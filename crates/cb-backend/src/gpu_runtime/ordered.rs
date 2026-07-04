//! GPUT-13 (Phase 13 Plan 08, W8): the ordered-boosting device **driver** — the per-permutation
//! historical approx TRAJECTORY that ordered boosting (`EBoostingType::Ordered`) keeps
//! DEVICE-RESIDENT across boosting iterations. It reproduces the frozen CPU
//! `cb_train::boosting::ordered_approx_delta_simple` body/tail approximant (the anti-leakage heart of
//! ORD-02: a TAIL document's approx delta is estimated from the BODY prefix PLUS only the tail
//! documents that PRECEDE it in the learn permutation — it never depends on itself; the BODY rows
//! keep delta `0`, the estimation prefix) at ε=1e-4, and accumulates the per-iteration deltas into a
//! resident trajectory handle updated ON DEVICE via `apply_leaf_delta` — NO n-length read-back per
//! iteration (Pitfall 5: only the O(1) per-iteration descriptor + the host-computed ordered delta
//! cross the seam; the trajectory handle never leaves the device mid-loop, read back exactly ONCE at
//! the end for the oracle comparison).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! - [`OrderedTree`] — one boosting iteration's frozen ordered descriptor (leaf assignment, per-object
//!   der/weight, the learn permutation, the body/tail boundary, leaf count, `scaled_l2`).
//! - [`ordered_approx_delta`] — ONE tree's ordered per-object approximant delta, transcribing
//!   `ordered_approx_delta_simple` INLINE (never a `cb-train` dep — the CPU ref is the FROZEN oracle,
//!   not a runtime call). The tail is walked in PERMUTATION order, add-then-read the running per-leaf
//!   gradient average via the SHARED `cb_compute::gradient_leaf_delta` (the same `calc_average` the
//!   CPU reference uses). Body rows stay `0`.
//! - [`accumulate_ordered_trajectory`] — the resident driver: iterate the trees, computing each
//!   tree's ordered delta (host, inherently sequential permutation scan) and folding it into the
//!   DEVICE-RESIDENT trajectory via [`crate::gpu_runtime::launch_apply_leaf_delta_into`] (an IDENTITY
//!   leaf map + unit rate, so `trajectory[i] += delta[i]`), reading the trajectory back exactly ONCE
//!   at the end.
//!
//! # f64-typed seam / landmines
//!
//! The ordered trajectory needs f64 device channels for the ε=1e-4 bar; WGSL has no f64, so a genuine
//! `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather than an opaque JIT crash (the
//! [`accumulate_ordered_trajectory`] reject — the `multiclass` / `mvs_device` precedent). cb-backend
//! must NEVER gain a `cb-train` dep (the feature-unification landmine that breaks the rocm runtime) —
//! this driver reaches only `cb_compute` (`gradient_leaf_delta`) + `cb_core`. No `-inf` literal in any
//! kernel body (the resident add reuses the existing `apply_leaf_delta_kernel`, which has none). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13); never reads a 0-len
//! handle.

use cb_compute::gradient_leaf_delta;
use cb_core::{CbError, CbResult};

#[cfg(not(feature = "wgpu"))]
use cubecl::server::Handle;

#[cfg(not(feature = "wgpu"))]
use crate::SelectedRuntime;

/// One boosting iteration's frozen ordered descriptor. All slices are in OBJECT order except
/// [`Self::permutation`], which maps a learn-order position to its object index (`permutation[p]` is
/// the object at learn position `p`). The body/tail boundary [`Self::body_finish`] /
/// [`Self::tail_finish`] are learn-order positions (the estimation prefix is `[0, body_finish)`; the
/// historically-approximated tail is `[body_finish, tail_finish)`).
pub(crate) struct OrderedTree<'a> {
    /// Object `doc`'s leaf index in the grown tree (OBJECT order), length ≥ `permutation.len()`.
    pub leaf_of: &'a [u32],
    /// Object `doc`'s first derivative (already weighted if weighted), OBJECT order.
    pub der: &'a [f64],
    /// Object `doc`'s weight (OBJECT order); EMPTY ⇒ all `1.0`.
    pub weights: &'a [f64],
    /// The learn permutation: `permutation[p]` is the object at learn-order position `p`.
    pub permutation: &'a [i32],
    /// The body/tail boundary (learn-order position): `[0, body_finish)` is the estimation prefix.
    pub body_finish: usize,
    /// The tail end (learn-order position): the historically-approximated tail is
    /// `[body_finish, tail_finish)`.
    pub tail_finish: usize,
    /// The tree's leaf count.
    pub n_leaves: usize,
    /// The L2 regularizer (`cb_compute::scale_l2_reg` output).
    pub scaled_l2: f64,
}

/// Object `doc`'s weight (empty `weights` ⇒ `1.0`), a bounds-safe read.
fn weight_of(weights: &[f64], doc: usize) -> f64 {
    if weights.is_empty() {
        1.0
    } else {
        weights.get(doc).copied().unwrap_or(1.0)
    }
}

/// Compute ONE tree's ordered per-object approximant delta, transcribing
/// `cb_train::boosting::ordered_approx_delta_simple` INLINE (never a `cb-train` dep). The running
/// per-leaf der/weight accumulator is seeded with the BODY prefix (`[0, body_finish)` in learn
/// order), then the TAIL rows `[body_finish, tail_finish)` are walked IN PERMUTATION order:
///   1. the row's own der/weight enters its leaf's running sum (`AddMethodDer`),
///   2. the running leaf average `gradient_leaf_delta(leafDer, leafWeight, l2)` — which NOW includes
///      this row (upstream adds-then-reads) — is written to `approx_delta[doc]`.
/// Body rows (and any object outside `[0, n)`) keep delta `0` (the anti-leakage estimation prefix).
/// Returns the per-object delta in OBJECT order (length `permutation.len()`).
///
/// # Errors
/// [`CbError::Degenerate`] if `leaf_of` / `der` are shorter than the permutation implies, or a
/// permutation index is negative / out of range.
pub(crate) fn ordered_approx_delta(tree: &OrderedTree) -> CbResult<Vec<f64>> {
    let n = tree.permutation.len();
    if tree.leaf_of.len() < n || tree.der.len() < n {
        return Err(CbError::Degenerate(
            "ordered_approx: leaf_of / der shorter than permutation".to_owned(),
        ));
    }
    let mut approx_delta = vec![0.0_f64; n];

    // Running per-leaf der/weight accumulator, seeded by the BODY prefix sums.
    let mut leaf_sum_der = vec![0.0_f64; tree.n_leaves];
    let mut leaf_sum_weight = vec![0.0_f64; tree.n_leaves];

    // Seed the body prefix: accumulate the first `body_finish` learn-order rows into their leaves.
    for p in 0..tree.body_finish.min(n) {
        let Some(&doc_i) = tree.permutation.get(p) else {
            break;
        };
        if doc_i < 0 {
            return Err(CbError::Degenerate(
                "ordered_approx: body permutation index is negative".to_owned(),
            ));
        }
        let doc = doc_i as usize;
        let (Some(&leaf), Some(&d)) = (tree.leaf_of.get(doc), tree.der.get(doc)) else {
            return Err(CbError::Degenerate(
                "ordered_approx: body permutation index out of range".to_owned(),
            ));
        };
        let w = weight_of(tree.weights, doc);
        let leaf = leaf as usize;
        if let (Some(sd), Some(sw)) = (leaf_sum_der.get_mut(leaf), leaf_sum_weight.get_mut(leaf)) {
            *sd += d;
            *sw += w;
        }
    }

    // Walk the TAIL rows in permutation order; add-then-read the running leaf average.
    for p in tree.body_finish..tree.tail_finish.min(n) {
        let Some(&doc_i) = tree.permutation.get(p) else {
            break;
        };
        if doc_i < 0 {
            return Err(CbError::Degenerate(
                "ordered_approx: tail permutation index is negative".to_owned(),
            ));
        }
        let doc = doc_i as usize;
        let (Some(&leaf), Some(&d)) = (tree.leaf_of.get(doc), tree.der.get(doc)) else {
            return Err(CbError::Degenerate(
                "ordered_approx: tail permutation index out of range".to_owned(),
            ));
        };
        let w = weight_of(tree.weights, doc);
        let leaf = leaf as usize;
        // AddMethodDer: this row's der/weight enters its leaf's running sum FIRST.
        if let (Some(sd), Some(sw)) = (leaf_sum_der.get_mut(leaf), leaf_sum_weight.get_mut(leaf)) {
            *sd += d;
            *sw += w;
        }
        // CalcMethodDelta (Gradient/RMSE simple path): the SHARED `calc_average` the CPU ref uses.
        let leaf_der = leaf_sum_der.get(leaf).copied().unwrap_or(0.0);
        let leaf_weight = leaf_sum_weight.get(leaf).copied().unwrap_or(0.0);
        let delta = gradient_leaf_delta(leaf_der, leaf_weight, tree.scaled_l2);
        if let Some(slot) = approx_delta.get_mut(doc) {
            *slot = delta;
        }
    }

    Ok(approx_delta)
}

/// Accumulate the ordered per-permutation approx TRAJECTORY across `trees` boosting iterations,
/// keeping the trajectory DEVICE-RESIDENT across iterations. The resident trajectory handle is
/// initialised all-zero ONCE, and each iteration folds that tree's ordered delta (from
/// [`ordered_approx_delta`]) into it ON DEVICE via
/// [`crate::gpu_runtime::launch_apply_leaf_delta_into`] with an IDENTITY leaf map + unit rate — so
/// the kernel computes `trajectory[i] += 1.0 * delta[identity[i]] = trajectory[i] += delta[i]`. The
/// handle is read back EXACTLY ONCE at the end (Pitfall 5: no n-length read-back per iteration — only
/// the O(1) per-iteration descriptor + the host-computed ordered delta cross the seam). Returns the
/// final trajectory in object order.
///
/// # Errors
/// [`CbError::OutOfRange`] on the (f64-less) wgpu backend; propagates any
/// [`ordered_approx_delta`] error or a device read-back failure.
pub(crate) fn accumulate_ordered_trajectory(trees: &[OrderedTree], n: usize) -> CbResult<Vec<f64>> {
    #[cfg(feature = "wgpu")]
    {
        let _ = (trees, n);
        return Err(CbError::OutOfRange(
            "ordered trajectory requires f64 device channels; the wgpu backend has none (WGSL has \
             no f64). Use the rocm/cuda/cpu backend for the ordered per-permutation trajectory."
                .to_owned(),
        ));
    }
    #[cfg(not(feature = "wgpu"))]
    {
        if n == 0 {
            return Ok(Vec::new());
        }
        let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
        let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

        // The resident trajectory, all-zero, RESIDENT across iterations (updated IN PLACE below).
        let mut trajectory_h: Handle =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
        // The IDENTITY leaf map (object → itself), uploaded ONCE and cloned per iteration.
        let identity: Vec<u32> = (0..n as u32).collect();
        let identity_h: Handle = client.create(cubecl::bytes::Bytes::from_elems(identity));

        for tree in trees {
            // The ordered delta is an inherently SEQUENTIAL permutation scan (host).
            let delta = ordered_approx_delta(tree)?;
            // Resident add via the existing `apply_leaf_delta` kernel: the trajectory handle is
            // updated ON DEVICE and returned WITHOUT a read-back (identity map + unit rate).
            trajectory_h = crate::gpu_runtime::launch_apply_leaf_delta_into(
                &client,
                trajectory_h,
                identity_h.clone(),
                &delta,
                1.0,
                n,
            )?;
        }

        // The ONE final read-back (the oracle comparison) — the only n-length crossing.
        let bytes = client.read_one(trajectory_h).map_err(|e| {
            CbError::Degenerate(format!("ordered trajectory read-back failed: {e:?}"))
        })?;
        Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
    }
}
