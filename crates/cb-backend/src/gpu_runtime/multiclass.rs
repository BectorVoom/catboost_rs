//! GPUT-12 (Phase 13 Plan 07, W7): the multi-output device **driver** — the block-leaf emission
//! that wires the Plan-06 K-dim Newton der2 block solve ([`crate::kernels::multi_newton`]) onto the
//! whole multi-output loss family (MultiClass softmax, MultiClassOneVsAll, MultiLogloss /
//! MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty). It grows ONE shared tree structure per
//! boosting step (the leaf assignment `leaf_of` is shared across the `K` approx dimensions — the
//! oblivious structure is one tree), solves a K-dim der2 block per leaf (COUPLED full-block for
//! MultiClass softmax, DIAGONAL per-component for the separable losses), and emits the
//! `leaf_count × approx_dim` ROW-MAJOR block ([`cb_compute::DeviceGrownTree`] layout, `approx_dim =
//! K`) that routes through the existing multi-output CPU apply
//! (`approx[d * n + i] += lr * leaf_block[leaf_of[i] * K + d]`).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! - [`map_multiclass_objective`] — the loss → [`MulticlassObjective`] classification (coupled vs
//!   diagonal dispatch, RESEARCH Pitfall 3, VERIFIED). MultiQuantile / any pointwise loss → `None`.
//! - [`assemble_multiclass_ders`] — the per-object der ASSEMBLY, calling the der functions that
//!   ALREADY exist in `cb-compute` (`softmax_ders` / `multiclass_onevsall_ders` /
//!   `multi_crossentropy_ders` / `rmse_with_uncertainty_ders`). The device transcribes the block
//!   EMISSION, not the der — this only gathers the existing CPU der into the dimension-major der1 +
//!   per-object packed der2 buffers the block solve consumes.
//! - [`grow_multiclass_block`] — the driver: accumulate per-leaf `sum_der` (K) + `sum_der2_packed`
//!   (`K·(K+1)/2`) via the ordered [`cb_core::sum_f64`] (D-08 — same reduction order as the CPU
//!   `compute_softmax_leaf_deltas` reference), dispatch the device [`launch_multi_newton_solve`]
//!   batched over the leaves (coupled/diagonal per the objective), and read back the
//!   `leaf_count × K` row-major block.
//!
//! # Coupled vs diagonal dispatch (RESEARCH Pitfall 3, VERIFIED)
//!
//! COUPLED (the full K×K solve) is used ONLY for `MultiClass` softmax — its off-diagonal
//! `−w·p_k·p_row` Hessian couples the classes, so the leaf delta is one dense symmetric solve.
//! DIAGONAL (a per-component 1×1 Newton solve) is used for the SEPARABLE losses (MultiClassOneVsAll,
//! MultiLogloss / MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty): their Hessian is diagonal, so
//! each component solves independently (== `solve_symmetric_newton` with `k == 1` per component,
//! the Plan-06 diagonal mode reading only the packed diagonal entries `H[d][d]`).
//!
//! # f64-typed seam / landmines
//!
//! The der/weight SUMS use the ordered host reduction; the per-leaf block solve accumulates in f64
//! on device (Plan 06 — the per-matrix arithmetic is not an atomic reduction, so gfx1100's missing
//! f64 atomic-add is irrelevant here). WGSL has no f64, so a genuine `wgpu` backend surfaces a typed
//! [`CbError::OutOfRange`] rather than an opaque JIT crash (the [`launch_multi_newton_solve`]
//! reject). cb-backend must NEVER gain a `cb-train` dep (the feature-unification landmine) — this
//! driver reaches only `cb_compute` for the host der functions + types. No `-inf` literal in any
//! kernel body (the block solve's pivot fallback is a finite zero — Plan 06). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13).

use cb_compute::{
    multi_crossentropy_ders, multiclass_onevsall_ders, rmse_with_uncertainty_ders, softmax_ders,
    Loss,
};
use cb_core::{sum_f64, CbError, CbResult};

use crate::kernels::multi_newton::launch_multi_newton_solve;
#[cfg(not(feature = "wgpu"))]
use crate::SelectedRuntime;

/// The multi-output objective the block solve dispatches on. `Softmax` is the ONLY coupled arm (the
/// full K×K softmax solve); the remaining four are SEPARABLE (diagonal per-component solve). The der
/// math for each already lives in `cb-compute`; this enum only records the dispatch decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MulticlassObjective {
    /// MultiClass softmax — the COUPLED full K×K dense symmetric Newton solve
    /// (`softmax_ders` + `solve_symmetric_newton`, the off-diagonal Hessian couples classes).
    Softmax,
    /// MultiClassOneVsAll — DIAGONAL per-dimension sigmoid Newton (`multiclass_onevsall_ders`).
    OneVsAll,
    /// MultiLogloss / MultiCrossEntropy — DIAGONAL per-label sigmoid Newton (`multi_crossentropy_ders`,
    /// the SAME upstream `TMultiCrossEntropyError` class for both losses).
    MultiCrossEntropy,
    /// MultiRMSE — DIAGONAL per-target regression Newton (`der1 = w·(target−approx)`, `der2 = −w`).
    /// (No `Loss` variant yet; classified here so the arm lands when the variant is added.)
    MultiRmse,
    /// RMSEWithUncertainty — DIAGONAL K=2 solve over `[mean, log-scale]`
    /// (`rmse_with_uncertainty_ders`; the weight is folded inside that der).
    RmseWithUncertainty,
}

impl MulticlassObjective {
    /// Whether this objective uses the COUPLED full-block solve (softmax only). Every other arm is
    /// separable and uses the diagonal per-component solve (Pitfall 3).
    pub(crate) fn is_coupled(self) -> bool {
        matches!(self, MulticlassObjective::Softmax)
    }
}

/// Classify a host [`Loss`] into its [`MulticlassObjective`], or `None` for a non-multi-output loss
/// (the family-gated `Option`-returning template — Pattern A). MultiQuantile is a multi-output loss
/// but NOT in the covered set (its exact-quantile leaf is a different estimator, Plan 09), so it
/// declines here. Every scalar / pairwise / ranking loss returns `None`.
pub(crate) fn map_multiclass_objective(loss: &Loss) -> Option<MulticlassObjective> {
    match loss {
        Loss::MultiClass => Some(MulticlassObjective::Softmax),
        Loss::MultiClassOneVsAll => Some(MulticlassObjective::OneVsAll),
        Loss::MultiLogloss | Loss::MultiCrossEntropy => Some(MulticlassObjective::MultiCrossEntropy),
        Loss::RmseWithUncertainty => Some(MulticlassObjective::RmseWithUncertainty),
        // MultiRMSE has no `Loss` variant yet; when added it maps to `MultiRmse` (diagonal).
        _ => None,
    }
}

/// The packed lower-triangular index of the diagonal entry `(d, d)` in the order
/// `[(0,0),(0,1),…,(0,k-1),(1,1),…]`: `d·k − d·(d−1)/2`. Guards the `d == 0` usize underflow (the
/// device kernel evaluates the identical expression in wrapping GPU arithmetic).
fn diag_index(d: usize, k: usize) -> usize {
    if d == 0 {
        return 0;
    }
    d * k - d * (d - 1) / 2
}

/// The packed lower-triangular Hessian length `k·(k+1)/2`, overflow-checked.
fn packed_len(k: usize) -> CbResult<usize> {
    k.checked_mul(k + 1)
        .map(|p| p / 2)
        .ok_or_else(|| CbError::OutOfRange(format!("packed hessian length overflows (k = {k})")))
}

/// Read object `i`'s remapped contiguous class index from a per-object class-label column
/// (`target[i]`, stored as `f64`). A negative / non-finite label clamps to `0` (the caller's
/// `Loss::validate` range guard is the real defense — this never indexes out of bounds).
fn class_of(target: &[f64], i: usize) -> usize {
    let t = target.get(i).copied().unwrap_or(0.0);
    if t.is_finite() && t >= 0.0 {
        t.round() as usize
    } else {
        0
    }
}

/// Assemble the per-object multi-output der from the der functions ALREADY in `cb-compute` (the
/// device transcribes the block EMISSION, not the der). Returns:
/// - `der1`: the DIMENSION-MAJOR weighted first derivative `der1[d * n + i]` (length `k · n`), and
/// - `der2_packed`: the PER-OBJECT packed lower-triangular Hessian `der2_packed[i * pk + j]` (length
///   `n · pk`, `pk = k·(k+1)/2`), weighted. For the separable (diagonal) objectives only the packed
///   DIAGONAL entries are filled (off-diagonal stays `0`) — the device diagonal mode reads only
///   those, matching the CPU per-component `solve_symmetric_newton(k == 1)` reference.
///
/// `approx` is dimension-major (`approx[d * n + i]`, length `k · n`). `target`'s layout depends on
/// the objective: a per-object class column (Softmax / OneVsAll), a dimension-major label matrix
/// (MultiCrossEntropy, `target[d * n + i]`), or a per-object regression target (MultiRMSE /
/// RMSEWithUncertainty). `weight` is per-object (length `n`); the covered regime is unit weights.
/// No `unwrap`/`expect`/`panic`/indexing (D-13).
pub(crate) fn assemble_multiclass_ders(
    objective: MulticlassObjective,
    approx: &[f64],
    target: &[f64],
    weight: &[f64],
    n: usize,
    k: usize,
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    let pk = packed_len(k)?;
    let k_n = k
        .checked_mul(n)
        .ok_or_else(|| CbError::OutOfRange(format!("k·n overflows (k = {k}, n = {n})")))?;
    let n_pk = n
        .checked_mul(pk)
        .ok_or_else(|| CbError::OutOfRange(format!("n·pk overflows (n = {n}, pk = {pk})")))?;
    if approx.len() != k_n {
        return Err(CbError::LengthMismatch {
            column: "approx".to_owned(),
            expected: k_n,
            actual: approx.len(),
        });
    }
    let mut der1 = vec![0.0_f64; k_n];
    let mut der2_packed = vec![0.0_f64; n_pk];

    // Set dimension `d` of object `i` in both der buffers (safe indexed writes).
    let mut set = |der1: &mut [f64], der2: &mut [f64], i: usize, d: usize, d1: f64, d2_diag: f64| {
        if let Some(slot) = der1.get_mut(d * n + i) {
            *slot = d1;
        }
        if let Some(slot) = der2.get_mut(i * pk + diag_index(d, k)) {
            *slot = d2_diag;
        }
    };

    for i in 0..n {
        let w = weight.get(i).copied().unwrap_or(1.0);
        match objective {
            MulticlassObjective::Softmax => {
                // Coupled: the FULL packed Hessian (diagonal + off-diagonal) is produced by
                // `softmax_ders`; fold the per-object weight into BOTH der1 and the packed der2.
                let approx_i: Vec<f64> =
                    (0..k).map(|d| approx.get(d * n + i).copied().unwrap_or(0.0)).collect();
                let (d1, d2) = softmax_ders(&approx_i, class_of(target, i));
                for d in 0..k {
                    if let Some(slot) = der1.get_mut(d * n + i) {
                        *slot = d1.get(d).copied().unwrap_or(0.0) * w;
                    }
                }
                for j in 0..pk {
                    if let Some(slot) = der2_packed.get_mut(i * pk + j) {
                        *slot = d2.get(j).copied().unwrap_or(0.0) * w;
                    }
                }
            }
            MulticlassObjective::OneVsAll => {
                let class = class_of(target, i);
                for d in 0..k {
                    let a = approx.get(d * n + i).copied().unwrap_or(0.0);
                    let (d1, d2) = multiclass_onevsall_ders(a, d == class);
                    set(&mut der1, &mut der2_packed, i, d, d1 * w, d2 * w);
                }
            }
            MulticlassObjective::MultiCrossEntropy => {
                for d in 0..k {
                    let a = approx.get(d * n + i).copied().unwrap_or(0.0);
                    let t = target.get(d * n + i).copied().unwrap_or(0.0);
                    let (d1, d2) = multi_crossentropy_ders(a, t);
                    set(&mut der1, &mut der2_packed, i, d, d1 * w, d2 * w);
                }
            }
            MulticlassObjective::MultiRmse => {
                // Separable RMSE per target dimension: der1 = w·(target_d − approx_d), der2 = −w.
                for d in 0..k {
                    let a = approx.get(d * n + i).copied().unwrap_or(0.0);
                    let t = target.get(d * n + i).copied().unwrap_or(0.0);
                    set(&mut der1, &mut der2_packed, i, d, w * (t - a), -w);
                }
            }
            MulticlassObjective::RmseWithUncertainty => {
                // K=2 `[mean, log-scale]`; `rmse_with_uncertainty_ders` folds the weight itself.
                let a0 = approx.get(i).copied().unwrap_or(0.0);
                let a1 = approx.get(n + i).copied().unwrap_or(0.0);
                let t = target.get(i).copied().unwrap_or(0.0);
                let (d1, d2) = rmse_with_uncertainty_ders(a0, a1, t, w);
                for d in 0..k.min(2) {
                    set(
                        &mut der1,
                        &mut der2_packed,
                        i,
                        d,
                        d1.get(d).copied().unwrap_or(0.0),
                        d2.get(d).copied().unwrap_or(0.0),
                    );
                }
            }
        }
    }
    Ok((der1, der2_packed))
}

/// Accumulate the per-leaf `sum_der` (length `n_leaves · k`, system-major `sum_der[leaf * k + d]`)
/// and `sum_der2_packed` (length `n_leaves · pk`, `sum_der2_packed[leaf * pk + j]`) from the
/// per-object dimension-major `der1` + per-object packed `der2_packed`, gathering each leaf's
/// members in ASCENDING object order and reducing through the ordered [`cb_core::sum_f64`] (D-08) —
/// the SAME reduction order the CPU `compute_softmax_leaf_deltas` reference uses. The layout is
/// system-major so it feeds [`launch_multi_newton_solve`] (one per-leaf system) and reads back as
/// the `leaf_count × k` ROW-MAJOR [`cb_compute::DeviceGrownTree`] block directly.
fn accumulate_leaf_blocks(
    leaf_of: &[u32],
    der1: &[f64],
    der2_packed: &[f64],
    n: usize,
    k: usize,
    pk: usize,
    n_leaves: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Per-leaf, per-dimension / per-packed-index member lists (ascending object order).
    let mut der1_members: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); k]; n_leaves];
    let mut der2_members: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); pk]; n_leaves];
    for i in 0..n {
        let leaf = leaf_of.get(i).copied().unwrap_or(0) as usize;
        if leaf >= n_leaves {
            continue;
        }
        for d in 0..k {
            let v = der1.get(d * n + i).copied().unwrap_or(0.0);
            if let Some(slot) = der1_members.get_mut(leaf).and_then(|r| r.get_mut(d)) {
                slot.push(v);
            }
        }
        for j in 0..pk {
            let v = der2_packed.get(i * pk + j).copied().unwrap_or(0.0);
            if let Some(slot) = der2_members.get_mut(leaf).and_then(|r| r.get_mut(j)) {
                slot.push(v);
            }
        }
    }

    let mut sum_der = vec![0.0_f64; n_leaves * k];
    let mut sum_der2_packed = vec![0.0_f64; n_leaves * pk];
    for leaf in 0..n_leaves {
        for d in 0..k {
            let members = der1_members
                .get(leaf)
                .and_then(|r| r.get(d))
                .map_or(&[][..], Vec::as_slice);
            if let Some(slot) = sum_der.get_mut(leaf * k + d) {
                *slot = sum_f64(members);
            }
        }
        for j in 0..pk {
            let members = der2_members
                .get(leaf)
                .and_then(|r| r.get(j))
                .map_or(&[][..], Vec::as_slice);
            if let Some(slot) = sum_der2_packed.get_mut(leaf * pk + j) {
                *slot = sum_f64(members);
            }
        }
    }
    (sum_der, sum_der2_packed)
}

/// Dispatch the device [`launch_multi_newton_solve`] batched over `n_leaves` per-leaf systems of
/// dimension `k`, reading back the `leaf_count × k` ROW-MAJOR block (`out[leaf * k + d]`). `coupled`
/// selects the full softmax solve; `!coupled` the per-component diagonal solve. A `wgpu` backend
/// surfaces the typed f64 reject from [`launch_multi_newton_solve`]. An empty problem
/// (`k == 0` / `n_leaves == 0`) returns an empty block WITHOUT a device launch (never a 0-len read).
fn solve_blocks_device(
    sum_der: &[f64],
    sum_der2_packed: &[f64],
    k: usize,
    n_leaves: usize,
    scaled_l2: f64,
    coupled: bool,
) -> CbResult<Vec<f64>> {
    if k == 0 || n_leaves == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (sum_der, sum_der2_packed, scaled_l2, coupled);
        return Err(CbError::OutOfRange(
            "multi-output block solve requires f64 device channels; the wgpu backend has none \
             (WGSL has no f64). Use the rocm/cuda/cpu backend for the multi-output leaf block."
                .to_owned(),
        ));
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
        let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
        let ders_h = client.create(cubecl::bytes::Bytes::from_elems(sum_der.to_vec()));
        let packed_h = client.create(cubecl::bytes::Bytes::from_elems(sum_der2_packed.to_vec()));
        let handle =
            launch_multi_newton_solve(&client, &ders_h, &packed_h, k, n_leaves, scaled_l2, coupled)?;
        let bytes = client.read_one(handle).map_err(|e| {
            CbError::Degenerate(format!("multi-output block solve read-back failed: {e:?}"))
        })?;
        Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
    }
}

/// Grow the multi-output leaf block for ONE shared tree: assemble the per-object der (the existing
/// `cb-compute` der), accumulate the per-leaf `sum_der` / `sum_der2_packed` (ordered, D-08),
/// dispatch the device K-dim Newton block solve (coupled for MultiClass softmax, diagonal for the
/// separable losses), and return the `leaf_count × K` ROW-MAJOR block leaf
/// (`out[leaf * K + d]` = dimension `d` of leaf `leaf`) UNSCALED (the learning-rate factor is
/// applied downstream — the [`cb_compute::DeviceGrownTree`] contract). This is the device analog of
/// the CPU `compute_softmax_leaf_deltas` / diagonal per-`d` leaf estimation, matched ≤1e-4 by the
/// `multiclass_test` self-oracle.
///
/// `leaf_of` (length `n`) is the shared leaf assignment; `approx` is dimension-major (`k · n`);
/// `target` / `weight` follow [`assemble_multiclass_ders`]; `k` is the approx dimension; `n_leaves`
/// the leaf count; `scaled_l2` the per-tree `scale_l2_reg` output. No `unwrap`/`expect`/`panic`/
/// indexing (D-13).
pub(crate) fn grow_multiclass_block(
    objective: MulticlassObjective,
    leaf_of: &[u32],
    approx: &[f64],
    target: &[f64],
    weight: &[f64],
    k: usize,
    n_leaves: usize,
    scaled_l2: f64,
) -> CbResult<Vec<f64>> {
    let n = leaf_of.len();
    let pk = packed_len(k)?;
    let (der1, der2_packed) = assemble_multiclass_ders(objective, approx, target, weight, n, k)?;
    let (sum_der, sum_der2_packed) =
        accumulate_leaf_blocks(leaf_of, &der1, &der2_packed, n, k, pk, n_leaves);
    solve_blocks_device(
        &sum_der,
        &sum_der2_packed,
        k,
        n_leaves,
        scaled_l2,
        objective.is_coupled(),
    )
}
