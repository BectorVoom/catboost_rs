//! GPUT-18 (Phase 12 Plan 04, W2b): the device **Region** grow driver. Grows one Region
//! PATH tree (upstream `TRegionModel` / `ComputeOptimalSplitsRegion`, §6.4 / §5.4) by
//! REUSING the whole-dataset device split scorer
//! ([`crate::gpu_runtime::launch_find_optimal_split_pointwise`]) per frontier subset —
//! the SAME host-driven per-node scoring spine as the Plan-03 non-symmetric grow
//! ([`crate::kernels::nonsym_grow::device_best_split_for_node`]) — but with a REGION
//! selection: at each level the device scores the SINGLE frontier, and the child whose
//! own best split has the higher gain becomes the next frontier while the other diverges
//! into that level's terminal bin (upstream `SelectLeavesToSplit`'s lower-Score child).
//!
//! # Region is a PATH, not a node graph (Pitfall 2)
//!
//! A depth-`d` Region extends ONE path: `d` per-level splits and exactly `d + 1` leaves
//! (`MaxLeaves = MaxDepth + 1`), NOT `2^d`. This driver therefore emits a
//! `region_path: Vec<(feature, bin, expected_direction, one_hot)>` onto [`DeviceGrownTree`]
//! (length `d`) plus `d + 1` `leaf_values` — it MUST NOT populate `step_nodes` /
//! `node_id_to_leaf_id` (that is the non-symmetric node-graph carrier; a `2^d` leaf count
//! is the failure signal for the "Region is a binary tree" bug).
//!
//! The region-walk direction test transcribed inline (`add_model_value.cu::AddRegionImpl` /
//! `ComputeRegionBinsImpl`, `takeEqualAndSplitDirection`): the recorded per-level direction
//! is `true` when the CONTINUING child is the `value > border` (passes) child, else `false`,
//! so the apply walk `(value > border) == expected_direction` descends into the frontier and
//! diverges on the first mismatch. Ties prefer the passes (right) child (`>=`), the SAME
//! deterministic rule as `cb_train::tree::region_grower` (the frozen ≤1e-5 CPU Region oracle
//! this driver reproduces, Plan 02).
//!
//! # `-inf` sentinel landmine (MEMORY `cubecl-hip-no-inf-literal`)
//!
//! This module contains NO `#[cube]` kernel of its own — it composes the existing
//! `launch_find_optimal_split_pointwise` (whose kernel already uses finite `f32::MIN` /
//! `f32::MAX` ARGMAX sentinels). The host-side child-gain comparison here uses
//! `f64::NEG_INFINITY` (host code, never a `#[cube]` region — allowed, exactly as
//! `cb_train::tree::region_grower` does). No `cb-train` / `cb-model` dependency (the
//! Region walk semantics are transcribed inline, Pattern B feature-unification landmine).

use cb_compute::{calc_average, DeviceGrownTree};
use cb_core::{sum_f64, CbError, CbResult};

use crate::kernels::nonsym_grow::device_best_split_for_node;

/// Grow ONE Region path tree on the device (GPUT-18, D-03a). Returns a [`DeviceGrownTree`]
/// with a `region_path` of length `depth` (per-level `(feature, bin, expected_direction,
/// one_hot)`), `leaf_values` of length `depth + 1` (UN-scaled by `learning_rate`,
/// `calc_average` — the RMSE / Gradient leaf, the SAME contract as the non-sym grow), and
/// `leaf_of` (per-object terminal bin, length `n`, for the self-oracle; the boosting fold
/// arm recomputes it via the walk). `step_nodes` / `node_id_to_leaf_id` stay EMPTY (Region
/// is a path, not a node graph).
///
/// # Errors
/// - [`CbError::Degenerate`] if `max_depth == 0`, or if the ROOT cannot be split at all (no
///   device candidate / gain below the `1e-9` cutoff), so no region path exists — mirroring
///   `cb_train::tree::region_grower`'s degenerate contract.
/// - [`CbError::LengthMismatch`] if `cindex` is not exactly `n_features * n` long.
/// - [`CbError::OutOfRange`] if `n_features * n` overflows `usize`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grow_region_tree(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    n: usize,
    n_bins: usize,
    n_features: usize,
    max_depth: usize,
    min_data_in_leaf: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<DeviceGrownTree> {
    if max_depth == 0 {
        return Err(CbError::Degenerate(
            "grow_region_tree: max_depth == 0 (no region path can be grown)".to_owned(),
        ));
    }
    let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!("n_features ({n_features}) * n ({n}) overflows usize"))
    })?;
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // The current path frontier's owning document subset (root = all objects).
    let mut frontier: Vec<usize> = (0..n).collect();
    // Per-level chosen split + continue direction, in walk order (`depth == region_path.len()`).
    let mut region_path: Vec<(u32, u32, bool, bool)> = Vec::new();
    // Per-bin terminal document subsets: `terminal_docs[k]` diverged at level `k`. The final
    // surviving frontier (bin == depth) is appended after the loop.
    let mut terminal_docs: Vec<Vec<usize>> = Vec::new();

    for _level in 0..max_depth {
        // Score the frontier's best split on the DEVICE (argmin over all (feature, bin)
        // candidates; `device_best_split_for_node` enforces `gain >= 1e-9` + min_data_in_leaf,
        // so a beneficial split is required to extend the path — the root splits only when a
        // beneficial candidate exists; a degenerate root errors below).
        let Some(bs) = device_best_split_for_node(
            &frontier, der1, weight, cindex, n, n_bins, n_features, min_data_in_leaf, scaled_l2,
            score_fn,
        )?
        else {
            break;
        };

        // Continue direction: the child whose OWN best split has the higher gain is the next
        // frontier (upstream picks the lower-Score == higher-gain child to split next). A
        // non-splittable child scores `NEG_INFINITY`. Ties prefer the passes (right) child
        // (`>=`), deterministic — the SAME rule as `region_grower`.
        let right_gain = device_best_split_for_node(
            &bs.right_docs, der1, weight, cindex, n, n_bins, n_features, min_data_in_leaf,
            scaled_l2, score_fn,
        )?
        .map_or(f64::NEG_INFINITY, |b| b.gain);
        let left_gain = device_best_split_for_node(
            &bs.left_docs, der1, weight, cindex, n, n_bins, n_features, min_data_in_leaf,
            scaled_l2, score_fn,
        )?
        .map_or(f64::NEG_INFINITY, |b| b.gain);

        // The device float Region grow emits `value > border` splits only (never one-hot).
        if right_gain >= left_gain {
            // The passes (`cindex > bin`) child continues → direction `true`; the not-passes
            // (left) child diverges into this level's terminal bin.
            region_path.push((bs.feature, bs.bin, true, false));
            terminal_docs.push(bs.left_docs);
            frontier = bs.right_docs;
        } else {
            region_path.push((bs.feature, bs.bin, false, false));
            terminal_docs.push(bs.right_docs);
            frontier = bs.left_docs;
        }
    }

    if region_path.is_empty() {
        // The root never split → no region path (degenerate, matching the CPU region_grower
        // and the leaf-wise `Degenerate` contract for a root with no beneficial candidate).
        return Err(CbError::Degenerate(
            "grow_region_tree produced no split (root gain below the 1e-9 cutoff or too few docs)"
                .to_owned(),
        ));
    }

    let depth = region_path.len();

    // Per-object terminal bin (`AddRegionImpl` walk bin): objects in `terminal_docs[k]`
    // diverged at level `k` → bin `k`; the surviving `frontier` matched every direction → bin
    // `depth`. Populated for the self-oracle; the boosting fold arm recomputes it via the walk.
    let mut leaf_of: Vec<u32> = vec![0; n];
    for (bin, docs) in terminal_docs.iter().enumerate() {
        for &obj in docs {
            if let Some(slot) = leaf_of.get_mut(obj) {
                *slot = bin as u32;
            }
        }
    }
    for &obj in &frontier {
        if let Some(slot) = leaf_of.get_mut(obj) {
            *slot = depth as u32;
        }
    }

    // Leaf values in bin order (`calc_average` over each terminal bin's docs, UN-scaled by lr —
    // the SAME contract as `grow_nonsym_tree`). Exactly `depth + 1` values: `terminal_docs[0..
    // depth]` then the surviving `frontier`.
    let mut leaf_values: Vec<f64> = Vec::with_capacity(depth + 1);
    for docs in terminal_docs.iter().chain(std::iter::once(&frontier)) {
        let der_sub: Vec<f64> = docs.iter().map(|&i| der1.get(i).copied().unwrap_or(0.0)).collect();
        let w_sub: Vec<f64> = docs.iter().map(|&i| weight.get(i).copied().unwrap_or(0.0)).collect();
        leaf_values.push(calc_average(sum_f64(&der_sub), sum_f64(&w_sub), scaled_l2));
    }

    Ok(DeviceGrownTree {
        // `splits` mirrors the region path's per-level `(feature, bin)` for structural
        // completeness; the boosting Region fold arm resolves its borders from `region_path`
        // (which additionally carries the direction + one-hot flags).
        splits: region_path.iter().map(|&(f, b, _, _)| (f, b)).collect(),
        leaf_values,
        // Scalar Region emission: one value per leaf ⇒ approx_dim == 1
        // (byte-unchanged block collapse, D-04).
        approx_dim: 1,
        leaf_of,
        // Region is a PATH — NO node graph (the boosting dispatch keys on `region_path`
        // non-empty BEFORE `step_nodes`, so these stay empty).
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_path,
    })
}
