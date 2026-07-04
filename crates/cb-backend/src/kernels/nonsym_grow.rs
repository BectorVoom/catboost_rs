//! GPUT-18 (Phase 12 Plan 03, W1): the device **Depthwise / Lossguide** non-symmetric
//! grow driver. Grows one non-symmetric tree by REUSING the whole-dataset device split
//! scorer ([`crate::gpu_runtime::launch_find_optimal_split_pointwise`]) per candidate
//! leaf-node, changing ONLY the leaf-selection ORDER (level-order for Depthwise,
//! best-gain-priority for Lossguide) and emitting a NODE GRAPH instead of a depth-length
//! per-level split list. The node-registration bookkeeping (`step_nodes` /
//! `node_id_to_leaf_id`, `u32::MAX` interior sentinel, checked `u16::try_from` child
//! diffs) is TRANSCRIBED VERBATIM from `cb_train::tree::leaf_wise_grower`
//! (`tree.rs:923-1247`) — NEVER `use cb_train` (the feature-unification landmine that
//! would pull `cb-backend/cpu` into a `rocm`/`wgpu`/`cuda` build).
//!
//! # Host-light per-node scoring (the D-05 boundary for the non-sym MVP)
//!
//! Unlike the resident oblivious grow loop, this driver is host-DRIVEN: for each
//! candidate node it slices that node's document subset (host) and hands it to the
//! device split scorer, which fills a fresh histogram over the subset device-resident
//! and returns the O(1) best `(feature, bin)` descriptor. The bulk per-candidate score
//! vector stays on the device (only the O(blocks) winner + the small subset buffers
//! cross the seam per node). This mirrors upstream's generic
//! `TGreedyTreeLikeStructureSearcher<TTreeModel>` → one `BuildTreeLikeModel<TModel>`
//! step: the device searches STRUCTURE + leaf values only, and host
//! `Model::from_trained` builds `TreeVariant::NonSymmetric` from the emitted plain host
//! structs (D-04, host-structs-only — no device-native tree type).
//!
//! # Structure parity (the STRICT bar) vs leaf VALUES (the ε=1e-4 bar)
//!
//! The device split SELECTION (`launch_find_optimal_split_pointwise`'s argmin) matches
//! the CPU `select_best_candidate` EXACTLY on a clear-gain-margin fixture (proven by the
//! sibling `score_split` oracle), so the emitted node graph is INTEGER-exact vs
//! `leaf_wise_grower`. The per-node GAIN that gates a split (`>= 1e-9`) and orders the
//! Lossguide priority queue is computed HOST-side via the SAME `cb_compute` score
//! functions `leaf_wise_grower` uses (ordered `sum_f64` reductions, D-08), so the queue
//! order is bit-identical. Leaf VALUES (`calc_average`, UN-scaled by `learning_rate`) are
//! the ε=1e-4 device-vs-CPU bar (Kaggle CUDA sign-off deferred to Plan 09).
//!
//! # `-inf` sentinel landmine (Pattern D / MEMORY `cubecl-hip-no-inf-literal`)
//!
//! This module contains NO `#[cube]` kernel of its own — it composes the existing
//! `launch_find_optimal_split_pointwise` (which already uses finite `f32::MIN`/`f32::MAX`
//! ARGMAX sentinels inside its kernel). The host-side argmax accumulators here use
//! `f64::NEG_INFINITY` (host code, never a `#[cube]` region — allowed).

use cb_compute::{
    calc_average, cosine_split_score, l2_split_score, multi_dim_split_score, DeviceGrownTree,
    EScoreFunction, LeafStats,
};
use cb_core::{sum_f64, CbError, CbResult};

use crate::gpu_runtime::launch_find_optimal_split_pointwise;
use crate::kernels::{
    SCORE_FN_COSINE, SCORE_FN_L2, SCORE_FN_LOO_L2, SCORE_FN_SAT_L2, SCORE_FN_SOLAR_L2,
};

/// The non-symmetric leaf-wise expansion strategy (mirrors `cb_train::tree::LeafWisePolicy`
/// BY VALUE — deliberately re-declared here, NOT `use cb_train`, so `cb-backend` never
/// gains a `cb-train` dependency via feature unification, Pattern B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NonsymPolicy {
    /// Level-order expansion: split every eligible leaf at the current level before
    /// descending, bounded by `max_depth`.
    Depthwise,
    /// Best-gain priority-queue expansion: pop the highest-gain splittable leaf, split
    /// it, enqueue its children; stop when the queue empties or the structure reaches
    /// `max_leaves` leaves.
    Lossguide,
}

/// The best split found for ONE node's document subset: the chosen `(feature, bin)`, its
/// GAIN (2-leaf split score minus the unsplit single-leaf score), and the left / right
/// child document subsets (`value <= border` / `value > border`). Mirrors
/// `cb_train::tree::LeafBestSplit`.
pub(crate) struct NodeBestSplit {
    pub(crate) feature: u32,
    pub(crate) bin: u32,
    pub(crate) gain: f64,
    /// Left-child docs (split FALSE: `cindex <= bin`).
    pub(crate) left_docs: Vec<usize>,
    /// Right-child docs (split TRUE: `cindex > bin`).
    pub(crate) right_docs: Vec<usize>,
}

/// One built node in the flat non-symmetric node graph (mirrors
/// `cb_train::tree::BuiltNode`).
enum BuiltNode {
    /// Interior node: its chosen `(feature, bin)` split and its left / right child ids.
    Interior { feature: u32, bin: u32, left: usize, right: usize },
    /// Terminal leaf node (distinct-leaf id assigned in node order at finalization).
    Leaf,
}

/// The host split-score dispatch over the covered score-function selectors, matching the
/// `cb_train::tree::split_score` dispatch VERBATIM (L2 / Cosine direct, the Solar / LOO /
/// Sat variants via `multi_dim_split_score` with the two leaves as one dimension). Used
/// ONLY for the O(1) GAIN of a chosen split (gate + Lossguide priority) — the per-candidate
/// SCAN stays on the device.
fn host_split_score(leaves: &[LeafStats], scaled_l2: f64, score_fn: u32) -> f64 {
    match score_fn {
        SCORE_FN_COSINE => cosine_split_score(leaves, scaled_l2),
        SCORE_FN_L2 => l2_split_score(leaves, scaled_l2),
        SCORE_FN_SOLAR_L2 => {
            multi_dim_split_score(EScoreFunction::SolarL2, &[leaves.to_vec()], scaled_l2)
        }
        SCORE_FN_LOO_L2 => {
            multi_dim_split_score(EScoreFunction::LOOL2, &[leaves.to_vec()], scaled_l2)
        }
        SCORE_FN_SAT_L2 => {
            multi_dim_split_score(EScoreFunction::SatL2, &[leaves.to_vec()], scaled_l2)
        }
        // Any other selector is a caller bug; L2 is the safe fold (never a wrong-arm
        // silent dispatch — the gate only ever passes the five above).
        _ => l2_split_score(leaves, scaled_l2),
    }
}

/// The single-leaf unsplit score over a document subset (the baseline the per-node gain
/// subtracts), mirroring `cb_train::tree::unsplit_leaf_score`.
fn unsplit_score(docs: &[usize], der1: &[f64], weight: &[f64], scaled_l2: f64, score_fn: u32) -> f64 {
    let der_sub: Vec<f64> = docs.iter().map(|&i| der1.get(i).copied().unwrap_or(0.0)).collect();
    let w_sub: Vec<f64> = docs.iter().map(|&i| weight.get(i).copied().unwrap_or(0.0)).collect();
    let stats = LeafStats { sum_weighted_delta: sum_f64(&der_sub), sum_weight: sum_f64(&w_sub) };
    host_split_score(&[stats], scaled_l2, score_fn)
}

/// Find the best split for ONE node's document subset using the DEVICE split scorer for
/// SELECTION and the host score dispatch for the O(1) GAIN. Returns `None` when the node
/// cannot be split (too few docs, no device candidate, or gain below the `1e-9` cutoff) —
/// EXACTLY `cb_train::tree::best_split_for_leaf`'s gating, so the emitted structure matches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_best_split_for_node(
    docs: &[usize],
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    n: usize,
    n_bins: usize,
    n_features: usize,
    min_data_in_leaf: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<Option<NodeBestSplit>> {
    if docs.len() < min_data_in_leaf || docs.len() < 2 {
        return Ok(None);
    }
    let m = docs.len();

    // Slice this node's subset in ascending original-object order (the deterministic
    // reduction order the CPU reference uses). cindex is feature-major over the FULL n,
    // so the subset is re-packed feature-major over m.
    let der_sub: Vec<f64> = docs.iter().map(|&i| der1.get(i).copied().unwrap_or(0.0)).collect();
    let w_sub: Vec<f64> = docs.iter().map(|&i| weight.get(i).copied().unwrap_or(0.0)).collect();
    let mut cindex_sub: Vec<u32> = vec![0; n_features.saturating_mul(m)];
    for feature in 0..n_features {
        for (j, &obj) in docs.iter().enumerate() {
            let src = cindex.get(feature.saturating_mul(n).saturating_add(obj)).copied().unwrap_or(0);
            if let Some(slot) = cindex_sub.get_mut(feature.saturating_mul(m).saturating_add(j)) {
                *slot = src;
            }
        }
    }
    let indices_sub: Vec<u32> = (0..m as u32).collect();

    // DEVICE split selection: the argmin over all (feature, bin) candidates for THIS
    // subset (histogram device-resident; only the O(1) BestSplit crosses back).
    let (best, _scores) = launch_find_optimal_split_pointwise(
        &der_sub, &w_sub, &cindex_sub, &indices_sub, n_bins, n_features, scaled_l2, score_fn,
    )?;
    let Some(best) = best else { return Ok(None) };
    let feature = best.feature_id;
    let bin = best.bin_id;

    // Partition the node's docs by the chosen split (forward-bit `cindex > bin`), and
    // compute the 2-leaf split score HOST-side over the SAME partition (ordered sums) so
    // the GAIN matches `leaf_wise_grower` bit-for-bit.
    let mut left_docs: Vec<usize> = Vec::new();
    let mut right_docs: Vec<usize> = Vec::new();
    let mut left_der: Vec<f64> = Vec::new();
    let mut left_w: Vec<f64> = Vec::new();
    let mut right_der: Vec<f64> = Vec::new();
    let mut right_w: Vec<f64> = Vec::new();
    for &obj in docs {
        let d = der1.get(obj).copied().unwrap_or(0.0);
        let w = weight.get(obj).copied().unwrap_or(0.0);
        let passes = cindex
            .get(feature as usize * n + obj)
            .copied()
            .unwrap_or(0)
            > bin;
        if passes {
            right_docs.push(obj);
            right_der.push(d);
            right_w.push(w);
        } else {
            left_docs.push(obj);
            left_der.push(d);
            left_w.push(w);
        }
    }
    let split_stats = [
        LeafStats { sum_weighted_delta: sum_f64(&left_der), sum_weight: sum_f64(&left_w) },
        LeafStats { sum_weighted_delta: sum_f64(&right_der), sum_weight: sum_f64(&right_w) },
    ];
    let split_sc = host_split_score(&split_stats, scaled_l2, score_fn);
    let baseline = unsplit_score(docs, der1, weight, scaled_l2, score_fn);
    let gain = split_sc - baseline;
    if gain < 1e-9 {
        return Ok(None);
    }

    Ok(Some(NodeBestSplit { feature, bin, gain, left_docs, right_docs }))
}

/// Grow ONE non-symmetric tree on the device (GPUT-18). Returns a [`DeviceGrownTree`] with
/// per-NODE `splits`, the `step_nodes` / `node_id_to_leaf_id` node graph, `leaf_values`
/// (UN-scaled by `learning_rate`, `calc_average` — the RMSE / Gradient leaf), and `leaf_of`
/// (distinct-leaf id per object, length `n`, for the self-oracle; the boosting fold arm
/// recomputes it via the pointer-walk).
///
/// # Errors
/// - [`CbError::DepthExceeded`] semantics: `max_depth == 0` yields a degenerate error (no
///   tree can be grown).
/// - [`CbError::Degenerate`] if the root cannot be split at all.
/// - [`CbError::OutOfRange`] if a child diff exceeds `u16` (extreme `max_depth`, CR-01).
#[allow(clippy::too_many_arguments)]
pub(crate) fn grow_nonsym_tree(
    policy: NonsymPolicy,
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    n: usize,
    n_bins: usize,
    n_features: usize,
    max_depth: usize,
    max_leaves: usize,
    min_data_in_leaf: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<DeviceGrownTree> {
    if max_depth == 0 {
        return Err(CbError::Degenerate(
            "grow_nonsym_tree: max_depth == 0 (no non-symmetric tree can be grown)".to_owned(),
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

    // Flat node graph under construction (mirrors leaf_wise_grower's node vectors).
    let mut nodes: Vec<BuiltNode> = Vec::new();
    let mut node_docs: Vec<Vec<usize>> = Vec::new();
    let mut node_depth: Vec<usize> = Vec::new();

    let mut new_node = |nodes: &mut Vec<BuiltNode>,
                        node_docs: &mut Vec<Vec<usize>>,
                        node_depth: &mut Vec<usize>,
                        docs: Vec<usize>,
                        depth: usize|
     -> usize {
        let id = nodes.len();
        nodes.push(BuiltNode::Leaf); // provisional; promoted to Interior on split
        node_docs.push(docs);
        node_depth.push(depth);
        id
    };

    let root_docs: Vec<usize> = (0..n).collect();
    let root = new_node(&mut nodes, &mut node_docs, &mut node_depth, root_docs, 0);
    // leaf_owner[obj] = the node id of the leaf object `obj` currently lands in.
    let mut leaf_owner: Vec<usize> = vec![root; n];

    // Split node `id` given its best split: register two children, route docs. Returns
    // (left_child_id, right_child_id). Mirrors leaf_wise_grower::do_split.
    let mut do_split = |nodes: &mut Vec<BuiltNode>,
                        node_docs: &mut Vec<Vec<usize>>,
                        node_depth: &mut Vec<usize>,
                        leaf_owner: &mut [usize],
                        id: usize,
                        bs: &NodeBestSplit|
     -> (usize, usize) {
        let depth = node_depth.get(id).copied().unwrap_or(0) + 1;
        let left = new_node(nodes, node_docs, node_depth, bs.left_docs.clone(), depth);
        let right = new_node(nodes, node_docs, node_depth, bs.right_docs.clone(), depth);
        if let Some(slot) = nodes.get_mut(id) {
            *slot = BuiltNode::Interior { feature: bs.feature, bin: bs.bin, left, right };
        }
        for &obj in &bs.left_docs {
            if let Some(o) = leaf_owner.get_mut(obj) {
                *o = left;
            }
        }
        for &obj in &bs.right_docs {
            if let Some(o) = leaf_owner.get_mut(obj) {
                *o = right;
            }
        }
        (left, right)
    };

    match policy {
        NonsymPolicy::Depthwise => {
            let mut current_level: Vec<usize> = vec![root];
            for _cur_depth in 0..max_depth {
                let mut next_level: Vec<usize> = Vec::new();
                for &leaf in &current_level {
                    let docs = node_docs.get(leaf).cloned().unwrap_or_default();
                    if let Some(bs) = device_best_split_for_node(
                        &docs, der1, weight, cindex, n, n_bins, n_features, min_data_in_leaf,
                        scaled_l2, score_fn,
                    )? {
                        let (l, r) = do_split(
                            &mut nodes, &mut node_docs, &mut node_depth, &mut leaf_owner, leaf, &bs,
                        );
                        next_level.push(l);
                        next_level.push(r);
                    }
                }
                if next_level.is_empty() {
                    break;
                }
                current_level = next_level;
            }
        }
        NonsymPolicy::Lossguide => {
            use std::cmp::Ordering;
            use std::collections::BinaryHeap;

            struct QItem {
                gain: f64,
                seq: u64,
                node: usize,
                best: NodeBestSplit,
            }
            impl PartialEq for QItem {
                fn eq(&self, other: &Self) -> bool {
                    self.gain == other.gain && self.seq == other.seq
                }
            }
            impl Eq for QItem {}
            impl Ord for QItem {
                fn cmp(&self, other: &Self) -> Ordering {
                    // Higher gain first; equal gain → EARLIER seq wins (reverse seq).
                    match self.gain.partial_cmp(&other.gain).unwrap_or(Ordering::Equal) {
                        Ordering::Equal => other.seq.cmp(&self.seq),
                        ord => ord,
                    }
                }
            }
            impl PartialOrd for QItem {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut heap: BinaryHeap<QItem> = BinaryHeap::new();
            let mut seq: u64 = 0;
            let mut leaf_count: usize = 1;

            // enqueue the best split of `node` iff its depth < max_depth. The best-split
            // computation calls the device scorer — surface any device error.
            macro_rules! enqueue {
                ($node:expr) => {{
                    let node = $node;
                    if node_depth.get(node).copied().unwrap_or(0) < max_depth {
                        let docs = node_docs.get(node).cloned().unwrap_or_default();
                        if let Some(bs) = device_best_split_for_node(
                            &docs, der1, weight, cindex, n, n_bins, n_features, min_data_in_leaf,
                            scaled_l2, score_fn,
                        )? {
                            heap.push(QItem { gain: bs.gain, seq, node, best: bs });
                            seq += 1;
                        }
                    }
                }};
            }

            enqueue!(root);

            while let Some(item) = heap.pop() {
                if leaf_count >= max_leaves {
                    break;
                }
                let (l, r) = do_split(
                    &mut nodes, &mut node_docs, &mut node_depth, &mut leaf_owner, item.node,
                    &item.best,
                );
                leaf_count += 1;
                enqueue!(l);
                enqueue!(r);
            }
        }
    }

    // ── Finalize the flat node graph → step_nodes + node_id_to_leaf_id + per-node splits
    //    + leaf_values (transcribed from leaf_wise_grower's finalization). ──────────────
    let node_count = nodes.len();
    let mut step_nodes: Vec<(u16, u16)> = Vec::with_capacity(node_count);
    let mut node_id_to_leaf_id: Vec<u32> = vec![u32::MAX; node_count];
    let mut splits: Vec<(u32, u32)> = Vec::with_capacity(node_count);
    let mut leaf_values: Vec<f64> = Vec::new();
    let mut node_to_leaf: Vec<Option<u32>> = vec![None; node_count];
    let mut next_leaf_id: u32 = 0;

    for (id, node) in nodes.iter().enumerate() {
        match node {
            BuiltNode::Interior { feature, bin, left, right } => {
                splits.push((*feature, *bin));
                if let Some(slot) = node_id_to_leaf_id.get_mut(id) {
                    *slot = u32::MAX;
                }
                let ld = u16::try_from(left.saturating_sub(id)).map_err(|_| {
                    CbError::OutOfRange(format!(
                        "non-symmetric step-node child diff {} at node {id} exceeds u16 range",
                        left.saturating_sub(id)
                    ))
                })?;
                let rd = u16::try_from(right.saturating_sub(id)).map_err(|_| {
                    CbError::OutOfRange(format!(
                        "non-symmetric step-node child diff {} at node {id} exceeds u16 range",
                        right.saturating_sub(id)
                    ))
                })?;
                step_nodes.push((ld, rd));
            }
            BuiltNode::Leaf => {
                // Placeholder split for a leaf node (never read — its step entry is (0,0)).
                splits.push((0, 0));
                step_nodes.push((0, 0));
                if let Some(slot) = node_to_leaf.get_mut(id) {
                    *slot = Some(next_leaf_id);
                }
                if let Some(slot) = node_id_to_leaf_id.get_mut(id) {
                    *slot = next_leaf_id;
                }
                // Leaf value via calc_average over this leaf's docs (UN-scaled by lr).
                let docs = node_docs.get(id).cloned().unwrap_or_default();
                let der_sub: Vec<f64> =
                    docs.iter().map(|&i| der1.get(i).copied().unwrap_or(0.0)).collect();
                let w_sub: Vec<f64> =
                    docs.iter().map(|&i| weight.get(i).copied().unwrap_or(0.0)).collect();
                leaf_values.push(calc_average(sum_f64(&der_sub), sum_f64(&w_sub), scaled_l2));
                next_leaf_id += 1;
            }
        }
    }

    if next_leaf_id < 2 {
        return Err(CbError::Degenerate(
            "grow_nonsym_tree produced no split (root gain below the 1e-9 cutoff or too few docs)"
                .to_owned(),
        ));
    }

    // Per-object distinct-leaf id (for the self-oracle; boosting recomputes via the walk).
    let leaf_of: Vec<u32> = leaf_owner
        .iter()
        .map(|&node| node_to_leaf.get(node).and_then(|o| *o).unwrap_or(0))
        .collect();

    Ok(DeviceGrownTree {
        splits,
        leaf_values,
        // Scalar non-symmetric emission: one value per leaf ⇒ approx_dim == 1
        // (byte-unchanged block collapse, D-04).
        approx_dim: 1,
        leaf_of,
        step_nodes,
        node_id_to_leaf_id,
        // Non-symmetric emission carries NO region path (the boosting fold arm keys the
        // Region dispatch on this being non-empty; a node-graph tree leaves it empty).
        region_path: Vec::new(),
    })
}
