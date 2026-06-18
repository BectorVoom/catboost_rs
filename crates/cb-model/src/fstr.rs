//! Feature-importance for the canonical [`crate::Model`] (MODEL-03 complete +
//! MODEL-05 partial):
//!
//! - [`prediction_values_change`] — the `PredictionValuesChange` importance
//!   (percent, Σ = 100),
//! - [`interaction`] — the pairwise `Interaction` importance
//!   (`Vec<(feature_i, feature_j, score)>`, percent-normalized), and
//! - [`loss_function_change`] — the `LossFunctionChange` importance (MODEL-03 /
//!   D-12, completing the deferred Phase-4 partial; D-6.6-09).
//!
//! `prediction_values_change` / `interaction` consume the per-leaf
//! training-document weights (`leaf_weights`) captured in Plan 01 (RESEARCH
//! Pitfall 1 — without them every importance silently short-circuits to zero).
//! Both branch on the tree variant: the OBLIVIOUS arm is the literal pre-6.6
//! bit-indexed loop (BYTE-IDENTICAL, D-6.6-05 — `fstr_oracle_test` stays locked);
//! the NON-SYMMETRIC arm walks the actual node graph (`step_nodes` children +
//! `node_id_to_leaf_id`) instead of bit-indexing leaf pairs (D-6.6-10).
//!
//! # Source of truth (RESEARCH Patterns 5 & 6)
//!
//! - **`prediction_values_change`** transcribes `CalcEffect`
//!   (`feature_str.h:233-270`): for each tree, for each split-level bit `feature`
//!   (source feature `srcIdx = splits[feature].feature`), and for each leaf pair
//!   `(leaf, leaf ^ (1<<feature))` with `inverted > leaf`, with `count1 =
//!   leaf_weight[leaf]`, `count2 = leaf_weight[inverted]` — SKIP if either is 0
//!   (`avrg = (val1·c1 + val2·c2)/(c1+c2)`; `dif = (val1−avrg)²·c1 +
//!   (val2−avrg)²·c2`; `res[srcIdx] += dif`), then `ConvertToPercents`
//!   (`feature_str.cpp`) normalizing so `Σ = 100`.
//! - **`interaction`** transcribes `CalcMostInteractingFeatures`
//!   (`feature_str.cpp:190-223`) + `CalcFeatureInteraction`
//!   (`calc_fstr.cpp:343-414`): for each tree and each pair of split levels
//!   `(firstIdx, secondIdx)`, `delta = Σ_leaf sign·leafValue` with `sign =
//!   (var1 XOR var2) ? +1 : −1` (`var_k = (leaf & (1<<idx)) != 0`); accumulate
//!   `|delta|` into the sorted source-feature pair (skip equal source features);
//!   then each pair's `score = sum / totalEffect · 100`, sorted by score
//!   descending. For numeric-only models each split's source feature IS its float
//!   feature index (identity layout, A3), so no internal→regular remap is needed.
//!
//! `loss_function_change` (`loss_change_fstr.cpp:154-356`) is the metric-delta
//! importance: for each feature `f`, subtract that feature's per-document SHAP
//! contribution from the raw approx and re-evaluate the additive objective
//! metric; `score[f] = finalError(approx − shap[·][f]) − finalError(approx)`.
//! For the supported single-dimensional `Logloss` objective the additive
//! `finalError` is the mean per-object Logloss (best value = Min, so the score
//! is used verbatim — `CalcFeatureEffectLossChangeFromScores`).
//!
//! # Parity discipline
//!
//! Every float fold routes through [`cb_core::sum_f64`] (D-08). All leaf / weight
//! access is checked `.get` (`indexing_slicing` deny); no `unwrap`/`expect`. The
//! `count1 == 0 || count2 == 0` short-circuit (PVC) and the `total == 0` guard
//! (normalization) reproduce the upstream div-by-zero protections exactly
//! (T-04-04-03), so no NaN leaks.

use cb_core::sum_f64;

use crate::{shap::shap_values, Model};

/// The kind of feature importance to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureImportanceType {
    /// Per-feature `PredictionValuesChange` (the default upstream importance):
    /// how much, on average, predictions change when a feature's split flips,
    /// normalized to percentages summing to 100.
    PredictionValuesChange,
    /// Pairwise `Interaction` importance between feature pairs (percent).
    Interaction,
    /// Per-feature `LossFunctionChange` (MODEL-03 / D-12): the change in the
    /// objective metric when a feature's per-document SHAP contribution is
    /// removed from the raw approx (`loss_change_fstr.cpp`). Requires a dataset
    /// (features + labels), unlike the two structure-only importances.
    LossFunctionChange,
}

/// The number of (flat = float, numeric-only) features the model splits on:
/// `max split feature index + 1`. Matches the width of the upstream importance
/// vector for a numeric-only pool (unused features keep a `0` importance).
/// Spans BOTH tree variants (oblivious `splits` and non-symmetric `tree_splits`)
/// so a non-symmetric model widens the vector correctly (D-6.6-10).
fn feature_count(model: &Model) -> usize {
    let oblivious_max = model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        // Numeric-only feature importance projects over FLOAT splits; a CTR split
        // has no single float-feature index and does not widen the float vector.
        .filter_map(crate::ModelSplit::float_feature)
        .map(|f| f + 1)
        .max()
        .unwrap_or(0);
    let non_symmetric_max = model
        .non_symmetric_trees
        .iter()
        .flat_map(|t| t.tree_splits.iter())
        .filter_map(crate::ModelSplit::float_feature)
        .map(|f| f + 1)
        .max()
        .unwrap_or(0);
    oblivious_max.max(non_symmetric_max)
}

/// `ConvertToPercents` (`feature_str.cpp`): scale `res` in place so it sums to
/// 100. A zero total leaves the vector untouched (upstream divides by `total`;
/// the importance is only computed for models with weighted leaves, so `total`
/// is positive in practice — the guard prevents a NaN, T-04-04-03).
fn convert_to_percents(res: &mut [f64]) {
    let total = sum_f64(res);
    if total == 0.0 {
        return;
    }
    for x in res.iter_mut() {
        *x *= 100.0 / total;
    }
}

/// `PredictionValuesChange` feature importance (MODEL-03). Branches on the tree
/// variant: the OBLIVIOUS arm transcribes the bit-indexed `CalcEffect`
/// (`feature_str.h:233-270`); the NON-SYMMETRIC arm transcribes the node-graph
/// `CalcEffectForNonObliviousModel` (`feature_str.h:149-228`, D-6.6-10). Returns
/// one percentage per feature (length = [`feature_count`]); percentages sum to
/// 100.
#[must_use]
pub fn prediction_values_change(model: &Model) -> Vec<f64> {
    let n_features = feature_count(model);
    let mut res = vec![0.0_f64; n_features];

    // OBLIVIOUS arm — the literal pre-6.6 bit-indexed loop (D-6.6-05).
    for tree in &model.oblivious_trees {
        pvc_accumulate_oblivious(tree, &mut res);
    }
    // NON-SYMMETRIC arm — node-graph recursion (D-6.6-10).
    for tree in &model.non_symmetric_trees {
        pvc_accumulate_non_symmetric(tree, &mut res);
    }

    convert_to_percents(&mut res);
    res
}

/// The OBLIVIOUS `PredictionValuesChange` accumulation (`CalcEffect`,
/// `feature_str.h:233-270`) — bit-indexed leaf pairs `(leaf, leaf ^ (1<<bit))`.
/// Kept BYTE-IDENTICAL to the pre-6.6 loop (D-6.6-05).
fn pvc_accumulate_oblivious(tree: &crate::ObliviousTree, res: &mut [f64]) {
    let leaf_count = tree.leaf_values.len();
    // for (feature = 0; feature < tree.SrcFeatures.size(); ++feature)
    for (feature_bit, split) in tree.splits.iter().enumerate() {
        // Numeric-only importance: a CTR split has no single float-feature
        // index, so it contributes nothing to the per-float-feature vector.
        let Some(src_idx) = split.float_feature() else {
            continue;
        };
        // for (leafIdx = 0; leafIdx < tree.Leaves.size(); ++leafIdx)
        for leaf_idx in 0..leaf_count {
            let inverted = leaf_idx ^ (1usize << feature_bit);
            if inverted < leaf_idx {
                continue;
            }
            let count1 = tree.leaf_weights.get(leaf_idx).copied().unwrap_or(0.0);
            let count2 = tree.leaf_weights.get(inverted).copied().unwrap_or(0.0);
            if count1 == 0.0 || count2 == 0.0 {
                continue;
            }
            let val1 = tree.leaf_values.get(leaf_idx).copied().unwrap_or(0.0);
            let val2 = tree.leaf_values.get(inverted).copied().unwrap_or(0.0);

            let avrg = (val1 * count1 + val2 * count2) / (count1 + count2);
            let dif = (val1 - avrg).powi(2) * count1 + (val2 - avrg).powi(2) * count2;

            if let Some(slot) = res.get_mut(src_idx) {
                *slot += dif;
            }
        }
    }
}

/// Per-node `(weighted-average leaf value, subtree document weight)` info,
/// matching upstream `TNodeInfo` in the node-graph `CalcEffect`.
#[derive(Clone, Copy)]
struct NodeInfo {
    value: f64,
    count: f64,
}

/// The NON-SYMMETRIC `PredictionValuesChange` accumulation,
/// `CalcEffectForNonObliviousModel` (`feature_str.h:149-228`, D-6.6-10).
///
/// Two passes per tree. Pass 1 (forward over nodes): a TERMINAL node
/// (`left_diff == 0 || right_diff == 0`) seeds its `NodeInfo` from
/// `node_id_to_leaf_id[node]` (value = leaf value, count = leaf weight); a node
/// that ALSO has a non-zero child (a one-sided `(d,0)`/`(0,d)` halt) is BOTH a
/// terminal AND an interior node, so it is additionally pushed as a triangle.
/// Pass 2 (LIFO over the triangle stack ⇒ bottom-up, since children appear after
/// their parent in node order): combine the left/right child `NodeInfo`,
/// accumulate `dif = (val1−avrg)²·c1 + (val2−avrg)²·c2` into `res[featureIdx]`,
/// and store the merged parent `NodeInfo`. All reductions are scalar adds in the
/// upstream order; `.get(...)` everywhere (depth-bounded by `node_count`,
/// T-06.6-15).
fn pvc_accumulate_non_symmetric(tree: &crate::NonSymmetricTree, res: &mut [f64]) {
    use std::collections::HashMap;

    let node_count = tree.step_nodes.len();
    let mut node_info: HashMap<usize, NodeInfo> = HashMap::new();
    // (parent_node, left_child, right_child, feature_idx)
    let mut triangles: Vec<(usize, usize, usize, usize)> = Vec::new();

    for node_idx in 0..node_count {
        let (left_diff, right_diff) = tree.step_nodes.get(node_idx).copied().unwrap_or((0, 0));
        let is_terminal = left_diff == 0 || right_diff == 0;
        if is_terminal {
            // leafValueIndex = NonSymmetricNodeIdToLeafId[nodeIdx]; for the 1-dim
            // numeric model the per-leaf value/weight are indexed directly by the
            // (already LOCAL, 06.6-05) leaf id.
            let leaf_id = tree
                .node_id_to_leaf_id
                .get(node_idx)
                .copied()
                .map_or(usize::MAX, |v| v as usize);
            let value = tree.leaf_values.get(leaf_id).copied().unwrap_or(0.0);
            let count = tree.leaf_weights.get(leaf_id).copied().unwrap_or(0.0);
            node_info.insert(node_idx, NodeInfo { value, count });
            if left_diff == 0 && right_diff == 0 {
                continue;
            }
        }
        // Interior (possibly one-sided): record the triangle. A CTR split has no
        // float-feature index and is skipped (numeric-only projection).
        let Some(feature_idx) = tree
            .tree_splits
            .get(node_idx)
            .and_then(crate::ModelSplit::float_feature)
        else {
            continue;
        };
        let left_child = node_idx.saturating_add(left_diff as usize);
        let right_child = node_idx.saturating_add(right_diff as usize);
        triangles.push((node_idx, left_child, right_child, feature_idx));
    }

    // LIFO pop = bottom-up (children were pushed after their parent).
    while let Some((parent, left, right, feature_idx)) = triangles.pop() {
        let left_info = node_info.remove(&left).unwrap_or(NodeInfo { value: 0.0, count: 0.0 });
        let right_info = node_info.remove(&right).unwrap_or(NodeInfo { value: 0.0, count: 0.0 });

        let count1 = left_info.count;
        let count2 = right_info.count;
        let sum_count = count1 + count2;

        let val1 = if count1 != 0.0 { left_info.value } else { 0.0 };
        let val2 = if count2 != 0.0 { right_info.value } else { 0.0 };

        let denom = if sum_count != 0.0 { sum_count } else { 1.0 };
        let avrg = (val1 * count1 + val2 * count2) / denom;
        let dif = (val1 - avrg).powi(2) * count1 + (val2 - avrg).powi(2) * count2;

        if let Some(slot) = res.get_mut(feature_idx) {
            *slot += dif;
        }
        node_info.insert(parent, NodeInfo { value: avrg, count: sum_count });
    }
}

/// Accumulate `|delta|` for a sorted source-feature pair into the insertion-order
/// `(pairs, sums)` accumulator (a Vec keyed by pair rather than a hash map, so
/// the iteration order is deterministic; the final score sort is by value).
fn interaction_add(pairs: &mut Vec<(usize, usize)>, sums: &mut Vec<f64>, a: usize, b: usize, contribution: f64) {
    if let Some(pos) = pairs.iter().position(|&p| p == (a, b)) {
        if let Some(slot) = sums.get_mut(pos) {
            *slot += contribution;
        }
    } else {
        pairs.push((a, b));
        sums.push(contribution);
    }
}

/// `Interaction` feature importance (MODEL-03). Branches on the tree variant:
/// the OBLIVIOUS arm transcribes the bit-indexed `CalcMostInteractingFeatures`
/// (`feature_str.cpp:190-223`); the NON-SYMMETRIC arm transcribes the node-graph
/// DFS `CalcMostInteractingFeatures(model, featureToIdx)` (`feature_str.cpp:226-295`,
/// D-6.6-10). Both feed the shared `CalcFeatureInteraction` scoring
/// (`calc_fstr.cpp:343-414`).
///
/// Returns `(feature_i, feature_j, score)` triples with `i < j`, `score` the
/// percent-of-total pairwise contribution, sorted by `score` descending (the
/// upstream `EXISTING_PAIRS_COUNT` behavior — only pairs that actually appear).
/// **NOT** SHAP interaction values (that is the advanced-fstr plan).
#[must_use]
pub fn interaction(model: &Model) -> Vec<(usize, usize, f64)> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut sums: Vec<f64> = Vec::new();

    // OBLIVIOUS arm — the literal pre-6.6 bit-indexed loop (D-6.6-05).
    for tree in &model.oblivious_trees {
        let split_count = tree.splits.len();
        let leaf_count = tree.leaf_values.len();
        // for (firstIdx = 0; firstIdx < splits-1; ++firstIdx)
        for first_idx in 0..split_count.saturating_sub(1) {
            for second_idx in (first_idx + 1)..split_count {
                let n1 = 1usize << first_idx;
                let n2 = 1usize << second_idx;

                // delta = Σ_leaf sign·leafValue (order-locked, D-08).
                let signed: Vec<f64> = (0..leaf_count)
                    .map(|leaf_idx| {
                        let var1 = (leaf_idx & n1) != 0;
                        let var2 = (leaf_idx & n2) != 0;
                        let sign = if var1 ^ var2 { 1.0 } else { -1.0 };
                        let val = tree.leaf_values.get(leaf_idx).copied().unwrap_or(0.0);
                        sign * val
                    })
                    .collect();
                let delta = sum_f64(&signed);

                // Numeric-only interaction: skip a pair that involves a CTR split
                // (no single float-feature index).
                let (Some(src1), Some(src2)) = (
                    tree.splits.get(first_idx).and_then(crate::ModelSplit::float_feature),
                    tree.splits.get(second_idx).and_then(crate::ModelSplit::float_feature),
                ) else {
                    continue;
                };
                if src1 == src2 {
                    continue;
                }
                let (a, b) = if src1 < src2 { (src1, src2) } else { (src2, src1) };
                interaction_add(&mut pairs, &mut sums, a, b, delta.abs());
            }
        }
    }

    // NON-SYMMETRIC arm — per-tree signed DFS accumulation, then `|·|` per tree
    // (D-6.6-10).
    for tree in &model.non_symmetric_trees {
        interaction_accumulate_non_symmetric(tree, &mut pairs, &mut sums);
    }

    // CalcFeatureInteraction: score = sum / totalEffect * 100.
    let total_effect = sum_f64(&sums);
    let mut result: Vec<(usize, usize, f64)> = pairs
        .iter()
        .zip(sums.iter())
        .map(|(&(a, b), &s)| {
            let score = if total_effect == 0.0 {
                0.0
            } else {
                s / total_effect * 100.0
            };
            (a, b, score)
        })
        .collect();

    // StableSort(rbegin, rend) by Score  =>  descending by score.
    result.sort_by(|l, r| r.2.partial_cmp(&l.2).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// The NON-SYMMETRIC `Interaction` accumulation, the DFS in
/// `CalcMostInteractingFeatures(model, featureToIdx)` (`feature_str.cpp:226-295`,
/// D-6.6-10).
///
/// Per tree: DFS from the root node carrying a `path` of `(feature_idx, sign)`.
/// At a TERMINAL node (`left_child == node || right_child == node`, i.e. one diff
/// is 0) accumulate `treeSum[(f1,f2)] += sign1·sign2·delta` for every pair in the
/// path (`delta` = Σ leaf values over the dimension; 1-dim ⇒ the single value).
/// Descending into the left child uses `sign = -1`, the right child `sign = +1`.
/// After the tree, fold `Σ|treeSum[pair]|` into the shared accumulator. The walk
/// is `node_count`-depth-bounded with checked `.get` (T-06.6-15) — a malformed
/// graph stops the descent rather than recursing forever.
fn interaction_accumulate_non_symmetric(
    tree: &crate::NonSymmetricTree,
    pairs: &mut Vec<(usize, usize)>,
    sums: &mut Vec<f64>,
) {
    let node_count = tree.step_nodes.len();
    // Per-tree signed accumulator (insertion-order keyed Vec, like the shared one).
    let mut tree_pairs: Vec<(usize, usize)> = Vec::new();
    let mut tree_sums: Vec<f64> = Vec::new();
    let mut path: Vec<(usize, i32)> = Vec::new();

    interaction_dfs(tree, 0, node_count, &mut path, &mut tree_pairs, &mut tree_sums);

    // sumInteractions[pair] += |treeSum[pair]|.
    for (&pair, &signed) in tree_pairs.iter().zip(tree_sums.iter()) {
        interaction_add(pairs, sums, pair.0, pair.1, signed.abs());
    }
}

/// One DFS step of the non-symmetric `Interaction` accumulation (the `DFS`
/// helper, `feature_str.cpp:226-264`). `remaining_depth` bounds the recursion at
/// `node_count` so a cyclic / escaping graph can never recurse without end
/// (T-06.6-15).
fn interaction_dfs(
    tree: &crate::NonSymmetricTree,
    node_idx: usize,
    remaining_depth: usize,
    path: &mut Vec<(usize, i32)>,
    tree_pairs: &mut Vec<(usize, usize)>,
    tree_sums: &mut Vec<f64>,
) {
    if remaining_depth == 0 {
        return;
    }
    let (left_diff, right_diff) = match tree.step_nodes.get(node_idx) {
        Some(&d) => d,
        None => return,
    };
    let left_child = node_idx.saturating_add(left_diff as usize);
    let right_child = node_idx.saturating_add(right_diff as usize);

    // Terminal node: accumulate over every path pair (DoSwap to sorted order,
    // skip equal source features). For a 1-dim numeric model `delta` is the
    // single leaf value at the node's leaf id.
    if left_child == node_idx || right_child == node_idx {
        let leaf_id = tree
            .node_id_to_leaf_id
            .get(node_idx)
            .copied()
            .map_or(usize::MAX, |v| v as usize);
        let delta = tree.leaf_values.get(leaf_id).copied().unwrap_or(0.0);

        for first in 0..path.len() {
            for second in (first + 1)..path.len() {
                let (Some(&(f1, s1)), Some(&(f2, s2))) = (path.get(first), path.get(second)) else {
                    continue;
                };
                let (mut a, mut b) = (f1, f2);
                if b < a {
                    std::mem::swap(&mut a, &mut b);
                }
                if a == b {
                    continue;
                }
                let sign = f64::from(s1 * s2);
                interaction_add(tree_pairs, tree_sums, a, b, sign * delta);
            }
        }
    }

    // The node's own split feature (skip a CTR split — no float-feature index).
    let Some(feature_idx) = tree
        .tree_splits
        .get(node_idx)
        .and_then(crate::ModelSplit::float_feature)
    else {
        return;
    };

    // Descend: left child sign = -1, right child sign = +1 (the upstream
    // `sign = -1; ... sign *= -1` order over `{left, right}`).
    let mut sign: i32 = -1;
    for &child_idx in &[left_child, right_child] {
        if child_idx != node_idx {
            path.push((feature_idx, sign));
            interaction_dfs(tree, child_idx, remaining_depth - 1, path, tree_pairs, tree_sums);
            path.pop();
        }
        sign = -sign;
    }
}

/// `LossFunctionChange` feature importance (MODEL-03 / D-12; D-6.6-09),
/// transcribing `CalcFeatureEffectLossChange` + `...MetricStats` +
/// `...FromScores` (`loss_change_fstr.cpp:154-356`) for the supported
/// single-dimensional `Logloss` objective.
///
/// For each feature `f`: `score[f] = finalError(approx − shap[·][f]) −
/// finalError(approx)`, where `approx` is the raw model output (the apply path),
/// `shap[obj][f]` is feature `f`'s per-document SHAP contribution
/// ([`crate::shap_values`]), and `finalError` is the additive `Logloss` metric's
/// mean per-object loss (best value = `Min`, so the difference is used verbatim —
/// `CalcFeatureEffectLossChangeFromScores`). The output is per-feature in feature
/// index order (length = [`feature_count`]); the Python
/// `get_feature_importance(type='LossFunctionChange')` maps the internal sorted
/// scores back to this layout.
///
/// `cols` are the per-feature object-major columns the apply / SHAP paths consume;
/// `labels` are the binary targets (`0.0` / `1.0`); `n_features` is the SHAP
/// feature width. Reductions route through [`cb_core::sum_f64`] (D-08); all access
/// is checked `.get` (no `unwrap` / `expect`).
#[must_use]
pub fn loss_function_change(
    model: &Model,
    cols: &[Vec<f32>],
    labels: &[f64],
    n_features: usize,
) -> Vec<f64> {
    let n_objects = cols.first().map_or(0, Vec::len);
    if n_objects == 0 || labels.len() != n_objects {
        return vec![0.0_f64; n_features];
    }

    // Raw approx (RawFormulaVal) over the documents.
    let approx = crate::apply::predict_raw(model, cols);
    // Per-document per-feature SHAP (shape [n_objects][n_features + 1]; the
    // trailing column is the bias / mean and is not subtracted).
    let shap = shap_values(model, cols, n_features);

    // finalError(approx) = mean per-object Logloss (additive metric / count).
    let base_score = logloss_final_error(&approx, labels);

    (0..n_features)
        .map(|feature| {
            // approx_f[obj] = approx[obj] − shap[obj][feature].
            let approx_f: Vec<f64> = (0..n_objects)
                .map(|obj| {
                    let a = approx.get(obj).copied().unwrap_or(0.0);
                    let s = shap
                        .get(obj)
                        .and_then(|row| row.get(feature))
                        .copied()
                        .unwrap_or(0.0);
                    a - s
                })
                .collect();
            // score = finalError(without feature f) − finalError(full). Logloss
            // best value is Min, so the difference is the importance verbatim.
            logloss_final_error(&approx_f, labels) - base_score
        })
        .collect()
}

/// The additive `Logloss` `finalError`: the mean per-object binary cross-entropy
/// (`metric.GetFinalError` = `error_sum / weight_sum`, here unit weights). The
/// per-object sigmoid + clamp mirrors the upstream additive Logloss metric. The
/// element sum routes through [`cb_core::sum_f64`] (D-08).
fn logloss_final_error(approx: &[f64], labels: &[f64]) -> f64 {
    const EPS: f64 = 1e-15;
    let losses: Vec<f64> = approx
        .iter()
        .zip(labels.iter())
        .map(|(&a, &t)| {
            let p = (1.0 / (1.0 + (-a).exp())).clamp(EPS, 1.0 - EPS);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .collect();
    let count = losses.len();
    if count == 0 {
        return 0.0;
    }
    sum_f64(&losses) / count as f64
}
