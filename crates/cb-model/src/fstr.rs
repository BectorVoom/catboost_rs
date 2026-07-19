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
//! # CTR-aware attribution (FSTR-01: FIC-01/FIC-02/FIC-03)
//!
//! `prediction_values_change` and `interaction` are CTR-aware: a
//! `ModelSplit::Ctr` split's effect is attributed to the underlying
//! categorical feature(s) its projection combines, not silently dropped. Both
//! functions place categorical contributions into a **combined flat index**
//! ([`cat_feature_count`], [`flat_cat_index`]) that extends the existing
//! float-only layout — floats occupy `[0, n_float)` (unchanged,
//! [`feature_count`]), categoricals occupy `[n_float, n_float +
//! cat_feature_count(model))`. For a float-only model this is byte-identical
//! to the pre-CTR-aware behavior (regression-locked by the existing
//! `fstr_oracle_test.rs` assertions).
//!
//! - **`interaction`** (matching upstream `CalcMostInteractingFeatures` +
//!   `CalcFeatureInteraction`, `feature_str.cpp:190-223` / `calc_fstr.cpp:343-414`
//!   v1.2.10): split levels are first grouped by INTERNAL feature identity
//!   ([`same_internal_feature`] — a `Float` split's feature index, or a `Ctr`
//!   split's full border-less CTR descriptor, upstream `TFeature`/`GetFeature`).
//!   Level pairs of the SAME internal feature are skipped outright (upstream's
//!   `srcFeature1 == srcFeature2 → continue`); each surviving pair's `|delta|`
//!   accumulates into a per-internal-PAIR score. Only then does
//!   `CalcFeatureInteraction` expand each internal pair to combined-flat
//!   indices ([`split_flat_indices`]): each cross-product cell `(f0 ∈ side0) ×
//!   (f1 ∈ side1)` gets `score / (side0.len() * side1.len())`, self-cells
//!   (`f0 == f1`) are dropped from the OUTPUT — but the pair's FULL `score`
//!   still enters `totalEffect` (upstream `totalEffect += effect` sits outside
//!   the cross-product loops), so dropped self-cell mass deflates every
//!   returned percentage. The oblivious arm's `delta` is a fully-aggregated,
//!   pre-`abs()` scalar; the non-symmetric DFS arm accumulates SIGNED
//!   per-internal-pair sums with magnitude taken exactly once per tree
//!   (`sumInteractions[pair] += fabs(treeSum)`), preserving cross-leaf
//!   sign-cancellation.
//! - **`prediction_values_change`** (matching upstream
//!   `CalcRegularFeatureEffect`, `calc_fstr.cpp` v1.2.10): a `Ctr` split's
//!   `dif` (unchanged math) is divided EQUALLY across its projection's
//!   constituent categorical features' flat slots (no cross-product — one
//!   split's own effect, not a pair). The output vector widens to
//!   `n_float + cat_feature_count(model)` to hold the categorical slots.
//!   [`prediction_values_change_with_data`] is upstream's `data=pool` mode
//!   (`CollectLeavesStatistics`, `fstr/util.cpp`): the per-leaf weights are
//!   recomputed from the provided dataset via the APPLY path instead of the
//!   model's stored training-time `leaf_weights` — the two genuinely differ
//!   for online-CTR models (training-time online CTR values assign documents
//!   to different leaves than the final baked apply-time tables).
//!
//! `interaction`/`prediction_values_change`'s public signatures and return
//! TYPES are unchanged; only the returned index range widens for CTR-bearing
//! models. Out of scope: one-hot cat splits / float-in-CTR (`BinFeatures`) —
//! this codebase's `TProjection` is categorical-only, so upstream's
//! `BinFeatures`/`OneHotFeatures` expansion terms are vacuous here.
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

/// The number of DISTINCT categorical feature indices referenced by any
/// `ModelSplit::Ctr` split's projection, across BOTH tree kinds — the
/// categorical analogue of [`feature_count`] (SPEC FIC-01). `max(local cat
/// index) + 1`, mirroring `feature_count`'s "widen based on observed usage"
/// convention; `0` if the model has no CTR splits.
fn cat_feature_count(model: &Model) -> usize {
    let oblivious_max = model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter_map(|s| match s {
            crate::ModelSplit::Ctr(c) => c.projection.cat_features().iter().copied().max(),
            crate::ModelSplit::Float(_) => None,
        })
        .map(|m| m + 1)
        .max()
        .unwrap_or(0);
    let non_symmetric_max = model
        .non_symmetric_trees
        .iter()
        .flat_map(|t| t.tree_splits.iter())
        .filter_map(|s| match s {
            crate::ModelSplit::Ctr(c) => c.projection.cat_features().iter().copied().max(),
            crate::ModelSplit::Float(_) => None,
        })
        .map(|m| m + 1)
        .max()
        .unwrap_or(0);
    oblivious_max.max(non_symmetric_max)
}

/// The combined flat index (§4) for a categorical feature's LOCAL index `c`
/// (as it appears in `TProjection::cat_features()`), given the model's float
/// width `n_float` (from [`feature_count`]). Always `n_float + c` — floats
/// occupy `[0, n_float)`, categoricals occupy `[n_float, n_float +
/// cat_feature_count(model))` (SPEC FIC-01).
const fn flat_cat_index(n_float: usize, local_cat_index: usize) -> usize {
    n_float + local_cat_index
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
    pvc_impl(model, None)
}

/// `PredictionValuesChange` with a dataset — upstream's
/// `get_feature_importance(type='PredictionValuesChange', data=pool)` mode
/// (`CalcFeatureEffectAverageChange` with a dataset, `calc_fstr.cpp` v1.2.10):
/// the per-leaf weights are recomputed from the provided columns via the APPLY
/// path ([`crate::collect_leaves_statistics`], upstream
/// `CollectLeavesStatistics`) instead of the model's stored training-time
/// `leaf_weights`. For online-CTR models the two genuinely differ (documents
/// land in different leaves under training-time online CTR values than under
/// the final baked tables), so oracle parity against a `data=pool` fixture
/// REQUIRES this mode. `feature_values` / `cat_columns` follow the
/// [`crate::predict_raw_cat`] SoA convention (unit document weights).
#[must_use]
pub fn prediction_values_change_with_data(
    model: &Model,
    feature_values: &[Vec<f32>],
    cat_columns: &[Vec<String>],
) -> Vec<f64> {
    let stats = crate::apply::collect_leaves_statistics(model, feature_values, cat_columns);
    pvc_impl(model, Some(&stats))
}

/// Shared `PredictionValuesChange` body: `weights_override` is the
/// dataset-recomputed per-tree leaf statistics (oblivious trees first, then
/// non-symmetric, matching [`crate::collect_leaves_statistics`]'s layout);
/// `None` uses each tree's stored `leaf_weights` (upstream's no-dataset arm).
fn pvc_impl(model: &Model, weights_override: Option<&[Vec<f64>]>) -> Vec<f64> {
    let n_features = feature_count(model);
    // The combined flat-index width (SPEC §4, FIC-03): a `Ctr` split's `dif`
    // redistributes equally across its projection's categorical members'
    // flat slots, `[n_features, n_features + cat_feature_count(model))`.
    // `feature_count` itself stays LOCKED unchanged (regression discipline);
    // widening `res` here is the ONLY place this vector's length is fixed —
    // the two accumulate helpers below receive it as an already-sized slice
    // and cannot widen it themselves.
    let n_float = n_features;
    let mut res = vec![0.0_f64; n_features + cat_feature_count(model)];

    // OBLIVIOUS arm — the literal pre-6.6 bit-indexed loop (D-6.6-05).
    for (tree_idx, tree) in model.oblivious_trees.iter().enumerate() {
        let weights = weights_override
            .and_then(|ws| ws.get(tree_idx))
            .map_or(tree.leaf_weights.as_slice(), Vec::as_slice);
        pvc_accumulate_oblivious(tree, weights, &mut res, n_float);
    }
    // NON-SYMMETRIC arm — node-graph recursion (D-6.6-10). Its statistics
    // follow the oblivious trees' in `collect_leaves_statistics`'s layout.
    let nonsym_base = model.oblivious_trees.len();
    for (tree_idx, tree) in model.non_symmetric_trees.iter().enumerate() {
        let weights = weights_override
            .and_then(|ws| ws.get(nonsym_base + tree_idx))
            .map_or(tree.leaf_weights.as_slice(), Vec::as_slice);
        pvc_accumulate_non_symmetric(tree, weights, &mut res, n_float);
    }

    convert_to_percents(&mut res);
    res
}

/// The OBLIVIOUS `PredictionValuesChange` accumulation (`CalcEffect`,
/// `feature_str.h:233-270`) — bit-indexed leaf pairs `(leaf, leaf ^ (1<<bit))`.
/// The `count1`/`count2`/`avrg`/`dif` computation is BYTE-IDENTICAL to the
/// pre-6.6 loop (D-6.6-05); only the post-`dif` "which slot(s) does this add
/// to" step is CTR-aware (FIC-03): a `Float` split adds the FULL `dif` to its
/// single slot (unchanged); a `Ctr` split divides `dif` EQUALLY across its
/// projection's constituent categorical features' flat slots (no
/// cross-product — a single split's own effect, not a pair).
///
/// `leaf_weights` is passed explicitly (rather than read off the tree) so the
/// dataset-recomputed statistics of [`prediction_values_change_with_data`] can
/// substitute for the stored training-time weights.
fn pvc_accumulate_oblivious(
    tree: &crate::ObliviousTree,
    leaf_weights: &[f64],
    res: &mut [f64],
    n_float: usize,
) {
    let leaf_count = tree.leaf_values.len();
    // for (feature = 0; feature < tree.SrcFeatures.size(); ++feature)
    for (feature_bit, split) in tree.splits.iter().enumerate() {
        let target_slots = split_flat_indices(split, n_float);
        if target_slots.is_empty() {
            continue;
        }
        let add_effect_divisor = target_slots.len() as f64;
        // for (leafIdx = 0; leafIdx < tree.Leaves.size(); ++leafIdx)
        for leaf_idx in 0..leaf_count {
            let inverted = leaf_idx ^ (1usize << feature_bit);
            if inverted < leaf_idx {
                continue;
            }
            let count1 = leaf_weights.get(leaf_idx).copied().unwrap_or(0.0);
            let count2 = leaf_weights.get(inverted).copied().unwrap_or(0.0);
            if count1 == 0.0 || count2 == 0.0 {
                continue;
            }
            let val1 = tree.leaf_values.get(leaf_idx).copied().unwrap_or(0.0);
            let val2 = tree.leaf_values.get(inverted).copied().unwrap_or(0.0);

            let avrg = (val1 * count1 + val2 * count2) / (count1 + count2);
            let dif = (val1 - avrg).powi(2) * count1 + (val2 - avrg).powi(2) * count2;
            let add_effect = dif / add_effect_divisor;

            for &slot_idx in &target_slots {
                if let Some(slot) = res.get_mut(slot_idx) {
                    *slot += add_effect;
                }
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
/// accumulate `dif = (val1−avrg)²·c1 + (val2−avrg)²·c2` into `res[featureIdx]`
/// (FIC-03: a `Ctr` split's `dif` is instead divided EQUALLY across its
/// projection's constituent categorical features' flat slots — see
/// [`split_flat_indices`]), and store the merged parent `NodeInfo`. All
/// reductions are scalar adds in the upstream order; `.get(...)` everywhere
/// (depth-bounded by `node_count`, T-06.6-15).
///
/// `leaf_weights` is passed explicitly (see [`pvc_accumulate_oblivious`]) so
/// dataset-recomputed statistics can substitute for the stored weights.
fn pvc_accumulate_non_symmetric(
    tree: &crate::NonSymmetricTree,
    leaf_weights: &[f64],
    res: &mut [f64],
    n_float: usize,
) {
    use std::collections::HashMap;

    let node_count = tree.step_nodes.len();
    let mut node_info: HashMap<usize, NodeInfo> = HashMap::new();
    // (parent_node, left_child, right_child, target_slots)
    let mut triangles: Vec<(usize, usize, usize, Vec<usize>)> = Vec::new();

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
            let count = leaf_weights.get(leaf_id).copied().unwrap_or(0.0);
            node_info.insert(node_idx, NodeInfo { value, count });
            if left_diff == 0 && right_diff == 0 {
                continue;
            }
        }
        // Interior (possibly one-sided): record the triangle. FIC-03: the
        // node's split expansion (a `Float` split's single slot, or a `Ctr`
        // split's constituent cat features' flat slots) — an empty
        // expansion (malformed empty-projection `Ctr` split, defensive)
        // skips the node the same way the pre-FIC-03 code skipped a CTR
        // split entirely.
        let Some(split) = tree.tree_splits.get(node_idx) else {
            continue;
        };
        let target_slots = split_flat_indices(split, n_float);
        if target_slots.is_empty() {
            continue;
        }
        let left_child = node_idx.saturating_add(left_diff as usize);
        let right_child = node_idx.saturating_add(right_diff as usize);
        triangles.push((node_idx, left_child, right_child, target_slots));
    }

    // LIFO pop = bottom-up (children were pushed after their parent).
    while let Some((parent, left, right, target_slots)) = triangles.pop() {
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
        let add_effect = dif / target_slots.len() as f64;

        for &slot_idx in &target_slots {
            if let Some(slot) = res.get_mut(slot_idx) {
                *slot += add_effect;
            }
        }
        node_info.insert(parent, NodeInfo { value: avrg, count: sum_count });
    }
}

/// Expand a single [`crate::ModelSplit`] to the list of combined-flat feature
/// indices (SPEC §4) its effect attributes to: a `Float` split expands to its
/// single float-feature index; a `Ctr` split expands to every constituent
/// categorical feature's flat index (`flat_cat_index(n_float, c)` for each `c`
/// in `projection.cat_features()`). Shared by `interaction()`'s oblivious and
/// non-symmetric arms (FIC-02) and `prediction_values_change()`'s two
/// accumulate helpers (FIC-03) so the expansion rule is defined exactly once.
/// An empty result (a `Ctr` split with an empty projection — should not occur
/// from `Model::from_trained`, but defensively) means "attributes to nothing".
fn split_flat_indices(split: &crate::ModelSplit, n_float: usize) -> Vec<usize> {
    match split {
        crate::ModelSplit::Float(s) => vec![s.feature],
        crate::ModelSplit::Ctr(c) => c
            .projection
            .cat_features()
            .iter()
            .map(|&x| flat_cat_index(n_float, x))
            .collect(),
    }
}

/// Whether two splits test the SAME internal feature (upstream `TFeature`
/// equality, `GetFeature` in `feature_str.cpp` v1.2.10): a `Float` split's
/// internal identity is its float-feature index (every border of a float
/// feature is the same internal feature); a `Ctr` split's is its FULL
/// border-less CTR descriptor (`TModelCtr`: projection + ctr type + prior +
/// target border idx + shift + scale — the split `border` is NOT part of the
/// identity, it lives in the binary split, not the feature). Two splits of
/// different kinds are never the same internal feature.
fn same_internal_feature(a: &crate::ModelSplit, b: &crate::ModelSplit) -> bool {
    match (a, b) {
        (crate::ModelSplit::Float(x), crate::ModelSplit::Float(y)) => x.feature == y.feature,
        (crate::ModelSplit::Ctr(x), crate::ModelSplit::Ctr(y)) => {
            x.projection == y.projection
                && x.ctr_type == y.ctr_type
                && x.prior == y.prior
                && x.target_border_idx == y.target_border_idx
                && x.shift == y.shift
                && x.scale == y.scale
        }
        _ => false,
    }
}

/// Intern `split`'s internal feature into `internal` (first-seen order,
/// mirroring upstream `GetFeatureToIdxMap`'s insertion-order indexing) and
/// return its index. Linear scan — internal feature counts are tiny (bounded
/// by distinct split features, not trees × splits).
fn intern_internal_feature(internal: &mut Vec<crate::ModelSplit>, split: &crate::ModelSplit) -> usize {
    if let Some(pos) = internal.iter().position(|s| same_internal_feature(s, split)) {
        pos
    } else {
        internal.push(split.clone());
        internal.len() - 1
    }
}

/// Accumulate a contribution for a sorted key pair into the insertion-order
/// `(pairs, sums)` accumulator (a Vec keyed by pair rather than a hash map, so
/// the iteration order is deterministic; the final score sort is by value).
/// Used both for INTERNAL-feature pairs (the `sumInteractions` stage) and for
/// the expanded combined-flat pairs (the `CalcFeatureInteraction` stage).
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
    // Stage 1 (upstream `CalcInternalFeatureInteraction`): accumulate
    // per-INTERNAL-feature-pair scores. `internal` interns each split's
    // border-less internal identity ([`same_internal_feature`], first-seen
    // order); `int_pairs`/`int_sums` are keyed by sorted internal-index pairs.
    let mut internal: Vec<crate::ModelSplit> = Vec::new();
    let mut int_pairs: Vec<(usize, usize)> = Vec::new();
    let mut int_sums: Vec<f64> = Vec::new();
    // The combined flat-index float width (SPEC §4) — a CTR split's
    // categorical members are attributed to `[n_float, n_float +
    // cat_feature_count(model))`.
    let n_float = feature_count(model);

    // OBLIVIOUS arm — the literal bit-indexed loop (`CalcMostInteractingFeatures`,
    // `feature_str.cpp:190-223`; D-6.6-05).
    for tree in &model.oblivious_trees {
        let split_count = tree.splits.len();
        let leaf_count = tree.leaf_values.len();
        // for (firstIdx = 0; firstIdx < splits-1; ++firstIdx)
        for first_idx in 0..split_count.saturating_sub(1) {
            for second_idx in (first_idx + 1)..split_count {
                let (Some(split1), Some(split2)) =
                    (tree.splits.get(first_idx), tree.splits.get(second_idx))
                else {
                    continue;
                };
                let src1 = intern_internal_feature(&mut internal, split1);
                let src2 = intern_internal_feature(&mut internal, split2);
                // Two borders of the SAME internal feature interact with
                // nothing — upstream `srcFeature1 == srcFeature2 → continue`
                // BEFORE any accumulation, so the pair contributes to neither
                // the output nor `totalEffect`.
                if src1 == src2 {
                    continue;
                }

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

                let (a, b) = if src1 < src2 { (src1, src2) } else { (src2, src1) };
                interaction_add(&mut int_pairs, &mut int_sums, a, b, delta.abs());
            }
        }
    }

    // NON-SYMMETRIC arm — per-tree signed DFS accumulation keyed by internal
    // pair, then `|·|` per tree (D-6.6-10).
    for tree in &model.non_symmetric_trees {
        interaction_accumulate_non_symmetric(tree, &mut internal, &mut int_pairs, &mut int_sums);
    }

    // Stage 2 (`CalcFeatureInteraction`, `calc_fstr.cpp:343-414`): expand each
    // internal pair to its combined-flat cross-product. Self-cells (`f0 ==
    // f1`) are dropped from the OUTPUT, but the pair's FULL score still
    // enters `totalEffect` (upstream `totalEffect += effect` sits OUTSIDE the
    // cross-product loops) — so e.g. a simple-CTR × combination-CTR pair
    // sharing a cat feature keeps its dropped self-cell mass in the
    // denominator, deflating every returned percentage (AT-FIC02d's fixture
    // demonstrates exactly this: upstream's five scores sum to ~92.5, not 100).
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut sums: Vec<f64> = Vec::new();
    let mut total_parts: Vec<f64> = Vec::new();
    for (&(ia, ib), &score) in int_pairs.iter().zip(int_sums.iter()) {
        let (Some(feat_a), Some(feat_b)) = (internal.get(ia), internal.get(ib)) else {
            continue;
        };
        let side0 = split_flat_indices(feat_a, n_float);
        let side1 = split_flat_indices(feat_b, n_float);
        // Defensive (an empty-projection `Ctr` cannot occur upstream): skip
        // the pair entirely — nothing in the output, nothing in the total.
        if side0.is_empty() || side1.is_empty() {
            continue;
        }
        let divisor = (side0.len() * side1.len()) as f64;
        for &f0 in &side0 {
            for &f1 in &side1 {
                if f0 == f1 {
                    continue;
                }
                let (a, b) = if f0 < f1 { (f0, f1) } else { (f1, f0) };
                interaction_add(&mut pairs, &mut sums, a, b, score / divisor);
            }
        }
        total_parts.push(score);
    }

    // score = sum / totalEffect * 100.
    let total_effect = sum_f64(&total_parts);
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
/// Per tree: DFS from the root node carrying a `path` of `(internal_idx, sign)`
/// — INTERNAL feature indices (upstream `featureToIdx.at(feature)`), NOT
/// expanded flat indices; expansion to combined-flat pairs happens once, in
/// `interaction()`'s `CalcFeatureInteraction` stage. At a TERMINAL node
/// (`left_child == node || right_child == node`, i.e. one diff is 0)
/// accumulate `treeSum[(i1,i2)] += sign1·sign2·delta` for every path pair with
/// DISTINCT internal features (upstream's `srcFeature1 == srcFeature2 →
/// continue`; `delta` = Σ leaf values over the dimension; 1-dim ⇒ the single
/// value). Descending into the left child uses `sign = -1`, the right child
/// `sign = +1`. After the tree, fold `Σ|treeSum[pair]|` into the shared
/// internal-pair accumulator. The walk is `node_count`-depth-bounded with
/// checked `.get` (T-06.6-15) — a malformed graph stops the descent rather
/// than recursing forever.
fn interaction_accumulate_non_symmetric(
    tree: &crate::NonSymmetricTree,
    internal: &mut Vec<crate::ModelSplit>,
    int_pairs: &mut Vec<(usize, usize)>,
    int_sums: &mut Vec<f64>,
) {
    let node_count = tree.step_nodes.len();
    // Per-tree signed accumulator (insertion-order keyed Vec, like the shared one).
    let mut tree_pairs: Vec<(usize, usize)> = Vec::new();
    let mut tree_sums: Vec<f64> = Vec::new();
    let mut path: Vec<(usize, i32)> = Vec::new();

    interaction_dfs(tree, 0, node_count, internal, &mut path, &mut tree_pairs, &mut tree_sums);

    // sumInteractions[pair] += |treeSum[pair]|. This is the SOLE place
    // magnitude is taken for this arm (FIC-02) — the terminal accumulation
    // stays SIGNED (never `.abs()`'d per-leaf), so same-pair contributions
    // from different leaves along the path can partially cancel before this
    // deferred `.abs()`.
    for (&pair, &signed) in tree_pairs.iter().zip(tree_sums.iter()) {
        interaction_add(int_pairs, int_sums, pair.0, pair.1, signed.abs());
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
    internal: &mut Vec<crate::ModelSplit>,
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

    // Terminal node: accumulate `sign1·sign2·delta` for every path-entry PAIR
    // with DISTINCT internal features (upstream's `srcFeature1 == srcFeature2
    // → continue`; the combined-flat cross-product expansion — including its
    // per-cell self-pair skip for PARTIAL overlaps, AT-FIC02e — happens once
    // in `interaction()`'s `CalcFeatureInteraction` stage, NOT here). For a
    // 1-dim numeric model `delta` is the single leaf value at the node's leaf
    // id. **STILL SIGNED here** (do NOT take `.abs()` at this call site) —
    // `interaction_accumulate_non_symmetric`'s `signed.abs()` remains the
    // sole place magnitude is taken for this arm.
    if left_child == node_idx || right_child == node_idx {
        let leaf_id = tree
            .node_id_to_leaf_id
            .get(node_idx)
            .copied()
            .map_or(usize::MAX, |v| v as usize);
        let delta = tree.leaf_values.get(leaf_id).copied().unwrap_or(0.0);

        for first in 0..path.len() {
            for second in (first + 1)..path.len() {
                let (Some(&(src1, sign1)), Some(&(src2, sign2))) =
                    (path.get(first), path.get(second))
                else {
                    continue;
                };
                if src1 == src2 {
                    continue;
                }
                let sign = f64::from(sign1 * sign2);
                let (a, b) = if src1 < src2 { (src1, src2) } else { (src2, src1) };
                interaction_add(tree_pairs, tree_sums, a, b, sign * delta);
            }
        }
    }

    // A pure-leaf node carries a placeholder `Float { feature: 0, .. }` split
    // (its step entry is `(0, 0)`); short-circuit on that sentinel BEFORE reading
    // the split so the placeholder is never mistaken for real feature 0
    // (WR-04). Behaviour-preserving: a `(0, 0)` node has both diffs zero, so the
    // descent loop below was already a no-op for it.
    if matches!(tree.step_nodes.get(node_idx), Some(&(0, 0))) {
        return;
    }

    // The node's own split, interned to its internal-feature index (FIC-02).
    let Some(split) = tree.tree_splits.get(node_idx) else {
        return;
    };
    let src = intern_internal_feature(internal, split);

    // Descend: left child sign = -1, right child sign = +1 (the upstream
    // `sign = -1; ... sign *= -1` order over `{left, right}`).
    let mut sign: i32 = -1;
    for &child_idx in &[left_child, right_child] {
        if child_idx != node_idx {
            path.push((src, sign));
            interaction_dfs(
                tree,
                child_idx,
                remaining_depth - 1,
                internal,
                path,
                tree_pairs,
                tree_sums,
            );
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
pub fn loss_function_change<F: Fn(&[f64], &[f64]) -> f64>(
    model: &Model,
    cols: &[Vec<f32>],
    labels: &[f64],
    n_features: usize,
    final_error: F,
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

    // finalError(approx) via the CALLER-supplied metric closure (its
    // `GetFinalError`); the metric MUST be Min-optimized so the per-feature
    // difference below is the importance verbatim (FL-01). Kept metric-agnostic
    // so this crate needs no metric implementation — the facade owns the
    // loss → closure mapping.
    let base_score = final_error(&approx, labels);

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
            // score = finalError(without feature f) − finalError(full). The
            // metric best value is Min, so the difference is the importance
            // verbatim.
            final_error(&approx_f, labels) - base_score
        })
        .collect()
}

/// Logloss-defaulted convenience wrapper — the pre-FL-01 behavior verbatim,
/// retained so existing binary-model callers/tests stay byte-identical. Delegates
/// to the generic [`loss_function_change`] with the built-in
/// [`logloss_final_error`] closure (`FL-02` back-compat).
#[must_use]
pub fn loss_function_change_logloss(
    model: &Model,
    cols: &[Vec<f32>],
    labels: &[f64],
    n_features: usize,
) -> Vec<f64> {
    loss_function_change(model, cols, labels, n_features, logloss_final_error)
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

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file), mirroring
// `crates/cb-model/src/ctr_data.rs:58-61`.
#[cfg(test)]
#[path = "fstr_test.rs"]
mod tests;
