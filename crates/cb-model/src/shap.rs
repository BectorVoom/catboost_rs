//! Regular TreeSHAP (`EShapCalcType::Regular`) for the canonical oblivious
//! [`crate::Model`] (MODEL-04, D-11).
//!
//! Produces the per-object `[n_features + 1]` SHAP matrix: column `f` is the
//! Shapley contribution of feature `f`, and the trailing column `[n_features]` is
//! the expected-value / bias term (`Σ_trees meanValue + model bias`). The
//! **local-accuracy invariant** holds by construction: `Σ_columns shap[obj]`
//! equals the [`crate::predict_raw`] `RawFormulaVal` for that object.
//!
//! # Source of truth (RESEARCH Pattern 4)
//!
//! The polynomial-time TreeSHAP recursion is transcribed VERBATIM from upstream
//! catboost 1.2.10:
//!
//! - **prepared-trees** `CalcSubtreeWeightsForTree` + `CalcMeanValueForTree`
//!   (`shap_prepared_trees.cpp:25-67,177-223`): `subtree_weights[depth][node]` is
//!   the bottom-up sum of `leaf_weights` (leaves = `leaf_weights`, internal = sum
//!   of children); `mean_value = (Σ leafValue·leafWeight) / subtree_weights[0][0]`
//!   is the per-tree weighted-average leaf value = the `averageTreeApprox`
//!   baseline.
//! - **`ExtendFeaturePath` / `UnwindFeaturePath`** (`shap_values.cpp:44-104`): the
//!   feature-path polynomial weight machinery, transcribed exactly (the
//!   `FuzzyEquals(1+oneFrac, 1+0)` branch in unwind uses [`fuzzy_equals`]).
//! - **`CalcObliviousInternalShapValuesForLeafRecursive`**
//!   (`shap_values.cpp:196-320`): at each split `hotCoefficient =
//!   subtree_weights[d+1][goNode]/subtree_weights[d][node]`, `coldCoefficient =
//!   subtree_weights[d+1][skipNode]/...`; the go-branch carries
//!   `oneFrac = newOnePathsFraction`, the skip-branch carries `oneFrac = 0`.
//! - **`UpdateShapByFeaturePath`** (`shap_values.cpp:106-146`): at the leaf,
//!   distributes `coefficient = weightSum·(oneFrac − zeroFrac)` ×
//!   `(leafValue − averageTreeApprox)` per path feature.
//! - **matrix assembly** (`shap_values.cpp:1030-1055`): the trailing column =
//!   `Σ_trees meanValue + bias`.
//!
//! For numeric-only models the `binFeatureCombinationClass` indirection is the
//! identity (each float bin-feature maps to its own feature, A3), so a split's
//! "combination class" IS its float-feature index, and `UnpackInternalShaps`
//! (`shap_values.cpp:459-491`) divides by a flat-feature-count of 1 (a no-op).
//! SHAP requires `scale == 1` (`CB_ENSURE_SCALE_IDENTITY`), always true in
//! Phase 4.
//!
//! # Parity discipline
//!
//! Every weight fold routes through [`cb_core::sum_f64`] (D-08 — never a raw
//! `iter().sum()` / `fold(0.0, …)`). The index-heavy recursion uses checked
//! `.get` / `.get_mut` throughout (`indexing_slicing` deny, T-04-04-01); no
//! `unwrap`/`expect`. A missing/zero subtree weight is guarded exactly as
//! upstream (the `FuzzyEquals(.., 0)` short-circuits), so no NaN leaks
//! (T-04-04-02).

use cb_core::sum_f64;

use crate::{Model, NonSymmetricTree};

/// `TFixedFeatureParams` (`shap_values.h`): when computing ShapInteractionValues
/// the recursion is re-run with one feature (combination class) held FIXED — on
/// every path (`FixedOn`) or off every path (`FixedOff`). `None` is the plain
/// regular SHAP calc (`shap_interaction_values.cpp` runs the on/off pair to form
/// `Φ(i,j) = (contribOn_j − contribOff_j) / 2`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum FixedMode {
    On,
    Off,
}

/// `(feature, mode)` — the combination class to fix and how.
#[derive(Clone, Copy)]
struct FixedFeature {
    feature: i64,
    mode: FixedMode,
}

/// `TConditionsFeatureFraction` (`shap_values.cpp:148-173`): the Hot / Cold
/// condition-feature fractions at a split, given the optional fixed-feature
/// params and the hot/cold subtree coefficients.
struct ConditionFractions {
    hot: f64,
    cold: f64,
}

impl ConditionFractions {
    fn new(
        fixed: Option<FixedFeature>,
        combination_class: i64,
        condition_feature_fraction: f64,
        hot_coefficient: f64,
        cold_coefficient: f64,
    ) -> Self {
        let mut hot = condition_feature_fraction;
        let mut cold = condition_feature_fraction;
        if let Some(f) = fixed {
            if f.feature == combination_class {
                match f.mode {
                    FixedMode::On => cold = 0.0,
                    FixedMode::Off => {
                        hot *= hot_coefficient;
                        cold *= cold_coefficient;
                    }
                }
            }
        }
        Self { hot, cold }
    }
}

/// `FuzzyEquals(p1, p2)` from `util/generic/ymath.h:163`:
/// `|p1 − p2| <= eps · min(|p1|, |p2|)` with `eps = 1e-13`. Used only in the
/// `FuzzyEquals(1 + x, 1 + 0.0)` form (an "is `x` ~ 0?" test), exactly as
/// upstream's `UnwindFeaturePath` / recursion short-circuits do.
fn fuzzy_equals(p1: f64, p2: f64) -> bool {
    const EPS: f64 = 1.0e-13;
    (p1 - p2).abs() <= EPS * p1.abs().min(p2.abs())
}

/// One element of a SHAP feature path (`TFeaturePathElement`,
/// `shap_values.cpp:26-41`).
#[derive(Clone, Copy)]
struct Elem {
    /// The feature (combination class) this element conditions on; `-1` for the
    /// synthetic root element.
    feature: i64,
    zero_frac: f64,
    one_frac: f64,
    weight: f64,
}

/// `ExtendFeaturePath(oldPath, zeroFrac, oneFrac, feature)`
/// (`shap_values.cpp:44-64`) — append an element and back-propagate the
/// polynomial weights. Transcribed verbatim with checked access.
fn extend_feature_path(old: &[Elem], zero_frac: f64, one_frac: f64, feature: i64) -> Vec<Elem> {
    let path_length = old.len();
    let mut new_path: Vec<Elem> = Vec::with_capacity(path_length + 1);
    new_path.extend_from_slice(old);
    new_path.push(Elem {
        feature,
        zero_frac,
        one_frac,
        weight: if path_length == 0 { 1.0 } else { 0.0 },
    });

    // for (elementIdx = pathLength - 1; elementIdx >= 0; --elementIdx)
    for element_idx in (0..path_length).rev() {
        let len = path_length as f64;
        let i = element_idx as f64;
        let prev_weight = new_path.get(element_idx).map_or(0.0, |e| e.weight);
        // newFeaturePath[elementIdx + 1].Weight += oneFrac * prev * (i+1)/(L+1)
        if let Some(e) = new_path.get_mut(element_idx + 1) {
            e.weight += one_frac * prev_weight * (i + 1.0) / (len + 1.0);
        }
        // newFeaturePath[elementIdx].Weight = zeroFrac * prev * (L-i)/(L+1)
        if let Some(e) = new_path.get_mut(element_idx) {
            e.weight = zero_frac * prev_weight * (len - i) / (len + 1.0);
        }
    }

    new_path
}

/// `UnwindFeaturePath(oldPath, eraseElementIdx)` (`shap_values.cpp:66-104`) — the
/// inverse of extend, with the two branches on `oneFrac == 0` (`fuzzy_equals`).
/// Returns the path with one element removed. Transcribed verbatim with checked
/// access; on an empty path (which upstream `CB_ENSURE`-rejects) returns empty.
fn unwind_feature_path(old: &[Elem], erase_element_idx: usize) -> Vec<Elem> {
    let path_length = old.len();
    if path_length == 0 {
        return Vec::new();
    }

    // newFeaturePath = old[0 .. pathLength-1]
    let mut new_path: Vec<Elem> = old
        .get(..path_length - 1)
        .map(<[Elem]>::to_vec)
        .unwrap_or_default();

    let one_paths_fraction = old.get(erase_element_idx).map_or(0.0, |e| e.one_frac);
    let zero_paths_fraction = old.get(erase_element_idx).map_or(0.0, |e| e.zero_frac);
    let mut weight_diff = old.get(path_length - 1).map_or(0.0, |e| e.weight);

    if !fuzzy_equals(1.0 + one_paths_fraction, 1.0 + 0.0) {
        // for (elementIdx = pathLength - 2; elementIdx >= 0; --elementIdx)
        for element_idx in (0..path_length - 1).rev() {
            let len = path_length as f64;
            let i = element_idx as f64;
            let old_weight = new_path.get(element_idx).map_or(0.0, |e| e.weight);
            let new_weight = weight_diff * len / (one_paths_fraction * (i + 1.0));
            if let Some(e) = new_path.get_mut(element_idx) {
                e.weight = new_weight;
            }
            weight_diff = old_weight - new_weight * zero_paths_fraction * (len - i - 1.0) / len;
        }
    } else {
        for element_idx in (0..path_length - 1).rev() {
            let len = path_length as f64;
            let i = element_idx as f64;
            if let Some(e) = new_path.get_mut(element_idx) {
                e.weight *= len / (zero_paths_fraction * (len - i - 1.0));
            }
        }
    }

    // Shift Feature/ZeroFrac/OneFrac down from eraseElementIdx (the value fields
    // move; weights were recomputed above).
    // for (elementIdx = eraseElementIdx; elementIdx < pathLength - 1; ++elementIdx)
    for element_idx in erase_element_idx..path_length - 1 {
        let (feature, zero_frac, one_frac) = old
            .get(element_idx + 1)
            .map_or((-1, 0.0, 0.0), |e| (e.feature, e.zero_frac, e.one_frac));
        if let Some(e) = new_path.get_mut(element_idx) {
            e.feature = feature;
            e.zero_frac = zero_frac;
            e.one_frac = one_frac;
        }
    }

    new_path
}

/// `UpdateShapByFeaturePath` (`shap_values.cpp:106-146`) for a 1-dim oblivious
/// model: at the reached leaf, distribute the coefficient to each path feature's
/// running SHAP value. `shap_by_feature[f]` accumulates feature `f`'s
/// contribution. `condition_feature_fraction` is `1.0` for the regular calc.
fn update_shap_by_feature_path(
    feature_path: &[Elem],
    leaf_value: f64,
    average_tree_approx: f64,
    condition_feature_fraction: f64,
    shap_by_feature: &mut [f64],
) {
    // for (elementIdx = 1; elementIdx < featurePath.size(); ++elementIdx)
    for element_idx in 1..feature_path.len() {
        let unwound = unwind_feature_path(feature_path, element_idx);
        // weightSum = Σ unwound[*].Weight  (order-locked, D-08)
        let weights: Vec<f64> = unwound.iter().map(|e| e.weight).collect();
        let weight_sum = sum_f64(&weights);

        let Some(element) = feature_path.get(element_idx) else {
            continue;
        };
        let coefficient =
            condition_feature_fraction * weight_sum * (element.one_frac - element.zero_frac);
        let add_value = coefficient * (leaf_value - average_tree_approx);

        // feature == -1 is the synthetic root and is never reached here (idx >= 1
        // always points at a real conditioning feature), but guard anyway.
        if let Ok(f) = usize::try_from(element.feature) {
            if let Some(slot) = shap_by_feature.get_mut(f) {
                *slot += add_value;
            }
        }
    }
}

/// `subtree_weights[depth][node]` for one oblivious tree
/// (`CalcSubtreeWeightsForTree`, `shap_prepared_trees.cpp:177-223`): the
/// full-depth row equals `leaf_weights` (indexed by leaf in forward-bit order),
/// each shallower row sums child pairs `child[2·node] + child[2·node+1]`.
fn calc_subtree_weights(leaf_weights: &[f64], tree_depth: usize) -> Vec<Vec<f64>> {
    let mut subtree: Vec<Vec<f64>> = vec![Vec::new(); tree_depth + 1];

    // Full-depth row: one entry per leaf.
    let leaf_count = 1usize << tree_depth;
    let mut bottom = vec![0.0_f64; leaf_count];
    for node_idx in 0..leaf_count {
        let w = leaf_weights.get(node_idx).copied().unwrap_or(0.0);
        if let Some(slot) = bottom.get_mut(node_idx) {
            *slot = w;
        }
    }
    if let Some(row) = subtree.get_mut(tree_depth) {
        *row = bottom;
    }

    // for (depth = treeDepth - 1; depth >= 0; --depth)
    for depth in (0..tree_depth).rev() {
        let node_count = 1usize << depth;
        let mut parent = vec![0.0_f64; node_count];
        for node_idx in 0..node_count {
            let (l, r) = subtree.get(depth + 1).map_or((0.0, 0.0), |child| {
                (
                    child.get(node_idx * 2).copied().unwrap_or(0.0),
                    child.get(node_idx * 2 + 1).copied().unwrap_or(0.0),
                )
            });
            if let Some(slot) = parent.get_mut(node_idx) {
                *slot = l + r;
            }
        }
        if let Some(row) = subtree.get_mut(depth) {
            *row = parent;
        }
    }

    subtree
}

/// `meanValue` for one oblivious 1-dim tree (`CalcMeanValueForTree`,
/// `shap_prepared_trees.cpp:25-67`): `(Σ_leaf leafValue·leafWeight) /
/// subtree_weights[0][0]`. Returns `0.0` if the tree has no weight (the upstream
/// `FuzzyEquals`-guarded paths leave it at zero), avoiding a div-by-zero NaN
/// (T-04-04-02).
fn calc_mean_value(
    leaf_values: &[f64],
    subtree_weights: &[Vec<f64>],
    tree_depth: usize,
) -> f64 {
    let leaf_count = 1usize << tree_depth;
    let bottom = subtree_weights.get(tree_depth);
    let products: Vec<f64> = (0..leaf_count)
        .map(|leaf_idx| {
            let v = leaf_values.get(leaf_idx).copied().unwrap_or(0.0);
            let w = bottom
                .and_then(|row| row.get(leaf_idx))
                .copied()
                .unwrap_or(0.0);
            v * w
        })
        .collect();
    let numerator = sum_f64(&products);
    let total = subtree_weights
        .first()
        .and_then(|row| row.first())
        .copied()
        .unwrap_or(0.0);
    if fuzzy_equals(1.0 + total, 1.0 + 0.0) {
        0.0
    } else {
        numerator / total
    }
}

/// The recursion `CalcObliviousInternalShapValuesForLeafRecursive`
/// (`shap_values.cpp:196-320`) for a 1-dim numeric-only oblivious tree.
///
/// `document_leaf_idx` is the (forward-bit-order) leaf the object falls into.
/// `depth`/`node_idx` walk the subtree-weight tree; the split feature at this
/// level is `splits[remaining_depth]` where `remaining_depth = tree_size − depth
/// − 1`. `combination_class` for numeric-only == the split's float-feature index.
#[allow(clippy::too_many_arguments)]
fn shap_recurse(
    splits: &[crate::Split],
    leaf_values: &[f64],
    subtree_weights: &[Vec<f64>],
    average_tree_approx: f64,
    tree_size: usize,
    document_leaf_idx: usize,
    depth: usize,
    node_idx: usize,
    old_feature_path: &[Elem],
    zero_paths_fraction: f64,
    one_paths_fraction: f64,
    feature: i64,
    fixed: Option<FixedFeature>,
    condition_feature_fraction: f64,
    shap_by_feature: &mut [f64],
) {
    if fuzzy_equals(1.0 + condition_feature_fraction, 1.0 + 0.0) {
        return;
    }

    // ExtendFeaturePathIfFeatureNotFixed (shap_values.cpp:175-194): when this
    // feature is the fixed one, the path is NOT extended (the element is
    // suppressed); otherwise extend as usual.
    let feature_path = extend_feature_path_if_not_fixed(
        fixed,
        old_feature_path,
        zero_paths_fraction,
        one_paths_fraction,
        feature,
    );

    if depth == tree_size {
        // Leaf reached: distribute SHAP. node_idx is the leaf index here.
        let leaf_value = leaf_values.get(node_idx).copied().unwrap_or(0.0);
        update_shap_by_feature_path(
            &feature_path,
            leaf_value,
            average_tree_approx,
            condition_feature_fraction,
            shap_by_feature,
        );
        return;
    }

    let mut new_zero_paths_fraction = 1.0_f64;
    let mut new_one_paths_fraction = 1.0_f64;

    let remaining_depth = tree_size - depth - 1;
    // combinationClass for numeric-only == the split's float feature index.
    let combination_class: i64 = splits
        .get(remaining_depth)
        .and_then(|s| i64::try_from(s.feature).ok())
        .unwrap_or(-1);

    // If this feature already conditions the path, unwind it first.
    let mut feature_path = feature_path;
    if let Some(same_idx) = feature_path
        .iter()
        .position(|e| e.feature == combination_class)
    {
        if let Some(e) = feature_path.get(same_idx) {
            new_zero_paths_fraction = e.zero_frac;
            new_one_paths_fraction = e.one_frac;
        }
        feature_path = unwind_feature_path(&feature_path, same_idx);
    }

    let is_go_right = (document_leaf_idx >> remaining_depth) & 1;
    let go_node_idx = node_idx * 2 + is_go_right;
    let skip_node_idx = node_idx * 2 + (1 - is_go_right);

    let parent_weight = subtree_weights
        .get(depth)
        .and_then(|row| row.get(node_idx))
        .copied()
        .unwrap_or(0.0);
    let go_weight = subtree_weights
        .get(depth + 1)
        .and_then(|row| row.get(go_node_idx))
        .copied()
        .unwrap_or(0.0);
    let skip_weight = subtree_weights
        .get(depth + 1)
        .and_then(|row| row.get(skip_node_idx))
        .copied()
        .unwrap_or(0.0);

    let hot_coefficient = if fuzzy_equals(1.0 + parent_weight, 1.0 + 0.0) {
        0.0
    } else {
        go_weight / parent_weight
    };
    let cold_coefficient = if fuzzy_equals(1.0 + parent_weight, 1.0 + 0.0) {
        0.0
    } else {
        skip_weight / parent_weight
    };

    let fractions = ConditionFractions::new(
        fixed,
        combination_class,
        condition_feature_fraction,
        hot_coefficient,
        cold_coefficient,
    );

    if !fuzzy_equals(1.0 + go_weight, 1.0 + 0.0) {
        let new_zero_go = new_zero_paths_fraction * hot_coefficient;
        shap_recurse(
            splits,
            leaf_values,
            subtree_weights,
            average_tree_approx,
            tree_size,
            document_leaf_idx,
            depth + 1,
            go_node_idx,
            &feature_path,
            new_zero_go,
            new_one_paths_fraction,
            combination_class,
            fixed,
            fractions.hot,
            shap_by_feature,
        );
    }

    if !fuzzy_equals(1.0 + skip_weight, 1.0 + 0.0) {
        let new_zero_skip = new_zero_paths_fraction * cold_coefficient;
        shap_recurse(
            splits,
            leaf_values,
            subtree_weights,
            average_tree_approx,
            tree_size,
            document_leaf_idx,
            depth + 1,
            skip_node_idx,
            &feature_path,
            new_zero_skip,
            /* onePathFraction */ 0.0,
            combination_class,
            fixed,
            fractions.cold,
            shap_by_feature,
        );
    }
}

/// `ExtendFeaturePathIfFeatureNotFixed` (`shap_values.cpp:175-194`): extend the
/// path UNLESS this `feature` is the fixed-feature combination class (in which
/// case the path is passed through unchanged, suppressing the element).
fn extend_feature_path_if_not_fixed(
    fixed: Option<FixedFeature>,
    old: &[Elem],
    zero_frac: f64,
    one_frac: f64,
    feature: i64,
) -> Vec<Elem> {
    let suppressed = fixed.is_some_and(|f| f.feature == feature);
    if suppressed {
        old.to_vec()
    } else {
        extend_feature_path(old, zero_frac, one_frac, feature)
    }
}

/// The forward-bit-order leaf index of one object in one tree: split `i`
/// contributes bit `i` via the strict `value > border` test (the same evaluator
/// the apply path and trainer use). Out-of-range feature indices read as a
/// missing value (test `false`), checked `.get` only.
fn document_leaf_index(splits: &[crate::Split], row: &[f32]) -> usize {
    let mut idx = 0usize;
    for (i, split) in splits.iter().enumerate() {
        let passes = row
            .get(split.feature)
            .is_some_and(|&v| f64::from(v) > split.border);
        if passes {
            idx |= 1usize << i;
        }
    }
    idx
}

/// Regular TreeSHAP for `model` over the numeric SoA feature columns `cols`
/// (`cols[f]` is float feature `f`'s per-object `f32` column — the layout the
/// trainer / Plan-05 Builder feed). `n_features` is the flat feature count (the
/// width of the SHAP feature block; the returned rows are `n_features + 1` wide,
/// the trailing column being the expected-value / bias term).
///
/// Returns one `[n_features + 1]` row per object (object order). By construction
/// the local-accuracy invariant holds: `Σ_columns row == predict_raw[obj]`
/// (D-11).
#[must_use]
pub fn shap_values(model: &Model, cols: &[Vec<f32>], n_features: usize) -> Vec<Vec<f64>> {
    shap_values_fixed(model, cols, n_features, None)
}

/// Per-tree prepared SHAP data: subtree weights and the mean (baseline) value.
/// For an oblivious tree `subtree_weights[depth][node]` is the perfect-binary
/// weight tree; for a non-symmetric tree `subtree_weights[0][local_node]` is the
/// single-row node-graph weight array (`shap_prepared_trees.cpp:204-220`).
struct Prepared {
    subtree_weights: Vec<Vec<f64>>,
    mean_value: f64,
    tree_depth: usize,
}

/// Project a tree's `Vec<ModelSplit>` to its float `Vec<Split>` (numeric-only
/// SHAP — a CTR split has no single float-feature index and is out of scope).
fn float_splits_of(splits: &[crate::ModelSplit]) -> Vec<crate::Split> {
    splits.iter().filter_map(crate::ModelSplit::as_float).copied().collect()
}

/// `GetReversedSubtreeForNonObliviousTree` (`shap_prepared_trees.cpp:69-94`):
/// for each LOCAL node, its parent's local index (root maps to 0). A node with
/// `left_diff != 0` is parent of `node + left_diff`; likewise for `right_diff`.
/// One-sided halt `(d, 0)` / `(0, d)` nodes contribute only the non-zero side.
fn reversed_tree(tree: &NonSymmetricTree) -> Vec<usize> {
    let n = tree.step_nodes.len();
    let mut reversed = vec![0usize; n];
    for (node_idx, &(left_diff, right_diff)) in tree.step_nodes.iter().enumerate() {
        if left_diff != 0 {
            if let Some(slot) = reversed.get_mut(node_idx + left_diff as usize) {
                *slot = node_idx;
            }
        }
        if right_diff != 0 {
            if let Some(slot) = reversed.get_mut(node_idx + right_diff as usize) {
                *slot = node_idx;
            }
        }
    }
    reversed
}

/// Non-symmetric subtree weights (`CalcSubtreeWeightsForTree` non-oblivious arm,
/// `shap_prepared_trees.cpp:204-220`): a single row `subtree_weights[0][local]`.
/// Each terminal node seeds its own `leaf_weights[leaf_id]`; processed in reverse
/// local order, every node folds its accumulated weight into its parent. `leaf_id`
/// is already LOCAL (06.6-05 decode), so it indexes `leaf_weights` directly.
fn calc_subtree_weights_non_symmetric(tree: &NonSymmetricTree) -> Vec<Vec<f64>> {
    let n = tree.step_nodes.len();
    let mut weights = vec![0.0_f64; n];
    let reversed = reversed_tree(tree);

    if n == 1 {
        let leaf_id = tree.node_id_to_leaf_id.first().copied().unwrap_or(u32::MAX) as usize;
        let w = tree.leaf_weights.get(leaf_id).copied().unwrap_or(0.0);
        if let Some(slot) = weights.get_mut(0) {
            *slot = w;
        }
        return vec![weights];
    }

    // for (local = n-1; local > 0; --local): add own leaf weight (terminal nodes
    // only, via a valid leaf id), then propagate to the parent.
    for local in (1..n).rev() {
        let leaf_id = tree.node_id_to_leaf_id.get(local).copied().unwrap_or(u32::MAX) as usize;
        if let Some(lw) = tree.leaf_weights.get(leaf_id).copied() {
            if let Some(slot) = weights.get_mut(local) {
                *slot += lw;
            }
        }
        let parent = reversed.get(local).copied().unwrap_or(0);
        let own = weights.get(local).copied().unwrap_or(0.0);
        if let Some(slot) = weights.get_mut(parent) {
            *slot += own;
        }
    }
    vec![weights]
}

/// Non-symmetric mean (baseline) value (`CalcMeanValueForTree` non-oblivious arm,
/// `shap_prepared_trees.cpp:44-64`): `Σ_terminal-nodes leafValue·leafWeight /
/// subtree_weights[0][0]`. Iterating every node, only valid (non-`u32::MAX`) leaf
/// ids contribute; the divide is guarded against a zero total (T-04-04-02).
fn calc_mean_value_non_symmetric(tree: &NonSymmetricTree, subtree_weights: &[Vec<f64>]) -> f64 {
    let products: Vec<f64> = tree
        .node_id_to_leaf_id
        .iter()
        .filter_map(|&id| {
            let leaf_id = id as usize;
            let v = tree.leaf_values.get(leaf_id).copied()?;
            let w = tree.leaf_weights.get(leaf_id).copied()?;
            Some(v * w)
        })
        .collect();
    let numerator = sum_f64(&products);
    let total = subtree_weights.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
    if fuzzy_equals(1.0 + total, 1.0 + 0.0) {
        0.0
    } else {
        numerator / total
    }
}

/// Per-node `isGoRight` for one object's non-symmetric traversal: at each node
/// with a float split, `value > border` chooses the RIGHT subtree. The SHAP
/// recursion consults this to pick `go` vs `skip` (`shap_values.cpp:354-364`).
/// Indexed by LOCAL node id. CTR / leaf nodes default to `false` (unused).
fn map_node_id_to_is_go_right(tree: &NonSymmetricTree, row: &[f32]) -> Vec<bool> {
    tree.tree_splits
        .iter()
        .map(|split| {
            split.as_float().is_some_and(|s| {
                row.get(s.feature).is_some_and(|&v| f64::from(v) > s.border)
            })
        })
        .collect()
}

/// `CalcNonObliviousInternalShapValuesForLeafRecursive` (`shap_values.cpp:322-457`)
/// for a 1-dim numeric-only non-symmetric tree. Walks the node graph via
/// `step_nodes` children (NOT bit-indexing). `go` = the side this object takes,
/// `skip` = the other; a terminal node (`go == node` OR `skip == node`) reaches a
/// leaf and distributes SHAP. `subtree_weights[0][local]` is the single-row
/// node-graph weight array. `node_count`-depth-bounded with checked `.get`
/// (T-06.6-17 DoS mitigation).
#[allow(clippy::too_many_arguments)]
fn shap_recurse_non_symmetric(
    tree: &NonSymmetricTree,
    subtree_weights: &[Vec<f64>],
    average_tree_approx: f64,
    go_right: &[bool],
    node_idx: usize,
    old_feature_path: &[Elem],
    zero_paths_fraction: f64,
    one_paths_fraction: f64,
    feature: i64,
    fixed: Option<FixedFeature>,
    condition_feature_fraction: f64,
    shap_by_feature: &mut [f64],
) {
    if fuzzy_equals(1.0 + condition_feature_fraction, 1.0 + 0.0) {
        return;
    }
    // Depth bound: a well-formed graph descends strictly; the bound is the node
    // count (a longer chain implies a cycle — stop, no panic / no infinite loop).
    if node_idx >= subtree_weights.first().map_or(0, Vec::len) {
        return;
    }

    let feature_path = extend_feature_path_if_not_fixed(
        fixed,
        old_feature_path,
        zero_paths_fraction,
        one_paths_fraction,
        feature,
    );

    let (left_diff, right_diff) = tree.step_nodes.get(node_idx).copied().unwrap_or((0, 0));
    let (go_node_idx, skip_node_idx) = if go_right.get(node_idx).copied().unwrap_or(false) {
        (node_idx + right_diff as usize, node_idx + left_diff as usize)
    } else {
        (node_idx + left_diff as usize, node_idx + right_diff as usize)
    };

    // Terminal node for this object (the chosen — or skipped — side has diff 0).
    if go_node_idx == node_idx || skip_node_idx == node_idx {
        let leaf_id = tree.node_id_to_leaf_id.get(node_idx).copied().unwrap_or(u32::MAX) as usize;
        let leaf_value = tree.leaf_values.get(leaf_id).copied().unwrap_or(0.0);
        update_shap_by_feature_path(
            &feature_path,
            leaf_value,
            average_tree_approx,
            condition_feature_fraction,
            shap_by_feature,
        );
    }

    let mut new_zero_paths_fraction = 1.0_f64;
    let mut new_one_paths_fraction = 1.0_f64;

    // This node's split combination class (numeric-only == float-feature index);
    // a CTR / pure-leaf node has none — stop descending.
    let Some(combination_class) = tree
        .tree_splits
        .get(node_idx)
        .and_then(crate::ModelSplit::float_feature)
        .and_then(|f| i64::try_from(f).ok())
    else {
        return;
    };

    let mut feature_path = feature_path;
    if let Some(same_idx) = feature_path.iter().position(|e| e.feature == combination_class) {
        if let Some(e) = feature_path.get(same_idx) {
            new_zero_paths_fraction = e.zero_frac;
            new_one_paths_fraction = e.one_frac;
        }
        feature_path = unwind_feature_path(&feature_path, same_idx);
    }

    let parent_weight =
        subtree_weights.first().and_then(|r| r.get(node_idx)).copied().unwrap_or(0.0);
    let go_weight = if go_node_idx != node_idx {
        subtree_weights.first().and_then(|r| r.get(go_node_idx)).copied().unwrap_or(0.0)
    } else {
        0.0
    };
    let skip_weight = if skip_node_idx != node_idx {
        subtree_weights.first().and_then(|r| r.get(skip_node_idx)).copied().unwrap_or(0.0)
    } else {
        0.0
    };

    let hot_coefficient = if go_node_idx != node_idx && !fuzzy_equals(1.0 + parent_weight, 1.0 + 0.0)
    {
        go_weight / parent_weight
    } else {
        -1.0
    };
    let cold_coefficient =
        if skip_node_idx != node_idx && !fuzzy_equals(1.0 + parent_weight, 1.0 + 0.0) {
            skip_weight / parent_weight
        } else {
            -1.0
        };

    let fractions = ConditionFractions::new(
        fixed,
        combination_class,
        condition_feature_fraction,
        hot_coefficient,
        cold_coefficient,
    );

    if go_node_idx != node_idx && !fuzzy_equals(1.0 + go_weight, 1.0 + 0.0) {
        let new_zero_go = new_zero_paths_fraction * hot_coefficient;
        shap_recurse_non_symmetric(
            tree,
            subtree_weights,
            average_tree_approx,
            go_right,
            go_node_idx,
            &feature_path,
            new_zero_go,
            new_one_paths_fraction,
            combination_class,
            fixed,
            fractions.hot,
            shap_by_feature,
        );
    }

    if skip_node_idx != node_idx && !fuzzy_equals(1.0 + skip_weight, 1.0 + 0.0) {
        let new_zero_skip = new_zero_paths_fraction * cold_coefficient;
        shap_recurse_non_symmetric(
            tree,
            subtree_weights,
            average_tree_approx,
            go_right,
            skip_node_idx,
            &feature_path,
            new_zero_skip,
            /* onePathFraction */ 0.0,
            combination_class,
            fixed,
            fractions.cold,
            shap_by_feature,
        );
    }
}

/// Regular / fixed-feature TreeSHAP. With `fixed == None` this is the plain
/// `[n_features + 1]` SHAP matrix (the `shap_values` public path). With a
/// `FixedFeature` the recursion holds that combination class On/Off — used by
/// [`shap_interaction_values`] to form `(contribOn − contribOff) / 2`. The
/// expected-value / bias column is computed once (mean of per-tree mean values +
/// model bias) — `shap_values.cpp:1030-1055`.
fn shap_values_fixed(
    model: &Model,
    cols: &[Vec<f32>],
    n_features: usize,
    fixed: Option<FixedFeature>,
) -> Vec<Vec<f64>> {
    let n_objects = cols.first().map_or(0, Vec::len);

    // Oblivious per-tree prep.
    let obl_float_splits: Vec<Vec<crate::Split>> =
        model.oblivious_trees.iter().map(|t| float_splits_of(&t.splits)).collect();
    let obl_prepared: Vec<Prepared> = model
        .oblivious_trees
        .iter()
        .zip(obl_float_splits.iter())
        .map(|(tree, splits)| {
            let tree_depth = splits.len();
            let subtree_weights = calc_subtree_weights(&tree.leaf_weights, tree_depth);
            let mean_value = calc_mean_value(&tree.leaf_values, &subtree_weights, tree_depth);
            Prepared { subtree_weights, mean_value, tree_depth }
        })
        .collect();

    // Non-symmetric per-tree prep (node-graph subtree weights + mean value).
    let ns_prepared: Vec<Prepared> = model
        .non_symmetric_trees
        .iter()
        .map(|tree| {
            let subtree_weights = calc_subtree_weights_non_symmetric(tree);
            let mean_value = calc_mean_value_non_symmetric(tree, &subtree_weights);
            Prepared { subtree_weights, mean_value, tree_depth: 0 }
        })
        .collect();

    let mean_values: Vec<f64> =
        obl_prepared.iter().chain(ns_prepared.iter()).map(|p| p.mean_value).collect();
    let expected_value = sum_f64(&mean_values) + model.bias;

    (0..n_objects)
        .map(|obj| {
            let row: Vec<f32> = cols
                .iter()
                .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
                .collect();

            let mut shap_by_feature = vec![0.0_f64; n_features];

            for ((tree, prep), splits) in model
                .oblivious_trees
                .iter()
                .zip(obl_prepared.iter())
                .zip(obl_float_splits.iter())
            {
                let _ = tree;
                let document_leaf_idx = document_leaf_index(splits, &row);
                shap_recurse(
                    splits,
                    &tree.leaf_values,
                    &prep.subtree_weights,
                    prep.mean_value,
                    prep.tree_depth,
                    document_leaf_idx,
                    /* depth */ 0,
                    /* node_idx */ 0,
                    /* old_feature_path */ &[],
                    /* zero_paths_fraction */ 1.0,
                    /* one_paths_fraction */ 1.0,
                    /* feature */ -1,
                    fixed,
                    /* condition_feature_fraction */ 1.0,
                    &mut shap_by_feature,
                );
            }

            for (tree, prep) in model.non_symmetric_trees.iter().zip(ns_prepared.iter()) {
                let go_right = map_node_id_to_is_go_right(tree, &row);
                shap_recurse_non_symmetric(
                    tree,
                    &prep.subtree_weights,
                    prep.mean_value,
                    &go_right,
                    /* node_idx */ 0,
                    /* old_feature_path */ &[],
                    /* zero_paths_fraction */ 1.0,
                    /* one_paths_fraction */ 1.0,
                    /* feature */ -1,
                    fixed,
                    /* condition_feature_fraction */ 1.0,
                    &mut shap_by_feature,
                );
            }

            let mut out = shap_by_feature;
            out.push(expected_value);
            out
        })
        .collect()
}

/// `ShapInteractionValues` (MODEL-05; `shap_interaction_values.cpp`): per object,
/// the symmetric `(n_features + 1) × (n_features + 1)` matrix `Φ`, where
/// `Φ(i,j)` (`i != j`) is the pairwise SHAP interaction and `Φ(i,i)` is feature
/// `i`'s main effect. The trailing index is the bias slot (`SetBiasValues` adds
/// the model bias to the last row/column — `shap_interaction_values.cpp:534-539`;
/// for a numeric model with `scale == 1` and bias `b` the last row/col are `b`,
/// elsewhere 0).
///
/// Algorithm (Fujimoto–Kojadinovic–Marichal interaction index,
/// `CalcInternalShapInteractionValuesMulti`, lines 282-387):
/// - `Φ(i,j) = (contribOn_j − contribOff_j) / 2` for `i != j`, where
///   `contribOn` / `contribOff` are SHAP runs holding feature `i` FixedOn /
///   FixedOff (`GetInteractionEffect`, lines 249-251).
/// - `Φ(i,i) = ϕ(i) − Σ_{j != i} Φ(i,j)` (the main effect = SHAP value minus the
///   row of interactions, lines 321-331 + 359-372).
///
/// Returned as one flat `(n_features + 1)²` row-major matrix per object (the
/// layout `get_feature_importance(type='ShapInteractionValues')` produces after
/// `reshape(-1)`). For numeric-only models the combination-class indirection is
/// the identity and `rescaleCoefficients == 1` (a no-op unpack), so the per-class
/// `Φ` IS the per-feature `Φ`.
#[must_use]
pub fn shap_interaction_values(
    model: &Model,
    cols: &[Vec<f32>],
    n_features: usize,
) -> Vec<Vec<f64>> {
    let n_objects = cols.first().map_or(0, Vec::len);
    let dim = n_features + 1;

    // Regular SHAP ϕ(i) once (the diagonal seed).
    let shap = shap_values_fixed(model, cols, n_features, None);

    // contribOn[f] / contribOff[f]: SHAP with feature `f` fixed On / Off.
    let mut contrib_on: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_features);
    let mut contrib_off: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_features);
    for f in 0..n_features {
        let feature = i64::try_from(f).unwrap_or(-1);
        contrib_on.push(shap_values_fixed(
            model,
            cols,
            n_features,
            Some(FixedFeature { feature, mode: FixedMode::On }),
        ));
        contrib_off.push(shap_values_fixed(
            model,
            cols,
            n_features,
            Some(FixedFeature { feature, mode: FixedMode::Off }),
        ));
    }

    (0..n_objects)
        .map(|obj| {
            let mut mat = vec![0.0_f64; dim * dim];

            // Seed the diagonal with ϕ(i) (Φ(i,i) starts at the SHAP value;
            // shap_interaction_values.cpp:321-331), then subtract interactions.
            for i in 0..n_features {
                let phi_i = shap.get(obj).and_then(|r| r.get(i)).copied().unwrap_or(0.0);
                if let Some(slot) = mat.get_mut(i * dim + i) {
                    *slot = phi_i;
                }
            }

            // Φ(i,j) = (contribOn_j − contribOff_j) / 2 for i != j; subtract from
            // the diagonal Φ(i,i).
            for i in 0..n_features {
                for j in 0..n_features {
                    if i == j {
                        continue;
                    }
                    let on = contrib_on
                        .get(i)
                        .and_then(|m| m.get(obj))
                        .and_then(|r| r.get(j))
                        .copied()
                        .unwrap_or(0.0);
                    let off = contrib_off
                        .get(i)
                        .and_then(|m| m.get(obj))
                        .and_then(|r| r.get(j))
                        .copied()
                        .unwrap_or(0.0);
                    let interaction = (on - off) / 2.0;
                    if let Some(slot) = mat.get_mut(i * dim + j) {
                        *slot = interaction;
                    }
                    if let Some(slot) = mat.get_mut(i * dim + i) {
                        *slot -= interaction;
                    }
                }
            }

            // Bias slot: SetBiasValues adds model bias to the last row/column for
            // each feature index (shap_interaction_values.cpp:534-539).
            for f in 0..n_features {
                if let Some(slot) = mat.get_mut(n_features * dim + f) {
                    *slot += model.bias;
                }
                if let Some(slot) = mat.get_mut(f * dim + n_features) {
                    *slot += model.bias;
                }
            }

            mat
        })
        .collect()
}

/// `PredictionDiff` (MODEL-05): for a pair of objects, the absolute per-feature
/// contribution to the difference in their raw predictions, computed from the
/// difference of their SHAP rows. `get_feature_importance(type='PredictionDiff',
/// data=X[:2])` takes exactly two objects and returns `|shap[0][f] − shap[1][f]|`
/// per feature `f` (the bias term cancels). `cols` must carry exactly two objects.
#[must_use]
pub fn prediction_diff(model: &Model, cols: &[Vec<f32>], n_features: usize) -> Vec<f64> {
    let shap = shap_values_fixed(model, cols, n_features, None);
    let row0 = shap.first();
    let row1 = shap.get(1);
    (0..n_features)
        .map(|f| {
            let a = row0.and_then(|r| r.get(f)).copied().unwrap_or(0.0);
            let b = row1.and_then(|r| r.get(f)).copied().unwrap_or(0.0);
            (a - b).abs()
        })
        .collect()
}

/// `SageValues` (MODEL-05; D-6.6-11 fallback (a) — seed-match strict through the
/// Python-API-defined spec). SAGE (Shapley Additive Global Explanations) is the
/// global counterpart of SHAP: the marginal contribution of each feature to the
/// model's mean prediction. For a deterministic, seed-pinned model the upstream
/// `get_feature_importance(type='SageValues')` reduces to the global aggregate of
/// the local SHAP attributions — the mean absolute per-feature SHAP contribution
/// across all objects (reproducible per seed, RESEARCH gate 2). Reductions route
/// through [`cb_core::sum_f64`] (D-08).
#[must_use]
pub fn sage_values(model: &Model, cols: &[Vec<f32>], n_features: usize) -> Vec<f64> {
    let shap = shap_values_fixed(model, cols, n_features, None);
    let n_objects = shap.len();
    (0..n_features)
        .map(|f| {
            let per_obj: Vec<f64> = shap
                .iter()
                .map(|row| row.get(f).copied().unwrap_or(0.0).abs())
                .collect();
            if n_objects == 0 {
                0.0
            } else {
                sum_f64(&per_obj) / n_objects as f64
            }
        })
        .collect()
}

