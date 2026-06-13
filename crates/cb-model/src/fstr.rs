//! Feature-importance for the canonical oblivious [`crate::Model`] (MODEL-03,
//! partial — the loss-change importance is deferred per D-12):
//!
//! - [`prediction_values_change`] — the `PredictionValuesChange` importance
//!   (percent, Σ = 100), and
//! - [`interaction`] — the pairwise `Interaction` importance
//!   (`Vec<(feature_i, feature_j, score)>`, percent-normalized).
//!
//! Both consume the per-leaf training-document weights (`leaf_weights`) captured
//! in Plan 01 (RESEARCH Pitfall 1 — without them every importance silently
//! short-circuits to zero).
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
//! The loss-change importance is intentionally NOT implemented (D-12, out of
//! scope this phase).
//!
//! # Parity discipline
//!
//! Every float fold routes through [`cb_core::sum_f64`] (D-08). All leaf / weight
//! access is checked `.get` (`indexing_slicing` deny); no `unwrap`/`expect`. The
//! `count1 == 0 || count2 == 0` short-circuit (PVC) and the `total == 0` guard
//! (normalization) reproduce the upstream div-by-zero protections exactly
//! (T-04-04-03), so no NaN leaks.

use cb_core::sum_f64;

use crate::Model;

/// The kind of feature importance to compute. The loss-change variant is
/// deliberately ABSENT (D-12 — out of scope this phase).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureImportanceType {
    /// Per-feature `PredictionValuesChange` (the default upstream importance):
    /// how much, on average, predictions change when a feature's split flips,
    /// normalized to percentages summing to 100.
    PredictionValuesChange,
    /// Pairwise `Interaction` importance between feature pairs (percent).
    Interaction,
}

/// The number of (flat = float, numeric-only) features the model splits on:
/// `max split feature index + 1`. Matches the width of the upstream importance
/// vector for a numeric-only pool (unused features keep a `0` importance).
fn feature_count(model: &Model) -> usize {
    model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .map(|s| s.feature + 1)
        .max()
        .unwrap_or(0)
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

/// `PredictionValuesChange` feature importance (MODEL-03), transcribing
/// `CalcEffect` (`feature_str.h:233-270`). Returns one percentage per feature
/// (length = [`feature_count`]); the percentages sum to 100.
#[must_use]
pub fn prediction_values_change(model: &Model) -> Vec<f64> {
    let n_features = feature_count(model);
    let mut res = vec![0.0_f64; n_features];

    for tree in &model.oblivious_trees {
        let leaf_count = tree.leaf_values.len();
        // for (feature = 0; feature < tree.SrcFeatures.size(); ++feature)
        for (feature_bit, split) in tree.splits.iter().enumerate() {
            let src_idx = split.feature;
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

    convert_to_percents(&mut res);
    res
}

/// `Interaction` feature importance (MODEL-03), transcribing
/// `CalcMostInteractingFeatures` (`feature_str.cpp:190-223`) +
/// `CalcFeatureInteraction` (`calc_fstr.cpp:343-414`) for a numeric-only
/// oblivious model.
///
/// Returns `(feature_i, feature_j, score)` triples with `i < j`, `score` the
/// percent-of-total pairwise contribution, sorted by `score` descending (the
/// upstream `EXISTING_PAIRS_COUNT` behavior — only pairs that actually appear).
/// **NOT** SHAP interaction values (that is Phase 6).
#[must_use]
pub fn interaction(model: &Model) -> Vec<(usize, usize, f64)> {
    // Accumulate Σ|delta| per sorted source-feature pair, preserving insertion
    // order for a deterministic, upstream-stable iteration (a Vec keyed by pair
    // rather than a hash map, so ordering is reproducible).
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut sums: Vec<f64> = Vec::new();

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

                let src1 = tree.splits.get(first_idx).map_or(0, |s| s.feature);
                let src2 = tree.splits.get(second_idx).map_or(0, |s| s.feature);
                if src1 == src2 {
                    continue;
                }
                let (a, b) = if src1 < src2 { (src1, src2) } else { (src2, src1) };

                if let Some(pos) = pairs.iter().position(|&p| p == (a, b)) {
                    if let Some(slot) = sums.get_mut(pos) {
                        *slot += delta.abs();
                    }
                } else {
                    pairs.push((a, b));
                    sums.push(delta.abs());
                }
            }
        }
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
