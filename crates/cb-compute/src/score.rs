//! Split-score (gain) computation — the L2 score calcer's `AddLeafPlain` fold
//! over a candidate split's leaf statistics (TRAIN-02, `score_function=L2`).
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/score_calcers.cpp:20-49` + `online_predictor.h`:
//! the L2 score calcer folds each leaf's contribution as
//! `avg * sumWeightedDelta` where `avg = CalcAverage(sumWeightedDelta, sumWeight,
//! L2Regularizer)`. For a candidate split the score is the sum of `AddLeafPlain`
//! over ALL leaves the split produces (for an oblivious tree at depth `d` that is
//! every leaf at the current level, with the candidate applied across the level).
//! A higher score is a better split.
//!
//! `MINIMAL_SCORE = std::numeric_limits<double>::lowest()` is the sentinel a
//! candidate must beat; here it is [`f64::NEG_INFINITY`] (any finite score
//! exceeds it, matching upstream's `bestScore == MINIMAL_SCORE -> stop`).
//!
//! # f64 discipline & summation routing (D-07 / D-08)
//!
//! Leaf stats arrive already reduced (via `cb_core::sum_f64` in `histogram.rs`).
//! The per-leaf `avg * sumWeightedDelta` terms are accumulated through
//! [`cb_core::sum_f64`] so even the score fold honors the single-primitive rule;
//! no raw iterator-sum or zero-seeded float fold is spelled here (D-08).

use cb_core::sum_f64;

use crate::histogram::LeafStats;
use crate::leaf::calc_average;

/// The minimal score sentinel (`std::numeric_limits<double>::lowest()` analogue):
/// any finite candidate score must strictly exceed it to be selected.
pub const MINIMAL_SCORE: f64 = f64::NEG_INFINITY;

/// One leaf's L2 `AddLeafPlain` contribution: `avg * sum_weighted_delta` with
/// `avg = CalcAverage(sum_weighted_delta, sum_weight, scaled_l2)`.
///
/// `score_calcers.cpp` — the per-leaf term the L2 score calcer accumulates.
#[must_use]
pub fn add_leaf_plain(stats: LeafStats, scaled_l2: f64) -> f64 {
    let avg = calc_average(stats.sum_weighted_delta, stats.sum_weight, scaled_l2);
    avg * stats.sum_weighted_delta
}

/// The total L2 score for a candidate split: the sum of [`add_leaf_plain`] over
/// every leaf the split produces, accumulated in the given leaf order through the
/// sanctioned reduction primitive (D-08). `leaves` MUST be supplied in the
/// canonical leaf-index order so the fold order is reproducible.
#[must_use]
pub fn l2_split_score(leaves: &[LeafStats], scaled_l2: f64) -> f64 {
    let terms: Vec<f64> = leaves
        .iter()
        .map(|&stats| add_leaf_plain(stats, scaled_l2))
        .collect();
    sum_f64(&terms)
}
