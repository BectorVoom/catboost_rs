//! Leaf-value estimation primitives — `CalcAverage`, `ScaleL2Reg`, and the
//! Gradient-method leaf delta (TRAIN-03 Gradient). These are the L2-regularized
//! averages over a leaf's member derivatives. The SUM over leaf members is done
//! by the caller through `cb_core::sum_f64` (D-02/D-05); these helpers consume an
//! already-reduced `sum_delta` / `sum_weight`.
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo_helpers/online_predictor.h:112-178`:
//! - `CalcAverage(sumDelta, count, scaledL2) = count > 0 ? sumDelta/(count +
//!   scaledL2) : 0.0` — the guarded average; an empty leaf returns `0.0` rather
//!   than dividing by zero (T-03-01-01 mitigation).
//! - `ScaleL2Reg(l2, sumAllWeights, allDocCount) = l2 * (sumAllWeights /
//!   allDocCount)` — the per-tree L2 scaling applied to every leaf's denominator.
//! - Gradient leaf delta = `CalcAverage(SumDer, SumWeights, scaledL2)`. For the
//!   unweighted path every object weight is `1.0`, so `SumWeights` is the leaf's
//!   object count and `sumAllWeights/allDocCount == 1`, giving `scaledL2 == l2`.
//!
//! # f64 discipline & summation routing (D-07 / D-08)
//!
//! All arguments and results are `f64`. This module performs only scalar
//! arithmetic on already-reduced sums; it never spells a float fold, so the
//! D-08 raw-sum ban does not apply here — the reduction lives in the caller via
//! `cb_core::sum_f64`.

/// L2-regularized guarded average: `count > 0 ? sum_delta/(count + scaled_l2) :
/// 0.0`.
///
/// `online_predictor.h` `CalcAverage`. The `count > 0` guard means a degenerate
/// (empty) leaf returns `0.0` — no division by zero, no panic (T-03-01-01).
/// `count` is the leaf's summed weight (object count in the unweighted path).
#[must_use]
pub fn calc_average(sum_delta: f64, count: f64, scaled_l2: f64) -> f64 {
    if count > 0.0 {
        sum_delta / (count + scaled_l2)
    } else {
        0.0
    }
}

/// Per-tree L2 scaling: `l2 * (sum_all_weights / doc_count)`.
///
/// `online_predictor.h` `ScaleL2Reg`. `doc_count` is the total object count;
/// `sum_all_weights` is the total weight. For the unweighted path
/// `sum_all_weights == doc_count`, so this returns `l2`. Returns `l2` directly
/// when `doc_count == 0` to avoid a division by zero on a degenerate dataset
/// (the trainer rejects empty datasets upstream of this primitive).
#[must_use]
pub fn scale_l2_reg(l2: f64, sum_all_weights: f64, doc_count: usize) -> f64 {
    if doc_count == 0 {
        l2
    } else {
        l2 * (sum_all_weights / doc_count as f64)
    }
}

/// The Gradient-method leaf delta: `CalcAverage(sum_der, sum_weight, scaled_l2)`.
///
/// `sum_der` is the leaf's reduced first-derivative sum, `sum_weight` its summed
/// weight (object count unweighted), `scaled_l2` the [`scale_l2_reg`] output.
/// This is the unscaled-by-learning-rate delta; the boosting loop multiplies the
/// stored leaf value by `learning_rate`.
#[must_use]
pub fn gradient_leaf_delta(sum_der: f64, sum_weight: f64, scaled_l2: f64) -> f64 {
    calc_average(sum_der, sum_weight, scaled_l2)
}
