//! Unit tests for the eval-set validation metrics (TRAIN-07, [`crate::metrics`]).
//!
//! Locks the RMSE / Logloss `eval_metric` values on HAND-COMPUTED pred/target/
//! weight sets (independent of any oracle fixture), the `eval_metric`-defaults-
//! to-objective rule, the degenerate-input guards (no panic / no div-by-zero,
//! T-03-06-01), and the per-eval-set history bookkeeping.
//!
//! Dedicated test file (CLAUDE.md source/test separation — no inline
//! `#[cfg(test)]`). Test-only lint relaxations match the crate policy.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::Loss;

use crate::metrics::{EvalMetric, EvalMetricHistory};

/// `eval_metric` defaults to the objective when unset.
#[test]
fn eval_metric_defaults_to_objective() {
    assert_eq!(EvalMetric::for_loss(Loss::Rmse), EvalMetric::Rmse);
    assert_eq!(EvalMetric::for_loss(Loss::Mae), EvalMetric::Rmse);
    assert_eq!(EvalMetric::for_loss(Loss::Logloss), EvalMetric::Logloss);
}

/// RMSE over a hand-computed unweighted set:
/// d = [-0.5, 1.0, -1.0], sq = [0.25, 1.0, 1.0], mean = 0.75, sqrt = 0.8660254…
#[test]
fn rmse_unweighted_hand_computed() {
    let approx = [1.0, 2.0, 3.0];
    let target = [1.5, 1.0, 4.0];
    let got = EvalMetric::Rmse.eval(&approx, &target, &[]).unwrap();
    assert!(
        (got - 0.866_025_403_784_438_6).abs() < 1e-12,
        "rmse unweighted = {got}"
    );
}

/// Weighted RMSE: weights [1,2,1] -> weighted_sq sum 3.25 / total weight 4.0 =
/// 0.8125, sqrt = 0.9013878…
#[test]
fn rmse_weighted_hand_computed() {
    let approx = [1.0, 2.0, 3.0];
    let target = [1.5, 1.0, 4.0];
    let weights = [1.0, 2.0, 1.0];
    let got = EvalMetric::Rmse.eval(&approx, &target, &weights).unwrap();
    assert!(
        (got - 0.901_387_818_865_997_3).abs() < 1e-12,
        "rmse weighted = {got}"
    );
}

/// Logloss over a hand-computed set: approx [0, 2], target [1, 0].
/// p0 = sigmoid(0) = 0.5 -> -ln 0.5 = 0.69314718…
/// p1 = sigmoid(2) = 0.88079707… -> -ln(1-p1) = 2.12692801…
/// mean = 1.41003759…
#[test]
fn logloss_hand_computed() {
    let approx = [0.0, 2.0];
    let target = [1.0, 0.0];
    let got = EvalMetric::Logloss.eval(&approx, &target, &[]).unwrap();
    assert!(
        (got - 1.410_037_595_801_458_8).abs() < 1e-12,
        "logloss = {got}"
    );
}

/// An empty eval set is degenerate (no div-by-zero / panic, T-03-06-01).
#[test]
fn empty_eval_set_is_degenerate() {
    assert!(EvalMetric::Rmse.eval(&[], &[], &[]).is_err());
    assert!(EvalMetric::Logloss.eval(&[], &[], &[]).is_err());
}

/// A zero total weight is degenerate (the division guard fires).
#[test]
fn zero_total_weight_is_degenerate() {
    let approx = [1.0, 2.0];
    let target = [1.0, 2.0];
    let weights = [0.0, 0.0];
    assert!(EvalMetric::Rmse.eval(&approx, &target, &weights).is_err());
}

/// A length mismatch (approx vs target, or weights) is degenerate.
#[test]
fn length_mismatch_is_degenerate() {
    assert!(EvalMetric::Rmse.eval(&[1.0, 2.0], &[1.0], &[]).is_err());
    assert!(EvalMetric::Rmse
        .eval(&[1.0, 2.0], &[1.0, 2.0], &[1.0])
        .is_err());
}

/// The per-eval-set history records per-iteration values per set and exposes the
/// primary (index 0) curve to the detector.
#[test]
fn history_tracks_per_set_and_primary() {
    let mut h = EvalMetricHistory::new(2);
    assert_eq!(h.len(), 2);
    assert!(!h.is_empty());
    h.push(0, 1.0);
    h.push(1, 10.0);
    h.push(0, 0.9);
    h.push(1, 9.0);
    assert_eq!(h.per_set[0], vec![1.0, 0.9]);
    assert_eq!(h.per_set[1], vec![10.0, 9.0]);
    assert_eq!(h.primary(), &[1.0, 0.9]);
    // Out-of-range push is ignored (defensive).
    h.push(5, 99.0);
    assert_eq!(h.len(), 2);
}

/// An empty history (no eval sets) yields an empty primary curve, never panics.
#[test]
fn empty_history_primary_is_empty() {
    let h = EvalMetricHistory::new(0);
    assert!(h.is_empty());
    assert!(h.primary().is_empty());
}
