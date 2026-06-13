//! Unit tests for the per-loss derivatives (TRAIN-01). RMSE `t-a`/`-1`; Logloss
//! `t-p`/`-p(1-p)` with `p = sigmoid(approx)` over the raw logit.

use crate::loss::{
    logloss_der1, logloss_der2, mae_der1, mae_der2, rmse_der1, rmse_der2, sigmoid,
};

#[test]
fn rmse_der1_is_target_minus_approx() {
    assert!((rmse_der1(0.5, 2.0) - 1.5).abs() < 1e-12);
    assert!((rmse_der1(3.0, 1.0) - (-2.0)).abs() < 1e-12);
    assert!((rmse_der1(0.0, 0.0)).abs() < 1e-12);
}

#[test]
fn rmse_der2_is_constant_negative_one() {
    assert!((rmse_der2(0.5, 2.0) - (-1.0)).abs() < 1e-12);
    assert!((rmse_der2(-100.0, 100.0) - (-1.0)).abs() < 1e-12);
}

#[test]
fn sigmoid_at_zero_is_half() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
}

#[test]
fn sigmoid_is_symmetric() {
    // sigmoid(-x) == 1 - sigmoid(x)
    let x = 1.7_f64;
    assert!((sigmoid(-x) - (1.0 - sigmoid(x))).abs() < 1e-12);
}

#[test]
fn logloss_der1_is_target_minus_prob() {
    // approx=0 -> p=0.5; target=1 -> der1 = 0.5; target=0 -> der1 = -0.5
    assert!((logloss_der1(0.0, 1.0) - 0.5).abs() < 1e-12);
    assert!((logloss_der1(0.0, 0.0) - (-0.5)).abs() < 1e-12);
    // raw-logit approx: p = sigmoid(2.0)
    let p = sigmoid(2.0);
    assert!((logloss_der1(2.0, 1.0) - (1.0 - p)).abs() < 1e-12);
}

#[test]
fn logloss_der2_is_neg_p_times_one_minus_p() {
    // approx=0 -> p=0.5 -> der2 = -0.25
    assert!((logloss_der2(0.0, 1.0) - (-0.25)).abs() < 1e-12);
    let p = sigmoid(1.3);
    assert!((logloss_der2(1.3, 0.0) - (-p * (1.0 - p))).abs() < 1e-12);
}

#[test]
fn mae_der1_is_signed_half_quantile() {
    // residual > delta -> +alpha (0.5); residual < -delta -> -(1-alpha) (-0.5).
    assert!((mae_der1(0.0, 2.0) - 0.5).abs() < 1e-12); // target above approx
    assert!((mae_der1(2.0, 0.0) - (-0.5)).abs() < 1e-12); // target below approx
}

#[test]
fn mae_der1_deadzone_returns_zero() {
    // |target - approx| < delta (1e-6) -> 0 (the deadzone).
    assert_eq!(mae_der1(1.0, 1.0), 0.0);
    assert_eq!(mae_der1(1.0, 1.0 + 1e-9), 0.0);
}

#[test]
fn mae_der2_is_zero() {
    assert_eq!(mae_der2(0.5, 2.0), 0.0);
    assert_eq!(mae_der2(-3.0, 7.0), 0.0);
}
