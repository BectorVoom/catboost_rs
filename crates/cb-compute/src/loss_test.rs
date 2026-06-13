//! Unit tests for the per-loss derivatives (TRAIN-01 / D-09). RMSE `t-a`/`-1`;
//! Logloss / CrossEntropy `t-p`/`-p(1-p)` with `p = sigmoid(approx)` over the raw
//! logit; Focal `alpha`/`gamma`-weighted der1/der2 (`error_functions.h`).

use crate::loss::{
    cross_entropy_der1, cross_entropy_der2, focal_der1, focal_der2, logloss_der1, logloss_der2,
    mae_der1, mae_der2, rmse_der1, rmse_der2, sigmoid,
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
fn cross_entropy_matches_logloss_math() {
    // CrossEntropy der1/der2 are IDENTICAL to Logloss (D-09); a SOFT target in
    // [0,1] is the only CrossEntropy-specific input.
    assert!((cross_entropy_der1(0.0, 0.7) - logloss_der1(0.0, 0.7)).abs() < 1e-15);
    assert!((cross_entropy_der2(0.0, 0.7) - logloss_der2(0.0, 0.7)).abs() < 1e-15);
    let p = sigmoid(1.4);
    assert!((cross_entropy_der1(1.4, 0.3) - (0.3 - p)).abs() < 1e-12);
    assert!((cross_entropy_der2(1.4, 0.3) - (-p * (1.0 - p))).abs() < 1e-12);
}

#[test]
fn focal_der1_matches_reference_positive_class() {
    // error_functions.h:1684-1709 transcription at (approx=0.5, target=1).
    let (alpha, gamma, approx) = (0.25_f64, 2.0_f64, 0.5_f64);
    let p = (1.0 / (1.0 + (-approx).exp())).clamp(1e-13, 1.0 - 1e-13);
    let (at, pt, y) = (alpha, p, 1.0_f64);
    let want = -(at * y * (1.0 - pt).powf(gamma) * (gamma * pt * pt.ln() + pt - 1.0));
    assert!((focal_der1(approx, 1.0, alpha, gamma) - want).abs() < 1e-12);
}

#[test]
fn focal_der2_matches_reference_positive_class() {
    let (alpha, gamma, approx) = (0.25_f64, 2.0_f64, 0.5_f64);
    let p = (1.0 / (1.0 + (-approx).exp())).clamp(1e-13, 1.0 - 1e-13);
    let (at, pt, y) = (alpha, p, 1.0_f64);
    let u = at * y * (1.0 - pt).powf(gamma);
    let du = -at * y * gamma * (1.0 - pt).powf(gamma - 1.0);
    let v = gamma * pt * pt.ln() + pt - 1.0;
    let dv = gamma * pt.ln() + gamma + 1.0;
    let want = -((du * v + u * dv) * y * (pt * (1.0 - pt)));
    assert!((focal_der2(approx, 1.0, alpha, gamma) - want).abs() < 1e-12);
}

#[test]
fn focal_clamps_saturated_logit_no_nan() {
    // A large positive logit with the negative class drives pt -> 0; the clamp
    // keeps ln(pt)/powf finite (T-04-02-02 — no NaN).
    let g1 = focal_der1(40.0, 0.0, 0.25, 2.0);
    let g2 = focal_der2(40.0, 0.0, 0.25, 2.0);
    assert!(g1.is_finite(), "focal der1 must stay finite under saturation");
    assert!(g2.is_finite(), "focal der2 must stay finite under saturation");
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
