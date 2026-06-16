//! Unit tests for the per-loss derivatives (TRAIN-01 / D-09). RMSE `t-a`/`-1`;
//! Logloss / CrossEntropy `t-p`/`-p(1-p)` with `p = sigmoid(approx)` over the raw
//! logit; Focal `alpha`/`gamma`-weighted der1/der2 (`error_functions.h`).

use crate::loss::{
    calc_softmax, cross_entropy_der1, cross_entropy_der2, expectile_der1, expectile_der2,
    focal_der1, focal_der2, huber_der1, huber_der2, logcosh_der1, logcosh_der2, logloss_der1,
    logloss_der2, lq_der1, lq_der2, mae_der1, mae_der2, mape_der1, mape_der2,
    multi_crossentropy_ders, multiclass_onevsall_ders, poisson_der1, poisson_der2, quantile_der1,
    quantile_der2, rmse_der1, rmse_der2, sigmoid, softmax_ders, tweedie_der1, tweedie_der2,
    QUANTILE_ALPHA, QUANTILE_DELTA,
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

#[test]
fn quantile_der1_at_alpha07_is_asymmetric() {
    // alpha=0.7: residual > delta -> +alpha (0.7); residual < -delta ->
    // -(1-alpha) (-0.3); |residual| < delta -> 0 (the deadzone).
    let alpha = 0.7;
    let delta = 1e-6;
    // target above approx (val = target - approx > 0) -> +0.7.
    assert!((quantile_der1(0.0, 2.0, alpha, delta) - 0.7).abs() < 1e-12);
    // target below approx (val < 0) -> -(1 - 0.7) = -0.3.
    assert!((quantile_der1(2.0, 0.0, alpha, delta) - (-0.3)).abs() < 1e-12);
    // |val| < delta -> deadzone 0.
    assert_eq!(quantile_der1(1.0, 1.0, alpha, delta), 0.0);
    assert_eq!(quantile_der1(1.0, 1.0 + 1e-9, alpha, delta), 0.0);
}

#[test]
fn quantile_der1_at_alpha05_equals_mae() {
    // The MAE-equivalence guarantee: quantile_der1(a, t, 0.5, 1e-6) == mae_der1(a,
    // t) at sample points (above, below, deadzone). MAE byte-stability hinges on
    // this — re-expressing MAE through Quantile must not move the math.
    for &(a, t) in &[(0.0, 2.0), (2.0, 0.0), (1.0, 1.0), (3.0, -4.5), (-1.0, -1.0)] {
        assert_eq!(
            quantile_der1(a, t, QUANTILE_ALPHA, QUANTILE_DELTA),
            mae_der1(a, t),
            "quantile_der1({a}, {t}, 0.5, 1e-6) must equal mae_der1({a}, {t})"
        );
    }
}

#[test]
fn quantile_der2_is_zero() {
    // der2 == 0 for any alpha/delta (TQuantileError QUANTILE_DER2 = 0).
    assert_eq!(quantile_der2(0.5, 2.0, 0.7, 1e-6), 0.0);
    assert_eq!(quantile_der2(-3.0, 7.0, 0.5, 1e-6), 0.0);
    assert_eq!(quantile_der2(1.0, 1.0, 0.9, 1e-3), 0.0);
}

// ---- Wave-1 smooth losses (D-6.1-02) ----

#[test]
fn logcosh_der1_is_neg_tanh_of_residual() {
    // error_functions.h:414 — der1 = -tanh(approx - target).
    assert!((logcosh_der1(2.0, 0.5) - (-(1.5_f64).tanh())).abs() < 1e-12);
    assert!((logcosh_der1(0.5, 2.0) - (-(-1.5_f64).tanh())).abs() < 1e-12);
    // At zero residual the gradient vanishes (tanh(0) = 0).
    assert!(logcosh_der1(3.0, 3.0).abs() < 1e-12);
}

#[test]
fn logcosh_der2_is_neg_sech_squared() {
    // error_functions.h:418 — der2 = -1/cosh(approx - target)^2.
    let want = -1.0 / ((1.5_f64).cosh() * (1.5_f64).cosh());
    assert!((logcosh_der2(2.0, 0.5) - want).abs() < 1e-12);
    // At zero residual cosh(0)=1 -> der2 = -1 (max curvature).
    assert!((logcosh_der2(3.0, 3.0) - (-1.0)).abs() < 1e-12);
}

#[test]
fn lq_der1_signed_power_residual() {
    // error_functions.h:553 — der1 = q*sign(target-approx)*|approx-target|^(q-1).
    // q=2: der1 = 2*sign(t-a)*|a-t| ; (approx=0.5,target=2.0) -> 2*(+1)*1.5 = 3.0
    assert!((lq_der1(0.5, 2.0, 2.0) - 3.0).abs() < 1e-12);
    // target below approx -> negative gradient.
    assert!((lq_der1(2.0, 0.5, 2.0) - (-3.0)).abs() < 1e-12);
    // q=3, (approx=0.0,target=2.0): 3*(+1)*2^2 = 12.0
    assert!((lq_der1(0.0, 2.0, 3.0) - 12.0).abs() < 1e-12);
}

#[test]
fn lq_der2_neg_q_qm1_power() {
    // error_functions.h:558 — der2 = -q*(q-1)*|target-approx|^(q-2).
    // q=2: collapses to constant -2 (pow(.,0)=1).
    assert!((lq_der2(0.5, 2.0, 2.0) - (-2.0)).abs() < 1e-12);
    assert!((lq_der2(7.0, -3.0, 2.0) - (-2.0)).abs() < 1e-12);
    // q=3, |t-a|=2: -3*2*2^1 = -12.0
    assert!((lq_der2(0.0, 2.0, 3.0) - (-12.0)).abs() < 1e-12);
}

#[test]
fn huber_der1_band_and_saturation() {
    // error_functions.h:1612 — diff=target-approx; |diff|<delta ? diff : sign*delta.
    let delta = 1.0;
    // Inside band (|diff|=0.5<1): der1 = diff = 0.5.
    assert!((huber_der1(0.0, 0.5, delta) - 0.5).abs() < 1e-12);
    // Outside band, positive diff -> +delta.
    assert!((huber_der1(0.0, 3.0, delta) - delta).abs() < 1e-12);
    // Outside band, negative diff -> -delta.
    assert!((huber_der1(3.0, 0.0, delta) - (-delta)).abs() < 1e-12);
}

#[test]
fn huber_der1_band_boundary_is_strict() {
    // |diff| == delta is NOT < delta (strict): saturates to sign*delta, not diff.
    let delta = 2.0;
    // diff = +2.0 == delta -> saturated +delta (== diff here, but via the sign arm).
    assert!((huber_der1(0.0, 2.0, delta) - delta).abs() < 1e-12);
    // diff = -2.0, |diff| == delta -> -delta.
    assert!((huber_der1(2.0, 0.0, delta) - (-delta)).abs() < 1e-12);
}

#[test]
fn huber_der2_minus_one_in_band_zero_outside() {
    // error_functions.h:1621 — |diff|<delta ? -1 : 0.
    let delta = 1.0;
    assert!((huber_der2(0.0, 0.5, delta) - (-1.0)).abs() < 1e-12); // in band
    assert_eq!(huber_der2(0.0, 3.0, delta), 0.0); // outside band
    // Boundary |diff| == delta is outside (strict <) -> 0.
    assert_eq!(huber_der2(0.0, 1.0, delta), 0.0);
}

#[test]
fn expectile_der1_asymmetric_l2() {
    // error_functions.h:527 — e=target-approx; (e>0)?2a*e:2(1-a)*e.
    let alpha = 0.3;
    // e = +2 (>0): 2*0.3*2 = 1.2
    assert!((expectile_der1(0.0, 2.0, alpha) - 1.2).abs() < 1e-12);
    // e = -2 (<0): 2*0.7*(-2) = -2.8
    assert!((expectile_der1(2.0, 0.0, alpha) - (-2.8)).abs() < 1e-12);
    // alpha=0.5 reduces to the RMSE gradient e.
    assert!((expectile_der1(0.0, 1.7, 0.5) - 1.7).abs() < 1e-12);
}

#[test]
fn expectile_der1_zero_residual_boundary() {
    // e == 0 is NOT > 0: selects the below-branch 2*(1-a)*e = 0 (continuous here).
    assert!(expectile_der1(1.0, 1.0, 0.3).abs() < 1e-12);
}

#[test]
fn expectile_der2_piecewise_constant() {
    // error_functions.h:532 — (e>0)?-2a:-2(1-a).
    let alpha = 0.3;
    assert!((expectile_der2(0.0, 2.0, alpha) - (-0.6)).abs() < 1e-12); // e>0 -> -2*0.3
    assert!((expectile_der2(2.0, 0.0, alpha) - (-1.4)).abs() < 1e-12); // e<0 -> -2*0.7
    // e == 0 -> below-branch -2*(1-a).
    assert!((expectile_der2(1.0, 1.0, alpha) - (-1.4)).abs() < 1e-12);
}

// --- Wave-2 positive-domain / link losses (Plan 06.1-02) -------------------

/// Poisson der1 = `target - exp(approx)`, exp computed INLINE on the raw approx.
#[test]
fn poisson_der1_is_target_minus_exp_approx() {
    // approx = 0 -> exp(0) = 1, der1 = target - 1.
    assert!((poisson_der1(0.0, 3.0) - 2.0).abs() < 1e-12);
    // approx = 1 -> exp(1) = e, der1 = 5 - e.
    assert!((poisson_der1(1.0, 5.0) - (5.0 - std::f64::consts::E)).abs() < 1e-12);
    // A known approx: approx = ln(4) -> exp = 4, der1 = 10 - 4 = 6.
    assert!((poisson_der1(4.0_f64.ln(), 10.0) - 6.0).abs() < 1e-12);
}

/// Poisson der2 = `-exp(approx)` (strictly negative, convex).
#[test]
fn poisson_der2_is_negative_exp_approx() {
    assert!((poisson_der2(0.0, 99.0) - (-1.0)).abs() < 1e-12);
    assert!((poisson_der2(4.0_f64.ln(), 0.0) - (-4.0)).abs() < 1e-12);
    // der2 does not depend on target.
    assert!((poisson_der2(2.0, 1.0) - poisson_der2(2.0, 7.0)).abs() < 1e-12);
}

/// Tweedie der1 = `target*e^((1-p)*approx) - e^((2-p)*approx)` at p=1.5 (exp
/// INSIDE the der; raw approx).
#[test]
fn tweedie_der1_at_p_1_5() {
    let p = 1.5;
    // approx = 0 -> e^0 = 1 both terms: der1 = target*1 - 1 = target - 1.
    assert!((tweedie_der1(0.0, 4.0, p) - 3.0).abs() < 1e-12);
    // approx = 2, target = 3: 3*e^(-1) - e^(1).
    let expected = 3.0 * (-1.0_f64).exp() - (1.0_f64).exp();
    assert!((tweedie_der1(2.0, 3.0, p) - expected).abs() < 1e-12);
}

/// Tweedie der2 = `target*(1-p)*e^((1-p)*approx) - (2-p)*e^((2-p)*approx)` at p=1.5.
#[test]
fn tweedie_der2_at_p_1_5() {
    let p = 1.5;
    // approx = 0: target*(1-p) - (2-p) = target*(-0.5) - 0.5.
    assert!((tweedie_der2(0.0, 4.0, p) - (4.0 * -0.5 - 0.5)).abs() < 1e-12);
    let expected = 3.0 * (-0.5) * (-1.0_f64).exp() - 0.5 * (1.0_f64).exp();
    assert!((tweedie_der2(2.0, 3.0, p) - expected).abs() < 1e-12);
}

/// MAPE der1 = `sign(target-approx)/max(1.0,|target|)` — test the |target|<1 vs
/// >1 divisor boundary (Pitfall 7) and the sign branch.
#[test]
fn mape_der1_divisor_boundary_and_sign() {
    // |target| = 5 > 1 -> divisor 5. target>approx -> +1/5 = 0.2.
    assert!((mape_der1(2.0, 5.0) - 0.2).abs() < 1e-12);
    // target<approx -> -1/5.
    assert!((mape_der1(7.0, 5.0) - (-0.2)).abs() < 1e-12);
    // |target| = 0.5 < 1 -> divisor floored to 1.0. target>approx -> +1.
    assert!((mape_der1(0.0, 0.5) - 1.0).abs() < 1e-12);
    // |target| = 0.5, target<approx -> -1 (divisor floor 1.0).
    assert!((mape_der1(2.0, 0.5) - (-1.0)).abs() < 1e-12);
    // tie target == approx maps to the -1 branch (upstream `> 0 ? 1 : -1`).
    assert!((mape_der1(3.0, 3.0) - (-1.0 / 3.0)).abs() < 1e-12);
}

/// MAPE der2 is constant 0 (Pitfall 5: Newton undefined -> Gradient leaf).
#[test]
fn mape_der2_is_zero() {
    assert!(mape_der2(0.0, 5.0).abs() < 1e-12);
    assert!(mape_der2(2.0, 0.5).abs() < 1e-12);
    assert!(mape_der2(-100.0, 100.0).abs() < 1e-12);
}

// --- MultiClass softmax (coupled der + packed symmetric Hessian) -------------

#[test]
fn calc_softmax_uniform_at_equal_approx() {
    // All-equal approx -> uniform distribution (max-subtraction makes every
    // exponent 1.0, so each p = 1/k).
    let p = calc_softmax(&[0.0, 0.0, 0.0]);
    for &pd in &p {
        assert!((pd - 1.0 / 3.0).abs() < 1e-12);
    }
    // Probabilities sum to 1.
    let s: f64 = p.iter().sum();
    assert!((s - 1.0).abs() < 1e-12);
}

#[test]
fn calc_softmax_max_subtraction_no_overflow() {
    // A large-magnitude approx must NOT overflow exp to Inf/NaN (T-6.2-02): the
    // max-subtraction keeps every exponent <= 1.0. The dominant dimension's
    // probability approaches 1, the rest approach 0; nothing is NaN/Inf.
    let p = calc_softmax(&[1000.0, 0.0, -1000.0]);
    for &pd in &p {
        assert!(pd.is_finite());
        assert!((0.0..=1.0).contains(&pd));
    }
    let s: f64 = p.iter().sum();
    assert!((s - 1.0).abs() < 1e-12);
    assert!(p[0] > 0.999_999);
}

#[test]
fn softmax_ders_match_hand_computed_three_class() {
    // 3-class object, approx = [0, 0, 0] -> p = [1/3, 1/3, 1/3], target_class = 1.
    // der1[d] = δ(d==1) - p[d]: [-1/3, 2/3, -1/3].
    let (der1, der2) = softmax_ders(&[0.0, 0.0, 0.0], 1);
    let third = 1.0 / 3.0;
    assert!((der1[0] - (-third)).abs() < 1e-12);
    assert!((der1[1] - (2.0 * third)).abs() < 1e-12);
    assert!((der1[2] - (-third)).abs() < 1e-12);

    // Packed Hessian order [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)] of length 6.
    // diag (y,y) = p_y*(p_y-1) = (1/3)*(1/3 - 1) = -2/9.
    // off  (y,x) = p_y*p_x     = (1/3)*(1/3)     = 1/9.
    assert_eq!(der2.len(), 6);
    let diag = third * (third - 1.0); // -2/9
    let off = third * third; // 1/9
    for &idx in &[0usize, 3, 5] {
        assert!((der2[idx] - diag).abs() < 1e-12, "diag at {idx}");
    }
    for &idx in &[1usize, 2, 4] {
        assert!((der2[idx] - off).abs() < 1e-12, "off at {idx}");
    }
}

#[test]
fn softmax_ders_out_of_range_target_does_not_panic() {
    // An out-of-range target_class (>= k) leaves der1 = -p[d] (no +1) without
    // panicking — the caller's range validation is the defense (T-6.2-01).
    let (der1, _) = softmax_ders(&[0.1, 0.2, 0.3], 99);
    let p = calc_softmax(&[0.1, 0.2, 0.3]);
    for d in 0..3 {
        assert!((der1[d] - (-p[d])).abs() < 1e-12);
    }
}

// --- MultiClassOneVsAll (diagonal per-dimension sigmoid der) -----------------

#[test]
fn multiclass_onevsall_ders_match_sigmoid() {
    // approx_d = 0 -> sigmoid = 0.5. Target dimension: der1 = 1 - 0.5 = 0.5;
    // non-target: der1 = -0.5. der2 = -0.5*0.5 = -0.25 in both.
    let (d1_t, d2_t) = multiclass_onevsall_ders(0.0, true);
    assert!((d1_t - 0.5).abs() < 1e-12);
    assert!((d2_t - (-0.25)).abs() < 1e-12);
    let (d1_n, d2_n) = multiclass_onevsall_ders(0.0, false);
    assert!((d1_n - (-0.5)).abs() < 1e-12);
    assert!((d2_n - (-0.25)).abs() < 1e-12);
}

// --- MultiLogloss / MultiCrossEntropy (diagonal TMultiCrossEntropyError der) --

#[test]
fn multi_crossentropy_ders_is_target_minus_sigmoid_diagonal() {
    // approx_d = 0 -> sigmoid = 0.5. der1 = target - 0.5; der2 = -0.5*0.5 = -0.25.
    let (d1_one, d2_one) = multi_crossentropy_ders(0.0, 1.0);
    assert!((d1_one - 0.5).abs() < 1e-12);
    assert!((d2_one - (-0.25)).abs() < 1e-12);
    let (d1_zero, d2_zero) = multi_crossentropy_ders(0.0, 0.0);
    assert!((d1_zero - (-0.5)).abs() < 1e-12);
    assert!((d2_zero - (-0.25)).abs() < 1e-12);
}

#[test]
fn multi_crossentropy_der1_matches_target_minus_sigmoid_over_range() {
    // Spot-check the der1 = target_d - sigmoid(approx_d) closed form at several
    // logits (the MultiLogloss/MultiCrossEntropy diagonal der reuses the scalar
    // sigmoid per dimension).
    for &a in &[-2.0_f64, -0.7, 0.3, 1.5, 3.0] {
        for &t in &[0.0_f64, 0.25, 1.0] {
            let (der1, der2) = multi_crossentropy_ders(a, t);
            let p = sigmoid(a);
            assert!((der1 - (t - p)).abs() < 1e-12, "der1 mismatch at a={a} t={t}");
            assert!((der2 - (-p * (1.0 - p))).abs() < 1e-12, "der2 mismatch");
        }
    }
}

#[test]
fn multi_crossentropy_der2_equals_onevsall_diagonal() {
    // MultiLogloss/MultiCrossEntropy and OneVsAll share the SAME diagonal Hessian
    // entry -sigmoid*(1-sigmoid); only the der1 target-vs-class-indicator differs.
    for &a in &[-1.0_f64, 0.0, 0.8, 2.2] {
        let (_, ce_d2) = multi_crossentropy_ders(a, 1.0);
        let (_, ova_d2) = multiclass_onevsall_ders(a, true);
        assert!((ce_d2 - ova_d2).abs() < 1e-12, "diagonal der2 must match OneVsAll at a={a}");
    }
}
