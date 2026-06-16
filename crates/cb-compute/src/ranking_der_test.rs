//! Unit tests for the grouped der seam ([`super::calc_ders_for_queries`],
//! [`super::group_reduce_weighted`], and the [`Runtime::compute_gradients_grouped`]
//! default) — LOSS-04, D-6.3-03.
//!
//! Source/test separation (INFRA-06): dedicated file, linked from `ranking_der.rs`
//! via the `#[path]` footer — never inline.

use super::{calc_ders_for_queries, group_reduce_weighted, Competitor, GroupSpan};
use crate::runtime::{Derivatives, Loss, Runtime};
use cb_core::{sum_f64, CbResult};

fn span(begin: usize, end: usize) -> GroupSpan {
    GroupSpan {
        begin,
        end,
        weight: 1.0,
        competitors: vec![Vec::new(); end - begin],
    }
}

#[test]
fn group_reduce_weighted_matches_hand_sum_f64_uniform() {
    let slice = [1.0, 2.0, 3.0, 4.0];
    // Uniform weights → plain sum.
    let got = group_reduce_weighted(&slice, &[]);
    let want = sum_f64(&[1.0, 2.0, 3.0, 4.0]);
    assert!((got - want).abs() < 1e-12);
    assert!((got - 10.0).abs() < 1e-12);
}

#[test]
fn group_reduce_weighted_matches_hand_sum_f64_weighted() {
    let slice = [1.0, 2.0, 3.0];
    let weights = [0.5, 2.0, 1.0];
    let got = group_reduce_weighted(&slice, &weights);
    // 1*0.5 + 2*2.0 + 3*1.0 = 7.5, reduced through sum_f64.
    let want = sum_f64(&[0.5, 4.0, 3.0]);
    assert!((got - want).abs() < 1e-12);
    assert!((got - 7.5).abs() < 1e-12);
}

#[test]
fn empty_groups_yields_empty_ders_no_panic() {
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0, 0.0];
    let out = calc_ders_for_queries(&Loss::Rmse, &approx, &target, &[], &[], 0).unwrap();
    assert!(out.is_empty());
}

#[test]
fn group_span_out_of_range_is_degenerate_error() {
    let approx = [0.1, 0.2];
    let target = [0.0, 1.0];
    // Group claims [0, 5) but only 2 objects exist.
    let groups = [span(0, 5)];
    let err = calc_ders_for_queries(&Loss::Rmse, &approx, &target, &[], &groups, 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::Degenerate(_)));
}

#[test]
fn approx_target_length_mismatch_is_degenerate_error() {
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0];
    let err = calc_ders_for_queries(&Loss::Rmse, &approx, &target, &[], &[], 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::Degenerate(_)));
}

#[test]
fn unwired_loss_variant_returns_typed_error() {
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0, 0.0];
    let groups = [span(0, 3)];
    // Every loss arm is unwired in Plan 06.3-01 → typed OutOfRange error.
    let err =
        calc_ders_for_queries(&Loss::Logloss, &approx, &target, &[], &groups, 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));
}

#[test]
fn per_group_slicing_validates_each_span() {
    // Two valid groups spanning [0,2) and [2,4): the seam slices each before
    // dispatch. With an unwired loss the first group's dispatch surfaces the
    // typed error — proving the slice bounds passed (no Degenerate span error).
    let approx = [0.1, 0.2, 0.3, 0.4];
    let target = [0.0, 1.0, 0.0, 1.0];
    let groups = [span(0, 2), span(2, 4)];
    let err =
        calc_ders_for_queries(&Loss::Logloss, &approx, &target, &[], &groups, 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));
}

#[test]
fn competitor_round_trips_in_group_span() {
    // Sanity: a GroupSpan carrying competitor edges is well-formed and slices.
    let mut g = span(0, 3);
    g.competitors[0].push(Competitor {
        id: 1,
        weight: 1.0,
    });
    assert_eq!(g.size(), 3);
    assert_eq!(g.competitors[0][0].id, 1);
}

/// Minimal `Runtime` exercising ONLY the grouped default — `compute_gradients`
/// is unused here, so it returns a degenerate error.
struct DummyRuntime;

impl Runtime for DummyRuntime {
    fn compute_gradients(
        &self,
        _loss: &Loss,
        _approx: &[f64],
        _target: &[f64],
        _approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        Err(cb_core::CbError::Degenerate("unused in this test".to_owned()))
    }
}

#[test]
fn runtime_compute_gradients_grouped_default_delegates_to_seam() {
    let rt = DummyRuntime;
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0, 0.0];
    let groups = [span(0, 3)];
    // The default impl delegates to calc_ders_for_queries → unwired typed error.
    let err = rt
        .compute_gradients_grouped(&Loss::Logloss, &approx, &target, &[], &groups, 0)
        .unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));

    // Empty groups → empty der set through the trait default (no panic).
    let out = rt
        .compute_gradients_grouped(&Loss::Rmse, &approx, &target, &[], &[], 0)
        .unwrap();
    assert!(out.is_empty());
}

// --- QueryRMSE grouped der (LOSS-04 Wave A) -----------------------------------

#[test]
fn queryrmse_two_group_der_matches_hand_computed() {
    // Two groups: [0,2) and [2,5). Unweighted (w=1).
    // Group 0: approx [0.0, 1.0], target [1.0, 0.0].
    //   residuals (t-a) = [1.0, -1.0]; queryAvrg = (1 + -1)/2 = 0.
    //   der1 = (t - a - 0)·1 = [1.0, -1.0]; der2 = [-1, -1].
    // Group 1: approx [0.0, 0.0, 1.0], target [2.0, 1.0, 0.0].
    //   residuals = [2, 1, -1]; queryAvrg = (2+1-1)/3 = 2/3.
    //   der1 = [2 - 2/3, 1 - 2/3, -1 - 2/3] = [4/3, 1/3, -5/3]; der2 = [-1,-1,-1].
    let approx = [0.0, 1.0, 0.0, 0.0, 1.0];
    let target = [1.0, 0.0, 2.0, 1.0, 0.0];
    let groups = [span(0, 2), span(2, 5)];
    let out = calc_ders_for_queries(&Loss::QueryRmse, &approx, &target, &[], &groups, 0).unwrap();
    assert_eq!(out.len(), 2);
    let g0 = &out[0];
    assert!((g0.der1[0] - 1.0).abs() < 1e-12);
    assert!((g0.der1[1] - (-1.0)).abs() < 1e-12);
    assert!(g0.der2.iter().all(|&d| (d - (-1.0)).abs() < 1e-12));
    let g1 = &out[1];
    assert!((g1.der1[0] - (4.0 / 3.0)).abs() < 1e-12);
    assert!((g1.der1[1] - (1.0 / 3.0)).abs() < 1e-12);
    assert!((g1.der1[2] - (-5.0 / 3.0)).abs() < 1e-12);
    assert!(g1.der2.iter().all(|&d| (d - (-1.0)).abs() < 1e-12));
}

#[test]
fn queryrmse_empty_group_yields_empty_ders_no_divide() {
    // A zero-size group contributes an empty der set; no divide-by-zero on Σw=0.
    let approx = [0.5, 0.5];
    let target = [1.0, 0.0];
    let groups = [
        GroupSpan { begin: 0, end: 0, weight: 1.0, competitors: Vec::new() },
        span(0, 2),
    ];
    let out = calc_ders_for_queries(&Loss::QueryRmse, &approx, &target, &[], &groups, 0).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out[0].der1.is_empty() && out[0].der2.is_empty());
    assert_eq!(out[1].der1.len(), 2);
}

#[test]
fn queryrmse_weighted_folds_weight_into_der() {
    // Group [0,2), weights [2.0, 1.0]. residuals (t-a) = [1.0, -2.0].
    // queryAvrg = (1·2 + -2·1)/(2+1) = (2 - 2)/3 = 0.
    // der1 = (t-a-0)·w = [1·2, -2·1] = [2.0, -2.0]; der2 = [-1·2, -1·1] = [-2,-1].
    let approx = [0.0, 2.0];
    let target = [1.0, 0.0];
    let weights = [2.0, 1.0];
    let groups = [span(0, 2)];
    let out =
        calc_ders_for_queries(&Loss::QueryRmse, &approx, &target, &weights, &groups, 0).unwrap();
    let g = &out[0];
    assert!((g.der1[0] - 2.0).abs() < 1e-12);
    assert!((g.der1[1] - (-2.0)).abs() < 1e-12);
    assert!((g.der2[0] - (-2.0)).abs() < 1e-12);
    assert!((g.der2[1] - (-1.0)).abs() < 1e-12);
}

// --- QuerySoftMax grouped der (LOSS-04 Wave A) --------------------------------

#[test]
fn querysoftmax_single_group_der_matches_hand_computed() {
    // One group [0,3); beta=1, lambda=0; unweighted.
    // approx [0.0, 0.0, ln2]; target [1.0, 0.0, 0.0].
    // maxApprox = ln2; shifted exp = [exp(-ln2), exp(-ln2), exp(0)] = [0.5, 0.5, 1.0].
    // weighted_exp (w=1) = [0.5, 0.5, 1.0]; sumExp = 2.0.
    // p = [0.25, 0.25, 0.5]; sumWTargets = 1·1 = 1.
    // der1 = 1·(-1·p + w·target):
    //   obj0: -0.25 + 1 = 0.75 ; obj1: -0.25 + 0 = -0.25 ; obj2: -0.5 + 0 = -0.5
    // der2 = 1·1·(1·p·(p-1) - 0):
    //   obj0: 0.25·(-0.75) = -0.1875 ; obj1: -0.1875 ; obj2: 0.5·(-0.5) = -0.25
    let ln2 = 2.0_f64.ln();
    let approx = [0.0, 0.0, ln2];
    let target = [1.0, 0.0, 0.0];
    let groups = [span(0, 3)];
    let loss = Loss::QuerySoftMax { lambda: 0.0, beta: 1.0 };
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];
    assert!((g.der1[0] - 0.75).abs() < 1e-12, "der1[0]={}", g.der1[0]);
    assert!((g.der1[1] - (-0.25)).abs() < 1e-12);
    assert!((g.der1[2] - (-0.5)).abs() < 1e-12);
    assert!((g.der2[0] - (-0.1875)).abs() < 1e-12);
    assert!((g.der2[1] - (-0.1875)).abs() < 1e-12);
    assert!((g.der2[2] - (-0.25)).abs() < 1e-12);
}

#[test]
fn querysoftmax_max_shift_avoids_exp_overflow() {
    // A group with huge approx values must NOT overflow exp to Inf/NaN: the
    // max-shift subtracts maxApprox before exp (error_functions.cpp:540-552).
    let approx = [1000.0, 1000.5, 999.0];
    let target = [1.0, 0.0, 0.0];
    let groups = [span(0, 3)];
    let loss = Loss::QuerySoftMax { lambda: 0.01, beta: 1.0 };
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];
    assert!(g.der1.iter().all(|d| d.is_finite()), "der1 must be finite (no exp overflow)");
    assert!(g.der2.iter().all(|d| d.is_finite()), "der2 must be finite (no exp overflow)");
}

#[test]
fn querysoftmax_zero_sum_targets_yields_zero_ders() {
    // sumWTargets <= 0 (all targets 0) → ders 0 (error_functions.cpp:571-576).
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 0.0, 0.0];
    let groups = [span(0, 3)];
    let loss = Loss::QuerySoftMax { lambda: 0.01, beta: 1.0 };
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];
    assert!(g.der1.iter().all(|&d| d == 0.0));
    assert!(g.der2.iter().all(|&d| d == 0.0));
}

#[test]
fn querysoftmax_validate_rejects_bad_params() {
    // lambda < 0 and beta <= 0 are rejected by Loss::validate (T-06.3-02-03).
    assert!(Loss::QuerySoftMax { lambda: -0.1, beta: 1.0 }.validate().is_err());
    assert!(Loss::QuerySoftMax { lambda: 0.0, beta: 0.0 }.validate().is_err());
    assert!(Loss::QuerySoftMax { lambda: f64::NAN, beta: 1.0 }.validate().is_err());
    assert!(Loss::QuerySoftMax { lambda: 0.01, beta: 1.0 }.validate().is_ok());
    // QueryRmse carries no params — always valid.
    assert!(Loss::QueryRmse.validate().is_ok());
}

// --- PairLogit grouped der over Competitors (LOSS-04 Wave B) ------------------

fn group_with_competitors(begin: usize, end: usize, comps: Vec<Vec<Competitor>>) -> GroupSpan {
    GroupSpan { begin, end, weight: 1.0, competitors: comps }
}

#[test]
fn pairlogit_single_pair_group_der_matches_hand_computed() {
    // One group [0,2); a single pair: winner = doc 0, loser = doc 1, weight 1.
    // approx [0.0, 0.0] (raw) ⇒ expApprox [1, 1].
    //   p = exp(loser)/(exp(loser)+exp(winner)) = 1/(1+1) = 0.5.
    //   winnerDer += w·p = 0.5 ; der1[loser] -= w·p = -0.5.
    //   winnerDer2 += w·p·(p-1) = 0.5·(-0.5) = -0.25 ; der2[loser] += -0.25.
    //   der1[winner] += 0.5 ; der2[winner] += -0.25.
    // ⇒ der1 = [0.5, -0.5]; der2 = [-0.25, -0.25].
    let approx = [0.0, 0.0];
    let target = [1.0, 0.0]; // target unused by PairLogit
    let comps = vec![vec![Competitor { id: 1, weight: 1.0 }], Vec::new()];
    let groups = [group_with_competitors(0, 2, comps)];
    let out = calc_ders_for_queries(&Loss::PairLogit, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];
    assert!((g.der1[0] - 0.5).abs() < 1e-12, "der1[0]={}", g.der1[0]);
    assert!((g.der1[1] - (-0.5)).abs() < 1e-12, "der1[1]={}", g.der1[1]);
    assert!((g.der2[0] - (-0.25)).abs() < 1e-12, "der2[0]={}", g.der2[0]);
    assert!((g.der2[1] - (-0.25)).abs() < 1e-12, "der2[1]={}", g.der2[1]);
}

#[test]
fn pairlogit_pairwise_shares_pairlogit_der() {
    // PairLogitPairwise uses the SAME der as PairLogit (only the leaf path differs).
    let approx = [0.5, -0.5];
    let target = [1.0, 0.0];
    let comps = vec![vec![Competitor { id: 1, weight: 2.0 }], Vec::new()];
    let groups = [group_with_competitors(0, 2, comps)];
    let a =
        calc_ders_for_queries(&Loss::PairLogit, &approx, &target, &[], &groups, 0).unwrap();
    let b = calc_ders_for_queries(
        &Loss::PairLogitPairwise,
        &approx,
        &target,
        &[],
        &groups,
        0,
    )
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn pairlogit_pair_weight_scales_der() {
    // weight 2 doubles the contribution vs weight 1 at the symmetric p=0.5 point.
    let approx = [0.0, 0.0];
    let target = [1.0, 0.0];
    let comps = vec![vec![Competitor { id: 1, weight: 2.0 }], Vec::new()];
    let groups = [group_with_competitors(0, 2, comps)];
    let out = calc_ders_for_queries(&Loss::PairLogit, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];
    // winnerDer = 2·0.5 = 1.0 ; der1[loser] = -1.0 ; der2 = 2·(-0.25) = -0.5.
    assert!((g.der1[0] - 1.0).abs() < 1e-12);
    assert!((g.der1[1] - (-1.0)).abs() < 1e-12);
    assert!((g.der2[0] - (-0.5)).abs() < 1e-12);
    assert!((g.der2[1] - (-0.5)).abs() < 1e-12);
}

// --- LambdaMart grouped der (LOSS-04 Wave B) ----------------------------------

#[test]
fn lambdamart_two_doc_ndcg_der_matches_hand_computed() {
    // One group [0,2); metric NDCG, sigma=1, top=-1, norm=false (isolate the core).
    // approx [1.0, 0.0]; target [1.0, 0.0]. Sort by approx desc ⇒ order [0,1].
    //   idealScore (target sorted desc [1,0], top=2):
    //     1/log2(2) + 0/log2(3) = 1/1 = 1.0.
    //   pair (firstId=0,secondId=1): firstTarget=1 > secondTarget=0.
    //     approxDiff = approx[0]-approx[1] = 1.0.
    //     dcgNum = 1 - 0 = 1 ; dcgDen = |1/log2(2) - 1/log2(3)| = |1 - 0.63093| = 0.36907.
    //     delta = 1·0.36907 / 1.0 = 0.36907.
    //     σ = 1/(1+exp(1)) = 0.268941 ; antigrad = -1·δ·σ ; hessian = 1·δ·σ(1-σ).
    let approx = [1.0_f64, 0.0];
    let target = [1.0_f64, 0.0];
    let groups = [span(0, 2)];
    let loss = Loss::LambdaMart {
        metric: crate::runtime::LambdaMartMetric::Ndcg,
        sigma: 1.0,
        top: -1,
        norm: false,
    };
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];

    let dcg_den = (1.0 / (2.0_f64).log2() - 1.0 / (3.0_f64).log2()).abs();
    let delta = 1.0 * dcg_den / 1.0;
    let sig = 1.0 / (1.0 + 1.0_f64.exp());
    let antigrad = -1.0 * delta * sig;
    let hessian = 1.0 * 1.0 * delta * sig * (1.0 - sig);
    // doc 0 is the high doc (firstId=0): der1 += antigrad, der2 += hessian.
    // doc 1 is the low doc: der1 -= antigrad, der2 += hessian.
    assert!((g.der1[0] - antigrad).abs() < 1e-12, "der1[0]={}", g.der1[0]);
    assert!((g.der1[1] - (-antigrad)).abs() < 1e-12, "der1[1]={}", g.der1[1]);
    assert!((g.der2[0] - hessian).abs() < 1e-12, "der2[0]={}", g.der2[0]);
    assert!((g.der2[1] - hessian).abs() < 1e-12, "der2[1]={}", g.der2[1]);
}

#[test]
fn lambdamart_singleton_group_yields_zero_der() {
    // A 1-doc group has no ordered pairs ⇒ zero der (no panic, count<=1 guard).
    let approx = [0.7];
    let target = [2.0];
    let groups = [span(0, 1)];
    let loss = Loss::LambdaMart {
        metric: crate::runtime::LambdaMartMetric::Ndcg,
        sigma: 1.0,
        top: -1,
        norm: true,
    };
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];
    assert_eq!(g.der1, vec![0.0]);
    assert_eq!(g.der2, vec![0.0]);
}

#[test]
fn lambdamart_equal_targets_yields_zero_der() {
    // No ordered pair has firstTarget > secondTarget ⇒ zero der.
    let approx = [0.5, 0.2, 0.9];
    let target = [1.0, 1.0, 1.0];
    let groups = [span(0, 3)];
    let loss = Loss::LambdaMart {
        metric: crate::runtime::LambdaMartMetric::Ndcg,
        sigma: 1.0,
        top: -1,
        norm: true,
    };
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &groups, 0).unwrap();
    let g = &out[0];
    assert!(g.der1.iter().all(|&d| d == 0.0));
    assert!(g.der2.iter().all(|&d| d == 0.0));
}

#[test]
fn lambdamart_validate_rejects_bad_sigma_and_top() {
    let bad_sigma = Loss::LambdaMart {
        metric: crate::runtime::LambdaMartMetric::Ndcg,
        sigma: 0.0,
        top: -1,
        norm: true,
    };
    assert!(bad_sigma.validate().is_err());
    let bad_top = Loss::LambdaMart {
        metric: crate::runtime::LambdaMartMetric::Ndcg,
        sigma: 1.0,
        top: 0,
        norm: true,
    };
    assert!(bad_top.validate().is_err());
    let ok = Loss::LambdaMart {
        metric: crate::runtime::LambdaMartMetric::Ndcg,
        sigma: 1.0,
        top: -1,
        norm: true,
    };
    assert!(ok.validate().is_ok());
}

#[test]
fn is_pairwise_scoring_and_plain_only_predicates() {
    use super::{is_pairwise_scoring, is_plain_only};
    assert!(is_pairwise_scoring(&Loss::PairLogitPairwise));
    assert!(is_plain_only(&Loss::PairLogitPairwise));
    // Non-pairwise ranking losses are neither.
    assert!(!is_pairwise_scoring(&Loss::PairLogit));
    assert!(!is_plain_only(&Loss::PairLogit));
    assert!(!is_pairwise_scoring(&Loss::QueryRmse));
    assert!(!is_pairwise_scoring(&Loss::Rmse));
}
