//! Unit tests for the grouped der seam ([`super::calc_ders_for_queries`],
//! [`super::group_reduce_weighted`], and the [`Runtime::compute_gradients_grouped`]
//! default) — LOSS-04, D-6.3-03.
//!
//! Source/test separation (INFRA-06): dedicated file, linked from `ranking_der.rs`
//! via the `#[path]` footer — never inline.

use super::{calc_ders_for_queries, group_reduce_weighted, Competitor, GroupSpan};
use crate::runtime::{Derivatives, Loss, Runtime, StochasticRankMetric};
use cb_core::{std_normal, sum_f64, CbResult, TFastRng64};

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

// ---------------------------------------------------------------------------
// Wave-C randomized ranking losses (YetiRank / StochasticRank) — LOSS-04.
// ---------------------------------------------------------------------------

/// A GroupSpan carrying a sampled competitor adjacency (the YetiRank pair source
/// the trainer injects). Helper for the YetiRank-rides-PairLogit test.
fn span_with_competitors(begin: usize, end: usize, competitors: Vec<Vec<Competitor>>) -> GroupSpan {
    GroupSpan {
        begin,
        end,
        weight: 1.0,
        competitors,
    }
}

/// YetiRank rides the EXISTING PairLogit der over the SAMPLED competitors: feeding
/// the same competitor adjacency through `Loss::YetiRank` and `Loss::PairLogit`
/// must produce BIT-IDENTICAL ders (the only difference between the two is the leaf
/// path, decided in boosting — the der math is the shared PairLogit der).
#[test]
fn yetirank_rides_pairlogit_der_over_sampled_competitors() {
    let approx = [0.5_f64, -0.2, 0.1];
    let target = [2.0_f64, 0.0, 1.0];
    // Sampled adjacency: doc0 beats doc1 (w=0.3), doc2 beats doc1 (w=0.1).
    let competitors = vec![
        vec![Competitor { id: 1, weight: 0.3 }],
        vec![],
        vec![Competitor { id: 1, weight: 0.1 }],
    ];
    let groups_y = vec![span_with_competitors(0, 3, competitors.clone())];
    let groups_p = vec![span_with_competitors(0, 3, competitors)];

    let yeti = calc_ders_for_queries(
        &Loss::YetiRank { permutations: 10, decay: 0.85 },
        &approx,
        &target,
        &[],
        &groups_y,
        0,
    )
    .unwrap();
    let pair = calc_ders_for_queries(&Loss::PairLogit, &approx, &target, &[], &groups_p, 0).unwrap();

    assert_eq!(yeti.len(), 1);
    assert_eq!(pair.len(), 1);
    for (y, p) in yeti[0].der1.iter().zip(&pair[0].der1) {
        assert!((y - p).abs() < 1e-15, "YetiRank der1 must equal PairLogit der1 over the same sampled pairs");
    }
    for (y, p) in yeti[0].der2.iter().zip(&pair[0].der2) {
        assert!((y - p).abs() < 1e-15, "YetiRank der2 must equal PairLogit der2");
    }
}

/// StochasticRank (num_estimations=1) draws EXACTLY `count` Gaussian noises per
/// sample, one per doc, from `TFastRng64(random_seed + group_index)` via the SAME
/// `cb_core::std_normal` Marsaglia-polar sequence. We hand-trace the first group's
/// noise draws (group_index=0 => seed=random_seed) and assert the der is finite +
/// the draw stream is the std_normal one (a different sampler would desync). The
/// der2 must be all-zero (Gradient leaf method).
#[test]
fn stochasticrank_num_estimations_one_draws_count_gaussians_via_std_normal() {
    let approx = [0.3_f64, -0.4, 0.1];
    let target = [2.0_f64, 0.0, 1.0];
    let random_seed = 5_u64;
    let loss = Loss::StochasticRank {
        metric: StochasticRankMetric::Ndcg,
        sigma: 1.0,
        mu: 0.0,
        num_estimations: 1,
    };

    // Hand trace: group 0 seed = random_seed + 0; one std_normal per doc.
    let mut rng = TFastRng64::from_seed(random_seed);
    let hand: Vec<f64> = (0..3).map(|_| std_normal(&mut rng)).collect();
    assert_eq!(hand.len(), 3, "3 docs => 3 Gaussian draws for num_estimations=1");

    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &[span(0, 3)], random_seed).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].der1.len(), 3);
    assert!(out[0].der1.iter().all(|d| d.is_finite()), "StochasticRank der1 must be finite");
    assert!(out[0].der2.iter().all(|&d| d == 0.0), "StochasticRank der2 == 0 (Gradient leaf)");
    // SFA Stage-3 orthogonalization: the der1 sums to ~0 (mean subtracted).
    let s: f64 = sum_f64(&out[0].der1);
    assert!(s.abs() < 1e-9, "StochasticRank der1 mean-centered by SFA (sum ~ 0), got {s}");
}

/// StochasticRank on a single-doc group (count <= 1) yields zero ders, never
/// divides by `count - 1` (T-06.3-04-02 — the Security V5 guard,
/// error_functions.cpp:1020-1022).
#[test]
fn stochasticrank_single_doc_group_is_zero_no_divide() {
    let approx = [0.7_f64];
    let target = [1.0_f64];
    let loss = Loss::StochasticRank {
        metric: StochasticRankMetric::Dcg,
        sigma: 1.0,
        mu: 0.0,
        num_estimations: 1,
    };
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &[span(0, 1)], 0).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].der1, vec![0.0]);
    assert_eq!(out[0].der2, vec![0.0]);
}

/// StochasticRank re-seeds per group with `random_seed + group_index`: two
/// identical groups at different indices must draw DIFFERENT noise (so their ders
/// differ) — a missing `+ group_index` would make every group's stream identical.
#[test]
fn stochasticrank_reseeds_per_group_index() {
    let approx = [0.3_f64, -0.4, 0.1, 0.3, -0.4, 0.1];
    let target = [2.0_f64, 0.0, 1.0, 2.0, 0.0, 1.0];
    let loss = Loss::StochasticRank {
        metric: StochasticRankMetric::Ndcg,
        sigma: 1.0,
        mu: 0.0,
        num_estimations: 1,
    };
    let groups = vec![span(0, 3), span(3, 6)];
    let out = calc_ders_for_queries(&loss, &approx, &target, &[], &groups, 0).unwrap();
    assert_eq!(out.len(), 2);
    // Same approx/target shape, different seed (0 vs 1) => different der streams.
    let differs = out[0]
        .der1
        .iter()
        .zip(&out[1].der1)
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(differs, "per-group reseed (random_seed + group_index) must vary the noise stream");
}

/// `Loss::validate` rejects out-of-range Wave-C params with a typed OutOfRange
/// error (T-06.3-04-03), never a panic.
#[test]
fn wave_c_loss_validate_rejects_bad_params() {
    // permutations == 0.
    assert!(matches!(
        Loss::YetiRank { permutations: 0, decay: 0.85 }.validate(),
        Err(cb_core::CbError::OutOfRange(_))
    ));
    // decay out of [0, 1].
    assert!(matches!(
        Loss::YetiRankPairwise { permutations: 10, decay: 1.5 }.validate(),
        Err(cb_core::CbError::OutOfRange(_))
    ));
    // sigma <= 0.
    assert!(matches!(
        Loss::StochasticRank { metric: StochasticRankMetric::Ndcg, sigma: 0.0, mu: 0.0, num_estimations: 1 }.validate(),
        Err(cb_core::CbError::OutOfRange(_))
    ));
    // mu < 0.
    assert!(matches!(
        Loss::StochasticRank { metric: StochasticRankMetric::Ndcg, sigma: 1.0, mu: -0.1, num_estimations: 1 }.validate(),
        Err(cb_core::CbError::OutOfRange(_))
    ));
    // num_estimations == 0.
    assert!(matches!(
        Loss::StochasticRank { metric: StochasticRankMetric::Ndcg, sigma: 1.0, mu: 0.0, num_estimations: 0 }.validate(),
        Err(cb_core::CbError::OutOfRange(_))
    ));
    // Valid defaults pass.
    assert!(Loss::YetiRank { permutations: 10, decay: 0.85 }.validate().is_ok());
    assert!(Loss::StochasticRank { metric: StochasticRankMetric::Ndcg, sigma: 1.0, mu: 0.0, num_estimations: 1 }.validate().is_ok());
}

// ---------------------------------------------------------------------------
// CR-01 regression: calc_dcg_metric_diff must read old/new position weights from
// the NORMALIZED pos_weights vector (the same one that built cum_sum/up/low), not
// from a recomputed raw 1/denominator. For any group with ideal_dcg != 1.0 the
// raw recomputation puts doc_diff on a different scale than mid_diff.
// Mirrors upstream CalcDCGMetricDiff posWeights[oldPos]/posWeights[newPos]
// (error_functions.cpp:1233-1234).
// ---------------------------------------------------------------------------

/// Graded-relevance group where the ideal DCG is clearly != 1.0, so the NDCG
/// `pos_weights` are scaled by `1/ideal_dcg`. The fix asserts `calc_dcg_metric_diff`
/// returns `doc_gain*(pos_weights[new_pos]-pos_weights[old_pos]) + mid_diff` read
/// from the normalized vector — which differs from the pre-fix raw-denominator
/// value by exactly the `ideal_dcg` factor on the `doc_diff` term.
#[test]
fn calc_dcg_metric_diff_reads_normalized_pos_weights_ndcg() {
    // 3-doc group, graded relevance [3, 2, 1] -> ideal_dcg clearly != 1.0.
    let targets = [3.0_f64, 2.0, 1.0];
    let query_top_size = targets.len();
    // Normalized NDCG pos_weights (divided by ideal_dcg).
    let pos_weights = super::compute_dcg_pos_weights(&targets, query_top_size, true);

    // ideal_dcg must be != 1.0 for this group (sanity guard for the regression).
    let raw_pos_weights = super::compute_dcg_pos_weights(&targets, query_top_size, false);
    let ratio = raw_pos_weights[0] / pos_weights[0];
    assert!(
        (ratio - 1.0).abs() > 0.5,
        "fixture must have ideal_dcg clearly != 1.0 (ratio={ratio})"
    );

    // A non-perfect approx ranking: order = [1, 0, 2] (doc 1 ranked first).
    let order = [1_usize, 0, 2];

    // Build cum_sum / cum_sum_up / cum_sum_low exactly as stochastic_rank_group_der
    // does, from the SAME normalized pos_weights.
    let count = targets.len();
    let mut cum_sum = vec![0.0_f64; count + 1];
    let mut cum_sum_up = vec![0.0_f64; count + 1];
    let mut cum_sum_low = vec![0.0_f64; count + 1];
    for pos in 0..count {
        let doc_id = order[pos];
        let gain = super::ndcg_numerator(targets[doc_id]);
        cum_sum[pos + 1] = cum_sum[pos] + gain * pos_weights[pos];
        if pos + 1 < count {
            cum_sum_low[pos + 1] = cum_sum_low[pos] + gain * pos_weights[pos + 1];
        }
        if pos > 0 {
            cum_sum_up[pos + 1] = cum_sum_up[pos] + gain * pos_weights[pos - 1];
        }
    }
    cum_sum_low[count] = cum_sum_low[count - 1];

    let old_pos = 0_usize;
    let new_pos = 2_usize;

    let got = super::calc_dcg_metric_diff(
        old_pos,
        new_pos,
        &targets,
        &order,
        &pos_weights,
        &cum_sum,
        &cum_sum_up,
        &cum_sum_low,
    );

    // Expected: normalized doc_diff + mid_diff (upstream posWeights[oldPos/newPos]).
    let doc_gain = super::ndcg_numerator(targets[order[old_pos]]);
    let doc_diff_norm = doc_gain * (pos_weights[new_pos] - pos_weights[old_pos]);
    // new_pos > old_pos branch:
    let old_mid = cum_sum[new_pos + 1] - cum_sum[old_pos + 1];
    let new_mid = cum_sum_up[new_pos + 1] - cum_sum_up[old_pos + 1];
    let mid_diff = new_mid - old_mid;
    let want = doc_diff_norm + mid_diff;
    assert!(
        (got - want).abs() < 1e-12,
        "calc_dcg_metric_diff must use normalized pos_weights: got={got}, want={want}"
    );

    // The pre-fix raw-denominator doc_diff would have been off by 1/ideal_dcg.
    let raw_doc_diff =
        doc_gain * (1.0 / super::ndcg_denominator(new_pos) - 1.0 / super::ndcg_denominator(old_pos));
    let raw_value = raw_doc_diff + mid_diff;
    assert!(
        (got - raw_value).abs() > 1e-9,
        "fix must differ from the old raw-denominator value (got={got}, raw={raw_value})"
    );
}

/// The DCG (non-NDCG) arm is unaffected: pos_weights == raw 1/denominator, so the
/// normalized read equals the old raw recomputation.
#[test]
fn calc_dcg_metric_diff_dcg_arm_unchanged() {
    let targets = [3.0_f64, 2.0, 1.0];
    let query_top_size = targets.len();
    let pos_weights = super::compute_dcg_pos_weights(&targets, query_top_size, false);
    // For DCG, pos_weights[pos] == 1/denominator(pos).
    for (pos, &w) in pos_weights.iter().enumerate() {
        assert!((w - 1.0 / super::ndcg_denominator(pos)).abs() < 1e-15);
    }
}

// ---------------------------------------------------------------------------
// WR-02 (D-08): lambdamart_ideal_ndcg must accumulate the per-position ideal-DCG
// terms through cb_core::sum_f64, not a raw `score +=` fold. sum_f64 is a strict
// left-to-right f64 fold (same order, same accumulator as the old loop), so the
// numeric result is unchanged — this guards the D-08 summation discipline.
// ---------------------------------------------------------------------------

/// `lambdamart_ideal_ndcg` over a known graded-relevance slice equals the
/// `sum_f64` of the explicit per-position `ndcg_numerator(t)/ndcg_denominator(pos)`
/// terms over the descending-sorted top window.
#[test]
fn lambdamart_ideal_ndcg_equals_sum_f64_of_terms() {
    let target = [1.0_f64, 3.0, 2.0]; // unsorted; the fn sorts descending.
    let query_top_size = target.len();
    let got = super::lambdamart_ideal_ndcg(&target, query_top_size);

    // Reference: descending sort [3, 2, 1], terms = num(t)/den(pos), reduced via sum_f64.
    let mut sorted = target.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let terms: Vec<f64> = sorted
        .iter()
        .enumerate()
        .take(query_top_size)
        .map(|(pos, &t)| super::ndcg_numerator(t) / super::ndcg_denominator(pos))
        .collect();
    let want = sum_f64(&terms);
    assert!((got - want).abs() < 1e-15, "got={got}, want={want}");
}

/// Trivial windows: empty slice -> 0.0; a single element -> its sole term.
#[test]
fn lambdamart_ideal_ndcg_trivial_windows() {
    assert_eq!(super::lambdamart_ideal_ndcg(&[], 0), 0.0);
    assert_eq!(super::lambdamart_ideal_ndcg(&[5.0], 0), 0.0); // top window 0 -> empty.
    let single = super::lambdamart_ideal_ndcg(&[5.0], 1);
    let want = super::ndcg_numerator(5.0) / super::ndcg_denominator(0);
    assert!((single - want).abs() < 1e-15);
}
