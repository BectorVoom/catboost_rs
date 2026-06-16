//! Unit tests for the L2 split-score calcer (`AddLeafPlain`) and the
//! `random_strength` perturbation (`TRandomScore::GetInstance`, `CalcScoreStDev`).

use cb_core::TFastRng64;

use crate::histogram::LeafStats;
use crate::runtime::EScoreFunction;
use crate::score::{
    add_leaf_plain, cosine_split_score, derivatives_std_dev_from_zero, l2_split_score,
    multi_dim_split_score, random_score_instance, score_st_dev, MINIMAL_SCORE,
};

#[test]
fn multi_dim_split_score_dim1_equals_scalar_score() {
    // dim=1 byte-identity (D-04 anchor): with ONE dimension's leaf stats, the
    // single-shared-accumulator score is bit-for-bit the scalar split score, for
    // BOTH Cosine and L2.
    let leaves = vec![
        LeafStats {
            sum_weighted_delta: 4.0,
            sum_weight: 3.0,
        },
        LeafStats {
            sum_weighted_delta: -2.5,
            sum_weight: 2.0,
        },
        LeafStats {
            sum_weighted_delta: 1.0,
            sum_weight: 1.0,
        },
    ];
    let scaled_l2 = 3.0;
    let per_dim = vec![leaves.clone()];
    let cos_multi = multi_dim_split_score(EScoreFunction::Cosine, &per_dim, scaled_l2);
    let cos_scalar = cosine_split_score(&leaves, scaled_l2);
    assert_eq!(
        cos_multi.to_bits(),
        cos_scalar.to_bits(),
        "Cosine dim=1 multi-dim score must be BIT-identical to the scalar score"
    );
    let l2_multi = multi_dim_split_score(EScoreFunction::L2, &per_dim, scaled_l2);
    let l2_scalar = l2_split_score(&leaves, scaled_l2);
    assert_eq!(
        l2_multi.to_bits(),
        l2_scalar.to_bits(),
        "L2 dim=1 multi-dim score must be BIT-identical to the scalar score"
    );
}

#[test]
fn multi_dim_split_score_couples_dimensions_inside_sqrt() {
    // Cosine couples num/den INSIDE the sqrt: it is NOT the sum of per-dimension
    // cosine scores. Build two distinct dimensions and assert the coupled form
    // num/sqrt(den) differs from Σ_d cosine_d (the wrong reduction).
    let dim0 = vec![
        LeafStats {
            sum_weighted_delta: 4.0,
            sum_weight: 3.0,
        },
        LeafStats {
            sum_weighted_delta: -1.0,
            sum_weight: 2.0,
        },
    ];
    let dim1 = vec![
        LeafStats {
            sum_weighted_delta: 2.0,
            sum_weight: 1.0,
        },
        LeafStats {
            sum_weighted_delta: -3.0,
            sum_weight: 4.0,
        },
    ];
    let scaled_l2 = 3.0;
    let coupled = multi_dim_split_score(
        EScoreFunction::Cosine,
        &[dim0.clone(), dim1.clone()],
        scaled_l2,
    );
    let sum_of_cosines =
        cosine_split_score(&dim0, scaled_l2) + cosine_split_score(&dim1, scaled_l2);
    assert!(
        (coupled - sum_of_cosines).abs() > 1e-9,
        "coupled num/sqrt(den) must differ from the (wrong) Σ_d cosine_d reduction"
    );
}

#[test]
fn minimal_score_is_neg_infinity() {
    assert_eq!(MINIMAL_SCORE, f64::NEG_INFINITY);
    assert!(0.0 > MINIMAL_SCORE);
    assert!(-1e300 > MINIMAL_SCORE);
}

#[test]
fn add_leaf_plain_is_avg_times_sum_delta() {
    // sum_weighted_delta = 10.0, sum_weight = 4.0, scaledL2 = 3.0
    // avg = 10/(4+3) = 10/7; term = avg * 10 = 100/7
    let stats = LeafStats {
        sum_weighted_delta: 10.0,
        sum_weight: 4.0,
    };
    let term = add_leaf_plain(stats, 3.0);
    assert!((term - 100.0 / 7.0).abs() < 1e-12);
}

#[test]
fn add_leaf_plain_empty_leaf_is_zero() {
    let stats = LeafStats::default();
    assert_eq!(add_leaf_plain(stats, 3.0), 0.0);
}

#[test]
fn l2_split_score_hand_computed_bucket() {
    // Two leaves; total score = sum of per-leaf avg*sumDelta.
    // leaf A: delta=6.0, w=3.0, scaledL2=3.0 -> avg=6/6=1.0 -> 1.0*6 = 6.0
    // leaf B: delta=4.0, w=1.0, scaledL2=3.0 -> avg=4/4=1.0 -> 1.0*4 = 4.0
    let leaves = [
        LeafStats {
            sum_weighted_delta: 6.0,
            sum_weight: 3.0,
        },
        LeafStats {
            sum_weighted_delta: 4.0,
            sum_weight: 1.0,
        },
    ];
    let score = l2_split_score(&leaves, 3.0);
    assert!((score - 10.0).abs() < 1e-12);
}

#[test]
fn derivatives_std_dev_from_zero_is_rms_of_weighted_ders() {
    // Plain boosting: sqrt(sum(wd^2)/n) (CalcDerivativesStDevFromZeroPlainBoosting).
    let wd = [1.0, -2.0, 3.0, -0.5];
    let got = derivatives_std_dev_from_zero(&wd);
    assert!(
        (got - 1.887_458_608_817_687_5).abs() < 1e-13,
        "dsdz mismatch: {got}"
    );
}

#[test]
fn derivatives_std_dev_from_zero_empty_is_zero() {
    // Guarded: an empty derivative vector yields 0.0 (no divide-by-zero).
    assert_eq!(derivatives_std_dev_from_zero(&[]), 0.0);
}

#[test]
fn score_st_dev_applies_model_size_multiplier() {
    // scoreStDev = random_strength * dsdz * modelLeft/(1+modelLeft),
    // modelLeft = exp(ln(n) - modelLength). n=4, modelLength=0.2, rs=1.0.
    let wd = [1.0, -2.0, 3.0, -0.5];
    let got = score_st_dev(1.0, &wd, 0.2);
    assert!(
        (got - 1.445_939_871_899_679_9).abs() < 1e-13,
        "scoreStDev mismatch: {got}"
    );
}

#[test]
fn score_st_dev_first_tree_multiplier() {
    // modelLength=0 -> modelLeft = n -> mult = n/(1+n) = 4/5 = 0.8.
    let wd = [1.0, -2.0, 3.0, -0.5];
    let dsdz = derivatives_std_dev_from_zero(&wd);
    let got = score_st_dev(1.0, &wd, 0.0);
    assert!((got - dsdz * 0.8).abs() < 1e-13, "first-tree scoreStDev: {got}");
}

#[test]
fn score_st_dev_zero_random_strength_is_zero() {
    // random_strength=0 -> no perturbation magnitude (first-slice behaviour).
    let wd = [1.0, -2.0, 3.0, -0.5];
    assert_eq!(score_st_dev(0.0, &wd, 0.2), 0.0);
}

// CR-01 CONTRACT (isolated RED->GREEN at the unit boundary): upstream
// `CalcDerivativesStDevFromZeroPlainBoosting` (greedy_tensor_search.cpp:92-107)
// reads `fold.BodyTailArr.front().WeightedDerivatives` — the FULL, un-sampled
// fold. The boosting score path historically fed the control-MASKED vector
// (`score_weighted_der1`, control-false entries ZEROED but length preserved)
// into `score_st_dev`. Zeroing entries shrinks the RMS numerator while `n`
// (the denominator AND the `ln(n)` model-size multiplier) stays the same, so
// the masked input yields a STRICTLY LOWER scoreStDev than the full fold
// whenever any object is dropped. This is exactly the CR-01 parity break that
// surfaces when `random_strength != 0` is combined with `bootstrap_type != No`.
// The end-to-end first-tree oracle (`regularization_oracle_random_strength_
// bernoulli`) cannot isolate this on the `numeric_tiny` corpus (the std-dev
// difference is entangled with the variable-length Box-Muller draw-stream
// residual and never flips a tree-0 split there); this unit test locks the
// contract at the boundary where the bug IS observable. See SUMMARY "Deviations".
#[test]
fn score_st_dev_masked_vector_biases_low_vs_full_fold_cr01() {
    // Full fold: 4 objects with non-trivial derivatives.
    let full = [1.0, -2.0, 3.0, -0.5];
    // Masked fold: same length (n=4) but two control-false entries zeroed —
    // the shape the buggy boosting score path passed in.
    let masked = [1.0, 0.0, 3.0, 0.0];

    let dsdz_full = derivatives_std_dev_from_zero(&full);
    let dsdz_masked = derivatives_std_dev_from_zero(&masked);

    // The masked numerator under-counts (zeroed entries contribute 0) while the
    // denominator n=4 is unchanged -> masked std-dev is strictly LOWER.
    assert!(
        dsdz_masked < dsdz_full,
        "masked dsdz ({dsdz_masked}) must be < full-fold dsdz ({dsdz_full}) — \
         this is the CR-01 bias the &weighted_der1 fix removes"
    );

    // Concretely: full = sqrt((1+4+9+0.25)/4) = sqrt(3.5625); masked =
    // sqrt((1+0+9+0)/4) = sqrt(2.5). Both divide by the SAME n=4.
    assert!((dsdz_full - 3.562_5_f64.sqrt()).abs() < 1e-13, "full: {dsdz_full}");
    assert!((dsdz_masked - 2.5_f64.sqrt()).abs() < 1e-13, "masked: {dsdz_masked}");

    // The model-size multiplier is identical for both (same n, same modelLength),
    // so the scoreStDev difference is carried entirely by the numerator: the
    // FULL fold is the upstream-correct input.
    let model_length = 0.0; // first tree
    let sd_full = score_st_dev(1.0, &full, model_length);
    let sd_masked = score_st_dev(1.0, &masked, model_length);
    assert!(
        sd_masked < sd_full,
        "masked scoreStDev ({sd_masked}) must be < full scoreStDev ({sd_full})"
    );
}

#[test]
fn random_score_instance_adds_normal_times_stdev() {
    // GetInstance(Normal) = Val + StdNormalDistribution(rand) * StDev.
    // seed=42 first std_normal == 0.196_927_155_406_922_8 (cb_core reference).
    let mut rng = TFastRng64::from_seed(42);
    let got = random_score_instance(5.0, 2.0, &mut rng);
    let expected = 5.0 + 0.196_927_155_406_922_8 * 2.0;
    assert!((got - expected).abs() < 1e-13, "instance mismatch: {got}");
}

#[test]
fn random_score_instance_zero_stdev_returns_val_but_still_draws() {
    // Even at StDev=0 upstream still CALLS StdNormalDistribution (the draw is
    // consumed); the product is 0 so the result is exactly Val. The draw MUST
    // advance the RNG so downstream draw order stays aligned.
    let mut rng = TFastRng64::from_seed(7);
    let mut probe = TFastRng64::from_seed(7);
    let got = random_score_instance(3.5, 0.0, &mut rng);
    assert_eq!(got, 3.5, "zero-stdev instance must equal Val");
    // The RNG advanced: drawing a normal from the probe leaves it at the same
    // state as `rng` (both consumed exactly one std_normal).
    assert_eq!(rng.gen_rand(), {
        let _ = cb_core::std_normal(&mut probe);
        probe.gen_rand()
    });
}

/// Build a 4-leaf candidate from `(sum_weighted_delta, sum_weight)` pairs.
fn leaves(stats: [(f64, f64); 4]) -> Vec<LeafStats> {
    stats
        .iter()
        .map(|&(sum_weighted_delta, sum_weight)| LeafStats {
            sum_weighted_delta,
            sum_weight,
        })
        .collect()
}

#[test]
fn cosine_picks_upstream_border_where_l2_diverges() {
    // The pc=1 tensor_ctr_e2e tree-0 second split under the S-order structure CTR
    // (plan 05-19). der1 = y-0.5, scaled_l2 = l2*(sumW/n) = 3*(30/30) = 3.
    let scaled_l2 = 3.0;
    let border2 = leaves([(-3.0, 6.0), (0.0, 0.0), (-0.5, 11.0), (5.5, 13.0)]);
    let border3 = leaves([(-3.5, 9.0), (0.0, 0.0), (0.0, 8.0), (5.5, 13.0)]);
    // L2 ranks border 3 ABOVE border 2 (the latent-bug behaviour cb-train shipped).
    assert!(
        l2_split_score(&border3, scaled_l2) > l2_split_score(&border2, scaled_l2),
        "L2 (incorrectly for the default config) prefers border 3"
    );
    // Cosine (catboost default) ranks border 2 ABOVE border 3 — matching upstream.
    assert!(
        cosine_split_score(&border2, scaled_l2) > cosine_split_score(&border3, scaled_l2),
        "Cosine must prefer border 2 (the upstream split), reversing L2's choice"
    );
}

#[test]
fn cosine_is_l2_numerator_over_sqrt_denominator() {
    let scaled_l2 = 3.0;
    let cand = leaves([(-3.0, 6.0), (1.0, 2.0), (-0.5, 11.0), (5.5, 13.0)]);
    let num = l2_split_score(&cand, scaled_l2);
    let mut den = 1e-100;
    for s in &cand {
        let avg = s.sum_weighted_delta / (s.sum_weight + scaled_l2);
        den += avg * avg * s.sum_weight;
    }
    let expected = num / den.sqrt();
    let got = cosine_split_score(&cand, scaled_l2);
    assert!(
        (got - expected).abs() <= 1e-12,
        "cosine == DP/sqrt(D2): got {got}, expected {expected}"
    );
}
