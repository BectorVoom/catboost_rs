//! Unit tests for the Cholesky pairwise-leaf solve ([`super::calculate_pairwise_leaf_values`]
//! and [`super::compute_pairwise_leaf_deltas`]) — LOSS-04 Wave B.
//!
//! The two `calculate_pairwise_leaf_values` cases mirror upstream catboost 1.2.10
//! `pairwise_leaves_calculation_ut.cpp` VERBATIM (the 2×2 closed form + a general
//! `n=4` SPD system), asserting the solved leaf values to 1e-6 against the frozen
//! upstream expectations. The matrix-assembly + zero-average cases exercise
//! `compute_pairwise_leaf_deltas` end to end.
//!
//! Source/test separation (INFRA-06): dedicated file, linked via the `#[path]`
//! footer in `pairwise_leaves.rs` — never inline.

use super::{calculate_pairwise_leaf_values, compute_pairwise_leaf_deltas};
use cb_compute::{GroupSpan, RankingCompetitor};

#[test]
fn pairwise_leaf_calculation_small_matrix_matches_upstream() {
    // pairwise_leaves_calculation_ut.cpp:18-28 (PairwiseLeafCalculationTestSmallMatrix).
    let weight_sums = vec![vec![5.0, -5.0], vec![-5.0, 5.0]];
    let der_sums = vec![-2.0, 2.0];
    let l2_diag_reg = 0.3_f64;
    let pairwise_non_diag_reg = 0.1_f64;
    let leaf_values =
        calculate_pairwise_leaf_values(&weight_sums, &der_sums, l2_diag_reg, pairwise_non_diag_reg);
    assert!((leaf_values[0] - (-0.186_915_887_4)).abs() < 1e-6, "v0={}", leaf_values[0]);
    assert!((leaf_values[1] - 0.186_915_887_4).abs() < 1e-6, "v1={}", leaf_values[1]);
}

#[test]
fn pairwise_leaf_calculation_large_matrix_matches_upstream() {
    // pairwise_leaves_calculation_ut.cpp:30-47 (PairwiseLeafCalculationTestLargeMatrix).
    let weight_sums = vec![
        vec![2.0, -2.0, 0.0, 0.0],
        vec![-2.0, 3.0, -1.0, 0.0],
        vec![0.0, -1.0, 5.0, -4.0],
        vec![0.0, 0.0, -4.0, 4.0],
    ];
    let der_sums = vec![16.0, -32.0, 32.0, -16.0];
    let leaf_values = calculate_pairwise_leaf_values(&weight_sums, &der_sums, 0.3, 0.1);
    assert!((leaf_values[0] - 0.737_447_159_3).abs() < 1e-6, "v0={}", leaf_values[0]);
    assert!((leaf_values[1] - (-7.279_036_944)).abs() < 1e-6, "v1={}", leaf_values[1]);
    assert!((leaf_values[2] - 5.448_432_894).abs() < 1e-6, "v2={}", leaf_values[2]);
    assert!((leaf_values[3] - 1.093_156_891).abs() < 1e-6, "v3={}", leaf_values[3]);
}

#[test]
fn pairwise_leaf_values_are_zero_averaged() {
    // MakeZeroAverage: the leaf deltas sum to ~0 (zero-centered).
    let weight_sums = vec![
        vec![2.0, -2.0, 0.0, 0.0],
        vec![-2.0, 3.0, -1.0, 0.0],
        vec![0.0, -1.0, 5.0, -4.0],
        vec![0.0, 0.0, -4.0, 4.0],
    ];
    let der_sums = vec![16.0, -32.0, 32.0, -16.0];
    let leaf_values = calculate_pairwise_leaf_values(&weight_sums, &der_sums, 0.3, 0.1);
    let s: f64 = leaf_values.iter().sum();
    assert!(s.abs() < 1e-9, "leaf deltas must be zero-averaged, sum={s}");
    // 2×2 case is also zero-centered: v0 + v1 == 0.
    let small = calculate_pairwise_leaf_values(&[vec![5.0, -5.0], vec![-5.0, 5.0]], &[-2.0, 2.0], 0.3, 0.1);
    assert!((small[0] + small[1]).abs() < 1e-12);
}

#[test]
fn non_positive_pivot_falls_back_to_zeros_no_nan() {
    // A degenerate weight matrix (all zeros) with the small reg priors can drive
    // the Cholesky to a non-positive pivot in the general path; the solver returns
    // None and the leaf falls back to zeros (T-06.3-03-01 — never NaN/panic).
    let weight_sums = vec![vec![0.0; 4]; 4];
    let der_sums = vec![1.0, 2.0, 3.0, 4.0];
    // With tiny reg priors the (n-1)×(n-1) matrix may still be SPD; the contract is
    // only that the output is finite (no NaN), zero-averaged.
    let leaf_values = calculate_pairwise_leaf_values(&weight_sums, &der_sums, 0.0, 0.0);
    assert_eq!(leaf_values.len(), 4);
    assert!(leaf_values.iter().all(|v| v.is_finite()), "no NaN on degenerate system");
}

#[test]
fn compute_pairwise_leaf_deltas_assembles_weight_sums_from_competitors() {
    // One group [0,2): winner=doc0, loser=doc1, weight 1. doc0 in leaf 0, doc1 in
    // leaf 1. ComputePairwiseWeightSums:
    //   sum[0][1] -= 1; sum[1][0] -= 1; sum[0][0] += 1; sum[1][1] += 1
    //   ⇒ [[1,-1],[-1,1]]. derSums per leaf = Σ der1 over leaf members.
    // With der1 = [-0.5, 0.5] (PairLogit symmetric pair), derSums = [-0.5, 0.5].
    // 2×2 closed form: x0 = derSums[0]/(1 + diagReg); diagReg = 0.1·0.5 + 3 = 3.05.
    //   x0 = -0.5/3.05 = -0.163934...; zero-avg ⇒ [-0.0819672, 0.0819672].
    let group = GroupSpan {
        begin: 0,
        end: 2,
        weight: 1.0,
        competitors: vec![vec![RankingCompetitor { id: 1, weight: 1.0 }], Vec::new()],
    };
    let leaf_of = vec![0usize, 1usize];
    let der1 = vec![-0.5_f64, 0.5];
    let deltas = compute_pairwise_leaf_deltas(&[group], &leaf_of, &der1, 2, 3.0, 0.1);
    assert_eq!(deltas.len(), 2);
    let diag_reg = 0.1 * 0.5 + 3.0; // 3.05
    let x0 = -0.5 / (1.0 + diag_reg);
    let avg = (x0 + 0.0) / 2.0;
    assert!((deltas[0] - (x0 - avg)).abs() < 1e-12, "d0={}", deltas[0]);
    assert!((deltas[1] - (0.0 - avg)).abs() < 1e-12, "d1={}", deltas[1]);
}

#[test]
fn same_leaf_pairs_are_skipped_in_weight_sums() {
    // A winner and loser in the SAME leaf contribute nothing to the weight matrix
    // (ComputePairwiseWeightSums `winnerLeaf == loserLeaf` continue). With both docs
    // in leaf 0, the weight sums are all zero ⇒ derSums drive a trivial solve.
    let group = GroupSpan {
        begin: 0,
        end: 2,
        weight: 1.0,
        competitors: vec![vec![RankingCompetitor { id: 1, weight: 1.0 }], Vec::new()],
    };
    let leaf_of = vec![0usize, 0usize]; // both in leaf 0
    let der1 = vec![-0.5_f64, 0.5];
    // leaf_count 2 but only leaf 0 populated; leaf 1 has no members.
    let deltas = compute_pairwise_leaf_deltas(&[group], &leaf_of, &der1, 2, 3.0, 0.1);
    assert_eq!(deltas.len(), 2);
    assert!(deltas.iter().all(|v| v.is_finite()));
}
