//! Self-oracle unit tests for the pairwise split-scoring subsystem
//! (`pairwise_scoring.rs`). Source/test separation per INFRA-06 (CLAUDE.md): no
//! inline `#[cfg(test)]` body in the production file — this sibling file is linked
//! via the `#[path = "pairwise_scoring_test.rs"] mod tests;` footer.
//!
//! Each test hand-derives a small reference matrix and asserts the transcription
//! reproduces it bit-for-bit (der-sums / weight statistics) or to ≤1e-9 (the
//! Cholesky-backed per-split score).

use super::*;

/// Fixture geometry shared across the der-sum / weight-statistics tests:
/// 3 docs, 2 leaves, 3 buckets.
///   doc0 → leaf 0, bucket 1
///   doc1 → leaf 1, bucket 0
///   doc2 → leaf 0, bucket 2
const LEAF_OF: [usize; 3] = [0, 1, 0];
const BUCKET_OF: [usize; 3] = [1, 0, 2];
const LEAF_COUNT: usize = 2;
const BUCKET_COUNT: usize = 3;

#[test]
fn der_sums_scatter_matches_reference() {
    // weighted_der per doc.
    let weighted_der = [0.5_f64, -0.3, 0.2];

    let got = compute_der_sums(&weighted_der, LEAF_COUNT, BUCKET_COUNT, &LEAF_OF, &BUCKET_OF)
        .expect("der sums");

    // Hand-derived reference:
    //   der_sums[0][1] += 0.5 (doc0), der_sums[0][2] += 0.2 (doc2)
    //   der_sums[1][0] += -0.3 (doc1)
    let expected = vec![vec![0.0, 0.5, 0.2], vec![-0.3, 0.0, 0.0]];
    assert_eq!(got, expected, "der-sum scatter must match hand-derived matrix");
}

#[test]
fn der_sums_rejects_out_of_range_index() {
    let weighted_der = [1.0_f64];
    // leaf index 5 >= leaf_count 2 → typed error, never a panic.
    let bad_leaf = [5_usize];
    let bucket = [0_usize];
    let err = compute_der_sums(&weighted_der, LEAF_COUNT, BUCKET_COUNT, &bad_leaf, &bucket)
        .expect_err("out-of-range leaf must error");
    matches!(err, cb_core::CbError::OutOfRange(_))
        .then_some(())
        .expect("expected OutOfRange");
}

#[test]
fn pair_weight_statistics_winner_loser_bucket_order() {
    // Pairs (winner_doc, loser_doc, weight) covering both winnerBucket>loserBucket
    // and winnerBucket<loserBucket branches plus a winner==loser skip.
    let pairs = [
        (0usize, 1usize, 1.0f64), // winnerBucket 1 > loserBucket 0  -> branch A
        (2, 1, 2.0),              // winnerBucket 2 > loserBucket 0  -> branch A
        (1, 0, 0.5),              // winnerBucket 0 < loserBucket 1  -> else branch
        (0, 0, 9.0),              // winner == loser                 -> skip
    ];

    let stats =
        compute_pair_weight_statistics(&pairs, LEAF_COUNT, BUCKET_COUNT, &LEAF_OF, &BUCKET_OF)
            .expect("weight statistics");

    // Hand-derived reference: every non-skipped pair lands in stats[1][0]
    // (loserLeaf=1,winnerLeaf=0 for branch A; winnerLeaf=1,loserLeaf=0 for else).
    //   bucket0.smaller = -1.0 -2.0 -0.5 = -3.5
    //   bucket1.greater = -1.0 (pair0,1) -0.5 (pair1,0) = -1.5
    //   bucket2.greater = -2.0 (pair2,1)
    let s10 = &stats[1][0];
    assert!((s10[0].smaller_border_weight_sum - (-3.5)).abs() < 1e-12);
    assert!((s10[0].greater_border_right_weight_sum - 0.0).abs() < 1e-12);
    assert!((s10[1].smaller_border_weight_sum - 0.0).abs() < 1e-12);
    assert!((s10[1].greater_border_right_weight_sum - (-1.5)).abs() < 1e-12);
    assert!((s10[2].smaller_border_weight_sum - 0.0).abs() < 1e-12);
    assert!((s10[2].greater_border_right_weight_sum - (-2.0)).abs() < 1e-12);

    // The winner==loser pair (0,0,9.0) must NOT have mutated stats[0][*].
    for leaf_b in 0..LEAF_COUNT {
        for b in 0..BUCKET_COUNT {
            let c = &stats[0][leaf_b][b];
            assert_eq!(c.smaller_border_weight_sum, 0.0);
            assert_eq!(c.greater_border_right_weight_sum, 0.0);
        }
    }
}

#[test]
fn pair_weight_statistics_rejects_out_of_range_doc() {
    // Pair references doc 9 which is out of range for the 3-element index arrays.
    let pairs = [(0usize, 9usize, 1.0f64)];
    let err =
        compute_pair_weight_statistics(&pairs, LEAF_COUNT, BUCKET_COUNT, &LEAF_OF, &BUCKET_OF)
            .expect_err("out-of-range loser doc must error");
    matches!(err, cb_core::CbError::OutOfRange(_))
        .then_some(())
        .expect("expected OutOfRange");
}
