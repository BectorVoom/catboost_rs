//! Unit tests for the TFold body/tail prefix state machine and multi-permutation
//! fold creation ([`crate::fold`]).
//!
//! Test names embed `fold_prefix` so `cargo test -p cb-train fold_prefix`
//! selects the boundary lock (the plan's verify command), alongside the
//! fold-count and fold-creation locks.

use super::{
    body_sum_weights, body_tail_boundaries, body_tail_segments, create_folds, learning_fold_count,
    plain_fold_body_tail, select_min_batch_size, select_tail_size,
};

const MULT: f64 = 2.0;

#[test]
fn select_min_batch_size_boundary() {
    // n > 500 ? min(100, n/50) : 1.
    assert_eq!(select_min_batch_size(1), 1);
    assert_eq!(select_min_batch_size(30), 1);
    assert_eq!(select_min_batch_size(500), 1); // not > 500
    assert_eq!(select_min_batch_size(501), 501 / 50); // = 10
    assert_eq!(select_min_batch_size(2500), 50); // n/50 < 100
    assert_eq!(select_min_batch_size(10_000), 100); // capped at 100
}

#[test]
fn select_tail_size_is_ceil_times_multiplier() {
    assert_eq!(select_tail_size(1, 2.0), 2);
    assert_eq!(select_tail_size(16, 2.0), 32);
    // ceil semantics: 3 * 1.5 = 4.5 -> 5.
    assert_eq!(select_tail_size(3, 1.5), 5);
    // ceil of an exact integer product stays put.
    assert_eq!(select_tail_size(4, 2.0), 8);
}

#[test]
fn fold_prefix_boundaries_small_n30() {
    // The committed ordered_boost/body_tail_boundaries.npy is [1 2 4 8 16 30].
    assert_eq!(body_tail_boundaries(30, MULT), vec![1, 2, 4, 8, 16, 30]);
}

#[test]
fn fold_prefix_boundaries_n1_single_segment() {
    // SelectMinBatchSize(1) == 1 == n; one segment, final == n.
    assert_eq!(body_tail_boundaries(1, MULT), vec![1]);
}

#[test]
fn fold_prefix_boundaries_empty_for_n0() {
    assert!(body_tail_boundaries(0, MULT).is_empty());
}

#[test]
fn fold_prefix_boundaries_large_n_over_500() {
    // n = 600 > 500: SelectMinBatchSize = min(100, 600/50) = 12.
    // leftPartLen: 12 -> 24 -> 48 -> 96 -> 192 -> 384 -> ceil(768) capped at 600.
    let b = body_tail_boundaries(600, MULT);
    assert_eq!(b.first().copied(), Some(12));
    assert_eq!(b.last().copied(), Some(600));
    assert_eq!(b, vec![12, 24, 48, 96, 192, 384, 600]);
}

#[test]
fn fold_prefix_boundaries_final_always_equals_n() {
    for &n in &[1usize, 2, 5, 7, 30, 100, 999] {
        let b = body_tail_boundaries(n, MULT);
        assert_eq!(b.last().copied(), Some(n), "final boundary must equal n={n}");
    }
}

#[test]
fn fold_prefix_segments_pair_consecutive_boundaries() {
    // segments = windows-of-2 over the boundary sequence.
    assert_eq!(
        body_tail_segments(30, MULT),
        vec![(1, 2), (2, 4), (4, 8), (8, 16), (16, 30)]
    );
    // A single-boundary fold (n=1) has no growing segment.
    assert!(body_tail_segments(1, MULT).is_empty());
}

#[test]
fn plain_fold_is_single_full_span() {
    assert_eq!(plain_fold_body_tail(30), (30, 30));
    assert_eq!(plain_fold_body_tail(0), (0, 0));
}

#[test]
fn learning_fold_count_for_permutation_count_1_2_4() {
    // permutation needed: max(1, pc - 1).
    assert_eq!(learning_fold_count(1, true), 1); // max(1, 0)
    assert_eq!(learning_fold_count(2, true), 1); // max(1, 1) -> 1 learning + 1 avg
    assert_eq!(learning_fold_count(4, true), 3); // max(1, 3)
    // permutation NOT needed (plain numeric / one-hot): always 1.
    assert_eq!(learning_fold_count(1, false), 1);
    assert_eq!(learning_fold_count(4, false), 1);
}

#[test]
fn create_folds_count_is_learning_plus_one_averaging() {
    // permutation_count = 2, needed: 1 learning + 1 averaging = 2 folds.
    let folds = create_folds(30, 2, true, true, MULT, 0);
    assert_eq!(folds.len(), 2);
    assert_eq!(folds.iter().filter(|f| f.is_averaging).count(), 1);
    assert_eq!(folds.iter().filter(|f| !f.is_averaging).count(), 1);
    // The LAST fold is the averaging fold.
    assert!(folds.last().map(|f| f.is_averaging).unwrap_or(false));

    // permutation_count = 4, needed: 3 learning + 1 averaging = 4 folds.
    let folds4 = create_folds(30, 4, true, true, MULT, 0);
    assert_eq!(folds4.len(), 4);
    assert_eq!(folds4.iter().filter(|f| f.is_averaging).count(), 1);
}

#[test]
fn create_folds_averaging_uses_plain_span_learning_uses_dynamic() {
    let folds = create_folds(30, 2, true, true, MULT, 0);
    // Learning fold (idx 0): dynamic growing body/tail.
    let learning = folds.iter().find(|f| !f.is_averaging).unwrap();
    assert_eq!(learning.body_tail_boundaries, vec![1, 2, 4, 8, 16, 30]);
    // Averaging fold: single full span [n].
    let averaging = folds.iter().find(|f| f.is_averaging).unwrap();
    assert_eq!(averaging.body_tail_boundaries, vec![30]);
}

#[test]
fn create_folds_plain_path_all_single_span() {
    // dynamic_body_tail = false (plain / one-hot path): every fold is full span.
    let folds = create_folds(30, 1, false, false, MULT, 0);
    for f in &folds {
        assert_eq!(f.body_tail_boundaries, vec![30]);
    }
}

#[test]
fn create_folds_permutations_drawn_in_continuous_order() {
    use crate::permutations;
    // The fold permutations must equal the continuous-stream draws (learning
    // folds first, then averaging) — not per-fold reseeds.
    let folds = create_folds(16, 2, true, true, MULT, 99);
    let expected = permutations(16, folds.len(), 99);
    let got: Vec<Vec<i32>> = folds.iter().map(|f| f.permutation.clone()).collect();
    assert_eq!(got, expected);
}

#[test]
fn body_sum_weights_unweighted_equals_body_finish() {
    // Unweighted: each segment's body weight == its body_finish.
    let w = body_sum_weights(30, MULT, &[]);
    assert_eq!(w, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
}

#[test]
fn body_sum_weights_weighted_sums_body_prefix() {
    let weights: Vec<f64> = (0..30).map(|i| (i as f64) + 1.0).collect();
    let w = body_sum_weights(30, MULT, &weights);
    // First segment body_finish = 1 -> sum of first weight = 1.0.
    assert_eq!(w.first().copied(), Some(1.0));
    // Second segment body_finish = 2 -> 1 + 2 = 3.0.
    assert_eq!(w.get(1).copied(), Some(3.0));
    // Fourth segment body_finish = 8 -> sum 1..=8 = 36.0.
    assert_eq!(w.get(3).copied(), Some(36.0));
}
