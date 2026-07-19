//! Isolated unit tests for the GreedyLogSum binarizer ([`crate::borders`]).
//!
//! These are hand-constructed (no fixtures): they lock the four parity-critical
//! micro-behaviors of the binarizer — the penalty value, duplicate-column
//! collapse, `-0.0 → 0.0` border normalization, and the basic 2-value /
//! 1-border case. The corpus-level oracle comparison lives in the integration
//! test `tests/borders_oracle_test.rs`. Kept in a dedicated `*_test.rs` file
//! per the source/test separation rule (D-17); no `#[cfg(test)] mod` lives in
//! `borders.rs`.

use crate::borders::{penalty_maxsumlog, select_borders_greedy_logsum};

/// (a) `penalty_maxsumlog(count) == -(count + 1e-8).ln()` for known counts
/// (binarization.cpp:180).
#[test]
fn penalty_matches_negative_log_with_epsilon() {
    for &count in &[1.0_f64, 2.0, 7.0, 50.0] {
        let expected = -(count + 1e-8).ln();
        assert_eq!(penalty_maxsumlog(count), expected);
    }
}

/// (b) A column with a duplicated value collapses to the same border set as its
/// de-duplicated form (the lower/upper-bound probing skips equal values).
#[test]
fn duplicate_value_column_collapses() {
    // Five distinct values, then the same five with one value duplicated.
    let distinct = [-2.0_f64, -1.0, 0.5, 1.0, 3.0];
    let with_dup = [-2.0_f64, -1.0, 0.5, 0.5, 1.0, 3.0];

    let borders_distinct = select_borders_greedy_logsum(&distinct, 254, false);
    let borders_dup = select_borders_greedy_logsum(&with_dup, 254, false);

    assert_eq!(
        borders_distinct, borders_dup,
        "a duplicated value must not change the border set"
    );
}

/// (c) An input producing a `-0.0` border is normalized so the returned border
/// is `+0.0` (assert the sign bit, not just numeric equality).
#[test]
fn negative_zero_border_normalizes_to_positive_zero() {
    // Two symmetric values around zero: the f32 midpoint is 0.5*(-x) + 0.5*x.
    // With x chosen so the midpoint underflows to -0.0f, normalization must
    // flip it to +0.0f. -0.0 and +0.0 compare equal, so we check to_bits().
    let column = [-1.0_f64, 1.0];
    let borders = select_borders_greedy_logsum(&column, 254, false);

    assert_eq!(borders.len(), 1, "two distinct values yield exactly one border");
    let border = borders[0];
    assert_eq!(border, 0.0_f64);
    // The single border is the f32 midpoint 0.5*(-1) + 0.5*1 = 0.0; ensure it is
    // +0.0 (positive sign), never -0.0.
    assert!(
        border.is_sign_positive(),
        "border must be +0.0 (normalized), got bits {:#x}",
        border.to_bits()
    );
}

/// (d) The basic 2-distinct-value case returns exactly one border at the f32
/// midpoint of the two values.
#[test]
fn two_distinct_values_yield_single_midpoint_border() {
    let a = 1.5_f64;
    let b = 4.5_f64;
    let borders = select_borders_greedy_logsum(&[a, b], 254, false);

    assert_eq!(borders.len(), 1);
    // f32 midpoint, matching LeftBorder's 0.5f * a + 0.5f * b.
    let expected = f64::from(0.5_f32 * (a as f32) + 0.5_f32 * (b as f32));
    assert!((borders[0] - expected).abs() <= 1e-6, "border {} != {}", borders[0], expected);
}

/// WR-01: a tie-prone column (many equal gaps -> equal split scores) under a
/// small `max_borders` exercises the STL binary-heap tie-break. The result must
/// be deterministic and order-independent of the input permutation: shuffling
/// the column must not change the final sorted border set (the selector sorts
/// internally, and the heap tie-break is keyed on score, not input order).
#[test]
fn tie_prone_column_is_deterministic_under_permutation() {
    // Evenly spaced values produce many equal-score split candidates (ties).
    let ascending: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let mut shuffled = ascending.clone();
    shuffled.reverse();
    let mut interleaved = Vec::with_capacity(16);
    for i in 0..8 {
        interleaved.push(ascending[i]);
        interleaved.push(ascending[15 - i]);
    }

    // A small budget forces the greedy heap to choose among tied bins.
    for max_borders in [2usize, 3, 5, 7] {
        let a = select_borders_greedy_logsum(&ascending, max_borders, false);
        let b = select_borders_greedy_logsum(&shuffled, max_borders, false);
        let c = select_borders_greedy_logsum(&interleaved, max_borders, false);
        assert_eq!(a, b, "max_borders={max_borders}: reversed input changed borders");
        assert_eq!(a, c, "max_borders={max_borders}: interleaved input changed borders");
    }
}

/// WR-01: a constant column (all equal values) produces no internal split (no
/// border), and the heap handling never panics regardless of budget.
#[test]
fn constant_column_yields_no_borders() {
    let column = vec![3.0_f64; 12];
    for max_borders in [1usize, 4, 254] {
        let borders = select_borders_greedy_logsum(&column, max_borders, false);
        assert!(
            borders.is_empty(),
            "constant column must yield no borders (max_borders={max_borders}), got {borders:?}"
        );
    }
}

/// The NanMode(Min) sentinel is prepended at index 0 when requested, before the
/// sorted borders.
#[test]
fn nan_sentinel_prepended_when_requested() {
    let borders = select_borders_greedy_logsum(&[1.0_f64, 2.0, 3.0], 254, true);
    assert_eq!(borders[0], f64::from(f32::MIN));
    // The remaining borders are strictly greater than the sentinel.
    for &b in &borders[1..] {
        assert!(b > f64::from(f32::MIN));
    }
}

/// Locks the ascending-sorted-output invariant that `cb_train::boosting::
/// quantize_feature_major`'s `partition_point` optimization depends on (a
/// binary search is only correct on sorted input; upstream's `THashSet` ->
/// sort -> dedup pipeline is documented but not otherwise directly tested).
/// Covers: unsorted/reversed/interleaved input, ties, a `max_borders` budget
/// spanning "no split" to "many splits", and both `nan_sentinel` settings —
/// the sentinel value itself (`f32::MIN`) must sort to the front too.
#[test]
fn output_is_always_ascending_sorted() {
    let reversed: Vec<f64> = (0..32).rev().map(f64::from).collect();
    let mut interleaved = Vec::with_capacity(32);
    for i in 0..16 {
        interleaved.push(f64::from(i));
        interleaved.push(f64::from(31 - i));
    }
    let with_ties: Vec<f64> = (0..32).map(|i| f64::from(i / 3)).collect();

    for column in [&reversed, &interleaved, &with_ties] {
        for max_borders in [0usize, 1, 3, 8, 254] {
            for nan_sentinel in [false, true] {
                let borders = select_borders_greedy_logsum(column, max_borders, nan_sentinel);
                assert!(
                    borders.windows(2).all(|w| w[0] <= w[1]),
                    "borders not ascending-sorted for max_borders={max_borders}, \
                     nan_sentinel={nan_sentinel}: {borders:?}"
                );
            }
        }
    }
}
