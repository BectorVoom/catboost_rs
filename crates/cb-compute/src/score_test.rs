//! Unit tests for the L2 split-score calcer (`AddLeafPlain`).

use crate::histogram::LeafStats;
use crate::score::{add_leaf_plain, l2_split_score, MINIMAL_SCORE};

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
