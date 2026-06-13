//! Unit tests for the whole-set (Plain-mode) CTR accumulation
//! ([`crate::ctr::online`]).

use crate::ctr::online::{accumulate_online, TCtrHistory, TCtrMeanHistory};

/// A small binclf categorical column: three distinct values, mixed classes.
/// `a` appears 3x (classes 1,1,0), `b` 2x (classes 0,1), `c` 1x (class 1).
fn small_column() -> (Vec<&'static str>, Vec<usize>, Vec<f64>) {
    let column = vec!["a", "a", "b", "a", "b", "c"];
    let target_class = vec![1, 1, 0, 0, 1, 1];
    let target = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    (column, target_class, target)
}

#[test]
fn whole_set_class_counts_are_exact_integers() {
    let (col, tc, t) = small_column();
    let acc = accumulate_online(&col, &tc, &t, 2, 1).expect("accumulate");
    assert_eq!(acc.bucket_count, 3, "three distinct categories -> 3 buckets");
    // Bucket 0 = "a" (first-seen): classes [1,1,0] -> N[0]=1, N[1]=2.
    assert_eq!(acc.class_histories[0].n, vec![1, 2]);
    // Bucket 1 = "b": classes [0,1] -> N[0]=1, N[1]=1.
    assert_eq!(acc.class_histories[1].n, vec![1, 1]);
    // Bucket 2 = "c": class [1] -> N[0]=0, N[1]=1.
    assert_eq!(acc.class_histories[2].n, vec![0, 1]);
}

#[test]
fn totals_match_per_bucket_document_counts() {
    let (col, tc, t) = small_column();
    let acc = accumulate_online(&col, &tc, &t, 2, 1).expect("accumulate");
    // total_counts is the Counter/FeatureFreq numerator: per-bucket doc count.
    assert_eq!(acc.total_counts, vec![3, 2, 1]);
    // TCtrHistory::total agrees with the explicit total.
    assert_eq!(acc.class_histories[0].total(), 3);
    assert_eq!(acc.class_histories[1].total(), 2);
    assert_eq!(acc.class_histories[2].total(), 1);
}

#[test]
fn binarized_mean_divides_by_target_border_count() {
    let (col, tc, t) = small_column();
    // target_border_count = 2: BinarizedTargetMeanValue adds class/2.
    let acc = accumulate_online(&col, &tc, &t, 2, 2).expect("accumulate");
    // Bucket "a": classes 1,1,0 -> (0.5 + 0.5 + 0.0) = 1.0 over count 3.
    assert!((acc.binarized_mean[0].sum - 1.0).abs() < 1e-6);
    assert_eq!(acc.binarized_mean[0].count, 3);
    // Bucket "b": classes 0,1 -> (0.0 + 0.5) = 0.5 over count 2.
    assert!((acc.binarized_mean[1].sum - 0.5).abs() < 1e-6);
    assert_eq!(acc.binarized_mean[1].count, 2);
}

#[test]
fn float_mean_adds_raw_target() {
    let column = vec!["x", "x", "y"];
    let target_class = vec![0, 1, 1];
    let target = vec![2.5, 3.5, 10.0];
    let acc = accumulate_online(&column, &target_class, &target, 2, 1).expect("accumulate");
    // Bucket "x": raw targets 2.5 + 3.5 = 6.0 over count 2.
    assert!((acc.float_mean[0].sum - 6.0).abs() < 1e-6);
    assert_eq!(acc.float_mean[0].count, 2);
    // Bucket "y": raw target 10.0 over count 1.
    assert!((acc.float_mean[1].sum - 10.0).abs() < 1e-6);
    assert_eq!(acc.float_mean[1].count, 1);
}

#[test]
fn length_mismatch_is_typed_error_not_panic() {
    let column = vec!["a", "b"];
    let target_class = vec![0]; // wrong length
    let target = vec![0.0, 1.0];
    assert!(accumulate_online(&column, &target_class, &target, 2, 1).is_err());
}

#[test]
fn zero_target_border_count_is_typed_error() {
    let (col, tc, t) = small_column();
    assert!(accumulate_online(&col, &tc, &t, 2, 0).is_err());
}

#[test]
fn empty_column_yields_zero_buckets() {
    let acc = accumulate_online(&[], &[], &[], 2, 1).expect("empty accumulate");
    assert_eq!(acc.bucket_count, 0);
    assert!(acc.class_histories.is_empty());
    assert!(acc.total_counts.is_empty());
}

#[test]
fn ctr_history_increment_is_bounds_checked() {
    let mut h = TCtrHistory::new(2);
    h.increment(1);
    h.increment(1);
    h.increment(0);
    assert_eq!(h.n, vec![1, 2]);
    // An out-of-range class is ignored (no panic), leaving counts unchanged.
    h.increment(5);
    assert_eq!(h.n, vec![1, 2]);
}

#[test]
fn mean_history_add_accumulates_sum_and_count() {
    let mut m = TCtrMeanHistory::default();
    m.add(1.0);
    m.add(0.5);
    assert!((m.sum - 1.5).abs() < 1e-6);
    assert_eq!(m.count, 2);
}
