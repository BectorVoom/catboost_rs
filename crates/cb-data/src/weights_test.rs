//! Unit tests for [`crate::weights`]. Dedicated `*_test.rs` file per the
//! source/test separation rule; no `#[cfg(test)] mod` lives in `weights.rs`.
//!
//! The oracle parity comparison (Balanced / SqrtBalanced vs the committed
//! `class_weights` fixtures) lives in `tests/weights_oracle_test.rs`; this file
//! covers the algebra and the degenerate-class floor branch.

use cb_core::CbError;

use crate::weights::{
    balanced_class_weights, resolve_object_weights, sqrt_balanced_class_weights,
    summary_class_weights,
};

/// 30 objects of class 0, 10 of class 1 — the `class_weights` fixture scenario.
fn binary_30_10() -> Vec<usize> {
    let mut classes = vec![0_usize; 30];
    classes.extend(std::iter::repeat(1_usize).take(10));
    classes
}

#[test]
fn summary_weights_count_objects_per_class_under_unit_weights() {
    let classes = binary_30_10();
    let weights = vec![1.0_f64; classes.len()];
    let summary = summary_class_weights(&classes, &weights, 2).unwrap();
    assert_eq!(summary, vec![30.0, 10.0]);
}

#[test]
fn balanced_matches_max_over_count() {
    let classes = binary_30_10();
    let weights = vec![1.0_f64; classes.len()];
    let balanced = balanced_class_weights(&classes, &weights, 2).unwrap();
    // max=30: class 0 -> 30/30 = 1.0, class 1 -> 30/10 = 3.0.
    assert_eq!(balanced, vec![1.0_f32, 3.0_f32]);
}

#[test]
fn sqrt_balanced_is_f32_sqrt_of_balanced() {
    let classes = binary_30_10();
    let weights = vec![1.0_f64; classes.len()];
    let sqrt = sqrt_balanced_class_weights(&classes, &weights, 2).unwrap();
    // class 1 -> sqrt(3.0) computed in f32 == sqrtf(3.0).
    assert_eq!(sqrt, vec![1.0_f32, 3.0_f32.sqrt()]);
}

#[test]
fn empty_class_hits_floor_branch_with_no_panic() {
    // 3 classes declared but class 2 has zero objects: its summary weight is 0,
    // which is <= 1e-8, so both calculators must return 1.0 (not inf / NaN).
    let classes = vec![0_usize, 0, 1, 1, 1];
    let weights = vec![1.0_f64; classes.len()];

    let summary = summary_class_weights(&classes, &weights, 3).unwrap();
    assert_eq!(summary, vec![2.0, 3.0, 0.0]);

    let balanced = balanced_class_weights(&classes, &weights, 3).unwrap();
    // max=3: class 0 -> 3/2=1.5, class 1 -> 3/3=1.0, class 2 (empty) -> floor 1.0.
    assert_eq!(balanced, vec![1.5_f32, 1.0_f32, 1.0_f32]);
    assert!(balanced.iter().all(|w| w.is_finite()));

    let sqrt = sqrt_balanced_class_weights(&classes, &weights, 3).unwrap();
    assert_eq!(sqrt, vec![(3.0_f32 / 2.0).sqrt(), 1.0_f32, 1.0_f32]);
    assert!(sqrt.iter().all(|w| w.is_finite()));
}

#[test]
fn zero_class_count_is_an_error() {
    let err = summary_class_weights(&[0], &[1.0], 0).unwrap_err();
    assert!(matches!(err, CbError::OutOfRange(_)));
}

#[test]
fn class_index_out_of_range_is_an_error() {
    let err = summary_class_weights(&[0, 2], &[1.0, 1.0], 2).unwrap_err();
    assert!(matches!(err, CbError::OutOfRange(_)));
}

#[test]
fn mismatched_weight_length_is_an_error() {
    let err = summary_class_weights(&[0, 1], &[1.0], 2).unwrap_err();
    assert!(matches!(err, CbError::LengthMismatch { .. }));
}

#[test]
fn resolve_expands_class_weights_to_per_object() {
    let classes = vec![0_usize, 1, 0, 1];
    let class_weights = vec![1.0_f32, 3.0_f32];
    // No explicit per-object weights -> base 1.0 each.
    let resolved = resolve_object_weights(&class_weights, &[], &classes).unwrap();
    assert_eq!(resolved, vec![1.0, 3.0, 1.0, 3.0]);
}

#[test]
fn resolve_combines_explicit_and_class_weights() {
    let classes = vec![0_usize, 1];
    let class_weights = vec![2.0_f32, 5.0_f32];
    let per_object = vec![10.0_f64, 0.5];
    let resolved = resolve_object_weights(&class_weights, &per_object, &classes).unwrap();
    // object 0: 10*2=20 ; object 1: 0.5*5=2.5.
    assert_eq!(resolved, vec![20.0, 2.5]);
}

#[test]
fn resolve_empty_class_weights_is_pass_through() {
    let classes = vec![0_usize, 1, 0];
    let per_object = vec![1.0_f64, 2.0, 3.0];
    let resolved = resolve_object_weights(&[], &per_object, &classes).unwrap();
    assert_eq!(resolved, per_object);
}

#[test]
fn resolve_mismatched_per_object_length_is_an_error() {
    let classes = vec![0_usize, 1];
    let err = resolve_object_weights(&[1.0, 1.0], &[1.0], &classes).unwrap_err();
    assert!(matches!(err, CbError::LengthMismatch { .. }));
}
