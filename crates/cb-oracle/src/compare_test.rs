//! Unit tests for the [`crate::compare`] comparator primitive. Dedicated
//! `*_test.rs` file per D-17.

use crate::compare::{assert_abs_close, Stage};
use crate::error::OracleError;

const TOL: f64 = 1e-5;

#[test]
fn ok_when_all_within_tolerance() {
    let expected = [0.0, 1.0, -2.5, 3.14159];
    // Every paired diff is < 1e-5.
    let actual = [0.000_001, 0.999_995, -2.500_004, 3.141_592];
    assert!(assert_abs_close(&expected, &actual, TOL).is_ok());
}

#[test]
fn exact_tolerance_boundary_is_ok() {
    // diff == tol must NOT diverge (strict `>` comparison).
    let expected = [0.0];
    let actual = [TOL];
    assert!(assert_abs_close(&expected, &actual, TOL).is_ok());
}

#[test]
fn length_mismatch_errors() {
    let expected = [0.0, 1.0, 2.0];
    let actual = [0.0, 1.0];
    match assert_abs_close(&expected, &actual, TOL) {
        Err(OracleError::LengthMismatch { expected: e, actual: a }) => {
            assert_eq!(e, 3);
            assert_eq!(a, 2);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn diverged_reports_first_offending_index() {
    let expected = [0.0, 1.0, 2.0];
    // index 1 diverges by 0.1 > tol; index 0 is fine.
    let actual = [0.0, 1.1, 2.0];
    match assert_abs_close(&expected, &actual, TOL) {
        Err(OracleError::Diverged { index, expected: e, actual: a, diff }) => {
            assert_eq!(index, 1);
            assert_eq!(e, 1.0);
            assert_eq!(a, 1.1);
            assert!((diff - 0.1).abs() < 1e-9);
        }
        other => panic!("expected Diverged, got {other:?}"),
    }
}

#[test]
fn empty_slices_are_ok() {
    let empty: [f64; 0] = [];
    assert!(assert_abs_close(&empty, &empty, TOL).is_ok());
}

#[test]
fn stage_variants_are_distinct() {
    // Smoke test that the Stage enum is usable/comparable by later phases.
    assert_ne!(Stage::Borders, Stage::Predictions);
    assert_eq!(Stage::LeafValues, Stage::LeafValues);
}
