//! Unit tests for the [`crate::compare`] comparator primitive. Dedicated
//! `*_test.rs` file per D-17.

use crate::compare::{assert_abs_close, compare_stage, Stage};
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

// --- compare_stage: FALSIFIABLE 1e-5 boundary gate (INFRA-04, T-01-10) -------
//
// These tests are the proof that `compare_stage` actually gates at 1e-5 — NOT a
// self-equality check. They are falsifiable in BOTH directions:
//   * a broken always-Ok comparator FAILS the 2e-5 -> Err assertions;
//   * a broken always-Err comparator FAILS the 9e-6 -> Ok assertions.

/// A reference vector to perturb. Magnitudes here are well clear of the 1e-5
/// boundary so the perturbation, not float noise, decides pass/fail.
fn reference() -> Vec<f64> {
    vec![0.0, 1.0, -2.5, 3.141_59, 42.0]
}

#[test]
fn compare_stage_predictions_passes_just_below_tolerance() {
    let expected = reference();
    let mut actual = expected.clone();
    // Perturb one element by 9e-6 (< 1e-5): must be Ok.
    actual[2] += 9e-6;
    assert!(compare_stage(Stage::Predictions, &expected, &actual).is_ok());
}

#[test]
fn compare_stage_predictions_fails_just_above_tolerance() {
    let expected = reference();
    let mut actual = expected.clone();
    // Perturb one element by 2e-5 (> 1e-5): must be Err(StageDiverged) tagged
    // with Stage::Predictions and the right index.
    actual[3] += 2e-5;
    match compare_stage(Stage::Predictions, &expected, &actual) {
        Err(OracleError::StageDiverged { stage, index, diff, .. }) => {
            assert_eq!(stage, Stage::Predictions);
            assert_eq!(index, 3);
            assert!((diff - 2e-5).abs() < 1e-9);
        }
        other => panic!("expected StageDiverged(Predictions, idx 3), got {other:?}"),
    }
}

#[test]
fn compare_stage_borders_is_stage_tagged_on_divergence() {
    // Prove stage tagging is wired per-stage, not hardcoded to Predictions.
    let expected = reference();
    let mut actual = expected.clone();
    actual[0] += 2e-5;
    match compare_stage(Stage::Borders, &expected, &actual) {
        Err(OracleError::StageDiverged { stage, index, .. }) => {
            assert_eq!(stage, Stage::Borders);
            assert_eq!(index, 0);
        }
        other => panic!("expected StageDiverged(Borders, idx 0), got {other:?}"),
    }
}

#[test]
fn compare_stage_borders_passes_just_below_tolerance() {
    let expected = reference();
    let mut actual = expected.clone();
    actual[0] += 9e-6;
    assert!(compare_stage(Stage::Borders, &expected, &actual).is_ok());
}

#[test]
fn compare_stage_length_mismatch_is_stage_tagged() {
    let expected = reference();
    let actual = vec![0.0, 1.0]; // shorter
    match compare_stage(Stage::StagedApprox, &expected, &actual) {
        Err(OracleError::StageLengthMismatch { stage, expected: e, actual: a }) => {
            assert_eq!(stage, Stage::StagedApprox);
            assert_eq!(e, 5);
            assert_eq!(a, 2);
        }
        other => panic!("expected StageLengthMismatch(StagedApprox), got {other:?}"),
    }
}
