//! Auto class-weight oracle: proves the Rust Balanced / SqrtBalanced calculators
//! ([`cb_data::balanced_class_weights`] / [`cb_data::sqrt_balanced_class_weights`])
//! reproduce upstream CatBoost's auto class-weight output to <= 1e-5 on the
//! frozen `class_weights` fixtures (DATA-08).
//!
//! The expected per-class weights are committed under
//! `cb-oracle/fixtures/class_weights/{balanced.npy,sqrt_balanced.npy}` and were
//! extracted from `CatBoostClassifier.get_all_params()['class_weights']` for a
//! binary dataset with `class_counts = [30, 10]` (see that scenario's
//! `config.json`). The SqrtBalanced value `1.7320507764816284` is CatBoost's
//! f32-precision `sqrt(3.0f)` widened to f64 (~3e-8 below the pure-f64
//! `sqrt(3.0)`); the <= 1e-5 oracle tolerance absorbs that, and the Rust
//! calculator computes in f32 to match.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors `borders_oracle_test.rs:14`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::{balanced_class_weights, sqrt_balanced_class_weights};
use cb_oracle::{assert_abs_close, load_f64_vec};

/// Resolve a path under `cb-oracle/fixtures/` from cb-data's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The `class_weights` fixture scenario: a binary dataset with `class_counts =
/// [30, 10]` (config.json). Reproduce it as 30 objects of class 0 followed by 10
/// of class 1, all unit-weighted.
fn binary_30_10_dataset() -> (Vec<usize>, Vec<f64>) {
    let mut classes = vec![0_usize; 30];
    classes.extend(std::iter::repeat(1_usize).take(10));
    let weights = vec![1.0_f64; classes.len()];
    (classes, weights)
}

/// Widen an `f32` class-weight vector to `f64` for the oracle comparison.
fn widen(weights: &[f32]) -> Vec<f64> {
    weights.iter().map(|&w| f64::from(w)).collect()
}

#[test]
fn balanced_class_weights_match_oracle() {
    let expected = load_f64_vec(&fixture("class_weights/balanced.npy")).unwrap();
    let (classes, weights) = binary_30_10_dataset();

    let actual = widen(&balanced_class_weights(&classes, &weights, 2).unwrap());

    assert_eq!(actual.len(), expected.len());
    assert_abs_close(&expected, &actual, 1e-5)
        .unwrap_or_else(|e| panic!("Balanced class weights diverged from oracle: {e:?}"));
}

#[test]
fn sqrt_balanced_class_weights_match_oracle() {
    let expected = load_f64_vec(&fixture("class_weights/sqrt_balanced.npy")).unwrap();
    let (classes, weights) = binary_30_10_dataset();

    let actual = widen(&sqrt_balanced_class_weights(&classes, &weights, 2).unwrap());

    assert_eq!(actual.len(), expected.len());
    assert_abs_close(&expected, &actual, 1e-5)
        .unwrap_or_else(|e| panic!("SqrtBalanced class weights diverged from oracle: {e:?}"));
}
