//! Integration test: proves the per-stage READ pipeline on REAL committed
//! catboost==1.2.10 oracle fixtures, and proves `compare_stage` gates falsifiably
//! at 1e-5 on that real data (INFRA-04).
//!
//! Comparison of these oracle per-stage values against Rust-COMPUTED actuals is
//! deferred to cb-train (P3) / cb-model (P4), which produce those actuals; Phase 1
//! has no Rust algorithm, so it proves the read pipeline + tolerance gate on real
//! fixtures only — it does NOT assert `expected == expected` self-equality.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::{compare_stage, load_f64_vec, OracleError, Stage};

fn fixture(rel: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/cb-oracle ; fixtures are committed under it.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

/// Proves the READ path on REAL oracle data: both borders.npy and predictions.npy
/// load to non-empty Vec<f64> through the real ndarray-npy pipeline.
#[test]
fn reads_real_borders_and_predictions_fixtures() {
    let borders = load_f64_vec(&fixture("regression_skeleton/borders.npy"))
        .expect("borders.npy must load via the ndarray-npy pipeline");
    let predictions = load_f64_vec(&fixture("regression_skeleton/predictions.npy"))
        .expect("predictions.npy must load via the ndarray-npy pipeline");

    assert!(!borders.is_empty(), "real borders fixture must be non-empty");
    assert!(!predictions.is_empty(), "real predictions fixture must be non-empty");
}

/// Proves the GATE on REAL oracle data: a 2e-5 perturbation of the real loaded
/// predictions is rejected with a stage-tagged Diverged error.
#[test]
fn compare_stage_rejects_2e5_perturbation_of_real_predictions() {
    let loaded = load_f64_vec(&fixture("regression_skeleton/predictions.npy"))
        .expect("predictions.npy must load");
    assert!(!loaded.is_empty());

    let mut perturbed = loaded.clone();
    perturbed[0] += 2e-5; // just above the 1e-5 boundary

    match compare_stage(Stage::Predictions, &loaded, &perturbed) {
        Err(OracleError::StageDiverged { stage, index, .. }) => {
            assert_eq!(stage, Stage::Predictions);
            assert_eq!(index, 0);
        }
        other => panic!("expected StageDiverged(Predictions, idx 0) on real data, got {other:?}"),
    }
}

/// Proves the GATE accepts a sub-tolerance perturbation of the real data: a 9e-6
/// shift of the real loaded predictions is within tolerance.
#[test]
fn compare_stage_accepts_9e6_perturbation_of_real_predictions() {
    let loaded = load_f64_vec(&fixture("regression_skeleton/predictions.npy"))
        .expect("predictions.npy must load");
    assert!(!loaded.is_empty());

    let mut perturbed = loaded.clone();
    perturbed[0] += 9e-6; // just below the 1e-5 boundary

    assert!(compare_stage(Stage::Predictions, &loaded, &perturbed).is_ok());
}
