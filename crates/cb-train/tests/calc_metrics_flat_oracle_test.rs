//! Flat `calc_metric` ≤1e-5 oracle (ORCH-04-S2) against frozen catboost 1.2.10
//! reference values.
//!
//! Metrics on FIXED `(label, approx)` predictions have no training/quantization
//! nondeterminism, so this is the cleanest possible oracle. The fixtures under
//! `calc_metrics/` are generated OFFLINE (RUN-ONCE/COMMIT) by:
//!     .venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --calc-metrics
//! and CI only READS the committed `.npy`. `label` is pinned to {0,1} so the one
//! shared pair satisfies RMSE + Logloss + MSLE simultaneously; `approx > -1`
//! satisfies the MSLE log-domain guard.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::{compare_stage, load_f64_vec, Stage};
use cb_train::{calc_metric, EvalMetric};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("calc_metrics")
        .join(rel)
}

/// Shared frozen flat inputs `(label, approx)`.
fn flat_inputs() -> (Vec<f64>, Vec<f64>) {
    let label = load_f64_vec(&fixture("label.npy")).unwrap();
    let approx = load_f64_vec(&fixture("approx.npy")).unwrap();
    (label, approx)
}

/// Evaluate one flat metric over the frozen inputs and gate it against its
/// committed upstream scalar at ≤1e-5.
fn gate(metric: EvalMetric, fixture_name: &str, weight: &[f64]) {
    let (label, approx) = flat_inputs();
    let expected = load_f64_vec(&fixture(fixture_name)).unwrap();
    assert_eq!(expected.len(), 1, "{fixture_name}: one scalar metric value");
    let actual = calc_metric(&metric, &label, &approx, weight, &[])
        .unwrap_or_else(|e| panic!("{fixture_name}: calc_metric failed: {e:?}"));
    compare_stage(Stage::Predictions, &expected, &[actual])
        .unwrap_or_else(|e| panic!("{fixture_name}: diverged from upstream: {e:?}"));
}

#[test]
fn rmse_matches_upstream() {
    gate(EvalMetric::Rmse, "rmse.npy", &[]);
}

#[test]
fn logloss_matches_upstream() {
    gate(EvalMetric::Logloss, "logloss.npy", &[]);
}

#[test]
fn msle_matches_upstream() {
    gate(EvalMetric::Msle, "msle.npy", &[]);
}

#[test]
fn rmse_weighted_matches_upstream() {
    let weight = load_f64_vec(&fixture("weight.npy")).unwrap();
    gate(EvalMetric::Rmse, "rmse_weighted.npy", &weight);
}
