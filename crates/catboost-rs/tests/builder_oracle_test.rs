//! End-to-end public-API slice (RAPI-01 / ROADMAP Phase-4 criterion 5): for BOTH
//! a numeric binary-classification and a numeric regression fixture, drive the
//! FULL train -> serialize -> load -> predict cycle through the published
//! `catboost-rs` facade ONLY (no direct `cb-train`/`cb-model` import on the
//! prediction path), and lock the result against upstream catboost 1.2.10.
//!
//! # Two layers of assertion
//!
//! 1. **In-env determinism (always runs).** `fit -> save_cbm -> load_cbm ->
//!    predict` and `fit -> save_json -> load_json -> predict` must reproduce the
//!    freshly-fit model's predictions EXACTLY (Rust<->Rust round-trip): the
//!    serialization layer is lossless and the public API is deterministic. This
//!    needs no upstream fixture and is the in-env gate.
//!
//! 2. **Upstream oracle (`compare_stage`, <= 1e-5), always runs.** The reloaded
//!    model's predictions are compared to the committed upstream
//!    `predictions.npy`. The facade's `fit` computes its OWN quantization borders
//!    from the pool (`select_borders_greedy_logsum`); for `numeric_tiny` those
//!    borders reproduce upstream's border selection exactly, so the full
//!    train -> serialize -> load -> predict cycle matches upstream catboost
//!    1.2.10 <= 1e-5 for BOTH binclf and regression (ROADMAP Phase-4 criterion 5).
//!    This leg runs unconditionally as the end-to-end public-API oracle lock.
//!
//! The upstream `model_serde/{binclf,regression}` fixtures were trained on
//! `numeric_tiny` with `boost_from_average`, `depth=2`, `iterations=5`,
//! `learning_rate=0.1`, `l2_leaf_reg=3.0`, `leaf_estimation_method=Gradient`,
//! `bootstrap_type=No`, `random_seed=0` (see each `config.json`) — the builder
//! is configured to match.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{
    CatBoostBuilder, IngestSource, LeafMethod, Loss, Model, OwnedColumns, Pool, PredictionType,
};
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from catboost-rs's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the `numeric_tiny` input matrix as per-feature SoA `f64` columns.
fn load_feature_columns() -> Vec<Vec<f64>> {
    let x: Array2<f64> =
        read_npy(fixture("inputs/numeric_tiny/X.npy")).expect("numeric_tiny/X.npy must load");
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().copied().collect())
        .collect()
}

/// Raw continuous target (RMSE regression).
fn load_regression_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).expect("numeric_tiny/y.npy must load")
}

/// Binary Logloss label `y_binary = (y > median(y))` (matches the generator's
/// `model_serde/binclf` label definition).
fn load_binclf_target() -> Vec<f64> {
    let y = load_regression_target();
    let mut sorted = y.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    y.iter()
        .map(|&v| if v > median { 1.0 } else { 0.0 })
        .collect()
}

/// Build a [`Pool`] from the numeric_tiny float columns + a target.
fn build_pool(target: Vec<f64>) -> Pool {
    OwnedColumns::new(load_feature_columns(), target)
        .into_pool()
        .expect("numeric_tiny pool must build")
}

/// A unique temp path under the OS temp dir.
fn unique_tmp(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("catboost_rs_{tag}_{nanos}"))
}

/// The builder configured to match the upstream `model_serde/*` `config.json`.
fn configured_builder(loss: Loss, boost_from_average: bool) -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(loss)
        .iterations(5)
        .depth(2)
        .learning_rate(0.1)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(boost_from_average)
        .leaf_method(LeafMethod::Gradient)
        .random_seed(0)
}

/// Drive the full public-API slice for one scenario.
fn run_scenario(scenario: &str, loss: Loss, boost_from_average: bool, target: Vec<f64>) {
    let pool = build_pool(target);

    // (1) Train through the public Builder facade.
    let model = configured_builder(loss, boost_from_average)
        .fit(&pool)
        .unwrap_or_else(|e| panic!("{scenario}: fit failed: {e:?}"));

    // Baseline predictions from the freshly-fit model (the in-env reference).
    let baseline = model
        .predict_with(&pool, PredictionType::RawFormulaVal)
        .unwrap_or_else(|e| panic!("{scenario}: predict failed: {e:?}"));

    // (2) .cbm round-trip: save -> load -> predict must reproduce baseline EXACTLY.
    let cbm_path = unique_tmp(&format!("{}_cbm", scenario.replace('/', "_")));
    model
        .save_cbm(&cbm_path)
        .unwrap_or_else(|e| panic!("{scenario}: save_cbm failed: {e:?}"));
    let reloaded_cbm =
        Model::load_cbm(&cbm_path).unwrap_or_else(|e| panic!("{scenario}: load_cbm failed: {e:?}"));
    let after_cbm = reloaded_cbm
        .predict_with(&pool, PredictionType::RawFormulaVal)
        .unwrap_or_else(|e| panic!("{scenario}: predict after load_cbm failed: {e:?}"));
    compare_stage(Stage::Predictions, &baseline, &after_cbm)
        .unwrap_or_else(|e| panic!("{scenario}: .cbm round-trip diverged: {e:?}"));
    let _ = std::fs::remove_file(&cbm_path);

    // (3) model.json round-trip: save -> load -> predict must reproduce baseline.
    let json_path = unique_tmp(&format!("{}_json", scenario.replace('/', "_")));
    model
        .save_json(&json_path)
        .unwrap_or_else(|e| panic!("{scenario}: save_json failed: {e:?}"));
    let reloaded_json = Model::load_json(&json_path)
        .unwrap_or_else(|e| panic!("{scenario}: load_json failed: {e:?}"));
    let after_json = reloaded_json
        .predict_with(&pool, PredictionType::RawFormulaVal)
        .unwrap_or_else(|e| panic!("{scenario}: predict after load_json failed: {e:?}"));
    compare_stage(Stage::Predictions, &baseline, &after_json)
        .unwrap_or_else(|e| panic!("{scenario}: model.json round-trip diverged: {e:?}"));
    let _ = std::fs::remove_file(&json_path);

    // (4) predict_proba shorthand is wired (two columns per object).
    let proba = reloaded_cbm
        .predict_proba(&pool)
        .unwrap_or_else(|e| panic!("{scenario}: predict_proba failed: {e:?}"));
    assert_eq!(
        proba.len(),
        baseline.len() * 2,
        "{scenario}: predict_proba must emit two columns per object"
    );

    // (5) Upstream oracle leg (<= 1e-5): the reloaded model's predictions match
    //     the committed upstream catboost 1.2.10 `predictions.npy` (ROADMAP
    //     Phase-4 criterion 5). The builder's fit-from-pool borders reproduce
    //     upstream's border selection for numeric_tiny, so the whole public-API
    //     cycle is oracle-locked.
    let expected = load_f64_vec(&fixture(&format!("model_serde/{scenario}/predictions.npy")))
        .unwrap_or_else(|e| panic!("{scenario}: predictions.npy must load: {e:?}"));
    compare_stage(Stage::Predictions, &expected, &after_cbm)
        .unwrap_or_else(|e| panic!("{scenario}: upstream <= 1e-5 oracle diverged: {e:?}"));
}

#[test]
fn builder_binclf_full_cycle() {
    run_scenario("binclf", Loss::Logloss, false, load_binclf_target());
}

#[test]
fn builder_regression_full_cycle() {
    run_scenario("regression", Loss::Rmse, true, load_regression_target());
}
