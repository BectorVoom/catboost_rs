//! Wave-B uncertainty PREDICTION-type oracle (LOSS-06 / Plan 06.4-03): load the
//! frozen RMSEWithUncertainty `model.json` and reproduce the three uncertainty
//! prediction types — `RmseWithUncertainty` (single-model, 2 cols),
//! `VirtEnsembles` (V x 2), and `TotalUncertainty` (3 cols) — against the
//! committed (OFFLINE-generated, frozen) upstream catboost 1.2.10
//! `uncertainty_predict` fixture at <= 1e-5 (D-6.4-02 full Python-reachable
//! oracle). Closes the Phase-4 D-10 uncertainty-prediction-type deferral.
//!
//! A dedicated `tests/` file (source/test separation, CLAUDE.md). NO `#[ignore]`,
//! NO weakened tolerance.
//!
//! Layouts (object-major, matching the fixture):
//!   - RMSEWithUncertainty: predict(prediction_type='RMSEWithUncertainty')
//!     (n, 2) row-major [mean, variance], variance = exp(2*log-scale) (Pitfall 6).
//!   - VirtEnsembles: virtual_ensembles_predict(...) (n, V, 2) row-major; per
//!     ensemble [mean, variance=exp(2*log-scale)].
//!   - TotalUncertainty: virtual_ensembles_predict(...) (n, 3) row-major
//!     [mean, knowledgeUncertainty, dataUncertainty] (CalcRegressionUncertaitny).
//!
//! Virtual ensembles slice the trained tree sequence (`apply.cpp:526-600`):
//! evalPeriod = end // (2*V); begin = end - evalPeriod*V; ensemble 0 seeds from
//! trees [0, begin) WITH the per-dim bias [mean, 0.5*log(var)] (treeStart==0);
//! each ensemble adds the apply of its evalPeriod-tree slice (NO bias,
//! treeStart>0) and copies the running sum forward. The per-dim bias is read from
//! `scale_and_bias[1]` of the model.json (the project's scalar `Model.bias` keeps
//! dim-0 only — the VE apply takes the per-dim bias as an explicit argument).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{
    apply_prediction_type, apply_ve_prediction_type, apply_virtual_ensembles, load_json,
    predict_raw_multi_biased, PredictionType,
};
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

const VIRTUAL_ENSEMBLES_COUNT: usize = 5;
const APPROX_DIMENSION: usize = 2;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the `numeric_tiny` input matrix as per-feature `f32` SoA columns (the same
/// public apply input the trainer/Builder feed).
fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Read the per-dimension bias `scale_and_bias[1]` straight from the model.json
/// (the project's `Model.bias` is the scalar dim-0 mean; the VE base ensemble
/// needs the full `[mean, 0.5*log(var)]` per-dim bias).
fn per_dim_bias() -> Vec<f64> {
    let raw = std::fs::read_to_string(fixture("uncertainty_predict/model.json"))
        .unwrap_or_else(|e| panic!("model.json must read: {e:?}"));
    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    doc["scale_and_bias"][1]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// Number of objects (from the regression input rows).
fn n_objects() -> usize {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy")).unwrap();
    x.nrows()
}

/// RMSEWithUncertainty single-model predict: load -> predict_raw_multi (dim-major
/// [mean(0..n), log-scale(n..2n)]) -> apply_prediction_type(RmseWithUncertainty)
/// -> object-major (n, 2) [mean, variance=exp(2*log-scale)].
#[test]
fn rmse_with_uncertainty_predict_oracle() {
    let columns = load_feature_columns();
    let model = load_json(&fixture("uncertainty_predict/model.json"))
        .unwrap_or_else(|e| panic!("model.json must load: {e:?}"));
    assert_eq!(model.approx_dimension, APPROX_DIMENSION);

    // Dim-major raw approx (length 2*n): [mean(0..n), log-scale(n..2n)], seeded
    // with the PER-DIM bias [mean, 0.5*log(var)] (the scalar `Model.bias` drops
    // dim-1 — `predict_raw_multi_biased` carries the full per-dim bias).
    let bias = per_dim_bias();
    let raw = predict_raw_multi_biased(&model, &columns, &bias);
    // The RmseWithUncertainty transform -> object-major (n, 2) [mean, variance].
    let predictions = apply_prediction_type(PredictionType::RmseWithUncertainty, &raw);

    let expected = load_f64_vec(&fixture("uncertainty_predict/rmse_with_uncertainty.npy")).unwrap();
    compare_stage(Stage::Predictions, &expected, &predictions)
        .unwrap_or_else(|e| panic!("RMSEWithUncertainty predict diverged: {e:?}"));
}

/// VirtEnsembles: load -> apply_virtual_ensembles (2V-row matrix) ->
/// apply_prediction_type(VirtEnsembles) -> object-major (n, V, 2) per-ensemble
/// [mean, variance=exp(2*log-scale)].
#[test]
fn virt_ensembles_predict_oracle() {
    let columns = load_feature_columns();
    let model = load_json(&fixture("uncertainty_predict/model.json")).unwrap();
    let bias = per_dim_bias();
    let n = n_objects();

    let ve = apply_virtual_ensembles(&model, &columns, &bias, VIRTUAL_ENSEMBLES_COUNT)
        .unwrap_or_else(|e| panic!("apply_virtual_ensembles failed: {e:?}"));
    let predictions = apply_ve_prediction_type(
        PredictionType::VirtEnsembles,
        &ve,
        VIRTUAL_ENSEMBLES_COUNT,
        APPROX_DIMENSION,
    );

    let expected = load_f64_vec(&fixture("uncertainty_predict/virt_ensembles.npy")).unwrap();
    assert_eq!(
        predictions.len(),
        n * VIRTUAL_ENSEMBLES_COUNT * APPROX_DIMENSION,
        "VirtEnsembles output must be (n, V, 2)"
    );
    compare_stage(Stage::Predictions, &expected, &predictions)
        .unwrap_or_else(|e| panic!("VirtEnsembles predict diverged: {e:?}"));
}

/// TotalUncertainty: load -> apply_virtual_ensembles (2V-row matrix) ->
/// apply_prediction_type(TotalUncertainty) -> object-major (n, 3)
/// [mean, knowledgeUncertainty, dataUncertainty].
#[test]
fn total_uncertainty_predict_oracle() {
    let columns = load_feature_columns();
    let model = load_json(&fixture("uncertainty_predict/model.json")).unwrap();
    let bias = per_dim_bias();
    let n = n_objects();

    let ve = apply_virtual_ensembles(&model, &columns, &bias, VIRTUAL_ENSEMBLES_COUNT)
        .unwrap_or_else(|e| panic!("apply_virtual_ensembles failed: {e:?}"));
    let predictions = apply_ve_prediction_type(
        PredictionType::TotalUncertainty,
        &ve,
        VIRTUAL_ENSEMBLES_COUNT,
        APPROX_DIMENSION,
    );

    let expected = load_f64_vec(&fixture("uncertainty_predict/total_uncertainty.npy")).unwrap();
    assert_eq!(predictions.len(), n * 3, "TotalUncertainty output must be (n, 3)");
    compare_stage(Stage::Predictions, &expected, &predictions)
        .unwrap_or_else(|e| panic!("TotalUncertainty predict diverged: {e:?}"));
}

/// Degenerate VE input (fewer than 2V+1 trees for the requested V) returns a
/// `CbResult` ERROR, NOT a panic (T-06.4C-01).
#[test]
fn virt_ensembles_too_few_trees_errors() {
    let columns = load_feature_columns();
    let model = load_json(&fixture("uncertainty_predict/model.json")).unwrap();
    let bias = per_dim_bias();
    // The model has 11 trees; V=6 needs >= 13 trees -> "Not enough trees" error.
    let result = apply_virtual_ensembles(&model, &columns, &bias, 6);
    assert!(
        result.is_err(),
        "VE with too few trees must return an error, not panic"
    );
}
