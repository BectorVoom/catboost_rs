//! PUBLIC load-model->predict oracle for all 5 multi-output losses (CR-01 / Plan
//! 06.2-06 Task 3). A dedicated `tests/` file (source/test separation, CLAUDE.md).
//!
//! This proves the LOAD->PREDICT surface CR-01 broke: for each frozen catboost
//! 1.2.10 fixture, `cb_model::load_json(model.json)` -> `predict_raw_multi` ->
//! `apply_multiclass_prediction` is compared element-wise to the frozen
//! `predictions.npy` at <= 1e-5. Distinct from the EXISTING training-staged oracle
//! (`cb-train/tests/multiclass_oracle_test.rs`), which feeds the trainer's staged
//! buffer; THIS test loads the published `model.json` through the PUBLIC
//! `load_json` (NO `from_trained` in the predict path).
//!
//! Prediction-type / kind per fixture (from each `config.json`):
//!   - multiclass_softmax    : Probability, Softmax   (softmax over dim)
//!   - multiclass_onevsall   : Probability, OneVsAll  (per-dim sigmoid)
//!   - multilogloss          : Probability, MultiLabel(per-dim sigmoid)
//!   - multicrossentropy     : Probability, MultiLabel(per-dim sigmoid)
//!   - multiquantile         : RawFormulaVal           (identity, dim-major->object-major)
//!
//! `predictions.npy` is OBJECT-MAJOR `(n_rows, dim)` row-major flat f64, which is
//! exactly `apply_multiclass_prediction`'s output layout. NO `#[ignore]`, NO
//! weakened tolerance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{
    apply_multiclass_prediction, load_json, predict_raw_multi, MultiClassKind, PredictionType,
};
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

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

/// Run the PUBLIC oracle for one fixture: load `model.json`, predict the raw
/// dim-major approx via `predict_raw_multi`, apply the multiclass transform, and
/// compare to the frozen object-major `predictions.npy` at <= 1e-5.
fn check_public_oracle(scenario: &str, prediction_type: PredictionType, kind: MultiClassKind) {
    let columns = load_feature_columns();

    // PUBLIC load surface — NOT `from_trained`.
    let model = load_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load via PUBLIC load_json: {e:?}"));
    assert_eq!(
        model.approx_dimension, 3,
        "{scenario}: expected 3 approx dimensions from the public load"
    );

    // Public N-dim apply -> dim-major raw approx (length dim * n).
    let raw = predict_raw_multi(&model, &columns);
    let dim = model.approx_dimension;
    assert_eq!(raw.len() % dim, 0, "{scenario}: raw approx length must be a multiple of dim");

    // Multiclass transform -> object-major (n, dim) row-major, matching predictions.npy.
    let predictions = apply_multiclass_prediction(
        prediction_type,
        kind,
        &raw,
        dim,
        &model.class_to_label,
    );

    let expected = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/predictions.npy must load: {e:?}"));
    compare_stage(Stage::Predictions, &expected, &predictions)
        .unwrap_or_else(|e| panic!("{scenario}: PUBLIC load->predict diverged: {e:?}"));
}

#[test]
fn public_oracle_multiclass_softmax() {
    check_public_oracle(
        "multiclass_softmax",
        PredictionType::Probability,
        MultiClassKind::Softmax,
    );
}

#[test]
fn public_oracle_multiclass_onevsall() {
    check_public_oracle(
        "multiclass_onevsall",
        PredictionType::Probability,
        MultiClassKind::OneVsAll,
    );
}

#[test]
fn public_oracle_multilogloss() {
    check_public_oracle(
        "multilogloss",
        PredictionType::Probability,
        MultiClassKind::MultiLabel,
    );
}

#[test]
fn public_oracle_multicrossentropy() {
    check_public_oracle(
        "multicrossentropy",
        PredictionType::Probability,
        MultiClassKind::MultiLabel,
    );
}

#[test]
fn public_oracle_multiquantile() {
    // MultiQuantile: RawFormulaVal identity (the dim-major raw approx transposed
    // dim-major -> object-major). Kind is irrelevant for the RawFormulaVal arm.
    check_public_oracle(
        "multiquantile",
        PredictionType::RawFormulaVal,
        MultiClassKind::MultiLabel,
    );
}

/// The PUBLIC multiclass `Class` transform recovers the ORIGINAL labels via the
/// `class_to_label` parsed from `model_info.class_params` (Task 2) — every
/// predicted class is one of the fixture's original `[0,1,2]` labels.
#[test]
fn public_oracle_class_labels_round_trip() {
    let columns = load_feature_columns();
    let model = load_json(&fixture("multiclass_softmax/model.json")).unwrap();
    assert_eq!(
        model.class_to_label,
        vec![0.0, 1.0, 2.0],
        "the public load must recover class_to_label [0,1,2] from model_info.class_params"
    );
    let raw = predict_raw_multi(&model, &columns);
    let classes = apply_multiclass_prediction(
        PredictionType::Class,
        MultiClassKind::Softmax,
        &raw,
        model.approx_dimension,
        &model.class_to_label,
    );
    let n = columns.first().map_or(0, Vec::len);
    assert_eq!(classes.len(), n, "one class per object");
    for &c in &classes {
        assert!(
            (0.0..=2.0).contains(&c) && c.fract() == 0.0,
            "class {c} must be an original label in {{0,1,2}}"
        );
    }
}
