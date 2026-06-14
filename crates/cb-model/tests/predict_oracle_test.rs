//! Prediction-type oracle (LOSS-06): apply the `binclf_skeleton` model to
//! `numeric_tiny`, transform the raw `approx` through each [`PredictionType`], and
//! assert each output matches its committed `prediction_types/*.npy` fixture at
//! <= 1e-5.
//!
//! The `prediction_types` fixtures were generated from the SAME model as
//! `binclf_skeleton` (their `rawformulaval.npy` equals
//! `binclf_skeleton/predictions.npy`), so the model.json here is shared with
//! `apply_oracle_test.rs`. `Probability` / `LogProbability` are two-column
//! (flattened row-major `[class-0, class-1]` per object), so their fixtures are
//! length `2 * n_rows`.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{apply_prediction_type, predict_raw, Model, ModelSplit, ObliviousTree, PredictionType, Split};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, ModelJson, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

fn model_from_json(mj: &ModelJson) -> Model {
    let oblivious_trees = mj
        .oblivious_trees
        .iter()
        .map(|t| {
            let splits = t
                .splits
                .iter()
                .map(|s| ModelSplit::Float(Split {
                    feature: usize::try_from(s.float_feature_index).expect("non-negative feature"),
                    border: s.border,
                }))
                .collect();
            ObliviousTree {
                splits,
                leaf_values: t.leaf_values.clone(),
                leaf_weights: vec![0.0; t.leaf_values.len()],
            }
        })
        .collect();
    Model {
        oblivious_trees,
        bias: mj.bias().expect("bias must parse"),
        float_feature_borders: mj.float_feature_borders(),
        ctr_data: None,
    }
}

/// Apply the shared model and return the per-object raw `approx`.
fn raw_approx() -> Vec<f64> {
    let columns = load_feature_columns();
    let mj = load_model_json(&fixture("binclf_skeleton/model.json"))
        .unwrap_or_else(|e| panic!("binclf_skeleton/model.json must load: {e:?}"));
    predict_raw(&model_from_json(&mj), &columns)
}

/// Assert one prediction type against its fixture.
fn check_type(prediction_type: PredictionType, fixture_name: &str, stage: Stage) {
    let approx = raw_approx();
    let actual = apply_prediction_type(prediction_type, &approx);
    let expected =
        load_f64_vec(&fixture(&format!("prediction_types/{fixture_name}.npy"))).unwrap();
    compare_stage(stage, &expected, &actual)
        .unwrap_or_else(|e| panic!("{fixture_name} diverged: {e:?}"));
}

#[test]
fn predict_oracle_rawformulaval() {
    check_type(PredictionType::RawFormulaVal, "rawformulaval", Stage::Predictions);
}

#[test]
fn predict_oracle_probability() {
    // Two columns per object: [1 - sigmoid(a), sigmoid(a)].
    let approx = raw_approx();
    let actual = apply_prediction_type(PredictionType::Probability, &approx);
    assert_eq!(actual.len(), approx.len() * 2, "Probability must emit two columns");
    check_type(PredictionType::Probability, "probability", Stage::Predictions);
}

#[test]
fn predict_oracle_logprobability() {
    // Two columns per object: [-log(1+exp(a)), -log(1+exp(-a))].
    let approx = raw_approx();
    let actual = apply_prediction_type(PredictionType::LogProbability, &approx);
    assert_eq!(actual.len(), approx.len() * 2, "LogProbability must emit two columns");
    check_type(PredictionType::LogProbability, "logprobability", Stage::Predictions);
}

#[test]
fn predict_oracle_class() {
    check_type(PredictionType::Class, "class", Stage::Predictions);
}

#[test]
fn predict_oracle_exponent() {
    // Exponent uses f64::exp; the 1e-5 gate absorbs upstream FastExp's gap (A2).
    check_type(PredictionType::Exponent, "exponent", Stage::Predictions);
}
