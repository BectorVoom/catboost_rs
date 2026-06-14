//! Apply-path oracle (MODEL-02): build the canonical [`cb_model::Model`] from the
//! committed upstream `binclf_skeleton/model.json` and assert that the pure-Rust
//! [`cb_model::predict_raw`] reproduces the upstream `RawFormulaVal` predictions
//! (`binclf_skeleton/predictions.npy`) on `numeric_tiny` at <= 1e-5.
//!
//! `binclf_skeleton` is the SAME model the `prediction_types` fixtures were
//! generated from (their `rawformulaval.npy` equals `binclf_skeleton/predictions`),
//! so this lock and the prediction-type lock share one model.json.
//!
//! The apply path is GPU-toolchain-free: this test (and `apply.rs`) names NO
//! `cb-backend` / `cubecl` symbol — `predict_raw` runs with no CubeCL present.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the cb-train oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{binarize_feature, predict_raw, Model, ModelSplit, ObliviousTree, Split};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, ModelJson, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from cb-model's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the `numeric_tiny` input matrix as per-feature `f32` SoA columns.
fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Build the canonical [`Model`] from an upstream [`ModelJson`] — splits
/// (`float_feature_index` + `border`), per-tree `leaf_values`, the model `bias`,
/// and the per-float-feature borders. `leaf_weights` are unused by apply, so they
/// are filled with zeros of the matching length.
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

#[test]
fn apply_oracle_binclf_rawformulaval() {
    let columns = load_feature_columns();
    let mj = load_model_json(&fixture("binclf_skeleton/model.json"))
        .unwrap_or_else(|e| panic!("binclf_skeleton/model.json must load: {e:?}"));
    let model = model_from_json(&mj);

    let actual = predict_raw(&model, &columns);
    let expected = load_f64_vec(&fixture("binclf_skeleton/predictions.npy")).unwrap();

    compare_stage(Stage::Predictions, &expected, &actual)
        .unwrap_or_else(|e| panic!("apply predict_raw diverged: {e:?}"));
}

/// `binarize_feature` is the STRICT `>` border count (Step A): a value below
/// every border bins to 0, above every border bins to `borders.len()`, and a
/// value EXACTLY on a border does NOT exceed it (strict `>`).
#[test]
fn binarize_strict_greater_count() {
    let borders = [1.0_f64, 2.0, 3.0];
    assert_eq!(binarize_feature(0.5, &borders), 0);
    assert_eq!(binarize_feature(1.0, &borders), 0); // exactly on border 1.0 -> not >
    assert_eq!(binarize_feature(1.5, &borders), 1);
    assert_eq!(binarize_feature(2.0, &borders), 1); // exactly on border 2.0 -> not >
    assert_eq!(binarize_feature(3.5, &borders), 3);
    assert_eq!(binarize_feature(0.0, &[]), 0); // no borders -> bin 0
}

/// Bias is added EXACTLY once (RESEARCH Pitfall 6): a model with no trees
/// predicts exactly `bias` for every object, not `0` or `2*bias`.
#[test]
fn bias_added_once_no_trees() {
    let model = Model {
        oblivious_trees: Vec::new(),
        bias: 0.42,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
    };
    let columns = vec![vec![0.1_f32, 0.9, 0.3]];
    let preds = predict_raw(&model, &columns);
    assert_eq!(preds, vec![0.42, 0.42, 0.42]);
}
