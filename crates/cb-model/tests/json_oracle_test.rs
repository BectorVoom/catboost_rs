//! `model.json` (de)serialization oracle (MODEL-06, D-04, Security V5).
//!
//! Three locks:
//!   (a) `save_json` then `cb_oracle::model_json::load_model_json` yields matching
//!       split_borders / leaf_values / leaf_weights / bias <= 1e-5 — the existing
//!       upstream-schema parser doubles as the round-trip oracle (D-04). Asserts
//!       the emitted JSON nests `leaf_weights` per tree (Pitfall 2) and emits
//!       `scale_and_bias = [1, [bias]]` (Pitfall 6).
//!   (b) `load_json` on the upstream `model_serde/{binclf,regression}/model.json`
//!       reconstructs a Model whose `predict_raw` matches the upstream prediction
//!       fixture <= 1e-5.
//!   (c) malformed JSON -> typed `ModelError`, never panic (Security V5).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{decode_json, load_json, predict_raw, save_json, Model, ModelError, ObliviousTree, Split};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/`.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the `numeric_tiny` input matrix as per-feature `f32` SoA columns.
fn load_numeric_tiny() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// A process+test-unique temp path so parallel tests never share a file.
fn unique_tmp(tag: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("cb_model_{tag}_{}_{nonce}.json", std::process::id()))
}

/// A small Rust-built model for the export round-trip.
fn rust_built_model() -> Model {
    Model {
        oblivious_trees: vec![
            ObliviousTree {
                splits: vec![
                    Split { feature: 0, border: 0.5 },
                    Split { feature: 1, border: 1.5 },
                ],
                leaf_values: vec![0.1, -0.2, 0.3, -0.4],
                leaf_weights: vec![10.0, 5.0, 7.0, 3.0],
            },
            ObliviousTree {
                splits: vec![Split { feature: 0, border: 2.5 }],
                leaf_values: vec![0.05, -0.05],
                leaf_weights: vec![12.0, 13.0],
            },
        ],
        bias: 0.25,
        float_feature_borders: vec![vec![0.5, 2.5], vec![1.5]],
    }
}

// ── (a) save_json round-trips through the cb-oracle parser ──────────────────

#[test]
fn save_json_round_trips_through_oracle_parser() {
    let model = rust_built_model();
    let path = unique_tmp("save_roundtrip");
    save_json(&model, &path).expect("save_json must succeed");

    let mj = load_model_json(&path).expect("cb-oracle must parse our model.json");
    let _ = std::fs::remove_file(&path);

    // split_borders match (tree order, flattened).
    let expected_borders: Vec<f64> = model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter().map(|s| s.border))
        .collect();
    compare_stage(Stage::Splits, &expected_borders, &mj.split_borders())
        .unwrap_or_else(|e| panic!("split_borders diverged: {e:?}"));

    // leaf_values match.
    compare_stage(Stage::LeafValues, &model.leaf_values(), &mj.leaf_values())
        .unwrap_or_else(|e| panic!("leaf_values diverged: {e:?}"));

    // leaf_weights match (per-tree nested -> flatten both sides).
    let oracle_weights: Vec<f64> = mj.leaf_weights().into_iter().flatten().collect();
    compare_stage(Stage::LeafValues, &model.leaf_weights(), &oracle_weights)
        .unwrap_or_else(|e| panic!("leaf_weights diverged: {e:?}"));

    // bias matches scale_and_bias[1][0].
    let bias = mj.bias().expect("oracle must read bias");
    assert!((bias - model.bias).abs() <= 1e-5, "bias diverged: {bias} vs {}", model.bias);
}

#[test]
fn save_json_nests_leaf_weights_per_tree_and_emits_scale_and_bias() {
    let model = rust_built_model();
    let path = unique_tmp("schema_check");
    save_json(&model, &path).expect("save_json must succeed");
    let raw = std::fs::read_to_string(&path).expect("read back emitted json");
    let _ = std::fs::remove_file(&path);

    let v: serde_json::Value = serde_json::from_str(&raw).expect("emitted json must parse");

    // leaf_weights nested INSIDE each oblivious_trees[] entry (Pitfall 2).
    let trees = v["oblivious_trees"].as_array().expect("oblivious_trees array");
    assert_eq!(trees.len(), 2);
    for (ti, t) in trees.iter().enumerate() {
        let lw = t["leaf_weights"].as_array().unwrap_or_else(|| {
            panic!("tree {ti} must carry a nested leaf_weights array")
        });
        let lv = t["leaf_values"].as_array().expect("leaf_values array");
        assert_eq!(lw.len(), lv.len(), "tree {ti} leaf_weights/leaf_values length");
    }

    // scale_and_bias emitted as [1, [bias]] (Pitfall 6).
    let sab = v["scale_and_bias"].as_array().expect("scale_and_bias array");
    assert_eq!(sab.len(), 2);
    assert_eq!(sab[0].as_f64(), Some(1.0));
    let bias_vec = sab[1].as_array().expect("bias vector");
    assert_eq!(bias_vec.len(), 1);
    assert!((bias_vec[0].as_f64().unwrap() - model.bias).abs() <= 1e-12);
}

#[test]
fn save_json_load_json_full_round_trip() {
    let model = rust_built_model();
    let path = unique_tmp("full_roundtrip");
    save_json(&model, &path).expect("save_json must succeed");
    let reloaded = load_json(&path).expect("load_json must succeed");
    let _ = std::fs::remove_file(&path);
    assert_eq!(model, reloaded, "save_json -> load_json must reproduce the Model");
}

// ── (b) load_json on upstream model.json applies within tol ─────────────────

fn assert_upstream_json_load_parity(scenario: &str) {
    let model = load_json(&fixture(&format!("model_serde/{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("upstream {scenario}/model.json must load: {e:?}"));
    let columns = load_numeric_tiny();
    let actual = predict_raw(&model, &columns);
    let expected = load_f64_vec(&fixture(&format!("model_serde/{scenario}/predictions.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/predictions.npy must load: {e:?}"));
    compare_stage(Stage::Predictions, &expected, &actual)
        .unwrap_or_else(|e| panic!("upstream {scenario} model.json apply diverged: {e:?}"));
}

#[test]
fn json_load_upstream_binclf_applies_within_tol() {
    assert_upstream_json_load_parity("binclf");
}

#[test]
fn json_load_upstream_regression_applies_within_tol() {
    assert_upstream_json_load_parity("regression");
}

// ── (c) malformed JSON → typed error, never panic (Security V5) ─────────────

#[test]
fn malformed_json_is_typed_error_not_panic() {
    // Not even valid JSON.
    match decode_json("{ this is not json ]") {
        Err(ModelError::Json(_)) => {}
        other => panic!("garbage must be a Json error, got {other:?}"),
    }
    // Valid JSON, wrong shape (missing required fields).
    match decode_json("{\"unexpected\": true}") {
        Err(ModelError::Json(_)) => {}
        other => panic!("wrong-shape json must be a Json error, got {other:?}"),
    }
    // Structurally valid but malformed scale_and_bias (empty bias vector).
    let bad_bias = "{\"features_info\":{\"float_features\":[]},\"oblivious_trees\":[],\"scale_and_bias\":[1,[]]}";
    match decode_json(bad_bias) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("malformed scale_and_bias must be Deserialize error, got {other:?}"),
    }
}
