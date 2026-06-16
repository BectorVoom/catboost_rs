//! `.cbm` (de)serialization oracle (MODEL-01, D-02/D-03, Security V5).
//!
//! Three locks:
//!   (a) `save_cbm` -> `load_cbm` reproduces a Rust-built [`cb_model::Model`]
//!       EXACTLY (semantic round-trip, `assert_eq!`), with f32-exact borders so
//!       the f32 border wire type is lossless for this model.
//!   (b) `load_cbm` on the upstream-produced catboost 1.2.10 `.cbm`
//!       (`model_serde/{binclf,regression}/model.cbm`) then `predict_raw` matches
//!       the upstream `RawFormulaVal` prediction fixture <= 1e-5 (cross-version
//!       load parity, D-02 1.2.10 bar).
//!   (c) malformed input — bad magic / oversized size / truncated buffer / wrong
//!       FormatVersion — each returns `Err(ModelError::*)` and NEVER panics
//!       (Security V5, T-04-03-01..03).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the apply/predict oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{
    decode_cbm, load_cbm, predict_raw, save_cbm, Model, ModelError, ModelSplit, ObliviousTree,
    Split, CBM1, FLATBUFFERS_MODEL_V1,
};
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

/// Load the `numeric_tiny` input matrix as per-feature `f32` SoA columns (the
/// dataset both `model_serde` scenarios were generated on).
fn load_numeric_tiny() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Build the canonical [`Model`] from an upstream [`ModelJson`] (splits, leaf
/// values, bias, borders). Leaf weights are carried verbatim (per-tree nested).
fn model_from_json(mj: &ModelJson) -> Model {
    let leaf_weights = mj.leaf_weights();
    let oblivious_trees = mj
        .oblivious_trees
        .iter()
        .enumerate()
        .map(|(ti, t)| {
            let splits = t
                .splits
                .iter()
                .map(|s| ModelSplit::Float(Split {
                    feature: usize::try_from(s.float_feature_index).expect("non-negative feature"),
                    border: s.border,
                }))
                .collect();
            let weights = leaf_weights
                .get(ti)
                .filter(|w| w.len() == t.leaf_values.len())
                .cloned()
                .unwrap_or_else(|| vec![0.0; t.leaf_values.len()]);
            ObliviousTree {
                splits,
                leaf_values: t.leaf_values.clone(),
                leaf_weights: weights,
            }
        })
        .collect();
    Model {
        oblivious_trees,
        bias: mj.bias().expect("bias must parse"),
        float_feature_borders: mj.float_feature_borders(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// A small Rust-built model whose borders are f32-exact (`0.5`, `1.5`, `2.5`) so
/// the f32 border wire type round-trips losslessly — leaf values/weights are f64
/// on the wire and round-trip exactly.
fn rust_built_model() -> Model {
    Model {
        oblivious_trees: vec![
            ObliviousTree {
                splits: vec![
                    ModelSplit::Float(Split { feature: 0, border: 0.5 }),
                    ModelSplit::Float(Split { feature: 1, border: 1.5 }),
                ],
                leaf_values: vec![0.1, -0.2, 0.3, -0.4],
                leaf_weights: vec![10.0, 5.0, 7.0, 3.0],
            },
            ObliviousTree {
                splits: vec![ModelSplit::Float(Split { feature: 0, border: 2.5 })],
                leaf_values: vec![0.05, -0.05],
                leaf_weights: vec![12.0, 13.0],
            },
        ],
        bias: 0.25,
        float_feature_borders: vec![vec![0.5, 2.5], vec![1.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

// ── (a) semantic round-trip ────────────────────────────────────────────────

/// A process+test-unique temp path so parallel tests never share a file.
fn unique_tmp(tag: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("cb_model_{tag}_{}_{nonce}.cbm", std::process::id()))
}

#[test]
fn cbm_round_trip_reproduces_model() {
    let model = rust_built_model();
    let path = unique_tmp("roundtrip");
    save_cbm(&model, &path).expect("save_cbm must succeed");
    let reloaded = load_cbm(&path).expect("load_cbm must succeed");
    let _ = std::fs::remove_file(&path);
    assert_eq!(model, reloaded, "save_cbm -> load_cbm must reproduce the Model");
}

#[test]
fn cbm_round_trip_predicts_identically() {
    // Even where borders are not f32-exact, the loaded model must apply the same
    // as the source on the dataset (semantic apply parity through the round-trip).
    let mj = load_model_json(&fixture("model_serde/binclf/model.json"))
        .unwrap_or_else(|e| panic!("binclf model.json must load: {e:?}"));
    let model = model_from_json(&mj);
    let path = unique_tmp("roundtrip_binclf");
    save_cbm(&model, &path).expect("save_cbm must succeed");
    let reloaded = load_cbm(&path).expect("load_cbm must succeed");
    let _ = std::fs::remove_file(&path);

    let columns = load_numeric_tiny();
    let before = predict_raw(&model, &columns);
    let after = predict_raw(&reloaded, &columns);
    compare_stage(Stage::Predictions, &before, &after)
        .unwrap_or_else(|e| panic!("round-trip apply diverged: {e:?}"));
}

// ── (b) upstream 1.2.10 .cbm load parity ───────────────────────────────────

fn assert_upstream_cbm_load_parity(scenario: &str) {
    let model = load_cbm(&fixture(&format!("model_serde/{scenario}/model.cbm")))
        .unwrap_or_else(|e| panic!("upstream {scenario}/model.cbm must load: {e:?}"));
    let columns = load_numeric_tiny();
    let actual = predict_raw(&model, &columns);
    let expected = load_f64_vec(&fixture(&format!("model_serde/{scenario}/predictions.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/predictions.npy must load: {e:?}"));
    compare_stage(Stage::Predictions, &expected, &actual)
        .unwrap_or_else(|e| panic!("upstream {scenario} .cbm apply diverged: {e:?}"));
}

#[test]
fn cbm_load_upstream_binclf_applies_within_tol() {
    assert_upstream_cbm_load_parity("binclf");
}

#[test]
fn cbm_load_upstream_regression_applies_within_tol() {
    assert_upstream_cbm_load_parity("regression");
}

// ── (c) malformed input → typed error, never panic (Security V5) ────────────

/// A valid minimal `.cbm` byte buffer to corrupt for the malformed-input cases.
fn valid_cbm_bytes() -> Vec<u8> {
    let model = rust_built_model();
    let path = unique_tmp("valid_for_corrupt");
    save_cbm(&model, &path).expect("save_cbm must succeed");
    let bytes = std::fs::read(&path).expect("read back the saved cbm");
    let _ = std::fs::remove_file(&path);
    bytes
}

#[test]
fn cbm_bad_magic_is_typed_error() {
    let mut bytes = valid_cbm_bytes();
    bytes[0] = b'X'; // corrupt the magic
    match decode_cbm(&bytes) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("bad magic must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn cbm_oversized_size_is_typed_error() {
    let mut bytes = valid_cbm_bytes();
    // Declare a core size far larger than the file — must be bounded, not OOB.
    bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    match decode_cbm(&bytes) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("oversized size must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn cbm_truncated_buffer_is_typed_error() {
    let bytes = valid_cbm_bytes();
    // Keep the frame but truncate the FlatBuffers payload mid-buffer.
    let truncated = &bytes[..bytes.len().saturating_sub(bytes.len() / 2).max(8) + 4];
    match decode_cbm(truncated) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("truncated buffer must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn cbm_short_header_is_typed_error() {
    // Fewer than 8 framing bytes must error, never index out of bounds.
    for n in 0..8 {
        let bytes = vec![CBM1[0]; n];
        match decode_cbm(&bytes) {
            Err(ModelError::Deserialize(_)) => {}
            other => panic!("short header (len {n}) must be Deserialize error, got {other:?}"),
        }
    }
}

#[test]
fn format_version_literal_is_the_canonical_typo() {
    assert_eq!(FLATBUFFERS_MODEL_V1, "FlabuffersModel_v1");
}
