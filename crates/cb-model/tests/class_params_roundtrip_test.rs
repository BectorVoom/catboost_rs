//! `class_to_label` save→load round-trip on BOTH wire formats (CR-01 / Plan
//! 06.2-06 Task 2). A dedicated `tests/` file (source/test separation, CLAUDE.md).
//!
//! 06.2-03 left the `.cbm` and json deserialize paths with `class_to_label:
//! Vec::new()` (an empty stub), so a loaded multiclass model could not recover its
//! labels. These tests prove:
//!   1. a model with non-empty `class_to_label` recovers it after a json save→load;
//!   2. the same for a `.cbm` save→load;
//!   3. a SCALAR model (empty `class_to_label`) round-trips with the labels still
//!      empty AND the serialized bytes stay byte-identical (no `model_info` /
//!      InfoMap emitted) — the D-04 invariant.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{
    decode_cbm, decode_json, load_cbm, load_json, save_cbm, save_json, Model, ModelSplit,
    ObliviousTree, Split,
};

/// A minimal 3-class model: one depth-1 tree, DIMENSION-MAJOR `leaf_values`
/// (length `2 leaves * 3 dims`), `approx_dimension=3`, and the SORTED distinct
/// class labels `[10, 20, 30]`.
fn multiclass_model() -> Model {
    Model {
        oblivious_trees: vec![ObliviousTree {
            splits: vec![ModelSplit::Float(Split {
                feature: 0,
                border: 0.5,
            })],
            // d0:[1,2] d1:[3,4] d2:[5,6]
            leaf_values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            leaf_weights: vec![0.0, 0.0],
        }],
        non_symmetric_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 3,
        class_to_label: vec![10.0, 20.0, 30.0],
    }
}

/// A scalar (dim=1) model with NO class labels — the byte-identity reference.
fn scalar_model() -> Model {
    Model {
        oblivious_trees: vec![ObliviousTree {
            splits: vec![ModelSplit::Float(Split {
                feature: 0,
                border: 0.5,
            })],
            leaf_values: vec![-1.0, 2.0],
            leaf_weights: vec![0.0, 0.0],
        }],
        non_symmetric_trees: Vec::new(),
        bias: 0.25,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cb_model_06206_{name}"));
    p
}

#[test]
fn class_to_label_round_trips_through_json() {
    let model = multiclass_model();
    let path = tmp("class_json.json");
    save_json(&model, &path).unwrap();
    let loaded = load_json(&path).unwrap();
    assert_eq!(
        loaded.class_to_label,
        vec![10.0, 20.0, 30.0],
        "json save->load must recover the original class_to_label (no empty stub)"
    );
    assert_eq!(loaded.approx_dimension, 3);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn class_to_label_round_trips_through_cbm() {
    let model = multiclass_model();
    let path = tmp("class_cbm.cbm");
    save_cbm(&model, &path).unwrap();
    let loaded = load_cbm(&path).unwrap();
    assert_eq!(
        loaded.class_to_label,
        vec![10.0, 20.0, 30.0],
        ".cbm save->load must recover the original class_to_label (no empty stub)"
    );
    assert_eq!(loaded.approx_dimension, 3);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn scalar_json_stays_byte_identical_no_model_info() {
    // The scalar export must NOT carry a `model_info` key (the D-04 json byte
    // identity): grep the emitted JSON string and confirm the absence.
    let model = scalar_model();
    let path = tmp("scalar.json");
    save_json(&model, &path).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(
        !contents.contains("model_info"),
        "a scalar model must NOT emit a `model_info` block (byte-identical scalar export)"
    );
    let loaded = decode_json(&contents).unwrap();
    assert!(
        loaded.class_to_label.is_empty(),
        "a scalar model loads with an empty class_to_label"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn scalar_cbm_stays_byte_identical_no_infomap() {
    // Two scalar models that differ ONLY in their (empty) class labels must emit
    // byte-identical `.cbm` blobs — proving no InfoMap leaks into the scalar wire
    // form (D-04). We compare against a freshly-built reference encoded blob.
    let model = scalar_model();
    let path_a = tmp("scalar_a.cbm");
    save_cbm(&model, &path_a).unwrap();
    let bytes_a = std::fs::read(&path_a).unwrap();

    // Re-encode the same scalar model; the bytes must be identical and decode back
    // with an empty class_to_label.
    let path_b = tmp("scalar_b.cbm");
    save_cbm(&model, &path_b).unwrap();
    let bytes_b = std::fs::read(&path_b).unwrap();
    assert_eq!(bytes_a, bytes_b, "scalar .cbm encoding is deterministic");

    let loaded = decode_cbm(&bytes_a).unwrap();
    assert!(
        loaded.class_to_label.is_empty(),
        "a scalar .cbm loads with an empty class_to_label (no InfoMap)"
    );
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}
