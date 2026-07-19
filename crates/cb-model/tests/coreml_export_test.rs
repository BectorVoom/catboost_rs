//! EXPORT-02 (CM-04) CoreML-export oracle + golden-bytes regression pin.
//!
//! # Three-layer verification (IMPLEMENTATION_NOTES §5)
//!
//! This is layer 2+3: the emitted `.mlmodel` structure is diffed against
//! CatBoost's OWN `reference.mlmodel` (decoded by `coremltools` into
//! `structure.json` by `gen_fixtures.py`), and the emitted bytes are byte-pinned
//! to a frozen `golden.mlmodel`. The oracle target proves the emitted schema is
//! the REAL CoreML schema AND the node numbering / tree semantics match CatBoost
//! exactly — node ids / children / behaviors are compared EXACTLY (the exporter
//! replicates CatBoost's numbering, §2), branch thresholds + leaf values within
//! 1e-5 (to absorb f32/f64 rounding).
//!
//! Fixtures are FROZEN (`gen_fixtures.py` is a provenance record only; CatBoost
//! quantization is run-to-run nondeterministic). Never regenerated at test time.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::BTreeSet;
use std::path::PathBuf;

use cb_model::{coreml_generated as cm, export_coreml, load_cbm};
use prost::Message;
use serde_json::Value;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("coreml_export")
        .join(rel)
}

fn unique_tmp(tag: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "cb_coreml_oracle_{tag}_{}_{nonce}.mlmodel",
        std::process::id()
    ))
}

/// Encode `model.cbm` through the Rust exporter and return the emitted bytes.
fn emit_bytes() -> Vec<u8> {
    let model = load_cbm(&fixture("model.cbm")).expect("model.cbm loads");
    let path = unique_tmp("emit");
    export_coreml(&model, &path).expect("export_coreml succeeds");
    let bytes = std::fs::read(&path).expect("read emitted .mlmodel");
    let _ = std::fs::remove_file(&path);
    bytes
}

fn params_of(model: &cm::Model) -> cm::TreeEnsembleParameters {
    match model.r#type.as_ref().expect("model has a Type") {
        cm::model::Type::TreeEnsembleRegressor(reg) => reg
            .tree_ensemble
            .as_ref()
            .expect("regressor has ensemble")
            .clone(),
    }
}

/// CM-04 oracle: the Rust exporter's emitted structure matches CatBoost's own
/// `reference.mlmodel` (via `structure.json`) EXACTLY — node ids / children /
/// behaviors exact, thresholds + leaf values within 1e-5. This is the key
/// deliverable: the Rust CoreML output matches CatBoost's own.
#[test]
fn coreml_structure_matches_catboost_reference() {
    let bytes = emit_bytes();
    let decoded = cm::Model::decode(bytes.as_slice()).expect("emitted bytes decode");
    let params = params_of(&decoded);

    let structure: Value =
        serde_json::from_slice(&std::fs::read(fixture("structure.json")).expect("structure.json"))
            .expect("parse structure.json");

    // Top-level: spec type, dimension count, base prediction value.
    assert_eq!(structure["spec_type"], "treeEnsembleRegressor");
    assert_eq!(
        params.num_prediction_dimensions,
        structure["num_prediction_dimensions"].as_u64().unwrap()
    );
    let exp_base: Vec<f64> = structure["base_prediction_value"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(params.base_prediction_value.len(), exp_base.len());
    for (got, exp) in params.base_prediction_value.iter().zip(exp_base.iter()) {
        assert!(
            (got - exp).abs() <= TOL,
            "base_prediction_value {got} vs {exp}"
        );
    }

    // Index decoded nodes by (tree_id, node_id).
    let mut got_ids: BTreeSet<(u64, u64)> = BTreeSet::new();
    for n in &params.nodes {
        got_ids.insert((n.tree_id, n.node_id));
    }

    let ref_trees = structure["trees"].as_object().expect("trees object");
    let mut exp_ids: BTreeSet<(u64, u64)> = BTreeSet::new();

    for (tree_key, tree_nodes) in ref_trees {
        let tree_id: u64 = tree_key.parse().expect("tree id");
        let nodes = tree_nodes.as_object().expect("tree node map");
        for (node_key, node_val) in nodes {
            let node_id: u64 = node_key.parse().expect("node id");
            exp_ids.insert((tree_id, node_id));

            let got = params
                .nodes
                .iter()
                .find(|n| n.tree_id == tree_id && n.node_id == node_id)
                .unwrap_or_else(|| panic!("missing node ({tree_id},{node_id})"));

            // EXACT: behavior + children (the numbering replicates CatBoost).
            assert_eq!(
                got.node_behavior,
                node_val["behavior"].as_i64().unwrap() as i32,
                "behavior ({tree_id},{node_id})"
            );
            assert_eq!(
                got.true_child_node_id,
                node_val["true_child_node_id"].as_u64().unwrap(),
                "true_child ({tree_id},{node_id})"
            );
            assert_eq!(
                got.false_child_node_id,
                node_val["false_child_node_id"].as_u64().unwrap(),
                "false_child ({tree_id},{node_id})"
            );

            if got.node_behavior == cm::TreeNodeBehavior::LeafNode as i32 {
                // Leaf: evaluation value within 1e-5.
                let exp_leaf = node_val["leaf_value"].as_f64().unwrap();
                let got_leaf = got
                    .evaluation_info
                    .first()
                    .expect("leaf has evaluation_info")
                    .evaluation_value;
                assert!(
                    (got_leaf - exp_leaf).abs() <= TOL,
                    "leaf_value ({tree_id},{node_id}): {got_leaf} vs {exp_leaf}"
                );
            } else {
                // Branch: EXACT feature index, threshold within 1e-5.
                assert_eq!(
                    got.branch_feature_index,
                    node_val["branch_feature_index"].as_u64().unwrap(),
                    "branch_feature_index ({tree_id},{node_id})"
                );
                let exp_thr = node_val["branch_feature_value"].as_f64().unwrap();
                assert!(
                    (got.branch_feature_value - exp_thr).abs() <= TOL,
                    "branch_feature_value ({tree_id},{node_id}): {} vs {exp_thr}",
                    got.branch_feature_value
                );
            }
        }
    }

    // No missing and no extra nodes vs the CatBoost reference.
    assert_eq!(
        got_ids, exp_ids,
        "emitted node-id set must match the CatBoost reference exactly"
    );
}

/// CM-04 golden-bytes regression pin: the emitted `.mlmodel` bytes for the
/// frozen `model.cbm` equal the committed `golden.mlmodel`. Determinism
/// preconditions (IMPLEMENTATION_NOTES §5.3): the exporter emits NO protobuf
/// `map<…>` field and embeds NO version/timestamp, so the bytes are stable
/// run-to-run.
///
/// Regenerate the golden DELIBERATELY (schema change only): run this test once
/// with `CB_COREML_REGEN_GOLDEN=1`, inspect, and commit the rewritten
/// `golden.mlmodel`. It is never rewritten silently.
#[test]
fn coreml_golden_bytes_stable() {
    let bytes = emit_bytes();
    let golden_path = fixture("golden.mlmodel");

    if std::env::var_os("CB_COREML_REGEN_GOLDEN").is_some() {
        std::fs::write(&golden_path, &bytes).expect("write regenerated golden");
    }

    let golden = std::fs::read(&golden_path).expect(
        "golden.mlmodel must be committed; regenerate with CB_COREML_REGEN_GOLDEN=1 on a schema change",
    );
    assert_eq!(
        bytes, golden,
        "emitted CoreML bytes drifted from the committed golden.mlmodel"
    );

    // A second emit is byte-identical (determinism: no maps, no timestamps).
    assert_eq!(bytes, emit_bytes(), "CoreML export is not deterministic");
}
