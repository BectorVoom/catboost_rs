//! Unit tests for `export/coreml.rs` (CM-01 guard, CM-02 round-trip). Sibling
//! `#[path]` mount (source/test separation, CLAUDE.md), mirroring
//! `onnx_test.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use prost::Message;

use super::*;
use crate::coreml_generated as cm;
use crate::ctr_data::{CtrData, ECtrType, Prior};
use crate::model::{
    CtrSplit, Model, ModelSplit, NonSymmetricTree, ObliviousTree, RegionTree, Split,
};

// ── Shared fixtures ─────────────────────────────────────────────────────────

/// An all-empty, all-oblivious, float-only, `ctr_data: None`, scalar model —
/// the baseline every disqualifying-condition test overrides one field of.
fn empty_model() -> Model {
    Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

fn minimal_non_symmetric_tree() -> NonSymmetricTree {
    NonSymmetricTree {
        tree_splits: Vec::new(),
        step_nodes: vec![(0, 0)],
        node_id_to_leaf_id: vec![0],
        leaf_values: vec![0.0],
        leaf_weights: vec![0.0],
    }
}

fn minimal_region_tree() -> RegionTree {
    RegionTree {
        levels: Vec::new(),
        leaf_values: vec![0.0],
        leaf_weights: vec![0.0],
    }
}

fn minimal_ctr_split() -> CtrSplit {
    CtrSplit {
        projection: cb_train::TProjection::single(0),
        ctr_type: ECtrType::Borders,
        prior: Prior { num: 0.0, denom: 1.0 },
        target_border_idx: 0,
        border: 0.0,
        shift: 0.0,
        scale: 1.0,
    }
}

fn ctr_split_tree() -> ObliviousTree {
    ObliviousTree {
        splits: vec![ModelSplit::Ctr(minimal_ctr_split())],
        leaf_values: vec![0.1, 0.2],
        leaf_weights: vec![1.0, 1.0],
    }
}

fn empty_ctr_data() -> CtrData {
    CtrData {
        tables: std::collections::BTreeMap::new(),
    }
}

fn float_split(feature: usize, border: f64) -> ModelSplit {
    ModelSplit::Float(Split { feature, border })
}

/// A deterministic depth-2 float-only tree: splits[0]=(feat 1, 5.0),
/// splits[1]=(feat 0, 1.0); leaf_values 10..13 (forward-bit order).
fn depth2_tree() -> ObliviousTree {
    ObliviousTree {
        splits: vec![float_split(1, 5.0), float_split(0, 1.0)],
        leaf_values: vec![10.0, 11.0, 12.0, 13.0],
        leaf_weights: vec![1.0; 4],
    }
}

/// A deterministic depth-3 float-only tree: splits[0]=(2,9.0),
/// splits[1]=(0,1.0), splits[2]=(1,5.0); leaf_values 0..8 (forward-bit order).
fn depth3_tree() -> ObliviousTree {
    ObliviousTree {
        splits: vec![
            float_split(2, 9.0),
            float_split(0, 1.0),
            float_split(1, 5.0),
        ],
        leaf_values: (0..8).map(f64::from).collect(),
        leaf_weights: vec![1.0; 8],
    }
}

fn unique_tmp(tag: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "cb_coreml_{tag}_{}_{nonce}.mlmodel",
        std::process::id()
    ))
}

/// Pull the `TreeEnsembleParameters` out of a decoded `cm::Model`.
fn ensemble(model: &cm::Model) -> &cm::TreeEnsembleParameters {
    match model.r#type.as_ref().expect("model has a Type") {
        cm::model::Type::TreeEnsembleRegressor(reg) => {
            reg.tree_ensemble.as_ref().expect("regressor has ensemble")
        }
    }
}

/// Index a decoded ensemble's nodes for one tree by `node_id`.
fn nodes_of_tree(params: &cm::TreeEnsembleParameters, tree_id: u64) -> Vec<&cm::TreeNode> {
    let mut v: Vec<&cm::TreeNode> =
        params.nodes.iter().filter(|n| n.tree_id == tree_id).collect();
    v.sort_by_key(|n| n.node_id);
    v
}

fn node_by_id(nodes: &[&cm::TreeNode], node_id: u64) -> cm::TreeNode {
    (*nodes
        .iter()
        .find(|n| n.node_id == node_id)
        .expect("node id present"))
    .clone()
}

// ── CM-01: guard rejection ──────────────────────────────────────────────────

/// CM-01: CTR / non-symmetric / region / multi-dim models each yield the
/// matching typed `Unsupported` error, and NO file is written.
#[test]
fn coreml_rejects_unsupported() {
    // (a) non-symmetric.
    let mut m = empty_model();
    m.non_symmetric_trees = vec![minimal_non_symmetric_tree()];
    let path = unique_tmp("reject_nonsym");
    assert!(matches!(
        export_coreml(&m, &path),
        Err(CoreMlExportError::NonObliviousTreesUnsupported)
    ));
    assert!(!path.exists());

    // (b) region.
    let mut m = empty_model();
    m.region_trees = vec![minimal_region_tree()];
    let path = unique_tmp("reject_region");
    assert!(matches!(
        export_coreml(&m, &path),
        Err(CoreMlExportError::RegionTreesUnsupported)
    ));
    assert!(!path.exists());

    // (c) CTR via a ModelSplit::Ctr split.
    let mut m = empty_model();
    m.oblivious_trees = vec![ctr_split_tree()];
    let path = unique_tmp("reject_ctr_split");
    assert!(matches!(
        export_coreml(&m, &path),
        Err(CoreMlExportError::CategoricalFeaturesUnsupported)
    ));
    assert!(!path.exists());

    // (c') CTR via baked ctr_data.
    let mut m = empty_model();
    m.ctr_data = Some(empty_ctr_data());
    let path = unique_tmp("reject_ctr_data");
    assert!(matches!(
        export_coreml(&m, &path),
        Err(CoreMlExportError::CategoricalFeaturesUnsupported)
    ));
    assert!(!path.exists());

    // (d) multi-dimensional.
    let mut m = empty_model();
    m.approx_dimension = 2;
    let path = unique_tmp("reject_multidim");
    assert!(matches!(
        export_coreml(&m, &path),
        Err(CoreMlExportError::MultiDimUnsupported)
    ));
    assert!(!path.exists());
}

/// A supported float-only oblivious scalar model exports (guard returns `Ok`).
#[test]
fn coreml_accepts_supported() {
    let mut m = empty_model();
    m.oblivious_trees = vec![depth2_tree()];
    m.float_feature_borders = vec![vec![1.0], vec![5.0]];
    let path = unique_tmp("accept");
    export_coreml(&m, &path).expect("supported model exports");
    assert!(path.exists());
    let _ = std::fs::remove_file(&path);
}

// ── CM-02: round-trip decode ────────────────────────────────────────────────

/// CM-02: a known 2-tree (depth-2 + depth-3) float-only model encodes then
/// re-decodes with the SAME prost structs; the decoded structure reproduces the
/// source per IMPLEMENTATION_NOTES §2 — leaves-first node ids, reversed
/// split-order per level, forward-bit leaf values, and
/// `base_prediction_value == [bias]`.
#[test]
fn coreml_nodes_match_source() {
    let mut m = empty_model();
    m.oblivious_trees = vec![depth2_tree(), depth3_tree()];
    m.bias = -0.25;
    m.float_feature_borders = vec![vec![1.0], vec![5.0], vec![9.0]];

    let proto = build_model(&m);
    let mut buf = Vec::new();
    proto.encode(&mut buf).expect("encode");
    let decoded = cm::Model::decode(buf.as_slice()).expect("decode");

    assert_eq!(decoded.specification_version, 1);
    let params = ensemble(&decoded);
    assert_eq!(params.num_prediction_dimensions, 1);
    assert_eq!(params.base_prediction_value, vec![-0.25]);

    // Two trees present.
    let n_trees = params
        .nodes
        .iter()
        .map(|n| n.tree_id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(n_trees, [0u64, 1u64].into_iter().collect());

    // ── Tree 0 (depth-2): leaves 0..3, internal 4,5 (level 1), 6 (root). ──
    let t0 = nodes_of_tree(params, 0);
    assert_eq!(t0.len(), 7);
    for leaf in 0..4u64 {
        let n = node_by_id(&t0, leaf);
        assert_eq!(n.node_behavior, cm::TreeNodeBehavior::LeafNode as i32);
        assert_eq!(n.evaluation_info.len(), 1);
        assert_eq!(n.evaluation_info[0].evaluation_index, 0);
        // forward-bit leaf value == leaf_values[leaf] == 10 + leaf.
        assert_eq!(n.evaluation_info[0].evaluation_value, 10.0 + leaf as f64);
    }
    // Level 1 (deepest internal) tests splits[0] == (feature 1, border 5.0).
    for (node_id, false_c, true_c) in [(4u64, 0u64, 1u64), (5u64, 2u64, 3u64)] {
        let n = node_by_id(&t0, node_id);
        assert_eq!(
            n.node_behavior,
            cm::TreeNodeBehavior::BranchOnValueGreaterThan as i32
        );
        assert_eq!(n.branch_feature_index, 1);
        assert_eq!(n.branch_feature_value, 5.0);
        assert_eq!(n.false_child_node_id, false_c);
        assert_eq!(n.true_child_node_id, true_c);
        assert!(!n.missing_value_tracks_true_child);
    }
    // Root (node 6) tests splits[1] == (feature 0, border 1.0), F->4, T->5.
    let root0 = node_by_id(&t0, 6);
    assert_eq!(root0.branch_feature_index, 0);
    assert_eq!(root0.branch_feature_value, 1.0);
    assert_eq!(root0.false_child_node_id, 4);
    assert_eq!(root0.true_child_node_id, 5);

    // ── Tree 1 (depth-3): leaves 0..7, internal 8..11 (lvl2), 12,13 (lvl1),
    //    14 (root). ──
    let t1 = nodes_of_tree(params, 1);
    assert_eq!(t1.len(), 15);
    for leaf in 0..8u64 {
        let n = node_by_id(&t1, leaf);
        assert_eq!(n.node_behavior, cm::TreeNodeBehavior::LeafNode as i32);
        assert_eq!(n.evaluation_info[0].evaluation_value, leaf as f64);
    }
    // Level 2 (@8): tests splits[0] == (feature 2, 9.0); F->2p, T->2p+1.
    for (idx, node_id) in (8u64..12).enumerate() {
        let n = node_by_id(&t1, node_id);
        assert_eq!(n.branch_feature_index, 2);
        assert_eq!(n.branch_feature_value, 9.0);
        assert_eq!(n.false_child_node_id, (idx as u64) * 2);
        assert_eq!(n.true_child_node_id, (idx as u64) * 2 + 1);
    }
    // Level 1 (@12): tests splits[1] == (feature 0, 1.0); children @8.
    for (idx, node_id) in (12u64..14).enumerate() {
        let n = node_by_id(&t1, node_id);
        assert_eq!(n.branch_feature_index, 0);
        assert_eq!(n.branch_feature_value, 1.0);
        assert_eq!(n.false_child_node_id, 8 + (idx as u64) * 2);
        assert_eq!(n.true_child_node_id, 8 + (idx as u64) * 2 + 1);
    }
    // Root (@14): tests splits[2] == (feature 1, 5.0), F->12, T->13.
    let root1 = node_by_id(&t1, 14);
    assert_eq!(root1.branch_feature_index, 1);
    assert_eq!(root1.branch_feature_value, 5.0);
    assert_eq!(root1.false_child_node_id, 12);
    assert_eq!(root1.true_child_node_id, 13);
}

/// CM-02 edge: a depth-0 tree (single leaf, no splits) emits exactly one leaf
/// node and NO internal nodes; `base_prediction_value == [bias]` even at bias
/// 0.0 (IMPLEMENTATION_NOTES §3 unconditional emission).
#[test]
fn coreml_depth0_single_leaf() {
    let mut m = empty_model();
    m.oblivious_trees = vec![ObliviousTree {
        splits: Vec::new(),
        leaf_values: vec![0.7],
        leaf_weights: vec![1.0],
    }];
    m.bias = 0.0;

    let proto = build_model(&m);
    let params_owned = ensemble(&proto).clone();
    assert_eq!(params_owned.base_prediction_value, vec![0.0]);
    assert_eq!(params_owned.nodes.len(), 1);
    let only = &params_owned.nodes[0];
    assert_eq!(only.node_id, 0);
    assert_eq!(only.node_behavior, cm::TreeNodeBehavior::LeafNode as i32);
    assert_eq!(only.evaluation_info[0].evaluation_value, 0.7);
}

/// CM-02: the input/output feature descriptors match §4 — one `feature_{i}`
/// doubleType input per float feature, one `prediction` multiArray DOUBLE
/// shape [1] output.
#[test]
fn coreml_feature_descriptors() {
    let mut m = empty_model();
    m.oblivious_trees = vec![depth2_tree()];
    m.float_feature_borders = vec![vec![1.0], vec![5.0]];

    let proto = build_model(&m);
    let desc = proto.description.as_ref().expect("description");
    assert_eq!(desc.predicted_feature_name, "prediction");
    assert_eq!(desc.input.len(), 2);
    for (i, fd) in desc.input.iter().enumerate() {
        assert_eq!(fd.name, format!("feature_{i}"));
        match fd.r#type.as_ref().and_then(|t| t.r#type.as_ref()) {
            Some(cm::feature_type::Type::DoubleType(_)) => {}
            other => panic!("input {i} expected doubleType, got {other:?}"),
        }
    }
    assert_eq!(desc.output.len(), 1);
    let out = &desc.output[0];
    assert_eq!(out.name, "prediction");
    match out.r#type.as_ref().and_then(|t| t.r#type.as_ref()) {
        Some(cm::feature_type::Type::MultiArrayType(a)) => {
            assert_eq!(a.shape, vec![1]);
            assert_eq!(a.data_type, cm::ArrayDataType::Double as i32);
        }
        other => panic!("output expected multiArrayType, got {other:?}"),
    }
}
