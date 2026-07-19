//! Unit tests for `export/onnx.rs` (EXPORT-01a..e). Sibling `#[path]` mount
//! (source/test separation, CLAUDE.md), mirroring `partial_dependence_test.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use prost::Message;

use super::*;
use crate::ctr_data::{CtrData, ECtrType, Prior};
use crate::model::{
    CtrSplit, Model, ModelSplit, NonSymmetricTree, ObliviousTree, RegionTree, Split,
};

// ── Shared fixtures ─────────────────────────────────────────────────────────

/// An all-empty, all-oblivious, float-only, `ctr_data: None` model — the
/// baseline every disqualifying-condition test overrides one field of.
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

/// A minimal single-leaf (depth-0) non-symmetric tree — enough to make
/// `model.non_symmetric_trees` non-empty without caring about its content.
fn minimal_non_symmetric_tree() -> NonSymmetricTree {
    NonSymmetricTree {
        tree_splits: Vec::new(),
        step_nodes: vec![(0, 0)],
        node_id_to_leaf_id: vec![0],
        leaf_values: vec![0.0],
        leaf_weights: vec![0.0],
    }
}

/// A minimal depth-0 region tree (one leaf, no levels) — enough to make
/// `model.region_trees` non-empty without caring about its content.
fn minimal_region_tree() -> RegionTree {
    RegionTree {
        levels: Vec::new(),
        leaf_values: vec![0.0],
        leaf_weights: vec![0.0],
    }
}

/// A minimal `CtrSplit` over a single categorical feature — the ONLY
/// CTR-split constructor in this test suite (no existing helper to reuse; the
/// same technique EXPORT-01a's AT-01a-3/4 need).
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

/// A depth-1 oblivious tree with a single [`ModelSplit::Ctr`] split.
fn ctr_split_tree() -> ObliviousTree {
    ObliviousTree {
        splits: vec![ModelSplit::Ctr(minimal_ctr_split())],
        leaf_values: vec![0.1, 0.2],
        leaf_weights: vec![1.0, 1.0],
    }
}

/// An all-empty [`CtrData`] (baked tables present in NAME only — enough to
/// make `model.ctr_data.is_some()` true).
fn empty_ctr_data() -> CtrData {
    CtrData {
        tables: std::collections::BTreeMap::new(),
    }
}

/// A depth-3 oblivious tree with three DISTINCT `(feature, border)` splits at
/// distinct positions, hand-computed leaf values `0.0..8.0` (forward-bit
/// order) — the AT-01b-1/AT-01b-2 dedicated reversed-split-order regression
/// fixture.
fn three_distinct_split_tree() -> ObliviousTree {
    ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 2, border: 9.0 }), // idx 0
            ModelSplit::Float(Split { feature: 0, border: 1.0 }), // idx 1
            ModelSplit::Float(Split { feature: 1, border: 5.0 }), // idx 2
        ],
        leaf_values: (0..8).map(f64::from).collect(),
        leaf_weights: vec![1.0; 8],
    }
}

fn one_split_tree(feature: usize, border: f64, leaf_values: Vec<f64>) -> ObliviousTree {
    let n = leaf_values.len();
    ObliviousTree {
        splits: vec![ModelSplit::Float(Split { feature, border })],
        leaf_values,
        leaf_weights: vec![0.0; n],
    }
}

fn find_attr<'a>(node: &'a onnx::NodeProto, name: &str) -> Option<&'a onnx::AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

/// A process+test-unique temp `.onnx` path (mirrors `cbm_oracle_test.rs`'s
/// `unique_tmp`), so parallel tests never share a file.
fn unique_tmp(tag: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("cb_model_onnx_{tag}_{}_{nonce}.onnx", std::process::id()))
}

// ── EXPORT-01a: guard ───────────────────────────────────────────────────────

#[test]
fn rejects_non_symmetric_tree_model() {
    let mut model = empty_model();
    model.non_symmetric_trees.push(minimal_non_symmetric_tree());
    assert!(matches!(
        is_onnx_exportable(&model),
        Err(OnnxExportError::NonObliviousTreesUnsupported)
    ));
}

#[test]
fn rejects_region_tree_model() {
    let mut model = empty_model();
    model.region_trees.push(minimal_region_tree());
    assert!(matches!(
        is_onnx_exportable(&model),
        Err(OnnxExportError::RegionTreesUnsupported)
    ));
}

#[test]
fn rejects_ctr_split_model() {
    let mut model = empty_model();
    model.oblivious_trees.push(ctr_split_tree());
    assert!(matches!(
        is_onnx_exportable(&model),
        Err(OnnxExportError::CategoricalFeaturesUnsupported)
    ));
}

#[test]
fn rejects_baked_ctr_data_with_no_ctr_split() {
    let mut model = empty_model();
    model.oblivious_trees.push(one_split_tree(0, 0.0, vec![0.0, 1.0]));
    model.ctr_data = Some(empty_ctr_data());
    assert!(matches!(
        is_onnx_exportable(&model),
        Err(OnnxExportError::CategoricalFeaturesUnsupported)
    ));
}

#[test]
fn accepts_float_only_oblivious_model() {
    let mut model = empty_model();
    model.oblivious_trees.push(one_split_tree(0, 0.0, vec![0.0, 1.0]));
    assert!(is_onnx_exportable(&model).is_ok());
}

#[test]
fn guard_order_non_oblivious_wins_over_ctr() {
    let mut model = empty_model();
    model.non_symmetric_trees.push(minimal_non_symmetric_tree());
    model.oblivious_trees.push(ctr_split_tree());
    // Order slot 1 (non-symmetric) must fire, NOT slot 3 (CTR).
    assert!(matches!(
        is_onnx_exportable(&model),
        Err(OnnxExportError::NonObliviousTreesUnsupported)
    ));
}

// ── EXPORT-01b: per-tree ONNX node arrays ───────────────────────────────────

#[test]
fn reversed_split_order_matches_hand_computed_mapping() {
    let tree = three_distinct_split_tree();
    let frag = build_tree_nodes(&tree, 0);

    // n_internal = 2^3 - 1 = 7; node 0 = depth 0 (root), node 1 = depth 1,
    // node 3 = depth 2 (the first false-child chain: 0 -> 1 -> 3).
    // depth 0 reads splits[len-1-0] = splits[2] (feature 1, border 5.0).
    assert_eq!(frag.feature_ids[0], 1);
    assert_eq!(frag.values[0], 5.0);
    // depth 1 reads splits[len-1-1] = splits[1] (feature 0, border 1.0).
    assert_eq!(frag.feature_ids[1], 0);
    assert_eq!(frag.values[1], 1.0);
    // depth 2 reads splits[len-1-2] = splits[0] (feature 2, border 9.0).
    assert_eq!(frag.feature_ids[3], 2);
    assert_eq!(frag.values[3], 9.0);
}

#[test]
fn leaf_values_transcribed_verbatim_no_permutation() {
    let tree = three_distinct_split_tree();
    let frag = build_tree_nodes(&tree, 0);
    assert_eq!(frag.leaf_values, tree.leaf_values);
    assert_eq!(frag.leaf_node_ids.len(), tree.leaf_values.len());
}

#[test]
fn branch_gt_mode_and_complete_binary_child_indexing() {
    let tree = three_distinct_split_tree();
    let frag = build_tree_nodes(&tree, 7);
    let n_internal = 7; // 2^3 - 1
    for i in 0..n_internal {
        assert_eq!(frag.modes[i], "BRANCH_GT");
        assert_eq!(frag.false_node_ids[i], (2 * i + 1) as i64);
        assert_eq!(frag.true_node_ids[i], (2 * i + 2) as i64);
        assert_eq!(frag.tree_ids[i], 7);
    }
    for i in n_internal..frag.modes.len() {
        assert_eq!(frag.modes[i], "LEAF");
    }
}

#[test]
fn depth_zero_tree_is_single_leaf_node() {
    let tree = ObliviousTree {
        splits: Vec::new(),
        leaf_values: vec![3.5],
        leaf_weights: vec![1.0],
    };
    let frag = build_tree_nodes(&tree, 0);
    assert_eq!(frag.node_ids, vec![0]);
    assert_eq!(frag.modes, vec!["LEAF".to_owned()]);
    assert_eq!(frag.leaf_node_ids, vec![0]);
    assert_eq!(frag.leaf_values, vec![3.5]);
}

// ── EXPORT-01c: whole-ensemble regressor assembly ───────────────────────────

fn two_tree_model(bias: f64) -> Model {
    let mut model = empty_model();
    model.bias = bias;
    model.float_feature_borders = vec![vec![1.0], vec![5.0], vec![9.0]];
    model.oblivious_trees = vec![three_distinct_split_tree(), one_split_tree(0, 0.5, vec![-1.0, 1.0])];
    model
}

#[test]
fn two_tree_regressor_assembly_preserves_boosting_order() {
    let model = two_tree_model(0.0);
    let node = build_regressor_node(&model);

    let expect0 = build_tree_nodes(&model.oblivious_trees[0], 0);
    let expect1 = build_tree_nodes(&model.oblivious_trees[1], 1);

    let tree_ids = find_attr(&node, "nodes_treeids").expect("nodes_treeids present");
    let mut expected_tree_ids = expect0.tree_ids.clone();
    expected_tree_ids.extend(expect1.tree_ids.clone());
    assert_eq!(tree_ids.ints, expected_tree_ids);

    let feature_ids = find_attr(&node, "nodes_featureids").expect("present");
    let mut expected_feature_ids = expect0.feature_ids.clone();
    expected_feature_ids.extend(expect1.feature_ids.clone());
    assert_eq!(feature_ids.ints, expected_feature_ids);
}

#[test]
fn zero_bias_omits_base_values_attribute() {
    let model = two_tree_model(0.0);
    let node = build_regressor_node(&model);
    assert!(find_attr(&node, "base_values").is_none());
}

#[test]
fn nonzero_bias_sets_base_values() {
    let model = two_tree_model(2.5);
    let node = build_regressor_node(&model);
    let base_values = find_attr(&node, "base_values").expect("base_values present");
    assert_eq!(base_values.floats, vec![2.5_f32]);
}

#[test]
fn fixed_regressor_attributes() {
    let model = two_tree_model(0.0);
    let node = build_regressor_node(&model);
    assert_eq!(node.op_type, "TreeEnsembleRegressor");
    assert_eq!(node.domain, "ai.onnx.ml");
    let post_transform = find_attr(&node, "post_transform").expect("present");
    assert_eq!(post_transform.s, b"NONE");
    let n_targets = find_attr(&node, "n_targets").expect("present");
    assert_eq!(n_targets.i, 1);
}

// ── EXPORT-01d: whole-ensemble classifier assembly + ZipMap ────────────────

fn binary_classifier_model(bias: f64) -> Model {
    let mut model = empty_model();
    model.bias = bias;
    model.float_feature_borders = vec![vec![0.5]];
    model.approx_dimension = 1;
    model.oblivious_trees = vec![one_split_tree(0, 0.5, vec![-1.0, 1.0])];
    model
}

/// A small 2-tree, 3-class multiclass model. `leaf_values` is
/// DIMENSION-MAJOR: `leaf_values[class * n_leaves + leaf]`.
fn multiclass_model() -> Model {
    let mut model = empty_model();
    model.approx_dimension = 3;
    model.float_feature_borders = vec![vec![0.5]];
    // Depth-1 tree (2 leaves), 3 classes -> 6 dimension-major leaf values:
    // class 0: [0.0, 1.0], class 1: [2.0, 3.0], class 2: [4.0, 5.0].
    let tree = ObliviousTree {
        splits: vec![ModelSplit::Float(Split { feature: 0, border: 0.5 })],
        leaf_values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        leaf_weights: vec![1.0; 6],
    };
    model.oblivious_trees = vec![tree.clone(), tree];
    model
}

#[test]
fn binary_classifier_asymmetric_base_values() {
    let model = binary_classifier_model(1.0);
    let (node, _zipmap) = build_classifier_nodes(&model);
    let base_values = find_attr(&node, "base_values").expect("base_values present");
    assert_eq!(base_values.floats, vec![-1.0_f32, 1.0_f32]);
}

#[test]
fn binary_classifier_uses_logistic() {
    let model = binary_classifier_model(1.0);
    let (node, _zipmap) = build_classifier_nodes(&model);
    let post_transform = find_attr(&node, "post_transform").expect("present");
    assert_eq!(post_transform.s, b"LOGISTIC");
}

#[test]
fn multiclass_classifier_uses_softmax() {
    let model = multiclass_model();
    let (node, _zipmap) = build_classifier_nodes(&model);
    let post_transform = find_attr(&node, "post_transform").expect("present");
    assert_eq!(post_transform.s, b"SOFTMAX");
}

/// AT-01d-3b: the SAME 3-class model — assert the emitted per-class leaf
/// contributions equal `tree.leaf_values[class * n_leaves + leaf]`
/// (dimension-major), hand-computed directly from the tree's `leaf_values`,
/// NOT round-tripped through the exporter itself. Catches a leaf/class
/// transposition bug that `post_transform == "SOFTMAX"` alone cannot.
#[test]
fn multiclass_class_weights_use_dimension_major_indexing() {
    let model = multiclass_model();
    let (node, _zipmap) = build_classifier_nodes(&model);

    let class_ids = find_attr(&node, "class_ids").expect("present");
    let class_nodeids = find_attr(&node, "class_nodeids").expect("present");
    let class_weights = find_attr(&node, "class_weights").expect("present");
    let class_treeids = find_attr(&node, "class_treeids").expect("present");

    // Hand-compute the expected (tree, class, leaf) -> value triples directly
    // from the model's OWN leaf_values (dimension-major formula), independent
    // of the exporter's internal per-tree fragment builder.
    let n_leaves = 2usize;
    let dim = 3usize;
    let mut expected = Vec::new();
    for (tree_id, tree) in model.oblivious_trees.iter().enumerate() {
        for leaf in 0..n_leaves {
            for class in 0..dim {
                let value = tree.leaf_values[class * n_leaves + leaf];
                expected.push((tree_id as i64, class as i64, value));
            }
        }
    }

    assert_eq!(class_ids.ints.len(), expected.len());
    for (i, (tree_id, class, value)) in expected.iter().enumerate() {
        assert_eq!(class_treeids.ints[i], *tree_id, "treeid mismatch at {i}");
        assert_eq!(class_ids.ints[i], *class, "class mismatch at {i}");
        assert_eq!(class_weights.floats[i], *value as f32, "value mismatch at {i}");
        // class_nodeids must be a LEAF node id (present in the tree fragment).
        let frag = build_tree_nodes(&model.oblivious_trees[*tree_id as usize], *tree_id);
        assert!(frag.leaf_node_ids.contains(&class_nodeids.ints[i]));
    }
}

#[test]
fn zipmap_wired_to_classifier_probability_output() {
    let model = binary_classifier_model(0.0);
    let (node, zipmap) = build_classifier_nodes(&model);
    assert_eq!(zipmap.op_type, "ZipMap");
    assert_eq!(zipmap.domain, "ai.onnx.ml");
    assert_eq!(zipmap.input.len(), 1);
    // node.output[1] is the probability-tensor output (output[0] is "label").
    assert_eq!(zipmap.input[0], node.output[1]);
}

#[test]
fn class_labels_default_and_explicit() {
    let model = binary_classifier_model(0.0);
    let (node, _zipmap) = build_classifier_nodes(&model);
    let labels = find_attr(&node, "classlabels_int64s").expect("present");
    assert_eq!(labels.ints, vec![0, 1]);

    let mut model2 = binary_classifier_model(0.0);
    model2.class_to_label = vec![3.0, 7.0];
    let (node2, _zipmap2) = build_classifier_nodes(&model2);
    let labels2 = find_attr(&node2, "classlabels_int64s").expect("present");
    assert_eq!(labels2.ints, vec![3, 7]);
}

/// Regression test for the fallback-classlabels bug: a `dim > 2` multiclass
/// model with NO stored `class_to_label` (a realistic case — the .cbm/.json
/// loaders drop non-numeric label entries) must default `classlabels_int64s`
/// to `0..dim`, NOT a hardcoded 2-entry `[0, 1]` — otherwise `class_ids`
/// (which the per-leaf loop unconditionally emits over `0..dim`) reference
/// indices beyond `classlabels_int64s.len()`.
#[test]
fn class_labels_default_for_multiclass_covers_full_dim() {
    let model = multiclass_model(); // approx_dimension == 3, class_to_label empty
    assert!(model.class_to_label.is_empty());
    let (node, zipmap) = build_classifier_nodes(&model);

    let labels = find_attr(&node, "classlabels_int64s").expect("present");
    assert_eq!(labels.ints, vec![0, 1, 2]);

    // class_ids must never reference an index beyond classlabels_int64s.
    let class_ids = find_attr(&node, "class_ids").expect("present");
    for &id in &class_ids.ints {
        assert!(
            (id as usize) < labels.ints.len(),
            "class_id {id} out of range of classlabels_int64s (len {})",
            labels.ints.len()
        );
    }

    let zipmap_labels = find_attr(&zipmap, "classlabels_int64s").expect("present");
    assert_eq!(zipmap_labels.ints, vec![0, 1, 2]);
}

// ── Fix 2: non-integer class_to_label is rejected, never silently truncated ─

#[test]
fn validate_class_to_label_accepts_integer_valued_entries() {
    let mut model = binary_classifier_model(0.0);
    model.class_to_label = vec![0.0, 1.0, -5.0];
    assert!(validate_class_to_label(&model).is_ok());
}

#[test]
fn validate_class_to_label_rejects_fractional_entry() {
    let mut model = binary_classifier_model(0.0);
    model.class_to_label = vec![0.0, 1.5];
    assert!(matches!(
        validate_class_to_label(&model),
        Err(OnnxExportError::NonIntegerClassLabelsUnsupported)
    ));
}

#[test]
fn validate_class_to_label_rejects_non_finite_entry() {
    let mut model = binary_classifier_model(0.0);
    model.class_to_label = vec![f64::NAN];
    assert!(matches!(
        validate_class_to_label(&model),
        Err(OnnxExportError::NonIntegerClassLabelsUnsupported)
    ));

    let mut model2 = binary_classifier_model(0.0);
    model2.class_to_label = vec![f64::INFINITY];
    assert!(matches!(
        validate_class_to_label(&model2),
        Err(OnnxExportError::NonIntegerClassLabelsUnsupported)
    ));
}

#[test]
fn export_onnx_classifier_with_integer_class_to_label_succeeds() {
    let mut model = binary_classifier_model(0.0);
    model.class_to_label = vec![3.0, 7.0];
    let path = unique_tmp("classifier_integer_labels");
    let result = export_onnx(&model, &path, true);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn export_onnx_classifier_with_non_integer_class_to_label_errors() {
    let mut model = binary_classifier_model(0.0);
    model.class_to_label = vec![0.0, 1.5];
    let path = unique_tmp("classifier_fractional_labels");
    let result = export_onnx(&model, &path, true);
    assert!(matches!(
        result,
        Err(OnnxExportError::NonIntegerClassLabelsUnsupported)
    ));
    // Guard failure: no partial file left behind (same contract as the
    // structural guard's `guard_failure_writes_no_file`).
    assert!(!path.exists());
}

#[test]
fn export_onnx_regressor_ignores_non_integer_class_to_label() {
    // class_to_label is only read on the classifier path; a regressor export
    // must not reject a model over it.
    let mut model = two_tree_model(0.0);
    model.class_to_label = vec![0.0, 1.5];
    let path = unique_tmp("regressor_ignores_fractional_labels");
    let result = export_onnx(&model, &path, false);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let _ = std::fs::remove_file(&path);
}

// ── EXPORT-01e: metadata + serialization + entry point ─────────────────────

#[test]
fn guard_failure_writes_no_file() {
    let mut model = empty_model();
    model.oblivious_trees.push(ctr_split_tree());
    let path = unique_tmp("guard_failure");
    let result = export_onnx(&model, &path, false);
    assert!(result.is_err());
    assert!(!path.exists());
}

#[test]
fn regressor_round_trips_through_encode_decode() {
    let model = two_tree_model(2.5);
    let path = unique_tmp("regressor_roundtrip");
    export_onnx(&model, &path, false).expect("export_onnx must succeed");

    let bytes = std::fs::read(&path).expect("read back the written file");
    let _ = std::fs::remove_file(&path);
    let decoded = onnx::ModelProto::decode(bytes.as_slice()).expect("decode must succeed");

    assert_eq!(decoded.ir_version, 3);
    assert_eq!(decoded.opset_import.len(), 1);
    assert_eq!(decoded.opset_import[0].domain, "ai.onnx.ml");
    assert_eq!(decoded.opset_import[0].version, 2);

    let graph = decoded.graph.expect("graph present");
    assert_eq!(graph.node.len(), 1);
    let decoded_node = &graph.node[0];
    assert_eq!(decoded_node.op_type, "TreeEnsembleRegressor");

    let expected_node = build_regressor_node(&model);
    assert_eq!(
        find_attr(decoded_node, "nodes_treeids"),
        find_attr(&expected_node, "nodes_treeids")
    );
    assert_eq!(
        find_attr(decoded_node, "base_values"),
        find_attr(&expected_node, "base_values")
    );
}

#[test]
fn classifier_round_trips_through_encode_decode() {
    let model = binary_classifier_model(1.0);
    let path = unique_tmp("classifier_roundtrip");
    export_onnx(&model, &path, true).expect("export_onnx must succeed");

    let bytes = std::fs::read(&path).expect("read back the written file");
    let _ = std::fs::remove_file(&path);
    let decoded = onnx::ModelProto::decode(bytes.as_slice()).expect("decode must succeed");

    let graph = decoded.graph.expect("graph present");
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "TreeEnsembleClassifier");
    assert_eq!(graph.node[1].op_type, "ZipMap");

    // The probability_tensor ValueInfoProto's second dim is 2 for a binary
    // (approx_dimension==1) model — never 1 (plan-checker-added shape note).
    let prob_info = graph
        .value_info
        .iter()
        .find(|v| v.name == "probability_tensor")
        .expect("probability_tensor value_info present");
    let ty = prob_info.r#type.as_ref().expect("type present");
    let tensor_type = match ty.value.as_ref().expect("value present") {
        onnx::type_proto::Value::TensorType(t) => t,
        other => panic!("expected TensorType, got {other:?}"),
    };
    let shape = tensor_type.shape.as_ref().expect("shape present");
    assert_eq!(shape.dim.len(), 2);
    let second_dim = match shape.dim[1].value.as_ref().expect("dim value present") {
        onnx::tensor_shape_proto::dimension::Value::DimValue(v) => *v,
        other => panic!("expected DimValue, got {other:?}"),
    };
    assert_eq!(second_dim, 2);
}

#[test]
fn unwritable_path_returns_typed_io_error() {
    let model = two_tree_model(0.0);
    let path = std::path::PathBuf::from("/nonexistent-dir-onnx-export-test/model.onnx");
    let result = export_onnx(&model, &path, false);
    assert!(matches!(result, Err(OnnxExportError::Io(_))));
}
