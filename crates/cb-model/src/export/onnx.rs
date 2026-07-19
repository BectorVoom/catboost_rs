//! ONNX export (EXPORT-01 / Phase 17 T0-T5): a float-only, oblivious,
//! identity-scale [`Model`] to a well-formed `ai.onnx.ml`
//! `TreeEnsembleRegressor` / `TreeEnsembleClassifier` (+`ZipMap`) graph.
//!
//! # Source of truth
//!
//! Transcribes upstream's `catboost/libs/model/model_export/onnx_helpers.cpp`
//! (`InitMetadata`, `AddTree`, `ConvertTreeToOnnxGraph`, `TTreesAttributes`)
//! and `model_exporter.cpp` (`ExportModel`'s guard ordering) — see
//! `.planning/phases/17-model-export/onnx-export/{research.md,SPEC.md}` for the
//! line-anchored upstream citations this module's structure is checked against.
//!
//! # Guard (EXPORT-01a)
//!
//! `cb_model::Model` has exactly two split variants (`ModelSplit::Float`,
//! `ModelSplit::Ctr`) and no distinct one-hot-categorical / text / embedding
//! representation, so "float-only AND oblivious" reduces to ONE structural
//! predicate ([`is_onnx_exportable`]) rather than upstream's four separate
//! `HasCategoricalFeatures`/`HasTextFeatures`/`HasEmbeddingFeatures`/
//! `IsOblivious` checks. The guard runs to completion BEFORE any byte is
//! written (upstream's own check-before-build ordering) — a rejected model
//! never leaves a partial file at the target path.
//!
//! # Reversed split-order tree walk (EXPORT-01b) — the load-bearing pitfall
//!
//! This port's [`ObliviousTree::leaf_values`] is stored in canonical
//! FORWARD-bit-order: split `i` contributes bit `i` of the leaf index (the
//! LOWEST-index split is evaluated closest to the leaves,
//! `crate::apply::binarize_feature` doc). ONNX's `TreeEnsembleRegressor`/
//! `TreeEnsembleClassifier` instead walk a COMPLETE BINARY TREE from a root at
//! depth 0, so depth `d`'s split must be `splits[len - 1 - d]` — the LAST split
//! is the ONNX root, the FIRST split is the deepest internal level
//! (`onnx_helpers.cpp AddTree`). [`build_tree_nodes`] implements this via a
//! flat node-id enumeration (`node_id`'s ONNX depth = `(node_id + 1).ilog2()`)
//! rather than an explicit recursive walk, which happens to make the
//! leaf-node-id enumeration order coincide EXACTLY with this port's
//! forward-bit-order `leaf_values` — no permutation is ever needed (verified
//! by hand-computation in `onnx_test.rs`, AT-01b-1/AT-01b-2).
//!
//! # Multiclass leaf-value indexing (EXPORT-01d)
//!
//! `ObliviousTree::leaf_values` is DIMENSION-MAJOR
//! (`leaf_values[class * n_leaves + leaf]`, `crate::model::Model::approx_dimension`
//! doc) — NOT upstream's own leaf-major layout. [`build_classifier_nodes`]
//! reads the dimension-major formula directly; reusing the single-dimension
//! leaf transcription from [`build_tree_nodes`] unmodified for `dim > 1` would
//! silently transpose leaf/class (SPEC.md EXPORT-01d).

use std::path::Path;

use crate::model::{Model, ModelSplit, ObliviousTree};
use crate::onnx_generated as onnx;

/// Typed failure at the ONNX-export boundary (no panic, no unwrap, no raw
/// indexing — workspace-denied restriction lints).
#[derive(Debug, thiserror::Error)]
pub enum OnnxExportError {
    /// The model contains at least one CTR split, or carries baked `ctr_data`
    /// — upstream's `HasCategoricalFeatures`-equivalent guard for this port's
    /// data model (a CTR split is the ONLY categorical-derived construct
    /// [`Model`] can represent).
    #[error("model uses categorical/CTR features, which ONNX export does not support")]
    CategoricalFeaturesUnsupported,

    /// The model has at least one non-symmetric (Lossguide/Depthwise) tree —
    /// upstream's `IsOblivious()` guard.
    #[error("model contains non-symmetric (Lossguide/Depthwise) trees, which ONNX export does not support")]
    NonObliviousTreesUnsupported,

    /// The model has at least one region-path tree — upstream's
    /// `IsOblivious()` guard (Region trees are a separate, non-oblivious
    /// variant in this port's `TreeVariant`).
    #[error("model contains region-path trees, which ONNX export does not support")]
    RegionTreesUnsupported,

    /// Failed to encode the built ONNX graph to protobuf bytes.
    #[error("ONNX protobuf encode error: {0}")]
    Encode(#[from] prost::EncodeError),

    /// Underlying I/O error while writing the `.onnx` file.
    #[error("ONNX export I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// `model.class_to_label` contains a value that is not exactly
    /// representable as an `i64` (non-finite, has a fractional part, or is
    /// out of `i64` range). Casting such a value with `as i64` silently
    /// truncates (e.g. `1.5 -> 1`), which can collide two distinct original
    /// labels onto the same ONNX `classlabels_int64s` entry and silently
    /// mis-map predictions to the wrong class — so this is rejected rather
    /// than truncated.
    #[error("model.class_to_label contains a non-integer value, which ONNX export does not support")]
    NonIntegerClassLabelsUnsupported,
}

/// EXPORT-01a guard: reject a model this exporter cannot represent, in the
/// deterministic check order SPEC.md §4/§5 specifies — (1) non-symmetric tree
/// present, (2) region tree present, (3) CTR split or baked `ctr_data`
/// present, (4) otherwise `Ok(())`. Pure; no I/O, no protobuf.
fn is_onnx_exportable(model: &Model) -> Result<(), OnnxExportError> {
    if !model.non_symmetric_trees.is_empty() {
        return Err(OnnxExportError::NonObliviousTreesUnsupported);
    }
    if !model.region_trees.is_empty() {
        return Err(OnnxExportError::RegionTreesUnsupported);
    }
    let has_ctr_split = model
        .oblivious_trees
        .iter()
        .flat_map(|tree| tree.splits.iter())
        .any(|split| matches!(split, ModelSplit::Ctr(_)));
    if model.ctr_data.is_some() || has_ctr_split {
        return Err(OnnxExportError::CategoricalFeaturesUnsupported);
    }
    Ok(())
}

/// EXPORT-01d classifier-only guard: every `model.class_to_label` entry must
/// be exactly representable as an `i64` before [`build_classifier_nodes`]
/// casts it into `classlabels_int64s`. Not part of [`is_onnx_exportable`]
/// because `class_to_label` is only ever read on the classifier export path
/// (a regressor export never touches it) — [`export_onnx`] calls this
/// separately, gated on `is_classifier`, but still BEFORE any node/graph is
/// built (same check-before-build ordering as the core guard). Pure; no I/O,
/// no protobuf.
fn validate_class_to_label(model: &Model) -> Result<(), OnnxExportError> {
    for &v in &model.class_to_label {
        let in_i64_range = v >= i64::MIN as f64 && v <= i64::MAX as f64;
        if !v.is_finite() || v.fract() != 0.0 || !in_i64_range {
            return Err(OnnxExportError::NonIntegerClassLabelsUnsupported);
        }
    }
    Ok(())
}

/// `2^k`, saturating instead of overflow-panicking for a pathological `k`
/// (unreachable for any real tree depth, but keeps this checked rather than a
/// bare `1 << k`).
fn pow2(k: usize) -> usize {
    let exp = u32::try_from(k).unwrap_or(u32::MAX);
    1usize.checked_shl(exp).unwrap_or(usize::MAX)
}

/// EXPORT-01b per-tree ONNX node-array fragment: the seven parallel
/// `nodes_*` arrays (in ONNX complete-binary-tree node-id order) plus, for
/// each LEAF node (in the same node-id order, which coincides with this
/// port's forward-bit-order `leaf_values`), that leaf's own node id and its
/// (single-dimension) leaf value. `tree_ids` is `tree_id` repeated once per
/// node, matching every other array's length.
struct TreeNodeFragment {
    tree_ids: Vec<i64>,
    node_ids: Vec<i64>,
    feature_ids: Vec<i64>,
    modes: Vec<String>,
    values: Vec<f64>,
    true_node_ids: Vec<i64>,
    false_node_ids: Vec<i64>,
    /// Node id of each leaf, in increasing node-id order (== forward-bit-order
    /// leaf index — see module doc).
    leaf_node_ids: Vec<i64>,
    /// `tree.leaf_values` verbatim, in the SAME order as `leaf_node_ids`
    /// (single-dimension read; the multiclass path re-derives per-class values
    /// separately, see [`build_classifier_nodes`]).
    leaf_values: Vec<f64>,
}

/// EXPORT-01b: transcribe one [`ObliviousTree`] into a [`TreeNodeFragment`].
///
/// Enumerates the complete-binary-tree node ids `0..(2^(k+1) - 1)` directly
/// (no recursion): for internal node `i` (`i < 2^k - 1`), its ONNX depth is
/// `(i + 1).ilog2()`, and depth `d` reads `tree.splits[k - 1 - d]` (the
/// REVERSED index — module doc). Every node at that depth shares the SAME
/// split (an oblivious tree tests one split per level), so no per-node lookup
/// beyond the depth is needed. The remaining `2^k` node ids (in increasing
/// order) are the leaves, and — because ONNX's false-child-first
/// (`2i+1`/`2i+2`) numbering enumerates leaves in exactly the same order this
/// port's forward-bit-order leaf index does — leaf `j`'s node id holds
/// `tree.leaf_values[j]` verbatim, with NO permutation.
fn build_tree_nodes(tree: &ObliviousTree, tree_id: i64) -> TreeNodeFragment {
    let k = tree.splits.len();
    let n_leaves = pow2(k);
    let n_internal = n_leaves.saturating_sub(1);
    let total_nodes = n_internal.saturating_add(n_leaves);

    let mut frag = TreeNodeFragment {
        tree_ids: Vec::with_capacity(total_nodes),
        node_ids: Vec::with_capacity(total_nodes),
        feature_ids: Vec::with_capacity(total_nodes),
        modes: Vec::with_capacity(total_nodes),
        values: Vec::with_capacity(total_nodes),
        true_node_ids: Vec::with_capacity(total_nodes),
        false_node_ids: Vec::with_capacity(total_nodes),
        leaf_node_ids: Vec::with_capacity(n_leaves),
        leaf_values: Vec::with_capacity(n_leaves),
    };

    for node_id in 0..total_nodes {
        let node_id_i64 = i64::try_from(node_id).unwrap_or(i64::MAX);
        frag.tree_ids.push(tree_id);
        frag.node_ids.push(node_id_i64);

        if node_id < n_internal {
            // ONNX depth of this internal node (0 == root).
            let depth = (node_id.saturating_add(1)).ilog2() as usize;
            // Reversed split-order: depth d reads splits[k - 1 - d] (module doc).
            let split = k
                .checked_sub(1)
                .and_then(|m| m.checked_sub(depth))
                .and_then(|idx| tree.splits.get(idx))
                .and_then(ModelSplit::as_float);
            let (feature, border) = match split {
                Some(s) => (s.feature, s.border),
                // Unreachable for a guard-passed (float-only) model; defensive
                // zero rather than a panic (workspace indexing_slicing deny).
                None => (0, 0.0),
            };
            frag.feature_ids
                .push(i64::try_from(feature).unwrap_or(i64::MAX));
            frag.modes.push("BRANCH_GT".to_owned());
            frag.values.push(border);
            frag.false_node_ids
                .push(i64::try_from(2 * node_id + 1).unwrap_or(i64::MAX));
            frag.true_node_ids
                .push(i64::try_from(2 * node_id + 2).unwrap_or(i64::MAX));
        } else {
            let leaf_idx = node_id - n_internal;
            frag.feature_ids.push(0);
            frag.modes.push("LEAF".to_owned());
            frag.values.push(0.0);
            frag.false_node_ids.push(0);
            frag.true_node_ids.push(0);
            frag.leaf_node_ids.push(node_id_i64);
            frag.leaf_values
                .push(tree.leaf_values.get(leaf_idx).copied().unwrap_or(0.0));
        }
    }
    frag
}

fn attr_int(name: &str, value: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_owned(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

fn attr_ints(name: &str, values: Vec<i64>) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_owned(),
        r#type: onnx::attribute_proto::AttributeType::Ints as i32,
        ints: values,
        ..Default::default()
    }
}

/// `AttributeProto.floats` is `f32` (ONNX tree-ensemble attributes are
/// single-precision — the "relaxed, export-specific float32 tolerance" the
/// roadmap locks for this whole phase); this is the single f64->f32 cast
/// point for every FLOATS attribute this module emits.
fn attr_floats(name: &str, values: &[f64]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_owned(),
        r#type: onnx::attribute_proto::AttributeType::Floats as i32,
        floats: values.iter().map(|&v| v as f32).collect(),
        ..Default::default()
    }
}

fn attr_string(name: &str, value: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_owned(),
        r#type: onnx::attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn attr_strings(name: &str, values: &[String]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_owned(),
        r#type: onnx::attribute_proto::AttributeType::Strings as i32,
        strings: values.iter().map(|s| s.as_bytes().to_vec()).collect(),
        ..Default::default()
    }
}

/// The seven `nodes_*` arrays shared VERBATIM by [`build_regressor_node`] and
/// [`build_classifier_nodes`] (T7 refactor — both concatenate the EXACT same
/// per-tree structural fragment across `model.oblivious_trees`; only the
/// per-leaf target/class contribution arrays differ between the two, so only
/// this shared piece is factored out).
#[derive(Default)]
struct SharedNodeArrays {
    tree_ids: Vec<i64>,
    node_ids: Vec<i64>,
    feature_ids: Vec<i64>,
    modes: Vec<String>,
    values: Vec<f64>,
    true_node_ids: Vec<i64>,
    false_node_ids: Vec<i64>,
}

impl SharedNodeArrays {
    fn extend_from(&mut self, frag: TreeNodeFragment) {
        self.tree_ids.extend(frag.tree_ids);
        self.node_ids.extend(frag.node_ids);
        self.feature_ids.extend(frag.feature_ids);
        self.modes.extend(frag.modes);
        self.values.extend(frag.values);
        self.true_node_ids.extend(frag.true_node_ids);
        self.false_node_ids.extend(frag.false_node_ids);
    }

    fn into_attrs(self) -> [onnx::AttributeProto; 7] {
        [
            attr_ints("nodes_treeids", self.tree_ids),
            attr_ints("nodes_nodeids", self.node_ids),
            attr_ints("nodes_featureids", self.feature_ids),
            attr_strings("nodes_modes", &self.modes),
            attr_floats("nodes_values", &self.values),
            attr_ints("nodes_truenodeids", self.true_node_ids),
            attr_ints("nodes_falsenodeids", self.false_node_ids),
        ]
    }
}

/// EXPORT-01c: assemble every tree in `model.oblivious_trees` (boosting
/// order) into one `TreeEnsembleRegressor` [`onnx::NodeProto`]. `base_values`
/// is emitted ONLY when `model.bias != 0.0` (upstream's `TTreesAttributes`
/// conditional allocation — a zero-bias model omits the attribute entirely,
/// never emits `[0.0]`).
fn build_regressor_node(model: &Model) -> onnx::NodeProto {
    let mut shared = SharedNodeArrays::default();
    let mut target_treeids = Vec::new();
    let mut target_nodeids = Vec::new();
    let mut target_ids = Vec::new();
    let mut target_weights = Vec::new();

    for (tree_id, tree) in model.oblivious_trees.iter().enumerate() {
        let tid = i64::try_from(tree_id).unwrap_or(i64::MAX);
        let frag = build_tree_nodes(tree, tid);
        for (leaf_node_id, leaf_value) in frag.leaf_node_ids.iter().zip(frag.leaf_values.iter()) {
            target_treeids.push(tid);
            target_nodeids.push(*leaf_node_id);
            target_ids.push(0i64);
            target_weights.push(*leaf_value);
        }
        shared.extend_from(frag);
    }

    let mut attrs = shared.into_attrs().to_vec();
    attrs.extend([
        attr_ints("target_treeids", target_treeids),
        attr_ints("target_nodeids", target_nodeids),
        attr_ints("target_ids", target_ids),
        attr_floats("target_weights", &target_weights),
        attr_int("n_targets", 1),
        attr_string("post_transform", "NONE"),
    ]);
    if model.bias != 0.0 {
        attrs.push(attr_floats("base_values", &[model.bias]));
    }

    onnx::NodeProto {
        op_type: "TreeEnsembleRegressor".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        input: vec!["features".to_owned()],
        output: vec!["predictions".to_owned()],
        attribute: attrs,
        ..Default::default()
    }
}

/// EXPORT-01d: assemble every tree into a `TreeEnsembleClassifier`
/// [`onnx::NodeProto`] plus a sibling `ZipMap` node consuming its probability
/// output. `post_transform` is `LOGISTIC` at `approx_dimension == 1`,
/// `SOFTMAX` otherwise. Per-class leaf contributions are read via the
/// DIMENSION-MAJOR formula `tree.leaf_values[class * n_leaves + leaf]`
/// (module doc) — at `dim == 1` this coincides with
/// [`build_tree_nodes`]'s single-dimension read, so the binary path needs no
/// special case. The binary case's `base_values` (when `bias != 0.0`) is the
/// upstream asymmetric pair `[-bias, +bias]`, NOT the regressor's single-value
/// form; multiclass `base_values` is omitted (this port's `Model` carries only
/// a scalar dim-0 bias, no per-class bias vector — see `apply.rs`'s
/// `predict_raw_multi` WR-03 note for the same limitation on the apply side).
fn build_classifier_nodes(model: &Model) -> (onnx::NodeProto, onnx::NodeProto) {
    let dim = model.approx_dimension.max(1);

    let mut shared = SharedNodeArrays::default();
    let mut class_treeids = Vec::new();
    let mut class_nodeids = Vec::new();
    let mut class_ids = Vec::new();
    let mut class_weights = Vec::new();

    for (tree_id, tree) in model.oblivious_trees.iter().enumerate() {
        let tid = i64::try_from(tree_id).unwrap_or(i64::MAX);
        let frag = build_tree_nodes(tree, tid);

        let n_leaves = frag.leaf_node_ids.len();
        for (leaf_pos, &leaf_node_id) in frag.leaf_node_ids.iter().enumerate() {
            for class in 0..dim {
                // Dimension-major: leaf_values[class * n_leaves + leaf] (module doc).
                let idx = class.saturating_mul(n_leaves).saturating_add(leaf_pos);
                let value = tree.leaf_values.get(idx).copied().unwrap_or(0.0);
                class_treeids.push(tid);
                class_nodeids.push(leaf_node_id);
                class_ids.push(i64::try_from(class).unwrap_or(i64::MAX));
                class_weights.push(value);
            }
        }
        shared.extend_from(frag);
    }

    // No stored class_to_label (e.g. dropped by a loader that only keeps
    // numeric label entries): default to 0..n_labels, where n_labels is the
    // SAME "binary is width-2" rule `export_onnx` uses for the
    // probability_tensor's second dim (dim==1 binary models emit class_ids
    // 0 ONLY but still need a 2-entry label set for the implicit
    // 1-p(class 0) complement; dim>1 multiclass models need exactly `dim`
    // entries, one per class_id actually emitted by the loop above). NOT a
    // hardcoded 2-entry vec unconditionally, which under-covers dim > 2.
    let n_labels = if dim == 1 { 2 } else { dim };
    let classlabels: Vec<i64> = if model.class_to_label.is_empty() {
        (0..n_labels as i64).collect()
    } else {
        // Safe: callers must run `validate_class_to_label` first (see
        // `export_onnx`), which rejects any non-integer-representable entry.
        model.class_to_label.iter().map(|&v| v as i64).collect()
    };
    let post_transform = if dim == 1 { "LOGISTIC" } else { "SOFTMAX" };

    let mut attrs = shared.into_attrs().to_vec();
    attrs.extend([
        attr_ints("class_treeids", class_treeids),
        attr_ints("class_nodeids", class_nodeids),
        attr_ints("class_ids", class_ids),
        attr_floats("class_weights", &class_weights),
        attr_ints("classlabels_int64s", classlabels.clone()),
        attr_string("post_transform", post_transform),
    ]);
    if dim == 1 && model.bias != 0.0 {
        // The binary-classifier asymmetric-bias trick (class_ids 0 and 1),
        // NOT the regressor's single-value form.
        attrs.push(attr_floats("base_values", &[-model.bias, model.bias]));
    }

    let classifier_node = onnx::NodeProto {
        op_type: "TreeEnsembleClassifier".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        input: vec!["features".to_owned()],
        output: vec!["label".to_owned(), "probability_tensor".to_owned()],
        attribute: attrs,
        ..Default::default()
    };
    let zipmap_node = onnx::NodeProto {
        op_type: "ZipMap".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        input: vec!["probability_tensor".to_owned()],
        output: vec!["probabilities".to_owned()],
        attribute: vec![attr_ints("classlabels_int64s", classlabels)],
        ..Default::default()
    };
    (classifier_node, zipmap_node)
}

fn dim_param() -> onnx::tensor_shape_proto::Dimension {
    onnx::tensor_shape_proto::Dimension {
        value: Some(onnx::tensor_shape_proto::dimension::Value::DimParam(
            "N".to_owned(),
        )),
        ..Default::default()
    }
}

fn dim_value(v: i64) -> onnx::tensor_shape_proto::Dimension {
    onnx::tensor_shape_proto::Dimension {
        value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)),
        ..Default::default()
    }
}

fn tensor_type(
    elem_type: onnx::tensor_proto::DataType,
    dims: Vec<onnx::tensor_shape_proto::Dimension>,
) -> onnx::TypeProto {
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
            elem_type: elem_type as i32,
            shape: Some(onnx::TensorShapeProto { dim: dims }),
        })),
        ..Default::default()
    }
}

/// `seq(map(int64, float))` — the `ZipMap` output type (upstream's
/// probabilities graph output for an int64-labeled classifier).
fn sequence_of_int64_float_map_type() -> onnx::TypeProto {
    let value_type = onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
            elem_type: onnx::tensor_proto::DataType::Float as i32,
            shape: None,
        })),
        ..Default::default()
    };
    let map_type = onnx::TypeProto {
        value: Some(onnx::type_proto::Value::MapType(Box::new(
            onnx::type_proto::Map {
                key_type: onnx::tensor_proto::DataType::Int64 as i32,
                value_type: Some(Box::new(value_type)),
            },
        ))),
        ..Default::default()
    };
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::SequenceType(Box::new(
            onnx::type_proto::Sequence {
                elem_type: Some(Box::new(map_type)),
            },
        ))),
        ..Default::default()
    }
}

/// EXPORT-01e: export `model` to a well-formed ONNX file at `path`.
///
/// `is_classifier` selects `TreeEnsembleClassifier`+`ZipMap`
/// (`post_transform="LOGISTIC"` for `approx_dimension==1`, `"SOFTMAX"` for
/// `approx_dimension>1`) when `true`, or `TreeEnsembleRegressor`
/// (`post_transform="NONE"`) when `false`. The caller supplies this because
/// [`Model`] carries no loss-function/objective metadata to infer it from.
///
/// The guard ([`is_onnx_exportable`]) runs to completion BEFORE any byte is
/// written — a rejected model never leaves a partial file at `path`.
///
/// # Errors
/// [`OnnxExportError::CategoricalFeaturesUnsupported`] /
/// [`OnnxExportError::NonObliviousTreesUnsupported`] /
/// [`OnnxExportError::RegionTreesUnsupported`] if the guard rejects `model`;
/// [`OnnxExportError::NonIntegerClassLabelsUnsupported`] if `is_classifier`
/// and `model.class_to_label` contains a value that would silently truncate
/// when cast to `i64`; [`OnnxExportError::Encode`] / [`OnnxExportError::Io`]
/// on a downstream failure. Never panics.
pub fn export_onnx(model: &Model, path: &Path, is_classifier: bool) -> Result<(), OnnxExportError> {
    is_onnx_exportable(model)?;
    if is_classifier {
        validate_class_to_label(model)?;
    }

    let n_float = i64::try_from(model.float_feature_borders.len()).unwrap_or(i64::MAX);
    let features_input = onnx::ValueInfoProto {
        name: "features".to_owned(),
        r#type: Some(tensor_type(
            onnx::tensor_proto::DataType::Float,
            vec![dim_param(), dim_value(n_float)],
        )),
        ..Default::default()
    };

    let (nodes, graph_outputs, value_info) = if is_classifier {
        let (classifier_node, zipmap_node) = build_classifier_nodes(model);
        let dim = model.approx_dimension.max(1);
        // Upstream declares the probability tensor's second dim as
        // `dims == 1 ? 2 : dims` — even a binary model is declared width-2
        // (module doc / SPEC EXPORT-01d output-shape note).
        let second_dim = if dim == 1 { 2 } else { dim };
        let probability_value_info = onnx::ValueInfoProto {
            name: "probability_tensor".to_owned(),
            r#type: Some(tensor_type(
                onnx::tensor_proto::DataType::Float,
                vec![dim_param(), dim_value(i64::try_from(second_dim).unwrap_or(i64::MAX))],
            )),
            ..Default::default()
        };
        let label_output = onnx::ValueInfoProto {
            name: "label".to_owned(),
            r#type: Some(tensor_type(
                onnx::tensor_proto::DataType::Int64,
                vec![dim_param()],
            )),
            ..Default::default()
        };
        let probabilities_output = onnx::ValueInfoProto {
            name: "probabilities".to_owned(),
            r#type: Some(sequence_of_int64_float_map_type()),
            ..Default::default()
        };
        (
            vec![classifier_node, zipmap_node],
            vec![label_output, probabilities_output],
            vec![probability_value_info],
        )
    } else {
        let regressor_node = build_regressor_node(model);
        let predictions_output = onnx::ValueInfoProto {
            name: "predictions".to_owned(),
            r#type: Some(tensor_type(
                onnx::tensor_proto::DataType::Float,
                vec![dim_param(), dim_value(1)],
            )),
            ..Default::default()
        };
        (vec![regressor_node], vec![predictions_output], vec![])
    };

    let graph = onnx::GraphProto {
        node: nodes,
        name: "CatBoostModel".to_owned(),
        input: vec![features_input],
        output: graph_outputs,
        value_info,
        ..Default::default()
    };

    let model_proto = onnx::ModelProto {
        ir_version: 3,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: "ai.onnx.ml".to_owned(),
            version: 2,
        }],
        producer_name: "catboost-rs".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        graph: Some(graph),
        ..Default::default()
    };

    let mut buf = Vec::new();
    prost::Message::encode(&model_proto, &mut buf)?;
    std::fs::write(path, &buf)?;
    Ok(())
}

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md —
// no test body in this production file). Mirrors ctr_data.rs:58-61.
#[cfg(test)]
#[path = "onnx_test.rs"]
mod tests;
