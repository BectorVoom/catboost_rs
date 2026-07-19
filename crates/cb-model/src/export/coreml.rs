//! CoreML export (EXPORT-02 / Phase 17): a float-only, oblivious, scalar
//! [`Model`] to a well-formed Apple CoreML `.mlmodel`
//! (`CoreML.Specification.Model` → `treeEnsembleRegressor`) protobuf.
//!
//! # Source of truth
//!
//! The emitted schema + node layout are the REAL CoreML schema and CatBoost's
//! OWN `.mlmodel` node numbering, extracted live on this host from
//! `coremltools==9.0` and CatBoost 1.2.10's `save_model(format="coreml")` — see
//! `.planning/plans/coreml-export/IMPLEMENTATION_NOTES.md` §1/§2/§3/§4, the
//! authoritative addendum this module is checked against (the oracle in
//! `tests/coreml_export_test.rs` diffs the emitted structure against CatBoost's
//! own reference, so any layout divergence fails a test rather than silently
//! shipping a semantically-wrong file).
//!
//! # Guard (CM-01)
//!
//! `cb_model::Model` reduces "float-only AND oblivious AND scalar" to a small
//! set of structural predicates ([`is_coreml_exportable`]): no non-symmetric
//! tree, no region tree, no CTR split / baked `ctr_data`, and
//! `approx_dimension == 1` (regressor-first — a multi-dimension model is
//! rejected rather than silently flattened). The guard runs to completion
//! BEFORE any byte is written — a rejected model never leaves a partial file at
//! the target path (mirrors [`crate::export::export_onnx`]'s check-before-build
//! ordering).
//!
//! # Node layout (CM-02) — replicates CatBoost EXACTLY
//!
//! For a scalar oblivious tree with `k` splits (`2^k` leaves, `2^k - 1`
//! internal nodes) CatBoost numbers nodes **leaves-first, internal bottom-up,
//! root last** (IMPLEMENTATION_NOTES §2):
//!
//! * Leaf nodes: `node_id = 0 .. 2^k - 1`; leaf `node_id` carries
//!   `evaluation_info = [{0, leaf_values[node_id]}]`. This is the canonical
//!   forward-bit leaf index — [`ObliviousTree::leaf_values`] maps DIRECTLY, no
//!   permutation (the same invariant the ONNX exporter relies on).
//! * Internal nodes: numbered by DECREASING depth. Level `L` (`L = 0` the root)
//!   holds `2^L` nodes with first id `firstId(L) = 2^(k+1) - 2^(L+1)`; the
//!   deepest internal level `k-1` starts at `2^k`, the root is the single node
//!   `2^(k+1) - 2`.
//! * Level `L` tests `splits[k-1-L]` (root tests the HIGHEST-index split, the
//!   deepest internal level tests `splits[0]`), `node_behavior =
//!   BranchOnValueGreaterThan`. Children of the node at level `L`, position `p`:
//!   if `L == k-1` (children are leaves) `false = 2p`, `true = 2p+1`; else
//!   `false = firstId(L+1) + 2p`, `true = firstId(L+1) + 2p + 1`.
//!
//! A depth-0 tree (`k == 0`, a single leaf) emits ONE leaf node and NO internal
//! nodes (the `k-1` math is never reached).

use std::path::Path;

use crate::coreml_generated as cm;
use crate::model::{Model, ModelSplit, ObliviousTree};

/// Typed failure at the CoreML-export boundary (no panic, no unwrap, no raw
/// indexing — workspace-denied restriction lints), mirroring
/// [`crate::export::OnnxExportError`].
#[derive(Debug, thiserror::Error)]
pub enum CoreMlExportError {
    /// The model contains at least one CTR split, or carries baked `ctr_data` —
    /// the `HasCategoricalFeatures`-equivalent guard for this port's data model
    /// (a CTR split is the ONLY categorical-derived construct [`Model`] can
    /// represent).
    #[error("model uses categorical/CTR features, which CoreML export does not support")]
    CategoricalFeaturesUnsupported,

    /// The model has at least one non-symmetric (Lossguide/Depthwise) tree —
    /// upstream's `IsOblivious()` guard.
    #[error("model contains non-symmetric (Lossguide/Depthwise) trees, which CoreML export does not support")]
    NonObliviousTreesUnsupported,

    /// The model has at least one region-path tree — upstream's `IsOblivious()`
    /// guard (Region trees are a separate, non-oblivious variant in this port's
    /// `TreeVariant`).
    #[error("model contains region-path trees, which CoreML export does not support")]
    RegionTreesUnsupported,

    /// The model has `approx_dimension > 1` (multiclass / multilabel /
    /// MultiQuantile). This first slice exports a SCALAR regressor only; a
    /// multi-dimension model would need a per-class leaf channel this exporter
    /// does not emit, so it is rejected rather than silently flattened.
    #[error("model is multi-dimensional (approx_dimension > 1), which CoreML export does not support")]
    MultiDimUnsupported,

    /// Failed to encode the built CoreML model to protobuf bytes.
    #[error("CoreML protobuf encode error: {0}")]
    Encode(#[from] prost::EncodeError),

    /// Underlying I/O error while writing the `.mlmodel` file.
    #[error("CoreML export I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// CM-01 guard: reject a model this exporter cannot represent, in the
/// deterministic order (1) non-symmetric tree present, (2) region tree present,
/// (3) CTR split or baked `ctr_data` present, (4) `approx_dimension > 1`,
/// otherwise `Ok(())`. Pure; no I/O, no protobuf. Mirrors
/// [`crate::export::onnx`]'s guard order + the extra multi-dim check.
fn is_coreml_exportable(model: &Model) -> Result<(), CoreMlExportError> {
    if !model.non_symmetric_trees.is_empty() {
        return Err(CoreMlExportError::NonObliviousTreesUnsupported);
    }
    if !model.region_trees.is_empty() {
        return Err(CoreMlExportError::RegionTreesUnsupported);
    }
    let has_ctr_split = model
        .oblivious_trees
        .iter()
        .flat_map(|tree| tree.splits.iter())
        .any(|split| matches!(split, ModelSplit::Ctr(_)));
    if model.ctr_data.is_some() || has_ctr_split {
        return Err(CoreMlExportError::CategoricalFeaturesUnsupported);
    }
    if model.approx_dimension > 1 {
        return Err(CoreMlExportError::MultiDimUnsupported);
    }
    Ok(())
}

/// `2^n` as `u64`, saturating instead of overflow-panicking for a pathological
/// `n` (unreachable for any real tree depth, but keeps this checked rather than
/// a bare `1 << n`).
fn pow2_u64(n: usize) -> u64 {
    let exp = u32::try_from(n).unwrap_or(u32::MAX);
    1u64.checked_shl(exp).unwrap_or(u64::MAX)
}

/// `2^n` as `usize`, saturating (leaf-count / loop-bound arithmetic).
fn pow2_usize(n: usize) -> usize {
    let exp = u32::try_from(n).unwrap_or(u32::MAX);
    1usize.checked_shl(exp).unwrap_or(usize::MAX)
}

/// `firstId(L) = 2^(k+1) - 2^(L+1)` — the first `node_id` of internal level `L`
/// for a `k`-split tree (IMPLEMENTATION_NOTES §2). Saturating arithmetic.
fn first_internal_id(k: usize, level: usize) -> u64 {
    let base = pow2_u64(k.saturating_add(1));
    let block = pow2_u64(level.saturating_add(1));
    base.saturating_sub(block)
}

/// CM-02: transcribe one [`ObliviousTree`] into its CoreML nodes, emitted in
/// ascending `node_id` order (leaves `0..2^k`, then internal levels `k-1` down
/// to `0` — i.e. still ascending `node_id`, since a deeper level has the lower
/// first id). Replicates CatBoost's numbering EXACTLY (module doc / §2).
fn build_tree_nodes(tree: &ObliviousTree, tree_id: u64) -> Vec<cm::TreeNode> {
    let k = tree.splits.len();
    let n_leaves = pow2_usize(k);
    let n_internal = n_leaves.saturating_sub(1);
    let mut nodes: Vec<cm::TreeNode> = Vec::with_capacity(n_leaves.saturating_add(n_internal));

    // Leaves: node_id 0..2^k-1, evaluation_value == leaf_values[node_id].
    for leaf in 0..n_leaves {
        let leaf_value = tree.leaf_values.get(leaf).copied().unwrap_or(0.0);
        nodes.push(cm::TreeNode {
            tree_id,
            node_id: u64::try_from(leaf).unwrap_or(u64::MAX),
            node_behavior: cm::TreeNodeBehavior::LeafNode as i32,
            evaluation_info: vec![cm::EvaluationInfo {
                evaluation_index: 0,
                evaluation_value: leaf_value,
            }],
            ..Default::default()
        });
    }

    // Internal nodes: level L = k-1 down to 0 (ascending node_id blocks).
    for level in (0..k).rev() {
        let first_id = first_internal_id(k, level);
        // Level L tests splits[k-1-L] (reversed split order — module doc).
        let split_idx = k
            .checked_sub(1)
            .and_then(|m| m.checked_sub(level));
        let split = split_idx
            .and_then(|idx| tree.splits.get(idx))
            .and_then(ModelSplit::as_float);
        let (feature, border) = match split {
            Some(s) => (s.feature, s.border),
            // Unreachable for a guard-passed (float-only) model; defensive
            // zero rather than a panic (workspace indexing_slicing deny).
            None => (0, 0.0),
        };
        let feature_index = u64::try_from(feature).unwrap_or(u64::MAX);

        let n_at_level = pow2_usize(level);
        let is_leaf_parent = level == k.saturating_sub(1);
        let children_base = if is_leaf_parent {
            0
        } else {
            first_internal_id(k, level.saturating_add(1))
        };
        for p in 0..n_at_level {
            let node_id = first_id.saturating_add(u64::try_from(p).unwrap_or(u64::MAX));
            let two_p = u64::try_from(p)
                .unwrap_or(u64::MAX)
                .saturating_mul(2);
            let false_child = children_base.saturating_add(two_p);
            let true_child = false_child.saturating_add(1);
            nodes.push(cm::TreeNode {
                tree_id,
                node_id,
                node_behavior: cm::TreeNodeBehavior::BranchOnValueGreaterThan as i32,
                branch_feature_index: feature_index,
                branch_feature_value: border,
                true_child_node_id: true_child,
                false_child_node_id: false_child,
                missing_value_tracks_true_child: false,
                ..Default::default()
            });
        }
    }

    nodes
}

/// CM-02: assemble every tree in `model.oblivious_trees` (boosting order) into
/// one [`cm::TreeEnsembleRegressor`]. `base_prediction_value` carries
/// `[model.bias]` UNCONDITIONALLY (IMPLEMENTATION_NOTES §3 — CatBoost always
/// emits it, even at bias 0.0, UNLIKE the ONNX exporter's `bias != 0.0`
/// conditional, so the oracle diff against CatBoost's reference stays clean).
fn build_regressor(model: &Model) -> cm::TreeEnsembleRegressor {
    let mut nodes: Vec<cm::TreeNode> = Vec::new();
    for (tree_index, tree) in model.oblivious_trees.iter().enumerate() {
        let tree_id = u64::try_from(tree_index).unwrap_or(u64::MAX);
        nodes.extend(build_tree_nodes(tree, tree_id));
    }

    let params = cm::TreeEnsembleParameters {
        nodes,
        num_prediction_dimensions: 1,
        base_prediction_value: vec![model.bias],
    };

    cm::TreeEnsembleRegressor {
        tree_ensemble: Some(params),
        post_evaluation_transform: cm::TreeEnsemblePostEvaluationTransform::NoTransform as i32,
    }
}

/// CM-02/CM-03: build the full [`cm::Model`] (description + regressor) for a
/// guard-passed float-only oblivious scalar `model`. One `input`
/// `FeatureDescription` per float feature (`feature_{i}`, `doubleType`), one
/// `output` (`prediction`, `multiArrayType` DOUBLE shape `[1]`),
/// `predicted_feature_name = "prediction"`, `specification_version = 1`
/// (IMPLEMENTATION_NOTES §4). Pure; no I/O.
fn build_model(model: &Model) -> cm::Model {
    let n_float = model.float_feature_borders.len();
    let inputs: Vec<cm::FeatureDescription> = (0..n_float)
        .map(|i| cm::FeatureDescription {
            name: format!("feature_{i}"),
            short_description: String::new(),
            r#type: Some(cm::FeatureType {
                r#type: Some(cm::feature_type::Type::DoubleType(cm::DoubleFeatureType {})),
            }),
        })
        .collect();

    let output = cm::FeatureDescription {
        name: "prediction".to_owned(),
        short_description: String::new(),
        r#type: Some(cm::FeatureType {
            r#type: Some(cm::feature_type::Type::MultiArrayType(cm::ArrayFeatureType {
                shape: vec![1],
                data_type: cm::ArrayDataType::Double as i32,
            })),
        }),
    };

    let description = cm::ModelDescription {
        input: inputs,
        output: vec![output],
        predicted_feature_name: "prediction".to_owned(),
    };

    cm::Model {
        specification_version: 1,
        description: Some(description),
        is_updatable: false,
        r#type: Some(cm::model::Type::TreeEnsembleRegressor(build_regressor(model))),
    }
}

/// CM-03: export `model` to a well-formed CoreML `.mlmodel` file at `path`.
///
/// The guard ([`is_coreml_exportable`]) runs to completion BEFORE any byte is
/// written — a rejected model never leaves a partial file at `path`.
///
/// # Errors
/// [`CoreMlExportError::CategoricalFeaturesUnsupported`] /
/// [`CoreMlExportError::NonObliviousTreesUnsupported`] /
/// [`CoreMlExportError::RegionTreesUnsupported`] /
/// [`CoreMlExportError::MultiDimUnsupported`] if the guard rejects `model`;
/// [`CoreMlExportError::Encode`] / [`CoreMlExportError::Io`] on a downstream
/// failure. Never panics.
pub fn export_coreml(model: &Model, path: &Path) -> Result<(), CoreMlExportError> {
    is_coreml_exportable(model)?;
    let proto = build_model(model);
    let mut buf = Vec::new();
    prost::Message::encode(&proto, &mut buf)?;
    std::fs::write(path, &buf)?;
    Ok(())
}

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md —
// no test body in this production file). Mirrors onnx.rs:635-637.
#[cfg(test)]
#[path = "coreml_test.rs"]
mod tests;
