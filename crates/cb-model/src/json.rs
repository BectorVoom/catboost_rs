//! `model.json` export/import on the upstream catboost schema (MODEL-06, D-04).
//!
//! `save_json` serializes the canonical [`crate::Model`] to the upstream
//! `model.json` shape and `load_json` parses it back. The field names match
//! upstream `json_model_helpers.cpp:160-526` VERBATIM so the export round-trips
//! through `cb_oracle::model_json::load_model_json` — the existing oracle parser
//! doubles as the schema oracle (D-04).
//!
//! # Layout discipline (RESEARCH Pitfalls 2 & 6)
//!
//! - `leaf_weights` is NESTED per tree (`oblivious_trees[i].leaf_weights`), NOT
//!   a flat array like the `.cbm` form (Pitfall 2). Each tree carries its own
//!   `leaf_values` + `leaf_weights` inner array.
//! - `scale_and_bias` is emitted as `[1, [bias]]` (scale 1, single-element bias
//!   vector) — leaf values stay bias-free, the bias term lives only here
//!   (Pitfall 6). `load_json` reads `scale_and_bias[1][0]` for the bias.
//! - `split_index` is a per-tree positional index here (a stable, self-consistent
//!   value); upstream's global split-pool index is reconstructed by `.cbm`, not
//!   needed for the JSON apply/round-trip.
//!
//! # Validation (Security V5, T-04-03-04)
//!
//! `load_json` reads the file then `serde_json::from_str`; `serde_json` is safe
//! by default (no unsafe, no unbounded recursion panic on the shapes here), and
//! every malformed-shape failure maps to a typed [`ModelError`] (the `#[from]
//! serde_json::Error` arm, or [`ModelError::Deserialize`] for a malformed
//! `scale_and_bias`). Nothing panics on hostile JSON. No `unwrap`/raw-index in
//! the production path (workspace deny-lints).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::{Model, ModelSplit, NonSymmetricTree, ObliviousTree, Split};

/// The maximum non-symmetric `"trees"` nested-node recursion depth accepted by
/// the converter (T-06.6-07): a crafted, deeply-nested model file cannot drive
/// unbounded stack recursion. Upstream caps practical tree depth far below this;
/// `64` is well past any real model while bounding the conversion.
const MAX_NON_SYMMETRIC_DEPTH: usize = 64;

/// One split in an oblivious tree (upstream `oblivious_trees[i].splits[j]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SplitJson {
    /// Split border (threshold) on the referenced float feature.
    border: f64,
    /// Index of the float feature this split tests.
    float_feature_index: i64,
    /// Positional split index (self-consistent within the export).
    split_index: i64,
    /// Split kind; numeric-only models emit `"FloatFeature"`.
    split_type: String,
}

/// One oblivious tree: per-tree NESTED `leaf_values` + `leaf_weights` (Pitfall 2)
/// and the ordered `splits`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObliviousTreeJson {
    /// Leaf values in canonical forward-bit order, length `2^depth`.
    leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights, same per-tree nested layout as
    /// `leaf_values` (Pitfall 2). `#[serde(default)]` tolerates older fixtures.
    #[serde(default)]
    leaf_weights: Vec<f64>,
    /// The ordered splits defining this tree's symmetric structure.
    splits: Vec<SplitJson>,
}

/// One node of a non-symmetric (Lossguide / Depthwise) tree in the upstream
/// `"trees"` recursive nested-node schema (FEAT-06, RESEARCH Pitfall 3). A node
/// is EITHER an interior node (`split` + `left` + `right`) OR a leaf (`value` +
/// `weight`). Both arms are `Option` so serde accepts either shape; the
/// converter ([`flatten_non_symmetric`]) discriminates on `split.is_some()`.
///
/// This is a STRUCTURALLY DIFFERENT schema from `oblivious_trees` (flat arrays);
/// it MUST NOT be routed through the [`ObliviousTreeJson`] path (Pitfall 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NonSymmetricNodeJson {
    /// The interior-node split (absent on a leaf).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    split: Option<SplitJson>,
    /// The left child (absent on a leaf). Boxed — the schema is recursive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    left: Option<Box<NonSymmetricNodeJson>>,
    /// The right child (absent on a leaf). Boxed — the schema is recursive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    right: Option<Box<NonSymmetricNodeJson>>,
    /// The leaf value (absent on an interior node).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    value: Option<f64>,
    /// The leaf's summed training-document weight (absent on an interior node).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weight: Option<f64>,
}

/// One float feature's metadata (upstream `features_info.float_features[i]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FloatFeatureJson {
    /// The ascending quantization borders (candidate split thresholds). May be
    /// empty when the model never split on the feature.
    #[serde(default)]
    borders: Vec<f64>,
    /// The feature's string id (empty for unnamed numeric features).
    #[serde(default)]
    feature_id: String,
    /// The feature's index among float features.
    feature_index: i64,
    /// The feature's flat (across-all-types) index; equals `feature_index` for
    /// numeric-only models.
    flat_feature_index: i64,
    /// Whether the feature has NaNs (always `false` for the Phase-4 fixtures).
    #[serde(default)]
    has_nans: bool,
    /// NaN treatment (`"AsIs"` for the Phase-4 numeric models).
    nan_value_treatment: String,
}

/// The `features_info` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeaturesInfoJson {
    /// Per-float-feature metadata in feature order.
    #[serde(default)]
    float_features: Vec<FloatFeatureJson>,
}

/// Top-level upstream `model.json` (the subset we round-trip).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelJsonDoc {
    /// Per-feature metadata, including each float feature's candidate borders.
    features_info: FeaturesInfoJson,
    /// All oblivious trees in boosting (iteration) order. EMPTY for a
    /// non-symmetric model (which populates `trees` instead). `#[serde(default)]`
    /// so a non-symmetric `model.json` (no `oblivious_trees` key) still parses.
    #[serde(default)]
    oblivious_trees: Vec<ObliviousTreeJson>,
    /// All NON-SYMMETRIC (Lossguide / Depthwise) trees in boosting order, in the
    /// recursive nested-node schema (FEAT-06, RESEARCH Pitfall 3 — a DISTINCT
    /// top-level key from `oblivious_trees`). EMPTY for a symmetric model;
    /// `#[serde(default)]` + `skip_serializing_if` so the symmetric export stays
    /// byte-identical (the key is omitted entirely).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    trees: Vec<NonSymmetricNodeJson>,
    /// Upstream `[scale, [bias, …]]`; emitted as `[1, [bias]]` (Pitfall 6).
    scale_and_bias: serde_json::Value,
    /// The upstream `model_info` block — carries `class_params` /
    /// `multiclass_params` for a multiclass model (the SORTED distinct class labels
    /// in `class_to_label`, LOSS-02 / CR-01). `#[serde(default)]` so scalar fixtures
    /// with NO `model_info` still parse; `skip_serializing_if` so a scalar export
    /// (no class labels) omits the key entirely (the D-04 json round-trip stays
    /// byte-identical to the pre-6.2 scalar export).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_info: Option<serde_json::Value>,
}

/// Extract the SORTED distinct class labels (`class_to_label`) from a parsed
/// `model_info` value: read `class_params.class_to_label` (the new generic key,
/// `model.cpp:1431`), falling back to `multiclass_params.class_to_label`. Each
/// label is coerced to `f64` (the canonical `Model::class_to_label` type, matching
/// the trainer-side sorted distinct labels). An absent / malformed block yields an
/// empty vector (a scalar model carries no class labels) — never a panic.
fn class_to_label_from_model_info(model_info: &serde_json::Value) -> Vec<f64> {
    for key in ["class_params", "multiclass_params"] {
        if let Some(arr) = model_info
            .get(key)
            .and_then(|p| p.get("class_to_label"))
            .and_then(serde_json::Value::as_array)
        {
            return arr.iter().filter_map(serde_json::Value::as_f64).collect();
        }
    }
    Vec::new()
}

/// Build a `model_info` value carrying the model's `class_to_label` under the
/// generic `class_params` key (matching upstream `model.cpp:1765`), or `None` when
/// the model has no class labels (a scalar / regression model — the key is then
/// omitted so the scalar json export stays byte-identical, D-04).
fn model_info_from_class_labels(class_to_label: &[f64]) -> Option<serde_json::Value> {
    if class_to_label.is_empty() {
        return None;
    }
    Some(serde_json::json!({
        "class_params": {
            "class_to_label": class_to_label,
        }
    }))
}

/// Flatten one non-symmetric `"trees"` nested-node tree into the canonical
/// [`NonSymmetricTree`] flat triple (`tree_splits` per node, `step_nodes`
/// `(left_diff, right_diff)`, `node_id_to_leaf_id`, `leaf_values`,
/// `leaf_weights`), RESEARCH Pitfall 3.
///
/// Node ids are assigned in a deterministic PRE-ORDER walk (root `0`, then the
/// whole left subtree, then the whole right subtree). For an interior node the
/// stored `(left_diff, right_diff)` are the offsets FROM that node's id to its
/// left / right child ids (matching the upstream `index += diff` pointer-walk);
/// a leaf stores `(0, 0)` and a `node_id_to_leaf_id` entry pointing at its slot
/// in `leaf_values`. The walk is depth-bounded by [`MAX_NON_SYMMETRIC_DEPTH`]
/// (T-06.6-07 — a crafted file cannot drive unbounded recursion).
///
/// # Errors
/// [`ModelError::Deserialize`] on a malformed node (an interior node missing a
/// child / split, a leaf missing its value, or a tree deeper than
/// [`MAX_NON_SYMMETRIC_DEPTH`]).
fn flatten_non_symmetric(root: &NonSymmetricNodeJson) -> Result<NonSymmetricTree, ModelError> {
    // First pass: assign a pre-order id to every node so child-offset diffs are
    // computable. We collect nodes in pre-order; a node's id is its position.
    let mut order: Vec<&NonSymmetricNodeJson> = Vec::new();
    collect_preorder(root, 0, &mut order)?;
    let node_count = order.len();

    // Map every node pointer to its assigned id (pointer identity is stable for
    // the lifetime of `root`).
    let id_of = |target: &NonSymmetricNodeJson| -> Option<usize> {
        order
            .iter()
            .position(|n| std::ptr::eq(*n, target))
    };

    let mut tree_splits: Vec<ModelSplit> = Vec::with_capacity(node_count);
    let mut step_nodes: Vec<(u16, u16)> = Vec::with_capacity(node_count);
    let mut node_id_to_leaf_id: Vec<u32> = vec![0; node_count];
    let mut leaf_values: Vec<f64> = Vec::new();
    let mut leaf_weights: Vec<f64> = Vec::new();

    for (id, node) in order.iter().enumerate() {
        if node.split.is_some() {
            // Interior node: record its float split and child diffs.
            let split = node.split.as_ref().ok_or_else(|| {
                ModelError::Deserialize("non-symmetric interior node missing split".to_owned())
            })?;
            let feature = usize::try_from(split.float_feature_index).map_err(|_| {
                ModelError::Deserialize(format!(
                    "negative float_feature_index {}",
                    split.float_feature_index
                ))
            })?;
            tree_splits.push(ModelSplit::Float(Split {
                feature,
                border: split.border,
            }));
            let left = node.left.as_deref().ok_or_else(|| {
                ModelError::Deserialize("non-symmetric interior node missing left".to_owned())
            })?;
            let right = node.right.as_deref().ok_or_else(|| {
                ModelError::Deserialize("non-symmetric interior node missing right".to_owned())
            })?;
            let left_id = id_of(left).ok_or_else(|| {
                ModelError::Deserialize("non-symmetric left child id not found".to_owned())
            })?;
            let right_id = id_of(right).ok_or_else(|| {
                ModelError::Deserialize("non-symmetric right child id not found".to_owned())
            })?;
            let left_diff = u16::try_from(left_id.saturating_sub(id)).map_err(|_| {
                ModelError::Deserialize("non-symmetric left subtree diff exceeds u16".to_owned())
            })?;
            let right_diff = u16::try_from(right_id.saturating_sub(id)).map_err(|_| {
                ModelError::Deserialize("non-symmetric right subtree diff exceeds u16".to_owned())
            })?;
            step_nodes.push((left_diff, right_diff));
        } else {
            // Leaf node: a no-op split placeholder, terminal step node, and a
            // leaf-id pointing at the next slot in the flat leaf arrays.
            let value = node.value.ok_or_else(|| {
                ModelError::Deserialize("non-symmetric leaf missing value".to_owned())
            })?;
            tree_splits.push(ModelSplit::Float(Split {
                feature: 0,
                border: f64::NEG_INFINITY,
            }));
            step_nodes.push((0, 0));
            let leaf_id = leaf_values.len();
            node_id_to_leaf_id[id] = u32::try_from(leaf_id).map_err(|_| {
                ModelError::Deserialize("non-symmetric leaf id exceeds u32".to_owned())
            })?;
            leaf_values.push(value);
            leaf_weights.push(node.weight.unwrap_or(0.0));
        }
    }

    Ok(NonSymmetricTree {
        tree_splits,
        step_nodes,
        node_id_to_leaf_id,
        leaf_values,
        leaf_weights,
    })
}

/// Pre-order traversal collecting node references in id order (root, then the
/// whole left subtree, then the whole right subtree). Depth-bounded by
/// [`MAX_NON_SYMMETRIC_DEPTH`] (T-06.6-07).
///
/// # Errors
/// [`ModelError::Deserialize`] if `depth` exceeds the bound or an interior node
/// is missing a child.
fn collect_preorder<'a>(
    node: &'a NonSymmetricNodeJson,
    depth: usize,
    out: &mut Vec<&'a NonSymmetricNodeJson>,
) -> Result<(), ModelError> {
    if depth > MAX_NON_SYMMETRIC_DEPTH {
        return Err(ModelError::Deserialize(format!(
            "non-symmetric tree exceeds max depth {MAX_NON_SYMMETRIC_DEPTH}"
        )));
    }
    out.push(node);
    if node.split.is_some() {
        let left = node.left.as_deref().ok_or_else(|| {
            ModelError::Deserialize("non-symmetric interior node missing left".to_owned())
        })?;
        let right = node.right.as_deref().ok_or_else(|| {
            ModelError::Deserialize("non-symmetric interior node missing right".to_owned())
        })?;
        collect_preorder(left, depth + 1, out)?;
        collect_preorder(right, depth + 1, out)?;
    }
    Ok(())
}

/// Rebuild the nested `"trees"` node form from a flat [`NonSymmetricTree`]
/// triple (the inverse of [`flatten_non_symmetric`]), for the json EXPORT path.
/// Walks from node `0`, following each interior node's `(left_diff, right_diff)`
/// to its children; a `(0, 0)` step node is a leaf carrying
/// `leaf_values[node_id_to_leaf_id[id]]`. Depth-bounded by
/// [`MAX_NON_SYMMETRIC_DEPTH`].
///
/// On a malformed in-memory tree (an out-of-range diff / leaf id) the node is
/// emitted as a zero leaf rather than panicking — the EXPORT path never trusts a
/// hostile in-memory model into an OOB index (the loader is the validating side).
fn unflatten_non_symmetric(tree: &NonSymmetricTree) -> NonSymmetricNodeJson {
    fn build(tree: &NonSymmetricTree, id: usize, depth: usize) -> NonSymmetricNodeJson {
        let leaf = |tree: &NonSymmetricTree, id: usize| -> NonSymmetricNodeJson {
            let leaf_id = tree
                .node_id_to_leaf_id
                .get(id)
                .map_or(0usize, |&l| l as usize);
            NonSymmetricNodeJson {
                split: None,
                left: None,
                right: None,
                value: Some(tree.leaf_values.get(leaf_id).copied().unwrap_or(0.0)),
                weight: Some(tree.leaf_weights.get(leaf_id).copied().unwrap_or(0.0)),
            }
        };
        if depth > MAX_NON_SYMMETRIC_DEPTH {
            return leaf(tree, id);
        }
        match tree.step_nodes.get(id) {
            Some(&(0, 0)) | None => leaf(tree, id),
            Some(&(left_diff, right_diff)) => {
                let split = tree.tree_splits.get(id).and_then(ModelSplit::as_float);
                let Some(split) = split else {
                    return leaf(tree, id);
                };
                let left_id = id.saturating_add(left_diff as usize);
                let right_id = id.saturating_add(right_diff as usize);
                NonSymmetricNodeJson {
                    split: Some(SplitJson {
                        border: split.border,
                        float_feature_index: i64::try_from(split.feature).unwrap_or(i64::MAX),
                        split_index: i64::try_from(id).unwrap_or(i64::MAX),
                        split_type: "FloatFeature".to_owned(),
                    }),
                    left: Some(Box::new(build(tree, left_id, depth + 1))),
                    right: Some(Box::new(build(tree, right_id, depth + 1))),
                    value: None,
                    weight: None,
                }
            }
        }
    }
    build(tree, 0, 0)
}

/// Build the serializable document from the canonical model.
fn to_doc(model: &Model) -> ModelJsonDoc {
    // Output dimensions (D-6.2-01 / Plan 06.2-02); `0`/unset means the scalar
    // default `1`. Drives the leaf-major transpose + per-dim bias vector below.
    let dim = model.approx_dimension.max(1);
    let float_features = model
        .float_feature_borders
        .iter()
        .enumerate()
        .map(|(idx, borders)| {
            let fi = i64::try_from(idx).unwrap_or(i64::MAX);
            FloatFeatureJson {
                borders: borders.clone(),
                feature_id: String::new(),
                feature_index: fi,
                flat_feature_index: fi,
                has_nans: false,
                nan_value_treatment: "AsIs".to_owned(),
            }
        })
        .collect();

    let oblivious_trees = model
        .oblivious_trees
        .iter()
        .map(|t| {
            // The numeric-only `model.json` schema emits FLOAT splits only; CTR
            // splits round-trip through the `.cbm` / `ctr_data` path, not this
            // numeric JSON export (a CTR split is skipped here, the json round-trip
            // covers float-only models — the apply path for CTR splits is exercised
            // via the trainer-lifted model + baked ctr_data, not this loader).
            let splits = t
                .splits
                .iter()
                .filter_map(ModelSplit::as_float)
                .enumerate()
                .map(|(si, s)| SplitJson {
                    border: s.border,
                    float_feature_index: i64::try_from(s.feature).unwrap_or(i64::MAX),
                    split_index: i64::try_from(si).unwrap_or(i64::MAX),
                    split_type: "FloatFeature".to_owned(),
                })
                .collect();
            // LEAF-MAJOR transpose (Pitfall 6): the in-memory buffer is
            // DIMENSION-MAJOR (`leaf_values[d * n_leaves + l]`); the upstream
            // `model.json` stores `leaf_values` LEAF-MAJOR (`leaf_values[l * dim
            // + d]`). At `dim == 1` the orders coincide, so the emitted array is
            // byte-identical to the pre-6.2 scalar export. `leaf_weights` stays
            // one-per-leaf.
            let leaf_values = transpose_dim_major_to_leaf_major(&t.leaf_values, dim);
            ObliviousTreeJson {
                leaf_values,
                leaf_weights: t.leaf_weights.clone(),
                splits,
            }
        })
        .collect();

    // Non-symmetric trees → the nested `"trees"` schema (Pitfall 3). EMPTY for a
    // symmetric model, so `skip_serializing_if` omits the key entirely and the
    // oblivious export stays byte-identical.
    let trees = model
        .non_symmetric_trees
        .iter()
        .map(unflatten_non_symmetric)
        .collect();

    ModelJsonDoc {
        features_info: FeaturesInfoJson { float_features },
        oblivious_trees,
        trees,
        // [1, [bias_d0, …]] — scale 1, a per-dimension bias vector (Pitfall 6).
        // At `dim == 1` this is exactly `[1, [bias]]` (byte-identical). The model
        // carries a single scalar bias this wave, so higher dimensions repeat it
        // (per-dim bias plumbing lands with the multi-output losses, Plans
        // 06.2-03..05); for the in-scope dim=1 models this branch is never taken.
        scale_and_bias: serde_json::json!([1, vec![model.bias; dim]]),
        // Emit the multiclass `class_params.class_to_label` only when present; a
        // scalar model omits `model_info` entirely (byte-identical scalar export).
        model_info: model_info_from_class_labels(&model.class_to_label),
    }
}

/// Transpose a DIMENSION-MAJOR leaf buffer (`src[d * n_leaves + l]`) into the
/// LEAF-MAJOR wire order (`dst[l * dim + d]`, Pitfall 6). At `dim == 1` (or `dim
/// == 0`) the input is returned verbatim, so the dim=1 path is byte-identical.
fn transpose_dim_major_to_leaf_major(src: &[f64], dim: usize) -> Vec<f64> {
    if dim <= 1 {
        return src.to_vec();
    }
    let n_leaves = src.len() / dim;
    let mut dst = vec![0.0_f64; src.len()];
    for l in 0..n_leaves {
        for d in 0..dim {
            if let (Some(slot), Some(&v)) = (dst.get_mut(l * dim + d), src.get(d * n_leaves + l)) {
                *slot = v;
            }
        }
    }
    dst
}

/// Transpose a LEAF-MAJOR wire buffer (`src[l * dim + d]`) back into the
/// canonical DIMENSION-MAJOR order (`dst[d * n_leaves + l]`). At `dim == 1` the
/// input is returned verbatim (byte-identical scalar load).
fn transpose_leaf_major_to_dim_major(src: &[f64], dim: usize) -> Vec<f64> {
    if dim <= 1 {
        return src.to_vec();
    }
    let n_leaves = src.len() / dim;
    let mut dst = vec![0.0_f64; src.len()];
    for l in 0..n_leaves {
        for d in 0..dim {
            if let (Some(slot), Some(&v)) = (dst.get_mut(d * n_leaves + l), src.get(l * dim + d)) {
                *slot = v;
            }
        }
    }
    dst
}

/// Serialize `model` to the upstream `model.json` schema at `path` (MODEL-06).
///
/// # Errors
/// [`ModelError::Json`] if serialization fails; [`ModelError::Io`] on a write
/// failure.
pub fn save_json(model: &Model, path: &Path) -> Result<(), ModelError> {
    let doc = to_doc(model);
    let s = serde_json::to_string_pretty(&doc)?;
    std::fs::write(path, s)?;
    Ok(())
}

/// Reconstruct the canonical [`Model`] from a parsed document.
fn from_doc(doc: &ModelJsonDoc) -> Result<Model, ModelError> {
    let float_feature_borders = doc
        .features_info
        .float_features
        .iter()
        .map(|f| f.borders.clone())
        .collect();

    // Output dimensions from the bias-vector length (Pitfall 6); `1` for scalar.
    let dim = read_approx_dimension(&doc.scale_and_bias);

    let oblivious_trees = doc
        .oblivious_trees
        .iter()
        .map(|t| {
            let splits = t
                .splits
                .iter()
                .map(|s| {
                    let feature = usize::try_from(s.float_feature_index).map_err(|_| {
                        ModelError::Deserialize(format!(
                            "negative float_feature_index {}",
                            s.float_feature_index
                        ))
                    })?;
                    Ok(ModelSplit::Float(Split {
                        feature,
                        border: s.border,
                    }))
                })
                .collect::<Result<Vec<_>, ModelError>>()?;
            // Un-transpose the wire LEAF-MAJOR `leaf_values` (`leaf_values[l*dim
            // + d]`) back into the canonical DIMENSION-MAJOR buffer
            // (`leaf_values[d*n_leaves + l]`). At `dim == 1` this is the verbatim
            // array (byte-identical scalar load). `leaf_weights` is one-per-leaf.
            let leaf_values = transpose_leaf_major_to_dim_major(&t.leaf_values, dim);
            // Zero-fill weights if a fixture predates leaf_weights (one weight per
            // leaf == `leaf_values.len() / dim`); keep SHAP/fstr shape-consistent.
            let n_leaves = if dim == 0 { t.leaf_values.len() } else { t.leaf_values.len() / dim };
            let leaf_weights = if t.leaf_weights.len() == n_leaves {
                t.leaf_weights.clone()
            } else {
                vec![0.0; n_leaves]
            };
            Ok(ObliviousTree {
                splits,
                leaf_values,
                leaf_weights,
            })
        })
        .collect::<Result<Vec<_>, ModelError>>()?;

    // Non-symmetric `"trees"` nested-node trees (FEAT-06, Pitfall 3) → flat
    // triple. EMPTY for an oblivious model (the `trees` key is absent → the
    // oblivious path above is byte-identical). A model is EITHER all-oblivious
    // or all-non-symmetric, never both.
    let non_symmetric_trees = doc
        .trees
        .iter()
        .map(flatten_non_symmetric)
        .collect::<Result<Vec<_>, ModelError>>()?;

    // Recover the SORTED distinct class labels from `model_info.class_params`
    // (falling back to `multiclass_params`), CR-01 / LOSS-02. Absent for a scalar
    // model (empty vector — the Class transform then falls back to the raw class
    // index, predict.rs).
    let class_to_label = doc
        .model_info
        .as_ref()
        .map(class_to_label_from_model_info)
        .unwrap_or_default();

    Ok(Model {
        oblivious_trees,
        non_symmetric_trees,
        bias: read_bias(&doc.scale_and_bias)?,
        float_feature_borders,
        ctr_data: None,
        approx_dimension: dim,
        class_to_label,
    })
}

/// Read `scale_and_bias[1][0]` without panicking (Pitfall 6).
/// Derive the model's `approx_dimension` from the `scale_and_bias` bias vector
/// length (`scale_and_bias[1]` has one entry per output dimension, Pitfall 6).
/// Defaults to `1` when the vector is absent / empty, so a scalar `[1, [bias]]`
/// model reads back as `approx_dimension == 1` (byte-identical).
fn read_approx_dimension(scale_and_bias: &serde_json::Value) -> usize {
    scale_and_bias
        .get(1)
        .and_then(serde_json::Value::as_array)
        .map_or(1, Vec::len)
        .max(1)
}

fn read_bias(scale_and_bias: &serde_json::Value) -> Result<f64, ModelError> {
    scale_and_bias
        .get(1)
        .and_then(|bias_vec| bias_vec.get(0))
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            ModelError::Deserialize(
                "scale_and_bias[1][0] (bias) missing or non-numeric".to_owned(),
            )
        })
}

/// Deserialize a `model.json` file at `path` into the canonical [`Model`]
/// (MODEL-06), validating the shape (Security V5).
///
/// # Errors
/// [`ModelError::Json`] on malformed JSON; [`ModelError::Deserialize`] on a
/// malformed `scale_and_bias` or a negative feature index; [`ModelError::Io`] if
/// the file cannot be read.
pub fn load_json(path: &Path) -> Result<Model, ModelError> {
    let contents = std::fs::read_to_string(path)?;
    decode_json(&contents)
}

/// Decode an in-memory `model.json` string (the core of [`load_json`], split out
/// so unit tests can exercise it without a file).
///
/// # Errors
/// As [`load_json`] (minus the I/O arm).
pub fn decode_json(contents: &str) -> Result<Model, ModelError> {
    let doc: ModelJsonDoc = serde_json::from_str(contents)?;
    from_doc(&doc)
}
