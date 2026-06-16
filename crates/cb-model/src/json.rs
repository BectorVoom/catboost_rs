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
use crate::{Model, ModelSplit, ObliviousTree, Split};

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
    /// All oblivious trees in boosting (iteration) order.
    oblivious_trees: Vec<ObliviousTreeJson>,
    /// Upstream `[scale, [bias, …]]`; emitted as `[1, [bias]]` (Pitfall 6).
    scale_and_bias: serde_json::Value,
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

    ModelJsonDoc {
        features_info: FeaturesInfoJson { float_features },
        oblivious_trees,
        // [1, [bias_d0, …]] — scale 1, a per-dimension bias vector (Pitfall 6).
        // At `dim == 1` this is exactly `[1, [bias]]` (byte-identical). The model
        // carries a single scalar bias this wave, so higher dimensions repeat it
        // (per-dim bias plumbing lands with the multi-output losses, Plans
        // 06.2-03..05); for the in-scope dim=1 models this branch is never taken.
        scale_and_bias: serde_json::json!([1, vec![model.bias; dim]]),
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

    Ok(Model {
        oblivious_trees,
        bias: read_bias(&doc.scale_and_bias)?,
        float_feature_borders,
        ctr_data: None,
        approx_dimension: dim,
        // The json model carries class labels in `model_info.class_params`; the
        // per-stage train oracle constructs the model via `from_trained`, so the
        // json deserialize path leaves it empty until a later plan wires the
        // class_params round-trip.
        class_to_label: Vec::new(),
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
