//! `model.json` parser for the training oracle (INFRA-04, D-11).
//!
//! Upstream `catboost==1.2.10` `save_model(format='json')` emits an oblivious
//! (symmetric) tree model whose per-tree `splits` (`float_feature_index`,
//! `border`) and `leaf_values` are the per-stage parity targets the Wave-1
//! first-slice oracle locks against via `compare_stage(Stage::Splits, …)` and
//! `compare_stage(Stage::LeafValues, …)`.
//!
//! Mirrors `fixture.rs`' serde + fallible-loader shape: `Deserialize` structs +
//! a `load_model_json` that returns [`OracleError`] (never panics) on a missing
//! file or malformed JSON (T-03-00-01 mitigation). No `unwrap`/`expect` in the
//! production path — the deny-lints stay satisfied.

use std::path::Path;

use serde::Deserialize;

use crate::error::OracleError;

/// One split in an oblivious tree (verified `model.json` schema:
/// `oblivious_trees[i].splits[j]`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SplitJson {
    /// Split border (threshold) on the referenced float feature.
    pub border: f64,
    /// Index of the float feature this split tests.
    pub float_feature_index: i64,
    /// Global split index in the model's split pool.
    pub split_index: i64,
    /// Split kind (e.g. `"FloatFeature"`); plain numeric models only emit
    /// `FloatFeature` splits this phase.
    pub split_type: String,
}

/// One oblivious (symmetric) tree: a flat list of `leaf_values` of length
/// `2^depth` and the `depth` ordered `splits` that index objects into them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ObliviousTree {
    /// Leaf values in canonical (binary-index) order; length is `2^depth`.
    pub leaf_values: Vec<f64>,
    /// The ordered splits defining this tree's symmetric structure.
    pub splits: Vec<SplitJson>,
}

/// One float feature's metadata (verified `model.json` schema:
/// `features_info.float_features[i]`). Only the `borders` are consumed by the
/// training oracle (they are the candidate split borders the trainer scores).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FloatFeatureJson {
    /// The feature's index among float features.
    pub feature_index: i64,
    /// The ascending quantization borders for this feature (the candidate split
    /// thresholds). May be empty when the model never split on the feature.
    #[serde(default)]
    pub borders: Vec<f64>,
}

/// The `features_info` block (the subset the oracle consumes).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FeaturesInfoJson {
    /// Per-float-feature metadata in feature order.
    #[serde(default)]
    pub float_features: Vec<FloatFeatureJson>,
}

/// Top-level upstream `model.json` (the subset the oracle consumes).
///
/// `scale_and_bias` is upstream's `[scale, [bias, …]]` pair; for the
/// single-target regression/binclf skeletons it is `[1, [bias]]`, so
/// [`ModelJson::bias`] reads `scale_and_bias[1][0]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelJson {
    /// Per-feature metadata, including each float feature's candidate borders.
    pub features_info: FeaturesInfoJson,
    /// All oblivious trees in boosting (iteration) order.
    pub oblivious_trees: Vec<ObliviousTree>,
    /// Upstream `[scale, [bias, …]]`. Untyped (`serde_json::Value`) because the
    /// outer array mixes a scalar scale and a nested bias vector; the typed
    /// accessor [`ModelJson::bias`] extracts the bias without `unwrap`.
    pub scale_and_bias: serde_json::Value,
}

impl ModelJson {
    /// Per-tree borders flattened in tree order, ready for
    /// `compare_stage(Stage::Splits, …)`.
    #[must_use]
    pub fn split_borders(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|tree| tree.splits.iter().map(|split| split.border))
            .collect()
    }

    /// Per-tree leaf values flattened in tree order, ready for
    /// `compare_stage(Stage::LeafValues, …)`.
    #[must_use]
    pub fn leaf_values(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|tree| tree.leaf_values.iter().copied())
            .collect()
    }

    /// Per-float-feature candidate borders, in feature order (each feature's
    /// ascending borders). Empty inner vectors are preserved so the index lines
    /// up with the float-feature index. Ready to feed the trainer as the
    /// candidate split borders.
    #[must_use]
    pub fn float_feature_borders(&self) -> Vec<Vec<f64>> {
        self.features_info
            .float_features
            .iter()
            .map(|f| f.borders.clone())
            .collect()
    }

    /// The model bias `scale_and_bias[1][0]`.
    ///
    /// # Errors
    /// [`OracleError::MalformedModel`] if `scale_and_bias` is not the expected
    /// `[scale, [bias, …]]` shape (e.g. an empty bias vector or a non-numeric
    /// entry) — surfaced as an error rather than a panic (T-03-00-01).
    pub fn bias(&self) -> Result<f64, OracleError> {
        self.scale_and_bias
            .get(1)
            .and_then(|bias_vec| bias_vec.get(0))
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| OracleError::MalformedModel {
                what: "scale_and_bias[1][0] (bias) missing or non-numeric".to_owned(),
            })
    }
}

/// Parses an upstream `model.json` into a [`ModelJson`].
///
/// # Errors
/// [`OracleError::Io`] if the file cannot be read; [`OracleError::Json`] if it
/// cannot be parsed as the expected schema.
pub fn load_model_json(path: &Path) -> Result<ModelJson, OracleError> {
    let contents = std::fs::read_to_string(path)?;
    let model = serde_json::from_str(&contents)?;
    Ok(model)
}
