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

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::OracleError;

/// One split in an oblivious tree (verified `model.json` schema:
/// `oblivious_trees[i].splits[j]`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SplitJson {
    /// Split border (threshold). For a `FloatFeature` split this is the float
    /// feature threshold; for an `OnlineCtr` split it is the CTR-value border.
    pub border: f64,
    /// Index of the float feature this split tests. Present on `FloatFeature`
    /// splits; ABSENT on `OnlineCtr` (CTR) splits — those reference a CTR table
    /// via `split_index` instead, so the field is `#[serde(default)]` (0) and is
    /// not meaningful for a CTR split. Categorical/tensor-CTR upstream models
    /// (ORD-05) emit `OnlineCtr` splits, so requiring this field would reject a
    /// real `model.json` outright.
    #[serde(default)]
    pub float_feature_index: i64,
    /// Global split index in the model's split pool. For an `OnlineCtr` split this
    /// indexes the model's CTR feature definitions.
    pub split_index: i64,
    /// Split kind: `"FloatFeature"` (numeric) or `"OnlineCtr"` (CTR). Unknown
    /// CTR-only fields (e.g. `ctr_target_border_idx`) are ignored by serde.
    pub split_type: String,
}

/// One oblivious (symmetric) tree: a flat list of `leaf_values` of length
/// `2^depth` and the `depth` ordered `splits` that index objects into them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ObliviousTree {
    /// Leaf values in canonical (binary-index) order; length is `2^depth`.
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights in the SAME per-tree nested
    /// layout as `leaf_values` (RESEARCH Pitfall 2: `model.json` `leaf_weights`
    /// is per-tree nested, NOT the `.cbm` flat array). `#[serde(default)]` keeps
    /// existing Phase-3 fixtures without `leaf_weights` parsing.
    #[serde(default)]
    pub leaf_weights: Vec<f64>,
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

/// One CTR table from the upstream `model.json` `ctr_data` hash-map (the value
/// half of one `"<ctr-base-string>": { … }` entry).
///
/// Upstream emits each table as (see `json_model_helpers.cpp:475-482`):
/// ```json
/// { "hash_map": ["<hash0>", n00, n01, "<hash1>", n10, n11, …],
///   "hash_stride": 3,
///   "counter_denominator": 0 }
/// ```
/// where `hash_map` is a FLAT, heterogeneous array: each stride is one bucket —
/// a hash STRING followed by `hash_stride - 1` integer counts. The interpretation
/// of those counts depends on the CTR type (encoded in the map key, per
/// `static_ctr_provider.cpp:14-126`):
/// - **Borders / Buckets / *TargetMeanValue (classes):** `hash_stride - 1` class
///   counts per bucket; CTR = `ctrHistory[1] / (ctrHistory[0] + ctrHistory[1])`
///   for Borders (and analogous per-class forms).
/// - **Counter / FeatureFreq:** a single `ctrTotal` per bucket; CTR =
///   `ctrTotal[bucket] / counter_denominator`.
/// - **Mean (TCtrMeanHistory):** a `Sum` then a `Count` per bucket (stride 3 incl.
///   hash); CTR = `Sum / Count`.
///
/// Every field is `#[serde(default)]` so Phase-3/4 fixtures WITHOUT a `ctr_data`
/// block keep parsing (RESEARCH A5 — the parser was borders-only before this).
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct CtrTableJson {
    /// Flat, heterogeneous bucket array: `[hash_string, count, count, …]`
    /// repeated every `hash_stride` elements. Stored as raw JSON values because
    /// the array mixes hash strings and integer counts.
    #[serde(default)]
    pub hash_map: Vec<serde_json::Value>,
    /// Number of elements per bucket (1 hash + `hash_stride - 1` counts). Zero
    /// for a degenerate/empty table.
    #[serde(default)]
    pub hash_stride: i64,
    /// `CounterDenominator` for `Counter`/`FeatureFreq` CTRs; `0` otherwise.
    #[serde(default)]
    pub counter_denominator: i64,
}

impl CtrTableJson {
    /// The integer counts of every bucket, in `hash_map` order, with the leading
    /// per-bucket hash string stripped. Each inner `Vec<i64>` has length
    /// `hash_stride - 1`.
    ///
    /// Non-integer entries (a malformed blob) yield [`OracleError::MalformedModel`]
    /// rather than a panic (T-05-01-01). A `hash_stride <= 0` is treated as an
    /// empty table (no buckets).
    ///
    /// # Errors
    /// [`OracleError::MalformedModel`] if a stride boundary is ragged or a count
    /// slot is not an integer.
    pub fn bucket_counts(&self) -> Result<Vec<Vec<i64>>, OracleError> {
        let stride = self.hash_stride;
        if stride <= 0 {
            return Ok(Vec::new());
        }
        let stride = stride as usize;
        if !self.hash_map.len().is_multiple_of(stride) {
            return Err(OracleError::MalformedModel {
                what: format!(
                    "ctr_data hash_map length {} is not a multiple of hash_stride {stride}",
                    self.hash_map.len()
                ),
            });
        }
        let mut out = Vec::with_capacity(self.hash_map.len() / stride);
        for bucket in self.hash_map.chunks_exact(stride) {
            // bucket[0] is the hash string; bucket[1..] are the integer counts.
            let mut counts = Vec::with_capacity(stride - 1);
            for slot in bucket.iter().skip(1) {
                let value = slot.as_i64().ok_or_else(|| OracleError::MalformedModel {
                    what: "ctr_data hash_map count slot is not an integer".to_owned(),
                })?;
                counts.push(value);
            }
            out.push(counts);
        }
        Ok(out)
    }
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
    /// The upstream `ctr_data` hash-map: CTR-base-string key → per-bucket count
    /// table (`json_model_helpers.cpp:524`). Only present when the model has
    /// CTR features; `#[serde(default)]` keeps borders-only Phase-3/4 fixtures
    /// parsing (RESEARCH A5). A `BTreeMap` gives a deterministic key order for
    /// reproducible per-CTR-type comparison.
    #[serde(default)]
    pub ctr_data: BTreeMap<String, CtrTableJson>,
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

    /// Per-tree leaf weights as a nested `Vec<Vec<f64>>` (one inner vector per
    /// tree, RESEARCH Pitfall 2). Trees whose fixture predates the
    /// `leaf_weights` regeneration yield an empty inner vector (the
    /// `#[serde(default)]` field), so callers can detect a missing fixture.
    #[must_use]
    pub fn leaf_weights(&self) -> Vec<Vec<f64>> {
        self.oblivious_trees
            .iter()
            .map(|tree| tree.leaf_weights.clone())
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

    /// The parsed `ctr_data` tables, keyed by upstream CTR-base string. Empty
    /// for a borders-only model (no CTR features). Each table exposes its
    /// per-bucket counts via [`CtrTableJson::bucket_counts`] for per-CTR-type
    /// comparison through `compare_stage`.
    #[must_use]
    pub fn ctr_data(&self) -> &BTreeMap<String, CtrTableJson> {
        &self.ctr_data
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
