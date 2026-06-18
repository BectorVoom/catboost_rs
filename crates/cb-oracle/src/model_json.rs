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

/// One node of a NON-SYMMETRIC (Lossguide / Depthwise) tree in the upstream
/// `"trees"` recursive nested-node schema (FEAT-06, RESEARCH Pitfall 3). A node
/// is EITHER interior (`split` + `left` + `right`) OR a leaf (`value` +
/// `weight`). This schema is STRUCTURALLY DIFFERENT from `oblivious_trees`
/// (flat arrays) and MUST NOT be routed through [`ObliviousTree`] (Pitfall 3).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NonSymmetricNodeJson {
    /// The interior-node split (absent on a leaf).
    #[serde(default)]
    pub split: Option<SplitJson>,
    /// The left child (absent on a leaf). Boxed — the schema is recursive.
    #[serde(default)]
    pub left: Option<Box<NonSymmetricNodeJson>>,
    /// The right child (absent on a leaf). Boxed — the schema is recursive.
    #[serde(default)]
    pub right: Option<Box<NonSymmetricNodeJson>>,
    /// The leaf value (absent on an interior node).
    #[serde(default)]
    pub value: Option<f64>,
    /// The leaf's summed training-document weight (absent on an interior node).
    #[serde(default)]
    pub weight: Option<f64>,
}

/// The flat triple a non-symmetric `"trees"` tree converts to (the per-stage
/// comparator consumes these). Mirrors the canonical `cb_model::NonSymmetricTree`
/// layout: per-node splits / step-node diffs / leaf ids + the distinct leaves.
#[derive(Debug, Clone, PartialEq)]
pub struct NonSymmetricFlatTree {
    /// Per-node split border (interior nodes) in flat pre-order. Leaf nodes carry
    /// a placeholder (`f64::NEG_INFINITY`) the apply walk never reads.
    pub split_borders: Vec<f64>,
    /// Per-node `(left_subtree_diff, right_subtree_diff)`; `(0, 0)` marks a leaf.
    pub step_nodes: Vec<(u16, u16)>,
    /// Per-node index into `leaf_values` (only meaningful for leaf nodes).
    pub node_id_to_leaf_id: Vec<u32>,
    /// The distinct leaf values in leaf-visitation (pre-order) order.
    pub leaf_values: Vec<f64>,
    /// The per-leaf summed weights, same order as `leaf_values`.
    pub leaf_weights: Vec<f64>,
}

impl NonSymmetricNodeJson {
    /// Flatten this nested-node tree into the [`NonSymmetricFlatTree`] triple
    /// (RESEARCH Pitfall 3 — a DISTINCT parser path, NOT the oblivious flat
    /// arrays). Node ids are assigned in a deterministic PRE-ORDER walk; an
    /// interior node's `(left_diff, right_diff)` are the offsets from its id to
    /// its children. Depth-bounded by [`MAX_NON_SYMMETRIC_DEPTH`] (no unbounded
    /// recursion on a crafted file).
    ///
    /// # Errors
    /// [`OracleError::MalformedModel`] on a malformed node (interior missing a
    /// child/split, leaf missing its value) or a tree deeper than the bound.
    pub fn flatten(&self) -> Result<NonSymmetricFlatTree, OracleError> {
        let mut order: Vec<&NonSymmetricNodeJson> = Vec::new();
        collect_preorder(self, 0, &mut order)?;
        let node_count = order.len();

        let id_of = |target: &NonSymmetricNodeJson| -> Option<usize> {
            order.iter().position(|n| std::ptr::eq(*n, target))
        };

        let mut split_borders: Vec<f64> = Vec::with_capacity(node_count);
        let mut step_nodes: Vec<(u16, u16)> = Vec::with_capacity(node_count);
        let mut node_id_to_leaf_id: Vec<u32> = vec![0; node_count];
        let mut leaf_values: Vec<f64> = Vec::new();
        let mut leaf_weights: Vec<f64> = Vec::new();

        for (id, node) in order.iter().enumerate() {
            if let Some(split) = node.split.as_ref() {
                split_borders.push(split.border);
                let left = node.left.as_deref().ok_or_else(|| OracleError::MalformedModel {
                    what: "non-symmetric interior node missing left".to_owned(),
                })?;
                let right = node.right.as_deref().ok_or_else(|| OracleError::MalformedModel {
                    what: "non-symmetric interior node missing right".to_owned(),
                })?;
                let left_id = id_of(left).ok_or_else(|| OracleError::MalformedModel {
                    what: "non-symmetric left child id not found".to_owned(),
                })?;
                let right_id = id_of(right).ok_or_else(|| OracleError::MalformedModel {
                    what: "non-symmetric right child id not found".to_owned(),
                })?;
                let left_diff = u16::try_from(left_id.saturating_sub(id)).map_err(|_| {
                    OracleError::MalformedModel {
                        what: "non-symmetric left subtree diff exceeds u16".to_owned(),
                    }
                })?;
                let right_diff = u16::try_from(right_id.saturating_sub(id)).map_err(|_| {
                    OracleError::MalformedModel {
                        what: "non-symmetric right subtree diff exceeds u16".to_owned(),
                    }
                })?;
                step_nodes.push((left_diff, right_diff));
            } else {
                let value = node.value.ok_or_else(|| OracleError::MalformedModel {
                    what: "non-symmetric leaf missing value".to_owned(),
                })?;
                split_borders.push(f64::NEG_INFINITY);
                step_nodes.push((0, 0));
                let leaf_id = leaf_values.len();
                node_id_to_leaf_id[id] =
                    u32::try_from(leaf_id).map_err(|_| OracleError::MalformedModel {
                        what: "non-symmetric leaf id exceeds u32".to_owned(),
                    })?;
                leaf_values.push(value);
                leaf_weights.push(node.weight.unwrap_or(0.0));
            }
        }

        Ok(NonSymmetricFlatTree {
            split_borders,
            step_nodes,
            node_id_to_leaf_id,
            leaf_values,
            leaf_weights,
        })
    }
}

/// The maximum non-symmetric `"trees"` nested-node recursion depth the oracle
/// converter accepts (T-06.6-07 — a crafted file cannot drive unbounded stack
/// recursion). Well past any real upstream tree depth.
const MAX_NON_SYMMETRIC_DEPTH: usize = 64;

/// Pre-order traversal collecting node references in id order (root, whole left
/// subtree, whole right subtree). Depth-bounded by [`MAX_NON_SYMMETRIC_DEPTH`].
///
/// # Errors
/// [`OracleError::MalformedModel`] if `depth` exceeds the bound or an interior
/// node is missing a child.
fn collect_preorder<'a>(
    node: &'a NonSymmetricNodeJson,
    depth: usize,
    out: &mut Vec<&'a NonSymmetricNodeJson>,
) -> Result<(), OracleError> {
    if depth > MAX_NON_SYMMETRIC_DEPTH {
        return Err(OracleError::MalformedModel {
            what: format!("non-symmetric tree exceeds max depth {MAX_NON_SYMMETRIC_DEPTH}"),
        });
    }
    out.push(node);
    if node.split.is_some() {
        let left = node.left.as_deref().ok_or_else(|| OracleError::MalformedModel {
            what: "non-symmetric interior node missing left".to_owned(),
        })?;
        let right = node.right.as_deref().ok_or_else(|| OracleError::MalformedModel {
            what: "non-symmetric interior node missing right".to_owned(),
        })?;
        collect_preorder(left, depth + 1, out)?;
        collect_preorder(right, depth + 1, out)?;
    }
    Ok(())
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
    /// All oblivious (symmetric) trees in boosting order. EMPTY for a
    /// non-symmetric model (which populates `trees` instead). `#[serde(default)]`
    /// so a non-symmetric `model.json` (no `oblivious_trees` key) still parses.
    #[serde(default)]
    pub oblivious_trees: Vec<ObliviousTree>,
    /// All NON-SYMMETRIC (Lossguide / Depthwise) trees in boosting order, in the
    /// recursive nested-node schema (FEAT-06, RESEARCH Pitfall 3 — a DISTINCT
    /// top-level key from `oblivious_trees`). EMPTY for a symmetric model;
    /// `#[serde(default)]` keeps oblivious fixtures parsing.
    #[serde(default)]
    pub trees: Vec<NonSymmetricNodeJson>,
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

    /// Whether this is a NON-SYMMETRIC (Lossguide / Depthwise) model — it
    /// populates `trees` (nested-node) rather than `oblivious_trees` (Pitfall 3).
    #[must_use]
    pub fn is_non_symmetric(&self) -> bool {
        !self.trees.is_empty()
    }

    /// Flatten every non-symmetric `"trees"` nested-node tree into the flat
    /// triple the per-stage comparator consumes (FEAT-06, RESEARCH Pitfall 3).
    /// Returns an EMPTY vector for an oblivious model. The result MUST be
    /// non-empty (and per-tree non-zero-length) for a non-symmetric fixture —
    /// a zero-length triple is the Pitfall-3 "parsed as oblivious" warning sign.
    ///
    /// # Errors
    /// [`OracleError::MalformedModel`] from [`NonSymmetricNodeJson::flatten`].
    pub fn non_symmetric_flat_trees(&self) -> Result<Vec<NonSymmetricFlatTree>, OracleError> {
        self.trees.iter().map(NonSymmetricNodeJson::flatten).collect()
    }

    /// Per-tree non-symmetric split borders flattened in tree order (the INTERIOR
    /// nodes' borders, leaf placeholders excluded), ready for
    /// `compare_stage(Stage::Splits, …)` — the SPLITS-FIRST lock (Open Question 1).
    ///
    /// # Errors
    /// [`OracleError::MalformedModel`] from flattening.
    pub fn non_symmetric_split_borders(&self) -> Result<Vec<f64>, OracleError> {
        let mut out = Vec::new();
        for tree in self.non_symmetric_flat_trees()? {
            for (border, &(l, r)) in tree.split_borders.iter().zip(tree.step_nodes.iter()) {
                if !(l == 0 && r == 0) {
                    out.push(*border);
                }
            }
        }
        Ok(out)
    }

    /// Per-tree non-symmetric leaf values flattened in tree order (distinct
    /// leaves, leaf-visitation order), ready for `compare_stage(Stage::LeafValues, …)`.
    ///
    /// # Errors
    /// [`OracleError::MalformedModel`] from flattening.
    pub fn non_symmetric_leaf_values(&self) -> Result<Vec<f64>, OracleError> {
        let mut out = Vec::new();
        for tree in self.non_symmetric_flat_trees()? {
            out.extend_from_slice(&tree.leaf_values);
        }
        Ok(out)
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
