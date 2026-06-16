//! The canonical serializable [`Model`] (re-homed from `cb-train`, RESEARCH
//! Primary Recommendation).
//!
//! This is the substrate all of Phase-4's serialize / apply / explain operate
//! on: it carries the boosting-order [`ObliviousTree`]s (each with `leaf_values`
//! AND `leaf_weights` — RESEARCH Pitfall 1), the model `bias`, and the
//! per-float-feature ascending `float_feature_borders` so apply / SHAP /
//! serialize need NO training pool.
//!
//! The float-split type is REUSED (`pub use cb_train::Split`) rather than
//! redefined — the canonical model shares the exact `Split { feature, border }`
//! semantics the trainer produces. A trained [`cb_train::Model`] is lifted into
//! the canonical model via [`Model::from_trained`], carrying the float-feature
//! borders that the trainer scored against.
//!
//! # CTR-split representation (ORD-05 / D-05)
//!
//! A tree's ordered splits are stored as [`ModelSplit`] — EITHER a float
//! threshold ([`ModelSplit::Float`], a [`Split`]) OR a tensor / combination CTR
//! test ([`ModelSplit::Ctr`], a [`CtrSplit`] over a combined categorical
//! projection). This mirrors the trainer-side `cb_train::AnySplit { Float, OneHot }`
//! precedent: a CTR split is a first-class STORED split. The baked
//! [`crate::CtrData`] tables live on the [`Model`] (`ctr_data`); the per-tree
//! split list stores only the `(projection, ctr_type, prior, target_border_idx,
//! border)` test, and the apply path reconstructs the table key from
//! `(projection, ctr_type)` to look up the document's combined-projection CTR
//! value (`crate::ctr_value_for_combined_projection`).

// Reuse the trainer's float-split type verbatim (no redefinition — the canonical
// model shares the exact `Split { feature, border }` semantics).
pub use cb_train::Split;

use crate::ctr_data::{CtrData, ECtrType, Prior};

/// A stored tensor / combination CTR split (ORD-05 / D-05): a `ctr_value > border`
/// test on a materialized CTR feature value computed over the combined
/// categorical [`cb_train::TProjection`]. The baked [`crate::CtrValueTable`] this
/// split tests lives in the model's [`Model::ctr_data`]; `projection` + `ctr_type`
/// reconstruct that table's key at apply time, and the per-document combined hash
/// is folded from the raw cat values via `cb_data::calc_cat_feature_hash` (NEVER
/// the model's stored `ctr_data` hash_map — RESEARCH Anti-Pattern).
#[derive(Debug, Clone, PartialEq)]
pub struct CtrSplit {
    /// The combined categorical projection (sorted cat-feature member set).
    pub projection: cb_train::TProjection,
    /// Which baked [`crate::CtrValueTable`] type to look up.
    pub ctr_type: ECtrType,
    /// The prior used at apply (carried from the model).
    pub prior: Prior,
    /// The Buckets per-class numerator selector (default `0`).
    pub target_border_idx: usize,
    /// The CTR-value threshold; the split passes when `ctr_value > border`.
    pub border: f64,
    /// The inference `Shift` (`calc_normalization(prior_num)` → `shift`), threaded
    /// into the apply `Calc` on BOTH the table-found and not-found branches so the
    /// CTR value reaches the same scaled-border space as the baked borders
    /// (`Borders:Prior=0.5/1` → `0.0`). Defaults to `0.0` for back-compat (Plan 05-14).
    pub shift: f64,
    /// The inference `Scale` (`ctr_border_count / norm`), threaded into the apply
    /// `Calc` on BOTH branches (`Borders:Prior=0.5/1`, `ctr_border_count=15` →
    /// `15.0`). Defaults to `1.0` for back-compat (Plan 05-14).
    pub scale: f64,
}

/// One stored oblivious-tree split: EITHER a float threshold ([`Split`]) OR a
/// tensor / combination CTR test ([`CtrSplit`]). Mirrors the trainer-side
/// `cb_train::AnySplit` precedent (ORD-05 / D-05). Every split consumer
/// (apply / SHAP / fstr / serialize) matches this enum exhaustively so no
/// consumer silently drops a CTR split (T-05-09-03).
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSplit {
    /// A float `value > border` threshold split.
    Float(Split),
    /// A tensor / combination CTR `ctr_value > border` split.
    Ctr(CtrSplit),
}

impl ModelSplit {
    /// The FLOAT feature index this split tests, or `None` for a CTR split. The
    /// numeric-only feature-importance / SHAP consumers project over this (a CTR
    /// split has no single float-feature index).
    #[must_use]
    pub fn float_feature(&self) -> Option<usize> {
        match self {
            Self::Float(s) => Some(s.feature),
            Self::Ctr(_) => None,
        }
    }

    /// The inner [`Split`] for a float split, or `None` for a CTR split.
    #[must_use]
    pub fn as_float(&self) -> Option<&Split> {
        match self {
            Self::Float(s) => Some(s),
            Self::Ctr(_) => None,
        }
    }
}

/// One oblivious (symmetric) tree in the canonical model: the ordered splits, the
/// per-leaf values (already `learning_rate`-scaled, matching upstream
/// `model.json`), and the per-leaf summed training-document weights
/// (`leaf_weights`, RESEARCH Pitfall 1 — required by SHAP /
/// PredictionValuesChange / Interaction).
#[derive(Debug, Clone, PartialEq)]
pub struct ObliviousTree {
    /// The ordered splits (float threshold or CTR test) defining the symmetric
    /// structure.
    pub splits: Vec<ModelSplit>,
    /// Leaf values in canonical forward-bit-order, length `2^depth`.
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights in the same forward-bit-order as
    /// `leaf_values`, length `2^depth`. For unweighted training a leaf weight
    /// equals its document count (RESEARCH A4).
    pub leaf_weights: Vec<f64>,
}

/// The canonical serializable model: boosting-order [`ObliviousTree`]s, the model
/// `bias` (the starting approx), the per-float-feature ascending candidate
/// borders, the baked [`CtrData`] tables CTR splits look up, and (for the
/// categorical apply path) the per-document raw categorical feature columns.
/// Carries everything apply / SHAP / serialize need without a training pool.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// The oblivious trees in boosting (iteration) order.
    pub oblivious_trees: Vec<ObliviousTree>,
    /// The starting approx / model bias.
    pub bias: f64,
    /// Per-float-feature ascending candidate borders (`float_feature_borders[f]`
    /// is feature `f`'s borders). Empty inner vectors are preserved so the index
    /// lines up with the float-feature index.
    pub float_feature_borders: Vec<Vec<f64>>,
    /// The baked `ctr_data` tables CTR splits look up at apply time (ORD-05).
    /// `None` for the numeric-only models (no CTR splits). Keyed by the upstream
    /// CTR-base string; a [`CtrSplit`] reconstructs its key from
    /// `(projection, ctr_type)`.
    pub ctr_data: Option<CtrData>,
    /// The number of output (approx) dimensions (D-6.2-01 / Plan 06.2-02). `1`
    /// for every scalar regression / binary model; `> 1` for multiclass /
    /// multilabel / MultiQuantile. Each tree's `leaf_values` is the
    /// DIMENSION-MAJOR training buffer `leaf_values[d * n_leaves + l]`; the `.cbm`
    /// / json wire format stores it LEAF-MAJOR (`leaf_values[l * dim + d]`,
    /// Pitfall 6). At `1` leaf-major == dim-major, so the serialized bytes are
    /// byte-identical to the pre-6.2 scalar model.
    pub approx_dimension: usize,
    /// The `ClassToLabel` map for a multiclass model (LOSS-02, Pitfall 4): the
    /// SORTED distinct original class labels, so `class_to_label[c]` is the original
    /// label for class index `c`. EMPTY for every scalar regression / binary model.
    /// Predictions recover the original labels via this map (`class_params` /
    /// `multiclass_params` model_info, verified empirically against catboost 1.2.10).
    pub class_to_label: Vec<f64>,
}

impl Model {
    /// Lift a trained [`cb_train::Model`] into the canonical model, carrying the
    /// `float_feature_borders` the trainer scored against (the model's
    /// quantization borders, e.g. from the quantized pool / `model_json`).
    ///
    /// The trained trees' float splits become [`ModelSplit::Float`] and any
    /// trainer-side CTR splits (`cb_train::CtrSplitSpec`) become
    /// [`ModelSplit::Ctr`] — no recomputation. The numeric / one-hot / ordered
    /// paths carry no CTR splits, so `ctr_data` is left `None`; the categorical
    /// train→predict path threads the baked tables in via [`Self::with_ctr_data`].
    #[must_use]
    pub fn from_trained(trained: &cb_train::Model, float_feature_borders: Vec<Vec<f64>>) -> Self {
        let oblivious_trees = trained
            .oblivious_trees
            .iter()
            .map(|t| {
                let mut splits: Vec<ModelSplit> = t
                    .splits
                    .iter()
                    .map(|s| ModelSplit::Float(*s))
                    .collect();
                // Lift any trainer-side tensor-CTR splits into ModelSplit::Ctr
                // (ORD-05 / D-05). The numeric / one-hot / ordered paths carry an
                // empty `ctr_splits`, so this is a no-op there.
                for c in &t.ctr_splits {
                    let ctr_type = ECtrType::from_i8(c.ctr_type).unwrap_or(ECtrType::Borders);
                    splits.push(ModelSplit::Ctr(CtrSplit {
                        projection: c.projection.clone(),
                        ctr_type,
                        prior: Prior {
                            num: c.prior_num,
                            denom: c.prior_denom,
                        },
                        target_border_idx: c.target_border_idx,
                        border: c.border,
                        // Carry the bake-derived (Shift, Scale) so the apply path
                        // scales the CTR value into the baked-border space on BOTH
                        // the found and not-found branches (Plan 05-14).
                        shift: c.shift,
                        scale: c.scale,
                    }));
                }
                ObliviousTree {
                    splits,
                    leaf_values: t.leaf_values.clone(),
                    leaf_weights: t.leaf_weights.clone(),
                }
            })
            .collect();
        Self {
            oblivious_trees,
            bias: trained.bias,
            float_feature_borders,
            ctr_data: None,
            approx_dimension: trained.approx_dimension,
            class_to_label: trained.class_to_label.clone(),
        }
    }

    /// Attach the baked [`CtrData`] tables CTR splits look up at apply time
    /// (the categorical train→predict path threads the model's `ctr_data` in).
    #[must_use]
    pub fn with_ctr_data(mut self, ctr_data: CtrData) -> Self {
        self.ctr_data = Some(ctr_data);
        self
    }

    /// Per-tree leaf values flattened in tree order.
    #[must_use]
    pub fn leaf_values(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|t| t.leaf_values.iter().copied())
            .collect()
    }

    /// Per-tree leaf weights flattened in tree order (RESEARCH Pitfall 1).
    #[must_use]
    pub fn leaf_weights(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|t| t.leaf_weights.iter().copied())
            .collect()
    }
}
