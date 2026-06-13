//! The canonical serializable [`Model`] (re-homed from `cb-train`, RESEARCH
//! Primary Recommendation).
//!
//! This is the substrate all of Phase-4's serialize / apply / explain operate
//! on: it carries the boosting-order [`ObliviousTree`]s (each with `leaf_values`
//! AND `leaf_weights` — RESEARCH Pitfall 1), the model `bias`, and the
//! per-float-feature ascending `float_feature_borders` so apply / SHAP /
//! serialize need NO training pool.
//!
//! The split type is REUSED (`pub use cb_train::Split`) rather than redefined —
//! the canonical model shares the exact `Split { feature, border }` semantics the
//! trainer produces. A trained [`cb_train::Model`] is lifted into the canonical
//! model via [`Model::from_trained`], carrying the float-feature borders that the
//! trainer scored against.

// Reuse the trainer's split type verbatim (no redefinition — the canonical model
// shares the exact `Split { feature, border }` semantics).
pub use cb_train::Split;

/// One oblivious (symmetric) tree in the canonical model: the ordered splits, the
/// per-leaf values (already `learning_rate`-scaled, matching upstream
/// `model.json`), and the per-leaf summed training-document weights
/// (`leaf_weights`, RESEARCH Pitfall 1 — required by SHAP /
/// PredictionValuesChange / Interaction).
#[derive(Debug, Clone, PartialEq)]
pub struct ObliviousTree {
    /// The ordered splits (feature + border) defining the symmetric structure.
    pub splits: Vec<Split>,
    /// Leaf values in canonical forward-bit-order, length `2^depth`.
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights in the same forward-bit-order as
    /// `leaf_values`, length `2^depth`. For unweighted training a leaf weight
    /// equals its document count (RESEARCH A4).
    pub leaf_weights: Vec<f64>,
}

/// The canonical serializable model: boosting-order [`ObliviousTree`]s, the model
/// `bias` (the starting approx), and the per-float-feature ascending candidate
/// borders. Carries everything apply / SHAP / serialize need without a training
/// pool.
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
}

impl Model {
    /// Lift a trained [`cb_train::Model`] into the canonical model, carrying the
    /// `float_feature_borders` the trainer scored against (the model's
    /// quantization borders, e.g. from the quantized pool / `model_json`).
    ///
    /// The trained trees' splits, leaf values, and captured leaf weights are
    /// copied verbatim; no recomputation occurs.
    #[must_use]
    pub fn from_trained(trained: &cb_train::Model, float_feature_borders: Vec<Vec<f64>>) -> Self {
        let oblivious_trees = trained
            .oblivious_trees
            .iter()
            .map(|t| ObliviousTree {
                splits: t.splits.clone(),
                leaf_values: t.leaf_values.clone(),
                leaf_weights: t.leaf_weights.clone(),
            })
            .collect();
        Self {
            oblivious_trees,
            bias: trained.bias,
            float_feature_borders,
        }
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
