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

/// One non-symmetric (Lossguide / Depthwise) tree in the canonical model
/// (FEAT-06 / D-6.6-05). Mirrors the upstream flat-node serialization triple
/// (`TreeSplits` per-node + `NonSymmetricStepNodes {LeftSubtreeDiff,
/// RightSubtreeDiff}` + `NonSymmetricNodeIdToLeafId`, `model.cpp:111-165`),
/// reusing the existing `TNonSymmetricTreeStepNode` FlatBuffers bindings.
///
/// The apply path (06.6-05) is a pointer-walk: start at node `0`, at each node
/// take `diff = (binarized_value >= split_idx) ? right_subtree_diff :
/// left_subtree_diff`, advance `index += diff`, stop when `diff == 0` (a leaf),
/// then `leaf = node_id_to_leaf_id[index]` (RESEARCH Pattern 1). The oblivious
/// arm is left BYTE-IDENTICAL — this is a SEPARATE variant, NOT a refactor of
/// the symmetric traversal (D-6.6-05).
#[derive(Debug, Clone, PartialEq)]
pub struct NonSymmetricTree {
    /// One split per node, in flat-node order. Each node's split is EITHER a
    /// float threshold ([`ModelSplit::Float`]) OR a CTR test ([`ModelSplit::Ctr`]),
    /// mirroring [`ObliviousTree::splits`]. A pure leaf node carries no split,
    /// so `tree_splits` is indexed by NODE id and only INTERIOR nodes hold a
    /// meaningful split (a leaf's `step_nodes` entry is `(0, 0)`).
    pub tree_splits: Vec<ModelSplit>,
    /// Per-node `(left_subtree_diff, right_subtree_diff)` offsets matching
    /// `TNonSymmetricTreeStepNode`. A `(0, 0)` entry marks a terminal (leaf)
    /// node — the walk stops there. For an interior node the chosen diff is
    /// ADDED to the current node index to reach the next node.
    pub step_nodes: Vec<(u16, u16)>,
    /// Per-node index into the flat `leaf_values` (`NonSymmetricNodeIdToLeafId`).
    /// Only meaningful for terminal nodes (a `(0, 0)` `step_nodes` entry); the
    /// apply walk reads `node_id_to_leaf_id[index]` once it halts.
    pub node_id_to_leaf_id: Vec<u32>,
    /// Leaf values in flat node-graph leaf order (DIMENSION-MAJOR for the
    /// multi-output case, identical layout discipline to [`ObliviousTree`]).
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights, same length / order as the
    /// distinct leaf values (RESEARCH Pitfall 1).
    pub leaf_weights: Vec<f64>,
}

/// One level of a [`RegionTree`] path (GPUT-18 / D-03a): a split test plus the
/// CONTINUE direction the walk follows while the test matches. Mirrors upstream's
/// `TRegionDirection { ui16 Bin; ui16 Value; }` + the `takeEqualAndSplitDirection`
/// bit-packing (`add_model_value.cu:229` — one-hot in bit 0, expected direction in
/// bit 1).
#[derive(Debug, Clone, PartialEq)]
pub struct RegionLevel {
    /// The split test at this level (a `value > border` float test, or a CTR test).
    /// The CPU grower emits float splits only; the CTR / device paths may add more.
    pub split: ModelSplit,
    /// The CONTINUE direction: the walk descends to the next level while
    /// `passes(split) == expected_direction`, otherwise it diverges into this
    /// level's terminal leaf.
    pub expected_direction: bool,
    /// One-hot flag (`feature.OneHotFeature`): when `true` the split is an equality
    /// test rather than `>`. Always `false` for the CPU float grower; carried for
    /// round-trip fidelity + device parity (Plan 04).
    pub one_hot: bool,
}

/// One REGION tree in the canonical model (GPUT-18 / D-03a): upstream's
/// `TRegionModel` — an oblivious-like PATH walked while each level's computed split
/// matches the stored direction, diverging into a terminal leaf otherwise. A
/// depth-`d` Region has `d` [`RegionLevel`]s and exactly `d + 1` leaves (leaf `k`
/// for `0 <= k < depth` holds objects that diverged at level `k`; leaf `depth`
/// holds objects that matched every direction).
///
/// This is a PATH model — the apply walk ([`crate::apply`] `region_leaf`) is
/// `bin = 0; for level { if passes == expected_direction { bin += 1 } else { break }
/// } leaf = bin`. It is DISTINCT from [`NonSymmetricTree`]'s binary node graph and
/// MUST NOT be modeled with `step_nodes` (leaf count `2^depth` is the failure
/// signal, Pitfall 2).
#[derive(Debug, Clone, PartialEq)]
pub struct RegionTree {
    /// The per-level path splits + directions, in walk order, length `depth`.
    pub levels: Vec<RegionLevel>,
    /// Leaf values in bin order (dimension-major for multi-output,
    /// `leaf_values[d * n_leaves + l]`), length `(depth + 1) * dim`. Indexed
    /// DIRECTLY by the walk's `bin` (0..=depth).
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights, same bin order as `leaf_values`,
    /// length `depth + 1`.
    pub leaf_weights: Vec<f64>,
}

/// A tree in the canonical model: an oblivious (symmetric) tree, a non-symmetric
/// (Lossguide / Depthwise) tree (FEAT-06 / D-6.6-05), OR a region path (GPUT-18 /
/// D-03a). Every tree consumer (apply / SHAP / fstr / serialize) matches this enum
/// exhaustively so no arm can be silently dropped, and the oblivious arm stays
/// byte-identical to the pre-6.6 path.
#[derive(Debug, Clone, PartialEq)]
pub enum TreeVariant {
    /// A symmetric oblivious tree (the pre-6.6 path, BYTE-IDENTICAL).
    Oblivious(ObliviousTree),
    /// A non-symmetric (Lossguide / Depthwise) tree.
    NonSymmetric(NonSymmetricTree),
    /// A region path tree (GPUT-18).
    Region(RegionTree),
}

impl TreeVariant {
    /// The oblivious tree if this is an [`TreeVariant::Oblivious`], else `None`.
    #[must_use]
    pub fn as_oblivious(&self) -> Option<&ObliviousTree> {
        match self {
            Self::Oblivious(t) => Some(t),
            Self::NonSymmetric(_) | Self::Region(_) => None,
        }
    }

    /// The non-symmetric tree if this is an [`TreeVariant::NonSymmetric`], else
    /// `None`.
    #[must_use]
    pub fn as_non_symmetric(&self) -> Option<&NonSymmetricTree> {
        match self {
            Self::NonSymmetric(t) => Some(t),
            Self::Oblivious(_) | Self::Region(_) => None,
        }
    }

    /// The region tree if this is an [`TreeVariant::Region`], else `None`.
    #[must_use]
    pub fn as_region(&self) -> Option<&RegionTree> {
        match self {
            Self::Region(t) => Some(t),
            Self::Oblivious(_) | Self::NonSymmetric(_) => None,
        }
    }

    /// This tree's leaf values (forward-bit order for oblivious, flat-node leaf
    /// order for non-symmetric, bin order for region).
    #[must_use]
    pub fn leaf_values(&self) -> &[f64] {
        match self {
            Self::Oblivious(t) => &t.leaf_values,
            Self::NonSymmetric(t) => &t.leaf_values,
            Self::Region(t) => &t.leaf_values,
        }
    }

    /// This tree's per-leaf weights (RESEARCH Pitfall 1).
    #[must_use]
    pub fn leaf_weights(&self) -> &[f64] {
        match self {
            Self::Oblivious(t) => &t.leaf_weights,
            Self::NonSymmetric(t) => &t.leaf_weights,
            Self::Region(t) => &t.leaf_weights,
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
    /// The non-symmetric (Lossguide / Depthwise) trees in boosting order
    /// (FEAT-06 / D-6.6-05). EMPTY for every pre-6.6 symmetric model, so the
    /// oblivious `.cbm` / json / apply paths stay byte-identical (a model is
    /// EITHER all-oblivious or all-non-symmetric — upstream never mixes grow
    /// policies within one model). The grower (06.6-04) populates this; the
    /// pointer-walk apply (06.6-05) consumes it.
    pub non_symmetric_trees: Vec<NonSymmetricTree>,
    /// The region path trees in boosting order (GPUT-18 / D-03a). EMPTY for every
    /// oblivious / non-symmetric model (a model is EITHER all-oblivious OR
    /// all-non-symmetric OR all-region — never mixed), so those `.cbm` / json /
    /// apply paths stay byte-identical. The Region grower (Plan 12-02) populates
    /// this; the walk-until-diverge apply (`predict_raw_one`) consumes it.
    pub region_trees: Vec<RegionTree>,
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
        // FEAT-06 / D-6.6-04: lift the non-symmetric grower output into
        // `TreeVariant::NonSymmetric`-shaped `NonSymmetricTree`s, mapping the
        // per-node float splits to `ModelSplit::Float` exactly as the oblivious lift
        // does and carrying the node graph (`step_nodes` / `node_id_to_leaf_id`)
        // verbatim. A model is EITHER all-oblivious or all-non-symmetric, so for an
        // oblivious model `trained.non_symmetric_trees` is empty and this is a no-op
        // (the oblivious lift path above stays byte-identical, D-6.6-05). Leaf VALUES
        // + the apply pointer-walk round-trip land in 06.6-05; this plan lifts the
        // STRUCTURE (per-node splits + node graph).
        let non_symmetric_trees = trained
            .non_symmetric_trees
            .iter()
            .map(|t| {
                let tree_splits: Vec<ModelSplit> =
                    t.splits.iter().map(|s| ModelSplit::Float(*s)).collect();
                NonSymmetricTree {
                    tree_splits,
                    step_nodes: t.step_nodes.clone(),
                    node_id_to_leaf_id: t.node_id_to_leaf_id.clone(),
                    leaf_values: t.leaf_values.clone(),
                    leaf_weights: t.leaf_weights.clone(),
                }
            })
            .collect();
        // GPUT-18 / D-03a: lift the Region grower output into `TreeVariant::Region`
        // shaped `RegionTree`s. Each per-level float split becomes a
        // `ModelSplit::Float` (exactly as the oblivious / non-sym lifts do), zipped
        // with its continue direction + one-hot flag. A model is EITHER all-oblivious
        // OR all-non-sym OR all-region, so for a non-region model
        // `trained.region_trees` is empty and this is a no-op (the other lift paths
        // stay byte-identical, D-04).
        let region_trees = trained
            .region_trees
            .iter()
            .map(|t| {
                let levels: Vec<RegionLevel> = t
                    .splits
                    .iter()
                    .enumerate()
                    .map(|(level, s)| RegionLevel {
                        split: ModelSplit::Float(*s),
                        expected_direction: t.directions.get(level).copied().unwrap_or(false),
                        one_hot: t.one_hot.get(level).copied().unwrap_or(false),
                    })
                    .collect();
                RegionTree {
                    levels,
                    leaf_values: t.leaf_values.clone(),
                    leaf_weights: t.leaf_weights.clone(),
                }
            })
            .collect();
        Self {
            oblivious_trees,
            non_symmetric_trees,
            region_trees,
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
