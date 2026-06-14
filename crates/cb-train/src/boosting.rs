//! Plain gradient-boosting loop (TRAIN-01) — drives [`crate::tree`] over the
//! generic `cb-compute` [`Runtime`] boundary to grow symmetric oblivious trees
//! with Gradient leaf estimation, oracle-locked to upstream catboost 1.2.10.
//!
//! # Source of truth
//!
//! `catboost/libs/train_lib/train_model.cpp` (boosting driver) +
//! `online_predictor.h` (leaf math):
//! - Starting approx (`CalcOptimumConstApprox`, Pitfall 2): for RMSE
//!   `boost_from_average=true` the starting approx is the target MEAN, stored as
//!   the model bias; for Logloss `boost_from_average=false` the starting approx
//!   is `0` (bias `0`).
//! - Per iteration: `compute_gradients(approx, target)` → grow one oblivious tree
//!   → Gradient leaf delta `CalcAverage(sumDer, sumWeight, scaledL2)` over each
//!   leaf's members (ordered `sum_f64`, D-05) → store `learning_rate * delta` as
//!   the leaf value → `approx[i] += leaf_value[leaf(i)]`.
//! - `leaf_estimation_iterations = 1` for this slice (auto-forced; Pitfall 5).
//!
//! # Parity discipline
//!
//! Every leaf SUM routes through `cb_core::sum_f64` (via
//! `cb_compute::reduce_leaf_stats`). The leaf values STORED already include the
//! `learning_rate` factor, matching the upstream `model.json` `leaf_values` the
//! oracle compares against. Degenerate inputs surface as [`CbError`]; no
//! `unwrap`/`expect`/raw float fold in production (deny-lints + D-08).

use cb_compute::{
    collect_leaf_residuals, exact_leaf_delta, gradient_leaf_delta, newton_leaf_delta,
    reduce_leaf_der2, reduce_leaf_stats, scale_l2_reg, score_st_dev, simple_leaf_delta,
    LeafMethod, Loss, Runtime, QUANTILE_ALPHA, QUANTILE_DELTA,
};
use cb_core::{sum_f64, CbError, CbResult, TFastRng64};

use crate::autolr::{self, TargetType};
use crate::bootstrap::{bootstrap, last_iter_mean_leaf_value, EBootstrapType};
use crate::ctr::{CounterCalcMethod, ECtrType};
use crate::fold::Fold;
use crate::metrics::{EvalMetric, EvalMetricHistory};
use crate::overfit::{BestModelTracker, EOverfittingDetectorType, OverfittingDetector};
use crate::candidates::tensor_ctr_candidates;
use crate::tree::{
    check_depth, greedy_tensor_search_oblivious_ordered, greedy_tensor_search_oblivious_perturbed,
    leaf_index, CtrSplitSpec, FeatureMatrix, GrownTree, Perturbation, Split,
};

/// Per-iteration PRE-bootstrap draws on the persistent RNG (train.cpp:208,211):
/// the fold pick (`Rand.GenRand() % foldCount`) and the derivative-seed draw
/// (`Rand.GenRand()` feeding `GenRandUI64Vector`). Consumed only when sampling
/// is active so the bootstrap draws land on the correct RNG phase every tree.
const PRE_TREE_DRAWS: usize = 2;

/// Per-iteration POST-bootstrap draws BEYOND the `depth` per-level
/// `CalcScores` random-strength seed draws (greedy_tensor_search.cpp:884): the
/// depth loop evaluates `depth + 1` candidate levels (the final level finds no
/// improving split and breaks), so `CalcScores` draws one extra `GenRand()`.
/// Verified end-to-end against the Bernoulli oracle (post = depth + 1).
const POST_TREE_EXTRA_DRAWS: usize = 1;

/// The boosting type (`EBoostingType`, `boosting_options.cpp:16`). The CPU
/// default is [`EBoostingType::Plain`]; [`EBoostingType::Ordered`] drives the
/// anti-leakage body/tail ordered approximant (ORD-02). Pinned EXPLICITLY on
/// [`BoostParams::boosting_type`] (never auto-selected — Ordered auto-select is
/// GPU-only, RESEARCH Pitfall 6 / Anti-Pattern).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EBoostingType {
    /// Plain boosting: a single body/tail spanning the whole fold; every
    /// document's approximant is estimated on the whole set (the 05-02..05-04
    /// path). The CPU default.
    #[default]
    Plain,
    /// Ordered boosting: growing body/tail segments; a tail document's
    /// approximant is estimated on the BODY prefix and never depends on itself
    /// (`approx_calcer.cpp:566-600`, ORD-02).
    Ordered,
}

/// Parameters for the plain boosting loop (the D-07 simplified isolating set).
///
/// No longer `Copy`: the CTR config carries an owned `Vec<f64>` of explicit
/// priors ([`Self::simple_ctr_priors`]); callers pass `&BoostParams` (as every
/// `train*` entry point already does) or `.clone()` it.
#[derive(Debug, Clone, PartialEq)]
pub struct BoostParams {
    /// Which loss / objective (RMSE or Logloss).
    pub loss: Loss,
    /// Number of boosting iterations (trees).
    pub iterations: usize,
    /// Tree depth (number of splits per tree; `2^depth` leaves).
    pub depth: usize,
    /// Learning rate scaling every leaf delta. Ignored when
    /// [`BoostParams::auto_learning_rate`] is `true` and the loss is auto-LR
    /// eligible (the value is then guessed pre-train via [`crate::autolr`]).
    pub learning_rate: f64,
    /// When `true`, the learning rate is selected automatically pre-train
    /// ([`crate::autolr`], TRAIN-08) — matching upstream's gate where
    /// `learning_rate` / `leaf_estimation_method` / `leaf_estimation_iterations`
    /// / `l2_leaf_reg` are all unset. The host caller maps "all four unset" to
    /// this flag; this struct carries concrete values for the latter three, so
    /// the flag is the single explicit auto-LR opt-in. When the loss is not in
    /// the auto-LR table (e.g. MAE) the explicit [`BoostParams::learning_rate`]
    /// is used unchanged (matches upstream `NeedToUpdate == false`).
    pub auto_learning_rate: bool,
    /// L2 leaf regularization (`l2_leaf_reg`).
    pub l2_leaf_reg: f64,
    /// Split-score perturbation strength (`random_strength`, TRAIN-05). `0.0`
    /// disables the perturbation (no normal draws — the first-slice path);
    /// non-zero turns on the per-candidate `TRandomScore::GetInstance` normal
    /// draw over the persistent RNG.
    pub random_strength: f64,
    /// Whether to start from the per-loss optimum constant approx (the target
    /// mean for RMSE), stored as the model bias. `false` starts from `0`.
    pub boost_from_average: bool,
    /// Which leaf-estimation method computes the per-leaf deltas (TRAIN-03 /
    /// D-09). The first-slice path is [`LeafMethod::Gradient`].
    pub leaf_method: LeafMethod,
    /// Bootstrap / sampling type (TRAIN-04). The first-slice path is
    /// [`EBootstrapType::No`].
    pub bootstrap_type: EBootstrapType,
    /// Object subsample fraction (`subsample`), used by Bernoulli and MVS. `1.0`
    /// disables subsampling. Ignored by `No`/`Bayesian`.
    pub subsample: f64,
    /// Bayesian bagging temperature (`bagging_temperature`); `0.0` makes Bayesian
    /// weights all `1.0`. Ignored by the other types.
    pub bagging_temperature: f32,
    /// The training random seed seeding the persistent sampling RNG
    /// (`random_seed`). Only consumed when `bootstrap_type != No`.
    pub random_seed: u64,
    /// Overfitting-detector type (`od_type`, TRAIN-06). [`EOverfittingDetectorType::None`]
    /// (or a non-positive `od_pval`) disables early stopping.
    pub od_type: EOverfittingDetectorType,
    /// Overfitting-detector stop threshold (`od_pval` / `AutoStopPValue`). `0`
    /// makes IncToDec / Wilcoxon inactive (the upstream default); Iter ignores it
    /// (the threshold is forced to `1.0`).
    pub od_pval: f64,
    /// Overfitting-detector wait iterations (`od_wait` / `IterationsWait`).
    pub od_wait: usize,
    /// `use_best_model`: when `true`, track the best eval-metric iteration and
    /// truncate the model's tree list to it (best_iteration + 1 trees).
    pub use_best_model: bool,
    /// The per-iteration eval-set validation metric (`eval_metric`, TRAIN-07).
    /// `None` defaults to the objective ([`EvalMetric::for_loss`]); `Some`
    /// overrides it. Only consumed when an eval set is supplied.
    pub eval_metric: Option<EvalMetric>,
    /// One-hot encoding threshold (`one_hot_max_size`,
    /// `cat_feature_options.cpp:231-232`, default 2 — pinned EXPLICITLY here,
    /// never auto-selected, RESEARCH Pitfall 6). A categorical column routes to
    /// the one-hot path when `1 < learn-set-cardinality <= one_hot_max_size`
    /// (inclusive boundary) and to the CTR path (deferred to later waves) when
    /// `cardinality > one_hot_max_size`. See [`crate::route_categorical`] /
    /// [`crate::EncodingPath`] (ORD-04 / D-04). Consumed by the categorical
    /// encoding-path selection; the numeric-only first slices leave it at the
    /// pinned default and never exercise the one-hot branch.
    pub one_hot_max_size: u32,
    /// Number of random permutations used by the multi-permutation fold
    /// machinery (`permutation_count`, default 4 — `boosting_options.cpp`).
    /// Pinned EXPLICITLY here, never auto-selected (RESEARCH Pitfall 6). The
    /// learning-fold count is `max(1, permutation_count - 1)` plus one averaging
    /// fold ([`crate::learning_fold_count`] / [`crate::create_folds`],
    /// `learn_context.cpp:48-49`). Consumed by ordered boosting / ordered CTR
    /// (later waves); the numeric/one-hot Plain slices need no permutation and
    /// leave it at the pinned default.
    pub permutation_count: usize,
    /// Tail-growth multiplier for the dynamic (ordered) fold body/tail
    /// (`fold_len_multiplier`, default 2.0 — `fold.cpp:39-41`
    /// `SelectTailSize(old, mult) = ceil(old * mult)`). Pinned EXPLICITLY
    /// (never auto). Consumed by [`crate::body_tail_boundaries`] /
    /// [`crate::create_folds`]; the plain single-span path ignores it.
    pub fold_len_multiplier: f64,
    /// The SINGLE `simple_ctr` type the high-cardinality categorical path bakes
    /// (ORD-03 / D-07 — one explicit CTR type per fixture, never the upstream
    /// auto default set `[Borders, Counter]`, RESEARCH Pitfall 6). Pinned
    /// EXPLICITLY ([`simple_ctr_default`]). Consumed by the Plain-CTR bake
    /// ([`crate::build_final_ctr`]); the numeric/one-hot slices leave it at the
    /// default and never exercise the CTR path.
    pub simple_ctr: ECtrType,
    /// The explicit per-prior numerators for [`Self::simple_ctr`] (D-07 — one
    /// prior per CTR column, never auto). Each entry is a unit-denominator prior
    /// numerator (`PriorDenom = 1`, RESEARCH A6 — so the online `+1` denom and
    /// the inference `+PriorDenom` coincide). Pinned EXPLICITLY
    /// ([`simple_ctr_priors_default`]).
    pub simple_ctr_priors: Vec<f64>,
    /// The `counter_calc_method` (`SkipTest` default, Pitfall 4 — pinned
    /// EXPLICITLY, never auto). In the whole-learn-set Plain build there are no
    /// test documents, so the flag does not change the counts; it is carried for
    /// the tensor-CTR path. [`counter_calc_method_default`].
    pub counter_calc_method: CounterCalcMethod,
    /// The boosting type ([`EBoostingType`], `boosting_options.cpp:16`). Pinned
    /// EXPLICITLY ([`boosting_type_default`] = [`EBoostingType::Plain`], the CPU
    /// default — Ordered auto-select is GPU-only, RESEARCH Pitfall 6). When
    /// [`EBoostingType::Ordered`] the ordered approximant path
    /// ([`ordered_approx_delta_simple`]) drives the anti-leakage body/tail update
    /// (ORD-02); the numeric/one-hot Plain slices leave it at the default.
    pub boosting_type: EBoostingType,
    /// The maximum feature-combination (tensor-CTR) projection length
    /// (`max_ctr_complexity` / upstream `MaxTensorComplexity`,
    /// `cat_feature_options.cpp:231-232`, default 4 — pinned EXPLICITLY here,
    /// never auto-selected, RESEARCH Pitfall 6). Bounds
    /// [`crate::TProjection::full_projection_length`] in
    /// [`crate::tensor_ctr_candidates`] (`GetFullProjectionLength` gate,
    /// `greedy_tensor_search.cpp:532-533`): `== 1` emits only SimpleCtrs, `>= 2`
    /// admits CombinationCtrs (tensors) of that length. The numeric/one-hot and
    /// single-feature CTR slices leave it at the pinned default and never form a
    /// combination (ORD-05 / D-05). [`max_ctr_complexity_default`].
    pub max_ctr_complexity: usize,
    /// The SINGLE `combinations_ctr` type the tensor-CTR (CombinationCtr) path
    /// bakes (ORD-05 / D-07 — one explicit CTR type per fixture, never the
    /// upstream auto default set, RESEARCH Pitfall 6). Pinned EXPLICITLY
    /// ([`combinations_ctr_default`]); the tensor CTR keys the SAME online/ordered
    /// accumulation (05-04/05-05) on the combined projection hash. The
    /// numeric/one-hot/simple-CTR slices leave it at the default and never
    /// exercise the combination path.
    pub combinations_ctr: ECtrType,
    /// The explicit per-prior numerators for [`Self::combinations_ctr`] (D-07 —
    /// one prior per combination CTR column, never auto; the tensor_ctr fixture
    /// pins `Borders:Prior=0.5`, so the online `+1` denom and the inference
    /// `+PriorDenom` coincide, RESEARCH A6). Pinned EXPLICITLY
    /// ([`combinations_ctr_priors_default`]).
    pub combinations_ctr_priors: Vec<f64>,
}

/// The canonical default `permutation_count` (`4`, `boosting_options.cpp`).
/// Pinned EXPLICITLY at every `BoostParams` construction site (RESEARCH
/// Pitfall 6 — never auto-selected).
#[must_use]
pub fn permutation_count_default() -> usize {
    4
}

/// The canonical default `fold_len_multiplier` (`2.0`, `fold.cpp:39-41`).
/// Pinned EXPLICITLY at every `BoostParams` construction site.
#[must_use]
pub fn fold_len_multiplier_default() -> f64 {
    2.0
}

/// The canonical default `simple_ctr` type ([`ECtrType::Borders`], the upstream
/// default CTR family head). Pinned EXPLICITLY at every `BoostParams`
/// construction site (RESEARCH Pitfall 6 — never auto-selected); the
/// numeric/one-hot slices leave it here and never exercise the CTR path.
#[must_use]
pub fn simple_ctr_default() -> ECtrType {
    ECtrType::Borders
}

/// The canonical default `simple_ctr` priors — a single unit-denominator prior
/// `0.5/1` (the in-scope plain_ctr fixture pins `Borders:Prior=0.5`, so the
/// online `+1` denom and the inference `+PriorDenom` coincide, RESEARCH A6).
/// Pinned EXPLICITLY at every `BoostParams` construction site.
#[must_use]
pub fn simple_ctr_priors_default() -> Vec<f64> {
    vec![0.5]
}

/// The canonical default `counter_calc_method` ([`CounterCalcMethod::SkipTest`],
/// `cat_feature_options.cpp:234`, Pitfall 4). Pinned EXPLICITLY (never auto).
#[must_use]
pub fn counter_calc_method_default() -> CounterCalcMethod {
    CounterCalcMethod::SkipTest
}

/// The canonical default `boosting_type` ([`EBoostingType::Plain`], the CPU
/// default — `boosting_options.cpp:16`; Ordered auto-select is GPU-only).
/// Pinned EXPLICITLY at every `BoostParams` construction site (RESEARCH
/// Pitfall 6 — never auto-selected).
#[must_use]
pub fn boosting_type_default() -> EBoostingType {
    EBoostingType::Plain
}

/// The canonical default `max_ctr_complexity` (`4`,
/// `cat_feature_options.cpp:231-232`; upstream `MaxTensorComplexity`). Pinned
/// EXPLICITLY at every `BoostParams` construction site (RESEARCH Pitfall 6 —
/// never auto-selected). Re-exports [`crate::projection::max_ctr_complexity_default`]
/// so the magic number lives in one place.
#[must_use]
pub fn max_ctr_complexity_default() -> usize {
    crate::projection::max_ctr_complexity_default()
}

/// The canonical default `combinations_ctr` type ([`ECtrType::Borders`], the
/// upstream default CTR family head). Pinned EXPLICITLY at every `BoostParams`
/// construction site (RESEARCH Pitfall 6 — never auto-selected); the
/// numeric/one-hot/simple-CTR slices leave it here and never exercise the
/// combination path.
#[must_use]
pub fn combinations_ctr_default() -> ECtrType {
    ECtrType::Borders
}

/// The canonical default `combinations_ctr` priors — a single unit-denominator
/// prior `0.5/1` (the in-scope tensor_ctr fixture pins `Borders:Prior=0.5`, so
/// the online `+1` denom and the inference `+PriorDenom` coincide, RESEARCH A6).
/// Pinned EXPLICITLY at every `BoostParams` construction site.
#[must_use]
pub fn combinations_ctr_priors_default() -> Vec<f64> {
    vec![0.5]
}

/// The canonical default Borders CTR border count (`15`, the upstream
/// `cat_feature_options.cpp` `ctr_border_count` default for the Borders CTR
/// family). Pinned EXPLICITLY by the caller (never auto-selected — RESEARCH
/// Pitfall 6); the materialized combined-projection online CTR feature is
/// quantized into `[0, 15]` integer CTR bins against this count
/// ([`crate::calc_ctr_online_bin`]).
#[must_use]
pub fn ctr_border_count_default() -> usize {
    15
}

/// The ORDERED-boosting per-object approximant delta for one tree iteration over
/// one body/tail segment (`UpdateApproxDeltasHistoricallyImpl`,
/// `approx_calcer.cpp:566-600`; the simple single-dim Gradient/Newton path,
/// `CalcApproxDeltaSimple` `:706`). This is the anti-leakage heart of ORD-02: a
/// TAIL document's approximant delta is estimated from the BODY prefix PLUS only
/// the tail documents that PRECEDE it in the permutation — it NEVER depends on
/// itself.
///
/// Walking the tail rows `[body_finish, tail_finish)` IN PERMUTATION (learn)
/// ORDER, the running per-leaf der/weight accumulator is seeded with the body
/// prefix sums (`body_sum_weight` and the body's per-leaf der sums), then each
/// successive tail row:
///   1. ADDS its own `der`/`weight` into its leaf's running sum (`AddMethodDer`),
///   2. computes the running delta `CalcMethodDelta(leafDer, l2, sumWeights)` —
///      for Gradient/RMSE that is `leafSumDer / (leafSumWeight + l2)` — using the
///      accumulator that NOW INCLUDES this row (upstream adds-then-reads), and
///   3. writes that delta to `approx_delta[row]`.
///
/// The "add then read" ordering is upstream-faithful (`:586-590`): the row's own
/// der enters its leaf sum before the delta is read, but because the delta is the
/// LEAF AVERAGE (a pooled statistic dominated by the body prefix), the row's
/// influence on its OWN delta vanishes as the body grows — the historical
/// (ordered) approximant. The body rows themselves keep delta `0` (they are the
/// estimation prefix, not updated here).
///
/// # Parameters
/// - `leaf_of[doc]` — object `doc`'s leaf index in the grown tree (OBJECT order).
/// - `der[doc]` — object `doc`'s first derivative (already weighted if weighted).
/// - `weights[doc]` — object `doc`'s weight (empty ⇒ all `1.0`).
/// - `permutation[p]` — the object at learn-order position `p`.
/// - `body_finish` / `tail_finish` — the segment boundary (learn-order positions).
/// - `_body_sum_weight` — the body prefix's summed weight (`fold.cpp:170-172`).
///   Part of the public signature (consumed by 05-05/05-10 wiring); the simple
///   Gradient delta reads the per-leaf running weight, so this prefix total is
///   not read here (WR-01 cleanup — the dead running-total accumulator that
///   carried it is removed). `_`-prefixed to mark it unused without changing the
///   parameter list/order callers depend on.
/// - `n_leaves` — the tree's leaf count.
/// - `scaled_l2` — the L2 regularizer ([`cb_compute::scale_l2_reg`]).
///
/// Returns the per-object approximant delta in OBJECT order (body rows and any
/// row outside `[0, n)` stay `0`). Every der/weight running sum routes through
/// integer-free `f64` accumulation seeded by the ordered [`sum_f64`] body sums
/// (D-08) — no hand-rolled whole-vector fold.
///
/// # Errors
/// [`CbError::Degenerate`] if `leaf_of` / `der` are shorter than the permutation
/// implies, or a permutation index is out of range.
#[allow(clippy::too_many_arguments)]
pub fn ordered_approx_delta_simple(
    leaf_of: &[usize],
    der: &[f64],
    weights: &[f64],
    permutation: &[i32],
    body_finish: usize,
    tail_finish: usize,
    _body_sum_weight: f64,
    n_leaves: usize,
    scaled_l2: f64,
) -> CbResult<Vec<f64>> {
    let n = permutation.len();
    if leaf_of.len() < n || der.len() < n {
        return Err(CbError::Degenerate(
            "ordered_approx: leaf_of / der shorter than permutation".to_owned(),
        ));
    }
    let mut approx_delta = vec![0.0f64; n];

    // Running per-leaf der/weight accumulator, seeded by the BODY prefix sums.
    let mut leaf_sum_der = vec![0.0f64; n_leaves];
    let mut leaf_sum_weight = vec![0.0f64; n_leaves];
    // Seed the body prefix: accumulate the first `body_finish` learn-order rows'
    // der/weight into their leaves (the estimation prefix the tail reads from).
    for p in 0..body_finish.min(n) {
        let Some(&doc_i) = permutation.get(p) else {
            break;
        };
        let doc = doc_i as usize;
        let (Some(&leaf), Some(&d)) = (leaf_of.get(doc), der.get(doc)) else {
            return Err(CbError::Degenerate(
                "ordered_approx: body permutation index out of range".to_owned(),
            ));
        };
        let w = if weights.is_empty() {
            1.0
        } else {
            weights.get(doc).copied().unwrap_or(1.0)
        };
        if let (Some(sd), Some(sw)) = (leaf_sum_der.get_mut(leaf), leaf_sum_weight.get_mut(leaf)) {
            *sd += d;
            *sw += w;
        }
    }

    // Walk the TAIL rows in permutation order; add-then-read the running delta.
    for p in body_finish..tail_finish.min(n) {
        let Some(&doc_i) = permutation.get(p) else {
            break;
        };
        let doc = doc_i as usize;
        let (Some(&leaf), Some(&d)) = (leaf_of.get(doc), der.get(doc)) else {
            return Err(CbError::Degenerate(
                "ordered_approx: tail permutation index out of range".to_owned(),
            ));
        };
        let w = if weights.is_empty() {
            1.0
        } else {
            weights.get(doc).copied().unwrap_or(1.0)
        };
        // AddMethodDer: this row's der/weight enters its leaf's running sum.
        if let (Some(sd), Some(sw)) = (leaf_sum_der.get_mut(leaf), leaf_sum_weight.get_mut(leaf)) {
            *sd += d;
            *sw += w;
        }
        // CalcMethodDelta (Gradient/RMSE simple path): leaf der / (leaf weight +
        // l2). The leaf running weight already includes this row + body prefix.
        let leaf_der = leaf_sum_der.get(leaf).copied().unwrap_or(0.0);
        let leaf_weight = leaf_sum_weight.get(leaf).copied().unwrap_or(0.0);
        let delta = gradient_leaf_delta(leaf_der, leaf_weight, scaled_l2);
        if let Some(slot) = approx_delta.get_mut(doc) {
            *slot = delta;
        }
    }

    Ok(approx_delta)
}

/// One trained oblivious tree: the ordered splits, the per-leaf values
/// (already scaled by `learning_rate`, matching upstream `model.json`), and the
/// per-leaf summed training-document weights (`leaf_weights`, RESEARCH Pitfall 1).
#[derive(Debug, Clone, PartialEq)]
pub struct ObliviousTree {
    /// The ordered FLOAT splits (feature + border) defining the symmetric
    /// structure. The numeric / one-hot / ordered boosting paths produce ONLY
    /// float splits here; tensor-CTR splits (when present) are carried separately
    /// in [`Self::ctr_splits`] so the widely-read `splits: Vec<Split>` surface the
    /// existing oracles consume stays byte-for-byte unchanged.
    pub splits: Vec<Split>,
    /// The ordered tensor / combination CTR splits chosen during tree growth
    /// (ORD-05 / D-05), one [`CtrSplitSpec`] per chosen CTR split. EMPTY for the
    /// numeric / one-hot / ordered-boosting paths (no CTR candidates emitted).
    /// `cb_model::Model::from_trained` lifts each into a `ModelSplit::Ctr`.
    pub ctr_splits: Vec<CtrSplitSpec>,
    /// Leaf values in canonical forward-bit-order, length `2^depth`.
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights in the same forward-bit-order
    /// as `leaf_values`, length `2^depth`. For unweighted training a leaf weight
    /// equals its document count (RESEARCH A4). Required by SHAP /
    /// PredictionValuesChange / Interaction (RESEARCH Pitfall 1).
    pub leaf_weights: Vec<f64>,
}

/// A trained plain-boosted model: the boosting-order trees plus the starting
/// approx (`boost_from_average`) stored as the model bias.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// The oblivious trees in boosting (iteration) order.
    pub oblivious_trees: Vec<ObliviousTree>,
    /// The starting approx / model bias.
    pub bias: f64,
}

impl Model {
    /// Per-tree split borders flattened in tree order (for
    /// `compare_stage(Stage::Splits, …)`).
    #[must_use]
    pub fn split_borders(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|t| t.splits.iter().map(|s| s.border))
            .collect()
    }

    /// Per-tree leaf values flattened in tree order (for
    /// `compare_stage(Stage::LeafValues, …)`).
    #[must_use]
    pub fn leaf_values(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|t| t.leaf_values.iter().copied())
            .collect()
    }

    /// Per-tree leaf weights flattened in tree order (RESEARCH Pitfall 1; for
    /// `compare_stage(Stage::LeafValues, …)` against the upstream `leaf_weights`).
    #[must_use]
    pub fn leaf_weights(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|t| t.leaf_weights.iter().copied())
            .collect()
    }
}

/// Map the boosting [`Loss`] to the auto-LR [`TargetType`] (upstream
/// `GetTargetType`, `options_helper.cpp:181-194`): RMSE -> RMSE, Logloss ->
/// Logloss, everything else (MAE / Quantile) -> [`TargetType::Unknown`] (not in
/// the auto-LR table, so no rate is guessed).
const fn autolr_target_type(loss: Loss) -> TargetType {
    match loss {
        Loss::Rmse => TargetType::Rmse,
        // CrossEntropy shares Logloss's auto-LR coefficient row (same objective
        // family); Focal is not in the upstream auto-LR table -> Unknown.
        Loss::Logloss | Loss::CrossEntropy => TargetType::Logloss,
        Loss::Focal { .. } | Loss::Mae => TargetType::Unknown,
    }
}

/// Compute the starting approx (and model bias): the target mean for RMSE with
/// `boost_from_average`, else `0` (Pitfall 2). The mean is folded through the
/// sanctioned `sum_f64` primitive (D-05).
fn starting_approx(params: &BoostParams, target: &[f64]) -> f64 {
    if params.boost_from_average && matches!(params.loss, Loss::Rmse) && !target.is_empty() {
        sum_f64(target) / target.len() as f64
    } else {
        0.0
    }
}

/// Compute the per-leaf deltas for the selected [`LeafMethod`] (TRAIN-03 / D-09).
///
/// Gradient/Newton/Simple are closed-form over each leaf's ordered reduced sums
/// (`cb_core::sum_f64` via `reduce_leaf_stats` / `reduce_leaf_der2`, D-05). Exact
/// takes the weighted median of each leaf's per-member residuals
/// (`target - approx`) via the quantile-style optimum. `weighted_der1[i]` is
/// `der1*weight`; `der2[i]` the per-object second derivative (weighted below for
/// the Newton sum); `approx`/`target` the running approximant/labels.
#[allow(clippy::too_many_arguments)]
fn compute_leaf_deltas(
    method: LeafMethod,
    leaf_of: &[usize],
    weighted_der1: &[f64],
    der2: &[f64],
    weights: &[f64],
    approx: &[f64],
    target: &[f64],
    scaled_l2: f64,
    n_leaves: usize,
) -> Vec<f64> {
    match method {
        LeafMethod::Gradient => {
            let stats = reduce_leaf_stats(leaf_of, weighted_der1, weights, n_leaves);
            stats
                .iter()
                .map(|s| gradient_leaf_delta(s.sum_weighted_delta, s.sum_weight, scaled_l2))
                .collect()
        }
        LeafMethod::Simple => {
            let stats = reduce_leaf_stats(leaf_of, weighted_der1, weights, n_leaves);
            stats
                .iter()
                .map(|s| simple_leaf_delta(s.sum_weighted_delta, s.sum_weight, scaled_l2))
                .collect()
        }
        LeafMethod::Newton => {
            let stats = reduce_leaf_stats(leaf_of, weighted_der1, weights, n_leaves);
            // Newton needs Σ der2*weight per leaf; build the weighted-der2 column
            // (elementwise product the host folds), then reduce ordered (D-05).
            let weighted_der2: Vec<f64> = der2
                .iter()
                .zip(weights.iter())
                .map(|(&d, &w)| d * w)
                .collect();
            let sum_der2 = reduce_leaf_der2(leaf_of, &weighted_der2, n_leaves);
            stats
                .iter()
                .zip(sum_der2.iter())
                .map(|(s, &d2)| newton_leaf_delta(s.sum_weighted_delta, d2, scaled_l2))
                .collect()
        }
        LeafMethod::Exact => {
            // Exact: per-leaf weighted median of residuals r_i = target_i -
            // approx_i (MAE / Quantile alpha=0.5, delta=1e-6). scaled_l2 is unused
            // (Exact has no L2 term — it is a rank statistic, not an average).
            let residuals: Vec<f64> = approx
                .iter()
                .zip(target.iter())
                .map(|(&a, &t)| t - a)
                .collect();
            let members = collect_leaf_residuals(leaf_of, &residuals, weights, n_leaves);
            members
                .iter()
                .map(|(r, w)| exact_leaf_delta(r, w, QUANTILE_ALPHA, QUANTILE_DELTA))
                .collect()
        }
    }
}

/// Accumulate per-leaf summed training-document weights (RESEARCH Pitfall 1,
/// `approx_calcer.cpp:154-160` = `leafWeights[leafIndex] += rowWeight`).
///
/// For each leaf, collect its member objects' weights (the FULL, un-sampled fold
/// weights used for leaf estimation) in object order, then reduce ordered through
/// the sanctioned `cb_core::sum_f64` primitive (D-08 — never a raw `iter().sum()`
/// / `fold(0.0, …)`). The result is in the same forward-bit-order as
/// `leaf_of` produces: `leaf_weights[leaf]` is `Σ weight` over members of `leaf`.
/// For unweighted training (`weights` all `1.0`) a leaf weight equals its
/// document count (RESEARCH A4).
fn accumulate_leaf_weights(leaf_of: &[usize], weights: &[f64], n_leaves: usize) -> Vec<f64> {
    // Bucket each leaf's member weights in object order (checked `.get` only —
    // `indexing_slicing` is deny).
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if let (Some(bucket), Some(&w)) = (members.get_mut(leaf), weights.get(i)) {
            bucket.push(w);
        }
    }
    members.iter().map(|bucket| sum_f64(bucket)).collect()
}

/// Map the tree's chosen tensor-CTR candidates into the persisted
/// [`CtrSplitSpec`] list (ORD-05 / D-05). For the numeric `train` driver the
/// `candidates` list is EMPTY (no categorical columns supply CTR-eligible
/// features), so this returns an empty `Vec` and the float-only oracles are
/// unchanged. The categorical train→predict path emits real candidates and (after
/// scoring the materialized combined-projection online CTR feature against
/// borders) records the chosen ones here; each carries its projection, the
/// `combinations_ctr` type, the prior, the per-class numerator selector, and the
/// CTR-value border.
///
/// `priors` is `params.combinations_ctr_priors` — the explicit per-prior
/// numerators (unit denominator, RESEARCH A6); the head prior (`0.5` for the
/// in-scope `Borders:Prior=0.5` fixture) seeds the spec. The split BORDER is left
/// `0.0` here (the candidate-emission stage); the categorical scorer overwrites it
/// with the chosen CTR-value threshold when a CTR split actually wins a level.
fn ctr_splits_for_tree(
    candidates: &[crate::candidates::CtrCandidate],
    priors: &[f64],
) -> Vec<CtrSplitSpec> {
    let prior_num = priors.first().copied().unwrap_or(0.5);
    candidates
        .iter()
        .map(|c| CtrSplitSpec {
            projection: c.projection.clone(),
            // combinations_ctr default head family is Borders (i8 == 0); pinned
            // explicitly at the BoostParams level (combinations_ctr_default).
            ctr_type: crate::ctr::ECtrType::Borders.as_i8(),
            prior_num,
            prior_denom: 1.0,
            target_border_idx: 0,
            border: 0.0,
        })
        .collect()
}

/// A held-out evaluation set feeding the overfitting detector (TRAIN-06). The
/// `feature_values` reuse the training feature borders (the model's float-feature
/// borders) for the `value > border` split tests.
pub struct EvalSet<'a> {
    /// `feature_values[f]` is eval float feature `f`'s per-object `f32` column.
    pub feature_values: &'a [Vec<f32>],
    /// Eval per-object target labels.
    pub target: &'a [f64],
}

/// Apply one oblivious tree to a single eval object: walk its splits to the leaf
/// and return that leaf's value. Out-of-range indices contribute `0` (defensive;
/// the trainer supplies valid trees).
fn tree_eval_contribution(tree: &ObliviousTree, matrix: &FeatureMatrix, obj: usize) -> f64 {
    let passes: Vec<bool> = tree
        .splits
        .iter()
        .map(|s| {
            matrix
                .feature_values
                .get(s.feature)
                .and_then(|col| col.get(obj))
                .is_some_and(|&v| f64::from(v) > s.border)
        })
        .collect();
    let leaf = leaf_index(&passes);
    tree.leaf_values.get(leaf).copied().unwrap_or(0.0)
}

/// Train a plain-boosted oblivious-tree model over the generic runtime `R`.
///
/// `feature_values[f]` is float feature `f`'s per-object `f32` column;
/// `feature_borders[f]` its ascending candidate borders (the model's float-feature
/// borders). `target`/`weights` are per-object; `staged_out`, when `Some`, is
/// filled with the per-iteration staged approximants (flat, `iterations * n`).
///
/// Delegates to [`train_with_eval_sets`] without an eval set (no early stopping).
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `params.depth > MAX_DEPTH`.
/// - [`CbError::Degenerate`] on an empty dataset or a level with no candidate
///   split.
/// - Any error the runtime's `compute_gradients` surfaces.
pub fn train<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    staged_out: Option<&mut Vec<f64>>,
) -> CbResult<Model> {
    train_with_eval_sets(
        runtime,
        feature_values,
        feature_borders,
        target,
        weights,
        params,
        staged_out,
        &[],
        None,
    )
}

/// Train with a SINGLE optional held-out eval set driving the overfitting
/// detector (TRAIN-06) and `use_best_model` truncation, plus an optional
/// `eval_loss_out` receiving the PRIMARY eval set's per-iteration `eval_metric`
/// curve (the detector's `AddError` sequence).
///
/// This is the single-eval-set convenience wrapper over [`train_with_eval_sets`]
/// (the TRAIN-06 entry point); the per-iteration eval value is now the formalized
/// `eval_metric` ([`crate::metrics`], TRAIN-07) rather than the Plan 05 inline
/// stub. When `params.od_type` is active the loop feeds the eval metric to the
/// detector and breaks on `IsNeedStop()`. When `params.use_best_model` is set the
/// model's trees are truncated to `best_iteration + 1` after the loop (upstream
/// `model.tree_count_` for a use_best_model run).
///
/// # Errors
/// As [`train`], plus any detector-construction error
/// ([`CbError::Degenerate`] for Wilcoxon without a test set) or a degenerate eval
/// set ([`CbError::Degenerate`] from the metric).
#[allow(clippy::too_many_arguments)]
pub fn train_with_eval<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    staged_out: Option<&mut Vec<f64>>,
    eval_set: Option<&EvalSet>,
    eval_loss_out: Option<&mut Vec<f64>>,
) -> CbResult<Model> {
    // Adapt the single eval set into the multi-set path. The primary (index 0)
    // set is the one the detector + best-model tracker consume; its per-iteration
    // metric curve is mirrored into `eval_loss_out` for backward compatibility.
    let sets: Vec<EvalSet> = eval_set
        .map(|es| {
            vec![EvalSet {
                feature_values: es.feature_values,
                target: es.target,
            }]
        })
        .unwrap_or_default();
    let mut history = eval_loss_out.as_ref().map(|_| EvalMetricHistory::new(sets.len()));
    let model = train_with_eval_sets(
        runtime,
        feature_values,
        feature_borders,
        target,
        weights,
        params,
        staged_out,
        &sets,
        history.as_mut(),
    )?;
    if let (Some(out), Some(h)) = (eval_loss_out, history) {
        out.clear();
        out.extend_from_slice(h.primary());
    }
    Ok(model)
}

/// Train with ZERO OR MORE held-out eval sets, computing the `eval_metric`
/// (TRAIN-07) over EACH set per iteration, logging the per-set per-iteration
/// values into `history`, and feeding the PRIMARY (index 0) set's metric to the
/// overfitting detector (TRAIN-06) + `use_best_model` tracker.
///
/// `eval_sets[0]` is the primary (validation_0) set the detector consumes;
/// further sets are logged only. `params.eval_metric` overrides the metric;
/// `None` defaults to the objective ([`EvalMetric::for_loss`]). When
/// `params.od_type` is active the loop breaks on `IsNeedStop()`; when
/// `params.use_best_model` is set the trees are truncated to `best_iteration + 1`.
///
/// This is the formalized replacement for the Plan 05 inline eval-set loss stub:
/// the metric set (multiple eval sets, `eval_metric` override, per-iteration
/// logging) lives in [`crate::metrics`]; the detector's stop/best-iteration path
/// is UNCHANGED — only the metric SOURCE changed.
///
/// # Errors
/// As [`train`], plus any detector-construction error
/// ([`CbError::Degenerate`] for Wilcoxon without a test set) or a degenerate eval
/// set ([`CbError::Degenerate`] from the metric).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn train_with_eval_sets<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    mut staged_out: Option<&mut Vec<f64>>,
    eval_sets: &[EvalSet],
    mut history: Option<&mut EvalMetricHistory>,
) -> CbResult<Model> {
    check_depth(params.depth)?;

    let n = target.len();
    if n == 0 {
        return Err(CbError::Degenerate("empty target".to_owned()));
    }

    // Automatic learning-rate selection (TRAIN-08): when the caller opted into
    // auto-LR AND the loss is in the upstream coefficient table, guess the rate
    // pre-train from (target, useBestModel, boostFromAverage, learnObjectCount,
    // iterations) — exactly upstream's `UpdateLearningRate` gate
    // (`options_helper.cpp:269-288`, fired when learning_rate /
    // leaf_estimation_method / leaf_estimation_iterations / l2_leaf_reg all
    // unset). When the loss is NOT auto-LR eligible the explicit
    // `params.learning_rate` is used unchanged (matches `NeedToUpdate == false`).
    let learning_rate = if params.auto_learning_rate {
        let target_type = autolr_target_type(params.loss);
        match autolr::guess(
            target_type,
            params.use_best_model,
            params.boost_from_average,
            n,
            params.iterations,
        ) {
            Ok(lr) => lr,
            // No coefficient row for this loss (Unknown target): keep the
            // explicit rate, matching upstream `NeedToUpdate == false`.
            Err(CbError::Degenerate(_)) => params.learning_rate,
            Err(e) => return Err(e),
        }
    } else {
        params.learning_rate
    };

    // Per-object weights: default to 1.0 when no weights are supplied.
    let weights: Vec<f64> = if weights.is_empty() {
        vec![1.0; n]
    } else {
        weights.to_vec()
    };
    let sum_all_weights = sum_f64(&weights);

    let bias = starting_approx(params, target);
    let mut approx = vec![bias; n];

    // Boosting type (ORD-02): the Plain path below estimates every document's
    // leaf delta on the whole fold (single body/tail span). The ORDERED path
    // (`EBoostingType::Ordered`) instead grows each tree's STRUCTURE via the
    // 05-08 ordered split-scoring subsystem
    // ([`greedy_tensor_search_oblivious_ordered`]) over the learning fold's
    // growing body/tail segments, then estimates the leaf VALUES on the AVERAGING
    // fold exactly as Plain (`CalcLeafValuesSimple` — leaf values are
    // Plain-identical; only the split scoring differs, STATE.md re-scope).
    // `params.boosting_type` is the explicit pin (never auto — Ordered
    // auto-select is GPU-only, Pitfall 6).
    //
    // FOLDS-BUILT-ONCE (learn_context.cpp:494-590): the fold set is created ONCE
    // here, BEFORE the tree-iteration loop, from the continuous-stream RNG
    // (`random_seed`) — the fold permutations are fixed for the whole run and are
    // NEVER redrawn per iteration. `create_folds` appears EXACTLY ONCE in this
    // production module (grep-enforced, FOLDS-BUILT-ONCE invariant). The Plain
    // path leaves `ordered_learning_perm` `None` and is byte-identical to before.
    let ordered_learning_perm: Option<Vec<i32>> = match params.boosting_type {
        EBoostingType::Plain => None,
        EBoostingType::Ordered => {
            // Build learning fold(s) (ordered ⇒ permutation needed, dynamic
            // body/tail) + one averaging fold. For permutation_count=1 →
            // learning_fold_count(1, true) == 1 learning fold + 1 averaging fold.
            let folds: Vec<Fold> = crate::fold::create_folds(
                n,
                params.permutation_count,
                /* permutation_needed_for_learning = */ true,
                /* dynamic_body_tail = */ true,
                params.fold_len_multiplier,
                params.random_seed,
            );
            // The learning fold (first non-averaging) supplies the object order
            // the ordered per-segment split score walks. Degenerate (no learning
            // fold) ⇒ surface a typed error rather than silently falling through.
            let perm = folds
                .iter()
                .find(|f| !f.is_averaging)
                .map(|f| f.permutation.clone())
                .ok_or_else(|| {
                    CbError::Degenerate("ordered boosting: no learning fold created".to_owned())
                })?;
            Some(perm)
        }
    };

    // Numeric-only training matrix (no categorical features in this path; the
    // one-hot categorical splits are exercised through the categorical-aware
    // tree search directly in the ORD-04 oracle test, D-04).
    let matrix = FeatureMatrix::new(feature_values, feature_borders);

    // Tensor / combination CTR candidate generation (ORD-05 / D-05, AddTreeCtrs,
    // greedy_tensor_search.cpp:491-551): emit the SimpleCtr / CombinationCtr
    // projections over the CTR-eligible cat features under the
    // `params.max_ctr_complexity` gate (:532-533). This whole-set numeric `train`
    // driver carries NO categorical columns (`cat_cardinalities = &[]`), so the
    // emission yields an EMPTY candidate set here and no CTR split is chosen —
    // the float-only oracles are byte-for-byte unchanged. The categorical
    // train→predict path (which supplies real cat columns, materializes each
    // candidate's combined-projection online CTR value via the 05-04/05/06
    // accumulation keyed on `TProjection::combined_hash`, scores it alongside the
    // float candidates, and persists a chosen CTR split as a `CtrSplitSpec`) is
    // exercised end-to-end by the `tensor_ctr_e2e` oracle. The candidate gate is
    // wired HERE so the `max_ctr_complexity` bound is honored at the train()
    // candidate-generation site (no unbounded enumeration, T-05-09-01).
    let cat_cardinalities: &[u32] = &[];
    let ctr_candidates =
        tensor_ctr_candidates(cat_cardinalities, params.one_hot_max_size, params.max_ctr_complexity);

    let n_leaves = 1usize << params.depth;
    let mut trees: Vec<ObliviousTree> = Vec::with_capacity(params.iterations);

    // Overfitting detection / use_best_model (TRAIN-06) + per-iteration eval-set
    // metric logging (TRAIN-07). The detector + best-model tracker consume the
    // PRIMARY (index 0) eval set's per-iteration `eval_metric`; ALL eval sets are
    // logged into `history`. Both are no-ops without any eval set. Each eval set's
    // raw approximant accumulates the bias plus every tree's leaf contribution as
    // trees are grown.
    //
    // `eval_metric` formalizes the Plan 05 inline eval-set loss STUB: the metric
    // (RMSE / Logloss, weighted, multi-set) lives in `crate::metrics`; it defaults
    // to the objective and may be overridden via `params.eval_metric`.
    let has_test = !eval_sets.is_empty();
    let eval_metric = params
        .eval_metric
        .unwrap_or_else(|| EvalMetric::for_loss(params.loss));
    let mut detector =
        OverfittingDetector::new(params.od_type, params.od_pval, params.od_wait, has_test)?;
    let mut best_model = BestModelTracker::new();
    let eval_matrices: Vec<FeatureMatrix> = eval_sets
        .iter()
        .map(|es| FeatureMatrix::new(es.feature_values, feature_borders))
        .collect();
    let mut eval_approx: Vec<Vec<f64>> = eval_sets
        .iter()
        .map(|es| vec![bias; es.target.len()])
        .collect();
    if let Some(h) = history.as_deref_mut() {
        *h = EvalMetricHistory::new(eval_sets.len());
    }

    // Persistent, continuously-advancing sampling RNG (`LearnProgress->Rand`,
    // seeded `random_seed`). Only consumed when bootstrap_type != No (Bayesian /
    // Bernoulli / MVS). The draw stream is NOT reseeded per tree (Pitfall 4).
    //
    // The bootstrap draws are NOT the only consumers of the persistent RNG:
    // upstream's per-iteration boosting body advances `LearnProgress->Rand` in a
    // FIXED pattern around each tree's `DoBootstrap` (train.cpp:206-243,
    // greedy_tensor_search.cpp:884,1916). Reproducing the draw ORDER (the parity
    // contract) requires consuming those non-bootstrap draws in the exact same
    // sequence so the bootstrap draws land on the correct RNG state every tree:
    //   * PRE-bootstrap, per iteration (train.cpp:208,211): `Rand.GenRand()`
    //     (fold pick `% foldCount`) + `Rand.GenRand()` (seed for
    //     `GenRandUI64Vector`) = [`PRE_TREE_DRAWS`] draws.
    //   * POST-bootstrap, per depth level (greedy_tensor_search.cpp:884):
    //     `CalcScores` draws ONE `Rand.GenRand()` per level (the
    //     random-strength seed, consumed even at `random_strength=0`) = `depth`
    //     draws per tree.
    let mut rng = TFastRng64::from_seed(params.random_seed);
    // The persistent RNG is consumed when EITHER sampling is active (bootstrap !=
    // No) OR the `random_strength` perturbation is on. With perturbation the
    // per-level `randSeed` draw and the `SelectBestCandidate` normal draws are
    // consumed INLINE by the perturbed tree search (in exact upstream order), so
    // the bulk POST per-level draws must NOT be applied in that case.
    let perturb_active = params.random_strength != 0.0;
    let draws_active = !matches!(params.bootstrap_type, EBootstrapType::No) || perturb_active;
    // MVS lambda for trees after the first uses the previous tree's mean leaf L2
    // norm (`CalculateLastIterMeanLeafValue`); `None` on the first tree.
    let mut prev_leaf_mean_l2: Option<f64> = None;

    for iter in 0..params.iterations {
        // 1. Per-object derivatives (UN-reduced; D-02) via the runtime kernel.
        let ders = runtime.compute_gradients(params.loss, &approx, target)?;

        // Weighted gradient contribution per object: der1 * weight (the
        // histogram-scatter elementwise product; the host reduces it ordered).
        let weighted_der1: Vec<f64> = ders
            .der1
            .iter()
            .zip(weights.iter())
            .map(|(&d, &w)| d * w)
            .collect();

        // 1a. PRE-bootstrap per-iteration draws (train.cpp:208,211): keep the RNG
        //     phase-aligned with upstream before the per-tree Bootstrap.
        if draws_active {
            for _ in 0..PRE_TREE_DRAWS {
                rng.gen_rand();
            }
        }

        // 1b. Bootstrap / sampling (TRAIN-04): once per tree, on the continuous
        //     RNG. MVS reads the weighted derivatives; the others ignore them.
        let sampled = bootstrap(
            params.bootstrap_type,
            &weighted_der1,
            params.subsample,
            params.bagging_temperature,
            prev_leaf_mean_l2,
            &mut rng,
        )?;

        // The SAMPLE WEIGHTS and CONTROL mask affect ONLY the SPLIT SCORING
        // (the `sampledDocs` histogram path); LEAF VALUES are estimated on the
        // FULL, UN-sampled AveragingFold derivatives (verified against upstream:
        // Bayesian/MVS sample weights never enter `CalcLeafValues`). So:
        //   * SCORE path: der1*weight*sampleWeight, restricted to control-true
        //     objects (zero score weight excludes an object from the ordered
        //     histogram reduction, exactly as `sampledDocs` drops it).
        //   * LEAF path: the raw weighted_der1 / weights (no sampling) —
        //     unchanged from the first slice.
        let score_weighted_der1: Vec<f64> = weighted_der1
            .iter()
            .zip(sampled.sample_weights.iter())
            .zip(sampled.control.iter())
            .map(|((&d, &sw), &c)| if c { d * sw } else { 0.0 })
            .collect();
        let score_weights: Vec<f64> = weights
            .iter()
            .zip(sampled.sample_weights.iter())
            .zip(sampled.control.iter())
            .map(|((&w, &sw), &c)| if c { w * sw } else { 0.0 })
            .collect();

        // 2. Grow one oblivious tree using the L2 split score over the ordered
        //    leaf-stat reduction (sampled subset / sample-weighted). When
        //    `random_strength != 0`, the per-candidate `TRandomScore` normal
        //    perturbation is drawn from the persistent RNG in upstream order
        //    (`scoreStDev = random_strength * derivativesStDevFromZero *
        //    modelSizeMultiplier`, `modelLength = iter * learning_rate`).
        //    `scoreStDev` / `derivativesStDevFromZero` is computed over the FULL,
        //    un-sampled AveragingFold derivatives (`weighted_der1`) — matching the
        //    LEAF path below and upstream `CalcDerivativesStDevFromZeroPlainBoosting`
        //    (greedy_tensor_search.cpp:92-107, which reads
        //    `fold.BodyTailArr.front().WeightedDerivatives`, the full fold). Only
        //    the split-scoring HISTOGRAM uses the masked `score_weighted_der1` /
        //    `score_weights` (the `sampledDocs` restriction). Feeding the masked
        //    vector into the std-dev biases it low whenever `bootstrap_type != No`
        //    drops objects (CR-01) — fixed here by passing `&weighted_der1`.
        let scaled_l2 = scale_l2_reg(params.l2_leaf_reg, sum_all_weights, n);
        let perturb = if perturb_active {
            let model_length = iter as f64 * learning_rate;
            let std_dev = score_st_dev(params.random_strength, &weighted_der1, model_length);
            Some(Perturbation {
                rng: &mut rng,
                score_st_dev: std_dev,
            })
        } else {
            None
        };
        let grown: GrownTree = match ordered_learning_perm.as_deref() {
            // ORDERED (ORD-02): grow the tree STRUCTURE via the 05-08 ordered
            // per-segment split-scoring subsystem over the learning fold's
            // BodyTailArr. At random_strength=0 + bootstrap_type=No there are no
            // perturbation/bootstrap draws, so the ordered split score consumes the
            // FULL (un-masked) `weighted_der1` / `weights` in learning-fold object
            // order; the function derives the body/tail segments + per-segment body
            // sum-weights internally from `fold_len_multiplier` (fold.rs, 05-03).
            // `leaf_of` is object-order (Plain-identical) so the SAME averaging-fold
            // leaf-value path below applies (`CalcLeafValuesSimple`).
            Some(learning_perm) => greedy_tensor_search_oblivious_ordered(
                &matrix,
                &weighted_der1,
                &weights,
                learning_perm,
                params.l2_leaf_reg,
                params.fold_len_multiplier,
                params.depth,
                n,
            )?,
            // PLAIN (unchanged): the perturbed whole-fold search over the
            // sampled/sample-weighted histogram (byte-identical to before).
            None => greedy_tensor_search_oblivious_perturbed(
                &matrix,
                &score_weighted_der1,
                &score_weights,
                scaled_l2,
                params.depth,
                n,
                perturb,
            )?,
        };

        // 3. Leaf values via the selected estimation method (TRAIN-03 / D-09),
        //    scaled by learning_rate (stored value matches model.json). Leaf
        //    estimation uses the FULL fold (all objects) with the RAW (un-sampled)
        //    derivatives/weights. Every reduction over leaf members routes through
        //    cb_core::sum_f64 (D-05).
        let leaf_deltas = compute_leaf_deltas(
            params.leaf_method,
            &grown.leaf_of,
            &weighted_der1,
            &ders.der2,
            &weights,
            &approx,
            target,
            scaled_l2,
            n_leaves,
        );
        let leaf_values: Vec<f64> = leaf_deltas
            .iter()
            .map(|&delta| learning_rate * delta)
            .collect();

        // Per-leaf summed training-document weights (RESEARCH Pitfall 1). Uses
        // the FULL un-sampled fold weights (same as leaf estimation), reduced
        // ordered through cb_core::sum_f64 (D-08).
        let leaf_weights = accumulate_leaf_weights(&grown.leaf_of, &weights, n_leaves);

        // 4. Update approx: approx[i] += leaf_value[leaf(i)].
        for (i, &leaf) in grown.leaf_of.iter().enumerate() {
            if let (Some(a), Some(&lv)) = (approx.get_mut(i), leaf_values.get(leaf)) {
                *a += lv;
            }
        }

        // Record the staged approximant for this iteration (raw value / logit).
        if let Some(out) = staged_out.as_deref_mut() {
            out.extend_from_slice(&approx);
        }

        // POST per-tree draws. Two distinct main-RNG consumers run AFTER the tree
        // structure is grown:
        //   (a) the per-level `CalcScores` randSeed (greedy_tensor_search.cpp:884)
        //       — ONE `Rand.GenRand()` per level; and
        //   (b) the leaf-estimation seed (train.cpp:303,
        //       `GenRandUI64Vector(foldCount, Rand.GenRand())`) — ONE
        //       `Rand.GenRand()` per TREE, drawn once the tree is built.
        // When the perturbation is OFF but sampling is on, (a) is not observable
        // individually, so the prior wave folds (a)+(b) into a single bulk
        // `depth + 1` advance that keeps the next tree's Bootstrap phase-aligned.
        // When the perturbation is ON, the perturbed search ALREADY consumed (a)'s
        // randSeed AND the `SelectBestCandidate` normal draws inline in exact
        // upstream order, so only (b) — the single leaf-estimation seed draw —
        // remains to be consumed here (train.cpp:303). This source-faithful draw
        // locks the FIRST tree end-to-end (splits + leaf values <= 1e-5); a
        // per-tree main-RNG phase drift remains for tree-1+ (the variable-length
        // normal-draw accounting could not be localized at tree granularity
        // without C++ instrumentation of `LearnProgress->Rand` — escalated to
        // D-11 / Open Q4, see the regularization oracle test header and SUMMARY).
        if perturb_active {
            for _ in 0..POST_TREE_EXTRA_DRAWS {
                rng.gen_rand();
            }
        } else if draws_active {
            for _ in 0..(params.depth + POST_TREE_EXTRA_DRAWS) {
                rng.gen_rand();
            }
        }

        // MVS lambda for the NEXT tree uses this tree's mean leaf L2 norm
        // (`CalculateLastIterMeanLeafValue`, mvs.cpp:21-35) over the stored
        // (learning_rate-scaled) leaf values.
        prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&leaf_values));

        // Persist any chosen tensor-CTR splits for this tree (ORD-05). With the
        // numeric `train` driver's empty candidate set this is EMPTY; the
        // categorical path supplies real candidates and a chosen CTR split is
        // lifted by `cb_model::Model::from_trained` into a `ModelSplit::Ctr`.
        let ctr_splits = ctr_splits_for_tree(&ctr_candidates, &params.combinations_ctr_priors);

        trees.push(ObliviousTree {
            splits: grown.splits,
            ctr_splits,
            leaf_values,
            leaf_weights,
        });

        // Overfitting detection / use_best_model (TRAIN-06): once the tree is
        // grown, update EACH eval set's raw approximant with this tree's leaf
        // contribution, compute the `eval_metric` over each set (TRAIN-07), log
        // the per-set per-iteration value, and feed the PRIMARY set's metric to
        // the detector + best-model tracker (TRAIN-06), breaking on IsNeedStop().
        if has_test {
            if let Some(tree) = trees.last() {
                for (set_idx, approx_col) in eval_approx.iter_mut().enumerate() {
                    if let Some(em) = eval_matrices.get(set_idx) {
                        for (obj, a) in approx_col.iter_mut().enumerate() {
                            *a += tree_eval_contribution(tree, em, obj);
                        }
                    }
                }
            }

            // The PRIMARY set's metric drives the stop decision (unchanged from
            // Plan 05 — only the metric source moved to `crate::metrics`).
            let mut primary_metric: Option<f64> = None;
            for (set_idx, es) in eval_sets.iter().enumerate() {
                if let Some(approx_col) = eval_approx.get(set_idx) {
                    // Eval sets carry no per-object weights in this phase — the
                    // metric uses uniform weight 1.0 (matching the upstream eval
                    // metric for unweighted eval pools).
                    let value = eval_metric.eval(approx_col, es.target, &[])?;
                    if let Some(h) = history.as_deref_mut() {
                        h.push(set_idx, value);
                    }
                    if set_idx == 0 {
                        primary_metric = Some(value);
                    }
                }
            }

            if let Some(value) = primary_metric {
                best_model.add_error(value);
                detector.add_error(value);
                if detector.is_need_stop() {
                    break;
                }
            }
        }
    }

    // use_best_model: truncate the model's trees to best_iteration + 1
    // (upstream `model.tree_count_` for a use_best_model run). Without an eval set
    // there is no best iteration, so the model keeps every grown tree.
    if params.use_best_model {
        if let Some(best) = best_model.best_iteration() {
            trees.truncate(best + 1);
        }
    }

    Ok(Model {
        oblivious_trees: trees,
        bias,
    })
}
