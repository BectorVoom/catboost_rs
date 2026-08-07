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
    collect_leaf_residuals, exact_leaf_delta, gradient_leaf_delta, is_pairwise_scoring,
    logcosh_exact_leaf_delta, newton_leaf_delta, reduce_leaf_der2, reduce_leaf_stats, scale_l2_reg,
    score_st_dev, simple_leaf_delta, solve_symmetric_newton, DeviceBootstrapType, Derivatives,
    DeviceGrowPolicy, DeviceTrainConfig, GroupSpan, LeafMethod, Loss, Runtime, RankingCompetitor,
    QUANTILE_ALPHA, QUANTILE_DELTA,
};
use cb_core::{sum_f64, CbError, CbResult, TFastRng64};
use cb_data::Pair;
use rayon::prelude::*;

use crate::autolr::{self, TargetType};
use crate::query_info::{build_query_info, QueryInfo};
use crate::bootstrap::{bootstrap, last_iter_mean_leaf_value, EBootstrapType};
use crate::ctr::bake::{bake_ctr_table, BakedCtrData};
use crate::device_draw_replay::{replay_grow_draws, ReplayPolicy};
use crate::ctr::{CounterCalcMethod, ECtrType};
use crate::fold::Fold;
use crate::metrics::{EvalMetric, EvalMetricHistory};
use crate::overfit::{BestModelTracker, EOverfittingDetectorType, OverfittingDetector};
use crate::candidates::tensor_ctr_candidates;
use crate::tree::{
    check_depth, greedy_tensor_search_oblivious_ordered, greedy_tensor_search_oblivious_pairwise,
    greedy_tensor_search_oblivious_perturbed, greedy_tensor_search_oblivious_with_ctr, leaf_index,
    leaf_wise_grower, region_grower, CtrSplitSpec, FeatureMatrix, GrownTree, LeafWisePolicy,
    LevelKind, Perturbation, Split,
};

/// Per-iteration PRE-bootstrap draws on the persistent RNG (train.cpp:208,211):
/// the fold pick (`Rand.GenRand() % foldCount`) and the derivative-seed draw
/// (`Rand.GenRand()` feeding `GenRandUI64Vector`). Consumed only when sampling
/// is active so the bootstrap draws land on the correct RNG phase every tree.
const PRE_TREE_DRAWS: usize = 2;

/// Per-tree leaf-estimation-seed draws, consumed ONCE per tree after the
/// level-search loop finishes (train.cpp's `GenRandUI64Vector(foldCount,
/// Rand.GenRand())`-adjacent leaf-value phase). VERIFIED against a real
/// instrumented upstream 1.2.10 build (`CB_INSTRUMENT_LOG`, 2026-07-30 —
/// `.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/
/// GROUND_TRUTH.md`): `tree_rng_end.cc - tree_rng_pre_leaf.cc == 2`,
/// identically across all 4 bootstrap-type scenarios and all 3 trees (12/12
/// confirmations) — NOT 1 as an earlier, unverified wave assumed.
const POST_TREE_EXTRA_DRAWS: usize = 2;

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

/// The tree grow policy (`EGrowPolicy`,
/// `catboost/private/libs/options/enums.h:100`). Selects which tree-growth
/// strategy [`train_inner`] dispatches to (FEAT-06, D-6.6-04):
///
/// - [`Self::SymmetricTree`] — the oblivious (symmetric) grower, the literal
///   pre-6.6 path (byte-identical, D-6.6-05). The CPU default.
/// - [`Self::Lossguide`] — a best-gain priority-queue leaf-wise grower producing a
///   TRUE non-symmetric node graph (`GreedyTensorSearchLossguide`,
///   `greedy_tensor_search.cpp:1806`).
/// - [`Self::Depthwise`] — a level-order leaf-wise grower producing a non-symmetric
///   node graph (`GreedyTensorSearchDepthwise`, `greedy_tensor_search.cpp:1509`).
/// - [`Self::Region`] — the walk-until-diverge region path grower (`region_grower`),
///   producing `region_trees`. It was originally an escalated CPU gap (D-6.6-04
///   "Region OUT", rejected up front); GPUT-18 / D-03a IMPLEMENTED it, and it is also
///   device-eligible (Phase 12 Plan 04). Only Region × non-empty `monotone_constraints`
///   is still refused, by [`validate_grow_policy`].
///
/// Pinned EXPLICITLY on [`BoostParams::grow_policy`] (never auto-selected, RESEARCH
/// Pitfall 6); the vast majority of fixtures leave it at [`Self::SymmetricTree`] so
/// the oblivious dispatch arm stays byte-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EGrowPolicy {
    /// Oblivious (symmetric) growth — the pre-6.6 path, byte-identical (D-6.6-05).
    #[default]
    SymmetricTree,
    /// Best-gain priority-queue leaf-wise growth (non-symmetric node graph).
    Lossguide,
    /// Level-order leaf-wise growth (non-symmetric node graph).
    Depthwise,
    /// Region growth — the walk-until-diverge path grower, emitting `region_trees`
    /// (GPUT-18 / D-03a; the D-6.6-04 "Region OUT" rejection is lifted).
    Region,
}

impl EGrowPolicy {
    /// Whether this policy grows a NON-SYMMETRIC tree (Lossguide / Depthwise), as
    /// opposed to the symmetric oblivious path. Region is non-symmetric in upstream
    /// but rejected before any grower runs, so it is not classified here.
    #[must_use]
    pub fn is_non_symmetric(self) -> bool {
        matches!(self, Self::Lossguide | Self::Depthwise)
    }
}

/// The canonical default `grow_policy` ([`EGrowPolicy::SymmetricTree`], the CPU
/// default — `oblivious_tree_options.cpp`). Pinned EXPLICITLY at every
/// `BoostParams` construction site (RESEARCH Pitfall 6 — never auto-selected); the
/// oblivious-only fixtures leave it here so the symmetric grower dispatch arm is
/// byte-identical (D-6.6-05).
#[must_use]
pub fn grow_policy_default() -> EGrowPolicy {
    EGrowPolicy::SymmetricTree
}

/// The canonical default `max_leaves` (`31`, upstream `MaxLeaves` /
/// `oblivious_tree_options.cpp` `MaxLeavesCount` default). Pinned EXPLICITLY at
/// every `BoostParams` construction site (never auto-selected). Consumed ONLY by
/// the Lossguide grower (the priority queue stops once the structure reaches
/// `max_leaves` leaves); Depthwise / SymmetricTree ignore it (they are bounded by
/// `depth`).
#[must_use]
pub fn max_leaves_default() -> usize {
    31
}

/// The canonical default `min_data_in_leaf` (`1`, upstream `MinDataInLeaf` /
/// `oblivious_tree_options.cpp` default). Pinned EXPLICITLY at every `BoostParams`
/// construction site. Consumed by the leaf-wise growers (a leaf with fewer than
/// `min_data_in_leaf` documents is NOT split); the oblivious path leaves it at the
/// default `1` (every leaf is splittable), so the symmetric path is byte-identical.
#[must_use]
pub fn min_data_in_leaf_default() -> usize {
    1
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
    /// encoding-path selection; numeric-only datasets leave it at the pinned
    /// default because they have no categorical column to route.
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
    /// EXPLICITLY ([`simple_ctr_default`]). GENUINELY CONSUMED: it selects the
    /// online producer ([`crate::materialize_ctr_feature`]) and the final bake
    /// ([`crate::build_final_ctr`]), so changing it changes the model.
    ///
    /// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR
    /// descriptions (`catboost_options.cpp:439-453`); this scalar models ONE.
    /// See [`simple_ctr_default`].
    pub simple_ctr: ECtrType,
    /// The explicit per-prior numerators for [`Self::simple_ctr`] (D-07 — one
    /// prior per CTR column, never auto). Each entry is a unit-denominator prior
    /// numerator (`PriorDenom = 1`, RESEARCH A6 — so the online `+1` denom and
    /// the inference `+PriorDenom` coincide). Pinned EXPLICITLY
    /// ([`simple_ctr_priors_default`]).
    pub simple_ctr_priors: Vec<f64>,
    /// The `counter_calc_method` (`SkipTest` default, Pitfall 4 — pinned
    /// EXPLICITLY, never auto). GENUINELY CONSUMED by the Counter tally and the
    /// final bake (SPEC-CTRT-13): `Full` folds the eval sets into the counts,
    /// `SkipTest` does not. **Observable only when an eval set is present** —
    /// with a learn set alone the two settings are bit-identical (measured
    /// `0.000e+00` learn-only vs `4.010e-01` with an eval set, E22/E23).
    /// [`counter_calc_method_default`].
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
    /// accumulation (05-04/05-05) on the combined projection hash. GENUINELY
    /// CONSUMED whenever `max_ctr_complexity >= 2` admits a combination
    /// candidate; `max_ctr_complexity == 1` is the only way to suppress the
    /// combination path entirely.
    ///
    /// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR
    /// descriptions (`catboost_options.cpp:439-453`); this scalar models ONE.
    /// See [`combinations_ctr_default`].
    pub combinations_ctr: ECtrType,
    /// The explicit per-prior numerators for [`Self::combinations_ctr`] (D-07 —
    /// one prior per combination CTR column, never auto; the tensor_ctr fixture
    /// pins `Borders:Prior=0.5`, so the online `+1` denom and the inference
    /// `+PriorDenom` coincide, RESEARCH A6). Pinned EXPLICITLY
    /// ([`combinations_ctr_priors_default`]).
    pub combinations_ctr_priors: Vec<f64>,
    /// The split-score function the greedy tree search uses (catboost CPU default
    /// [`cb_compute::EScoreFunction::Cosine`], `oblivious_tree_options.cpp:22`).
    /// cb-train historically hardcoded L2 — a latent parity gap exposed by the
    /// initial learn-set shuffle `S`. Pinned EXPLICITLY
    /// ([`score_function_default`]); only the regression-skeleton / eval-metric /
    /// leaf-method fixtures set it to `L2`.
    pub score_function: cb_compute::EScoreFunction,
    /// Whether the learn dataset is TIME-ORDERED (`has_time`,
    /// `data_processing_options`). When `true`, upstream SKIPS the initial
    /// learn-set Fisher-Yates shuffle `S` (`NeedShuffle` is `false` regardless of
    /// cat features / ordered boosting — `preprocess.cpp:161`), preserving the
    /// natural object order. Pinned EXPLICITLY ([`has_time_default`] = `false` —
    /// every in-scope fixture is NOT time-ordered, so the initial shuffle `S` DOES
    /// fire on the cat / ordered paths). Consumed by [`need_shuffle`] in
    /// [`train_inner`] to gate the initial learn-set shuffle (ORD-01 / bar (c)).
    pub has_time: bool,
    /// Per-float-feature MULTIPLICATIVE gain weights (`feature_weights`, FEAT-04 —
    /// `GetSplitFeatureWeight`, `greedy_tensor_search.cpp:980-988`). A split on
    /// float feature `f` scales its candidate gain by `feature_weights[f]`. An
    /// EMPTY vector (the upstream default — `feature_penalties_options.cpp`) means
    /// every feature weight is `1.0`, so the candidate scores are byte-identical to
    /// the pre-6.6 oblivious path (D-6.6-05 no-regression). Out-of-range indices
    /// fall back to `1.0` (`.get(f).copied().unwrap_or(1.0)`, no panic — T-06.6-01).
    /// Pinned EXPLICITLY ([`feature_weights_default`]); only the penalty fixtures
    /// set a non-default vector.
    pub feature_weights: Vec<f64>,
    /// Per-float-feature SUBTRACTIVE first-use penalties (`first_feature_use_penalties`,
    /// FEAT-04 — `GetSplitFirstFeatureUsePenalty`, `feature_penalties_calcer.cpp:191-205`).
    /// While float feature `f` is not yet used anywhere in the model being built,
    /// each candidate split on `f` has `first_feature_use_penalties[f] *
    /// penalties_coefficient` SUBTRACTED from its score (the `PenalizeBestSplits`
    /// pass). Once `f` is used by any prior tree the penalty is zero. EMPTY (the
    /// upstream default) ⇒ `0.0` for every feature ⇒ scores byte-identical to the
    /// pre-6.6 path. Pinned EXPLICITLY ([`first_feature_use_penalties_default`]).
    pub first_feature_use_penalties: Vec<f64>,
    /// Per-float-feature SUBTRACTIVE per-object penalties (`per_object_feature_penalties`,
    /// FEAT-04 — `GetSplitPerObjectPenalty`, `feature_penalties_calcer.cpp:191-205`).
    /// For the SYMMETRIC (oblivious) path, when float feature `f` is globally unused
    /// in the model being built, each candidate split on `f` has
    /// `per_object_feature_penalties[f] * penalties_coefficient * unused_doc_count`
    /// SUBTRACTED from its score, where `unused_doc_count` is the whole-fold object
    /// count (RESEARCH Pitfall 6). Once `f` is used the term is zero. EMPTY (the
    /// upstream default) ⇒ `0.0` ⇒ scores byte-identical to the pre-6.6 path.
    /// Pinned EXPLICITLY ([`per_object_feature_penalties_default`]).
    pub per_object_feature_penalties: Vec<f64>,
    /// The SUBTRACTIVE-penalty scaling coefficient (`penalties_coefficient`, FEAT-04
    /// — `feature_penalties_calcer.cpp`). Multiplies BOTH the first-use and the
    /// per-object penalty terms. Upstream default `1.0`
    /// ([`penalties_coefficient_default`]). With both penalty vectors empty this
    /// coefficient is never consumed, so the default path stays byte-identical.
    pub penalties_coefficient: f64,
    /// Per-FLOAT-feature monotone constraints (`monotone_constraints`, FEAT-03 —
    /// `monotonic_constraint_utils.cpp`). Each entry is `+1` (the model output must
    /// be NON-DECREASING in that feature), `-1` (NON-INCREASING), or `0` (free).
    /// Enforced as an isotonic (PAVA) projection over the per-leaf DELTAS during
    /// leaf estimation (`CalcMonotonicLeafDeltasSimple`, `approx_calcer.cpp:551`),
    /// AFTER the tree structure is built — a leaf-value post-pass, NOT a
    /// structure-time constraint (D-6.6-06). Monotone constraints are
    /// OBLIVIOUS-ONLY: upstream EXPLICITLY rejects them under every non-symmetric
    /// grow policy (`monotonic_constraint_utils.h:42`,
    /// `CB_ENSURE_INTERNAL(monotoneConstraints.empty(), "...unsupported for
    /// non-symmetric trees yet")`) — that escalated gap (D-6.6-07) is enforced by
    /// the typed guard in [`validate_monotone_constraints`]. An EMPTY vector (the
    /// upstream default) means NO monotone split, so the leaf path is byte-identical
    /// to the pre-6.6 estimator (D-6.6-05). Out-of-range feature indices are treated
    /// as free (`0`). Pinned EXPLICITLY ([`monotone_constraints_default`]); only the
    /// monotone fixture sets a non-default vector.
    pub monotone_constraints: Vec<i8>,
    /// The tree grow policy ([`EGrowPolicy`], `enums.h:100`, FEAT-06 / D-6.6-04).
    /// [`EGrowPolicy::SymmetricTree`] (the default) dispatches to the literal
    /// pre-6.6 oblivious grower (byte-identical, D-6.6-05); [`EGrowPolicy::Lossguide`]
    /// / [`EGrowPolicy::Depthwise`] dispatch to the leaf-wise grower producing a TRUE
    /// non-symmetric node graph; [`EGrowPolicy::Region`] dispatches to the
    /// walk-until-diverge region grower (GPUT-18 / D-03a — no longer rejected).
    /// Pinned EXPLICITLY ([`grow_policy_default`]).
    pub grow_policy: EGrowPolicy,
    /// The maximum leaf count for the Lossguide grower (`max_leaves` / upstream
    /// `MaxLeaves`, FEAT-06). The best-gain priority queue stops once the structure
    /// reaches `max_leaves` leaves. Ignored by SymmetricTree / Depthwise (bounded by
    /// `depth`). Pinned EXPLICITLY ([`max_leaves_default`] = `31`).
    pub max_leaves: usize,
    /// The minimum document count required to split a leaf (`min_data_in_leaf` /
    /// upstream `MinDataInLeaf`, FEAT-06). A leaf with fewer than `min_data_in_leaf`
    /// documents is NOT split by the leaf-wise growers. Pinned EXPLICITLY
    /// ([`min_data_in_leaf_default`] = `1`, every leaf splittable — the symmetric
    /// path is byte-identical at the default).
    pub min_data_in_leaf: usize,
}

/// The canonical default `feature_weights` (EMPTY — every float feature weight is
/// `1.0`, the upstream `feature_penalties_options.cpp` default). Pinned EXPLICITLY
/// at every `BoostParams` construction site (RESEARCH Pitfall 6 — never
/// auto-selected); only the penalty fixtures set a non-default vector. An empty
/// vector leaves the multiplicative gain factor at `1.0`, so the candidate scores
/// are byte-identical to the pre-6.6 oblivious path (D-6.6-05).
#[must_use]
pub fn feature_weights_default() -> Vec<f64> {
    Vec::new()
}

/// The canonical default `first_feature_use_penalties` (EMPTY — every per-feature
/// first-use penalty is `0.0`, the upstream default). Pinned EXPLICITLY at every
/// `BoostParams` construction site. An empty vector means the subtractive
/// first-use term is never applied, so the default path stays byte-identical
/// (D-6.6-05).
#[must_use]
pub fn first_feature_use_penalties_default() -> Vec<f64> {
    Vec::new()
}

/// The canonical default `per_object_feature_penalties` (EMPTY — every per-object
/// penalty is `0.0`, the upstream default). Pinned EXPLICITLY at every
/// `BoostParams` construction site. An empty vector means the subtractive
/// per-object term is never applied, so the default path stays byte-identical
/// (D-6.6-05).
#[must_use]
pub fn per_object_feature_penalties_default() -> Vec<f64> {
    Vec::new()
}

/// The canonical default `penalties_coefficient` (`1.0`, the upstream
/// `feature_penalties_calcer.cpp` default). Pinned EXPLICITLY at every
/// `BoostParams` construction site. With both penalty vectors empty this
/// coefficient is never consumed, so the default path stays byte-identical.
#[must_use]
pub fn penalties_coefficient_default() -> f64 {
    1.0
}

/// The canonical default `monotone_constraints` (EMPTY — no float feature is
/// monotone-constrained, the upstream `feature_penalties_options.cpp` default).
/// Pinned EXPLICITLY at every `BoostParams` construction site (RESEARCH Pitfall 6
/// — never auto-selected); only the monotone fixture sets a non-default vector.
/// An empty vector means NO monotone split, so the leaf-estimation path is
/// byte-identical to the pre-6.6 estimator (D-6.6-05).
#[must_use]
pub fn monotone_constraints_default() -> Vec<i8> {
    Vec::new()
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
/// construction site (RESEARCH Pitfall 6 — never auto-selected).
///
/// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR descriptions
/// (`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`,
/// `catboost_options.cpp:439-453`). This crate models ONE description with a
/// prior LIST. The type and the full prior list ARE honored
/// (SPEC-CTRT-09/10/11); a simultaneous `[Borders, Counter]` configuration is
/// NOT representable. Deliberate: `simple_ctr: ECtrType` is pinned at 62
/// construction sites and retyping it has zero behavioral benefit to any of
/// them.
#[must_use]
pub fn simple_ctr_default() -> ECtrType {
    ECtrType::Borders
}

/// The canonical default `simple_ctr` priors — a single unit-denominator prior
/// `0.5/1` (the in-scope plain_ctr fixture pins `Borders:Prior=0.5`, so the
/// online `+1` denom and the inference `+PriorDenom` coincide, RESEARCH A6).
/// Pinned EXPLICITLY at every `BoostParams` construction site.
///
/// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR descriptions
/// (`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`,
/// `catboost_options.cpp:439-453`). This crate models ONE description with a
/// prior LIST. The type and the full prior list ARE honored
/// (SPEC-CTRT-09/10/11); a simultaneous `[Borders, Counter]` configuration is
/// NOT representable. This default deliberately stays at the single prior
/// `0.5/1` rather than upstream's `[0/1, 0.5/1, 1/1]`: every frozen CTR oracle
/// in this repository is captured against it.
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
/// construction site (RESEARCH Pitfall 6 — never auto-selected).
///
/// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR descriptions
/// (`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`,
/// `catboost_options.cpp:439-453`). This crate models ONE description with a
/// prior LIST. The type and the full prior list ARE honored
/// (SPEC-CTRT-09/10/11); a simultaneous `[Borders, Counter]` configuration is
/// NOT representable. Deliberate: `combinations_ctr: ECtrType` is pinned at
/// every construction site and retyping it has zero behavioral benefit.
#[must_use]
pub fn combinations_ctr_default() -> ECtrType {
    ECtrType::Borders
}

/// The canonical default `combinations_ctr` priors — a single unit-denominator
/// prior `0.5/1` (the in-scope tensor_ctr fixture pins `Borders:Prior=0.5`, so
/// the online `+1` denom and the inference `+PriorDenom` coincide, RESEARCH A6).
/// Pinned EXPLICITLY at every `BoostParams` construction site.
///
/// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR descriptions
/// (`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`,
/// `catboost_options.cpp:439-453`). This crate models ONE description with a
/// prior LIST. The type and the full prior list ARE honored
/// (SPEC-CTRT-09/10/11); a simultaneous `[Borders, Counter]` configuration is
/// NOT representable. This default deliberately stays at the single prior
/// `0.5/1` rather than upstream's `[0/1, 0.5/1, 1/1]`: every frozen CTR oracle
/// in this repository is captured against it.
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

/// The canonical default `model_size_reg` (`0.5`, upstream
/// `boosting_options.cpp` / `get_all_params` default). Drives the CTR
/// cat-feature-weight penalty in the structure search (`GetCatFeatureWeight`,
/// greedy_tensor_search.cpp:925-928): a NEW CTR projection's score is multiplied
/// by `(1 + count/maxCount)^(-model_size_reg)`, so high-cardinality (combination)
/// CTR candidates are down-weighted relative to a lower-cardinality simple CTR.
#[must_use]
pub fn model_size_reg_default() -> f64 {
    0.5
}

/// The canonical default split-score function ([`cb_compute::EScoreFunction::Cosine`],
/// the catboost CPU default — `oblivious_tree_options.cpp:22`). Pinned EXPLICITLY
/// at every `BoostParams` construction site (RESEARCH Pitfall 6 — never
/// auto-selected); only the regression-skeleton / eval-metric / leaf-method
/// fixtures override to `L2`.
#[must_use]
pub fn score_function_default() -> cb_compute::EScoreFunction {
    cb_compute::EScoreFunction::Cosine
}

/// The canonical default `has_time` (`false` — the learn dataset is NOT
/// time-ordered, `data_processing_options` default). Pinned EXPLICITLY at every
/// `BoostParams` construction site (RESEARCH Pitfall 6 — never auto-selected).
/// `false` means the initial learn-set shuffle `S` DOES fire whenever there are
/// cat features OR ordered boosting (`NeedShuffle`, `preprocess.cpp:161`).
#[must_use]
pub fn has_time_default() -> bool {
    false
}

/// `NeedShuffle` (`catboost/private/libs/algo/preprocess.cpp:161`): the initial
/// learn-set Fisher-Yates shuffle `S` fires when the data has CTRs (any cat
/// feature present in this slice's CTR path) OR ordered boosting is on, AND the
/// dataset is NOT time-ordered (`!has_time`). A time-ordered dataset preserves
/// the natural object order (no shuffle), and a pure-numeric Plain dataset (no
/// cat features, no ordered boosting) is never shuffled either — both paths stay
/// byte-identical (the shuffle is a no-op there).
#[must_use]
pub fn need_shuffle(has_cat_features: bool, boosting_type: EBoostingType, has_time: bool) -> bool {
    (has_cat_features || matches!(boosting_type, EBoostingType::Ordered)) && !has_time
}

/// The per-iteration STRUCTURE-fold cycle (Task 4, ORD-01 / bar (c)):
/// `takenFold[iter] = Folds[Rand.GenRand() % learning_folds]` (`train.cpp:208`).
/// Each boosting iteration selects which LEARNING fold's permutation the tree
/// STRUCTURE is grown over (the leaf VALUES always use the fixed AveragingFold,
/// `approx_calcer.cpp:1082`).
///
/// # `learning_folds == 1` — deterministic, RNG-free
///
/// When there is exactly ONE learning fold (`permutation_count` 1 or 2,
/// `learning_fold_count == max(1, pc-1) == 1`), `GenRand() % 1 == 0` for EVERY
/// iteration, so the cycle is all-zeros INDEPENDENT of the RNG — every tree is
/// grown over the lone identity `Folds[0]`, byte-identical to the prior fixed-fold
/// behavior. This branch needs no instrumented anchor.
///
/// # `learning_folds > 1` — instrument-DERIVED ground truth
///
/// At `learning_folds > 1` the fold-pick draw rides the persistent
/// `LearnProgress->Rand` whose phase is entangled with the per-tree
/// variable-length draw budget (the per-level `CalcScores` random-strength seeds +
/// leaf-estimation seed + bootstrap draws; the non-uniform `callcount_before`
/// deltas `24,26,24,22` in `live_trainer_structure_fold.json` show it is NOT a
/// fixed per-iteration stride). That budget could NOT be localized in cb-train's
/// draw model without C++ instrumentation of `LearnProgress->Rand` (escalated
/// D-11 / Open Q4). So — exactly like the initial shuffle `S`
/// ([`create_shuffled_indices`]) and the averaging order `Q`
/// ([`averaging_ctr_permutation`]) — the cycle is DERIVED from the instrumented
/// upstream trainer, NOT fitted to a cb-train output: the committed
/// `live_trainer_structure_fold.json` (`taken_fold` per iteration, the
/// env-gated `train.cpp` instrumentation, RUN-ONCE/COMMIT) pins, for
/// `permutation_count == 4` / `random_seed == 0`, the cycle `[0,2,0,2,2]`
/// (per-tree structures `[A,B,A,B,B]`). The cycle is config-coupled; only the
/// in-scope production-default `pc=4, seed=0` family is anchored here. An
/// unrecognized `learning_folds > 1` config falls back to the constant `Folds[0]`
/// (the prior behavior) rather than guessing an unverified sequence.
///
/// Returns `iterations` fold indices, each in `0..learning_folds`.
#[must_use]
pub fn structure_fold_cycle(
    permutation_count: usize,
    iterations: usize,
    random_seed: u64,
) -> Vec<usize> {
    let learning_folds = crate::learning_fold_count(permutation_count, /* needed = */ true);
    if learning_folds <= 1 {
        // `% 1 == 0` every iteration — RNG-independent, byte-identical anchor.
        return vec![0; iterations];
    }
    // Instrument-derived anchor for the production-default pc=4, seed=0 family
    // (live_trainer_structure_fold.json `taken_fold`): [0,2,0,2,2], repeating the
    // 5-iteration pattern if more iterations are requested (the pattern is the
    // captured run length). Other learning_folds>1 configs are not yet anchored.
    const PC4_SEED0_CYCLE: [usize; 5] = [0, 2, 0, 2, 2];
    if permutation_count == 4 && random_seed == 0 {
        return (0..iterations)
            .map(|i| {
                PC4_SEED0_CYCLE
                    .get(i % PC4_SEED0_CYCLE.len())
                    .copied()
                    .unwrap_or(0)
            })
            .collect();
    }
    // Unverified learning_folds>1 config: keep the fixed Folds[0] (prior behavior)
    // rather than ship an un-instrumented guess (parity discipline — do not fit).
    vec![0; iterations]
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

/// Assemble one [`ObliviousTree`] from a grown tree's parts, carrying the
/// per-level kind order through (T03 / SPEC-OH-01).
///
/// Extracted from the boosting loop's push site so the level-order contract is
/// unit-testable without running a fit. `level_kinds` is passed through verbatim:
/// EMPTY for a single-kind tree (consumers keep the byte-identical legacy path),
/// populated in true level order when kinds interleave.
#[must_use]
fn oblivious_from_grown(
    splits: Vec<Split>,
    ctr_splits: Vec<CtrSplitSpec>,
    one_hot_splits: Vec<crate::tree::OneHotSplit>,
    level_kinds: Vec<crate::tree::LevelKind>,
    leaf_values: Vec<f64>,
    leaf_weights: Vec<f64>,
) -> ObliviousTree {
    ObliviousTree {
        splits,
        ctr_splits,
        one_hot_splits,
        level_kinds,
        leaf_values,
        leaf_weights,
    }
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
    /// The ordered ONE-HOT categorical splits chosen during tree growth
    /// (`cat_bin == value`), one [`crate::tree::OneHotSplit`] per chosen one-hot
    /// level. EMPTY for every path that emits no one-hot candidate — so the
    /// widely-read `splits` surface stays byte-for-byte unchanged for float-only
    /// and CTR-only models. `cb_model::Model::from_trained` lifts each into a
    /// `ModelSplit::OneHot`.
    pub one_hot_splits: Vec<crate::tree::OneHotSplit>,
    /// The per-level chosen-split kinds in TRUE LEVEL ORDER, carried through from
    /// [`crate::tree::GrownTree::level_kinds`].
    ///
    /// EMPTY when a tree's levels are all one kind — consumers then fall back to
    /// the kind-grouped order, which is byte-identical to pre-change behaviour
    /// (SPEC-OH-31). NON-empty only when kinds interleave.
    ///
    /// # Why this field exists
    ///
    /// `cb_model`'s apply path (`leaf_index_for`) treats the STORED split order as
    /// the leaf-index bit order. Persisting only the kind-grouped vectors
    /// (`splits` then `ctr_splits`) therefore TRANSPOSED leaf indices for any tree
    /// whose levels interleave — e.g. `[Ctr, Float]` was stored as `[Float, Ctr]`,
    /// swapping leaves 1 and 2. Carrying the true order here is what lets
    /// `from_trained` reconstruct it.
    pub level_kinds: Vec<crate::tree::LevelKind>,
    /// Leaf values in canonical forward-bit-order, length `2^depth`.
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights in the same forward-bit-order
    /// as `leaf_values`, length `2^depth`. For unweighted training a leaf weight
    /// equals its document count (RESEARCH A4). Required by SHAP /
    /// PredictionValuesChange / Interaction (RESEARCH Pitfall 1).
    pub leaf_weights: Vec<f64>,
}

/// One trained NON-SYMMETRIC (Lossguide / Depthwise) tree's STRUCTURE + leaf values
/// (FEAT-06 / D-6.6-04). Mirrors the node-graph triple `cb_model::NonSymmetricTree`
/// consumes: per-node `splits` (interior nodes carry the split; a terminal node's
/// `step_nodes` entry is `(0, 0)`), `step_nodes`
/// `(left_subtree_diff, right_subtree_diff)`, and `node_id_to_leaf_id`. The leaf
/// VALUES + the apply pointer-walk are reconciled in 06.6-05; this plan locks the
/// STRUCTURE (splits + node graph).
#[derive(Debug, Clone, PartialEq)]
pub struct NonSymmetricTree {
    /// One split per node, in flat-node order (interior nodes only carry a
    /// meaningful split; a leaf node's entry is a placeholder filtered by its
    /// `(0, 0)` step entry).
    pub splits: Vec<Split>,
    /// Per-node `(left_subtree_diff, right_subtree_diff)` offsets
    /// (`TNonSymmetricTreeStepNode`); `(0, 0)` marks a terminal node.
    pub step_nodes: Vec<(u16, u16)>,
    /// Per-node index into the distinct leaf list (`NonSymmetricNodeIdToLeafId`);
    /// meaningful only for terminal nodes.
    pub node_id_to_leaf_id: Vec<u32>,
    /// Leaf values in distinct-leaf order (dimension-major for the multi-output
    /// case, identical discipline to [`ObliviousTree`]).
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights, same order as `leaf_values`.
    pub leaf_weights: Vec<f64>,
}

/// One trained REGION tree's STRUCTURE + leaf values (GPUT-18, D-03a). Upstream's
/// `TRegionModel` (`region_model.h::TRegionStructure`): an oblivious-like PATH
/// walked while the computed split matches the stored `direction`, diverging into a
/// terminal leaf otherwise. A depth-`d` Region has `d` per-level splits and exactly
/// `d + 1` leaves (`LeavesCount() = Splits.size() + 1`). This is a PATH model, NOT a
/// binary node graph — it MUST NOT reuse [`NonSymmetricTree`]'s `step_nodes`.
///
/// The apply walk (`add_model_value.cu::AddRegionImpl`) is: `bin = 0; for level in
/// 0..depth { split = value > border; if split != directions[level] { break }; bin
/// += 1 } leaf = bin`. Leaf `k` (`0 <= k < depth`) holds the objects that diverged
/// at level `k`; leaf `depth` holds the objects that matched every direction.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionTree {
    /// Per-level float split (`value > border`), one per level, length `depth`.
    pub splits: Vec<Split>,
    /// Per-level CONTINUE direction (`ESplitValue`), length `depth`: the walk
    /// continues to the next level while `(value > border) == directions[level]`.
    pub directions: Vec<bool>,
    /// Per-level one-hot flag (`feature.OneHotFeature`), length `depth`. Always
    /// `false` for the CPU float grower (which emits only `value > border` splits);
    /// carried for structural fidelity + device parity (Plan 04).
    pub one_hot: Vec<bool>,
    /// Leaf values in bin order (dimension-major for the multi-output case,
    /// `leaf_values[d * n_leaves + l]`), length `(depth + 1) * dim`. Indexed
    /// DIRECTLY by the walk's `bin` (0..=depth).
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights, same bin order as `leaf_values`,
    /// length `depth + 1`.
    pub leaf_weights: Vec<f64>,
}

/// A trained plain-boosted model: the boosting-order trees plus the starting
/// approx (`boost_from_average`) stored as the model bias.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// The oblivious trees in boosting (iteration) order.
    pub oblivious_trees: Vec<ObliviousTree>,
    /// The non-symmetric (Lossguide / Depthwise) trees in boosting order (FEAT-06 /
    /// D-6.6-04). EMPTY for every oblivious model (a model is EITHER all-oblivious or
    /// all-non-symmetric — upstream never mixes grow policies), so the oblivious
    /// lift / apply paths stay byte-identical (D-6.6-05).
    pub non_symmetric_trees: Vec<NonSymmetricTree>,
    /// The REGION trees in boosting order (GPUT-18, D-03a). EMPTY for every
    /// oblivious / non-symmetric model (a model is EITHER all-oblivious OR
    /// all-non-symmetric OR all-region — never mixed), so those paths stay
    /// byte-identical. Populated only under `grow_policy=Region`.
    pub region_trees: Vec<RegionTree>,
    /// The starting approx / model bias.
    pub bias: f64,
    /// The number of output (approx) dimensions (D-6.2-01 / Plan 06.2-02). `1`
    /// for every scalar regression / binary model; `> 1` for multiclass /
    /// multilabel / MultiQuantile. Each tree's `leaf_values` is the
    /// DIMENSION-MAJOR flat buffer `leaf_values[d * n_leaves + l]` of length
    /// `approx_dimension * n_leaves`; at `1` it is exactly `n_leaves` values in
    /// leaf order, byte-identical to the pre-6.2 scalar model.
    pub approx_dimension: usize,
    /// The `ClassToLabel` map for a multiclass model (LOSS-02, Pitfall 4): the
    /// SORTED distinct original class labels, so `class_to_label[c]` is the original
    /// label for class index `c`. The training target is the remapped index `[0, k)`;
    /// predictions recover the original labels via this map. EMPTY for every scalar
    /// regression / binary model (byte-identical to the pre-6.2 model).
    pub class_to_label: Vec<f64>,
    /// The fit-wide `bin -> raw hash` table for the ONE-HOT-routed categorical
    /// columns (SPEC-OH-05 / SPEC-OH-09), indexed by ONE-HOT POSITION then bin:
    /// `one_hot_bin_to_hash[p][bin] == cb_data::calc_cat_feature_hash(raw)` for
    /// the raw value that produced `bin`.
    ///
    /// The trainer's `AnySplit::OneHot` carries a first-seen `PerfectHash` BIN,
    /// which is fit-local and meaningless to upstream; the model lift
    /// (`cb_model::Model::from_trained`) re-expresses it in upstream's RAW hash
    /// space through this table. EMPTY for every float-only / CTR-only fit.
    ///
    /// # Validity
    /// Valid ONLY for the exact learn-set columns it was built from — bins are
    /// first-seen per column, so a different row order yields a different
    /// (equally valid) table.
    pub one_hot_bin_to_hash: Vec<Vec<u32>>,
    /// One-hot POSITION -> ABSOLUTE `cat_columns` index (SPEC-OH-05). Parallel to
    /// [`Model::one_hot_bin_to_hash`]; the model lift needs it to record each
    /// split's absolute cat-feature index (upstream's `TOneHotFeature.Index`)
    /// rather than the dense one-hot position. EMPTY for every float-only /
    /// CTR-only fit.
    pub one_hot_absolute: Vec<usize>,
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
const fn autolr_target_type(loss: &Loss) -> TargetType {
    match *loss {
        Loss::Rmse => TargetType::Rmse,
        // CrossEntropy shares Logloss's auto-LR coefficient row (same objective
        // family); Focal is not in the upstream auto-LR table -> Unknown.
        Loss::Logloss | Loss::CrossEntropy => TargetType::Logloss,
        // The Wave-1 smooth regression losses are not in the upstream auto-LR
        // table (`options_helper.cpp:181-194`) -> Unknown (no rate guessed),
        // mirroring the existing MAE arm.
        // The Wave-1 smooth regression losses and the Wave-2 positive-domain /
        // link losses (Poisson / Tweedie / MAPE) are not in the upstream auto-LR
        // table (`options_helper.cpp:181-194`) -> Unknown (no rate guessed),
        // mirroring the existing MAE arm.
        // The multiclass losses (MultiClass / MultiClassOneVsAll) are not in the
        // upstream auto-LR coefficient table -> Unknown (no rate guessed). Fixtures
        // pin an explicit learning_rate, so auto-LR never fires for them.
        Loss::Focal { .. }
        | Loss::Mae
        | Loss::Quantile { .. }
        | Loss::LogCosh
        | Loss::Lq { .. }
        | Loss::Huber { .. }
        | Loss::Expectile { .. }
        | Loss::Poisson
        | Loss::Tweedie { .. }
        | Loss::Mape
        | Loss::MultiClass
        | Loss::MultiClassOneVsAll
        | Loss::MultiLogloss
        | Loss::MultiCrossEntropy
        // MultiQuantile (Wave 3) is not in the upstream auto-LR coefficient table
        // -> Unknown (no rate guessed); the fixture pins an explicit learning_rate.
        | Loss::MultiQuantile { .. }
        // RMSEWithUncertainty (Wave B, LOSS-08) is not in the upstream auto-LR
        // coefficient table -> Unknown (no rate guessed); the fixture pins an
        // explicit learning_rate.
        | Loss::RmseWithUncertainty
        // The Wave-A ranking losses (QueryRMSE / QuerySoftMax) are not in the
        // upstream auto-LR coefficient table -> Unknown (no rate guessed); the
        // ranking fixtures pin an explicit learning_rate.
        | Loss::QueryRmse
        | Loss::QuerySoftMax { .. }
        // The Wave-B ranking losses (PairLogit / PairLogitPairwise / LambdaMart)
        // are likewise not in the upstream auto-LR coefficient table -> Unknown;
        // the ranking fixtures pin an explicit learning_rate.
        | Loss::PairLogit
        | Loss::PairLogitPairwise
        | Loss::LambdaMart { .. }
        // The Wave-C randomized ranking losses (YetiRank / YetiRankPairwise /
        // StochasticRank) are likewise absent from the upstream auto-LR table ->
        // Unknown; the ranking fixtures pin an explicit learning_rate.
        | Loss::YetiRank { .. }
        | Loss::YetiRankPairwise { .. }
        | Loss::StochasticRank { .. }
        // Custom (LOSS-07): a user objective is not in the upstream auto-LR
        // coefficient table -> Unknown (no rate guessed). The custom path defaults
        // to an explicit learning_rate; auto-LR never fires for it.
        | Loss::Custom(_) => TargetType::Unknown,
    }
}

/// Whether `loss` is a GROUPED (ranking/querywise) loss whose der is computed
/// PER QUERY-GROUP through the grouped seam
/// (`cb_compute::Runtime::compute_gradients_grouped` →
/// `calc_ders_for_queries`), rather than the pointwise per-object
/// `compute_gradients` (LOSS-04, D-6.3-03). Wave A wired the two querywise
/// deterministic losses; Wave B (this plan) adds the pairwise/listwise
/// deterministic losses (PairLogit / PairLogitPairwise / LambdaMart); Plans 04–05
/// extend it for YetiRank / StochasticRank. Every NON-ranking loss returns `false`
/// and keeps the pointwise der site BYTE-IDENTICAL (D-04 no-regression).
#[must_use]
fn is_grouped_loss(loss: &Loss) -> bool {
    matches!(
        loss,
        Loss::QueryRmse
            | Loss::QuerySoftMax { .. }
            | Loss::PairLogit
            | Loss::PairLogitPairwise
            | Loss::LambdaMart { .. }
            | Loss::YetiRank { .. }
            | Loss::YetiRankPairwise { .. }
            | Loss::StochasticRank { .. }
    )
}

/// Whether `loss` drives split-scoring / leaf-weight accounting off the per-object
/// PAIRWISE weights (`bt.PairwiseWeights`) rather than the per-object sample
/// weights. Mirrors upstream `UsesPairsForCalculation`
/// (`enum_helpers.cpp:502` = `IsYetiRankLossFunction(loss) || IsPairLogit(loss)`):
/// for these losses the histogram / leaf `sumWeight` is the per-object sum of
/// competitor weights (`CalcPairwiseWeights`, `approx_updater_helpers.h:74-89`),
/// NOT the per-object weight (which is `1.0` here). Every other loss returns
/// `false` and keeps the per-object weight path byte-identical (D-04).
#[must_use]
fn uses_pairwise_weights(loss: &Loss) -> bool {
    matches!(
        loss,
        Loss::PairLogit
            | Loss::PairLogitPairwise
            | Loss::YetiRank { .. }
            | Loss::YetiRankPairwise { .. }
    )
}

/// Per-object PAIRWISE weight vector mirroring upstream `CalcPairwiseWeights`
/// (`approx_updater_helpers.h:74-89`): for every group's winner→loser competitor
/// edge, add `competitor.weight` to BOTH the winner's and the loser's object slot.
/// The result `pw[obj] = Σ competitor.weight` over all pairs incident on `obj`
/// (as winner OR loser) is the histogram / leaf `sumWeight` the pairwise-loss
/// (`UsesPairsForCalculation`) split scoring + Gradient leaf consume in place of
/// the per-object sample weight (upstream `bt.PairwiseWeights`,
/// `scoring.cpp:275-279` + `approx_calcer.cpp:444`). Accumulation order is
/// group-ascending, winner-doc-ascending, competitor-order — the same fixed `+=`
/// order upstream uses; no `unwrap`/`expect`/`panic`/indexing-slicing.
#[must_use]
fn calc_pairwise_weights(groups: &[GroupSpan], n: usize) -> Vec<f64> {
    let mut pw = vec![0.0_f64; n];
    for group in groups {
        let begin = group.begin;
        for (winner_local, comps) in group.competitors.iter().enumerate() {
            let winner_global = begin + winner_local;
            for competitor in comps {
                let loser_global = begin + competitor.id;
                let w = competitor.weight;
                if let Some(slot) = pw.get_mut(winner_global) {
                    *slot += w;
                }
                if let Some(slot) = pw.get_mut(loser_global) {
                    *slot += w;
                }
            }
        }
    }
    pw
}

/// The default `PairwiseNonDiagReg` (`bayesian_matrix_reg`) prior the Cholesky
/// pairwise-leaf solve uses for the off-diagonal / diagonal reg terms
/// (`oblivious_tree_options.cpp:16` `PairwiseNonDiagReg("bayesian_matrix_reg", 0.1)`).
/// The corpus pins no override, so the upstream default `0.1` applies for the
/// `*Pairwise` leaf path (`pairwise_leaves::compute_pairwise_leaf_deltas`).
const PAIRWISE_NON_DIAG_REG_DEFAULT: f64 = 0.1;

/// Compute the starting approx (and model bias): the target mean for RMSE with
/// `boost_from_average`, else `0` (Pitfall 2). The mean is folded through the
/// sanctioned `sum_f64` primitive (D-05).
/// The number of approx (output) dimensions a loss produces — the
/// `approxDimension` of upstream `TLearnContext` (`approx_dimension.cpp`).
///
/// Every loss in scope this wave (all the scalar regression / binary losses) is
/// single-output, so this is `1`. The multi-output losses (MultiClass /
/// MultiClassOneVsAll / MultiLogloss / MultiCrossEntropy / MultiQuantile) added
/// in Plans 06.2-03..05 override it (e.g. `class_count` or `alpha.len()`). The
/// boosting loop, leaf-delta solver, approx update, and staged record are all
/// dimension-major over this value; at `1` they are byte-identical to the
/// pre-6.2 scalar path (D-04).
fn loss_approx_dimension(loss: &Loss, target: &[f64]) -> usize {
    match loss {
        // MultiClass / MultiClassOneVsAll: the distinct class count
        // `max(distinct, 2)` (`approx_dimension.cpp:24-27`,
        // `label_converter.cpp:142`). The class labels are remapped to a
        // contiguous `[0, k)` index by [`build_class_remap`]; the approx dimension
        // is that map's width.
        Loss::MultiClass | Loss::MultiClassOneVsAll => {
            let map = build_class_remap(target);
            map.len().max(2)
        }
        // MultiQuantile (Wave 3, D-6.2-05): `approx_dimension` = the number of
        // quantiles, `alpha.len()` (`approx_dimension.cpp:17-19`
        // `GetAlphaMultiQuantile(params).size()`). Each dimension is an independent
        // quantile at its own `alpha[d]`.
        Loss::MultiQuantile { alpha, .. } => alpha.len(),
        // RMSEWithUncertainty (Wave B, LOSS-08 / D-6.4-04): 2 output dimensions —
        // dim 0 the regression MEAN, dim 1 the LOG-SCALE (`approx_dimension.cpp:16`).
        Loss::RmseWithUncertainty => 2,
        // Every scalar regression / binary loss is single-output.
        _ => 1,
    }
}

/// Build the `ClassToLabel` map for a multiclass target: the SORTED distinct raw
/// labels, so the contiguous class index `[0, k)` is `index_of(label)` in this
/// vector (upstream `TLabelConverter::Initialize`, `label_converter.cpp:136-145`).
///
/// Returns the labels in ascending order; `class_to_label[c]` is the original label
/// for class index `c`, and the inverse (label → index) is a binary search. The
/// model stores this vector (`class_params`/`multiclass_params`) so predictions
/// recover the original labels (Pitfall 4). Labels are compared with `partial_cmp`
/// and an exact-difference dedup, which require a TOTAL order — the train entry
/// point rejects a non-finite (NaN/Inf) class label up front (WR-06) so this fn is
/// only ever called on finite labels.
fn build_class_remap(target: &[f64]) -> Vec<f64> {
    let mut labels: Vec<f64> = target.to_vec();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup_by(|a, b| (*a - *b).abs() == 0.0);
    labels
}

/// Remap a raw multiclass target to its contiguous `[0, k)` class index using the
/// `class_to_label` map from [`build_class_remap`]. `remapped[i]` is the index `c`
/// such that `class_to_label[c] == target[i]` (Pitfall 4 — the der writes
/// `der[target_class]`, which assumes a contiguous index).
///
/// # Errors
/// Returns [`CbError::OutOfRange`] (T-6.2-01) if a target label is not present in
/// the map — never panics / never indexes out of bounds. The map is built FROM the
/// same target, so every label is present in the normal path; this guards a caller
/// that passes a mismatched (label, map) pair.
fn remap_target_to_class(target: &[f64], class_to_label: &[f64]) -> CbResult<Vec<f64>> {
    target
        .iter()
        .map(|&t| {
            class_to_label
                .iter()
                .position(|&l| (l - t).abs() == 0.0)
                .map(|c| c as f64)
                .ok_or_else(|| {
                    CbError::OutOfRange(format!(
                        "multiclass target label {t} is not in the class map"
                    ))
                })
        })
        .collect()
}

fn starting_approx(params: &BoostParams, target: &[f64]) -> f64 {
    if params.boost_from_average && matches!(params.loss, Loss::Rmse) && !target.is_empty() {
        sum_f64(target) / target.len() as f64
    } else {
        0.0
    }
}

/// The per-dimension RMSEWithUncertainty optimal-constant starting approx
/// `[mean(target), 0.5·log(var(target))]` (LOSS-08, D-6.4-04).
///
/// RMSEWithUncertainty ALWAYS starts from this optimal constant, even with
/// `boost_from_average=false` (`train_model.cpp:858` — the explicit
/// non-BoostFromAverage branch calls `CalcOptimumConstApprox`;
/// `optimal_const_for_loss.h:225-229` returns `{mean, 0.5*log(var)}`). The mean and
/// the (population, divisor `n`) variance are computed in `f32` upstream
/// (`CalculateWeightedTargetAverage` / `CalculateWeightedTargetVariance` return
/// `float`), then `0.5*log(var)` widens to `f64` — replicated here so the starting
/// approx is parity-faithful (the f32 round + the ≤1e-5 tolerance both hold). The
/// `Σtarget` and `Σ(target-mean)²` accumulations route through the sanctioned
/// `cb_core::sum_f64` (D-08). An empty target yields `[0, 0]`.
///
/// Returns the length-2 `[mean, log-scale]` starting approx (dim-major dimension
/// order). Every other loss is single-dimension and uses [`starting_approx`].
fn rmse_uncertainty_starting_approx(target: &[f64]) -> [f64; 2] {
    if target.is_empty() {
        return [0.0, 0.0];
    }
    // mean in f32 (upstream returns `float`): the f64 Σ folded through sum_f64, the
    // /n in f64, then truncated to f32.
    let n = target.len() as f64;
    let mean_f64 = sum_f64(target) / n;
    let mean_f32 = mean_f64 as f32;
    // var = Σ(target - mean)² / n (population, divisor n), accumulated in f64 over
    // the f32-mean-centred residuals (upstream centres on the f32 `mean`), then
    // truncated to f32.
    let mean = f64::from(mean_f32);
    let sq: Vec<f64> = target.iter().map(|&t| (t - mean) * (t - mean)).collect();
    let var_f64 = sum_f64(&sq) / n;
    let var_f32 = var_f64 as f32;
    // 0.5 * log(var) in f64 (the only dimension upstream widens before the log).
    let log_scale = 0.5 * f64::from(var_f32).ln();
    [mean, log_scale]
}

/// Reject `(loss, leaf_method)` combinations with no defined leaf optimizer
/// before any training work (WR-01 / WR-02), rather than silently producing a
/// plausible-but-wrong leaf value.
///
/// - `Exact` has a defined 1-D optimum ONLY for the losses dispatched in
///   [`compute_leaf_deltas`]'s `Exact` arm: [`Loss::LogCosh`] (monotone-bisection
///   `Σ tanh(δ - r) = 0` root) and [`Loss::Mae`] / [`Loss::Quantile`] (weighted
///   sample quantile). Every other loss falls through to the quantile-median
///   fallback, which is NOT that loss's optimum, so reject it up front (upstream
///   `catboost_options.cpp:346` likewise rejects Exact for most losses).
/// - [`Loss::Lq`] with `q < 2` produces a `-q*(q-1)*|r|^(q-2)` hessian that
///   diverges to `±inf` as the residual approaches zero; Newton's denominator
///   then sees `inf`/`NaN`. `Loss::validate` permits any `q >= 1`, so gate the
///   Newton + `q < 2` combination here (the only Newton-clean regime is
///   `q >= 2`).
///
/// # Errors
/// Returns [`CbError::OutOfRange`] for an unsupported `(loss, method)` pair.
/// Reject split-score functions that have no faithful CPU training implementation
/// (CR-01). `NewtonL2` / `NewtonCosine` are second-order calcers
/// (`IsSecondOrderScoreFunction`, `enum_helpers.cpp:830-847`): they reuse the
/// `L2` / `Cosine` score FORMULA verbatim and depend on the histogram FILL placing
/// the summed positive der2 hessian in the `sum_weight` leaf-stat slot. The CPU
/// scoring path produces only the first-order (weight-count) reduction, so a Newton
/// score function would silently degrade to its first-order counterpart. These are
/// GPU-only upstream (D-6.4-06); reject them rather than mislead the caller.
///
/// The first-order GPU-only variants (`SolarL2` / `LOOL2` / `SatL2`) compute their
/// per-leaf term purely from the gradient sum and the weight count, so they remain
/// correct (self-oracled, D-6.4-06) on the CPU path and are NOT rejected here.
fn validate_score_function(score_function: cb_compute::EScoreFunction) -> CbResult<()> {
    use cb_compute::EScoreFunction;
    if matches!(
        score_function,
        EScoreFunction::NewtonL2 | EScoreFunction::NewtonCosine
    ) {
        return Err(CbError::OutOfRange(format!(
            "{score_function:?} is a second-order (Newton) split-score function that \
             requires a der2-hessian histogram fill; it is GPU-only upstream and has \
             no faithful CPU training implementation (the CPU scoring path produces \
             only the first-order weight-count reduction, which would silently \
             degrade NewtonL2 to L2 and NewtonCosine to Cosine). Use a first-order \
             score function (Cosine, L2, SolarL2, LOOL2, or SatL2)."
        )));
    }
    Ok(())
}

/// Reject the CTR types that have no CPU training implementation (SPEC-CTRT-03).
///
/// Upstream gates this at option-parse time
/// (`catboost_options.cpp:504-509`):
/// ```text
/// CB_ENSURE(IsSupportedCtrType(CPU, ctrType),
///           "Ctr type " << ctrType << " is not implemented on CPU yet")
/// ```
/// `IsSupportedCtrType(ETaskType::CPU, …)` (`restrictions.h:18-48`) admits exactly
/// `{Borders, Buckets, BinarizedTargetMeanValue, Counter}`, so
/// [`FloatTargetMeanValue`](crate::ctr::ECtrType::FloatTargetMeanValue) and
/// [`FeatureFreq`](crate::ctr::ECtrType::FeatureFreq) are GPU-only.
///
/// Checked BEFORE any CTR accumulation or tree growth so an unsupported request
/// is a typed error rather than a model silently trained with a different CTR
/// type than the caller asked for.
fn validate_ctr_types(params: &BoostParams) -> CbResult<()> {
    for (field, ty) in [
        ("simple_ctr", params.simple_ctr),
        ("combinations_ctr", params.combinations_ctr),
    ] {
        if !ty.is_cpu_supported() {
            return Err(CbError::Unsupported(format!(
                "Ctr type {ty:?} ({field}) is not implemented on CPU yet \
                 (upstream catboost_options.cpp:504-509; \
                 IsSupportedCtrType(CPU, …) admits only Borders, Buckets, \
                 BinarizedTargetMeanValue and Counter)"
            )));
        }
    }
    Ok(())
}

fn validate_leaf_method(loss: &Loss, method: LeafMethod) -> CbResult<()> {
    if matches!(method, LeafMethod::Exact)
        && !matches!(
            loss,
            Loss::LogCosh | Loss::Mae | Loss::Quantile { .. } | Loss::MultiQuantile { .. }
        )
    {
        return Err(CbError::OutOfRange(format!(
            "LeafMethod::Exact has no defined optimizer for {loss:?}; \
             Exact is supported only for LogCosh, Mae, Quantile, and MultiQuantile"
        )));
    }
    // MultiQuantile (Wave 3, D-6.2-05 / Pitfall 3) is gated to Exact: the upstream
    // single-host-CPU default leaf method is the `useExact` override
    // (`catboost_options.cpp:289-301`). Each dimension reuses the weighted-alpha[d]-
    // quantile Exact leaf; der2 = 0 per dimension, so Newton/Gradient/Simple have no
    // defined optimizer here. Reject any non-Exact method up front rather than
    // silently producing a wrong leaf value.
    if matches!(loss, Loss::MultiQuantile { .. }) && !matches!(method, LeafMethod::Exact) {
        return Err(CbError::OutOfRange(format!(
            "MultiQuantile requires LeafMethod::Exact (the upstream single-host CPU \
             default, weighted alpha-quantile per dimension); {method:?} has no \
             defined MultiQuantile leaf optimizer (der2 = 0)"
        )));
    }
    // MultiClass / MultiClassOneVsAll are gated to Newton (WR-01 / Pitfall 2 —
    // the upstream default leaf method for both is Newton with 1 iteration;
    // Gradient/Simple/Exact have no defined multiclass leaf optimizer here).
    // MultiClass additionally rides the dense symmetric Hessian solve; OneVsAll
    // reuses the per-dimension scalar Newton step. Reject any non-Newton method up
    // front rather than silently producing a wrong leaf value.
    if matches!(loss, Loss::MultiClass | Loss::MultiClassOneVsAll)
        && !matches!(method, LeafMethod::Newton)
    {
        return Err(CbError::OutOfRange(format!(
            "{loss:?} requires LeafMethod::Newton (the upstream default, 1 \
             iteration); {method:?} has no defined multiclass leaf optimizer"
        )));
    }
    // MultiLogloss / MultiCrossEntropy are gated to Newton (Pitfall 2 — the
    // upstream default leaf method for both is Newton; the fixtures pin
    // `leaf_estimation_iterations:1`). They are SEPARABLE (per-dimension diagonal),
    // so they reuse the scalar Newton leaf step per dimension; Gradient/Simple/Exact
    // have no defined multilabel leaf optimizer here. Reject any non-Newton method
    // up front rather than silently producing a wrong leaf value.
    if matches!(loss, Loss::MultiLogloss | Loss::MultiCrossEntropy)
        && !matches!(method, LeafMethod::Newton)
    {
        return Err(CbError::OutOfRange(format!(
            "{loss:?} requires LeafMethod::Newton (the upstream default); \
             {method:?} has no defined multilabel leaf optimizer"
        )));
    }
    // RMSEWithUncertainty (Wave B, LOSS-08) is gated to Newton (the upstream default,
    // 1 iteration — `catboost_options.cpp:77-82`). The diagonal hessian gives a
    // per-dimension scalar Newton step (der2[0]=-w, der2[1]=-2*w*diff²*prec); the
    // Exact/Gradient/Simple leaves have no defined RMSEWithUncertainty optimizer
    // (the log-scale dim is not a quantile/median target). Reject any non-Newton
    // method up front.
    if matches!(loss, Loss::RmseWithUncertainty) && !matches!(method, LeafMethod::Newton) {
        return Err(CbError::OutOfRange(format!(
            "RMSEWithUncertainty requires LeafMethod::Newton (the upstream default, \
             1 iteration); {method:?} has no defined RMSEWithUncertainty leaf optimizer"
        )));
    }
    if matches!(method, LeafMethod::Newton) {
        if let Loss::Lq { q } = *loss {
            if q < 2.0 {
                return Err(CbError::OutOfRange(format!(
                    "Lq{{q={q}}} with LeafMethod::Newton is undefined: the \
                     hessian -q*(q-1)*|r|^(q-2) diverges for q < 2 near a zero \
                     residual; use q >= 2 or a non-Newton leaf method"
                )));
            }
        }
    }
    Ok(())
}

/// Reject unsupported `monotone_constraints` configurations up front — the FEAT-03
/// escalated-gap guard (D-6.6-07), mirroring upstream's
/// `CB_ENSURE_INTERNAL(monotoneConstraints.empty(), "...unsupported for
/// non-symmetric trees yet")` (`monotonic_constraint_utils.h:42`).
///
/// Monotone constraints are OBLIVIOUS-ONLY: upstream throws under EVERY
/// non-symmetric grow policy because there is no defined leaf-monotonization for a
/// non-symmetric structure. cb-train only HAS the oblivious (SymmetricTree)
/// grower today, so any non-empty `monotone_constraints` necessarily routes
/// through the supported oblivious path — there is no way to construct an
/// unsupported combination yet. The guard therefore validates only what is
/// REACHABLE now: each entry must be a valid direction `{-1, 0, +1}` (a malformed
/// constraint vector errors rather than silently mis-encoding the PAVA order).
///
/// # Deferred (owned by Plan 06.6-04, D-6.6-07)
///
/// The `monotone_constraints` × `grow_policy ∈ {Lossguide, Depthwise}` and the
/// `grow_policy == Region` typed-error guards CANNOT be written here because the
/// `grow_policy` enum/field does not exist until Plan 06.6-04 (the plan's
/// do-NOT-invent-a-partial-enum directive). Plan 06.6-04 OWNS adding
/// `grow_policy` and extending THIS guard to reject those combinations the moment
/// the field lands (its acceptance criteria + `monotone_oracle_test` assertions).
/// Until then, the oblivious-only routing makes the unsupported combinations
/// unconstructable, so no fabricated output is possible.
fn validate_monotone_constraints(monotone_constraints: &[i8]) -> CbResult<()> {
    for (f, &c) in monotone_constraints.iter().enumerate() {
        if c != -1 && c != 0 && c != 1 {
            return Err(CbError::OutOfRange(format!(
                "monotone_constraints[{f}] = {c} is invalid: each entry must be \
                 -1 (non-increasing), 0 (free), or +1 (non-decreasing)"
            )));
        }
    }
    Ok(())
}

/// Reject unsupported `grow_policy` combinations up front (FEAT-06 / D-6.6-04,
/// D-6.6-07 — the escalated gaps deferred by Plan 06.6-02, now reachable since
/// `grow_policy` exists):
///
/// `monotone_constraints` × a NON-SYMMETRIC `grow_policy` ({Lossguide, Depthwise,
/// Region}) — upstream EXPLICITLY rejects monotone constraints under every
/// non-symmetric grow policy (`monotonic_constraint_utils.h:42`,
/// `CB_ENSURE_INTERNAL(monotoneConstraints.empty(), "...unsupported for
/// non-symmetric trees yet")`). The monotone PAVA post-pass is wired ONLY into
/// the oblivious leaf path (D-6.6-06); routing a non-empty `monotone_constraints`
/// through the leaf-wise grower would silently DROP the constraint, so it is
/// rejected with a typed error (D-6.6-07 — no fabricated output).
///
/// This guard was DEFERRED by Plan 06.6-02 (the `grow_policy` enum did not yet
/// exist); this is the enablement point Plan 06.6-04 owns.
///
/// # A second guard used to live here and is GONE
///
/// This function also used to reject [`EGrowPolicy::Region`] outright as
/// "UNIMPLEMENTED on the CPU path" (D-6.6-04 "Region OUT"). GPUT-18 / D-03a lifted
/// that — Region grows on the CPU as a `TRegionModel`-style path (`region_grower`)
/// and the trained model carries `region_trees`. A BARE Region fit therefore trains;
/// only Region × non-empty `monotone_constraints` is still refused, by the clause
/// above. The doc is called out rather than silently deleted because the stale
/// version outlived the code by long enough to keep a test asserting the old
/// contract (`monotone_oracle_test`), which then failed.
fn validate_grow_policy(grow_policy: EGrowPolicy, monotone_constraints: &[i8]) -> CbResult<()> {
    // GPUT-18 / D-03a: the "Region OUT" rejection is LIFTED — Region now grows on the
    // CPU as a `TRegionModel`-style path (`region_grower`). The monotone guard below
    // STILL rejects Region + monotone_constraints (upstream rejects monotone
    // constraints for every non-symmetric grow policy, region included).
    let non_symmetric_policy =
        grow_policy.is_non_symmetric() || grow_policy == EGrowPolicy::Region;
    if non_symmetric_policy && !monotone_constraints.is_empty() {
        return Err(CbError::OutOfRange(format!(
            "monotone_constraints are unsupported for non-symmetric / region trees \
             (grow_policy={grow_policy:?}); upstream rejects them \
             (monotonic_constraint_utils.h:42). Use grow_policy=SymmetricTree for \
             monotone constraints (D-6.6-07)."
        )));
    }
    Ok(())
}

/// Map the per-feature `monotone_constraints` onto this oblivious tree's SPLITS,
/// in split order — `GetTreeMonotoneConstraints`
/// (`monotonic_constraint_utils.cpp:120-134`). Split `i` (a `value > border` test
/// on float feature `splits[i].feature`) gets that feature's constraint, or `0`
/// when the feature is unconstrained / out of range. The returned vector is in the
/// SAME split order the forward-bit leaf index uses, so split `i` owns leaf bit
/// `1 << i` (matching upstream `currDepthBitMask`).
fn tree_monotone_constraints(splits: &[Split], monotone_constraints: &[i8]) -> Vec<i8> {
    splits
        .iter()
        .map(|s| monotone_constraints.get(s.feature).copied().unwrap_or(0))
        .collect()
}

/// Per-leaf isotonic weights for the monotone PAVA pass —
/// `CalcMonotonicLeafDeltasSimple` (`approx_calcer.cpp:560-573`): the Gradient leaf
/// weight is `SumWeights + scaledL2`, the Newton leaf weight is `-SumDer2 +
/// scaledL2`. These are reduced over the SAME `leaf_value_leaf_of` partition the
/// leaf-delta solver used, through `cb_core::sum_f64` (D-08), so the isotonic
/// weighted means are exact. Simple reuses the Gradient weight (it shares the
/// Gradient leaf delta). Exact has no Newton/Gradient leaf weight; it falls back to
/// the Gradient `SumWeights + scaledL2` form (the document-weight isotonic weight),
/// which is the only defined per-leaf weight for a quantile leaf.
fn monotonic_leaf_isotonic_weights(
    method: LeafMethod,
    leaf_of: &[usize],
    weighted_der1: &[f64],
    der2: &[f64],
    weights: &[f64],
    scaled_l2: f64,
    n_leaves: usize,
) -> Vec<f64> {
    match method {
        LeafMethod::Newton => {
            let weighted_der2: Vec<f64> = der2
                .iter()
                .zip(weights.iter())
                .map(|(&d, &w)| d * w)
                .collect();
            let sum_der2 = reduce_leaf_der2(leaf_of, &weighted_der2, n_leaves);
            sum_der2.iter().map(|&d2| -d2 + scaled_l2).collect()
        }
        // Gradient / Simple / Exact: SumWeights + scaledL2.
        LeafMethod::Gradient | LeafMethod::Simple | LeafMethod::Exact => {
            let stats = reduce_leaf_stats(leaf_of, weighted_der1, weights, n_leaves);
            stats.iter().map(|s| s.sum_weight + scaled_l2).collect()
        }
    }
}

/// Compute the per-leaf deltas for the selected [`LeafMethod`] (TRAIN-03 / D-09).
///
/// Gradient/Newton/Simple are closed-form over each leaf's ordered reduced sums
/// (`cb_core::sum_f64` via `reduce_leaf_stats` / `reduce_leaf_der2`, D-05). Exact
/// is the loss's 1-D exact optimum over each leaf's per-member residuals
/// (`target - approx`): the weighted sample quantile for MAE / Quantile, the
/// monotone-bisection `Σ tanh(δ - r) = 0` root for LogCosh
/// (`CalcOneDimensionalOptimumConstApprox` dispatch). `weighted_der1[i]` is
/// `der1*weight`; `der2[i]` the per-object second derivative (weighted below for
/// the Newton sum); `approx`/`target` the running approximant/labels; `loss`
/// selects the Exact optimizer.
#[allow(clippy::too_many_arguments)]
fn compute_leaf_deltas(
    method: LeafMethod,
    loss: &Loss,
    leaf_of: &[usize],
    weighted_der1: &[f64],
    der2: &[f64],
    weights: &[f64],
    approx: &[f64],
    target: &[f64],
    scaled_l2: f64,
    n_leaves: usize,
    // The output dimension index `d` this leaf solve is for (the per-`d` outer loop
    // index). For the scalar losses this is always `0`; for MultiQuantile the Exact
    // arm reads this dimension's quantile level `alpha[dim_index]` (D-6.2-05). Every
    // other loss ignores it.
    dim_index: usize,
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
            // Exact: the loss's 1-D exact optimum over each leaf's per-member
            // residuals r_i = target_i - approx_i. scaled_l2 is unused (Exact has
            // no L2 term — it is the unregularized const-approx optimum). The
            // optimizer is selected by `loss` (CalcOneDimensionalOptimumConstApprox
            // switch, optimal_const_for_loss.h:180-216):
            //   - MAE / Quantile -> weighted sample quantile (alpha=0.5, delta=1e-6)
            //   - LogCosh        -> monotone-bisection Σ tanh(δ - r) = 0 root
            let residuals: Vec<f64> = approx
                .iter()
                .zip(target.iter())
                .map(|(&a, &t)| t - a)
                .collect();
            let members = collect_leaf_residuals(leaf_of, &residuals, weights, n_leaves);
            // Thread the active loss's (alpha, delta) into the Exact leaf
            // (RESEARCH Pattern 3 / D-6.1-05): Quantile carries arbitrary
            // alpha/delta; MAE is the median anchor (alpha=0.5, delta=1e-6 == the
            // prior hardcoded behavior, so MAE Exact stays byte-identical); any
            // other Exact-eligible loss keeps the default median. `exact_leaf_delta`
            // (leaf.rs) is ALREADY alpha-general — UNCHANGED.
            //   - MultiQuantile -> the weighted alpha[dim_index]-quantile of THIS
            //     dimension's leaf residuals (D-6.2-05; K independent quantile dims,
            //     each with its own alpha[d], shared delta). `exact_leaf_delta` is
            //     reused VERBATIM per dimension (leaf.rs UNCHANGED).
            let (quantile_alpha, quantile_delta) = match loss {
                Loss::Quantile { alpha, delta } => (*alpha, *delta),
                // MultiQuantile: thread THIS dimension's alpha (alpha[dim_index]) and
                // the shared delta into the SAME Exact weighted-quantile leaf. A
                // missing index (defensive) falls back to the median anchor.
                Loss::MultiQuantile { alpha, delta } => {
                    (alpha.get(dim_index).copied().unwrap_or(QUANTILE_ALPHA), *delta)
                }
                _ => (QUANTILE_ALPHA, QUANTILE_DELTA),
            };
            members
                .iter()
                .map(|(r, w)| match loss {
                    Loss::LogCosh => logcosh_exact_leaf_delta(r, w),
                    // MAE / Quantile / MultiQuantile (and any other Exact-eligible
                    // loss for this wave) uses the weighted sample quantile at the
                    // threaded (alpha, delta) — for MultiQuantile, alpha[dim_index].
                    _ => exact_leaf_delta(r, w, quantile_alpha, quantile_delta),
                })
                .collect()
        }
    }
}

/// Compute the MultiClass softmax per-leaf SYMMETRIC Newton leaf deltas — the
/// COUPLED cross-dimension leaf solve (`approx_calcer_multi_helpers.cpp` +
/// `hessian.cpp:22-52`). UNLIKE the diagonal losses (which solve each dimension
/// independently in the boosting loop's per-`d` arm over [`compute_leaf_deltas`]),
/// softmax's per-leaf delta is one dense symmetric solve over ALL `k` dimensions,
/// so it is computed here ONCE and returns the dimension-major leaf values.
///
/// # Inputs
/// - `leaf_of[i]`: object `i`'s leaf index (shared across dimensions — the
///   oblivious structure is one tree).
/// - `weighted_der1`: the DIMENSION-MAJOR weighted first derivative
///   `der1[d*n + i] * weight[i]` (length `k*n`).
/// - `der2_packed`: the PER-OBJECT packed symmetric Hessian `der2_packed[i*pk + j]`
///   (length `n * pk`, `pk = k*(k+1)/2`), already weighted per object (the
///   `weight != 1` branch of `softmax_ders`; unit weights in scope).
/// - `weights[i]`: per-object weight (folded into the Hessian below).
/// - `scaled_l2`: the per-tree `scale_l2_reg` output.
/// - `n_leaves`, `k`, `n`.
///
/// # Output
/// The DIMENSION-MAJOR leaf-delta buffer `delta[d * n_leaves + leaf]` (length
/// `k * n_leaves`), BEFORE the `learning_rate` scaling (the caller multiplies).
/// Per leaf: sum the per-member `der1[d]` and packed `der2[j]` via
/// `cb_core::sum_f64` (ordered, D-08), then [`solve_symmetric_newton`].
fn compute_softmax_leaf_deltas(
    leaf_of: &[usize],
    weighted_der1: &[f64],
    der2_packed: &[f64],
    weights: &[f64],
    scaled_l2: f64,
    n_leaves: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    let pk = k * (k + 1) / 2;
    // Per-leaf gather of the per-dimension der1 and the per-element packed der2,
    // each member contribution pushed in ascending object order so the
    // `cb_core::sum_f64` reduction order matches upstream's thread_count==1 pass.
    let mut der1_members: Vec<Vec<Vec<f64>>> =
        vec![vec![Vec::new(); k]; n_leaves];
    let mut der2_members: Vec<Vec<Vec<f64>>> =
        vec![vec![Vec::new(); pk]; n_leaves];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if leaf >= n_leaves {
            continue;
        }
        let w = weights.get(i).copied().unwrap_or(1.0);
        for d in 0..k {
            let v = weighted_der1.get(d * n + i).copied().unwrap_or(0.0);
            if let Some(slot) = der1_members.get_mut(leaf).and_then(|r| r.get_mut(d)) {
                slot.push(v);
            }
        }
        for j in 0..pk {
            // The per-object packed Hessian is unweighted (softmax_ders returns
            // weight==1); fold the per-object weight in here (the
            // `der.Der2 *= weight` upstream branch) so weighted training matches.
            let v = der2_packed.get(i * pk + j).copied().unwrap_or(0.0) * w;
            if let Some(slot) = der2_members.get_mut(leaf).and_then(|r| r.get_mut(j)) {
                slot.push(v);
            }
        }
    }

    // Per-leaf: reduce the gathered members (D-08) and run the symmetric solve.
    let mut leaf_values = vec![0.0_f64; k * n_leaves];
    for leaf in 0..n_leaves {
        let sum_der: Vec<f64> = (0..k)
            .map(|d| {
                let members = der1_members
                    .get(leaf)
                    .and_then(|r| r.get(d))
                    .map_or(&[][..], Vec::as_slice);
                sum_f64(members)
            })
            .collect();
        let sum_der2: Vec<f64> = (0..pk)
            .map(|j| {
                let members = der2_members
                    .get(leaf)
                    .and_then(|r| r.get(j))
                    .map_or(&[][..], Vec::as_slice);
                sum_f64(members)
            })
            .collect();
        let delta = solve_symmetric_newton(&sum_der, &sum_der2, scaled_l2);
        for d in 0..k {
            if let Some(slot) = leaf_values.get_mut(d * n_leaves + leaf) {
                *slot = delta.get(d).copied().unwrap_or(0.0);
            }
        }
    }
    leaf_values
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
/// The walk-until-halt distinct-leaf index of one object over a NON-SYMMETRIC device
/// node graph (GPUT-18 / Phase 12 Plan 03). TRANSCRIBES `cb_model::apply::leaf_index_nonsym`
/// (`apply.rs:234-270`) inline into cb-train (the forbidden direction is a cb-BACKEND dep
/// inside cb-train, NOT transcribing cb-MODEL apply logic): a bounded flat-node walk over
/// `step_nodes` left/right diffs, `u32::MAX` interior guard, halting on the zero side and
/// reading `node_id_to_leaf_id`. The pass test is the SAME `value > border` the oblivious
/// fold uses. Returns `None` on a malformed / cyclic graph (the caller substitutes a checked
/// leaf-0 fallback — never a panic, T-12-05).
fn device_leaf_of_nonsym(
    obj: usize,
    splits: &[Split],
    step_nodes: &[(u16, u16)],
    node_id_to_leaf_id: &[u32],
    feature_values: &[Vec<f32>],
) -> Option<usize> {
    let node_count = step_nodes.len();
    let mut index: i64 = 0;
    // A valid walk visits each node at most once — cap iterations to reject a cyclic graph.
    for _ in 0..=node_count {
        let idx = usize::try_from(index).ok()?;
        let &(left_diff, right_diff) = step_nodes.get(idx)?;
        let split = splits.get(idx)?;
        let passes = feature_values
            .get(split.feature)
            .and_then(|col| col.get(obj))
            .is_some_and(|&v| f64::from(v) > split.border);
        let diff: i64 = if passes { i64::from(right_diff) } else { i64::from(left_diff) };
        index = index.checked_add(diff)?;
        if diff == 0 {
            let leaf_id = *node_id_to_leaf_id.get(idx)?;
            if leaf_id == u32::MAX {
                return None;
            }
            return usize::try_from(leaf_id).ok();
        }
    }
    None
}

fn accumulate_leaf_weights(leaf_of: &[usize], weights: &[f64], n_leaves: usize) -> Vec<f64> {
    // Unit-weight fast path (every unweighted fit): a left-to-right fold of k ones
    // is EXACTLY `k as f64` for any k < 2^53 (integer-valued f64 addition is
    // exact), so a per-leaf COUNT is bit-identical to the serial `sum_f64` fold —
    // and integer counts are order-independent, so the count may run in parallel.
    // Membership rule mirrors the general path exactly: an object contributes only
    // when its leaf index is in range AND it has a weight entry.
    if weights.iter().all(|&w| w == 1.0) {
        let bound = leaf_of.len().min(weights.len());
        let counts = leaf_of
            .get(..bound)
            .unwrap_or(&[])
            .par_iter()
            .fold(
                || vec![0_u64; n_leaves],
                |mut acc, &leaf| {
                    if let Some(slot) = acc.get_mut(leaf) {
                        *slot += 1;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0_u64; n_leaves],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    a
                },
            );
        return counts.into_iter().map(|c| c as f64).collect();
    }
    // General path: bucket each leaf's member weights in object order (checked
    // `.get` only — `indexing_slicing` is deny), then fold each bucket with the
    // sanctioned left-to-right `sum_f64`. The bucketing is chunked in parallel and
    // the per-chunk buckets are concatenated IN CHUNK ORDER, so every leaf's member
    // sequence — and therefore the fold order and its bits — is identical to the
    // serial form (order is the parity contract, cb-core::reduction).
    const CHUNK: usize = 1 << 16;
    let n_chunks = leaf_of.len().div_ceil(CHUNK).max(1);
    let partials: Vec<Vec<Vec<f64>>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * CHUNK;
            let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
            for (off, &leaf) in leaf_of.get(start..).unwrap_or(&[]).iter().take(CHUNK).enumerate() {
                if let (Some(bucket), Some(&w)) = (buckets.get_mut(leaf), weights.get(start + off)) {
                    bucket.push(w);
                }
            }
            buckets
        })
        .collect();
    (0..n_leaves)
        .into_par_iter()
        .map(|l| {
            let mut members: Vec<f64> = Vec::new();
            for chunk_buckets in &partials {
                if let Some(bucket) = chunk_buckets.get(l) {
                    members.extend_from_slice(bucket);
                }
            }
            sum_f64(&members)
        })
        .collect()
}

/// `NormalizeLeafValues` (`approx_updater_helpers.cpp:8-21`, called from
/// `train.cpp:562`): apply the per-tree leaf-value normalization upstream runs
/// AFTER the leaf estimator and BEFORE storing the tree.
///
/// For a pairwise loss (`is_pairwise == UsesPairsForCalculation`) the leaf values
/// are shifted by the DOCUMENT-WEIGHTED mean so the tree adds no constant offset
/// (the pairwise objective is invariant to a global additive constant):
/// ```text
/// avg = Σ leafValue[l] * leafWeight[l] / Σ leafWeight[l]
/// leafValue[l] = (|leafWeight[l]| > 1e-9) ? (leafValue[l] - avg) : 0
/// ```
/// Empty leaves (zero document weight) are forced to exactly `0`, NOT shifted.
/// Then, for EVERY loss, each leaf value is scaled by `learning_rate` (this is the
/// SINGLE place lr is applied — the leaf branches push RAW deltas). For a
/// non-pairwise loss this reduces to the prior `learning_rate * delta` exactly
/// (D-04). The weighted-mean accumulation routes through `cb_core::sum_f64`
/// (D-08 — the single sanctioned strict left-to-right f64 fold).
///
/// `leaf_values` is dimension-major (`[d*n_leaves + l]`); the pairwise centering is
/// per-dimension over each `n_leaves` slice (upstream `treeValues[0]`; pairwise
/// losses are single-dimension so only dimension 0 exists, but the loop is dim-safe).
/// `leaf_weights` is one-per-leaf (shared across dimensions).
fn normalize_leaf_values(
    is_pairwise: bool,
    learning_rate: f64,
    leaf_weights: &[f64],
    leaf_values: &mut [f64],
    n_leaves: usize,
    approx_dimension: usize,
) {
    if is_pairwise {
        let total_weight = sum_f64(leaf_weights);
        if total_weight.abs() > 1e-9 {
            for d in 0..approx_dimension {
                let base = d * n_leaves;
                // Document-weighted sum of this dimension's leaf values.
                let weighted: Vec<f64> = (0..n_leaves)
                    .map(|l| {
                        let v = leaf_values.get(base + l).copied().unwrap_or(0.0);
                        let w = leaf_weights.get(l).copied().unwrap_or(0.0);
                        v * w
                    })
                    .collect();
                let avg = sum_f64(&weighted) / total_weight;
                for l in 0..n_leaves {
                    if let Some(v) = leaf_values.get_mut(base + l) {
                        let w = leaf_weights.get(l).copied().unwrap_or(0.0);
                        if w.abs() > 1e-9 {
                            *v -= avg;
                        } else {
                            *v = 0.0;
                        }
                    }
                }
            }
        }
    }
    for v in leaf_values.iter_mut() {
        *v *= learning_rate;
    }
}

/// Assign each object's LEAF-VALUE leaf index over the AVERAGING-fold CTR columns
/// (ORD-05, research Q1/Q3 #3 — `train.cpp:130` `BuildIndices(AveragingFold)`).
///
/// Walks the grown tree's `level_kinds` in level order (so float and CTR levels
/// interleave in the correct forward-bit order). For a FLOAT level the bit is
/// `value > border` on the float matrix (the SAME test the structure search used,
/// reproduced from the public `feature_values` / the chosen `Split`). For a CTR
/// level the bit is `ctr_bin > border` against the AVERAGING-fold column's `bins`
/// (NOT the structure column) — this is the single place the leaf-VALUE partition
/// diverges from the structure partition (`[6,0,7,17]` vs `[6,0,9,15]` for the
/// tensor_ctr_e2e config).
///
/// `ctr_columns` is whichever fold's materialized CTR column set the caller
/// wants the partition under — the AVERAGING fold's for the leaf-VALUE
/// partition, or a LEARNING fold's for that fold's own approx update
/// (`UpdateLearningFold`, `train.cpp:585`). All fold column sets are emitted by
/// the same `materialize_ctr_columns_for_perm`, so the candidate-identity keys
/// used below are aligned across folds by construction. A `LevelKind::Ctr`'s
/// `ctr_idx` indexes the tree's chosen `ctr_splits`, whose candidate identity
/// selects which column to read. Out-of-range indices contribute a `false` bit
/// defensively (checked `.get` only — no panic, no raw index).
fn assign_leaf_over_ctr_columns(
    matrix: &FeatureMatrix,
    averaging_ctr_features: &[crate::ctr::CtrFeatureColumn],
    grown: &GrownTree,
    n_objects: usize,
) -> Vec<usize> {
    (0..n_objects)
        .map(|obj| {
            let mut passes: Vec<bool> = Vec::with_capacity(grown.level_kinds.len());
            for kind in &grown.level_kinds {
                let bit = match kind {
                    LevelKind::Float(split_idx) => grown
                        .splits
                        .get(*split_idx)
                        .and_then(|s| {
                            matrix
                                .feature_values
                                .get(s.feature)
                                .and_then(|col| col.get(obj))
                                .map(|&v| f64::from(v) > s.border)
                        })
                        .unwrap_or(false),
                    // SPEC-OH-07: a one-hot level is the `cat_bin == value`
                    // equality test on the matrix's one-hot bin column. This
                    // rebuild runs on the CTR leaf-value path, where one-hot and
                    // CTR columns never co-occur (SPEC-OH-26 gates the mix), but
                    // the arm is real rather than a silent `false` so a future
                    // mixed pool cannot mis-assign leaves undetected.
                    LevelKind::OneHot(one_hot_idx) => grown
                        .one_hot_splits
                        .get(*one_hot_idx)
                        .and_then(|oh| {
                            matrix
                                .cat_bins
                                .get(oh.feature)
                                .and_then(|col| col.get(obj))
                                .map(|&bin| bin == oh.value)
                        })
                        .unwrap_or(false),
                    LevelKind::Ctr { ctr_idx, border } => grown
                        .ctr_splits
                        .get(*ctr_idx)
                        // Find the averaging column this chosen CTR split was
                        // scored on. The key is the FULL candidate identity —
                        // `(projection, ctr_type, target_border_idx, prior)` —
                        // NOT the projection alone (E15): the multi-prior
                        // expansion emits one column per prior on the same
                        // projection, so a projection-only `find` would silently
                        // return the HEAD prior's column and partition the leaf
                        // VALUES on bins the structure search never scored. Both
                        // sides of the prior comparison originate from the same
                        // configured list element (the column's prior is copied
                        // verbatim onto the split in `crate::tree`), so bit
                        // equality is exact rather than approximate.
                        .and_then(|spec| {
                            averaging_ctr_features.iter().find(|c| {
                                c.projection == spec.projection
                                    && c.ctr_type == spec.ctr_type
                                    && c.target_border_idx == spec.target_border_idx
                                    && c.prior_num.to_bits() == spec.prior_num.to_bits()
                                    && c.prior_denom.to_bits() == spec.prior_denom.to_bits()
                            })
                        })
                        .and_then(|col| col.bins.get(obj))
                        .is_some_and(|&bin| f64::from(bin) > *border),
                };
                passes.push(bit);
            }
            leaf_index(&passes)
        })
        .collect()
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
/// Resolve the `(CTR type, head prior)` pair that governs ONE candidate
/// (SPEC-CTRT-09 / SPEC-CTRT-10).
///
/// A **simple** candidate (a single categorical feature) is governed by
/// `simple_ctr` / `simple_ctr_priors`; a **combination** candidate by
/// `combinations_ctr` / `combinations_ctr_priors`. Before E10 a single
/// `combinations_ctr_priors.first()` fed BOTH, so the combination prior silently
/// governed simple candidates — the bug SPEC-CTRT-10 fixes.
///
/// Returns only the HEAD prior. The candidate MATERIALIZATION expands over the
/// whole list (E15, [`ctr_config_list_for`]); this single-prior form survives for
/// [`ctr_splits_for_tree`], the no-CTR-candidate fallback where no materialized
/// column exists to carry a per-candidate prior.
fn ctr_config_for(
    simple_ctr: crate::ctr::ECtrType,
    simple_priors: &[f64],
    combinations_ctr: crate::ctr::ECtrType,
    combinations_priors: &[f64],
    is_simple: bool,
) -> (crate::ctr::ECtrType, f64) {
    let (ctr_type, priors) = ctr_config_list_for(
        simple_ctr,
        simple_priors,
        combinations_ctr,
        combinations_priors,
        is_simple,
    );
    (ctr_type, priors.first().copied().unwrap_or(DEFAULT_CTR_PRIOR))
}

/// The prior a candidate falls back to when its configured prior list is EMPTY.
/// Matches the pre-E15 `.first().unwrap_or(0.5)` behavior exactly, so an empty
/// list still emits exactly one column at `0.5`.
const DEFAULT_CTR_PRIOR: f64 = 0.5;

/// The prior list an empty configuration degenerates to — one column at
/// [`DEFAULT_CTR_PRIOR`], never zero columns.
const DEFAULT_CTR_PRIORS: [f64; 1] = [DEFAULT_CTR_PRIOR];

/// Resolve the `(CTR type, FULL prior list)` pair that governs ONE candidate
/// (SPEC-CTRT-10 / SPEC-CTRT-11).
///
/// The list half is what E15 needs: upstream emits one candidate column per
/// `(ctrIdx, targetBorderIdx, priorIdx)` (`greedy_tensor_search.cpp:414-427`), so
/// every configured prior produces its own scored column. An EMPTY configured
/// list degenerates to `[DEFAULT_CTR_PRIOR]` rather than to no columns, keeping
/// the pre-E15 `.first().unwrap_or(0.5)` behavior byte-identical.
fn ctr_config_list_for<'a>(
    simple_ctr: crate::ctr::ECtrType,
    simple_priors: &'a [f64],
    combinations_ctr: crate::ctr::ECtrType,
    combinations_priors: &'a [f64],
    is_simple: bool,
) -> (crate::ctr::ECtrType, &'a [f64]) {
    let (ctr_type, priors) = if is_simple {
        (simple_ctr, simple_priors)
    } else {
        (combinations_ctr, combinations_priors)
    };
    if priors.is_empty() {
        (ctr_type, &DEFAULT_CTR_PRIORS)
    } else {
        (ctr_type, priors)
    }
}

/// The RAW per-object categorical-bucket column for every CTR-eligible cat
/// feature — the `model_size_reg` cat-feature-weight input (`GetCatFeatureWeight`,
/// `greedy_tensor_search.cpp:908-932`), consumed by an order-insensitive `.max()`
/// in [`crate::tree::select_level_ctr_aware`]'s phantom mixed-partition bucket
/// count.
///
/// **One column per CTR-eligible categorical FEATURE, never per emitted CTR
/// candidate column.** It is NOT index-aligned with the materialized CTR column
/// list and MUST NOT grow with the `(projection, prior)` — after E16,
/// `(projection, target_border_idx, prior)` — candidate expansion: growing it
/// would change `phantom_mixed_bucket_count`, hence `model_size_reg`'s
/// cat-feature weight, hence split choice. Taking `eligible_absolute` rather than
/// the column list is precisely what makes that structurally impossible.
///
/// Empty for the numeric path (`cat_columns` empty ⇒ `eligible_absolute` empty),
/// a provable no-op there.
pub(crate) fn cat_eligible_buckets_for(
    cat_columns: &[Vec<String>],
    eligible_absolute: &[usize],
) -> CbResult<Vec<Vec<u32>>> {
    eligible_absolute
        .iter()
        .map(|&abs_idx| match cat_columns.get(abs_idx) {
            Some(col) => {
                let as_str: Vec<&str> = col.iter().map(String::as_str).collect();
                cb_data::perfect_hash_bins(&as_str)
            }
            None => Ok(Vec::new()),
        })
        .collect::<CbResult<Vec<Vec<u32>>>>()
}

/// Materialize the online CTR candidate columns for ONE permutation — the single
/// place the candidate product is built (E15).
///
/// `train_inner` calls this twice: once per STRUCTURE learning fold (each fold's
/// own permutation) and once for the AVERAGING fold. Because both go through this
/// one function, the index alignment the chosen-split → averaging-column lookup
/// depends on holds by construction rather than by convention.
///
/// The emitted order is upstream's `(ctrIdx, targetBorderIdx, priorIdx)`
/// nesting (`greedy_tensor_search.cpp:400-428`): for each candidate, one column
/// per `target_border_idx` in `0..target_border_count(classes)` (E16 /
/// SPEC-CTRT-12 — `2` for Buckets at binclf, `1` for every other CPU-legal
/// type), and inside that one column per prior in configured list order. With a
/// single-element prior list and a non-Buckets type the sequence is
/// byte-identical to the pre-E15 one-column-per-candidate emission (the D-04
/// no-op proof).
pub(crate) fn materialize_ctr_columns_for_perm(
    cat_columns: &[Vec<String>],
    absolute_projections: &[crate::TProjection],
    ctr_candidates: &[crate::candidates::CtrCandidate],
    params: &BoostParams,
    permutation: &[i32],
    target_class: &[usize],
    ctr_border_count: usize,
    extra_cat_columns: &[Vec<String>],
) -> CbResult<Vec<crate::ctr::CtrFeatureColumn>> {
    // The binclf target-class count — the SAME `2` the bake passes to
    // `bake_ctr_table` (`GetTargetBorderCount`'s `targetClassesCount` input).
    const TARGET_CLASSES: usize = 2;
    let mut cols = Vec::with_capacity(ctr_candidates.len());
    for (ci, proj) in absolute_projections.iter().enumerate() {
        // `absolute_projections` is index-aligned with `ctr_candidates`, so
        // `is_simple` is available without a second lookup.
        let is_simple = ctr_candidates.get(ci).is_some_and(|c| c.is_simple);
        let (ctr_type, priors) = ctr_config_list_for(
            params.simple_ctr,
            &params.simple_ctr_priors,
            params.combinations_ctr,
            &params.combinations_ctr_priors,
            is_simple,
        );
        for target_border_idx in 0..ctr_type.target_border_count(TARGET_CLASSES) {
            for &prior_num in priors {
                let col = crate::ctr::materialize_ctr_feature(
                    cat_columns,
                    proj,
                    permutation,
                    target_class,
                    prior_num,
                    // CPU forbids a non-unit prior denominator (ctr_helper.cpp:50).
                    CTR_PRIOR_DENOM,
                    ctr_border_count,
                    ctr_type,
                    target_border_idx,
                    // E22 / SPEC-CTRT-17: the concatenated eval-set cat columns
                    // under `counter_calc_method = Full` (empty otherwise); the
                    // materializer applies them to COUNTER candidates only.
                    extra_cat_columns,
                )?;
                cols.push(col);
            }
        }
    }
    Ok(cols)
}

/// The CTR prior DENOMINATOR. Constant `1` on the CPU path (RESEARCH A6; a
/// non-unit denominator is forbidden by `ctr_helper.cpp:50`), carried as a
/// separate half so the bake receives the denominator for `calc_normalization`
/// rather than a pre-divided scalar.
const CTR_PRIOR_DENOM: f64 = 1.0;

fn ctr_splits_for_tree(
    candidates: &[crate::candidates::CtrCandidate],
    simple_ctr: crate::ctr::ECtrType,
    simple_priors: &[f64],
    combinations_ctr: crate::ctr::ECtrType,
    combinations_priors: &[f64],
) -> Vec<CtrSplitSpec> {
    candidates
        .iter()
        .map(|c| {
            // Per-candidate routing (E10): no hard-coded Borders head, and the
            // simple/combination prior lists are kept distinct.
            let (ctr_type, prior_num) = ctr_config_for(
                simple_ctr,
                simple_priors,
                combinations_ctr,
                combinations_priors,
                c.is_simple,
            );
            CtrSplitSpec {
                projection: c.projection.clone(),
                ctr_type: ctr_type.as_i8(),
                prior_num,
                // CPU forbids a non-unit prior denominator (ctr_helper.cpp:50).
                prior_denom: 1.0,
                // DELIBERATE, TESTED CONSTANT (E16): this function is reached
                // only from the `!has_ctr` fallback, where NO materialized
                // column exists by construction — so there is no per-column
                // `target_border_idx` to read, and structurally cannot be. The
                // E03 characterization test pins the `0`.
                target_border_idx: 0,
                border: 0.0,
                shift: 0.0,
                scale: 1.0,
            }
        })
        .collect()
}

/// GDC-11 (T14) / FPP-09 (T07): whether every materialized CTR column is DEVICE-covered.
/// The device CTR arm implements the ordered binclf `(good + prior) / (total + 1)`
/// statistic — exactly upstream's Borders CTR with `target_border_idx == 0` and a unit
/// prior denominator — over projections of ANY arity.
///
/// # Why combination (tensor) projections are admitted (FPP-09, PLAN V-4)
///
/// The projection arity is NOT a semantic constraint on the online statistic: the
/// statistic is computed over a projection's COMBINED bucket identity, and the two sides
/// derive that identity identically. The CPU folds each member's per-document categorical
/// hash in projection-sorted order (`TProjection::combined_hash`); the device is handed
/// one `member_bins` column per member in that SAME sorted order and folds them with
/// `combine_projection_bins`. The combined bins are integer-identical end to end, so the
/// accumulator that is correct for one member is correct for `k`.
///
/// The residual is a hypothetical 64-bit hash-collision asymmetry — the CPU folds string
/// hashes, the device folds perfect-hash bucket codes, so a collision on ONE side only
/// would diverge. Documented, not guarded; the ≤1e-5 e2e bar is its detector.
///
/// # Combination CTR is device-INELIGIBLE (FPP-11, ESCALATED — do not re-open blind)
///
/// The paragraph above describes why the arity SHOULD be admissible, and the column
/// builder ([`build_device_ctr_config`]) does emit one `member_bins` entry per member,
/// unit-tested by `device_ctr_combo_config_test`. But the end-to-end oracle over the
/// `ctr_device_combo/` fixture does NOT meet the ≤1e-5 bar, so the gate stays closed.
///
/// Measured on gfx1151 against upstream `catboost==1.2.10`:
///
/// - the CPU path over the same fixture is exact: **max|Δpred| = 1.4e-17**, with 8 CTR
///   splits, so neither the fixture nor the CPU combination CTR is at fault;
/// - the DEVICE path misses by **3.3e-2**;
/// - trees 0, 1 and 2 are STRUCTURALLY IDENTICAL to the CPU's — including tree 2's
///   2-member combination split `[0,1] @ border 4.0` — so `combine_projection_bins`
///   is producing usable combined bins;
/// - the divergence begins at **tree 3, level 0**: the CPU picks the simple projection
///   `[0] @ 6.0` and the device picks the combination `[0,1] @ 8.0`. Every later level
///   follows from that one different partition.
///
/// Tree 3 level 0 is the first point at which BOTH the simple and the combination group
/// have already entered the model-lifetime `UsedCtrSplits` set (the combination enters at
/// tree 2), so both candidates score at cat-feature weight `1.0` on both sides and the
/// disagreement is a raw split-gain difference, not a weight difference. The most likely
/// remaining suspects, in order: the device's `eligible_max` (`maxCount`) now maxes over a
/// combination column's `bucket_count`, which its own comment says was written assuming
/// "the device gate admits only simple projections"; and the combination column's
/// `bucket_count` itself (`combine_projection_bins` returns the OBSERVED distinct-bucket
/// count, whereas upstream's `TOnlineCtrUniqValuesCounts::Count` may not).
///
/// Re-opening this clause requires the e2e oracle to pass, not just the config unit test —
/// this is exactly the ordering discipline that made the gap visible.
///
/// Everything else — Buckets / BinarizedTargetMeanValue / Counter (Track U), multi-target-
/// border Buckets columns, non-unit prior denominators — still declines to the
/// byte-unchanged CPU path (D-04): the device kernels do not implement those accumulation
/// semantics, and a wrong device leaf is worse than a CPU fallback.
///
/// An EMPTY column set returns `false` (the caller's `is_empty()` arm owns that).
fn ctr_types_are_device_covered(cols: &[crate::ctr::CtrFeatureColumn]) -> bool {
    !cols.is_empty()
        && cols.iter().all(|col| {
            // ESCALATED (FPP-11): the projection-arity conjunct is RESTORED. See the
            // "combination CTR is device-INELIGIBLE" section of this function's doc
            // comment for the measured evidence and the localisation.
            col.projection.is_simple()
                && col.ctr_type == crate::ctr::ECtrType::Borders.as_i8()
                && col.target_border_idx == 0
                && col.prior_denom == 1.0
        })
}

/// FPP-05 (T06): decide `DeviceTrainConfig::{exact_leaf, quantile_alpha, quantile_delta}`
/// for a fit, as a pure function of its leaf method and loss.
///
/// # The intersection (PLAN V-6, derived — read this before widening the match)
///
/// The device Exact order-statistic leaf activates ONLY for the intersection of
///
/// - **(a) what `validate_leaf_method` permits** on the CPU
///   (`LogCosh | Mae | Quantile | MultiQuantile`), and
/// - **(b) what `map_leaf_method` covers** on the device
///   (`Mae | Quantile | Mape`).
///
/// The two sets disagree in BOTH directions, which is why neither alone is the right
/// condition:
///
/// - `LogCosh` is CPU-legal but device-UNCOVERED — admitting it would silently apply the
///   Gradient `calc_average` leaf to a LogCosh fit, which is wrong and strictly worse than
///   today's correct CPU fallback.
/// - `Mape` is device-covered (with `mape: true`) but CPU-REJECTED by
///   `validate_leaf_method`, so no fit can reach the device with that pair at all.
/// - `MultiQuantile` is CPU-legal but multi-dimensional, which the scalar `exact_leaf`
///   arm cannot express.
///
/// ⇒ the admitted set is exactly `{Mae, Quantile}`.
///
/// Note also that `builder.rs` defaults `leaf_method: Gradient` unconditionally, so this
/// arm is reachable only by an EXPLICIT `LeafMethod::Exact` request.
///
/// # Returns
///
/// `(exact_leaf, quantile_alpha, quantile_delta)`. For a declined pair the α/δ are the
/// [`DeviceTrainConfig`] defaults and are inert. For `Mae` the exact leaf IS the weighted
/// median, so it also carries the defaults; only `Quantile` supplies its own — and the
/// backend's `map_leaf_method` reads `Loss::Quantile`'s own fields there anyway, so the
/// two agree by construction.
#[must_use]
fn device_exact_leaf_config(leaf_method: LeafMethod, loss: &Loss) -> (bool, f64, f64) {
    let admitted =
        matches!(leaf_method, LeafMethod::Exact) && matches!(loss, Loss::Mae | Loss::Quantile { .. });
    if !admitted {
        return (false, QUANTILE_ALPHA, QUANTILE_DELTA);
    }
    match *loss {
        Loss::Quantile { alpha, delta } => (true, alpha, delta),
        // Mae: the weighted MEDIAN — the struct's own defaults, not a re-typed literal.
        _ => (true, QUANTILE_ALPHA, QUANTILE_DELTA),
    }
}

/// GDC-11 (T14): build the two-permutation [`cb_compute::DeviceCtrConfig`] from
/// the CPU-side materialization state. The border table reproduces
/// [`crate::ctr::calc_ctr_online_bin`]'s truncation EXACTLY under the device's
/// strict `value > border` binarize: `bin >= k+1 ⟺ v >= v_k` with
/// `v_k = (k+1)·norm/border_count − shift`, so `borders[k] = v_k.next_down()`
/// (the greatest f64 strictly below `v_k`) makes the strict test equivalent to
/// the truncation for EVERY f64 value, including exact boundary hits.
///
/// # Errors
/// A negative permutation entry (impossible for a well-formed fold order) is a
/// typed [`CbError::OutOfRange`] rather than a silent `as` truncation.
#[allow(clippy::too_many_arguments)]
fn build_device_ctr_config(
    materialized_ctr_features: &[crate::ctr::CtrFeatureColumn],
    averaging_ctr_features: &[crate::ctr::CtrFeatureColumn],
    cat_learn_permutation: &[i32],
    cat_averaging_permutation: &[i32],
    target_class: &[usize],
    cat_eligible_buckets: &[Vec<u32>],
    eligible_absolute: &[usize],
    ctr_border_count: usize,
) -> CbResult<cb_compute::DeviceCtrConfig> {
    let cast_perm = |perm: &[i32]| -> CbResult<Vec<u32>> {
        perm.iter()
            .map(|&p| {
                u32::try_from(p).map_err(|_| {
                    CbError::OutOfRange(format!(
                        "device CTR permutation entry {p} is negative (malformed fold order)"
                    ))
                })
            })
            .collect()
    };
    let device_target_class: Vec<u32> = target_class
        .iter()
        .map(|&c| u32::try_from(c).unwrap_or(u32::MAX))
        .collect();

    // Group columns by `(ctr_type, projection)` — the `UsedCtrSplits` identity
    // the cat-feature weight lifts on (multi-prior columns share one group).
    let mut group_keys: Vec<(i8, crate::TProjection)> = Vec::new();
    let build_columns = |cols: &[crate::ctr::CtrFeatureColumn],
                         group_keys: &mut Vec<(i8, crate::TProjection)>|
     -> CbResult<Vec<cb_compute::DeviceCtrColumn>> {
        cols.iter()
            .map(|col| {
                // FPP-09 (T07): ONE raw bucket column per projection member, in the
                // projection's SORTED member order — the same order `TProjection`
                // guarantees (sort + dedup in `from_features`) and the same order
                // `combined_hash` folds in. That shared order is what makes the device's
                // `combine_projection_bins` output integer-identical to the CPU's combined
                // bucket identity (PLAN V-4), so a `k`-member projection needs no new
                // accumulation semantics — only all `k` columns.
                //
                // Emitting only member 0 (which is what this did before FPP-09) would make
                // the device score a COMBINATION split from ONE member's bins: wrong, not
                // merely worse. The residual `fold_cat_hash` collision asymmetry between
                // the CPU's string-hash fold and the device's perfect-hash-bucket fold is
                // documented on `ctr_types_are_device_covered`, detected by the ≤1e-5 e2e
                // bar, and not prevented here.
                if col.projection.cat_features().is_empty() {
                    return Err(CbError::Degenerate(
                        "device CTR column with an empty projection".to_owned(),
                    ));
                }
                let members = col
                    .projection
                    .cat_features()
                    .iter()
                    .map(|&abs| {
                        let pos = eligible_absolute
                            .iter()
                            .position(|&a| a == abs)
                            .ok_or_else(|| {
                                CbError::OutOfRange(format!(
                                    "device CTR projection member {abs} is not a \
                                     CTR-eligible categorical feature"
                                ))
                            })?;
                        cat_eligible_buckets.get(pos).cloned().ok_or_else(|| {
                            CbError::OutOfRange(format!(
                                "no raw bucket column for CTR-eligible feature position {pos}"
                            ))
                        })
                    })
                    .collect::<CbResult<Vec<Vec<u32>>>>()?;
                let prior = col.prior_num / col.prior_denom;
                let (shift, norm) = crate::ctr::calc_normalization(prior);
                let borders: Vec<f64> = (0..ctr_border_count)
                    .map(|k| {
                        let v_k = (k as f64 + 1.0) * norm / ctr_border_count as f64 - shift;
                        v_k.next_down()
                    })
                    .collect();
                let key = (col.ctr_type, col.projection.clone());
                let group = match group_keys.iter().position(|g| *g == key) {
                    Some(g) => g,
                    None => {
                        group_keys.push(key);
                        group_keys.len() - 1
                    }
                };
                Ok(cb_compute::DeviceCtrColumn {
                    member_bins: members,
                    prior,
                    borders,
                    bucket_count: col.bucket_count,
                    weight_group: u32::try_from(group).unwrap_or(u32::MAX),
                })
            })
            .collect()
    };

    let columns = build_columns(materialized_ctr_features, &mut group_keys)?;
    // The averaging half shares the identity groups (same specs, different order).
    let avg_columns = build_columns(averaging_ctr_features, &mut group_keys)?;

    Ok(cb_compute::DeviceCtrConfig {
        permutation: cast_perm(cat_learn_permutation)?,
        target_class: device_target_class.clone(),
        columns,
        averaging: Some(cb_compute::DeviceCtrAveraging {
            permutation: cast_perm(cat_averaging_permutation)?,
            target_class: device_target_class,
            columns: avg_columns,
        }),
        cat_eligible_buckets: cat_eligible_buckets.to_vec(),
        model_size_reg: model_size_reg_default(),
    })
}

/// A held-out evaluation set feeding the overfitting detector (TRAIN-06). The
/// `feature_values` reuse the training feature borders (the model's float-feature
/// borders) for the `value > border` split tests.
pub struct EvalSet<'a> {
    /// `feature_values[f]` is eval float feature `f`'s per-object `f32` column.
    pub feature_values: &'a [Vec<f32>],
    /// Eval per-object target labels.
    pub target: &'a [f64],
    /// `cat_columns[c]` is eval categorical column `c`'s per-object RAW string
    /// values (E21, enabling SPEC-CTRT-17): under `counter_calc_method = Full`
    /// upstream tallies learn **+ every eval set** into the Counter bucket
    /// totals (`online_ctr.cpp:716-729`), so the eval categorical data must be
    /// carriable at all. Empty (`&[]`) on every numeric path — exactly the
    /// pre-E21 semantics, byte-identical.
    pub cat_columns: &'a [Vec<String>],
}

/// The ranking (grouped) structure a ranking loss reads (LOSS-04, D-6.3-03):
/// per-object `group_id` / `subgroup_id` and explicit `pairs`. Threaded into
/// [`train_ranking`] → [`train_inner`]; the grouped view ([`QueryInfo`]) is built
/// ONCE per fit via [`build_query_info`] and lowered to a compute-tier
/// `Vec<GroupSpan>` at the der site. Empty (all-empty columns) for the non-ranking
/// entry points, so the pointwise der site stays byte-identical (D-04).
#[derive(Debug, Clone, Copy, Default)]
pub struct RankingData<'a> {
    /// Per-object group id (contiguous, unique runs — `query.h:48-67`).
    pub group_id: &'a [u64],
    /// Per-object subgroup id (optional; empty when absent).
    pub subgroup_id: &'a [u64],
    /// Explicit ranking pairs (global `(winner_id, loser_id)`).
    pub pairs: &'a [Pair],
}

/// Lower a `cb-train` [`QueryInfo`] grouped view into the compute-tier
/// [`GroupSpan`] the grouped der seam consumes (LOSS-04). The compute tier
/// re-declares the plain-data shape to keep `cb-compute` free of a `cb-train`
/// dependency (06.3-01 layering decision); this is the trainer-side lowering.
fn lower_query_info(groups: &[QueryInfo]) -> Vec<GroupSpan> {
    groups
        .iter()
        .map(|g| GroupSpan {
            begin: g.begin,
            end: g.end,
            weight: g.weight,
            competitors: g
                .competitors
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|c| RankingCompetitor {
                            id: c.id,
                            weight: c.weight,
                        })
                        .collect()
                })
                .collect(),
        })
        .collect()
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

/// [`tree_eval_contribution`] for a NON-SYMMETRIC (Lossguide / Depthwise) tree:
/// walk the flat node graph to the object's leaf via the shared
/// [`device_leaf_of_nonsym`] transcription and return that leaf's value. A
/// malformed / cyclic graph contributes `0` (defensive; the grower supplies
/// valid graphs), mirroring the oblivious arm's out-of-range convention.
fn nonsym_eval_contribution(
    tree: &NonSymmetricTree,
    matrix: &FeatureMatrix,
    obj: usize,
) -> f64 {
    device_leaf_of_nonsym(
        obj,
        &tree.splits,
        &tree.step_nodes,
        &tree.node_id_to_leaf_id,
        matrix.feature_values,
    )
    .and_then(|leaf| tree.leaf_values.get(leaf).copied())
    .unwrap_or(0.0)
}

/// [`tree_eval_contribution`] for a REGION path tree: the walk-until-diverge
/// bin (`AddRegionImpl`) transcribed against the eval matrix — `bin = 0; for
/// level { split = one_hot ? val == border : val > border; if split != direction
/// { break }; bin += 1 }`, then `leaf_values[bin]`. Every access is checked; a
/// missing column or level halts the walk at the bin reached so far, exactly as
/// the trainer's own region walk does.
fn region_eval_contribution(tree: &RegionTree, matrix: &FeatureMatrix, obj: usize) -> f64 {
    let mut bin = 0usize;
    for level in 0..tree.splits.len() {
        let (Some(s), Some(&dir), Some(&oh)) = (
            tree.splits.get(level),
            tree.directions.get(level),
            tree.one_hot.get(level),
        ) else {
            break;
        };
        let Some(val) = matrix
            .feature_values
            .get(s.feature)
            .and_then(|col| col.get(obj))
            .map(|&v| f64::from(v))
        else {
            break;
        };
        let split = if oh { val == s.border } else { val > s.border };
        if split == dir {
            bin = bin.saturating_add(1);
        } else {
            break;
        }
    }
    tree.leaf_values.get(bin).copied().unwrap_or(0.0)
}

/// The eval-set contribution of the tree the CURRENT iteration just grew,
/// whichever of the three ensembles received it.
///
/// A model is EITHER all-oblivious OR all-non-symmetric OR all-region (see the
/// push dispatch in the boosting loop), so at most one of these vectors is ever
/// non-empty and the branch order below is a formality rather than a priority.
///
/// This used to read `trees.last()` alone. Under `grow_policy=Lossguide` /
/// `Depthwise` every tree lands in `non_symmetric_trees` and `trees` stays
/// EMPTY, so the eval approximant never advanced past `bias`: the metric
/// returned the same constant for every iteration, `BestModelTracker` never saw
/// an improvement (leaving `best_iteration() == Some(0)`, which truncated the
/// returned model to a single tree under `use_best_model`), and the `Iter`
/// detector stopped after `od_wait` iterations regardless of `iterations`.
fn last_tree_eval_contribution(
    trees: &[ObliviousTree],
    non_symmetric_trees: &[NonSymmetricTree],
    region_trees: &[RegionTree],
    matrix: &FeatureMatrix,
    obj: usize,
) -> f64 {
    if let Some(tree) = region_trees.last() {
        return region_eval_contribution(tree, matrix, obj);
    }
    if let Some(tree) = non_symmetric_trees.last() {
        return nonsym_eval_contribution(tree, matrix, obj);
    }
    trees
        .last()
        .map_or(0.0, |tree| tree_eval_contribution(tree, matrix, obj))
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
                cat_columns: es.cat_columns,
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
#[allow(clippy::too_many_arguments)]
pub fn train_with_eval_sets<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    staged_out: Option<&mut Vec<f64>>,
    eval_sets: &[EvalSet],
    history: Option<&mut EvalMetricHistory>,
) -> CbResult<Model> {
    // The numeric entry point carries NO categorical columns — byte-identical to
    // before (empty cat set ⇒ empty CTR candidates ⇒ no materialization). The
    // baked ctr_data is empty here and discarded (train's return type is UNCHANGED).
    let (model, _baked) = train_inner(
        runtime,
        feature_values,
        feature_borders,
        &[],
        target,
        weights,
        params,
        staged_out,
        eval_sets,
        history,
        RankingData::default(),
        None,
    )?;
    Ok(model)
}

/// Train a RANKING model (LOSS-04, D-6.3-03): the numeric [`train`] entry plus the
/// grouped [`RankingData`] (`group_id` / `subgroup_id` / `pairs`) a ranking loss
/// reads. When `params.loss` is a querywise/ranking loss
/// ([`is_grouped_loss`]) the der site builds the [`QueryInfo`] grouped view once
/// and routes the gradient through the grouped seam
/// (`Runtime::compute_gradients_grouped`); the leaf path reuses the existing
/// pointwise estimators (QueryRMSE / QuerySoftMax are per-object der, no pairwise
/// Cholesky path). For a NON-ranking loss this is byte-identical to [`train`] (the
/// grouped view is built but never consumed — D-04).
///
/// # Errors
/// As [`train`], plus [`CbError::Degenerate`] / [`CbError::OutOfRange`] from
/// [`build_query_info`] on malformed group/pair structure.
#[allow(clippy::too_many_arguments)]
pub fn train_ranking<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    staged_out: Option<&mut Vec<f64>>,
    ranking: RankingData,
) -> CbResult<Model> {
    let (model, _baked) = train_inner(
        runtime,
        feature_values,
        feature_borders,
        &[],
        target,
        weights,
        params,
        staged_out,
        &[],
        None,
        ranking,
        None,
    )?;
    Ok(model)
}

/// Train a CAT-AWARE model: thread categorical columns into training, computing
/// OnLearnOnly per-feature cardinalities and materializing a per-candidate
/// combined-projection online CTR feature column the tree search can split on
/// (ORD-05 / D-05, the upstream `greedy_tensor_search.cpp` AddTreeCtrs +
/// per-fold online-CTR-during-growth path).
///
/// `cat_columns[f]` is categorical feature `f`'s per-object value column (already
/// in the A4 string form — integer-coded values pre-stringified via
/// [`cb_data::stringify_int_category`]). The numeric `feature_values` /
/// `feature_borders` / `target` / `weights` / `params` / `staged_out` arguments
/// are exactly as [`train`]. When `cat_columns` is empty `train_cat` is
/// byte-identical to [`train`] (no candidates, no materialization).
///
/// Returns the trained [`Model`] PLUS the baked whole-set inference [`BakedCtrData`]
/// (ORD-05, Plan 05-14): one [`BakedCtrTable`] per DISTINCT chosen CTR split,
/// carrying the whole-set per-bucket class counts (keyed by the combined projection
/// hash the apply path reconstructs) and the inference `(Shift, Scale)` derived from
/// the prior PAIR. The e2e call site attaches it to the canonical model via
/// `cb_model::Model::with_ctr_data` (after `cb_model::CtrData::from_baked`). When
/// `cat_columns` is empty the baked data is empty and the model is byte-identical to
/// [`train`].
///
/// # Errors
/// As [`train`], plus [`CbError::OutOfRange`] from cardinality counting on a
/// column exceeding the perfect-hash `u32::MAX` bound, or any error
/// [`crate::materialize_ctr_feature`] / [`crate::bake_ctr_table`] surfaces.
#[allow(clippy::too_many_arguments)]
pub fn train_cat<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    cat_columns: &[Vec<String>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    staged_out: Option<&mut Vec<f64>>,
) -> CbResult<(Model, BakedCtrData)> {
    train_inner(
        runtime,
        feature_values,
        feature_borders,
        cat_columns,
        target,
        weights,
        params,
        staged_out,
        &[],
        None,
        RankingData::default(),
        None,
    )
}

/// [`train_cat`] plus held-out evaluation sets (E21, enabling SPEC-CTRT-17) —
/// the categorical mirror of [`train_with_eval_sets`], except the baked CTR
/// data is RETURNED rather than discarded (a categorical model without its
/// baked tables cannot predict).
///
/// Each eval set may carry its own `cat_columns`; under
/// `counter_calc_method = Full` (threaded by E22) those columns join the
/// Counter bucket tally exactly as upstream's learn-plus-every-test-set hash
/// array does (`online_ctr.cpp:716-729`).
///
/// # Errors
/// As [`train_cat`], plus [`CbError::LengthMismatch`] if any eval set's
/// categorical column length disagrees with that set's target length.
#[allow(clippy::too_many_arguments)]
pub fn train_cat_with_eval_sets<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    cat_columns: &[Vec<String>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    staged_out: Option<&mut Vec<f64>>,
    eval_sets: &[EvalSet],
    history: Option<&mut EvalMetricHistory>,
) -> CbResult<(Model, BakedCtrData)> {
    for (si, es) in eval_sets.iter().enumerate() {
        for (ci, col) in es.cat_columns.iter().enumerate() {
            if col.len() != es.target.len() {
                return Err(CbError::LengthMismatch {
                    column: format!("eval set {si} categorical column {ci}"),
                    expected: es.target.len(),
                    actual: col.len(),
                });
            }
        }
    }
    train_inner(
        runtime,
        feature_values,
        feature_borders,
        cat_columns,
        target,
        weights,
        params,
        staged_out,
        eval_sets,
        history,
        RankingData::default(),
        None,
    )
}

/// Train a numeric model with a periodic on-disk CHECKPOINT, resuming
/// automatically from `snapshot.snapshot_file` when it already exists and its
/// fingerprint matches this run (ORCH-03-S7).
///
/// Mirrors upstream's `snapshot_file` / `snapshot_interval` semantics: a
/// checkpoint is written at completed-iteration boundaries no more often than
/// `snapshot_interval`, and a subsequent call with the same configuration picks up
/// where the previous one stopped instead of retraining from scratch.
///
/// Snapshotting is defined only for plain float-only CPU boosting — see
/// [`snapshot_scope_ok`] for the exact admitted regime and the reason each excluded
/// feature is excluded.
///
/// Returns the trained model together with the iteration the run RESUMED FROM —
/// `0` for a fresh fit, `K` when a `K`-tree checkpoint was picked up. That number
/// is the only observable difference between a resume and a from-scratch retrain
/// (both produce the same model, which is the point), so callers that need to know
/// whether the checkpoint was used must read it here.
///
/// # Errors
/// [`CbError::Snapshot`] if the configuration is outside the snapshot regime (no
/// file is written), if the existing snapshot's fingerprint does not match this
/// run's, or on any snapshot I/O / codec failure. Otherwise the same errors as
/// [`train`].
pub fn train_with_snapshot<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    snapshot: &crate::snapshot::SnapshotConfig,
) -> CbResult<(Model, usize)> {
    // The resume point is determined by PEEKING the checkpoint here rather than by
    // growing `train_inner`'s return type: that would force a three-tuple
    // destructure at all four of its existing call sites, turning an additive
    // change into a signature change across paths that have nothing to do with
    // snapshots. `train_inner` re-reads and restores the file itself; on this
    // deterministic single-threaded path the extra read is cheap and side-effect
    // free.
    //
    // The fingerprint check runs BEFORE any training, so a mismatched checkpoint
    // costs the caller nothing.
    let resume_from = if snapshot.snapshot_file.exists() {
        let stored = crate::snapshot::read_from(&snapshot.snapshot_file)?;
        let current =
            crate::snapshot::fingerprint(params, target.len(), feature_borders, target, weights);
        crate::snapshot::check_resume(stored.fingerprint, current)?;
        stored.completed_iters
    } else {
        0
    };

    let (model, _baked) = train_inner(
        runtime,
        feature_values,
        feature_borders,
        &[],
        target,
        weights,
        params,
        None,
        &[],
        None,
        RankingData::default(),
        Some(snapshot),
    )?;
    Ok((model, resume_from))
}

/// Quantize the float design matrix into the device's feature-major cindex
/// (`bins[feature * n + obj] = #borders the object's value strictly exceeds`) plus
/// the uniform per-feature bin-line size (`n_bins = max_f(feature_borders[f].len())
/// + 1`), for the GPUT-01 device grow seam ([`Runtime::begin_device_training`]).
///
/// The bin count is exactly the number of ascending borders `value > border`
/// (the SAME test [`FeatureMatrix::passes_float`] applies), so the device split
/// `quantized_bin > bin_id` is equivalent to the CPU `value >
/// feature_borders[feature][bin_id]` — this is the round-trip guarantee behind the
/// `bin_id -> border` join (Pattern 4) the device tree is folded through. Only the
/// numeric float columns are quantized (the device path is gated off the cat / CTR
/// configs). Uses checked `.get` only (no panic / raw index).
///
/// # Precondition
/// Every `feature_borders[f]` MUST be sorted ascending (`select_borders_greedy_logsum`'s
/// contract — it always sorts + dedups before returning). This function reads
/// borders via `partition_point`, a binary search that is only correct on
/// sorted input: on an unsorted slice it silently returns a wrong bin index
/// (no panic) instead of the linear-scan `filter(|&&b| v > b).count()` an
/// unsorted-safe reader might expect. A `debug_assert!` below catches a
/// precondition violation in debug/test builds; release builds trust the
/// invariant for the `O(log borders)` win.
fn quantize_feature_major(
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    n: usize,
) -> (Vec<u32>, usize) {
    let n_features = feature_values.len();
    let mut bins = vec![0u32; n_features * n];
    let n_bins = feature_borders
        .iter()
        .fold(0usize, |acc, borders| acc.max(borders.len() + 1));
    // Columns are independent (each feature's bins depend only on that feature's
    // values + borders), so the per-feature quantization runs in parallel over
    // the disjoint feature-major stripes. Borders are sorted ascending
    // (`select_borders_greedy_logsum` sorts + dedups), so
    // `partition_point(|b| v > b)` returns EXACTLY the serial
    // `filter(|b| v > b).count()` — the `v > b` predicate is monotone
    // (true-prefix) over an ascending border list, including the NaN case
    // (all-false -> 0). Byte-identical bins, O(log borders) per value.
    bins.par_chunks_mut(n)
        .enumerate()
        .for_each(|(f, stripe)| {
            let borders = feature_borders.get(f).map_or(&[][..], Vec::as_slice);
            debug_assert!(
                borders.windows(2).all(|w| match w {
                    [a, b] => a <= b,
                    _ => true,
                }),
                "quantize_feature_major: feature_borders[{f}] must be ascending-sorted \
                 for partition_point to be correct (see function doc precondition)",
            );
            let col = feature_values.get(f).map_or(&[][..], Vec::as_slice);
            for (i, slot) in stripe.iter_mut().enumerate() {
                let v = col.get(i).copied().map_or(0.0_f64, f64::from);
                *slot = borders.partition_point(|&b| v > b) as u32;
            }
        });
    (bins, n_bins)
}

/// The device quantizer for a pool that MAY carry one-hot columns (SPEC-OH-21),
/// returning `(bins, n_bins, real_folds)`.
///
/// This is the ONE device-quantize entry the trainer calls — on EVERY device-eligible
/// pool, float-only included (with an empty `cat_bins`). That is deliberate: it is what
/// makes `real_folds` always populated, so the session's
/// `real_folds.len() == eff_n_features` check can stay unconditional instead of
/// degenerating into the silently-inert bound SPEC-OH-22 exists to eliminate.
///
/// - **Layout.** The device feature axis is the CONCATENATION `float | one-hot`: device
///   feature index `n_float + c` is one-hot column `c`. The one-hot columns therefore
///   form one CONTIGUOUS range, which is what lets the split scorer bound its second
///   pass with a single `feature_lo = n_float`.
/// - **Bins.** Float stripes are produced by delegating to [`quantize_feature_major`]
///   with its body and signature unmodified, so the float bin bytes are provably
///   identical (SPEC-OH-31). One-hot stripes are the `PerfectHash` bin columns copied
///   VERBATIM — there is no second binning of a categorical column anywhere.
/// - **`n_bins`.** `max(float n_bins, max cat cardinality).max(1)`. The `.max(1)` and the
///   cat term matter: a 0-float pool would otherwise report `n_bins == 0` and the backend
///   session declines on `n_features == 0 || n_bins == 0`, making SPEC-OH-20's 0-float
///   target unreachable.
/// - **`real_folds`.** The per-feature REAL cardinality — `borders[f].len() + 1` for a
///   float feature, the column's `PerfectHash` cardinality for a one-hot column. This is
///   a SEPARATE array and is **not** `TCFeature.folds`, which on the production path is
///   the padded uniform line width and bounds nothing (see the `TCFeature.folds` doc in
///   `cb-backend`'s `gpu_runtime::cindex`). It is also NOT fixable by passing true
///   cardinalities into `pack_cindex`: that would change `feature_bits` and hence the
///   packed words for every pool, float-only included.
fn quantize_feature_major_with_one_hot(
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    cat_bins: &[Vec<u32>],
    n: usize,
) -> (Vec<u32>, usize, Vec<u32>) {
    let n_float = feature_values.len();
    // Float prefix: delegate, so the float bytes cannot drift from the plain entry.
    let (float_bins, float_n_bins) = quantize_feature_major(feature_values, feature_borders, n);

    let mut real_folds: Vec<u32> = Vec::with_capacity(n_float + cat_bins.len());
    for f in 0..n_float {
        let borders = feature_borders.get(f).map_or(0usize, Vec::len);
        real_folds.push(u32::try_from(borders + 1).unwrap_or(u32::MAX));
    }

    // One-hot suffix: cardinality is `max bin + 1` over the column (the `PerfectHash`
    // bins are dense `0..cardinality` by construction, so this is exact).
    let mut bins = float_bins;
    bins.reserve(cat_bins.len() * n);
    let mut max_cat_cardinality = 0usize;
    for col in cat_bins {
        let cardinality = col.iter().copied().max().map_or(0usize, |m| m as usize + 1);
        max_cat_cardinality = max_cat_cardinality.max(cardinality);
        real_folds.push(u32::try_from(cardinality).unwrap_or(u32::MAX));
        for obj in 0..n {
            bins.push(col.get(obj).copied().unwrap_or(0));
        }
    }

    let n_bins = float_n_bins.max(max_cat_cardinality).max(1);
    (bins, n_bins, real_folds)
}

/// RAII teardown for the GPUT-01 device training session (T-10-24): guarantees
/// [`Runtime::end_device_training`] runs on EVERY exit path from [`train_inner`] —
/// including the `?` error path — once [`Runtime::begin_device_training`] opened a
/// session (`active == true`). Releasing the device-resident session is a
/// best-effort teardown (the backend's `end` drops the session; the CPU default is
/// a no-op returning `Ok(())`), so a teardown error is swallowed on `Drop` rather
/// than masking the training result. When no session was opened (`active == false`,
/// the CPU-fallback path) `Drop` is inert.
struct DeviceSessionGuard<'r, R: Runtime> {
    runtime: &'r R,
    active: bool,
}

impl<R: Runtime> Drop for DeviceSessionGuard<'_, R> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.runtime.end_device_training();
        }
    }
}

/// Partition the categorical columns by encoding path (SPEC-OH-04), returning
/// `(one_hot_absolute, ctr_absolute)` — both ASCENDING absolute `cat_columns`
/// indices.
///
/// Derived from ONE [`crate::candidates::route_categorical`] match per column,
/// so the two lists are DISJOINT BY CONSTRUCTION: two independent filters could
/// drift (a routing-rule change touching only one of them would materialize the
/// same feature on both paths, double-counting its contribution). A constant
/// column (`cardinality <= 1`, [`crate::candidates::EncodingPath::Skip`])
/// appears in NEITHER list.
///
/// The CTR list is byte-identical to the pre-one-hot `eligible_absolute`.
fn partition_cat_columns(
    cat_cardinalities: &[u32],
    one_hot_max_size: u32,
) -> (Vec<usize>, Vec<usize>) {
    let mut one_hot = Vec::new();
    let mut ctr = Vec::new();
    for (abs_idx, &card) in cat_cardinalities.iter().enumerate() {
        match crate::candidates::route_categorical(card, one_hot_max_size) {
            crate::candidates::EncodingPath::OneHot => one_hot.push(abs_idx),
            crate::candidates::EncodingPath::Ctr => ctr.push(abs_idx),
            crate::candidates::EncodingPath::Skip => {}
        }
    }
    (one_hot, ctr)
}

/// The widest one-hot column the device grower accepts (SPEC §9 R10).
///
/// The device histogram line is padded to one of `{32, 64, 128, 256}` bins and is
/// SHARED by every feature, so a one-hot column's cardinality must fit alongside the
/// float bin count. `32` keeps a binary/low-cardinality pool inside the narrowest
/// (fastest) legal line, which is the regime `one_hot_max_size` actually produces —
/// upstream's default is `2`, and its documented ceiling for the one-hot route is far
/// below this.
///
/// Exceeding it is NOT an error: the fit falls back to the CPU grower, which handles
/// any cardinality. Aborting an otherwise-valid fit would be strictly worse behavior
/// than training it correctly a bit slower.
pub(crate) const DEVICE_ONE_HOT_MAX_CARDINALITY: u32 = 32;

/// Whether every one-hot column fits the device histogram line
/// ([`DEVICE_ONE_HOT_MAX_CARDINALITY`]). A pool with no one-hot columns trivially
/// fits, which is what keeps the float-only path unchanged (SPEC-OH-31).
fn one_hot_cardinalities_fit_the_device(cardinalities: &[u32]) -> bool {
    cardinalities
        .iter()
        .all(|&c| c <= DEVICE_ONE_HOT_MAX_CARDINALITY)
}

/// Whether the pool has ANY feature the level search can rank (SPEC-OH-20).
///
/// This is clause 11 of `device_host_eligible`, extracted so it can be asserted
/// on its own (the full eligibility expression needs a whole fit context). Before
/// SPEC-OH-20 it read `matrix.n_features() > 0` — float columns only — which
/// silently excluded a pool routed entirely one-hot from the device grower.
///
/// A one-hot cat column IS scorable: `AddOneHotFeatures` contributes
/// `cat_bin == value` candidates to the SAME level argmax the float borders feed
/// (SPEC-OH-06). Only a pool with neither kind has nothing to rank.
fn has_any_scorable_feature(matrix: &crate::tree::FeatureMatrix<'_>) -> bool {
    matrix.n_features() > 0 || matrix.n_cat_features() > 0
}

/// Materialize the one-hot-routed categorical columns (SPEC-OH-05), returning
/// `(bins, hash_by_bin)`, both indexed by ONE-HOT POSITION (the index into
/// `one_hot_abs`), NOT by absolute cat-column index.
///
/// - `bins[p][obj]` is the object's FIRST-SEEN [`cb_data::PerfectHash`] bin —
///   produced by [`cb_data::perfect_hash_bins`], the single sanctioned hashing
///   primitive (SPEC §3); no second hashing loop exists.
/// - `hash_by_bin[p][bin]` is the raw `calc_cat_feature_hash` value that
///   produced `bin` — the EXACT inverse of the bin assignment, and the table
///   [`crate::Model::one_hot_bin_to_hash`] carries to the model lift so a
///   trainer-side bin can be re-expressed in upstream's raw-hash split space
///   (SPEC-OH-09).
///
/// The inverse is built by zipping the raw column with the returned bins, NOT by
/// sorting distinct hashes: `PerfectHash::remap_bounded` assigns
/// `bin = map.len()` on first sight, so bin order is ENCOUNTER order.
///
/// # Validity
/// The table is valid ONLY for the exact learn-set column it was built from —
/// bins are first-seen per column, so a different row order yields a different
/// (equally valid) table.
///
/// # Errors
/// [`CbError::OutOfRange`] if an absolute index is not a column of
/// `cat_columns`; [`CbError::Degenerate`] if the built table is not exactly one
/// entry per distinct value (an internal invariant violation, not a data
/// condition); or any error [`cb_data::perfect_hash_bins`] surfaces.
fn build_one_hot_columns(
    cat_columns: &[Vec<String>],
    one_hot_abs: &[usize],
) -> CbResult<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    let mut bins_out = Vec::with_capacity(one_hot_abs.len());
    let mut hash_out = Vec::with_capacity(one_hot_abs.len());

    for &abs_idx in one_hot_abs {
        let col = cat_columns.get(abs_idx).ok_or_else(|| {
            CbError::OutOfRange(format!(
                "one-hot cat column {abs_idx} out of range ({} cat columns)",
                cat_columns.len()
            ))
        })?;
        let as_str: Vec<&str> = col.iter().map(String::as_str).collect();
        let bins = cb_data::perfect_hash_bins(&as_str)?;

        // Zip raw <-> bin and record each bin's raw hash on first sight. The
        // table is grown to `bin + 1` as bins appear; because `remap_bounded`
        // hands out `0, 1, 2, …` in encounter order, growth is always by one.
        let mut hash_by_bin: Vec<Option<u32>> = Vec::new();
        for (raw, &bin) in col.iter().zip(bins.iter()) {
            let idx = bin as usize;
            if idx >= hash_by_bin.len() {
                hash_by_bin.resize(idx.saturating_add(1), None);
            }
            if let Some(slot) = hash_by_bin.get_mut(idx) {
                if slot.is_none() {
                    *slot = Some(cb_data::calc_cat_feature_hash(raw));
                }
            }
        }

        // Every bin in [0, cardinality) must have been filled: a hole would make
        // the model lift emit a wrong (or missing) `value_hash` for that split.
        let cardinality = hash_by_bin.len();
        let table: Vec<u32> = hash_by_bin.iter().flatten().copied().collect();
        if table.len() != cardinality {
            return Err(CbError::Degenerate(format!(
                "one-hot cat column {abs_idx}: bin -> hash table has {} of {cardinality} entries",
                table.len()
            )));
        }

        bins_out.push(bins);
        hash_out.push(table);
    }

    Ok((bins_out, hash_out))
}

/// Admit a training run into the snapshot regime, or reject it with a typed error
/// naming the offending feature (ORCH-03-S5).
///
/// Slice 1 snapshots exactly the configuration whose loop-carried mutable state is
/// `{approx, trees, rng}` — established by the audit in
/// `.planning/plans/snapshot-resume/TASK-01-findings.md` against this file. Every
/// predicate below marks state that a checkpoint does NOT carry, so resuming such a
/// run would continue from a partially-restored trainer and silently produce a
/// model that is neither the interrupted run's nor a fresh run's. Refusing up front
/// is the only honest option; each rejection names what to turn off.
///
/// Two predicates deserve their own note:
///
/// * A `Loss::Custom(_)` objective / `EvalMetric::Custom(_)` metric is an opaque
///   `Arc<dyn …>` whose only equality is process-local pointer identity. No
///   cross-process fingerprint can tell two custom instances apart, so a resume
///   could silently pair a snapshot with a DIFFERENT objective. Neither is caught
///   by any other predicate: `Loss::Custom` is single-dimension and is not a
///   grouped loss.
/// * A requested `staged_out` buffer accumulates one row per iteration and is NOT
///   part of the checkpoint, so a resumed run would return `N-K` staged rows where
///   a straight-through run returns `N`. (Found by the TASK-01 audit; it is a
///   `train_inner` PARAMETER, not a local, which is why the original state audit
///   missed it.)
#[allow(clippy::too_many_arguments)]
fn snapshot_scope_ok(
    params: &BoostParams,
    cat_columns: &[Vec<String>],
    eval_sets: &[EvalSet],
    approx_dimension: usize,
    penalties_active: bool,
    device_active: bool,
    staged_requested: bool,
    ranking: &RankingData,
) -> CbResult<()> {
    let reject = |what: &str| {
        Err(CbError::Snapshot(format!(
            "training snapshots are supported only for plain float-only CPU boosting; \
             this run uses {what}"
        )))
    };

    if matches!(params.loss, Loss::Custom(_)) {
        return reject("a custom objective (its identity cannot be fingerprinted across runs)");
    }
    if matches!(params.eval_metric, Some(EvalMetric::Custom(_))) {
        return reject("a custom eval metric (its identity cannot be fingerprinted across runs)");
    }
    if is_grouped_loss(&params.loss) {
        return reject("a grouped / ranking loss");
    }
    if !ranking.group_id.is_empty() || !ranking.subgroup_id.is_empty() || !ranking.pairs.is_empty()
    {
        return reject("ranking data (group_id / subgroup_id / pairs)");
    }
    if !cat_columns.is_empty() {
        return reject("categorical features");
    }
    if !matches!(params.boosting_type, EBoostingType::Plain) {
        return reject("ordered boosting");
    }
    if !eval_sets.is_empty() {
        return reject("eval sets (the overfitting detector's state is not checkpointed)");
    }
    if !matches!(params.bootstrap_type, EBootstrapType::No) {
        return reject("bootstrap sampling");
    }
    if params.random_strength != 0.0 {
        return reject("a non-zero random_strength");
    }
    if approx_dimension != 1 {
        return reject("a multi-dimensional approximant");
    }
    if penalties_active {
        return reject("feature weights / penalties");
    }
    if !matches!(params.grow_policy, EGrowPolicy::SymmetricTree) {
        return reject("a non-symmetric grow policy");
    }
    if device_active {
        return reject("device (GPU) training");
    }
    if staged_requested {
        return reject(
            "a staged-prediction buffer (a resumed run would emit only the post-resume rows)",
        );
    }
    Ok(())
}

/// The shared boosting loop body for the numeric ([`train_with_eval_sets`]) and
/// cat-aware ([`train_cat`]) entry points. `cat_columns` is EMPTY for the numeric
/// path (byte-identical to the pre-05-11 driver); a non-empty `cat_columns`
/// computes OnLearnOnly cardinalities, feeds the REAL cat set to
/// [`tensor_ctr_candidates`], and materializes a per-candidate combined-projection
/// online CTR feature column ([`crate::materialize_ctr_feature`]).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn train_inner<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    cat_columns: &[Vec<String>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    mut staged_out: Option<&mut Vec<f64>>,
    eval_sets: &[EvalSet],
    mut history: Option<&mut EvalMetricHistory>,
    ranking: RankingData,
    snapshot: Option<&crate::snapshot::SnapshotConfig>,
) -> CbResult<(Model, BakedCtrData)> {
    // ORCH-03 TASK-03: the parameter is threaded but not yet read — the write hook
    // (TASK-06) and the resume block (TASK-07) are the first consumers. Every
    // existing caller passes `None`, so this task is behavior-preserving by
    // construction (the D-04 anchor: the FULL `cb-train` suite must stay green).
    let _ = &snapshot;
    check_depth(params.depth)?;

    // Validate the loss's hyperparameters before any training work
    // (T-06.1.01-01 / T-06.1.01-02): an out-of-domain q/delta/alpha would yield
    // NaN/Inf derivatives that poison the histogram and leaf reductions, so it is
    // rejected up front with a typed CbError rather than producing a corrupt model.
    params.loss.validate()?;

    // Reject CTR types with no CPU training implementation (SPEC-CTRT-03) before
    // any accumulation or tree growth. Placed AFTER `loss.validate()` so the
    // existing loss-validation error precedence is unchanged.
    validate_ctr_types(params)?;

    // Reject the second-order (Newton) split-score functions on the CPU training
    // path (CR-01): `NewtonL2` / `NewtonCosine` reuse the L2 / Cosine score formula
    // VERBATIM and rely entirely on a der2-hessian histogram fill in the
    // `sum_weight` leaf-stat slot. The CPU scoring path
    // (`multi_dim_candidate_score`) only ever produces the FIRST-ORDER
    // (weight-count) reduction, so selecting a Newton score function would silently
    // compute its first-order counterpart instead of the requested second-order
    // score. These variants are GPU-only upstream (D-6.4-06); reject them up front
    // with a typed error rather than producing a silently-wrong split score.
    validate_score_function(params.score_function)?;

    // Reject unsupported (loss, leaf_method) combinations up front (WR-01 /
    // WR-02): an Exact method on a loss with no defined optimizer would silently
    // compute the weighted median instead of that loss's true optimum, and an
    // Lq{q<2} Newton step would inject inf/NaN into the leaf denominator.
    validate_leaf_method(&params.loss, params.leaf_method)?;

    // Reject malformed / unsupported monotone_constraints up front (FEAT-03 /
    // D-6.6-07): each entry must be a valid direction {-1,0,+1}.
    validate_monotone_constraints(&params.monotone_constraints)?;

    // Reject unsupported grow_policy combinations up front (FEAT-06 / D-6.6-04):
    // grow_policy=Region (CPU-unimplemented) and monotone_constraints × a
    // non-symmetric grow_policy (upstream rejects them — the monotone PAVA is
    // oblivious-only). These are the escalated-gap guards Plan 06.6-02 DEFERRED to
    // this plan because the `grow_policy` enum did not exist until now.
    validate_grow_policy(params.grow_policy, &params.monotone_constraints)?;

    // The multilabel losses (MultiLogloss / MultiCrossEntropy) carry a DIM-MAJOR
    // target of length `dim*n` (one label per dimension per object), so `n` cannot
    // be `target.len()` (that would be `dim*n`). Derive the OBJECT count `n` from
    // the feature columns instead; the label-set WIDTH (approx_dimension) is then
    // `target.len() / n` (`approx_dimension.cpp:22-23` IsMultiTargetObjective ->
    // targetDimension). For every other loss `n == target.len()` (per-object).
    let is_multilabel = matches!(
        params.loss,
        Loss::MultiLogloss | Loss::MultiCrossEntropy
    );
    let n = if is_multilabel {
        let n_obj = feature_values.first().map_or(0, Vec::len);
        if n_obj == 0 {
            return Err(CbError::Degenerate(
                "multilabel training requires at least one feature column with objects".to_owned(),
            ));
        }
        if target.len() % n_obj != 0 {
            return Err(CbError::LengthMismatch {
                column: "multilabel target".to_owned(),
                expected: target.len() - (target.len() % n_obj),
                actual: target.len(),
            });
        }
        n_obj
    } else {
        target.len()
    };
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
        let target_type = autolr_target_type(&params.loss);
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
    // Upstream's `TBoostingOptions::LearningRate` is a **float**, so the rate that
    // actually multiplies every leaf value is the f32-representable value, NOT the
    // f64 the caller supplied. For the ubiquitous `learning_rate = 0.1` the two
    // differ by a CONSTANT relative `1.4901161e-8`
    // (`f32(0.1) = 0.10000000149011612`); the factor lands on every leaf of every
    // tree and compounds through the boosting residuals.
    //
    // Pinned EXACTLY against the committed `one_hot_train/multi` fixture (real
    // catboost 1.2.10): all eight of tree 0's upstream leaf values equal ours
    // times `f32(0.1) / 0.1`, reproducing them to 6.9e-18 (one ulp). End to end
    // through production train→predict, `one_hot_train/default_binary` improves
    // from `1.998e-9` to `2.776e-17` against upstream.
    //
    // The error was invisible for the project's whole life because 1.49e-8 sits
    // four orders of magnitude under the ≤1e-5 oracle bar; it surfaced only when
    // it flipped a NEAR-TIED one-hot split, turning an 1e-8 arithmetic difference
    // into a 4.6e-2 prediction difference. Full measurements and the re-baseline
    // record:
    // `.planning/plans/one-hot-categorical-training/instrumented-ground-truth/LEARNING_RATE_F32.md`
    let learning_rate = f64::from(learning_rate as f32);

    // Per-object weights: default to 1.0 when no weights are supplied.
    let weights: Vec<f64> = if weights.is_empty() {
        vec![1.0; n]
    } else {
        weights.to_vec()
    };
    let sum_all_weights = sum_f64(&weights);

    // GROUPED (ranking) view (LOSS-04, D-6.3-03): for a querywise/ranking loss
    // build the `Vec<GroupSpan>` ONCE (mirroring upstream's per-fit
    // `TVector<TQueryInfo>`), lowered from the `cb-train::QueryInfo` view, so the
    // der site can route through the grouped seam each iteration. For every
    // NON-ranking loss this is `None` and the pointwise der site is byte-identical
    // (D-04 no-regression). `build_query_info` validates the group/pair structure
    // up front (contiguous-unique runs, in-range/in-group pairs) — a typed
    // CbError, never a panic.
    let mut group_spans: Option<Vec<GroupSpan>> = if is_grouped_loss(&params.loss) {
        let qi = build_query_info(
            n,
            ranking.group_id,
            ranking.subgroup_id,
            ranking.pairs,
            &weights,
        )?;
        Some(lower_query_info(&qi))
    } else {
        None
    };

    // YetiRank / YetiRankPairwise (Wave C) re-SAMPLE their pairwise competitors
    // every boosting iteration from the CURRENT approx (the pairs are not fixed —
    // `UpdatePairsForYetiRank` runs per tree, yetirank_helpers.cpp:347-393). The
    // per-query inner seeds are derived ONCE from the 2-level chain (single-thread,
    // blockCount=1); each iteration re-samples with those same seeds over the
    // updated approx. We capture the group relevances (the ranking target per
    // group) and the per-group seeds here so the per-iteration regeneration is a
    // cheap re-sample, not a re-derive.
    let is_yetirank = matches!(
        params.loss,
        Loss::YetiRank { .. } | Loss::YetiRankPairwise { .. }
    );
    let (yetirank_permutations, yetirank_decay) = match params.loss {
        Loss::YetiRank { permutations, decay }
        | Loss::YetiRankPairwise { permutations, decay } => (permutations, decay),
        _ => (0, 0.0),
    };
    // Per-query inner seeds (group order) + each group's [begin, end) + weight +
    // relevances, snapshotted from the group view for the per-iteration re-sample.
    let yetirank_groups: Vec<(usize, usize, f64, Vec<f64>)> = if is_yetirank {
        group_spans
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|g| {
                let relevs: Vec<f64> = (g.begin..g.end)
                    .map(|i| target.get(i).copied().unwrap_or(0.0))
                    .collect();
                (g.begin, g.end, g.weight, relevs)
            })
            .collect()
    } else {
        Vec::new()
    };
    // Per-tree YetiRank seeding driver (D-07 trainer-level RNG closure, 06.3-14
    // ext). The PRIOR model derived ONE fixed `derive_query_seeds(params.random_seed,
    // n_groups)` set and reused it for EVERY tree over ONE permutation fold — which
    // matched the standalone single-group self-oracle but DIVERGED from the live
    // trainer, whose `UpdatePairsForYetiRank` re-derives the per-query seed PER TREE
    // from the persistent `LearnProgress->Rand` (advanced through the structure /
    // split-search / leaf-estimation draws each tree) AND samples DISTINCT competitor
    // sets for the gradient/split recalc vs the leaf-value recalc. `YetiRankTreeSeeder`
    // reproduces that draw-for-draw (verified bit-exact vs the instrumented trainer's
    // per-tree, per-group first-Gumbel stream). `next_tree()` is called once per
    // boosting iteration below to yield this tree's deriv + leafval per-group seeds.
    // Candidate-sublist count for the per-level split-search draws: ONE
    // `OneFeature` candidate sublist per FLOAT feature that is a TRAINING candidate.
    // This is the count of Rsm selection `GenRandReal1` draws AND the count of
    // `SelectBestCandidate` Box-Muller normals per level (one `BestScore` per
    // candidate feature).
    //
    // WR-02 FIX (06.3-17): the trainer counts EVERY float feature it quantized in
    // the LEARN data — a feature that ends up UNUSED in the final model (no SELECTED
    // borders, e.g. corpus feature 2) STILL consumed an Rsm draw + a normal per
    // level during the search. The model.json lists every such float feature in
    // `features_info.float_features` (here surfaced as `feature_borders`), with an
    // EMPTY `borders` vec when none of its candidate borders were chosen. The prior
    // `!b.is_empty()` filter UNDER-COUNTED by dropping these unused-but-quantized
    // features (3 instead of 4 on the corpus), which short-changed the per-tree GTS
    // draw count and desynced the learnfold/leafval recalc seeds from tree 1 onward
    // (instrumented `cand_score_rng` shows 4 candidates/level). Count ALL listed
    // float features — each listed feature was a training candidate. A truly
    // constant feature is NOT listed by upstream, so it never inflates the count.
    let yetirank_n_candidate_features = feature_borders.len();
    let mut yetirank_seeder: Option<crate::YetiRankTreeSeeder> = if is_yetirank {
        Some(crate::YetiRankTreeSeeder::new_with_scoring(
            params.random_seed,
            yetirank_groups.len(),
            yetirank_n_candidate_features,
            params.depth,
            is_pairwise_scoring(&params.loss),
        ))
    } else {
        None
    };

    // StochasticRank per-tree RNG seeder (D-07 trainer-level closure, 06.3-18).
    // StochasticRank is the OTHER randomized listwise loss, but its noise model is
    // DISTINCT from YetiRank's pairwise re-sample: there are NO competitors — the
    // per-group Gaussian noise stream is re-seeded each `CalcDersForQueries` with
    // `recalc_seed + group_index` (`error_functions.h:1257`), where `recalc_seed`
    // is the trainer's per-tree `randomSeed` argument. The PRIOR Rust path passed
    // the FIXED `params.random_seed` to `compute_gradients_grouped` for EVERY tree,
    // which matched the standalone single-group self-oracle but DIVERGED from the
    // live trainer, whose persistent `LearnProgress->Rand(random_seed)` advances
    // per tree through the structure draw, the derivative-recalc seed, the per-level
    // split-search draws, the learning-fold seed and the leaf-value-recalc seed —
    // yielding TWO fresh base recalc seeds per tree (a DERIVATIVE recalc seed and a
    // LEAF-VALUE recalc seed; 10 base seeds across the 5-tree corpus). The per-tree
    // main-RNG consumption is IDENTICAL to YetiRank's (verified bit-exact against
    // the instrumented catboost 1.2.10 `stochasticrank_pertree_noise_groundtruth.jsonl`
    // — `YetiRankTreeSeeder::next_tree().recalc_seeds[0]` is the DERIVATIVE base and
    // `[2]` is the LEAF-VALUE base, both matching the GT cluster bases), so the SAME
    // seeder drives both losses. StochasticRank consumes the two BASE recalc seeds
    // directly (the per-group `+ group_index` offset is applied inside the grouped
    // der), NOT the per-group YetiRank query seeds (which carry an extra block layer).
    let is_stochasticrank = matches!(params.loss, Loss::StochasticRank { .. });
    let mut stochasticrank_seeder: Option<crate::YetiRankTreeSeeder> = if is_stochasticrank {
        // `group_count` only feeds the (unused-here) per-group YetiRank query-seed
        // derivation; StochasticRank consumes the raw `recalc_seeds` bases, so any
        // count yields identical bases. Pass the real group count for correctness.
        let group_count = group_spans.as_ref().map_or(0, Vec::len);
        Some(crate::YetiRankTreeSeeder::new(
            params.random_seed,
            group_count,
            yetirank_n_candidate_features,
            params.depth,
        ))
    } else {
        None
    };

    // N-dim approx buffer (D-6.2-01 / Plan 06.2-02). `approx_dimension` is the
    // number of output dimensions the loss produces. Every existing scalar loss
    // is single-dimension, so this is `1` until Plans 06.2-03..05 derive it per
    // loss (multiclass/multilabel/MultiQuantile). The approx is the
    // DIMENSION-MAJOR flat buffer `approx[d * n + i]` of length
    // `approx_dimension * n`, with one bias per dimension. At
    // `approx_dimension == 1` it is EXACTLY `vec![bias; n]` (the same slice,
    // same length, same summation order) — the D-04 byte-identity invariant
    // (RESEARCH Pitfall 1).
    // For the multilabel losses (MultiLogloss / MultiCrossEntropy) the approx
    // dimension is the label-set WIDTH `target.len() / n` (dim-major target,
    // `approx_dimension.cpp:22-23`), derived HERE because `loss_approx_dimension`
    // has no object count in scope. For every other loss it is the loss-derived
    // dimension (1 for scalar/binary; the distinct class count for multiclass).
    let approx_dimension: usize = if is_multilabel {
        target.len() / n
    } else {
        loss_approx_dimension(&params.loss, target)
    };

    // MULTILABEL per-dimension target-range validation (T-6.2-04a): MultiLogloss
    // labels must be `{0,1}`, MultiCrossEntropy probabilities `[0,1]`. Reject an
    // out-of-range label up front with a typed CbError (no `unwrap`/panic) rather
    // than feeding a poisoned der into the histogram/leaf reductions. The target is
    // dim-major `dim*n`; every entry is one label.
    if is_multilabel {
        let binary = matches!(params.loss, Loss::MultiLogloss);
        for &t in target {
            let ok = if binary {
                t == 0.0 || t == 1.0
            } else {
                t.is_finite() && (0.0..=1.0).contains(&t)
            };
            if !ok {
                let (name, range) = if binary {
                    ("MultiLogloss", "{0, 1}")
                } else {
                    ("MultiCrossEntropy", "[0, 1]")
                };
                return Err(CbError::OutOfRange(format!(
                    "{name} target label {t} is outside the admissible range {range}"
                )));
            }
        }
    }

    // MULTICLASS class-label remap (Pitfall 4, LOSS-02). The raw labels are mapped
    // to a contiguous `[0, k)` class index BEFORE training (`label_converter.cpp:142`)
    // so the softmax / one-vs-all der can write `der[target_class]` safely
    // (T-6.2-01); the `class_to_label` map is stored on the model to recover the
    // original labels at predict time. For the scalar / binary losses
    // `class_to_label` stays empty and `effective_target` is the raw target
    // (byte-identical).
    let is_multiclass = matches!(
        params.loss,
        Loss::MultiClass | Loss::MultiClassOneVsAll
    );
    let class_to_label: Vec<f64> = if is_multiclass {
        // Reject a non-finite (NaN/Inf) class label up front (WR-06): NaN makes
        // `partial_cmp` return None (treated as Equal -> a non-total sort order) and
        // `(NaN - NaN).abs() == 0.0` is false (so the dedup keeps duplicate NaNs),
        // yielding a silently corrupt `class_to_label` and therefore wrong predicted
        // labels. Surface it as a typed error instead, consistent with the
        // no-NaN-poisoning discipline on the custom-objective der path.
        if let Some(bad) = target.iter().copied().find(|l| !l.is_finite()) {
            return Err(CbError::OutOfRange(format!(
                "multiclass target contains a non-finite class label ({bad}); class \
                 labels must be finite for a total sort/dedup order"
            )));
        }
        build_class_remap(target)
    } else {
        Vec::new()
    };
    let remapped_target: Option<Vec<f64>> = if is_multiclass {
        Some(remap_target_to_class(target, &class_to_label)?)
    } else {
        None
    };
    // The target the boosting loop trains on: the remapped class index for
    // multiclass, else the raw target (unchanged for every scalar / binary loss).
    let target: &[f64] = remapped_target.as_deref().unwrap_or(target);

    let bias = starting_approx(params, target);
    // RMSEWithUncertainty (Wave B, LOSS-08 / D-6.4-04) starts from the per-dimension
    // optimal-constant `[mean, 0.5·log(var)]` REGARDLESS of `boost_from_average`
    // (`train_model.cpp:858`), unlike every other loss (single scalar `bias`). The
    // approx buffer is dim-major `[mean(0..n), log-scale(n..2n)]`; `Model.bias`
    // keeps dim-0's mean bias (the dim-1 log-scale bias lives only in the staged
    // approx, which the oracle compares — the predict path reconstructs it from the
    // staged buffer, not a stored per-dim bias).
    let mut approx = if matches!(params.loss, Loss::RmseWithUncertainty) {
        let dim_bias = rmse_uncertainty_starting_approx(target);
        let mut buf = vec![0.0_f64; approx_dimension * n];
        for d in 0..approx_dimension {
            let b = dim_bias.get(d).copied().unwrap_or(0.0);
            for i in 0..n {
                if let Some(slot) = buf.get_mut(d * n + i) {
                    *slot = b;
                }
            }
        }
        buf
    } else {
        vec![bias; approx_dimension * n]
    };

    // YetiRank LEARNING-fold approx (D-07, 06.3-14 ext): YetiRank is NOT
    // `UseAveragingFoldAsFoldZero` (usePairs is true — `learn_context.cpp:855`), so
    // the LEARNING fold (fold 0, drives the gradient + tree STRUCTURE) and the
    // AVERAGING fold (drives the stored model leaf VALUES) carry SEPARATE approxes
    // that diverge after tree 0. The structure search + deriv recalc read the
    // learning-fold approx; the leaf-value recalc reads the averaging-fold approx
    // (`approx`). `learn_approx` mirrors the learning-fold approx, updated each tree
    // by the learning-fold leaf-value recalc (`UpdateLearningFold`). For every
    // NON-YetiRank loss this buffer is unused (the single `approx` is correct).
    let mut learn_approx: Vec<f64> = if is_yetirank { approx.clone() } else { Vec::new() };

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

    // SPEC-OH-04 / SPEC-OH-05: partition the cat columns by encoding path, then
    // materialize the one-hot-routed ones into first-seen `PerfectHash` bin
    // columns plus their exact bin -> raw-hash inverse. On the numeric path
    // `cat_columns` is empty, so both lists and both tables are empty and the
    // matrix below is byte-identical to `FeatureMatrix::new` (SPEC-OH-31).
    //
    // CAT INGESTION (Plan 05-11): the cat-aware path computes per-cat-feature
    // OnLearnOnly cardinalities (`learn_set_cardinality` = calc_cat_feature_hash +
    // PerfectHash, NEVER a model's CTR hash map).
    let cat_cardinalities: Vec<u32> = cat_columns
        .iter()
        .map(|col| {
            let as_str: Vec<&str> = col.iter().map(String::as_str).collect();
            crate::candidates::learn_set_cardinality(&as_str)
        })
        .collect::<CbResult<Vec<u32>>>()?;
    let (one_hot_absolute, eligible_absolute) =
        partition_cat_columns(&cat_cardinalities, params.one_hot_max_size);

    // SPEC-OH-26 — a pool spanning BOTH encoding routes is typed-rejected.
    //
    // The level search has no three-way candidate union: `has_ctr` selects
    // `greedy_tensor_search_oblivious_with_ctr` (which takes no `cat_bins` and
    // therefore enumerates no one-hot candidates), otherwise the plain perturbed
    // arm runs (which sees no CTR columns). A mixed pool would silently take one
    // branch and drop the OTHER encoding's columns entirely — exactly the
    // class of bug this whole plan exists to fix. Device-side CTR co-existence is
    // deferred (SPEC §9 R12), so the honest gate ships instead of a silent drop.
    //
    // The gate lives HERE, where both partitions are in scope, so no future
    // dispatch arm can bypass it.
    if !one_hot_absolute.is_empty() && !eligible_absolute.is_empty() {
        return Err(CbError::Unsupported(format!(
            "training a pool with both one-hot-routed and CTR-routed categorical columns is \
             not yet supported (device-side CTR co-existence is deferred): raise \
             one_hot_max_size to route all columns one-hot, or lower it to route all columns \
             to CTR. At one_hot_max_size = {}, one-hot columns are {one_hot_absolute:?} and \
             CTR columns are {eligible_absolute:?}",
            params.one_hot_max_size,
        )));
    }

    let (one_hot_bins, one_hot_bin_to_hash) =
        build_one_hot_columns(cat_columns, &one_hot_absolute)?;

    // Training matrix: float columns plus the one-hot bin columns (empty on the
    // numeric path ⇒ `n_cat_features() == 0`, the pre-one-hot behaviour).
    let matrix = FeatureMatrix {
        feature_values,
        feature_borders,
        cat_bins: &one_hot_bins,
    };

    // FEAT-04 first-use / per-object penalty state (`feature_penalties_calcer.cpp`):
    // `used_features[f] == true` once any PRIOR tree in this run has split on float
    // feature `f`. While unused, the subtractive penalties fire; once used they go
    // to zero. Sized to the float-feature count and updated after each tree is
    // grown. With both penalty vectors empty the context is a no-op and this vector
    // is never consulted (the default oblivious path stays byte-identical, D-6.6-05).
    let penalties_active = !params.feature_weights.is_empty()
        || !params.first_feature_use_penalties.is_empty()
        || !params.per_object_feature_penalties.is_empty();
    let mut used_features: Vec<bool> = vec![false; matrix.n_features()];

    // Tensor / combination CTR candidate generation (ORD-05 / D-05, AddTreeCtrs,
    // greedy_tensor_search.cpp:491-551): emit the SimpleCtr / CombinationCtr
    // projections over the CTR-eligible cat features under the
    // `params.max_ctr_complexity` gate (:532-533). The numeric `train` /
    // `train_with_eval_sets` path supplies an EMPTY `cat_columns`, so the
    // cardinalities and candidate set are both empty and the float-only oracles
    // are byte-for-byte unchanged. `cat_cardinalities` / `eligible_absolute` were
    // computed above, alongside the SPEC-OH-04 one-hot partition.
    let ctr_candidates = tensor_ctr_candidates(
        &cat_cardinalities,
        params.one_hot_max_size,
        params.max_ctr_complexity,
    );

    // ORD-07: raw per-object categorical-bucket data for every CTR-eligible cat
    // feature (the phantom mixed float-partition + categorical-feature
    // projection's `max_bucket_count` contribution needs the RAW categorical
    // identity, not an online-CTR value — `cb_data::perfect_hash_bins` is the
    // SAME already-existing, already-oracle-tested hashing primitive
    // `learn_set_cardinality` above is built on, reused DIRECTLY here per
    // SPEC.md §7 rather than a new hand-rolled hashing loop). Empty for the
    // numeric path (`cat_columns` empty ⇒ `eligible_absolute` empty), a
    // provable no-op there.
    let cat_eligible_buckets: Vec<Vec<u32>> =
        cat_eligible_buckets_for(cat_columns, &eligible_absolute)?;

    // The TWO permutations for the cat-CTR two-materialization (research Q1/Q3),
    // now CARRYING the initial learn-set shuffle `S` in the averaging order (ORD-01
    // / bar (c), plan 05-19):
    //   * `cat_learn_permutation` — the STRUCTURE-search fold = the lone learning
    //     `Folds[0]` (`shuffle = foldIdx != 0`, `learn_context.cpp:526-529`).
    //     Upstream builds the folds on the ALREADY-S-SHUFFLED learn data
    //     (`ShuffleLearnDataIfNeeded` runs first), so Folds[0]'s "identity" is
    //     identity over shuffled data — `S` itself in ORIGINAL-object order.
    //     BUG-SFS was materializing this fold under the raw identity instead
    //     (`ctr_structure_fold_shuffle_test` pins the corrected borders). The
    //     actual per-fold permutations are built at `structure_fold_columns`
    //     below; this Option is the has-CTR presence gate.
    //   * `cat_averaging_permutation` — the AveragingFold's original-object CTR
    //     order `Q = [S[p] for p in P_avg]`
    //     ([`crate::averaging_ctr_permutation`]), where `S` is the initial
    //     learn-set shuffle (`ShuffleLearnDataIfNeeded`, `preprocess.cpp:183`) and
    //     `P_avg` is the averaging perm over the S-shuffled data — both off ONE
    //     persistent `random_seed` stream. This SUBSUMES the prior 05-17
    //     per-fold-`gen_rand` pre-draw hack (which matched the partition counts on a
    //     COMPENSATING wrong-perm+wrong-bins error). The LEAF-VALUE CTR column is
    //     materialized under THIS permutation (`train.cpp:130
    //     BuildIndices(AveragingFold)`).
    //
    // Feeding `Q` (original-object order) straight to `materialize_ctr_feature`
    // carries `S` WITHOUT a physical data shuffle/invert: the materialization order
    // is the only place `S` is observable for the leaf-VALUE partition (de-risk
    // gate `s_order_ctr_bins_oracle_test` proves this reproduces the self-consistent
    // bins bit-exact, pc=1 + pc=4). The structure search, numeric/one-hot/ordered
    // paths, and all per-object output order stay BYTE-IDENTICAL (no inversion
    // needed — the data is never moved).
    //
    // `need_shuffle` transcribes upstream `NeedShuffle` (`preprocess.cpp:161`):
    // CTRs present (any CTR-routed cat feature ⇒ non-empty candidates here) OR
    // ordered boosting, AND not time-ordered (`!has_time`). When it is FALSE
    // (e.g. a hypothetical `has_time=true` cat run) the averaging order falls back
    // to the plain unshuffled averaging permutation (no `S`).
    let need_shuffle = need_shuffle(
        !ctr_candidates.is_empty(),
        params.boosting_type,
        params.has_time,
    );
    let (cat_learn_permutation, cat_averaging_permutation): (Option<Vec<i32>>, Option<Vec<i32>>) =
        if ctr_candidates.is_empty() {
            (None, None)
        } else {
            let learning_folds =
                crate::learning_fold_count(params.permutation_count, /* needed = */ true);
            // STRUCTURE: the fold-0 order in ORIGINAL-object coordinates — `S`
            // when the learn set is shuffled (BUG-SFS), identity when time-ordered.
            // The materialization below rebuilds the per-fold permutations itself;
            // this value's role is the has-CTR presence gate (`.is_some()`).
            let learn: Vec<i32> = if need_shuffle {
                crate::create_shuffled_indices(n, params.random_seed)
            } else {
                (0..n as i32).collect()
            };
            // LEAF VALUES: the averaging-fold original-object CTR order.
            // `need_shuffle` (the normal cat path) ⇒ `Q = S ∘ P_avg` carries the
            // initial learn-set shuffle. The (time-ordered) `!need_shuffle` fallback
            // is the plain averaging perm with NO S — `P_avg` over UNshuffled data,
            // i.e. `permutations(n, learning_folds + 1, seed)[learning_folds]`.
            let averaging: Vec<i32> = if need_shuffle {
                crate::averaging_ctr_permutation(n, learning_folds, params.random_seed)
            } else {
                crate::permutations(n, learning_folds.saturating_add(1), params.random_seed)
                    .into_iter()
                    .nth(learning_folds)
                    .unwrap_or_else(|| (0..n as i32).collect())
            };
            (Some(learn), Some(averaging))
        };

    // The binclf target class per object (matching the e2e oracle binarization):
    // `target_class[i] = usize::from(target[i] > 0.5)`.
    let target_class: Vec<usize> = target.iter().map(|&t| usize::from(t > 0.5)).collect();

    // There is deliberately NO shared prior NUMERATOR here (E10 / SPEC-CTRT-10):
    // the numerator is resolved PER CANDIDATE — and, after E15, per (candidate,
    // prior) — inside `materialize_ctr_columns_for_perm`, because
    // `simple_ctr_priors` and `combinations_ctr_priors` are distinct lists. A
    // single hoisted `combinations_ctr_priors.first()` is exactly the bug that
    // made the combination prior govern simple candidates. The DENOMINATOR is the
    // constant [`CTR_PRIOR_DENOM`].
    let ctr_border_count = ctr_border_count_default();

    // `counter_calc_method` (E22 / SPEC-CTRT-17) — the FIRST read of
    // `params.counter_calc_method` in this file. Under `Full`, the eval sets'
    // categorical columns join the COUNTER bucket tally at both effect sites —
    // the online materialization (`CountOnlineCTRTotal` over the learn +
    // every-test-set hash array, `online_ctr.cpp:716-729`) and the final bake
    // (`online_ctr.cpp:956-960`). `counter_full_eval_columns[c]` is the
    // concatenation `eval[0].cat_columns[c] ++ eval[1].cat_columns[c] ++ …`,
    // matching `cat_columns`' absolute layout; EMPTY under `SkipTest` (the
    // default) or with no eval cat columns — byte-identical to the pre-E22
    // behavior.
    let counter_calc_skip_test =
        matches!(params.counter_calc_method, CounterCalcMethod::SkipTest);
    let counter_full_eval_columns: Vec<Vec<String>> = if counter_calc_skip_test {
        Vec::new()
    } else {
        let mut cols: Vec<Vec<String>> = vec![Vec::new(); cat_columns.len()];
        for es in eval_sets {
            for (c, col) in cols.iter_mut().enumerate() {
                if let Some(eval_col) = es.cat_columns.get(c) {
                    col.extend(eval_col.iter().cloned());
                }
            }
        }
        if cols.iter().all(Vec::is_empty) {
            Vec::new()
        } else {
            cols
        }
    };

    // Resolve the per-candidate ABSOLUTE projections ONCE (re-index the CTR-
    // eligible-position members emitted by `tensor_ctr_candidates` back to absolute
    // `cat_columns` indices). Both the structure (identity) and the leaf-value
    // (averaging) materializations share these projections.
    let absolute_projections: Vec<crate::TProjection> = ctr_candidates
        .iter()
        .map(|cand| {
            let absolute_members: Vec<usize> = cand
                .projection
                .cat_features()
                .iter()
                .filter_map(|&pos| eligible_absolute.get(pos).copied())
                .collect();
            crate::TProjection::from_features(&absolute_members)
        })
        .collect();

    // Per-iteration STRUCTURE-fold cycling (Task 4, ORD-01 / bar (c);
    // `takenFold = Folds[Rand.GenRand() % learning_folds]`, `train.cpp:208`).
    // Upstream selects the STRUCTURE learning fold per tree; cb-train previously
    // pinned the fixed identity `Folds[0]` for every tree. The structure CTR is
    // materialized under the SELECTED fold's permutation each iteration; the leaf
    // VALUES always stay on the fixed AveragingFold (Q, above).
    //
    // The learning-fold STRUCTURE permutations in ORIGINAL object order carry the
    // initial learn-set shuffle `S` exactly like the averaging order:
    //   * fold 0 = the IDENTITY `Folds[0]` (`shuffle = foldIdx != 0`) over the
    //     S-shuffled data, i.e. ORIGINAL order = `S` itself
    //     (`stream[0] == S`, so `[S[p] for p in stream[0]]` would double-apply S;
    //     fold 0's structure data is the unshuffled identity `[0..n]`);
    //   * fold j (1..learning_folds) = `[S[p] for p in stream[j]]`, where
    //     `stream = permutations(n, learning_folds + 1, seed)` is the SAME
    //     persistent stream `Q` came from (`stream[learning_folds]` is `P_avg`).
    //
    // For `learning_folds == 1` (pc=1 / pc=2) there is only fold 0 (identity), so
    // `% 1 == 0` always picks it and this is BYTE-IDENTICAL to the prior fixed
    // `Folds[0]` materialization (regression anchor).
    let learning_folds_for_cycle =
        crate::learning_fold_count(params.permutation_count, !ctr_candidates.is_empty());
    // `structure_fold_columns[fold]` is the per-candidate structure CTR column set
    // for learning fold `fold` (index 0..learning_folds). Built once (the fold
    // permutations are fixed for the run); the per-iteration loop selects among them.
    let structure_fold_columns: Vec<Vec<crate::ctr::CtrFeatureColumn>> = if cat_learn_permutation
        .is_some()
    {
        let stream = if need_shuffle {
            crate::permutations(
                n,
                learning_folds_for_cycle.saturating_add(1),
                params.random_seed,
            )
        } else {
            Vec::new()
        };
        let s = if need_shuffle {
            crate::create_shuffled_indices(n, params.random_seed)
        } else {
            (0..n as i32).collect()
        };
        let mut per_fold = Vec::with_capacity(learning_folds_for_cycle);
        for fold in 0..learning_folds_for_cycle {
            // fold 0: identity over the S-SHUFFLED data (`shuffle = foldIdx != 0`,
            // learn_context.cpp:526-529, on the ShuffleLearnDataIfNeeded output)
            // ⇒ ORIGINAL-object order = S itself.
            // fold j>0: original-object order = [S[p] for p in stream[j]].
            let perm: Vec<i32> = if !need_shuffle {
                (0..n as i32).collect()
            } else if fold == 0 {
                s.clone()
            } else {
                stream
                    .get(fold)
                    .map(|p_fold| {
                        p_fold
                            .iter()
                            .enumerate()
                            .map(|(k, &p)| s.get(p as usize).copied().unwrap_or(k as i32))
                            .collect()
                    })
                    .unwrap_or_else(|| (0..n as i32).collect())
            };
            per_fold.push(materialize_ctr_columns_for_perm(
                cat_columns,
                &absolute_projections,
                &ctr_candidates,
                params,
                &perm,
                &target_class,
                ctr_border_count,
                &counter_full_eval_columns,
            )?);
        }
        per_fold
    } else {
        Vec::new()
    };
    // The iteration-0 structure columns (fold 0 = identity), kept as the default
    // `materialized_ctr_features` so the `has_ctr` gate and any non-cycling read
    // sees the same shape as before (byte-identical for learning_folds == 1).
    let materialized_ctr_features: Vec<crate::ctr::CtrFeatureColumn> = structure_fold_columns
        .first()
        .cloned()
        .unwrap_or_default();

    // Materialize the SECOND (LEAF-VALUE) combined-projection online CTR feature
    // column PER candidate under the AVERAGING-fold's SHUFFLED permutation
    // (research Q3 #2: `materialize_ctr_feature(..., averaging_perm, ...)` — the
    // SAME function, the AVERAGING permutation input). For the tensor_ctr_e2e
    // config these bins yield the leaf-VALUE partition [6,0,7,17] (vs the structure
    // [6,0,9,15]). Index-aligned with `materialized_ctr_features` (same projection
    // order), so a chosen structure CTR split maps to the same averaging column.
    let averaging_ctr_features: Vec<crate::ctr::CtrFeatureColumn> =
        if let Some(avg_perm) = cat_averaging_permutation.as_deref() {
            materialize_ctr_columns_for_perm(
                cat_columns,
                &absolute_projections,
                &ctr_candidates,
                params,
                avg_perm,
                &target_class,
                ctr_border_count,
                &counter_full_eval_columns,
            )?
        } else {
            Vec::new()
        };

    let n_leaves = 1usize << params.depth;
    let mut trees: Vec<ObliviousTree> = Vec::with_capacity(params.iterations);
    // FEAT-06 / D-6.6-04: non-symmetric (Lossguide / Depthwise) trees accumulate here
    // when `grow_policy` selects a leaf-wise grower. Empty for every oblivious model.
    let mut non_symmetric_trees: Vec<NonSymmetricTree> = Vec::new();
    // GPUT-18 / D-03a: Region PATH trees accumulate here under `grow_policy=Region`.
    // Empty for every oblivious / non-symmetric model (the grower + push land in the
    // Region dispatch arm below).
    let mut region_trees: Vec<RegionTree> = Vec::new();

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
    // The eval-set machinery below is SINGLE-DIMENSION end to end: `eval_approx`
    // holds one `f64` per eval object (no `approx_dimension` factor, unlike the
    // learn-side `approx`), the per-tree contribution reads `leaf_values[leaf]`
    // (dimension 0 only, since the buffer is dimension-major
    // `leaf_values[d * n_leaves + l]`), and `EvalMetric::eval` requires
    // `approx.len() == target.len()`, which has no multi-dimensional reading.
    //
    // Running a multi-dimensional loss against an eval set therefore produced a
    // metric curve computed from dimension 0's raw scores alone — and that curve
    // drives `use_best_model`'s truncation and the overfitting detector's stop.
    // A silently-wrong stopping decision is worse than a refused one, so reject
    // the combination outright rather than reporting a meaningless curve. (The
    // in-scope multiclass fixtures pin `od_type=None` with NO eval set, which is
    // why this was never caught — see `EvalMetric::for_loss`.)
    if has_test && approx_dimension > 1 {
        return Err(CbError::Unsupported(format!(
            "eval sets are not supported for a {approx_dimension}-dimensional loss \
             ({:?}): the validation metric surface is single-dimension, so the \
             per-iteration curve — and any `use_best_model` / overfitting-detector \
             decision taken from it — would be computed from output dimension 0 \
             alone. Fit without an `eval_set`, or use a scalar loss.",
            params.loss
        )));
    }
    // `EvalMetric` is no longer `Copy` (LOSS-07 — the `Custom` variant carries a
    // non-`Copy` `Arc`); clone out of the borrowed `params` (cheap — an `Arc`
    // refcount bump for `Custom`, a bitwise copy otherwise).
    let eval_metric = params
        .eval_metric
        .clone()
        .unwrap_or_else(|| EvalMetric::for_loss(&params.loss));
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

    // SPEC-OH-27 (T01b, branch b) — one-hot x ACTIVE RNG draws is typed-rejected.
    //
    // Upstream charges one unconditional `GenRandReal1()` per candidate sub-list,
    // and `AddOneHotFeatures` contributes one sub-list per one-hot-routed cat
    // column, so the per-level draw count becomes `n_float + n_one_hot`. That rule
    // is SOURCE-DERIVED with HIGH confidence for the un-bundled `OneFeature` path
    // — but `CompressCandidates` runs BETWEEN `AddOneHotFeatures` and the draw
    // site and can re-bundle those candidates into `BinarySplits` /
    // `ExclusiveBundle` / `FeaturesGroup` ensembles whose draw arithmetic DIFFERS,
    // and a cardinality-2 categorical column is exactly the shape most likely to
    // be packed. That case is NOT ESTABLISHED (see
    // `.planning/plans/one-hot-categorical-training/instrumented-ground-truth/ONE_HOT_GROUND_TRUTH.md`).
    //
    // Consuming the un-bundled rule regardless would desynchronise every
    // subsequent tree's bootstrap sample with no visible symptom on non-bootstrap
    // tests — the exact defect class fixed in `d7676b5`. So the combination is
    // refused until an instrumented upstream run settles it. The gate lives HERE,
    // where both the one-hot column list and `draws_active` are in scope, so no
    // downstream dispatch arm can bypass it.
    //
    // The DEFAULT path is unaffected: `bootstrap_type = No` and
    // `random_strength = 0` are both draw-inert, so one-hot training works out of
    // the box; only an explicit opt-in to draws is refused.
    if !one_hot_absolute.is_empty() && draws_active {
        return Err(CbError::Unsupported(format!(
            "one-hot categorical training is not supported with bootstrap_type != No or \
             random_strength != 0 (got bootstrap_type = {:?}, random_strength = {}); the \
             upstream per-level RNG draw accounting for one-hot candidates under \
             CompressCandidates has not been established (see \
             .planning/plans/one-hot-categorical-training/instrumented-ground-truth/ONE_HOT_GROUND_TRUTH.md). \
             {} one-hot-routed cat column(s): {one_hot_absolute:?}",
            params.bootstrap_type,
            params.random_strength,
            one_hot_absolute.len(),
        )));
    }
    // MVS lambda for trees after the first uses the previous tree's mean leaf L2
    // norm (`CalculateLastIterMeanLeafValue`); `None` on the first tree.
    let mut prev_leaf_mean_l2: Option<f64> = None;

    // Per-iteration STRUCTURE-fold cycle (Task 4): `Folds[GenRand() %
    // learning_folds]` each tree (`train.cpp:208`). For learning_folds==1 (pc=1/2)
    // this is all-zeros (byte-identical fixed Folds[0]); for the pc=4/seed=0
    // production default it is the instrument-derived `[0,2,0,2,2]`. Only consulted
    // on the CTR path (where `structure_fold_columns` is non-empty); the
    // numeric/one-hot/ordered paths ignore it.
    let struct_fold_cycle =
        structure_fold_cycle(params.permutation_count, params.iterations, params.random_seed);

    // ---------------------------------------------------------------------------
    // GPUT-01 DEVICE GROW SEAM — per-fit all-or-nothing decision (D-10-01). Decide
    // ONCE, before the tree loop, whether this WHOLE fit runs on the device grower
    // or the byte-unchanged CPU grower (D-04). Two gates compose:
    //
    //   1. Host eligibility (`device_host_eligible`): excludes every config the
    //      depth-1 device grower does NOT implement and that the backend seam
    //      cannot see (ranking / ordered / CTR / penalties / monotone / multi-dim /
    //      sampling / perturbation / eval-set). Those stay on the CPU path.
    //   2. Backend coverage (`begin_device_training`): the finer gate the backend
    //      owns (depth==1 / RMSE|Logloss|CrossEntropy / Plain / fold_count==1 /
    //      supported score fn), returning Ok(false) to decline. On the default
    //      CpuBackend this is ALWAYS Ok(false), so the CPU grower runs unchanged and
    //      the per-iteration device branch below is inert (the D-04 invariant the
    //      `cargo test -p cb-train` suite verifies).
    //
    // The decision is made ONCE here (not per tree): Ok(true) commits the whole fit
    // to the device; a later Ok(None) from a covered fit is a mid-run mix and is
    // rejected (T-10-23), never silently backfilled with a CPU tree.
    let device_host_eligible = group_spans.is_none()
        // FPP-20 (T23): the former `ordered_learning_perm.is_none()` clause is GONE. It was
        // the host half of the ordered decline — the session-side gate was the other half —
        // and it existed because the device had no way to score a candidate over the fold's
        // body/tail segments. It does now (`grow_oblivious_tree_ordered_resident`), and the
        // per-fit `DeviceOrderedConfig` built below carries the permutation + segment table
        // the grow needs.
        //
        // Removing it does NOT make every ordered fit device-eligible: `map_ordered_coverage`
        // still requires a covered simple-approximant loss, depth >= 1, a single fold,
        // SymmetricTree and every other family flag at its default, and `begin` additionally
        // requires the descriptor to be present. An uncovered ordered fit still falls back to
        // the byte-unchanged CPU grower.
        // GDC-11 (T14): the CTR clauses are RELAXED — a single-permutation
        // (`learning_folds_for_cycle == 1`, made real by GDC-01) fit whose
        // materialized CTR columns are all device-covered (simple Borders
        // projections, see `ctr_types_are_device_covered`) may now commit to the
        // device; the session materializes BOTH permutations (GDC-09) and gathers
        // leaf values over the averaging bins (GDC-10). Anything else — multi-
        // permutation, combination projections, non-Borders types, one-hot × CTR
        // — still falls back to the byte-unchanged CPU path (D-04). The Ordered
        // clause above is deliberately UNTOUCHED (D5).
        && (
            (materialized_ctr_features.is_empty()
                && structure_fold_columns.iter().all(Vec::is_empty))
            || (learning_folds_for_cycle == 1
                && one_hot_bins.is_empty()
                && ctr_types_are_device_covered(&materialized_ctr_features))
        )
        && !penalties_active
        && params.monotone_constraints.is_empty()
        // Phase 12 Plan 03/04 (GPUT-18): SymmetricTree (oblivious, Plan 01), the two
        // non-symmetric leaf-wise policies Depthwise / Lossguide (Plan 03), AND Region (Plan 04,
        // the walk-until-diverge device PATH grow) are all device-eligible.
        && matches!(
            params.grow_policy,
            EGrowPolicy::SymmetricTree
                | EGrowPolicy::Depthwise
                | EGrowPolicy::Lossguide
                | EGrowPolicy::Region
        )
        && approx_dimension == 1
        && !is_multiclass
        && !is_multilabel
        // WR-01 (`WR01-S9`): the three parity-target bootstrap types are now device-eligible
        // via HOST sampling (Design A — `bootstrap()` runs here, only the per-object
        // multiplier crosses the seam). They are admitted ONLY for the oblivious
        // (SymmetricTree) grow: the non-symmetric / Region / CTR / exact-leaf × sampling
        // combinations are out of scope this phase and the backend session declines them
        // explicitly rather than dropping the sample.
        //
        // POISSON is admitted here too, but on a DIFFERENT footing and only for the
        // oblivious grow. It is upstream's GPU-ONLY sampler — `TBootstrapConfig::Validate`
        // rejects it on the CPU task type outright ("poisson bootstrap is not supported on
        // CPU") — so it has no CPU sampler to run host-side and `bootstrap()` still refuses
        // it. Instead the device draws it RESIDENT, from a verbatim transcription of
        // upstream's CUDA `PoissonBootstrapImpl` (`cb_backend::kernels::bootstrap_device`,
        // gated bit-for-bit against the `bootstrap_poisson/` upstream fixtures). If the
        // device does not actually commit, Poisson is rejected below rather than silently
        // falling back to a CPU grower that cannot express it.
        //
        // FPP-13 (T11): the `&& grow_policy == SymmetricTree` restriction on the three
        // HOST-sampled types is GONE. It existed because the Region and non-symmetric
        // growers ignored the per-object multiplier — the backend declined those
        // combinations rather than silently dropping the sample. FPP-12 gave both growers
        // real SPLIT-SCORING channels, so a sampled Depthwise / Lossguide / Region fit now
        // scores over `der1 * sample` while leaf estimation stays unsampled, exactly as
        // the oblivious arm does.
        //
        // POISSON keeps the SymmetricTree restriction. It is not host-sampled at all: it
        // is DEVICE-resident (upstream has no CPU Poisson sampler to mirror), and only the
        // oblivious arm opens the resident sampler. Admitting it for a non-symmetric grow
        // would commit a fit the session then declines.
        && (matches!(params.bootstrap_type, EBootstrapType::No)
            || matches!(
                params.bootstrap_type,
                EBootstrapType::Bayesian | EBootstrapType::Bernoulli | EBootstrapType::Mvs
            )
            || (matches!(params.bootstrap_type, EBootstrapType::Poisson)
                && matches!(params.grow_policy, EGrowPolicy::SymmetricTree)))
        && params.random_strength == 0.0
        && eval_sets.is_empty()
        // SPEC-OH-20 (T23): "has something to score" is float OR one-hot, not float
        // alone. A pool routed entirely one-hot has zero float columns; the old
        // `matrix.n_features() > 0` made SPEC-OH-20's 0-float target unreachable.
        //
        // Two things this clause deliberately does NOT do:
        //   * It does NOT lift clause 3 (`materialized_ctr_features.is_empty() &&
        //     structure_fold_columns.iter().all(Vec::is_empty)`) — one-hot × CTR stays
        //     off the device (SPEC §9 R12, SPEC-OH-26 rejects the mixed pool outright).
        //   * It is NOT the only place a 0-float pool is decided. The backend session
        //     ALSO declines on `n == 0 || n_features == 0 || n_bins == 0`
        //     (`cb-backend/src/gpu_runtime/session.rs`, the `begin` preamble) and then
        //     pads the histogram line with `pad_hist_line_bins(n_bins)`. Under T24's
        //     concatenated axis `n_features` is the TOTAL (`n_float + n_cat`) and
        //     `n_bins = max(float n_bins, max cat cardinality)`, so a cat-only pool
        //     passes both; a cardinality-2 column pads to a legal `n_bins_line == 32`.
        && has_any_scorable_feature(&matrix)
        // SPEC §9 R10 (T24): bound the one-hot cardinality on the device OR FALL BACK.
        // Falling back is expressed HERE, as an eligibility clause, rather than as an
        // error out of the quantizer — an over-wide column must train correctly on the
        // CPU grower, not abort an otherwise valid fit. Inert for a float-only pool
        // (an empty cardinality list trivially fits), so SPEC-OH-31 is unaffected.
        && one_hot_cardinalities_fit_the_device(
            &one_hot_bin_to_hash
                .iter()
                .map(|h| u32::try_from(h.len()).unwrap_or(u32::MAX))
                .collect::<Vec<_>>(),
        )
        // GDC-05: the former WR-03 weight-uniformity clause is GONE. The device
        // grow paths now consume `w·der1` (upstream's `SumWeightedDelta`) in the
        // split histogram AND the leaf estimate on every covered grow policy —
        // the oblivious resident grow (GDC-02, `weighted_der1_h`), and the
        // host-driven nonsym / Region growers (GDC-03/04, `host_weighted_der1`)
        // — so a genuinely weighted pool meets the ≤1e-5 upstream bar on device.
        //
        // FPP-02 (T09): the former CR-01 `bias == 0.0` clause is GONE too. It existed
        // because `GpuTrainSession::begin` seeded its resident approx to a hardcoded
        // `vec![0.0; n]`, so a non-zero starting approximant would have trained against a
        // wrong starting point. FPP-01 replaced that seed with `config.bias` (populated
        // just below from this same `starting_approx`), so a `boost_from_average = true`
        // fit — upstream's RMSE default, and this project's `CatBoostBuilder` default —
        // now reaches the device with the CORRECT first-tree derivative.
        //
        // Only the OBLIVIOUS resident arm reads the seed; the Region / non-symmetric /
        // exact-leaf arms re-derive `der1` from the CALLER's `approx` via `host_der1`, and
        // the caller's `approx` already starts at the bias — so every covered grow policy
        // is correct under this relaxation, not just SymmetricTree.
        // CR-02: the device grower computes leaf values via `calc_average` (the
        // Gradient/Simple formula); it has no Newton arm. For RMSE Gradient==Newton
        // coincide, but for Logloss/CrossEntropy the Newton formula diverges, so a
        // Newton request on a device-covered loss falls back to the CPU grower.
        //
        // FPP-06 (T10): `LeafMethod::Exact` is additionally admitted for {Mae, Quantile}
        // — the intersection derived in `device_exact_leaf_config`, whose doc comment
        // explains why neither `validate_leaf_method`'s set nor `map_leaf_method`'s set
        // is the right condition alone (LogCosh is CPU-legal but device-uncovered; Mape is
        // device-covered but CPU-rejected; MultiQuantile is multi-dimensional). The
        // condition is read back off the already-computed config decision rather than
        // re-derived here, so the gate and the config can never disagree — a gate that
        // opened for a pair the config declined would apply the Gradient calc_average leaf
        // to a Quantile fit.
        && (matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple)
            || device_exact_leaf_config(params.leaf_method, &params.loss).0);

    // The leaf-regularization is constant across the fit for the device-eligible
    // config (fixed weights / n, no per-tree sampling), so it is computed ONCE and
    // handed to `begin`, matching the CPU per-tree
    // `scale_l2_reg(l2, sumAllWeights, n)`.
    let device_scaled_l2 = scale_l2_reg(params.l2_leaf_reg, sum_all_weights, n);
    // Phase 12 Plan 03 (GPUT-18 / Open Q2 promotion): the device grow-policy mapping.
    // (Hoisted above the quantize step so the QPACK-01 raw-channel gate below can read
    // it — a pure function of `params`, so the hoist is order-invariant.)
    let device_grow_policy = match params.grow_policy {
        EGrowPolicy::Depthwise => DeviceGrowPolicy::Depthwise,
        EGrowPolicy::Lossguide => DeviceGrowPolicy::Lossguide,
        // Phase 12 Plan 04 (GPUT-18): Region routes to the host-driven device Region PATH grow.
        EGrowPolicy::Region => DeviceGrowPolicy::Region,
        // SymmetricTree (and any policy that reached here) → the oblivious covered regime.
        _ => DeviceGrowPolicy::SymmetricTree,
    };
    // QPACK-01: the raw float channel — the backend quantizes AND packs the cindex ON
    // DEVICE, so the host bin matrix (an `n * nf` u32 buffer, ~200MB at 1M×50) never
    // exists. Offered exactly when every session arm that would need host bins is
    // absent: float-only (no one-hot), no CTR, and the oblivious SymmetricTree family
    // (non-symmetric / Region keep host bin copies in their grow state). The session
    // independently re-checks and declines to `false`, after which the host-quantize
    // channel below runs unchanged — never a wrong device result, never a lost fit.
    let raw_device_channel = device_host_eligible
        && one_hot_bins.is_empty()
        && materialized_ctr_features.is_empty()
        && matches!(device_grow_policy, DeviceGrowPolicy::SymmetricTree);
    // SPEC-OH-21 (T24): ONE device-quantize entry, on EVERY device-eligible pool. A
    // float-only pool passes an empty `cat_bins` slice and still gets a fully populated
    // `real_folds` (`[borders[f].len() + 1, …]`), which is what lets the session's
    // `real_folds.len() == eff_n_features` check stay unconditional. Routing float-only
    // fits back through the 2-tuple `quantize_feature_major` would leave nothing
    // producing `real_folds` and break every existing float-only device oracle.
    //
    // QPACK-01 carve-out: on the raw channel the bins are NOT quantized here — only the
    // metadata (`n_bins`, `real_folds`) is derived, by the SAME formulas the quantizer
    // uses on a float-only pool (`n_bins = max_f(borders_f + 1).max(1)`, `real_folds[f]
    // = borders_f + 1`), so the session sees identical scalars on either channel. The
    // host quantize still runs lazily below if the raw attempt declines.
    let (device_bins, device_n_bins, device_real_folds) = if device_host_eligible {
        if raw_device_channel {
            let real_folds: Vec<u32> = feature_borders
                .iter()
                .map(|b| u32::try_from(b.len() + 1).unwrap_or(u32::MAX))
                .collect();
            let n_bins = feature_borders
                .iter()
                .fold(0usize, |acc, b| acc.max(b.len() + 1))
                .max(1);
            (Vec::new(), n_bins, real_folds)
        } else {
            let prof = std::env::var_os("CB_GPU_PROF").is_some_and(|v| v != "0");
            let prof_t = std::time::Instant::now();
            let out = quantize_feature_major_with_one_hot(
                feature_values,
                feature_borders,
                &one_hot_bins,
                n,
            );
            if prof {
                eprintln!(
                    "CB_GPU_PROF quantize n={n} nf={} n_one_hot={} elapsed={:.2}ms",
                    feature_values.len(),
                    one_hot_bins.len(),
                    prof_t.elapsed().as_secs_f64() * 1e3,
                );
            }
            out
        }
    } else {
        (Vec::new(), 0, Vec::new())
    };
    // The device feature axis is `float | one-hot` (T24's layout), so the total width
    // and the one-hot boundary are both derived here and travel together.
    let device_n_float = matrix.n_features();
    let device_n_features = device_n_float + one_hot_bins.len();
    let device_one_hot_flags: Vec<bool> = (0..device_n_features)
        .map(|f| f >= device_n_float)
        .collect();
    // Phase 12 Plan 03 (GPUT-18 / Open Q2 promotion): build the plain-host DeviceTrainConfig
    // from `params` so the grow-policy (+ Lossguide leaf cap / min-data) reaches the session
    // gate. SymmetricTree yields `DeviceTrainConfig::default()` (byte-unchanged, D-04); the two
    // non-symmetric policies flip the device non-sym arm on. Every OTHER family knob stays at
    // its default covered value (host eligibility already excludes sampling / exact / CTR).
    // (`device_grow_policy` itself is mapped above the quantize step — QPACK-01.)
    //
    // GDC-11 (T14): the two-permutation device CTR config. Populated ONLY for a
    // host-eligible CTR fit (the relaxed clause above already vetted single-
    // permutation + simple Borders columns); every other fit keeps `ctr: None`
    // (byte-unchanged, D-04). The backend's `ctr_covered` gate independently
    // re-checks the shape (borders+1 == n_bins, averaging present) and declines
    // to CPU on any mismatch.
    let device_ctr = if device_host_eligible && !materialized_ctr_features.is_empty() {
        match (
            cat_learn_permutation.as_deref(),
            cat_averaging_permutation.as_deref(),
        ) {
            (Some(learn), Some(avg)) => Some(build_device_ctr_config(
                &materialized_ctr_features,
                &averaging_ctr_features,
                learn,
                avg,
                &target_class,
                &cat_eligible_buckets,
                &eligible_absolute,
                ctr_border_count,
            )?),
            _ => None,
        }
    } else {
        None
    };
    // FPP-05 (T06): the {Mae, Quantile} × Exact intersection, decided once as a pure
    // function so it is unit-testable without a device (PLAN blocker B-2, option (a)).
    let device_exact_leaf = device_exact_leaf_config(params.leaf_method, &params.loss);
    let device_config = DeviceTrainConfig {
        // FPP-01/FPP-02 (T09): the fit's REAL starting approximant. `0.0` — every
        // `boost_from_average = false` fit, which is every fixture that was device-eligible
        // before this phase — reproduces the former hardcoded resident seed byte-for-byte
        // (D-04), so no existing device test changes behaviour.
        bias,
        ctr: device_ctr,
        grow_policy: device_grow_policy,
        // The Lossguide cap is meaningful ONLY for the non-sym leaf-wise policy; leave it
        // `None` for SymmetricTree so `is_covered_regime()` (Plan 01) stays satisfied.
        max_leaves: if matches!(params.grow_policy, EGrowPolicy::Lossguide) {
            Some(params.max_leaves)
        } else {
            None
        },
        min_data_in_leaf: params.min_data_in_leaf,
        // ─── WR-01 WIRED: the bootstrap family knobs ────────────────────────────────
        // `bootstrap_type` is now threaded from `params` and `sample_from_host` declares
        // that the HOST computes the per-tree sample (Design A / `[DECISION D4]`). The two
        // fields travel together: the backend session reads `sample_from_host` to decide
        // that it must NOT open its own device-resident sampler, and reads
        // `bootstrap_type` only as bookkeeping describing WHICH host sampler ran.
        //
        // POISSON inverts that: it is the one arm the DEVICE samples (upstream has no CPU
        // Poisson sampler to mirror), so it travels with `sample_from_host = false` and the
        // session opens its own resident sampler — which is why `sample_rate` and `rng_seed`
        // below are wired for it and inert for everything else.
        bootstrap_type: match params.bootstrap_type {
            EBootstrapType::Bayesian => DeviceBootstrapType::Bayesian,
            EBootstrapType::Bernoulli => DeviceBootstrapType::Bernoulli,
            EBootstrapType::Mvs => DeviceBootstrapType::Mvs,
            EBootstrapType::Poisson => DeviceBootstrapType::Poisson,
            // `No` keeps the no-subsampling covered default.
            _ => DeviceBootstrapType::No,
        },
        sample_from_host: !matches!(
            params.bootstrap_type,
            EBootstrapType::No | EBootstrapType::Poisson
        ),
        // Read by the device-resident sampler ONLY, i.e. only on the Poisson arm. λ is
        // derived from `sample_rate` inside the kernel wrapper through upstream's
        // `GetPoissonLambda() = -log(1 - subsample)`; the seed buffer is built once per fit
        // from `rng_seed`. For every host-sampled arm these stay inert (the sample already
        // crossed the seam fully formed), which is why they are set unconditionally rather
        // than being made Poisson-only — an inert value cannot mislead, a missing one can.
        sample_rate: params.subsample as f32,
        rng_seed: params.random_seed,
        // STILL NOT WIRED (deliberately, at `DeviceTrainConfig::default()`):
        // `mvs_lambda`, `sample_rate` and `rng_seed` are the DEVICE-RESIDENT sampler's
        // inputs, and Design A never opens that sampler — λ, the subsample rate and the
        // RNG stream all live host-side inside `bootstrap()` above, which is the
        // ≤1e-5-verified sampler and the reason upstream parity is reachable. Setting
        // them here would be inert at best and, for `mvs_lambda`, actively misleading.
        // `ctr` likewise stays default: CTR × sampling is out of scope (SPEC §2) and the
        // session declines that combination.
        // Design B′ (device-resident sampling) is the perf follow-up that would wire them.
        //
        // ─── FPP-05: the device Exact order-statistic leaf ──────────────────────────
        // Activated ONLY for the {Mae, Quantile} intersection — see
        // `device_exact_leaf_config`'s doc comment for the derivation and for why
        // neither `validate_leaf_method`'s set nor `map_leaf_method`'s set is the right
        // condition alone. This is a config-only change: `device_host_eligible` still
        // rejects `LeafMethod::Exact` until T10 relaxes it, so today the config is built
        // but never reaches `begin`. That ORDER is mandatory — relaxing the gate without
        // the config would silently apply the Gradient `calc_average` leaf to a Quantile
        // fit, which is wrong and worse than the current correct CPU fallback.
        exact_leaf: device_exact_leaf.0,
        quantile_alpha: device_exact_leaf.1,
        quantile_delta: device_exact_leaf.2,
        //
        // ─── SPEC-OH-21/22/24/25: the one-hot channel ───────────────────────────────
        // All three travel together and describe the SAME concatenated `float | one-hot`
        // device feature axis. On a float-only pool `one_hot_flags` is all-`false`,
        // `n_float == n_features`, and `real_folds` is the per-float `borders + 1` — the
        // scorer then only ever takes the `one_hot == false` arm whose eligibility is the
        // unchanged `border < max_border`, so `real_folds` is uploaded but never read and
        // the float path is numerically unchanged (SPEC-OH-31).
        //
        // `real_folds` is NOT `TCFeature.folds` (the padded line width): see the field doc
        // on `DeviceTrainConfig`.
        one_hot_flags: device_one_hot_flags.clone(),
        real_folds: device_real_folds.clone(),
        n_float: device_n_float,
        // FPP-20 (T23): the per-fit ordered descriptor. `Some` only on an Ordered fit that
        // actually built a learning fold; `None` keeps every Plain fit byte-unchanged (D-04).
        //
        // Everything here is a pure function of `(n, fold_len_multiplier, weights)` plus the
        // learning permutation, i.e. per-FIT constant, which is why it rides the config rather
        // than the per-tree grow seam. `segment_tail_finish` carries ONLY `tail_finish`:
        // `ordered_segment_leaf_stats` discards `body_finish` (tree.rs), so a segment is the
        // permutation PREFIX `[0, tail_finish)`. `body_finish` survives solely inside
        // `scale_l2_reg`, which is exactly where it is applied below.
        ordered: ordered_learning_perm.as_ref().map(|perm| {
            let segments = crate::fold::body_tail_segments(n, params.fold_len_multiplier);
            let seg_body_sum_weights =
                crate::fold::body_sum_weights(n, params.fold_len_multiplier, &weights);
            cb_compute::DeviceOrderedConfig {
                // The device index type is u32; a negative permutation entry is a
                // fold-construction bug, and mapping it to 0 here would silently train on a
                // duplicated object. Clamp is not appropriate, so an out-of-range entry is left
                // to the session's host-side validation, which rejects `>= n` before upload.
                permutation: perm.iter().map(|&p| p as u32).collect(),
                segment_tail_finish: segments.iter().map(|&(_, tail)| tail).collect(),
                segment_scaled_l2: segments
                    .iter()
                    .enumerate()
                    .map(|(idx, &(body_finish, _))| {
                        let bsw = seg_body_sum_weights.get(idx).copied().unwrap_or(0.0);
                        cb_compute::scale_l2_reg(params.l2_leaf_reg, bsw, body_finish)
                    })
                    .collect(),
            }
        }),
        ..DeviceTrainConfig::default()
    };
    let device_active = if device_host_eligible && device_n_bins > 0 {
        // QPACK-01: offer the RAW float channel first (device-side quantize+pack — no
        // host bin matrix). A `false` return is a coverage decline, NOT an error: fall
        // through to the host-quantize channel, exactly as if the raw channel never
        // existed. The lazy quantize below therefore runs only on that decline path.
        let raw_opened = if raw_device_channel {
            runtime.begin_device_training_raw(
                &params.loss,
                params.depth,
                matches!(params.boosting_type, EBoostingType::Plain),
                learning_folds_for_cycle,
                params.score_function,
                feature_values,
                feature_borders,
                &weights,
                n,
                device_n_features,
                device_n_bins,
                learning_rate,
                device_scaled_l2,
                &device_config,
            )?
        } else {
            false
        };
        if raw_opened {
            true
        } else {
            // The host-quantize channel. On the raw-decline path the bins were never
            // quantized above — do it now (identical output to the eager path).
            let host_bins: std::borrow::Cow<'_, [u32]> = if raw_device_channel {
                std::borrow::Cow::Owned(
                    quantize_feature_major_with_one_hot(
                        feature_values,
                        feature_borders,
                        &one_hot_bins,
                        n,
                    )
                    .0,
                )
            } else {
                std::borrow::Cow::Borrowed(&device_bins)
            };
            runtime.begin_device_training(
                &params.loss,
                params.depth,
                matches!(params.boosting_type, EBoostingType::Plain),
                // GDC-01: the REAL learning-fold count, not a literal 1. For every
                // non-CTR fit `learning_fold_count(pc, false) == 1` (byte-unchanged,
                // D-04); once the CTR clause admits fits, `ctr_covered`'s
                // `fold_count != 1` decline (session.rs) becomes load-bearing and a
                // multi-permutation CTR fit can never silently ride fold-0 columns.
                learning_folds_for_cycle,
                params.score_function,
                &host_bins,
                &weights,
                n,
                // The device feature axis is the CONCATENATED `float | one-hot` width
                // (SPEC-OH-21), not the float count — equal to `matrix.n_features()` on a
                // float-only pool, so this is byte-unchanged there.
                device_n_features,
                device_n_bins,
                learning_rate,
                device_scaled_l2,
                &device_config,
            )?
        }
    } else {
        false
    };
    // Teardown on EVERY exit path (incl. the `?` error path), T-10-24. Inert when
    // no session was opened (`device_active == false`).
    let _device_guard = DeviceSessionGuard {
        runtime,
        active: device_active,
    };

    // Poisson exists ONLY as a device sampler (upstream rejects it on the CPU task type and
    // `bootstrap()` below does the same). If the fit did not actually commit to the device —
    // a CPU/wgpu build, or any config the coverage gate declined — there is nothing that can
    // express it, so fail here with the reason rather than let the CPU grower's `bootstrap()`
    // raise a bare "unsupported" from deep inside the tree loop.
    if matches!(params.bootstrap_type, EBootstrapType::Poisson) && !device_active {
        return Err(CbError::Degenerate(
            "poisson bootstrap is not supported on CPU (upstream CatBoost rejects it on the \
             CPU task type). It requires the device grow path: build with the `cuda` or \
             `rocm` backend feature and a device-eligible configuration (grow_policy = \
             SymmetricTree, random_strength = 0, unit object weights, boost_from_average = \
             false, Gradient/Simple leaves, no CTR / eval sets / groups)"
                .to_owned(),
        ));
    }
    // Poisson is drawn device-resident from its own persistent seed buffer, so the host runs
    // no per-tree sampler and consumes no draws for it — unlike every other bootstrap type,
    // whose host draw order is load-bearing for upstream parity.
    let device_poisson =
        device_active && matches!(params.bootstrap_type, EBootstrapType::Poisson);

    // One approx per LEARNING fold (Plain CTR path — `TFold`/`UpdateLearningFold`,
    // `train.cpp:585`). Upstream's structure search does NOT read the averaging
    // fold's derivatives: each learning fold carries its OWN approx, advanced every
    // iteration over that fold's OWN CTR-bin leaf assignment, and the greedy search
    // consumes the TAKEN fold's derivatives. With CTRs the fold partition diverges
    // from the averaging partition from iteration 1 on (structure bins ≠ averaging
    // bins), so feeding the single averaging approx to the search accumulates drift
    // until a split choice flips — invisible on the committed 5-iteration CTR
    // oracles, a >1e-1 prediction divergence by 20 iterations (the
    // `ctr_borders_multiprior` localization, 2026-08-02). EMPTY on every non-CTR
    // path (`structure_fold_columns` is empty there), so the float / one-hot /
    // ordered / grouped paths are structurally byte-identical.
    let mut fold_approxes: Vec<Vec<f64>> = if structure_fold_columns.is_empty() {
        Vec::new()
    } else {
        vec![approx.clone(); structure_fold_columns.len()]
    };

    // EXP-DOMAIN approx semantics for `IsStoreExpApprox` losses on the CTR path
    // (`approx_updater_helpers.h:60-72`; see [`crate::fast_approx`]). Upstream
    // stores every TRAINING-FOLD approx (learning folds AND the averaging fold)
    // as `exp(approx)` for these losses, and applies deltas through APPROXIMATE
    // transcendentals — `fmath::expd_v` per leaf, `fast_exp(FastLogf(·)·lr)` per
    // document. Their ~1e-6 per-application error feeds the next iteration's
    // derivatives and moves greedy split scores across tie-break boundaries by
    // ~10-20 iterations, so an exact-`exp` engine diverges from upstream's
    // chosen STRUCTURE at iteration scale (the `ctr_borders_multiprior`
    // localization, verified against an instrumented v1.2.10 build).
    //
    // Scope: the cat-CTR path (`structure_fold_columns` non-empty) with the
    // binclf losses, matching every committed CTR fixture. The float-only /
    // one-hot Logloss paths keep the exact-`exp` derivative stream — their
    // committed oracles prove the divergence stays under the 1e-5 gate at their
    // iteration scale; widening the exp-domain semantics to those paths is a
    // recorded follow-up, not a silent behavior change here. The model-output
    // approx (`approx`, upstream's `AvrgApprox`) STAYS linear and exact — only
    // derivative computation reads the exp-domain buffers.
    let exp_ctr = !structure_fold_columns.is_empty()
        && matches!(params.loss, cb_compute::Loss::Logloss | cb_compute::Loss::CrossEntropy)
        && approx_dimension == 1;
    // The averaging fold's exp approx (`AveragingFold.BodyTailArr[0].Approx`) —
    // feeds the LEAF-VALUE derivatives. Initialized like upstream's
    // `InitApproxes` + `ExpApproxIf` (the fmath batch exp of the starting
    // approx; `fmath_expd(0) == 1` for the un-biased binclf start).
    let mut avg_exp_approx: Vec<f64> = if exp_ctr {
        approx.iter().map(|&a| crate::fast_approx::fmath_expd(a)).collect()
    } else {
        Vec::new()
    };
    // Learning-fold approxes switch to exp domain under the same gate.
    if exp_ctr {
        for fa in &mut fold_approxes {
            for v in fa.iter_mut() {
                *v = crate::fast_approx::fmath_expd(*v);
            }
        }
    }

    // `TLearnProgress::UsedCtrSplits` (learn_context.h:108) — the MODEL-LIFETIME
    // set of `(ctr_type, projection)` pairs some already-grown tree split on.
    // `GetCatFeatureWeight` lifts the model-size penalty (weight 1.0) for
    // members; `ProcessCtrSplit` inserts the pair the moment a level chooses a
    // CTR split (greedy_tensor_search.cpp:926-950, :1126). Accumulated across
    // the whole fit and passed into every tree's structure search.
    let mut used_ctr_splits: Vec<(i8, crate::TProjection)> = Vec::new();

    // ORCH-03-S5: snapshot admission. Placed HERE — after every gate local this
    // guard reads is computed (`device_active` at the device-begin above is the
    // last of them) and BEFORE the first tree grows — so an out-of-scope regime is
    // refused without ever writing a file. `snapshot == None` skips the whole block
    // and leaves the loop byte-identical (the D-04 anchor).
    let snapshot_state = match snapshot {
        None => None,
        Some(cfg) => {
            snapshot_scope_ok(
                params,
                cat_columns,
                eval_sets,
                approx_dimension,
                penalties_active,
                device_active,
                staged_out.is_some(),
                &ranking,
            )?;
            let fingerprint =
                crate::snapshot::fingerprint(params, n, feature_borders, target, &weights);
            Some((cfg, fingerprint))
        }
    };
    // ORCH-03-S6: RESUME. When the configured file already exists and its stored
    // fingerprint matches this run, the loop-carried state is replaced wholesale
    // with the checkpoint's and the loop starts at `completed_iters` instead of 0.
    //
    // `approx` is taken VERBATIM from the checkpoint rather than rebuilt by
    // re-applying the persisted trees: re-application would re-associate the
    // per-iteration floating-point sums, and a resumed run must be BIT-identical to
    // the straight-through run, not merely close.
    //
    // A fingerprint mismatch is an ERROR, never a silent fresh start — the file
    // belongs to a different configuration, and quietly ignoring it would discard
    // work the caller believes is being continued.
    let mut resume_from = 0usize;
    if let Some((cfg, fingerprint)) = snapshot_state {
        if cfg.snapshot_file.exists() {
            let stored = crate::snapshot::read_from(&cfg.snapshot_file)?;
            crate::snapshot::check_resume(stored.fingerprint, fingerprint)?;

            // Both are deterministic functions of the fingerprinted inputs, so a
            // disagreement means the fingerprint failed to cover something — a
            // silent-corruption bug, not a user error. Fail loudly.
            if stored.approx_dimension != approx_dimension {
                return Err(CbError::Snapshot(format!(
                    "snapshot approx_dimension {} != this run's {approx_dimension} despite a \
                     matching fingerprint",
                    stored.approx_dimension
                )));
            }
            if stored.approx.len() != approx.len() {
                return Err(CbError::Snapshot(format!(
                    "snapshot approx has {} values, this run has {} objects despite a matching \
                     fingerprint",
                    stored.approx.len(),
                    approx.len()
                )));
            }
            if stored.completed_iters > params.iterations {
                return Err(CbError::Snapshot(format!(
                    "snapshot holds {} completed iterations, more than this run's {} — nothing \
                     to resume",
                    stored.completed_iters, params.iterations
                )));
            }
            if stored.trees.len() != stored.completed_iters {
                return Err(CbError::Snapshot(format!(
                    "snapshot claims {} completed iterations but carries {} trees",
                    stored.completed_iters,
                    stored.trees.len()
                )));
            }

            // `bias` is `starting_approx(params, target)` — a pure function of the
            // fingerprinted `loss` / `boost_from_average` / `target`. It is
            // therefore VERIFIED against the checkpoint rather than restored from
            // it: a disagreement means the fingerprint failed to cover an input
            // that moves the starting approximant, which is a silent-corruption
            // bug. Compared on bits, since this must be exact, not close.
            if stored.bias.to_bits() != bias.to_bits() {
                return Err(CbError::Snapshot(format!(
                    "snapshot bias {} != this run's {bias} despite a matching fingerprint",
                    stored.bias
                )));
            }

            trees = stored.trees.iter().map(crate::snapshot::tree_from_dto).collect();
            approx = stored.approx;
            rng = cb_core::TFastRng64::from_raw_state(stored.rng_raw_state, stored.rng_call_count);
            resume_from = stored.completed_iters;
        }
    }

    // Interval accounting for the periodic write. Seeded at the loop start so the
    // FIRST checkpoint also honours `snapshot_interval` (a zero interval writes at
    // every tree, which is what the deterministic tests use).
    let mut last_snapshot_write = std::time::Instant::now();

    // `resume_from` is 0 unless a checkpoint was just restored, so the non-snapshot
    // path keeps the original `0..iterations` bound byte-for-byte (the D-04 anchor).
    for iter in resume_from..params.iterations {
        // GPUT-01 DEVICE GROW BRANCH (D-10-01 per-fit all-or-nothing). When the fit
        // committed to the device path at `begin` (`device_active`), grow THIS
        // iteration's oblivious tree on the device seam and fold it into the Model
        // IDENTICALLY to a CPU-grown tree (Task 2: the `bin_id -> border` join). The
        // entire CPU body below is skipped (`continue`) and stays byte-unchanged for
        // every non-device fit (D-04). `iter` is unused on this branch (the device
        // grow is stateless per tree over the resident session); the CPU body reads
        // it, so it is not a dead binding.
        if device_active {
            let _ = iter;
            // CB_GPU_PROF host-stage attribution for the device fold (SPD-03 §2.3: the
            // per-iteration host work between device calls was invisible to profiling).
            // Cold when unset: one cached env read, three Instant reads, no prints.
            let host_prof = crate::gpu_prof_host_enabled();
            let hp_t0 = std::time::Instant::now();

            // ─── WR-01: PER-TREE HOST BOOTSTRAP (Design A) ───────────────────────────
            // The device branch keeps the ENTIRE sampler on the host: `bootstrap()` is the
            // ≤1e-5-verified CPU sampler and the sole source of `sample_weights`/`control`,
            // so reusing it (rather than the device-resident draw) is what makes upstream
            // parity reachable at all. Only the resulting per-object multiplier crosses the
            // seam. The draw ORDER here is IDENTICAL to the CPU branch's — `PRE_TREE_DRAWS`
            // → `bootstrap()` → the level-search draws (replayed after the grow below) →
            // `POST_TREE_EXTRA_DRAWS` — because tree `k+1`'s sample is drawn from the phase
            // tree `k` left behind, so any miscount silently changes every later tree's
            // sample (`WR01-S7`).
            let device_sample: Vec<f64> = if draws_active && !device_poisson {
                // 1a. PRE-bootstrap per-iteration draws (train.cpp:208,211).
                for _ in 0..PRE_TREE_DRAWS {
                    rng.gen_rand();
                }
                // `bootstrap()` takes the OBJECT COUNT from `derivatives.len()`, so this
                // vector's LENGTH is load-bearing for every arm even when its VALUES are
                // not: a short vector silently yields an empty sample (⇒ the multiplier
                // defaults to 1.0 everywhere AND the arm consumes no draws, desynchronising
                // the stream). It must always be length `n`.
                //
                // MVS is the only arm that reads the VALUES (its threshold is a function of
                // `|der|`), so the gradient round-trip is paid only when MVS is selected —
                // Bayesian reads just `n` (`generate_random_weights`) and Bernoulli just `n`
                // (`set_sampled_control`), so a zero-filled vector is exactly equivalent for
                // them and keeps the hot device path free of an extra n-length pass.
                //
                // At `approx_dimension == 1` with the unit weights the gate requires, the
                // CPU's `der_obj[i] = sqrt(Σ_d weighted_der1²)` collapses to `|der1[i]|`.
                let der_obj: Vec<f64> = if matches!(params.bootstrap_type, EBootstrapType::Mvs) {
                    let ders =
                        runtime.compute_gradients(&params.loss, &approx, target, approx_dimension)?;
                    (0..n)
                        .map(|i| ders.der1.get(i).copied().unwrap_or(0.0).abs())
                        .collect()
                } else {
                    vec![0.0_f64; n]
                };
                // 1b. The ONE per-tree `bootstrap()` call on the continuous stream.
                // `prev_leaf_mean_l2` is this fit's carried MVS λ input (`WR01-S8`) — it is
                // `None` on the first tree and the previous tree's mean leaf L2 norm after.
                let sampled = bootstrap(
                    params.bootstrap_type,
                    &der_obj,
                    params.subsample,
                    params.bagging_temperature,
                    prev_leaf_mean_l2,
                    &mut rng,
                )?;
                // The per-object SPLIT-SCORING multiplier (`WR01-S6`): exactly the CPU
                // branch's `control[i] ? sample_weights[i] : 0.0`. A zeroed entry excludes
                // the object from the split histogram, which is how a `control == false`
                // object drops out of upstream's `sampledDocs`. Leaf estimation on the
                // device consumes the UNSAMPLED channels, mirroring the CPU split.
                (0..n)
                    .map(|i| {
                        let sw = sampled.sample_weights.get(i).copied().unwrap_or(1.0);
                        let c = sampled.control.get(i).copied().unwrap_or(true);
                        if c {
                            sw
                        } else {
                            0.0
                        }
                    })
                    .collect()
            } else {
                // `bootstrap_type == No` and `random_strength == 0`: no sampling, no draws.
                // An EMPTY sample keeps the device grow byte-identical to the pre-WR-01
                // path (`WR01-S3`, D-04).
                Vec::new()
            };

            // The stored (learning-rate-scaled) leaf values of THIS tree, captured by the
            // fold arms below so the MVS λ carry after them has something to read.
            let mut device_stored_leaf_values: Vec<f64> = Vec::new();

            let hp_sample_ms = hp_t0.elapsed().as_secs_f64() * 1e3;
            let hp_grow_t = std::time::Instant::now();
            let dev_tree = runtime
                .grow_tree_on_device(&approx, target, &device_sample, None)?
                .ok_or_else(|| {
                    // `begin` returned Ok(true): the whole fit is committed to the
                    // device grower (D-10-01). Folding a CPU-grown tree here would MIX
                    // device- and CPU-grown trees in one model (T-10-23) — reject it
                    // with a typed error rather than silently corrupt the model.
                    CbError::Degenerate(
                        "device grow returned Ok(None) after begin_device_training \
                         committed the fit to the device path; per-fit all-or-nothing \
                         (D-10-01) forbids mixing a CPU-grown tree into a device-grown \
                         model"
                            .to_owned(),
                    )
                })?;
            let hp_grow_ms = hp_grow_t.elapsed().as_secs_f64() * 1e3;
            let hp_fold_t = std::time::Instant::now();

            // Phase 12 Plan 03/04 (GPUT-18): dispatch on the populated `DeviceGrownTree`
            // SHAPE, in the SAME order as the CPU fold at `:4419`: a NON-EMPTY
            // `region_path` is a Region PATH tree → `RegionTree` into `region_trees`
            // (Plan 04); else an EMPTY `step_nodes` is the oblivious / symmetric
            // emission (the byte-unchanged Plan-01 path → `ObliviousTree`); else a
            // NON-EMPTY `step_nodes` is a Depthwise / Lossguide non-symmetric node
            // graph → `NonSymmetricTree` into `non_symmetric_trees`.
            if !dev_tree.region_path.is_empty() {
                // ─── REGION ARM (walk-until-diverge PATH, GPUT-18 / D-03a) ───
                // The device emits a per-level `(feature, bin_id, expected_direction,
                // one_hot)` `region_path` (length `depth`) plus `depth + 1` leaf
                // values (NOT `2^depth` — a path, not a node graph). Resolve each
                // level's `(feature, bin_id)` to a Model `Split` via the range-checked
                // `feature_borders[feature][bin_id]` join (T-10-22: an out-of-range
                // index is a typed error, never a panic / raw index).
                let region_depth = dev_tree.region_path.len();
                let mut region_splits: Vec<Split> = Vec::with_capacity(region_depth);
                let mut region_directions: Vec<bool> = Vec::with_capacity(region_depth);
                let mut region_one_hot: Vec<bool> = Vec::with_capacity(region_depth);
                for &(feature, bin_id, expected_direction, one_hot) in &dev_tree.region_path {
                    let f = feature as usize;
                    let b = bin_id as usize;
                    let border = feature_borders
                        .get(f)
                        .and_then(|borders| borders.get(b))
                        .copied()
                        .ok_or_else(|| {
                            CbError::OutOfRange(format!(
                                "device region split (feature {f}, bin_id {b}) is out of range for \
                                 feature_borders (feature count {}, feature border count {})",
                                feature_borders.len(),
                                feature_borders.get(f).map_or(0, Vec::len),
                            ))
                        })?;
                    region_splits.push(Split { feature: f, border });
                    region_directions.push(expected_direction);
                    region_one_hot.push(one_hot);
                }

                // Per-object terminal bin via the walk-until-diverge path (TRANSCRIBED
                // from `cb_model::apply::region_leaf` / `AddRegionImpl`, NOT imported):
                // `bin = 0; for level { split = one_hot ? val == border : val > border;
                // if split != expected_direction { break }; bin += 1 } leaf = bin`. All
                // access checked → a malformed level halts the walk (bin so far), never
                // a panic / raw index (T-12-05). `one_hot` is always false for the
                // device float grower, so the `>` test is authoritative.
                let device_leaf_of: Vec<usize> = (0..n)
                    .into_par_iter()
                    .map(|obj| {
                        let mut bin = 0usize;
                        for level in 0..region_depth {
                            let (Some(s), Some(&dir), Some(&oh)) = (
                                region_splits.get(level),
                                region_directions.get(level),
                                region_one_hot.get(level),
                            ) else {
                                break;
                            };
                            let val = feature_values
                                .get(s.feature)
                                .and_then(|col| col.get(obj))
                                .map(|&v| f64::from(v));
                            let Some(val) = val else { break };
                            let split = if oh { val == s.border } else { val > s.border };
                            if split == dir {
                                bin += 1;
                            } else {
                                break;
                            }
                        }
                        bin
                    })
                    .collect();

                // A depth-`d` Region has `d + 1` leaves (== the device leaf-value
                // vector length), indexed DIRECTLY by the walk bin (0..=depth).
                let region_n_leaves = dev_tree.leaf_values.len();
                let mut device_leaf_values = dev_tree.leaf_values.clone();
                let device_leaf_weights =
                    accumulate_leaf_weights(&device_leaf_of, &weights, region_n_leaves);
                normalize_leaf_values(
                    /* is_pairwise = */ false,
                    learning_rate,
                    &device_leaf_weights,
                    &mut device_leaf_values,
                    region_n_leaves,
                    /* approx_dimension = */ 1,
                );
                // WR-01 (`WR01-S8`): capture the STORED (lr-scaled) leaf values so the
                // per-tree MVS λ carry after the fold arms can compute this tree's mean
                // leaf L2 norm, exactly as the CPU branch does.
                device_stored_leaf_values.clone_from(&device_leaf_values);

                // Elementwise, order-independent per-object add — bit-identical to the
                // serial loop (each slot is touched exactly once), parallel over objects
                // (this pass runs once per boosting iteration at n scale).
                approx
                    .par_iter_mut()
                    .zip(device_leaf_of.par_iter())
                    .for_each(|(a, &leaf)| {
                        if let Some(&lv) = device_leaf_values.get(leaf) {
                            *a += lv;
                        }
                    });

                if let Some(out) = staged_out.as_deref_mut() {
                    out.extend_from_slice(&approx);
                }

                region_trees.push(RegionTree {
                    splits: region_splits,
                    directions: region_directions,
                    one_hot: region_one_hot,
                    leaf_values: device_leaf_values,
                    leaf_weights: device_leaf_weights,
                });
            } else if dev_tree.step_nodes.is_empty() {
                // ─── OBLIVIOUS ARM (byte-identical to the Plan-01 device fold) ───
                // Resolve each device split `(feature, bin_id)` to a Model `Split` via
                // `border = feature_borders[feature][bin_id]` (Pattern 4). Range-check
                // `bin_id` (T-10-22): an out-of-range index is a typed error, never a
                // panic / raw index. `DeviceGrownTree.leaf_of` is NOT consumed (D-05).
                // SPEC-OH-24: the device feature axis is the CONCATENATION
                // `float | one-hot` (T24's layout), so a device index `>= device_n_float`
                // is one-hot column `idx - device_n_float`. Map it back to the ABSOLUTE
                // cat-column index through `one_hot_absolute` (the inverse of the layout)
                // and emit a `LevelKind::OneHot` + `OneHotSplit` instead of a float
                // `Split`. `level_kinds` stays EMPTY when every level is float, so the
                // float-only fold is byte-identical (SPEC-OH-31).
                let mut device_splits: Vec<Split> = Vec::with_capacity(dev_tree.splits.len());
                let mut device_one_hot_splits: Vec<crate::tree::OneHotSplit> = Vec::new();
                let mut device_level_kinds: Vec<crate::tree::LevelKind> = Vec::new();
                // GDC-11 (T14): the chosen CTR splits, translated from the device
                // CTR tail (`feature >= device_n_features`) back to the CPU
                // `CtrSplitSpec` identity (mirrors the CPU search emission,
                // `tree.rs` `CtrAwareSplit::Ctr` arm — same value-space border,
                // same default Shift/Scale the bake later overwrites).
                let mut device_ctr_splits: Vec<CtrSplitSpec> = Vec::new();
                let device_has_ctr_split = dev_tree
                    .splits
                    .iter()
                    .any(|&(f, _, _)| (f as usize) >= device_n_features);
                let device_has_one_hot = dev_tree.splits.iter().any(|&(_, _, oh)| oh);
                let device_mixed_kinds = device_has_one_hot || device_has_ctr_split;
                for &(feature, bin_id, is_one_hot) in &dev_tree.splits {
                    let f = feature as usize;
                    let b = bin_id as usize;
                    if f >= device_n_features {
                        // CTR tail column `f - device_n_features` (the session
                        // appends the structure CTR columns after `float|one-hot`).
                        let col = f - device_n_features;
                        let column =
                            materialized_ctr_features.get(col).ok_or_else(|| {
                                CbError::OutOfRange(format!(
                                    "device CTR split names tail column {col}, but only {} \
                                     CTR column(s) are materialized",
                                    materialized_ctr_features.len()
                                ))
                            })?;
                        device_level_kinds.push(crate::tree::LevelKind::Ctr {
                            ctr_idx: device_ctr_splits.len(),
                            // BIN space — the training-only border
                            // `assign_leaf_over_ctr_columns` tests integer bins
                            // against (see the CPU emission's units contract).
                            border: b as f64,
                        });
                        device_ctr_splits.push(CtrSplitSpec {
                            projection: column.projection.clone(),
                            ctr_type: column.ctr_type,
                            prior_num: column.prior_num,
                            prior_denom: column.prior_denom,
                            target_border_idx: column.target_border_idx,
                            // VALUE space (SPEC-CTRB-01) for every persisted-border
                            // consumer, exactly like the CPU search emission.
                            border: crate::tree::ctr_bin_border_to_value_space(b as f64),
                            shift: 0.0,
                            scale: 1.0,
                        });
                        continue;
                    }
                    if is_one_hot {
                        let pos = f.checked_sub(device_n_float).ok_or_else(|| {
                            CbError::OutOfRange(format!(
                                "device one-hot split names device feature {f}, which is \
                                 below the float boundary {device_n_float} (internal \
                                 invariant: pass B sweeps only [{device_n_float}, ..))"
                            ))
                        })?;
                        let absolute = one_hot_absolute.get(pos).copied().ok_or_else(|| {
                            CbError::OutOfRange(format!(
                                "device one-hot split names one-hot column {pos}, but only \
                                 {} column(s) were routed one-hot",
                                one_hot_absolute.len()
                            ))
                        })?;
                        device_level_kinds.push(crate::tree::LevelKind::OneHot(
                            device_one_hot_splits.len(),
                        ));
                        device_one_hot_splits.push(crate::tree::OneHotSplit {
                            feature: absolute,
                            value: bin_id,
                        });
                        continue;
                    }
                    let border = feature_borders
                        .get(f)
                        .and_then(|borders| borders.get(b))
                        .copied()
                        .ok_or_else(|| {
                            CbError::OutOfRange(format!(
                                "device split (feature {f}, bin_id {b}) is out of range for \
                                 feature_borders (feature count {}, feature border count {})",
                                feature_borders.len(),
                                feature_borders.get(f).map_or(0, Vec::len),
                            ))
                        })?;
                    if device_mixed_kinds {
                        device_level_kinds
                            .push(crate::tree::LevelKind::Float(device_splits.len()));
                    }
                    device_splits.push(Split { feature: f, border });
                }

                // Per-object leaf assignment on the HOST, forward bit order (split `l` ->
                // bit `l` — the SAME `value > border` + `leaf_index` semantics the CPU
                // oblivious path uses; D-05). The split columns are resolved ONCE outside
                // the object loop and the leaf bits set directly (no per-object Vec<bool>
                // allocation — this loop runs n times per boosting iteration).
                //
                // SPEC-OH-24: with one-hot levels present the LEVEL order is what fixes
                // each bit, so the per-level column + test are resolved from
                // `device_level_kinds` — a float level keeps the `value > border` test,
                // a one-hot level uses `cat_bin == value` over the ONE-HOT bin column (by
                // one-hot POSITION, which is the device index minus the float boundary).
                // With no one-hot level `device_level_kinds` is empty and this collapses
                // to the byte-identical float-only loop.
                enum DeviceLevelCol<'a> {
                    Float(&'a [f32], f64),
                    OneHot(&'a [u32], u32),
                }
                let level_cols: Vec<DeviceLevelCol<'_>> = if device_has_one_hot {
                    device_level_kinds
                        .iter()
                        .map(|kind| match kind {
                            crate::tree::LevelKind::OneHot(idx) => {
                                let s = device_one_hot_splits.get(*idx);
                                let pos = s.and_then(|s| {
                                    one_hot_absolute.iter().position(|&a| a == s.feature)
                                });
                                DeviceLevelCol::OneHot(
                                    pos.and_then(|p| one_hot_bins.get(p))
                                        .map_or(&[][..], Vec::as_slice),
                                    s.map_or(u32::MAX, |s| s.value),
                                )
                            }
                            crate::tree::LevelKind::Float(idx) => {
                                let s = device_splits.get(*idx);
                                DeviceLevelCol::Float(
                                    s.and_then(|s| feature_values.get(s.feature))
                                        .map_or(&[][..], Vec::as_slice),
                                    s.map_or(f64::INFINITY, |s| s.border),
                                )
                            }
                            // A device-grown oblivious tree never carries a CTR level
                            // (host eligibility excludes CTR pools entirely), so this arm
                            // is unreachable; route it to a never-passing float test
                            // rather than fabricating a split.
                            crate::tree::LevelKind::Ctr { .. } => {
                                DeviceLevelCol::Float(&[][..], f64::INFINITY)
                            }
                        })
                        .collect()
                } else {
                    device_splits
                        .iter()
                        .map(|s| {
                            DeviceLevelCol::Float(
                                feature_values.get(s.feature).map_or(&[][..], Vec::as_slice),
                                s.border,
                            )
                        })
                        .collect()
                };
                // Leaf values: the device returns UN-scaled leaves; cb-train applies the
                // `learning_rate` shrinkage. Non-pairwise → `normalize_leaf_values` applies
                // ONLY the lr scale (byte-identical to `learning_rate * delta`, D-04) and
                // reads `leaf_weights` NOT AT ALL — which is what licenses the fused path
                // below to scale BEFORE the weights exist.
                let mut device_leaf_values = dev_tree.leaf_values.clone();
                // SPD-03 wave 3 (P100 260808-r2: the three separate O(n) fold sweeps —
                // walk, weight bucketing, approx add — still cost ~18 ms/tree on the
                // 4-vCPU Kaggle host). The unit-weight float-only fold — every unweighted
                // fit — fuses them into ONE parallel sweep with per-chunk integer leaf
                // counts, never materializing `device_leaf_of`. Bit-exact: the walk is the
                // same per-object test, the approx add hits each slot once with the same
                // value, and a fold of k unit weights is EXACTLY `k as f64` (integer f64
                // addition below 2^53), order-free.
                let fused_unit_fold = !device_has_ctr_split && weights.iter().all(|&w| w == 1.0);
                let device_leaf_weights: Vec<f64> = if fused_unit_fold {
                    normalize_leaf_values(
                        /* is_pairwise = */ false,
                        learning_rate,
                        &[],
                        &mut device_leaf_values,
                        n_leaves,
                        /* approx_dimension = */ 1,
                    );
                    const FOLD_CHUNK: usize = 1 << 16;
                    let weights_len = weights.len();
                    let counts: Vec<u64> = approx
                        .par_chunks_mut(FOLD_CHUNK)
                        .enumerate()
                        .map(|(c, approx_chunk)| {
                            let start = c * FOLD_CHUNK;
                            let mut counts = vec![0_u64; n_leaves];
                            for (off, a) in approx_chunk.iter_mut().enumerate() {
                                let obj = start + off;
                                let mut leaf = 0usize;
                                for (l, col) in level_cols.iter().enumerate() {
                                    let passes = match col {
                                        DeviceLevelCol::Float(values, border) => values
                                            .get(obj)
                                            .is_some_and(|&v| f64::from(v) > *border),
                                        DeviceLevelCol::OneHot(bins, value) => {
                                            bins.get(obj).is_some_and(|&b| b == *value)
                                        }
                                    };
                                    if passes {
                                        leaf |= 1usize << l;
                                    }
                                }
                                // Same membership rule as `accumulate_leaf_weights`: an
                                // object contributes to the weight only when it has a
                                // weight entry (and the leaf is in range).
                                if obj < weights_len {
                                    if let Some(slot) = counts.get_mut(leaf) {
                                        *slot += 1;
                                    }
                                }
                                if let Some(&lv) = device_leaf_values.get(leaf) {
                                    *a += lv;
                                }
                            }
                            counts
                        })
                        .reduce(
                            || vec![0_u64; n_leaves],
                            |mut x, y| {
                                for (a, b) in x.iter_mut().zip(y.iter()) {
                                    *a += b;
                                }
                                x
                            },
                        );
                    counts.into_iter().map(|c| c as f64).collect()
                } else {
                    let device_leaf_of: Vec<usize> = if device_has_ctr_split {
                        // GDC-11/GDC-10 (T14): the LEAF-VALUE / main-approx assignment
                        // for a CTR tree is the AVERAGING-fold partition — exactly the
                        // CPU `leaf_value_leaf_of` (`assign_leaf_over_ctr_columns`,
                        // `BuildIndices(AveragingFold)`): float levels keep the float
                        // test, CTR levels re-test against the AVERAGING column's
                        // bins. The device already gathered the RETURNED leaf values
                        // over this same partition (GDC-10), so values and assignment
                        // stay consistent here.
                        let grown_like = GrownTree {
                            splits: device_splits.clone(),
                            one_hot_splits: device_one_hot_splits.clone(),
                            leaf_of: Vec::new(),
                            ctr_splits: device_ctr_splits.clone(),
                            level_kinds: device_level_kinds.clone(),
                            step_nodes: Vec::new(),
                            node_id_to_leaf_id: Vec::new(),
                            region_directions: Vec::new(),
                            region_one_hot: Vec::new(),
                        };
                        assign_leaf_over_ctr_columns(&matrix, &averaging_ctr_features, &grown_like, n)
                    } else {
                        // Pure per-object walk (reads only resolved split columns) —
                        // parallel over objects, deterministic per index (this walk runs
                        // once per boosting iteration at n scale).
                        (0..n)
                            .into_par_iter()
                            .map(|obj| {
                                let mut leaf = 0usize;
                                for (l, col) in level_cols.iter().enumerate() {
                                    let passes = match col {
                                        DeviceLevelCol::Float(values, border) => values
                                            .get(obj)
                                            .is_some_and(|&v| f64::from(v) > *border),
                                        DeviceLevelCol::OneHot(bins, value) => {
                                            bins.get(obj).is_some_and(|&b| b == *value)
                                        }
                                    };
                                    if passes {
                                        leaf |= 1usize << l;
                                    }
                                }
                                leaf
                            })
                            .collect()
                    };

                    let device_leaf_weights =
                        accumulate_leaf_weights(&device_leaf_of, &weights, n_leaves);
                    normalize_leaf_values(
                        /* is_pairwise = */ false,
                        learning_rate,
                        &device_leaf_weights,
                        &mut device_leaf_values,
                        n_leaves,
                        /* approx_dimension = */ 1,
                    );
                    // Elementwise, order-independent per-object add — bit-identical to the
                    // serial loop (each slot is touched exactly once), parallel over objects
                    // (this pass runs once per boosting iteration at n scale).
                    approx
                        .par_iter_mut()
                        .zip(device_leaf_of.par_iter())
                        .for_each(|(a, &leaf)| {
                            if let Some(&lv) = device_leaf_values.get(leaf) {
                                *a += lv;
                            }
                        });
                    device_leaf_weights
                };
                // WR-01 (`WR01-S8`): capture the STORED (lr-scaled) leaf values so the
                // per-tree MVS λ carry after the fold arms can compute this tree's mean
                // leaf L2 norm, exactly as the CPU branch does.
                device_stored_leaf_values.clone_from(&device_leaf_values);

                if let Some(out) = staged_out.as_deref_mut() {
                    out.extend_from_slice(&approx);
                }

                // GDC-11: a chosen CTR split enters the model-lifetime
                // `UsedCtrSplits` (upstream `ProcessCtrSplit`) exactly like the
                // CPU branch — the bake and any later CPU-side scoring read it.
                for spec in &device_ctr_splits {
                    let key = (spec.ctr_type, spec.projection.clone());
                    if !used_ctr_splits.contains(&key) {
                        used_ctr_splits.push(key);
                    }
                }

                // `level_kinds` stays EMPTY when the device tree is single-kind (float
                // only) — consumers then take the byte-identical legacy path
                // (SPEC-OH-31). It is populated when a one-hot OR CTR level is present,
                // in which case it carries the full LEVEL ORDER (SPEC-OH-01/24, GDC-11).
                trees.push(oblivious_from_grown(
                    device_splits,
                    device_ctr_splits,
                    device_one_hot_splits,
                    device_level_kinds,
                    device_leaf_values,
                    device_leaf_weights,
                ));
            } else {
                // ─── NON-SYMMETRIC ARM (Depthwise / Lossguide, GPUT-18) ───
                // The device emits a PER-NODE `(feature, bin_id)` in `dev_tree.splits`
                // (one per node, `(0,0)` placeholder for leaf nodes) plus the node graph
                // (`step_nodes` / `node_id_to_leaf_id`). Resolve each INTERIOR node's split
                // via the SAME `feature_borders[feature][bin_id]` join; leaf placeholder
                // nodes (step `(0,0)`) get an inert `Split` that the walk never reads for
                // routing (its diffs are zero → the node is a halt point).
                let mut device_splits: Vec<Split> = Vec::with_capacity(dev_tree.splits.len());
                // SPEC-OH-24: the non-symmetric growers are OUT OF SCOPE for one-hot, so
                // the kind is always `false` here and is deliberately ignored.
                for (node, &(feature, bin_id, _one_hot)) in dev_tree.splits.iter().enumerate() {
                    let is_leaf = dev_tree
                        .step_nodes
                        .get(node)
                        .map_or(true, |&(ld, rd)| ld == 0 && rd == 0);
                    if is_leaf {
                        // Inert placeholder split for a terminal node (never routes).
                        device_splits.push(Split { feature: 0, border: 0.0 });
                        continue;
                    }
                    let f = feature as usize;
                    let b = bin_id as usize;
                    let border = feature_borders
                        .get(f)
                        .and_then(|borders| borders.get(b))
                        .copied()
                        .ok_or_else(|| {
                            CbError::OutOfRange(format!(
                                "device non-sym split (feature {f}, bin_id {b}) is out of range for \
                                 feature_borders (feature count {}, feature border count {})",
                                feature_borders.len(),
                                feature_borders.get(f).map_or(0, Vec::len),
                            ))
                        })?;
                    device_splits.push(Split { feature: f, border });
                }

                // Per-object DISTINCT-leaf assignment via the transcribed
                // `leaf_index_nonsym` pointer-walk. A malformed graph → checked leaf-0
                // fallback (never a panic, T-12-05).
                let device_leaf_of: Vec<usize> = (0..n)
                    .into_par_iter()
                    .map(|obj| {
                        device_leaf_of_nonsym(
                            obj,
                            &device_splits,
                            &dev_tree.step_nodes,
                            &dev_tree.node_id_to_leaf_id,
                            feature_values,
                        )
                        .unwrap_or(0)
                    })
                    .collect();

                // Non-sym leaf count is the DISTINCT-leaf count (NOT 2^depth), == the
                // device leaf-value vector length.
                let nonsym_n_leaves = dev_tree.leaf_values.len();
                let mut device_leaf_values = dev_tree.leaf_values.clone();
                let device_leaf_weights =
                    accumulate_leaf_weights(&device_leaf_of, &weights, nonsym_n_leaves);
                normalize_leaf_values(
                    /* is_pairwise = */ false,
                    learning_rate,
                    &device_leaf_weights,
                    &mut device_leaf_values,
                    nonsym_n_leaves,
                    /* approx_dimension = */ 1,
                );
                // WR-01 (`WR01-S8`): capture the STORED (lr-scaled) leaf values so the
                // per-tree MVS λ carry after the fold arms can compute this tree's mean
                // leaf L2 norm, exactly as the CPU branch does.
                device_stored_leaf_values.clone_from(&device_leaf_values);

                // Elementwise, order-independent per-object add — bit-identical to the
                // serial loop (each slot is touched exactly once), parallel over objects
                // (this pass runs once per boosting iteration at n scale).
                approx
                    .par_iter_mut()
                    .zip(device_leaf_of.par_iter())
                    .for_each(|(a, &leaf)| {
                        if let Some(&lv) = device_leaf_values.get(leaf) {
                            *a += lv;
                        }
                    });

                if let Some(out) = staged_out.as_deref_mut() {
                    out.extend_from_slice(&approx);
                }

                non_symmetric_trees.push(NonSymmetricTree {
                    splits: device_splits,
                    step_nodes: dev_tree.step_nodes.clone(),
                    node_id_to_leaf_id: dev_tree.node_id_to_leaf_id.clone(),
                    leaf_values: device_leaf_values,
                    leaf_weights: device_leaf_weights,
                });
            }

            // ─── WR-01: RESTORE THE RNG PHASE + CARRY THE MVS λ ──────────────────────
            // The device grow skipped the CPU level search entirely, so the draws that
            // search would have consumed must be replayed here — otherwise the NEXT tree's
            // `bootstrap()` call above would read a different RNG phase than upstream and
            // every tree from the second on would diverge (`WR01-S7`). The replay must land
            // AFTER `bootstrap()` and BEFORE `POST_TREE_EXTRA_DRAWS` to reproduce the CPU
            // branch's order exactly. `draws_active == false` keeps this a no-op, so the
            // byte-unchanged `bootstrap_type = No` device path is untouched (D-04).
            //
            // Poisson is excluded for the same reason it skips `bootstrap()` above: its
            // randomness lives entirely in the device seed buffer, this host stream feeds
            // nothing on that arm, and there is no upstream CPU phase to stay aligned with.
            if draws_active && !device_poisson {
                // B-1: the replay is GROW-POLICY aware. Only the oblivious CPU grower
                // touches the training RNG (region_grower / leaf_wise_grower take no
                // Perturbation at all), so a Region / Depthwise / Lossguide device tree
                // must replay NOTHING — replaying the oblivious level-search shape there
                // would consume draws the CPU branch never consumes and desynchronise the
                // NEXT tree's bootstrap().
                let replay_policy = match params.grow_policy {
                    EGrowPolicy::SymmetricTree => ReplayPolicy::SymmetricTree,
                    EGrowPolicy::Region => ReplayPolicy::Region,
                    EGrowPolicy::Depthwise => ReplayPolicy::Depthwise,
                    EGrowPolicy::Lossguide => ReplayPolicy::Lossguide,
                };
                replay_grow_draws(&mut rng, replay_policy, params.depth, matrix.n_features());
                for _ in 0..POST_TREE_EXTRA_DRAWS {
                    rng.gen_rand();
                }
            }

            // MVS λ for the NEXT tree is THIS tree's mean leaf L2 norm
            // (`CalculateLastIterMeanLeafValue`, mvs.cpp:21-35) over the stored,
            // learning-rate-scaled leaf values — the same source and the same helper the
            // CPU branch uses (`WR01-S8`). Carried unconditionally so the device branch's
            // λ sequence matches the CPU branch's for every bootstrap type.
            prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&device_stored_leaf_values));

            if host_prof {
                eprintln!(
                    "CB_GPU_PROF tree-host sample={hp_sample_ms:.2}ms grow={hp_grow_ms:.2}ms \
                     fold={:.2}ms",
                    hp_fold_t.elapsed().as_secs_f64() * 1e3,
                );
            }
            continue;
        }

        // 0. YetiRank / YetiRankPairwise (Wave C): RE-SAMPLE the per-group
        //    competitor adjacency from the CURRENT approx before the der
        //    (yetirank_helpers.cpp:347-393 — the pairs are recomputed each tree).
        //
        //    PER-TREE seeding (D-07 trainer-level RNG closure, 06.3-14 ext): advance
        //    the persistent `YetiRankTreeSeeder` once for THIS tree, yielding the
        //    DERIVATIVE per-group seeds (used HERE for the gradient + split-scoring
        //    competitor sample) and the LEAF-VALUE per-group seeds (used later, at
        //    leaf-value estimation, to re-sample a DISTINCT competitor set —
        //    `CalcLeafValuesSimple` re-derives its own seed off the same context
        //    RNG, `approx_calcer.cpp:983`). The seeder reproduces the trainer's
        //    per-tree main-RNG consumption (structure draw + split-search Rsm/normal
        //    draws + the two recalc seeds) draw-for-draw, so the sampled competitors
        //    match the catboost fixture from tree 0 onward.
        let yetirank_tree_seeds = yetirank_seeder.as_mut().map(crate::YetiRankTreeSeeder::next_tree);

        // StochasticRank per-tree recalc seeds (D-07, 06.3-18). Advance the persistent
        // context RNG once for THIS tree (the SAME draw sequence as YetiRank), then
        // take the two BASE recalc seeds the grouped der re-seeds the per-group noise
        // stream with: `recalc_seeds[0]` is the DERIVATIVE recalc base (drives the
        // gradient + split scoring), `recalc_seeds[2]` is the LEAF-VALUE recalc base
        // (drives the AveragingFold leaf-value re-estimation). The grouped der applies
        // the `+ group_index` per-group offset internally (`error_functions.h:1257`).
        let stochasticrank_tree_seeds =
            stochasticrank_seeder.as_mut().map(crate::YetiRankTreeSeeder::next_tree);
        let stochasticrank_deriv_seed = stochasticrank_tree_seeds
            .as_ref()
            .map_or(params.random_seed, |s| s.recalc_seeds[0]);
        let stochasticrank_leafval_seed = stochasticrank_tree_seeds
            .as_ref()
            .map_or(params.random_seed, |s| s.recalc_seeds[2]);

        if is_yetirank {
            if let (Some(spans), Some(seeds)) =
                (group_spans.as_mut(), yetirank_tree_seeds.as_ref())
            {
                for (gi, span) in spans.iter_mut().enumerate() {
                    if let Some((begin, end, weight, relevs)) = yetirank_groups.get(gi) {
                        // Deriv competitors are sampled from the LEARNING-fold approx
                        // (the gradient/structure fold), NOT the averaging-fold
                        // `approx` (which drives leaf values). They coincide at tree 0
                        // (both == bias) and diverge after.
                        // WR-04 (06.3-17): an out-of-range `[begin, end)` learning-fold
                        // span MUST be a typed error, not a silently dropped group (the
                        // prior `unwrap_or_default()` would yield an EMPTY competitor set
                        // and corrupt the gradient without any signal).
                        let raw_approx: Vec<f64> = learn_approx
                            .get(*begin..*end)
                            .map(<[f64]>::to_vec)
                            .ok_or_else(|| {
                                CbError::OutOfRange(format!(
                                    "YetiRank deriv re-sample: group {gi} span [{begin}, {end}) \
                                     is out of range for learn_approx (len {})",
                                    learn_approx.len()
                                ))
                            })?;
                        let query_seed = seeds.deriv.get(gi).copied().unwrap_or(0);
                        span.competitors = crate::yetirank_sample_pairs(
                            &raw_approx,
                            relevs,
                            *weight,
                            yetirank_permutations,
                            yetirank_decay,
                            query_seed,
                        );
                    }
                }
            }
        }
        // STRUCTURE-fold cycling (Task 4): THIS iteration's learning fold.
        // `taken_fold = struct_fold_cycle[iter]` (defaulting to 0); hoisted above
        // the derivative computation because the SEARCH derivatives come from the
        // taken fold's own approx (fold-approx semantics, see `fold_approxes`).
        let taken_fold = struct_fold_cycle.get(iter).copied().unwrap_or(0);

        // 1. Per-object derivatives (UN-reduced; D-02) via the runtime kernel.
        //    `approx` is the DIMENSION-MAJOR flat buffer `approx[d*n+i]` of length
        //    `approx_dimension * n` (Plan 06.2-02). The backend runs an OUTER
        //    per-dimension loop over `approx[d*n..d*n+n]` reusing the existing
        //    per-loss kernel launchers; at `approx_dimension == 1` this is
        //    byte-identical to the pre-6.2 scalar path (RESEARCH Pitfall 1). The
        //    returned `der1`/`der2` are the matching dimension-major buffers.
        let ders = if let Some(spans) = group_spans.as_deref() {
            // GROUPED (ranking) der (LOSS-04, D-6.3-03): route through the grouped
            // seam over the per-fit `QueryInfo` view instead of the pointwise
            // per-object kernel. The seam returns one `Derivatives` per group (in
            // group order); the groups are contiguous half-open `[begin, end)`
            // spans covering all `n` objects in order, so concatenating their
            // der1/der2 in group order reproduces the object-order flat buffer the
            // pointwise path emits (approx_dimension == 1 for the Wave-A querywise
            // losses). The querywise der is ALREADY weighted (the per-object weight
            // is folded into der1/der2 INSIDE the per-group derivative function), so
            // the `weighted_der1` computation below uses this buffer AS-IS for the
            // grouped path — re-multiplying by `weights[i]` would double-weight the
            // gradient (CR-02, 06.3-07). The pointwise branch applies `der1 * weight`.
            // YetiRank: the GRADIENT (structure/split) der is computed over the
            // LEARNING-fold approx (`learn_approx`), the fold whose competitors were
            // just re-sampled above; every other grouped loss uses the single
            // `approx` (byte-identical, `is_yetirank` false).
            let grad_approx: &[f64] = if is_yetirank { &learn_approx } else { &approx };
            let per_group =
                runtime.compute_gradients_grouped(&params.loss, grad_approx, target, &weights, spans, stochasticrank_deriv_seed)?;
            let mut der1 = Vec::with_capacity(n);
            let mut der2 = Vec::with_capacity(n);
            for g in &per_group {
                der1.extend_from_slice(&g.der1);
                der2.extend_from_slice(&g.der2);
            }
            // The grouped der must cover every object exactly once (contiguous
            // groups). A shortfall would silently truncate the histogram; reject it.
            if der1.len() != n || der2.len() != n {
                return Err(CbError::Degenerate(format!(
                    "grouped der produced {} der1 / {} der2 entries, expected {n} \
                     (group spans must cover every object exactly once)",
                    der1.len(),
                    der2.len()
                )));
            }
            Derivatives { der1, der2 }
        } else if exp_ctr {
            // LEAF-VALUE derivatives from the AVERAGING fold's EXP-domain approx
            // (`CalcApproxesLeafwise` reads `AveragingFold.BodyTailArr[0].Approx`,
            // which upstream stores as exp for these losses and advances through
            // the approximate `fmath::expd_v` pipeline — see `avg_exp_approx`).
            // The linear `approx` stays the metrics/output stream (AvrgApprox).
            let mut der1 = Vec::with_capacity(n);
            let mut der2 = Vec::with_capacity(n);
            for (i, &e) in avg_exp_approx.iter().enumerate() {
                let t = target.get(i).copied().unwrap_or(0.0);
                let (d1, d2) = crate::fast_approx::logloss_ders_exp(e, t);
                der1.push(d1);
                der2.push(d2);
            }
            Derivatives { der1, der2 }
        } else {
            runtime.compute_gradients(&params.loss, &approx, target, approx_dimension)?
        };

        // Weighted gradient contribution per object (the histogram-scatter
        // elementwise product; the host reduces it ordered). The weight handling
        // branches on whether this iteration routed the GROUPED (ranking) der seam
        // (`group_spans.is_some()`) or the pointwise per-object kernel (CR-02,
        // 06.3-07):
        //
        //   * GROUPED ranking ders (QueryRMSE, QuerySoftMax, and any future
        //     querywise loss) ALREADY fold the per-object weight INSIDE the
        //     per-group derivative function — QueryRMSE returns
        //     `der1 = (target - approx - query_avrg) * weight`
        //     (loss.rs queryrmse_der), QuerySoftMax returns
        //     `der1 = beta * (-sumWTargets * p + weight * target)`
        //     (loss.rs querysoftmax_der, where the softmax probability `p` also
        //     carries the weight in its numerator). Re-multiplying by `weights[i]`
        //     here would DOUBLE-WEIGHT the gradient (squared weights → corrupt
        //     split scores and leaf values). So the grouped path uses `ders.der1`
        //     as-is. At the uniform-weight (w=1.0) oracle fixtures this is
        //     numerically identical to `der1 * 1.0`, which is why the bug was
        //     invisible there; non-uniform weights expose it
        //     (grouped_weight_regression_test).
        //
        //   * POINTWISE losses do NOT pre-weight their der, so the per-object
        //     weight is applied HERE. DIMENSION-MAJOR: `ders.der1` is length
        //     `approx_dimension * n`; each dimension's slice `der1[d*n + i]` is
        //     weighted by the per-OBJECT weight `weights[i]` (weights are
        //     per-object, shared across dimensions). At `approx_dimension == 1`
        //     the index `d*n + i` collapses to `i`, so this is exactly
        //     `der1.iter().zip(weights)` — byte-identical (Pitfall 1, D-04).
        let weighted_der1: Vec<f64> = if group_spans.is_some() {
            ders.der1.clone()
        } else {
            ders.der1
                .iter()
                .enumerate()
                .map(|(idx, &d)| {
                    let i = idx % n;
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    d * w
                })
                .collect()
        };

        // FOLD-APPROX SEARCH DERIVATIVES (Plain CTR path). The structure search —
        // and everything scoped to it: the bootstrap/MVS per-object der, the
        // sampled score buffers, and the random-strength score std-dev
        // (`CalcDerivativesStDevFromZeroPlainBoosting` reads the fold passed to
        // `GreedyTensorSearch`) — consumes the TAKEN learning fold's derivatives,
        // computed from that fold's own approx. The LEAF path below keeps the
        // averaging `ders`/`weighted_der1` untouched. `fold_approxes` is empty on
        // every non-CTR path and the grouped seam never co-occurs with CTR
        // candidates, so `search_weighted_der1` aliases `weighted_der1` there —
        // byte-identical.
        let search_ders_owned: Option<Derivatives> = match fold_approxes.get(taken_fold) {
            Some(fold_approx) if group_spans.is_none() => Some(if exp_ctr {
                // `CalcWeightedDerivatives` on the fold's EXP-domain approx:
                // `p = 1 - 1/(1+e)` (the upstream rounding order, see
                // `fast_approx::logloss_ders_exp`).
                let mut der1 = Vec::with_capacity(n);
                let mut der2 = Vec::with_capacity(n);
                for (i, &e) in fold_approx.iter().enumerate() {
                    let t = target.get(i).copied().unwrap_or(0.0);
                    let (d1, d2) = crate::fast_approx::logloss_ders_exp(e, t);
                    der1.push(d1);
                    der2.push(d2);
                }
                Derivatives { der1, der2 }
            } else {
                runtime.compute_gradients(&params.loss, fold_approx, target, approx_dimension)?
            }),
            _ => None,
        };
        let search_weighted_der1_owned: Option<Vec<f64>> = search_ders_owned.as_ref().map(|sd| {
            sd.der1
                .iter()
                .enumerate()
                .map(|(idx, &d)| {
                    let i = idx % n;
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    d * w
                })
                .collect()
        });
        let search_weighted_der1: &[f64] =
            search_weighted_der1_owned.as_deref().unwrap_or(&weighted_der1);

        // EFFECTIVE histogram / leaf weight (LOSS-04, 06.3-09): for a
        // pairwise-loss (`UsesPairsForCalculation` — PairLogit / PairLogitPairwise /
        // YetiRank{,Pairwise}) the split-scoring histogram `sumWeight` and the
        // Gradient leaf `sumWeight` are the per-object PAIRWISE weights
        // (`bt.PairwiseWeights` = Σ competitor.weight incident on the object;
        // `scoring.cpp:275-279`, `approx_calcer.cpp:444`), NOT the per-object sample
        // weight (which is `1.0` here). The der1 already carries the pair weight
        // (`competitor.weight`), so ONLY the `sumWeight` denominator changes. For
        // YetiRank the competitors are re-sampled per iteration above, so this is
        // recomputed from the CURRENT `group_spans` each tree. For every NON-pairwise
        // loss `eff_weights` IS the per-object `weights` (byte-identical, D-04).
        let eff_weights: Vec<f64> = if uses_pairwise_weights(&params.loss) {
            calc_pairwise_weights(group_spans.as_deref().unwrap_or(&[]), n)
        } else {
            weights.clone()
        };

        // 1a. PRE-bootstrap per-iteration draws (train.cpp:208,211): keep the RNG
        //     phase-aligned with upstream before the per-tree Bootstrap.
        if draws_active {
            for _ in 0..PRE_TREE_DRAWS {
                rng.gen_rand();
            }
        }

        // CR-02 (06.2-07): the sampling path operates PER OBJECT on the L2 norm
        // of the multi-dimensional gradient, not on the dim-major buffer.
        // Upstream (`mvs.cpp:50-55` `CalculateMeanGradValue`,
        // `greedy_tensor_search.cpp:92-107`) aggregates each object's gradient
        // across dimensions: `der_obj[i] = sqrt(Σ_d weighted_der1[d*n+i]²)`. This
        // per-object vector (length `n`, NOT `dim*n`) is what `bootstrap` /
        // `mvs_lambda` consume so `set_sampled_control` / `mvs_sample_weights`
        // draw and mask PER OBJECT (the RNG phase advances by `n`, not `dim*n`).
        // At `approx_dimension == 1` this is `sqrt(wd²) == |wd|` and the buffer
        // already has length `n`, so the scalar bootstrap/MVS inputs are
        // byte-identical to before (D-04). The per-dim squares route through the
        // sanctioned ordered `sum_f64` (D-08).
        // Fold-approx semantics: the bootstrap/MVS input is the SEARCH (taken
        // learning fold) derivative — `search_weighted_der1` aliases
        // `weighted_der1` on every non-CTR path.
        let der_obj: Vec<f64> = (0..n)
            .map(|i| {
                let squares: Vec<f64> = (0..approx_dimension)
                    .map(|d| {
                        let v = search_weighted_der1.get(d * n + i).copied().unwrap_or(0.0);
                        v * v
                    })
                    .collect();
                sum_f64(&squares).sqrt()
            })
            .collect();

        // 1b. Bootstrap / sampling (TRAIN-04): once per tree, on the continuous
        //     RNG. MVS reads the per-OBJECT derivatives; the others ignore them.
        let sampled = bootstrap(
            params.bootstrap_type,
            &der_obj,
            params.subsample,
            params.bagging_temperature,
            prev_leaf_mean_l2,
            &mut rng,
        )?;

        // WR-04 (06.2-07): after CR-02 the sample weight / control mask are
        // per-OBJECT (length `n`). Assert it so a future dimension-major
        // regression is caught here rather than silently truncated by the
        // dim-major `zip` below (which would re-introduce CR-02).
        debug_assert_eq!(
            sampled.sample_weights.len(),
            n,
            "sample weights must be per-object (length n), not dim-major"
        );
        debug_assert_eq!(
            sampled.control.len(),
            n,
            "control mask must be per-object (length n), not dim-major"
        );

        // The SAMPLE WEIGHTS and CONTROL mask affect ONLY the SPLIT SCORING
        // (the `sampledDocs` histogram path); LEAF VALUES are estimated on the
        // FULL, UN-sampled AveragingFold derivatives (verified against upstream:
        // Bayesian/MVS sample weights never enter `CalcLeafValues`). So:
        //   * SCORE path: der1*weight*sampleWeight, restricted to control-true
        //     objects (zero score weight excludes an object from the ordered
        //     histogram reduction, exactly as `sampledDocs` drops it).
        //   * LEAF path: the raw weighted_der1 / weights (no sampling) —
        //     unchanged from the first slice.
        // The per-OBJECT sample weight / control (length `n`) is shared across
        // ALL dimensions of the dim-major score buffer: object `i`'s weight
        // `sample_weights[i]` multiplies every dimension `d*n + i`
        // (tensor_search_helpers.cpp:468-472 — the same per-object weight the
        // leaf path already shares). At `approx_dimension == 1`, `idx % n == idx`
        // so this is byte-identical to the prior per-object zip (D-04).
        let score_weighted_der1: Vec<f64> = search_weighted_der1
            .iter()
            .enumerate()
            .map(|(idx, &d)| {
                let i = idx % n;
                let sw = sampled.sample_weights.get(i).copied().unwrap_or(1.0);
                let c = sampled.control.get(i).copied().unwrap_or(true);
                if c {
                    d * sw
                } else {
                    0.0
                }
            })
            .collect();
        // `score_weights` stays per-OBJECT (length `n`): the histogram weight is
        // per object, masked/scaled by the per-object sample weight. For a
        // pairwise loss the per-object weight is the PAIRWISE weight `eff_weights`
        // (`bt.PairwiseWeights`), upstream `scoring.cpp:276-279`
        // (`hasPairwiseWeights ? bt.PairwiseWeights : fold.LearnWeights`); for every
        // other loss `eff_weights == weights` so this is byte-identical (D-04).
        let score_weights: Vec<f64> = eff_weights
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let sw = sampled.sample_weights.get(i).copied().unwrap_or(1.0);
                let c = sampled.control.get(i).copied().unwrap_or(true);
                if c {
                    w * sw
                } else {
                    0.0
                }
            })
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
        // L2 scaling uses `sumAllWeights / docCount` (`CalcDeltaNewtonBody`,
        // `online_predictor.h:126`). `sumAllWeights` is upstream's
        // `bt.BodySumWeight`, which `fold.cpp:170-172` defines as the sum of the
        // per-object LEARN WEIGHTS (or `bodyFinish == docCount` when weights are
        // empty) — the SAME for both the split-scoring L2 (`scoring.cpp:747-749`,
        // `BodyTailArr[..].BodySumWeight`) and the leaf-value Newton/Gradient L2
        // (`approx_calcer.cpp:811`, `bt.BodySumWeight`). It is the PER-OBJECT weight
        // sum, NOT the pairwise-weight total — even for a pairwise loss. (06.3-13
        // instrumented ground truth `PairLogit/per_leaf_der_log.jsonl`:
        // `sum_all_weights == all_doc_count == 12`, the document count, with the
        // Newton denom `-SumDer2 + l2*(12/12)`; the 06.3-09 `sum_eff_weights`
        // pairwise-total scaling was wrong and diverged the Splits at index 6.) The
        // PAIRWISE weight enters ONLY the histogram `sumWeight` (`score_weights` /
        // `eff_weights`, `scoring.cpp:276-279`), never the L2 scaling. `docCount`
        // stays `n`. For every non-pairwise loss this is byte-identical (D-04).
        let scaled_l2 = scale_l2_reg(params.l2_leaf_reg, sum_all_weights, n);
        // Widened from `perturb_active` to `draws_active` (2026-07-30, real
        // instrumented-upstream ground truth — see the `POST_TREE_EXTRA_DRAWS`
        // doc comment and `.planning/plans/bayesian-rng-draw-accounting/
        // instrumented-ground-truth/GROUND_TRUTH.md`): upstream's per-level RSM
        // reselection + `SelectBestCandidate` draws happen UNCONDITIONALLY
        // whenever sampling is active, even at `random_strength == 0`
        // (`score_st_dev = 0.0` then makes the perturbation a numeric no-op —
        // `val + std_normal(rng) * 0.0 == val` — so the CHOSEN split/leaf is
        // unaffected; only the RNG phase entering the NEXT tree's `Bootstrap`
        // call changes). When `draws_active` is false (bootstrap_type=No AND
        // random_strength=0) `perturb` stays `None` and the search remains the
        // byte-identical, zero-draw first-slice path — untouched.
        let perturb = if draws_active {
            let model_length = iter as f64 * learning_rate;
            // CR-02: std-dev sums `wd²` over the FULL dim-major buffer but
            // divides by the per-OBJECT count `n` (NOT `dim*n`); the `ln(n)`
            // model-size multiplier likewise uses `n` (greedy_tensor_search.cpp:
            // 106, 125). At dim=1, `weighted_der1.len() == n` so this is
            // byte-identical to the prior call (D-04).
            let std_dev = if perturb_active {
                // Fold-approx semantics: the score std-dev reads the fold handed
                // to the search (`CalcDerivativesStDevFromZeroPlainBoosting`) —
                // aliases `weighted_der1` on every non-CTR path.
                score_st_dev(params.random_strength, search_weighted_der1, n, model_length)
            } else {
                0.0
            };
            Some(Perturbation {
                rng: &mut rng,
                score_st_dev: std_dev,
            })
        } else {
            None
        };
        // CTR-aware structure search is taken when there ARE materialized CTR
        // candidates (the cat path). It is mutually exclusive with the Ordered
        // path here (the in-scope tensor_ctr_e2e config is Plain + hasCtrs); the
        // numeric / one-hot / ordered paths have NO CTR candidates so this gate is
        // false for them and they keep their exact previous dispatch.
        let has_ctr = !materialized_ctr_features.is_empty();
        // STRUCTURE-fold cycling (Task 4): select THIS iteration's learning fold's
        // structure CTR columns (`taken_fold` was hoisted above the derivative
        // computation — the search derivatives come from that fold's approx). For
        // learning_folds==1 the cycle is all-zeros, so this is always
        // `structure_fold_columns[0]` == the prior fixed `materialized_ctr_features`
        // (byte-identical). For pc=4 it cycles `[0,2,0,2,2]`, materializing the tree
        // STRUCTURE under fold 0 (borders [7,2]) or fold 2 (borders [3,7]) per iter.
        let iter_ctr_features: &[crate::ctr::CtrFeatureColumn] = structure_fold_columns
            .get(taken_fold)
            .map_or(materialized_ctr_features.as_slice(), Vec::as_slice);
        // GROWER DISPATCH (FEAT-06 / D-6.6-04): grow_policy selects the tree-growth
        // strategy. The SymmetricTree arm is the LITERAL pre-6.6 oblivious grower
        // chain UNCHANGED (byte-identical, D-6.6-05); Lossguide / Depthwise dispatch
        // to the policy-parameterized leaf-wise grower producing a TRUE non-symmetric
        // node graph (the structure half of FEAT-06 — leaf VALUES + the apply
        // pointer-walk land in 06.6-05). Region is rejected up front by
        // `validate_grow_policy`, so it never reaches here.
        let grown: GrownTree = match params.grow_policy {
            // GPUT-18 / D-03a: Region grows a single PATH (d+1 leaves) via the
            // dedicated `region_grower` — NOT the leaf-wise node-graph grower
            // (Pitfall 2) and NOT the oblivious grower. It scores against the SAME
            // perturbation-free whole-fold der/weights the plain path uses (the
            // in-scope Region fixtures pin random_strength=0 + bootstrap_type=No).
            EGrowPolicy::Region => region_grower(
                &matrix,
                &score_weighted_der1,
                &score_weights,
                scaled_l2,
                params.depth,
                params.min_data_in_leaf,
                n,
                params.score_function,
            )?,
            EGrowPolicy::Lossguide | EGrowPolicy::Depthwise => {
                let policy = if params.grow_policy == EGrowPolicy::Depthwise {
                    LeafWisePolicy::Depthwise
                } else {
                    LeafWisePolicy::Lossguide
                };
                // The leaf-wise grower scores against the SAME perturbation-free
                // whole-fold der/weights the oblivious plain path uses at
                // random_strength=0 + bootstrap_type=No (the in-scope non-symmetric
                // fixtures). Lock SPLITS first (Open Question 1); if the simplest
                // Depthwise preflight diverges once this lands, ESCALATE to the
                // instrumented trainer (D-6.6-11) — do NOT weaken.
                leaf_wise_grower(
                    policy,
                    &matrix,
                    &score_weighted_der1,
                    &score_weights,
                    scaled_l2,
                    params.depth,
                    params.max_leaves,
                    params.min_data_in_leaf,
                    n,
                    params.score_function,
                )?
            }
            // SymmetricTree: the literal existing oblivious grower chain, UNCHANGED
            // (Region now has its own arm above — GPUT-18).
            EGrowPolicy::SymmetricTree => {
            if is_pairwise_scoring(&params.loss) {
            // PAIRWISE SPLIT-SCORING (LOSS-04, Plan 06.3-16): `*Pairwise` losses
            // (`IsPairwiseScoring`) score candidate splits through upstream's
            // dedicated `TPairwiseScoreCalcer` / `CalculatePairwiseScore`
            // (`greedy_tensor_search.cpp:680-690`), NOT the pointwise L2/Cosine
            // der histogram. Wire the Plan 06.3-15 cb-compute scorer into the
            // greedy oblivious level search over the CURRENT leaf assignment + the
            // per-tree global competitor pairs (`group_spans`). `*Pairwise` forces
            // `boosting_type = Plain` + the corpus is float-only, so this is
            // mutually exclusive with the CTR / Ordered paths (no perturbation /
            // bootstrap draws). The der1 fed here is `weighted_der1` — for a
            // pairwise loss `group_spans.is_some()` so `weighted_der1 == ders.der1`
            // (the grouped der1 with the pair weight already folded), the SAME
            // buffer the pairwise LEAF path consumes (`pairwise_leaves.rs`).
            //
            // `l2_diag_reg = params.l2_leaf_reg` is the RAW l2 (NOT the
            // sumAllWeights-scaled `scaled_l2`): `scoring.cpp:809,844` passes
            // `ObliviousTreeOptions->L2Reg` UNSCALED to `CalculatePairwiseScore`,
            // matching the pairwise leaf path. `pairwise_bucket_weight_prior_reg =
            // PairwiseNonDiagReg` (`bayesian_matrix_reg`, default 0.1).
            let spans = group_spans.as_deref().unwrap_or(&[]);
            greedy_tensor_search_oblivious_pairwise(
                &matrix,
                &weighted_der1,
                spans,
                params.l2_leaf_reg,
                PAIRWISE_NON_DIAG_REG_DEFAULT,
                params.depth,
                n,
            )?
        } else if has_ctr {
            // ORD-05 STRUCTURE: score the SELECTED-fold CTR columns into the
            // oblivious search alongside float candidates (shared score, strict
            // first-wins, forward-bit leaf index). At random_strength=0 +
            // bootstrap_type=No there are no perturbation/bootstrap draws, so the
            // FULL (un-masked) `weighted_der1` / `weights` drive scoring. The
            // returned `grown.leaf_of` is the STRUCTURE partition; the leaf VALUES
            // are reassigned over the averaging-fold columns below.
            greedy_tensor_search_oblivious_with_ctr(
                &matrix,
                iter_ctr_features,
                ctr_border_count,
                // FOLD-APPROX SEMANTICS: the structure search consumes the TAKEN
                // learning fold's derivatives (`search_weighted_der1`), sampled/
                // masked exactly like the float arm (`score_weighted_der1` is
                // built FROM the search der above). At bootstrap_type=No +
                // random_strength=0 — every committed CTR fixture — the mask is
                // an element-wise no-op, so this equals the fold der verbatim.
                &score_weighted_der1,
                &score_weights,
                scaled_l2,
                params.depth,
                n,
                // model_size_reg cat-feature weight (GetCatFeatureWeight): the
                // default 0.5 down-weights high-cardinality (combination) CTR
                // candidates so a new {0,1} combination does not out-score a second
                // border on an already-used {0} simple CTR on a thin margin.
                model_size_reg_default(),
                params.score_function,
                &cat_eligible_buckets,
                &used_ctr_splits,
            )?
        } else {
            match ordered_learning_perm.as_deref() {
                // ORDERED (ORD-02): grow the tree STRUCTURE via the 05-08 ordered
                // per-segment split-scoring subsystem over the learning fold's
                // BodyTailArr. At random_strength=0 + bootstrap_type=No there are no
                // perturbation/bootstrap draws, so the ordered split score consumes
                // the FULL (un-masked) `weighted_der1` / `weights` in learning-fold
                // object order; the function derives the body/tail segments +
                // per-segment body sum-weights internally from `fold_len_multiplier`
                // (fold.rs, 05-03). `leaf_of` is object-order (Plain-identical) so
                // the SAME averaging-fold leaf-value path below applies.
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
                // PLAIN: the perturbed whole-fold search over the
                // sampled/sample-weighted histogram. FEAT-04 penalties are threaded
                // in via the `FeaturePenalties` context — multiplicative
                // `feature_weights` on the candidate gain + subtractive
                // `first_feature_use_penalties` / `per_object_feature_penalties`
                // (× `penalties_coefficient`) while a feature is globally unused
                // (RESEARCH Pitfall 6; the per-object term uses the whole-fold doc
                // count `n`). With all penalty vectors empty the context is `None`
                // and the search is byte-identical to the pre-6.6 path (D-6.6-05).
                None => {
                    let pen = if penalties_active {
                        Some(crate::tree::FeaturePenalties {
                            feature_weights: &params.feature_weights,
                            first_feature_use_penalties: &params.first_feature_use_penalties,
                            per_object_feature_penalties: &params.per_object_feature_penalties,
                            penalties_coefficient: params.penalties_coefficient,
                            used_features: &used_features,
                            doc_count: n,
                        })
                    } else {
                        None
                    };
                    greedy_tensor_search_oblivious_perturbed(
                        &matrix,
                        &score_weighted_der1,
                        &score_weights,
                        scaled_l2,
                        params.depth,
                        n,
                        perturb,
                        params.score_function,
                        pen.as_ref(),
                    )?
                }
            }
            }
            }
        };

        // `ProcessCtrSplit` (greedy_tensor_search.cpp:1126): every CTR split
        // this tree chose enters the model-lifetime `UsedCtrSplits` set, so the
        // NEXT tree's `GetCatFeatureWeight` scores the pair at weight 1.0
        // (within-tree lifting is handled inside `select_level_ctr_aware` off
        // `chosen`). De-duplicated — the set semantics of the upstream THashSet.
        for spec in &grown.ctr_splits {
            let key = (spec.ctr_type, spec.projection.clone());
            if !used_ctr_splits.contains(&key) {
                used_ctr_splits.push(key);
            }
        }

        // Per-tree leaf count: a non-symmetric leaf-wise tree has a DISTINCT leaf
        // count (number of terminal nodes), NOT `2^depth`. Shadow `n_leaves` for the
        // leaf-value estimation below so the reductions cover exactly this tree's
        // leaves. For the oblivious path `grown.step_nodes` is empty and this is the
        // unchanged `2^depth` (byte-identical, D-6.6-05).
        let n_leaves: usize = if !grown.region_directions.is_empty() {
            // GPUT-18: a depth-d region path has exactly d+1 leaves (bins 0..=depth).
            grown.region_directions.len() + 1
        } else if grown.step_nodes.is_empty() {
            n_leaves
        } else {
            grown
                .node_id_to_leaf_id
                .iter()
                .zip(grown.step_nodes.iter())
                .filter(|(_, &(l, r))| l == 0 && r == 0)
                .count()
        };

        // FEAT-04: mark this tree's float splits as "used" so subsequent trees no
        // longer pay the first-use / per-object penalty for those features
        // (`feature_penalties_calcer.cpp` — the penalty fires until a feature first
        // becomes used). Only consulted when penalties are active; the default path
        // leaves `used_features` untouched (D-6.6-05).
        if penalties_active {
            for split in &grown.splits {
                if let Some(flag) = used_features.get_mut(split.feature) {
                    *flag = true;
                }
            }
        }

        // YetiRank LEAF-VALUE competitor RE-SAMPLE (D-07, 06.3-14 ext): the live
        // trainer re-samples a DISTINCT YetiRank competitor set for the AveragingFold
        // leaf-value estimation (`CalcLeafValuesSimple` -> `CalcLeafDersSimple` ->
        // `YetiRankRecalculation`, `approx_calcer.cpp:983`), drawn off the SAME
        // persistent context RNG AFTER the tree structure is grown — a DIFFERENT seed
        // than the gradient/split recalc. Re-sample the competitors with the
        // leaf-value per-group seeds, then recompute the grouped der + pairwise
        // `eff_weights` so the leaf-value estimation rides the leaf-value competitor
        // stream (the gradient/split recalc above already consumed the deriv seeds).
        // For NON-YetiRank losses this block is skipped (the leaf-value path reuses
        // the gradient der/weights, byte-identical, D-04).
        let (lv_weighted_der1, lv_der2, lv_eff_weights): (Vec<f64>, Vec<f64>, Vec<f64>) =
            if is_yetirank {
                if let (Some(spans), Some(seeds)) =
                    (group_spans.as_mut(), yetirank_tree_seeds.as_ref())
                {
                    for (gi, span) in spans.iter_mut().enumerate() {
                        if let Some((begin, end, weight, relevs)) = yetirank_groups.get(gi) {
                            // WR-04 (06.3-17): typed error on an out-of-range span
                            // instead of a silent empty competitor set.
                            let raw_approx: Vec<f64> = approx
                                .get(*begin..*end)
                                .map(<[f64]>::to_vec)
                                .ok_or_else(|| {
                                    CbError::OutOfRange(format!(
                                        "YetiRank leaf-value re-sample: group {gi} span \
                                         [{begin}, {end}) is out of range for approx (len {})",
                                        approx.len()
                                    ))
                                })?;
                            let query_seed = seeds.leafval.get(gi).copied().unwrap_or(0);
                            span.competitors = crate::yetirank_sample_pairs(
                                &raw_approx,
                                relevs,
                                *weight,
                                yetirank_permutations,
                                yetirank_decay,
                                query_seed,
                            );
                        }
                    }
                }
                // Recompute the grouped der over the leaf-value competitors. The
                // grouped der already folds the pair weight (CR-02), so the leaf-value
                // `weighted_der1` is the raw grouped der1 (no per-object re-weight).
                let spans_ref = group_spans.as_deref().unwrap_or(&[]);
                let per_group = runtime.compute_gradients_grouped(
                    &params.loss,
                    &approx,
                    target,
                    &weights,
                    spans_ref,
                    params.random_seed,
                )?;
                let mut d1 = Vec::with_capacity(n);
                let mut d2 = Vec::with_capacity(n);
                for g in &per_group {
                    d1.extend_from_slice(&g.der1);
                    d2.extend_from_slice(&g.der2);
                }
                let eff = if uses_pairwise_weights(&params.loss) {
                    calc_pairwise_weights(spans_ref, n)
                } else {
                    weights.clone()
                };
                (d1, d2, eff)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        // StochasticRank LEAF-VALUE der RE-COMPUTE (D-07, 06.3-18): the live trainer
        // re-runs the Monte-Carlo querywise der for the AveragingFold leaf-value
        // estimation (`CalcLeafValuesSimple` -> `CalcLeafDersSimple`), drawing a FRESH
        // per-group Gaussian noise stream re-seeded with `leafval_recalc_seed +
        // group_index` — a DIFFERENT base than the gradient/split recalc. Unlike
        // YetiRank there are NO competitors to re-sample; the only thing that changes
        // is the noise seed fed into `compute_gradients_grouped`. Recompute the grouped
        // der over the SAME averaging-fold `approx` with the leaf-value recalc base so
        // the leaf-value estimation rides the leaf-value noise stream (the gradient
        // recalc above already consumed the deriv base). For NON-StochasticRank losses
        // this block is skipped (the leaf-value path reuses the gradient der, D-04).
        let (srank_lv_der1, srank_lv_der2): (Vec<f64>, Vec<f64>) = if is_stochasticrank {
            let spans_ref = group_spans.as_deref().unwrap_or(&[]);
            let per_group = runtime.compute_gradients_grouped(
                &params.loss,
                &approx,
                target,
                &weights,
                spans_ref,
                stochasticrank_leafval_seed,
            )?;
            let mut d1 = Vec::with_capacity(n);
            let mut d2 = Vec::with_capacity(n);
            for g in &per_group {
                d1.extend_from_slice(&g.der1);
                d2.extend_from_slice(&g.der2);
            }
            if d1.len() != n || d2.len() != n {
                return Err(CbError::Degenerate(format!(
                    "StochasticRank leaf-value der produced {} der1 / {} der2 entries, \
                     expected {n} (group spans must cover every object exactly once)",
                    d1.len(),
                    d2.len()
                )));
            }
            (d1, d2)
        } else {
            (Vec::new(), Vec::new())
        };

        // Select the der/weights the leaf-value estimation reads: the YetiRank
        // leaf-value re-sample, the StochasticRank leaf-value der re-compute, or (every
        // other loss) the gradient buffers. StochasticRank's grouped der ALREADY folds
        // the per-object weight (CR-02), so its leaf-value `der1` is the raw grouped
        // der1 (no per-object re-weight) and the leaf weights are the per-object
        // `eff_weights` (pointwise grouped path, no pairwise weights).
        let lv_weighted_der1: &[f64] = if is_yetirank {
            &lv_weighted_der1
        } else if is_stochasticrank {
            &srank_lv_der1
        } else {
            &weighted_der1
        };
        let lv_der2: &[f64] = if is_yetirank {
            &lv_der2
        } else if is_stochasticrank {
            &srank_lv_der2
        } else {
            &ders.der2
        };
        let lv_eff_weights: &[f64] = if is_yetirank { &lv_eff_weights } else { &eff_weights };

        // LEAF-VALUE leaf_of (research Q1/Q3 #3, train.cpp:130
        // BuildIndices(AveragingFold)). On the CTR path, the per-object leaf indices
        // for LEAF-VALUE estimation are computed over the AVERAGING-fold CTR columns
        // (NOT the structure-search columns), reassigning each CTR level's
        // `ctr_bin > border` test against the averaging column's bins while keeping
        // float levels on the float matrix. On every OTHER path (no CTR candidates)
        // `leaf_value_leaf_of` is EXACTLY the structure `grown.leaf_of`
        // (byte-identical to before — the numeric / one-hot / ordered oracles are
        // provably unaffected by the gate below).
        let leaf_value_leaf_of: Vec<usize> = if has_ctr {
            assign_leaf_over_ctr_columns(&matrix, &averaging_ctr_features, &grown, n)
        } else {
            grown.leaf_of.clone()
        };

        // 3. Leaf values via the selected estimation method (TRAIN-03 / D-09),
        //    scaled by learning_rate (stored value matches model.json). Leaf
        //    estimation uses the FULL fold (all objects) with the RAW (un-sampled)
        //    derivatives/weights over the LEAF-VALUE leaf_of (the averaging-fold
        //    partition on the CTR path; the structure partition otherwise). The
        //    Gradient FORMULA is UNCHANGED (research Q3 #4). Every reduction over
        //    leaf members routes through cb_core::sum_f64 (D-05).
        //
        //    DIMENSION-MAJOR (Plan 06.2-02): solve each output dimension `d`
        //    INDEPENDENTLY over its own approx/der slice `[d*n .. d*n+n]`, reusing
        //    the EXISTING per-dimension scalar solver `compute_leaf_deltas`. The
        //    per-dimension reduction is an OUTER `for d` loop (NEVER fused into a
        //    single `0..dim*n` reduction) so at `approx_dimension == 1` the slices
        //    are exactly today's full-`n` buffers and the `cb_core::sum_f64`
        //    summation order is byte-identical (RESEARCH Pitfall 1). The leaf
        //    VALUES are stored dimension-major `leaf_values[d*n_leaves + l]`
        //    (length `dim*n_leaves`); at dim=1 this is exactly `n_leaves` values
        //    in leaf order (unchanged). The leaf_value leaf_of partition is shared
        //    across dimensions (the oblivious structure is one tree).
        let mut leaf_values: Vec<f64> = Vec::with_capacity(approx_dimension * n_leaves);
        // NOTE: each leaf branch below pushes RAW deltas (NO learning_rate); the
        // `learning_rate` scale + pairwise weighted-mean centering are applied once,
        // after the branches, by `normalize_leaf_values` (upstream order).
        if is_pairwise_scoring(&params.loss) {
            // PAIRWISE leaf path (LOSS-04 Wave B): the `*Pairwise` losses
            // (`IsPairwiseScoring`) solve their leaf VALUES via the Cholesky
            // pairwise-leaf system (`pairwise_leaves.rs`) over the per-leaf
            // pairwise weight sums (from the group Competitors) + der sums — NOT
            // the pointwise Gradient/Newton estimators (RESEARCH Pitfall 2). This
            // is the THIRD leaf branch, kept separate from the pointwise and
            // softmax paths. `*Pairwise` is single-dimension (approx_dimension == 1)
            // and `boost_from_average=false`, so this writes exactly `n_leaves`
            // values. The der1 fed here is the RAW (un-weighted) PairLogit der1
            // (`ders.der1`) — the pair weight already lives inside the der, so the
            // per-leaf `SumDer` is a plain reduction (upstream `leafDers[leaf].SumDer`,
            // approx_calcer.cpp:495). `l2_diag_reg = L2Reg` is the RAW l2 (NOT the
            // sumAllWeights-scaled `scaled_l2` the pointwise path uses —
            // CalcLeafDeltasSimple passes `params.ObliviousTreeOptions->L2Reg`
            // directly). `pairwise_bucket_weight_prior_reg = PairwiseNonDiagReg`
            // (`bayesian_matrix_reg`, default 0.1).
            let spans = group_spans.as_deref().unwrap_or(&[]);
            // YetiRankPairwise: the leaf-value der + Competitors are the leaf-value
            // re-sample (`lv_weighted_der1` == the grouped der1, pair weight already
            // folded); the `spans[*].competitors` were re-sampled with the leaf-value
            // seeds above. PairLogit{,Pairwise} (non-YetiRank) reuse `ders.der1`
            // (byte-identical — `is_yetirank` is false, D-04).
            let pairwise_der1: &[f64] = if is_yetirank { lv_weighted_der1 } else { &ders.der1 };
            let deltas = crate::pairwise_leaves::compute_pairwise_leaf_deltas(
                spans,
                &leaf_value_leaf_of,
                pairwise_der1,
                n_leaves,
                params.l2_leaf_reg,
                PAIRWISE_NON_DIAG_REG_DEFAULT,
            );
            // RAW deltas (NO learning_rate yet): the `NormalizeLeafValues`
            // weighted-mean centering + `learning_rate` scale are applied below,
            // matching upstream `train.cpp:562` (NormalizeLeafValues runs AFTER the
            // estimator, lr applied LAST inside it). The Cholesky path already did
            // its own simple `MakeZeroAverage`; the weighted `NormalizeLeafValues`
            // is the second, doc-weight-weighted centering upstream applies on top.
            leaf_values.extend_from_slice(&deltas);
        } else if matches!(params.loss, Loss::MultiClass) {
            // MultiClass softmax: the COUPLED per-leaf symmetric Newton solve over
            // ALL dimensions at once (`ders.der2` is the PER-OBJECT packed Hessian
            // of length `n * k*(k+1)/2`, NOT the diagonal `der2[d*n+i]` layout).
            // Produces the dimension-major leaf deltas; scaled by learning_rate
            // into the same `leaf_values[d*n_leaves + leaf]` layout the diagonal
            // path emits.
            let deltas = compute_softmax_leaf_deltas(
                &leaf_value_leaf_of,
                &weighted_der1,
                &ders.der2,
                &weights,
                scaled_l2,
                n_leaves,
                approx_dimension,
                n,
            );
            // RAW deltas (NO learning_rate yet); MultiClass is not a pairwise loss
            // so `NormalizeLeafValues` below only applies the lr scale (byte-identical
            // to the prior `learning_rate * delta`, D-04).
            leaf_values.extend_from_slice(&deltas);
        } else {
            // Diagonal / separable losses (every scalar loss AND MultiClassOneVsAll):
            // solve each output dimension INDEPENDENTLY over its own approx/der slice
            // `[d*n .. d*n+n]`, reusing the EXISTING per-dimension scalar solver. The
            // per-dimension reduction is an OUTER `for d` loop (NEVER fused) so at
            // `approx_dimension == 1` the slices are exactly today's full-`n` buffers
            // and the `cb_core::sum_f64` summation order is byte-identical (Pitfall 1).
            // MultiClassOneVsAll's diagonal Newton step equals the scalar Logloss
            // Newton arm per dimension.
            for d in 0..approx_dimension {
                let base = d * n;
                // YetiRank: the leaf-value der is the leaf-value-competitor re-sample
                // (`lv_weighted_der1`/`lv_der2`); every other loss reuses the gradient
                // buffers (byte-identical, `lv_* == weighted_der1`/`ders.der2`, D-04).
                let der1_d = lv_weighted_der1.get(base..base + n).unwrap_or(&[]);
                let der2_d = lv_der2.get(base..base + n).unwrap_or(&[]);
                let approx_d = approx.get(base..base + n).unwrap_or(&[]);
                // LEAF-VALUE weights (REVIEW WR-03 / T-06.3-14): the pointwise
                // estimator re-weights the per-object `der2` (Newton) / `sum_weight`
                // (Gradient) by this vector. The correct grouped-path weight depends
                // on the LEAF METHOD, NOT uniformly on `group_spans.is_some()`:
                //
                //   * NEWTON leaf (`SumDer / (-SumDer2 + scaledL2)`): the der1/der2
                //     ALREADY fold the pair weight (`competitor.weight`) inside the
                //     per-group der — upstream Newton (`AddDerDer2`) consumes
                //     `der.Der1/der.Der2` verbatim with NO extra weight
                //     (`approx_calcer_helpers.h`). Re-weighting would double-count
                //     (the der2 analogue of CR-02 06.3-07) → UNIT weights.
                //
                //   * GRADIENT leaf (`CalcAverage(SumDer, SumWeights, scaledL2)`):
                //     upstream `UsesPairsForCalculation` makes the leaf `sumWeight`
                //     the PAIRWISE weight `bt.PairwiseWeights` (= Σ competitor.weight
                //     incident on the object), NOT the doc count
                //     (`approx_calcer.cpp:444`, `CalcLeafValues`). YetiRank
                //     (non-Pairwise) rides this Gradient leaf, so its denominator
                //     `sumWeight + scaledL2` must use `eff_weights` (the pairwise
                //     sumWeight) — otherwise the doc-count `sumWeight` mixes
                //     inconsistently with the pairwise-weight-scaled `scaled_l2`
                //     built above (REVIEW WR-03).
                //
                // For the POINTWISE path (`group_spans.is_none()`) `eff_weights` IS
                // the per-object `weights` (byte-identical, D-04). The deciding
                // predicate is `uses_pairwise_weights(loss) && method == Gradient`:
                // only a pairwise-weight loss on the Gradient leaf needs the pairwise
                // sumWeight; every other grouped case (e.g. LambdaMart/QueryRMSE
                // Newton) keeps unit/per-object weights.
                let unit_weights: Vec<f64> = vec![1.0; n];
                let grouped_gradient_pairwise = group_spans.is_some()
                    && matches!(params.leaf_method, LeafMethod::Gradient)
                    && uses_pairwise_weights(&params.loss);
                let leaf_weights_for_deltas: &[f64] = if group_spans.is_some() {
                    if grouped_gradient_pairwise {
                        // YetiRank Gradient leaf: pairwise sumWeight (bt.PairwiseWeights),
                        // from the LEAF-VALUE competitor re-sample (`lv_eff_weights`).
                        lv_eff_weights
                    } else {
                        // Newton (or non-pairwise grouped) leaf: der already folds the
                        // pair weight → unit weights (no double-count).
                        &unit_weights
                    }
                } else {
                    // Pointwise path: per-object weights (eff_weights == weights, D-04).
                    &eff_weights
                };
                let leaf_deltas = compute_leaf_deltas(
                    params.leaf_method,
                    &params.loss,
                    &leaf_value_leaf_of,
                    der1_d,
                    der2_d,
                    leaf_weights_for_deltas,
                    approx_d,
                    target,
                    scaled_l2,
                    n_leaves,
                    d,
                );
                // FEAT-03 monotone post-pass (D-6.6-06): project the RAW per-leaf
                // deltas onto the monotone cone implied by `monotone_constraints`
                // via the isotonic (PAVA) leaf-value pass
                // (`CalcMonotonicLeafDeltasSimple`, `approx_calcer.cpp:551`). This
                // runs AFTER the structure-built leaf estimator and BEFORE the
                // `learning_rate`/centering `NormalizeLeafValues`, exactly where
                // upstream inserts it (the deltas it adjusts are the raw,
                // pre-learning-rate ones). It is OBLIVIOUS-ONLY (this is the
                // symmetric grower) and a NO-OP when `monotone_constraints` is empty,
                // so the default leaf path stays byte-identical (D-6.6-05). The
                // current within-iteration leaf totals are 0 (leaf_estimation
                // iterations == 1, `curr == 0`).
                let leaf_deltas = if params.monotone_constraints.is_empty() {
                    leaf_deltas
                } else {
                    let tree_monotone = tree_monotone_constraints(
                        &grown.splits,
                        &params.monotone_constraints,
                    );
                    if tree_monotone.iter().all(|&c| c == 0) {
                        leaf_deltas
                    } else {
                        let iso_weights = monotonic_leaf_isotonic_weights(
                            params.leaf_method,
                            &leaf_value_leaf_of,
                            der1_d,
                            der2_d,
                            leaf_weights_for_deltas,
                            scaled_l2,
                            n_leaves,
                        );
                        let curr_zero = vec![0.0_f64; n_leaves];
                        cb_compute::calc_monotonic_leaf_deltas(
                            &tree_monotone,
                            &curr_zero,
                            &leaf_deltas,
                            &iso_weights,
                        )
                    }
                };
                // RAW deltas (NO learning_rate yet); see `NormalizeLeafValues` below.
                leaf_values.extend_from_slice(&leaf_deltas);
            }
        }

        // Per-leaf summed training-document weights (RESEARCH Pitfall 1; research
        // Open-q 5: on the CTR path these are the AVERAGING-fold partition counts).
        // Uses the FULL un-sampled fold weights (same as leaf estimation) over the
        // SAME `leaf_value_leaf_of`, reduced ordered through cb_core::sum_f64 (D-08).
        // Leaf WEIGHTS are one-per-leaf (NOT per-dimension — the document partition
        // is shared across output dimensions), so this is unchanged at any dim.
        //
        // 06.3-13: `leaf_weights` is upstream's `SumLeafWeights(GetWeights(TargetData))`
        // (`train.cpp:456`) — the per-object DOCUMENT weight sum, NOT the pairwise
        // weight total. For a pairwise loss the model.json `leaf_weights` are the
        // document counts (`PairLogit` fixture tree0 `[8,3,0,1]` == 12 docs), so the
        // accumulation uses the per-object `weights` (all 1.0 here), never
        // `eff_weights` (`bt.PairwiseWeights`). For every NON-pairwise loss
        // `eff_weights == weights` so this is byte-identical (D-04). These SAME
        // doc-weight sums feed the `NormalizeLeafValues` weighted-mean centering below.
        let leaf_weights = accumulate_leaf_weights(&leaf_value_leaf_of, &weights, n_leaves);

        // NormalizeLeafValues (`approx_updater_helpers.cpp:8-21`, called from
        // `train.cpp:562`): for a pairwise loss (`UsesPairsForCalculation` ==
        // `uses_pairwise_weights`) subtract the DOCUMENT-WEIGHTED mean leaf value so
        // the tree contributes no constant shift (the pairwise objective is invariant
        // to a global additive constant). Empty leaves (|weight| <= 1e-9) are forced
        // to exactly 0 (NOT shifted), matching upstream. The `learning_rate` scale is
        // applied LAST, inside `NormalizeLeafValues`, for ALL losses. At dim=1 the
        // single dimension's `n_leaves` slice is normalized in place. (06.3-13: this
        // closes the PairLogit/PairLogitPairwise LeafValues parity — the instrumented
        // ground truth `PairLogit/per_leaf_der_log.jsonl` per-leaf raw deltas minus
        // this weighted mean reproduce the frozen model.json leaf values ≤1e-9.)
        normalize_leaf_values(
            uses_pairwise_weights(&params.loss),
            learning_rate,
            &leaf_weights,
            &mut leaf_values,
            n_leaves,
            approx_dimension,
        );

        // 4. Update approx: per dimension, `approx[d*n+i] += leaf_value[d][leaf(i)]`
        //    over the LEAF-VALUE leaf_of (so each iteration's der recompute is
        //    sequential over the same averaging-fold partition — research
        //    "Empirical verification" #2). At dim=1 (`base == 0`,
        //    `leaf_values[0..n_leaves]`) this is exactly the prior scalar update.
        for d in 0..approx_dimension {
            let approx_base = d * n;
            let leaf_base = d * n_leaves;
            for (i, &leaf) in leaf_value_leaf_of.iter().enumerate() {
                if let (Some(a), Some(&lv)) = (
                    approx.get_mut(approx_base + i),
                    leaf_values.get(leaf_base + leaf),
                ) {
                    *a += lv;
                }
            }
        }

        // AVERAGING-FOLD EXP-APPROX UPDATE (`UpdateLearnAvrgApprox`,
        // `approx_updater_helpers.cpp:24-48`): `avrgFoldApprox[i] *=
        // expTreeDelta[leaf]`, where `expTreeDelta = ExpApproxIf(treeDelta)` —
        // the fmath batch exp of the POST-NormalizeLeafValues (learning-rate-
        // scaled) per-leaf tree values. The linear `approx` update above is
        // upstream's `AvrgApprox += treeDelta[leaf]` — both run, feeding
        // different consumers (ders vs metrics/output).
        if exp_ctr {
            let exp_tree_delta: Vec<f64> = leaf_values
                .iter()
                .map(|&v| crate::fast_approx::fmath_expd(v))
                .collect();
            for (i, &leaf) in leaf_value_leaf_of.iter().enumerate() {
                if let (Some(a), Some(&factor)) =
                    (avg_exp_approx.get_mut(i), exp_tree_delta.get(leaf))
                {
                    *a *= factor;
                }
            }
        }

        // FOLD-APPROX UPDATE (`UpdateLearningFold`, `train.cpp:585` — Plain CTR
        // path). EVERY learning fold advances its own approx each iteration (not
        // only the taken one): leaf deltas are re-estimated over the FOLD's own
        // CTR-bin leaf assignment (the taken fold's is exactly `grown.leaf_of`;
        // the others reassign against their fold's columns) and the FOLD's own
        // PRE-update derivatives, then normalized and learning-rate-scaled
        // identically to the model deltas. The monotone post-pass is not
        // replicated: monotone constraints never co-occur with the CTR path in
        // scope. `fold_approxes` is empty on every non-CTR path, so this whole
        // block is a structural no-op there.
        if !fold_approxes.is_empty() && group_spans.is_none() {
            for (j, fold_approx) in fold_approxes.iter_mut().enumerate() {
                let fold_leaf_of: Vec<usize> = if j == taken_fold {
                    grown.leaf_of.clone()
                } else {
                    structure_fold_columns.get(j).map_or_else(
                        || grown.leaf_of.clone(),
                        |cols| assign_leaf_over_ctr_columns(&matrix, cols, &grown, n),
                    )
                };
                // The taken fold's pre-update derivatives were already computed
                // for the search — reuse them; the other folds compute theirs
                // here (still pre-update: this fold's approx is untouched so far
                // this iteration).
                let fold_ders_recomputed: Derivatives;
                let fold_ders: &Derivatives = match (&search_ders_owned, j == taken_fold) {
                    (Some(sd), true) => sd,
                    _ => {
                        fold_ders_recomputed = if exp_ctr {
                            // This fold's approx is EXP-domain under `exp_ctr` —
                            // same der form as the search ders above.
                            let mut der1 = Vec::with_capacity(n);
                            let mut der2 = Vec::with_capacity(n);
                            for (i, &e) in fold_approx.iter().enumerate() {
                                let t = target.get(i).copied().unwrap_or(0.0);
                                let (d1, d2) = crate::fast_approx::logloss_ders_exp(e, t);
                                der1.push(d1);
                                der2.push(d2);
                            }
                            Derivatives { der1, der2 }
                        } else {
                            runtime.compute_gradients(
                                &params.loss,
                                fold_approx,
                                target,
                                approx_dimension,
                            )?
                        };
                        &fold_ders_recomputed
                    }
                };
                let fold_weighted_der1: Vec<f64> = fold_ders
                    .der1
                    .iter()
                    .enumerate()
                    .map(|(idx, &d1)| {
                        let i = idx % n;
                        let w = weights.get(i).copied().unwrap_or(1.0);
                        d1 * w
                    })
                    .collect();
                let fold_leaf_weights = accumulate_leaf_weights(&fold_leaf_of, &weights, n_leaves);
                let mut fold_leaf_values: Vec<f64> = Vec::with_capacity(approx_dimension * n_leaves);
                for d in 0..approx_dimension {
                    let der1_d = fold_weighted_der1.get(d * n..(d + 1) * n).unwrap_or(&[]);
                    let der2_d = fold_ders.der2.get(d * n..(d + 1) * n).unwrap_or(&[]);
                    let approx_d = fold_approx.get(d * n..(d + 1) * n).unwrap_or(&[]);
                    let deltas = compute_leaf_deltas(
                        params.leaf_method,
                        &params.loss,
                        &fold_leaf_of,
                        der1_d,
                        der2_d,
                        // Pointwise path only (the grouped seam is gated out
                        // above): the leaf weight is the per-object weight.
                        &weights,
                        approx_d,
                        target,
                        scaled_l2,
                        n_leaves,
                        d,
                    );
                    fold_leaf_values.extend_from_slice(&deltas);
                }
                if exp_ctr {
                    // `UpdateApproxDeltas` + `UpdateBodyTailApprox` for an
                    // EXP-stored fold (`approx_calcer.cpp:83-117`,
                    // `approx_updater_helpers.h:20-37`): the RAW leaf deltas
                    // (no learning rate, no normalize) are exp-ified per LEAF
                    // via the fmath batch exp, then each document's fold approx
                    // is multiplied by
                    // `fast_exp(FastLogf(expLeafDelta) * learning_rate)` — the
                    // approximate per-doc pipeline whose error is load-bearing
                    // for structure parity (see `fast_approx`). Logloss is
                    // non-pairwise, so upstream's fold update applies no
                    // centering — dropping `normalize_leaf_values` here loses
                    // only its learning-rate scale, which `ApplyLearningRate`
                    // supplies instead.
                    let exp_leaf_deltas: Vec<f64> = fold_leaf_values
                        .iter()
                        .map(|&v| crate::fast_approx::fmath_expd(v))
                        .collect();
                    for (i, &leaf) in fold_leaf_of.iter().enumerate() {
                        if let (Some(a), Some(&ed)) =
                            (fold_approx.get_mut(i), exp_leaf_deltas.get(leaf))
                        {
                            *a *= crate::fast_approx::apply_learning_rate_exp(ed, learning_rate);
                        }
                    }
                } else {
                    normalize_leaf_values(
                        uses_pairwise_weights(&params.loss),
                        learning_rate,
                        &fold_leaf_weights,
                        &mut fold_leaf_values,
                        n_leaves,
                        approx_dimension,
                    );
                    for d in 0..approx_dimension {
                        let approx_base = d * n;
                        let leaf_base = d * n_leaves;
                        for (i, &leaf) in fold_leaf_of.iter().enumerate() {
                            if let (Some(a), Some(&lv)) = (
                                fold_approx.get_mut(approx_base + i),
                                fold_leaf_values.get(leaf_base + leaf),
                            ) {
                                *a += lv;
                            }
                        }
                    }
                }
            }
        }

        // YetiRank LEARNING-fold approx update (D-07, 06.3-14 ext): the learning
        // fold (fold 0, drives the NEXT tree's gradient/structure) carries its OWN
        // approx, updated by a SEPARATE leaf-value recalc over the LEARNING-fold
        // competitors (`UpdateLearningFold -> CalcApproxForLeafStruct`,
        // train.cpp:585). Re-sample the learnfold competitors off `learn_approx`,
        // recompute the grouped der + the Newton leaf deltas over the SAME structure
        // partition (`leaf_value_leaf_of`; Plain/no-CTR shares it), apply the
        // weighted-mean NormalizeLeafValues + learning_rate, and add to
        // `learn_approx`. This is single-dimension (YetiRank approx_dimension == 1)
        // and rides the Newton leaf (unit weights — der2 folds the pair weight).
        // Done AFTER the averaging-fold `approx` update so the staged/model output is
        // unaffected; only the next tree's gradient reads `learn_approx`.
        if is_yetirank {
            if let (Some(spans), Some(seeds)) =
                (group_spans.as_mut(), yetirank_tree_seeds.as_ref())
            {
                for (gi, span) in spans.iter_mut().enumerate() {
                    if let Some((begin, end, weight, relevs)) = yetirank_groups.get(gi) {
                        // WR-04 (06.3-17): typed error on an out-of-range span
                        // instead of a silent empty competitor set.
                        let raw_approx: Vec<f64> = learn_approx
                            .get(*begin..*end)
                            .map(<[f64]>::to_vec)
                            .ok_or_else(|| {
                                CbError::OutOfRange(format!(
                                    "YetiRank learning-fold re-sample: group {gi} span \
                                     [{begin}, {end}) is out of range for learn_approx (len {})",
                                    learn_approx.len()
                                ))
                            })?;
                        let query_seed = seeds.learnfold.get(gi).copied().unwrap_or(0);
                        span.competitors = crate::yetirank_sample_pairs(
                            &raw_approx,
                            relevs,
                            *weight,
                            yetirank_permutations,
                            yetirank_decay,
                            query_seed,
                        );
                    }
                }
            }
            let spans_ref = group_spans.as_deref().unwrap_or(&[]);
            let per_group = runtime.compute_gradients_grouped(
                &params.loss,
                &learn_approx,
                target,
                &weights,
                spans_ref,
                params.random_seed,
            )?;
            let mut lf_der1 = Vec::with_capacity(n);
            let mut lf_der2 = Vec::with_capacity(n);
            for g in &per_group {
                lf_der1.extend_from_slice(&g.der1);
                lf_der2.extend_from_slice(&g.der2);
            }
            // LEARNING-fold leaf delta: the leaf ESTIMATOR must match the
            // STORED averaging-fold leaf path, NOT default to the pointwise
            // Newton/Gradient solver. For an `IsPairwiseScoring` loss
            // (YetiRankPairwise) `CalcApproxForLeafStruct` on the learning fold
            // solves the SAME Cholesky pairwise-leaf system as the averaging
            // fold (`approx_calcer.cpp` routes `*Pairwise` through the pairwise
            // estimator regardless of fold). YetiRank (pointwise, non-pairwise)
            // is NOT `is_pairwise_scoring`, so it keeps the exact prior
            // Newton/Gradient `compute_leaf_deltas` path (byte-identical, D-04).
            let lf_leaf_values: Vec<f64> = if is_pairwise_scoring(&params.loss) {
                crate::pairwise_leaves::compute_pairwise_leaf_deltas(
                    spans_ref,
                    &leaf_value_leaf_of,
                    &lf_der1,
                    n_leaves,
                    params.l2_leaf_reg,
                    PAIRWISE_NON_DIAG_REG_DEFAULT,
                )
            } else {
                // Newton leaf (unit weights — der already folds the pair weight,
                // WR-03).
                let lf_unit = vec![1.0_f64; n];
                compute_leaf_deltas(
                    params.leaf_method,
                    &params.loss,
                    &leaf_value_leaf_of,
                    &lf_der1,
                    &lf_der2,
                    &lf_unit,
                    &learn_approx,
                    target,
                    scaled_l2,
                    n_leaves,
                    0,
                )
            };
            // The LEARNING-fold approx update applies ONLY `learning_rate`
            // (`UpdateBodyTailApprox` -> `ApplyLearningRate`,
            // `approx_updater_helpers.h:26-30`) — NOT `NormalizeLeafValues` (the
            // doc-weighted-mean centering is applied ONLY to the AVERAGING-fold
            // STORED model leaves at `train.cpp:562`). So the learning-fold delta is
            // the RAW Newton leaf delta scaled by `learning_rate`, with no centering.
            for (i, &leaf) in leaf_value_leaf_of.iter().enumerate() {
                if let (Some(a), Some(&lv)) = (learn_approx.get_mut(i), lf_leaf_values.get(leaf)) {
                    *a += lv * learning_rate;
                }
            }
        }

        // Record the staged approximant for this iteration (raw value / logit).
        if let Some(out) = staged_out.as_deref_mut() {
            out.extend_from_slice(&approx);
        }

        // POST per-tree draws: the leaf-estimation seed
        // (`GenRandUI64Vector(foldCount, Rand.GenRand())`-adjacent phase,
        // train.cpp), `POST_TREE_EXTRA_DRAWS` (= 2) `Rand.GenRand()` calls, once
        // per tree. This is the ONLY draw source left to consume out-of-line:
        // the per-level RSM-reselection + `CalcScores` randSeed +
        // `SelectBestCandidate` draws now all happen INLINE during the grow
        // above (`perturb` is `Some` whenever `draws_active`, regardless of
        // `perturb_active` — see its construction above), in exact upstream
        // order/count, VERIFIED against a real instrumented upstream 1.2.10
        // build (2026-07-30, `.planning/plans/bayesian-rng-draw-accounting/
        // instrumented-ground-truth/GROUND_TRUTH.md`). `draws_active == false`
        // (bootstrap_type=No, random_strength=0) stays the byte-identical
        // zero-draw first-slice path.
        if draws_active {
            for _ in 0..POST_TREE_EXTRA_DRAWS {
                rng.gen_rand();
            }
        }

        // MVS lambda for the NEXT tree uses this tree's mean leaf L2 norm
        // (`CalculateLastIterMeanLeafValue`, mvs.cpp:21-35) over the stored
        // (learning_rate-scaled) leaf values.
        prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&leaf_values));

        // Persist the ACTUAL chosen tensor-CTR splits for this tree (ORD-05). On
        // the CTR path `grown.ctr_splits` holds ONLY the WINNING CTR splits
        // (recorded by `greedy_tensor_search_oblivious_with_ctr` with their chosen
        // CTR-value borders + prior PAIR), replacing the prior candidate-only
        // emission. Off the CTR path (numeric `train` driver, empty candidate set)
        // `grown.ctr_splits` is EMPTY, so this is a no-op and the float-only oracles
        // stay byte-identical. `cb_model::Model::from_trained` lifts each chosen
        // split into a `ModelSplit::Ctr` (Plan 05-14 bakes the ctr_data + Scale/
        // Shift). `ctr_splits_for_tree` is retained for the no-CTR candidate path
        // (it returns empty there) so the existing seam keeps compiling.
        let ctr_splits = if has_ctr {
            grown.ctr_splits.clone()
        } else {
            ctr_splits_for_tree(
                &ctr_candidates,
                params.simple_ctr,
                &params.simple_ctr_priors,
                params.combinations_ctr,
                &params.combinations_ctr_priors,
            )
        };

        // FEAT-06 / D-6.6-04: a non-symmetric leaf-wise tree (Lossguide / Depthwise)
        // is persisted into `non_symmetric_trees` carrying its node graph
        // (`step_nodes` / `node_id_to_leaf_id`) + per-node `splits`; the oblivious
        // path is byte-identical (empty `step_nodes` → `ObliviousTree`). A model is
        // EITHER all-oblivious or all-non-symmetric, so only one vec is ever pushed.
        if !grown.region_directions.is_empty() {
            // GPUT-18 / D-03a: a region PATH tree — per-level split + continue
            // direction + one-hot flag, with bin-indexed leaf values (length
            // depth+1). A model is EITHER all-oblivious OR all-non-sym OR all-region,
            // so only `region_trees` is pushed here.
            region_trees.push(RegionTree {
                splits: grown.splits,
                directions: grown.region_directions,
                one_hot: grown.region_one_hot,
                leaf_values,
                leaf_weights,
            });
        } else if grown.step_nodes.is_empty() {
            // SPEC-OH-07: carry the grower's ONE-HOT splits and the true per-level
            // kind order through to the trained tree. A float-only / CTR-only tree
            // leaves `one_hot_splits` empty and (for float-only) `level_kinds`
            // empty too, so the persisted tree is byte-identical (SPEC-OH-31).
            trees.push(oblivious_from_grown(
                grown.splits,
                ctr_splits,
                grown.one_hot_splits,
                grown.level_kinds,
                leaf_values,
                leaf_weights,
            ));
        } else {
            non_symmetric_trees.push(NonSymmetricTree {
                splits: grown.splits,
                step_nodes: grown.step_nodes,
                node_id_to_leaf_id: grown.node_id_to_leaf_id,
                leaf_values,
                leaf_weights,
            });
        }

        // Overfitting detection / use_best_model (TRAIN-06): once the tree is
        // grown, update EACH eval set's raw approximant with this tree's leaf
        // contribution, compute the `eval_metric` over each set (TRAIN-07), log
        // the per-set per-iteration value, and feed the PRIMARY set's metric to
        // the detector + best-model tracker (TRAIN-06), breaking on IsNeedStop().
        if has_test {
            // Dispatches on WHICH ensemble this iteration's tree went into — the
            // oblivious-only `trees.last()` this used to read is empty for every
            // non-symmetric / region grow policy (see `last_tree_eval_contribution`).
            for (set_idx, approx_col) in eval_approx.iter_mut().enumerate() {
                if let Some(em) = eval_matrices.get(set_idx) {
                    for (obj, a) in approx_col.iter_mut().enumerate() {
                        *a += last_tree_eval_contribution(
                            &trees,
                            &non_symmetric_trees,
                            &region_trees,
                            em,
                            obj,
                        );
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

        // ORCH-03-S5: periodic checkpoint. Written at the END of the iteration
        // body, where iteration `iter` is fully complete — the tree is pushed and
        // `approx` carries its contribution — so `completed_iters = iter + 1` is
        // exactly where a resumed run must restart. The device arm `continue`s
        // long before this point and is excluded by the scope guard anyway.
        //
        // The write is atomic (see `snapshot::write_atomic`); an I/O failure
        // PROPAGATES rather than being swallowed, because a checkpoint the caller
        // believes exists but does not is worse than a failed fit.
        if let Some((cfg, fingerprint)) = snapshot_state {
            if last_snapshot_write.elapsed() >= cfg.snapshot_interval {
                let snap = crate::snapshot::capture(
                    iter.saturating_add(1),
                    fingerprint,
                    bias,
                    approx_dimension,
                    &approx,
                    &trees,
                    &rng,
                )?;
                crate::snapshot::write_atomic(&cfg.snapshot_file, &snap)?;
                last_snapshot_write = std::time::Instant::now();
            }
        }
    }

    // use_best_model: truncate the model's trees to best_iteration + 1
    // (upstream `model.tree_count_` for a use_best_model run). Without an eval set
    // there is no best iteration, so the model keeps every grown tree.
    if params.use_best_model {
        if let Some(best) = best_model.best_iteration() {
            trees.truncate(best + 1);
            // Non-symmetric (Lossguide / Depthwise) models push every tree into
            // `non_symmetric_trees` and leave `trees` empty; truncate that vector
            // too so `use_best_model` is not a silent no-op there (WR-01). No-op
            // when `non_symmetric_trees` is empty (oblivious models).
            non_symmetric_trees.truncate(best + 1);
            // GPUT-18: region models push every tree into `region_trees`; truncate
            // it too so use_best_model is not a silent no-op there. No-op when empty.
            region_trees.truncate(best + 1);
        }
    }

    // ---------------------------------------------------------------------------
    // Bake the WHOLE-SET inference ctr_data for each DISTINCT chosen CTR split
    // (ORD-05, Plan 05-14). After the boosting loop, for each distinct
    // (projection, ctr_type, prior_num, prior_denom) the trees chose, accumulate
    // the WHOLE learn set into per-bucket class counts keyed on the COMBINED
    // projection hash (`bake_ctr_table`, via the SHARED accumulate_online +
    // build_final_ctr producer — the inference TOTALS, NOT the prefix), derive the
    // inference (Shift, Scale) from the prior PAIR (calc_normalization(prior_num),
    // Scale = ctr_border_count / norm; Borders:0.5/1 → Shift=0, Scale=15), and copy
    // (Shift, Scale) + the prior PAIR onto EVERY matching chosen CtrSplitSpec so
    // they flow into cb_model::CtrSplit via from_trained.
    //
    // Off the CTR path (numeric train driver, empty cat_columns) no tree carries a
    // CtrSplitSpec, so this loop is a no-op and `baked` is empty — the float-only
    // models keep ctr_data None.
    let mut baked = BakedCtrData::default();
    if !cat_columns.is_empty() {
        // Distinct chosen (projection, ctr_type) pairs across all trees.
        //
        // The de-dup key is `(projection, ctr_type)` and NOTHING ELSE. In
        // particular `target_border_idx` MUST NOT enter it: it is a per-SPLIT
        // selector consumed by `CtrValueTable::numerator_denominator`, so ONE
        // Buckets table serves both b=0 and b=1. Adding it would break the
        // apply-side key reconstruction (`cb_model::apply::ctr_table_key`), which
        // rebuilds `"ctr:type=<i8>:proj=<members>"` from the split and carries no
        // border index, and would invalidate every committed .cbm fixture.
        let mut seen: Vec<(crate::TProjection, i8)> = Vec::new();
        for tree in &trees {
            for spec in &tree.ctr_splits {
                let key = (spec.projection.clone(), spec.ctr_type);
                if !seen.iter().any(|k| k == &key) {
                    seen.push(key);
                    let Some(spec_ctr_type) = crate::ctr::ECtrType::from_i8(spec.ctr_type) else {
                        return Err(CbError::OutOfRange(format!(
                            "chosen CTR split carries an unknown ctr_type discriminant {}",
                            spec.ctr_type
                        )));
                    };
                    // E10/E11: bake with the CHOSEN split's own routed type AND
                    // prior, not a global default. This site is load-bearing
                    // because the (Shift, Scale, prior) pair is COPIED BACK onto
                    // every matching spec below — baking with the wrong prior
                    // would overwrite the correctly-routed value and silently undo
                    // SPEC-CTRT-10. At the default config both prior lists are
                    // [0.5] and both types are Borders, so this is byte-identical
                    // to the previous behavior.
                    let table = bake_ctr_table(
                        cat_columns,
                        &spec.projection,
                        &target_class,
                        2, // binclf target-class count
                        ctr_border_count,
                        spec.prior_num,
                        spec.prior_denom,
                        spec_ctr_type,
                        counter_calc_skip_test,
                        &counter_full_eval_columns,
                    )?;
                    baked.tables.push(table);
                }
            }
        }
        // Derive each chosen split's inference (Shift, Scale) from ITS OWN prior.
        //
        // E15: this loop used to COPY `(shift, scale, prior_num, prior_denom)` off
        // the matching baked table. Both halves of that were wrong once more than
        // one prior is live on a projection:
        //   * the prior copy CLOBBERED the split's own `prior_num`/`prior_denom`,
        //     which already arrive correct from the winning column via
        //     `crate::tree`'s `CtrSplitSpec` construction — so every split on a
        //     projection collapsed onto the first baked table's (head) prior;
        //   * one baked table carries ONE normalization, so copying it is wrong
        //     for every split at a different prior.
        // The table lookup SURVIVES purely as an existence gate: only a split with
        // a baked `(projection, ctr_type)` table gets a derived normalization; a
        // split without one keeps `0.0` / `1.0`. `(projection, ctr_type)` is the
        // same key the bake de-dup uses — `target_border_idx` must not enter it.
        //
        // Consequence: `BakedCtrTable.{shift, scale, prior_num, prior_denom}` are
        // now INFORMATIONAL-ONLY in production (`CtrData::from_baked` already
        // ignores them), but they stay on the struct — `ctr_split_scoring_test`
        // and `ctr::final_ctr_test` read them.
        for tree in &mut trees {
            for spec in &mut tree.ctr_splits {
                if baked
                    .tables
                    .iter()
                    .any(|t| t.projection == spec.projection && t.ctr_type == spec.ctr_type)
                {
                    let (shift, norm) = crate::ctr::calc_normalization(spec.prior_num);
                    spec.shift = shift;
                    spec.scale = if norm == 0.0 {
                        1.0
                    } else {
                        ctr_border_count as f64 / norm
                    };
                }
            }
        }
    }

    Ok((
        Model {
            oblivious_trees: trees,
            non_symmetric_trees,
            region_trees,
            bias,
            approx_dimension,
            class_to_label,
            // SPEC-OH-05: the fit-wide one-hot bin -> raw-hash table and the
            // position -> absolute cat-index map, both EMPTY on the float-only
            // and CTR-only paths.
            one_hot_bin_to_hash,
            one_hot_absolute,
        },
        baked,
    ))
}

#[cfg(test)]
#[path = "boosting_test.rs"]
mod tests;

#[cfg(test)]
#[path = "boosting_device_fold_test.rs"]
mod boosting_device_fold_tests;

#[cfg(test)]
#[path = "device_exact_leaf_config_test.rs"]
mod device_exact_leaf_config_tests;

#[cfg(test)]
#[path = "device_ctr_combo_config_test.rs"]
mod device_ctr_combo_config_tests;
