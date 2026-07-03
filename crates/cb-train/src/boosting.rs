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
    score_st_dev, simple_leaf_delta, solve_symmetric_newton, Derivatives, GroupSpan, LeafMethod,
    Loss, Runtime, RankingCompetitor, QUANTILE_ALPHA, QUANTILE_DELTA,
};
use cb_core::{sum_f64, CbError, CbResult, TFastRng64};
use cb_data::Pair;

use crate::autolr::{self, TargetType};
use crate::query_info::{build_query_info, QueryInfo};
use crate::bootstrap::{bootstrap, last_iter_mean_leaf_value, EBootstrapType};
use crate::ctr::bake::{bake_ctr_table, BakedCtrData};
use crate::ctr::{CounterCalcMethod, ECtrType};
use crate::fold::Fold;
use crate::metrics::{EvalMetric, EvalMetricHistory};
use crate::overfit::{BestModelTracker, EOverfittingDetectorType, OverfittingDetector};
use crate::candidates::tensor_ctr_candidates;
use crate::tree::{
    check_depth, greedy_tensor_search_oblivious_ordered, greedy_tensor_search_oblivious_pairwise,
    greedy_tensor_search_oblivious_perturbed, greedy_tensor_search_oblivious_with_ctr, leaf_index,
    leaf_wise_grower, CtrSplitSpec, FeatureMatrix, GrownTree, LeafWisePolicy, LevelKind,
    Perturbation, Split,
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
/// - [`Self::Region`] — present in the upstream enum for parity but UNIMPLEMENTED
///   on the CPU path (escalated gap, D-6.6-04 "Region OUT"); rejected up front by
///   [`validate_grow_policy`] with a typed [`CbError`] — there is no grower arm and
///   no fabricated structure.
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
    /// Region growth — UNIMPLEMENTED on CPU (escalated gap, D-6.6-04); rejected by
    /// [`validate_grow_policy`].
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
    /// non-symmetric node graph; [`EGrowPolicy::Region`] is rejected up front
    /// ([`validate_grow_policy`]). Pinned EXPLICITLY ([`grow_policy_default`]).
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
/// 1. [`EGrowPolicy::Region`] — UNIMPLEMENTED on the CPU path ("Region OUT",
///    D-6.6-04). There is no Region grower arm; selecting it errors rather than
///    silently falling back to another policy and fabricating a structure.
/// 2. `monotone_constraints` × a NON-SYMMETRIC `grow_policy` ({Lossguide,
///    Depthwise}) — upstream EXPLICITLY rejects monotone constraints under every
///    non-symmetric grow policy (`monotonic_constraint_utils.h:42`,
///    `CB_ENSURE_INTERNAL(monotoneConstraints.empty(), "...unsupported for
///    non-symmetric trees yet")`). The monotone PAVA post-pass is wired ONLY into
///    the oblivious leaf path (D-6.6-06); routing a non-empty `monotone_constraints`
///    through the leaf-wise grower would silently DROP the constraint, so it is
///    rejected with a typed error (D-6.6-07 — no fabricated output).
///
/// Both guards were DEFERRED by Plan 06.6-02 (the `grow_policy` enum did not yet
/// exist); this is the enablement point Plan 06.6-04 owns.
fn validate_grow_policy(grow_policy: EGrowPolicy, monotone_constraints: &[i8]) -> CbResult<()> {
    if grow_policy == EGrowPolicy::Region {
        return Err(CbError::OutOfRange(
            "grow_policy=Region is not supported on the CPU training path \
             (escalated gap, D-6.6-04 \"Region OUT\")"
                .to_owned(),
        ));
    }
    if grow_policy.is_non_symmetric() && !monotone_constraints.is_empty() {
        return Err(CbError::OutOfRange(format!(
            "monotone_constraints are unsupported for non-symmetric trees \
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
/// `averaging_ctr_features` is index-aligned with the structure
/// `materialized_ctr_features` (same projection order), and a `LevelKind::Ctr`'s
/// `ctr_idx` indexes the tree's chosen `ctr_splits`, whose projection identifies
/// which averaging column to read. Out-of-range indices contribute a `false` bit
/// defensively (checked `.get` only — no panic, no raw index).
fn assign_leaf_of_averaging(
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
                    LevelKind::Ctr { ctr_idx, border } => grown
                        .ctr_splits
                        .get(*ctr_idx)
                        // Find the averaging column whose projection matches this
                        // chosen CTR split (index-aligned with the structure
                        // columns; the projection is the stable key).
                        .and_then(|spec| {
                            averaging_ctr_features
                                .iter()
                                .find(|c| c.projection == spec.projection)
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
            shift: 0.0,
            scale: 1.0,
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
    )
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
fn quantize_feature_major(
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    n: usize,
) -> (Vec<u32>, usize) {
    let n_features = feature_values.len();
    let mut bins = vec![0u32; n_features * n];
    let mut n_bins = 0usize;
    for f in 0..n_features {
        let borders = feature_borders.get(f).map_or(&[][..], Vec::as_slice);
        n_bins = n_bins.max(borders.len() + 1);
        let col = feature_values.get(f).map_or(&[][..], Vec::as_slice);
        for i in 0..n {
            let v = col.get(i).copied().map_or(0.0_f64, f64::from);
            let bin = borders.iter().filter(|&&b| v > b).count() as u32;
            if let Some(slot) = bins.get_mut(f * n + i) {
                *slot = bin;
            }
        }
    }
    (bins, n_bins)
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
) -> CbResult<(Model, BakedCtrData)> {
    check_depth(params.depth)?;

    // Validate the loss's hyperparameters before any training work
    // (T-06.1.01-01 / T-06.1.01-02): an out-of-domain q/delta/alpha would yield
    // NaN/Inf derivatives that poison the histogram and leaf reductions, so it is
    // rejected up front with a typed CbError rather than producing a corrupt model.
    params.loss.validate()?;

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

    // Numeric-only training matrix (no categorical features in this path; the
    // one-hot categorical splits are exercised through the categorical-aware
    // tree search directly in the ORD-04 oracle test, D-04).
    let matrix = FeatureMatrix::new(feature_values, feature_borders);

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
    // `params.max_ctr_complexity` gate (:532-533).
    //
    // CAT INGESTION (Plan 05-11): the cat-aware path computes per-cat-feature
    // OnLearnOnly cardinalities (`learn_set_cardinality` = calc_cat_feature_hash +
    // PerfectHash, NEVER a model's CTR hash map) and feeds the REAL cat set to
    // `tensor_ctr_candidates`. The numeric `train` / `train_with_eval_sets` path
    // supplies an EMPTY `cat_columns`, so the cardinalities and candidate set are
    // both empty and the float-only oracles are byte-for-byte unchanged.
    let cat_cardinalities: Vec<u32> = cat_columns
        .iter()
        .map(|col| {
            let as_str: Vec<&str> = col.iter().map(String::as_str).collect();
            crate::candidates::learn_set_cardinality(&as_str)
        })
        .collect::<CbResult<Vec<u32>>>()?;
    let ctr_candidates = tensor_ctr_candidates(
        &cat_cardinalities,
        params.one_hot_max_size,
        params.max_ctr_complexity,
    );

    // Map the CTR-eligible-position projection members emitted by
    // `tensor_ctr_candidates` (dense positions into the CTR-eligible feature list,
    // candidates.rs) back to ABSOLUTE `cat_columns` indices so
    // `materialize_ctr_feature` reads the right columns. The eligible list is the
    // cat features routing to the CTR path (cardinality > one_hot_max_size), in
    // ascending absolute-index order.
    let eligible_absolute: Vec<usize> = cat_cardinalities
        .iter()
        .enumerate()
        .filter(|(_, &card)| {
            crate::candidates::route_categorical(card, params.one_hot_max_size)
                == crate::candidates::EncodingPath::Ctr
        })
        .map(|(abs_idx, _)| abs_idx)
        .collect();

    // The TWO permutations for the cat-CTR two-materialization (research Q1/Q3),
    // now CARRYING the initial learn-set shuffle `S` in the averaging order (ORD-01
    // / bar (c), plan 05-19):
    //   * `cat_learn_permutation` — the STRUCTURE-search fold = the lone learning
    //     `Folds[0]`, the IDENTITY (`shuffle = foldIdx != 0`,
    //     `learn_context.cpp:524`). The structure-search CTR column is materialized
    //     under this permutation. (Per-iteration structure-fold cycling
    //     `[0,2,0,2,2]` is Task 4; T3 keeps the fixed identity Folds[0].)
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
            // STRUCTURE: identity Folds[0] (the structure-search fold).
            let learn: Vec<i32> = (0..n as i32).collect();
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

    // The combination/simple CTR prior PAIR (numerator + unit denominator). The
    // head prior of the explicit `combinations_ctr_priors` (`0.5` for the in-scope
    // `Borders:Prior=0.5` fixture); the denominator is `1` (RESEARCH A6) — both
    // halves are carried so the Plan 05-12 bake receives the denominator for
    // `calc_normalization`, never a pre-divided scalar.
    let ctr_prior_num = params.combinations_ctr_priors.first().copied().unwrap_or(0.5);
    let ctr_prior_denom = 1.0;
    let ctr_border_count = ctr_border_count_default();

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
            // fold 0: identity (unshuffled structure data, the lone Folds[0]).
            // fold j>0: original-object order = [S[p] for p in stream[j]].
            let perm: Vec<i32> = if fold == 0 || !need_shuffle {
                (0..n as i32).collect()
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
            let mut cols = Vec::with_capacity(ctr_candidates.len());
            for proj in &absolute_projections {
                let col = crate::ctr::materialize_ctr_feature(
                    cat_columns,
                    proj,
                    &perm,
                    &target_class,
                    ctr_prior_num,
                    ctr_prior_denom,
                    ctr_border_count,
                )?;
                cols.push(col);
            }
            per_fold.push(cols);
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
            let mut cols = Vec::with_capacity(ctr_candidates.len());
            for proj in &absolute_projections {
                let col = crate::ctr::materialize_ctr_feature(
                    cat_columns,
                    proj,
                    avg_perm,
                    &target_class,
                    ctr_prior_num, ctr_prior_denom,
                    ctr_border_count,
                )?;
                cols.push(col);
            }
            cols
        } else {
            Vec::new()
        };

    let n_leaves = 1usize << params.depth;
    let mut trees: Vec<ObliviousTree> = Vec::with_capacity(params.iterations);
    // FEAT-06 / D-6.6-04: non-symmetric (Lossguide / Depthwise) trees accumulate here
    // when `grow_policy` selects a leaf-wise grower. Empty for every oblivious model.
    let mut non_symmetric_trees: Vec<NonSymmetricTree> = Vec::new();

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
        && ordered_learning_perm.is_none()
        && materialized_ctr_features.is_empty()
        && structure_fold_columns.iter().all(Vec::is_empty)
        && !penalties_active
        && params.monotone_constraints.is_empty()
        && params.grow_policy == EGrowPolicy::SymmetricTree
        && approx_dimension == 1
        && !is_multiclass
        && !is_multilabel
        && matches!(params.bootstrap_type, EBootstrapType::No)
        && params.random_strength == 0.0
        && eval_sets.is_empty()
        && matrix.n_features() > 0;

    // The leaf-regularization is constant across the fit for the device-eligible
    // config (fixed weights / n, no per-tree sampling), so it is computed ONCE and
    // handed to `begin`, matching the CPU per-tree
    // `scale_l2_reg(l2, sumAllWeights, n)`.
    let device_scaled_l2 = scale_l2_reg(params.l2_leaf_reg, sum_all_weights, n);
    let (device_bins, device_n_bins) = if device_host_eligible {
        quantize_feature_major(feature_values, feature_borders, n)
    } else {
        (Vec::new(), 0)
    };
    let device_active = if device_host_eligible && device_n_bins > 0 {
        runtime.begin_device_training(
            &params.loss,
            params.depth,
            matches!(params.boosting_type, EBoostingType::Plain),
            /* fold_count = */ 1,
            params.score_function,
            &device_bins,
            &weights,
            n,
            matrix.n_features(),
            device_n_bins,
            learning_rate,
            device_scaled_l2,
        )?
    } else {
        false
    };
    // Teardown on EVERY exit path (incl. the `?` error path), T-10-24. Inert when
    // no session was opened (`device_active == false`).
    let _device_guard = DeviceSessionGuard {
        runtime,
        active: device_active,
    };

    for iter in 0..params.iterations {
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
            let dev_tree = runtime
                .grow_tree_on_device(&approx, target)?
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

            // Resolve each device split `(feature, bin_id)` to a Model `Split` via
            // `border = feature_borders[feature][bin_id]` (Pattern 4 — the one
            // non-obvious correctness join). Range-check `bin_id` against the
            // feature's border count (T-10-22): an out-of-range index is a typed
            // error, never a panic / raw index. `DeviceGrownTree.leaf_of` is NOT
            // consumed (D-05 — empty in the production hot path).
            let mut device_splits: Vec<Split> = Vec::with_capacity(dev_tree.splits.len());
            for &(feature, bin_id) in &dev_tree.splits {
                let f = feature as usize;
                let b = bin_id as usize;
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
                device_splits.push(Split { feature: f, border });
            }

            // Per-object leaf assignment on the HOST from the resolved splits (D-05:
            // the device does NOT cross the seam with an `n`-length `leaf_of`).
            // This is the SAME `value > border` + forward-bit `leaf_index` the CPU
            // oblivious path uses, so the folded partition is CPU-identical.
            let device_leaf_of: Vec<usize> = (0..n)
                .map(|obj| {
                    let passes: Vec<bool> = device_splits
                        .iter()
                        .map(|s| {
                            feature_values
                                .get(s.feature)
                                .and_then(|col| col.get(obj))
                                .is_some_and(|&v| f64::from(v) > s.border)
                        })
                        .collect();
                    leaf_index(&passes)
                })
                .collect();

            // Leaf values: the device returns UN-scaled leaves (the 10-02 contract);
            // cb-train applies the `learning_rate` shrinkage. RMSE / Logloss /
            // CrossEntropy are non-pairwise, so `normalize_leaf_values` applies ONLY
            // the lr scale (no weighted-mean centering) — byte-identical to the CPU
            // `learning_rate * delta` store (D-04).
            let mut device_leaf_values = dev_tree.leaf_values.clone();
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

            // Update approx: `approx[i] += leaf_values[leaf(i)]` (single dimension) —
            // the SAME per-object accumulation the CPU path applies, so the staged
            // approximant and the next tree's device derivatives stay CPU-consistent.
            for (i, &leaf) in device_leaf_of.iter().enumerate() {
                if let (Some(a), Some(&lv)) =
                    (approx.get_mut(i), device_leaf_values.get(leaf))
                {
                    *a += lv;
                }
            }

            // Record the staged approximant (raw value / logit) for this iteration.
            if let Some(out) = staged_out.as_deref_mut() {
                out.extend_from_slice(&approx);
            }

            trees.push(ObliviousTree {
                splits: device_splits,
                ctr_splits: Vec::new(),
                leaf_values: device_leaf_values,
                leaf_weights: device_leaf_weights,
            });
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
        let der_obj: Vec<f64> = (0..n)
            .map(|i| {
                let squares: Vec<f64> = (0..approx_dimension)
                    .map(|d| {
                        let v = weighted_der1.get(d * n + i).copied().unwrap_or(0.0);
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
        let score_weighted_der1: Vec<f64> = weighted_der1
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
        let perturb = if perturb_active {
            let model_length = iter as f64 * learning_rate;
            // CR-02: std-dev sums `wd²` over the FULL dim-major buffer but
            // divides by the per-OBJECT count `n` (NOT `dim*n`); the `ln(n)`
            // model-size multiplier likewise uses `n` (greedy_tensor_search.cpp:
            // 106, 125). At dim=1, `weighted_der1.len() == n` so this is
            // byte-identical to the prior call (D-04).
            let std_dev = score_st_dev(params.random_strength, &weighted_der1, n, model_length);
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
        // structure CTR columns. `taken_fold = struct_fold_cycle[iter]` (defaulting
        // to 0). For learning_folds==1 the cycle is all-zeros, so this is always
        // `structure_fold_columns[0]` == the prior fixed `materialized_ctr_features`
        // (byte-identical). For pc=4 it cycles `[0,2,0,2,2]`, materializing the tree
        // STRUCTURE under fold 0 (borders [7,2]) or fold 2 (borders [3,7]) per iter.
        let taken_fold = struct_fold_cycle.get(iter).copied().unwrap_or(0);
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
            // SymmetricTree (and the never-reached Region): the literal existing
            // oblivious grower chain, UNCHANGED.
            EGrowPolicy::SymmetricTree | EGrowPolicy::Region => {
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
                &weighted_der1,
                &weights,
                scaled_l2,
                params.depth,
                n,
                0,
                // model_size_reg cat-feature weight (GetCatFeatureWeight): the
                // default 0.5 down-weights high-cardinality (combination) CTR
                // candidates so a new {0,1} combination does not out-score a second
                // border on an already-used {0} simple CTR on a thin margin.
                model_size_reg_default(),
                params.score_function,
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

        // Per-tree leaf count: a non-symmetric leaf-wise tree has a DISTINCT leaf
        // count (number of terminal nodes), NOT `2^depth`. Shadow `n_leaves` for the
        // leaf-value estimation below so the reductions cover exactly this tree's
        // leaves. For the oblivious path `grown.step_nodes` is empty and this is the
        // unchanged `2^depth` (byte-identical, D-6.6-05).
        let n_leaves: usize = if grown.step_nodes.is_empty() {
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
            assign_leaf_of_averaging(&matrix, &averaging_ctr_features, &grown, n)
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
            ctr_splits_for_tree(&ctr_candidates, &params.combinations_ctr_priors)
        };

        // FEAT-06 / D-6.6-04: a non-symmetric leaf-wise tree (Lossguide / Depthwise)
        // is persisted into `non_symmetric_trees` carrying its node graph
        // (`step_nodes` / `node_id_to_leaf_id`) + per-node `splits`; the oblivious
        // path is byte-identical (empty `step_nodes` → `ObliviousTree`). A model is
        // EITHER all-oblivious or all-non-symmetric, so only one vec is ever pushed.
        if grown.step_nodes.is_empty() {
            trees.push(ObliviousTree {
                splits: grown.splits,
                ctr_splits,
                leaf_values,
                leaf_weights,
            });
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
            // Non-symmetric (Lossguide / Depthwise) models push every tree into
            // `non_symmetric_trees` and leave `trees` empty; truncate that vector
            // too so `use_best_model` is not a silent no-op there (WR-01). No-op
            // when `non_symmetric_trees` is empty (oblivious models).
            non_symmetric_trees.truncate(best + 1);
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
        // Distinct chosen projections (by the sorted member set) across all trees.
        let mut seen: Vec<crate::TProjection> = Vec::new();
        for tree in &trees {
            for spec in &tree.ctr_splits {
                if !seen.iter().any(|p| p == &spec.projection) {
                    seen.push(spec.projection.clone());
                    let table = bake_ctr_table(
                        cat_columns,
                        &spec.projection,
                        &target_class,
                        2, // binclf target-class count
                        ctr_border_count,
                        ctr_prior_num,
                        ctr_prior_denom,
                    )?;
                    baked.tables.push(table);
                }
            }
        }
        // Copy the bake-derived (Shift, Scale) + prior PAIR onto each chosen split.
        for tree in &mut trees {
            for spec in &mut tree.ctr_splits {
                if let Some(table) = baked
                    .tables
                    .iter()
                    .find(|t| t.projection == spec.projection)
                {
                    spec.shift = table.shift;
                    spec.scale = table.scale;
                    spec.prior_num = table.prior_num;
                    spec.prior_denom = table.prior_denom;
                }
            }
        }
    }

    Ok((
        Model {
            oblivious_trees: trees,
            non_symmetric_trees,
            bias,
            approx_dimension,
            class_to_label,
        },
        baked,
    ))
}

#[cfg(test)]
#[path = "boosting_test.rs"]
mod tests;
