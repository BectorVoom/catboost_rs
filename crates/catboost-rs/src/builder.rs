//! [`CatBoostBuilder`] — the single unified Builder facade (D-05).
//!
//! The Rust-native Builder pattern (CLAUDE.md) over the internal
//! `cb_train::BoostParams` surface: `new()` + chained `#[must_use]` setters +
//! `fit(&pool) -> Result<Model, CatBoostError>`. The `loss` field SELECTS the
//! task — classification vs regression — with NO typed `Classifier`/`Regressor`
//! split (D-05). Regression losses (RMSE/MAE) train on the raw label;
//! classification losses (Logloss/CrossEntropy/Focal) train on the `{0,1}` /
//! `[0,1]` label.
//!
//! `fit` computes the model's per-float-feature quantization borders from the
//! pool (`cb_data::select_borders_greedy_logsum`, the Phase-2 greedy-logsum
//! binarizer), runs the plain boosting loop over the Phase-3 `cb_backend::CpuBackend`
//! runtime, and lifts the trained model into the canonical `cb_model::Model`
//! (carrying `leaf_weights` + `float_feature_borders`) wrapped in the facade
//! [`crate::Model`].

use std::sync::Arc;

// Compile-time backend selection (08-08): the facade picks `CpuBackend` under
// `cpu` and the generic `GpuBackend` under any of wgpu/cuda/rocm, so no cpu-only
// symbol is named under a non-cpu build. Both implement `cb_compute::Runtime`, the
// only bound `cb_train::train<R: Runtime>` requires.
#[cfg(feature = "cpu")]
use cb_backend::CpuBackend;
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use cb_backend::GpuBackend;
use cb_compute::{
    CustomMetric, CustomMetricHandle, CustomObjective, CustomObjectiveHandle, EScoreFunction,
    LeafEstimationBacktracking, LeafMethod, Loss,
};
use cb_data::{
    select_borders_greedy_logsum_f32, AutoClassWeights, EBorderSelectionType, NanMode, Pool,
    QuantizeParams,
};
use rayon::prelude::*;
use cb_train::{
    boosting_type_default, combinations_ctr_default, combinations_ctr_priors_default,
    counter_calc_method_default, feature_weights_default, first_feature_use_penalties_default,
    fold_len_multiplier_default, grow_policy_default, has_time_default, max_ctr_complexity_default,
    max_leaves_default, min_data_in_leaf_default, monotone_constraints_default,
    one_hot_max_size_default, penalties_coefficient_default, per_object_feature_penalties_default,
    permutation_count_default, score_function_default, simple_ctr_default,
    simple_ctr_priors_default, train_cat_with_eval_sets, train_with_eval_sets, BoostParams,
    CounterCalcMethod, EBoostingType, EBootstrapType, ECtrType, EGrowPolicy,
    EOverfittingDetectorType, EvalMetric, EvalMetricHistory, EvalSet,
};

use crate::error::CatBoostError;
use crate::model::Model;

/// The published Builder for training a CatBoost model (D-05, RAPI-01).
///
/// Start with [`CatBoostBuilder::new`], chain the `#[must_use]` setters to
/// override defaults, then call [`CatBoostBuilder::fit`] with a
/// [`cb_data::Pool`]. The `loss` selects the task: a regression loss
/// ([`Loss::Rmse`] / [`Loss::Mae`]) trains on the raw label; a classification
/// loss ([`Loss::Logloss`] / [`Loss::CrossEntropy`] / [`Loss::Focal`]) trains on
/// the binary label. There is intentionally no separate `Classifier`/`Regressor`
/// type (D-05).
///
/// Defaults mirror catboost 1.2.10 for the in-scope plain-boosting surface
/// (`depth = 6`, `learning_rate = 0.03`, `l2_leaf_reg = 3.0`,
/// `iterations = 1000`, no sampling, no early stopping) so a bare
/// `CatBoostBuilder::new().fit(&pool)` is a sensible default run.
// `Copy` is NOT derived: the `loss: Loss` field is non-Copy (Phase 6.2,
// D-6.2-05 — the Wave-3 MultiQuantile variant carries an owned Vec<f64>). The
// builder remains `Clone`; the consuming-`self` builder methods move rather than
// copy, so dropping `Copy` is source-compatible here.
#[derive(Debug, Clone, PartialEq)]
pub struct CatBoostBuilder {
    loss: Loss,
    /// Optional explicit eval metric (LOSS-07). `None` derives it from `loss`
    /// (`EvalMetric::for_loss`); a `Some(EvalMetric::Custom(..))` is set via
    /// [`CatBoostBuilder::custom_metric`].
    eval_metric: Option<EvalMetric>,
    iterations: usize,
    depth: usize,
    learning_rate: f64,
    auto_learning_rate: bool,
    l2_leaf_reg: f64,
    random_strength: f64,
    boost_from_average: bool,
    leaf_method: LeafMethod,
    bootstrap_type: EBootstrapType,
    subsample: f64,
    bagging_temperature: f32,
    random_seed: u64,
    border_count: usize,
    /// The `feature_border_type` binarizer. Only observable when `border_count`
    /// binds — see [`CatBoostBuilder::feature_border_type`].
    feature_border_type: EBorderSelectionType,
    /// How `NaN` float values are quantized (`nan_mode`) — see
    /// [`CatBoostBuilder::nan_mode`].
    nan_mode: NanMode,
    /// The leaf-step backtracking policy (`leaf_estimation_backtracking`) — see
    /// [`CatBoostBuilder::leaf_estimation_backtracking`].
    leaf_estimation_backtracking: LeafEstimationBacktracking,
    score_function: EScoreFunction,
    /// Cardinality ceiling for the one-hot categorical encoding path
    /// (`one_hot_max_size`, upstream default 2). A categorical column with
    /// `1 < learn-cardinality <= one_hot_max_size` routes to one-hot; above it,
    /// to the CTR path.
    one_hot_max_size: u32,
    /// Maximum feature-combination (tensor-CTR) projection length
    /// (`max_ctr_complexity`, upstream default 4). `1` emits SimpleCtrs only and
    /// is the ONLY in-engine way to suppress combination CTRs entirely.
    max_ctr_complexity: usize,
    /// The SINGLE simple-CTR type (`simple_ctr`).
    ///
    /// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR
    /// descriptions (`catboost_options.cpp:439-453`); this crate models ONE
    /// description with a prior LIST. The type and the full prior list ARE
    /// honored (SPEC-CTRT-09/10/11); a simultaneous `[Borders, Counter]`
    /// configuration is NOT representable.
    simple_ctr: ECtrType,
    /// Per-prior NUMERATORS for [`Self::simple_ctr`]. Each denominator is pinned
    /// to `1`: `Prior=<n>/<d>` with `d != 1` is illegal on CPU upstream
    /// (`ctr_helper.cpp:50`), which is what vindicates the engine's
    /// `prior_denom: 1.0` pin.
    simple_ctr_priors: Vec<f64>,
    /// The SINGLE combination-CTR type (`combinations_ctr`). Same
    /// single-description parity gap as [`Self::simple_ctr`].
    combinations_ctr: ECtrType,
    /// Per-prior NUMERATORS for [`Self::combinations_ctr`]; unit denominators,
    /// as for [`Self::simple_ctr_priors`].
    combinations_ctr_priors: Vec<f64>,
    /// Whether the Counter CTR tally folds the eval sets in (`Full`) or skips
    /// them (`SkipTest`, the upstream default).
    ///
    /// **Observable only when an eval set is present**: with a learn set alone
    /// the two settings produce bit-identical models (measured `0.000e+00`
    /// learn-only vs `4.010e-01` with an eval set — the E23 gate).
    counter_calc_method: CounterCalcMethod,

    // ---- PARAM-01: the overfitting-detector / best-model controls -----------
    // Each of these was previously PINNED inside `boost_params()` to the
    // engine-inert value, so `cb_train` implemented the behaviour but no caller
    // — Rust or Python — could reach it. The defaults below reproduce those
    // pins exactly, so an untouched builder is byte-identical to the pre-PARAM-01
    // form (asserted by `untouched_builder_emits_the_pinned_engine_defaults`).
    /// Overfitting-detector type (`od_type`). [`EOverfittingDetectorType::None`]
    /// (the default) never stops. Consumed ONLY through an eval-set fit
    /// ([`CatBoostBuilder::fit_with_eval`]): with no eval set there is no metric
    /// curve to detect on, so the detector never fires.
    od_type: EOverfittingDetectorType,
    /// Overfitting-detector stop threshold (`od_pval` / `AutoStopPValue`). `0.0`
    /// makes IncToDec / Wilcoxon inactive (the upstream default); `Iter` ignores
    /// it (its threshold is forced to `1.0`).
    od_pval: f64,
    /// Overfitting-detector wait iterations (`od_wait` / `IterationsWait`) — the
    /// number of non-improving iterations tolerated before stopping. This is what
    /// upstream's `early_stopping_rounds` sets (together with `od_type = Iter`).
    od_wait: usize,
    /// `use_best_model`: track the best eval-metric iteration and truncate the
    /// model to `best_iteration + 1` trees. Eval-set-only, like the detector.
    use_best_model: bool,

    // ---- PARAM-01: the boosting-scheme controls ----------------------------
    /// The boosting type (`boosting_type`). [`EBoostingType::Plain`] is the CPU
    /// default; [`EBoostingType::Ordered`] drives the anti-leakage ordered
    /// approximant path (ORD-02).
    boosting_type: EBoostingType,
    /// Whether the learn dataset is TIME-ORDERED (`has_time`). `true` SKIPS the
    /// initial learn-set Fisher-Yates shuffle (`preprocess.cpp:161`), preserving
    /// the natural object order.
    has_time: bool,
    /// Number of random permutations for the multi-permutation fold machinery
    /// (`permutation_count`, upstream default 4).
    permutation_count: usize,
    /// Tail-growth multiplier for the dynamic (ordered) fold body/tail
    /// (`fold_len_multiplier`, upstream default 2.0).
    fold_len_multiplier: f64,

    // ---- PARAM-01: the per-feature weighting / penalty surface -------------
    /// Per-float-feature MULTIPLICATIVE gain weights (`feature_weights`). EMPTY
    /// (the upstream default) means every weight is `1.0`.
    feature_weights: Vec<f64>,
    /// Per-float-feature SUBTRACTIVE first-use penalties
    /// (`first_feature_use_penalties`). EMPTY ⇒ `0.0` for every feature.
    first_feature_use_penalties: Vec<f64>,
    /// Per-float-feature SUBTRACTIVE per-object penalties
    /// (`per_object_feature_penalties`). EMPTY ⇒ `0.0` for every feature.
    per_object_feature_penalties: Vec<f64>,
    /// The scaling coefficient multiplying BOTH penalty terms
    /// (`penalties_coefficient`, upstream default `1.0`). Never consumed while
    /// both penalty vectors are empty.
    penalties_coefficient: f64,
    /// Per-float-feature monotone constraints (`monotone_constraints`): `+1`
    /// non-decreasing, `-1` non-increasing, `0` free. Enforced as an isotonic
    /// (PAVA) projection over the per-leaf deltas. OBLIVIOUS-ONLY — upstream
    /// rejects them under every non-symmetric grow policy, and so does the
    /// engine's typed guard.
    monotone_constraints: Vec<i8>,

    // ---- PARAM-01: the tree grow policy ------------------------------------
    /// The tree grow policy (`grow_policy`). [`EGrowPolicy::SymmetricTree`] is
    /// the oblivious default; `Lossguide` / `Depthwise` grow a true non-symmetric
    /// node graph; `Region` is rejected by the engine's validator.
    grow_policy: EGrowPolicy,
    /// Maximum leaf count for the Lossguide grower (`max_leaves`, default 31).
    /// Ignored by SymmetricTree / Depthwise (both bounded by `depth`).
    max_leaves: usize,
    /// Minimum document count required to split a leaf (`min_data_in_leaf`,
    /// default 1). Read by the leaf-wise growers.
    min_data_in_leaf: usize,

    // ---- PARAM-03: the class-weighting surface (classification only) -------
    //
    // `cb_data::weights` already implemented the upstream-faithful computation
    // (and `weights_oracle_test` gates it against the frozen
    // `class_weights/` fixture), but NOTHING applied it during a fit — the
    // resolved per-object weights never reached `train`. These three fields are
    // the missing application path.
    /// Explicit per-class weights (`class_weights`). EMPTY (the default) means no
    /// class reweighting. Multiplies each object's weight by its class's entry.
    class_weights: Vec<f64>,
    /// Automatic class weights derived from the class distribution
    /// (`auto_class_weights`). Mutually exclusive with
    /// [`Self::class_weights`] / [`Self::scale_pos_weight`].
    auto_class_weights: AutoClassWeights,
    /// Binary-classification positive-class multiplier (`scale_pos_weight`),
    /// upstream default `1.0` — exactly `class_weights = [1.0, w]`. Mutually
    /// exclusive with the other two.
    scale_pos_weight: f64,

    // ---- PARAM-03: ignored_features ----------------------------------------
    /// Float-feature indices the tree search must never split on
    /// (`ignored_features`). EMPTY (the default) ignores nothing.
    ignored_features: Vec<usize>,
}

/// The result of an eval-set fit ([`CatBoostBuilder::fit_with_eval_sets`]): the
/// trained model plus the per-eval-set per-iteration metric curves the run
/// produced.
///
/// `eval_history[k]` is eval set `k`'s ordered `eval_metric` values, one per
/// COMPLETED boosting iteration. When the overfitting detector stops the run
/// early the curves are shorter than `iterations` — their length is therefore the
/// observable "how many iterations actually ran", which a bare [`Model`] cannot
/// report once `use_best_model` has truncated its tree list.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// The trained model (already truncated to `best_iteration + 1` trees when
    /// [`CatBoostBuilder::use_best_model`] was set).
    pub model: Model,
    /// Per-eval-set per-iteration metric curves; `eval_history[0]` is the PRIMARY
    /// set, the one the detector and the best-model tracker consume.
    pub eval_history: Vec<Vec<f64>>,
}

impl Default for CatBoostBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// PARAM-03: blank every categorical column named in `ignored` (LOCAL categorical
/// indices), leaving every other column — and every column's INDEX — untouched.
///
/// A blanked column holds one repeated value, so its learn-set cardinality is `1`
/// and `route_categorical` classifies it `Skip`: it enters neither the one-hot nor
/// the CTR candidate list, while the columns around it keep their absolute
/// indices. That is upstream's own `AddOneHotFeatures` skip-if-`<=1` rule, and it
/// is the exact categorical analogue of clearing a float feature's border set —
/// no candidate split at any level, no renumbering of anything else.
///
/// Returns [`std::borrow::Cow::Borrowed`] — the zero-copy identity — when
/// `ignored` is empty, so a fit that ignores no categorical feature is
/// byte-identical to one built without the parameter and copies nothing. The
/// owned arm does clone the pool's categorical columns; that cost is paid only by
/// a fit that actually ignores a categorical column, and the alternative (a
/// masking seam threaded through the trainer's public signatures) would push the
/// same concern into every caller that has no categorical features at all.
fn mask_ignored_cat_columns<'a>(
    cat_columns: &'a [Vec<String>],
    ignored: &[usize],
) -> std::borrow::Cow<'a, [Vec<String>]> {
    if ignored.is_empty() {
        return std::borrow::Cow::Borrowed(cat_columns);
    }
    let mut cols = cat_columns.to_vec();
    for &c in ignored {
        if let Some(col) = cols.get_mut(c) {
            // Clear IN PLACE (the clone above already allocated the column) —
            // every value becomes the same empty string, so the column is
            // constant and its cardinality is 1.
            for v in col.iter_mut() {
                v.clear();
            }
        }
    }
    std::borrow::Cow::Owned(cols)
}

impl CatBoostBuilder {
    /// Create a builder with catboost 1.2.10 defaults for the in-scope
    /// plain-boosting surface. The default loss is [`Loss::Rmse`] (regression);
    /// call [`CatBoostBuilder::loss`] to select classification.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loss: Loss::Rmse,
            eval_metric: None,
            iterations: 1000,
            depth: 6,
            learning_rate: 0.03,
            auto_learning_rate: false,
            l2_leaf_reg: 3.0,
            random_strength: 0.0,
            boost_from_average: true,
            leaf_method: LeafMethod::Gradient,
            bootstrap_type: EBootstrapType::No,
            subsample: 1.0,
            bagging_temperature: 0.0,
            random_seed: 0,
            border_count: QuantizeParams::default().border_count,
            feature_border_type: EBorderSelectionType::default(),
            // catboost's default nan_mode is Min.
            nan_mode: NanMode::Min,
            leaf_estimation_backtracking: LeafEstimationBacktracking::default(),
            score_function: score_function_default(),
            one_hot_max_size: one_hot_max_size_default(),
            max_ctr_complexity: max_ctr_complexity_default(),
            simple_ctr: simple_ctr_default(),
            simple_ctr_priors: simple_ctr_priors_default(),
            combinations_ctr: combinations_ctr_default(),
            combinations_ctr_priors: combinations_ctr_priors_default(),
            counter_calc_method: counter_calc_method_default(),
            // PARAM-01: every value below reproduces the literal constant
            // `boost_params()` used to pin, so `CatBoostBuilder::new()` emits the
            // SAME `BoostParams` as before the setters existed (D-04).
            od_type: EOverfittingDetectorType::None,
            od_pval: 0.0,
            od_wait: 0,
            use_best_model: false,
            boosting_type: boosting_type_default(),
            has_time: has_time_default(),
            permutation_count: permutation_count_default(),
            fold_len_multiplier: fold_len_multiplier_default(),
            feature_weights: feature_weights_default(),
            first_feature_use_penalties: first_feature_use_penalties_default(),
            per_object_feature_penalties: per_object_feature_penalties_default(),
            penalties_coefficient: penalties_coefficient_default(),
            monotone_constraints: monotone_constraints_default(),
            grow_policy: grow_policy_default(),
            max_leaves: max_leaves_default(),
            min_data_in_leaf: min_data_in_leaf_default(),
            // PARAM-03: all four are the upstream "off" values, so an untouched
            // builder resolves weights to the pool's own and ignores nothing.
            class_weights: Vec::new(),
            auto_class_weights: AutoClassWeights::None,
            scale_pos_weight: 1.0,
            ignored_features: Vec::new(),
        }
    }

    /// Select the loss / objective. The loss SELECTS the task (regression vs
    /// classification) — D-05.
    #[must_use]
    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    /// Select a user-supplied custom training objective (LOSS-07, D-6.4-05). The
    /// `Arc<dyn CustomObjective>` is plugged into the SAME loss dispatch the
    /// built-ins ride via [`Loss::Custom`]; its per-object `(der1, der2)` from
    /// `calc_ders_range` drive leaf estimation. The Phase-8 PyO3 callback bridge
    /// (D-09) wraps the SAME trait through this surface — no `pyo3` here.
    #[must_use]
    pub fn custom_objective(mut self, objective: Arc<dyn CustomObjective>) -> Self {
        self.loss = Loss::Custom(CustomObjectiveHandle::new(objective));
        self
    }

    /// Select a user-supplied custom evaluation metric (LOSS-07, D-6.4-05),
    /// plugged into the SAME [`cb_train::EvalMetric`] dispatch via
    /// [`EvalMetric::Custom`]. The Phase-8 PyO3 callback (D-09) wraps the SAME
    /// [`cb_compute::CustomMetric`] trait through this setter.
    #[must_use]
    pub fn custom_metric(mut self, metric: Arc<dyn CustomMetric>) -> Self {
        self.eval_metric = Some(EvalMetric::Custom(CustomMetricHandle::new(metric)));
        self
    }

    /// Number of boosting iterations (trees).
    #[must_use]
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Tree depth (`2^depth` leaves per oblivious tree).
    #[must_use]
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Learning rate scaling every leaf delta. Ignored when
    /// [`CatBoostBuilder::auto_learning_rate`] is set and the loss is auto-LR
    /// eligible.
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Enable automatic learning-rate selection pre-train (TRAIN-08). When the
    /// loss is not in the upstream auto-LR table the explicit
    /// [`CatBoostBuilder::learning_rate`] is used unchanged.
    #[must_use]
    pub fn auto_learning_rate(mut self, auto_learning_rate: bool) -> Self {
        self.auto_learning_rate = auto_learning_rate;
        self
    }

    /// L2 leaf regularization (`l2_leaf_reg`).
    #[must_use]
    pub fn l2_leaf_reg(mut self, l2_leaf_reg: f64) -> Self {
        self.l2_leaf_reg = l2_leaf_reg;
        self
    }

    /// Split-score perturbation strength (`random_strength`). `0.0` disables it.
    #[must_use]
    pub fn random_strength(mut self, random_strength: f64) -> Self {
        self.random_strength = random_strength;
        self
    }

    /// Whether to start from the per-loss optimum constant approx (the target
    /// mean for RMSE), stored as the model bias. `false` starts from `0`.
    #[must_use]
    pub fn boost_from_average(mut self, boost_from_average: bool) -> Self {
        self.boost_from_average = boost_from_average;
        self
    }

    /// Leaf-estimation method (`leaf_estimation_method`, TRAIN-03 / D-09).
    #[must_use]
    pub fn leaf_method(mut self, leaf_method: LeafMethod) -> Self {
        self.leaf_method = leaf_method;
        self
    }

    /// Bootstrap / sampling type (`bootstrap_type`, TRAIN-04).
    #[must_use]
    pub fn bootstrap_type(mut self, bootstrap_type: EBootstrapType) -> Self {
        self.bootstrap_type = bootstrap_type;
        self
    }

    /// Object subsample fraction (`subsample`); `1.0` disables subsampling.
    #[must_use]
    pub fn subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }

    /// Bayesian bagging temperature (`bagging_temperature`).
    #[must_use]
    pub fn bagging_temperature(mut self, bagging_temperature: f32) -> Self {
        self.bagging_temperature = bagging_temperature;
        self
    }

    /// Training random seed (`random_seed`); consumed only when sampling /
    /// perturbation is active.
    #[must_use]
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = random_seed;
        self
    }

    /// Per-feature border budget for quantization (`border_count`, catboost
    /// default 254).
    #[must_use]
    pub fn border_count(mut self, border_count: usize) -> Self {
        self.border_count = border_count;
        self
    }

    /// Which border-selection algorithm quantization uses
    /// (`feature_border_type`, catboost default
    /// [`EBorderSelectionType::GreedyLogSum`]).
    ///
    /// All seven upstream binarizers are supported; see
    /// [`cb_data::EBorderSelectionType`]. The choice is only observable when the
    /// `border_count` budget BINDS — below the number of representable splits
    /// every algorithm returns the same border set.
    #[must_use]
    pub fn feature_border_type(mut self, feature_border_type: EBorderSelectionType) -> Self {
        self.feature_border_type = feature_border_type;
        self
    }

    /// How `NaN` float values are quantized (`nan_mode`, catboost default
    /// [`NanMode::Min`]).
    ///
    /// A float column containing `NaN` gets a SENTINEL border that isolates the
    /// missing values into their own bin: [`NanMode::Min`] prepends `f32::MIN`
    /// (NaN sorts below every real value), [`NanMode::Max`] appends `f32::MAX`
    /// (above every real value). [`NanMode::Forbidden`] rejects a `NaN`-bearing
    /// column at fit time with a [`CatBoostError`], mirroring upstream's
    /// `quantization.cpp:320` refusal.
    ///
    /// The sentinel costs one slot of the feature's `border_count` budget, and a
    /// `NaN`-free column is unaffected by this setting.
    #[must_use]
    pub fn nan_mode(mut self, nan_mode: NanMode) -> Self {
        self.nan_mode = nan_mode;
        self
    }

    /// The leaf-step backtracking policy (`leaf_estimation_backtracking`,
    /// catboost default [`LeafEstimationBacktracking::AnyImprovement`]).
    ///
    /// [`LeafEstimationBacktracking::Armijo`] is GPU-ONLY upstream and is
    /// rejected at [`Self::fit`] with a [`CatBoostError::InvalidConfig`],
    /// mirroring `catboost_options.cpp:664`.
    ///
    /// `No` and `AnyImprovement` are provably EQUIVALENT in this engine's
    /// supported regime: backtracking shrinks a leaf step that would worsen the
    /// loss, so it needs more than one leaf-estimation step to have anything to
    /// fall back to, and `leaf_estimation_iterations` is not implemented here
    /// (the estimator takes exactly one step). Measured against catboost 1.2.10
    /// over 64 loss x leaf-method x learning-rate configurations at one
    /// iteration, the two policies never differ; at more than one iteration they
    /// differ in 53. See [`LeafEstimationBacktracking`] and the
    /// `leaf_estimation_backtracking/` oracle.
    #[must_use]
    pub fn leaf_estimation_backtracking(
        mut self,
        leaf_estimation_backtracking: LeafEstimationBacktracking,
    ) -> Self {
        self.leaf_estimation_backtracking = leaf_estimation_backtracking;
        self
    }

    /// Split-score function (`score_function`). [`EScoreFunction::Cosine`] is the
    /// catboost CPU default; [`EScoreFunction::L2`] is the variance-reduction
    /// alternative used by the upstream `model_serde/*` oracle fixtures.
    #[must_use]
    pub fn score_function(mut self, score_function: EScoreFunction) -> Self {
        self.score_function = score_function;
        self
    }

    /// Cardinality ceiling for the one-hot categorical encoding path
    /// (`one_hot_max_size`, upstream default 2 —
    /// `cat_feature_options.cpp:232-233`). A categorical column whose learn-set
    /// cardinality satisfies `1 < cardinality <= one_hot_max_size` routes to the
    /// one-hot path; above it, to the CTR path.
    #[must_use]
    pub fn one_hot_max_size(mut self, one_hot_max_size: u32) -> Self {
        self.one_hot_max_size = one_hot_max_size;
        self
    }

    /// Maximum feature-combination (tensor-CTR) projection length
    /// (`max_ctr_complexity` / upstream `MaxTensorComplexity`, default 4 —
    /// `cat_feature_options.cpp:231`). `1` emits SimpleCtrs only; `>= 2` admits
    /// CombinationCtrs of that length. Setting `1` is the ONLY in-engine way to
    /// suppress combination CTRs entirely.
    #[must_use]
    pub fn max_ctr_complexity(mut self, max_ctr_complexity: usize) -> Self {
        self.max_ctr_complexity = max_ctr_complexity;
        self
    }

    /// The simple-CTR type (`simple_ctr`). GENUINELY CONSUMED: it selects both
    /// the online producer and the final bake, so changing it changes the model.
    ///
    /// KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR
    /// descriptions (`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`,
    /// `catboost_options.cpp:439-453`). This crate models ONE description with a
    /// prior LIST — set the type here and the full prior list via
    /// [`CatBoostBuilder::simple_ctr_priors`]. A simultaneous
    /// `[Borders, Counter]` configuration is NOT representable.
    #[must_use]
    pub fn simple_ctr(mut self, simple_ctr: ECtrType) -> Self {
        self.simple_ctr = simple_ctr;
        self
    }

    /// The FULL prior list for [`CatBoostBuilder::simple_ctr`] — one CTR column
    /// is generated per prior.
    ///
    /// Each value is a prior NUMERATOR over an implicit denominator of `1`:
    /// upstream rejects `Prior=<n>/<d>` with `d != 1` on CPU
    /// (`ctr_helper.cpp:50`), which is exactly why the engine pins
    /// `prior_denom: 1.0`.
    #[must_use]
    pub fn simple_ctr_priors(mut self, simple_ctr_priors: Vec<f64>) -> Self {
        self.simple_ctr_priors = simple_ctr_priors;
        self
    }

    /// The combination-CTR type (`combinations_ctr`), applied to tensor
    /// (multi-column) projections. Consumed whenever
    /// [`CatBoostBuilder::max_ctr_complexity`] is `>= 2`. Carries the same
    /// single-description parity gap as [`CatBoostBuilder::simple_ctr`].
    #[must_use]
    pub fn combinations_ctr(mut self, combinations_ctr: ECtrType) -> Self {
        self.combinations_ctr = combinations_ctr;
        self
    }

    /// The FULL prior list for [`CatBoostBuilder::combinations_ctr`]; unit
    /// denominators, as for [`CatBoostBuilder::simple_ctr_priors`].
    #[must_use]
    pub fn combinations_ctr_priors(mut self, combinations_ctr_priors: Vec<f64>) -> Self {
        self.combinations_ctr_priors = combinations_ctr_priors;
        self
    }

    /// Whether the Counter CTR tally folds the eval sets into the counts
    /// ([`CounterCalcMethod::Full`]) or skips them
    /// ([`CounterCalcMethod::SkipTest`], the upstream default —
    /// `cat_feature_options.cpp:234`).
    ///
    /// **This flag is observable only when an eval set is present.** Training on
    /// a learn set alone produces bit-identical models under either setting
    /// (measured max|diff| `0.000e+00` learn-only versus `4.010e-01` with an
    /// eval set — the E23 gate). It is therefore NOT a blanket parity control:
    /// it only bites through the eval-set training entrypoint.
    #[must_use]
    pub fn counter_calc_method(mut self, counter_calc_method: CounterCalcMethod) -> Self {
        self.counter_calc_method = counter_calc_method;
        self
    }

    /// The per-iteration eval-set validation metric (`eval_metric`, TRAIN-07).
    /// Unset derives it from the loss ([`EvalMetric::for_loss`]).
    ///
    /// Consumed ONLY through an eval-set fit ([`CatBoostBuilder::fit_with_eval`] /
    /// [`CatBoostBuilder::fit_with_eval_sets`]) — a learn-only [`CatBoostBuilder::fit`]
    /// computes no validation metric, so setting this alone changes nothing.
    ///
    /// Note this REPLACES any metric previously installed by
    /// [`CatBoostBuilder::custom_metric`] (both write the same field).
    #[must_use]
    pub fn eval_metric(mut self, eval_metric: EvalMetric) -> Self {
        self.eval_metric = Some(eval_metric);
        self
    }

    /// Overfitting-detector type (`od_type`, TRAIN-06). Eval-set-only: with no
    /// eval set there is no metric curve to detect on.
    #[must_use]
    pub fn od_type(mut self, od_type: EOverfittingDetectorType) -> Self {
        self.od_type = od_type;
        self
    }

    /// Overfitting-detector stop threshold (`od_pval`). `0.0` (the default) makes
    /// IncToDec / Wilcoxon inactive; [`EOverfittingDetectorType::Iter`] ignores it.
    #[must_use]
    pub fn od_pval(mut self, od_pval: f64) -> Self {
        self.od_pval = od_pval;
        self
    }

    /// Overfitting-detector wait iterations (`od_wait`) — how many non-improving
    /// iterations to tolerate before stopping.
    #[must_use]
    pub fn od_wait(mut self, od_wait: usize) -> Self {
        self.od_wait = od_wait;
        self
    }

    /// Upstream's `early_stopping_rounds`: stop after `rounds` iterations without
    /// an eval-metric improvement.
    ///
    /// This is the exact pair upstream sets for that kwarg — `od_type = Iter`
    /// (the threshold-free "N rounds since the best" detector) plus
    /// `od_wait = rounds` — surfaced as ONE setter so a caller cannot set the
    /// wait while leaving the detector off, which would silently disable early
    /// stopping entirely.
    #[must_use]
    pub fn early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.od_type = EOverfittingDetectorType::Iter;
        self.od_wait = rounds;
        self
    }

    /// `use_best_model`: track the best eval-metric iteration and truncate the
    /// returned model to `best_iteration + 1` trees. Eval-set-only.
    #[must_use]
    pub fn use_best_model(mut self, use_best_model: bool) -> Self {
        self.use_best_model = use_best_model;
        self
    }

    /// The boosting type (`boosting_type`): [`EBoostingType::Plain`] (the CPU
    /// default) or [`EBoostingType::Ordered`] (the anti-leakage ordered
    /// approximant path).
    #[must_use]
    pub fn boosting_type(mut self, boosting_type: EBoostingType) -> Self {
        self.boosting_type = boosting_type;
        self
    }

    /// Whether the learn dataset is TIME-ORDERED (`has_time`). `true` skips the
    /// initial learn-set shuffle, preserving the natural object order.
    #[must_use]
    pub fn has_time(mut self, has_time: bool) -> Self {
        self.has_time = has_time;
        self
    }

    /// Number of random permutations for the multi-permutation fold machinery
    /// (`permutation_count`, upstream default 4).
    #[must_use]
    pub fn permutation_count(mut self, permutation_count: usize) -> Self {
        self.permutation_count = permutation_count;
        self
    }

    /// Tail-growth multiplier for the dynamic (ordered) fold body/tail
    /// (`fold_len_multiplier`, upstream default 2.0).
    #[must_use]
    pub fn fold_len_multiplier(mut self, fold_len_multiplier: f64) -> Self {
        self.fold_len_multiplier = fold_len_multiplier;
        self
    }

    /// Per-float-feature MULTIPLICATIVE candidate-gain weights
    /// (`feature_weights`). An empty vector (the default) weights every feature
    /// `1.0`; an out-of-range index also falls back to `1.0`.
    #[must_use]
    pub fn feature_weights(mut self, feature_weights: Vec<f64>) -> Self {
        self.feature_weights = feature_weights;
        self
    }

    /// Per-float-feature SUBTRACTIVE first-use penalties
    /// (`first_feature_use_penalties`), scaled by
    /// [`CatBoostBuilder::penalties_coefficient`] and applied while the feature is
    /// still unused anywhere in the model being built.
    #[must_use]
    pub fn first_feature_use_penalties(mut self, penalties: Vec<f64>) -> Self {
        self.first_feature_use_penalties = penalties;
        self
    }

    /// Per-float-feature SUBTRACTIVE per-object penalties
    /// (`per_object_feature_penalties`), scaled by
    /// [`CatBoostBuilder::penalties_coefficient`] and by the unused-document count.
    #[must_use]
    pub fn per_object_feature_penalties(mut self, penalties: Vec<f64>) -> Self {
        self.per_object_feature_penalties = penalties;
        self
    }

    /// The scaling coefficient multiplying BOTH penalty vectors
    /// (`penalties_coefficient`, upstream default `1.0`). Inert while both
    /// vectors are empty.
    #[must_use]
    pub fn penalties_coefficient(mut self, penalties_coefficient: f64) -> Self {
        self.penalties_coefficient = penalties_coefficient;
        self
    }

    /// Per-float-feature monotone constraints (`monotone_constraints`): `+1`
    /// (output non-decreasing in that feature), `-1` (non-increasing), `0`
    /// (free). Applied as an isotonic (PAVA) projection over the per-leaf deltas
    /// after the tree structure is built.
    ///
    /// OBLIVIOUS-ONLY: combining a non-empty constraint vector with a
    /// non-symmetric [`CatBoostBuilder::grow_policy`] is rejected at `fit()` by
    /// the engine's typed guard (upstream rejects it too), never silently ignored.
    #[must_use]
    pub fn monotone_constraints(mut self, monotone_constraints: Vec<i8>) -> Self {
        self.monotone_constraints = monotone_constraints;
        self
    }

    /// The tree grow policy (`grow_policy`). [`EGrowPolicy::SymmetricTree`] is
    /// the oblivious default; `Lossguide` / `Depthwise` grow a non-symmetric node
    /// graph; `Region` is rejected at `fit()`.
    #[must_use]
    pub fn grow_policy(mut self, grow_policy: EGrowPolicy) -> Self {
        self.grow_policy = grow_policy;
        self
    }

    /// Maximum leaf count for the Lossguide grower (`max_leaves`, default 31).
    /// Ignored by SymmetricTree / Depthwise, which `depth` bounds.
    #[must_use]
    pub fn max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    /// Minimum document count required to split a leaf (`min_data_in_leaf`,
    /// default 1). Read by the leaf-wise growers.
    #[must_use]
    pub fn min_data_in_leaf(mut self, min_data_in_leaf: usize) -> Self {
        self.min_data_in_leaf = min_data_in_leaf;
        self
    }

    /// Explicit per-class weights (`class_weights`): each object's weight is
    /// multiplied by `class_weights[class_of(object)]`, where the class is the
    /// object's integer label.
    ///
    /// CLASSIFICATION ONLY — combining it with a regression loss is rejected at
    /// `fit()` (upstream rejects it too), because "the class of an object" is
    /// undefined for a continuous target.
    #[must_use]
    pub fn class_weights(mut self, class_weights: Vec<f64>) -> Self {
        self.class_weights = class_weights;
        self
    }

    /// Derive the class weights automatically from the class distribution
    /// (`auto_class_weights`).
    ///
    /// Mutually exclusive with [`CatBoostBuilder::class_weights`] and
    /// [`CatBoostBuilder::scale_pos_weight`]: all three write the SAME per-object
    /// weight, so combining them is rejected at `fit()` rather than resolved by
    /// precedence, which would silently drop one the caller set on purpose.
    #[must_use]
    pub fn auto_class_weights(mut self, auto_class_weights: AutoClassWeights) -> Self {
        self.auto_class_weights = auto_class_weights;
        self
    }

    /// Binary-classification positive-class weight (`scale_pos_weight`, upstream
    /// default `1.0`) — exactly `class_weights = [1.0, scale_pos_weight]`.
    ///
    /// Binary only, and mutually exclusive with the other two class-weight
    /// controls (see [`CatBoostBuilder::auto_class_weights`]).
    #[must_use]
    pub fn scale_pos_weight(mut self, scale_pos_weight: f64) -> Self {
        self.scale_pos_weight = scale_pos_weight;
        self
    }

    /// Feature indices the tree search must never split on
    /// (`ignored_features`).
    ///
    /// Indices are in upstream's FLAT space, spanning BOTH feature kinds: a
    /// pool's categorical columns are always a trailing block, so index `i`
    /// names float feature `i` while `i >= n_float` names categorical column
    /// `i - n_float`. Both kinds are accepted, which is what upstream does and
    /// what a migrating script expects.
    ///
    /// Implemented WITHOUT physically dropping the column, so the model's
    /// feature INDEXING is unchanged: `predict` still takes the full-width pool,
    /// and every feature-importance / SHAP index keeps meaning the same input
    /// column. An ignored feature simply contributes nothing. A float feature is
    /// given an EMPTY border set; a categorical column is blanked to a single
    /// repeated value, which routes it to `Skip` (upstream's own
    /// skip-cardinality-`<=1` rule). Neither yields a candidate split at any
    /// level.
    ///
    /// An out-of-range index — one past the END of the flat space — is rejected
    /// at `fit()` rather than ignored: a typo that silently ignores nothing is
    /// exactly the failure this parameter is used to prevent.
    #[must_use]
    pub fn ignored_features(mut self, ignored_features: Vec<usize>) -> Self {
        self.ignored_features = ignored_features;
        self
    }

    /// Map the builder fields onto the internal [`BoostParams`].
    ///
    /// PARAM-01: the overfitting-detector / `use_best_model` / `eval_metric` /
    /// penalty / grow-policy / boosting-scheme fields are now READ FROM THE
    /// BUILDER instead of being pinned to literals here. `new()` seeds each with
    /// the value that was pinned, so an untouched builder emits an identical
    /// `BoostParams` (the D-04 no-regression gate).
    fn boost_params(&self) -> BoostParams {
        BoostParams {
            // `Loss` is no longer `Copy` (Phase 6.2, D-6.2-05 — the Wave-3
            // MultiQuantile variant carries an owned Vec<f64>); clone out of the
            // borrowed builder. Cheap for the current parameter-light variants.
            loss: self.loss.clone(),
            iterations: self.iterations,
            depth: self.depth,
            learning_rate: self.learning_rate,
            auto_learning_rate: self.auto_learning_rate,
            l2_leaf_reg: self.l2_leaf_reg,
            random_strength: self.random_strength,
            boost_from_average: self.boost_from_average,
            leaf_method: self.leaf_method,
            bootstrap_type: self.bootstrap_type,
            subsample: self.subsample,
            bagging_temperature: self.bagging_temperature,
            random_seed: self.random_seed,
            od_type: self.od_type,
            od_pval: self.od_pval,
            od_wait: self.od_wait,
            use_best_model: self.use_best_model,
            // Custom eval metric (LOSS-07) when set via `custom_metric`; else the
            // train loop derives it from the loss (`EvalMetric::for_loss`).
            eval_metric: self.eval_metric.clone(),
            // The facade now surfaces the categorical / CTR config (F01-F05);
            // each field defaults to the upstream value in `new()`, so a
            // builder that never touches them is byte-equivalent to the
            // previously-pinned form (guarded by `builder_test`'s
            // `untouched_builder_emits_the_canonical_ctr_defaults`).
            one_hot_max_size: self.one_hot_max_size,
            permutation_count: self.permutation_count,
            fold_len_multiplier: self.fold_len_multiplier,
            simple_ctr: self.simple_ctr,
            simple_ctr_priors: self.simple_ctr_priors.clone(),
            counter_calc_method: self.counter_calc_method,
            boosting_type: self.boosting_type,
            max_ctr_complexity: self.max_ctr_complexity,
            combinations_ctr: self.combinations_ctr,
            combinations_ctr_priors: self.combinations_ctr_priors.clone(),
            // Split-score function (`score_function`). The facade now surfaces
            // this via `.score_function()`, defaulting to the catboost CPU
            // default (Cosine, oblivious_tree_options.cpp:22) through
            // `score_function_default()` in `new()`.
            score_function: self.score_function,
            has_time: self.has_time,
            feature_weights: self.feature_weights.clone(),
            first_feature_use_penalties: self.first_feature_use_penalties.clone(),
            per_object_feature_penalties: self.per_object_feature_penalties.clone(),
            penalties_coefficient: self.penalties_coefficient,
            monotone_constraints: self.monotone_constraints.clone(),
            grow_policy: self.grow_policy,
            max_leaves: self.max_leaves,
            min_data_in_leaf: self.min_data_in_leaf,
        }
    }

    /// Train on `pool`, returning the trained facade [`Model`].
    ///
    /// Computes each float feature's quantization borders from the pool via the
    /// Phase-2 greedy-logsum binarizer, narrows the SoA float columns to `f32`
    /// (the feature storage type the apply path uses), and runs the plain
    /// boosting loop over [`CpuBackend`]. The resulting canonical
    /// [`cb_model::Model`] carries the per-tree `leaf_weights` and the
    /// `float_feature_borders` it was scored against (so later
    /// predict/serialize/explain need no pool).
    ///
    /// # Errors
    /// Returns [`CatBoostError::Train`] for any training failure (degenerate
    /// input, depth exceeded, runtime gradient error).
    pub fn fit(&self, pool: &Pool) -> Result<Model, CatBoostError> {
        // PARAM-01: `fit` is now the no-eval-set case of the shared inner. The
        // engine's own `train`/`train_cat` are themselves literal delegations to
        // `train_with_eval_sets`/`train_cat_with_eval_sets` with `&[]` sets and a
        // `None` history (boosting.rs:2325-2350, 2527-2551), so routing through
        // the inner with an empty set list is byte-identical to the previous
        // direct call — the D-04 / F21 no-regression gate.
        Ok(self.fit_inner(pool, &[])?.model)
    }

    /// Train on `pool` with ONE held-out eval set, returning the model together
    /// with its per-iteration eval-metric curve.
    ///
    /// This is the entry point that makes the eval-set-only parameters
    /// observable: [`CatBoostBuilder::od_type`] /
    /// [`CatBoostBuilder::early_stopping_rounds`] (stop early),
    /// [`CatBoostBuilder::use_best_model`] (truncate to the best iteration),
    /// [`CatBoostBuilder::eval_metric`] (which metric is tracked), and
    /// [`CatBoostBuilder::counter_calc_method`] (whether the Counter CTR tally
    /// folds the eval set in). With a learn set alone every one of those is inert.
    ///
    /// # Errors
    /// As [`CatBoostBuilder::fit`], plus [`CatBoostError::FeatureMismatch`] when
    /// the eval pool's float / categorical width disagrees with the learn pool's,
    /// and any detector-construction or eval-metric failure.
    pub fn fit_with_eval(&self, pool: &Pool, eval_pool: &Pool) -> Result<FitResult, CatBoostError> {
        self.fit_inner(pool, &[eval_pool])
    }

    /// Train on `pool` with ZERO OR MORE held-out eval sets.
    ///
    /// `eval_pools[0]` is the PRIMARY set — the only one the overfitting detector
    /// and the `use_best_model` tracker consume; the rest are evaluated and
    /// logged into [`FitResult::eval_history`] only. Passing an empty slice is
    /// exactly [`CatBoostBuilder::fit`].
    ///
    /// # Errors
    /// As [`CatBoostBuilder::fit_with_eval`].
    pub fn fit_with_eval_sets(
        &self,
        pool: &Pool,
        eval_pools: &[&Pool],
    ) -> Result<FitResult, CatBoostError> {
        self.fit_inner(pool, eval_pools)
    }

    /// Whether `loss` defines a per-object CLASS, which is what the class-weight
    /// controls need in order to mean anything.
    fn loss_is_classification(loss: &Loss) -> bool {
        matches!(
            loss,
            Loss::Logloss
                | Loss::CrossEntropy
                | Loss::Focal { .. }
                | Loss::MultiClass
                | Loss::MultiClassOneVsAll
                | Loss::MultiLogloss
                | Loss::MultiCrossEntropy
        )
    }

    /// PARAM-03: resolve the effective per-object training weights from the
    /// class-weight controls, or `None` when none is active (in which case the
    /// caller passes the pool's own weights through UNCHANGED — the byte-identical
    /// default path).
    ///
    /// The three controls all write the same per-object weight, so at most one may
    /// be active; the conflict is an error rather than a precedence rule.
    fn resolve_weights(&self, pool: &Pool) -> Result<Option<Vec<f64>>, CatBoostError> {
        let explicit = !self.class_weights.is_empty();
        let auto = self.auto_class_weights != AutoClassWeights::None;
        // Compared against the default rather than "is it set", because the
        // builder cannot distinguish an unset field from one explicitly set to
        // the default — and `scale_pos_weight = 1.0` is a no-op either way.
        let scaled = (self.scale_pos_weight - 1.0).abs() > f64::EPSILON;
        let active = usize::from(explicit) + usize::from(auto) + usize::from(scaled);
        if active == 0 {
            return Ok(None);
        }
        if active > 1 {
            return Err(CatBoostError::InvalidConfig(
                "class_weights, auto_class_weights and scale_pos_weight all set the same \
                 per-object weight; set at most one (combining them would silently discard \
                 all but one)"
                    .to_owned(),
            ));
        }
        if !Self::loss_is_classification(&self.loss) {
            return Err(CatBoostError::InvalidConfig(format!(
                "the class-weight controls (class_weights / auto_class_weights / \
                 scale_pos_weight) need a per-object CLASS, which loss {:?} does not define; \
                 they apply to classification losses only",
                self.loss
            )));
        }

        // Derive each object's class index from its label. A classification label
        // is an integer class id (0/1 for the binary losses), so a non-integral or
        // negative label is a typed error rather than a silent `as usize` truncation
        // that would bucket 0.7 and 0.2 into the same class.
        let mut classes: Vec<usize> = Vec::with_capacity(pool.label().len());
        for (i, &label) in pool.label().iter().enumerate() {
            if !label.is_finite() || label < 0.0 || (label.fract()).abs() > f64::EPSILON {
                return Err(CatBoostError::InvalidConfig(format!(
                    "the class-weight controls require integer class labels, but label[{i}] \
                     is {label}; for a probabilistic CrossEntropy target there is no class \
                     to weight"
                )));
            }
            classes.push(label as usize);
        }

        let observed = classes.iter().copied().max().map_or(0, |m| m + 1);
        let (weights, class_count) = if explicit {
            (
                self.class_weights.iter().map(|&w| w as f32).collect(),
                self.class_weights.len(),
            )
        } else if scaled {
            (vec![1.0_f32, self.scale_pos_weight as f32], 2)
        } else {
            // Auto weights are DERIVED from the observed distribution, so the class
            // count is whatever the labels span.
            //
            // `summary_class_weights` requires one item weight PER OBJECT, while an
            // unweighted `Pool` carries an EMPTY weight vector (the all-ones
            // convention `train` and `resolve_object_weights` both accept). Densify
            // here rather than loosening the primitive: its length check is what
            // catches a genuinely mismatched caller.
            let dense: Vec<f64> = if pool.weights().is_empty() {
                vec![1.0; classes.len()]
            } else {
                pool.weights().to_vec()
            };
            (
                cb_data::auto_class_weights(self.auto_class_weights, &classes, &dense, observed)?,
                observed,
            )
        };
        if observed > class_count {
            return Err(CatBoostError::InvalidConfig(format!(
                "the labels span {observed} classes but only {class_count} class weight(s) \
                 were given"
            )));
        }
        Ok(Some(cb_data::resolve_object_weights(
            &weights,
            pool.weights(),
            &classes,
        )?))
    }

    /// PARAM-03: apply `ignored_features`, whose indices are in upstream's FLAT
    /// (float-then-categorical) space.
    ///
    /// A pool's categorical columns are always a TRAILING block — every ingest
    /// path rejects a categorical column followed by a float one — so the flat
    /// space is exactly `[0, n_float)` float features followed by
    /// `[n_float, n_float + n_cat)` categorical ones, and a float feature's flat
    /// index equals its float index.
    ///
    /// A FLOAT index is handled here: its border set is blanked, leaving it with
    /// no candidate split at any level while preserving every other feature's
    /// index. A CATEGORICAL index cannot be handled here (it has no border set),
    /// so it is returned as a LOCAL categorical index for the caller to mask out
    /// of the categorical columns; see `mask_ignored_cat_columns`.
    ///
    /// Before this split existed, every index `>= n_float` was rejected as
    /// out-of-range, so `ignored_features` naming a categorical column — which
    /// upstream accepts, and which is the only way to ignore one at all — aborted
    /// the fit.
    ///
    /// Returns the ASCENDING, de-duplicated local categorical indices to ignore.
    fn apply_ignored_features(
        &self,
        borders: &mut [Vec<f64>],
        n_cat: usize,
    ) -> Result<Vec<usize>, CatBoostError> {
        // Read BEFORE the loop so the error message can name both widths without
        // borrowing `borders` immutably inside the `get_mut` arm (E0502).
        let n_float = borders.len();
        let flat_width = n_float.saturating_add(n_cat);
        let mut ignored_cat = Vec::new();
        for &f in &self.ignored_features {
            if let Some(slot) = borders.get_mut(f) {
                slot.clear();
            } else if f < flat_width {
                // `f >= n_float` and inside the flat width -> a categorical column.
                ignored_cat.push(f - n_float);
            } else {
                return Err(CatBoostError::InvalidConfig(format!(
                    "ignored_features index {f} is out of range for a pool with {n_float} \
                     float feature(s) and {n_cat} categorical feature(s) (flat indices \
                     0..{flat_width})"
                )));
            }
        }
        ignored_cat.sort_unstable();
        ignored_cat.dedup();
        Ok(ignored_cat)
    }

    /// The single training implementation behind `fit` / `fit_with_eval` /
    /// `fit_with_eval_sets`.
    fn fit_inner(&self, pool: &Pool, eval_pools: &[&Pool]) -> Result<FitResult, CatBoostError> {
        // CB_GPU_PROF host-stage attribution (shares the device profiler's env gate; cold
        // when unset — the checks below never allocate or print).
        let prof = std::env::var_os("CB_GPU_PROF").is_some_and(|v| v != "0");
        let prof_t = std::time::Instant::now();

        let params = self.boost_params();
        // SPD-03: kick off the background device-kernel warm-up NOW so JIT
        // compilation overlaps the host-side fit-prep below (and the caller's pool
        // ingestion already behind us) instead of serializing inline with training —
        // a cold GPU fit otherwise pays ~2-3 s of driver compilation inside
        // `begin`/tree-0 (P100 diag 2026-08-08). Best-effort and detached; also
        // enables CubeCL's disk compilation cache so later processes skip JIT
        // entirely. No-op effect on the trained model: warm-up only compiles.
        #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
        cb_backend::gpu_runtime::warmup::spawn_fit_warmup(
            params.loss.clone(),
            params.depth,
            self.border_count,
            params.score_function,
        );

        // SoA float columns as f32 (the feature storage type; the apply path
        // binarizes f32 against the borders) AND the per-float-feature
        // quantization borders (Phase-2 greedy logsum; NaN sentinel is off for the
        // numeric first-slice surface — NaN-free features are always Forbidden
        // regardless). The f64 pool columns are narrowed ONCE (parallel over the
        // disjoint per-column data), and the borders are then derived from the
        // narrowed f32 columns via `select_borders_greedy_logsum_f32` — the same
        // `v as f32` narrowing the f64 entry performs internally, so the border
        // set is byte-identical while the full-width f64 columns are read exactly
        // once instead of twice (this fit-prep stage is the largest host term at
        // scale — the CB_GPU_PROF timer below attributes to it). Each `par_iter`'s
        // indexed map preserves output order, so both results stay byte-identical
        // to the fully-serial form.
        // SPD-03 wave 3: ONE fused parallel pass per column — narrow, then derive
        // the borders from the just-narrowed column while it is still cache-warm —
        // instead of two full passes over all columns (the second re-read of the
        // whole matrix from RAM was pure waste on the 4-vCPU cloud hosts where this
        // stage is the largest remaining host term). Outputs are byte-identical:
        // same narrowing, same border derivation, indexed map preserves order.
        // The two AtomicU64s accumulate per-column CPU nanos for the CB_GPU_PROF
        // attribution line (thread-time, not wall — labeled as such).
        let narrow_ns = std::sync::atomic::AtomicU64::new(0);
        let borders_ns = std::sync::atomic::AtomicU64::new(0);
        // SPD-03 wave 3: an ingestion source whose input was ALREADY f32 (the
        // Python NumPy path) attaches a bit-exact narrowing cache — then the whole
        // narrowing pass vanishes and only the border derivation runs. The cache is
        // trusted only when its shape matches exactly (Pool doc: never load-bearing).
        let cached_f32 = pool.float_features_f32();
        let cache_valid = cached_f32.len() == pool.float_features().len()
            && cached_f32
                .iter()
                .zip(pool.float_features().iter())
                .all(|(c, f)| c.len() == f.len());
        // SPD-03 wave 6: the over-cap border sample is drawn from a FIXED-seed
        // stream parameterized only by the object count, so every column samples
        // the SAME index set. Draw it once here (sorted ascending — the gather
        // becomes a forward streaming read) instead of once per column; the
        // per-column border set is byte-identical (`_presampled` doc). Columns
        // at or under the cap keep the full-column path (`None`).
        let shared_sample: Option<Vec<u32>> = (pool.n_rows()
            > cb_data::MAX_SUBSET_SIZE_FOR_BUILD_BORDERS)
            .then(|| {
                cb_data::sample_indices_for_build_borders(
                    pool.n_rows(),
                    cb_data::MAX_SUBSET_SIZE_FOR_BUILD_BORDERS,
                )
            });
        // `leaf_estimation_backtracking=Armijo` is GPU-ONLY upstream
        // (`catboost_options.cpp:664`). Refusing it here is real parity: silently
        // training with AnyImprovement instead would hand back a model the user
        // did not ask for.
        if !self.leaf_estimation_backtracking.is_cpu_supported() {
            return Err(CatBoostError::InvalidConfig(format!(
                "Backtracking type {} is supported only on GPU; the CPU training \
                 path admits No and AnyImprovement",
                self.leaf_estimation_backtracking.as_str()
            )));
        }

        // `nan_mode=Forbidden` refuses a NaN-bearing learn column, mirroring
        // upstream `quantization.cpp:320`. Checked BEFORE any border work so the
        // rejection is cheap and cannot race the parallel pass below.
        if self.nan_mode == NanMode::Forbidden {
            for (f, col) in pool.float_features().iter().enumerate() {
                if col.iter().any(|v| v.is_nan()) {
                    return Err(CatBoostError::InvalidConfig(format!(
                        "Feature #{f}: There are nan factors and nan values for float \
                         features are not allowed. Set nan_mode != Forbidden."
                    )));
                }
            }
        }

        let border_type = self.feature_border_type;
        let nan_mode = self.nan_mode;
        let borders_for_col = |col: &[f32]| -> Vec<f64> {
            // A NaN-bearing column gets the NanMode sentinel that isolates the
            // missing values into their own bin. `Min` PREPENDS `f32::MIN`;
            // `Max` APPENDS `f32::MAX` after selection — a tail value, so it
            // cannot perturb which real borders get chosen (the same ordering
            // `cb_data::quantize` documents). Either way the sentinel consumes
            // one slot of the border budget.
            //
            // A NaN-FREE column takes the untouched pre-existing path, so no
            // existing fixture moves.
            let has_nan = col.iter().any(|v| v.is_nan());
            if has_nan {
                return match nan_mode {
                    NanMode::Min => {
                        cb_data::select_borders_f32(col, self.border_count, border_type, true)
                    }
                    NanMode::Max => {
                        let mut borders = cb_data::select_borders_f32(
                            col,
                            self.border_count.saturating_sub(1),
                            border_type,
                            false,
                        );
                        borders.push(f64::from(f32::MAX));
                        borders
                    }
                    // Rejected above; a NaN column never reaches here.
                    NanMode::Forbidden => {
                        cb_data::select_borders_f32(col, self.border_count, border_type, false)
                    }
                };
            }
            // The presampled fast path exists only for the default GreedyLogSum
            // binarizer (it is the fit-prep hot path). The other six route
            // through `select_borders_f32`, which draws the SAME fixed-seed
            // sample internally — the index set depends only on the object
            // count, so the borders are identical; only the (once-per-column)
            // draw is repeated.
            if border_type == EBorderSelectionType::GreedyLogSum {
                return match shared_sample.as_deref() {
                    // The shared draw is valid only for full-length columns; the
                    // Pool invariant makes every float column pool.n_rows() long,
                    // and the guard keeps a hypothetical short column correct by
                    // falling back to the self-sampling entry.
                    Some(sample) if col.len() == pool.n_rows() => {
                        cb_data::select_borders_greedy_logsum_f32_presampled(
                            col,
                            sample,
                            self.border_count,
                            false,
                        )
                    }
                    _ => select_borders_greedy_logsum_f32(col, self.border_count, false),
                };
            }
            cb_data::select_borders_f32(col, self.border_count, border_type, false)
        };
        let (owned_values, mut feature_borders): (Option<Vec<Vec<f32>>>, Vec<Vec<f64>>) =
            if cache_valid && !cached_f32.is_empty() {
                let borders: Vec<Vec<f64>> = cached_f32
                    .par_iter()
                    .map(|col| {
                        let t1 = std::time::Instant::now();
                        let borders = borders_for_col(col);
                        if prof {
                            let ord = std::sync::atomic::Ordering::Relaxed;
                            borders_ns.fetch_add(t1.elapsed().as_nanos() as u64, ord);
                        }
                        borders
                    })
                    .collect();
                (None, borders)
            } else {
                let (values, borders) = pool
                    .float_features()
                    .par_iter()
                    .map(|col| {
                        let t0 = std::time::Instant::now();
                        let narrowed: Vec<f32> = col.iter().map(|&v| v as f32).collect();
                        let t1 = std::time::Instant::now();
                        let borders = borders_for_col(&narrowed);
                        if prof {
                            let ord = std::sync::atomic::Ordering::Relaxed;
                            narrow_ns.fetch_add((t1 - t0).as_nanos() as u64, ord);
                            borders_ns.fetch_add(t1.elapsed().as_nanos() as u64, ord);
                        }
                        (narrowed, borders)
                    })
                    .unzip();
                (Some(values), borders)
            };
        let feature_values: &[Vec<f32>] = owned_values.as_deref().unwrap_or(cached_f32);
        // PARAM-03: an ignored feature keeps its column and its index but loses
        // its borders, so the candidate enumeration never proposes a split on it.
        // Applied AFTER border selection (not before) so the borders of every
        // OTHER feature are byte-identical to a run without the parameter.
        //
        // `ignored_features` is indexed in upstream's FLAT space, so an index
        // naming a categorical column comes back here to be masked separately
        // (the categorical arm below); it is EMPTY for a float-only pool and for
        // every fit that ignores only float features.
        let ignored_cat = self.apply_ignored_features(&mut feature_borders, pool.n_cat_features())?;
        // PARAM-03: the effective training weights. `None` = no class-weight
        // control is active, and the pool's own weights pass through unchanged.
        let resolved_weights = self.resolve_weights(pool)?;
        let weights: &[f64] = resolved_weights.as_deref().unwrap_or_else(|| pool.weights());
        if prof {
            let ord = std::sync::atomic::Ordering::Relaxed;
            eprintln!(
                "CB_GPU_PROF fit-prep copy+borders elapsed={:.2}ms \
                 (cpu-time narrow={:.2}ms borders={:.2}ms)",
                prof_t.elapsed().as_secs_f64() * 1e3,
                narrow_ns.load(ord) as f64 / 1e6,
                borders_ns.load(ord) as f64 / 1e6,
            );
        }
        // Narrow each eval pool to the SAME f32 storage type the learn columns use
        // (the apply path binarizes f32 against the learn borders, so an eval set
        // held at f64 would be scored against a different quantization than the
        // learn set — the borders come from the LEARN pool only, never recomputed
        // per eval set).
        //
        // The width check is EAGER and typed: `train_with_eval_sets` indexes the
        // eval columns positionally, so a narrower eval pool would otherwise be a
        // silent wrong-feature evaluation rather than an error.
        let mut eval_values: Vec<Vec<Vec<f32>>> = Vec::with_capacity(eval_pools.len());
        for (i, ep) in eval_pools.iter().enumerate() {
            if ep.n_float_features() != pool.n_float_features() {
                return Err(CatBoostError::FeatureMismatch(format!(
                    "eval set {i} has {} float features, the learn pool has {}",
                    ep.n_float_features(),
                    pool.n_float_features()
                )));
            }
            if ep.n_cat_features() != pool.n_cat_features() {
                return Err(CatBoostError::FeatureMismatch(format!(
                    "eval set {i} has {} categorical features, the learn pool has {}",
                    ep.n_cat_features(),
                    pool.n_cat_features()
                )));
            }
            eval_values.push(
                ep.float_features()
                    .par_iter()
                    .map(|col| col.iter().map(|&v| v as f32).collect())
                    .collect(),
            );
        }
        let eval_sets: Vec<EvalSet> = eval_values
            .iter()
            .zip(eval_pools.iter())
            .map(|(values, ep)| EvalSet {
                feature_values: values,
                target: ep.label(),
                cat_columns: ep.cat_features(),
            })
            .collect();
        // Track the per-set metric curves ONLY when there is a set to track, so a
        // plain `fit()` allocates nothing and passes the same `None` the engine's
        // `train` does.
        let mut history = if eval_sets.is_empty() {
            None
        } else {
            Some(EvalMetricHistory::new(eval_sets.len()))
        };

        let prof_train_t = std::time::Instant::now();

        // (`params` was built at fit entry — before fit-prep — so the SPD-03 warm-up
        // could read the loss/depth/score keys; it is byte-identical to building it
        // here, `boost_params` is a pure constructor over `&self`.)
        // Compile-time backend selection (08-08): exactly one feature is active, so
        // exactly one `backend` binding is in scope. `train` is already generic over
        // `R: Runtime`, so it accepts either zero-sized backend with no other change.
        #[cfg(feature = "cpu")]
        let backend = CpuBackend;
        #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
        let backend = GpuBackend::default();

        // F09 / SPEC-CATF-08: a pool that DECLARES categorical columns routes
        // through `train_cat`, which returns the baked CTR tables alongside the
        // model. Before this branch existed, `fit()` always called the float-only
        // `train` and `pool.cat_features()` was never read, so every categorical
        // column was SILENTLY DROPPED — the fit succeeded and produced a model
        // that scored as though the column did not exist.
        //
        // The cat-free arm is left EXACTLY as it was (same `train` call, same
        // arguments, same order), so a numeric pool is bit-identical to the
        // pre-F09 result — the F21 no-regression gate.
        let canonical = if pool.cat_features().is_empty() {
            let trained = train_with_eval_sets(
                &backend,
                feature_values,
                &feature_borders,
                pool.label(),
                weights,
                &params,
                None,
                &eval_sets,
                history.as_mut(),
            )?;
            if prof {
                eprintln!(
                    "CB_GPU_PROF fit-train elapsed={:.2}ms",
                    prof_train_t.elapsed().as_secs_f64() * 1e3,
                );
            }
            cb_model::Model::from_trained(&trained, feature_borders)
        } else {
            // PARAM-03: mask out any categorical column named by
            // `ignored_features`. `Cow::Borrowed` — the zero-copy identity — for
            // every fit that ignores no categorical feature, which is the norm.
            let cat_columns = mask_ignored_cat_columns(pool.cat_features(), &ignored_cat);
            let (trained, baked) = train_cat_with_eval_sets(
                &backend,
                feature_values,
                &feature_borders,
                cat_columns.as_ref(),
                pool.label(),
                weights,
                &params,
                None,
                &eval_sets,
                history.as_mut(),
            )?;
            if prof {
                eprintln!(
                    "CB_GPU_PROF fit-train-cat elapsed={:.2}ms",
                    prof_train_t.elapsed().as_secs_f64() * 1e3,
                );
            }
            let model = cb_model::Model::from_trained(&trained, feature_borders);
            // A purely ONE-HOT-routed pool bakes no CTR table. Attaching an
            // empty `CtrData` would make `ctr_data.is_some()` true for a model
            // with no CTR split at all, which the predict-side
            // `needs_cat_columns()` predicate reads as "this is a CTR model".
            let model = if baked.tables.is_empty() {
                model
            } else {
                model.with_ctr_data(cb_model::CtrData::from_baked(&baked))
            };
            // Δ4: record the pool's DECLARED cat width, so the predict-side
            // width check never compares against a width derived from the
            // splits the model happened to choose (PLAN-CHECK CRITICAL-3).
            model.with_cat_feature_count(pool.n_cat_features())
        };
        Ok(FitResult {
            model: Model::from_canonical(canonical),
            eval_history: history.map(|h| h.per_set).unwrap_or_default(),
        })
    }
}

#[cfg(test)]
#[path = "builder_test.rs"]
mod builder_test;
