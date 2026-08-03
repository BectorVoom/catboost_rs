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
    LeafMethod, Loss,
};
use cb_data::{select_borders_greedy_logsum, Pool, QuantizeParams};
use rayon::prelude::*;
use cb_train::{
    boosting_type_default, combinations_ctr_default, combinations_ctr_priors_default,
    counter_calc_method_default, fold_len_multiplier_default, has_time_default,
    max_ctr_complexity_default,
    one_hot_max_size_default, permutation_count_default, score_function_default,
    simple_ctr_default, simple_ctr_priors_default, train, train_cat, BoostParams,
    CounterCalcMethod, EBootstrapType, ECtrType, EOverfittingDetectorType, EvalMetric,
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
}

impl Default for CatBoostBuilder {
    fn default() -> Self {
        Self::new()
    }
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
            score_function: score_function_default(),
            one_hot_max_size: one_hot_max_size_default(),
            max_ctr_complexity: max_ctr_complexity_default(),
            simple_ctr: simple_ctr_default(),
            simple_ctr_priors: simple_ctr_priors_default(),
            combinations_ctr: combinations_ctr_default(),
            combinations_ctr_priors: combinations_ctr_priors_default(),
            counter_calc_method: counter_calc_method_default(),
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

    /// Map the builder fields onto the internal [`BoostParams`]. The
    /// overfitting-detector / `use_best_model` / `eval_metric` controls are off
    /// (the Phase-4 first-slice surface does not expose an eval set through the
    /// facade).
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
            od_type: EOverfittingDetectorType::None,
            od_pval: 0.0,
            od_wait: 0,
            use_best_model: false,
            // Custom eval metric (LOSS-07) when set via `custom_metric`; else the
            // train loop derives it from the loss (`EvalMetric::for_loss`).
            eval_metric: self.eval_metric.clone(),
            // The facade now surfaces the categorical / CTR config (F01-F05);
            // each field defaults to the upstream value in `new()`, so a
            // builder that never touches them is byte-equivalent to the
            // previously-pinned form (guarded by `builder_test`'s
            // `untouched_builder_emits_the_canonical_ctr_defaults`).
            one_hot_max_size: self.one_hot_max_size,
            // Pinned to the upstream defaults (RESEARCH Pitfall 6); the numeric
            // facade path needs no permutation, so these are inert here.
            permutation_count: permutation_count_default(),
            fold_len_multiplier: fold_len_multiplier_default(),
            simple_ctr: self.simple_ctr,
            simple_ctr_priors: self.simple_ctr_priors.clone(),
            counter_calc_method: self.counter_calc_method,
            boosting_type: boosting_type_default(),
            max_ctr_complexity: self.max_ctr_complexity,
            combinations_ctr: self.combinations_ctr,
            combinations_ctr_priors: self.combinations_ctr_priors.clone(),
            // Split-score function (`score_function`). The facade now surfaces
            // this via `.score_function()`, defaulting to the catboost CPU
            // default (Cosine, oblivious_tree_options.cpp:22) through
            // `score_function_default()` in `new()`.
            score_function: self.score_function,
            has_time: has_time_default(),
            feature_weights: cb_train::feature_weights_default(),
            first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
            per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
            penalties_coefficient: cb_train::penalties_coefficient_default(),
            monotone_constraints: cb_train::monotone_constraints_default(),
            grow_policy: cb_train::grow_policy_default(),
            max_leaves: cb_train::max_leaves_default(),
            min_data_in_leaf: cb_train::min_data_in_leaf_default(),
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
        // CB_GPU_PROF host-stage attribution (shares the device profiler's env gate; cold
        // when unset — the checks below never allocate or print).
        let prof = std::env::var_os("CB_GPU_PROF").is_some_and(|v| v != "0");
        let prof_t = std::time::Instant::now();

        // SoA float columns as f32 (the feature storage type; the apply path
        // binarizes f32 against the borders) AND the per-float-feature
        // quantization borders (Phase-2 greedy logsum; NaN sentinel is off for the
        // numeric first-slice surface — NaN-free features are always Forbidden
        // regardless). Both are independent `rayon`-parallel reductions over the
        // SAME `pool.float_features()`, reading disjoint per-column data and
        // writing disjoint outputs (`feature_values` narrows f64->f32,
        // `feature_borders` derives borders — neither reads the other's output),
        // so they run CONCURRENTLY via `rayon::join` rather than sequentially, to
        // shrink this fit-prep stage's latency (the CB_GPU_PROF timer below
        // specifically attributes to this stage). Each inner `par_iter`'s indexed
        // map preserves output order, so both results stay byte-identical to the
        // fully-serial form.
        let (feature_values, feature_borders): (Vec<Vec<f32>>, Vec<Vec<f64>>) = rayon::join(
            || {
                pool.float_features()
                    .par_iter()
                    .map(|col| col.iter().map(|&v| v as f32).collect())
                    .collect()
            },
            || {
                pool.float_features()
                    .par_iter()
                    .map(|col| select_borders_greedy_logsum(col, self.border_count, false))
                    .collect()
            },
        );
        if prof {
            eprintln!(
                "CB_GPU_PROF fit-prep copy+borders elapsed={:.2}ms",
                prof_t.elapsed().as_secs_f64() * 1e3,
            );
        }
        let prof_train_t = std::time::Instant::now();

        let params = self.boost_params();
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
            let trained = train(
                &backend,
                &feature_values,
                &feature_borders,
                pool.label(),
                pool.weights(),
                &params,
                None,
            )?;
            if prof {
                eprintln!(
                    "CB_GPU_PROF fit-train elapsed={:.2}ms",
                    prof_train_t.elapsed().as_secs_f64() * 1e3,
                );
            }
            cb_model::Model::from_trained(&trained, feature_borders)
        } else {
            let (trained, baked) = train_cat(
                &backend,
                &feature_values,
                &feature_borders,
                pool.cat_features(),
                pool.label(),
                pool.weights(),
                &params,
                None,
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
        Ok(Model::from_canonical(canonical))
    }
}

#[cfg(test)]
#[path = "builder_test.rs"]
mod builder_test;
