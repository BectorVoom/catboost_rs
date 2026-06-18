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

use cb_backend::CpuBackend;
use cb_compute::{
    CustomMetric, CustomMetricHandle, CustomObjective, CustomObjectiveHandle, LeafMethod, Loss,
};
use cb_data::{select_borders_greedy_logsum, Pool, QuantizeParams};
use cb_train::{
    boosting_type_default, combinations_ctr_default, combinations_ctr_priors_default,
    counter_calc_method_default, fold_len_multiplier_default, has_time_default,
    max_ctr_complexity_default,
    one_hot_max_size_default, permutation_count_default, score_function_default,
    simple_ctr_default, simple_ctr_priors_default, train, BoostParams, EBootstrapType,
    EOverfittingDetectorType, EvalMetric,
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
            // Pinned to the upstream default (cat_feature_options.cpp:231-232);
            // the facade does not yet surface categorical config, and the
            // numeric-only train path never exercises the one-hot branch.
            one_hot_max_size: one_hot_max_size_default(),
            // Pinned to the upstream defaults (RESEARCH Pitfall 6); the numeric
            // facade path needs no permutation, so these are inert here.
            permutation_count: permutation_count_default(),
            fold_len_multiplier: fold_len_multiplier_default(),
            // CTR config pinned to the upstream defaults (D-07 / Pitfall 6); the
            // numeric facade path bakes no CTR table, so these are inert here.
            simple_ctr: simple_ctr_default(),
            simple_ctr_priors: simple_ctr_priors_default(),
            counter_calc_method: counter_calc_method_default(),
            boosting_type: boosting_type_default(),
            // Tensor-CTR config pinned to the upstream defaults (D-07 / Pitfall
            // 6); the numeric facade path forms no combination, so these are
            // inert here.
            max_ctr_complexity: max_ctr_complexity_default(),
            combinations_ctr: combinations_ctr_default(),
            combinations_ctr_priors: combinations_ctr_priors_default(),
            // catboost CPU default split-score function (Cosine,
            // oblivious_tree_options.cpp:22); the facade does not surface it.
            score_function: score_function_default(),
            has_time: has_time_default(),
            feature_weights: cb_train::feature_weights_default(),
            first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
            per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
            penalties_coefficient: cb_train::penalties_coefficient_default(),
            monotone_constraints: cb_train::monotone_constraints_default(),
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
        // SoA float columns as f32 (the feature storage type; the apply path
        // binarizes f32 against the borders).
        let feature_values: Vec<Vec<f32>> = pool
            .float_features()
            .iter()
            .map(|col| col.iter().map(|&v| v as f32).collect())
            .collect();

        // Per-float-feature quantization borders from the pool (Phase-2 greedy
        // logsum). NaN sentinel is off for the numeric first-slice surface
        // (NaN-free features are always Forbidden regardless).
        let feature_borders: Vec<Vec<f64>> = pool
            .float_features()
            .iter()
            .map(|col| select_borders_greedy_logsum(col, self.border_count, false))
            .collect();

        let params = self.boost_params();
        let trained = train(
            &CpuBackend,
            &feature_values,
            &feature_borders,
            pool.label(),
            pool.weights(),
            &params,
            None,
        )?;

        let canonical = cb_model::Model::from_trained(&trained, feature_borders);
        Ok(Model::from_canonical(canonical))
    }
}
