//! The facade [`Model`] (D-06 / D-07): a cohesive wrapper over the canonical
//! [`cb_model::Model`] exposing predict / predict_proba / save / load / SHAP /
//! feature-importance on a single object.
//!
//! - Prediction has an enum CORE — [`Model::predict_with`] taking a
//!   [`PredictionType`] — plus the [`Model::predict`] (`RawFormulaVal`) and
//!   [`Model::predict_proba`] (`Probability`) shorthands (D-06).
//! - Serialization delegates to `cb_model::{cbm, json}`; SHAP to
//!   `cb_model::shap_values`; feature importance to `cb_model::fstr` (D-07).
//! - Every fallible method returns [`crate::CatBoostError`]; a wrong-width pool
//!   surfaces as [`CatBoostError::FeatureMismatch`] (T-04-05-02) — never an
//!   out-of-bounds access.

use std::path::Path;

use cb_data::Pool;
use cb_model::{
    apply_prediction_type, export_onnx, interaction, load_cbm, load_json, partial_dependence,
    predict_raw, predict_raw_staged, prediction_values_change, prediction_values_change_with_data,
    save_cbm, save_json, shap_values, sum_models, FeatureImportanceType, PartialDependence,
    PredictionType,
};
use cb_train::{parse_metric, EvalMetric};

use crate::error::CatBoostError;

/// Map a trained-loss name to its Min-optimized [`EvalMetric`] for
/// `LossFunctionChange` (FL-03). Only the supported numeric losses `RMSE`,
/// `MAE`, `MAPE`, `Quantile[:alpha=..]`, and `Logloss` are accepted — the base
/// name (before any `:param` suffix) is checked case-insensitively against this
/// explicit allow-list, then the full descriptor is parsed by
/// [`cb_train::parse_metric`] (which resolves e.g. `Quantile`'s `alpha`). A
/// Max-optimized (`AUC`/`Accuracy`/`R²`), ranking (`NDCG`/…), out-of-scope
/// (`MSLE`), or unknown loss yields [`CatBoostError::UnsupportedLoss`], never a
/// silent fallback. A malformed *param* on a supported base name also yields
/// `UnsupportedLoss`, but with the underlying [`cb_train::parse_metric`] reason
/// appended so the caller can see the loss IS supported and only the param is
/// wrong.
///
/// The base name is `trim`med (like `parse_metric` itself) so a cosmetically
/// whitespaced descriptor (`" RMSE"`) is not spuriously rejected.
///
/// **Quantile caveat:** the model file carries no loss metadata (Q1), so the
/// caller supplies `loss`; a bare `"Quantile"` resolves the default `alpha=0.5`.
/// If the model was trained at a different `alpha`, pass the full
/// `"Quantile:alpha=.."` descriptor — otherwise the pinball final error is
/// evaluated at the wrong quantile (the facade cannot detect the mismatch).
fn eval_metric_for_loss(loss: &str) -> Result<EvalMetric, CatBoostError> {
    let lower = loss.to_ascii_lowercase();
    let base = lower.split(':').next().unwrap_or(lower.as_str()).trim();
    match base {
        "rmse" | "logloss" | "mae" | "mape" | "quantile" => parse_metric(loss)
            .map_err(|e| CatBoostError::UnsupportedLoss(format!("{loss} ({e})"))),
        _ => Err(CatBoostError::UnsupportedLoss(loss.to_owned())),
    }
}

/// A trained CatBoost model (the published facade over [`cb_model::Model`]).
///
/// Obtain one from [`crate::CatBoostBuilder::fit`] or by loading a file with
/// [`Model::load_cbm`] / [`Model::load_json`].
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    inner: cb_model::Model,
}

impl Model {
    /// Wrap a canonical [`cb_model::Model`] in the facade (used by the Builder
    /// and the loaders).
    #[must_use]
    pub(crate) fn from_canonical(inner: cb_model::Model) -> Self {
        Self { inner }
    }

    /// Borrow the underlying canonical model (escape hatch for advanced callers
    /// that need the internal representation directly).
    #[must_use]
    pub fn as_canonical(&self) -> &cb_model::Model {
        &self.inner
    }

    /// The number of float features the model expects (the
    /// `float_feature_borders` width).
    #[must_use]
    pub fn n_float_features(&self) -> usize {
        self.inner.float_feature_borders.len()
    }

    /// Narrow a pool's SoA float columns to `f32` (the apply storage type) after
    /// checking the float-feature count matches the model (T-04-05-02). A
    /// mismatch returns [`CatBoostError::FeatureMismatch`] rather than reading
    /// out-of-range columns.
    fn feature_columns(&self, pool: &Pool) -> Result<Vec<Vec<f32>>, CatBoostError> {
        let expected = self.n_float_features();
        let actual = pool.n_float_features();
        if actual != expected {
            return Err(CatBoostError::FeatureMismatch(format!(
                "pool has {actual} float features, model expects {expected}"
            )));
        }
        Ok(pool
            .float_features()
            .iter()
            .map(|col| col.iter().map(|&v| v as f32).collect())
            .collect())
    }

    /// Predict with an explicit [`PredictionType`] (the D-06 enum core).
    ///
    /// Returns the flattened (row-major) output: one value per object for
    /// single-column types ([`PredictionType::RawFormulaVal`],
    /// [`PredictionType::Class`], [`PredictionType::Exponent`]); two values per
    /// object (`[class-0, class-1]`) for the two-column types
    /// ([`PredictionType::Probability`], [`PredictionType::LogProbability`]).
    ///
    /// # Errors
    /// [`CatBoostError::FeatureMismatch`] if `pool`'s float-feature count differs
    /// from the model's.
    pub fn predict_with(
        &self,
        pool: &Pool,
        prediction_type: PredictionType,
    ) -> Result<Vec<f64>, CatBoostError> {
        let columns = self.feature_columns(pool)?;
        let raw = predict_raw(&self.inner, &columns);
        Ok(apply_prediction_type(prediction_type, &raw))
    }

    /// Predict the raw model scores ([`PredictionType::RawFormulaVal`]) — the
    /// D-06 shorthand. One value per object.
    ///
    /// # Errors
    /// [`CatBoostError::FeatureMismatch`] (see [`Model::predict_with`]).
    pub fn predict(&self, pool: &Pool) -> Result<Vec<f64>, CatBoostError> {
        self.predict_with(pool, PredictionType::RawFormulaVal)
    }

    /// Reject a model the scalar float-only staged path cannot handle (SP-03
    /// guard): a non-scalar (`approx_dimension > 1`), non-oblivious
    /// (non-symmetric / Region), or CTR/categorical model. On such a model
    /// [`predict_raw_staged`] would silently drop dimensions or ignore trees, so
    /// it is rejected with a typed [`CatBoostError::UnsupportedModel`].
    fn ensure_scalar_oblivious(&self) -> Result<(), CatBoostError> {
        let inner = self.as_canonical();
        if inner.approx_dimension > 1 {
            return Err(CatBoostError::UnsupportedModel(format!(
                "staged_predict supports only scalar (approx_dimension == 1) models, got approx_dimension = {}",
                inner.approx_dimension
            )));
        }
        if !inner.non_symmetric_trees.is_empty() {
            return Err(CatBoostError::UnsupportedModel(
                "staged_predict supports only oblivious models; this model has non-symmetric trees"
                    .to_owned(),
            ));
        }
        if !inner.region_trees.is_empty() {
            return Err(CatBoostError::UnsupportedModel(
                "staged_predict supports only oblivious models; this model has region trees"
                    .to_owned(),
            ));
        }
        if inner.ctr_data.is_some() {
            return Err(CatBoostError::UnsupportedModel(
                "staged_predict supports only float-only models; this model has CTR data"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Cumulative raw predictions ([`PredictionType::RawFormulaVal`]) over an
    /// increasing prefix of the ensemble — one `Vec<f64>` per stage, each length
    /// `n_objects` in object order (scalar oblivious float-only models).
    ///
    /// `ntree_start` / `ntree_end` / `eval_period` default to `0` / `0` / `1`
    /// (all trees, one stage per tree) when `None`, matching upstream
    /// `staged_predict`. Stages step by `eval_period` and always include
    /// `ntree_end` (`0` ⇒ all trees) as the final stage; `stages.last()` equals
    /// [`Model::predict`] for the default schedule.
    ///
    /// # Errors
    /// [`CatBoostError::UnsupportedModel`] if the model is not scalar
    /// (`approx_dimension > 1`), not oblivious (non-symmetric / Region trees), or
    /// carries CTR data — checked BEFORE the pool so the scope error is
    /// deterministic. [`CatBoostError::FeatureMismatch`] if `pool`'s
    /// float-feature count differs from the model's.
    pub fn staged_predict(
        &self,
        pool: &Pool,
        ntree_start: Option<usize>,
        ntree_end: Option<usize>,
        eval_period: Option<usize>,
    ) -> Result<Vec<Vec<f64>>, CatBoostError> {
        self.ensure_scalar_oblivious()?;
        let columns = self.feature_columns(pool)?;
        let start = ntree_start.unwrap_or(0);
        let end = ntree_end.unwrap_or(0);
        let period = eval_period.unwrap_or(1);
        Ok(predict_raw_staged(
            self.as_canonical(),
            &columns,
            start,
            end,
            period,
        ))
    }

    /// Predict class probabilities ([`PredictionType::Probability`]) — the D-06
    /// shorthand. Two values per object (`[class-0, class-1]`, row-major).
    ///
    /// # Errors
    /// [`CatBoostError::FeatureMismatch`] (see [`Model::predict_with`]).
    pub fn predict_proba(&self, pool: &Pool) -> Result<Vec<f64>, CatBoostError> {
        self.predict_with(pool, PredictionType::Probability)
    }

    /// Per-object regular TreeSHAP matrix (`[n_features + 1]` per object,
    /// row-major; the trailing column is the expected value) — D-07.
    ///
    /// # Errors
    /// [`CatBoostError::FeatureMismatch`] if `pool`'s float-feature count differs
    /// from the model's.
    pub fn shap_values(&self, pool: &Pool) -> Result<Vec<Vec<f64>>, CatBoostError> {
        let columns = self.feature_columns(pool)?;
        Ok(shap_values(&self.inner, &columns, self.n_float_features()))
    }

    /// Compute a feature-importance vector / pairing for the requested
    /// [`FeatureImportanceType`] (D-07).
    ///
    /// [`FeatureImportanceType::PredictionValuesChange`] returns a per-feature
    /// percentage vector (summing to 100) as `(feature, _, score)` tuples with
    /// the second index unused (set to the same feature index);
    /// [`FeatureImportanceType::Interaction`] returns the descending-sorted
    /// `(feature_a, feature_b, score)` pairs. Both share one return shape so a
    /// single method covers the structure-only D-07 importance surface.
    ///
    /// [`FeatureImportanceType::LossFunctionChange`] needs a dataset (features +
    /// labels) and is NOT available through this data-free method — it returns an
    /// empty vector here; use [`Model::feature_importance_with_data`] instead.
    #[must_use]
    pub fn feature_importance(
        &self,
        importance_type: FeatureImportanceType,
    ) -> Vec<(usize, usize, f64)> {
        match importance_type {
            FeatureImportanceType::PredictionValuesChange => prediction_values_change(&self.inner)
                .into_iter()
                .enumerate()
                .map(|(feature, score)| (feature, feature, score))
                .collect(),
            FeatureImportanceType::Interaction => interaction(&self.inner),
            // LossFunctionChange requires labels; not serviceable without a Pool.
            FeatureImportanceType::LossFunctionChange => Vec::new(),
        }
    }

    /// Compute a feature-importance vector for the requested
    /// [`FeatureImportanceType`] using a dataset (MODEL-03 / D-12; D-6.6-09).
    ///
    /// This is the data-bearing companion to [`Model::feature_importance`]. It is
    /// the ONLY way to obtain [`FeatureImportanceType::LossFunctionChange`], which
    /// re-evaluates the objective metric with each feature's per-document SHAP
    /// contribution removed; the result is `(feature, feature, score)` tuples in
    /// feature-index order.
    ///
    /// [`FeatureImportanceType::PredictionValuesChange`] recomputes each tree's
    /// per-leaf weights from `pool`'s columns
    /// ([`cb_model::prediction_values_change_with_data`], upstream's
    /// `data=pool` mode) instead of using the model's stored training-time
    /// `leaf_weights` — for online-CTR models the two genuinely differ (documents
    /// land in different leaves under training-time online CTR values than under
    /// the final baked tables), so this is required for oracle parity against a
    /// `data=pool` fixture on a CTR model.
    ///
    /// [`FeatureImportanceType::Interaction`] has no dataset-aware mode in this
    /// crate yet ([`cb_model::interaction`] is structure-only — it reads each
    /// tree's baked `leaf_values`, never `leaf_weights`, so there is currently no
    /// dataset-recomputed variant to call); it ignores `pool` and delegates to
    /// [`Model::feature_importance`].
    ///
    /// For [`FeatureImportanceType::LossFunctionChange`], `loss` names the
    /// model's trained objective (case-insensitive); the model file carries no
    /// loss metadata (Q1), so the caller supplies it. Only the Min-optimized
    /// numeric losses `RMSE`, `MAE`, `MAPE`, `Quantile[:alpha=..]`, and `Logloss`
    /// are supported in this slice — any other (Max-optimized, ranking, or
    /// unknown) loss yields [`CatBoostError::UnsupportedLoss`] rather than a
    /// silent wrong-metric fallback (FL-03). `loss` is ignored by the
    /// structure-only variants.
    ///
    /// # Errors
    /// [`CatBoostError::FeatureMismatch`] if the pool's feature columns cannot be
    /// projected; [`CatBoostError::UnsupportedLoss`] for an out-of-scope
    /// `LossFunctionChange` loss name.
    pub fn feature_importance_with_data(
        &self,
        importance_type: FeatureImportanceType,
        pool: &Pool,
        loss: &str,
    ) -> Result<Vec<(usize, usize, f64)>, CatBoostError> {
        match importance_type {
            FeatureImportanceType::PredictionValuesChange => {
                let columns = self.feature_columns(pool)?;
                let cat_columns = pool.cat_features().to_vec();
                Ok(
                    prediction_values_change_with_data(&self.inner, &columns, &cat_columns)
                        .into_iter()
                        .enumerate()
                        .map(|(feature, score)| (feature, feature, score))
                        .collect(),
                )
            }
            FeatureImportanceType::Interaction => Ok(self.feature_importance(importance_type)),
            FeatureImportanceType::LossFunctionChange => {
                // Map the trained loss → its Min-optimized `EvalMetric`; reject
                // anything outside the supported numeric set (no silent Logloss).
                let metric = eval_metric_for_loss(loss)?;
                let columns = self.feature_columns(pool)?;
                // Inject the metric's `GetFinalError` as a non-panicking closure.
                // Weights: unit only — this first slice ignores per-object `Pool`
                // weights (SPEC R3), matching the retained Logloss path; a weighted
                // pool would diverge from upstream's weighted `GetFinalError`, so
                // weighted `LossFunctionChange` stays out of scope until a weighted
                // oracle exists. A degenerate `eval` (length/empty/weight guard)
                // yields `NaN` rather than panicking across the public boundary.
                let scores = cb_model::loss_function_change(
                    &self.inner,
                    &columns,
                    pool.label(),
                    self.n_float_features(),
                    |approx, target| metric.eval(approx, target, &[]).unwrap_or(f64::NAN),
                );
                Ok(scores
                    .into_iter()
                    .enumerate()
                    .map(|(feature, score)| (feature, feature, score))
                    .collect())
            }
        }
    }

    /// Compute partial dependence for one or two float features over `pool`
    /// (FSTR-03). Each target feature is swept across its per-bin grid while the
    /// other features keep their per-object `pool` values; the `RawFormulaVal` is
    /// averaged over all objects, matching upstream `plot_partial_dependence`
    /// within `1e-5`. Returns the per-feature grids and the averaged surface
    /// (row-major, first feature outer, for two features) — see
    /// [`cb_model::PartialDependence`].
    ///
    /// `features` indexes the model's float-feature space (`0..n_float_features`);
    /// pass 1 or 2 distinct indices.
    ///
    /// # Errors
    /// [`CatBoostError::FeatureMismatch`] if `pool`'s float-feature count differs
    /// from the model's; [`CatBoostError::PartialDependence`] for an invalid
    /// request (bad arity, out-of-range / duplicate feature, empty dataset).
    pub fn partial_dependence(
        &self,
        pool: &Pool,
        features: &[usize],
    ) -> Result<PartialDependence, CatBoostError> {
        let columns = self.feature_columns(pool)?;
        Ok(partial_dependence(&self.inner, &columns, features)?)
    }

    /// Save the model to a native `.cbm` file (D-07).
    ///
    /// # Errors
    /// [`CatBoostError::Model`] / [`CatBoostError::Io`] on a serialization or I/O
    /// failure.
    pub fn save_cbm(&self, path: &Path) -> Result<(), CatBoostError> {
        save_cbm(&self.inner, path)?;
        Ok(())
    }

    /// Load a model from a native `.cbm` file (D-07).
    ///
    /// # Errors
    /// [`CatBoostError::Model`] on malformed input (bad magic, corrupt
    /// FlatBuffers, wrong schema) — never panics (T-04-05-01, Security V5).
    pub fn load_cbm(path: &Path) -> Result<Self, CatBoostError> {
        let inner = load_cbm(path)?;
        Ok(Self::from_canonical(inner))
    }

    /// Save the model to a `model.json` file on the upstream schema (D-07).
    ///
    /// # Errors
    /// [`CatBoostError::Model`] / [`CatBoostError::Io`] on a serialization or I/O
    /// failure.
    pub fn save_json(&self, path: &Path) -> Result<(), CatBoostError> {
        save_json(&self.inner, path)?;
        Ok(())
    }

    /// Load a model from a `model.json` file on the upstream schema (D-07).
    ///
    /// # Errors
    /// [`CatBoostError::Model`] on malformed input (bad JSON, wrong shape) —
    /// never panics (T-04-05-01, Security V5).
    pub fn load_json(path: &Path) -> Result<Self, CatBoostError> {
        let inner = load_json(path)?;
        Ok(Self::from_canonical(inner))
    }

    /// Export to ONNX (EXPORT-01): a float-only, oblivious, identity-scale
    /// model only — categorical/CTR and non-oblivious (Lossguide/Depthwise/
    /// Region) models are rejected with a typed error, never a panic.
    /// `is_classifier` selects `TreeEnsembleClassifier`+`ZipMap`
    /// (`post_transform="LOGISTIC"`/`"SOFTMAX"`) vs `TreeEnsembleRegressor`
    /// (`post_transform="NONE"`) — see [`cb_model::export_onnx`]. The caller
    /// supplies this because the model carries no loss-function/objective
    /// metadata to infer it from.
    ///
    /// # Errors
    /// [`CatBoostError::Export`] on an unsupported model (categorical/CTR,
    /// non-oblivious) or a downstream encode/I/O failure.
    pub fn save_onnx(&self, path: &Path, is_classifier: bool) -> Result<(), CatBoostError> {
        export_onnx(&self.inner, path, is_classifier)?;
        Ok(())
    }

    /// Combine several models into one weighted-sum model (D-07 analogue).
    /// `weights[i]` scales model `i`'s leaf contributions; `None` defaults
    /// every model to weight `1.0`.
    ///
    /// # Errors
    /// [`CatBoostError::Model`] on an unmergeable set — see
    /// [`cb_model::sum_models`].
    pub fn sum_models(models: &[&Model], weights: Option<&[f64]>) -> Result<Model, CatBoostError> {
        let canonical: Vec<&cb_model::Model> = models.iter().map(|m| &m.inner).collect();
        let merged = sum_models(&canonical, weights.unwrap_or(&[]))?;
        Ok(Self::from_canonical(merged))
    }
}
