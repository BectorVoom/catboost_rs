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
    predict_raw, prediction_values_change, save_cbm, save_json, shap_values,
    FeatureImportanceType, PartialDependence, PredictionType,
};

use crate::error::CatBoostError;

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
    /// feature-index order. The structure-only variants
    /// ([`FeatureImportanceType::PredictionValuesChange`] /
    /// [`FeatureImportanceType::Interaction`]) ignore the dataset and delegate to
    /// [`Model::feature_importance`].
    ///
    /// # Errors
    /// [`CatBoostError`] if the pool's feature columns cannot be projected.
    pub fn feature_importance_with_data(
        &self,
        importance_type: FeatureImportanceType,
        pool: &Pool,
    ) -> Result<Vec<(usize, usize, f64)>, CatBoostError> {
        match importance_type {
            FeatureImportanceType::PredictionValuesChange
            | FeatureImportanceType::Interaction => Ok(self.feature_importance(importance_type)),
            FeatureImportanceType::LossFunctionChange => {
                let columns = self.feature_columns(pool)?;
                let labels = pool.label().to_vec();
                let scores = cb_model::loss_function_change(
                    &self.inner,
                    &columns,
                    &labels,
                    self.n_float_features(),
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
}
