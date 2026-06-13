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
    apply_prediction_type, interaction, load_cbm, load_json, predict_raw, prediction_values_change,
    save_cbm, save_json, shap_values, FeatureImportanceType, PredictionType,
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
    /// single method covers the D-07 importance surface.
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
        }
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
}
