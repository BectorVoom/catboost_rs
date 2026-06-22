//! `CatBoostRanker` — the ranking arm of the native estimator trio (08-04,
//! PYAPI-03).
//!
//! Mirrors [`crate::regressor::CatBoostRegressor`]'s store-verbatim / validate /
//! ingest / detach / fit structure (shared base in [`crate::estimator`]), with one
//! ranking-specific contract: `fit` REQUIRES a `group_id`-bearing dataset. Because
//! a bare framework object (NumPy / Pandas / Arrow / Polars) carries no grouping
//! metadata, the user passes a native [`crate::pool::Pool`] constructed with
//! `group_id=...`. `fit` validates `group_id` presence on the materialized facade
//! `Pool` (else an actionable `CatBoostValueError`, threat T-08-14); `predict`
//! returns raw ranking scores via [`Model::predict`].
//!
//! The sklearn presentation for the ranker (tags / `score`) is Deferred to 08-05
//! (RESEARCH 483-487); this slice exposes the native `fit` / `predict` only.

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{not_fitted_err, CatBoostValueError, PyCbError};
use crate::estimator::{build_sklearn_tags, data_to_pool, fit_pool, EstimatorBase};
use crate::params::{make_builder, validate_params};

/// CatBoost-mirror ranker. Reuses the shared estimator base, param registry, and
/// ingestion; requires a `group_id`-bearing `Pool` at `fit` and returns raw
/// ranking scores at `predict`.
#[pyclass(name = "CatBoostRanker", subclass, dict)]
pub struct CatBoostRanker {
    base: EstimatorBase,
}

#[pymethods]
impl CatBoostRanker {
    /// Store every kwarg verbatim (D-06: NO work in `__init__`). Validation and
    /// the `group_id`-presence check fire at `fit()` time.
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            base: EstimatorBase::from_kwargs(kwargs)?,
        })
    }

    /// Fit on a `group_id`-bearing dataset. `x` is normally a native
    /// [`crate::pool::Pool`] built with `group_id=...` (a bare framework object
    /// carries no grouping metadata and so is rejected by the `group_id` check).
    ///
    /// Validates kwargs (D-07 registry), ingests + OWNS the input under the GIL
    /// (D-11), asserts `group_id` is present (else an actionable
    /// `CatBoostValueError`, threat T-08-14), then releases the GIL (`py.detach`)
    /// for training.
    ///
    /// # Errors
    /// `CatBoostParameterError` on an unknown / unsupported kwarg;
    /// `CatBoostValueError` when `group_id` is absent or on a dtype / layout
    /// mismatch (D-12); the typed taxonomy on a training failure.
    #[pyo3(signature = (x, y = None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        validate_params(&slf.base.params)?;
        let pool = data_to_pool(py, x, y)?;
        // Ranking REQUIRES grouping. Reject a group-less dataset with an
        // actionable typed error rather than silently training a non-ranking model
        // or indexing unchecked group structure (threat T-08-14).
        if pool.group_id().is_empty() {
            return Err(CatBoostValueError::new_err(
                "CatBoostRanker.fit requires a grouped dataset: pass a `Pool` constructed with \
                 `group_id=...` (a bare NumPy / Pandas / Arrow / Polars array carries no grouping \
                 metadata)",
            ));
        }
        let builder = make_builder(&slf.base.params, py)?;
        let model = py.detach(|| fit_pool(builder, &pool)).map_err(PyCbError)?;
        slf.base.model = Some(model);
        Ok(slf.into())
    }

    /// Predict raw ranking SCORES for a C-contiguous float32 NumPy `X` `(n, k)`
    /// (or a native `Pool`). Returns a NumPy `float64` array of length `n` — the
    /// per-object score by which a group's objects are ranked.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on a dtype / layout /
    /// feature mismatch; the typed taxonomy on a prediction failure.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostRanker is not fitted yet; call `fit` before `predict`",
            )
        })?;
        let pool = data_to_pool(py, x, None)?;
        let preds = py.detach(|| model.predict(&pool)).map_err(PyCbError)?;
        Ok(preds.to_pyarray(py))
    }

    /// Return the verbatim constructor kwargs (sklearn `get_params`). Enables
    /// `sklearn.base.clone` / `GridSearchCV` even though the ranker is excluded
    /// from the structural `check_estimator` gate (Task 2, RESEARCH Open Q2).
    ///
    /// # Errors
    /// Propagates any failure building the params dict.
    #[pyo3(signature = (deep = None))]
    fn get_params<'py>(
        &self,
        py: Python<'py>,
        deep: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.base.get_params(py, deep)
    }

    /// Merge `**params` into the verbatim store and return `self`. Validation stays
    /// at `fit` (D-06).
    ///
    /// # Errors
    /// Propagates any failure extracting a param key.
    #[pyo3(signature = (**params))]
    fn set_params(
        mut slf: PyRefMut<'_, Self>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Self>> {
        slf.base.set_params(params)?;
        Ok(slf.into())
    }

    /// The sklearn ≥1.6 `Tags` dataclass. sklearn has no native ranker estimator
    /// type, so the ranker presents with the regressor-like `"regressor"` tag set
    /// (continuous per-object score output) per RESEARCH Open Q2. It is EXCLUDED
    /// from the structural `check_estimator` gate (no native sklearn ranker
    /// contract to satisfy), documented in `test_check_estimator.py`.
    ///
    /// # Errors
    /// Propagates any failure constructing the `Tags` dataclass.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        build_sklearn_tags(py, "regressor")
    }

    /// sklearn estimator-type marker. The ranker presents regressor-like
    /// (continuous score) per RESEARCH Open Q2.
    #[classattr]
    fn _estimator_type() -> &'static str {
        "regressor"
    }

    /// `True` once `fit` has populated the model.
    #[getter]
    fn is_fitted(&self) -> bool {
        self.base.is_fitted()
    }

    /// sklearn's fitted-state hook (the fitted model is an opaque Rust field).
    fn __sklearn_is_fitted__(&self) -> bool {
        self.base.is_fitted()
    }
}
