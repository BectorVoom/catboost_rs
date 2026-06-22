//! `CatBoostRegressor` — the 08-01 walking-skeleton estimator.
//!
//! The thinnest end-to-end vertical slice: store kwargs verbatim in `__init__`
//! (D-06), ingest a float32 contiguous NumPy `X` (+ `y`) into an owned `Pool`
//! under the GIL (D-11), release the GIL for the fit (`py.detach`), and return
//! raw predictions as a NumPy array. The sklearn contract (`get_params` /
//! `set_params` / `__sklearn_tags__` / clone / `NotFittedError`), the classifier
//! / ranker, param validation, and the typed error taxonomy land in later plans.

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{not_fitted_err, PyCbError};
use crate::estimator::{data_to_pool, fit_pool, load_model_path, EstimatorBase};
use crate::params::{make_builder, validate_params};

/// CatBoost-mirror regressor (sklearn-compatible). Plan 08-01 implements the
/// thinnest `__init__` / `fit` / `predict` path; the full sklearn contract and
/// param surface land in later plans.
#[pyclass(name = "CatBoostRegressor", subclass)]
pub struct CatBoostRegressor {
    base: EstimatorBase,
}

#[pymethods]
impl CatBoostRegressor {
    /// Store every kwarg verbatim (D-06: NO work in `__init__`). Validation fires
    /// at `fit()` time in later plans.
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            base: EstimatorBase::from_kwargs(kwargs)?,
        })
    }

    /// Fit on a C-contiguous float32 NumPy `X` `(n, k)` and `y` `(n,)`.
    ///
    /// Ingests + OWNS the input under the GIL (D-11), then releases the GIL
    /// (`py.detach`) for the training compute. Returns `self`-less `None`
    /// (sklearn `fit` returns the estimator; the Python wrapper returns `self` —
    /// here the in-place mutation suffices for the smoke and `m.fit(...)` chains
    /// because pyo3 returns the bound receiver implicitly via the caller).
    ///
    /// # Errors
    /// [`PyValueError`] on a dtype / layout / shape mismatch (D-12) or a training
    /// failure (typed taxonomy: 08-02).
    #[pyo3(signature = (x, y = None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        // Validate kwargs against the D-07 registry BEFORE ingest (D-06): reject
        // known-not-yet (parity gap) and unknown (typo) params with a typed
        // CatBoostParameterError, so no unsupported param is silently ignored
        // (threat T-08-05).
        validate_params(&slf.base.params)?;
        // --- GIL HELD: own the input before any detach (D-11) ---
        // data_to_pool accepts EITHER a framework object (NumPy / Pandas / Arrow /
        // Polars) OR a native Pool; in every case it COPIES into owned columns
        // before returning, so the py.detach below never sees a live Python-buffer
        // borrow (PYAPI-06).
        let pool = data_to_pool(py, x, y)?;
        let builder = make_builder(&slf.base.params, py)?;
        // --- owned/quantized data only: safe to release the GIL ---
        let model = py.detach(|| fit_pool(builder, &pool)).map_err(PyCbError)?;
        slf.base.model = Some(model);
        Ok(slf.into())
    }

    /// Load a reference model from a `.cbm` (or `.json`) file into a fitted
    /// `CatBoostRegressor` WITHOUT training (mirrors upstream `load_model`). The
    /// returned estimator's `model` is `Some(loaded)`; this is the single
    /// deterministic oracle path (RESEARCH Open Q3, Path (a)).
    ///
    /// # Errors
    /// `CatBoostValueError` on a malformed / unreadable model file (T-08-12).
    #[staticmethod]
    fn load_model(path: &str) -> PyResult<Self> {
        let model = load_model_path(path)?;
        Ok(Self {
            base: EstimatorBase::from_model(model),
        })
    }

    /// Predict raw model scores for a C-contiguous float32 NumPy `X` `(n, k)`.
    /// Returns a NumPy `float64` array of length `n`.
    ///
    /// # Errors
    /// [`PyValueError`] if the estimator is not fitted (placeholder for the typed
    /// `NotFittedError`, 08-05), on a dtype / layout mismatch (D-12), or a
    /// prediction failure.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostRegressor is not fitted yet; call `fit` before `predict`",
            )
        })?;
        // --- GIL HELD: own the input before any detach (D-11) ---
        // Accept a framework object OR a native Pool, same as fit.
        let pool = data_to_pool(py, x, None)?;
        let preds = py.detach(|| model.predict(&pool)).map_err(PyCbError)?;
        Ok(preds.to_pyarray(py))
    }
}
