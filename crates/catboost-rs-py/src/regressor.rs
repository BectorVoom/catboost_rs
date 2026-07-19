//! `CatBoostRegressor` — the 08-01 walking-skeleton estimator.
//!
//! The thinnest end-to-end vertical slice: store kwargs verbatim in `__init__`
//! (D-06), ingest a float32 contiguous NumPy `X` (+ `y`) into an owned `Pool`
//! under the GIL (D-11), release the GIL for the fit (`py.detach`), and return
//! raw predictions as a NumPy array. The sklearn contract (`get_params` /
//! `set_params` / `__sklearn_tags__` / clone / `NotFittedError`), the classifier
//! / ranker, param validation, and the typed error taxonomy land in later plans.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{not_fitted_err, CatBoostValueError, PyCbError};
use crate::estimator::{
    build_sklearn_tags, data_to_pool, fit_pool, load_model_path, r2_score, EstimatorBase,
};
use crate::params::{make_builder, validate_params};

/// CatBoost-mirror regressor (sklearn-compatible). Plan 08-01 implements the
/// thinnest `__init__` / `fit` / `predict` path; the full sklearn contract and
/// param surface land in later plans.
#[pyclass(name = "CatBoostRegressor", subclass, dict)]
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

    /// Export the fitted model to ONNX (EXPORT-01f) as a `TreeEnsembleRegressor`
    /// (`post_transform="NONE"`). Categorical/CTR and non-oblivious models are
    /// rejected with a typed `CatBoostValueError`, never a panic.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on an unsupported
    /// model; `IOError` on a downstream file-write failure.
    fn save_onnx(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostRegressor is not fitted yet; call `fit` before `save_onnx`",
            )
        })?;
        py.detach(|| model.save_onnx(std::path::Path::new(path), false))
            .map_err(PyCbError)?;
        Ok(())
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

    /// Partial dependence for one or two float features (FSTR-03), mirroring
    /// upstream `plot_partial_dependence`. Returns a dict `{features: list[int],
    /// grids: list[np.ndarray], values: np.ndarray}`; `values` is row-major over
    /// the Cartesian product of `grids` (first feature outer). `features` indexes
    /// the model's float features (`0..n_float_features`); pass 1 or 2 distinct
    /// indices.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on a bad `X` (dtype /
    /// layout) or an invalid feature request (arity / out-of-range / duplicate /
    /// empty dataset).
    #[pyo3(signature = (x, features))]
    fn partial_dependence<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        features: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostRegressor is not fitted yet; call `fit` before `partial_dependence`",
            )
        })?;
        crate::estimator::partial_dependence_py(model, py, x, features)
    }

    /// Return the verbatim constructor kwargs (sklearn `get_params`). `deep` is
    /// accepted for signature parity (no nested estimators). Enables
    /// `sklearn.base.clone` and `GridSearchCV` (T-08-15).
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

    /// Merge `**params` into the verbatim store and return `self` (sklearn
    /// `set_params` chaining). No validation here — that stays at `fit` (D-06).
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

    /// The sklearn ≥1.6 `Tags` dataclass marking this as a `"regressor"`
    /// (RESEARCH Pitfall 5). Read by modern `check_estimator`.
    ///
    /// # Errors
    /// Propagates any failure constructing the `Tags` dataclass.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        build_sklearn_tags(py, "regressor")
    }

    /// sklearn estimator-type marker (`"regressor"`) read by some older sklearn
    /// dispatch paths alongside `__sklearn_tags__`.
    #[classattr]
    fn _estimator_type() -> &'static str {
        "regressor"
    }

    /// `True` once `fit`/`load_model` has populated the model. Exposed so sklearn's
    /// `check_is_fitted` (and users) can introspect the fitted state.
    #[getter]
    fn is_fitted(&self) -> bool {
        self.base.is_fitted()
    }

    /// sklearn's fitted-state hook. Because the fitted model lives in an opaque
    /// Rust field (not a trailing-underscore Python attribute), `check_is_fitted`
    /// cannot detect it by attribute scan; this method is the documented escape
    /// hatch sklearn uses instead.
    fn __sklearn_is_fitted__(&self) -> bool {
        self.base.is_fitted()
    }

    /// R² score of `predict(X)` vs `y` (the sklearn `RegressorMixin.score`
    /// default). `y` is a C-contiguous float32 1-D NumPy array.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on a bad `y` dtype/layout
    /// or a length mismatch; the typed taxonomy on a prediction failure.
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<f64> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostRegressor is not fitted yet; call `fit` before `score`",
            )
        })?;
        let pool = data_to_pool(py, x, None)?;
        let preds = py.detach(|| model.predict(&pool)).map_err(PyCbError)?;
        let y_true = y_to_vec(y)?;
        if y_true.len() != preds.len() {
            return Err(CatBoostValueError::new_err(format!(
                "y length ({}) does not match X row count ({})",
                y_true.len(),
                preds.len()
            )));
        }
        Ok(r2_score(&y_true, &preds))
    }
}

/// Read a C-contiguous float32 1-D NumPy array into an owned `Vec<f64>` (the
/// `score` true-label contract; mirrors the strict D-12 ingest rule).
///
/// # Errors
/// [`CatBoostValueError`] if `y` is not a C-contiguous 1-D float32 array.
pub(crate) fn y_to_vec(y: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let arr: PyReadonlyArray1<f32> = y.extract().map_err(|_| {
        CatBoostValueError::new_err("y must be a 1-D float32 NumPy array; pass `y.astype(np.float32)`")
    })?;
    Ok(arr.as_array().iter().map(|&v| f64::from(v)).collect())
}
