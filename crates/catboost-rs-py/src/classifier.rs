//! `CatBoostClassifier` — the classification arm of the native estimator trio
//! (08-04, PYAPI-03).
//!
//! Mirrors [`crate::regressor::CatBoostRegressor`]'s store-verbatim / validate /
//! ingest / detach / fit structure (the shared base lives in
//! [`crate::estimator`]), differing in two ways:
//!
//! 1. **Default loss.** When the user does NOT pass `loss_function` / `objective`,
//!    the classifier defaults to a CLASSIFICATION loss (`Logloss`) — the loss
//!    SELECTS the task (D-05). A regressor would default to `RMSE`.
//! 2. **Prediction surface.** `predict` returns CLASS LABELS (`(n,)`, via
//!    [`PredictionType::Class`]); `predict_proba` returns CLASS PROBABILITIES
//!    shaped `(n, 2)` (`[class-0, class-1]` per object, via
//!    [`PredictionType::Probability`]) — the upstream binary convention.

use catboost_rs::{Loss, PredictionType};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{not_fitted_err, CatBoostValueError, PyCbError};
use crate::estimator::{
    accuracy_score, build_sklearn_tags, data_to_pool, fit_pool, load_model_path, EstimatorBase,
};
use crate::params::{make_builder, validate_params};
use crate::regressor::y_to_vec;

/// CatBoost-mirror classifier (sklearn-compatible). Reuses the shared estimator
/// base, param registry, and ingestion; defaults to `Logloss` and exposes
/// `predict` (class labels) + `predict_proba` (`(n, 2)` probabilities).
#[pyclass(name = "CatBoostClassifier", subclass, dict)]
pub struct CatBoostClassifier {
    base: EstimatorBase,
}

#[pymethods]
impl CatBoostClassifier {
    /// Store every kwarg verbatim (D-06: NO work in `__init__`). Validation and
    /// the classification-default loss fire at `fit()` time.
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            base: EstimatorBase::from_kwargs(kwargs)?,
        })
    }

    /// Fit on a C-contiguous float32 NumPy `X` `(n, k)` (or a native `Pool`) and
    /// a binary `y` `(n,)`.
    ///
    /// Validates kwargs (D-07 registry), ingests + OWNS the input under the GIL
    /// (D-11), then releases the GIL (`py.detach`) for training. When the user did
    /// not set `loss_function` / `objective`, the builder's loss is overridden to
    /// `Logloss` (a classification loss) so the model is a classifier (D-05).
    ///
    /// # Errors
    /// `CatBoostParameterError` on an unknown / unsupported kwarg;
    /// `CatBoostValueError` on a dtype / layout / shape mismatch (D-12); the typed
    /// taxonomy on a training failure (08-02 / PYAPI-05).
    #[pyo3(signature = (x, y = None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        validate_params(&slf.base.params)?;
        let pool = data_to_pool(py, x, y)?;
        let mut builder = make_builder(&slf.base.params, py)?;
        // The classifier defaults to a CLASSIFICATION loss (D-05). Only override
        // when the user supplied neither the canonical name nor its alias, so an
        // explicit `loss_function="CrossEntropy"` (etc.) is honored.
        if !slf.base.params.contains_key("loss_function") && !slf.base.params.contains_key("objective")
        {
            builder = builder.loss(Loss::Logloss);
        }
        let model = py.detach(|| fit_pool(builder, &pool)).map_err(PyCbError)?;
        slf.base.model = Some(model);
        Ok(slf.into())
    }

    /// Predict CLASS LABELS for a C-contiguous float32 NumPy `X` `(n, k)` (or a
    /// native `Pool`). Returns a NumPy `float64` array of length `n` carrying the
    /// predicted class (`0.0` / `1.0`) via [`PredictionType::Class`].
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
                "this CatBoostClassifier is not fitted yet; call `fit` before `predict`",
            )
        })?;
        let pool = data_to_pool(py, x, None)?;
        let preds = py
            .detach(|| model.predict_with(&pool, PredictionType::Class))
            .map_err(PyCbError)?;
        Ok(preds.to_pyarray(py))
    }

    /// Predict CLASS PROBABILITIES for a C-contiguous float32 NumPy `X` `(n, k)`
    /// (or a native `Pool`). Returns a NumPy `float64` array shaped `(n, 2)` with
    /// `[P(class 0), P(class 1)]` per row (the upstream binary convention) via
    /// [`PredictionType::Probability`].
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on a dtype / layout /
    /// feature mismatch; the typed taxonomy on a prediction failure.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostClassifier is not fitted yet; call `fit` before `predict_proba`",
            )
        })?;
        let pool = data_to_pool(py, x, None)?;
        // The facade returns the two-column probability output flattened row-major
        // (`[class-0, class-1]` per object). Reshape to `(n, 2)` for the upstream
        // binary convention.
        let flat = py
            .detach(|| model.predict_with(&pool, PredictionType::Probability))
            .map_err(PyCbError)?;
        // The (n, 2) contract requires an even flat length. Assert it rather than
        // silently truncating a trailing element via `chunks_exact` (WR-01): a
        // single-column or otherwise odd output is a model/contract violation, not
        // something to drop the last object's probabilities over.
        if flat.len() % 2 != 0 {
            return Err(CatBoostValueError::new_err(format!(
                "probability output length {} is not divisible by 2 (expected an (n, 2) \
                 row-major buffer of [P(class 0), P(class 1)] pairs)",
                flat.len()
            )));
        }
        // Empty input: `PyArray2::from_vec2` on an empty `rows` yields shape (0, 0),
        // violating the (n, 2) column-count contract downstream consumers rely on
        // (np.concatenate / vstack). Construct an explicit (0, 2) array (WR-02).
        if flat.is_empty() {
            return Ok(PyArray2::zeros(py, [0, 2], false));
        }
        let rows: Vec<Vec<f64>> = flat.chunks_exact(2).map(<[f64]>::to_vec).collect();
        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    /// Load a reference model from a `.cbm` (or `.json`) file into a fitted
    /// `CatBoostClassifier` WITHOUT training (mirrors upstream `load_model`). The
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

    /// Export the fitted model to ONNX (EXPORT-01f) as a
    /// `TreeEnsembleClassifier`+`ZipMap` pair (`post_transform="LOGISTIC"` for
    /// binary, `"SOFTMAX"` for multiclass). Categorical/CTR and non-oblivious
    /// models are rejected with a typed `CatBoostValueError`, never a panic.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on an unsupported
    /// model; `IOError` on a downstream file-write failure.
    fn save_onnx(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostClassifier is not fitted yet; call `fit` before `save_onnx`",
            )
        })?;
        py.detach(|| model.save_onnx(std::path::Path::new(path), true))
            .map_err(PyCbError)?;
        Ok(())
    }

    /// Partial dependence for one or two float features (FSTR-03), mirroring
    /// upstream `plot_partial_dependence`. Returns a dict `{features: list[int],
    /// grids: list[np.ndarray], values: np.ndarray}` (values row-major, first
    /// feature outer). `features` indexes the model's float features.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on a bad `X` or an
    /// invalid feature request (arity / out-of-range / duplicate / empty dataset).
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
                "this CatBoostClassifier is not fitted yet; call `fit` before `partial_dependence`",
            )
        })?;
        crate::estimator::partial_dependence_py(model, py, x, features)
    }

    /// Return the verbatim constructor kwargs (sklearn `get_params`); enables
    /// `sklearn.base.clone` / `GridSearchCV` (T-08-15).
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
    /// `set_params` chaining). Validation stays at `fit` (D-06).
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

    /// The sklearn ≥1.6 `Tags` dataclass marking this as a `"classifier"`.
    ///
    /// # Errors
    /// Propagates any failure constructing the `Tags` dataclass.
    fn __sklearn_tags__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        build_sklearn_tags(py, "classifier")
    }

    /// sklearn estimator-type marker (`"classifier"`).
    #[classattr]
    fn _estimator_type() -> &'static str {
        "classifier"
    }

    /// `True` once `fit`/`load_model` has populated the model.
    #[getter]
    fn is_fitted(&self) -> bool {
        self.base.is_fitted()
    }

    /// sklearn's fitted-state hook (the fitted model is an opaque Rust field, not a
    /// trailing-underscore attribute `check_is_fitted` can scan).
    fn __sklearn_is_fitted__(&self) -> bool {
        self.base.is_fitted()
    }

    /// Mean accuracy of `predict(X)` vs `y` (the sklearn `ClassifierMixin.score`
    /// default). `y` is a C-contiguous float32 1-D NumPy array.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` on a bad `y` dtype/layout
    /// or a length mismatch; the typed taxonomy on a prediction failure.
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<f64> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostClassifier is not fitted yet; call `fit` before `score`",
            )
        })?;
        let pool = data_to_pool(py, x, None)?;
        let preds = py
            .detach(|| model.predict_with(&pool, PredictionType::Class))
            .map_err(PyCbError)?;
        let y_true = y_to_vec(y)?;
        if y_true.len() != preds.len() {
            return Err(CatBoostValueError::new_err(format!(
                "y length ({}) does not match X row count ({})",
                y_true.len(),
                preds.len()
            )));
        }
        Ok(accuracy_score(&y_true, &preds))
    }
}
