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
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{not_fitted_err, CatBoostValueError, PyCbError};
use crate::estimator::{
    accuracy_score, build_sklearn_tags, data_to_pool, eval_set_to_pools, fit_pool_with_eval,
    load_model_path, resolve_cat_features, scoring_data_to_pool, EstimatorBase,
};
use crate::params::{
    make_builder, validate_eval_set_only_params, validate_params, EVAL_SET_REMEDY_FIT,
};
use crate::regressor::y_to_vec;

/// Read the `class_names` kwarg, if present, as an ordered label list.
///
/// Upstream accepts it on the CLASSIFIER only (`CatBoostRegressor.__init__()` raises
/// `unexpected keyword argument 'class_names'`), and the ORDER is meaningful: class
/// index `i` is position `i`, so passing the labels non-sorted flips which label is
/// the positive class and reorders `predict_proba`'s columns.
fn read_class_names(
    params: &std::collections::BTreeMap<String, Py<PyAny>>,
    py: Python<'_>,
) -> PyResult<Vec<Py<PyAny>>> {
    let Some(obj) = params.get("class_names") else {
        return Ok(Vec::new());
    };
    let bound = obj.bind(py);
    if bound.is_none() {
        return Ok(Vec::new());
    }
    let items: Vec<Py<PyAny>> = bound
        .try_iter()
        .map_err(|_| {
            CatBoostValueError::new_err(
                "class_names must be a sequence of class labels (e.g. [\"neg\", \"pos\"])",
            )
        })?
        .map(|it| it.map(pyo3::Bound::unbind))
        .collect::<PyResult<_>>()?;
    if items.len() < 2 {
        return Err(CatBoostValueError::new_err(format!(
            "class_names must list at least 2 classes; got {}",
            items.len()
        )));
    }
    // This surface's classifier is BINARY throughout (`predict_proba` returns the
    // upstream two-column convention), so a longer list would silently train a
    // binary model and label it as multiclass.
    if items.len() > 2 {
        return Err(CatBoostValueError::new_err(format!(
            "class_names lists {} classes, but this classifier is binary \
             (predict_proba returns 2 columns); multiclass class_names is not \
             implemented",
            items.len()
        )));
    }
    // Duplicates would make the label -> index map ambiguous.
    for i in 0..items.len() {
        for j in (i + 1)..items.len() {
            if items[i].bind(py).eq(items[j].bind(py)).unwrap_or(false) {
                return Err(CatBoostValueError::new_err(
                    "class_names must not contain duplicate labels",
                ));
            }
        }
    }
    Ok(items)
}

/// Map a target of arbitrary labels onto class INDICES (`0.0` / `1.0`) so the
/// existing float32 ingestion path can carry it.
///
/// A label absent from `class_names` is REJECTED with upstream's wording
/// (`Unknown class label: "..."`) rather than silently dropped or coerced — a
/// mislabelled row would otherwise train as the wrong class.
fn encode_labels<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    class_names: &[Py<PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    let mut encoded: Vec<f32> = Vec::new();
    for item in y.try_iter().map_err(|_| {
        CatBoostValueError::new_err("y must be a sequence of class labels when class_names is set")
    })? {
        let item = item?;
        let mut found = None;
        for (idx, name) in class_names.iter().enumerate() {
            if item.eq(name.bind(py)).unwrap_or(false) {
                found = Some(idx);
                break;
            }
        }
        match found {
            Some(idx) => encoded.push(idx as f32),
            None => {
                return Err(CatBoostValueError::new_err(format!(
                    "Unknown class label: \"{}\"; class_names declares {:?}",
                    item,
                    class_names
                        .iter()
                        .map(|c| c.bind(py).str().map(|s| s.to_string()).unwrap_or_default())
                        .collect::<Vec<_>>()
                )));
            }
        }
    }
    Ok(encoded.into_pyarray(py).into_any())
}

/// Map predicted class indices back to the caller's labels.
fn decode_labels<'py>(
    py: Python<'py>,
    preds: &[f64],
    class_names: &[Py<PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    let out = pyo3::types::PyList::empty(py);
    for &p in preds {
        // Predictions are class indices; anything outside the declared range is a
        // bug in the model layer, not user input, so surface it rather than clamp.
        let idx = p as isize;
        let label = usize::try_from(idx)
            .ok()
            .and_then(|i| class_names.get(i))
            .ok_or_else(|| {
                CatBoostValueError::new_err(format!(
                    "predicted class index {p} is outside the {} declared class_names",
                    class_names.len()
                ))
            })?;
        out.append(label.bind(py))?;
    }
    Ok(out.into_any())
}

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
    #[pyo3(signature = (x, y = None, cat_features = None, eval_set = None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<&Bound<'_, PyAny>>,
        cat_features: Option<Vec<usize>>,
        eval_set: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        validate_params(&slf.base.params)?;
        // F17: `cat_features` from the fit kwarg, else from the constructor.
        // Remembered on the base because PREDICT must declare the same width
        // (F10 checks the pool's declared width against the model's).
        let cats = resolve_cat_features(&slf.base.params, py, cat_features)?;
        // `class_names` maps arbitrary labels onto class indices BEFORE ingestion,
        // because the ingestion path carries a float32 target and cannot hold
        // strings. Absent the parameter this is a no-op and the path is unchanged.
        let class_names = read_class_names(&slf.base.params, py)?;
        let encoded_y: Option<Bound<'_, PyAny>> = match (y, class_names.is_empty()) {
            (Some(y), false) => Some(encode_labels(py, y, &class_names)?),
            _ => None,
        };
        let y = match encoded_y.as_ref() {
            Some(e) => Some(e),
            None => y,
        };
        let pool = data_to_pool(py, x, y, Some(&cats))?;
        // PARAM-02: see the regressor for the eval-set contract.
        let eval_pools = match eval_set {
            Some(es) => eval_set_to_pools(py, es, &cats)?,
            None => Vec::new(),
        };
        if eval_pools.is_empty() {
            validate_eval_set_only_params(py, &slf.base.params, EVAL_SET_REMEDY_FIT)?;
        }
        let mut builder = make_builder(&slf.base.params, py)?;
        // The classifier defaults to a CLASSIFICATION loss (D-05). Only override
        // when the user supplied neither the canonical name nor its alias, so an
        // explicit `loss_function="CrossEntropy"` (etc.) is honored.
        if !slf.base.params.contains_key("loss_function") && !slf.base.params.contains_key("objective")
        {
            builder = builder.loss(Loss::Logloss);
        }
        let model = py
            .detach(|| fit_pool_with_eval(builder, &pool, &eval_pools))
            .map_err(PyCbError)?;
        slf.base.model = Some(model);
        slf.base.cat_features = cats;
        slf.base.class_names = class_names;
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
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(
                py,
                "this CatBoostClassifier is not fitted yet; call `fit` before `predict`",
            )
        })?;
        let pool = scoring_data_to_pool(py, x, &self.base.cat_features)?;
        let preds = py
            .detach(|| model.predict_with(&pool, PredictionType::Class))
            .map_err(PyCbError)?;
        // Without `class_names` the return is the UNCHANGED float64 class array, so
        // every existing caller is unaffected; with it, the caller gets their own
        // labels back (upstream's behaviour).
        if self.base.class_names.is_empty() {
            Ok(preds.to_pyarray(py).into_any())
        } else {
            decode_labels(py, &preds, &self.base.class_names)
        }
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
        let pool = scoring_data_to_pool(py, x, &self.base.cat_features)?;
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
        crate::estimator::partial_dependence_py(model, py, x, features, &self.base.cat_features)
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

    /// The class labels in CLASS-INDEX order (sklearn's `classes_`), i.e. exactly
    /// the `class_names` this estimator was fitted with.
    ///
    /// Present only when `class_names` was supplied. Without it this surface has no
    /// label mapping at all -- `predict` returns the raw `0.0`/`1.0` class indices --
    /// so reporting a `classes_` would imply a mapping that does not exist. Upstream
    /// derives `classes_` from the data in that case; that is a separate gap, and
    /// raising here is the honest signal rather than inventing `[0, 1]`.
    ///
    /// # Errors
    /// `NotFittedError` if unfitted; `CatBoostValueError` if fitted without
    /// `class_names`.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if !self.base.is_fitted() {
            return Err(not_fitted_err(
                py,
                "this CatBoostClassifier is not fitted yet; call `fit` before `classes_`",
            ));
        }
        if self.base.class_names.is_empty() {
            return Err(CatBoostValueError::new_err(
                "classes_ is available only when the estimator was fitted with \
                 `class_names`; without it this classifier does not map labels and \
                 `predict` returns raw 0.0/1.0 class indices",
            ));
        }
        let out = pyo3::types::PyList::empty(py);
        for c in &self.base.class_names {
            out.append(c.bind(py))?;
        }
        Ok(out.into_any())
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
        let pool = scoring_data_to_pool(py, x, &self.base.cat_features)?;
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
