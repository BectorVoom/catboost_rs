//! Shared estimator base logic: the verbatim kwargs store (D-06) + the fitted
//! model handle.
//!
//! `__init__` stores constructor kwargs verbatim (D-06: NO work / validation /
//! coercion). The param-vocabulary registry, alias handling, the
//! kwargs -> [`CatBoostBuilder`] map, and unknown/unsupported-param rejection
//! (D-05 / D-07) live in [`crate::params`] and run at `fit()` time.

use std::collections::BTreeMap;
use std::path::Path;

use catboost_rs::{CatBoostBuilder, CatBoostError, IngestSource, Model, Pool};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::CatBoostValueError;
use crate::ingest_py::ingest_to_owned;

/// Shared estimator state: kwargs stored verbatim (D-06) + the fitted model
/// (`None` until `fit` runs — the not-fitted sentinel; the typed `NotFittedError`
/// lands in 08-05).
pub(crate) struct EstimatorBase {
    /// Constructor kwargs, stored exactly as received (D-06). Keyed by name so
    /// `get_params`/`set_params` round-trip in later plans.
    pub(crate) params: BTreeMap<String, Py<PyAny>>,
    /// The fitted model; `None` means not-yet-fitted.
    pub(crate) model: Option<Model>,
}

impl EstimatorBase {
    /// Build an empty (unfitted) base from optional `**kwargs`, storing every key
    /// verbatim. No validation or coercion happens here (D-06).
    ///
    /// # Errors
    /// Propagates any failure extracting a kwargs key as a string.
    pub(crate) fn from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut params = BTreeMap::new();
        if let Some(dict) = kwargs {
            for (key, value) in dict.iter() {
                let name: String = key.extract()?;
                params.insert(name, value.unbind());
            }
        }
        Ok(Self {
            params,
            model: None,
        })
    }

    /// Build a fitted base directly from a loaded [`Model`] (no kwargs), used by
    /// the `load_model` constructors. `params` is empty (the loaded model already
    /// embeds its trained configuration); `model` is `Some(model)`.
    #[must_use]
    pub(crate) fn from_model(model: Model) -> Self {
        Self {
            params: BTreeMap::new(),
            model: Some(model),
        }
    }

    /// Return the verbatim constructor kwargs as a fresh `dict` (the sklearn
    /// `get_params` contract). The store is keyed by the EXACT name the user passed
    /// (D-06), so `set_params(**get_params())` is an identity round-trip and
    /// `sklearn.base.clone` (which does `__init__(**get_params())`) reconstructs an
    /// equal-params unfitted estimator (T-08-15). `deep` is accepted for sklearn
    /// signature parity; there are no nested sub-estimators, so it is a no-op.
    ///
    /// # Errors
    /// Propagates any failure cloning a stored value into the new dict.
    pub(crate) fn get_params<'py>(
        &self,
        py: Python<'py>,
        _deep: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, value) in &self.params {
            dict.set_item(name, value.bind(py))?;
        }
        Ok(dict)
    }

    /// Merge `**params` into the verbatim store (the sklearn `set_params`
    /// contract). Each key overwrites verbatim; no validation or coercion happens
    /// here (validation stays at `fit`, D-06). Keys NOT already present are still
    /// accepted (sklearn's `set_params` allows setting any valid `__init__` param).
    ///
    /// # Errors
    /// Propagates any failure extracting a key as a string.
    pub(crate) fn set_params(&mut self, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(dict) = params {
            for (key, value) in dict.iter() {
                let name: String = key.extract()?;
                self.params.insert(name, value.unbind());
            }
        }
        Ok(())
    }

    /// `true` once `fit` (or `load_model`) has populated the model handle.
    #[must_use]
    pub(crate) fn is_fitted(&self) -> bool {
        self.model.is_some()
    }
}

/// Build the sklearn ≥1.6 `Tags` dataclass for an estimator of `estimator_type`
/// (`"classifier"` | `"regressor"`). sklearn 1.6 replaced the old `_get_tags()`
/// dict with the `__sklearn_tags__()` dataclass (RESEARCH Pitfall 5); modern
/// `check_estimator` reads `estimator_type` and the per-kind sub-tags off this
/// object. We construct it by calling into Python (`sklearn.utils.Tags` +
/// `TargetTags`/`ClassifierTags`/`RegressorTags`/`InputTags`) so we always match
/// the installed sklearn's exact dataclass shape rather than hard-coding fields.
///
/// `required=True` on the target tags marks the estimators as supervised (both
/// the classifier and regressor require `y` at fit). The Ranker presents with the
/// regressor-like `"regressor"` tag set (continuous score output) per RESEARCH
/// Open Q2; it is EXCLUDED from the `check_estimator` gate (Task 2).
///
/// # Errors
/// Propagates any failure importing `sklearn.utils` or constructing the dataclass.
pub(crate) fn build_sklearn_tags<'py>(
    py: Python<'py>,
    estimator_type: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let utils = py.import(intern!(py, "sklearn.utils"))?;
    let tags_cls = utils.getattr(intern!(py, "Tags"))?;
    let target_tags = utils
        .getattr(intern!(py, "TargetTags"))?
        .call1((true,))?; // TargetTags(required=True)
    let input_tags = utils.getattr(intern!(py, "InputTags"))?.call0()?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "estimator_type"), estimator_type)?;
    kwargs.set_item(intern!(py, "target_tags"), target_tags)?;
    kwargs.set_item(intern!(py, "input_tags"), input_tags)?;
    if estimator_type == "classifier" {
        let clf = utils.getattr(intern!(py, "ClassifierTags"))?.call0()?;
        kwargs.set_item(intern!(py, "classifier_tags"), clf)?;
    } else {
        let reg = utils.getattr(intern!(py, "RegressorTags"))?.call0()?;
        kwargs.set_item(intern!(py, "regressor_tags"), reg)?;
    }
    tags_cls.call((), Some(&kwargs))
}

/// Coefficient of determination R² of `pred` vs the true `y` (the sklearn
/// `RegressorMixin.score` default). `R² = 1 - SS_res / SS_tot`; when `SS_tot == 0`
/// (constant `y`) sklearn returns `0.0` for a non-perfect fit, `1.0` for a perfect
/// one — mirror that.
#[must_use]
pub(crate) fn r2_score(y: &[f64], pred: &[f64]) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    if ss_tot == 0.0 {
        return if ss_res == 0.0 { 1.0 } else { 0.0 };
    }
    1.0 - ss_res / ss_tot
}

/// Mean accuracy of class predictions vs `y` (the sklearn
/// `ClassifierMixin.score` default). Labels are compared after rounding the
/// f64 predictions to the nearest integer (the classifier emits `0.0`/`1.0`).
#[must_use]
pub(crate) fn accuracy_score(y: &[f64], pred: &[f64]) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    // Compare the rounded labels as integers (the intent is integer equality).
    // The previous `< f64::EPSILON` form was correct only for 0.0/1.0 binary
    // labels — EPSILON (~2.2e-16) is the representable gap near 1.0, so equal
    // integer-valued f64s with magnitude > ~2 could compare unequal, and it is a
    // latent bug for multiclass labels (WR-06). A non-finite (NaN) rounded value
    // never matches (the `i64` guard short-circuits via the finite check).
    let correct = y
        .iter()
        .zip(pred.iter())
        .filter(|(t, p)| {
            let (tr, pr) = (t.round(), p.round());
            tr.is_finite() && pr.is_finite() && (tr as i64) == (pr as i64)
        })
        .count();
    correct as f64 / n as f64
}

/// Build a [`CatBoostBuilder`] from the params and fit it on an OWNED pool. The
/// caller is expected to invoke this under `py.detach` (the `pool` is owned, so
/// no Python buffer borrow is alive — D-11). Returns the typed facade
/// [`CatBoostError`]; the caller maps it via `errors::to_pyerr` (PYAPI-05).
///
/// # Errors
/// Returns the facade [`CatBoostError`] training error.
pub(crate) fn fit_pool(builder: CatBoostBuilder, pool: &Pool) -> Result<Model, CatBoostError> {
    builder.fit(pool)
}

/// Build a facade [`Pool`] from `x` (+ optional `y`), accepting EITHER a native
/// [`crate::pool::Pool`] OR a framework object (NumPy / Pandas / Arrow / Polars).
///
/// Shared by `CatBoostRegressor`, `CatBoostClassifier`, and `CatBoostRanker` so
/// the three estimators ingest identically (prep for the 08-05 sklearn contract).
///
/// When `x` is a `Pool`, its inherited `into_pool()` validation runs (and any `y`
/// is ignored — the Pool already carries its label). Otherwise `x`/`y` route
/// through the shared ingest adapter. In both cases the result is fully owned, so
/// the caller may `py.detach()` immediately (D-11 / PYAPI-06).
///
/// # Error-surface asymmetry (WR-04)
///
/// The two input kinds validate at DIFFERENT points, by design:
/// - A NumPy / Pandas / Arrow / Polars `x` runs the strict D-12 input checks
///   (float32 / contiguity / nullability) eagerly during ingestion here.
/// - A native `Pool` already had those checks run at its OWN construction, so the
///   fast-path runs only `OwnedColumns::into_pool()`'s length check. A
///   feature-width mismatch against the fitted model is therefore NOT caught here;
///   it surfaces later as the facade's `FeatureMismatch` inside `predict_with`
///   (still a typed error, just raised deeper in the call stack).
///
/// Additionally, on the `Pool` fast-path the `y` argument is IGNORED — the `Pool`
/// already carries its own label, so a `y` passed alongside a `Pool` is silently
/// dropped (the Pool is the single source of truth).
///
/// # Errors
/// [`CatBoostValueError`] on any dtype / layout / length / nullability failure.
pub(crate) fn data_to_pool(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<Pool> {
    if let Ok(pool_ref) = x.cast::<crate::pool::Pool>() {
        // Pool fast-path (WR-04): `y` is intentionally ignored (the Pool carries its
        // own label) and only the inherited length check runs here — a feature-width
        // mismatch defers to the facade's `FeatureMismatch` inside `predict_with`.
        return pool_ref.borrow().to_pool();
    }
    ingest_to_owned(py, x, y, None)?
        .into_pool()
        .map_err(|e| CatBoostValueError::new_err(e.to_string()))
}

/// Load a reference model from `path`, dispatching on the file extension: a
/// `.json` path loads via [`Model::load_json`], anything else (notably `.cbm`)
/// loads via [`Model::load_cbm`]. Shared by the `load_model` constructors on the
/// regressor and classifier (the single deterministic oracle path, Path (a)).
///
/// A malformed model surfaces as the facade `CatBoostError::Deserialize` /
/// `SchemaVersion`, mapped by [`crate::errors::to_pyerr`] to `CatBoostValueError`
/// (threat T-08-12) — never a panic.
///
/// # Errors
/// `CatBoostValueError` (via [`crate::errors::PyCbError`]) on a malformed /
/// unreadable model file.
pub(crate) fn load_model_path(path: &str) -> PyResult<Model> {
    let p = Path::new(path);
    let model = if p.extension().and_then(|e| e.to_str()) == Some("json") {
        Model::load_json(p)
    } else {
        Model::load_cbm(p)
    };
    model.map_err(|e| crate::errors::PyCbError(e).into())
}
