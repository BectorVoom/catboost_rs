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
/// # Errors
/// [`CatBoostValueError`] on any dtype / layout / length / nullability failure.
pub(crate) fn data_to_pool(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<Pool> {
    if let Ok(pool_ref) = x.cast::<crate::pool::Pool>() {
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
