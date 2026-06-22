//! Shared estimator base logic: the verbatim kwargs store (D-06) + the fitted
//! model handle.
//!
//! `__init__` stores constructor kwargs verbatim (D-06: NO work / validation /
//! coercion). The param-vocabulary registry, alias handling, the
//! kwargs -> [`CatBoostBuilder`] map, and unknown/unsupported-param rejection
//! (D-05 / D-07) live in [`crate::params`] and run at `fit()` time.

use std::collections::BTreeMap;

use catboost_rs::{CatBoostBuilder, CatBoostError, Model, Pool};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
