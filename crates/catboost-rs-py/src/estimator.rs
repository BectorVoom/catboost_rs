//! Shared estimator base logic for the 08-01 walking-skeleton slice.
//!
//! Stores constructor kwargs verbatim (D-06: NO work / validation / coercion in
//! `__init__`) and maps the five smoke parameters onto the Rust-only
//! [`CatBoostBuilder`] at `fit()` time. The full param-vocabulary registry, alias
//! handling, and unknown/unsupported-param rejection (D-05 / D-07) land in plan
//! 08-02 — here unknown keys are ignored for the smoke.

use std::collections::BTreeMap;

use catboost_rs::{CatBoostBuilder, CatBoostError, Model, Pool};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The smoke parameters the 08-01 slice reads off the verbatim kwargs store.
/// Everything else is ignored here (full registry: 08-02).
const SMOKE_PARAMS: [&str; 5] = [
    "iterations",
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "random_seed",
];

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

    /// Read a stored param as `f64`, if present and numeric.
    fn get_f64(&self, py: Python<'_>, name: &str) -> PyResult<Option<f64>> {
        match self.params.get(name) {
            Some(v) => Ok(Some(v.bind(py).extract::<f64>()?)),
            None => Ok(None),
        }
    }

    /// Read a stored param as `usize`, if present and integral.
    fn get_usize(&self, py: Python<'_>, name: &str) -> PyResult<Option<usize>> {
        match self.params.get(name) {
            Some(v) => Ok(Some(v.bind(py).extract::<usize>()?)),
            None => Ok(None),
        }
    }

    /// Read a stored param as `u64`, if present and integral.
    fn get_u64(&self, py: Python<'_>, name: &str) -> PyResult<Option<u64>> {
        match self.params.get(name) {
            Some(v) => Ok(Some(v.bind(py).extract::<u64>()?)),
            None => Ok(None),
        }
    }

    /// Construct a [`CatBoostBuilder`] from the stored smoke params. Unknown keys
    /// are ignored in this slice (full validation: 08-02).
    ///
    /// # Errors
    /// A `PyTypeError`/`PyValueError` (via `extract`) if a smoke param is present
    /// but not the expected numeric type.
    pub(crate) fn make_builder(&self, py: Python<'_>) -> PyResult<CatBoostBuilder> {
        let _ = &SMOKE_PARAMS; // documents the read set; alias/registry is 08-02.
        let mut builder = CatBoostBuilder::new();
        if let Some(iterations) = self.get_usize(py, "iterations")? {
            builder = builder.iterations(iterations);
        }
        if let Some(depth) = self.get_usize(py, "depth")? {
            builder = builder.depth(depth);
        }
        if let Some(learning_rate) = self.get_f64(py, "learning_rate")? {
            builder = builder.learning_rate(learning_rate);
        }
        if let Some(l2_leaf_reg) = self.get_f64(py, "l2_leaf_reg")? {
            builder = builder.l2_leaf_reg(l2_leaf_reg);
        }
        if let Some(random_seed) = self.get_u64(py, "random_seed")? {
            builder = builder.random_seed(random_seed);
        }
        Ok(builder)
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
