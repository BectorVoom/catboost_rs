#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `catboost_rs` — the PyO3 binding crate (Phase 8).
//!
//! Wraps the published [`catboost_rs`](catboost_rs) facade
//! (`CatBoostBuilder` -> `Model::predict`) into a CatBoost-mirror Python module
//! whose import name is `catboost_rs` (D-09). Plan 08-01 stands up the thinnest
//! end-to-end vertical slice: `CatBoostRegressor().fit(X, y).predict(X)` over a
//! float32 contiguous NumPy array, wired through the real facade — not a stub.
//!
//! Param validation, the full registry, multi-source ingestion, the classifier /
//! ranker, the error taxonomy, and packaging land in later slices.

use pyo3::prelude::*;

mod classifier;
mod errors;
mod estimator;
mod ingest_py;
mod params;
mod pool;
mod ranker;
mod regressor;
mod utils;

pub use classifier::CatBoostClassifier;
pub use pool::Pool;
pub use ranker::CatBoostRanker;
pub use regressor::CatBoostRegressor;

/// The `catboost_rs` Python module (D-09 import name; `module-name` in
/// `pyproject.toml`). Plan 08-01 registers only [`CatBoostRegressor`]; the
/// classifier / ranker / `Pool` and the exception taxonomy land in later plans.
///
/// `gil_used = false` (PyO3 0.29, PYAPI-06): declares this module
/// free-threaded-aware. The contract is upheld by the **own-before-detach**
/// discipline (08-03 / D-11) — every Python buffer is copied into Rust-owned
/// `OwnedColumns` *before* the GIL is released, so no borrow into a live Python
/// buffer ever survives a `Python::detach`. Without this flag, importing the
/// module on a free-threaded interpreter (`python3.13t`) would silently
/// re-enable the GIL with a warning. The one documented exception is the
/// custom-loss / custom-metric callback path, which re-enters Python during
/// compute via per-call `Python::attach` (serialized) — see `FREE_THREADING.md`.
#[pymodule(gil_used = false)]
fn catboost_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CatBoostRegressor>()?;
    // PYAPI-03 native classifier + ranker (the CatBoost-mirror estimator trio).
    m.add_class::<CatBoostClassifier>()?;
    m.add_class::<CatBoostRanker>()?;
    // PYAPI-03 native Pool — a user can pass an explicit Pool to fit/predict.
    m.add_class::<Pool>()?;
    // PYAPI-05 typed-exception taxonomy (CatBoostError base + Parameter/Value/
    // NotFitted subclasses), importable as `catboost_rs.<Name>`.
    errors::register(py, m)?;
    // D-07 registry introspection helper for the param-coverage test.
    m.add_function(wrap_pyfunction!(params::_param_status, m)?)?;
    // ORCH-04-S6 standalone metric surface: `catboost_rs.utils.eval_metric`.
    utils::register(py, m)?;
    Ok(())
}

// Source/test separation (CLAUDE.md): `*_test` modules are declared at the crate
// root, mirroring the facade crate's `#[cfg(test)] mod error_test;` idiom.
#[cfg(test)]
mod ingest_py_test;
#[cfg(test)]
mod errors_test;
#[cfg(test)]
mod estimator_test;
#[cfg(test)]
mod params_test;
