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

/// The `catboost_rs` Python module (D-09 import name; `module-name` in
/// `pyproject.toml`). Plan 08-01 Task 2 stands up the empty module; Task 3
/// registers [`CatBoostRegressor`].
#[pymodule]
fn catboost_rs(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
