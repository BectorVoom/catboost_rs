//! ORCH-01-S6 — the module-level `catboost_rs.cv` cross-validation surface.
//!
//! A thin PyO3 wrapper over the published [`catboost_rs::cv`] facade (ORCH-01-S5):
//! ingest the `pool` (a native [`crate::pool::Pool`] or an `(X, y)` framework
//! object), build a `CatBoostBuilder` from the `params` dict via
//! [`make_builder`], derive the metric list, OWN every Python-borrowed buffer
//! under the GIL, release the GIL for the per-fold train/eval compute
//! (`py.detach`, D-11), and return the upstream cv columns as a `dict`.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::errors::{CatBoostValueError, PyCbError};
use crate::estimator::data_to_pool;
use crate::params::make_builder;

/// Resolve the requested metric descriptors: an explicit `str` / `list[str]`, or
/// (when `None`) the canonical metric derived from `params["loss_function"]`
/// (default `"RMSE"`). An empty `list[str]` is passed through unchanged — the
/// facade `cv` rejects it with a typed error (mapped through [`PyCbError`]).
///
/// # Errors
/// [`CatBoostValueError`] if `metrics` is neither a `str` nor a `list[str]`.
///
/// Shared with `search.rs` (grid/randomized search) so the metric-default rule
/// is single-sourced.
pub(crate) fn resolve_metrics(
    metrics: Option<&Bound<'_, PyAny>>,
    loss_function: Option<&str>,
) -> PyResult<Vec<String>> {
    match metrics {
        None => Ok(vec![loss_function.unwrap_or("RMSE").to_owned()]),
        Some(obj) => {
            if let Ok(name) = obj.extract::<String>() {
                Ok(vec![name])
            } else if let Ok(names) = obj.extract::<Vec<String>>() {
                Ok(names)
            } else {
                Err(CatBoostValueError::new_err(
                    "`metrics` must be a str or a list of str",
                ))
            }
        }
    }
}

/// Ingest the `pool` argument into an owned facade [`Pool`](catboost_rs::Pool):
/// a 2-tuple `(X, y)` splits into data + label; anything else (a native `Pool`
/// or a lone framework array) forwards to [`data_to_pool`] with no separate `y`.
///
/// # Errors
/// [`CatBoostValueError`] on any dtype / layout / length failure (via
/// [`data_to_pool`]).
fn pool_from_arg(py: Python<'_>, pool: &Bound<'_, PyAny>) -> PyResult<catboost_rs::Pool> {
    if let Ok(tuple) = pool.cast::<PyTuple>() {
        if tuple.len() == 2 {
            let x = tuple.get_item(0)?;
            let y = tuple.get_item(1)?;
            return data_to_pool(py, &x, Some(&y));
        }
    }
    data_to_pool(py, pool, None)
}

/// `catboost_rs.cv(pool, params, fold_count=3, inverted=False,
/// partition_random_seed=0, seed=None, shuffle=True, folds=None, metrics=None)`
/// — k-fold cross-validate `params` on `pool`, returning the upstream cv columns
/// as a dict: `iterations` plus `test-<M>-mean/std` and `train-<M>-mean/std`.
///
/// `seed` (upstream alias) overrides `partition_random_seed` when supplied. When
/// `metrics` is `None` the metric list is derived from `params["loss_function"]`
/// (default `"RMSE"`).
///
/// # Errors
/// [`CatBoostValueError`] on a bad `pool` / `params` / `metrics` at the boundary;
/// the mapped facade [`catboost_rs::CatBoostError`] (via [`PyCbError`]) on an
/// empty / unknown metric, an unsupported (categorical / CTR / non-oblivious)
/// model, an empty explicit `folds` list, or any training / evaluation failure.
/// Never panics.
#[pyfunction]
#[pyo3(signature = (
    pool,
    params,
    fold_count = 3,
    inverted = false,
    partition_random_seed = 0,
    seed = None,
    shuffle = true,
    folds = None,
    metrics = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cv(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    params: &Bound<'_, PyDict>,
    fold_count: usize,
    inverted: bool,
    partition_random_seed: u64,
    seed: Option<u64>,
    shuffle: bool,
    folds: Option<Vec<Vec<usize>>>,
    metrics: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    // --- GIL HELD: own every Python-borrowed input before any detach (D-11). ---
    // Copy the `params` dict into the owned `BTreeMap` `make_builder` consumes
    // (verbatim keys, mirroring `EstimatorBase::from_kwargs`).
    let mut params_map: BTreeMap<String, Py<PyAny>> = BTreeMap::new();
    for (key, value) in params.iter() {
        let name: String = key.extract()?;
        params_map.insert(name, value.unbind());
    }

    let pool = pool_from_arg(py, pool)?;
    let builder = make_builder(&params_map, py)?;

    // The default metric derives from `loss_function` (default "RMSE").
    let loss_function: Option<String> = params_map
        .get("loss_function")
        .and_then(|v| v.bind(py).extract::<String>().ok());
    let metric_names = resolve_metrics(metrics, loss_function.as_deref())?;

    // `seed` (the upstream alias) overrides `partition_random_seed` when supplied.
    let seed_val = seed.unwrap_or(partition_random_seed);

    // Intentional divergence from upstream (SPEC §8): upstream `catboost.cv()`
    // RAISES when `folds` is combined with any of `fold_count` / `shuffle` /
    // `partition_random_seed` / `inverted`. This binding does NOT replicate that
    // check — an explicit `folds` list silently OVERRIDES the partition knobs,
    // matching the Rust `cv()` signature (whose `fold_count: usize` cannot
    // represent "unset"). This is a deliberate, documented adaptation, not a
    // functional bug — do NOT "fix" it into an error without revisiting SPEC §8.

    // --- owned data only: release the GIL for the per-fold train/eval compute. ---
    let result = py
        .detach(|| {
            let metric_refs: Vec<&str> = metric_names.iter().map(String::as_str).collect();
            catboost_rs::cv(
                &pool,
                &builder,
                &metric_refs,
                fold_count,
                shuffle,
                seed_val,
                inverted,
                folds.as_deref(),
            )
        })
        .map_err(PyCbError)?;

    // Build the upstream cv columns dict: `iterations` + one list per column.
    let dict = PyDict::new(py);
    dict.set_item("iterations", PyList::new(py, &result.iterations)?)?;
    for (name, column) in &result.columns {
        dict.set_item(name, PyList::new(py, column)?)?;
    }
    Ok(dict.unbind())
}
