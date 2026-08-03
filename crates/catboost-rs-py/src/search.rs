//! ORCH-02-S5 / -S6 — the module-level `catboost_rs.grid_search` /
//! `catboost_rs.randomized_search` hyperparameter-search surface.
//!
//! Thin PyO3 wrappers over the published [`catboost_rs::grid_search`] /
//! [`catboost_rs::randomized_search`] facades (ORCH-02-S3/S4). Because Rust has
//! no dict-based param-grid concept, the dict/product ergonomics live HERE (in
//! Python-land) and the failure-isolating value-selection + cv loop stays in the
//! facade:
//!
//! 1. Read the `estimator`'s base params via its sklearn `get_params()` contract.
//! 2. Expand `param_grid` (`dict[str, list]`) into the Cartesian product of
//!    param-override dicts (STABLE sorted key order for determinism); for each
//!    grid point, merge over the base params, `validate_params`, and build a
//!    [`CatBoostBuilder`] via [`make_builder`] — so a bad param RAISES before any
//!    training ([`expand_param_grid`]).
//! 3. Ingest `(X, y)` into an owned [`Pool`](catboost_rs::Pool) under the GIL
//!    (D-11), OWN every input, then release the GIL (`py.detach`) for the
//!    per-candidate cv compute.
//! 4. Return `{"params": <best grid point>, "cv_results": {columns}}`.
//!
//! **Refit choice (SPEC §9 Q3).** The facade is always called with `refit=false`
//! (we never need the facade's own `best_model`); when the Python `refit=true`,
//! the passed `estimator` is refit IN PLACE — `set_params(**best_grid_point)`
//! then `fit(X, y)` — matching upstream's in-place refit and avoiding a wasted
//! second training. The estimator API (`set_params` / `fit`, sklearn contract)
//! supports this cleanly for the regressor / classifier / ranker uniformly.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use catboost_rs::{CatBoostBuilder, ErrorScore, SearchResult};

use crate::errors::{CatBoostValueError, FitFailedWarning, PyCbError};
use crate::estimator::data_to_pool;
use crate::params::{
    make_builder, validate_eval_set_only_params, validate_params, EVAL_SET_REMEDY_SEARCH,
};

/// Read the `estimator`'s base params through its sklearn `get_params()` contract
/// (implemented uniformly by the regressor / classifier / ranker), copying every
/// key verbatim into an owned `BTreeMap` — the same store [`make_builder`]
/// consumes.
///
/// # Errors
/// Propagates any failure calling `get_params()` or extracting a key as a string.
fn base_params(estimator: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, Py<PyAny>>> {
    let params_obj = estimator.call_method0("get_params")?;
    let params_dict = params_obj.cast::<PyDict>()?;
    let mut base: BTreeMap<String, Py<PyAny>> = BTreeMap::new();
    for (key, value) in params_dict.iter() {
        let name: String = key.extract()?;
        base.insert(name, value.unbind());
    }
    Ok(base)
}

/// Resolve the requested metric descriptors: an explicit `str` / `list[str]`, or
/// (when `None`) the canonical metric derived from `loss_function` (default
/// `"RMSE"`). Mirrors the `cv.rs` resolution so the search-metric default matches
/// the cross-validation default. An empty `list[str]` is passed through — the
/// facade rejects it with a typed error (mapped via [`PyCbError`]).
///
// `resolve_metrics` is single-sourced in `cv.rs` and shared here so the
// metric-default rule (str/list[str], else `loss_function` default "RMSE")
// cannot drift between the cv and search Python surfaces.
use crate::cv::resolve_metrics;

/// (ORCH-02-S5) Expand `grid` (`dict[str, list]`) into the Cartesian product of
/// candidate builders. For determinism the grid keys are sorted, and the product
/// is built in that stable key order. For EACH grid point the chosen values are
/// merged OVER `base` (overriding on key collision), `validate_params` runs (so a
/// KnownNotYet / unknown param RAISES before any training), and a
/// [`CatBoostBuilder`] is produced via [`make_builder`].
///
/// Returns the parallel `(builders, grid_points)` — `grid_points[i]` is the
/// per-combination override dict (the winning one becomes the `"params"` in the
/// returned search dict). An empty per-parameter list yields an empty product
/// (no candidates), which the facade rejects with a typed error; an empty `grid`
/// yields a single candidate equal to `base`.
///
/// # Errors
/// [`crate::errors::CatBoostParameterError`] (via `validate_params`) on a bad
/// grid param, a `PyTypeError`/`PyValueError` (via `make_builder`) on a bad
/// value, or a non-iterable grid value.
pub(crate) fn expand_param_grid(
    base: &BTreeMap<String, Py<PyAny>>,
    grid: &Bound<'_, PyDict>,
    py: Python<'_>,
) -> PyResult<(Vec<CatBoostBuilder>, Vec<Py<PyDict>>)> {
    // (key, [values...]) pairs in STABLE sorted-key order (determinism).
    let mut dims: Vec<(String, Vec<Py<PyAny>>)> = Vec::with_capacity(grid.len());
    for (key, value) in grid.iter() {
        let name: String = key.extract()?;
        let mut values: Vec<Py<PyAny>> = Vec::new();
        for item in value.try_iter()? {
            values.push(item?.unbind());
        }
        dims.push((name, values));
    }
    dims.sort_by(|a, b| a.0.cmp(&b.0));

    // Iterative Cartesian product of the per-dimension value indices, aligned
    // with `dims` order. Avoids any slice indexing (clippy `indexing_slicing`).
    let mut combos: Vec<Vec<Py<PyAny>>> = vec![Vec::new()];
    for (_, values) in &dims {
        let mut next: Vec<Vec<Py<PyAny>>> = Vec::with_capacity(combos.len() * values.len());
        for combo in &combos {
            for value in values {
                let mut extended: Vec<Py<PyAny>> =
                    combo.iter().map(|p| p.clone_ref(py)).collect();
                extended.push(value.clone_ref(py));
                next.push(extended);
            }
        }
        combos = next;
    }

    let mut builders: Vec<CatBoostBuilder> = Vec::with_capacity(combos.len());
    let mut grid_points: Vec<Py<PyDict>> = Vec::with_capacity(combos.len());
    for combo in combos {
        // merged = base (deep-cloned refs) overlaid with this grid point.
        let mut merged: BTreeMap<String, Py<PyAny>> = BTreeMap::new();
        for (name, value) in base {
            merged.insert(name.clone(), value.clone_ref(py));
        }
        let point = PyDict::new(py);
        for ((name, _), value) in dims.iter().zip(combo.iter()) {
            merged.insert(name.clone(), value.clone_ref(py));
            point.set_item(name, value.bind(py))?;
        }
        validate_params(&merged)?;
        // Each candidate is fit through `builder.fit(&train_pool)` — the
        // learn-only path, no eval set — so the detector / best-model params
        // would be silently inert. Checked on the MERGED map so a grid that
        // sweeps one (`{"early_stopping_rounds": [10, 20]}`) is caught as well as
        // one inherited from the estimator's constructor kwargs.
        validate_eval_set_only_params(py, &merged, EVAL_SET_REMEDY_SEARCH)?;
        let builder = make_builder(&merged, py)?;
        builders.push(builder);
        grid_points.push(point.unbind());
    }
    Ok((builders, grid_points))
}

/// Refit the passed `estimator` IN PLACE with the winning grid point
/// (`set_params(**best_grid_point)` then `fit(X, y)`), mirroring upstream's
/// in-place refit (SPEC §9 Q3).
///
/// # Errors
/// A typed error if `best_index` is out of range, or any failure from the
/// estimator's own `set_params` / `fit`.
fn refit_estimator(
    py: Python<'_>,
    estimator: &Bound<'_, PyAny>,
    grid_points: &[Py<PyDict>],
    best_index: usize,
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let best = grid_points.get(best_index).ok_or_else(|| {
        CatBoostValueError::new_err("search: selected index out of range for refit")
    })?;
    estimator.call_method("set_params", (), Some(best.bind(py)))?;
    match y {
        Some(y) => {
            estimator.call_method1("fit", (x, y))?;
        }
        None => {
            estimator.call_method1("fit", (x,))?;
        }
    }
    Ok(())
}

/// Build the upstream `{"params": <best grid point>, "cv_results": {"iterations":
/// [...], "test-<M>-mean": [...], ...}}` return dict. Shared by both
/// [`grid_search`] and [`randomized_search`] (dedupe).
///
/// # Errors
/// A typed error if `res.best_index` is out of range of `grid_points`.
fn result_to_pydict(
    py: Python<'_>,
    res: &SearchResult,
    grid_points: &[Py<PyDict>],
) -> PyResult<Py<PyDict>> {
    let params = grid_points.get(res.best_index).ok_or_else(|| {
        CatBoostValueError::new_err("search: selected index out of range for result")
    })?;
    let out = PyDict::new(py);
    out.set_item("params", params.bind(py))?;

    let cv_dict = PyDict::new(py);
    cv_dict.set_item("iterations", PyList::new(py, &res.cv_results.iterations)?)?;
    for (name, column) in &res.cv_results.columns {
        cv_dict.set_item(name, PyList::new(py, column)?)?;
    }
    out.set_item("cv_results", cv_dict)?;
    Ok(out.unbind())
}

/// Parse the Python `error_score` argument into the facade [`ErrorScore`]
/// policy, matching scikit-learn's `GridSearchCV(error_score=...)` contract:
///
/// - `None` (the kwarg omitted / Python `None`) ⇒ [`ErrorScore::Value`]`(NaN)`
///   (sklearn's default `np.nan`): a failed candidate is ranked worst.
/// - the string `"raise"` ⇒ [`ErrorScore::Raise`] (re-raise the first failure);
///   any OTHER string ⇒ [`CatBoostValueError`].
/// - a float / int (incl. a `NaN` float) ⇒ [`ErrorScore::Value`] of that number
///   (a finite value competes normally).
/// - anything else ⇒ [`CatBoostValueError`].
///
/// # Errors
/// [`CatBoostValueError`] for a non-`"raise"` string or a non-numeric,
/// non-string object.
fn parse_error_score(obj: Option<&Bound<'_, PyAny>>) -> PyResult<ErrorScore> {
    let Some(obj) = obj else {
        return Ok(ErrorScore::Value(f64::NAN));
    };
    if obj.is_none() {
        return Ok(ErrorScore::Value(f64::NAN));
    }
    // A string: only "raise" is valid (check BEFORE the numeric branch, since a
    // string never extracts as f64 but the ordering keeps the error message
    // specific).
    if let Ok(s) = obj.extract::<String>() {
        if s == "raise" {
            return Ok(ErrorScore::Raise);
        }
        return Err(CatBoostValueError::new_err(
            "error_score must be 'raise' or a number",
        ));
    }
    // A number (int / float, including a NaN float).
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(ErrorScore::Value(v));
    }
    Err(CatBoostValueError::new_err(
        "error_score must be 'raise' or a number",
    ))
}

/// Emit a [`FitFailedWarning`] (sklearn parity: warn, do NOT raise) when one or
/// more candidate fits were assigned `error_score`. Must run with the GIL held
/// (after `py.detach` returns). `total` is the number of candidate fits
/// attempted.
///
/// # Errors
/// Propagates any failure importing `warnings` or calling `warnings.warn`.
fn warn_fit_failed(py: Python<'_>, failures: &[(usize, String)], total: usize) -> PyResult<()> {
    let n = failures.len();
    let first = failures.first().map_or("", |(_, m)| m.as_str());
    let msg =
        format!("{n} of {total} candidate fits failed and were assigned error_score; first error: {first}");
    let category = py.get_type::<FitFailedWarning>();
    py.import("warnings")?
        .getattr("warn")?
        .call1((msg, category))?;
    Ok(())
}

/// `catboost_rs.grid_search(estimator, param_grid, X, y=None, cv=3,
/// partition_random_seed=0, shuffle=True, inverted=False, folds=None,
/// metrics=None, refit=True)` — cross-validate the Cartesian product of
/// `param_grid` (merged over the estimator's base params) and keep the best by
/// the primary metric, returning `{"params", "cv_results"}`. When `refit`, the
/// `estimator` is refit in place on `(X, y)` with the winning params.
///
/// When `metrics` is `None` the metric derives from the estimator's
/// `loss_function` (default `"RMSE"`).
///
/// # Errors
/// [`crate::errors::CatBoostParameterError`] on a bad grid param; the mapped
/// facade [`catboost_rs::CatBoostError`] (via [`PyCbError`]) on an empty /
/// unknown metric, an unsupported (categorical / CTR / non-oblivious) candidate,
/// an empty explicit `folds` list, or any training / evaluation failure; a
/// [`CatBoostValueError`] on a bad `X` / `y`. Never panics.
#[pyfunction]
#[pyo3(signature = (
    estimator,
    param_grid,
    x,
    y = None,
    cv = 3,
    partition_random_seed = 0,
    shuffle = true,
    inverted = false,
    folds = None,
    metrics = None,
    refit = true,
    error_score = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn grid_search(
    py: Python<'_>,
    estimator: &Bound<'_, PyAny>,
    param_grid: &Bound<'_, PyDict>,
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
    cv: usize,
    partition_random_seed: u64,
    shuffle: bool,
    inverted: bool,
    folds: Option<Vec<Vec<usize>>>,
    metrics: Option<&Bound<'_, PyAny>>,
    refit: bool,
    error_score: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    // --- GIL HELD: own every Python-borrowed input before any detach (D-11). ---
    let base = base_params(estimator)?;
    // F17: honour the estimator's `cat_features` so a categorical pool reaches
    // F14's fail-fast guard instead of being silently ingested float-only.
    let cats = crate::estimator::resolve_cat_features(&base, py, None)?;
    let pool = data_to_pool(py, x, y, Some(&cats))?;
    let (builders, grid_points) = expand_param_grid(&base, param_grid, py)?;
    let error_score = parse_error_score(error_score)?;

    let loss_function: Option<String> = base
        .get("loss_function")
        .and_then(|v| v.bind(py).extract::<String>().ok());
    let metric_names = resolve_metrics(metrics, loss_function.as_deref())?;

    // --- owned data only: release the GIL for the per-candidate cv compute. ---
    // `refit=false` in the facade: we refit the estimator IN PLACE below (§9 Q3).
    let res = py
        .detach(|| {
            let metric_refs: Vec<&str> = metric_names.iter().map(String::as_str).collect();
            catboost_rs::grid_search(
                &pool,
                &builders,
                &metric_refs,
                cv,
                shuffle,
                partition_random_seed,
                inverted,
                folds.as_deref(),
                false,
                error_score,
            )
        })
        .map_err(PyCbError)?;

    // sklearn parity: WARN (never raise) when candidates were assigned
    // `error_score`; must run with the GIL held (after detach returns).
    if !res.failures.is_empty() {
        warn_fit_failed(py, &res.failures, builders.len())?;
    }

    if refit {
        refit_estimator(py, estimator, &grid_points, res.best_index, x, y)?;
    }
    result_to_pydict(py, &res, &grid_points)
}

/// `catboost_rs.randomized_search(estimator, param_distributions, X, y=None,
/// cv=3, n_iter=10, partition_random_seed=0, shuffle=True, inverted=False,
/// folds=None, metrics=None, refit=True)` — as [`grid_search`], but the facade
/// deterministically subsamples `min(n_iter, product)` candidates (the project's
/// own `TFastRng64`, seeded by `partition_random_seed`) before selecting.
///
/// # Errors
/// As [`grid_search`], plus the facade's typed error when `n_iter == 0`. Never
/// panics.
#[pyfunction]
#[pyo3(signature = (
    estimator,
    param_distributions,
    x,
    y = None,
    cv = 3,
    n_iter = 10,
    partition_random_seed = 0,
    shuffle = true,
    inverted = false,
    folds = None,
    metrics = None,
    refit = true,
    error_score = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn randomized_search(
    py: Python<'_>,
    estimator: &Bound<'_, PyAny>,
    param_distributions: &Bound<'_, PyDict>,
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
    cv: usize,
    n_iter: usize,
    partition_random_seed: u64,
    shuffle: bool,
    inverted: bool,
    folds: Option<Vec<Vec<usize>>>,
    metrics: Option<&Bound<'_, PyAny>>,
    refit: bool,
    error_score: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    // --- GIL HELD: own every Python-borrowed input before any detach (D-11). ---
    let base = base_params(estimator)?;
    // F17: honour the estimator's `cat_features` so a categorical pool reaches
    // F14's fail-fast guard instead of being silently ingested float-only.
    let cats = crate::estimator::resolve_cat_features(&base, py, None)?;
    let pool = data_to_pool(py, x, y, Some(&cats))?;
    let (builders, grid_points) = expand_param_grid(&base, param_distributions, py)?;
    let error_score = parse_error_score(error_score)?;

    let loss_function: Option<String> = base
        .get("loss_function")
        .and_then(|v| v.bind(py).extract::<String>().ok());
    let metric_names = resolve_metrics(metrics, loss_function.as_deref())?;

    // --- owned data only: release the GIL for the per-candidate cv compute. ---
    let res = py
        .detach(|| {
            let metric_refs: Vec<&str> = metric_names.iter().map(String::as_str).collect();
            catboost_rs::randomized_search(
                &pool,
                &builders,
                &metric_refs,
                n_iter,
                cv,
                shuffle,
                partition_random_seed,
                inverted,
                folds.as_deref(),
                false,
                error_score,
            )
        })
        .map_err(PyCbError)?;

    // sklearn parity: WARN (never raise) when candidates were assigned
    // `error_score`; the number of fits attempted is the sampled subset size.
    if !res.failures.is_empty() {
        warn_fit_failed(py, &res.failures, n_iter.min(builders.len()))?;
    }

    if refit {
        refit_estimator(py, estimator, &grid_points, res.best_index, x, y)?;
    }
    result_to_pydict(py, &res, &grid_points)
}
