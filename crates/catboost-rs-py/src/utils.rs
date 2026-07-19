//! `catboost_rs.utils` — the standalone metric surface (ORCH-04-S6), mirroring
//! upstream `catboost.utils.eval_metric`.
//!
//! `eval_metric(label, approx, metric, weight=None, group_id=None)` returns a
//! `float` for a single metric string and a `list[float]` for a list of metric
//! strings, delegating to the published facade
//! [`catboost_rs::eval_metric`] / [`catboost_rs::eval_metrics`]. Facade errors
//! map through the [`crate::errors::PyCbError`] chokepoint (a bad metric string
//! raises `catboost_rs.CatBoostError`, never a panic).
//!
//! Own-before-detach (D-11): every Python buffer is copied into a Rust-owned
//! `Vec` up front, so no borrow into a live Python buffer survives.

use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList, PyModule};

use crate::errors::{CatBoostValueError, PyCbError};

/// Copy a 1-D numeric Python sequence / NumPy array into an owned `Vec<f64>`.
fn extract_f64_seq(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        CatBoostValueError::new_err(format!(
            "`{name}` must be a 1-D sequence / NumPy array of numbers"
        ))
    })
}

/// Copy a 1-D numeric Python sequence into an owned `Vec<f64>` and require its
/// length to equal `expected` (the `label`/`approx` object count). A supplied
/// array whose length disagrees — including an explicitly-passed empty array — is
/// a value error, matching upstream `catboost.utils.eval_metric` (a 0-length
/// `weight` is a length mismatch, NOT "uniform weights"); only an OMITTED
/// argument (`None`) means uniform / single group.
fn extract_len_checked_f64(
    obj: &Bound<'_, PyAny>,
    name: &str,
    expected: usize,
) -> PyResult<Vec<f64>> {
    let v = extract_f64_seq(obj, name)?;
    if v.len() != expected {
        return Err(CatBoostValueError::new_err(format!(
            "`{name}` has length {} but must match the `label`/`approx` object count {expected}",
            v.len()
        )));
    }
    Ok(v)
}

/// Copy a 1-D group-id sequence into an owned `Vec<u64>`, validated and length-
/// checked against `expected`.
///
/// Integer inputs (Python `int` lists, NumPy integer arrays) extract LOSSLESSLY
/// to `u64` — preserving ids above `2^53` and rejecting negatives (which cannot
/// be a `u64`). Only when that fails (a float array) do we fall back to an f64
/// pass that accepts non-negative WHOLE numbers and rejects negative / NaN /
/// non-integral values with a typed error — never the previous silent
/// `g as u64` saturating cast, which collapsed distinct query groups.
fn extract_group_ids(
    obj: &Bound<'_, PyAny>,
    name: &str,
    expected: usize,
) -> PyResult<Vec<u64>> {
    let ids = match obj.extract::<Vec<u64>>() {
        Ok(u) => u,
        Err(_) => {
            let v = extract_f64_seq(obj, name)?;
            let mut out = Vec::with_capacity(v.len());
            for &g in &v {
                if !g.is_finite() || g < 0.0 || g.fract() != 0.0 {
                    return Err(CatBoostValueError::new_err(format!(
                        "`{name}` must contain non-negative integer group ids; got `{g}`"
                    )));
                }
                out.push(g as u64);
            }
            out
        }
    };
    if ids.len() != expected {
        return Err(CatBoostValueError::new_err(format!(
            "`{name}` has length {} but must match the `label`/`approx` object count {expected}",
            ids.len()
        )));
    }
    Ok(ids)
}

/// `catboost_rs.utils.eval_metric(label, approx, metric, weight=None,
/// group_id=None)`.
///
/// A `str` `metric` returns a `float`; a list of `str` returns a `list[float]`.
///
/// # Errors
/// `catboost_rs.CatBoostValueError` on a malformed input array, a `label`/`approx`/
/// `weight`/`group_id` length mismatch, a non-integer / negative `group_id`, or a
/// `metric` that is neither a `str` nor a list of `str`. `catboost_rs.CatBoostError`
/// (mapped from the facade) on an unknown metric, a bad param, a degenerate eval
/// set (e.g. non-positive total weight), or a non-contiguous `group_id`.
#[pyfunction]
#[pyo3(signature = (label, approx, metric, weight=None, group_id=None))]
fn eval_metric(
    py: Python<'_>,
    label: &Bound<'_, PyAny>,
    approx: &Bound<'_, PyAny>,
    metric: &Bound<'_, PyAny>,
    weight: Option<&Bound<'_, PyAny>>,
    group_id: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Own-before-detach: materialize all Python buffers into Rust-owned Vecs.
    let label = extract_f64_seq(label, "label")?;
    let approx = extract_len_checked_f64(approx, "approx", label.len())?;
    // A supplied `weight`/`group_id` is validated and length-checked here (a value
    // error at the boundary); only an OMITTED (`None`) argument means uniform
    // weights / single group. This keeps an explicitly-empty array from being
    // silently coerced to `None`, and surfaces malformed input as the documented
    // `CatBoostValueError` rather than the seam's base `CatBoostError`.
    let weight = match weight {
        Some(w) => Some(extract_len_checked_f64(w, "weight", label.len())?),
        None => None,
    };
    let group_id = match group_id {
        Some(g) => Some(extract_group_ids(g, "group_id", label.len())?),
        None => None,
    };
    let weight_opt = weight.as_deref();
    let group_opt = group_id.as_deref();

    // A `str` metric -> scalar float; a list of `str` -> list[float].
    if let Ok(name) = metric.extract::<String>() {
        let value = catboost_rs::eval_metric(&label, &approx, &name, weight_opt, group_opt)
            .map_err(PyCbError)?;
        Ok(PyFloat::new(py, value).into_any().unbind())
    } else if let Ok(names) = metric.extract::<Vec<String>>() {
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let values = catboost_rs::eval_metrics(&label, &approx, &refs, weight_opt, group_opt)
            .map_err(PyCbError)?;
        Ok(PyList::new(py, values)?.into_any().unbind())
    } else {
        Err(CatBoostValueError::new_err(
            "`metric` must be a str or a list of str",
        ))
    }
}

/// Build and register the `catboost_rs.utils` submodule on `parent`, and insert
/// it into `sys.modules["catboost_rs.utils"]` so `import catboost_rs.utils` and
/// `from catboost_rs.utils import eval_metric` both work.
///
/// # Errors
/// Propagates any failure creating the submodule, adding the function, or
/// updating `sys.modules`.
pub(crate) fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let utils = PyModule::new(py, "utils")?;
    utils.add_function(wrap_pyfunction!(eval_metric, &utils)?)?;
    parent.add_submodule(&utils)?;
    // A submodule added via `add_submodule` is NOT automatically importable as
    // `catboost_rs.utils`; register it in sys.modules so the dotted import works.
    py.import("sys")?
        .getattr("modules")?
        .set_item("catboost_rs.utils", &utils)?;
    Ok(())
}
