//! Standalone metric evaluation on the published facade (ORCH-04-S5).
//!
//! Free functions that mirror upstream `catboost.utils.eval_metric`: compute a
//! CatBoost metric's final value on caller-supplied fixed predictions, entirely
//! through the published crate. They are a thin, non-panicking wrapper over
//! [`cb_train::calc_metrics::eval_metric`] (`Option` weight/group -> empty
//! slice; `cb_core::CbError` -> [`CatBoostError`] via `?`).

use crate::error::CatBoostError;

/// Compute several CatBoost metrics on fixed predictions; returns one value per
/// metric string, in order (facade over
/// [`cb_train::calc_metrics::eval_metric`]).
///
/// `label`/`approx` are the fixed per-object label and RAW model output;
/// `weight`/`group_id` default to empty (uniform weight / a single group) when
/// `None`. Ranking metrics use `group_id`; flat metrics ignore it.
///
/// # Errors
/// [`CatBoostError::Train`] wrapping the underlying [`cb_core::CbError`] on an
/// unknown metric name, a bad param, a length mismatch, a degenerate eval set,
/// or a non-contiguous `group_id`. Never panics.
pub fn eval_metrics(
    label: &[f64],
    approx: &[f64],
    metrics: &[&str],
    weight: Option<&[f64]>,
    group_id: Option<&[u64]>,
) -> Result<Vec<f64>, CatBoostError> {
    let weight = weight.unwrap_or(&[]);
    let group_id = group_id.unwrap_or(&[]);
    let values = cb_train::calc_metrics::eval_metric(label, approx, metrics, weight, group_id)?;
    Ok(values)
}

/// Compute a single CatBoost metric on fixed predictions (facade over
/// [`eval_metrics`]). Mirrors `catboost.utils.eval_metric` for one metric string.
///
/// # Errors
/// [`CatBoostError::Train`] wrapping the underlying [`cb_core::CbError`] (see
/// [`eval_metrics`]). Never panics: the single value is extracted with a
/// non-indexing `Iterator::next`, and the (unreachable — `eval_metric` returns
/// exactly one value per metric) empty case maps to a typed error.
pub fn eval_metric(
    label: &[f64],
    approx: &[f64],
    metric: &str,
    weight: Option<&[f64]>,
    group_id: Option<&[u64]>,
) -> Result<f64, CatBoostError> {
    eval_metrics(label, approx, &[metric], weight, group_id)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            CatBoostError::Train(cb_core::CbError::Degenerate(
                "eval_metric produced no value".to_owned(),
            ))
        })
}
