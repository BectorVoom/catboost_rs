//! Standalone `eval_metric` / `calc_metric` surface (ORCH-04).
//!
//! This module adds the missing *standalone* metric surface: compute a CatBoost
//! metric's FINAL value on caller-supplied `(label, approx, weight, group_id)`,
//! mirroring upstream `catboost.utils.eval_metric`. It is pure routing over the
//! already-oracle-locked arithmetic in [`crate::EvalMetric`] — it adds NO new
//! float summation (D-08) and does NOT modify the reused seams (D-04): flat
//! metrics delegate to [`EvalMetric::eval`], ranking metrics to
//! [`EvalMetric::eval_grouped`].
//!
//! Every error path returns a typed [`cb_core::CbError`]; there is no
//! `unwrap`/`expect`/`panic`/indexing (clippy deny-lints).

use cb_core::{CbError, CbResult};

use crate::ranking_metrics::{AucType, DcgDenominator, DcgMetricType};
use crate::EvalMetric;

/// The upstream metric names NOT yet present in [`EvalMetric`] (documented so the
/// parser's error message steers callers), plus the note that `Custom` is
/// program-constructed only and never string-parsed (SPEC §9 Q2).
const DEFERRED_METRICS: &str =
    "deferred (not yet in EvalMetric): AUC, Accuracy, F1, Precision, Recall, R2, MAE, MAPE; \
     `Custom` is program-constructed only (never parsed from a string)";

/// A parsed `key=value` param token list (lowercased keys, raw values).
struct Params<'a> {
    entries: Vec<(String, &'a str)>,
}

impl<'a> Params<'a> {
    /// Split the descriptor tail (`key=value` tokens) and reject a malformed
    /// token (missing `=`) or a duplicate key.
    fn parse(tail: impl Iterator<Item = &'a str>, metric: &str) -> CbResult<Self> {
        let mut entries: Vec<(String, &'a str)> = Vec::new();
        for tok in tail {
            let (key, value) = tok.split_once('=').ok_or_else(|| {
                CbError::Degenerate(format!(
                    "metric `{metric}`: malformed param token `{tok}` (expected key=value)"
                ))
            })?;
            let key = key.to_ascii_lowercase();
            if entries.iter().any(|(k, _)| *k == key) {
                return Err(CbError::Degenerate(format!(
                    "metric `{metric}`: duplicate param key `{key}`"
                )));
            }
            entries.push((key, value));
        }
        Ok(Self { entries })
    }

    /// Reject any param key not in `allowed` for this metric.
    fn reject_unknown(&self, metric: &str, allowed: &[&str]) -> CbResult<()> {
        for (k, _) in &self.entries {
            if !allowed.iter().any(|a| a == k) {
                return Err(CbError::Degenerate(format!(
                    "metric `{metric}`: unknown param key `{k}`; supported keys: {allowed:?}"
                )));
            }
        }
        Ok(())
    }

    fn find(&self, key: &str) -> Option<&'a str> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    fn i64_or(&self, key: &str, default: i64, metric: &str) -> CbResult<i64> {
        match self.find(key) {
            Some(v) => v.trim().parse::<i64>().map_err(|e| {
                CbError::Degenerate(format!(
                    "metric `{metric}`: param `{key}={v}` is not an integer: {e}"
                ))
            }),
            None => Ok(default),
        }
    }

    fn f64_or(&self, key: &str, default: f64, metric: &str) -> CbResult<f64> {
        match self.find(key) {
            Some(v) => v.trim().parse::<f64>().map_err(|e| {
                CbError::Degenerate(format!(
                    "metric `{metric}`: param `{key}={v}` is not a float: {e}"
                ))
            }),
            None => Ok(default),
        }
    }

    fn dcg_type_or_default(&self, metric: &str) -> CbResult<DcgMetricType> {
        match self.find("type") {
            None => Ok(DcgMetricType::Base),
            Some(v) => match v.trim().to_ascii_lowercase().as_str() {
                "base" => Ok(DcgMetricType::Base),
                "exp" => Ok(DcgMetricType::Exp),
                other => Err(CbError::Degenerate(format!(
                    "metric `{metric}`: param `type={other}` must be Base|Exp"
                ))),
            },
        }
    }

    fn denominator_or_default(&self, metric: &str) -> CbResult<DcgDenominator> {
        match self.find("denominator") {
            None => Ok(DcgDenominator::LogPosition),
            Some(v) => match v.trim().to_ascii_lowercase().as_str() {
                "logposition" => Ok(DcgDenominator::LogPosition),
                "position" => Ok(DcgDenominator::Position),
                other => Err(CbError::Degenerate(format!(
                    "metric `{metric}`: param `denominator={other}` must be LogPosition|Position"
                ))),
            },
        }
    }

    fn auc_type_or_default(&self, metric: &str) -> CbResult<AucType> {
        match self.find("type") {
            None => Ok(AucType::Classic),
            Some(v) => match v.trim().to_ascii_lowercase().as_str() {
                "classic" => Ok(AucType::Classic),
                "ranking" => Ok(AucType::Ranking),
                other => Err(CbError::Degenerate(format!(
                    "metric `{metric}`: param `type={other}` must be Classic|Ranking"
                ))),
            },
        }
    }
}

/// Parse a CatBoost metric-descriptor string into an [`EvalMetric`] (ORCH-04-S1).
///
/// Grammar (case-insensitive metric name; `:key=value` params in any order):
///   "RMSE" | "Logloss" | "MSLE"
///   "NDCG[:top=<i64>][:type=Base|Exp][:denominator=LogPosition|Position]"
///   "DCG[:top=<i64>][:type=Base|Exp][:denominator=LogPosition|Position]"
///   "MAP[:top=<i64>][:border=<f64>]"
///   "MRR[:top=<i64>][:border=<f64>]"
///   "ERR[:top=<i64>]"
///   "PFound[:top=<i64>][:decay=<f64>]"
///   "PrecisionAt[:top=<i64>][:border=<f64>]"
///   "RecallAt[:top=<i64>][:border=<f64>]"
///   "QueryAUC[:type=Classic|Ranking]"
/// Omitted params take the upstream defaults encoded in [`EvalMetric`]
/// (top=-1, border=0.5, decay=0.85, dcg_type=Base, denominator=LogPosition,
/// auc_type=Classic).
///
/// # Errors
/// [`CbError::Degenerate`] on an unknown metric name, an unknown/duplicate param
/// key, or an unparseable param value. Never panics (deny-lints).
pub fn parse_metric(descr: &str) -> CbResult<EvalMetric> {
    let mut parts = descr.split(':');
    let name = parts.next().unwrap_or("");
    let lower = name.trim().to_ascii_lowercase();
    let p = Params::parse(parts, name)?;

    let metric = match lower.as_str() {
        "rmse" => {
            p.reject_unknown(name, &[])?;
            EvalMetric::Rmse
        }
        "logloss" => {
            p.reject_unknown(name, &[])?;
            EvalMetric::Logloss
        }
        "msle" => {
            p.reject_unknown(name, &[])?;
            EvalMetric::Msle
        }
        "ndcg" => {
            p.reject_unknown(name, &["top", "type", "denominator"])?;
            EvalMetric::Ndcg {
                top: p.i64_or("top", -1, name)?,
                dcg_type: p.dcg_type_or_default(name)?,
                denominator: p.denominator_or_default(name)?,
            }
        }
        "dcg" => {
            p.reject_unknown(name, &["top", "type", "denominator"])?;
            EvalMetric::Dcg {
                top: p.i64_or("top", -1, name)?,
                dcg_type: p.dcg_type_or_default(name)?,
                denominator: p.denominator_or_default(name)?,
            }
        }
        "map" => {
            p.reject_unknown(name, &["top", "border"])?;
            EvalMetric::Map {
                top: p.i64_or("top", -1, name)?,
                border: p.f64_or("border", 0.5, name)?,
            }
        }
        "mrr" => {
            p.reject_unknown(name, &["top", "border"])?;
            EvalMetric::Mrr {
                top: p.i64_or("top", -1, name)?,
                border: p.f64_or("border", 0.5, name)?,
            }
        }
        "err" => {
            p.reject_unknown(name, &["top"])?;
            EvalMetric::Err {
                top: p.i64_or("top", -1, name)?,
            }
        }
        "pfound" => {
            p.reject_unknown(name, &["top", "decay"])?;
            EvalMetric::PFound {
                top: p.i64_or("top", -1, name)?,
                decay: p.f64_or("decay", 0.85, name)?,
            }
        }
        "precisionat" => {
            p.reject_unknown(name, &["top", "border"])?;
            EvalMetric::PrecisionAt {
                top: p.i64_or("top", -1, name)?,
                border: p.f64_or("border", 0.5, name)?,
            }
        }
        "recallat" => {
            p.reject_unknown(name, &["top", "border"])?;
            EvalMetric::RecallAt {
                top: p.i64_or("top", -1, name)?,
                border: p.f64_or("border", 0.5, name)?,
            }
        }
        "queryauc" => {
            p.reject_unknown(name, &["type"])?;
            EvalMetric::QueryAuc {
                auc_type: p.auc_type_or_default(name)?,
            }
        }
        other => {
            return Err(CbError::Degenerate(format!(
                "unknown metric name `{other}`; supported: RMSE, Logloss, MSLE, NDCG, DCG, MAP, \
                 MRR, ERR, PFound, PrecisionAt, RecallAt, QueryAUC; {DEFERRED_METRICS}"
            )));
        }
    };
    Ok(metric)
}

/// Whether `metric` is a per-query-group (ranking) metric — routed through
/// [`EvalMetric::eval_grouped`] — versus a flat metric routed through
/// [`EvalMetric::eval`]. The ranking set is exactly the arms the grouped seam
/// accepts (`eval_grouped` / `eval_one_group`), i.e. every variant `EvalMetric::eval`
/// rejects as non-flat — which INCLUDES `Map`/`PrecisionAt`/`RecallAt`. (Do not
/// confuse this with `metrics.rs`'s narrower `use_group_weight` set, which weights
/// groups and deliberately excludes those three.)
fn is_ranking(metric: &EvalMetric) -> bool {
    matches!(
        metric,
        EvalMetric::Ndcg { .. }
            | EvalMetric::Dcg { .. }
            | EvalMetric::Map { .. }
            | EvalMetric::Mrr { .. }
            | EvalMetric::Err { .. }
            | EvalMetric::PFound { .. }
            | EvalMetric::PrecisionAt { .. }
            | EvalMetric::RecallAt { .. }
            | EvalMetric::QueryAuc { .. }
    )
}

/// Compute one metric's FINAL value on fixed predictions (the standalone
/// `catboost.utils.eval_metric` surface, ORCH-04-S2/S3).
///
/// - Flat metrics (RMSE/Logloss/MSLE/Custom) route through [`EvalMetric::eval`];
///   `group_id` is ignored.
/// - Ranking metrics route through [`EvalMetric::eval_grouped`]; an empty
///   `group_id` is treated as one group, `subgroup_id` is passed empty.
///
/// `weight` empty => uniform 1.0. `approx[i]` is the RAW model output
/// (`RawFormulaVal`; Logloss applies the sigmoid internally). The arg order
/// mirrors upstream `(label, approx, ...)`; the reused seams take
/// `(approx, target, ...)`, so `label` maps to `target`.
///
/// # Errors
/// [`CbError::Degenerate`] on length mismatch, empty eval set, non-positive total
/// weight, or non-contiguous `group_id` (delegated to the underlying seam).
pub fn calc_metric(
    metric: &EvalMetric,
    label: &[f64],
    approx: &[f64],
    weight: &[f64],
    group_id: &[u64],
) -> CbResult<f64> {
    if is_ranking(metric) {
        metric.eval_grouped(approx, label, weight, group_id, &[])
    } else {
        metric.eval(approx, label, weight)
    }
}

/// Parse each metric descriptor and evaluate it on the fixed predictions,
/// returning one `f64` per descriptor in input order (ORCH-04-S4). This is the
/// multi-metric dispatch form of the standalone surface.
///
/// # Errors
/// The first parse error (unknown metric name / bad param) or evaluation error
/// (length mismatch, degenerate eval set, non-contiguous `group_id`) as a typed
/// [`CbError`], short-circuiting. Never panics.
pub fn eval_metric(
    label: &[f64],
    approx: &[f64],
    metrics: &[&str],
    weight: &[f64],
    group_id: &[u64],
) -> CbResult<Vec<f64>> {
    metrics
        .iter()
        .map(|descr| {
            let metric = parse_metric(descr)?;
            calc_metric(&metric, label, approx, weight, group_id)
        })
        .collect()
}

#[cfg(test)]
#[path = "calc_metrics_test.rs"]
mod tests;
