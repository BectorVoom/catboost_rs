//! Eval-set validation metrics (TRAIN-07) — per-iteration `eval_metric`
//! computation over one or more held-out eval sets, with per-eval-set
//! per-iteration logging. This module FORMALIZES the minimal inline eval-set
//! loss the Plan 05 boosting stub carried (the stop-decision loss): it adds the
//! `eval_metric` abstraction (defaulting to the objective), a weighted
//! formulation, and a multi-eval-set history structure, then feeds the PRIMARY
//! eval set's metric to the overfitting detector.
//!
//! # Source of truth
//!
//! The per-iteration metric reported by upstream catboost 1.2.10 for an eval set
//! (`model.get_evals_result()[validation_k][eval_metric]`):
//!
//! - **RMSE** (`catboost/libs/metrics/metric.cpp`, `TRMSEMetric`):
//!   `sqrt(sum_w (pred - target)^2 / sum_w)` — the weighted root-mean-square
//!   error over the eval set's raw approximants.
//! - **Logloss** (`TLoglossMetric`): the weighted cross-entropy
//!   `sum_w -(y*ln(p) + (1-y)*ln(1-p)) / sum_w`, with `p = sigmoid(approx)` over
//!   the raw logit (Pitfall 6 — the approx is `RawFormulaVal`, sigmoid applied
//!   exactly once).
//!
//! `eval_metric` DEFAULTS to the objective when unset (RMSE for an RMSE
//! objective, Logloss for a Logloss objective) — [`EvalMetric::for_loss`].
//!
//! # Parity discipline
//!
//! EVERY fold — the squared-error numerator, the cross-entropy numerator, and
//! the weight denominator — routes through `cb_core::sum_f64` in canonical object
//! order (D-05/D-08). A degenerate eval set (empty, or total weight `<= 0`)
//! surfaces as [`CbError::Degenerate`]; there is no div-by-zero and no panic
//! (T-03-06-01, deny-lints). No raw float summation exists in this module — every
//! fold is the sanctioned `cb_core::sum_f64`.

use cb_compute::{sigmoid, Loss};
use cb_core::{sum_f64, CbError, CbResult};

use crate::ranking_metrics::{
    dcg_group, err_group, map_at_group, mrr_group, ndcg_group, pfound_group, precision_at_group,
    query_auc_group, recall_at_group, AucType, DcgDenominator, DcgMetricType,
};

/// The validation metric reported per iteration per eval set (`eval_metric`).
///
/// `eval_metric` defaults to the objective ([`EvalMetric::for_loss`]); it can be
/// overridden explicitly. This phase covers the two metrics matching the two
/// objectives — RMSE and Logloss.
///
/// The ranking metrics (NDCG/DCG/MAP/MRR/ERR/PFound/PrecisionAt/RecallAt/
/// QueryAUC) carry their upstream params (`top`/`border`/`decay`/type) via the
/// established variant-with-params pattern and are evaluated through
/// [`EvalMetric::eval_grouped`] (group seam, D-6.3-05); they are eval-only (no
/// derivative). `Eq` is NOT derived because the ranking variants carry `f64`
/// params (`border`/`decay`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvalMetric {
    /// Root-mean-square error: `sqrt(sum_w (pred-target)^2 / sum_w)`.
    Rmse,
    /// Weighted cross-entropy over `p = sigmoid(approx)`.
    Logloss,
    /// Mean squared logarithmic error (metric-ONLY, D-6.1-06): `sum_w (log(1+
    /// approx) - log(1+target))^2 / sum_w`. MSLE is NOT a trainable objective
    /// upstream (`enum_helpers.cpp:200,533-549`) — it has no `Loss` variant and is
    /// selected only via an explicit `eval_metric`, never as an objective default
    /// (`EvalMetric::for_loss` has no MSLE arm). The approx is RAW (`isExpApprox`
    /// asserted false upstream, `metric.cpp:1912`).
    Msle,
    /// Normalized DCG per group, averaged over groups (`metric.cpp:3079`).
    /// `top=-1` → full group; upstream defaults `type=Base`,
    /// `denominator=LogPosition`, `normalized=true`. IDCG==0 → 1.
    Ndcg {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
        /// Numerator gain type (`Base` default).
        dcg_type: DcgMetricType,
        /// Position-discount denominator (`LogPosition` default).
        denominator: DcgDenominator,
    },
    /// DCG per group, averaged over groups (`metric.cpp:3079`, `normalized=false`).
    Dcg {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
        /// Numerator gain type (`Base` default).
        dcg_type: DcgMetricType,
        /// Position-discount denominator (`LogPosition` default).
        denominator: DcgDenominator,
    },
    /// Mean Average Precision @k per group (`metric.cpp:4564`). `border` default
    /// `0.5` (`GetDefaultTargetBorder`).
    Map {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
        /// Relevance binarization threshold (`target > border`).
        border: f64,
    },
    /// Mean Reciprocal Rank per group (`metric.cpp:6062`).
    Mrr {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
        /// Relevance binarization threshold (`target > border`).
        border: f64,
    },
    /// Expected Reciprocal Rank per group (`metric.cpp:6166`).
    Err {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
    },
    /// PFound cascade per group (`metric.cpp:2918`). `decay` default `0.85`.
    PFound {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
        /// Cascade decay factor (default `0.85`).
        decay: f64,
    },
    /// Precision @k per group (`metric.cpp:4369`). `border` default `0.5`.
    PrecisionAt {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
        /// Relevance binarization threshold (`target > border`).
        border: f64,
    },
    /// Recall @k per group (`metric.cpp:4466`). `border` default `0.5`.
    RecallAt {
        /// `top` size; `-1` (default) uses the full group.
        top: i64,
        /// Relevance binarization threshold (`target > border`).
        border: f64,
    },
    /// Per-group AUC averaged over groups (`metric.cpp:5606`). Singleclass
    /// `Classic` (binary-class AUC) default; `Ranking` for graded relevance.
    QueryAuc {
        /// Singleclass AUC type (`Classic` default).
        auc_type: AucType,
    },
}

impl EvalMetric {
    /// The default `eval_metric` for an objective (`eval_metric` unset): RMSE for
    /// the RMSE/MAE family, Logloss for the Logloss objective.
    ///
    /// Mirrors upstream's `eval_metric == objective` default. MAE maps to RMSE
    /// here only as a numeric eval surface — the Phase-3 eval-metric set is the
    /// two losses this phase locks; MAE training uses RMSE eval reporting until a
    /// later phase adds the MAE metric.
    #[must_use]
    pub fn for_loss(loss: &Loss) -> Self {
        match *loss {
            // The Wave-1 smooth regression losses (LogCosh / Lq / Huber /
            // Expectile) each default upstream to their own-named regression
            // metric; for Wave 1 they report the parity-neutral RMSE eval surface
            // (mirroring the MAE arm) unless a fixture pins `eval_metric`.
            // The Wave-2 positive-domain / link losses (Poisson / Tweedie / MAPE)
            // each default upstream to their own-named regression metric; for
            // Wave 2 they report the parity-neutral RMSE eval surface (mirroring
            // the MAE arm) unless a fixture pins `eval_metric`. MSLE is NOT mapped
            // here — it is metric-only (D-6.1-06) and selected explicitly, never a
            // default for any objective.
            // The multiclass losses (MultiClass / MultiClassOneVsAll) report the
            // parity-neutral RMSE eval surface by default; the in-scope fixtures pin
            // `od_type=None` with no eval set, so this default is never exercised
            // (a named multiclass eval metric is a later phase).
            Loss::Rmse
            | Loss::Mae
            | Loss::Quantile { .. }
            | Loss::LogCosh
            | Loss::Lq { .. }
            | Loss::Huber { .. }
            | Loss::Expectile { .. }
            | Loss::Poisson
            | Loss::Tweedie { .. }
            | Loss::Mape
            | Loss::MultiClass
            | Loss::MultiClassOneVsAll
            | Loss::MultiLogloss
            | Loss::MultiCrossEntropy
            // MultiQuantile (Wave 3) reports the parity-neutral RMSE eval surface
            // by default; the in-scope fixture pins no eval set, so this default is
            // never exercised (a named MultiQuantile eval metric is a later phase).
            | Loss::MultiQuantile { .. }
            // RMSEWithUncertainty (Wave B, LOSS-08) reports the parity-neutral RMSE
            // eval surface by default; the in-scope fixture pins `od_type=None` with
            // no eval set, so this default is never exercised (a named uncertainty
            // eval metric is out of scope for this loss oracle).
            | Loss::RmseWithUncertainty
            // The Wave-A ranking losses (QueryRMSE / QuerySoftMax) report the
            // parity-neutral RMSE eval surface by default; the in-scope ranking
            // fixtures pin `od_type=None` with no eval set, so this default is
            // never exercised (a named ranking eval metric — NDCG etc. — is Wave D).
            | Loss::QueryRmse
            | Loss::QuerySoftMax { .. }
            // The Wave-B ranking losses (PairLogit / PairLogitPairwise / LambdaMart)
            // likewise report the parity-neutral RMSE eval surface by default; the
            // in-scope fixtures pin `od_type=None` with no eval set, so this default
            // is never exercised (named ranking eval metrics are Wave D).
            | Loss::PairLogit
            | Loss::PairLogitPairwise
            | Loss::LambdaMart { .. }
            // The Wave-C randomized ranking losses (YetiRank / YetiRankPairwise /
            // StochasticRank) likewise report the parity-neutral RMSE eval surface
            // by default; the in-scope fixtures pin `od_type=None` with no eval set,
            // so this default is never exercised (named ranking eval metrics are
            // Wave D).
            | Loss::YetiRank { .. }
            | Loss::YetiRankPairwise { .. }
            | Loss::StochasticRank { .. } => Self::Rmse,
            // The binary-classification family (Logloss / CrossEntropy / Focal)
            // reports the Logloss eval metric by default.
            Loss::Logloss | Loss::CrossEntropy | Loss::Focal { .. } => Self::Logloss,
        }
    }

    /// Compute this eval metric over an eval set's raw approximants.
    ///
    /// `approx[i]` is object `i`'s running raw approximant (bias + Σ tree
    /// contributions; the raw logit for Logloss). `target[i]` its label.
    /// `weights`, when non-empty, are per-object; an empty slice means uniform
    /// weight `1.0`. All folds route through `cb_core::sum_f64` (D-08).
    ///
    /// # Errors
    /// - [`CbError::Degenerate`] if `approx`/`target` lengths disagree, the eval
    ///   set is empty, or the total weight is `<= 0` (T-03-06-01 — no div-by-zero).
    pub fn eval(self, approx: &[f64], target: &[f64], weights: &[f64]) -> CbResult<f64> {
        if approx.len() != target.len() {
            return Err(CbError::Degenerate(
                "eval metric: approx/target length mismatch".to_owned(),
            ));
        }
        if approx.is_empty() {
            return Err(CbError::Degenerate("eval metric: empty eval set".to_owned()));
        }
        if !weights.is_empty() && weights.len() != approx.len() {
            return Err(CbError::Degenerate(
                "eval metric: weights length mismatch".to_owned(),
            ));
        }

        // The weight column the denominator and the weighted numerator share:
        // uniform 1.0 when no weights are supplied.
        let weight_at = |i: usize| -> f64 {
            if weights.is_empty() {
                1.0
            } else {
                weights.get(i).copied().unwrap_or(0.0)
            }
        };

        let weight_col: Vec<f64> = (0..approx.len()).map(weight_at).collect();
        let total_weight = sum_f64(&weight_col);
        // Guard the division: a non-finite (NaN/Inf) or non-positive total weight
        // is degenerate (T-03-06-01 — no div-by-zero, no NaN leaking to the gate).
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(CbError::Degenerate(
                "eval metric: non-positive total weight".to_owned(),
            ));
        }

        match self {
            Self::Rmse => {
                // sqrt(sum_w (pred-target)^2 / sum_w).
                let weighted_sq: Vec<f64> = approx
                    .iter()
                    .zip(target.iter())
                    .enumerate()
                    .map(|(i, (&a, &t))| {
                        let d = a - t;
                        weight_at(i) * d * d
                    })
                    .collect();
                Ok((sum_f64(&weighted_sq) / total_weight).sqrt())
            }
            Self::Logloss => {
                // sum_w -(y*ln p + (1-y)*ln(1-p)) / sum_w, p = sigmoid(approx).
                // p is clamped away from {0,1} so a saturated logit cannot produce
                // -inf (matches the inline stub it replaces; T-03-06-01).
                let weighted_ce: Vec<f64> = approx
                    .iter()
                    .zip(target.iter())
                    .enumerate()
                    .map(|(i, (&a, &y))| {
                        let p = sigmoid(a).clamp(1e-15, 1.0 - 1e-15);
                        weight_at(i) * -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                    })
                    .collect();
                Ok(sum_f64(&weighted_ce) / total_weight)
            }
            Self::Msle => {
                // sum_w (log(1+approx) - log(1+target))^2 / sum_w, approx RAW
                // (`metric.cpp:1899-1926`: error.Stats[0] += Sqr(log(1+approx) -
                // log(1+target))*w; GetFinalError = Stats[0]/(Stats[1]+1e-38)).
                // NOT sqrt'd — MSLE is the MEAN squared log error (cf. RMSLE).
                // Log-domain guard (T-06.1.02-03): `1+approx` / `1+target` must be
                // strictly positive, else the `ln` is a domain violation (NaN);
                // surface a typed CbError rather than leaking a NaN to the gate
                // (mirrors the Logloss clamp discipline; never `unwrap`/panic).
                let mut weighted_sq: Vec<f64> = Vec::with_capacity(approx.len());
                for (i, (&a, &t)) in approx.iter().zip(target.iter()).enumerate() {
                    let one_plus_a = 1.0 + a;
                    let one_plus_t = 1.0 + t;
                    if !(one_plus_a > 0.0) || !(one_plus_t > 0.0) {
                        return Err(CbError::Degenerate(format!(
                            "MSLE log-domain violation at object {i}: 1+approx={one_plus_a}, \
                             1+target={one_plus_t} (both must be > 0)"
                        )));
                    }
                    let d = one_plus_a.ln() - one_plus_t.ln();
                    weighted_sq.push(weight_at(i) * d * d);
                }
                Ok(sum_f64(&weighted_sq) / total_weight)
            }
            // The ranking metrics are per-query-group quantities — they are
            // evaluated through `eval_grouped` (D-6.3-05), never the flat path.
            // Selecting one here is a misuse, surfaced as a typed error (never a
            // panic). The flat path above stays byte-identical (D-04).
            Self::Ndcg { .. }
            | Self::Dcg { .. }
            | Self::Map { .. }
            | Self::Mrr { .. }
            | Self::Err { .. }
            | Self::PFound { .. }
            | Self::PrecisionAt { .. }
            | Self::RecallAt { .. }
            | Self::QueryAuc { .. } => Err(CbError::Degenerate(
                "ranking eval metric requires the grouped seam (use eval_grouped)".to_owned(),
            )),
        }
    }

    /// Compute a ranking eval metric over a grouped eval set (D-6.3-05).
    ///
    /// `approx[i]`/`target[i]` are object `i`'s raw approximant and graded
    /// relevance label; `weights` is empty (uniform `1.0`) or per-object.
    /// `group_id` carries the per-object query-group id (contiguous, unique runs —
    /// mirrors upstream `GroupSamples`); when empty the whole eval set is treated
    /// as one group. `subgroup_id` is accepted for API symmetry (only PFound uses
    /// it upstream to dedup positions; the in-scope fixtures do not set it).
    ///
    /// Each metric is computed per group and averaged over groups. The weighting
    /// matches upstream exactly: DCG/NDCG/PFound/ERR/MRR/QueryAUC weight each group
    /// by its group weight (mean of member weights, else `1.0`); MAP/PrecisionAt/
    /// RecallAt weight every group uniformly (`Stats[1]++`,
    /// `metric.cpp:4390/4487/4586`). Every per-group and cross-group reduction
    /// routes through `cb_core::sum_f64` (group asc — D-08).
    ///
    /// The non-ranking metrics (RMSE/Logloss/MSLE) are rejected here with a typed
    /// error — they use the flat [`EvalMetric::eval`] path (kept byte-identical,
    /// D-04).
    ///
    /// # Errors
    /// - [`CbError::Degenerate`] if `approx`/`target`/`group_id`/`weights` lengths
    ///   disagree, the eval set is empty, `group_id` is non-contiguous, or a
    ///   non-ranking metric is selected.
    pub fn eval_grouped(
        self,
        approx: &[f64],
        target: &[f64],
        weights: &[f64],
        group_id: &[u64],
        subgroup_id: &[u64],
    ) -> CbResult<f64> {
        let _ = subgroup_id; // accepted for API symmetry; unused by the in-scope metrics.
        if approx.len() != target.len() {
            return Err(CbError::Degenerate(
                "ranking eval: approx/target length mismatch".to_owned(),
            ));
        }
        if approx.is_empty() {
            return Err(CbError::Degenerate("ranking eval: empty eval set".to_owned()));
        }
        if !weights.is_empty() && weights.len() != approx.len() {
            return Err(CbError::Degenerate(
                "ranking eval: weights length mismatch".to_owned(),
            ));
        }
        if !group_id.is_empty() && group_id.len() != approx.len() {
            return Err(CbError::Degenerate(
                "ranking eval: group_id length mismatch".to_owned(),
            ));
        }

        // (1) Detect contiguous group runs as half-open [begin, end) spans
        //     (mirrors `build_query_info` / upstream `GroupSamples`). An empty
        //     group_id is one big group.
        let spans = group_spans(group_id, approx.len())?;

        // (2) Per-group weight (mean of member weights, else 1.0) — reduced
        //     through cb_core::sum_f64 (D-08).
        let weight_at = |i: usize| -> f64 {
            if weights.is_empty() {
                1.0
            } else {
                weights.get(i).copied().unwrap_or(1.0)
            }
        };

        // (3) Whether this metric weights groups by group weight (DCG family,
        //     PFound, ERR, MRR, QueryAUC) or uniformly (MAP/Precision/Recall —
        //     `Stats[1]++`).
        let use_group_weight = matches!(
            self,
            Self::Ndcg { .. }
                | Self::Dcg { .. }
                | Self::PFound { .. }
                | Self::Err { .. }
                | Self::Mrr { .. }
                | Self::QueryAuc { .. }
        );

        let mut numerators: Vec<f64> = Vec::with_capacity(spans.len());
        let mut denominators: Vec<f64> = Vec::with_capacity(spans.len());
        for &(begin, end) in &spans {
            let a = approx.get(begin..end).ok_or_else(|| {
                CbError::Degenerate("ranking eval: group span out of range".to_owned())
            })?;
            let t = target.get(begin..end).ok_or_else(|| {
                CbError::Degenerate("ranking eval: group span out of range".to_owned())
            })?;
            if a.is_empty() {
                continue; // empty group contributes nothing (never divides).
            }
            let group_weight = if use_group_weight {
                let members: Vec<f64> = (begin..end).map(weight_at).collect();
                sum_f64(&members) / (members.len() as f64)
            } else {
                1.0
            };
            let value = self.eval_one_group(a, t)?;
            numerators.push(group_weight * value);
            denominators.push(group_weight);
        }

        let total = sum_f64(&denominators);
        if !total.is_finite() || total <= 0.0 {
            // No groups / zero total weight: upstream `GetFinalError` returns 0
            // (PFound/DCG/NDCG/ERR/MRR/MAP) or 1 (Precision/Recall). Mirror the
            // per-metric default rather than dividing.
            return Ok(self.empty_metric_default());
        }
        Ok(sum_f64(&numerators) / total)
    }

    /// The per-group metric value for ONE group's approx/target slice (no
    /// weighting — the caller applies the group weight).
    fn eval_one_group(self, approx: &[f64], target: &[f64]) -> CbResult<f64> {
        Ok(match self {
            Self::Ndcg {
                top,
                dcg_type,
                denominator,
            } => ndcg_group(approx, target, top, dcg_type, denominator),
            Self::Dcg {
                top,
                dcg_type,
                denominator,
            } => dcg_group(approx, target, top, dcg_type, denominator),
            Self::Map { top, border } => map_at_group(approx, target, top, border),
            Self::Mrr { top, border } => mrr_group(approx, target, top, border),
            Self::Err { top } => err_group(approx, target, top),
            Self::PFound { top, decay } => pfound_group(approx, target, top, decay),
            Self::PrecisionAt { top, border } => precision_at_group(approx, target, top, border),
            Self::RecallAt { top, border } => recall_at_group(approx, target, top, border),
            Self::QueryAuc { auc_type } => query_auc_group(approx, target, auc_type),
            Self::Rmse | Self::Logloss | Self::Msle => {
                return Err(CbError::Degenerate(
                    "non-ranking metric passed to the grouped seam (use eval)".to_owned(),
                ))
            }
        })
    }

    /// The empty-eval-set default per metric (`GetFinalError` with `Stats[1]==0`):
    /// Precision/Recall return `1`, every other ranking metric returns `0`.
    fn empty_metric_default(self) -> f64 {
        match self {
            Self::PrecisionAt { .. } | Self::RecallAt { .. } => 1.0,
            _ => 0.0,
        }
    }
}

/// Detect contiguous group runs as half-open `[begin, end)` spans (mirrors
/// `build_query_info` / upstream `GroupSamples`, `query.h:48-67`): each run is one
/// group; a group id reappearing after a different id intervened is rejected
/// (typed `CbError::Degenerate`, never a panic). An empty `group_id` yields a
/// single span covering all `n_rows` objects.
fn group_spans(group_id: &[u64], n_rows: usize) -> CbResult<Vec<(usize, usize)>> {
    if group_id.is_empty() {
        return Ok(if n_rows == 0 {
            Vec::new()
        } else {
            vec![(0, n_rows)]
        });
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut seen: Vec<u64> = Vec::new();
    let mut i = 0usize;
    while i < group_id.len() {
        let current = group_id.get(i).copied().unwrap_or_default();
        let begin = i;
        i += 1;
        while i < group_id.len() && group_id.get(i).copied() == Some(current) {
            i += 1;
        }
        runs.push((begin, i));
        seen.push(current);
    }
    seen.sort_unstable();
    if seen.windows(2).any(|w| w.first() == w.get(1)) {
        return Err(CbError::Degenerate(
            "ranking eval: group_id is not contiguous (queryIds should be grouped)".to_owned(),
        ));
    }
    Ok(runs)
}

/// Per-eval-set per-iteration metric history (TRAIN-07 logging).
///
/// `per_set[k]` is eval set `k`'s ordered per-iteration `eval_metric` values
/// (one entry per boosting iteration up to the stop point). The PRIMARY eval set
/// is index `0` — its curve is what the overfitting detector consumes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalMetricHistory {
    /// `per_set[k]` is eval set `k`'s per-iteration metric values, in iteration
    /// order.
    pub per_set: Vec<Vec<f64>>,
}

impl EvalMetricHistory {
    /// Create an empty history sized for `n_sets` eval sets.
    #[must_use]
    pub fn new(n_sets: usize) -> Self {
        Self {
            per_set: vec![Vec::new(); n_sets],
        }
    }

    /// Number of eval sets tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.per_set.len()
    }

    /// Whether no eval sets are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.per_set.is_empty()
    }

    /// Append eval set `set_idx`'s metric value for the current iteration. A
    /// `set_idx` beyond the tracked sets is ignored (defensive; the trainer only
    /// pushes valid indices).
    pub fn push(&mut self, set_idx: usize, value: f64) {
        if let Some(curve) = self.per_set.get_mut(set_idx) {
            curve.push(value);
        }
    }

    /// The PRIMARY eval set's per-iteration curve (index `0`) — the curve the
    /// overfitting detector consumes. Empty when no eval sets are tracked.
    #[must_use]
    pub fn primary(&self) -> &[f64] {
        self.per_set.first().map_or(&[], Vec::as_slice)
    }
}

#[cfg(test)]
#[path = "metrics_test.rs"]
mod tests;
