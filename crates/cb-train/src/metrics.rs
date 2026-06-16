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

/// The validation metric reported per iteration per eval set (`eval_metric`).
///
/// `eval_metric` defaults to the objective ([`EvalMetric::for_loss`]); it can be
/// overridden explicitly. This phase covers the two metrics matching the two
/// objectives — RMSE and Logloss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub fn for_loss(loss: Loss) -> Self {
        match loss {
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
            Loss::Rmse
            | Loss::Mae
            | Loss::Quantile { .. }
            | Loss::LogCosh
            | Loss::Lq { .. }
            | Loss::Huber { .. }
            | Loss::Expectile { .. }
            | Loss::Poisson
            | Loss::Tweedie { .. }
            | Loss::Mape => Self::Rmse,
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
        }
    }
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
