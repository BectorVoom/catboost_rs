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

use cb_compute::{sigmoid, CustomMetricHandle, Loss};
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
///
/// `Copy` is intentionally NOT derived (LOSS-07, 06.4-RESEARCH Pitfall 7): the
/// [`EvalMetric::Custom`] variant carries a [`CustomMetricHandle`] (`Arc<dyn>`),
/// and `Arc` is not `Copy`. `Copy` was dropped HERE in a one-time mechanical
/// refactor (mirroring the 6.2 `Loss` Copy-drop) — the by-value `self` methods
/// (`eval` / `eval_grouped` / `eval_one_group` / `empty_metric_default`) became
/// `&self`; the call sites that matched `*self` now match `self` by reference.
/// `Clone` is retained and cheap (an `Arc` refcount bump for `Custom`, a bitwise
/// copy for every other variant).
#[derive(Debug, Clone, PartialEq)]
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
    /// Mean absolute error: `Σ w·|approx−target| / Σ w` (flat, Min-optimized,
    /// `is_max_optimal == false`). Evaluated via the flat [`EvalMetric::eval`]
    /// path, never the grouped seam.
    Mae,
    /// Mean absolute percentage error: `Σ w·|approx−target|/D(target) / Σ w`
    /// with a zero-target-guarded divisor `D` (flat, Min-optimized,
    /// `is_max_optimal == false`). Evaluated via the flat [`EvalMetric::eval`]
    /// path, never the grouped seam.
    Mape,
    /// Quantile (pinball) loss at `alpha`: `Σ w·pinball(a,t,alpha) / Σ w`
    /// (flat, Min-optimized, `is_max_optimal == false`); default `alpha == 0.5`
    /// (see `cb_compute::QUANTILE_ALPHA`), at which it equals `0.5·MAE`.
    /// Evaluated via the flat [`EvalMetric::eval`] path, never the grouped seam.
    Quantile {
        /// Pinball quantile level in `(0, 1)`; default `0.5`.
        alpha: f64,
    },
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
    /// Custom user metric (LOSS-07, D-6.4-05): a user-supplied
    /// [`cb_compute::CustomMetric`] trait object (`evaluate` / `get_final_error`
    /// / `is_max_optimal`, mirroring the Python `CustomMetric` callback) plugged
    /// into the ONE [`EvalMetric::eval`] dispatch via the [`CustomMetricHandle`]
    /// `Arc<dyn>` newtype. Carrying a non-`Copy` `Arc` is why `EvalMetric` drops
    /// `Copy` (06.4-RESEARCH Pitfall 7). The Phase-8 PyO3 callback (D-09) wraps
    /// the SAME trait — no `pyo3` dependency is added here.
    Custom(CustomMetricHandle),
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
            // Custom objective (LOSS-07): with no explicit `eval_metric` we cannot
            // auto-derive a `CustomMetric` from an opaque objective, so default to
            // the parity-neutral RMSE eval surface (the same default the other
            // non-classification arms use). A user wanting a custom eval surface
            // sets `eval_metric = EvalMetric::Custom(..)` explicitly.
            Loss::Custom(_) => Self::Rmse,
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
    pub fn eval(&self, approx: &[f64], target: &[f64], weights: &[f64]) -> CbResult<f64> {
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
        // uniform 1.0 when no weights are supplied. A missing per-object weight
        // falls back to 1.0 (the upstream "missing weight == 1.0" convention),
        // MATCHING `eval_grouped` so both metric paths weight an out-of-range index
        // identically (WR-05). After the up-front `weights.len() == approx.len()`
        // check this branch is unreachable; the shared fallback keeps the two paths
        // consistent if that guard is ever relaxed.
        let weight_at = |i: usize| -> f64 {
            if weights.is_empty() {
                1.0
            } else {
                weights.get(i).copied().unwrap_or(1.0)
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
            // Flat Min-optimized metrics (EM-01/EM-02/EM-03). The real
            // final-error math lands in EMT-2 (MAE) / EMT-3 (MAPE) / EMT-4
            // (Quantile); these are inert typed placeholders so the crate
            // compiles with the new variants present (EMT-1 compile-safety
            // spine). Kept next to the flat arms (Rmse/Logloss/Msle) above.
            Self::Mae => {
                // sum_w |approx - target| / sum_w — weighted mean absolute error
                // (EM-01). Min-optimized flat metric; mirrors the `Rmse` fold
                // shape above (build the weighted per-object column, reduce via
                // `cb_core::sum_f64` (D-08), divide by the shared `total_weight`).
                let weighted_abs: Vec<f64> = approx
                    .iter()
                    .zip(target.iter())
                    .enumerate()
                    .map(|(i, (&a, &t))| weight_at(i) * (a - t).abs())
                    .collect();
                Ok(sum_f64(&weighted_abs) / total_weight)
            }
            Self::Mape => {
                // sum_w |approx - target| / D(target) / sum_w — weighted mean
                // absolute percentage error (EM-02), where D(t) is the guarded,
                // zero-target-safe divisor. Min-optimized flat metric; mirrors the
                // `Mae` fold shape above (build the weighted per-object column,
                // reduce via `cb_core::sum_f64` (D-08), divide by `total_weight`).
                //
                // R1 (zero-target divisor) — RESOLVED (EMT-6): `D(t) = max(1.0, |t|)`,
                // the upstream `TMAPEMetric` convention. Pinned against the frozen
                // `catboost==1.2.10` scalar in
                // `calc_metrics_flat_oracle_test::mape_matches_upstream`, whose `{0,1}`
                // label carries zero-target rows: `max(1.0,|t|)` reproduces the
                // upstream value to ~1e-16, while SPEC §4's `max(|t|, EPS)` explodes
                // (zero row → ~1e37) and skip-zero undershoots — so this convention is
                // the unique arbiter-confirmed one. The divisor is >= 1 for every row,
                // so a `target == 0` row is FINITE (no div-by-zero, no NaN/Inf).
                let weighted_ape: Vec<f64> = approx
                    .iter()
                    .zip(target.iter())
                    .enumerate()
                    .map(|(i, (&a, &t))| {
                        let divisor = t.abs().max(1.0);
                        weight_at(i) * (a - t).abs() / divisor
                    })
                    .collect();
                Ok(sum_f64(&weighted_ape) / total_weight)
            }
            Self::Quantile { alpha } => {
                // sum_w pinball(approx, target, alpha) / sum_w — weighted mean
                // pinball (quantile) loss (EM-03), where
                // `pinball(a,t,alpha) = t>=a ? alpha·(t−a) : (1−alpha)·(a−t)`.
                // Min-optimized flat metric; mirrors the `Mae` fold shape above
                // (build the weighted per-object column, reduce via
                // `cb_core::sum_f64` (D-08), divide by the shared `total_weight`).
                // `self` is matched by reference, so `alpha` is `&f64` — deref for
                // the arithmetic. At `alpha == 0.5` (the parse default,
                // `cb_compute::QUANTILE_ALPHA`) every row contributes `0.5·|a−t|`,
                // so the metric equals `0.5·MAE`.
                let a_lvl = *alpha;
                let weighted_pinball: Vec<f64> = approx
                    .iter()
                    .zip(target.iter())
                    .enumerate()
                    .map(|(i, (&a, &t))| {
                        let d = t - a;
                        let pinball = if d >= 0.0 {
                            a_lvl * d
                        } else {
                            (1.0 - a_lvl) * -d
                        };
                        weight_at(i) * pinball
                    })
                    .collect();
                Ok(sum_f64(&weighted_pinball) / total_weight)
            }
            // Custom user metric (LOSS-07, D-6.4-05): the user trait accumulates
            // `(error_sum, weight_sum)` via `evaluate`, then `get_final_error`
            // reduces them (e.g. `error / weight`). The trait owns its own folding
            // (it may, like the built-ins, route through a stable order); the
            // result is REJECTED if non-finite (T-06.4D-02 — a NaN/Inf metric must
            // not reach the overfitting-detector gate). No `unwrap`/`panic`
            // (T-06.4D-01) — the trait returns a typed `CbError`.
            Self::Custom(handle) => {
                let (error, weight) = handle.0.evaluate(approx, target, weights)?;
                let value = handle.0.get_final_error(error, weight);
                if !value.is_finite() {
                    return Err(CbError::Degenerate(format!(
                        "custom eval metric produced a non-finite value: {value} \
                         (error={error}, weight={weight})"
                    )));
                }
                Ok(value)
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
        &self,
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
    fn eval_one_group(&self, approx: &[f64], target: &[f64]) -> CbResult<f64> {
        // Matched by reference (`&self`, the Copy-drop ripple of LOSS-07): the
        // `Custom` variant carries a non-`Copy` `Arc`, so `self` cannot be matched
        // by value. The ranking arms bind their `Copy` params by reference and
        // dereference at the call site (byte-identical to the pre-LOSS-07
        // by-value path; D-04 no-regression).
        Ok(match self {
            Self::Ndcg {
                top,
                dcg_type,
                denominator,
            } => ndcg_group(approx, target, *top, *dcg_type, *denominator),
            Self::Dcg {
                top,
                dcg_type,
                denominator,
            } => dcg_group(approx, target, *top, *dcg_type, *denominator),
            Self::Map { top, border } => map_at_group(approx, target, *top, *border),
            Self::Mrr { top, border } => mrr_group(approx, target, *top, *border),
            Self::Err { top } => err_group(approx, target, *top),
            Self::PFound { top, decay } => pfound_group(approx, target, *top, *decay),
            Self::PrecisionAt { top, border } => precision_at_group(approx, target, *top, *border),
            Self::RecallAt { top, border } => recall_at_group(approx, target, *top, *border),
            Self::QueryAuc { auc_type } => query_auc_group(approx, target, *auc_type),
            // Flat metrics (the `eval` path), never the grouped seam:
            // Rmse/Logloss/Msle, the Min-optimized Mae/Mape/Quantile, and Custom.
            Self::Rmse
            | Self::Logloss
            | Self::Msle
            | Self::Mae
            | Self::Mape
            | Self::Quantile { .. }
            | Self::Custom(_) => {
                return Err(CbError::Degenerate(
                    "non-ranking metric passed to the grouped seam (use eval)".to_owned(),
                ))
            }
        })
    }

    /// The empty-eval-set default per metric (`GetFinalError` with `Stats[1]==0`):
    /// Precision/Recall return `1`, every other ranking metric returns `0`.
    fn empty_metric_default(&self) -> f64 {
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
