//! The grouped der seam (LOSS-04, D-6.3-03) — the design hinge every ranking
//! loss funnels through. Mirrors upstream catboost 1.2.10
//! `IDerCalcer::CalcDersForQueries`
//! (`catboost-master/catboost/private/libs/algo_helpers/error_functions.h:831-841`):
//! given the flat approx/target/weights and a `Vec` of per-group spans (the
//! `TQueryInfo` mirror), it slices `approx[begin..end]` per group and routes each
//! group to a per-loss der arm.
//!
//! # This plan delivers the seam, not the loss math
//!
//! Plan 06.3-01 lands ONLY the dispatch skeleton + per-group slicing
//! infrastructure + the shared per-group normalizer [`group_reduce_weighted`].
//! Every concrete [`Loss`] arm returns a typed "ranking loss not yet wired"
//! [`CbError`]; Plans 02–05 replace these arms with the transcribed QueryRMSE /
//! QuerySoftMax / PairLogit / LambdaMart / YetiRank / StochasticRank der.
//!
//! # No cb-train dependency (crate layering)
//!
//! `cb-train` depends on `cb-compute`, never the reverse. So this seam does NOT
//! import `cb-train::QueryInfo`; it re-declares a compute-tier [`GroupSpan`] /
//! [`Competitor`] carrying the same shape as plain data (`cb-train::QueryInfo`
//! lowers into `Vec<GroupSpan>` at the call site). This keeps `cb-compute` free
//! of the trainer and matches the existing layering (RESEARCH Architectural
//! Responsibility Map: grouped der is host-side `cb-compute`, NOT a kernel).
//!
//! # Parity discipline
//!
//! Every per-group / per-pair reduction routes through `cb_core::sum_f64` in
//! upstream iteration order (group index ascending, doc index ascending —
//! RESEARCH Pitfall 4). Empty-group / zero-weight division returns `0.0`, never
//! divides (Security V5 — mirrors `cb-train::metrics.rs:145-149`); no
//! `unwrap`/`expect`/`panic`/indexing-slicing on the grouped input.

use cb_core::{sum_f64, CbError, CbResult};

use crate::loss::{queryrmse_der, querysoftmax_der};
use crate::runtime::{Derivatives, Loss};

/// A competitor edge inside a group: the group-local loser index plus the pair
/// weight. Compute-tier mirror of `cb-train::Competitor` / upstream
/// `TCompetitor` (`query.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Competitor {
    /// Group-local index of the losing object.
    pub id: usize,
    /// Pair weight.
    pub weight: f64,
}

/// One query group's boundary span + per-group weight + competitor adjacency,
/// the compute-tier mirror of `cb-train::QueryInfo` (and upstream `TQueryInfo`,
/// `query.h:19-44`). The seam slices `approx[begin..end]` per group; per-pair
/// losses read `competitors`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupSpan {
    /// Inclusive start object index (half-open `[begin, end)`).
    pub begin: usize,
    /// Exclusive end object index.
    pub end: usize,
    /// Per-group weight.
    pub weight: f64,
    /// `competitors[winner_local]` → losers `winner_local` is preferred over.
    pub competitors: Vec<Vec<Competitor>>,
}

impl GroupSpan {
    /// Number of objects in the group (`end - begin`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.end - self.begin
    }
}

/// Weighted per-group reduction `Σ_i slice[i] * weights[i]` — the per-group
/// normalizer every querywise ranking loss uses (e.g. QueryRMSE's `queryAvrg`
/// numerator, QuerySoftMax's `Σ expApprox·w`). Reduced through `cb_core::sum_f64`
/// in object order (D-08 — no raw float fold). `weights` is either empty
/// (uniform `1.0`) or the same length as `slice`; a length disagreement folds
/// only the in-range prefix's weights (callers pass matched slices).
#[must_use]
pub fn group_reduce_weighted(slice: &[f64], weights: &[f64]) -> f64 {
    let products: Vec<f64> = slice
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = if weights.is_empty() {
                1.0
            } else {
                weights.get(i).copied().unwrap_or(0.0)
            };
            v * w
        })
        .collect();
    sum_f64(&products)
}

/// Compute the grouped first/second derivatives for `loss` over the grouped view
/// `groups`, mirroring upstream `CalcDersForQueries`
/// (`error_functions.h:831-841`).
///
/// `approx` / `target` / `weights` are the flat per-object buffers (length
/// `n_objects`; `weights` may be empty for uniform `1.0`). Each [`GroupSpan`]
/// carries the half-open `[begin, end)` object span this group owns; the seam
/// slices `approx[begin..end]` and routes to the per-loss arm.
///
/// Plan 06.3-01 implements the dispatch skeleton + per-group slicing only — every
/// loss arm returns a typed "ranking loss not yet wired" error (Plans 02–05 fill
/// them in). The per-group reductions that the real arms will use route through
/// [`group_reduce_weighted`] / `cb_core::sum_f64`.
///
/// # Errors
/// - [`CbError::Degenerate`] if a [`GroupSpan`] span is out of range for
///   `approx`/`target`, or `approx`/`target` lengths disagree.
/// - [`CbError::OutOfRange`] (kind "ranking loss not yet wired") for every loss
///   variant in this plan — the seam is structural; loss math lands in 02–05.
pub fn calc_ders_for_queries(
    loss: &Loss,
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    groups: &[GroupSpan],
    _random_seed: u64,
) -> CbResult<Vec<Derivatives>> {
    if approx.len() != target.len() {
        return Err(CbError::Degenerate(format!(
            "calc_ders_for_queries: approx len {} != target len {}",
            approx.len(),
            target.len()
        )));
    }
    if !weights.is_empty() && weights.len() != approx.len() {
        return Err(CbError::Degenerate(format!(
            "calc_ders_for_queries: weights len {} != approx len {}",
            weights.len(),
            approx.len()
        )));
    }

    let mut per_group: Vec<Derivatives> = Vec::with_capacity(groups.len());
    for group in groups {
        // Per-group slice bounds, validated before any indexing (Security V5 — no
        // unchecked slice on the grouped input).
        if group.begin > group.end || group.end > approx.len() {
            return Err(CbError::Degenerate(format!(
                "calc_ders_for_queries: group span [{}, {}) out of range for n={}",
                group.begin,
                group.end,
                approx.len()
            )));
        }
        let approx_slice = approx.get(group.begin..group.end).ok_or_else(|| {
            CbError::Degenerate("calc_ders_for_queries: approx slice out of range".to_owned())
        })?;
        let target_slice = target.get(group.begin..group.end).ok_or_else(|| {
            CbError::Degenerate("calc_ders_for_queries: target slice out of range".to_owned())
        })?;
        let weight_slice: &[f64] = if weights.is_empty() {
            &[]
        } else {
            weights.get(group.begin..group.end).ok_or_else(|| {
                CbError::Degenerate("calc_ders_for_queries: weight slice out of range".to_owned())
            })?
        };

        // Per-object weight accessor for this group: uniform 1.0 when unweighted.
        let weight_at = |i: usize| -> f64 {
            if weight_slice.is_empty() {
                1.0
            } else {
                weight_slice.get(i).copied().unwrap_or(0.0)
            }
        };

        // Empty group → an empty der set for the group (Security V5 — never
        // divides, mirror metrics.rs:145-149). Plans 02–05 wired arms below all
        // short-circuit on a zero-size group to the empty der.
        if group.size() == 0 {
            per_group.push(Derivatives {
                der1: Vec::new(),
                der2: Vec::new(),
            });
            continue;
        }

        // Dispatch to the per-loss arm (Plan 06.3-02 wires the first two querywise
        // ranking losses; PairLogit / LambdaMart / YetiRank / StochasticRank stay
        // unwired until Plans 03–05).
        let ders = match loss {
            // QueryRMSE (Wave A): per-group weighted residual mean `queryAvrg`,
            // then per-object `der1 = (target - approx - queryAvrg)·w`,
            // `der2 = -1·w` (error_functions.h:879-933). The `queryAvrg`
            // numerator `Σ (target - approx)·w` and denominator `Σ w` both route
            // through cb_core::sum_f64 (D-08, doc-ascending order).
            Loss::QueryRmse => {
                // residual[i] = target[i] - approx[i] (group-local).
                let residuals: Vec<f64> = (0..group.size())
                    .map(|i| {
                        target_slice.get(i).copied().unwrap_or(0.0)
                            - approx_slice.get(i).copied().unwrap_or(0.0)
                    })
                    .collect();
                // numerator = Σ residual·w ; denominator = Σ w (both sum_f64).
                let numerator = group_reduce_weighted(&residuals, weight_slice);
                let weight_col: Vec<f64> = (0..group.size()).map(weight_at).collect();
                let denominator = sum_f64(&weight_col);
                // queryCount > 0 guard (error_functions.h:928): zero-weight group →
                // queryAvrg 0, never divides.
                let query_avrg = if denominator > 0.0 {
                    numerator / denominator
                } else {
                    0.0
                };
                let mut der1 = Vec::with_capacity(group.size());
                let mut der2 = Vec::with_capacity(group.size());
                for i in 0..group.size() {
                    let a = approx_slice.get(i).copied().unwrap_or(0.0);
                    let t = target_slice.get(i).copied().unwrap_or(0.0);
                    let (d1, d2) = queryrmse_der(a, t, weight_at(i), query_avrg);
                    der1.push(d1);
                    der2.push(d2);
                }
                Derivatives { der1, der2 }
            }
            // QuerySoftMax (Wave A): per-group softmax over `Beta·approx`,
            // MAX-SHIFTED before exp (error_functions.cpp:540-552). The exp share
            // `p = expApprox·w / Σ expApprox·w`; der per error_functions.cpp:560-565.
            // `sumWTargets <= 0` (or `weight <= 0`) → ders 0 (no divide).
            Loss::QuerySoftMax { lambda, beta } => {
                // (1) maxApprox + sumWeightedTargets over weight>0 objects
                // (error_functions.cpp:540-550). maxApprox seeds at the most
                // negative finite f64 (upstream `-numeric_limits::max()`); an
                // all-zero-weight group leaves it at that seed but the
                // `sumWTargets <= 0` guard short-circuits before exp.
                let mut max_approx = f64::MIN;
                let mut target_terms: Vec<f64> = Vec::with_capacity(group.size());
                for i in 0..group.size() {
                    let w = weight_at(i);
                    let a = approx_slice.get(i).copied().unwrap_or(0.0);
                    let t = target_slice.get(i).copied().unwrap_or(0.0);
                    if w > 0.0 {
                        if a > max_approx {
                            max_approx = a;
                        }
                        if t > 0.0 {
                            target_terms.push(w * t);
                        }
                    }
                }
                // Σ w·target through the sanctioned ordered reduction (D-08).
                let sum_weighted_targets = sum_f64(&target_terms);
                if sum_weighted_targets > 0.0 {
                    // (2) expApprox[i] = exp(Beta·(approx[i] - maxApprox)) · w, the
                    // numerator the share p divides by. `calc_softmax` already
                    // max-subtracts + exps, but it normalizes UNWEIGHTED over ALL
                    // objects; here the share is WEIGHTED and the shift uses Beta.
                    // Transcribe the shift directly (the calc_softmax NaN-guard
                    // discipline) so weight>0/weight==0 objects are handled per
                    // upstream.
                    let weighted_exp: Vec<f64> = (0..group.size())
                        .map(|i| {
                            let w = weight_at(i);
                            let a = approx_slice.get(i).copied().unwrap_or(0.0);
                            // exp(Beta·(approx - maxApprox)) · w.
                            (beta * (a - max_approx)).exp() * w
                        })
                        .collect();
                    // sumExpApprox = Σ weighted_exp (sum_f64, doc-ascending — D-08).
                    let sum_exp = sum_f64(&weighted_exp);
                    let mut der1 = Vec::with_capacity(group.size());
                    let mut der2 = Vec::with_capacity(group.size());
                    for i in 0..group.size() {
                        let w = weight_at(i);
                        if w > 0.0 && sum_exp > 0.0 {
                            let p = weighted_exp.get(i).copied().unwrap_or(0.0) / sum_exp;
                            let t = target_slice.get(i).copied().unwrap_or(0.0);
                            let (d1, d2) = querysoftmax_der(
                                p,
                                sum_weighted_targets,
                                w,
                                t,
                                *beta,
                                *lambda,
                            );
                            der1.push(d1);
                            der2.push(d2);
                        } else {
                            // weight <= 0 → ders 0 (error_functions.cpp:566-569).
                            der1.push(0.0);
                            der2.push(0.0);
                        }
                    }
                    Derivatives { der1, der2 }
                } else {
                    // sumWTargets <= 0 → all ders 0 (error_functions.cpp:571-576).
                    Derivatives {
                        der1: vec![0.0; group.size()],
                        der2: vec![0.0; group.size()],
                    }
                }
            }
            // PairLogit / LambdaMart / YetiRank / StochasticRank (and every non-
            // ranking loss) stay unwired here — Plans 03–05 fill them; the
            // pointwise losses never reach the grouped seam.
            _ => {
                return Err(CbError::OutOfRange(format!(
                    "calc_ders_for_queries: ranking loss not yet wired for {loss:?} \
                     (grouped der arms land in Plans 06.3-03..05)"
                )));
            }
        };
        per_group.push(ders);
    }

    // Reached only when `groups` is empty (no group ever dispatched): the seam
    // returns an empty der set, never panicking on degenerate ungrouped input.
    Ok(per_group)
}

#[cfg(test)]
#[path = "ranking_der_test.rs"]
mod tests;
