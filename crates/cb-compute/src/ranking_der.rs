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
        let _target_slice = target.get(group.begin..group.end).ok_or_else(|| {
            CbError::Degenerate("calc_ders_for_queries: target slice out of range".to_owned())
        })?;
        let weight_slice: &[f64] = if weights.is_empty() {
            &[]
        } else {
            weights.get(group.begin..group.end).ok_or_else(|| {
                CbError::Degenerate("calc_ders_for_queries: weight slice out of range".to_owned())
            })?
        };

        // Empty-group guard: a zero-size group contributes nothing and never
        // divides (Security V5 — mirror metrics.rs:145-149). The per-group
        // normalizer is computed (and discarded for now) to exercise the seam's
        // reduction path through cb_core::sum_f64.
        let _normalizer = if group.size() == 0 {
            0.0
        } else {
            group_reduce_weighted(approx_slice, weight_slice)
        };

        // Dispatch to the per-loss arm. Plan 06.3-01: every variant is unwired —
        // Plans 02–05 replace this with the transcribed per-loss der arms
        // (QueryRMSE / QuerySoftMax / PairLogit / LambdaMart / YetiRank /
        // StochasticRank). The per-group slicing + normalizer above is the
        // infrastructure those arms attach to.
        let _ = &mut per_group;
        return Err(CbError::OutOfRange(format!(
            "calc_ders_for_queries: ranking loss not yet wired for {loss:?} \
             (grouped der arms land in Plans 06.3-02..05)"
        )));
    }

    // Reached only when `groups` is empty (no group ever dispatched): the seam
    // returns an empty der set, never panicking on degenerate ungrouped input.
    Ok(per_group)
}

#[cfg(test)]
#[path = "ranking_der_test.rs"]
mod tests;
