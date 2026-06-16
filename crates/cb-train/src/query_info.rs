//! `QueryInfo` — the grouped (ranking) view built once per fit from a [`Pool`]'s
//! `group_id` / `subgroup_id` / `pairs` columns (LOSS-04, D-6.3-03).
//!
//! This is the design hinge of the ranking-loss surface: every ranking loss
//! funnels through a `QueryInfo`-shaped grouped view that mirrors upstream
//! catboost 1.2.10's `TQueryInfo`
//! (`catboost-master/catboost/private/libs/data_types/query.h:19-44`):
//! a half-open object span `[begin, end)`, a per-group `weight`, an optional
//! `subgroup_id` vector, and the `competitors` adjacency (winner-local-index →
//! list of `{loser-local-index, weight}`) that both explicit pairs and (later)
//! YetiRank-sampled pairs populate.
//!
//! # Group detection mirrors upstream `GroupSamples`
//!
//! [`build_query_info`] scans `group_id` for contiguous equal runs — one
//! [`QueryInfo`] per run, `begin`/`end` the half-open object span — exactly as
//! upstream `GroupSamples` (`query.h:48-67`): it collects the run bounds, then
//! asserts the set of seen group ids has **no duplicate after sorting** (a group
//! id reappearing after a *different* id has intervened). Upstream uses
//! `CB_ENSURE(... "queryIds should be grouped")`; here that becomes a typed
//! [`CbError::Degenerate`] — never a panic (CLAUDE.md bans
//! `unwrap`/`expect`/`panic`/indexing-slicing in library code; T-06.3-01-01).
//!
//! # Pairs → competitors mirrors upstream `data_providers.cpp`
//!
//! Explicit [`Pair`]s map into per-group `competitors` exactly as upstream
//! (`data_providers.cpp:315-340`):
//! `competitors[winner - group.begin].push({loser - group.begin, weight})`.
//! Winner and loser must fall in the same group and in range, else a typed
//! [`CbError`] is returned (T-06.3-01-02) — no index arithmetic on unvalidated
//! input.
//!
//! # Group weight convention
//!
//! Upstream sets `group.Weight = groupWeights[group.Begin]` (the first member's
//! weight; `data_providers.cpp:297`). When per-object `weights` are supplied this
//! builder follows the documented per-group convention of the **mean of member
//! weights** (a single-member group reduces to that member's weight, matching the
//! upstream first-member value for the common uniform-within-group case); when
//! `weights` is empty every group weight defaults to `1.0`. The mean is reduced
//! through `cb_core::sum_f64` (D-08 — no raw float fold).

use cb_core::{sum_f64, CbError, CbResult};
use cb_data::Pair;

/// A single competitor edge inside a query group: the loser this winner is
/// preferred over, plus the pair weight. Mirrors upstream `TCompetitor`
/// (`query.h`): `{ Id, Weight }`, where `Id` is the loser's **group-local**
/// object index (`loser_global - group.begin`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Competitor {
    /// Group-local index of the losing object (`loser_global - begin`).
    pub id: usize,
    /// Pair weight (defaults to `1.0` for an unweighted explicit [`Pair`]).
    pub weight: f64,
}

/// One query group's grouped view, mirroring upstream `TQueryInfo`
/// (`query.h:19-44`): the half-open object span `[begin, end)`, the per-group
/// `weight`, the optional per-member `subgroup_id`, and the `competitors`
/// adjacency (indexed by **group-local** winner index).
#[derive(Debug, Clone, PartialEq)]
pub struct QueryInfo {
    /// Inclusive start object index of the group (half-open `[begin, end)`).
    pub begin: usize,
    /// Exclusive end object index of the group.
    pub end: usize,
    /// Per-group weight (see module doc: mean of member weights, else `1.0`).
    pub weight: f64,
    /// Per-member subgroup id (empty when no subgroup column is present);
    /// length `end - begin` when present, indexed group-locally.
    pub subgroup_id: Vec<u64>,
    /// `competitors[winner_local]` is the list of losers `winner_local` is
    /// preferred over. Empty (all-empty rows) when the group has no pairs.
    pub competitors: Vec<Vec<Competitor>>,
}

impl QueryInfo {
    /// Number of objects in this group (`end - begin`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.end - self.begin
    }
}

/// Build the grouped (ranking) view `Vec<QueryInfo>` from a [`Pool`]'s raw
/// `group_id` / `subgroup_id` / `pairs` columns plus per-object `weights`
/// (D-6.3-03).
///
/// Mirrors upstream `GroupSamples` (`query.h:48-67`) for run detection and the
/// `data_providers.cpp:290-340` pairs→competitors mapping. Ungrouped input
/// (empty `group_id`) yields a single [`QueryInfo`] spanning all `n_rows`
/// objects (the natural "one big group" view).
///
/// `weights` is either empty (every object weight `1.0`) or length `n_rows`.
/// `subgroup_id` is either empty or length `n_rows`. `n_rows` is the object
/// count the columns describe; when `group_id` is non-empty its length must
/// equal `n_rows`.
///
/// # Errors
/// - [`CbError::Degenerate`] if `group_id` is non-contiguous (a group id
///   reappears after a different id — upstream's "queryIds should be grouped"),
///   or if a supplied column length disagrees with `n_rows`.
/// - [`CbError::OutOfRange`] if a [`Pair`]'s `winner_id`/`loser_id` is out of
///   range, or the two endpoints fall in different groups (T-06.3-01-02).
pub fn build_query_info(
    n_rows: usize,
    group_id: &[u64],
    subgroup_id: &[u64],
    pairs: &[Pair],
    weights: &[f64],
) -> CbResult<Vec<QueryInfo>> {
    if !group_id.is_empty() && group_id.len() != n_rows {
        return Err(CbError::Degenerate(format!(
            "build_query_info: group_id length {} != n_rows {n_rows}",
            group_id.len()
        )));
    }
    if !subgroup_id.is_empty() && subgroup_id.len() != n_rows {
        return Err(CbError::Degenerate(format!(
            "build_query_info: subgroup_id length {} != n_rows {n_rows}",
            subgroup_id.len()
        )));
    }
    if !weights.is_empty() && weights.len() != n_rows {
        return Err(CbError::Degenerate(format!(
            "build_query_info: weights length {} != n_rows {n_rows}",
            weights.len()
        )));
    }

    // Per-object weight accessor: uniform 1.0 when no weights supplied.
    let weight_at = |i: usize| -> f64 {
        if weights.is_empty() {
            1.0
        } else {
            weights.get(i).copied().unwrap_or(1.0)
        }
    };

    // (1) Detect contiguous group runs as half-open [begin, end) spans, mirroring
    //     upstream `GroupSamples` (query.h:50-61). Ungrouped input → one span.
    let bounds: Vec<(usize, usize)> = if group_id.is_empty() {
        if n_rows == 0 {
            Vec::new()
        } else {
            vec![(0, n_rows)]
        }
    } else {
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
        // Upstream asserts the seen-id set is unique after sorting (query.h:62-66):
        // a group id that reappears after a DIFFERENT id intervened is rejected.
        seen.sort_unstable();
        if seen.windows(2).any(|w| w.first() == w.get(1)) {
            return Err(CbError::Degenerate(
                "build_query_info: group_id is not contiguous (queryIds should be grouped)"
                    .to_owned(),
            ));
        }
        runs
    };

    // (2) Materialize one QueryInfo per run: weight (mean of members), subgroup
    //     ids, and empty competitor rows (filled in step 3).
    let mut groups: Vec<QueryInfo> = Vec::with_capacity(bounds.len());
    for &(begin, end) in &bounds {
        let size = end - begin;
        // Per-group weight: mean of member weights (single-member groups reduce
        // to that member, matching upstream `groupWeights[begin]` for uniform
        // groups). Reduced through cb_core::sum_f64 (D-08).
        let weight = if size == 0 {
            1.0
        } else {
            let members: Vec<f64> = (begin..end).map(weight_at).collect();
            sum_f64(&members) / (size as f64)
        };
        let sub: Vec<u64> = if subgroup_id.is_empty() {
            Vec::new()
        } else {
            (begin..end)
                .map(|i| subgroup_id.get(i).copied().unwrap_or_default())
                .collect()
        };
        groups.push(QueryInfo {
            begin,
            end,
            weight,
            subgroup_id: sub,
            competitors: vec![Vec::new(); size],
        });
    }

    // (3) Map explicit pairs into per-group competitors, re-indexing the global
    //     winner/loser ids to group-local indices (data_providers.cpp:315-340).
    //     A pair whose endpoints are out of range or in different groups is a
    //     typed error (T-06.3-01-02) — never an unchecked index.
    if !pairs.is_empty() {
        // Object → group index lookup, built once (mirrors upstream
        // objectToGroupIdxMap). Objects outside any group stay None.
        let mut object_group: Vec<Option<usize>> = vec![None; n_rows];
        for (g_idx, g) in groups.iter().enumerate() {
            for obj in g.begin..g.end {
                if let Some(slot) = object_group.get_mut(obj) {
                    *slot = Some(g_idx);
                }
            }
        }
        for pair in pairs {
            let winner = pair.winner_id as usize;
            let loser = pair.loser_id as usize;
            let w_group = object_group.get(winner).copied().flatten();
            let l_group = object_group.get(loser).copied().flatten();
            match (w_group, l_group) {
                (Some(wg), Some(lg)) if wg == lg => {
                    let group = groups.get_mut(wg).ok_or_else(|| {
                        CbError::OutOfRange(format!(
                            "build_query_info: pair winner {winner} group index out of range"
                        ))
                    })?;
                    let winner_local = winner - group.begin;
                    let loser_local = loser - group.begin;
                    let row = group.competitors.get_mut(winner_local).ok_or_else(|| {
                        CbError::OutOfRange(format!(
                            "build_query_info: pair winner local index {winner_local} out of range"
                        ))
                    })?;
                    row.push(Competitor {
                        id: loser_local,
                        weight: 1.0,
                    });
                }
                _ => {
                    return Err(CbError::OutOfRange(format!(
                        "build_query_info: pair ({winner}, {loser}) is out of range or crosses \
                         group boundaries"
                    )));
                }
            }
        }
    }

    Ok(groups)
}

#[cfg(test)]
#[path = "query_info_test.rs"]
mod tests;
