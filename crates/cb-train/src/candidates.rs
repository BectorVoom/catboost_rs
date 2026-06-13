//! One-hot vs CTR encoding-path selection for categorical features (ORD-04,
//! D-04). This is the narrowest first slice of the high-risk categorical phase:
//! it decides, purely from a categorical column's LEARN-SET cardinality and the
//! `one_hot_max_size` knob, whether the column is materialized as a set of
//! one-hot binary splits (low cardinality) or deferred to the CTR path (high
//! cardinality, built in later waves). NO permutation and NO CTR math live here
//! — a one-hot-only model rides the EXISTING plain boosting + oblivious trees.
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/greedy_tensor_search.cpp`:
//! - `:171-197` — `AddOneHotFeatures`: a categorical feature is emitted as a
//!   one-hot candidate iff `!((onLearnOnlyCount > oneHotMaxSize) || (onLearnOnlyCount <= 1))`,
//!   i.e. one-hot when `1 < onLearnOnlyCount <= oneHotMaxSize` (boundary
//!   INCLUSIVE at `== oneHotMaxSize`). `onLearnOnlyCount` is the learn-set-only
//!   unique value count (RESEARCH Pitfall 3).
//! - `:457-551` — the CTR candidate path is taken for the complementary range
//!   `onLearnOnlyCount > oneHotMaxSize` (boundary EXCLUSIVE); that path is
//!   deferred to later waves of this phase.
//!
//! `catboost/private/libs/options/cat_feature_options.cpp:231-232`:
//! - default `one_hot_max_size = 2` (pinned EXPLICITLY by the caller, never
//!   auto-selected — RESEARCH Pitfall 6).
//!
//! # Cardinality (RESEARCH Pitfall 3)
//!
//! Cardinality is the LEARN-SET-only count of DISTINCT categorical values,
//! computed via the single categorical-hash source
//! [`cb_data::calc_cat_feature_hash`] + the first-seen perfect-hash bins
//! ([`cb_data::PerfectHash`]) — NEVER a model's `ctr_data` hash_map (D Carried
//! Forward / RESEARCH Anti-Pattern). Two raw values collide into one bin iff
//! their CityHash64-derived ui32 hashes are equal, exactly as upstream's
//! `GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly` counts.
//!
//! # Fallibility
//!
//! Cardinality counting can surface [`cb_core::CbError::OutOfRange`] when a
//! column exceeds the perfect-hash `u32::MAX` bound (propagated from
//! [`cb_data::PerfectHash::remap`]). Checked access only (`.get`); no
//! `unwrap`/`expect`/`panic`/raw index in production (INFRA-02). No `anyhow`.

use cb_core::CbResult;
use cb_data::{calc_cat_feature_hash, PerfectHash};

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file). Mounted as a CHILD module
// of `candidates` so the canonical filter `cargo test -p cb-train candidates::`
// selects them; the boundary tests are reachable via the `one_hot_threshold`
// substring filter the plan's verify command uses
// (`cargo test -p cb-train one_hot_threshold`).
#[cfg(test)]
#[path = "candidates_test.rs"]
mod tests;

/// The encoding path a categorical column routes to, decided from its learn-set
/// cardinality and `one_hot_max_size` ([`one_hot_max_size_default`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingPath {
    /// One-hot encoding (`1 < cardinality <= one_hot_max_size`): the column is
    /// materialized as one-hot binary splits the oblivious tree consumes
    /// directly (ORD-04, this wave). No permutation, no CTR.
    OneHot,
    /// CTR path (`cardinality > one_hot_max_size`): the column is deferred to
    /// the target-statistic / CTR machinery built in later waves of this phase.
    Ctr,
    /// Neither path (`cardinality <= 1`): a constant column carries no split
    /// information and is skipped entirely (`AddOneHotFeatures` skip-if-`<=1`).
    Skip,
}

/// The upstream default `one_hot_max_size`
/// (`cat_feature_options.cpp:231-232`). Pinned explicitly by the caller (never
/// auto-selected — RESEARCH Pitfall 6); exposed so a [`crate::BoostParams`] /
/// fixture config can reference the canonical default without re-encoding the
/// magic number.
pub const fn one_hot_max_size_default() -> u32 {
    2
}

/// Route a categorical column to its encoding path purely from its learn-set
/// cardinality and `one_hot_max_size` (`AddOneHotFeatures`,
/// `greedy_tensor_search.cpp:171-197`).
///
/// The boundary is INCLUSIVE for one-hot at `cardinality == one_hot_max_size`
/// and EXCLUSIVE above it (CTR), with a constant column (`cardinality <= 1`)
/// skipped entirely — reproducing the upstream skip predicate
/// `(onLearnOnlyCount > oneHotMaxSize) || (onLearnOnlyCount <= 1)` (RESEARCH
/// Pitfall 3, no off-by-one).
#[must_use]
pub fn route_categorical(cardinality: u32, one_hot_max_size: u32) -> EncodingPath {
    if cardinality <= 1 {
        // A constant (or empty) column carries no split information
        // (`AddOneHotFeatures` skip-if-`<=1`).
        EncodingPath::Skip
    } else if cardinality <= one_hot_max_size {
        // `1 < cardinality <= one_hot_max_size` (inclusive boundary): one-hot.
        EncodingPath::OneHot
    } else {
        // `cardinality > one_hot_max_size` (exclusive boundary): CTR (deferred).
        EncodingPath::Ctr
    }
}

/// Count a categorical column's LEARN-SET distinct-value cardinality via the
/// single categorical-hash source ([`cb_data::calc_cat_feature_hash`] +
/// [`cb_data::PerfectHash`]) — the `GetUniqueValuesCounts(...).OnLearnOnly`
/// equivalent (RESEARCH Pitfall 3). Each value is hashed to its ui32 CityHash64
/// reduction and remapped to a first-seen dense bin; the cardinality is the
/// number of distinct bins (`PerfectHash::len`).
///
/// `column` holds the categorical values already in the A4 string form the
/// hasher expects (integer-coded values pre-stringified via
/// [`cb_data::stringify_int_category`] by the caller).
///
/// # Errors
///
/// Propagates [`cb_core::CbError::OutOfRange`] from [`cb_data::PerfectHash::remap`]
/// if the column has more than `u32::MAX` distinct values (no panic).
pub fn learn_set_cardinality(column: &[&str]) -> CbResult<u32> {
    let mut ph = PerfectHash::new();
    for &value in column {
        let hash = calc_cat_feature_hash(value);
        ph.remap(hash)?;
    }
    // `PerfectHash::len` is the distinct-bin count, bounded to `u32::MAX` by the
    // remap guard above, so the `as u32` is loss-free.
    Ok(ph.len() as u32)
}

/// Route a categorical column directly from its raw learn-set values: count its
/// distinct-value cardinality ([`learn_set_cardinality`]) then route it
/// ([`route_categorical`]). The single-call convenience the tree-growth caller
/// uses per categorical feature.
///
/// # Errors
///
/// Propagates [`cb_core::CbError::OutOfRange`] from [`learn_set_cardinality`].
pub fn route_column(column: &[&str], one_hot_max_size: u32) -> CbResult<EncodingPath> {
    let cardinality = learn_set_cardinality(column)?;
    Ok(route_categorical(cardinality, one_hot_max_size))
}
