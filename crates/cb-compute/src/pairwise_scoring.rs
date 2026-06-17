//! Pairwise split-scoring subsystem (LOSS-04, Rule-4 architectural piece) — the
//! `OneFeature` (float-feature) path of catboost 1.2.10's `TPairwiseScoreCalcer`
//! / `CalculatePairwiseScore`
//! (`catboost-master/catboost/private/libs/algo/pairwise_scoring.{h,cpp}`).
//!
//! # Why a separate split path
//!
//! `*Pairwise` losses (`IsPairwiseScoring`: `YetiRankPairwise`, `PairLogitPairwise`,
//! `QueryCrossEntropy`) score candidate splits through the per-(leaf,leaf,bucket)
//! pair-weight statistics + a per-split Cholesky leaf solve, NOT the pointwise der
//! histogram the L2/Cosine split path reuses (`enum_helpers.cpp:469-475`). This is
//! the SPLIT-SELECTION divergence the 06.3-13/06.3-14 verification isolated as the
//! deferral cause for both `PairLogitPairwise` and `YetiRankPairwise`.
//!
//! This file is a PURE LIBRARY: it lands the scored-candidate primitive and
//! self-oracles it against hand-derived references. It adds NO tree-search wiring
//! (Plan 06.3-16) and freezes NO fixtures.
//!
//! # OneFeature only
//!
//! The float-only frozen ranking corpus needs no `BinarySplits` /
//! `ExclusiveFeaturesBundle` / `FeaturesGroup` branches; this ports ONLY the
//! `OneFeature` arm (`pairwise_scoring.cpp:140-232`). The caller supplies
//! `bucket_of[doc]` (the candidate float feature's bucket index per doc), so the
//! der-sum / weight-statistics fns are feature-agnostic.
//!
//! # No new crate
//!
//! The per-split leaf solve reuses the in-house Cholesky routine vendored in
//! `cb-compute/src/leaf.rs` ([`crate::pairwise_cholesky_solve`]) — the SAME
//! primitive `cb_train::calculate_pairwise_leaf_values` calls (RESEARCH Open Q1
//! RESOLVED). It returns `None` on a non-positive pivot so a degenerate solve
//! falls back to zeros, never a NaN/panic (T-06.3-15-02).
//!
//! # Summation discipline (D-08)
//!
//! Every der-sum / weight-sum / score accumulation that occupies a REDUCTION
//! position routes through `cb_core::sum_f64` (the strict left-to-right f64 fold,
//! `reduction.rs`). The per-cell scatter accumulations (`der_sums[leaf][bucket]`,
//! the pair-weight `-= weight` decrements) replicate upstream's `+=`/`-=` order
//! (doc-ascending, pair-ascending) and carry a doc-comment naming that order as the
//! parity contract (T-06.3-15-03).
//!
//! # Bounds safety (T-06.3-15-01)
//!
//! Leaf/bucket indices arrive from the trainer (a trust boundary). A malformed
//! index must not panic a library: every scatter index is bounds-guarded via
//! `.get`/`.get_mut`, an out-of-range index surfaces a typed `cb_core::CbError`
//! (`OutOfRange`), never `unwrap`/`expect`/`panic`/raw-index (CLAUDE.md Security V5).

use cb_core::{CbError, CbResult};

/// Per-(winner-leaf, loser-leaf, bucket) pair-weight statistics — mirrors
/// `TBucketPairWeightStatistics` (`pairwise_scoring.h:16-27`).
///
/// `smaller_border_weight_sum` is the weight sum of pair elements with the SMALLER
/// border; `greater_border_right_weight_sum` the weight sum of pair elements with
/// the GREATER border. Both accumulate as negative decrements in
/// [`compute_pair_weight_statistics`] (upstream's `-= weight`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BucketPairWeightStatistics {
    /// The weight sum of pair elements with the smaller border.
    pub smaller_border_weight_sum: f64,
    /// The weight sum of pair elements with the greater border.
    pub greater_border_right_weight_sum: f64,
}

impl BucketPairWeightStatistics {
    /// Element-wise merge, mirroring `TBucketPairWeightStatistics::Add`
    /// (`pairwise_scoring.h:23-26`). Used when merging per-range partial
    /// statistics; the single-thread library never splits ranges, but the method
    /// is provided to keep the struct a faithful analogue. (Named `merge` rather
    /// than `add` to avoid `clippy::should_implement_trait` confusion with
    /// `std::ops::Add` — this is not a numeric addition operator.)
    #[must_use]
    pub fn merge(self, rhs: Self) -> Self {
        Self {
            smaller_border_weight_sum: self.smaller_border_weight_sum + rhs.smaller_border_weight_sum,
            greater_border_right_weight_sum: self.greater_border_right_weight_sum
                + rhs.greater_border_right_weight_sum,
        }
    }
}

/// Scatter `weighted_der[doc]` into `der_sums[leaf_of[doc]][bucket_of[doc]]`,
/// mirroring `ComputeDerSums` (`pairwise_scoring.h:52-68`).
///
/// Result is `[leaf_count][bucket_count]`, zero where unhit. The per-cell `+=`
/// replicates upstream's doc-ascending accumulation order exactly (the summation
/// order IS the parity contract); the result is bit-identical to upstream's
/// `derSums[leafIndex][bucketIndex] += weightedDerivativesData[docId]`.
///
/// # Errors
///
/// Returns [`CbError::OutOfRange`] if any `leaf_of[doc] >= leaf_count`,
/// `bucket_of[doc] >= bucket_count`, or the per-doc slices disagree in length
/// (a malformed index from the trainer trust boundary — never a panic,
/// T-06.3-15-01).
pub fn compute_der_sums(
    weighted_der: &[f64],
    leaf_count: usize,
    bucket_count: usize,
    leaf_of: &[usize],
    bucket_of: &[usize],
) -> CbResult<Vec<Vec<f64>>> {
    if leaf_of.len() != weighted_der.len() || bucket_of.len() != weighted_der.len() {
        return Err(CbError::OutOfRange(format!(
            "compute_der_sums: leaf_of ({}), bucket_of ({}) must match weighted_der ({})",
            leaf_of.len(),
            bucket_of.len(),
            weighted_der.len()
        )));
    }
    let mut der_sums = vec![vec![0.0_f64; bucket_count]; leaf_count];
    // Doc-ascending scatter (upstream order = parity contract): each cell is an
    // in-place accumulation, NOT a reduction over a collected Vec, so the raw `+=`
    // here mirrors `pairwise_scoring.h:65` verbatim (D-08 exception is scoped to
    // the documented per-cell scatter only).
    for (doc, &der) in weighted_der.iter().enumerate() {
        let leaf = leaf_of.get(doc).copied().unwrap_or(leaf_count);
        let bucket = bucket_of.get(doc).copied().unwrap_or(bucket_count);
        let cell = der_sums
            .get_mut(leaf)
            .and_then(|row| row.get_mut(bucket))
            .ok_or_else(|| {
                CbError::OutOfRange(format!(
                    "compute_der_sums: doc {doc} has leaf {leaf} (<{leaf_count}) / bucket {bucket} (<{bucket_count}) out of range"
                ))
            })?;
        *cell += der;
    }
    Ok(der_sums)
}

/// Build the `[leaf][leaf][bucket]` pair-weight statistics from the flat pair list,
/// mirroring `ComputePairWeightStatistics` (`pairwise_scoring.h:72-103`),
/// `OneFeature` branch.
///
/// Each pair is `(winner_doc, loser_doc, weight)`. For each pair with
/// `winner != loser`, the `winnerBucket > loserBucket` vs else branch decrements
/// `smaller_border_weight_sum` / `greater_border_right_weight_sum` exactly as
/// `pairwise_scoring.h:93-99` (loser-leaf vs winner-leaf ordering preserved):
///
/// ```text
/// if winnerBucket > loserBucket:
///     stats[loserLeaf][winnerLeaf][loserBucket].smaller        -= weight
///     stats[loserLeaf][winnerLeaf][winnerBucket].greater_right -= weight
/// else:
///     stats[winnerLeaf][loserLeaf][winnerBucket].smaller        -= weight
///     stats[winnerLeaf][loserLeaf][loserBucket].greater_right   -= weight
/// ```
///
/// `winner == loser` pairs are skipped (no statistics mutation). The `-= weight`
/// scatter replicates upstream's pair-ascending order (the summation order IS the
/// parity contract).
///
/// # Errors
///
/// Returns [`CbError::OutOfRange`] if any pair's winner/loser doc index, or its
/// derived leaf/bucket index, is out of range (T-06.3-15-01 — never a panic).
pub fn compute_pair_weight_statistics(
    pairs: &[(usize, usize, f64)],
    leaf_count: usize,
    bucket_count: usize,
    leaf_of: &[usize],
    bucket_of: &[usize],
) -> CbResult<Vec<Vec<Vec<BucketPairWeightStatistics>>>> {
    let mut weight_sums =
        vec![
            vec![vec![BucketPairWeightStatistics::default(); bucket_count]; leaf_count];
            leaf_count
        ];

    // Pair-ascending scatter (upstream order = parity contract).
    for &(winner_doc, loser_doc, weight) in pairs {
        if winner_doc == loser_doc {
            continue;
        }
        let winner_bucket = lookup(bucket_of, winner_doc, "winner bucket")?;
        let winner_leaf = lookup(leaf_of, winner_doc, "winner leaf")?;
        let loser_bucket = lookup(bucket_of, loser_doc, "loser bucket")?;
        let loser_leaf = lookup(leaf_of, loser_doc, "loser leaf")?;

        if winner_bucket > loser_bucket {
            decrement_smaller(&mut weight_sums, loser_leaf, winner_leaf, loser_bucket, weight)?;
            decrement_greater(&mut weight_sums, loser_leaf, winner_leaf, winner_bucket, weight)?;
        } else {
            decrement_smaller(&mut weight_sums, winner_leaf, loser_leaf, winner_bucket, weight)?;
            decrement_greater(&mut weight_sums, winner_leaf, loser_leaf, loser_bucket, weight)?;
        }
    }

    Ok(weight_sums)
}

/// Bounds-guarded lookup of an index array, surfacing a typed `OutOfRange` rather
/// than a panic (T-06.3-15-01).
fn lookup(indices: &[usize], doc: usize, what: &str) -> CbResult<usize> {
    indices.get(doc).copied().ok_or_else(|| {
        CbError::OutOfRange(format!(
            "compute_pair_weight_statistics: {what} doc index {doc} out of range (len {})",
            indices.len()
        ))
    })
}

/// `stats[leaf_a][leaf_b][bucket].smaller_border_weight_sum -= weight`, bounds-guarded.
fn decrement_smaller(
    stats: &mut [Vec<Vec<BucketPairWeightStatistics>>],
    leaf_a: usize,
    leaf_b: usize,
    bucket: usize,
    weight: f64,
) -> CbResult<()> {
    let cell = stats
        .get_mut(leaf_a)
        .and_then(|m| m.get_mut(leaf_b))
        .and_then(|row| row.get_mut(bucket))
        .ok_or_else(|| {
            CbError::OutOfRange(format!(
                "compute_pair_weight_statistics: smaller cell [{leaf_a}][{leaf_b}][{bucket}] out of range"
            ))
        })?;
    cell.smaller_border_weight_sum -= weight;
    Ok(())
}

/// `stats[leaf_a][leaf_b][bucket].greater_border_right_weight_sum -= weight`,
/// bounds-guarded.
fn decrement_greater(
    stats: &mut [Vec<Vec<BucketPairWeightStatistics>>],
    leaf_a: usize,
    leaf_b: usize,
    bucket: usize,
    weight: f64,
) -> CbResult<()> {
    let cell = stats
        .get_mut(leaf_a)
        .and_then(|m| m.get_mut(leaf_b))
        .and_then(|row| row.get_mut(bucket))
        .ok_or_else(|| {
            CbError::OutOfRange(format!(
                "compute_pair_weight_statistics: greater cell [{leaf_a}][{leaf_b}][{bucket}] out of range"
            ))
        })?;
    cell.greater_border_right_weight_sum -= weight;
    Ok(())
}

#[cfg(test)]
#[path = "pairwise_scoring_test.rs"]
mod tests;
