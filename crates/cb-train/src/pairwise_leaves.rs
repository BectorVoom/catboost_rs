//! Cholesky pairwise-leaf solve (LOSS-04 Wave B) — the dedicated leaf-value path
//! the `*Pairwise` ranking losses (`IsPairwiseScoring`) require, transcribed from
//! catboost 1.2.10
//! `catboost-master/catboost/private/libs/algo_helpers/pairwise_leaves_calculation.cpp`
//! (+ the `ComputePairwiseWeightSums` matrix assembly and `CalcLeafDeltasSimple`
//! caller, `approx_calcer.cpp:470-501`).
//!
//! # Why a separate path
//!
//! Only the `*Pairwise` variants (here [`Loss::PairLogitPairwise`]) solve their
//! leaf VALUES as an `(n-1)×(n-1)` SPD linear system over the per-leaf pairwise
//! weight sums + der sums (`IsPairwiseScoring`, `enum_helpers.cpp:469-475`). The
//! pointwise PairLogit/LambdaMart variants use the existing Gradient/Newton
//! estimators (RESEARCH Pitfall 2 — mis-routing diverges leaf values). The boosting
//! loop gates this path on `cb_compute::is_pairwise_scoring`.
//!
//! # No new crate
//!
//! The SPD solve reuses the in-house Cholesky routine already vendored in
//! `cb-compute/src/leaf.rs` ([`cb_compute::pairwise_cholesky_solve`]) — RESEARCH
//! Open Q1 RESOLVED, no linear-algebra dependency added. It returns `None` on a
//! non-positive pivot so the leaf falls back to zeros, never a NaN/panic
//! (T-06.3-03-01). Every `weightSums` / `derSums` accumulation routes through
//! `cb_core::sum_f64` (D-08 — no raw float fold).
//!
//! # Reg priors
//!
//! `diagReg = pairwiseBucketWeightPriorReg·(1 - 1/n) + l2DiagReg`,
//! `nonDiagReg = -pairwiseBucketWeightPriorReg/n` (`pairwise_leaves_calculation.cpp:20-22`),
//! where `l2DiagReg = L2Reg` and `pairwiseBucketWeightPriorReg = PairwiseNonDiagReg`
//! (`bayesian_matrix_reg`, default `0.1`; `approx_calcer.cpp:490-501`).

use cb_compute::pairwise_cholesky_solve;
use cb_compute::GroupSpan;
use cb_core::sum_f64;

/// Compute the `leafCount × leafCount` pairwise weight-sum matrix from the grouped
/// `competitors` adjacency and the per-object leaf assignment `leaf_of`
/// (`ComputePairwiseWeightSums`, `pairwise_leaves_calculation.cpp:54-97`).
///
/// For each group's winner→loser pair, with `winnerLeaf = leaf_of[winner_global]`
/// and `loserLeaf = leaf_of[loser_global]` (skipping same-leaf pairs):
/// ```text
/// sum[winnerLeaf][loserLeaf] -= weight; sum[loserLeaf][winnerLeaf] -= weight;
/// sum[winnerLeaf][winnerLeaf] += weight; sum[loserLeaf][loserLeaf] += weight;
/// ```
/// The matrix is row-major `leaf_count × leaf_count`. The `-= / +=` scatter mirrors
/// upstream's accumulation order exactly (group-ascending, doc-ascending,
/// competitor-order) — the summation order IS the parity contract.
#[must_use]
fn compute_pairwise_weight_sums(
    groups: &[GroupSpan],
    leaf_of: &[usize],
    leaf_count: usize,
) -> Vec<Vec<f64>> {
    let mut sums = vec![vec![0.0_f64; leaf_count]; leaf_count];
    for group in groups {
        let begin = group.begin;
        for (winner_local, comps) in group.competitors.iter().enumerate() {
            let winner_global = begin + winner_local;
            let winner_leaf = leaf_of.get(winner_global).copied().unwrap_or(0);
            for competitor in comps {
                let loser_global = begin + competitor.id;
                let loser_leaf = leaf_of.get(loser_global).copied().unwrap_or(0);
                if winner_leaf == loser_leaf {
                    continue;
                }
                let w = competitor.weight;
                if let Some(c) = sums.get_mut(winner_leaf).and_then(|r| r.get_mut(loser_leaf)) {
                    *c -= w;
                }
                if let Some(c) = sums.get_mut(loser_leaf).and_then(|r| r.get_mut(winner_leaf)) {
                    *c -= w;
                }
                if let Some(c) = sums.get_mut(winner_leaf).and_then(|r| r.get_mut(winner_leaf)) {
                    *c += w;
                }
                if let Some(c) = sums.get_mut(loser_leaf).and_then(|r| r.get_mut(loser_leaf)) {
                    *c += w;
                }
            }
        }
    }
    sums
}

/// Per-leaf der sums `derSums[leaf] = Σ der1[member]` over the leaf's members
/// (`approx_calcer.cpp:493-496`, `leafDers[leaf].SumDer`). The PairLogit der1 is
/// NOT pre-multiplied by the per-object weight (the pair weight lives inside the
/// der), so this is a plain per-leaf reduction routed through `cb_core::sum_f64`
/// (D-08, member order ascending).
#[must_use]
fn compute_der_sums(leaf_of: &[usize], der1: &[f64], leaf_count: usize) -> Vec<f64> {
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); leaf_count];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if let (Some(bucket), Some(&d)) = (members.get_mut(leaf), der1.get(i)) {
            bucket.push(d);
        }
    }
    members.iter().map(|m| sum_f64(m)).collect()
}

/// Solve one leaf-system via the in-house Cholesky path
/// (`CalculatePairwiseLeafValues`, `pairwise_leaves_calculation.cpp:9-52`).
///
/// `pairwise_weight_sums` is the `n × n` SPD weight matrix (`n = leaf_count`),
/// `der_sums` the per-leaf der sums, `l2_diag_reg = L2Reg`,
/// `pairwise_bucket_weight_prior_reg = PairwiseNonDiagReg`. Returns the `n`
/// zero-averaged leaf deltas. The `systemSize == 2` closed form and the general
/// `(n-1)×(n-1)` Cholesky case + `MakeZeroAverage` are transcribed verbatim; a
/// non-positive pivot (cholesky_solve → `None`) falls back to zeros (no NaN/panic).
#[must_use]
pub fn calculate_pairwise_leaf_values(
    pairwise_weight_sums: &[Vec<f64>],
    der_sums: &[f64],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> Vec<f64> {
    let system_size = der_sums.len();
    if system_size == 0 {
        return Vec::new();
    }
    let cell_prior = 1.0 / system_size as f64;
    let non_diag_reg = -pairwise_bucket_weight_prior_reg * cell_prior;
    let diag_reg = pairwise_bucket_weight_prior_reg * (1.0 - cell_prior) + l2_diag_reg;

    if system_size == 1 {
        // A degenerate single-leaf system: the only delta is zero after
        // MakeZeroAverage (mean-subtract of a singleton is 0). Upstream asserts
        // GetXSize() > 1, but the trainer can hit a single-leaf depth-0 tree; keep
        // it safe (no panic) and return the zero-centered delta.
        return vec![0.0];
    }

    if system_size == 2 {
        // 2×2 closed form (pairwise_leaves_calculation.cpp:25-34):
        //   res = { derSums[0] / (weightSums[0][0] + diagReg), 0 }; MakeZeroAverage.
        let a11 = pairwise_weight_sums
            .first()
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(0.0);
        let denom = a11 + diag_reg;
        let x0 = if denom != 0.0 {
            der_sums.first().copied().unwrap_or(0.0) / denom
        } else {
            0.0
        };
        let mut res = vec![x0, 0.0];
        make_zero_average(&mut res);
        return res;
    }

    // General case: build the (n-1)×(n-1) SPD matrix (upper-tri + reg priors,
    // pairwise_leaves_calculation.cpp:36-43). cholesky_solve reconstructs the
    // symmetric matrix from the full row-major `a`, so fill BOTH triangles.
    let m = system_size - 1;
    let mut matrix = vec![vec![0.0_f64; m]; m];
    for y in 0..m {
        for x in 0..y {
            let v = pairwise_weight_sums
                .get(y)
                .and_then(|r| r.get(x))
                .copied()
                .unwrap_or(0.0)
                + non_diag_reg;
            if let Some(c) = matrix.get_mut(y).and_then(|r| r.get_mut(x)) {
                *c = v;
            }
            if let Some(c) = matrix.get_mut(x).and_then(|r| r.get_mut(y)) {
                *c = v;
            }
        }
        let diag = pairwise_weight_sums
            .get(y)
            .and_then(|r| r.get(y))
            .copied()
            .unwrap_or(0.0)
            + diag_reg;
        if let Some(c) = matrix.get_mut(y).and_then(|r| r.get_mut(y)) {
            *c = diag;
        }
    }

    // res = derSums[..n-1]; solve; push 0; MakeZeroAverage.
    let rhs: Vec<f64> = der_sums.iter().take(m).copied().collect();
    let mut res = match pairwise_cholesky_solve(&matrix, &rhs) {
        Some(x) => x,
        // Non-positive pivot → fall back to all-zeros (T-06.3-03-01: no NaN/panic).
        None => vec![0.0; m],
    };
    res.push(0.0);
    make_zero_average(&mut res);
    res
}

/// `MakeZeroAverage` (`catboost/libs/helpers/matrix.h:5-15`): subtract the mean so
/// the leaf deltas are zero-centered. A sequential accumulation matching upstream's
/// `average += res[i]` loop order (the parity contract is the loop order).
fn make_zero_average(res: &mut [f64]) {
    let n = res.len();
    if n == 0 {
        return;
    }
    let mut average = 0.0_f64;
    for &v in res.iter() {
        average += v;
    }
    average /= n as f64;
    for v in res.iter_mut() {
        *v -= average;
    }
}

/// The full pairwise leaf-delta path: assemble the weight-sum matrix + der sums
/// from the grouped view and per-object leaf assignment, then solve. Returns the
/// `leaf_count` zero-averaged leaf deltas (BEFORE the `learning_rate` scale the
/// caller applies). This is the entry point `boosting.rs` calls for the
/// `*Pairwise` losses.
#[must_use]
pub fn compute_pairwise_leaf_deltas(
    groups: &[GroupSpan],
    leaf_of: &[usize],
    der1: &[f64],
    leaf_count: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> Vec<f64> {
    if leaf_count == 0 {
        return Vec::new();
    }
    let weight_sums = compute_pairwise_weight_sums(groups, leaf_of, leaf_count);
    let der_sums = compute_der_sums(leaf_of, der1, leaf_count);
    calculate_pairwise_leaf_values(
        &weight_sums,
        &der_sums,
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg,
    )
}

#[cfg(test)]
#[path = "pairwise_leaves_test.rs"]
mod tests;
