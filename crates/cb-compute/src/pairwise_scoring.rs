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

use cb_core::{sum_f64, CbError, CbResult};

use crate::pairwise_cholesky_solve;

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

/// `MakeZeroAverage` (`catboost/libs/helpers/matrix.h:5-15`): subtract the mean so
/// the leaf deltas are zero-centered. The mean accumulation routes through
/// `cb_core::sum_f64` per D-08 — the single sanctioned strict left-to-right f64
/// fold (`sum_f64` IS upstream's sequential `average += res[i]` order).
fn make_zero_average(res: &mut [f64]) {
    let n = res.len();
    if n == 0 {
        return;
    }
    let average = sum_f64(res) / n as f64;
    for v in res.iter_mut() {
        *v -= average;
    }
}

/// Solve one per-split leaf system over the local `2*leaf_count` weight matrix —
/// a cb-compute-local twin of `CalculatePairwiseLeafValues`
/// (`pairwise_leaves_calculation.cpp:9-52`), the SAME routine
/// `cb_train::calculate_pairwise_leaf_values` calls. It cannot depend on cb-train
/// (layering), so the matrix-assembly + 2×2 closed form + `MakeZeroAverage` are
/// transcribed here and the general case reuses the in-house
/// [`crate::pairwise_cholesky_solve`] (no new crate).
///
/// `weight_sum` is the `n × n` SPD weight matrix (`n = 2*leaf_count`), `der_sum`
/// the length-`n` der vector, `l2_diag_reg = L2Reg`,
/// `pairwise_bucket_weight_prior_reg = PairwiseNonDiagReg`. Returns the `n`
/// zero-averaged leaf values. A non-positive pivot (cholesky → `None`) falls back
/// to zeros (T-06.3-15-02: no NaN/panic).
#[must_use]
fn calculate_pairwise_leaf_values(
    weight_sum: &[Vec<f64>],
    der_sum: &[f64],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> Vec<f64> {
    let system_size = der_sum.len();
    if system_size == 0 {
        return Vec::new();
    }
    let cell_prior = 1.0 / system_size as f64;
    let non_diag_reg = -pairwise_bucket_weight_prior_reg * cell_prior;
    let diag_reg = pairwise_bucket_weight_prior_reg * (1.0 - cell_prior) + l2_diag_reg;

    if system_size == 1 {
        // Degenerate singleton: MakeZeroAverage of a single element is 0.
        return vec![0.0];
    }

    if system_size == 2 {
        // 2×2 closed form (pairwise_leaves_calculation.cpp:25-34):
        //   res = { derSum[0] / (weightSum[0][0] + diagReg), 0 }; MakeZeroAverage.
        let a11 = weight_sum
            .first()
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(0.0);
        let denom = a11 + diag_reg;
        let x0 = if denom != 0.0 {
            der_sum.first().copied().unwrap_or(0.0) / denom
        } else {
            0.0
        };
        let mut res = vec![x0, 0.0];
        make_zero_average(&mut res);
        return res;
    }

    // General case: build the (n-1)×(n-1) SPD matrix (upper-tri + reg priors,
    // pairwise_leaves_calculation.cpp:36-43). pairwise_cholesky_solve reconstructs
    // the symmetric matrix from the full row-major matrix, so fill BOTH triangles.
    let m = system_size - 1;
    let mut matrix = vec![vec![0.0_f64; m]; m];
    for y in 0..m {
        for x in 0..y {
            let v = weight_sum
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
        let diag = weight_sum
            .get(y)
            .and_then(|r| r.get(y))
            .copied()
            .unwrap_or(0.0)
            + diag_reg;
        if let Some(c) = matrix.get_mut(y).and_then(|r| r.get_mut(y)) {
            *c = diag;
        }
    }

    let rhs: Vec<f64> = der_sum.iter().take(m).copied().collect();
    let mut res = match pairwise_cholesky_solve(&matrix, &rhs) {
        Some(x) => x,
        // Non-positive pivot → all-zeros fallback (T-06.3-15-02: no NaN/panic).
        None => vec![0.0; m],
    };
    res.push(0.0);
    make_zero_average(&mut res);
    res
}

/// Score one split border: `Σ_x avrg[x]·(sumDer[x] − 0.5·Σ_y avrg[y]·weightSum[x][y])`,
/// mirroring `TPairwiseScoreCalcer::CalculateScore` (`pairwise_scoring.cpp:51-81`).
///
/// The SIMD `FusedMultiplyAdd` inner fold is replaced by a scalar
/// `avrg[y]·weightSum[x][y]` product collected into a `Vec` and reduced through
/// `cb_core::sum_f64` (single-thread parity — no SIMD needed; D-08 reduction order).
fn calculate_score(avrg: &[f64], sum_der: &[f64], weight_sum: &[Vec<f64>]) -> f64 {
    let n = sum_der.len();
    let mut outer: Vec<f64> = Vec::with_capacity(n);
    for x in 0..n {
        let avrg_x = avrg.get(x).copied().unwrap_or(0.0);
        let der_x = sum_der.get(x).copied().unwrap_or(0.0);
        // Inner fold Σ_y avrg[y]·weightSum[x][y] via sum_f64.
        let products: Vec<f64> = (0..n)
            .map(|y| {
                let avrg_y = avrg.get(y).copied().unwrap_or(0.0);
                let w_xy = weight_sum
                    .get(x)
                    .and_then(|row| row.get(y))
                    .copied()
                    .unwrap_or(0.0);
                avrg_y * w_xy
            })
            .collect();
        let sub_score = sum_f64(&products);
        outer.push(avrg_x * (der_x - 0.5 * sub_score));
    }
    sum_f64(&outer)
}

/// `UpdateWeightSumFromTotal` (`pairwise_scoring.cpp:84-90`): fold the off-diagonal
/// leaf-pair total into the `2y+1 / 2x+1` block. Bounds-guarded `get_mut`.
fn update_weight_sum_from_total(y: usize, x: usize, total: f64, weight_sum: &mut [Vec<f64>]) {
    add_at(weight_sum, 2 * y + 1, 2 * x + 1, total);
    add_at(weight_sum, 2 * x + 1, 2 * y + 1, total);
    add_at(weight_sum, 2 * x + 1, 2 * x + 1, -total);
    add_at(weight_sum, 2 * y + 1, 2 * y + 1, -total);
}

/// `UpdateWeightSumFromNonDiagStats` (`pairwise_scoring.cpp:93-120`): apply the per-
/// split off-diagonal weight deltas from the `xy`/`yx` bucket statistics.
fn update_weight_sum_from_non_diag_stats(
    y: usize,
    x: usize,
    xy: BucketPairWeightStatistics,
    yx: BucketPairWeightStatistics,
    weight_sum: &mut [Vec<f64>],
) {
    let w00 = xy.greater_border_right_weight_sum + yx.greater_border_right_weight_sum;
    let w01 = xy.smaller_border_weight_sum - xy.greater_border_right_weight_sum;
    let w10 = yx.smaller_border_weight_sum - yx.greater_border_right_weight_sum;
    let w11 = -(xy.smaller_border_weight_sum + yx.smaller_border_weight_sum);

    add_at(weight_sum, 2 * x, 2 * y, w00);
    add_at(weight_sum, 2 * y, 2 * x, w00);
    add_at(weight_sum, 2 * x, 2 * y + 1, w01);
    add_at(weight_sum, 2 * y + 1, 2 * x, w01);
    add_at(weight_sum, 2 * x + 1, 2 * y, w10);
    add_at(weight_sum, 2 * y, 2 * x + 1, w10);
    add_at(weight_sum, 2 * x + 1, 2 * y + 1, w11);
    add_at(weight_sum, 2 * y + 1, 2 * x + 1, w11);

    add_at(weight_sum, 2 * y, 2 * y, -(w00 + w10));
    add_at(weight_sum, 2 * x, 2 * x, -(w00 + w01));
    add_at(weight_sum, 2 * x + 1, 2 * x + 1, -(w10 + w11));
    add_at(weight_sum, 2 * y + 1, 2 * y + 1, -(w01 + w11));
}

/// `weight_sum[r][c] += delta`, bounds-guarded (a malformed leaf index from the
/// trainer never panics; an out-of-range cell is silently skipped because every
/// `r`/`c` here is derived from in-range `0..2*leaf_count` indices — the guard is
/// the Security-V5 belt-and-suspenders, T-06.3-15-01).
fn add_at(weight_sum: &mut [Vec<f64>], r: usize, c: usize, delta: f64) {
    if let Some(cell) = weight_sum.get_mut(r).and_then(|row| row.get_mut(c)) {
        *cell += delta;
    }
}

/// Compute one pairwise split-score per candidate border (`0..bucket_count-1`),
/// mirroring `CalculatePairwiseScore` + `TPairwiseScoreCalcer::CalculateScore`
/// (`pairwise_scoring.cpp:140-232`, `OneFeature` branch).
///
/// Inputs:
/// - `der_sums`: `[leaf_count][bucket_count]` from [`compute_der_sums`].
/// - `pair_weight_statistics`: `[leaf][leaf][bucket]` from
///   [`compute_pair_weight_statistics`].
/// - `bucket_count`: number of buckets; produces `bucket_count - 1` scores.
/// - `l2_diag_reg = l2_leaf_reg` and
///   `pairwise_bucket_weight_prior_reg = bayesian_matrix_reg` (default `0.1`) — the
///   regularization priors the per-split Cholesky leaf solve uses
///   (`scoring.cpp:839-847`; `oblivious_tree_options.cpp:15-16`: `L2Reg` default
///   `3.0`, `PairwiseNonDiagReg` default `0.1`).
///
/// Algorithm (`pairwise_scoring.cpp:144-231`):
/// 1. Build the `2*leaf_count` square `weight_sum` matrix and `der_sum` vector.
/// 2. Seed `der_sum[2*leaf+1] += Σ_bucket der_sums[leaf][bucket]`.
/// 3. Apply `UpdateWeightSumFromTotal` once per off-diagonal leaf pair (`y<x`),
///    with `total = Σ_bucket (xy.smaller + yx.smaller)` over all buckets.
/// 4. For each split border `splitId` (0..bucket_count-1), accumulate the per-split
///    `derDelta`/`weightDelta`/`UpdateWeightSumFromNonDiagStats` deltas onto the
///    RUNNING `der_sum`/`weight_sum` (they carry across borders), call the per-split
///    Cholesky leaf solve, then `CalculateScore`.
///
/// A degenerate (non-positive-pivot) per-split solve yields the zeros the leaf
/// solver returns, so its score is `0.0`, NOT a NaN (T-06.3-15-02).
///
/// # Errors
///
/// Returns [`CbError::OutOfRange`] if `der_sums` / `pair_weight_statistics`
/// dimensions are inconsistent with `bucket_count` / each other (T-06.3-15-01).
pub fn calculate_pairwise_score(
    der_sums: &[Vec<f64>],
    pair_weight_statistics: &[Vec<Vec<BucketPairWeightStatistics>>],
    bucket_count: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> CbResult<Vec<f64>> {
    let leaf_count = der_sums.len();
    if bucket_count == 0 {
        return Ok(Vec::new());
    }
    // Validate der_sums row widths.
    for (leaf, row) in der_sums.iter().enumerate() {
        if row.len() != bucket_count {
            return Err(CbError::OutOfRange(format!(
                "calculate_pairwise_score: der_sums[{leaf}] has width {} != bucket_count {bucket_count}",
                row.len()
            )));
        }
    }
    // Validate pair_weight_statistics shape: [leaf_count][leaf_count][bucket_count].
    if pair_weight_statistics.len() != leaf_count {
        return Err(CbError::OutOfRange(format!(
            "calculate_pairwise_score: pair_weight_statistics outer len {} != leaf_count {leaf_count}",
            pair_weight_statistics.len()
        )));
    }
    for (a, plane) in pair_weight_statistics.iter().enumerate() {
        if plane.len() != leaf_count {
            return Err(CbError::OutOfRange(format!(
                "calculate_pairwise_score: pair_weight_statistics[{a}] len {} != leaf_count {leaf_count}",
                plane.len()
            )));
        }
        for (b, row) in plane.iter().enumerate() {
            if row.len() != bucket_count {
                return Err(CbError::OutOfRange(format!(
                    "calculate_pairwise_score: pair_weight_statistics[{a}][{b}] width {} != bucket_count {bucket_count}",
                    row.len()
                )));
            }
        }
    }

    let system_size = 2 * leaf_count;
    let mut weight_sum = vec![vec![0.0_f64; system_size]; system_size];
    let mut der_sum = vec![0.0_f64; system_size];

    // Step 2: der_sum[2*leaf+1] += Σ_bucket der_sums[leaf][bucket].
    for (leaf, row) in der_sums.iter().enumerate() {
        let total = sum_f64(row);
        add_to(&mut der_sum, 2 * leaf + 1, total);
    }

    // Step 3: UpdateWeightSumFromTotal per off-diagonal leaf pair (y<x).
    for y in 0..leaf_count {
        for x in (y + 1)..leaf_count {
            // total = Σ_bucket (xy.smaller + yx.smaller), xy = stats[x][y], yx = stats[y][x].
            let xy_row = stats_row(pair_weight_statistics, x, y);
            let yx_row = stats_row(pair_weight_statistics, y, x);
            let mut bucket_terms: Vec<f64> = Vec::with_capacity(2 * bucket_count);
            for b in 0..bucket_count {
                bucket_terms.push(smaller_at(xy_row, b));
                bucket_terms.push(smaller_at(yx_row, b));
            }
            let total = sum_f64(&bucket_terms);
            update_weight_sum_from_total(y, x, total, &mut weight_sum);
        }
    }

    // Step 4: per-split running deltas + solve + score.
    let n_splits = bucket_count - 1;
    let mut scores = vec![0.0_f64; n_splits];
    for split_id in 0..n_splits {
        for y in 0..leaf_count {
            let der_delta = der_sums
                .get(y)
                .and_then(|row| row.get(split_id))
                .copied()
                .unwrap_or(0.0);
            add_to(&mut der_sum, 2 * y, der_delta);
            add_to(&mut der_sum, 2 * y + 1, -der_delta);

            let diag = stats_at(pair_weight_statistics, y, y, split_id);
            let weight_delta = diag.smaller_border_weight_sum - diag.greater_border_right_weight_sum;
            add_at(&mut weight_sum, 2 * y, 2 * y + 1, weight_delta);
            add_at(&mut weight_sum, 2 * y + 1, 2 * y, weight_delta);
            add_at(&mut weight_sum, 2 * y, 2 * y, -weight_delta);
            add_at(&mut weight_sum, 2 * y + 1, 2 * y + 1, -weight_delta);

            for x in (y + 1)..leaf_count {
                let xy = stats_at(pair_weight_statistics, x, y, split_id);
                let yx = stats_at(pair_weight_statistics, y, x, split_id);
                update_weight_sum_from_non_diag_stats(y, x, xy, yx, &mut weight_sum);
            }
        }

        let leaf_values = calculate_pairwise_leaf_values(
            &weight_sum,
            &der_sum,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
        );
        let score = calculate_score(&leaf_values, &der_sum, &weight_sum);
        if let Some(slot) = scores.get_mut(split_id) {
            *slot = score;
        }
    }

    Ok(scores)
}

/// `der_sum[idx] += delta`, bounds-guarded (idx is always in `0..2*leaf_count`).
fn add_to(der_sum: &mut [f64], idx: usize, delta: f64) {
    if let Some(slot) = der_sum.get_mut(idx) {
        *slot += delta;
    }
}

/// Read the `[a][b]` bucket-statistics row (empty slice if out of range).
fn stats_row(
    stats: &[Vec<Vec<BucketPairWeightStatistics>>],
    a: usize,
    b: usize,
) -> &[BucketPairWeightStatistics] {
    stats
        .get(a)
        .and_then(|plane| plane.get(b))
        .map_or(&[][..], |row| row.as_slice())
}

/// `stats[a][b][bucket]` (default if out of range).
fn stats_at(
    stats: &[Vec<Vec<BucketPairWeightStatistics>>],
    a: usize,
    b: usize,
    bucket: usize,
) -> BucketPairWeightStatistics {
    stats
        .get(a)
        .and_then(|plane| plane.get(b))
        .and_then(|row| row.get(bucket))
        .copied()
        .unwrap_or_default()
}

/// `row[bucket].smaller_border_weight_sum` (0 if out of range).
fn smaller_at(row: &[BucketPairWeightStatistics], bucket: usize) -> f64 {
    row.get(bucket)
        .map(|s| s.smaller_border_weight_sum)
        .unwrap_or(0.0)
}

#[cfg(test)]
#[path = "pairwise_scoring_test.rs"]
mod tests;
