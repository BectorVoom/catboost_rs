//! Self-oracle unit tests for the pairwise split-scoring subsystem
//! (`pairwise_scoring.rs`). Source/test separation per INFRA-06 (CLAUDE.md): no
//! inline `#[cfg(test)]` body in the production file — this sibling file is linked
//! via the `#[path = "pairwise_scoring_test.rs"] mod tests;` footer.
//!
//! Each test hand-derives a small reference matrix and asserts the transcription
//! reproduces it bit-for-bit (der-sums / weight statistics) or to ≤1e-9 (the
//! Cholesky-backed per-split score).

use super::*;

/// Fixture geometry shared across the der-sum / weight-statistics tests:
/// 3 docs, 2 leaves, 3 buckets.
///   doc0 → leaf 0, bucket 1
///   doc1 → leaf 1, bucket 0
///   doc2 → leaf 0, bucket 2
const LEAF_OF: [usize; 3] = [0, 1, 0];
const BUCKET_OF: [usize; 3] = [1, 0, 2];
const LEAF_COUNT: usize = 2;
const BUCKET_COUNT: usize = 3;

#[test]
fn der_sums_scatter_matches_reference() {
    // weighted_der per doc.
    let weighted_der = [0.5_f64, -0.3, 0.2];

    let got = compute_der_sums(&weighted_der, LEAF_COUNT, BUCKET_COUNT, &LEAF_OF, &BUCKET_OF)
        .expect("der sums");

    // Hand-derived reference:
    //   der_sums[0][1] += 0.5 (doc0), der_sums[0][2] += 0.2 (doc2)
    //   der_sums[1][0] += -0.3 (doc1)
    let expected = vec![vec![0.0, 0.5, 0.2], vec![-0.3, 0.0, 0.0]];
    assert_eq!(got, expected, "der-sum scatter must match hand-derived matrix");
}

#[test]
fn der_sums_rejects_out_of_range_index() {
    let weighted_der = [1.0_f64];
    // leaf index 5 >= leaf_count 2 → typed error, never a panic.
    let bad_leaf = [5_usize];
    let bucket = [0_usize];
    let err = compute_der_sums(&weighted_der, LEAF_COUNT, BUCKET_COUNT, &bad_leaf, &bucket)
        .expect_err("out-of-range leaf must error");
    matches!(err, cb_core::CbError::OutOfRange(_))
        .then_some(())
        .expect("expected OutOfRange");
}

#[test]
fn pair_weight_statistics_winner_loser_bucket_order() {
    // Pairs (winner_doc, loser_doc, weight) covering both winnerBucket>loserBucket
    // and winnerBucket<loserBucket branches plus a winner==loser skip.
    let pairs = [
        (0usize, 1usize, 1.0f64), // winnerBucket 1 > loserBucket 0  -> branch A
        (2, 1, 2.0),              // winnerBucket 2 > loserBucket 0  -> branch A
        (1, 0, 0.5),              // winnerBucket 0 < loserBucket 1  -> else branch
        (0, 0, 9.0),              // winner == loser                 -> skip
    ];

    let stats =
        compute_pair_weight_statistics(&pairs, LEAF_COUNT, BUCKET_COUNT, &LEAF_OF, &BUCKET_OF)
            .expect("weight statistics");

    // Hand-derived reference: every non-skipped pair lands in stats[1][0]
    // (loserLeaf=1,winnerLeaf=0 for branch A; winnerLeaf=1,loserLeaf=0 for else).
    //   bucket0.smaller = -1.0 -2.0 -0.5 = -3.5
    //   bucket1.greater = -1.0 (pair0,1) -0.5 (pair1,0) = -1.5
    //   bucket2.greater = -2.0 (pair2,1)
    let s10 = &stats[1][0];
    assert!((s10[0].smaller_border_weight_sum - (-3.5)).abs() < 1e-12);
    assert!((s10[0].greater_border_right_weight_sum - 0.0).abs() < 1e-12);
    assert!((s10[1].smaller_border_weight_sum - 0.0).abs() < 1e-12);
    assert!((s10[1].greater_border_right_weight_sum - (-1.5)).abs() < 1e-12);
    assert!((s10[2].smaller_border_weight_sum - 0.0).abs() < 1e-12);
    assert!((s10[2].greater_border_right_weight_sum - (-2.0)).abs() < 1e-12);

    // The winner==loser pair (0,0,9.0) must NOT have mutated stats[0][*].
    for leaf_b in 0..LEAF_COUNT {
        for b in 0..BUCKET_COUNT {
            let c = &stats[0][leaf_b][b];
            assert_eq!(c.smaller_border_weight_sum, 0.0);
            assert_eq!(c.greater_border_right_weight_sum, 0.0);
        }
    }
}

#[test]
fn pair_weight_statistics_rejects_out_of_range_doc() {
    // Pair references doc 9 which is out of range for the 3-element index arrays.
    let pairs = [(0usize, 9usize, 1.0f64)];
    let err =
        compute_pair_weight_statistics(&pairs, LEAF_COUNT, BUCKET_COUNT, &LEAF_OF, &BUCKET_OF)
            .expect_err("out-of-range loser doc must error");
    matches!(err, cb_core::CbError::OutOfRange(_))
        .then_some(())
        .expect("expected OutOfRange");
}

/// Default reg priors (`oblivious_tree_options.cpp:15-16`): `L2Reg = 3.0`,
/// `PairwiseNonDiagReg = bayesian_matrix_reg = 0.1`.
const L2_DIAG_REG: f64 = 3.0;
const PRIOR_REG: f64 = 0.1;

#[test]
fn single_leaf_two_bucket_score_uses_cholesky_leaf() {
    // Degenerate single-leaf, 2-bucket case → the 2×2 closed-form leaf solve.
    // der_sums[0] = [d0, d1]; stats[0][0][0] supplies the weightDelta for split 0.
    let d0 = 1.0_f64;
    let d1 = 0.5_f64;
    let der_sums = vec![vec![d0, d1]];

    // stats[0][0][0].smaller = -2.0, .greater = 0.0 → weight_delta = -2.0.
    let mut stats = vec![vec![vec![BucketPairWeightStatistics::default(); 2]; 1]; 1];
    stats[0][0][0].smaller_border_weight_sum = -2.0;
    stats[0][0][0].greater_border_right_weight_sum = 0.0;

    let scores = calculate_pairwise_score(&der_sums, &stats, 2, L2_DIAG_REG, PRIOR_REG)
        .expect("score");
    assert_eq!(scores.len(), 1, "bucket_count-1 == 1 score");

    // Independent closed-form re-derivation (system_size == 2):
    //   der_sum after step 2+4: [d0, d1]; weight_delta wd = smaller - greater = -2.0.
    //   weightSum = [[-wd, wd],[wd, -wd]]; diag_reg = PRIOR*(1-1/2) + L2.
    //   x0 = der_sum[0] / (weightSum[0][0] + diag_reg) = d0 / (-wd + diag_reg).
    //   avrg = MakeZeroAverage([x0, 0]) = [x0/2, -x0/2].
    //   score = Σ_x avrg[x]·(der[x] − 0.5·Σ_y avrg[y]·weightSum[x][y]).
    let wd = -2.0_f64;
    let diag_reg = PRIOR_REG * 0.5 + L2_DIAG_REG;
    let x0 = d0 / (-wd + diag_reg);
    let avrg = [x0 / 2.0, -x0 / 2.0];
    let w = [[-wd, wd], [wd, -wd]];
    let der = [d0, d1];
    let mut expected = 0.0_f64;
    for x in 0..2 {
        let inner: f64 = (0..2).map(|y| avrg[y] * w[x][y]).sum();
        expected += avrg[x] * (der[x] - 0.5 * inner);
    }

    assert!(scores[0].is_finite(), "degenerate solve must be finite, not NaN");
    assert!(
        (scores[0] - expected).abs() < 1e-9,
        "single-leaf score {} != closed-form reference {}",
        scores[0],
        expected
    );
}

/// An independent, transparent re-implementation of the OneFeature pairwise score
/// algorithm (`pairwise_scoring.cpp:140-232`) used as the self-oracle reference.
/// Deliberately written in a different style (no shared helpers with the production
/// code) so a transcription bug in either side is caught.
fn reference_pairwise_score(
    der_sums: &[Vec<f64>],
    stats: &[Vec<Vec<BucketPairWeightStatistics>>],
    bucket_count: usize,
    l2: f64,
    prior: f64,
) -> Vec<f64> {
    let leaf_count = der_sums.len();
    let n = 2 * leaf_count;
    let mut weight_sum = vec![vec![0.0_f64; n]; n];
    let mut der_sum = vec![0.0_f64; n];

    // Step 2.
    for leaf in 0..leaf_count {
        let mut s = 0.0;
        for b in 0..bucket_count {
            s += der_sums[leaf][b];
        }
        der_sum[2 * leaf + 1] += s;
    }
    // Step 3.
    for y in 0..leaf_count {
        for x in (y + 1)..leaf_count {
            let mut total = 0.0;
            for b in 0..bucket_count {
                total += stats[x][y][b].smaller_border_weight_sum
                    + stats[y][x][b].smaller_border_weight_sum;
            }
            weight_sum[2 * y + 1][2 * x + 1] += total;
            weight_sum[2 * x + 1][2 * y + 1] += total;
            weight_sum[2 * x + 1][2 * x + 1] -= total;
            weight_sum[2 * y + 1][2 * y + 1] -= total;
        }
    }
    // Step 4.
    let mut scores = Vec::new();
    for split in 0..(bucket_count - 1) {
        for y in 0..leaf_count {
            let der_delta = der_sums[y][split];
            der_sum[2 * y] += der_delta;
            der_sum[2 * y + 1] -= der_delta;
            let wd = stats[y][y][split].smaller_border_weight_sum
                - stats[y][y][split].greater_border_right_weight_sum;
            weight_sum[2 * y][2 * y + 1] += wd;
            weight_sum[2 * y + 1][2 * y] += wd;
            weight_sum[2 * y][2 * y] -= wd;
            weight_sum[2 * y + 1][2 * y + 1] -= wd;
            for x in (y + 1)..leaf_count {
                let xy = stats[x][y][split];
                let yx = stats[y][x][split];
                let w00 = xy.greater_border_right_weight_sum + yx.greater_border_right_weight_sum;
                let w01 = xy.smaller_border_weight_sum - xy.greater_border_right_weight_sum;
                let w10 = yx.smaller_border_weight_sum - yx.greater_border_right_weight_sum;
                let w11 = -(xy.smaller_border_weight_sum + yx.smaller_border_weight_sum);
                weight_sum[2 * x][2 * y] += w00;
                weight_sum[2 * y][2 * x] += w00;
                weight_sum[2 * x][2 * y + 1] += w01;
                weight_sum[2 * y + 1][2 * x] += w01;
                weight_sum[2 * x + 1][2 * y] += w10;
                weight_sum[2 * y][2 * x + 1] += w10;
                weight_sum[2 * x + 1][2 * y + 1] += w11;
                weight_sum[2 * y + 1][2 * x + 1] += w11;
                weight_sum[2 * y][2 * y] -= w00 + w10;
                weight_sum[2 * x][2 * x] -= w00 + w01;
                weight_sum[2 * x + 1][2 * x + 1] -= w10 + w11;
                weight_sum[2 * y + 1][2 * y + 1] -= w01 + w11;
            }
        }
        let leaf_values = reference_leaf_solve(&weight_sum, &der_sum, l2, prior);
        // CalculateScore.
        let mut score = 0.0;
        for x in 0..n {
            let mut inner = 0.0;
            for y in 0..n {
                inner += leaf_values[y] * weight_sum[x][y];
            }
            score += leaf_values[x] * (der_sum[x] - 0.5 * inner);
        }
        scores.push(score);
    }
    scores
}

/// Independent reference leaf solve (the 2×2 / general Cholesky path of
/// `CalculatePairwiseLeafValues`) via a nalgebra-free Gaussian elimination on the
/// `(n-1)×(n-1)` regularized system — a DIFFERENT solver than the production
/// Cholesky, so the two agreeing validates the production path.
fn reference_leaf_solve(
    weight_sum: &[Vec<f64>],
    der_sum: &[f64],
    l2: f64,
    prior: f64,
) -> Vec<f64> {
    let n = der_sum.len();
    let cell_prior = 1.0 / n as f64;
    let non_diag = -prior * cell_prior;
    let diag = prior * (1.0 - cell_prior) + l2;
    if n == 2 {
        let x0 = der_sum[0] / (weight_sum[0][0] + diag);
        let mut res = vec![x0, 0.0];
        let mean = (res[0] + res[1]) / 2.0;
        for v in &mut res {
            *v -= mean;
        }
        return res;
    }
    let m = n - 1;
    let mut a = vec![vec![0.0_f64; m]; m];
    for y in 0..m {
        for x in 0..y {
            let v = weight_sum[y][x] + non_diag;
            a[y][x] = v;
            a[x][y] = v;
        }
        a[y][y] = weight_sum[y][y] + diag;
    }
    let mut b: Vec<f64> = der_sum[..m].to_vec();
    // Gaussian elimination with partial pivoting.
    for col in 0..m {
        let mut piv = col;
        for r in (col + 1)..m {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-300 {
            return vec![0.0; n];
        }
        for r in (col + 1)..m {
            let f = a[r][col] / d;
            for c in col..m {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0_f64; m];
    for i in (0..m).rev() {
        let mut s = b[i];
        for c in (i + 1)..m {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    let mut res = x;
    res.push(0.0);
    let mean: f64 = res.iter().sum::<f64>() / res.len() as f64;
    for v in &mut res {
        *v -= mean;
    }
    res
}

#[test]
fn calculate_pairwise_score_matches_hand_derived_first_candidate() {
    // 2-leaf, 3-bucket case with non-trivial cross-leaf statistics.
    let der_sums = vec![vec![0.4_f64, -0.2, 0.1], vec![-0.3, 0.25, 0.05]];

    let mut stats =
        vec![vec![vec![BucketPairWeightStatistics::default(); 3]; 2]; 2];
    // Cross-leaf statistics (stats[0][1] and stats[1][0]) + self-leaf diagonals.
    stats[0][1][0].smaller_border_weight_sum = -1.0;
    stats[0][1][1].greater_border_right_weight_sum = -1.0;
    stats[1][0][0].smaller_border_weight_sum = -2.0;
    stats[1][0][2].greater_border_right_weight_sum = -2.0;
    stats[0][0][0].smaller_border_weight_sum = -0.5;
    stats[1][1][1].smaller_border_weight_sum = -0.75;
    stats[1][1][1].greater_border_right_weight_sum = -0.25;

    let scores = calculate_pairwise_score(&der_sums, &stats, 3, L2_DIAG_REG, PRIOR_REG)
        .expect("score");
    let reference = reference_pairwise_score(&der_sums, &stats, 3, L2_DIAG_REG, PRIOR_REG);

    assert_eq!(scores.len(), 2, "bucket_count-1 == 2 scores");
    assert_eq!(reference.len(), 2);
    for (i, (&got, &want)) in scores.iter().zip(reference.iter()).enumerate() {
        assert!(got.is_finite(), "score[{i}] must be finite");
        assert!(
            (got - want).abs() < 1e-9,
            "score[{i}] {got} != reference {want} (diff {})",
            (got - want).abs()
        );
    }
}

#[test]
fn calculate_pairwise_score_rejects_inconsistent_shape() {
    // der_sums width (2) disagrees with bucket_count (3).
    let der_sums = vec![vec![0.1_f64, 0.2]];
    let stats = vec![vec![vec![BucketPairWeightStatistics::default(); 3]; 1]; 1];
    let err = calculate_pairwise_score(&der_sums, &stats, 3, L2_DIAG_REG, PRIOR_REG)
        .expect_err("inconsistent width must error");
    matches!(err, cb_core::CbError::OutOfRange(_))
        .then_some(())
        .expect("expected OutOfRange");
}
