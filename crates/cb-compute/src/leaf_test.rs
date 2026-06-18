//! Unit tests for leaf-value estimation primitives (TRAIN-03 all four methods).

use crate::leaf::{
    build_monotonic_linear_orders, calc_average, calc_monotonic_leaf_deltas, exact_leaf_delta,
    gradient_leaf_delta, logcosh_exact_leaf_delta, newton_leaf_delta, scale_l2_reg,
    simple_leaf_delta, solve_symmetric_newton,
};

#[test]
fn solve_symmetric_newton_reproduces_hessian_cpp_three_class() {
    // A leaf with a single 3-class object at approx = [0,0,0], target_class = 1:
    //   der1 = [-1/3, 2/3, -1/3] (softmax_ders der1)
    //   packed Hessian der2 = [p_y(p_y-1) diag, p_y*p_x off] with p = 1/3:
    //     [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)] = [-2/9, 1/9, 1/9, -2/9, 1/9, -2/9]
    // Solving hessian.cpp:22-52 (trace-eps at f32 precision) with scaled_l2 = 3.0
    // gives the leaf delta [-0.1, 0.2, -0.1] (verified against numpy.linalg.solve).
    let third = 1.0 / 3.0;
    let der1 = [-third, 2.0 * third, -third];
    let diag = third * (third - 1.0); // -2/9
    let off = third * third; // 1/9
    let der2_packed = [diag, off, off, diag, off, diag];
    let delta = solve_symmetric_newton(&der1, &der2_packed, 3.0);
    let want = [-0.1, 0.2, -0.1];
    for d in 0..3 {
        assert!(
            (delta[d] - want[d]).abs() < 1e-9,
            "delta[{d}] = {} want {}",
            delta[d],
            want[d]
        );
    }
}

#[test]
fn solve_symmetric_newton_empty_leaf_returns_zeros() {
    // An all-zero leaf (empty / degenerate) returns zeros, not NaN (T-6.2-01: no
    // panic / div-by-zero). With der1 = [0,0] and a zero Hessian, the negated
    // matrix is `adjustedL2·I` (SPD), so the solve yields exactly 0.
    let delta = solve_symmetric_newton(&[0.0, 0.0], &[0.0, 0.0, 0.0], 3.0);
    assert_eq!(delta.len(), 2);
    for &d in &delta {
        assert_eq!(d, 0.0);
    }
}

#[test]
fn calc_average_guards_empty_leaf_returns_zero() {
    // count == 0 -> 0.0, no division by zero / panic (T-03-01-01).
    assert_eq!(calc_average(5.0, 0.0, 3.0), 0.0);
    assert_eq!(calc_average(0.0, 0.0, 0.0), 0.0);
    // A negative-or-zero count is also degenerate -> 0.0.
    assert_eq!(calc_average(5.0, -1.0, 3.0), 0.0);
}

#[test]
fn calc_average_l2_regularized_average() {
    // sumDer=-2.92126*23 path mirrored: just the formula sum/(count+l2).
    let v = calc_average(10.0, 4.0, 3.0);
    assert!((v - 10.0 / 7.0).abs() < 1e-12);
}

#[test]
fn scale_l2_reg_unweighted_is_l2() {
    // sum_all_weights == doc_count (every weight 1.0) -> scaledL2 == l2.
    assert!((scale_l2_reg(3.0, 50.0, 50) - 3.0).abs() < 1e-12);
}

#[test]
fn scale_l2_reg_weighted_scales_by_mean_weight() {
    // sum_all_weights/doc_count = 100/50 = 2.0 -> 3.0 * 2.0 = 6.0
    assert!((scale_l2_reg(3.0, 100.0, 50) - 6.0).abs() < 1e-12);
}

#[test]
fn scale_l2_reg_zero_doc_count_returns_l2() {
    assert!((scale_l2_reg(3.0, 0.0, 0) - 3.0).abs() < 1e-12);
}

#[test]
fn gradient_leaf_delta_matches_oracle_leaf0() {
    // Tree-0 leaf-0 of regression_skeleton: sumDer=-2.92126*(23+3) numerator;
    // verified in the plan against model.json: delta = -2.92126 for
    // sumDer=-75.9528, count=23, scaledL2=3.0 (cnt 23 -> /26).
    let sum_der = -2.921261_f64 * 26.0; // delta * (count + scaledL2)
    let delta = gradient_leaf_delta(sum_der, 23.0, 3.0);
    assert!((delta - (-2.921261)).abs() < 1e-5);
}

#[test]
fn newton_leaf_delta_formula() {
    // sum_der / (-sum_der2 + scaledL2). With sum_der2 = -4 (e.g. Logloss-like),
    // denom = 4 + 3 = 7, so 10/7.
    let v = newton_leaf_delta(10.0, -4.0, 3.0);
    assert!((v - 10.0 / 7.0).abs() < 1e-12);
}

#[test]
fn newton_equals_gradient_for_rmse_hessian() {
    // RMSE der2 == -1 per object, so sum_der2 == -sum_weight => -sum_der2 ==
    // sum_weight, making Newton's denom == Gradient's denom (sum_weight+scaledL2).
    let sum_der = -75.95;
    let sum_weight = 23.0;
    let scaled_l2 = 3.0;
    let sum_der2 = -sum_weight; // RMSE: sum of (-1)*weight over 23 unit-weight objs.
    let g = gradient_leaf_delta(sum_der, sum_weight, scaled_l2);
    let n = newton_leaf_delta(sum_der, sum_der2, scaled_l2);
    assert!((g - n).abs() < 1e-12);
}

#[test]
fn newton_guards_only_exact_zero_denominator() {
    // der2 == 0 (MAE/Quantile) with scaledL2 == 0 -> denom exactly 0 -> guarded 0.0
    // (the ONLY guard — avoids the 0/0 NaN; T-03-02-01).
    assert_eq!(newton_leaf_delta(5.0, 0.0, 0.0), 0.0);
    // Empty leaf: sum_der2 == 0, scaledL2 == 0 -> 0.0, no div-by-zero.
    assert_eq!(newton_leaf_delta(0.0, 0.0, 0.0), 0.0);
    // A NEGATIVE denominator (-sum_der2 + scaledL2 < 0) must DIVIDE verbatim like
    // upstream `CalcDeltaNewtonBody` (online_predictor.h:162-170), NOT be clamped:
    // the listwise LambdaMart loss fills a strictly POSITIVE hessian, so its
    // denominator is legitimately negative and upstream divides by it (LOSS-04
    // Wave B; clamping it to 0 zeroed every LambdaMart leaf). denom = -4 + 1 = -3;
    // 5/-3 = -1.6667.
    assert!((newton_leaf_delta(5.0, 4.0, 1.0) - (-5.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn simple_equals_gradient_delta() {
    // A6: Simple dispatches to the Gradient branch upstream (CalcLeafDeltasSimple).
    let s = simple_leaf_delta(10.0, 4.0, 3.0);
    let g = gradient_leaf_delta(10.0, 4.0, 3.0);
    assert!((s - g).abs() < 1e-12);
}

#[test]
fn exact_leaf_delta_unweighted_median_odd_count() {
    // Residuals {-3, -1, 2}; alpha=0.5 weighted (unit weights) -> needWeight=1.5;
    // sorted [-3,-1,2], running weights [1,2,3]; first >= 1.5-eps is the 2nd
    // element (-1). delta adjustment: q=-1, less={-3}(w=1), equal={-1}(w=1):
    // lessWeight + equalWeight*alpha = 1 + 0.5 = 1.5 >= needWeight (1.5) -> q-=delta.
    let v = exact_leaf_delta(&[2.0, -3.0, -1.0], &[1.0, 1.0, 1.0], 0.5, 1e-6);
    assert!((v - (-1.0 - 1e-6)).abs() < 1e-9, "got {v}");
}

#[test]
fn exact_leaf_delta_empty_leaf_returns_zero() {
    assert_eq!(exact_leaf_delta(&[], &[], 0.5, 1e-6), 0.0);
}

#[test]
fn exact_leaf_delta_weighted_quantile() {
    // Residuals {0.0(w=3), 5.0(w=1)}; total weight 4, alpha=0.5 -> needWeight=2.
    // sorted [0(w3),5(w1)] running [3,4]; first >=2-eps is 0.0. delta: q=0,
    // less={}, equal={0(w3)}: 0 + 3*0.5 = 1.5 < 2 -> q += delta.
    let v = exact_leaf_delta(&[5.0, 0.0], &[1.0, 3.0], 0.5, 1e-6);
    assert!((v - (0.0 + 1e-6)).abs() < 1e-9, "got {v}");
}

#[test]
fn exact_leaf_delta_alpha_zero_is_min() {
    // alpha <= 0 -> min element (CalcSampleQuantile early return).
    let v = exact_leaf_delta(&[5.0, -2.0, 3.0], &[1.0, 1.0, 1.0], 0.0, 1e-6);
    assert!((v - (-2.0)).abs() < 1e-9, "got {v}");
}

#[test]
fn logcosh_exact_leaf_delta_empty_leaf_returns_zero() {
    assert_eq!(logcosh_exact_leaf_delta(&[], &[]), 0.0);
}

#[test]
fn logcosh_exact_leaf_delta_symmetric_residuals_is_zero() {
    // Σ tanh(δ - r) = 0 for symmetric residuals {-1, +1} at δ = 0 (tanh is odd).
    // The bracket is [-1, 1]; the bisection converges to ~0 within 1e-9.
    let v = logcosh_exact_leaf_delta(&[-1.0, 1.0], &[]);
    assert!(v.abs() < 1e-8, "got {v}");
}

#[test]
fn logcosh_exact_leaf_delta_matches_bisection_reference() {
    // Reproduce the upstream bisection (optimal_const_for_loss.h:110-154) over an
    // asymmetric residual set with weights; the root of g(δ)=Σ w·tanh(δ-r) must
    // match the production fn bit-for-bit (same op order, same return-`left`).
    let residuals: [f32; 4] = [-2.0, 0.5, 1.0, 3.0];
    let weights = [1.0_f64, 2.0, 0.5, 1.5];

    let g = |approx: f64| -> f64 {
        residuals
            .iter()
            .zip(weights.iter())
            .map(|(&r, &w)| (approx - f64::from(r)).tanh() * w)
            .sum::<f64>()
    };
    let mut left = -2.0_f64;
    let mut right = 3.0_f64;
    for _ in 0..100 {
        if (right - left) <= 1e-9 {
            break;
        }
        let m = (left + right) / 2.0;
        if g(m) > 0.0 {
            right = m;
        } else {
            left = m;
        }
    }

    let v = logcosh_exact_leaf_delta(&residuals, &weights);
    assert!((v - left).abs() < 1e-9, "got {v}, want {left}");
}

// --- Monotone-constraint isotonic (PAVA) projection (FEAT-03) ---------------

/// `build_monotonic_linear_orders` for a single monotonic feature (+1) on a
/// depth-1 tree reproduces `BuildMonotonicLinearOrdersOnLeafs`
/// (`monotonic_constraint_utils.cpp:38-52`): one non-monotonic feature count = 0
/// → exactly one subtree; its order over the 2 leaves is `[leaf0, leaf1]` (the
/// least-leaf is 0 for a `+1` constraint, then leafRank 1 XORs in bit `1`).
#[test]
fn build_monotonic_linear_orders_single_increasing_split() {
    let orders = build_monotonic_linear_orders(&[1]);
    assert_eq!(orders, vec![vec![0u32, 1u32]]);
}

/// A single DECREASING (`-1`) split flips the least-leaf: `leastLeafIndex |=
/// bit` so the order starts at leaf 1 and descends to leaf 0
/// (`cpp:19-21`). The PAVA over this order then enforces NON-INCREASING values
/// along the split.
#[test]
fn build_monotonic_linear_orders_single_decreasing_split() {
    let orders = build_monotonic_linear_orders(&[-1]);
    assert_eq!(orders, vec![vec![1u32, 0u32]]);
}

/// Two splits, the FIRST non-monotonic (0) and the SECOND monotonic (+1):
/// `nonMonotonicFeatureCount == 1` → 2 subtrees. Split 0 is bit `1`, split 1 is
/// bit `2`. Subtree 0 (the non-monotonic bit = 0) orders leaves `[0, 2]`;
/// subtree 1 (the non-monotonic bit = 1) orders leaves `[1, 3]` — each subtree
/// is the monotonic order along the second split within a fixed first-split
/// value (`cpp:4-52`).
#[test]
fn build_monotonic_linear_orders_one_free_one_increasing() {
    let orders = build_monotonic_linear_orders(&[0, 1]);
    assert_eq!(orders, vec![vec![0u32, 2u32], vec![1u32, 3u32]]);
}

/// Empty constraints → exactly one trivial subtree containing every leaf in
/// natural order; the PAVA over it is a NO-OP for any value vector. (Upstream
/// `BuildMonotonicLinearOrdersOnLeafs([])` returns `[[0]]` for a 0-split tree;
/// for the no-op path the caller never reaches PAVA — see the no-op test.)
#[test]
fn calc_monotonic_leaf_deltas_empty_constraints_is_noop() {
    let curr = [0.0_f64, 0.0, 0.0, 0.0];
    let deltas = [0.5_f64, -0.3, 0.9, 0.1];
    let weights = [4.0_f64, 4.0, 4.0, 4.0];
    let out = calc_monotonic_leaf_deltas(&[], &curr, &deltas, &weights);
    assert_eq!(out, deltas.to_vec());
}

/// A depth-1 increasing constraint where the raw deltas VIOLATE monotonicity
/// (leaf0 delta > leaf1 delta). PAVA pools the two equal-weight leaves to their
/// weighted average; the returned deltas, added to `curr` (== 0), must be
/// NON-DECREASING along `[0, 1]`. With equal weights the pooled value is the
/// arithmetic mean `(1.0 + 0.0)/2 = 0.5`, so both deltas become `0.5`.
#[test]
fn calc_monotonic_leaf_deltas_pools_increasing_violation() {
    let curr = [0.0_f64, 0.0];
    let deltas = [1.0_f64, 0.0]; // violates +1 (leaf0 > leaf1)
    let weights = [3.0_f64, 3.0];
    let out = calc_monotonic_leaf_deltas(&[1], &curr, &deltas, &weights);
    assert!((out[0] - 0.5).abs() < 1e-12, "out[0] = {}", out[0]);
    assert!((out[1] - 0.5).abs() < 1e-12, "out[1] = {}", out[1]);
    // Monotone non-decreasing along the order [0, 1].
    assert!(out[0] <= out[1] + 1e-12);
}

/// Weighted pooling: an increasing violation with UNEQUAL weights pools to the
/// WEIGHTED average (`SumWeightedValue / Weight`, `TIsotonicLevelSet::Average`).
/// leaf0 (value 1.0, weight 1) and leaf1 (value 0.0, weight 3) pool to
/// `(1*1 + 3*0)/(1+3) = 0.25`.
#[test]
fn calc_monotonic_leaf_deltas_weighted_pool() {
    let curr = [0.0_f64, 0.0];
    let deltas = [1.0_f64, 0.0];
    let weights = [1.0_f64, 3.0];
    let out = calc_monotonic_leaf_deltas(&[1], &curr, &deltas, &weights);
    assert!((out[0] - 0.25).abs() < 1e-12, "out[0] = {}", out[0]);
    assert!((out[1] - 0.25).abs() < 1e-12, "out[1] = {}", out[1]);
}

/// An ALREADY-monotone increasing delta vector is returned UNCHANGED (PAVA never
/// merges when each new level-set average exceeds the previous).
#[test]
fn calc_monotonic_leaf_deltas_already_monotone_unchanged() {
    let curr = [0.0_f64, 0.0];
    let deltas = [-0.2_f64, 0.7];
    let weights = [5.0_f64, 5.0];
    let out = calc_monotonic_leaf_deltas(&[1], &curr, &deltas, &weights);
    assert!((out[0] - (-0.2)).abs() < 1e-12);
    assert!((out[1] - 0.7).abs() < 1e-12);
}

/// A decreasing (`-1`) constraint enforces NON-INCREASING in leaf index. The
/// linear order is `[1, 0]` (larger leaf first), so PAVA enforces
/// `value[leaf1] <= value[leaf0]`. A raw pair leaf0=0.0, leaf1=1.0 VIOLATES that
/// (leaf1 > leaf0): along the order `[1, 0]` the sequence is 1.0 then 0.0
/// (decreasing) → PAVA pools the two equal-weight leaves to their mean 0.5.
#[test]
fn calc_monotonic_leaf_deltas_decreasing_pools_violation() {
    let curr = [0.0_f64, 0.0];
    let deltas = [0.0_f64, 1.0]; // leaf0=0.0, leaf1=1.0 — violates -1 (leaf1 > leaf0)
    let weights = [2.0_f64, 2.0];
    let out = calc_monotonic_leaf_deltas(&[-1], &curr, &deltas, &weights);
    // Order [1,0]: values 1.0 (leaf1) then 0.0 (leaf0) decreasing → pool to 0.5.
    assert!((out[0] - 0.5).abs() < 1e-12, "out[0] = {}", out[0]);
    assert!((out[1] - 0.5).abs() < 1e-12, "out[1] = {}", out[1]);
}

/// The projection operates on `curr + delta` and returns the ADJUSTED DELTA
/// (`updatedLeafValues[l] - currLeafValues[l]`, `approx_calcer.cpp:587-589`).
/// With a non-zero `curr`, an already-monotone TOTAL is unchanged so the delta
/// is returned verbatim.
#[test]
fn calc_monotonic_leaf_deltas_returns_delta_relative_to_curr() {
    let curr = [10.0_f64, 20.0];
    let deltas = [0.1_f64, 0.2]; // totals 10.1, 20.2 — monotone increasing
    let weights = [4.0_f64, 4.0];
    let out = calc_monotonic_leaf_deltas(&[1], &curr, &deltas, &weights);
    assert!((out[0] - 0.1).abs() < 1e-12);
    assert!((out[1] - 0.2).abs() < 1e-12);
}
