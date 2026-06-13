//! Unit tests for leaf-value estimation primitives (TRAIN-03 all four methods).

use crate::leaf::{
    calc_average, exact_leaf_delta, gradient_leaf_delta, newton_leaf_delta, scale_l2_reg,
    simple_leaf_delta,
};

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
fn newton_guards_degenerate_denominator_returns_zero() {
    // der2 == 0 (MAE/Quantile) with scaledL2 == 0 -> denom 0 -> guarded 0.0.
    assert_eq!(newton_leaf_delta(5.0, 0.0, 0.0), 0.0);
    // Empty leaf: sum_der2 == 0, scaledL2 == 0 -> 0.0, no div-by-zero (T-03-02-01).
    assert_eq!(newton_leaf_delta(0.0, 0.0, 0.0), 0.0);
    // A negative denominator (-sum_der2 + scaledL2 < 0) is also degenerate -> 0.0.
    assert_eq!(newton_leaf_delta(5.0, 4.0, 1.0), 0.0);
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
