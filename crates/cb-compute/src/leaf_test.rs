//! Unit tests for leaf-value estimation primitives (TRAIN-03 Gradient).

use crate::leaf::{calc_average, gradient_leaf_delta, scale_l2_reg};

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
