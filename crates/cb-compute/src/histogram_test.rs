//! Unit tests for the host-side ordered bucket reduction (D-02/D-05).

use crate::histogram::{
    collect_leaf_residuals, reduce_leaf_der2, reduce_leaf_stats, reduce_leaf_stats_newton,
    LeafStats,
};

#[test]
fn reduce_leaf_stats_groups_by_leaf() {
    // 4 objects, 2 leaves. leaf 0 = {obj0, obj2}, leaf 1 = {obj1, obj3}.
    let leaf_of = [0usize, 1, 0, 1];
    let der1 = [1.0, 2.0, 3.0, 4.0];
    let weight = [1.0, 1.0, 1.0, 1.0];
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, 2);
    assert_eq!(stats.len(), 2);
    assert_eq!(
        stats[0],
        LeafStats {
            sum_weighted_delta: 4.0, // 1.0 + 3.0
            sum_weight: 2.0,
        }
    );
    assert_eq!(
        stats[1],
        LeafStats {
            sum_weighted_delta: 6.0, // 2.0 + 4.0
            sum_weight: 2.0,
        }
    );
}

#[test]
fn reduce_leaf_stats_empty_leaf_is_zero() {
    // All objects fall in leaf 0; leaf 1 stays empty -> zero stats.
    let leaf_of = [0usize, 0, 0];
    let der1 = [1.0, 2.0, 3.0];
    let weight = [1.0, 1.0, 1.0];
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, 2);
    assert_eq!(stats[1], LeafStats::default());
    assert_eq!(stats[0].sum_weighted_delta, 6.0);
    assert_eq!(stats[0].sum_weight, 3.0);
}

#[test]
fn reduce_leaf_stats_ordered_determinism() {
    // The reduction must fold in ascending object order (D-05). Build a leaf
    // whose member order matters under the naive sequential sum: the adversarial
    // [1e16, 1.0, -1e16] sequence sums to 0.0 left-to-right (the 1.0 is lost).
    let leaf_of = [0usize, 0, 0];
    let der1 = [1e16, 1.0, -1e16];
    let weight = [1.0, 1.0, 1.0];
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, 1);
    // Object order preserved -> sequential fold loses the 1.0, exactly as
    // cb_core::sum_f64 (the parity contract) does.
    assert_eq!(stats[0].sum_weighted_delta, 0.0);
}

#[test]
fn reduce_leaf_stats_newton_fills_positive_hessian_in_weight_slot() {
    // Newton fill (D-6.4-03): `sum_weighted_delta` is the gradient sum (as in
    // reduce_leaf_stats), but `sum_weight` carries the POSITIVE summed hessian
    // Σ(-der2*weight) — der2 is ≤0 (RMSE=-1, Logloss=-p(1-p)), negated to +.
    // leaf 0 = {obj0, obj2}, leaf 1 = {obj1, obj3}.
    let leaf_of = [0usize, 1, 0, 1];
    let der1 = [1.0, 2.0, 3.0, 4.0];
    // weighted der2 = der2 * weight, all ≤0 (e.g. RMSE der2=-1 * weight).
    let weighted_der2 = [-1.0, -0.5, -1.0, -0.25];
    let stats = reduce_leaf_stats_newton(&leaf_of, &der1, &weighted_der2, 2);
    assert_eq!(stats.len(), 2);
    // gradient sums match reduce_leaf_stats; hessian is negated der2 sum.
    assert!((stats[0].sum_weighted_delta - 4.0).abs() < 1e-12); // 1+3
    assert!((stats[0].sum_weight - 2.0).abs() < 1e-12); // -(-1 + -1)
    assert!((stats[1].sum_weighted_delta - 6.0).abs() < 1e-12); // 2+4
    assert!((stats[1].sum_weight - 0.75).abs() < 1e-12); // -(-0.5 + -0.25)
}

#[test]
fn reduce_leaf_stats_newton_empty_leaf_is_zero() {
    // A degenerate (empty) leaf yields zero gradient AND zero hessian — finite,
    // never NaN/Inf (T-06.4A-01 robustness).
    let leaf_of = [0usize, 0, 0];
    let der1 = [1.0, 2.0, 3.0];
    let weighted_der2 = [-1.0, -1.0, -1.0];
    let stats = reduce_leaf_stats_newton(&leaf_of, &der1, &weighted_der2, 2);
    assert_eq!(stats[1], LeafStats::default());
    assert!(stats[1].sum_weight.is_finite());
    // leaf 0: gradient 6.0, positive hessian 3.0.
    assert!((stats[0].sum_weighted_delta - 6.0).abs() < 1e-12);
    assert!((stats[0].sum_weight - 3.0).abs() < 1e-12);
}

#[test]
fn reduce_leaf_der2_groups_by_leaf() {
    // leaf 0 = {obj0, obj2}, leaf 1 = {obj1, obj3}; weighted der2 per object.
    let leaf_of = [0usize, 1, 0, 1];
    let weighted_der2 = [-1.0, -0.5, -1.0, -0.25];
    let d2 = reduce_leaf_der2(&leaf_of, &weighted_der2, 2);
    assert_eq!(d2.len(), 2);
    assert!((d2[0] - (-2.0)).abs() < 1e-12); // -1 + -1
    assert!((d2[1] - (-0.75)).abs() < 1e-12); // -0.5 + -0.25
}

#[test]
fn reduce_leaf_der2_empty_leaf_is_zero() {
    let leaf_of = [0usize, 0];
    let weighted_der2 = [-1.0, -1.0];
    let d2 = reduce_leaf_der2(&leaf_of, &weighted_der2, 2);
    assert_eq!(d2[1], 0.0);
    assert!((d2[0] - (-2.0)).abs() < 1e-12);
}

#[test]
fn collect_leaf_residuals_groups_members() {
    // leaf 0 = {obj0, obj2}, leaf 1 = {obj1}. residuals widen through f32.
    let leaf_of = [0usize, 1, 0];
    let residuals = [1.5_f64, -2.0, 3.25];
    let weight = [1.0, 1.0, 1.0];
    let members = collect_leaf_residuals(&leaf_of, &residuals, &weight, 2);
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].0, vec![1.5_f32, 3.25_f32]);
    assert_eq!(members[0].1, vec![1.0_f64, 1.0_f64]);
    assert_eq!(members[1].0, vec![-2.0_f32]);
}
