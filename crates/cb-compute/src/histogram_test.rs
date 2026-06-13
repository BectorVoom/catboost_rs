//! Unit tests for the host-side ordered bucket reduction (D-02/D-05).

use crate::histogram::{reduce_leaf_stats, LeafStats};

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
