//! Unit tests for the [`super::DeviceGrownTree`] multi-output block-leaf extension
//! (Phase 13 Plan 06, GPUT-12, D-03 / D-04).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the plain host seam
//! struct lives in `runtime.rs`; these `#[test]` + `.unwrap()`/indexing assertions
//! live here. The tests pin two contracts:
//!
//! 1. **Block reinterpretation** — `leaf_values` is a `leaf_count × approx_dim`
//!    row-major block; dimension `d` of leaf `l` is `leaf_values[l * approx_dim + d]`,
//!    and the multi-output CPU apply layout is
//!    `approx[d * n + i] += lr * leaf_values[leaf_of[i] * approx_dim + d]`.
//! 2. **Scalar byte-invariance (GPUT-14 / D-04)** — at `approx_dim == 1` the block
//!    collapses to the flat scalar vector: `leaf_values[l * 1 + 0] == leaf_values[l]`,
//!    so the scalar emission is byte-unchanged.

use crate::DeviceGrownTree;

/// A depth-1 scalar tree carries `approx_dim == 1` and the block index collapses to
/// the flat leaf-index vector (byte-unchanged scalar path, D-04).
#[test]
fn device_grown_tree_scalar_approx_dim_is_one() {
    let tree = DeviceGrownTree {
        splits: vec![(0, 1)],
        leaf_values: vec![2.0, -3.0],
        approx_dim: 1,
        leaf_of: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_path: Vec::new(),
    };
    assert_eq!(tree.approx_dim, 1, "scalar tree must carry approx_dim == 1");
    // Block index `l * approx_dim + d` collapses to `l` when approx_dim == 1, d == 0.
    for l in 0..tree.leaf_values.len() {
        let block_idx = l * tree.approx_dim; // d == 0
        assert_eq!(
            tree.leaf_values[block_idx], tree.leaf_values[l],
            "scalar block index must equal the flat leaf index (byte-unchanged)"
        );
    }
}

/// A K=3 multi-output tree reinterprets `leaf_values` as a `leaf_count × 3` row-major
/// block: leaf `l`, dimension `d` is `leaf_values[l * 3 + d]`.
#[test]
fn device_grown_tree_block_reinterpretation() {
    let k = 3usize;
    let leaf_count = 2usize;
    // Leaf 0 = [0.1, 0.2, 0.3]; leaf 1 = [0.4, 0.5, 0.6] (row-major per leaf).
    let leaf_values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let tree = DeviceGrownTree {
        splits: vec![(0, 1)],
        leaf_values,
        approx_dim: k,
        leaf_of: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_path: Vec::new(),
    };
    assert_eq!(tree.leaf_values.len(), leaf_count * k, "block length == leaf_count × K");
    assert_eq!(tree.leaf_values[0 * k + 0], 0.1);
    assert_eq!(tree.leaf_values[0 * k + 2], 0.3);
    assert_eq!(tree.leaf_values[1 * k + 0], 0.4);
    assert_eq!(tree.leaf_values[1 * k + 2], 0.6);
}

/// The multi-output CPU apply layout `approx[d * n + i] += lr * leaf_block[leaf_of[i] * K + d]`
/// updates the dimension-major approximant buffer from the block leaves. This pins the
/// exact index arithmetic the device block apply routes through (D-03).
#[test]
fn device_grown_tree_block_apply_layout() {
    let k = 2usize;
    let n = 3usize;
    let lr = 0.5_f64;
    // Two leaves, K=2: leaf 0 = [1.0, 2.0]; leaf 1 = [3.0, 4.0].
    let leaf_block = vec![1.0, 2.0, 3.0, 4.0];
    let leaf_of = [0u32, 1, 0];
    // Dimension-major approx buffer, length K*n, starts at zero.
    let mut approx = vec![0.0_f64; k * n];
    for d in 0..k {
        for i in 0..n {
            let leaf = leaf_of[i] as usize;
            approx[d * n + i] += lr * leaf_block[leaf * k + d];
        }
    }
    // Expected: object 0 (leaf 0): d0 += 0.5*1.0, d1 += 0.5*2.0.
    //           object 1 (leaf 1): d0 += 0.5*3.0, d1 += 0.5*4.0.
    //           object 2 (leaf 0): d0 += 0.5*1.0, d1 += 0.5*2.0.
    let expected = vec![0.5, 1.5, 0.5, 1.0, 2.0, 1.0];
    for (got, want) in approx.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-12, "block apply {got} vs {want}");
    }
}
