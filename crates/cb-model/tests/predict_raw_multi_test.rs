//! Unit tests for the public N-dimensional apply path `predict_raw_multi`
//! (CR-01 / Plan 06.2-06). A dedicated `tests/` file (source/test separation,
//! CLAUDE.md) so no `#[cfg(test)]` block lives in `apply.rs`.
//!
//! Three behaviors (06.2-06 Task 1):
//!   1. dim=1 BYTE-IDENTITY: for any scalar model, `predict_raw_multi` equals
//!      `predict_raw` element-for-element via `to_bits()` (the D-04 invariant on
//!      the public apply surface — the multi path collapses to the scalar
//!      accumulator at dim=1).
//!   2. multi-dim accumulation: a hand-built 1-tree `approx_dimension=3` model with
//!      a DIMENSION-MAJOR `leaf_values` (`[d*n_leaves + l]`) predicts
//!      `bias + leaf_values[d*n_leaves + leaf]` for EACH `d`, output length
//!      `3 * n_objects`, dim-major (`out[d*n + i]`).
//!   3. checked access: a leaf index out of range contributes 0.0 for every
//!      dimension (no panic), matching the scalar `unwrap_or(0.0)`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_model::{predict_raw, predict_raw_multi, Model, ModelSplit, ObliviousTree, Split};

/// A depth-1 oblivious tree splitting feature 0 at `border`, with the supplied
/// (forward-bit-order) `leaf_values` (and matching-length zero weights).
fn one_split_tree(feature: usize, border: f64, leaf_values: Vec<f64>) -> ObliviousTree {
    let n = leaf_values.len();
    ObliviousTree {
        splits: vec![ModelSplit::Float(Split { feature, border })],
        leaf_values,
        leaf_weights: vec![0.0; n],
    }
}

/// A scalar (dim=1) model: one depth-1 tree, two leaves `[lo, hi]`, bias `b`.
fn scalar_model(border: f64, lo: f64, hi: f64, bias: f64) -> Model {
    Model {
        oblivious_trees: vec![one_split_tree(0, border, vec![lo, hi])],
        non_symmetric_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![border]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

#[test]
fn dim1_byte_identical_to_predict_raw() {
    let model = scalar_model(0.5, -1.25, 3.5, 0.42);
    // A mix of objects below / above the border (NaN-free f32 column).
    let columns = vec![vec![0.1_f32, 0.9, 0.3, 0.7, 0.5]];

    let scalar = predict_raw(&model, &columns);
    let multi = predict_raw_multi(&model, &columns);

    assert_eq!(
        scalar.len(),
        multi.len(),
        "dim=1 multi output must have one value per object"
    );
    for (i, (&s, &m)) in scalar.iter().zip(multi.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            m.to_bits(),
            "object {i}: predict_raw_multi must be BYTE-IDENTICAL to predict_raw at dim=1 \
             (scalar={s}, multi={m})"
        );
    }
}

#[test]
fn dim1_byte_identical_empty_and_biasonly() {
    // Bias-only model (no trees) — every object predicts exactly `bias`, identical
    // on both paths.
    let model = Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: Vec::new(),
        bias: -0.75,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    };
    let columns = vec![vec![0.1_f32, 0.9, 0.3]];
    let scalar = predict_raw(&model, &columns);
    let multi = predict_raw_multi(&model, &columns);
    assert_eq!(scalar.len(), 3);
    for (&s, &m) in scalar.iter().zip(multi.iter()) {
        assert_eq!(s.to_bits(), m.to_bits());
    }
}

#[test]
fn multi_dim_accumulation_dim_major_output() {
    // approx_dimension=3, depth-1 tree (2 leaves). DIMENSION-MAJOR leaf_values:
    //   leaf_values[d * n_leaves + l], n_leaves=2, dim=3, length 6.
    //   d0: [10, 20]   d1: [30, 40]   d2: [50, 60]
    let leaf_values = vec![10.0, 20.0, /*d0*/ 30.0, 40.0, /*d1*/ 50.0, 60.0 /*d2*/];
    let bias = 1.0;
    let model = Model {
        oblivious_trees: vec![one_split_tree(0, 0.5, leaf_values)],
        non_symmetric_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 3,
        class_to_label: Vec::new(),
    };
    // Object 0 below border -> leaf 0; object 1 above border -> leaf 1.
    let columns = vec![vec![0.1_f32, 0.9]];
    let out = predict_raw_multi(&model, &columns);

    let n = 2usize;
    let dim = 3usize;
    assert_eq!(out.len(), dim * n, "output length must be approx_dimension * n");

    // n_leaves = 2. Expected: bias + leaf_values[d*2 + leaf].
    // obj0 -> leaf0: d0=10,d1=30,d2=50 ; obj1 -> leaf1: d0=20,d1=40,d2=60.
    // dim-major out[d*n + i]:
    let expected = vec![
        bias + 10.0, // d0,obj0
        bias + 20.0, // d0,obj1
        bias + 30.0, // d1,obj0
        bias + 40.0, // d1,obj1
        bias + 50.0, // d2,obj0
        bias + 60.0, // d2,obj1
    ];
    assert_eq!(out, expected, "dim-major out[d*n+i] = bias + leaf_values[d*n_leaves+leaf]");
}

#[test]
fn out_of_range_leaf_contributes_zero_no_panic() {
    // A malformed tree: a depth-1 split implies 2 structural leaves, but at dim=2
    // a well-formed `leaf_values` would be length 4 (`2 leaves * 2 dims`). We
    // supply only length 2, so `n_leaves = leaf_values.len() / dim = 2/2 = 1`.
    // For the object landing on structural leaf 1, dimension d=1 reads index
    // `d*n_leaves + leaf = 1*1 + 1 = 2`, which is OUT OF RANGE (len 2) -> must
    // contribute 0.0 (checked `.get`), never panic.
    let leaf_values = vec![7.0, 9.0];
    let bias = 0.5;
    let model = Model {
        oblivious_trees: vec![one_split_tree(0, 0.5, leaf_values)],
        non_symmetric_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 2,
        class_to_label: Vec::new(),
    };
    // obj0 below border -> structural leaf 0 ; obj1 above border -> structural leaf 1.
    let columns = vec![vec![0.1_f32, 0.9]];
    let out = predict_raw_multi(&model, &columns);
    let n = 2usize;
    let dim = 2usize;
    assert_eq!(out.len(), dim * n);
    // n_leaves = 1. Reads: out[d*n + i] = bias + leaf_values.get(d*1 + leaf).
    //   d0,obj0(leaf0): get(0)=7 -> 7.5
    //   d0,obj1(leaf1): get(1)=9 -> 9.5
    //   d1,obj0(leaf0): get(1)=9 -> 9.5
    //   d1,obj1(leaf1): get(2)=OOR -> 0 -> 0.5  (the checked-access guarantee)
    let expected = vec![bias + 7.0, bias + 9.0, bias + 9.0, bias];
    assert_eq!(
        out, expected,
        "an out-of-range leaf read contributes 0.0 (no panic): last slot is bias only"
    );
}
