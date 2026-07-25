//! Integration oracle for [`cb_backend::gpu_runtime::launch_apply_oblivious_f64`]
//! (GINF-01-S4).
//!
//! Builds the flat arrays of a known 2-tree float-only oblivious scalar model +
//! a feature-major object matrix by hand, launches the device apply helper under
//! the compile-time-selected runtime (f64 under the default `cpu` backend), and
//! asserts the returned vector equals a host reconstruction within the report
//! bound. Also exercises the empty-input and shape-mismatch error paths.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::gpu_runtime::launch_apply_oblivious_f64;
use cb_core::CbError;

struct FlatModel {
    split_features: Vec<u32>,
    split_borders: Vec<f64>,
    tree_split_offsets: Vec<u32>,
    leaf_values: Vec<f64>,
    tree_leaf_offsets: Vec<u32>,
    bias: f64,
}

fn fixed_model() -> FlatModel {
    FlatModel {
        split_features: vec![0, 1, 1, 0],
        split_borders: vec![0.5, 1.5, 0.0, 2.5],
        tree_split_offsets: vec![0, 2, 4],
        leaf_values: vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        tree_leaf_offsets: vec![0, 4, 8],
        bias: 0.25,
    }
}

/// Independent host reconstruction: forward-bit-order leaf gather per tree, per-tree
/// contributions summed in ascending tree order, `bias` added exactly ONCE last.
fn host_reference(m: &FlatModel, features: &[f64], n_objects: usize) -> Vec<f64> {
    let n_trees = m.tree_split_offsets.len() - 1;
    let mut out = vec![0.0_f64; n_objects];
    for obj in 0..n_objects {
        let mut acc = 0.0_f64;
        for t in 0..n_trees {
            let s0 = m.tree_split_offsets[t] as usize;
            let s1 = m.tree_split_offsets[t + 1] as usize;
            let mut leaf = 0u32;
            let mut i = 0u32;
            for s in s0..s1 {
                let f = m.split_features[s] as usize;
                let v = features[f * n_objects + obj];
                let bit = if v > m.split_borders[s] { 1u32 } else { 0u32 };
                leaf += bit << i;
                i += 1;
            }
            let li = m.tree_leaf_offsets[t] as usize + leaf as usize;
            if (li as u32) < m.tree_leaf_offsets[t + 1] {
                acc += m.leaf_values[li];
            }
        }
        acc += m.bias;
        out[obj] = acc;
    }
    out
}

#[test]
fn launch_matches_host_reference() {
    let m = fixed_model();
    let n_features = 2usize;
    let n_objects = 3usize;
    // feature-major: [f0o0, f0o1, f0o2, f1o0, f1o1, f1o2]
    let features = vec![1.0, 0.0, 3.0, 2.0, 0.0, 1.0];

    let got = launch_apply_oblivious_f64(
        &features,
        n_objects,
        n_features,
        &m.split_features,
        &m.split_borders,
        &m.tree_split_offsets,
        &m.leaf_values,
        &m.tree_leaf_offsets,
        m.bias,
    )
    .expect("launch should succeed for a well-formed model");

    let host = host_reference(&m, &features, n_objects);
    assert_eq!(host, vec![24.25, 11.25, 42.25]);
    assert_eq!(got.len(), n_objects);
    for (obj, (g, h)) in got.iter().zip(host.iter()).enumerate() {
        assert!(
            (g - h).abs() <= 1e-9,
            "object {obj}: device={g} host={h} diff={:.3e}",
            (g - h).abs()
        );
    }
}

#[test]
fn launch_empty_is_empty() {
    let m = fixed_model();
    let got = launch_apply_oblivious_f64(
        &[],
        0,
        2,
        &m.split_features,
        &m.split_borders,
        &m.tree_split_offsets,
        &m.leaf_values,
        &m.tree_leaf_offsets,
        m.bias,
    )
    .expect("empty input must return Ok(vec![])");
    assert!(got.is_empty());
}

#[test]
fn launch_bad_offsets_errs() {
    let m = fixed_model();
    let n_features = 2usize;
    let n_objects = 3usize;
    let features = vec![1.0, 0.0, 3.0, 2.0, 0.0, 1.0];

    // (a) Terminal violation: an offset array whose last element overshoots the
    // array it indexes. `[0, 2, 100]` is length 3 (== tree_leaf_offsets.len()) and
    // monotonic, so it slips past the length/equal-length checks, yet drives the
    // kernel to read `split_features[s]` for `s` up to 100 → OOB. Must be a typed
    // error, never a panic/OOB.
    let bad_terminal: Vec<u32> = vec![0, 2, 100];
    let err = launch_apply_oblivious_f64(
        &features,
        n_objects,
        n_features,
        &m.split_features,
        &m.split_borders,
        &bad_terminal,
        &m.leaf_values,
        &m.tree_leaf_offsets,
        m.bias,
    )
    .expect_err("an out-of-terminal CSR offset array must be a typed error");
    assert!(matches!(err, CbError::OutOfRange(_)), "expected OutOfRange, got {err:?}");

    // (b) Monotonicity violation: `[0, 5, 4]` decreases (5 → 4) while still
    // terminating at split_features.len() == 4. Must also be OutOfRange.
    let non_monotonic: Vec<u32> = vec![0, 5, 4];
    let err2 = launch_apply_oblivious_f64(
        &features,
        n_objects,
        n_features,
        &m.split_features,
        &m.split_borders,
        &non_monotonic,
        &m.leaf_values,
        &m.tree_leaf_offsets,
        m.bias,
    )
    .expect_err("a non-monotonic CSR offset array must be a typed error");
    assert!(matches!(err2, CbError::OutOfRange(_)), "expected OutOfRange, got {err2:?}");
}

#[test]
fn launch_shape_mismatch_errs() {
    let m = fixed_model();
    // features.len() (5) != n_features (2) * n_objects (3) == 6.
    let features = vec![1.0, 0.0, 3.0, 2.0, 0.0];
    let err = launch_apply_oblivious_f64(
        &features,
        3,
        2,
        &m.split_features,
        &m.split_borders,
        &m.tree_split_offsets,
        &m.leaf_values,
        &m.tree_leaf_offsets,
        m.bias,
    )
    .expect_err("a length/shape violation must be a typed error");
    assert!(matches!(err, CbError::OutOfRange(_)), "expected OutOfRange, got {err:?}");
}
