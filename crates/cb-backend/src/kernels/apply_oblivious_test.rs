//! Device-vs-host oracle for [`apply_oblivious_float_kernel`] (GINF-01-S3).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the production
//! `#[cube]` body lives in `kernels.rs`; the launch shell + assertions live here,
//! mounted at `kernels::apply_oblivious_test`.
//!
//! The kernel computes one raw prediction per object for a float-only oblivious
//! scalar ensemble. This oracle builds a tiny FIXED flat model (2 trees, depth-2,
//! known borders/leaves, nonzero bias) + a 3-object feature-major matrix by hand,
//! launches the kernel under [`crate::SelectedRuntime`] (the `cpu` `CpuRuntime`,
//! f64, in ordinary `cargo test`), reads back `out`, and asserts it equals a HOST
//! reference computed independently as `sum(tree contributions ascending) + bias`
//! — bias added LAST, NOT pre-seeded, so a bias-ordering regression is caught.

use cubecl::prelude::*;

use crate::kernels::apply_oblivious_float_kernel;

/// A tiny fixed float-only oblivious scalar model in the flat (CSR) layout the
/// kernel consumes. Two depth-2 trees over two features.
struct FlatModel {
    split_features: Vec<u32>,
    split_borders: Vec<f64>,
    tree_split_offsets: Vec<u32>,
    leaf_values: Vec<f64>,
    tree_leaf_offsets: Vec<u32>,
    bias: f64,
    n_trees: u32,
}

fn fixed_model() -> FlatModel {
    FlatModel {
        // tree0: split0 on feat0, split1 on feat1; tree1: split0 on feat1, split1 on feat0.
        split_features: vec![0, 1, 1, 0],
        split_borders: vec![0.5, 1.5, 0.0, 2.5],
        tree_split_offsets: vec![0, 2, 4],
        leaf_values: vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        tree_leaf_offsets: vec![0, 4, 8],
        bias: 0.25,
        n_trees: 2,
    }
}

/// HOST reference: forward-bit-order leaf gather per tree, per-tree contributions
/// summed in ASCENDING tree order, with `bias` added exactly ONCE at the very end
/// (NOT pre-seeded). Mirrors `predict_raw_one`'s `bias + sum_f64(&oblivious)`.
fn host_reference(m: &FlatModel, features: &[f64], _n_features: usize, n_objects: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_objects];
    for obj in 0..n_objects {
        let mut acc = 0.0_f64;
        for t in 0..(m.n_trees as usize) {
            let s0 = m.tree_split_offsets[t] as usize;
            let s1 = m.tree_split_offsets[t + 1] as usize;
            let mut leaf = 0u32;
            let mut i = 0u32;
            for s in s0..s1 {
                let f = m.split_features[s] as usize;
                let v = features[f * n_objects + obj];
                let b = m.split_borders[s];
                let bit = if v > b { 1u32 } else { 0u32 };
                leaf += bit << i;
                i += 1;
            }
            let li = m.tree_leaf_offsets[t] as usize + leaf as usize;
            let mut contrib = 0.0_f64;
            if (li as u32) < m.tree_leaf_offsets[t + 1] {
                contrib = m.leaf_values[li];
            }
            acc += contrib;
        }
        acc += m.bias;
        out[obj] = acc;
    }
    out
}

/// Launch the kernel under the selected runtime (f64 under the default `cpu`
/// backend) and read back one prediction per object.
fn run_kernel(m: &FlatModel, features: &[f64], n_features: usize, n_objects: usize) -> Vec<f64> {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let features_h = client.create(cubecl::bytes::Bytes::from_elems(features.to_vec()));
    let split_features_h =
        client.create(cubecl::bytes::Bytes::from_elems(m.split_features.clone()));
    let split_borders_h = client.create(cubecl::bytes::Bytes::from_elems(m.split_borders.clone()));
    let tree_split_offsets_h =
        client.create(cubecl::bytes::Bytes::from_elems(m.tree_split_offsets.clone()));
    let leaf_values_h = client.create(cubecl::bytes::Bytes::from_elems(m.leaf_values.clone()));
    let tree_leaf_offsets_h =
        client.create(cubecl::bytes::Bytes::from_elems(m.tree_leaf_offsets.clone()));
    let bias_h = client.create(cubecl::bytes::Bytes::from_elems(vec![m.bias]));
    let out_h = client.empty(n_objects * std::mem::size_of::<f64>());

    let cube_dim = 32usize;
    let num_cubes = n_objects.div_ceil(cube_dim).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim { x: cube_dim as u32, y: 1, z: 1 };

    apply_oblivious_float_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(features_h, features.len()) },
        unsafe { ArrayArg::from_raw_parts(split_features_h, m.split_features.len()) },
        unsafe { ArrayArg::from_raw_parts(split_borders_h, m.split_borders.len()) },
        unsafe { ArrayArg::from_raw_parts(tree_split_offsets_h, m.tree_split_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(leaf_values_h, m.leaf_values.len()) },
        unsafe { ArrayArg::from_raw_parts(tree_leaf_offsets_h, m.tree_leaf_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), n_objects) },
        unsafe { ArrayArg::from_raw_parts(bias_h, 1) },
        n_objects as u32,
        n_features as u32,
        m.n_trees,
    );

    let bytes = client.read_one(out_h).unwrap();
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

/// A fixed one-tree depth-2 model whose SECOND split references float feature
/// index 5 — deliberately `>= n_features` (2) so the per-split gather has NO
/// in-range column for it. Mirrors a model that references more float features
/// than the caller supplies (`predict_raw` treats an absent feature as bit 0 via
/// the checked `.get`); the kernel must do the same and NOT read out of bounds.
fn missing_feature_model() -> FlatModel {
    FlatModel {
        // split0 on feat0 (in range); split1 on feat5 (OUT OF RANGE for n_features=2).
        // split1's border is NEGATIVE so an unguarded OOB gather (which reads
        // zeroed/adjacent memory ~0.0) would evaluate `~0.0 > -1.0` → bit 1,
        // diverging in VALUE from the guarded "absent → bit 0" reference. This
        // makes the missing-feature bug observable even when the OOB read does
        // not crash under `CpuRuntime`.
        split_features: vec![0, 5],
        split_borders: vec![0.5, -1.0],
        tree_split_offsets: vec![0, 2],
        leaf_values: vec![1.0, 2.0, 3.0, 4.0],
        tree_leaf_offsets: vec![0, 4],
        bias: 0.25,
        n_trees: 1,
    }
}

/// HOST reference that mirrors `passes_float_split`'s checked `.get`: a split
/// feature index `>= n_features` contributes bit 0 (feature absent), NEVER an OOB
/// read. Otherwise identical to [`host_reference`].
fn host_reference_guarded(
    m: &FlatModel,
    features: &[f64],
    n_features: usize,
    n_objects: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_objects];
    for obj in 0..n_objects {
        let mut acc = 0.0_f64;
        for t in 0..(m.n_trees as usize) {
            let s0 = m.tree_split_offsets[t] as usize;
            let s1 = m.tree_split_offsets[t + 1] as usize;
            let mut leaf = 0u32;
            let mut i = 0u32;
            for s in s0..s1 {
                let f = m.split_features[s] as usize;
                // Guarded gather: an out-of-range feature is absent → bit 0.
                let mut bit = 0u32;
                if f < n_features {
                    let v = features[f * n_objects + obj];
                    if v > m.split_borders[s] {
                        bit = 1;
                    }
                }
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

/// GINF-01 CR (Finding #1): a split feature index `>= n_features` must be
/// treated as ABSENT (bit 0), matching `passes_float_split`, and must NOT read
/// past the uploaded `n_features * n_objects` feature array (UB/garbage/crash).
#[test]
fn apply_oblivious_kernel_missing_feature() {
    let m = missing_feature_model();
    let n_features = 2usize;
    let n_objects = 3usize;
    // feature-major, length n_features * n_objects = 6. No column for feature 5.
    let features = vec![1.0, 0.0, 3.0, 2.0, 0.0, 1.0];

    let device = run_kernel(&m, &features, n_features, n_objects);
    let host = host_reference_guarded(&m, &features, n_features, n_objects);

    // By hand: split1 (feat5) is absent → bit 0; leaf == feat0-bit only.
    // obj0: 1.0>0.5 → leaf1 = 2.0 + 0.25; obj1: 0.0 → leaf0 = 1.0 + 0.25;
    // obj2: 3.0>0.5 → leaf1 = 2.0 + 0.25.
    assert_eq!(host, vec![2.25, 1.25, 2.25]);

    assert_eq!(device.len(), n_objects);
    for (obj, (d, h)) in device.iter().zip(host.iter()).enumerate() {
        assert!(
            (d - h).abs() <= 1e-9,
            "object {obj}: device={d} host={h} diff={:.3e}",
            (d - h).abs()
        );
    }
}

#[test]
fn apply_oblivious_kernel_oracle() {
    let m = fixed_model();
    let n_features = 2usize;
    let n_objects = 3usize;
    // feature-major: [f0o0, f0o1, f0o2, f1o0, f1o1, f1o2]
    let features = vec![1.0, 0.0, 3.0, 2.0, 0.0, 1.0];

    let device = run_kernel(&m, &features, n_features, n_objects);
    let host = host_reference(&m, &features, n_features, n_objects);

    // Sanity: the by-hand expected values (bias 0.25 added LAST).
    assert_eq!(host, vec![24.25, 11.25, 42.25]);

    assert_eq!(device.len(), n_objects);
    for (obj, (d, h)) in device.iter().zip(host.iter()).enumerate() {
        assert!(
            (d - h).abs() <= 1e-9,
            "object {obj}: device={d} host={h} diff={:.3e}",
            (d - h).abs()
        );
    }
}
