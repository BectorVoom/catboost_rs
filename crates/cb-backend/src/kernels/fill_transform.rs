//! Self-oracle for the fill / gather / vector-arithmetic transforms (Plan 10-04,
//! GPUT-16): each trivial transform must match an INLINE SERIAL elementwise reference
//! (D-02: no `cb-train` reach, no upstream fixture). These are the trivial primitives
//! whose FULL validation is transitive through the depth-1 tree + cindex (D-01), so the
//! oracle asserts elementwise equality on small inputs (plus one `n >> CUBE_DIM` case).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernels live in
//! `kernels.rs`; all assertions + `.unwrap()`/indexing live here.
//!
//! Runs on `rocm` in-env on gfx1100 (wave32) and under every other backend over the
//! generic [`crate::SelectedRuntime`]. Reported bounds are generous run-stable values
//! (NOT the signed-off GPU-06 epsilon); f64 arithmetic here is elementwise so exact.

use cubecl::prelude::*;

use crate::kernels::{
    fill_kernel, gather_kernel, vector_add_kernel, vector_div_kernel, vector_mul_kernel,
    vector_sub_kernel,
};

/// 32-wide cubes (wave32 gfx1100); enough cubes to cover every element.
const CUBE_DIM: usize = 32;

/// Tight elementwise bound: these transforms are per-lane with no reassociation, so the
/// device result is bit-exact with the serial f64 reference for f64 inputs.
const ABS_TOL: f64 = 1e-12;

fn client_dim(n: usize) -> (
    cubecl::client::ComputeClient<crate::SelectedRuntime>,
    CubeCount,
    CubeDim,
) {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let n_cubes = n.div_ceil(CUBE_DIM).max(1);
    (
        client,
        CubeCount::Static(n_cubes as u32, 1, 1),
        CubeDim { x: CUBE_DIM as u32, y: 1, z: 1 },
    )
}

/// Launch `fill_kernel::<f64>` filling an `n`-element buffer with `value`.
fn run_fill(n: usize, value: f64) -> Vec<f64> {
    let (client, count, dim) = client_dim(n);
    let buf_h = client.empty(n * std::mem::size_of::<f64>());
    let val_h = client.create(cubecl::bytes::Bytes::from_elems(vec![value]));
    fill_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(buf_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(val_h, 1) },
    );
    let b = client.read_one(buf_h).unwrap();
    bytemuck::cast_slice::<u8, f64>(&b).to_vec()
}

/// Launch `gather_kernel::<f64>`: `out[i] = src[idx[i]]`.
fn run_gather(src: &[f64], idx: &[u32]) -> Vec<f64> {
    let n = idx.len();
    let (client, count, dim) = client_dim(n);
    let src_h = client.create(cubecl::bytes::Bytes::from_elems(src.to_vec()));
    let idx_h = client.create(cubecl::bytes::Bytes::from_elems(idx.to_vec()));
    let out_h = client.empty(n * std::mem::size_of::<f64>());
    gather_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(src_h, src.len()) },
        unsafe { ArrayArg::from_raw_parts(idx_h, n) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) },
    );
    let b = client.read_one(out_h).unwrap();
    bytemuck::cast_slice::<u8, f64>(&b).to_vec()
}

/// Which elementwise binary op to launch (keeps the launch boilerplate in one place).
enum VecOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Launch the selected elementwise `vector_*_kernel::<f64>` over `a` and `b`.
fn run_vector(a: &[f64], b: &[f64], op: VecOp) -> Vec<f64> {
    let n = a.len();
    assert_eq!(n, b.len(), "a/b length mismatch");
    let (client, count, dim) = client_dim(n);
    let a_h = client.create(cubecl::bytes::Bytes::from_elems(a.to_vec()));
    let b_h = client.create(cubecl::bytes::Bytes::from_elems(b.to_vec()));
    let out_h = client.empty(n * std::mem::size_of::<f64>());
    let a_arg = unsafe { ArrayArg::from_raw_parts(a_h, n) };
    let b_arg = unsafe { ArrayArg::from_raw_parts(b_h, n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) };
    match op {
        VecOp::Add => {
            vector_add_kernel::launch::<f64, crate::SelectedRuntime>(&client, count, dim, a_arg, b_arg, out_arg)
        }
        VecOp::Sub => {
            vector_sub_kernel::launch::<f64, crate::SelectedRuntime>(&client, count, dim, a_arg, b_arg, out_arg)
        }
        VecOp::Mul => {
            vector_mul_kernel::launch::<f64, crate::SelectedRuntime>(&client, count, dim, a_arg, b_arg, out_arg)
        }
        VecOp::Div => {
            vector_div_kernel::launch::<f64, crate::SelectedRuntime>(&client, count, dim, a_arg, b_arg, out_arg)
        }
    }
    let bytes = client.read_one(out_h).unwrap();
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

fn assert_close(device: &[f64], expected: &[f64], what: &str) {
    assert_eq!(device.len(), expected.len(), "{what}: length mismatch");
    for (i, (&d, &e)) in device.iter().zip(expected).enumerate() {
        assert!((d - e).abs() <= ABS_TOL, "{what}: elem {i} device={d} expected={e}");
    }
}

#[test]
fn fill_sets_every_element() {
    // Behaviour: fill(buf, c) sets every element to c. Small + n >> CUBE_DIM.
    let small = run_fill(5, -2.5);
    assert_close(&small, &vec![-2.5; 5], "fill small");

    let big = run_fill(5000, 3.75);
    assert_close(&big, &vec![3.75; 5000], "fill n>>CUBE_DIM");
}

#[test]
fn gather_matches_indexed_read() {
    // Behaviour: gather(src, idx)[i] = src[idx[i]].
    let src = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
    let idx = vec![4u32, 0, 2, 2, 1];
    let device = run_gather(&src, &idx);
    let expected: Vec<f64> = idx.iter().map(|&j| src[j as usize]).collect();
    assert_eq!(device, vec![50.0_f64, 10.0, 30.0, 30.0, 20.0], "gather behaviour example");
    assert_close(&device, &expected, "gather");

    // n >> CUBE_DIM: reversed-index gather.
    let n = 4096usize;
    let src2: Vec<f64> = (0..n).map(|k| (k as f64) * 0.25 - 7.0).collect();
    let idx2: Vec<u32> = (0..n as u32).rev().collect();
    let device2 = run_gather(&src2, &idx2);
    let expected2: Vec<f64> = idx2.iter().map(|&j| src2[j as usize]).collect();
    assert_close(&device2, &expected2, "gather n>>CUBE_DIM");
}

#[test]
fn vector_arithmetic_is_elementwise() {
    let n = 3000usize; // >> CUBE_DIM
    let a: Vec<f64> = (0..n).map(|k| (k as f64) * 0.5 - 11.0).collect();
    let b: Vec<f64> = (0..n).map(|k| ((k % 13) as f64) + 1.0).collect(); // non-zero for div

    let add = run_vector(&a, &b, VecOp::Add);
    let sub = run_vector(&a, &b, VecOp::Sub);
    let mul = run_vector(&a, &b, VecOp::Mul);
    let div = run_vector(&a, &b, VecOp::Div);

    let exp_add: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    let exp_sub: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
    let exp_mul: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
    let exp_div: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x / y).collect();

    assert_close(&add, &exp_add, "vector add");
    assert_close(&sub, &exp_sub, "vector sub");
    assert_close(&mul, &exp_mul, "vector mul");
    assert_close(&div, &exp_div, "vector div");
}
