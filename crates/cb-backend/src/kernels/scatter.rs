//! Tests for the histogram-scatter kernel (`contrib[i] = der1[i] * weight[i]`):
//! assert per-element output equals a host reference and that the kernel performs
//! NO reduction (the output is per-object, same length as the input).
//!
//! Source/test separation (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs`; all assertions live here.

use cubecl::prelude::*;

use crate::kernels::histogram_scatter_kernel;

/// Launch `histogram_scatter_kernel::<F>` on `CpuRuntime` and read back `contrib`.
fn run_scatter<F>(der1: &[F], weight: &[F]) -> Vec<F>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = der1.len();
    let device = cubecl::cpu::CpuDevice::default();
    let client = <cubecl::cpu::CpuRuntime as Runtime>::client(&device);

    let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1.to_vec()));
    let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight.to_vec()));
    let contrib_handle = client.empty(n * std::mem::size_of::<F>());

    let cube_dim = 32usize;
    let num_cubes = n.div_ceil(cube_dim).max(1);

    histogram_scatter_kernel::launch::<F, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: cube_dim as u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
        unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
        unsafe { ArrayArg::from_raw_parts(contrib_handle.clone(), n) },
    );

    let bytes = client.read_one(contrib_handle).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

#[test]
fn scatter_kernel_matches_host_reference_f64() {
    let der1: Vec<f64> = vec![1.0, -2.0, 3.5, 0.0, -7.25, 10.0];
    let weight: Vec<f64> = vec![1.0, 1.0, 2.0, 0.5, 1.0, 0.0];

    let contrib = run_scatter(&der1, &weight);

    // No reduction: output length equals input length (per-object scatter).
    assert_eq!(contrib.len(), der1.len());
    for i in 0..der1.len() {
        let expected = der1[i] * weight[i];
        assert!(
            (contrib[i] - expected).abs() <= 1e-12,
            "scatter mismatch at {i}: kernel={}, host={expected}",
            contrib[i]
        );
    }
}

#[test]
fn scatter_kernel_unweighted_is_identity() {
    // Every weight == 1.0 -> contrib == der1 (the unweighted path).
    let der1: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.5 - 3.0).collect();
    let weight: Vec<f64> = vec![1.0; 37];

    let contrib = run_scatter(&der1, &weight);

    assert_eq!(contrib.len(), der1.len());
    for i in 0..der1.len() {
        assert!((contrib[i] - der1[i]).abs() <= 1e-12);
    }
}
