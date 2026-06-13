//! Build/run spike for the CubeCL CPU seam (RESEARCH Open Q2, Wave-0): launch
//! the `#[cube]` `gradient_kernel` on `CpuRuntime`, transfer data with
//! `bytemuck`, and assert per-element output equals a host-computed reference.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives
//! in `kernels.rs`; all assertions live here.

use cubecl::prelude::*;

use crate::kernels::gradient_kernel;

/// Launch `gradient_kernel::<F>` on `CpuRuntime` and read back `der1`.
fn run_gradient<F>(approx: &[F], target: &[F]) -> Vec<F>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice::default();
    let client = <cubecl::cpu::CpuRuntime as Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let target_handle = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
    let der1_handle = client.empty(n * std::mem::size_of::<F>());

    // Ceiling division so a non-multiple `n` still covers every element; the
    // kernel's bounds check idles the surplus threads (multi-threading manual).
    let cube_dim = 32usize;
    let num_cubes = n.div_ceil(cube_dim);

    // cubecl 0.10.0 `ArrayArg::from_raw_parts(handle, length)` consumes the
    // `Handle` (which is `Clone`); clone the output handle so the original is
    // still readable after launch.
    gradient_kernel::launch::<F, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: cube_dim as u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
        unsafe { ArrayArg::from_raw_parts(target_handle, n) },
        unsafe { ArrayArg::from_raw_parts(der1_handle.clone(), n) },
    );

    let bytes = client.read_one(der1_handle).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

#[test]
fn gradient_kernel_matches_host_reference_f32() {
    let approx: Vec<f32> = vec![0.0, 1.0, -2.5, 3.25, 10.0, -0.5, 7.0];
    let target: Vec<f32> = vec![1.0, 0.0, 2.5, -3.25, 4.0, 0.5, 7.0];

    let der1 = run_gradient(&approx, &target);

    assert_eq!(der1.len(), approx.len());
    for i in 0..approx.len() {
        let expected = target[i] - approx[i];
        assert!(
            (der1[i] - expected).abs() <= 1e-6,
            "f32 mismatch at {i}: kernel={}, host={expected}",
            der1[i]
        );
    }
}

#[test]
fn gradient_kernel_matches_host_reference_f64() {
    // A non-cube-multiple length exercises the bounds-check idle path.
    let approx: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.5).collect();
    let target: Vec<f64> = (0..37).map(|k| f64::from(k) * -0.25 + 1.0).collect();

    let der1 = run_gradient(&approx, &target);

    assert_eq!(der1.len(), approx.len());
    for i in 0..approx.len() {
        let expected = target[i] - approx[i];
        assert!(
            (der1[i] - expected).abs() <= 1e-12,
            "f64 mismatch at {i}: kernel={}, host={expected}",
            der1[i]
        );
    }
}
