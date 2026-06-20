//! Self-oracle for the device-resident RMSE der seam (GPU-01 der, Phase 7.2): the
//! GPU der1 computed over [`crate::SelectedRuntime`] must match the
//! `cb-compute::loss` CPU baseline (`rmse_der1`/`rmse_der2`) within a REPORTED (not
//! signed-off) tolerance, over f32 and f64 fixtures including edge cases (empty,
//! n=1, non-cube-multiple, large N), and the device-residency hand-off must return
//! der1/der2 HANDLES with no host fold inserted on the seam (SC-3 / D-7.2-04).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs` and the launch seam in `gpu_runtime.rs`; all assertions live here.
//! Test code may use `.unwrap()`/indexing (the `lib.rs:1` `#[cfg(test)]` allow) —
//! the production `gpu_runtime.rs` may not.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32), and builds/runs under every
//! backend (like `kernels::reduce`/`kernels::scan`). The reported max abs/rel
//! divergence is informational: the GPU-06 epsilon is signed off in Phase 7.6, NOT
//! hard-coded here (D-7.2-06). The asserted tolerances are generous, run-stable
//! bounds (f32 ~1e-3 relative, f64 ~1e-9 relative) that catch a wrong der without
//! pinning the final epsilon.

use cubecl::prelude::*;

use crate::gpu_runtime::{const_der_handle, launch_der_binary, launch_der_binary_handle, DerBinaryKernel};

/// Compare the device der (cast to f64) to the CPU baseline element-wise, returning
/// the max abs and max rel divergence over the vector. Copied verbatim from the
/// `kernels::scan` reporter (REPORT-not-sign-off, D-7.2-06).
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for i in 0..baseline.len() {
        let abs = (device[i] - baseline[i]).abs();
        let rel = if baseline[i].abs() > 0.0 {
            abs / baseline[i].abs()
        } else {
            abs
        };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

/// The `cb-compute::rmse_der1` CPU baseline for a fixture, computed elementwise in
/// the frozen host order (`target - approx`). This is the D-7.2-06 baseline the
/// device der1 is REPORTED against.
fn rmse_der1_baseline(approx: &[f64], target: &[f64]) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::rmse_der1(a, t))
        .collect()
}

/// Launch the device RMSE der1 over an f32 fixture by upcasting to f64 at the seam
/// boundary (the seam is f64-typed, matching the `cb-compute` baseline). The f32
/// fixture is cast to f64 BEFORE the launch and the baseline is computed on the same
/// cast inputs, so the reported divergence is the device-vs-host der divergence at
/// f64, exercised with f32-magnitude values (the reduce/scan f32 bound applies).
fn run_der_binary_f32(approx: &[f32], target: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let approx_f64: Vec<f64> = approx.iter().map(|&v| f64::from(v)).collect();
    let target_f64: Vec<f64> = target.iter().map(|&v| f64::from(v)).collect();
    let device = launch_der_binary(&approx_f64, &target_f64, DerBinaryKernel::RmseGradient).unwrap();
    let baseline = rmse_der1_baseline(&approx_f64, &target_f64);
    (device, baseline)
}

#[test]
fn rmse_gradient_matches_cpu_baseline_f64_non_cube_multiple() {
    // n=37 is a non-cube-multiple length: it exercises the `if ABSOLUTE_POS <
    // approx.len()` idle-guard path in `gradient_kernel` (Pitfall 4) — the surplus
    // threads in the last cube must not write past `der1[n-1]`.
    let approx: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.5 - 3.0).collect();
    let target: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.25 + 1.0).collect();

    let device = launch_der_binary(&approx, &target, DerBinaryKernel::RmseGradient).unwrap();
    let baseline = rmse_der1_baseline(&approx, &target);

    assert_eq!(device.len(), approx.len(), "der1 output length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der rmse f64 n=37] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-9 || abs <= 1e-9,
        "f64 RMSE der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn rmse_gradient_matches_cpu_baseline_f32() {
    // f32-magnitude fixture (cast to f64 at the seam): the device der1 vs the f64
    // host baseline over the same cast inputs. A generous f32 relative bound
    // (~1e-3) catches a wrong der without pinning the GPU-06 epsilon.
    let approx: Vec<f32> = (0..64).map(|k| (k as f32) * 0.5 - 8.0 + 0.125 * (k as f32)).collect();
    let target: Vec<f32> = (0..64).map(|k| (k as f32) * 0.25 + 2.0).collect();

    let (device, baseline) = run_der_binary_f32(&approx, &target);

    assert_eq!(device.len(), approx.len(), "der1 output length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der rmse f32 n=64] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-3 || abs <= 1e-3,
        "f32 RMSE der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn rmse_der_edge_cases() {
    // Empty (n=0): the production seam short-circuits with NO launch; der1 is empty.
    // NOTE: a zero-length device buffer is NOT read back here — `read_one` on an
    // empty HIP buffer dereferences a null/zero-length pointer (the HIP IO
    // controller's `slice::from_raw_parts` precondition), so the empty case asserts
    // the host-wrapper result is empty and that the const-der2 handle CONSTRUCTS
    // (the production seam likewise never reads an empty handle — it short-circuits).
    {
        let approx: Vec<f64> = Vec::new();
        let target: Vec<f64> = Vec::new();
        let device = launch_der_binary(&approx, &target, DerBinaryKernel::RmseGradient).unwrap();
        assert!(device.is_empty(), "empty input must yield an empty der1 (no launch)");
        // The const-der2 handle is length-0 for the empty case: assert it builds
        // (Ok) without reading the empty device buffer back.
        assert!(
            const_der_handle(-1.0, 0).is_ok(),
            "empty der2 const handle must construct (no read-back of an empty buffer)"
        );
    }

    // n=1: a single object, one thread, no surplus.
    {
        let approx = vec![1.5_f64];
        let target = vec![4.25_f64];
        let device = launch_der_binary(&approx, &target, DerBinaryKernel::RmseGradient).unwrap();
        let baseline = rmse_der1_baseline(&approx, &target);
        let (abs, rel) = max_divergence(&device, &baseline);
        println!("[der rmse f64 n=1] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
        assert!(
            rel <= 1e-9 || abs <= 1e-9,
            "f64 RMSE der1 (n=1) diverged too far: abs={abs:.3e} rel={rel:.3e}"
        );
    }

    // Large N: many cubes (10_000 >> CUBE_DIM=32) — the elementwise der has no
    // cross-cube dependency, so every cube is independent and the result is exact.
    {
        let n = 10_000usize;
        let approx: Vec<f64> = (0..n).map(|k| (k as f64) * 0.001 - 5.0).collect();
        let target: Vec<f64> = (0..n).map(|k| (k as f64) * 0.002 + 0.5).collect();
        let device = launch_der_binary(&approx, &target, DerBinaryKernel::RmseGradient).unwrap();
        let baseline = rmse_der1_baseline(&approx, &target);
        assert_eq!(device.len(), n, "large-N der1 length must equal input length");
        let (abs, rel) = max_divergence(&device, &baseline);
        println!("[der rmse f64 n=10000] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
        assert!(
            rel <= 1e-9 || abs <= 1e-9,
            "f64 RMSE der1 (large-N) diverged too far: abs={abs:.3e} rel={rel:.3e}"
        );
    }
}

#[test]
fn rmse_der_device_resident_handoff() {
    // The SC-3 device-residency assertion: `launch_der_binary_handle` returns the
    // der1 as a device HANDLE with NO host fold inserted on the seam, and
    // `const_der_handle(-1.0, n)` returns the der2 (constant -1.0) as a device
    // HANDLE. The read-back happens ONCE here (in the test), confirming the handles
    // carry the correct values, and never on the hand-off path itself.
    let approx: Vec<f64> = (0..50).map(|k| f64::from(k) * 0.3 - 7.0).collect();
    let target: Vec<f64> = (0..50).map(|k| f64::from(k) * 0.1 + 2.0).collect();
    let n = approx.len();

    // Handle-in -> handles-out: BOTH return Ok(handle) with no read-back on the seam.
    let der1_handle = launch_der_binary_handle(&approx, &target, DerBinaryKernel::RmseGradient).unwrap();
    let der2_handle = const_der_handle(-1.0, n).unwrap();

    // Read each handle back ONCE (the test is the only place a read-back is allowed).
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let der1_bytes = client.read_one(der1_handle).unwrap();
    let der1_host = bytemuck::cast_slice::<u8, f64>(&der1_bytes).to_vec();
    let der1_baseline = rmse_der1_baseline(&approx, &target);
    assert_eq!(der1_host.len(), n, "der1 handle length must equal n");
    let (abs1, rel1) = max_divergence(&der1_host, &der1_baseline);
    println!("[der rmse handoff der1 n=50] REPORTED max abs_div={abs1:.3e} rel_div={rel1:.3e}");
    assert!(
        rel1 <= 1e-9 || abs1 <= 1e-9,
        "device-resident der1 diverged from cb_compute::rmse_der1: abs={abs1:.3e} rel={rel1:.3e}"
    );

    let der2_bytes = client.read_one(der2_handle).unwrap();
    let der2_host = bytemuck::cast_slice::<u8, f64>(&der2_bytes).to_vec();
    assert_eq!(der2_host.len(), n, "der2 const handle length must equal n");
    // The RMSE der2 is the constant -1.0 (== cb_compute::rmse_der2 for every object).
    for (i, &v) in der2_host.iter().enumerate() {
        let baseline = cb_compute::rmse_der2(approx[i], target[i]);
        assert_eq!(
            v, baseline,
            "der2 const handle slot {i} = {v}, expected cb_compute::rmse_der2 = {baseline}"
        );
    }
}
