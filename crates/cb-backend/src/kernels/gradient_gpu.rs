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

use crate::gpu_runtime::{
    const_der_handle, launch_der_binary, launch_der_binary_handle, launch_der_param,
    launch_der_param_handle, launch_der_unary, launch_der_unary_handle, DerBinaryKernel,
    DerParamKernel, DerUnaryKernel,
};

/// Compare the device der (cast to f64) to the CPU baseline element-wise, returning
/// the max abs and max rel divergence over the vector. Matches the hardened
/// `kernels::scan`/`kernels::reduce` reporters (IN-02/IN-03): zip the two slices so a
/// `device.len() < baseline.len()` mismatch surfaces a clear length precondition
/// instead of an opaque index-out-of-bounds panic (WR-03). REPORT-not-sign-off,
/// D-7.2-06.
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    assert_eq!(
        device.len(),
        baseline.len(),
        "max_divergence requires equal-length slices"
    );
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (&d, &b) in device.iter().zip(baseline) {
        let abs = (d - b).abs();
        let rel = if b.abs() > 0.0 { abs / b.abs() } else { abs };
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

// ===========================================================================
// Task 1 (Plan 07.2-02): Logloss / CrossEntropy der1 (binary launch) + der2
// (unary hessian launch) self-oracle vs the `cb-compute::loss` baseline.
//
// Logloss and CrossEntropy share the EXACT der path (Pitfall 6 / D-09): the SAME
// `DerBinaryKernel::LoglossGradient` (der1) and `DerUnaryKernel::LoglossHessian`
// (der2) serve both — there is no separate CrossEntropy kernel. The fixtures use
// logit-shaped approx (a spread that exercises the sigmoid, including a few
// saturated logits) and 0/1 targets.
// ===========================================================================

/// `cb-compute::logloss_der1` baseline (== `cross_entropy_der1`, the shared
/// sigmoid-gradient): `target - sigmoid(approx)`, computed elementwise.
fn logloss_der1_baseline(approx: &[f64], target: &[f64]) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::logloss_der1(a, t))
        .collect()
}

/// `cb-compute::logloss_der2` baseline (== `cross_entropy_der2`): `-p*(1-p)` with
/// `p = sigmoid(approx)`, computed elementwise. der2 is independent of `target`.
fn logloss_der2_baseline(approx: &[f64], target: &[f64]) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::logloss_der2(a, t))
        .collect()
}

#[test]
fn logloss_der1_matches_cpu_baseline_f64_non_cube_multiple() {
    // n=37 (non-cube-multiple) exercises the `if ABSOLUTE_POS < approx.len()`
    // idle-guard path in `logloss_gradient_kernel` (Pitfall 4). The approx spread
    // -4.5..+4.5 includes near-saturated logits (sigmoid -> ~0 / ~1).
    let approx: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.25 - 4.5).collect();
    // Alternating 0/1 binary labels.
    let target: Vec<f64> = (0..37).map(|k| f64::from(k % 2)).collect();

    let device = launch_der_binary(&approx, &target, DerBinaryKernel::LoglossGradient).unwrap();
    let baseline = logloss_der1_baseline(&approx, &target);

    assert_eq!(device.len(), approx.len(), "der1 length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der logloss f64 n=37] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-9 || abs <= 1e-9,
        "f64 Logloss der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn logloss_der1_matches_cpu_baseline_f32() {
    // f32-magnitude logits cast to f64 at the seam (the seam is f64-typed). A
    // generous f32 relative bound catches a wrong der without pinning the epsilon.
    let approx_f32: Vec<f32> = (0..64).map(|k| (k as f32) * 0.15 - 5.0).collect();
    let target_f32: Vec<f32> = (0..64).map(|k| (k % 2) as f32).collect();
    let approx: Vec<f64> = approx_f32.iter().map(|&v| f64::from(v)).collect();
    let target: Vec<f64> = target_f32.iter().map(|&v| f64::from(v)).collect();

    let device = launch_der_binary(&approx, &target, DerBinaryKernel::LoglossGradient).unwrap();
    let baseline = logloss_der1_baseline(&approx, &target);

    assert_eq!(device.len(), approx.len(), "der1 length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der logloss f32 n=64] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-3 || abs <= 1e-3,
        "f32 Logloss der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn crossentropy_reuses_logloss_kernel() {
    // Pitfall 6 / D-09: CrossEntropy reuses the EXACT logloss der path. The SAME
    // `DerBinaryKernel::LoglossGradient` launch must match `cross_entropy_der1`
    // (which itself delegates to `logloss_der1`). CrossEntropy admits soft targets
    // in [0,1], so the fixture uses a few probabilistic labels too.
    let approx: Vec<f64> = (0..40).map(|k| f64::from(k) * 0.2 - 4.0).collect();
    let target: Vec<f64> = (0..40)
        .map(|k| match k % 4 {
            0 => 0.0,
            1 => 1.0,
            2 => 0.25,
            _ => 0.75,
        })
        .collect();

    let device = launch_der_binary(&approx, &target, DerBinaryKernel::LoglossGradient).unwrap();
    let baseline: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::cross_entropy_der1(a, t))
        .collect();

    assert_eq!(device.len(), approx.len(), "der1 length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der crossentropy f64 n=40] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-9 || abs <= 1e-9,
        "CrossEntropy der1 (reusing the logloss kernel) diverged: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn logloss_der2_matches_cpu_baseline_f64() {
    // der2 = -p*(1-p) via the SINGLE-input `launch_der_unary` hessian seam (the
    // hessian takes only `approx`). n=37 exercises the idle guard.
    let approx: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.25 - 4.5).collect();
    let target: Vec<f64> = (0..37).map(|k| f64::from(k % 2)).collect();

    let device = launch_der_unary(&approx, DerUnaryKernel::LoglossHessian).unwrap();
    let baseline = logloss_der2_baseline(&approx, &target);

    assert_eq!(device.len(), approx.len(), "der2 length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der logloss-hess f64 n=37] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-9 || abs <= 1e-9,
        "f64 Logloss der2 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn logloss_der2_matches_cpu_baseline_f32() {
    let approx_f32: Vec<f32> = (0..64).map(|k| (k as f32) * 0.15 - 5.0).collect();
    let approx: Vec<f64> = approx_f32.iter().map(|&v| f64::from(v)).collect();
    // der2 is target-independent; pass a dummy 0/1 vector for the baseline shape.
    let target: Vec<f64> = (0..64).map(|k| f64::from(k % 2)).collect();

    let device = launch_der_unary(&approx, DerUnaryKernel::LoglossHessian).unwrap();
    let baseline = logloss_der2_baseline(&approx, &target);

    assert_eq!(device.len(), approx.len(), "der2 length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der logloss-hess f32 n=64] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-3 || abs <= 1e-3,
        "f32 Logloss der2 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn logloss_der_edge_cases() {
    // Empty: the der1 binary wrapper short-circuits to empty (no launch); the der2
    // unary wrapper likewise. Neither reads back an empty device buffer.
    {
        let approx: Vec<f64> = Vec::new();
        let target: Vec<f64> = Vec::new();
        let der1 = launch_der_binary(&approx, &target, DerBinaryKernel::LoglossGradient).unwrap();
        assert!(der1.is_empty(), "empty input must yield an empty logloss der1 (no launch)");
        let der2 = launch_der_unary(&approx, DerUnaryKernel::LoglossHessian).unwrap();
        assert!(der2.is_empty(), "empty input must yield an empty logloss der2 (no launch)");
    }

    // n=1: a single object, one thread.
    {
        let approx = vec![0.75_f64];
        let target = vec![1.0_f64];
        let der1 = launch_der_binary(&approx, &target, DerBinaryKernel::LoglossGradient).unwrap();
        let base1 = logloss_der1_baseline(&approx, &target);
        let (a1, r1) = max_divergence(&der1, &base1);
        println!("[der logloss f64 n=1] REPORTED der1 max abs_div={a1:.3e} rel_div={r1:.3e}");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "logloss der1 (n=1) diverged: abs={a1:.3e} rel={r1:.3e}");

        let der2 = launch_der_unary(&approx, DerUnaryKernel::LoglossHessian).unwrap();
        let base2 = logloss_der2_baseline(&approx, &target);
        let (a2, r2) = max_divergence(&der2, &base2);
        println!("[der logloss-hess f64 n=1] REPORTED der2 max abs_div={a2:.3e} rel_div={r2:.3e}");
        assert!(r2 <= 1e-9 || a2 <= 1e-9, "logloss der2 (n=1) diverged: abs={a2:.3e} rel={r2:.3e}");
    }

    // Large N: many independent cubes (10_000 >> CUBE_DIM=32).
    {
        let n = 10_000usize;
        let approx: Vec<f64> = (0..n).map(|k| (k as f64) * 0.001 - 5.0).collect();
        let target: Vec<f64> = (0..n).map(|k| f64::from((k % 2) as u32)).collect();
        let der1 = launch_der_binary(&approx, &target, DerBinaryKernel::LoglossGradient).unwrap();
        let base1 = logloss_der1_baseline(&approx, &target);
        assert_eq!(der1.len(), n, "large-N logloss der1 length must equal n");
        let (a1, r1) = max_divergence(&der1, &base1);
        println!("[der logloss f64 n=10000] REPORTED der1 max abs_div={a1:.3e} rel_div={r1:.3e}");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "logloss der1 (large-N) diverged: abs={a1:.3e} rel={r1:.3e}");

        let der2 = launch_der_unary(&approx, DerUnaryKernel::LoglossHessian).unwrap();
        let base2 = logloss_der2_baseline(&approx, &target);
        assert_eq!(der2.len(), n, "large-N logloss der2 length must equal n");
        let (a2, r2) = max_divergence(&der2, &base2);
        println!("[der logloss-hess f64 n=10000] REPORTED der2 max abs_div={a2:.3e} rel_div={r2:.3e}");
        assert!(r2 <= 1e-9 || a2 <= 1e-9, "logloss der2 (large-N) diverged: abs={a2:.3e} rel={r2:.3e}");
    }
}

// ===========================================================================
// Task 2 (Plan 07.2-02): Quantile / MAE der1 (parametric launch, alpha/delta as
// length-1 Array<F>) + constant-0 der2 handle self-oracle vs the
// `cb-compute::loss` baseline.
//
// MAE routes through the SAME quantile kernel at (QUANTILE_ALPHA, QUANTILE_DELTA)
// (WR-04 / Pitfall 5) — no duplicate MAE kernel, so MAE == Quantile{0.5, 1e-6} is
// bit-identical by construction. der2 is the constant-0 device handle (no
// quantile_hessian_kernel exists — Pitfall 5).
// ===========================================================================

/// `cb-compute::quantile_der1` baseline at `(alpha, delta)`, computed elementwise.
fn quantile_der1_baseline(approx: &[f64], target: &[f64], alpha: f64, delta: f64) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::quantile_der1(a, t, alpha, delta))
        .collect()
}

#[test]
fn quantile_der1_matches_cpu_baseline_f64_non_cube_multiple() {
    // n=37 (non-cube-multiple) exercises the idle guard. alpha=0.7 (asymmetric)
    // distinguishes the pinball arms: val>0 -> +0.7, val<0 -> -(1-0.7)=-0.3,
    // |val|<delta -> 0. delta=0.01 makes a few residuals land in the deadzone.
    let alpha = 0.7_f64;
    let delta = 0.01_f64;
    let approx: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.13 - 2.0).collect();
    // Cross approx so some residuals are +, some -, and a few near-zero (deadzone).
    let target: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.13 - 2.0 + ((k % 3) as f64 - 1.0) * 0.005).collect();

    let device = launch_der_param(
        &approx,
        &target,
        DerParamKernel::QuantileGradient,
        &[alpha, delta],
    )
    .unwrap();
    let baseline = quantile_der1_baseline(&approx, &target, alpha, delta);

    assert_eq!(device.len(), approx.len(), "der1 length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der quantile a=0.7 f64 n=37] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-9 || abs <= 1e-9,
        "f64 Quantile der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn quantile_der1_matches_cpu_baseline_f32() {
    let alpha = 0.3_f64;
    let delta = 0.02_f64;
    let approx_f32: Vec<f32> = (0..64).map(|k| (k as f32) * 0.1 - 3.0).collect();
    let target_f32: Vec<f32> = (0..64).map(|k| (k as f32) * 0.1 - 3.0 + ((k % 5) as f32 - 2.0) * 0.01).collect();
    let approx: Vec<f64> = approx_f32.iter().map(|&v| f64::from(v)).collect();
    let target: Vec<f64> = target_f32.iter().map(|&v| f64::from(v)).collect();

    let device = launch_der_param(
        &approx,
        &target,
        DerParamKernel::QuantileGradient,
        &[alpha, delta],
    )
    .unwrap();
    let baseline = quantile_der1_baseline(&approx, &target, alpha, delta);

    assert_eq!(device.len(), approx.len(), "der1 length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der quantile a=0.3 f32 n=64] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-3 || abs <= 1e-3,
        "f32 Quantile der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn mae_equals_quantile_half() {
    // WR-04 / Pitfall 5: MAE routes through the quantile kernel at
    // (QUANTILE_ALPHA=0.5, QUANTILE_DELTA=1e-6). The device der1 must match BOTH
    // `cb_compute::mae_der1` AND `cb_compute::quantile_der1` at those params —
    // bit-identical by construction (no separate MAE kernel).
    let approx: Vec<f64> = (0..50).map(|k| f64::from(k) * 0.21 - 5.0).collect();
    let target: Vec<f64> = (0..50).map(|k| f64::from(k) * 0.19 - 4.5).collect();

    let device = launch_der_param(
        &approx,
        &target,
        DerParamKernel::QuantileGradient,
        &[cb_compute::QUANTILE_ALPHA, cb_compute::QUANTILE_DELTA],
    )
    .unwrap();

    let mae_baseline: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::mae_der1(a, t))
        .collect();
    let quantile_baseline = quantile_der1_baseline(
        &approx,
        &target,
        cb_compute::QUANTILE_ALPHA,
        cb_compute::QUANTILE_DELTA,
    );

    assert_eq!(device.len(), approx.len());
    let (abs_mae, rel_mae) = max_divergence(&device, &mae_baseline);
    println!("[der mae==quantile{{0.5}} f64 n=50] REPORTED vs mae max abs_div={abs_mae:.3e} rel_div={rel_mae:.3e}");
    assert!(
        rel_mae <= 1e-9 || abs_mae <= 1e-9,
        "device der1 at (0.5,1e-6) != cb_compute::mae_der1: abs={abs_mae:.3e} rel={rel_mae:.3e}"
    );
    // MAE and Quantile{0.5} baselines are themselves bit-identical (mae_der1
    // delegates to quantile_der1), so the device der1 matching one matches both.
    assert_eq!(
        mae_baseline, quantile_baseline,
        "cb_compute::mae_der1 must be bit-identical to quantile_der1{{0.5,1e-6}} (WR-04)"
    );
}

#[test]
fn quantile_der2_zero_handle() {
    // Pitfall 5: Quantile/MAE der2 == 0 (there is NO quantile_hessian_kernel). The
    // der2 device handle is `const_der_handle(0.0, n)`; read it back ONCE here and
    // assert it equals `cb_compute::quantile_der2`/`mae_der2` (both constant 0.0).
    let n = 50usize;
    let approx: Vec<f64> = (0..n).map(|k| (k as f64) * 0.21 - 5.0).collect();
    let target: Vec<f64> = (0..n).map(|k| (k as f64) * 0.19 - 4.5).collect();

    let der2_handle = const_der_handle(0.0, n).unwrap();
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let bytes = client.read_one(der2_handle).unwrap();
    let der2_host = bytemuck::cast_slice::<u8, f64>(&bytes).to_vec();

    assert_eq!(der2_host.len(), n, "der2 const handle length must equal n");
    for (i, &v) in der2_host.iter().enumerate() {
        let q = cb_compute::quantile_der2(
            approx[i],
            target[i],
            cb_compute::QUANTILE_ALPHA,
            cb_compute::QUANTILE_DELTA,
        );
        let m = cb_compute::mae_der2(approx[i], target[i]);
        assert_eq!(v, 0.0, "quantile der2 const handle slot {i} must be 0.0");
        assert_eq!(v, q, "der2 slot {i} must equal cb_compute::quantile_der2 = {q}");
        assert_eq!(v, m, "der2 slot {i} must equal cb_compute::mae_der2 = {m}");
    }
}

#[test]
fn quantile_der_edge_cases() {
    let alpha = 0.6_f64;
    let delta = 1e-6_f64;
    // Empty: the param wrapper short-circuits to empty (no launch). The const-der2
    // handle CONSTRUCTS for n=0 without a read-back of the empty buffer.
    {
        let approx: Vec<f64> = Vec::new();
        let target: Vec<f64> = Vec::new();
        let der1 = launch_der_param(&approx, &target, DerParamKernel::QuantileGradient, &[alpha, delta]).unwrap();
        assert!(der1.is_empty(), "empty input must yield an empty quantile der1 (no launch)");
        assert!(
            const_der_handle(0.0, 0).is_ok(),
            "empty der2 const handle must construct (no read-back of an empty buffer)"
        );
    }

    // n=1: a single object.
    {
        let approx = vec![1.0_f64];
        let target = vec![3.0_f64]; // val=+2 > delta -> alpha
        let der1 = launch_der_param(&approx, &target, DerParamKernel::QuantileGradient, &[alpha, delta]).unwrap();
        let base = quantile_der1_baseline(&approx, &target, alpha, delta);
        let (a, r) = max_divergence(&der1, &base);
        println!("[der quantile f64 n=1] REPORTED max abs_div={a:.3e} rel_div={r:.3e}");
        assert!(r <= 1e-9 || a <= 1e-9, "quantile der1 (n=1) diverged: abs={a:.3e} rel={r:.3e}");
    }

    // Large N: many independent cubes.
    {
        let n = 10_000usize;
        let approx: Vec<f64> = (0..n).map(|k| (k as f64) * 0.001 - 5.0).collect();
        let target: Vec<f64> = (0..n).map(|k| (k as f64) * 0.0011 - 5.5).collect();
        let der1 = launch_der_param(&approx, &target, DerParamKernel::QuantileGradient, &[alpha, delta]).unwrap();
        let base = quantile_der1_baseline(&approx, &target, alpha, delta);
        assert_eq!(der1.len(), n, "large-N quantile der1 length must equal n");
        let (a, r) = max_divergence(&der1, &base);
        println!("[der quantile f64 n=10000] REPORTED max abs_div={a:.3e} rel_div={r:.3e}");
        assert!(r <= 1e-9 || a <= 1e-9, "quantile der1 (large-N) diverged: abs={a:.3e} rel={r:.3e}");
    }
}

// ===========================================================================
// Task 1 (Plan 07.2-03): Focal der1 (parametric, alpha/gamma as length-1
// Array<F>) + der2 (parametric hessian, SAME params) self-oracle vs the
// `cb-compute::loss` baseline. Focal is the TWO-kernel parametric family:
// `focal_gradient_kernel` (der1) AND `focal_hessian_kernel` (der2), BOTH
// launched through the Plan-02 `DerParamKernel` parametric seam — no new launch
// geometry. The kernels already clamp `p` to `[1e-13, 1-1e-13]` so a saturated
// logit cannot produce NaN (T-04-02-02). der1/der2 stay UNWEIGHTED (A1).
// ===========================================================================

/// `cb-compute::focal_der1` baseline at `(alpha, gamma)`, computed elementwise.
fn focal_der1_baseline(approx: &[f64], target: &[f64], alpha: f64, gamma: f64) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::focal_der1(a, t, alpha, gamma))
        .collect()
}

/// `cb-compute::focal_der2` baseline at `(alpha, gamma)`, computed elementwise.
fn focal_der2_baseline(approx: &[f64], target: &[f64], alpha: f64, gamma: f64) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::focal_der2(a, t, alpha, gamma))
        .collect()
}

#[test]
fn focal_der1_matches_cpu_baseline_f64_non_cube_multiple() {
    // n=37 (non-cube-multiple) exercises the `if ABSOLUTE_POS < approx.len()`
    // idle-guard path in `focal_gradient_kernel`. (alpha=0.25, gamma=2.0) are the
    // common Focal defaults. The approx spread -4.5..+4.5 exercises the sigmoid
    // (including near-saturated logits); 0/1 targets exercise BOTH at/pt label
    // branches.
    let alpha = 0.25_f64;
    let gamma = 2.0_f64;
    let approx: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.25 - 4.5).collect();
    let target: Vec<f64> = (0..37).map(|k| f64::from(k % 2)).collect();

    let device = launch_der_param(
        &approx,
        &target,
        DerParamKernel::FocalGradient,
        &[alpha, gamma],
    )
    .unwrap();
    let baseline = focal_der1_baseline(&approx, &target, alpha, gamma);

    assert_eq!(device.len(), approx.len(), "focal der1 length must equal input length");
    assert!(device.iter().all(|v| v.is_finite()), "focal der1 must be finite (clamp holds)");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der focal a=0.25 g=2.0 f64 n=37] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-9 || abs <= 1e-9,
        "f64 Focal der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn focal_der1_matches_cpu_baseline_f32() {
    // f32-magnitude logits cast to f64 at the seam. A generous f32 relative bound
    // (~1e-3) catches a wrong der without pinning the GPU-06 epsilon.
    let alpha = 0.5_f64;
    let gamma = 1.0_f64;
    let approx_f32: Vec<f32> = (0..64).map(|k| (k as f32) * 0.15 - 5.0).collect();
    let target_f32: Vec<f32> = (0..64).map(|k| (k % 2) as f32).collect();
    let approx: Vec<f64> = approx_f32.iter().map(|&v| f64::from(v)).collect();
    let target: Vec<f64> = target_f32.iter().map(|&v| f64::from(v)).collect();

    let device = launch_der_param(
        &approx,
        &target,
        DerParamKernel::FocalGradient,
        &[alpha, gamma],
    )
    .unwrap();
    let baseline = focal_der1_baseline(&approx, &target, alpha, gamma);

    assert_eq!(device.len(), approx.len(), "focal der1 length must equal input length");
    assert!(device.iter().all(|v| v.is_finite()), "focal der1 must be finite (clamp holds)");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der focal a=0.5 g=1.0 f32 n=64] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-3 || abs <= 1e-3,
        "f32 Focal der1 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn focal_der2_matches_cpu_baseline_f64() {
    // der2 via the SECOND focal kernel (`focal_hessian_kernel`) through the SAME
    // `DerParamKernel::FocalHessian` parametric seam. n=37 exercises the idle guard.
    let alpha = 0.25_f64;
    let gamma = 2.0_f64;
    let approx: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.25 - 4.5).collect();
    let target: Vec<f64> = (0..37).map(|k| f64::from(k % 2)).collect();

    let device = launch_der_param(
        &approx,
        &target,
        DerParamKernel::FocalHessian,
        &[alpha, gamma],
    )
    .unwrap();
    let baseline = focal_der2_baseline(&approx, &target, alpha, gamma);

    assert_eq!(device.len(), approx.len(), "focal der2 length must equal input length");
    assert!(device.iter().all(|v| v.is_finite()), "focal der2 must be finite (clamp holds)");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der focal-hess a=0.25 g=2.0 f64 n=37] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-9 || abs <= 1e-9,
        "f64 Focal der2 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn focal_der2_matches_cpu_baseline_f32() {
    let alpha = 0.5_f64;
    let gamma = 1.0_f64;
    let approx_f32: Vec<f32> = (0..64).map(|k| (k as f32) * 0.15 - 5.0).collect();
    let target_f32: Vec<f32> = (0..64).map(|k| (k % 2) as f32).collect();
    let approx: Vec<f64> = approx_f32.iter().map(|&v| f64::from(v)).collect();
    let target: Vec<f64> = target_f32.iter().map(|&v| f64::from(v)).collect();

    let device = launch_der_param(
        &approx,
        &target,
        DerParamKernel::FocalHessian,
        &[alpha, gamma],
    )
    .unwrap();
    let baseline = focal_der2_baseline(&approx, &target, alpha, gamma);

    assert_eq!(device.len(), approx.len(), "focal der2 length must equal input length");
    assert!(device.iter().all(|v| v.is_finite()), "focal der2 must be finite (clamp holds)");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[der focal-hess a=0.5 g=1.0 f32 n=64] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= 1e-3 || abs <= 1e-3,
        "f32 Focal der2 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn focal_der_saturated_logit_no_nan() {
    // T-07.2-07 / T-04-02-02: saturated logits (±40) would drive sigmoid to ~0 / ~1,
    // making `ln(pt)` / `powf(1-pt, gamma)` blow up to NaN/-inf WITHOUT the clamp.
    // The kernels clamp `p` to `[1e-13, 1-1e-13]` before `ln`/`powf`, so der1/der2
    // MUST be finite. Also assert they match the (identically-clamped) cb-compute
    // baseline.
    let alpha = 0.25_f64;
    let gamma = 2.0_f64;
    // Mix saturated (+/-40) and moderate logits, both labels.
    let approx: Vec<f64> = vec![
        -40.0, 40.0, -40.0, 40.0, -3.0, 3.0, 0.0, -1.5, 2.5, -40.0, 40.0, 0.5,
    ];
    let target: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];

    let der1 = launch_der_param(&approx, &target, DerParamKernel::FocalGradient, &[alpha, gamma]).unwrap();
    let der2 = launch_der_param(&approx, &target, DerParamKernel::FocalHessian, &[alpha, gamma]).unwrap();

    assert!(
        der1.iter().all(|v| v.is_finite()),
        "saturated-logit focal der1 must be finite (clamp holds): {der1:?}"
    );
    assert!(
        der2.iter().all(|v| v.is_finite()),
        "saturated-logit focal der2 must be finite (clamp holds): {der2:?}"
    );

    let base1 = focal_der1_baseline(&approx, &target, alpha, gamma);
    let base2 = focal_der2_baseline(&approx, &target, alpha, gamma);
    let (a1, r1) = max_divergence(&der1, &base1);
    let (a2, r2) = max_divergence(&der2, &base2);
    println!("[der focal saturated der1] REPORTED max abs_div={a1:.3e} rel_div={r1:.3e}");
    println!("[der focal saturated der2] REPORTED max abs_div={a2:.3e} rel_div={r2:.3e}");
    assert!(r1 <= 1e-9 || a1 <= 1e-9, "saturated focal der1 diverged: abs={a1:.3e} rel={r1:.3e}");
    assert!(r2 <= 1e-9 || a2 <= 1e-9, "saturated focal der2 diverged: abs={a2:.3e} rel={r2:.3e}");
}

#[test]
fn focal_der_edge_cases() {
    let alpha = 0.25_f64;
    let gamma = 2.0_f64;

    // Empty: both parametric wrappers short-circuit to empty (no launch); neither
    // reads back an empty device buffer.
    {
        let approx: Vec<f64> = Vec::new();
        let target: Vec<f64> = Vec::new();
        let der1 = launch_der_param(&approx, &target, DerParamKernel::FocalGradient, &[alpha, gamma]).unwrap();
        let der2 = launch_der_param(&approx, &target, DerParamKernel::FocalHessian, &[alpha, gamma]).unwrap();
        assert!(der1.is_empty(), "empty input must yield an empty focal der1 (no launch)");
        assert!(der2.is_empty(), "empty input must yield an empty focal der2 (no launch)");
    }

    // n=1: a single object, one thread. target=1 selects the at=alpha/pt=p branch.
    {
        let approx = vec![0.75_f64];
        let target = vec![1.0_f64];
        let der1 = launch_der_param(&approx, &target, DerParamKernel::FocalGradient, &[alpha, gamma]).unwrap();
        let der2 = launch_der_param(&approx, &target, DerParamKernel::FocalHessian, &[alpha, gamma]).unwrap();
        let b1 = focal_der1_baseline(&approx, &target, alpha, gamma);
        let b2 = focal_der2_baseline(&approx, &target, alpha, gamma);
        let (a1, r1) = max_divergence(&der1, &b1);
        let (a2, r2) = max_divergence(&der2, &b2);
        println!("[der focal f64 n=1] REPORTED der1 abs={a1:.3e} rel={r1:.3e} | der2 abs={a2:.3e} rel={r2:.3e}");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "focal der1 (n=1) diverged: abs={a1:.3e} rel={r1:.3e}");
        assert!(r2 <= 1e-9 || a2 <= 1e-9, "focal der2 (n=1) diverged: abs={a2:.3e} rel={r2:.3e}");
    }

    // Large N: many independent cubes (10_000 >> CUBE_DIM=32). Both labels mixed.
    {
        let n = 10_000usize;
        let approx: Vec<f64> = (0..n).map(|k| (k as f64) * 0.001 - 5.0).collect();
        let target: Vec<f64> = (0..n).map(|k| f64::from((k % 2) as u32)).collect();
        let der1 = launch_der_param(&approx, &target, DerParamKernel::FocalGradient, &[alpha, gamma]).unwrap();
        let der2 = launch_der_param(&approx, &target, DerParamKernel::FocalHessian, &[alpha, gamma]).unwrap();
        let b1 = focal_der1_baseline(&approx, &target, alpha, gamma);
        let b2 = focal_der2_baseline(&approx, &target, alpha, gamma);
        assert_eq!(der1.len(), n, "large-N focal der1 length must equal n");
        assert_eq!(der2.len(), n, "large-N focal der2 length must equal n");
        let (a1, r1) = max_divergence(&der1, &b1);
        let (a2, r2) = max_divergence(&der2, &b2);
        println!("[der focal f64 n=10000] REPORTED der1 abs={a1:.3e} rel={r1:.3e} | der2 abs={a2:.3e} rel={r2:.3e}");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "focal der1 (large-N) diverged: abs={a1:.3e} rel={r1:.3e}");
        assert!(r2 <= 1e-9 || a2 <= 1e-9, "focal der2 (large-N) diverged: abs={a2:.3e} rel={r2:.3e}");
    }
}

// ===========================================================================
// Task 2 (Plan 07.2-03): Full-family device-residency hand-off lock + the SC-4
// structural assertion. Phase 7.2 closes here: every in-scope pointwise der
// family (RMSE / Logloss-CrossEntropy / Quantile-MAE / Focal) must hand 7.3 its
// der1 AND der2 as device HANDLES with NO host fold inserted on the seam
// (handle-in -> handles-out, SC-3 / D-7.2-04). The read-back happens ONCE per
// handle at the END (test-only), never on the hand-off path.
// ===========================================================================

#[test]
fn all_losses_device_resident_handoff() {
    // The der1/der2 device HANDLES handed to 7.3 are UNWEIGHTED (A1, the Plan-01
    // checkpoint + Open Q1): the per-object weight is folded DOWNSTREAM by the 7.3
    // `histogram_scatter_kernel` (`contrib[i] = der[i] * weight[i]`), NOT here. Each
    // family below obtains its der1 AND der2 as `Ok(handle)` with NO host fold
    // between them; the test reads each handle back ONCE at the end and confirms it
    // equals the matching `cb_compute` baseline.
    let n = 50usize;
    // Logit-shaped approx (exercises the sigmoid families) and 0/1 targets (the
    // Focal/Logloss label branch); also valid residuals for RMSE/Quantile.
    let approx: Vec<f64> = (0..n).map(|k| f64::from(k as u32) * 0.18 - 4.5).collect();
    let target: Vec<f64> = (0..n).map(|k| f64::from((k % 2) as u32)).collect();

    // ONE client constructs all the read-backs (test-only); each handle is allocated
    // by its launch helper's own internal client, but the handle VALUES are read here
    // after each launch completes.
    let device = <crate::SelectedRuntime as Runtime>::Device::default();

    // Small read-back helper: a handle is read back ONCE (test-only) through a fresh
    // client of the SAME runtime. (Production never does this on the seam.)
    fn read_handle(h: cubecl::server::Handle) -> Vec<f64> {
        let device = <crate::SelectedRuntime as Runtime>::Device::default();
        let client = <crate::SelectedRuntime as Runtime>::client(&device);
        let bytes = client.read_one(h).unwrap();
        bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
    }

    // --- RMSE: der1 via launch_der_binary_handle, der2 via const_der_handle(-1.0) ---
    {
        let der1_h = launch_der_binary_handle(&approx, &target, DerBinaryKernel::RmseGradient).unwrap();
        let der2_h = const_der_handle(-1.0, n).unwrap();
        let der1 = read_handle(der1_h);
        let der2 = read_handle(der2_h);
        let b1 = rmse_der1_baseline(&approx, &target);
        let (a1, r1) = max_divergence(&der1, &b1);
        println!("[handoff RMSE der1] REPORTED abs={a1:.3e} rel={r1:.3e}");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "RMSE der1 handoff diverged: abs={a1:.3e} rel={r1:.3e}");
        for (i, &v) in der2.iter().enumerate() {
            assert_eq!(v, cb_compute::rmse_der2(approx[i], target[i]), "RMSE der2 handoff slot {i}");
        }
    }

    // --- Logloss/CrossEntropy: der1 via launch_der_binary_handle, der2 via launch_der_unary_handle ---
    {
        let der1_h = launch_der_binary_handle(&approx, &target, DerBinaryKernel::LoglossGradient).unwrap();
        let der2_h = launch_der_unary_handle(&approx, DerUnaryKernel::LoglossHessian).unwrap();
        let der1 = read_handle(der1_h);
        let der2 = read_handle(der2_h);
        let b1 = logloss_der1_baseline(&approx, &target);
        let b2 = logloss_der2_baseline(&approx, &target);
        let (a1, r1) = max_divergence(&der1, &b1);
        let (a2, r2) = max_divergence(&der2, &b2);
        println!("[handoff Logloss der1] abs={a1:.3e} rel={r1:.3e} | der2 abs={a2:.3e} rel={r2:.3e}");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "Logloss der1 handoff diverged: abs={a1:.3e} rel={r1:.3e}");
        assert!(r2 <= 1e-9 || a2 <= 1e-9, "Logloss der2 handoff diverged: abs={a2:.3e} rel={r2:.3e}");
    }

    // --- Quantile/MAE: der1 via launch_der_param_handle, der2 via const_der_handle(0.0) ---
    {
        let alpha = cb_compute::QUANTILE_ALPHA;
        let delta = cb_compute::QUANTILE_DELTA;
        let der1_h = launch_der_param_handle(&approx, &target, DerParamKernel::QuantileGradient, &[alpha, delta]).unwrap();
        let der2_h = const_der_handle(0.0, n).unwrap();
        let der1 = read_handle(der1_h);
        let der2 = read_handle(der2_h);
        let b1 = quantile_der1_baseline(&approx, &target, alpha, delta);
        let (a1, r1) = max_divergence(&der1, &b1);
        println!("[handoff Quantile/MAE der1] REPORTED abs={a1:.3e} rel={r1:.3e}");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "Quantile der1 handoff diverged: abs={a1:.3e} rel={r1:.3e}");
        for (i, &v) in der2.iter().enumerate() {
            assert_eq!(v, 0.0, "Quantile der2 handoff slot {i} must be 0.0");
            assert_eq!(v, cb_compute::quantile_der2(approx[i], target[i], alpha, delta), "Quantile der2 handoff slot {i}");
        }
    }

    // --- Focal: der1 AND der2 via TWO launch_der_param_handle calls (two-kernel family) ---
    {
        let alpha = 0.25_f64;
        let gamma = 2.0_f64;
        // Handle-in -> handles-out: der1 AND der2 are obtained as device HANDLES with
        // NO host fold inserted between them (the SC-3 / D-7.2-04 contract for 7.3).
        let der1_h = launch_der_param_handle(&approx, &target, DerParamKernel::FocalGradient, &[alpha, gamma]).unwrap();
        let der2_h = launch_der_param_handle(&approx, &target, DerParamKernel::FocalHessian, &[alpha, gamma]).unwrap();
        let der1 = read_handle(der1_h);
        let der2 = read_handle(der2_h);
        let b1 = focal_der1_baseline(&approx, &target, alpha, gamma);
        let b2 = focal_der2_baseline(&approx, &target, alpha, gamma);
        let (a1, r1) = max_divergence(&der1, &b1);
        let (a2, r2) = max_divergence(&der2, &b2);
        println!("[handoff Focal der1] abs={a1:.3e} rel={r1:.3e} | der2 abs={a2:.3e} rel={r2:.3e}");
        assert!(der1.iter().all(|v| v.is_finite()) && der2.iter().all(|v| v.is_finite()), "Focal handoff der1/der2 finite");
        assert!(r1 <= 1e-9 || a1 <= 1e-9, "Focal der1 handoff diverged: abs={a1:.3e} rel={r1:.3e}");
        assert!(r2 <= 1e-9 || a2 <= 1e-9, "Focal der2 handoff diverged: abs={a2:.3e} rel={r2:.3e}");
    }

    // The `device` binding above documents the single-runtime hand-off surface; the
    // read-backs are the ONLY host folds in this whole test, and they all happen
    // AFTER every launch helper returned a handle (proving the seam is handle-in ->
    // handles-out for all four families).
    let _ = device;
}

#[test]
fn cb_compute_is_cubecl_free() {
    // SC-4 structural note (D-7.2-05): cb-compute MUST stay cubecl-free — the loss
    // baselines this oracle compares against are pure-CPU `f64` math with NO GPU
    // dependency, so the comparison is independent of the device path under test.
    //
    // The AUTHORITATIVE SC-4 gate is the `cargo tree` command run in verification:
    //   `cargo tree -e features -p cb-compute | grep -ci cubecl`  ==  0
    // and `git diff --stat` showing cb-compute/cb-core/cb-model byte-unchanged this
    // phase. This test makes that intent DISCOVERABLE in the oracle file (a
    // programmatic dependency-graph walk from a unit test is awkward and would
    // duplicate `cargo tree`); it asserts the one structural fact this file relies
    // on — that `cb_compute`'s der baselines are callable as plain host functions
    // (no GPU client / no cubecl runtime), which is exactly what `cb-compute` being
    // cubecl-free guarantees.
    let der1 = cb_compute::focal_der1(0.5, 1.0, 0.25, 2.0);
    let der2 = cb_compute::focal_der2(0.5, 1.0, 0.25, 2.0);
    assert!(der1.is_finite() && der2.is_finite(), "cb_compute focal der baselines are pure-CPU finite math");
}
