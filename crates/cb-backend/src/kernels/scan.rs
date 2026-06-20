//! Self-oracle for the block-scan kernel (GPU-01 scan, D-7.1-06): the device
//! inclusive/exclusive prefix-scan must match a Rust CPU prefix-sum within a
//! REPORTED (not signed-off) tolerance, over f32 and f64 inputs including the
//! n=1 edge case.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs`; all assertions live here. Test code may use `.unwrap()`/indexing
//! (the `lib.rs:1` `#[cfg(test)]` allow) — production `gpu_runtime.rs` may not.
//!
//! SCOPE (RESEARCH Open Q2): `block_scan_kernel` performs the scan WITHIN a single
//! cube; the cross-cube running carry is the documented first forward dependency
//! for 7.2/7.3. The oracle therefore exercises N <= CUBE_DIM (32) — exactly one
//! cube/plane on wave32 gfx1100, where the within-plane plane-op prefix is the
//! whole answer and the Hillis-Steele cross-plane carry collapses to the identity.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32). The reported max abs/rel
//! divergence is informational: the GPU-06 epsilon is signed off in Phase 7.6, NOT
//! hard-coded here (D-7.1-07/09). The asserted tolerances are generous, run-stable
//! bounds (f32 ~1e-3 relative, f64 ~1e-9 relative) that catch a wrong scan without
//! pinning the final epsilon.

use cubecl::prelude::*;

use crate::kernels::block_scan_kernel;

/// Launch geometry: one cube of CUBE_DIM units. The oracle is scoped to a single
/// cube (N <= CUBE_DIM, Open Q2), so a single static cube is launched.
const CUBE_DIM: usize = 32;

// IN-03: the "generous, run-stable" oracle bounds, hoisted into named consts shared
// across this module's assertions so the Phase-7.6 epsilon sign-off edits ONE place
// (and so the reduce and scan oracles cannot drift apart). These are NOT the final
// GPU-06 epsilon — they only catch a wrong scan without pinning the signed-off bound.

/// Generous relative bound for an f32 device scan vs the f64 CPU baseline.
const F32_REL_TOL: f64 = 1e-3;
/// Generous absolute bound for an f32 device scan vs the f64 CPU baseline.
const F32_ABS_TOL: f64 = 1e-3;
/// Generous relative bound for an f64 device scan vs the f64 CPU baseline.
const F64_REL_TOL: f64 = 1e-9;
/// Tight absolute bound for the small/exact f64 cases (n=1, exclusive scan[0] == 0).
const F64_ABS_TOL_TIGHT: f64 = 1e-12;

/// Launch `block_scan_kernel::<F>` on the selected runtime and read back the
/// per-element prefix-scan. `inclusive` is passed through as the kernel's comptime
/// flag. The output is the SAME length as the input (a scan is not a reduction).
fn run_scan<F>(input: &[F], inclusive: bool) -> Vec<F>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = input.len();
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    // Single cube covers N <= CUBE_DIM (the documented oracle scope).
    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let out_handle = client.empty(n * std::mem::size_of::<F>());

    block_scan_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: CUBE_DIM as u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        inclusive,
    );

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

/// Rust CPU inclusive prefix-sum (running total includes self): the parity baseline.
fn cpu_inclusive_scan(input: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0.0_f64;
    for &v in input {
        acc += v;
        out.push(acc);
    }
    out
}

/// Rust CPU exclusive prefix-sum (each element = sum of strictly-prior elements;
/// `output[0] == 0`): the parity baseline.
fn cpu_exclusive_scan(input: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0.0_f64;
    for &v in input {
        out.push(acc);
        acc += v;
    }
    out
}

/// Compare the device scan (cast to f64) to the CPU baseline element-wise,
/// returning the max abs and max rel divergence over the vector.
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    // IN-02: zip the two slices so the equal-length precondition is structural (and
    // consistent with the already-hardened sibling, commit 252c33a / IN-03 in 07.3)
    // rather than relying on `device[i]` not panicking when `device` is shorter.
    debug_assert_eq!(
        device.len(),
        baseline.len(),
        "max_divergence requires device and baseline to be equal length"
    );
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (d, b) in device.iter().zip(baseline) {
        let abs = (d - b).abs();
        let rel = if b.abs() > 0.0 { abs / b.abs() } else { abs };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

#[test]
fn block_scan_inclusive_matches_cpu_prefix_sum_f64() {
    // A single-cube length (24 <= CUBE_DIM) with a mix of signs/magnitudes.
    let input: Vec<f64> = (0..24).map(|k| f64::from(k) * 0.5 - 3.0).collect();

    let device = run_scan(&input, true);
    let baseline = cpu_inclusive_scan(&input);

    assert_eq!(device.len(), input.len(), "scan output length must equal input length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[scan f64 inclusive n=24] max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= F64_REL_TOL || abs <= F64_REL_TOL,
        "f64 inclusive scan diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn block_scan_exclusive_matches_cpu_prefix_sum_f64() {
    let input: Vec<f64> = (0..24).map(|k| f64::from(k) * 0.5 - 3.0).collect();

    let device = run_scan(&input, false);
    let baseline = cpu_exclusive_scan(&input);

    assert_eq!(device.len(), input.len());
    // Exclusive scan: the first element MUST be exactly 0 (sum of nothing prior).
    assert!(
        device[0].abs() <= F64_ABS_TOL_TIGHT,
        "exclusive scan[0] must be 0.0, got {}",
        device[0]
    );
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[scan f64 exclusive n=24] device[0]={} max abs_div={abs:.3e} rel_div={rel:.3e}", device[0]);
    assert!(
        rel <= F64_REL_TOL || abs <= F64_REL_TOL,
        "f64 exclusive scan diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn block_scan_inclusive_matches_cpu_prefix_sum_f32() {
    let input: Vec<f32> = (0..30).map(|k| ((k % 7) as f32) - 3.0 + 0.25 * (k as f32)).collect();

    let device_f32 = run_scan(&input, true);
    let device: Vec<f64> = device_f32.iter().map(|&v| f64::from(v)).collect();
    let input_f64: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
    let baseline = cpu_inclusive_scan(&input_f64);

    assert_eq!(device.len(), input.len());
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[scan f32 inclusive n=30] max abs_div={abs:.3e} rel_div={rel:.3e}");
    // f32 device scan vs f64 baseline: a generous, run-stable relative bound.
    assert!(
        rel <= F32_REL_TOL || abs <= F32_ABS_TOL,
        "f32 inclusive scan diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn block_scan_edge_case_single_element() {
    // n = 1: inclusive scan = [x]; exclusive scan = [0].
    let input = vec![42.5_f64];

    let incl = run_scan(&input, true);
    assert_eq!(incl.len(), 1);
    assert!(
        (incl[0] - 42.5).abs() <= F64_ABS_TOL_TIGHT,
        "n=1 inclusive scan must be [x]={}, got {}",
        42.5,
        incl[0]
    );

    let excl = run_scan(&input, false);
    assert_eq!(excl.len(), 1);
    assert!(
        excl[0].abs() <= F64_ABS_TOL_TIGHT,
        "n=1 exclusive scan must be [0], got {}",
        excl[0]
    );
    println!("[scan f64 n=1] inclusive={} exclusive={}", incl[0], excl[0]);
}
