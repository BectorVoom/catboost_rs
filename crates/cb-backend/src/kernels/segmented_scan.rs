//! Self-oracle for the flag-array segmented-scan kernel (GPUT-16, Plan 10-01, D-02):
//! the device inclusive/exclusive SEGMENTED prefix-scan must reset the running sum at
//! each segment boundary and match an inline serial segmented-prefix reference within a
//! REPORTED (not signed-off) tolerance.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the `#[cube]` kernel
//! lives in `kernels.rs`; ALL assertions / `.unwrap()` / indexing live here (the
//! `lib.rs:1` `#[cfg(test)]` allow). The serial reference is transcribed INLINE — no
//! `cb-train` reach, no upstream/CUB fixture (D-02).
//!
//! SCOPE (mirrors `kernels/scan.rs` Open Q2): `segmented_scan_kernel` performs the scan
//! WITHIN a single cube; the cross-cube segmented carry is the documented forward
//! dependency. The oracle therefore exercises `n <= CUBE_DIM (32)` — exactly one cube on
//! wave32 gfx1100. This runs on `rocm` in-env; the reported divergence is informational
//! (the GPU-06 epsilon is signed off in Phase 7.6, NOT hard-coded here). The asserted
//! bounds are generous, run-stable values that catch a wrong segmented scan without
//! pinning the final epsilon.

use cubecl::prelude::*;

use crate::kernels::segmented_scan_kernel;

/// Launch geometry: one cube of CUBE_DIM units (single-cube scope, `n <= CUBE_DIM`).
const CUBE_DIM: usize = 32;

/// Generous relative bound for an f64 device segmented scan vs the f64 CPU baseline.
const F64_REL_TOL: f64 = 1e-9;
/// Tight absolute bound for exact small cases (segment starts, exclusive zeros).
const F64_ABS_TOL_TIGHT: f64 = 1e-12;

/// Launch `segmented_scan_kernel::<F>` on the selected runtime and read back the
/// per-element segmented prefix-scan. `inclusive` passes through as the kernel's
/// comptime flag. Single-cube scope: `input.len() <= CUBE_DIM`.
fn run_segmented_scan<F>(input: &[F], flags: &[F], inclusive: bool) -> Vec<F>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = input.len();
    assert_eq!(n, flags.len(), "input and flags must be equal length");
    assert!(n <= CUBE_DIM, "single-cube segmented-scan oracle scope: n <= {CUBE_DIM}");

    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let flag_handle = client.create(cubecl::bytes::Bytes::from_elems(flags.to_vec()));
    let out_handle = client.empty(n * std::mem::size_of::<F>());

    segmented_scan_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim {
            x: CUBE_DIM as u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(flag_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        inclusive,
    );

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

/// Inline serial segmented INCLUSIVE prefix-sum reference (D-02): the running sum resets
/// to the current element at each segment start (`flag != 0`), otherwise accumulates.
fn cpu_segmented_inclusive(input: &[f64], flags: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0.0_f64;
    for (&v, &f) in input.iter().zip(flags) {
        if f != 0.0 {
            acc = v;
        } else {
            acc += v;
        }
        out.push(acc);
    }
    out
}

/// Inline serial segmented EXCLUSIVE prefix-sum reference (D-02): a segment start emits
/// `0` (nothing prior in the segment); otherwise the sum of strictly-prior elements
/// within the segment.
fn cpu_segmented_exclusive(input: &[f64], flags: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0.0_f64;
    for (&v, &f) in input.iter().zip(flags) {
        if f != 0.0 {
            out.push(0.0);
            acc = v;
        } else {
            out.push(acc);
            acc += v;
        }
    }
    out
}

/// Max abs / rel divergence over equal-length device vs baseline vectors.
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
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
fn segmented_scan_behaviour_example() {
    // Plan behaviour: values [1,1,1,1] with segment flags [1,0,1,0] → inclusive [1,2,1,2].
    let input = vec![1.0_f64, 1.0, 1.0, 1.0];
    let flags = vec![1.0_f64, 0.0, 1.0, 0.0];

    let device = run_segmented_scan(&input, &flags, true);
    assert_eq!(device, vec![1.0, 2.0, 1.0, 2.0], "inclusive segmented scan of the Plan example");
}

#[test]
fn segmented_scan_inclusive_matches_serial_multi_segment() {
    // A richer multi-segment case with varied magnitudes/signs (n <= CUBE_DIM).
    let input: Vec<f64> = (0..20).map(|k| f64::from(k) * 0.5 - 2.0).collect();
    //          seg starts at 0, 5, 6, 13 — segments of differing lengths (incl. length-1).
    let mut flags = vec![0.0_f64; 20];
    for &start in &[0usize, 5, 6, 13] {
        flags[start] = 1.0;
    }

    let device = run_segmented_scan(&input, &flags, true);
    let baseline = cpu_segmented_inclusive(&input, &flags);

    assert_eq!(device.len(), input.len());
    // A segment start's inclusive value must equal exactly its own input (reset).
    for (i, &f) in flags.iter().enumerate() {
        if f != 0.0 {
            assert!(
                (device[i] - input[i]).abs() <= F64_ABS_TOL_TIGHT,
                "segment start at {i} must reset to input[{i}]={}, got {}",
                input[i],
                device[i]
            );
        }
    }
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[seg-scan f64 inclusive n=20] max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= F64_REL_TOL || abs <= F64_REL_TOL,
        "f64 inclusive segmented scan diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn segmented_scan_exclusive_matches_serial_multi_segment() {
    let input: Vec<f64> = (0..20).map(|k| f64::from(k) * 0.5 - 2.0).collect();
    let mut flags = vec![0.0_f64; 20];
    for &start in &[0usize, 5, 6, 13] {
        flags[start] = 1.0;
    }

    let device = run_segmented_scan(&input, &flags, false);
    let baseline = cpu_segmented_exclusive(&input, &flags);

    assert_eq!(device.len(), input.len());
    // Every segment start's exclusive value must be exactly 0 (nothing prior in-segment).
    for (i, &f) in flags.iter().enumerate() {
        if f != 0.0 {
            assert!(
                device[i].abs() <= F64_ABS_TOL_TIGHT,
                "segment start at {i} exclusive value must be 0, got {}",
                device[i]
            );
        }
    }
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[seg-scan f64 exclusive n=20] max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= F64_REL_TOL || abs <= F64_REL_TOL,
        "f64 exclusive segmented scan diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn segmented_scan_full_cube_all_boundaries() {
    // Fill the whole cube (n == CUBE_DIM = 32) with alternating and back-to-back
    // segment heads (incl. every-element-a-segment and long runs) to exercise the
    // Hillis-Steele stride to its full 32-wide extent.
    let n = CUBE_DIM;
    let input: Vec<f64> = (0..n).map(|k| 1.0 + ((k % 5) as f64) * 0.25).collect();
    let mut flags = vec![0.0_f64; n];
    flags[0] = 1.0;
    for &start in &[1usize, 2, 3, 10, 11, 20, 31] {
        flags[start] = 1.0;
    }

    let device = run_segmented_scan(&input, &flags, true);
    let baseline = cpu_segmented_inclusive(&input, &flags);

    assert_eq!(device.len(), n);
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[seg-scan f64 inclusive n={n} full-cube] max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= F64_REL_TOL || abs <= F64_REL_TOL,
        "f64 full-cube segmented scan diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}
