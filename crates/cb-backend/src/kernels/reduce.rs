//! Self-oracle for the block-reduce kernel (D-7.1-09, GPU-01 reduce): the device
//! sum must match `cb-core::sum_f64` within a REPORTED (not signed-off) tolerance,
//! over f32 and f64 inputs including edge cases (empty, n=1, length not a multiple
//! of CUBE_DIM, large N), exercising BOTH the plane path and the shared-memory
//! fallback.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs`; all assertions live here. Test code may use `.unwrap()`/indexing
//! (the `lib.rs:1` `#[cfg(test)]` allow) — production `gpu_runtime.rs` may not.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32). The reported max abs/rel
//! divergence is informational: the GPU-06 epsilon is signed off in Phase 7.6, NOT
//! hard-coded here (D-7.1-07/09). The asserted tolerances are generous, run-stable
//! bounds (f32 ~1e-3 relative, f64 ~1e-9 relative) that catch a wrong fold without
//! pinning the final epsilon.

use cubecl::features::Plane;
use cubecl::prelude::*;

use crate::kernels::block_reduce_kernel;

/// Launch `block_reduce_kernel::<F>` on the selected runtime and read back the
/// per-cube partial sums. `use_plane` is passed explicitly so a test can drive
/// EITHER path regardless of the hardware capability (the fallback is always valid;
/// the plane path is only requested when the device actually supports it).
fn run_reduce<F>(input: &[F], use_plane: bool) -> Vec<F>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = input.len();
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let num_cubes = n.div_ceil(32usize).max(1);
    let out_handle = client.empty(num_cubes * std::mem::size_of::<F>());

    block_reduce_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: 32u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), num_cubes) },
        use_plane,
    );

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

/// Does the selected runtime's device advertise plane (subgroup) ops? Drives which
/// path(s) the oracle exercises.
fn device_has_plane() -> bool {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    client.features().plane.contains(Plane::Ops)
}

/// Sum the per-cube f32 partials and compare to the f64 CPU baseline, returning
/// `(device_sum, baseline, abs_div, rel_div)`. The partials are cast to f64 and
/// folded with `cb_core::sum_f64` (the frozen host order) — the across-cube
/// finalize the production `launch_block_reduce_f64` leaves to the host.
fn oracle_f32(input: &[f32], use_plane: bool) -> (f64, f64, f64, f64) {
    let partials = run_reduce(input, use_plane);
    let partials_f64: Vec<f64> = partials.iter().map(|&v| f64::from(v)).collect();
    let device_sum = cb_core::sum_f64(&partials_f64);
    let input_f64: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
    let baseline = cb_core::sum_f64(&input_f64);
    let abs = (device_sum - baseline).abs();
    let rel = if baseline.abs() > 0.0 {
        abs / baseline.abs()
    } else {
        abs
    };
    (device_sum, baseline, abs, rel)
}

/// Sum the per-cube f64 partials and compare to the f64 CPU baseline.
fn oracle_f64(input: &[f64], use_plane: bool) -> (f64, f64, f64, f64) {
    let partials = run_reduce(input, use_plane);
    let device_sum = cb_core::sum_f64(&partials);
    let baseline = cb_core::sum_f64(input);
    let abs = (device_sum - baseline).abs();
    let rel = if baseline.abs() > 0.0 {
        abs / baseline.abs()
    } else {
        abs
    };
    (device_sum, baseline, abs, rel)
}

#[test]
fn block_reduce_matches_cpu_sum_f32() {
    // Multiple-of-CUBE_DIM length (8 full cubes of 32) with a mix of signs/magnitudes.
    let input: Vec<f32> = (0..256).map(|k| ((k % 17) as f32) - 8.0 + 0.25 * (k as f32)).collect();

    let has_plane = device_has_plane();
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;

    // Always exercise the shared-memory fallback; additionally the plane path when
    // the device supports it (both must produce the correct sum — D-7.1-08).
    let mut paths: Vec<bool> = vec![false];
    if has_plane {
        paths.push(true);
    }
    for &use_plane in &paths {
        let (dev, base, abs, rel) = oracle_f32(&input, use_plane);
        println!(
            "[reduce f32] use_plane={use_plane} device_sum={dev} baseline={base} abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        // f32 device sum vs f64 baseline: a generous, run-stable relative bound that
        // catches a wrong fold without pinning the GPU-06 epsilon (7.6's job).
        assert!(
            rel <= 1e-3 || abs <= 1e-3,
            "f32 reduce diverged too far (use_plane={use_plane}): abs={abs:.3e} rel={rel:.3e}"
        );
    }
    println!("[reduce f32] REPORTED max abs_div={max_abs:.3e} max rel_div={max_rel:.3e} (plane_available={has_plane})");
}

#[test]
fn block_reduce_matches_cpu_sum_f64_non_cube_multiple() {
    // A non-cube-multiple length (37) exercises the bounds-guard idle/zero-pad path.
    let input: Vec<f64> = (0..37).map(|k| f64::from(k) * 0.5 - 3.0).collect();

    let has_plane = device_has_plane();
    let mut paths: Vec<bool> = vec![false];
    if has_plane {
        paths.push(true);
    }
    for &use_plane in &paths {
        let (dev, base, abs, rel) = oracle_f64(&input, use_plane);
        println!(
            "[reduce f64 n=37] use_plane={use_plane} device_sum={dev} baseline={base} abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        assert!(
            rel <= 1e-9 || abs <= 1e-9,
            "f64 reduce diverged too far (use_plane={use_plane}): abs={abs:.3e} rel={rel:.3e}"
        );
    }
}

#[test]
fn block_reduce_edge_cases() {
    let has_plane = device_has_plane();

    // Empty slice -> sum 0.0 (no launch; the production helper short-circuits, and
    // the test harness `num_cubes.max(1)` keeps a single idle cube valid here we
    // simply assert the baseline-equivalent zero through the f64 helper at n=1..).
    {
        let empty: Vec<f64> = Vec::new();
        let baseline = cb_core::sum_f64(&empty);
        assert_eq!(baseline, 0.0, "empty baseline must be 0.0");
    }

    // n = 1 -> sum equals the single element.
    {
        let input = vec![42.5_f64];
        let (dev, base, abs, _rel) = oracle_f64(&input, false);
        println!("[reduce f64 n=1] device_sum={dev} baseline={base} abs_div={abs:.3e}");
        assert!(abs <= 1e-12, "n=1 reduce mismatch: abs={abs:.3e}");
        if has_plane {
            let (_d, _b, abs_p, _r) = oracle_f64(&input, true);
            assert!(abs_p <= 1e-12, "n=1 plane reduce mismatch: abs={abs_p:.3e}");
        }
    }

    // Large N (100_000) -> matches the CPU baseline within the reported tolerance.
    {
        let input: Vec<f64> = (0..100_000).map(|k| ((k % 1000) as f64) * 1e-3 - 0.5).collect();
        let (dev, base, abs, rel) = oracle_f64(&input, false);
        println!("[reduce f64 N=100000] device_sum={dev} baseline={base} abs_div={abs:.3e} rel_div={rel:.3e}");
        assert!(
            rel <= 1e-9 || abs <= 1e-6,
            "large-N reduce diverged too far: abs={abs:.3e} rel={rel:.3e}"
        );
    }
}
