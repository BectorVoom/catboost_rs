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

use cubecl::features::{AtomicUsage, Plane};
use cubecl::prelude::*;

use crate::kernels::{
    block_reduce_atomic_kernel, block_reduce_fixedpoint_kernel, block_reduce_kernel, full_scan_into,
    key_head_flag_kernel, reduce_by_key_kernel, segment_offset_scatter_kernel,
    segmented_reduce_kernel, REDUCE_FIXEDPOINT_SCALE_F64,
};

// IN-03: the "generous, run-stable" oracle bounds, hoisted into named consts shared
// across this module's assertions so the Phase-7.6 epsilon sign-off edits ONE place
// (and so the reduce and scan oracles cannot drift apart). These are NOT the final
// GPU-06 epsilon — they only catch a wrong fold without pinning the signed-off bound.

/// Generous relative bound for an f32 device sum vs the f64 CPU baseline.
const F32_REL_TOL: f64 = 1e-3;
/// Generous absolute bound for an f32 device sum vs the f64 CPU baseline.
const F32_ABS_TOL: f64 = 1e-3;
/// Generous relative bound for an f64 device sum vs the f64 CPU baseline.
const F64_REL_TOL: f64 = 1e-9;
/// Tight absolute bound for the small/exact f64 cases (n=1).
const F64_ABS_TOL_TIGHT: f64 = 1e-12;
/// Looser absolute bound for the large-N f64 accumulation case.
const F64_ABS_TOL_LARGE_N: f64 = 1e-6;

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

/// Launch `block_reduce_atomic_kernel::<F>` DIRECTLY on the selected runtime and
/// read back the single in-kernel `Atomic::fetch_add` accumulator (WR-01). This
/// bypasses the capability gate in
/// [`crate::gpu_runtime::launch_block_reduce_atomic_f64`] (which routes to the
/// host-sum fallback whenever the device does not ADVERTISE f64 atomic-add) so the
/// atomic kernel is actually executed on gfx1100 — where HIP runs f64 atomics even
/// though it does not advertise them. `use_plane` selects the intra-cube fold path
/// exactly as `run_reduce` does for the non-atomic kernel.
fn run_atomic_reduce<F>(input: &[F], use_plane: bool) -> F
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = input.len();
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let num_cubes = n.div_ceil(32usize).max(1);
    // Zero-initialized length-1 accumulator: the in-kernel `fetch_add`s accumulate
    // from the additive identity.
    let acc_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![F::new(0.0)]));

    block_reduce_atomic_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: 32u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), 1) },
        use_plane,
    );

    let bytes = client.read_one(acc_handle).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes)[0]
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
            rel <= F32_REL_TOL || abs <= F32_ABS_TOL,
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
            rel <= F64_REL_TOL || abs <= F64_REL_TOL,
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

        // IN-05: exercise the PRODUCTION empty short-circuit, not only the host
        // baseline. `launch_block_reduce_f64(&[])` must return `Ok(vec![])` without
        // touching the device.
        let out = crate::gpu_runtime::launch_block_reduce_f64(&empty).unwrap();
        assert!(out.is_empty(), "launch_block_reduce_f64(&[]) must return Ok(vec![])");
    }

    // n = 1 -> sum equals the single element.
    {
        let input = vec![42.5_f64];
        let (dev, base, abs, _rel) = oracle_f64(&input, false);
        println!("[reduce f64 n=1] device_sum={dev} baseline={base} abs_div={abs:.3e}");
        assert!(abs <= F64_ABS_TOL_TIGHT, "n=1 reduce mismatch: abs={abs:.3e}");
        if has_plane {
            let (_d, _b, abs_p, _r) = oracle_f64(&input, true);
            assert!(abs_p <= F64_ABS_TOL_TIGHT, "n=1 plane reduce mismatch: abs={abs_p:.3e}");
        }
    }

    // Large N (100_000) -> matches the CPU baseline within the reported tolerance.
    {
        let input: Vec<f64> = (0..100_000).map(|k| ((k % 1000) as f64) * 1e-3 - 0.5).collect();
        let (dev, base, abs, rel) = oracle_f64(&input, false);
        println!("[reduce f64 N=100000] device_sum={dev} baseline={base} abs_div={abs:.3e} rel_div={rel:.3e}");
        assert!(
            rel <= F64_REL_TOL || abs <= F64_ABS_TOL_LARGE_N,
            "large-N reduce diverged too far: abs={abs:.3e} rel={rel:.3e}"
        );
    }
}

/// Atomic-finalize reduce variant (D-03 / D-7.1-07): the in-kernel `Atomic::fetch_add`
/// cross-cube finalize must match `cb-core::sum_f64`, and — because the cross-cube
/// summation ORDER is non-deterministic — the oracle runs it MANY times to OBSERVE
/// and report the run-to-run variance. The reported variance / divergence is
/// informational ONLY: the GPU-06 epsilon is signed off in Phase 7.6, NOT here.
///
/// This exercises [`crate::gpu_runtime::launch_block_reduce_atomic_f64`], which
/// reports WHICH finalize ran (in-kernel f64 atomic vs the documented host-sum
/// fallback when the backend lacks f64 atomic-add — Pitfall 4). The chosen path is
/// printed so the SUMMARY can record it (NOT a silent omission).
#[test]
fn block_reduce_atomic_finalize_matches_cpu_sum_and_reports_variance() {
    use crate::gpu_runtime::{launch_block_reduce_atomic_f64, AtomicFinalizePath};

    // Multi-cube input (300 elements -> ~10 cubes at CUBE_DIM 32) so several cubes
    // race to fetch_add into the single accumulator — the setup that exposes any
    // cross-cube order non-determinism.
    let input: Vec<f64> = (0..300).map(|k| ((k % 23) as f64) - 11.0 + 0.125 * (k as f64)).collect();
    let baseline = cb_core::sum_f64(&input);

    // Run the atomic finalize repeatedly and collect the device sums.
    let runs = 32;
    let mut sums: Vec<f64> = Vec::with_capacity(runs);
    let mut path = AtomicFinalizePath::HostSumFallback;
    for _ in 0..runs {
        let (sum, p) = launch_block_reduce_atomic_f64(&input).unwrap();
        sums.push(sum);
        path = p;
    }

    // Observe the run-to-run spread (the D-03 non-determinism signal).
    let mut min_sum = sums[0];
    let mut max_sum = sums[0];
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for &s in &sums {
        min_sum = min_sum.min(s);
        max_sum = max_sum.max(s);
        let abs = (s - baseline).abs();
        let rel = if baseline.abs() > 0.0 {
            abs / baseline.abs()
        } else {
            abs
        };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    let variance_spread = max_sum - min_sum;

    println!(
        "[reduce atomic-finalize] path={path:?} runs={runs} baseline={baseline} \
         min={min_sum} max={max_sum} run_to_run_spread={variance_spread:.3e} \
         REPORTED max abs_div={max_abs:.3e} max rel_div={max_rel:.3e}"
    );
    println!(
        "[reduce atomic-finalize] NOTE: run-to-run spread is the accepted D-03 \
         in-kernel-atomic non-determinism (T-7.1-05); the GPU-06 epsilon is signed \
         off in Phase 7.6, NOT here."
    );

    // WR-02: the helper returns `(f64, AtomicFinalizePath)` and its `HostSumFallback`
    // branch silently substitutes a DETERMINISTIC host sum for the "atomic" entry
    // point. Assert that the returned path is consistent with what the device's f64
    // atomic-add capability ACTUALLY advertises, so a silent atomic-to-deterministic
    // mode switch on a device that is EXPECTED to support atomics surfaces as a
    // failure rather than passing silently. On gfx1100 HIP does not advertise f64
    // atomic-add, so `HostSumFallback` is the consistent (expected) path here.
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let advertises_atomic = {
        let ty = <cubecl::prelude::Atomic<f64> as CubePrimitive>::as_type_native_unchecked();
        client
            .properties()
            .atomic_type_usage(ty)
            .contains(cubecl::features::AtomicUsage::Add)
    };
    let expected_path = if advertises_atomic {
        AtomicFinalizePath::InKernelAtomicF64
    } else {
        AtomicFinalizePath::HostSumFallback
    };
    assert_eq!(
        path, expected_path,
        "launch_block_reduce_atomic_f64 returned {path:?} but the device advertises \
         f64 atomic-add = {advertises_atomic} (expected {expected_path:?}) — a silent \
         atomic-to-deterministic mode switch would otherwise pass unnoticed (WR-02)"
    );

    // The atomic finalize must still land on the CPU baseline within a generous,
    // run-stable bound that catches a wrong fold without pinning the GPU-06 epsilon.
    assert!(
        max_rel <= F64_REL_TOL || max_abs <= F64_REL_TOL,
        "atomic-finalize reduce diverged too far from baseline: abs={max_abs:.3e} rel={max_rel:.3e}"
    );
}

/// Drive `block_reduce_atomic_kernel` DIRECTLY (WR-01), bypassing the capability
/// gate that routes the production helper to the host-sum fallback on gfx1100
/// (where HIP does not ADVERTISE f64 atomic-add but DOES run it). This is the test
/// that actually EXERCISES the in-kernel `acc[0].fetch_add(cube_partial)` cross-cube
/// finalize — the headline deliverable of commit 4916c8e — on the in-env hardware,
/// regardless of the advertised capability. Without this, the gated oracle takes the
/// deterministic host-sum branch and the atomic kernel is never launched in-env.
#[test]
fn block_reduce_atomic_kernel_direct_matches_cpu_sum() {
    // Multi-cube input (300 elements -> ~10 cubes at CUBE_DIM 32) so several cubes
    // race to fetch_add into the single accumulator — the setup that drives the
    // cross-cube atomic finalize.
    let input: Vec<f64> = (0..300).map(|k| ((k % 23) as f64) - 11.0 + 0.125 * (k as f64)).collect();
    let baseline = cb_core::sum_f64(&input);

    let has_plane = device_has_plane();
    let mut paths: Vec<bool> = vec![false];
    if has_plane {
        paths.push(true);
    }

    // Run the atomic kernel repeatedly so the run-to-run spread (the D-03 cross-cube
    // order non-determinism) is observed on the path that ACTUALLY launches the
    // kernel, not the host-sum fallback.
    let runs = 32;
    for &use_plane in &paths {
        let mut min_sum = f64::INFINITY;
        let mut max_sum = f64::NEG_INFINITY;
        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for _ in 0..runs {
            let sum = run_atomic_reduce(&input, use_plane);
            min_sum = min_sum.min(sum);
            max_sum = max_sum.max(sum);
            let abs = (sum - baseline).abs();
            let rel = if baseline.abs() > 0.0 { abs / baseline.abs() } else { abs };
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
        let spread = max_sum - min_sum;
        println!(
            "[reduce atomic-kernel DIRECT] use_plane={use_plane} runs={runs} baseline={baseline} \
             min={min_sum} max={max_sum} run_to_run_spread={spread:.3e} \
             REPORTED max abs_div={max_abs:.3e} max rel_div={max_rel:.3e}"
        );
        assert!(
            max_rel <= F64_REL_TOL || max_abs <= F64_REL_TOL,
            "direct atomic-kernel reduce diverged too far (use_plane={use_plane}): abs={max_abs:.3e} rel={max_rel:.3e}"
        );
    }
}

// ===========================================================================
// Reduce family self-oracles (GPUT-16, Plan 10-03): segmented-reduce +
// reduce-by-key. The device output is checked against an INLINE serial CPU
// reference (D-02 — no cb-train reach, no upstream fixture) baselined via
// `cb_core::sum_f64` (the single sanctioned ordered reduction — never a naive
// `.sum()`). Both primitives fold in f64 with a FIXED-ORDER tree reduce, so they
// are deterministic; the asserted bounds are generous/run-stable (NOT the GPU-06
// epsilon, which is signed off on Kaggle CUDA via 10-09).
// ===========================================================================

/// Launch `segmented_reduce_kernel::<F>` (one cube per segment) and read back the
/// `num_segments` per-segment sums. `seg_offsets` has `num_segments + 1` entries.
fn run_segmented_reduce<F>(input: &[F], seg_offsets: &[u32]) -> Vec<F>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let num_segments = seg_offsets.len() - 1;
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let off_handle = client.create(cubecl::bytes::Bytes::from_elems(seg_offsets.to_vec()));
    let out_handle = client.empty(num_segments * std::mem::size_of::<F>());

    segmented_reduce_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(num_segments as u32, 1, 1),
        CubeDim { x: 32u32, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(in_handle, input.len()) },
        unsafe { ArrayArg::from_raw_parts(off_handle, seg_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), num_segments) },
    );

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

/// Inline serial per-segment sum reference (D-02): each segment folded through
/// `cb_core::sum_f64` in ascending order — the frozen host reduction.
fn cpu_segmented_reduce(input: &[f64], seg_offsets: &[u32]) -> Vec<f64> {
    (0..seg_offsets.len() - 1)
        .map(|s| {
            let a = seg_offsets[s] as usize;
            let b = seg_offsets[s + 1] as usize;
            cb_core::sum_f64(&input[a..b])
        })
        .collect()
}

/// Launch the on-device reduce-by-key pipeline: `key_head_flag_kernel` (phase 1) →
/// exclusive `full_scan_into` of the flags (phase 2, the 10-01 scan reused for
/// key-run detection) → `segment_offset_scatter_kernel` (phase 3, heads write their
/// start into `seg_offsets`) → `reduce_by_key_kernel` (one cube per run). Returns the
/// compacted `(keys, sums)` in key-run order. `num_segments` (a scalar count) is
/// derived host-side; the boundary POSITIONS and the sums are computed on-device.
fn run_reduce_by_key<F>(keys: &[u32], values: &[F]) -> (Vec<u32>, Vec<F>)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = keys.len();
    assert_eq!(n, values.len(), "keys/values length mismatch");

    // Distinct-key-run count (scalar; the positions are scattered on-device below).
    let mut num_segments = 0usize;
    for i in 0..n {
        if i == 0 || keys[i] != keys[i - 1] {
            num_segments += 1;
        }
    }

    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let dim32 = CubeDim { x: 32u32, y: 1, z: 1 };
    let n_cubes = n.div_ceil(32usize).max(1);

    let keys_h = client.create(cubecl::bytes::Bytes::from_elems(keys.to_vec()));
    let values_h = client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()));

    // Phase 1: key-head flags.
    let flags_h = client.empty(n * std::mem::size_of::<F>());
    key_head_flag_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(n_cubes as u32, 1, 1),
        dim32,
        unsafe { ArrayArg::from_raw_parts(keys_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(flags_h.clone(), n) },
    );

    // Phase 2: exclusive scan of the flags → per-element segment index (10-01 reuse,
    // device-resident — `flags_h` is cloned since `full_scan_into` consumes its input).
    let seg_ids_h = full_scan_into::<F>(&client, flags_h.clone(), n, false).unwrap();

    // Phase 3: scatter each head's position into `seg_offsets[seg_id]`. Init every slot
    // to `n` so the trailing slot `[num_segments]` is `n`; heads overwrite `[0..num_segments]`.
    let seg_offsets_h =
        client.create(cubecl::bytes::Bytes::from_elems(vec![n as u32; num_segments + 1]));
    segment_offset_scatter_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(n_cubes as u32, 1, 1),
        dim32,
        unsafe { ArrayArg::from_raw_parts(flags_h, n) },
        unsafe { ArrayArg::from_raw_parts(seg_ids_h, n) },
        unsafe { ArrayArg::from_raw_parts(seg_offsets_h.clone(), num_segments + 1) },
    );

    // Phase 4: per-run key + f64 sum (one cube per run).
    let out_keys_h = client.empty(num_segments * std::mem::size_of::<u32>());
    let out_sums_h = client.empty(num_segments * std::mem::size_of::<F>());
    reduce_by_key_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(num_segments as u32, 1, 1),
        dim32,
        unsafe { ArrayArg::from_raw_parts(keys_h, n) },
        unsafe { ArrayArg::from_raw_parts(values_h, n) },
        unsafe { ArrayArg::from_raw_parts(seg_offsets_h, num_segments + 1) },
        unsafe { ArrayArg::from_raw_parts(out_keys_h.clone(), num_segments) },
        unsafe { ArrayArg::from_raw_parts(out_sums_h.clone(), num_segments) },
    );

    let keys_bytes = client.read_one(out_keys_h).unwrap();
    let sums_bytes = client.read_one(out_sums_h).unwrap();
    (
        bytemuck::cast_slice::<u8, u32>(&keys_bytes).to_vec(),
        bytemuck::cast_slice::<u8, F>(&sums_bytes).to_vec(),
    )
}

/// Inline serial reduce-by-key reference (D-02): walk contiguous equal-key runs,
/// summing each run's values through `cb_core::sum_f64` in ascending order.
fn cpu_reduce_by_key(keys: &[u32], values: &[f64]) -> (Vec<u32>, Vec<f64>) {
    let mut out_keys = Vec::new();
    let mut out_sums = Vec::new();
    let mut i = 0usize;
    while i < keys.len() {
        let k = keys[i];
        let start = i;
        while i < keys.len() && keys[i] == k {
            i += 1;
        }
        out_keys.push(k);
        out_sums.push(cb_core::sum_f64(&values[start..i]));
    }
    (out_keys, out_sums)
}

#[test]
fn segmented_reduce_matches_serial() {
    // Behaviour example (plan): values [1,2,3,4] with segment offsets {0,2,4} → [3, 7].
    {
        let input = vec![1.0_f64, 2.0, 3.0, 4.0];
        let offsets = vec![0u32, 2, 4];
        let dev = run_segmented_reduce(&input, &offsets);
        assert_eq!(dev, vec![3.0_f64, 7.0], "behaviour example [3,7] mismatch");
    }

    // Larger, varied-size segments (several > CUBE_DIM=32 so the grid-stride intra-segment
    // fold is exercised) in both f32 and f64.
    let seg_sizes = [1usize, 5, 32, 33, 64, 100, 7, 50];
    let mut offsets: Vec<u32> = vec![0];
    let mut acc = 0u32;
    for &sz in &seg_sizes {
        acc += sz as u32;
        offsets.push(acc);
    }
    let n = acc as usize;
    let input_f64: Vec<f64> = (0..n).map(|k| ((k % 19) as f64) - 9.0 + 0.125 * (k as f64)).collect();
    let input_f32: Vec<f32> = input_f64.iter().map(|&v| v as f32).collect();
    let expected = cpu_segmented_reduce(&input_f64, &offsets);

    let dev_f64 = run_segmented_reduce(&input_f64, &offsets);
    for (s, (&d, &e)) in dev_f64.iter().zip(expected.iter()).enumerate() {
        let abs = (d - e).abs();
        let rel = if e.abs() > 0.0 { abs / e.abs() } else { abs };
        assert!(
            rel <= F64_REL_TOL || abs <= F64_ABS_TOL_LARGE_N,
            "segmented-reduce f64 seg {s} diverged: dev={d} exp={e} abs={abs:.3e}"
        );
    }

    let dev_f32 = run_segmented_reduce(&input_f32, &offsets);
    for (s, (&d, &e)) in dev_f32.iter().zip(expected.iter()).enumerate() {
        let abs = (f64::from(d) - e).abs();
        let rel = if e.abs() > 0.0 { abs / e.abs() } else { abs };
        assert!(
            rel <= F32_REL_TOL || abs <= F32_ABS_TOL,
            "segmented-reduce f32 seg {s} diverged: dev={d} exp={e} abs={abs:.3e}"
        );
    }
    println!("[segmented-reduce] {} segments, n={n} — f32+f64 match serial", seg_sizes.len());
}

#[test]
fn reduce_by_key_matches_serial() {
    // Behaviour example (plan): keys [a,a,b,b,b] values [1,1,1,1,1] → keys [a,b] sums [2,3].
    {
        let keys = vec![7u32, 7, 3, 3, 3];
        let values = vec![1.0_f64; 5];
        let (dk, ds) = run_reduce_by_key(&keys, &values);
        assert_eq!(dk, vec![7u32, 3], "behaviour example keys [a,b] mismatch");
        assert_eq!(ds, vec![2.0_f64, 3.0], "behaviour example sums [2,3] mismatch");
    }

    // Larger multi-run case with runs longer than CUBE_DIM and mixed magnitudes (f64+f32).
    let run_spec: [(u32, usize); 6] = [(10, 40), (20, 1), (30, 65), (10, 5), (40, 33), (55, 12)];
    let mut keys: Vec<u32> = Vec::new();
    let mut values_f64: Vec<f64> = Vec::new();
    let mut ctr = 0usize;
    for &(k, len) in &run_spec {
        for _ in 0..len {
            keys.push(k);
            values_f64.push(((ctr % 13) as f64) - 6.0 + 0.25 * (ctr as f64));
            ctr += 1;
        }
    }
    let values_f32: Vec<f32> = values_f64.iter().map(|&v| v as f32).collect();
    let (exp_keys, exp_sums) = cpu_reduce_by_key(&keys, &values_f64);

    let (dk64, ds64) = run_reduce_by_key(&keys, &values_f64);
    assert_eq!(dk64, exp_keys, "reduce-by-key f64 keys mismatch");
    for (s, (&d, &e)) in ds64.iter().zip(exp_sums.iter()).enumerate() {
        let abs = (d - e).abs();
        let rel = if e.abs() > 0.0 { abs / e.abs() } else { abs };
        assert!(
            rel <= F64_REL_TOL || abs <= F64_ABS_TOL_LARGE_N,
            "reduce-by-key f64 run {s} diverged: dev={d} exp={e} abs={abs:.3e}"
        );
    }

    let (dk32, ds32) = run_reduce_by_key(&keys, &values_f32);
    assert_eq!(dk32, exp_keys, "reduce-by-key f32 keys mismatch");
    for (s, (&d, &e)) in ds32.iter().zip(exp_sums.iter()).enumerate() {
        let abs = (f64::from(d) - e).abs();
        let rel = if e.abs() > 0.0 { abs / e.abs() } else { abs };
        assert!(
            rel <= F32_REL_TOL || abs <= F32_ABS_TOL,
            "reduce-by-key f32 run {s} diverged: dev={d} exp={e} abs={abs:.3e}"
        );
    }
    println!("[reduce-by-key] {} runs — f32+f64 keys+sums match serial", run_spec.len());
}

// ===========================================================================
// Deterministic finalize strategies + run-to-run variance harness (Plan 10-03
// Task 2, D-03/D-04). The scalar cross-cube reduce finalize is delivered as 2-3
// SELECTABLE deterministic strategies, each of which must show ZERO run-to-run
// spread (byte-identical over 32 launches) and REPORT which strategy actually ran
// (a capability downgrade is reported, never a silent switch — T-10-07). The
// measured winner ships as the library reduce (D-04, no throwaway) and feeds
// Phase 11's ε=1e-4 histogram gate.
//
//   (a) FixedOrderTree    — recursive `block_reduce_kernel` (shared-mem tree, no
//                           plane, no atomics) → a fixed pairing on EVERY backend.
//   (b) HostSum           — per-cube partials + host `cb_core::sum_f64` (the
//                           in-tree Phase-7.6 `HostSumFallback` precedent).
//   (c) FixedPointAtomic  — `round(v*2^30) → i64 → u64` integer atomics
//                           (`block_reduce_fixedpoint_kernel`); exact + order-
//                           independent. Capability-gated on `Atomic<u64>` add;
//                           where unadvertised it reports the (b) downgrade.
// ===========================================================================

/// Which deterministic finalize strategy actually ran (reported so a capability
/// downgrade surfaces explicitly — never a silent atomic→deterministic switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceFinalizeStrategy {
    /// Recursive fixed-order shared-mem tree reduce (deterministic on any backend).
    FixedOrderTree,
    /// Per-cube partials + host `cb_core::sum_f64` (the in-tree `HostSumFallback`).
    HostSum,
    /// Fixed-point `Atomic<u64>` integer-atomic finalize (deterministic where the
    /// device advertises `Atomic<u64>` add).
    FixedPointAtomic,
}

/// Does the selected runtime's device advertise `Atomic<u64>` add? Gates strategy (c).
fn device_supports_u64_atomic_add() -> bool {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let ty = <cubecl::prelude::Atomic<u64> as CubePrimitive>::as_type_native_unchecked();
    client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::Add)
}

/// One recursion level of the fixed-order tree reduce: launch `block_reduce_kernel`
/// with `use_plane = false` (the FIXED-ORDER shared-mem tree — no plane, no atomics),
/// reducing `n` elements to `ceil(n/32)` per-cube partials, recursing until one scalar
/// remains. Deterministic on every backend.
fn tree_reduce_into(
    client: &cubecl::client::ComputeClient<crate::SelectedRuntime>,
    in_handle: cubecl::server::Handle,
    n: usize,
) -> cubecl::server::Handle {
    let num_cubes = n.div_ceil(32usize).max(1);
    let out = client.empty(num_cubes * std::mem::size_of::<f64>());
    block_reduce_kernel::launch::<f64, crate::SelectedRuntime>(
        client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim { x: 32u32, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), num_cubes) },
        false,
    );
    if num_cubes <= 1 {
        return out;
    }
    tree_reduce_into(client, out, num_cubes)
}

/// Strategy (a): fully-deterministic on-device recursive fixed-order tree reduce.
fn run_fixed_order_tree_reduce(input: &[f64]) -> f64 {
    if input.is_empty() {
        return 0.0;
    }
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let out = tree_reduce_into(&client, in_handle, input.len());
    let bytes = client.read_one(out).unwrap();
    bytemuck::cast_slice::<u8, f64>(&bytes)[0]
}

/// Strategy (b): per-cube partials (`launch_block_reduce_f64`) + host `cb_core::sum_f64`.
fn run_host_sum_finalize(input: &[f64]) -> f64 {
    let partials = crate::gpu_runtime::launch_block_reduce_f64(input).unwrap();
    cb_core::sum_f64(&partials)
}

/// Strategy (c): fixed-point `Atomic<u64>` finalize. Runs the integer-atomic kernel when
/// the device advertises `Atomic<u64>` add; otherwise reports the deterministic host-sum
/// DOWNGRADE (never a silent switch). Returns `(sum, actual_strategy)`.
fn run_fixedpoint_reduce(input: &[f64]) -> (f64, ReduceFinalizeStrategy) {
    if input.is_empty() {
        return (0.0, ReduceFinalizeStrategy::HostSum);
    }
    if !device_supports_u64_atomic_add() {
        // Documented capability downgrade (gfx1100 case): the deterministic host sum
        // stands in and the returned strategy says so — the harness asserts this.
        return (run_host_sum_finalize(input), ReduceFinalizeStrategy::HostSum);
    }

    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let n = input.len();
    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    // Zero-initialized single u64 fixed-point accumulator.
    let acc_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u64]));
    let num_cubes = n.div_ceil(32usize).max(1);

    block_reduce_fixedpoint_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim { x: 32u32, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), 1) },
    );

    let bytes = client.read_one(acc_handle).unwrap();
    let bits = bytemuck::cast_slice::<u8, u64>(&bytes)[0];
    // Reinterpret the u64 two's-complement bits as i64 and scale back (manual §3).
    let sum = (bits as i64) as f64 / REDUCE_FIXEDPOINT_SCALE_F64;
    (sum, ReduceFinalizeStrategy::FixedPointAtomic)
}

/// Run `f` `runs` times, assert BYTE-IDENTICAL results (zero run-to-run spread),
/// assert the reported strategy equals `expected` (no silent switch), and assert the
/// result lands on `baseline` within a generous run-stable bound.
fn assert_zero_spread<Fp>(
    name: &str,
    runs: usize,
    baseline: f64,
    mut f: Fp,
    expected: ReduceFinalizeStrategy,
) where
    Fp: FnMut() -> (f64, ReduceFinalizeStrategy),
{
    let mut first_bits: Option<u64> = None;
    let mut reported = expected;
    for _ in 0..runs {
        let (sum, strat) = f();
        reported = strat;
        assert_eq!(
            strat, expected,
            "[{name}] reported strategy {strat:?} != expected {expected:?} — a silent \
             capability switch must FAIL, not pass (T-10-07)"
        );
        let bits = sum.to_bits();
        match first_bits {
            None => first_bits = Some(bits),
            Some(b) => assert_eq!(
                bits, b,
                "[{name}] run-to-run spread detected ({} vs {}) — a deterministic \
                 strategy must be byte-identical across launches (T-10-06)",
                f64::from_bits(b),
                sum
            ),
        }
    }
    let sum = f64::from_bits(first_bits.unwrap());
    let abs = (sum - baseline).abs();
    let rel = if baseline.abs() > 0.0 { abs / baseline.abs() } else { abs };
    assert!(
        rel <= F64_REL_TOL || abs <= F64_ABS_TOL_LARGE_N,
        "[{name}] diverged from baseline: sum={sum} baseline={baseline} abs={abs:.3e}"
    );
    println!(
        "[reduce finalize: {name}] path={reported:?} runs={runs} sum={sum} baseline={baseline} \
         run_to_run_spread=0 abs_div={abs:.3e}"
    );
}

#[test]
fn reduce_finalize_strategies_are_deterministic_and_report_path() {
    // Multi-cube input (300 elements → ~10 cubes at CUBE_DIM 32) so the cross-cube
    // finalize (where any nondeterminism would live) is exercised by every strategy.
    let input: Vec<f64> = (0..300).map(|k| ((k % 23) as f64) - 11.0 + 0.125 * (k as f64)).collect();
    let baseline = cb_core::sum_f64(&input);
    let runs = 32;

    let u64_atomic = device_supports_u64_atomic_add();

    // (a) fixed-order tree reduce — deterministic on every backend.
    assert_zero_spread(
        "fixed-order-tree",
        runs,
        baseline,
        || (run_fixed_order_tree_reduce(&input), ReduceFinalizeStrategy::FixedOrderTree),
        ReduceFinalizeStrategy::FixedOrderTree,
    );

    // (b) block-then-host-final-sum — the in-tree HostSumFallback (Phase 7.6 precedent).
    assert_zero_spread(
        "host-sum",
        runs,
        baseline,
        || (run_host_sum_finalize(&input), ReduceFinalizeStrategy::HostSum),
        ReduceFinalizeStrategy::HostSum,
    );

    // (c) fixed-point u64 atomics — capability-gated. On gfx1100 (no advertised u64
    // atomic-add) this reports the deterministic host-sum downgrade; on CUDA it exercises
    // the integer-atomic path. Either way: zero run-to-run spread + the REPORTED path
    // matches the device capability (a silent switch fails the assertion above).
    let expected_c = if u64_atomic {
        ReduceFinalizeStrategy::FixedPointAtomic
    } else {
        ReduceFinalizeStrategy::HostSum
    };
    assert_zero_spread("fixed-point-atomic", runs, baseline, || run_fixedpoint_reduce(&input), expected_c);

    println!(
        "[reduce finalize strategies] u64_atomic_advertised={u64_atomic} — all 3 strategies \
         show ZERO run-to-run spread; fixed-point path reported as {expected_c:?}. NOTE: \
         gfx1100 ADVERTISES Atomic<u64> add (unlike Atomic<f64> add), so the fixed-point \
         path runs the integer-atomic kernel in-env; CUDA err+ms numbers are filled on \
         Kaggle via 10-09."
    );
}
