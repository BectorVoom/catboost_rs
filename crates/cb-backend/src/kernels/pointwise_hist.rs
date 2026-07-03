//! Self-oracle for the device-resident 2-channel pointwise histogram fill
//! (GPU-01 histogram slice, Phase 7.3): the GPU `pointwise_hist2` 8-bit non-binary
//! fill computed over [`crate::SelectedRuntime`] must match an ORDERED host-reference
//! 2-channel histogram (Σ der1, Σ weight per (feature, bin)) within a REPORTED (not
//! signed-off) tolerance, over f32 and f64 fixtures including edge cases (empty,
//! n=1, non-cube-multiple, large N), and the device-residency hand-off must return
//! the `binSums` as a HANDLE with no host fold inserted on the seam (D-7.3-05 /
//! SC-3).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs`, the launch seam in `gpu_runtime.rs`; ALL assertions live here. Test
//! code may use `.unwrap()`/indexing (the `lib.rs:1` `#[cfg(test)]` allow) — the
//! production `gpu_runtime.rs` may not.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32), and builds/runs under every
//! backend (like `kernels::reduce`/`kernels::scan`/`kernels::gradient_gpu`). The
//! reported max abs/rel divergence is informational: the GPU-06 epsilon is signed
//! off in Phase 7.6, NOT hard-coded here (D-7.3-04). The asserted tolerances are
//! generous, run-stable bounds (f32 ~1e-3 relative, f64 ~1e-9 relative) that catch a
//! wrong histogram without pinning the final epsilon. The in-kernel atomic merge
//! (D-03) makes the cross-thread accumulation ORDER non-deterministic, so the f64
//! bound is intentionally not tighter than ~1e-9.
//!
//! # FROZEN `binSums` device-handle layout (D-7.3-01 / Pitfall 2)
//!
//! The host reference writes into the SAME flat buffer layout the device kernel
//! writes and the 7.5 score/split seam will consume — it MUST be frozen here so
//! Plans B/C/D and 7.5 reuse it unchanged:
//!
//! ```text
//! histLineSize = 2 * totalBinFeatures            (2 = the 2 interleaved channels)
//! index(part, fold, feature, bin, channel) =
//!     (GetHistogramOffset(part, fold) * histLineSize
//!      + (FirstFoldIndex(feature) + bin)) * 2 + channel
//! ```
//!
//! mirroring upstream `split_properties_helpers.cuh::ShiftPartAndBinSumsPtr` +
//! `pointwise_hist2_one_byte_templ.cuh:132-145` (`... * 2 + w`, w in {0,1}). For the
//! single-tree fill this phase delivers (`partCount = foldCount = 1`,
//! `GetHistogramOffset(0, 0) = 0`), with a single feature group whose
//! `FirstFoldIndex = 0` and `totalBinFeatures = n_features * (1 << bits)`, the index
//! collapses to:
//!
//! ```text
//! index(feature, bin, channel) = (feature * n_bins + bin) * 2 + channel
//! ```
//!
//! channel 0 = Σ der1 ("target"), channel 1 = Σ weight. The buffer length is
//! `histLineSize * 2 = n_features * n_bins * 2` floats.

use cubecl::prelude::*;

use cb_core::sum_f64;

use crate::gpu_runtime::cindex::{pack_cindex, read_bin_host};
use crate::gpu_runtime::{launch_pointwise_hist2, launch_pointwise_hist2_handle, AtomicFinalizePath};

/// The asserted run-stable divergence bound for the device histogram channel. The
/// device channel is f64 on rocm/cuda/cpu (HIP/CUDA support/emulate the f64 atomic
/// add) and f32 on wgpu (WGSL has no f64 atomics — RESEARCH A1), so the bound is the
/// f32 magnitude (~1e-3) under `wgpu` and the f64 magnitude (~1e-9) elsewhere. This is
/// a REPORTED run-stable bound, NOT the GPU-06 epsilon (7.6's job).
#[cfg(feature = "wgpu")]
const HIST_BOUND: f64 = 1e-3;
#[cfg(not(feature = "wgpu"))]
const HIST_BOUND: f64 = 1e-9;

/// Compare the device histogram (cast to f64) to the host reference element-wise,
/// returning the max abs and max rel divergence over the buffer. Cloned verbatim
/// from the `kernels::gradient_gpu` reporter (REPORT-not-sign-off, D-7.3-04).
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    // IN-03: make the length precondition explicit (every caller already asserts equal
    // lengths). Zipping the two slices removes the implicit coupling — a length mismatch
    // truncates to the shorter slice rather than panicking with an opaque OOB index, and
    // the debug_assert documents the contract loudly in debug builds.
    debug_assert_eq!(device.len(), baseline.len());
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

/// The ORDERED host-reference 2-channel histogram — the parity baseline the device
/// `binSums` is REPORTED against (D-7.3-04 / D-05). This GENERALIZES
/// `cb-compute::histogram::reduce_leaf_stats` (the `leaf -> bin` shape) from leaves to
/// `(feature, bin)` cells, WITHOUT modifying the frozen `cb-compute` baseline
/// (D-7.3-07): the host-reference lives HERE, in the `cb-backend` test file.
///
/// For each (feature, bin) cell it gathers the member objects in ascending OBJECT
/// order (`bin_of[i] = cindex[feature * n + indices[i]]`), then folds each gathered
/// `Vec` through `cb_core::sum_f64` (the single sanctioned ordered reduction — never
/// a naive iterator `.sum()`, D-05). The result is written into the FROZEN flat
/// `binSums` layout `index(feature, bin, channel) = (feature * n_bins + bin) * 2 +
/// channel`, channel 0 = Σ der1, channel 1 = Σ weight — so the host index == the
/// kernel write index, cell-for-cell.
///
/// `der1`/`weight` are length `n` (per object, in object order). `cindex` is the
/// quantized bin matrix laid out feature-major: `cindex[feature * n + obj]` is object
/// `obj`'s bin for `feature`. `indices` is the object visiting order (length `n`,
/// values in `0..n`).
fn host_reference_hist2(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> Vec<f64> {
    let n = der1.len();
    // GPUT-15: extract each bin through the SAME shift-mask the device fill now uses —
    // pack the plain feature-major cindex into the bit-packed grouped layout and read via
    // `read_bin_host` (the host replica of the `read_bin` accessor). The extracted bin
    // VALUE is identical to the old plain `cindex[feature * n + obj]` load (the packing
    // identity is proven bit-exact by the independent `kernels::cindex` oracle), so this
    // reference stays the ordered ground truth while agreeing cell-for-cell with the
    // packed device path.
    let n_buckets = vec![n_bins; n_features];
    let packed = pack_cindex(cindex, &n_buckets, n).unwrap();
    // Gather each (feature, bin) cell's per-object contributions in ascending object
    // order, then fold through the ordered primitive (the reduce_leaf_stats shape).
    let mut delta_members: Vec<Vec<f64>> = vec![Vec::new(); n_features * n_bins];
    let mut weight_members: Vec<Vec<f64>> = vec![Vec::new(); n_features * n_bins];

    for feature in 0..n_features {
        let tcf = packed.features[feature];
        // Visit objects in the `indices` order (ascending object-visiting order),
        // exactly as the device kernel walks them.
        for &obj in indices.iter() {
            let obj = obj as usize;
            let bin = read_bin_host(&packed.words, tcf.offset, obj, tcf.shift, tcf.mask) as usize;
            // WR-02: the reference indexes `delta_members[feature * n_bins + bin]` RAW,
            // so it REQUIRES every bin to be in `0..n_bins` (the host-side range guard in
            // `launch_pointwise_hist2_into` enforces the same invariant for the kernels).
            // Assert it here so the oracle cannot silently diverge from the masking
            // kernels on an out-of-range bin — instead of indexing out of bounds, it
            // surfaces a clear assertion in the test harness.
            assert!(
                bin < n_bins,
                "host_reference_hist2 requires in-range bins: bin {bin} >= n_bins {n_bins} \
                 (feature {feature}, obj {obj})"
            );
            let cell = feature * n_bins + bin;
            delta_members[cell].push(der1[obj]);
            weight_members[cell].push(weight[obj]);
        }
    }

    let mut out = vec![0.0_f64; n_features * n_bins * 2];
    for feature in 0..n_features {
        for bin in 0..n_bins {
            let cell = feature * n_bins + bin;
            let base = (feature * n_bins + bin) * 2;
            out[base] = sum_f64(&delta_members[cell]); // channel 0: Σ der1
            out[base + 1] = sum_f64(&weight_members[cell]); // channel 1: Σ weight
        }
    }
    out
}

/// Build a deterministic f64 fixture of `n` objects over `n_features` 8-bit features:
/// returns `(der1, weight, cindex, indices)` in the [`host_reference_hist2`] /
/// [`launch_pointwise_hist2`] layout. `der1` is the UNWEIGHTED first derivative (the
/// 7.2 seam contract); `weight` is a non-trivial per-object weight folded HERE as the
/// histogram's second channel (D-7.3-05). Bins span `0..n_bins` so the 8-bit range is
/// exercised without requiring n >= 256.
fn make_fixture_f64(n: usize, n_features: usize, n_bins: usize) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
    let der1: Vec<f64> = (0..n).map(|k| (k as f64) * 0.37 - 4.0).collect();
    // Non-trivial weights (never all-1) so the weight channel is a real sum.
    let weight: Vec<f64> = (0..n).map(|k| 0.5 + ((k % 7) as f64) * 0.25).collect();
    // Feature-major cindex: spread bins across the 8-bit range deterministically.
    let mut cindex = vec![0u32; n_features * n];
    for feature in 0..n_features {
        for obj in 0..n {
            let bin = ((obj * (feature + 1) + feature) % n_bins) as u32;
            cindex[feature * n + obj] = bin;
        }
    }
    let indices: Vec<u32> = (0..n as u32).collect();
    (der1, weight, cindex, indices)
}

/// Read a device `binSums` HANDLE back ONCE through a fresh client of the SAME
/// runtime (test-only — production never reads the hand-off handle, D-7.3-05). The
/// handle carries f64 values (the device channel is f64 when the f64-atomic gate
/// reports support, else the host-sum fallback already produced f64).
fn read_handle_f64(h: cubecl::server::Handle) -> Vec<f64> {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    // WR-03: re-resolving the client via `Runtime::client(&device)` for the SAME
    // runtime/device does NOT construct a foreign allocator — CubeCL caches one client
    // per device in a global pool, so this returns the SAME client (allocator/stream)
    // that the seam used to allocate `h`. The cross-client hazard the production seam
    // comment warns about (`launch_der_binary_into`) is reading a handle through a client
    // of a DIFFERENT device/runtime; that is not happening here. This is the established
    // read-back pattern shared by the reduce/scan/gradient_gpu oracles.
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let bytes = client.read_one(h).expect("hand-off handle read-back failed");
    // The channel is f32 on wgpu (RESEARCH A1) and f64 elsewhere — upcast to f64 so the
    // oracle compares against the f64 host reference uniformly.
    #[cfg(feature = "wgpu")]
    {
        bytemuck::cast_slice::<u8, f32>(&bytes).iter().map(|&v| f64::from(v)).collect()
    }
    #[cfg(not(feature = "wgpu"))]
    {
        bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
    }
}

#[test]
fn nonbinary_8bit() {
    // The 8-bit non-binary fill self-oracle: the device 2-channel histogram must
    // match the ordered host reference within the REPORTED bound over n=1, n=37
    // (non-cube-multiple), and large N, plus the empty short-circuit. The reported
    // max abs/rel divergence + the AtomicFinalizePath are printed (D-03 / Pitfall 1).
    let n_features = 2usize;
    let n_bins = 256usize; // 8-bit -> 1 << 8

    // Empty (n=0): NO launch, NO read-back (Pitfall 5). The readback wrapper returns
    // an empty Vec; the handle wrapper returns a zero-length handle (asserted in
    // `handoff`).
    {
        let (der1, weight, cindex, indices) = make_fixture_f64(0, n_features, n_bins);
        let (device, path) =
            launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
        println!("[hist2 8bit n=0] REPORTED AtomicFinalizePath={path:?}");
        assert!(device.is_empty(), "empty input must yield an empty histogram (no launch)");
    }

    // n=1, n=37 (non-cube-multiple), large N: device vs ordered host reference.
    for &n in &[1usize, 37usize, 10_000usize] {
        let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);
        let (device, path) =
            launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
        let baseline = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);

        assert_eq!(
            device.len(),
            baseline.len(),
            "device binSums length must equal the host-reference layout length"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[hist2 8bit f64 n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} AtomicFinalizePath={path:?}"
        );
        assert!(
            rel <= HIST_BOUND || abs <= HIST_BOUND,
            "8-bit hist2 (n={n}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
        );
    }
}

#[test]
fn nonbinary_8bit_f32() {
    // f32-magnitude fixture cast to f64 at the seam (the seam is f64-typed, matching
    // the cb-compute reduction order). A generous f32 relative bound (~1e-3) catches a
    // wrong histogram without pinning the GPU-06 epsilon.
    let n_features = 2usize;
    let n_bins = 256usize;
    let n = 64usize;

    let der1_f32: Vec<f32> = (0..n).map(|k| (k as f32) * 0.37 - 4.0).collect();
    let weight_f32: Vec<f32> = (0..n).map(|k| 0.5 + ((k % 7) as f32) * 0.25).collect();
    let der1: Vec<f64> = der1_f32.iter().map(|&v| f64::from(v)).collect();
    let weight: Vec<f64> = weight_f32.iter().map(|&v| f64::from(v)).collect();
    let mut cindex = vec![0u32; n_features * n];
    for feature in 0..n_features {
        for obj in 0..n {
            cindex[feature * n + obj] = ((obj * (feature + 1) + feature) % n_bins) as u32;
        }
    }
    let indices: Vec<u32> = (0..n as u32).collect();

    let (device, path) =
        launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
    let baseline = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);

    assert_eq!(device.len(), baseline.len(), "device binSums length must equal host-reference length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[hist2 8bit f32 n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} AtomicFinalizePath={path:?}");
    assert!(
        rel <= 1e-3 || abs <= 1e-3,
        "f32 8-bit hist2 diverged too far: abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn handoff() {
    // The SC-3 / D-7.3-05 device-residency assertion: der1(UNWEIGHTED)/weight/cindex/
    // indices handles in -> `launch_pointwise_hist2_handle` returns the `binSums` as a
    // device HANDLE with NO host fold inserted on the seam. The read-back happens ONCE
    // here (test-only), confirming the handle carries the correct histogram, and never
    // on the hand-off path itself.
    let n_features = 2usize;
    let n_bins = 256usize;
    let n = 50usize;

    let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);

    // Handle-out: NO read-back on the seam.
    let bin_sums_handle =
        launch_pointwise_hist2_handle(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();

    // Read the handle back ONCE (the test is the only place a read-back is allowed).
    let device = read_handle_f64(bin_sums_handle);
    let baseline = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);

    assert_eq!(device.len(), baseline.len(), "binSums handle length must equal host-reference layout length");
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[hist2 8bit handoff n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= HIST_BOUND || abs <= HIST_BOUND,
        "device-resident binSums handle diverged from the host reference: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
    );

    // Empty hand-off: the handle wrapper returns a zero-length handle with NO launch
    // and NO read-back (Pitfall 5) — assert it CONSTRUCTS without reading the empty
    // buffer back.
    let (der1, weight, cindex, indices) = make_fixture_f64(0, n_features, n_bins);
    assert!(
        launch_pointwise_hist2_handle(&der1, &weight, &cindex, &indices, n_bins, n_features).is_ok(),
        "empty hand-off must construct a zero-length handle (no read-back of an empty buffer)"
    );
}

#[test]
fn nonbinary_bits() {
    // The 5/6/7-bit non-binary fill self-oracle (Plan B): the SAME
    // `pointwise_hist2_nonbinary_kernel` selected through the comptime `bits` arg —
    // no new kernel family, no runtime bit-count branch (D-7.3-02). For each bit-width
    // the device 2-channel histogram must match the ORDERED host reference within the
    // REPORTED bound, exercised at the matching border count `(1 << bits)` over the
    // edge cases n=1 / n=37 (non-cube-multiple) / large N, plus the empty
    // short-circuit. An 8-bit pass is kept as a regression anchor so the slice
    // generalization cannot silently break the Plan A path. The per-bit max abs/rel
    // divergence + the AtomicFinalizePath are REPORTED, not signed off (D-7.3-04 —
    // the GPU-06 epsilon is 7.6's job).
    let n_features = 2usize;

    // {5,6,7} are the new Plan B cases; 8 is the Plan A regression anchor. The border
    // (bin) count is `1 << bits` per bit-width, mirroring upstream
    // `pointwise_kernels.cpp`'s `DISPATCH_ONE_BYTE(..., 5/6/7/8)` (a `b`-bit feature
    // group has up to `1 << b` borders).
    for &bits in &[5u32, 6u32, 7u32, 8u32] {
        let n_bins = 1usize << bits;

        // Empty (n=0): NO launch, NO read-back (Pitfall 5) at every bit-width.
        {
            let (der1, weight, cindex, indices) = make_fixture_f64(0, n_features, n_bins);
            let (device, path) =
                launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features)
                    .unwrap();
            println!("[hist2 {bits}bit n=0] REPORTED AtomicFinalizePath={path:?}");
            assert!(
                device.is_empty(),
                "empty input must yield an empty histogram (no launch) at bits={bits}"
            );
        }

        // n=1, n=37 (non-cube-multiple), large N: device vs ordered host reference,
        // each at the bit-width's `(1 << bits)` border count.
        for &n in &[1usize, 37usize, 10_000usize] {
            let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);
            let (device, path) =
                launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features)
                    .unwrap();
            let baseline =
                host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);

            assert_eq!(
                device.len(),
                baseline.len(),
                "device binSums length must equal the host-reference layout length (bits={bits}, n={n})"
            );
            let (abs, rel) = max_divergence(&device, &baseline);
            println!(
                "[hist2 {bits}bit f64 n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} AtomicFinalizePath={path:?}"
            );
            assert!(
                rel <= HIST_BOUND || abs <= HIST_BOUND,
                "{bits}-bit hist2 (n={n}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
            );
        }
    }
}

#[test]
fn half_byte() {
    // The half-byte (4-bit) fill self-oracle (Plan C — D-7.3-02/03/04): the SEPARATE
    // `pointwise_hist2_half_byte_kernel` family (NOT a comptime case of the non-binary
    // kernel) selected host-side from the half-byte border count `n_bins == 16`. Its
    // shared-memory layout / load-unroll structurally mirror upstream
    // `pointwise_hist2_half_byte_template.cuh` (the 16-bin TPointHistHalfByte working
    // histogram), but it writes the SAME FROZEN binSums layout
    // `(feature * 16 + bin) * 2 + channel` through the UNCHANGED der-handle-in ->
    // binSums-handle-out seam (the seam stays byte-identical — D-7.3-01 / Pitfall 2).
    //
    // For the 4-bit width the device 2-channel histogram must match the ORDERED host
    // reference within the REPORTED bound, exercised at the 16-bin border count over
    // the edge cases n=1 / n=37 (non-cube-multiple) / large N, plus the empty
    // short-circuit. The Plan A `host_reference_hist2` / `max_divergence` /
    // `make_fixture_f64` harness is reused verbatim (the host reference is generic over
    // `n_bins`, so 16 needs no new reference). The per-fixture max abs/rel divergence +
    // the AtomicFinalizePath are REPORTED, not signed off (D-7.3-04 — the GPU-06
    // epsilon is 7.6's job).
    let n_features = 2usize;
    let n_bins = 16usize; // 4-bit half-byte -> 1 << 4

    // Empty (n=0): NO launch, NO read-back (Pitfall 5) on the half-byte path too.
    {
        let (der1, weight, cindex, indices) = make_fixture_f64(0, n_features, n_bins);
        let (device, path) =
            launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
        println!("[hist2 half_byte n=0] REPORTED AtomicFinalizePath={path:?}");
        assert!(
            device.is_empty(),
            "empty input must yield an empty half-byte histogram (no launch)"
        );
    }

    // n=1, n=37 (non-cube-multiple), large N: device vs ordered host reference, at the
    // 16-bin half-byte border count, through the readback wrapper.
    for &n in &[1usize, 37usize, 10_000usize] {
        let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);
        let (device, path) =
            launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
        let baseline = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);

        assert_eq!(
            device.len(),
            baseline.len(),
            "device half-byte binSums length must equal the host-reference layout length (n={n})"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[hist2 half_byte f64 n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} AtomicFinalizePath={path:?}"
        );
        assert!(
            rel <= HIST_BOUND || abs <= HIST_BOUND,
            "half-byte hist2 (n={n}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
        );
    }

    // Device-residency hand-off on the half-byte family: der/weight/cindex/indices
    // handles in -> `launch_pointwise_hist2_handle` returns the half-byte `binSums` as a
    // device HANDLE with NO host fold on the seam (D-7.3-05). Read back ONCE here only.
    {
        let n = 50usize;
        let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);
        let bin_sums_handle =
            launch_pointwise_hist2_handle(&der1, &weight, &cindex, &indices, n_bins, n_features)
                .unwrap();
        let device = read_handle_f64(bin_sums_handle);
        let baseline = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);
        assert_eq!(
            device.len(),
            baseline.len(),
            "half-byte binSums handle length must equal the host-reference layout length"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!("[hist2 half_byte handoff n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
        assert!(
            rel <= HIST_BOUND || abs <= HIST_BOUND,
            "device-resident half-byte binSums handle diverged from the host reference: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
        );
    }
}

#[test]
fn binary() {
    // The binary (1-bit) fill self-oracle (Plan D — D-7.3-02/03/04): the SEPARATE
    // `pointwise_hist2_binary_kernel` family (NOT a comptime case of the non-binary or
    // half-byte kernels) selected host-side from the binary border count `n_bins == 2`.
    // It structurally mirrors upstream `pointwise_hist2_binary.cu`'s `ComputeHist2Binary`
    // (the binary `TPointHistHalfByte`-based decomposition collapsed to 2 bins, bit 0/1),
    // but it writes the SAME FROZEN binSums layout `(feature * 2 + bin) * 2 + channel`
    // through the UNCHANGED der-handle-in -> binSums-handle-out seam (the seam stays
    // byte-identical — D-7.3-01 / Pitfall 2), with `n_bins = 2`.
    //
    // For the 1-bit width the device 2-channel histogram must match the ORDERED host
    // reference within the REPORTED bound, exercised at the 2-bin border count over the
    // edge cases n=1 / n=37 (non-cube-multiple) / large N, plus the empty short-circuit.
    // The Plan A `host_reference_hist2` / `max_divergence` / `make_fixture_f64` harness
    // is reused verbatim (the host reference is generic over `n_bins`, so 2 needs no new
    // reference). The per-fixture max abs/rel divergence + the AtomicFinalizePath are
    // REPORTED, not signed off (D-7.3-04 — the GPU-06 epsilon is 7.6's job).
    let n_features = 2usize;
    let n_bins = 2usize; // 1-bit binary -> bit 0/1

    // Empty (n=0): NO launch, NO read-back (Pitfall 5) on the binary path too.
    {
        let (der1, weight, cindex, indices) = make_fixture_f64(0, n_features, n_bins);
        let (device, path) =
            launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
        println!("[hist2 binary n=0] REPORTED AtomicFinalizePath={path:?}");
        assert!(
            device.is_empty(),
            "empty input must yield an empty binary histogram (no launch)"
        );
    }

    // n=1, n=37 (non-cube-multiple), large N: device vs ordered host reference, at the
    // 2-bin binary border count, through the readback wrapper.
    for &n in &[1usize, 37usize, 10_000usize] {
        let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);
        let (device, path) =
            launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
        let baseline = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);

        assert_eq!(
            device.len(),
            baseline.len(),
            "device binary binSums length must equal the host-reference layout length (n={n})"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[hist2 binary f64 n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} AtomicFinalizePath={path:?}"
        );
        assert!(
            rel <= HIST_BOUND || abs <= HIST_BOUND,
            "binary hist2 (n={n}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
        );
    }

    // Device-residency hand-off on the binary family: der/weight/cindex/indices handles
    // in -> `launch_pointwise_hist2_handle` returns the binary `binSums` as a device
    // HANDLE with NO host fold on the seam (D-7.3-05). Read back ONCE here only.
    {
        let n = 50usize;
        let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);
        let bin_sums_handle =
            launch_pointwise_hist2_handle(&der1, &weight, &cindex, &indices, n_bins, n_features)
                .unwrap();
        let device = read_handle_f64(bin_sums_handle);
        let baseline = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);
        assert_eq!(
            device.len(),
            baseline.len(),
            "binary binSums handle length must equal the host-reference layout length"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!("[hist2 binary handoff n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
        assert!(
            rel <= HIST_BOUND || abs <= HIST_BOUND,
            "device-resident binary binSums handle diverged from the host reference: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
        );
    }
}

/// The whole-family rocm regression assertion (SC-1 close): runs ALL the pointwise
/// histogram variants — general 8-bit non-binary, 5/6/7-bit non-binary, half-byte
/// (4-bit), and binary (1-bit) — through the SINGLE frozen seam in one pass, proving the
/// full `pointwise_hist2` family fills the device-resident 2-channel histogram on rocm
/// and matches the ordered host reference within the REPORTED bound. This is the phase's
/// headline gate (D-7.3-01): with binary landed, the variant set is complete. It reuses
/// the Plan A `host_reference_hist2` / `max_divergence` / `make_fixture_f64` harness
/// verbatim; each variant is the host-dispatch border count (`n_bins`) the seam routes to
/// its kernel family (binary=2, half-byte=16, non-binary 5/6/7/8-bit=32/64/128/256).
#[test]
fn whole_family() {
    let n_features = 2usize;
    // Every covered border count -> the kernel family the frozen seam dispatches to:
    //   2   -> pointwise_hist2_binary_kernel      (1-bit, Plan D)
    //   16  -> pointwise_hist2_half_byte_kernel    (4-bit, Plan C)
    //   32  -> pointwise_hist2_nonbinary_kernel(5) (5-bit, Plan B)
    //   64  -> pointwise_hist2_nonbinary_kernel(6) (6-bit, Plan B)
    //   128 -> pointwise_hist2_nonbinary_kernel(7) (7-bit, Plan B)
    //   256 -> pointwise_hist2_nonbinary_kernel(8) (8-bit, Plan A)
    for &n_bins in &[2usize, 16, 32, 64, 128, 256] {
        // Empty short-circuit on every family (Pitfall 5).
        {
            let (der1, weight, cindex, indices) = make_fixture_f64(0, n_features, n_bins);
            let (device, _path) =
                launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features)
                    .unwrap();
            assert!(
                device.is_empty(),
                "empty input must yield an empty histogram for n_bins={n_bins} (no launch)"
            );
        }

        // n=1 / n=37 (non-cube-multiple) / large N: device vs ordered host reference.
        for &n in &[1usize, 37usize, 10_000usize] {
            let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);
            let (device, path) =
                launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features)
                    .unwrap();
            let baseline =
                host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);
            assert_eq!(
                device.len(),
                baseline.len(),
                "whole-family device binSums length must match the host layout (n_bins={n_bins} n={n})"
            );
            let (abs, rel) = max_divergence(&device, &baseline);
            println!(
                "[hist2 whole_family n_bins={n_bins} n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} AtomicFinalizePath={path:?}"
            );
            assert!(
                rel <= HIST_BOUND || abs <= HIST_BOUND,
                "whole-family hist2 (n_bins={n_bins} n={n}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={HIST_BOUND:.0e})"
            );
        }
    }
}

#[test]
fn length_mismatch_is_typed_error() {
    // T-07.3-01: mismatched der1/weight/cindex/indices lengths must surface a typed
    // `CbError::LengthMismatch` BEFORE any launch (a host-side guard), never an OOB
    // device read. (Test-only assertion that the production guard fires.)
    let n_features = 2usize;
    let n_bins = 256usize;
    let n = 16usize;
    let (der1, weight, cindex, indices) = make_fixture_f64(n, n_features, n_bins);

    // weight one element short of der1.
    let short_weight = &weight[..n - 1];
    let err = launch_pointwise_hist2(&der1, short_weight, &cindex, &indices, n_bins, n_features);
    assert!(
        matches!(err, Err(cb_core::CbError::LengthMismatch { .. })),
        "mismatched weight length must surface CbError::LengthMismatch, got {err:?}"
    );

    // The AtomicFinalizePath enum is part of the reported seam surface (suppress the
    // unused-import lint when the path constants are not otherwise named in a build).
    let _ = AtomicFinalizePath::HostSumFallback;
}

#[test]
fn out_of_range_value_is_typed_error() {
    // CR-01: the length guards bound only buffer POSITIONS; the VALUES inside `cindex`
    // (the bin) and `indices` (the object id) drive unchecked device array indices. A
    // malformed value must surface a typed `CbError::OutOfRange` BEFORE any launch (a
    // host-side guard), never an OOB device read/store. Mirrors
    // `length_mismatch_is_typed_error`. (Test-only assertion that the production guard
    // fires.)
    let n_features = 2usize;
    let n_bins = 256usize;
    let n = 16usize;

    // (a) Out-of-range `cindex` bin value: a single bin == n_bins (>= n_bins) would write
    // bin_sums out of bounds in the non-binary path (which does not mask).
    {
        let (der1, weight, mut cindex, indices) = make_fixture_f64(n, n_features, n_bins);
        cindex[0] = n_bins as u32; // exactly n_bins -> out of the valid 0..n_bins range
        let err = launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);
        assert!(
            matches!(err, Err(cb_core::CbError::OutOfRange(_))),
            "out-of-range cindex bin must surface CbError::OutOfRange, got {err:?}"
        );
    }

    // (b) Out-of-range `indices` object id: a single entry == n (>= n) would read
    // der1[obj]/weight[obj]/cindex[feature*n+obj] out of bounds on the device.
    {
        let (der1, weight, cindex, mut indices) = make_fixture_f64(n, n_features, n_bins);
        indices[0] = n as u32; // exactly n -> out of the valid 0..n range
        let err = launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);
        assert!(
            matches!(err, Err(cb_core::CbError::OutOfRange(_))),
            "out-of-range indices object id must surface CbError::OutOfRange, got {err:?}"
        );
    }
}
