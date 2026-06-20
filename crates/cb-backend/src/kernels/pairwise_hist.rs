//! Self-oracle for the device-resident **4-channel weight-only** pairwise histogram
//! fill (GPU-01 histogram slice, Phase 7.4) — the pairwise SIBLING of the 7.3
//! `kernels::pointwise_hist` oracle. The GPU `pairwise_hist` non-binary fill computed
//! over [`crate::SelectedRuntime`] (parameterized by a `#[comptime] bits` in {5,6,7})
//! must match an ORDERED host-reference 4-channel pairwise histogram within a REPORTED
//! (not signed-off) tolerance, over f32 and f64 fixtures including edge cases (empty,
//! n_pairs=1, non-cube-multiple n_pairs=37, large N), and the device-residency hand-off
//! must return the `binSums` as a HANDLE with no host fold inserted on the seam
//! (D-7.4-03 / SC-3).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs`, the launch seam in `gpu_runtime.rs`; ALL assertions live here. Test
//! code may use `.unwrap()`/indexing (the `lib.rs:1` `#[cfg(test)]` allow) — the
//! production `gpu_runtime.rs` may not.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32), and builds/runs under every backend
//! (like `kernels::pointwise_hist`/`kernels::gradient_gpu`). The reported max abs/rel
//! divergence is informational: the GPU-06 epsilon is signed off in Phase 7.6, NOT
//! hard-coded here (D-7.4-05). The asserted tolerances are generous, run-stable bounds
//! (f32 ~1e-3 relative, f64 ~1e-9 relative) that catch a wrong histogram without
//! pinning the final epsilon. The in-kernel atomic merge (D-03) makes the cross-thread
//! accumulation ORDER non-deterministic, so the f64 bound is intentionally not tighter
//! than ~1e-9.
//!
//! # FROZEN `binSums` device-handle layout — 4-channel WEIGHT-ONLY (D-7.4-03 / Pitfall 2)
//!
//! The host reference writes into the SAME flat buffer layout the device kernel writes
//! and the 7.5 pairwise score/split seam will consume — it MUST be frozen here so Plans
//! B/C/D/E and 7.5 reuse it unchanged. Unlike 7.3's 2-channel (Σ der1, Σ weight)
//! pointwise layout, the pairwise histogram is **4-channel weight-only** (`histId in
//! {0,1,2,3}`), mirroring upstream `pairwise_hist_one_byte_5bit.cuh:255-256` (the
//! `4 * (maxFoldCount * f + fold) + histId` merge) and `split_properties_helpers.cuh`'s
//! `Compare` predicate:
//!
//! ```text
//! PAIR_HIST_CHANNELS = 4
//! histLineSize = 4 * totalBinFeatures            (totalBinFeatures = n_features * n_bins)
//! index(feature, bin, histId) = (feature * n_bins + bin) * 4 + histId,  histId in {0,1,2,3}
//! buffer length = n_features * n_bins * 4
//! ```
//!
//! For the single-tree fill this phase delivers (`part = fold = 0`, single feature
//! group with `FirstFoldIndex = 0`), the multi-part `ShiftPartAndBinSumsPtr` offset and
//! `BuildBinaryFeatureHistograms` transform are 7.5 forward dependencies (documented,
//! not cut). **Anti-pattern: any `* 2` here silently breaks the 7.5 seam (Pitfall 2) —
//! the buffer is ALWAYS `* 4`.**
//!
//! # The per-pair `Compare -> histId` channel mapping (the genuinely-new logic, D-7.4-02)
//!
//! Distilled from upstream `pairwise_hist_one_byte_5bit.cuh::AddPair` (lines 68-119) +
//! the final merge (lines 240-256), with the warp-tile distribution (`flag =
//! threadIdx.x & 1`, the RotateRight, the 4-iteration unroll, the `tiled_partition<16>`
//! syncs) reduced to its ACCUMULATION SEMANTICS (A6 — the tile is perf, not semantics;
//! Pitfall 4). The 4 channels per `(feature, bin)` are
//! `histId = 2 * isGe + isSecondBin` where `weightLeq -> {0,1}` and `weightGe -> {2,3}`
//! (`isSecondBin` distinguishes the pair's first/second bin, merge lines 255-256). For a
//! pair `(b1, b2)` with weight `w`, summing the warp's flag=0 AND flag=1 contributions
//! collapses (non-one-hot `Compare(x,y,flag) = (x >= y) == flag`) to:
//!
//! ```text
//! let ge = (b1 >= b2);   let gt = (b1 > b2);          // b1==b2 -> ge=1, gt=0
//! bin b1, histId 2*ge + 0  += w;   bin b1, histId 2*gt + 0  += w;   // isSecondBin = 0
//! bin b2, histId 2*ge + 1  += w;   bin b2, histId 2*gt + 1  += w;   // isSecondBin = 1
//! ```
//!
//! (when `b1 > b2`: ge=gt=1, so bin b1 channel 2 gets 2w, bin b2 channel 3 gets 2w;
//! when `b1 < b2`: bin b1 channel 0 gets 2w, bin b2 channel 1 gets 2w; when `b1 == b2`:
//! bin b1 channels 0 and 2 each get w, bin b2 channels 1 and 3 each get w). The one-hot
//! overlay (Plan E) replaces the predicate with `bin1 == bin2`; here the `one_hot` arg
//! threads through but is exercised only by Plan E. The host reference and the kernel
//! use the IDENTICAL formula, so a wrong channel mapping diverges immediately.

use cubecl::prelude::*;

use cb_core::sum_f64;

use crate::gpu_runtime::{
    launch_pairwise_hist, launch_pairwise_hist_8bit, launch_pairwise_hist_8bit_handle,
    launch_pairwise_hist_binary, launch_pairwise_hist_binary_handle,
    launch_pairwise_hist_half_byte, launch_pairwise_hist_half_byte_handle,
    launch_pairwise_hist_handle,
};

/// The asserted run-stable divergence bound for the device histogram channel. The
/// device channel is f64 on rocm/cuda/cpu (HIP/CUDA support/emulate the f64 atomic add)
/// and f32 on wgpu (WGSL has no f64 atomics — RESEARCH A1), so the bound is the f32
/// magnitude (~1e-3) under `wgpu` and the f64 magnitude (~1e-9) elsewhere. This is a
/// REPORTED run-stable bound, NOT the GPU-06 epsilon (7.6's job).
#[cfg(feature = "wgpu")]
const PAIR_HIST_BOUND: f64 = 1e-3;
#[cfg(not(feature = "wgpu"))]
const PAIR_HIST_BOUND: f64 = 1e-9;

/// Compare the device histogram (cast to f64) to the host reference element-wise,
/// returning the max abs and max rel divergence over the buffer. Cloned verbatim from
/// `kernels::pointwise_hist::max_divergence` (REPORT-not-sign-off, D-7.4-05).
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    // Make the length precondition explicit (every caller already asserts equal
    // lengths). Zipping removes the implicit coupling — a length mismatch truncates to
    // the shorter slice rather than panicking with an opaque OOB index.
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

/// The per-pair, per-feature 4-channel contribution helper — the SINGLE source of the
/// `(b1, b2, one_hot) -> (bin, histId, w) writes` semantics shared by the host reference
/// here and (transcribed structurally) by the device kernel in `kernels.rs`. Pushes
/// `w` into the four selected `(bin, histId)` cells via `push_cell`, applying the
/// distilled `Compare -> histId` mapping documented in the module header.
///
/// Non-one-hot: `ge = (b1 >= b2)`, `gt = (b1 > b2)`. One-hot (Plan E only): the
/// `Compare` predicate becomes `bin1 == bin2`, i.e. `eq = (b1 == b2)` drives both the
/// "Leq" and "Ge" channel selection (upstream `split_properties_helpers.cuh:261`).
fn add_pair_contrib<P: FnMut(usize, usize, f64)>(b1: usize, b2: usize, w: f64, one_hot: bool, mut push_cell: P) {
    // `pred_first`/`pred_second` are the `Compare`-predicate truth for the two warp
    // passes (flag-collapsed). Non-one-hot: Compare(x,y,flag) = (x >= y) == flag, which
    // over the flag=0/flag=1 collapse yields the (ge, gt) pair of writes per bin. One-hot
    // (Plan E only): Compare = (bin1 == bin2), so both predicate evaluations are `eq`.
    // histId = 2 * isGe + isSecondBin, where isSecondBin distinguishes the pair's first
    // bin (b1, isSecondBin=0, histIds {0,2}) from its second bin (b2, isSecondBin=1,
    // histIds {1,3}), and isGe selects the "Ge" (8-offset) slot when the predicate is
    // false (merge lines 255-256 of pairwise_hist_one_byte_5bit.cuh).
    if one_hot {
        // One-hot Compare = (bin1 == bin2). Both flag-collapsed predicate evaluations
        // are `eq`, so the two per-bin writes coincide on the same slot (Plan E refines
        // the exact one-hot fold; here host-ref and kernel stay self-consistent).
        let is_ge = usize::from(b1 != b2); // predicate false -> Ge slot
        push_cell(b1, 2 * is_ge, w);
        push_cell(b1, 2 * is_ge, w);
        push_cell(b2, 2 * is_ge + 1, w);
        push_cell(b2, 2 * is_ge + 1, w);
    } else {
        let ge = usize::from(b1 >= b2);
        let gt = usize::from(b1 > b2);
        // bin b1 (isSecondBin=0): the two flag-collapsed writes land in histId 2*ge+0
        // and 2*gt+0 (equal bins -> ge=1,gt=0 -> channels 2 and 0 each get w).
        push_cell(b1, 2 * ge, w);
        push_cell(b1, 2 * gt, w);
        // bin b2 (isSecondBin=1): histId 2*ge+1 and 2*gt+1.
        push_cell(b2, 2 * ge + 1, w);
        push_cell(b2, 2 * gt + 1, w);
    }
}

/// The ORDERED host-reference 4-channel pairwise histogram — the parity baseline the
/// device `binSums` is REPORTED against (D-7.4-05 / D-05). GENERALIZES
/// `kernels::pointwise_hist::host_reference_hist2` from per-OBJECT / 2-channel to
/// per-PAIR / 4-channel, WITHOUT modifying the frozen `cb-compute` baseline (D-7.4-08):
/// the host reference lives HERE, in the `cb-backend` test file.
///
/// Iterates PAIRS in ascending pair order; per feature computes
/// `bin1 = cindex[feature * n_objects + oi]`, `bin2 = cindex[feature * n_objects + oj]`
/// (cindex stride over OBJECTS, not pairs — Pitfall 3), derives the four
/// `(bin, histId)` writes via [`add_pair_contrib`], and gathers each cell's per-pair
/// weights in ascending pair order. Each cell is then folded through `cb_core::sum_f64`
/// (the single sanctioned ordered reduction — never a naive iterator `.sum()`, D-05),
/// written into the FROZEN flat layout `(feature * n_bins + bin) * 4 + hist_id`.
///
/// `pair_i`/`pair_j`/`pair_weight` are length `n_pairs`. `cindex` is the quantized bin
/// matrix laid out feature-major: `cindex[feature * n_objects + obj]` is object `obj`'s
/// bin for `feature`.
fn host_reference_pairwise_hist(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    one_hot: bool,
) -> Vec<f64> {
    let n_pairs = pair_weight.len();
    // Gather each (feature, bin, histId) cell's per-pair contributions in ascending pair
    // order, then fold through the ordered primitive.
    let cells = n_features * n_bins * 4;
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); cells];

    for feature in 0..n_features {
        for p in 0..n_pairs {
            let oi = pair_i[p] as usize;
            let oj = pair_j[p] as usize;
            let w = pair_weight[p];
            let b1 = cindex[feature * n_objects + oi] as usize;
            let b2 = cindex[feature * n_objects + oj] as usize;
            // The reference indexes RAW, so it REQUIRES every bin to be in `0..n_bins`
            // (the host-side range guard in `launch_pairwise_hist_into` enforces the same
            // invariant for the kernel). Assert it here so the oracle cannot silently
            // diverge from the kernel on an out-of-range bin.
            assert!(
                b1 < n_bins && b2 < n_bins,
                "host_reference_pairwise_hist requires in-range bins: b1={b1} b2={b2} \
                 n_bins={n_bins} (feature {feature}, pair {p})"
            );
            add_pair_contrib(b1, b2, w, one_hot, |bin, hist_id, ww| {
                let cell = (feature * n_bins + bin) * 4 + hist_id;
                members[cell].push(ww);
            });
        }
    }

    let mut out = vec![0.0_f64; cells];
    for (cell, vals) in members.iter().enumerate() {
        out[cell] = sum_f64(vals);
    }
    out
}

/// Build a deterministic fixture of `n_pairs` object pairs over `n_objects` objects and
/// `n_features` features with `n_bins` bins: returns `(pair_i, pair_j, pair_weight,
/// cindex)` in the [`host_reference_pairwise_hist`] / [`launch_pairwise_hist`] layout.
/// `pair_weight` is the ONLY per-pair value (NO der1 — D-7.4-03). Bins span `0..n_bins`
/// feature-major over objects so the bit range is exercised without requiring
/// `n_objects >= n_bins`. Pair endpoints span `0..n_objects` and deliberately include
/// equal-bin pairs (so the `b1 == b2` channel path is covered).
fn make_pair_fixture(
    n_objects: usize,
    n_features: usize,
    n_bins: usize,
    n_pairs: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<u32>) {
    // Pair endpoints: deterministic spread across objects; oi != oj for n_objects > 1,
    // but bins MAY coincide (covering the b1==b2 split-channel case).
    let mut pair_i = vec![0u32; n_pairs];
    let mut pair_j = vec![0u32; n_pairs];
    let mut pair_weight = vec![0.0_f64; n_pairs];
    for p in 0..n_pairs {
        let oi = if n_objects == 0 { 0 } else { (p * 3 + 1) % n_objects };
        let oj = if n_objects == 0 { 0 } else { (p * 7 + 2) % n_objects };
        pair_i[p] = oi as u32;
        pair_j[p] = oj as u32;
        // Non-trivial per-pair weights (never all-1) so each channel is a real sum.
        pair_weight[p] = 0.5 + ((p % 11) as f64) * 0.25;
    }
    // Feature-major cindex: spread bins across the range deterministically.
    let mut cindex = vec![0u32; n_features * n_objects];
    for feature in 0..n_features {
        for obj in 0..n_objects {
            let bin = if n_bins == 0 { 0 } else { ((obj * (feature + 1) + feature) % n_bins) as u32 };
            cindex[feature * n_objects + obj] = bin;
        }
    }
    (pair_i, pair_j, pair_weight, cindex)
}

/// Read a device `binSums` HANDLE back ONCE through a fresh client of the SAME runtime
/// (test-only — production never reads the hand-off handle, D-7.4-03). Clone of
/// `kernels::pointwise_hist::read_handle_f64` with the f32->f64 upcast for the wgpu
/// channel.
fn read_pair_handle(h: cubecl::server::Handle) -> Vec<f64> {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    // Re-resolving the client via `Runtime::client(&device)` for the SAME runtime/device
    // returns the SAME cached client (allocator/stream) that the seam used to allocate
    // `h` — the established read-back pattern shared by the reduce/scan/pointwise_hist
    // oracles.
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let bytes = client.read_one(h).expect("hand-off handle read-back failed");
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
fn nonbinary_bits() {
    // The 5/6/7-bit non-binary pairwise fill self-oracle: the device 4-channel histogram
    // must match the ordered host reference within the REPORTED bound over the edge cases
    // n_pairs=0 (empty, NO launch/read-back), n_pairs=1, n_pairs=37 (non-cube-multiple),
    // and large N, at each bit-width's `1 << bits` border count. The reported max abs/rel
    // divergence is printed (REPORT-not-sign-off, D-7.4-05).
    let n_features = 2usize;
    let n_objects = 64usize;
    let one_hot = false;

    for &bits in &[5u32, 6u32, 7u32] {
        let n_bins = 1usize << bits;

        // Empty (n_pairs=0): NO launch, NO read-back (Pitfall 5).
        {
            let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, 0);
            let device =
                launch_pairwise_hist(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, bits, one_hot)
                    .unwrap();
            assert!(
                device.is_empty(),
                "empty input must yield an empty pairwise histogram (no launch) at bits={bits}"
            );
        }

        for &n_pairs in &[1usize, 37usize, 10_000usize] {
            let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
            let device =
                launch_pairwise_hist(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, bits, one_hot)
                    .unwrap();
            let baseline = host_reference_pairwise_hist(
                &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
            );

            assert_eq!(
                device.len(),
                baseline.len(),
                "device binSums length must equal the host-reference 4-channel layout (bits={bits}, n_pairs={n_pairs})"
            );
            let (abs, rel) = max_divergence(&device, &baseline);
            println!(
                "[pair_hist {bits}bit f64 n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}"
            );
            assert!(
                rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
                "{bits}-bit pairwise hist (n_pairs={n_pairs}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
            );
        }
    }
}

#[test]
fn eightbit_atomics() {
    // The 8-bit-atomics pairwise fill self-oracle (D-7.4-02 — the structurally DISTINCT
    // global-atomics family; upstream `pairwise_hist_one_byte_8bit_atomics.cuh`). At 8 bits
    // a 256-bin x 4-channel line does not fit the per-block shared-memory budget, so
    // upstream accumulates via TRUE GLOBAL ATOMICS; the MVP mirrors the non-binary kernel
    // body with `bits = 8` always using direct global `Atomic<F>::fetch_add` (upstream's
    // per-thread CachedBins cache is a documented perf follow-up over the SAME atomic
    // structure). It is a SEPARATE `#[cube]` symbol with a SEPARATE launch arm — exercised
    // here through `launch_pairwise_hist_8bit`.
    //
    // The device 4-channel histogram (n_bins = 256) must match the ordered host reference
    // within the REPORTED bound over the edge cases n_pairs=0 (empty, NO launch/read-back),
    // n_pairs=1, n_pairs=37 (non-cube-multiple), and large N. The reported max abs/rel
    // divergence is printed (REPORT-not-sign-off, D-7.4-05).
    let n_features = 2usize;
    let n_objects = 300usize; // > 256 so bins span the full 8-bit range
    let one_hot = false;
    let bits = 8u32;
    let n_bins = 1usize << bits; // 256 — the distinct 8-bit-atomics line size

    // Empty (n_pairs=0): NO launch, NO read-back (Pitfall 5).
    {
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, 0);
        let device =
            launch_pairwise_hist_8bit(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot)
                .unwrap();
        assert!(
            device.is_empty(),
            "empty input must yield an empty 8-bit-atomics pairwise histogram (no launch)"
        );
    }

    for &n_pairs in &[1usize, 37usize, 10_000usize] {
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        let device =
            launch_pairwise_hist_8bit(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot)
                .unwrap();
        let baseline = host_reference_pairwise_hist(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
        );

        assert_eq!(
            device.len(),
            baseline.len(),
            "device binSums length must equal the host-reference 4-channel layout (8-bit, n_pairs={n_pairs})"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[pair_hist 8bit-atomics f64 n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        assert!(
            rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
            "8-bit-atomics pairwise hist (n_pairs={n_pairs}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
        );
    }

    // Device-residency hand-off (SC-3): the 8-bit handle arm returns the 4-channel binSums
    // as a device HANDLE with NO host fold on the seam; read it back ONCE here.
    {
        let n_pairs = 50usize;
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        let handle = launch_pairwise_hist_8bit_handle(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
        )
        .unwrap();
        let device = read_pair_handle(handle);
        let baseline = host_reference_pairwise_hist(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
        );
        assert_eq!(
            device.len(),
            baseline.len(),
            "8-bit binSums handle length must equal the host-reference 4-channel layout length"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[pair_hist 8bit-atomics handoff n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        assert!(
            rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
            "8-bit-atomics device-resident binSums handle diverged from the host reference: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
        );
    }
}

#[test]
fn half_byte() {
    // The half-byte (4-bit, 16-bin) pairwise fill self-oracle (D-7.4-02 — the structurally
    // DISTINCT half-byte family; upstream `pairwise_hist_half_byte.cu`). The half-byte line
    // is a FIXED 16-bin (4-bit) histogram (the comptime `HALF_BYTE_BINS` precedent from the
    // shipped 7.3 half-byte kernel, NOT a runtime `bits` arg), and the family takes NO
    // one-hot overlay upstream (there is no `pairwise_hist_half_byte_one_hot.cu`). It is a
    // SEPARATE `#[cube]` symbol with a SEPARATE launch arm — exercised here through
    // `launch_pairwise_hist_half_byte`.
    //
    // The device 4-channel histogram (n_bins = 16) must match the ordered host reference
    // within the REPORTED bound over the edge cases n_pairs=0 (empty, NO launch/read-back),
    // n_pairs=1, n_pairs=37 (non-cube-multiple), and large N. The reported max abs/rel
    // divergence is printed (REPORT-not-sign-off, D-7.4-05).
    let n_features = 2usize;
    let n_objects = 64usize; // > 16 so bins span the full 4-bit range
    let n_bins = 16usize; // HALF_BYTE_BINS — the distinct half-byte line size
    // The half-byte family has no one-hot overlay; the host reference is the non-one-hot
    // Compare path (the kernel hard-codes it).
    let one_hot = false;

    // Empty (n_pairs=0): NO launch, NO read-back (Pitfall 5).
    {
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, 0);
        let device =
            launch_pairwise_hist_half_byte(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features)
                .unwrap();
        assert!(
            device.is_empty(),
            "empty input must yield an empty half-byte pairwise histogram (no launch)"
        );
    }

    for &n_pairs in &[1usize, 37usize, 10_000usize] {
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        let device =
            launch_pairwise_hist_half_byte(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features)
                .unwrap();
        let baseline = host_reference_pairwise_hist(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
        );

        assert_eq!(
            device.len(),
            baseline.len(),
            "device binSums length must equal the host-reference 4-channel layout (half-byte, n_pairs={n_pairs})"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[pair_hist half-byte f64 n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        assert!(
            rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
            "half-byte pairwise hist (n_pairs={n_pairs}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
        );
    }

    // Device-residency hand-off (SC-3): the half-byte handle arm returns the 4-channel
    // binSums as a device HANDLE with NO host fold on the seam; read it back ONCE here.
    {
        let n_pairs = 50usize;
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        let handle = launch_pairwise_hist_half_byte_handle(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features,
        )
        .unwrap();
        let device = read_pair_handle(handle);
        let baseline = host_reference_pairwise_hist(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
        );
        assert_eq!(
            device.len(),
            baseline.len(),
            "half-byte binSums handle length must equal the host-reference 4-channel layout length"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[pair_hist half-byte handoff n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        assert!(
            rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
            "half-byte device-resident binSums handle diverged from the host reference: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
        );
    }
}

#[test]
fn binary() {
    // The binary (1-bit, 2-bin) pairwise fill self-oracle (D-7.4-02 — the structurally
    // DISTINCT binary family; upstream `pairwise_hist_binary.cu`). The binary line is a
    // FIXED 2-bin (1-bit) histogram (a bin COUNT, NOT a warp literal), and the family takes
    // NO one-hot overlay upstream (there is no `pairwise_hist_binary_one_hot.cu`). It is a
    // SEPARATE `#[cube]` symbol with a SEPARATE launch arm — exercised here through
    // `launch_pairwise_hist_binary`. The upstream 2x2 `(invBin1&invBin2)|...` channel
    // decomposition reduces to the SAME non-one-hot `Compare(bin1,bin2)->histId` predicate
    // the other families use (validated bit-exact by this oracle).
    //
    // The device 4-channel histogram (n_bins = 2) must match the ordered host reference
    // within the REPORTED bound over the edge cases n_pairs=0 (empty, NO launch/read-back),
    // n_pairs=1, n_pairs=37 (non-cube-multiple), and large N. The reported max abs/rel
    // divergence is printed (REPORT-not-sign-off, D-7.4-05).
    let n_features = 2usize;
    let n_objects = 8usize; // bins are masked to {0,1}; n_objects only sizes the cindex stride
    let n_bins = 2usize; // the distinct binary line size (1 << 1)
    // The binary family has no one-hot overlay; the host reference is the non-one-hot
    // Compare path (the kernel hard-codes it).
    let one_hot = false;

    // Empty (n_pairs=0): NO launch, NO read-back (Pitfall 5).
    {
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, 0);
        let device =
            launch_pairwise_hist_binary(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features)
                .unwrap();
        assert!(
            device.is_empty(),
            "empty input must yield an empty binary pairwise histogram (no launch)"
        );
    }

    for &n_pairs in &[1usize, 37usize, 10_000usize] {
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        let device =
            launch_pairwise_hist_binary(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features)
                .unwrap();
        let baseline = host_reference_pairwise_hist(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
        );

        assert_eq!(
            device.len(),
            baseline.len(),
            "device binSums length must equal the host-reference 4-channel layout (binary, n_pairs={n_pairs})"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[pair_hist binary f64 n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        assert!(
            rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
            "binary pairwise hist (n_pairs={n_pairs}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
        );
    }

    // Device-residency hand-off (SC-3): the binary handle arm returns the 4-channel
    // binSums as a device HANDLE with NO host fold on the seam; read it back ONCE here.
    {
        let n_pairs = 50usize;
        let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        let handle = launch_pairwise_hist_binary_handle(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features,
        )
        .unwrap();
        let device = read_pair_handle(handle);
        let baseline = host_reference_pairwise_hist(
            &pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot,
        );
        assert_eq!(
            device.len(),
            baseline.len(),
            "binary binSums handle length must equal the host-reference 4-channel layout length"
        );
        let (abs, rel) = max_divergence(&device, &baseline);
        println!(
            "[pair_hist binary handoff n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}"
        );
        assert!(
            rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
            "binary device-resident binSums handle diverged from the host reference: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
        );
    }
}

#[test]
fn handoff() {
    // The SC-3 / D-7.4-03 device-residency assertion: pair_i/pair_j/pair_weight/cindex
    // handles in -> `launch_pairwise_hist_handle` returns the 4-channel `binSums` as a
    // device HANDLE with NO host fold on the seam. The read-back happens ONCE here
    // (test-only), confirming the handle carries the correct histogram, and never on the
    // hand-off path itself.
    let n_features = 2usize;
    let n_objects = 64usize;
    let n_bins = 32usize; // 5-bit
    let n_pairs = 50usize;
    let bits = 5u32;
    let one_hot = false;

    let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);

    // Handle-out: NO read-back on the seam.
    let handle =
        launch_pairwise_hist_handle(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, bits, one_hot)
            .unwrap();

    // Read the handle back ONCE (the test is the only place a read-back is allowed).
    let device = read_pair_handle(handle);
    let baseline =
        host_reference_pairwise_hist(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, one_hot);

    assert_eq!(
        device.len(),
        baseline.len(),
        "binSums handle length must equal the host-reference 4-channel layout length"
    );
    let (abs, rel) = max_divergence(&device, &baseline);
    println!("[pair_hist handoff n_pairs={n_pairs}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}");
    assert!(
        rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
        "device-resident binSums handle diverged from the host reference: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
    );

    // Empty hand-off: the handle wrapper returns a zero-length handle with NO launch and
    // NO read-back (Pitfall 5) — assert it CONSTRUCTS without reading the empty buffer.
    let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, 0);
    assert!(
        launch_pairwise_hist_handle(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, bits, one_hot)
            .is_ok(),
        "empty hand-off must construct a zero-length handle (no read-back of an empty buffer)"
    );
}

#[test]
fn pairlogit_fixture() {
    // SC-2: a PairLogitPairwise-derived pair/weight fixture grounds the oracle in a
    // realistic ranking-loss pair list. The pairs + per-pair weights are derived from
    // `cb_compute::loss::pairlogit_pair_prob` (read-only — D-7.4-08): for each
    // winner/loser pair in a query group, the pair weight is the PairLogit probability
    // `p = exp(loser) / (exp(winner) + exp(loser))` scaled by the pair's base weight
    // (the `winnerDer += w * p` magnitude, `error_functions.h:859`). The device 4-channel
    // pairwise histogram over these pairs must match the ordered host reference (SC-2).
    use cb_compute::pairlogit_pair_prob;

    let n_features = 2usize;
    let n_objects = 24usize; // one query group of 24 documents
    let n_bins = 32usize; // 5-bit
    let bits = 5u32;
    let one_hot = false;

    // Synthetic group: document approxes (the ranking score) drive the PairLogit prob.
    let approx: Vec<f64> = (0..n_objects).map(|k| (k as f64) * 0.1 - 1.0).collect();

    // Winner -> loser pairs: each adjacent (winner=i, loser=j>i) competitor edge, the
    // PairLogitPairwise adjacency shape (winner preferred over loser). Per-pair weight =
    // base_weight * p (the pairwise weight the ranking der path produces).
    let mut pair_i: Vec<u32> = Vec::new();
    let mut pair_j: Vec<u32> = Vec::new();
    let mut pair_weight: Vec<f64> = Vec::new();
    for w in 0..n_objects {
        for l in (w + 1)..n_objects {
            let base_weight = 0.5 + ((w + l) % 5) as f64 * 0.3;
            let p = pairlogit_pair_prob(approx[w], approx[l]);
            pair_i.push(w as u32);
            pair_j.push(l as u32);
            pair_weight.push(base_weight * p);
        }
    }

    // Feature-major cindex over the group's documents.
    let mut cindex = vec![0u32; n_features * n_objects];
    for feature in 0..n_features {
        for obj in 0..n_objects {
            cindex[feature * n_objects + obj] = ((obj * (feature + 1) + feature) % n_bins) as u32;
        }
    }

    let device = launch_pairwise_hist(
        &pair_i, &pair_j, &pair_weight, &cindex, n_objects, n_bins, n_features, bits, one_hot,
    )
    .unwrap();
    let baseline = host_reference_pairwise_hist(
        &pair_i, &pair_j, &pair_weight, &cindex, n_objects, n_bins, n_features, one_hot,
    );

    assert_eq!(
        device.len(),
        baseline.len(),
        "PairLogit fixture device binSums length must equal the host-reference layout length"
    );
    let (abs, rel) = max_divergence(&device, &baseline);
    println!(
        "[pair_hist pairlogit_fixture n_pairs={}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e}",
        pair_weight.len()
    );
    assert!(
        rel <= PAIR_HIST_BOUND || abs <= PAIR_HIST_BOUND,
        "PairLogit pairwise hist diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={PAIR_HIST_BOUND:.0e})"
    );
}

#[test]
fn length_mismatch_is_typed_error() {
    // T-07.4-01: mismatched pair_i/pair_j/pair_weight/cindex lengths must surface a typed
    // `CbError::LengthMismatch` BEFORE any launch (a host-side guard), never an OOB device
    // read.
    let n_features = 2usize;
    let n_objects = 32usize;
    let n_bins = 32usize;
    let n_pairs = 16usize;
    let bits = 5u32;
    let (pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);

    // pair_j one element short of pair_i.
    let short_pj = &pj[..n_pairs - 1];
    let err = launch_pairwise_hist(&pi, short_pj, &pw, &cindex, n_objects, n_bins, n_features, bits, false);
    assert!(
        matches!(err, Err(cb_core::CbError::LengthMismatch { .. })),
        "mismatched pair_j length must surface CbError::LengthMismatch, got {err:?}"
    );
}

#[test]
fn out_of_range_value_is_typed_error() {
    // T-07.4-02: the length guards bound only buffer POSITIONS; the VALUES inside
    // `pair_i`/`pair_j` (object ids) and `cindex` (bins) drive unchecked device array
    // indices. A malformed value must surface a typed `CbError::OutOfRange` BEFORE any
    // launch, never an OOB device read.
    let n_features = 2usize;
    let n_objects = 32usize;
    let n_bins = 32usize;
    let n_pairs = 16usize;
    let bits = 5u32;

    // (a) Out-of-range pair_i object id (== n_objects).
    {
        let (mut pi, pj, pw, cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        pi[0] = n_objects as u32;
        let err = launch_pairwise_hist(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, bits, false);
        assert!(
            matches!(err, Err(cb_core::CbError::OutOfRange(_))),
            "out-of-range pair_i object id must surface CbError::OutOfRange, got {err:?}"
        );
    }

    // (b) Out-of-range cindex bin value (== n_bins).
    {
        let (pi, pj, pw, mut cindex) = make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        cindex[0] = n_bins as u32;
        let err = launch_pairwise_hist(&pi, &pj, &pw, &cindex, n_objects, n_bins, n_features, bits, false);
        assert!(
            matches!(err, Err(cb_core::CbError::OutOfRange(_))),
            "out-of-range cindex bin must surface CbError::OutOfRange, got {err:?}"
        );
    }
}
