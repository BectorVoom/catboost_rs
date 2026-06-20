//! Cross-oracle for the device-resident **host-light single-tree grow loop** (GPU-01
//! grow-loop slice, Phase 7.5 Plan C; D-7.5-02 / D-05): the GPU `grow_oblivious_tree`
//! driver over [`crate::SelectedRuntime`] grows one complete oblivious tree
//! device-resident — per level chaining the FROZEN 7.3 histogram fill → the Plan-B
//! scan/update → the Plan-A score + deterministic argmin → ONE O(1) ~16-byte
//! [`crate::gpu_runtime::BestSplit`] read-back → the Plan-C `partition_split` (forward-bit
//! doc-routing) → the Plan-C `partition_update` (per-partition Σ der1 / Σ weight reduce),
//! over persistent device buffers threaded through ONE `ComputeClient`, then reads back
//! ONLY the `2^depth` part-stats at the leaves and computes leaf values via the FROZEN
//! `cb_compute::calc_average` formula. Histograms / partitions / doc-routing stay
//! device-resident across launches (reading the full histogram/partition buffer to score
//! or partition on host is the FORBIDDEN D-05 hybrid).
//!
//! # The STRICT structure bar (SC-3 / D-7.5-06) vs the REPORTED leaf-value tolerance
//!
//! The GPU tree's STRUCTURE — the per-level split `(feature, bin)` sequence AND the
//! per-object leaf assignment (`leaf_of`) — must match the CPU reference EXACTLY on the
//! clear-gain-margin fixture (the strict bar). Leaf VALUES (computed via
//! `cb_compute::calc_average` over the read-back part-stats) are REPORTED within a
//! generous run-stable bound (f32 ~1e-3 wgpu / f64 ~1e-9 elsewhere) — informational, NOT
//! the GPU-06 epsilon (7.6's job). A structure mismatch on a near-tie boundary is the
//! tolerance boundary to REPORT, never signed off here.
//!
//! # D-7.5-04 boundary — transcribe, do NOT import `cb-train`
//!
//! `cb_compute` (a normal dep) is imported READ-ONLY for the leaf-value oracle
//! (`calc_average` / `scale_l2_reg`) and the score oracle (`l2_split_score` / `LeafStats`,
//! reused from the sibling `score_split` harness shape). The TREE-STRUCTURE oracle —
//! `cb_train::greedy_tensor_search_oblivious`'s strict-first-wins greedy search + the
//! forward-bit `cb_train::leaf_index` — is TRANSCRIBED VERBATIM here rather than imported
//! (the Plan-A landmine: importing `cb-train` pulls its `cb-backend = {path}` default-`cpu`
//! dependency into the test build graph, cargo feature unification then activates
//! `cb-backend/cpu` ALONGSIDE `rocm`/`wgpu`/`cuda`, `SelectedRuntime` resolves to the
//! CpuRuntime which lacks `Atomic<f64>`/`Atomic<f32>`, and the histogram fill cannot run
//! at all — see the 07.5-01 SUMMARY). The transcription cross-oracles against the EXACT
//! documented CPU semantics (`tree.rs:272-302`, `:486-580`).
//!
//! This runs on `rocm` in-env on gfx1100 (wave32) and builds/runs under every backend
//! over [`crate::SelectedRuntime`], like `kernels::score_split`/`pointwise_hist`.

use cubecl::prelude::*;

use cb_core::sum_f64;

use crate::gpu_runtime::{
    launch_partition_split_into, launch_partition_update_into, read_part_stats_f64, read_u32_handle,
    upload_channel_floats,
};

/// The asserted run-stable leaf-VALUE divergence bound (REPORTED, not the GPU-06 epsilon —
/// 7.6's job): f32 magnitude (~1e-3) on wgpu (no f64 channel), f64 magnitude (~1e-9)
/// elsewhere — the SAME channel-driven split as `score_split::SCORE_BOUND`.
#[cfg(feature = "wgpu")]
const LEAF_BOUND: f64 = 1e-3;
#[cfg(not(feature = "wgpu"))]
const LEAF_BOUND: f64 = 1e-9;

/// Max abs / rel divergence over two equal-length buffers (the `score_split`/`pointwise_hist`
/// reporter shape — REPORT-not-sign-off, D-7.5-05).
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
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

/// The FORWARD-bit leaf index — TRANSCRIBED VERBATIM from the FROZEN
/// `cb_train::leaf_index` (`tree.rs:272-280`): split `i` → bit `i`, `idx |= 1usize << i`.
/// `passes[i]` is whether the object passes split `i`. This is the parity-critical
/// convention `partition_split_kernel` must replicate (Pitfall 6).
fn cpu_leaf_index(passes: &[bool]) -> usize {
    let mut idx = 0usize;
    for (i, &p) in passes.iter().enumerate() {
        if p {
            idx |= 1usize << i;
        }
    }
    idx
}

/// Build a deterministic fixture for the grow-loop / partition cross-oracle: `n_features`
/// quantized features over `n_bins` bins each, with a CLEAR per-feature gain margin so
/// the greedy structure is unambiguous (Pitfall 2). Returns the FROZEN 7.3 inputs
/// `(der1, weight, cindex, indices)` in the `launch_pointwise_hist2_handle` layout
/// (`cindex[feature * n + obj]`, feature-major). Reuses the `score_split` fixture shape:
/// feature 0's bins climb monotonically with the object index so the der1 ramp aligns
/// with the bin axis (a clear best border), other features get a different deterministic
/// spread (lower gain).
fn make_fixture(
    n: usize,
    n_features: usize,
    n_bins: usize,
) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
    let der1: Vec<f64> = (0..n).map(|k| (k as f64) - (n as f64) / 2.0).collect();
    let weight: Vec<f64> = (0..n).map(|k| 0.5 + ((k % 5) as f64) * 0.25).collect();
    let mut cindex = vec![0u32; n_features * n];
    for feature in 0..n_features {
        for obj in 0..n {
            let bin = if feature == 0 {
                ((obj * n_bins) / n.max(1)).min(n_bins - 1)
            } else {
                (obj * (feature + 2) + feature) % n_bins
            };
            cindex[feature * n + obj] = bin as u32;
        }
    }
    let indices: Vec<u32> = (0..n as u32).collect();
    (der1, weight, cindex, indices)
}

// NOTE (Task 2): the inline whole-dataset L2 score reference + the strict-first-wins CPU
// greedy-search transcription (mirroring `cb_train::greedy_tensor_search_oblivious` +
// `select_best_candidate`) and the `cb_compute::calc_average` leaf-value oracle land with
// the `single_tree` cross-oracle below (it imports `cb_compute::{l2_split_score,
// calc_average, scale_l2_reg, LeafStats}` at that point).

// ===========================================================================
// Partition primitives self-oracle (Phase 7.5 Plan C, Task 1): the device
// `partition_split_kernel` forward-bit doc-routing reorder must match the CPU
// `leaf_index` per object EXACTLY (Pitfall 6 — the SC-3 structure check), and the device
// `partition_update_kernel` per-partition Σ der1 / Σ weight reduce must match the host
// ORDERED `sum_f64` per-leaf reference within the reported tolerance (D-7.5-05). The bulk
// doc-routing stays device-resident (handle-in / handle-out); only the test reads the
// final leaf_of / part-stats back to assert parity.
// ===========================================================================

mod partition {
    use super::*;

    /// Apply a known split sequence on-device via repeated `launch_partition_split_into`
    /// (threading ONE client, all handles device-resident), read back the final per-object
    /// `leaf_of`, and assert it equals the CPU `leaf_index` over the SAME split sequence
    /// for EVERY object (forward-bit order, bit `i` for split `i` — Pitfall 6).
    #[test]
    fn leaf_of_matches_cpu_leaf_index() {
        let n_features = 3usize;
        let n_bins = 32usize;

        for &n in &[1usize, 37usize, 1000usize] {
            let (der1, _weight, cindex, indices) = make_fixture(n, n_features, n_bins);

            // A known split sequence: 3 levels on distinct features at mid-range borders.
            // (feature, bin) per level; the CPU passes-test is `cindex bin > bin`.
            let splits: Vec<(usize, usize)> = vec![(0, 15), (1, 10), (2, 20)];

            let device = <crate::SelectedRuntime as Runtime>::Device::default();
            let client = <crate::SelectedRuntime as Runtime>::client(&device);

            // Resident handles uploaded ONCE (the grow-loop seam: one client, persistent
            // buffers). der1 is threaded to keep the kernel's F generic real.
            let der1_h = upload_channel_floats(&client, &der1);
            let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.clone()));
            let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.clone()));
            // leaf_of starts all-zero (every object in partition 0).
            let mut leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));

            let cindex_stride = n_features * n;
            for (level, &(feature, bin)) in splits.iter().enumerate() {
                leaf_of_h = launch_partition_split_into(
                    &client,
                    der1_h.clone(),
                    cindex_h.clone(),
                    indices_h.clone(),
                    leaf_of_h,
                    n,
                    cindex_stride,
                    feature as u32,
                    bin as u32,
                    level as u32,
                )
                .expect("partition split must launch");
            }

            let device_leaf_of = read_u32_handle(&client, leaf_of_h).expect("read leaf_of");

            // CPU reference: forward-bit leaf_index over the same passes per object.
            let cpu_leaf_of: Vec<u32> = (0..n)
                .map(|obj| {
                    let passes: Vec<bool> = splits
                        .iter()
                        .map(|&(feature, bin)| (cindex[feature * n + obj] as usize) > bin)
                        .collect();
                    cpu_leaf_index(&passes) as u32
                })
                .collect();

            assert_eq!(
                device_leaf_of.len(),
                cpu_leaf_of.len(),
                "device leaf_of length must equal n (n={n})"
            );
            assert_eq!(
                device_leaf_of, cpu_leaf_of,
                "device partition_split leaf_of must equal CPU leaf_index forward-bit (n={n})"
            );
        }
    }

    /// After applying a split sequence, the device `partition_update_kernel` per-partition
    /// Σ der1 / Σ weight must equal the host ORDERED `sum_f64` per-leaf reference within
    /// the reported tolerance (D-7.5-05). Validates the per-partition reduce
    /// (== upstream `UpdatePartitionProps`) over the SAME device-resident routing.
    #[test]
    fn update_matches_ordered_reference() {
        let n_features = 3usize;
        let n_bins = 32usize;

        for &n in &[1usize, 37usize, 1000usize] {
            let (der1, weight, cindex, indices) = make_fixture(n, n_features, n_bins);
            let splits: Vec<(usize, usize)> = vec![(0, 15), (1, 10)];
            let depth = splits.len();
            let n_parts = 1usize << depth;

            let device = <crate::SelectedRuntime as Runtime>::Device::default();
            let client = <crate::SelectedRuntime as Runtime>::client(&device);

            let der1_h = upload_channel_floats(&client, &der1);
            let weight_h = upload_channel_floats(&client, &weight);
            let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.clone()));
            let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.clone()));
            let mut leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));

            let cindex_stride = n_features * n;
            for (level, &(feature, bin)) in splits.iter().enumerate() {
                leaf_of_h = launch_partition_split_into(
                    &client,
                    der1_h.clone(),
                    cindex_h.clone(),
                    indices_h.clone(),
                    leaf_of_h,
                    n,
                    cindex_stride,
                    feature as u32,
                    bin as u32,
                    level as u32,
                )
                .expect("partition split must launch");
            }

            // Device per-partition reduce (the leaf_of handle is consumed; clone it so the
            // host reference can read the SAME routing back).
            let part_stats_h = launch_partition_update_into(
                &client,
                der1_h.clone(),
                weight_h.clone(),
                indices_h.clone(),
                leaf_of_h.clone(),
                n,
                n_parts,
            )
            .expect("partition update must launch");
            let device_stats =
                read_part_stats_f64(&client, part_stats_h).expect("read part-stats");

            // Host ordered reference: read the device routing back, fold each partition's
            // der1/weight in ascending object order via sum_f64 (NEVER naive .sum(), D-08).
            let device_leaf_of = read_u32_handle(&client, leaf_of_h).expect("read leaf_of");
            let mut baseline = vec![0.0_f64; n_parts * 2];
            for part in 0..n_parts {
                let mut der_seg: Vec<f64> = Vec::new();
                let mut w_seg: Vec<f64> = Vec::new();
                for obj in 0..n {
                    if device_leaf_of[obj] as usize == part {
                        der_seg.push(der1[obj]);
                        w_seg.push(weight[obj]);
                    }
                }
                baseline[part * 2] = sum_f64(&der_seg);
                baseline[part * 2 + 1] = sum_f64(&w_seg);
            }

            assert_eq!(
                device_stats.len(),
                baseline.len(),
                "device part-stats length must equal n_parts * 2 (n={n})"
            );
            let (abs, rel) = max_divergence(&device_stats, &baseline);
            println!(
                "[partition_update n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} \
                 (bound={LEAF_BOUND:.0e})"
            );
            assert!(
                rel <= LEAF_BOUND || abs <= LEAF_BOUND,
                "device partition_update (n={n}) diverged too far: abs={abs:.3e} rel={rel:.3e} \
                 (bound={LEAF_BOUND:.0e})"
            );
        }
    }
}
