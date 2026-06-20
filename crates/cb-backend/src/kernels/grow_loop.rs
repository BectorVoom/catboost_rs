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

/// The whole-dataset L2 score of ONE binary split `(feature, bin)` over the fixture —
/// the depth-1 stump score, TRANSCRIBED from the FROZEN
/// `cb_compute::{l2_split_score, LeafStats}` semantics (`score.rs:39-55`) + the leaf
/// partition `cindex[feature * n + obj] > bin` (forward-bit, == the device
/// `partition_split` test). LEFT leaf = bins `0..=bin`, RIGHT leaf = bins `bin+1..`.
/// Each side's Σ der1 / Σ weight is folded in ASCENDING OBJECT ORDER via `sum_f64`
/// (NEVER naive `.sum()`, D-08), matching `cb_compute::reduce_leaf_stats`. This is the
/// inline transcription the D-7.5-04 boundary mandates (do NOT import `cb-train`).
fn cpu_stump_score(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    n: usize,
    feature: usize,
    bin: usize,
    scaled_l2: f64,
) -> f64 {
    let mut left_der: Vec<f64> = Vec::new();
    let mut left_w: Vec<f64> = Vec::new();
    let mut right_der: Vec<f64> = Vec::new();
    let mut right_w: Vec<f64> = Vec::new();
    for obj in 0..n {
        // Forward-bit pass test == the device `partition_split_kernel` (`cindex > bin`).
        if (cindex[feature * n + obj] as usize) > bin {
            right_der.push(der1[obj]);
            right_w.push(weight[obj]);
        } else {
            left_der.push(der1[obj]);
            left_w.push(weight[obj]);
        }
    }
    let left = cb_compute::LeafStats {
        sum_weighted_delta: sum_f64(&left_der),
        sum_weight: sum_f64(&left_w),
    };
    let right = cb_compute::LeafStats {
        sum_weighted_delta: sum_f64(&right_der),
        sum_weight: sum_f64(&right_w),
    };
    // L2 score = Σ add_leaf_plain(leaf) over the two leaves (l2_split_score:49-55).
    cb_compute::l2_split_score(&[left, right], scaled_l2)
}

/// The inline CPU greedy LEVEL-0 search — the strict-first-wins L2 argmax over the
/// candidates in upstream ascending `(feature, bin)` order, TRANSCRIBED from
/// `cb_train::greedy_tensor_search_oblivious` + `cb_train::select_best_candidate`
/// (`tree.rs:291-302`, `:486-643`): iterate features ascending, bins ascending, keep the
/// FIRST candidate whose score STRICTLY exceeds the running best (strict `>`, NOT `>=` —
/// the load-bearing first-wins tie-break, Pitfall 1). Returns the chosen `(feature, bin)`
/// or `None` if no candidate exists. Mirrors the device's lowest-`(feature, bin)`-index
/// tie-break so the two agree on a near-tie (Pattern 4).
fn cpu_best_stump(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    n: usize,
    n_features: usize,
    n_bins: usize,
    scaled_l2: f64,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut best_score = f64::NEG_INFINITY;
    for feature in 0..n_features {
        for bin in 0..n_bins {
            let score = cpu_stump_score(der1, weight, cindex, n, feature, bin, scaled_l2);
            // STRICT `>` (first-wins on equal score, ascending (feature, bin) order).
            if score > best_score {
                best_score = score;
                best = Some((feature, bin));
            }
        }
    }
    best
}

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

// ===========================================================================
// Single-tree cross-oracle (Phase 7.5 Plan C, Task 2; SC-3): grow ONE oblivious tree
// device-resident via `grow_oblivious_tree` and assert its STRUCTURE (the split
// `(feature, bin)` sequence AND the per-object `leaf_of`) matches the inline CPU
// greedy-search transcription (`cb_train::greedy_tensor_search_oblivious` /
// `leaf_index`) EXACTLY — the strict bar. Leaf-value divergence vs
// `cb_compute::calc_average` is REPORTED within the run-stable bound (NOT the GPU-06
// epsilon — 7.6's job). The loop is host-light: only the O(1) BestSplit per level + ONE
// 2^depth part-stats read-back cross host<->device (D-05, enforced by construction in
// `grow_oblivious_tree`).
// ===========================================================================

mod single_tree {
    use super::*;
    use crate::gpu_runtime::grow_oblivious_tree;

    /// The per-tree L2 scaling — `cb_compute::scale_l2_reg(l2, Σweight, n)`. For the
    /// fixture's per-object weights this is the FROZEN per-tree scaling the CPU oracle
    /// and the device leaf-value step both consume.
    fn scaled_l2_for(weight: &[f64], n: usize, l2: f64) -> f64 {
        let sum_w = sum_f64(weight);
        cb_compute::scale_l2_reg(l2, sum_w, n)
    }

    /// Grow a depth-1 oblivious tree (the MVP vertical slice: one split / stump, the
    /// strict O(1)-per-level device-resident path) on the clear-gain-margin fixture and
    /// assert the device STRUCTURE matches the inline CPU greedy search EXACTLY (split
    /// `(feature, bin)` + per-object `leaf_of` == `leaf_index`), then REPORT the
    /// leaf-value divergence vs `cb_compute::calc_average`. f64 channel (rocm/cuda/cpu)
    /// and f32 channel (wgpu) both run over `SelectedRuntime`.
    #[test]
    fn matches_cpu_greedy_search() {
        let n_features = 3usize;
        let n_bins = 32usize;
        let depth = 1usize;
        let l2 = 3.0_f64;

        for &n in &[1usize, 37usize, 1000usize] {
            let (der1, weight, cindex, indices) = make_fixture(n, n_features, n_bins);
            let scaled_l2 = scaled_l2_for(&weight, n, l2);

            // Device: grow the tree host-light over SelectedRuntime.
            let tree = grow_oblivious_tree(
                &der1, &weight, &cindex, &indices, n_bins, n_features, depth, scaled_l2,
                crate::kernels::SCORE_FN_L2,
            )
            .expect("grow_oblivious_tree must succeed on the clear-margin fixture");

            // CPU reference: the strict-first-wins level-0 stump (inline transcription).
            let cpu_split = cpu_best_stump(&der1, &weight, &cindex, n, n_features, n_bins, scaled_l2)
                .expect("CPU reference must find a candidate split");

            // (A) STRUCTURE — the split (feature, bin) sequence must match EXACTLY.
            assert_eq!(
                tree.splits.len(),
                depth,
                "device tree must have exactly `depth` splits (n={n})"
            );
            let (dev_feat, dev_bin) = tree.splits[0];
            assert_eq!(
                (dev_feat as usize, dev_bin as usize),
                cpu_split,
                "device split (feature, bin) must match CPU greedy first-wins (n={n}): \
                 device=({dev_feat}, {dev_bin}) cpu={cpu_split:?}"
            );

            // (B) STRUCTURE — per-object leaf_of must equal CPU leaf_index over the SAME
            //     split (forward-bit, Pitfall 6) for EVERY object.
            let (cpu_feature, cpu_bin) = cpu_split;
            let cpu_leaf_of: Vec<u32> = (0..n)
                .map(|obj| {
                    let passes = [(cindex[cpu_feature * n + obj] as usize) > cpu_bin];
                    cpu_leaf_index(&passes) as u32
                })
                .collect();
            assert_eq!(
                tree.leaf_of, cpu_leaf_of,
                "device leaf_of must equal CPU leaf_index forward-bit (n={n})"
            );

            // (C) LEAF VALUES — REPORTED divergence vs the CPU calc_average over the SAME
            //     leaf partition (NOT signed off — 7.6 owns the epsilon, D-7.5-05).
            let n_leaves = 1usize << depth;
            let mut cpu_leaf_values = vec![0.0_f64; n_leaves];
            for leaf in 0..n_leaves {
                let mut der_seg: Vec<f64> = Vec::new();
                let mut w_seg: Vec<f64> = Vec::new();
                for obj in 0..n {
                    if cpu_leaf_of[obj] as usize == leaf {
                        der_seg.push(der1[obj]);
                        w_seg.push(weight[obj]);
                    }
                }
                cpu_leaf_values[leaf] =
                    cb_compute::calc_average(sum_f64(&der_seg), sum_f64(&w_seg), scaled_l2);
            }
            assert_eq!(
                tree.leaf_values.len(),
                cpu_leaf_values.len(),
                "device leaf_values length must equal n_leaves (n={n})"
            );
            let (abs, rel) = max_divergence(&tree.leaf_values, &cpu_leaf_values);
            println!(
                "[single_tree n={n}] STRUCTURE match: split={cpu_split:?}; REPORTED leaf-value \
                 max abs_div={abs:.3e} rel_div={rel:.3e} (bound={LEAF_BOUND:.0e})"
            );
            assert!(
                rel <= LEAF_BOUND || abs <= LEAF_BOUND,
                "device leaf values (n={n}) diverged beyond the REPORTED bound: \
                 abs={abs:.3e} rel={rel:.3e} (bound={LEAF_BOUND:.0e})"
            );
        }
    }

    /// `depth > 1` surfaces the typed `CbError::OutOfRange` documenting the
    /// partition-aware-histogram forward dependency — NOT a silently-mislabeled stump.
    #[test]
    fn depth_gt_one_is_tracked_forward_dependency() {
        let n_features = 2usize;
        let n_bins = 16usize;
        let n = 50usize;
        let (der1, weight, cindex, indices) = make_fixture(n, n_features, n_bins);
        let scaled_l2 = scaled_l2_for(&weight, n, 3.0);

        let err = grow_oblivious_tree(
            &der1, &weight, &cindex, &indices, n_bins, n_features, 2, scaled_l2,
            crate::kernels::SCORE_FN_L2,
        )
        .expect_err("depth > 1 must surface a typed forward-dependency error, not a stump");
        // It must be the OutOfRange forward-dependency guard (not a panic / wrong tree).
        let msg = format!("{err:?}");
        assert!(
            msg.contains("depth") && msg.contains("forward dependency"),
            "depth>1 error must name the partition-aware-histogram forward dependency: {msg}"
        );
    }
}

// ===========================================================================
// Multi-tree cross-oracle (Phase 7.5 Plan 04; SC-3 / D-7.5-06): run a full multi-tree
// Plain-boosting pass device-resident via `grow_boosting_pass` and assert EVERY tree's
// STRUCTURE (the split `(feature, bin)` sequence AND the per-object `leaf_of`) matches
// the inline CPU multi-tree Plain-boosting reference EXACTLY across the WHOLE run (the
// strict bar — atomic-order jitter must NOT flip a split on the clear-gain-margin
// fixture). Per-tree leaf-value divergence vs `cb_compute::calc_average` is REPORTED
// (NOT the GPU-06 epsilon — 7.6's job). An N-run stability check REPORTS whether per-tree
// structure is stable run-to-run; an observed argmax-flip on a near-tie is the D-7.5-06
// boundary to REPORT, never to sign off here.
//
// The CPU reference is the EXACT device convention transcribed inline (the D-7.5-04
// boundary — do NOT import `cb-train`): start approx all-zero, per iteration der1 =
// `target - approx` (RMSE gradient, `cb_compute::loss::rmse_der1`), grow the depth-1
// stump via `cpu_best_stump` over der1, leaf delta `calc_average(Σder1, Σweight,
// scaled_l2)`, leaf value `learning_rate * delta`, then `approx[i] += leaf_value[leaf(i)]`
// (`cb_train::boosting` boosting.rs:9-16). der2 is the constant `-1.0` (folded; the L2
// score uses Σ der1 / Σ weight, not the Newton hessian — MVP).
// ===========================================================================

mod multi_tree {
    use super::*;
    use crate::gpu_runtime::{grow_boosting_pass, GrownModel};

    /// The per-tree L2 scaling — `cb_compute::scale_l2_reg(l2, Σweight, n)` (the SAME
    /// scaling the device leaf-value step and the CPU reference both consume).
    fn scaled_l2_for(weight: &[f64], n: usize, l2: f64) -> f64 {
        let sum_w = sum_f64(weight);
        cb_compute::scale_l2_reg(l2, sum_w, n)
    }

    /// A clear-gain-margin multi-tree REGRESSION fixture (target-based, the boosting
    /// input — NOT the der1-based `make_fixture`). Feature 0's bins climb monotonically
    /// with the object index AND the target ramps with the object index, so feature 0
    /// carries a clear, stable best border every boosting iteration (the residual stays
    /// monotone-ish in feature 0). Other features get a deterministic lower-gain spread.
    /// Returns `(target, weight, cindex, indices)` in the FROZEN feature-major layout
    /// (`cindex[feature * n + obj]`).
    fn make_boosting_fixture(
        n: usize,
        n_features: usize,
        n_bins: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        // A clear monotone target ramp (centred) — feature 0's bins track it, so the
        // greedy stump split on feature 0 is unambiguous and stable across iterations.
        let target: Vec<f64> = (0..n).map(|k| (k as f64) - (n as f64) / 2.0).collect();
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
        (target, weight, cindex, indices)
    }

    /// The CPU multi-tree Plain-boosting reference (inline transcription, NOT imported):
    /// returns the per-tree `(split, leaf_of, leaf_values)` over the whole run, mirroring
    /// the device `grow_boosting_pass` convention EXACTLY. Depth-1 (the MVP scope, same as
    /// `grow_oblivious_tree`): one stump per tree.
    fn cpu_boosting_pass(
        target: &[f64],
        weight: &[f64],
        cindex: &[u32],
        n: usize,
        n_features: usize,
        n_bins: usize,
        iterations: usize,
        learning_rate: f64,
        scaled_l2: f64,
    ) -> Vec<((usize, usize), Vec<u32>, Vec<f64>)> {
        let mut approx = vec![0.0_f64; n];
        let mut out: Vec<((usize, usize), Vec<u32>, Vec<f64>)> = Vec::with_capacity(iterations);
        for _iter in 0..iterations {
            // Residual gradient der1 = target - approx (RMSE, cb_compute::loss::rmse_der1).
            let der1: Vec<f64> = (0..n).map(|i| cb_compute::rmse_der1(approx[i], target[i])).collect();

            // Greedy depth-1 stump over the current residual (strict first-wins).
            let split = cpu_best_stump(&der1, weight, cindex, n, n_features, n_bins, scaled_l2)
                .expect("CPU reference must find a candidate split each iteration");
            let (feature, bin) = split;

            // Forward-bit leaf_of over the chosen stump (== cpu_leaf_index, Pitfall 6).
            let leaf_of: Vec<u32> = (0..n)
                .map(|obj| {
                    let passes = [(cindex[feature * n + obj] as usize) > bin];
                    cpu_leaf_index(&passes) as u32
                })
                .collect();

            // Per-leaf delta via calc_average over der1 (ordered sum_f64), scaled by lr.
            let n_leaves = 2usize; // depth == 1
            let mut leaf_values = vec![0.0_f64; n_leaves];
            for leaf in 0..n_leaves {
                let mut der_seg: Vec<f64> = Vec::new();
                let mut w_seg: Vec<f64> = Vec::new();
                for obj in 0..n {
                    if leaf_of[obj] as usize == leaf {
                        der_seg.push(der1[obj]);
                        w_seg.push(weight[obj]);
                    }
                }
                let delta = cb_compute::calc_average(sum_f64(&der_seg), sum_f64(&w_seg), scaled_l2);
                leaf_values[leaf] = learning_rate * delta;
            }

            // Update the running approx (approx[i] += leaf_value[leaf(i)]).
            for obj in 0..n {
                approx[obj] += leaf_values[leaf_of[obj] as usize];
            }

            out.push((split, leaf_of, leaf_values));
        }
        out
    }

    /// Run `grow_boosting_pass` (device) and the inline CPU multi-tree Plain-boosting
    /// reference on the SAME clear-gain-margin fixture/seed/params (f32 + f64 channel via
    /// `SelectedRuntime`, several iterations); assert EVERY tree's split `(feature, bin)`
    /// + per-object `leaf_of` matches the CPU reference EXACTLY (the SC-3 strict bar
    /// across the whole run); REPORT the per-tree + aggregate leaf-value divergence.
    #[test]
    fn matches_cpu_multi_tree_boosting() {
        let n_features = 3usize;
        let n_bins = 32usize;
        let depth = 1usize;
        let l2 = 3.0_f64;
        let iterations = 5usize;
        let learning_rate = 0.3_f64;

        for &n in &[1usize, 37usize, 1000usize] {
            let (target, weight, cindex, indices) = make_boosting_fixture(n, n_features, n_bins);
            let scaled_l2 = scaled_l2_for(&weight, n, l2);

            // Device: the full multi-tree boosting pass over SelectedRuntime.
            let model: GrownModel = grow_boosting_pass(
                &target, &weight, &cindex, &indices, n_bins, n_features, iterations,
                learning_rate, depth, scaled_l2,
            )
            .expect("grow_boosting_pass must succeed on the clear-margin fixture");

            // CPU multi-tree reference on the SAME fixture/params.
            let cpu = cpu_boosting_pass(
                &target, &weight, &cindex, n, n_features, n_bins, iterations, learning_rate,
                scaled_l2,
            );

            assert_eq!(
                model.iterations(),
                iterations,
                "device model must have exactly `iterations` trees (n={n})"
            );
            assert_eq!(
                cpu.len(),
                iterations,
                "CPU reference must produce `iterations` trees (n={n})"
            );

            let mut run_abs = 0.0_f64;
            let mut run_rel = 0.0_f64;
            for (k, (dev_tree, (cpu_split, cpu_leaf_of, cpu_leaf_values))) in
                model.trees.iter().zip(cpu.iter()).enumerate()
            {
                // (A) STRUCTURE — the split (feature, bin) must match EXACTLY (this tree).
                assert_eq!(
                    dev_tree.splits.len(),
                    depth,
                    "device tree {k} must have exactly `depth` splits (n={n})"
                );
                let (dev_feat, dev_bin) = dev_tree.splits[0];
                assert_eq!(
                    (dev_feat as usize, dev_bin as usize),
                    *cpu_split,
                    "device tree {k} split must match CPU multi-tree first-wins (n={n}): \
                     device=({dev_feat}, {dev_bin}) cpu={cpu_split:?}"
                );

                // (B) STRUCTURE — per-object leaf_of must equal the CPU leaf_of EXACTLY.
                assert_eq!(
                    &dev_tree.leaf_of, cpu_leaf_of,
                    "device tree {k} leaf_of must equal CPU forward-bit leaf_index (n={n})"
                );

                // (C) LEAF VALUES — REPORTED divergence (NOT signed off — 7.6 owns the
                //     epsilon, D-7.5-05). The device leaf values are already lr-scaled.
                assert_eq!(
                    dev_tree.leaf_values.len(),
                    cpu_leaf_values.len(),
                    "device tree {k} leaf_values length must equal n_leaves (n={n})"
                );
                let (abs, rel) = max_divergence(&dev_tree.leaf_values, cpu_leaf_values);
                run_abs = run_abs.max(abs);
                run_rel = run_rel.max(rel);
                println!(
                    "[multi_tree n={n} tree={k}] STRUCTURE match: split={cpu_split:?}; REPORTED \
                     leaf-value max abs_div={abs:.3e} rel_div={rel:.3e}"
                );
            }
            println!(
                "[multi_tree n={n}] AGGREGATE REPORTED leaf-value divergence over {iterations} \
                 trees: max abs_div={run_abs:.3e} rel_div={run_rel:.3e} (bound={LEAF_BOUND:.0e})"
            );
            assert!(
                run_rel <= LEAF_BOUND || run_abs <= LEAF_BOUND,
                "device multi-tree leaf values (n={n}) diverged beyond the REPORTED bound: \
                 abs={run_abs:.3e} rel={run_rel:.3e} (bound={LEAF_BOUND:.0e})"
            );
        }
    }

    /// The D-7.5-06 non-determinism budget: re-run `grow_boosting_pass` N times on the
    /// clear-gain-margin fixture and REPORT whether per-tree STRUCTURE (every tree's
    /// split + leaf_of) is stable run-to-run. On the clear-margin fixture atomic-order
    /// jitter must NOT flip a split (structure stays exact); if a flip IS observed it is
    /// the REPORTED D-7.5-06 boundary (characterize — do NOT fail the suite on a
    /// documented flip, do NOT sign off an epsilon — 7.6's job). Also REPORTS the max
    /// run-to-run leaf-value drift.
    #[test]
    fn run_to_run_structure_stability_reported() {
        let n_features = 3usize;
        let n_bins = 32usize;
        let depth = 1usize;
        let l2 = 3.0_f64;
        let iterations = 5usize;
        let learning_rate = 0.3_f64;
        let n = 1000usize;
        let runs = 4usize;

        let (target, weight, cindex, indices) = make_boosting_fixture(n, n_features, n_bins);
        let scaled_l2 = scaled_l2_for(&weight, n, l2);

        // Run the pass `runs` times; the first run is the structure baseline.
        let mut models: Vec<GrownModel> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let m = grow_boosting_pass(
                &target, &weight, &cindex, &indices, n_bins, n_features, iterations,
                learning_rate, depth, scaled_l2,
            )
            .expect("grow_boosting_pass must succeed");
            models.push(m);
        }

        let baseline = &models[0];
        let mut structure_stable = true;
        let mut max_value_drift = 0.0_f64;
        for (r, m) in models.iter().enumerate().skip(1) {
            for (k, (bt, mt)) in baseline.trees.iter().zip(m.trees.iter()).enumerate() {
                // STRUCTURE stability: split + leaf_of must be identical run-to-run on the
                // clear-margin fixture. A divergence is the REPORTED argmax-flip boundary.
                if bt.splits != mt.splits || bt.leaf_of != mt.leaf_of {
                    structure_stable = false;
                    println!(
                        "[multi_tree D-7.5-06] REPORTED argmax-flip: run {r} tree {k} structure \
                         differs from baseline (split base={:?} run={:?})",
                        bt.splits, mt.splits
                    );
                }
                let (abs, _rel) = max_divergence(&mt.leaf_values, &bt.leaf_values);
                max_value_drift = max_value_drift.max(abs);
            }
        }

        println!(
            "[multi_tree D-7.5-06] REPORTED run-to-run structure_stable={structure_stable} over \
             {runs} runs x {iterations} trees (n={n}); max run-to-run leaf-value drift={max_value_drift:.3e}"
        );
        // The clear-margin fixture is engineered so structure stays exact (Pitfall 2): a
        // flip here would be the D-7.5-06 boundary REPORTED above. We assert structure
        // stability on the clear-margin fixture (the strict SC-3 bar) — if this ever fails,
        // the printed REPORT is the 7.6 epsilon input, NOT a sign-off here.
        assert!(
            structure_stable,
            "clear-gain-margin multi-tree structure must be stable run-to-run (any flip is the \
             REPORTED D-7.5-06 boundary above — characterize in 7.6, do not mask)"
        );
    }
}
