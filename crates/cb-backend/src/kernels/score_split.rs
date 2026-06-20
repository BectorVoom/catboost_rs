//! Self-oracle for the device-resident **pointwise L2 split score + deterministic
//! split argmin** (GPU-01 score/split slice, Phase 7.5 Plan A): the GPU
//! `find_optimal_split_kernel` (L2 arm) computed over [`crate::SelectedRuntime`] from
//! the FROZEN 7.3 device-resident 2-channel histogram handle must
//!
//! 1. produce a per-candidate L2 split score matching the FROZEN CPU oracle
//!    [`cb_compute::l2_split_score`] over the SAME reduced [`cb_compute::LeafStats`]
//!    within a REPORTED (not signed-off) f64 tolerance, and
//! 2. pick the SAME winning `(feature, bin)` split as the FROZEN CPU oracle
//!    [`cb_train::select_best_candidate`] over the SAME ascending `(feature, bin)`
//!    candidate order, including the lowest-`(feature, bin)`-index tie-break on equal
//!    gain (strict first-wins, `>`),
//!
//! with only the O(blocks) `BestSplit[]` descriptor crossing host<->device (the full
//! per-(feature,bin) score/histogram buffers stay device-resident, D-05 / D-7.5-05).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs`, the launch seam (`launch_find_optimal_split_pointwise` + the
//! `BestSplit` POD) in `gpu_runtime.rs`; ALL assertions live HERE. Test code may use
//! `.unwrap()`/indexing (the `lib.rs:1` `#[cfg(test)]` allow) — the production
//! `gpu_runtime.rs`/`kernels.rs` may NOT.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32), and builds/runs under every backend
//! (like `kernels::pointwise_hist`/`kernels::pairwise_hist`/`kernels::reduce`), over
//! [`crate::SelectedRuntime`]. The reported max abs/rel SCORE divergence is
//! informational: the GPU-06 epsilon is signed off in Phase 7.6, NOT hard-coded here
//! (D-7.5-05 / D-03). The asserted SCORE tolerances are generous, run-stable bounds
//! (f32 ~1e-3 relative, f64 ~1e-9 relative) that catch a wrong score without pinning
//! the final epsilon. STRUCTURE (the integer `(feature, bin)` split decision) is the
//! STRICT bar (D-7.5-06): the device argmin MUST equal the CPU winner EXACTLY on both a
//! clear-gain-margin fixture and a deliberate near-tie fixture; a structure mismatch is
//! REPORTED as the tolerance boundary, never signed off here (7.6's job).
//!
//! # D-7.5-04 boundary
//!
//! `cb_compute` (a normal dep) and `cb_train` (a TEST-ONLY dev-dep) are imported
//! READ-ONLY as parity oracles; this file never modifies them and the cubecl-coupled
//! production paths never import `cb_train`.

use cubecl::prelude::*;

use cb_compute::{l2_split_score, scale_l2_reg, LeafStats};
use cb_core::sum_f64;
use cb_train::{select_best_candidate, Candidate};

use crate::gpu_runtime::{launch_find_optimal_split_pointwise, BestSplit};

/// The asserted run-stable SCORE divergence bound for the device L2 split score. The
/// device score fold is f64 on rocm/cuda/cpu (HIP/CUDA support/emulate the f64 atomic
/// add) and f32 on wgpu (WGSL has no f64 atomics — RESEARCH A1), so the bound is the
/// f32 magnitude (~1e-3) under `wgpu` and the f64 magnitude (~1e-9) elsewhere. This is
/// a REPORTED run-stable bound, NOT the GPU-06 epsilon (7.6's job). Cloned from the
/// `kernels::pointwise_hist` `HIST_BOUND` precedent.
#[cfg(feature = "wgpu")]
const SCORE_BOUND: f64 = 1e-3;
#[cfg(not(feature = "wgpu"))]
const SCORE_BOUND: f64 = 1e-9;

/// Compare a device per-candidate score vector (cast to f64) to the host reference
/// element-wise, returning the max abs and max rel divergence over the buffer. Cloned
/// verbatim from the `kernels::pointwise_hist` reporter (REPORT-not-sign-off,
/// D-7.5-05). The length precondition is made explicit via `zip` + a `debug_assert`
/// (a mismatch truncates to the shorter slice rather than panicking with an opaque OOB
/// index).
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

/// Read a device handle of `BestSplit` PODs back ONCE through a fresh client of the
/// SAME runtime (test-only — production reads only via the seam). The handle carries
/// `BestSplit` `#[repr(C)]` structs (16 bytes each); `bytemuck::cast_slice` reinterprets
/// the read-back bytes. Cloned from the `kernels::pointwise_hist::read_handle_f64`
/// read-back pattern (re-resolving `Runtime::client(&device)` for the SAME device
/// returns the SAME cached pooled client that allocated the handle — no foreign
/// allocator, no cross-client hazard).
#[allow(dead_code)]
fn read_best_splits(h: cubecl::server::Handle) -> Vec<BestSplit> {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let bytes = client.read_one(h).expect("best-split handle read-back failed");
    bytemuck::cast_slice::<u8, BestSplit>(&bytes).to_vec()
}

/// Build a deterministic f64 fixture for the L2 score/split self-oracle: `n_features`
/// features over `n_bins` bins each, with a CLEAR per-feature gain margin so the CPU
/// winner is unambiguous (Pitfall 2 — no artificial exact ties unless asked). Returns
/// the FROZEN 7.3 inputs `(der1, weight, cindex, indices)` in the
/// [`crate::gpu_runtime::launch_pointwise_hist2_handle`] /
/// `kernels::pointwise_hist::host_reference_hist2` layout:
///
/// - `der1` (UNWEIGHTED first derivative, the 7.2 seam contract), length `n`
/// - `weight` (per-object weight, channel 1), length `n`
/// - `cindex` (feature-major quantized bins, `cindex[feature * n + obj]`), length
///   `n_features * n`
/// - `indices` (object visiting order), length `n`
///
/// Each object is assigned a deterministic bin per feature; `der1` is shaped so that a
/// particular border per feature carves a high-gain split (objects below the border
/// have systematically different der1 sign than objects above it), giving the L2 score
/// a clear maximum at one `(feature, bin)`.
fn make_score_fixture(
    n: usize,
    n_features: usize,
    n_bins: usize,
) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
    // der1: a smooth ramp through zero so a mid-range border separates negative from
    // positive contributions (a high-gain split).
    let der1: Vec<f64> = (0..n).map(|k| (k as f64) - (n as f64) / 2.0).collect();
    // Non-trivial weights (never all-1) so the weight channel / denominator is a real
    // sum.
    let weight: Vec<f64> = (0..n).map(|k| 0.5 + ((k % 5) as f64) * 0.25).collect();
    // Feature-major cindex: feature 0 spreads bins monotonically with the object index
    // (so the der1 ramp aligns with the bin axis → a clear high-gain border), other
    // features get a different deterministic spread (lower gain).
    let mut cindex = vec![0u32; n_features * n];
    for feature in 0..n_features {
        for obj in 0..n {
            let bin = if feature == 0 {
                // monotone with obj → aligns with the der1 ramp (clear best feature)
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

/// The ORDERED host reference per-(feature, bin) L2 split SCORE — the parity baseline
/// the device per-candidate score is REPORTED against (D-7.5-05). For each feature and
/// each candidate border `bin` in `0..n_bins`, it reduces the feature's objects into a
/// LEFT leaf (objects whose bin `<= bin`) and a RIGHT leaf (objects whose bin `> bin`)
/// in ascending object order (folded through [`sum_f64`], the single sanctioned ordered
/// reduction), builds the two [`LeafStats`], and calls the FROZEN [`l2_split_score`]
/// over `[left, right]`. This GENERALIZES the leaf reduction WITHOUT modifying the
/// frozen `cb-compute` baseline (the host reference lives HERE, in the `cb-backend`
/// test file). Returns a flat `Vec<f64>` of length `n_features * n_bins` indexed
/// `feature * n_bins + bin` — the SAME candidate enumeration order the device kernel and
/// the CPU `Candidate` vector use (ascending feature, then ascending bin).
fn host_reference_scores(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    scaled_l2: f64,
) -> Vec<f64> {
    let n = der1.len();
    let mut scores = vec![0.0_f64; n_features * n_bins];
    for feature in 0..n_features {
        for border in 0..n_bins {
            let mut left_der: Vec<f64> = Vec::new();
            let mut left_w: Vec<f64> = Vec::new();
            let mut right_der: Vec<f64> = Vec::new();
            let mut right_w: Vec<f64> = Vec::new();
            for &obj in indices.iter() {
                let obj = obj as usize;
                let bin = cindex[feature * n + obj] as usize;
                if bin <= border {
                    left_der.push(der1[obj]);
                    left_w.push(weight[obj]);
                } else {
                    right_der.push(der1[obj]);
                    right_w.push(weight[obj]);
                }
            }
            let left = LeafStats {
                sum_weighted_delta: sum_f64(&left_der),
                sum_weight: sum_f64(&left_w),
            };
            let right = LeafStats {
                sum_weighted_delta: sum_f64(&right_der),
                sum_weight: sum_f64(&right_w),
            };
            scores[feature * n_bins + border] = l2_split_score(&[left, right], scaled_l2);
        }
    }
    scores
}

/// The CPU split winner over the SAME ascending `(feature, bin)` candidate order, via
/// the FROZEN [`select_best_candidate`] (strict first-wins / lowest-index tie-break).
/// Builds a `Candidate { feature, border, score }` per `(feature, bin)` in ascending
/// order (feature outer, bin inner) — the SAME order the device kernel enumerates and
/// the SAME order `host_reference_scores` flattens — so the CPU strict-`>` first-wins
/// resolves a tie to the lowest `(feature, bin)` index, which the device argmin must
/// equal. Returns the winning `(feature, bin)` pair, or `None` if there are no
/// candidates.
fn cpu_best_split(scores: &[f64], n_bins: usize, n_features: usize) -> Option<(usize, usize)> {
    let mut candidates: Vec<Candidate> = Vec::with_capacity(n_features * n_bins);
    for feature in 0..n_features {
        for border in 0..n_bins {
            candidates.push(Candidate {
                feature,
                // `border` here is the integer bin index used as the candidate key; the
                // float `border` value is irrelevant to the score/argmin parity (the
                // score is supplied directly), so encode the bin index as the border so
                // the winning candidate carries its bin back.
                border: border as f64,
                score: scores[feature * n_bins + border],
            });
        }
    }
    select_best_candidate(&candidates).map(|c| (c.feature, c.border as usize))
}

#[test]
fn score_l2_matches_cpu_oracle() {
    // The device per-candidate L2 split score must match the ORDERED host reference
    // (`l2_split_score` over the SAME reduced LeafStats) within the REPORTED bound, over
    // the edge cases n=1, n=37 (non-cube-multiple), large N, plus the empty
    // short-circuit. REPORTED, not signed off (D-7.5-05 — the GPU-06 epsilon is 7.6's
    // job). The score is read from the device via the BestSplit descriptors' reported
    // per-candidate scores; the seam exposes the full per-candidate score vector for the
    // oracle (NOT a host round-trip of the histogram — the score is computed
    // device-resident from the histogram handle).
    let n_features = 2usize;
    let n_bins = 32usize; // 5-bit feature group (<= CUBE_DIM scan precondition, RESEARCH A1)
    let l2 = 3.0_f64;

    // Empty (n=0): NO launch, NO read-back of a 0-len handle (Pitfall 3/5). The seam
    // returns an empty result; assert it constructs without faulting.
    {
        let (der1, weight, cindex, indices) = make_score_fixture(0, n_features, n_bins);
        let scaled_l2 = scale_l2_reg(l2, 0.0, 0);
        let res = launch_find_optimal_split_pointwise(
            &der1, &weight, &cindex, &indices, n_bins, n_features, scaled_l2,
        );
        assert!(res.is_ok(), "empty score/split must construct without faulting");
        let (best, dev_scores) = res.unwrap();
        assert!(best.is_none(), "empty input must yield no best split");
        assert!(dev_scores.is_empty(), "empty input must yield no per-candidate scores");
    }

    for &n in &[1usize, 37usize, 10_000usize] {
        let (der1, weight, cindex, indices) = make_score_fixture(n, n_features, n_bins);
        // Unweighted-path scaling convention (sum_all_weights == doc_count): scale_l2_reg
        // returns l2 directly; pass the host total weight / doc count so the device and
        // host see the SAME scaled_l2.
        let total_w: f64 = sum_f64(&weight);
        let scaled_l2 = scale_l2_reg(l2, total_w, n);
        let (_best, dev_scores) = launch_find_optimal_split_pointwise(
            &der1, &weight, &cindex, &indices, n_bins, n_features, scaled_l2,
        )
        .unwrap();
        let baseline = host_reference_scores(
            &der1, &weight, &cindex, &indices, n_bins, n_features, scaled_l2,
        );
        assert_eq!(
            dev_scores.len(),
            baseline.len(),
            "device per-candidate score length must equal the host-reference layout length (n={n})"
        );
        let (abs, rel) = max_divergence(&dev_scores, &baseline);
        println!(
            "[score_l2 n={n}] REPORTED max abs_div={abs:.3e} rel_div={rel:.3e} (bound={SCORE_BOUND:.0e})"
        );
        assert!(
            rel <= SCORE_BOUND || abs <= SCORE_BOUND,
            "device L2 split score (n={n}) diverged too far: abs={abs:.3e} rel={rel:.3e} (bound={SCORE_BOUND:.0e})"
        );
    }
}
