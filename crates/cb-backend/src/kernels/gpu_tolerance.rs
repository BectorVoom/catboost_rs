//! GPU-06 measurement harness (Phase 7.6 Plan 01): the EVIDENCE roll-up that the
//! epsilon sign-off (Plan 02) and the tolerance doc are derived from.
//!
//! # What this module measures (and what it does NOT)
//!
//! This is a `#[cfg(test)]` MEASUREMENT module — it does NOT add a new kernel. It
//! COMPOSES the existing per-kernel-family divergence comparisons (der/hess,
//! pointwise histogram, pairwise histogram, score/split, reduce) into one
//! `[GPU-06 EVIDENCE]` console line per family, adds an N≥30 run-to-run variance
//! loop with `stddev` + an `observed_max + 3σ` headroom term, and measures the
//! end-to-end GPU-trained-vs-CPU-trained model leaf values (the numbers Phase 7.5
//! left REPORTED-not-signed-off, 07.5-03/04/06). Running the rocm suite in-env on
//! gfx1100 then emits the evidence lines.
//!
//! # The seam: `SelectedRuntime` rocm arm vs the CpuRuntime baseline
//!
//! Every device launch goes through [`crate::SelectedRuntime`] (the rocm arm on
//! gfx1100, wave32), and the comparison baseline is the `cb-core::sum_f64` /
//! `cb-compute` CPU reference TRANSCRIBED INLINE here. The measured number is the
//! rocm-device-vs-Rust-CPU divergence — which is meaningful ONLY when
//! `SelectedRuntime` actually resolves to the GPU. Under the default `cpu` feature
//! it would resolve to `CpuRuntime` and measure CPU-vs-CPU (always bit-exact,
//! meaningless), so the harness MUST be run `--no-default-features --features rocm`
//! (T-07.6-01). The evidence lines print `channel=f64(rocm/gfx1100)` so a wrong
//! channel is visible in the output.
//!
//! # REPORT, not sign-off
//!
//! The asserted [`TOL_BOUND_F64`] / [`TOL_BOUND_F32`] are generous, run-stable
//! BOUNDS that catch a WRONG result without pinning the final epsilon. They are NOT
//! the GPU-06 epsilon — that is Plan 02's signed-off DOCUMENTED number, never a test
//! constant (D-7.6-02 / the precedent set by `reduce.rs` / `gradient_gpu.rs` /
//! `pointwise_hist.rs`). The `observed_max + 3σ` headroom term emitted here is the
//! EMPIRICAL INPUT to that epsilon proposal, not the epsilon itself.
//!
//! # Source/test separation + the cb-train landmine
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernels live in
//! `kernels.rs` and the launch seams in `gpu_runtime.rs`; all assertions live HERE
//! in a standalone `#[cfg(test)] mod gpu_tolerance` file (NEVER an inline
//! `#[cfg(test)] mod tests` in a production file). This module NEVER imports
//! `cb-train`: doing so would pull its `cb-backend = {path}` default-`cpu` dependency
//! into the test build graph, cargo feature unification would then activate
//! `cb-backend/cpu` ALONGSIDE `rocm`, `SelectedRuntime` would resolve to `CpuRuntime`
//! (which lacks `Atomic<f64>`), and the harness would silently report a FAKE 0.0
//! divergence (T-07.6-02 / the grow_loop.rs:29-36 LANDMINE). Every CPU reference is
//! TRANSCRIBED INLINE; `cb_compute` / `cb_core` are imported READ-ONLY. Like
//! `gradient_gpu` / `pointwise_hist` (and UNLIKE the cpu-only `gradient`/`scatter`
//! spikes), this runs over the generic `SelectedRuntime`, so it builds under EVERY
//! backend (rocm in-env + wgpu host run + cuda/cpu compile).

use cubecl::prelude::*;

use cb_core::sum_f64;

use crate::gpu_runtime::{
    grow_oblivious_tree, launch_block_reduce_atomic_f64, launch_der_binary, launch_der_unary,
    launch_find_optimal_split_pointwise, launch_pairwise_hist, launch_pointwise_hist2,
    AtomicFinalizePath, DerBinaryKernel, DerUnaryKernel,
};

/// The N≥30 floor for the run-to-run variance loop (D-7.6-02). 32 runs both clears
/// the floor and matches the `reduce.rs` atomic-finalize loop precedent.
const VARIANCE_RUNS: usize = 32;

/// The asserted run-stable divergence BOUND for the f64 device channel
/// (rocm/cuda/cpu — HIP/CUDA support or emulate the f64 atomic add). This is a
/// REPORTED bound that catches a wrong result, NOT the GPU-06 epsilon (Plan 02's
/// signed-off documented number, never a test constant). Mirrors
/// `reduce::F64_REL_TOL` / `pointwise_hist::HIST_BOUND` / `grow_loop::LEAF_BOUND`.
#[cfg(not(feature = "wgpu"))]
const TOL_BOUND_F64: f64 = 1e-9;
/// On wgpu the device channel is f32 (WGSL has no f64 atomics — RESEARCH A1), so the
/// run-stable bound is the f32 magnitude. Same REPORT-not-epsilon caveat.
#[cfg(feature = "wgpu")]
const TOL_BOUND_F64: f64 = 1e-3;

/// The f32-magnitude run-stable bound (used for the der/hess f32-fixture family and
/// the wgpu channel). REPORTED, NOT the epsilon.
const TOL_BOUND_F32: f64 = 1e-3;

/// Compare the device result (cast to f64) to the CPU baseline element-wise,
/// returning the max abs and max rel divergence over the vector. Copied VERBATIM from
/// `gradient_gpu.rs:34-49` (the canonical abs/rel reporter — IN-02/IN-03): zip the two
/// slices so a length mismatch surfaces a clear precondition instead of an opaque
/// index-out-of-bounds panic (WR-03). REPORT-not-sign-off (D-7.6-02).
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

/// Mean / variance / population-stddev of a sample, folded through the SANCTIONED
/// ordered reduction `cb_core::sum_f64` (NEVER `std::iter::sum` — D-08 reduction
/// order). Returns `(mean, variance, stddev)`. The `observed_max + 3σ` headroom term
/// the caller emits is built from `stddev` here.
fn mean_variance_stddev(samples: &[f64]) -> (f64, f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = sum_f64(samples) / (n as f64);
    let squared_devs: Vec<f64> = samples.iter().map(|&s| (s - mean) * (s - mean)).collect();
    let variance = sum_f64(&squared_devs) / (n as f64);
    let stddev = variance.sqrt();
    (mean, variance, stddev)
}

/// The device's ADVERTISED f64-atomic-add capability and the consistent
/// [`AtomicFinalizePath`] it implies. The WR-02 path-consistency guard asserts the
/// path the reduce loop RETURNED equals this — a silent atomic→host-sum mode switch
/// (or a skipped GPU test counted as passed) surfaces as a FAILURE, so a 0.0
/// divergence cannot pass as fake validation (T-07.6-03). Transcribed VERBATIM from
/// `reduce.rs:315-328`.
fn expected_atomic_finalize_path() -> AtomicFinalizePath {
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let advertises_atomic = {
        let ty = <cubecl::prelude::Atomic<f64> as CubePrimitive>::as_type_native_unchecked();
        client
            .properties()
            .atomic_type_usage(ty)
            .contains(cubecl::features::AtomicUsage::Add)
    };
    if advertises_atomic {
        AtomicFinalizePath::InKernelAtomicF64
    } else {
        AtomicFinalizePath::HostSumFallback
    }
}

/// True when a GPU backend feature is active, i.e. `SelectedRuntime` resolves to a
/// real device runtime. The WR-01 anti-false-pass guard for the tests that have no
/// `AtomicFinalizePath` return to assert against (Tests A and C): the module is only
/// meaningful on a GPU backend — under the default `cpu` feature every
/// `max_divergence` collapses to a CPU-vs-CPU comparison that is bit-exact (0.0) and
/// would silently PASS while measuring nothing (the fake-validation outcome the
/// module exists to prevent, T-07.6-01). Tests A/C call this and SKIP (early-return
/// with a printed notice) under `cpu` rather than emitting a fake `[GPU-06 EVIDENCE]`
/// line — a skip stays green under `cargo test --workspace` (default `cpu`) without
/// claiming a measurement that never ran. The check is on the active backend feature
/// (not the device-advertised atomic capability) so it stays correct on gfx1100,
/// where `HostSumFallback` is the LEGITIMATE GPU reduce path.
fn gpu_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda", feature = "wgpu"))
}

// ===========================================================================
// Inline CPU references (TRANSCRIBED — never import cb-train, T-07.6-02). Each is
// the FROZEN ordered-host reference for one kernel family, folded through
// `cb_core::sum_f64` (D-08). These are the SAME shapes the sibling oracles use.
// ===========================================================================

/// The RMSE der1 CPU baseline (`target - approx`) — `cb_compute::rmse_der1`
/// elementwise in the frozen host order (transcribed from `gradient_gpu.rs:54-60`).
fn rmse_der1_baseline(approx: &[f64], target: &[f64]) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::rmse_der1(a, t))
        .collect()
}

/// The Logloss der2 CPU baseline (`der2 = -p*(1-p)`, `p = sigmoid(approx)`) —
/// `cb_compute::logloss_der2` elementwise in the frozen host order. The hessian does
/// NOT depend on `target` (the signature's `_target` is unused), matching the
/// device `launch_der_unary(approx, LoglossHessian)` single-input seam; pass `0.0`.
fn logloss_der2_baseline(approx: &[f64]) -> Vec<f64> {
    approx
        .iter()
        .map(|&a| cb_compute::logloss_der2(a, 0.0))
        .collect()
}

/// The ORDERED host-reference 2-channel pointwise histogram, transcribed from
/// `pointwise_hist.rs:105-152`: per (feature, bin) gather der1 / weight in ascending
/// object-visiting order, fold through `sum_f64`, flat layout
/// `(feature * n_bins + bin) * 2 + channel`.
fn host_reference_hist2(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> Vec<f64> {
    let n = der1.len();
    let mut delta_members: Vec<Vec<f64>> = vec![Vec::new(); n_features * n_bins];
    let mut weight_members: Vec<Vec<f64>> = vec![Vec::new(); n_features * n_bins];
    for feature in 0..n_features {
        for &obj in indices.iter() {
            let obj = obj as usize;
            let bin = cindex[feature * n + obj] as usize;
            assert!(
                bin < n_bins,
                "host_reference_hist2 requires in-range bins: bin {bin} >= n_bins {n_bins}"
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
            out[base] = sum_f64(&delta_members[cell]);
            out[base + 1] = sum_f64(&weight_members[cell]);
        }
    }
    out
}

/// The `(b1, b2, one_hot) -> (bin, histId, w)` per-pair contribution writer,
/// transcribed VERBATIM from `pairwise_hist.rs:122-151`: `histId = 2*isGe +
/// isSecondBin` over the (ge, gt) flag collapse.
fn add_pair_contrib<P: FnMut(usize, usize, f64)>(
    b1: usize,
    b2: usize,
    w: f64,
    one_hot: bool,
    mut push_cell: P,
) {
    if one_hot {
        let is_ge = usize::from(b1 != b2);
        push_cell(b1, 2 * is_ge, w);
        push_cell(b1, 2 * is_ge, w);
        push_cell(b2, 2 * is_ge + 1, w);
        push_cell(b2, 2 * is_ge + 1, w);
    } else {
        let ge = usize::from(b1 >= b2);
        let gt = usize::from(b1 > b2);
        push_cell(b1, 2 * ge, w);
        push_cell(b1, 2 * gt, w);
        push_cell(b2, 2 * ge + 1, w);
        push_cell(b2, 2 * gt + 1, w);
    }
}

/// The ORDERED host-reference 4-channel WEIGHT-ONLY pairwise histogram, transcribed
/// from `pairwise_hist.rs:170-214`: per pair, per feature, derive the four
/// `(bin, histId)` writes, gather per-cell in ascending pair order, fold through
/// `sum_f64`, flat layout `(feature * n_bins + bin) * 4 + hist_id`.
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
    let cells = n_features * n_bins * 4;
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); cells];
    for feature in 0..n_features {
        for p in 0..n_pairs {
            let oi = pair_i[p] as usize;
            let oj = pair_j[p] as usize;
            let w = pair_weight[p];
            let b1 = cindex[feature * n_objects + oi] as usize;
            let b2 = cindex[feature * n_objects + oj] as usize;
            assert!(
                b1 < n_bins && b2 < n_bins,
                "host_reference_pairwise_hist requires in-range bins: b1={b1} b2={b2} n_bins={n_bins}"
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

/// The ORDERED host-reference per-(feature, bin) L2 split SCORE, transcribed from
/// `score_split.rs:167-207`: LEFT leaf = bins `<= border`, RIGHT leaf = bins
/// `> border`, each side folded through `sum_f64` in ascending object order, scored
/// via the FROZEN `cb_compute::l2_split_score`. Flat layout `feature * n_bins + bin`.
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
            let left = cb_compute::LeafStats {
                sum_weighted_delta: sum_f64(&left_der),
                sum_weight: sum_f64(&left_w),
            };
            let right = cb_compute::LeafStats {
                sum_weighted_delta: sum_f64(&right_der),
                sum_weight: sum_f64(&right_w),
            };
            scores[feature * n_bins + border] =
                cb_compute::l2_split_score(&[left, right], scaled_l2);
        }
    }
    scores
}

// ===========================================================================
// Shared fixtures (the `score_split` / `grow_loop` fixture shape, transcribed inline
// so this module never depends on a sibling test module's privates).
// ===========================================================================

/// The clear-gain-margin grow-loop / score fixture (`grow_loop.rs:95-115`): feature 0
/// climbs monotonically with the object index (clear best border), other features get
/// a deterministic lower-gain spread. Returns `(der1, weight, cindex, indices)`
/// feature-major.
fn make_fixture(n: usize, n_features: usize, n_bins: usize) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
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

/// The pairwise fixture (`pairwise_hist.rs:223-249`): deterministic pair endpoints
/// (including equal-bin pairs) and non-trivial per-pair weights. Returns
/// `(pair_i, pair_j, pair_weight, cindex)`.
fn make_pair_fixture(
    n_objects: usize,
    n_features: usize,
    n_bins: usize,
    n_pairs: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<u32>) {
    let mut pair_i = vec![0u32; n_pairs];
    let mut pair_j = vec![0u32; n_pairs];
    let mut pair_weight = vec![0.0_f64; n_pairs];
    for p in 0..n_pairs {
        let oi = if n_objects == 0 { 0 } else { (p * 3 + 1) % n_objects };
        let oj = if n_objects == 0 { 0 } else { (p * 7 + 2) % n_objects };
        pair_i[p] = oi as u32;
        pair_j[p] = oj as u32;
        pair_weight[p] = 0.5 + ((p % 11) as f64) * 0.25;
    }
    let mut cindex = vec![0u32; n_features * n_objects];
    for feature in 0..n_features {
        for obj in 0..n_objects {
            let bin = if n_bins == 0 {
                0
            } else {
                ((obj * (feature + 1) + feature) % n_bins) as u32
            };
            cindex[feature * n_objects + obj] = bin;
        }
    }
    (pair_i, pair_j, pair_weight, cindex)
}

/// The FORWARD-bit leaf index (`grow_loop.rs:77-85`): split `i` → bit `i`. The
/// parity-critical convention `partition_split_kernel` replicates (Pitfall 6).
fn cpu_leaf_index(passes: &[bool]) -> usize {
    let mut idx = 0usize;
    for (i, &p) in passes.iter().enumerate() {
        if p {
            idx |= 1usize << i;
        }
    }
    idx
}

/// The inline CPU L2 stump score of ONE `(feature, bin)` split (`grow_loop.rs:125-158`),
/// forward-bit partition `cindex > bin`, each side folded through `sum_f64` (D-08).
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
    cb_compute::l2_split_score(&[left, right], scaled_l2)
}

/// The inline CPU greedy LEVEL-0 search — strict-first-wins L2 argmax in ascending
/// `(feature, bin)` order (`grow_loop.rs:168-190`, STRICT `>` tie-break — the
/// load-bearing first-wins / lowest-index rule, Pitfall 1). TRANSCRIBED inline (never
/// `cb-train`). Returns the chosen `(feature, bin)` or `None`.
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
            if score > best_score {
                best_score = score;
                best = Some((feature, bin));
            }
        }
    }
    best
}

// ===========================================================================
// Task 1 — Test A: the per-family aggregation evidence roll-up.
// ===========================================================================

/// Re-run the existing per-kernel-family divergence comparisons over
/// `SelectedRuntime` and emit exactly ONE `[GPU-06 EVIDENCE]` line per family
/// (der_hess, pointwise_hist, pairwise_hist, score_split). Each family's abs/rel is
/// asserted ≤ the run-stable BOUND (NOT the GPU-06 epsilon). The aggregated numbers
/// feed Plan 02's epsilon proposal.
#[test]
fn gpu06_per_family_aggregation_reports_evidence() {
    // WR-01: under the cpu backend this would be a CPU-vs-CPU false-pass — SKIP
    // (not a silent fake measurement, not a panic that reddens `cargo test --workspace`).
    if !gpu_backend_active() {
        eprintln!(
            "[GPU-06] SKIP gpu06_per_family_aggregation: cpu backend active — \
             measurement requires a GPU feature (--no-default-features --features rocm)."
        );
        return;
    }
    let n_features = 2usize;
    let n_bins = 32usize;
    let l2 = 3.0_f64;

    // --- family = der_hess: RMSE der1 + Logloss der2 over SelectedRuntime ---
    {
        let n = 1000usize;
        let approx: Vec<f64> = (0..n).map(|k| (k as f64) * 0.001 - 5.0).collect();
        let target: Vec<f64> = (0..n).map(|k| (k as f64) * 0.002 + 0.5).collect();

        let dev_der1 = launch_der_binary(&approx, &target, DerBinaryKernel::RmseGradient).unwrap();
        let base_der1 = rmse_der1_baseline(&approx, &target);
        let (abs1, rel1) = max_divergence(&dev_der1, &base_der1);

        let dev_der2 = launch_der_unary(&approx, DerUnaryKernel::LoglossHessian).unwrap();
        let base_der2 = logloss_der2_baseline(&approx);
        let (abs2, rel2) = max_divergence(&dev_der2, &base_der2);

        let observed_max_abs = abs1.max(abs2);
        let observed_max_rel = rel1.max(rel2);
        println!(
            "[GPU-06 EVIDENCE] family=der_hess channel=f64(rocm/gfx1100) \
             observed_max_abs={observed_max_abs:.3e} observed_max_rel={observed_max_rel:.3e} \
             stddev=n/a(single-shot) observed_max_plus_3sigma={observed_max_abs:.3e} \
             AtomicFinalizePath=n/a(elementwise)"
        );
        assert!(
            observed_max_rel <= TOL_BOUND_F64 || observed_max_abs <= TOL_BOUND_F64,
            "der_hess diverged beyond the run-stable bound: abs={observed_max_abs:.3e} rel={observed_max_rel:.3e}"
        );
    }

    // --- family = pointwise_hist: 2-channel histogram over SelectedRuntime ---
    {
        let n = 1000usize;
        let (der1, weight, cindex, indices) = make_fixture(n, n_features, n_bins);
        let (dev_hist, path) =
            launch_pointwise_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features).unwrap();
        let base_hist = host_reference_hist2(&der1, &weight, &cindex, &indices, n_bins, n_features);
        let (abs, rel) = max_divergence(&dev_hist, &base_hist);
        println!(
            "[GPU-06 EVIDENCE] family=pointwise_hist channel=f64(rocm/gfx1100) \
             observed_max_abs={abs:.3e} observed_max_rel={rel:.3e} stddev=n/a(single-shot) \
             observed_max_plus_3sigma={abs:.3e} AtomicFinalizePath={path:?}"
        );
        assert!(
            rel <= TOL_BOUND_F64 || abs <= TOL_BOUND_F64,
            "pointwise_hist diverged beyond the run-stable bound: abs={abs:.3e} rel={rel:.3e}"
        );
    }

    // --- family = pairwise_hist: 4-channel weight-only histogram over SelectedRuntime ---
    {
        let n_objects = 1000usize;
        let n_pairs = 4000usize;
        let bits = 5u32; // 32-bin 5-bit feature group
        let (pair_i, pair_j, pair_weight, cindex) =
            make_pair_fixture(n_objects, n_features, n_bins, n_pairs);
        let dev_hist = launch_pairwise_hist(
            &pair_i, &pair_j, &pair_weight, &cindex, n_objects, n_bins, n_features, bits, false,
        )
        .unwrap();
        let base_hist = host_reference_pairwise_hist(
            &pair_i, &pair_j, &pair_weight, &cindex, n_objects, n_bins, n_features, false,
        );
        let (abs, rel) = max_divergence(&dev_hist, &base_hist);
        println!(
            "[GPU-06 EVIDENCE] family=pairwise_hist channel=f64(rocm/gfx1100) \
             observed_max_abs={abs:.3e} observed_max_rel={rel:.3e} stddev=n/a(single-shot) \
             observed_max_plus_3sigma={abs:.3e} AtomicFinalizePath=in-kernel(global)"
        );
        assert!(
            rel <= TOL_BOUND_F64 || abs <= TOL_BOUND_F64,
            "pairwise_hist diverged beyond the run-stable bound: abs={abs:.3e} rel={rel:.3e}"
        );
    }

    // --- family = score_split: per-candidate L2 split score over SelectedRuntime ---
    {
        let n = 1000usize;
        let (der1, weight, cindex, indices) = make_fixture(n, n_features, n_bins);
        let total_w = sum_f64(&weight);
        let scaled_l2 = cb_compute::scale_l2_reg(l2, total_w, n);
        let (_best, dev_scores) = launch_find_optimal_split_pointwise(
            &der1, &weight, &cindex, &indices, n_bins, n_features, scaled_l2,
            crate::kernels::SCORE_FN_L2,
        )
        .unwrap();
        let base_scores =
            host_reference_scores(&der1, &weight, &cindex, &indices, n_bins, n_features, scaled_l2);
        let (abs, rel) = max_divergence(&dev_scores, &base_scores);
        println!(
            "[GPU-06 EVIDENCE] family=score_split channel=f64(rocm/gfx1100) \
             observed_max_abs={abs:.3e} observed_max_rel={rel:.3e} stddev=n/a(single-shot) \
             observed_max_plus_3sigma={abs:.3e} AtomicFinalizePath=n/a(device-resident-score)"
        );
        assert!(
            rel <= TOL_BOUND_F64 || abs <= TOL_BOUND_F64,
            "score_split diverged beyond the run-stable bound: abs={abs:.3e} rel={rel:.3e}"
        );
    }

    // The f32 bound is referenced so the wgpu channel's looser magnitude is documented
    // alongside the f64 channel (the evidence consumer reads both — REPORT-not-epsilon).
    let _ = TOL_BOUND_F32;
}

// ===========================================================================
// Task 1 — Test B: the N≥30 variance loop with stddev + observed_max+3σ headroom.
// ===========================================================================

/// The `reduce.rs` runs=32 atomic-finalize loop, EXTENDED per D-7.6-02: after the
/// min/max/spread/max_abs block, compute mean/variance/stddev through
/// `cb_core::sum_f64` and the `observed_max + 3σ` headroom term, then emit the
/// `[GPU-06 EVIDENCE] family=reduce ...` line. The WR-02 path-consistency guard is
/// PRESERVED VERBATIM — a silent atomic→host-sum substitution (or a skipped GPU test
/// counted as passed) FAILS rather than passing as a fake 0.0 divergence (T-07.6-03).
#[test]
fn gpu06_variance_with_stddev_reports_evidence() {
    // Multi-cube input (300 elements -> ~10 cubes at CUBE_DIM 32) so several cubes
    // race to fetch_add into the single accumulator — the cross-cube non-determinism
    // setup (reduce.rs:262-265).
    let input: Vec<f64> = (0..300)
        .map(|k| ((k % 23) as f64) - 11.0 + 0.125 * (k as f64))
        .collect();
    let baseline = sum_f64(&input);

    let runs = VARIANCE_RUNS; // N≥30 floor (D-7.6-02)
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

    // D-7.6-02 EXTENSION: stddev of the device sums + the observed_max+3σ headroom
    // term that feeds the epsilon proposal. Folded through `sum_f64` (D-08), NOT
    // `std::iter::sum`.
    let (mean, variance, stddev) = mean_variance_stddev(&sums);
    let headroom_input = max_abs + 3.0 * stddev;

    println!(
        "[GPU-06 EVIDENCE] family=reduce channel=f64(rocm/gfx1100) runs={runs} baseline={baseline} \
         mean={mean} variance={variance:.3e} stddev={stddev:.3e} \
         observed_max_abs={max_abs:.3e} observed_max_rel={max_rel:.3e} \
         run_to_run_spread={variance_spread:.3e} observed_max_plus_3sigma={headroom_input:.3e} \
         AtomicFinalizePath={path:?}"
    );
    println!(
        "[GPU-06 EVIDENCE] family=reduce NOTE: observed_max_plus_3sigma is the EMPIRICAL \
         headroom INPUT to the Plan-02 epsilon proposal, NOT the epsilon."
    );

    // WR-02 path-consistency guard (reduce.rs:308-334), PRESERVED VERBATIM: the
    // returned path must match the device's advertised f64-atomic capability so a
    // silent atomic→deterministic-host-sum mode switch surfaces as a FAILURE. On
    // gfx1100 HIP does not advertise f64 atomic-add, so HostSumFallback is the
    // consistent (expected) path. A 0.0 divergence cannot pass as fake validation.
    let expected_path = expected_atomic_finalize_path();
    assert_eq!(
        path, expected_path,
        "launch_block_reduce_atomic_f64 returned {path:?} but the device-advertised path is \
         {expected_path:?} — a silent atomic→deterministic mode switch (or a skipped GPU test \
         counted as passed) would otherwise pass unnoticed (WR-02 / T-07.6-03)"
    );

    // The atomic finalize must still land on the CPU baseline within the generous,
    // run-stable bound (catches a wrong fold without pinning the GPU-06 epsilon).
    assert!(
        max_rel <= TOL_BOUND_F64 || max_abs <= TOL_BOUND_F64,
        "atomic-finalize reduce diverged too far from baseline: abs={max_abs:.3e} rel={max_rel:.3e}"
    );
}

// ===========================================================================
// Task 2 — Test C: end-to-end GPU-vs-CPU leaf-value measurement (structure EXACT,
// leaf values REPORTED into the evidence roll-up).
// ===========================================================================

/// The run-stable leaf-VALUE divergence bound, the SAME channel-driven value as
/// `grow_loop::LEAF_BOUND` (f32 ~1e-3 on wgpu, f64 ~1e-9 elsewhere): REPORTED, NOT the
/// GPU-06 epsilon. Aliased to [`TOL_BOUND_F64`] so the module edits ONE place.
const LEAF_BOUND: f64 = TOL_BOUND_F64;

/// Drive `grow_oblivious_tree` over `SelectedRuntime` on the LARGEST-N multi-cube
/// fixture (n=10000 — maximizing cross-cube atomic contention for the variance/headroom
/// story, D-7.6-01), assert the tree STRUCTURE matches the inline CPU greedy
/// first-wins search EXACTLY (split `(feature, bin)` sequence + per-object `leaf_of` ==
/// `leaf_index`), then REPORT the leaf-value `(abs, rel)` divergence vs
/// `cb_compute::calc_average` over the SAME partition. These are the numbers Phase 7.5
/// left REPORTED-not-signed-off (07.5-03/04/06) — the fresh measurement target for the
/// gate. Modeled on `grow_loop.rs:467-549` (`matches_cpu_greedy_search`); STRUCTURE is
/// the STRICT bar, leaf VALUES are REPORTED (the GPU-06 epsilon is Plan 02's job).
#[test]
fn gpu06_end_to_end_leaf_values_report_evidence() {
    // WR-01: under the cpu backend this would be a CPU-vs-CPU false-pass — SKIP
    // (not a silent fake measurement, not a panic that reddens `cargo test --workspace`).
    if !gpu_backend_active() {
        eprintln!(
            "[GPU-06] SKIP gpu06_end_to_end_leaf_values: cpu backend active — \
             measurement requires a GPU feature (--no-default-features --features rocm)."
        );
        return;
    }
    let n_features = 3usize;
    let n_bins = 32usize;
    let depth = 1usize; // the MVP vertical slice (the strict O(1)-per-level device path)
    let l2 = 3.0_f64;

    // Prefer the largest-N multi-cube fixture (n=10000 >> CUBE_DIM=32 → many cubes race
    // into the cross-cube finalize, the atomic-contention setup). n=1000 is included so
    // the evidence covers more than one scale.
    for &n in &[1000usize, 10_000usize] {
        let (der1, weight, cindex, indices) = make_fixture(n, n_features, n_bins);

        // The per-tree L2 scaling — cb_compute::scale_l2_reg(l2, Σweight, n) — the FROZEN
        // scaling the CPU oracle and the device leaf-value step both consume.
        let sum_w = sum_f64(&weight);
        let scaled_l2 = cb_compute::scale_l2_reg(l2, sum_w, n);

        // Device: grow the tree host-light over SelectedRuntime (the rocm arm in-env).
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

        // (C) LEAF VALUES — REPORTED divergence vs cb_compute::calc_average over the SAME
        //     leaf partition (NOT signed off — Plan 02 owns the epsilon).
        let n_leaves = 1usize << depth;
        let mut cpu_leaf_values = vec![0.0_f64; n_leaves];
        for (leaf, slot) in cpu_leaf_values.iter_mut().enumerate() {
            let mut der_seg: Vec<f64> = Vec::new();
            let mut w_seg: Vec<f64> = Vec::new();
            for obj in 0..n {
                if cpu_leaf_of[obj] as usize == leaf {
                    der_seg.push(der1[obj]);
                    w_seg.push(weight[obj]);
                }
            }
            *slot = cb_compute::calc_average(sum_f64(&der_seg), sum_f64(&w_seg), scaled_l2);
        }
        assert_eq!(
            tree.leaf_values.len(),
            cpu_leaf_values.len(),
            "device leaf_values length must equal n_leaves (n={n})"
        );
        let (abs, rel) = max_divergence(&tree.leaf_values, &cpu_leaf_values);
        println!(
            "[GPU-06 EVIDENCE] family=end_to_end_leaf_values channel=f64(rocm/gfx1100) n={n} \
             split={cpu_split:?} structure=EXACT observed_max_abs={abs:.3e} \
             observed_max_rel={rel:.3e} observed_max_plus_3sigma={abs:.3e} \
             AtomicFinalizePath=device-resident-grow-loop (bound={LEAF_BOUND:.0e})"
        );
        assert!(
            rel <= LEAF_BOUND || abs <= LEAF_BOUND,
            "device leaf values (n={n}) diverged beyond the REPORTED bound: \
             abs={abs:.3e} rel={rel:.3e} (bound={LEAF_BOUND:.0e})"
        );
    }
}
