//! GPUT-02/03 residency cross-oracle for [`crate::gpu_runtime::GpuTrainSession`]: the
//! session uploads the quantized matrix + weights + indices ONCE at `begin` and reuses the
//! resident handles across `grow_one` calls (the running approx updated ON DEVICE via
//! `apply_leaf_delta`, the residual `der1` chained device-resident — no per-tree re-upload,
//! no der1 read-back). The device-resident depth-1 Plain-boosting sequence (Cosine default
//! score, GPUT-08) must match a CPU multi-tree Cosine boosting reference EXACTLY in structure
//! (every tree's split `(feature, bin)` + per-object `leaf_of`), with the leaf VALUES
//! REPORTED within a generous run-stable bound (the GPU-06 epsilon is 7.6's job).
//!
//! Source/test separation (CLAUDE.md / AGENTS.md): the session is production code
//! (`gpu_runtime/session.rs`); ALL `#[test]` + `.unwrap()`/indexing live here. The
//! TREE-STRUCTURE oracle (greedy first-wins search + forward-bit `leaf_index`) is
//! TRANSCRIBED VERBATIM here — importing `cb-train` would pull its `cb-backend` default-`cpu`
//! dep into the test build graph and break `SelectedRuntime` under rocm (the landmine). The
//! read-only SCORE oracle (`cosine_split_score`) + leaf formula (`calc_average`) come from
//! `cb_compute` (already a dep). Runs on rocm in-env on gfx1100 (wave32).

#![cfg(not(feature = "wgpu"))]

use cb_compute::{DeviceTrainConfig, EScoreFunction, Loss};
use cb_core::sum_f64;

use crate::gpu_runtime::GpuTrainSession;

/// Generous run-stable leaf-value bound (NOT the signed-off epsilon).
const LEAF_BOUND: f64 = 1e-3;

// Fixture builders — byte-identical to `kernels::test_fixtures` (private to `kernels`; the
// depth-1 clear-gain-margin boosting fixture is transcribed here for the gpu_runtime tree).

/// Centred ramp `k - n/2` (the boosting target, monotone in the object index).
fn ramp_centred(n: usize) -> Vec<f64> {
    (0..n).map(|k| (k as f64) - (n as f64) / 2.0).collect()
}

/// Non-trivial per-object weight `0.5 + (k % 5) * 0.25` (never all-1).
fn weight_mod5(n: usize) -> Vec<f64> {
    (0..n).map(|k| 0.5 + ((k % 5) as f64) * 0.25).collect()
}

/// Feature-major quantized cindex: feature 0's bins climb monotonically with the object index
/// (a clear best border), other features get a deterministic lower-gain spread.
fn cindex_feature_major(n: usize, n_features: usize, n_bins: usize) -> Vec<u32> {
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
    cindex
}

/// Forward-bit leaf index over a per-level pass sequence (`idx |= 1 << i` for pass `i` —
/// `cb_train::leaf_index`, transcribed). Depth-1 has one pass.
fn cpu_leaf_index(passes: &[bool]) -> usize {
    let mut idx = 0usize;
    for (i, &pass) in passes.iter().enumerate() {
        if pass {
            idx |= 1usize << i;
        }
    }
    idx
}

/// The whole-dataset Cosine score of ONE binary split `(feature, bin)` (the depth-1 stump
/// score under the catboost DEFAULT score fn), TRANSCRIBED from the FROZEN
/// `cb_compute::cosine_split_score` over the forward-bit left/right partition
/// (`cindex[feature * n + obj] > bin`, == the device `partition_split`). Each side's
/// Σ der1 / Σ weight is folded in ascending object order via `sum_f64` (D-08).
fn cpu_stump_score_cosine(
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
    cb_compute::cosine_split_score(&[left, right], scaled_l2)
}

/// Greedy depth-1 Cosine stump search (strict first-wins over ascending `(feature, bin)`,
/// enumerating only the `0..n_bins-1` real borders — lockstep with the device argmin guard).
fn cpu_best_stump_cosine(
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
        let last_real = n_bins.saturating_sub(1);
        for bin in 0..last_real {
            let score = cpu_stump_score_cosine(der1, weight, cindex, n, feature, bin, scaled_l2);
            if score > best_score {
                best_score = score;
                best = Some((feature, bin));
            }
        }
    }
    best
}

/// The CPU multi-tree Plain-boosting reference under the Cosine split score, mirroring the
/// device session EXACTLY: RMSE residual `der1 = target - approx`, Cosine greedy stump,
/// forward-bit `leaf_of`, UNSCALED `calc_average` leaf delta (the 10-02 `DeviceGrownTree`
/// contract), and the running approx update `approx[i] += lr * delta[leaf(i)]`. Returns the
/// per-tree `(split, leaf_of, unscaled_leaf_values)`.
#[allow(clippy::type_complexity)]
fn cpu_multi_tree_cosine(
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
        let der1: Vec<f64> = (0..n).map(|i| cb_compute::rmse_der1(approx[i], target[i])).collect();
        let (feature, bin) = cpu_best_stump_cosine(&der1, weight, cindex, n, n_features, n_bins, scaled_l2)
            .expect("CPU reference must find a candidate split each iteration");

        let leaf_of: Vec<u32> = (0..n)
            .map(|obj| {
                let passes = [(cindex[feature * n + obj] as usize) > bin];
                cpu_leaf_index(&passes) as u32
            })
            .collect();

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
            // UNSCALED delta (the DeviceGrownTree contract); approx applies lr below.
            leaf_values[leaf] = cb_compute::calc_average(sum_f64(&der_seg), sum_f64(&w_seg), scaled_l2);
        }

        for obj in 0..n {
            approx[obj] += learning_rate * leaf_values[leaf_of[obj] as usize];
        }

        out.push(((feature, bin), leaf_of, leaf_values));
    }
    out
}

/// Max abs/rel divergence (informational).
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    let mut abs = 0.0_f64;
    let mut rel = 0.0_f64;
    for (&d, &b) in device.iter().zip(baseline.iter()) {
        let a = (d - b).abs();
        abs = abs.max(a);
        let denom = b.abs().max(1e-12);
        rel = rel.max(a / denom);
    }
    (abs, rel)
}

/// The residency proof: `begin` once, then `grow_one` several times reusing the resident
/// handles (der1 chained ON DEVICE, approx updated ON DEVICE via `apply_leaf_delta` — no
/// per-tree matrix re-upload, no der1 read-back). EVERY tree's structure must match the CPU
/// multi-tree Cosine reference EXACTLY; leaf values REPORTED within the generous bound.
#[test]
fn session_residency_matches_cpu_multi_tree_boosting() {
    // WR-01 anti-false-pass convention (Phase 7.6 / matches the grow_loop depth6 tests): the
    // resident grow routes through the fixed-point `Atomic<u64>` partition histogram, which
    // cpu/wgpu do not advertise — SKIP (not panic) on those backends so a default-`cpu`
    // `cargo test` run does not fail on the device-only grow. Runs on rocm/cuda in-env.
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!(
            "[10-07] SKIP session_residency_matches_cpu_multi_tree_boosting: active backend lacks \
             Atomic<u64> add (cpu/wgpu) — the resident partition histogram path needs rocm/cuda"
        );
        return;
    }

    let n_features = 3usize;
    let n_bins = 32usize;
    let l2 = 3.0_f64;
    let iterations = 5usize;
    let learning_rate = 0.3_f64;

    for &n in &[1usize, 37usize, 1000usize] {
        let target = ramp_centred(n);
        let weight = weight_mod5(n);
        let cindex = cindex_feature_major(n, n_features, n_bins);
        let scaled_l2 = cb_compute::scale_l2_reg(l2, sum_f64(&weight), n);

        // Open the session ONCE (uploads the matrix + weights + indices once). Cosine default.
        let mut session = GpuTrainSession::begin(
            &Loss::Rmse,
            1, // depth
            true, // Plain
            1, // fold_count
            EScoreFunction::Cosine,
            &cindex,
            &weight,
            n,
            n_features,
            n_bins,
            learning_rate,
            scaled_l2,
            &DeviceTrainConfig::default(),
        )
        .expect("session begin must not error on a covered config")
        .expect("covered config (depth1/RMSE/Plain/fold1/Cosine) must open a session");

        assert_eq!(session.n(), n, "session n must equal the fixture n (n={n})");

        // Grow `iterations` trees reusing the resident handles (NO re-upload per call).
        let mut device_trees = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            // Oblivious resident session: `approx` is ignored (the resident approx is
            // authoritative), so a zero vector satisfies the length-agreement guard.
            let tree = session
                .grow_one(&vec![0.0_f64; n], &target)
                .expect("grow_one must succeed on the clear-margin fixture");
            device_trees.push(tree);
        }

        let cpu = cpu_multi_tree_cosine(
            &target, &weight, &cindex, n, n_features, n_bins, iterations, learning_rate, scaled_l2,
        );

        assert_eq!(device_trees.len(), iterations);
        assert_eq!(cpu.len(), iterations);

        let mut run_abs = 0.0_f64;
        let mut run_rel = 0.0_f64;
        for (k, (dev, (cpu_split, cpu_leaf_of, cpu_leaf_values))) in
            device_trees.iter().zip(cpu.iter()).enumerate()
        {
            assert_eq!(dev.splits.len(), 1, "device tree {k} must be a depth-1 stump (n={n})");
            let (df, db) = dev.splits[0];
            assert_eq!(
                (df as usize, db as usize),
                *cpu_split,
                "device tree {k} split must match CPU Cosine first-wins (n={n}): \
                 device=({df},{db}) cpu={cpu_split:?}"
            );
            assert_eq!(
                &dev.leaf_of, cpu_leaf_of,
                "device tree {k} leaf_of must equal CPU forward-bit leaf_index (n={n})"
            );
            assert_eq!(
                dev.leaf_values.len(),
                cpu_leaf_values.len(),
                "device tree {k} leaf_values length must equal n_leaves (n={n})"
            );
            let (abs, rel) = max_divergence(&dev.leaf_values, cpu_leaf_values);
            run_abs = run_abs.max(abs);
            run_rel = run_rel.max(rel);
            println!(
                "[session_residency n={n} tree={k}] STRUCTURE match split={cpu_split:?}; \
                 REPORTED leaf-value abs={abs:.3e} rel={rel:.3e}"
            );
        }
        assert!(
            run_rel <= LEAF_BOUND || run_abs <= LEAF_BOUND,
            "device session leaf values (n={n}) diverged beyond the REPORTED bound: \
             abs={run_abs:.3e} rel={run_rel:.3e} (bound={LEAF_BOUND:.0e})"
        );

        // Sanity: the residual must actually decrease across iterations on this fixture — a
        // proof the resident der1 is chained (not re-derived from a stale zero approx). The
        // tree-0 and tree-4 splits agreeing with the CPU chained reference already implies
        // this; assert the reference itself moved (structure/values evolved).
        if n > 1 {
            let (_s0, _l0, v0) = &cpu[0];
            let (_s4, _l4, v4) = &cpu[iterations - 1];
            assert!(
                max_divergence(v0, v4).0 > 0.0,
                "residual should evolve across iterations (chained der1) (n={n})"
            );
        }

        // The session (client + resident handles) frees deterministically on drop here.
        drop(session);
    }
}

/// The coverage gate (D-10-02): `begin` returns `None` (→ CPU fallback) for depth==0 /
/// non-RMSE-Logloss / non-Plain / fold_count>1 / unsupported score fn, and `Some` for the
/// covered depth>=1 RMSE/Logloss Plain fold-1 config (Phase 12 Plan 01: depth>1 now covered).
#[test]
fn session_residency_coverage_gate_declines_uncovered() {
    let n = 37usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let weight = weight_mod5(n);
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);
    let lr = 0.3_f64;

    let cfg = DeviceTrainConfig::default();
    let open = |loss: &Loss, depth: usize, plain: bool, folds: usize, sf: EScoreFunction| {
        GpuTrainSession::begin(
            loss, depth, plain, folds, sf, &cindex, &weight, n, n_features, n_bins, lr, scaled_l2,
            &cfg,
        )
        .expect("begin must not error while classifying coverage")
        .is_some()
    };

    // Covered: depth1 / RMSE / Plain / fold1 / Cosine.
    assert!(open(&Loss::Rmse, 1, true, 1, EScoreFunction::Cosine), "RMSE depth1 must be covered");
    assert!(open(&Loss::Logloss, 1, true, 1, EScoreFunction::Cosine), "Logloss depth1 must be covered");
    assert!(open(&Loss::CrossEntropy, 1, true, 1, EScoreFunction::L2), "CrossEntropy depth1 L2 must be covered");

    // Phase 12 Plan 01 (GPUT-18, A3 gap): depth>1 is now DEVICE-COVERED (Phase-11 substrate);
    // the former `depth>1 must decline` assertion is INVERTED — a depth-2 Plain/fold1/RMSE
    // config now opens a session (the depth>1 grow self-oracle lives in session_depth_gt1_test).
    assert!(open(&Loss::Rmse, 2, true, 1, EScoreFunction::Cosine), "depth>1 must now be covered");

    // Uncovered → None (CPU fallback).
    assert!(!open(&Loss::Rmse, 1, false, 1, EScoreFunction::Cosine), "non-Plain (Ordered) must decline");
    assert!(!open(&Loss::Rmse, 1, true, 2, EScoreFunction::Cosine), "fold_count>1 must decline");
    assert!(!open(&Loss::Mae, 1, true, 1, EScoreFunction::Cosine), "non-RMSE/Logloss loss must decline");
}
