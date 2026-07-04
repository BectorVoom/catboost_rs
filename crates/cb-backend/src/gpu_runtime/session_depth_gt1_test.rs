//! Phase 12 Plan 01 (GPUT-18, A3 gap): the depth>1 device-grow self-oracle for
//! [`crate::gpu_runtime::GpuTrainSession`]. Task 1 relaxed the coverage gate so a depth>1
//! Plain/fold_count==1/RMSE/covered-score config now reaches the device grow through the
//! already-shipped Phase-11 partition-aware substrate (`grow_oblivious_tree_resident` loops
//! `0..depth`) instead of being force-declined to CPU. This test proves:
//!
//! 1. (structure) a depth-6 tree grown via the SESSION path (`begin` → `grow_one`) matches a
//!    DIRECT [`crate::gpu_runtime::grow_oblivious_tree`] call over the SAME first-tree residual
//!    (`der1 = rmse_der1(0, target)`) bit-for-bit on the integer `(feature, bin)` split
//!    sequence AND the per-object `leaf_of`, with the leaf VALUES within the ε=1e-4 GPU bar;
//! 2. (gate) depth>1 Plain/fold1/RMSE/Cosine now returns `Ok(Some(_))`, while every
//!    still-uncovered config (non-Plain, fold_count>1, unmapped loss, depth==0, and any
//!    non-default [`cb_compute::DeviceTrainConfig`] family flag) still returns `Ok(None)`
//!    (D-10-01 all-or-nothing → the byte-unchanged CPU grower).
//!
//! Source/test separation (CLAUDE.md / AGENTS.md): the session + grow driver are production
//! code; ALL `#[test]` + `.unwrap()`/indexing live here. The structure oracle is the DIRECT
//! device `grow_oblivious_tree` call (NOT a `cb-train` import — that would pull the landmine
//! `cb-backend` default-`cpu` dep into the test build graph and break `SelectedRuntime` under
//! rocm). The device grow needs `Atomic<u64>` (the fixed-point partition histogram) so the
//! grow arm SKIPS on cpu/wgpu (the Phase-7.6 WR-01 anti-false-pass convention); the gate arm
//! runs on every backend (it only classifies — it never grows). rocm in-env on gfx1100.

#![cfg(not(feature = "wgpu"))]

use cb_compute::{DeviceGrowPolicy, DeviceTrainConfig, EScoreFunction, Loss};
use cb_core::sum_f64;

use crate::gpu_runtime::{grow_oblivious_tree, GpuTrainSession};
use crate::kernels::SCORE_FN_COSINE;

/// The ε=1e-4 GPU bar (D-04).
const GROW_EPS: f64 = 1e-4;

// Fixture builders — byte-identical to `session_residency` (the same clear-gain-margin fixture,
// transcribed for source/test separation).

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

/// Max abs/rel divergence (informational + the ε assert).
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

/// (Structure oracle) a depth-6 tree grown through the SESSION path must match a DIRECT
/// `grow_oblivious_tree` call over the SAME first-tree residual bit-for-bit on the integer
/// splits + `leaf_of`, with the leaf values within ε=1e-4. Needs `Atomic<u64>` (the fixed-point
/// partition histogram) — SKIPS on cpu/wgpu, runs on rocm/cuda in-env.
#[test]
fn session_depth_gt1_grows_and_matches_direct() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!(
            "[12-01] SKIP session_depth_gt1_grows_and_matches_direct: active backend lacks \
             Atomic<u64> add (cpu/wgpu) — the depth>1 partition histogram path needs rocm/cuda"
        );
        return;
    }

    let n_features = 3usize;
    let n_bins = 32usize;
    let depth = 6usize;
    let l2 = 3.0_f64;
    let learning_rate = 0.3_f64;

    for &n in &[200usize, 2000usize] {
        let target = ramp_centred(n);
        let weight = weight_mod5(n);
        let cindex = cindex_feature_major(n, n_features, n_bins);
        let scaled_l2 = cb_compute::scale_l2_reg(l2, sum_f64(&weight), n);

        // Session path: open a DEPTH-6 covered session (the A3 gate relaxation), grow ONE tree.
        // The session's first `grow_one` derives `der1 = rmse_der1(0, target)` from the resident
        // zero-start approx, so the direct call below feeds that SAME residual.
        let mut session = GpuTrainSession::begin(
            &Loss::Rmse,
            depth,
            true, // Plain
            1,    // fold_count
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
        .expect("begin must not error on a covered depth-6 config")
        .expect("depth-6 Plain/fold1/RMSE/Cosine must now open a session (A3 gap closed)");

        assert_eq!(session.n(), n, "session n must equal the fixture n (n={n})");

        let dev_tree = session
            .grow_one(&vec![0.0_f64; n], &target)
            .expect("depth-6 grow_one must succeed on the clear-margin fixture");

        // Direct reference: `grow_oblivious_tree` over the SAME first-tree residual + Cosine.
        let der1: Vec<f64> = (0..n).map(|i| cb_compute::rmse_der1(0.0, target[i])).collect();
        let indices: Vec<u32> = (0..n as u32).collect();
        let direct = grow_oblivious_tree(
            &der1, &weight, &cindex, &indices, n_bins, n_features, depth, scaled_l2, SCORE_FN_COSINE,
        )
        .expect("direct depth-6 grow_oblivious_tree must succeed");

        // (A) STRUCTURE — the full 6-level integer split sequence must match EXACTLY.
        assert_eq!(
            dev_tree.splits.len(),
            depth,
            "session depth-6 tree must have exactly {depth} splits (n={n})"
        );
        assert_eq!(
            dev_tree.splits, direct.splits,
            "session depth-6 split sequence must match the direct grow bit-for-bit (n={n}): \
             session={:?} direct={:?}",
            dev_tree.splits, direct.splits
        );

        // (B) STRUCTURE — per-object `leaf_of` must match EXACTLY.
        assert_eq!(
            dev_tree.leaf_of, direct.leaf_of,
            "session depth-6 leaf_of must equal the direct grow leaf_of (n={n})"
        );

        // (C) LEAF VALUES — divergence within the ε=1e-4 GPU bar.
        assert_eq!(
            dev_tree.leaf_values.len(),
            direct.leaf_values.len(),
            "session depth-6 leaf_values length must equal 2^{depth} (n={n})"
        );
        let (abs, rel) = max_divergence(&dev_tree.leaf_values, &direct.leaf_values);
        println!(
            "[session_depth_gt1 n={n}] STRUCTURE match ({depth} splits, leaf_of exact); leaf-value \
             max abs_div={abs:.3e} rel_div={rel:.3e} (bar={GROW_EPS:.0e})"
        );
        assert!(
            abs <= GROW_EPS || rel <= GROW_EPS,
            "session depth-6 leaf values (n={n}) exceeded the ε=1e-4 bar: abs={abs:.3e} rel={rel:.3e}"
        );

        drop(session);
    }
}

/// (Gate) depth>1 Plain/fold1/RMSE/Cosine now opens a session; every still-uncovered config —
/// non-Plain, fold_count>1, unmapped loss, depth==0, and any non-default `DeviceTrainConfig`
/// family flag — still declines to `Ok(None)` (D-10-01 all-or-nothing). Runs on EVERY backend
/// (it only classifies at `begin` — it never grows), so no rocm skip.
#[test]
fn session_depth_gt1_gate_declines_uncovered() {
    let n = 37usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let weight = weight_mod5(n);
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);
    let lr = 0.3_f64;

    let open = |depth: usize,
                plain: bool,
                folds: usize,
                loss: &Loss,
                sf: EScoreFunction,
                cfg: &DeviceTrainConfig| {
        GpuTrainSession::begin(
            loss, depth, plain, folds, sf, &cindex, &weight, n, n_features, n_bins, lr, scaled_l2,
            cfg,
        )
        .expect("begin must not error while classifying coverage")
        .is_some()
    };
    let def = DeviceTrainConfig::default();

    // Covered: depth>1 Plain / fold1 / RMSE-or-Logloss / covered score (A3 gap closed).
    assert!(
        open(6, true, 1, &Loss::Rmse, EScoreFunction::Cosine, &def),
        "depth-6 Plain/fold1/RMSE/Cosine must now be covered"
    );
    assert!(
        open(2, true, 1, &Loss::Logloss, EScoreFunction::Cosine, &def),
        "depth-2 Plain/fold1/Logloss/Cosine must now be covered"
    );

    // Still-uncovered → None (byte-unchanged CPU fallback).
    assert!(
        !open(6, false, 1, &Loss::Rmse, EScoreFunction::Cosine, &def),
        "non-Plain (Ordered) must decline even at depth>1"
    );
    assert!(
        !open(6, true, 2, &Loss::Rmse, EScoreFunction::Cosine, &def),
        "fold_count>1 must decline even at depth>1"
    );
    assert!(
        !open(6, true, 1, &Loss::Mae, EScoreFunction::Cosine, &def),
        "an unmapped loss must decline even at depth>1"
    );
    assert!(
        !open(0, true, 1, &Loss::Rmse, EScoreFunction::Cosine, &def),
        "depth==0 (no tree to grow) must decline"
    );

    // Phase 12 Plan 03 (GPUT-18): the Depthwise / Lossguide grow policies are now COVERED
    // (the non-symmetric device grow arm flipped on this wave). Region stays declined (Plan 04).
    let depthwise = DeviceTrainConfig {
        grow_policy: DeviceGrowPolicy::Depthwise,
        ..DeviceTrainConfig::default()
    };
    assert!(
        open(6, true, 1, &Loss::Rmse, EScoreFunction::Cosine, &depthwise),
        "Depthwise grow_policy must now be covered (Plan 03 device non-sym arm)"
    );
    let lossguide = DeviceTrainConfig {
        grow_policy: DeviceGrowPolicy::Lossguide,
        max_leaves: Some(8),
        ..DeviceTrainConfig::default()
    };
    assert!(
        open(6, true, 1, &Loss::Rmse, EScoreFunction::Cosine, &lossguide),
        "Lossguide grow_policy (with a leaf cap) must now be covered (Plan 03)"
    );
    let region = DeviceTrainConfig {
        grow_policy: DeviceGrowPolicy::Region,
        ..DeviceTrainConfig::default()
    };
    assert!(
        !open(6, true, 1, &Loss::Rmse, EScoreFunction::Cosine, &region),
        "Region grow_policy must still decline (no device kernel until Plan 04)"
    );
    let exact = DeviceTrainConfig {
        exact_leaf: true,
        ..DeviceTrainConfig::default()
    };
    assert!(
        !open(6, true, 1, &Loss::Rmse, EScoreFunction::Cosine, &exact),
        "an exact-leaf config with a NON-quantile loss (RMSE) must decline (Plan 05: exact-leaf \
         is covered ONLY for the Quantile/MAE/MAPE family)"
    );
    assert!(
        !open(6, true, 1, &Loss::Logloss, EScoreFunction::Cosine, &exact),
        "an exact-leaf config with Logloss (non-quantile) must decline (Plan 05)"
    );
}

/// (Gate, Plan 05 GPUT-19) the EXACT-LEAF arm: an `exact_leaf` config with a covered
/// quantile-family loss (MAE / Quantile / MAPE) now opens a session (`Ok(Some)`); the Newton
/// path (exact_leaf unset) is unchanged, and exact-leaf with a non-quantile loss OR a second
/// non-default family flag still declines. Classify-only at `begin` — runs on EVERY backend.
#[test]
fn session_exact_leaf_gate_covers_quantile_family() {
    let n = 41usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let weight = weight_mod5(n);
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);
    let lr = 0.3_f64;

    let open = |loss: &Loss, cfg: &DeviceTrainConfig| {
        GpuTrainSession::begin(
            loss, 6, true, 1, EScoreFunction::Cosine, &cindex, &weight, n, n_features, n_bins, lr,
            scaled_l2, cfg,
        )
        .expect("begin must not error while classifying the exact-leaf arm")
        .is_some()
    };

    let exact = DeviceTrainConfig { exact_leaf: true, ..DeviceTrainConfig::default() };
    let exact_q = DeviceTrainConfig {
        exact_leaf: true,
        quantile_alpha: 0.25,
        quantile_delta: 1e-6,
        ..DeviceTrainConfig::default()
    };

    // COVERED: exact-leaf + the Quantile/MAE/MAPE family → Ok(Some).
    assert!(open(&Loss::Mae, &exact), "exact-leaf + MAE must open a session (Plan 05)");
    assert!(
        open(&Loss::Quantile { alpha: 0.25, delta: 1e-6 }, &exact_q),
        "exact-leaf + Quantile must open a session (Plan 05)"
    );
    assert!(open(&Loss::Mape, &exact), "exact-leaf + MAPE must open a session (Plan 05)");

    // NOT covered: exact-leaf + non-quantile loss → None.
    assert!(!open(&Loss::Rmse, &exact), "exact-leaf + RMSE must decline (non-quantile)");

    // NOT covered: exact-leaf PLUS another non-default family flag (e.g. bootstrap) → None
    // (D-10-01 all-or-nothing — only exact_leaf may differ).
    let exact_plus_bootstrap = DeviceTrainConfig {
        exact_leaf: true,
        bootstrap_type: cb_compute::DeviceBootstrapType::Bernoulli,
        ..DeviceTrainConfig::default()
    };
    assert!(
        !open(&Loss::Mae, &exact_plus_bootstrap),
        "exact-leaf + a second non-default family flag must still decline (all-or-nothing)"
    );

    // The Newton path (exact_leaf unset) is unchanged: MAE without exact_leaf still declines
    // (no device der arm), RMSE still opens.
    let def = DeviceTrainConfig::default();
    assert!(!open(&Loss::Mae, &def), "MAE without exact_leaf keeps the Newton path (declines)");
    assert!(open(&Loss::Rmse, &def), "RMSE (Newton) still opens — default path unchanged");
}

/// (End-to-end wiring, Plan 05 GPUT-19) an exact-leaf Quantile session grows a real device tree
/// whose leaf VALUES are the device Exact order statistic (finite, non-NaN). The leaf-VALUE
/// numerics are locked ≤1e-4 by the `kernels::exact_quantile` self-oracle; here we prove the
/// begin→grow_one wiring produces a valid tree end-to-end. Needs `Atomic<u64>` (the resident
/// partition histogram) + the device sort → SKIPS on cpu/wgpu, runs on rocm/cuda in-env.
#[test]
fn session_exact_leaf_grows_finite_quantile_leaves() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!(
            "[12-05] SKIP session_exact_leaf_grows_finite_quantile_leaves: active backend lacks \
             Atomic<u64> / the device sort composition (cpu/wgpu) — needs rocm/cuda"
        );
        return;
    }
    let n = 300usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let depth = 3usize;
    let target = ramp_centred(n);
    let weight = vec![1.0_f64; n]; // covered exact regime is unit weight
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);

    let exact = DeviceTrainConfig {
        exact_leaf: true,
        quantile_alpha: 0.5,
        quantile_delta: 1e-6,
        ..DeviceTrainConfig::default()
    };
    let mut session = GpuTrainSession::begin(
        &Loss::Quantile { alpha: 0.5, delta: 1e-6 },
        depth,
        true,
        1,
        EScoreFunction::Cosine,
        &cindex,
        &weight,
        n,
        n_features,
        n_bins,
        0.3,
        scaled_l2,
        &exact,
    )
    .expect("begin must not error on a covered exact-leaf Quantile config")
    .expect("exact-leaf Quantile must open a session (Plan 05 gate arm)");

    let tree = session
        .grow_one(&vec![0.0_f64; n], &target)
        .expect("exact-leaf grow_one must succeed");

    assert_eq!(tree.splits.len(), depth, "exact-leaf tree must have {depth} splits");
    assert_eq!(tree.leaf_of.len(), n, "leaf_of length must equal n");
    assert_eq!(tree.leaf_values.len(), 1usize << depth, "2^depth leaf values");
    for (l, &v) in tree.leaf_values.iter().enumerate() {
        assert!(v.is_finite(), "exact leaf value {l} must be finite, got {v}");
    }
    // At least one leaf must carry a non-zero exact quantile (the target is a non-trivial ramp,
    // so the residual median per leaf is not identically zero).
    assert!(
        tree.leaf_values.iter().any(|&v| v.abs() > 1e-9),
        "exact-leaf grow produced all-zero leaves — the quantile override did not run"
    );
    drop(session);
}

/// (Gate, Plan 06 GPUT-09) the BOOTSTRAP arm: a covered non-`No` `bootstrap_type`
/// (Bernoulli/Bayesian/Poisson) opens a session (`Ok(Some)`); MVS declines (Plan 07); the default
/// (`No`) path is unchanged; and bootstrap PLUS a second non-default family flag still declines
/// (D-10-01 all-or-nothing). Classify-only at `begin` — runs on EVERY backend.
#[test]
fn session_bootstrap_gate_covers_bernoulli_bayesian_poisson() {
    use cb_compute::DeviceBootstrapType;
    let n = 41usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let weight = weight_mod5(n);
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);

    let open = |cfg: &DeviceTrainConfig| {
        GpuTrainSession::begin(
            &Loss::Rmse, 6, true, 1, EScoreFunction::Cosine, &cindex, &weight, n, n_features,
            n_bins, 0.3, scaled_l2, cfg,
        )
        .expect("begin must not error while classifying the bootstrap arm")
        .is_some()
    };

    let bern = DeviceTrainConfig {
        bootstrap_type: DeviceBootstrapType::Bernoulli,
        sample_rate: 0.7,
        rng_seed: 17,
        ..DeviceTrainConfig::default()
    };
    let bayes = DeviceTrainConfig {
        bootstrap_type: DeviceBootstrapType::Bayesian,
        rng_seed: 42,
        ..DeviceTrainConfig::default()
    };
    let pois = DeviceTrainConfig {
        bootstrap_type: DeviceBootstrapType::Poisson,
        rng_seed: 7,
        ..DeviceTrainConfig::default()
    };
    assert!(open(&bern), "Bernoulli bootstrap must open a session (Plan 06)");
    assert!(open(&bayes), "Bayesian bootstrap must open a session (Plan 06)");
    assert!(open(&pois), "Poisson bootstrap must open a session (Plan 06)");

    // MVS declines until Plan 07.
    let mvs = DeviceTrainConfig {
        bootstrap_type: DeviceBootstrapType::Mvs,
        ..DeviceTrainConfig::default()
    };
    assert!(!open(&mvs), "MVS bootstrap must still decline (Plan 07)");

    // Bootstrap + a second non-default family flag (exact leaf) → decline (all-or-nothing).
    let bern_plus_exact = DeviceTrainConfig {
        bootstrap_type: DeviceBootstrapType::Bernoulli,
        exact_leaf: true,
        ..DeviceTrainConfig::default()
    };
    assert!(
        !open(&bern_plus_exact),
        "bootstrap + a second non-default family flag must decline (all-or-nothing)"
    );

    // The default (No bootstrap) path is unchanged.
    assert!(open(&DeviceTrainConfig::default()), "the No-bootstrap default still opens");
}

/// (Gate, Plan 08 GPUT-10) the CTR arm: a covered SINGLE-PERMUTATION CTR config opens a session
/// (`Ok(Some)`), actually accumulating the ordered CTR + binarizing the extra cindex columns ON
/// device during `begin` (the serial read-before-increment scan runs in-env on cpu — no
/// `Atomic<u64>`); a MULTI-FOLD CTR config declines (`Ok(None)`, Open Q3 deferred); a CTR config
/// whose columns do not binarize to `n_bins` buckets declines; and CTR PLUS a second non-default
/// family flag declines (D-10-01 all-or-nothing). Runs on every non-wgpu backend.
#[test]
fn session_ctr_gate_covers_single_permutation() {
    use cb_compute::{DeviceCtrColumn, DeviceCtrConfig};

    let n = 64usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let weight = weight_mod5(n);
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);

    // A single covered CTR column: one small-cardinality cat member, identity permutation, binclf
    // classes, and 31 borders spanning (0,1) so `borders.len() + 1 == n_bins` (uniform histogram).
    let cat: Vec<u32> = (0..n).map(|k| (k % 5) as u32).collect();
    let target_class: Vec<u32> = (0..n).map(|k| (k % 2) as u32).collect();
    let permutation: Vec<u32> = (0..n as u32).collect();
    let borders: Vec<f64> = (1..n_bins).map(|b| b as f64 / n_bins as f64).collect();
    assert_eq!(borders.len() + 1, n_bins, "borders must yield exactly n_bins buckets");
    let ctr_column = DeviceCtrColumn {
        member_bins: vec![cat.clone()],
        prior: 0.5,
        borders: borders.clone(),
    };
    let covered_ctr = DeviceCtrConfig {
        permutation: permutation.clone(),
        target_class: target_class.clone(),
        columns: vec![ctr_column.clone()],
    };

    let open = |folds: usize, cfg: &DeviceTrainConfig| {
        GpuTrainSession::begin(
            &Loss::Rmse, 6, true, folds, EScoreFunction::Cosine, &cindex, &weight, n, n_features,
            n_bins, 0.3, scaled_l2, cfg,
        )
        .expect("begin must not error while classifying / building the CTR arm")
        .is_some()
    };

    // COVERED: single-permutation CTR → Ok(Some) (begin accumulates + binarizes on device).
    let cfg_covered = DeviceTrainConfig {
        ctr: Some(covered_ctr.clone()),
        ..DeviceTrainConfig::default()
    };
    assert!(
        open(1, &cfg_covered),
        "a single-permutation covered CTR config must open a session (Plan 08 gate arm)"
    );

    // NOT covered: multi-fold / multi-permutation CTR → Ok(None) (Open Q3 deferred behind None).
    assert!(
        !open(2, &cfg_covered),
        "a multi-fold (fold_count>1) CTR config must decline (Open Q3 multi-permutation deferral)"
    );

    // NOT covered: a CTR column that does NOT binarize to n_bins buckets (wrong border count).
    let wrong_bins = DeviceTrainConfig {
        ctr: Some(DeviceCtrConfig {
            permutation: permutation.clone(),
            target_class: target_class.clone(),
            columns: vec![DeviceCtrColumn {
                member_bins: vec![cat.clone()],
                prior: 0.5,
                borders: vec![0.5_f64], // 2 buckets != n_bins
            }],
        }),
        ..DeviceTrainConfig::default()
    };
    assert!(
        !open(1, &wrong_bins),
        "a CTR column not binarized to n_bins buckets must decline (uniform-histogram invariant)"
    );

    // NOT covered: CTR PLUS a second non-default family flag (exact leaf) → decline (all-or-nothing).
    let ctr_plus_exact = DeviceTrainConfig {
        ctr: Some(covered_ctr.clone()),
        exact_leaf: true,
        ..DeviceTrainConfig::default()
    };
    assert!(
        !open(1, &ctr_plus_exact),
        "CTR + a second non-default family flag must decline (D-10-01 all-or-nothing)"
    );

    // The default (no CTR) path is unchanged.
    assert!(open(1, &DeviceTrainConfig::default()), "the no-CTR default still opens");
}

/// (End-to-end residency, Plan 08 GPUT-10) a covered CTR session actually augments its resident
/// cindex with the binarized CTR columns during `begin` — the effective feature count grows by the
/// number of CTR columns, and the session opens with the extra columns device-resident. The ordered
/// CTR VALUE numerics are locked ≤1e-4 by the `kernels::ctr_device` self-oracle; here we prove the
/// begin wiring accumulates + joins the CTR columns on device end-to-end. The serial CTR scan runs
/// in-env on cpu (no `Atomic<u64>`); a full-tree grow over the augmented cindex is the Plan-09
/// Kaggle CUDA sign-off (needs the resident partition histogram), so this test does NOT call
/// grow_one.
#[test]
fn session_ctr_augments_resident_cindex() {
    use cb_compute::{DeviceCtrColumn, DeviceCtrConfig};

    let n = 80usize;
    let n_features = 2usize;
    let n_bins = 16usize;
    let weight = vec![1.0_f64; n];
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);

    let cat0: Vec<u32> = (0..n).map(|k| (k % 4) as u32).collect();
    let cat1: Vec<u32> = (0..n).map(|k| (k % 6) as u32).collect();
    let target_class: Vec<u32> = (0..n).map(|k| ((k / 3) % 2) as u32).collect();
    let permutation: Vec<u32> = (0..n as u32).collect();
    let borders: Vec<f64> = (1..n_bins).map(|b| b as f64 / n_bins as f64).collect();

    // TWO CTR columns: one plain single-feature, one tensor/feature-combination (2 members, A5).
    let ctr = DeviceCtrConfig {
        permutation,
        target_class,
        columns: vec![
            DeviceCtrColumn { member_bins: vec![cat0.clone()], prior: 0.5, borders: borders.clone() },
            DeviceCtrColumn {
                member_bins: vec![cat0.clone(), cat1.clone()],
                prior: 1.0,
                borders: borders.clone(),
            },
        ],
    };
    let cfg = DeviceTrainConfig { ctr: Some(ctr), ..DeviceTrainConfig::default() };

    let session = GpuTrainSession::begin(
        &Loss::Rmse, 3, true, 1, EScoreFunction::Cosine, &cindex, &weight, n, n_features, n_bins,
        0.3, scaled_l2, &cfg,
    )
    .expect("begin must not error building the CTR-augmented resident session")
    .expect("a covered 2-column CTR config must open a session (Plan 08)");

    // The effective feature count grew by the two binarized CTR columns (they are now resident
    // cindex features the histogram loop reads).
    assert_eq!(
        session.n_features_effective(),
        n_features + 2,
        "the resident cindex must gain one feature per binarized CTR column"
    );
    assert_eq!(session.n(), n, "object count unchanged by the CTR augmentation");
    drop(session);
}

/// (End-to-end wiring, Plan 06 GPUT-09) a Bernoulli-bootstrap session grows a real device tree:
/// `grow_one` draws the device-resident keep-mask, folds it into the resident weight, and grows a
/// finite tree. Proves the begin→grow_one bootstrap wiring runs on device; the DRAW numerics are
/// locked bit-for-bit by the `kernels::bootstrap_device` self-oracle. Needs `Atomic<u64>` (the
/// resident histogram) + the u64 RNG kernel → SKIPS on cpu/wgpu, runs on rocm/cuda in-env.
#[test]
fn session_bootstrap_grows_finite_tree() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!(
            "[12-06] SKIP session_bootstrap_grows_finite_tree: active backend lacks Atomic<u64> / \
             the u64 RNG kernel (cpu/wgpu) — needs rocm/cuda"
        );
        return;
    }
    let n = 256usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let depth = 3usize;
    let target = ramp_centred(n);
    let weight = vec![1.0_f64; n];
    let cindex = cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, sum_f64(&weight), n);

    let bern = DeviceTrainConfig {
        bootstrap_type: cb_compute::DeviceBootstrapType::Bernoulli,
        sample_rate: 0.7,
        rng_seed: 2024,
        ..DeviceTrainConfig::default()
    };
    let mut session = GpuTrainSession::begin(
        &Loss::Rmse, depth, true, 1, EScoreFunction::Cosine, &cindex, &weight, n, n_features,
        n_bins, 0.3, scaled_l2, &bern,
    )
    .expect("begin must not error on a covered Bernoulli-bootstrap config")
    .expect("Bernoulli bootstrap must open a session (Plan 06 gate arm)");

    // Grow two trees to exercise the continuous-stream advance between trees.
    let approx = vec![0.0_f64; n];
    let t0 = match session.grow_one(&approx, &target) {
        Ok(t) => t,
        Err(cb_core::CbError::Unsupported(msg)) if msg.contains("Atomic<u64>") => {
            // The resident partition histogram needs an ADVERTISED Atomic<u64> add. When the
            // in-env ROCm runtime does not advertise it (an environment/driver capability state,
            // NOT a bootstrap-draw defect — the `kernels::bootstrap_device` self-oracle proves the
            // draw bit-for-bit), the WHOLE resident grow is unavailable, so skip the e2e wiring
            // check (WR-01 capability-skip pattern) rather than fail on an environmental gate.
            println!("[12-06] SKIP session_bootstrap_grows_finite_tree: {msg}");
            return;
        }
        Err(e) => panic!("bootstrap grow_one (tree 0) must succeed: {e:?}"),
    };
    let t1 = session.grow_one(&approx, &target).expect("bootstrap grow_one (tree 1) must succeed");
    for tree in [&t0, &t1] {
        assert_eq!(tree.splits.len(), depth, "bootstrap tree must have {depth} splits");
        assert_eq!(tree.leaf_of.len(), n, "leaf_of length must equal n");
        assert_eq!(tree.leaf_values.len(), 1usize << depth, "2^depth leaf values");
        for (l, &v) in tree.leaf_values.iter().enumerate() {
            assert!(v.is_finite(), "bootstrap leaf value {l} must be finite, got {v}");
        }
    }
    drop(session);
}
