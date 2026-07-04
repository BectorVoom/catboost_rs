//! Self-oracle for the multi-output device block-leaf driver (Phase 13 Plan 07, GPUT-12): the
//! device [`crate::gpu_runtime::multiclass::grow_multiclass_block`] must reproduce the CPU
//! multi-output leaf values — per leaf, the ordered `cb_core::sum_f64` der/der2 reduction followed by
//! `cb_compute::solve_symmetric_newton` (COUPLED full K×K for MultiClass softmax; DIAGONAL
//! per-component `k == 1` for the separable losses) — within the ε=1e-4 GPU bar (D-07).
//!
//! Three frozen fixtures cover BOTH hessian structures with minimal inputs (RESEARCH A2):
//!
//! 1. **Coupled softmax MultiClass K=3** — the full K×K dense symmetric solve.
//! 2. **Diagonal RMSEWithUncertainty K=2** — distinct row-0 (`der2 = −w`) / row-1
//!    (`der2 = −2w·diff²·prec`) hessians, the per-component diagonal path.
//! 3. **Diagonal MultiClassOneVsAll K=3** — the separable sigmoid diagonal path.
//!
//! Plus the coverage gate: a multi-output loss classifies via
//! [`crate::gpu_runtime::multiclass::map_multiclass_objective`], and `GpuTrainSession::begin`
//! declines a covered multi-output config to `Ok(None)` (the per-tree shared multi-dim grow seam is
//! a forward dependency — the pairwise / ranking precedent), never fabricating a scalar grow.
//!
//! Source/test separation (CLAUDE.md / AGENTS.md): the driver + coverage gate are production code;
//! ALL `#[test]` + `.unwrap()`/indexing live here. cb-backend must NEVER gain a `cb-train` dep even
//! in the test — the CPU reference is `cb_compute::solve_symmetric_newton` directly. The numeric ε
//! assertion SKIPS off rocm/cuda (record-only) so a default `cpu`-backend run does not silently
//! "pass" a CPU-vs-CPU compare without a real device (WR-01 anti-false-pass). The whole file is
//! gated off the (f64-less) wgpu backend.

#![cfg(not(feature = "wgpu"))]

use cb_compute::{solve_symmetric_newton, DeviceTrainConfig, EScoreFunction, Loss};
use cb_core::sum_f64;

use crate::gpu_runtime::multiclass::{
    assemble_multiclass_ders, grow_multiclass_block, map_multiclass_objective, MulticlassObjective,
};
use crate::gpu_runtime::GpuTrainSession;

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device solve actually runs on a real device backend (rocm/cuda). On the default
/// `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU false-pass
/// (WR-01) — record-only there, hard-assert on rocm/cuda.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// Max abs divergence over equal-length buffers (`INFINITY` on a length mismatch).
fn max_abs_divergence(device: &[f64], reference: &[f64]) -> f64 {
    if device.len() != reference.len() {
        return f64::INFINITY;
    }
    device
        .iter()
        .zip(reference.iter())
        .map(|(&d, &r)| (d - r).abs())
        .fold(0.0_f64, f64::max)
}

/// The packed lower-triangular index of the diagonal entry `(d, d)` in `[(0,0),(0,1),…]`.
fn diag_index(d: usize, k: usize) -> usize {
    if d == 0 {
        return 0;
    }
    d * k - d * (d - 1) / 2
}

/// The CPU multi-output leaf-block reference: assemble the per-object der (the SAME
/// `cb-compute`-backed assembly the driver uses), accumulate each leaf's members in ascending object
/// order through the ordered `cb_core::sum_f64`, then run the host `solve_symmetric_newton` (coupled
/// full-block for softmax; per-component `k == 1` diagonal for the separable losses). Returns the
/// `leaf_count × K` ROW-MAJOR block `out[leaf * k + d]` — the SAME layout the device driver emits.
fn cpu_reference_block(
    objective: MulticlassObjective,
    approx: &[f64],
    target: &[f64],
    weight: &[f64],
    leaf_of: &[u32],
    k: usize,
    n_leaves: usize,
    scaled_l2: f64,
) -> Vec<f64> {
    let n = leaf_of.len();
    let pk = k * (k + 1) / 2;
    let (der1, der2_packed) =
        assemble_multiclass_ders(objective, approx, target, weight, n, k).expect("assemble der");

    let mut der1_members: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); k]; n_leaves];
    let mut der2_members: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); pk]; n_leaves];
    for i in 0..n {
        let leaf = leaf_of[i] as usize;
        if leaf >= n_leaves {
            continue;
        }
        for d in 0..k {
            der1_members[leaf][d].push(der1[d * n + i]);
        }
        for j in 0..pk {
            der2_members[leaf][j].push(der2_packed[i * pk + j]);
        }
    }

    let mut out = vec![0.0_f64; n_leaves * k];
    for leaf in 0..n_leaves {
        let sum_der: Vec<f64> = (0..k).map(|d| sum_f64(&der1_members[leaf][d])).collect();
        let sum_der2: Vec<f64> = (0..pk).map(|j| sum_f64(&der2_members[leaf][j])).collect();
        let delta: Vec<f64> = if objective.is_coupled() {
            solve_symmetric_newton(&sum_der, &sum_der2, scaled_l2)
        } else {
            (0..k)
                .map(|d| {
                    solve_symmetric_newton(&[sum_der[d]], &[sum_der2[diag_index(d, k)]], scaled_l2)
                        .first()
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect()
        };
        for d in 0..k {
            out[leaf * k + d] = delta.get(d).copied().unwrap_or(0.0);
        }
    }
    out
}

/// Assert the device block equals the CPU reference within `TOL` on a real device (record-only on
/// cpu). Also asserts every device value is finite.
fn assert_block_matches(label: &str, device: &[f64], reference: &[f64]) {
    assert_eq!(
        device.len(),
        reference.len(),
        "[{label}] device/reference block length mismatch"
    );
    for &v in device {
        assert!(v.is_finite(), "[{label}] device block value must be finite, got {v}");
    }
    let divergence = max_abs_divergence(device, reference);
    println!(
        "[multiclass:{label}] device-vs-CPU max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "[{label}] device block diverged from CPU: {divergence:e} > {TOL:e}\n\
             device = {device:?}\nreference = {reference:?}"
        );
    }
}

/// Test 1: coupled softmax MultiClass K=3 — the device full K×K block equals the CPU
/// `solve_symmetric_newton` multi-output leaf values within `TOL`.
#[test]
fn coupled_softmax_k3_matches_cpu_multi_output() {
    let k = 3usize;
    let n = 6usize;
    let n_leaves = 2usize;
    // Dimension-major approx (approx[d*n+i]): three classes with a spread of logits per object.
    let approx: Vec<f64> = vec![
        // dim 0
        0.4, -0.2, 0.1, 0.9, -0.6, 0.3, // dim 1
        -0.1, 0.5, -0.3, 0.2, 0.7, -0.4, // dim 2
        0.2, 0.0, 0.6, -0.5, 0.1, 0.8,
    ];
    // Per-object remapped class label in [0, k).
    let target: Vec<f64> = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    let weight = vec![1.0_f64; n];
    // Two leaves split at the midpoint.
    let leaf_of: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
    let scaled_l2 = 3.0_f64;

    let objective = MulticlassObjective::Softmax;
    let device = grow_multiclass_block(objective, &leaf_of, &approx, &target, &weight, k, n_leaves, scaled_l2)
        .expect("device coupled softmax block must not error");
    let reference =
        cpu_reference_block(objective, &approx, &target, &weight, &leaf_of, k, n_leaves, scaled_l2);
    assert_eq!(reference.len(), n_leaves * k, "reference block is leaf_count × K");
    assert_block_matches("coupled_softmax_k3", &device, &reference);
}

/// Test 2: diagonal RMSEWithUncertainty K=2 (distinct row-0/row-1 hessian) — the device
/// per-component diagonal block equals the CPU diagonal path within `TOL`.
#[test]
fn diagonal_rmse_uncertainty_k2_matches_cpu_multi_output() {
    let k = 2usize;
    let n = 5usize;
    let n_leaves = 2usize;
    // Dimension-major approx: dim 0 = regression MEAN, dim 1 = LOG-SCALE (distinct hessians —
    // der2[0] = −w, der2[1] = −2w·diff²·exp(−2·a1), so the two rows differ per object).
    let approx: Vec<f64> = vec![
        // dim 0 (mean)
        1.0, 2.0, 0.5, -0.5, 1.5, // dim 1 (log-scale)
        0.2, -0.3, 0.4, 0.1, -0.2,
    ];
    // Per-object regression target.
    let target: Vec<f64> = vec![1.2, 1.7, 0.9, -0.2, 1.1];
    let weight = vec![1.0_f64; n];
    let leaf_of: Vec<u32> = vec![0, 0, 1, 1, 1];
    let scaled_l2 = 2.0_f64;

    let objective = MulticlassObjective::RmseWithUncertainty;
    let device = grow_multiclass_block(objective, &leaf_of, &approx, &target, &weight, k, n_leaves, scaled_l2)
        .expect("device diagonal RMSEWithUncertainty block must not error");
    let reference =
        cpu_reference_block(objective, &approx, &target, &weight, &leaf_of, k, n_leaves, scaled_l2);
    assert_eq!(reference.len(), n_leaves * k, "reference block is leaf_count × K");
    assert_block_matches("diagonal_rmse_uncertainty_k2", &device, &reference);
}

/// Test 3: diagonal MultiClassOneVsAll K=3 — the device separable sigmoid diagonal block equals the
/// CPU diagonal path within `TOL`.
#[test]
fn diagonal_onevsall_k3_matches_cpu_multi_output() {
    let k = 3usize;
    let n = 6usize;
    let n_leaves = 3usize;
    let approx: Vec<f64> = vec![
        // dim 0
        0.3, -0.4, 0.7, 0.1, -0.2, 0.5, // dim 1
        -0.5, 0.6, 0.2, -0.1, 0.4, -0.3, // dim 2
        0.1, 0.0, -0.6, 0.8, -0.4, 0.2,
    ];
    let target: Vec<f64> = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    let weight = vec![1.0_f64; n];
    let leaf_of: Vec<u32> = vec![0, 0, 1, 1, 2, 2];
    let scaled_l2 = 1.5_f64;

    let objective = MulticlassObjective::OneVsAll;
    let device = grow_multiclass_block(objective, &leaf_of, &approx, &target, &weight, k, n_leaves, scaled_l2)
        .expect("device diagonal OneVsAll block must not error");
    let reference =
        cpu_reference_block(objective, &approx, &target, &weight, &leaf_of, k, n_leaves, scaled_l2);
    assert_eq!(reference.len(), n_leaves * k, "reference block is leaf_count × K");
    assert_block_matches("diagonal_onevsall_k3", &device, &reference);
}

/// The loss → objective classification: the covered multi-output losses map to their objective
/// (coupled ONLY for MultiClass softmax), and every scalar / non-covered loss declines.
#[test]
fn objective_classification_and_coupled_dispatch() {
    assert_eq!(map_multiclass_objective(&Loss::MultiClass), Some(MulticlassObjective::Softmax));
    assert_eq!(
        map_multiclass_objective(&Loss::MultiClassOneVsAll),
        Some(MulticlassObjective::OneVsAll)
    );
    assert_eq!(
        map_multiclass_objective(&Loss::MultiLogloss),
        Some(MulticlassObjective::MultiCrossEntropy)
    );
    assert_eq!(
        map_multiclass_objective(&Loss::MultiCrossEntropy),
        Some(MulticlassObjective::MultiCrossEntropy)
    );
    assert_eq!(
        map_multiclass_objective(&Loss::RmseWithUncertainty),
        Some(MulticlassObjective::RmseWithUncertainty)
    );
    // Coupled is used ONLY for MultiClass softmax; every separable arm is diagonal.
    assert!(MulticlassObjective::Softmax.is_coupled());
    for obj in [
        MulticlassObjective::OneVsAll,
        MulticlassObjective::MultiCrossEntropy,
        MulticlassObjective::MultiRmse,
        MulticlassObjective::RmseWithUncertainty,
    ] {
        assert!(!obj.is_coupled(), "{obj:?} must use the diagonal path");
    }
    // A scalar loss is not a multi-output objective.
    assert_eq!(map_multiclass_objective(&Loss::Rmse), None);
    assert_eq!(map_multiclass_objective(&Loss::Logloss), None);
}

/// The coverage gate: `GpuTrainSession::begin` declines a multi-output loss to `Ok(None)` (the
/// per-tree shared multi-dim grow seam is a forward dependency — the pairwise / ranking precedent),
/// for BOTH a covered config and a genuinely uncovered one. Never a fabricated scalar grow, never an
/// error while classifying.
#[test]
fn begin_declines_multi_output_to_cpu() {
    let n = 6usize;
    let n_features = 2usize;
    let n_bins = 32usize;
    let weight = vec![1.0_f64; n];
    // Minimal valid feature-major cindex (bins < n_bins). The multi-output gate returns before the
    // host-side cindex validation, but keep it well-formed.
    let mut cindex = vec![0u32; n_features * n];
    for f in 0..n_features {
        for obj in 0..n {
            cindex[f * n + obj] = (obj % n_bins) as u32;
        }
    }
    let scaled_l2 = 2.0_f64;
    let lr = 0.3_f64;

    let open = |loss: &Loss, depth: usize, plain: bool, folds: usize, cfg: &DeviceTrainConfig| {
        GpuTrainSession::begin(
            loss,
            depth,
            plain,
            folds,
            EScoreFunction::Cosine,
            &cindex,
            &weight,
            n,
            n_features,
            n_bins,
            lr,
            scaled_l2,
            cfg,
        )
        .expect("begin must not error while classifying multi-output coverage")
        .is_some()
    };
    let def = DeviceTrainConfig::default();

    // Covered multi-output config (SymmetricTree, depth≥1, Plain, single fold) → declines to CPU.
    assert!(
        !open(&Loss::MultiClass, 2, true, 1, &def),
        "covered MultiClass declines to CPU pending the shared multi-dim grow seam"
    );
    assert!(
        !open(&Loss::RmseWithUncertainty, 2, true, 1, &def),
        "covered RMSEWithUncertainty declines to CPU pending the shared multi-dim grow seam"
    );
    // Genuinely uncovered multi-output configs → also Ok(None) (all-or-nothing per family).
    assert!(
        !open(&Loss::MultiClassOneVsAll, 2, false, 1, &def),
        "non-Plain multi-output must decline"
    );
    assert!(
        !open(&Loss::MultiClass, 2, true, 2, &def),
        "fold_count>1 multi-output must decline"
    );
}
