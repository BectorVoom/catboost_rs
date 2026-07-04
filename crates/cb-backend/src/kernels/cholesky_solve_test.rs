//! Self-oracle for the device batched f64 Cholesky solver (Phase 13 Plan 02, GPUT-21, Pattern F):
//! the device [`crate::kernels::cholesky_solve`] solve must reproduce the frozen Rust CPU parity
//! references within `TOL` (ε=1e-4, the D-07 GPU bar):
//!
//! - **Leaf values** — `solve_pairwise_leaf_values_host` == `calculate_pairwise_leaf_values`
//!   (`cb_train::pairwise_leaves:113-195`), including the `system_size == 2` closed form, the
//!   general `(n-1)×(n-1)` build, `res.push(0.0)` and `make_zero_average`.
//! - **Split score** — `score_pairwise_cholesky_host` == the `calculate_pairwise_score` inner
//!   (`cb_compute::pairwise_scoring`): the leaf solve composed with `calculate_score`
//!   (`score = Σ_x avrg[x]·(sumDer[x] − ½·Σ_y avrg[y]·weightSum[x][y])`).
//! - **Degenerate** — a non-positive-definite system returns the zeros fallback on BOTH the device
//!   and the CPU (no NaN/panic, T-13-03).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernel + launcher live
//! in the production `kernels::cholesky_solve` module; ALL assertions / `.unwrap()` / indexing live
//! HERE. The CPU references transcribe the SAME catboost regularization constants + solve/score
//! composition INLINE, reusing ONLY `cb_compute::pairwise_cholesky_solve` (the shared in-house SPD
//! primitive, already a dep) — NO `cb-train` dep even in the test (the feature-unification
//! landmine). The device kernel is a separate CubeCL JIT codepath, so this is an independent
//! implementation, NOT a tautology.
//!
//! Runs over [`crate::SelectedRuntime`]. The serial f64 solver is validated on ROCm/CUDA in-env;
//! the numeric assertion SKIPS off rocm/cuda (record-only) so a default `cpu`-backend run does not
//! silently "pass" a CPU-vs-CPU compare without a real device (WR-01 anti-false-pass). The whole
//! file is gated `not(feature = "wgpu")` — the f64 solve has no wgpu backend (the launcher rejects
//! it with a typed error, never a JIT crash).
#![cfg(not(feature = "wgpu"))]

use crate::kernels::cholesky_solve::{score_pairwise_cholesky_host, solve_pairwise_leaf_values_host};
use cb_compute::pairwise_cholesky_solve;

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// The catboost pairwise bucket-weight prior reg default (`bayesian_matrix_reg` / `PairwiseNonDiagReg`,
/// upstream default `0.1`) — transcribed inline (NO `cb-train` dep).
const PRIOR: f64 = 0.1;

/// Whether the device solve actually runs on a real device backend (rocm/cuda). On the default
/// `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU false-pass
/// (WR-01) — record-only there, hard-assert on rocm/cuda.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// Zero-center a leaf-delta vector (`MakeZeroAverage`, `matrix.h:5-15`) via a plain ascending fold
/// (== `cb_core::sum_f64` order for these small systems; the ε=1e-4 bar is far above the fold-order
/// residual). Transcribed inline so the reference stays independent.
fn make_zero_average(res: &mut [f64]) {
    let n = res.len();
    if n == 0 {
        return;
    }
    let average = res.iter().sum::<f64>() / n as f64;
    for v in res.iter_mut() {
        *v -= average;
    }
}

/// Independent CPU reference for `calculate_pairwise_leaf_values` (`pairwise_leaves.rs:113-195`),
/// transcribing the reg constants + `system_size == 2` closed form + general `(n-1)×(n-1)` build +
/// `push(0.0)` + `make_zero_average` INLINE, reusing `cb_compute::pairwise_cholesky_solve` for the
/// SPD solve. `weight_sums` is row-major `n × n` (no ridge); `der_sums` is length `n`.
fn cpu_pairwise_leaf_values(
    weight_sums: &[f64],
    der_sums: &[f64],
    l2_diag_reg: f64,
    prior: f64,
) -> Vec<f64> {
    let n = der_sums.len();
    if n == 0 {
        return Vec::new();
    }
    let cell_prior = 1.0 / n as f64;
    let non_diag_reg = -prior * cell_prior;
    let diag_reg = prior * (1.0 - cell_prior) + l2_diag_reg;

    if n == 1 {
        return vec![0.0];
    }
    if n == 2 {
        let a11 = weight_sums[0];
        let denom = a11 + diag_reg;
        let x0 = if denom != 0.0 { der_sums[0] / denom } else { 0.0 };
        let mut res = vec![x0, 0.0];
        make_zero_average(&mut res);
        return res;
    }

    let m = n - 1;
    let mut matrix = vec![vec![0.0_f64; m]; m];
    for y in 0..m {
        for x in 0..y {
            let v = weight_sums[y * n + x] + non_diag_reg;
            matrix[y][x] = v;
            matrix[x][y] = v;
        }
        matrix[y][y] = weight_sums[y * n + y] + diag_reg;
    }
    let rhs: Vec<f64> = der_sums.iter().take(m).copied().collect();
    let mut res = pairwise_cholesky_solve(&matrix, &rhs).unwrap_or_else(|| vec![0.0; m]);
    res.push(0.0);
    make_zero_average(&mut res);
    res
}

/// Independent CPU reference for `calculate_score` (`pairwise_scoring.cpp:51-81`):
/// `Σ_x avrg[x]·(sumDer[x] − ½·Σ_y avrg[y]·weightSum[x][y])`. `weight_sum` is row-major `n × n`.
fn cpu_calculate_score(avrg: &[f64], sum_der: &[f64], weight_sum: &[f64], n: usize) -> f64 {
    let mut outer = 0.0_f64;
    for x in 0..n {
        let avrg_x = avrg.get(x).copied().unwrap_or(0.0);
        let der_x = sum_der.get(x).copied().unwrap_or(0.0);
        let mut sub = 0.0_f64;
        for y in 0..n {
            sub += avrg.get(y).copied().unwrap_or(0.0) * weight_sum[x * n + y];
        }
        outer += avrg_x * (der_x - 0.5 * sub);
    }
    outer
}

/// Max absolute divergence over two equal-length buffers (∞ if the lengths disagree).
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

/// Test 1: a 3-leaf leaf-value system — the device solve equals `calculate_pairwise_leaf_values`
/// within `TOL`, including the trailing-zero + zero-average steps (rowSize == 2 → 2×2 reduced solve).
#[test]
fn device_leaf_values_match_cpu_three_leaves() {
    let weight_sums = vec![
        4.0, -1.5, -2.5, //
        -1.5, 3.0, -1.5, //
        -2.5, -1.5, 4.0,
    ];
    let der_sums = vec![0.75, -0.25, -0.5];
    let l2 = 3.0;

    let reference = cpu_pairwise_leaf_values(&weight_sums, &der_sums, l2, PRIOR);
    assert_eq!(reference.len(), 3, "leaf-value vector has length leaf_count");
    // Zero-averaged → mean ≈ 0.
    assert!(reference.iter().sum::<f64>().abs() < 1e-9, "CPU leaf values must be zero-centered");

    let device = solve_pairwise_leaf_values_host(&weight_sums, &der_sums, l2, PRIOR)
        .expect("device leaf-value solve must not error on a covered SPD fixture");

    for &v in &device {
        assert!(v.is_finite(), "device leaf value must be finite, got {v}");
    }
    let divergence = max_abs_divergence(&device, &reference);
    println!(
        "[cholesky_solve] 3-leaf leaf-value device-vs-CPU max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device leaf values diverged from CPU: {divergence:e} > {TOL:e}\n\
             device = {device:?}\nreference = {reference:?}"
        );
    }
}

/// Test 1b: a 4-leaf leaf-value system (rowSize == 3 → a deeper 3×3 reduced Cholesky) — the device
/// solve equals the CPU reference within `TOL`.
#[test]
fn device_leaf_values_match_cpu_four_leaves() {
    let weight_sums = vec![
        5.0, -1.0, -2.0, -2.0, //
        -1.0, 4.0, -1.5, -1.5, //
        -2.0, -1.5, 6.0, -2.5, //
        -2.0, -1.5, -2.5, 6.0,
    ];
    let der_sums = vec![1.0, -0.5, 0.25, -0.75];
    let l2 = 1.0;

    let reference = cpu_pairwise_leaf_values(&weight_sums, &der_sums, l2, PRIOR);
    let device = solve_pairwise_leaf_values_host(&weight_sums, &der_sums, l2, PRIOR)
        .expect("device 4-leaf leaf-value solve must not error");

    for &v in &device {
        assert!(v.is_finite(), "device leaf value must be finite, got {v}");
    }
    let divergence = max_abs_divergence(&device, &reference);
    println!("[cholesky_solve] 4-leaf leaf-value device-vs-CPU max abs divergence = {divergence:e}");
    if device_backend_active() {
        assert!(divergence <= TOL, "device leaf values diverged: {divergence:e} > {TOL:e}");
    }
}

/// Test 2: a split-score system (`n = 4` == 2 leaves) — the device `CalcScoresCholesky` score equals
/// the CPU `calculate_score(calculate_pairwise_leaf_values(...), der, weightSum)` within `TOL`.
#[test]
fn device_split_score_matches_cpu() {
    // Symmetric 4×4 running pairwise weight matrix (SPD after ridge) + der vector.
    let weight_sum = vec![
        6.0, -1.0, -1.0, -1.0, //
        -1.0, 6.0, -1.0, -1.0, //
        -1.0, -1.0, 6.0, -1.0, //
        -1.0, -1.0, -1.0, 6.0,
    ];
    let der_sum = vec![0.5, -0.3, 0.2, -0.4];
    let l2 = 3.0;
    let n = 4;

    let cpu_leaves = cpu_pairwise_leaf_values(&weight_sum, &der_sum, l2, PRIOR);
    let cpu_score = cpu_calculate_score(&cpu_leaves, &der_sum, &weight_sum, n);

    let device_score = score_pairwise_cholesky_host(&weight_sum, &der_sum, l2, PRIOR)
        .expect("device split-score solve must not error on a covered SPD fixture");

    assert!(device_score.is_finite(), "device split score must be finite, got {device_score}");
    let divergence = (device_score - cpu_score).abs();
    println!(
        "[cholesky_solve] split-score device={device_score:e} cpu={cpu_score:e} \
         abs divergence = {divergence:e} (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device split score diverged from CPU: {divergence:e} > {TOL:e} \
             (device={device_score}, cpu={cpu_score})"
        );
    }
}

/// Test 3 (degenerate): a non-positive-definite system (all-zero weight sums, no ridge) returns the
/// zeros fallback on BOTH the device and the CPU — a FINITE zero vector, never a NaN/panic
/// (T-13-03). The Cholesky decomposition hits a non-positive pivot immediately.
#[test]
fn degenerate_non_pd_system_returns_zeros_both_paths() {
    // 3-leaf all-zero weight sums; no ridge (l2 = 0, prior = 0) → reduced 2×2 is all zeros → the
    // first pivot is 0 ≤ 0 → the zeros fallback (cholesky_solve returns None).
    let weight_sums = vec![0.0; 9];
    let der_sums = vec![1.0, -1.0, 0.0];
    let l2 = 0.0;
    let prior = 0.0;

    let cpu = cpu_pairwise_leaf_values(&weight_sums, &der_sums, l2, prior);
    // CPU: reduced solve → None → zeros; push 0; zero-average of zeros → all zeros.
    for &v in &cpu {
        assert_eq!(v, 0.0, "CPU degenerate leaf values must be exactly zero");
    }

    let device = solve_pairwise_leaf_values_host(&weight_sums, &der_sums, l2, prior)
        .expect("device degenerate solve must not error");
    assert_eq!(device.len(), 3, "degenerate system still emits leaf_count deltas");
    for &v in &device {
        assert!(v.is_finite(), "device degenerate leaf value must be finite (no NaN), got {v}");
    }
    // The zeros fallback is exact on both paths (a non-positive pivot yields all-zeros before any
    // divergent arithmetic), so this holds even on the cpu backend.
    let divergence = max_abs_divergence(&device, &cpu);
    assert!(
        divergence <= TOL,
        "device degenerate fallback diverged from CPU zeros: {divergence:e} > {TOL:e}\n\
         device = {device:?}"
    );
}
