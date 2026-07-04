//! Self-oracle for the device K-dim Newton der2 block solve (Phase 13 Plan 06, GPUT-12, Pattern F):
//! the device [`crate::kernels::multi_newton`] block solve must reproduce the frozen Rust CPU parity
//! reference [`cb_compute::solve_symmetric_newton`] within `TOL` (ε=1e-4, the D-07 GPU bar):
//!
//! - **Coupled (K=3 softmax)** — `solve_multi_newton_host(coupled = true)` == `solve_symmetric_newton`
//!   over the full packed K×K hessian (the MultiClass softmax path).
//! - **Diagonal (K=2 separable)** — `solve_multi_newton_host(coupled = false)` == the CPU diagonal
//!   path (per-component `solve_symmetric_newton` with `k == 1`), reading only the packed diagonal.
//! - **Non-PD** — a non-positive-definite block returns the zeros fallback on BOTH the device and
//!   the CPU (no NaN/panic, T-13-11).
//! - **Scalar K=1 no-regression** — the K=1 block equals the prior scalar Newton result.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernel + launcher live in
//! the production `kernels::multi_newton` module; ALL assertions / `.unwrap()` / indexing live HERE.
//! The CPU reference reuses ONLY `cb_compute::solve_symmetric_newton` (already a dep) — NO `cb-train`
//! dep even in the test (the feature-unification landmine). The device kernel is a separate CubeCL
//! JIT codepath, so this is an independent implementation, NOT a tautology.
//!
//! Runs over [`crate::SelectedRuntime`]. The serial f64 solve is validated on ROCm/CUDA in-env; the
//! numeric assertion SKIPS off rocm/cuda (record-only) so a default `cpu`-backend run does not
//! silently "pass" a CPU-vs-CPU compare without a real device (WR-01 anti-false-pass). The whole
//! file is gated `not(feature = "wgpu")` — the f64 solve has no wgpu backend (the launcher rejects
//! it with a typed error, never a JIT crash).
#![cfg(not(feature = "wgpu"))]

use crate::kernels::multi_newton::solve_multi_newton_host;
use cb_compute::solve_symmetric_newton;

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device solve actually runs on a real device backend (rocm/cuda). On the default
/// `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU false-pass
/// (WR-01) — record-only there, hard-assert on rocm/cuda.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
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

/// CPU DIAGONAL reference: the separable losses solve each component as an independent 1×1 Newton
/// system — `solve_symmetric_newton(&[sum_der[d]], &[H[d][d]], scaled_l2)[0]` per dimension. This is
/// the frozen `cb_compute::solve_symmetric_newton` restricted to `k == 1` per component; the device
/// diagonal mode reads only the packed diagonal entries and must reproduce this.
fn cpu_diagonal(sum_der: &[f64], packed_diag: &[f64], scaled_l2: f64) -> Vec<f64> {
    sum_der
        .iter()
        .zip(packed_diag.iter())
        .map(|(&d, &h)| {
            solve_symmetric_newton(&[d], &[h], scaled_l2)
                .first()
                .copied()
                .unwrap_or(0.0)
        })
        .collect()
}

/// The packed diagonal index of `(d, d)` in the order `[(0,0),(0,1),…,(0,k-1),(1,1),…]`:
/// `d·k − d·(d−1)/2`.
fn diag_index(d: usize, k: usize) -> usize {
    // d·k − d·(d−1)/2; guard the d == 0 underflow (the kernel runs this in wrapping GPU arithmetic).
    if d == 0 {
        return 0;
    }
    d * k - d * (d - 1) / 2
}

/// Test 1: a coupled K=3 softmax block — the device full K×K solve equals `solve_symmetric_newton`
/// within `TOL`. The packed hessian is an SPD softmax-shaped block (diagonal `p(1−p) > 0` dominant).
#[test]
fn coupled_k3_softmax_matches_solve_symmetric_newton() {
    // Packed lower-triangular order for K=3: [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)].
    // A softmax-shaped hessian: diag positive-dominant, off-diagonals negative.
    let sum_der2_packed = vec![
        0.21,  // (0,0)
        -0.06, // (0,1)
        -0.05, // (0,2)
        0.24,  // (1,1)
        -0.07, // (1,2)
        0.20,  // (2,2)
    ];
    let sum_der = vec![0.30, -0.15, -0.15];
    let scaled_l2 = 3.0;

    let reference = solve_symmetric_newton(&sum_der, &sum_der2_packed, scaled_l2);
    assert_eq!(reference.len(), 3, "reference block has length K");

    let device = solve_multi_newton_host(&sum_der, &sum_der2_packed, scaled_l2, true)
        .expect("device coupled K=3 block solve must not error on a covered SPD fixture");

    for &v in &device {
        assert!(v.is_finite(), "device coupled block value must be finite, got {v}");
    }
    let divergence = max_abs_divergence(&device, &reference);
    println!(
        "[multi_newton] coupled K=3 device-vs-CPU max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device coupled block diverged from CPU: {divergence:e} > {TOL:e}\n\
             device = {device:?}\nreference = {reference:?}"
        );
    }
}

/// Test 2: a diagonal K=2 block — the device per-component solve equals the CPU diagonal path
/// (per-component `solve_symmetric_newton` with `k == 1`) within `TOL`.
#[test]
fn diagonal_k2_matches_cpu_diagonal_path() {
    let k = 2usize;
    // Packed K=2: [(0,0),(0,1),(1,1)]. Off-diagonal present in the buffer but IGNORED by the
    // diagonal mode (it reads only (0,0) and (1,1)).
    let sum_der2_packed = vec![0.36, -0.09, 0.49];
    let sum_der = vec![0.20, -0.30];
    let scaled_l2 = 2.0;

    let packed_diag: Vec<f64> = (0..k).map(|d| sum_der2_packed[diag_index(d, k)]).collect();
    let reference = cpu_diagonal(&sum_der, &packed_diag, scaled_l2);
    assert_eq!(reference.len(), 2, "diagonal reference has length K");

    let device = solve_multi_newton_host(&sum_der, &sum_der2_packed, scaled_l2, false)
        .expect("device diagonal K=2 block solve must not error");

    for &v in &device {
        assert!(v.is_finite(), "device diagonal block value must be finite, got {v}");
    }
    let divergence = max_abs_divergence(&device, &reference);
    println!(
        "[multi_newton] diagonal K=2 device-vs-CPU max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device diagonal block diverged from CPU: {divergence:e} > {TOL:e}\n\
             device = {device:?}\nreference = {reference:?}"
        );
    }
}

/// Test 3a: a non-PD coupled block — an all-zero hessian with no ridge yields a non-positive first
/// pivot → the zeros fallback on BOTH the device and the CPU (a FINITE zero vector, T-13-11).
#[test]
fn non_pd_coupled_block_returns_zeros_both_paths() {
    // K=3 all-zero packed hessian, scaled_l2 = 0 → M = -(-adjL2·I) with adjL2 = 0 → all-zero M →
    // the first Cholesky pivot is 0 ≤ 0 → the zeros fallback (matches solve_symmetric_newton None).
    let sum_der2_packed = vec![0.0; 6];
    let sum_der = vec![1.0, -1.0, 0.5];
    let scaled_l2 = 0.0;

    let reference = solve_symmetric_newton(&sum_der, &sum_der2_packed, scaled_l2);
    for &v in &reference {
        assert_eq!(v, 0.0, "CPU non-PD block must be exactly zero");
    }

    let device = solve_multi_newton_host(&sum_der, &sum_der2_packed, scaled_l2, true)
        .expect("device non-PD coupled block solve must not error");
    assert_eq!(device.len(), 3, "non-PD block still emits K deltas");
    for &v in &device {
        assert!(v.is_finite(), "device non-PD block value must be finite (no NaN), got {v}");
    }
    // The zeros fallback is exact on both paths (non-positive pivot yields all-zeros before any
    // divergent arithmetic), so this holds even on the cpu backend.
    let divergence = max_abs_divergence(&device, &reference);
    assert!(
        divergence <= TOL,
        "device non-PD fallback diverged from CPU zeros: {divergence:e} > {TOL:e}\n\
         device = {device:?}"
    );
}

/// Test 3b: scalar K=1 no-regression — the K=1 block (both coupled and diagonal, which coincide at
/// K=1) equals the prior scalar Newton result `solve_symmetric_newton(&[der], &[der2], l2)`.
#[test]
fn scalar_k1_matches_prior_scalar_newton() {
    let sum_der = vec![0.42];
    let sum_der2_packed = vec![0.7]; // K=1: packed length 1·2/2 == 1, the lone diagonal.
    let scaled_l2 = 1.5;

    let reference = solve_symmetric_newton(&sum_der, &sum_der2_packed, scaled_l2);
    assert_eq!(reference.len(), 1, "scalar reference has length 1");

    // Coupled and diagonal must both reproduce the scalar Newton delta at K=1.
    let device_coupled = solve_multi_newton_host(&sum_der, &sum_der2_packed, scaled_l2, true)
        .expect("device K=1 coupled block solve must not error");
    let device_diagonal = solve_multi_newton_host(&sum_der, &sum_der2_packed, scaled_l2, false)
        .expect("device K=1 diagonal block solve must not error");

    for &v in device_coupled.iter().chain(device_diagonal.iter()) {
        assert!(v.is_finite(), "device K=1 block value must be finite, got {v}");
    }
    // The two device paths must agree with each other at K=1 (structural — no device needed).
    let coupled_vs_diagonal = max_abs_divergence(&device_coupled, &device_diagonal);
    assert!(
        coupled_vs_diagonal <= TOL,
        "K=1 coupled and diagonal paths must coincide: {coupled_vs_diagonal:e} > {TOL:e}"
    );
    let divergence = max_abs_divergence(&device_coupled, &reference);
    println!(
        "[multi_newton] scalar K=1 device-vs-CPU max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device K=1 block diverged from the scalar Newton result: {divergence:e} > {TOL:e}\n\
             device = {device_coupled:?}\nreference = {reference:?}"
        );
    }
}
