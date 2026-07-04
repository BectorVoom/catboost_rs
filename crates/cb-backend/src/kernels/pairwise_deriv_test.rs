//! Self-oracle for the device pairwise per-leaf linear-system assembly (Phase 13 Plan 01,
//! GPUT-11 / GPUT-21 prep, Pattern F): the device
//! [`crate::gpu_runtime::assemble_pairwise_system_host`] must reproduce the packed
//! lower-triangular `linearSystem` the Rust CPU parity oracle
//! `cb_train::pairwise_leaves::calculate_pairwise_leaf_values` (:154-186) builds — the
//! `rowSize*(rowSize+1)/2` lower-triangle matrix cells (row-major, `x` in `0..=y`, with the
//! `diag_reg`/`non_diag_reg` catboost prior) FOLLOWED by the `rowSize` RHS, where
//! `rowSize = leaf_count - 1` — within `TOL` (ε=1e-4, the D-07 GPU bar).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device assembly kernel +
//! launcher live in the production `kernels`/`gpu_runtime::pairwise` modules; ALL assertions /
//! `.unwrap()` / indexing live HERE. The CPU reference transcribes the SAME catboost
//! regularization constants INLINE (NO `cb-train` dep even in the test — the feature-unification
//! landmine), so it is an independent implementation of the packing (the device kernel is a
//! separate CubeCL JIT codepath), NOT a tautology.
//!
//! Runs over [`crate::SelectedRuntime`]. The serial f64 assembly kernel is validated on ROCm/CUDA
//! in-env; the numeric assertion SKIPS off rocm/cuda (record-only) so a default `cpu`-backend run
//! does not silently "pass" a CPU-vs-CPU compare without exercising a real device (WR-01
//! anti-false-pass). The whole file is gated `not(feature = "wgpu")` — the f64 packed system has
//! no wgpu backend (the launcher rejects it with a typed error, never a JIT crash).
#![cfg(not(feature = "wgpu"))]

use crate::gpu_runtime::{assemble_pairwise_system_host, PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT};

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device assembly actually runs on a real device backend (rocm/cuda). On the default
/// `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU false-pass
/// (WR-01) — record-only there, hard-assert on rocm/cuda.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// Independent CPU reference: pack the same lower-triangular `linearSystem` the device kernel emits,
/// transcribing the `calculate_pairwise_leaf_values` reg constants inline (NO `cb-train` dep).
/// `rowSize = leaf_count - 1` (leaf gauge freedom drops the last row). Returns the empty system for
/// `leaf_count <= 1` (a singleton/empty system has no reduced matrix or RHS).
fn cpu_assemble_reference(
    weight_sums: &[f64],
    der_sums: &[f64],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> Vec<f64> {
    let leaf_count = der_sums.len();
    if leaf_count <= 1 {
        return Vec::new();
    }
    let cell_prior = 1.0 / leaf_count as f64;
    let non_diag_reg = -pairwise_bucket_weight_prior_reg * cell_prior;
    let diag_reg = pairwise_bucket_weight_prior_reg * (1.0 - cell_prior) + l2_diag_reg;
    let m = leaf_count - 1;
    let mut out = Vec::with_capacity(m * (m + 1) / 2 + m);
    for y in 0..m {
        for x in 0..=y {
            let reg = if x == y { diag_reg } else { non_diag_reg };
            out.push(weight_sums[y * leaf_count + x] + reg);
        }
    }
    for r in 0..m {
        out.push(der_sums[r]);
    }
    out
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

/// Test 1: a small 3-leaf fixture — the device-assembled packed system equals the CPU reference
/// within `TOL` over equal-length buffers (rowSize == 2: 3 matrix cells `[(0,0),(1,0),(1,1)]` then
/// 2 RHS).
#[test]
fn device_pairwise_system_matches_cpu_reference_three_leaves() {
    // Symmetric per-leaf pairwise weight-sum matrix (row-major 3×3) + der sums (len 3).
    let weight_sums = vec![
        4.0, -1.5, -2.5, //
        -1.5, 3.0, -1.5, //
        -2.5, -1.5, 4.0,
    ];
    let der_sums = vec![0.75, -0.25, -0.5];
    let l2_diag_reg = 3.0;
    let prior = PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT;

    let reference = cpu_assemble_reference(&weight_sums, &der_sums, l2_diag_reg, prior);
    // rowSize = 2 → 3 matrix cells + 2 RHS = 5 entries.
    assert_eq!(reference.len(), 5, "packed system length = rowSize*(rowSize+1)/2 + rowSize");

    let device = assemble_pairwise_system_host(&weight_sums, &der_sums, l2_diag_reg, prior)
        .expect("device pairwise system assembly must not error on a covered fixture");

    let divergence = max_abs_divergence(&device, &reference);
    println!(
        "[pairwise_deriv] 3-leaf device-vs-CPU max abs divergence = {divergence:e} \
         (device_backend_active = {})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device pairwise system diverged from CPU reference: {divergence:e} > {TOL:e}\n\
             device = {device:?}\nreference = {reference:?}"
        );
    }
}

/// Test 1b: a 4-leaf fixture (rowSize == 3: 6 matrix cells then 3 RHS) — exercises a deeper
/// lower-triangle packing than the minimal 3-leaf case.
#[test]
fn device_pairwise_system_matches_cpu_reference_four_leaves() {
    let weight_sums = vec![
        5.0, -1.0, -2.0, -2.0, //
        -1.0, 4.0, -1.5, -1.5, //
        -2.0, -1.5, 6.0, -2.5, //
        -2.0, -1.5, -2.5, 6.0,
    ];
    let der_sums = vec![1.0, -0.5, 0.25, -0.75];
    let l2_diag_reg = 1.0;
    let prior = PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT;

    let reference = cpu_assemble_reference(&weight_sums, &der_sums, l2_diag_reg, prior);
    assert_eq!(reference.len(), 6 + 3, "rowSize == 3 → 6 matrix cells + 3 RHS");

    let device = assemble_pairwise_system_host(&weight_sums, &der_sums, l2_diag_reg, prior)
        .expect("device pairwise system assembly must not error on the 4-leaf fixture");

    let divergence = max_abs_divergence(&device, &reference);
    println!("[pairwise_deriv] 4-leaf device-vs-CPU max abs divergence = {divergence:e}");
    if device_backend_active() {
        assert!(
            divergence <= TOL,
            "device pairwise system diverged: {divergence:e} > {TOL:e}"
        );
    }
}

/// Test 2: an uncovered / degenerate single-leaf system yields the EMPTY assembly (not a fabricated
/// result) — the "Ok(None)"-analog at the assembly level. `leaf_count == 1` has rowSize 0, so there
/// is no reduced matrix or RHS; its lone zero-averaged leaf delta is 0.
#[test]
fn single_leaf_system_is_empty_not_fabricated() {
    let weight_sums = vec![7.0]; // 1×1
    let der_sums = vec![0.9];
    let device = assemble_pairwise_system_host(&weight_sums, &der_sums, 3.0, 0.1)
        .expect("single-leaf assembly must succeed as an empty (no-op) result");
    assert!(
        device.is_empty(),
        "single-leaf pairwise system must assemble to EMPTY (no fabricated cells), got {device:?}"
    );
    // The CPU reference agrees the system is empty.
    assert!(cpu_assemble_reference(&weight_sums, &der_sums, 3.0, 0.1).is_empty());
}

/// Test 3: the empty-`n` (zero-leaf) case constructs the no-op launcher WITHOUT reading a 0-length
/// handle (the HIP fault guard) — `assemble_pairwise_system_host` short-circuits to an empty `Vec`
/// before any device read.
#[test]
fn empty_input_constructs_no_op_without_zero_len_read() {
    let device = assemble_pairwise_system_host(&[], &[], 3.0, 0.1)
        .expect("empty pairwise assembly must not error or read a 0-len handle");
    assert!(device.is_empty(), "empty input must yield an empty assembled system");
}
