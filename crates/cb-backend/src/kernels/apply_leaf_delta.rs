//! Self-oracle for the device-resident `apply_leaf_delta` approx update (GPUT-03): the
//! device `approx[i] += lr * leaf_values[leaf_of[i]]` must match a serial CPU reference
//! bit-exactly (integer-clean gather + one fused multiply-add; a generous f64 bound catches
//! a wrong update without pinning the signed-off epsilon — the GPU-06 epsilon is 7.6's job).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel
//! ([`crate::kernels::apply_leaf_delta_kernel`]) and the launcher
//! ([`crate::gpu_runtime::launch_apply_leaf_delta_into`]) are production code; ALL
//! `#[test]` + `.unwrap()`/indexing live here.
//!
//! The launcher is f64-channel on cpu/cuda/rocm and f32-channel on wgpu; the resident
//! session (which this update feeds) is f64-only (the der seam rejects wgpu — WR-02), so the
//! oracle is gated to the non-wgpu f64 channel it is exercised under in-env (rocm gfx1100),
//! mirroring the der-seam / gpu_tolerance cpu-vs-f64 discipline.

#![cfg(not(feature = "wgpu"))]

use cubecl::prelude::*;

use crate::gpu_runtime::launch_apply_leaf_delta_into;

/// Launch the resident approx update on the selected runtime and read the updated approx
/// back. Uploads `approx`/`leaf_of` as the (non-wgpu) f64/u32 device handles the launcher
/// consumes, then reads back the returned resident approx handle on the SAME client.
fn run_apply(approx: &[f64], leaf_of: &[u32], leaf_values: &[f64], lr: f64) -> Vec<f64> {
    let n = approx.len();
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let approx_h = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(leaf_of.to_vec()));

    let out = launch_apply_leaf_delta_into(&client, approx_h, leaf_of_h, leaf_values, lr, n).unwrap();
    let bytes = client.read_one(out).unwrap();
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

/// Serial CPU reference `approx[i] += lr * leaf_values[leaf_of[i]]` — the parity baseline.
fn cpu_apply(approx: &[f64], leaf_of: &[u32], leaf_values: &[f64], lr: f64) -> Vec<f64> {
    approx
        .iter()
        .zip(leaf_of.iter())
        .map(|(&a, &leaf)| a + lr * leaf_values[leaf as usize])
        .collect()
}

/// A generous run-stable f64 bound (NOT the signed-off epsilon): the update is a single
/// gather + fused multiply-add, so on f64 it is effectively exact.
const F64_TOL: f64 = 1e-12;

#[test]
fn apply_leaf_delta_matches_cpu_reference_depth1() {
    // depth == 1 (2 leaves): a mix of both leaves, positive/negative deltas + approx.
    let approx = vec![0.5, -1.0, 2.0, 0.0, 3.5, -2.5, 1.0, 0.25];
    let leaf_of = vec![0u32, 1, 0, 1, 1, 0, 1, 0];
    let leaf_values = vec![0.3_f64, -0.7];
    let lr = 0.1_f64;

    let dev = run_apply(&approx, &leaf_of, &leaf_values, lr);
    let cpu = cpu_apply(&approx, &leaf_of, &leaf_values, lr);

    assert_eq!(dev.len(), cpu.len(), "length mismatch");
    for (i, (d, c)) in dev.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (d - c).abs() <= F64_TOL,
            "apply_leaf_delta[{i}] device {d} vs cpu {c} (diff {})",
            (d - c).abs()
        );
    }
}

#[test]
fn apply_leaf_delta_grid_stride_large_n() {
    // n > CUBE_DIM so the grid-stride loop runs; deterministic pseudo-random gather.
    let n = 1000usize;
    let n_leaves = 4usize; // depth == 2 leaf count, still a valid gather test
    let approx: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 3.0).collect();
    let leaf_of: Vec<u32> = (0..n).map(|i| (i % n_leaves) as u32).collect();
    let leaf_values = vec![0.11_f64, -0.22, 0.33, -0.44];
    let lr = 0.3_f64;

    let dev = run_apply(&approx, &leaf_of, &leaf_values, lr);
    let cpu = cpu_apply(&approx, &leaf_of, &leaf_values, lr);

    for (i, (d, c)) in dev.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (d - c).abs() <= F64_TOL,
            "apply_leaf_delta[{i}] device {d} vs cpu {c}"
        );
    }
}

/// Serial CPU reference for the MULTI-OUTPUT block apply (Phase 13 Plan 06, GPUT-12, D-03):
/// `approx[d * n + i] += lr * leaf_block[leaf_of[i] * k + d]` over a dimension-major approx
/// buffer (length `k * n`) and a `leaf_count × k` row-major leaf block. At `k == 1` this is
/// byte-identical to [`cpu_apply`] (the scalar path — GPUT-14 / D-04 no-regression).
fn cpu_apply_block(approx: &[f64], leaf_of: &[u32], leaf_block: &[f64], lr: f64, k: usize) -> Vec<f64> {
    let n = approx.len() / k;
    let mut out = approx.to_vec();
    for d in 0..k {
        for i in 0..n {
            let leaf = leaf_of[i] as usize;
            out[d * n + i] += lr * leaf_block[leaf * k + d];
        }
    }
    out
}

/// Device block apply that ROUTES THROUGH the existing scalar multi-output apply layout: for each
/// dimension `d` it slices the dimension-major approx span `approx[d * n .. (d + 1) * n]`, extracts
/// the per-dimension leaf column `leaf_col[l] = leaf_block[l * k + d]`, and applies the SAME device
/// `apply_leaf_delta_kernel` the scalar path uses (multilogit.cu applies one dimension at a time).
/// The K slices are reassembled into the dimension-major buffer — exactly the CPU block layout.
fn run_apply_block(approx: &[f64], leaf_of: &[u32], leaf_block: &[f64], lr: f64, k: usize) -> Vec<f64> {
    let n = approx.len() / k;
    let n_leaves = leaf_block.len() / k;
    let mut out = vec![0.0_f64; approx.len()];
    for d in 0..k {
        let approx_dim: Vec<f64> = approx[d * n..(d + 1) * n].to_vec();
        let leaf_col: Vec<f64> = (0..n_leaves).map(|l| leaf_block[l * k + d]).collect();
        let updated = run_apply(&approx_dim, leaf_of, &leaf_col, lr);
        out[d * n..(d + 1) * n].copy_from_slice(&updated);
    }
    out
}

#[test]
fn apply_leaf_delta_block_matches_cpu_reference_k3() {
    // K=3 multi-output, depth-1 (2 leaves), n=4 objects. Dimension-major approx buffer (length K*n).
    let k = 3usize;
    let n = 4usize;
    let approx: Vec<f64> = (0..k * n).map(|x| (x as f64) * 0.1 - 0.5).collect();
    let leaf_of = vec![0u32, 1, 1, 0];
    // Leaf 0 = [0.10, 0.20, 0.30]; leaf 1 = [-0.40, -0.50, -0.60] (row-major per leaf).
    let leaf_block = vec![0.10, 0.20, 0.30, -0.40, -0.50, -0.60];
    let lr = 0.25_f64;

    let dev = run_apply_block(&approx, &leaf_of, &leaf_block, lr, k);
    let cpu = cpu_apply_block(&approx, &leaf_of, &leaf_block, lr, k);

    assert_eq!(dev.len(), cpu.len(), "block length mismatch");
    for (i, (d, c)) in dev.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (d - c).abs() <= F64_TOL,
            "block apply[{i}] device {d} vs cpu {c} (diff {})",
            (d - c).abs()
        );
    }
}

#[test]
fn apply_leaf_delta_block_k1_is_scalar_byte_unchanged() {
    // K == 1: the block apply collapses to the scalar path — byte-identical (GPUT-14 / D-04).
    let approx = vec![0.5, -1.0, 2.0, 0.0, 3.5, -2.5];
    let leaf_of = vec![0u32, 1, 0, 1, 1, 0];
    let leaf_values = vec![0.3_f64, -0.7]; // K == 1 ⇒ leaf_block == flat leaf vector.
    let lr = 0.1_f64;

    let block = run_apply_block(&approx, &leaf_of, &leaf_values, lr, 1);
    let scalar = cpu_apply(&approx, &leaf_of, &leaf_values, lr);

    for (i, (b, s)) in block.iter().zip(scalar.iter()).enumerate() {
        assert!(
            (b - s).abs() <= F64_TOL,
            "K=1 block apply must equal the scalar apply at index {i}: {b} vs {s}"
        );
    }
}

#[test]
fn apply_leaf_delta_empty_is_noop() {
    // n == 0: no launch, the resident approx handle is returned unchanged (Pitfall 5).
    // Do NOT read a 0-len handle back (HIP faults on a 0-len read); assert the launcher
    // constructs the no-op without faulting.
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let approx_h = client.empty(0);
    let leaf_of_h = client.empty(0);
    let res = launch_apply_leaf_delta_into(&client, approx_h, leaf_of_h, &[0.5_f64, -0.5], 0.1, 0);
    assert!(res.is_ok(), "empty apply_leaf_delta must construct without faulting");
}
