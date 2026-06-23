//! Unit tests for [`crate::gpu_backend::GpuBackend`] (08-08). Source/test
//! separation (CLAUDE.md): this lives in a dedicated `*_test.rs` declared
//! `#[cfg(test)] mod gpu_backend_test;` under the GPU-feature cfg in `lib.rs`, so it
//! is NEVER part of the default cpu workspace test (which has no `GpuBackend` symbol).
//!
//! The parity tests run under whatever GPU feature is active in-env (rocm on the
//! gfx1100 box). Under `wgpu` the der seam typed-rejects f64 (WR-02 — WGSL has no
//! f64), so the on-device der1/der2 launches return an error rather than a buffer;
//! the parity tests are guarded to SKIP (return early) under wgpu rather than
//! falsely failing, while the typed-error tests still run on every GPU backend.

use cb_compute::{Loss, Runtime};

use crate::gpu_backend::GpuBackend;

/// Phase-7 GPU parity tolerance (D-04 / Phase 7.6 sign-off): the GPU der path is
/// accepted within <=1e-4 of the host `cb-compute::loss` baseline (frequently
/// bit-exact in-env on gfx1100).
const GPU_TOL: f64 = 1e-4;

/// `true` under the wgpu backend, where the f64 der seam is typed-rejected (WR-02).
/// The parity tests skip on wgpu (the seam cannot produce f64 derivatives there);
/// they exercise the real device math on rocm/cuda.
const WGPU_F64_UNSUPPORTED: bool = cfg!(feature = "wgpu");

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff <= GPU_TOL,
            "{label}: object {i} GPU der {g} vs host baseline {w} (|diff|={diff} > {GPU_TOL})"
        );
    }
}

/// GpuBackend RMSE der1/der2 match the host `cb-compute::loss` baseline within the
/// Phase-7 GPU tolerance: der1 = target - approx, der2 = -1.0.
#[test]
fn gpu_backend_rmse_matches_host_baseline() {
    if WGPU_F64_UNSUPPORTED {
        return; // wgpu has no f64 der seam (WR-02) — skip the on-device parity check.
    }
    let approx = [0.5_f64, -1.0, 2.0, 0.0, -0.25];
    let target = [1.0_f64, -0.5, 1.5, 0.5, 0.0];

    let want_der1: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::rmse_der1(a, t))
        .collect();
    let want_der2: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::rmse_der2(a, t))
        .collect();

    let ders = GpuBackend
        .compute_gradients(&Loss::Rmse, &approx, &target, 1)
        .expect("GpuBackend RMSE compute_gradients should succeed in-env");

    assert_close("RMSE der1", &ders.der1, &want_der1);
    assert_close("RMSE der2", &ders.der2, &want_der2);
}

/// GpuBackend Logloss der1/der2 match the host baseline: der1 = target -
/// sigmoid(approx), der2 = -p*(1-p).
#[test]
fn gpu_backend_logloss_matches_host_baseline() {
    if WGPU_F64_UNSUPPORTED {
        return;
    }
    let approx = [0.5_f64, -1.0, 2.0, 0.0, -0.25];
    let target = [1.0_f64, 0.0, 1.0, 0.0, 1.0];

    let want_der1: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::logloss_der1(a, t))
        .collect();
    let want_der2: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| cb_compute::logloss_der2(a, t))
        .collect();

    let ders = GpuBackend
        .compute_gradients(&Loss::Logloss, &approx, &target, 1)
        .expect("GpuBackend Logloss compute_gradients should succeed in-env");

    assert_close("Logloss der1", &ders.der1, &want_der1);
    assert_close("Logloss der2", &ders.der2, &want_der2);
}

/// An empty input short-circuits to empty derivatives (no launch), on every backend
/// including wgpu (no f64 launch is reached).
#[test]
fn gpu_backend_empty_input_short_circuits() {
    let ders = GpuBackend
        .compute_gradients(&Loss::Rmse, &[], &[], 1)
        .expect("empty input should short-circuit to empty derivatives");
    assert!(ders.der1.is_empty());
    assert!(ders.der2.is_empty());
}

/// A loss with no Phase-7.2 GPU der kernel (LogCosh here) returns a typed error
/// naming it unsupported on the GPU backend — never a silent wrong gradient, never a
/// panic. Runs on every GPU backend (the reject is reached before any launch).
#[test]
fn gpu_backend_unsupported_loss_returns_typed_error() {
    let approx = [0.5_f64, -1.0, 2.0];
    let target = [1.0_f64, -0.5, 1.5];

    let err = GpuBackend
        .compute_gradients(&Loss::LogCosh, &approx, &target, 1)
        .expect_err("LogCosh has no GPU der kernel — must return a typed error");

    let msg = format!("{err}");
    assert!(
        msg.contains("not yet supported on the GPU backend"),
        "unexpected error message: {msg}"
    );
}

/// A zero `approx_dimension` is rejected with a typed length-mismatch error (mirrors
/// CpuBackend), on every backend (validated before any launch).
#[test]
fn gpu_backend_zero_dimension_rejected() {
    let approx = [0.5_f64, -1.0];
    let target = [1.0_f64, -0.5];
    let err = GpuBackend
        .compute_gradients(&Loss::Rmse, &approx, &target, 0)
        .expect_err("approx_dimension == 0 must be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("approx_dimension"), "unexpected error: {msg}");
}
