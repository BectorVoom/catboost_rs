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

use cb_compute::{EScoreFunction, Loss, Runtime};

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

    let ders = GpuBackend::default()
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

    let ders = GpuBackend::default()
        .compute_gradients(&Loss::Logloss, &approx, &target, 1)
        .expect("GpuBackend Logloss compute_gradients should succeed in-env");

    assert_close("Logloss der1", &ders.der1, &want_der1);
    assert_close("Logloss der2", &ders.der2, &want_der2);
}

/// An empty input short-circuits to empty derivatives (no launch), on every backend
/// including wgpu (no f64 launch is reached).
#[test]
fn gpu_backend_empty_input_short_circuits() {
    let ders = GpuBackend::default()
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

    let err = GpuBackend::default()
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
    let err = GpuBackend::default()
        .compute_gradients(&Loss::Rmse, &approx, &target, 0)
        .expect_err("approx_dimension == 0 must be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("approx_dimension"), "unexpected error: {msg}");
}

// ===========================================================================
// GPUT-02/03/04: the device grow-tree seam lifecycle over the GpuBackend RefCell session.
// Under wgpu the resident der seam typed-rejects f64 (WR-02), so the covered-path grow is
// skipped there; the coverage gate (Ok(false)/Ok(None)) is exercised on every GPU backend.
// ===========================================================================

/// A small feature-major fixture: feature 0's bins climb monotonically with the object
/// index (a clear best border); returns `(target, weight, cindex)`.
fn seam_fixture(n: usize, n_features: usize, n_bins: usize) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let target: Vec<f64> = (0..n).map(|k| (k as f64) - (n as f64) / 2.0).collect();
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
    (target, weight, cindex)
}

/// GPUT-04: the covered depth-1 RMSE/Plain/fold-1/Cosine config opens a session
/// (`begin -> Ok(true)`), grows a device tree (`grow_tree_on_device -> Ok(Some(tree))` with a
/// depth-1 split + per-object leaf_of), and tears it down (`end -> Ok(())`). rocm/cuda only
/// (the resident der seam is f64; wgpu is skipped — WR-02).
#[test]
fn gpu_backend_device_grow_lifecycle_covered() {
    if WGPU_F64_UNSUPPORTED {
        return;
    }
    let n = 64usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let (target, weight, cindex) = seam_fixture(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, cb_core::sum_f64(&weight), n);

    let backend = GpuBackend::default();
    let covered = backend
        .begin_device_training(
            &Loss::Rmse, 1, true, 1, EScoreFunction::Cosine, &cindex, &weight, n, n_features,
            n_bins, 0.3, scaled_l2,
        )
        .expect("begin must not error on a covered config");
    assert!(covered, "depth1/RMSE/Plain/fold1/Cosine must open a device session (Ok(true))");

    let approx = vec![0.0_f64; n];
    let tree = backend
        .grow_tree_on_device(&approx, &target)
        .expect("grow_tree_on_device must not error over an open session")
        .expect("an open session must grow a device tree (Ok(Some))");
    assert_eq!(tree.splits.len(), 1, "depth-1 device tree must have exactly one split");
    assert_eq!(tree.leaf_of.len(), n, "device tree leaf_of must be populated length n");
    assert_eq!(tree.leaf_values.len(), 2, "depth-1 device tree must have 2 leaf values");

    backend.end_device_training().expect("end must release the session cleanly");

    // After end, the session is gone -> the grow seam falls back to Ok(None).
    let after = backend
        .grow_tree_on_device(&approx, &target)
        .expect("grow after end must not error");
    assert!(after.is_none(), "after end_device_training the grow seam must return Ok(None)");
}

/// GPUT-04: an UNCOVERED config declines to the CPU path — `begin -> Ok(false)` and the grow
/// seam returns `Ok(None)` (no session stored). Runs on every GPU backend (no device grow).
#[test]
fn gpu_backend_device_grow_uncovered_falls_back() {
    let n = 32usize;
    let n_features = 2usize;
    let n_bins = 32usize;
    let (target, weight, cindex) = seam_fixture(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, cb_core::sum_f64(&weight), n);

    let backend = GpuBackend::default();
    // depth == 2 is not covered (the per-partition histogram forward dependency).
    let covered = backend
        .begin_device_training(
            &Loss::Rmse, 2, true, 1, EScoreFunction::Cosine, &cindex, &weight, n, n_features,
            n_bins, 0.3, scaled_l2,
        )
        .expect("begin must not error while classifying coverage");
    assert!(!covered, "depth>1 must decline the device path (Ok(false))");

    let approx = vec![0.0_f64; n];
    let tree = backend
        .grow_tree_on_device(&approx, &target)
        .expect("grow must not error with no open session");
    assert!(tree.is_none(), "uncovered config must route through Ok(None) -> CPU fallback");

    backend.end_device_training().expect("end is a no-op with no open session");
}
