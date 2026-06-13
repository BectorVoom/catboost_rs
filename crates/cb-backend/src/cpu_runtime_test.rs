//! Tests for the `CpuBackend` impl of `cb_compute::Runtime`: the launched
//! kernels' UN-reduced per-object derivatives must match a host reference for
//! both RMSE and Logloss (D-02 — the kernel does elementwise work only).

use cb_compute::{Loss, Runtime};

use crate::cpu_runtime::CpuBackend;

fn sigmoid_host(x: f64) -> f64 {
    let e = x.exp();
    1.0 - 1.0 / (1.0 + e)
}

#[test]
fn rmse_gradients_match_host_reference() {
    let approx = [0.0, 1.0, -2.5, 3.25, 10.0, -0.5, 7.0];
    let target = [1.0, 0.0, 2.5, -3.25, 4.0, 0.5, 7.0];

    let ders = CpuBackend.compute_gradients(Loss::Rmse, &approx, &target).unwrap();

    assert_eq!(ders.der1.len(), approx.len());
    assert_eq!(ders.der2.len(), approx.len());
    for i in 0..approx.len() {
        assert!((ders.der1[i] - (target[i] - approx[i])).abs() <= 1e-12);
        assert!((ders.der2[i] - (-1.0)).abs() <= 1e-12);
    }
}

#[test]
fn logloss_gradients_match_host_reference() {
    let approx = [0.0, 0.5, -1.3, 2.0, -3.0];
    let target = [1.0, 0.0, 1.0, 0.0, 1.0];

    let ders = CpuBackend
        .compute_gradients(Loss::Logloss, &approx, &target)
        .unwrap();

    assert_eq!(ders.der1.len(), approx.len());
    for i in 0..approx.len() {
        let p = sigmoid_host(approx[i]);
        assert!((ders.der1[i] - (target[i] - p)).abs() <= 1e-12);
        assert!((ders.der2[i] - (-p * (1.0 - p))).abs() <= 1e-12);
    }
}

#[test]
fn length_mismatch_is_error_not_panic() {
    let approx = [0.0, 1.0];
    let target = [1.0];
    assert!(CpuBackend
        .compute_gradients(Loss::Rmse, &approx, &target)
        .is_err());
}

#[test]
fn empty_input_yields_empty_derivatives() {
    let ders = CpuBackend.compute_gradients(Loss::Rmse, &[], &[]).unwrap();
    assert!(ders.der1.is_empty());
    assert!(ders.der2.is_empty());
}
