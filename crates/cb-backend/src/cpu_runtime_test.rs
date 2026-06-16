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
fn mae_gradients_match_host_reference() {
    // MAE / Quantile(alpha=0.5, delta=1e-6): der1 = sign(residual)*half-quantile
    // with a deadzone; der2 = 0. Includes an exact-tie object (approx==target).
    let approx = [0.0, 1.0, -2.5, 3.25, 7.0];
    let target = [1.0, 0.0, 2.5, -3.25, 7.0];

    let ders = CpuBackend.compute_gradients(Loss::Mae, &approx, &target).unwrap();

    assert_eq!(ders.der1.len(), approx.len());
    assert_eq!(ders.der2.len(), approx.len());
    for i in 0..approx.len() {
        let val = target[i] - approx[i];
        let expected = if val.abs() < 1e-6 {
            0.0
        } else if val > 0.0 {
            0.5
        } else {
            -0.5
        };
        assert!((ders.der1[i] - expected).abs() <= 1e-12, "i={i}");
        assert_eq!(ders.der2[i], 0.0);
    }
}

#[test]
fn quantile_gradients_match_host_reference_alpha07() {
    // Quantile{alpha=0.7, delta=1e-6}: der1 = |val|<delta ? 0 : (val>0 ? 0.7 :
    // -0.3); der2 = 0. Includes an exact-tie object (approx==target -> deadzone 0).
    let approx = [0.0, 1.0, -2.5, 3.25, 7.0];
    let target = [1.0, 0.0, 2.5, -3.25, 7.0];
    let alpha = 0.7;
    let delta = 1e-6;

    let ders = CpuBackend
        .compute_gradients(Loss::Quantile { alpha, delta }, &approx, &target)
        .unwrap();

    assert_eq!(ders.der1.len(), approx.len());
    assert_eq!(ders.der2.len(), approx.len());
    for i in 0..approx.len() {
        let val = target[i] - approx[i];
        let expected = if val.abs() < delta {
            0.0
        } else if val > 0.0 {
            alpha
        } else {
            -(1.0 - alpha)
        };
        assert!((ders.der1[i] - expected).abs() <= 1e-12, "i={i}");
        assert_eq!(ders.der2[i], 0.0);
    }
}

#[test]
fn quantile_alpha05_gradients_equal_mae() {
    // MAE == Quantile{alpha=0.5, delta=1e-6} at the dispatch level: the launched
    // der1/der2 must be bit-identical between Loss::Mae and Loss::Quantile{0.5}.
    let approx = [0.0, 1.0, -2.5, 3.25, 7.0, -0.5];
    let target = [1.0, 0.0, 2.5, -3.25, 7.0, 0.5];

    let mae = CpuBackend.compute_gradients(Loss::Mae, &approx, &target).unwrap();
    let q05 = CpuBackend
        .compute_gradients(
            Loss::Quantile {
                alpha: 0.5,
                delta: 1e-6,
            },
            &approx,
            &target,
        )
        .unwrap();

    assert_eq!(mae.der1, q05.der1, "Quantile{{0.5}} der1 must equal MAE der1");
    assert_eq!(mae.der2, q05.der2, "Quantile{{0.5}} der2 must equal MAE der2");
}

#[test]
fn logcosh_gradients_match_host_reference() {
    // der1 = -tanh(approx-target); der2 = -1/cosh(approx-target)^2.
    let approx = [0.0, 1.0, -2.5, 3.25, 7.0];
    let target = [1.0, 0.0, 2.5, -3.25, 7.0];

    let ders = CpuBackend
        .compute_gradients(Loss::LogCosh, &approx, &target)
        .unwrap();

    assert_eq!(ders.der1.len(), approx.len());
    assert_eq!(ders.der2.len(), approx.len());
    for i in 0..approx.len() {
        let r = approx[i] - target[i];
        assert!((ders.der1[i] - (-r.tanh())).abs() <= 1e-12, "der1 i={i}");
        let c = r.cosh();
        assert!((ders.der2[i] - (-1.0 / (c * c))).abs() <= 1e-12, "der2 i={i}");
    }
}

#[test]
fn lq_gradients_match_host_reference() {
    // q=2: der1 = 2*sign(t-a)*|a-t|; der2 = constant -2.
    let approx = [0.0, 1.0, -2.5, 3.25, 7.0];
    let target = [1.0, 0.0, 2.5, -3.25, 7.0];
    let q = 2.0;

    let ders = CpuBackend
        .compute_gradients(Loss::Lq { q }, &approx, &target)
        .unwrap();

    for i in 0..approx.len() {
        let abs_loss = (approx[i] - target[i]).abs();
        let sign = if target[i] - approx[i] > 0.0 { 1.0 } else { -1.0 };
        let want1 = q * sign * abs_loss.powf(q - 1.0);
        let want2 = -q * (q - 1.0) * (target[i] - approx[i]).abs().powf(q - 2.0);
        assert!((ders.der1[i] - want1).abs() <= 1e-12, "der1 i={i}");
        assert!((ders.der2[i] - want2).abs() <= 1e-12, "der2 i={i}");
    }
}

#[test]
fn huber_gradients_match_host_reference() {
    // diff=target-approx; der1=|diff|<delta?diff:sign*delta; der2=|diff|<delta?-1:0.
    let approx = [0.0, 0.0, 3.0, 0.0, 2.0];
    let target = [0.5, 3.0, 0.0, 1.0, 0.0]; // residuals 0.5(in), 3(out+), -3(out-), 1(==delta out), -2(out-)
    let delta = 1.0;

    let ders = CpuBackend
        .compute_gradients(Loss::Huber { delta }, &approx, &target)
        .unwrap();

    for i in 0..approx.len() {
        let diff = target[i] - approx[i];
        let want1 = if diff.abs() < delta {
            diff
        } else if diff > 0.0 {
            delta
        } else {
            -delta
        };
        let want2 = if diff.abs() < delta { -1.0 } else { 0.0 };
        assert!((ders.der1[i] - want1).abs() <= 1e-12, "der1 i={i}");
        assert!((ders.der2[i] - want2).abs() <= 1e-12, "der2 i={i}");
    }
}

#[test]
fn expectile_gradients_match_host_reference() {
    // e=target-approx; der1=(e>0)?2a*e:2(1-a)*e; der2=(e>0)?-2a:-2(1-a).
    let approx = [0.0, 2.0, 1.0, -1.0, 5.0];
    let target = [2.0, 0.0, 1.0, 3.0, 5.0]; // e = +2, -2, 0, +4, 0
    let alpha = 0.3;

    let ders = CpuBackend
        .compute_gradients(Loss::Expectile { alpha }, &approx, &target)
        .unwrap();

    for i in 0..approx.len() {
        let e = target[i] - approx[i];
        let (want1, want2) = if e > 0.0 {
            (2.0 * alpha * e, -2.0 * alpha)
        } else {
            (2.0 * (1.0 - alpha) * e, -2.0 * (1.0 - alpha))
        };
        assert!((ders.der1[i] - want1).abs() <= 1e-12, "der1 i={i}");
        assert!((ders.der2[i] - want2).abs() <= 1e-12, "der2 i={i}");
    }
}

// --- Wave-2 positive-domain / link losses (Plan 06.1-02) -------------------

/// Poisson kernels: der1 = `target - exp(approx)`, der2 = `-exp(approx)` over the
/// RAW approx (inline exp). Moderate approx range so exp stays finite
/// (T-06.1.02-01).
#[test]
fn poisson_gradients_match_host_reference() {
    let approx = [0.0, 0.5, -1.0, 1.5, 2.0, -0.25, 1.0];
    let target = [3.0, 1.0, 4.0, 2.0, 5.0, 1.5, 2.0];

    let ders = CpuBackend
        .compute_gradients(Loss::Poisson, &approx, &target)
        .unwrap();

    for i in 0..approx.len() {
        let e = approx[i].exp();
        assert!((ders.der1[i] - (target[i] - e)).abs() <= 1e-12, "poisson der1 i={i}");
        assert!((ders.der2[i] - (-e)).abs() <= 1e-12, "poisson der2 i={i}");
    }
}

/// Tweedie{variance_power=1.5} kernels: exp INSIDE the der over the RAW approx.
#[test]
fn tweedie_gradients_match_host_reference() {
    let p = 1.5;
    let approx = [0.0, 0.5, -1.0, 1.0, 2.0, -0.5, 0.25];
    let target = [3.0, 1.0, 4.0, 2.0, 5.0, 1.5, 2.0];

    let ders = CpuBackend
        .compute_gradients(Loss::Tweedie { variance_power: p }, &approx, &target)
        .unwrap();

    for i in 0..approx.len() {
        let a = approx[i];
        let t = target[i];
        let e1 = ((1.0 - p) * a).exp();
        let e2 = ((2.0 - p) * a).exp();
        let want1 = t * e1 - e2;
        let want2 = t * (1.0 - p) * e1 - (2.0 - p) * e2;
        assert!((ders.der1[i] - want1).abs() <= 1e-12, "tweedie der1 i={i}");
        assert!((ders.der2[i] - want2).abs() <= 1e-12, "tweedie der2 i={i}");
    }
}

/// MAPE kernel: der1 = `sign(target-approx)/max(1,|target|)`, der2 = 0
/// (constant). Exercises the |target|<1 vs >1 divisor floor and the sign branch.
#[test]
fn mape_gradients_match_host_reference() {
    let approx = [2.0, 7.0, 0.0, 2.0, 3.0, -1.0, 0.5];
    let target = [5.0, 5.0, 0.5, 0.5, 3.0, 4.0, 0.5];

    let ders = CpuBackend
        .compute_gradients(Loss::Mape, &approx, &target)
        .unwrap();

    for i in 0..approx.len() {
        let denom = 1.0_f64.max(target[i].abs());
        let sign = if target[i] - approx[i] > 0.0 { 1.0 } else { -1.0 };
        let want1 = sign / denom;
        assert!((ders.der1[i] - want1).abs() <= 1e-12, "mape der1 i={i}");
        assert!(ders.der2[i].abs() <= 1e-12, "mape der2 i={i} must be 0");
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
