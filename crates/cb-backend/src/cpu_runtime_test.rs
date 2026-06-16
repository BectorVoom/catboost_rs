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

    let ders = CpuBackend.compute_gradients(&Loss::Rmse, &approx, &target, 1).unwrap();

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
        .compute_gradients(&Loss::Logloss, &approx, &target, 1)
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

    let ders = CpuBackend.compute_gradients(&Loss::Mae, &approx, &target, 1).unwrap();

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
        .compute_gradients(&Loss::Quantile { alpha, delta }, &approx, &target, 1)
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

    let mae = CpuBackend.compute_gradients(&Loss::Mae, &approx, &target, 1).unwrap();
    let q05 = CpuBackend
        .compute_gradients(
            &Loss::Quantile {
                alpha: 0.5,
                delta: 1e-6,
            },
            &approx,
            &target,
            1,
        )
        .unwrap();

    assert_eq!(mae.der1, q05.der1, "Quantile{{0.5}} der1 must equal MAE der1");
    assert_eq!(mae.der2, q05.der2, "Quantile{{0.5}} der2 must equal MAE der2");
}

#[test]
fn quantile_kernel_matches_scalar_at_deadzone_boundary() {
    // CR-01 regression: a residual EXACTLY equal to `delta` is OUTSIDE the
    // scalar `|val| < delta` deadzone, so it must return the signed quantile
    // weight, NOT 0. Construct `val == +delta` (approx=0, target=delta) and
    // `val == -delta` (approx=delta, target=0) and assert the launched kernel
    // matches the scalar `quantile_der1`/`mae_der1` reference bit-for-bit.
    let delta = 1e-6;
    let alpha = 0.7;
    // [+delta, -delta] residuals plus a strictly-inside-deadzone control.
    let approx = [0.0, delta, 0.0];
    let target = [delta, 0.0, 0.0];

    let q = CpuBackend
        .compute_gradients(&Loss::Quantile { alpha, delta }, &approx, &target, 1)
        .unwrap();
    for i in 0..approx.len() {
        let expected = cb_compute::quantile_der1(approx[i], target[i], alpha, delta);
        assert!(
            (q.der1[i] - expected).abs() <= 1e-12,
            "quantile boundary i={i}: kernel={} scalar={expected}",
            q.der1[i]
        );
    }

    // MAE (Quantile{0.5}) must likewise honor the boundary.
    let m = CpuBackend.compute_gradients(&Loss::Mae, &approx, &target, 1).unwrap();
    for i in 0..approx.len() {
        let expected = cb_compute::mae_der1(approx[i], target[i]);
        assert!(
            (m.der1[i] - expected).abs() <= 1e-12,
            "mae boundary i={i}: kernel={} scalar={expected}",
            m.der1[i]
        );
    }
}

#[test]
fn logcosh_gradients_match_host_reference() {
    // der1 = -tanh(approx-target); der2 = -1/cosh(approx-target)^2.
    let approx = [0.0, 1.0, -2.5, 3.25, 7.0];
    let target = [1.0, 0.0, 2.5, -3.25, 7.0];

    let ders = CpuBackend
        .compute_gradients(&Loss::LogCosh, &approx, &target, 1)
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
        .compute_gradients(&Loss::Lq { q }, &approx, &target, 1)
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
        .compute_gradients(&Loss::Huber { delta }, &approx, &target, 1)
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
        .compute_gradients(&Loss::Expectile { alpha }, &approx, &target, 1)
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
        .compute_gradients(&Loss::Poisson, &approx, &target, 1)
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
        .compute_gradients(&Loss::Tweedie { variance_power: p }, &approx, &target, 1)
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
        .compute_gradients(&Loss::Mape, &approx, &target, 1)
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
        .compute_gradients(&Loss::Rmse, &approx, &target, 1)
        .is_err());
}

#[test]
fn empty_input_yields_empty_derivatives() {
    let ders = CpuBackend.compute_gradients(&Loss::Rmse, &[], &[], 1).unwrap();
    assert!(ders.der1.is_empty());
    assert!(ders.der2.is_empty());
}

// --- Phase 6.2 Wave-0 N-dim refactor: dim=1 byte-identity + per-dim dispatch ---

/// D-04 anchor (RESEARCH Pitfall 1): at `approx_dimension == 1` the dim-major
/// buffer is the historical scalar buffer, and the per-dimension loop runs exactly
/// once over `approx[0..n]`. The output MUST be BYTE-IDENTICAL (exact `==`, NOT a
/// `<= 1e-5` tolerance) to the scalar `Rmse` host reference computed in the same
/// order, and deterministic for `Logloss` — proving the widening perturbs nothing
/// at dim=1.
#[test]
fn dim1_is_byte_identical_to_scalar_path() {
    let approx = [0.0, 1.0, -2.5, 3.25, 10.0, -0.5, 7.0];
    let target = [1.0, 0.0, 2.5, -3.25, 4.0, 0.5, 7.0];

    // Rmse: der1 = target - approx (exact), der2 = -1.0 (exact).
    let rmse = CpuBackend
        .compute_gradients(&Loss::Rmse, &approx, &target, 1)
        .unwrap();
    assert_eq!(rmse.der1.len(), approx.len());
    for i in 0..approx.len() {
        assert_eq!(rmse.der1[i], target[i] - approx[i], "rmse der1 i={i}");
        assert_eq!(rmse.der2[i], -1.0, "rmse der2 i={i}");
    }

    // Logloss: der1 = target - sigmoid(approx), der2 = -p*(1-p), the SAME launch
    // path as the scalar arm — two dim-1 launches must agree bit-for-bit.
    let ll_a = CpuBackend
        .compute_gradients(&Loss::Logloss, &approx, &target, 1)
        .unwrap();
    let ll_b = CpuBackend
        .compute_gradients(&Loss::Logloss, &approx, &target, 1)
        .unwrap();
    assert_eq!(ll_a.der1, ll_b.der1, "logloss der1 dim=1 must be deterministic");
    assert_eq!(ll_a.der2, ll_b.der2, "logloss der2 dim=1 must be deterministic");
}

/// For a separable loss at `approx_dimension == k > 1`, the backend launches the
/// per-loss kernel once per dimension over `approx[d*n..d*n+n]` and concatenates
/// into a dim-major output of length `k*n`. Each dimension's slice must produce
/// exactly the same der1/der2 as an independent dim=1 call on that slice (the
/// outer loop introduces no cross-dimension coupling for separable losses).
#[test]
fn multi_dim_separable_concatenates_per_dimension() {
    // dim=3, n=4. Distinct per-dimension approx blocks; one shared target column.
    let n = 4;
    let dim = 3;
    let approx: Vec<f64> = vec![
        // d=0
        0.0, 1.0, -2.0, 3.0, //
        // d=1
        0.5, -0.5, 2.5, -3.5, //
        // d=2
        1.5, -1.5, 0.25, -0.25,
    ];
    let target = [1.0, 0.0, 2.0, -3.0];
    assert_eq!(approx.len(), dim * n);

    let nd = CpuBackend
        .compute_gradients(&Loss::Rmse, &approx, &target, dim)
        .unwrap();
    assert_eq!(nd.der1.len(), dim * n);
    assert_eq!(nd.der2.len(), dim * n);

    for d in 0..dim {
        let approx_d = &approx[d * n..d * n + n];
        let scalar = CpuBackend
            .compute_gradients(&Loss::Rmse, approx_d, &target, 1)
            .unwrap();
        // Byte-identical per-dimension block (exact ==).
        assert_eq!(&nd.der1[d * n..d * n + n], scalar.der1.as_slice(), "der1 d={d}");
        assert_eq!(&nd.der2[d * n..d * n + n], scalar.der2.as_slice(), "der2 d={d}");
    }
}

/// Shape validation (T-6.2-01a): a non-divisible `approx.len()` vs
/// `approx_dimension`, a zero dimension, or an inconsistent `target` length all
/// return a typed `CbError` (no panic, no `unwrap` in production).
#[test]
fn ndim_shape_mismatch_is_error_not_panic() {
    let approx = [0.0, 1.0, 2.0, 3.0, 4.0]; // len 5, not divisible by 2
    let target = [0.0, 1.0];
    assert!(CpuBackend
        .compute_gradients(&Loss::Rmse, &approx, &target, 2)
        .is_err());

    // Zero dimension is rejected.
    assert!(CpuBackend
        .compute_gradients(&Loss::Rmse, &approx, &target, 0)
        .is_err());

    // target length inconsistent with n = approx.len()/dim (4/2 = 2 != 3).
    let approx4 = [0.0, 1.0, 2.0, 3.0];
    let target3 = [0.0, 1.0, 2.0];
    assert!(CpuBackend
        .compute_gradients(&Loss::Rmse, &approx4, &target3, 2)
        .is_err());
}

/// MultiQuantile (Wave 3, D-6.2-05): K INDEPENDENT quantile dimensions. The
/// backend der for each dimension `d` must equal the scalar `quantile_der1`
/// applied per dimension with THAT dimension's level `alpha[d]` (the shared
/// `delta`), and `der2` must be the constant `0` across every dimension
/// (QUANTILE_DER2 = 0 -> Exact leaf). The target stays PER-OBJECT length `n`.
#[test]
fn multiquantile_der_equals_per_dimension_quantile_der_and_der2_zero() {
    let n = 4;
    let dim = 3;
    // Distinct per-dimension approx blocks (dim-major); one shared per-object target.
    let approx: Vec<f64> = vec![
        // d=0
        0.0, 1.0, -2.0, 3.0, //
        // d=1
        0.5, -0.5, 2.5, -3.5, //
        // d=2
        1.5, -1.5, 0.25, -0.25,
    ];
    let target = [1.0, 0.0, 2.0, -3.0];
    let alpha = vec![0.1, 0.5, 0.9];
    let delta = 1e-6_f64;
    assert_eq!(approx.len(), dim * n);

    let nd = CpuBackend
        .compute_gradients(
            &Loss::MultiQuantile {
                alpha: alpha.clone(),
                delta,
            },
            &approx,
            &target,
            dim,
        )
        .unwrap();
    assert_eq!(nd.der1.len(), dim * n);
    assert_eq!(nd.der2.len(), dim * n);

    // Host reference: the scalar quantile_der1 per dimension at alpha[d].
    let quantile_der1 = |a: f64, t: f64, alpha: f64, delta: f64| -> f64 {
        let val = t - a;
        if val.abs() < delta {
            0.0
        } else if val > 0.0 {
            alpha
        } else {
            -(1.0 - alpha)
        }
    };
    for d in 0..dim {
        for i in 0..n {
            let idx = d * n + i;
            let want1 = quantile_der1(approx[idx], target[i], alpha[d], delta);
            assert!((nd.der1[idx] - want1).abs() <= 1e-12, "der1 d={d} i={i}");
            // der2 = 0 across every dimension.
            assert_eq!(nd.der2[idx], 0.0, "der2 must be 0 at d={d} i={i}");
        }
    }
}

/// MultiQuantile at `dim == 1` with `alpha = [a]` is byte-identical to the scalar
/// `Loss::Quantile { alpha: a, delta }` backend path (the degenerate-equivalence
/// anchor at the gradient level — D-6.2-05).
#[test]
fn multiquantile_dim1_equals_scalar_quantile() {
    let approx = [0.0_f64, 1.0, -2.0, 3.0, 0.5];
    let target = [1.0_f64, 0.0, 2.0, -3.0, 0.5];
    let alpha = 0.7_f64;
    let delta = 1e-6_f64;

    let mq = CpuBackend
        .compute_gradients(
            &Loss::MultiQuantile {
                alpha: vec![alpha],
                delta,
            },
            &approx,
            &target,
            1,
        )
        .unwrap();
    let scalar = CpuBackend
        .compute_gradients(&Loss::Quantile { alpha, delta }, &approx, &target, 1)
        .unwrap();
    assert_eq!(mq.der1, scalar.der1, "der1 must be byte-identical at dim=1");
    assert_eq!(mq.der2, scalar.der2, "der2 must be byte-identical at dim=1");
}

/// MultiQuantile shape validation: `alpha.len()` must equal `approx_dimension`
/// (the backend rejects a mismatch with a typed `CbError`, no panic).
#[test]
fn multiquantile_alpha_dim_mismatch_is_error_not_panic() {
    let approx = [0.0_f64, 1.0, 2.0, 3.0]; // dim=2, n=2
    let target = [0.0_f64, 1.0];
    // alpha.len()=3 != approx_dimension=2.
    assert!(CpuBackend
        .compute_gradients(
            &Loss::MultiQuantile {
                alpha: vec![0.1, 0.5, 0.9],
                delta: 1e-6,
            },
            &approx,
            &target,
            2,
        )
        .is_err());
}
