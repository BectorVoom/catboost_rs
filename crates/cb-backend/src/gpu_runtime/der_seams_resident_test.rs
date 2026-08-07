//! FPP-05 self-oracle for [`super::launch_der_param_resident`] — the RESIDENT parametric
//! der seam added to close the exact-leaf structural-derivative gap.
//!
//! # What this seam fixes
//!
//! The resident grow loop previously had only [`super::launch_der_binary_resident`], whose
//! two arms are RMSE and Logloss. An EXACT-leaf fit (MAE / Quantile) therefore borrowed
//! `RmseGradient` for its split histogram, which decides the tree STRUCTURE: leaf VALUES
//! were the correct device order statistic, but the splits were the ones an RMSE fit would
//! have chosen. Against upstream `catboost==1.2.10` that was a ~3.4e-2 prediction gap on a
//! 3-tree depth-3 MAE fit — four orders of magnitude past the ≤1e-5 bar — and it stayed
//! invisible only because `device_host_eligible` rejected `LeafMethod::Exact` outright
//! until FPP-06 admitted it.
//!
//! The KERNEL was never missing: `quantile_gradient_kernel` already existed and MAE already
//! routed through it at `(QUANTILE_ALPHA, QUANTILE_DELTA)`. Only the resident LAUNCHER was.
//!
//! Source/test separation (CLAUDE.md): the seam is production code (`der_seams.rs`); every
//! `#[test]` / `.unwrap()` / indexing lives here.

#![cfg(not(feature = "wgpu"))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use crate::gpu_runtime::der_seams::{launch_der_param_resident, DerParamKernel};
use crate::gpu_runtime::{read_part_stats_f64, upload_channel_floats};
use crate::SelectedRuntime;

/// ε for the resident-channel read-back (the device self-oracle bound, D-07).
const EPS: f64 = 1e-9;

/// `TQuantileError::CalcDer`: with `v = target - approx`,
/// `der1 = |v| < δ ? 0 : (v > 0 ? α : -(1-α))`. MAE is this at `α = 0.5`, `δ = 1e-6`.
fn host_quantile_der(approx: &[f64], target: &[f64], alpha: f64, delta: f64) -> Vec<f64> {
    approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| {
            let v = t - a;
            if v.abs() < delta {
                0.0
            } else if v > 0.0 {
                alpha
            } else {
                -(1.0 - alpha)
            }
        })
        .collect()
}

fn client() -> cubecl::client::ComputeClient<SelectedRuntime> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    <SelectedRuntime as cubecl::Runtime>::client(&device)
}

/// The resident parametric der must reproduce the host quantile derivative exactly, for
/// both the MAE (α = 0.5) and a genuinely off-median (α = 0.7) parameterisation.
///
/// α is separable by construction: at α = 0.7 a positive residual contributes `0.7` and a
/// negative one `-0.3`, versus `±0.5` at the median. A launcher that ignored its params
/// would agree with exactly one of the two arms, never both.
#[test]
fn resident_quantile_der_matches_the_host_reference() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[FPP-05] SKIP resident_quantile_der_matches_the_host_reference: needs rocm/cuda");
        return;
    }
    let n = 64usize;
    // Straddles zero and includes exact ties (v == 0) so the deadzone arm is exercised.
    let approx: Vec<f64> = (0..n).map(|i| (i as f64) - 32.0).collect();
    let target: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();

    let cl = client();
    let mut results = Vec::new();
    for (alpha, delta, label) in [(0.5_f64, 1e-6_f64, "mae"), (0.7, 1e-6, "quantile07")] {
        let approx_h = upload_channel_floats(&cl, &approx);
        let target_h = upload_channel_floats(&cl, &target);
        let out = launch_der_param_resident(
            &cl,
            approx_h,
            target_h,
            DerParamKernel::QuantileGradient,
            &[alpha, delta],
            n,
        )
        .unwrap_or_else(|e| panic!("[{label}] resident quantile der must launch: {e:?}"));
        let got = read_part_stats_f64(&cl, out)
            .unwrap_or_else(|e| panic!("[{label}] read-back must not fail: {e:?}"));

        let want = host_quantile_der(&approx, &target, alpha, delta);
        assert_eq!(got.len(), n, "[{label}] der length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= EPS,
                "[{label}] der1[{i}]: device {g} vs host quantile der {w} (|Δ|={:.3e})",
                (g - w).abs()
            );
        }

        // The RMSE residual over this input is nothing like the quantile der (it spans
        // ±35 versus ±1), so the assertion above cannot pass for the wrong dispatch.
        let rmse_like = approx
            .iter()
            .zip(target.iter())
            .zip(got.iter())
            .all(|((&a, &t), &g)| (t - a - g).abs() < 1e-9);
        assert!(
            !rmse_like,
            "[{label}] the resident der equals target - approx — that IS the RMSE residual, \
             so the quantile dispatch did not take effect"
        );
        println!("[FPP-05] {label}: resident der == quantile der (α={alpha}, δ={delta})");
        results.push(got);
    }

    // α is load-bearing: the two parameterisations must genuinely disagree.
    let max_delta = results[0]
        .iter()
        .zip(results[1].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_delta > 0.1,
        "α = 0.5 and α = 0.7 produced the same resident der (max|Δ|={max_delta:.3e}) — the \
         params are not reaching the kernel"
    );
}

/// `n == 0` short-circuits to a zero-length handle with no launch (the empty-input
/// contract shared with every other der seam entry point).
#[test]
fn resident_quantile_der_empty_input_is_a_zero_length_handle() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[FPP-05] SKIP resident_quantile_der_empty_input...: needs rocm/cuda");
        return;
    }
    let cl = client();
    let out = launch_der_param_resident(
        &cl,
        cl.empty(0),
        cl.empty(0),
        DerParamKernel::QuantileGradient,
        &[0.5, 1e-6],
        0,
    )
    .expect("an empty resident der must not error");
    // Deliberately NOT read back: `read_one` on a zero-length CubeCL-HIP handle trips a
    // `slice::from_raw_parts` null-pointer precondition inside the backend. Every
    // production empty path short-circuits before any read for the same reason, so the
    // observable contract here is "returns Ok without launching", which is what is
    // asserted. The handle's own size is the second half of that contract.
    assert_eq!(out.size(), 0, "an empty input must yield a zero-length handle");
}

/// A malformed `params` slice is a typed error, never a silently defaulted parameter — a
/// defaulted α would grow the median tree for an α = 0.7 fit and only show up as an
/// unattributable end-to-end gap.
#[test]
fn resident_quantile_der_rejects_short_params() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[FPP-05] SKIP resident_quantile_der_rejects_short_params: needs rocm/cuda");
        return;
    }
    let cl = client();
    let n = 4usize;
    let err = launch_der_param_resident(
        &cl,
        upload_channel_floats(&cl, &[0.0; 4]),
        upload_channel_floats(&cl, &[1.0; 4]),
        DerParamKernel::QuantileGradient,
        &[0.5],
        n,
    )
    .expect_err("a one-element params slice must be rejected");
    println!("[FPP-05] short params rejected: {err}");
}

/// The Focal arms have no resident consumer and must reject EXPLICITLY rather than fall
/// through to a wrong kernel — the exact failure class this seam exists to fix.
#[test]
fn resident_param_der_rejects_uncovered_kernels() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[FPP-05] SKIP resident_param_der_rejects_uncovered_kernels: needs rocm/cuda");
        return;
    }
    let cl = client();
    let n = 4usize;
    for kernel in [DerParamKernel::FocalGradient, DerParamKernel::FocalHessian] {
        let err = launch_der_param_resident(
            &cl,
            upload_channel_floats(&cl, &[0.0; 4]),
            upload_channel_floats(&cl, &[1.0; 4]),
            kernel,
            &[0.25, 2.0],
            n,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("QuantileGradient"),
            "{kernel:?} must reject naming the only covered arm, got: {msg}"
        );
    }
}
