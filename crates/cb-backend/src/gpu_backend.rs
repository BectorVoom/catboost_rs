//! `impl cb_compute::Runtime` for the CubeCL GPU runtimes (wgpu / cuda / rocm),
//! generic over [`crate::SelectedRuntime`] (Phase 8 gap plan 08-08).
//!
//! This is the GPU sibling of [`crate::cpu_runtime::CpuBackend`]: it satisfies
//! `cb-compute`'s abstract [`Runtime`] trait so the facade train path
//! (`cb_train::train<R: Runtime>`) can run over a GPU backend WITHOUT touching the
//! already-generic boosting loop. A SINGLE zero-sized [`GpuBackend`] serves ALL the
//! GPU backends — it computes derivatives on-device through the EXISTING,
//! oracle-validated Phase-7.2 der seam ([`crate::gpu_runtime::launch_der_binary`] et
//! al.), routed over [`crate::SelectedRuntime`] (cpu->CpuRuntime arm excluded here;
//! wgpu->WgpuRuntime / cuda->CudaRuntime / rocm->HipRuntime). There is NO
//! per-backend duplication and NO concrete runtime named in this file (selection is
//! the compile-time `SelectedRuntime` alias).
//!
//! # MVP loss support (08-08)
//!
//! GPU derivatives are computed for exactly the losses with an existing Phase-7.2
//! der kernel: `Rmse`, `Logloss`/`CrossEntropy`, `Mae`, `Quantile`, and `Focal`.
//! Every other loss (multiclass, multilabel, ranking, the smooth/positive-domain
//! losses without a GPU der kernel, custom objectives, RMSEWithUncertainty) returns
//! a typed [`CbError`] naming the loss as not-yet-supported on the GPU backend
//! (a documented parity gap, NOT a bug) — never a silent fallback, never a panic.
//! No new `#[cube]` kernels are authored here: the seam is reused verbatim.
//!
//! # Layout parity with CpuBackend
//!
//! The shape validation (`approx_dimension == 0` reject, non-divisible length
//! reject, per-object `target.len() == n` check, empty short-circuit) and the
//! per-dimension outer launch loop mirror [`crate::cpu_runtime::CpuBackend`] EXACTLY
//! (RESEARCH Pitfall 1 — never fuse the per-dimension pass), so the UN-reduced
//! `Derivatives` the host loop folds via `cb_core::sum_f64` are bit-compatible with
//! the CPU path within the Phase-7 GPU tolerance (D-04, <=1e-4).

use cb_compute::{Derivatives, Loss, Runtime, QUANTILE_ALPHA, QUANTILE_DELTA};
use cb_core::{CbError, CbResult};

use crate::gpu_runtime::{
    launch_der_binary, launch_der_param, launch_der_unary, DerBinaryKernel, DerParamKernel,
    DerUnaryKernel,
};

/// The CubeCL GPU runtime as `cb-compute`'s [`Runtime`], generic over
/// [`crate::SelectedRuntime`] (the der seam resolves the concrete runtime
/// internally). A zero-sized handle — the CubeCL client is created per call inside
/// the seam, mirroring [`crate::cpu_runtime::CpuBackend`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBackend;

/// Host-materialize a length-`n` constant der2 buffer for the CONSTANT-der2 losses
/// (RMSE der2 = `-1.0`, Quantile/MAE der2 = `0.0`).
///
/// The constant der2 has NO GPU kernel and its value is identical on host and
/// device, so it is materialized directly as `vec![value; n]` — mirroring the
/// host-side `vec![value; n]` the CpuBackend uses for these losses. The 08-08 GPU
/// path folds der2 in the host loop (the on-device `const_der_handle` buffer is for
/// the 7.3 histogram hand-off, which 08-08 does not use), so no device round-trip
/// is performed here: the previous `const_der_handle` call (WR-07) allocated and
/// immediately discarded a device buffer without ever reading it back, doing no
/// validation and wasting a device allocation per call.
fn const_der_host(value: f64, n: usize) -> CbResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    Ok(vec![value; n])
}

/// Compute the per-object der1/der2 for `loss` over a SINGLE dimension's slices
/// (`approx_d` and `target_d` both length `n`) through the Phase-7.2 GPU der seam.
/// This is the GPU analog of [`crate::cpu_runtime`]'s `compute_gradients_one_dim`:
/// same per-loss dispatch shape, but routing the math to the device seam. Only the
/// losses with an existing GPU der kernel are supported; every other loss returns a
/// typed [`CbError`] (parity gap, not a bug) — no silent fallback, no panic.
fn compute_gradients_one_dim_gpu(
    loss: &Loss,
    approx_d: &[f64],
    target_d: &[f64],
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    let n = approx_d.len();
    match *loss {
        Loss::Rmse => {
            let der1 = launch_der_binary(approx_d, target_d, DerBinaryKernel::RmseGradient)?;
            // RMSE hessian is the constant -1.0 (no kernel — the const der seam).
            let der2 = const_der_host(-1.0_f64, n)?;
            Ok((der1, der2))
        }
        // CrossEntropy shares Logloss's der1/der2 EXACTLY (D-09 / Pitfall 6): the
        // der seam collapses both to the same sigmoid-gradient + hessian kernels.
        Loss::Logloss | Loss::CrossEntropy => {
            let der1 = launch_der_binary(approx_d, target_d, DerBinaryKernel::LoglossGradient)?;
            let der2 = launch_der_unary(approx_d, DerUnaryKernel::LoglossHessian)?;
            Ok((der1, der2))
        }
        // MAE == Quantile{alpha=0.5, delta=1e-6} (WR-04): route through the
        // parametric quantile kernel at the MAE constants, bit-identical to the
        // CpuBackend Mae arm. der2 = 0 (the constant der seam — no quantile hessian).
        Loss::Mae => {
            let der1 = launch_der_param(
                approx_d,
                target_d,
                DerParamKernel::QuantileGradient,
                &[QUANTILE_ALPHA, QUANTILE_DELTA],
            )?;
            let der2 = const_der_host(0.0_f64, n)?;
            Ok((der1, der2))
        }
        // Quantile{alpha, delta}: the parametric pinball gradient; der2 = 0.
        Loss::Quantile { alpha, delta } => {
            let der1 = launch_der_param(
                approx_d,
                target_d,
                DerParamKernel::QuantileGradient,
                &[alpha, delta],
            )?;
            let der2 = const_der_host(0.0_f64, n)?;
            Ok((der1, der2))
        }
        // Focal{alpha, gamma}: a TWO-kernel parametric family — both gradient and
        // hessian have real GPU kernels on the 7.2 seam.
        Loss::Focal { alpha, gamma } => {
            let der1 = launch_der_param(
                approx_d,
                target_d,
                DerParamKernel::FocalGradient,
                &[alpha, gamma],
            )?;
            let der2 = launch_der_param(
                approx_d,
                target_d,
                DerParamKernel::FocalHessian,
                &[alpha, gamma],
            )?;
            Ok((der1, der2))
        }
        // Every other loss lacks a Phase-7.2 GPU der kernel. Reject with a typed
        // CbError naming the loss as not-yet-supported on the GPU backend (a
        // documented parity gap per 08-08, NOT a bug). NO silent fallback to a wrong
        // gradient, NO panic. The CPU backend supports these; the GPU der seam will
        // grow arms in later GPU phases.
        ref other => Err(CbError::OutOfRange(format!(
            "loss {other:?} is not yet supported on the GPU backend (GpuBackend): no \
             Phase-7.2 GPU derivative kernel exists for it. Supported on GPU: Rmse, \
             Logloss, CrossEntropy, Mae, Quantile, Focal. Use the cpu backend for \
             other losses (parity gap, not a bug)."
        ))),
    }
}

impl Runtime for GpuBackend {
    fn compute_gradients(
        &self,
        loss: &Loss,
        approx: &[f64],
        target: &[f64],
        approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        // Shape validation — mirrors CpuBackend EXACTLY (T-6.2-01a): reject a zero
        // dimension or a non-divisible length up front with a typed CbError (no
        // panic / no `unwrap`).
        if approx_dimension == 0 {
            return Err(CbError::LengthMismatch {
                column: "approx_dimension".to_owned(),
                expected: 1,
                actual: 0,
            });
        }
        if approx.len() % approx_dimension != 0 {
            return Err(CbError::LengthMismatch {
                column: "approx".to_owned(),
                expected: approx.len() - (approx.len() % approx_dimension),
                actual: approx.len(),
            });
        }
        let n = approx.len() / approx_dimension;
        // `target` stays per-object length `n` for the scalar losses the GPU der
        // seam supports. The multilabel dim-major target + the multiclass/ranking/
        // custom dispatch are CPU-only in 08-08 — those losses are rejected by
        // `compute_gradients_one_dim_gpu` with a typed CbError, so they never reach
        // a malformed launch here.
        if target.len() != n {
            return Err(CbError::LengthMismatch {
                column: "target".to_owned(),
                expected: n,
                actual: target.len(),
            });
        }
        if approx.is_empty() {
            return Ok(Derivatives {
                der1: Vec::new(),
                der2: Vec::new(),
            });
        }

        // Per-dimension outer launch loop — mirrors CpuBackend (RESEARCH Pitfall 1):
        // at `approx_dimension == 1` it runs once over `approx[0..n]`, so the seam
        // inputs/outputs and the downstream `cb_core::sum_f64` order match the
        // scalar path. The supported GPU losses are all scalar (the multi-output
        // losses are rejected per dimension by the dispatch below), so this loop is
        // effectively single-iteration for the supported set.
        let mut der1 = Vec::with_capacity(approx.len());
        let mut der2 = Vec::with_capacity(approx.len());
        for d in 0..approx_dimension {
            let lo = d * n;
            let hi = lo + n;
            let approx_d = approx.get(lo..hi).ok_or_else(|| CbError::LengthMismatch {
                column: "approx (dim slice)".to_owned(),
                expected: hi,
                actual: approx.len(),
            })?;
            let (der1_d, der2_d) = compute_gradients_one_dim_gpu(loss, approx_d, target)?;
            der1.extend_from_slice(&der1_d);
            der2.extend_from_slice(&der2_d);
        }
        Ok(Derivatives { der1, der2 })
    }
}
