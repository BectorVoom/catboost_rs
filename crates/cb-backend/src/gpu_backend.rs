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

use std::cell::RefCell;

use cb_compute::{
    DeviceGrownTree, Derivatives, EScoreFunction, Loss, Runtime, QUANTILE_ALPHA, QUANTILE_DELTA,
};
use cb_core::{CbError, CbResult};

use crate::gpu_runtime::{
    launch_der_binary, launch_der_param, launch_der_unary, DerBinaryKernel, DerParamKernel,
    DerUnaryKernel, GpuTrainSession,
};

/// The CubeCL GPU runtime as `cb-compute`'s [`Runtime`], generic over
/// [`crate::SelectedRuntime`] (the der seam resolves the concrete runtime internally). The
/// der path ([`Runtime::compute_gradients`]) creates its CubeCL client per call inside the
/// seam (stateless, mirroring [`crate::cpu_runtime::CpuBackend`]); the GPUT-02/03 device
/// grow-tree path holds a per-fit [`GpuTrainSession`] behind a `RefCell` so the `&self`
/// [`Runtime`] seam signatures stay unchanged (Pitfall 6 — the interior mutability drops the
/// former `Copy`/zero-sized derive).
///
/// `RefCell` is `!Sync`; that is fine here — the `Model` Send+Sync contract is about the
/// TRAINED model, not this transient training-time backend (which is bound once and used by
/// `&reference` within a single training call). `Default` constructs it with NO open session.
#[derive(Default)]
pub struct GpuBackend {
    /// The per-fit device-resident training session (GPUT-02): `Some` between a covered
    /// [`Runtime::begin_device_training`] and [`Runtime::end_device_training`], `None`
    /// otherwise (the CPU-fallback state, D-04).
    session: RefCell<Option<GpuTrainSession>>,
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A GpuTrainSession holds device handles (not Debug); surface only whether a session
        // is currently open so `GpuBackend` stays `Debug` without requiring the session to be.
        let active = self.session.borrow().is_some();
        f.debug_struct("GpuBackend")
            .field("session_active", &active)
            .finish()
    }
}

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

    /// GPUT-02/04: open a device-resident training session for a covered fit. Constructs a
    /// [`GpuTrainSession`] via its coverage gate (D-10-02) and stores it behind the `RefCell`,
    /// returning `Ok(true)` when the device path is selected or `Ok(false)` (→ CPU fallback,
    /// D-04) when the config is not covered (depth>1 / non-RMSE-Logloss / non-Plain /
    /// fold_count>1 / unsupported score fn). Overrides the trait default (which always
    /// declines).
    #[allow(clippy::too_many_arguments)]
    fn begin_device_training(
        &self,
        loss: &Loss,
        depth: usize,
        boosting_type_is_plain: bool,
        fold_count: usize,
        score_function: EScoreFunction,
        bins_feature_major: &[u32],
        weight: &[f64],
        n: usize,
        n_features: usize,
        n_bins: usize,
        learning_rate: f64,
        scaled_l2: f64,
    ) -> CbResult<bool> {
        // Phase 12 Plan 01 (Open Q2): the config surface is a single plain host
        // `DeviceTrainConfig`. The `Runtime::begin_device_training` trait method keeps its
        // arg list (its sole caller lives in `cb_train::boosting`, owned by a later wave), so
        // the backend constructs the DEFAULT covered regime here — every non-default family
        // knob (grow policy / sampling / exact / CTR) is promoted to the trait method by the
        // wave that lands its device kernel (Plan 02+). `default()` == today's covered path,
        // so this is byte-unchanged (D-04).
        let config = cb_compute::DeviceTrainConfig::default();
        let session = GpuTrainSession::begin(
            loss,
            depth,
            boosting_type_is_plain,
            fold_count,
            score_function,
            bins_feature_major,
            weight,
            n,
            n_features,
            n_bins,
            learning_rate,
            scaled_l2,
            &config,
        )?;
        let covered = session.is_some();
        *self.session.borrow_mut() = session;
        Ok(covered)
    }

    /// GPUT-03/04: grow one oblivious tree on the device over the resident session, or signal
    /// the CPU fallback. When a session is open it advances the device-resident boosting
    /// (der1 chained on device, approx updated via `apply_leaf_delta` — no per-tree der1
    /// read-back) and returns the host-typed [`DeviceGrownTree`]; when no session is open
    /// (uncovered config) it returns `Ok(None)` so the caller uses the CPU grow loop (D-04).
    ///
    /// The passed `approx` is validated against the session's object count; the resident
    /// approx is authoritative for the device pass (in the covered Plain / fold=1 / from-zero
    /// regime it tracks the caller's `approx` exactly, since both start at zero and apply the
    /// SAME per-tree `lr * leaf_values`). `target` is uploaded once by the session and reused.
    fn grow_tree_on_device(
        &self,
        approx: &[f64],
        target: &[f64],
    ) -> CbResult<Option<DeviceGrownTree>> {
        let mut guard = self.session.borrow_mut();
        match guard.as_mut() {
            None => Ok(None),
            Some(session) => {
                // Defensive length agreement (the session holds the authoritative resident
                // approx; a mismatch signals a caller inconsistency, surfaced typed not UB).
                if approx.len() != session.n() {
                    return Err(CbError::LengthMismatch {
                        column: "approx".to_owned(),
                        expected: session.n(),
                        actual: approx.len(),
                    });
                }
                let tree = session.grow_one(target)?;
                Ok(Some(tree))
            }
        }
    }

    /// GPUT-02/04: end the device training session, dropping the resident client + handles
    /// deterministically (a no-op if no session was open — the CPU-fallback path).
    fn end_device_training(&self) -> CbResult<()> {
        // `take()` moves the session out and drops it here (frees the client + handles).
        let _ = self.session.borrow_mut().take();
        Ok(())
    }
}
