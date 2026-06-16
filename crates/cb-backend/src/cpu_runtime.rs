//! `impl cb_compute::Runtime` for the CubeCL `CpuRuntime` (D-01/D-03).
//!
//! This is the concrete backend behind `cb-compute`'s abstract [`Runtime`] trait:
//! it launches the `#[cube]` elementwise kernels ([`crate::kernels`]) on the
//! CubeCL CPU runtime and returns the per-object derivative buffers UN-reduced
//! (D-02) — every parity-critical SUM is finalized host-side by `cb-compute` /
//! `cb-train` via `cb-core::sum_f64`. Phase 7 adds GPU runtimes implementing the
//! SAME trait additively, leaving `cb-compute`/`cb-train` untouched.
//!
//! # Data transfer
//!
//! Host buffers cross to/from the device as `f64` via `bytemuck::Pod`
//! (`Bytes::from_elems`), the transfer pattern the CubeCL manual prescribes.
//! Parity reductions are always finalized in `f64` regardless of device element
//! width, so this backend computes in `f64` end-to-end.

use cubecl::prelude::*;

use cb_compute::{Derivatives, Loss, Runtime, QUANTILE_ALPHA, QUANTILE_DELTA};
use cb_core::{CbError, CbResult};

use crate::kernels::{
    expectile_gradient_kernel, expectile_hessian_kernel, focal_gradient_kernel,
    focal_hessian_kernel, gradient_kernel, huber_gradient_kernel, huber_hessian_kernel,
    logcosh_gradient_kernel, logcosh_hessian_kernel, logloss_gradient_kernel, logloss_hessian_kernel,
    lq_gradient_kernel, lq_hessian_kernel, mape_gradient_kernel,
    poisson_gradient_kernel, poisson_hessian_kernel, quantile_gradient_kernel,
    tweedie_gradient_kernel, tweedie_hessian_kernel,
};

/// Launch geometry: threads per cube (the cube `x` dimension) shared by every
/// launch helper below. Extracted to a single constant (IN-02) so the launch
/// geometry lives in one place instead of being repeated verbatim per helper.
const CUBE_DIM: usize = 32;

/// The CubeCL CPU runtime as `cb-compute`'s [`Runtime`]. A zero-sized handle —
/// the actual CubeCL client is created per call from the default device (the
/// CPU runtime's client construction is cheap and stateless for this slice).
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

/// Launch a single elementwise `der1 = f(approx, target)` kernel on `CpuRuntime`
/// and read back the `f64` output, in object order.
fn launch_binary_f64(
    approx: &[f64],
    target: &[f64],
    kernel: BinaryKernel,
) -> CbResult<Vec<f64>> {
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice;
    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let target_handle = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
    let out_handle = client.empty(std::mem::size_of_val(approx));

    let cube_dim = CUBE_DIM;
    let num_cubes = n.div_ceil(cube_dim).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: cube_dim as u32,
        y: 1,
        z: 1,
    };

    match kernel {
        BinaryKernel::RmseGradient => gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
            unsafe { ArrayArg::from_raw_parts(target_handle, n) },
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        ),
        BinaryKernel::LoglossGradient => {
            logloss_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            )
        }
        BinaryKernel::LogCoshGradient => {
            logcosh_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            )
        }
        BinaryKernel::LogCoshHessian => {
            logcosh_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            )
        }
        BinaryKernel::PoissonGradient => {
            poisson_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            )
        }
        BinaryKernel::MapeGradient => {
            mape_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            )
        }
    }

    // Propagate a device read-back failure as a typed CbError (WR-05): mapping it
    // to a zero buffer would masquerade as a valid all-zero derivative vector,
    // silently producing a degenerate (no-gradient) tree instead of surfacing the
    // backend failure. `compute_gradients` returns CbResult, so the error channel
    // exists — use it.
    let bytes = client.read_one(out_handle).map_err(|e| {
        CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Which elementwise binary (approx, target) -> der kernel to launch. LogCosh is
/// non-parametric, so BOTH its gradient and hessian are binary kernels here (no
/// length-1 param array, unlike Lq/Huber/Expectile).
#[derive(Clone, Copy)]
enum BinaryKernel {
    RmseGradient,
    LoglossGradient,
    LogCoshGradient,
    LogCoshHessian,
    /// Poisson gradient `target - exp(approx)` (the hessian is the unary
    /// [`launch_poisson_hessian`] — no target input).
    PoissonGradient,
    /// MAPE gradient `sign(target-approx)/max(1,|target|)` (der2=0 — no hessian
    /// kernel; the dispatch fills a constant-0 vec, the Mae precedent).
    MapeGradient,
}

/// Launch the Logloss hessian kernel (`der2 = -p*(1-p)`) on `CpuRuntime`.
fn launch_logloss_hessian(approx: &[f64]) -> CbResult<Vec<f64>> {
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice;
    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let out_handle = client.empty(std::mem::size_of_val(approx));

    let cube_dim = CUBE_DIM;
    let num_cubes = n.div_ceil(cube_dim).max(1);

    logloss_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: cube_dim as u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
    );

    // Propagate a device read-back failure as a typed CbError (WR-05): mapping it
    // to a zero buffer would masquerade as a valid all-zero derivative vector,
    // silently producing a degenerate (no-gradient) tree instead of surfacing the
    // backend failure. `compute_gradients` returns CbResult, so the error channel
    // exists — use it.
    let bytes = client.read_one(out_handle).map_err(|e| {
        CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Launch the Poisson hessian kernel (`der2 = -exp(approx)`) on `CpuRuntime`. A
/// unary (approx-only) kernel like [`launch_logloss_hessian`] — the Poisson
/// hessian does not depend on the target.
fn launch_poisson_hessian(approx: &[f64]) -> CbResult<Vec<f64>> {
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice;
    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let out_handle = client.empty(std::mem::size_of_val(approx));

    let cube_dim = CUBE_DIM;
    let num_cubes = n.div_ceil(cube_dim).max(1);

    poisson_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: cube_dim as u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
    );

    // Propagate a device read-back failure as a typed CbError (WR-05): mapping it
    // to a zero buffer would masquerade as a valid all-zero derivative vector,
    // silently producing a degenerate (no-gradient) tree instead of surfacing the
    // backend failure. `compute_gradients` returns CbResult, so the error channel
    // exists — use it.
    let bytes = client.read_one(out_handle).map_err(|e| {
        CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Launch a Focal elementwise derivative kernel (`gradient` or `hessian`) on
/// `CpuRuntime`, passing the scalar `alpha`/`gamma` loss parameters, and read
/// back the `f64` output in object order.
fn launch_focal_f64(
    approx: &[f64],
    target: &[f64],
    alpha: f64,
    gamma: f64,
    hessian: bool,
) -> CbResult<Vec<f64>> {
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice;
    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let target_handle = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
    let out_handle = client.empty(std::mem::size_of_val(approx));
    // Loss params pass as length-1 device arrays (the kernel stays generic over
    // F — a generic scalar arg would need the non-generic ScalarArgType bound).
    let alpha_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![alpha]));
    let gamma_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![gamma]));

    let cube_dim = CUBE_DIM;
    let num_cubes = n.div_ceil(cube_dim).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: cube_dim as u32,
        y: 1,
        z: 1,
    };

    if hessian {
        focal_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
            unsafe { ArrayArg::from_raw_parts(target_handle, n) },
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
            unsafe { ArrayArg::from_raw_parts(gamma_handle, 1) },
        );
    } else {
        focal_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
            unsafe { ArrayArg::from_raw_parts(target_handle, n) },
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
            unsafe { ArrayArg::from_raw_parts(gamma_handle, 1) },
        );
    }

    // Propagate a device read-back failure as a typed CbError (WR-05): mapping it
    // to a zero buffer would masquerade as a valid all-zero derivative vector,
    // silently producing a degenerate (no-gradient) tree instead of surfacing the
    // backend failure. `compute_gradients` returns CbResult, so the error channel
    // exists — use it.
    let bytes = client.read_one(out_handle).map_err(|e| {
        CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Launch the Quantile gradient kernel (`val = target - approx; der1 = |val| <
/// delta ? 0 : (val > 0 ? alpha : -(1-alpha))`) on `CpuRuntime`, passing the
/// `alpha`/`delta` loss parameters as length-1 device arrays, and read back the
/// `f64` der1 in object order. Gradient-only — the Quantile der2 is the constant
/// `0` (the dispatch fills a zero vec, the Mae precedent). Mirrors
/// [`launch_focal_f64`] for the two-parameter Quantile gradient.
fn launch_quantile_f64(
    approx: &[f64],
    target: &[f64],
    alpha: f64,
    delta: f64,
) -> CbResult<Vec<f64>> {
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice;
    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let target_handle = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
    let out_handle = client.empty(std::mem::size_of_val(approx));
    // The loss params pass as length-1 device arrays (the kernel stays generic
    // over F — a generic scalar arg would need the non-generic ScalarArgType bound).
    let alpha_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![alpha]));
    let delta_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![delta]));

    let cube_dim = CUBE_DIM;
    let num_cubes = n.div_ceil(cube_dim).max(1);

    quantile_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        CubeDim {
            x: cube_dim as u32,
            y: 1,
            z: 1,
        },
        unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
        unsafe { ArrayArg::from_raw_parts(target_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
        unsafe { ArrayArg::from_raw_parts(delta_handle, 1) },
    );

    // Propagate a device read-back failure as a typed CbError (WR-05): mapping it
    // to a zero buffer would masquerade as a valid all-zero derivative vector,
    // silently producing a degenerate (no-gradient) tree instead of surfacing the
    // backend failure. `compute_gradients` returns CbResult, so the error channel
    // exists — use it.
    let bytes = client.read_one(out_handle).map_err(|e| {
        CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Which single-parameter smooth-loss derivative kernel to launch (Lq{q},
/// Huber{delta}, Expectile{alpha}). Each carries one scalar loss parameter passed
/// as a length-1 device `Array<F>` (the `focal` length-1-array precedent — keeps
/// the kernel generic over `F: Float`, AGENTS.md), and has a gradient and a
/// hessian form selected by the `hessian` flag.
#[derive(Clone, Copy)]
enum ParamKernel {
    Lq,
    Huber,
    Expectile,
    /// Tweedie{variance_power}: gradient + hessian both carry the `variance_power`
    /// scalar as a length-1 device array (the exp lives inside the der; raw approx).
    Tweedie,
}

/// Launch a single-parameter smooth-loss derivative kernel (`gradient` or
/// `hessian`) on `CpuRuntime`, passing the scalar loss `param` (q / delta /
/// alpha) as a length-1 device array, and read back the `f64` output in object
/// order. Mirrors [`launch_focal_f64`] for the one-parameter losses.
fn launch_param_f64(
    approx: &[f64],
    target: &[f64],
    param: f64,
    kind: ParamKernel,
    hessian: bool,
) -> CbResult<Vec<f64>> {
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice;
    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let target_handle = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
    let out_handle = client.empty(std::mem::size_of_val(approx));
    // The loss parameter passes as a length-1 device array (the kernel stays
    // generic over F — a generic scalar arg would need the non-generic
    // ScalarArgType bound).
    let param_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![param]));

    let cube_dim = CUBE_DIM;
    let num_cubes = n.div_ceil(cube_dim).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: cube_dim as u32,
        y: 1,
        z: 1,
    };

    let approx_arg = unsafe { ArrayArg::from_raw_parts(approx_handle, n) };
    let target_arg = unsafe { ArrayArg::from_raw_parts(target_handle, n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };
    let param_arg = unsafe { ArrayArg::from_raw_parts(param_handle, 1) };

    match (kind, hessian) {
        (ParamKernel::Lq, false) => lq_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
            &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
        ),
        (ParamKernel::Lq, true) => lq_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
            &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
        ),
        (ParamKernel::Huber, false) => {
            huber_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
            )
        }
        (ParamKernel::Huber, true) => {
            huber_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
            )
        }
        (ParamKernel::Expectile, false) => {
            expectile_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
            )
        }
        (ParamKernel::Expectile, true) => {
            expectile_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
            )
        }
        (ParamKernel::Tweedie, false) => {
            tweedie_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
            )
        }
        (ParamKernel::Tweedie, true) => {
            tweedie_hessian_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
                &client, count, dim, approx_arg, target_arg, out_arg, param_arg,
            )
        }
    }

    // Propagate a device read-back failure as a typed CbError (WR-05): mapping it
    // to a zero buffer would masquerade as a valid all-zero derivative vector,
    // silently producing a degenerate (no-gradient) tree instead of surfacing the
    // backend failure. `compute_gradients` returns CbResult, so the error channel
    // exists — use it.
    let bytes = client.read_one(out_handle).map_err(|e| {
        CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}"))
    })?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Compute the per-object der1/der2 for `loss` over a SINGLE dimension's slices
/// (`approx_d` and `target_d` both length `n`), reusing the existing per-loss
/// CubeCL kernel launchers. This is the scalar body of the historical
/// `compute_gradients` match, extracted verbatim so the outer per-dimension loop
/// in [`CpuBackend::compute_gradients`] can call it once per dimension over the
/// dim-major slices. At `approx_dimension == 1` the loop runs this exactly once
/// over `approx[0..n]`, so the kernel inputs, the `cb_core::sum_f64` order
/// downstream, and the output are byte-identical to the pre-6.2 scalar path
/// (RESEARCH Pitfall 1). No new loss arm is added here.
fn compute_gradients_one_dim(
    loss: &Loss,
    approx_d: &[f64],
    target_d: &[f64],
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    match *loss {
        Loss::Rmse => {
            let der1 = launch_binary_f64(approx_d, target_d, BinaryKernel::RmseGradient)?;
            // RMSE hessian is the constant -1.0 (no kernel needed).
            let der2 = vec![-1.0_f64; approx_d.len()];
            Ok((der1, der2))
        }
        // CrossEntropy shares Logloss's der1/der2 EXACTLY (D-09): reuse the
        // Logloss gradient + hessian kernels (no separate kernel needed).
        Loss::Logloss | Loss::CrossEntropy => {
            let der1 = launch_binary_f64(approx_d, target_d, BinaryKernel::LoglossGradient)?;
            let der2 = launch_logloss_hessian(approx_d)?;
            Ok((der1, der2))
        }
        Loss::Focal { alpha, gamma } => {
            let der1 = launch_focal_f64(approx_d, target_d, alpha, gamma, false)?;
            let der2 = launch_focal_f64(approx_d, target_d, alpha, gamma, true)?;
            Ok((der1, der2))
        }
        // MAE == Quantile{alpha=0.5, delta=1e-6} (WR-04): route through the
        // parametric quantile kernel (alpha/delta as length-1 device arrays)
        // rather than a duplicate `mae_gradient_kernel` that hardcodes
        // `F::new(1e-6)` / `F::new(0.5)` — the hardcoded path would drift from
        // the host scalar under a future f32 instantiation. This makes MAE and
        // Quantile{0.5} bit-identical by construction.
        Loss::Mae => {
            let der1 = launch_quantile_f64(approx_d, target_d, QUANTILE_ALPHA, QUANTILE_DELTA)?;
            // MAE / Quantile hessian is the constant 0.0 (no kernel needed).
            let der2 = vec![0.0_f64; approx_d.len()];
            Ok((der1, der2))
        }
        // Quantile{alpha, delta} (Wave 3): the alpha-general pinball gradient
        // (the alpha/delta passed as length-1 device arrays). der2 = 0 (the
        // constant-0 vec, the Mae precedent). At alpha=0.5,delta=1e-6 the
        // gradient equals the Mae arm above (MAE == Quantile{0.5}).
        Loss::Quantile { alpha, delta } => {
            let der1 = launch_quantile_f64(approx_d, target_d, alpha, delta)?;
            let der2 = vec![0.0_f64; approx_d.len()];
            Ok((der1, der2))
        }
        // Wave-1 smooth losses (D-6.1-02): all four have a real der2, so each
        // launches BOTH a gradient and a hessian kernel.
        Loss::LogCosh => {
            let der1 = launch_binary_f64(approx_d, target_d, BinaryKernel::LogCoshGradient)?;
            let der2 = launch_binary_f64(approx_d, target_d, BinaryKernel::LogCoshHessian)?;
            Ok((der1, der2))
        }
        Loss::Lq { q } => {
            let der1 = launch_param_f64(approx_d, target_d, q, ParamKernel::Lq, false)?;
            let der2 = launch_param_f64(approx_d, target_d, q, ParamKernel::Lq, true)?;
            Ok((der1, der2))
        }
        Loss::Huber { delta } => {
            let der1 = launch_param_f64(approx_d, target_d, delta, ParamKernel::Huber, false)?;
            let der2 = launch_param_f64(approx_d, target_d, delta, ParamKernel::Huber, true)?;
            Ok((der1, der2))
        }
        Loss::Expectile { alpha } => {
            let der1 = launch_param_f64(approx_d, target_d, alpha, ParamKernel::Expectile, false)?;
            let der2 = launch_param_f64(approx_d, target_d, alpha, ParamKernel::Expectile, true)?;
            Ok((der1, der2))
        }
        // Wave-2 positive-domain / link losses (D-6.1-02 / Plan 06.1-02).
        // Poisson: exp-link der (inline F::exp); gradient is a binary kernel,
        // the hessian is the unary -exp(approx) kernel (no target input).
        Loss::Poisson => {
            let der1 = launch_binary_f64(approx_d, target_d, BinaryKernel::PoissonGradient)?;
            let der2 = launch_poisson_hessian(approx_d)?;
            Ok((der1, der2))
        }
        // Tweedie: exp INSIDE the der (raw approx, NOT exp-approx); both
        // gradient and hessian carry the variance_power scalar param.
        Loss::Tweedie { variance_power } => {
            let der1 =
                launch_param_f64(approx_d, target_d, variance_power, ParamKernel::Tweedie, false)?;
            let der2 =
                launch_param_f64(approx_d, target_d, variance_power, ParamKernel::Tweedie, true)?;
            Ok((der1, der2))
        }
        // MAPE: der2 = 0 (Pitfall 5 — Newton undefined). Only a gradient
        // kernel; the hessian is the constant 0.0 vec (the Mae precedent).
        Loss::Mape => {
            let der1 = launch_binary_f64(approx_d, target_d, BinaryKernel::MapeGradient)?;
            let der2 = vec![0.0_f64; approx_d.len()];
            Ok((der1, der2))
        }
        // MultiClass / MultiClassOneVsAll are multi-output losses handled in
        // `compute_gradients` BEFORE the per-dimension scalar loop reaches here
        // (softmax is cross-dimension-coupled; OneVsAll needs the per-object class
        // index to test `d == target_class`). They never enter this single-dimension
        // scalar dispatch, so reject them defensively rather than silently producing
        // a wrong gradient (no `unwrap`/panic — a typed CbError).
        Loss::MultiClass | Loss::MultiClassOneVsAll => Err(CbError::Degenerate(
            "multiclass losses are dispatched in compute_gradients, not the \
             single-dimension scalar path"
                .to_owned(),
        )),
        // MultiLogloss / MultiCrossEntropy are multi-output (multilabel) losses
        // dispatched in `compute_gradients` BEFORE the per-dimension scalar loop
        // (they need the dim-major target to read `target[d*n+i]`). They never enter
        // this single-dimension scalar dispatch; reject defensively (typed CbError,
        // no `unwrap`/panic).
        Loss::MultiLogloss | Loss::MultiCrossEntropy => Err(CbError::Degenerate(
            "multilabel losses are dispatched in compute_gradients, not the \
             single-dimension scalar path"
                .to_owned(),
        )),
    }
}

/// Compute the MultiLogloss / MultiCrossEntropy SEPARABLE per-dimension diagonal
/// der1/der2 over the dimension-major `approx` buffer. Both losses are the SAME
/// upstream `TMultiCrossEntropyError` class (`error_functions.h:781-820`), so they
/// share THIS one der path — only the admissible target range differs (validated
/// upstream of training). Each label dimension `d` is an independent binary sigmoid
/// cross-entropy with `der1 = target[d*n+i] - sigmoid(approx[d*n+i])`,
/// `der2 = -p*(1-p)`.
///
/// Unlike the multiclass losses (per-OBJECT class index), the multilabel `target`
/// is DIM-MAJOR length `dim*n` — one `{0,1}`/`[0,1]` label per dimension per object.
/// Both outputs are dimension-major length `dim*n` (`buf[d*n+i]`), reusing the
/// diagonal-loss leaf path per dimension.
fn compute_multilabel_gradients(approx: &[f64], target: &[f64], dim: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut der1 = vec![0.0_f64; dim * n];
    let mut der2 = vec![0.0_f64; dim * n];
    for d in 0..dim {
        for i in 0..n {
            let idx = d * n + i;
            let a = approx.get(idx).copied().unwrap_or(0.0);
            let t = target.get(idx).copied().unwrap_or(0.0);
            let (od1, od2) = cb_compute::multi_crossentropy_ders(a, t);
            if let Some(slot) = der1.get_mut(idx) {
                *slot = od1;
            }
            if let Some(slot) = der2.get_mut(idx) {
                *slot = od2;
            }
        }
    }
    (der1, der2)
}

/// Compute the MultiClass softmax COUPLED der1 (dimension-major) + PACKED symmetric
/// Hessian (per-object) over the full dimension-major `approx` buffer.
///
/// `approx` is `approx[d*n + i]` (length `k*n`); `target[i]` is object `i`'s
/// REMAPPED contiguous class index `[0, k)` (cast from `f64`). Returns
/// `(der1, der2_packed)` where:
/// - `der1` is dimension-major length `k*n` (`der1[d*n + i] = softmax_ders(...).0[d]`),
///   matching the diagonal losses' layout, and
/// - `der2_packed` is PER-OBJECT length `n * k*(k+1)/2`, object `i`'s packed
///   lower-triangular Hessian at `der2_packed[i*pk .. i*pk + pk]` (`pk = k*(k+1)/2`).
///
/// The coupled Hessian cannot ride the dimension-major `der2[d*n+i]` layout (it is
/// not diagonal), so the boosting loop reads it per-object via the packed stride.
fn compute_softmax_gradients(
    approx: &[f64],
    target: &[f64],
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let pk = k * (k + 1) / 2;
    let mut der1 = vec![0.0_f64; k * n];
    let mut der2_packed = vec![0.0_f64; n * pk];
    // Per-object gather of the k-dimensional approx slice (dim-major → object view).
    let mut obj_approx = vec![0.0_f64; k];
    for i in 0..n {
        for d in 0..k {
            obj_approx[d] = approx.get(d * n + i).copied().unwrap_or(0.0);
        }
        let target_class = target.get(i).copied().unwrap_or(0.0) as usize;
        let (od1, od2) = cb_compute::softmax_ders(&obj_approx, target_class);
        for d in 0..k {
            if let Some(slot) = der1.get_mut(d * n + i) {
                *slot = od1.get(d).copied().unwrap_or(0.0);
            }
        }
        for (j, &v) in od2.iter().enumerate() {
            if let Some(slot) = der2_packed.get_mut(i * pk + j) {
                *slot = v;
            }
        }
    }
    (der1, der2_packed)
}

/// Compute the MultiClassOneVsAll SEPARABLE per-dimension diagonal der1/der2 over
/// the dimension-major `approx` buffer. Each dimension `d` is an independent binary
/// one-vs-rest sigmoid with `der1 = δ(d == class_i) - sigmoid(approx_d)`,
/// `der2 = -p*(1-p)` (`error_functions.h:746-779`). Both outputs are dimension-major
/// length `k*n` (`buf[d*n + i]`), reusing the diagonal-loss leaf path per dimension.
fn compute_onevsall_gradients(approx: &[f64], target: &[f64], k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut der1 = vec![0.0_f64; k * n];
    let mut der2 = vec![0.0_f64; k * n];
    for i in 0..n {
        let class_i = target.get(i).copied().unwrap_or(0.0) as usize;
        for d in 0..k {
            let a = approx.get(d * n + i).copied().unwrap_or(0.0);
            let (od1, od2) = cb_compute::multiclass_onevsall_ders(a, d == class_i);
            if let Some(slot) = der1.get_mut(d * n + i) {
                *slot = od1;
            }
            if let Some(slot) = der2.get_mut(d * n + i) {
                *slot = od2;
            }
        }
    }
    (der1, der2)
}

impl Runtime for CpuBackend {
    fn compute_gradients(
        &self,
        loss: &Loss,
        approx: &[f64],
        target: &[f64],
        approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        // Shape validation (T-6.2-01a): `approx_dimension` partitions the
        // dim-major `approx` into `approx_dimension` contiguous per-dimension
        // slices of length `n`. Reject a zero dimension or a non-divisible
        // length up front with a typed CbError (no panic / no `unwrap`).
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
        // The multilabel losses (MultiLogloss / MultiCrossEntropy) carry a DIM-MAJOR
        // target of length `dim*n` (one label per dimension per object), unlike the
        // scalar / multiclass losses whose target is per-object length `n`. Validate
        // and dispatch them here, BEFORE the per-object `target.len() == n` check.
        let is_multilabel = matches!(loss, Loss::MultiLogloss | Loss::MultiCrossEntropy);
        if is_multilabel {
            if target.len() != approx_dimension * n {
                return Err(CbError::LengthMismatch {
                    column: "target".to_owned(),
                    expected: approx_dimension * n,
                    actual: target.len(),
                });
            }
            if approx.is_empty() {
                return Ok(Derivatives {
                    der1: Vec::new(),
                    der2: Vec::new(),
                });
            }
            let (der1, der2) = compute_multilabel_gradients(approx, target, approx_dimension, n);
            return Ok(Derivatives { der1, der2 });
        }
        // `target` stays per-object length `n` for the scalar / class losses
        // dispatched in this wave (the dim-major target widening is a later plan).
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

        // MULTICLASS dispatch (LOSS-02): handle the two multi-output classification
        // losses BEFORE the per-dimension scalar loop. MultiClass softmax is
        // cross-dimension-COUPLED — its `der2` is a PER-OBJECT packed symmetric
        // Hessian (length `n * k*(k+1)/2`), NOT the diagonal `der2[d*n+i]` layout;
        // the boosting loop reads it via the packed stride and runs the symmetric
        // Newton solve. MultiClassOneVsAll is SEPARABLE (per-dim diagonal sigmoid),
        // so its outputs ride the standard dimension-major `buf[d*n+i]` layout and
        // reuse the scalar Newton leaf path per dimension.
        match loss {
            Loss::MultiClass => {
                let (der1, der2) = compute_softmax_gradients(approx, target, approx_dimension, n);
                return Ok(Derivatives { der1, der2 });
            }
            Loss::MultiClassOneVsAll => {
                let (der1, der2) = compute_onevsall_gradients(approx, target, approx_dimension, n);
                return Ok(Derivatives { der1, der2 });
            }
            _ => {}
        }

        // Per-dimension kernel-launch loop (D-6.2-01 / RESEARCH Pitfall 1). The
        // reduction is an OUTER loop over dimensions; each iteration launches the
        // existing scalar kernel over the dim-major slice `approx[d*n..d*n+n]` and
        // concatenates into the dim-major output (`der1[d*n+i]`). This is NOT fused
        // into a single `0..approx_dimension * n` launch: at `approx_dimension==1`
        // the loop runs once over `approx[0..n]`, so the kernel inputs and outputs
        // are byte-identical to the pre-6.2 scalar path.
        let mut der1 = Vec::with_capacity(approx.len());
        let mut der2 = Vec::with_capacity(approx.len());
        for d in 0..approx_dimension {
            let approx_d = &approx[d * n..d * n + n];
            let (der1_d, der2_d) = compute_gradients_one_dim(loss, approx_d, target)?;
            der1.extend_from_slice(&der1_d);
            der2.extend_from_slice(&der2_d);
        }
        Ok(Derivatives { der1, der2 })
    }
}
