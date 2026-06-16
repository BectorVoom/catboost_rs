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

use cb_compute::{Derivatives, Loss, Runtime};
use cb_core::{CbError, CbResult};

use crate::kernels::{
    expectile_gradient_kernel, expectile_hessian_kernel, focal_gradient_kernel,
    focal_hessian_kernel, gradient_kernel, huber_gradient_kernel, huber_hessian_kernel,
    logcosh_gradient_kernel, logcosh_hessian_kernel, logloss_gradient_kernel, logloss_hessian_kernel,
    lq_gradient_kernel, lq_hessian_kernel, mae_gradient_kernel, mape_gradient_kernel,
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
) -> Vec<f64> {
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
        BinaryKernel::MaeGradient => mae_gradient_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
            unsafe { ArrayArg::from_raw_parts(target_handle, n) },
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        ),
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

    let bytes = client
        .read_one(out_handle)
        .unwrap_or_else(|_| cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

/// Which elementwise binary (approx, target) -> der kernel to launch. LogCosh is
/// non-parametric, so BOTH its gradient and hessian are binary kernels here (no
/// length-1 param array, unlike Lq/Huber/Expectile).
#[derive(Clone, Copy)]
enum BinaryKernel {
    RmseGradient,
    LoglossGradient,
    MaeGradient,
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
fn launch_logloss_hessian(approx: &[f64]) -> Vec<f64> {
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

    let bytes = client
        .read_one(out_handle)
        .unwrap_or_else(|_| cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

/// Launch the Poisson hessian kernel (`der2 = -exp(approx)`) on `CpuRuntime`. A
/// unary (approx-only) kernel like [`launch_logloss_hessian`] — the Poisson
/// hessian does not depend on the target.
fn launch_poisson_hessian(approx: &[f64]) -> Vec<f64> {
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

    let bytes = client
        .read_one(out_handle)
        .unwrap_or_else(|_| cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
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
) -> Vec<f64> {
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

    let bytes = client
        .read_one(out_handle)
        .unwrap_or_else(|_| cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

/// Launch the Quantile gradient kernel (`val = target - approx; der1 = |val| <
/// delta ? 0 : (val > 0 ? alpha : -(1-alpha))`) on `CpuRuntime`, passing the
/// `alpha`/`delta` loss parameters as length-1 device arrays, and read back the
/// `f64` der1 in object order. Gradient-only — the Quantile der2 is the constant
/// `0` (the dispatch fills a zero vec, the Mae precedent). Mirrors
/// [`launch_focal_f64`] for the two-parameter Quantile gradient.
fn launch_quantile_f64(approx: &[f64], target: &[f64], alpha: f64, delta: f64) -> Vec<f64> {
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

    let bytes = client
        .read_one(out_handle)
        .unwrap_or_else(|_| cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
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
) -> Vec<f64> {
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

    let bytes = client
        .read_one(out_handle)
        .unwrap_or_else(|_| cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

impl Runtime for CpuBackend {
    fn compute_gradients(
        &self,
        loss: Loss,
        approx: &[f64],
        target: &[f64],
    ) -> CbResult<Derivatives> {
        if approx.len() != target.len() {
            return Err(CbError::LengthMismatch {
                column: "target".to_owned(),
                expected: approx.len(),
                actual: target.len(),
            });
        }
        if approx.is_empty() {
            return Ok(Derivatives {
                der1: Vec::new(),
                der2: Vec::new(),
            });
        }

        match loss {
            Loss::Rmse => {
                let der1 = launch_binary_f64(approx, target, BinaryKernel::RmseGradient);
                // RMSE hessian is the constant -1.0 (no kernel needed).
                let der2 = vec![-1.0_f64; approx.len()];
                Ok(Derivatives { der1, der2 })
            }
            // CrossEntropy shares Logloss's der1/der2 EXACTLY (D-09): reuse the
            // Logloss gradient + hessian kernels (no separate kernel needed).
            Loss::Logloss | Loss::CrossEntropy => {
                let der1 = launch_binary_f64(approx, target, BinaryKernel::LoglossGradient);
                let der2 = launch_logloss_hessian(approx);
                Ok(Derivatives { der1, der2 })
            }
            Loss::Focal { alpha, gamma } => {
                let der1 = launch_focal_f64(approx, target, alpha, gamma, false);
                let der2 = launch_focal_f64(approx, target, alpha, gamma, true);
                Ok(Derivatives { der1, der2 })
            }
            Loss::Mae => {
                let der1 = launch_binary_f64(approx, target, BinaryKernel::MaeGradient);
                // MAE / Quantile hessian is the constant 0.0 (no kernel needed).
                let der2 = vec![0.0_f64; approx.len()];
                Ok(Derivatives { der1, der2 })
            }
            // Quantile{alpha, delta} (Wave 3): the alpha-general pinball gradient
            // (the alpha/delta passed as length-1 device arrays). der2 = 0 (the
            // constant-0 vec, the Mae precedent). At alpha=0.5,delta=1e-6 the
            // gradient equals the Mae arm above (MAE == Quantile{0.5}).
            Loss::Quantile { alpha, delta } => {
                let der1 = launch_quantile_f64(approx, target, alpha, delta);
                let der2 = vec![0.0_f64; approx.len()];
                Ok(Derivatives { der1, der2 })
            }
            // Wave-1 smooth losses (D-6.1-02): all four have a real der2, so each
            // launches BOTH a gradient and a hessian kernel.
            Loss::LogCosh => {
                let der1 = launch_binary_f64(approx, target, BinaryKernel::LogCoshGradient);
                let der2 = launch_binary_f64(approx, target, BinaryKernel::LogCoshHessian);
                Ok(Derivatives { der1, der2 })
            }
            Loss::Lq { q } => {
                let der1 = launch_param_f64(approx, target, q, ParamKernel::Lq, false);
                let der2 = launch_param_f64(approx, target, q, ParamKernel::Lq, true);
                Ok(Derivatives { der1, der2 })
            }
            Loss::Huber { delta } => {
                let der1 = launch_param_f64(approx, target, delta, ParamKernel::Huber, false);
                let der2 = launch_param_f64(approx, target, delta, ParamKernel::Huber, true);
                Ok(Derivatives { der1, der2 })
            }
            Loss::Expectile { alpha } => {
                let der1 = launch_param_f64(approx, target, alpha, ParamKernel::Expectile, false);
                let der2 = launch_param_f64(approx, target, alpha, ParamKernel::Expectile, true);
                Ok(Derivatives { der1, der2 })
            }
            // Wave-2 positive-domain / link losses (D-6.1-02 / Plan 06.1-02).
            // Poisson: exp-link der (inline F::exp); gradient is a binary kernel,
            // the hessian is the unary -exp(approx) kernel (no target input).
            Loss::Poisson => {
                let der1 = launch_binary_f64(approx, target, BinaryKernel::PoissonGradient);
                let der2 = launch_poisson_hessian(approx);
                Ok(Derivatives { der1, der2 })
            }
            // Tweedie: exp INSIDE the der (raw approx, NOT exp-approx); both
            // gradient and hessian carry the variance_power scalar param.
            Loss::Tweedie { variance_power } => {
                let der1 =
                    launch_param_f64(approx, target, variance_power, ParamKernel::Tweedie, false);
                let der2 =
                    launch_param_f64(approx, target, variance_power, ParamKernel::Tweedie, true);
                Ok(Derivatives { der1, der2 })
            }
            // MAPE: der2 = 0 (Pitfall 5 — Newton undefined). Only a gradient
            // kernel; the hessian is the constant 0.0 vec (the Mae precedent).
            Loss::Mape => {
                let der1 = launch_binary_f64(approx, target, BinaryKernel::MapeGradient);
                let der2 = vec![0.0_f64; approx.len()];
                Ok(Derivatives { der1, der2 })
            }
        }
    }
}
