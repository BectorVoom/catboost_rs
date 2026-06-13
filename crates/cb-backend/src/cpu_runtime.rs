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
    gradient_kernel, logloss_gradient_kernel, logloss_hessian_kernel,
};

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

    let cube_dim = 32usize;
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
    }

    let bytes = client
        .read_one(out_handle)
        .unwrap_or_else(|_| cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n]));
    bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
}

/// Which elementwise binary (approx, target) -> der1 kernel to launch.
#[derive(Clone, Copy)]
enum BinaryKernel {
    RmseGradient,
    LoglossGradient,
}

/// Launch the Logloss hessian kernel (`der2 = -p*(1-p)`) on `CpuRuntime`.
fn launch_logloss_hessian(approx: &[f64]) -> Vec<f64> {
    let n = approx.len();
    let device = cubecl::cpu::CpuDevice;
    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let out_handle = client.empty(std::mem::size_of_val(approx));

    let cube_dim = 32usize;
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
            Loss::Logloss => {
                let der1 = launch_binary_f64(approx, target, BinaryKernel::LoglossGradient);
                let der2 = launch_logloss_hessian(approx);
                Ok(Derivatives { der1, der2 })
            }
        }
    }
}
