//! Generic device launch helpers over [`crate::SelectedRuntime`] (D-7.1-04).
//!
//! This is the Phase-7.1 GPU analog of [`crate::cpu_runtime`]: it mirrors that
//! file's per-call client construction, `bytemuck`/`Bytes` host<->device transfer,
//! and WR-05 typed-error read-back, but is parameterized over the compile-time
//! selected runtime so the SAME launch path serves `cpu`/`wgpu`/`cuda`/`rocm`
//! (D-7.1-01). It hosts the Phase-7.1 device primitives — a block sum reduction
//! ([`launch_block_reduce_f64`], used by the `kernels::reduce` rocm self-oracle)
//! and the block inclusive/exclusive prefix-scan ([`launch_block_scan_f64`], used
//! by the `kernels::scan` rocm self-oracle, GPU-01 scan / D-7.1-06).
//!
//! # Reduction contract (D-7.1-05 / Open Q1)
//!
//! [`block_reduce_kernel`] folds each cube's slice into ONE partial; this helper
//! returns the per-cube partials UN-summed. The parity-critical FINAL fold is the
//! host's job (`cb-core::sum_f64`, the frozen sequential order), exactly as the
//! elementwise gradient kernels leave the leaf reduction to the host (D-02). This
//! default finalize carries NO atomic-float dependency (the in-kernel
//! atomic-finalize variant is Plan 02), so it builds and runs regardless of
//! f64-atomic support on a given backend.
//!
//! # Wave-size policy (D-09)
//!
//! `use_plane` is queried ONCE host-side from
//! `client.features().plane.contains(Plane::Ops)` and passed as the kernel's
//! `#[comptime]` flag, selecting the plane fold or the shared-memory fallback at
//! JIT time. No warp/wave-size literal appears in the kernel reduction stride.
//!
//! # Error handling (T-7.1-02 / WR-05)
//!
//! A device read-back failure is mapped to a typed [`CbError::Degenerate`], never
//! a silent all-zero buffer that would masquerade as a valid reduction. No
//! `unwrap`/`expect`/`panic`/indexing in this production file (workspace lints + D-13).

use cubecl::features::Plane;
use cubecl::prelude::*;

use cb_core::{CbError, CbResult};

use crate::kernels::{block_reduce_kernel, block_scan_kernel};
use crate::SelectedRuntime;

/// Launch geometry: threads per cube (the cube `x` dimension), shared with the
/// `cpu_runtime.rs` launch helpers (IN-02 — one place, not repeated per helper).
/// This is the launch-geometry const AND the `SharedMemory` size the kernel
/// allocates (a comptime-const size — Pitfall 3); it is NOT a wave/warp-size
/// literal in any reduction stride (D-09).
const CUBE_DIM: usize = 32;

/// Reduce `input` to its sum on the compile-time [`SelectedRuntime`], returning
/// the per-cube PARTIAL sums (the host finalizes the across-cube fold via
/// `cb-core::sum_f64` — the default atomic-free finalize, Open Q1).
///
/// The empty input short-circuits to an empty vec (no launch). For `n > 0` the
/// helper transfers `input` to the device, launches [`block_reduce_kernel`] over
/// `ceil(n / CUBE_DIM)` cubes, and reads back one `f64` partial per cube. A
/// read-back failure surfaces as [`CbError::Degenerate`] (WR-05), never a zero
/// buffer.
///
/// `use_plane` is resolved from the device's `Plane::Ops` capability and passed as
/// the kernel's `#[comptime]` flag (D-7.1-08): the plane fold compiles where the
/// hardware supports subgroup ops, the shared-memory tree-reduce fallback
/// otherwise — both produce the same partials, exercised by the `kernels::reduce`
/// oracle.
pub fn launch_block_reduce_f64(input: &[f64]) -> CbResult<Vec<f64>> {
    let n = input.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let out_handle = client.empty(num_cubes * std::mem::size_of::<f64>());

    // Query the plane capability ONCE on the host and drive the comptime branch
    // (comptime specialization manual): zero device-side feature check.
    let use_plane = client.features().plane.contains(Plane::Ops);

    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    block_reduce_kernel::launch::<f64, SelectedRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), num_cubes) },
        use_plane,
    );

    // Propagate a device read-back failure as a typed CbError (WR-05): mapping it
    // to a zero buffer would masquerade as a valid all-zero reduction, silently
    // producing a degenerate sum instead of surfacing the backend failure.
    let bytes = client
        .read_one(out_handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Compute the block inclusive/exclusive prefix-scan of `input` on the
/// compile-time [`SelectedRuntime`] (GPU-01 scan, D-7.1-06). The returned vector is
/// the SAME length as the input (a scan is NOT a reduction): `output[i]` is the
/// running prefix-sum up to and including (inclusive) or strictly before
/// (exclusive) element `i`.
///
/// The empty input short-circuits to an empty vec (no launch). For `n > 0` the
/// helper transfers `input` to the device, launches [`block_scan_kernel`] over
/// `ceil(n / CUBE_DIM)` cubes, and reads back one `f64` per element. A read-back
/// failure surfaces as [`CbError::Degenerate`] (WR-05), never a silent zero buffer.
///
/// `inclusive` passes through as the kernel's `#[comptime]` flag (no runtime
/// branch in the kernel). SCOPE (RESEARCH Open Q2): the kernel performs the scan
/// WITHIN a single cube; the cross-cube running carry is the documented forward
/// dependency for 7.2/7.3, so this helper is correct for `n <= CUBE_DIM`. There is
/// no wave/warp-size literal: the plane prefix uses `PLANE_DIM` and the cross-plane
/// carry stride derives from `CUBE_DIM_X` / `PLANE_DIM` (D-09). No
/// `unwrap`/`expect`/`panic`/indexing in this production file (workspace lints + D-13).
pub fn launch_block_scan_f64(input: &[f64], inclusive: bool) -> CbResult<Vec<f64>> {
    let n = input.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    // The scan output is per-element (same length as the input), NOT one slot per
    // cube — a scan is not a reduction.
    let out_handle = client.empty(n * std::mem::size_of::<f64>());

    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    block_scan_kernel::launch::<f64, SelectedRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        inclusive,
    );

    // Typed-error read-back (WR-05): never a silent all-zero buffer masquerading as
    // a valid scan.
    let bytes = client
        .read_one(out_handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}
