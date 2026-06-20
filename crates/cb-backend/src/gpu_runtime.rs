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

use cubecl::features::{AtomicUsage, Plane};
use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::kernels::{
    block_reduce_atomic_kernel, block_reduce_kernel, block_scan_kernel, focal_gradient_kernel,
    focal_hessian_kernel, gradient_kernel, logloss_gradient_kernel, logloss_hessian_kernel,
    pointwise_hist2_binary_kernel, pointwise_hist2_half_byte_kernel,
    pointwise_hist2_nonbinary_kernel, quantile_gradient_kernel,
};
use crate::SelectedRuntime;

/// Which cross-cube finalize path the atomic-reduce helper actually ran. The f64
/// in-kernel atomic add is not guaranteed on every backend (Pitfall 4 — HIP
/// supports f32 natively, f64 is emulated/optional; wgpu needs an atomic-float
/// extension), so the helper queries the device capability and reports the path it
/// took rather than crashing or silently producing a wrong result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicFinalizePath {
    /// f64 in-kernel `Atomic::fetch_add` cross-cube finalize ran (the D-03 path).
    InKernelAtomicF64,
    /// The device lacks f64 atomic-add; the portable Plan-01 shared-mem-partial +
    /// host `cb-core::sum_f64` finalize ran instead (documented fallback, NOT a
    /// silent drop).
    HostSumFallback,
}

/// Launch geometry: threads per cube (the cube `x` dimension), shared with the
/// `cpu_runtime.rs` launch helpers (IN-02 — one place, not repeated per helper).
/// This is the launch-geometry const AND the `SharedMemory` size the kernel
/// allocates (a comptime-const size — Pitfall 3); it is NOT a wave/warp-size
/// literal in any reduction stride (D-09).
const CUBE_DIM: usize = 32;

/// CR-02 guard: the launch-geometry cube width MUST NOT exceed the kernels'
/// comptime `SharedMemory` allocation ([`crate::kernels::BLOCK_REDUCE_SHMEM`]).
/// The fallback tree-reduce writes `shared[tid]` for every unit `tid in
/// 0..CUBE_DIM_X`, so a launch wider than the allocation would write out of
/// bounds device-side (UB). Coupling the two constants here makes any future
/// drift a COMPILE error rather than a silent OOB write.
const _: () = assert!(
    CUBE_DIM <= crate::kernels::BLOCK_REDUCE_SHMEM,
    "CUBE_DIM (launch width) exceeds BLOCK_REDUCE_SHMEM (shared-mem allocation) — \
     a wider launch would write past the kernels' SharedMemory (device-side OOB)"
);

/// Pointwise-histogram geometry guard (Phase 7.3 / Pitfall 3): the 8-bit non-binary
/// fill's worst-case used prefix is `2 channels * (1 << 8) bins = 512`, which MUST
/// fit the kernels' [`crate::kernels::HIST_SHMEM`] worst-case allocation. Coupling the
/// two here makes any future drift (e.g. raising the max bit-width) a COMPILE error
/// rather than a silent device-side OOB on the shared-mem follow-up path.
const _: () = assert!(
    2 * (1usize << HIST_MAX_BITS) <= crate::kernels::HIST_SHMEM,
    "2 * (1 << HIST_MAX_BITS) exceeds HIST_SHMEM (the per-block histogram allocation) — \
     a wider bit-width would overflow the kernels' worst-case SharedMemory reservation"
);

/// The maximum one-byte non-binary bit-width this phase fills (8-bit; Plans B/C/D
/// extend the SAME kernel to 5/6/7-bit via the comptime `bits` arg, all <= this).
const HIST_MAX_BITS: u32 = 8;

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
    // CR-01: `block_scan_kernel` scans only WITHIN a single cube; the cross-cube
    // running carry is the documented 7.2/7.3 forward dependency (Open Q2). Launching
    // `ceil(n/CUBE_DIM)` cubes for `n > CUBE_DIM` would return WRONG prefix sums
    // (each cube restarts from 0, ignoring earlier cubes' running total) while
    // reporting `Ok` — a silent wrong result. Enforce the documented single-cube
    // precondition with a typed error until the cross-cube carry lands.
    if n > CUBE_DIM {
        return Err(CbError::Degenerate(format!(
            "launch_block_scan_f64 supports n <= {CUBE_DIM} until the cross-cube carry \
             lands (Open Q2, 7.2/7.3); got n = {n}"
        )));
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    // n <= CUBE_DIM (guarded above), so this is always a single cube.
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

/// Does the device support f64 in-kernel atomic-add? Drives the HOST choice between
/// the D-03 in-kernel-atomic finalize and the portable host-sum fallback (Pitfall
/// 4). Generic over the runtime (`ComputeClient<R>` is parameterized by the
/// `Runtime`, not its `Server`/`Channel`); mirrors the cubecl
/// `runtime_tests/atomic.rs::supports_feature` capability query.
fn device_supports_f64_atomic_add<R: cubecl::Runtime>(
    client: &cubecl::client::ComputeClient<R>,
) -> bool {
    let ty = <Atomic<f64> as CubePrimitive>::as_type_native_unchecked();
    client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::Add)
}

/// Reduce `input` to a SINGLE scalar sum on the compile-time [`SelectedRuntime`]
/// using the D-03 IN-KERNEL ATOMIC finalize (D-7.1-07) — the cross-cube sum is
/// performed on-device via `Atomic::fetch_add`, NOT by the host.
///
/// Returns `(sum, path)` where `path` records which finalize actually ran. When the
/// device advertises f64 atomic-add ([`AtomicFinalizePath::InKernelAtomicF64`]) the
/// atomic kernel runs and the cross-cube summation ORDER is non-deterministic (the
/// accepted D-03 source of run-to-run float-order variance — T-7.1-05). When the
/// device LACKS f64 atomic-add the helper falls back to the portable Plan-01
/// shared-mem-partial + host `cb-core::sum_f64` finalize
/// ([`AtomicFinalizePath::HostSumFallback`]) — a DOCUMENTED fallback, never a silent
/// drop of the atomic variant.
///
/// The empty input short-circuits to `(0.0, HostSumFallback)`. A device read-back
/// failure surfaces as [`CbError::Degenerate`] (WR-05). No
/// `unwrap`/`expect`/`panic`/indexing in this production helper (workspace lints +
/// D-13). The atomic path uses no wave/warp-size literal (the intra-cube fold reuses
/// the wave-agnostic plane / `CUBE_DIM_X`-strided shared-mem reduce, D-09).
///
/// # Caller contract (WR-02 — best-effort atomic, NOT guaranteed atomic)
///
/// This is a BEST-EFFORT atomic helper: on a device that does not advertise f64
/// atomic-add it returns a DETERMINISTIC host sum via the `HostSumFallback` branch.
/// Callers that require the in-kernel atomic finalize (e.g. to observe the D-03
/// cross-cube non-determinism) MUST inspect the returned [`AtomicFinalizePath`] and
/// must NOT discard it with `let (sum, _) = ...` — doing so silently accepts the
/// deterministic fallback. The `kernels::reduce` oracle asserts the returned path
/// matches the device's advertised capability so this substitution cannot pass
/// unnoticed in-env.
pub fn launch_block_reduce_atomic_f64(input: &[f64]) -> CbResult<(f64, AtomicFinalizePath)> {
    let n = input.len();
    if n == 0 {
        return Ok((0.0, AtomicFinalizePath::HostSumFallback));
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    // Pitfall 4: if the backend lacks f64 atomic-add, take the portable host-sum
    // fallback (the Plan-01 atomic-free path) and REPORT it — do not crash, do not
    // silently produce a wrong result.
    if !device_supports_f64_atomic_add(&client) {
        let partials = launch_block_reduce_f64(input)?;
        let sum = cb_core::sum_f64(&partials);
        return Ok((sum, AtomicFinalizePath::HostSumFallback));
    }

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let num_cubes = n.div_ceil(CUBE_DIM).max(1);

    // The accumulator is a single f64 slot, zero-initialized so the in-kernel
    // `fetch_add`s accumulate from 0.0 (the additive identity).
    let acc_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64]));

    let use_plane = client.features().plane.contains(Plane::Ops);

    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    block_reduce_atomic_kernel::launch::<f64, SelectedRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), 1) },
        use_plane,
    );

    let bytes = client
        .read_one(acc_handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    let scalars = bytemuck::cast_slice::<u8, f64>(&bytes);
    // The accumulator is length-1; surface a malformed read-back as a typed error
    // rather than indexing (D-13 — no production indexing).
    let sum = scalars
        .first()
        .copied()
        .ok_or_else(|| CbError::Degenerate("atomic accumulator read-back was empty".to_string()))?;
    Ok((sum, AtomicFinalizePath::InKernelAtomicF64))
}

/// Which elementwise binary `(approx, target) -> der1` kernel the device-resident
/// der seam launches (Phase 7.2, GPU-01 der). This is the GPU analog of the
/// `cpu_runtime::BinaryKernel` selector, but constrained to the elementwise der
/// family the on-device residency seam carries: it starts with the RMSE gradient
/// (`target - approx`); `LoglossGradient` and the parametric/unary arms are added
/// in Plans 02/03 on this SAME seam (no new launch geometry per arm).
///
/// All arms produce UNWEIGHTED der1 — byte-identical in structure to the
/// `cb-compute::loss` baseline the self-oracle compares against (D-7.2-01/02,
/// approved Task-1 contract). The per-object weight is folded DOWNSTREAM by the
/// 7.3 `histogram_scatter_kernel` (`contrib[i] = der1[i] * weight[i]`), NOT in
/// this kernel: the seam hands 7.3 the unweighted der1 handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerBinaryKernel {
    /// RMSE first derivative `der1[i] = target[i] - approx[i]` (the reused
    /// [`gradient_kernel`], D-7.2-03 — no new math). Its der2 is the constant
    /// `-1.0`, produced by [`const_der_handle`] (no kernel).
    RmseGradient,
    /// Logloss / CrossEntropy first derivative `der1[i] = target[i] -
    /// sigmoid(approx[i])` (the reused [`logloss_gradient_kernel`], D-7.2-03 — no
    /// new math). Logloss AND CrossEntropy route to THIS one arm (Pitfall 6 / D-09):
    /// the Rust seam collapses both to the same sigmoid-gradient kernel (there is no
    /// separate CrossEntropy kernel). Its der2 is the (single-input) hessian
    /// [`DerUnaryKernel::LoglossHessian`].
    LoglossGradient,
}

/// Launch an elementwise binary der1 kernel on the compile-time
/// [`SelectedRuntime`] and return the der1 as a DEVICE BUFFER HANDLE — WITHOUT
/// reading it back to the host (SC-3 / D-7.2-04 / Pitfall 2). This is the
/// device-residency hand-off seam the 7.3 histogram kernels plug into: the
/// returned der1 handle stays on-device and is multiplied by the weight handle
/// downstream by `histogram_scatter_kernel`, never folded here.
///
/// Mirrors [`launch_block_reduce_f64`]'s per-call client + `Bytes::from_elems`
/// host->device transfer + `ArrayArg::from_raw_parts` launch shape EXACTLY, but
/// the output is per-ELEMENT (length `n`, NOT one slot per cube — a der is not a
/// reduction). The empty input short-circuits to a zero-length device handle
/// (no launch). NO `read_one` on this path. No `unwrap`/`expect`/`panic`/indexing
/// (workspace lints + D-13).
pub fn launch_der_binary_handle(
    approx: &[f64],
    target: &[f64],
    kernel: DerBinaryKernel,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_der_binary_into(&client, approx, target, kernel)
}

/// The ONE der launch geometry (IN-02 — one place, not duplicated per public
/// entry point). Transfers `approx`/`target` onto `client`, launches the selected
/// der kernel, and returns the der1 output Handle WITHOUT reading it back. The
/// caller owns the `client` lifecycle so a read-back (the self-oracle wrapper)
/// uses the SAME client that allocated the handle — the canonical CubeCL idiom keeps
/// one client per op through read-back (basic-operations manual). This is the SAFE,
/// recommended pattern; it is what the production hand-off path does and never reads
/// the handle back at all. (Note on the test harnesses: re-resolving the client via
/// `Runtime::client(&device)` for the SAME runtime/device returns the cached global
/// client from CubeCL's per-device pool, NOT a foreign allocator — so the oracle
/// read-back wrappers that "construct a fresh client" actually share this client's
/// allocator/stream; see `kernels/pointwise_hist.rs::read_handle_f64`. The hazard to
/// avoid is reading a handle through a client of a DIFFERENT device/runtime, which
/// would violate a `slice::from_raw_parts` precondition in the HIP IO controller.)
/// Both the handle-returning public fn and the host-readback wrapper route through
/// here, so the launch geometry stays single.
fn launch_der_binary_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    approx: &[f64],
    target: &[f64],
    kernel: DerBinaryKernel,
) -> CbResult<Handle> {
    let n = approx.len();
    // Shape guard: the kernel reads `approx[i]` and `target[i]` for the same `i`,
    // so a mismatched length would read out of bounds on the device. Surface it as
    // a typed error (no panic) rather than launching a malformed kernel.
    if target.len() != n {
        return Err(CbError::LengthMismatch {
            column: "target".to_owned(),
            expected: n,
            actual: target.len(),
        });
    }

    // Empty input: hand back a zero-length device handle (no launch), mirroring the
    // reduce/scan empty short-circuit. 7.3 still receives a valid (empty) der handle.
    if n == 0 {
        return Ok(client.empty(0));
    }

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let target_handle = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
    // The der output is per-element (length `n`), NOT one slot per cube.
    let out_handle = client.empty(n * std::mem::size_of::<f64>());

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // `from_raw_parts` consumes the handle; clone the output so the original stays
    // returnable on-device (the 7.1 idiom). NO `read_one` here (SC-3). The
    // approx/target input handles are also consumed by `from_raw_parts`, so each
    // launch arm rebuilds them from the already-uploaded `*_handle.clone()`.
    match kernel {
        DerBinaryKernel::RmseGradient => gradient_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
            unsafe { ArrayArg::from_raw_parts(target_handle, n) },
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        ),
        DerBinaryKernel::LoglossGradient => logloss_gradient_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
            unsafe { ArrayArg::from_raw_parts(target_handle, n) },
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        ),
    }

    Ok(out_handle)
}

/// Host-readback wrapper over the der launch: launch the der1 kernel
/// device-resident, then read the handle back to a host `Vec<f64>`. This is the
/// seam the all-backend self-oracle exercises (it compares the device der1 to the
/// `cb-compute::loss` CPU baseline); it is NOT the histogram hand-off path (that is
/// [`launch_der_binary_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_der_binary_into`]); this wrapper
/// constructs the client ONCE and uses that SAME client for both the launch and the
/// read-back, so the handle is read by the client that allocated it (required — see
/// [`launch_der_binary_into`]). A device read-back failure surfaces as
/// [`CbError::Degenerate`] (WR-05), never a silent all-zero buffer masquerading as a
/// valid derivative.
pub fn launch_der_binary(
    approx: &[f64],
    target: &[f64],
    kernel: DerBinaryKernel,
) -> CbResult<Vec<f64>> {
    if approx.is_empty() {
        return Ok(Vec::new());
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    let handle = launch_der_binary_into(&client, approx, target, kernel)?;

    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Which elementwise UNARY (single-input) `approx -> der` kernel the device-resident
/// der seam launches (Phase 7.2, GPU-01 der). Unlike [`DerBinaryKernel`], the unary
/// family reads only `approx` (no `target`) — the Logloss/CrossEntropy hessian
/// `der2[i] = -p*(1-p)` with `p = sigmoid(approx[i])` is target-independent
/// (`logloss_hessian_kernel`). This is the seam shape for any single-input
/// derivative; Plans 03+ add arms on this SAME geometry.
///
/// All arms produce UNWEIGHTED der2 — byte-identical in structure to the
/// `cb-compute::loss` baseline the self-oracle compares against (D-7.2-01/02). The
/// per-object weight is folded DOWNSTREAM by the 7.3 histogram kernels, NOT here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerUnaryKernel {
    /// Logloss / CrossEntropy second derivative `der2[i] = -p*(1-p)`, `p =
    /// sigmoid(approx[i])` (the reused [`logloss_hessian_kernel`], D-7.2-03 — no new
    /// math). Logloss AND CrossEntropy share it (Pitfall 6 / D-09).
    LoglossHessian,
}

/// Launch an elementwise UNARY (single-input) der kernel on the compile-time
/// [`SelectedRuntime`] and return the der as a DEVICE BUFFER HANDLE — WITHOUT
/// reading it back (SC-3 / D-7.2-04 / Pitfall 2). This is the device-residency
/// hand-off seam for the single-input hessians (Logloss/CrossEntropy der2): the
/// returned handle stays on-device for the 7.3 histogram kernels.
///
/// Mirrors [`launch_der_binary_handle`] but with a SINGLE input array (`approx`);
/// the hessian takes no `target`. The empty input short-circuits to a zero-length
/// device handle (no launch). NO `read_one` on this path. No
/// `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
pub fn launch_der_unary_handle(approx: &[f64], kernel: DerUnaryKernel) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_der_unary_into(&client, approx, kernel)
}

/// The ONE unary-der launch geometry (IN-02 — one place). Transfers `approx` onto
/// `client`, launches the selected single-input der kernel, and returns the output
/// Handle WITHOUT reading it back. The caller owns the `client` lifecycle so a
/// read-back (the self-oracle wrapper) uses the SAME client that allocated the
/// handle — a CubeCL Handle is bound to its originating client (see
/// [`launch_der_binary_into`] for the full rationale).
fn launch_der_unary_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    approx: &[f64],
    kernel: DerUnaryKernel,
) -> CbResult<Handle> {
    let n = approx.len();

    // Empty input: hand back a zero-length device handle (no launch), mirroring the
    // binary short-circuit. 7.3 still receives a valid (empty) der handle.
    if n == 0 {
        return Ok(client.empty(0));
    }

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    // The der output is per-element (length `n`), NOT one slot per cube.
    let out_handle = client.empty(n * std::mem::size_of::<f64>());

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    match kernel {
        DerUnaryKernel::LoglossHessian => logloss_hessian_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
            // Clone so the original output handle stays returnable (the 7.1 idiom).
            // NO `read_one` here (SC-3).
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        ),
    }

    Ok(out_handle)
}

/// Host-readback wrapper over the unary der launch: launch the single-input der
/// kernel device-resident, then read the handle back to a host `Vec<f64>`. This is
/// the seam the all-backend self-oracle exercises (it compares the device der2 to
/// the `cb-compute::loss` CPU baseline); it is NOT the histogram hand-off path
/// (that is [`launch_der_unary_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_der_unary_into`]); this wrapper
/// constructs the client ONCE and uses that SAME client for both the launch and the
/// read-back. A device read-back failure surfaces as [`CbError::Degenerate`]
/// (WR-05), never a silent all-zero buffer masquerading as a valid derivative.
pub fn launch_der_unary(approx: &[f64], kernel: DerUnaryKernel) -> CbResult<Vec<f64>> {
    if approx.is_empty() {
        return Ok(Vec::new());
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    let handle = launch_der_unary_into(&client, approx, kernel)?;

    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Which PARAMETRIC elementwise der kernel the device-resident der seam launches
/// (Phase 7.2, GPU-01 der). The parametric family carries one or more scalar loss
/// parameters passed as length-1 `Array<F>` device buffers read at index 0 — NOT
/// scalar kernel args — to keep the kernels fully generic over `F: Float` (AGENTS.md
/// generics-float; the `launch_quantile_f64` / `launch_focal_f64` precedent: a
/// generic scalar arg would force the non-generic `F: ScalarArgType` bound).
///
/// All arms produce UNWEIGHTED der1 — byte-identical in structure to the
/// `cb-compute::loss` baseline (D-7.2-01/02). The per-object weight is folded
/// DOWNSTREAM by the 7.3 histogram kernels, NOT here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerParamKernel {
    /// Quantile{alpha, delta} first derivative (the reused
    /// [`quantile_gradient_kernel`], D-7.2-03 — no new math): with `val = target -
    /// approx`, `der1 = |val| < delta ? 0 : (val > 0 ? alpha : -(1-alpha))`. The
    /// params are `[alpha, delta]`. MAE routes through THIS arm at
    /// `(QUANTILE_ALPHA, QUANTILE_DELTA)` (WR-04 — no separate MAE kernel), so MAE
    /// and Quantile{0.5, 1e-6} are bit-identical. Its der2 is the constant `0.0`,
    /// produced by [`const_der_handle`] (Pitfall 5 — there is no quantile hessian
    /// kernel).
    QuantileGradient,
    /// Focal{alpha, gamma} FIRST derivative (the reused [`focal_gradient_kernel`],
    /// D-7.2-03 — no new math): `p = clamp(sigmoid(approx), 1e-13, 1-1e-13)`; with
    /// `at`/`pt` selected by the binary label and `y = 2*target - 1`, `der1 = -(at*y*
    /// pow(1-pt, gamma) * (gamma*pt*ln(pt) + pt - 1))`. The params are `[alpha,
    /// gamma]`. Unlike Quantile, Focal is a TWO-kernel family — its der2 is
    /// [`DerParamKernel::FocalHessian`] (a real hessian kernel, NOT a constant). The
    /// kernel clamps `p` so a saturated logit cannot produce `NaN` (T-04-02-02 /
    /// T-07.2-07).
    FocalGradient,
    /// Focal{alpha, gamma} SECOND derivative (the reused [`focal_hessian_kernel`],
    /// D-7.2-03 — no new math): the analytic hessian of [`Self::FocalGradient`]
    /// (`u*dv + du*v` chain, `error_functions.h:1684-1709`). Same `[alpha, gamma]`
    /// params, same length-1 `Array<F>` discipline, same `p` clamp (T-04-02-02). This
    /// is the SECOND kernel of the Focal two-kernel family; both run through this ONE
    /// parametric seam (no new launch geometry).
    FocalHessian,
}

/// Read the two-element `[alpha, delta]` (or `[param0, param1]`) param slice without
/// indexing (D-13 — no production indexing/panic). A malformed slice (fewer than the
/// kernel's required params) surfaces a typed [`CbError::Degenerate`], never a panic.
fn param_pair(params: &[f64], who: &str) -> CbResult<(f64, f64)> {
    let p0 = params.first().copied().ok_or_else(|| {
        CbError::Degenerate(format!("{who}: missing parameter 0 (param slice was empty)"))
    })?;
    let p1 = params.get(1).copied().ok_or_else(|| {
        CbError::Degenerate(format!(
            "{who}: missing parameter 1 (param slice had {} elements, need 2)",
            params.len()
        ))
    })?;
    Ok((p0, p1))
}

/// Launch a PARAMETRIC elementwise der1 kernel on the compile-time
/// [`SelectedRuntime`] and return the der1 as a DEVICE BUFFER HANDLE — WITHOUT
/// reading it back (SC-3 / D-7.2-04 / Pitfall 2). This is the device-residency
/// hand-off seam for the parametric losses (Quantile/MAE; Focal in Plan 03): the
/// returned handle stays on-device for the 7.3 histogram kernels.
///
/// The loss params pass as length-1 `Array<F>` device buffers (read at index 0) —
/// the `launch_quantile_f64` precedent — keeping the kernel generic over `F: Float`.
/// A malformed `params` slice surfaces [`CbError::Degenerate`] (no panic/indexing,
/// D-13). The empty input short-circuits to a zero-length device handle (no launch).
/// NO `read_one` on this path.
pub fn launch_der_param_handle(
    approx: &[f64],
    target: &[f64],
    kernel: DerParamKernel,
    params: &[f64],
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_der_param_into(&client, approx, target, kernel, params)
}

/// The ONE parametric-der launch geometry (IN-02 — one place). Transfers
/// `approx`/`target` and the length-1 param buffers onto `client`, launches the
/// selected param der kernel, and returns the output Handle WITHOUT reading it back.
/// The caller owns the `client` lifecycle so a read-back (the self-oracle wrapper)
/// uses the SAME client that allocated the handle (see [`launch_der_binary_into`]).
fn launch_der_param_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    approx: &[f64],
    target: &[f64],
    kernel: DerParamKernel,
    params: &[f64],
) -> CbResult<Handle> {
    let n = approx.len();
    // Shape guard: the kernel reads `approx[i]` and `target[i]` for the same `i`.
    if target.len() != n {
        return Err(CbError::LengthMismatch {
            column: "target".to_owned(),
            expected: n,
            actual: target.len(),
        });
    }

    // Empty input: hand back a zero-length device handle (no launch).
    if n == 0 {
        return Ok(client.empty(0));
    }

    let approx_handle = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let target_handle = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
    let out_handle = client.empty(n * std::mem::size_of::<f64>());

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    match kernel {
        DerParamKernel::QuantileGradient => {
            // `[alpha, delta]` — read without indexing (D-13). The length-1 device
            // buffers mirror `launch_quantile_f64` exactly (SelectedRuntime swapped in).
            let (alpha, delta) = param_pair(params, "QuantileGradient")?;
            let alpha_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![alpha]));
            let delta_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![delta]));
            quantile_gradient_kernel::launch::<f64, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                // Clone so the original output handle stays returnable (SC-3, no read).
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
                unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
                unsafe { ArrayArg::from_raw_parts(delta_handle, 1) },
            );
        }
        DerParamKernel::FocalGradient => {
            // `[alpha, gamma]` — read without indexing (D-13). The length-1 device
            // buffers mirror the `launch_focal_f64` precedent (SelectedRuntime swapped
            // in). The kernel clamps `p` so a saturated logit cannot produce NaN.
            let (alpha, gamma) = param_pair(params, "FocalGradient")?;
            let alpha_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![alpha]));
            let gamma_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![gamma]));
            focal_gradient_kernel::launch::<f64, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                // Clone so the original output handle stays returnable (SC-3, no read).
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
                unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
                unsafe { ArrayArg::from_raw_parts(gamma_handle, 1) },
            );
        }
        DerParamKernel::FocalHessian => {
            // The SECOND Focal kernel (der2), identical arg shape to FocalGradient.
            let (alpha, gamma) = param_pair(params, "FocalHessian")?;
            let alpha_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![alpha]));
            let gamma_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![gamma]));
            focal_hessian_kernel::launch::<f64, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_handle, n) },
                unsafe { ArrayArg::from_raw_parts(target_handle, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
                unsafe { ArrayArg::from_raw_parts(alpha_handle, 1) },
                unsafe { ArrayArg::from_raw_parts(gamma_handle, 1) },
            );
        }
    }

    Ok(out_handle)
}

/// Host-readback wrapper over the parametric der launch: launch the param der1
/// kernel device-resident, then read the handle back to a host `Vec<f64>`. This is
/// the seam the all-backend self-oracle exercises; it is NOT the histogram hand-off
/// path (that is [`launch_der_param_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_der_param_into`]); this wrapper
/// constructs the client ONCE and uses that SAME client for both the launch and the
/// read-back. A device read-back failure surfaces as [`CbError::Degenerate`]
/// (WR-05). A malformed `params` slice surfaces [`CbError::Degenerate`] too (no
/// indexing/panic, D-13).
pub fn launch_der_param(
    approx: &[f64],
    target: &[f64],
    kernel: DerParamKernel,
    params: &[f64],
) -> CbResult<Vec<f64>> {
    if approx.is_empty() {
        return Ok(Vec::new());
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    let handle = launch_der_param_into(&client, approx, target, kernel, params)?;

    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Build a length-`n` device buffer HANDLE filled with the constant `value`, for
/// the CONSTANT-der2 losses that have no hessian kernel (RMSE der2 = `-1.0`;
/// Quantile/MAE der2 = `0.0` — the [`DerParamKernel::QuantileGradient`] der2,
/// Pitfall 5: there is no quantile hessian kernel). The 7.3 histogram seam still receives
/// a der2 HANDLE for these losses (Pitfall 5) — there is NO `rmse_hessian_kernel`
/// to launch; the constant is materialized host-side and uploaded once.
///
/// The empty case short-circuits to a zero-length handle (no allocation churn). No
/// `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
pub fn const_der_handle(value: f64, n: usize) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    if n == 0 {
        return Ok(client.empty(0));
    }
    Ok(client.create(cubecl::bytes::Bytes::from_elems(vec![value; n])))
}

// ===========================================================================
// Phase 7.3 — the device-resident 2-channel pointwise histogram FILL seam
// (GPU-01 histogram slice). The 8-bit non-binary `ComputeHist2NonBinary<8>` analog:
// der1(UNWEIGHTED)/weight + cindex/indices in -> `binSums` device handle out, NO host
// round-trip (D-7.3-05). The FROZEN `binSums` layout + the der-handle-in ->
// binSums-handle-out seam this defines are reused UNCHANGED by Plans B/C/D and the
// 7.5 score/split consumer.
// ===========================================================================

/// The number of channels in a `hist2` cell (Σ der1, Σ weight) — the `* 2` in the
/// FROZEN interleaved `binSums` index. Naming the `2` removes the magic literal from
/// the layout arithmetic (it is the channel count, NOT a stride/warp literal — D-09).
const HIST_CHANNELS: usize = 2;

/// Compute the FROZEN `binSums` buffer length for the single-tree fill:
/// `histLineSize * 2 = (2 * totalBinFeatures) ... ` collapses, for `partCount =
/// foldCount = 1` and a single feature group with `FirstFoldIndex = 0`, to
/// `n_features * n_bins * HIST_CHANNELS` floats (the host-reference layout the
/// `kernels::pointwise_hist` oracle indexes cell-for-cell). See
/// [`launch_pointwise_hist2_handle`] for the full index formula.
#[inline]
fn hist2_binsums_len(n_bins: usize, n_features: usize) -> usize {
    n_features * n_bins * HIST_CHANNELS
}

/// Overflow-checked companion to [`hist2_binsums_len`] (WR-04). Returns `None` if
/// `n_features * n_bins * HIST_CHANNELS` overflows `usize`, so the host seam can reject a
/// degenerate dimension with a typed range error instead of wrapping silently. Kept next
/// to [`hist2_binsums_len`] so the two stay in lockstep on the FROZEN layout arithmetic.
#[inline]
fn hist2_binsums_len_checked(n_bins: usize, n_features: usize) -> Option<usize> {
    n_features
        .checked_mul(n_bins)
        .and_then(|v| v.checked_mul(HIST_CHANNELS))
}

/// Fill the device-resident 2-channel pointwise histogram (8-bit non-binary) on the
/// compile-time [`SelectedRuntime`] and return `binSums` as a DEVICE BUFFER HANDLE —
/// WITHOUT reading it back (SC-3 / D-7.3-05 / Pitfall 2/5). This is the load-bearing
/// hand-off seam the 7.5 score/split path plugs into: the returned histogram handle
/// stays on-device, consumed with no host round-trip.
///
/// # FROZEN `binSums` device-handle layout (D-7.3-01 / Pitfall 2)
///
/// The histogram is a flat `[partCount * foldCount * histLineSize]`-floats buffer with
/// the 2 channels (target, weight) interleaved per (feature, bin):
///
/// ```text
/// histLineSize = HIST_CHANNELS * totalBinFeatures             (totalBinFeatures = n_features * n_bins)
/// index(part, fold, feature, bin, channel) =
///     (GetHistogramOffset(part, fold) * histLineSize
///      + (FirstFoldIndex(feature) + bin)) * HIST_CHANNELS + channel
/// ```
///
/// mirroring upstream `split_properties_helpers.cuh::ShiftPartAndBinSumsPtr` +
/// `pointwise_hist2_one_byte_templ.cuh:132-145` (`... * 2 + w`). This phase delivers
/// the SINGLE-TREE fill: `partCount = foldCount = 1`, `GetHistogramOffset(0, 0) = 0`,
/// one feature group with `FirstFoldIndex = 0`, so the index collapses to the
/// kernel's write index `(feature * n_bins + bin) * HIST_CHANNELS + channel` and the
/// buffer length is [`hist2_binsums_len`]. The `fullPass = false` multi-part offset
/// (`ShiftPartAndBinSumsPtr`'s else-branch) is a 7.5 FORWARD DEPENDENCY (RESEARCH A2),
/// NOT filled here — documented, not silently cut. This layout is FROZEN across Plans
/// B/C/D and the 7.5 seam.
///
/// # Inputs (D-7.3-05)
///
/// `der1` (UNWEIGHTED, the 7.2 seam contract), `weight` (folded HERE as channel 1),
/// both length `n` in object order; `cindex` (length `n_features * n`, feature-major:
/// `cindex[feature * n + obj]` is object `obj`'s quantized bin for `feature`);
/// `indices` (length `n`, the object visiting order). `n_bins` is `1 << bits` (8-bit
/// -> 256 here); `n_features` is the feature-group width.
///
/// # Atomic merge (D-03 / Pitfall 1)
///
/// The cross-thread merge into `binSums` is ALWAYS the in-kernel `Atomic<F>::fetch_add`
/// (a handle cannot host-sum without a read-back). The only capability adaptation is the
/// channel float type: f64 on rocm/cuda/cpu, f32 on wgpu (WGSL has no f64 atomics —
/// RESEARCH A1). The [`AtomicFinalizePath`] reported by [`launch_pointwise_hist2`] is
/// INFORMATIONAL (the device's advertised f64-atomic capability, for the 7.6 epsilon
/// sign-off), NOT a selector — unlike the 7.1 reduce helper
/// ([`launch_block_reduce_atomic_f64`]) there is no host-sum fallback on this path; the
/// kernel always runs the in-kernel atomic regardless of advertised capability.
///
/// Empty input (`n == 0` or `n_features == 0` or `n_bins == 0`) short-circuits to a
/// zero-length handle with NO launch and NO read-back (Pitfall 5). Mismatched
/// der1/weight/cindex/indices lengths surface [`CbError::LengthMismatch`] BEFORE
/// launch (T-07.3-01). No `unwrap`/`expect`/`panic`/indexing in this production helper
/// (workspace lints + D-13).
pub fn launch_pointwise_hist2_handle(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pointwise_hist2_into(&client, der1, weight, cindex, indices, n_bins, n_features)
}

/// The ONE pointwise-histogram launch geometry (IN-02 — one place, not duplicated per
/// public entry point). Transfers `der1`/`weight`/`cindex`/`indices` onto `client`,
/// zero-initialises the `binSums` buffer, launches the 8-bit non-binary fill kernel,
/// and returns the `binSums` Handle WITHOUT reading it back. The caller owns the
/// `client` lifecycle so a read-back (the self-oracle wrapper) uses the SAME client
/// that allocated the handle — a CubeCL Handle is bound to its originating client (see
/// [`launch_der_binary_into`] for the full rationale).
fn launch_pointwise_hist2_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let n = der1.len();

    // Shape guards (T-07.3-01): the kernel reads der1[obj]/weight[obj] for the same
    // object, cindex[feature * n + obj] for each feature, and walks `indices` (length
    // n). A mismatch would read out of bounds on the device — surface a typed error
    // BEFORE launching a malformed kernel (no panic).
    if weight.len() != n {
        return Err(CbError::LengthMismatch {
            column: "weight".to_owned(),
            expected: n,
            actual: weight.len(),
        });
    }
    if indices.len() != n {
        return Err(CbError::LengthMismatch {
            column: "indices".to_owned(),
            expected: n,
            actual: indices.len(),
        });
    }

    // Overflow guards (WR-04) FIRST — before any unchecked product is formed (WR-01).
    // The cindex stride `n_features * n` and the binSums length
    // `n_features * n_bins * HIST_CHANNELS` are products of unbounded caller-supplied
    // dimensions. A wrapping `usize` multiply would silently address the wrong cell (no
    // fault, wrong histogram); a debug build would panic on multiply overflow, violating
    // the "no panic in this production helper" contract. Reject a degenerate dimension
    // with a typed range error BEFORE the product is ever computed unchecked, then REUSE
    // the checked product for the length guard below (never re-multiplying) — so the
    // "reject overflow with a typed error, never wrap" contract holds at the seam.
    let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n ({n}) overflows usize (cindex stride)"
        ))
    })?;
    if hist2_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    // Now the cindex length guard uses the already-checked product, never re-multiplying.
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (CR-01): the length guards above bound only the buffer
    // *positions*; the *values* inside `indices` and `cindex` drive unchecked device
    // array indices. Validate them HOST-SIDE so a malformed object id or bin surfaces a
    // typed `CbError::OutOfRange` rather than an out-of-bounds device read/store (UB).
    // This makes the "typed error, not UB" contract hold UNIFORMLY across the non-binary,
    // half-byte, and binary families — the non-binary kernel does NOT mask its bin, so it
    // relies entirely on this guard (WR-01).
    //
    // Object ids (`indices[i]`) index der1[obj]/weight[obj]/cindex[feature*n+obj]; a value
    // >= n would read those buffers out of bounds on the device.
    if let Some(&bad) = indices.iter().find(|&&ix| (ix as usize) >= n) {
        return Err(CbError::OutOfRange(format!(
            "indices value {bad} >= n ({n}); object id would read der1/cindex out of bounds"
        )));
    }
    // Every `cindex` bin must fit the dispatched line size (`n_bins`); a value >= n_bins
    // would write `bin_sums` out of bounds in the non-binary path (which does not mask).
    // This guard is applied UNIFORMLY across all families (IN-04): the binary (`& 1`) and
    // half-byte (`& 15`) kernels would tolerate larger raw values via masking, but we
    // intentionally hold them to the SAME `< n_bins` bound. That keeps the host reference
    // — which does NOT mask — bit-exact with the device for every family (stricter than
    // strictly necessary for the masked paths, never UB). Relax to mask-aware per-family
    // bounds only if a caller ever needs raw multi-bit cindex on a masked feature.
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    // 7.5 still receives a valid (empty) histogram handle.
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // Launch geometry: enough cubes to cover `n` objects (the grid-stride loop in
    // every fill kernel handles any surplus via the total-thread-count stride). Shared
    // by the half-byte branch below and the non-binary branch (IN-02 — one geometry).
    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // Binary (1-bit) family branch (Plan D — D-7.3-02). A border count of
    // `BINARY_BINS == 2` selects the SEPARATE `pointwise_hist2_binary_kernel` (NOT a
    // comptime case of the non-binary kernel, NOR the half-byte kernel): upstream's binary
    // path (`pointwise_hist2_binary.cu`'s `ComputeHist2Binary`) is a structurally distinct
    // kernel family (2-bucket split-bit decomposition), so we dispatch to its own kernel
    // here, mirroring `pointwise_kernels.cpp`'s `BinaryFeatures -> ComputeHist2Binary`
    // dispatch. It writes the SAME FROZEN binSums layout `(feature * 2 + bin) * 2 +
    // channel` through this UNCHANGED seam (the seam stays byte-identical — D-7.3-01). The
    // same channel-float dispatch applies (f64 on rocm/cuda/cpu, f32 on wgpu — RESEARCH
    // A1). NO read-back here (SC-3 / Pitfall 5).
    if n_bins == crate::kernels::BINARY_BINS {
        let bin_sums_len = hist2_binsums_len(n_bins, n_features);
        let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
        let indices_handle = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

        #[cfg(feature = "wgpu")]
        {
            let der1_f32: Vec<f32> = der1.iter().map(|&v| v as f32).collect();
            let weight_f32: Vec<f32> = weight.iter().map(|&v| v as f32).collect();
            let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1_f32));
            let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
            let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
            pointwise_hist2_binary_kernel::launch::<f32, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
                unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
                unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
                unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
                unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                n_features as u32,
            );
            return Ok(h);
        }

        #[cfg(not(feature = "wgpu"))]
        {
            let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1.to_vec()));
            let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight.to_vec()));
            let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
            pointwise_hist2_binary_kernel::launch::<f64, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
                unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
                unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
                unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
                unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                n_features as u32,
            );
            return Ok(h);
        }
    }

    // Half-byte (4-bit) family branch (Plan C — D-7.3-02). A border count of
    // `HALF_BYTE_BINS == 16` selects the SEPARATE `pointwise_hist2_half_byte_kernel`
    // (NOT a comptime case of the non-binary kernel): upstream's half-byte path
    // (`pointwise_hist2_half_byte_template.cuh`) is a structurally distinct kernel
    // family (16-bin working histogram + nibble decomposition), so we dispatch to its
    // own kernel here, mirroring `pointwise_kernels.cpp`'s `HalfByteFeatures ->
    // ComputeHist2HalfByte` dispatch. It writes the SAME FROZEN binSums layout
    // `(feature * 16 + bin) * 2 + channel` through this UNCHANGED seam (the seam stays
    // byte-identical — D-7.3-01). The same channel-float dispatch applies (f64 on
    // rocm/cuda/cpu, f32 on wgpu — RESEARCH A1). NO read-back here (SC-3 / Pitfall 5).
    if n_bins == crate::kernels::HALF_BYTE_BINS {
        let bin_sums_len = hist2_binsums_len(n_bins, n_features);
        let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
        let indices_handle = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

        #[cfg(feature = "wgpu")]
        {
            let der1_f32: Vec<f32> = der1.iter().map(|&v| v as f32).collect();
            let weight_f32: Vec<f32> = weight.iter().map(|&v| v as f32).collect();
            let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1_f32));
            let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
            let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
            pointwise_hist2_half_byte_kernel::launch::<f32, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
                unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
                unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
                unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
                unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                n_features as u32,
            );
            return Ok(h);
        }

        #[cfg(not(feature = "wgpu"))]
        {
            let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1.to_vec()));
            let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight.to_vec()));
            let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
            pointwise_hist2_half_byte_kernel::launch::<f64, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
                unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
                unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
                unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
                unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                n_features as u32,
            );
            return Ok(h);
        }
    }

    // One-byte non-binary bit-width selection (Plan B — D-7.3-02). The bit-count is
    // chosen HOST-SIDE from the feature group's border count `n_bins`, mirroring
    // upstream `pointwise_kernels.cpp`'s `DISPATCH_ONE_BYTE(..., 5/6/7/8)` (a `b`-bit
    // group has exactly `1 << b` bins). The selected `bits` is passed as the SAME
    // `#[comptime]` arg of the SAME `pointwise_hist2_nonbinary_kernel` (the FROZEN Plan
    // A kernel + binSums seam, reused UNCHANGED) — the comptime value drives the
    // histogram line size / used-prefix at JIT time, so there is NO runtime bit-count
    // branch in the device hot loop (the 7.1 `use_plane`/`inclusive` comptime pattern).
    // `n_bins` MUST equal `1 << bits` for a one-byte non-binary group; anything else
    // (not a power of two in the 5..=8 range) is not this family — surface a typed error
    // rather than launch a malformed line size (half-byte/binary are separate kernels,
    // Plans C/D). Half-byte (<5-bit, e.g. n_bins <= 16) is explicitly NOT routed here.
    let bits: u32 = match n_bins {
        32 => 5,
        64 => 6,
        128 => 7,
        256 => 8,
        _ => {
            return Err(CbError::Degenerate(format!(
                "pointwise_hist2 one-byte non-binary fill expects n_bins in {{32,64,128,256}} \
                 (1 << bits for bits in 5..=8), got {n_bins}"
            )));
        }
    };
    // Invariant (IN-01): `bits` is one of {5,6,7,8} from the match arms above and
    // `HIST_MAX_BITS == 8`, so `bits <= HIST_MAX_BITS` is a compile-time fact already
    // enforced by the module-level `const _: () = assert!(2 * (1 << HIST_MAX_BITS) <=
    // HIST_SHMEM)`. The used prefix `2 * (1 << bits)` can never exceed the 8-bit
    // shared-histogram allocation here — no runtime assert is needed.

    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let indices_handle = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

    // The binSums accumulator is zero-initialized so the in-kernel `fetch_add`s
    // accumulate from 0.0 (the additive identity). Length = the FROZEN layout length.
    let bin_sums_len = hist2_binsums_len(n_bins, n_features);

    // Launch geometry (IN-02): reuse the single `count`/`dim` already computed above and
    // shared with the binary/half-byte branches — there is exactly ONE geometry, not a
    // per-branch recomputation.

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1). The in-kernel atomic
    // merge needs `Atomic<F>::fetch_add` device-side. HIP (rocm) and CUDA support /
    // emulate the f64 atomic add, so the histogram channel is f64 there (the project's
    // f64-finalization preference, D-03). wgpu's WGSL has NO f64 atomics at all (the
    // capability is genuinely absent, not just unadvertised), so the wgpu arm uses an
    // f32 channel — upstream-faithful `atomicAdd(float*)`, natively supported — read
    // back and UPCAST to f64. The buffer length (the FROZEN layout) is channel-type
    // independent; only the element width differs, transparently to the 7.5 seam which
    // reads the handle through this same crate. The der1/weight inputs are cast to the
    // channel type at upload. `bytes::from_elems` keeps the buffer byte-typed.
    #[cfg(feature = "wgpu")]
    let bin_sums_handle = {
        let der1_f32: Vec<f32> = der1.iter().map(|&v| v as f32).collect();
        let weight_f32: Vec<f32> = weight.iter().map(|&v| v as f32).collect();
        let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1_f32));
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pointwise_hist2_nonbinary_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            bits,
        );
        h
    };

    #[cfg(not(feature = "wgpu"))]
    let bin_sums_handle = {
        let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1.to_vec()));
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pointwise_hist2_nonbinary_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
            // The atomic accumulator binds the same f64 storage as a plain buffer (the
            // `block_reduce_atomic_kernel` precedent); `clone` so the original stays
            // returnable on-device. NO `read_one` here (SC-3).
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            bits,
        );
        h
    };

    Ok(bin_sums_handle)
}

/// Read a `binSums` device handle back to a host `Vec<f64>`, transparently UPCASTING
/// the f32 channel on the wgpu arm (RESEARCH A1) and reading the f64 channel directly
/// elsewhere. Centralizes the channel-type read so both the readback wrapper and the
/// `kernels::pointwise_hist` oracle observe the SAME f64 layout regardless of backend.
/// A read-back failure surfaces [`CbError::Degenerate`] (WR-05).
fn read_binsums_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    #[cfg(feature = "wgpu")]
    {
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .map(|&v| f64::from(v))
            .collect())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
    }
}

/// Host-readback wrapper over the pointwise-histogram fill: launch the 8-bit
/// non-binary fill device-resident, then read the `binSums` handle back to a host
/// `Vec<f64>` AND report which [`AtomicFinalizePath`] the merge took. This is the seam
/// the all-backend self-oracle exercises (it compares the device histogram to the
/// ordered host reference); it is NOT the histogram hand-off path (that is
/// [`launch_pointwise_hist2_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pointwise_hist2_into`]); this
/// wrapper constructs the client ONCE and uses that SAME client for both the launch
/// and the read-back, so the handle is read by the client that allocated it (required
/// — see [`launch_der_binary_into`]). A device read-back failure surfaces as
/// [`CbError::Degenerate`] (WR-05), never a silent all-zero buffer masquerading as a
/// valid histogram.
///
/// The reported [`AtomicFinalizePath`] records whether the device advertised the
/// f64-atomic-add support the in-kernel merge relies on
/// ([`AtomicFinalizePath::InKernelAtomicF64`]) or not
/// ([`AtomicFinalizePath::HostSumFallback`] — informational on this path, since the
/// kernel always ran the in-kernel atomic; the report surfaces the capability for the
/// 7.6 epsilon sign-off and the RESEARCH A1 f32/f64 decision). REPORT-not-sign-off.
pub fn launch_pointwise_hist2(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<(Vec<f64>, AtomicFinalizePath)> {
    if der1.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok((Vec::new(), AtomicFinalizePath::HostSumFallback));
    }

    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    // Report the atomic capability the in-kernel merge depends on (Pitfall 1 / D-03 /
    // RESEARCH A1). The dispatch in `launch_pointwise_hist2_into` uses an f64 channel
    // on rocm/cuda/cpu (HIP/CUDA support or emulate the f64 atomic add) and an f32
    // channel on wgpu (WGSL has no f64 atomics). `device_supports_f64_atomic_add` is
    // the cubecl capability query: on gfx1100 it returns false (HIP does not ADVERTISE
    // f64 atomic-add even though it runs it), so this report is informational, not a
    // hard gate — it surfaces the device-advertised capability for the 7.6 epsilon
    // sign-off. The actual channel width is the compile-time backend choice above.
    let path = if device_supports_f64_atomic_add(&client) {
        AtomicFinalizePath::InKernelAtomicF64
    } else {
        AtomicFinalizePath::HostSumFallback
    };

    let handle =
        launch_pointwise_hist2_into(&client, der1, weight, cindex, indices, n_bins, n_features)?;

    let device_hist = read_binsums_f64(&client, handle)?;
    Ok((device_hist, path))
}
