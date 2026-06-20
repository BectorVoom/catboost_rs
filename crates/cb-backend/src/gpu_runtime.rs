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
//! # Module contents (WR-03 — actual surface, broader than 7.1 reduce/scan)
//!
//! This module has grown to host THREE phases' device-launch seams. A single-
//! responsibility split into `gpu_runtime/{der,histogram}.rs` is an EXPLICIT tracked
//! follow-up (IN-03) — scheduled as dedicated refactor work, NOT an open-ended
//! "someday"; until that split lands this doc is the authoritative inventory:
//!
//! - **Phase 7.1 reduce/scan primitives** (this file's original scope):
//!   [`launch_block_reduce_f64`], [`launch_block_reduce_atomic_f64`],
//!   [`launch_block_scan_f64`], [`AtomicFinalizePath`].
//! - **Phase 7.2 der seam** (device-resident der1/der2): `DerBinaryKernel`,
//!   `DerUnaryKernel`, `DerParamKernel` and the six `launch_der_*` helpers.
//! - **Phase 7.3 pointwise-histogram seam**: `launch_pointwise_hist2*` and
//!   `read_binsums_f64`.
//!
//! The `kernels::{...}` import below therefore pulls in kernels for all three
//! seams, not only the 7.1 reduce/scan pair.
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
    block_reduce_atomic_kernel, block_reduce_kernel, block_scan_kernel, find_optimal_split_kernel,
    focal_gradient_kernel,
    focal_hessian_kernel, gradient_kernel, logloss_gradient_kernel, logloss_hessian_kernel,
    pairwise_hist_8bit_atomics_kernel, pairwise_hist_binary_kernel,
    pairwise_hist_half_byte_kernel, pairwise_hist_nonbinary_kernel,
    pointwise_hist2_binary_kernel,
    pointwise_hist2_half_byte_kernel, pointwise_hist2_nonbinary_kernel,
    quantile_gradient_kernel, SCORE_FN_L2,
};
use crate::SelectedRuntime;

/// Which cross-cube finalize path the atomic-reduce helper actually ran. The f64
/// in-kernel atomic add is not guaranteed on every backend (Pitfall 4 — HIP
/// supports f32 natively, f64 is emulated/optional; wgpu needs an atomic-float
/// extension), so the helper queries the device capability and reports the path it
/// took rather than crashing or silently producing a wrong result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicFinalizePath {
    /// f64 in-kernel `Atomic::fetch_add` cross-cube finalize ran AND the device
    /// advertised f64 atomic-add support (the D-03 path).
    InKernelAtomicF64,
    /// The device lacks f64 atomic-add; the portable Plan-01 shared-mem-partial +
    /// host `cb-core::sum_f64` finalize ran instead (documented fallback, NOT a
    /// silent drop). This is the GENUINE host-sum path — a consumer that inspects for
    /// it to confirm a deterministic host sum occurred (the contract
    /// [`launch_block_reduce_atomic_f64`] honors) can rely on this meaning.
    HostSumFallback,
    /// The in-kernel f64 atomic merge ran on-device even though the device did NOT
    /// ADVERTISE f64 atomic-add (WR-02). On HIP/gfx1100 the f64 atomic executes but
    /// `device_supports_f64_atomic_add` returns `false`; the histogram seam still
    /// finalizes entirely on-device via the kernel atomic. This is distinct from
    /// [`HostSumFallback`] (no host sum ran) and informational — it surfaces the
    /// device-advertised capability bit for the 7.6 epsilon sign-off WITHOUT claiming
    /// a host-sum fallback that never occurred.
    InKernelAtomicF64Unadvertised,
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

/// WR-04 guard: the shared-mem tree-reduce in [`crate::kernels::block_reduce_kernel`]
/// and [`crate::kernels::block_reduce_atomic_kernel`] halves the stride
/// (`s = CUBE_DIM_X / 2; ...; s /= 2`), which only covers EVERY element when the cube
/// width is a power of two. For a non-power-of-two `CUBE_DIM` the integer-halved
/// stride never reaches the top element(s), silently dropping them from the sum.
/// Coupling the precondition here makes any future launch-geometry change to a
/// non-power-of-two width a COMPILE error rather than a silent wrong reduction.
const _: () = assert!(
    CUBE_DIM.is_power_of_two(),
    "CUBE_DIM (launch width) must be a power of two — the shared-mem tree-reduce \
     halves its stride and would silently drop the top element(s) otherwise"
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
    // Validate length BEFORE the empty short-circuit so this readback wrapper and the
    // sibling `*_into`/`*_handle` entry points agree on the malformed-input contract
    // (WR-01): a (empty-approx, non-empty-target) input must reject, not silently
    // return `Ok([])`. Mirrors the shape guard in `launch_der_binary_into`.
    if target.len() != approx.len() {
        return Err(CbError::LengthMismatch {
            column: "target".to_owned(),
            expected: approx.len(),
            actual: target.len(),
        });
    }
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
    // Validate length BEFORE the empty short-circuit so this readback wrapper and the
    // sibling `*_into`/`*_handle` entry points agree on the malformed-input contract
    // (WR-01): a (empty-approx, non-empty-target) input must reject, not silently
    // return `Ok([])`. Mirrors the shape guard in `launch_der_param_into`.
    if target.len() != approx.len() {
        return Err(CbError::LengthMismatch {
            column: "target".to_owned(),
            expected: approx.len(),
            actual: target.len(),
        });
    }
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

    // IN-02: the per-family upload + zero-init + channel-float `#[cfg]` split + launch +
    // `return Ok(h)` boilerplate is IDENTICAL across the binary / half-byte / non-binary
    // arms (they differ only in the kernel launcher and, for non-binary, a trailing
    // `bits` comptime arg). Extract it into ONE macro so the channel-float split (f64 on
    // rocm/cuda/cpu, f32 on wgpu — RESEARCH A1) and the zero-init live in exactly one
    // place; a fix to the channel dispatch or the zero-init is then applied once, not
    // three times. The macro captures `client`/`count`/`dim`/`n`/`cindex`/`indices`/
    // `der1`/`weight`/`cindex_stride`/`n_features`/`n_bins` from the enclosing scope and
    // takes the kernel launcher path plus any trailing comptime launch args. It zero-
    // initialises `binSums` so the in-kernel `fetch_add`s accumulate from 0.0, casts the
    // der1/weight inputs to the channel type at upload, and returns the `binSums` Handle
    // WITHOUT a read-back (SC-3 / Pitfall 5). `from_raw_parts` consumes each input handle;
    // the output is cloned so the original stays returnable on-device.
    macro_rules! launch_hist2_family {
        ($kernel:ident $(, $extra:expr )* $(,)?) => {{
            let bin_sums_len = hist2_binsums_len(n_bins, n_features);
            let cindex_handle =
                client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
            let indices_handle =
                client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

            #[cfg(feature = "wgpu")]
            {
                let der1_f32: Vec<f32> = der1.iter().map(|&v| v as f32).collect();
                let weight_f32: Vec<f32> = weight.iter().map(|&v| v as f32).collect();
                let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1_f32));
                let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
                let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
                $kernel::launch::<f32, SelectedRuntime>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
                    unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
                    unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
                    unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
                    unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                    n_features as u32,
                    $( $extra, )*
                );
                return Ok(h);
            }

            #[cfg(not(feature = "wgpu"))]
            {
                let der1_handle = client.create(cubecl::bytes::Bytes::from_elems(der1.to_vec()));
                let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight.to_vec()));
                let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
                $kernel::launch::<f64, SelectedRuntime>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(der1_handle, n) },
                    unsafe { ArrayArg::from_raw_parts(weight_handle, n) },
                    unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
                    unsafe { ArrayArg::from_raw_parts(indices_handle, n) },
                    unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                    n_features as u32,
                    $( $extra, )*
                );
                return Ok(h);
            }
        }};
    }

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
        launch_hist2_family!(pointwise_hist2_binary_kernel);
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
        launch_hist2_family!(pointwise_hist2_half_byte_kernel);
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

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1) + upload + zero-init + launch
    // are handled by the shared `launch_hist2_family!` macro (IN-02 — one place): the
    // in-kernel atomic merge needs `Atomic<F>::fetch_add` device-side; HIP (rocm) and CUDA
    // support / emulate the f64 atomic add, so the channel is f64 there (D-03), while
    // wgpu's WGSL has NO f64 atomics, so the wgpu arm uses an f32 channel (read back and
    // UPCAST to f64). The buffer length (the FROZEN layout) is channel-type independent.
    // The non-binary arm passes the host-selected `bits` comptime arg (the only structural
    // difference from the binary/half-byte arms). NO read-back here (SC-3 / Pitfall 5).
    // The macro `return`s the `binSums` handle from whichever channel `#[cfg]` arm is
    // compiled (exactly one is always active), so control never reaches the trailing
    // `unreachable!()` below — it only satisfies the type checker (the macro expands to a
    // `()`-typed statement, not the function's `CbResult<Handle>` tail).
    launch_hist2_family!(pointwise_hist2_nonbinary_kernel, bits);
    #[allow(unreachable_code)]
    {
        unreachable!("launch_hist2_family! returns the binSums handle from a channel #[cfg] arm")
    }
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
/// The reported [`AtomicFinalizePath`] records whether the device ADVERTISED the
/// f64-atomic-add support the in-kernel merge relies on
/// ([`AtomicFinalizePath::InKernelAtomicF64`]) or ran the in-kernel atomic WITHOUT the
/// device advertising it ([`AtomicFinalizePath::InKernelAtomicF64Unadvertised`] — the
/// gfx1100 case). This path NEVER returns [`AtomicFinalizePath::HostSumFallback`]: the
/// in-kernel atomic merge always runs (there is no host-sum fallback here), so reporting
/// `HostSumFallback` would falsely claim a deterministic host sum occurred (WR-02). The
/// report surfaces the device-advertised capability for the 7.6 epsilon sign-off and the
/// RESEARCH A1 f32/f64 decision. REPORT-not-sign-off.
pub fn launch_pointwise_hist2(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<(Vec<f64>, AtomicFinalizePath)> {
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
    //
    // WR-02: NEVER report `HostSumFallback` here — the in-kernel atomic ALWAYS runs on
    // this path (no host sum), so the unadvertised case is `InKernelAtomicF64Unadvertised`,
    // NOT `HostSumFallback` (which would collide with the reduce helper's genuine
    // host-sum semantics).
    let path = if device_supports_f64_atomic_add(&client) {
        AtomicFinalizePath::InKernelAtomicF64
    } else {
        AtomicFinalizePath::InKernelAtomicF64Unadvertised
    };

    if der1.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok((Vec::new(), path));
    }

    let handle =
        launch_pointwise_hist2_into(&client, der1, weight, cindex, indices, n_bins, n_features)?;

    let device_hist = read_binsums_f64(&client, handle)?;
    Ok((device_hist, path))
}

// ===========================================================================
// Phase 7.5 Plan A — the device-resident pointwise L2 split-SCORE + split-ARGMIN seam
// (GPU-01 score/split slice). Consumes the FROZEN 7.3 2-channel histogram handle in
// place (NO host round-trip of the histogram, D-05), computes the per-candidate L2 split
// score device-resident, and finishes the deterministic argmin (lowest-(feature,bin)-
// index tie-break) so the chosen split matches the CPU `select_best_candidate` exactly.
// Only the small per-candidate score buffer (the self-oracle observation) and the
// O(blocks) per-block winner descriptor cross host<->device; the bulk histogram never
// leaves the device. Cross-oracled in `kernels/score_split.rs` against the FROZEN CPU
// references `cb-compute/src/score.rs::l2_split_score` + `cb-train/src/tree.rs::
// select_best_candidate`.
// ===========================================================================

/// The O(1) best-split descriptor read back per level (upstream `TBestSplitProperties`
/// analogue, `pointwise_scores.cu:303-309`): the chosen `(feature, bin)` split, its
/// L2 score, and its gain. `#[repr(C)]` 16-byte POD so the device-written / host-read
/// bytes reinterpret with `bytemuck::cast_slice` with no padding surprises (the
/// `gpu_runtime.rs` read-back idiom). For Plan A `score == gain` (the L2 split score IS
/// the gain to maximize); the two fields are kept distinct for the Cosine/variant arms
/// (Plan E) and the upstream descriptor shape.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BestSplit {
    /// The winning candidate's feature index.
    pub feature_id: u32,
    /// The winning candidate's split border (bin index).
    pub bin_id: u32,
    /// The winning candidate's L2 split score.
    pub score: f32,
    /// The winning candidate's gain (== `score` for the L2 arm).
    pub gain: f32,
}

/// Fill the FROZEN 7.3 device-resident 2-channel histogram, then compute the
/// pointwise **L2 split score per candidate** and the **best `(feature, bin)` split**
/// device-resident on the compile-time [`SelectedRuntime`], returning
/// `(best_split, per_candidate_scores)`.
///
/// The histogram is filled with [`launch_pointwise_hist2_into`] (the FROZEN 7.3 seam)
/// and consumed IN PLACE by [`find_optimal_split_kernel`] — the bulk histogram NEVER
/// crosses to the host (D-05 / D-7.5-05). The per-candidate score vector (length
/// `n_features * n_bins`, flat order `feature * n_bins + bin`) is read back as the
/// self-oracle observation (the analog of reading `binSums` back ONCE in the histogram
/// oracle); the across-block argmin is finished host-side over the small O(blocks)
/// per-block winner array with the SAME lowest-`(feature, bin)`-index tie-break the
/// kernel uses, so the chosen split equals `cb_train::select_best_candidate` over the
/// ascending `(feature, bin)` order (Pitfall 1 / RESEARCH Pattern 4).
///
/// `der1` (UNWEIGHTED, the 7.2 seam contract) / `weight` (channel 1), length `n`;
/// `cindex` (feature-major quantized bins), length `n_features * n`; `indices` (object
/// visiting order), length `n`; `n_bins` is `1 << bits`; `scaled_l2` is the per-tree
/// `scale_l2_reg` output.
///
/// Empty input (`n == 0` / `n_features == 0` / `n_bins == 0`) short-circuits to
/// `(None, Vec::new())` with NO launch and NO read-back of a 0-length handle (Pitfall
/// 3/5). Mismatched input lengths / out-of-range bin/object values surface a typed
/// [`CbError`] BEFORE launch (the FROZEN 7.3 guards in `launch_pointwise_hist2_into`).
/// No `unwrap`/`expect`/`panic`/indexing in this production helper (workspace lints +
/// D-13).
#[allow(clippy::too_many_arguments)]
pub fn launch_find_optimal_split_pointwise(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    scaled_l2: f64,
) -> CbResult<(Option<BestSplit>, Vec<f64>)> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_find_optimal_split_pointwise_into(
        &client, der1, weight, cindex, indices, n_bins, n_features, scaled_l2,
    )
}

/// The ONE score/split launch geometry (the histogram-seam IN-02 precedent — one place).
/// Fills the device histogram, launches the score/argmin kernel consuming that handle,
/// reads back the per-candidate scores + the per-block winners, and finishes the
/// host-side argmin. The caller owns the `client` lifecycle so every read-back uses the
/// SAME client that allocated the handles (a CubeCL Handle is bound to its originating
/// client — see [`launch_der_binary_into`]).
#[allow(clippy::too_many_arguments)]
fn launch_find_optimal_split_pointwise_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    scaled_l2: f64,
) -> CbResult<(Option<BestSplit>, Vec<f64>)> {
    let n = der1.len();

    // Empty short-circuit FIRST (Pitfall 3/5): no histogram, no launch, no 0-len read.
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok((None, Vec::new()));
    }

    // Candidate-count overflow guard (T-07.5-01-02): the per-candidate score buffer and
    // the kernel's candidate index math are products of caller-supplied dimensions. Reject
    // a degenerate dimension with a typed range error BEFORE forming the product unchecked
    // (a wrapping multiply would address the wrong cell; a debug build would panic).
    let n_candidates = n_features.checked_mul(n_bins).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) overflows usize (candidate count)"
        ))
    })?;

    // Only the L2 arm exists this plan (D-7.5-01); a non-32/64/128/256 n_bins would not be
    // a one-byte non-binary feature group either, but the histogram fill (below) already
    // rejects that. The score kernel needs n_bins as a comptime u32; bound it to u32 here
    // so the cast cannot silently truncate a degenerate dimension.
    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel comptime line size)"))
    })?;

    // Fill the FROZEN 7.3 device-resident 2-channel histogram (this also runs the FROZEN
    // length / value-range guards on der1/weight/cindex/indices BEFORE any launch, and
    // returns a device HANDLE with NO read-back). The bulk histogram stays device-resident.
    let bin_sums = launch_pointwise_hist2_into(client, der1, weight, cindex, indices, n_bins, n_features)?;

    // Launch geometry: a SINGLE cube of CUBE_DIM units strides over all candidates and
    // block-reduces to one winner. (A multi-cube grid is a perf follow-up; one cube keeps
    // the across-block argmin trivial while staying device-resident.) The shared-mem argmin
    // size is the comptime ARGMIN_SHMEM == CUBE_DIM (Pitfall 3).
    let num_cubes = 1usize;
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // The score / argmin output buffers. `scores` is the per-candidate L2 score (the
    // self-oracle observation); `best_gain`/`best_idx` carry one winner per cube. They are
    // zero-initialised (the kernel writes every candidate it strides; the block-reduce
    // writes the per-cube winner). The channel float type matches the histogram channel:
    // f64 on rocm/cuda/cpu, f32 on wgpu (RESEARCH A1) — read back and UPCAST to f64.
    let bin_sums_len = hist2_binsums_len(n_bins, n_features);

    #[cfg(feature = "wgpu")]
    let (scores_handle, best_gain_handle, best_idx_handle) = {
        let scores_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; n_candidates]));
        let best_gain_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; num_cubes]));
        let best_idx_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_cubes]));
        let lambda_h = client.create(cubecl::bytes::Bytes::from_elems(vec![scaled_l2 as f32]));
        find_optimal_split_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, bin_sums_len) },
            unsafe { ArrayArg::from_raw_parts(scores_h.clone(), n_candidates) },
            unsafe { ArrayArg::from_raw_parts(best_gain_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(best_idx_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(lambda_h, 1) },
            n_features as u32,
            n_bins_u32,
            SCORE_FN_L2,
        );
        (scores_h, best_gain_h, best_idx_h)
    };

    #[cfg(not(feature = "wgpu"))]
    let (scores_handle, best_gain_handle, best_idx_handle) = {
        let scores_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; n_candidates]));
        let best_gain_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; num_cubes]));
        let best_idx_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_cubes]));
        let lambda_h = client.create(cubecl::bytes::Bytes::from_elems(vec![scaled_l2]));
        find_optimal_split_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, bin_sums_len) },
            unsafe { ArrayArg::from_raw_parts(scores_h.clone(), n_candidates) },
            unsafe { ArrayArg::from_raw_parts(best_gain_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(best_idx_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(lambda_h, 1) },
            n_features as u32,
            n_bins_u32,
            SCORE_FN_L2,
        );
        (scores_h, best_gain_h, best_idx_h)
    };

    // Read back the per-candidate scores (UPCAST f32->f64 on wgpu). A read-back failure
    // surfaces as CbError::Degenerate, never a silent zero buffer (WR-05 / T-07.5-01-04).
    let scores = read_scores_f64(client, scores_handle)?;

    // Read back the O(blocks) per-block winner descriptors (gain + candidate index) and
    // finish the across-block argmin host-side with the SAME lowest-index tie-break the
    // kernel uses. This is the ONLY O(1)-class crossing for the split decision (D-05).
    let best_gains = read_scores_f64(client, best_gain_handle)?;
    let best_idx_bytes = client
        .read_one(best_idx_handle)
        .map_err(|e| CbError::Degenerate(format!("best-idx read-back failed: {e:?}")))?;
    let best_idxs: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&best_idx_bytes).to_vec();

    // Finish the across-block argmin: highest gain wins; on an EXACT tie the LOWER
    // candidate index wins (strict first-wins parity). For a single cube this is the one
    // winner, but the loop keeps the contract for a future multi-cube grid.
    let mut best: Option<BestSplit> = None;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_c = u32::MAX;
    for (block, &gain) in best_gains.iter().enumerate() {
        let cand = best_idxs.get(block).copied().unwrap_or(u32::MAX);
        // Skip a block that found no candidate (cand == n_candidates sentinel).
        if (cand as usize) >= n_candidates {
            continue;
        }
        let take = gain > best_gain || (gain == best_gain && cand < best_c);
        if take {
            best_gain = gain;
            best_c = cand;
        }
    }
    if (best_c as usize) < n_candidates {
        let feature = (best_c as usize) / n_bins;
        let bin = (best_c as usize) % n_bins;
        let score = scores.get(best_c as usize).copied().unwrap_or(best_gain);
        best = Some(BestSplit {
            feature_id: feature as u32,
            bin_id: bin as u32,
            score: score as f32,
            gain: best_gain as f32,
        });
    }

    Ok((best, scores))
}

/// Read a device score handle back to a host `Vec<f64>`, UPCASTING the f32 channel on
/// the wgpu arm (RESEARCH A1) and reading the f64 channel directly elsewhere — the score
/// sibling of [`read_binsums_f64`]. A read-back failure surfaces [`CbError::Degenerate`]
/// (WR-05), never a silent zero buffer.
fn read_scores_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL score read-back failed: {e:?}")))?;
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

// ===========================================================================
// Phase 7.4 — the device-resident 4-channel WEIGHT-ONLY pairwise histogram FILL seam
// (GPU-01 histogram slice). The general one-byte non-binary
// `ComputePairwiseHistogramOneByte{5,6,7}Bits` analog: pair_i/pair_j (= `uint2* pairs`)
// + per-pair weight + cindex in -> 4-channel `binSums` device handle out, NO host
// round-trip (D-7.4-03). DISTINCT from the 7.2/7.3 der1/der2/pointwise seam: NO der1
// input, and the histogram is 4-channel weight-only (histId in {0,1,2,3}), NEVER the
// 7.3 2-channel der1/weight layout. The FROZEN 4-channel layout + the
// pair-handles-in -> binSums-handle-out seam this defines are reused UNCHANGED by Plans
// B/C/D/E and the 7.5 pairwise score/split consumer.
// ===========================================================================

/// The number of channels in a pairwise-histogram cell — the `* 4` in the FROZEN
/// pairwise `binSums` index (`histId in {0,1,2,3}`). Naming the `4` removes the magic
/// literal from the layout arithmetic (it is the channel count, NOT a stride/warp
/// literal — D-09). **This is 4, NEVER 2** (the 7.3 pointwise channel count) — Pitfall 2.
pub(crate) const PAIR_HIST_CHANNELS: usize = 4;

/// Compute the FROZEN pairwise `binSums` buffer length for the single-tree fill:
/// `histLineSize = PAIR_HIST_CHANNELS * totalBinFeatures` collapses, for `part = fold =
/// 0` and a single feature group with `FirstFoldIndex = 0`, to
/// `n_features * n_bins * PAIR_HIST_CHANNELS` floats. See [`launch_pairwise_hist_handle`]
/// for the full index formula.
#[inline]
fn pair_hist_binsums_len(n_bins: usize, n_features: usize) -> usize {
    n_features * n_bins * PAIR_HIST_CHANNELS
}

/// Overflow-checked companion to [`pair_hist_binsums_len`]. Returns `None` if
/// `n_features * n_bins * PAIR_HIST_CHANNELS` overflows `usize`, so the host seam can
/// reject a degenerate dimension with a typed range error instead of wrapping silently.
#[inline]
fn pair_hist_binsums_len_checked(n_bins: usize, n_features: usize) -> Option<usize> {
    n_features
        .checked_mul(n_bins)
        .and_then(|v| v.checked_mul(PAIR_HIST_CHANNELS))
}

/// Fill the device-resident 4-channel WEIGHT-ONLY pairwise histogram (5/6/7-bit
/// non-binary) on the compile-time [`SelectedRuntime`] and return `binSums` as a DEVICE
/// BUFFER HANDLE — WITHOUT reading it back (SC-3 / D-7.4-03 / Pitfall 2/5). This is the
/// load-bearing hand-off seam the 7.5 pairwise score/split path plugs into: the returned
/// histogram handle stays on-device, consumed with no host round-trip.
///
/// # FROZEN pairwise `binSums` device-handle layout (D-7.4-03 / Pitfall 2)
///
/// The histogram is a flat `[partCount * histLineSize]`-floats buffer with the 4
/// weight-only channels (`histId in {0,1,2,3}`) interleaved per (feature, bin):
///
/// ```text
/// histLineSize = PAIR_HIST_CHANNELS * totalBinFeatures      (totalBinFeatures = n_features * n_bins)
/// index(feature, bin, histId) = (feature * n_bins + bin) * PAIR_HIST_CHANNELS + histId
/// ```
///
/// mirroring upstream `pairwise_hist_one_byte_5bit.cuh:255-256` (`4 * (maxFoldCount * f
/// + fold) + histId`) + `split_properties_helpers.cuh`'s `Compare` predicate. This
/// phase delivers the SINGLE-TREE fill: `partCount = foldCount = 1`, one feature group
/// with `FirstFoldIndex = 0`, so the index collapses to the kernel's write index and
/// the buffer length is [`pair_hist_binsums_len`]. The `fullPass = false` multi-part
/// offset (`ShiftPartAndBinSumsPtr`'s else-branch) and the `BuildBinaryFeatureHistograms`
/// transform are 7.5 FORWARD DEPENDENCIES (RESEARCH Open Q3), NOT filled here —
/// documented, not silently cut. This raw 4-channel fill is what 7.5 reduces. This
/// layout is FROZEN across Plans B/C/D/E and the 7.5 seam. **Anti-pattern: any `* 2`
/// here silently breaks the 7.5 seam (Pitfall 2) — the buffer is ALWAYS `* 4`.**
///
/// # Inputs (D-7.4-03)
///
/// `pair_i`/`pair_j` (two parallel `u32` arrays = upstream `uint2* pairs`, length
/// `n_pairs`, OBJECT ids), `pair_weight` (the ONLY per-pair value — NO der1, length
/// `n_pairs`); `cindex` (length `n_features * n_objects`, feature-major:
/// `cindex[feature * n_objects + obj]` is object `obj`'s quantized bin for `feature` —
/// stride over OBJECTS, NOT pairs, Pitfall 3); `n_objects` the object count; `n_bins`
/// is `1 << bits`; `n_features` the feature-group width; `bits` in {5,6,7}; `one_hot`
/// threaded for Plan E.
///
/// # Atomic merge (D-03 / Pitfall 1)
///
/// The cross-thread merge into `binSums` is ALWAYS the in-kernel `Atomic<F>::fetch_add`.
/// The only capability adaptation is the channel float type: f64 on rocm/cuda/cpu, f32
/// on wgpu (WGSL has no f64 atomics — RESEARCH A1).
///
/// Empty input (`n_pairs == 0` or `n_features == 0` or `n_bins == 0`) short-circuits to
/// a zero-length handle with NO launch and NO read-back (Pitfall 5). Mismatched
/// pair_i/pair_j/pair_weight/cindex lengths surface [`CbError::LengthMismatch`] BEFORE
/// launch (T-07.4-01); out-of-range pair ids / bins surface [`CbError::OutOfRange`]
/// (T-07.4-02). No `unwrap`/`expect`/`panic`/indexing in this production helper.
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )
}

/// The ONE pairwise-histogram launch geometry (IN-02 — one place, not duplicated per
/// public entry point). Transfers `pair_i`/`pair_j`/`pair_weight`/`cindex` onto
/// `client`, zero-initialises the 4-channel `binSums` buffer, launches the non-binary
/// pairwise fill kernel, and returns the `binSums` Handle WITHOUT reading it back. The
/// caller owns the `client` lifecycle so a read-back (the self-oracle wrapper) uses the
/// SAME client that allocated the handle — a CubeCL Handle is bound to its originating
/// client.
#[allow(clippy::too_many_arguments)]
fn launch_pairwise_hist_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-01): the kernel reads pair_i[p]/pair_j[p]/pair_weight[p] for
    // each pair, and cindex[feature * n_objects + obj] for each feature. A mismatch would
    // read out of bounds on the device — surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST — before any unchecked product is formed. The cindex stride
    // `n_features * n_objects` and the binSums length `n_features * n_bins * 4` are
    // products of unbounded caller-supplied dimensions. Reject a degenerate dimension
    // with a typed range error BEFORE the product is ever computed unchecked, then REUSE
    // the checked product for the length guard below (never re-multiplying).
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
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

    // Value-range guards (T-07.4-02): the length guards above bound only buffer
    // *positions*; the *values* inside pair_i/pair_j (object ids) and cindex (bins) drive
    // unchecked device array indices. Validate them HOST-SIDE so a malformed object id or
    // bin surfaces a typed `CbError::OutOfRange` rather than an out-of-bounds device
    // read/store (UB). BOTH pair endpoints index cindex (Pitfall 3).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    // Every cindex bin must fit the dispatched line size (`n_bins`); a value >= n_bins
    // would write `bin_sums` out of bounds (the non-binary kernel does not mask).
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    // 7.5 still receives a valid (empty) histogram handle.
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // Validate the bit-width belongs to the one-byte non-binary family (5/6/7-bit). The
    // border count `n_bins` MUST equal `1 << bits`; anything else is not this family.
    if bits < 5 || bits > 7 || n_bins != (1usize << bits) {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist non-binary fill expects bits in {{5,6,7}} with n_bins == 1 << bits, \
             got bits={bits} n_bins={n_bins}"
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): the in-kernel atomic merge
    // needs `Atomic<F>::fetch_add` device-side; HIP (rocm) and CUDA support / emulate the
    // f64 atomic add, so the channel is f64 there (D-03), while wgpu's WGSL has NO f64
    // atomics, so the wgpu arm uses an f32 channel (read back and UPCAST to f64). The
    // buffer length (the FROZEN 4-channel layout) is channel-type independent. NO
    // read-back here (SC-3 / Pitfall 5). `from_raw_parts` consumes each input handle; the
    // output is cloned so the original stays returnable on-device.
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_nonbinary_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            bits,
            one_hot,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_nonbinary_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            bits,
            one_hot,
        );
        Ok(h)
    }
}

/// Read a pairwise `binSums` device handle back to a host `Vec<f64>`, transparently
/// UPCASTING the f32 channel on the wgpu arm (RESEARCH A1) and reading the f64 channel
/// directly elsewhere. Centralizes the channel-type read so both the readback wrapper
/// and the `kernels::pairwise_hist` oracle observe the SAME f64 layout regardless of
/// backend. A read-back failure surfaces [`CbError::Degenerate`].
fn read_pair_binsums_f64(
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

/// Host-readback wrapper over the pairwise-histogram fill: launch the non-binary fill
/// device-resident, then read the 4-channel `binSums` handle back to a host `Vec<f64>`.
/// This is the seam the all-backend self-oracle exercises (it compares the device
/// histogram to the ordered host reference); it is NOT the histogram hand-off path (that
/// is [`launch_pairwise_hist_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_into`]); this wrapper
/// constructs the client ONCE and uses that SAME client for both the launch and the
/// read-back, so the handle is read by the client that allocated it. A device read-back
/// failure surfaces as [`CbError::Degenerate`], never a silent all-zero buffer
/// masquerading as a valid histogram. Empty input returns an empty `Vec` (no launch).
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    bits: u32,
    one_hot: bool,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )?;

    read_pair_binsums_f64(&client, handle)
}

/// Launch the 8-bit-atomics pairwise histogram fill device-resident and return the
/// 4-channel `binSums` Handle WITHOUT reading it back (the SC-3 / D-7.4-03 hand-off path).
/// This is the SEPARATE launch arm for the structurally DISTINCT 8-bit-atomics family
/// (D-7.4-02 — upstream `pairwise_hist_one_byte_8bit_atomics.cuh`): the 256-bin x
/// 4-channel line exceeds the per-block shared-memory budget, so the kernel accumulates
/// via TRUE GLOBAL atomics. It reuses the Plan A FROZEN 4-channel layout
/// ([`pair_hist_binsums_len`], index `(feature * n_bins + bin) * 4 + histId`), the guard
/// block, the backend-dispatched channel, the empty short-circuit, and the no-readback
/// seam UNCHANGED — the only differences are `n_bins = 256` (validated below) and the
/// dispatched kernel symbol.
///
/// `n_bins` MUST be `256` (the 8-bit line size); anything else is rejected with
/// [`CbError::Degenerate`]. Inputs, guards, and the empty/typed-error semantics match
/// [`launch_pairwise_hist_handle`] (see its docs). No `unwrap`/`expect`/`panic`/indexing
/// in this production helper.
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist_8bit_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    one_hot: bool,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_8bit_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, one_hot,
    )
}

/// The ONE 8-bit-atomics pairwise launch geometry (IN-02 — one place). Clones
/// [`launch_pairwise_hist_into`] with `n_bins` pinned to `256` and dispatches
/// [`pairwise_hist_8bit_atomics_kernel`] (direct global atomics, no shared-mem
/// pre-reduce). Reuses the Plan A guard block, the 4-channel layout, the
/// backend-dispatched channel, and the no-readback seam unchanged. The caller owns the
/// `client` lifecycle so a read-back uses the SAME client that allocated the handle.
#[allow(clippy::too_many_arguments)]
fn launch_pairwise_hist_8bit_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    one_hot: bool,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-07): identical to the non-binary arm — a mismatch would read
    // out of bounds on the device. Surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST (the 256-bin line makes `n_features * 256 * 4` overflow
    // checking load-bearing — T-07.4-07). Reuse the checked cindex stride for the length
    // guard below, never re-multiplying.
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-07.4-07): BOTH pair endpoints index cindex (Pitfall 3); every
    // cindex bin must fit the 256-bin line (the kernel does not mask).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // The 8-bit-atomics family is defined for the 256-bin line ONLY (n_bins == 1 << 8);
    // anything else is not this family.
    if n_bins != 256 {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist 8-bit-atomics fill expects n_bins == 256 (1 << 8), got n_bins={n_bins}"
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): f64 on rocm/cuda/cpu, f32 on
    // wgpu (WGSL has no f64 atomics — read back and UPCAST). Reused VERBATIM from the
    // non-binary arm. NO read-back here (SC-3 / Pitfall 5).
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_8bit_atomics_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            one_hot,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_8bit_atomics_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
            one_hot,
        );
        Ok(h)
    }
}

/// Host-readback wrapper over the 8-bit-atomics pairwise-histogram fill: launch
/// device-resident, then read the 4-channel `binSums` handle back to a host `Vec<f64>`.
/// This is the seam the all-backend self-oracle exercises; it is NOT the histogram
/// hand-off path (that is [`launch_pairwise_hist_8bit_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_8bit_into`]); this
/// wrapper constructs the client ONCE and uses that SAME client for both the launch and
/// the read-back. A device read-back failure surfaces [`CbError::Degenerate`], never a
/// silent all-zero buffer. Empty input returns an empty `Vec` (no launch).
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_hist_8bit(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
    one_hot: bool,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_8bit_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, one_hot,
    )?;

    read_pair_binsums_f64(&client, handle)
}

/// Launch the half-byte (4-bit, 16-bin) pairwise histogram fill device-resident and return
/// the 4-channel `binSums` Handle WITHOUT reading it back (the SC-3 / D-7.4-03 hand-off
/// path). This is the SEPARATE launch arm for the structurally DISTINCT half-byte family
/// (D-7.4-02 — upstream `pairwise_hist_half_byte.cu`): the histogram line is a FIXED 16-bin
/// (4-bit) line (the comptime `HALF_BYTE_BINS`), and the family takes NO one-hot overlay
/// upstream (there is no `pairwise_hist_half_byte_one_hot.cu`). It reuses the Plan A FROZEN
/// 4-channel layout ([`pair_hist_binsums_len`], index `(feature * n_bins + bin) * 4 +
/// histId`), the guard block, the backend-dispatched channel, the empty short-circuit, and
/// the no-readback seam UNCHANGED — the only differences are `n_bins = 16` (validated below)
/// and the dispatched kernel symbol.
///
/// `n_bins` MUST be `16` (the half-byte line size, [`crate::kernels::HALF_BYTE_BINS`]);
/// anything else is rejected with [`CbError::Degenerate`]. There is no `one_hot` parameter
/// (the half-byte family has no one-hot overlay). Inputs, guards, and the empty/typed-error
/// semantics match [`launch_pairwise_hist_handle`] (see its docs). No
/// `unwrap`/`expect`/`panic`/indexing in this production helper.
pub fn launch_pairwise_hist_half_byte_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_half_byte_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )
}

/// The ONE half-byte pairwise launch geometry (IN-02 — one place). Clones
/// [`launch_pairwise_hist_into`] with `n_bins` pinned to [`crate::kernels::HALF_BYTE_BINS`]
/// (16) and dispatches [`pairwise_hist_half_byte_kernel`] (fixed 16-bin line, nibble-masked
/// bins, no one-hot). Reuses the Plan A guard block, the 4-channel layout, the
/// backend-dispatched channel, and the no-readback seam unchanged. The caller owns the
/// `client` lifecycle so a read-back uses the SAME client that allocated the handle.
fn launch_pairwise_hist_half_byte_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-11): identical to the non-binary/8-bit arms — a mismatch would
    // read out of bounds on the device. Surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST. Reuse the checked cindex stride for the length guard below,
    // never re-multiplying.
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-07.4-11): BOTH pair endpoints index cindex (Pitfall 3); every
    // cindex bin must fit the 16-bin line (the kernel masks to the nibble, but the host
    // reference indexes RAW, so the value-range guard keeps the two in lock-step).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // The half-byte family is defined for the 16-bin line ONLY (n_bins == HALF_BYTE_BINS);
    // anything else is not this family.
    if n_bins != crate::kernels::HALF_BYTE_BINS {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist half-byte fill expects n_bins == {} (1 << 4), got n_bins={n_bins}",
            crate::kernels::HALF_BYTE_BINS
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): f64 on rocm/cuda/cpu, f32 on
    // wgpu (WGSL has no f64 atomics — read back and UPCAST). Reused VERBATIM from the
    // non-binary/8-bit arms. NO read-back here (SC-3 / Pitfall 5).
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_half_byte_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_half_byte_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }
}

/// Host-readback wrapper over the half-byte pairwise-histogram fill: launch device-resident,
/// then read the 4-channel `binSums` handle back to a host `Vec<f64>`. This is the seam the
/// all-backend self-oracle exercises; it is NOT the histogram hand-off path (that is
/// [`launch_pairwise_hist_half_byte_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_half_byte_into`]); this
/// wrapper constructs the client ONCE and uses that SAME client for both the launch and the
/// read-back. A device read-back failure surfaces [`CbError::Degenerate`], never a silent
/// all-zero buffer. Empty input returns an empty `Vec` (no launch).
pub fn launch_pairwise_hist_half_byte(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_half_byte_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )?;

    read_pair_binsums_f64(&client, handle)
}

/// Launch the binary (1-bit, 2-bin) pairwise histogram fill device-resident and return the
/// 4-channel `binSums` Handle WITHOUT reading it back (the SC-3 / D-7.4-03 hand-off path).
/// This is the SEPARATE launch arm for the structurally DISTINCT binary family (D-7.4-02 —
/// upstream `pairwise_hist_binary.cu`): the histogram line is a FIXED 2-bin (1-bit) line (a
/// bin COUNT, NOT a warp literal), and the family takes NO one-hot overlay upstream (there is
/// no `pairwise_hist_binary_one_hot.cu`). It reuses the Plan A FROZEN 4-channel layout
/// ([`pair_hist_binsums_len`], index `(feature * n_bins + bin) * 4 + histId`), the guard
/// block, the backend-dispatched channel, the empty short-circuit, and the no-readback seam
/// UNCHANGED — the only differences are `n_bins = 2` (validated below) and the dispatched
/// kernel symbol.
///
/// `n_bins` MUST be `2` (the binary line size); anything else is rejected with
/// [`CbError::Degenerate`]. There is no `one_hot` parameter (the binary family has no one-hot
/// overlay). Inputs, guards, and the empty/typed-error semantics match
/// [`launch_pairwise_hist_handle`] (see its docs). No `unwrap`/`expect`/`panic`/indexing in
/// this production helper.
pub fn launch_pairwise_hist_binary_handle(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_hist_binary_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )
}

/// The ONE binary pairwise launch geometry (IN-02 — one place). Clones
/// [`launch_pairwise_hist_into`] with `n_bins` pinned to `2` and dispatches
/// [`pairwise_hist_binary_kernel`] (fixed 2-bin line, bit-masked bins, no one-hot). Reuses the
/// Plan A guard block, the 4-channel layout, the backend-dispatched channel, and the
/// no-readback seam unchanged. The caller owns the `client` lifecycle so a read-back uses the
/// SAME client that allocated the handle.
fn launch_pairwise_hist_binary_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    let n_pairs = pair_i.len();

    // Shape guards (T-07.4-15): identical to the non-binary/8-bit/half-byte arms — a mismatch
    // would read out of bounds on the device. Surface a typed error BEFORE launch (no panic).
    if pair_j.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_j".to_owned(),
            expected: n_pairs,
            actual: pair_j.len(),
        });
    }
    if pair_weight.len() != n_pairs {
        return Err(CbError::LengthMismatch {
            column: "pair_weight".to_owned(),
            expected: n_pairs,
            actual: pair_weight.len(),
        });
    }

    // Overflow guards FIRST. Reuse the checked cindex stride for the length guard below,
    // never re-multiplying.
    let cindex_stride = n_features.checked_mul(n_objects).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_objects ({n_objects}) overflows usize (cindex stride)"
        ))
    })?;
    if pair_hist_binsums_len_checked(n_bins, n_features).is_none() {
        return Err(CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize (binSums length)"
        )));
    }

    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-07.4-15): BOTH pair endpoints index cindex (Pitfall 3); every
    // cindex bin must fit the 2-bin line (the kernel masks to the bit, but the host reference
    // indexes RAW, so the value-range guard keeps the two in lock-step).
    if let Some(&bad) = pair_i.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_i value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = pair_j.iter().find(|&&ix| (ix as usize) >= n_objects) {
        return Err(CbError::OutOfRange(format!(
            "pair_j value {bad} >= n_objects ({n_objects}); object id would read cindex out of bounds"
        )));
    }
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    if n_pairs == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // The binary family is defined for the 2-bin line ONLY (n_bins == 2); anything else is not
    // this family.
    if n_bins != 2 {
        return Err(CbError::Degenerate(format!(
            "pairwise_hist binary fill expects n_bins == 2 (1 << 1), got n_bins={n_bins}"
        )));
    }

    // Launch geometry: enough cubes to cover `n_pairs` (the grid-stride loop handles any
    // surplus via the total-thread-count stride).
    let num_cubes = n_pairs.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let bin_sums_len = pair_hist_binsums_len(n_bins, n_features);
    let cindex_handle = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let pair_i_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_i.to_vec()));
    let pair_j_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_j.to_vec()));

    // Channel float-type dispatch (RESEARCH A1 / Pitfall 1): f64 on rocm/cuda/cpu, f32 on wgpu
    // (WGSL has no f64 atomics — read back and UPCAST). Reused VERBATIM from the
    // non-binary/8-bit/half-byte arms. NO read-back here (SC-3 / Pitfall 5).
    #[cfg(feature = "wgpu")]
    {
        let weight_f32: Vec<f32> = pair_weight.iter().map(|&v| v as f32).collect();
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(weight_f32));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
        pairwise_hist_binary_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let weight_handle = client.create(cubecl::bytes::Bytes::from_elems(pair_weight.to_vec()));
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
        pairwise_hist_binary_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(pair_i_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(pair_j_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(weight_handle, n_pairs) },
            unsafe { ArrayArg::from_raw_parts(cindex_handle, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
            n_features as u32,
            n_objects as u32,
        );
        Ok(h)
    }
}

/// Host-readback wrapper over the binary pairwise-histogram fill: launch device-resident, then
/// read the 4-channel `binSums` handle back to a host `Vec<f64>`. This is the seam the
/// all-backend self-oracle exercises; it is NOT the histogram hand-off path (that is
/// [`launch_pairwise_hist_binary_handle`], which never reads back).
///
/// The launch geometry lives in ONE place ([`launch_pairwise_hist_binary_into`]); this wrapper
/// constructs the client ONCE and uses that SAME client for both the launch and the read-back.
/// A device read-back failure surfaces [`CbError::Degenerate`], never a silent all-zero
/// buffer. Empty input returns an empty `Vec` (no launch).
pub fn launch_pairwise_hist_binary(
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    n_objects: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);

    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    let handle = launch_pairwise_hist_binary_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features,
    )?;

    read_pair_binsums_f64(&client, handle)
}
