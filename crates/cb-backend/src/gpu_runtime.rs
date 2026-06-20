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
    partition_split_kernel, partition_update_kernel,
    pointwise_hist2_binary_kernel,
    pointwise_hist2_half_byte_kernel, pointwise_hist2_nonbinary_kernel,
    pairwise_make_derivatives_kernel, quantile_gradient_kernel, scan_update_pairwise_kernel,
    scan_update_pointwise_kernel, select_best_split_kernel, SCORE_FN_COSINE, SCORE_FN_L2,
    SCORE_FN_LOO_L2, SCORE_FN_SAT_L2, SCORE_FN_SOLAR_L2,
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
///
/// `score_fn` selects the comptime score calcer arm of [`find_optimal_split_kernel`]:
/// [`SCORE_FN_L2`] (Plan A), or the Plan-E [`SCORE_FN_COSINE`] /
/// [`SCORE_FN_SOLAR_L2`] / [`SCORE_FN_LOO_L2`] / [`SCORE_FN_SAT_L2`] — each transcribed
/// VERBATIM from the FROZEN `cb-compute/src/score.rs`. An unknown `score_fn` surfaces a
/// typed [`CbError::OutOfRange`] BEFORE launch (no silent wrong-arm dispatch).
#[allow(clippy::too_many_arguments)]
pub fn launch_find_optimal_split_pointwise(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<(Option<BestSplit>, Vec<f64>)> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_find_optimal_split_pointwise_into(
        &client, der1, weight, cindex, indices, n_bins, n_features, scaled_l2, score_fn,
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
    score_fn: u32,
) -> CbResult<(Option<BestSplit>, Vec<f64>)> {
    let n = der1.len();

    // Reject an unknown score-fn selector BEFORE any launch (no silent wrong-arm
    // dispatch / no garbage score buffer). The kernel only monomorphizes the five
    // transcribed arms; an out-of-range selector is a caller bug, surfaced typed.
    if score_fn != SCORE_FN_L2
        && score_fn != SCORE_FN_COSINE
        && score_fn != SCORE_FN_SOLAR_L2
        && score_fn != SCORE_FN_LOO_L2
        && score_fn != SCORE_FN_SAT_L2
    {
        return Err(CbError::OutOfRange(format!(
            "unknown score_fn selector ({score_fn}); expected one of \
             L2/Cosine/SolarL2/LOOL2/SatL2"
        )));
    }

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
            score_fn,
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
            score_fn,
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
// Phase 7.5 Plan B — the device-resident pointwise scan/update bridge (GPU-01
// scan-update slice; D-7.5-03 — the `ScanPointwiseHistograms` /
// `UpdatePointwiseHistograms` transform 7.3 explicitly deferred). It consumes the
// FROZEN 7.3 2-channel histogram handle IN PLACE (NO host round-trip inserted at the
// FILL->scan seam, D-7.5-03), turning per-bin (Σ der1, Σ weight) into cumulative
// "left-of-border" leaf stats (an inclusive prefix-sum over the per-feature bin axis)
// the Plan-A scorer consumes (`left = scan[b]`, `right = total - scan[b]`). The bulk
// histogram never leaves the device; only the cumulative result is read back ONCE as
// the self-oracle observation (the analog of the histogram oracle reading binSums
// back once). Cross-oracled in `kernels/score_split.rs::scan` against the host ordered
// `sum_f64` prefix reference. SCOPE: n_bins <= CUBE_DIM (single-cube scan
// precondition); n_bins > CUBE_DIM surfaces a typed error (the EXPLICIT tracked
// cross-cube-carry forward dependency, RESEARCH A1 / Open Q1).
// ===========================================================================

/// Fill the FROZEN 7.3 device-resident 2-channel histogram, then run the
/// device-resident **scan/update** that turns it into cumulative "left-of-border"
/// leaf stats, returning the cumulative buffer as a host `Vec<f64>` (the self-oracle
/// observation, length `n_features * n_bins * 2`, flat `(feature * n_bins + bin) * 2 +
/// channel` order — the SAME FROZEN layout the histogram uses).
///
/// The histogram is filled with [`launch_pointwise_hist2_into`] (the FROZEN 7.3 seam)
/// and the `binSums` Handle is consumed IN PLACE by [`scan_update_pointwise_kernel`] —
/// the bulk histogram NEVER crosses to the host at the FILL->scan seam (D-7.5-03 /
/// D-05). For each feature `f`, channel `c`, border `b` the output cell holds the
/// INCLUSIVE prefix `Σ_{bin=0}^{b} binSums[(f, bin, c)]`, so a candidate at border `b`
/// reads `left = cumulative[b]`, `right = cumulative[n_bins - 1] - cumulative[b]`.
///
/// `der1` (UNWEIGHTED, the 7.2 seam contract) / `weight` (channel 1), length `n`;
/// `cindex` (feature-major quantized bins), length `n_features * n`; `indices` (object
/// visiting order), length `n`; `n_bins` is the per-feature border count.
///
/// Empty input (`n == 0` / `n_features == 0` / `n_bins == 0`) short-circuits to
/// `Vec::new()` with NO launch and NO read-back of a 0-length handle (Pitfall 3/5).
/// Mismatched input lengths / out-of-range bin/object values surface a typed
/// [`CbError`] BEFORE launch (the FROZEN 7.3 guards in `launch_pointwise_hist2_into`).
///
/// SCOPE GUARD (RESEARCH A1 / Open Q1): `n_bins > CUBE_DIM` surfaces a typed
/// [`CbError::OutOfRange`] — the single-cube `scan_update_pointwise_kernel` (which
/// reuses `block_scan_kernel`) cannot carry across cubes, and the cross-cube carry for
/// 8-bit (256-bin) features is the EXPLICIT tracked forward dependency (recorded in the
/// 07.5-02 SUMMARY — NOT a silent truncation). No `unwrap`/`expect`/`panic`/indexing in
/// this production helper (workspace lints + D-13).
pub fn launch_scan_update_pointwise(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_scan_update_pointwise_into(&client, der1, weight, cindex, indices, n_bins, n_features)
}

/// The ONE scan/update launch geometry (the histogram-seam IN-02 precedent — one
/// place). Fills the device histogram, launches the scan/update kernel consuming that
/// handle IN PLACE, and reads back the cumulative buffer. The caller owns the `client`
/// lifecycle so the read-back uses the SAME client that allocated the handles (a CubeCL
/// Handle is bound to its originating client — see [`launch_der_binary_into`]).
fn launch_scan_update_pointwise_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let n = der1.len();

    // Empty short-circuit FIRST (Pitfall 3/5): no histogram, no launch, no 0-len read.
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    // SCOPE GUARD (RESEARCH A1 / Open Q1): the single-cube scan_update_pointwise_kernel
    // (reusing block_scan_kernel) is correct only for n_bins <= CUBE_DIM. Launching it
    // for n_bins > CUBE_DIM would scan only the first CUBE_DIM bins per (feature,
    // channel) and silently return a WRONG prefix for the rest — reject with a typed
    // error until the cross-cube carry (the tracked 7.2/7.3 forward dependency) lands.
    if n_bins > CUBE_DIM {
        return Err(CbError::OutOfRange(format!(
            "launch_scan_update_pointwise supports n_bins <= {CUBE_DIM} until the \
             cross-cube scan carry lands (RESEARCH A1 / Open Q1, the tracked 7.2/7.3 \
             forward dependency for 8-bit/256-bin features); got n_bins = {n_bins}"
        )));
    }

    // Cumulative-buffer length overflow guard (T-07.5-02-02): the output length and the
    // kernel's cell index are products of caller-supplied dimensions. Reject a degenerate
    // dimension with a typed range error BEFORE forming the product unchecked (a wrapping
    // multiply would address the wrong cell; a debug build would panic). REUSE the FROZEN
    // 7.3 checked length helper so the cumulative buffer matches the binSums layout exactly.
    let cumulative_len = hist2_binsums_len_checked(n_bins, n_features).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {HIST_CHANNELS} overflows usize \
             (cumulative length)"
        ))
    })?;

    // The kernel needs n_bins as a u32 (one unit per bin within a single cube). Bound it
    // so the cast cannot silently truncate a degenerate dimension (n_bins <= CUBE_DIM is
    // already guaranteed above, so this never actually fails, but the guard keeps the
    // "typed error, never truncate" contract uniform with the other seams).
    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel bin axis)"))
    })?;

    // Number of (feature, channel) scan axes = n_features * HIST_CHANNELS, each scanned
    // by ONE cube. Guard the cube-count product against overflow before the cast.
    let num_cubes = n_features.checked_mul(HIST_CHANNELS).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * {HIST_CHANNELS} overflows usize (scan cube count)"
        ))
    })?;
    let num_cubes_u32 = u32::try_from(num_cubes).map_err(|_| {
        CbError::OutOfRange(format!("scan cube count ({num_cubes}) exceeds u32"))
    })?;

    // Fill the FROZEN 7.3 device-resident 2-channel histogram (this runs the FROZEN
    // length / value-range guards on der1/weight/cindex/indices BEFORE any launch, and
    // returns a device HANDLE with NO read-back). The bulk histogram stays device-resident.
    let bin_sums = launch_pointwise_hist2_into(client, der1, weight, cindex, indices, n_bins, n_features)?;

    // Launch geometry: ONE cube of CUBE_DIM units per (feature, channel) scan axis. The
    // kernel decodes feature = CUBE_POS / 2, channel = CUBE_POS % 2, and scans n_bins
    // (<= CUBE_DIM) bins via the single-cube block-scan mechanism.
    let count = CubeCount::Static(num_cubes_u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // The cumulative output buffer matches the FROZEN binSums layout / channel float
    // type: f64 on rocm/cuda/cpu, f32 on wgpu (RESEARCH A1) — read back and UPCAST to
    // f64 via read_binsums_f64. Zero-initialised (the kernel writes every real bin cell).
    #[cfg(feature = "wgpu")]
    let cumulative_handle = {
        let cumulative_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; cumulative_len]));
        scan_update_pointwise_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, cumulative_len) },
            unsafe { ArrayArg::from_raw_parts(cumulative_h.clone(), cumulative_len) },
            n_bins_u32,
        );
        cumulative_h
    };

    #[cfg(not(feature = "wgpu"))]
    let cumulative_handle = {
        let cumulative_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; cumulative_len]));
        scan_update_pointwise_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, cumulative_len) },
            unsafe { ArrayArg::from_raw_parts(cumulative_h.clone(), cumulative_len) },
            n_bins_u32,
        );
        cumulative_h
    };

    // Read back the cumulative buffer (UPCAST f32->f64 on wgpu, direct f64 elsewhere) via
    // the FROZEN binSums read-back path (same layout). A read-back failure surfaces as
    // CbError::Degenerate, never a silent zero buffer (WR-05 / T-07.5-02-04).
    read_binsums_f64(client, cumulative_handle)
}

// ===========================================================================
// Phase 7.5 Plan C — the device-resident partition split + partition update seams, and
// the host-light single-tree grow loop driver (GPU-01 grow-loop slice; D-7.5-02 / D-05).
// The partition split (forward-bit doc-routing reorder == `cb_train::leaf_index`) and
// partition update (per-partition Σ der1 / Σ weight reduce == upstream
// `UpdatePartitionProps`) stay ENTIRELY device-resident: handle-in / handle-out, NO
// `read_one` on the bulk routing (D-05). The grow loop threads ONE `ComputeClient`
// through the whole tree, reading back ONLY the O(1) BestSplit descriptor per level and
// the final `2^depth` part-stats — never the full histogram/partition buffer (the
// forbidden host hybrid). Cross-oracled in `kernels/grow_loop.rs` against an INLINE
// transcription of `cb_train::greedy_tensor_search_oblivious`/`leaf_index` + the
// read-only `cb_compute::calc_average` leaf formula.
// ===========================================================================

/// Upload a host der1/weight float column onto `client` as a device handle, casting to
/// the channel float type (f32 on wgpu, f64 elsewhere — RESEARCH A1). Used by the grow
/// loop to materialize the resident der1/weight/cumulative buffers ONCE per tree. The
/// empty case short-circuits to a zero-length handle (Pitfall 5).
pub(crate) fn upload_channel_floats(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    values: &[f64],
) -> Handle {
    if values.is_empty() {
        return client.empty(0);
    }
    #[cfg(feature = "wgpu")]
    {
        let v: Vec<f32> = values.iter().map(|&x| x as f32).collect();
        client.create(cubecl::bytes::Bytes::from_elems(v))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()))
    }
}

/// Read a device part-stats handle (channel float type) back to a host `Vec<f64>`,
/// UPCASTING the f32 channel on wgpu (RESEARCH A1) — the part-stats sibling of
/// [`read_binsums_f64`]/[`read_scores_f64`]. A read-back failure surfaces
/// [`CbError::Degenerate`] (WR-05), never a silent zero buffer.
pub(crate) fn read_part_stats_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("part-stats read-back failed: {e:?}")))?;
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

/// Read a device `u32` handle (the partition `leaf_of`) back to a host `Vec<u32>`. A
/// read-back failure surfaces [`CbError::Degenerate`] (WR-05). Used by the cross-oracle
/// to validate the device per-object leaf assignment against `cb_train::leaf_index`; the
/// grow loop itself never reads the bulk routing back (D-05) — this is the test seam.
pub(crate) fn read_u32_handle(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<u32>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("u32 handle read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
}

/// Apply ONE split's forward-bit doc-routing reorder device-resident: returns a NEW
/// `leaf_of` handle with bit `level_bit` set for every object whose quantized bin on
/// `feature` is `> bin` (== `cb_train::leaf_index`'s `idx |= 1 << i`, Pitfall 6). The
/// bulk routing stays device-resident — NO `read_one` here (D-05). The caller threads
/// ONE `&client`; `der1` (resident handle), `cindex`/`indices`/`leaf_of` are device
/// handles bound to that client.
///
/// `n` is the object count; `cindex_stride` is `n_features * n` (the resident cindex
/// length). `feature`/`bin` are the chosen split; `level_bit` is the level's leaf bit.
/// Mismatched/degenerate dimensions and a `feature`/`bin` out of range surface a typed
/// [`CbError`] BEFORE launch (the host validated the VALUE ranges on upload). Empty
/// (`n == 0`) returns a zero-length handle with NO launch (Pitfall 5). No
/// `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_partition_split_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: Handle,
    cindex: Handle,
    indices: Handle,
    leaf_of: Handle,
    n: usize,
    cindex_stride: usize,
    feature: u32,
    bin: u32,
    level_bit: u32,
) -> CbResult<Handle> {
    if n == 0 {
        return Ok(client.empty(0));
    }

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // The new leaf_of starts as a fresh device buffer (the kernel writes every object it
    // strides; idle lanes write nothing, but every object is covered by the grid-stride).
    let new_leaf_of = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));

    #[cfg(feature = "wgpu")]
    partition_split_kernel::launch::<f32, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(der1, n) },
        unsafe { ArrayArg::from_raw_parts(cindex, cindex_stride) },
        unsafe { ArrayArg::from_raw_parts(indices, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of, n) },
        unsafe { ArrayArg::from_raw_parts(new_leaf_of.clone(), n) },
        feature,
        bin,
        level_bit,
    );

    #[cfg(not(feature = "wgpu"))]
    partition_split_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(der1, n) },
        unsafe { ArrayArg::from_raw_parts(cindex, cindex_stride) },
        unsafe { ArrayArg::from_raw_parts(indices, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of, n) },
        unsafe { ArrayArg::from_raw_parts(new_leaf_of.clone(), n) },
        feature,
        bin,
        level_bit,
    );

    Ok(new_leaf_of)
}

/// Recompute the per-partition `Σ der1` / `Σ weight` device-resident after a split:
/// returns a NEW `part_stats` handle of length `n_parts * 2` (channel 0 = Σ der1,
/// channel 1 = Σ weight) reduced over the resident `leaf_of` partition via the in-kernel
/// atomic merge (D-03). The bulk routing stays device-resident — NO `read_one` here
/// (D-05); the grow loop reads back the part-stats ONCE at the leaves.
///
/// `n` is the object count; `n_parts` is `2^level` (the current partition count). The
/// host validated `leaf_of[obj] < n_parts` on upload so the atomic store stays in bounds.
/// `n_parts * 2` overflow surfaces a typed [`CbError::OutOfRange`]. Empty (`n == 0` or
/// `n_parts == 0`) returns a zero-length handle with NO launch (Pitfall 5). No
/// `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
pub(crate) fn launch_partition_update_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: Handle,
    weight: Handle,
    indices: Handle,
    leaf_of: Handle,
    n: usize,
    n_parts: usize,
) -> CbResult<Handle> {
    if n == 0 || n_parts == 0 {
        return Ok(client.empty(0));
    }

    let part_stats_len = n_parts.checked_mul(2).ok_or_else(|| {
        CbError::OutOfRange(format!("n_parts ({n_parts}) * 2 overflows usize (part-stats length)"))
    })?;

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    #[cfg(feature = "wgpu")]
    let part_stats = {
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; part_stats_len]));
        partition_update_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1, n) },
            unsafe { ArrayArg::from_raw_parts(weight, n) },
            unsafe { ArrayArg::from_raw_parts(indices, n) },
            unsafe { ArrayArg::from_raw_parts(leaf_of, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), part_stats_len) },
        );
        h
    };

    #[cfg(not(feature = "wgpu"))]
    let part_stats = {
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; part_stats_len]));
        partition_update_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1, n) },
            unsafe { ArrayArg::from_raw_parts(weight, n) },
            unsafe { ArrayArg::from_raw_parts(indices, n) },
            unsafe { ArrayArg::from_raw_parts(leaf_of, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), part_stats_len) },
        );
        h
    };

    Ok(part_stats)
}

/// The output of [`grow_oblivious_tree`]: the per-level chosen splits, the per-object
/// leaf assignment (`leaf_of`, the SC-3 structure observation), and the per-leaf values.
///
/// `splits[level] = (feature, bin)` is the level's chosen split in FORWARD-bit order
/// (split `level` -> leaf bit `level`, matching `cb_train::leaf_index`). `leaf_of[obj]`
/// is object `obj`'s final leaf index (`0..2^depth`). `leaf_values[leaf]` is the
/// `cb_compute::calc_average`-estimated value of leaf `leaf` over the read-back
/// `2^depth` part-stats. `part_stats` is the raw `[Σ der1, Σ weight]`-per-leaf buffer
/// (length `2^depth * 2`) the leaf values were estimated from (kept for the oracle's
/// leaf-value divergence report).
#[derive(Clone, Debug, PartialEq)]
pub struct GrownTree {
    /// The per-level chosen `(feature, bin)` split sequence (FORWARD-bit order).
    pub splits: Vec<(u32, u32)>,
    /// The per-object final leaf index (`0..2^depth`) — the SC-3 structure observation.
    pub leaf_of: Vec<u32>,
    /// The per-leaf estimated value (`calc_average(Σ der1, Σ weight, scaled_l2)`).
    pub leaf_values: Vec<f64>,
    /// The raw read-back `[Σ der1, Σ weight]`-per-leaf part-stats (length `2^depth * 2`).
    pub part_stats: Vec<f64>,
}

// ===========================================================================
// Phase 7.5 Plan C — the host-light single-tree grow loop driver (GPU-01 grow-loop
// slice; D-7.5-02 / D-05). Mirrors upstream
// `oblivious_tree_doc_parallel_structure_searcher.cpp:63-158` and the CPU
// `greedy_tensor_search_oblivious_perturbed` per-level skeleton: per depth fill (7.3)
// -> scan/update (Plan B, implicit in the score kernel's left/right fold) -> score +
// deterministic argmin (Plan A) -> ONE O(1) BestSplit read-back -> host integer split
// decision (strict first-wins / lowest-(feature,bin) tie-break == `select_best_candidate`)
// -> partition-split (forward-bit) -> partition-update, over persistent device buffers
// threaded through ONE `ComputeClient`, then at the leaves ONE read-back of the
// `2^depth` part-stats and host leaf values via `cb_compute::calc_average`. The bulk
// histogram / partition / doc-routing stays device-resident across launches; reading
// the full histogram/partition buffer to host per level is the FORBIDDEN D-05 hybrid.
// ===========================================================================

/// Grow ONE oblivious tree device-resident over the compile-time [`SelectedRuntime`],
/// returning the chosen split sequence + per-object leaf assignment + per-leaf values
/// ([`GrownTree`]). The genuinely-new host-light driver (D-05 / D-7.5-02): per level it
/// chains the FROZEN 7.3 histogram fill -> the Plan-A device score + deterministic argmin
/// -> ONE O(1) [`BestSplit`] read-back -> the host integer split decision -> the Plan-C
/// device `partition_split` (forward-bit doc-routing) -> the Plan-C device
/// `partition_update` (per-partition Σ der1 / Σ weight reduce), over persistent device
/// handles threaded through ONE `ComputeClient`. At the leaves it reads back ONLY the
/// `2^depth` part-stats and computes leaf values via the FROZEN
/// `cb_compute::calc_average` formula. The bulk histogram / partition / doc-routing NEVER
/// crosses to host per level (the FORBIDDEN D-05 host hybrid).
///
/// # Inputs
///
/// `der1` (UNWEIGHTED, the 7.2 seam contract) / `weight` (channel 1), length `n` in
/// object order; `cindex` (feature-major quantized bins, length `n_features * n`:
/// `cindex[feature * n + obj]`); `indices` (object visiting order, length `n`); `n_bins`
/// is the per-feature border count (`1 << bits`); `n_features` the feature-group width;
/// `depth` the tree depth; `scaled_l2` the per-tree `cb_compute::scale_l2_reg` output.
///
/// # MVP scope (depth == 1) and the depth>1 forward dependency
///
/// The MVP grows a depth-1 oblivious tree (a single split / stump) — the genuinely
/// complete vertical slice the existing kernels support EXACTLY with the strict
/// O(1)-per-level read-back: level 0 has ONE partition (the root), so the whole-dataset
/// [`launch_find_optimal_split_pointwise_into`] stump score IS the exact CPU level-0
/// score, the O(1) [`BestSplit`] read-back is the only crossing, then one
/// `partition_split` + `partition_update` + the final 2-leaf part-stats read-back. A
/// `depth > 1` tree scores each level's candidate over the CURRENT 2^level partitions
/// (the per-partition / `fullPass = false` histogram, upstream `SubmitCompute(subsets,
/// ...)`); that partition-aware histogram is the EXPLICIT tracked forward dependency
/// (the FROZEN 7.3 fill is whole-dataset / `partCount = 1`, documented in
/// [`launch_pointwise_hist2_handle`]). `depth > 1` surfaces a typed
/// [`CbError::OutOfRange`] until the partition-aware fill lands — documented, NOT
/// silently cut, and NOT a wrong-structure stump score masquerading as a deep tree.
///
/// # Errors
///
/// - [`CbError::OutOfRange`] if `depth > 1` (the partition-aware-histogram forward
///   dependency), or if `2^depth * 2` / `n_features * n_bins` overflows `usize`.
/// - [`CbError::Degenerate`] if a level finds no candidate split, or any device
///   read-back fails (never a silent zero buffer, WR-05 / T-07.5-03-04).
/// - the FROZEN 7.3 length / value-range guards (via the score / partition launches).
///
/// No `unwrap`/`expect`/`panic`/indexing in this production driver (workspace lints +
/// D-13). Threads ONE `&client` through the whole tree (Pitfall 3); never reads a 0-len
/// handle.
///
/// `score_fn` selects the per-level split-score calcer ([`SCORE_FN_L2`] /
/// [`SCORE_FN_COSINE`] / [`SCORE_FN_SOLAR_L2`] / [`SCORE_FN_LOO_L2`] /
/// [`SCORE_FN_SAT_L2`]) — Cosine is a primary path alongside L2 (D-7.5-01). It is
/// forwarded verbatim to [`launch_find_optimal_split_pointwise_into`], which rejects an
/// unknown selector with a typed error before launch.
#[allow(clippy::too_many_arguments)]
pub fn grow_oblivious_tree(
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    depth: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<GrownTree> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    grow_oblivious_tree_into(
        &client, der1, weight, cindex, indices, n_bins, n_features, depth, scaled_l2, score_fn,
    )
}

/// The ONE grow-loop geometry (the histogram-seam IN-02 precedent — one place). Uploads
/// the resident der1/weight/cindex/indices/leaf_of handles ONCE onto `client`, runs the
/// per-depth launch chain over those persistent handles, and reads back ONLY the O(1)
/// BestSplit per level + the final `2^depth` part-stats. The caller owns the `client`
/// lifecycle so every read-back uses the SAME client that allocated the handles (a
/// CubeCL Handle is bound to its originating client — Pitfall 3 / see
/// [`launch_der_binary_into`]).
#[allow(clippy::too_many_arguments)]
fn grow_oblivious_tree_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    depth: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<GrownTree> {
    let n = der1.len();

    // Empty short-circuit (Pitfall 3/5): no objects / no candidates -> an empty tree.
    if n == 0 || n_features == 0 || n_bins == 0 || depth == 0 {
        return Ok(GrownTree {
            splits: Vec::new(),
            leaf_of: vec![0u32; n],
            leaf_values: Vec::new(),
            part_stats: Vec::new(),
        });
    }

    // MVP scope guard (the partition-aware-histogram forward dependency): the FROZEN 7.3
    // fill is whole-dataset (partCount == 1), so a single whole-dataset stump score is the
    // EXACT CPU level-0 score but NOT the per-partition level-L>0 score. Reject depth>1
    // with a typed error rather than silently scoring a stump and mislabeling it a deep
    // tree (which would be a wrong-structure fabrication). Documented, not cut.
    if depth > 1 {
        return Err(CbError::OutOfRange(format!(
            "grow_oblivious_tree supports depth <= 1 until the per-partition \
             (fullPass = false) histogram fill lands (the FROZEN 7.3 whole-dataset fill \
             scores a single binary partition; a depth>1 level scores over 2^level \
             partitions — the EXPLICIT tracked forward dependency, RESEARCH A2); got \
             depth = {depth}"
        )));
    }

    // 2^depth leaf count, overflow-checked (T-07.5-03-02): the part-stats buffer length and
    // the leaf-value loop bound are derived from this. checked_shl rejects a degenerate
    // depth before forming the product unchecked.
    let n_leaves = 1usize
        .checked_shl(depth as u32)
        .ok_or_else(|| CbError::OutOfRange(format!("2^depth overflows usize (depth = {depth})")))?;

    // The resident cindex stride (feature-major) and its overflow guard (the partition
    // split reads cindex[feature * n + obj]).
    let cindex_stride = n_features
        .checked_mul(n)
        .ok_or_else(|| CbError::OutOfRange(format!("n_features ({n_features}) * n ({n}) overflows usize")))?;
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Resident device handles uploaded ONCE (the D-05 persistent-buffer contract: one
    // client, buffers live across the whole tree's launches). der1/weight are channel-typed
    // (f32 on wgpu, f64 elsewhere); cindex/indices are u32; leaf_of starts all-zero (every
    // object in the root partition 0).
    let der1_h = upload_channel_floats(client, der1);
    let weight_h = upload_channel_floats(client, weight);
    let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));
    let mut leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));

    let mut splits: Vec<(u32, u32)> = Vec::with_capacity(depth);

    // The host-light per-depth loop (upstream :63-158). For the MVP depth == 1 this runs
    // once over the whole dataset (the single root partition); the loop is kept so the
    // partition-aware depth>1 path slots in at the per-level score step unchanged.
    for level in 0..depth {
        // (1) Device fill + score + deterministic argmin over the CURRENT partition. For
        //     level 0 (one partition) this is the whole-dataset stump score over the
        //     resident der1/weight/cindex/indices — the EXACT CPU level-0 score. The bulk
        //     histogram stays device-resident IN the score launch; only the O(1) BestSplit
        //     descriptor crosses back (the score kernel's per-candidate vector is the Plan-A
        //     self-oracle observation, not read here in the driver — D-05).
        let (best, _scores) = launch_find_optimal_split_pointwise_into(
            client, der1, weight, cindex, indices, n_bins, n_features, scaled_l2, score_fn,
        )?;

        // (2) The O(1) host integer split decision. A level with no candidate at all is a
        //     degenerate dataset (no feature has any bin) — surface a typed error rather
        //     than fabricating a split (T-07.5-03-04).
        let split = best.ok_or_else(|| {
            CbError::Degenerate(format!(
                "grow_oblivious_tree level {level}: no candidate split (degenerate histogram)"
            ))
        })?;
        splits.push((split.feature_id, split.bin_id));

        // (3) Device partition-split (forward-bit doc-routing, level -> bit level == the CPU
        //     `leaf_index` convention, Pitfall 6) — IN-PLACE on device, NO read-back here
        //     (D-05). The resident handles are cloned (a CubeCL Handle is a ref-counted
        //     buffer binding; cloning shares the device buffer, it does NOT copy).
        leaf_of_h = launch_partition_split_into(
            client,
            der1_h.clone(),
            cindex_h.clone(),
            indices_h.clone(),
            leaf_of_h,
            n,
            cindex_stride,
            split.feature_id,
            split.bin_id,
            level as u32,
        )?;
    }

    // (4) Device partition-update: the per-partition Σ der1 / Σ weight reduce over the
    //     final 2^depth partitions (upstream `UpdatePartitionProps`), device-resident.
    let part_stats_h = launch_partition_update_into(
        client,
        der1_h.clone(),
        weight_h.clone(),
        indices_h.clone(),
        leaf_of_h.clone(),
        n,
        n_leaves,
    )?;

    // (5) The leaves: ONE read-back of the 2^depth part-stats (the ONLY bulk-data crossing
    //     besides the O(1) per-level BestSplit — D-05). A read-back failure surfaces
    //     CbError::Degenerate, never a silent zero buffer (WR-05 / T-07.5-03-04).
    let part_stats = read_part_stats_f64(client, part_stats_h)?;

    // (6) Host leaf values via the FROZEN cb_compute::calc_average formula (leaf.rs:83-89):
    //     mu = count > 0 ? Σder1 / (Σweight + scaled_l2) : 0.0 (the count>0 guard transcribed,
    //     T-07.5-03-06 — no NaN/Inf from an empty leaf). part_stats is [Σder1, Σweight] per
    //     leaf in leaf-index order.
    let mut leaf_values = vec![0.0_f64; n_leaves];
    for leaf in 0..n_leaves {
        let sum = part_stats.get(leaf * 2).copied().unwrap_or(0.0);
        let cnt = part_stats.get(leaf * 2 + 1).copied().unwrap_or(0.0);
        if let Some(slot) = leaf_values.get_mut(leaf) {
            *slot = cb_compute::calc_average(sum, cnt, scaled_l2);
        }
    }

    // (7) The per-object leaf assignment (the SC-3 structure observation): the grow loop
    //     itself never reads the bulk routing back (D-05); this single read-back at the
    //     END is the oracle seam (the same crossing class as the final part-stats). A
    //     read-back failure surfaces CbError::Degenerate (WR-05).
    let leaf_of = read_u32_handle(client, leaf_of_h)?;

    Ok(GrownTree {
        splits,
        leaf_of,
        leaf_values,
        part_stats,
    })
}

// ===========================================================================
// Phase 7.5 Plan 04 — the device-resident MULTI-TREE Plain-boosting pass (GPU-01
// grow-loop slice; D-7.5-06 non-determinism budget). Loops `grow_oblivious_tree`
// (Plan C) over a full Plain-boosting run device-resident: each tree's leaf updates
// feed the next tree's residuals, with der1 recomputed DEVICE-SIDE between trees via
// the 7.2 der seam (`launch_der_binary_into` RMSE gradient), over ONE `ComputeClient`
// threaded through the whole run. Mirrors the CPU `cb_train::boosting` Plain skeleton
// (`boosting.rs:9-16`): per iteration `compute_gradients(approx, target)` -> grow one
// oblivious tree -> leaf delta `calc_average` -> store `learning_rate * delta` ->
// `approx[i] += leaf_value[leaf(i)]`. MVP: Plain boosting, foldCount == 1, symmetric
// oblivious, RMSE/L2 (ordered / non-symmetric OUT of scope).
// ===========================================================================

/// The output of [`grow_boosting_pass`]: the per-tree grown structure + values of a
/// full multi-tree Plain-boosting run. `trees[k]` is the `k`-th boosting iteration's
/// [`GrownTree`] (its `leaf_values` already scaled by `learning_rate` — the stored
/// `model.json` `leaf_values` convention, `cb_train::boosting` `boosting.rs:15-16`).
/// `iterations` is `trees.len()`; `learning_rate` is the per-tree leaf-value scale.
///
/// This is the SC-3 structure observation surface for the multi-tree cross-oracle:
/// EVERY tree's `splits` + `leaf_of` is asserted against the CPU multi-tree reference
/// EXACTLY across the whole run, and the per-tree `leaf_values` divergence is REPORTED
/// (the D-7.5-06 non-determinism budget — NOT the GPU-06 epsilon, 7.6's job).
#[derive(Clone, Debug, PartialEq)]
pub struct GrownModel {
    /// The per-iteration grown trees (each tree's `leaf_values` already `learning_rate`-scaled).
    pub trees: Vec<GrownTree>,
    /// The per-tree leaf-value scale applied per iteration.
    pub learning_rate: f64,
}

impl GrownModel {
    /// The boosting iteration count (`trees.len()`).
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.trees.len()
    }
}

/// Run a full MULTI-TREE Plain-boosting pass device-resident over the compile-time
/// [`SelectedRuntime`], returning the per-iteration grown trees ([`GrownModel`]). The
/// genuinely-new multi-iteration driver (D-05 / D-7.5-06): it loops
/// [`grow_oblivious_tree_into`] (Plan C) over `iterations`, threading ONE
/// `ComputeClient` through the whole run; between trees it recomputes the running
/// residual gradient `der1` DEVICE-SIDE via the 7.2 der seam
/// ([`launch_der_binary_into`], [`DerBinaryKernel::RmseGradient`] = `target - approx`),
/// after updating the running per-object approx from the just-grown tree
/// (`approx[i] += learning_rate * leaf_value[leaf_of[i]]`, the
/// `cb_train::boosting` Plain convention). The per-tree leaf values are scaled by
/// `learning_rate` in place (the stored `model.json` `leaf_values` convention).
///
/// # Inputs
///
/// `target` (the regression label, length `n`); `weight` (the per-object weight,
/// channel 1, length `n` — folded downstream by the 7.3 histogram, the 7.2 UNWEIGHTED
/// der contract); `cindex` (feature-major quantized bins, length `n_features * n`:
/// `cindex[feature * n + obj]`); `indices` (object visiting order, length `n`);
/// `n_bins` the per-feature border count; `n_features` the feature-group width;
/// `iterations` the boosting-iteration count; `learning_rate` the per-tree leaf-value
/// scale; `depth` the per-tree depth (MVP `depth == 1`); `scaled_l2` the per-tree
/// `cb_compute::scale_l2_reg` output.
///
/// The starting approx is all-zero (the RMSE-from-zero MVP — `boost_from_average` /
/// the target-mean bias is OUT of scope, the cross-oracle uses the SAME zero start);
/// the initial der1 is therefore `target - 0 = target`.
///
/// # MVP scope and the depth>1 forward dependency
///
/// Plain boosting, `foldCount == 1`, symmetric oblivious, RMSE/L2. Each tree is grown
/// by [`grow_oblivious_tree_into`], which itself supports `depth == 1` and surfaces a
/// typed [`CbError::OutOfRange`] for `depth > 1` (the partition-aware-histogram forward
/// dependency, documented in [`grow_oblivious_tree`]). The der recompute between trees
/// runs device-side via the 7.2 seam; the recomputed der1 is read back once per tree —
/// the SAME crossing class as the existing per-tree `leaf_of` / part-stats read-backs
/// (D-05: no bulk histogram / partition / doc-routing crosses, only the O(1)/leaf
/// data). der2 for RMSE is the constant `-1.0` ([`const_der_handle`]) — not materialized
/// here (the L2 score path uses Σ der1 / Σ weight, the Newton der2 is the
/// per-partition-histogram score follow-up, RESEARCH).
///
/// # Errors
///
/// - [`CbError::LengthMismatch`] if `weight`/`cindex` lengths disagree with `target`/`n`.
/// - [`CbError::OutOfRange`] on `iterations`/leaf bookkeeping overflow, or `depth > 1`
///   (propagated from [`grow_oblivious_tree_into`]).
/// - [`CbError::Degenerate`] if a tree finds no candidate split, or any device read-back
///   (the per-tree der recompute, leaf_of, or part-stats) fails — never a silent zero
///   buffer (WR-05 / T-07.5-04-04).
///
/// No `unwrap`/`expect`/`panic`/indexing in this production driver (workspace lints +
/// D-13). Threads ONE `&client` through the whole run (Pitfall 3 / T-07.5-04-03); never
/// reads a 0-len handle.
#[allow(clippy::too_many_arguments)]
pub fn grow_boosting_pass(
    target: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    iterations: usize,
    learning_rate: f64,
    depth: usize,
    scaled_l2: f64,
) -> CbResult<GrownModel> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    grow_boosting_pass_into(
        &client, target, weight, cindex, indices, n_bins, n_features, iterations,
        learning_rate, depth, scaled_l2,
    )
}

/// The ONE multi-tree boosting-pass geometry (the histogram-seam IN-02 precedent — one
/// place). Threads ONE `client` through every iteration: per tree it grows one
/// [`grow_oblivious_tree_into`] over the running residual `der1`, scales the tree's
/// leaf values by `learning_rate`, updates the running per-object approx from the
/// tree's `leaf_of` + (scaled) `leaf_values`, and recomputes `der1` DEVICE-SIDE on the
/// SAME client via the 7.2 der seam for the next iteration. The caller owns the
/// `client` lifecycle so every device read-back uses the SAME client that allocated the
/// handles (a CubeCL Handle is bound to its originating client — Pitfall 3 / see
/// [`launch_der_binary_into`]).
#[allow(clippy::too_many_arguments)]
fn grow_boosting_pass_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    target: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    iterations: usize,
    learning_rate: f64,
    depth: usize,
    scaled_l2: f64,
) -> CbResult<GrownModel> {
    let n = target.len();

    // Shape guards (T-07.5-04-01): every per-object input must agree on `n` so the
    // residual / approx update and the per-tree launches stay in bounds. Surface a typed
    // error rather than launching a malformed kernel (no panic).
    if weight.len() != n {
        return Err(CbError::LengthMismatch {
            column: "weight".to_owned(),
            expected: n,
            actual: weight.len(),
        });
    }

    // Empty short-circuit (Pitfall 3/5): no objects / no iterations / no candidates ->
    // an empty model. Never read a 0-len handle.
    if n == 0 || n_features == 0 || n_bins == 0 || iterations == 0 || depth == 0 {
        return Ok(GrownModel {
            trees: Vec::new(),
            learning_rate,
        });
    }

    // Leaf-bookkeeping overflow guard (T-07.5-04-02): the running approx update indexes
    // `leaf_values[leaf_of[obj]]` with `leaf_of[obj] < 2^depth`. Reject a degenerate
    // depth before the per-tree loop forms the product unchecked. (`grow_oblivious_tree_into`
    // re-checks per tree, but pin it here so the bound holds for the whole pass.)
    let _n_leaves = 1usize
        .checked_shl(depth as u32)
        .ok_or_else(|| CbError::OutOfRange(format!("2^depth overflows usize (depth = {depth})")))?;

    // The running per-object approx (starts all-zero — the RMSE-from-zero MVP; the
    // target-mean bias / boost_from_average is OUT of scope, the cross-oracle uses the
    // SAME zero start). The initial residual der1 is therefore `target - 0 = target`.
    let mut approx = vec![0.0_f64; n];

    // The running residual gradient feeding the NEXT tree. Iteration 0 sees the initial
    // residual (target, from the zero approx); each subsequent iteration recomputes it
    // device-side via the 7.2 der seam after the approx update.
    let mut der1: Vec<f64> = target.to_vec();

    let mut trees: Vec<GrownTree> = Vec::with_capacity(iterations);

    for _iter in 0..iterations {
        // (1) Grow ONE oblivious tree over the CURRENT residual der1 (host-light, Plan C)
        //     on the threaded client. The bulk histogram / partition / doc-routing stays
        //     device-resident inside this call; only the O(1) BestSplit per level + the
        //     2^depth part-stats + leaf_of cross back (the existing D-05 crossings).
        // The multi-tree Plain-boosting pass scores with L2 (Plan 04 scope); the
        // Cosine/Solar/LOO/Sat arms are exercised single-tree in Plan E.
        let mut tree = grow_oblivious_tree_into(
            client, &der1, weight, cindex, indices, n_bins, n_features, depth, scaled_l2,
            SCORE_FN_L2,
        )?;

        // (2) Scale the tree's leaf values by `learning_rate` in place — the stored
        //     `model.json` `leaf_values` convention (`cb_train::boosting` boosting.rs:15-16:
        //     store `learning_rate * delta` as the leaf value). The unscaled
        //     `calc_average` delta is the GrownTree's leaf_values; scale it here so the
        //     model carries the per-tree contribution actually added to the approx.
        for v in tree.leaf_values.iter_mut() {
            *v *= learning_rate;
        }

        // (3) Update the running per-object approx from the just-grown tree
        //     (`approx[i] += leaf_value[leaf_of[i]]`, the Plain convention). This reuses
        //     the tree's ALREADY-read-back leaf_of + (scaled) leaf_values — NO new bulk
        //     crossing (D-05). A leaf index out of range is impossible by construction
        //     (leaf_of[obj] < 2^depth), but read without indexing (D-13).
        for (obj, slot) in approx.iter_mut().enumerate() {
            if let Some(&leaf) = tree.leaf_of.get(obj) {
                if let Some(&v) = tree.leaf_values.get(leaf as usize) {
                    *slot += v;
                }
            }
        }

        trees.push(tree);

        // (4) Recompute the running residual der1 DEVICE-SIDE for the NEXT iteration via
        //     the 7.2 der seam (RMSE gradient `der1 = target - approx`), on the SAME
        //     client. Skip the final iteration (no next tree needs it). The der is read
        //     back once — the SAME crossing class as the per-tree leaf_of / part-stats
        //     read-backs (D-05). A read-back failure surfaces CbError::Degenerate (WR-05 /
        //     T-07.5-04-04), never a silent zero buffer.
        if _iter + 1 < iterations {
            let der1_h = launch_der_binary_into(client, &approx, target, DerBinaryKernel::RmseGradient)?;
            let bytes = client.read_one(der1_h).map_err(|e| {
                CbError::Degenerate(format!(
                    "boosting-pass der1 recompute read-back failed (iter {_iter}): {e:?}"
                ))
            })?;
            // The der kernel output is f64 on rocm/cuda/cpu; on wgpu the elementwise der
            // kernel still launches over f64 inputs/outputs (it uploads the f64 approx/
            // target and writes an f64 out_handle — see `launch_der_binary_into`).
            der1 = bytemuck::cast_slice::<u8, f64>(&bytes).to_vec();
            // Defensive length guard (T-07.5-04-04): a truncated read-back must not feed a
            // mis-sized residual into the next tree's launch. Surface it, never pad/zero.
            if der1.len() != n {
                return Err(CbError::Degenerate(format!(
                    "boosting-pass der1 recompute returned {} elements, expected {n} (iter {_iter})",
                    der1.len()
                )));
            }
        }
    }

    Ok(GrownModel {
        trees,
        learning_rate,
    })
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

// ===========================================================================
// Phase 7.5 Plan 06 — the PAIRWISE split scorer device seam (split_pairwise.cuh),
// the LAST GPU-01 slice (D-7.5-01). It consumes the FROZEN 7.4 4-channel pairwise
// histogram handle + a device der-sum scatter, scan/updates them device-resident
// (the deferred 7.4 transform, D-7.4-06), then performs a BOUNDED host read-back of
// the assembled per-(feature,bucket) statistics and runs the small per-leaf Cholesky
// solve + score via `cb_compute::calculate_pairwise_score` (RESEARCH Open Q3: a
// `#[cube]` dense SPD solve is awkward and the FROZEN CPU `pairwise_cholesky_solve`
// is the parity oracle; the bulk pairwise histogram stays device-resident — minimal
// round-trips, D-05). Single-client threaded; typed guards; never reads a 0-len
// handle. Cross-oracled in `kernels/score_split.rs::pairwise` +
// `kernels/grow_loop.rs::pairwise`.
// ===========================================================================

/// Read a pairwise der-sum device handle (channel float type) back to a host
/// `Vec<f64>`, UPCASTING the f32 channel on wgpu (RESEARCH A1) — the der-sum sibling of
/// [`read_pair_binsums_f64`]. A read-back failure surfaces [`CbError::Degenerate`]
/// (WR-05), never a silent zero buffer.
fn read_pair_der_sums_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("pairwise der-sums read-back failed: {e:?}")))?;
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

/// The device-resident pairwise **scan/update** seam: fill the FROZEN 7.4 4-channel
/// pairwise histogram device-resident, scan/update it IN PLACE (inclusive prefix per
/// channel), and read back the cumulative 4-channel buffer (the self-oracle observation
/// — the FILL→scan seam stays device-resident, only the cumulative result crosses,
/// D-7.5-03 / D-05). Mirrors [`launch_scan_update_pointwise`] for the 4-channel layout.
///
/// SCOPE (inherited from Plan B): correct only for `n_bins <= CUBE_DIM`; `n_bins >
/// CUBE_DIM` (8-bit/256-bin) surfaces a typed [`CbError::OutOfRange`] — the EXPLICIT
/// tracked cross-cube-carry follow-up (RESEARCH Open Q3), never a silent truncation.
///
/// Inputs are the FROZEN 7.4 pairwise-fill inputs (`pair_i`/`pair_j` object ids,
/// `pair_weight`, `cindex` feature-major, `n_objects`, `n_bins`, `n_features`, `bits`
/// in {5,6,7}, `one_hot`). Empty input returns an empty `Vec` (no launch). No
/// `unwrap`/`expect`/`panic`/indexing in this production helper.
#[allow(clippy::too_many_arguments)]
pub fn launch_scan_update_pairwise(
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
    launch_scan_update_pairwise_into(
        &client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )
}

/// The ONE pairwise scan/update geometry (IN-02 — one place). Fills the FROZEN 7.4
/// 4-channel histogram device-resident, launches the per-(feature, histId)-cube
/// inclusive prefix-sum consuming that handle IN PLACE, and reads back ONLY the
/// cumulative 4-channel buffer. The caller owns the `client` lifecycle so the read-back
/// uses the SAME client that allocated the handle.
#[allow(clippy::too_many_arguments)]
fn launch_scan_update_pairwise_into(
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
) -> CbResult<Vec<f64>> {
    // Empty short-circuit FIRST (Pitfall 3/5): no histogram, no launch, no 0-len read.
    if pair_i.is_empty() || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    // SCOPE GUARD (RESEARCH Open Q3 / inherited from Plan B): the single-cube
    // scan_update_pairwise_kernel is correct only for n_bins <= CUBE_DIM. Reject
    // n_bins > CUBE_DIM with a typed error until the cross-cube carry lands (the
    // tracked cross-cube-carry follow-up for 8-bit/256-bin features).
    if n_bins > CUBE_DIM {
        return Err(CbError::OutOfRange(format!(
            "launch_scan_update_pairwise supports n_bins <= {CUBE_DIM} until the \
             cross-cube scan carry lands (RESEARCH Open Q3, the tracked pairwise \
             cross-cube-carry follow-up for 8-bit/256-bin features); got n_bins = {n_bins}"
        )));
    }

    // Cumulative-buffer length overflow guard: REUSE the FROZEN 7.4 checked 4-channel
    // length helper so the cumulative buffer matches the binSums layout exactly.
    let cumulative_len = pair_hist_binsums_len_checked(n_bins, n_features).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {PAIR_HIST_CHANNELS} overflows usize \
             (pairwise cumulative length)"
        ))
    })?;

    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel bin axis)"))
    })?;

    // Number of (feature, histId) scan axes = n_features * PAIR_HIST_CHANNELS, each
    // scanned by ONE cube. Guard the cube-count product against overflow before the cast.
    let num_cubes = n_features.checked_mul(PAIR_HIST_CHANNELS).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * {PAIR_HIST_CHANNELS} overflows usize (scan cube count)"
        ))
    })?;
    let num_cubes_u32 = u32::try_from(num_cubes).map_err(|_| {
        CbError::OutOfRange(format!("pairwise scan cube count ({num_cubes}) exceeds u32"))
    })?;

    // Fill the FROZEN 7.4 device-resident 4-channel histogram (this runs the FROZEN
    // length / value-range guards on pair_i/pair_j/pair_weight/cindex BEFORE any launch,
    // and returns a device HANDLE with NO read-back). The bulk histogram stays resident.
    let bin_sums = launch_pairwise_hist_into(
        client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )?;

    // Launch geometry: ONE cube of CUBE_DIM units per (feature, histId) scan axis.
    let count = CubeCount::Static(num_cubes_u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // The cumulative output buffer matches the FROZEN 4-channel layout / channel float
    // type: f64 on rocm/cuda/cpu, f32 on wgpu (RESEARCH A1) — read back via
    // read_pair_binsums_f64. Zero-initialised (the kernel writes every real bin cell).
    #[cfg(feature = "wgpu")]
    let cumulative_handle = {
        let cumulative_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; cumulative_len]));
        scan_update_pairwise_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, cumulative_len) },
            unsafe { ArrayArg::from_raw_parts(cumulative_h.clone(), cumulative_len) },
            n_bins_u32,
        );
        cumulative_h
    };

    #[cfg(not(feature = "wgpu"))]
    let cumulative_handle = {
        let cumulative_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; cumulative_len]));
        scan_update_pairwise_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, cumulative_len) },
            unsafe { ArrayArg::from_raw_parts(cumulative_h.clone(), cumulative_len) },
            n_bins_u32,
        );
        cumulative_h
    };

    read_pair_binsums_f64(client, cumulative_handle)
}

/// The device-resident pairwise **make-derivatives** seam: scatter the pairwise-weighted
/// `der1` into the per-(feature, bucket) der-sum buffer device-resident (== upstream
/// `MakePointwiseDerivatives` over the single root leaf), and read back the bounded
/// `n_features * n_bins` der-sum descriptor. This is the pointwise der row the pairwise
/// scorer assembles into `der_sum[2*leaf+1] += Σ_bucket der_sums[leaf][bucket]`; the
/// heavy per-object scatter stays device-resident, only the bounded der-sum crosses
/// (RESEARCH Open Q3 / D-05). The output is laid out `der_sums[feature * n_bins + bin]`
/// (leaf 0 / root, the depth-1 MVP), the SAME order `cb_compute::compute_der_sums`
/// produces for `leaf_count == 1`.
///
/// `der1` (the pairwise-weighted first derivative, length `n`), `cindex` (feature-major
/// quantized bins, `cindex[feature * n + obj]`, length `n_features * n`), `indices`
/// (object visiting order, length `n`), `n_bins`/`n_features`. Empty input returns an
/// empty `Vec` (no launch). Mismatched lengths / out-of-range bin values surface a typed
/// [`CbError`] BEFORE launch. No `unwrap`/`expect`/`panic`/indexing.
pub fn launch_pairwise_make_derivatives(
    der1: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_make_derivatives_into(&client, der1, cindex, indices, n_bins, n_features)
}

/// The ONE pairwise make-derivatives geometry (IN-02 — one place). Uploads the resident
/// der1/cindex/indices handles onto `client`, launches the der-sum scatter, and reads
/// back ONLY the bounded `n_features * n_bins` der-sum descriptor. The caller owns the
/// `client` lifecycle so the read-back uses the SAME client that allocated the handles.
fn launch_pairwise_make_derivatives_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
) -> CbResult<Vec<f64>> {
    let n = der1.len();

    // Empty short-circuit FIRST (Pitfall 3/5).
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(Vec::new());
    }

    // Overflow guards BEFORE any unchecked product. The cindex stride `n_features * n`
    // and the der-sum length `n_features * n_bins` are products of caller dimensions.
    let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n ({n}) overflows usize (cindex stride)"
        ))
    })?;
    let der_sums_len = n_features.checked_mul(n_bins).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) overflows usize (der-sums length)"
        ))
    })?;

    // Shape guards (the kernel reads cindex[feature * n + obj] / indices[i]).
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }
    if indices.len() != n {
        return Err(CbError::LengthMismatch {
            column: "indices".to_owned(),
            expected: n,
            actual: indices.len(),
        });
    }

    // Value-range guards (T-07.5-06-01): a bin >= n_bins would write der_sums out of
    // bounds. An object id >= n would read der1/cindex out of bounds.
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write der_sums out of bounds"
        )));
    }
    if let Some(&bad) = indices.iter().find(|&&ix| (ix as usize) >= n) {
        return Err(CbError::OutOfRange(format!(
            "indices value {bad} >= n ({n}); object id would read der1/cindex out of bounds"
        )));
    }

    // The kernel needs n_bins as a comptime u32 (the per-feature line size).
    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel comptime line size)"))
    })?;

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    let der1_h = upload_channel_floats(client, der1);
    let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

    #[cfg(feature = "wgpu")]
    let der_sums_handle = {
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; der_sums_len]));
        pairwise_make_derivatives_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1_h, n) },
            unsafe { ArrayArg::from_raw_parts(cindex_h, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(indices_h, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), der_sums_len) },
            n_features as u32,
            n_bins_u32,
        );
        h
    };

    #[cfg(not(feature = "wgpu"))]
    let der_sums_handle = {
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; der_sums_len]));
        pairwise_make_derivatives_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der1_h, n) },
            unsafe { ArrayArg::from_raw_parts(cindex_h, cindex_stride) },
            unsafe { ArrayArg::from_raw_parts(indices_h, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), der_sums_len) },
            n_features as u32,
            n_bins_u32,
        );
        h
    };

    read_pair_der_sums_f64(client, der_sums_handle)
}

/// The pairwise split scorer result: the per-candidate pairwise scores (one per border,
/// `scores[feature * (bucket_count - 1) + border]`), the device-resident-assembled
/// per-(feature, bucket) der-sums (the bounded descriptor read back, leaf 0 / root), and
/// the deterministic best split (lowest-index tie-break == `select_best_candidate`).
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseSplitScore {
    /// The per-candidate pairwise score, flat `feature * (bucket_count - 1) + border`.
    pub scores: Vec<f64>,
    /// The device der-sum descriptor (leaf 0), flat `feature * n_bins + bin` — the
    /// bounded read-back the host solve consumed (kept for the self-oracle).
    pub der_sums: Vec<f64>,
    /// The deterministic best split (`feature_id`, `bin_id`) + its score, or `None` on a
    /// degenerate (no-candidate) system.
    pub best: Option<BestSplit>,
}

/// Compute the device pairwise split score over the FROZEN 7.4 4-channel pairwise
/// histogram handle (GPU-01 final slice). The heavy work is device-resident: the 4-channel
/// pair-weight histogram fill (7.4) + the der-sum scatter
/// ([`launch_pairwise_make_derivatives_into`]) build the per-(feature, bucket) statistics
/// on device; this seam then reads back ONLY the bounded `n_features * n_bins` der-sum
/// descriptor (the bulk pairwise histogram stays device-resident, D-05) and runs the
/// small per-leaf Cholesky solve + `CalculateScore` host-side via the FROZEN
/// `cb_compute::calculate_pairwise_score` (RESEARCH Open Q3: a `#[cube]` dense SPD solve
/// is awkward, and the FROZEN CPU `pairwise_cholesky_solve` IS the parity oracle, so the
/// small assembled system is solved over the bounded host read-back). The best split is
/// then selected with the SAME lowest-(feature,bin)-index tie-break as the pointwise path.
///
/// # Depth-1 / leaf_count == 1 MVP scope
///
/// This plan grows a depth-1 pairwise stump (leaf_count == 1 at level 0, the root), the
/// genuinely-complete vertical slice consistent with 07.5-03/04/05. The pairwise scorer's
/// per-leaf system is `2*leaf_count == 2` at the root; the general `leaf_count > 1` (deep)
/// path needs the partition-aware der-sum + pair-weight assembly (the SAME forward
/// dependency the pointwise grow loop surfaces for depth > 1).
///
/// # Inputs
///
/// `der1` (the pairwise-weighted first derivative, length `n`), `pair_i`/`pair_j` (object
/// ids, length `n_pairs`), `pair_weight` (length `n_pairs`), `cindex` (feature-major
/// quantized bins, length `n_features * n`), `indices` (object visiting order, length
/// `n`), `n_bins` (`1 << bits`, `bits` in {5,6,7}), `n_features`, `l2_diag_reg` (raw
/// `l2_leaf_reg`, NOT `scaled_l2`), `pairwise_bucket_weight_prior_reg`
/// (`bayesian_matrix_reg`, default `0.1`), `one_hot`. Empty input returns an empty score.
///
/// # Errors
///
/// [`CbError::OutOfRange`] on a degenerate dimension / overflow, [`CbError::Degenerate`]
/// on a read-back failure (never a silent zero buffer, WR-05), and the FROZEN 7.4 guards
/// (via the fill / der-sum launches). No `unwrap`/`expect`/`panic`/indexing.
#[allow(clippy::too_many_arguments)]
pub fn launch_pairwise_split_score(
    der1: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    one_hot: bool,
) -> CbResult<PairwiseSplitScore> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    launch_pairwise_split_score_into(
        &client,
        der1,
        pair_i,
        pair_j,
        pair_weight,
        cindex,
        indices,
        n_bins,
        n_features,
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg,
        one_hot,
    )
}

/// The ONE pairwise split-score geometry (IN-02 — one place). Threads ONE `&client`
/// through the 4-channel fill + the der-sum scatter (both device-resident), reads back
/// ONLY the bounded der-sum + 4-channel statistics descriptor, assembles the per-leaf
/// systems, and runs the FROZEN host `calculate_pairwise_score` + best-split argmin.
#[allow(clippy::too_many_arguments)]
fn launch_pairwise_split_score_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    one_hot: bool,
) -> CbResult<PairwiseSplitScore> {
    let n = der1.len();

    // Empty short-circuit FIRST (Pitfall 3/5).
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(PairwiseSplitScore {
            scores: Vec::new(),
            der_sums: Vec::new(),
            best: None,
        });
    }

    // The bit-width drives the FROZEN 7.4 fill (5/6/7-bit one-byte non-binary family).
    // n_bins MUST equal 1 << bits; derive bits and reject anything else with a typed error.
    let bits: u32 = match n_bins {
        32 => 5,
        64 => 6,
        128 => 7,
        _ => {
            return Err(CbError::Degenerate(format!(
                "launch_pairwise_split_score expects n_bins in {{32,64,128}} (1 << bits for \
                 bits in 5..=7, the FROZEN 7.4 one-byte non-binary family); got n_bins = {n_bins}"
            )));
        }
    };
    let n_objects = n;

    // (0) DEVICE: consume the FROZEN 7.4 4-channel pairwise histogram device-resident
    //     (the load-bearing 7.4 hand-off seam, D-7.4-03 / D-7.4-06). The 4-channel
    //     pair-weight statistics histogram is filled IN PLACE over the same pairs the
    //     scorer's pair-weight statistics derive from; the bulk histogram stays resident
    //     (NO read-back here, D-05). This launch also runs the FROZEN 7.4 pair/bin
    //     value-range guards BEFORE any device store. The handle is dropped after launch
    //     (the scan path reads it back as the self-oracle; the score path proves the SAME
    //     statistics via the FROZEN cb_compute oracle below, the parity reference D-7.4-05).
    let _pair_hist = launch_pairwise_hist_into(
        client, pair_i, pair_j, pair_weight, cindex, n_objects, n_bins, n_features, bits, one_hot,
    )?;

    // (1) DEVICE: scatter the pairwise-weighted der1 into the per-(feature, bucket)
    //     der-sum buffer device-resident, read back ONLY the bounded n_features * n_bins
    //     der-sum descriptor (== compute_der_sums over leaf_count == 1). This also runs
    //     the length / value-range guards on der1/cindex/indices BEFORE any launch.
    let der_sums_flat =
        launch_pairwise_make_derivatives_into(client, der1, cindex, indices, n_bins, n_features)?;

    // (2) HOST: assemble the per-leaf systems + score over the BOUNDED descriptor
    //     (RESEARCH Open Q3 — the small per-leaf dense Cholesky solve runs host-side over
    //     the assembled systems via the FROZEN cb_compute::calculate_pairwise_score; the
    //     bulk per-object scatter stayed device-resident). For the depth-1 MVP there is
    //     ONE leaf (the root), so the pair-weight statistics + der_sums are leaf_count == 1.
    //
    //     The pair-weight statistics are reconstructed from the SAME pair/bucket inputs the
    //     FROZEN 7.4 4-channel fill consumes via the FROZEN cb_compute::compute_pair_weight_statistics
    //     (the parity reference D-7.4-05 names): leaf_count == 1, bucket_count == n_bins, all
    //     objects in the root leaf 0. This keeps the score a TRANSCRIPTION of the FROZEN CPU
    //     oracle, never a re-derivation of the 4-channel histId->statistics transform.
    let leaf_count = 1usize;
    let bucket_count = n_bins;

    // bucket_of[obj] from the feature's cindex column (the candidate feature's bucket).
    // leaf_of is all-zero (the root). Build per-feature der_sums / pair-weight stats and
    // score each feature's borders, flattening to scores[feature * (bucket_count-1) + border].
    let n_splits = bucket_count.saturating_sub(1);
    let n_candidates = n_features.checked_mul(n_splits).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * (bucket_count-1) ({n_splits}) overflows usize"
        ))
    })?;
    let mut scores = vec![0.0_f64; n_candidates];
    let leaf_of = vec![0usize; n];

    // The global pairs as (winner, loser, weight) — the FROZEN compute_pair_weight_statistics
    // input shape (the SAME pairs the 7.4 fill consumes).
    let pairs: Vec<(usize, usize, f64)> = pair_i
        .iter()
        .zip(pair_j.iter())
        .zip(pair_weight.iter())
        .map(|((&i, &j), &w)| (i as usize, j as usize, w))
        .collect();

    for feature in 0..n_features {
        // bucket_of[obj] = the candidate feature's quantized bin (feature-major cindex).
        let mut bucket_of = vec![0usize; n];
        for obj in 0..n {
            let bin = cindex.get(feature * n + obj).copied().unwrap_or(0) as usize;
            if let Some(slot) = bucket_of.get_mut(obj) {
                *slot = bin;
            }
        }

        // der_sums[leaf=0][bucket] = the device-scattered der-sum row for this feature
        // (the bounded device descriptor, leaf 0). Shaped [leaf_count][bucket_count].
        let mut feat_der_sums = vec![vec![0.0_f64; bucket_count]; leaf_count];
        for bucket in 0..bucket_count {
            let v = der_sums_flat.get(feature * n_bins + bucket).copied().unwrap_or(0.0);
            if let Some(cell) = feat_der_sums.get_mut(0).and_then(|row| row.get_mut(bucket)) {
                *cell = v;
            }
        }

        // pair_weight_statistics[leaf][leaf][bucket] via the FROZEN cb_compute oracle over
        // the SAME pairs / bucket assignment (leaf_count == 1, the root).
        let pair_weight_statistics = cb_compute::compute_pair_weight_statistics(
            &pairs,
            leaf_count,
            bucket_count,
            &leaf_of,
            &bucket_of,
        )?;

        // The per-feature, per-border pairwise scores via the FROZEN host scorer (the small
        // per-leaf Cholesky solve runs here over the bounded assembled system).
        let feat_scores = cb_compute::calculate_pairwise_score(
            &feat_der_sums,
            &pair_weight_statistics,
            bucket_count,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
        )?;

        for border in 0..n_splits {
            let s = feat_scores.get(border).copied().unwrap_or(f64::NEG_INFINITY);
            if let Some(slot) = scores.get_mut(feature * n_splits + border) {
                *slot = s;
            }
        }
    }

    // (3) DEVICE: the deterministic best-split argmax (== select_best_candidate strict
    //     first-wins / lowest-index tie-break) over the host-solved scores, threaded
    //     through the device select_best_split_kernel so the selection is device-resident
    //     (only the O(1) winner crosses back). Empty candidate set -> no best.
    let best = if n_candidates == 0 {
        None
    } else {
        select_best_split_over_scores(client, &scores, n_splits)?
    };

    Ok(PairwiseSplitScore {
        scores,
        der_sums: der_sums_flat,
        best,
    })
}

/// Run the device [`select_best_split_kernel`] over a host-solved per-candidate score
/// vector and finish the O(blocks) across-block argmax host-side with the SAME
/// lowest-index tie-break (== `select_best_candidate`). Returns the winning
/// [`BestSplit`] (`feature_id = candidate / n_splits`, `bin_id = candidate % n_splits`),
/// or `None` if no candidate beat the sentinel. A read-back failure surfaces
/// [`CbError::Degenerate`] (WR-05), never a silent zero buffer.
fn select_best_split_over_scores(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    scores: &[f64],
    n_splits: usize,
) -> CbResult<Option<BestSplit>> {
    let n_candidates = scores.len();
    if n_candidates == 0 {
        return Ok(None);
    }
    let n_candidates_u32 = u32::try_from(n_candidates).map_err(|_| {
        CbError::OutOfRange(format!("n_candidates ({n_candidates}) exceeds u32 (argmin axis)"))
    })?;

    // A SINGLE cube of CUBE_DIM units strides over all candidates and block-reduces to one
    // winner (the FROZEN pointwise argmin geometry).
    let num_cubes = 1usize;
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    #[cfg(feature = "wgpu")]
    let (best_gain_handle, best_idx_handle) = {
        let scores_f32: Vec<f32> = scores.iter().map(|&v| v as f32).collect();
        let scores_h = client.create(cubecl::bytes::Bytes::from_elems(scores_f32));
        let best_gain_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; num_cubes]));
        let best_idx_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_cubes]));
        select_best_split_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(scores_h, n_candidates) },
            unsafe { ArrayArg::from_raw_parts(best_gain_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(best_idx_h.clone(), num_cubes) },
            n_candidates_u32,
        );
        (best_gain_h, best_idx_h)
    };

    #[cfg(not(feature = "wgpu"))]
    let (best_gain_handle, best_idx_handle) = {
        let scores_h = client.create(cubecl::bytes::Bytes::from_elems(scores.to_vec()));
        let best_gain_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; num_cubes]));
        let best_idx_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_cubes]));
        select_best_split_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(scores_h, n_candidates) },
            unsafe { ArrayArg::from_raw_parts(best_gain_h.clone(), num_cubes) },
            unsafe { ArrayArg::from_raw_parts(best_idx_h.clone(), num_cubes) },
            n_candidates_u32,
        );
        (best_gain_h, best_idx_h)
    };

    let best_gains = read_scores_f64(client, best_gain_handle)?;
    let best_idx_bytes = client
        .read_one(best_idx_handle)
        .map_err(|e| CbError::Degenerate(format!("pairwise best-idx read-back failed: {e:?}")))?;
    let best_idxs: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&best_idx_bytes).to_vec();

    // Finish the across-block argmax: highest score wins; on an EXACT tie the LOWER
    // candidate index wins (strict first-wins parity == select_best_candidate).
    let mut best_score = f64::NEG_INFINITY;
    let mut best_c = u32::MAX;
    for (block, &gain) in best_gains.iter().enumerate() {
        let cand = best_idxs.get(block).copied().unwrap_or(u32::MAX);
        if (cand as usize) >= n_candidates {
            continue;
        }
        let take = gain > best_score || (gain == best_score && cand < best_c);
        if take {
            best_score = gain;
            best_c = cand;
        }
    }

    if (best_c as usize) < n_candidates && n_splits > 0 {
        let feature = (best_c as usize) / n_splits;
        let bin = (best_c as usize) % n_splits;
        Ok(Some(BestSplit {
            feature_id: feature as u32,
            bin_id: bin as u32,
            score: best_score as f32,
            gain: best_score as f32,
        }))
    } else {
        Ok(None)
    }
}

/// Grow ONE oblivious tree device-resident over the compile-time [`SelectedRuntime`]
/// using the PAIRWISE split scorer path (GPU-01 ranking slice). The pairwise sibling of
/// [`grow_oblivious_tree`]: per level it chains the FROZEN 7.4 4-channel pairwise fill +
/// the device der-sum scatter -> [`launch_pairwise_split_score`] (the per-leaf
/// linear-system build + the host Cholesky solve, RESEARCH Open Q3) -> ONE O(1)
/// [`BestSplit`] read-back -> the host integer split decision -> the loss-agnostic Plan-C
/// [`launch_partition_split_into`] (forward-bit doc-routing) -> [`launch_partition_update_into`]
/// (per-partition Σ der1 / Σ weight reduce), over persistent device handles threaded
/// through ONE `ComputeClient`. At the leaves it reads back ONLY the `2^depth` part-stats
/// and computes leaf values via the FROZEN `cb_compute::calc_average`. The bulk pairwise
/// histogram / partition / doc-routing NEVER crosses to host per level (D-05).
///
/// # MVP scope (depth == 1) — the partition-aware forward dependency
///
/// The MVP grows a depth-1 pairwise stump (leaf_count == 1 at level 0, the root) — the
/// genuinely-complete vertical slice consistent with 07.5-03/04/05. The pairwise scorer's
/// per-leaf system is `2*leaf_count == 2` at the root. A `depth > 1` tree scores each
/// level over the CURRENT 2^level partitions (the partition-aware der-sum + pair-weight
/// assembly), the SAME forward dependency the pointwise [`grow_oblivious_tree`] surfaces;
/// `depth > 1` returns a typed [`CbError::OutOfRange`] until it lands (documented, NOT
/// silently cut). PairLogit/ranking, foldCount == 1, Plain.
///
/// # Inputs
///
/// `der1` (the pairwise-weighted first derivative, length `n`), `weight` (channel 1,
/// length `n`), `pair_i`/`pair_j`/`pair_weight` (the global competitor pairs), `cindex`
/// (feature-major quantized bins, length `n_features * n`), `indices` (object visiting
/// order, length `n`), `n_bins` (`1 << bits`, bits in {5,6,7}), `n_features`, `depth`,
/// `l2_diag_reg` (raw `l2_leaf_reg`), `pairwise_bucket_weight_prior_reg`
/// (`bayesian_matrix_reg`, default `0.1`), `scaled_l2` (the per-tree leaf-value scaling),
/// `one_hot`.
///
/// # Errors
///
/// [`CbError::OutOfRange`] if `depth > 1` or a dimension overflows; [`CbError::Degenerate`]
/// if a level finds no candidate split or a read-back fails (never a silent zero buffer);
/// the FROZEN 7.4 guards (via the fill / score / partition launches). No
/// `unwrap`/`expect`/`panic`/indexing.
#[allow(clippy::too_many_arguments)]
pub fn grow_oblivious_tree_pairwise(
    der1: &[f64],
    weight: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    depth: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    scaled_l2: f64,
    one_hot: bool,
) -> CbResult<GrownTree> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    grow_oblivious_tree_pairwise_into(
        &client,
        der1,
        weight,
        pair_i,
        pair_j,
        pair_weight,
        cindex,
        indices,
        n_bins,
        n_features,
        depth,
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg,
        scaled_l2,
        one_hot,
    )
}

/// The ONE pairwise grow-loop geometry (IN-02 — one place). Uploads the resident
/// der1/weight/cindex/indices/leaf_of handles ONCE onto `client`, runs the per-depth
/// pairwise launch chain over those persistent handles, and reads back ONLY the O(1)
/// BestSplit per level + the final `2^depth` part-stats. The caller owns the `client`
/// lifecycle so every read-back uses the SAME client that allocated the handles.
#[allow(clippy::too_many_arguments)]
fn grow_oblivious_tree_pairwise_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    weight: &[f64],
    pair_i: &[u32],
    pair_j: &[u32],
    pair_weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    n_bins: usize,
    n_features: usize,
    depth: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    scaled_l2: f64,
    one_hot: bool,
) -> CbResult<GrownTree> {
    let n = der1.len();

    // Empty short-circuit (Pitfall 3/5).
    if n == 0 || n_features == 0 || n_bins == 0 || depth == 0 {
        return Ok(GrownTree {
            splits: Vec::new(),
            leaf_of: vec![0u32; n],
            leaf_values: Vec::new(),
            part_stats: Vec::new(),
        });
    }

    // MVP scope guard: the depth-1 pairwise stump scores over the single root leaf
    // (leaf_count == 1). A depth>1 level scores over 2^level partitions (the
    // partition-aware der-sum + pair-weight assembly) — the EXPLICIT tracked forward
    // dependency (the SAME class as the pointwise grow loop's depth>1 guard). Reject with
    // a typed error rather than fabricating a wrong-structure deep tree.
    if depth > 1 {
        return Err(CbError::OutOfRange(format!(
            "grow_oblivious_tree_pairwise supports depth <= 1 until the partition-aware \
             (per-leaf, leaf_count > 1) pairwise der-sum + pair-weight assembly lands (the \
             depth-1 stump scores the single root leaf; a depth>1 level scores over 2^level \
             partitions — the EXPLICIT tracked forward dependency); got depth = {depth}"
        )));
    }

    // 2^depth leaf count, overflow-checked.
    let n_leaves = 1usize
        .checked_shl(depth as u32)
        .ok_or_else(|| CbError::OutOfRange(format!("2^depth overflows usize (depth = {depth})")))?;

    let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!("n_features ({n_features}) * n ({n}) overflows usize"))
    })?;
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Resident device handles uploaded ONCE (the D-05 persistent-buffer contract).
    let der1_h = upload_channel_floats(client, der1);
    let weight_h = upload_channel_floats(client, weight);
    let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));
    let mut leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));

    let mut splits: Vec<(u32, u32)> = Vec::with_capacity(depth);

    for level in 0..depth {
        // (1) Device pairwise fill + der-sum scatter + score + deterministic argmax over
        //     the CURRENT (root) partition. The bulk 4-channel histogram + der-sum scatter
        //     stay device-resident IN the score launch; only the O(1) BestSplit crosses back.
        let scored = launch_pairwise_split_score_into(
            client,
            der1,
            pair_i,
            pair_j,
            pair_weight,
            cindex,
            indices,
            n_bins,
            n_features,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
            one_hot,
        )?;

        // (2) The O(1) host integer split decision. No candidate -> a degenerate dataset.
        let split = scored.best.ok_or_else(|| {
            CbError::Degenerate(format!(
                "grow_oblivious_tree_pairwise level {level}: no candidate split (degenerate)"
            ))
        })?;
        splits.push((split.feature_id, split.bin_id));

        // (3) Device partition-split (forward-bit doc-routing, level -> bit level == the CPU
        //     `leaf_index` convention) — IN-PLACE on device, NO read-back here (D-05).
        leaf_of_h = launch_partition_split_into(
            client,
            der1_h.clone(),
            cindex_h.clone(),
            indices_h.clone(),
            leaf_of_h,
            n,
            cindex_stride,
            split.feature_id,
            split.bin_id,
            level as u32,
        )?;
    }

    // (4) Device partition-update: the per-partition Σ der1 / Σ weight reduce over the
    //     final 2^depth partitions, device-resident.
    let part_stats_h = launch_partition_update_into(
        client,
        der1_h.clone(),
        weight_h.clone(),
        indices_h.clone(),
        leaf_of_h.clone(),
        n,
        n_leaves,
    )?;

    // (5) ONE read-back of the 2^depth part-stats (the ONLY bulk-data crossing besides the
    //     O(1) per-level BestSplit — D-05). A read-back failure surfaces CbError::Degenerate.
    let part_stats = read_part_stats_f64(client, part_stats_h)?;

    // (6) Host leaf values via the FROZEN cb_compute::calc_average formula (count>0 guard).
    let mut leaf_values = vec![0.0_f64; n_leaves];
    for leaf in 0..n_leaves {
        let sum = part_stats.get(leaf * 2).copied().unwrap_or(0.0);
        let cnt = part_stats.get(leaf * 2 + 1).copied().unwrap_or(0.0);
        if let Some(slot) = leaf_values.get_mut(leaf) {
            *slot = cb_compute::calc_average(sum, cnt, scaled_l2);
        }
    }

    // (7) The per-object leaf assignment (the SC-3 structure observation): the grow loop
    //     never reads the bulk routing back (D-05); this single read-back at the END is
    //     the oracle seam (the same crossing class as the final part-stats).
    let leaf_of = read_u32_handle(client, leaf_of_h)?;

    Ok(GrownTree {
        splits,
        leaf_of,
        leaf_values,
        part_stats,
    })
}
