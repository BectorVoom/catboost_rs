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
//! This module hosts several phases' device-launch seams. IN-03 (the tracked
//! single-responsibility split) has LANDED: the Phase 7.2 der seam now lives in
//! [`der_seams`] and the Phase 7.4/7.5 pairwise seam in [`pairwise`], both re-exported
//! here (`pub use der_seams::*` / `pub use pairwise::*`) so the crate-internal
//! `crate::gpu_runtime::X` API surface is unchanged. This `mod.rs` retains:
//!
//! - **Phase 7.1 reduce/scan primitives** (this file's original scope):
//!   [`launch_block_reduce_f64`], [`launch_block_reduce_atomic_f64`],
//!   [`launch_block_scan_f64`], [`AtomicFinalizePath`].
//! - **Phase 7.3 pointwise-histogram seam**: `launch_pointwise_hist2*` and
//!   `read_binsums_f64`.
//! - **Phase 7.5 pointwise split/score + scan/update seam**, the partition primitives,
//!   and the pointwise grow-loop + boosting drivers.
//!
//! The relocation is a PURE mechanical move (ZERO logic changes); see [`der_seams`] and
//! [`pairwise`]. The `kernels::{...}` import below pulls in kernels for the seams retained
//! here; the relocated submodules import their own kernels via `use super::*`.
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
    apply_leaf_delta_kernel,
    block_reduce_atomic_kernel, block_reduce_kernel, block_scan_kernel, find_optimal_split_kernel,
    find_optimal_split_partition_kernel,
    focal_gradient_kernel,
    focal_hessian_kernel, gradient_kernel, logloss_gradient_kernel, logloss_hessian_kernel,
    pairwise_hist_8bit_atomics_kernel, pairwise_hist_binary_kernel,
    pairwise_hist_half_byte_kernel, pairwise_hist_nonbinary_kernel,
    derive_sibling_partition_hist_kernel, fold_hist_copies_kernel,
    partition_hist2_lds_kernel, partition_hist2_nonbinary_kernel, partition_split_kernel,
    partition_update_kernel,
    pointwise_hist2_binary_kernel, zero_u64_kernel,
    HIST_LDS_CELLS_LARGE, HIST_LDS_CELLS_MEDIUM, HIST_LDS_CELLS_SMALL,
    pointwise_hist2_half_byte_kernel, pointwise_hist2_nonbinary_kernel,
    subtract_histograms_kernel,
    pairwise_assemble_system_kernel,
    pairwise_make_derivatives_kernel, quantile_gradient_kernel, scan_update_pairwise_kernel,
    scan_update_pointwise_kernel, select_best_split_kernel, SCORE_FN_COSINE, SCORE_FN_L2,
    SCORE_FN_LOO_L2, SCORE_FN_SAT_L2, SCORE_FN_SOLAR_L2,
};
// The fixed-point scale is only used by the TEST-SEAM decoder `read_fixedpoint_hist_f64`
// (`#[cfg(test)]` — production NEVER reads the histogram back, D-05).
#[cfg(test)]
use crate::kernels::REDUCE_FIXEDPOINT_SCALE_F64;
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

/// Launch geometry for the PARTITION-HISTOGRAM family (fill / zero / fold / derive):
/// 256 threads per cube. These kernels are pure grid-stride loops with NO
/// `SharedMemory` allocation, so they are decoupled from the [`CUBE_DIM`]-sized
/// shared-mem reduce family above. 32-thread cubes (one warp per block on CUDA)
/// cap occupancy well below the SM limit on NVIDIA/AMD; 256 is the conventional
/// occupancy-friendly width for atomic-scatter kernels (upstream CatBoost's
/// histogram kernels use 256..768-thread blocks).
const HIST_CUBE_DIM: usize = 256;

/// Maximum privatized-copy count for the multi-copy partition-histogram fill (the
/// contention fix): at `2^level` partitions the launcher allocates
/// `max(1, HIST_MAX_COPIES >> level)` copies, so the total allocation stays
/// ~`HIST_MAX_COPIES × one-partition line` at every level while same-cell atomic
/// contention drops by the copy count where it is worst (the shallow levels, where
/// few hot cells absorb every object's 2-atomic scatter).
const HIST_MAX_COPIES: usize = 64;

/// Target TOTAL cube count for the 2-D LDS-privatized fill dispatch (object chunks ×
/// feature tiles). ~4-8 cubes per SM/CU on the mid-size parts this targets (P100 = 56
/// SMs, gfx11 APUs ~20 CUs) — enough resident cubes to hide LDS-atomic latency without
/// exploding the per-cube merge traffic (`cubes × tile_cells` global atomics). A soft
/// target: the object axis is capped by [`HIST_LDS_MIN_OBJ_PER_THREAD`] and the feature
/// axis by the LDS budget, so tiny inputs launch fewer cubes.
const HIST_LDS_TARGET_CUBES: usize = 256;

/// Minimum objects per thread on the LDS fill's object axis: more chunks than
/// `n / (HIST_CUBE_DIM * this)` would spend more time on per-cube zero+merge overhead
/// than on scatter work (each extra chunk re-merges its whole tile).
const HIST_LDS_MIN_OBJ_PER_THREAD: usize = 8;

/// `CB_GPU_PROF=1` gates the per-stage device profiling prints (stage attribution for
/// the resident grow loop). The env var is read ONCE; unset (or `"0"`) keeps every
/// profiling branch cold — no sync, no timing, no output — so the hot path is unchanged.
pub(crate) fn gpu_prof_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("CB_GPU_PROF").is_some_and(|v| v != "0"))
}

/// Drain the device queue (a profiling FENCE — only ever called under
/// [`gpu_prof_enabled`], never on the un-profiled hot path). Errors are swallowed:
/// a failed fence skews a timing report, it must not fail training.
pub(crate) fn prof_sync(client: &cubecl::client::ComputeClient<SelectedRuntime>) {
    let _ = cubecl::reader::try_read_sync(client.sync());
}

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

/// Does the device advertise `Atomic<u64>` add? Gates the fixed-point partition-histogram
/// fill (`partition_hist2_nonbinary_kernel`), which accumulates into `&Array<Atomic<u64>>`
/// and therefore CANNOT run on a backend without u64 atomic-add (cpu/wgpu). Mirrors
/// [`device_supports_f64_atomic_add`] and the `kernels::reduce` u64 capability query. WR-02:
/// the partition-fill launcher gates on this and surfaces a typed error before launch rather
/// than attempting a kernel the backend cannot execute.
fn device_supports_u64_atomic_add<R: cubecl::Runtime>(
    client: &cubecl::client::ComputeClient<R>,
) -> bool {
    let ty = <Atomic<u64> as CubePrimitive>::as_type_native_unchecked();
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


// ===========================================================================
// IN-03 — module split. The oversized single-file `gpu_runtime` is split into this
// `mod.rs` (the 7.1 reduce/scan primitives, the 7.3 pointwise-histogram seam, the
// 7.5 pointwise split/score + scan/update seam, the partition primitives, and the
// pointwise grow-loop + boosting drivers) plus two relocated leaf submodules. The
// relocation is a PURE mechanical move (ZERO logic changes); every public item is
// re-exported here so the crate-internal `crate::gpu_runtime::X` API surface is
// IDENTICAL and all existing `use` paths still resolve.
// ===========================================================================

mod der_seams; // Phase 7.2 der1/der2 seam (DerBinaryKernel/.../launch_der_*).
pub use der_seams::*;

mod pairwise; // Phase 7.4/7.5 pairwise histogram + scan/score + pairwise grow driver.
pub use pairwise::*;

mod session; // Phase 10-07 (GPUT-02/03): the per-fit device-resident training session.
pub use session::*;

// Phase 13 Plan 07 (GPUT-12): the multi-output device driver — the block-leaf emission that wires
// the Plan-06 K-dim Newton block solve onto the multi-output loss family (MultiClass softmax,
// MultiClassOneVsAll, MultiLogloss / MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty). `pub(crate)`
// so the session multiclass coverage gate reaches `map_multiclass_objective` / `MulticlassObjective`
// and the `multiclass_test` self-oracle reaches the block driver.
pub(crate) mod multiclass;

// Phase 13 Plan 08 (GPUT-13): the ordered-boosting device driver — the per-permutation historical
// approx trajectory (`ordered_approx_delta_simple` body/tail approximant) kept device-resident across
// iterations, folded into the resident trajectory via `apply_leaf_delta` (identity leaf map + unit
// rate). `pub(crate)` so the session ordered coverage gate + the `ordered_test` self-oracle reach the
// `OrderedTree` descriptor + `ordered_approx_delta` / `accumulate_ordered_trajectory` driver.
pub(crate) mod ordered;

// Phase 13 Plan 08 (GPUT-13): the ordered trajectory self-oracle (source/test separation) — device
// resident trajectory (folded via `apply_leaf_delta`) vs the frozen CPU `ordered_approx_delta_simple`
// trajectory at ε=1e-4, the body-rows-keep-0 anti-leakage assertion, the residency (single final
// read-back) check, and the uncovered-config `Ok(None)` gate. rocm in-env on gfx1100 (numeric ε
// assertions device-gated; cpu records-only, WR-01).
#[cfg(test)]
mod ordered_test;

// Phase 13 Plan 07 (GPUT-12): the multi-output block-emission self-oracle (source/test separation) —
// device coupled-softmax K=3 + diagonal RMSEWithUncertainty K=2 + diagonal MultiClassOneVsAll block
// leaves vs the CPU `cb_compute::solve_symmetric_newton` multi-output leaf values at ε=1e-4, plus the
// uncovered-config `Ok(None)` gate. rocm in-env on gfx1100 (numeric ε assertions device-gated; cpu
// records-only, WR-01).
#[cfg(test)]
mod multiclass_test;

// Phase 13 Plan 04 (GPUT-22): the deterministic query/listwise objective device driver
// (QueryRMSE / QuerySoftMax / QueryCrossEntropy) over the Plan-03 query-grouping infra. `pub(crate)`
// so the session ranking coverage gate reaches `RankingObjective` / `ranking_objective_covered` and
// the `ranking_det_test` self-oracle reaches the der drivers.
pub(crate) mod ranking;

// Phase 13 Plan 04 (GPUT-22): the deterministic ranking der self-oracle (source/test separation) —
// device QueryRMSE / QuerySoftMax der vs the CPU `cb_compute::ranking_der` at ε=1e-4, plus the
// QueryCrossEntropy bounded-shift self-consistency + independent Ok(None) gate. rocm in-env on
// gfx1100 (numeric ε assertions device-gated; cpu records-only, WR-01).
#[cfg(test)]
mod ranking_det_test;

// Phase 13 Plan 05 (GPUT-22, D-08): the STOCHASTIC ranking pair self-oracle (source/test
// separation) — device YetiRank / PFound-F der vs the FROZEN pinned-seed CPU
// `yetirank_sample_pairs` + `calc_ders_for_queries` reference at ε=1e-4, plus the per-query seed
// chain + draw-count asserts (Pitfall 4 / T-13-10). rocm in-env on gfx1100 (numeric ε assertions
// device-gated; cpu records-only, WR-01).
#[cfg(test)]
mod ranking_stoch_test;

// Phase 10-07 (GPUT-02/03): the GpuTrainSession residency cross-oracle (source/test
// separation) — begin uploads once, grow_one reuses the resident handles + chains der1 on
// device, structure matches the CPU multi-tree boosting reference; the coverage gate
// declines depth>1 / non-RMSE-Logloss / non-Plain / fold_count>1. rocm in-env on gfx1100.
#[cfg(test)]
mod session_residency;

// Phase 12 Plan 01 (GPUT-18, A3 gap): the depth>1 device-grow self-oracle. The coverage gate
// no longer force-declines depth>1 — a depth-6 Plain/fold1/RMSE config grows through the
// Phase-11 partition-aware substrate and matches a direct `grow_oblivious_tree` call within
// ε=1e-4; every still-uncovered config returns Ok(None). rocm in-env on gfx1100.
#[cfg(test)]
mod session_depth_gt1_test;

// Phase 10-06 (GPUT-15): the bit-packed compressed index (cindex) builder — the
// grouped `WriteCompressedIndex` layout + per-feature `TCFeature` table the histogram /
// partition consumers read through the ONE `kernels::read_bin` accessor. `pub(crate)` so
// the `kernels::cindex` bit-exact oracle can reach `pack_cindex`/`TCFeature`.
pub(crate) mod cindex;

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

    // GPUT-15/GPUT-03: build the bit-packed grouped cindex + per-feature TCFeature table
    // (Offset/Shift/Mask) and upload the channel-typed der1/weight + the packed cindex
    // `words` + per-feature (offsets, shifts, masks) + indices, then delegate to the ONE
    // resident histogram geometry [`hist2_launch_resident`]. This SLICE entry uploads fresh
    // handles per call; the GPUT-03 resident session path uploads these ONCE at `begin` and
    // clones the persistent handles into the SAME geometry (no per-tree re-upload). Every
    // fill kernel reads bins through the ONE `read_bin` accessor over the packed words
    // (`(cindex[offset + obj] >> shift) & mask`), NEVER the old plain
    // `cindex[feature * n + obj]` load (T-10-15). The bin VALUE is unchanged (the
    // bin->border join is identical); only its storage/extraction changes. Host-pack-then-
    // upload-once (Open Q1 / A2 — see `gpu_runtime::cindex`). All features in this fill
    // share the same `n_bins` bucket count; the cindex value-range guard above already
    // rejected any bin >= n_bins, so `pack_cindex` masks each field without truncation.
    let n_buckets_per_feature = vec![n_bins; n_features];
    let packed = crate::gpu_runtime::cindex::pack_cindex(cindex, &n_buckets_per_feature, n)?;
    let (offsets_v, shifts_v, masks_v) = packed.device_arrays()?;
    let num_words = packed.words.len();

    // Upload der1/weight as the channel float type (f32 on wgpu, f64 elsewhere — RESEARCH
    // A1) via the shared helper, and the packed cindex `words` + TCFeature arrays + indices
    // as u32. These are the SAME handles the resident session holds persistently (GPUT-03).
    let der1_h = upload_channel_floats(client, der1);
    let weight_h = upload_channel_floats(client, weight);
    let cindex_words_h = client.create(cubecl::bytes::Bytes::from_elems(packed.words.clone()));
    let offsets_h = client.create(cubecl::bytes::Bytes::from_elems(offsets_v));
    let shifts_h = client.create(cubecl::bytes::Bytes::from_elems(shifts_v));
    let masks_h = client.create(cubecl::bytes::Bytes::from_elems(masks_v));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

    hist2_launch_resident(
        client, der1_h, weight_h, cindex_words_h, offsets_h, shifts_h, masks_h, indices_h,
        num_words, n, n_bins, n_features,
    )
}

/// The ONE resident pointwise-histogram launch geometry (GPUT-03 / IN-02 — one place). Takes
/// PRE-UPLOADED, channel-typed device handles (der1/weight already f32-on-wgpu / f64-else,
/// the packed cindex `words`, the per-feature TCFeature offsets/shifts/masks, and indices),
/// picks the bit-width family HOST-SIDE from `n_bins`, zero-initialises the `binSums`
/// buffer, launches the fill kernel, and returns the `binSums` Handle WITHOUT reading it
/// back (SC-3 / Pitfall 5). Both the slice entry [`launch_pointwise_hist2_into`] (fresh
/// per-call handles) and the resident session grow loop (persistent handles, cloned per
/// call — no per-tree re-upload) route through here, so the launch geometry stays single.
///
/// The input handles are `.clone()`d into each launch arm (a CubeCL Handle is a ref-counted
/// buffer binding; cloning shares the device buffer, it does NOT copy) so the SAME resident
/// buffers feed every boosting iteration. The caller is responsible for the value-range
/// guards (`indices[i] < n`, `cindex bin < n_bins`) — the slice entry runs them before
/// upload; the resident session runs them ONCE at `begin`. No `unwrap`/`expect`/`panic`/
/// indexing (workspace lints + D-13).
#[allow(clippy::too_many_arguments)]
fn hist2_launch_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1_h: Handle,
    weight_h: Handle,
    cindex_words_h: Handle,
    offsets_h: Handle,
    shifts_h: Handle,
    masks_h: Handle,
    indices_h: Handle,
    num_words: usize,
    n: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<Handle> {
    // Empty fill: hand back a zero-length handle (no launch, no read-back — Pitfall 5).
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // Launch geometry: enough cubes to cover `n` objects (the grid-stride loop in every fill
    // kernel handles any surplus via the total-thread-count stride). Shared by all families.
    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // IN-02: the per-family zero-init + channel-float `#[cfg]` split + launch + `return
    // Ok(h)` boilerplate is IDENTICAL across the binary / half-byte / non-binary arms (they
    // differ only in the kernel launcher and, for non-binary, a trailing `bits` comptime
    // arg). Each input handle is `.clone()`d into the launch (share, not copy) so the SAME
    // resident buffers feed every call; the fresh `binSums` output `h` is returned WITHOUT a
    // read-back. Exactly one channel `#[cfg]` arm is compiled per build; the taken family
    // `if` early-returns.
    macro_rules! launch_hist2_family {
        ($kernel:ident $(, $extra:expr )* $(,)?) => {{
            let bin_sums_len = hist2_binsums_len(n_bins, n_features);
            #[cfg(feature = "wgpu")]
            {
                let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; bin_sums_len]));
                $kernel::launch::<f32, SelectedRuntime>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(der1_h.clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(weight_h.clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(cindex_words_h.clone(), num_words) },
                    unsafe { ArrayArg::from_raw_parts(offsets_h.clone(), n_features) },
                    unsafe { ArrayArg::from_raw_parts(shifts_h.clone(), n_features) },
                    unsafe { ArrayArg::from_raw_parts(masks_h.clone(), n_features) },
                    unsafe { ArrayArg::from_raw_parts(indices_h.clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                    n_features as u32,
                    $( $extra, )*
                );
                return Ok(h);
            }

            #[cfg(not(feature = "wgpu"))]
            {
                let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; bin_sums_len]));
                $kernel::launch::<f64, SelectedRuntime>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(der1_h.clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(weight_h.clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(cindex_words_h.clone(), num_words) },
                    unsafe { ArrayArg::from_raw_parts(offsets_h.clone(), n_features) },
                    unsafe { ArrayArg::from_raw_parts(shifts_h.clone(), n_features) },
                    unsafe { ArrayArg::from_raw_parts(masks_h.clone(), n_features) },
                    unsafe { ArrayArg::from_raw_parts(indices_h.clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(h.clone(), bin_sums_len) },
                    n_features as u32,
                    $( $extra, )*
                );
                return Ok(h);
            }
        }};
    }

    // Binary (1-bit) family branch (Plan D — D-7.3-02): the SEPARATE
    // `pointwise_hist2_binary_kernel` (a structurally distinct 2-bucket split-bit
    // decomposition). Writes the SAME FROZEN binSums layout through this UNCHANGED seam.
    if n_bins == crate::kernels::BINARY_BINS {
        launch_hist2_family!(pointwise_hist2_binary_kernel);
    }

    // Half-byte (4-bit) family branch (Plan C — D-7.3-02): the SEPARATE
    // `pointwise_hist2_half_byte_kernel` (16-bin working histogram + nibble decomposition).
    if n_bins == crate::kernels::HALF_BYTE_BINS {
        launch_hist2_family!(pointwise_hist2_half_byte_kernel);
    }

    // One-byte non-binary bit-width selection (Plan B — D-7.3-02). `bits` is chosen
    // HOST-SIDE from `n_bins` (a `b`-bit group has `1 << b` bins) and passed as the SAME
    // `#[comptime]` arg of `pointwise_hist2_nonbinary_kernel` — no runtime bit-count branch.
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

    // Fill the FROZEN 7.3 device-resident 2-channel histogram (this also runs the FROZEN
    // length / value-range guards on der1/weight/cindex/indices BEFORE any launch, and
    // returns a device HANDLE with NO read-back). The bulk histogram stays device-resident,
    // then the per-candidate score + best split is computed over that handle via the ONE
    // score geometry [`score_over_binsums`] (shared with the resident session grow loop,
    // GPUT-03 — so the histogram fill and the score/argmin each live in exactly one place).
    let bin_sums = launch_pointwise_hist2_into(client, der1, weight, cindex, indices, n_bins, n_features)?;
    score_over_binsums(client, bin_sums, n_bins, n_features, scaled_l2, score_fn)
}

/// The ONE score/argmin geometry over a resident `binSums` handle (GPUT-03 / IN-02 — one
/// place). Launches [`find_optimal_split_kernel`] over the device-resident histogram handle
/// (consumed in place, NEVER read to host), reads back ONLY the per-candidate score vector
/// (the self-oracle observation) + the O(blocks) per-block winner descriptors, and finishes
/// the across-block argmin host-side with the strict lowest-`(feature, bin)` tie-break.
/// Both the slice entry [`launch_find_optimal_split_pointwise_into`] and the resident
/// session grow loop route through here.
///
/// `bin_sums` is the FROZEN 2-channel histogram handle (length [`hist2_binsums_len`]);
/// `scaled_l2` the per-tree `scale_l2_reg` output; `score_fn` the comptime score calcer arm
/// (already validated by the caller). Returns `(best_split, per_candidate_scores)`. A
/// read-back failure surfaces [`CbError::Degenerate`] (WR-05), never a silent zero buffer.
/// No `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
fn score_over_binsums(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    bin_sums: Handle,
    n_bins: usize,
    n_features: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<(Option<BestSplit>, Vec<f64>)> {
    // Candidate-count overflow guard (T-07.5-01-02): the per-candidate score buffer and the
    // kernel's candidate index math are products of caller-supplied dimensions. Reject a
    // degenerate dimension with a typed range error BEFORE forming the product unchecked.
    let n_candidates = n_features.checked_mul(n_bins).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) overflows usize (candidate count)"
        ))
    })?;
    // The score kernel needs n_bins as a comptime u32; bound it so the cast cannot silently
    // truncate a degenerate dimension.
    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel comptime line size)"))
    })?;

    // Launch geometry: a SINGLE cube of CUBE_DIM units strides over all candidates and
    // block-reduces to one winner. The shared-mem argmin size is the comptime
    // ARGMIN_SHMEM == CUBE_DIM (Pitfall 3).
    let num_cubes = 1usize;
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // The score / argmin output buffers. `scores` is the per-candidate L2 score (the
    // self-oracle observation); `best_gain`/`best_idx` carry one winner per cube. The
    // channel float type matches the histogram channel: f64 on rocm/cuda/cpu, f32 on wgpu
    // (RESEARCH A1) — read back and UPCAST to f64.
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
        // WR-05: the trailing `border == n_bins - 1` candidate is the no-op split (all bins
        // LEFT / none RIGHT); the device kernel already excludes it, this is the host belt.
        if (cand as usize) % n_bins == n_bins - 1 {
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

/// Create an `n`-element channel-typed device buffer filled with ONE constant, ON DEVICE
/// (a length-1 value upload + a [`crate::kernels::fill_kernel`] launch) — the transfer-lean
/// sibling of [`upload_channel_floats`] for constant vectors (e.g. the RMSE `der2 = -1`
/// channel), replacing a per-call O(n) host alloc + PCIe upload with O(1) bytes crossing.
pub(crate) fn create_channel_const(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    value: f64,
    n: usize,
) -> Handle {
    if n == 0 {
        return client.empty(0);
    }
    let num_cubes = n.div_ceil(HIST_CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: HIST_CUBE_DIM as u32,
        y: 1,
        z: 1,
    };
    #[cfg(feature = "wgpu")]
    {
        let out = client.empty(n * std::mem::size_of::<f32>());
        let val_h = client.create(cubecl::bytes::Bytes::from_elems(vec![value as f32]));
        crate::kernels::fill_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(val_h, 1) },
        );
        out
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let out = client.empty(n * std::mem::size_of::<f64>());
        let val_h = client.create(cubecl::bytes::Bytes::from_elems(vec![value]));
        crate::kernels::fill_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(val_h, 1) },
        );
        out
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

/// Read the per-tree part-stats (channel float) AND the final `leaf_of` routing in ONE
/// blocking read — the same two D-05 end-of-tree crossings as
/// [`read_part_stats_f64`] + [`read_u32_handle`], batched so the resident grow pays a
/// single pipeline sync per tree instead of two. wgpu upcasts the f32 channel exactly
/// like `read_part_stats_f64`. A read-back failure surfaces [`CbError::Degenerate`]
/// (WR-05), never a silent zero buffer.
pub(crate) fn read_part_stats_and_leaf_of(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    part_stats_h: Handle,
    leaf_of_h: Handle,
) -> CbResult<(Vec<f64>, Vec<u32>)> {
    let buffers = cubecl::reader::try_read_sync(client.read_async(vec![part_stats_h, leaf_of_h]))
        .ok_or_else(|| {
            CbError::Degenerate(
                "part-stats/leaf_of read-back: blocking reads are unsupported on this platform"
                    .to_owned(),
            )
        })?
        .map_err(|e| CbError::Degenerate(format!("part-stats/leaf_of read-back failed: {e:?}")))?;
    let mut buffers = buffers.into_iter();
    let (Some(stats_bytes), Some(leaf_bytes)) = (buffers.next(), buffers.next()) else {
        return Err(CbError::Degenerate(
            "part-stats/leaf_of read-back returned fewer than 2 buffers".to_owned(),
        ));
    };
    #[cfg(feature = "wgpu")]
    let part_stats = bytemuck::cast_slice::<u8, f32>(&stats_bytes)
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    #[cfg(not(feature = "wgpu"))]
    let part_stats = bytemuck::cast_slice::<u8, f64>(&stats_bytes).to_vec();
    Ok((part_stats, bytemuck::cast_slice::<u8, u32>(&leaf_bytes).to_vec()))
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

    // GPUT-15: the kernel reads the split bin through the ONE `read_bin` accessor
    // (T-10-15). The split feature's plain feature-major cindex is read_bin's DEGENERATE
    // `TCFeature`: `offset = feature * n` (word base), `shift = 0`, `mask = 0xFFFF_FFFF`
    // — so the read is byte-identical to the former `cindex[feature * n + obj]` load,
    // routed through the single accessor. Guard the offset against the u32 device index
    // range (T-10-16); `feature * n <= cindex_stride`, already validated by the caller.
    let split_offset_usize = (feature as usize).checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "partition split: feature ({feature}) * n ({n}) overflows usize"
        ))
    })?;
    let split_offset = u32::try_from(split_offset_usize).map_err(|_| {
        CbError::OutOfRange(format!(
            "partition split: cindex offset {split_offset_usize} exceeds u32 device index range"
        ))
    })?;
    let split_shift = 0u32;
    let split_mask = u32::MAX;

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
        split_offset,
        split_shift,
        split_mask,
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
        split_offset,
        split_shift,
        split_mask,
        bin,
        level_bit,
    );

    Ok(new_leaf_of)
}

/// Packed-cindex variant of [`launch_partition_split_into`] (round-3 perf): the split
/// feature's bin is read through the ONE `read_bin` accessor with its REAL `TCFeature`
/// `(offset, shift, mask)` descriptor over the resident bit-packed `words` — `read_bin`
/// reproduces the plain feature-major bin bit-exactly (the `kernels/cindex.rs` pack→read
/// oracle), so the forward-bit routing decision is byte-identical while the
/// 4-byte-per-cell plain replica never needs uploading.
///
/// The output buffer is `client.empty` (NOT zero-created + uploaded): the grid-stride
/// covers every `i in 0..n` and writes `new_leaf_of[indices[i]]` unconditionally, so as
/// long as `indices` covers `0..n` (the resident session's `indices` is the IDENTITY
/// permutation by construction) every cell is written before any read. A caller whose
/// `indices` does not cover `0..n` must use [`launch_partition_split_into`] (which
/// zero-creates) instead.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_partition_split_packed_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: Handle,
    cindex_words: Handle,
    indices: Handle,
    leaf_of: Handle,
    n: usize,
    num_words: usize,
    offset: u32,
    shift: u32,
    mask: u32,
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

    // No host-side zero buffer + upload here: the kernel writes every covered object
    // (see the doc invariant above), so an uninitialized device allocation suffices.
    let new_leaf_of = client.empty(n * std::mem::size_of::<u32>());

    #[cfg(feature = "wgpu")]
    partition_split_kernel::launch::<f32, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(der1, n) },
        unsafe { ArrayArg::from_raw_parts(cindex_words, num_words) },
        unsafe { ArrayArg::from_raw_parts(indices, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of, n) },
        unsafe { ArrayArg::from_raw_parts(new_leaf_of.clone(), n) },
        offset,
        shift,
        mask,
        bin,
        level_bit,
    );

    #[cfg(not(feature = "wgpu"))]
    partition_split_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(der1, n) },
        unsafe { ArrayArg::from_raw_parts(cindex_words, num_words) },
        unsafe { ArrayArg::from_raw_parts(indices, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of, n) },
        unsafe { ArrayArg::from_raw_parts(new_leaf_of.clone(), n) },
        offset,
        shift,
        mask,
        bin,
        level_bit,
    );

    Ok(new_leaf_of)
}

/// Recompute the per-partition `Σ der1` / `Σ weight` / `Σ (der2·weight)` device-resident
/// after a split: returns a NEW `part_stats` handle of length `n_parts * 3` (channel 0 =
/// Σ der1, channel 1 = Σ weight, channel 2 = Σ (der2·weight) — the GPUT-07 Newton hessian
/// channel) reduced over the resident `leaf_of` partition via the in-kernel atomic merge
/// (D-03). The bulk routing stays device-resident — NO `read_one` here (D-05); the grow
/// loop reads back the part-stats ONCE at the leaves.
///
/// `der2` is the per-object UNWEIGHTED second derivative handle (the Phase 7.2 seam
/// contract): [`DerUnaryKernel::LoglossHessian`] (`-p(1-p)`) for the Logloss/Newton arm,
/// or the constant `-1.0` ([`const_der_handle`], via a resident `-1` upload) for the RMSE
/// arm. The weight is folded IN-KERNEL as `der2·weight` (A3 landmine — matches
/// `cb_compute::reduce_leaf_der2`). It MUST be bound to THIS `client` (a CubeCL Handle is
/// bound to its allocating client — Pitfall 3); the RMSE `calc_average` arm ignores channel
/// 2 but still supplies a valid `n`-length der2 handle so the launch is well-formed.
///
/// `n` is the object count; `n_parts` is `2^level` (the current partition count). The
/// host validated `leaf_of[obj] < n_parts` on upload so the atomic store stays in bounds.
/// `n_parts * 3` overflow surfaces a typed [`CbError::OutOfRange`]. Empty (`n == 0` or
/// `n_parts == 0`) returns a zero-length handle with NO launch (Pitfall 5). No
/// `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
pub(crate) fn launch_partition_update_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: Handle,
    weight: Handle,
    der2: Handle,
    indices: Handle,
    leaf_of: Handle,
    n: usize,
    n_parts: usize,
) -> CbResult<Handle> {
    if n == 0 || n_parts == 0 {
        return Ok(client.empty(0));
    }

    let part_stats_len = n_parts.checked_mul(3).ok_or_else(|| {
        CbError::OutOfRange(format!("n_parts ({n_parts}) * 3 overflows usize (part-stats length)"))
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
            unsafe { ArrayArg::from_raw_parts(der2, n) },
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
            unsafe { ArrayArg::from_raw_parts(der2, n) },
            unsafe { ArrayArg::from_raw_parts(indices, n) },
            unsafe { ArrayArg::from_raw_parts(leaf_of, n) },
            unsafe { ArrayArg::from_raw_parts(h.clone(), part_stats_len) },
        );
        h
    };

    Ok(part_stats)
}

// ===========================================================================
// Phase 11 Plan 02 (GPUT-05 / GPUT-06) — host launch fns for the partition-aware
// `pointwise_hist2` (`fullPass = false`, `2^level` leaf slots, fixed-point Atomic<u64>
// accumulate) and the histogram SUBTRACTION trick (parent − smaller, weight-channel
// max(0) clamp). Both validate ALL buffer sizing + VALUE ranges with `checked_*` →
// typed `CbError` BEFORE launch (T-11-02-01/02). Cross-oracled in `kernels/grow_loop.rs`
// against the CPU leaf-keyed scatter `cb_compute::reduce_leaf_stats`.
// ===========================================================================

/// Fill the device-resident partition-aware 2-channel pointwise histogram
/// (`fullPass = false`) on the compile-time [`SelectedRuntime`] and return the
/// FIXED-POINT `u64` `binSums` DEVICE BUFFER HANDLE — WITHOUT reading it back (the
/// GPUT-05 depth>1 capability). `leaf_of[obj]` routes each object into its partition's
/// histogram line, so the buffer holds `2^level` concatenated leaf slots
/// (`n_parts * n_features * n_bins * HIST_CHANNELS` `u64` cells). The merge uses the
/// LOCKED deterministic fixed-point `Atomic<u64>` path (GPUT-06); decode each cell as
/// `(bits as i64) as f64 / 2^30` on read-back ([`read_fixedpoint_hist_f64`]).
///
/// # Buffer sizing + value-range guards (T-11-02-01/02)
///
/// `2^level` (`checked_shl`), the cindex stride `n_features * n` (`checked_mul`), the
/// per-leaf line `n_features * n_bins * HIST_CHANNELS` ([`hist2_binsums_len_checked`]),
/// and the total `per_leaf * 2^level` (`checked_mul`) are ALL overflow-checked →
/// [`CbError::OutOfRange`] BEFORE any product is formed. `der1`/`weight`/`indices`/
/// `leaf_of` length mismatches and any `indices[i] >= n`, `cindex bin >= n_bins`, or
/// `leaf_of[obj] >= 2^level` surface a typed [`CbError`] BEFORE launch — so a
/// device-side OOB store is impossible. Empty input short-circuits to a zero-length
/// handle (no launch). No `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_partition_hist2_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    indices: &[u32],
    leaf_of_h: Handle,
    n_bins: usize,
    n_features: usize,
    level: u32,
) -> CbResult<Handle> {
    let n = der1.len();

    // Shape guards.
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

    // cindex stride overflow guard + length guard (host slice — validated before packing).
    let cindex_stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n ({n}) overflows usize (cindex stride)"
        ))
    })?;
    if cindex.len() != cindex_stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: cindex_stride,
            actual: cindex.len(),
        });
    }

    // Value-range guards (T-11-02-01): the VALUES inside indices/cindex drive unchecked
    // device array indices; validate them host-side so a malformed id/bin surfaces a typed
    // error rather than an out-of-bounds device store (UB). `leaf_of` is now a device-resident
    // HANDLE (D-05 — the grow loop's routing NEVER crosses to host); its `< 2^level` range is
    // guaranteed by construction (`launch_partition_split_into` only sets bits up to `level`),
    // and the resident core's `checked_shl`/`checked_mul` still bound the buffer sizing.
    if let Some(&bad) = indices.iter().find(|&&ix| (ix as usize) >= n) {
        return Err(CbError::OutOfRange(format!(
            "indices value {bad} >= n ({n}); object id would read der1/cindex out of bounds"
        )));
    }
    if let Some(&bad) = cindex.iter().find(|&&b| (b as usize) >= n_bins) {
        return Err(CbError::OutOfRange(format!(
            "cindex bin value {bad} >= n_bins ({n_bins}); would write bin_sums out of bounds"
        )));
    }

    // Empty fill: hand back a zero-length handle (no launch, no read-back).
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // Build the bit-packed grouped cindex + per-feature TCFeature table and upload the
    // channel-typed der1/weight + packed words + (offsets, shifts, masks) + indices. All
    // features share `n_bins` buckets (the value-range guard above rejected any bin >= n_bins,
    // so `pack_cindex` masks each field losslessly). The device-resident `leaf_of_h` routes
    // each object into its partition slot (D-05).
    let n_buckets_per_feature = vec![n_bins; n_features];
    let packed = crate::gpu_runtime::cindex::pack_cindex(cindex, &n_buckets_per_feature, n)?;
    let (offsets_v, shifts_v, masks_v) = packed.device_arrays()?;
    let num_words = packed.words.len();

    let der1_h = upload_channel_floats(client, der1);
    let weight_h = upload_channel_floats(client, weight);
    let cindex_words_h = client.create(cubecl::bytes::Bytes::from_elems(packed.words.clone()));
    let offsets_h = client.create(cubecl::bytes::Bytes::from_elems(offsets_v));
    let shifts_h = client.create(cubecl::bytes::Bytes::from_elems(shifts_v));
    let masks_h = client.create(cubecl::bytes::Bytes::from_elems(masks_v));
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));

    launch_partition_hist2_resident_into(
        client,
        der1_h,
        weight_h,
        cindex_words_h,
        offsets_h,
        shifts_h,
        masks_h,
        indices_h,
        leaf_of_h,
        num_words,
        n,
        n_bins,
        n_features,
        level,
        /* filter_mask = */ 0,
    )
}

/// The RESIDENT-HANDLE core of the partition-aware `pointwise_hist2` fill (Phase 11 Plan 03):
/// fills the `2^level` leaf slots of the FIXED-POINT `u64` histogram from ALREADY-resident
/// device handles (der1/weight channel floats, bit-packed grouped cindex + per-feature
/// offsets/shifts/masks, indices, and the resident `leaf_of` routing) — WITHOUT re-uploading or
/// reading anything back (the D-05 depth>1 grow-loop seam). Both the host-slice entry
/// [`launch_partition_hist2_into`] (which packs + uploads once, then calls here) and the
/// resident session grow loop ([`grow_oblivious_tree_resident`], which threads its persistent
/// packed handles) route through this ONE launcher.
///
/// # Buffer-sizing guards (T-11-03-01)
///
/// `2^level` (`checked_shl`), the per-leaf line `n_features * n_bins * HIST_CHANNELS`
/// ([`hist2_binsums_len_checked`]), and the total `per_leaf * 2^level` (`checked_mul`) are ALL
/// overflow-checked → [`CbError::OutOfRange`] BEFORE any product is formed, covering the
/// per-level `2^level` slot sizing. The caller guarantees `leaf_of[obj] < 2^level` (by
/// construction / host validation). Empty input short-circuits to a zero-length handle (no
/// launch). No `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_partition_hist2_resident_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1_h: Handle,
    weight_h: Handle,
    cindex_words_h: Handle,
    offsets_h: Handle,
    shifts_h: Handle,
    masks_h: Handle,
    indices_h: Handle,
    leaf_of_h: Handle,
    num_words: usize,
    n: usize,
    n_bins: usize,
    n_features: usize,
    level: u32,
    filter_mask: u32,
) -> CbResult<Handle> {
    // Partition count `2^level` (checked shift — a `level >= usize::BITS` would wrap).
    let n_parts = 1usize.checked_shl(level).ok_or_else(|| {
        CbError::OutOfRange(format!("2^level overflows usize (level={level})"))
    })?;
    let per_leaf = hist2_binsums_len_checked(n_bins, n_features).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {HIST_CHANNELS} overflows usize (per-leaf line)"
        ))
    })?;
    let total = per_leaf.checked_mul(n_parts).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "per_leaf ({per_leaf}) * n_parts ({n_parts}) overflows usize (partition binSums length)"
        ))
    })?;

    // Empty fill: hand back a zero-length handle (no launch, no read-back).
    if n == 0 || n_features == 0 || n_bins == 0 {
        return Ok(client.empty(0));
    }

    // WR-02 / IN-02: the fill accumulates into `&Array<Atomic<u64>>`, so a backend without
    // u64 atomic-add (cpu/wgpu) physically cannot run this kernel. Gate on the device's
    // ADVERTISED capability here — the single choke point both grow paths route through
    // (`launch_partition_hist2_into` and `grow_oblivious_tree_resident`) — and surface a
    // typed error BEFORE launch rather than attempting an unsupported kernel.
    if !device_supports_u64_atomic_add(client) {
        return Err(CbError::Unsupported(
            "partition-aware histogram fill requires Atomic<u64> add, which the active backend \
             does not advertise (cpu/wgpu lack u64 atomics — use the rocm/cuda backend)"
                .to_owned(),
        ));
    }

    // Bit-width family selection (same host-side pick as the depth-1 seam).
    let bits: u32 = match n_bins {
        32 => 5,
        64 => 6,
        128 => 7,
        256 => 8,
        _ => {
            return Err(CbError::Degenerate(format!(
                "partition_hist2 one-byte non-binary fill expects n_bins in {{32,64,128,256}} \
                 (1 << bits for bits in 5..=8), got {n_bins}"
            )));
        }
    };

    // ---- Round-4 Tier-1 LDS privatization (CubeCL manual 08_atomic_contention.md §4):
    // when every cube can hold its (active-partition × feature-tile) slice of the
    // fixed-point histogram in shared memory, scatter into the per-cube LDS copy and
    // merge with ONE global atomic per non-zero cell — global-atomic traffic drops from
    // O(n · n_features) to O(cubes · tile_cells). Integer fixed-point adds are exact and
    // commutative, so this is BIT-IDENTICAL to the multi-copy global-atomic path
    // (GPUT-06 unchanged). The filter compression needs `filter_mask == half` (the grow
    // loop's `1 << (L-1)` invariant); any other non-zero mask keeps the proven path.
    //
    // WHICH arm is faster is HARDWARE-dependent (measured 2026-07-16): shared-memory
    // u64 atomics win on modern parts but LOSE to L2-side global u64 atomics on Pascal
    // (Kaggle P100: LDS ~6.9 ms/tree vs multi-copy ~5 ms/tree). Since both arms are
    // bit-identical, the choice is pure throughput — so it is PROBED once per process
    // on the first eligible fill (both arms fenced and timed on the real inputs, the
    // faster latched; see [`hist_fill_path`]) instead of hard-coding either.
    // `CB_HIST_LDS` (=1 force LDS / =0 force multi-copy) overrides the probe for A/B
    // provenance.
    let n_active = if filter_mask == 0 { n_parts } else { filter_mask as usize };
    let mask_compressible =
        filter_mask == 0 || (filter_mask as usize).checked_mul(2) == Some(n_parts);
    let lds_cpf = if mask_compressible {
        n_active
            .checked_mul(n_bins)
            .and_then(|v| v.checked_mul(HIST_CHANNELS))
            .filter(|&cpf| cpf <= HIST_LDS_CELLS_LARGE)
    } else {
        None
    };

    if let Some(cpf) = lds_cpf {
        match hist_fill_path(cpf) {
            Some(HistFillPath::Lds) => {
                return launch_partition_hist2_lds(
                    client,
                    der1_h,
                    weight_h,
                    cindex_words_h,
                    offsets_h,
                    shifts_h,
                    masks_h,
                    indices_h,
                    leaf_of_h,
                    num_words,
                    n,
                    n_features,
                    total,
                    n_active,
                    cpf,
                    filter_mask,
                    bits,
                );
            }
            Some(HistFillPath::MultiCopy) => {}
            None => {
                // One-time probe (JIT-warmed, fenced, best-of-2 per arm). Every launch
                // below is a REAL fill of these exact inputs — all results are
                // bit-identical, so the winner's LAST handle is returned as the level's
                // histogram and the discarded ones just return to the memory pool.
                let run_mc = || -> CbResult<Handle> {
                    launch_partition_hist2_multicopy(
                        client,
                        der1_h.clone(),
                        weight_h.clone(),
                        cindex_words_h.clone(),
                        offsets_h.clone(),
                        shifts_h.clone(),
                        masks_h.clone(),
                        indices_h.clone(),
                        leaf_of_h.clone(),
                        num_words,
                        n,
                        n_parts,
                        n_features,
                        total,
                        filter_mask,
                        bits,
                    )
                };
                let run_lds = || -> CbResult<Handle> {
                    launch_partition_hist2_lds(
                        client,
                        der1_h.clone(),
                        weight_h.clone(),
                        cindex_words_h.clone(),
                        offsets_h.clone(),
                        shifts_h.clone(),
                        masks_h.clone(),
                        indices_h.clone(),
                        leaf_of_h.clone(),
                        num_words,
                        n,
                        n_features,
                        total,
                        n_active,
                        cpf,
                        filter_mask,
                        bits,
                    )
                };
                // JIT warm-up launches (untimed — the first launch of each kernel
                // family compiles it; timing that would swamp the exec comparison).
                drop(run_mc()?);
                drop(run_lds()?);
                prof_sync(client);
                // Best-of-2 fenced timings per arm.
                let mut t_mc = f64::INFINITY;
                let mut t_lds = f64::INFINITY;
                let mut mc_out: Option<Handle> = None;
                let mut lds_out: Option<Handle> = None;
                for _ in 0..2 {
                    let t0 = std::time::Instant::now();
                    let h = run_mc()?;
                    prof_sync(client);
                    t_mc = t_mc.min(t0.elapsed().as_secs_f64());
                    mc_out = Some(h);
                    let t1 = std::time::Instant::now();
                    let h = run_lds()?;
                    prof_sync(client);
                    t_lds = t_lds.min(t1.elapsed().as_secs_f64());
                    lds_out = Some(h);
                }
                let chosen = if t_lds <= t_mc {
                    HistFillPath::Lds
                } else {
                    HistFillPath::MultiCopy
                };
                if let Ok(mut probed) = HIST_FILL_PROBED.lock() {
                    probed.insert(cpf, chosen);
                }
                if gpu_prof_enabled() {
                    println!(
                        "CB_GPU_PROF hist-fill probe cpf={cpf}: multicopy={:.2}ms lds={:.2}ms -> {chosen:?}",
                        t_mc * 1e3,
                        t_lds * 1e3,
                    );
                }
                // Return the winner's (bit-identical) histogram; the loser's drops.
                let winner = match chosen {
                    HistFillPath::Lds => lds_out,
                    HistFillPath::MultiCopy => mc_out,
                };
                return winner.ok_or_else(|| {
                    CbError::Degenerate(
                        "hist-fill probe produced no histogram handle (internal invariant)"
                            .to_owned(),
                    )
                });
            }
        }
    }

    launch_partition_hist2_multicopy(
        client,
        der1_h,
        weight_h,
        cindex_words_h,
        offsets_h,
        shifts_h,
        masks_h,
        indices_h,
        leaf_of_h,
        num_words,
        n,
        n_parts,
        n_features,
        total,
        filter_mask,
        bits,
    )
}

/// The fill-arm selector for the LDS-eligible partition histogram fill, keyed by the
/// fill's `cells_per_feature` (`n_active * n_bins * HIST_CHANNELS` — the level SHAPE).
/// Per-shape latching matters: measured on Kaggle P100 (2026-07-16, r4b), the LDS arm
/// WINS the shallow 1-partition fill (probe 1.18 ms vs 1.69 ms) but LOSES the deep
/// multi-slot fills (Pascal's shared-mem u64 atomics vs its L2 global atomics), so one
/// global choice is wrong at one end — each distinct shape gets its own probe instead.
/// Resolution order: the `CB_HIST_LDS` env override (`"1"` → LDS, `"0"` → multi-copy;
/// latched at first read like `CB_GPU_PROF`), then the shape's probe-latched winner,
/// then `None` (→ the caller runs the one-time probe for this shape). Process-global:
/// one device per process by construction (compile-time backend selection, D-02), and
/// both arms are bit-identical, so a latched choice can never change results — only
/// throughput.
fn hist_fill_path(cells_per_feature: usize) -> Option<HistFillPath> {
    static ENV_OVERRIDE: std::sync::OnceLock<Option<HistFillPath>> = std::sync::OnceLock::new();
    let env = ENV_OVERRIDE.get_or_init(|| match std::env::var("CB_HIST_LDS").ok().as_deref() {
        Some("1") => Some(HistFillPath::Lds),
        Some("0") => Some(HistFillPath::MultiCopy),
        _ => None,
    });
    if let Some(p) = env {
        return Some(*p);
    }
    HIST_FILL_PROBED
        .lock()
        .ok()
        .and_then(|probed| probed.get(&cells_per_feature).copied())
}

/// The two bit-identical fill arms of the LDS-eligible partition histogram fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistFillPath {
    /// Per-cube shared-memory privatization ([`partition_hist2_lds_kernel`]).
    Lds,
    /// Multi-copy global-atomic privatization ([`partition_hist2_nonbinary_kernel`] +
    /// fold).
    MultiCopy,
}

/// The probe-latched fill arm per level shape (`cells_per_feature` → winner; each shape
/// is probed exactly once per process, on its first eligible fill).
static HIST_FILL_PROBED: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<usize, HistFillPath>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// The MULTI-COPY global-atomic arm of [`launch_partition_hist2_resident_into`] — the
/// pre-round-4 body, extracted verbatim so the one-time probe can time it against the
/// LDS arm. Allocates `n_copies` privatized copies (level-scaled — see
/// [`HIST_MAX_COPIES`]), zeroes them on device, scatters with global fixed-point
/// `Atomic<u64>` adds, and folds the copies into copy 0 (integer adds — exact,
/// order-independent, GPUT-06).
#[allow(clippy::too_many_arguments)]
fn launch_partition_hist2_multicopy(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1_h: Handle,
    weight_h: Handle,
    cindex_words_h: Handle,
    offsets_h: Handle,
    shifts_h: Handle,
    masks_h: Handle,
    indices_h: Handle,
    leaf_of_h: Handle,
    num_words: usize,
    n: usize,
    n_parts: usize,
    n_features: usize,
    total: usize,
    filter_mask: u32,
    bits: u32,
) -> CbResult<Handle> {
    // Level-scaled privatized copy count (the contention fix): shallow levels concentrate
    // every object's 2-atomic scatter on few hot cells, so they get the most copies; the
    // total allocation stays ~`HIST_MAX_COPIES × per-leaf line` at every level.
    let n_copies = (HIST_MAX_COPIES / n_parts).clamp(1, HIST_MAX_COPIES);
    let alloc_len = total.checked_mul(n_copies).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "total ({total}) * n_copies ({n_copies}) overflows usize (multi-copy binSums length)"
        ))
    })?;

    // Allocate the multi-copy fixed-point u64 histogram and zero it ON DEVICE (0u64 is the
    // additive identity in two's complement too — manual §3; no O(histogram) zero bytes
    // cross the bus). The fill kernel accumulates into it.
    let out = client.empty(alloc_len * std::mem::size_of::<u64>());
    let hist_dim = CubeDim {
        x: HIST_CUBE_DIM as u32,
        y: 1,
        z: 1,
    };
    let zero_cubes = alloc_len.div_ceil(HIST_CUBE_DIM).max(1);
    zero_u64_kernel::launch::<SelectedRuntime>(
        client,
        CubeCount::Static(zero_cubes as u32, 1, 1),
        hist_dim,
        unsafe { ArrayArg::from_raw_parts(out.clone(), alloc_len) },
    );

    let num_cubes = n.div_ceil(HIST_CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = hist_dim;

    // The der1/weight channel float type is f32 on wgpu, f64 elsewhere (RESEARCH A1); the
    // fixed-point `Atomic<u64>` accumulator output is `u64` on both. (IN-02: wgpu/cpu lack
    // u64 atomics — this launcher itself gates on the device's advertised `Atomic<u64>` add
    // capability above and returns `CbError::Unsupported` before reaching this launch, so the
    // kernel below only runs on a backend that actually supports it.)
    #[cfg(feature = "wgpu")]
    partition_hist2_nonbinary_kernel::launch::<f32, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(der1_h, n) },
        unsafe { ArrayArg::from_raw_parts(weight_h, n) },
        unsafe { ArrayArg::from_raw_parts(cindex_words_h, num_words) },
        unsafe { ArrayArg::from_raw_parts(offsets_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(shifts_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(masks_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(indices_h, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of_h, n) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), alloc_len) },
        n_features as u32,
        n_copies as u32,
        filter_mask,
        bits,
    );

    #[cfg(not(feature = "wgpu"))]
    partition_hist2_nonbinary_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(der1_h, n) },
        unsafe { ArrayArg::from_raw_parts(weight_h, n) },
        unsafe { ArrayArg::from_raw_parts(cindex_words_h, num_words) },
        unsafe { ArrayArg::from_raw_parts(offsets_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(shifts_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(masks_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(indices_h, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of_h, n) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), alloc_len) },
        n_features as u32,
        n_copies as u32,
        filter_mask,
        bits,
    );

    // Fold the privatized copies into copy 0 (integer adds — exact, order-independent,
    // GPUT-06 determinism unchanged). Downstream consumers read the handle with length
    // `total` (copy 0 only), so the extra copies never leave the device.
    if n_copies > 1 {
        let fold_cubes = total.div_ceil(HIST_CUBE_DIM).max(1);
        fold_hist_copies_kernel::launch::<SelectedRuntime>(
            client,
            CubeCount::Static(fold_cubes as u32, 1, 1),
            hist_dim,
            unsafe { ArrayArg::from_raw_parts(out.clone(), alloc_len) },
            n_copies as u32,
        );
    }

    // Return a COPY-0 VIEW (`total` cells, `Handle::offset_end` trims the tail copies):
    // kernel consumers always bind an explicit `total` length, but WHOLE-handle reads
    // (`read_fixedpoint_hist_f64`, the grow-loop self-oracles) see the handle's logical
    // size — returning the full `alloc_len` buffer leaked the folded-away tail copies
    // into those read-backs (a silent round-1 multi-copy regression, caught by the
    // round-4 gfx1151 suite once a local device could actually run it). The trim also
    // makes this arm's return SHAPE identical to the LDS arm's, which the probe relies
    // on (either arm's handle is a drop-in level histogram).
    let tail_bytes = (alloc_len - total) * std::mem::size_of::<u64>();
    Ok(out.offset_end(tail_bytes as u64))
}

/// The LDS-privatized arm of [`launch_partition_hist2_resident_into`] (round 4, Tier-1
/// shared-memory privatization — CubeCL manual `08_atomic_contention.md` §4). Allocates
/// the SINGLE-copy fixed-point histogram (`total` cells — no multi-copy, no fold pass:
/// privatization lives in each cube's shared memory instead), zeroes it on device, and
/// dispatches [`partition_hist2_lds_kernel`] on a 2-D grid (X = object chunks, Y =
/// feature tiles).
///
/// # Tile geometry
///
/// `cells_per_feature = n_active * n_bins * HIST_CHANNELS` is the caller-checked
/// (`<= HIST_LDS_CELLS_LARGE`) per-feature LDS cost. The tile width is capped by the
/// LDS budget (`tile_cap`), then the launch splits its parallelism between object
/// chunks and feature tiles to land near [`HIST_LDS_TARGET_CUBES`] total cubes; the
/// smallest comptime LDS tier covering `cells_per_feature * tile_f` is selected so
/// shallow levels keep a small footprint (higher occupancy). `ceil(f / ceil(f/t)) <= t`
/// keeps `tile_f` within the cap; the final `clamp` is the belt.
///
/// Bit-exactness: the kernel accumulates the SAME `fixedpoint_encode` contributions with
/// integer atomics (LDS then global) — order-independent, so the result is byte-identical
/// to the multi-copy global-atomic arm (GPUT-06). No read-back; the returned handle is
/// consumed with length `total` exactly like the multi-copy arm's copy 0.
#[allow(clippy::too_many_arguments)]
fn launch_partition_hist2_lds(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1_h: Handle,
    weight_h: Handle,
    cindex_words_h: Handle,
    offsets_h: Handle,
    shifts_h: Handle,
    masks_h: Handle,
    indices_h: Handle,
    leaf_of_h: Handle,
    num_words: usize,
    n: usize,
    n_features: usize,
    total: usize,
    n_active: usize,
    cells_per_feature: usize,
    filter_mask: u32,
    bits: u32,
) -> CbResult<Handle> {
    // Allocate + device-zero the single-copy fixed-point u64 histogram (0u64 is the
    // additive identity — no O(histogram) zero bytes cross the bus).
    let out = client.empty(total * std::mem::size_of::<u64>());
    let hist_dim = CubeDim {
        x: HIST_CUBE_DIM as u32,
        y: 1,
        z: 1,
    };
    let zero_cubes = total.div_ceil(HIST_CUBE_DIM).max(1);
    zero_u64_kernel::launch::<SelectedRuntime>(
        client,
        CubeCount::Static(zero_cubes as u32, 1, 1),
        hist_dim,
        unsafe { ArrayArg::from_raw_parts(out.clone(), total) },
    );

    // Tile geometry (see the doc header). `cells_per_feature >= 1` (caller guards
    // n_bins/n_features > 0), so no division by zero.
    let tile_cap = (HIST_LDS_CELLS_LARGE / cells_per_feature).clamp(1, n_features);
    let groups_min = n_features.div_ceil(tile_cap);
    let obj_chunk_cap = n.div_ceil(HIST_CUBE_DIM * HIST_LDS_MIN_OBJ_PER_THREAD).max(1);
    let chunks = (HIST_LDS_TARGET_CUBES / groups_min).clamp(1, obj_chunk_cap);
    let groups_desired = HIST_LDS_TARGET_CUBES.div_ceil(chunks).max(groups_min);
    let tile_f = n_features.div_ceil(groups_desired).clamp(1, tile_cap);
    let groups = n_features.div_ceil(tile_f);

    // Smallest comptime LDS tier covering the tile (JIT-instantiated per tier).
    let lds_needed = cells_per_feature * tile_f;
    let lds_cells = if lds_needed <= HIST_LDS_CELLS_SMALL {
        HIST_LDS_CELLS_SMALL
    } else if lds_needed <= HIST_LDS_CELLS_MEDIUM {
        HIST_LDS_CELLS_MEDIUM
    } else {
        HIST_LDS_CELLS_LARGE
    };

    let count = CubeCount::Static(chunks as u32, groups as u32, 1);

    // The der1/weight channel float type is f32 on wgpu, f64 elsewhere (RESEARCH A1);
    // unreachable on wgpu in practice (the caller's Atomic<u64> capability gate).
    #[cfg(feature = "wgpu")]
    partition_hist2_lds_kernel::launch::<f32, SelectedRuntime>(
        client,
        count,
        hist_dim,
        unsafe { ArrayArg::from_raw_parts(der1_h, n) },
        unsafe { ArrayArg::from_raw_parts(weight_h, n) },
        unsafe { ArrayArg::from_raw_parts(cindex_words_h, num_words) },
        unsafe { ArrayArg::from_raw_parts(offsets_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(shifts_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(masks_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(indices_h, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of_h, n) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), total) },
        n_features as u32,
        filter_mask,
        n_active as u32,
        tile_f as u32,
        bits,
        lds_cells as u32,
    );

    #[cfg(not(feature = "wgpu"))]
    partition_hist2_lds_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        hist_dim,
        unsafe { ArrayArg::from_raw_parts(der1_h, n) },
        unsafe { ArrayArg::from_raw_parts(weight_h, n) },
        unsafe { ArrayArg::from_raw_parts(cindex_words_h, num_words) },
        unsafe { ArrayArg::from_raw_parts(offsets_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(shifts_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(masks_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(indices_h, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of_h, n) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), total) },
        n_features as u32,
        filter_mask,
        n_active as u32,
        tile_f as u32,
        bits,
        lds_cells as u32,
    );

    Ok(out)
}

/// Complete a level's partition histogram from the PARENT level's histogram via the
/// subtraction trick (WR-01 wired, upstream §5.5): the level-`L` fill ran with
/// `filter_mask = 1 << (L-1)`, accumulating ONLY the newest-bit-set partitions
/// `p + half`; this launch derives every unfilled sibling slot as
/// `hist[p] = parent[p] − hist[p + half]` for `p < half = 2^(L-1)` — elementwise over
/// each slot's per-leaf line, device-resident, bit-exact (see
/// [`derive_sibling_partition_hist_kernel`]). `parent_h` is the previous level's
/// (already copy-folded) histogram (`half` slots); `hist_h` the current level's
/// (`2 * half` slots). No read-back; empty dimensions are a no-op.
pub(crate) fn launch_derive_sibling_hist_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    parent_h: &Handle,
    hist_h: &Handle,
    half: usize,
    n_bins: usize,
    n_features: usize,
) -> CbResult<()> {
    if half == 0 || n_bins == 0 || n_features == 0 {
        return Ok(());
    }
    let per_leaf = hist2_binsums_len_checked(n_bins, n_features).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {HIST_CHANNELS} overflows usize (per-leaf line)"
        ))
    })?;
    let cells = per_leaf.checked_mul(half).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "per_leaf ({per_leaf}) * half ({half}) overflows usize (sibling-derive cell count)"
        ))
    })?;
    let hist_len = cells.checked_mul(2).ok_or_else(|| {
        CbError::OutOfRange(format!("cells ({cells}) * 2 overflows usize (child histogram length)"))
    })?;
    let half_u32 = u32::try_from(half).map_err(|_| {
        CbError::OutOfRange(format!("half ({half}) exceeds u32 (sibling-derive slot count)"))
    })?;
    let leaf_stride_u32 = u32::try_from(per_leaf).map_err(|_| {
        CbError::OutOfRange(format!("per-leaf line ({per_leaf}) exceeds u32 (sibling-derive stride)"))
    })?;

    let num_cubes = cells.div_ceil(HIST_CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: HIST_CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    #[cfg(feature = "wgpu")]
    derive_sibling_partition_hist_kernel::launch::<f32, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(parent_h.clone(), cells) },
        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
        half_u32,
        leaf_stride_u32,
    );

    #[cfg(not(feature = "wgpu"))]
    derive_sibling_partition_hist_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(parent_h.clone(), cells) },
        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
        half_u32,
        leaf_stride_u32,
    );

    Ok(())
}

/// Score the MULTI-LEAF FIXED-POINT partition histogram device-resident and return the single
/// best oblivious `(feature, bin)` split for the level (GPUT-05 depth>1 score step, Phase 11
/// Plan 03) — the depth>1 sibling of [`score_over_binsums`]. Launches
/// [`find_optimal_split_partition_kernel`] over the resident `bin_sums` handle (consumed IN
/// PLACE, NEVER read to host), folds the single split's score over the `n_parts = 2^level`
/// active leaves, and reads back ONLY the O(1) per-block `(best_gain, best_idx)` winner
/// descriptor — the sole crossing (D-05, T-11-03-02: the full histogram NEVER crosses).
///
/// `bin_sums` is the fixed-point `u64` partition histogram (length `n_parts * n_features *
/// n_bins * HIST_CHANNELS`); `scaled_l2` the per-tree regularizer; `score_fn` the comptime
/// score calcer arm (validated here). Returns the chosen [`BestSplit`] or `None` on a degenerate
/// (no-candidate) level. A read-back failure surfaces [`CbError::Degenerate`] (WR-05). No
/// `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
fn score_partition_over_binsums(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    bin_sums: Handle,
    n_parts: usize,
    n_bins: usize,
    n_bins_used: usize,
    n_features: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CbResult<Option<BestSplit>> {
    // `n_bins` is the (possibly PADDED) histogram line width the fill dispatched
    // ({32,64,128,256}); `n_bins_used` the ACTUAL quantized bin count (<= n_bins). The
    // padding cells are zero and their phantom borders are excluded from the argmin
    // (device-side in the kernel + the host belt below), so the scored candidate set is
    // exactly the CPU reference's real-border enumeration.
    if n_bins_used == 0 || n_bins_used > n_bins {
        return Err(CbError::OutOfRange(format!(
            "n_bins_used ({n_bins_used}) must be in 1..=n_bins ({n_bins})"
        )));
    }
    // Reject an unknown score-fn selector BEFORE any launch (no silent wrong-arm dispatch).
    if score_fn != SCORE_FN_L2
        && score_fn != SCORE_FN_COSINE
        && score_fn != SCORE_FN_SOLAR_L2
        && score_fn != SCORE_FN_LOO_L2
        && score_fn != SCORE_FN_SAT_L2
    {
        return Err(CbError::OutOfRange(format!(
            "unknown score_fn selector ({score_fn}); expected one of L2/Cosine/SolarL2/LOOL2/SatL2"
        )));
    }

    // Empty short-circuit (Pitfall 3/5): no partitions / no candidates -> no split, no launch.
    if n_parts == 0 || n_features == 0 || n_bins == 0 {
        return Ok(None);
    }

    let n_candidates = n_features.checked_mul(n_bins).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) overflows usize (candidate count)"
        ))
    })?;
    let per_leaf = hist2_binsums_len_checked(n_bins, n_features).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "n_features ({n_features}) * n_bins ({n_bins}) * {HIST_CHANNELS} overflows usize (per-leaf line)"
        ))
    })?;
    let bin_sums_len = per_leaf.checked_mul(n_parts).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "per_leaf ({per_leaf}) * n_parts ({n_parts}) overflows usize (partition binSums length)"
        ))
    })?;
    let n_bins_u32 = u32::try_from(n_bins).map_err(|_| {
        CbError::OutOfRange(format!("n_bins ({n_bins}) exceeds u32 (kernel comptime line size)"))
    })?;
    let n_parts_u32 = u32::try_from(n_parts).map_err(|_| {
        CbError::OutOfRange(format!("n_parts ({n_parts}) exceeds u32 device range"))
    })?;

    // MULTI-cube candidate sweep (perf pass): one CUBE_DIM-thread cube per CUBE_DIM
    // candidates, so the argmin uses the whole device instead of one SM (the old
    // `num_cubes = 1` dispatch serialized ~n_candidates * n_parts * n_bins decode+add
    // iterations onto a single 32-thread warp — the dominant per-level scorer cost at
    // depth > 1). Every candidate is still scored by exactly ONE thread and every fold
    // keeps the strict-`>`/lowest-index tie-break, so the chosen split is bit-identical
    // (see the kernel doc). The shared-mem argmin size stays the comptime ARGMIN_SHMEM
    // (== CUBE_DIM, Pitfall 3).
    let num_cubes = n_candidates.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // Packed per-block winner buffer: gains in slots `0..num_cubes`, candidate indices
    // (widened to the channel float — exact for every u32 in f64) in slots
    // `num_cubes..2*num_cubes`, retrieved with ONE read-back (one sync per level, not two).
    #[cfg(feature = "wgpu")]
    let best_out_handle = {
        let best_out_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32; 2 * num_cubes]));
        let lambda_h = client.create(cubecl::bytes::Bytes::from_elems(vec![scaled_l2 as f32]));
        find_optimal_split_partition_kernel::launch::<f32, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, bin_sums_len) },
            unsafe { ArrayArg::from_raw_parts(best_out_h.clone(), 2 * num_cubes) },
            unsafe { ArrayArg::from_raw_parts(lambda_h, 1) },
            n_parts_u32,
            n_features as u32,
            n_bins_used as u32,
            n_bins_u32,
            score_fn,
        );
        best_out_h
    };

    #[cfg(not(feature = "wgpu"))]
    let best_out_handle = {
        let best_out_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f64; 2 * num_cubes]));
        let lambda_h = client.create(cubecl::bytes::Bytes::from_elems(vec![scaled_l2]));
        find_optimal_split_partition_kernel::launch::<f64, SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(bin_sums, bin_sums_len) },
            unsafe { ArrayArg::from_raw_parts(best_out_h.clone(), 2 * num_cubes) },
            unsafe { ArrayArg::from_raw_parts(lambda_h, 1) },
            n_parts_u32,
            n_features as u32,
            n_bins_used as u32,
            n_bins_u32,
            score_fn,
        );
        best_out_h
    };

    // Read back ONLY the O(blocks) per-block winner descriptors (the sole D-05 crossing) and
    // finish the across-block argmin host-side with the SAME lowest-index tie-break the kernel
    // uses. The bulk histogram never leaves the device (T-11-03-02).
    let best_out = read_scores_f64(client, best_out_handle)?;
    let (best_gains, best_idx_f) = best_out
        .split_at_checked(num_cubes)
        .ok_or_else(|| {
            CbError::Degenerate(format!(
                "partition best-split read-back returned {} slots, expected {}",
                best_out.len(),
                2 * num_cubes
            ))
        })?;

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_c = u32::MAX;
    for (block, &gain) in best_gains.iter().enumerate() {
        // The idx slot is an exact small integer in the channel float; an out-of-range or
        // non-finite slot falls to u32::MAX and is skipped by the candidate-range guard.
        let cand = best_idx_f
            .get(block)
            .copied()
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= f64::from(u32::MAX))
            .map_or(u32::MAX, |v| v as u32);
        if (cand as usize) >= n_candidates {
            continue;
        }
        // Trailing no-op border AND phantom padded borders (`border >= n_bins_used - 1`)
        // are all-LEFT no-op splits (the device kernel already excludes them; host belt, WR-05).
        if (cand as usize) % n_bins >= n_bins_used - 1 {
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
        return Ok(Some(BestSplit {
            feature_id: feature as u32,
            bin_id: bin as u32,
            score: best_gain as f32,
            gain: best_gain as f32,
        }));
    }

    Ok(None)
}

/// Derive the LARGER sibling's histogram device-resident via the SUBTRACTION trick:
/// `bigger = parent − smaller` per fixed-point cell, clamping the `statId == 0`
/// (weight/hessian) channel to `max(0)` (D-04, memory-lean — the smaller child is filled
/// directly by [`launch_partition_hist2_into`], the larger is derived in O(bins)). Returns
/// a fresh `u64` `bigger` handle of length `cells`. `parent`/`smaller` may be multi-slot
/// buffers addressed at `parent_base`/`smaller_base` (the parent-resident slot + the
/// directly-computed smaller sibling, selected by partition Size host-side); `cells` is
/// one leaf line's length.
///
/// # Buffer-size guards (T-11-02-01/02)
///
/// `parent_base + cells <= parent_len`, `smaller_base + cells <= smaller_len`
/// (`checked_add`), and the base/cells fit the `u32` device index range (`try_from`) —
/// all validated → typed [`CbError`] BEFORE launch. `cells == 0` short-circuits to a
/// zero-length handle. No `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
// WR-01 (Phase 11 review): the subtraction trick is not yet wired into a scored grow path;
// this launcher is retained and unit-tested standalone, so it is dead code only in non-test
// builds until the memory-lean parent-reuse path is actually implemented.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_subtract_histograms_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    parent: Handle,
    parent_len: usize,
    parent_base: usize,
    smaller: Handle,
    smaller_len: usize,
    smaller_base: usize,
    cells: usize,
) -> CbResult<Handle> {
    if cells == 0 {
        return Ok(client.empty(0));
    }

    // Slot-offset bounds: base + cells must not run past either source buffer.
    let parent_end = parent_base.checked_add(cells).ok_or_else(|| {
        CbError::OutOfRange(format!("parent_base ({parent_base}) + cells ({cells}) overflows usize"))
    })?;
    if parent_end > parent_len {
        return Err(CbError::OutOfRange(format!(
            "parent slot [{parent_base}, {parent_end}) exceeds parent buffer length {parent_len}"
        )));
    }
    let smaller_end = smaller_base.checked_add(cells).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "smaller_base ({smaller_base}) + cells ({cells}) overflows usize"
        ))
    })?;
    if smaller_end > smaller_len {
        return Err(CbError::OutOfRange(format!(
            "smaller slot [{smaller_base}, {smaller_end}) exceeds smaller buffer length {smaller_len}"
        )));
    }

    // The kernel takes u32 base/cells scalars (device index domain).
    let parent_base_u32 = u32::try_from(parent_base).map_err(|_| {
        CbError::OutOfRange(format!("parent_base {parent_base} exceeds u32 device index range"))
    })?;
    let smaller_base_u32 = u32::try_from(smaller_base).map_err(|_| {
        CbError::OutOfRange(format!("smaller_base {smaller_base} exceeds u32 device index range"))
    })?;
    let cells_u32 = u32::try_from(cells).map_err(|_| {
        CbError::OutOfRange(format!("cells {cells} exceeds u32 device index range"))
    })?;

    let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u64; cells]));

    let num_cubes = cells.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    #[cfg(feature = "wgpu")]
    subtract_histograms_kernel::launch::<f32, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(parent, parent_len) },
        unsafe { ArrayArg::from_raw_parts(smaller, smaller_len) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), cells) },
        parent_base_u32,
        smaller_base_u32,
        0u32,
        cells_u32,
        HIST_CHANNELS as u32,
    );

    #[cfg(not(feature = "wgpu"))]
    subtract_histograms_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(parent, parent_len) },
        unsafe { ArrayArg::from_raw_parts(smaller, smaller_len) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), cells) },
        parent_base_u32,
        smaller_base_u32,
        0u32,
        cells_u32,
        HIST_CHANNELS as u32,
    );

    Ok(out)
}

/// Read a FIXED-POINT `u64` histogram handle back and DECODE each cell to `f64`
/// (`(bits as i64) as f64 / 2^30` — CubeCL fixed-point-atomics manual §3), the
/// deterministic-accumulator sibling of [`read_binsums_f64`]. `len` is the expected cell
/// count. A read-back failure surfaces [`CbError::Degenerate`] (WR-05), never a silent
/// zero buffer; a length mismatch surfaces [`CbError::LengthMismatch`].
///
/// TEST-SEAM ONLY (`#[cfg(test)]`): decoding the FULL histogram to host is the FORBIDDEN D-05
/// hybrid (T-11-03-02) — the depth>1 grow loop scores the histogram DEVICE-side
/// ([`score_partition_over_binsums`]) and NEVER reads it back. This decoder exists solely for
/// the kernel self-oracles (`kernels::grow_loop`) that assert the fill/subtraction/determinism
/// against the CPU reference; it is compiled out of production builds.
#[cfg(test)]
pub(crate) fn read_fixedpoint_hist_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
    len: usize,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("fixed-point hist read-back failed: {e:?}")))?;
    let bits = bytemuck::cast_slice::<u8, u64>(&bytes);
    if bits.len() != len {
        return Err(CbError::LengthMismatch {
            column: "fixedpoint_hist".to_owned(),
            expected: len,
            actual: bits.len(),
        });
    }
    Ok(bits
        .iter()
        .map(|&b| (b as i64) as f64 / REDUCE_FIXEDPOINT_SCALE_F64)
        .collect())
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
/// # Depth coverage (GPUT-05: depth>1 device-covered)
///
/// Grows a full depth-`depth` oblivious tree. Each level scores its candidate over the
/// CURRENT `2^level` partitions via the partition-aware (`fullPass = false`) histogram fill
/// keyed by the resident `leaf_of` ([`launch_partition_hist2_into`]) + the subtraction trick
/// ([`launch_subtract_histograms_into`], D-04 memory-lean parent reuse) + the per-active-leaf
/// device score/argmin ([`score_partition_over_binsums`]). Level 0 has ONE partition (the
/// root), so its fill is the whole-dataset histogram; level L fills the `2^L` active-leaf
/// slots. Only the O(1) [`BestSplit`] per level + the final `2^depth` part-stats cross
/// host<->device (D-05 / T-11-03-02 — the bulk histogram / doc-routing stay resident; no
/// full-buffer read path). Phase 11 Plan 03 removed the former depth>1 `OutOfRange` reject.
///
/// # Errors
///
/// - [`CbError::OutOfRange`] if `2^depth * 2` / `n_features * n_bins` / a per-level `2^level`
///   slot product overflows `usize`.
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

    // GPUT-05 (Phase 11 Plan 03): depth>1 is now DEVICE-COVERED. Each level's score step fills
    // the partition-aware (fullPass = false) histogram keyed by the resident leaf_of over the
    // current 2^level partitions and scores the single oblivious split across every active leaf
    // (the depth>1 forward dependency the FROZEN 7.3 whole-dataset fill could not serve). The
    // D-05 boundary is preserved — only the O(1) BestSplit per level + the final 2^depth
    // part-stats cross host<->device; the histogram / partition / doc-routing stay resident.

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

    // NOTE (WR-01, Phase 11 review): the D-04 "memory-lean" subtraction trick is NOT wired into
    // this scored path. The partition-aware fill below computes every `2^level` slot directly, so
    // there is no smaller/larger sibling to derive here. The standalone subtraction kernel
    // (`launch_subtract_histograms_into`) is retained and unit-tested, but until it actually
    // replaces the direct fill of the larger sibling it delivers no memory saving — so the claim
    // and the discarded per-level derivation have been removed rather than left as dead code.

    // The host-light per-depth loop (upstream :63-158). Per level: partition-aware fill keyed
    // by the resident leaf_of over the current 2^level partitions ->
    // device score/argmin over every active leaf -> ONE O(1) BestSplit read-back -> host split
    // decision -> device partition-split. The bulk histogram / doc-routing stay resident (D-05).
    for level in 0..depth {
        // (1) Partition count for this level and the partition-aware (fullPass = L>0) fill,
        //     keyed by the DEVICE-RESIDENT leaf_of (D-05 — the routing NEVER crosses to host).
        //     At level 0 (leaf_of all-zero) this fills the single root slot == the whole-dataset
        //     histogram; at level L it fills the 2^L active-leaf slots.
        let n_parts = 1usize.checked_shl(level as u32).ok_or_else(|| {
            CbError::OutOfRange(format!("2^level overflows usize (level={level})"))
        })?;
        let hist_h = launch_partition_hist2_into(
            client, der1, weight, cindex, indices, leaf_of_h.clone(), n_bins, n_features,
            level as u32,
        )?;

        // (2) Device score + deterministic argmin over the CURRENT 2^level partitions. The bulk
        //     histogram stays device-resident IN the score launch; only the O(1) BestSplit
        //     descriptor crosses back (D-05 / T-11-03-02 — no full-buffer read path).
        let best = score_partition_over_binsums(
            client, hist_h.clone(), n_parts, n_bins, n_bins, n_features, scaled_l2, score_fn,
        )?;

        // (3) The O(1) host integer split decision. A level with no candidate at all is a
        //     degenerate dataset (no feature has any bin) — surface a typed error rather
        //     than fabricating a split (T-07.5-03-04).
        let split = best.ok_or_else(|| {
            CbError::Degenerate(format!(
                "grow_oblivious_tree level {level}: no candidate split (degenerate histogram)"
            ))
        })?;
        splits.push((split.feature_id, split.bin_id));

        // (4) Device partition-split (forward-bit doc-routing, level -> bit level == the CPU
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

    // (4) Device partition-update: the per-partition Σ der1 / Σ weight / Σ(der2·weight) reduce
    //     over the final 2^depth partitions (upstream `UpdatePartitionProps`), device-resident.
    //     RMSE arm: der2 = const -1 (RMSE hessian), so the Σ(der2·weight) channel is -Σweight
    //     and `newton_leaf_delta` collapses to `calc_average` (see step 6). The channel is
    //     computed for symmetry with the Newton path but the RMSE leaf reads channels 0/1.
    let der2_rmse_h = create_channel_const(client, -1.0, n);
    let part_stats_h = launch_partition_update_into(
        client,
        der1_h.clone(),
        weight_h.clone(),
        der2_rmse_h,
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
    //     T-07.5-03-06 — no NaN/Inf from an empty leaf). part_stats is now stride-3
    //     [Σder1, Σweight, Σ(der2·weight)] per leaf in leaf-index order; the RMSE arm reads
    //     channels 0/1 (der2=−1 ⇒ `newton_leaf_delta` == `calc_average` exactly, GPUT-07).
    let mut leaf_values = vec![0.0_f64; n_leaves];
    for leaf in 0..n_leaves {
        let sum = part_stats.get(leaf * 3).copied().unwrap_or(0.0);
        let cnt = part_stats.get(leaf * 3 + 1).copied().unwrap_or(0.0);
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

/// The single closed-form Newton leaf step (GPUT-07 / A1 — `leaf_estimation_iterations = 1`),
/// TRANSCRIBED INLINE from `cb_compute::leaf::newton_leaf_delta` (leaf.rs:145-154) rather than
/// imported (the D-7.5-04 boundary — keep cb-backend's cb-compute surface to the already-used
/// `calc_average`/`scale_l2_reg`). `sum_der` is Σ der1 (channel 0, UNWEIGHTED — the numerator
/// the RMSE arm also uses); `sum_der2` is Σ(der2·weight) (channel 2, the Newton hessian —
/// NON-positive). The `denom == 0` guard is the 0/0 empty-leaf case only (never a NaN); every
/// non-zero denominator divides verbatim. RMSE collapses: der2 = -1 ⇒ sum_der2 = -Σweight ⇒
/// denom = Σweight + scaled_l2 ⇒ this == `calc_average` exactly (A3 / the Pitfall-2 collapse
/// check). NO iterative walker, NO backtracking — a single closed-form step (A1/D-02).
fn newton_leaf_delta(sum_der: f64, sum_der2: f64, scaled_l2: f64) -> f64 {
    let denom = -sum_der2 + scaled_l2;
    if denom == 0.0 {
        0.0
    } else {
        sum_der / denom
    }
}

/// Public wrapper over [`grow_oblivious_tree_newton_into`]: construct the client ONCE and grow
/// a depth-`depth` oblivious tree with Newton der2 leaf estimation (GPUT-07 — the Logloss
/// classification default). See [`grow_oblivious_tree`] for the RMSE (`calc_average`) sibling.
#[allow(clippy::too_many_arguments)]
pub fn grow_oblivious_tree_newton(
    der1: &[f64],
    der2: &[f64],
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
    grow_oblivious_tree_newton_into(
        &client, der1, der2, weight, cindex, indices, n_bins, n_features, depth, scaled_l2,
        score_fn,
    )
}

/// Grow a depth-`depth` oblivious tree with device-resident **Newton der2 leaf estimation**
/// (GPUT-07 — the genuinely-new Logloss path; RMSE collapses to `calc_average`, see
/// [`newton_leaf_delta`]).
///
/// The tree STRUCTURE (per-level split sequence + per-object `leaf_of`) is grown by
/// [`grow_oblivious_tree_into`] — the split scoring is identical (Σ der1 / Σ weight, A2), so the
/// structure is reused verbatim rather than duplicating the 130-line partition-aware level loop.
/// Only the LEAF VALUES differ: this re-reduces the per-partition stats with the REAL per-object
/// `der2` handle (the Phase 7.2 seam contract — [`DerUnaryKernel::LoglossHessian`] `-p(1-p)` for
/// Logloss), so the Σ(der2·weight) channel-2 carries the true Newton hessian, then computes each
/// leaf via the inline single-step [`newton_leaf_delta`] (`leaf_estimation_iterations = 1`, A1 —
/// NO iterative walker / backtracking). The re-reduce is device-resident (`launch_partition_update_into`
/// over re-uploaded resident handles on the SAME client); only the O(1) BestSplit per level + the
/// final `2^depth` part-stats + the `leaf_of` cross host<->device (the SAME D-05 crossing class as
/// the RMSE path).
///
/// `der2` (length `n`) is the per-object UNWEIGHTED second derivative (weight is folded in the
/// reduce, A3). `score_fn` selects the split calcer (forwarded to the structure grow). Errors and
/// the empty short-circuit mirror [`grow_oblivious_tree_into`]. No `unwrap`/`expect`/`panic`/
/// indexing (workspace lints + D-13).
#[allow(clippy::too_many_arguments)]
pub(crate) fn grow_oblivious_tree_newton_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der1: &[f64],
    der2: &[f64],
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

    // der2 length must agree with n (the per-object hessian) — surface a typed error rather
    // than launching a malformed reduce (no panic, D-13).
    if der2.len() != n {
        return Err(CbError::LengthMismatch {
            column: "der2".to_owned(),
            expected: n,
            actual: der2.len(),
        });
    }

    // (1) Grow the tree STRUCTURE (splits + leaf_of) via the shared partition-aware loop. Its
    //     leaf_values (RMSE `calc_average`) and part_stats (der2 = -1 channel) are DISCARDED —
    //     only the structure is reused (the split scoring is der2-independent, A2).
    let structure = grow_oblivious_tree_into(
        client, der1, weight, cindex, indices, n_bins, n_features, depth, scaled_l2, score_fn,
    )?;

    // Empty tree (no objects / no candidates): no leaves to estimate — hand back the structure.
    if structure.splits.is_empty() && structure.leaf_values.is_empty() {
        return Ok(structure);
    }

    let n_leaves = 1usize
        .checked_shl(depth as u32)
        .ok_or_else(|| CbError::OutOfRange(format!("2^depth overflows usize (depth = {depth})")))?;

    // (2) Re-reduce the per-partition stats with the REAL der2 (the Newton hessian channel).
    //     Re-upload the resident handles on THIS client (Pitfall 3 — the der2 handle MUST be
    //     bound to the same client as the reduce). leaf_of comes back from the structure grow
    //     (the SAME D-05 crossing class); re-upload it to key the reduce device-side.
    let der1_h = upload_channel_floats(client, der1);
    let weight_h = upload_channel_floats(client, weight);
    let der2_h = upload_channel_floats(client, der2);
    let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));
    let leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(structure.leaf_of.clone()));

    let part_stats_h = launch_partition_update_into(
        client, der1_h, weight_h, der2_h, indices_h, leaf_of_h, n, n_leaves,
    )?;
    let part_stats = read_part_stats_f64(client, part_stats_h)?;

    // (3) Newton leaf values via the inline single-step closed form (A1): channel 0 = Σder1,
    //     channel 2 = Σ(der2·weight); leaf = Σder1 / (-Σ(der2·weight) + scaled_l2). Empty-leaf
    //     denom==0 guard inside `newton_leaf_delta` (never a NaN).
    let mut leaf_values = vec![0.0_f64; n_leaves];
    for leaf in 0..n_leaves {
        let sum_der = part_stats.get(leaf * 3).copied().unwrap_or(0.0);
        let sum_der2 = part_stats.get(leaf * 3 + 2).copied().unwrap_or(0.0);
        if let Some(slot) = leaf_values.get_mut(leaf) {
            *slot = newton_leaf_delta(sum_der, sum_der2, scaled_l2);
        }
    }

    Ok(GrownTree {
        splits: structure.splits,
        leaf_of: structure.leaf_of,
        leaf_values,
        part_stats,
    })
}

/// GPUT-03: apply ONE tree's per-leaf delta to the running approx device-resident
/// (`approx[i] += lr * leaf_values[leaf_of[i]]`) via [`apply_leaf_delta_kernel`], returning
/// the UPDATED approx handle WITHOUT any n-length read-back. The running approx stays a
/// resident device buffer across boosting iterations so the next tree's residual `der1` is
/// recomputed device-side (the resident der seam) with no host round-trip (the must-have
/// no-read-back contract).
///
/// `approx_h` (resident, updated IN PLACE and returned) and `leaf_of_h` (resident, cloned in
/// — NEVER read to host) are length `n`; `leaf_values` (length `2^depth`, small) is the
/// per-leaf UNSCALED delta; `lr` is the learning-rate scale. The small `leaf_values` + `lr`
/// scalar are uploaded as the channel float type (f32 on wgpu, f64 elsewhere). Empty
/// (`n == 0`) or empty `leaf_values` returns `approx_h` unchanged (no launch, Pitfall 5). No
/// `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
pub(crate) fn launch_apply_leaf_delta_into(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    approx_h: Handle,
    leaf_of_h: Handle,
    leaf_values: &[f64],
    lr: f64,
    n: usize,
) -> CbResult<Handle> {
    if n == 0 || leaf_values.is_empty() {
        return Ok(approx_h);
    }

    let n_leaves = leaf_values.len();
    let leaf_values_h = upload_channel_floats(client, leaf_values);
    let lr_h = upload_channel_floats(client, &[lr]);

    let num_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim {
        x: CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    // The kernel writes `approx` in place; clone the resident handle into the `&mut` launch
    // arg so the ORIGINAL stays returnable on-device (a CubeCL Handle clone shares the
    // buffer — the write goes to the shared allocation). NO read-back (SC-3 / Pitfall 1).
    #[cfg(feature = "wgpu")]
    apply_leaf_delta_kernel::launch::<f32, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(approx_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of_h, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_values_h, n_leaves) },
        unsafe { ArrayArg::from_raw_parts(lr_h, 1) },
    );

    #[cfg(not(feature = "wgpu"))]
    apply_leaf_delta_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(approx_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(leaf_of_h, n) },
        unsafe { ArrayArg::from_raw_parts(leaf_values_h, n_leaves) },
        unsafe { ArrayArg::from_raw_parts(lr_h, 1) },
    );

    Ok(approx_h)
}

/// GPUT-02/03: grow ONE depth-1 oblivious tree over PRE-UPLOADED, session-resident device
/// handles (the residency variant of [`grow_oblivious_tree_into`]), keeping the running
/// approx / residual `der1` device-resident across boosting iterations. It re-uploads
/// NOTHING per tree: the quantized matrix (both the packed cindex `words` the histogram
/// reads AND the plain feature-major layout the partition split reads), the weights, the
/// TCFeature table, the indices, and the target are uploaded ONCE by the session's `begin`
/// and cloned into the SAME `hist2_launch_resident` / `score_over_binsums` /
/// `launch_partition_*` geometry here.
///
/// Per tree (MVP depth == 1) it: (1) fills the resident histogram from the resident `der1`
/// handle + resident weight/cindex/indices → the FROZEN score/argmin (Cosine or L2, the
/// depth-1 device default is Cosine, GPUT-08) → ONE O(1) [`BestSplit`] read-back; (2)
/// device `partition_split` (forward-bit doc-routing, resident); (3) device
/// `partition_update` → ONE `2^depth` part-stats read-back → host leaf values via the
/// FROZEN `cb_compute::calc_average`; (4) `apply_leaf_delta` updates the resident approx ON
/// DEVICE; (5) recomputes the resident `der1` for the NEXT tree via the resident der seam —
/// so NO n-length der1/approx read-back crosses per tree (only the O(1) BestSplit + the
/// `2^depth` part-stats, plus ONE `leaf_of` read-back at the end for the structure oracle,
/// the SAME crossing class as [`grow_oblivious_tree_into`]).
///
/// Returns `(GrownTree, approx_h_updated, der1_h_next)`: the grown structure (its
/// `leaf_values` UNSCALED — the caller/`cb-train` applies `learning_rate` downstream, the
/// 10-02 `DeviceGrownTree` contract; the approx update above already applied `lr` on
/// device), the updated resident approx handle, and the recomputed resident `der1` handle
/// for the next iteration.
///
/// # Depth coverage (GPUT-05: depth>1 device-covered)
///
/// Depth>1 is device-covered via the RESIDENT partition-aware fill
/// ([`launch_partition_hist2_resident_into`]) keyed by the resident `leaf_of` + the
/// subtraction trick + the per-active-leaf score ([`score_partition_over_binsums`]), exactly
/// as [`grow_oblivious_tree_into`] (Phase 11 Plan 03 removed the former depth>1 reject).
/// `der_kernel` selects the residual recompute (RMSE `target - approx` / Logloss `target -
/// sigmoid(approx)`). No `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13). Threads
/// ONE `&client`; never reads a 0-len handle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grow_oblivious_tree_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    approx_h: Handle,
    der1_h: &Handle,
    weight_h: &Handle,
    feat_offsets: &[u32],
    feat_shifts: &[u32],
    feat_masks: &[u32],
    cindex_words_h: &Handle,
    offsets_h: &Handle,
    shifts_h: &Handle,
    masks_h: &Handle,
    indices_h: &Handle,
    target_h: &Handle,
    num_words: usize,
    n: usize,
    n_bins: usize,
    n_bins_used: usize,
    n_features: usize,
    depth: usize,
    scaled_l2: f64,
    score_fn: u32,
    learning_rate: f64,
    der_kernel: DerBinaryKernel,
) -> CbResult<(GrownTree, Handle, Handle)> {
    // Empty short-circuit (Pitfall 3/5): no objects / no candidates -> an empty tree, approx
    // unchanged, der1 carried forward.
    if n == 0 || n_features == 0 || n_bins == 0 || depth == 0 {
        return Ok((
            GrownTree {
                splits: Vec::new(),
                leaf_of: vec![0u32; n],
                leaf_values: Vec::new(),
                part_stats: Vec::new(),
            },
            approx_h,
            der1_h.clone(),
        ));
    }

    // MVP scope guard (the partition-aware-histogram forward dependency) — mirrors
    // `grow_oblivious_tree_into`: depth>1 is now DEVICE-COVERED via the partition-aware
    // (fullPass = false) fill keyed by the resident leaf_of + subtraction trick + per-active-leaf
    // score (GPUT-05, Phase 11 Plan 03). The D-05 boundary is preserved — only the O(1) BestSplit
    // per level + the final 2^depth part-stats cross; the histogram / routing stay resident.

    let n_leaves = 1usize
        .checked_shl(depth as u32)
        .ok_or_else(|| CbError::OutOfRange(format!("2^depth overflows usize (depth = {depth})")))?;

    // WR-01 WIRED (perf pass): the D-04 subtraction trick now drives every level > 0 of this
    // scored path. The level-L fill accumulates ONLY the newest-bit-set partitions
    // (`filter_mask = 1 << (L-1)` — roughly half the objects); the sibling slots are then
    // derived device-side as `parent − filled` (bit-exact — integer fixed-point arithmetic),
    // so the per-level atomic-scatter traffic is roughly halved beyond the root level.

    // leaf_of starts all-zero (every object in the root partition 0), resident on device.
    let mut leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));
    let mut splits: Vec<(u32, u32)> = Vec::with_capacity(depth);
    // The previous level's (copy-folded, complete) histogram — the subtraction parent.
    let mut prev_bin_sums: Option<Handle> = None;

    // CB_GPU_PROF stage attribution (cold unless the env var is set): each lap fences the
    // queue so the elapsed wall time is the STAGE's device time, not launch-enqueue time.
    let prof = gpu_prof_enabled();
    let (mut t_fill, mut t_derive, mut t_score, mut t_split, mut t_stats, mut t_tail) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut prof_t = std::time::Instant::now();

    for level in 0..depth {
        // (1) Partition-aware (fullPass = L>0) fill over the RESIDENT session handles (cloned,
        //     NOT re-uploaded — the packed cindex + der1/weight/indices stay resident) keyed by
        //     the resident leaf_of. The bulk histogram stays device-resident; only the O(1)
        //     BestSplit crosses back (D-05). Levels > 0 fill only the newest-bit-set sibling
        //     of each partition pair (the subtraction-trick filter).
        let n_parts = 1usize.checked_shl(level as u32).ok_or_else(|| {
            CbError::OutOfRange(format!("2^level overflows usize (level={level})"))
        })?;
        let filter_mask: u32 = if level == 0 { 0 } else { 1u32 << (level - 1) };
        let bin_sums = launch_partition_hist2_resident_into(
            client,
            der1_h.clone(),
            weight_h.clone(),
            cindex_words_h.clone(),
            offsets_h.clone(),
            shifts_h.clone(),
            masks_h.clone(),
            indices_h.clone(),
            leaf_of_h.clone(),
            num_words,
            n,
            n_bins,
            n_features,
            level as u32,
            filter_mask,
        )?;
        if prof {
            prof_sync(client);
            t_fill += prof_t.elapsed().as_secs_f64();
            prof_t = std::time::Instant::now();
        }

        // (1b) Derive the unfilled sibling slots from the parent level's histogram
        //      (`hist[p] = parent[p] − hist[p + half]`, device-resident, bit-exact).
        if level > 0 {
            let parent = prev_bin_sums.as_ref().ok_or_else(|| {
                CbError::Degenerate(format!(
                    "grow_oblivious_tree_resident level {level}: missing parent histogram \
                     for the subtraction trick (internal invariant)"
                ))
            })?;
            launch_derive_sibling_hist_into(client, parent, &bin_sums, n_parts / 2, n_bins, n_features)?;
        }
        if prof {
            prof_sync(client);
            t_derive += prof_t.elapsed().as_secs_f64();
            prof_t = std::time::Instant::now();
        }

        // (2) Device score + deterministic argmin over the CURRENT 2^level partitions (D-05).
        let best = score_partition_over_binsums(
            client, bin_sums.clone(), n_parts, n_bins, n_bins_used, n_features, scaled_l2, score_fn,
        )?;
        prev_bin_sums = Some(bin_sums);
        if prof {
            // The scorer's own read-back already drained the queue — this lap is pure elapsed.
            t_score += prof_t.elapsed().as_secs_f64();
            prof_t = std::time::Instant::now();
        }

        // (3) The O(1) host integer split decision. No candidate -> degenerate dataset.
        let split = best.ok_or_else(|| {
            CbError::Degenerate(format!(
                "grow_oblivious_tree_resident level {level}: no candidate split (degenerate histogram)"
            ))
        })?;
        splits.push((split.feature_id, split.bin_id));

        // (4) Device partition-split (forward-bit doc-routing) over the resident PACKED
        //     cindex words — IN-PLACE on device, NO read-back (D-05). The split feature's
        //     bin is read through the ONE `read_bin` accessor with its REAL TCFeature
        //     (offset, shift, mask) descriptor — bit-exact vs the former plain
        //     feature-major replica (the `kernels/cindex.rs` pack→read oracle), which is
        //     therefore no longer uploaded at `begin` (round-3 perf).
        let fi = split.feature_id as usize;
        let (split_offset, split_shift, split_mask) =
            match (feat_offsets.get(fi), feat_shifts.get(fi), feat_masks.get(fi)) {
                (Some(&o), Some(&s), Some(&m)) => (o, s, m),
                _ => {
                    return Err(CbError::OutOfRange(format!(
                        "grow_oblivious_tree_resident level {level}: split feature {fi} out of \
                         the {n_features}-feature packed descriptor table"
                    )))
                }
            };
        leaf_of_h = launch_partition_split_packed_into(
            client,
            der1_h.clone(),
            cindex_words_h.clone(),
            indices_h.clone(),
            leaf_of_h,
            n,
            num_words,
            split_offset,
            split_shift,
            split_mask,
            split.bin_id,
            level as u32,
        )?;
        if prof {
            prof_sync(client);
            t_split += prof_t.elapsed().as_secs_f64();
            prof_t = std::time::Instant::now();
        }
    }

    // (4) Device partition-update -> ONE combined read-back of the 2^depth part-stats AND
    //     the final leaf_of routing (the ONLY bulk-data crossing besides the O(1) per-level
    //     BestSplit — D-05; batched into a single sync, see `read_part_stats_and_leaf_of`).
    //     leaf_of is FINAL after the level loop — the launches below never touch it — so
    //     reading it here is byte-identical to the old end-of-fn read. RMSE arm: der2 =
    //     const -1 (the Σ(der2·weight) channel is computed but the calc_average leaf reads
    //     channels 0/1; the Newton/Logloss leaf estimation is the dedicated
    //     `grow_oblivious_tree_newton_into` path, GPUT-07).
    let der2_rmse_h = create_channel_const(client, -1.0, n);
    let part_stats_h = launch_partition_update_into(
        client,
        der1_h.clone(),
        weight_h.clone(),
        der2_rmse_h,
        indices_h.clone(),
        leaf_of_h.clone(),
        n,
        n_leaves,
    )?;
    let (part_stats, leaf_of) =
        read_part_stats_and_leaf_of(client, part_stats_h, leaf_of_h.clone())?;
    if prof {
        // The combined read-back drained the queue — pure elapsed.
        t_stats += prof_t.elapsed().as_secs_f64();
        prof_t = std::time::Instant::now();
    }

    // (5) Host leaf values via the FROZEN cb_compute::calc_average (UNSCALED delta — the
    //     10-02 DeviceGrownTree contract; the approx update below applies learning_rate).
    //     part_stats is stride-3 [Σder1, Σweight, Σ(der2·weight)]; the RMSE arm reads 0/1.
    let mut leaf_values = vec![0.0_f64; n_leaves];
    for leaf in 0..n_leaves {
        let sum = part_stats.get(leaf * 3).copied().unwrap_or(0.0);
        let cnt = part_stats.get(leaf * 3 + 1).copied().unwrap_or(0.0);
        if let Some(slot) = leaf_values.get_mut(leaf) {
            *slot = cb_compute::calc_average(sum, cnt, scaled_l2);
        }
    }

    // (6) Update the resident approx ON DEVICE (`approx[i] += lr * leaf_values[leaf_of[i]]`)
    //     — NO n-length read-back.
    let approx_h = launch_apply_leaf_delta_into(client, approx_h, leaf_of_h, &leaf_values, learning_rate, n)?;

    // (7) Recompute the resident residual der1 for the NEXT tree DEVICE-SIDE from the updated
    //     resident approx — NO approx/der1 read-back (the must-have no-read-back contract).
    let der1_next = launch_der_binary_resident(client, approx_h.clone(), target_h.clone(), der_kernel, n)?;

    // (leaf_of crossed with the part-stats in step 4 — the SC-3 structure observation, the
    // SAME crossing class, one sync instead of two; the per-level routing never crossed.)

    if prof {
        prof_sync(client);
        t_tail += prof_t.elapsed().as_secs_f64();
        eprintln!(
            "CB_GPU_PROF tree n={n} nf={n_features} bins={n_bins} depth={depth} \
             fill={:.2}ms derive={:.2}ms score={:.2}ms split={:.2}ms stats_read={:.2}ms \
             leaf_apply_der={:.2}ms",
            t_fill * 1e3,
            t_derive * 1e3,
            t_score * 1e3,
            t_split * 1e3,
            t_stats * 1e3,
            t_tail * 1e3,
        );
    }

    Ok((
        GrownTree {
            splits,
            leaf_of,
            leaf_values,
            part_stats,
        },
        approx_h,
        der1_next,
    ))
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
/// # Scope (depth>1 device-covered, GPUT-05)
///
/// Plain boosting, `foldCount == 1`, symmetric oblivious, RMSE/L2. Each tree is grown
/// by [`grow_oblivious_tree_into`], which grows a full depth-`depth` tree via the
/// partition-aware per-level score (Phase 11 Plan 03 — the former depth>1 reject is gone).
/// The der recompute between trees
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
/// - [`CbError::OutOfRange`] on `iterations`/leaf bookkeeping / per-level `2^level` slot
///   overflow (propagated from [`grow_oblivious_tree_into`]).
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
