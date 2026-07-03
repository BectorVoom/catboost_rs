//! IN-03: Phase 7.2 device-resident der1/der2 seam, mechanically relocated out of the
//! oversized `gpu_runtime.rs` with ZERO logic changes. `DerBinaryKernel` /
//! `DerUnaryKernel` / `DerParamKernel` and the `launch_der_*` helpers are re-exported
//! from `gpu_runtime` (`pub use der_seams::*`) so every existing `crate::gpu_runtime::X`
//! path still resolves. `use super::*` brings in the shared parent items (`CUBE_DIM`,
//! `launch_block_reduce_f64`, the cubecl/cb-core imports) this seam consumes.
#![allow(unused_imports)]
use super::*;

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
// Under wgpu the body early-returns the WR-02 typed reject before touching the launch
// args, so they read as unused and the f64 launch below is dead — on that cfg only (the
// args ARE used and the launch IS reached on every other backend).
#[cfg_attr(feature = "wgpu", allow(unused_variables, unreachable_code))]
// IN-03: `pub(crate)` (was private) so the relocated der seam stays reachable from the
// `gpu_runtime` parent (the boosting pass calls it) after the module split. Still
// crate-internal — no external API change.
pub(crate) fn launch_der_binary_into(
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

    // WR-02: the whole der seam (handle layout, the `out_handle = empty(n *
    // size_of::<f64>())` below, the `cast_slice::<u8, f64>` read-backs, and the
    // `grow_boosting_pass_into` f64 read-back) is f64-typed. WGSL has no `f64` type, so a
    // genuine wgpu backend cannot JIT this `launch::<f64, _>`. Reject it with a typed
    // error here (the reviewer-sanctioned typed-reject alternative) rather than letting
    // it surface as an opaque JIT crash. The in-env rocm/cuda/cpu f64 path is unaffected.
    // (The wgpu-only dead f64 launch below is allowed at the fn-level cfg_attr.)
    #[cfg(feature = "wgpu")]
    {
        return Err(CbError::OutOfRange(
            "binary der seam requires an f64 device channel; the wgpu backend has no f64 \
             type (WR-02). Use the rocm/cuda/cpu backend for derivative computation."
                .to_owned(),
        ));
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

/// GPUT-03 resident der seam: recompute der1 from PRE-UPLOADED, device-resident
/// `approx_h`/`target_h` handles and return the der1 output Handle WITHOUT reading either
/// input OR the output back to host (no n-length crossing). This is the residency variant
/// of [`launch_der_binary_into`]: the boosting loop keeps `approx` resident (updated in
/// place by `apply_leaf_delta`) and chains `der1_h = der(approx_h, target_h)` into the
/// next tree entirely on-device (the must-have no-read-back contract). The input handles
/// are read-only (`&Array`), so the caller passes `.clone()`s and keeps its own resident
/// copies (a CubeCL Handle clone shares the device buffer, it does NOT copy).
///
/// `n` is the object count (both inputs are length `n`). Empty (`n == 0`) returns a
/// zero-length handle with NO launch (Pitfall 5). The whole der seam is f64-typed (WGSL
/// has no f64 type), so a genuine wgpu backend surfaces a typed [`CbError::OutOfRange`]
/// rather than an opaque JIT crash (WR-02) — the in-env rocm/cuda/cpu f64 path is
/// unaffected. No `unwrap`/`expect`/`panic`/indexing (workspace lints + D-13).
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_der_binary_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    approx_h: Handle,
    target_h: Handle,
    kernel: DerBinaryKernel,
    n: usize,
) -> CbResult<Handle> {
    if n == 0 {
        return Ok(client.empty(0));
    }

    #[cfg(feature = "wgpu")]
    {
        return Err(CbError::OutOfRange(
            "resident binary der seam requires an f64 device channel; the wgpu backend has \
             no f64 type (WR-02). Use the rocm/cuda/cpu backend for derivative computation."
                .to_owned(),
        ));
    }

    #[cfg(not(feature = "wgpu"))]
    {
        // The der output is per-element (length `n`), NOT one slot per cube.
        let out_handle = client.empty(n * std::mem::size_of::<f64>());
        let num_cubes = n.div_ceil(CUBE_DIM).max(1);
        let count = CubeCount::Static(num_cubes as u32, 1, 1);
        let dim = CubeDim {
            x: CUBE_DIM as u32,
            y: 1,
            z: 1,
        };
        // `from_raw_parts` consumes each input handle; clone the output so the original
        // stays returnable on-device. NO `read_one` here (SC-3). Each match arm consumes
        // approx_h/target_h — mutually exclusive at runtime, so the moves do not conflict.
        match kernel {
            DerBinaryKernel::RmseGradient => gradient_kernel::launch::<f64, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_h, n) },
                unsafe { ArrayArg::from_raw_parts(target_h, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            ),
            DerBinaryKernel::LoglossGradient => logloss_gradient_kernel::launch::<f64, SelectedRuntime>(
                client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(approx_h, n) },
                unsafe { ArrayArg::from_raw_parts(target_h, n) },
                unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            ),
        }
        Ok(out_handle)
    }
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
// See `launch_der_binary_into`: wgpu early-returns the WR-02 typed reject, so the launch
// args read as unused and the f64 launch below is dead on that cfg only.
#[cfg_attr(feature = "wgpu", allow(unused_variables, unreachable_code))]
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

    // WR-02: f64-typed seam (see `launch_der_binary_into`); WGSL has no f64 type, so a
    // genuine wgpu backend cannot JIT this. Typed-reject rather than an opaque JIT crash.
    #[cfg(feature = "wgpu")]
    {
        return Err(CbError::OutOfRange(
            "unary der seam requires an f64 device channel; the wgpu backend has no f64 \
             type (WR-02). Use the rocm/cuda/cpu backend for derivative computation."
                .to_owned(),
        ));
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
// See `launch_der_binary_into`: wgpu early-returns the WR-02 typed reject, so the launch
// args read as unused and the f64 launch below is dead on that cfg only.
#[cfg_attr(feature = "wgpu", allow(unused_variables, unreachable_code))]
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

    // WR-02: f64-typed seam (see `launch_der_binary_into`); WGSL has no f64 type, so a
    // genuine wgpu backend cannot JIT this. Typed-reject rather than an opaque JIT crash.
    #[cfg(feature = "wgpu")]
    {
        return Err(CbError::OutOfRange(
            "parametric der seam requires an f64 device channel; the wgpu backend has no \
             f64 type (WR-02). Use the rocm/cuda/cpu backend for derivative computation."
                .to_owned(),
        ));
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
