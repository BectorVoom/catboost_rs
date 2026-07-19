//! GPUT-12 (Phase 13 Plan 06, W6): the K-dim **Newton der2 block-leaf solve** on device — the
//! shared multi-output leaf kernel amortized across the whole Phase-13 Plan-07 multi-output family
//! (MultiClass softmax, MultiClassOneVsAll, MultiCrossEntropy/multilabel, MultiRMSE,
//! RMSEWithUncertainty). It runs the packed-hessian Newton solve
//! (`cb_compute::leaf::solve_symmetric_newton`) on device, batched over per-leaf systems, matching
//! the Rust CPU path at ε=1e-4 (D-07).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! A serial `#[cube]` kernel that transcribes the frozen CPU parity oracle INLINE — the kernel body
//! cannot reach `cb_compute`/`cb_core`/`cb_train`, and cb-backend must NEVER gain a `cb-train` dep
//! (the feature-unification landmine, Pattern B). Two CPU references are reproduced:
//!
//! 1. **`cb_compute::leaf::solve_symmetric_newton`** (`leaf.rs:201-260`) — reconstruct the dense
//!    symmetric Hessian `H` from the packed lower-triangular order
//!    `[(0,0),(0,1),…,(0,K-1),(1,1),…]`; `maxTrace = max(scaled_l2, max_d(-H[d][d]))` at **f32
//!    precision** (Pitfall 5); `adjustedL2 = max(scaled_l2, maxTrace · f32::EPSILON)`;
//!    `M = -(H − adjustedL2·I)`; solve `M · x = -sum_der`; `res = -x`. A non-positive pivot yields
//!    the zeros fallback (matching the CPU `None`→zeros — no NaN/panic, T-13-11).
//! 2. **`cb_compute::leaf::cholesky_solve`** (`leaf.rs:282-347`) — the dense SPD `a = L·Lᵀ` /
//!    forward / back substitution, transcribed inline (shares the Plan-02
//!    `crate::kernels::cholesky_solve` numerics).
//!
//! # Coupled vs diagonal dispatch (RESEARCH Pitfall 3, VERIFIED)
//!
//! `mode == 0` COUPLED — the FULL K×K solve, ONLY for `MultiClass` softmax (the off-diagonal
//! `−w·p_k·p_row` hessian couples the classes). `mode == 1` DIAGONAL — a per-component 1×1 Newton
//! solve, for the separable losses (`MultiClassOneVsAll`, `MultiCrossEntropy`/multilabel,
//! `MultiRMSE`, `RMSEWithUncertainty`); `multilogit.cu` emits der2 one row at a time, so the
//! diagonal path reads only the packed diagonal entries `H[d][d]` and solves each component
//! independently (== `solve_symmetric_newton` with `k == 1` per component). The der functions
//! themselves already exist in `cb-compute` — this kernel is ONLY the block leaf solve.
//!
//! # f64-typed seam
//!
//! The whole solve accumulates in f64 (D-07 — the per-matrix arithmetic is NOT an atomic reduction,
//! so gfx1100's missing f64 atomic-add is irrelevant here; only the der/weight SUMS use the
//! fixed-point reduction, upstream of this kernel). WGSL has no f64, so a genuine `wgpu` backend
//! surfaces a typed [`CbError::OutOfRange`] rather than an opaque JIT crash. No `-inf` literal in any
//! `#[cube]` body (Pattern D — the pivot fallback is a finite zero, never a sentinel). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::SelectedRuntime;

/// `f32::EPSILON` (`2^-23`) — the `maxTrace` regularizer epsilon (`hessian.cpp:35-38`,
/// `solve_symmetric_newton` Pitfall 5), transcribed inline (the `#[cube]` body cannot reach
/// `f32::EPSILON` through a path type). The regularization is computed at f32 precision exactly as
/// the CPU, then cast back to f64.
const F32_EPSILON: f32 = 1.192_092_9e-7;

// ===========================================================================
// #[cube] batched serial K-dim Newton der2 block solve
// ===========================================================================

/// Serial (unit 0) batched f64 K-dim Newton der2 block solve over `batch` independent per-leaf
/// systems, each of dimension `k = dims[0]`. Transcribes `solve_symmetric_newton` + `cholesky_solve`
/// inline.
///
/// - `ders`: `batch · k` per-leaf `sum_der` (the RHS source; `neg_der = -sum_der`).
/// - `der2_packed`: `batch · P` packed lower-triangular hessians, `P = k·(k+1)/2` per leaf, in the
///   order `[(0,0),(0,1),…,(0,k-1),(1,1),…]` (the SAME packing `softmax_ders` uses). The DIAGONAL
///   mode reads only the packed diagonal entries.
/// - `params`: `[scaled_l2]`.
/// - `dims`: `[k, batch, mode]` — `mode == 0` COUPLED (full k×k solve, MultiClass softmax);
///   `mode == 1` DIAGONAL (per-component 1×1 solve, the separable losses).
/// - `out`: `batch · k` — the per-leaf block `res = -x` (the K-vector leaf delta).
/// - `scratch`: reused per-system workspace of length `2·k·k + 3·k` — `H` (`k·k`), `L` (`k·k`),
///   `b`/`neg_der` (`k`), `y` (`k`), `x` (`k`). Serial reuse across the batch (no
///   cross-contamination: `H`/`L` are rebuilt/re-zeroed each system).
///
/// A non-positive pivot (or a zero diagonal in substitution) sets the per-system fallback so the
/// emitted block is all-zeros — matching the CPU `cholesky_solve` → `None` → `vec![0.0; k]` path (no
/// NaN, T-13-11). No `-inf` literal; no host reach.
#[cube(launch)]
fn multi_newton_solve_kernel(
    ders: &Array<f64>,
    der2_packed: &Array<f64>,
    params: &Array<f64>,
    dims: &Array<u32>,
    out: &mut Array<f64>,
    scratch: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let k = dims[0] as usize;
        let batch = dims[1] as usize;
        let mode = dims[2];
        let scaled_l2 = params[0];
        let scaled_l2_f32 = f32::cast_from(scaled_l2);
        let eps = F32_EPSILON;

        // Packed lower-triangular length P = k·(k+1)/2 (k·(k+1) is even so /2 is exact).
        let packed = k * (k + 1usize) / 2usize;

        // Scratch region offsets (per-system; reused serially across the batch).
        let h_off = 0usize;
        let l_off = k * k;
        let b_off = 2usize * k * k;
        let y_off = 2usize * k * k + k;
        let x_off = 2usize * k * k + 2usize * k;

        let mut sysid = 0usize;
        while sysid < batch {
            let der_base = sysid * k;
            let pack_base = sysid * packed;

            if mode == 1u32 {
                // ---- DIAGONAL: per-component 1×1 Newton solve (separable losses). ----
                let mut d = 0usize;
                while d < k {
                    // Diagonal packed index: (d,d) in [(0,0),(0,1),…] is d·k − d·(d−1)/2.
                    let diag_idx = d * k - d * (d - 1usize) / 2usize;
                    let h_dd = der2_packed[pack_base + diag_idx];
                    // maxTrace at f32 (start scaled_l2, max with −H[d][d]).
                    let mut max_trace = scaled_l2_f32;
                    let neg_diag = f32::cast_from(-h_dd);
                    if neg_diag > max_trace {
                        max_trace = neg_diag;
                    }
                    let mut adj_f32 = scaled_l2_f32;
                    let te = max_trace * eps;
                    if te > adj_f32 {
                        adj_f32 = te;
                    }
                    let adjusted_l2 = f64::cast_from(adj_f32);
                    // M = -(H − adjustedL2): m_dd = -(h_dd - adjusted_l2).
                    let m_dd = -(h_dd - adjusted_l2);
                    let neg_der = -ders[der_base + d];
                    // Solve m_dd · x = neg_der; res = -x. Non-positive pivot → 0.
                    let mut res = 0.0_f64;
                    if m_dd > 0.0_f64 {
                        let x = neg_der / m_dd;
                        res = -x;
                    }
                    out[der_base + d] = res;
                    d += 1usize;
                }
            } else {
                // ---- COUPLED: full k×k solve (MultiClass softmax). ----
                // 1. Reconstruct dense symmetric H from the packed lower-triangular order.
                let mut idx = 0usize;
                let mut i = 0usize;
                while i < k {
                    let mut j = i;
                    while j < k {
                        let v = der2_packed[pack_base + idx];
                        idx += 1usize;
                        scratch[h_off + i * k + j] = v;
                        scratch[h_off + j * k + i] = v;
                        j += 1usize;
                    }
                    i += 1usize;
                }

                // 2. maxTrace at f32 precision (scaled_l2, then max with each −H[d][d]).
                let mut max_trace = scaled_l2_f32;
                let mut dd = 0usize;
                while dd < k {
                    let neg_diag = f32::cast_from(-scratch[h_off + dd * k + dd]);
                    if neg_diag > max_trace {
                        max_trace = neg_diag;
                    }
                    dd += 1usize;
                }
                let mut adj_f32 = scaled_l2_f32;
                let te = max_trace * eps;
                if te > adj_f32 {
                    adj_f32 = te;
                }
                let adjusted_l2 = f64::cast_from(adj_f32);

                // 3. M = -(H − adjustedL2·I): subtract from the diagonal, negate the whole matrix.
                //    Build M directly into the L-scratch working matrix (H stays as read model for
                //    the score path is not needed here). We overwrite H in place as M.
                let mut da = 0usize;
                while da < k {
                    scratch[h_off + da * k + da] -= adjusted_l2;
                    da += 1usize;
                }
                let mut ni = 0usize;
                let kk = k * k;
                while ni < kk {
                    scratch[h_off + ni] = -scratch[h_off + ni];
                    ni += 1usize;
                }

                // neg_der = -sum_der (the RHS b).
                let mut bi = 0usize;
                while bi < k {
                    scratch[b_off + bi] = -ders[der_base + bi];
                    bi += 1usize;
                }

                // 4. Cholesky decompose M = L·Lᵀ (lower). Non-positive pivot → zeros fallback.
                let mut ok = true;
                // Zero L.
                let mut z = 0usize;
                while z < kk {
                    scratch[l_off + z] = 0.0_f64;
                    z += 1usize;
                }
                let mut ci = 0usize;
                while ci < k {
                    let mut cj = 0usize;
                    while cj <= ci {
                        let mut s = scratch[h_off + ci * k + cj];
                        let mut p = 0usize;
                        while p < cj {
                            s -= scratch[l_off + ci * k + p] * scratch[l_off + cj * k + p];
                            p += 1usize;
                        }
                        if ci == cj {
                            if s <= 0.0_f64 {
                                ok = false;
                            }
                            let mut diag = 0.0_f64;
                            if ok {
                                diag = s.sqrt();
                            }
                            scratch[l_off + ci * k + cj] = diag;
                        } else {
                            let ljj = scratch[l_off + cj * k + cj];
                            let mut v = 0.0_f64;
                            if ljj != 0.0_f64 {
                                v = s / ljj;
                            } else {
                                ok = false;
                            }
                            scratch[l_off + ci * k + cj] = v;
                        }
                        cj += 1usize;
                    }
                    ci += 1usize;
                }

                // Forward solve L·y = b.
                if ok {
                    let mut fi = 0usize;
                    while fi < k {
                        let mut s = scratch[b_off + fi];
                        let mut p = 0usize;
                        while p < fi {
                            s -= scratch[l_off + fi * k + p] * scratch[y_off + p];
                            p += 1usize;
                        }
                        let lii = scratch[l_off + fi * k + fi];
                        if lii != 0.0_f64 {
                            scratch[y_off + fi] = s / lii;
                        } else {
                            ok = false;
                        }
                        fi += 1usize;
                    }
                }

                // Back solve Lᵀ·x = y (i descending).
                if ok {
                    let mut b2 = k;
                    while b2 > 0usize {
                        b2 -= 1usize;
                        let mut s = scratch[y_off + b2];
                        let mut p = b2 + 1usize;
                        while p < k {
                            s -= scratch[l_off + p * k + b2] * scratch[x_off + p];
                            p += 1usize;
                        }
                        let lii = scratch[l_off + b2 * k + b2];
                        if lii != 0.0_f64 {
                            scratch[x_off + b2] = s / lii;
                        } else {
                            ok = false;
                        }
                    }
                }

                // 5. res = -x (all-zeros on a failed pivot).
                let mut ri = 0usize;
                while ri < k {
                    let mut v = 0.0_f64;
                    if ok {
                        v = -scratch[x_off + ri];
                    }
                    out[der_base + ri] = v;
                    ri += 1usize;
                }
            }

            sysid += 1usize;
        }
    }
}

// ===========================================================================
// Host launch wrappers (device-resident Handle + readback oracle wrapper)
// ===========================================================================

/// Reject the (impossible) wgpu f64 path with a typed error, mirroring
/// [`crate::kernels::cholesky_solve`]. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device Newton der2 block solve requires f64 device channels; the wgpu backend has none \
         (WGSL has no f64). Use the rocm/cuda/cpu backend for the multi-output leaf solve."
            .to_owned(),
    )
}

/// The scratch length `2·k·k + 3·k` the kernel needs per system, overflow-checked.
fn scratch_len(k: usize) -> CbResult<usize> {
    let kk = k
        .checked_mul(k)
        .ok_or_else(|| CbError::OutOfRange(format!("Newton scratch k·k overflows (k = {k})")))?;
    let two_kk = kk
        .checked_mul(2)
        .ok_or_else(|| CbError::OutOfRange(format!("Newton scratch 2·k·k overflows (k = {k})")))?;
    let three_k = k
        .checked_mul(3)
        .ok_or_else(|| CbError::OutOfRange(format!("Newton scratch 3·k overflows (k = {k})")))?;
    two_kk
        .checked_add(three_k)
        .ok_or_else(|| CbError::OutOfRange(format!("Newton scratch length overflows (k = {k})")))
}

/// The packed lower-triangular hessian length `k·(k+1)/2`, overflow-checked.
fn packed_len(k: usize) -> CbResult<usize> {
    k.checked_mul(k + 1)
        .map(|p| p / 2)
        .ok_or_else(|| CbError::OutOfRange(format!("packed hessian length overflows (k = {k})")))
}

/// Launch the batched K-dim Newton der2 block solve over `batch` resident systems of dimension `k`,
/// returning the resident output HANDLE WITHOUT reading it back (D-05 residency). `ders_h` is
/// `batch·k` `sum_der`; `der2_packed_h` is `batch·(k·(k+1)/2)` packed lower-triangular hessians.
/// `coupled == true` runs the full k×k softmax solve; `coupled == false` runs the per-component
/// diagonal solve. `client` owns every handle (residency, Pitfall 3). `k == 0` or `batch == 0`
/// yields an empty handle (no launch, no 0-len read).
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_multi_newton_solve(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    ders_h: &Handle,
    der2_packed_h: &Handle,
    k: usize,
    batch: usize,
    scaled_l2: f64,
    coupled: bool,
) -> CbResult<Handle> {
    if k == 0 || batch == 0 {
        return Ok(client.empty(0));
    }

    #[cfg(feature = "wgpu")]
    {
        return Err(wgpu_reject());
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let der_len = batch.checked_mul(k).ok_or_else(|| {
            CbError::OutOfRange(format!("ders length overflows (k = {k}, batch = {batch})"))
        })?;
        let packed = packed_len(k)?;
        let pack_total = batch.checked_mul(packed).ok_or_else(|| {
            CbError::OutOfRange(format!("packed length overflows (k = {k}, batch = {batch})"))
        })?;
        let scr_len = scratch_len(k)?;

        let mode: u32 = if coupled { 0 } else { 1 };
        let k_u32 =
            u32::try_from(k).map_err(|_| CbError::OutOfRange(format!("k ({k}) exceeds u32")))?;
        let batch_u32 = u32::try_from(batch)
            .map_err(|_| CbError::OutOfRange(format!("batch ({batch}) exceeds u32")))?;

        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![scaled_l2]));
        let dims_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![k_u32, batch_u32, mode]));
        let out = client.empty(der_len * std::mem::size_of::<f64>());
        let scratch = client.empty(scr_len * std::mem::size_of::<f64>());

        // Serial single-thread launch (unit 0 loops the batch); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        multi_newton_solve_kernel::launch::<SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(ders_h.clone(), der_len) },
            unsafe { ArrayArg::from_raw_parts(der2_packed_h.clone(), pack_total) },
            unsafe { ArrayArg::from_raw_parts(params_h, 1) },
            unsafe { ArrayArg::from_raw_parts(dims_h, 3) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), der_len) },
            unsafe { ArrayArg::from_raw_parts(scratch, scr_len) },
        );
        Ok(out)
    }
}

/// Read a resident f64 handle back to a host `Vec<f64>` (the self-oracle seam; NOT the residency
/// path). A read-back failure surfaces [`CbError::Degenerate`], never a silent zero buffer.
#[cfg(not(feature = "wgpu"))]
fn read_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("Newton block solve read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Host-readback wrapper: solve ONE K-dim Newton der2 block on device and read back the `k`-vector
/// leaf delta — the device analog of `cb_compute::leaf::solve_symmetric_newton`. `sum_der` is the
/// per-dimension der sum (length `k`); `sum_der2_packed` is the packed lower-triangular hessian
/// (length `k·(k+1)/2`). `coupled == true` runs the full softmax solve; `coupled == false` runs the
/// per-component diagonal solve. A `k <= 0` system returns an empty `Vec` WITHOUT a device launch.
#[allow(dead_code)] // consumed by the #[cfg(test)] multi_newton_test self-oracle (source/test separation)
pub(crate) fn solve_multi_newton_host(
    sum_der: &[f64],
    sum_der2_packed: &[f64],
    scaled_l2: f64,
    coupled: bool,
) -> CbResult<Vec<f64>> {
    let k = sum_der.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    let expected = packed_len(k)?;
    if sum_der2_packed.len() != expected {
        return Err(CbError::LengthMismatch {
            column: "sum_der2_packed".to_owned(),
            expected,
            actual: sum_der2_packed.len(),
        });
    }
    #[cfg(feature = "wgpu")]
    {
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
        let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
        let ders_h = client.create(cubecl::bytes::Bytes::from_elems(sum_der.to_vec()));
        let packed_h = client.create(cubecl::bytes::Bytes::from_elems(sum_der2_packed.to_vec()));
        let handle =
            launch_multi_newton_solve(&client, &ders_h, &packed_h, k, 1, scaled_l2, coupled)?;
        read_f64(&client, handle)
    }
}
