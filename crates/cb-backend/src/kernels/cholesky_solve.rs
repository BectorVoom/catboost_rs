//! GPUT-21 (Phase 13 Plan 02, Wave B): the batched f64 SPD **Cholesky solver** on device — the one
//! genuinely-new `#[cube]` kernel of the phase and the item Phase 7.5 deliberately deferred to the
//! host (the documented "RESEARCH Open Q3" at `score_split.rs:852` / `pairwise.rs:993`). It runs
//! decomposition + forward/back substitution + ridge regularization + score-from-decomposition on
//! device, batched over per-leaf systems, matching the **Rust CPU path** at ε=1e-4 (D-07).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! A serial `#[cube]` kernel that transcribes the frozen CPU parity oracles INLINE — the kernel
//! body cannot reach `cb_compute`/`cb_core`/`cb_train`, and cb-backend must NEVER gain a `cb-train`
//! dep (the feature-unification landmine, Pattern B). Two CPU references are reproduced:
//!
//! 1. **`cb_compute::leaf::cholesky_solve`** (`leaf.rs:282-347`) — the dense SPD solve `a = L·Lᵀ`,
//!    forward `L·y = b`, back `Lᵀ·x = y`; a **non-positive pivot** yields the zeros fallback
//!    (matching the CPU `None`→zeros — no NaN/panic, T-13-03).
//! 2. **`cb_train::pairwise_leaves::calculate_pairwise_leaf_values`** (`pairwise_leaves.rs:113-195`)
//!    — the ridge constants (`cell_prior = 1/system_size`,
//!    `non_diag_reg = -prior·cell_prior`, `diag_reg = prior·(1 - cell_prior) + l2`), the
//!    `system_size == 2` closed form, the general `(n-1)×(n-1)` build (drop the last row: leaf
//!    gauge freedom), `res.push(0.0)`, and `make_zero_average` (mean over the leaf deltas). These
//!    are the EXACT CPU-oracle constants, NOT upstream `linear_solver.cu::RegularizeImpl`'s
//!    bump-heuristics (RESEARCH Pitfall 2).
//!
//! The optional `Score` mode adds the `CalcScoresCholesky` path
//! (`cb_compute::pairwise_scoring::calculate_score`, `pairwise_scoring.cpp:51-81`):
//! `score = Σ_x avrg[x]·(sumDer[x] − ½·Σ_y avrg[y]·weightSum[x][y])` over the SAME zero-averaged
//! solution — so ONE parameterized kernel serves both the leaf-value system (size `leaf_count`) and
//! the split-score system (size `2·PartCount`), per RESEARCH Open Q1.
//!
//! # f64-typed seam
//!
//! The whole solve accumulates in f64 (D-07 — f64 holds ε=1e-4 across hundreds of trees; the solve
//! is NOT an atomic reduction, so gfx1100's missing f64 atomic-add is irrelevant here). WGSL has no
//! f64, so a genuine `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather than an opaque
//! JIT crash. No `-inf` literal in any `#[cube]` body (Pattern D — the pivot fallback is a finite
//! zero, never a sentinel). No `unwrap`/`expect`/`panic`/indexing in production (workspace lints).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::SelectedRuntime;

// ===========================================================================
// #[cube] batched serial Cholesky solver
// ===========================================================================

/// Serial (unit 0) batched f64 Cholesky solver over `batch` independent per-leaf SPD systems, each
/// of size `n = dims[0]`. Transcribes `calculate_pairwise_leaf_values` + `cholesky_solve` inline.
///
/// - `matrices`: `batch · n · n` row-major weight-sum matrices (NO ridge; ridge is applied on the
///   fly to the reduced `(n-1)×(n-1)` system). In `Score` mode the FULL `n×n` matrix is also read
///   back for `calculate_score`.
/// - `ders`: `batch · n` per-leaf der sums (the RHS).
/// - `reg_params`: `[l2_diag_reg, pairwise_bucket_weight_prior_reg]`.
/// - `dims`: `[n, batch, mode]` — `mode == 0` emits the `n` zero-averaged leaf deltas per system;
///   `mode == 1` emits one `calculate_score` per system.
/// - `out`: `Leaf` mode `batch · n`; `Score` mode `batch`.
/// - `scratch`: reused per-system workspace of length `m·m + 2·m + n` (`m = n - 1`) — `L` (`m·m`),
///   `y` (`m`), `x` (`m`), `res` (`n`). Serial reuse across the batch (no cross-contamination:
///   `L`/`res` are re-zeroed each system).
///
/// A non-positive pivot (or a zero diagonal in substitution) sets the per-system fallback so `res`
/// is all-zeros — matching the CPU `cholesky_solve` → `None` → `vec![0.0; m]` path (no NaN). No
/// `-inf` literal; no host reach.
#[cube(launch)]
fn cholesky_solve_kernel(
    matrices: &Array<f64>,
    ders: &Array<f64>,
    reg_params: &Array<f64>,
    dims: &Array<u32>,
    out: &mut Array<f64>,
    scratch: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let n = dims[0] as usize;
        let batch = dims[1] as usize;
        let mode = dims[2];
        let l2 = reg_params[0];
        let prior = reg_params[1];

        // Reduced system rank m = n - 1 (drop the last row: leaf gauge freedom). n <= 1 → m = 0.
        let mut m = 0usize;
        if n > 0 {
            m = n - 1;
        }

        // Ridge constants (calculate_pairwise_leaf_values:123-125), computed ONCE per launch.
        let n_f = f64::cast_from(u64::cast_from(n));
        let mut cell_prior = 0.0_f64;
        if n > 0 {
            cell_prior = 1.0_f64 / n_f;
        }
        let non_diag_reg = -prior * cell_prior;
        let diag_reg = prior * (1.0_f64 - cell_prior) + l2;

        // Scratch region offsets (per-system; reused serially across the batch).
        let l_off = 0usize;
        let y_off = m * m;
        let x_off = m * m + m;
        let res_off = m * m + m + m;

        let mut b = 0usize;
        while b < batch {
            let mat_base = b * n * n;
            let der_base = b * n;

            // --- res <- 0 (length n). n == 1 keeps the lone delta 0; n == 0 is a no-op. ---
            let mut t = 0usize;
            while t < n {
                scratch[res_off + t] = 0.0_f64;
                t += 1usize;
            }

            if n == 2usize {
                // 2×2 closed form (pairwise_leaves_calculation.cpp:25-34):
                //   res = { derSums[0] / (weightSums[0][0] + diagReg), 0 }.
                let a11 = matrices[mat_base];
                let denom = a11 + diag_reg;
                let mut x0 = 0.0_f64;
                if denom != 0.0_f64 {
                    x0 = ders[der_base] / denom;
                }
                scratch[res_off] = x0;
                scratch[res_off + 1usize] = 0.0_f64;
            } else if n > 2usize {
                // General case: Cholesky-decompose the reduced m×m SPD matrix into L (scratch).
                let mut ok = true;

                // Zero L.
                let mut z = 0usize;
                let ll = m * m;
                while z < ll {
                    scratch[l_off + z] = 0.0_f64;
                    z += 1usize;
                }

                // Decomposition: for i in 0..m, j in 0..=i (lower triangle), a = L·Lᵀ.
                let mut i = 0usize;
                while i < m {
                    let mut j = 0usize;
                    while j <= i {
                        // M[i][j] = weightSums[i][j] + reg (diag vs off-diag prior).
                        let mut s = matrices[mat_base + i * n + j];
                        if i == j {
                            s += diag_reg;
                        } else {
                            s += non_diag_reg;
                        }
                        let mut p = 0usize;
                        while p < j {
                            s -= scratch[l_off + i * m + p] * scratch[l_off + j * m + p];
                            p += 1usize;
                        }
                        if i == j {
                            // Non-positive pivot → fall back to zeros (CPU `None`; T-13-03).
                            if s <= 0.0_f64 {
                                ok = false;
                            }
                            let mut d = 0.0_f64;
                            if ok {
                                d = s.sqrt();
                            }
                            scratch[l_off + i * m + j] = d;
                        } else {
                            let ljj = scratch[l_off + j * m + j];
                            let mut v = 0.0_f64;
                            if ljj != 0.0_f64 {
                                v = s / ljj;
                            } else {
                                ok = false;
                            }
                            scratch[l_off + i * m + j] = v;
                        }
                        j += 1usize;
                    }
                    i += 1usize;
                }

                // Forward solve L·y = b (b = ders[0..m]).
                if ok {
                    let mut fi = 0usize;
                    while fi < m {
                        let mut s = ders[der_base + fi];
                        let mut p = 0usize;
                        while p < fi {
                            s -= scratch[l_off + fi * m + p] * scratch[y_off + p];
                            p += 1usize;
                        }
                        let lii = scratch[l_off + fi * m + fi];
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
                    let mut bi = m;
                    while bi > 0usize {
                        bi -= 1usize;
                        let mut s = scratch[y_off + bi];
                        let mut p = bi + 1usize;
                        while p < m {
                            s -= scratch[l_off + p * m + bi] * scratch[x_off + p];
                            p += 1usize;
                        }
                        let lii = scratch[l_off + bi * m + bi];
                        if lii != 0.0_f64 {
                            scratch[x_off + bi] = s / lii;
                        } else {
                            ok = false;
                        }
                    }
                }

                // res[0..m] = x (all-zeros on a failed pivot), res[m] = 0 (push 0).
                let mut ri = 0usize;
                while ri < m {
                    let mut v = 0.0_f64;
                    if ok {
                        v = scratch[x_off + ri];
                    }
                    scratch[res_off + ri] = v;
                    ri += 1usize;
                }
                scratch[res_off + m] = 0.0_f64;
            }

            // --- make_zero_average over res[0..n] (mean-subtract; matrix.h:5-15). ---
            if n > 0usize {
                let mut ssum = 0.0_f64;
                let mut k = 0usize;
                while k < n {
                    ssum += scratch[res_off + k];
                    k += 1usize;
                }
                let avg = ssum / n_f;
                let mut k2 = 0usize;
                while k2 < n {
                    scratch[res_off + k2] -= avg;
                    k2 += 1usize;
                }
            }

            // --- Emit: leaf deltas, or the CalcScoresCholesky score. ---
            if mode == 0u32 {
                let mut e = 0usize;
                while e < n {
                    out[der_base + e] = scratch[res_off + e];
                    e += 1usize;
                }
            } else {
                // score = Σ_x avrg[x]·(sumDer[x] − ½·Σ_y avrg[y]·weightSum[x][y]) (no ridge).
                let mut outer = 0.0_f64;
                let mut sx = 0usize;
                while sx < n {
                    let avrg_x = scratch[res_off + sx];
                    let der_x = ders[der_base + sx];
                    let mut sub = 0.0_f64;
                    let mut sy = 0usize;
                    while sy < n {
                        sub += scratch[res_off + sy] * matrices[mat_base + sx * n + sy];
                        sy += 1usize;
                    }
                    outer += avrg_x * (der_x - 0.5_f64 * sub);
                    sx += 1usize;
                }
                out[b] = outer;
            }

            b += 1usize;
        }
    }
}

// ===========================================================================
// Host launch wrappers (device-resident Handle + readback oracle wrappers)
// ===========================================================================

/// Reject the (impossible) wgpu f64 path with a typed error, mirroring
/// [`crate::kernels::mvs_device`]. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device Cholesky solve requires f64 device channels; the wgpu backend has none (WGSL has \
         no f64). Use the rocm/cuda/cpu backend for the pairwise Cholesky path."
            .to_owned(),
    )
}

/// The scratch length `m·m + 2·m + n` (`m = n - 1`) the kernel needs per system, overflow-checked.
fn scratch_len(n: usize) -> CbResult<usize> {
    let m = n.saturating_sub(1);
    let mm = m.checked_mul(m).ok_or_else(|| {
        CbError::OutOfRange(format!("Cholesky scratch m·m overflows (m = {m})"))
    })?;
    let two_m = m.checked_mul(2).ok_or_else(|| {
        CbError::OutOfRange(format!("Cholesky scratch 2·m overflows (m = {m})"))
    })?;
    mm.checked_add(two_m)
        .and_then(|v| v.checked_add(n))
        .ok_or_else(|| CbError::OutOfRange(format!("Cholesky scratch length overflows (n = {n})")))
}

/// Launch the batched Cholesky solver over `batch` resident systems of size `n`, returning the
/// resident output HANDLE WITHOUT reading it back (D-05 residency). `matrices_h` is `batch·n·n`
/// row-major weight matrices (no ridge), `ders_h` is `batch·n` der sums. `score_mode == false`
/// emits `batch·n` zero-averaged leaf deltas; `score_mode == true` emits `batch` scores. `client`
/// owns every handle (residency, Pitfall 3). `n == 0` or `batch == 0` yields an empty handle (no
/// launch, no 0-len read).
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_cholesky_solve(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    matrices_h: &Handle,
    ders_h: &Handle,
    n: usize,
    batch: usize,
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    score_mode: bool,
) -> CbResult<Handle> {
    if n == 0 || batch == 0 {
        return Ok(client.empty(0));
    }

    #[cfg(feature = "wgpu")]
    {
        return Err(wgpu_reject());
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let mat_len = batch
            .checked_mul(n)
            .and_then(|v| v.checked_mul(n))
            .ok_or_else(|| {
                CbError::OutOfRange(format!("matrices length overflows (n = {n}, batch = {batch})"))
            })?;
        let der_len = batch.checked_mul(n).ok_or_else(|| {
            CbError::OutOfRange(format!("ders length overflows (n = {n}, batch = {batch})"))
        })?;
        let out_len = if score_mode { batch } else { der_len };
        let scr_len = scratch_len(n)?;

        let mode: u32 = if score_mode { 1 } else { 0 };
        let n_u32 = u32::try_from(n)
            .map_err(|_| CbError::OutOfRange(format!("n ({n}) exceeds u32")))?;
        let batch_u32 = u32::try_from(batch)
            .map_err(|_| CbError::OutOfRange(format!("batch ({batch}) exceeds u32")))?;

        let reg_h = client.create(cubecl::bytes::Bytes::from_elems(vec![
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
        ]));
        let dims_h =
            client.create(cubecl::bytes::Bytes::from_elems(vec![n_u32, batch_u32, mode]));
        let out = client.empty(out_len * std::mem::size_of::<f64>());
        let scratch = client.empty(scr_len * std::mem::size_of::<f64>());

        // Serial single-thread launch (unit 0 loops the batch); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        cholesky_solve_kernel::launch::<SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(matrices_h.clone(), mat_len) },
            unsafe { ArrayArg::from_raw_parts(ders_h.clone(), der_len) },
            unsafe { ArrayArg::from_raw_parts(reg_h, 2) },
            unsafe { ArrayArg::from_raw_parts(dims_h, 3) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), out_len) },
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
        .map_err(|e| CbError::Degenerate(format!("Cholesky solve read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Host-readback wrapper: solve ONE pairwise leaf-value system on device and read back the
/// `leaf_count` zero-averaged leaf deltas — the device analog of
/// `cb_train::pairwise_leaves::calculate_pairwise_leaf_values`. `weight_sums` is the row-major
/// `leaf_count × leaf_count` per-leaf pairwise weight-sum matrix (no ridge), `der_sums` the
/// per-leaf der sums. A `leaf_count <= 1` system returns `vec![0.0; leaf_count]` WITHOUT a device
/// launch (its lone zero-averaged delta is 0; matches the CPU singleton/empty guard).
#[allow(dead_code)] // consumed by the #[cfg(test)] cholesky_solve_test self-oracle (source/test separation)
pub(crate) fn solve_pairwise_leaf_values_host(
    weight_sums: &[f64],
    der_sums: &[f64],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> CbResult<Vec<f64>> {
    let n = der_sums.len();
    if n <= 1 {
        return Ok(vec![0.0; n]);
    }
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CbError::OutOfRange(format!("leaf_count ({n}) squared overflows")))?;
    if weight_sums.len() != expected {
        return Err(CbError::LengthMismatch {
            column: "weight_sums".to_owned(),
            expected,
            actual: weight_sums.len(),
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
        let mat_h = client.create(cubecl::bytes::Bytes::from_elems(weight_sums.to_vec()));
        let der_h = client.create(cubecl::bytes::Bytes::from_elems(der_sums.to_vec()));
        let handle = launch_cholesky_solve(
            &client,
            &mat_h,
            &der_h,
            n,
            1,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
            false,
        )?;
        read_f64(&client, handle)
    }
}

/// Host-readback wrapper: compute ONE `CalcScoresCholesky` pairwise split score on device — the
/// device analog of `cb_compute::pairwise_scoring::calculate_score` composed with the leaf solve.
/// `weight_sum` is the row-major `n × n` running pairwise weight matrix (no ridge, `n = 2·PartCount`),
/// `der_sum` the running der vector. Returns the scalar score (`0.0` for a degenerate `n < 2`
/// system, matching the CPU zeros-leaf → 0-score path).
#[allow(dead_code)] // consumed by the #[cfg(test)] cholesky_solve_test self-oracle (source/test separation)
pub(crate) fn score_pairwise_cholesky_host(
    weight_sum: &[f64],
    der_sum: &[f64],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
) -> CbResult<f64> {
    let n = der_sum.len();
    if n < 2 {
        return Ok(0.0);
    }
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CbError::OutOfRange(format!("system_size ({n}) squared overflows")))?;
    if weight_sum.len() != expected {
        return Err(CbError::LengthMismatch {
            column: "weight_sum".to_owned(),
            expected,
            actual: weight_sum.len(),
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
        let mat_h = client.create(cubecl::bytes::Bytes::from_elems(weight_sum.to_vec()));
        let der_h = client.create(cubecl::bytes::Bytes::from_elems(der_sum.to_vec()));
        let handle = launch_cholesky_solve(
            &client,
            &mat_h,
            &der_h,
            n,
            1,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
            true,
        )?;
        let out = read_f64(&client, handle)?;
        Ok(out.first().copied().unwrap_or(0.0))
    }
}
