//! Device Exact weighted-quantile leaf estimation (GPUT-19, Phase 12 Plan 05) — the
//! order-statistic leaf method for the Quantile / MAE / MAPE family, DISTINCT from the
//! Newton der2 path (GPUT-07, Phase 11; RESEARCH Pitfall 6: Exact is a weighted quantile,
//! NOT `g/(h+ε)`).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! Unlike the `kernels/sort.rs` / `kernels/segmented_scan.rs` SELF-ORACLE harnesses (which
//! are `#[cfg(test)]` mounts), this module is a PRODUCTION submodule: the session gate arm
//! ([`crate::gpu_runtime::session`]) and MVS (Plan 07) call into it. It hosts:
//!
//! - [`segmented_radix_sort`] — the **shared segmented (per-leaf-bin) radix-sort primitive**
//!   the W3 audit added (Open Q1 / A1; see the audit note in `sort.rs`). It sorts keys+values
//!   independently WITHIN flag-delimited segments, reusing the Phase-10 whole-buffer radix
//!   machinery (`radix_bit_flag_kernel` → `full_scan` onesBefore → `reorder_one_bit_scatter`)
//!   per segment — NO second hand-rolled sort. Consumed by Exact (below) AND MVS (Plan 07).
//! - [`device_exact_leaf_delta`] — the device weighted-quantile leaf delta reproducing
//!   `cb-compute/src/leaf.rs::exact_leaf_delta` (≤1e-4): segmented sort of residuals per
//!   leaf-bin → device weight prefix scan (`full_scan`) → deterministic `totalWeight`
//!   (fixed-point `Atomic<u64>` k=30 reduce, `launch_block_reduce_atomic_f64`) → binary
//!   search for `needWeights = totalWeight·α` → the α/δ adjustment.
//!
//! # No new `#[cube]` kernel (HIP JIT safety)
//!
//! Every device step composes ALREADY-VALIDATED Phase-10 `#[cube]` kernels
//! (`radix_bit_flag_kernel`, `full_scan`, `reorder_one_bit_scatter_kernel`) + the Phase-7.1
//! deterministic reduce helper — so there is NO new `#[cube]` body and therefore NO `-inf`
//! / HIP-JIT-reject surface (project landmine: `F::new(f32::NEG_INFINITY)` in a `#[cube]`
//! kernel is a gfx1100 JIT reject invisible to cpu/wgpu). The parity-critical device SUMs
//! (`totalWeight`, the weight prefix) are deterministic (fixed-point reduce + fixed-order
//! Hillis-Steele scan) — never a non-deterministic float atomic (Pattern 3 / Pitfall).
//!
//! # No `cb-train` / `cb-compute` reach (landmine)
//!
//! `exact_leaf_delta`'s semantics are TRANSCRIBED inline here; cb-backend never gains a
//! `cb-train` dep (the feature-unification landmine), and this module reaches only
//! `cb_core` (the ordered `sum_f64` primitive) — no `cb_compute` import.

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::gpu_runtime::launch_block_reduce_atomic_f64;
use crate::kernels::{full_scan_into, radix_bit_flag_kernel, reorder_one_bit_scatter_kernel};
use crate::SelectedRuntime;

/// Launch geometry: 32-wide cubes (wave32 gfx1100), enough cubes to cover every element —
/// mirrors the `sort.rs` / `kernels.rs` scatter geometry (one bounds-guarded write per lane).
const CUBE_DIM: usize = 32;

/// `DBL_EPSILON` — the tolerance the upstream weighted-quantile search compares against
/// (`quantile.cpp:67/98`, `optimal_const_for_loss.h:95`), matching
/// `cb-compute/src/leaf.rs::exact_leaf_delta`'s `DBL_EPSILON`.
const DBL_EPSILON: f64 = f64::EPSILON;

/// STABLE whole-buffer LSD radix sort of `keys` (with paired `values`) over the selected
/// runtime — the production twin of the `kernels::sort` self-oracle's `run_radix_sort`
/// (transcribed here so a PRODUCTION caller need not reach into a `#[cfg(test)]` module).
///
/// Ping-pongs the resident key/value buffers, applying one STABLE single-bit reorder per bit
/// from LSB up to the highest set bit of any key (`radix_bit_flag_kernel` → exclusive
/// `full_scan` onesBefore → `reorder_one_bit_scatter_kernel`). Each pass is stable, so the
/// composition is a stable full sort; only the FINAL buffers are read back (device-resident
/// across passes). `total_zeros[bit]` is order-invariant, computed host-side per bit.
///
/// Returns a typed [`CbError`] on a device read-back failure (never a silent zero buffer);
/// no `unwrap`/`expect`/`panic` (D-13 / workspace lints).
///
/// A FRESH client is obtained per call (mirroring the proven `kernels::sort` self-oracle's
/// `run_radix_sort`): the ping-pong buffers + per-bit `full_scan` handles are scoped to one
/// client and read back before return, so composing many per-segment calls in a loop cannot
/// alias buffers across segments (the memory-pool reuse hazard of threading one shared client
/// through interleaved `empty()`/`read` in a tight loop).
fn run_radix_sort_device(keys: &[u32], values: &[u32]) -> CbResult<(Vec<u32>, Vec<u32>)> {
    let n = keys.len();
    if n != values.len() {
        return Err(CbError::LengthMismatch {
            column: "radix_sort values".to_owned(),
            expected: n,
            actual: values.len(),
        });
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let device = <SelectedRuntime as Runtime>::Device::default();
    let client = <SelectedRuntime as Runtime>::client(&device);
    let dim32 = CubeDim { x: CUBE_DIM as u32, y: 1, z: 1 };
    let n_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(n_cubes as u32, 1, 1);

    // Number of LSD passes = bit-width of the largest key (0 keys ⇒ already sorted).
    let max_key = keys.iter().copied().max().unwrap_or(0);
    let num_bits = if max_key == 0 { 0 } else { 32 - max_key.leading_zeros() };

    let mut cur_keys = client.create(cubecl::bytes::Bytes::from_elems(keys.to_vec()));
    let mut cur_vals = client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()));
    let mut alt_keys = client.empty(n * std::mem::size_of::<u32>());
    let mut alt_vals = client.empty(n * std::mem::size_of::<u32>());

    for bit in 0..num_bits {
        let flags_h = client.empty(n * std::mem::size_of::<f64>());
        radix_bit_flag_kernel::launch::<f64, SelectedRuntime>(
            &client,
            count.clone(),
            dim32,
            unsafe { ArrayArg::from_raw_parts(cur_keys.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(flags_h.clone(), n) },
            bit,
        );
        let ones_before_h = full_scan_into::<f64>(&client, flags_h, n, false)?;
        let total_zeros = keys.iter().filter(|&&k| (k >> bit) & 1 == 0).count() as u32;
        reorder_one_bit_scatter_kernel::launch::<f64, SelectedRuntime>(
            &client,
            count.clone(),
            dim32,
            unsafe { ArrayArg::from_raw_parts(cur_keys.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(cur_vals.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(ones_before_h, n) },
            unsafe { ArrayArg::from_raw_parts(alt_keys.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(alt_vals.clone(), n) },
            bit,
            total_zeros,
        );
        std::mem::swap(&mut cur_keys, &mut alt_keys);
        std::mem::swap(&mut cur_vals, &mut alt_vals);
    }

    let kb = client
        .read_one(cur_keys)
        .map_err(|e| CbError::Degenerate(format!("radix sort key read-back failed: {e:?}")))?;
    let vb = client
        .read_one(cur_vals)
        .map_err(|e| CbError::Degenerate(format!("radix sort value read-back failed: {e:?}")))?;
    Ok((
        bytemuck::cast_slice::<u8, u32>(&kb).to_vec(),
        bytemuck::cast_slice::<u8, u32>(&vb).to_vec(),
    ))
}

/// **Segmented radix sort** (Open Q1 / A1 — the W3 primitive shared by Exact + MVS). Sorts
/// `keys` (with paired `values`) STABLY and ASCENDING *independently within each segment*,
/// where a segment START is marked by `head_flags[i] == 1` (mirroring the `segmented_scan.rs`
/// head-flag geometry). `head_flags[0]` MUST be `1` (the first segment opens at 0). Elements
/// never move across a segment boundary — a truncated/mixed result is impossible by
/// construction (each segment is sorted over its OWN slice).
///
/// # Why this is not a second sort algorithm
///
/// The audit (`sort.rs`) found the Phase-10 sort exposes only a WHOLE-BUFFER radix sort
/// (`run_radix_sort`, keys+values). This primitive REUSES that exact radix machinery
/// ([`run_radix_sort_device`]) once per flag-delimited segment and stitches the per-segment
/// results back at their original positions — so no new sort algorithm is introduced (the
/// plan's "reuse the radix machinery; do NOT hand-roll a second sort" constraint). For the
/// self-oracle / Exact leaf-bin sizes this per-segment orchestration is correct and
/// deterministic; a single fused segmented-radix kernel (composite `(seg<<bits)|key`) is a
/// performance follow-up, not a correctness need.
///
/// Returns a typed [`CbError`] on a malformed head-flag array or a device failure.
pub(crate) fn segmented_radix_sort(
    head_flags: &[u32],
    keys: &[u32],
    values: &[u32],
) -> CbResult<(Vec<u32>, Vec<u32>)> {
    let n = keys.len();
    if n != values.len() || n != head_flags.len() {
        return Err(CbError::LengthMismatch {
            column: "segmented_radix_sort inputs".to_owned(),
            expected: n,
            actual: values.len().min(head_flags.len()),
        });
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if head_flags[0] != 1 {
        return Err(CbError::Degenerate(
            "segmented_radix_sort: head_flags[0] must be 1 (the first segment opens at 0)"
                .to_owned(),
        ));
    }

    let mut out_keys = vec![0u32; n];
    let mut out_vals = vec![0u32; n];

    // Walk the head flags, sorting each [seg_start, seg_end) slice in isolation.
    let mut seg_start = 0usize;
    let mut i = 1usize;
    while i <= n {
        let boundary = i == n || head_flags.get(i).copied().unwrap_or(0) == 1;
        if boundary {
            let seg_keys = keys.get(seg_start..i).unwrap_or(&[]);
            let seg_vals = values.get(seg_start..i).unwrap_or(&[]);
            let (sk, sv) = run_radix_sort_device(seg_keys, seg_vals)?;
            // Stitch the per-segment sorted result back at its original positions.
            for (off, (&k, &v)) in sk.iter().zip(sv.iter()).enumerate() {
                if let (Some(dk), Some(dv)) =
                    (out_keys.get_mut(seg_start + off), out_vals.get_mut(seg_start + off))
                {
                    *dk = k;
                    *dv = v;
                }
            }
            seg_start = i;
        }
        i += 1;
    }

    Ok((out_keys, out_vals))
}

/// Map an `f32` to an order-preserving `u32` radix key: sorting the keys ASCENDING sorts the
/// original `f32` residuals ascending, negatives before positives. Flip the sign bit for a
/// positive/zero value; flip ALL bits for a negative value (the standard IEEE-754 monotone
/// bijection). Matches the STABLE ascending order `exact_leaf_delta` sorts residuals in
/// (`f32::total_cmp` for all-finite inputs). `#[inline]` host helper — NOT a `#[cube]` body.
#[inline]
fn f32_to_ordered_u32(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Device Exact weighted-quantile leaf delta for ONE leaf-bin's residuals — the device twin
/// of `cb-compute/src/leaf.rs::exact_leaf_delta` (≤1e-4), for the Quantile / MAE / MAPE
/// family (GPUT-19, D-09). DISTINCT from Newton (`g/(h+ε)`): this is a weighted order
/// statistic.
///
/// `residuals[i]` is member `i`'s `target_i - approx_i` (widened through `f32` to match
/// upstream's `TVector<float> leafSamples`); `weights[i]` its object weight (for MAPE the
/// caller pre-divides by `max(1, |target|)` — the `weightsWithTargets` transform, A4).
/// `alpha`/`delta` are the loss's quantile parameters. Reproduces the CPU reference exactly:
///
/// 1. Empty leaf → `0.0`; `alpha <= 0` → the min residual (`CalcSampleQuantile:113-115`).
/// 2. Segmented radix sort of residuals ASCENDING (single segment here), carrying the
///    original index — [`segmented_radix_sort`] over the [`f32_to_ordered_u32`] keys.
/// 3. `weightsPrefixSum` = inclusive [`full_scan`] of the sorted weights (device, fixed
///    order → deterministic).
/// 4. `totalWeight` = deterministic fixed-point `Atomic<u64>` reduce
///    ([`launch_block_reduce_atomic_f64`]); `needWeight = totalWeight·alpha`.
/// 5. BINARY search over the monotone weight prefix for the FIRST doc whose cumulative
///    weight `>= needWeight - DBL_EPSILON` (same first-crossing index the CPU linear search
///    returns) → `quantile = sorted_residual[doc]`; fall back to the last (largest) value.
/// 6. The α/δ adjustment (`CalculateWeightedTargetQuantile`): `q -= delta` if
///    `lessWeight + equalWeight·alpha >= needWeight - DBL_EPSILON`, else `q += delta`
///    (`less`/`equal` weights folded via the ordered `cb_core::sum_f64`, matching the CPU
///    reference's `sum_f64`).
///
/// No `unwrap`/`expect`/`panic`/indexing (D-13); no `cb-train`/`cb-compute` reach.
pub(crate) fn device_exact_leaf_delta(
    residuals: &[f32],
    weights: &[f64],
    alpha: f64,
    delta: f64,
) -> CbResult<f64> {
    let n = residuals.len();
    if n == 0 {
        return Ok(0.0);
    }
    // alpha <= 0 -> min element (CalcSampleQuantile:113-115).
    if alpha <= 0.0 {
        let mut min = f64::INFINITY;
        for &v in residuals {
            let v = f64::from(v);
            if v < min {
                min = v;
            }
        }
        return Ok(min);
    }

    // 2. Segmented radix sort ASCENDING, carrying the original index (single segment).
    let keys: Vec<u32> = residuals.iter().map(|&v| f32_to_ordered_u32(v)).collect();
    let idx: Vec<u32> = (0..n as u32).collect();
    let mut head_flags = vec![0u32; n];
    if let Some(h) = head_flags.get_mut(0) {
        *h = 1;
    }
    let (_sorted_keys, sorted_idx) = segmented_radix_sort(&head_flags, &keys, &idx)?;

    // Gather the sorted residuals + weights (weights default to 1.0 when unsupplied, exactly
    // as `exact_leaf_delta` pairs `weights.get(i).copied().unwrap_or(1.0)`).
    let sorted_res: Vec<f32> = sorted_idx
        .iter()
        .map(|&j| residuals.get(j as usize).copied().unwrap_or(0.0))
        .collect();
    let sorted_w: Vec<f64> = sorted_idx
        .iter()
        .map(|&j| weights.get(j as usize).copied().unwrap_or(1.0))
        .collect();

    // 3. Device inclusive weight prefix scan (fixed-order Hillis-Steele → deterministic).
    let device = <SelectedRuntime as Runtime>::Device::default();
    let client = <SelectedRuntime as Runtime>::client(&device);
    let w_handle: Handle = client.create(cubecl::bytes::Bytes::from_elems(sorted_w.clone()));
    let prefix_handle = full_scan_into::<f64>(&client, w_handle, n, true)?;
    let prefix_bytes = client
        .read_one(prefix_handle)
        .map_err(|e| CbError::Degenerate(format!("weight prefix read-back failed: {e:?}")))?;
    let prefix: Vec<f64> = bytemuck::cast_slice::<u8, f64>(&prefix_bytes).to_vec();
    if prefix.len() != n {
        return Err(CbError::LengthMismatch {
            column: "weight prefix scan".to_owned(),
            expected: n,
            actual: prefix.len(),
        });
    }

    // 4. Deterministic totalWeight (fixed-point Atomic<u64> k=30) + needWeight.
    let (total_weight, _path) = launch_block_reduce_atomic_f64(&sorted_w)?;
    let need_weight = total_weight * alpha;

    // 5. Binary search over the monotone prefix for the FIRST index with
    //    prefix[j] >= need_weight - DBL_EPSILON (== the CPU linear search's first crossing).
    //    Fixed O(log n) iterations, no unbounded loop (T-12-08). Fallback: the last value.
    let threshold = need_weight - DBL_EPSILON;
    let mut lo = 0usize;
    let mut hi = n; // n == "not found" sentinel.
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if prefix.get(mid).copied().unwrap_or(f64::INFINITY) >= threshold {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let quantile = if lo < n {
        f64::from(sorted_res.get(lo).copied().unwrap_or(0.0))
    } else {
        // CalcSampleQuantileLinearSearch's `return elements.back().Value` fallback.
        f64::from(sorted_res.get(n - 1).copied().unwrap_or(0.0))
    };

    // 6. Delta adjustment (CalculateWeightedTargetQuantile, optimal_const_for_loss.h:82-100).
    let mut quantile = quantile;
    if delta > 0.0 {
        let q_f32 = quantile as f32;
        let mut less_members: Vec<f64> = Vec::new();
        let mut equal_members: Vec<f64> = Vec::new();
        for (r, w) in sorted_res.iter().zip(sorted_w.iter()) {
            if *r < q_f32 {
                less_members.push(*w);
            } else if *r == q_f32 {
                equal_members.push(*w);
            }
        }
        let less_weight = cb_core::sum_f64(&less_members);
        let equal_weight = cb_core::sum_f64(&equal_members);
        if less_weight + equal_weight * alpha >= need_weight - DBL_EPSILON {
            quantile -= delta;
        } else {
            quantile += delta;
        }
    }

    Ok(quantile)
}
