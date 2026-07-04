//! GPUT-22 (Phase 13 Plan 03, Wave 3): the shared device **query-grouping infrastructure** built
//! ONCE, before any ranking objective, so all five query/listwise objectives (Plans 04–05) amortize
//! it. This is the `query_helper` kernel surface — group ids, group means/max, group-bias removal,
//! in-query sort keys (feeding the EXISTING segmented radix sort), and taken-docs / query-end masks
//! — each matching the CPU `cb_compute::ranking_der` group reductions at ε=1e-4.
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! Serial (`ComputeGroupIds` / `ComputeGroupMeans` / `ComputeGroupMax` / `FillTakenDocsMask` /
//! `FillQueryEndMask` / `ComputeSampledSizes`) and doc-parallel (`RemoveGroupMeans` /
//! `CreateSortKeys`) `#[cube]` kernels. The upstream `query_helper.cu` (§6.6a) names are kept in the
//! doc comments; the Rust identifiers are snake_case per CLAUDE.md. Only O(1) group descriptors
//! (query offsets) + the resident value/der buffer cross the host↔device seam.
//!
//! # Determinism — the fixed-point group reduction (D-08 / T-13-06)
//!
//! `ComputeGroupMeans` folds the per-query der/weight SUMS through the k=30 fixed-point path
//! ([`crate::kernels::REDUCE_FIXEDPOINT_SCALE_F64`]): each `value·weight` (and `weight`) is quantized
//! `round(x · 2^30) → i64 → u64 bits` and accumulated with a wrapping integer add, which is EXACT and
//! ORDER-INDEPENDENT (two's-complement `u64` add == `i64` add) — the property gfx1100 cannot offer via
//! f64 atomic-add (it advertises none). This matches the CPU `group_reduce_weighted` /
//! `cb_core::sum_f64` group normalizer within ε=1e-4 (the 2^-30 quantization residual is ~1e-9, far
//! inside the bar). `ComputeGroupMax` / `RemoveGroupMeans` are order-invariant (max / per-doc subtract)
//! so they need no fixed-point channel.
//!
//! # In-query sampling — reuse the segmented radix sort (do NOT hand-roll a second sort)
//!
//! [`shuffle_within_queries_host`] composes `CreateSortKeys` (`key = (qid<<32) | random_low_32` via
//! the inline PCG RNG transcribed from [`crate::kernels::mvs_device`]) with the EXISTING
//! [`crate::kernels::exact_quantile::segmented_radix_sort`]: the query head-flags keep queries
//! contiguous (the high `qid` bits are implicit in the segmentation) while the random low 32 bits
//! shuffle docs WITHIN each query. No second sort algorithm is introduced.
//!
//! # f64/u64-typed seam (WR-01)
//!
//! The group reductions accumulate in f64 / fixed-point u64 and the RNG is a u64 quantity; WGSL has
//! neither f64 nor u64, so a genuine `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather
//! than an opaque JIT crash. The in-env rocm/cuda/cpu path is unaffected. No `-inf` literal in any
//! `#[cube]` body (a finite seed is used for the group max). No `unwrap`/`expect`/`panic`/indexing in
//! production (workspace lints + D-13).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::kernels::REDUCE_FIXEDPOINT_SCALE_F64;
use crate::SelectedRuntime;

/// LCG multiplier `A` (`cb_core::rng::LCG_MULTIPLIER`, `0x5851F42D4C957F2D`) — transcribed inline
/// (the `#[cube]` body cannot reach `cb_core`), matching [`crate::kernels::mvs_device`].
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

// ===========================================================================
// #[cube] RNG primitives (transcribed from cb_core::TFastRng64 — bit-for-bit,
// mirroring crate::kernels::mvs_device)
// ===========================================================================

/// `RotateBitsRight(v, r)` for a 32-bit word (`fast.h` `TPCGMixer`). `r = x >> 59` is in `0..32`
/// (never 32); the `r == 0` guard avoids the `v << 32` UB shift.
#[cube]
fn rotate_right_u32(v: u32, r: u32) -> u32 {
    let mut out = v;
    if r != 0u32 {
        out = (v >> r) | (v << (32u32 - r));
    }
    out
}

/// `TPCGMixer::Mix` (`fast.h`): XSH-RR on the 64-bit state → 32-bit output, matching
/// [`cb_core::rng::pcg_mix`] exactly.
#[cube]
fn pcg_mix(x: u64) -> u32 {
    let xorshifted = u32::cast_from(((x >> 18u32) ^ x) >> 27u32);
    let rot = u32::cast_from(x >> 59u32);
    rotate_right_u32(xorshifted, rot)
}

/// `SampledQuerySize(sampleRate, qSize)` — the per-query sampled document count, FLOORED at 2
/// (upstream `query_helper.cu` §6.6a: a sampled query keeps at least a competitor pair). A truncating
/// `u32::cast_from(f64)` is `floor` for the non-negative `rate·qSize`; the result is clamped into
/// `[min(2, qSize), qSize]` so a 1-doc query yields its single doc (never an out-of-range 2).
#[cube]
fn sampled_query_size(sample_rate: f64, q_size: u32) -> u32 {
    let raw = sample_rate * f64::cast_from(q_size);
    let mut s = u32::cast_from(raw);
    if s < 2u32 {
        s = 2u32;
    }
    if s > q_size {
        s = q_size;
    }
    s
}

// ===========================================================================
// #[cube] grouping kernels
// ===========================================================================

/// `ComputeGroupIds` (`query_helper.cu` §6.6a): scatter each doc's query index `qid` to `qids[d]`.
/// Serial (unit 0) walk over the `n_groups` query spans `[q_offsets[g], q_offsets[g+1])`. `q_offsets`
/// has `n_groups + 1` entries.
#[cube(launch)]
fn compute_group_ids_kernel(q_offsets: &Array<u32>, qids: &mut Array<u32>) {
    if ABSOLUTE_POS == 0 {
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            let gid = u32::cast_from(g);
            let mut d = begin;
            while d < end {
                qids[d as usize] = gid;
                d += 1u32;
            }
            g += 1usize;
        }
    }
}

/// `ComputeGroupMeans` (`query_helper.cu` §6.6a), q-offsets overload: the per-query WEIGHTED mean
/// `Σ(value·weight) / Σ weight` (the `queryAvrg` numerator / denominator every querywise ranking loss
/// uses; `cb_compute::ranking_der::group_reduce_weighted`). Serial (unit 0) per query; both SUMS run
/// through the k=30 fixed-point path ([`REDUCE_FIXEDPOINT_SCALE_F64`]) so the reduction is
/// deterministic (T-13-06). A zero-weight query yields mean `0` (never divides — Security V5).
/// `weights` is length `n` (the host expands a uniform `1.0` column when unweighted).
#[cube(launch)]
fn compute_group_means_kernel(
    values: &Array<f64>,
    weights: &Array<f64>,
    q_offsets: &Array<u32>,
    out_means: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let scale = REDUCE_FIXEDPOINT_SCALE_F64;
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            // Wrapping fixed-point accumulators (exact, order-independent integer add).
            let mut num_acc = 0u64;
            let mut den_acc = 0u64;
            let mut d = begin;
            while d < end {
                let w = weights[d as usize];
                let prod = values[d as usize] * w;
                num_acc += u64::cast_from(i64::cast_from(f64::round(prod * scale)));
                den_acc += u64::cast_from(i64::cast_from(f64::round(w * scale)));
                d += 1u32;
            }
            let num = f64::cast_from(i64::cast_from(num_acc)) / scale;
            let den = f64::cast_from(i64::cast_from(den_acc)) / scale;
            let mut mean = 0.0f64;
            if den > 0.0f64 {
                mean = num / den;
            }
            out_means[g] = mean;
            g += 1usize;
        }
    }
}

/// `ComputeGroupMax` (`query_helper.cu` §6.6a): the per-query maximum `value`. Serial (unit 0) per
/// query, seeded with the query's FIRST element (NO `-inf` literal — Pattern D); an empty query
/// writes `0`. Order-invariant, so no fixed-point channel is needed.
#[cube(launch)]
fn compute_group_max_kernel(
    values: &Array<f64>,
    q_offsets: &Array<u32>,
    out_max: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            let mut m = 0.0f64;
            if end > begin {
                m = values[begin as usize];
                let mut d = begin + 1u32;
                while d < end {
                    let v = values[d as usize];
                    if v > m {
                        m = v;
                    }
                    d += 1u32;
                }
            }
            out_max[g] = m;
            g += 1usize;
        }
    }
}

/// `RemoveGroupMeans` (`query_helper.cu` §6.6a): the doc-parallel per-query bias removal
/// `out[d] = values[d] - group_means[qids[d]]`. Each doc is independent (ABSOLUTE_POS-indexed), so the
/// write is order-independent.
#[cube(launch)]
fn remove_group_means_kernel(
    values: &Array<f64>,
    qids: &Array<u32>,
    group_means: &Array<f64>,
    out: &mut Array<f64>,
) {
    let d = ABSOLUTE_POS;
    if d < values.len() {
        let g = qids[d];
        out[d] = values[d] - group_means[g as usize];
    }
}

/// `CreateSortKeys` (`query_helper.cu` §6.6a): each doc's in-query sort key `random_low_32` (the low
/// 32 bits of the conceptual `key = (qid<<32) | random_low_32`; the high `qid` bits are supplied by
/// the segmentation of the downstream [`crate::kernels::exact_quantile::segmented_radix_sort`], so
/// queries stay contiguous while the low bits shuffle docs WITHIN a query). Per-doc inline PCG seed
/// `base_seed + d` mixed once — doc-parallel, deterministic for a pinned `base_seed`.
#[cube(launch)]
fn create_sort_keys_kernel(seed: &Array<u64>, keys: &mut Array<u32>) {
    let d = ABSOLUTE_POS;
    if d < keys.len() {
        let a = LCG_MULTIPLIER;
        let base = seed[0];
        let mut x = base + u64::cast_from(d);
        x = x * a + 1u64;
        keys[d] = pcg_mix(x);
    }
}

/// `FillTakenDocsMask` (`query_helper.cu` §6.6a): mark the first [`sampled_query_size`] docs of each
/// query as taken (`mask == 1`), the rest `0`. Serial (unit 0) per query. `params = [sample_rate]`.
/// The mask is over the CURRENT doc order (compose with [`shuffle_within_queries_host`] first for a
/// random in-query sample).
#[cube(launch)]
fn fill_taken_docs_mask_kernel(
    q_offsets: &Array<u32>,
    params: &Array<f64>,
    mask: &mut Array<u32>,
) {
    if ABSOLUTE_POS == 0 {
        let rate = params[0];
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            let q_size = end - begin;
            let sampled = sampled_query_size(rate, q_size);
            let mut rank = 0u32;
            let mut d = begin;
            while d < end {
                let mut m = 0u32;
                if rank < sampled {
                    m = 1u32;
                }
                mask[d as usize] = m;
                rank += 1u32;
                d += 1u32;
            }
            g += 1usize;
        }
    }
}

/// The per-query sampled sizes `sampled_query_size(rate, qSize)` — a thin readback surface over the
/// [`sampled_query_size`] floor (`ComputeSampledSizes`), so the ≥2 floor is directly observable.
/// Serial (unit 0) per query; `params = [sample_rate]`.
#[cube(launch)]
fn compute_sampled_sizes_kernel(
    q_offsets: &Array<u32>,
    params: &Array<f64>,
    out_sizes: &mut Array<u32>,
) {
    if ABSOLUTE_POS == 0 {
        let rate = params[0];
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            out_sizes[g] = sampled_query_size(rate, end - begin);
            g += 1usize;
        }
    }
}

/// `FillQueryEndMask` (`query_helper.cu` §6.6a): mark the LAST doc of each query (`mask == 1`), the
/// rest `0` (the host pre-zeroes the buffer). Serial (unit 0) per query — the per-query segment-end
/// flags a scan uses to reset per-query prefix state.
#[cube(launch)]
fn fill_query_end_mask_kernel(q_offsets: &Array<u32>, mask: &mut Array<u32>) {
    if ABSOLUTE_POS == 0 {
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let end = q_offsets[g + 1];
            let begin = q_offsets[g];
            if end > begin {
                mask[(end - 1u32) as usize] = 1u32;
            }
            g += 1usize;
        }
    }
}

// ===========================================================================
// Host launch wrappers (device-resident Handle + readback oracle wrappers)
// ===========================================================================

/// Reject the (impossible) wgpu f64/u64 path with a typed error (WR-01), mirroring
/// [`crate::kernels::mvs_device`]. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device query-grouping requires f64 + u64 device channels; the wgpu backend has neither \
         (WR-01). Use the rocm/cuda/cpu backend for the group reductions."
            .to_owned(),
    )
}

/// The selected-runtime client (one per call, mirroring [`crate::kernels::mvs_device`]).
#[cfg(not(feature = "wgpu"))]
fn selected_client() -> cubecl::client::ComputeClient<SelectedRuntime> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    <SelectedRuntime as cubecl::Runtime>::client(&device)
}

/// Build the per-query head-flag array (`head_flags[begin] == 1` at every query start, else `0`;
/// `head_flags[0] == 1`) that [`crate::kernels::exact_quantile::segmented_radix_sort`] consumes.
/// `q_offsets` has `n_groups + 1` entries; out-of-range starts are ignored (defensive).
fn build_query_head_flags(q_offsets: &[u32], n: usize) -> Vec<u32> {
    let mut flags = vec![0u32; n];
    for &start in q_offsets.iter().take(q_offsets.len().saturating_sub(1)) {
        if let Some(slot) = flags.get_mut(start as usize) {
            *slot = 1;
        }
    }
    if let Some(first) = flags.get_mut(0) {
        *first = 1;
    }
    flags
}

/// `ComputeGroupIds` host readback: the per-doc query id for the `n` docs described by `q_offsets`.
pub(crate) fn compute_group_ids_host(q_offsets: &[u32], n: usize) -> CbResult<Vec<u32>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = q_offsets;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let client = selected_client();
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));
        compute_group_ids_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        read_u32(&client, out, "compute_group_ids")
    }
}

/// `ComputeGroupMeans` host readback: the per-query weighted means. `weights` is either empty
/// (uniform `1.0`) or length `n`.
pub(crate) fn compute_group_means_host(
    values: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
) -> CbResult<Vec<f64>> {
    let n = values.len();
    let n_groups = q_offsets.len().saturating_sub(1);
    if n_groups == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (weights, n);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let weight_col: Vec<f64> = if weights.is_empty() {
            vec![1.0; n]
        } else {
            weights.to_vec()
        };
        if weight_col.len() != n {
            return Err(CbError::Degenerate(format!(
                "compute_group_means_host: weights len {} != values len {n}",
                weight_col.len()
            )));
        }
        // Reject magnitudes that would overflow the i64 fixed-point accumulator (IN-01): the kernel
        // quantizes `value·weight` (and `weight`) as `i64::cast_from(round(prod · scale))`, whose
        // f64 → i64 cast is backend-defined once `|prod| · scale` exceeds `i64::MAX`, silently
        // corrupting the group mean. The covered mean-removed residuals stay far inside this bound;
        // still reject an out-of-range magnitude up front with a typed error rather than trust it.
        let max_mag = (i64::MAX as f64) / REDUCE_FIXEDPOINT_SCALE_F64;
        for (v, w) in values.iter().zip(weight_col.iter()) {
            if (v * w).abs() > max_mag || w.abs() > max_mag {
                return Err(CbError::OutOfRange(format!(
                    "compute_group_means_host: |value·weight| or |weight| exceeds the i64 \
                     fixed-point range (±{max_mag:.3e}); the deterministic group reduction would \
                     overflow"
                )));
            }
        }
        let client = selected_client();
        let val_h = client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()));
        let w_h = client.create(cubecl::bytes::Bytes::from_elems(weight_col));
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let out = client.empty(n_groups * std::mem::size_of::<f64>());
        compute_group_means_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(val_h, n) },
            unsafe { ArrayArg::from_raw_parts(w_h, n) },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n_groups) },
        );
        read_f64(&client, out, "compute_group_means")
    }
}

/// `ComputeGroupMax` host readback: the per-query maxima.
pub(crate) fn compute_group_max_host(values: &[f64], q_offsets: &[u32]) -> CbResult<Vec<f64>> {
    let n = values.len();
    let n_groups = q_offsets.len().saturating_sub(1);
    if n_groups == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = n;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let client = selected_client();
        let val_h = client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()));
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let out = client.empty(n_groups * std::mem::size_of::<f64>());
        compute_group_max_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(val_h, n) },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n_groups) },
        );
        read_f64(&client, out, "compute_group_max")
    }
}

/// `RemoveGroupMeans` host readback: `values[d] - group_means[qids[d]]` per doc. `qids` / `values`
/// are length `n`; `group_means` is length `n_groups`.
pub(crate) fn remove_group_means_host(
    values: &[f64],
    qids: &[u32],
    group_means: &[f64],
) -> CbResult<Vec<f64>> {
    let n = values.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if qids.len() != n {
        return Err(CbError::Degenerate(format!(
            "remove_group_means_host: qids len {} != values len {n}",
            qids.len()
        )));
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = group_means;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let client = selected_client();
        let val_h = client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()));
        let qid_h = client.create(cubecl::bytes::Bytes::from_elems(qids.to_vec()));
        let means_h = client.create(cubecl::bytes::Bytes::from_elems(group_means.to_vec()));
        let out = client.empty(n * std::mem::size_of::<f64>());
        let block = 64u32;
        let cubes = n.div_ceil(block as usize).max(1) as u32;
        remove_group_means_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim { x: block, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(val_h, n) },
            unsafe { ArrayArg::from_raw_parts(qid_h, n) },
            unsafe { ArrayArg::from_raw_parts(means_h, group_means.len()) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        read_f64(&client, out, "remove_group_means")
    }
}

/// `CreateSortKeys` host readback: the `n` per-doc `random_low_32` in-query sort keys for the pinned
/// `base_seed`. Returns the resident buffer read back to host `u32`.
pub(crate) fn create_sort_keys_host(base_seed: u64, n: usize) -> CbResult<Vec<u32>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = base_seed;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let client = selected_client();
        let seed_h = client.create(cubecl::bytes::Bytes::from_elems(vec![base_seed]));
        let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));
        let block = 64u32;
        let cubes = n.div_ceil(block as usize).max(1) as u32;
        create_sort_keys_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim { x: block, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(seed_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        read_u32(&client, out, "create_sort_keys")
    }
}

/// The per-query sampled sizes (`ComputeSampledSizes`) — the ≥2-floor readback surface.
pub(crate) fn compute_sampled_sizes_host(
    q_offsets: &[u32],
    sample_rate: f64,
) -> CbResult<Vec<u32>> {
    let n_groups = q_offsets.len().saturating_sub(1);
    if n_groups == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = sample_rate;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let client = selected_client();
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![sample_rate]));
        let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n_groups]));
        compute_sampled_sizes_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(params_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n_groups) },
        );
        read_u32(&client, out, "compute_sampled_sizes")
    }
}

/// `FillTakenDocsMask` host readback: the per-doc taken mask for the given `sample_rate` over the
/// CURRENT doc order (`n` docs).
pub(crate) fn fill_taken_docs_mask_host(
    q_offsets: &[u32],
    sample_rate: f64,
    n: usize,
) -> CbResult<Vec<u32>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (q_offsets, sample_rate);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let client = selected_client();
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![sample_rate]));
        let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));
        fill_taken_docs_mask_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(params_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        read_u32(&client, out, "fill_taken_docs_mask")
    }
}

/// `FillQueryEndMask` host readback: the per-doc query-end mask (`1` at each query's last doc).
pub(crate) fn fill_query_end_mask_host(q_offsets: &[u32], n: usize) -> CbResult<Vec<u32>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = q_offsets;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let client = selected_client();
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));
        fill_query_end_mask_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        read_u32(&client, out, "fill_query_end_mask")
    }
}

/// In-query random shuffle: compose `CreateSortKeys` with the EXISTING segmented radix sort
/// ([`crate::kernels::exact_quantile::segmented_radix_sort`]) so queries stay contiguous while docs
/// shuffle WITHIN each query. Returns the permuted doc indices `[0, n)`. `q_offsets` delimits the
/// queries; `base_seed` pins the draw.
pub(crate) fn shuffle_within_queries_host(
    q_offsets: &[u32],
    base_seed: u64,
    n: usize,
) -> CbResult<Vec<u32>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let keys = create_sort_keys_host(base_seed, n)?;
    let head_flags = build_query_head_flags(q_offsets, n);
    let doc_ids: Vec<u32> = (0..n as u32).collect();
    let (_sorted_keys, sorted_docs) =
        crate::kernels::exact_quantile::segmented_radix_sort(&head_flags, &keys, &doc_ids)?;
    Ok(sorted_docs)
}

/// Read a resident `f64` handle back to host, mapping a failure to [`CbError::Degenerate`] (WR-05),
/// never a silent zero buffer.
#[cfg(not(feature = "wgpu"))]
fn read_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
    who: &str,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("query_helper {who} f64 read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Read a resident `u32` handle back to host, mapping a failure to [`CbError::Degenerate`] (WR-05).
#[cfg(not(feature = "wgpu"))]
fn read_u32(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
    who: &str,
) -> CbResult<Vec<u32>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("query_helper {who} u32 read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
}
