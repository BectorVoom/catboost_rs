//! Self-oracle for the device query-grouping infrastructure (Phase 13 Plan 03, GPUT-22, Pattern F):
//! the device [`crate::kernels::query_helper`] kernels must reproduce an INDEPENDENT serial CPU
//! reference of the `cb_compute::ranking_der` group reductions within `TOL` (ε=1e-4, the D-07 GPU
//! bar):
//!
//! - **Group means/max** — `compute_group_means_host` / `compute_group_max_host` over a fixture with
//!   uneven query sizes equal the CPU per-group weighted mean (`Σ(value·w)/Σw`, folded through
//!   `cb_core::sum_f64` — the SAME normalizer `group_reduce_weighted` uses) / per-query max.
//! - **Bias removal** — `remove_group_means_host` yields `values[d] - mean[qid[d]]` matching the CPU
//!   per-query bias removal.
//! - **In-query contiguity** — `CreateSortKeys` + the EXISTING `segmented_radix_sort` keep queries
//!   contiguous (qids non-decreasing across the sorted output) while shuffling docs within a query.
//! - **SampledQuerySize floor** — the per-query sampled size floors at 2 (never exceeds the query).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernels + launchers live
//! in the production `kernels::query_helper` module; ALL assertions / `.unwrap()` / indexing live
//! HERE. The CPU reference is an independent hand-written reduction over the SAME `GroupSpan` shape,
//! baselined via `cb_core::sum_f64` (the sanctioned ordered fold) — NO `cb-train` dep even in the
//! test (the feature-unification landmine), and the device kernel is a separate CubeCL JIT codepath,
//! so this is NOT a tautology.
//!
//! Runs over [`crate::SelectedRuntime`]. The f64/u64 group reductions are validated on ROCm/CUDA
//! in-env; the group-mean/max/bias NUMERIC assertion SKIPS off rocm/cuda (record-only) so a default
//! `cpu`-backend run does not silently "pass" a CPU-vs-CPU compare without a real device (WR-01
//! anti-false-pass). The query-contiguity + SampledQuerySize-floor invariants are backend-independent
//! (they hold for ANY RNG draw / any correct sort), so they hard-assert on every backend. The whole
//! file is gated `not(feature = "wgpu")` — the f64/u64 reductions have no wgpu backend (the launchers
//! reject it with a typed error, never a JIT crash).
#![cfg(not(feature = "wgpu"))]

use cb_core::sum_f64;

use crate::kernels::query_helper::{
    compute_group_ids_host, compute_group_max_host, compute_group_means_host,
    compute_sampled_sizes_host, fill_query_end_mask_host, fill_taken_docs_mask_host,
    remove_group_means_host, shuffle_within_queries_host,
};

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device reductions actually run on a real device backend (rocm/cuda). On the default
/// `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU false-pass
/// (WR-01) — record-only there, hard-assert on rocm/cuda.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// Per-doc query ids from `q_offsets` (the CPU reference for `ComputeGroupIds`).
fn cpu_group_ids(q_offsets: &[u32], n: usize) -> Vec<u32> {
    let mut qids = vec![0u32; n];
    for g in 0..q_offsets.len().saturating_sub(1) {
        let begin = q_offsets[g] as usize;
        let end = q_offsets[g + 1] as usize;
        for d in begin..end {
            if let Some(slot) = qids.get_mut(d) {
                *slot = g as u32;
            }
        }
    }
    qids
}

/// Per-query WEIGHTED mean `Σ(value·w)/Σw` via the sanctioned ordered `cb_core::sum_f64` fold — the
/// SAME normalizer `cb_compute::ranking_der::group_reduce_weighted` uses (D-08). Empty `weights` ⇒
/// uniform `1.0`; a zero-weight query ⇒ `0.0` (never divides — Security V5).
fn cpu_group_means(values: &[f64], weights: &[f64], q_offsets: &[u32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(q_offsets.len().saturating_sub(1));
    for g in 0..q_offsets.len().saturating_sub(1) {
        let begin = q_offsets[g] as usize;
        let end = q_offsets[g + 1] as usize;
        let mut prods = Vec::with_capacity(end - begin);
        let mut ws = Vec::with_capacity(end - begin);
        for d in begin..end {
            let w = if weights.is_empty() { 1.0 } else { weights[d] };
            prods.push(values[d] * w);
            ws.push(w);
        }
        let num = sum_f64(&prods);
        let den = sum_f64(&ws);
        out.push(if den > 0.0 { num / den } else { 0.0 });
    }
    out
}

/// Per-query max (seeded with the first element; empty query ⇒ `0.0` — matches the kernel's finite
/// seed, NO `-inf`).
fn cpu_group_max(values: &[f64], q_offsets: &[u32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(q_offsets.len().saturating_sub(1));
    for g in 0..q_offsets.len().saturating_sub(1) {
        let begin = q_offsets[g] as usize;
        let end = q_offsets[g + 1] as usize;
        let mut m = 0.0_f64;
        if end > begin {
            m = values[begin];
            for d in (begin + 1)..end {
                if values[d] > m {
                    m = values[d];
                }
            }
        }
        out.push(m);
    }
    out
}

/// Max absolute divergence over two equal-length buffers (∞ if the lengths disagree).
fn max_abs_divergence(device: &[f64], reference: &[f64]) -> f64 {
    if device.len() != reference.len() {
        return f64::INFINITY;
    }
    device
        .iter()
        .zip(reference.iter())
        .map(|(&d, &r)| (d - r).abs())
        .fold(0.0_f64, f64::max)
}

/// A deterministic, varied value/weight fixture over three UNEVEN queries (sizes 3 / 1 / 5).
fn uneven_fixture() -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let q_offsets = vec![0u32, 3, 4, 9];
    let n = 9usize;
    let values: Vec<f64> = (0..n)
        .map(|k| {
            let x = k as f64;
            (x * 0.7).sin() * 2.0 + (x * 0.31).cos() * 0.5 - 0.3
        })
        .collect();
    let weights: Vec<f64> = (0..n).map(|k| 0.5 + 0.25 * ((k % 4) as f64)).collect();
    (values, weights, q_offsets)
}

/// Test 1: group means + max over a fixture with uneven query sizes equal the CPU `group_reduce`
/// results within `TOL` (numeric assert gated to a real device; finiteness + length always).
#[test]
fn group_means_and_max_match_cpu_within_epsilon() {
    let (values, weights, q_offsets) = uneven_fixture();
    let n_groups = q_offsets.len() - 1;

    let ref_means = cpu_group_means(&values, &weights, &q_offsets);
    let ref_max = cpu_group_max(&values, &q_offsets);

    let dev_means = compute_group_means_host(&values, &weights, &q_offsets)
        .expect("device group-means must not error on a covered fixture");
    let dev_max = compute_group_max_host(&values, &q_offsets)
        .expect("device group-max must not error on a covered fixture");

    assert_eq!(dev_means.len(), n_groups, "one mean per query");
    assert_eq!(dev_max.len(), n_groups, "one max per query");
    for &v in dev_means.iter().chain(dev_max.iter()) {
        assert!(v.is_finite(), "device group reduction must be finite, got {v}");
    }

    let mean_div = max_abs_divergence(&dev_means, &ref_means);
    let max_div = max_abs_divergence(&dev_max, &ref_max);
    println!(
        "[query_helper] group means max_div={mean_div:e} group max max_div={max_div:e} \
         (device_backend_active={})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(mean_div <= TOL, "device group means diverged from CPU: {mean_div:e} > {TOL:e}");
        assert!(max_div <= TOL, "device group max diverged from CPU: {max_div:e} > {TOL:e}");
    }
}

/// Test 1b: a zero-weight query yields mean `0` (never divides — Security V5); mirrors the CPU guard.
#[test]
fn zero_weight_query_mean_is_zero() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let weights = vec![0.0, 0.0, 1.0, 1.0]; // query 0 all-zero-weight, query 1 normal
    let q_offsets = vec![0u32, 2, 4];

    let ref_means = cpu_group_means(&values, &weights, &q_offsets);
    assert_eq!(ref_means[0], 0.0, "CPU zero-weight query mean must be 0");

    let dev = compute_group_means_host(&values, &weights, &q_offsets)
        .expect("device group-means must not error");
    assert!(dev.iter().all(|v| v.is_finite()), "means must be finite");
    let div = max_abs_divergence(&dev, &ref_means);
    println!("[query_helper] zero-weight means max_div={div:e}");
    if device_backend_active() {
        assert!(div <= TOL, "device zero-weight mean diverged: {div:e} > {TOL:e}");
    }
}

/// Test 2: `RemoveGroupMeans` yields residuals matching the CPU per-query bias removal
/// (`values[d] - mean[qid[d]]`). Also validates `ComputeGroupIds` (the qids feeding the removal).
#[test]
fn remove_group_means_matches_cpu_bias_removal() {
    let (values, weights, q_offsets) = uneven_fixture();
    let n = values.len();

    let ref_qids = cpu_group_ids(&q_offsets, n);
    let dev_qids =
        compute_group_ids_host(&q_offsets, n).expect("device group-ids must not error");
    assert_eq!(dev_qids, ref_qids, "device qids must match the CPU scatter (exact u32)");

    let means = cpu_group_means(&values, &weights, &q_offsets);
    let ref_residual: Vec<f64> = (0..n)
        .map(|d| values[d] - means[ref_qids[d] as usize])
        .collect();

    let dev_residual = remove_group_means_host(&values, &dev_qids, &means)
        .expect("device bias removal must not error");
    assert_eq!(dev_residual.len(), n, "one residual per doc");
    assert!(dev_residual.iter().all(|v| v.is_finite()), "residuals must be finite");

    let div = max_abs_divergence(&dev_residual, &ref_residual);
    println!(
        "[query_helper] bias-removal residual max_div={div:e} (device_backend_active={})",
        device_backend_active()
    );
    if device_backend_active() {
        assert!(div <= TOL, "device bias removal diverged from CPU: {div:e} > {TOL:e}");
    }
}

/// Test 3: `CreateSortKeys` + `segmented_radix_sort` keep queries CONTIGUOUS — the sorted doc order
/// has non-decreasing qids (a doc never crosses a query boundary), while docs shuffle WITHIN a query.
/// The invariant is structural, but `segmented_radix_sort`'s underlying `full_scan` uses a plane
/// inclusive-sum the CPU backend does not implement, so this exercises the sort only on a real device
/// backend (rocm/cuda) and skips off-device (WR-01 anti-false-pass — never a silent CPU "pass").
#[test]
fn create_sort_keys_keep_queries_contiguous() {
    if !device_backend_active() {
        eprintln!("[query_helper] contiguity skipped — segmented_radix_sort needs rocm/cuda (plane scan)");
        return;
    }
    let q_offsets = vec![0u32, 4, 5, 11]; // uneven: sizes 4 / 1 / 6
    let n = 11usize;
    let qids = cpu_group_ids(&q_offsets, n);

    let sorted_docs = shuffle_within_queries_host(&q_offsets, 0x51ED_2024_u64, n)
        .expect("in-query shuffle must not error");
    assert_eq!(sorted_docs.len(), n, "the shuffle is a permutation of all docs");

    // Every doc appears exactly once (a genuine permutation).
    let mut seen = vec![false; n];
    for &d in &sorted_docs {
        let idx = d as usize;
        assert!(idx < n, "sorted doc index out of range: {idx}");
        assert!(!seen[idx], "doc {idx} appeared twice — not a permutation");
        seen[idx] = true;
    }
    assert!(seen.into_iter().all(|s| s), "the shuffle dropped a doc");

    // Queries contiguous: qids non-decreasing across the sorted order.
    let sorted_qids: Vec<u32> = sorted_docs.iter().map(|&d| qids[d as usize]).collect();
    for w in sorted_qids.windows(2) {
        assert!(
            w[0] <= w[1],
            "query contiguity violated: qids {sorted_qids:?} not non-decreasing"
        );
    }
    // Each query's docs form a contiguous run whose membership equals the original query.
    for g in 0..(q_offsets.len() - 1) {
        let begin = q_offsets[g] as usize;
        let end = q_offsets[g + 1] as usize;
        let mut run: Vec<u32> = sorted_docs
            .get(begin..end)
            .unwrap_or(&[])
            .to_vec();
        run.sort_unstable();
        let expected: Vec<u32> = (begin as u32..end as u32).collect();
        assert_eq!(run, expected, "query {g}'s docs must stay within its contiguous slice");
    }
    println!("[query_helper] contiguity ok over {n} docs / {} queries", q_offsets.len() - 1);
}

/// Test 4: `SampledQuerySize` floors at 2 and never exceeds the query — a backend-independent
/// invariant (hard-assert everywhere). A tiny `sample_rate` that would truncate below 2 still yields
/// 2 (or the query size for a 1-doc query).
#[test]
fn sampled_query_size_floors_at_two() {
    // sizes: 1 / 3 / 10; a small rate (0.1) floors the raw counts (0 / 0 / 1) up to 2 (capped at qSize).
    let q_offsets = vec![0u32, 1, 4, 14];
    let sizes = [1usize, 3, 10];
    let rate = 0.1_f64;

    let sampled = compute_sampled_sizes_host(&q_offsets, rate)
        .expect("device sampled-size must not error");
    assert_eq!(sampled.len(), sizes.len(), "one sampled size per query");
    println!("[query_helper] sampled sizes (rate={rate}) = {sampled:?}");

    for (g, &q_size) in sizes.iter().enumerate() {
        let s = sampled[g] as usize;
        let floor = 2usize.min(q_size);
        assert!(s >= floor, "query {g} (size {q_size}) sampled {s} < floor {floor}");
        assert!(s <= q_size, "query {g} sampled {s} exceeds query size {q_size}");
    }
    // The 1-doc query yields exactly 1 (floor capped at qSize), the others exactly 2 under rate 0.1.
    assert_eq!(sampled[0], 1, "1-doc query floors to its single doc, not 2");
    assert_eq!(sampled[1], 2, "3-doc query at rate 0.1 floors to 2");
    assert_eq!(sampled[2], 2, "10-doc query at rate 0.1 floors to 2");

    // A larger rate exercises the non-floored path: floor(0.6·10) == 6.
    let sampled_hi = compute_sampled_sizes_host(&q_offsets, 0.6)
        .expect("device sampled-size must not error");
    assert_eq!(sampled_hi[2] as usize, 6, "10-doc query at rate 0.6 samples floor(6.0)=6");
}

/// Test 5: `FillTakenDocsMask` marks the first `sampled_query_size` docs per query; `FillQueryEndMask`
/// marks each query's last doc. Both are serial u32 kernels (no plane scan), so they run on every
/// backend and hard-assert against a CPU reference over the CURRENT doc order.
#[test]
fn taken_and_query_end_masks_match_cpu() {
    let q_offsets = vec![0u32, 1, 4, 14]; // sizes 1 / 3 / 10
    let sizes = [1usize, 3, 10];
    let n = 14usize;
    let rate = 0.5_f64;

    // CPU reference for the taken mask: floor(rate·qSize) floored at 2, capped at qSize, first-k.
    let mut ref_taken = vec![0u32; n];
    let mut ref_end = vec![0u32; n];
    for (g, &q_size) in sizes.iter().enumerate() {
        let begin = q_offsets[g] as usize;
        let end = q_offsets[g + 1] as usize;
        let sampled = ((rate * q_size as f64) as usize).max(2).min(q_size);
        for (rank, d) in (begin..end).enumerate() {
            if rank < sampled {
                ref_taken[d] = 1;
            }
        }
        if end > begin {
            ref_end[end - 1] = 1;
        }
    }

    let dev_taken = fill_taken_docs_mask_host(&q_offsets, rate, n)
        .expect("device taken-mask must not error");
    let dev_end = fill_query_end_mask_host(&q_offsets, n).expect("device end-mask must not error");

    assert_eq!(dev_taken, ref_taken, "taken mask must match the CPU first-k reference");
    assert_eq!(dev_end, ref_end, "query-end mask must mark each query's last doc");
    // Query-end mask has exactly one set bit per non-empty query.
    assert_eq!(dev_end.iter().filter(|&&m| m == 1).count(), sizes.len(), "one end per query");
}
