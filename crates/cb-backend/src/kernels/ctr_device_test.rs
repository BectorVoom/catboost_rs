//! Self-oracle for the device CTR accumulation (Phase 12 Plan 08, GPUT-10, Pattern F): the device
//! ordered / one-hot / tensor CTR column ([`crate::kernels::ctr_device`]) must reproduce a FROZEN
//! CPU reference — an inline serial transcription of `cb_train::ctr::online::online_ctr_prefix_binclf`
//! (read-before-increment, object-order output) + `calc_ctr_online` — within the ε=1e-4 device bar,
//! and the CTR→cindex binarize JOIN must reproduce the host binarization bit-exact.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernels live in the
//! production `kernels::ctr_device` module; ALL assertions / `.unwrap()` / indexing live here. The
//! CPU reference is TRANSCRIBED inline (NOT `use cb_train` — the landmine is cb-TRAIN, the whole
//! reason cb-backend must not depend on it), so this is a genuine device-`#[cube]`-vs-independent-
//! host-scan oracle, not a tautology.
//!
//! Runs over [`crate::SelectedRuntime`]. Unlike the resident-grow oracles, the ordered CTR is a
//! SERIAL SCAN with EXACT INTEGER prefix counting (no `Atomic<u64>`, no resident histogram), so it
//! executes IN-ENV on the default cpu backend as well as rocm/cuda. Only the wgpu backend (no f64)
//! cannot run the f64 CTR value seam — the assertions SKIP there (WR-01 anti-false-pass).

use crate::kernels::ctr_device::{
    binarize_ctr_column_host, combine_projection_bins, compute_ordered_ctr_host,
};

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device f64 CTR seam runs on this backend (cpu/rocm/cuda have f64; wgpu does not).
fn device_ctr_active() -> bool {
    !cfg!(feature = "wgpu")
}

/// Deterministic pseudo-random `u32` stream (LCG — no `rand` dep), matching the cindex oracle's
/// generator so the fixtures are reproducible.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state >> 8
}

/// A deterministic categorical bin column (`obj -> bucket in 0..cardinality`) + a binclf class
/// column + a learn permutation, all in object order, from one seed.
fn synth_fixture(n: usize, cardinality: u32, seed: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let bins: Vec<u32> = (0..n).map(|_| lcg(&mut s) % cardinality).collect();
    let class: Vec<u32> = (0..n).map(|_| lcg(&mut s) % 2).collect();
    // A non-identity learn permutation (rotate + swap) so read-order matters.
    let mut perm: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        let j = (lcg(&mut s) as usize) % n;
        perm.swap(i, j);
    }
    (bins, class, perm)
}

/// The FROZEN CPU ordered CTR reference: an inline serial transcription of
/// `online_ctr_prefix_binclf` (read-before-increment, object order) + `calc_ctr_online`
/// (`(good + prior) / (total + 1)`). Returns `(good, total, value)` in OBJECT order.
fn cpu_ordered_ctr(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let n = perm.len();
    let bucket_count = bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut counts = vec![[0i64; 2]; bucket_count];
    let mut good = vec![0i64; n];
    let mut total = vec![0i64; n];
    let mut value = vec![0f64; n];
    for &doc_i in perm {
        let doc = doc_i as usize;
        let bucket = bins[doc] as usize;
        // READ before increment (the no-leakage invariant).
        let n0 = counts[bucket][0];
        let n1 = counts[bucket][1];
        let g = n1;
        let t = n0 + n1;
        good[doc] = g;
        total[doc] = t;
        value[doc] = (g as f64 + prior) / (t as f64 + 1.0);
        counts[bucket][class[doc] as usize] += 1;
    }
    (good, total, value)
}

/// `max |a - b|` over two equal-length f64 vectors; `INFINITY` on a length mismatch so a truncated
/// device buffer fails loudly (WR-06, mirroring `grow_loop::max_divergence`).
fn max_divergence(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The bucket count `max(bins) + 1` a device launch must size its scratch to.
fn bucket_count(bins: &[u32]) -> usize {
    bins.iter().copied().max().map_or(0, |m| m as usize + 1)
}

#[test]
fn ordered_ts_matches_cpu_reference() {
    // Ordered target statistic (the headline): the device read-before-increment prefix column
    // must reproduce the frozen CPU reference — good/total EXACT integers, value ≤1e-4.
    if !device_ctr_active() {
        eprintln!("SKIP ordered_ts_matches_cpu_reference: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 400usize;
    let prior = 0.5;
    for &seed in &[1u32, 42, 12345] {
        let (bins, class, perm) = synth_fixture(n, 7, seed);
        let (cg, ct, cv) = cpu_ordered_ctr(&perm, &bins, &class, prior);
        let (dg, dt, dv) =
            compute_ordered_ctr_host(&perm, &bins, &class, prior, bucket_count(&bins)).unwrap();
        assert_eq!(dg, cg, "device good count != CPU (seed {seed})");
        assert_eq!(dt, ct, "device total count != CPU (seed {seed})");
        let div = max_divergence(&dv, &cv);
        assert!(div <= TOL, "ordered CTR value divergence {div} > {TOL} (seed {seed})");
    }
}

#[test]
fn first_doc_in_segment_reads_prior() {
    // The FIRST document to land in each bucket (in permutation order) reads the PRIOR alone —
    // good = total = 0, value = prior / 1 — never its own label (the no-leakage invariant).
    if !device_ctr_active() {
        eprintln!("SKIP first_doc_in_segment_reads_prior: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 200usize;
    let prior = 0.5;
    let (bins, class, perm) = synth_fixture(n, 5, 777);
    let (dg, dt, dv) =
        compute_ordered_ctr_host(&perm, &bins, &class, prior, bucket_count(&bins)).unwrap();

    // Walk the permutation, tracking each bucket's first appearance.
    let mut seen = vec![false; bucket_count(&bins)];
    for &doc_i in &perm {
        let doc = doc_i as usize;
        let bucket = bins[doc] as usize;
        if !seen[bucket] {
            seen[bucket] = true;
            assert_eq!(dg[doc], 0, "first doc in bucket {bucket} must read good=0");
            assert_eq!(dt[doc], 0, "first doc in bucket {bucket} must read total=0");
            assert!(
                (dv[doc] - prior).abs() <= TOL,
                "first doc in bucket {bucket} must read value=prior ({prior}), got {}",
                dv[doc]
            );
        }
    }
}

#[test]
fn one_hot_ctr_matches_cpu_reference() {
    // One-hot regime: a SMALL-cardinality categorical (each category its own bucket) rides the
    // SAME device prefix kernel as the ordered TS (A5 — only the bucket source differs).
    if !device_ctr_active() {
        eprintln!("SKIP one_hot_ctr_matches_cpu_reference: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 256usize;
    let prior = 1.0;
    let (bins, class, perm) = synth_fixture(n, 3, 9091); // cardinality 3 => one-hot regime
    let (cg, ct, cv) = cpu_ordered_ctr(&perm, &bins, &class, prior);
    let (dg, dt, dv) =
        compute_ordered_ctr_host(&perm, &bins, &class, prior, bucket_count(&bins)).unwrap();
    assert_eq!(dg, cg, "one-hot device good != CPU");
    assert_eq!(dt, ct, "one-hot device total != CPU");
    assert!(max_divergence(&dv, &cv) <= TOL, "one-hot CTR value divergence > {TOL}");
}

#[test]
fn tensor_combination_ctr_matches_cpu_reference() {
    // Tensor / feature-combination CTR (A5): TWO cat columns fold into one combined-projection
    // bin column (device-host `combine_projection_bins`), then the SAME ordered prefix runs on the
    // combined bins. Both the device path and the CPU reference consume the SAME combined bins, so
    // the oracle validates the ordered-CTR-over-combined-bins MATH (the exact upstream combined_hash
    // parity is the Kaggle CUDA sign-off, Plan 09). Identical (categoryA, categoryB) pairs must
    // share the SAME combined bucket (asserted below).
    if !device_ctr_active() {
        eprintln!("SKIP tensor_combination_ctr_matches_cpu_reference: wgpu has no f64 seam (WR-01)");
        return;
    }
    let n = 300usize;
    let prior = 0.5;
    let (col_a, class, perm) = synth_fixture(n, 4, 202);
    let (col_b, _c2, _p2) = synth_fixture(n, 3, 606);
    let (combined, buckets) = combine_projection_bins(&[col_a.clone(), col_b.clone()], n).unwrap();
    assert_eq!(combined.len(), n);
    // Same (a, b) pair => same combined bin (canonical CTR per category-within-combination).
    for i in 0..n {
        for j in 0..n {
            if col_a[i] == col_a[j] && col_b[i] == col_b[j] {
                assert_eq!(combined[i], combined[j], "same (a,b) pair must share a combined bucket");
            }
        }
    }
    assert!(buckets <= 4 * 3, "combined buckets bounded by the cartesian product");

    let (cg, ct, cv) = cpu_ordered_ctr(&perm, &combined, &class, prior);
    let (dg, dt, dv) = compute_ordered_ctr_host(&perm, &combined, &class, prior, buckets).unwrap();
    assert_eq!(dg, cg, "tensor device good != CPU");
    assert_eq!(dt, ct, "tensor device total != CPU");
    assert!(max_divergence(&dv, &cv) <= TOL, "tensor CTR value divergence > {TOL}");
}

#[test]
fn ctr_binarized_cindex_column_bit_exact() {
    // The CTR→cindex binarize JOIN: the device binarizes the accumulated CTR VALUES into an extra
    // cindex bin column (`bin = #{borders < value}`, the `> bin` convention the histogram loop
    // reads). It must reproduce the host binarization of the CPU CTR column BIT-EXACT (integer
    // equality — tighter than the ε=1e-4 float bar, mirroring the cindex D-07 discipline).
    if !device_ctr_active() {
        eprintln!("SKIP ctr_binarized_cindex_column_bit_exact: wgpu has no f64 seam (WR-01)");
        return;
    }
    let n = 300usize;
    let prior = 0.5;
    let (bins, class, perm) = synth_fixture(n, 6, 314);
    // CTR values lie in (0, 1] for these priors; borders split that range into cindex buckets.
    let borders = vec![0.2_f64, 0.4, 0.5, 0.6, 0.8];

    // Host reference: CPU CTR column, then binarize each value against the borders.
    let (_cg, _ct, cv) = cpu_ordered_ctr(&perm, &bins, &class, prior);
    let host_cindex: Vec<u32> = cv
        .iter()
        .map(|&v| borders.iter().filter(|&&b| v > b).count() as u32)
        .collect();

    // Device: accumulate CTR on device, binarize on device, read the extra cindex column back.
    let dev_cindex =
        binarize_ctr_column_host(&perm, &bins, &class, prior, bucket_count(&bins), &borders).unwrap();

    assert_eq!(dev_cindex.len(), n, "device CTR cindex column truncated");
    assert_eq!(
        dev_cindex, host_cindex,
        "device CTR->cindex binarization must be bit-exact vs the host CTR column"
    );
    // Every emitted bin is a valid index into the borders+1 buckets (bounds sanity).
    assert!(
        dev_cindex.iter().all(|&b| (b as usize) <= borders.len()),
        "a CTR cindex bin exceeds the borders+1 bucket count"
    );
}
