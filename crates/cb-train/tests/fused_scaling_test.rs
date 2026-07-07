//! PHASE 21.5 — CPU parallel-scaling proof for the FUSED feature-parallel histogram.
//!
//! This is the phase's headline evidence harness. Plans 02/03 fused the per-feature
//! histogram accumulate + `O(n_bins)` scan-score into ONE `rayon` parallel-over-features
//! pass inside `select_level_plain` / `select_level_perturbed` (the Amdahl fix, Spike
//! 006), byte-for-byte preserving parity. This test PROVES the recovered multi-core
//! scaling:
//!
//!   * PRIMARY (hard gate) — measure the ISOLATED per-level fused pass (the exact
//!     accumulate+score work `select_level_plain` now performs, driven through the
//!     production `cb_compute::fused_feature_scan_and_score` +
//!     `FusedFeatureScratch` map_init primitive) across sized rayon pools of
//!     1/2/4/8/16 threads on the Spike-002 baseline shape (n=10000, nf=20, nbins=128),
//!     and `assert!` the 16-thread speedup is >= 3.0 (Spike 006 measured ~5.0x).
//!   * SECONDARY (documented, not gated) — drive end-to-end `train()` per-tree time in
//!     the same sized pools at depth 2/4/6, printing the whole-tree-grow 1->16-thread
//!     curve (it still includes remaining out-of-scope serial phases — binning setup,
//!     leaf values — so it is REPORTED, not gated).
//!
//! INERT by default: real work ONLY when `CB_PERF` is set. Always run `--release`.
//! Greppable records are printed with the `RSBENCH215` prefix.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use rayon::prelude::*;

/// A sized rayon thread pool (the sweep unit).
fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build rayon pool")
}

/// Same splitmix64 generator as `spike006_fused_parallel_test.rs` /
/// `perf_baseline_test.rs` (row-for-row comparable): feature-major bins
/// (`bins[f * n + i]`), a per-object der1 (dim=1), unit weights, and a mid-tree
/// partition spread across `n_leaves`.
fn make_inputs(
    n: usize,
    nf: usize,
    nbins: usize,
    n_leaves: usize,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut bins = vec![0u32; nf * n];
    let mut der1 = vec![0.0f64; n];
    for f in 0..nf {
        for i in 0..n {
            let mut z = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((f as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let v = (z >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            bins[f * n + i] = ((v * nbins as f64) as usize).min(nbins - 1) as u32;
            if f < 5 {
                der1[i] += v * if f % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    let weight = vec![1.0f64; n];
    let leaf_of: Vec<usize> = (0..n).map(|i| i % n_leaves).collect();
    (bins, der1, weight, leaf_of)
}

/// Median-of-`reps` wall time in ms for a closure (warmup handled by the caller).
fn median_ms<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    use std::time::Instant;
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// `true` when `CB_PERF` is set; otherwise the caller prints SKIP and returns (so
/// `cargo test` without `CB_PERF` stays fast and green).
fn perf_enabled() -> bool {
    std::env::var("CB_PERF").is_ok()
}

#[test]
fn fused_scaling_curve() {
    if !perf_enabled() {
        eprintln!(
            "SKIP fused_scaling_curve: set CB_PERF=1 (and run --release) to measure the \
             fused per-level scaling curve"
        );
        return;
    }
    // Task 2 fills in the isolated per-level sweep (hard gate: 16t speedup >= 3.0) and
    // the end-to-end per-tree curve, then writes 21.5-EVIDENCE.md. The shared harness
    // (make_inputs / pool / median_ms) lives above.
    let cores = std::thread::available_parallelism().map_or(0, |c| c.get());
    println!("RSBENCH215_META cores={cores}");

    // Smoke: the harness builds valid inputs + a sized pool (scaffold self-check).
    let (bins, der1, weight, leaf_of) = make_inputs(10_000, 20, 128, 32);
    assert_eq!(bins.len(), 20 * 10_000);
    let _ = median_ms(1, || {});
    pool(2).install(|| {
        let s: u64 = (0..bins.len()).into_par_iter().map(|i| bins[i] as u64).sum();
        assert!(s > 0);
        assert_eq!(der1.len(), leaf_of.len());
        assert_eq!(weight.len(), leaf_of.len());
    });
    println!("RSBENCH215_SCAFFOLD_OK");
}
