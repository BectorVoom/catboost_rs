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
//!     production `cb_compute::fused_feature_scan_and_score` + `FusedFeatureScratch`
//!     `map_init` primitive) across sized rayon pools of 1/2/4/8/16 threads on the
//!     Spike-002 baseline shape (n=10000, nf=20, nbins=128), and `assert!` the
//!     16-thread speedup is >= 3.0 (Spike 006 measured ~5.0x). The two-pass shape
//!     (serial whole-partition build, then parallel score — the pre-fusion tree.rs
//!     hot path) is measured alongside for the before/after contrast (~1.7x ceiling).
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

use cb_compute::{
    build_bucket_histogram, fused_feature_scan_and_score, rmse_der1, rmse_der2,
    scan_and_score_borders, Derivatives, EScoreFunction, FusedFeatureScratch, LeafMethod, Loss,
    Runtime,
};
use cb_core::CbResult;
use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const THREAD_SWEEP: [usize; 5] = [1, 2, 4, 8, 16];

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
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

// --------------------------------------------------------------------------
// PRIMARY — the isolated per-level FUSED pass (production primitive).
// --------------------------------------------------------------------------
/// Drives EXACTLY the accumulate+score work `select_level_plain` performs per level:
/// `(0..nf).into_par_iter().map_init(FusedFeatureScratch::new, |scratch, f| build the
/// 1-feature histogram over that feature's contiguous feature-major column + scan-score
/// its borders)`. This is the production `fused_feature_scan_and_score` primitive from
/// Plan 01 that `tree.rs` is built on — the direct measurement of the parallel region.
#[allow(clippy::too_many_arguments)]
fn run_fused_level(
    bins: &[u32],
    der1: &[f64],
    weight: &[f64],
    leaf_of: &[usize],
    n_leaves: usize,
    nf: usize,
    n: usize,
    nbins: usize,
    dim: usize,
    sf: EScoreFunction,
    l2: f64,
) -> Vec<Vec<f64>> {
    (0..nf)
        .into_par_iter()
        .map_init(FusedFeatureScratch::new, |scratch, f| {
            let col = &bins[f * n..(f + 1) * n];
            fused_feature_scan_and_score(
                scratch, col, der1, weight, leaf_of, n_leaves, nbins, dim, nbins - 1, sf, l2,
            )
            .to_vec()
        })
        .collect()
}

/// The PRE-FUSION tree.rs hot path (Spike 005 shape): serial whole-partition
/// `build_bucket_histogram` over ALL features (rayon-free accumulation OUTSIDE the
/// parallel region), then parallel-per-feature `scan_and_score_borders`. Measured for
/// the before/after contrast — its 16-thread speedup is the ~1.7x Amdahl ceiling.
#[allow(clippy::too_many_arguments)]
fn run_two_pass(
    bins: &[u32],
    der1: &[f64],
    weight: &[f64],
    leaf_of: &[usize],
    n_leaves: usize,
    nf: usize,
    nbins: usize,
    dim: usize,
    sf: EScoreFunction,
    l2: f64,
) -> Vec<Vec<f64>> {
    let hist = build_bucket_histogram(bins, der1, weight, leaf_of, n_leaves, nf, nbins, dim);
    (0..nf)
        .into_par_iter()
        .map(|f| scan_and_score_borders(&hist, f, nbins - 1, dim, sf, l2))
        .collect()
}

/// Measure the isolated per-level pass across the thread sweep and return the fused
/// 16-thread speedup (the hard gate). Prints greppable `RSBENCH215` records.
fn primary_per_level_sweep() -> f64 {
    let (n, nf, nbins, n_leaves, dim) = (10_000usize, 20usize, 128usize, 32usize, 1usize);
    let sf = EScoreFunction::L2;
    let l2 = 3.0f64;
    let reps = 41usize;
    let (bins, der1, weight, leaf_of) = make_inputs(n, nf, nbins, n_leaves);
    println!("RSBENCH215 part=per_level_meta n={n} nf={nf} nbins={nbins} n_leaves={n_leaves} dim={dim} reps={reps}");

    // Parity self-check: the fused per-level pass is byte-identical to two-pass.
    {
        let fused = run_fused_level(&bins, &der1, &weight, &leaf_of, n_leaves, nf, n, nbins, dim, sf, l2);
        let two = run_two_pass(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim, sf, l2);
        assert_eq!(fused.len(), two.len(), "feature count mismatch");
        for f in 0..nf {
            assert_eq!(fused[f].len(), two[f].len(), "border count mismatch at feature {f}");
            for b in 0..fused[f].len() {
                assert_eq!(
                    fused[f][b].to_bits(),
                    two[f][b].to_bits(),
                    "fused per-level score diverged from two-pass at (feature={f}, border={b})"
                );
            }
        }
        println!("RSBENCH215 part=per_level_parity byte_identical=true features={nf}");
    }

    let mut fused_base = 0.0f64;
    let mut two_base = 0.0f64;
    let mut fused_16 = 0.0f64;
    for &threads in &THREAD_SWEEP {
        let pl = pool(threads);
        // warmup both inside this pool
        pl.install(|| {
            black_box(run_fused_level(&bins, &der1, &weight, &leaf_of, n_leaves, nf, n, nbins, dim, sf, l2));
            black_box(run_two_pass(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim, sf, l2));
        });
        let fused_ms = pl.install(|| {
            median_ms(reps, || {
                black_box(run_fused_level(&bins, &der1, &weight, &leaf_of, n_leaves, nf, n, nbins, dim, sf, l2));
            })
        });
        let two_ms = pl.install(|| {
            median_ms(reps, || {
                black_box(run_two_pass(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim, sf, l2));
            })
        });
        if threads == 1 {
            fused_base = fused_ms;
            two_base = two_ms;
        }
        let fused_sp = if fused_ms > 0.0 { fused_base / fused_ms } else { 0.0 };
        let two_sp = if two_ms > 0.0 { two_base / two_ms } else { 0.0 };
        if threads == 16 {
            fused_16 = fused_sp;
        }
        println!(
            "RSBENCH215 part=per_level threads={threads} \
             fused_ms={fused_ms:.4} fused_speedup={fused_sp:.2} \
             two_pass_ms={two_ms:.4} two_pass_speedup={two_sp:.2}"
        );
    }
    fused_16
}

// --------------------------------------------------------------------------
// SECONDARY — end-to-end train() per-tree curve (documented, not gated).
// --------------------------------------------------------------------------
struct CpuHostRuntime;
impl Runtime for CpuHostRuntime {
    fn compute_gradients(
        &self,
        _loss: &Loss,
        approx: &[f64],
        target: &[f64],
        _approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        let der1 = approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect();
        let der2 = approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect();
        Ok(Derivatives { der1, der2 })
    }
}

/// Same splitmix64 columns as `spike005_parallel_scaling_test.rs::gen`.
fn gen(n: usize, nf: usize, nbins: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut cols = Vec::with_capacity(nf);
    for f in 0..nf {
        let col: Vec<f32> = (0..n)
            .map(|i| {
                let mut z = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((f as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 / (1u64 << 53) as f64) as f32
            })
            .collect();
        cols.push(col);
    }
    let borders: Vec<Vec<f64>> = (0..nf)
        .map(|_| (1..nbins).map(|k| k as f64 / nbins as f64).collect())
        .collect();
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let mut acc = 0.0;
            for f in 0..nf.min(5) {
                acc += (cols[f][i] as f64) * (if f % 2 == 0 { 1.0 } else { -1.0 });
            }
            acc
        })
        .collect();
    (cols, borders, target)
}

fn params(depth: usize, iterations: usize) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth,
        learning_rate: 0.03,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 42,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: EGrowPolicy::SymmetricTree,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

fn secondary_end_to_end_curve() {
    let (n, nf, nbins, iters) = (10_000usize, 20usize, 128usize, 8usize);
    println!("RSBENCH215 part=e2e_meta n={n} nf={nf} nbins={nbins} iters={iters} threads_swept=1,2,4,8,16");
    for &depth in &[2usize, 4, 6] {
        let (cols, borders, target) = gen(n, nf, nbins);
        let p = params(depth, iters);
        let mut base_ptm = 0.0f64;
        for &threads in &THREAD_SWEEP {
            let pl = pool(threads);
            // warmup tree-set (JIT / allocator warm), then timed.
            let _ = pl.install(|| train(&CpuHostRuntime, &cols, &borders, &target, &[], &p, None));
            let t = Instant::now();
            let m = pl
                .install(|| train(&CpuHostRuntime, &cols, &borders, &target, &[], &p, None))
                .unwrap_or_else(|e| panic!("train failed: {e:?}"));
            let ptm = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            if threads == 1 {
                base_ptm = ptm;
            }
            let speedup = if ptm > 0.0 { base_ptm / ptm } else { 0.0 };
            println!(
                "RSBENCH215 part=e2e depth={depth} threads={threads} \
                 per_tree_ms={ptm:.3} speedup={speedup:.2} trees={}",
                m.oblivious_trees.len()
            );
        }
    }
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
    let cores = std::thread::available_parallelism().map_or(0, |c| c.get());
    println!("RSBENCH215_META cores={cores}");

    // PRIMARY (hard gate): isolated per-level fused pass, 16-thread speedup >= 3.0.
    let fused_16 = primary_per_level_sweep();

    // SECONDARY (documented, not gated): end-to-end per-tree curve at depth 2/4/6.
    secondary_end_to_end_curve();

    println!("RSBENCH215_DONE fused_per_level_speedup_16t={fused_16:.2}");
    assert!(
        fused_16 >= 3.0,
        "isolated per-level fused 16-thread speedup = {fused_16:.2}x, expected >= 3.0x \
         (Spike 006 measured ~5.0x); the serial accumulation fraction did NOT collapse"
    );
}
