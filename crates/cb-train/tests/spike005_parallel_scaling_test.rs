//! SPIKE 005 (parallel-scaling-root-cause): proves WHY catboost-rs only gets
//! ~1.5-1.7x from 1->16 threads (vs CatBoost's 2.0-3.0x) now that the histogram
//! rewrite (Phase 21) closed the per-core algorithmic gap.
//!
//! Three parts, all printing greppable `RSBENCH005 ...`:
//!   A. end-to-end scaling curve  — train inside local rayon pools of 1/2/4/8/16
//!      threads across a depth sweep; report per_tree_ms + speedup vs 1 thread.
//!   B. phase split               — microbench the SERIAL `build_bucket_histogram`
//!      accumulation vs the PARALLEL per-feature `scan_and_score_borders` scoring;
//!      report the serial fraction and the Amdahl 16-thread ceiling it implies.
//!   C. parity escape-hatch PoC   — build the histogram FEATURE-PARALLEL and assert
//!      the result is BYTE-IDENTICAL to the serial build (no oracle re-baseline),
//!      while timing it to show the serial phase parallelizes.
//!
//! INERT by default: real work ONLY when `CB_PERF` is set. Always run `--release`.
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
    build_bucket_histogram, rmse_der1, rmse_der2, scan_and_score_borders, Derivatives,
    EScoreFunction, LeafMethod, Loss, Runtime,
};
use cb_core::CbResult;
use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

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

/// Same splitmix64 generator as `perf_baseline_test.rs` (row-for-row comparable).
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

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build rayon pool")
}

// --------------------------------------------------------------------------
// Part A — end-to-end scaling curve
// --------------------------------------------------------------------------
fn part_a() {
    let (n, nf, nbins, iters) = (10_000usize, 20usize, 128usize, 8usize);
    println!("RSBENCH005 part=A_meta n={n} nf={nf} nbins={nbins} iters={iters} threads_swept=1,2,4,8,16");
    for &depth in &[2usize, 4, 6] {
        let (cols, borders, target) = gen(n, nf, nbins);
        let p = params(depth, iters);
        let mut base_ptm = 0.0_f64;
        for &threads in &[1usize, 2, 4, 8, 16] {
            let pl = pool(threads);
            // one warmup tree-set (JIT caches / allocator warm), then timed.
            let _ = pl.install(|| train(&CpuHostRuntime, &cols, &borders, &target, &[], &p, None));
            let t = Instant::now();
            let m = pl
                .install(|| train(&CpuHostRuntime, &cols, &borders, &target, &[], &p, None))
                .unwrap_or_else(|e| panic!("train failed: {e:?}"));
            let secs = t.elapsed().as_secs_f64();
            let ptm = secs * 1000.0 / iters as f64;
            if threads == 1 {
                base_ptm = ptm;
            }
            let speedup = if ptm > 0.0 { base_ptm / ptm } else { 0.0 };
            println!(
                "RSBENCH005 part=A depth={depth} threads={threads} \
                 per_tree_ms={ptm:.3} speedup={speedup:.2} trees={}",
                m.oblivious_trees.len()
            );
        }
    }
}

// --------------------------------------------------------------------------
// Synthetic histogram inputs (feature-major bins + der1 + weight + leaf_of).
// --------------------------------------------------------------------------
fn make_hist_inputs(
    n: usize,
    nf: usize,
    nbins: usize,
    n_leaves: usize,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let (cols, _borders, target) = gen(n, nf, nbins);
    // feature-major bins: bins[feature * n + obj]
    let mut bins = vec![0u32; nf * n];
    for f in 0..nf {
        for i in 0..n {
            let b = ((cols[f][i] as f64) * nbins as f64) as usize;
            bins[f * n + i] = b.min(nbins - 1) as u32;
        }
    }
    // dim = 1: der1 is per-object; use the target residual magnitudes.
    let der1: Vec<f64> = target.clone();
    let weight: Vec<f64> = vec![1.0; n];
    // mid-tree partition: spread objects across n_leaves deterministically.
    let leaf_of: Vec<usize> = (0..n).map(|i| i % n_leaves).collect();
    (bins, der1, weight, leaf_of)
}

// --------------------------------------------------------------------------
// Part B — serial-accumulate vs parallel-score phase split.
// --------------------------------------------------------------------------
fn part_b() {
    let (n, nf, nbins, n_leaves, dim) = (10_000usize, 20usize, 128usize, 32usize, 1usize);
    let reps = 40usize;
    let (bins, der1, weight, leaf_of) = make_hist_inputs(n, nf, nbins, n_leaves);
    let scaled_l2 = 3.0_f64;

    // SERIAL accumulation (the rayon-free build_bucket_histogram).
    let t = Instant::now();
    let mut hist = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim);
    for _ in 1..reps {
        hist = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim);
    }
    let t_build_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    // PARALLEL-pass WORK, measured single-threaded (sum over features of the scan/score).
    let t = Instant::now();
    for _ in 0..reps {
        for f in 0..nf {
            let s = scan_and_score_borders(&hist, f, nbins - 1, dim, EScoreFunction::L2, scaled_l2);
            black_box(&s);
        }
    }
    let t_score_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let f_serial = t_build_ms / (t_build_ms + t_score_ms);
    let amdahl16 = 1.0 / (f_serial + (1.0 - f_serial) / 16.0);
    println!(
        "RSBENCH005 part=B n={n} nf={nf} nbins={nbins} n_leaves={n_leaves} \
         t_build_ms={t_build_ms:.3} t_score_ms={t_score_ms:.3} \
         serial_fraction={f_serial:.3} amdahl_ceiling_16={amdahl16:.2}"
    );
}

// --------------------------------------------------------------------------
// Part C — feature-parallel histogram: byte-identical + faster.
// --------------------------------------------------------------------------
/// Feature-outer / object-inner parallel build. Each feature accumulates its own
/// cells in ASCENDING OBJECT ORDER within a single thread, so every cell's
/// `+=` fold order is identical to the serial object-outer build => bit-identical.
fn feature_parallel_build(
    bins: &[u32],
    der1: &[f64],
    weight: &[f64],
    leaf_of: &[usize],
    n_leaves: usize,
    nf: usize,
    nbins: usize,
    dim: usize,
) -> Vec<f64> {
    let n = leaf_of.len();
    let n_channels = dim + 1;
    // Per-feature compact buffers (only this feature's (leaf,bin,channel) cells).
    let compact: Vec<Vec<f64>> = (0..nf)
        .into_par_iter()
        .map(|feature| {
            let mut buf = vec![0.0_f64; n_leaves * nbins * n_channels];
            for obj in 0..n {
                let leaf = leaf_of[obj];
                if leaf >= n_leaves {
                    continue;
                }
                let bin = bins[feature * n + obj] as usize;
                if bin >= nbins {
                    continue;
                }
                let base = (leaf * nbins + bin) * n_channels;
                for d in 0..dim {
                    buf[base + d] += der1[d * n + obj];
                }
                buf[base + dim] += weight[obj];
            }
            buf
        })
        .collect();
    // Assemble into the frozen ((leaf*nf+feature)*nbins+bin)*n_channels layout
    // (cheap O(n_leaves*nf*nbins) — no dependence on n).
    let mut data = vec![0.0_f64; n_leaves * nf * nbins * n_channels];
    for feature in 0..nf {
        let buf = &compact[feature];
        for leaf in 0..n_leaves {
            for bin in 0..nbins {
                for c in 0..n_channels {
                    let dst = ((leaf * nf + feature) * nbins + bin) * n_channels + c;
                    let src = (leaf * nbins + bin) * n_channels + c;
                    data[dst] = buf[src];
                }
            }
        }
    }
    data
}

fn part_c() {
    let (n, nf, nbins, n_leaves, dim) = (10_000usize, 20usize, 128usize, 32usize, 1usize);
    let reps = 40usize;
    let (bins, der1, weight, leaf_of) = make_hist_inputs(n, nf, nbins, n_leaves);
    let n_channels = dim + 1;

    let serial = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim);

    // Parity: every cell byte-identical (exact f64 equality is the whole claim).
    let par = feature_parallel_build(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim);
    let mut identical = true;
    let mut first_mismatch = (usize::MAX, usize::MAX, usize::MAX, usize::MAX);
    'outer: for leaf in 0..n_leaves {
        for feature in 0..nf {
            for bin in 0..nbins {
                for c in 0..n_channels {
                    let dst = ((leaf * nf + feature) * nbins + bin) * n_channels + c;
                    let got = par[dst];
                    let exp = serial.channel(leaf, feature, bin, c);
                    if got.to_bits() != exp.to_bits() {
                        identical = false;
                        first_mismatch = (leaf, feature, bin, c);
                        break 'outer;
                    }
                }
            }
        }
    }
    assert!(
        identical,
        "feature-parallel histogram diverged at (leaf,feature,bin,channel)={first_mismatch:?}"
    );

    // Timing: serial vs feature-parallel on 16 threads.
    let t = Instant::now();
    for _ in 0..reps {
        let h = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim);
        black_box(&h);
    }
    let t_serial_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let pl = pool(16);
    let t = Instant::now();
    pl.install(|| {
        for _ in 0..reps {
            let h = feature_parallel_build(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim);
            black_box(&h);
        }
    });
    let t_par_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    let speedup = if t_par_ms > 0.0 { t_serial_ms / t_par_ms } else { 0.0 };

    println!(
        "RSBENCH005 part=C n={n} nf={nf} nbins={nbins} n_leaves={n_leaves} \
         parity_byte_identical={identical} t_serial_ms={t_serial_ms:.3} \
         t_featurepar16_ms={t_par_ms:.3} speedup={speedup:.2}"
    );
}

#[test]
fn spike005_parallel_scaling() {
    if std::env::var("CB_PERF").is_err() {
        eprintln!("SKIP spike005_parallel_scaling: set CB_PERF=1 to run the parallel-scaling diagnostic");
        return;
    }
    println!("RSBENCH005_META cores={}", std::thread::available_parallelism().map_or(0, |c| c.get()));
    part_a();
    part_b();
    part_c();
    println!("RSBENCH005_DONE");
}
