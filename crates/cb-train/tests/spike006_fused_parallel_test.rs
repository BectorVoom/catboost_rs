//! SPIKE 006 (fused-feature-parallel-histogram): prototypes the fix 005 pointed to.
//!
//! 005 proved the parallel-scaling ceiling is the SERIAL `build_bucket_histogram`
//! accumulation being left OUT of the parallel region (only scoring is threaded),
//! and that a feature-outer parallel build is byte-identical to the serial one.
//!
//! This spike prototypes CatBoost's `CalcStatsAndScores` shape: FUSE accumulate +
//! score into ONE `into_par_iter` over features — each task builds its own feature's
//! histogram (the O(n) column scan) AND scores it — so the accumulation is inside the
//! parallel region and the serial fraction goes to ~0. It compares, at 1/2/4/8/16
//! threads:
//!   - `two_pass`  (mimics current tree.rs): serial `build_bucket_histogram` over ALL
//!                 features, then parallel per-feature `scan_and_score_borders`.
//!   - `fused`     (the fix): parallel per-feature (build own 1-feature hist + score).
//! and asserts the fused per-border scores are BYTE-IDENTICAL to the two-pass path.
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

use cb_compute::{build_bucket_histogram, scan_and_score_borders, EScoreFunction};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build rayon pool")
}

/// Same splitmix64 generator as the perf baseline — feature-major bins + a per-object
/// der1 (dim=1) + unit weight + a mid-tree partition.
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

/// CURRENT shape: serial build over ALL features (rayon-free), then parallel score.
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
    // SERIAL accumulation (outside the parallel region).
    let hist = build_bucket_histogram(bins, der1, weight, leaf_of, n_leaves, nf, nbins, dim);
    // PARALLEL score only.
    (0..nf)
        .into_par_iter()
        .map(|f| scan_and_score_borders(&hist, f, nbins - 1, dim, sf, l2))
        .collect()
}

/// FIX shape (CalcStatsAndScores): fused accumulate+score, everything in the parallel
/// region. Each task builds its OWN 1-feature histogram (the O(n) column scan) then
/// scores it. Feature-major storage makes the column a contiguous slice, and a
/// 1-feature build folds each cell in ascending object order => byte-identical cells.
fn run_fused(
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
        .map(|f| {
            let col = &bins[f * n..(f + 1) * n];
            let h = build_bucket_histogram(col, der1, weight, leaf_of, n_leaves, 1, nbins, dim);
            scan_and_score_borders(&h, 0, nbins - 1, dim, sf, l2)
        })
        .collect()
}

fn time_ms<F: Fn()>(reps: usize, f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

fn bench_shape(n: usize, nf: usize, nbins: usize, n_leaves: usize) {
    let dim = 1usize;
    let sf = EScoreFunction::L2;
    let l2 = 3.0f64;
    let reps = 30usize;
    let (bins, der1, weight, leaf_of) = make_inputs(n, nf, nbins, n_leaves);
    println!("RSBENCH006 shape_meta n={n} nf={nf} nbins={nbins} n_leaves={n_leaves}");

    let mut two_base = 0.0;
    let mut fused_base = 0.0;
    for &threads in &[1usize, 2, 4, 8, 16] {
        let pl = pool(threads);
        // warmup both
        pl.install(|| {
            black_box(run_two_pass(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim, sf, l2));
            black_box(run_fused(&bins, &der1, &weight, &leaf_of, n_leaves, nf, n, nbins, dim, sf, l2));
        });
        let two_ms = pl.install(|| {
            time_ms(reps, || {
                black_box(run_two_pass(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim, sf, l2));
            })
        });
        let fused_ms = pl.install(|| {
            time_ms(reps, || {
                black_box(run_fused(&bins, &der1, &weight, &leaf_of, n_leaves, nf, n, nbins, dim, sf, l2));
            })
        });
        if threads == 1 {
            two_base = two_ms;
            fused_base = fused_ms;
        }
        let two_sp = if two_ms > 0.0 { two_base / two_ms } else { 0.0 };
        let fused_sp = if fused_ms > 0.0 { fused_base / fused_ms } else { 0.0 };
        println!(
            "RSBENCH006 shape=n{n}_nf{nf}_nb{nbins}_L{n_leaves} threads={threads} \
             two_pass_ms={two_ms:.3} two_pass_speedup={two_sp:.2} \
             fused_ms={fused_ms:.3} fused_speedup={fused_sp:.2}"
        );
    }
}

/// Byte-identity: fused per-border scores == two-pass per-border scores, exactly.
fn assert_parity(n: usize, nf: usize, nbins: usize, n_leaves: usize) {
    let dim = 1usize;
    let sf = EScoreFunction::L2;
    let l2 = 3.0f64;
    let (bins, der1, weight, leaf_of) = make_inputs(n, nf, nbins, n_leaves);
    let two = run_two_pass(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim, sf, l2);
    let fused = run_fused(&bins, &der1, &weight, &leaf_of, n_leaves, nf, n, nbins, dim, sf, l2);
    assert_eq!(two.len(), fused.len(), "feature count mismatch");
    let mut identical = true;
    let mut first = (usize::MAX, usize::MAX);
    'outer: for f in 0..nf {
        assert_eq!(two[f].len(), fused[f].len(), "border count mismatch at feature {f}");
        for b in 0..two[f].len() {
            if two[f][b].to_bits() != fused[f][b].to_bits() {
                identical = false;
                first = (f, b);
                break 'outer;
            }
        }
    }
    assert!(identical, "fused scores diverged from two-pass at (feature,border)={first:?}");
    println!(
        "RSBENCH006 part=parity n={n} nf={nf} nbins={nbins} n_leaves={n_leaves} \
         byte_identical=true features={nf}"
    );
}

#[test]
fn spike006_fused_parallel() {
    if std::env::var("CB_PERF").is_err() {
        eprintln!("SKIP spike006_fused_parallel: set CB_PERF=1 to run the fused-parallel prototype");
        return;
    }
    println!("RSBENCH006_META cores={}", std::thread::available_parallelism().map_or(0, |c| c.get()));
    // Parity first (cheap, and the gate for the whole approach).
    assert_parity(10_000, 20, 128, 32);
    assert_parity(40_000, 8, 254, 16);
    // Scaling: baseline shape, a low-nf shape (where two-pass starves worst), and a
    // large-n shape (where the O(n) accumulation dominates — the biggest fused win).
    bench_shape(10_000, 20, 128, 32);
    bench_shape(10_000, 8, 128, 32);
    bench_shape(40_000, 20, 128, 32);
    println!("RSBENCH006_DONE");
}
