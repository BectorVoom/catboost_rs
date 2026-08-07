//! Opt-in END-TO-END fit bench through the public `CatBoostBuilder::fit` facade —
//! the full pipeline the Python bench times (fit-prep borders → quantize/pack →
//! device begin → per-tree grow), unlike `gpu_oblivious_bench_test` which enters at
//! `train()` with borders pre-made. This is the local attribution probe for the
//! host-prep stages (`CB_GPU_PROF fit-prep / quantize / begin / tree`), so a
//! host-prep change can be measured without a Colab/Kaggle round-trip.
//!
//! Run (mirror bench/full_param_gpu_speed/bench.py's device-eligible RMSE cell):
//!   CB_FIT_BENCH=1 CB_GPU_PROF=1 cargo test -p catboost-rs --release \
//!     --no-default-features --features rocm --test fit_full_bench_test -- --nocapture
//! Emulate the 2-vCPU Colab host with: RAYON_NUM_THREADS=2 taskset -c 0,1 <cmd>.
//! Knobs: BN (rows, default 1_000_000), BNF (features, 50), BITERS (30), BDEPTH (6),
//! BREPS (best-of, 2).
//!
//! Timing discipline: an untimed small warm fit absorbs CubeCL JIT; the timed fit
//! returns only after `fit` materializes the model, so the lazy device queue is
//! drained inside the timed region; best-of-BREPS is reported.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use catboost_rs::{CatBoostBuilder, EBootstrapType, IngestSource, Loss, OwnedColumns, Pool};
use std::time::Instant;

fn envn(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}

/// Deterministic LCG columns (values spread across the bin range) + a smooth target.
fn gen(n: usize, nf: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let cols: Vec<Vec<f64>> = (0..nf)
        .map(|_| (0..n).map(|_| next() * 10.0 - 5.0).collect())
        .collect();
    let target: Vec<f64> = (0..n)
        .map(|i| (cols[0][i] * 0.31).sin() + (cols[1 % nf][i] * 0.17).cos() * 0.5)
        .collect();
    (cols, target)
}

fn builder(iters: usize, depth: usize) -> CatBoostBuilder {
    // The device-eligible RMSE cell of bench/full_param_gpu_speed/bench.py, verbatim:
    // 30 iters, depth 6, lr 0.1, l2 3.0, 32 borders, random_strength 0,
    // bootstrap No, boost_from_average false.
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(iters)
        .depth(depth)
        .learning_rate(0.1)
        .l2_leaf_reg(3.0)
        .border_count(32)
        .random_strength(0.0)
        .bootstrap_type(EBootstrapType::No)
        .boost_from_average(false)
        .random_seed(0)
}

#[test]
fn tmp_full_fit_bench() {
    if std::env::var("CB_FIT_BENCH").ok().as_deref() != Some("1") {
        eprintln!("[fit_full_bench] SKIP: set CB_FIT_BENCH=1 to run");
        return;
    }
    let n = envn("BN", 1_000_000);
    let nf = envn("BNF", 50);
    let iters = envn("BITERS", 30);
    let depth = envn("BDEPTH", 6);
    let reps = envn("BREPS", 2);

    // Untimed warm fit (small n, same shape family) to absorb CubeCL JIT.
    let (wcols, wtarget) = gen(20_000, nf);
    let wpool: Pool = OwnedColumns::new(wcols, wtarget).into_pool().unwrap();
    builder(2, depth).fit(&wpool).unwrap();

    let (cols, target) = gen(n, nf);
    let pool: Pool = OwnedColumns::new(cols, target).into_pool().unwrap();
    let b = builder(iters, depth);
    let mut best = f64::INFINITY;
    for rep in 0..reps.max(1) {
        let t = Instant::now();
        let model = b.fit(&pool).unwrap();
        let dt = t.elapsed().as_secs_f64();
        eprintln!("[fit_full_bench] rep {rep}: n={n} nf={nf} iters={iters} depth={depth} fit={dt:.3}s");
        // Sanity: the fitted model scores (cheap — the small warm pool, not the 1M one).
        let preds = model.predict(&wpool).unwrap();
        assert_eq!(preds.len(), wpool.n_rows(), "fit produced a non-scoring model");
        best = best.min(dt);
    }
    eprintln!("[fit_full_bench] BEST fit={best:.3}s (n={n} nf={nf} iters={iters} depth={depth})");
}
