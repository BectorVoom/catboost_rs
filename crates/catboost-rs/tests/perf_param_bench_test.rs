//! Opt-in CPU benchmark for the three perf-critical parameters of this wave:
//! `thread_count`, `rsm` and `langevin`.
//!
//! Run:
//!   CB_PERF_BENCH=1 cargo test -p catboost-rs --release \
//!     --test perf_param_bench_test -- --nocapture --test-threads=1
//!
//! Knobs: `BN` (rows, default 200_000), `BNF` (features, 20), `BITERS` (20),
//! `BDEPTH` (6), `BREPS` (best-of, 3).
//!
//! `--test-threads=1` is REQUIRED, not cosmetic: these cells compete for the same
//! cores, so running them concurrently would measure contention rather than the
//! parameter.
//!
//! # What each cell is for
//!
//! * `thread_count` — the scaling curve. Reported as speedup against
//!   `thread_count=1` alongside the ideal, because the headline "Nx faster" is
//!   meaningless without knowing how far from linear it is. This is the parameter
//!   with the largest measured headroom: an earlier wave found this engine's
//!   `GreedyLogSum` baseline at 0.62 s against official CatBoost's 0.17 s, but
//!   official at `thread_count=1` was 0.58 s — i.e. essentially the whole 3.6x
//!   gap was threading, not the algorithm.
//! * `rsm` — feature subsampling is the rare parameter that should make training
//!   FASTER (fewer candidates scored per level) while changing the model. The
//!   cell reports both, so a speedup that comes with a wrecked fit is visible.
//! * `langevin` — pure overhead: two extra noise passes per tree. Expected to be
//!   small; measured so "negligible" is a number rather than an assumption.
//!
//! Timing discipline: an untimed warm fit first, best-of-`BREPS` reported, and the
//! same corpus reused across cells so only the parameter varies.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::time::Instant;

use catboost_rs::{CatBoostBuilder, IngestSource, Loss, OwnedColumns, Pool};

fn envn(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}

fn enabled() -> bool {
    std::env::var("CB_PERF_BENCH").is_ok_and(|v| v != "0")
}

/// Deterministic LCG columns + a smooth target (the `fit_full_bench_test` corpus,
/// so numbers are comparable across the repository's benches).
fn gen(n: usize, nf: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f64) / ((1_u64 << 31) as f64)
    };
    let cols: Vec<Vec<f64>> = (0..nf)
        .map(|_| (0..n).map(|_| next() * 10.0 - 5.0).collect())
        .collect();
    let target: Vec<f64> = (0..n)
        .map(|i| (cols[0][i] * 0.31).sin() + (cols[1 % nf][i] * 0.17).cos() * 0.5)
        .collect();
    (cols, target)
}

fn corpus() -> (Pool, usize, usize) {
    let n = envn("BN", 200_000);
    let nf = envn("BNF", 20);
    let (cols, target) = gen(n, nf);
    (
        OwnedColumns::new(cols, target).into_pool().expect("pool"),
        n,
        nf,
    )
}

fn builder() -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(envn("BITERS", 20))
        .depth(envn("BDEPTH", 6))
        .learning_rate(0.03)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(false)
        .random_seed(42)
        .border_count(254)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
}

/// One timed fit.
fn one_fit(pool: &Pool, make: &dyn Fn() -> CatBoostBuilder) -> f64 {
    let t = Instant::now();
    let m = make().fit(pool).expect("timed fit");
    let secs = t.elapsed().as_secs_f64();
    // Touch the model so nothing can be optimized away.
    assert!(!m.as_canonical().oblivious_trees.is_empty());
    secs
}

/// Best-of-`BREPS` for EVERY configuration, with the repetitions INTERLEAVED
/// (round-robin) rather than run as a block per configuration.
///
/// This matters: measuring one config to completion before starting the next lets
/// a transient — another process, a thermal excursion — land entirely inside one
/// cell and be reported as that parameter's cost. An early version of this bench
/// did exactly that and showed `rsm=0.75` at 0.65x, then `rsm=0.5` at 0.57x on a
/// re-run: the outlier followed the schedule, not the parameter. Interleaving
/// spreads any transient across cells, and best-of keeps the cleanest run.
fn time_all(pool: &Pool, cells: &[(String, Box<dyn Fn() -> CatBoostBuilder>)]) -> Vec<f64> {
    for (_, make) in cells {
        let _warm = make().fit(pool).expect("warm fit");
    }
    let reps = envn("BREPS", 3);
    let mut best = vec![f64::INFINITY; cells.len()];
    for _ in 0..reps {
        for (i, (_, make)) in cells.iter().enumerate() {
            best[i] = best[i].min(one_fit(pool, make.as_ref()));
        }
    }
    best
}

#[test]
fn thread_count_scaling() {
    if !enabled() {
        eprintln!("SKIP thread_count_scaling: set CB_PERF_BENCH=1");
        return;
    }
    let (pool, n, nf) = corpus();
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    println!(
        "\n=== thread_count scaling  (n={n}, features={nf}, iters={}, depth={}, cores={cores}) ===",
        envn("BITERS", 20),
        envn("BDEPTH", 6)
    );
    let counts: Vec<usize> = vec![1, 2, 4, 8, 16, 0];
    let cells: Vec<(String, Box<dyn Fn() -> CatBoostBuilder>)> = counts
        .iter()
        .map(|&tc| {
            let label = if tc == 0 { "0 (all)".to_owned() } else { tc.to_string() };
            let f: Box<dyn Fn() -> CatBoostBuilder> =
                Box::new(move || builder().thread_count(tc));
            (label, f)
        })
        .collect();
    let secs = time_all(&pool, &cells);
    let base = secs[0];
    println!("{:>10}  {:>10}  {:>10}  {:>10}  {:>10}", "threads", "secs", "speedup", "ideal", "efficiency");
    for (i, &tc) in counts.iter().enumerate() {
        let ideal = if tc == 0 { cores } else { tc.min(cores) } as f64;
        let sp = base / secs[i];
        println!(
            "{:>10}  {:>10.3}  {:>9.2}x  {:>9.2}x  {:>9.0}%",
            cells[i].0,
            secs[i],
            sp,
            ideal,
            sp / ideal * 100.0
        );
    }
}

#[test]
fn rsm_cost() {
    if !enabled() {
        eprintln!("SKIP rsm_cost: set CB_PERF_BENCH=1");
        return;
    }
    let (pool, n, nf) = corpus();
    println!("\n=== rsm cost  (n={n}, features={nf}) ===");
    println!(
        "NOTE: rsm < 1 DECLINES the device grower and forces the CPU path, so these\n\
         numbers are the CPU cost. rsm=1.0 and 'unset' are timed separately because\n\
         only values BELOW 1 enable the per-level candidate draws."
    );
    let values: Vec<Option<f64>> = vec![None, Some(1.0), Some(0.75), Some(0.5), Some(0.25)];
    let cells: Vec<(String, Box<dyn Fn() -> CatBoostBuilder>)> = values
        .iter()
        .map(|&v| {
            let label = v.map_or_else(|| "unset".to_owned(), |r| format!("{r}"));
            let f: Box<dyn Fn() -> CatBoostBuilder> = match v {
                None => Box::new(builder),
                Some(r) => Box::new(move || builder().rsm(r)),
            };
            (label, f)
        })
        .collect();
    let secs = time_all(&pool, &cells);

    // Training RMSE, measured OUTSIDE the timed region so the extra fits cannot
    // perturb the timings.
    let (_, target) = gen(n, nf);
    let rmse = |b: CatBoostBuilder| -> f64 {
        let m = b.fit(&pool).expect("fit");
        let p = m.predict(&pool).expect("predict");
        (p.iter().zip(target.iter()).map(|(a, t)| (a - t) * (a - t)).sum::<f64>()
            / p.len() as f64)
            .sqrt()
    };

    println!("{:>8}  {:>10}  {:>10}  {:>12}", "rsm", "secs", "vs unset", "train RMSE");
    for (i, &v) in values.iter().enumerate() {
        let b = match v {
            None => builder(),
            Some(r) => builder().rsm(r),
        };
        println!(
            "{:>8}  {:>10.3}  {:>9.2}x  {:>12.5}",
            cells[i].0,
            secs[i],
            secs[0] / secs[i],
            rmse(b)
        );
    }
}

#[test]
fn langevin_overhead() {
    if !enabled() {
        eprintln!("SKIP langevin_overhead: set CB_PERF_BENCH=1");
        return;
    }
    let (pool, n, nf) = corpus();
    println!("\n=== langevin overhead  (n={n}, features={nf}) ===");
    println!(
        "Two extra passes per tree: one Gaussian per object (derivatives) and one\n\
         per leaf (leaf sums). Both are O(n) / O(leaves) against an O(n * features *\n\
         bins) histogram build, so the expectation is 'small'; this makes it a number."
    );
    let cells: Vec<(String, Box<dyn Fn() -> CatBoostBuilder>)> = vec![
        ("off".to_owned(), Box::new(builder)),
        (
            "dt=10000 (default)".to_owned(),
            Box::new(|| builder().langevin(true).diffusion_temperature(10_000.0)),
        ),
        (
            "dt=100".to_owned(),
            Box::new(|| builder().langevin(true).diffusion_temperature(100.0)),
        ),
        (
            "posterior_sampling".to_owned(),
            Box::new(|| builder().posterior_sampling(true)),
        ),
    ];
    let secs = time_all(&pool, &cells);
    println!("{:>22}  {:>10}  {:>10}", "config", "secs", "overhead");
    for (i, (label, _)) in cells.iter().enumerate() {
        if i == 0 {
            println!("{:>22}  {:>10.3}  {:>10}", label, secs[i], "-");
        } else {
            println!(
                "{:>22}  {:>10.3}  {:>9.1}%",
                label,
                secs[i],
                (secs[i] / secs[0] - 1.0) * 100.0
            );
        }
    }
}
