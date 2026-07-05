//! SPIKE 002 (perf-baseline-and-scaling): times the pure-CPU host boosting loop
//! (`cb_train::train` with a device-declining `Runtime`, so the byte-unchanged
//! host oblivious grow runs) across a synthetic scaling grid. The companion
//! `catboost_grid.py` trains official CatBoost CPU on the same shapes/params so
//! the two can be compared row-for-row.
//!
//! INERT by default: does real work ONLY when `CB_PERF` is set, so a normal
//! `cargo test` run just prints SKIP. Prints machine-greppable `RSBENCH ...`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{rmse_der1, rmse_der2, Derivatives, EScoreFunction, LeafMethod, Loss, Runtime};
use cb_core::CbResult;
use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};
use std::time::Instant;

/// Pure host-CPU runtime: computes RMSE derivatives on the host and declines the
/// device grow (trait defaults → `Ok(None)`), so `train` runs the byte-unchanged
/// host oblivious boosting loop. This isolates the tree-growing / split-finding
/// cost — the design-doc pipeline — from any cubecl-cpu derivative overhead.
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

/// Deterministic continuous workload matching `catboost_grid.py`'s generator:
/// `nf` features in `[0, 1)` from a splittable hash, a linear-combination
/// regression target. `nbins` bins ⇒ `nbins - 1` evenly spaced borders per
/// feature (so the histogram width matches CatBoost's `border_count = nbins-1`).
fn gen(n: usize, nf: usize, nbins: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut cols = Vec::with_capacity(nf);
    for f in 0..nf {
        let col: Vec<f32> = (0..n)
            .map(|i| {
                // splitmix64-ish hash → uniform [0,1)
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

fn time_row(n: usize, nf: usize, nbins: usize, depth: usize, iters: usize) {
    let (cols, borders, target) = gen(n, nf, nbins);
    let p = params(depth, iters);
    let t = Instant::now();
    let m = train(&CpuHostRuntime, &cols, &borders, &target, &[], &p, None)
        .unwrap_or_else(|e| panic!("train failed: {e:?}"));
    let secs = t.elapsed().as_secs_f64();
    let trees = m.oblivious_trees.len();
    println!(
        "RSBENCH n={n} nf={nf} nbins={nbins} depth={depth} iters={iters} \
         train_s={secs:.4} trees={trees} per_tree_ms={:.3}",
        secs * 1000.0 / iters as f64
    );
}

#[test]
fn perf_baseline_grid() {
    if std::env::var("CB_PERF").is_err() {
        eprintln!("SKIP perf_baseline_grid: set CB_PERF=1 to run the SPIKE-002 CPU timing grid");
        return;
    }
    // A small fixed iteration count keeps wall-clock bounded; per-tree time is the
    // reported figure. One axis varies at a time around a baseline config.
    // Lighter grid tuned to COMPLETE (the naive O(candidates*n*depth) rescan makes
    // large configs run for minutes). per_tree_ms is the reported, iters-normalized
    // metric, so a small iters is fine. Baseline n kept modest; the diagnostic
    // axes (n_bins, n_features) sweep widest — n_bins is the cleanest "missing
    // histogram" signal (Rust cost ∝ n_bins; CatBoost histogram scoring is ~flat).
    let base_n = 10_000usize;
    let base_nf = 20usize;
    let base_nbins = 128usize;
    let base_depth = 6usize;
    let iters = 3usize;

    println!("RSBENCH_META base_n={base_n} base_nf={base_nf} base_nbins={base_nbins} base_depth={base_depth} iters={iters}");

    // Baseline
    time_row(base_n, base_nf, base_nbins, base_depth, iters);
    // Sweep n_rows
    for &n in &[5_000usize, 20_000, 40_000] {
        time_row(n, base_nf, base_nbins, base_depth, iters);
    }
    // Sweep n_features
    for &nf in &[5usize, 10, 40] {
        time_row(base_n, nf, base_nbins, base_depth, iters);
    }
    // Sweep n_bins (border_count) — the key diagnostic axis
    for &nb in &[16usize, 32, 64, 254] {
        time_row(base_n, base_nf, nb, base_depth, iters);
    }
    // Sweep depth
    for &d in &[2usize, 4] {
        time_row(base_n, base_nf, base_nbins, d, iters);
    }
    println!("RSBENCH_DONE");
}
