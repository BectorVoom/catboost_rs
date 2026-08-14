//! OBLIVIOUS resident device-grow bench (opt-in, like `bench_grow_speed_test`).
//!
//! `bench_grow_speed_test.rs` covers only the non-symmetric Depthwise / Region device
//! families; the default SymmetricTree (oblivious) resident grow — the path a plain
//! `bootstrap_type=No` fit actually takes, and the one compared against official
//! CatBoost GPU — had no bench at all. This fills that gap so a device-grow change can
//! be attributed locally instead of via a Kaggle round-trip.
//!
//! Run: CB_OBL_BENCH=1 cargo test -p cb-train --release --no-default-features \
//!        --features rocm --test gpu_oblivious_bench_test -- --nocapture
//! Add CB_GPU_PROF=1 for per-stage attribution (fill / derive / score / split /
//! stats_read / leaf_apply_der), which is how the `partition_update` contention
//! hotspot was localized.
//! Knobs: BN (rows), BNF (features), BDEPTH, BITERS, BREPS.
//!
//! Timing discipline: an untimed 1-iteration warm run absorbs CubeCL JIT, `train()`
//! blocks on read-back so the lazy queue is drained inside the timed region, and the
//! reported number is best-of-BREPS.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

fn envn(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod bench {
    use super::envn;
    use cb_backend::GpuBackend;
    use cb_compute::{EScoreFunction, LeafMethod, Loss};
    use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};
    use std::time::Instant;

    fn gen(n: usize, nf: usize, nbins: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
        let mut cols = Vec::with_capacity(nf);
        let mut borders = Vec::with_capacity(nf);
        // Deterministic LCG so the data is spread across bins (not a modulo comb, which
        // makes histograms unrealistically uniform).
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / ((1u64 << 31) as f64)
        };
        for _ in 0..nf {
            let col: Vec<f32> = (0..n).map(|_| (next() * (nbins as f64)) as f32).collect();
            cols.push(col);
            borders.push((0..nbins - 1).map(|k| k as f64 + 0.5).collect::<Vec<f64>>());
        }
        let target: Vec<f64> = (0..n)
            .map(|i| {
                let a = cols[0][i] as f64;
                let b = cols[1 % nf][i] as f64;
                (a * 0.31).sin() + (b * 0.17).cos() * 0.5
            })
            .collect();
        (cols, borders, target)
    }

    fn params(depth: usize, iters: usize) -> BoostParams {
        BoostParams {
            loss: Loss::Rmse,
            iterations: iters,
            depth,
            learning_rate: 0.1,
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
            extra: Default::default(),
        }
    }

    pub fn run() {
        let n = envn("BN", 300_000);
        let nf = envn("BNF", 50);
        let depth = envn("BDEPTH", 6);
        let iters = envn("BITERS", 30);
        let reps = envn("BREPS", 3);
        let nbins = 32usize;
        let (cols, borders, target) = gen(n, nf, nbins);
        let p = params(depth, iters);

        // Warm run (1 iter) to absorb CubeCL JIT — excluded from timing.
        let mut pw = p.clone();
        pw.iterations = 1;
        let t_jit = Instant::now();
        let _ = train(&GpuBackend::default(), &cols, &borders, &target, &[], &pw, None)
            .unwrap_or_else(|e| panic!("device warm-run failed: {e:?}"));
        let jit_s = t_jit.elapsed().as_secs_f64();

        let mut best = f64::INFINITY;
        let mut all = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            let m = train(&GpuBackend::default(), &cols, &borders, &target, &[], &p, None)
                .unwrap_or_else(|e| panic!("device train failed: {e:?}"));
            let s = t0.elapsed().as_secs_f64();
            assert_eq!(m.oblivious_trees.len(), iters, "must grow one tree per iteration");
            all.push(s);
            best = best.min(s);
        }
        println!(
            "OBL_BENCH n={n} nf={nf} depth={depth} iters={iters} nbins={nbins} \
             warm_jit_s={jit_s:.4} best_s={best:.4} all={all:?}"
        );
    }
}

#[test]
fn tmp_gpu_obl_bench() {
    if std::env::var("CB_OBL_BENCH").is_err() {
        eprintln!("SKIP tmp_gpu_obl_bench: set CB_OBL_BENCH=1");
        return;
    }
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    bench::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = envn("BN", 1);
        eprintln!("SKIP tmp_gpu_obl_bench: needs rocm/cuda");
    }
}
