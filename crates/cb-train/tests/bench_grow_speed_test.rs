//! BENCH-02 grow-loop speed harness (GPUT-18): device-accelerated non-symmetric /
//! Region grow vs the host-CPU boosting loop, train-only, warm-run (JIT excluded),
//! on a large-n synthetic workload. Times `cb_train::train` with `GpuBackend`
//! (device path) against a CPU-declining `Runtime` (host `leaf_wise_grower` /
//! Region grower) — both compiled in ONE `--features cuda` binary, so the only
//! variable is the backend. Prints machine-greppable `BENCH ...` lines.
//!
//! INERT by default: does real work ONLY when the `CB_BENCH` env var is set, so a
//! normal `cargo test` run just prints SKIP (this is a benchmark, not a gate).
//! Runs on the real device backends only (rocm/cuda); cpu/wgpu skip.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod bench {
    use cb_backend::GpuBackend;
    use cb_compute::{rmse_der1, rmse_der2, Derivatives, EScoreFunction, LeafMethod, Loss, Runtime};
    use cb_core::CbResult;
    use cb_train::{
        train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType,
    };
    use std::time::Instant;

    /// CPU reference runtime: declines the device grow (trait defaults → `Ok(None)`),
    /// so `train` uses the byte-unchanged host boosting loop, and supplies the RMSE
    /// gradients the real CPU backend would.
    struct CpuRefRuntime;
    impl Runtime for CpuRefRuntime {
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

    /// Deterministic large-n binned workload: `nf` features each in `[0, nbins)`,
    /// target a clean sign on a 2-feature combination so the grow has real gain.
    fn gen(n: usize, nf: usize, nbins: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
        let mut cols = Vec::with_capacity(nf);
        let mut borders = Vec::with_capacity(nf);
        for f in 0..nf {
            let col: Vec<f32> = (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(f.wrapping_mul(40_503));
                    (h % nbins) as f32
                })
                .collect();
            cols.push(col);
            borders.push((0..nbins - 1).map(|k| k as f64 + 0.5).collect());
        }
        let thresh = (nbins as f64) * 0.75;
        let target: Vec<f64> = (0..n)
            .map(|i| {
                let a = cols[0][i] as f64;
                let b = cols[1 % nf][i] as f64;
                if a + 0.5 * b > thresh { 1.0 } else { -1.0 }
            })
            .collect();
        (cols, borders, target)
    }

    fn params(grow_policy: EGrowPolicy, depth: usize, iterations: usize) -> BoostParams {
        BoostParams {
            loss: Loss::Rmse,
            iterations,
            depth,
            learning_rate: 0.3,
            l2_leaf_reg: 0.0,
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
            grow_policy,
            max_leaves: cb_train::max_leaves_default(),
            min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        }
    }

    fn time_train<R: Runtime>(
        backend: &R,
        cols: &[Vec<f32>],
        borders: &[Vec<f64>],
        target: &[f64],
        p: &BoostParams,
    ) -> (f64, usize) {
        let t = Instant::now();
        let m = train(backend, cols, borders, target, &[], p, None)
            .unwrap_or_else(|e| panic!("train failed: {e:?}"));
        let trees = m.non_symmetric_trees.len() + m.region_trees.len();
        (t.elapsed().as_secs_f64(), trees)
    }

    pub fn run() {
        let depth = 6usize;
        let iters = 20usize;
        let nf = 20usize;
        let nbins = 32usize;
        let ns: Vec<usize> = std::env::var("BENCH_NS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![10_000, 100_000, 300_000]);
        let families = [
            ("depthwise", EGrowPolicy::Depthwise),
            ("region", EGrowPolicy::Region),
        ];
        println!("BENCH_META depth={depth} iters={iters} nf={nf} nbins={nbins} ns={ns:?}");
        for (label, gp) in families {
            for &n in &ns {
                let (cols, borders, target) = gen(n, nf, nbins);
                let p = params(gp, depth, iters);
                // warm-run device: JIT the kernels (excluded from the timed run).
                let mut pw = p.clone();
                pw.iterations = 1;
                let _ = train(&GpuBackend::default(), &cols, &borders, &target, &[], &pw, None)
                    .unwrap_or_else(|e| panic!("[{label} n={n}] device warm-run failed: {e:?}"));
                let (dev_s, dtrees) =
                    time_train(&GpuBackend::default(), &cols, &borders, &target, &p);
                let (cpu_s, ctrees) = time_train(&CpuRefRuntime, &cols, &borders, &target, &p);
                let speedup = if dev_s > 0.0 { cpu_s / dev_s } else { f64::NAN };
                println!(
                    "BENCH family={label} n={n} device_s={dev_s:.4} cpu_s={cpu_s:.4} \
                     speedup={speedup:.3}x dev_trees={dtrees} cpu_trees={ctrees}"
                );
            }
        }
        println!("BENCH_DONE");
    }
}

#[test]
fn bench_grow_speed() {
    if std::env::var("CB_BENCH").is_err() {
        eprintln!("SKIP bench_grow_speed: set CB_BENCH=1 to run the BENCH-02 grow-loop timing");
        return;
    }
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    bench::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP bench_grow_speed: needs the cuda/rocm device backend");
}
