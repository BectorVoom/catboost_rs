//! WR-01 speed evidence: a sampled fit must now run at DEVICE speed, not at the
//! CPU-fallback speed the pre-WR-01 gate forced it to.
//!
//! This is the in-repo, hardware-agnostic form of the phase's performance claim. The
//! archived Kaggle P100 run (`bench/bootstrap_gpu/kaggle-output-260730`) measured the
//! gap this phase exists to close, on 300k×50 / depth 6 / 30 iters:
//!
//! | bootstrap | catboost-rs | ran on | vs CatBoost GPU |
//! |---|---|---|---|
//! | `No`        |  1.93 s | GPU | 1.47× |
//! | `Bayesian`  | 16.56 s | CPU | 12.97× |
//! | `Bernoulli` | 16.26 s | CPU | 12.28× |
//! | `MVS`       | 16.72 s | CPU | 12.10× |
//!
//! The three sampled rows were ~8.4× slower than the `No` row purely because
//! `device_host_eligible` excluded them and the GPU sat idle. The assertion here is
//! therefore deliberately RATIO-based against the `No` row on the SAME machine, which
//! makes it meaningful on any device without hard-coding a wall-clock number:
//! a sampled fit that is still falling back to the CPU shows up as a multiple-× ratio,
//! while a genuinely device-resident sampled fit costs only its per-tree host sample
//! plus two extra elementwise device products.
//!
//! Timing is inherently noisy, so the bar is loose (see `MAX_RATIO`): this test exists
//! to catch a REGRESSION to CPU fallback, not to certify a specific speedup. The
//! precise cross-implementation numbers come from the CUDA bench harness.
//!
//! rocm/cuda only; cpu/wgpu print a SKIP line.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use cb_backend::GpuBackend;
    use cb_compute::{EScoreFunction, LeafMethod, Loss};
    use cb_train::{
        train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType,
    };
    use std::time::Instant;

    /// The pre-WR-01 CPU-fallback penalty was ~8.4× the `No` row. A sampled fit that
    /// still reaches the device costs only the host sample + two elementwise products,
    /// so anything at or above this ratio means the fit fell back to the CPU grower.
    /// Set well below 8.4 to catch the regression, and well above 1.0 to absorb timing
    /// noise and the genuine host-sampling cost.
    const MAX_RATIO: f64 = 3.0;

    fn fixture(n: usize, nf: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
        let mut columns = Vec::with_capacity(nf);
        let mut borders = Vec::with_capacity(nf);
        for f in 0..nf {
            columns.push((0..n).map(|i| ((i * 7 + f * 13) % 32) as f32).collect::<Vec<f32>>());
            borders.push((0..31).map(|k| k as f64 + 0.5).collect::<Vec<f64>>());
        }
        let target = (0..n)
            .map(|i| {
                let a = f64::from(columns[0][i]);
                (a * 0.31).sin() + ((i % 11) as f64) * 0.05
            })
            .collect();
        (columns, borders, target)
    }

    fn params(bt: EBootstrapType, subsample: f64, temp: f32) -> BoostParams {
        BoostParams {
            loss: Loss::Rmse,
            iterations: 15,
            depth: 6,
            learning_rate: 0.1,
            l2_leaf_reg: 3.0,
            random_strength: 0.0,
            boost_from_average: false,
            leaf_method: LeafMethod::Gradient,
            bootstrap_type: bt,
            subsample,
            bagging_temperature: temp,
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

    pub fn sampled_fits_run_at_device_speed() {
        let (n, nf) = (60_000usize, 24usize);
        let (columns, borders, target) = fixture(n, nf);

        let timed = |bt, ss, tp| -> f64 {
            let p = params(bt, ss, tp);
            // Warm run absorbs kernel JIT / allocator warm-up so the timed run measures
            // steady-state training, not first-launch compilation.
            let _ = train(&GpuBackend::default(), &columns, &borders, &target, &[], &p, None)
                .expect("warm fit must succeed");
            let t0 = Instant::now();
            let m = train(&GpuBackend::default(), &columns, &borders, &target, &[], &p, None)
                .expect("timed fit must succeed");
            let s = t0.elapsed().as_secs_f64();
            assert_eq!(m.oblivious_trees.len(), p.iterations, "all trees must be grown");
            s
        };

        let base = timed(EBootstrapType::No, 1.0, 0.0);
        println!("[speed] No        {base:.3}s (device baseline, n={n} nf={nf})");

        for (name, bt, ss, tp) in [
            ("Bernoulli", EBootstrapType::Bernoulli, 0.8_f64, 0.0_f32),
            ("Bayesian", EBootstrapType::Bayesian, 1.0, 1.0),
            ("MVS", EBootstrapType::Mvs, 0.8, 0.0),
        ] {
            let s = timed(bt, ss, tp);
            let ratio = s / base;
            println!("[speed] {name:9} {s:.3}s  ratio_vs_No={ratio:.2}x");
            assert!(
                ratio <= MAX_RATIO,
                "[speed] {name} took {ratio:.2}x the unsampled device fit (> {MAX_RATIO}x); \
                 that is the signature of a silent fallback to the CPU grower, which is \
                 exactly the ~8.4x penalty WR-01 removed"
            );
        }
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn wr01_sampled_fits_run_at_device_speed() {
    device::sampled_fits_run_at_device_speed();
}

#[cfg(not(any(feature = "rocm", feature = "cuda")))]
#[test]
fn wr01_sampled_fits_run_at_device_speed() {
    println!("SKIP wr01_sampled_fits_run_at_device_speed: needs rocm/cuda");
}
