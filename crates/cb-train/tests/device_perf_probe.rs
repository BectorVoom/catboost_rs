//! GDC perf probe (T-perf, run explicitly with `-- --ignored --nocapture` on a
//! real GPU): wall-clock cost of the TWO new device channels this phase added.
//!
//! 1. WEIGHTED-DER overhead — the same large RMSE depth-6 fit with uniform vs
//!    non-uniform weights. The weighted path adds ONE elementwise multiply per
//!    tree (`fold_weights_resident`), so the ratio should be ≈1.0; a large ratio
//!    means the fast path regressed.
//! 2. CTR device fit vs the CPU reference — the same mixed float+cat Logloss
//!    fit through `train_cat` on the device session vs a CPU-declining runtime.
//!
//! Timing protocol: one untimed warm fit per configuration (absorbs JIT / first
//! launch), then the timed fit. `#[ignore]` keeps it out of every normal suite
//! run (it is a REPORTING probe, not a correctness gate — the correctness bars
//! live in the device oracle tests).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

fn probe_params(loss: Loss, depth: usize, iterations: usize) -> BoostParams {
    BoostParams {
        loss,
        iterations,
        depth,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::ECtrType::Borders,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
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

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::time::Instant;

    use super::probe_params;
    use cb_backend::GpuBackend;
    use cb_compute::{rmse_der1, rmse_der2, Derivatives, Loss, Runtime};
    use cb_core::CbResult;
    use cb_train::{train, train_cat};

    struct CpuRefRuntime;

    impl Runtime for CpuRefRuntime {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            _approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            match loss {
                Loss::Logloss => {
                    let der1: Vec<f64> = approx
                        .iter()
                        .zip(target)
                        .map(|(&a, &t)| t - 1.0 / (1.0 + (-a).exp()))
                        .collect();
                    let der2: Vec<f64> = approx
                        .iter()
                        .map(|&a| {
                            let p = 1.0 / (1.0 + (-a).exp());
                            -p * (1.0 - p)
                        })
                        .collect();
                    Ok(Derivatives { der1, der2 })
                }
                _ => {
                    let der1: Vec<f64> =
                        approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect();
                    let der2: Vec<f64> =
                        approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect();
                    Ok(Derivatives { der1, der2 })
                }
            }
        }
    }

    /// Deterministic pseudo-random f32 in [0, 1) (no external RNG dep).
    fn lcg_stream(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (u32::MAX >> 1) as f32
            })
            .collect()
    }

    pub fn run() {
        // ── (1) weighted-der overhead on a large oblivious fit ────────────────
        let n = 200_000usize;
        let n_features = 10usize;
        let columns: Vec<Vec<f32>> =
            (0..n_features).map(|f| lcg_stream(n, 0x9e37 + f as u64)).collect();
        let borders: Vec<Vec<f64>> = (0..n_features)
            .map(|_| (1..32).map(|k| k as f64 / 32.0).collect())
            .collect();
        let target: Vec<f64> = columns[0]
            .iter()
            .zip(columns[1].iter())
            .map(|(&a, &b)| f64::from(a) * 3.0 - f64::from(b) * 2.0)
            .collect();
        let weights: Vec<f64> = (0..n).map(|i| [1.0, 2.0, 1.0, 3.0][i % 4]).collect();
        let params = probe_params(Loss::Rmse, 6, 40);

        let timed = |w: &[f64]| -> f64 {
            let _ = train(&GpuBackend::default(), &columns, &borders, &target, w, &params, None)
                .expect("warm fit");
            let t0 = Instant::now();
            let m = train(&GpuBackend::default(), &columns, &borders, &target, w, &params, None)
                .expect("timed fit");
            assert_eq!(m.oblivious_trees.len(), params.iterations, "device arm fired");
            t0.elapsed().as_secs_f64()
        };
        let uniform_s = timed(&[]);
        let weighted_s = timed(&weights);
        println!(
            "[perf-probe weighted] n={n} f={n_features} depth=6 iters=40: \
             uniform={uniform_s:.3}s weighted={weighted_s:.3}s ratio={:.3}",
            weighted_s / uniform_s
        );

        // ── (2) CTR device fit vs the CPU reference ───────────────────────────
        let n = 50_000usize;
        let columns: Vec<Vec<f32>> = (0..2).map(|f| lcg_stream(n, 0xc0ffee + f as u64)).collect();
        let borders: Vec<Vec<f64>> =
            (0..2).map(|_| (1..16).map(|k| k as f64 / 16.0).collect()).collect();
        let cat_raw = lcg_stream(n, 0xfeed);
        let cat_columns: Vec<Vec<String>> = vec![cat_raw
            .iter()
            .map(|&v| format!("{}", (v * 8.0) as u32 % 8))
            .collect()];
        let target: Vec<f64> = columns[0]
            .iter()
            .zip(cat_raw.iter())
            .map(|(&a, &c)| f64::from(a + c > 1.0))
            .collect();
        let params = probe_params(Loss::Logloss, 2, 20);

        let timed_cat = |dev: bool| -> f64 {
            let go = |t0: Option<Instant>| -> f64 {
                let started = t0.unwrap_or_else(Instant::now);
                if dev {
                    let (m, _) = train_cat(
                        &GpuBackend::default(), &columns, &borders, &cat_columns, &target, &[],
                        &params, None,
                    )
                    .expect("ctr fit");
                    assert_eq!(m.oblivious_trees.len(), params.iterations);
                } else {
                    let (m, _) = train_cat(
                        &CpuRefRuntime, &columns, &borders, &cat_columns, &target, &[], &params,
                        None,
                    )
                    .expect("ctr fit");
                    assert_eq!(m.oblivious_trees.len(), params.iterations);
                }
                started.elapsed().as_secs_f64()
            };
            let _ = go(None); // warm
            go(Some(Instant::now()))
        };
        let ctr_device_s = timed_cat(true);
        let ctr_cpu_s = timed_cat(false);
        println!(
            "[perf-probe ctr] n={n} f=2+1cat depth=2 iters=20: \
             device={ctr_device_s:.3}s cpu={ctr_cpu_s:.3}s speedup={:.2}x",
            ctr_cpu_s / ctr_device_s
        );
    }
}

#[test]
#[ignore = "perf probe — run explicitly on a real GPU with -- --ignored --nocapture"]
fn device_weighted_and_ctr_perf_probe() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = probe_params(Loss::Rmse, 6, 40);
        eprintln!("SKIP device_weighted_and_ctr_perf_probe: needs rocm/cuda");
    }
}
