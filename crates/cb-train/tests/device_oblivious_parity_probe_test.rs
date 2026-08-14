//! PROBE (WR-01 prerequisite): measure the ACTUAL device-vs-CPU agreement of the
//! oblivious (SymmetricTree) device grower at `bootstrap_type = No`, on a workload
//! large enough for round-off to accumulate.
//!
//! Why this exists: the device-bootstrap-parity phase targets the project's ≤1e-5
//! parity bar, but every shipped device fit oracle asserts only ε=1e-4 (see
//! `device_nonsym_fit_test.rs`). If the BASE oblivious device grower cannot already
//! hold ≤1e-5 against the CPU grower with sampling switched OFF, then no amount of
//! bootstrap wiring can reach ≤1e-5 either, and the base grower must be fixed first.
//! This probe answers that question with a measured number instead of an assumption.
//!
//! It asserts only the SHIPPED ε=1e-4 (so it cannot turn the suite red) and PRINTS the
//! measured max |Δpred| plus which of 1e-4 / 1e-5 / 1e-6 that margin would satisfy.
//!
//! Runs on the REAL device only (rocm / cuda); `GpuBackend` is not compiled under the
//! `cpu` feature, so cpu/wgpu SKIP — the WR-01 anti-false-pass convention.
//! Lives under `tests/` (integration) per the same dev-dep diamond note as
//! `device_nonsym_fit_test.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// A non-separable regression fixture big enough that per-tree round-off accumulates:
/// `n` objects x `n_features` quantized ramps, target a smooth nonlinear combination so
/// no single tree fits it exactly and all `iterations` trees carry real residual.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn fixture(n: usize, n_features: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut columns = Vec::with_capacity(n_features);
    let mut borders = Vec::with_capacity(n_features);
    for f in 0..n_features {
        // Deterministic pseudo-spread per feature, values in 0..32.
        let col: Vec<f32> = (0..n).map(|i| ((i * 7 + f * 13) % 32) as f32).collect();
        columns.push(col);
        // 31 borders at k+0.5 → 32 bins.
        borders.push((0..31).map(|k| k as f64 + 0.5).collect::<Vec<f64>>());
    }
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let a = columns[0][i] as f64;
            let b = columns[1 % n_features][i] as f64;
            // Nonlinear, non-separable by any single split.
            (a * 0.31).sin() + (b * 0.17).cos() * 0.5 + ((i % 11) as f64) * 0.05
        })
        .collect();
    (columns, borders, target)
}

/// Device-eligible OBLIVIOUS params: RMSE / Plain / fold-1 / unit-weight / bias-0
/// (`boost_from_average = false`, required by the CR-01 bias gate) / Gradient leaf /
/// `bootstrap_type = No` / `random_strength = 0` — i.e. every `device_host_eligible`
/// precondition satisfied, so the fit commits to the device oblivious grow.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn oblivious_params(depth: usize, iterations: usize) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth,
        learning_rate: 0.3,
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

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use super::{fixture, oblivious_params};
    use cb_backend::GpuBackend;
    use cb_compute::{rmse_der1, rmse_der2, Derivatives, Loss, Runtime};
    use cb_core::CbResult;
    use cb_model::Model as CbModel;
    use cb_train::train;

    /// CPU reference runtime that DECLINES the device seam (trait defaults: `begin →
    /// Ok(false)`, `grow → Ok(None)`) so `train` runs the byte-unchanged CPU grower,
    /// while computing the same RMSE derivatives the real `CpuBackend` would
    /// (`CpuBackend` is not compiled under rocm/cuda).
    struct CpuRefRuntime;

    impl Runtime for CpuRefRuntime {
        fn compute_gradients(
            &self,
            _loss: &Loss,
            approx: &[f64],
            target: &[f64],
            _approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            let der1: Vec<f64> =
                approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect();
            let der2: Vec<f64> =
                approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect();
            Ok(Derivatives { der1, der2 })
        }
    }

    pub fn probe(n: usize, n_features: usize, depth: usize, iterations: usize) {
        let (columns, borders, target) = fixture(n, n_features);
        let params = oblivious_params(depth, iterations);
        let label = format!("n={n} nf={n_features} depth={depth} iters={iterations}");

        let dev = train(&GpuBackend::default(), &columns, &borders, &target, &[], &params, None)
            .unwrap_or_else(|e| panic!("[{label}] device oblivious train failed: {e:?}"));
        let cpu = train(&CpuRefRuntime, &columns, &borders, &target, &[], &params, None)
            .unwrap_or_else(|e| panic!("[{label}] cpu oblivious train failed: {e:?}"));

        assert_eq!(
            dev.oblivious_trees.len(),
            params.iterations,
            "[{label}] device fit must produce one oblivious tree per iteration"
        );

        // Structural agreement: identical split sequence is a stronger signal than
        // prediction closeness, and tells us whether any Δ is pure arithmetic
        // round-off or an actual different tree.
        let mut split_mismatches = 0usize;
        for (t, (d, c)) in dev.oblivious_trees.iter().zip(cpu.oblivious_trees.iter()).enumerate() {
            if d.splits != c.splits {
                split_mismatches += 1;
                if split_mismatches <= 3 {
                    println!("[{label}] tree {t} SPLIT MISMATCH: device {:?} vs cpu {:?}",
                             d.splits, c.splits);
                }
            }
        }

        // DIAGNOSTIC: for each structurally-mismatched tree, apply THAT TREE ALONE on
        // both sides. If the per-tree contributions still agree, the divergent split is
        // functionally degenerate (a gain tie whose partition does not change any
        // object's value). If they differ, the trees are genuinely different models and
        // the summed agreement is boosting compensation instead — a far worse story.
        for (t, (d, c)) in dev.oblivious_trees.iter().zip(cpu.oblivious_trees.iter()).enumerate() {
            if d.splits == c.splits {
                continue;
            }
            let mut dev_one = dev.clone();
            dev_one.oblivious_trees = vec![d.clone()];
            let mut cpu_one = cpu.clone();
            cpu_one.oblivious_trees = vec![c.clone()];
            let dm = CbModel::from_trained(&dev_one, borders.clone());
            let cm = CbModel::from_trained(&cpu_one, borders.clone());
            let dp = cb_model::predict_raw(&dm, &columns);
            let cp = cb_model::predict_raw(&cm, &columns);
            let mut tree_max = 0.0_f64;
            for (&a, &b) in dp.iter().zip(cp.iter()) {
                tree_max = tree_max.max((a - b).abs());
            }
            println!(
                "[{label}]   tree {t} ALONE: max|Δcontribution|={tree_max:.3e} \
                 => {}",
                if tree_max <= 1e-12 { "DEGENERATE TIE (same partition value)" }
                else { "GENUINELY DIFFERENT TREE" }
            );
        }

        let dev_model = CbModel::from_trained(&dev, borders.clone());
        let cpu_model = CbModel::from_trained(&cpu, borders.clone());
        let dev_pred = cb_model::predict_raw(&dev_model, &columns);
        let cpu_pred = cb_model::predict_raw(&cpu_model, &columns);
        assert_eq!(dev_pred.len(), cpu_pred.len());

        let mut max_abs = 0.0_f64;
        for (&d, &c) in dev_pred.iter().zip(cpu_pred.iter()) {
            max_abs = max_abs.max((d - c).abs());
        }
        // Leaf-value agreement, independent of the apply path.
        let mut max_leaf = 0.0_f64;
        for (d, c) in dev.oblivious_trees.iter().zip(cpu.oblivious_trees.iter()) {
            for (&dv, &cv) in d.leaf_values.iter().zip(c.leaf_values.iter()) {
                max_leaf = max_leaf.max((dv - cv).abs());
            }
        }

        println!(
            "[PROBE {label}] max|Δpred|={max_abs:.3e} max|Δleaf|={max_leaf:.3e} \
             split_mismatched_trees={split_mismatches}/{} | \
             holds_1e-4={} holds_1e-5={} holds_1e-6={}",
            params.iterations,
            max_abs <= 1e-4,
            max_abs <= 1e-5,
            max_abs <= 1e-6,
        );

        // Assert ONLY the shipped bar so this probe never reddens the suite.
        assert!(
            max_abs <= 1e-4,
            "[{label}] device vs cpu max|Δpred|={max_abs:.3e} exceeds the SHIPPED ε=1e-4"
        );
    }
}

#[test]
fn device_oblivious_parity_probe() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        // Small/shallow → large/deep, so we can see whether any gap grows with
        // accumulation (round-off) or is present from the start (a real divergence).
        device::probe(512, 4, 3, 5);
        device::probe(2048, 8, 6, 10);
        device::probe(20_000, 16, 6, 20);
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = oblivious_params(6, 10);
        let _ = fixture(16, 2);
        eprintln!("SKIP device_oblivious_parity_probe: needs rocm/cuda");
    }
}
