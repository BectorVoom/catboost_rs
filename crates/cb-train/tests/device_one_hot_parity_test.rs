//! T28 / SPEC-OH-25 — the device one-hot grower matches the CPU grower within 1e-5.
//!
//! This is a PURE GATE: every piece of wiring belongs to T24/T25/T26/T27/T27b. If a
//! failure here needs a production change, the owning task is what must be re-opened.
//!
//! # Anti-false-pass discipline (both mandatory, SPEC-OH-25)
//!
//! 1. **A silent CPU fallback makes "device == CPU" trivially true.** Closed by
//!    `CountingGpu`, which wraps the real `GpuBackend` and counts the trees that actually
//!    came back from `grow_tree_on_device`; every scenario asserts that count equals
//!    `iterations`.
//! 2. **The device could train a one-hot pool while choosing only FLOAT splits.** Then the
//!    equality fold is never exercised. Closed by asserting the trained model carries at
//!    least one `ModelSplit::OneHot`.
//!
//! Runs on the REAL device only (rocm / cuda); `GpuBackend` is not compiled under the
//! `cpu` feature, so cpu/wgpu print a SKIP line rather than passing silently.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// The ≤1e-5 project parity bar.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
const EPS: f64 = 1e-5;

/// Device-eligible params for a one-hot pool. Every `device_host_eligible` precondition is
/// satisfied: RMSE / Plain / fold-1 / unit weights / bias 0 (`boost_from_average = false`)
/// / Gradient leaf / `random_strength = 0` / SymmetricTree / no eval sets / no CTR.
///
/// `one_hot_max_size` is set so EVERY cat column routes one-hot — a mixed one-hot × CTR
/// pool is typed-rejected by SPEC-OH-26 and could not reach the device at all.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn one_hot_params(depth: usize, iterations: usize, one_hot_max_size: u32) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth,
        learning_rate: 0.3,
        auto_learning_rate: false,
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
        one_hot_max_size,
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::Cosine,
        has_time: cb_train::has_time_default(),
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

#[cfg(not(any(feature = "rocm", feature = "cuda")))]
#[test]
fn device_one_hot_parity_skips_without_a_device() {
    println!(
        "[T28] SKIP device one-hot parity: `GpuBackend` is not compiled under the cpu/wgpu \
         features. Run with: cargo test -p cb-train --no-default-features --features rocm \
         --test device_one_hot_parity_test"
    );
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use super::{one_hot_params, EPS};
    use cb_backend::GpuBackend;
    use cb_compute::{
        rmse_der1, rmse_der2, Derivatives, DeviceGrownTree, DeviceTrainConfig, EScoreFunction,
        Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_model::{predict_raw_cat, Model as CbModel, ModelSplit};
    use cb_train::{train_cat, BoostParams};
    use std::cell::Cell;

    /// CPU reference runtime that DECLINES the device seam (trait defaults: `begin →
    /// Ok(false)`, `grow → Ok(None)`) so `train_cat` runs the byte-unchanged CPU grower,
    /// while computing the same RMSE derivatives the real `CpuBackend` would (`CpuBackend`
    /// is not compiled under rocm/cuda).
    struct CpuRefRuntime;

    impl Runtime for CpuRefRuntime {
        fn compute_gradients(
            &self,
            _loss: &Loss,
            approx: &[f64],
            target: &[f64],
            _approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            Ok(Derivatives {
                der1: approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect(),
                der2: approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect(),
            })
        }
    }

    /// The real [`GpuBackend`], instrumented so the suite can PROVE the device grow ran.
    struct CountingGpu {
        inner: GpuBackend,
        grown: Cell<usize>,
        begun: Cell<usize>,
    }

    impl CountingGpu {
        fn new() -> Self {
            Self {
                inner: GpuBackend::default(),
                grown: Cell::new(0),
                begun: Cell::new(0),
            }
        }
    }

    impl Runtime for CountingGpu {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            self.inner.compute_gradients(loss, approx, target, approx_dimension)
        }

        #[allow(clippy::too_many_arguments)]
        fn begin_device_training(
            &self,
            loss: &Loss,
            depth: usize,
            boosting_type_is_plain: bool,
            fold_count: usize,
            score_function: EScoreFunction,
            bins_feature_major: &[u32],
            weight: &[f64],
            n: usize,
            n_features: usize,
            n_bins: usize,
            learning_rate: f64,
            scaled_l2: f64,
            config: &DeviceTrainConfig,
        ) -> CbResult<bool> {
            let accepted = self.inner.begin_device_training(
                loss,
                depth,
                boosting_type_is_plain,
                fold_count,
                score_function,
                bins_feature_major,
                weight,
                n,
                n_features,
                n_bins,
                learning_rate,
                scaled_l2,
                config,
            )?;
            if accepted {
                self.begun.set(self.begun.get() + 1);
            }
            Ok(accepted)
        }

        fn grow_tree_on_device(
            &self,
            approx: &[f64],
            target: &[f64],
            sample: &[f64],
        ) -> CbResult<Option<DeviceGrownTree>> {
            let out = self.inner.grow_tree_on_device(approx, target, sample)?;
            if out.is_some() {
                self.grown.set(self.grown.get() + 1);
            }
            Ok(out)
        }

        fn end_device_training(&self) -> CbResult<()> {
            self.inner.end_device_training()
        }
    }

    /// A pool: `n_float` quantized float ramps (32 bins each) + the given cat columns.
    fn pool(
        n: usize,
        n_float: usize,
        cats: Vec<Vec<String>>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<Vec<String>>, Vec<f64>) {
        let mut columns = Vec::with_capacity(n_float);
        let mut borders = Vec::with_capacity(n_float);
        for f in 0..n_float {
            columns.push((0..n).map(|i| ((i * 7 + f * 13) % 32) as f32).collect::<Vec<f32>>());
            borders.push((0..31).map(|k| k as f64 + 0.5).collect::<Vec<f64>>());
        }
        // A target that genuinely depends on BOTH the float ramp and the cat columns, so a
        // correct one-hot split is worth choosing.
        let target: Vec<f64> = (0..n)
            .map(|i| {
                let a = if n_float > 0 { f64::from(columns[0][i]) } else { 0.0 };
                let c: f64 = cats
                    .iter()
                    .map(|col| if col[i] == "a" { 3.0 } else { -2.0 })
                    .sum();
                (a * 0.31).sin() + c + ((i % 11) as f64) * 0.05
            })
            .collect();
        (columns, borders, cats, target)
    }

    /// Train the same pool on the device and on the CPU grower, asserting BOTH
    /// anti-false-pass guards, and return `(device_predictions, cpu_predictions)`.
    fn both(
        label: &str,
        params: &BoostParams,
        n: usize,
        n_float: usize,
        cats: Vec<Vec<String>>,
    ) -> (Vec<f64>, Vec<f64>) {
        let (columns, borders, cats, target) = pool(n, n_float, cats);

        let gpu = CountingGpu::new();
        let (dev_trained, _) = train_cat(&gpu, &columns, &borders, &cats, &target, &[], params, None)
            .unwrap_or_else(|e| panic!("[{label}] device train_cat failed: {e:?}"));
        let (cpu_trained, _) =
            train_cat(&CpuRefRuntime, &columns, &borders, &cats, &target, &[], params, None)
                .unwrap_or_else(|e| panic!("[{label}] cpu train_cat failed: {e:?}"));

        // ── Guard 1: the device really grew every tree. ──
        assert_eq!(
            gpu.begun.get(),
            1,
            "[{label}] the backend must ACCEPT exactly one device session; 0 means the \
             eligibility gate declined and the 'device' fit is really a CPU fit"
        );
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "[{label}] the device must grow every tree ({} expected); a shortfall means a \
             silent CPU fallback",
            params.iterations
        );

        let dev_model = CbModel::from_trained(&dev_trained, borders.clone());
        let cpu_model = CbModel::from_trained(&cpu_trained, borders.clone());

        // ── Guard 2: the device tree actually USED a one-hot split. ──
        let one_hot_count = dev_model
            .oblivious_trees
            .iter()
            .flat_map(|t| &t.splits)
            .filter(|s| matches!(s, ModelSplit::OneHot(_)))
            .count();
        assert!(
            one_hot_count >= 1,
            "[{label}] the device-trained model carries NO one-hot split, so the equality \
             fold was never exercised and the parity assertion below proves nothing"
        );

        let dev_pred = predict_raw_cat(&dev_model, &columns, &cats);
        let cpu_pred = predict_raw_cat(&cpu_model, &columns, &cats);
        (dev_pred, cpu_pred)
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f64::max)
    }

    fn binary_cat(n: usize, period: usize) -> Vec<String> {
        (0..n)
            .map(|i| if (i / period) % 2 == 0 { "a" } else { "b" }.to_owned())
            .collect()
    }

    /// The headline gate: 1 float column + 2 binary cat columns, depth 3, 5 iterations.
    #[test]
    fn device_one_hot_training_matches_cpu_within_1e5() {
        let n = 512usize;
        let params = one_hot_params(3, 5, 2);
        let cats = vec![binary_cat(n, 3), binary_cat(n, 7)];
        let (dev, cpu) = both("headline", &params, n, 1, cats);
        let diff = max_abs_diff(&dev, &cpu);
        println!("[T28 headline] max|device - cpu| = {diff:.3e} (bar {EPS:.0e})");
        assert!(diff <= EPS, "device/CPU one-hot parity {diff:.3e} exceeds {EPS:.0e}");
    }

    /// The ONLY assertion in the plan that can catch a wrong `real_folds` DATA SOURCE.
    ///
    /// It runs through the PRODUCTION path `train_cat` → `device_host_eligible` →
    /// `begin_device_training` → `grow_oblivious_tree_resident`, never a hand-supplied
    /// `real_folds`. That is the whole point: the scorer unit test
    /// (`gpu_runtime::one_hot_split_score_test::one_hot_padded_bins_never_win`)
    /// hand-supplies the array and therefore passes even if production feeds the padded
    /// line width instead.
    ///
    /// The pool maximizes the gap between the two: a float column with 31 borders
    /// (`n_bins = 32 = n_bins_line`) plus cardinality-2 cat columns whose padded bins
    /// `2..32` would be eligible under the wrong bound. The second sub-case adds a column
    /// with an INTERIOR bin absent from the data (a gap), so the CPU and device candidate
    /// sets differ in a second, independent way if the rule is wrong.
    ///
    /// Failure localization for the seam itself lives in
    /// `cb_backend::gpu_runtime::one_hot_session_wiring_test`, which asserts
    /// `real_folds == [32, 2, 2]` directly; a session read-back is unreachable from here
    /// (`GpuTrainSession` is private to `cb-backend`).
    #[test]
    fn device_one_hot_parity_with_a_padded_and_a_gap_bin() {
        let n = 512usize;
        let params = one_hot_params(3, 4, 4);

        // (a) cardinality-2 columns against a 32-wide padded line.
        let cats = vec![binary_cat(n, 5), binary_cat(n, 11)];
        let (dev, cpu) = both("padded", &params, n, 1, cats);
        let diff = max_abs_diff(&dev, &cpu);
        println!("[T28 padded] max|device - cpu| = {diff:.3e}");
        assert!(
            diff <= EPS,
            "padded-bin parity {diff:.3e} exceeds {EPS:.0e}: a phantom padded-bin candidate \
             won device-side, i.e. one-hot candidates are bounded by the PADDED line width \
             instead of `real_folds`"
        );

        // (b) a column whose interior category is absent from the training data. Its
        // `PerfectHash` cardinality is 3 while only 2 distinct bins actually occur in a
        // contiguous block, so a wrong bound diverges here in a second, independent way.
        let gap: Vec<String> = (0..n)
            .map(|i| match i % 6 {
                0 | 1 => "x",
                2 | 3 => "y",
                _ => "z",
            }
            .to_owned())
            .collect();
        let cats = vec![gap, binary_cat(n, 9)];
        let (dev, cpu) = both("gap", &params, n, 1, cats);
        let diff = max_abs_diff(&dev, &cpu);
        println!("[T28 gap] max|device - cpu| = {diff:.3e}");
        assert!(diff <= EPS, "gap-bin parity {diff:.3e} exceeds {EPS:.0e}");
    }

    /// Depth 3 over a mixed pool: exercises the `leaf_stride` addressing on partitions
    /// >= 1, which a single-partition test cannot reach.
    #[test]
    fn device_one_hot_parity_at_depth_three_for_a_mixed_pool() {
        let n = 1024usize;
        let params = one_hot_params(3, 3, 2);
        let cats = vec![binary_cat(n, 2), binary_cat(n, 13)];
        let (dev, cpu) = both("depth3-mixed", &params, n, 3, cats);
        let diff = max_abs_diff(&dev, &cpu);
        println!("[T28 depth3-mixed] max|device - cpu| = {diff:.3e}");
        assert!(diff <= EPS, "depth-3 mixed parity {diff:.3e} exceeds {EPS:.0e}");
    }
}
