//! Device ROUTING for every parameter this wave added: each one either commits
//! the fit to the device or DECLINES to the CPU grower, and which it does is
//! asserted here rather than left to accident.
//!
//! A silent decline is the dangerous case. The device branch `continue`s past the
//! whole CPU leaf-value section, so a parameter implemented only there would be
//! quietly ignored on a device fit — the caller would get a model they did not
//! ask for, with no error. `leaf_estimation_iterations` was exactly that bug
//! until this file's `multi_step_leaf_estimation_declines` was written.
//!
//! The observation mechanism is the established `CountingGpu` wrapper: it counts
//! device tree grows, so `0` means the fit fell back to the CPU grower and
//! `iterations` means it committed.
//!
//! Run with: `cargo test -p cb-train --no-default-features --features rocm \
//!            --test string_param_device_routing_test`
//! (a blanket `--features rocm` leaves the default `cpu` feature on and
//! `SelectedRuntime` silently resolves to cubecl-cpu — a false negative).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EBoostingType, EGrowPolicy, EOverfittingDetectorType};

/// A device-eligible baseline: plain, symmetric, unweighted, no sampling.
fn base_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 4,
        depth: 3,
        learning_rate: 0.3,
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
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
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
    use std::cell::Cell;

    use super::base_params;
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_train::{train, BoostParams};

    pub struct CountingGpu {
        pub inner: GpuBackend,
        pub grown: Cell<usize>,
    }

    impl CountingGpu {
        fn new() -> Self {
            Self {
                inner: GpuBackend::default(),
                grown: Cell::new(0),
            }
        }
    }

    impl Runtime for CountingGpu {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            dim: usize,
        ) -> CbResult<cb_compute::Derivatives> {
            self.inner.compute_gradients(loss, approx, target, dim)
        }

        #[allow(clippy::too_many_arguments)]
        fn begin_device_training(
            &self,
            loss: &Loss,
            depth: usize,
            plain: bool,
            fold_count: usize,
            score_function: EScoreFunction,
            bins: &[u32],
            weight: &[f64],
            n: usize,
            n_features: usize,
            n_bins: usize,
            lr: f64,
            scaled_l2: f64,
            config: &DeviceTrainConfig,
        ) -> CbResult<bool> {
            self.inner.begin_device_training(
                loss, depth, plain, fold_count, score_function, bins, weight, n, n_features,
                n_bins, lr, scaled_l2, config,
            )
        }

        /// Forwarding this is NOT optional. The `Runtime` trait's default for the
        /// QPACK-01 raw seam DECLINES, so a wrapper that only overrides
        /// `begin_device_training` silently routes every fit through the
        /// host-quantize channel — the fit still commits, `grown` still counts, and
        /// the test looks like it covered the device while the on-GPU quantizer
        /// never ran. That is exactly what hid the `nan_mode=Max` sentinel bug from
        /// the first version of `nan_max_commits_and_matches_cpu`.
        #[allow(clippy::too_many_arguments)]
        fn begin_device_training_raw(
            &self,
            loss: &Loss,
            depth: usize,
            plain: bool,
            fold_count: usize,
            score_function: EScoreFunction,
            float_columns: &[Vec<f32>],
            feature_borders: &[Vec<f64>],
            weight: &[f64],
            n: usize,
            n_features: usize,
            n_bins: usize,
            lr: f64,
            scaled_l2: f64,
            config: &DeviceTrainConfig,
        ) -> CbResult<bool> {
            self.inner.begin_device_training_raw(
                loss,
                depth,
                plain,
                fold_count,
                score_function,
                float_columns,
                feature_borders,
                weight,
                n,
                n_features,
                n_bins,
                lr,
                scaled_l2,
                config,
            )
        }

        fn grow_tree_on_device(
            &self,
            approx: &[f64],
            target: &[f64],
            sample: &[f64],
            family: Option<&FamilyTreeArgs<'_>>,
        ) -> CbResult<Option<DeviceGrownTree>> {
            let out = self.inner.grow_tree_on_device(approx, target, sample, family)?;
            if out.is_some() {
                self.grown.set(self.grown.get() + 1);
            }
            Ok(out)
        }

        fn end_device_training(&self) -> CbResult<()> {
            self.inner.end_device_training()
        }
    }

    /// The CPU baseline for a differential. `CpuBackend` is gated behind the `cpu`
    /// feature, which is OFF in this lane (`--no-default-features --features rocm`),
    /// so the CPU grower is reached instead by declining at the backend seam: this
    /// wrapper implements only `compute_gradients` and inherits the `Runtime` trait's
    /// default `begin_device_training` (`Ok(false)`), which is precisely the
    /// "backend declines" path the D-04 invariant covers. Same params, same corpus,
    /// byte-unchanged CPU grower.
    struct HostOnly(GpuBackend);

    impl Runtime for HostOnly {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            dim: usize,
        ) -> CbResult<cb_compute::Derivatives> {
            self.0.compute_gradients(loss, approx, target, dim)
        }
    }

    /// A deterministic numeric corpus with enough distinct values to give every
    /// feature real borders.
    fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
        let n = 256;
        let mut state = 987_654_321_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n); 3];
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let a = next();
            let b = next();
            let c = next();
            cols[0].push(a as f32);
            cols[1].push(b as f32);
            cols[2].push(c as f32);
            target.push(2.0 * a - b + 0.5 * c);
        }
        let borders: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| cb_data::select_borders_greedy_logsum_f32(c, 32, false))
            .collect();
        (cols, borders, target)
    }

    /// Run one fit and report how many trees the DEVICE grew.
    fn device_grows(params: &BoostParams) -> usize {
        let (cols, borders, target) = corpus();
        let weights = vec![1.0_f64; target.len()];
        let gpu = CountingGpu::new();
        train(&gpu, &cols, &borders, &target, &weights, params, None)
            .expect("the fit must succeed on whichever path it takes");
        gpu.grown.get()
    }

    /// The baseline MUST commit, otherwise every "declines" assertion below is
    /// vacuous — a fit that never reaches the device would trivially report 0.
    pub fn baseline_commits() {
        let params = base_params();
        assert_eq!(
            device_grows(&params),
            params.iterations,
            "the device-eligible baseline must commit every tree to the device; \
             without this the decline assertions prove nothing"
        );
    }

    /// `model_shrink_rate != 0` declines: the shrink rescales the running approx
    /// each iteration, but the device keeps its approx RESIDENT and never reads
    /// the host copy back per tree, so a host-side rescale would be dropped.
    pub fn model_shrink_declines() {
        let params = BoostParams {
            extra: cb_train::ExtraBoostParams {
                model_shrink_rate: 0.1,
                ..Default::default()
            },
            ..base_params()
        };
        assert_eq!(device_grows(&params), 0, "model_shrink_rate must decline");
    }

    /// `leaf_estimation_iterations > 1` declines. The accumulate-and-recompute
    /// loop lives in the CPU leaf-value section, which the device branch skips —
    /// committing would silently take ONE step and ignore the parameter.
    ///
    /// `backtracking = No` is required, not incidental: the default
    /// `AnyImprovement` is rejected outright at N > 1 (the step-shrinking SEARCH
    /// is unimplemented), so leaving it would make this test observe THAT guard
    /// instead of the routing decision it is here to pin.
    pub fn multi_step_leaf_estimation_declines() {
        let params = BoostParams {
            extra: cb_train::ExtraBoostParams {
                leaf_estimation_iterations: 3,
                leaf_estimation_backtracking: cb_compute::LeafEstimationBacktracking::No,
                ..Default::default()
            },
            ..base_params()
        };
        assert_eq!(
            device_grows(&params),
            0,
            "leaf_estimation_iterations > 1 must decline — the multi-step estimator \
             is CPU-only, so committing would silently ignore it"
        );
    }

    /// `rsm < 1` declines. The device grow loop scores every quantized feature at
    /// every level and never touches the host learn RNG, so committing would both
    /// ignore the per-level candidate mask (training the `rsm = 1` model) and skip
    /// the `GenRandReal1()` draws that the mask is made of — leaving the RNG phase
    /// wrong for everything downstream. Exactly the silently-wrong-model class
    /// this file exists to catch.
    pub fn rsm_declines() {
        let params = BoostParams {
            extra: cb_train::ExtraBoostParams {
                rsm: 0.5,
                ..Default::default()
            },
            ..base_params()
        };
        assert_eq!(
            device_grows(&params),
            0,
            "rsm < 1 must decline — the device grower has no per-level candidate mask"
        );
    }

    /// ...and the DEFAULT `rsm` must still commit, so the decline above is a real
    /// routing decision rather than this parameter having disabled the device path
    /// outright.
    pub fn default_rsm_still_commits() {
        let params = BoostParams {
            extra: cb_train::ExtraBoostParams {
                rsm: 1.0,
                ..Default::default()
            },
            ..base_params()
        };
        assert_eq!(
            device_grows(&params),
            params.iterations,
            "rsm = 1.0 is the default and must stay device-eligible"
        );
    }

    /// `nan_mode=Max` COMMITS to the device, and the device model must equal the CPU
    /// model. Quantization params reach `cb-train` only as BORDERS, so there is no
    /// eligibility clause to read — the honest question is not "does it decline" but
    /// "is the model the same on both paths", and only a differential answers it.
    ///
    /// The regime matters: `Max` is carried by an appended `f32::MAX` SENTINEL
    /// border, and the fit is float-only + SymmetricTree, which is exactly the
    /// QPACK-01 raw channel where the device quantizes on-GPU. A device quantizer
    /// blind to the sentinel bins NaN to 0 — the `Min` answer — and this test is
    /// what separates the two.
    pub fn nan_max_commits_and_matches_cpu() {
        let (cols, mut borders, target) = corpus();
        // Make feature 0 NaN-bearing and give it the `nan_mode=Max` sentinel.
        let mut cols = cols;
        for (i, v) in cols[0].iter_mut().enumerate() {
            if i % 5 == 0 {
                *v = f32::NAN;
            }
        }
        borders[0].push(f64::from(f32::MAX));
        let weights = vec![1.0_f64; target.len()];
        let params = base_params();

        let gpu = CountingGpu::new();
        let device_model = train(&gpu, &cols, &borders, &target, &weights, &params, None)
            .expect("the NaN-sentinel device fit must succeed");
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "a NaN-sentinel column must still COMMIT to the device — quantization is a \
             host-side border choice with no eligibility clause of its own"
        );

        let cpu_model = train(
            &HostOnly(GpuBackend::default()),
            &cols,
            &borders,
            &target,
            &weights,
            &params,
            None,
        )
        .expect("the CPU baseline fit must succeed");

        let device_leaves: Vec<f64> = device_model
            .oblivious_trees
            .iter()
            .flat_map(|t| t.leaf_values.iter().copied())
            .collect();
        let cpu_leaves: Vec<f64> = cpu_model
            .oblivious_trees
            .iter()
            .flat_map(|t| t.leaf_values.iter().copied())
            .collect();
        assert_eq!(
            device_leaves.len(),
            cpu_leaves.len(),
            "device and CPU models differ in shape on a NaN-sentinel fit"
        );
        for (i, (d, c)) in device_leaves.iter().zip(cpu_leaves.iter()).enumerate() {
            assert!(
                (d - c).abs() <= 1e-9,
                "leaf {i}: device {d} vs CPU {c} — the device quantizer is treating \
                 the f32::MAX sentinel differently from the host"
            );
        }
    }

    /// The two CTR-mode params and `allow_const_label` do not touch the grow loop,
    /// so they must leave routing alone.
    pub fn non_grow_params_stay_device_eligible() {
        let params = BoostParams {
            extra: cb_train::ExtraBoostParams {
                final_ctr_computation_mode: cb_train::EFinalCtrComputationMode::Skip,
                allow_const_label: true,
                ..Default::default()
            },
            ..base_params()
        };
        assert_eq!(
            device_grows(&params),
            params.iterations,
            "final_ctr_computation_mode / allow_const_label must not change routing \
             on a numeric fit"
        );
    }

    /// EVERY `feature_border_type` must produce the same model on the device as on
    /// the CPU. The parameter never reaches `cb-train` as a flag — it reaches it only
    /// as BORDER VALUES — so "does it decline" is the wrong question and a
    /// differential is the only honest one.
    ///
    /// It is not a formality. The raw device channel requires each border to
    /// round-trip `f64 -> f32 -> f64` EXACTLY and declines otherwise, and the seven
    /// types build their borders by three different midpoint formulas over different
    /// value sets. A type whose borders fail the round-trip falls back to the
    /// host-quantize channel — correct, but a DIFFERENT path — and a type whose
    /// borders pass it exercises the on-GPU quantizer. This asserts the model is the
    /// same either way, so which path a type takes stays an implementation detail
    /// rather than a silent behaviour change.
    pub fn every_border_type_matches_cpu_on_device() {
        let (cols, _, target) = corpus();
        let weights = vec![1.0_f64; target.len()];
        let params = base_params();

        for border_type in cb_data::EBorderSelectionType::all() {
            let borders: Vec<Vec<f64>> = cols
                .iter()
                .map(|c| cb_data::select_borders_f32(c, 32, border_type, false))
                .collect();
            // A border type that produced NO borders would make every fit trivially
            // agree; that is a vacuous cell, not a passing one.
            assert!(
                borders.iter().any(|b| !b.is_empty()),
                "{border_type:?} produced no borders on this corpus — the comparison \
                 below would be vacuous"
            );

            let gpu = CountingGpu::new();
            let device_model = train(&gpu, &cols, &borders, &target, &weights, &params, None)
                .unwrap_or_else(|e| panic!("{border_type:?}: device fit failed: {e}"));
            assert_eq!(
                gpu.grown.get(),
                params.iterations,
                "{border_type:?}: a border choice must not change device routing"
            );

            let cpu_model = train(
                &HostOnly(GpuBackend::default()),
                &cols,
                &borders,
                &target,
                &weights,
                &params,
                None,
            )
            .unwrap_or_else(|e| panic!("{border_type:?}: CPU baseline fit failed: {e}"));

            let d: Vec<f64> = device_model
                .oblivious_trees
                .iter()
                .flat_map(|t| t.leaf_values.iter().copied())
                .collect();
            let c: Vec<f64> = cpu_model
                .oblivious_trees
                .iter()
                .flat_map(|t| t.leaf_values.iter().copied())
                .collect();
            assert_eq!(d.len(), c.len(), "{border_type:?}: model shape differs");
            for (i, (dv, cv)) in d.iter().zip(c.iter()).enumerate() {
                assert!(
                    (dv - cv).abs() <= 1e-9,
                    "{border_type:?} leaf {i}: device {dv} vs CPU {cv}"
                );
            }
        }
    }

    /// `random_score_type` is only consulted when `random_strength != 0`, and a
    /// non-zero strength already declines — so the score type can never reach the
    /// device path. Pinned so a future relaxation of the random_strength clause
    /// cannot silently start ignoring the distribution.
    pub fn random_score_type_is_unreachable_on_device() {
        let params = BoostParams {
            random_strength: 1.0,
            extra: cb_train::ExtraBoostParams {
                random_score_type: cb_compute::ERandomScoreType::Gumbel,
                ..Default::default()
            },
            ..base_params()
        };
        assert_eq!(
            device_grows(&params),
            0,
            "random_strength != 0 must decline, which is what keeps random_score_type \
             off the device path"
        );
    }
}

macro_rules! device_test {
    ($name:ident, $body:path) => {
        #[test]
        fn $name() {
            #[cfg(any(feature = "rocm", feature = "cuda"))]
            $body();
            #[cfg(not(any(feature = "rocm", feature = "cuda")))]
            eprintln!(
                "SKIP {}: needs --no-default-features --features rocm (or cuda)",
                stringify!($name)
            );
        }
    };
}

device_test!(device_baseline_commits, device::baseline_commits);
device_test!(model_shrink_rate_declines, device::model_shrink_declines);
device_test!(
    multi_step_leaf_estimation_declines,
    device::multi_step_leaf_estimation_declines
);
device_test!(
    nan_max_commits_and_matches_cpu,
    device::nan_max_commits_and_matches_cpu
);
device_test!(
    non_grow_params_stay_device_eligible,
    device::non_grow_params_stay_device_eligible
);
device_test!(
    every_border_type_matches_cpu_on_device,
    device::every_border_type_matches_cpu_on_device
);
device_test!(
    random_score_type_is_unreachable_on_device,
    device::random_score_type_is_unreachable_on_device
);
device_test!(rsm_declines, device::rsm_declines);
device_test!(default_rsm_still_commits, device::default_rsm_still_commits);
