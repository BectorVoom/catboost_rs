//! WR-01 `WR01-S14` / `WR01-S15` — the BIAS-0 upstream bootstrap oracle.
//!
//! The committed `bootstrap/` family trains with `boost_from_average=True`, which
//! the device eligibility gate excludes (the device grower seeds its resident
//! approx to zero, so a non-zero starting bias would train against the wrong
//! starting point — the CR-01 gate). That family therefore cannot, even in
//! principle, hold the device to an UPSTREAM oracle.
//!
//! `bootstrap_dev/` is its bias-0 sibling: identical dataset, seed, iteration
//! count, depth and sampler parameters, differing ONLY in
//! `boost_from_average=False`. It gates two distinct claims at ≤1e-5:
//!
//! - **CPU vs upstream** (`WR01-S14`) — runs everywhere, and is the gate that
//!   proves the FIXTURE itself is sound. Without it, a device-vs-upstream failure
//!   could not be attributed between the fixture and the device.
//! - **Device vs upstream** (`WR01-S15`) — runs on rocm/cuda only. This is the
//!   phase's strongest claim: the device grower reproduces upstream CatBoost
//!   1.2.10, not merely the in-repo CPU grower.
//!
//! Poisson has no scenario here for the same reason it has none in
//! `bootstrap_oracle_test`: upstream rejects it on CPU, so no Python oracle
//! exists. Its backend-independent rejection is covered by
//! `device_bootstrap_parity_test`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{BoostParams, EBootstrapType, EOverfittingDetectorType};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The frozen multi-block input, shared byte-for-byte with the `bootstrap/` family.
fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/bootstrap_multiblock/X.npy"))
        .unwrap_or_else(|e| panic!("bootstrap_multiblock/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

fn load_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/bootstrap_multiblock/y.npy")).unwrap()
}

/// The bias-0 params the `bootstrap_dev/` generator pinned. Every knob catboost's
/// raw dict API defaults differently from this builder is set EXPLICITLY (notably
/// `random_strength = 0`), so a silent default drift on either side cannot masquerade
/// as a sampler bug.
fn params(
    bootstrap_type: EBootstrapType,
    subsample: f64,
    bagging_temperature: f32,
) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 3,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        // The single deliberate difference from the `bootstrap/` family.
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type,
        subsample,
        bagging_temperature,
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
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra: Default::default(),
    }
}

/// The four scenarios, each gated over ALL THREE trees against upstream
/// (`(dir, type, subsample, temperature)`).
///
/// MVS used to carry a reduced-tree carve-out (`MVS_GATED_TREES = 2`) because the CPU
/// sampler diverged from upstream partway through a fit. The cause was two fabricated
/// RNG draws in `cb_train::bootstrap`'s MVS arm — a per-tree phase drift, fixed by
/// deleting them — so MVS is now gated exactly like every other scenario.
const SCENARIOS: &[(&str, EBootstrapType, f64, f32)] = &[
    ("no", EBootstrapType::No, 1.0, 0.0),
    ("bayesian", EBootstrapType::Bayesian, 1.0, 1.0),
    ("bernoulli", EBootstrapType::Bernoulli, 0.8, 0.0),
    ("mvs", EBootstrapType::Mvs, 0.8, 0.0),
];

/// Gate one trained model + staged approximant against the upstream fixture at
/// the ≤1e-5 bar (`compare_stage`'s tolerance).
///
fn gate_against_upstream(
    who: &str,
    scenario: &str,
    model: &cb_train::Model,
    staged: &[f64],
) {
    let dir = format!("bootstrap_dev/{scenario}");
    let model_json = load_model_json(&fixture(&format!("{dir}/model.json")))
        .unwrap_or_else(|e| panic!("{dir}/model.json must load: {e:?}"));

    let n_trees = model.oblivious_trees.len();

    let up_splits = model_json.split_borders();
    let our_splits = model.split_borders();
    compare_stage(Stage::Splits, &up_splits, &our_splits)
        .unwrap_or_else(|e| panic!("[{who}] {dir}: splits diverged from upstream: {e:?}"));

    let up_leaves = model_json.leaf_values();
    let our_leaves = model.leaf_values();
    compare_stage(Stage::LeafValues, &up_leaves, &our_leaves)
        .unwrap_or_else(|e| panic!("[{who}] {dir}: leaf values diverged from upstream: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{dir}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, staged)
    .unwrap_or_else(|e| panic!("[{who}] {dir}: staged approx diverged from upstream: {e:?}"));

    // The fixture must be a bias-0 fit — this is the property that makes the family
    // device-reachable at all. A regenerated fixture that silently flipped
    // `boost_from_average` back on would otherwise turn the device test below into a
    // permanent, confusing failure.
    assert!(
        model.bias.abs() < 1e-12,
        "[{who}] {dir}: bootstrap_dev fixtures must be BIAS-0 (boost_from_average=False), \
         got bias {}",
        model.bias
    );
    println!(
        "[{who}] {dir}: splits + leaf values + staged within 1e-5 of upstream \
         over all {n_trees} trees"
    );
}

/// `WR01-S14`: the CPU path reproduces upstream on the bias-0 family. Runs on the
/// default `cpu` feature; this is what proves the FIXTURE is sound independently of
/// any device.
#[cfg(not(any(feature = "rocm", feature = "cuda")))]
#[test]
fn bootstrap_dev_cpu_matches_upstream() {
    use cb_backend::CpuBackend;
    use cb_train::train;

    let columns = load_feature_columns();
    let target = load_target();
    for &(scenario, bt, subsample, temp) in SCENARIOS {
        let model_json =
            load_model_json(&fixture(&format!("bootstrap_dev/{scenario}/model.json"))).unwrap();
        let borders = model_json.float_feature_borders();
        let mut staged = Vec::new();
        let model = train(
            &CpuBackend,
            &columns,
            &borders,
            &target,
            &[],
            &params(bt, subsample, temp),
            Some(&mut staged),
        )
        .unwrap_or_else(|e| panic!("bootstrap_dev/{scenario}: CPU training failed: {e:?}"));
        gate_against_upstream("cpu", scenario, &model, &staged);
    }
}

/// `WR01-S15`: the DEVICE path reproduces upstream on the bias-0 family — the
/// phase's headline claim. rocm/cuda only.
#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn bootstrap_dev_device_matches_upstream() {
    use cb_backend::GpuBackend;
    use cb_compute::{DeviceGrownTree, DeviceTrainConfig, FamilyTreeArgs, Runtime};
    use cb_core::CbResult;
    use cb_train::train;
    use std::cell::Cell;

    /// Counts the trees the device actually returned, so a silent CPU fallback
    /// cannot make this test pass while proving nothing about the device.
    struct CountingGpu {
        inner: GpuBackend,
        grown: Cell<usize>,
        sampled_trees: Cell<usize>,
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
                loss,
                depth,
                plain,
                fold_count,
                score_function,
                bins,
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
                if !sample.is_empty() {
                    self.sampled_trees.set(self.sampled_trees.get() + 1);
                }
            }
            Ok(out)
        }

        fn end_device_training(&self) -> CbResult<()> {
            self.inner.end_device_training()
        }
    }

    let columns = load_feature_columns();
    let target = load_target();
    for &(scenario, bt, subsample, temp) in SCENARIOS {
        let model_json =
            load_model_json(&fixture(&format!("bootstrap_dev/{scenario}/model.json"))).unwrap();
        let borders = model_json.float_feature_borders();
        let p = params(bt, subsample, temp);
        let gpu = CountingGpu {
            inner: GpuBackend::default(),
            grown: Cell::new(0),
            sampled_trees: Cell::new(0),
        };
        let mut staged = Vec::new();
        let model = train(&gpu, &columns, &borders, &target, &[], &p, Some(&mut staged))
            .unwrap_or_else(|e| panic!("bootstrap_dev/{scenario}: device training failed: {e:?}"));

        // Anti-false-pass: every tree must have come from the device, and every
        // SAMPLED scenario must have carried a real sample across the seam.
        assert_eq!(
            gpu.grown.get(),
            p.iterations,
            "bootstrap_dev/{scenario}: the device must grow all {} trees; a shortfall means \
             the fit silently fell back to the CPU grower and this is not a device oracle",
            p.iterations
        );
        let expect_sampled = if matches!(bt, EBootstrapType::No) { 0 } else { p.iterations };
        assert_eq!(
            gpu.sampled_trees.get(),
            expect_sampled,
            "bootstrap_dev/{scenario}: expected {expect_sampled} trees to carry a host sample \
             across the seam, saw {}",
            gpu.sampled_trees.get()
        );

        gate_against_upstream("device", scenario, &model, &staged);
    }
}
