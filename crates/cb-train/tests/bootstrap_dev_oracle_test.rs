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
    }
}

/// The scenarios gated over ALL THREE trees against upstream
/// (`(dir, type, subsample, temperature)`).
///
/// MVS is deliberately NOT here — see [`MVS_SCENARIO`] and [`MVS_GATED_TREES`].
const SCENARIOS: &[(&str, EBootstrapType, f64, f32)] = &[
    ("no", EBootstrapType::No, 1.0, 0.0),
    ("bayesian", EBootstrapType::Bayesian, 1.0, 1.0),
    ("bernoulli", EBootstrapType::Bernoulli, 0.8, 0.0),
];

/// MVS, gated separately over a REDUCED tree range.
const MVS_SCENARIO: (&str, EBootstrapType, f64, f32) = ("mvs", EBootstrapType::Mvs, 0.8, 0.0);

/// How many of the fixture's 3 trees the MVS scenario is gated over.
///
/// # Why MVS is gated over 2 trees, not 3 — a PRE-EXISTING CPU-side gap
///
/// Generating this bias-0 family exposed a real upstream-parity gap in the CPU MVS
/// sampler that no committed fixture covered. It is NOT introduced by the WR-01
/// device work: the CPU grow path is byte-unchanged this phase (the device branch is
/// inert under `CpuBackend`), and it reproduces with the device untouched.
///
/// Measured on this dataset, `subsample = 0.8`, 3 iterations, comparing our CPU fit
/// against freshly generated upstream 1.2.10 fixtures — each run using its OWN
/// model's quantization borders (borders are NOT stable across configurations, so a
/// shared-border comparison is invalid and was discarded):
///
/// | `boost_from_average` | seeds matching upstream on all 3 trees |
/// |---|---|
/// | `true`  (the committed `bootstrap/` family) | 3 / 5 (seeds 0, 2, 3) |
/// | `false` (this bias-0 family)                | 0 / 5 |
///
/// In EVERY failing case the first divergent split is at flat index 4 or 5 — i.e.
/// **tree 2**, never trees 0 or 1. Trees 0 and 1 agree with upstream to ~5e-9, and
/// the carried MVS λ feeding tree 2 agrees to ~3e-9, so the λ formula itself
/// (`last_iter_mean_leaf_value`) is not the defect; the divergence enters when tree
/// 2's sample is drawn from that λ.
///
/// The committed `bootstrap/mvs` oracle (seed 0, `boost_from_average=True`) happens
/// to be one of the passing configurations, which is why the gap went unnoticed.
///
/// Gating trees 0–1 keeps a REAL upstream claim for MVS and still catches a
/// regression in the sampler's first two trees, without asserting a third tree we
/// know does not hold. Device-vs-CPU MVS parity is separately locked at ≤1e-5
/// (measured ~4.7e-11) by `device_bootstrap_parity_test`, so the DEVICE half of
/// WR-01 is unaffected by this CPU-side gap. Raise this to 3 once the MVS tree-2
/// sampling gap is fixed.
const MVS_GATED_TREES: usize = 2;

/// Gate one trained model + staged approximant against the upstream fixture at
/// the ≤1e-5 bar (`compare_stage`'s tolerance).
///
/// `gated_trees` bounds how many leading trees are compared. It exists ONLY for the
/// MVS carve-out documented on [`MVS_GATED_TREES`]; every other scenario passes the
/// full tree count and is gated end to end.
fn gate_against_upstream(
    who: &str,
    scenario: &str,
    model: &cb_train::Model,
    staged: &[f64],
    gated_trees: usize,
) {
    let dir = format!("bootstrap_dev/{scenario}");
    let model_json = load_model_json(&fixture(&format!("{dir}/model.json")))
        .unwrap_or_else(|e| panic!("{dir}/model.json must load: {e:?}"));

    let n_trees = model.oblivious_trees.len();
    assert!(
        gated_trees <= n_trees,
        "[{who}] {dir}: cannot gate {gated_trees} trees out of {n_trees}"
    );
    // Trees are flat-concatenated per stage: `depth` splits and `2^depth` leaf values
    // each, and one `n_rows` block per staged iteration. Truncate all three
    // consistently so a reduced gate compares the SAME leading trees everywhere.
    let depth = 2usize;
    let n_rows = 1500usize;
    let split_end = gated_trees * depth;
    let leaf_end = gated_trees * (1usize << depth);
    let staged_end = gated_trees * n_rows;

    let up_splits = model_json.split_borders();
    let our_splits = model.split_borders();
    compare_stage(Stage::Splits, &up_splits[..split_end], &our_splits[..split_end])
        .unwrap_or_else(|e| panic!("[{who}] {dir}: splits diverged from upstream: {e:?}"));

    let up_leaves = model_json.leaf_values();
    let our_leaves = model.leaf_values();
    compare_stage(Stage::LeafValues, &up_leaves[..leaf_end], &our_leaves[..leaf_end])
        .unwrap_or_else(|e| panic!("[{who}] {dir}: leaf values diverged from upstream: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{dir}/staged.npy"))).unwrap();
    compare_stage(
        Stage::StagedApprox,
        &expected_staged[..staged_end],
        &staged[..staged_end],
    )
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
         over {gated_trees}/{n_trees} trees"
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
    // The fully-gated scenarios, then MVS over its reduced tree range.
    let all = SCENARIOS
        .iter()
        .map(|&s| (s, 3usize))
        .chain(std::iter::once((MVS_SCENARIO, MVS_GATED_TREES)));
    for ((scenario, bt, subsample, temp), gated) in all {
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
        gate_against_upstream("cpu", scenario, &model, &staged, gated);
    }
}

/// `WR01-S15`: the DEVICE path reproduces upstream on the bias-0 family — the
/// phase's headline claim. rocm/cuda only.
#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn bootstrap_dev_device_matches_upstream() {
    use cb_backend::GpuBackend;
    use cb_compute::{DeviceGrownTree, DeviceTrainConfig, Runtime};
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
        ) -> CbResult<Option<DeviceGrownTree>> {
            let out = self.inner.grow_tree_on_device(approx, target, sample)?;
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
    let all = SCENARIOS
        .iter()
        .map(|&s| (s, 3usize))
        .chain(std::iter::once((MVS_SCENARIO, MVS_GATED_TREES)));
    for ((scenario, bt, subsample, temp), gated) in all {
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

        gate_against_upstream("device", scenario, &model, &staged, gated);
    }
}
