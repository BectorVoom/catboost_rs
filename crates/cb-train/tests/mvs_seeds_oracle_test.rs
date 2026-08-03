//! `MVS-S3` — the MVS multi-seed × bias upstream oracle.
//!
//! # Why this file exists
//!
//! `bootstrap/mvs` pins exactly ONE configuration (`random_seed = 0`,
//! `boost_from_average = True`). A defect that made the CPU MVS sampler consume two
//! extra RNG draws per tree — shifting the sampled subset of every tree from the
//! first onward — left that single scenario **green** while breaking **7 of 10**
//! seed/bias combinations. A wrong subset only becomes visible when it actually flips
//! a split argmax, so one scenario cannot discriminate. This family can.
//!
//! Measured against upstream 1.2.10 with the defect present:
//! `boost_from_average=true` failed on seeds 1 and 4; `boost_from_average=false`
//! failed on all five. After the fix, all ten agree to ≤1e-5 over all 3 trees.
//!
//! # The border trap
//!
//! CatBoost quantization borders are **not** stable across configurations — they were
//! observed to move with `subsample` on identical data. Every scenario therefore reads
//! **its own** `model.json`'s `float_feature_borders()`. Reusing one border set across
//! scenarios silently invalidates the comparison and can turn a real divergence into a
//! pass (or vice versa).
//!
//! CPU-feature oracle: it gates the host sampler, which the device path shares
//! verbatim, so there is nothing device-specific to assert here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{BoostParams, EBootstrapType, EOverfittingDetectorType};
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

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

/// The generator's pinned parameters. Every knob catboost's raw dict API defaults
/// differently from `BoostParams` is set EXPLICITLY (`random_strength = 0` above all),
/// on BOTH sides, so a silent default drift cannot masquerade as a sampler bug.
fn params(seed: u64, boost_from_average: bool) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 3,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::Mvs,
        subsample: 0.8,
        bagging_temperature: 0.0,
        random_seed: seed,
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

/// `MVS-S3`: every `(seed, bias)` scenario matches upstream at ≤1e-5 over all 3 trees.
#[cfg(not(any(feature = "rocm", feature = "cuda")))]
#[test]
fn mvs_seeds_cpu_matches_upstream_across_seeds_and_bias() {
    use cb_backend::CpuBackend;
    use cb_train::train;

    let columns = load_feature_columns();
    let target = load_target();
    let mut checked = 0usize;

    for seed in 0..5_u64 {
        for bfa in [false, true] {
            let name = format!("s{seed}_bfa{}", u8::from(bfa));
            let dir = format!("mvs_seeds/{name}");
            // THIS scenario's own borders — never a shared set (see the module docs).
            let model_json = load_model_json(&fixture(&format!("{dir}/model.json")))
                .unwrap_or_else(|e| panic!("{dir}/model.json must load: {e:?}"));
            let borders = model_json.float_feature_borders();

            let mut staged = Vec::new();
            let model = train(
                &CpuBackend,
                &columns,
                &borders,
                &target,
                &[],
                &params(seed, bfa),
                Some(&mut staged),
            )
            .unwrap_or_else(|e| panic!("{dir}: training failed: {e:?}"));

            compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
                .unwrap_or_else(|e| panic!("{dir}: splits diverged from upstream: {e:?}"));
            compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
                .unwrap_or_else(|e| panic!("{dir}: leaf values diverged from upstream: {e:?}"));
            let expected_staged =
                load_f64_vec(&fixture(&format!("{dir}/staged.npy"))).unwrap();
            compare_stage(Stage::StagedApprox, &expected_staged, &staged)
                .unwrap_or_else(|e| panic!("{dir}: staged approx diverged from upstream: {e:?}"));

            // The bias setting must actually differ across the family, or half the
            // scenarios are duplicates and the discriminating power is halved.
            assert_eq!(
                model.bias.abs() < 1e-12,
                !bfa,
                "{dir}: boost_from_average={bfa} must yield a {} bias",
                if bfa { "non-zero" } else { "zero" }
            );

            checked += 1;
            println!("[mvs_seeds] {dir}: within 1e-5 of upstream over all 3 trees");
        }
    }

    // Guard against a silently-empty loop (e.g. a fixture root that failed to
    // generate would otherwise make this test vacuously green).
    assert_eq!(checked, 10, "all 10 (seed, bias) scenarios must be gated");
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn mvs_seeds_oracle_skipped_on_device_build() {
    // `CpuBackend` is not compiled under rocm/cuda. This oracle gates the HOST
    // sampler, which the device path shares verbatim, so there is nothing lost by
    // skipping it here — but print rather than pass silently.
    println!("SKIP mvs_seeds_oracle: CPU-feature oracle");
}
