//! permutation_count=4 end-to-end train→predict oracle (ORD-01 / SC-1 closure,
//! Plan 05-17) — the PRODUCTION-DEFAULT multi-permutation hard gate.
//!
//! The production default is `permutation_count=4` (`permutation_count_default()
//! == 4`). Plan 05-17 corrected `create_folds`'s per-fold RNG draw accounting (the
//! AveragingFold shuffle begins at call-count == learning_folds, discovered by the
//! instrumented C++ harness and committed as `rng_draw_accounting.json`) so the
//! pc=4 AveragingFold partition reproduces catboost 1.2.10's `[6,0,10,14]`
//! integer-exact. This oracle proves the fix END-TO-END: training the tensor_ctr_e2e
//! config family at `permutation_count=4` and predicting through the PRODUCTION
//! `cb_model::predict_raw_cat` apply path matches the committed upstream catboost
//! 1.2.10 pc=4 RawFormulaVal predictions (`predictions_pc4.npy`) ≤1e-5 across ALL
//! objects/trees.
//!
//! This is the closure of the SC-1 / ORD-01 blocking gap at the production default:
//! a wrong per-fold advance count would yield a different AveragingFold permutation,
//! hence different leaf values, hence predictions diverging > 1e-5 here. The test
//! runs unconditionally (never skipped / never ignore-attributed).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_model::Model as CbModel;
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use cb_train::{
    train_cat, BoostParams, EBootstrapType, EBoostingType, EOverfittingDetectorType,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const FIXTURE_SEED: u64 = 0;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
fn oracle_fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Resolve a path under `cb-train/tests/fixtures/` (the committed pc=4 predictions).
fn train_fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(rel)
}

/// Load the two categorical columns from `X_cat.npy` as per-feature `Vec<String>`
/// SoA columns (each integer code stringified via `stringify_int_category` — the
/// PLAIN integer form upstream's Pool hashed when the fixture was generated).
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(oracle_fixture("tensor_ctr_e2e/X_cat.npy"))
        .unwrap_or_else(|e| panic!("tensor_ctr_e2e/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The isolating TENSOR-CTR config at the PRODUCTION DEFAULT `permutation_count=4`
/// (otherwise identical to `tensor_ctr_e2e/config.json`): Plain boosting,
/// one_hot_max_size=1, max_ctr_complexity=2, simple_ctr/combinations_ctr
/// Borders:Prior=0.5, fold_len_multiplier=2.0, depth=2, iterations=5, lr=0.1,
/// l2=3.0, Gradient, bootstrap=No, random_strength=0, seed=0, Logloss.
fn tensor_ctr_params_pc4() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: FIXTURE_SEED,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 1,
        // The production default and the focus of this gate.
        permutation_count: 4,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
        max_ctr_complexity: 2,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_train::score_function_default(),
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
    }
}

/// FULL multi-tree pc=4 tensor-CTR train→predict ≤1e-5 vs upstream catboost
/// 1.2.10, through the production `cb_model::predict_raw_cat` apply path. The
/// closure gate for SC-1 / ORD-01 at the production default. Runs unconditionally.
#[test]
fn permutation_count_four_predictions_match_upstream() {
    let cat_cols = load_cat_columns();
    let target = load_f64_vec(&oracle_fixture("tensor_ctr_e2e/y.npy")).unwrap();
    let expected_predictions =
        load_f64_vec(&train_fixture("multi_permutation_fold/predictions_pc4.npy"))
            .unwrap_or_else(|e| panic!("multi_permutation_fold/predictions_pc4.npy must load: {e:?}"));

    // Categorical-only model: no float feature columns / borders. Train the
    // tensor-CTR model at the PRODUCTION-DEFAULT permutation_count=4, driving the
    // cat columns through `train_cat` (the cat-aware entry).
    let borders: Vec<Vec<f64>> = Vec::new();
    let (trained, baked_ctr_data) = train_cat(
        &CpuBackend,
        &[],
        &borders,
        &cat_cols,
        &target,
        &[],
        &tensor_ctr_params_pc4(),
        None,
    )
    .unwrap_or_else(|e| panic!("pc=4 tensor-CTR e2e training failed: {e:?}"));

    // Lift into the canonical model with the baked ctr_data and predict via the
    // PRODUCTION apply path (cb_model::predict_raw_cat over ModelSplit::Ctr).
    let model = CbModel::from_trained(&trained, borders.clone())
        .with_ctr_data(cb_model::CtrData::from_baked(&baked_ctr_data));
    let actual = cb_model::predict_raw_cat(&model, &[], &cat_cols);

    assert_eq!(
        actual.len(),
        expected_predictions.len(),
        "pc=4 prediction count must match upstream (N objects, all trees applied)"
    );
    // ≤1e-5 over ALL objects (covering ALL 5 trees), the SC-1 closure at pc=4.
    compare_stage(Stage::Predictions, &expected_predictions, &actual).unwrap_or_else(|e| {
        panic!("pc=4 tensor-CTR e2e predictions diverged from upstream catboost 1.2.10: {e:?}")
    });
}
