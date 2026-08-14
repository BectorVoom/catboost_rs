//! BUG-CTRB / SPEC-CTRB-01 — the persisted CTR border must match upstream
//! catboost 1.2.10's own committed value-space convention.
//!
//! C01 proves INTERNAL consistency (training and apply agree). This proves
//! UPSTREAM agreement. Both are needed: a border of `b + 0.5` would satisfy C01
//! and still be wrong on the wire.
//!
//! Upstream stores CTR split borders as `(bin + 1) - 2^-20`, in VALUE space.
//! The borders here are read from the COMMITTED fixtures, never regenerated.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_oracle::{load_f64_vec, load_model_json};
use cb_train::{
    train_cat, BoostParams, EBootstrapType, EOverfittingDetectorType,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Every CTR border upstream committed for `scenario`, sorted.
///
/// Read with raw `serde_json` because `cb_oracle`'s model-json accessor exposes
/// float-feature borders only, not CTR borders.
fn upstream_ctr_borders(scenario: &str) -> Vec<f64> {
    let raw = std::fs::read_to_string(fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must be readable: {e:?}"));
    let json: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{scenario}/model.json: {e:?}"));

    let ctrs = json["features_info"]["ctrs"]
        .as_array()
        .unwrap_or_else(|| panic!("{scenario}: features_info.ctrs must be an array"));
    assert!(
        !ctrs.is_empty(),
        "{scenario}: features_info.ctrs is empty — the fixture carries no CTR at all"
    );

    let mut borders: Vec<f64> = ctrs
        .iter()
        .flat_map(|c| {
            c["borders"]
                .as_array()
                .unwrap_or_else(|| panic!("{scenario}: a ctr entry has no borders array"))
                .iter()
                .map(|b| b.as_f64().expect("border must be a number"))
        })
        .collect();
    borders.sort_by(f64::total_cmp);
    borders
}

fn load_cat_columns(scenario: &str) -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{scenario}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/X_cat.npy must load as int32: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The shared skeleton; per-scenario deltas are applied by the caller.
fn base_params() -> BoostParams {
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
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 2,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_train::score_function_default(),
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

fn params_for(scenario: &str) -> BoostParams {
    let mut p = base_params();
    if scenario == "ctr_counter_simple" {
        p.simple_ctr = cb_train::ECtrType::Counter;
        p.simple_ctr_priors = vec![0.5];
        p.combinations_ctr_priors = vec![0.5];
        p.max_ctr_complexity = 1;
    }
    p
}

/// Every CTR border this repository's trainer persists for `scenario`.
fn trained_ctr_borders(scenario: &str) -> Vec<f64> {
    let cat_cols = load_cat_columns(scenario);
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture(&format!("{scenario}/y.npy"))).unwrap();

    let (trained, _baked) = train_cat(
        &CpuBackend,
        &[],
        &borders,
        &cat_cols,
        &target,
        &[],
        &params_for(scenario),
        None,
    )
    .unwrap_or_else(|e| panic!("{scenario}: training failed: {e:?}"));

    let mut out: Vec<f64> = trained
        .oblivious_trees
        .iter()
        .flat_map(|t| t.ctr_splits.iter().map(|s| s.border))
        .collect();
    out.sort_by(f64::total_cmp);
    out.dedup_by(|a, b| a.to_bits() == b.to_bits());
    out
}

const SCENARIOS: [&str; 2] = ["tensor_ctr_e2e", "ctr_counter_simple"];

#[test]
fn trained_ctr_borders_follow_the_upstream_value_space_convention() {
    for scenario in SCENARIOS {
        let borders = trained_ctr_borders(scenario);
        assert!(
            !borders.is_empty(),
            "{scenario}: the trained model carries no CTR split at all — this gate \
             would be vacuous"
        );

        for x in &borders {
            // Upstream's form is (b+1) - 2^-20, so x + 2^-20 must be an integer.
            let k = x + f64::from(f32::powi(2.0, -20));
            assert_eq!(
                k,
                k.round(),
                "{scenario}: border {x} is not of upstream's form (b+1) - 2^-20 \
                 (x + 2^-20 = {k}, which is not integral). All trained borders: \
                 {borders:?}"
            );

            // The .cbm codec narrows Borders to f32 and decodes via f64::from, so a
            // border that is not an f32 fixed point would shift on save/load.
            assert_eq!(
                f64::from(*x as f32),
                *x,
                "{scenario}: border {x} is not an f32 fixed point; the .cbm codec \
                 narrows Borders to f32 and a non-fixed-point border shifts on \
                 save/load"
            );

            assert!(*x > 0.0, "{scenario}: border {x} must be strictly positive");
        }
    }
}

#[test]
fn trained_ctr_borders_are_members_of_the_upstream_border_set() {
    for scenario in SCENARIOS {
        let upstream = upstream_ctr_borders(scenario);
        let borders = trained_ctr_borders(scenario);
        assert!(!borders.is_empty(), "{scenario}: no trained CTR border");

        for x in &borders {
            assert!(
                upstream.iter().any(|u| u.to_bits() == x.to_bits()),
                "trained CTR border {x} is not among upstream's {upstream:?} for \
                 {scenario}. If the CONVENTION assertions pass and this one fails, \
                 the trainer chose a DIFFERENT bin threshold than catboost 1.2.10 — \
                 that is a structural parity finding, NOT a test to weaken. \
                 STOP AND REPORT."
            );
        }
    }
}
