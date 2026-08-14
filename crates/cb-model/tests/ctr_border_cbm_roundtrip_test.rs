//! BUG-CTRB / SPEC-CTRB-05 — a trainer-produced CTR model must survive a `.cbm`
//! save → load round-trip with every border bit-identical.
//!
//! # Why this matters
//!
//! The value-space border is the FIRST non-integer CTR border this trainer has
//! ever written. `.cbm` narrows `Borders` to `f32` on save and widens via
//! `f64::from` on load, so a border that is not an f32 fixed point would shift
//! on the round-trip and silently move the split.
//!
//! # This is a GUARD, not a Red
//!
//! It passes both before and after the fix: an integer border is trivially an
//! f32 fixed point too. Its falsifiability comes from the mutation check
//! recorded in the task's completion evidence, not from an initial Red.
//!
//! No prior test exercises this shape: `save_cbm` had never been called on a
//! TRAINER-PRODUCED CTR model — the one existing CTR caller builds its `CtrData`
//! by hand rather than via `CtrData::from_baked`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_model::{load_cbm, save_cbm, CtrData, Model as CbModel, ModelSplit};
use cb_oracle::{load_f64_vec, load_model_json};
use cb_train::{train_cat, BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType};
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_counter_simple";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The fixture's pinned config, every field explicit (Pitfall-6).
fn counter_params() -> BoostParams {
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
        simple_ctr: ECtrType::Counter,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: vec![0.5],
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

fn fit() -> (CbModel, Vec<Vec<String>>) {
    let cat_cols = load_cat_columns();
    let model_json = load_model_json(&fixture(&format!("{SCENARIO}/model.json")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();

    let (trained, baked) = train_cat(
        &CpuBackend,
        &[],
        &borders,
        &cat_cols,
        &target,
        &[],
        &counter_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("training failed: {e:?}"));

    let model =
        CbModel::from_trained(&trained, borders).with_ctr_data(CtrData::from_baked(&baked));
    (model, cat_cols)
}

/// Every CTR split border in the model, in tree/split order.
fn ctr_borders(model: &CbModel) -> Vec<f64> {
    model
        .oblivious_trees
        .iter()
        .flat_map(|t| {
            t.splits.iter().filter_map(|s| match s {
                ModelSplit::Ctr(c) => Some(c.border),
                _ => None,
            })
        })
        .collect()
}

fn tmp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ctr_border_cbm_roundtrip_{tag}_{}.cbm",
        std::process::id()
    ))
}

#[test]
fn ctr_model_cbm_roundtrip_preserves_every_border_bitwise() {
    let (model, _cat_cols) = fit();

    let in_memory = ctr_borders(&model);
    assert!(
        !in_memory.is_empty(),
        "the trained model carries no CTR split — this gate would be vacuous"
    );

    let path = tmp_path("borders");
    // A border absent from its identity's `borders` set would surface here as
    // ModelError::Serialize("CTR split border missing from its identity") from
    // `ctr_split_to_global_index`, so a future regression in `build_ctr_features`
    // is attributed correctly rather than showing up as a mismatch below.
    save_cbm(&model, &path).unwrap_or_else(|e| {
        panic!(
            "save_cbm must succeed on a trainer-produced CTR model; a \
             ModelError::Serialize(\"CTR split border missing from its identity\") \
             here means the encode-side split->border lookup no longer resolves: {e:?}"
        )
    });
    let reloaded = load_cbm(&path).unwrap_or_else(|e| panic!("load_cbm must succeed: {e:?}"));
    let _ = std::fs::remove_file(&path);

    let after = ctr_borders(&reloaded);
    assert_eq!(
        after.len(),
        in_memory.len(),
        "the round-trip changed the CTR split count"
    );
    for (i, (a, b)) in in_memory.iter().zip(after.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "CTR border {i} shifted across the .cbm round-trip: {a} -> {b}. The \
             codec narrows Borders to f32 on save and widens via f64::from on \
             load, so the border must be an f32 fixed point (SPEC-CTRB-01 \
             Invariant 2)."
        );
    }
}

#[test]
fn ctr_model_cbm_roundtrip_predictions_are_bit_equal() {
    let (model, cat_cols) = fit();

    let before = cb_model::predict_raw_cat(&model, &[], &cat_cols);

    let path = tmp_path("preds");
    save_cbm(&model, &path).expect("save_cbm must succeed");
    let reloaded = load_cbm(&path).expect("load_cbm must succeed");
    let _ = std::fs::remove_file(&path);

    let after = cb_model::predict_raw_cat(&reloaded, &[], &cat_cols);

    assert_eq!(before.len(), after.len(), "prediction count changed");
    assert!(
        before.iter().any(|v| *v != before[0]),
        "predictions are constant — this gate would be vacuous"
    );
    for (i, (a, b)) in before.iter().zip(after.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "prediction {i} changed across the .cbm round-trip: {a} -> {b}"
        );
    }
}
