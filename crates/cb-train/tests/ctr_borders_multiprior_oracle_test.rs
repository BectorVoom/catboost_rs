//! E15 / SPEC-CTRT-11, acceptance A4 — multi-prior CTR candidate expansion,
//! end-to-end ≤1e-5 vs `catboost==1.2.10`.
//!
//! # What this locks
//!
//! 1. **≤1e-5 parity** for `simple_ctr_priors = [0.0, 0.5, 1.0]` through the
//!    production `train_cat` → `predict_raw_cat` path. Upstream emits ONE
//!    candidate column per `(ctrIdx, targetBorderIdx, priorIdx)`
//!    (`greedy_tensor_search.cpp:414-427`); the engine emitted one per
//!    projection, built from `priors.first()`, so every prior past the head was
//!    inert and the scored candidate set was strictly smaller.
//! 2. **The bake copy-back no longer flattens per-split priors.** The copy-back
//!    was keyed on `projection` alone and overwrote each split's own
//!    `prior_num` / `prior_denom` with the first baked table's, and copied that
//!    one table's `(shift, scale)` onto every split of the projection. With
//!    several priors live on one projection that is wrong for all but one split.
//!    Test fn 2 pins both halves.
//!
//! Test fn 2 lives here rather than in `boosting_test.rs` (where the plan
//! sketched it) because its observation channel is an INTEGRATION-level
//! observable — `CtrSplitSpec.{prior_num, scale}` read off the model
//! `train_cat` returns, all public — and it needs this directory's committed
//! fixture corpus, which is exactly the corpus proven (by E14's generator guard)
//! to put splits on projection {0} at three distinct priors. No private item is
//! touched, so the child-module placement buys nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_model::Model as CbModel;
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    train_cat, BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType, TProjection,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_borders_multiprior";

/// The three priors the fixture was generated with
/// (`simple_ctr = ["Borders:Prior=0:Prior=0.5:Prior=1"]`).
const PRIORS: [f64; 3] = [0.0, 0.5, 1.0];

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The two categorical columns as SoA `Vec<String>`, stringified via
/// `stringify_int_category` — the A4 plain-integer form upstream's Pool hashed
/// when the fixture was generated.
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The fixture's pinned config, EVERY field explicit (Pitfall-6 discipline: a
/// changed builder default must not silently alter what this gate exercises).
fn multiprior_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 20,
        depth: 3,
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
        simple_ctr: ECtrType::Borders,
        simple_ctr_priors: PRIORS.to_vec(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: PRIORS.to_vec(),
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

/// Train the fixture through production `train_cat`.
fn fit() -> (
    cb_train::Model,
    cb_train::BakedCtrData,
    Vec<Vec<String>>,
    Vec<Vec<f64>>,
) {
    let cat_cols = load_cat_columns();
    let model_json = load_model_json(&fixture(&format!("{SCENARIO}/model.json")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();

    let (trained, baked) = train_cat(
        &CpuBackend,
        &[], // categorical-only fixture: zero float columns
        &borders,
        &cat_cols,
        &target,
        &[],
        &multiprior_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("multi-prior Borders-CTR training failed: {e:?}"));

    (trained, baked, cat_cols, borders)
}

#[test]
fn borders_multiprior_predictions_match_upstream_within_1e_minus_5() {
    let expected = load_f64_vec(&fixture(&format!("{SCENARIO}/predictions.npy"))).unwrap();
    let (trained, baked, cat_cols, borders) = fit();

    let model = CbModel::from_trained(&trained, borders)
        .with_ctr_data(cb_model::CtrData::from_baked(&baked));
    let actual = cb_model::predict_raw_cat(&model, &[], &cat_cols);

    assert_eq!(
        actual.len(),
        expected.len(),
        "prediction count must match upstream"
    );
    assert!(
        actual.iter().any(|v| *v != actual[0]),
        "predictions are constant — the gate would be vacuous"
    );

    let max_div = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    compare_stage(Stage::Predictions, &expected, &actual).unwrap_or_else(|e| {
        panic!(
            "multi-prior Borders-CTR predictions diverged from upstream \
             (max |diff| = {max_div:e}): {e:?}"
        )
    });
    println!("borders_multiprior max |diff| = {max_div:e}");
}

#[test]
fn splits_on_one_projection_keep_distinct_priors_and_scales_after_the_bake() {
    let (trained, _baked, _cat_cols, _borders) = fit();

    // Every chosen CTR split on the SINGLE-member projection {0} — the one E14's
    // generator guard proved upstream splits on at three distinct priors.
    let proj0 = TProjection::from_features(&[0]);
    let splits: Vec<_> = trained
        .oblivious_trees
        .iter()
        .flat_map(|t| t.ctr_splits.iter())
        .filter(|s| s.projection == proj0)
        .collect();

    assert!(
        splits.len() >= 2,
        "the fixture must produce at least two CTR splits on projection {{0}} for \
         this pin to mean anything; got {}",
        splits.len()
    );

    let priors: Vec<f64> = splits.iter().map(|s| s.prior_num).collect();
    let distinct_priors: Vec<f64> = {
        let mut p = priors.clone();
        p.sort_by(f64::total_cmp);
        p.dedup();
        p
    };
    assert!(
        distinct_priors.len() >= 2,
        "the bake copy-back keyed on `projection` alone and OVERWROTE every split's \
         prior with the first baked table's — each split must keep its OWN prior. \
         Observed priors on projection {{0}}: {priors:?}"
    );

    // shift/scale must be derived PER SPLIT from calc_normalization(prior_num),
    // not copied off one shared table. `Borders:Prior=0/0.5` both normalize to
    // norm = 1 (scale 15); `Prior=1` does too — so `scale` alone cannot separate
    // them, and the honest pin is the (prior -> shift/scale) RELATION: two splits
    // that carry different priors must carry the normalization of THEIR prior.
    for s in &splits {
        let left = f64::min(0.0, s.prior_num);
        let right = f64::max(1.0, s.prior_num);
        let (want_shift, norm) = (-left, right - left);
        let want_scale = 15.0 / norm;
        assert_eq!(
            s.shift.to_bits(),
            want_shift.to_bits(),
            "split at prior {} must carry calc_normalization's OWN shift {want_shift}, \
             not a shared table's {}",
            s.prior_num,
            s.shift
        );
        assert_eq!(
            s.scale.to_bits(),
            want_scale.to_bits(),
            "split at prior {} must carry its OWN scale {want_scale}, not {}",
            s.prior_num,
            s.scale
        );
    }
}
