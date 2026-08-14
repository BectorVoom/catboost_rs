//! E23 / SPEC-CTRT-17 (parity half), acceptance A6 — `counter_calc_method`
//! discriminated end-to-end with an eval set, ≤1e-5 vs `catboost==1.2.10`.
//!
//! `counter_calc_method` is UNOBSERVABLE without an eval set (measured
//! `maxdiff = 0.000e+00` learn-only vs `4.010e-01` with a 40-row eval set,
//! research §B) — a learn-only test passes trivially and is FORBIDDEN. This
//! gate therefore trains through `train_cat_with_eval_sets` with the fixture's
//! eval cat columns supplied, under BOTH settings, against the two separately
//! frozen upstream prediction vectors, and asserts our own two runs genuinely
//! differ (test fn 3 — without it, two identical wrong answers pass both
//! parity gates).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_model::Model as CbModel;
use cb_oracle::load_f64_vec;
use cb_train::{
    train_cat_with_eval_sets, BoostParams, CounterCalcMethod, EBootstrapType, ECtrType,
    EOverfittingDetectorType, EvalSet,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_counter_full_eval";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_cat(file: &str) -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/{file}")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/{file} must load as int32: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

fn counter_params(method: CounterCalcMethod) -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 10,
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
        counter_calc_method: method,
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

fn fit(method: CounterCalcMethod) -> (Vec<f64>, cb_train::BakedCtrData) {
    let learn_cats = load_cat("X_cat.npy");
    let eval_cats = load_cat("X_cat_eval.npy");
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();
    let eval_target = load_f64_vec(&fixture(&format!("{SCENARIO}/y_eval.npy"))).unwrap();
    let weights = vec![1.0_f64; target.len()];

    let eval_sets = vec![EvalSet {
        feature_values: &[],
        target: &eval_target,
        cat_columns: &eval_cats,
    }];

    let (trained, baked) = train_cat_with_eval_sets(
        &CpuBackend,
        &[],
        &[],
        &learn_cats,
        &target,
        &weights,
        &counter_params(method),
        None,
        &eval_sets,
        None,
    )
    .unwrap_or_else(|e| panic!("Counter training with an eval set failed ({method:?}): {e:?}"));

    let model = CbModel::from_trained(&trained, Vec::new())
        .with_ctr_data(cb_model::CtrData::from_baked(&baked));
    let preds = cb_model::predict_raw_cat(&model, &[], &learn_cats);
    (preds, baked)
}

fn max_div(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn counter_full_with_eval_set_matches_upstream_within_1e_minus_5() {
    let expected = load_f64_vec(&fixture(&format!("{SCENARIO}/predictions_full.npy"))).unwrap();
    let (ours, _baked) = fit(CounterCalcMethod::Full);
    assert_eq!(ours.len(), expected.len());
    let d = max_div(&ours, &expected);
    assert!(d <= 1e-5, "Full diverged from upstream: max |diff| = {d:e}");
    println!("counter Full max |diff| = {d:e}");
}

#[test]
fn counter_skiptest_with_eval_set_matches_upstream_within_1e_minus_5() {
    let expected =
        load_f64_vec(&fixture(&format!("{SCENARIO}/predictions_skiptest.npy"))).unwrap();
    let (ours, _baked) = fit(CounterCalcMethod::SkipTest);
    assert_eq!(ours.len(), expected.len());
    let d = max_div(&ours, &expected);
    assert!(d <= 1e-5, "SkipTest diverged from upstream: max |diff| = {d:e}");
    println!("counter SkipTest max |diff| = {d:e}");
}

#[test]
fn full_and_skiptest_predictions_actually_differ() {
    // THE DISCRIMINATOR (mirroring the generator's guard): without it, two
    // identical wrong answers pass both parity gates. Research measured
    // 0.000e+00 learn-only vs 4.010e-01 with an eval set; this fixture froze
    // 2.240e-01.
    let (full, _) = fit(CounterCalcMethod::Full);
    let (skiptest, _) = fit(CounterCalcMethod::SkipTest);
    let d = max_div(&full, &skiptest);
    assert!(
        d > 1e-3,
        "our Full and SkipTest runs agree (maxdiff = {d:e}) — the threading is \
         inert and both parity gates are vacuous"
    );
}

#[test]
fn baked_counter_denominator_is_larger_under_full() {
    // The structural twin of the measured probe (22 vs 14): the eval documents
    // join the Counter tally, so the MAX bucket total can only grow.
    let (_, baked_full) = fit(CounterCalcMethod::Full);
    let (_, baked_skiptest) = fit(CounterCalcMethod::SkipTest);
    let denom = |b: &cb_train::BakedCtrData| {
        b.tables
            .iter()
            .find(|t| t.ctr_type == ECtrType::Counter.as_i8())
            .map(|t| t.counter_denominator)
            .expect("a baked Counter table must exist")
    };
    let df = denom(&baked_full);
    let ds = denom(&baked_skiptest);
    assert!(
        df > ds,
        "the Full denominator must see the eval documents (Full {df} vs SkipTest {ds})"
    );
}
