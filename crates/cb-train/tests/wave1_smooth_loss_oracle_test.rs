//! Wave-1 smooth-regression-loss training oracle (LOSS-03 / Plan 06.1-01): train a
//! plain boosted regression model under each of the four smooth losses with a real
//! der2 — LogCosh, Lq{q}, Huber{delta}, Expectile{alpha} — and gate per-tree
//! splits, per-tree leaf values, and per-iteration staged approximants against the
//! committed upstream catboost 1.2.10 `{logcosh,lq,huber,expectile}` fixtures at
//! <= 1e-5.
//!
//! Per-loss leaf method is PINNED per the upstream default (RESEARCH Pitfall 2):
//!   - LogCosh  : Exact   (catboost_options.cpp:65-70 — NOT Newton)
//!   - Lq(q=2.0): Newton  (q>=2 so der2 is Newton-clean, Pitfall 6)
//!   - Huber    : Newton  (catboost_options.cpp:187-192)
//!   - Expectile: Newton, leaf_estimation_iterations:1 PINNED (override upstream 5)
//!
//! All four share the D-07 isolating discipline (depth 2, 5 iterations,
//! learning_rate 0.1, l2_leaf_reg 3.0, no sampling / no random strength,
//! boost_from_average=false, score_function=L2) so a divergence is attributable to
//! the loss math alone.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the other cb-train oracle tests.
//!
//! RED until Tasks 2-3 land the four losses + their dispatch/leaf wiring — the
//! `Loss::{LogCosh,Lq,Huber,Expectile}` variants do not exist yet, so this file
//! will not even compile against the current `cb_compute::Loss`. That RED state is
//! the Task-1 acceptance signal.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
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
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

fn load_y() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

fn base_params(loss: Loss, leaf_method: LeafMethod) -> BoostParams {
    BoostParams {
        loss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method,
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
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
    }
}

fn train_scenario(loss: Loss, leaf_method: LeafMethod, scenario: &str) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let target = load_y();
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &base_params(loss, leaf_method),
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("{scenario}: training failed: {e:?}"));
    (model, staged)
}

fn check_scenario(loss: Loss, leaf_method: LeafMethod, scenario: &str) {
    let (model, staged) = train_scenario(loss, leaf_method, scenario);
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json"))).unwrap();

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("{scenario}: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("{scenario}: leaf values diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("{scenario}: staged approx diverged: {e:?}"));
}

#[test]
fn wave1_oracle_logcosh() {
    // LogCosh is non-parametric; upstream default leaf method is Exact (Pitfall 2).
    check_scenario(Loss::LogCosh, LeafMethod::Exact, "logcosh");
}

#[test]
fn wave1_oracle_lq() {
    // Lq{q=2.0}: q>=2 so der2 is Newton-clean (Pitfall 6); leaf method Newton.
    check_scenario(Loss::Lq { q: 2.0 }, LeafMethod::Newton, "lq");
}

#[test]
fn wave1_oracle_huber() {
    // Huber{delta=1.0}: smooth band der2=-1, leaf method Newton.
    check_scenario(Loss::Huber { delta: 1.0 }, LeafMethod::Newton, "huber");
}

#[test]
fn wave1_oracle_expectile() {
    // Expectile{alpha=0.3}: leaf method Newton, leaf_estimation_iterations:1 pinned.
    check_scenario(Loss::Expectile { alpha: 0.3 }, LeafMethod::Newton, "expectile");
}
