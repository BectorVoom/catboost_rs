//! `leaf_estimation_backtracking` parity oracle against catboost 1.2.10.
//!
//! # What this parameter can and cannot do here
//!
//! Backtracking HALVES a leaf-value step that would not improve the loss, so it
//! is a property of the leaf-estimation ITERATION loop — with a single step
//! there is no earlier step to fall back to.
//!
//! `leaf_estimation_iterations` is not implemented in this engine (the leaf
//! estimator takes exactly one step). Measured against catboost 1.2.10 over
//! {RMSE, Logloss, MAE, Poisson, Tweedie, Huber, LogCosh, Quantile} x
//! {Gradient, Newton} x learning_rate {0.3, 1, 3, 10}:
//!
//! - at one leaf iteration, **0 of 64** configurations distinguish `No` from
//!   `AnyImprovement`;
//! - at more than one, **53** do — the frozen counter-example
//!   (`Huber:delta=1.0`, Newton, 5 iterations) separates them by 7.12.
//!
//! So the two CPU policies are provably equivalent in the supported regime, and
//! that is pinned here as an oracle-verified fact rather than assumed. The
//! backtracking SEARCH is deliberately not written: it would be unreachable code
//! no test could exercise. When `leaf_estimation_iterations` lands, the
//! equivalence assertion below is exactly what should start failing.
//!
//! `Armijo` is GPU-ONLY upstream (`catboost_options.cpp:664`) and its rejection
//! IS observable behaviour, so it is gated here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, IngestSource, LeafEstimationBacktracking, OwnedColumns, Pool};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("leaf_estimation_backtracking")
        .join(rel)
}

fn load_x(name: &str) -> Vec<Vec<f64>> {
    let x: Array2<f64> = read_npy(fixture(name)).expect("fixture matrix");
    (0..x.ncols()).map(|f| x.column(f).to_vec()).collect()
}

fn load_y(name: &str) -> Vec<f64> {
    let y: Array1<f64> = read_npy(fixture(name)).expect("fixture vector");
    y.to_vec()
}

fn pool_of(cols: Vec<Vec<f64>>, target: Vec<f64>) -> Pool {
    OwnedColumns::new(cols, target)
        .into_pool()
        .expect("pool must build")
}

fn eval_pool() -> Pool {
    let cols = load_x("X_eval.npy");
    let n = cols.first().map_or(0, Vec::len);
    pool_of(cols, vec![0.0; n])
}

/// The pinned fit from `gen_leaf_backtracking_fixtures.py::PARAMS`.
fn builder(policy: LeafEstimationBacktracking) -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(true)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
        .leaf_estimation_backtracking(policy)
}

fn fit_and_predict(policy: LeafEstimationBacktracking) -> Vec<f64> {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let model = builder(policy).fit(&pool).expect("the fit must succeed");
    model.predict(&eval_pool()).expect("predict must succeed")
}

fn check(policy: LeafEstimationBacktracking, preds_file: &str) {
    let actual = fit_and_predict(policy);
    let expected = load_y(preds_file);
    assert_eq!(actual.len(), expected.len(), "{policy:?}: prediction count");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{policy:?}: row {i} predicted {a} but catboost 1.2.10 says {e}"
        );
    }
}

#[test]
fn backtracking_no_matches_catboost() {
    check(LeafEstimationBacktracking::No, "preds_No.npy");
}

#[test]
fn backtracking_any_improvement_matches_catboost() {
    check(
        LeafEstimationBacktracking::AnyImprovement,
        "preds_AnyImprovement.npy",
    );
}

/// `Armijo` is GPU-only upstream, so the CPU fit must REFUSE it rather than
/// quietly substituting `AnyImprovement` — which would hand back a model the
/// caller did not ask for.
#[test]
fn backtracking_armijo_is_rejected_on_cpu() {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let err = builder(LeafEstimationBacktracking::Armijo)
        .fit(&pool)
        .expect_err("Armijo must be rejected on the CPU path");
    let msg = err.to_string();
    assert!(
        msg.contains("Armijo") && msg.contains("GPU"),
        "the rejection must name the policy and say it is GPU-only; got: {msg}"
    );
}

/// The two CPU policies agree in this engine's supported regime, matching the
/// frozen catboost measurement. This is the assertion that must START FAILING
/// once `leaf_estimation_iterations > 1` is implemented — at which point the
/// backtracking search has to be written for real.
#[test]
fn no_and_any_improvement_agree_at_one_leaf_iteration() {
    let oracle_no = load_y("preds_No.npy");
    let oracle_any = load_y("preds_AnyImprovement.npy");
    assert_eq!(
        oracle_no, oracle_any,
        "the frozen catboost predictions for No and AnyImprovement must be identical \
         at leaf_estimation_iterations=1"
    );

    let ours_no = fit_and_predict(LeafEstimationBacktracking::No);
    let ours_any = fit_and_predict(LeafEstimationBacktracking::AnyImprovement);
    assert_eq!(
        ours_no, ours_any,
        "this engine must reproduce that equivalence"
    );
}

/// Every legal token round-trips; an unknown one is rejected.
#[test]
fn backtracking_parses_every_legal_token() {
    for p in LeafEstimationBacktracking::all() {
        assert_eq!(LeafEstimationBacktracking::parse(p.as_str()), Some(p));
    }
    assert_eq!(LeafEstimationBacktracking::parse("anyimprovement"), None);
    assert_eq!(LeafEstimationBacktracking::parse("ZzBogusValue"), None);
    // Only Armijo is GPU-gated.
    assert!(LeafEstimationBacktracking::No.is_cpu_supported());
    assert!(LeafEstimationBacktracking::AnyImprovement.is_cpu_supported());
    assert!(!LeafEstimationBacktracking::Armijo.is_cpu_supported());
}
