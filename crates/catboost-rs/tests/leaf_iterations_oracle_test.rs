//! `leaf_estimation_iterations` parity oracle against catboost 1.2.10, and the
//! backtracking search it makes reachable.
//!
//! Upstream's `CalcApproxDeltaSimple` runs the leaf solve N times per tree,
//! ACCUMULATING the per-leaf delta and RECOMPUTING the derivatives at
//! `approx + accumulated_delta` before each further step. `N = 1` is the
//! single-step solve this port already had.
//!
//! # Why the fixture uses Poisson + Newton
//!
//! Reached by elimination — see `gen_leaf_iterations_fixtures.py`. RMSE+Gradient
//! barely separates N=1 from N=5 (its leaf solve is already the exact optimum);
//! Logloss+Newton separates the iteration counts but never fires backtracking
//! (its Newton step always improves); Huber+Newton fires backtracking but
//! collapses `AnyImprovement` to the ALL-ZERO model, which an implementation
//! that simply refused every step would also produce. Poisson+Newton separates
//! the policies with both models non-trivial, so some steps are accepted and
//! some shrunk.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{
    CatBoostBuilder, IngestSource, LeafEstimationBacktracking, OwnedColumns, Pool,
};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("leaf_estimation_iterations")
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
    pool_of(cols, vec![1.0; n])
}

/// The pinned fit from `gen_leaf_iterations_fixtures.py::PARAMS`.
fn builder(iters: usize, policy: LeafEstimationBacktracking) -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Poisson)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(false)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Newton)
        .leaf_estimation_iterations(iters)
        .leaf_estimation_backtracking(policy)
}

fn fit_and_predict(iters: usize, policy: LeafEstimationBacktracking) -> Vec<f64> {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let model = builder(iters, policy)
        .fit(&pool)
        .expect("the fit must succeed");
    model.predict(&eval_pool()).expect("predict must succeed")
}

fn check(iters: usize, policy: LeafEstimationBacktracking, file: &str) {
    let actual = fit_and_predict(iters, policy);
    let expected = load_y(file);
    assert_eq!(actual.len(), expected.len(), "{policy:?}/{iters}: count");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{policy:?} iters={iters}: row {i} predicted {a} but catboost says {e} \
             (|diff| {})",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// The multi-step estimator itself, with backtracking DISABLED (pure accumulation)
// ---------------------------------------------------------------------------

#[test]
fn one_leaf_iteration_matches_catboost() {
    check(1, LeafEstimationBacktracking::No, "preds_No_1.npy");
}

#[test]
fn two_leaf_iterations_match_catboost() {
    check(2, LeafEstimationBacktracking::No, "preds_No_2.npy");
}

#[test]
fn five_leaf_iterations_match_catboost() {
    check(5, LeafEstimationBacktracking::No, "preds_No_5.npy");
}

/// More steps must actually CHANGE the model, otherwise every test above would
/// pass for an implementation that ignores the parameter.
#[test]
fn the_frozen_fixture_separates_one_iteration_from_five() {
    let one = load_y("preds_No_1.npy");
    let five = load_y("preds_No_5.npy");
    let sep = one
        .iter()
        .zip(five.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        sep > TOL,
        "1 and 5 leaf iterations differ by only {sep}; the fixture cannot detect a \
         single-step implementation"
    );
}

// ---------------------------------------------------------------------------
// Coverage gate: refused rather than silently single-stepped
// ---------------------------------------------------------------------------

/// Zero iterations would produce no leaf values at all.
#[test]
fn zero_leaf_iterations_is_rejected() {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let err = builder(0, LeafEstimationBacktracking::No)
        .fit(&pool)
        .expect_err("0 leaf iterations must be rejected");
    assert!(err.to_string().contains("leaf_estimation_iterations"));
}

/// `LeafMethod::Exact` computes a one-shot exact optimum, not a step to iterate,
/// so N > 1 is refused rather than silently ignored.
#[test]
fn multi_step_with_exact_leaf_is_rejected() {
    let cols = load_x("X.npy");
    let y = load_y("y.npy");
    let pool = pool_of(cols, y);
    let err = CatBoostBuilder::new()
        .loss(Loss::Mae)
        .iterations(3)
        .depth(3)
        .leaf_method(cb_compute::LeafMethod::Exact)
        .leaf_estimation_iterations(5)
        .fit(&pool)
        .expect_err("Exact + multi-step must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("Exact") && msg.contains("leaf_estimation_iterations"),
        "the refusal must name both the method and the parameter; got: {msg}"
    );
}

/// A non-symmetric grow policy computes leaf values on a different path.
#[test]
fn multi_step_with_a_non_symmetric_grow_policy_is_rejected() {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let err = builder(5, LeafEstimationBacktracking::No)
        .grow_policy(catboost_rs::EGrowPolicy::Depthwise)
        .fit(&pool)
        .expect_err("non-symmetric + multi-step must be rejected");
    assert!(err.to_string().contains("leaf_estimation_iterations"));
}

/// A single iteration is unaffected by every gate above.
#[test]
fn one_iteration_is_allowed_everywhere_the_gate_refuses_many() {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    builder(1, LeafEstimationBacktracking::No)
        .grow_policy(catboost_rs::EGrowPolicy::Depthwise)
        .fit(&pool)
        .expect("one leaf iteration must stay allowed under every grow policy");
}
