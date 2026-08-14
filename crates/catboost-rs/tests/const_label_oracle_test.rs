//! `allow_const_label` parity oracle against catboost 1.2.10.
//!
//! A learn set whose targets are ALL EQUAL is REFUSED by default
//! (`metric.cpp:7011`, "All train targets are equal") — most metrics are
//! undefined on a constant target, so a silently-trained degenerate model is
//! worse than a named error. With the flag set, upstream trains 5 trees that
//! predict the constant, and so does this port.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, IngestSource, OwnedColumns, Pool};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("logging_const_label")
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

/// The pinned fit from `gen_logging_const_fixtures.py::PARAMS`.
fn builder() -> CatBoostBuilder {
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
}

/// A constant learn target is refused by default, naming the opt-in.
#[test]
fn a_constant_target_is_refused_by_default() {
    let pool = pool_of(load_x("X.npy"), load_y("y_const.npy"));
    let err = builder()
        .fit(&pool)
        .expect_err("an all-equal target must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("equal") && msg.contains("allow_const_label"),
        "the refusal must state the cause and name the opt-in; got: {msg}"
    );
}

/// With the opt-in, training succeeds and the model predicts the constant —
/// matching catboost, which produces 3.5 everywhere.
#[test]
fn allow_const_label_trains_and_predicts_the_constant() {
    let pool = pool_of(load_x("X.npy"), load_y("y_const.npy"));
    let model = builder()
        .allow_const_label(true)
        .fit(&pool)
        .expect("allow_const_label must permit the fit");
    let actual = model.predict(&eval_pool()).expect("predict must succeed");
    let expected = load_y("preds_const_allowed.npy");
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "row {i} predicted {a} but catboost 1.2.10 says {e}"
        );
    }
}

/// The flag must be INERT on a normal (non-constant) target — it only removes a
/// guard, it must not change a fit that was already legal.
#[test]
fn allow_const_label_is_inert_on_a_normal_target() {
    let baseline = {
        let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
        let model = builder().fit(&pool).expect("fit");
        model.predict(&eval_pool()).expect("predict")
    };
    let expected = load_y("preds.npy");
    for (i, (a, e)) in baseline.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "row {i}: the default fit diverged from catboost ({a} vs {e})"
        );
    }

    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let with_flag = builder()
        .allow_const_label(true)
        .fit(&pool)
        .expect("fit")
        .predict(&eval_pool())
        .expect("predict");
    assert_eq!(
        baseline, with_flag,
        "allow_const_label must not change a fit that was already legal"
    );
}

/// A single-row learn set must not trip the CONSTANT-TARGET guard: with one
/// target there is nothing to compare, and refusing it as "all targets equal"
/// would be a restriction upstream does not have.
///
/// Such a fit still fails, for an unrelated and pre-existing reason — a one-row
/// column yields no borders, so there is no candidate split — which is exactly
/// what this asserts: the error must NOT be the const-label one.
#[test]
fn a_single_row_target_is_not_treated_as_constant() {
    let pool = pool_of(vec![vec![1.0], vec![2.0], vec![3.0]], vec![7.0]);
    let err = builder()
        .iterations(1)
        .depth(1)
        .fit(&pool)
        .expect_err("a one-row fit fails for lack of any candidate split");
    let msg = err.to_string();
    assert!(
        !msg.contains("allow_const_label"),
        "a one-row target must not be reported as a constant target; got: {msg}"
    );
    assert!(
        msg.contains("candidate split"),
        "the failure should be the pre-existing no-border one; got: {msg}"
    );
}
