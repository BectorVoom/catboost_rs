//! PARAM-01 — the eval-set fit path through the PUBLISHED facade only.
//!
//! Four `BoostParams` fields (`od_type` / `od_pval` / `od_wait`,
//! `use_best_model`, `eval_metric`, and `counter_calc_method`) are
//! **eval-set-only**: with a learn set alone the trainer computes no validation
//! curve, so each is inert and setting it changes nothing. Before PARAM-01 the
//! facade had no way to supply an eval set at all, which is why all four were
//! pinned off inside `boost_params()`.
//!
//! These tests therefore assert BEHAVIOUR, not plumbing — the builder unit tests
//! already cover "the value reaches `BoostParams`". What can only be checked here
//! is that the value CHANGES THE RUN:
//!
//!   * the eval curve exists and has one entry per completed iteration;
//!   * `early_stopping_rounds` makes the run stop SHORT of `iterations`;
//!   * `use_best_model` TRUNCATES the returned model's tree list;
//!   * an empty eval-set list is byte-identical to a plain `fit` (the D-04
//!     no-regression gate for routing `fit` through the eval-set entry point).
//!
//! # The fixture: a deliberately anti-correlated eval set
//!
//! Early stopping is only observable when the eval metric gets WORSE as training
//! proceeds. A random split of one dataset would not guarantee that within a few
//! iterations. So the eval set here carries the SAME feature matrix as the learn
//! set but the NEGATED target: every tree that fits the learn signal moves the
//! eval predictions further from the eval labels, so the eval RMSE increases
//! monotonically from iteration 0 and the best iteration is unambiguously the
//! first. That makes both the stop point and the truncated tree count exact
//! rather than statistical.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use catboost_rs::{
    CatBoostBuilder, EOverfittingDetectorType, EvalMetric, IngestSource, OwnedColumns, Pool,
};

/// Objects in the synthetic dataset.
const N: usize = 64;

/// A learn pool whose single float feature perfectly predicts the target.
fn learn_pool() -> Pool {
    let x: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    OwnedColumns::new(vec![x], y)
        .into_pool()
        .expect("learn pool must build")
}

/// An eval pool with the SAME features and the NEGATED target — so every tree
/// fitted on the learn signal makes this set's error strictly worse.
fn anticorrelated_eval_pool() -> Pool {
    let x: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| -(2.0 * v + 1.0)).collect();
    OwnedColumns::new(vec![x], y)
        .into_pool()
        .expect("eval pool must build")
}

/// A builder configured for a short, fully deterministic regression run.
fn builder(iterations: usize) -> CatBoostBuilder {
    CatBoostBuilder::new()
        .iterations(iterations)
        .depth(2)
        .learning_rate(0.3)
}

/// The eval curve exists and carries exactly one value per completed iteration
/// when no detector is active.
#[test]
fn fit_with_eval_returns_one_metric_value_per_iteration() {
    let out = builder(10)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");

    assert_eq!(
        out.eval_history.len(),
        1,
        "one curve per eval set was supplied"
    );
    assert_eq!(
        out.eval_history[0].len(),
        10,
        "no detector is active, so every iteration must be recorded"
    );
    assert_eq!(
        out.model.as_canonical().oblivious_trees.len(),
        10,
        "without use_best_model the full ensemble is returned"
    );
}

/// The anti-correlated fixture behaves as the other tests assume: the eval metric
/// is monotonically WORSE with each iteration, so iteration 0 is the best.
///
/// Asserted explicitly rather than left implicit — if this ever stopped holding,
/// the early-stopping and truncation tests below would still pass for the wrong
/// reason (a run that stops because the metric plateaued, not because it worsened).
#[test]
fn the_eval_curve_degrades_monotonically_on_the_anticorrelated_set() {
    let out = builder(8)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");
    let curve = &out.eval_history[0];
    for w in curve.windows(2) {
        assert!(
            w[1] > w[0],
            "eval error must strictly increase on the anti-correlated set, got {curve:?}"
        );
    }
}

/// `early_stopping_rounds` STOPS the run short of `iterations`.
///
/// With `od_wait = 2` and a metric that worsens from the very first iteration,
/// the detector fires two iterations after the best (iteration 0), so the run
/// must record far fewer than the 50 iterations requested.
#[test]
fn early_stopping_rounds_stops_the_run_short() {
    let requested = 50;
    let out = builder(requested)
        .early_stopping_rounds(2)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");

    let ran = out.eval_history[0].len();
    assert!(
        ran < requested,
        "early stopping must cut the run short, but all {requested} iterations ran"
    );
    assert_eq!(
        out.model.as_canonical().oblivious_trees.len(),
        ran,
        "the model must carry exactly the trees the run actually grew"
    );
}

/// The SAME configuration WITHOUT the detector runs to completion — the control
/// that proves the previous test's short run came from early stopping and not
/// from some unrelated property of the fixture.
#[test]
fn without_a_detector_the_same_run_uses_every_iteration() {
    let out = builder(50)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");
    assert_eq!(out.eval_history[0].len(), 50);
}

/// `use_best_model` TRUNCATES the returned model to `best_iteration + 1` trees.
/// On the anti-correlated fixture the best iteration is 0, so exactly one tree
/// survives — while the same run without the flag keeps all of them.
#[test]
fn use_best_model_truncates_the_model_to_the_best_iteration() {
    let truncated = builder(10)
        .use_best_model(true)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");
    let full = builder(10)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");

    assert_eq!(
        truncated.model.as_canonical().oblivious_trees.len(),
        1,
        "the best iteration is 0, so use_best_model must keep exactly one tree"
    );
    assert_eq!(
        full.model.as_canonical().oblivious_trees.len(),
        10,
        "the control run must keep the full ensemble"
    );
}

/// An explicit `eval_metric` overrides the objective-derived default: MAE and
/// RMSE disagree numerically on this fixture, so the recorded curve differs.
///
/// This is what proves the setter is CONSUMED rather than merely stored — the
/// builder unit test can only see that it reached `BoostParams`.
#[test]
fn eval_metric_override_changes_the_recorded_curve() {
    let default_metric = builder(5)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");
    let mae = builder(5)
        .eval_metric(EvalMetric::Mae)
        .fit_with_eval(&learn_pool(), &anticorrelated_eval_pool())
        .expect("eval fit must succeed");

    assert_ne!(
        default_metric.eval_history[0], mae.eval_history[0],
        "an explicit MAE eval_metric must produce a different curve than the \
         RMSE objective default"
    );
    assert_eq!(
        default_metric.model.as_canonical().oblivious_trees.len(),
        mae.model.as_canonical().oblivious_trees.len(),
        "the metric is a MEASUREMENT: with no detector active it must not change \
         the model itself"
    );
}

/// D-04 NO-REGRESSION GATE. `fit_with_eval_sets(pool, &[])` must be identical to
/// `fit(pool)` — the routing change (`fit` now delegates to the eval-set entry
/// point with an empty set list) may not perturb a learn-only run.
#[test]
fn an_empty_eval_set_list_is_identical_to_a_plain_fit() {
    let pool = learn_pool();
    let plain = builder(12).fit(&pool).expect("plain fit must succeed");
    let via_eval = builder(12)
        .fit_with_eval_sets(&pool, &[])
        .expect("empty-eval fit must succeed");

    assert!(
        via_eval.eval_history.is_empty(),
        "no eval set was supplied, so no curve may be recorded"
    );
    assert_eq!(
        plain.predict(&pool).expect("predict"),
        via_eval.model.predict(&pool).expect("predict"),
        "an empty eval-set list must reproduce the plain fit EXACTLY"
    );
}

/// A detector configured WITHOUT an eval set stays inert rather than erroring:
/// there is no curve to detect on, so the run completes normally. (Upstream
/// behaves the same way — `od_type` without a validation set is a no-op.)
#[test]
fn a_detector_without_an_eval_set_is_inert() {
    let pool = learn_pool();
    let out = builder(10)
        .od_type(EOverfittingDetectorType::Iter)
        .od_wait(1)
        .fit(&pool)
        .expect("a detector with no eval set must not fail the fit");
    assert_eq!(
        out.as_canonical().oblivious_trees.len(),
        10,
        "with no eval set the detector cannot fire, so every iteration runs"
    );
}

/// An eval pool whose float width disagrees with the learn pool is rejected with
/// a typed `FeatureMismatch` — never scored positionally against the wrong
/// columns.
#[test]
fn a_width_mismatched_eval_pool_is_rejected() {
    let learn = learn_pool();
    let x: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let wide = OwnedColumns::new(vec![x.clone(), x], vec![0.0; N])
        .into_pool()
        .expect("two-column pool must build");

    let err = builder(3)
        .fit_with_eval(&learn, &wide)
        .expect_err("a width mismatch must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("eval set 0") && msg.contains("float features"),
        "the error must name the offending eval set and the mismatched width, got: {msg}"
    );
}

/// Several eval sets are all evaluated, and the PRIMARY (index 0) is the one the
/// detector consumes. Asserting the curve count is what catches a facade that
/// silently keeps only the first set.
#[test]
fn every_supplied_eval_set_gets_its_own_curve() {
    let learn = learn_pool();
    let a = anticorrelated_eval_pool();
    let b = learn_pool();

    let out = builder(6)
        .fit_with_eval_sets(&learn, &[&a, &b])
        .expect("multi-eval fit must succeed");

    assert_eq!(out.eval_history.len(), 2, "one curve per supplied eval set");
    assert_eq!(out.eval_history[0].len(), 6);
    assert_eq!(out.eval_history[1].len(), 6);
    assert_ne!(
        out.eval_history[0], out.eval_history[1],
        "the two sets carry different targets, so their curves must differ"
    );
}
