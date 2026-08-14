//! `sampling_frequency` / `sampling_unit` parity oracle against catboost 1.2.10.
//!
//! This engine implements exactly one value of each: `sampling_frequency=PerTree`
//! (it draws the bootstrap sample once per tree, before the level loop) and
//! `sampling_unit=Object` (the sampler draws per object and has no group spans).
//!
//! The claim worth testing is therefore NOT "the parameter changes something" — it is
//! **"the value we implement is the one upstream computes"**. A default nobody
//! exercises against the oracle is exactly how an engine drifts while every test
//! passes, so each drawing sampler gets a frozen `PerTree` target here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{
    CatBoostBuilder, EBootstrapType, ESamplingFrequency, ESamplingUnit, IngestSource, OwnedColumns,
    Pool,
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
        .join("sampling")
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

fn learn_pool() -> Pool {
    pool_of(load_x("X.npy"), load_y("y.npy"))
}

fn eval_pool() -> Pool {
    let cols = load_x("X_eval.npy");
    let n = cols.first().map_or(0, Vec::len);
    pool_of(cols, vec![0.0; n])
}

/// The pinned fit from `gen_sampling_fixtures.py::BASE`.
fn builder() -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(false)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
}

/// Apply the per-sampler knobs upstream accepts. They cannot be pinned globally:
/// `No`/`Bayesian` REJECT `subsample`, so each type carries exactly its own.
fn with_sampler(b: CatBoostBuilder, bt: EBootstrapType) -> CatBoostBuilder {
    match bt {
        EBootstrapType::Bernoulli | EBootstrapType::Mvs => b.bootstrap_type(bt).subsample(0.7),
        EBootstrapType::Bayesian => b.bootstrap_type(bt).bagging_temperature(1.0),
        _ => b.bootstrap_type(bt),
    }
}

fn check(bt: EBootstrapType, fixture_name: &str) {
    let model = with_sampler(builder(), bt)
        .sampling_frequency(ESamplingFrequency::PerTree)
        .sampling_unit(ESamplingUnit::Object)
        .fit(&learn_pool())
        .unwrap_or_else(|e| panic!("{bt:?}: fit must succeed: {e}"));
    let actual = model.predict(&eval_pool()).expect("predict must succeed");
    let expected = load_y(fixture_name);
    assert_eq!(actual.len(), expected.len(), "{bt:?}: prediction count");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{bt:?} row {i}: predicted {a} but catboost 1.2.10 (sampling_frequency=PerTree) \
             says {e} (|diff| = {})",
            (a - e).abs()
        );
    }
}

#[test]
fn per_tree_matches_catboost_under_bernoulli() {
    check(EBootstrapType::Bernoulli, "preds_pertree_bernoulli.npy");
}

#[test]
fn per_tree_matches_catboost_under_bayesian() {
    check(EBootstrapType::Bayesian, "preds_pertree_bayesian.npy");
}

#[test]
fn per_tree_matches_catboost_under_mvs() {
    check(EBootstrapType::Mvs, "preds_pertree_mvs.npy");
}

/// `PerTreeLevel` is ACCEPTED under `bootstrap_type=No` and produces the baseline
/// model — the no-draw carve-out, end to end through the public builder.
///
/// The fixture proves upstream agrees: `inert_without_draw_max_abs_diff` is exactly
/// 0 at depths 1, 2 and 4.
#[test]
fn per_tree_level_is_accepted_and_inert_without_a_draw() {
    let predict = |freq: ESamplingFrequency| {
        builder()
            .bootstrap_type(EBootstrapType::No)
            .sampling_frequency(freq)
            .fit(&learn_pool())
            .expect("bootstrap_type=No must accept either sampling_frequency")
            .predict(&eval_pool())
            .expect("predict must succeed")
    };
    let per_tree = predict(ESamplingFrequency::PerTree);
    let per_level = predict(ESamplingFrequency::PerTreeLevel);
    assert_eq!(
        per_tree, per_level,
        "with no draw the two frequencies must produce identical predictions"
    );
    assert!(
        per_tree.iter().any(|v| *v != 0.0),
        "vacuous: the fit produced all-zero predictions, so the comparison proves nothing"
    );
}

/// `PerTreeLevel` with a DRAWING sampler is refused through the public surface, and
/// the error names the alternative rather than just failing.
#[test]
fn per_tree_level_is_refused_with_a_drawing_sampler() {
    let err = with_sampler(builder(), EBootstrapType::Bernoulli)
        .sampling_frequency(ESamplingFrequency::PerTreeLevel)
        .fit(&learn_pool())
        .expect_err("PerTreeLevel + a drawing sampler must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("sampling_frequency") && msg.contains("PerTree"),
        "the refusal must name the parameter and the supported value; got: {msg}"
    );
}

/// `sampling_unit=Group` is refused through the public surface. Upstream refuses it
/// too on this ungrouped pool — `meta.json`'s `group_sampling_unit_rejection` records
/// its message — so the refusal is parity here rather than a gap.
#[test]
fn group_sampling_unit_is_refused() {
    let err = with_sampler(builder(), EBootstrapType::Bernoulli)
        .sampling_unit(ESamplingUnit::Group)
        .fit(&learn_pool())
        .expect_err("sampling_unit=Group must be refused");
    assert!(
        err.to_string().contains("sampling_unit"),
        "the refusal must name the parameter; got: {err}"
    );
}
