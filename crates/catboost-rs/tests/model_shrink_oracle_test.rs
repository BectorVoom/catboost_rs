//! `model_shrink_rate` / `model_shrink_mode` parity oracle against catboost
//! 1.2.10.
//!
//! Upstream multiplies the ENTIRE accumulated model — the bias and every
//! already-grown tree's leaf values — before growing each new tree, which also
//! rescales the running approximant the next tree's gradients come from. The
//! model-level scale stays 1; the shrinkage is baked into leaves and bias.
//!
//! The first tree is never shrunk, so `iterations = 5` performs FOUR
//! applications. Both multipliers were derived from catboost's own saved leaves
//! at `rate = 0.1`, `learning_rate = 0.3`:
//!
//! | mode         | multiplier                | tree 0 scaled by |
//! |--------------|---------------------------|------------------|
//! | `Constant`   | `1 - rate * learning_rate`| `0.97^4 = 0.88529` |
//! | `Decreasing` | `1 - rate / i`            | `0.9*0.95*0.96667*0.975 = 0.80583` |
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, EModelShrinkMode, IngestSource, OwnedColumns, Pool};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;
const SHRINK_RATE: f64 = 0.1;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("model_shrink")
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

/// The pinned fit from `gen_model_shrink_fixtures.py::PARAMS`.
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

fn predict_with(rate: f64, mode: EModelShrinkMode) -> Vec<f64> {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let model = builder()
        .model_shrink_rate(rate)
        .model_shrink_mode(mode)
        .fit(&pool)
        .expect("the fit must succeed");
    model.predict(&eval_pool()).expect("predict must succeed")
}

fn check(mode: EModelShrinkMode, preds_file: &str) {
    let actual = predict_with(SHRINK_RATE, mode);
    let expected = load_y(preds_file);
    assert_eq!(actual.len(), expected.len(), "{mode:?}: prediction count");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{mode:?}: row {i} predicted {a} but catboost 1.2.10 says {e} (|diff| {})",
            (a - e).abs()
        );
    }
}

#[test]
fn model_shrink_constant_matches_catboost() {
    check(EModelShrinkMode::Constant, "preds_Constant.npy");
}

#[test]
fn model_shrink_decreasing_matches_catboost() {
    check(EModelShrinkMode::Decreasing, "preds_Decreasing.npy");
}

/// `model_shrink_rate = 0` must be completely INERT — the guard in the boosting
/// loop skips the whole shrink block, so an unshrunk fit is byte-identical to
/// one that never mentions the parameter. This is what proves the wave did not
/// perturb the default training path.
#[test]
fn model_shrink_rate_zero_is_inert() {
    let expected = load_y("preds_none.npy");
    for mode in EModelShrinkMode::all() {
        let actual = predict_with(0.0, mode);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= TOL,
                "rate=0 {mode:?}: row {i} moved to {a} from the unshrunk {e}"
            );
        }
    }
}

/// The frozen fixture must actually separate the two modes AND separate each
/// from an unshrunk fit; otherwise the parity tests above would pass for an
/// implementation that ignores the parameters entirely.
#[test]
fn the_frozen_fixture_separates_the_modes_and_the_unshrunk_fit() {
    let none = load_y("preds_none.npy");
    let constant = load_y("preds_Constant.npy");
    let decreasing = load_y("preds_Decreasing.npy");

    let sep = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };

    assert!(
        sep(&constant, &decreasing) > TOL,
        "Constant and Decreasing must differ; got {}",
        sep(&constant, &decreasing)
    );
    assert!(
        sep(&constant, &none) > TOL,
        "Constant must differ from an unshrunk fit"
    );
    assert!(
        sep(&decreasing, &none) > TOL,
        "Decreasing must differ from an unshrunk fit"
    );
}

/// The multipliers themselves, checked against the values read out of catboost's
/// saved leaf scaling (see the module doc).
#[test]
fn shrink_multipliers_match_the_values_read_from_catboost() {
    let rate = 0.1;
    let lr = 0.3;
    let constant: f64 = (1..=4)
        .map(|i| EModelShrinkMode::Constant.multiplier(rate, lr, i))
        .product();
    let decreasing: f64 = (1..=4)
        .map(|i| EModelShrinkMode::Decreasing.multiplier(rate, lr, i))
        .product();
    assert!(
        (constant - 0.885_292_81).abs() < 1e-8,
        "Constant 4-step product should be 0.97^4 = 0.88529281, got {constant}"
    );
    assert!(
        (decreasing - 0.805_837_5).abs() < 1e-8,
        "Decreasing 4-step product should be 0.8058375, got {decreasing}"
    );
}

/// Every legal token round-trips; an unknown one is rejected.
#[test]
fn model_shrink_mode_parses_every_legal_token() {
    for m in EModelShrinkMode::all() {
        assert_eq!(EModelShrinkMode::parse(m.as_str()), Some(m));
    }
    assert_eq!(EModelShrinkMode::parse("constant"), None);
    assert_eq!(EModelShrinkMode::parse("ZzBogusValue"), None);
}
