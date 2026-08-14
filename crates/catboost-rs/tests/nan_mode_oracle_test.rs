//! `nan_mode` parity oracle (Min / Max / Forbidden) against catboost 1.2.10.
//!
//! # What this covers
//!
//! Upstream isolates missing float values into their own quantization bin with a
//! SENTINEL border, and records the routing on the float feature:
//!
//! | `nan_mode`  | sentinel                 | `nan_value_treatment` |
//! |-------------|--------------------------|-----------------------|
//! | `Min`       | `f32::MIN` PREPENDED     | `AsFalse`             |
//! | `Max`       | `f32::MAX` APPENDED      | `AsTrue`              |
//! | `Forbidden` | none — a NaN column is REJECTED at fit time    |          |
//!
//! Before this wave the Rust fit path added NO sentinel at all, so NaN silently
//! shared bin 0 with the smallest real values on ANY NaN-bearing dataset.
//!
//! # Why the fixture's target is value-driven
//!
//! A target driven by MISSINGNESS makes this test vacuous: the sentinel isolates
//! the NaN bin completely, both modes learn the same "NaN -> c" leaf, and their
//! predictions coincide exactly. The frozen corpus therefore drives `y` from
//! feature 0's VALUE, so NaN rows ride along at ordinary borders and the two
//! modes separate by ~2.0 (see `generator/gen_nan_mode_fixtures.py`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, IngestSource, OwnedColumns, Pool};
use cb_compute::Loss;
use cb_data::NanMode;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

/// The parity bar for this repository (CLAUDE.md).
const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("nan_mode")
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

/// The pinned fit from `gen_nan_mode_fixtures.py::PARAMS`.
fn builder(nan_mode: NanMode) -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(true)
        .random_seed(0)
        .border_count(16)
        .score_function(cb_compute::EScoreFunction::L2)
        .nan_mode(nan_mode)
}

/// Build a [`Pool`] from float columns plus a target.
fn pool_of(cols: Vec<Vec<f64>>, target: Vec<f64>) -> Pool {
    OwnedColumns::new(cols, target)
        .into_pool()
        .expect("pool must build")
}

/// The held-out eval pool. The label is never read by the apply path, so a zero
/// vector keeps the fixture prediction-only.
fn eval_pool() -> Pool {
    let cols = load_x("X_eval.npy");
    let n = cols.first().map_or(0, Vec::len);
    pool_of(cols, vec![0.0; n])
}

fn fit_and_predict(nan_mode: NanMode) -> Vec<f64> {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let model = builder(nan_mode).fit(&pool).expect("the fit must succeed");
    model.predict(&eval_pool()).expect("predict must succeed")
}

fn check_mode(nan_mode: NanMode, preds_file: &str) {
    let actual = fit_and_predict(nan_mode);
    let expected = load_y(preds_file);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{nan_mode:?}: prediction count mismatch"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{nan_mode:?}: row {i} predicted {a} but catboost 1.2.10 says {e} \
             (|diff| {} > {TOL})",
            (a - e).abs()
        );
    }
}

#[test]
fn nan_mode_min_matches_catboost() {
    check_mode(NanMode::Min, "preds_Min.npy");
}

#[test]
fn nan_mode_max_matches_catboost() {
    check_mode(NanMode::Max, "preds_Max.npy");
}

/// `nan_mode=Forbidden` must REJECT a NaN-bearing learn column rather than
/// quietly training on it, mirroring upstream `quantization.cpp:320`.
#[test]
fn nan_mode_forbidden_rejects_a_nan_column() {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let err = builder(NanMode::Forbidden)
        .fit(&pool)
        .expect_err("a NaN column under nan_mode=Forbidden must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("nan") && msg.contains("Forbidden"),
        "the rejection must name the offending condition and the setting; got: {msg}"
    );
}

/// A NaN-FREE pool is unaffected by `nan_mode` — the sentinel only exists for a
/// column that actually carries missing values, so all three settings must give
/// byte-identical predictions. This is what proves the wave did not perturb the
/// ordinary numeric path.
#[test]
fn nan_mode_is_inert_on_a_nan_free_pool() {
    let mut cols = load_x("X.npy");
    // Replace the NaNs with a finite value, keeping everything else identical.
    for v in cols[0].iter_mut() {
        if v.is_nan() {
            *v = 0.0;
        }
    }
    let y = load_y("y.npy");

    let predict_with = |mode: NanMode| -> Vec<f64> {
        let pool = pool_of(cols.clone(), y.clone());
        let model = builder(mode).fit(&pool).expect("the fit must succeed");
        model.predict(&eval_pool()).expect("predict must succeed")
    };

    let min = predict_with(NanMode::Min);
    let max = predict_with(NanMode::Max);
    let forbidden = predict_with(NanMode::Forbidden);
    assert_eq!(min, max, "nan_mode must be inert without NaNs (Min vs Max)");
    assert_eq!(
        min, forbidden,
        "nan_mode must be inert without NaNs (Min vs Forbidden)"
    );
}

/// The fixture itself must be able to tell the two modes apart. If Min and Max
/// ever collapse onto the same predictions the parity tests above would pass for
/// an implementation that ignores `nan_mode` entirely.
#[test]
fn the_frozen_fixture_separates_min_from_max() {
    let min = load_y("preds_Min.npy");
    let max = load_y("preds_Max.npy");
    let sep = min
        .iter()
        .zip(max.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        sep > TOL,
        "the frozen Min and Max predictions differ by only {sep}, so this fixture \
         cannot detect an implementation that ignores nan_mode"
    );
}
