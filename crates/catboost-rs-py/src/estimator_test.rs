//! Unit tests for the Rust-visible sklearn-contract helpers in
//! [`crate::estimator`] (source/test separation, CLAUDE.md): the `score` math
//! (`r2_score` / `accuracy_score`) and the verbatim `get_params`/`set_params`
//! round-trip over [`EstimatorBase`].

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::estimator::{accuracy_score, r2_score, EstimatorBase};

#[test]
fn r2_perfect_fit_is_one() {
    let y = [1.0, 2.0, 3.0, 4.0];
    let pred = [1.0, 2.0, 3.0, 4.0];
    assert!((r2_score(&y, &pred) - 1.0).abs() < 1e-12);
}

#[test]
fn r2_mean_predictor_is_zero() {
    // Predicting the mean of y gives SS_res == SS_tot => R² == 0.
    let y = [1.0, 2.0, 3.0, 4.0];
    let mean = 2.5;
    let pred = [mean, mean, mean, mean];
    assert!(r2_score(&y, &pred).abs() < 1e-12);
}

#[test]
fn r2_constant_y_perfect_is_one_else_zero() {
    let y = [5.0, 5.0, 5.0];
    assert!((r2_score(&y, &[5.0, 5.0, 5.0]) - 1.0).abs() < 1e-12);
    assert!(r2_score(&y, &[5.0, 5.0, 6.0]).abs() < 1e-12);
}

#[test]
fn accuracy_counts_rounded_matches() {
    let y = [0.0, 1.0, 1.0, 0.0];
    // predict emits f64 labels; round before comparing.
    let pred = [0.0, 1.0, 0.0, 0.0];
    assert!((accuracy_score(&y, &pred) - 0.75).abs() < 1e-12);
}

#[test]
fn accuracy_empty_is_zero() {
    assert!(accuracy_score(&[], &[]).abs() < 1e-12);
}

#[test]
fn get_set_params_round_trip_is_verbatim() {
    Python::attach(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("iterations", 5_i64).unwrap();
        kwargs.set_item("learning_rate", 0.1_f64).unwrap();

        let mut base = EstimatorBase::from_kwargs(Some(&kwargs)).unwrap();

        // get_params returns exactly the stored kwargs.
        let got = base.get_params(py, None).unwrap();
        let it: i64 = got.get_item("iterations").unwrap().unwrap().extract().unwrap();
        let lr: f64 = got
            .get_item("learning_rate")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(it, 5);
        assert!((lr - 0.1).abs() < 1e-12);
        assert_eq!(got.len(), 2);

        // set_params(**get_params()) is an identity round-trip.
        base.set_params(Some(&got)).unwrap();
        let again = base.get_params(py, None).unwrap();
        let it2: i64 = again
            .get_item("iterations")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(it2, 5);
        assert_eq!(again.len(), 2);

        // set_params can add a new key (sklearn allows setting any __init__ param).
        let extra = PyDict::new(py);
        extra.set_item("depth", 3_i64).unwrap();
        base.set_params(Some(&extra)).unwrap();
        let merged = base.get_params(py, None).unwrap();
        assert_eq!(merged.len(), 3);
        let d: i64 = merged.get_item("depth").unwrap().unwrap().extract().unwrap();
        assert_eq!(d, 3);
    });
}

#[test]
fn is_fitted_false_before_fit() {
    Python::attach(|py| {
        let kwargs = PyDict::new(py);
        let base = EstimatorBase::from_kwargs(Some(&kwargs)).unwrap();
        assert!(!base.is_fitted());
    });
}
