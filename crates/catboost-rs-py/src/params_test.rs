//! Unit asserts for the D-07 param-vocabulary registry, alias resolution, the
//! fit()-time validator, and the kwargs -> [`CatBoostBuilder`] map.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};

use crate::errors::CatBoostParameterError;
use crate::params::{make_builder, status_of, status_of_user, validate_params, ParamStatus};

/// Build a verbatim kwargs map (mirroring `EstimatorBase::from_kwargs`) from a
/// Python dict literal, for driving the validator / builder map.
fn params_from<'py>(_py: Python<'py>, dict: &Bound<'py, PyDict>) -> BTreeMap<String, Py<PyAny>> {
    let mut params = BTreeMap::new();
    for (k, v) in dict.iter() {
        let name: String = k.extract().expect("str key");
        params.insert(name, v.unbind());
    }
    params
}

/// The IMPLEMENTED canonical params tag as Implemented.
#[test]
fn implemented_params_tag_implemented() {
    for name in [
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "loss_function",
        "border_count",
        "random_seed",
        "random_strength",
        "bagging_temperature",
        "bootstrap_type",
        "subsample",
        "score_function",
        "boost_from_average",
        "leaf_estimation_method",
    ] {
        assert_eq!(
            status_of(name),
            Some(ParamStatus::Implemented),
            "{name} should be Implemented"
        );
    }
}

/// A real-but-unimplemented upstream param tags KnownNotYet (parity gap).
#[test]
fn known_not_yet_params_tag_known_not_yet() {
    for name in ["nan_mode", "od_wait", "rsm", "max_ctr_complexity", "thread_count"] {
        assert_eq!(
            status_of(name),
            Some(ParamStatus::KnownNotYet),
            "{name} should be KnownNotYet"
        );
    }
}

/// A name outside the upstream vocabulary is unknown (None).
#[test]
fn unknown_param_is_none() {
    assert_eq!(status_of("not_a_real_param"), None);
    assert_eq!(status_of("iteratons"), None); // typo
}

/// Aliases resolve to their canonical status: implemented aliases -> Implemented,
/// `colsample_bylevel` -> `rsm` -> KnownNotYet.
#[test]
fn aliases_resolve_to_canonical_status() {
    for name in [
        "max_depth",
        "n_estimators",
        "num_trees",
        "num_boost_round",
        "random_state",
        "reg_lambda",
        "objective",
        "eta",
        "max_bin",
    ] {
        assert_eq!(
            status_of_user(name),
            Some(ParamStatus::Implemented),
            "{name} alias should resolve to Implemented"
        );
    }
    // colsample_bylevel -> rsm (no builder setter) is honestly a parity gap.
    assert_eq!(
        status_of_user("colsample_bylevel"),
        Some(ParamStatus::KnownNotYet)
    );
}

/// `validate_params` accepts only IMPLEMENTED params (incl. aliases).
#[test]
fn validate_accepts_implemented_and_aliases() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("iterations", 10).unwrap();
        dict.set_item("n_estimators", 5).unwrap();
        dict.set_item("max_depth", 3).unwrap();
        dict.set_item("reg_lambda", 2.0).unwrap();
        let params = params_from(py, &dict);
        assert!(validate_params(&params).is_ok());
    });
}

/// `validate_params` rejects a KnownNotYet param as a parity gap.
#[test]
fn validate_rejects_known_not_yet_as_parity_gap() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("nan_mode", "Min").unwrap();
        let params = params_from(py, &dict);
        let err = validate_params(&params).unwrap_err();
        assert!(err.is_instance_of::<CatBoostParameterError>(py));
        let msg = err.value(py).to_string();
        assert!(msg.contains("nan_mode"), "msg: {msg}");
        assert!(msg.contains("parity gap"), "msg: {msg}");
    });
}

/// `validate_params` rejects an unknown param and suggests the closest match.
#[test]
fn validate_rejects_unknown_with_suggestion() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("iteratons", 10).unwrap(); // typo of iterations
        let params = params_from(py, &dict);
        let err = validate_params(&params).unwrap_err();
        assert!(err.is_instance_of::<CatBoostParameterError>(py));
        let msg = err.value(py).to_string();
        assert!(msg.contains("iteratons"), "msg: {msg}");
        assert!(msg.contains("iterations"), "should suggest iterations: {msg}");
    });
}

/// The kwargs -> Builder map accepts every IMPLEMENTED param with correct typed
/// extraction (no panic, no extract error).
#[test]
fn builder_map_applies_implemented_params() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("iterations", 7).unwrap();
        dict.set_item("depth", 4).unwrap();
        dict.set_item("learning_rate", 0.1).unwrap();
        dict.set_item("l2_leaf_reg", 2.5).unwrap();
        dict.set_item("random_strength", 1.0).unwrap();
        dict.set_item("random_seed", 42).unwrap();
        dict.set_item("border_count", 128).unwrap();
        dict.set_item("subsample", 0.8).unwrap();
        dict.set_item("bagging_temperature", 0.5).unwrap();
        dict.set_item("bootstrap_type", "Bernoulli").unwrap();
        dict.set_item("score_function", "L2").unwrap();
        dict.set_item("loss_function", "RMSE").unwrap();
        dict.set_item("boost_from_average", true).unwrap();
        dict.set_item("leaf_estimation_method", "Newton").unwrap();
        let params = params_from(py, &dict);
        assert!(validate_params(&params).is_ok());
        assert!(make_builder(&params, py).is_ok());
    });
}

/// The builder map resolves aliases (n_estimators/max_depth/reg_lambda).
#[test]
fn builder_map_resolves_aliases() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("n_estimators", 9).unwrap();
        dict.set_item("max_depth", 3).unwrap();
        dict.set_item("reg_lambda", 4.0).unwrap();
        let params = params_from(py, &dict);
        assert!(validate_params(&params).is_ok());
        assert!(make_builder(&params, py).is_ok());
    });
}

/// An unsupported enum string surfaces as a CatBoostParameterError from the
/// builder map.
#[test]
fn builder_map_rejects_bad_enum_string() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("bootstrap_type", "Nonsense").unwrap();
        let params = params_from(py, &dict);
        let err = make_builder(&params, py).unwrap_err();
        assert!(err.is_instance_of::<CatBoostParameterError>(py));
    });
}
