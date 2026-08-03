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
///
/// `max_ctr_complexity` was in this list until F15 PROMOTED it (along with the
/// other five categorical / CTR params) — it is now genuinely consumed, so it
/// belongs in `IMPLEMENTED`. `ctr_leaf_count_limit` replaces it here as a real
/// upstream CTR param that is still unimplemented, keeping the CTR area
/// represented in this guard.
#[test]
fn known_not_yet_params_tag_known_not_yet() {
    for name in [
        "nan_mode",
        "od_wait",
        "rsm",
        "ctr_leaf_count_limit",
        "thread_count",
    ] {
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

// ---------------------------------------------------------------------------
// F15 / F16 (SPEC-CATF-13, SPEC-CATF-Δ2) — the six promoted categorical params
// ---------------------------------------------------------------------------

/// F15 — the three scalar categorical knobs reach the builder.
#[test]
fn builder_map_applies_the_scalar_categorical_params() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("one_hot_max_size", 5).unwrap();
        d.set_item("max_ctr_complexity", 2).unwrap();
        d.set_item("counter_calc_method", "Full").unwrap();
        let built = make_builder(&params_from(py, &d), py).expect("builder must build");

        let expected = catboost_rs::CatBoostBuilder::new()
            .one_hot_max_size(5)
            .max_ctr_complexity(2)
            .counter_calc_method(catboost_rs::CounterCalcMethod::Full);
        assert_eq!(built, expected);
    });
}

/// F16 — a single CTR description with its FULL prior list reaches the builder,
/// for both the simple and the combination family.
#[test]
fn builder_map_applies_the_ctr_description_grammar() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("simple_ctr", vec!["Buckets:Prior=0/1:Prior=0.5"])
            .unwrap();
        d.set_item("combinations_ctr", vec!["Counter:Prior=0.5:Prior=1"])
            .unwrap();
        let built = make_builder(&params_from(py, &d), py).expect("builder must build");

        let expected = catboost_rs::CatBoostBuilder::new()
            .simple_ctr(catboost_rs::ECtrType::Buckets)
            .simple_ctr_priors(vec![0.0, 0.5])
            .combinations_ctr(catboost_rs::ECtrType::Counter)
            .combinations_ctr_priors(vec![0.5, 1.0]);
        assert_eq!(built, expected);
    });
}

/// F16 — **THE RECORDED `combinations_ctr=[]` MAPPING (option (a))**.
///
/// The engine field is a scalar `ECtrType` with NO "disabled" representation;
/// `max_ctr_complexity = 1` is the only in-engine way to suppress combination
/// CTRs. Option (a) was chosen over rejecting `[]` because the committed
/// `crates/cb-oracle/fixtures/plain_ctr/config.json` uses exactly the `[]` form,
/// so F19 requires it be reachable through the Python surface.
///
/// This asserts the mapping ACTUALLY CHANGES `max_ctr_complexity` — not merely
/// that `fit()` does not raise, which a silently-ignored kwarg would also pass.
#[test]
fn empty_combinations_ctr_maps_to_max_ctr_complexity_one() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("combinations_ctr", Vec::<String>::new()).unwrap();
        let built = make_builder(&params_from(py, &d), py).expect("builder must build");

        assert_eq!(built, catboost_rs::CatBoostBuilder::new().max_ctr_complexity(1));
        assert_ne!(
            built,
            catboost_rs::CatBoostBuilder::new(),
            "combinations_ctr=[] must ACTUALLY suppress combination CTRs, not be ignored"
        );
    });
}

/// F16 — a CPU-illegal CTR type is rejected by NAME, mirroring the engine-side
/// E02 guard and upstream `catboost_options.cpp:504-509`.
#[test]
fn cpu_illegal_ctr_types_are_rejected() {
    Python::attach(|py| {
        for bad in ["FloatTargetMeanValue", "FeatureFreq"] {
            let d = PyDict::new(py);
            d.set_item("simple_ctr", vec![bad]).unwrap();
            let err = make_builder(&params_from(py, &d), py)
                .expect_err("a CPU-illegal CTR type must be rejected");
            assert!(err.is_instance_of::<CatBoostParameterError>(py));
            let msg = err.to_string();
            assert!(msg.contains(bad), "the error must name the type: {msg}");
            assert!(msg.contains("not implemented on CPU"), "got: {msg}");
            for ok in ["Borders", "Buckets", "BinarizedTargetMeanValue", "Counter"] {
                assert!(msg.contains(ok), "the error must list `{ok}`: {msg}");
            }
        }
    });
}

/// F16 — more than one CTR description is REJECTED, naming the parity gap.
/// Silently narrowing to the first entry is exactly what BLOCKER-2 was about.
#[test]
fn multiple_ctr_descriptions_are_rejected_as_a_named_parity_gap() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("simple_ctr", vec!["Borders:Prior=0.5", "Counter:Prior=0"])
            .unwrap();
        let err = make_builder(&params_from(py, &d), py)
            .expect_err("a two-description list must be rejected, never narrowed");
        let msg = err.to_string();
        assert!(msg.contains("one CTR description"), "got: {msg}");
    });
}

/// F16 — `Prior=<n>/<d>` with `d != 1` is illegal on CPU upstream
/// (`ctr_helper.cpp:50`), which is what vindicates the engine's
/// `prior_denom: 1.0` pin.
#[test]
fn non_unit_prior_denominator_is_rejected() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("simple_ctr", vec!["Borders:Prior=1/2"]).unwrap();
        let err = make_builder(&params_from(py, &d), py)
            .expect_err("a non-unit prior denominator must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("denominator"), "got: {msg}");
    });
}

/// F16 — a unit denominator written explicitly (`Prior=0.5/1`) is ACCEPTED and
/// equals the bare form, so upstream's own descriptor spelling round-trips.
#[test]
fn explicit_unit_prior_denominator_is_accepted() {
    Python::attach(|py| {
        let bare = PyDict::new(py);
        bare.set_item("simple_ctr", vec!["Borders:Prior=0.5"]).unwrap();
        let explicit = PyDict::new(py);
        explicit
            .set_item("simple_ctr", vec!["Borders:Prior=0.5/1"])
            .unwrap();
        assert_eq!(
            make_builder(&params_from(py, &bare), py).expect("bare"),
            make_builder(&params_from(py, &explicit), py).expect("explicit"),
        );
    });
}
