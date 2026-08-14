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
        // PARAM-02 promotions (each now maps onto a PARAM-01 builder setter).
        "od_type",
        "od_pval",
        "od_wait",
        "early_stopping_rounds",
        "use_best_model",
        "eval_metric",
        "boosting_type",
        "has_time",
        "fold_len_multiplier",
        "monotone_constraints",
        "feature_weights",
        "penalties_coefficient",
        "first_feature_use_penalties",
        "per_object_feature_penalties",
        "grow_policy",
        "max_leaves",
        "min_data_in_leaf",
        // PARAM-03 promotions.
        "class_weights",
        "auto_class_weights",
        "scale_pos_weight",
        "ignored_features",
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
///
/// `od_wait` left this list in PARAM-02, for the same reason: `CatBoostBuilder`
/// grew an `od_wait` setter (PARAM-01), so calling it a parity gap became
/// dishonest in the opposite direction. `leaf_estimation_iterations` replaces it
/// as a still-unimplemented member of the same (leaf/boosting-control) area.
#[test]
fn known_not_yet_params_tag_known_not_yet() {
    for name in [
        // `nan_mode` used to sit here; the string-valued-parameter wave
        // implemented it (sentinel-border quantization + a frozen catboost
        // oracle), so calling it a parity gap became dishonest in the opposite
        // direction. `model_shrink_rate` replaces it as a still-unimplemented
        // member of the same (data/boosting-control) area.
        "model_shrink_rate",
        "leaf_estimation_iterations",
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
        // PARAM-02: the two LightGBM-style leaf aliases.
        "num_leaves",
        "min_child_samples",
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
        dict.set_item("model_shrink_rate", 0.1).unwrap();
        let params = params_from(py, &dict);
        let err = validate_params(&params).unwrap_err();
        assert!(err.is_instance_of::<CatBoostParameterError>(py));
        let msg = err.value(py).to_string();
        assert!(msg.contains("model_shrink_rate"), "msg: {msg}");
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
// The string-valued-parameter wave — `feature_border_type` + `nan_mode`
// ---------------------------------------------------------------------------

/// Both quantization params are tagged IMPLEMENTED, so they are ACCEPTED rather
/// than rejected as a parity gap. Each is genuinely consumed by `fit`'s
/// quantization stage and gated by a frozen catboost 1.2.10 oracle.
#[test]
fn quantization_string_params_are_implemented() {
    for name in ["feature_border_type", "nan_mode"] {
        assert_eq!(
            status_of(name),
            Some(ParamStatus::Implemented),
            "{name} should be Implemented"
        );
    }
}

/// EVERY legal value of both params reaches the builder. The legal sets are the
/// ones the catboost 1.2.10 enum parser reports for `EBorderSelectionType` and
/// `ENanMode` — probed from the wheel, not transcribed.
#[test]
fn builder_map_applies_every_legal_quantization_string() {
    Python::attach(|py| {
        for value in [
            "Median",
            "GreedyLogSum",
            "UniformAndQuantiles",
            "MinEntropy",
            "MaxLogSum",
            "Uniform",
            "GreedyMinEntropy",
        ] {
            let d = PyDict::new(py);
            d.set_item("feature_border_type", value).unwrap();
            make_builder(&params_from(py, &d), py)
                .unwrap_or_else(|e| panic!("feature_border_type={value} must build: {e:?}"));
        }
        for value in ["Min", "Max", "Forbidden"] {
            let d = PyDict::new(py);
            d.set_item("nan_mode", value).unwrap();
            make_builder(&params_from(py, &d), py)
                .unwrap_or_else(|e| panic!("nan_mode={value} must build: {e:?}"));
        }
    });
}

/// An out-of-vocabulary value for either is a typed error naming the legal set,
/// never a silent fallback to the default binarizer / NaN policy.
#[test]
fn builder_map_rejects_bad_quantization_strings() {
    Python::attach(|py| {
        for (key, bad) in [
            ("feature_border_type", "greedylogsum"),
            ("feature_border_type", "Nonsense"),
            ("nan_mode", "min"),
            ("nan_mode", "Nonsense"),
        ] {
            let d = PyDict::new(py);
            d.set_item(key, bad).unwrap();
            let err = make_builder(&params_from(py, &d), py)
                .expect_err(&format!("{key}={bad} must be rejected"));
            assert!(err.is_instance_of::<CatBoostParameterError>(py));
        }
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

// ─── PARAM-02: the newly-promoted param surface ─────────────────────────────
//
// `make_builder` returns a `CatBoostBuilder`, which is `PartialEq` but exposes
// no field getters — so these tests assert the SAME way the pre-existing ones
// do: a builder built with the kwarg must DIFFER from one built without it
// (proving the value was consumed), and an invalid value must be REJECTED with a
// message naming the parameter.

/// PARAM-02 — each promoted param actually MOVES the builder. A param that
/// parsed but was never applied would produce an equal builder, which is exactly
/// the silent-ignore failure the registry exists to prevent.
#[test]
fn every_promoted_param_changes_the_builder() {
    Python::attach(|py| {
        let baseline = make_builder(&params_from(py, &PyDict::new(py)), py).expect("baseline");
        // (kwarg, a value that differs from the default)
        let cases: Vec<(&str, Box<dyn Fn(&Bound<'_, PyDict>)>)> = vec![
            ("od_type", Box::new(|d: &Bound<'_, PyDict>| d.set_item("od_type", "Iter").unwrap())),
            ("od_pval", Box::new(|d: &Bound<'_, PyDict>| d.set_item("od_pval", 0.05).unwrap())),
            ("od_wait", Box::new(|d: &Bound<'_, PyDict>| d.set_item("od_wait", 11).unwrap())),
            (
                "early_stopping_rounds",
                Box::new(|d: &Bound<'_, PyDict>| d.set_item("early_stopping_rounds", 9).unwrap()),
            ),
            (
                "use_best_model",
                Box::new(|d: &Bound<'_, PyDict>| d.set_item("use_best_model", true).unwrap()),
            ),
            (
                "eval_metric",
                Box::new(|d: &Bound<'_, PyDict>| d.set_item("eval_metric", "MAE").unwrap()),
            ),
            (
                "boosting_type",
                Box::new(|d: &Bound<'_, PyDict>| d.set_item("boosting_type", "Ordered").unwrap()),
            ),
            ("has_time", Box::new(|d: &Bound<'_, PyDict>| d.set_item("has_time", true).unwrap())),
            (
                "fold_len_multiplier",
                Box::new(|d: &Bound<'_, PyDict>| d.set_item("fold_len_multiplier", 3.0).unwrap()),
            ),
            (
                "monotone_constraints",
                Box::new(|d: &Bound<'_, PyDict>| {
                    d.set_item("monotone_constraints", vec![1, 0, -1]).unwrap()
                }),
            ),
            (
                "feature_weights",
                Box::new(|d: &Bound<'_, PyDict>| {
                    d.set_item("feature_weights", vec![2.0, 1.0]).unwrap()
                }),
            ),
            (
                "penalties_coefficient",
                Box::new(|d: &Bound<'_, PyDict>| {
                    d.set_item("penalties_coefficient", 4.0).unwrap()
                }),
            ),
            (
                "first_feature_use_penalties",
                Box::new(|d: &Bound<'_, PyDict>| {
                    d.set_item("first_feature_use_penalties", vec![0.5]).unwrap()
                }),
            ),
            (
                "per_object_feature_penalties",
                Box::new(|d: &Bound<'_, PyDict>| {
                    d.set_item("per_object_feature_penalties", vec![0.5]).unwrap()
                }),
            ),
            (
                "grow_policy",
                Box::new(|d: &Bound<'_, PyDict>| d.set_item("grow_policy", "Lossguide").unwrap()),
            ),
            ("max_leaves", Box::new(|d: &Bound<'_, PyDict>| d.set_item("max_leaves", 17).unwrap())),
            (
                "min_data_in_leaf",
                Box::new(|d: &Bound<'_, PyDict>| d.set_item("min_data_in_leaf", 4).unwrap()),
            ),
        ];
        for (name, set) in cases {
            let d = PyDict::new(py);
            set(&d);
            let built = make_builder(&params_from(py, &d), py)
                .unwrap_or_else(|e| panic!("{name} must build: {e}"));
            assert_ne!(built, baseline, "`{name}` was accepted but never applied");
        }
    });
}

/// PARAM-02 — the two LightGBM aliases reach the SAME builder as their canonical
/// names, which is what makes them aliases rather than separately-parsed kwargs.
#[test]
fn leaf_aliases_build_the_same_builder_as_their_canonical_names() {
    Python::attach(|py| {
        let canonical = PyDict::new(py);
        canonical.set_item("max_leaves", 21).unwrap();
        canonical.set_item("min_data_in_leaf", 6).unwrap();
        let aliased = PyDict::new(py);
        aliased.set_item("num_leaves", 21).unwrap();
        aliased.set_item("min_child_samples", 6).unwrap();
        assert_eq!(
            make_builder(&params_from(py, &canonical), py).expect("canonical"),
            make_builder(&params_from(py, &aliased), py).expect("aliased"),
        );
    });
}

/// PARAM-02 — `early_stopping_rounds` is REJECTED alongside an explicit
/// `od_type` OR `od_wait`. It is shorthand for that exact pair, so honouring one
/// would silently discard the other.
#[test]
fn early_stopping_rounds_conflicts_with_explicit_od_params() {
    Python::attach(|py| {
        for other in ["od_type", "od_wait"] {
            let d = PyDict::new(py);
            d.set_item("early_stopping_rounds", 5).unwrap();
            if other == "od_type" {
                d.set_item("od_type", "IncToDec").unwrap();
            } else {
                d.set_item("od_wait", 5).unwrap();
            }
            assert!(
                make_builder(&params_from(py, &d), py).is_err(),
                "`{other}` alongside early_stopping_rounds must be rejected"
            );
        }
    });
}

/// PARAM-02 — the conflict above raises with a message naming both spellings.
#[test]
fn early_stopping_rounds_conflict_names_both_spellings() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("early_stopping_rounds", 5).unwrap();
        d.set_item("od_wait", 7).unwrap();
        let err = make_builder(&params_from(py, &d), py)
            .expect_err("the shorthand and the pair must not be combined");
        let msg = err.to_string();
        assert!(
            msg.contains("early_stopping_rounds") && msg.contains("od_wait"),
            "got: {msg}"
        );
    });
}

/// PARAM-02 — `grow_policy="Region"` is rejected BY NAME with the parity-gap
/// reason, rather than reaching the engine's internal validator.
#[test]
fn region_grow_policy_is_rejected_with_its_own_reason() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("grow_policy", "Region").unwrap();
        let err = make_builder(&params_from(py, &d), py).expect_err("Region must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("Region") && msg.contains("parity gap"),
            "got: {msg}"
        );
        assert!(
            err.is_instance_of::<CatBoostParameterError>(py),
            "must be a CatBoostParameterError"
        );
    });
}

/// PARAM-02 — an unsupported `eval_metric` is rejected NAMING the metric, using
/// the engine's own parser so the accepted set cannot drift from what the
/// trainer computes.
#[test]
fn an_unsupported_eval_metric_is_rejected() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("eval_metric", "NotAMetric").unwrap();
        let err = make_builder(&params_from(py, &d), py).expect_err("must be rejected");
        assert!(err.to_string().contains("NotAMetric"), "got: {err}");
    });
}

/// PARAM-02 — a parametric `eval_metric` descriptor round-trips through the
/// engine parser (the reason we reuse it rather than matching on bare names).
#[test]
fn a_parametric_eval_metric_descriptor_is_accepted() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("eval_metric", "Quantile:alpha=0.9").unwrap();
        make_builder(&params_from(py, &d), py).expect("a parametric descriptor must be accepted");
    });
}

/// PARAM-02 — all three `monotone_constraints` spellings produce the SAME
/// builder. The dict form is keyed by feature INDEX and pads unlisted features
/// with `0` (free).
#[test]
fn the_monotone_constraint_spellings_agree() {
    Python::attach(|py| {
        let list = PyDict::new(py);
        list.set_item("monotone_constraints", vec![1, 0, -1]).unwrap();
        let text = PyDict::new(py);
        text.set_item("monotone_constraints", "(1,0,-1)").unwrap();
        let map = PyDict::new(py);
        let inner = PyDict::new(py);
        inner.set_item(0, 1).unwrap();
        inner.set_item(2, -1).unwrap();
        map.set_item("monotone_constraints", inner).unwrap();

        let from_list = make_builder(&params_from(py, &list), py).expect("list form");
        let from_text = make_builder(&params_from(py, &text), py).expect("string form");
        let from_map = make_builder(&params_from(py, &map), py).expect("dict form");
        assert_eq!(from_list, from_text, "the string form must equal the list form");
        assert_eq!(
            from_list, from_map,
            "the index-keyed dict form must equal the list form"
        );
    });
}

/// PARAM-02 — an out-of-domain monotone constraint is rejected naming the value.
#[test]
fn an_invalid_monotone_constraint_value_is_rejected() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        d.set_item("monotone_constraints", vec![1, 2]).unwrap();
        let err = make_builder(&params_from(py, &d), py).expect_err("2 is not a constraint");
        assert!(err.to_string().contains('2'), "got: {err}");
    });
}

/// PARAM-02 — a NAME-keyed monotone dict is rejected rather than silently
/// dropped: the engine applies constraints positionally and a Pool carries no
/// feature names, so honouring it is impossible.
#[test]
fn a_name_keyed_monotone_dict_is_rejected() {
    Python::attach(|py| {
        let d = PyDict::new(py);
        let inner = PyDict::new(py);
        inner.set_item("age", 1).unwrap();
        d.set_item("monotone_constraints", inner).unwrap();
        let err = make_builder(&params_from(py, &d), py).expect_err("named keys must be rejected");
        assert!(err.to_string().contains("feature NAME"), "got: {err}");
    });
}

/// PARAM-02 — the numeric range guards fire on the new params.
#[test]
fn promoted_numeric_params_are_range_checked() {
    Python::attach(|py| {
        // od_pval is a p-value: outside [0, 1] is meaningless.
        let d = PyDict::new(py);
        d.set_item("od_pval", 1.5).unwrap();
        assert!(make_builder(&params_from(py, &d), py).is_err(), "od_pval=1.5");

        // A one-leaf "tree" is a constant; the grower cannot produce it.
        let d = PyDict::new(py);
        d.set_item("max_leaves", 1).unwrap();
        assert!(make_builder(&params_from(py, &d), py).is_err(), "max_leaves=1");

        // fold_len_multiplier must exceed 1 or the dynamic tail never grows.
        let d = PyDict::new(py);
        d.set_item("fold_len_multiplier", 1.0).unwrap();
        assert!(
            make_builder(&params_from(py, &d), py).is_err(),
            "fold_len_multiplier=1.0"
        );

        // early_stopping_rounds=0 would stop at the first non-improving round.
        let d = PyDict::new(py);
        d.set_item("early_stopping_rounds", 0).unwrap();
        assert!(
            make_builder(&params_from(py, &d), py).is_err(),
            "early_stopping_rounds=0"
        );
    });
}

/// PARAM-02 — `validate_eval_set_only_params` rejects the detector / best-model
/// params on a learn-only fit, and leaves everything else alone.
#[test]
fn eval_set_only_params_are_rejected_without_an_eval_set() {
    Python::attach(|py| {
        for name in [
            "od_type",
            "od_pval",
            "od_wait",
            "early_stopping_rounds",
            "use_best_model",
        ] {
            let d = PyDict::new(py);
            d.set_item(name, 1).unwrap();
            let err = crate::params::validate_eval_set_only_params(
                py,
                &params_from(py, &d),
                crate::params::EVAL_SET_REMEDY_FIT,
            )
            .expect_err("an eval-set-only param must be rejected on a learn-only fit");
            let msg = err.value(py).to_string();
            assert!(
                msg.contains(name) && msg.contains("eval_set"),
                "the error must name the param and point at eval_set, got: {msg}"
            );
        }
        // A param that is NOT eval-set-only passes through untouched — the guard
        // must not become a blanket rejection of every kwarg.
        let d = PyDict::new(py);
        d.set_item("iterations", 10).unwrap();
        d.set_item("eval_metric", "MAE").unwrap();
        assert!(
            crate::params::validate_eval_set_only_params(
                py,
                &params_from(py, &d),
                crate::params::EVAL_SET_REMEDY_FIT,
            )
            .is_ok(),
            "only the detector / best-model params are eval-set-only"
        );
    });
}

/// The guard keys on the VALUE, not the name: passing an eval-set-only param's
/// own DISABLING value is not a request for behaviour the learn-only path cannot
/// deliver, so it must be accepted. Rejecting it blocked every wrapper / config
/// layer that materializes a full default parameter dict, and told the caller
/// their explicit *disabling* of early stopping was the problem.
#[test]
fn inert_eval_set_only_values_are_accepted_without_an_eval_set() {
    Python::attach(|py| {
        let accepts = |label: &str, d: &Bound<'_, PyDict>| {
            assert!(
                crate::params::validate_eval_set_only_params(
                    py,
                    &params_from(py, d),
                    crate::params::EVAL_SET_REMEDY_FIT,
                )
                .is_ok(),
                "`{label}` disables the parameter, so a learn-only fit loses nothing \
                 by it — it must not be rejected"
            );
        };

        let d = PyDict::new(py);
        d.set_item("od_type", "None").unwrap();
        accepts("od_type=\"None\"", &d);

        let d = PyDict::new(py);
        d.set_item("od_type", py.None()).unwrap();
        accepts("od_type=None", &d);

        let d = PyDict::new(py);
        d.set_item("od_pval", 0.0_f64).unwrap();
        accepts("od_pval=0.0", &d);

        let d = PyDict::new(py);
        d.set_item("use_best_model", false).unwrap();
        accepts("use_best_model=False", &d);

        let d = PyDict::new(py);
        d.set_item("early_stopping_rounds", py.None()).unwrap();
        accepts("early_stopping_rounds=None", &d);

        let d = PyDict::new(py);
        d.set_item("od_wait", py.None()).unwrap();
        accepts("od_wait=None", &d);

        // The whole default-dict shape a `clone` / config round-trip produces.
        let d = PyDict::new(py);
        d.set_item("iterations", 10).unwrap();
        d.set_item("od_type", "None").unwrap();
        d.set_item("od_pval", 0.0_f64).unwrap();
        d.set_item("od_wait", py.None()).unwrap();
        d.set_item("early_stopping_rounds", py.None()).unwrap();
        d.set_item("use_best_model", false).unwrap();
        accepts("a materialized full-default parameter dict", &d);
    });
}

/// The Python literal `None` is upstream's universal "not set", so every param
/// whose declared default is `None` must build a builder rather than raise a
/// PyO3 `TypeError` from `extract`. This is what `sklearn.clone(est)` /
/// `set_params(**est.get_params())` round-trips and explicit `param=None`
/// wrapper defaults produce.
#[test]
fn none_valued_params_are_treated_as_unset() {
    Python::attach(|py| {
        for name in [
            "auto_class_weights",
            "od_type",
            "eval_metric",
            "monotone_constraints",
            "early_stopping_rounds",
            "class_weights",
            "scale_pos_weight",
            "ignored_features",
            "grow_policy",
        ] {
            let d = PyDict::new(py);
            d.set_item("iterations", 10).unwrap();
            d.set_item(name, py.None()).unwrap();
            assert!(
                make_builder(&params_from(py, &d), py).is_ok(),
                "`{name}=None` means UNSET upstream and must leave the builder on its \
                 default, not raise"
            );
        }
        // ...and the shorthand/pair ambiguity check must not fire on a `None`
        // either: `od_type=None` is not an explicit od_type.
        let d = PyDict::new(py);
        d.set_item("iterations", 10).unwrap();
        d.set_item("early_stopping_rounds", 10).unwrap();
        d.set_item("od_type", py.None()).unwrap();
        d.set_item("od_wait", py.None()).unwrap();
        assert!(
            make_builder(&params_from(py, &d), py).is_ok(),
            "`early_stopping_rounds` alongside od_type=None/od_wait=None is the \
             shorthand alone, not the ambiguous both-forms combination"
        );
    });
}

// ─── FPP-16 (T16): `task_type` is VALIDATED-INFORMATIONAL ───────────────────────────────

/// Whether this test binary was compiled with a device backend feature — the same
/// compile-time question `validate_task_type` asks. Backend selection in catboost-rs is a
/// Cargo feature, so the expected outcome for `task_type="GPU"` differs per wheel and the
/// test has to split on it rather than pick one arm.
const DEVICE_FEATURE_COMPILED: bool =
    cfg!(any(feature = "wgpu", feature = "cuda", feature = "rocm"));

fn validate_one(py: Python<'_>, name: &str, value: &str) -> PyResult<()> {
    let dict = PyDict::new(py);
    dict.set_item(name, value).unwrap();
    validate_params(&params_from(py, &dict))
}

/// `task_type` is no longer rejected as a parity gap — it is an IMPLEMENTED param in the
/// VALIDATED-INFORMATIONAL sense (honoured by validating consistency, not by acting).
#[test]
fn task_type_is_no_longer_known_not_yet() {
    assert_eq!(
        status_of("task_type"),
        Some(ParamStatus::Implemented),
        "task_type must be IMPLEMENTED (validated-informational), not KnownNotYet"
    );
}

/// `task_type="CPU"` is accepted on EVERY wheel and changes nothing.
#[test]
fn task_type_cpu_is_accepted() {
    Python::attach(|py| {
        validate_one(py, "task_type", "CPU").expect("task_type=CPU must be accepted");
        // Case-insensitively, like every other string param here.
        validate_one(py, "task_type", "cpu").expect("task_type=cpu must be accepted");
    });
}

/// `task_type="GPU"` must agree with the COMPILED backend: accepted on a device wheel,
/// and an actionable error on a `cpu`-only wheel. Silently training on the CPU after an
/// explicit GPU request is the silently-wrong-model failure the honesty policy exists to
/// prevent, so the CPU-wheel arm asserts a real error, not a warning.
#[test]
fn task_type_gpu_matches_compiled_backend() {
    Python::attach(|py| {
        let result = validate_one(py, "task_type", "GPU");
        if DEVICE_FEATURE_COMPILED {
            result.expect("task_type=GPU must be accepted on a device-feature wheel");
        } else {
            let err = result.expect_err("task_type=GPU must be rejected on a cpu-only wheel");
            let msg = err.value(py).to_string();
            assert!(msg.contains("task_type"), "the message must name the parameter: {msg}");
            assert!(
                msg.contains("cuda") && msg.contains("rocm") && msg.contains("wgpu"),
                "the message must name the Cargo features that would enable it: {msg}"
            );
            assert!(
                msg.contains("compile-time"),
                "the message must explain WHY it cannot switch at runtime: {msg}"
            );
        }
    });
}

/// An unknown VALUE lists the legal values. It must NOT produce the Levenshtein
/// "unknown parameter — did you mean …?" suggestion: the parameter NAME is known and
/// perfectly spelled; only the value is wrong, and suggesting a different parameter name
/// would send the caller in the wrong direction entirely.
#[test]
fn task_type_unknown_value_is_rejected_with_legal_values() {
    Python::attach(|py| {
        let err = validate_one(py, "task_type", "TPU")
            .expect_err("task_type=TPU must be rejected");
        let msg = err.value(py).to_string();
        assert!(msg.contains("CPU") && msg.contains("GPU"), "must list the legal values: {msg}");
        assert!(
            !msg.contains("did you mean"),
            "a wrong VALUE must not be reported as a misspelled NAME: {msg}"
        );
    });
}

/// Python `None` is inert — upstream's universal "not set".
#[test]
fn task_type_none_is_inert() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("task_type", py.None()).unwrap();
        validate_params(&params_from(py, &dict)).expect("task_type=None must be inert");
    });
}

/// A non-string value is a typed error naming the legal values, never a silent accept.
#[test]
fn task_type_non_string_is_rejected() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("task_type", 1_i64).unwrap();
        let err = validate_params(&params_from(py, &dict))
            .expect_err("a non-string task_type must be rejected");
        let msg = err.value(py).to_string();
        assert!(msg.contains("task_type"), "must name the parameter: {msg}");
    });
}

/// `task_type` must not disturb the other params' validation — it rides the same loop.
#[test]
fn task_type_composes_with_other_params() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("task_type", "CPU").unwrap();
        dict.set_item("iterations", 10).unwrap();
        dict.set_item("max_depth", 3).unwrap();
        validate_params(&params_from(py, &dict)).expect("a mixed param set must validate");

        // …and a KnownNotYet param alongside it must still be rejected.
        let dict = PyDict::new(py);
        dict.set_item("task_type", "CPU").unwrap();
        dict.set_item("model_shrink_rate", 0.1).unwrap();
        validate_params(&params_from(py, &dict))
            .expect_err("a KnownNotYet param must still be rejected alongside task_type");
    });
}
