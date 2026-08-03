//! F07 — the CTR knob types are reachable from the published crate alone.
//!
//! This is an INTEGRATION target, so it compiles as a SEPARATE crate: it can
//! only name what `catboost_rs` actually re-exports. If `ECtrType` or
//! `CounterCalcMethod` were reachable only via `cb_train`, an external caller
//! could name `CatBoostBuilder::simple_ctr` but not construct its argument.

use catboost_rs::{CatBoostBuilder, CounterCalcMethod, ECtrType};

#[test]
fn ctr_knob_types_are_nameable_from_the_published_crate() {
    // Naming each variant is the assertion: this file does not import
    // `cb_train` at all, so it compiles only if the re-exports exist.
    let types = [
        ECtrType::Borders,
        ECtrType::Buckets,
        ECtrType::BinarizedTargetMeanValue,
        ECtrType::Counter,
    ];
    assert_eq!(types.len(), 4);
    assert_ne!(ECtrType::Borders, ECtrType::Counter);
    assert_ne!(CounterCalcMethod::Full, CounterCalcMethod::SkipTest);
}

#[test]
fn a_categorical_run_is_configurable_through_the_published_crate_alone() {
    // The whole point of F07: every argument below is named without reaching
    // into `cb_train`.
    let _builder = CatBoostBuilder::new()
        .iterations(3)
        .depth(3)
        .one_hot_max_size(2)
        .max_ctr_complexity(1)
        .simple_ctr(ECtrType::Borders)
        .simple_ctr_priors(vec![0.5])
        .combinations_ctr(ECtrType::Counter)
        .combinations_ctr_priors(vec![0.0, 0.5])
        .counter_calc_method(CounterCalcMethod::Full);
}
