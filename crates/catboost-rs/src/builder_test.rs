//! Unit tests for [`CatBoostBuilder`]'s categorical/CTR setters (F01–F06).
//!
//! Mounted as a **CHILD** module of `builder.rs` (`#[path]`), not a sibling in
//! `lib.rs`: `boost_params()` is a PRIVATE `fn` (`builder.rs`), so only a child
//! module can see it. Precedent: `crates/catboost-rs/src/cv.rs`'s `cv_test`
//! mount.

use super::CatBoostBuilder;
use cb_train::{
    combinations_ctr_default, combinations_ctr_priors_default, counter_calc_method_default,
    max_ctr_complexity_default, one_hot_max_size_default, simple_ctr_default,
    simple_ctr_priors_default, CounterCalcMethod, ECtrType,
};

/// F01 — `one_hot_max_size` is settable and reaches `BoostParams`.
#[test]
fn one_hot_max_size_setter_reaches_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(
        default_params.one_hot_max_size,
        one_hot_max_size_default(),
        "the unset builder must still emit the upstream default"
    );

    let params = CatBoostBuilder::new().one_hot_max_size(7).boost_params();
    assert_eq!(params.one_hot_max_size, 7);
}

/// F02 — `max_ctr_complexity` is settable and reaches `BoostParams`.
#[test]
fn max_ctr_complexity_setter_reaches_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(
        default_params.max_ctr_complexity,
        max_ctr_complexity_default()
    );

    let params = CatBoostBuilder::new().max_ctr_complexity(1).boost_params();
    assert_eq!(params.max_ctr_complexity, 1);
}

/// F03 — `simple_ctr` and `simple_ctr_priors` are settable in lockstep.
#[test]
fn simple_ctr_and_priors_setters_reach_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(default_params.simple_ctr, simple_ctr_default());
    assert_eq!(default_params.simple_ctr_priors, simple_ctr_priors_default());

    let params = CatBoostBuilder::new()
        .simple_ctr(ECtrType::Counter)
        .simple_ctr_priors(vec![0.0, 0.5, 1.0])
        .boost_params();
    assert_eq!(params.simple_ctr, ECtrType::Counter);
    assert_eq!(params.simple_ctr_priors, vec![0.0, 0.5, 1.0]);
}

/// F04 — `combinations_ctr` and `combinations_ctr_priors` are settable in
/// lockstep, and are INDEPENDENT of the simple-CTR pair (a cross-wire between
/// the two families is exactly what this asserts against).
#[test]
fn combinations_ctr_and_priors_setters_reach_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(default_params.combinations_ctr, combinations_ctr_default());
    assert_eq!(
        default_params.combinations_ctr_priors,
        combinations_ctr_priors_default()
    );

    let params = CatBoostBuilder::new()
        .simple_ctr(ECtrType::Buckets)
        .simple_ctr_priors(vec![0.25])
        .combinations_ctr(ECtrType::BinarizedTargetMeanValue)
        .combinations_ctr_priors(vec![0.75, 1.5])
        .boost_params();
    assert_eq!(params.combinations_ctr, ECtrType::BinarizedTargetMeanValue);
    assert_eq!(params.combinations_ctr_priors, vec![0.75, 1.5]);
    // The simple pair must NOT have been overwritten by the combination pair.
    assert_eq!(params.simple_ctr, ECtrType::Buckets);
    assert_eq!(params.simple_ctr_priors, vec![0.25]);
}

/// F05 — `counter_calc_method` is settable and reaches `BoostParams`.
#[test]
fn counter_calc_method_setter_reaches_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(
        default_params.counter_calc_method,
        counter_calc_method_default()
    );

    let params = CatBoostBuilder::new()
        .counter_calc_method(CounterCalcMethod::Full)
        .boost_params();
    assert_eq!(params.counter_calc_method, CounterCalcMethod::Full);
}

/// F06 — the default-equivalence guard: an untouched builder's `boost_params()`
/// must equal the canonical upstream defaults FIELD BY FIELD for every one of
/// the seven newly-promoted fields.
///
/// This is the guard the two mandated mutations (§3.1) must break:
///  1. a WRITE cross-wire — `new()` seeded with a non-default value;
///  2. a READ cross-wire — `boost_params()` reading the wrong `self` field
///     (e.g. `one_hot_max_size: self.max_ctr_complexity as u32`). Mutation 1
///     alone cannot detect this, because `simple_ctr_default()` and
///     `combinations_ctr_default()` both return `Borders`.
#[test]
fn untouched_builder_emits_the_canonical_ctr_defaults() {
    let p = CatBoostBuilder::new().boost_params();
    assert_eq!(p.one_hot_max_size, one_hot_max_size_default());
    assert_eq!(p.max_ctr_complexity, max_ctr_complexity_default());
    assert_eq!(p.simple_ctr, simple_ctr_default());
    assert_eq!(p.simple_ctr_priors, simple_ctr_priors_default());
    assert_eq!(p.combinations_ctr, combinations_ctr_default());
    assert_eq!(p.combinations_ctr_priors, combinations_ctr_priors_default());
    assert_eq!(p.counter_calc_method, counter_calc_method_default());
}

/// F06 (read-cross-wire detector). `one_hot_max_size` and `max_ctr_complexity`
/// are DISTINCT fields with DISTINCT defaults (2 vs 4), so a `boost_params()`
/// that reads one where it means the other is observable. Setting only one and
/// asserting the other is unmoved is what makes mutation 2 fail.
#[test]
fn one_hot_max_size_and_max_ctr_complexity_do_not_cross_wire() {
    let p = CatBoostBuilder::new().one_hot_max_size(9).boost_params();
    assert_eq!(p.one_hot_max_size, 9);
    assert_eq!(
        p.max_ctr_complexity,
        max_ctr_complexity_default(),
        "setting one_hot_max_size must not move max_ctr_complexity"
    );

    let p = CatBoostBuilder::new().max_ctr_complexity(3).boost_params();
    assert_eq!(p.max_ctr_complexity, 3);
    assert_eq!(
        p.one_hot_max_size,
        one_hot_max_size_default(),
        "setting max_ctr_complexity must not move one_hot_max_size"
    );
}
