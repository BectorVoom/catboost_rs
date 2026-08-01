//! Unit tests for the canonical model split representation ([`crate::ModelSplit`]).
//!
//! SPEC-OH-08: the one-hot variant exists in the UPSTREAM value space (a raw
//! `calc_cat_feature_hash` `i32`, never a `PerfectHash` bin) and carries NO
//! float-feature identity, so the numeric-only consumers that project over
//! [`crate::ModelSplit::float_feature`] / [`crate::ModelSplit::as_float`] never
//! mistake it for a float threshold.

use crate::{ModelSplit, OneHotModelSplit};

/// SPEC-OH-08 — a one-hot split has no float identity on either projection, and
/// the variant is `Clone`/`PartialEq` like the two it joins.
#[test]
fn one_hot_split_has_no_float_identity() {
    let s = ModelSplit::OneHot(OneHotModelSplit {
        cat_feature: 1,
        value_hash: -1_438_285_038_i32,
    });

    assert!(s.float_feature().is_none(), "a one-hot split has no float-feature index");
    assert!(s.as_float().is_none(), "a one-hot split is not a float threshold split");
    assert_eq!(s, s.clone(), "the one-hot variant round-trips through Clone");
}

/// SPEC-OH-08 — the value space is the raw upstream `i32` hash, so a negative
/// hash (the common case: `calc_cat_feature_hash` fills the whole `u32` range)
/// survives construction and comparison unchanged.
#[test]
fn one_hot_value_hash_is_the_raw_signed_upstream_hash() {
    let raw: u32 = 2_856_682_258;
    let s = OneHotModelSplit {
        cat_feature: 0,
        value_hash: raw as i32,
    };
    assert!(s.value_hash < 0, "the upstream i32 space is signed");
    assert_eq!(s.value_hash as u32, raw, "the cast is bit-preserving");
}
