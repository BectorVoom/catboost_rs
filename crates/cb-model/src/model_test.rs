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

// ---------------------------------------------------------------------------
// F08 (SPEC-CATF-Δ4) — the model carries the DECLARED trained categorical width
// ---------------------------------------------------------------------------

use crate::{Model, ObliviousTree};

/// A minimal one-tree, one-float-feature model to hang the width tests on.
fn tiny_model() -> Model {
    Model::new(
        vec![ObliviousTree {
            splits: vec![ModelSplit::Float(crate::Split {
                feature: 0,
                border: 0.5,
            })],
            leaf_values: vec![-1.0, 1.0],
            leaf_weights: vec![2.0, 3.0],
        }],
        0.25,
        vec![vec![0.5]],
    )
}

/// F08 test fn 1 — the width is readable and defaults to zero.
///
/// It must be the pool's DECLARED cat width, never a width derived from the
/// splits the model happened to choose: `max(projection member) + 1` equals the
/// true training width only if the highest-indexed cat column is both
/// CTR-eligible AND chosen by some split (PLAN-CHECK CRITICAL-3).
#[test]
fn with_cat_feature_count_is_readable_and_defaults_to_zero() {
    let m = tiny_model();
    assert_eq!(
        m.cat_feature_count(),
        0,
        "a model built without a categorical pool declares zero cat columns"
    );

    let m = m.with_cat_feature_count(3);
    assert_eq!(m.cat_feature_count(), 3);
}

/// F08 test fn 2 — THE BYTE-IDENTITY GUARD. `cat_feature_count` is runtime-only:
/// neither codec writes or reads it, so E00's frozen non-mean CTR baseline and
/// `float_only_byte_identity`'s baseline stay valid.
#[test]
fn adding_the_cat_feature_count_does_not_change_cbm_bytes() {
    let dir = std::env::temp_dir().join(format!(
        "cb_model_f08_bytes_{}_{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");

    let zero_path = dir.join("zero.cbm");
    let seven_path = dir.join("seven.cbm");
    crate::save_cbm(&tiny_model().with_cat_feature_count(0), &zero_path).expect("save zero");
    crate::save_cbm(&tiny_model().with_cat_feature_count(7), &seven_path).expect("save seven");

    let zero = std::fs::read(&zero_path).expect("read zero");
    let seven = std::fs::read(&seven_path).expect("read seven");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        zero, seven,
        "cat_feature_count is runtime-only and MUST NOT reach the .cbm bytes — \
         otherwise every frozen byte-identity baseline in the repo is invalidated"
    );
}
