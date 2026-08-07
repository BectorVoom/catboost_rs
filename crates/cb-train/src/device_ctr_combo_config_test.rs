//! FPP-09 (T07) unit self-oracle: COMBINATION (tensor) CTR projections are device-covered
//! and every projection member reaches the device as its own `member_bins` entry.
//!
//! Mounted as a sibling `#[path]` submodule of `boosting` (source/test separation,
//! CLAUDE.md; the `boosting_device_fold_test.rs` precedent), so it reaches the private
//! `super::{build_device_ctr_config, ctr_types_are_device_covered}` directly. PLAN blocker
//! **B-2 is resolved here the same way as T06 — option (a)-equivalent**: the function is
//! already free-standing, so it is tested in place rather than widened to `pub(crate)` for
//! an integration test that would then also need a device-free construction path.
//!
//! # The discriminating assertion
//!
//! Before this task, `build_device_ctr_config` extracted only
//! `projection.cat_features().first()` and emitted `member_bins: vec![member]`. A 2-member
//! combination therefore arrived at the device as ONE member's raw bucket column, and the
//! device would have scored a combination split from the wrong bins — WRONG, not merely
//! worse. `combination_projection_carries_every_member` is the test that fails on that.

use cb_compute::DeviceCtrAveraging;

use super::{build_device_ctr_config, ctr_types_are_device_covered};
use crate::ctr::{CtrFeatureColumn, ECtrType};
use crate::TProjection;

const N: usize = 8;
const CTR_BORDER_COUNT: usize = 15;

/// A device-covered CTR column over `projection`: Borders, target border 0, unit prior
/// denominator. `bins`/`ctr_value` are not read by `build_device_ctr_config` (it rebuilds
/// the border table from the prior), so they stay minimal.
fn covered_column(projection: TProjection, bucket_count: usize) -> CtrFeatureColumn {
    CtrFeatureColumn {
        projection,
        ctr_type: ECtrType::Borders.as_i8(),
        target_border_idx: 0,
        prior_num: 0.5,
        prior_denom: 1.0,
        bins: vec![0; N],
        ctr_value: vec![0.0; N],
        bucket_count,
    }
}

/// Raw bucket columns for three CTR-eligible cat features, each DISTINCT so a builder that
/// repeated member 0 is detectable.
fn buckets() -> Vec<Vec<u32>> {
    vec![
        (0..N).map(|i| (i % 2) as u32).collect(),
        (0..N).map(|i| (i % 3) as u32).collect(),
        (0..N).map(|i| (i % 4) as u32).collect(),
    ]
}

fn build(cols: &[CtrFeatureColumn]) -> cb_compute::DeviceCtrConfig {
    let perm: Vec<i32> = (0..N as i32).collect();
    let eligible_absolute = vec![0usize, 1, 2];
    build_device_ctr_config(
        cols,
        cols,
        &perm,
        &perm,
        &vec![0usize; N],
        &buckets(),
        &eligible_absolute,
        CTR_BORDER_COUNT,
    )
    .expect("a device-covered CTR column set must build a config")
}

#[test]
fn combination_projection_carries_every_member() {
    let projection = TProjection::from_features(&[0, 2]);
    assert!(projection.is_combination(), "the fixture must be a real 2-member projection");
    let cfg = build(&[covered_column(projection, 8)]);

    let col = cfg
        .columns
        .first()
        .expect("one structure column per input column");
    assert_eq!(
        col.member_bins.len(),
        2,
        "a 2-member combination projection must contribute TWO member_bins entries; \
         got {} — the device would score the combination split from one member's bins",
        col.member_bins.len()
    );
    assert_ne!(
        col.member_bins[0], col.member_bins[1],
        "the two members must be DISTINCT raw bucket columns, not a repeat of the first"
    );

    // Members arrive in projection-sorted order — the SAME order `combined_hash` folds in,
    // which is what makes the device's combined bin identical to the CPU's (PLAN V-4).
    let expected = buckets();
    assert_eq!(col.member_bins[0], expected[0], "member 0 == cat feature 0's buckets");
    assert_eq!(col.member_bins[1], expected[2], "member 1 == cat feature 2's buckets");
}

#[test]
fn simple_projection_is_byte_unchanged() {
    // D-04 regression: a simple projection must produce exactly what the pre-change
    // single-member extraction produced.
    let cfg = build(&[covered_column(TProjection::from_features(&[1]), 3)]);
    let col = cfg.columns.first().expect("one column");
    assert_eq!(col.member_bins.len(), 1, "a simple projection stays single-member");
    assert_eq!(col.member_bins[0], buckets()[1], "the single member is cat feature 1");
    assert_eq!(col.bucket_count, 3);
    assert!((col.prior - 0.5).abs() < f64::EPSILON, "prior = prior_num / prior_denom");
    assert_eq!(
        col.borders.len(),
        CTR_BORDER_COUNT,
        "the border table is unchanged for a simple projection"
    );
}

#[test]
fn a_combination_column_set_is_NOT_device_covered_yet() {
    // ESCALATED (FPP-11). The column BUILDER handles combinations correctly — that is
    // what the tests above pin — but the end-to-end oracle over `ctr_device_combo/` misses
    // the ≤1e-5 bar by 3.3e-2 while the CPU path is exact at 1.4e-17, so
    // `ctr_types_are_device_covered` still rejects non-simple projections and the fit
    // takes the correct CPU grower.
    //
    // This test pins the CLOSED gate deliberately: re-opening it must be a conscious act
    // accompanied by a passing `device_ctr_combo_fit_test` (currently `#[ignore]`d), not
    // an accident. The evidence and localisation live on `ctr_types_are_device_covered`.
    let cols = vec![
        covered_column(TProjection::from_features(&[0]), 2),
        covered_column(TProjection::from_features(&[0, 1]), 6),
    ];
    assert!(
        !ctr_types_are_device_covered(&cols),
        "a set containing a COMBINATION projection must still decline to the CPU path \
         until the device combination-CTR e2e gap is closed"
    );

    // …while an all-SIMPLE Borders set stays covered (D-04: the shipped CTR arm is
    // untouched by the escalation).
    let simple = vec![
        covered_column(TProjection::from_features(&[0]), 2),
        covered_column(TProjection::from_features(&[1]), 3),
    ];
    assert!(
        ctr_types_are_device_covered(&simple),
        "an all-simple Borders set must remain device-covered"
    );
}

#[test]
fn a_non_borders_column_still_declines() {
    // Track U (Buckets / BinarizedTargetMeanValue / Counter) is NOT this task. Relaxing
    // the projection conjunct must not accidentally relax the ctr_type one.
    let mut cols = vec![covered_column(TProjection::from_features(&[0, 1]), 6)];
    cols[0].ctr_type = ECtrType::Counter.as_i8();
    assert!(
        !ctr_types_are_device_covered(&cols),
        "a Counter CTR must still decline to the CPU path"
    );

    let mut cols = vec![covered_column(TProjection::from_features(&[0, 1]), 6)];
    cols[0].prior_denom = 2.0;
    assert!(
        !ctr_types_are_device_covered(&cols),
        "a non-unit prior denominator must still decline"
    );

    let mut cols = vec![covered_column(TProjection::from_features(&[0, 1]), 6)];
    cols[0].target_border_idx = 1;
    assert!(
        !ctr_types_are_device_covered(&cols),
        "a multi-target-border column must still decline"
    );
}

#[test]
fn an_unknown_projection_member_is_a_typed_error() {
    // The per-member `eligible_absolute` lookup and its typed error must survive the
    // rewrite — a combination naming a non-CTR-eligible feature is a caller bug, not a
    // silent drop.
    let perm: Vec<i32> = (0..N as i32).collect();
    let cols = vec![covered_column(TProjection::from_features(&[0, 9]), 6)];
    let err = build_device_ctr_config(
        &cols,
        &cols,
        &perm,
        &perm,
        &vec![0usize; N],
        &buckets(),
        &[0usize, 1, 2],
        CTR_BORDER_COUNT,
    )
    .expect_err("member 9 is not CTR-eligible");
    assert!(
        format!("{err}").contains('9'),
        "the typed error must name the offending member, got: {err}"
    );
}

#[test]
fn averaging_columns_get_the_same_member_treatment() {
    // The averaging permutation's columns run through the SAME closure; a fix applied to
    // only one call would leave the leaf-value gather reading one member's bins.
    let cfg = build(&[covered_column(TProjection::from_features(&[0, 2]), 8)]);
    let averaging: &DeviceCtrAveraging = cfg
        .averaging
        .as_ref()
        .expect("a covered CTR fit always populates the averaging arm");
    let col = averaging
        .columns
        .first()
        .expect("one averaging column per input column");
    assert_eq!(
        col.member_bins.len(),
        2,
        "the averaging arm must carry both members too"
    );
}
