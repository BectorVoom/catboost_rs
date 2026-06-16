//! Unit tests for the grouped der seam ([`super::calc_ders_for_queries`],
//! [`super::group_reduce_weighted`], and the [`Runtime::compute_gradients_grouped`]
//! default) — LOSS-04, D-6.3-03.
//!
//! Source/test separation (INFRA-06): dedicated file, linked from `ranking_der.rs`
//! via the `#[path]` footer — never inline.

use super::{calc_ders_for_queries, group_reduce_weighted, Competitor, GroupSpan};
use crate::runtime::{Derivatives, Loss, Runtime};
use cb_core::{sum_f64, CbResult};

fn span(begin: usize, end: usize) -> GroupSpan {
    GroupSpan {
        begin,
        end,
        weight: 1.0,
        competitors: vec![Vec::new(); end - begin],
    }
}

#[test]
fn group_reduce_weighted_matches_hand_sum_f64_uniform() {
    let slice = [1.0, 2.0, 3.0, 4.0];
    // Uniform weights → plain sum.
    let got = group_reduce_weighted(&slice, &[]);
    let want = sum_f64(&[1.0, 2.0, 3.0, 4.0]);
    assert!((got - want).abs() < 1e-12);
    assert!((got - 10.0).abs() < 1e-12);
}

#[test]
fn group_reduce_weighted_matches_hand_sum_f64_weighted() {
    let slice = [1.0, 2.0, 3.0];
    let weights = [0.5, 2.0, 1.0];
    let got = group_reduce_weighted(&slice, &weights);
    // 1*0.5 + 2*2.0 + 3*1.0 = 7.5, reduced through sum_f64.
    let want = sum_f64(&[0.5, 4.0, 3.0]);
    assert!((got - want).abs() < 1e-12);
    assert!((got - 7.5).abs() < 1e-12);
}

#[test]
fn empty_groups_yields_empty_ders_no_panic() {
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0, 0.0];
    let out = calc_ders_for_queries(&Loss::Rmse, &approx, &target, &[], &[], 0).unwrap();
    assert!(out.is_empty());
}

#[test]
fn group_span_out_of_range_is_degenerate_error() {
    let approx = [0.1, 0.2];
    let target = [0.0, 1.0];
    // Group claims [0, 5) but only 2 objects exist.
    let groups = [span(0, 5)];
    let err = calc_ders_for_queries(&Loss::Rmse, &approx, &target, &[], &groups, 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::Degenerate(_)));
}

#[test]
fn approx_target_length_mismatch_is_degenerate_error() {
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0];
    let err = calc_ders_for_queries(&Loss::Rmse, &approx, &target, &[], &[], 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::Degenerate(_)));
}

#[test]
fn unwired_loss_variant_returns_typed_error() {
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0, 0.0];
    let groups = [span(0, 3)];
    // Every loss arm is unwired in Plan 06.3-01 → typed OutOfRange error.
    let err =
        calc_ders_for_queries(&Loss::Logloss, &approx, &target, &[], &groups, 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));
}

#[test]
fn per_group_slicing_validates_each_span() {
    // Two valid groups spanning [0,2) and [2,4): the seam slices each before
    // dispatch. With an unwired loss the first group's dispatch surfaces the
    // typed error — proving the slice bounds passed (no Degenerate span error).
    let approx = [0.1, 0.2, 0.3, 0.4];
    let target = [0.0, 1.0, 0.0, 1.0];
    let groups = [span(0, 2), span(2, 4)];
    let err =
        calc_ders_for_queries(&Loss::Logloss, &approx, &target, &[], &groups, 0).unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));
}

#[test]
fn competitor_round_trips_in_group_span() {
    // Sanity: a GroupSpan carrying competitor edges is well-formed and slices.
    let mut g = span(0, 3);
    g.competitors[0].push(Competitor {
        id: 1,
        weight: 1.0,
    });
    assert_eq!(g.size(), 3);
    assert_eq!(g.competitors[0][0].id, 1);
}

/// Minimal `Runtime` exercising ONLY the grouped default — `compute_gradients`
/// is unused here, so it returns a degenerate error.
struct DummyRuntime;

impl Runtime for DummyRuntime {
    fn compute_gradients(
        &self,
        _loss: &Loss,
        _approx: &[f64],
        _target: &[f64],
        _approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        Err(cb_core::CbError::Degenerate("unused in this test".to_owned()))
    }
}

#[test]
fn runtime_compute_gradients_grouped_default_delegates_to_seam() {
    let rt = DummyRuntime;
    let approx = [0.1, 0.2, 0.3];
    let target = [0.0, 1.0, 0.0];
    let groups = [span(0, 3)];
    // The default impl delegates to calc_ders_for_queries → unwired typed error.
    let err = rt
        .compute_gradients_grouped(&Loss::Logloss, &approx, &target, &[], &groups, 0)
        .unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));

    // Empty groups → empty der set through the trait default (no panic).
    let out = rt
        .compute_gradients_grouped(&Loss::Rmse, &approx, &target, &[], &[], 0)
        .unwrap();
    assert!(out.is_empty());
}
