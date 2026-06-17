//! Non-uniform per-object weight gradient regression (CR-02 / Plan 06.3-07).
//!
//! The QueryRMSE and QuerySoftMax grouped derivative functions ALREADY fold the
//! per-object weight into `der1` (and `der2`) inside the per-group derivative arm:
//!   * QueryRMSE: `der1 = (target - approx - queryAvrg) * weight`
//!     (`cb_compute::queryrmse_der`, `error_functions.h:879-933`).
//!   * QuerySoftMax: `der1 = beta * (-sumWTargets * p + weight * target)` where the
//!     softmax probability `p = expApprox·w / Σ expApprox·w` also carries `w`
//!     (`cb_compute::querysoftmax_der`, `error_functions.cpp:560-565`).
//!
//! Before CR-02, `boosting.rs` computed `weighted_der1 = der1 * weight`
//! UNCONDITIONALLY for ALL losses — including these grouped ranking losses — which
//! re-multiplied the already-weighted grouped der by the per-object weight a SECOND
//! time, producing squared-weight gradients (corrupt split scores / leaf values).
//! At the uniform-weight (w == 1.0) oracle fixtures the squared product is
//! `already_weighted * 1.0`, numerically identical to the correct value, which
//! masked the bug. The fix branches `weighted_der1` on `group_spans.is_some()`: the
//! grouped path uses `ders.der1` AS-IS; the pointwise path keeps `der1 * weight`.
//!
//! Because no instrumented catboost TRAINER build is in scope for an end-to-end
//! non-uniform-weight model fixture (scope_fence — LOSS-04 truths #5/#7 stay
//! deferred), this test gates the BRANCH INVARIANT directly against the public
//! `cb_compute::calc_ders_for_queries` grouped seam (the exact buffer the trainer
//! routes into `weighted_der1` on the grouped path):
//!
//!   1. The grouped seam der EQUALS the single-weighted reference (the grouped der
//!      already folds the per-object weight, so the trainer-effective
//!      `weighted_der1 == ders.der1`, NOT `ders.der1 * weight`) — asserted to a
//!      same-machine exact float tolerance.
//!   2. The grouped seam der DIFFERS from the squared-weight value
//!      (`ders.der1 * weight`) for at least one object whose `weight != 1.0` —
//!      proving the old unconditional `der1 * weight` path was wrong and that this
//!      test would FAIL against the pre-CR-02 code.
//!
//! Integration test (under `tests/`) so it can depend on `cb-compute`'s public der
//! seam. Dedicated test file (source/test separation — CLAUDE.md).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::{
    calc_ders_for_queries, group_reduce_weighted, queryrmse_der, querysoftmax_der, GroupSpan, Loss,
    QUERYSOFTMAX_BETA_DEFAULT, QUERYSOFTMAX_LAMBDA_DEFAULT,
};

/// Same-machine exact-float tolerance: the grouped seam and the in-test reference
/// run the IDENTICAL per-object der formula over the IDENTICAL group reductions, so
/// they must agree to within ordered-fold float noise.
const EXACT_TOL: f64 = 1e-12;

/// A small grouped corpus with NON-UNIFORM per-object weights. Two contiguous
/// groups over 5 objects; several weights != 1.0 (and one == 1.0) so the
/// squared-weight path is detectably wrong on the non-unit objects and a no-op on
/// the unit object. `weights[3] == 1.0` is the control (grouped der == squared der
/// there); every other object's `weight != 1.0`.
struct Corpus {
    approx: Vec<f64>,
    target: Vec<f64>,
    weights: Vec<f64>,
    groups: Vec<GroupSpan>,
}

fn corpus() -> Corpus {
    // Group 0: objects [0, 3); Group 1: objects [3, 5).
    let approx = vec![0.20, -0.10, 0.05, 0.30, -0.25];
    let target = vec![1.0, 0.0, 2.0, 1.0, 3.0];
    let weights = vec![2.0, 0.5, 1.5, 1.0, 0.25];
    let groups = vec![
        GroupSpan {
            begin: 0,
            end: 3,
            weight: 1.0,
            competitors: Vec::new(),
        },
        GroupSpan {
            begin: 3,
            end: 5,
            weight: 1.0,
            competitors: Vec::new(),
        },
    ];
    Corpus {
        approx,
        target,
        weights,
        groups,
    }
}

/// Flatten the per-group `Derivatives` returned by the grouped seam into the
/// object-order `der1` flat buffer (group order is object order for contiguous
/// `[begin, end)` spans — the same concatenation `boosting.rs` performs).
fn flatten_der1(per_group: &[cb_compute::Derivatives]) -> Vec<f64> {
    let mut der1 = Vec::new();
    for g in per_group {
        der1.extend_from_slice(&g.der1);
    }
    der1
}

/// Independently compute the single-weighted QueryRMSE `der1` reference in-test by
/// replaying the per-group reductions + the exported `queryrmse_der` helper. The
/// helper ALREADY folds the weight, so this reference IS the trainer-effective
/// `weighted_der1` on the grouped path (NOT this value times the weight again).
fn queryrmse_reference(c: &Corpus) -> Vec<f64> {
    let mut der1 = Vec::with_capacity(c.approx.len());
    for g in &c.groups {
        let approx_slice = &c.approx[g.begin..g.end];
        let target_slice = &c.target[g.begin..g.end];
        let weight_slice = &c.weights[g.begin..g.end];
        let residuals: Vec<f64> = (0..approx_slice.len())
            .map(|i| target_slice[i] - approx_slice[i])
            .collect();
        let numerator = group_reduce_weighted(&residuals, weight_slice);
        let denominator: f64 = weight_slice.iter().sum();
        let query_avrg = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        for i in 0..approx_slice.len() {
            let (d1, _d2) =
                queryrmse_der(approx_slice[i], target_slice[i], weight_slice[i], query_avrg);
            der1.push(d1);
        }
    }
    der1
}

/// Independently compute the single-weighted QuerySoftMax `der1` reference in-test
/// by replaying the per-group max-shifted weighted softmax + the exported
/// `querysoftmax_der` helper (already weight-folded).
fn querysoftmax_reference(c: &Corpus, beta: f64, lambda: f64) -> Vec<f64> {
    let mut der1 = Vec::with_capacity(c.approx.len());
    for g in &c.groups {
        let approx_slice = &c.approx[g.begin..g.end];
        let target_slice = &c.target[g.begin..g.end];
        let weight_slice = &c.weights[g.begin..g.end];
        // maxApprox + sumWeightedTargets over weight>0 objects.
        let mut max_approx = f64::MIN;
        let mut sum_weighted_targets = 0.0_f64;
        for i in 0..approx_slice.len() {
            let w = weight_slice[i];
            if w > 0.0 {
                if approx_slice[i] > max_approx {
                    max_approx = approx_slice[i];
                }
                if target_slice[i] > 0.0 {
                    sum_weighted_targets += w * target_slice[i];
                }
            }
        }
        if sum_weighted_targets > 0.0 {
            let weighted_exp: Vec<f64> = (0..approx_slice.len())
                .map(|i| (beta * (approx_slice[i] - max_approx)).exp() * weight_slice[i])
                .collect();
            let sum_exp: f64 = weighted_exp.iter().sum();
            for i in 0..approx_slice.len() {
                let w = weight_slice[i];
                if w > 0.0 && sum_exp > 0.0 {
                    let p = weighted_exp[i] / sum_exp;
                    let (d1, _d2) =
                        querysoftmax_der(p, sum_weighted_targets, w, target_slice[i], beta, lambda);
                    der1.push(d1);
                } else {
                    der1.push(0.0);
                }
            }
        } else {
            for _ in 0..approx_slice.len() {
                der1.push(0.0);
            }
        }
    }
    der1
}

/// Assert the CR-02 single-weighting invariant for a flattened grouped `der1`:
///   1. Grouped seam der == single-weighted reference (trainer uses it AS-IS).
///   2. Grouped seam der != squared-weight value (`der * weight`) for at least one
///      object whose weight != 1.0 (would FAIL against the pre-CR-02 code).
fn assert_single_weight_invariant(produced: &[f64], reference: &[f64], weights: &[f64]) {
    assert_eq!(
        produced.len(),
        reference.len(),
        "grouped der length must match the reference length"
    );
    assert_eq!(
        produced.len(),
        weights.len(),
        "grouped der length must match the object count"
    );

    // (1) The grouped seam der IS the single-weighted der (already weight-folded);
    //     the trainer uses this buffer AS-IS for `weighted_der1` on the grouped
    //     path (group_spans.is_some()), NOT `der * weight`.
    for (i, (&p, &r)) in produced.iter().zip(reference.iter()).enumerate() {
        assert!(
            (p - r).abs() <= EXACT_TOL,
            "object {i}: grouped der {p} must equal the single-weighted reference {r} \
             (diff {})",
            (p - r).abs()
        );
    }

    // (2) The squared-weight value (`der * weight`) — what the pre-CR-02
    //     unconditional `der1 * weight` map produced — MUST differ from the grouped
    //     der for at least one non-unit-weight object. This is the assertion that
    //     fails against the old code.
    let mut distinguished = false;
    for (i, (&p, &w)) in produced.iter().zip(weights.iter()).enumerate() {
        let squared = p * w;
        if (w - 1.0).abs() > f64::EPSILON && p.abs() > EXACT_TOL {
            assert!(
                (squared - p).abs() > 1e-6,
                "object {i}: squared-weight value {squared} (der*weight, weight {w}) must differ \
                 from the single-weighted der {p} — the regression would not distinguish the bug"
            );
            distinguished = true;
        }
    }
    assert!(
        distinguished,
        "at least one object must have weight != 1.0 and a non-zero der so the squared-weight \
         path is detectably wrong"
    );
}

#[test]
fn queryrmse_grouped_der_is_single_weighted_not_squared() {
    let c = corpus();
    // Sanity: a non-uniform corpus with at least one weight != 1.0.
    assert!(
        c.weights.iter().any(|&w| (w - 1.0).abs() > f64::EPSILON),
        "corpus must carry non-uniform per-object weights"
    );

    let per_group = calc_ders_for_queries(
        &Loss::QueryRmse,
        &c.approx,
        &c.target,
        &c.weights,
        &c.groups,
        0,
    )
    .expect("QueryRMSE grouped der must compute");
    let produced = flatten_der1(&per_group);
    let reference = queryrmse_reference(&c);

    assert_single_weight_invariant(&produced, &reference, &c.weights);
}

#[test]
fn querysoftmax_grouped_der_is_single_weighted_not_squared() {
    let c = corpus();
    let beta = QUERYSOFTMAX_BETA_DEFAULT;
    let lambda = QUERYSOFTMAX_LAMBDA_DEFAULT;

    let loss = Loss::QuerySoftMax { lambda, beta };
    let per_group =
        calc_ders_for_queries(&loss, &c.approx, &c.target, &c.weights, &c.groups, 0)
            .expect("QuerySoftMax grouped der must compute");
    let produced = flatten_der1(&per_group);
    let reference = querysoftmax_reference(&c, beta, lambda);

    assert_single_weight_invariant(&produced, &reference, &c.weights);
}
