//! Wave-A (LOSS-09, Plan 06.4-01) split-score self-oracle for the five remaining
//! `EScoreFunction` variants: `SolarL2`, `NewtonL2`, `NewtonCosine`, `LOOL2`, `SatL2`.
//!
//! # D-6.4-06 WEAKENED-ORACLE CAVEAT (NON-NEGOTIABLE — DO NOT mistake for a strict lock)
//!
//! These five score functions are **GPU-only upstream**; catboost 1.2.10 rejects them
//! on CPU at `oblivious_tree_options.cpp:146` ("Only Cosine and L2 score functions are
//! supported for CPU."), verified live. There is therefore NO upstream-CPU training
//! ground truth for them. This file is a **transcribe-then-self-oracle** against the
//! hand-computed CUDA `score_calcers.cuh` arithmetic on chosen leaf-stat inputs —
//! it is **NOT a ≤1e-5-vs-upstream-CPU training lock** and must never be reported as one.
//! The per-leaf formulas below are transcribed verbatim from
//! `catboost/cuda/methods/kernel/score_calcers.cuh` (cited per case) and the Newton
//! dispatch `pointwise_scores.cu:484-527` (NewtonL2 reuses the L2 calcer; NewtonCosine
//! reuses the Cosine calcer — only the histogram fill, which feeds summed positive
//! `der2` into the `sum_weight` slot, differs, exercised here by constructing the
//! der2-filled `LeafStats` directly).
//!
//! All reference-side summation that mirrors a production fold routes through
//! `cb_core::sum_f64` (D-08), so the expected values are order-faithful to the
//! production `multi_dim_split_score` arms.
//!
//! Integration test under `tests/` (source/test separation, CLAUDE.md — NO inline
//! `#[cfg(test)]`).

use cb_compute::{calc_average, multi_dim_split_score, EScoreFunction, LeafStats};
use cb_core::sum_f64;

/// Tolerance: the five formulas are exact rational/log expressions, so the self-oracle
/// holds to near machine-epsilon; the plan's 1e-12 bar is generous (the LOSS-09 ≤1e-5
/// requirement is comfortably met).
const TOL: f64 = 1e-12;

/// A hand-chosen single-dimension leaf set with a spread of weights (including the
/// guard-boundary and degenerate cases) used across the first-order self-oracles.
fn sample_leaves() -> Vec<LeafStats> {
    vec![
        LeafStats {
            sum_weighted_delta: 4.0,
            sum_weight: 3.0,
        },
        LeafStats {
            sum_weighted_delta: -2.5,
            sum_weight: 5.0,
        },
        LeafStats {
            sum_weighted_delta: 1.0,
            sum_weight: 1.5,
        },
    ]
}

// ---------------------------------------------------------------------------
// SolarL2 — score_calcers.cuh:22-24
//   per-leaf term = weight > 1e-20 ? (-sum*sum) * (1 + 2*ln(weight + 1.0)) / weight : 0
// ---------------------------------------------------------------------------
#[test]
fn solar_l2_reproduces_cuda_arithmetic() {
    let leaves = sample_leaves();
    let scaled_l2 = 3.0;
    let per_dim = vec![leaves.clone()];

    // Hand-computed CUDA arithmetic, folded through sum_f64 (D-08 order-faithful).
    let terms: Vec<f64> = leaves
        .iter()
        .map(|s| {
            let sum = s.sum_weighted_delta;
            let weight = s.sum_weight;
            if weight > 1e-20 {
                (-sum * sum) * (1.0 + 2.0 * (weight + 1.0).ln()) / weight
            } else {
                0.0
            }
        })
        .collect();
    let expected = sum_f64(&terms);

    let got = multi_dim_split_score(EScoreFunction::SolarL2, &per_dim, scaled_l2);
    assert!(
        (got - expected).abs() <= TOL,
        "SolarL2 self-oracle: got {got}, expected {expected}"
    );
}

#[test]
fn solar_l2_degenerate_zero_weight_is_finite_zero() {
    // weight <= 1e-20 guard returns 0.0 (never NaN/Inf on a degenerate leaf).
    let leaves = vec![LeafStats {
        sum_weighted_delta: 0.0,
        sum_weight: 0.0,
    }];
    let per_dim = vec![leaves];
    let got = multi_dim_split_score(EScoreFunction::SolarL2, &per_dim, 3.0);
    assert!(got.is_finite(), "SolarL2 degenerate leaf must be finite");
    assert_eq!(got, 0.0, "SolarL2 weight<=1e-20 leaf must contribute 0.0");
}

// ---------------------------------------------------------------------------
// LOOL2 — score_calcers.cuh:83-87
//   adjust = weight>1 ? weight/(weight-1) : 0; adjust*=adjust;
//   weight>0 ? adjust*(-sum*sum)/weight : 0
// ---------------------------------------------------------------------------
#[test]
fn loo_l2_reproduces_cuda_arithmetic() {
    let leaves = sample_leaves();
    let scaled_l2 = 3.0;
    let per_dim = vec![leaves.clone()];

    let terms: Vec<f64> = leaves
        .iter()
        .map(|s| {
            let sum = s.sum_weighted_delta;
            let weight = s.sum_weight;
            let mut adjust = if weight > 1.0 {
                weight / (weight - 1.0)
            } else {
                0.0
            };
            adjust *= adjust;
            if weight > 0.0 {
                adjust * (-sum * sum) / weight
            } else {
                0.0
            }
        })
        .collect();
    let expected = sum_f64(&terms);

    let got = multi_dim_split_score(EScoreFunction::LOOL2, &per_dim, scaled_l2);
    assert!(
        (got - expected).abs() <= TOL,
        "LOOL2 self-oracle: got {got}, expected {expected}"
    );
}

#[test]
fn loo_l2_weight_one_boundary_adjust_is_zero() {
    // weight == 1.0 is NOT > 1.0 → adjust 0 → leaf contributes 0.0.
    let leaves = vec![LeafStats {
        sum_weighted_delta: 5.0,
        sum_weight: 1.0,
    }];
    let per_dim = vec![leaves];
    let got = multi_dim_split_score(EScoreFunction::LOOL2, &per_dim, 3.0);
    assert!(got.is_finite(), "LOOL2 boundary leaf must be finite");
    assert_eq!(got, 0.0, "LOOL2 weight==1.0 leaf adjust must be 0.0");
}

// ---------------------------------------------------------------------------
// SatL2 — score_calcers.cuh:114-117
//   adjust = weight>2 ? weight*(weight-2)/(weight*weight-3*weight+1) : 0;
//   weight>0 ? adjust*(-sum*sum)/weight : 0
// ---------------------------------------------------------------------------
#[test]
fn sat_l2_reproduces_cuda_arithmetic() {
    let leaves = sample_leaves();
    let scaled_l2 = 3.0;
    let per_dim = vec![leaves.clone()];

    let terms: Vec<f64> = leaves
        .iter()
        .map(|s| {
            let sum = s.sum_weighted_delta;
            let weight = s.sum_weight;
            let adjust = if weight > 2.0 {
                weight * (weight - 2.0) / (weight * weight - 3.0 * weight + 1.0)
            } else {
                0.0
            };
            if weight > 0.0 {
                adjust * (-sum * sum) / weight
            } else {
                0.0
            }
        })
        .collect();
    let expected = sum_f64(&terms);

    let got = multi_dim_split_score(EScoreFunction::SatL2, &per_dim, scaled_l2);
    assert!(
        (got - expected).abs() <= TOL,
        "SatL2 self-oracle: got {got}, expected {expected}"
    );
}

#[test]
fn sat_l2_weight_two_boundary_adjust_is_zero() {
    // weight == 2.0 is NOT > 2.0 → adjust 0 → leaf contributes 0.0.
    let leaves = vec![LeafStats {
        sum_weighted_delta: 5.0,
        sum_weight: 2.0,
    }];
    let per_dim = vec![leaves];
    let got = multi_dim_split_score(EScoreFunction::SatL2, &per_dim, 3.0);
    assert!(got.is_finite(), "SatL2 boundary leaf must be finite");
    assert_eq!(got, 0.0, "SatL2 weight==2.0 leaf adjust must be 0.0");
}

// ---------------------------------------------------------------------------
// NewtonL2 — pointwise_scores.cu:504-510 reuses the L2 calcer VERBATIM.
// The der2-vs-weight difference is the histogram FILL (the `sum_weight` slot is
// fed the summed positive der2 hessian), exercised here by constructing the
// der2-filled LeafStats directly: NewtonL2(stats) MUST equal L2(stats).
// ---------------------------------------------------------------------------
#[test]
fn newton_l2_reuses_l2_formula_on_der2_filled_stats() {
    // `sum_weight` carries the summed positive hessian (Σ -der2), not the count.
    let der2_filled = vec![
        LeafStats {
            sum_weighted_delta: 4.0,
            sum_weight: 3.0, // = Σ(-der2) for this leaf
        },
        LeafStats {
            sum_weighted_delta: -2.5,
            sum_weight: 5.0,
        },
        LeafStats {
            sum_weighted_delta: 1.0,
            sum_weight: 1.5,
        },
    ];
    let scaled_l2 = 3.0;
    let per_dim = vec![der2_filled];

    let newton = multi_dim_split_score(EScoreFunction::NewtonL2, &per_dim, scaled_l2);
    let l2 = multi_dim_split_score(EScoreFunction::L2, &per_dim, scaled_l2);
    assert_eq!(
        newton.to_bits(),
        l2.to_bits(),
        "NewtonL2 must reuse the L2 formula VERBATIM on der2-filled stats"
    );
}

// ---------------------------------------------------------------------------
// NewtonCosine — pointwise_scores.cu:512-521 reuses the Cosine calcer VERBATIM.
// NewtonCosine(stats) MUST equal Cosine(stats) on the same der2-filled stats.
// ---------------------------------------------------------------------------
#[test]
fn newton_cosine_reuses_cosine_formula_on_der2_filled_stats() {
    let der2_filled = vec![
        LeafStats {
            sum_weighted_delta: 4.0,
            sum_weight: 3.0,
        },
        LeafStats {
            sum_weighted_delta: -2.5,
            sum_weight: 5.0,
        },
        LeafStats {
            sum_weighted_delta: 1.0,
            sum_weight: 1.5,
        },
    ];
    let scaled_l2 = 3.0;
    let per_dim = vec![der2_filled];

    let newton = multi_dim_split_score(EScoreFunction::NewtonCosine, &per_dim, scaled_l2);
    let cosine = multi_dim_split_score(EScoreFunction::Cosine, &per_dim, scaled_l2);
    assert_eq!(
        newton.to_bits(),
        cosine.to_bits(),
        "NewtonCosine must reuse the Cosine formula VERBATIM on der2-filled stats"
    );
}

// ---------------------------------------------------------------------------
// Cross-check the SolarL2 numerator factor against calc_average so the test is
// non-vacuous: SolarL2 is NOT the plain L2 score (the ln-weighted factor differs).
// ---------------------------------------------------------------------------
#[test]
fn solar_l2_differs_from_l2_when_factor_nontrivial() {
    let leaves = sample_leaves();
    let scaled_l2 = 3.0;
    let per_dim = vec![leaves.clone()];
    let solar = multi_dim_split_score(EScoreFunction::SolarL2, &per_dim, scaled_l2);
    let l2 = multi_dim_split_score(EScoreFunction::L2, &per_dim, scaled_l2);
    // Reference: L2 uses calc_average(SWD, SW, scaled_l2)*SWD, NOT the ln factor.
    let l2_ref: Vec<f64> = leaves
        .iter()
        .map(|s| calc_average(s.sum_weighted_delta, s.sum_weight, scaled_l2) * s.sum_weighted_delta)
        .collect();
    assert!(
        (l2 - sum_f64(&l2_ref)).abs() <= TOL,
        "L2 reference sanity check"
    );
    assert!(
        (solar - l2).abs() > 1.0,
        "SolarL2 must differ materially from L2 (ln-weighted factor) — test non-vacuous"
    );
}
