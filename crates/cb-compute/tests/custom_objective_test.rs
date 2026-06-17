//! LOSS-07 self-oracle (trait-contract half): a built-in loss reimplemented AS a
//! Rust [`cb_compute::CustomObjective`] / [`cb_compute::CustomMetric`] must
//! reproduce the in-tree built-in math (`logloss_der1`/`logloss_der2`,
//! `rmse_der1`/`rmse_der2`, the Logloss metric formula) bit-faithfully (<= 1e-5,
//! in practice 1e-12). This proves the trait CONTRACT and the `Arc<dyn>` handle
//! mechanism are faithful using ONLY `cb-compute`'s dependency budget (D-03 —
//! `cb-compute` is `cubecl`-free and cannot reach the `cb-backend` train loop;
//! the END-TO-END training-dispatch self-oracle against the shipped built-in
//! oracle lives in `cb-train/tests/custom_objective_oracle_test.rs`).
//!
//! Integration test (under `tests/`) so the `#[cfg(test)]` source/test
//! separation rule (CLAUDE.md) is honored — no inline `mod tests`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::sync::Arc;

use cb_compute::{
    logloss_der1, logloss_der2, rmse_der1, rmse_der2, sigmoid, CustomMetric, CustomMetricHandle,
    CustomObjective, CustomObjectiveHandle, Loss,
};
use cb_core::{CbError, CbResult};

/// Logloss reimplemented as a custom objective. der1 = `target -
/// sigmoid(approx)`, der2 = `-p*(1-p)` (the stronger der2 test — a non-trivial
/// hessian). Mirrors the catboost Python `calc_ders_range` shape.
struct LoglossCustom;

impl CustomObjective for LoglossCustom {
    fn calc_ders_range(
        &self,
        approxes: &[f64],
        targets: &[f64],
        weights: &[f64],
        ders: &mut [(f64, f64)],
    ) -> CbResult<()> {
        if approxes.len() != targets.len() || approxes.len() != ders.len() {
            return Err(CbError::Degenerate(
                "LoglossCustom: approx/target/ders length mismatch".to_owned(),
            ));
        }
        if !weights.is_empty() && weights.len() != approxes.len() {
            return Err(CbError::Degenerate(
                "LoglossCustom: weights length mismatch".to_owned(),
            ));
        }
        for (i, ((&a, &t), d)) in approxes.iter().zip(targets.iter()).zip(ders.iter_mut()).enumerate()
        {
            let p = sigmoid(a);
            let der1 = t - p;
            let der2 = -p * (1.0 - p);
            let w = if weights.is_empty() { 1.0 } else { weights[i] };
            *d = (w * der1, w * der2);
        }
        Ok(())
    }
}

/// RMSE reimplemented as a custom objective. der1 = `target - approx`, der2 =
/// `-1`.
struct RmseCustom;

impl CustomObjective for RmseCustom {
    fn calc_ders_range(
        &self,
        approxes: &[f64],
        targets: &[f64],
        _weights: &[f64],
        ders: &mut [(f64, f64)],
    ) -> CbResult<()> {
        if approxes.len() != targets.len() || approxes.len() != ders.len() {
            return Err(CbError::Degenerate(
                "RmseCustom: approx/target/ders length mismatch".to_owned(),
            ));
        }
        for (((&a, &t), d), _) in approxes
            .iter()
            .zip(targets.iter())
            .zip(ders.iter_mut())
            .zip(0..)
        {
            *d = (t - a, -1.0);
        }
        Ok(())
    }
}

/// Logloss reimplemented as a custom metric: `evaluate` accumulates the weighted
/// cross-entropy numerator and the weight denominator; `get_final_error` divides;
/// smaller is better.
struct LoglossMetric;

impl CustomMetric for LoglossMetric {
    fn evaluate(&self, approxes: &[f64], target: &[f64], weight: &[f64]) -> CbResult<(f64, f64)> {
        if approxes.len() != target.len() {
            return Err(CbError::Degenerate(
                "LoglossMetric: approx/target length mismatch".to_owned(),
            ));
        }
        let mut err = 0.0;
        let mut wsum = 0.0;
        for (i, (&a, &y)) in approxes.iter().zip(target.iter()).enumerate() {
            let w = if weight.is_empty() { 1.0 } else { weight[i] };
            let p = sigmoid(a).clamp(1e-15, 1.0 - 1e-15);
            err += w * -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            wsum += w;
        }
        Ok((err, wsum))
    }

    fn get_final_error(&self, error: f64, weight: f64) -> f64 {
        error / weight
    }

    fn is_max_optimal(&self) -> bool {
        false
    }
}

const TOL: f64 = 1e-5;

/// The custom Logloss objective reproduces the in-tree `logloss_der1` /
/// `logloss_der2` per object (the trait dispatch faithfully carries the loss
/// math). Tolerance 1e-5 (the LOSS parity bar); actual divergence is 0.
#[test]
fn logloss_custom_matches_builtin_ders() {
    let approxes = [-2.0_f64, -0.5, 0.0, 0.5, 1.0, 3.0, -1.3, 2.7];
    let targets = [0.0_f64, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let mut ders = vec![(0.0_f64, 0.0_f64); approxes.len()];

    LoglossCustom
        .calc_ders_range(&approxes, &targets, &[], &mut ders)
        .unwrap();

    for (i, &(d1, d2)) in ders.iter().enumerate() {
        let want1 = logloss_der1(approxes[i], targets[i]);
        let want2 = logloss_der2(approxes[i], targets[i]);
        assert!(
            (d1 - want1).abs() <= TOL,
            "object {i}: custom der1 {d1} vs builtin {want1}"
        );
        assert!(
            (d2 - want2).abs() <= TOL,
            "object {i}: custom der2 {d2} vs builtin {want2}"
        );
    }
}

/// The custom RMSE objective reproduces the in-tree `rmse_der1` / `rmse_der2`.
#[test]
fn rmse_custom_matches_builtin_ders() {
    let approxes = [-2.0_f64, -0.5, 0.0, 0.5, 1.0, 3.0];
    let targets = [1.0_f64, 1.5, -0.2, 0.5, 0.0, 2.0];
    let mut ders = vec![(0.0_f64, 0.0_f64); approxes.len()];

    RmseCustom
        .calc_ders_range(&approxes, &targets, &[], &mut ders)
        .unwrap();

    for (i, &(d1, d2)) in ders.iter().enumerate() {
        let want1 = rmse_der1(approxes[i], targets[i]);
        let want2 = rmse_der2(approxes[i], targets[i]);
        assert!((d1 - want1).abs() <= TOL, "object {i}: der1 {d1} vs {want1}");
        assert!((d2 - want2).abs() <= TOL, "object {i}: der2 {d2} vs {want2}");
    }
}

/// The custom Logloss metric reproduces the weighted cross-entropy the built-in
/// `EvalMetric::Logloss` computes: `get_final_error(evaluate)` equals
/// `sum_w -(y ln p + (1-y) ln(1-p)) / sum_w`.
#[test]
fn logloss_custom_metric_matches_formula() {
    let approxes = [-2.0_f64, -0.5, 0.0, 0.5, 1.0, 3.0];
    let target = [0.0_f64, 1.0, 0.0, 1.0, 1.0, 0.0];

    let (err, w) = LoglossMetric.evaluate(&approxes, &target, &[]).unwrap();
    let got = LoglossMetric.get_final_error(err, w);

    // Independent recomputation of the same formula.
    let mut num = 0.0;
    for (&a, &y) in approxes.iter().zip(target.iter()) {
        let p = sigmoid(a).clamp(1e-15, 1.0 - 1e-15);
        num += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }
    let want = num / (approxes.len() as f64);
    assert!((got - want).abs() <= TOL, "custom metric {got} vs formula {want}");
    assert!(!LoglossMetric.is_max_optimal(), "Logloss: smaller is better");
}

/// The `Arc<dyn>` handle is `Clone` (cheap refcount bump sharing the SAME
/// instance), `PartialEq` is `Arc::ptr_eq` identity (a clone equals its source;
/// two distinct instances never do), and `Debug` prints the opaque marker — so
/// `Loss::Custom` keeps the derived `Debug + Clone + PartialEq`.
#[test]
fn handle_clone_and_ptr_eq_identity() {
    let h = CustomObjectiveHandle::new(Arc::new(LoglossCustom));
    let h_clone = h.clone();
    // A clone shares the same Arc → ptr-equal.
    assert_eq!(h, h_clone, "a cloned handle shares the same instance");

    // A distinct instance is never equal (identity equality).
    let other = CustomObjectiveHandle::new(Arc::new(LoglossCustom));
    assert_ne!(h, other, "distinct instances are never ptr-equal");

    // Debug is the opaque marker (does not leak the inner type).
    assert_eq!(format!("{h:?}"), "CustomObjectiveHandle(<dyn>)");

    // The metric handle has the same contract.
    let m = CustomMetricHandle::new(Arc::new(LoglossMetric));
    assert_eq!(m, m.clone());
    assert_eq!(format!("{m:?}"), "CustomMetricHandle(<dyn>)");
}

/// `Loss::Custom` carries the handle and keeps the enum's derived `Clone` /
/// `PartialEq`: a cloned `Loss::Custom` is equal to its source (same Arc), and a
/// `Loss::Custom` over a distinct instance is not.
#[test]
fn loss_custom_clone_and_eq() {
    let loss = Loss::Custom(CustomObjectiveHandle::new(Arc::new(LoglossCustom)));
    assert_eq!(loss, loss.clone(), "Loss::Custom clone shares the Arc");

    let other = Loss::Custom(CustomObjectiveHandle::new(Arc::new(LoglossCustom)));
    assert_ne!(loss, other, "distinct custom objectives are not equal");

    // validate() accepts a custom objective (no variant-level params to check).
    assert!(loss.validate().is_ok());
}
