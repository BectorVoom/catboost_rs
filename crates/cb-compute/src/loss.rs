//! Per-loss first/second derivatives (gradient/hessian) for the two losses this
//! phase covers: RMSE (regression) and Logloss / CrossEntropy (binary
//! classification). All elementwise scalars — the per-object loop that calls
//! them lives in the `cb-backend` kernel (D-02); every parity-critical SUM over
//! these derivatives is finalized host-side via `cb_core::sum_f64`.
//!
//! # Source of truth
//!
//! - `catboost/private/libs/algo_helpers/error_functions.h:391-402` (RMSE):
//!   `TRMSEError::CalcDer(approx, target) = target - approx` (der1);
//!   `TRMSEError::CalcDer2(...) = -1.0` (der2). The per-object weight is applied
//!   AFTER the derivative (`ders[i].Der1 *= weights[i]`), so these helpers take
//!   the unweighted scalar form.
//! - `catboost/private/libs/algo_helpers/error_functions.cpp:317-340`
//!   (Logloss / CrossEntropy):
//!   `e = exp(approx); p = 1 - 1/(1+e)` (== `sigmoid(approx)`);
//!   `der1 = target - p`; `der2 = -p*(1-p)`. The stored approx is the raw logit
//!   (`RawFormulaVal`) — sigmoid is applied here, not twice (Pitfall 6).
//!
//! # f64 discipline
//!
//! Approx and derivatives are computed in `f64` (matching upstream's `double`
//! accumulator path); `target` is logically an `f32` label widened to `f64`.
//! These are pure scalars and never spell a float SUM, so the D-08 raw-sum ban
//! does not touch this module.

/// RMSE first derivative for one object: `der1 = target - approx`.
///
/// `error_functions.h:391` — `TRMSEError::CalcDer(approx, target) = target -
/// approx`. The per-object weight is multiplied in by the caller afterward.
#[must_use]
pub fn rmse_der1(approx: f64, target: f64) -> f64 {
    target - approx
}

/// RMSE second derivative for one object: `der2 = -1.0` (constant).
///
/// `error_functions.h:392` — `TRMSEError::CalcDer2(...) = -1.0`.
#[must_use]
pub fn rmse_der2(_approx: f64, _target: f64) -> f64 {
    -1.0
}

/// The logistic sigmoid `p = 1 / (1 + exp(-approx))`, written as upstream's
/// `1 - 1/(1+exp(approx))` to match the `error_functions.cpp:317-340` arithmetic
/// path bit-for-bit.
///
/// `e = exp(approx); p = 1 - 1/(1+e)`. Algebraically identical to
/// `1/(1+exp(-approx))` but transcribed in the upstream form so rounding matches.
#[must_use]
pub fn sigmoid(approx: f64) -> f64 {
    let e = approx.exp();
    1.0 - 1.0 / (1.0 + e)
}

/// Logloss / CrossEntropy first derivative for one object: `der1 = target - p`
/// where `p = sigmoid(approx)` and `approx` is the raw logit.
///
/// `error_functions.cpp:320-330` — `der1 = target - p`. The raw-logit approx is
/// the model's `RawFormulaVal`; sigmoid is applied exactly once here (Pitfall 6).
#[must_use]
pub fn logloss_der1(approx: f64, target: f64) -> f64 {
    let p = sigmoid(approx);
    target - p
}

/// Logloss / CrossEntropy second derivative for one object: `der2 = -p*(1-p)`
/// where `p = sigmoid(approx)`.
///
/// `error_functions.cpp:331` — `der2 = -p*(1-p)`.
#[must_use]
pub fn logloss_der2(approx: f64, _target: f64) -> f64 {
    let p = sigmoid(approx);
    -p * (1.0 - p)
}
