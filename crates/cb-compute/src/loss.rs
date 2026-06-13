//! Per-loss first/second derivatives (gradient/hessian) for the losses this
//! phase covers: RMSE (regression), MAE / Quantile (robust regression), and the
//! binary-classification family Logloss / CrossEntropy / Focal (D-09).
//! CrossEntropy shares the Logloss sigmoid-gradient math exactly; Focal carries
//! its own `alpha`/`gamma`-weighted derivatives (`error_functions.h:1684-1709`).
//! All elementwise scalars — the per-object loop that calls
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

/// CrossEntropy first derivative for one object. IDENTICAL to Logloss
/// (`error_functions.cpp:304-336` `CalcCrossEntropyDerRangeImpl`): `der1 =
/// target - p`, `p = sigmoid(approx)`. CrossEntropy admits a soft `target ∈
/// [0,1]` (probabilistic label) where Logloss admits `{0,1}`, but the derivative
/// formula is the same — so this delegates to [`logloss_der1`] (the shared
/// sigmoid-gradient helper), never duplicating the math.
#[must_use]
pub fn cross_entropy_der1(approx: f64, target: f64) -> f64 {
    logloss_der1(approx, target)
}

/// CrossEntropy second derivative for one object: `der2 = -p*(1-p)`,
/// `p = sigmoid(approx)` — identical to Logloss (`error_functions.cpp:331`).
/// Delegates to [`logloss_der2`] (shared sigmoid-gradient helper).
#[must_use]
pub fn cross_entropy_der2(approx: f64, target: f64) -> f64 {
    logloss_der2(approx, target)
}

/// The Focal-loss probability clamp bounds (`error_functions.h` `TFocalError`):
/// `p` is clamped to `[FOCAL_P_MIN, 1 - FOCAL_P_MIN]` before the `log`/`pow` so a
/// saturated logit cannot drive `log(pt)` / `pow(1-pt, …)` to `NaN`/`-inf`
/// (T-04-02-02). `1e-13` is the upstream constant.
pub const FOCAL_P_MIN: f64 = 1e-13;

/// Focal loss first derivative for one object (`error_functions.h:1684-1709`
/// `TFocalError`, D-09), transcribed verbatim:
/// ```text
/// p  = 1/(1+exp(-approx));  p = clamp(p, 1e-13, 1-1e-13)
/// at = (target==1) ? alpha : 1-alpha
/// pt = (target==1) ? p     : 1-p
/// y  = 2*target - 1
/// der1 = -( at*y*pow(1-pt, gamma) * (gamma*pt*log(pt) + pt - 1) )
/// ```
/// Uses `std` `exp`/`pow`/`log` — Rust `f64` matches directly. `target` is the
/// binary label (`0.0`/`1.0`); the `target==1` branches test `target` exactly.
#[must_use]
pub fn focal_der1(approx: f64, target: f64, alpha: f64, gamma: f64) -> f64 {
    let p = sigmoid(approx).clamp(FOCAL_P_MIN, 1.0 - FOCAL_P_MIN);
    let is_pos = target == 1.0;
    let at = if is_pos { alpha } else { 1.0 - alpha };
    let pt = if is_pos { p } else { 1.0 - p };
    let y = 2.0 * target - 1.0;
    -(at * y * (1.0 - pt).powf(gamma) * (gamma * pt * pt.ln() + pt - 1.0))
}

/// Focal loss second derivative for one object (`error_functions.h:1684-1709`
/// `TFocalError`, D-09), transcribed verbatim:
/// ```text
/// u  = at*y*pow(1-pt, gamma);        du = -at*y*gamma*pow(1-pt, gamma-1)
/// v  = gamma*pt*log(pt) + pt - 1;    dv = gamma*log(pt) + gamma + 1
/// der2 = -( (du*v + u*dv) * y * (pt*(1-pt)) )
/// ```
/// `p` is clamped identically to [`focal_der1`] (T-04-02-02).
#[must_use]
pub fn focal_der2(approx: f64, target: f64, alpha: f64, gamma: f64) -> f64 {
    let p = sigmoid(approx).clamp(FOCAL_P_MIN, 1.0 - FOCAL_P_MIN);
    let is_pos = target == 1.0;
    let at = if is_pos { alpha } else { 1.0 - alpha };
    let pt = if is_pos { p } else { 1.0 - p };
    let y = 2.0 * target - 1.0;
    let u = at * y * (1.0 - pt).powf(gamma);
    let du = -at * y * gamma * (1.0 - pt).powf(gamma - 1.0);
    let v = gamma * pt * pt.ln() + pt - 1.0;
    let dv = gamma * pt.ln() + gamma + 1.0;
    -((du * v + u * dv) * y * (pt * (1.0 - pt)))
}

/// The MAE / Quantile default parameters: `alpha = 0.5` (the median) and the
/// `delta = 1e-6` deadzone (`error_functions.h:468-469` `TQuantileError`).
pub const QUANTILE_ALPHA: f64 = 0.5;
/// The MAE / Quantile deadzone half-width.
pub const QUANTILE_DELTA: f64 = 1e-6;

/// MAE / Quantile(alpha, delta) first derivative for one object:
/// `(target - approx > 0) ? alpha : -(1 - alpha)`, with a `|residual| < delta`
/// deadzone returning `0` (`error_functions.h:485-489` `TQuantileError::CalcDer`).
///
/// For the median (`alpha = 0.5`) the non-deadzone gradient is `+0.5` above the
/// approx and `-0.5` below — the sign of the residual scaled by the half-quantile.
#[must_use]
pub fn mae_der1(approx: f64, target: f64) -> f64 {
    let val = target - approx;
    if val.abs() < QUANTILE_DELTA {
        0.0
    } else if val > 0.0 {
        QUANTILE_ALPHA
    } else {
        -(1.0 - QUANTILE_ALPHA)
    }
}

/// MAE / Quantile second derivative for one object: `der2 = 0`
/// (`error_functions.h:491-493` — `QUANTILE_DER2_AND_DER3 = 0.0`). The Exact leaf
/// method does not use the hessian (it takes the weighted median), and Newton is
/// undefined for this loss (its denominator would be `scaledL2` alone).
#[must_use]
pub fn mae_der2(_approx: f64, _target: f64) -> f64 {
    0.0
}
