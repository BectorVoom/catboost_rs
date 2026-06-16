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

/// Quantile(alpha, delta) first derivative for one object: with `val = target -
/// approx`, `|val| < delta ? 0 : (val > 0 ? alpha : -(1 - alpha))`
/// (`error_functions.h:485-489` `TQuantileError::CalcDer`).
///
/// The asymmetric pinball gradient: residuals where the target sits ABOVE the
/// approx (`val > 0`, i.e. the model under-predicts) are pushed up with weight
/// `alpha`; residuals BELOW (`val < 0`, over-prediction) are pushed down with
/// weight `1 - alpha`. The `|val| < delta` deadzone returns `0` to avoid a
/// vanishing-residual chatter. At the median (`alpha = 0.5`, `delta = 1e-6`) this
/// is exactly [`mae_der1`] (the MAE-equivalence — MAE == Quantile{0.5}).
/// `alpha ∈ [0, 1]`, `delta >= 0` are validated by [`crate::Loss::validate`].
#[must_use]
pub fn quantile_der1(approx: f64, target: f64, alpha: f64, delta: f64) -> f64 {
    let val = target - approx;
    if val.abs() < delta {
        0.0
    } else if val > 0.0 {
        alpha
    } else {
        -(1.0 - alpha)
    }
}

/// Quantile(alpha, delta) second derivative for one object: `der2 = 0`
/// (`error_functions.h:491-493` — `QUANTILE_DER2_AND_DER3 = 0.0`), independent of
/// `alpha`/`delta`. The Exact leaf method takes the weighted alpha-quantile (it
/// does not use the hessian), and Newton is undefined for this loss (its
/// denominator would be `scaledL2` alone).
#[must_use]
pub fn quantile_der2(_approx: f64, _target: f64, _alpha: f64, _delta: f64) -> f64 {
    0.0
}

/// MAE first derivative for one object — the median quantile: delegates to
/// [`quantile_der1`] with the default `alpha = 0.5`, `delta = 1e-6`
/// (`error_functions.h:485-489`; MAE == Quantile{0.5}). The non-deadzone gradient
/// is `+0.5` above the approx and `-0.5` below.
#[must_use]
pub fn mae_der1(approx: f64, target: f64) -> f64 {
    quantile_der1(approx, target, QUANTILE_ALPHA, QUANTILE_DELTA)
}

/// MAE second derivative for one object: `der2 = 0` — delegates to
/// [`quantile_der2`] (`error_functions.h:491-493`). The Exact leaf method does not
/// use the hessian (it takes the weighted median).
#[must_use]
pub fn mae_der2(approx: f64, target: f64) -> f64 {
    quantile_der2(approx, target, QUANTILE_ALPHA, QUANTILE_DELTA)
}

/// LogCosh first derivative for one object: `der1 = -tanh(approx - target)`.
///
/// `error_functions.h:414-415` — `TLogCoshError::CalcDer(approx, target) =
/// -tanh(approx - target)`. Non-parametric; the smooth analogue of the MAE sign
/// gradient (`tanh` saturates to `±1` for large residuals, is ~linear near zero).
#[must_use]
pub fn logcosh_der1(approx: f64, target: f64) -> f64 {
    -(approx - target).tanh()
}

/// LogCosh second derivative for one object:
/// `der2 = -1 / cosh(approx - target)^2`.
///
/// `error_functions.h:418-419` — `TLogCoshError::CalcDer2(approx, target) =
/// -1 / (cosh(approx - target) * cosh(approx - target))`. Always strictly
/// negative (the loss is convex), so Newton is well-defined; the Wave-1 fixture
/// nonetheless uses upstream's Exact default (Pitfall 2).
#[must_use]
pub fn logcosh_der2(approx: f64, target: f64) -> f64 {
    let c = (approx - target).cosh();
    -1.0 / (c * c)
}

/// Lq{q} first derivative for one object:
/// `der1 = q * sign(target - approx) * |approx - target|^(q-1)`.
///
/// `error_functions.h:553-556` — `TLqError::CalcDer`: `absLoss = |approx -
/// target|; return Q * (target - approx > 0 ? 1 : -1) * pow(absLoss, Q - 1)`.
/// The sign is taken on `target - approx` (note: ties `target == approx` map to
/// `-1` upstream, the `> 0 ? 1 : -1` branch). `q` is the caller-supplied exponent
/// (`>= 1`, validated by [`crate::Loss::validate`]).
#[must_use]
pub fn lq_der1(approx: f64, target: f64, q: f64) -> f64 {
    let abs_loss = (approx - target).abs();
    let abs_loss_q = abs_loss.powf(q - 1.0);
    let sign = if target - approx > 0.0 { 1.0 } else { -1.0 };
    q * sign * abs_loss_q
}

/// Lq{q} second derivative for one object:
/// `der2 = -q * (q-1) * |target - approx|^(q-2)`.
///
/// `error_functions.h:558-561` — `TLqError::CalcDer2`: `absLoss = |target -
/// approx|; return -Q * (Q - 1) * pow(absLoss, Q - 2)`. Newton-clean only for
/// `q >= 2`; for `q < 2` the `^(q-2)` term diverges as the residual approaches
/// zero (RESEARCH Pitfall 6), so the Wave-1 fixture pins `q = 2.0`. At `q = 2`
/// this collapses to the constant `-2` (`pow(absLoss, 0) == 1`).
#[must_use]
pub fn lq_der2(approx: f64, target: f64, q: f64) -> f64 {
    let abs_loss = (target - approx).abs();
    -q * (q - 1.0) * abs_loss.powf(q - 2.0)
}

/// Huber{delta} first derivative for one object: with `diff = target - approx`,
/// `der1 = |diff| < delta ? diff : (diff > 0 ? delta : -delta)`.
///
/// `error_functions.h:1612-1619` — `THuberError::CalcDer`: inside the band
/// (`|diff| < delta`) the gradient is the raw residual (L2-like); outside it
/// saturates to `±delta` (L1-like). The band boundary is the `<` (strict) test,
/// matching upstream. `delta > 0` is validated by [`crate::Loss::validate`].
#[must_use]
pub fn huber_der1(approx: f64, target: f64, delta: f64) -> f64 {
    let diff = target - approx;
    if diff.abs() < delta {
        diff
    } else if diff > 0.0 {
        delta
    } else {
        -delta
    }
}

/// Huber{delta} second derivative for one object: with `diff = target - approx`,
/// `der2 = |diff| < delta ? -1 : 0`.
///
/// `error_functions.h:1621-1627` — `THuberError::CalcDer2`: `HUBER_DER2 = -1.0`
/// inside the band, `0.0` outside (the saturated L1 region has zero curvature).
/// The same strict `<` band boundary as [`huber_der1`].
#[must_use]
pub fn huber_der2(approx: f64, target: f64, delta: f64) -> f64 {
    let diff = target - approx;
    if diff.abs() < delta {
        -1.0
    } else {
        0.0
    }
}

/// Expectile{alpha} first derivative for one object: with `e = target - approx`,
/// `der1 = (e > 0) ? 2*alpha*e : 2*(1-alpha)*e`.
///
/// `error_functions.h:527-530` — `TExpectileError::CalcDer`: `e = target -
/// approx; return (e > 0) ? 2*Alpha*e : 2*(1-Alpha)*e`. The asymmetric L2
/// gradient — above-prediction residuals are weighted `alpha`, below `1-alpha`.
/// At `alpha = 0.5` this is exactly the RMSE gradient (`2*0.5*e = e`).
/// `alpha ∈ [0, 1]` is validated by [`crate::Loss::validate`].
#[must_use]
pub fn expectile_der1(approx: f64, target: f64, alpha: f64) -> f64 {
    let e = target - approx;
    if e > 0.0 {
        2.0 * alpha * e
    } else {
        2.0 * (1.0 - alpha) * e
    }
}

/// Expectile{alpha} second derivative for one object: with `e = target - approx`,
/// `der2 = (e > 0) ? -2*alpha : -2*(1-alpha)`.
///
/// `error_functions.h:532-535` — `TExpectileError::CalcDer2`: `e = target -
/// approx; return (e > 0) ? -2*Alpha : -2*(1-Alpha)`. Piecewise-constant
/// (`-2*alpha` above prediction, `-2*(1-alpha)` below), so Newton is well-defined
/// everywhere. The `e > 0` boundary at `e == 0` selects the below-branch
/// (`-2*(1-alpha)`), matching upstream's `> 0` test exactly.
#[must_use]
pub fn expectile_der2(approx: f64, target: f64, alpha: f64) -> f64 {
    let e = target - approx;
    if e > 0.0 {
        -2.0 * alpha
    } else {
        -2.0 * (1.0 - alpha)
    }
}

/// Poisson first derivative for one object: `der1 = target - exp(approx)` over the
/// RAW approx.
///
/// `error_functions.h:657-676` — `TPoissonError::CalcDer`: upstream receives the
/// already-exponentiated approx (`expApprox`) and returns `target - expApprox`.
/// Poisson is `IsStoreExpApprox` upstream (`approx_updater_helpers.h:60-72`), but
/// cb-train stores RAW approx and computes `exp()` INLINE here — the same inline-
/// link equivalence as Logloss's [`sigmoid`] (`approx` is the model's
/// `RawFormulaVal`; the final prediction applies the `Exponent` transform). No
/// exp-approx storage is implemented (RESEARCH Pattern 2 / Pitfall 4).
#[must_use]
pub fn poisson_der1(approx: f64, target: f64) -> f64 {
    target - approx.exp()
}

/// Poisson second derivative for one object: `der2 = -exp(approx)` over the RAW
/// approx.
///
/// `error_functions.h:657-676` — `TPoissonError::CalcDer2 = -expApprox`. Always
/// strictly negative (the loss is convex), so Newton is well-defined. `exp()` is
/// computed INLINE on the raw approx (the [`poisson_der1`] inline-link discipline).
#[must_use]
pub fn poisson_der2(approx: f64, _target: f64) -> f64 {
    -approx.exp()
}

/// Tweedie{variance_power} first derivative for one object: with `p =
/// variance_power` and the RAW approx,
/// `der1 = target*e^((1-p)*approx) - e^((2-p)*approx)`.
///
/// `error_functions.h:1648-1652` — `TTweedieError::CalcDer`:
/// `der1 = -(-target*exp((1-p)*approx) + exp((2-p)*approx))` =
/// `target*exp((1-p)*approx) - exp((2-p)*approx)`. Tweedie is NOT exp-approx
/// (`isExpApprox==false`, `error_functions.h:1644`) — the `exp` lives INSIDE the
/// der formula directly over the raw approx; there is NO `Exponent` predict
/// transform (the prediction is the raw approx — A4). `variance_power ∈ (1, 2)` is
/// validated by [`crate::Loss::validate`].
#[must_use]
pub fn tweedie_der1(approx: f64, target: f64, variance_power: f64) -> f64 {
    let p = variance_power;
    target * ((1.0 - p) * approx).exp() - ((2.0 - p) * approx).exp()
}

/// Tweedie{variance_power} second derivative for one object: with `p =
/// variance_power` and the RAW approx,
/// `der2 = target*(1-p)*e^((1-p)*approx) - (2-p)*e^((2-p)*approx)`.
///
/// `error_functions.h:1654-1658` — `TTweedieError::CalcDer2`:
/// `der2 = -(target*(1-p)*exp((1-p)*approx) ... )` simplifies to
/// `target*(1-p)*exp((1-p)*approx) - (2-p)*exp((2-p)*approx)` (the partial
/// derivative of [`tweedie_der1`] w.r.t. `approx`). exp INSIDE the der (raw
/// approx, NOT exp-approx — `error_functions.h:1644`).
#[must_use]
pub fn tweedie_der2(approx: f64, target: f64, variance_power: f64) -> f64 {
    let p = variance_power;
    target * (1.0 - p) * ((1.0 - p) * approx).exp() - (2.0 - p) * ((2.0 - p) * approx).exp()
}

/// MAPE first derivative for one object: `der1 = sign(target - approx) /
/// max(1.0, |target|)`.
///
/// `error_functions.h:607-630` — `TMAPEError::CalcDer`: `der = (target - approx >
/// 0 ? 1 : -1) / Max(1.f, Abs(target))`. The `1.f` divisor floor is an f32-domain
/// literal upstream (Pitfall 7); transcribed as `f64::max(1.0, target.abs())` —
/// the divisor is always `>= 1.0` so the division is never by zero (T-06.1.02-04).
/// The `> 0 ? 1 : -1` branch maps the tie `target == approx` to `-1` (upstream's
/// exact form).
#[must_use]
pub fn mape_der1(approx: f64, target: f64) -> f64 {
    let denom = 1.0_f64.max(target.abs());
    let sign = if target - approx > 0.0 { 1.0 } else { -1.0 };
    sign / denom
}

/// MAPE second derivative for one object: `der2 = 0`.
///
/// `error_functions.h:607-630` — `TMAPEError` has `MAPE_DER2 = 0.0` (the absolute-
/// percentage residual is piecewise-linear, zero curvature). der2=0 makes Newton
/// undefined (Pitfall 5), so upstream routes MAPE to the Gradient leaf method.
#[must_use]
pub fn mape_der2(_approx: f64, _target: f64) -> f64 {
    0.0
}

/// The softmax `p[d] = exp(approx[d] - maxApprox) / Σ_d exp(approx[d] - maxApprox)`
/// over one object's per-dimension raw approx, reproducing upstream `CalcSoftmax`
/// (`eval_processing.h:15-26`) with the MAX-SUBTRACTION before `exp` (T-6.2-02 NaN
/// guard — a large approx magnitude cannot overflow `exp` to `Inf`/`NaN`).
///
/// `eval_processing.h:16` — `maxApprox = *MaxElement(approx)`; `:18` —
/// `softmax[d] = approx[d] - maxApprox`; FastExp; normalize by the sum. Uses
/// `f64::exp` (the oracle absorbs the upstream `FastExpWithInfInplace` table gap,
/// A2 — the established 6.1 precedent). Returns the normalized probabilities in
/// dimension order. An empty input returns an empty vector; an all-equal input
/// returns the uniform distribution.
#[must_use]
pub fn calc_softmax(approx: &[f64]) -> Vec<f64> {
    if approx.is_empty() {
        return Vec::new();
    }
    // maxApprox = MaxElement(approx) (eval_processing.h:16). `fold` over a copied
    // start avoids `partial_cmp().unwrap()` on NaN; the approx is finite here.
    let mut max_approx = f64::NEG_INFINITY;
    for &a in approx {
        if a > max_approx {
            max_approx = a;
        }
    }
    // exp(approx[d] - maxApprox) (eval_processing.h:18-20); f64::exp (A2).
    let exps: Vec<f64> = approx.iter().map(|&a| (a - max_approx).exp()).collect();
    // sumExpApprox = Σ exps (eval_processing.h:21-24). The k summands (k <= ~10
    // classes in scope) are summed left-to-right matching upstream's scalar loop;
    // this is a per-object normalizer, NOT a parity-critical histogram/leaf SUM, so
    // it does not route through `cb_core::sum_f64` (which is reserved for the
    // ordered cross-object reductions, D-08).
    let mut sum_exp = 0.0_f64;
    for &e in &exps {
        sum_exp += e;
    }
    exps.iter().map(|&e| e / sum_exp).collect()
}

/// The MultiClass softmax coupled first derivative + packed symmetric Hessian for
/// one object (`error_functions.h:687-728` `TMultiClassError::CalcDersMulti`),
/// transcribed verbatim.
///
/// Given the object's per-dimension raw `approx` (length `k`) and its REMAPPED
/// contiguous class index `target_class ∈ [0, k)` (Pitfall 4 — the raw label MUST
/// be remapped before this call), returns:
/// - `der1[d] = δ(d == target_class) - p[d]` with `p = calc_softmax(approx)`
///   (max-subtracted), and
/// - `der2` = the PACKED lower-triangular (== row-major upper-triangular, the
///   matrix is symmetric) Hessian of length `k*(k+1)/2`, ordered
///   `[(0,0),(0,1),…,(0,k-1),(1,1),(1,2),…,(k-1,k-1)]`: diagonal `(y,y)` entry is
///   `p_y*(p_y - 1)`, off-diagonal `(y,x>y)` entry is `p_y*p_x`
///   (`error_functions.h:704-712`).
///
/// This is the ONLY cross-dimension-coupled der this phase: the off-diagonal
/// Hessian entries couple dimensions, so the leaf delta is a dense symmetric solve
/// (`crate::solve_symmetric_newton`), not a per-dimension scalar Newton step.
///
/// The unweighted form is returned (`weight == 1`); the caller folds the per-object
/// weight in afterward (the `error_functions.h:716-726` `weight != 1` branch), the
/// same convention as the scalar `*_der1` helpers.
///
/// # Panics
/// Does not panic. An out-of-range `target_class` (`>= k`) leaves `der1` unchanged
/// at `-p[d]` for every dimension (no `+1`), so the caller's range validation
/// (T-6.2-01, `Loss::validate` / boosting gate) is the defense; this helper never
/// indexes out of bounds.
#[must_use]
pub fn softmax_ders(approx: &[f64], target_class: usize) -> (Vec<f64>, Vec<f64>) {
    let k = approx.len();
    let p = calc_softmax(approx);
    // Packed symmetric Hessian (error_functions.h:704-712): diag p_y*(p_y-1),
    // off-diag p_y*p_x, in `idx++` row-major-upper order.
    let mut der2 = Vec::with_capacity(k * (k + 1) / 2);
    for dim_y in 0..k {
        let p_y = p.get(dim_y).copied().unwrap_or(0.0);
        der2.push(p_y * (p_y - 1.0));
        for dim_x in (dim_y + 1)..k {
            let p_x = p.get(dim_x).copied().unwrap_or(0.0);
            der2.push(p_y * p_x);
        }
    }
    // der1[d] = -p[d], then der1[target_class] += 1 (error_functions.h:714-717).
    let mut der1: Vec<f64> = p.iter().map(|&pd| -pd).collect();
    if let Some(slot) = der1.get_mut(target_class) {
        *slot += 1.0;
    }
    (der1, der2)
}

/// The MultiClassOneVsAll per-dimension diagonal der for one object and one
/// dimension `d` (`error_functions.h:746-779` `TMultiClassOneVsAllError`),
/// transcribed verbatim — separable, so this is a SCALAR pair per dimension (no
/// cross-dimension coupling).
///
/// With `p = sigmoid(approx_d)` (the per-dimension positive-class probability,
/// `error_functions.h:758-759` — the SAME upstream sigmoid arithmetic as
/// [`sigmoid`]):
/// - `der1 = δ(d == target_class) - p` (`:760-762`), and
/// - `der2 = -p*(1 - p)` (`:763`, the diagonal Hessian entry).
///
/// `is_target` is `d == target_class`. Because the Hessian is diagonal, the leaf
/// delta is the EXISTING scalar Newton step per dimension
/// (`crate::newton_leaf_delta`), identical to the binary Logloss path — no dense
/// solve. Returns `(der1, der2)` unweighted (the caller folds in the weight).
#[must_use]
pub fn multiclass_onevsall_ders(approx_d: f64, is_target: bool) -> (f64, f64) {
    let p = sigmoid(approx_d);
    let der1 = if is_target { 1.0 - p } else { -p };
    let der2 = -p * (1.0 - p);
    (der1, der2)
}

/// The MultiLogloss / MultiCrossEntropy per-dimension DIAGONAL der for one object
/// and one label dimension `d` (`error_functions.h:781-820`
/// `TMultiCrossEntropyError`), transcribed verbatim.
///
/// MultiLogloss and MultiCrossEntropy are the SAME upstream class
/// (`tensor_search_helpers.cpp:236-238` dispatches both to
/// `TMultiCrossEntropyError`); they differ ONLY in the admissible target range
/// (MultiLogloss = `{0,1}` binary labels; MultiCrossEntropy = `[0,1]` probability
/// targets), validated in [`crate::Loss::validate`]. The der is identical, so both
/// losses call THIS single helper — there is no per-loss branch in the der math.
///
/// The loss is fully SEPARABLE (each label dimension independent), so this is a
/// SCALAR pair per dimension reusing the existing scalar [`sigmoid`]. Upstream
/// computes (`error_functions.h:791-808`):
/// ```text
/// derRef[d] = -sigmoid(approx_d);             // = -FastExp(a)/(1+FastExp(a))
/// der2[d]   = derRef[d] * (1 + derRef[d]);    // diagonal: -sigmoid*(1-sigmoid)
/// derRef[d] += target[d];                     // => target_d - sigmoid(approx_d)
/// ```
/// so with `p = sigmoid(approx_d)`:
/// - `der1 = target_d - p` (`:807`), and
/// - `der2 = (-p)*(1 - p) = -p*(1 - p)` (`:802`, the diagonal Hessian entry —
///   identical to the binary Logloss / OneVsAll diagonal).
///
/// Because the Hessian is diagonal, the leaf delta is the EXISTING scalar Newton
/// step per dimension (`crate::newton_leaf_delta`) — no dense solve. `target_d` is
/// the per-dimension label (`{0,1}` for MultiLogloss, `[0,1]` for
/// MultiCrossEntropy). Returns `(der1, der2)` unweighted (the caller folds in the
/// weight).
#[must_use]
pub fn multi_crossentropy_ders(approx_d: f64, target_d: f64) -> (f64, f64) {
    let p = sigmoid(approx_d);
    let der1 = target_d - p;
    let der2 = -p * (1.0 - p);
    (der1, der2)
}

/// QueryRMSE per-object first/second derivative for one object, given the
/// already-computed per-group `query_avrg` (LOSS-04, Wave A). RAW approx
/// (`isExpApprox == false`).
///
/// `error_functions.h:901-907` — `TQueryRmseError::CalcDersForQueries`:
/// ```text
/// ders[d].Der1 = targets[d] - approxes[d] - queryAvrg;
/// ders[d].Der2 = -1;
/// if (!weights.empty()) { ders[d].Der1 *= w; ders[d].Der2 *= w; }
/// ```
/// so `der1 = (target - approx - query_avrg) * weight` and `der2 = -1 * weight`.
/// Unlike the scalar `*_der1` helpers (which return the UNWEIGHTED scalar for the
/// caller to weight), the upstream querywise der folds the per-object `weight`
/// INTO the der here (the leaf reduction consumes it directly without re-
/// weighting); the per-group wrapper ([`crate::calc_ders_for_queries`]) passes the
/// object's `weight` (or `1.0` when unweighted) and the group's `query_avrg`
/// (computed via [`crate::group_reduce_weighted`] over `target - approx`).
///
/// The `query_avrg` is the per-group weighted residual mean
/// `Σ_g (target - approx)·w / Σ_g w` (`CalcQueryAvrg`, `error_functions.h:912-932`),
/// `0` for an empty / zero-weight group (the upstream `queryCount > 0` guard).
#[must_use]
pub fn queryrmse_der(approx: f64, target: f64, weight: f64, query_avrg: f64) -> (f64, f64) {
    let der1 = (target - approx - query_avrg) * weight;
    let der2 = -1.0 * weight;
    (der1, der2)
}

/// QuerySoftMax per-object first/second derivative for one object within a group,
/// given the already-computed per-group softmax probability `p` (the object's
/// weighted exp-share `expApprox·w / Σ_g expApprox·w`), the per-group
/// `sum_weighted_targets`, the object's `weight` and `target`, and the loss
/// `beta` / `lambda_reg` (LOSS-04, Wave A). RAW approx (`isExpApprox == false`);
/// the `exp` is taken INLINE in the per-group wrapper, max-shifted before `exp`.
///
/// `error_functions.cpp:560-569` — `TQuerySoftMaxError::CalcDersForSingleQuery`:
/// ```text
/// p = ders[dim].Der1 / sumExpApprox;       // Der1 held expApprox*weight
/// ders[dim].Der2 = Beta*sumWTargets*(Beta*p*(p-1) - LambdaReg);
/// ders[dim].Der1 = Beta*(-sumWTargets*p + weight*target);
/// ```
/// for `weight > 0`; objects with `weight <= 0` (and the whole group when
/// `sumWTargets <= 0`) get `der1 = der2 = 0` (the per-group wrapper applies those
/// guards). Like [`queryrmse_der`] this returns the der ALREADY weighted (the
/// `weight·target` term and the `p` numerator fold the per-object weight in), so
/// the caller does NOT re-multiply by `weight`.
#[must_use]
pub fn querysoftmax_der(
    p: f64,
    sum_weighted_targets: f64,
    weight: f64,
    target: f64,
    beta: f64,
    lambda_reg: f64,
) -> (f64, f64) {
    let der2 = beta * sum_weighted_targets * (beta * p * (p - 1.0) - lambda_reg);
    let der1 = beta * (-sum_weighted_targets * p + weight * target);
    (der1, der2)
}
