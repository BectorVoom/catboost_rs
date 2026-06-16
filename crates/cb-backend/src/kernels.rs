//! First-class `#[cube]` compute kernels for the CPU backend (D-01/D-03).
//!
//! Every kernel here does ONLY order-independent, per-element work (D-02/D-06):
//! one output element per thread, indexed by [`ABSOLUTE_POS`], guarded by a
//! bounds check. Kernels NEVER perform a float reduction (sum/scan) — all
//! parity-critical reductions are finalized host-side through `cb-core::sum_f64`
//! in the frozen sequential order (D-02/D-05). This preserves the Phase-2
//! reduction invariant so CubeCL's parallelism cannot drift the 1e-5 oracle bar.
//!
//! Kernels are generic over `F: Float` (AGENTS.md generics-float rule) — no
//! float type is hard-coded in a kernel signature.

use cubecl::prelude::*;

/// First-order RMSE gradient kernel: `der1[i] = target[i] - approx[i]`.
///
/// CatBoost's RMSE first derivative for object `i` is `target[i] - approx[i]`
/// (`error_functions.*`); it is purely elementwise, so it maps to one thread per
/// object with no cross-thread communication. The bounds check `ABSOLUTE_POS <
/// approx.len()` lets the host launch a thread count rounded up to a cube
/// multiple without reading out of bounds (T-03-00-01 mitigation).
///
/// This kernel does NO reduction (D-02): the per-object gradients it emits are
/// later summed host-side via `cb-core::sum_f64` when building histograms / leaf
/// values in the Wave-1 training slice.
#[cube(launch)]
pub fn gradient_kernel<F: Float>(approx: &Array<F>, target: &Array<F>, der1: &mut Array<F>) {
    if ABSOLUTE_POS < approx.len() {
        der1[ABSOLUTE_POS] = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS];
    }
}

/// First-order Logloss / CrossEntropy gradient kernel:
/// `p = sigmoid(approx[i]); der1[i] = target[i] - p`.
///
/// `error_functions.cpp:317-340`: `e = exp(approx); p = 1 - 1/(1+e)`
/// (== `sigmoid(approx)`), `der1 = target - p`. The approx is the raw logit
/// (`RawFormulaVal`) — sigmoid is applied exactly once here (Pitfall 6). All
/// `Float` ops (`exp`) are kernel-legal. Order-independent, no reduction (D-02).
#[cube(launch)]
pub fn logloss_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let e = F::exp(approx[ABSOLUTE_POS]);
        let p = F::new(1.0) - F::new(1.0) / (F::new(1.0) + e);
        der1[ABSOLUTE_POS] = target[ABSOLUTE_POS] - p;
    }
}

/// First-order MAE / Quantile(alpha=0.5, delta=1e-6) gradient kernel:
/// `val = target - approx; der1 = |val|<delta ? 0 : (val>0 ? alpha : -(1-alpha))`.
///
/// `error_functions.h:485-489` (`TQuantileError::CalcDer`, alpha=0.5, delta=1e-6).
/// Elementwise, order-independent, no reduction (D-02). Per the CubeCL
/// conditionals manual the branch result is assigned to a `mut` variable
/// initialized to the deadzone value (avoiding `if`-as-expression IR pitfalls).
#[cube(launch)]
pub fn mae_gradient_kernel<F: Float>(approx: &Array<F>, target: &Array<F>, der1: &mut Array<F>) {
    if ABSOLUTE_POS < approx.len() {
        let val = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS];
        let alpha = F::new(0.5);
        let delta = F::new(1e-6);
        let mut g = F::new(0.0);
        if val > delta {
            g = alpha;
        } else if val < F::new(0.0) - delta {
            g = F::new(0.0) - (F::new(1.0) - alpha);
        }
        der1[ABSOLUTE_POS] = g;
    }
}

/// First-order Quantile{alpha, delta} gradient kernel: with `val = target -
/// approx`, `der1 = |val| < delta ? 0 : (val > 0 ? alpha : -(1-alpha))`.
///
/// `error_functions.h:485-489` (`TQuantileError::CalcDer`). Generalizes
/// [`mae_gradient_kernel`] (which hardcodes `alpha = 0.5`, `delta = 1e-6`) to
/// arbitrary `alpha`/`delta`, passed as length-1 `Array<F>` arguments (read at
/// index 0) — NOT scalar args — to keep the kernel fully generic over `F: Float`
/// (AGENTS.md generics-float; the [`focal_gradient_kernel`] / [`lq_gradient_kernel`]
/// length-1-array precedent). der2 is the constant `0` (no kernel — the dispatch
/// fills a zero vec, the Mae precedent). The branch result is assigned to a `mut`
/// variable initialized to the deadzone value via the if-as-STATEMENT pattern
/// (CubeCL conditionals manual). Elementwise, order-independent, no reduction
/// (D-02). At `alpha = 0.5`, `delta = 1e-6` this equals [`mae_gradient_kernel`].
#[cube(launch)]
pub fn quantile_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
    alpha: &Array<F>,
    delta: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let a = alpha[0];
        let d = delta[0];
        let val = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS];
        let mut g = F::new(0.0);
        if val > d {
            g = a;
        } else if val < F::new(0.0) - d {
            g = F::new(0.0) - (one - a);
        }
        der1[ABSOLUTE_POS] = g;
    }
}

/// Second-order Logloss / CrossEntropy hessian kernel:
/// `p = sigmoid(approx[i]); der2[i] = -p*(1-p)`.
///
/// `error_functions.cpp:331` — `der2 = -p*(1-p)`. Elementwise, no reduction
/// (D-02). The RMSE hessian is the constant `-1.0`, so it needs no kernel; the
/// host fills it directly.
#[cube(launch)]
pub fn logloss_hessian_kernel<F: Float>(approx: &Array<F>, der2: &mut Array<F>) {
    if ABSOLUTE_POS < approx.len() {
        let e = F::exp(approx[ABSOLUTE_POS]);
        let p = F::new(1.0) - F::new(1.0) / (F::new(1.0) + e);
        der2[ABSOLUTE_POS] = F::new(0.0) - p * (F::new(1.0) - p);
    }
}

/// First-order Focal gradient kernel (`error_functions.h:1684-1709` `TFocalError`):
/// `p = clamp(sigmoid(approx), 1e-13, 1-1e-13)`; with `at`/`pt` selected by the
/// binary label and `y = 2*target - 1`,
/// `der1 = -( at*y*pow(1-pt, gamma) * (gamma*pt*log(pt) + pt - 1) )`.
///
/// Elementwise, order-independent, no reduction (D-02). The loss parameters
/// `alpha`/`gamma` are passed as length-1 `Array<F>` arguments (read at index 0)
/// rather than as scalar kernel args — this keeps the kernel FULLY generic over
/// `F: Float` (AGENTS.md generics-float; a generic scalar arg would require the
/// non-generic `F: ScalarArgType + CubeElement + …` bound). The `target == 1`
/// branch selects `at`/`pt` via the if-as-STATEMENT pattern (CubeCL conditionals
/// manual — never if-as-expression). `p` is clamped before `ln`/`powf` so a
/// saturated logit cannot produce `NaN` (T-04-02-02).
#[cube(launch)]
pub fn focal_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
    alpha: &Array<F>,
    gamma: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let p_min = F::new(1e-13);
        let a = alpha[0];
        let g = gamma[0];
        let e = F::exp(F::new(0.0) - approx[ABSOLUTE_POS]);
        let p = F::clamp(one / (one + e), p_min, one - p_min);

        let is_pos = target[ABSOLUTE_POS] == one;
        let mut at = one - a;
        let mut pt = one - p;
        if is_pos {
            at = a;
            pt = p;
        }
        let y = F::new(2.0) * target[ABSOLUTE_POS] - one;

        let factor = F::powf(one - pt, g);
        let inner = g * pt * F::ln(pt) + pt - one;
        der1[ABSOLUTE_POS] = F::new(0.0) - (at * y * factor * inner);
    }
}

/// Second-order Focal hessian kernel (`error_functions.h:1684-1709`
/// `TFocalError`):
/// ```text
/// u = at*y*pow(1-pt, gamma);        du = -at*y*gamma*pow(1-pt, gamma-1)
/// v = gamma*pt*log(pt) + pt - 1;    dv = gamma*log(pt) + gamma + 1
/// der2 = -( (du*v + u*dv) * y * (pt*(1-pt)) )
/// ```
/// Same clamp / label-branch / generics-float discipline as
/// [`focal_gradient_kernel`].
#[cube(launch)]
pub fn focal_hessian_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der2: &mut Array<F>,
    alpha: &Array<F>,
    gamma: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let p_min = F::new(1e-13);
        let a = alpha[0];
        let g = gamma[0];
        let e = F::exp(F::new(0.0) - approx[ABSOLUTE_POS]);
        let p = F::clamp(one / (one + e), p_min, one - p_min);

        let is_pos = target[ABSOLUTE_POS] == one;
        let mut at = one - a;
        let mut pt = one - p;
        if is_pos {
            at = a;
            pt = p;
        }
        let y = F::new(2.0) * target[ABSOLUTE_POS] - one;

        let u = at * y * F::powf(one - pt, g);
        let du = (F::new(0.0) - at) * y * g * F::powf(one - pt, g - one);
        let v = g * pt * F::ln(pt) + pt - one;
        let dv = g * F::ln(pt) + g + one;
        der2[ABSOLUTE_POS] = F::new(0.0) - ((du * v + u * dv) * y * (pt * (one - pt)));
    }
}

/// First-order LogCosh gradient kernel: `der1[i] = -tanh(approx[i] - target[i])`.
///
/// `error_functions.h:414` (`TLogCoshError::CalcDer`). Non-parametric, smooth
/// (the saturating analogue of MAE's sign gradient). Elementwise, no reduction
/// (D-02). `F::tanh` is a kernel-legal `Float` op.
#[cube(launch)]
pub fn logcosh_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let r = approx[ABSOLUTE_POS] - target[ABSOLUTE_POS];
        der1[ABSOLUTE_POS] = F::new(0.0) - F::tanh(r);
    }
}

/// Second-order LogCosh hessian kernel:
/// `der2[i] = -1 / (cosh(approx[i] - target[i]))^2`.
///
/// `error_functions.h:418` (`TLogCoshError::CalcDer2`). Always strictly negative
/// (convex loss). Elementwise, no reduction (D-02). `F::cosh` is kernel-legal.
#[cube(launch)]
pub fn logcosh_hessian_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der2: &mut Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let r = approx[ABSOLUTE_POS] - target[ABSOLUTE_POS];
        let c = F::cosh(r);
        der2[ABSOLUTE_POS] = F::new(0.0) - F::new(1.0) / (c * c);
    }
}

/// First-order Lq{q} gradient kernel:
/// `der1[i] = q * sign(target-approx) * |approx-target|^(q-1)`.
///
/// `error_functions.h:553` (`TLqError::CalcDer`). The loss exponent `q` is passed
/// as a length-1 `Array<F>` (read at index 0) — NOT a scalar arg — to keep the
/// kernel fully generic over `F: Float` (AGENTS.md generics-float; the
/// `focal_gradient_kernel` length-1-array precedent). The `target - approx > 0`
/// sign is selected via the if-as-STATEMENT pattern (CubeCL conditionals manual —
/// never if-as-expression): `sign` is initialized to `-1` and flipped to `+1`
/// only when the residual is positive, matching upstream's `> 0 ? 1 : -1`.
#[cube(launch)]
pub fn lq_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
    q: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let qv = q[0];
        let a = approx[ABSOLUTE_POS];
        let t = target[ABSOLUTE_POS];
        let abs_loss = F::abs(a - t);
        let abs_loss_q = F::powf(abs_loss, qv - one);
        let mut sign = F::new(0.0) - one;
        if t - a > F::new(0.0) {
            sign = one;
        }
        der1[ABSOLUTE_POS] = qv * sign * abs_loss_q;
    }
}

/// Second-order Lq{q} hessian kernel:
/// `der2[i] = -q * (q-1) * |target-approx|^(q-2)`.
///
/// `error_functions.h:558` (`TLqError::CalcDer2`). Newton-clean only for
/// `q >= 2` (RESEARCH Pitfall 6); the Wave-1 fixture pins `q = 2.0`, where this
/// collapses to the constant `-2`. `q` passes as a length-1 `Array<F>` like
/// [`lq_gradient_kernel`]. Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn lq_hessian_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der2: &mut Array<F>,
    q: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let qv = q[0];
        let abs_loss = F::abs(target[ABSOLUTE_POS] - approx[ABSOLUTE_POS]);
        let pow_term = F::powf(abs_loss, qv - F::new(2.0));
        der2[ABSOLUTE_POS] = (F::new(0.0) - qv) * (qv - one) * pow_term;
    }
}

/// First-order Huber{delta} gradient kernel: with `diff = target - approx`,
/// `der1[i] = |diff| < delta ? diff : (diff > 0 ? delta : -delta)`.
///
/// `error_functions.h:1612` (`THuberError::CalcDer`). `delta` passes as a
/// length-1 `Array<F>` (read at index 0) — generics-float discipline. The
/// in-band / saturated branch and the `diff > 0` sign both use the
/// if-as-STATEMENT pattern (CubeCL conditionals manual): `g` is initialized to
/// the in-band value `diff`, then overwritten by `±delta` only when
/// `|diff| >= delta`. Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn huber_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
    delta: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let d = delta[0];
        let diff = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS];
        let mut g = diff;
        if F::abs(diff) >= d {
            g = F::new(0.0) - d;
            if diff > F::new(0.0) {
                g = d;
            }
        }
        der1[ABSOLUTE_POS] = g;
    }
}

/// Second-order Huber{delta} hessian kernel: with `diff = target - approx`,
/// `der2[i] = |diff| < delta ? -1 : 0`.
///
/// `error_functions.h:1621` (`THuberError::CalcDer2`). `delta` passes as a
/// length-1 `Array<F>` like [`huber_gradient_kernel`]. The strict `<` band
/// boundary matches upstream. if-as-STATEMENT: `h` initialized to the saturated
/// `0`, set to `-1` only inside the band. Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn huber_hessian_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der2: &mut Array<F>,
    delta: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let d = delta[0];
        let diff = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS];
        let mut h = F::new(0.0);
        if F::abs(diff) < d {
            h = F::new(0.0) - F::new(1.0);
        }
        der2[ABSOLUTE_POS] = h;
    }
}

/// First-order Expectile{alpha} gradient kernel: with `e = target - approx`,
/// `der1[i] = (e > 0) ? 2*alpha*e : 2*(1-alpha)*e`.
///
/// `error_functions.h:527` (`TExpectileError::CalcDer`). `alpha` passes as a
/// length-1 `Array<F>` (read at index 0) — generics-float discipline. The
/// `e > 0` asymmetry uses the if-as-STATEMENT pattern: `g` is initialized to the
/// below-branch (`2*(1-alpha)*e`, which also covers the `e == 0` boundary like
/// upstream's `> 0` test) and overwritten by the above-branch only for `e > 0`.
/// Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn expectile_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
    alpha: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let two = F::new(2.0);
        let a = alpha[0];
        let e = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS];
        let mut g = two * (one - a) * e;
        if e > F::new(0.0) {
            g = two * a * e;
        }
        der1[ABSOLUTE_POS] = g;
    }
}

/// Second-order Expectile{alpha} hessian kernel: with `e = target - approx`,
/// `der2[i] = (e > 0) ? -2*alpha : -2*(1-alpha)`.
///
/// `error_functions.h:532` (`TExpectileError::CalcDer2`). `alpha` passes as a
/// length-1 `Array<F>` like [`expectile_gradient_kernel`]. Piecewise-constant; the
/// `e == 0` boundary selects the below-branch (`-2*(1-alpha)`), matching
/// upstream's `> 0`. if-as-STATEMENT. Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn expectile_hessian_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der2: &mut Array<F>,
    alpha: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let two = F::new(2.0);
        let a = alpha[0];
        let e = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS];
        let mut h = (F::new(0.0) - two) * (one - a);
        if e > F::new(0.0) {
            h = (F::new(0.0) - two) * a;
        }
        der2[ABSOLUTE_POS] = h;
    }
}

/// First-order Poisson gradient kernel: `der1[i] = target[i] - exp(approx[i])`
/// over the RAW approx.
///
/// `error_functions.h:657-676` (`TPoissonError::CalcDer`). Poisson is
/// IsStoreExpApprox upstream but cb-train stores RAW approx and computes `F::exp`
/// INLINE here (the [`logloss_gradient_kernel`] inline-link precedent — the final
/// prediction applies the `Exponent` transform). `F::exp` is kernel-legal.
/// Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn poisson_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let e = F::exp(approx[ABSOLUTE_POS]);
        der1[ABSOLUTE_POS] = target[ABSOLUTE_POS] - e;
    }
}

/// Second-order Poisson hessian kernel: `der2[i] = -exp(approx[i])` over the RAW
/// approx.
///
/// `error_functions.h:657-676` (`TPoissonError::CalcDer2 = -expApprox`). Always
/// strictly negative (convex). `F::exp` INLINE on the raw approx (the Poisson
/// inline-link discipline). Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn poisson_hessian_kernel<F: Float>(approx: &Array<F>, der2: &mut Array<F>) {
    if ABSOLUTE_POS < approx.len() {
        let e = F::exp(approx[ABSOLUTE_POS]);
        der2[ABSOLUTE_POS] = F::new(0.0) - e;
    }
}

/// First-order Tweedie{variance_power} gradient kernel: with `p = variance_power`,
/// `der1[i] = target*e^((1-p)*approx) - e^((2-p)*approx)` over the RAW approx.
///
/// `error_functions.h:1648-1652` (`TTweedieError::CalcDer`). The `variance_power`
/// passes as a length-1 `Array<F>` (read at index 0) — generics-float discipline
/// (the [`focal_gradient_kernel`] length-1-array precedent). Tweedie is NOT
/// exp-approx (`error_functions.h:1644`): the `F::exp` lives INSIDE the der over
/// the raw approx; no `Exponent` predict transform (A4). `F::exp` is kernel-legal.
/// Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn tweedie_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
    variance_power: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let two = F::new(2.0);
        let p = variance_power[0];
        let a = approx[ABSOLUTE_POS];
        let t = target[ABSOLUTE_POS];
        let e1 = F::exp((one - p) * a);
        let e2 = F::exp((two - p) * a);
        der1[ABSOLUTE_POS] = t * e1 - e2;
    }
}

/// Second-order Tweedie{variance_power} hessian kernel: with `p = variance_power`,
/// `der2[i] = target*(1-p)*e^((1-p)*approx) - (2-p)*e^((2-p)*approx)` over the RAW
/// approx.
///
/// `error_functions.h:1654-1658` (`TTweedieError::CalcDer2`). `variance_power`
/// passes as a length-1 `Array<F>` like [`tweedie_gradient_kernel`]. exp INSIDE
/// the der (raw approx, NOT exp-approx). Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn tweedie_hessian_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der2: &mut Array<F>,
    variance_power: &Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let two = F::new(2.0);
        let p = variance_power[0];
        let a = approx[ABSOLUTE_POS];
        let t = target[ABSOLUTE_POS];
        let e1 = F::exp((one - p) * a);
        let e2 = F::exp((two - p) * a);
        der2[ABSOLUTE_POS] = t * (one - p) * e1 - (two - p) * e2;
    }
}

/// First-order MAPE gradient kernel: `der1[i] = sign(target-approx) /
/// max(1.0, |target|)`.
///
/// `error_functions.h:607-630` (`TMAPEError::CalcDer`). Non-parametric; the divisor
/// `max(1.0, |target|) >= 1.0` so the division is always safe (T-06.1.02-04). The
/// `1.f` divisor floor is f32-domain upstream (Pitfall 7); `F::max(1.0, |t|)`
/// reproduces it. The `target - approx > 0` sign uses the if-as-STATEMENT pattern
/// (CubeCL conditionals manual): `sign` is initialized to `-1` (covering the tie
/// `target == approx`, upstream's `> 0 ? 1 : -1`) and flipped to `+1` only when the
/// residual is positive. der2 is the constant 0 (no kernel — the dispatch fills a
/// zero vec, the Mae precedent). Elementwise, no reduction (D-02).
#[cube(launch)]
pub fn mape_gradient_kernel<F: Float>(
    approx: &Array<F>,
    target: &Array<F>,
    der1: &mut Array<F>,
) {
    if ABSOLUTE_POS < approx.len() {
        let one = F::new(1.0);
        let a = approx[ABSOLUTE_POS];
        let t = target[ABSOLUTE_POS];
        let denom = F::max(one, F::abs(t));
        let mut sign = F::new(0.0) - one;
        if t - a > F::new(0.0) {
            sign = one;
        }
        der1[ABSOLUTE_POS] = sign / denom;
    }
}

/// Histogram-scatter kernel: per-object, write the object's weighted gradient
/// contribution into its OWN per-object output slot (`contrib[i] = der1[i] *
/// weight[i]`).
///
/// This is the order-independent "scatter" half of the histogram (D-02/D-05):
/// each thread writes a single per-object value with NO cross-thread
/// accumulation — there is no `+=` into a shared bin total inside the kernel.
/// The host then folds these per-object contributions into per-bin / per-leaf
/// totals through `cb-core::sum_f64` in canonical object order (the ORDERED
/// reduction the kernel deliberately does not do). For the unweighted path every
/// `weight[i] == 1`, so `contrib[i] == der1[i]`.
#[cube(launch)]
pub fn histogram_scatter_kernel<F: Float>(
    der1: &Array<F>,
    weight: &Array<F>,
    contrib: &mut Array<F>,
) {
    if ABSOLUTE_POS < der1.len() {
        contrib[ABSOLUTE_POS] = der1[ABSOLUTE_POS] * weight[ABSOLUTE_POS];
    }
}

// Spike tests live in the dedicated `kernels/gradient.rs` file (source/test
// separation, CLAUDE.md / AGENTS.md — only a module declaration lives here, no
// test body). Mounted at `kernels::gradient` so `cargo test kernels::gradient`
// selects them.
#[cfg(test)]
mod gradient;

// Histogram-scatter kernel tests (source/test separation): assertions live in
// `kernels/scatter.rs`, mounted at `kernels::scatter`.
#[cfg(test)]
mod scatter;
