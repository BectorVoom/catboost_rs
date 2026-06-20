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

/// Comptime `SharedMemory` size for [`block_reduce_kernel`] (Pitfall 3 — the size
/// MUST be a compile-time `usize` const, not a runtime/topology value). It equals
/// the launch-geometry cube width (`CUBE_DIM = 32` in `gpu_runtime.rs` /
/// `cpu_runtime.rs`): one shared slot per unit (fallback tree-reduce) and an upper
/// bound on the per-cube plane count (plane carry). This is the ONE permitted `32`
/// (the launch-geometry / shared-memory size) — NOT a wave/warp-size literal in any
/// reduction STRIDE (the strides derive from `CUBE_DIM_X` / `PLANE_DIM`, D-09).
pub(crate) const BLOCK_REDUCE_SHMEM: usize = 32;

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

/// First-order Quantile{alpha, delta} gradient kernel: with `val = target -
/// approx`, `der1 = |val| < delta ? 0 : (val > 0 ? alpha : -(1-alpha))`.
///
/// `error_functions.h:485-489` (`TQuantileError::CalcDer`). `alpha`/`delta` pass
/// as length-1 `Array<F>` arguments (read at index 0) — NOT scalar args — to keep
/// the kernel fully generic over `F: Float` (AGENTS.md generics-float; the
/// [`focal_gradient_kernel`] / [`lq_gradient_kernel`] length-1-array precedent).
/// der2 is the constant `0` (no kernel — the dispatch fills a zero vec). The
/// branch result is assigned to a `mut` variable initialized to the deadzone
/// value via the if-as-STATEMENT pattern (CubeCL conditionals manual).
/// Elementwise, order-independent, no reduction (D-02). MAE routes through THIS
/// kernel at `alpha = 0.5`, `delta = 1e-6` (WR-04 — no duplicate MAE kernel), so
/// MAE and Quantile{0.5} are bit-identical by construction.
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
        // Band complement of the scalar `|val| < delta` deadzone: `|val| >= delta`
        // is OUTSIDE the deadzone, so the boundary `|val| == delta` returns the
        // signed quantile weight (matching `quantile_der1`), NOT 0. This mirrors
        // the correct `huber_gradient_kernel` `>= delta` band. The if-as-STATEMENT
        // pattern (CubeCL conditionals manual): `g` starts at the deadzone `0`,
        // then is overwritten with the `val < 0` arm and finally the `val > 0` arm.
        let mut g = F::new(0.0);
        if F::abs(val) >= d {
            g = F::new(0.0) - (one - a);
            if val > F::new(0.0) {
                g = a;
            }
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

/// Block-level sum reduction (the Phase-7.1 device primitive, D-7.1-04..09;
/// GPU-01 reduce). Each cube folds its `CUBE_DIM`-wide slice of `input` into a
/// SINGLE partial written to `output[CUBE_POS]`; the host finalizes the across-cube
/// sum via `cb-core::sum_f64` (the default atomic-free finalize — Open Q1; the
/// in-kernel atomic-finalize variant is Plan 02, NOT here). UNLIKE the elementwise
/// loss kernels above (D-02), this kernel DOES reduce on-device — but only the
/// intra-cube fold, leaving the parity-critical final sum to the frozen host order.
///
/// Wave-size-agnostic (D-09 / D-7.1-08): the `use_plane` path folds via
/// `plane_sum` over `PLANE_DIM` (the runtime plane width — NEVER a literal 32/64),
/// combining per-plane partials in shared memory keyed by `PLANE_POS` when a cube
/// spans more than one plane. The fallback path (no `Plane::Ops` capability) is a
/// `SharedMemory`-backed tree reduction whose stride derives from `CUBE_DIM_X`
/// (again no warp-size literal). `use_plane` is a `#[comptime]` flag resolved ONCE
/// host-side from `client.features().plane.contains(Plane::Ops)`, so the unused
/// branch is pruned at JIT time with zero device divergence (comptime
/// specialization manual). Mirrors the structure of upstream
/// `cuda_util/kernel/reduce.cuh::FastInBlockReduce` (shared-mem tree-reduce down to
/// plane width, then a plane reduction) — D-01 structural parity.
///
/// Generic over `F: Float` (AGENTS.md generics-float — no hard-coded float type).
/// Out-of-range lanes are zero-padded (`F::new(0.0)`) under the
/// `ABSOLUTE_POS < input.len()` bounds guard (T-7.1-01) so a non-multiple-of-cube
/// length stays correct. The `SharedMemory::new` SIZE is the comptime
/// [`BLOCK_REDUCE_SHMEM`] `usize` const (Pitfall 3 — a runtime/topology size will
/// not compile); the reduction STRIDE is `CUBE_DIM_X` / `PLANE_DIM`, never a
/// literal. Uses the if-as-STATEMENT pattern only (CubeCL conditionals manual).
#[cube(launch)]
pub fn block_reduce_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] use_plane: bool,
) {
    let tid = UNIT_POS;

    // Load this unit's element, zero-padding the idle out-of-range lanes
    // (if-as-STATEMENT: init to 0, overwrite inside the bounds guard).
    let mut acc = F::new(0.0);
    if ABSOLUTE_POS < input.len() {
        acc = input[ABSOLUTE_POS];
    }

    if use_plane {
        // Wave-agnostic plane fold: `plane_sum` gives EVERY unit its plane-wide
        // sum (width = PLANE_DIM, never a literal). When the cube spans more than
        // one plane (CUBE_DIM > PLANE_DIM), each plane's representative writes its
        // plane total into shared memory keyed by PLANE_POS, then unit 0 folds the
        // per-plane partials. The shared array is sized to the comptime CUBE_DIM
        // (an upper bound on the plane count — Pitfall 3).
        let plane_total = plane_sum(acc);
        let mut partials = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
        if UNIT_POS_PLANE == 0u32 {
            partials[PLANE_POS as usize] = plane_total;
        }
        sync_cube();
        if tid == 0u32 {
            // Number of planes in this cube = ceil(CUBE_DIM_X / PLANE_DIM). Fold the
            // per-plane partials sequentially (the count is small: 1 on wave32 at
            // CUBE_DIM 32). The loop bound derives from PLANE_DIM, not a literal.
            let num_planes = (CUBE_DIM_X + PLANE_DIM - 1u32) / PLANE_DIM;
            let mut sum = F::new(0.0);
            let mut p = 0u32;
            while p < num_planes {
                sum += partials[p as usize];
                p += 1u32;
            }
            output[CUBE_POS] = sum;
        }
    } else {
        // Fallback: shared-memory tree reduction (cubecl_reduce_sum.md). The array
        // SIZE is the comptime CUBE_DIM const; the stride starts at CUBE_DIM_X / 2
        // (the runtime cube width) — NEVER a literal 32/64 (D-09).
        let mut shared = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
        shared[tid as usize] = acc;
        sync_cube();
        let mut s = CUBE_DIM_X / 2u32;
        while s > 0u32 {
            if tid < s {
                let v = shared[(tid + s) as usize];
                shared[tid as usize] += v;
            }
            sync_cube();
            s /= 2u32;
        }
        if tid == 0u32 {
            output[CUBE_POS] = shared[0usize];
        }
    }
}

/// Block-level sum reduction with IN-KERNEL ATOMIC FINALIZE (D-03 / D-7.1-07; the
/// CUDA in-kernel-atomic reduction structure — Plan 02's second half of the reduce
/// primitive). Each cube folds its `CUBE_DIM`-wide slice into ONE partial (the SAME
/// wave-agnostic plane / shared-mem fold as [`block_reduce_kernel`]), then the
/// cube's representative unit (`UNIT_POS == 0`) `fetch_add`s that partial into a
/// length-1 global `Atomic<F>` accumulator (`acc[0]`). The cross-cube summation
/// ORDER is therefore non-deterministic (the hardware schedules the cubes' atomic
/// adds in an arbitrary order) — this matches CUDA's in-kernel atomic adds (D-03)
/// and the resulting run-to-run float-order variance is REPORTED by the
/// `kernels::reduce` oracle, NOT signed off here (the GPU-06 epsilon is 7.6's job —
/// D-7.1-07).
///
/// This is a SIBLING kernel to [`block_reduce_kernel`]: the Plan-01 atomic-free
/// finalize (one partial per cube + host `cb-core::sum_f64`) remains the portable
/// DEFAULT (Pitfall 4 — f64 atomic-add may be unsupported/slow on some backends;
/// the host gates this atomic path behind a `client.properties().atomic_type_usage`
/// capability check and falls back to the atomic-free helper when absent —
/// [`crate::gpu_runtime::launch_block_reduce_atomic_f64`]). Keeping it separate
/// means the default reduce path is byte-for-byte unchanged.
///
/// Wave-size-agnostic (D-09): the intra-cube fold uses `plane_sum` over `PLANE_DIM`
/// (plane path) or a `CUBE_DIM_X`-strided shared-mem tree-reduce (fallback) —
/// NEVER a literal 32/64 in any stride. `use_plane` is the `#[comptime]` flag
/// resolved host-side. Generic over `F: Float` (AGENTS.md generics-float). The
/// `SharedMemory::new` SIZE is the comptime [`BLOCK_REDUCE_SHMEM`] `usize` const
/// (Pitfall 3). Out-of-range lanes zero-padded under the `ABSOLUTE_POS <
/// input.len()` bounds guard (T-7.1-01). if-as-STATEMENT only.
///
/// NOTE on `&Array<Atomic<F>>`: the atomic accumulator is bound as a length-1
/// array; the underlying storage is plain `F`, so the host reads it back with the
/// same `bytemuck::cast_slice::<u8, F>` path as a non-atomic buffer (cubecl
/// `runtime_tests/atomic.rs` precedent). `fetch_add` takes `&self`, so the array is
/// `&Array<Atomic<F>>` (the per-element atomic provides interior mutability).
#[cube(launch)]
pub fn block_reduce_atomic_kernel<F: Float>(
    input: &Array<F>,
    acc: &Array<Atomic<F>>,
    #[comptime] use_plane: bool,
) {
    let tid = UNIT_POS;

    // Load this unit's element, zero-padding idle out-of-range lanes (T-7.1-01).
    let mut val = F::new(0.0);
    if ABSOLUTE_POS < input.len() {
        val = input[ABSOLUTE_POS];
    }

    // Intra-cube fold into a single per-cube partial held by unit 0 — identical
    // structure to `block_reduce_kernel`, but the finalize differs (atomic add into
    // a global accumulator instead of writing one slot per cube).
    let mut cube_partial = F::new(0.0);
    if use_plane {
        let plane_total = plane_sum(val);
        let mut partials = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
        if UNIT_POS_PLANE == 0u32 {
            partials[PLANE_POS as usize] = plane_total;
        }
        sync_cube();
        if tid == 0u32 {
            let num_planes = (CUBE_DIM_X + PLANE_DIM - 1u32) / PLANE_DIM;
            let mut sum = F::new(0.0);
            let mut p = 0u32;
            while p < num_planes {
                sum += partials[p as usize];
                p += 1u32;
            }
            cube_partial = sum;
        }
    } else {
        let mut shared = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
        shared[tid as usize] = val;
        sync_cube();
        let mut s = CUBE_DIM_X / 2u32;
        while s > 0u32 {
            if tid < s {
                let v = shared[(tid + s) as usize];
                shared[tid as usize] += v;
            }
            sync_cube();
            s /= 2u32;
        }
        if tid == 0u32 {
            cube_partial = shared[0usize];
        }
    }

    // In-kernel atomic finalize (D-03): the cube's representative adds its partial
    // into the single global accumulator. The order across cubes is
    // non-deterministic — the documented, accepted D-03 source of run-to-run
    // float-order variance (T-7.1-05).
    if tid == 0u32 {
        acc[0].fetch_add(cube_partial);
    }
}

/// Block-level inclusive/exclusive prefix-scan (the Phase-7.1 device primitive,
/// D-7.1-06; GPU-01 scan). Each unit reads `input[ABSOLUTE_POS]` (bounds-guarded,
/// zero-padded out-of-range) and writes the running prefix-sum to
/// `output[ABSOLUTE_POS]`. The `#[comptime] inclusive` flag selects the
/// CatBoost-CUDA `InplaceInclusiveScan` semantics (running total includes self)
/// vs the exclusive variant (sum of strictly-prior elements; `output[0] == 0`).
///
/// Structural parity (D-01): the cross-plane carry is a shared-memory
/// Hillis-Steele stride-doubling scan (`s = 1,2,4,…` with `sync_cube()` between
/// stages over per-plane partials), mirroring
/// `cuda_util/kernel/inplace_scan.cuh::InplaceInclusiveScan`. The within-plane
/// segment uses `plane_inclusive_sum` / `plane_exclusive_sum`, so the wave-level
/// prefix is expressed through CubeCL plane ops with NO warp/wave-size literal in
/// any stride (D-09 / D-7.1-08): the plane width is `PLANE_DIM` and the carry
/// stride loop runs over the per-plane count derived from `CUBE_DIM_X` / `PLANE_DIM`.
///
/// SCOPE (RESEARCH Open Q2): this kernel is correct WITHIN a single cube
/// (N ≤ CUBE_DIM — exactly one plane on wave32 gfx1100, where the cross-plane
/// carry collapses to the identity). The CROSS-CUBE carry (adding each cube's
/// running offset to the next) is NOT performed here and is the first forward
/// dependency for 7.2/7.3 (documented in the Plan-02 SUMMARY — NOT a silent scope
/// cut). The launch helper [`crate::gpu_runtime::launch_block_scan_f64`] and the
/// `kernels::scan` oracle therefore exercise N ≤ CUBE_DIM.
///
/// Generic over `F: Float` (AGENTS.md generics-float — no hard-coded float type).
/// `SharedMemory::new` SIZE is the comptime [`BLOCK_REDUCE_SHMEM`] `usize` const
/// (Pitfall 3 — a runtime/topology size will not compile); it holds one slot per
/// plane (an upper bound at CUBE_DIM units). Uses the if-as-STATEMENT pattern only
/// (CubeCL conditionals manual — never if-as-expression).
#[cube(launch)]
pub fn block_scan_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] inclusive: bool,
) {
    let tid = UNIT_POS;

    // Load this unit's element, zero-padding idle out-of-range lanes
    // (if-as-STATEMENT: init to 0, overwrite inside the bounds guard, T-7.1-01).
    let mut val = F::new(0.0);
    if ABSOLUTE_POS < input.len() {
        val = input[ABSOLUTE_POS];
    }

    // 1) Within-plane prefix via wave-agnostic plane ops (width = PLANE_DIM, never
    //    a literal). `scanned` is this unit's prefix WITHIN its own plane; `incl`
    //    is the plane-inclusive prefix (always includes self), used both to derive
    //    each plane's total and — for the exclusive request — to recover the
    //    inclusive value needed to seed the per-plane partial.
    let scanned_in_plane = plane_inclusive_sum(val);
    let mut scanned = scanned_in_plane;
    if !inclusive {
        scanned = plane_exclusive_sum(val);
    }

    // 2) Cross-plane carry (Hillis-Steele over per-plane inclusive totals — the
    //    `InplaceInclusiveScan` structure). The LAST unit of each plane holds that
    //    plane's inclusive total (`scanned_in_plane`); write it into a per-plane
    //    shared slot keyed by PLANE_POS.
    let mut partials = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
    if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
        partials[PLANE_POS as usize] = scanned_in_plane;
    }
    sync_cube();

    // Number of planes in this cube = ceil(CUBE_DIM_X / PLANE_DIM) (== 1 on wave32
    // at CUBE_DIM 32 — the carry below then adds nothing). The stride loop derives
    // its bound from CUBE_DIM_X / PLANE_DIM, NOT a literal 32/64 (D-09).
    let num_planes = (CUBE_DIM_X + PLANE_DIM - 1u32) / PLANE_DIM;

    // Hillis-Steele INCLUSIVE scan over the per-plane partials (mirrors
    // `inplace_scan.cuh`'s `val += data[tid - s]`, `s = 1,2,4,…`, `sync_cube()`
    // between stages). Only the first `num_planes` slots participate; one unit
    // (tid == its plane index) owns each slot.
    let mut s = 1u32;
    while s < num_planes {
        let mut add = F::new(0.0);
        // tid drives a slot iff tid < num_planes and tid >= s.
        if tid < num_planes {
            if tid >= s {
                add = partials[(tid - s) as usize];
            }
        }
        sync_cube();
        if tid < num_planes {
            if tid >= s {
                partials[tid as usize] += add;
            }
        }
        sync_cube();
        s *= 2u32;
    }

    // 3) Each plane's EXCLUSIVE carry = inclusive-scan of partials, shifted by one
    //    plane (carry for plane p = sum of all strictly-prior planes' totals).
    //    PLANE_POS == 0 has zero carry; otherwise carry = partials[PLANE_POS - 1].
    let mut carry = F::new(0.0);
    if PLANE_POS >= 1u32 {
        carry = partials[(PLANE_POS - 1u32) as usize];
    }
    sync_cube();

    let result = scanned + carry;
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = result;
    }
}

// Spike tests live in the dedicated `kernels/gradient.rs` file (source/test
// separation, CLAUDE.md / AGENTS.md — only a module declaration lives here, no
// test body). Mounted at `kernels::gradient` so `cargo test kernels::gradient`
// selects them. These spike harnesses hard-code `cubecl::cpu::CpuRuntime`, so the
// mount stays cpu-only (the `kernels` module now compiles under every backend, but
// these CPU-specific tests must not — they reference the cpu runtime by name).
#[cfg(all(test, feature = "cpu"))]
mod gradient;

// Block-reduce primitive oracle (source/test separation): the self-oracle vs
// `cb-core::sum_f64` lives in `kernels/reduce.rs`, mounted at `kernels::reduce`
// so `cargo test -p cb-backend ... kernels::reduce` selects it (D-7.1-09). UNLIKE
// the cpu-only spike harnesses above, it runs over the generic `SelectedRuntime`,
// so it builds/runs under EVERY backend (the rocm in-env oracle + wgpu host run).
#[cfg(test)]
mod reduce;

// Block-scan primitive oracle (source/test separation): the inclusive/exclusive
// prefix-sum self-oracle vs a Rust CPU prefix-sum lives in `kernels/scan.rs`,
// mounted at `kernels::scan` so `cargo test -p cb-backend ... kernels::scan`
// selects it (GPU-01 scan). Runs over the generic `SelectedRuntime`, so it
// builds/runs under EVERY backend (the rocm in-env oracle + wgpu host run).
#[cfg(test)]
mod scan;

// Histogram-scatter kernel tests (source/test separation): assertions live in
// `kernels/scatter.rs`, mounted at `kernels::scatter`. Cpu-only for the same reason
// as `gradient` above (the harness names `cubecl::cpu::CpuRuntime`).
#[cfg(all(test, feature = "cpu"))]
mod scatter;

// Device-resident RMSE der self-oracle (GPU-01 der, Phase 7.2): the GPU der1 over
// `SelectedRuntime` vs the `cb-compute::loss` CPU baseline, plus the SC-3
// device-residency hand-off assertion, live in `kernels/gradient_gpu.rs`, mounted
// at `kernels::gradient_gpu`. UNLIKE the cpu-only `gradient` spike above, it runs
// over the generic `SelectedRuntime`, so it builds/runs under EVERY backend (the
// rocm in-env oracle + wgpu host run).
#[cfg(test)]
mod gradient_gpu;
