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

use cb_core::{CbError, CbResult};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Comptime `SharedMemory` size for [`block_reduce_kernel`] (Pitfall 3 — the size
/// MUST be a compile-time `usize` const, not a runtime/topology value). It equals
/// the launch-geometry cube width (`CUBE_DIM = 32` in `gpu_runtime.rs` /
/// `cpu_runtime.rs`): one shared slot per unit (fallback tree-reduce) and an upper
/// bound on the per-cube plane count (plane carry). This is the ONE permitted `32`
/// (the launch-geometry / shared-memory size) — NOT a wave/warp-size literal in any
/// reduction STRIDE (the strides derive from `CUBE_DIM_X` / `PLANE_DIM`, D-09).
///
/// PRECONDITION (WR-04): the launch cube width (`CUBE_DIM` in `gpu_runtime.rs`) MUST
/// be a power of two. The fallback tree-reduce below halves its stride
/// (`s = CUBE_DIM_X / 2; ...; s /= 2`), which only covers every element when the
/// width is a power of two; a non-power-of-two width silently drops the top
/// element(s). `gpu_runtime.rs` enforces this with a `const _: () =
/// assert!(CUBE_DIM.is_power_of_two())` guard so any drift is a compile error.
pub(crate) const BLOCK_REDUCE_SHMEM: usize = 32;

/// Comptime worst-case `SharedMemory` size for the per-block 2-channel pointwise
/// histogram (Phase 7.3 / Pitfall 3 — the size MUST be a compile-time `usize` const,
/// not a runtime/topology value). It is the 8-bit worst case: `2 channels * (1 <<
/// 8) bins = 512`, the upper bound across the one-byte non-binary bit-widths (5/6/7/8
/// — the comptime `bits` arg selects the USED PREFIX `2 * (1 << bits)` of this
/// allocation, mirroring upstream `pointwise_hist2_one_byte_templ.cuh`'s worst-case
/// `__shared__ float counters[...]` sizing). This is a shared-memory SIZE (the
/// allocation), NOT a wave/warp-size literal in any stride (D-09) — the analog of
/// [`BLOCK_REDUCE_SHMEM`].
pub(crate) const HIST_SHMEM: usize = 2 * (1 << 8);

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

/// 2-channel pointwise histogram fill — the 8-bit non-binary `ComputeHist2NonBinary`
/// analog (Phase 7.3, GPU-01 histogram slice; D-7.3-01..05). For every (feature, bin)
/// it accumulates two interleaved channels: channel 0 = Σ der1 ("target"), channel 1
/// = Σ weight, written into the global `bin_sums` buffer at the FROZEN layout index
///
/// ```text
/// index(feature, bin, channel) = (feature * n_bins + bin) * 2 + channel
/// ```
///
/// (the single-tree collapse of `ShiftPartAndBinSumsPtr`: `histLineSize = 2 *
/// totalBinFeatures`, `part = fold = 0`, `FirstFoldIndex = 0` — see the module doc of
/// `kernels/pointwise_hist.rs`). `n_bins = 1 << bits` is derived at COMPTIME from the
/// `bits` arg, so the SAME kernel covers the whole one-byte non-binary family —
/// 5/6/7/8-bit — selected host-side per the feature group's border count (Plan B
/// landed 5/6/7 over the Plan A 8-bit; D-7.3-02), with NO runtime bit-count branch
/// (the comptime value is resolved at JIT time, mirroring the 7.1 `use_plane` pattern).
/// The histogram line size (`feature * n_bins`) and the used shared/global prefix
/// (`2 * (1 << bits)` cells per feature) both derive from `bits` at comptime; the
/// `HIST_SHMEM` allocation stays the fixed 8-bit worst case (only the USED prefix
/// shrinks for the narrower widths).
///
/// # In-kernel atomic merge (D-03 / D-7.3-03)
///
/// `bin_sums: &Array<Atomic<F>>` is the GLOBAL histogram; each thread `fetch_add`s its
/// per-object contribution directly into the global cell (the genuine D-03 in-kernel
/// atomic merge — the `block_reduce_atomic_kernel` `acc[0].fetch_add(...)` primitive
/// generalized to a (feature, bin, channel)-indexed buffer). Because many threads
/// contribute to the same cell, the cross-thread accumulation ORDER is
/// non-deterministic — the accepted D-03 source of run-to-run float-order variance,
/// REPORTED (not signed off) by the `kernels::pointwise_hist` oracle (GPU-06 epsilon
/// is 7.6's job). Upstream's per-block shared-memory working histogram + the
/// `BLOCKS_PER_FEATURE > 1 ? atomicAdd : WriteThrough` merge guard is a PERFORMANCE
/// refinement over this same atomic-merge STRUCTURE (it reduces global-atomic traffic
/// by pre-reducing within a block); the MVP fill uses the direct global atomic merge,
/// which is structurally faithful (D-01) and provably correct — the shared-mem
/// pre-reduction is an additive perf follow-up (RESEARCH Open Q3). The comptime
/// [`HIST_SHMEM`] worst-case size is reserved for that follow-up.
///
/// # Wave-size policy (D-09)
///
/// The per-object loop strides by the TOTAL thread count `CUBE_COUNT_X * CUBE_DIM_X`
/// (a grid-stride loop) — derived from the launch topology intrinsics, NEVER a literal
/// 32/64. No `& 31`/`tiled_partition<32>` appears: the bin index comes from
/// `cindex[feature * n + indices[i]]`, not a warp-lane partition. Generic over `F:
/// Float` (AGENTS.md generics-float). Every device read is under a POSITION bounds
/// guard (`i < indices.len()`) so a non-cube-multiple object count stays correct
/// (T-7.1-01). The VALUE-derived reads (`indices[i]` as an object id, `cindex[...]`
/// as a bin) are NOT guarded in-kernel; their ranges are validated HOST-SIDE in
/// `launch_pointwise_hist2_into` (CR-01) before launch, which is what keeps a
/// malformed object id / bin from faulting on the device. if-as-STATEMENT only
/// (CubeCL conditionals manual).
///
/// `der1`/`weight` are length `n` (per object, object order). `cindex` is the
/// quantized bin matrix laid out feature-major (`cindex[feature * n + obj]`).
/// `indices` (length `n`) is the object visiting order. `n` and `n_features` are
/// passed as comptime so the bounds and the feature loop are JIT-resolved.
#[cube(launch)]
pub fn pointwise_hist2_nonbinary_kernel<F: Float>(
    der1: &Array<F>,
    weight: &Array<F>,
    cindex: &Array<u32>,
    indices: &Array<u32>,
    bin_sums: &Array<Atomic<F>>,
    n_features: u32,
    #[comptime] bits: u32,
) {
    // n_bins = 1 << bits (comptime; the USED prefix of the HIST_SHMEM worst case).
    // Held as `usize` because it participates in the (feature, bin) index arithmetic
    // (cubecl array indexers are `usize` — `Cubecl_shared_memory.md` Indexing Safety).
    let n_bins = comptime!((1u32 << bits) as usize);
    // n (object count) and the feature-major cindex stride are derived from the input
    // lengths — no comptime `n` arg needed (the bounds and stride are runtime values
    // from the device arrays, exactly like the elementwise kernels' `approx.len()`).
    // `indices.len()` is `usize`, so all index arithmetic below stays in `usize`; the
    // `u32` values read from `cindex`/`indices` are cast to `usize` at the index site.
    let n = indices.len();
    let n_features_usize = n_features as usize;

    // Grid-stride loop over the object-visiting order. The stride is the total thread
    // count (CUBE_COUNT * CUBE_DIM) — a topology-derived value, NEVER a literal 32/64
    // (D-09). Each unit processes objects ABSOLUTE_POS, ABSOLUTE_POS + stride, … so a
    // launch narrower than `n` still covers every object (T-7.1-01). `ABSOLUTE_POS` and
    // `CUBE_COUNT` are `usize` intrinsics; `CUBE_DIM` is `u32` — cast it once to keep
    // the stride arithmetic in `usize`.
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut i = ABSOLUTE_POS;
    while i < n {
        // Bounds guard (T-7.1-01); `indices` is length n, indexed directly like the
        // elementwise kernels (`approx[ABSOLUTE_POS]`). The bin/object VALUES are `u32`
        // (from the `&Array<u32>` inputs); cast them to `usize` for the index math.
        let obj = indices[i] as usize;
        let d = der1[obj];
        let w = weight[obj];

        // For each feature, read the object's quantized bin and atomic-merge both
        // channels into the global histogram at the FROZEN interleaved index.
        let mut feature = 0usize;
        while feature < n_features_usize {
            // The non-binary bin is read RAW — intentionally NOT masked (WR-01). Masking
            // an up-to-8-bit value to 8 bits is a no-op, so it cannot be the kernel's
            // safety net; instead this family relies on the host-side range guard in
            // `launch_pointwise_hist2_into` (CR-01), which rejects any `cindex` value
            // >= n_bins BEFORE launch. The half-byte/binary kernels mask (`& 15`/`& 1`)
            // because their nibble/bit decomposition makes the mask structurally
            // meaningful; here it would not be. The divergence is deliberate, not an
            // oversight.
            let bin = cindex[feature * n + obj] as usize;
            let cell = (feature * n_bins + bin) * 2usize;
            // channel 0 = Σ der1, channel 1 = Σ weight (in-kernel atomic, D-03).
            bin_sums[cell].fetch_add(d);
            bin_sums[cell + 1usize].fetch_add(w);
            feature += 1usize;
        }

        i += stride;
    }
}

/// Comptime number of half-byte (4-bit) bins: `1 << 4 = 16`. This is the half-byte
/// family's FIXED histogram line size (NOT a comptime `bits` arg like the non-binary
/// kernel) — upstream `pointwise_hist2_half_byte_template.cuh`'s `TPointHistHalfByte`
/// is structurally a 16-entry working histogram (`offset = (bins >> ...) & 15`,
/// `HIST_SIZE = 16 * BlockSize`). It is a bin COUNT, NOT a wave/warp-size literal in
/// any stride (D-09). Held `usize` for the (feature, bin) index arithmetic.
pub(crate) const HALF_BYTE_BINS: usize = 16;

/// 2-channel pointwise histogram fill — the **half-byte (4-bit)** `ComputeHist2HalfByte`
/// analog (Phase 7.3, GPU-01 histogram slice; D-7.3-01..05). A SEPARATE `#[cube]`
/// kernel family from [`pointwise_hist2_nonbinary_kernel`] (D-7.3-02 — half-byte is
/// structurally distinct, NOT a comptime `bits` case of the non-binary kernel): it
/// mirrors upstream `pointwise_hist2_half_byte_template.cuh`'s `TPointHistHalfByte`,
/// whose working histogram is a FIXED 16-entry (4-bit) line (`HIST_SIZE = 16 *
/// BlockSize`, `offset = (bins >> ...) & 15`) — distinct from the non-binary kernel's
/// runtime-`bits`-driven `1 << bits` line size and its per-object direct-global merge.
///
/// For every (feature, bin) it accumulates two interleaved channels: channel 0 = Σ der1
/// ("target"), channel 1 = Σ weight, written into the global `bin_sums` buffer at the
/// SAME FROZEN layout index as the non-binary kernel (the seam stays byte-identical —
/// D-7.3-01 / Pitfall 2), with `n_bins = HALF_BYTE_BINS = 16`:
///
/// ```text
/// index(feature, bin, channel) = (feature * 16 + bin) * 2 + channel
/// ```
///
/// (the single-tree collapse of `ShiftPartAndBinSumsPtr`, `part = fold = 0`,
/// `FirstFoldIndex = 0` — see the module doc of `kernels/pointwise_hist.rs`).
///
/// # Structurally-distinct half-byte layout (D-7.3-02)
///
/// The half-byte structural distinctness vs the non-binary family is encoded by the
/// FIXED comptime 16-bin line and the half-byte 4-bit bin DECOMPOSITION: upstream packs
/// several half-byte features in one `ci` word and extracts each 4-bit nibble via
/// `(bins >> (28 - 4*i)) & 15`; here each feature's quantized bin is masked to its 4
/// bits (`bin & 15`, mirroring the `& 15` nibble select) and routed to one of the 16
/// half-byte bins. The line size is the comptime [`HALF_BYTE_BINS`] (`1 << 4`) — NOT a
/// runtime `bits` arg like [`pointwise_hist2_nonbinary_kernel`] (`1 << bits`, bits in
/// 5..=8) — so this is a genuinely SEPARATE kernel family (the plan's D-7.3-02
/// requirement), not a comptime case of the non-binary kernel. The nibble mask and the
/// 16-bin line are the half-byte family's structural signature.
///
/// # In-kernel atomic merge (D-03 / D-7.3-03)
///
/// `bin_sums: &Array<Atomic<F>>` is the GLOBAL histogram; each thread `fetch_add`s its
/// per-object contribution directly into the global cell — the genuine D-03 in-kernel
/// atomic merge (the same direct-global-atomic MVP fill Plan A chose for the non-binary
/// family; upstream's per-block shared-memory `TPointHistHalfByte` working histogram +
/// the `BLOCKS_PER_FEATURE > 1 ? atomicAdd : WriteThrough` merge guard is the additive
/// PERFORMANCE refinement over this same atomic-merge STRUCTURE — it pre-reduces within
/// a block — reserved as a follow-up, [`HIST_SHMEM`] kept for it). Because many threads
/// contribute to the same cell, the cross-thread accumulation ORDER is
/// non-deterministic — the accepted D-03 source of run-to-run float-order variance,
/// REPORTED (not signed off) by the `kernels::pointwise_hist` oracle (the GPU-06
/// epsilon is 7.6's job).
///
/// # Wave-size policy (D-09)
///
/// The per-object loop strides by the TOTAL thread count `CUBE_COUNT * CUBE_DIM` (a
/// grid-stride loop) — derived from the launch topology intrinsics, NEVER a literal
/// 32/64. No `& 31`/`tiled_partition<32>`/`512 * (threadIdx/32)` warp-tile construct
/// appears (upstream's `SliceOffset`/`SyncTile` warp partitioning is replaced by the
/// wave-agnostic grid-stride loop + global atomic merge). The bin index comes from the
/// masked 4-bit `cindex` value, not a warp-lane partition. Generic over `F: Float`
/// (AGENTS.md generics-float). Every device read is under a POSITION bounds guard (`i <
/// indices.len()`); the VALUE ranges (`indices[i]` object id, `cindex[...]` bin) are
/// validated HOST-SIDE in `launch_pointwise_hist2_into` (CR-01) before launch — the
/// 4-bit nibble mask (`& 15`) additionally bounds the bin structurally here.
/// if-as-STATEMENT only (CubeCL conditionals manual).
///
/// `der1`/`weight` are length `n` (per object, object order). `cindex` is the quantized
/// bin matrix laid out feature-major (`cindex[feature * n + obj]`). `indices` (length
/// `n`) is the object visiting order. `n_features` is a runtime `u32` scalar; the
/// 16-bin line size is the comptime [`HALF_BYTE_BINS`] (no runtime bit-count branch).
#[cube(launch)]
pub fn pointwise_hist2_half_byte_kernel<F: Float>(
    der1: &Array<F>,
    weight: &Array<F>,
    cindex: &Array<u32>,
    indices: &Array<u32>,
    bin_sums: &Array<Atomic<F>>,
    n_features: u32,
) {
    // FIXED 16-bin (4-bit) line — the comptime HALF_BYTE_BINS (NOT a runtime `bits`
    // value): the structural mark of the half-byte family (`TPointHistHalfByte` is a
    // 16-entry working histogram). Held `usize` for the (feature, bin) index math.
    let n_bins = comptime!(HALF_BYTE_BINS);
    // 4-bit nibble mask (upstream `& 15` half-byte bin select). Applied to the raw
    // `cindex` `u32` value before the `usize` index cast.
    let nibble_mask = comptime!((HALF_BYTE_BINS as u32) - 1u32);

    let n = indices.len();
    let n_features_usize = n_features as usize;

    // Grid-stride loop over the object-visiting order; stride = total thread count
    // (CUBE_COUNT * CUBE_DIM) — topology-derived, never a literal 32/64 (D-09).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut i = ABSOLUTE_POS;
    while i < n {
        let obj = indices[i] as usize;
        let d = der1[obj];
        let w = weight[obj];

        let mut feature = 0usize;
        while feature < n_features_usize {
            // Half-byte 4-bit nibble select (upstream `(bins >> ...) & 15`): mask the
            // quantized bin to its 4 bits, routing it to one of the 16 half-byte bins.
            let bin = (cindex[feature * n + obj] & nibble_mask) as usize;
            let cell = (feature * n_bins + bin) * 2usize;
            // channel 0 = Σ der1, channel 1 = Σ weight (in-kernel atomic merge, D-03)
            // into the FROZEN binSums layout (byte-identical to the non-binary seam).
            bin_sums[cell].fetch_add(d);
            bin_sums[cell + 1usize].fetch_add(w);
            feature += 1usize;
        }

        i += stride;
    }
}

/// Comptime number of binary (1-bit) bins: `1 << 1 = 2`. This is the binary family's
/// FIXED histogram line size (NOT a comptime `bits` arg like the non-binary kernel, NOR
/// the half-byte family's 16). Upstream `pointwise_hist2_binary.cu`'s `ComputeHist2Binary`
/// reuses the `TPointHistHalfByte` working histogram but each binary feature contributes
/// to exactly TWO buckets (the split bit 0/1), and the result is a 2-channel sum per
/// feature × bit. It is a bin COUNT, NOT a wave/warp-size literal in any stride (D-09).
/// Held `usize` for the (feature, bin) index arithmetic.
pub(crate) const BINARY_BINS: usize = 2;

/// 2-channel pointwise histogram fill — the **binary (1-bit)** `ComputeHist2Binary`
/// analog (Phase 7.3, GPU-01 histogram slice; D-7.3-01..05). A SEPARATE `#[cube]` kernel
/// family from BOTH [`pointwise_hist2_nonbinary_kernel`] and
/// [`pointwise_hist2_half_byte_kernel`] (D-7.3-02 — binary is structurally distinct, NOT
/// a comptime `bits` case of the non-binary kernel and NOT the half-byte kernel): it
/// mirrors upstream `pointwise_hist2_binary.cu`'s `ComputeHist2Binary`, whose binary
/// features decompose to exactly 2 buckets (the split bit 0/1) — a FIXED 2-entry
/// (1-bit) histogram line per feature, distinct from the half-byte's 16-entry line and
/// the non-binary kernel's runtime-`bits`-driven `1 << bits` line size.
///
/// For every (feature, bin) it accumulates two interleaved channels: channel 0 = Σ der1
/// ("target"), channel 1 = Σ weight, written into the global `bin_sums` buffer at the
/// SAME FROZEN layout index as the non-binary and half-byte kernels (the seam stays
/// byte-identical — D-7.3-01 / Pitfall 2), with `n_bins = BINARY_BINS = 2`:
///
/// ```text
/// index(feature, bin, channel) = (feature * 2 + bin) * 2 + channel
/// ```
///
/// (the single-tree collapse of `ShiftPartAndBinSumsPtr`, `part = fold = 0`,
/// `FirstFoldIndex = 0` — see the module doc of `kernels/pointwise_hist.rs`).
///
/// # Structurally-distinct binary layout (D-7.3-02)
///
/// The binary structural distinctness vs the non-binary and half-byte families is
/// encoded by the FIXED comptime 2-bin line and the binary 1-bit bin DECOMPOSITION:
/// upstream's `ComputeSplitPropertiesBImpl` routes each binary feature's split bit
/// (`fMask = 1 << (3 - (fid & 3))`) into one of two channels; here each feature's
/// quantized bin is masked to its low bit (`bin & 1`, the 1-bit split select) and routed
/// to one of the 2 binary bins. The line size is the comptime [`BINARY_BINS`] (`1 << 1`)
/// — NOT a runtime `bits` arg like [`pointwise_hist2_nonbinary_kernel`] (`1 << bits`,
/// bits in 5..=8), NOR the half-byte's fixed 16 — so this is a genuinely SEPARATE kernel
/// family (the plan's D-7.3-02 requirement). The 1-bit mask and the 2-bin line are the
/// binary family's structural signature.
///
/// # In-kernel atomic merge (D-03 / D-7.3-03)
///
/// `bin_sums: &Array<Atomic<F>>` is the GLOBAL histogram; each thread `fetch_add`s its
/// per-object contribution directly into the global cell — the genuine D-03 in-kernel
/// atomic merge (the same direct-global-atomic MVP fill Plan A chose for the non-binary
/// family and Plan C for half-byte; upstream's per-block shared-memory `TPointHistHalfByte`
/// working histogram + the `BlocksPerFeatureCount > 1 ? atomicAdd : WriteThrough` merge
/// guard is the additive PERFORMANCE refinement over this same atomic-merge STRUCTURE —
/// it pre-reduces within a block — reserved as a follow-up, [`HIST_SHMEM`] kept for it).
/// Because many threads contribute to the same cell, the cross-thread accumulation ORDER
/// is non-deterministic — the accepted D-03 source of run-to-run float-order variance,
/// REPORTED (not signed off) by the `kernels::pointwise_hist` oracle (the GPU-06 epsilon
/// is 7.6's job).
///
/// # Wave-size policy (D-09)
///
/// The per-object loop strides by the TOTAL thread count `CUBE_COUNT * CUBE_DIM` (a
/// grid-stride loop) — derived from the launch topology intrinsics, NEVER a literal
/// 32/64. No `& 31`/`threadIdx & 1`/`tiled_partition<32>` warp-tile/lane construct
/// appears (upstream's `threadIdx.x & 1` channel select + warp partitioning is replaced
/// by the wave-agnostic grid-stride loop + global atomic merge). The bin index comes from
/// the masked 1-bit `cindex` value, not a warp-lane partition. Generic over `F: Float`
/// (AGENTS.md generics-float). Every device read is under a POSITION bounds guard (`i <
/// indices.len()`); the VALUE ranges (`indices[i]` object id, `cindex[...]` bin) are
/// validated HOST-SIDE in `launch_pointwise_hist2_into` (CR-01) before launch — the
/// 1-bit mask (`& 1`) additionally bounds the bin structurally here. if-as-STATEMENT
/// only (CubeCL conditionals manual).
///
/// `der1`/`weight` are length `n` (per object, object order). `cindex` is the quantized
/// bin matrix laid out feature-major (`cindex[feature * n + obj]`). `indices` (length
/// `n`) is the object visiting order. `n_features` is a runtime `u32` scalar; the 2-bin
/// line size is the comptime [`BINARY_BINS`] (no runtime bit-count branch).
#[cube(launch)]
pub fn pointwise_hist2_binary_kernel<F: Float>(
    der1: &Array<F>,
    weight: &Array<F>,
    cindex: &Array<u32>,
    indices: &Array<u32>,
    bin_sums: &Array<Atomic<F>>,
    n_features: u32,
) {
    // FIXED 2-bin (1-bit) line — the comptime BINARY_BINS (NOT a runtime `bits` value,
    // NOR the half-byte's 16): the structural mark of the binary family. Held `usize` for
    // the (feature, bin) index math.
    let n_bins = comptime!(BINARY_BINS);
    // 1-bit split mask (upstream's binary split bit select). Applied to the raw `cindex`
    // `u32` value before the `usize` index cast.
    let bit_mask = comptime!((BINARY_BINS as u32) - 1u32);

    let n = indices.len();
    let n_features_usize = n_features as usize;

    // Grid-stride loop over the object-visiting order; stride = total thread count
    // (CUBE_COUNT * CUBE_DIM) — topology-derived, never a literal 32/64 (D-09).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut i = ABSOLUTE_POS;
    while i < n {
        let obj = indices[i] as usize;
        let d = der1[obj];
        let w = weight[obj];

        let mut feature = 0usize;
        while feature < n_features_usize {
            // Binary 1-bit split select (upstream's split bit): mask the quantized bin to
            // its low bit, routing it to one of the 2 binary bins.
            let bin = (cindex[feature * n + obj] & bit_mask) as usize;
            let cell = (feature * n_bins + bin) * 2usize;
            // channel 0 = Σ der1, channel 1 = Σ weight (in-kernel atomic merge, D-03)
            // into the FROZEN binSums layout (byte-identical to the non-binary/half-byte
            // seam).
            bin_sums[cell].fetch_add(d);
            bin_sums[cell + 1usize].fetch_add(w);
            feature += 1usize;
        }

        i += stride;
    }
}

/// 4-channel WEIGHT-ONLY pairwise histogram fill — the general **one-byte non-binary**
/// `ComputePairwiseHistogramOneByte{5,6,7}Bits` analog (Phase 7.4, GPU-01 histogram
/// slice; D-7.4-01..05). The pairwise SIBLING of [`pointwise_hist2_nonbinary_kernel`]:
/// where the pointwise kernel accumulates per SINGLE object into a 2-channel
/// (Σ der1, Σ weight) histogram, this kernel accumulates per OBJECT PAIR `(oi, oj)`
/// (upstream's `uint2* pairs`, encoded here as two parallel `u32` arrays per D-7.4-03
/// discretion) into a **4-channel weight-only** histogram (`histId in {0,1,2,3}`).
///
/// The bit-count is carried as a `#[comptime] bits` in {5,6,7} (SAME mechanism as the
/// shipped pointwise kernel's `#[comptime] bits`), so `n_bins = 1 << bits` is resolved
/// at JIT time with no runtime bit-count branch. `#[comptime] one_hot` selects the
/// `Compare` predicate at JIT time (non-one-hot `(bin1 >= bin2) == flag`, one-hot
/// `bin1 == bin2`); the one-hot overlay is THREADED now but exercised only by Plan E.
///
/// # FROZEN 4-channel WEIGHT-ONLY layout (D-7.4-03 / Pitfall 2)
///
/// For each (feature, bin) the kernel atomic-merges `pair_weight` into the four channels
/// selected by the per-pair `Compare -> histId` mapping (distilled from upstream
/// `pairwise_hist_one_byte_5bit.cuh::AddPair` + the `4 * (maxFoldCount * f + fold) +
/// histId` merge, with the warp-tile distribution reduced to its accumulation semantics
/// — A6 / Pitfall 4; the tile is perf, not semantics):
///
/// ```text
/// index(feature, bin, histId) = (feature * n_bins + bin) * 4 + histId,  histId in {0,1,2,3}
/// non-one-hot, pair (b1, b2, w):  ge = (b1>=b2), gt = (b1>b2)
///   bin b1, histId 2*ge+0 += w;   bin b1, histId 2*gt+0 += w;
///   bin b2, histId 2*ge+1 += w;   bin b2, histId 2*gt+1 += w;
/// ```
///
/// The buffer length is `n_features * n_bins * 4` (NEVER `* 2` — Pitfall 2). The
/// `part = fold = 0` single-tree collapse; the multi-part `ShiftPartAndBinSumsPtr`
/// offset is a 7.5 forward dependency.
///
/// # Stride discipline (Pitfall 3) + bounds (D-09)
///
/// `pair_i`/`pair_j` hold OBJECT ids; the cindex stride is over OBJECTS (`n_objects`, a
/// runtime scalar), NOT `n_pairs` — `bin = cindex[feature * n_objects + obj]`. The
/// grid-stride is the total thread count (`CUBE_COUNT * CUBE_DIM`), never a literal
/// 32/64 (D-09). Bin/object VALUE ranges are validated HOST-SIDE in
/// `launch_pairwise_hist_into` (T-07.4-01/02) before launch. Generic over `F: Float`
/// (AGENTS.md generics-float). if-as-STATEMENT only (CubeCL conditionals manual).
#[cube(launch)]
pub fn pairwise_hist_nonbinary_kernel<F: Float>(
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    pair_weight: &Array<F>,
    cindex: &Array<u32>,
    bin_sums: &Array<Atomic<F>>,
    n_features: u32,
    n_objects: u32,
    #[comptime] bits: u32,
    #[comptime] one_hot: bool,
) {
    // n_bins = 1 << bits (comptime). Held `usize` for the (feature, bin) index math.
    let n_bins = comptime!((1u32 << bits) as usize);
    // n_pairs (the loop bound) is the per-pair value count; the cindex stride is
    // n_objects (Pitfall 3 — NEVER n_pairs).
    let n_pairs = pair_weight.len();
    let n_features_usize = n_features as usize;
    let n_objects_usize = n_objects as usize;

    // Grid-stride loop over PAIRS; stride = total thread count (CUBE_COUNT * CUBE_DIM)
    // — topology-derived, never a literal 32/64 (D-09). Each unit processes pairs
    // ABSOLUTE_POS, ABSOLUTE_POS + stride, … so a launch narrower than n_pairs still
    // covers every pair (idle-guard `p < n_pairs`).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut p = ABSOLUTE_POS;
    while p < n_pairs {
        let oi = pair_i[p] as usize;
        let oj = pair_j[p] as usize;
        let w = pair_weight[p];

        let mut feature = 0usize;
        while feature < n_features_usize {
            // Two bins per (pair, feature): the quantized bins of the two paired objects,
            // read RAW (range-guarded host-side, like the pointwise non-binary kernel).
            // cindex stride is OBJECTS (Pitfall 3).
            let b1 = cindex[feature * n_objects_usize + oi] as usize;
            let b2 = cindex[feature * n_objects_usize + oj] as usize;

            // The per-pair Compare -> histId channel selection (the genuinely-new logic,
            // D-7.4-02). histId = 2 * isGe + isSecondBin; isSecondBin = 0 for b1, 1 for
            // b2. if-as-STATEMENT only (init the selector vars, overwrite in branches).
            let base = (feature * n_bins) * 4usize;
            if one_hot {
                // One-hot Compare = (bin1 == bin2); both flag passes coincide on the same
                // slot. Threaded now; refined by Plan E.
                let mut is_ge = 1usize; // predicate false (b1 != b2) -> Ge slot
                if b1 == b2 {
                    is_ge = 0usize;
                }
                let cell1 = base + b1 * 4usize + 2usize * is_ge;
                let cell2 = base + b2 * 4usize + 2usize * is_ge + 1usize;
                bin_sums[cell1].fetch_add(w);
                bin_sums[cell1].fetch_add(w);
                bin_sums[cell2].fetch_add(w);
                bin_sums[cell2].fetch_add(w);
            } else {
                // Non-one-hot: ge = (b1>=b2), gt = (b1>b2). The two flag-collapsed writes
                // per bin land in histId 2*ge+isSecondBin and 2*gt+isSecondBin.
                let mut ge = 0usize;
                if b1 >= b2 {
                    ge = 1usize;
                }
                let mut gt = 0usize;
                if b1 > b2 {
                    gt = 1usize;
                }
                // bin b1 (isSecondBin = 0)
                let b1_base = base + b1 * 4usize;
                bin_sums[b1_base + 2usize * ge].fetch_add(w);
                bin_sums[b1_base + 2usize * gt].fetch_add(w);
                // bin b2 (isSecondBin = 1)
                let b2_base = base + b2 * 4usize;
                bin_sums[b2_base + 2usize * ge + 1usize].fetch_add(w);
                bin_sums[b2_base + 2usize * gt + 1usize].fetch_add(w);
            }

            feature += 1usize;
        }

        p += stride;
    }
}

/// The 8-bit-atomics pairwise histogram fill — upstream's structurally DISTINCT
/// `pairwise_hist_one_byte_8bit_atomics.cuh::ComputePairwiseHistogramOneByte8BitAtomics`
/// family (D-7.4-02). At 8 bits a 256-bin x 4-channel histogram line does NOT fit the
/// per-block shared-memory budget the sub-8-bit paths use, so upstream accumulates via
/// TRUE GLOBAL ATOMICS (a per-thread `CachedBinsLeq/Ge` run flushed with global
/// `atomicAdd`). This is kept a SEPARATE `#[cube]` symbol with a SEPARATE launch arm to
/// preserve structural parity with upstream's separate `.cu` — even though the MVP body
/// is exactly the non-binary kernel with `bits` fixed at 8 and `n_bins = 256`, ALWAYS
/// using the direct per-pair global `Atomic<F>::fetch_add` the shipped 7.3 MVP already
/// uses for every width (RESEARCH Pattern 3 / A2). Upstream's per-thread cached-bin run
/// is a documented PERF FOLLOW-UP over the SAME atomic structure, not semantics.
///
/// # FROZEN 4-channel WEIGHT-ONLY layout (reused unchanged from Plan A — D-7.4-03)
///
/// The 8-bit family reuses the Plan A FROZEN 4-channel weight-only `binSums` layout
/// VERBATIM (never `* 2` — Pitfall 2); the only difference from
/// [`pairwise_hist_nonbinary_kernel`] is the comptime `n_bins = 256` (so the buffer
/// length is `n_features * 256 * 4`) and that this family ALWAYS uses direct global
/// atomics (the sub-8-bit shared-mem pre-reduce never applies here):
///
/// ```text
/// index(feature, bin, histId) = (feature * 256 + bin) * 4 + histId,  histId in {0,1,2,3}
/// non-one-hot, pair (b1, b2, w):  ge = (b1>=b2), gt = (b1>b2)
///   bin b1, histId 2*ge+0 += w;   bin b1, histId 2*gt+0 += w;
///   bin b2, histId 2*ge+1 += w;   bin b2, histId 2*gt+1 += w;
/// ```
///
/// # Stride discipline (Pitfall 3) + bounds (D-09)
///
/// `pair_i`/`pair_j` hold OBJECT ids; the cindex stride is over OBJECTS (`n_objects`, a
/// runtime scalar), NOT `n_pairs` — `bin = cindex[feature * n_objects + obj]`. The
/// grid-stride is the total thread count (`CUBE_COUNT * CUBE_DIM`), never a literal
/// 32/64 (D-09). Bin/object VALUE ranges are validated HOST-SIDE in
/// `launch_pairwise_hist_8bit_into` (T-07.4-07/08) before launch. Generic over
/// `F: Float` (AGENTS.md generics-float). if-as-STATEMENT only (CubeCL conditionals
/// manual).
#[cube(launch)]
pub fn pairwise_hist_8bit_atomics_kernel<F: Float>(
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    pair_weight: &Array<F>,
    cindex: &Array<u32>,
    bin_sums: &Array<Atomic<F>>,
    n_features: u32,
    n_objects: u32,
    #[comptime] one_hot: bool,
) {
    // n_bins fixed at 256 (the 8-bit-atomics line size — comptime). Held `usize` for the
    // (feature, bin) index math.
    let n_bins = comptime!(256usize);
    // n_pairs (the loop bound) is the per-pair value count; the cindex stride is
    // n_objects (Pitfall 3 — NEVER n_pairs).
    let n_pairs = pair_weight.len();
    let n_features_usize = n_features as usize;
    let n_objects_usize = n_objects as usize;

    // Grid-stride loop over PAIRS; stride = total thread count (CUBE_COUNT * CUBE_DIM)
    // — topology-derived, never a literal 32/64 (D-09). Each unit processes pairs
    // ABSOLUTE_POS, ABSOLUTE_POS + stride, … so a launch narrower than n_pairs still
    // covers every pair (idle-guard `p < n_pairs`).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut p = ABSOLUTE_POS;
    while p < n_pairs {
        let oi = pair_i[p] as usize;
        let oj = pair_j[p] as usize;
        let w = pair_weight[p];

        let mut feature = 0usize;
        while feature < n_features_usize {
            // Two bins per (pair, feature): the quantized bins of the two paired objects,
            // read RAW (range-guarded host-side). cindex stride is OBJECTS (Pitfall 3).
            let b1 = cindex[feature * n_objects_usize + oi] as usize;
            let b2 = cindex[feature * n_objects_usize + oj] as usize;

            // Direct global atomics ALWAYS (the 8-bit line exceeds the shared-mem budget —
            // D-7.4-02/04). The per-pair Compare -> histId channel selection is identical
            // to the non-binary kernel. if-as-STATEMENT only.
            let base = (feature * n_bins) * 4usize;
            if one_hot {
                // One-hot Compare = (bin1 == bin2). Threaded now; refined by Plan E.
                let mut is_ge = 1usize; // predicate false (b1 != b2) -> Ge slot
                if b1 == b2 {
                    is_ge = 0usize;
                }
                let cell1 = base + b1 * 4usize + 2usize * is_ge;
                let cell2 = base + b2 * 4usize + 2usize * is_ge + 1usize;
                bin_sums[cell1].fetch_add(w);
                bin_sums[cell1].fetch_add(w);
                bin_sums[cell2].fetch_add(w);
                bin_sums[cell2].fetch_add(w);
            } else {
                // Non-one-hot: ge = (b1>=b2), gt = (b1>b2). The two flag-collapsed writes
                // per bin land in histId 2*ge+isSecondBin and 2*gt+isSecondBin.
                let mut ge = 0usize;
                if b1 >= b2 {
                    ge = 1usize;
                }
                let mut gt = 0usize;
                if b1 > b2 {
                    gt = 1usize;
                }
                // bin b1 (isSecondBin = 0)
                let b1_base = base + b1 * 4usize;
                bin_sums[b1_base + 2usize * ge].fetch_add(w);
                bin_sums[b1_base + 2usize * gt].fetch_add(w);
                // bin b2 (isSecondBin = 1)
                let b2_base = base + b2 * 4usize;
                bin_sums[b2_base + 2usize * ge + 1usize].fetch_add(w);
                bin_sums[b2_base + 2usize * gt + 1usize].fetch_add(w);
            }

            feature += 1usize;
        }

        p += stride;
    }
}

/// The half-byte (4-bit, 16-bin) pairwise histogram fill — upstream's structurally
/// DISTINCT `pairwise_hist_half_byte.cu::ComputePairwiseHistogramHalfByte` family
/// (D-7.4-02). The half-byte path packs several 4-bit features per `ci` word and extracts
/// each via `(bins >> (28 - 4*i)) & 15`; its working histogram is a FIXED 16-entry (4-bit)
/// line — structurally distinct from the non-binary kernel's runtime-`bits`-driven
/// `1 << bits` line. It is kept a SEPARATE `#[cube]` symbol with a SEPARATE launch arm to
/// preserve structural parity with upstream's separate `.cu` — the MVP body is the
/// non-binary kernel with `n_bins` fixed at the comptime [`HALF_BYTE_BINS`] (16) and the
/// read bins masked to the nibble (`& 15`, the shipped 7.3 half-byte precedent at
/// [`pointwise_hist2_half_byte_kernel`]).
///
/// # No one-hot overlay (RESEARCH Pattern 2)
///
/// The half-byte family takes NO `one_hot` arg: upstream has no
/// `pairwise_hist_half_byte_one_hot.cu`. This kernel ALWAYS uses the non-one-hot `Compare`
/// predicate `(bin1 >= bin2) == flag` (the flag-collapsed `(ge, gt)` writes), exactly the
/// `else`-branch of [`pairwise_hist_nonbinary_kernel`].
///
/// # FROZEN 4-channel WEIGHT-ONLY layout (reused unchanged from Plan A — D-7.4-03)
///
/// The half-byte family reuses the Plan A FROZEN 4-channel weight-only `binSums` layout
/// VERBATIM (never `* 2` — Pitfall 2); the only differences from
/// [`pairwise_hist_nonbinary_kernel`] are the comptime `n_bins = HALF_BYTE_BINS` (16, so
/// the buffer length is `n_features * 16 * 4`), the nibble mask on the read bins, and the
/// absence of the one-hot branch:
///
/// ```text
/// index(feature, bin, histId) = (feature * 16 + bin) * 4 + histId,  histId in {0,1,2,3}
/// non-one-hot, pair (b1, b2, w):  ge = (b1>=b2), gt = (b1>b2)
///   bin b1, histId 2*ge+0 += w;   bin b1, histId 2*gt+0 += w;
///   bin b2, histId 2*ge+1 += w;   bin b2, histId 2*gt+1 += w;
/// ```
///
/// # Stride discipline (Pitfall 3) + bounds (D-09)
///
/// `pair_i`/`pair_j` hold OBJECT ids; the cindex stride is over OBJECTS (`n_objects`, a
/// runtime scalar), NOT `n_pairs` — `bin = cindex[feature * n_objects + obj]`. The
/// grid-stride is the total thread count (`CUBE_COUNT * CUBE_DIM`), never a literal 32/64
/// (D-09 — the 16-bin line is a bin COUNT, not a warp literal). The nibble mask (`& 15`)
/// additionally bounds the bin into `0..16` structurally; Bin/object VALUE ranges are also
/// validated HOST-SIDE in `launch_pairwise_hist_half_byte_into` before launch. Generic
/// over `F: Float` (AGENTS.md generics-float). if-as-STATEMENT only (CubeCL conditionals
/// manual).
#[cube(launch)]
pub fn pairwise_hist_half_byte_kernel<F: Float>(
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    pair_weight: &Array<F>,
    cindex: &Array<u32>,
    bin_sums: &Array<Atomic<F>>,
    n_features: u32,
    n_objects: u32,
) {
    // FIXED 16-bin (4-bit) line — the comptime HALF_BYTE_BINS (NOT a runtime `bits` value):
    // the structural mark of the half-byte family. Held `usize` for the (feature, bin)
    // index math.
    let n_bins = comptime!(HALF_BYTE_BINS);
    // 4-bit nibble mask (upstream `& 15` half-byte bin select). Applied to the raw `cindex`
    // `u32` value before the `usize` index cast.
    let nibble_mask = comptime!((HALF_BYTE_BINS as u32) - 1u32);
    // n_pairs (the loop bound) is the per-pair value count; the cindex stride is n_objects
    // (Pitfall 3 — NEVER n_pairs).
    let n_pairs = pair_weight.len();
    let n_features_usize = n_features as usize;
    let n_objects_usize = n_objects as usize;

    // Grid-stride loop over PAIRS; stride = total thread count (CUBE_COUNT * CUBE_DIM) —
    // topology-derived, never a literal 32/64 (D-09). Each unit processes pairs
    // ABSOLUTE_POS, ABSOLUTE_POS + stride, … so a launch narrower than n_pairs still covers
    // every pair (idle-guard `p < n_pairs`).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut p = ABSOLUTE_POS;
    while p < n_pairs {
        let oi = pair_i[p] as usize;
        let oj = pair_j[p] as usize;
        let w = pair_weight[p];

        let mut feature = 0usize;
        while feature < n_features_usize {
            // Two bins per (pair, feature): the quantized bins of the two paired objects,
            // masked to the 4-bit nibble (upstream `(bins >> ...) & 15`). cindex stride is
            // OBJECTS (Pitfall 3).
            let b1 = (cindex[feature * n_objects_usize + oi] & nibble_mask) as usize;
            let b2 = (cindex[feature * n_objects_usize + oj] & nibble_mask) as usize;

            // Non-one-hot Compare -> histId channel selection (the half-byte family has no
            // one-hot overlay). ge = (b1>=b2), gt = (b1>b2); the two flag-collapsed writes
            // per bin land in histId 2*ge+isSecondBin and 2*gt+isSecondBin. if-as-STATEMENT
            // only.
            let base = (feature * n_bins) * 4usize;
            let mut ge = 0usize;
            if b1 >= b2 {
                ge = 1usize;
            }
            let mut gt = 0usize;
            if b1 > b2 {
                gt = 1usize;
            }
            // bin b1 (isSecondBin = 0)
            let b1_base = base + b1 * 4usize;
            bin_sums[b1_base + 2usize * ge].fetch_add(w);
            bin_sums[b1_base + 2usize * gt].fetch_add(w);
            // bin b2 (isSecondBin = 1)
            let b2_base = base + b2 * 4usize;
            bin_sums[b2_base + 2usize * ge + 1usize].fetch_add(w);
            bin_sums[b2_base + 2usize * gt + 1usize].fetch_add(w);

            feature += 1usize;
        }

        p += stride;
    }
}

/// The binary (1-bit, 2-bin) pairwise histogram fill — upstream's structurally DISTINCT
/// `pairwise_hist_binary.cu::ComputePairwiseHistogramBinary` family (D-7.4-02). The binary
/// path packs several 1-bit features per `ci` word; upstream extracts a 4-bit nibble and
/// builds the 2x2 cross-product `(invBin1 & invBin2) | ((invBin1 & bin2) << 8) | ((bin1 &
/// invBin2) << 16) | ((bin1 & bin2) << 24)` over the warp tile. Reduced to per-pair
/// accumulation semantics that 2x2 `(i leq/ge) x (j leq/ge)` decomposition is EXACTLY the
/// non-one-hot `Compare(bin1,bin2)->histId` predicate the other families use (the self-oracle
/// validates this bit-exact). Its working histogram is a FIXED 2-entry (1-bit) line — a bin
/// COUNT, NOT a warp literal (D-09). It is kept a SEPARATE `#[cube]` symbol with a SEPARATE
/// launch arm to preserve structural parity with upstream's separate `.cu` — the MVP body is
/// the non-binary kernel with `n_bins` fixed at `2` and the read bins masked to the bit
/// (`& 1`, the shipped 7.3 binary precedent at `pointwise_hist2_binary_kernel`).
///
/// # No one-hot overlay (RESEARCH Pattern 2)
///
/// The binary family takes NO `one_hot` arg: upstream has no `pairwise_hist_binary_one_hot.cu`.
/// This kernel ALWAYS uses the non-one-hot `Compare` predicate `(bin1 >= bin2) == flag` (the
/// flag-collapsed `(ge, gt)` writes), exactly the `else`-branch of
/// [`pairwise_hist_nonbinary_kernel`].
///
/// # FROZEN 4-channel WEIGHT-ONLY layout (reused unchanged from Plan A — D-7.4-03)
///
/// The binary family reuses the Plan A FROZEN 4-channel weight-only `binSums` layout VERBATIM
/// (never `* 2` — Pitfall 2); the only differences from [`pairwise_hist_nonbinary_kernel`] are
/// the comptime `n_bins = 2` (so the buffer length is `n_features * 2 * 4`), the bit mask on
/// the read bins, and the absence of the one-hot branch:
///
/// ```text
/// index(feature, bin, histId) = (feature * 2 + bin) * 4 + histId,  histId in {0,1,2,3}
/// non-one-hot, pair (b1, b2, w):  ge = (b1>=b2), gt = (b1>b2)
///   bin b1, histId 2*ge+0 += w;   bin b1, histId 2*gt+0 += w;
///   bin b2, histId 2*ge+1 += w;   bin b2, histId 2*gt+1 += w;
/// ```
///
/// # Stride discipline (Pitfall 3) + bounds (D-09)
///
/// `pair_i`/`pair_j` hold OBJECT ids; the cindex stride is over OBJECTS (`n_objects`, a
/// runtime scalar), NOT `n_pairs` — `bin = cindex[feature * n_objects + obj]`. The grid-stride
/// is the total thread count (`CUBE_COUNT * CUBE_DIM`), never a literal 32/64 (D-09 — the
/// 2-bin line is a bin COUNT, not a warp literal). The bit mask (`& 1`) additionally bounds
/// the bin into `0..2` structurally; Bin/object VALUE ranges are also validated HOST-SIDE in
/// `launch_pairwise_hist_binary_into` before launch. Generic over `F: Float` (AGENTS.md
/// generics-float). if-as-STATEMENT only (CubeCL conditionals manual).
#[cube(launch)]
pub fn pairwise_hist_binary_kernel<F: Float>(
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    pair_weight: &Array<F>,
    cindex: &Array<u32>,
    bin_sums: &Array<Atomic<F>>,
    n_features: u32,
    n_objects: u32,
) {
    // FIXED 2-bin (1-bit) line — a bin COUNT, NOT a runtime `bits` value or warp literal
    // (D-09): the structural mark of the binary family. Held `usize` for the (feature, bin)
    // index math.
    let n_bins = comptime!(2usize);
    // 1-bit mask (upstream binary bin select). Applied to the raw `cindex` `u32` value before
    // the `usize` index cast.
    let bit_mask = comptime!(1u32);
    // n_pairs (the loop bound) is the per-pair value count; the cindex stride is n_objects
    // (Pitfall 3 — NEVER n_pairs).
    let n_pairs = pair_weight.len();
    let n_features_usize = n_features as usize;
    let n_objects_usize = n_objects as usize;

    // Grid-stride loop over PAIRS; stride = total thread count (CUBE_COUNT * CUBE_DIM) —
    // topology-derived, never a literal 32/64 (D-09). Each unit processes pairs ABSOLUTE_POS,
    // ABSOLUTE_POS + stride, … so a launch narrower than n_pairs still covers every pair
    // (idle-guard `p < n_pairs`).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut p = ABSOLUTE_POS;
    while p < n_pairs {
        let oi = pair_i[p] as usize;
        let oj = pair_j[p] as usize;
        let w = pair_weight[p];

        let mut feature = 0usize;
        while feature < n_features_usize {
            // Two bins per (pair, feature): the quantized bins of the two paired objects,
            // masked to the 1-bit value (upstream binary bin select). cindex stride is OBJECTS
            // (Pitfall 3).
            let b1 = (cindex[feature * n_objects_usize + oi] & bit_mask) as usize;
            let b2 = (cindex[feature * n_objects_usize + oj] & bit_mask) as usize;

            // Non-one-hot Compare -> histId channel selection (the binary family has no one-hot
            // overlay). ge = (b1>=b2), gt = (b1>b2); the two flag-collapsed writes per bin land
            // in histId 2*ge+isSecondBin and 2*gt+isSecondBin. This is the per-pair reduction
            // of upstream's 2x2 `(invBin1&invBin2)|...` cross-product. if-as-STATEMENT only.
            let base = (feature * n_bins) * 4usize;
            let mut ge = 0usize;
            if b1 >= b2 {
                ge = 1usize;
            }
            let mut gt = 0usize;
            if b1 > b2 {
                gt = 1usize;
            }
            // bin b1 (isSecondBin = 0)
            let b1_base = base + b1 * 4usize;
            bin_sums[b1_base + 2usize * ge].fetch_add(w);
            bin_sums[b1_base + 2usize * gt].fetch_add(w);
            // bin b2 (isSecondBin = 1)
            let b2_base = base + b2 * 4usize;
            bin_sums[b2_base + 2usize * ge + 1usize].fetch_add(w);
            bin_sums[b2_base + 2usize * gt + 1usize].fetch_add(w);

            feature += 1usize;
        }

        p += stride;
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
    // IN-01: no post-read `sync_cube()` here — the barrier this read depends on is the
    // trailing sync of the Hillis-Steele loop above; a barrier AFTER the read
    // synchronizes nothing relevant to `carry` and is a needless cube-wide barrier on
    // the scan hot path.

    let result = scanned + carry;
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = result;
    }
}

/// Launch-geometry cube width for the two-level [`full_scan`] (Plan 10-01, GPUT-16).
/// It equals [`BLOCK_REDUCE_SHMEM`] (a power-of-two width = the shared-mem allocation
/// each block-scan reuses); it is the launch-geometry const, NOT a wave/warp-size
/// literal in any reduction stride (the strides derive from `CUBE_DIM_X` / `PLANE_DIM`,
/// D-09).
pub(crate) const SCAN_CUBE_DIM: usize = BLOCK_REDUCE_SHMEM;

/// Per-block prefix scan that ALSO emits each block's total to `block_sums` — the
/// first phase of the cross-cube two-level [`full_scan`] (GPUT-16; RESEARCH Open Q2:
/// the single-cube [`block_scan_kernel`] carries no cross-cube offset). Each cube
/// independently scans its own `CUBE_DIM`-wide slice (exactly the `block_scan_kernel`
/// geometry) and writes `output[ABSOLUTE_POS]` = the WITHIN-BLOCK prefix (inclusive or
/// exclusive per the `#[comptime]` flag). Independently of that flag, the last VALID
/// unit of each block writes the block's INCLUSIVE total (`scanned_in_plane + carry`)
/// to `block_sums[CUBE_POS]`, so phase 2 can exclusive-scan those totals into per-block
/// offsets. The cross-cube offset add is phase 3 ([`add_block_offset_kernel`]).
///
/// Generic over `F: Float` (AGENTS.md generics-float). Every device read/write is under
/// an `ABSOLUTE_POS < input.len()` bounds guard (T-10-01); no `-inf` literal (T-10-02);
/// the `SharedMemory::new` SIZE is the comptime [`BLOCK_REDUCE_SHMEM`] const (Pitfall 3);
/// the cross-plane carry stride derives from `CUBE_DIM_X` / `PLANE_DIM`, never a literal
/// wave width (D-09). if-as-STATEMENT only.
#[cube(launch)]
pub fn block_scan_total_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    block_sums: &mut Array<F>,
    #[comptime] inclusive: bool,
) {
    let tid = UNIT_POS;
    let n = input.len();

    // Load this unit's element, zero-padding idle out-of-range lanes (T-10-01).
    let mut val = F::new(0.0);
    if ABSOLUTE_POS < n {
        val = input[ABSOLUTE_POS];
    }

    // Within-plane prefix via wave-agnostic plane ops (width = PLANE_DIM, never a
    // literal). `scanned_in_plane` is always the plane-INCLUSIVE prefix; `scanned` is
    // the requested (inclusive/exclusive) variant.
    let scanned_in_plane = plane_inclusive_sum(val);
    let mut scanned = scanned_in_plane;
    if !inclusive {
        scanned = plane_exclusive_sum(val);
    }

    // Cross-plane carry (Hillis-Steele over per-plane inclusive totals) — identical
    // structure to `block_scan_kernel`. The last unit of each plane writes that plane's
    // inclusive total into a per-plane shared slot keyed by PLANE_POS.
    let mut partials = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
    if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
        partials[PLANE_POS as usize] = scanned_in_plane;
    }
    sync_cube();

    let num_planes = (CUBE_DIM_X + PLANE_DIM - 1u32) / PLANE_DIM;
    let mut s = 1u32;
    while s < num_planes {
        let mut add = F::new(0.0);
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

    let mut carry = F::new(0.0);
    if PLANE_POS >= 1u32 {
        carry = partials[(PLANE_POS - 1u32) as usize];
    }

    let result = scanned + carry;
    // The block's INCLUSIVE within-block prefix (flag-independent): its value at the
    // last valid unit IS the block total consumed by phase 2.
    let inclusive_val = scanned_in_plane + carry;

    if ABSOLUTE_POS < n {
        output[ABSOLUTE_POS] = result;
    }

    // The last VALID unit of each block writes the block total to block_sums[CUBE_POS].
    // Two disjoint cases (same block, same semantics): a FULL block's last lane
    // (`UNIT_POS == CUBE_DIM_X - 1`), and the final PARTIAL block's global-last lane
    // (`ABSOLUTE_POS == n - 1`). `n >= 1` is guaranteed by the launcher (this kernel
    // only runs when there is more than one block).
    if ABSOLUTE_POS < n {
        if UNIT_POS == CUBE_DIM_X - 1u32 {
            block_sums[CUBE_POS as usize] = inclusive_val;
        }
    }
    if ABSOLUTE_POS < n {
        if ABSOLUTE_POS == n - 1usize {
            block_sums[CUBE_POS as usize] = inclusive_val;
        }
    }
}

/// Phase 3 of the two-level [`full_scan`]: add each block's exclusive-scanned offset
/// (`block_offsets[CUBE_POS]`, the running total of all PRIOR blocks) to every element
/// of that block, turning the per-block prefixes from phase 1 into the global prefix.
/// One write per lane, bounds-guarded; generic-float; no `-inf` literal. `CUBE_POS` is
/// bounded by the launch geometry (`num_cubes` cubes) == `block_offsets.len()`.
#[cube(launch)]
pub fn add_block_offset_kernel<F: Float>(data: &mut Array<F>, block_offsets: &Array<F>) {
    if ABSOLUTE_POS < data.len() {
        let offset = block_offsets[CUBE_POS as usize];
        data[ABSOLUTE_POS] += offset;
    }
}

/// One recursion level of the two-level scan over resident device handles. Returns the
/// handle holding the full prefix scan of `in_handle` (`n` elements) WITHOUT reading it
/// back — every handle stays bound to the SAME `client` that allocated it (Pitfall 3).
///
/// - `num_cubes <= 1` (n <= [`SCAN_CUBE_DIM`]): single-cube base case via
///   [`block_scan_kernel`] (the cross-cube carry collapses to identity).
/// - otherwise: phase 1 [`block_scan_total_kernel`] (per-block scan + block totals),
///   phase 2 = an EXCLUSIVE `full_scan_into` of the block totals (recurses until a
///   single cube suffices — so arbitrary `n` is covered, not just `n <= SCAN_CUBE_DIM^2`),
///   phase 3 [`add_block_offset_kernel`].
///
/// No `unwrap`/`expect`/`panic`/host indexing (workspace lints + D-13): a device
/// read-back failure only surfaces at the top-level [`full_scan`].
///
/// `pub(crate)` so the reduce-by-key launcher (`kernels/reduce.rs`, Plan 10-03) can
/// reuse this two-level exclusive scan for on-device key-run detection (flags →
/// per-element segment index) without a host round-trip through [`full_scan`].
pub(crate) fn full_scan_into<F>(
    client: &cubecl::client::ComputeClient<crate::SelectedRuntime>,
    in_handle: Handle,
    n: usize,
    inclusive: bool,
) -> CbResult<Handle>
where
    F: Float + CubeElement,
{
    let out_handle = client.empty(n * std::mem::size_of::<F>());
    let num_cubes = n.div_ceil(SCAN_CUBE_DIM).max(1);
    let dim = CubeDim {
        x: SCAN_CUBE_DIM as u32,
        y: 1,
        z: 1,
    };

    if num_cubes <= 1 {
        // Single-cube base case: n <= SCAN_CUBE_DIM, the documented `block_scan_kernel`
        // scope where the whole scan fits one cube.
        block_scan_kernel::launch::<F, crate::SelectedRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            dim,
            unsafe { ArrayArg::from_raw_parts(in_handle, n) },
            unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
            inclusive,
        );
        return Ok(out_handle);
    }

    // Phase 1: per-block scan + per-block totals.
    let block_sums = client.empty(num_cubes * std::mem::size_of::<F>());
    block_scan_total_kernel::launch::<F, crate::SelectedRuntime>(
        client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        dim,
        unsafe { ArrayArg::from_raw_parts(in_handle, n) },
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(block_sums.clone(), num_cubes) },
        inclusive,
    );

    // Phase 2: exclusive scan of the block totals → each block's additive offset
    // (recurses; terminates once the totals fit a single cube).
    let block_offsets = full_scan_into::<F>(client, block_sums, num_cubes, false)?;

    // Phase 3: add each block's offset to its elements.
    add_block_offset_kernel::launch::<F, crate::SelectedRuntime>(
        client,
        CubeCount::Static(num_cubes as u32, 1, 1),
        dim,
        unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(block_offsets, num_cubes) },
    );

    Ok(out_handle)
}

/// Cross-cube two-level full prefix scan over the compile-time [`crate::SelectedRuntime`]
/// (GPUT-16, RESEARCH Open Q2 — the generalization of the single-cube
/// [`block_scan_kernel`] to arbitrary `n`). `inclusive` selects the inclusive
/// (running total includes self) vs exclusive (`output[0] == 0`) variant; it threads
/// through every recursion level as the `#[comptime]` flag. The output has the SAME
/// length as the input (a scan is not a reduction).
///
/// The empty input short-circuits to an empty vec (no launch); `n = 1` returns `[x]`
/// (inclusive) / `[0]` (exclusive) via the single-cube base case. A device read-back
/// failure surfaces as [`CbError::Degenerate`] (never a silent zero buffer). No
/// `unwrap`/`expect`/`panic`/host indexing (workspace lints + D-13).
pub fn full_scan<F>(input: &[F], inclusive: bool) -> CbResult<Vec<F>>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n = input.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let device = <crate::SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <crate::SelectedRuntime as cubecl::Runtime>::client(&device);

    let in_handle = client.create(cubecl::bytes::Bytes::from_elems(input.to_vec()));
    let out_handle = full_scan_into::<F>(&client, in_handle, n, inclusive)?;

    let bytes = client
        .read_one(out_handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL device read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, F>(&bytes).to_vec())
}

/// Flag-array SEGMENTED prefix scan (GPUT-16, Plan 10-01; upstream `segmented_scan.cu`
/// `TSegmentedSum` pair-combiner). Each unit carries a `(value, flag)` pair where
/// `flag = 1.0` marks a SEGMENT START; the running sum RESETS at each segment boundary
/// so interior positions accumulate within their segment only. The `#[comptime]
/// inclusive` flag selects inclusive (running total includes self) vs exclusive
/// (segment start → `0`, otherwise sum of strictly-prior elements IN THE SEGMENT).
///
/// Structural parity: a shared-memory Hillis-Steele scan whose combine operator is the
/// segmented pair-sum — `if flag[tid] == 0 { val[tid] += val[tid - s] }` and
/// `flag[tid] |= flag[tid - s]`. Once a unit's accumulated flag is set, it stops
/// absorbing left neighbours (the segment reset). Reads are snapshotted into registers
/// before the barrier so the in-place shared update has no read/write hazard.
///
/// SCOPE (mirrors `block_scan_kernel` Open Q2): this performs the segmented scan WITHIN
/// a single cube (`n <= SCAN_CUBE_DIM`). The cross-cube segmented carry (propagating a
/// block's tail sum into the next block only until its first segment head) is the
/// documented forward dependency — it reuses the SAME two-level pattern as [`full_scan`]
/// plus a head-seen mask, and is NOT performed here (documented, not a silent cut).
///
/// Generic over `F: Float` (AGENTS.md generics-float). Every device access is under an
/// `ABSOLUTE_POS < input.len()` bounds guard (T-10-01); no `-inf` literal (T-10-02); the
/// `SharedMemory::new` SIZE is the comptime [`BLOCK_REDUCE_SHMEM`] const (Pitfall 3); the
/// Hillis-Steele stride is bounded by `CUBE_DIM_X`, never a literal wave width (D-09).
/// The `sflags` values are exactly `0.0`/`1.0` (assigned, never accumulated as floats),
/// so the `== 0.0` / `> 0.0` segment tests are exact. if-as-STATEMENT only.
#[cube(launch)]
pub fn segmented_scan_kernel<F: Float>(
    input: &Array<F>,
    flags: &Array<F>,
    output: &mut Array<F>,
    #[comptime] inclusive: bool,
) {
    let tid = UNIT_POS;
    let n = input.len();

    // Load this unit's (value, flag), zero-padding idle out-of-range lanes (T-10-01).
    let mut val = F::new(0.0);
    let mut flag = F::new(0.0);
    if ABSOLUTE_POS < n {
        val = input[ABSOLUTE_POS];
        flag = flags[ABSOLUTE_POS];
    }

    let mut svals = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
    let mut sflags = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
    svals[tid as usize] = val;
    sflags[tid as usize] = flag;
    sync_cube();

    // Hillis-Steele segmented inclusive scan (TSegmentedSum). Stride bound = CUBE_DIM_X
    // (the runtime cube width), never a literal.
    let mut s = 1u32;
    while s < CUBE_DIM_X {
        // Snapshot the left neighbour into registers BEFORE any in-place write so the
        // shared update below has no read/write hazard.
        let mut nb_val = F::new(0.0);
        let mut nb_flag = F::new(0.0);
        let mut active = false;
        if tid >= s {
            nb_val = svals[(tid - s) as usize];
            nb_flag = sflags[(tid - s) as usize];
            active = true;
        }
        sync_cube();
        if active {
            // Absorb the neighbour only while no segment head has been reached yet.
            if sflags[tid as usize] == F::new(0.0) {
                svals[tid as usize] += nb_val;
            }
            // Propagate the head flag (logical OR over 0.0/1.0 values).
            if nb_flag > F::new(0.0) {
                sflags[tid as usize] = F::new(1.0);
            }
        }
        sync_cube();
        s *= 2u32;
    }

    let incl = svals[tid as usize];
    let mut result = incl;
    if !inclusive {
        // Exclusive: subtract this lane's own value; a segment start resets to 0.
        result = incl - val;
        if flag > F::new(0.0) {
            result = F::new(0.0);
        }
    }

    if ABSOLUTE_POS < n {
        output[ABSOLUTE_POS] = result;
    }
}

// ===========================================================================
// Reduce family (GPUT-16, Plan 10-03): segmented-reduce + reduce-by-key.
// ---------------------------------------------------------------------------
// Upstream analog: `catboost/cuda/cuda_util/kernel/reduce.cu`
// `SegmentedReduce*PerSegment` (one block per segment, `meanSize<600` fast path)
// and `ReduceByKey`. Structural analog in-tree: `block_reduce_kernel` (intra-cube
// fold) + `block_scan_total_kernel`/`full_scan` (the 10-01 flag scan reused for
// key-run detection). Every partial is accumulated in **f64** regardless of the
// channel float type `F` (bounds float error — mirrors upstream `update_part_props`
// `M=8` double re-accumulate; RESEARCH §Reduction), and the per-segment fold is a
// FIXED-ORDER shared-memory tree reduce, so both primitives are DETERMINISTIC
// (zero run-to-run spread) with no float atomics. Serial CPU self-oracles live in
// `kernels/reduce.rs` (source/test separation).
// ===========================================================================

/// Segmented reduce (GPUT-16, Plan 10-03; upstream `SegmentedReducePerSegment`). One
/// cube per segment: cube `CUBE_POS` sums `input[seg_offsets[CUBE_POS] ..
/// seg_offsets[CUBE_POS + 1])` into `output[CUBE_POS]`. `seg_offsets` has
/// `num_segments + 1` entries (the classic exclusive-prefix boundary array produced
/// by the 10-01 scan / partitions primitives). An empty segment (`start == end`)
/// yields `0`.
///
/// Accumulation is in **f64** regardless of `F` (each lane widens `input[i]` via
/// `f64::cast_from` before folding); the intra-segment fold is a FIXED-ORDER
/// `CUBE_DIM_X`-strided shared-memory tree reduce (deterministic — a fixed pairing,
/// no atomics), then the f64 result is narrowed back to `F` for `output`.
///
/// Generic over `F: Float` (AGENTS.md generics-float; integer offsets are `u32`).
/// Every device access is under a bounds guard (T-10-01); no `-inf` literal
/// (T-10-08); the `SharedMemory::new` SIZE is the comptime [`BLOCK_REDUCE_SHMEM`]
/// const (Pitfall 3); the reduce stride derives from `CUBE_DIM_X`, never a literal
/// wave width (D-09). if-as-STATEMENT only.
#[cube(launch)]
pub fn segmented_reduce_kernel<F: Float>(
    input: &Array<F>,
    seg_offsets: &Array<u32>,
    output: &mut Array<F>,
) {
    let tid = UNIT_POS;
    let seg = CUBE_POS;

    // Segment bounds [start, end). `seg_offsets` has num_segments+1 entries, so
    // reading `seg + 1` is in-bounds for every launched cube (CUBE_POS < num_segments).
    let start = seg_offsets[seg];
    let end = seg_offsets[seg + 1usize];

    // Each lane folds a CUBE_DIM_X-strided slice of the segment in f64.
    let mut acc = 0.0f64;
    let mut i = start + tid;
    while i < end {
        acc += f64::cast_from(input[i as usize]);
        i += CUBE_DIM_X;
    }

    // Fixed-order f64 shared-mem tree reduce (deterministic; stride from CUBE_DIM_X).
    let mut shared = SharedMemory::<f64>::new(BLOCK_REDUCE_SHMEM);
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
        output[seg] = F::cast_from(shared[0usize]);
    }
}

/// Key-head flag kernel (reduce-by-key phase 1): `flags[i] = 1.0` iff element `i`
/// STARTS a new key run (`i == 0` or `keys[i] != keys[i-1]`), else `0.0`. Feeds the
/// 10-01 exclusive [`full_scan`] (phase 2) whose result is the segment index per
/// element. Flags are exactly `0.0`/`1.0` (assigned, never accumulated), so the
/// downstream `> 0.5` head test is exact. Generic over `F: Float` (the scan channel);
/// keys are `u32`. One write per lane, bounds-guarded; no `-inf` literal.
#[cube(launch)]
pub fn key_head_flag_kernel<F: Float>(keys: &Array<u32>, flags: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < keys.len() {
        let mut head = F::new(0.0);
        if i == 0usize {
            head = F::new(1.0);
        } else {
            if keys[i] != keys[i - 1usize] {
                head = F::new(1.0);
            }
        }
        flags[i] = head;
    }
}

/// Segment-offset scatter kernel (reduce-by-key phase 3): every key-run HEAD writes
/// its start position into `seg_offsets[seg_id]`, where `seg_id` is the exclusive
/// flag-scan value at that head (its distinct-key index). Each head owns a distinct
/// `seg_id`, so the writes never collide — deterministic. The trailing slot
/// `seg_offsets[num_segments] = n` is host-initialized (see `kernels/reduce.rs`).
/// `seg_ids` values are exact small integers held in `F` (sums of `0.0`/`1.0`), so
/// `u32::cast_from` is exact. One write per lane, bounds-guarded; no `-inf` literal.
#[cube(launch)]
pub fn segment_offset_scatter_kernel<F: Float>(
    flags: &Array<F>,
    seg_ids: &Array<F>,
    seg_offsets: &mut Array<u32>,
) {
    let i = ABSOLUTE_POS;
    if i < flags.len() {
        if flags[i] > F::new(0.5) {
            let seg = u32::cast_from(seg_ids[i]);
            seg_offsets[seg as usize] = i as u32;
        }
    }
}

/// reduce-by-key (GPUT-16, Plan 10-03; upstream `ReduceByKey`). Given `keys` sorted so
/// equal keys form contiguous RUNS, plus the per-run `seg_offsets` derived on-device by
/// [`key_head_flag_kernel`] -> exclusive [`full_scan`] -> [`segment_offset_scatter_kernel`],
/// this emits one `(out_keys[seg], out_sums[seg])` per distinct key: `out_keys[seg]` is
/// the run's key and `out_sums[seg]` is the f64 sum of its `values`.
///
/// Structurally identical fold to [`segmented_reduce_kernel`] (one cube per run, f64
/// accumulation, fixed-order shared-mem tree reduce — DETERMINISTIC), but it ALSO writes
/// the representative key so the result is a compacted key/sum pair list. Accumulation is
/// in f64 regardless of `F`; the sum is narrowed to `F` on write.
///
/// Generic over `F: Float` (AGENTS.md generics-float); keys/offsets are `u32`. Bounds-
/// guarded (T-10-01); no `-inf` literal (T-10-08); `SharedMemory` SIZE is the comptime
/// [`BLOCK_REDUCE_SHMEM`] const (Pitfall 3); reduce stride from `CUBE_DIM_X` (D-09).
#[cube(launch)]
pub fn reduce_by_key_kernel<F: Float>(
    keys: &Array<u32>,
    values: &Array<F>,
    seg_offsets: &Array<u32>,
    out_keys: &mut Array<u32>,
    out_sums: &mut Array<F>,
) {
    let tid = UNIT_POS;
    let seg = CUBE_POS;

    let start = seg_offsets[seg];
    let end = seg_offsets[seg + 1usize];

    // Representative key of the run (its first element; runs are non-empty by
    // construction — a head always has at least itself).
    if tid == 0u32 {
        out_keys[seg] = keys[start as usize];
    }

    // f64 fold of the run's values, CUBE_DIM_X-strided.
    let mut acc = 0.0f64;
    let mut i = start + tid;
    while i < end {
        acc += f64::cast_from(values[i as usize]);
        i += CUBE_DIM_X;
    }

    let mut shared = SharedMemory::<f64>::new(BLOCK_REDUCE_SHMEM);
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
        out_sums[seg] = F::cast_from(shared[0usize]);
    }
}

/// Fixed-point scale for the deterministic reduce finalize (`2^30`, the well-tested
/// gradient/hessian default from the CubeCL fixed-point-atomics manual §2.1). A
/// power of two keeps `v * S` and `sum / S` exact mantissa shifts. Used by
/// [`block_reduce_fixedpoint_kernel`] and its host read-back in `kernels/reduce.rs`.
pub(crate) const REDUCE_FIXEDPOINT_SCALE_F64: f64 = 1_073_741_824.0; // 2^30

/// Deterministic FIXED-POINT reduce finalize (Plan 10-03 Task 2, strategy (c); CubeCL
/// fixed-point-atomics manual §3). Each cube folds its `CUBE_DIM`-wide slice of `input`
/// into ONE f64 partial (fixed-order shared-mem tree reduce), then unit 0 quantizes that
/// partial `round(partial * 2^30) -> i64 -> u64` and `fetch_add`s it into the single
/// global `Atomic<u64>` accumulator. Integer atomic add is EXACT and ORDER-INDEPENDENT
/// (two's-complement wrapping `u64` add == `i64` add), so the cross-cube finalize is
/// DETERMINISTIC regardless of the hardware contention schedule — the property float
/// atomics cannot offer. The host reinterprets `acc[0]` bits as `i64` and divides by
/// `2^30` (see `kernels/reduce.rs`).
///
/// Guarded host-side by the device's `Atomic<u64>` add capability (the launcher only
/// runs this when advertised; otherwise it reports the deterministic host-sum
/// downgrade — no silent switch). Accumulation is in f64 (a full-magnitude fixed-point
/// range at `k=30`); `SharedMemory` SIZE is the comptime [`BLOCK_REDUCE_SHMEM`] const
/// (Pitfall 3); reduce stride from `CUBE_DIM_X` (D-09); bounds-guarded (T-10-01); no
/// `-inf` literal (T-10-08). Generic over `F: Float` (the input channel).
#[cube(launch)]
pub fn block_reduce_fixedpoint_kernel<F: Float>(input: &Array<F>, acc: &Array<Atomic<u64>>) {
    let tid = UNIT_POS;

    // Load + widen to f64, zero-padding idle out-of-range lanes (T-10-01).
    let mut val = 0.0f64;
    if ABSOLUTE_POS < input.len() {
        val = f64::cast_from(input[ABSOLUTE_POS]);
    }

    // Fixed-order f64 shared-mem tree reduce → one per-cube partial in unit 0.
    let mut shared = SharedMemory::<f64>::new(BLOCK_REDUCE_SHMEM);
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

    // Quantize the per-cube partial and integer-atomic-add into the global accumulator.
    // `round(partial * 2^30) → i64 → reinterpret bits as u64`; wrapping u64 add is exact
    // signed accumulation (manual §2.2). Order-independent ⇒ deterministic.
    if tid == 0u32 {
        let scaled = shared[0usize] * REDUCE_FIXEDPOINT_SCALE_F64;
        let q = u64::cast_from(i64::cast_from(f64::round(scaled)));
        acc[0].fetch_add(q);
    }
}

// ===========================================================================
// Plan 10-04 (GPUT-16): sort / reorder primitives — the stable single-bit reorder
// (`reorder_one_bit`) and the LSD radix sort composed from it (upstream
// `cuda_util/kernel/sort`, CATBOOST_CUDA_KERNELS_DESIGN §6.1/§6.2). Both consume the
// 10-01 exclusive `full_scan` primitive; the self-oracles live in `kernels/sort.rs`.
// ===========================================================================

/// reorder_one_bit phase A: extract bit `bit` of each `u32` key into a `0.0`/`1.0`
/// float flag (the additive-scan channel). `flags[i] = 1.0` iff `((keys[i]>>bit)&1) ==
/// 1`, else `0.0`. The exact `0.0`/`1.0` values (assigned by branch, never
/// accumulated) feed the 10-01 exclusive [`full_scan`] so the downstream
/// `onesBefore[i]` prefix is exact for every `n <= 2^53`.
///
/// Generic over `F: Float` (AGENTS.md generics-float; keys are `u32`, the runtime
/// `bit` selector is `u32`). One write per lane, bounds-guarded (T-10-09); no `-inf`
/// literal (T-10-10); if-as-STATEMENT only.
#[cube(launch)]
pub fn radix_bit_flag_kernel<F: Float>(keys: &Array<u32>, flags: &mut Array<F>, bit: u32) {
    let i = ABSOLUTE_POS;
    if i < keys.len() {
        // Branch-assign the exact float flag (avoid a u32->F numeric cast).
        let mut f = F::new(0.0);
        if (keys[i] >> bit) & 1u32 == 1u32 {
            f = F::new(1.0);
        }
        flags[i] = f;
    }
}

/// reorder_one_bit phase C: the STABLE single-bit scatter (upstream `reorder_one_bit`,
/// CATBOOST_CUDA_KERNELS_DESIGN §6.1). Given the current `keys` (+ paired `values`) and
/// `ones_before` = the EXCLUSIVE scan of the bit flags (`ones_before[i]` = count of
/// keys with `bit==1` strictly before `i`), each element is scattered to
///
/// ```text
/// pos = (bit==0) ? zeroesBefore[i]              // = i - ones_before[i]
///                : total_zeros + ones_before[i]
/// ```
///
/// Zeros keep their relative order at the front, ones keep theirs after the last zero —
/// STABLE within each group. `total_zeros` (the count of keys with `bit==0`) is passed
/// as a runtime scalar; it is order-invariant, so the host computes it once per pass.
/// Each source index maps to a DISTINCT destination, so the scatter is
/// order-independent (no cross-lane `+=`; T-10-09).
///
/// Generic over `F: Float` (the scan channel); keys/values/positions are `u32`. Every
/// device access is under a bounds guard (T-10-09); no `-inf` literal (T-10-10).
#[cube(launch)]
pub fn reorder_one_bit_scatter_kernel<F: Float>(
    keys: &Array<u32>,
    values: &Array<u32>,
    ones_before: &Array<F>,
    out_keys: &mut Array<u32>,
    out_values: &mut Array<u32>,
    bit: u32,
    total_zeros: u32,
) {
    let i = ABSOLUTE_POS;
    if i < keys.len() {
        let ones_b = u32::cast_from(ones_before[i]);
        // zeroesBefore[i] = i - ones_before[i] (exactly i-onesBefore zeros precede i).
        let mut pos = (i as u32) - ones_b;
        if (keys[i] >> bit) & 1u32 == 1u32 {
            pos = total_zeros + ones_b;
        }
        out_keys[pos as usize] = keys[i];
        out_values[pos as usize] = values[i];
    }
}

// ===========================================================================
// Plan 10-04 (GPUT-16): TDataPartition {Offset,Size} update (upstream
// `partitions.cu` `UpdatePartitionOffsets` / `UpdatePartitionSizes`,
// CATBOOST_CUDA_KERNELS_DESIGN §6.1/§2). From a SORTED partition-id array the two
// kernels produce, per partition, the contiguous `{Offset,Size}` the depth>1
// histograms (Phase 11) key on. They CONSUME the 10-01 exclusive `full_scan` (the
// head-flag scan → per-element run index) exactly as reduce-by-key does; the
// self-oracle lives in `kernels/partitions.rs`.
// ===========================================================================

/// TDataPartition offsets (phase A). Given a SORTED `part_ids` array, the per-element
/// key-run HEAD flags (`i==0 || part_ids[i]!=part_ids[i-1]`, produced by
/// [`key_head_flag_kernel`]) and `run_ids` = the EXCLUSIVE [`full_scan`] of those flags
/// (each element's 0-based distinct-run index — the 10-01 scan reuse), every HEAD writes:
/// - its start position `i` into the COMPACT `run_starts[run_id]` and its partition value
///   into `run_keys[run_id]` (the run→partition map consumed by the size kernel), AND
/// - its start position `i` directly into `offsets[part_ids[i]]` (the partition's Offset).
///
/// Each HEAD owns a DISTINCT `run_id` and a DISTINCT partition value, so no write
/// collides — deterministic. Partitions with NO elements are never written, so they keep
/// their host-seeded `{Offset:0}` (a well-defined empty offset). `run_ids` values are
/// exact small integers held in `F` (sums of `0.0`/`1.0`), so `u32::cast_from` is exact.
///
/// Generic over `F: Float` (the scan channel); ids/positions are `u32`. One write per
/// lane, bounds-guarded (T-10-09); no `-inf` literal (T-10-10).
#[cube(launch)]
pub fn update_partition_offsets_kernel<F: Float>(
    part_ids: &Array<u32>,
    flags: &Array<F>,
    run_ids: &Array<F>,
    offsets: &mut Array<u32>,
    run_keys: &mut Array<u32>,
    run_starts: &mut Array<u32>,
) {
    let i = ABSOLUTE_POS;
    if i < part_ids.len() {
        if flags[i] > F::new(0.5) {
            let r = u32::cast_from(run_ids[i]);
            let p = part_ids[i];
            run_keys[r as usize] = p;
            run_starts[r as usize] = i as u32;
            offsets[p as usize] = i as u32;
        }
    }
}

/// TDataPartition sizes (phase B). Given the COMPACT `run_keys` / `run_starts` from
/// [`update_partition_offsets_kernel`] (one entry per distinct run) plus the total
/// element count `n`, each run `r` writes its length into `sizes[run_keys[r]]`:
///
/// ```text
/// end  = (r + 1 < num_runs) ? run_starts[r+1] : n
/// size = end - run_starts[r]
/// ```
///
/// Runs are contiguous and cover `[0, n)`, so this is exactly each partition's element
/// count. Partitions with NO run keep their host-seeded `{Size:0}`. Launch geometry is
/// one lane per run (`num_runs` lanes); bounds-guarded (T-10-09); no `-inf` literal
/// (T-10-10). Generic over `F: Float` for launch-signature symmetry (ids are `u32`).
#[cube(launch)]
pub fn update_partition_sizes_kernel<F: Float>(
    run_keys: &Array<u32>,
    run_starts: &Array<u32>,
    sizes: &mut Array<u32>,
    n: u32,
    num_runs: u32,
) {
    let _ = F::new(0.0);
    let r = ABSOLUTE_POS;
    if r < run_keys.len() {
        let start = run_starts[r as usize];
        let mut end = n;
        if (r as u32) + 1u32 < num_runs {
            end = run_starts[(r + 1) as usize];
        }
        let p = run_keys[r as usize];
        sizes[p as usize] = end - start;
    }
}

// ===========================================================================
// Plan 10-04 (GPUT-16): fill / gather / vector-arithmetic transforms (upstream
// `fill.cu` + `transform.cu`, CATBOOST_CUDA_KERNELS_DESIGN §6.1). Trivial
// one-write-per-lane transforms (mirror [`histogram_scatter_kernel`]); their FULL
// validation is transitive through the depth-1 tree + cindex (D-01), so the
// `kernels/fill_transform.rs` self-oracle asserts elementwise equality on small inputs.
// ===========================================================================

/// `fill`: set every element of `buf` to the constant `value[0]` (upstream `fill.cu`
/// `FillBuffer`). The constant is passed as a length-1 array (the `lambda_h` scalar
/// precedent — a Float launch scalar as a resident length-1 handle). One write per lane,
/// bounds-guarded (T-10-10); no `-inf` literal; generic-float.
#[cube(launch)]
pub fn fill_kernel<F: Float>(buf: &mut Array<F>, value: &Array<F>) {
    if ABSOLUTE_POS < buf.len() {
        buf[ABSOLUTE_POS] = value[0];
    }
}

/// `gather`: `out[i] = src[idx[i]]` (upstream `transform.cu` gather). One indexed read
/// + one write per lane, bounds-guarded (T-10-09/10 — every scatter/gather index access
/// is guarded); no `-inf` literal; generic-float (indices are `u32`).
#[cube(launch)]
pub fn gather_kernel<F: Float>(src: &Array<F>, idx: &Array<u32>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let j = idx[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = src[j as usize];
    }
}

/// Elementwise vector `add`: `out[i] = a[i] + b[i]` (upstream `transform.cu`
/// AddVector). One write per lane, bounds-guarded; no `-inf` literal; generic-float.
#[cube(launch)]
pub fn vector_add_kernel<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// Elementwise vector `sub`: `out[i] = a[i] - b[i]`. One write per lane, bounds-guarded;
/// no `-inf` literal; generic-float.
#[cube(launch)]
pub fn vector_sub_kernel<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] - b[ABSOLUTE_POS];
    }
}

/// Elementwise vector `mul`: `out[i] = a[i] * b[i]`. One write per lane, bounds-guarded;
/// no `-inf` literal; generic-float.
#[cube(launch)]
pub fn vector_mul_kernel<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS];
    }
}

/// Elementwise vector `div`: `out[i] = a[i] / b[i]` (caller guarantees non-zero
/// divisors — the oracle uses non-zero `b`). One write per lane, bounds-guarded; no
/// `-inf` literal; generic-float.
#[cube(launch)]
pub fn vector_div_kernel<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS];
    }
}

// ===========================================================================
// Plan 10-05 (GPUT-16): bit-compression pack / unpack (upstream `compression.cu`
// `TCompressionHelper`, CATBOOST_CUDA_KERNELS_DESIGN §6.1). Packs an ≤8-bit bin
// column into shared 32-bit words (`bitsPerKey = ceil(log2(n_bins+1))`,
// `keysPerWord = 32/bitsPerKey`, `Mask = (1<<bitsPerKey)-1`) so the bit-packed
// cindex (10-06) can address each field by (Offset, Shift, Mask). Both kernels are
// pure `u32` integer transforms (bit-exact — tighter than the ≤1e-4 float bar, D-07);
// the `<F: Float>` phantom (`let _ = F::new(0.0)`) keeps the launch signature uniform
// with the other primitives (mirrors [`update_partition_sizes_kernel`]). The host-side
// layout helper [`bit_pack_layout`] computes the comptime bit geometry with checked
// arithmetic (T-10-12). The self-oracle lives in `kernels/compression.rs`.
// ===========================================================================

/// Bit geometry for packing an `n`-length ≤8-bit bin column into shared 32-bit words
/// (upstream `TCompressionHelper`). `bits_per_key = ceil(log2(n_bins+1))` (bits needed
/// to hold bin values `0..=n_bins`), `keys_per_word = 32/bits_per_key`,
/// `mask = (1<<bits_per_key)-1`, `num_words = ceil(n / keys_per_word)`.
// `pub(crate)` + consumed by the 10-05 self-oracle now and by the 10-06 bit-packed
// cindex builder next; `allow(dead_code)` so the default (cpu, non-test) build does not
// warn on the not-yet-wired production consumer.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitPackLayout {
    /// Bits per packed key (`ceil(log2(n_bins+1))`, clamped to `1..=32`).
    pub bits_per_key: u32,
    /// Keys packed per 32-bit word (`32 / bits_per_key`).
    pub keys_per_word: u32,
    /// Extraction mask (`(1<<bits_per_key)-1`).
    pub mask: u32,
    /// Number of 32-bit words needed to hold `n` keys.
    pub num_words: usize,
}

/// Compute the [`BitPackLayout`] for a feature with `n_bins` distinct border bins over
/// `n` objects. Every arithmetic step that can overflow (`n_bins+1`, the mask shift, the
/// word-count reconstruction) is CHECKED and surfaces as [`CbError::OutOfRange`] (T-10-12
/// — no unguarded shift/mul crosses the host→device boundary). Borders/quantization stay
/// host-side; this only sizes the packing.
#[allow(dead_code)]
pub(crate) fn bit_pack_layout(n_bins: u32, n: usize) -> CbResult<BitPackLayout> {
    // Distinct values to represent = n_bins + 1 (bins 0..=n_bins).
    let distinct = n_bins.checked_add(1).ok_or_else(|| {
        CbError::OutOfRange(format!("bit_pack_layout: n_bins {n_bins} + 1 overflows u32"))
    })?;
    // bits_per_key = ceil(log2(distinct)), at least 1 (a single-value column still needs
    // one bit so keys_per_word is well-defined).
    let bits_per_key = if distinct <= 1 { 1u32 } else { (distinct - 1).ilog2() + 1 };
    if bits_per_key == 0 || bits_per_key > 32 {
        return Err(CbError::OutOfRange(format!(
            "bit_pack_layout: bits_per_key {bits_per_key} out of 1..=32 for n_bins {n_bins}"
        )));
    }
    let keys_per_word = 32u32 / bits_per_key;
    if keys_per_word == 0 {
        return Err(CbError::OutOfRange(format!(
            "bit_pack_layout: keys_per_word 0 (bits_per_key {bits_per_key} > 32)"
        )));
    }
    // mask = (1 << bits_per_key) - 1. `checked_shl` rejects a 32-shift (Rust UB); the
    // full-width case is all-ones.
    let mask = if bits_per_key == 32 {
        u32::MAX
    } else {
        1u32.checked_shl(bits_per_key)
            .ok_or_else(|| {
                CbError::OutOfRange(format!("bit_pack_layout: 1 << {bits_per_key} overflows u32"))
            })?
            - 1
    };
    let num_words = n.div_ceil(keys_per_word as usize);
    // Guard the word-count × keys_per_word reconstruction the kernels index with
    // (`word_idx * keys_per_word + slot`); a silent usize overflow here would corrupt
    // the pack address (T-10-12).
    num_words.checked_mul(keys_per_word as usize).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "bit_pack_layout: num_words {num_words} * keys_per_word {keys_per_word} overflows usize"
        ))
    })?;
    Ok(BitPackLayout {
        bits_per_key,
        keys_per_word,
        mask,
        num_words,
    })
}

/// `pack`: pack an ≤8-bit bin column `bins` (each `u32` holds one bin value) into shared
/// 32-bit `words` (upstream `compression.cu` pack). Each thread OWNS one output word and
/// loops over the `keys_per_word` keys that land in it, OR-ing `(bins[key] & mask) <<
/// (slot * bits_per_key)` into its private accumulator — one thread per word ⇒ NO
/// cross-lane `|=` race, so the pack is deterministic and order-independent (T-10-14).
///
/// `bits_per_key` / `keys_per_word` / `mask` are `#[comptime]` (the JIT resolves the bit
/// geometry — no runtime bit-width branch). Bounds-guarded (T-10-12); no `-inf` literal
/// (T-10-14). Pure `u32` transform; the `<F: Float>` phantom keeps the launch signature
/// uniform (`let _ = F::new(0.0)`).
#[cube(launch)]
pub fn pack_bins_kernel<F: Float>(
    bins: &Array<u32>,
    words: &mut Array<u32>,
    #[comptime] bits_per_key: u32,
    #[comptime] keys_per_word: u32,
    #[comptime] mask: u32,
) {
    let _ = F::new(0.0);
    let word_idx = ABSOLUTE_POS;
    if word_idx < words.len() {
        let n = bins.len();
        let base = word_idx * (keys_per_word as usize);
        let mut w = 0u32;
        let mut slot = 0u32;
        while slot < keys_per_word {
            let key_idx = base + slot as usize;
            if key_idx < n {
                let field = (bins[key_idx] & mask) << (slot * bits_per_key);
                w |= field;
            }
            slot += 1u32;
        }
        words[word_idx] = w;
    }
}

/// `unpack`: extract key `i`'s bin field from the packed `words` (upstream
/// `compression.cu` unpack / the `read_bin` accessor the cindex consumes). Key `i` lives
/// in word `i / keys_per_word` at slot `i % keys_per_word`, so its value is
/// `(words[word_idx] >> (slot * bits_per_key)) & mask`. One indexed read + one write per
/// lane — multiple keys sharing a word each extract their OWN field via their distinct
/// `Shift`, so pack∘unpack is bit-exact.
///
/// `bits_per_key` / `keys_per_word` / `mask` are `#[comptime]`. Bounds-guarded (T-10-12);
/// no `-inf` literal (T-10-14). Pure `u32` transform; `<F: Float>` phantom.
#[cube(launch)]
pub fn unpack_bins_kernel<F: Float>(
    words: &Array<u32>,
    out: &mut Array<u32>,
    #[comptime] bits_per_key: u32,
    #[comptime] keys_per_word: u32,
    #[comptime] mask: u32,
) {
    let _ = F::new(0.0);
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let kpw = keys_per_word as usize;
        let word_idx = i / kpw;
        let slot = (i % kpw) as u32;
        let shift = slot * bits_per_key;
        out[i] = (words[word_idx] >> shift) & mask;
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

// Flag-array segmented-scan primitive oracle (source/test separation, Plan 10-01
// GPUT-16): the inclusive/exclusive segmented prefix-scan self-oracle vs an inline
// serial segmented-prefix reference (D-02) lives in `kernels/segmented_scan.rs`,
// mounted at `kernels::segmented_scan`. Runs over the generic `SelectedRuntime`, so it
// builds/runs under EVERY backend (the rocm in-env oracle + wgpu host run).
#[cfg(test)]
mod segmented_scan;

// Sort/reorder primitive oracle (source/test separation, Plan 10-04 GPUT-16): the
// stable single-bit reorder + LSD radix sort self-oracle vs inline serial stable
// partition-by-bit / stable sort references (D-02) lives in `kernels/sort.rs`, mounted
// at `kernels::sort`. Runs over the generic `SelectedRuntime`, so it builds/runs under
// EVERY backend (the rocm in-env oracle + wgpu host run).
#[cfg(test)]
mod sort;

// TDataPartition {Offset,Size} update oracle (source/test separation, Plan 10-04
// GPUT-16): the offset/size update self-oracle vs an inline serial partition-
// bookkeeping reference (D-02), including an empty-partition case, lives in
// `kernels/partitions.rs`, mounted at `kernels::partitions`. Runs over the generic
// `SelectedRuntime`, so it builds/runs under EVERY backend.
#[cfg(test)]
mod partitions;

// Fill / gather / vector-arithmetic transform oracle (source/test separation, Plan
// 10-04 GPUT-16): elementwise self-oracles vs inline serial references (D-02) live in
// `kernels/fill_transform.rs`, mounted at `kernels::fill_transform`. Full validation is
// transitive through the depth-1 tree + cindex (D-01). Runs over the generic
// `SelectedRuntime`, so it builds/runs under EVERY backend.
#[cfg(test)]
mod fill_transform;

// Bit-compression pack/unpack primitive oracle (source/test separation, Plan 10-05
// GPUT-16): the pack∘unpack BIT-EXACT round-trip self-oracle vs an inline serial
// pack/unpack reference (D-02, integer equality — tighter than ≤1e-4, D-07) plus the
// `bit_pack_layout` checked-arithmetic guard tests live in `kernels/compression.rs`,
// mounted at `kernels::compression`. Runs over the generic `SelectedRuntime`, so it
// builds/runs under EVERY backend (the rocm in-env oracle + wgpu host run).
#[cfg(test)]
mod compression;

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
/// Comptime score-function selector for [`find_optimal_split_kernel`] (the 7.1
/// `use_plane` / 7.3 comptime-`bits` precedent): the per-leaf `AddLeaf` arithmetic is
/// monomorphized at JIT time so there is NO runtime score-function branch in the inner
/// leaf loop (RESEARCH Pattern 1). Plan A (Phase 7.5) implemented ONLY the L2 arm
/// ([`SCORE_FN_L2`]); the Cosine/Solar/LOO/Sat arms ride the SAME kernel and land in
/// Plan E ([`SCORE_FN_COSINE`] / [`SCORE_FN_SOLAR_L2`] / [`SCORE_FN_LOO_L2`] /
/// [`SCORE_FN_SAT_L2`]).
pub(crate) const SCORE_FN_L2: u32 = 0;

/// Comptime selector for the **Cosine** (default) / **NewtonCosine** arm of
/// [`find_optimal_split_kernel`] (`cb-compute/src/score.rs::cosine_split_score`,
/// `multi_dim_split_score` Cosine/NewtonCosine dispatch): the numerator is the SAME
/// `Σ avg·sum` L2 fold and the denominator is `1e-100 + Σ avg²·weight` (the `1e-100`
/// seed is the FIRST summand, `score.rs:78`), with `score = num / sqrt(den)`.
/// NewtonCosine reuses this formula VERBATIM (the second-order distinction is the
/// histogram FILL, not the score — `pointwise_scores.cu:512-521`).
pub(crate) const SCORE_FN_COSINE: u32 = 1;

/// Comptime selector for the **SolarL2** arm (`score.rs::solar_l2_terms`,
/// `score_calcers.cuh:22-24`): per-leaf
/// `weight > 1e-20 ? (-sum*sum)*(1 + 2*ln(weight + 1))/weight : 0`. Takes NO
/// `scaled_l2` regularizer (IN-04 — do NOT add it).
pub(crate) const SCORE_FN_SOLAR_L2: u32 = 2;

/// Comptime selector for the **LOOL2** (leave-one-out L2) arm
/// (`score.rs::loo_l2_terms`, `score_calcers.cuh:83-87`): per-leaf
/// `adjust = weight>1 ? weight/(weight-1) : 0; adjust*=adjust;
/// weight>0 ? adjust*(-sum*sum)/weight : 0`.
pub(crate) const SCORE_FN_LOO_L2: u32 = 3;

/// Comptime selector for the **SatL2** (saturated L2) arm (`score.rs::sat_l2_terms`,
/// `score_calcers.cuh:114-117`): per-leaf
/// `adjust = weight>2 ? weight*(weight-2)/(weight²-3*weight+1) : 0;
/// weight>0 ? adjust*(-sum*sum)/weight : 0`.
pub(crate) const SCORE_FN_SAT_L2: u32 = 4;

/// Comptime shared-memory size for [`find_optimal_split_kernel`]'s block-reduce argmin
/// (Pitfall 3 — a compile-time `usize` const, never a runtime/topology value). It holds
/// one `(gain, candidate-index)` slot per unit at the launch cube width, so it equals
/// [`BLOCK_REDUCE_SHMEM`] (the same `CUBE_DIM` allocation the reduce kernels use). This
/// is a shared-memory SIZE, NOT a wave/warp-size literal in any stride (D-09).
pub(crate) const ARGMIN_SHMEM: usize = BLOCK_REDUCE_SHMEM;

/// Per-leaf `calc_average` — the FROZEN `cb-compute/src/leaf.rs::calc_average` `count > 0`
/// guard transcribed VERBATIM: `avg = weight > 0 ? sum / (weight + lambda) : 0`. Used by
/// the L2 and Cosine arms of [`find_optimal_split_kernel`] (numerator avg and the Cosine
/// denominator `avg²·weight`). Pure device helper (no runtime branch beyond the guard);
/// inlined at JIT, so the monomorphized kernel body is byte-identical to the prior
/// hand-inlined arms (IN-02 behavior-preserving extraction). `if`-as-statement only.
#[cube]
fn cb_leaf_avg<F: Float>(sum: F, w: F, lambda: F) -> F {
    let mut avg = F::new(0.0);
    if w > F::new(0.0) {
        avg = sum / (w + lambda);
    }
    avg
}

/// The per-leaf ADDITIVE score contribution for the comptime-selected calcer (IN-02:
/// the shared per-leaf term that replaces the five copy-pasted arms of
/// [`find_optimal_split_kernel`]). Given ONE leaf's `(Σ der1, Σ weight)` it returns that
/// leaf's contribution to the split score; the kernel calls it twice (left then right)
/// and sums in the SAME left-then-right order the prior inlined arms used, so the
/// monomorphized result is identical.
///
/// Each `if score_fn == comptime!(...)` is resolved at JIT (RESEARCH Pattern 1), so the
/// caller carries NO runtime score-fn branch. Every arm is transcribed VERBATIM from the
/// FROZEN CPU oracle `cb-compute/src/score.rs` (the citations live on the kernel doc):
/// - L2 / Cosine: `avg·sum` where `avg = calc_average` (the Cosine numerator is the SAME
///   `avg·sum`; the Cosine denominator is folded in the kernel via [`cb_leaf_avg`]).
/// - SolarL2: `w > 1e-20 ? (-sum²)·(1 + 2·ln(w+1))/w : 0` (no scaled_l2).
/// - LOOL2: `adjust = w>1 ? w/(w-1) : 0; adjust²·(-sum²)/w` guarded `w>0` (no scaled_l2).
/// - SatL2: `adjust = w>2 ? w(w-2)/(w²-3w+1) : 0; adjust²·(-sum²)/w` guarded `w>0`.
///
/// `if`-as-statement only (CubeCL conditionals). Generic over `F: Float` (AGENTS.md
/// generics-float — no hard-coded float type).
#[cube]
fn cb_leaf_score_term<F: Float>(sum: F, w: F, lambda: F, #[comptime] score_fn: u32) -> F {
    let mut term = F::new(0.0);

    // --- L2 / Cosine numerator: avg·sum, avg = calc_average (count>0 guard).
    if score_fn == comptime!(SCORE_FN_L2) {
        let avg = cb_leaf_avg::<F>(sum, w, lambda);
        term = avg * sum;
    }
    if score_fn == comptime!(SCORE_FN_COSINE) {
        let avg = cb_leaf_avg::<F>(sum, w, lambda);
        term = avg * sum;
    }

    // --- SolarL2: weight>1e-20 ? (-sum*sum)*(1 + 2*ln(weight+1))/weight : 0. NO scaled_l2.
    if score_fn == comptime!(SCORE_FN_SOLAR_L2) {
        if w > F::new(1e-20) {
            let one = F::new(1.0);
            let two = F::new(2.0);
            term = (-sum * sum) * (one + two * F::ln(w + one)) / w;
        }
    }

    // --- LOOL2: adjust = weight>1 ? weight/(weight-1) : 0; adjust*=adjust;
    //     weight>0 ? adjust*(-sum*sum)/weight : 0. NO scaled_l2.
    if score_fn == comptime!(SCORE_FN_LOO_L2) {
        let one = F::new(1.0);
        let mut adjust = F::new(0.0);
        if w > one {
            adjust = w / (w - one);
        }
        adjust = adjust * adjust;
        if w > F::new(0.0) {
            term = adjust * (-sum * sum) / w;
        }
    }

    // --- SatL2: adjust = weight>2 ? weight*(weight-2)/(weight²-3*weight+1) : 0;
    //     weight>0 ? adjust*(-sum*sum)/weight : 0. NO scaled_l2.
    if score_fn == comptime!(SCORE_FN_SAT_L2) {
        let two = F::new(2.0);
        let three = F::new(3.0);
        let one = F::new(1.0);
        let mut adjust = F::new(0.0);
        if w > two {
            adjust = w * (w - two) / (w * w - three * w + one);
        }
        if w > F::new(0.0) {
            term = adjust * (-sum * sum) / w;
        }
    }

    term
}

/// Device-resident **pointwise L2 split score + deterministic split argmin** over the
/// FROZEN 7.3 device-resident 2-channel histogram handle (GPU-01 score/split slice,
/// Phase 7.5 Plan A; D-7.5-01/05/06). The `#[comptime] score_fn` selects the score
/// calcer (ONLY [`SCORE_FN_L2`] this plan; Cosine/Solar/LOO/Sat reserved for Plan E).
///
/// # Inputs / layout
///
/// `bin_sums` is the FROZEN 7.3 2-channel histogram (read-only, device-resident, NO
/// host round-trip): cell index `(feature * n_bins + bin) * 2 + channel`, channel 0 =
/// Σ der1, channel 1 = Σ weight (the layout `pointwise_hist2_nonbinary_kernel` writes).
/// `scaled_l2` is the per-tree `scale_l2_reg` output (the L2 regularizer). The candidate
/// enumeration order is ascending `(feature, bin)` flattened as `feature * n_bins + bin`
/// — the SAME order the CPU `cb_train::select_best_candidate` `Candidate` vector uses,
/// so the tie-break agrees (RESEARCH Pattern 4 / A4).
///
/// # Score fold (D-03 f64-finalize, Pitfall 4 of 7.5)
///
/// For each candidate `(feature, border)` the split produces a LEFT leaf (bins
/// `0..=border`) and a RIGHT leaf (bins `border+1..n_bins`). The per-bin
/// `(Σ der1, Σ weight)` are folded — IN f64 (`F::Float` widened to `f64` via a running
/// `f64` accumulator is NOT expressible generically here; instead the device channel is
/// f64 on rocm/cuda/cpu and f32 on wgpu, matching the histogram channel, so `F` IS the
/// finalize type) — into `left`/`right` [`cb_compute::LeafStats`], then the L2 score is
/// transcribed VERBATIM from the FROZEN oracle `cb-compute/src/score.rs::l2_split_score`
/// + `cb-compute/src/leaf.rs::calc_average`:
///
/// ```text
/// avg(sum, weight) = weight + scaled_l2 > 0 ? sum / (weight + scaled_l2) : 0   // calc_average (count>0 guard)
/// add_leaf_plain(leaf) = avg(leaf.sum, leaf.weight) * leaf.sum                  // score.rs:39-42
/// score = add_leaf_plain(left) + add_leaf_plain(right)                          // l2_split_score:49-55
/// ```
///
/// The `count > 0` guard (transcribed from `calc_average`) means a degenerate (empty)
/// leaf contributes 0.0 — no division by zero, no NaN/Inf (Security V5 / T-07.5-01-05).
/// A higher score is a better split.
///
/// # Deterministic argmin (Pitfall 1 / RESEARCH Pattern 4)
///
/// Each thread keeps a running best `(gain, candidate-index)` over the candidates it
/// strides through, then the cube block-reduces those locals: on `gain[a] == gain[b]`
/// it keeps the LOWER candidate index (== ascending `(feature, bin)`), which equals the
/// CPU strict-`>` first-wins over the same order (`select_best_candidate`,
/// `tree.rs:291-302`; upstream `pointwise_scores.cu:140-141`). One `(best_gain,
/// best_idx)` pair is written per cube (block). The block-reduce is the wave-agnostic
/// `CUBE_DIM_X`-strided shared-mem tree (D-09 — no warp/wave-size literal in the stride;
/// `ARGMIN_SHMEM` is the comptime allocation size).
///
/// # Outputs (no host round-trip of the histogram — D-05)
///
/// `scores` (length `n_features * n_bins`) receives the per-candidate L2 score (the
/// self-oracle observation, the analog of `pointwise_hist` reading binSums back ONCE).
/// `best_gain` / `best_idx` (length = the cube count) receive one block winner each; the
/// host finishes the across-block argmin over this small O(blocks) array with the SAME
/// lowest-index tie-break and reads ONLY this descriptor back (D-05). The bulk histogram
/// never leaves the device.
///
/// Generic over `F: Float` (AGENTS.md generics-float — no hard-coded float type). Every
/// device read is under a POSITION bounds guard; the candidate/feature/bin VALUE ranges
/// are validated HOST-SIDE in `launch_find_optimal_split_pointwise_into` before launch.
/// if-as-STATEMENT only (CubeCL conditionals manual). The `MINIMAL_SCORE` sentinel
/// (`f64::NEG_INFINITY` analogue, matching the CPU oracle) is `F::new(f32::MIN)` — the
/// HIP-safe finite stand-in for `-inf` (a literal `-inf` fails the gfx1100 JIT, WR-01) —
/// so any real candidate wins on the first strict-greater compare.
///
/// # Real split borders only (WR-05)
///
/// A candidate's `border` ranges `0..n_bins`, but the trailing `border == n_bins - 1`
/// puts ALL bins in the LEFT leaf / none in the RIGHT — a no-op (non-split) that upstream
/// and the pairwise path (`n_splits = n_bins - 1`) never enumerate. The per-candidate
/// `scores[c]` slot for that border IS still written (so the buffer geometry stays
/// `n_features * n_bins` and the element-wise oracle compare is unchanged — no `-inf`
/// sentinel that would NaN under that compare), but it is EXCLUDED from the argmin
/// (`border < n_bins - 1` guard). The host winner decode (`gpu_runtime`) and the host
/// reference winner (`score_split::reference_best_split`, `grow_loop::cpu_best_stump*`)
/// skip the SAME trailing border in EXACT lockstep, so device and CPU oracle agree.
#[cube(launch)]
pub fn find_optimal_split_kernel<F: Float>(
    bin_sums: &Array<F>,
    scores: &mut Array<F>,
    best_gain: &mut Array<F>,
    best_idx: &mut Array<u32>,
    scaled_l2: &Array<F>,
    n_features: u32,
    #[comptime] n_bins: u32,
    #[comptime] score_fn: u32,
) {
    let tid = UNIT_POS;
    let n_bins_usize = n_bins as usize;
    let n_features_usize = n_features as usize;
    let n_candidates = n_features_usize * n_bins_usize;

    // The per-tree L2 regularizer is passed as a length-1 device array (the codebase
    // passes float values through `Array<F>`, never as a generic-`F` launch scalar — a
    // `#[cube(launch)]` scalar must be a concrete `CubeElement`, not the generic `F`).
    let lambda = scaled_l2[0usize];

    // The minimal-score sentinel any real candidate must beat (the
    // `score.rs::MINIMAL_SCORE` = `f64::NEG_INFINITY` analogue). It must be more negative
    // than (a) every realizable split score, so EVERY candidate wins the per-thread loop's
    // first strict-greater compare, AND (b) every other lane's gain, so an unseen lane
    // (still holding this seed) always LOSES the shared-memory reduction below. The finite
    // `f32::MIN` (~-3.4e38) satisfies both: L2/Cosine scores are >=0, and SolarL2/LOOL2/
    // SatL2 terms are gradient-derived and bounded — never anywhere near -3.4e38 — so this
    // is behaviorally identical to `f64::NEG_INFINITY` for all reachable inputs.
    //
    // WR-01 (rocm re-confirm): a literal `-inf` (`F::new(f32::NEG_INFINITY)`) is NOT usable
    // here — CubeCL emits a bare `double(-inf)` literal that the HIP/comgr compiler rejects
    // on gfx1100 (`error: use of undeclared identifier 'inf'`), failing the kernel JIT for
    // all calcers. The finite `f32::MIN` is the HIP-safe sentinel; closing the (unreachable)
    // sub-`f32::MIN` tail would require threading a per-lane "seen" flag through the
    // reduction, deferred as it has no reachable effect.
    let minimal_score = F::new(f32::MIN);

    // This thread's running best over the candidates it strides through. `best_c` is the
    // candidate index (== feature * n_bins + bin); ties keep the LOWER index, so seed it
    // to the max so any real candidate replaces it on the first strict-greater compare.
    let mut my_gain = minimal_score;
    let mut my_idx = n_candidates as u32;

    // Grid-stride over candidates (D-09: the stride is the cube width CUBE_DIM_X, a
    // topology value, never a literal). Each candidate is one (feature, border) split.
    let mut c = tid as usize;
    while c < n_candidates {
        let feature = c / n_bins_usize;
        let border = c % n_bins_usize;

        // Fold the feature's bins into LEFT (bins 0..=border) / RIGHT (bins
        // border+1..n_bins) leaf stats, reading the FROZEN 2-channel histogram in place.
        let mut left_sum = F::new(0.0);
        let mut left_w = F::new(0.0);
        let mut right_sum = F::new(0.0);
        let mut right_w = F::new(0.0);
        let mut bin = 0usize;
        while bin < n_bins_usize {
            let cell = (feature * n_bins_usize + bin) * 2usize;
            let d = bin_sums[cell];
            let w = bin_sums[cell + 1usize];
            if bin <= border {
                left_sum += d;
                left_w += w;
            } else {
                right_sum += d;
                right_w += w;
            }
            bin += 1usize;
        }

        // The candidate's split score, computed by the comptime-selected calcer arm.
        // Each arm is monomorphized at JIT time (the `if score_fn == comptime!(...)`
        // is resolved away — RESEARCH Pattern 1), so the inner leaf loop carries NO
        // runtime score-function branch. Every arm is transcribed VERBATIM from the
        // FROZEN CPU oracle `cb-compute/src/score.rs` (cited per arm) and folded in the
        // f64 device channel (D-03; f32 only on wgpu, matching the histogram channel).
        let mut score = F::new(0.0);

        // IN-02: the five copy-pasted comptime calcer arms are factored into the shared
        // per-leaf [`cb_leaf_score_term`] (the additive leaf contribution for the
        // comptime-selected calcer) called once per leaf in the SAME left-then-right
        // order the prior inlined arms summed, plus — for Cosine ONLY — the seeded
        // `avg²·weight` denominator folded here via [`cb_leaf_avg`]. The
        // `#[comptime] score_fn` is resolved at JIT, so the monomorphized body computes
        // the IDENTICAL value the hand-inlined arms did (behavior-preserving extraction).

        // L2 numerator (== whole L2 score) / Solar / LOO / Sat (their whole score) and the
        // Cosine NUMERATOR all reduce to `term(left) + term(right)` in left-then-right order.
        let left_term = cb_leaf_score_term::<F>(left_sum, left_w, lambda, score_fn);
        let right_term = cb_leaf_score_term::<F>(right_sum, right_w, lambda, score_fn);
        let folded = left_term + right_term;

        if score_fn == comptime!(SCORE_FN_L2) {
            score = folded;
        }
        if score_fn == comptime!(SCORE_FN_SOLAR_L2) {
            score = folded;
        }
        if score_fn == comptime!(SCORE_FN_LOO_L2) {
            score = folded;
        }
        if score_fn == comptime!(SCORE_FN_SAT_L2) {
            score = folded;
        }

        // --- Cosine / NewtonCosine (SCORE_FN_COSINE): numerator = the SAME `folded`
        // L2 fold (Σ avg·sum); denominator = seeded `1e-100 + Σ avg²·weight` with the
        // `1e-100` as the FIRST summand (score.rs:76-84, the seed-first accumulation
        // order matching the CPU), then `score = num / sqrt(den)`. `avg` reuses
        // `cb_leaf_avg` (the SAME `count>0`-guarded `calc_average`), in the SAME
        // left-then-right leaf order, so the value is byte-identical to the prior arm.
        if score_fn == comptime!(SCORE_FN_COSINE) {
            let left_avg = cb_leaf_avg::<F>(left_sum, left_w, lambda);
            let right_avg = cb_leaf_avg::<F>(right_sum, right_w, lambda);
            let mut denominator = F::new(1e-100);
            denominator += left_avg * left_avg * left_w;
            denominator += right_avg * right_avg * right_w;
            score = folded / denominator.sqrt();
        }

        scores[c] = score;

        // Update this thread's running best with the strict-first-wins / lowest-index
        // tie-break: take the candidate only if its score STRICTLY exceeds the running
        // best (a later equal-gain candidate never displaces an earlier one → lowest
        // index wins on a tie, matching select_best_candidate's strict `>`).
        //
        // WR-05: the trailing `border == n_bins - 1` candidate puts ALL bins in the LEFT
        // leaf / NONE in the RIGHT leaf — a NON-SPLIT (no-op) that upstream and the
        // pairwise path (which uses `n_splits = n_bins - 1`) NEVER enumerate as a real
        // split. It must NOT be argmin-eligible. We still WRITE `scores[c]` for that slot
        // (so the per-candidate `scores` buffer geometry stays `n_features * n_bins` —
        // buffer allocation, the `feature * n_bins + bin` index decode, and the
        // element-wise `max_divergence` oracle compare are all unchanged, and no `-inf`
        // sentinel that would produce NaN under that compare is introduced) but we SKIP
        // the argmin update for it. The host reference winner decode
        // (`reference_best_split` in `score_split.rs`) and the host winner decode
        // (`gpu_runtime.rs`) skip the SAME trailing border in EXACT lockstep, so device
        // and CPU oracle agree on a real (`border < n_bins - 1`) split.
        if border < n_bins_usize - 1usize {
            if score > my_gain {
                my_gain = score;
                my_idx = c as u32;
            }
        }

        c += CUBE_DIM_X as usize;
    }

    // Block-reduce the per-thread bests into ONE (gain, candidate-index) winner for the
    // cube, with the lowest-index tie-break. Wave-agnostic shared-mem tree (D-09): the
    // SIZE is the comptime ARGMIN_SHMEM; the stride starts at CUBE_DIM_X / 2 (the runtime
    // cube width), never a literal 32/64.
    let mut sh_gain = SharedMemory::<F>::new(ARGMIN_SHMEM);
    let mut sh_idx = SharedMemory::<u32>::new(ARGMIN_SHMEM);
    sh_gain[tid as usize] = my_gain;
    sh_idx[tid as usize] = my_idx;
    sync_cube();

    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s {
            let other_gain = sh_gain[(tid + s) as usize];
            let other_idx = sh_idx[(tid + s) as usize];
            let cur_gain = sh_gain[tid as usize];
            let cur_idx = sh_idx[tid as usize];
            // Keep the higher gain; on an EXACT tie keep the LOWER candidate index
            // (== lowest (feature, bin) — strict first-wins parity, Pitfall 1).
            let mut take_other = false;
            if other_gain > cur_gain {
                take_other = true;
            } else if other_gain == cur_gain {
                if other_idx < cur_idx {
                    take_other = true;
                }
            }
            if take_other {
                sh_gain[tid as usize] = other_gain;
                sh_idx[tid as usize] = other_idx;
            }
        }
        sync_cube();
        s /= 2u32;
    }

    if tid == 0u32 {
        best_gain[CUBE_POS] = sh_gain[0usize];
        best_idx[CUBE_POS] = sh_idx[0usize];
    }
}

/// Device-resident **scan/update** over the FROZEN 7.3 device-resident 2-channel
/// histogram handle (GPU-01 scan-update slice, Phase 7.5 Plan B; D-7.5-03 — the
/// `ScanPointwiseHistograms` / `UpdatePointwiseHistograms` transform 7.3 deferred).
///
/// It turns the per-bin `(Σ der1, Σ weight)` histogram into cumulative
/// "left-of-border" leaf stats: for each feature `f`, channel `c`, and border `b`,
/// `cumulative[(f * n_bins + b) * 2 + c] = Σ_{bin = 0}^{b} bin_sums[(f, bin, c)]`
/// (an INCLUSIVE prefix-sum over the per-feature bin axis). A candidate split at
/// border `b` then reads `left = cumulative[b]`, `right = cumulative[n_bins - 1] -
/// cumulative[b]` — the upstream `FindOptimalSplitSingleFoldImpl` convention
/// (`pointwise_scores.cu:259-263`, `weightRight = part.Weight - weightLeft`). The
/// output is therefore directly consumable by the Plan-A scorer.
///
/// # Launch geometry / reuse of the block-scan mechanism
///
/// ONE cube per `(feature, channel)` pair: `CUBE_POS` decodes `feature = CUBE_POS /
/// 2`, `channel = CUBE_POS % 2`. Within the cube each unit `UNIT_POS` owns one bin
/// `0..n_bins` and reads that bin's channel value from the FROZEN 2-channel layout
/// `(feature * n_bins + bin) * 2 + channel`. The prefix-sum itself REUSES the exact
/// wave-agnostic single-cube scan mechanism of [`block_scan_kernel`] VERBATIM (the
/// within-plane `plane_inclusive_sum` + the Hillis-Steele cross-plane carry over
/// per-plane partials), so the bin axis is scanned with NO hand-rolled scan and NO
/// warp/wave-size literal in any stride (D-09 — strides derive from `CUBE_DIM_X` /
/// `PLANE_DIM`). The `#[comptime] inclusive` flag is fixed to `true` here (the
/// "left of and INCLUDING border b" cumulative the scorer needs).
///
/// # SCOPE (RESEARCH A1 / Open Q1) — single-cube precondition
///
/// Like the underlying `block_scan_kernel`, this is correct only for `n_bins <=
/// CUBE_DIM` (exactly one plane on wave32 gfx1100, where the cross-plane carry
/// collapses to the identity). The CROSS-CUBE carry for `n_bins > CUBE_DIM` (8-bit,
/// 256-bin features) is NOT performed here; the host seam
/// [`crate::gpu_runtime::launch_scan_update_pointwise`] enforces the precondition
/// with a typed error (the EXPLICIT tracked forward dependency — NOT a silent cut).
///
/// # Precondition — `bin_sums` validity (IN-05)
///
/// This kernel consumes an ALREADY-FILLED `bin_sums` device handle and does NOT
/// re-validate the quantized-bin (`cindex`) value range itself. Its correctness relies
/// IMPLICITLY on the producing fill path having range-guarded `cindex` before populating
/// `bin_sums` (the `cindex` value-range guards in
/// [`crate::gpu_runtime::launch_scan_update_pointwise_into`], which fills then scans in
/// one place). Any FUTURE caller that supplies an EXTERNALLY-produced `bin_sums` (one not
/// built through that guarded fill path) MUST enforce the same `cindex < n_bins` /
/// layout-length preconditions before calling this kernel — otherwise a malformed
/// `bin_sums` would be prefix-summed verbatim into a wrong cumulative buffer.
///
/// Generic over `F: Float` (AGENTS.md generics-float). Every device read/write is
/// under a POSITION bounds guard. if-as-STATEMENT only (CubeCL conditionals manual).
#[cube(launch)]
pub fn scan_update_pointwise_kernel<F: Float>(
    bin_sums: &Array<F>,
    cumulative: &mut Array<F>,
    n_bins: u32,
) {
    let tid = UNIT_POS;
    let n_bins_usize = n_bins as usize;

    // Decode which (feature, channel) axis this cube scans. Two channels per feature
    // (channel 0 = Σ der1, channel 1 = Σ weight), so cube `k` handles feature `k / 2`,
    // channel `k % 2`. The flat cell index in the FROZEN 2-channel layout is
    // `(feature * n_bins + bin) * 2 + channel`.
    let feature = CUBE_POS / 2usize;
    let channel = CUBE_POS % 2usize;

    // Load this unit's bin value for the (feature, channel) axis, zero-padding idle
    // out-of-range lanes (this cube has CUBE_DIM units; only the first n_bins own a
    // bin). if-as-STATEMENT: init to 0, overwrite inside the bounds guard.
    let mut val = F::new(0.0);
    if tid < n_bins {
        let cell = (feature * n_bins_usize + tid as usize) * 2usize + channel;
        if cell < bin_sums.len() {
            val = bin_sums[cell];
        }
    }

    // --- Inclusive prefix-sum over the bin axis, REUSING the block_scan_kernel
    //     mechanism VERBATIM (within-plane plane scan + Hillis-Steele cross-plane
    //     carry over per-plane partials). inclusive = true (cumulative includes self).

    // 1) Within-plane inclusive prefix (width = PLANE_DIM, never a literal).
    let scanned_in_plane = plane_inclusive_sum(val);
    let scanned = scanned_in_plane;

    // 2) Cross-plane carry: the LAST unit of each plane writes that plane's inclusive
    //    total into a per-plane shared slot keyed by PLANE_POS.
    let mut partials = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
    if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
        partials[PLANE_POS as usize] = scanned_in_plane;
    }
    sync_cube();

    // Number of planes in this cube = ceil(CUBE_DIM_X / PLANE_DIM) (== 1 on wave32 at
    // CUBE_DIM 32 — the carry below then adds nothing). The stride bound derives from
    // CUBE_DIM_X / PLANE_DIM, NOT a literal 32/64 (D-09).
    let num_planes = (CUBE_DIM_X + PLANE_DIM - 1u32) / PLANE_DIM;

    // Hillis-Steele inclusive scan over the per-plane partials.
    let mut s = 1u32;
    while s < num_planes {
        let mut add = F::new(0.0);
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

    // 3) Each plane's exclusive carry = sum of all strictly-prior planes' totals.
    let mut carry = F::new(0.0);
    if PLANE_POS >= 1u32 {
        carry = partials[(PLANE_POS - 1u32) as usize];
    }

    let result = scanned + carry;

    // Write the inclusive cumulative for this (feature, channel, bin) back into the
    // FROZEN 2-channel layout (only the n_bins real bins; idle lanes write nothing).
    if tid < n_bins {
        let out_cell = (feature * n_bins_usize + tid as usize) * 2usize + channel;
        if out_cell < cumulative.len() {
            cumulative[out_cell] = result;
        }
    }
}

/// Device-resident **partition split** — the per-object doc-routing reorder that
/// extends the current per-object leaf assignment by ONE level (GPU-01 grow-loop
/// slice, Phase 7.5 Plan C; D-7.5-02). Mirrors upstream `TSubsetsHelper::Split`'s
/// in-place subset reorder, but expressed as a forward-bit leaf-index update so it
/// matches the CPU `cb_train::leaf_index` convention EXACTLY (Pitfall 6,
/// parity-critical: an off-by-one in the bit order silently permutes leaves).
///
/// # Forward-bit leaf convention (parity-critical, Pitfall 6)
///
/// For the split chosen at level `d` on `(feature, bin)`, every object whose
/// quantized bin on `feature` is STRICTLY GREATER than `bin` "passes" the split and
/// gets bit `d` set: `new_leaf_of[obj] = leaf_of[obj] | (pass ? (1 << level_bit) :
/// 0)`. This is `idx |= 1usize << i` from `cb_train::leaf_index` (`tree.rs:272-280`)
/// with `i == level_bit == d`. The `> bin` test mirrors the cross-oracle's CPU
/// `value > border` split mapped onto the quantized bin axis (border `b` ↔ bin > b),
/// so the device leaf assignment is bit-identical to the CPU `assign_leaves`/`leaf_index`.
///
/// The routing stays ENTIRELY device-resident: `leaf_of` (in) and `new_leaf_of` (out)
/// are device handles; the bulk doc-routing is NEVER read back to host (D-05). Only
/// the O(1) `(feature, bin)` decision crosses host→device as the launch scalars.
///
/// # Wave-size policy (D-09) / generics-float (AGENTS.md)
///
/// The per-object loop is a grid-stride loop over the total thread count
/// (`CUBE_COUNT * CUBE_DIM`), a topology-derived value — NEVER a literal 32/64. The
/// kernel is generic over `F: Float` (AGENTS.md generics-float): the resident der1
/// handle is threaded in as `&Array<F>` so the SAME persistent float buffer the grow
/// loop already holds is bound without a fresh upload; the routing itself reads only
/// the integer `cindex` (a `_ = der1.len()` keeps the generic real without a value
/// read). Every device read is under a POSITION bounds guard; the `feature`/`bin`/
/// object VALUE ranges are validated HOST-SIDE before launch. if-as-STATEMENT only
/// (CubeCL conditionals manual).
///
/// `leaf_of`/`new_leaf_of` are length `n` (per object, object order). `cindex` is the
/// quantized bin matrix laid out feature-major (`cindex[feature * n + obj]`).
/// `indices` (length `n`) is the object visiting order. `feature`/`bin`/`level_bit`
/// are the chosen split's feature index, split border (bin), and the level's leaf bit.
#[cube(launch)]
pub fn partition_split_kernel<F: Float>(
    der1: &Array<F>,
    cindex: &Array<u32>,
    indices: &Array<u32>,
    leaf_of: &Array<u32>,
    new_leaf_of: &mut Array<u32>,
    feature: u32,
    bin: u32,
    level_bit: u32,
) {
    // Keep the `F: Float` generic real (AGENTS.md generics-float) while routing on the
    // integer bin axis only — the resident der1 handle is threaded but not value-read
    // here (the split decision is purely on the quantized bins).
    let _ = der1.len();

    let n = indices.len();

    // Grid-stride loop over the object-visiting order (the stride is the total thread
    // count CUBE_COUNT * CUBE_DIM — a topology value, NEVER a literal 32/64, D-09).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut i = ABSOLUTE_POS;
    while i < n {
        let obj = indices[i] as usize;
        // The feature-major cindex stride is `feature * n + obj` (n == indices.len()):
        // the SAME layout the histogram fill reads.
        let cell = (feature as usize) * n + obj;
        let mut new_leaf = leaf_of[obj];
        // Forward-bit: object passes (gets bit `level_bit`) iff its bin > the border.
        if cindex[cell] > bin {
            new_leaf = new_leaf | (1u32 << level_bit);
        }
        new_leaf_of[obj] = new_leaf;
        i += stride;
    }
}

/// Device-resident **partition update** — the per-partition `Σ der1` / `Σ weight`
/// reduce after a split (GPU-01 grow-loop slice, Phase 7.5 Plan C; D-7.5-02). Mirrors
/// upstream `UpdatePartitionProps` / `PartitionUpdateImpl` (`pointwise_scores.cu:624-697`):
/// recompute each new leaf's summed first-derivative and summed weight so the leaf
/// values can be estimated from the device-resident partition without re-reading the
/// full doc routing to host.
///
/// # Per-partition reduce (D-03 in-kernel atomic + f64 finalize)
///
/// Each object atomic-adds its `(der1[obj], weight[obj])` into its partition's two
/// channels of the global `part_stats` buffer at `part_stats[leaf_of[obj] * 2 + 0]`
/// (Σ der1) and `part_stats[leaf_of[obj] * 2 + 1]` (Σ weight). The cross-thread merge
/// is ALWAYS the in-kernel `Atomic<F>::fetch_add` (the `block_reduce_atomic_kernel`
/// `acc[0].fetch_add(...)` primitive generalized to a per-partition-indexed buffer,
/// D-03); the accumulation ORDER is non-deterministic (the accepted D-03 float-order
/// variance, REPORTED not signed off). The channel float type is f64 on rocm/cuda/cpu
/// and f32 on wgpu (RESEARCH A1), matching the histogram channel. ALWAYS runs the
/// in-kernel atomic — never a host-fallback selector mid-loop (Pitfall 4).
///
/// # Wave-size policy (D-09) / generics-float (AGENTS.md)
///
/// The per-object loop is a grid-stride loop over the total thread count
/// (`CUBE_COUNT * CUBE_DIM`) — NEVER a literal 32/64. Generic over `F: Float`
/// (AGENTS.md generics-float). Every device read is under a POSITION bounds guard; the
/// `leaf_of` partition VALUE range (`< n_parts`) is validated HOST-SIDE before launch
/// so the atomic store cannot address `part_stats` out of bounds. if-as-STATEMENT only.
///
/// `der1` (UNWEIGHTED, the 7.2 seam contract) / `weight` are length `n` (per object,
/// object order). `leaf_of` (length `n`) is the per-object partition (`0..n_parts`),
/// produced DEVICE-SIDE by `partition_split_kernel` and consumed here WITHOUT a host
/// read-back/re-validation. Because the host therefore cannot vouch for the partition
/// VALUE range (WR-04), the atomic store is guarded in-kernel by `part * 2 + 1 <
/// part_stats.len()` (matching the scan kernel's `cell < bin_sums.len()` precedent) so a
/// drifting `leaf_of` — e.g. a future depth>1 partition that mis-numbers a leaf — can
/// never address `part_stats` out of bounds (which would be a device-atomic UB). `indices`
/// (length `n`) is the object visiting order. `part_stats` is length `n_parts * 2`
/// (zero-initialised by the host), channel 0 = Σ der1, channel 1 = Σ weight.
#[cube(launch)]
pub fn partition_update_kernel<F: Float>(
    der1: &Array<F>,
    weight: &Array<F>,
    indices: &Array<u32>,
    leaf_of: &Array<u32>,
    part_stats: &Array<Atomic<F>>,
) {
    let n = indices.len();

    // Grid-stride loop over the object-visiting order (stride == total thread count,
    // a topology value — NEVER a literal 32/64, D-09).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut i = ABSOLUTE_POS;
    while i < n {
        let obj = indices[i] as usize;
        let part = leaf_of[obj] as usize;
        let d = der1[obj];
        let w = weight[obj];
        // In-kernel atomic merge (D-03): the per-partition Σ der1 / Σ weight, folded
        // device-resident. `leaf_of` is DEVICE-PRODUCED (no host read-back), so guard the
        // partition VALUE range in-kernel (WR-04) instead of relying on a host check that
        // never runs — `part * 2 + 1 < part_stats.len()` ensures BOTH channel stores are
        // in bounds (matching the scan kernel's `cell < bin_sums.len()` precedent).
        // if-as-STATEMENT only.
        if part * 2usize + 1usize < part_stats.len() {
            part_stats[part * 2usize].fetch_add(d);
            part_stats[part * 2usize + 1usize].fetch_add(w);
        }
        i += stride;
    }
}

// ===========================================================================
// Phase 7.5 Plan 06 — the PAIRWISE split scorer (split_pairwise.cuh), the
// genuinely-new structurally-heaviest piece (Pitfall 5), sequenced LAST per
// D-7.5-01. It closes the GPU-01 kernel surface: the per-leaf linear-system build
// from the FROZEN 7.4 4-channel pairwise histogram + the der-sum scatter
// (== upstream `MakePairwiseDerivatives` / `MakePointwiseDerivatives`), the
// pairwise scan/update over the 4-channel handle (D-7.4-06 deferred-from-7.4),
// and the deterministic best-split argmin (== `SelectBestSplit`). The small
// per-leaf dense Cholesky solve + the `CalculateScore` fold run host-side over the
// BOUNDED assembled per-(feature,bucket) statistics (RESEARCH Open Q3: a
// `#[cube]` dense SPD solve is awkward and the FROZEN CPU
// `cb_compute::pairwise_cholesky_solve` is the parity oracle), so the bulk
// pairwise histogram stays device-resident and only the assembled
// `O(n_features * bucket_count)` der-sum + pair-weight statistics descriptor
// crosses to host (minimal round-trips, D-05). See
// `crates/cb-backend/src/gpu_runtime.rs::launch_pairwise_split_score`.
// ===========================================================================

/// 4-channel pairwise **scan/update** — the post-fill transform deferred from 7.4
/// (`UpdatePairwiseHistograms`/`ScanPairwiseHistograms`, D-7.4-06) over the FROZEN
/// 4-channel `(feature * n_bins + bin) * 4 + histId` layout. The pairwise sibling of
/// [`scan_update_pointwise_kernel`]: where the pointwise scan does an inclusive
/// prefix over the 2-channel (Σ der1, Σ weight) bin axis, this does the SAME
/// wave-agnostic inclusive prefix over EACH of the 4 weight-only channels
/// (`histId in {0,1,2,3}`) so the scorer can read cumulative "left-of-border" pair
/// weights per channel.
///
/// # Reuse of the block-scan mechanism (VERBATIM, D-09)
///
/// ONE cube per `(feature, histId)` pair: `CUBE_POS` decodes `feature = CUBE_POS /
/// 4`, `histId = CUBE_POS % 4`. Within the cube each unit `UNIT_POS` owns one bin
/// `0..n_bins` and reads that bin's channel from `(feature * n_bins + bin) * 4 +
/// histId`. The prefix-sum REUSES the exact wave-agnostic single-cube scan mechanism
/// of [`block_scan_kernel`] / [`scan_update_pointwise_kernel`] VERBATIM (the
/// within-plane `plane_inclusive_sum` + the Hillis-Steele cross-plane carry over
/// per-plane partials), so NO hand-rolled scan and NO warp/wave-size literal in any
/// stride (D-09 — strides derive from `CUBE_DIM_X` / `PLANE_DIM`). `inclusive = true`.
///
/// # SCOPE (RESEARCH Open Q3 / inherited from Plan B) — single-cube precondition
///
/// Like the underlying `block_scan_kernel`, this is correct only for `n_bins <=
/// CUBE_DIM` (one plane on wave32 gfx1100, where the cross-plane carry collapses to
/// the identity). The CROSS-CUBE carry for `n_bins > CUBE_DIM` (8-bit, 256-bin
/// features) is NOT performed here; the host seam
/// [`crate::gpu_runtime::launch_scan_update_pairwise`] enforces the precondition with
/// a typed error (the EXPLICIT tracked cross-cube-carry follow-up — NOT a silent cut).
///
/// Generic over `F: Float` (AGENTS.md generics-float). Every device read/write is
/// under a POSITION bounds guard. if-as-STATEMENT only (CubeCL conditionals manual).
#[cube(launch)]
pub fn scan_update_pairwise_kernel<F: Float>(
    bin_sums: &Array<F>,
    cumulative: &mut Array<F>,
    n_bins: u32,
) {
    let tid = UNIT_POS;
    let n_bins_usize = n_bins as usize;

    // Decode which (feature, histId) axis this cube scans. Four channels per feature
    // (the FROZEN 4-channel pairwise layout), so cube `k` handles feature `k / 4`,
    // histId `k % 4`. The flat cell index is `(feature * n_bins + bin) * 4 + histId`.
    let feature = CUBE_POS / 4usize;
    let hist_id = CUBE_POS % 4usize;

    // Load this unit's bin value for the (feature, histId) axis, zero-padding idle
    // out-of-range lanes (this cube has CUBE_DIM units; only the first n_bins own a
    // bin). if-as-STATEMENT: init to 0, overwrite inside the bounds guard.
    let mut val = F::new(0.0);
    if tid < n_bins {
        let cell = (feature * n_bins_usize + tid as usize) * 4usize + hist_id;
        if cell < bin_sums.len() {
            val = bin_sums[cell];
        }
    }

    // --- Inclusive prefix-sum over the bin axis, REUSING the block_scan_kernel
    //     mechanism VERBATIM (within-plane plane scan + Hillis-Steele cross-plane
    //     carry over per-plane partials). inclusive = true (cumulative includes self).

    // 1) Within-plane inclusive prefix (width = PLANE_DIM, never a literal).
    let scanned_in_plane = plane_inclusive_sum(val);
    let scanned = scanned_in_plane;

    // 2) Cross-plane carry: the LAST unit of each plane writes that plane's inclusive
    //    total into a per-plane shared slot keyed by PLANE_POS.
    let mut partials = SharedMemory::<F>::new(BLOCK_REDUCE_SHMEM);
    if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
        partials[PLANE_POS as usize] = scanned_in_plane;
    }
    sync_cube();

    // Number of planes = ceil(CUBE_DIM_X / PLANE_DIM) (== 1 on wave32 at CUBE_DIM 32 —
    // the carry below then adds nothing). The stride bound derives from CUBE_DIM_X /
    // PLANE_DIM, NOT a literal 32/64 (D-09).
    let num_planes = (CUBE_DIM_X + PLANE_DIM - 1u32) / PLANE_DIM;

    // Hillis-Steele inclusive scan over the per-plane partials.
    let mut s = 1u32;
    while s < num_planes {
        let mut add = F::new(0.0);
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

    // 3) Each plane's exclusive carry = sum of all strictly-prior planes' totals.
    let mut carry = F::new(0.0);
    if PLANE_POS >= 1u32 {
        carry = partials[(PLANE_POS - 1u32) as usize];
    }

    let result = scanned + carry;

    // Write the inclusive cumulative for this (feature, histId, bin) back into the
    // FROZEN 4-channel layout (only the n_bins real bins; idle lanes write nothing).
    if tid < n_bins {
        let out_cell = (feature * n_bins_usize + tid as usize) * 4usize + hist_id;
        if out_cell < cumulative.len() {
            cumulative[out_cell] = result;
        }
    }
}

/// **Pairwise make-derivatives** — the per-(feature, bucket) der-sum scatter that
/// builds the pointwise der portion of the pairwise linear system (== upstream
/// `MakePointwiseDerivatives` over the single root leaf, the per-leaf system row the
/// pairwise scorer's `der_sum[2*leaf+1] += Σ_bucket der_sums[leaf][bucket]` consumes;
/// `split_pairwise.cuh:11-18`). For the depth-1 MVP there is ONE leaf (the root), so
/// this scatters every object's pairwise-weighted `der1[obj]` into its feature bucket:
/// `der_sums[feature * n_bins + bin] += der1[obj]`, bin = `cindex[feature * n + obj]`.
///
/// This is the device twin of the host `cb_compute::compute_der_sums` (it produces the
/// SAME `der_sums[leaf=0][bucket]` row, flattened per feature) — the heavy per-object
/// scatter stays on device (D-03 in-kernel atomic), and only the bounded
/// `n_features * n_bins` der-sum descriptor crosses to host for the small per-leaf
/// Cholesky solve (RESEARCH Open Q3). The 4-channel pair-weight statistics come from
/// the FROZEN 7.4 fill ([`pairwise_hist_nonbinary_kernel`]); this kernel adds the
/// pointwise der row.
///
/// # Per-bucket reduce (D-03 in-kernel atomic + f64 finalize)
///
/// Each object atomic-adds its `der1[obj]` into `der_sums[feature * n_bins + bin]` for
/// every feature. The cross-thread merge is ALWAYS the in-kernel `Atomic<F>::fetch_add`
/// (D-03); the channel float type is f64 on rocm/cuda/cpu, f32 on wgpu (RESEARCH A1).
///
/// # Wave-size policy (D-09) / generics-float (AGENTS.md)
///
/// The per-object loop is a grid-stride loop over the total thread count
/// (`CUBE_COUNT * CUBE_DIM`) — NEVER a literal 32/64. Generic over `F: Float`. Every
/// device read is under a POSITION bounds guard; the bin/object VALUE ranges are
/// validated HOST-SIDE before launch. if-as-STATEMENT only.
///
/// `der1` (the pairwise-weighted first derivative, length `n`), `cindex` (feature-major
/// quantized bins, `cindex[feature * n + obj]`, length `n_features * n`), `indices`
/// (object visiting order, length `n`), `der_sums` (length `n_features * n_bins`,
/// zero-initialised). `n_features` the feature-group width.
#[cube(launch)]
pub fn pairwise_make_derivatives_kernel<F: Float>(
    der1: &Array<F>,
    cindex: &Array<u32>,
    indices: &Array<u32>,
    der_sums: &mut Array<Atomic<F>>,
    n_features: u32,
    #[comptime] n_bins: u32,
) {
    let n = indices.len();
    let n_bins_usize = n_bins as usize;
    let n_features_usize = n_features as usize;

    // Grid-stride loop over the object-visiting order (stride == total thread count,
    // a topology value — NEVER a literal 32/64, D-09).
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut i = ABSOLUTE_POS;
    while i < n {
        let obj = indices[i] as usize;
        let d = der1[obj];
        let mut feature = 0usize;
        while feature < n_features_usize {
            // The feature-major cindex stride is `feature * n + obj` (n == indices.len()).
            let bin = cindex[feature * n + obj] as usize;
            // der_sums[feature * n_bins + bin] += der1[obj] (per-leaf=root row scatter,
            // == compute_der_sums for leaf_count==1). The host validated bin < n_bins.
            let cell = feature * n_bins_usize + bin;
            der_sums[cell].fetch_add(d);
            feature += 1usize;
        }
        i += stride;
    }
}

/// Comptime `SharedMemory` size for the [`select_best_split_kernel`] argmin tree —
/// the SAME `BLOCK_REDUCE_SHMEM` (== `CUBE_DIM`) the [`find_optimal_split_kernel`]
/// argmin uses ([`ARGMIN_SHMEM`]). One shared slot per unit. NOT a wave/warp-size
/// literal in any stride (D-09).
pub(crate) const PAIRWISE_ARGMIN_SHMEM: usize = BLOCK_REDUCE_SHMEM;

/// **Select best split** — the deterministic argmax over the host-solved pairwise
/// scores with the SAME lowest-(candidate)-index tie-break as [`find_optimal_split_kernel`]
/// (== upstream `SelectBestSplit`, `split_pairwise.cuh:27-31`). The pairwise scores are
/// solved host-side (the small per-leaf Cholesky; RESEARCH Open Q3) and uploaded as a
/// per-candidate `scores` array (`scores[feature * (bucket_count-1) + border]`); this
/// kernel reduces them to ONE best `(candidate-index, score)` per cube via the
/// wave-agnostic shared-mem tree-reduce, mirroring the `find_optimal_split_kernel`
/// argmin VERBATIM. Threading the argmin through a device kernel keeps the
/// best-candidate selection device-resident (only the O(1) winner descriptor crosses
/// back), structurally matching `SelectBestSplit`.
///
/// `n_candidates` is the total scored candidate count (`n_features * (bucket_count-1)`).
/// `best_gain`/`best_idx` carry one winner per cube (length = cube count). Ties keep the
/// LOWER candidate index (strict-`>` first-wins, == `select_best_candidate`). Generic
/// over `F: Float`. if-as-STATEMENT only. Every shared/global access is bounds-guarded.
#[cube(launch)]
pub fn select_best_split_kernel<F: Float>(
    scores: &Array<F>,
    best_gain: &mut Array<F>,
    best_idx: &mut Array<u32>,
    n_candidates: u32,
) {
    let tid = UNIT_POS;
    let n_candidates_usize = n_candidates as usize;

    // The minimal-score sentinel any real candidate must beat (the
    // `score.rs::MINIMAL_SCORE` = `f64::NEG_INFINITY` analogue). It must be more negative
    // than every realizable pairwise score (so each candidate wins the first strict-greater
    // compare) AND than every other lane's gain (so an unseen lane loses the reduction). The
    // finite `f32::MIN` (~-3.4e38) satisfies both for gradient-derived scores, identical to
    // `f64::NEG_INFINITY` for all reachable inputs.
    //
    // WR-01 (rocm re-confirm): a literal `-inf` (`F::new(f32::NEG_INFINITY)`) is NOT usable —
    // CubeCL emits `double(-inf)`, which the HIP/comgr compiler rejects on gfx1100
    // (`use of undeclared identifier 'inf'`), failing the kernel JIT. `f32::MIN` is the
    // HIP-safe sentinel.
    let minimal_score = F::new(f32::MIN);

    // This thread's running best over the candidates it strides through. `my_idx` is the
    // candidate index; ties keep the LOWER index, so seed it to the max so any real
    // candidate replaces it on the first strict-greater compare.
    let mut my_gain = minimal_score;
    let mut my_idx = n_candidates;

    // Grid-stride over candidates (D-09: the stride is CUBE_DIM_X, a topology value).
    let mut c = tid as usize;
    while c < n_candidates_usize {
        let g = scores[c];
        // STRICT `>` (first-wins on equal score, ascending candidate index).
        if g > my_gain {
            my_gain = g;
            my_idx = c as u32;
        }
        c += CUBE_DIM_X as usize;
    }

    // Shared-mem tree-reduce argmax with the lowest-index tie-break (mirrors the
    // find_optimal_split_kernel argmin VERBATIM). SIZE is the comptime
    // PAIRWISE_ARGMIN_SHMEM; the stride starts at CUBE_DIM_X / 2 and halves.
    let mut sh_gain = SharedMemory::<F>::new(PAIRWISE_ARGMIN_SHMEM);
    let mut sh_idx = SharedMemory::<u32>::new(PAIRWISE_ARGMIN_SHMEM);
    sh_gain[tid as usize] = my_gain;
    sh_idx[tid as usize] = my_idx;
    sync_cube();

    let mut step = CUBE_DIM_X / 2u32;
    while step >= 1u32 {
        if tid < step {
            let other_gain = sh_gain[(tid + step) as usize];
            let other_idx = sh_idx[(tid + step) as usize];
            let cur_gain = sh_gain[tid as usize];
            let cur_idx = sh_idx[tid as usize];
            // Higher gain wins; on an EXACT tie the LOWER candidate index wins.
            let mut take = false;
            if other_gain > cur_gain {
                take = true;
            }
            if other_gain == cur_gain {
                if other_idx < cur_idx {
                    take = true;
                }
            }
            if take {
                sh_gain[tid as usize] = other_gain;
                sh_idx[tid as usize] = other_idx;
            }
        }
        sync_cube();
        step /= 2u32;
    }

    // Unit 0 writes this cube's winner.
    if tid == 0u32 {
        best_gain[CUBE_POS] = sh_gain[0usize];
        best_idx[CUBE_POS] = sh_idx[0usize];
    }
}

#[cfg(test)]
mod gradient_gpu;

// Device-resident 2-channel pointwise histogram self-oracle (GPU-01 histogram slice,
// Phase 7.3): the GPU `pointwise_hist2` 8-bit non-binary fill over `SelectedRuntime`
// vs an ORDERED host-reference 2-channel histogram (`cb-core::sum_f64` leaf->bin
// generalization), plus the D-7.3-05 device-residency hand-off assertion, live in
// `kernels/pointwise_hist.rs`, mounted at `kernels::pointwise_hist`. Like
// `gradient_gpu` (and UNLIKE the cpu-only `gradient`/`scatter` spikes), it runs over
// the generic `SelectedRuntime`, so it builds/runs under EVERY backend (the rocm
// in-env oracle on gfx1100 + the wgpu host run + cuda compile-only).
#[cfg(test)]
mod pointwise_hist;

// Device-resident 4-channel WEIGHT-ONLY pairwise histogram self-oracle (GPU-01
// histogram slice, Phase 7.4 — the pairwise SIBLING of `pointwise_hist`): the GPU
// `pairwise_hist` non-binary fill (comptime `bits` in {5,6,7}) over `SelectedRuntime`
// vs an ORDERED host-reference 4-channel pairwise histogram (`cb-core::sum_f64`
// per-pair generalization), plus the D-7.4-03 device-residency hand-off assertion and
// the SC-2 PairLogitPairwise fixture, live in `kernels/pairwise_hist.rs`, mounted at
// `kernels::pairwise_hist`. Like `pointwise_hist`/`gradient_gpu` (and UNLIKE the
// cpu-only `gradient`/`scatter` spikes), it runs over the generic `SelectedRuntime`, so
// it builds/runs under EVERY backend (the rocm in-env oracle on gfx1100 + the wgpu host
// run + cuda compile-only).
#[cfg(test)]
mod pairwise_hist;

// Device-resident pointwise L2 split-score + deterministic split-argmin self-oracle
// (GPU-01 score/split slice, Phase 7.5 Plan A — the scorer SIBLING of `pointwise_hist`):
// the GPU `find_optimal_split_kernel` (L2 arm) over `SelectedRuntime`, computing the
// per-(feature,bin) L2 split score from the FROZEN 7.3 device-resident 2-channel
// histogram and a deterministic block-reduce argmin (lowest-index tie-break), is
// cross-oracled against the FROZEN CPU references `cb_compute::l2_split_score` (score)
// and `cb_train::select_best_candidate` (winner / strict first-wins). Lives in
// `kernels/score_split.rs`, mounted at `kernels::score_split`. Like
// `pointwise_hist`/`pairwise_hist` it runs over the generic `SelectedRuntime`, so it
// builds/runs under EVERY backend (the rocm in-env oracle on gfx1100 + the wgpu host run
// + cuda compile-only). REPORTS divergence; the GPU-06 epsilon is 7.6's job.
// IN-04: shared `#[cfg(test)]` fixture-construction primitives composed by the Phase 7.5
// cross-oracle modules `score_split` and `grow_loop`. Factors ONLY the genuinely-shared
// per-channel construction (the centred ramp, the mod-5 weight, the feature-major cindex,
// the identity indices, the competitor-pair list); each builder still emits byte-identical
// oracle inputs. Lives in `kernels/test_fixtures.rs`, mounted at `kernels::test_fixtures`.
#[cfg(test)]
mod test_fixtures;

#[cfg(test)]
mod score_split;

// Device-resident host-light single-tree grow-loop cross-oracle (GPU-01 grow-loop
// slice, Phase 7.5 Plan C — the integration SIBLING of `score_split`): the GPU
// `grow_oblivious_tree` driver (fill→scan→score+argmin→ONE O(1) BestSplit read-back→
// partition-split→partition-update per level, then ONE 2^depth part-stats read-back at
// the leaves) over `SelectedRuntime`, cross-oracled against a FROZEN-CPU-reference
// transcription of `cb_train::greedy_tensor_search_oblivious` + `cb_train::leaf_index`
// (transcribed INLINE — never importing cb-train, the Plan-A feature-unification
// landmine) and `cb_compute::calc_average` (imported read-only) for leaf values. The
// `partition_split_kernel` / `partition_update_kernel` doc-routing/reduce primitives are
// exercised here too. Lives in `kernels/grow_loop.rs`, mounted at `kernels::grow_loop`.
// Like `score_split`/`pointwise_hist` it runs over the generic `SelectedRuntime`, so it
// builds/runs under EVERY backend (the rocm in-env oracle on gfx1100 + the wgpu host run
// + cuda compile-only). STRUCTURE is the STRICT bar (asserted EXACT); leaf VALUES are
// REPORTED (the GPU-06 epsilon is 7.6's job).
#[cfg(test)]
mod grow_loop;

// GPU-06 tolerance MEASUREMENT harness (Phase 7.6 Plan 01 — the EVIDENCE roll-up the
// epsilon sign-off in Plan 02 consumes): aggregates the existing per-kernel-family
// divergence comparisons (der/hess, pointwise hist, pairwise hist, score/split,
// reduce) into one `[GPU-06 EVIDENCE]` line per family, adds an N≥30 run-to-run
// variance loop with stddev + an `observed_max + 3σ` headroom term, and measures the
// end-to-end GPU-vs-CPU model leaf values (the 7.5 REPORTED-not-signed-off numbers).
// Adds NO new kernel — it COMPOSES the sibling oracles over the generic
// `SelectedRuntime` (the rocm in-env oracle on gfx1100 + the wgpu host run + cuda/cpu
// compile). NEVER imports `cb-train` (feature-unification landmine — would activate
// `cb-backend/cpu` alongside `rocm` and fake a 0.0 divergence); every CPU reference is
// transcribed INLINE, `cb_compute`/`cb_core` read-only. Lives in
// `kernels/gpu_tolerance.rs`, mounted at `kernels::gpu_tolerance`. REPORTS divergence;
// the GPU-06 epsilon is signed off in Plan 02, NOT hard-coded here.
#[cfg(test)]
mod gpu_tolerance;
