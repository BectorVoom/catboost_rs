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
