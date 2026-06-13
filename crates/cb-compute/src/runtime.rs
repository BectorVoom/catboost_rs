//! The abstract `R: Runtime` / `F: Float` compute boundary (D-03/D-04). This
//! crate stays `cubecl`-free: it defines the trait the `cb-backend` CubeCL
//! `CpuRuntime` implements (and, in Phase 7, the GPU runtimes implement
//! additively). The host orchestration in `cb-train` is generic over `R: Runtime`
//! so swapping backends never touches the boosting loop.
//!
//! # Design (D-04 — coarse domain-level ops)
//!
//! The trait exposes ML-level operations, not raw kernel launches. For Wave 1 the
//! single coarse op is [`Runtime::compute_gradients`]: given the per-object raw
//! approximants and targets it returns the per-object first/second derivatives,
//! UN-reduced (D-02 — the backend kernel does order-independent elementwise work;
//! the parity-critical SUM is finalized host-side via `cb_core::sum_f64` in
//! `cb-compute`/`cb-train`). Histogram scatter and split evaluation are likewise
//! "return un-reduced buffers; host folds" — they are added to this trait as the
//! later slices need them; for the first slice the histogram reduction is done by
//! `cb-compute::histogram` directly over the gradient buffers, keeping the trait
//! minimal and the parity surface obvious.
//!
//! # No `cubecl` (D-03)
//!
//! Nothing in this module names `cubecl`. The associated [`Loss`] selector lets
//! the backend dispatch the correct elementwise derivative without `cb-backend`
//! depending on `cb-compute::loss` internals.

use cb_core::CbResult;

/// The numeric element type a runtime computes over. A pure marker bound so
/// `cb-compute` can stay generic without naming any backend's float trait
/// (`cubecl::Float`); the backend maps its concrete element types onto this.
///
/// `f32` and `f64` implement it (below). The parity-critical reductions are
/// always finalized in `f64` host-side regardless of the kernel element type.
pub trait Float: Copy + Send + Sync + 'static {}

impl Float for f32 {}
impl Float for f64 {}

/// Which loss's elementwise derivatives a [`Runtime::compute_gradients`] call
/// should emit. Lets the backend pick the right per-object kernel without
/// reaching into `cb-compute::loss`.
///
/// `Eq` is intentionally NOT derived: [`Loss::Focal`] carries `f64` parameters
/// (`alpha` / `gamma`), and `f64` is not `Eq`. `PartialEq` is retained for the
/// match-and-compare call sites.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Loss {
    /// RMSE (regression): der1 = `target - approx`, der2 = `-1`.
    Rmse,
    /// Logloss / CrossEntropy (binary classification): der1 = `target - p`,
    /// der2 = `-p*(1-p)`, `p = sigmoid(approx)` over the raw logit.
    Logloss,
    /// CrossEntropy (binary classification, D-09): IDENTICAL math to
    /// [`Loss::Logloss`] — der1 = `target - p`, der2 = `-p*(1-p)`,
    /// `p = sigmoid(approx)`. The only difference is the admissible target range
    /// (`[0,1]` probabilities vs `{0,1}` labels); the derivatives are the same, so
    /// the backend reuses the Logloss gradient/hessian kernels.
    CrossEntropy,
    /// Focal loss (binary classification, D-09) with `alpha` (class balance) and
    /// `gamma` (focusing exponent). der1/der2 transcribed verbatim from
    /// `error_functions.h:1684-1709` `TFocalError`; `p` is clamped to
    /// `[1e-13, 1-1e-13]` before the `log`/`pow` (T-04-02-02 — no NaN).
    Focal {
        /// Class-balance weight `alpha ∈ (0,1)` (`focal_alpha`).
        alpha: f64,
        /// Focusing exponent `gamma > 0` (`focal_gamma`).
        gamma: f64,
    },
    /// MAE / Quantile(alpha=0.5, delta=1e-6) (robust regression): der1 =
    /// `(target - approx > 0) ? alpha : -(1 - alpha)` with a `|residual| < delta`
    /// deadzone, der2 = `0`. Used by the Exact leaf-estimation method, whose leaf
    /// delta is the weighted median of the leaf residuals
    /// (`error_functions.h:457-498` `TQuantileError`).
    Mae,
}

/// The per-object first and second derivatives returned by
/// [`Runtime::compute_gradients`], UN-reduced (D-02). Parallel to the input
/// `approx`/`target` slices, in object order.
#[derive(Debug, Clone, PartialEq)]
pub struct Derivatives {
    /// Per-object first derivative (gradient), object order.
    pub der1: Vec<f64>,
    /// Per-object second derivative (hessian), object order.
    pub der2: Vec<f64>,
}

/// The abstract compute runtime the boosting loop drives (D-04). A backend
/// (`cb-backend`'s CubeCL `CpuRuntime` now; GPU runtimes in Phase 7) implements
/// this by launching its `#[cube]` kernels and returning UN-reduced per-object
/// buffers; the host (`cb-compute`/`cb-train`) finalizes every parity-critical
/// SUM via `cb_core::sum_f64`.
pub trait Runtime {
    /// Compute the per-object derivatives for `loss` from the raw approximants
    /// and targets, returning them UN-reduced in object order (D-02).
    ///
    /// `approx` and `target` MUST be the same length (`n` objects). The
    /// elementwise work is order-independent (a per-object kernel on the
    /// backend); no reduction happens here — the histogram / leaf SUM is the
    /// caller's ordered host-side pass.
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if the backend cannot launch the kernel or
    /// the input lengths disagree.
    fn compute_gradients(&self, loss: Loss, approx: &[f64], target: &[f64])
        -> CbResult<Derivatives>;
}
