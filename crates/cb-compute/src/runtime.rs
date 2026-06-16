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
    /// Quantile{alpha, delta} (robust regression, D-6.1-02 Wave 3; D-6.1-03
    /// parametric variant generalizing [`Loss::Mae`]): with `val = target -
    /// approx`, der1 = `|val| < delta ? 0 : (val > 0 ? alpha : -(1 - alpha))`,
    /// der2 = `0` (`error_functions.h:457-498` `TQuantileError`). `alpha` defaults
    /// to `0.5` and `delta` to `1e-6` upstream (`error_functions.h:468-469`), the
    /// median case — so **MAE == Quantile{alpha: 0.5, delta: 1e-6}**. Like
    /// [`Loss::Mae`] it uses the Exact leaf-estimation method, whose leaf delta is
    /// the weighted alpha-quantile of the leaf residuals (`exact_leaf_delta` is
    /// already alpha-general; D-6.1-05). `alpha ∈ [0, 1]`, `delta >= 0` are
    /// validated by [`Loss::validate`].
    Quantile {
        /// Quantile level `alpha ∈ [0, 1]` (`Quantile:alpha=<value>`; default
        /// `0.5`, the median). The Exact leaf takes the weighted alpha-quantile.
        alpha: f64,
        /// Deadzone half-width `delta >= 0` (`Quantile:delta=<value>`; default
        /// `1e-6`). Residuals with `|target - approx| < delta` contribute `0`.
        delta: f64,
    },
    /// LogCosh (smooth regression, D-6.1-02 Wave 1): der1 =
    /// `-tanh(approx - target)`, der2 = `-1/cosh(approx - target)^2`
    /// (`error_functions.h:405-425` `TLogCoshError`). Non-parametric. Upstream
    /// default leaf method is Exact (`catboost_options.cpp:65-70` — NOT Newton;
    /// RESEARCH Pitfall 2), so the fixture pins `leaf_estimation_method: Exact`.
    LogCosh,
    /// Lq{q} (smooth regression, D-6.1-02 Wave 1; D-6.1-03 parametric variant):
    /// der1 = `q*sign(target-approx)*|approx-target|^(q-1)`, der2 =
    /// `-q*(q-1)*|target-approx|^(q-2)` (`error_functions.h:539-568` `TLqError`).
    /// `q` is MANDATORY (no default) and must be `>= 1`; the der2 above is only
    /// Newton-clean for `q >= 2` (the `^(q-2)` term diverges near a zero residual
    /// for `q < 2`; RESEARCH Pitfall 6), so the Wave-1 fixture pins `q = 2.0`.
    Lq {
        /// Loss exponent `q >= 1` (`Lq:q=<value>`).
        q: f64,
    },
    /// Huber{delta} (smooth regression, D-6.1-02 Wave 1; D-6.1-03 parametric):
    /// with `diff = target - approx`, der1 =
    /// `|diff| < delta ? diff : sign(diff)*delta`, der2 =
    /// `|diff| < delta ? -1 : 0` (`error_functions.h:1596-1632` `THuberError`).
    /// `delta` is MANDATORY (no default) and must be `> 0`. Upstream default leaf
    /// method is Newton (`catboost_options.cpp:187-192`).
    Huber {
        /// Huber transition half-width `delta > 0` (`Huber:delta=<value>`).
        delta: f64,
    },
    /// Expectile{alpha} (smooth regression, D-6.1-02 Wave 1; D-6.1-03 parametric):
    /// with `e = target - approx`, der1 = `(e > 0) ? 2*alpha*e : 2*(1-alpha)*e`,
    /// der2 = `(e > 0) ? -2*alpha : -2*(1-alpha)` (`error_functions.h:500-537`
    /// `TExpectileError`). `alpha` defaults to `0.5` upstream and must lie in
    /// `[0, 1]`. Leaf method Newton; the fixture pins
    /// `leaf_estimation_iterations: 1` (override upstream default 5; Pitfall 3).
    Expectile {
        /// Expectile asymmetry `alpha ∈ [0, 1]` (`Expectile:alpha=<value>`).
        alpha: f64,
    },
    /// Poisson (positive-domain / exp-link regression, D-6.1-02 Wave 2): der1 =
    /// `target - exp(approx)`, der2 = `-exp(approx)` over the RAW approx
    /// (`error_functions.h:657-676` `TPoissonError`). Poisson is `IsStoreExpApprox`
    /// upstream (`approx_updater_helpers.h:60-72`) — cb-train stores RAW approx and
    /// computes `exp()` INLINE in the der (the Logloss inline-sigmoid precedent),
    /// with the final prediction transformed via `Exponent` (raw staged approx,
    /// `exp(raw)` predictions; Open Q1 / Pitfall 4). Non-parametric. Upstream
    /// default leaf method is Newton with `leaf_estimation_iterations:10`; the
    /// fixture pins iterations:1 (Pitfall 3).
    Poisson,
    /// Tweedie{variance_power} (positive-domain regression, D-6.1-02 Wave 2;
    /// D-6.1-03 parametric variant): with `p = variance_power`, der1 =
    /// `target*e^((1-p)*approx) - e^((2-p)*approx)`, der2 =
    /// `target*(1-p)*e^((1-p)*approx) - (2-p)*e^((2-p)*approx)` over the RAW approx
    /// (`error_functions.h:1634-1665` `TTweedieError`). `variance_power` is
    /// MANDATORY (no default) and must lie in `(1, 2)` (`error_functions.h:643`).
    /// NOT exp-approx (`isExpApprox==false`, `error_functions.h:1644`) — the `exp`
    /// lives INSIDE the der formula; the prediction is the RAW approx (NO Exponent
    /// transform — A4). Upstream default leaf method is Newton.
    Tweedie {
        /// Tweedie variance power `p ∈ (1, 2)` (`Tweedie:variance_power=<value>`).
        variance_power: f64,
    },
    /// MAPE (positive-domain robust regression, D-6.1-02 Wave 2): with the divisor
    /// `max(1.0, |target|)`, der1 = `sign(target - approx) / max(1.0, |target|)`,
    /// der2 = `0` (`error_functions.h:607-630` `TMAPEError`). Non-parametric. The
    /// `1.f` divisor floor is an f32-domain literal upstream (Pitfall 7);
    /// transcribed as `f64::max(1.0, target.abs())`. der2=0 so Newton is undefined
    /// (Pitfall 5) — upstream default leaf method is Gradient
    /// (`catboost_options.cpp:113-124`), which the fixture pins.
    Mape,
}

/// The default Expectile asymmetry: `alpha = 0.5` (`TExpectileError`'s
/// one-argument constructor, `error_functions.h:512`), the symmetric L2 case.
pub const EXPECTILE_ALPHA_DEFAULT: f64 = 0.5;

impl Loss {
    /// Validate the loss's hyperparameters before training (the
    /// constructor-path range guard, T-06.1.01-01 / T-06.1.01-02). Out-of-domain
    /// `q`/`delta`/`alpha` produce `NaN`/`Inf` derivatives that would poison the
    /// histogram and leaf reductions, so they are rejected up front with a typed
    /// [`cb_core::CbError`] rather than `unwrap`/`panic` (CLAUDE.md: no `unwrap`
    /// in production).
    ///
    /// Mirrors upstream's `Y_ASSERT` domain checks:
    /// - `Lq`: `Q >= 1` (`error_functions.h:548`).
    /// - `Huber`: `delta > 0` (positive transition width).
    /// - `Expectile`: `alpha ∈ [0, 1]` (`error_functions.h:520` — the
    ///   `Alpha > -1e-6 && Alpha < 1.0 + 1e-6` assert, tightened to the exact
    ///   closed interval).
    ///
    /// # Errors
    /// Returns [`cb_core::CbError::OutOfRange`] when a parameter is outside its
    /// admissible domain (or is non-finite).
    pub fn validate(&self) -> CbResult<()> {
        match *self {
            Self::Lq { q } => {
                if !q.is_finite() || q < 1.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Lq exponent q must be finite and >= 1, got {q}"
                    )));
                }
            }
            Self::Huber { delta } => {
                if !delta.is_finite() || delta <= 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Huber delta must be finite and > 0, got {delta}"
                    )));
                }
            }
            Self::Expectile { alpha } => {
                if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Expectile alpha must be finite and in [0, 1], got {alpha}"
                    )));
                }
            }
            // Tweedie variance_power MUST be in the open interval (1, 2)
            // (`error_functions.h:643` `CB_ENSURE(VariancePower > 1 &&
            // VariancePower < 2)`). Outside this range the `e^((1-p)*a)` /
            // `e^((2-p)*a)` der terms degenerate (T-06.1.02-02), so reject up front
            // with a typed CbError (no `unwrap`/`panic`).
            Self::Tweedie { variance_power } => {
                if !variance_power.is_finite() || variance_power <= 1.0 || variance_power >= 2.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Tweedie variance_power must be finite and in (1, 2), got {variance_power}"
                    )));
                }
            }
            // Quantile alpha MUST be in [0, 1] and delta >= 0 (T-06.1.03-01;
            // `error_functions.h:479-480`). An out-of-domain alpha/delta yields an
            // ill-defined quantile der/leaf, so reject up front with a typed
            // CbError (no `unwrap`/`panic`).
            Self::Quantile { alpha, delta } => {
                if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Quantile alpha must be finite and in [0, 1], got {alpha}"
                    )));
                }
                if !delta.is_finite() || delta < 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Quantile delta must be finite and >= 0, got {delta}"
                    )));
                }
            }
            Self::Rmse
            | Self::Logloss
            | Self::CrossEntropy
            | Self::Focal { .. }
            | Self::Mae
            | Self::LogCosh
            | Self::Poisson
            | Self::Mape => {}
        }
        Ok(())
    }
}

/// Which split-score function the greedy tree search uses to rank candidate
/// splits. catboost's CPU default is [`EScoreFunction::Cosine`]
/// (`oblivious_tree_options.cpp:22 EScoreFunction::Cosine`); `L2` is the only
/// other CPU-supported option. cb-train historically hardcoded `L2`, which is a
/// latent parity gap exposed by the initial learn-set shuffle `S` (pc=1 tree-0
/// second split: L2 picks border 3, Cosine picks border 2 = upstream).
///
/// `Default` is `Cosine` to match catboost; configs that need the regression
/// skeleton's L2 set it explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EScoreFunction {
    /// Cosine split score (`score_calcers.h:47-72`): `score = sum(leafVal·sumDer)
    /// / sqrt(sum(count·leafVal²))`. catboost CPU default.
    #[default]
    Cosine,
    /// L2 split score (variance reduction). cb-train's historical hardcoded
    /// choice; only correct for configs that select it explicitly.
    L2,
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
