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
///
/// `Copy` is intentionally NOT derived (Phase 6.2, D-6.2-05 / D-6.1-03): the
/// Wave-3 `MultiQuantile { alpha: Vec<f64> }` variant carries an owned `Vec<f64>`
/// and so cannot be `Copy`. The `Copy` derive is dropped HERE, in the Wave-0
/// mechanical refactor, before any new variant is added, so the cross-crate
/// "by-value `Loss` → borrow/clone" ripple is a ONE-TIME refactor rather than a
/// second pass when MultiQuantile lands. `Clone` is retained and is cheap for the
/// current parameter-light variants; by-value call sites now pass `&Loss` or
/// `.clone()`.
#[derive(Debug, Clone, PartialEq)]
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
    /// MultiClass (softmax multiclass classification, D-6.2-04 / LOSS-02): the ONLY
    /// cross-dimension-COUPLED loss this phase. Over one object's `k`-dimensional
    /// raw approx, `p = softmax(approx)` (max-subtracted, `eval_processing.h:18`);
    /// `der1[d] = δ(d == target_class) - p[d]`; the second derivative is a PACKED
    /// symmetric Hessian (`der2[(y,y)] = p_y*(p_y-1)`, `der2[(y,x)] = p_y*p_x`,
    /// `error_functions.h:687-728`). The leaf delta is a dense symmetric Newton
    /// solve per leaf (`hessian.cpp:22-52`,
    /// [`crate::solve_symmetric_newton`]) — NOT a per-dimension scalar step.
    ///
    /// `approx_dimension` = the distinct class count `max(k, 2)` derived from the
    /// target (`approx_dimension.cpp:24-27`); the training target is the REMAPPED
    /// contiguous class index `[0, k)` (Pitfall 4). No params on the variant (the
    /// class count is target-derived, not stored). Upstream default leaf method is
    /// Newton with 1 iteration (Pitfall 2); fixtures pin `leaf_estimation_iterations:1`.
    MultiClass,
    /// MultiClassOneVsAll (multiclass classification, D-6.2-04 / LOSS-02): a
    /// SEPARABLE (per-dimension diagonal) multiclass loss — each dimension is an
    /// independent binary one-vs-rest sigmoid, so it reuses the scalar
    /// Newton/Logloss leaf math per dimension (no dense solve). Over one object's
    /// dimension `d`: `p = sigmoid(approx_d)`; `der1 = δ(d == target_class) - p`;
    /// `der2 = -p*(1 - p)` (`error_functions.h:746-779`).
    ///
    /// `approx_dimension` = the distinct class count `max(k, 2)`; the training
    /// target is the REMAPPED contiguous class index `[0, k)`. Upstream default
    /// leaf method is Newton with 1 iteration (Pitfall 2). Predictions are per-dim
    /// sigmoid probabilities (which do NOT sum to 1, unlike softmax) + argmax class.
    MultiClassOneVsAll,
    /// MultiLogloss (multilabel binary classification, D-6.2-04 / LOSS-02): a
    /// SEPARABLE (per-dimension diagonal) multilabel loss. Each label dimension `d`
    /// is an independent binary sigmoid cross-entropy over a `{0,1}` label column,
    /// so it reuses the scalar Logloss / Newton leaf math per dimension (no dense
    /// solve, no softmax coupling). Over one object's label dimension `d` with
    /// `p = sigmoid(approx_d)`: `der1 = target_d - p`, `der2 = -p*(1 - p)`
    /// (`error_functions.h:781-820` `TMultiCrossEntropyError`,
    /// [`crate::multi_crossentropy_ders`]).
    ///
    /// `approx_dimension` = the label-set WIDTH (number of binary label columns,
    /// `approx_dimension.cpp:22-23` `IsMultiTargetObjective`); the training target
    /// is dim-major length `dim*n` (one `{0,1}` label per dimension per object).
    /// Identical der path to [`Loss::MultiCrossEntropy`] — they map to the SAME
    /// upstream `TMultiCrossEntropyError` class (`tensor_search_helpers.cpp:236-238`);
    /// only the admissible target range differs (`{0,1}` here). Upstream default
    /// leaf method is Newton with `leaf_estimation_iterations:10`; fixtures pin it
    /// to 1 (Pitfall 2). Predictions are per-dim sigmoid probabilities.
    MultiLogloss,
    /// MultiCrossEntropy (multilabel classification with soft probability targets,
    /// D-6.2-04 / LOSS-02): the SAME `TMultiCrossEntropyError` per-dimension
    /// diagonal der path as [`Loss::MultiLogloss`] — they dispatch to ONE shared
    /// [`crate::multi_crossentropy_ders`] helper (`tensor_search_helpers.cpp:236-238`).
    /// The ONLY difference is the admissible target range: MultiCrossEntropy admits
    /// soft probability targets in `[0,1]` (vs MultiLogloss's binary `{0,1}`).
    ///
    /// `approx_dimension` = the label-set width; the training target is dim-major
    /// length `dim*n` (one `[0,1]` probability per dimension per object). Upstream
    /// default leaf method is Newton with `leaf_estimation_iterations:10`; fixtures
    /// pin it to 1 (Pitfall 2). Predictions are per-dim sigmoid probabilities.
    MultiCrossEntropy,
    /// MultiQuantile (multi-output robust regression, D-6.2-05 / LOSS-03): `K`
    /// INDEPENDENT [`Loss::Quantile`] dimensions — each dimension `d` is a
    /// standalone quantile at its own level `alpha[d]`, sharing the deadzone
    /// `delta`. Fully SEPARABLE (no cross-dimension coupling): each dimension
    /// reuses the scalar [`crate::quantile_der1`] der VERBATIM
    /// (`der1[d*n+i] = (|target_i - approx[d*n+i]| < delta) ? 0 : (arg > 0 ?
    /// alpha[d] : -(1 - alpha[d]))`, `der2 = 0`; `error_functions.cpp:453-478`
    /// `CalcDersMulti`) AND the 6.1 Exact weighted-`alpha`-quantile leaf path
    /// ([`crate::exact_leaf_delta`], already `alpha`-general — D-6.1-05) with the
    /// dimension's own `alpha[d]`. NO new der/leaf math.
    ///
    /// `approx_dimension` = `alpha.len()` (`approx_dimension.cpp:17-19`
    /// `GetAlphaMultiQuantile(params).size()`). The training target stays
    /// PER-OBJECT length `n` (every dimension predicts a quantile of the SAME
    /// scalar target, unlike the dim-major target of the multilabel losses). The
    /// upstream single-host-CPU default leaf method is **Exact**
    /// (`catboost_options.cpp:289-301` `useExact` override — Pitfall 3), `der2 = 0`
    /// per dimension. Predictions are RAW (identity — the per-quantile approx; no
    /// link transform).
    ///
    /// The per-quantile `alpha` is an owned `Vec<f64>` (the `Loss::Variant {
    /// params }` pattern, D-6.1-03), which is why `Loss` dropped `Copy` in the
    /// Wave-0 refactor. `alpha[k] ∈ [0, 1]` (each) and `delta >= 0` are validated by
    /// [`Loss::validate`].
    MultiQuantile {
        /// Per-dimension quantile levels `alpha[d] ∈ [0, 1]`
        /// (`MultiQuantile:alpha=<a0>,<a1>,...`). `approx_dimension = alpha.len()`.
        alpha: Vec<f64>,
        /// Shared deadzone half-width `delta >= 0` across all dimensions
        /// (`MultiQuantile:delta=<value>`; default `1e-6`). Residuals with
        /// `|target - approx| < delta` contribute `0` in every dimension.
        delta: f64,
    },
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
        // Matched by reference (`match self`, NOT `match *self`): the
        // `MultiQuantile { alpha: Vec<f64>, .. }` variant carries an owned,
        // non-`Copy` `Vec<f64>`, so the place cannot be matched by value. The
        // scalar-parameter arms bind their `f64` fields by reference (default
        // binding modes) and dereference at use.
        match self {
            Self::Lq { q } => {
                if !q.is_finite() || *q < 1.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Lq exponent q must be finite and >= 1, got {q}"
                    )));
                }
            }
            Self::Huber { delta } => {
                if !delta.is_finite() || *delta <= 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Huber delta must be finite and > 0, got {delta}"
                    )));
                }
            }
            Self::Expectile { alpha } => {
                if !alpha.is_finite() || !(0.0..=1.0).contains(alpha) {
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
                if !variance_power.is_finite() || *variance_power <= 1.0 || *variance_power >= 2.0 {
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
                if !alpha.is_finite() || !(0.0..=1.0).contains(alpha) {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Quantile alpha must be finite and in [0, 1], got {alpha}"
                    )));
                }
                if !delta.is_finite() || *delta < 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "Quantile delta must be finite and >= 0, got {delta}"
                    )));
                }
            }
            // MultiQuantile (Wave 3, D-6.2-05 / LOSS-03): K INDEPENDENT Quantile
            // dimensions, one `alpha` per dimension. The per-quantile `alpha`
            // values are an owned `Vec<f64>` (the `Loss::Variant { params }`
            // pattern, D-6.1-03 — this is why `Copy` was dropped on `Loss` in the
            // Wave-0 refactor). Validate each `alpha[k]` finite ∈ `[0, 1]` and the
            // shared `delta` finite `>= 0` (clone of the Quantile arm; T-6.2-03,
            // typed `CbError::OutOfRange`, no panic). An empty `alpha` is rejected
            // (`approx_dimension = alpha.len()` must be `>= 1`).
            Self::MultiQuantile { alpha, delta } => {
                if alpha.is_empty() {
                    return Err(cb_core::CbError::OutOfRange(
                        "MultiQuantile alpha must contain at least one quantile level".to_owned(),
                    ));
                }
                for (k, &a) in alpha.iter().enumerate() {
                    if !a.is_finite() || !(0.0..=1.0).contains(&a) {
                        return Err(cb_core::CbError::OutOfRange(format!(
                            "MultiQuantile alpha[{k}] must be finite and in [0, 1], got {a}"
                        )));
                    }
                }
                if !delta.is_finite() || *delta < 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "MultiQuantile delta must be finite and >= 0, got {delta}"
                    )));
                }
            }
            // MultiClass / MultiClassOneVsAll carry NO hyperparameters on the
            // variant (the class count is derived from the target, D-6.2-04). The
            // class-index range check (T-6.2-01) is enforced at training time
            // against the REMAPPED `[0, k)` index — `Loss::validate` has no target
            // in scope, so there is nothing to reject here.
            //
            // MultiLogloss / MultiCrossEntropy likewise carry no hyperparameters
            // (the label-set width is target-derived, D-6.2-04). Their per-dimension
            // target-range guard (MultiLogloss ∈ `{0,1}`, MultiCrossEntropy ∈
            // `[0,1]`, T-6.2-04a) needs the target, which `Loss::validate` does not
            // see, so it is enforced at training time (the multiclass remap
            // precedent) — nothing to reject here.
            Self::Rmse
            | Self::Logloss
            | Self::CrossEntropy
            | Self::Focal { .. }
            | Self::Mae
            | Self::LogCosh
            | Self::Poisson
            | Self::Mape
            | Self::MultiClass
            | Self::MultiClassOneVsAll
            | Self::MultiLogloss
            | Self::MultiCrossEntropy => {}
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
/// [`Runtime::compute_gradients`], UN-reduced (D-02).
///
/// # Dimension-major flat layout (Phase 6.2, D-6.2-01)
///
/// `approx`, `der1`, and `der2` are dimension-major flat buffers of length
/// `approx_dimension * n_objects`, indexed `buf[d * n_objects + i]` for dimension
/// `d` and object `i` (the OUTER index is the dimension). For the diagonal /
/// separable losses handled in this wave, `der1` and `der2` share the input
/// `approx`'s `approx_dimension * n_objects` length and per-object/per-dimension
/// ordering.
///
/// At `approx_dimension == 1` this collapses to the historical per-object scalar
/// layout `buf[i]`, and the per-dimension reduction MUST run as an outer loop with
/// a single iteration over `approx[0..n]` so the values are BYTE-IDENTICAL to the
/// pre-6.2 scalar path (RESEARCH Pitfall 1 — never fuse the per-dimension
/// reduction into a single `0..approx_dimension * n` pass, which would perturb the
/// `cb_core::sum_f64` order downstream).
#[derive(Debug, Clone, PartialEq)]
pub struct Derivatives {
    /// Per-object first derivative (gradient), dimension-major
    /// (`der1[d * n_objects + i]`); see the struct-level layout note.
    pub der1: Vec<f64>,
    /// Per-object second derivative (hessian), dimension-major
    /// (`der2[d * n_objects + i]`); see the struct-level layout note.
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
    /// `approx` is the dimension-major flat buffer of length
    /// `approx_dimension * n_objects` (`approx[d * n_objects + i]`, D-6.2-01).
    /// `target` stays per-object (length `n_objects`) for the scalar / class
    /// losses. The returned [`Derivatives`] share `approx`'s dimension-major length
    /// for the diagonal / separable losses. The elementwise work is
    /// order-independent (a per-object kernel on the backend); no reduction happens
    /// here — the histogram / leaf SUM is the caller's ordered host-side pass.
    ///
    /// At `approx_dimension == 1` the output is byte-identical to the pre-6.2
    /// scalar path: the backend runs the per-dimension kernel launch as an outer
    /// loop with a single iteration over `approx[0..n_objects]` (RESEARCH
    /// Pitfall 1 — no fused `0..approx_dimension * n` pass).
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if the backend cannot launch the kernel, the
    /// `approx` length is not a multiple of `approx_dimension`, or the input
    /// lengths disagree.
    fn compute_gradients(
        &self,
        loss: &Loss,
        approx: &[f64],
        target: &[f64],
        approx_dimension: usize,
    ) -> CbResult<Derivatives>;

    /// Compute the GROUPED per-object derivatives for a ranking `loss` over the
    /// query-group structure `groups` (LOSS-04, D-6.3-03), mirroring upstream
    /// `IDerCalcer::CalcDersForQueries` (`error_functions.h:831-841`).
    ///
    /// This is the sibling grouped seam to [`Runtime::compute_gradients`]: the
    /// pointwise signature above stays BYTE-IDENTICAL (D-04 no-regression on the
    /// shipped scalar / N-dim oracles), and ranking losses route here instead.
    /// The reduction is host-side (NO CubeCL kernel — RESEARCH Architectural
    /// Responsibility Map; AGENTS.md: 6.3 is host reductions), so the trait
    /// supplies a default implementation delegating to
    /// [`crate::ranking_der::calc_ders_for_queries`]; backends do not override it.
    ///
    /// Returns one [`Derivatives`] per group, in group order. Plan 06.3-01 lands
    /// the seam; every concrete ranking-loss arm is filled by Plans 02–05.
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if a group span is out of range, the input
    /// lengths disagree, or (in this plan) the ranking loss is not yet wired.
    fn compute_gradients_grouped(
        &self,
        loss: &Loss,
        approx: &[f64],
        target: &[f64],
        weights: &[f64],
        groups: &[crate::ranking_der::GroupSpan],
        random_seed: u64,
    ) -> CbResult<Vec<Derivatives>> {
        crate::ranking_der::calc_ders_for_queries(loss, approx, target, weights, groups, random_seed)
    }
}
