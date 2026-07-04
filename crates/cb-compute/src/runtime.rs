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

use crate::custom::CustomObjectiveHandle;

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
    /// RMSEWithUncertainty (regression with predicted uncertainty, D-6.4-04 /
    /// LOSS-08): a 2-dimensional DIAGONAL-hessian loss riding the 6.2 N-dim spine.
    /// Dim 0 is the regression MEAN (`approx[0]`), dim 1 is the LOG-SCALE
    /// (`approx[1]`). Over one object with a scalar `target` and weight `w`
    /// (`error_functions.h:280-313` `TRMSEWithUncertaintyError`,
    /// [`crate::rmse_with_uncertainty_ders`]):
    /// ```text
    /// diff    = target - approx[0]
    /// prec    = exp(-2 * approx[1])
    /// der1    = [w*diff, w*(diff²·prec - 1)]
    /// der2-diag = [-w, -2·w·diff²·prec]
    /// ```
    /// It is the ngboost "natural gradient" (regular gradient × Fisher info). The
    /// hessian is `EHessianType::Diagonal`, so each dimension is an INDEPENDENT
    /// scalar Newton step (the [`Self::MultiClassOneVsAll`]/[`Self::MultiLogloss`]
    /// diagonal leaf path), NOT the [`Self::MultiClass`] dense symmetric solve
    /// (06.4-RESEARCH Pitfall 4).
    ///
    /// `approx_dimension = 2` (`approx_dimension.cpp:16`). The training target stays
    /// PER-OBJECT length `n` (a single scalar target; the mean and log-scale are the
    /// two approx dims). RMSEWithUncertainty ALWAYS starts from the optimal constant
    /// approx `[mean(target), 0.5·log(var(target))]` even with
    /// `boost_from_average=false` (`train_model.cpp:858`,
    /// `optimal_const_for_loss.h:225-229`). No params on the variant. Upstream
    /// default leaf method is Newton with `leaf_estimation_iterations:1`
    /// (`catboost_options.cpp:77-82`); the fixture pins both. Predictions are RAW
    /// 2-dim (RawFormulaVal — identity; the variance transform `exp(2·approx[1])` is
    /// a separate prediction-type concern, not the loss der).
    RmseWithUncertainty,
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
    /// QueryRMSE (querywise ranking, D-6.3-03 / LOSS-04 Wave A): a per-query-group
    /// RMSE whose der subtracts the group's weighted residual mean so the model
    /// learns the WITHIN-group ranking, not the absolute target. RAW approx
    /// (`isExpApprox == false`, `error_functions.h:876`). Over one group:
    /// `queryAvrg = Σ_g (target - approx)·w / Σ_g w`; per object
    /// `der1 = (target - approx - queryAvrg)·w`, `der2 = -1·w`
    /// (`error_functions.h:879-933` `TQueryRmseError`,
    /// [`crate::queryrmse_der`] / [`crate::calc_ders_for_queries`]). Empty /
    /// zero-weight group → `queryAvrg = 0`, ders `0` (the upstream `queryCount > 0`
    /// guard — no divide). No params on the variant. Rides the EXISTING pointwise
    /// leaf estimators (Gradient/Newton — the der is per-object); no pairwise
    /// Cholesky path. Predictions are RAW (identity — no link transform).
    QueryRmse,
    /// QuerySoftMax (querywise ranking, D-6.3-03 / LOSS-04 Wave A): a per-query-
    /// group softmax cross-entropy over `Beta·approx`, MAX-SHIFTED before `exp`
    /// (`error_functions.cpp:540-552`; the [`crate::calc_softmax`] NaN guard,
    /// Security V5 / T-06.3-02-01). RAW approx (`isExpApprox == false`,
    /// `error_functions.h:1040`). Over one group with
    /// `p = expApprox·w / Σ_g expApprox·w` and
    /// `sumWTargets = Σ_g w·target` (over `target > 0`, `weight > 0`):
    /// `der1 = Beta·(-sumWTargets·p + w·target)`,
    /// `der2 = Beta·sumWTargets·(Beta·p·(p-1) - LambdaReg)`
    /// (`error_functions.cpp:560-565` `TQuerySoftMaxError`,
    /// [`crate::querysoftmax_der`]). `sumWTargets <= 0` (or `weight <= 0`) → ders
    /// `0` (T-06.3-02-02 — no divide). Rides the EXISTING pointwise leaf
    /// estimators (Gradient/Newton); no pairwise path. Predictions are RAW.
    ///
    /// `lambda` (`LambdaReg`) defaults to `0.01`, `beta` to `1.0`
    /// (`loss_description.cpp:209-216`). Both are owned `f64` params (the
    /// `Loss::Variant { params }` pattern). `lambda` finite `>= 0` and `beta`
    /// finite `> 0` are validated by [`Loss::validate`] (T-06.3-02-03).
    QuerySoftMax {
        /// L2 regularization on the softmax der `LambdaReg >= 0`
        /// (`QuerySoftMax:lambda=<value>`; default `0.01`).
        lambda: f64,
        /// Inverse-temperature `Beta > 0` scaling the approx before `exp`
        /// (`QuerySoftMax:beta=<value>`; default `1.0`).
        beta: f64,
    },
    /// PairLogit (pairwise ranking, D-6.3-04 / LOSS-04 Wave B): the pairwise
    /// logistic loss over explicit `Pool.pairs`. EXP-approx (`isExpApprox == true`,
    /// `error_functions.h:828` `CB_ENSURE(isExpApprox == true)`) — cb-train stores
    /// the RAW approx and computes `exp()` INLINE in the der (the Poisson
    /// precedent, [`Loss::Poisson`]). Over one group, per winner `docId` and each
    /// of its `Competitors` (the explicit losers it should outrank):
    /// `p = expApprox[loser] / (expApprox[loser] + expApprox[winner])`;
    /// `winnerDer += w·p`, `der1[loser] -= w·p`; `winnerDer2 += w·p·(p-1)`,
    /// `der2[loser] += w·p·(p-1)`; then `der1[winner] += winnerDer`,
    /// `der2[winner] += winnerDer2` (`error_functions.h:849-866`
    /// `TPairLogitError::CalcDersForQueries`, [`crate::pairlogit_competitors_der`]).
    /// The pair weight `w` is `competitor.weight` (NOT the per-object weight). No
    /// params on the variant. Rides the EXISTING pointwise leaf estimators
    /// (POINTWISE, NOT pairwise scoring — `IsPairwiseScoring` is false for the
    /// non-`Pairwise` variant; upstream default leaf method Newton). Predictions
    /// are RAW (identity — the ranking score; no link transform on apply).
    PairLogit,
    /// PairLogitPairwise (pairwise ranking, D-6.3-04 / LOSS-04 Wave B): the SAME
    /// pairwise-logit der as [`Loss::PairLogit`] (it maps to the same upstream
    /// `TPairLogitError`, `tensor_search_helpers.cpp:259-262`), but `IsPairwise`
    /// scoring (`enum_helpers.cpp:469-475`) — so the leaf VALUES are solved via the
    /// dedicated Cholesky pairwise-leaf path (`pairwise_leaves_calculation.cpp:9`,
    /// [`cb_train::pairwise_leaves`]) over the per-leaf pairwise weight sums + der
    /// sums, NOT the pointwise Gradient/Newton estimators. `*Pairwise` losses force
    /// `boosting_type = Plain` (`IsPlainOnlyModeLoss`, `enum_helpers.cpp:460-467`).
    /// EXP-approx (same as PairLogit). No params on the variant. Predictions are
    /// RAW.
    PairLogitPairwise,
    /// LambdaMart (listwise ranking, LOSS-04 Wave B): a per-group lambda gradient
    /// toward a target ranking `metric` (NDCG default). RAW approx
    /// (`isExpApprox == false`, ctor `IDerCalcer(false, 1, …)`,
    /// `error_functions.cpp:594`). Per group: stable-sort docs by approx
    /// descending, then for each ordered pair with `firstTarget > secondTarget`:
    /// `delta = (dcgNum·dcgDen) / idealScore` (the metric-specific position weight,
    /// `error_functions.cpp:653-658`), optionally `delta /= 0.01 + |approxDiff|`
    /// when `norm`; `antigrad = -Sigma·delta / (1 + exp(Sigma·approxDiff))`,
    /// `hessian = Sigma²·delta · σ(Sigma·approxDiff)·(1 - σ(...))`; accumulate
    /// `±antigrad` into `der1` and `+hessian` into `der2` for the high/low doc
    /// (`error_functions.cpp:664-674` `CalcDersNDCG`). The der2 hessian is filled
    /// despite `maxDerivativeOrder == 1` (RESEARCH Pitfall 5), so the upstream
    /// default leaf method is Newton. Optional `norm` post-scales all ders by
    /// `log2(1 + Σder1)/Σder1` (`error_functions.cpp:916-922`). Rides the EXISTING
    /// pointwise leaf estimators (POINTWISE — `IsPairwiseScoring` false). Predictions
    /// are RAW. Defaults: `metric = NDCG`, `sigma = 1.0`, `top = -1` (full group),
    /// `norm = true` (`tensor_search_helpers.cpp:273-278`).
    LambdaMart {
        /// The target ranking metric the lambda gradient optimizes
        /// (`LambdaMart:metric=<NDCG|DCG|MRR|ERR|MAP>`; default `NDCG`).
        metric: LambdaMartMetric,
        /// Scale parameter `Sigma > 0` in the pairwise sigmoid
        /// (`LambdaMart:sigma=<value>`; default `1.0`).
        sigma: f64,
        /// Top-`k` cutoff for the metric (`LambdaMart:top=<value>`; `-1` = the full
        /// group, the default). Stored as `i64` so `-1` is representable.
        top: i64,
        /// `norm` flag: when `true` (the default) the per-pair `delta` is divided by
        /// `0.01 + |approxDiff|` and the whole group's ders are rescaled by
        /// `log2(1 + Σder1)/Σder1` (`error_functions.cpp:660-662,916-922`).
        norm: bool,
    },
    /// YetiRank (randomized listwise ranking, LOSS-04 Wave C / D-6.3-02). The
    /// RNG-STREAM loss: each group's pairwise weights are SAMPLED via a 2-level
    /// `TFastRng64` seed derivation + Gumbel noise over the exp-approxes
    /// (`yetirank_helpers.cpp:146-163,305-393`), NOT a closed-form der. For a
    /// single-thread fit (`blockCount == 1`,
    /// `restorable_rng.cpp:3-9 GenRandUI64Vector(1, seed)` → one block seed):
    /// 1. block `rand = TFastRng64(GenRandUI64Vector(1, seed)[0])`;
    /// 2. per query: `querySeed = rand.GenRand()` re-seeds the inner per-query
    ///    `TFastRng64(querySeed)`;
    /// 3. per permutation (`permutations`, default 10): AddNoise draws one
    ///    `gen_rand_real1()` Gumbel uniform PER doc (`u`, then
    ///    `expApprox[d] *= u / (1.000001 - u)`), stable-sorts the indices by the
    ///    bootstrapped approx DESCENDING, and accumulates the Classic pairwise
    ///    weights (`magicConst 0.15 · decay^k · |Δrelev|` along the sorted
    ///    adjacency, `decay` default 0.85);
    /// 4. `competitorsWeight[w][l] = queryWeight · weights[w][l] / permutations`;
    ///    nonzero entries become the SAMPLED competitor pairs.
    /// Those sampled pairs then feed the EXISTING `TPairLogitError` der over the
    /// group (POINTWISE leaf path — `IsPairwiseScoring` false). The der is
    /// recomputed every boosting iteration (the pairs are re-sampled,
    /// `yetirank_helpers.cpp:347-393`). RAW approx is exp()'d INLINE for the noise
    /// (the bootstrappedApprox is the exp-approx). Predictions are RAW. The RNG
    /// draw order is the parity crux (RESEARCH Pitfall 1) and is validated against
    /// the instrumented `CB_INSTRUMENT_LOG` ground truth BEFORE the der is checked.
    /// Defaults: `permutations = 10`, `decay = 0.85`
    /// (`loss_description.cpp:181-193`).
    YetiRank {
        /// Number of noise permutations sampled per group (`permutations`,
        /// `loss_description.cpp:185`; default 10). Each permutation draws
        /// `querySize` Gumbel uniforms. Validated `>= 1` by [`Loss::validate`].
        permutations: u32,
        /// Classic-weight geometric decay `decay ∈ [0, 1]`
        /// (`yetirank_helpers.cpp:203` `decayCoefficient *= Config.Decay`;
        /// `loss_description.cpp:192`, default 0.85). Validated by
        /// [`Loss::validate`].
        decay: f64,
    },
    /// YetiRankPairwise (randomized listwise ranking, PAIRWISE leaf path, LOSS-04
    /// Wave C). The SAME sampled-pair RNG stream as [`Loss::YetiRank`] (identical
    /// 2-level seed + Gumbel noise + Classic weights), but the leaf values are
    /// solved via the Cholesky PAIRWISE-leaf path
    /// ([`cb_train::pairwise_leaves`], the Plan 06.3-03 machinery
    /// `PairLogitPairwise` rides) instead of the pointwise estimators
    /// (`IsPairwiseScoring` true). Forces `boosting_type = Plain`
    /// (`IsPlainOnlyModeLoss`, `enum_helpers.cpp:460-467`). Predictions are RAW.
    /// Defaults match [`Loss::YetiRank`] (`permutations = 10`, `decay = 0.85`).
    YetiRankPairwise {
        /// Number of noise permutations per group (default 10); see
        /// [`Loss::YetiRank::permutations`].
        permutations: u32,
        /// Classic-weight geometric decay (default 0.85); see
        /// [`Loss::YetiRank::decay`].
        decay: f64,
    },
    /// StochasticRank (randomized querywise ranking, LOSS-04 Wave C /
    /// D-6.3-02). The OTHER RNG-stream loss: a Monte-Carlo gradient estimator that
    /// perturbs each group's tie-broken/centered approxes with Gaussian noise and
    /// averages the per-position metric-difference gradient over `num_estimations`
    /// samples (`error_functions.cpp:1008-1102`). Per group (single-thread,
    /// `randomSeed = randomSeed + queryIndex`,
    /// `error_functions.h:1257 GenRandUI64Vector`):
    /// 1. shift `shifted[d] = approx[d] - Sigma·Mu·target[d]`, then center by
    ///    subtracting the group mean (non-FilteredDCG);
    /// 2. `rng = TFastRng64(randomSeed)`; per sample (`num_estimations`): draw one
    ///    `std_normal(rng)` Gaussian PER doc (`noise[d]`), `scores[d] = shifted[d]
    ///    + Sigma·noise[d]`, stable-sort the order DESCENDING by score, compute the
    ///    metric position weights + cumulative statistics, and accumulate the
    ///    per-doc `Σ metricDiff · densityDiff / num_estimations` into `der1`;
    /// 3. SFA: subtract the mean der (orthogonalize), then (count > 2) project out
    ///    the approx direction (`Lambda`/`Nu`). `der2 = 0` (Gradient leaf method).
    /// RAW approx; querywise POINTWISE (no pairs). The Gaussian draws go through
    /// [`cb_core::std_normal`] (the SAME variable-length Marsaglia-polar draw
    /// sequence — a different sampler desyncs the stream, RESEARCH Pitfall 1).
    /// Predictions are RAW. Defaults: `sigma = 1.0`, `mu = 0.0`,
    /// `num_estimations = 1`, `nu = 0.01`, `lambda = 1.0`
    /// (`tensor_search_helpers.cpp:284-289`).
    StochasticRank {
        /// The target ranking metric the Monte-Carlo gradient optimizes. Only the
        /// DCG/NDCG arm is transcribed in this phase (the most common; the
        /// PFound/ERR/MRR arms are out of scope, gated by [`Loss::validate`]).
        metric: StochasticRankMetric,
        /// Noise scale `Sigma > 0` (`sigma`, default 1.0). Validated by
        /// [`Loss::validate`].
        sigma: f64,
        /// Tie-resolving coefficient `Mu >= 0` (`mu`, default 0.0). Validated by
        /// [`Loss::validate`].
        mu: f64,
        /// Number of Monte-Carlo samples per group (`num_estimations`, default 1).
        /// Each sample draws `querySize` Gaussian noises. Validated `>= 1` by
        /// [`Loss::validate`].
        num_estimations: u32,
    },
    /// Custom user objective (LOSS-07, D-6.4-05): a user-supplied
    /// [`crate::CustomObjective`] trait object (per-object `(der1, der2)`,
    /// mirroring the Python `calc_ders_range`) plugged into the ONE
    /// `compute_gradients` dispatch via the [`CustomObjectiveHandle`] `Arc<dyn>`
    /// newtype (the workspace's first trait-object loss variant). `Loss`'s
    /// `#[derive(Debug, Clone, PartialEq)]` still compiles because the handle
    /// implements all three manually (`Arc::ptr_eq` identity equality;
    /// 06.4-RESEARCH Strand 3 / Pitfall 7). The custom der buffer is consumed by
    /// the loss-agnostic leaf/tree pipeline unchanged; the leaf method is the
    /// CALLER-SELECTED one (the Builder default is `Gradient`, which uses only der1
    /// — the user-supplied der2 is consumed only under the Newton leaf method,
    /// WR-02). The Phase-8 PyO3 callback (D-09) wraps
    /// the SAME trait — no `pyo3` dependency is added here.
    Custom(CustomObjectiveHandle),
}

/// The target ranking metric a [`Loss::StochasticRank`] Monte-Carlo gradient
/// optimizes. Mirrors the `EqualToOneOf(TargetMetric, DCG, NDCG, PFound,
/// FilteredDCG, ERR, MRR)` set the upstream `TStochasticRankError` ctor admits
/// (`error_functions.cpp:962-966`). This phase transcribes ONLY the DCG/NDCG arm
/// (`CalcDCGMetricDiff` / `CalcDCGCumulativeStatistics` /
/// `ComputeDCGPosWeights`); the other metrics are admitted by the enum for
/// future waves but rejected by [`Loss::validate`] until transcribed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StochasticRankMetric {
    /// Discounted Cumulative Gain — `CalcDCGMetricDiff` arm
    /// (`error_functions.cpp:1222-1256`).
    Dcg,
    /// Normalized DCG (the default StochasticRank metric); same
    /// `CalcDCGMetricDiff` arm as [`StochasticRankMetric::Dcg`], differing only in
    /// the `ComputeDCGPosWeights` ideal-DCG normalization
    /// (`error_functions.cpp:1525-1536`).
    #[default]
    Ndcg,
}

/// The target ranking metric a [`Loss::LambdaMart`] optimizes its per-group lambda
/// gradient toward. Mirrors the `EqualToOneOf(TargetMetric, DCG, NDCG, MRR, ERR,
/// MAP)` set the upstream `TLambdaMartError` ctor admits
/// (`error_functions.cpp:602-603`). `DCG`/`NDCG` share the same `CalcDersNDCG`
/// arm (the ideal-metric normalization differs only in `CalcIdealMetric`, which
/// the der treats identically — both route through `CalcDersNDCG`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LambdaMartMetric {
    /// Discounted Cumulative Gain — the `CalcDersNDCG` arm (`error_functions.cpp:882`).
    Dcg,
    /// Normalized DCG (the default upstream LambdaMART metric); same
    /// `CalcDersNDCG` arm as [`LambdaMartMetric::Dcg`].
    #[default]
    Ndcg,
    /// Mean Reciprocal Rank — the `CalcDersMRR` arm (`error_functions.cpp:679`).
    Mrr,
    /// Expected Reciprocal Rank — the `CalcDersERR` arm (`error_functions.cpp:748`).
    Err,
    /// Mean Average Precision — the `CalcDersMAP` arm (`error_functions.cpp:805`).
    Map,
}

/// The default QuerySoftMax L2 regularization `lambda = 0.01`
/// (`NCatboostOptions::GetQuerySoftMaxLambdaReg`, `loss_description.cpp:211`).
pub const QUERYSOFTMAX_LAMBDA_DEFAULT: f64 = 0.01;

/// The default QuerySoftMax inverse-temperature `beta = 1.0`
/// (`NCatboostOptions::GetQuerySoftMaxBeta`, `loss_description.cpp:215`).
pub const QUERYSOFTMAX_BETA_DEFAULT: f64 = 1.0;

/// The default Expectile asymmetry: `alpha = 0.5` (`TExpectileError`'s
/// one-argument constructor, `error_functions.h:512`), the symmetric L2 case.
pub const EXPECTILE_ALPHA_DEFAULT: f64 = 0.5;

/// The default YetiRank noise permutation count (`permutations = 10`,
/// `loss_description.cpp:185`).
pub const YETIRANK_PERMUTATIONS_DEFAULT: u32 = 10;

/// The default YetiRank Classic-weight geometric decay (`decay = 0.85`,
/// `loss_description.cpp:192`).
pub const YETIRANK_DECAY_DEFAULT: f64 = 0.85;

/// The YetiRank Classic-weight magic constant `0.15` ("Like in GPU",
/// `yetirank_helpers.cpp:198`).
pub const YETIRANK_MAGIC_CONST: f64 = 0.15;

/// The default StochasticRank Gaussian noise scale (`sigma = 1.0`,
/// `tensor_search_helpers.cpp:284`).
pub const STOCHASTIC_RANK_SIGMA_DEFAULT: f64 = 1.0;

/// The default StochasticRank tie-resolving coefficient (`mu = 0.0`,
/// `tensor_search_helpers.cpp:286`).
pub const STOCHASTIC_RANK_MU_DEFAULT: f64 = 0.0;

/// The default StochasticRank Monte-Carlo sample count (`num_estimations = 1`,
/// `tensor_search_helpers.cpp:285`).
pub const STOCHASTIC_RANK_NUM_ESTIMATIONS_DEFAULT: u32 = 1;

/// The default StochasticRank approx-norm addition (`nu = 0.01`,
/// `tensor_search_helpers.cpp:287`). Used in the Stage-3 SFA projection.
pub const STOCHASTIC_RANK_NU_DEFAULT: f64 = 0.01;

/// The default StochasticRank SFA coefficient for DCG/NDCG (`lambda = 1.0`,
/// `tensor_search_helpers.cpp:288-289`; FilteredDCG would use 0.0).
pub const STOCHASTIC_RANK_LAMBDA_DEFAULT: f64 = 1.0;

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
            // QuerySoftMax (Wave A, D-6.3-03 / LOSS-04): `lambda` (LambdaReg) must be
            // finite and `>= 0` (an L2 regularizer), `beta` (the inverse-temperature
            // scaling the approx before `exp`) finite and `> 0`. An out-of-domain
            // lambda/beta yields a NaN/Inf softmax der that would poison the leaf
            // reductions (T-06.3-02-03), so reject up front with a typed CbError (no
            // `unwrap`/`panic`). QueryRmse carries no params (nothing to reject).
            Self::QuerySoftMax { lambda, beta } => {
                if !lambda.is_finite() || *lambda < 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "QuerySoftMax lambda must be finite and >= 0, got {lambda}"
                    )));
                }
                if !beta.is_finite() || *beta <= 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "QuerySoftMax beta must be finite and > 0, got {beta}"
                    )));
                }
            }
            // LambdaMart (Wave B, LOSS-04): `sigma` (the pairwise-sigmoid scale)
            // must be finite and `> 0` (`error_functions.cpp:604`
            // `CB_ENSURE(Sigma > 0)`). A non-positive sigma collapses the sigmoid
            // and yields a degenerate gradient (T-06.3-03-04), so reject up front
            // with a typed CbError (no `unwrap`/`panic`). `top` is unbounded
            // (`-1` = full group; any positive `k` is clamped to the group size by
            // `GetQueryTopSize`), but `top == 0` would make every group's metric
            // window empty — reject it as out of range. `metric` is an exhaustive
            // enum (nothing to reject); `norm` is a bool.
            Self::LambdaMart { sigma, top, .. } => {
                if !sigma.is_finite() || *sigma <= 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "LambdaMart sigma must be finite and > 0, got {sigma}"
                    )));
                }
                if *top == 0 {
                    return Err(cb_core::CbError::OutOfRange(
                        "LambdaMart top must be -1 (full group) or a positive cutoff, got 0"
                            .to_owned(),
                    ));
                }
            }
            // YetiRank / YetiRankPairwise (Wave C, LOSS-04): `permutations >= 1`
            // (`loss_description.cpp:185`; a zero count would sample no pairs and
            // divide by zero in `competitorsWeight / permutationCount`,
            // `yetirank_helpers.cpp:339`, T-06.3-04-03) and `decay ∈ [0, 1]`
            // (`yetirank_helpers.cpp:203` — a decay outside the unit interval
            // either explodes or sign-flips the geometric Classic weights). Both
            // variants share the same RNG-stream params. `u32` is non-negative;
            // the `>= 1` check rejects only `0`.
            Self::YetiRank {
                permutations,
                decay,
            }
            | Self::YetiRankPairwise {
                permutations,
                decay,
            } => {
                if *permutations < 1 {
                    return Err(cb_core::CbError::OutOfRange(
                        "YetiRank permutations must be >= 1, got 0".to_owned(),
                    ));
                }
                if !decay.is_finite() || !(0.0..=1.0).contains(decay) {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "YetiRank decay must be finite and in [0, 1], got {decay}"
                    )));
                }
            }
            // StochasticRank (Wave C, LOSS-04): `sigma > 0` (the Gaussian noise
            // scale; a non-positive sigma collapses the Monte-Carlo perturbation,
            // `error_functions.cpp:1045`, T-06.3-04-03), `mu >= 0` (the
            // tie-resolving coefficient, `error_functions.cpp:1027`), and
            // `num_estimations >= 1` (a zero sample count averages over nothing →
            // `der / 0`, `error_functions.cpp:1219`). The `metric` enum admits
            // only DCG/NDCG (the transcribed arm); any other future variant would
            // be a non-exhaustive bug — both current variants are accepted.
            Self::StochasticRank {
                sigma,
                mu,
                num_estimations,
                metric: _,
            } => {
                if !sigma.is_finite() || *sigma <= 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "StochasticRank sigma must be finite and > 0, got {sigma}"
                    )));
                }
                if !mu.is_finite() || *mu < 0.0 {
                    return Err(cb_core::CbError::OutOfRange(format!(
                        "StochasticRank mu must be finite and >= 0, got {mu}"
                    )));
                }
                if *num_estimations < 1 {
                    return Err(cb_core::CbError::OutOfRange(
                        "StochasticRank num_estimations must be >= 1, got 0".to_owned(),
                    ));
                }
            }
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
            | Self::MultiCrossEntropy
            // RMSEWithUncertainty (Wave B, LOSS-08): no hyperparameters — the two
            // approx dims (mean, log-scale) and the diagonal hessian are fixed.
            | Self::RmseWithUncertainty
            | Self::QueryRmse
            | Self::PairLogit
            // Custom (LOSS-07): the user objective is an opaque trait object —
            // there are no variant-level hyperparameters for `validate` to range-
            // check. Any der-domain precondition is the implementor's; the
            // consumer (`compute_gradients`) rejects non-finite ders at runtime
            // (T-06.4D-02). Nothing to reject here.
            | Self::Custom(_)
            | Self::PairLogitPairwise => {}
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
    /// SolarL2 split score (`score_calcers.cuh:22-24`): per-leaf term
    /// `weight > 1e-20 ? (-sum*sum)*(1 + 2*ln(weight + 1.0))/weight : 0.0`.
    /// GPU-only upstream (D-6.4-06): no CPU training ground truth — the Rust port
    /// is self-oracled against the hand-computed CUDA arithmetic, NOT a strict lock.
    SolarL2,
    /// NewtonL2 split score (`pointwise_scores.cu:504-510`): reuses the [`Self::L2`]
    /// formula VERBATIM; the only difference is the histogram fill, which feeds the
    /// summed positive `der2` hessian into the `sum_weight` leaf-stat slot
    /// (second-order, `enum_helpers.cpp:830-847`). GPU-only upstream (D-6.4-06).
    NewtonL2,
    /// NewtonCosine split score (`pointwise_scores.cu:512-521`): reuses the
    /// [`Self::Cosine`] formula VERBATIM; like [`Self::NewtonL2`] the difference is
    /// the der2 histogram fill, not the score math. GPU-only upstream (D-6.4-06).
    NewtonCosine,
    /// LOOL2 (leave-one-out L2) split score (`score_calcers.cuh:83-87`):
    /// `adjust = weight>1.0 ? weight/(weight-1.0) : 0.0; adjust*=adjust;
    /// weight>0.0 ? adjust*(-sum*sum)/weight : 0.0`. GPU-only upstream (D-6.4-06).
    LOOL2,
    /// SatL2 (saturated L2) split score (`score_calcers.cuh:114-117`):
    /// `adjust = weight>2.0 ? weight*(weight-2.0)/(weight*weight-3.0*weight+1.0) : 0.0;
    /// weight>0.0 ? adjust*(-sum*sum)/weight : 0.0`. GPU-only upstream (D-6.4-06).
    SatL2,
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
/// A host-typed, CubeCL-free description of a single tree grown entirely on the
/// device (GPUT-01). This is the ONLY structure that crosses the
/// [`Runtime::grow_tree_on_device`] seam back to cb-train; it deliberately holds
/// only plain host types so the `Runtime` trait — and therefore cb-train — never
/// gains a `cubecl`/`cb-backend` dependency via feature unification (T-10-04).
///
/// # Fields
/// - `splits`: the per-level chosen oblivious split as `(feature_index, bin_id)`.
///   The pass test the caller applies is `quantized_bin[feature] > bin_id`, and
///   the caller resolves the concrete threshold with
///   `border = feature_borders[feature][bin_id]` (see
///   `cb_train::tree::FeatureMatrix`). Length equals the tree depth.
/// - `leaf_values`: the per-leaf values in leaf-index order, a
///   `leaf_count × approx_dim` ROW-MAJOR block (`approx_dim == 1` for the scalar
///   path ⇒ the flat `2^depth`-length vector, byte-unchanged, D-04). These are
///   **UN-scaled** by `learning_rate`; cb-train applies the shrinkage downstream
///   so the device path matches the CPU leaf-update contract.
/// - `approx_dim`: the per-leaf approximant dimension `K` (`1` for every scalar
///   construction; `> 1` only for the Phase-13 multi-output leaf blocks).
/// - `leaf_of`: the per-object leaf index. In the production hot path this is
///   **empty** (length 0) so the `n`-length buffer never crosses the seam per
///   tree (D-05); it is populated to length `n` ONLY for the oracle structure
///   check that compares device vs. CPU leaf assignment.
/// - `step_nodes` / `node_id_to_leaf_id`: the NON-SYMMETRIC node-graph carrier
///   (Phase 12 Plan 01, GPUT-18) that mirrors `cb_train::tree::GrownTree`
///   verbatim so the Plan-03 Depthwise/Lossguide fold arm and the Plan-04 Region
///   fold arm in `cb_train::boosting` can materialize a `NonSymmetricTree` /
///   `RegionTree` (NOT always an `ObliviousTree`). For the oblivious / depth>1
///   symmetric path these vectors are **EMPTY** — the oblivious emission is
///   byte-unchanged (D-04 no-regression); only the non-symmetric emission fills
///   them. Like every other field these are PLAIN HOST types — no `cubecl` /
///   `cb-backend` type may appear on this struct (T-10-04 feature-unification
///   landmine: the seam must never pull a backend dep into `cb-train`).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceGrownTree {
    /// Per-level chosen split as `(feature_index, bin_id)`; length = tree depth.
    /// Pass test: `quantized_bin[feature] > bin_id`. The caller resolves the
    /// border via `feature_borders[feature][bin_id]`.
    pub splits: Vec<(u32, u32)>,
    /// Per-leaf values in leaf-index order. UN-scaled by `learning_rate`
    /// (cb-train applies the shrinkage).
    ///
    /// # Multi-output block layout (Phase 13 Plan 06, GPUT-12, D-03)
    /// `leaf_values` is a `leaf_count × approx_dim` ROW-MAJOR block: leaf `l`'s
    /// per-dimension vector occupies the contiguous span
    /// `leaf_values[l * approx_dim .. (l + 1) * approx_dim]`, so dimension `d` of
    /// leaf `l` is `leaf_values[l * approx_dim + d]`. The block routes through the
    /// existing multi-output CPU apply layout
    /// `approx[d * n + i] += lr * leaf_values[leaf_of[i] * approx_dim + d]`.
    ///
    /// At `approx_dim == 1` (the scalar / oblivious path) this collapses to the
    /// pre-Phase-13 flat `leaf_count`-length vector and the bytes are IDENTICAL
    /// (GPUT-14 / D-04 no-regression): `leaf_values[l * 1 + 0] == leaf_values[l]`.
    pub leaf_values: Vec<f64>,
    /// The per-leaf approximant dimension `K` — `leaf_values` is a
    /// `leaf_count × approx_dim` row-major block (see the [`Self::leaf_values`]
    /// layout note). `1` for every scalar / oblivious / non-symmetric / Region
    /// construction (the block collapses to the flat scalar vector, byte-unchanged,
    /// D-04); `> 1` only for the multi-output (multiclass / multi-target) leaf
    /// blocks the Phase-13 Plan-07 family emits. Plain host `usize` — never a
    /// `cubecl` / `cb-backend` type (T-10-04 feature-unification landmine).
    pub approx_dim: usize,
    /// Per-object leaf index. EMPTY (length 0) in the production hot path so the
    /// `n`-length buffer never crosses the seam per tree (D-05); length `n` only
    /// for the oracle structure check.
    pub leaf_of: Vec<u32>,
    /// NON-SYMMETRIC node graph, `(left_subtree_diff, right_subtree_diff)` per
    /// node (mirrors `cb_train::tree::GrownTree::step_nodes` verbatim, CR-01/02).
    /// `(0, 0)` marks a terminal leaf. **EMPTY** for the oblivious / symmetric
    /// path (that emission is byte-unchanged, D-04); ONLY the Plan-03 non-sym
    /// device grow fills it. Plain host type — never a `cubecl` type.
    pub step_nodes: Vec<(u16, u16)>,
    /// NON-SYMMETRIC node → leaf id map (mirrors
    /// `cb_train::tree::GrownTree::node_id_to_leaf_id` verbatim). `u32::MAX` marks
    /// an interior node (the sentinel); any other value is that node's distinct
    /// leaf id. **EMPTY** for the oblivious / symmetric path; ONLY the Plan-03
    /// non-sym device grow fills it. Plain host type — never a `cubecl` type.
    pub node_id_to_leaf_id: Vec<u32>,
    /// REGION path carrier (Phase 12 Plan 04, GPUT-18, D-03a). The per-level
    /// walk-until-diverge Region path (upstream `TRegionModel`), one entry per
    /// level as `(feature_index, bin_id, expected_direction, one_hot)`:
    /// - `feature_index` / `bin_id`: the level's chosen split; the caller resolves
    ///   `border = feature_borders[feature][bin_id]` (the SAME `bin > bin_id` /
    ///   `value > border` forward-bit test the oblivious path uses).
    /// - `expected_direction`: the CONTINUE direction — the walk descends to the
    ///   next level while `(value > border) == expected_direction`, and diverges
    ///   into the current terminal bin on the first mismatch.
    /// - `one_hot`: the categorical one-hot flag (always `false` for the device
    ///   float Region grow; carried for structural fidelity / future parity).
    ///
    /// A depth-`d` Region has `d` path entries and exactly `d + 1` leaf values
    /// (NOT `2^d` — the node-graph failure signal). **EMPTY** for the oblivious /
    /// non-symmetric path (`step_nodes` empty AND `region_path` empty ⇒ oblivious;
    /// non-empty `step_nodes` ⇒ non-symmetric; non-empty `region_path` ⇒ Region);
    /// ONLY the Plan-04 device Region grow fills it. Plain host type — never a
    /// `cubecl` type (the seam must never pull a backend dep into `cb-train`).
    pub region_path: Vec<(u32, u32, bool, bool)>,
}

/// The device grow policy (Phase 12 Plan 01, GPUT-18), a PLAIN HOST enum mirroring
/// upstream `EGrowPolicy` BY VALUE. Deliberately re-declared here (NOT `use cb_train`)
/// so the [`Runtime`] seam — and therefore `cb-train` — never gains a `cubecl` /
/// `cb-backend` dependency via feature unification (T-10-04 landmine). The device
/// path covers only [`Self::SymmetricTree`] today; the non-symmetric arms are flipped
/// on by later Phase-12 waves (Plan 03 Depthwise/Lossguide, Plan 04 Region).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceGrowPolicy {
    /// Oblivious / symmetric tree (the only device-covered policy in Plan 01).
    #[default]
    SymmetricTree,
    /// Depth-wise non-symmetric growth (Plan 03).
    Depthwise,
    /// Loss-guided leaf-wise non-symmetric growth (Plan 03).
    Lossguide,
    /// Region growth — no CPU path upstream (Plan 04, built device-first).
    Region,
}

/// The device bootstrap (sampling) type (Phase 12 Plan 01), a PLAIN HOST enum mirroring
/// upstream `EBootstrapType` BY VALUE. The Plan-01 covered regime does NO subsampling
/// ([`Self::No`], every object at full weight); the sampling arms are flipped on by
/// Plan 06 (bootstrap) / Plan 07 (MVS).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceBootstrapType {
    /// No subsampling — the full dataset at uniform weight (the covered regime).
    #[default]
    No,
    /// Bayesian bootstrap (Plan 06).
    Bayesian,
    /// Bernoulli subsampling (Plan 06).
    Bernoulli,
    /// Minimal-Variance Sampling (Plan 07).
    Mvs,
    /// Poisson bootstrap (Plan 06).
    Poisson,
}

/// One CTR column to accumulate on device and binarize into an ADDITIONAL cindex column (Phase 12
/// Plan 08, GPUT-10). A PLAIN HOST struct (no `cubecl` type — the T-10-04 seam landmine). Covers
/// the ordered target statistic, one-hot (a single small-cardinality member), and the tensor /
/// feature-combination projection (`member_bins.len() >= 2`, A5 — the device CTR math is shared;
/// only the combined projection differs, and it is a host pre-step).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceCtrColumn {
    /// The categorical member bin columns (object order, already hashed to category codes).
    /// Length-1 for a plain single-feature / one-hot CTR; length `>= 2` for a tensor /
    /// feature-combination projection (folded into one combined bin column on device-host, A5).
    pub member_bins: Vec<Vec<u32>>,
    /// The additive CTR prior numerator (`calc_ctr_online` `(good + prior) / (total + 1)`).
    pub prior: f64,
    /// The border table binarizing the accumulated CTR VALUE into cindex bins (the `> bin`
    /// threshold convention; uploaded once per fit — quantization stays the CPU ≤1e-5 reference).
    pub borders: Vec<f64>,
}

/// The device CTR (categorical target statistics) config (Phase 12 Plan 08, GPUT-10). A PLAIN
/// HOST struct carrying the ORDERED CTR inputs the device kernel + cindex-join need: the single
/// learn permutation (single-permutation covered regime, Open Q3 — multi-fold defers behind
/// `Ok(None)`), the binclf target class per object, and the per-column CTR specs (priors,
/// projections, border tables). Its mere PRESENCE (`DeviceTrainConfig::ctr == Some(_)`) declines
/// via [`DeviceTrainConfig::is_covered_regime`]; the session's CTR gate arm (Plan 08) flips a
/// single-permutation covered CTR config to the device path. No `cubecl` type appears here.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceCtrConfig {
    /// The single learn permutation (learn-order object indices) the ordered CTR reads-before-
    /// increments over. Single-permutation covered regime (Open Q3); a multi-fold / multi-
    /// permutation CTR is NOT covered this wave (the session declines it to `Ok(None)`).
    pub permutation: Vec<u32>,
    /// The binclf target class per object (object order), each in `{0, 1}`.
    pub target_class: Vec<u32>,
    /// The CTR columns to accumulate + binarize into ADDITIONAL cindex columns.
    pub columns: Vec<DeviceCtrColumn>,
}

/// The single PLAIN HOST-typed device training config (Phase 12 Plan 01, Open Q2). It
/// carries the grow-policy / leaf-method / sampling / exact / CTR knobs the later
/// Phase-12 waves need, so [`Runtime::begin_device_training`]'s config surface widens by
/// mutating THIS struct instead of growing the per-call argument list per wave. Every
/// field is a plain host type (no `cubecl` / `cb-backend` type may appear here — the
/// T-10-04 seam landmine).
///
/// [`Self::default`] reproduces TODAY'S device-covered regime exactly (SymmetricTree,
/// no subsampling, RMSE/Newton leaf, no exact, no CTR), so an existing caller passing the
/// default is byte-unchanged (D-04). For Plan 01 EVERY non-default value still routes to
/// `Ok(None)` at the coverage gate — later waves flip each arm on as its device kernel lands.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceTrainConfig {
    /// The grow policy; only [`DeviceGrowPolicy::SymmetricTree`] is device-covered in Plan 01.
    pub grow_policy: DeviceGrowPolicy,
    /// Max leaves for the leaf-wise policies (`None` == unbounded / not applicable). Any
    /// `Some(_)` declines in Plan 01.
    pub max_leaves: Option<usize>,
    /// Minimum objects per leaf (the covered regime uses `1`).
    pub min_data_in_leaf: usize,
    /// The bootstrap / subsampling type; only [`DeviceBootstrapType::No`] is covered in Plan 01.
    pub bootstrap_type: DeviceBootstrapType,
    /// The subsample rate (the covered regime is `1.0` — the full dataset).
    pub sample_rate: f32,
    /// The MVS regularization λ (`None` unless MVS sampling is active — Plan 07).
    pub mvs_lambda: Option<f64>,
    /// The pinned per-fit RNG seed (device-resident sampling reproducibility, Plan 06/07).
    pub rng_seed: u64,
    /// The exact-leaf (order-statistic) leaf method flag; `true` declines in Plan 01 (Plan 05).
    pub exact_leaf: bool,
    /// The quantile α for the exact-leaf method (Plan 05).
    pub quantile_alpha: f64,
    /// The quantile δ boundary for the exact-leaf method (Plan 05).
    pub quantile_delta: f64,
    /// The device CTR config; `Some(_)` declines in Plan 01 (Plan 08 fills + covers it).
    pub ctr: Option<DeviceCtrConfig>,
}

impl Default for DeviceTrainConfig {
    fn default() -> Self {
        Self {
            grow_policy: DeviceGrowPolicy::SymmetricTree,
            max_leaves: None,
            min_data_in_leaf: 1,
            bootstrap_type: DeviceBootstrapType::No,
            sample_rate: 1.0,
            mvs_lambda: None,
            rng_seed: 0,
            exact_leaf: false,
            quantile_alpha: 0.5,
            quantile_delta: 1e-6,
            ctr: None,
        }
    }
}

impl DeviceTrainConfig {
    /// `true` iff this config is the exact Plan-01 device-covered regime (SymmetricTree,
    /// no subsampling, RMSE/Newton leaf, no exact-leaf, no CTR, no leaf-count cap). The
    /// coverage gate declines (`Ok(None)`) on anything else — every later Phase-12 wave
    /// flips its own arm here as its device kernel lands (D-10-01 all-or-nothing).
    #[must_use]
    pub fn is_covered_regime(&self) -> bool {
        self.grow_policy == DeviceGrowPolicy::SymmetricTree
            && self.bootstrap_type == DeviceBootstrapType::No
            && self.mvs_lambda.is_none()
            && !self.exact_leaf
            && self.ctr.is_none()
            && self.max_leaves.is_none()
    }
}

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

    /// Begin a device-resident training session (GPUT-01). Uploads the
    /// once-per-fit, tree-invariant state — the bins-feature-major quantized
    /// design matrix (the cindex), per-object `weight`, and the scalar training
    /// hyperparameters — to the device so [`Self::grow_tree_on_device`] can grow
    /// each tree without re-crossing the seam with `n`-length buffers (D-05).
    ///
    /// `bins_feature_major` is the feature-major flat quantized matrix
    /// (`bins_feature_major[f * n + i]`); `score_function` selects the split-score
    /// calcer (GPUT-08; catboost's CPU default is [`EScoreFunction::Cosine`]).
    /// `scaled_l2` is the leaf-regularization already folded to the device's
    /// scale. `boosting_type_is_plain` distinguishes Plain from Ordered.
    ///
    /// Returns `Ok(true)` when the backend accepted the session and will grow
    /// trees on the device; `Ok(false)` (the DEFAULT) means the backend declines
    /// and the caller must use the CPU path (D-04). The default implementation
    /// binds every parameter and returns `Ok(false)` so every existing
    /// `Runtime` impl inherits the transparent CPU fallback unchanged.
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if a backend override fails to allocate or
    /// upload the session state. The default implementation never errors.
    ///
    /// `config` is the single plain-host [`DeviceTrainConfig`] carrying the grow-policy /
    /// sampling / exact-leaf / CTR knobs (Phase 12 Plan 03 promotion — the config surface
    /// widens by mutating this struct, not the argument list). The default covered regime is
    /// [`DeviceTrainConfig::default`]; a Depthwise / Lossguide `grow_policy` flips the
    /// non-symmetric device arm on.
    #[allow(clippy::too_many_arguments)]
    fn begin_device_training(
        &self,
        loss: &Loss,
        depth: usize,
        boosting_type_is_plain: bool,
        fold_count: usize,
        score_function: EScoreFunction,
        bins_feature_major: &[u32],
        weight: &[f64],
        n: usize,
        n_features: usize,
        n_bins: usize,
        learning_rate: f64,
        scaled_l2: f64,
        config: &DeviceTrainConfig,
    ) -> CbResult<bool> {
        let _ = (
            loss,
            depth,
            boosting_type_is_plain,
            fold_count,
            score_function,
            bins_feature_major,
            weight,
            n,
            n_features,
            n_bins,
            learning_rate,
            scaled_l2,
            config,
        );
        Ok(false)
    }

    /// Grow one oblivious tree on the device from the current `approx` and
    /// `target`, using the session state uploaded by
    /// [`Self::begin_device_training`] (GPUT-01). The only `n`-length buffers
    /// that cross the seam are `approx`/`target` in and — behind the oracle flag —
    /// `DeviceGrownTree::leaf_of` out; the hot path keeps `leaf_of` empty (D-05).
    ///
    /// Returns `Ok(Some(tree))` with the host-typed [`DeviceGrownTree`] when the
    /// backend grew the tree, or `Ok(None)` (the DEFAULT) to signal the caller to
    /// fall back to the CPU grow loop (D-04 / the oracle safety net T-10-05 —
    /// never a fabricated device result). The default implementation binds its
    /// parameters and returns `Ok(None)`.
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if a backend override fails mid-grow. The
    /// default implementation never errors.
    fn grow_tree_on_device(
        &self,
        approx: &[f64],
        target: &[f64],
    ) -> CbResult<Option<DeviceGrownTree>> {
        let _ = (approx, target);
        Ok(None)
    }

    /// End the device training session opened by
    /// [`Self::begin_device_training`], releasing the device-resident session
    /// state (GPUT-01). The default implementation is a no-op returning `Ok(())`
    /// so backends that never opened a session (the CPU fallback) are unaffected.
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if a backend override fails to release
    /// device resources. The default implementation never errors.
    fn end_device_training(&self) -> CbResult<()> {
        Ok(())
    }
}
