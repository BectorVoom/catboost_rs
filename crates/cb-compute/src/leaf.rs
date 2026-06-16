//! Leaf-value estimation primitives — all four TRAIN-03 leaf-estimation methods
//! (D-09): Gradient, Newton, Simple, and Exact, plus the shared `CalcAverage` /
//! `ScaleL2Reg` helpers. Gradient/Newton/Simple are L2-regularized closed-form
//! deltas over a leaf's already-reduced derivative sums; Exact is the
//! quantile-style exact optimum over the leaf's per-member residuals. Every SUM
//! over leaf members is done by the caller through `cb_core::sum_f64` (D-02/D-05);
//! the closed-form helpers consume an already-reduced `sum_der` / `sum_weight` /
//! `sum_der2`.
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo_helpers/online_predictor.h:112-178`:
//! - `CalcAverage(sumDelta, count, scaledL2) = count > 0 ? sumDelta/(count +
//!   scaledL2) : 0.0` — the guarded average; an empty leaf returns `0.0` rather
//!   than dividing by zero (T-03-02-01 mitigation).
//! - `ScaleL2Reg(l2, sumAllWeights, allDocCount) = l2 * (sumAllWeights /
//!   allDocCount)` — the per-tree L2 scaling applied to every leaf's denominator.
//! - Gradient leaf delta = `CalcAverage(SumDer, SumWeights, scaledL2)`. For the
//!   unweighted path every object weight is `1.0`, so `SumWeights` is the leaf's
//!   object count and `sumAllWeights/allDocCount == 1`, giving `scaledL2 == l2`.
//! - `CalcDeltaNewtonBody(sumDer, sumDer2, l2, sumAllW, docCount) = sumDer /
//!   (-sumDer2 + scaledL2)` (`online_predictor.h:162-170`) — the Newton delta.
//!   For RMSE `der2 == -1` so `-sumDer2 == sumWeight` and Newton == Gradient;
//!   Logloss `der2 == -p(1-p)` makes Newton genuinely distinct.
//!
//! `catboost/private/libs/algo/approx_calcer.cpp:482-525` (`CalcLeafDeltasSimple`)
//! dispatches `ELeavesEstimation::{Newton, Gradient}`; the `Simple` enum value
//! falls into the same Gradient branch — so the SIMPLE method's leaf delta is
//! identical to GRADIENT for these losses (A6 resolved empirically against
//! catboost 1.2.10).
//!
//! `catboost/private/libs/algo/approx_calcer.cpp:681-704` (`CalcExactLeafDeltas`)
//! collects each leaf's residuals `r_i = target_i - approx_i` (as `f32`) and
//! weights `w_i`, then sets the leaf delta to
//! `CalcOneDimensionalOptimumConstApprox(loss, r, w)`
//! (`optimal_const_for_loss.h:180-216`). For MAE / Quantile(alpha, delta) that is
//! the weighted sample quantile `CalcSampleQuantile`
//! (`quantile.cpp` — `CalcSampleQuantileLinearSearch` for `< 100` samples:
//! stable-sort `r` ascending, accumulate `w`, return the first value whose
//! running weight `>= totalWeight*alpha - DBL_EPSILON`), then the delta
//! adjustment from `CalculateWeightedTargetQuantile`
//! (`optimal_const_for_loss.h:69-103`): `q -= delta` if
//! `lessWeight + equalWeight*alpha >= needWeight - DBL_EPSILON`, else `q += delta`.
//!
//! # f64 discipline & summation routing (D-07 / D-08)
//!
//! The closed-form helpers perform only scalar arithmetic on already-reduced
//! sums; they never spell a float fold, so the D-08 raw-sum ban does not touch
//! them — the reduction lives in the caller via `cb_core::sum_f64`. The Exact
//! quantile sorts per-leaf members (no float SUM of derivatives spelled here; the
//! only accumulation is the ascending weight scan that defines the quantile, which
//! mirrors upstream's `sumWeight +=` linear search exactly).

use cb_core::sum_f64;

/// `DBL_EPSILON` — the C++ `<cfloat>` `DBL_EPSILON` the upstream quantile search
/// compares against (`quantile.cpp:67/98`, `optimal_const_for_loss.h:95`).
const DBL_EPSILON: f64 = f64::EPSILON;

/// Which leaf-estimation method computes a tree's leaf deltas (TRAIN-03 / D-09).
/// Mirrors upstream `ELeavesEstimation` (`enums.h:64-70`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafMethod {
    /// `CalcAverage(SumDer, SumWeights, scaledL2)`.
    Gradient,
    /// `SumDer / (-SumDer2 + scaledL2)` (`CalcDeltaNewtonBody`).
    Newton,
    /// Dispatches to the Gradient leaf delta upstream (`CalcLeafDeltasSimple`
    /// Gradient branch; A6) — identical closed form to [`LeafMethod::Gradient`].
    Simple,
    /// Quantile-style exact optimum over the leaf's per-member residuals
    /// (`CalcExactLeafDeltas` -> `CalcOneDimensionalOptimumConstApprox`).
    Exact,
}

/// L2-regularized guarded average: `count > 0 ? sum_delta/(count + scaled_l2) :
/// 0.0`.
///
/// `online_predictor.h` `CalcAverage`. The `count > 0` guard means a degenerate
/// (empty) leaf returns `0.0` — no division by zero, no panic (T-03-01-01).
/// `count` is the leaf's summed weight (object count in the unweighted path).
#[must_use]
pub fn calc_average(sum_delta: f64, count: f64, scaled_l2: f64) -> f64 {
    if count > 0.0 {
        sum_delta / (count + scaled_l2)
    } else {
        0.0
    }
}

/// Per-tree L2 scaling: `l2 * (sum_all_weights / doc_count)`.
///
/// `online_predictor.h` `ScaleL2Reg`. `doc_count` is the total object count;
/// `sum_all_weights` is the total weight. For the unweighted path
/// `sum_all_weights == doc_count`, so this returns `l2`. Returns `l2` directly
/// when `doc_count == 0` to avoid a division by zero on a degenerate dataset
/// (the trainer rejects empty datasets upstream of this primitive).
#[must_use]
pub fn scale_l2_reg(l2: f64, sum_all_weights: f64, doc_count: usize) -> f64 {
    if doc_count == 0 {
        l2
    } else {
        l2 * (sum_all_weights / doc_count as f64)
    }
}

/// The Gradient-method leaf delta: `CalcAverage(sum_der, sum_weight, scaled_l2)`.
///
/// `sum_der` is the leaf's reduced first-derivative sum, `sum_weight` its summed
/// weight (object count unweighted), `scaled_l2` the [`scale_l2_reg`] output.
/// This is the unscaled-by-learning-rate delta; the boosting loop multiplies the
/// stored leaf value by `learning_rate`.
#[must_use]
pub fn gradient_leaf_delta(sum_der: f64, sum_weight: f64, scaled_l2: f64) -> f64 {
    calc_average(sum_der, sum_weight, scaled_l2)
}

/// The Newton-method leaf delta: `sum_der / (-sum_der2 + scaled_l2)`
/// (`CalcDeltaNewtonBody`, `online_predictor.h:162-170`).
///
/// `sum_der` / `sum_der2` are the leaf's reduced first/second-derivative sums
/// (the per-object weight already folded in by the caller, matching upstream
/// `TSum`); `scaled_l2` is the [`scale_l2_reg`] output. The denominator
/// `-sum_der2 + scaled_l2` is guarded: a degenerate `<= 0` denominator (an empty
/// leaf, or a loss with `der2 == 0` such as MAE/Quantile where Newton is
/// undefined) returns `0.0` rather than dividing by zero or producing a NaN/inf
/// (T-03-02-01 mitigation — never panic/div-by-zero). For RMSE `der2 == -1` so
/// `-sum_der2 == sum_weight` and this equals the Gradient delta.
#[must_use]
pub fn newton_leaf_delta(sum_der: f64, sum_der2: f64, scaled_l2: f64) -> f64 {
    let denom = -sum_der2 + scaled_l2;
    if denom > 0.0 {
        sum_der / denom
    } else {
        0.0
    }
}

/// The Simple-method leaf delta. Upstream `CalcLeafDeltasSimple`
/// (`approx_calcer.cpp:506-524`) routes `ELeavesEstimation::Simple` through the
/// Gradient branch, so this is identical to [`gradient_leaf_delta`] (A6).
#[must_use]
pub fn simple_leaf_delta(sum_der: f64, sum_weight: f64, scaled_l2: f64) -> f64 {
    gradient_leaf_delta(sum_der, sum_weight, scaled_l2)
}

/// The MultiClass softmax per-leaf SYMMETRIC Newton solve — the single piece of
/// genuinely-new numerical machinery this phase (RESEARCH "Key insight"),
/// transcribing `TSymmetricHessian::SolveNewtonEquation`
/// (`hessian.cpp:22-52`) verbatim with a hand-rolled symmetric positive-definite
/// solve (the RESEARCH-recommended hand-roll default — NO new linalg crate, so the
/// T-6.2-SC supply-chain gate is a no-op).
///
/// # Inputs
/// - `sum_der`: the leaf's summed per-dimension first derivative `Σ der1[d]`
///   (length `k`), the OUTPUT of `softmax_ders`' `der1` summed over the leaf's
///   members through `cb_core::sum_f64` (D-08).
/// - `sum_der2_packed`: the leaf's summed PACKED lower-triangular Hessian (length
///   `k*(k+1)/2`), `Σ` of `softmax_ders`' `der2` over the same members, in the same
///   `[(0,0),(0,1),…]` order.
/// - `scaled_l2`: the per-tree `scale_l2_reg(l2, sumAllWeights, docCount)` output
///   (upstream `l2Regularizer *= sumAllWeights/allDocCount` is applied by the
///   caller, `online_predictor.cpp:17`).
///
/// # Algorithm (`hessian.cpp:30-50`)
/// 1. `negativeDer = -sum_der`.
/// 2. `maxTraceElement = max(scaled_l2, max_d(-H[d][d]))` at **`f32` precision** —
///    the trace-epsilon regularizer uses `f32::EPSILON` (Pitfall 5; reproduced
///    bit-faithfully so leaf values match `<= 1e-5`).
/// 3. `adjustedL2 = max(scaled_l2, maxTraceElement * f32::EPSILON)`.
/// 4. Subtract `adjustedL2` from each diagonal, NEGATE the whole matrix, solve
///    `M · x = negativeDer`, then NEGATE the result: `res = -x` (the leaf delta per
///    dimension).
///
/// The negated matrix `M = -(H - adjustedL2·I)` is symmetric positive definite for
/// the softmax Hessian (the diagonal `-(p_y(p_y-1)) = p_y(1-p_y) > 0` dominates),
/// so a Cholesky `M = L·Lᵀ` solve reproduces LAPACK `dppsv_`'s result to machine
/// precision. The `k <= ~10` in-scope class counts make the dense solve trivial.
///
/// Returns the per-dimension leaf delta (length `k`). A non-positive-definite or
/// degenerate system (an empty leaf — all-zero stats) returns all-zeros rather than
/// panicking (no `unwrap`, no div-by-zero — T-6.2-01 discipline).
#[must_use]
pub fn solve_symmetric_newton(
    sum_der: &[f64],
    sum_der2_packed: &[f64],
    scaled_l2: f64,
) -> Vec<f64> {
    let k = sum_der.len();
    if k == 0 {
        return Vec::new();
    }
    // Reconstruct the dense symmetric Hessian H[i][j] from the packed
    // [(0,0),(0,1),…,(0,k-1),(1,1),…] order (mirrors `softmax_ders`' packing).
    // Each packed entry is written to BOTH H[i][j] and H[j][i] via index writes
    // (the borrow checker forbids holding two `&mut` rows at once).
    let mut h = vec![vec![0.0_f64; k]; k];
    let mut idx = 0usize;
    for i in 0..k {
        for j in i..k {
            let v = sum_der2_packed.get(idx).copied().unwrap_or(0.0);
            idx += 1;
            if let Some(cell) = h.get_mut(i).and_then(|r| r.get_mut(j)) {
                *cell = v;
            }
            if let Some(cell) = h.get_mut(j).and_then(|r| r.get_mut(i)) {
                *cell = v;
            }
        }
    }

    // maxTraceElement at f32 precision (hessian.cpp:35-38): start at scaled_l2 (the
    // `l2Regularizer` arg), then max with each `-H[d][d]`. The Max<float> cast and
    // the `* numeric_limits<float>::epsilon()` are reproduced at f32 (Pitfall 5).
    let mut max_trace = scaled_l2 as f32;
    for d in 0..k {
        let diag = h.get(d).and_then(|r| r.get(d)).copied().unwrap_or(0.0);
        max_trace = max_trace.max((-diag) as f32);
    }
    let adjusted_l2 = (scaled_l2 as f32).max(max_trace * f32::EPSILON) as f64;

    // M = -(H - adjustedL2·I): subtract adjustedL2 from the diagonal, then negate
    // the entire matrix (hessian.cpp:41-47).
    for d in 0..k {
        if let Some(cell) = h.get_mut(d).and_then(|r| r.get_mut(d)) {
            *cell -= adjusted_l2;
        }
    }
    for row in &mut h {
        for cell in row.iter_mut() {
            *cell = -*cell;
        }
    }

    // negativeDer = -sum_der (hessian.cpp:32 via online_predictor.cpp:13-16).
    let neg_der: Vec<f64> = sum_der.iter().map(|&d| -d).collect();

    // Solve M · x = neg_der via Cholesky (M is SPD). Then res = -x.
    match cholesky_solve(&h, &neg_der) {
        Some(x) => x.into_iter().map(|v| -v).collect(),
        None => vec![0.0; k],
    }
}

/// Solve the dense symmetric positive-definite system `a · x = b` via a Cholesky
/// factorization `a = L·Lᵀ` followed by forward/back substitution. `a` is `k×k`
/// row-major; `b` is length `k`. Returns `None` if `a` is not positive definite (a
/// non-positive pivot), so the caller can fall back to zeros rather than producing
/// a NaN. Used only by [`solve_symmetric_newton`] for the tiny (`k <= ~10`) softmax
/// leaf systems.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let k = b.len();
    let mut l = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
            for p in 0..j {
                let lip = l.get(i).and_then(|r| r.get(p)).copied().unwrap_or(0.0);
                let ljp = l.get(j).and_then(|r| r.get(p)).copied().unwrap_or(0.0);
                sum -= lip * ljp;
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                let diag = sum.sqrt();
                if let Some(cell) = l.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *cell = diag;
                }
            } else {
                let ljj = l.get(j).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                if ljj == 0.0 {
                    return None;
                }
                if let Some(cell) = l.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *cell = sum / ljj;
                }
            }
        }
    }
    // Forward solve L·y = b.
    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        let mut sum = b.get(i).copied().unwrap_or(0.0);
        for p in 0..i {
            let lip = l.get(i).and_then(|r| r.get(p)).copied().unwrap_or(0.0);
            let yp = y.get(p).copied().unwrap_or(0.0);
            sum -= lip * yp;
        }
        let lii = l.get(i).and_then(|r| r.get(i)).copied().unwrap_or(0.0);
        if lii == 0.0 {
            return None;
        }
        if let Some(slot) = y.get_mut(i) {
            *slot = sum / lii;
        }
    }
    // Back solve Lᵀ·x = y.
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut sum = y.get(i).copied().unwrap_or(0.0);
        for p in (i + 1)..k {
            let lpi = l.get(p).and_then(|r| r.get(i)).copied().unwrap_or(0.0);
            let xp = x.get(p).copied().unwrap_or(0.0);
            sum -= lpi * xp;
        }
        let lii = l.get(i).and_then(|r| r.get(i)).copied().unwrap_or(0.0);
        if lii == 0.0 {
            return None;
        }
        if let Some(slot) = x.get_mut(i) {
            *slot = sum / lii;
        }
    }
    Some(x)
}

/// The Exact-method leaf delta — the weighted sample quantile of a leaf's
/// per-member residuals, with the upstream alpha/delta adjustment
/// (`CalcExactLeafDeltas` -> `CalcOneDimensionalOptimumConstApprox` ->
/// `CalculateWeightedTargetQuantile`).
///
/// `residuals[i]` is member `i`'s `target_i - approx_i` (the caller widens through
/// `f32` to match upstream's `TVector<float> leafSamples`); `weights[i]` its
/// object weight. `alpha`/`delta` are the loss's quantile parameters (MAE /
/// Quantile default `alpha = 0.5`, `delta = 1e-6`). An empty leaf returns `0.0`
/// (`CalcSampleQuantile` empty guard). For `< 100` samples this is the linear
/// search; the binary search (`>= 100`) is not needed for the Phase-3 corpora and
/// would be added additively if a larger leaf appears.
///
/// The leaf members are processed in the caller-supplied order; the quantile is
/// order-independent (a stable sort over `(value, weight)` pairs), so no canonical
/// SUM order is at stake here (D-05 governs derivative sums, not this rank
/// statistic).
#[must_use]
pub fn exact_leaf_delta(residuals: &[f32], weights: &[f64], alpha: f64, delta: f64) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    // alpha <= 0 -> min element (CalcSampleQuantile:113-115).
    if alpha <= 0.0 {
        let mut min = f64::INFINITY;
        for &v in residuals {
            let v = f64::from(v);
            if v < min {
                min = v;
            }
        }
        return min;
    }

    // Pair each residual with its weight (default 1.0 when no weights supplied),
    // then STABLE-sort ascending by value — CalcSampleQuantileLinearSearch's
    // StableSort over TValueWithWeight (quantile.cpp:90-92).
    let mut elements: Vec<(f32, f64)> = residuals
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, weights.get(i).copied().unwrap_or(1.0)))
        .collect();
    // Use `f32::total_cmp` rather than `partial_cmp(...).unwrap_or(Equal)`: the
    // latter treats a NaN residual as equal to EVERYTHING, collapsing the total
    // order the stable quantile relies on and yielding an arbitrary, unstable
    // rank statistic (WR-06). `total_cmp` is a true total order (NaN sorts to a
    // deterministic end), so a non-finite residual produces a stable, repeatable
    // ordering instead of silent nondeterminism. For all-finite inputs (the
    // tested/oracle regime) `total_cmp` agrees with `partial_cmp`, preserving
    // upstream `StableSort` parity.
    elements.sort_by(|a, b| a.0.total_cmp(&b.0));

    // totalWeight = Accumulate(weights) — ordered f64 sum via the sanctioned
    // primitive (D-08); needWeight = totalWeight * alpha.
    let weight_col: Vec<f64> = elements.iter().map(|&(_, w)| w).collect();
    let total_weight = sum_f64(&weight_col);
    let need_weight = total_weight * alpha;

    // Linear search: first value whose running weight >= needWeight - DBL_EPSILON.
    let mut sum_weight = 0.0_f64;
    // Fallback to the last (largest) value, as CalcSampleQuantileLinearSearch
    // does (`return elements.back().Value`). `elements` is non-empty here.
    let mut quantile = elements.last().map_or(0.0, |&(v, _)| f64::from(v));
    for &(value, weight) in &elements {
        sum_weight += weight;
        if sum_weight >= need_weight - DBL_EPSILON {
            quantile = f64::from(value);
            break;
        }
    }

    // Delta adjustment (CalculateWeightedTargetQuantile, optimal_const_for_loss.h:
    // 82-100). lessWeight/equalWeight are computed against the chosen quantile q.
    if delta > 0.0 {
        let q_f32 = quantile as f32;
        let mut less_members: Vec<f64> = Vec::new();
        let mut equal_members: Vec<f64> = Vec::new();
        for &(value, weight) in &elements {
            if value < q_f32 {
                less_members.push(weight);
            } else if value == q_f32 {
                equal_members.push(weight);
            }
        }
        let less_weight = sum_f64(&less_members);
        let equal_weight = sum_f64(&equal_members);
        if less_weight + equal_weight * alpha >= need_weight - DBL_EPSILON {
            quantile -= delta;
        } else {
            quantile += delta;
        }
    }

    quantile
}

/// The binary-search iteration count and precision for the LogCosh exact optimum
/// (`optimal_const_for_loss.h:122-123` — `BINSEARCH_ITERATIONS = 100`,
/// `APPROX_PRECISION = 1e-9`).
const LOGCOSH_BINSEARCH_ITERATIONS: usize = 100;
const LOGCOSH_APPROX_PRECISION: f64 = 1e-9;

/// The Exact-method leaf delta for **LogCosh** — the 1-D optimum `δ` minimizing
/// `Σ_i w_i · logcosh(δ - r_i)`, found by the monotone-bisection root of its
/// derivative `g(δ) = Σ_i w_i · tanh(δ - r_i)`
/// (`CalcOneDimensionalOptimumConstApprox` -> `CalculateOptimalConstApproxForLogCosh`,
/// `optimal_const_for_loss.h:110-154`).
///
/// `residuals[i]` is member `i`'s `r_i = target_i - approx_i` (widened through
/// `f32` to match upstream's `TVector<float> leafSamples`); `weights[i]` its
/// object weight (an empty `weights` slice means uniform weight `1.0`, matching
/// the `weights.empty()` dispatch). The bracket is `[min(r), max(r)]`
/// (`minmax_element`); each of up to `100` bisection steps evaluates `g` at the
/// midpoint `m` and keeps the half where the sign of `g` flips (`g > 0 ->`
/// right=m, else left=m), returning `left` once the bracket narrows below `1e-9`
/// (or the iteration cap is hit) — transcribed verbatim from upstream, including
/// returning `left` (not the midpoint).
///
/// An empty leaf returns `0.0` (`target.empty()` guard). `g` is monotone
/// increasing in `δ` (`tanh` is), so the bisection is well-defined; the per-step
/// `Σ` is the same order upstream uses (member order), and is a `tanh`-weighted
/// fold — routed through `cb_core::sum_f64` to honor D-08.
#[must_use]
pub fn logcosh_exact_leaf_delta(residuals: &[f32], weights: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }

    let has_weights = !weights.is_empty();
    // g(approx) = Σ_i tanh(approx - r_i) * w_i, member order, ordered f64 fold.
    let g = |approx: f64| -> f64 {
        let terms: Vec<f64> = residuals
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                let w = if has_weights {
                    weights.get(i).copied().unwrap_or(1.0)
                } else {
                    1.0
                };
                (approx - f64::from(r)).tanh() * w
            })
            .collect();
        sum_f64(&terms)
    };

    // Bracket [min(r), max(r)] (minmax_element over the f32 residuals).
    let mut left = f64::INFINITY;
    let mut right = f64::NEG_INFINITY;
    for &r in residuals {
        let v = f64::from(r);
        if v < left {
            left = v;
        }
        if v > right {
            right = v;
        }
    }

    let mut id = 0;
    while id < LOGCOSH_BINSEARCH_ITERATIONS && (right - left) > LOGCOSH_APPROX_PRECISION {
        let m = (left + right) / 2.0;
        if g(m) > 0.0 {
            right = m;
        } else {
            left = m;
        }
        id += 1;
    }

    left
}
