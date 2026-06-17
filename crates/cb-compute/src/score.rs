//! Split-score (gain) computation — the L2 score calcer's `AddLeafPlain` fold
//! over a candidate split's leaf statistics (TRAIN-02, `score_function=L2`).
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/score_calcers.cpp:20-49` + `online_predictor.h`:
//! the L2 score calcer folds each leaf's contribution as
//! `avg * sumWeightedDelta` where `avg = CalcAverage(sumWeightedDelta, sumWeight,
//! L2Regularizer)`. For a candidate split the score is the sum of `AddLeafPlain`
//! over ALL leaves the split produces (for an oblivious tree at depth `d` that is
//! every leaf at the current level, with the candidate applied across the level).
//! A higher score is a better split.
//!
//! `MINIMAL_SCORE = std::numeric_limits<double>::lowest()` is the sentinel a
//! candidate must beat; here it is [`f64::NEG_INFINITY`] (any finite score
//! exceeds it, matching upstream's `bestScore == MINIMAL_SCORE -> stop`).
//!
//! # f64 discipline & summation routing (D-07 / D-08)
//!
//! Leaf stats arrive already reduced (via `cb_core::sum_f64` in `histogram.rs`).
//! The per-leaf `avg * sumWeightedDelta` terms are accumulated through
//! [`cb_core::sum_f64`] so even the score fold honors the single-primitive rule;
//! no raw iterator-sum or zero-seeded float fold is spelled here (D-08).

use cb_core::{std_normal, sum_f64, TFastRng64};

use crate::histogram::LeafStats;
use crate::leaf::calc_average;

/// The minimal score sentinel (`std::numeric_limits<double>::lowest()` analogue):
/// any finite candidate score must strictly exceed it to be selected.
pub const MINIMAL_SCORE: f64 = f64::NEG_INFINITY;

/// One leaf's L2 `AddLeafPlain` contribution: `avg * sum_weighted_delta` with
/// `avg = CalcAverage(sum_weighted_delta, sum_weight, scaled_l2)`.
///
/// `score_calcers.cpp` — the per-leaf term the L2 score calcer accumulates.
#[must_use]
pub fn add_leaf_plain(stats: LeafStats, scaled_l2: f64) -> f64 {
    let avg = calc_average(stats.sum_weighted_delta, stats.sum_weight, scaled_l2);
    avg * stats.sum_weighted_delta
}

/// The total L2 score for a candidate split: the sum of [`add_leaf_plain`] over
/// every leaf the split produces, accumulated in the given leaf order through the
/// sanctioned reduction primitive (D-08). `leaves` MUST be supplied in the
/// canonical leaf-index order so the fold order is reproducible.
#[must_use]
pub fn l2_split_score(leaves: &[LeafStats], scaled_l2: f64) -> f64 {
    let terms: Vec<f64> = leaves
        .iter()
        .map(|&stats| add_leaf_plain(stats, scaled_l2))
        .collect();
    sum_f64(&terms)
}

/// The total **Cosine** score for a candidate split — catboost's DEFAULT
/// `score_function` (`oblivious_tree_options.cpp:22 EScoreFunction::Cosine`).
///
/// `TCosineScoreCalcer` (`score_calcers.h:47-72`) accumulates a `{DP, D2}` pair
/// per split — `DP += leafApprox * SumWeightedDelta`, `D2 += leafApprox² *
/// SumWeight`, seeded `{0, 1e-100}` — and the final score is `DP / sqrt(D2)`. The
/// `leafApprox` is the SAME leaf value the L2 calcer uses
/// (`avg = CalcAverage(SumWeightedDelta, SumWeight, scaled_l2)`), so the numerator
/// `DP` is exactly the [`l2_split_score`] fold; only the `sqrt(D2)` normalization
/// differs. The `1e-100` seed (a) avoids a divide-by-zero on an all-empty split and
/// (b) is far below f64 resolution relative to any real `avg²·SumWeight` term, so it
/// does not perturb the ≤1e-5 parity. A higher score is a better split.
///
/// Like L2, every fold routes through [`sum_f64`] (D-08); the `1e-100` seed is the
/// first summand so the accumulation order mirrors upstream's seeded `Scores[1]`.
#[must_use]
pub fn cosine_split_score(leaves: &[LeafStats], scaled_l2: f64) -> f64 {
    // Numerator DP == the L2 score fold (sum of avg * SumWeightedDelta).
    let numerator = l2_split_score(leaves, scaled_l2);
    // Denominator D2 = 1e-100 (the seed, first summand) + sum(avg² * SumWeight).
    let mut den_terms: Vec<f64> = Vec::with_capacity(leaves.len() + 1);
    den_terms.push(1e-100);
    for &stats in leaves {
        let avg = calc_average(stats.sum_weighted_delta, stats.sum_weight, scaled_l2);
        den_terms.push(avg * avg * stats.sum_weight);
    }
    let denominator = sum_f64(&den_terms);
    numerator / denominator.sqrt()
}

/// The multi-dimension CROSS-DIMENSION split score — a SINGLE shared accumulator
/// fed every dimension's per-leaf bucket stats, with the score transform applied
/// ONCE after the dimension loop (RESEARCH "Multi-dim split-score reduction",
/// transcribed from `scoring.cpp:751-766`/`:1033-1049`, `score_calcers.h:47-97`,
/// `short_vector_ops.h:61-81`). It is NOT a sum of per-dimension scalar scores.
///
/// `per_dim_leaves[d]` is dimension `d`'s per-leaf [`LeafStats`] (in canonical
/// leaf-index order); `scaled_l2` is the per-tree `scale_l2_reg` output. For each
/// `(d, leaf)`: `avrg = SWD/(SW + scaled_l2)`; the shared accumulator does
/// `num += avrg·SWD` and `den += avrg²·SW`. Then ONCE:
/// - **Cosine** (default): `score = num / sqrt(den)` — dimensions COUPLED inside
///   the sqrt (sum numerators, sum denominators, THEN one division + sqrt).
/// - **L2**: `score = num` (linear; still routed through the single accumulator).
///
/// # dim=1 byte-identity (D-04 anchor, Pitfall 1)
/// At `per_dim_leaves.len() == 1` the dimension loop runs exactly once, the
/// `num`/`den` accumulators receive precisely one dimension's contribution through
/// the SAME `cb_core::sum_f64` reduction order as [`l2_split_score`] /
/// [`cosine_split_score`], and `GetScores()` applies the identical transform — so
/// the score is bit-for-bit today's scalar split score. (For L2 the per-dim sum is
/// literally [`l2_split_score`]; for Cosine the num is [`l2_split_score`] and the
/// den is the same seeded `1e-100 + Σ avrg²·SW` fold.)
#[must_use]
pub fn multi_dim_split_score(
    score_function: crate::runtime::EScoreFunction,
    per_dim_leaves: &[Vec<LeafStats>],
    scaled_l2: f64,
) -> f64 {
    use crate::runtime::EScoreFunction;
    // Numerator accumulator: the per-(dim,leaf) `avrg·SWD` terms across ALL dims,
    // folded through the sanctioned ordered primitive (D-08). This is exactly the
    // concatenation of each dimension's `l2_split_score` summands in dimension then
    // leaf order, so at dim=1 it is byte-identical to `l2_split_score`.
    // Pre-size to the total per-(dim,leaf) term count (IN-01) — consistent with the
    // Cosine denominator's `with_capacity` and avoids reallocations as the nested
    // loop pushes. Does not change the fold ORDER (still dimension-then-leaf), so
    // the dim=1 byte-identity (D-04) is preserved.
    let total_terms: usize = per_dim_leaves.iter().map(Vec::len).sum();
    let mut num_terms: Vec<f64> = Vec::with_capacity(total_terms);
    for leaves in per_dim_leaves {
        for &stats in leaves {
            num_terms.push(add_leaf_plain(stats, scaled_l2));
        }
    }
    let numerator = sum_f64(&num_terms);
    match score_function {
        // NewtonL2 (`pointwise_scores.cu:504-510`) reuses the L2 calcer VERBATIM —
        // the second-order distinction is the histogram FILL (summed positive der2 in
        // the `sum_weight` slot, wired in cb-train/histogram), NOT this score formula.
        EScoreFunction::L2 | EScoreFunction::NewtonL2 => numerator,
        // NewtonCosine (`pointwise_scores.cu:512-521`) reuses the Cosine calcer
        // VERBATIM — same der2-fill-vs-formula split as NewtonL2.
        EScoreFunction::Cosine | EScoreFunction::NewtonCosine => {
            // Denominator: the seeded `1e-100` first summand (matching the scalar
            // Cosine seed so dim=1 accumulation order is identical), then the
            // per-(dim,leaf) `avrg²·SW` terms across all dims.
            let mut den_terms: Vec<f64> = Vec::with_capacity(num_terms.len() + 1);
            den_terms.push(1e-100);
            for leaves in per_dim_leaves {
                for &stats in leaves {
                    let avrg = calc_average(stats.sum_weighted_delta, stats.sum_weight, scaled_l2);
                    den_terms.push(avrg * avrg * stats.sum_weight);
                }
            }
            let denominator = sum_f64(&den_terms);
            numerator / denominator.sqrt()
        }
        // SolarL2 (`score_calcers.cuh:22-24`): per-leaf scalar
        // `weight > 1e-20 ? (-sum*sum)*(1 + 2*ln(weight + 1.0))/weight : 0.0`, folded
        // through `sum_f64` in dimension-then-leaf order (mirroring the L2 num fold).
        // IN-04: the GPU Solar/LOO/Sat calcers compute `(-sum*sum)/weight`-style
        // terms directly and do NOT apply the `scaled_l2` regularizer that L2 /
        // Cosine thread in (correct per `score_calcers.cuh` — do not "fix" by adding
        // it). Hence `scaled_l2` is intentionally unused by these three arms.
        EScoreFunction::SolarL2 => {
            let terms = solar_l2_terms(per_dim_leaves);
            sum_f64(&terms)
        }
        // LOOL2 (`score_calcers.cuh:83-87`).
        EScoreFunction::LOOL2 => {
            let terms = loo_l2_terms(per_dim_leaves);
            sum_f64(&terms)
        }
        // SatL2 (`score_calcers.cuh:114-117`).
        EScoreFunction::SatL2 => {
            let terms = sat_l2_terms(per_dim_leaves);
            sum_f64(&terms)
        }
    }
}

/// SolarL2 per-leaf terms across all dimensions (dimension-then-leaf order):
/// `weight > 1e-20 ? (-sum*sum) * (1 + 2*ln(weight + 1.0)) / weight : 0.0`
/// (`score_calcers.cuh:22-24`). Returned as a `Vec` so the caller folds through the
/// single sanctioned reduction primitive (D-08).
fn solar_l2_terms(per_dim_leaves: &[Vec<LeafStats>]) -> Vec<f64> {
    let mut terms: Vec<f64> = Vec::with_capacity(per_dim_leaves.iter().map(Vec::len).sum());
    for leaves in per_dim_leaves {
        for &stats in leaves {
            let sum = stats.sum_weighted_delta;
            let weight = stats.sum_weight;
            let term = if weight > 1e-20 {
                (-sum * sum) * (1.0 + 2.0 * (weight + 1.0).ln()) / weight
            } else {
                0.0
            };
            terms.push(term);
        }
    }
    terms
}

/// LOOL2 per-leaf terms (`score_calcers.cuh:83-87`):
/// `adjust = weight>1.0 ? weight/(weight-1.0) : 0.0; adjust*=adjust;
/// weight>0.0 ? adjust*(-sum*sum)/weight : 0.0`.
fn loo_l2_terms(per_dim_leaves: &[Vec<LeafStats>]) -> Vec<f64> {
    let mut terms: Vec<f64> = Vec::with_capacity(per_dim_leaves.iter().map(Vec::len).sum());
    for leaves in per_dim_leaves {
        for &stats in leaves {
            let sum = stats.sum_weighted_delta;
            let weight = stats.sum_weight;
            let mut adjust = if weight > 1.0 {
                weight / (weight - 1.0)
            } else {
                0.0
            };
            adjust *= adjust;
            let term = if weight > 0.0 {
                adjust * (-sum * sum) / weight
            } else {
                0.0
            };
            terms.push(term);
        }
    }
    terms
}

/// SatL2 per-leaf terms (`score_calcers.cuh:114-117`):
/// `adjust = weight>2.0 ? weight*(weight-2.0)/(weight*weight-3.0*weight+1.0) : 0.0;
/// weight>0.0 ? adjust*(-sum*sum)/weight : 0.0`.
fn sat_l2_terms(per_dim_leaves: &[Vec<LeafStats>]) -> Vec<f64> {
    let mut terms: Vec<f64> = Vec::with_capacity(per_dim_leaves.iter().map(Vec::len).sum());
    for leaves in per_dim_leaves {
        for &stats in leaves {
            let sum = stats.sum_weighted_delta;
            let weight = stats.sum_weight;
            let adjust = if weight > 2.0 {
                weight * (weight - 2.0) / (weight * weight - 3.0 * weight + 1.0)
            } else {
                0.0
            };
            let term = if weight > 0.0 {
                adjust * (-sum * sum) / weight
            } else {
                0.0
            };
            terms.push(term);
        }
    }
    terms
}

// ----------------------------------------------------------------------------
// random_strength split-score perturbation (TRAIN-05).
//
// `random_strength != 0` adds a normal perturbation to every candidate split
// score, drawn from `TFastRng64` via the exact Box-Muller/Marsaglia-polar
// `std_normal`. Parity hinges on (a) the perturbation MAGNITUDE `scoreStDev`
// (`CalcScoreStDev`) and (b) the per-candidate draw ORDER (Pitfall 3 — wired in
// `cb-train::tree`). These helpers own (a) and the single-candidate `GetInstance`.
// ----------------------------------------------------------------------------

/// `CalcDerivativesStDevFromZeroPlainBoosting`
/// (`greedy_tensor_search.cpp:92-107`): the RMS of the (weighted) first
/// derivatives, `sqrt(sum2 / n)` where `sum2 = Σ_d L2NormSquared(der[d])` is
/// summed over the FULL dimension-major buffer (all `approx_dimension * n`
/// elements) and the divisor is the per-OBJECT count `n`
/// (`weightedDerivatives.front().size()`), NOT the buffer length. This is the
/// per-tree scale the `random_strength` perturbation is measured in.
///
/// `weighted_der1` is the dimension-major weighted first-derivative buffer
/// (length `approx_dimension * n`); `n` is the per-object count. At
/// `approx_dimension == 1` the buffer length equals `n` and this reduces to the
/// historical scalar `sqrt(sum(wd_i^2) / n)` (D-04).
///
/// The sum of squares routes through the sanctioned ordered reduction
/// ([`sum_f64`], D-08); `n == 0` returns `0.0` (guarded, no divide-by-zero —
/// the trainer never grows a tree on an empty fold).
#[must_use]
pub fn derivatives_std_dev_from_zero(weighted_der1: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let squares: Vec<f64> = weighted_der1.iter().map(|&d| d * d).collect();
    (sum_f64(&squares) / n as f64).sqrt()
}

/// `CalcDerivativesStDevFromZeroMultiplier` (`greedy_tensor_search.cpp:125-129`):
/// the model-size-decrease multiplier of the default `random_score_type`
/// (`NormalWithModelSizeDecrease`). `modelLength` is `treeIndex * learning_rate`;
/// `modelLeft = exp(ln(n) - modelLength)` and the multiplier is
/// `modelLeft / (1 + modelLeft)`, shrinking the perturbation as the model grows.
#[must_use]
fn model_size_multiplier(n: usize, model_length: f64) -> f64 {
    let model_exp_length = (n as f64).ln();
    let model_left = (model_exp_length - model_length).exp();
    model_left / (1.0 + model_left)
}

/// `CalcScoreStDev` (`greedy_tensor_search.cpp:851-866`): the standard deviation
/// of the `random_strength` split-score perturbation,
/// `random_strength * derivativesStDevFromZero * modelSizeMultiplier` for the
/// default `random_score_type = NormalWithModelSizeDecrease`.
///
/// `weighted_der1` is the dimension-major weighted first-derivative buffer
/// (length `approx_dimension * n`, the same fold the score histogram reduces);
/// `n` is the per-OBJECT count (the `learnSampleCount` upstream feeds BOTH the
/// std-dev divisor AND the `ln(n)` model-size multiplier — never `dim*n`);
/// `model_length = tree_index * learning_rate`. A `random_strength` of `0.0`
/// yields `0.0` (no perturbation — the first-slice behaviour where no normal
/// magnitude is applied).
#[must_use]
pub fn score_st_dev(random_strength: f64, weighted_der1: &[f64], n: usize, model_length: f64) -> f64 {
    let dsdz = derivatives_std_dev_from_zero(weighted_der1, n);
    let mult = model_size_multiplier(n, model_length);
    random_strength * dsdz * mult
}

/// `TRandomScore::GetInstance` for the Normal distribution (`rand_score.h:42-49`):
/// `Val + NormalDistribution<double>(rand, 0, StDev) = Val + std_normal(rand) *
/// StDev`. The normal is ALWAYS drawn (even at `StDev == 0`), so the RNG advances
/// by exactly one `std_normal` per call — the per-candidate draw order the
/// parity contract depends on (Pitfall 3) stays aligned regardless of `StDev`.
#[must_use]
pub fn random_score_instance(val: f64, std_dev: f64, rng: &mut TFastRng64) -> f64 {
    val + std_normal(rng) * std_dev
}
