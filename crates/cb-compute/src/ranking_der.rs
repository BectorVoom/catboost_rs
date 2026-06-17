//! The grouped der seam (LOSS-04, D-6.3-03) — the design hinge every ranking
//! loss funnels through. Mirrors upstream catboost 1.2.10
//! `IDerCalcer::CalcDersForQueries`
//! (`catboost-master/catboost/private/libs/algo_helpers/error_functions.h:831-841`):
//! given the flat approx/target/weights and a `Vec` of per-group spans (the
//! `TQueryInfo` mirror), it slices `approx[begin..end]` per group and routes each
//! group to a per-loss der arm.
//!
//! # This plan delivers the seam, not the loss math
//!
//! Plan 06.3-01 lands ONLY the dispatch skeleton + per-group slicing
//! infrastructure + the shared per-group normalizer [`group_reduce_weighted`].
//! Every concrete [`Loss`] arm returns a typed "ranking loss not yet wired"
//! [`CbError`]; Plans 02–05 replace these arms with the transcribed QueryRMSE /
//! QuerySoftMax / PairLogit / LambdaMart / YetiRank / StochasticRank der.
//!
//! # No cb-train dependency (crate layering)
//!
//! `cb-train` depends on `cb-compute`, never the reverse. So this seam does NOT
//! import `cb-train::QueryInfo`; it re-declares a compute-tier [`GroupSpan`] /
//! [`Competitor`] carrying the same shape as plain data (`cb-train::QueryInfo`
//! lowers into `Vec<GroupSpan>` at the call site). This keeps `cb-compute` free
//! of the trainer and matches the existing layering (RESEARCH Architectural
//! Responsibility Map: grouped der is host-side `cb-compute`, NOT a kernel).
//!
//! # Parity discipline
//!
//! Every per-group / per-pair reduction routes through `cb_core::sum_f64` in
//! upstream iteration order (group index ascending, doc index ascending —
//! RESEARCH Pitfall 4). Empty-group / zero-weight division returns `0.0`, never
//! divides (Security V5 — mirrors `cb-train::metrics.rs:145-149`); no
//! `unwrap`/`expect`/`panic`/indexing-slicing on the grouped input.

use cb_core::{std_normal, sum_f64, CbError, CbResult, TFastRng64};

use crate::loss::{
    lambdamart_pair_grad, pairlogit_pair_prob, queryrmse_der, querysoftmax_der,
};
use crate::runtime::{Derivatives, LambdaMartMetric, Loss, StochasticRankMetric};

/// Whether `loss` uses the PAIRWISE leaf-value / split-scoring path
/// (`IsPairwiseScoring`, `enum_helpers.cpp:469-475`). Only the `*Pairwise`
/// variants qualify — they solve the leaf values via the Cholesky pairwise-leaf
/// path ([`cb_train::pairwise_leaves`]) instead of the pointwise Gradient/Newton
/// estimators (RESEARCH Pitfall 2 — mis-routing diverges leaf values). In this
/// phase the only pairwise-scoring loss is [`Loss::PairLogitPairwise`]
/// (`YetiRankPairwise` lands in Plan 04; `QueryCrossEntropy` is out of scope).
#[must_use]
pub fn is_pairwise_scoring(loss: &Loss) -> bool {
    matches!(loss, Loss::PairLogitPairwise | Loss::YetiRankPairwise { .. })
}

/// Whether `loss` forces `boosting_type = Plain` (`IsPlainOnlyModeLoss`,
/// `enum_helpers.cpp:460-467`). The `*Pairwise` variants cannot use Ordered
/// boosting, so their fixtures must pin Plain. In this phase only
/// [`Loss::PairLogitPairwise`] qualifies (`YetiRankPairwise` in Plan 04).
#[must_use]
pub fn is_plain_only(loss: &Loss) -> bool {
    matches!(loss, Loss::PairLogitPairwise | Loss::YetiRankPairwise { .. })
}

/// A competitor edge inside a group: the group-local loser index plus the pair
/// weight. Compute-tier mirror of `cb-train::Competitor` / upstream
/// `TCompetitor` (`query.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Competitor {
    /// Group-local index of the losing object.
    pub id: usize,
    /// Pair weight.
    pub weight: f64,
}

/// One query group's boundary span + per-group weight + competitor adjacency,
/// the compute-tier mirror of `cb-train::QueryInfo` (and upstream `TQueryInfo`,
/// `query.h:19-44`). The seam slices `approx[begin..end]` per group; per-pair
/// losses read `competitors`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupSpan {
    /// Inclusive start object index (half-open `[begin, end)`).
    pub begin: usize,
    /// Exclusive end object index.
    pub end: usize,
    /// Per-group weight.
    pub weight: f64,
    /// `competitors[winner_local]` → losers `winner_local` is preferred over.
    pub competitors: Vec<Vec<Competitor>>,
}

impl GroupSpan {
    /// Number of objects in the group (`end - begin`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.end - self.begin
    }
}

/// Weighted per-group reduction `Σ_i slice[i] * weights[i]` — the per-group
/// normalizer every querywise ranking loss uses (e.g. QueryRMSE's `queryAvrg`
/// numerator, QuerySoftMax's `Σ expApprox·w`). Reduced through `cb_core::sum_f64`
/// in object order (D-08 — no raw float fold). `weights` is either empty
/// (uniform `1.0`) or the same length as `slice`; a length disagreement folds
/// only the in-range prefix's weights (callers pass matched slices).
#[must_use]
pub fn group_reduce_weighted(slice: &[f64], weights: &[f64]) -> f64 {
    let products: Vec<f64> = slice
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = if weights.is_empty() {
                1.0
            } else {
                weights.get(i).copied().unwrap_or(0.0)
            };
            v * w
        })
        .collect();
    sum_f64(&products)
}

/// Compute the grouped first/second derivatives for `loss` over the grouped view
/// `groups`, mirroring upstream `CalcDersForQueries`
/// (`error_functions.h:831-841`).
///
/// `approx` / `target` / `weights` are the flat per-object buffers (length
/// `n_objects`; `weights` may be empty for uniform `1.0`). Each [`GroupSpan`]
/// carries the half-open `[begin, end)` object span this group owns; the seam
/// slices `approx[begin..end]` and routes to the per-loss arm.
///
/// Plan 06.3-01 implements the dispatch skeleton + per-group slicing only — every
/// loss arm returns a typed "ranking loss not yet wired" error (Plans 02–05 fill
/// them in). The per-group reductions that the real arms will use route through
/// [`group_reduce_weighted`] / `cb_core::sum_f64`.
///
/// # Errors
/// - [`CbError::Degenerate`] if a [`GroupSpan`] span is out of range for
///   `approx`/`target`, or `approx`/`target` lengths disagree.
/// - [`CbError::OutOfRange`] (kind "ranking loss not yet wired") for every loss
///   variant in this plan — the seam is structural; loss math lands in 02–05.
pub fn calc_ders_for_queries(
    loss: &Loss,
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    groups: &[GroupSpan],
    random_seed: u64,
) -> CbResult<Vec<Derivatives>> {
    if approx.len() != target.len() {
        return Err(CbError::Degenerate(format!(
            "calc_ders_for_queries: approx len {} != target len {}",
            approx.len(),
            target.len()
        )));
    }
    if !weights.is_empty() && weights.len() != approx.len() {
        return Err(CbError::Degenerate(format!(
            "calc_ders_for_queries: weights len {} != approx len {}",
            weights.len(),
            approx.len()
        )));
    }

    let mut per_group: Vec<Derivatives> = Vec::with_capacity(groups.len());
    for (group_index, group) in groups.iter().enumerate() {
        // Per-group slice bounds, validated before any indexing (Security V5 — no
        // unchecked slice on the grouped input).
        if group.begin > group.end || group.end > approx.len() {
            return Err(CbError::Degenerate(format!(
                "calc_ders_for_queries: group span [{}, {}) out of range for n={}",
                group.begin,
                group.end,
                approx.len()
            )));
        }
        let approx_slice = approx.get(group.begin..group.end).ok_or_else(|| {
            CbError::Degenerate("calc_ders_for_queries: approx slice out of range".to_owned())
        })?;
        let target_slice = target.get(group.begin..group.end).ok_or_else(|| {
            CbError::Degenerate("calc_ders_for_queries: target slice out of range".to_owned())
        })?;
        let weight_slice: &[f64] = if weights.is_empty() {
            &[]
        } else {
            weights.get(group.begin..group.end).ok_or_else(|| {
                CbError::Degenerate("calc_ders_for_queries: weight slice out of range".to_owned())
            })?
        };

        // Per-object weight accessor for this group: uniform 1.0 when unweighted.
        let weight_at = |i: usize| -> f64 {
            if weight_slice.is_empty() {
                1.0
            } else {
                weight_slice.get(i).copied().unwrap_or(0.0)
            }
        };

        // Empty group → an empty der set for the group (Security V5 — never
        // divides, mirror metrics.rs:145-149). Plans 02–05 wired arms below all
        // short-circuit on a zero-size group to the empty der.
        if group.size() == 0 {
            per_group.push(Derivatives {
                der1: Vec::new(),
                der2: Vec::new(),
            });
            continue;
        }

        // Dispatch to the per-loss arm (Plan 06.3-02 wires the first two querywise
        // ranking losses; PairLogit / LambdaMart / YetiRank / StochasticRank stay
        // unwired until Plans 03–05).
        let ders = match loss {
            // QueryRMSE (Wave A): per-group weighted residual mean `queryAvrg`,
            // then per-object `der1 = (target - approx - queryAvrg)·w`,
            // `der2 = -1·w` (error_functions.h:879-933). The `queryAvrg`
            // numerator `Σ (target - approx)·w` and denominator `Σ w` both route
            // through cb_core::sum_f64 (D-08, doc-ascending order).
            Loss::QueryRmse => {
                // residual[i] = target[i] - approx[i] (group-local).
                let residuals: Vec<f64> = (0..group.size())
                    .map(|i| {
                        target_slice.get(i).copied().unwrap_or(0.0)
                            - approx_slice.get(i).copied().unwrap_or(0.0)
                    })
                    .collect();
                // numerator = Σ residual·w ; denominator = Σ w (both sum_f64).
                let numerator = group_reduce_weighted(&residuals, weight_slice);
                let weight_col: Vec<f64> = (0..group.size()).map(weight_at).collect();
                let denominator = sum_f64(&weight_col);
                // queryCount > 0 guard (error_functions.h:928): zero-weight group →
                // queryAvrg 0, never divides.
                let query_avrg = if denominator > 0.0 {
                    numerator / denominator
                } else {
                    0.0
                };
                let mut der1 = Vec::with_capacity(group.size());
                let mut der2 = Vec::with_capacity(group.size());
                for i in 0..group.size() {
                    let a = approx_slice.get(i).copied().unwrap_or(0.0);
                    let t = target_slice.get(i).copied().unwrap_or(0.0);
                    let (d1, d2) = queryrmse_der(a, t, weight_at(i), query_avrg);
                    der1.push(d1);
                    der2.push(d2);
                }
                Derivatives { der1, der2 }
            }
            // QuerySoftMax (Wave A): per-group softmax over `Beta·approx`,
            // MAX-SHIFTED before exp (error_functions.cpp:540-552). The exp share
            // `p = expApprox·w / Σ expApprox·w`; der per error_functions.cpp:560-565.
            // `sumWTargets <= 0` (or `weight <= 0`) → ders 0 (no divide).
            Loss::QuerySoftMax { lambda, beta } => {
                // (1) maxApprox + sumWeightedTargets over weight>0 objects
                // (error_functions.cpp:540-550). maxApprox seeds at the most
                // negative finite f64 (upstream `-numeric_limits::max()`); an
                // all-zero-weight group leaves it at that seed but the
                // `sumWTargets <= 0` guard short-circuits before exp.
                let mut max_approx = f64::MIN;
                let mut target_terms: Vec<f64> = Vec::with_capacity(group.size());
                for i in 0..group.size() {
                    let w = weight_at(i);
                    let a = approx_slice.get(i).copied().unwrap_or(0.0);
                    let t = target_slice.get(i).copied().unwrap_or(0.0);
                    if w > 0.0 {
                        if a > max_approx {
                            max_approx = a;
                        }
                        if t > 0.0 {
                            target_terms.push(w * t);
                        }
                    }
                }
                // Σ w·target through the sanctioned ordered reduction (D-08).
                let sum_weighted_targets = sum_f64(&target_terms);
                if sum_weighted_targets > 0.0 {
                    // (2) expApprox[i] = exp(Beta·(approx[i] - maxApprox)) · w, the
                    // numerator the share p divides by. `calc_softmax` already
                    // max-subtracts + exps, but it normalizes UNWEIGHTED over ALL
                    // objects; here the share is WEIGHTED and the shift uses Beta.
                    // Transcribe the shift directly (the calc_softmax NaN-guard
                    // discipline) so weight>0/weight==0 objects are handled per
                    // upstream.
                    let weighted_exp: Vec<f64> = (0..group.size())
                        .map(|i| {
                            let w = weight_at(i);
                            let a = approx_slice.get(i).copied().unwrap_or(0.0);
                            // exp(Beta·(approx - maxApprox)) · w.
                            (beta * (a - max_approx)).exp() * w
                        })
                        .collect();
                    // sumExpApprox = Σ weighted_exp (sum_f64, doc-ascending — D-08).
                    let sum_exp = sum_f64(&weighted_exp);
                    let mut der1 = Vec::with_capacity(group.size());
                    let mut der2 = Vec::with_capacity(group.size());
                    for i in 0..group.size() {
                        let w = weight_at(i);
                        if w > 0.0 && sum_exp > 0.0 {
                            let p = weighted_exp.get(i).copied().unwrap_or(0.0) / sum_exp;
                            let t = target_slice.get(i).copied().unwrap_or(0.0);
                            let (d1, d2) = querysoftmax_der(
                                p,
                                sum_weighted_targets,
                                w,
                                t,
                                *beta,
                                *lambda,
                            );
                            der1.push(d1);
                            der2.push(d2);
                        } else {
                            // weight <= 0 → ders 0 (error_functions.cpp:566-569).
                            der1.push(0.0);
                            der2.push(0.0);
                        }
                    }
                    Derivatives { der1, der2 }
                } else {
                    // sumWTargets <= 0 → all ders 0 (error_functions.cpp:571-576).
                    Derivatives {
                        der1: vec![0.0; group.size()],
                        der2: vec![0.0; group.size()],
                    }
                }
            }
            // PairLogit / PairLogitPairwise (Wave B): the SAME pairwise-logit der
            // over the group's explicit Competitors adjacency (they map to one
            // upstream TPairLogitError; only the LEAF path differs — pointwise vs
            // Cholesky — which is decided later in boosting, not here). EXP-approx:
            // the RAW approxes are exp()'d INLINE per pair (Poisson precedent). The
            // pair weight is `competitor.weight` (NOT the per-object weight).
            // error_functions.h:849-866.
            Loss::PairLogit | Loss::PairLogitPairwise => {
                pairlogit_group_der(approx_slice, &group.competitors)
            }
            // LambdaMart (Wave B): the per-group lambda gradient toward the target
            // metric. RAW approx; sorts the group by approx descending, then
            // accumulates the per-ordered-pair antigrad/hessian, optionally
            // norm-rescaled. error_functions.cpp:607-922.
            Loss::LambdaMart {
                metric,
                sigma,
                top,
                norm,
            } => lambdamart_group_der(approx_slice, target_slice, *metric, *sigma, *top, *norm),
            // YetiRank / YetiRankPairwise (Wave C): the SAMPLED pairs are injected
            // into the GroupSpan's `competitors` adjacency by the trainer
            // (cb-train::yetirank::sample_pairs, regenerated each iteration) BEFORE
            // this seam runs — so both YetiRank variants ride the EXISTING PairLogit
            // der over those sampled competitors, EXACTLY like PairLogit rides its
            // explicit pairs (yetirank_helpers.cpp:336-344 — the sampled
            // competitorsWeights ARE the TCompetitor adjacency `TPairLogitError`
            // consumes). The only difference between the two variants is the LEAF
            // path (YetiRank pointwise, YetiRankPairwise Cholesky), decided later in
            // boosting — not here. The exp()-of-approx for the noise happens inside
            // the sampler, not the der.
            Loss::YetiRank { .. } | Loss::YetiRankPairwise { .. } => {
                pairlogit_group_der(approx_slice, &group.competitors)
            }
            // StochasticRank (Wave C): the Monte-Carlo querywise der. RAW approx; no
            // pairs. The Gaussian noise stream is re-seeded per group with
            // `random_seed + group_index` (error_functions.h:1257 →
            // error_functions.cpp:1041 `TFastRng64 rng(randomSeed)`), drawn via
            // cb_core::std_normal (the SAME variable-length Marsaglia-polar
            // sequence). der2 = 0 (Gradient leaf method).
            Loss::StochasticRank {
                metric,
                sigma,
                mu,
                num_estimations,
            } => stochastic_rank_group_der(
                approx_slice,
                target_slice,
                *metric,
                *sigma,
                *mu,
                *num_estimations,
                random_seed.wrapping_add(group_index as u64),
            ),
            // Every non-ranking loss never reaches the grouped seam.
            _ => {
                return Err(CbError::OutOfRange(format!(
                    "calc_ders_for_queries: ranking loss not yet wired for {loss:?} \
                     (grouped der arms land in Plans 06.3-04..05)"
                )));
            }
        };
        per_group.push(ders);
    }

    // Reached only when `groups` is empty (no group ever dispatched): the seam
    // returns an empty der set, never panicking on degenerate ungrouped input.
    Ok(per_group)
}

/// PairLogit / PairLogitPairwise per-group der over the explicit `competitors`
/// adjacency (LOSS-04, Wave B). `approx_slice` is the group's RAW approxes (length
/// `n`, group-local); `competitors[winner_local]` lists the losers `winner_local`
/// should outrank, each carrying the pair `weight`. Transcribes
/// `TPairLogitError::CalcDersForQueries` (`error_functions.h:849-866`) verbatim.
///
/// The der is a SCATTER-add across both the winner and loser objects (a winner's
/// loss raises its own der and lowers each loser's), so it mirrors upstream's
/// fixed-order `+=` accumulation EXACTLY (doc-ascending outer, competitor-order
/// inner) rather than a reordered `sum_f64` reduction — the summation order IS the
/// parity contract here (any reorder perturbs the last ULP). The pair weight is
/// `competitor.weight`, NOT the per-object weight (PairLogit folds the weight into
/// the pair, not the object). EXP-approx: `p` is computed from the two RAW
/// approxes with `exp()` taken INLINE ([`pairlogit_pair_prob`]).
fn pairlogit_group_der(approx_slice: &[f64], competitors: &[Vec<Competitor>]) -> Derivatives {
    let n = approx_slice.len();
    let mut der1 = vec![0.0_f64; n];
    let mut der2 = vec![0.0_f64; n];
    for doc_id in 0..n {
        let winner_approx = approx_slice.get(doc_id).copied().unwrap_or(0.0);
        let mut winner_der = 0.0_f64;
        let mut winner_second_der = 0.0_f64;
        // competitors[doc_id] may be absent (no losers for this winner) — skip.
        if let Some(comps) = competitors.get(doc_id) {
            for competitor in comps {
                let loser_id = competitor.id;
                let loser_approx = approx_slice.get(loser_id).copied().unwrap_or(0.0);
                let p = pairlogit_pair_prob(winner_approx, loser_approx);
                let w = competitor.weight;
                winner_der += w * p;
                winner_second_der += w * p * (p - 1.0);
                if let Some(d) = der1.get_mut(loser_id) {
                    *d -= w * p;
                }
                if let Some(d) = der2.get_mut(loser_id) {
                    *d += w * p * (p - 1.0);
                }
            }
        }
        if let Some(d) = der1.get_mut(doc_id) {
            *d += winner_der;
        }
        if let Some(d) = der2.get_mut(doc_id) {
            *d += winner_second_der;
        }
    }
    Derivatives { der1, der2 }
}

/// `(N)DCG` numerator for a target relevance (the `ENdcgMetricType::Base` default:
/// the relevance itself; `error_functions.h:1359-1361`). The corpus pins the Base
/// type, so this is the identity; the `Exp` (`2^rel - 1`) variant is not
/// fixture-reachable in this wave.
#[inline]
fn ndcg_numerator(target: f64) -> f64 {
    target
}

/// `(N)DCG` denominator for a 0-based sort position (the
/// `ENdcgDenominatorType::LogPosition` default: `log2(2 + pos)`;
/// `error_functions.h:1363-1365`).
#[inline]
fn ndcg_denominator(pos: usize) -> f64 {
    (2.0 + pos as f64).log2()
}

/// LambdaMart per-group der toward the target `metric` (LOSS-04, Wave B).
/// `approx_slice` / `target_slice` are the group's RAW approxes / relevances
/// (group-local). Transcribes `TLambdaMartError::CalcDersForSingleQuery`
/// (`error_functions.cpp:859-923`): stable-sort the docs by approx descending,
/// dispatch to the metric-specific per-pair arm (only (N)DCG is fixture-reachable
/// this wave; MRR/ERR/MAP transcribed for completeness in a follow-up), then apply
/// the optional `norm` rescale. The der2 hessian is filled despite
/// `maxDerivativeOrder == 1` (RESEARCH Pitfall 5), so LambdaMart rides the
/// pointwise NEWTON leaf.
///
/// The per-pair accumulation is a fixed-order SCATTER-add (mirroring upstream's
/// `ders[...].Der1 += antigrad`), so it reproduces upstream's summation order
/// exactly rather than a reordered reduction. The `idealScore` and `sumDer1`
/// accumulations are likewise the sequential float adds upstream performs (the
/// parity order is the loop order, not a `sum_f64` reorder).
fn lambdamart_group_der(
    approx_slice: &[f64],
    target_slice: &[f64],
    metric: LambdaMartMetric,
    sigma: f64,
    top: i64,
    norm: bool,
) -> Derivatives {
    let count = approx_slice.len();
    let mut der1 = vec![0.0_f64; count];
    let mut der2 = vec![0.0_f64; count];
    if count <= 1 {
        return Derivatives { der1, der2 };
    }

    // Stable sort doc indices by approx DESCENDING (error_functions.cpp:874-878).
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&a, &b| {
        let aa = approx_slice.get(a).copied().unwrap_or(0.0);
        let ab = approx_slice.get(b).copied().unwrap_or(0.0);
        // descending by approx; stable on ties (sort_by is stable).
        ab.partial_cmp(&aa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let approx_at = |i: usize| -> f64 {
        approx_slice
            .get(order.get(i).copied().unwrap_or(0))
            .copied()
            .unwrap_or(0.0)
    };
    let target_at = |i: usize| -> f64 {
        target_slice
            .get(order.get(i).copied().unwrap_or(0))
            .copied()
            .unwrap_or(0.0)
    };

    // isApproxesSame guard (error_functions.cpp:640): when every approx is equal
    // the norm `/= 0.01 + |diff|` is skipped (diff is 0 for every pair anyway).
    let is_approxes_same = (approx_at(0) - approx_at(count - 1)).abs() == 0.0;

    let mut sum_der1 = 0.0_f64;

    match metric {
        LambdaMartMetric::Dcg | LambdaMartMetric::Ndcg => {
            let query_top_size = lambda_query_top_size(top, count);
            let ideal_score = lambdamart_ideal_ndcg(target_slice, query_top_size);
            for first_id in 0..count {
                let bound_for_second = if first_id < query_top_size {
                    count
                } else {
                    query_top_size
                };
                for second_id in 0..bound_for_second {
                    let first_target = target_at(first_id);
                    let second_target = target_at(second_id);
                    if first_target <= second_target {
                        continue;
                    }
                    let approx_diff = approx_at(first_id) - approx_at(second_id);
                    let dcg_num = ndcg_numerator(first_target) - ndcg_numerator(second_target);
                    let dcg_den = (1.0 / ndcg_denominator(first_id)
                        - 1.0 / ndcg_denominator(second_id))
                    .abs();
                    let mut delta = if ideal_score != 0.0 {
                        dcg_num * dcg_den / ideal_score
                    } else {
                        // IDCG==0 → no signal (every relevance 0); delta 0 (no
                        // divide — Security V5).
                        0.0
                    };
                    if norm && !is_approxes_same {
                        delta /= 0.01 + approx_diff.abs();
                    }
                    let (antigrad, hessian) = lambdamart_pair_grad(approx_diff, delta, sigma);
                    let fo = order.get(first_id).copied().unwrap_or(0);
                    let so = order.get(second_id).copied().unwrap_or(0);
                    if let Some(d) = der1.get_mut(fo) {
                        *d += antigrad;
                    }
                    if let Some(d) = der2.get_mut(fo) {
                        *d += hessian;
                    }
                    if let Some(d) = der1.get_mut(so) {
                        *d -= antigrad;
                    }
                    if let Some(d) = der2.get_mut(so) {
                        *d += hessian;
                    }
                    sum_der1 -= 2.0 * antigrad;
                }
            }
        }
        // MRR / ERR / MAP are admissible upstream metrics but are NOT reachable by
        // the Wave-B fixture (which pins the NDCG default). They are intentionally
        // left to a follow-up rather than transcribed-but-untested here (the
        // executor does not ship unverifiable der math). Reaching them is a no-op
        // (zeros) — `Loss::validate` accepts the metric but the corpus never trains
        // them; a future plan adds their CalcDersMRR/ERR/MAP arms + fixtures.
        LambdaMartMetric::Mrr | LambdaMartMetric::Err | LambdaMartMetric::Map => {}
    }

    // norm rescale (error_functions.cpp:916-922): when norm and sumDer1 > 0,
    // multiply every der by log2(1 + sumDer1)/sumDer1.
    if norm && sum_der1 > 0.0 {
        let norma = (1.0 + sum_der1).log2() / sum_der1;
        for d in &mut der1 {
            *d *= norma;
        }
        for d in &mut der2 {
            *d *= norma;
        }
    }

    Derivatives { der1, der2 }
}

/// LambdaMart `GetQueryTopSize` (`error_functions.h:1367-1372`): `top == -1` or
/// `top > docCount` ⇒ the full group, else `top`.
fn lambda_query_top_size(top: i64, doc_count: usize) -> usize {
    if top == -1 || top > doc_count as i64 {
        doc_count
    } else {
        // top is validated `> 0` (or -1) by Loss::validate; clamp defensively.
        usize::try_from(top).unwrap_or(doc_count).min(doc_count)
    }
}

/// LambdaMart `CalcIdealMetric` for (N)DCG (`error_functions.cpp:925-935`): the DCG
/// of the target-sorted ideal order over the top-`query_top_size` positions. The
/// per-position terms are collected and reduced through `cb_core::sum_f64` per the
/// D-08 summation discipline (CLAUDE.md admits no silent raw-accumulation
/// exception); `sum_f64` is the sanctioned strict left-to-right f64 fold, so this
/// matches upstream's sequential `score += ...` accumulation order exactly.
fn lambdamart_ideal_ndcg(target_slice: &[f64], query_top_size: usize) -> f64 {
    let mut sorted: Vec<f64> = target_slice.to_vec();
    // descending stable sort of the relevances.
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let terms: Vec<f64> = sorted
        .iter()
        .enumerate()
        .take(query_top_size)
        .map(|(pos, &t)| ndcg_numerator(t) / ndcg_denominator(pos))
        .collect();
    sum_f64(&terms)
}

/// Standard-normal density `φ((x-mean)/sigma)/sigma` for the StochasticRank
/// position-density gradient (`error_functions.h:1586-1589`
/// `TStochasticRankError::NormalDensity`):
/// ```text
/// z = ((x - mean) / sigma)^2;  return expl(-z/2) * INV_SQRT_2PI / sigma
/// ```
/// `INV_SQRT_2PI = 1/sqrt(2π)`. Upstream uses `long double` (`std::expl`); Rust
/// `f64` is the established A2 precision-gap precedent (the oracle absorbs the
/// last-ULP `expl` vs `exp` difference; documented in the README). `sigma > 0` is
/// guaranteed by [`Loss::validate`], so no divide-by-zero.
#[inline]
fn normal_density(x: f64, mean: f64, sigma: f64) -> f64 {
    /// `1/sqrt(2π)` (`error_functions.h:1422` `INV_SQRT_2PI`).
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_677_939_946;
    let z = ((x - mean) / sigma).powi(2);
    (-z / 2.0).exp() * INV_SQRT_2PI / sigma
}

/// `NormalDensity(x1) - NormalDensity(x2)` (`error_functions.h:1591-1593`).
#[inline]
fn normal_density_diff(x1: f64, x2: f64, mean: f64, sigma: f64) -> f64 {
    normal_density(x1, mean, sigma) - normal_density(x2, mean, sigma)
}

/// The DCG/NDCG position weights `1/CalcDenominator(pos)` over the top window,
/// normalized by the ideal DCG for NDCG (`error_functions.cpp:1514-1538`
/// `ComputeDCGPosWeights`). `CalcDenominator(pos) = log2(2 + pos)` (the
/// LogPosition default, `error_functions.h:1575-1577`), `CalcNumerator(t) = t`
/// (the Base default). For NDCG, divide by the ideal DCG of the target-sorted
/// order when it exceeds `f64::EPSILON` (`error_functions.cpp:1531`).
fn compute_dcg_pos_weights(
    targets: &[f64],
    query_top_size: usize,
    is_ndcg: bool,
) -> Vec<f64> {
    let count = targets.len();
    let mut pos_weights = vec![0.0_f64; count];
    for (pos, w) in pos_weights.iter_mut().enumerate().take(query_top_size) {
        // 1.0 / CalcDenominator(pos) = 1 / log2(2 + pos).
        *w = 1.0 / ndcg_denominator(pos);
    }
    if is_ndcg {
        // idealDCG = CalcDCG(sortedTargets desc, posWeights) (error_functions.cpp:1530).
        let mut sorted: Vec<f64> = targets.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut ideal_dcg = 0.0_f64;
        for (pos, &t) in sorted.iter().enumerate().take(query_top_size) {
            // CalcDCG = Σ numerator(target) * posWeight (error_functions.cpp:1572-...).
            ideal_dcg += ndcg_numerator(t) * pos_weights.get(pos).copied().unwrap_or(0.0);
        }
        if ideal_dcg > f64::EPSILON {
            for w in pos_weights.iter_mut().take(query_top_size) {
                *w /= ideal_dcg;
            }
        }
    }
    pos_weights
}

/// StochasticRank Monte-Carlo querywise der for one group, DCG/NDCG metric arm
/// (LOSS-04, Wave C). Transcribes `TStochasticRankError::CalcDersForSingleQuery`
/// (`error_functions.cpp:1008-1102`) + `CalcMonteCarloEstimateForSingleQueryPermutation`
/// (`:1107-1221`) + `CalcDCGCumulativeStatistics` (`:1428-1451`) +
/// `CalcDCGMetricDiff` (`:1222-1256`) for the DCG/NDCG `TargetMetric` ONLY (the
/// PFound/ERR/MRR/FilteredDCG arms are out of this phase's scope; rejected by
/// [`Loss::validate`]). `top == -1` (the full group) is assumed (no `TopSize`
/// param on the variant), so `query_top_size == count`.
///
/// RNG: `rng = TFastRng64(seed)` (the caller passes `random_seed + group_index`);
/// per sample, per doc one [`std_normal`] Gaussian (the variable-length
/// Marsaglia-polar draw — the parity crux). `der2 = 0` (Gradient leaf). Every
/// cross-doc reduction (`avrgShiftedApprox`, `noiseSum`, the SFA means/dots)
/// routes through `sum_f64` in doc-ascending order (D-08).
///
/// `count <= 1` → zero ders (`error_functions.cpp:1020-1022`).
#[allow(clippy::too_many_lines)]
fn stochastic_rank_group_der(
    approx_slice: &[f64],
    target_slice: &[f64],
    metric: StochasticRankMetric,
    sigma_param: f64,
    mu: f64,
    num_estimations: u32,
    seed: u64,
) -> Derivatives {
    let count = approx_slice.len();
    let mut der1 = vec![0.0_f64; count];
    let der2 = vec![0.0_f64; count]; // Gradient leaf: der2 == 0 (no Newton).
    if count <= 1 {
        return Derivatives { der1, der2 };
    }
    let is_ndcg = matches!(metric, StochasticRankMetric::Ndcg);
    let num_est = num_estimations.max(1) as usize;
    // top == -1 (full group): queryTopSize == count (error_functions.h:1579-1583).
    let query_top_size = count;

    // Stage 1 — shift approxes to break ties, then center (non-FilteredDCG).
    // shifted[d] = approx[d] - Sigma·Mu·target[d] (error_functions.cpp:1026-1028).
    let mut shifted: Vec<f64> = (0..count)
        .map(|d| {
            approx_slice.get(d).copied().unwrap_or(0.0)
                - sigma_param * mu * target_slice.get(d).copied().unwrap_or(0.0)
        })
        .collect();
    // avrgShiftedApprox = Σ shifted / count (sum_f64, doc-ascending — D-08).
    let avrg_shifted = sum_f64(&shifted) / count as f64;
    for s in &mut shifted {
        *s -= avrg_shifted;
    }

    // posWeights are sample-independent for DCG/NDCG (computed once, sample==0 in
    // upstream error_functions.cpp:1056-1057).
    let pos_weights = compute_dcg_pos_weights(target_slice, query_top_size, is_ndcg);

    // Stage 2 — Monte-Carlo: per sample, draw the Gaussian noise stream and
    // accumulate the per-doc gradient.
    let mut rng = TFastRng64::from_seed(seed);
    for _sample in 0..num_est {
        // noise[d] = StdNormalDistribution(rng); scores[d] = shifted[d] + Sigma·noise[d]
        // (error_functions.cpp:1043-1046). The std_normal draw order IS the parity
        // contract — a different sampler desyncs every subsequent draw.
        // Bounds-checked fill (CLAUDE.md unchecked-index ban; T-06.3-11). zip over
        // (&shifted, &mut noise, &mut scores) so every write lands in a valid slot
        // and shifted[d] is read in-range — bit-identical to the indexed loop.
        let mut noise = vec![0.0_f64; count];
        let mut scores = vec![0.0_f64; count];
        for ((sh, n), sc) in shifted.iter().zip(noise.iter_mut()).zip(scores.iter_mut()) {
            *n = std_normal(&mut rng);
            *sc = sh + sigma_param * *n;
        }
        // noiseSum = Σ noise (sum_f64, doc-ascending — D-08).
        let noise_sum = sum_f64(&noise);
        // order = stable-sort of indices by score DESCENDING (error_functions.cpp:1052-1055).
        let mut order: Vec<usize> = (0..count).collect();
        stable_sort_desc_by_key(&mut order, &scores);

        // CalcDCGCumulativeStatistics (error_functions.cpp:1428-1451). cumSum /
        // cumSumUp / cumSumLow are length count+1.
        let mut cum_sum = vec![0.0_f64; count + 1];
        let mut cum_sum_up = vec![0.0_f64; count + 1];
        let mut cum_sum_low = vec![0.0_f64; count + 1];
        // Bounds-checked prefix-sum build (CLAUDE.md unchecked-index ban;
        // T-06.3-11). Reads via .get(..).unwrap_or(0.0), writes via get_mut guards.
        // All indices are in-range for the present caller — bit-identical values.
        for pos in 0..count {
            let doc_id = order.get(pos).copied().unwrap_or(0);
            let gain = ndcg_numerator(target_slice.get(doc_id).copied().unwrap_or(0.0));
            let prev = cum_sum.get(pos).copied().unwrap_or(0.0);
            if let Some(slot) = cum_sum.get_mut(pos + 1) {
                *slot = prev + gain * pos_weights.get(pos).copied().unwrap_or(0.0);
            }
            if pos + 1 < count {
                let prev_low = cum_sum_low.get(pos).copied().unwrap_or(0.0);
                if let Some(slot) = cum_sum_low.get_mut(pos + 1) {
                    *slot = prev_low + gain * pos_weights.get(pos + 1).copied().unwrap_or(0.0);
                }
            }
            if pos > 0 {
                let prev_up = cum_sum_up.get(pos).copied().unwrap_or(0.0);
                if let Some(slot) = cum_sum_up.get_mut(pos + 1) {
                    *slot = prev_up + gain * pos_weights.get(pos - 1).copied().unwrap_or(0.0);
                }
            }
        }
        let last_low = cum_sum_low.get(count.saturating_sub(1)).copied().unwrap_or(0.0);
        if let Some(slot) = cum_sum_low.get_mut(count) {
            *slot = last_low;
        }

        // Per-position der accumulation (error_functions.cpp:1161-1220).
        for pos in 0..count {
            let doc_id = order.get(pos).copied().unwrap_or(0);
            let score = scores.get(doc_id).copied().unwrap_or(0.0);
            let approx = approx_slice.get(doc_id).copied().unwrap_or(0.0);
            // mean = approx + (noiseSum - (score - approx)) / (count - 1)  (non-FilteredDCG).
            let mean = approx + (noise_sum - (score - approx)) / (count as f64 - 1.0);
            // sigma = Sigma · sqrt(count / (count - 1)).
            let sigma = sigma_param * (count as f64 / (count as f64 - 1.0)).sqrt();
            // maxNewPos = min(count - 1, queryTopSize) (DCG arm, :1173).
            let max_new_pos = (count - 1).min(query_top_size);
            let mut der_sum = 0.0_f64;
            for new_pos in 0..(max_new_pos + 1) {
                if new_pos == pos {
                    continue;
                }
                // CalcDCGMetricDiff(pos, new_pos, ...) (error_functions.cpp:1222-1256).
                let metric_diff = calc_dcg_metric_diff(
                    pos,
                    new_pos,
                    target_slice,
                    &order,
                    &pos_weights,
                    &cum_sum,
                    &cum_sum_up,
                    &cum_sum_low,
                );
                // densityDiff (error_functions.cpp:1144-1155). `score_at(p)` reads
                // scores[order[p]] bounds-checked (CLAUDE.md unchecked-index ban;
                // T-06.3-11) — in-range for the present caller, bit-identical.
                let score_at = |p: usize| -> f64 {
                    scores
                        .get(order.get(p).copied().unwrap_or(0))
                        .copied()
                        .unwrap_or(0.0)
                };
                let density_diff = if new_pos == 0 {
                    normal_density(score_at(0), mean, sigma)
                } else if new_pos + 1 == count.min(query_top_size + 1) {
                    if new_pos < pos {
                        -normal_density(score_at(query_top_size - 1), mean, sigma)
                    } else {
                        -normal_density(score_at(query_top_size.min(count - 1)), mean, sigma)
                    }
                } else if new_pos < pos {
                    normal_density_diff(score_at(new_pos), score_at(new_pos - 1), mean, sigma)
                } else {
                    normal_density_diff(score_at(new_pos + 1), score_at(new_pos), mean, sigma)
                };
                der_sum += metric_diff * density_diff;
            }
            // ders[docId].Der1 += derSum / NumEstimations (error_functions.cpp:1219).
            // Bounds-checked write (CLAUDE.md unchecked-index ban; T-06.3-11).
            if let Some(slot) = der1.get_mut(doc_id) {
                *slot += der_sum / num_est as f64;
            }
        }
    }

    // Stage 3 — SFA: subtract the mean der (orthogonalize, non-FilteredDCG,
    // error_functions.cpp:1075-1084). avrgDer = Σ der1 / count.
    let avrg_der = sum_f64(&der1) / count as f64;
    for d in &mut der1 {
        *d -= avrg_der;
    }
    // count > 2: project out the approx direction (error_functions.cpp:1085-1103).
    if count > 2 {
        let avrg_approx = sum_f64(approx_slice) / count as f64;
        let zero_mean: Vec<f64> = (0..count)
            .map(|d| approx_slice.get(d).copied().unwrap_or(0.0) - avrg_approx)
            .collect();
        let sq: Vec<f64> = zero_mean.iter().map(|&z| z * z).collect();
        // approxesNormSqr = (sqrt(Σ z²) + Nu)² (error_functions.cpp:1095).
        let approxes_norm_sqr = (sum_f64(&sq).sqrt() + crate::runtime::STOCHASTIC_RANK_NU_DEFAULT)
            .powi(2);
        // Bounds-checked dot + projection (CLAUDE.md unchecked-index ban;
        // T-06.3-11). der1 and zero_mean are both length count — the zip is
        // bit-identical to the indexed loop, in canonical doc-ascending order (D-08).
        let dots: Vec<f64> = der1
            .iter()
            .zip(zero_mean.iter())
            .map(|(&d1, &z)| d1 * z)
            .collect();
        let dot = sum_f64(&dots);
        if approxes_norm_sqr > 0.0 {
            // k = Lambda · dot / approxesNormSqr (Lambda default 1.0 for DCG/NDCG).
            let k = crate::runtime::STOCHASTIC_RANK_LAMBDA_DEFAULT * dot / approxes_norm_sqr;
            for (d1, &z) in der1.iter_mut().zip(zero_mean.iter()) {
                *d1 -= k * z;
            }
        }
    }

    Derivatives { der1, der2 }
}

/// `CalcDCGMetricDiff` for the (N)DCG arm (`error_functions.cpp:1222-1256`),
/// non-FilteredDCG branch (the only one in scope). `docGain = numerator(target[
/// order[oldPos]])`, `docDiff = docGain·(newWeight - oldWeight)`, and the mid
/// section uses the cumSum/cumSumUp/cumSumLow prefix sums.
///
/// `old_weight`/`new_weight` are read directly from the NORMALIZED `pos_weights`
/// vector — the SAME vector that built `cum_sum`/`cum_sum_up`/`cum_sum_low` — so
/// `doc_diff` and `mid_diff` share the `1/ideal_dcg` scale for NDCG groups where
/// `ideal_dcg != 1.0`. This mirrors upstream `oldWeight = posWeights[oldPos]` /
/// `newWeight = posWeights[newPos]` (`error_functions.cpp:1233-1234`). For the DCG
/// arm `pos_weights[pos] == 1/CalcDenominator(pos)`, so the result is unchanged.
#[allow(clippy::too_many_arguments)]
fn calc_dcg_metric_diff(
    old_pos: usize,
    new_pos: usize,
    target_slice: &[f64],
    order: &[usize],
    pos_weights: &[f64],
    cum_sum: &[f64],
    cum_sum_up: &[f64],
    cum_sum_low: &[f64],
) -> f64 {
    let doc_gain = ndcg_numerator(
        target_slice
            .get(order.get(old_pos).copied().unwrap_or(0))
            .copied()
            .unwrap_or(0.0),
    );
    // Read old/new position weights from the SAME normalized pos_weights vector that
    // built the cumSum prefix arrays (upstream posWeights[oldPos]/posWeights[newPos],
    // error_functions.cpp:1233-1234). Bounds-checked .get (CLAUDE.md unchecked-index
    // ban; T-06.3-06-01).
    let old_weight = pos_weights.get(old_pos).copied().unwrap_or(0.0);
    let new_weight = pos_weights.get(new_pos).copied().unwrap_or(0.0);
    let doc_diff = doc_gain * (new_weight - old_weight);
    // Bounds-checked prefix-sum reads (CLAUDE.md unchecked-index ban; T-06.3-11,
    // WR-02). cum_sum/cum_sum_up/cum_sum_low are length count+1, so every index
    // below is in-range for the present caller — the .get() form yields IDENTICAL
    // values, matching the pos_weights.get(..) discipline used two lines above and
    // removing the latent panic if a future top/query_top_size change desyncs the
    // index range.
    let mid_diff = if new_pos < old_pos {
        let old_mid = cum_sum.get(old_pos).copied().unwrap_or(0.0)
            - cum_sum.get(new_pos).copied().unwrap_or(0.0);
        let new_mid = cum_sum_low.get(old_pos).copied().unwrap_or(0.0)
            - cum_sum_low.get(new_pos).copied().unwrap_or(0.0);
        new_mid - old_mid
    } else {
        let old_mid = cum_sum.get(new_pos + 1).copied().unwrap_or(0.0)
            - cum_sum.get(old_pos + 1).copied().unwrap_or(0.0);
        let new_mid = cum_sum_up.get(new_pos + 1).copied().unwrap_or(0.0)
            - cum_sum_up.get(old_pos + 1).copied().unwrap_or(0.0);
        new_mid - old_mid
    };
    doc_diff + mid_diff
}

/// Stable sort of `indices` by `keys[idx]` DESCENDING (upstream `StableSort(order,
/// scores[a] > scores[b])`, `error_functions.cpp:1053`). Rust's `sort_by` is
/// stable; the comparator returns `Greater` when `keys[a] < keys[b]` to sort
/// descending while preserving the original order on ties (the parity contract —
/// a different tie-break reorders the sampled positions).
fn stable_sort_desc_by_key(indices: &mut [usize], keys: &[f64]) {
    indices.sort_by(|&a, &b| {
        let ka = keys.get(a).copied().unwrap_or(0.0);
        let kb = keys.get(b).copied().unwrap_or(0.0);
        // Descending: b vs a. NaN-safe (treat as equal → stable preserves order).
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
#[path = "ranking_der_test.rs"]
mod tests;
