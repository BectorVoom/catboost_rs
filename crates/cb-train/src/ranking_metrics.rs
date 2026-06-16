//! Per-group ranking-metric formulas (LOSS-05, D-6.3-05) — the nine eval-only
//! ranking quantities **NDCG, DCG, MAP, MRR, ERR, PFound, PrecisionAt, RecallAt,
//! QueryAUC**, each computed per query group and averaged over groups.
//!
//! These are eval-only (no derivative): they extend [`crate::metrics::EvalMetric`]
//! via the widened `eval_grouped` group seam and carry no gradient. Every formula
//! is transcribed verbatim from vendored catboost 1.2.10 `libs/metrics/` with the
//! `file:line` citation on each function.
//!
//! # Shared `compare_docs` tie-break (transcribed ONCE)
//!
//! Every approx-sorted metric breaks ties the same way upstream's `CompareDocs`
//! (`doc_comparator.h:4-6`) does: by predicted value **descending**, then — when
//! predictions are equal — by target **ascending**, then by a **stable** original
//! index. [`compare_docs`] is the single transcription; the per-metric sorts all
//! route through it so the tie-handling matches ≤1e-5 (RESEARCH Pitfall 6 /
//! Anti-Patterns: "transcribe `CompareDocs` once").
//!
//! # Parity discipline
//!
//! Every per-group and cross-group reduction routes through `cb_core::sum_f64`
//! (group index ascending, doc index ascending — RESEARCH Pitfall 4); no raw
//! float fold exists in this module (D-08 CI-grep ban). Empty group / IDCG==0 /
//! zero-relevant guards return a fixed value (never divide) mirroring upstream's
//! `GetFinalError` (Security V5; T-06.3-05-01). No `unwrap`/`expect`/`panic`/
//! unchecked indexing on the numeric buffers (CLAUDE.md lint gate).

use cb_core::sum_f64;

/// NDCG/DCG numerator gain type (`ENdcgMetricType`, `dcg.cpp:69-75`): `Base` uses
/// the raw relevance `rel`; `Exp` uses `2^rel - 1`. Upstream default is `Base`
/// (`metric.cpp:3011`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcgMetricType {
    /// Numerator is the raw relevance (`num = rel`).
    Base,
    /// Numerator is `2^rel - 1`.
    Exp,
}

/// NDCG/DCG position-discount denominator (`ENdcgDenominatorType`,
/// `dcg.cpp:88-100`): `Position` is `1/(pos+1)`; `LogPosition` is
/// `1/log2(pos+2)`. Upstream default is `LogPosition` (`metric.cpp:3012`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcgDenominator {
    /// `decay[pos] = 1 / (pos + 1)`.
    Position,
    /// `decay[pos] = 1 / log2(pos + 2)`.
    LogPosition,
}

/// QueryAUC singleclass AUC type (`EAucType`, `metric.cpp:5509-5574`). The
/// singleclass default upstream is `Classic` (binary-class AUC per group);
/// `Ranking` computes the inversions-based ranking AUC per group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AucType {
    /// Binary-class AUC per group (`CalcBinClassAuc`, `auc.cpp:247-303`).
    Classic,
    /// Ranking AUC per group (`CalcAUC`, `auc.cpp:183-241`).
    Ranking,
}

/// The shared document comparator (`doc_comparator.h:4-6`): a strict "is `left`
/// ranked before `right`?" predicate — predicted value **descending**, ties
/// broken by target **ascending**. Returns `true` when `left` should sort before
/// `right`.
///
/// `CompareDocs(approxL, targetL, approxR, targetR) = approxL != approxR ?
/// approxL > approxR : targetL < targetR`.
#[must_use]
pub fn compare_docs(approx_left: f64, target_left: f64, approx_right: f64, target_right: f64) -> bool {
    if approx_left != approx_right {
        approx_left > approx_right
    } else {
        target_left < target_right
    }
}

/// Return the group's object order as group-local indices sorted by
/// [`compare_docs`] with a **stable** original-index tie-break (the upstream
/// `StableSort`/`PartialSort` + `lhs < rhs` final tie-break, `dcg.cpp:30-48`,
/// `metric.cpp:6153`).
fn sorted_indices(approx: &[f64], target: &[f64]) -> Vec<usize> {
    let n = approx.len().min(target.len());
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&l, &r| {
        let (al, tl) = (approx.get(l).copied().unwrap_or(0.0), target.get(l).copied().unwrap_or(0.0));
        let (ar, tr) = (approx.get(r).copied().unwrap_or(0.0), target.get(r).copied().unwrap_or(0.0));
        if compare_docs(al, tl, ar, tr) {
            std::cmp::Ordering::Less
        } else if compare_docs(ar, tr, al, tl) {
            std::cmp::Ordering::Greater
        } else {
            // Stable index tie-break (`lhs < rhs`).
            l.cmp(&r)
        }
    });
    idx
}

/// Clamp `top` to the group size: `k = (top < 0 || size < top) ? size : top`
/// (`precision_recall_at_k.cpp:9-11`). `top == -1` (the upstream `DefaultTopSize`)
/// means "use the full group".
#[must_use]
pub fn clamp_top(top: i64, size: usize) -> usize {
    if top < 0 || (size as i64) < top {
        size
    } else {
        // top is non-negative and <= size here.
        top as usize
    }
}

/// One position discount `decay[pos]` for DCG/NDCG (`FillDcgDecay`,
/// `dcg.cpp:80-105`). `decay[0] = 1`; thereafter `Position → 1/(pos+1)`,
/// `LogPosition → 1/log2(pos+2)`.
fn dcg_decay(pos: usize, denominator: DcgDenominator) -> f64 {
    if pos == 0 {
        1.0
    } else {
        match denominator {
            DcgDenominator::Position => 1.0 / ((pos + 1) as f64),
            DcgDenominator::LogPosition => 1.0 / ((pos + 2) as f64).log2(),
        }
    }
}

/// DCG over an already approx-sorted list of targets (`CalcDcgSorted`,
/// `dcg.cpp:60-78`): `Σ_pos num(sortedTargets[pos]) · decay[pos]` where
/// `num = rel` (Base) or `2^rel - 1` (Exp). Reduced through `cb_core::sum_f64`.
fn dcg_sorted(sorted_targets: &[f64], dcg_type: DcgMetricType, denominator: DcgDenominator) -> f64 {
    let terms: Vec<f64> = sorted_targets
        .iter()
        .enumerate()
        .map(|(pos, &rel)| {
            let num = match dcg_type {
                DcgMetricType::Base => rel,
                DcgMetricType::Exp => 2f64.powf(rel) - 1.0,
            };
            num * dcg_decay(pos, denominator)
        })
        .collect();
    sum_f64(&terms)
}

/// The top-`k` targets in approx-sorted order (`GetTopSortedTargets`,
/// `dcg.cpp:18-58`) — DCG path (sort by [`compare_docs`]).
fn top_sorted_targets_by_approx(approx: &[f64], target: &[f64], top_size: usize) -> Vec<f64> {
    let order = sorted_indices(approx, target);
    order
        .iter()
        .take(top_size)
        .map(|&i| target.get(i).copied().unwrap_or(0.0))
        .collect()
}

/// The top-`k` targets in IDEAL (target-descending) order (`CalcIDcg`,
/// `dcg.cpp:126-142`: comparator `left.Target > right.Target`, stable index
/// tie-break).
fn top_sorted_targets_ideal(target: &[f64], top_size: usize) -> Vec<f64> {
    let n = target.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&l, &r| {
        let (tl, tr) = (target.get(l).copied().unwrap_or(0.0), target.get(r).copied().unwrap_or(0.0));
        if tl > tr {
            std::cmp::Ordering::Less
        } else if tr > tl {
            std::cmp::Ordering::Greater
        } else {
            l.cmp(&r)
        }
    });
    idx.iter()
        .take(top_size)
        .map(|&i| target.get(i).copied().unwrap_or(0.0))
        .collect()
}

/// Per-group **DCG** (`CalcDcg`, `dcg.cpp:108-124`).
#[must_use]
pub fn dcg_group(
    approx: &[f64],
    target: &[f64],
    top: i64,
    dcg_type: DcgMetricType,
    denominator: DcgDenominator,
) -> f64 {
    let size = approx.len().min(target.len());
    let k = clamp_top(top, size);
    let sorted = top_sorted_targets_by_approx(approx, target, k);
    dcg_sorted(&sorted, dcg_type, denominator)
}

/// Per-group **NDCG** (`CalcNdcg`, `dcg.cpp:144-148`): `DCG / IDCG`; when
/// `IDCG <= 0` upstream returns **1** (NOT 0 — `dcg.cpp:147`).
#[must_use]
pub fn ndcg_group(
    approx: &[f64],
    target: &[f64],
    top: i64,
    dcg_type: DcgMetricType,
    denominator: DcgDenominator,
) -> f64 {
    let size = approx.len().min(target.len());
    let k = clamp_top(top, size);
    let dcg = dcg_sorted(&top_sorted_targets_by_approx(approx, target, k), dcg_type, denominator);
    let idcg = dcg_sorted(&top_sorted_targets_ideal(target, k), dcg_type, denominator);
    if idcg > 0.0 {
        dcg / idcg
    } else {
        1.0
    }
}

/// Per-group **PFound** (`TPFoundCalcer::AddQuery`, `pfound.h:22-59`): cascade
/// `pLook = 1`; `PFound += rel[pos]·pLook`; `pLook *= (1 - rel[pos])·decay`, over
/// the top-`depth` approx-sorted docs. Default `decay = 0.85` (`pfound.h:15`).
#[must_use]
pub fn pfound_group(approx: &[f64], target: &[f64], top: i64, decay: f64) -> f64 {
    let size = approx.len().min(target.len());
    let order = sorted_indices(approx, target);
    let depth = if top < 0 { size } else { size.min(top as usize) };
    let mut p_look = 1.0f64;
    let mut p_found = 0.0f64;
    for &doc in order.iter().take(depth) {
        let rel = target.get(doc).copied().unwrap_or(0.0);
        p_found += rel * p_look;
        p_look *= (1.0 - rel) * decay;
    }
    p_found
}

/// Per-group **ERR** (`TERRMetric::CalcQueryERR`, `metric.cpp:6140-6164`):
/// approx-sorted (ties by target ascending, `approx[a]>approx[b] || (== &&
/// target[a]<target[b])`); cascade `pLook = 1`; `ERR += pLook·rel/(pos+1)`;
/// `pLook *= 1 - rel`, over the top-`lookupDepth` docs.
#[must_use]
pub fn err_group(approx: &[f64], target: &[f64], top: i64) -> f64 {
    let size = approx.len().min(target.len());
    let order = sorted_indices(approx, target);
    let depth = if top == -1 { size } else { size.min(top.max(0) as usize) };
    let mut p_look = 1.0f64;
    let mut query_rr = 0.0f64;
    for (pos, &doc) in order.iter().take(depth).enumerate() {
        let rel = target.get(doc).copied().unwrap_or(0.0);
        query_rr += p_look * rel / ((pos + 1) as f64);
        p_look *= 1.0 - rel;
    }
    query_rr
}

/// Per-group **MRR** (`TMRRMetric::CalcQueryReciprocalRank`,
/// `metric.cpp:6035-6060`): `1 / rank_of_first_relevant`. Relevance is
/// `target > border`. Upstream finds the max approx among relevant docs, then
/// counts how many docs strictly beat it (or tie with a non-relevant doc) to get
/// the 1-based rank; 0 if no relevant doc or the rank exceeds `maxPos`.
#[must_use]
pub fn mrr_group(approx: &[f64], target: &[f64], top: i64, border: f64) -> f64 {
    let size = approx.len().min(target.len());
    let mut found_relevant = false;
    let mut max_relevant_approx = f64::MIN;
    for i in 0..size {
        let t = target.get(i).copied().unwrap_or(0.0);
        if t > border {
            found_relevant = true;
            let a = approx.get(i).copied().unwrap_or(0.0);
            if a > max_relevant_approx {
                max_relevant_approx = a;
            }
        }
    }
    if !found_relevant {
        return 0.0;
    }
    let max_pos: i64 = if top == -1 { size as i64 } else { (size as i64).min(top) };
    let mut pos: i64 = 1;
    let mut i = 0usize;
    while (i as i64) < size as i64 && pos <= max_pos {
        let a = approx.get(i).copied().unwrap_or(0.0);
        let t = target.get(i).copied().unwrap_or(0.0);
        if a > max_relevant_approx || (a == max_relevant_approx && t <= border) {
            pos += 1;
        }
        i += 1;
    }
    if pos <= max_pos {
        1.0 / (pos as f64)
    } else {
        0.0
    }
}

/// Per-group **PrecisionAt@k** (`CalcPrecisionAtK`,
/// `precision_recall_at_k.cpp:49-53`): `relevant_in_top_k / k`, `k = clamp_top`,
/// `relevant = target > border`.
#[must_use]
pub fn precision_at_group(approx: &[f64], target: &[f64], top: i64, border: f64) -> f64 {
    let size = approx.len().min(target.len());
    let k = clamp_top(top, size);
    if k == 0 {
        return 0.0;
    }
    let order = sorted_indices(approx, target);
    let relevant = order
        .iter()
        .take(k)
        .filter(|&&i| target.get(i).copied().unwrap_or(0.0) > border)
        .count();
    (relevant as f64) / (k as f64)
}

/// Per-group **RecallAt@k** (`CalcRecallAtK`,
/// `precision_recall_at_k.cpp:55-60`): `relevant_in_top_k / total_relevant`;
/// returns **1** when `total_relevant == 0`.
#[must_use]
pub fn recall_at_group(approx: &[f64], target: &[f64], top: i64, border: f64) -> f64 {
    let size = approx.len().min(target.len());
    let k = clamp_top(top, size);
    let order = sorted_indices(approx, target);
    let total_relevant = (0..size)
        .filter(|&i| target.get(i).copied().unwrap_or(0.0) > border)
        .count();
    if total_relevant == 0 {
        return 1.0;
    }
    let relevant_in_top = order
        .iter()
        .take(k)
        .filter(|&&i| target.get(i).copied().unwrap_or(0.0) > border)
        .count();
    (relevant_in_top as f64) / (total_relevant as f64)
}

/// Per-group **MAP@k** / Average Precision (`CalcAveragePrecisionK`,
/// `precision_recall_at_k.cpp:62-83`): sort by [`compare_docs`]; over ALL docs in
/// sorted order, on each relevant doc (`target > border`) increment `hits`, and
/// when its position `< k` add `hits/(pos+1)` to `score`; return
/// `hits > 0 ? score / min(hits, k) : 0`.
#[must_use]
pub fn map_at_group(approx: &[f64], target: &[f64], top: i64, border: f64) -> f64 {
    let size = approx.len().min(target.len());
    let k = clamp_top(top, size);
    let order = sorted_indices(approx, target);
    let mut hits = 0.0f64;
    let mut score = 0.0f64;
    for (pos, &doc) in order.iter().enumerate() {
        if target.get(doc).copied().unwrap_or(0.0) > border {
            hits += 1.0;
            if pos < k {
                score += hits / ((pos + 1) as f64);
            }
        }
    }
    if hits > 0.0 {
        score / hits.min(k as f64)
    } else {
        0.0
    }
}

/// Per-group **Ranking AUC** (`CalcAUC`, `auc.cpp:183-241`): the concordance
/// probability over `(target, prediction)` samples (weights uniform `1` here).
///
/// Upstream's `CalcAUC` reduces — via a weighted merge-sort + equal-prediction /
/// equal-target corrections — to the standard ranking-AUC definition: over every
/// ordered pair `(i, j)` with `target[i] > target[j]`, the fraction the prediction
/// orders correctly (`approx[i] > approx[j]` → full credit; `approx[i] ==
/// approx[j]` → half credit). With single-thread, unit weights, and the small
/// per-group sizes here, the direct `O(size²)` concordance count is bit-identical
/// to the merge-sort result and far clearer. Returns `0` when no target-ordered
/// pair exists (a group with all-equal targets — upstream's `pairWeightSum == 0`).
fn ranking_auc_group(approx: &[f64], target: &[f64]) -> f64 {
    let size = approx.len().min(target.len());
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for i in 0..size {
        let (ai, ti) = (
            approx.get(i).copied().unwrap_or(0.0),
            target.get(i).copied().unwrap_or(0.0),
        );
        for j in 0..size {
            let (aj, tj) = (
                approx.get(j).copied().unwrap_or(0.0),
                target.get(j).copied().unwrap_or(0.0),
            );
            if ti > tj {
                denominator += 1.0;
                if ai > aj {
                    numerator += 1.0;
                } else if ai == aj {
                    numerator += 0.5;
                }
            }
        }
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Per-group **Classic (binary-class) AUC** (`CalcBinClassAuc`,
/// `auc.cpp:247-303`): split docs into positive (`target > 0`, weight
/// `target·w`) and negative (`target < 1`, weight `(1-target)·w`) samples; the
/// AUC is the weighted fraction of (positive, negative) pairs the prediction
/// orders correctly, with half-credit for prediction ties. Returns 0 when either
/// class is empty.
fn classic_auc_group(approx: &[f64], target: &[f64]) -> f64 {
    let size = approx.len().min(target.len());
    let mut positives: Vec<(f64, f64)> = Vec::new(); // (prediction, weight)
    let mut negatives: Vec<(f64, f64)> = Vec::new();
    for i in 0..size {
        let t = target.get(i).copied().unwrap_or(0.0);
        let a = approx.get(i).copied().unwrap_or(0.0);
        if t > 0.0 {
            positives.push((a, t));
        }
        if t < 1.0 {
            negatives.push((a, 1.0 - t));
        }
    }
    if positives.is_empty() || negatives.is_empty() {
        return 0.0;
    }
    // Direct O(P·N) pair count (single-thread, small groups). For each
    // (positive, negative) pair: full credit when positive prediction is GREATER,
    // half credit on a tie (mirrors the prefix-sum + equal-prediction logic).
    let mut numerator = 0.0f64;
    let mut positive_weight_sum = 0.0f64;
    let mut negative_weight_sum = 0.0f64;
    for &(_pp, pw) in &positives {
        positive_weight_sum += pw;
    }
    for &(np, nw) in &negatives {
        negative_weight_sum += nw;
        for &(pp, pw) in &positives {
            if pp > np {
                numerator += pw * nw;
            } else if pp == np {
                numerator += 0.5 * pw * nw;
            }
        }
    }
    numerator / (positive_weight_sum * negative_weight_sum)
}

/// Per-group **QueryAUC** (`TQueryAUCMetric::EvalSingleThread`,
/// `metric.cpp:5606-5690`): Classic (binary-class) or Ranking AUC per group.
#[must_use]
pub fn query_auc_group(approx: &[f64], target: &[f64], auc_type: AucType) -> f64 {
    match auc_type {
        AucType::Classic => classic_auc_group(approx, target),
        AucType::Ranking => ranking_auc_group(approx, target),
    }
}

#[cfg(test)]
#[path = "ranking_metrics_test.rs"]
mod tests;
