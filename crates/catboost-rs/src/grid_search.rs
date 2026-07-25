//! Hyperparameter search facade (ORCH-02) — `grid_search` / `randomized_search`
//! plus the pure direction / scoring / selection primitives they compose.
//!
//! This module is orchestration-layer glue: it CALLS the already-oracle-locked
//! seams ([`crate::cv::cv`], [`CatBoostBuilder::fit`],
//! [`cb_train::parse_metric`], [`cb_train::fisher_yates_permutation`]) and never
//! re-implements them (D-04 no-regression). It lives in the `catboost-rs` facade
//! because only the facade can reach `catboost_rs::cv` /
//! `CatBoostBuilder::fit` / `Model`.
//!
//! Selection rule (first slice): the PRIMARY metric is `metrics[0]`; a
//! candidate's score is the best value over iterations of its
//! `columns["test-<metric>-mean"]` cv column — the `min` if the metric is
//! Min-optimized, the `max` if Max-optimized (see [`metric_is_max_optimal`]);
//! ties resolve to the LOWEST index (deterministic). No `unwrap` / `expect` /
//! `panic` / raw indexing appears on any path (the four workspace restriction
//! lints are denied in prod).

use cb_data::Pool;
use cb_train::{fisher_yates_permutation, parse_metric, EvalMetric};
// Per-candidate cv runs under rayon ONLY when the `cpu` feature is compiled;
// under any GPU feature candidate evaluation is serial (see `grid_search`),
// mirroring `cv.rs`, so the prelude is imported only where it is used.
#[cfg(feature = "cpu")]
use rayon::prelude::*;

use crate::cv::{cv, CvResult};
use crate::error::CatBoostError;
use crate::{CatBoostBuilder, Model};

// The degenerate-error front door is single-sourced in `cv.rs` and shared here.
use crate::cv::degenerate;

/// scikit-learn `error_score` policy for a candidate that fails to evaluate
/// (its `cv()` errors, or its primary-metric column is empty/non-finite).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorScore {
    /// Re-raise the first candidate failure (sklearn `error_score='raise'`).
    Raise,
    /// Assign this score to a failed candidate (sklearn numeric / `np.nan`
    /// default). `NaN` ⇒ the candidate is ranked worst (never chosen over a
    /// finite-scored one); a finite value competes normally.
    Value(f64),
}

/// (ORCH-02-S1) Whether a LARGER value of `metric` is better.
///
/// `true` for the ranking metrics
/// (`Ndcg`/`Dcg`/`Map`/`Mrr`/`Err`/`PFound`/`PrecisionAt`/`RecallAt`/`QueryAuc`)
/// — larger is better; `false` for the error metrics
/// (`Rmse`/`Logloss`/`Msle`/`Mae`/`Mape`/`Quantile`) — smaller is better;
/// `Custom` delegates to [`cb_compute::CustomMetric::is_max_optimal`]. Total
/// function; never panics.
#[must_use]
pub fn metric_is_max_optimal(metric: &EvalMetric) -> bool {
    // Exhaustive match (no wildcard): a future `EvalMetric` variant forces a
    // compile error here rather than a silent wrong default.
    match metric {
        EvalMetric::Rmse
        | EvalMetric::Logloss
        | EvalMetric::Msle
        | EvalMetric::Mae
        | EvalMetric::Mape
        | EvalMetric::Quantile { .. } => false,
        EvalMetric::Ndcg { .. }
        | EvalMetric::Dcg { .. }
        | EvalMetric::Map { .. }
        | EvalMetric::Mrr { .. }
        | EvalMetric::Err { .. }
        | EvalMetric::PFound { .. }
        | EvalMetric::PrecisionAt { .. }
        | EvalMetric::RecallAt { .. }
        | EvalMetric::QueryAuc { .. } => true,
        EvalMetric::Custom(h) => h.0.is_max_optimal(),
    }
}

/// The best value over a per-iteration cv column: the `max` when `max` is set,
/// else the `min`. Only FINITE entries are considered — isolated `NaN`/`inf`
/// values are skipped — and `None` is returned for an empty column OR one whose
/// every entry is non-finite, so the caller can surface a typed error rather
/// than a `NaN`/sentinel score.
///
/// This folds from the first finite value (not a `f64::MIN`/`f64::MAX` seed):
/// seeding with the sentinel and relying on `f64::max`/`f64::min` to ignore
/// `NaN` would return `Some(f64::MIN)`/`Some(f64::MAX)` for an all-`NaN`
/// column — a FINITE sentinel that defeats the "non-finite ⇒ None" guard.
fn best_over_iters(col: &[f64], max: bool) -> Option<f64> {
    let mut best: Option<f64> = None;
    for &v in col {
        if !v.is_finite() {
            continue;
        }
        best = Some(match best {
            None => v,
            Some(b) => {
                if max {
                    b.max(v)
                } else {
                    b.min(v)
                }
            }
        });
    }
    best
}

/// (ORCH-02-S2) Reduce one candidate's [`CvResult`] to its scalar "best cv
/// score" for the PRIMARY `metric`: the best value over iterations of
/// `columns["test-<metric>-mean"]` — the `min` if `!metric_is_max_optimal`, the
/// `max` if it is.
///
/// The column key is built as `format!("test-{metric}-mean")`, matching the
/// `cv()` naming convention verbatim (`cv.rs` inserts `test-<metric>-mean` with
/// the raw metric descriptor — no canonicalization — so `"RMSE"` maps to
/// `"test-RMSE-mean"`).
///
/// # Errors
/// [`CatBoostError::Train`] wrapping [`cb_core::CbError::Degenerate`] when
/// `metric` fails to [`cb_train::parse_metric`], or the expected
/// `test-<metric>-mean` column is absent / empty / non-finite. Never panics.
pub fn score_candidate(cv: &CvResult, metric: &str) -> Result<f64, CatBoostError> {
    let parsed = parse_metric(metric)?;
    let key = format!("test-{metric}-mean");
    let col = cv
        .columns
        .get(&key)
        .ok_or_else(|| degenerate(format!("score_candidate: cv result is missing column `{key}`")))?;
    best_over_iters(col, metric_is_max_optimal(&parsed)).ok_or_else(|| {
        degenerate(format!(
            "score_candidate: column `{key}` is empty or non-finite"
        ))
    })
}

/// (ORCH-02-S2) Index of the best candidate among per-candidate scores for the
/// primary `metric`. Picks the argmax when the metric is Max-optimal, else the
/// argmin; ties resolve to the LOWEST index (strict `>`/`<`, deterministic).
///
/// # Errors
/// [`CatBoostError::Train`] on an empty `cv_results` slice, an unparseable
/// `metric`, or any per-candidate [`score_candidate`] failure. Never panics.
///
/// Retained as part of the published scoring surface (and exercised by the unit
/// tests); the tolerant `error_score` search path uses [`select_index`] over
/// pre-computed scores instead, so this has no production caller.
#[cfg_attr(not(test), allow(dead_code))]
pub fn select_best(cv_results: &[CvResult], metric: &str) -> Result<usize, CatBoostError> {
    if cv_results.is_empty() {
        return Err(degenerate("select_best: `cv_results` must be non-empty"));
    }
    let max = metric_is_max_optimal(&parse_metric(metric)?);

    let mut best: Option<(usize, f64)> = None;
    for (i, cv) in cv_results.iter().enumerate() {
        let score = score_candidate(cv, metric)?;
        let is_better = match best {
            None => true,
            // Strict comparison ⇒ the FIRST (lowest-index) candidate wins ties.
            Some((_, prev)) => {
                if max {
                    score > prev
                } else {
                    score < prev
                }
            }
        };
        if is_better {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| degenerate("select_best: no candidate could be scored"))
}

/// Index of the best score among an already-computed per-candidate `scores`
/// slice, IGNORING every non-finite entry (a `NaN` `error_score` never wins over
/// a finite one). Argmax when `max`, else argmin; strict `>`/`<` so ties resolve
/// to the LOWEST index (deterministic). Returns `None` when `scores` is empty or
/// every entry is non-finite. Total function; never panics — the tolerant
/// `error_score` selection path (`run_over`) uses this instead of the
/// short-circuiting [`select_best`]. Exposed `pub(crate)` so the tie-break /
/// all-NaN rules are unit-testable without training.
#[must_use]
pub(crate) fn select_index(scores: &[f64], max: bool) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_finite() {
            continue;
        }
        let is_better = match best {
            None => true,
            // Strict comparison ⇒ the FIRST (lowest-index) candidate wins ties.
            Some((_, prev)) => {
                if max {
                    s > prev
                } else {
                    s < prev
                }
            }
        };
        if is_better {
            best = Some((i, s));
        }
    }
    best.map(|(i, _)| i)
}

/// The result of a hyperparameter search: the winning candidate + its cv
/// columns (+ the refit model when requested).
///
/// Derives only [`Debug`] — [`Model`] is not `PartialEq`, so `best_model` is
/// excluded from any equality; refit equality is asserted via prediction
/// comparison, not `==`.
#[derive(Debug)]
pub struct SearchResult {
    /// Index into the supplied `candidates` slice of the winning builder.
    pub best_index: usize,
    /// A clone of the winning [`CatBoostBuilder`] (the selected hyperparameters).
    pub best_builder: CatBoostBuilder,
    /// The [`cv`] output columns for the winning candidate (upstream
    /// `cv_results`).
    pub cv_results: CvResult,
    /// The model refit on the FULL `pool` with `best_builder` when
    /// `refit == true`; `None` otherwise.
    pub best_model: Option<Model>,
    /// One `(candidate_index, error_message)` per candidate that FAILED to
    /// evaluate and was assigned the [`ErrorScore::Value`] fallback (sklearn's
    /// `error_score`), in ascending candidate-index order; EMPTY when every
    /// candidate scored (and always empty under [`ErrorScore::Raise`], which
    /// aborts on the first failure). Candidate indices are into the ORIGINAL
    /// `candidates` slice (`randomized_search` maps sampled positions back).
    pub failures: Vec<(usize, String)>,
}

/// Cross-validate every candidate in `cands` with [`cv`], select the best by the
/// primary `metric`, and — when `refit` — refit the winner on the full `pool`.
/// The shared loop behind both [`grid_search`] and [`randomized_search`];
/// `best_index` here indexes `cands` (the caller maps it back to the original
/// slice for randomized search).
///
/// Per-candidate cv runs under `rayon` (order-preserving `collect`) ONLY when
/// the `cpu` feature is compiled; under any GPU feature it is SERIAL, matching
/// the `cv.rs` per-fold precedent. The result is identical either way.
///
/// # Errors
/// Under [`ErrorScore::Raise`], the first per-candidate [`cv`] failure (including
/// [`CatBoostError::UnsupportedModel`]) or first empty/non-finite primary column
/// propagates typed. Under [`ErrorScore::Value`], per-candidate failures are
/// TOLERATED (recorded in the returned failures list); a typed error is returned
/// only when NO candidate is scoreable (every score non-finite), or when the
/// selected candidate carries no [`CvResult`] (its cv errored yet a finite
/// `error_score` picked it). Never panics.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_over(
    pool: &Pool,
    cands: &[CatBoostBuilder],
    metrics: &[&str],
    fold_count: usize,
    shuffle: bool,
    partition_random_seed: u64,
    inverted: bool,
    folds: Option<&[Vec<usize>]>,
    error_score: ErrorScore,
) -> Result<(usize, CvResult, Vec<(usize, String)>), CatBoostError> {
    let primary = *metrics
        .first()
        .ok_or_else(|| degenerate("search: `metrics` must be non-empty"))?;

    // Per-candidate cv, collected TOLERANTLY as one `Result` per candidate
    // (`collect::<Vec<Result<_, _>>>()`, NOT `Result<Vec<_>, _>`) so a failing
    // candidate does not short-circuit the whole search — sklearn `error_score`
    // semantics. Candidate order is preserved either way.
    #[cfg(feature = "cpu")]
    let results: Vec<Result<CvResult, CatBoostError>> = cands
        .par_iter()
        .map(|b| {
            cv(
                pool,
                b,
                metrics,
                fold_count,
                shuffle,
                partition_random_seed,
                inverted,
                folds,
            )
        })
        .collect();
    #[cfg(not(feature = "cpu"))]
    let results: Vec<Result<CvResult, CatBoostError>> = cands
        .iter()
        .map(|b| {
            cv(
                pool,
                b,
                metrics,
                fold_count,
                shuffle,
                partition_random_seed,
                inverted,
                folds,
            )
        })
        .collect();

    let max = metric_is_max_optimal(&parse_metric(primary)?);

    // Derive per-candidate `(score, cv_opt)` applying the `error_score` policy,
    // recording each failed candidate's index + message.
    let mut scores: Vec<f64> = Vec::with_capacity(results.len());
    let mut cv_opts: Vec<Option<CvResult>> = Vec::with_capacity(results.len());
    let mut failures: Vec<(usize, String)> = Vec::new();
    for (i, result) in results.into_iter().enumerate() {
        let (score, cv_opt) = match result {
            Ok(cv) => match score_candidate(&cv, primary) {
                Ok(s) => (s, Some(cv)),
                // The cv ran but the primary column is empty / non-finite.
                Err(e) => match error_score {
                    ErrorScore::Raise => return Err(e),
                    ErrorScore::Value(v) => {
                        failures.push((i, e.to_string()));
                        (v, Some(cv))
                    }
                },
            },
            // The candidate's cv itself errored (fit / eval / unsupported model).
            Err(e) => match error_score {
                ErrorScore::Raise => return Err(e),
                ErrorScore::Value(v) => {
                    failures.push((i, e.to_string()));
                    (v, None)
                }
            },
        };
        scores.push(score);
        cv_opts.push(cv_opt);
    }

    let best = select_index(&scores, max).ok_or_else(|| {
        degenerate("all candidates failed to evaluate (error_score); no scoreable candidate")
    })?;
    // The winner MUST carry a CvResult: a candidate whose cv errored but was
    // picked by a finite `error_score` has `None` here ⇒ a typed error, not a
    // panic (the numeric-error_score-all-cv-errored edge).
    let cv_results = cv_opts
        .get(best)
        .cloned()
        .flatten()
        .ok_or_else(|| degenerate("selected candidate produced no cv result"))?;
    Ok((best, cv_results, failures))
}

/// (ORCH-02-S3) Grid search: cross-validate every `candidates` builder with
/// [`cv`] and keep the best by the PRIMARY metric (`metrics[0]`). When `refit`,
/// refit the winner on the full `pool`.
///
/// - `candidates`: one [`CatBoostBuilder`] per hyperparameter combination (the
///   Python layer expands `param_grid` into this). Empty ⇒ typed error.
/// - `metrics`, `fold_count`, `shuffle`, `partition_random_seed`, `inverted`,
///   `folds`: forwarded verbatim to [`cv`] per candidate.
///
/// # Errors
/// [`CatBoostError::Train`] wrapping [`cb_core::CbError::Degenerate`] for empty
/// `candidates` / `metrics` or a bad partition;
/// [`CatBoostError::UnsupportedModel`] propagated from `cv`/`staged_predict` for
/// an out-of-class candidate; any training/eval failure surfaces typed. Never
/// panics.
#[allow(clippy::too_many_arguments)]
pub fn grid_search(
    pool: &Pool,
    candidates: &[CatBoostBuilder],
    metrics: &[&str],
    fold_count: usize,
    shuffle: bool,
    partition_random_seed: u64,
    inverted: bool,
    folds: Option<&[Vec<usize>]>,
    refit: bool,
    error_score: ErrorScore,
) -> Result<SearchResult, CatBoostError> {
    if candidates.is_empty() {
        return Err(degenerate("grid_search: `candidates` must be non-empty"));
    }
    if metrics.is_empty() {
        return Err(degenerate("grid_search: `metrics` must be non-empty"));
    }

    let (best_index, cv_results, failures) = run_over(
        pool,
        candidates,
        metrics,
        fold_count,
        shuffle,
        partition_random_seed,
        inverted,
        folds,
        error_score,
    )?;
    let best_builder = candidates
        .get(best_index)
        .ok_or_else(|| degenerate("grid_search: selected index out of range"))?
        .clone();
    let best_model = if refit {
        Some(best_builder.fit(pool)?)
    } else {
        None
    };
    Ok(SearchResult {
        best_index,
        best_builder,
        cv_results,
        best_model,
        failures,
    })
}

/// (ORCH-02-S4) The deterministic candidate subsample: the first
/// `min(n_iter, m)` indices of `fisher_yates_permutation(m, seed)` (the
/// project's own `TFastRng64`, no new `rand` dependency). Exposed
/// `pub(crate)` so the subsampling is unit-testable WITHOUT training. Total
/// function; never panics — the permutation entries are all in `0..m`
/// (non-negative), so every `usize::try_from` succeeds.
#[must_use]
pub(crate) fn sample_indices(m: usize, n_iter: usize, seed: u64) -> Vec<usize> {
    let take = n_iter.min(m);
    fisher_yates_permutation(m, seed)
        .into_iter()
        .filter_map(|i| usize::try_from(i).ok())
        .take(take)
        .collect()
}

/// (ORCH-02-S4) Randomized search: the [`grid_search`] mechanism over a
/// deterministic subset of `min(n_iter, candidates.len())` candidates chosen by
/// [`sample_indices`]. `best_index` in the returned [`SearchResult`] indexes the
/// ORIGINAL `candidates` slice (the sampled position is mapped back).
///
/// Note (SPEC §9): `partition_random_seed` is double-duty — it seeds BOTH the
/// candidate subsample AND every per-candidate `cv` fold assignment (when
/// `shuffle`). Both draws are independent, freshly-seeded streams; the relative
/// comparison across sampled candidates stays fair (every candidate gets the
/// same fold assignment).
///
/// # Errors
/// As [`grid_search`], plus a typed [`CatBoostError::Train`] when `n_iter == 0`
/// or `candidates` is empty. Never panics.
#[allow(clippy::too_many_arguments)]
pub fn randomized_search(
    pool: &Pool,
    candidates: &[CatBoostBuilder],
    metrics: &[&str],
    n_iter: usize,
    fold_count: usize,
    shuffle: bool,
    partition_random_seed: u64,
    inverted: bool,
    folds: Option<&[Vec<usize>]>,
    refit: bool,
    error_score: ErrorScore,
) -> Result<SearchResult, CatBoostError> {
    if candidates.is_empty() {
        return Err(degenerate("randomized_search: `candidates` must be non-empty"));
    }
    if n_iter == 0 {
        return Err(degenerate("randomized_search: `n_iter` must be > 0"));
    }
    if metrics.is_empty() {
        return Err(degenerate("randomized_search: `metrics` must be non-empty"));
    }

    // The deterministically sampled ORIGINAL indices, in permuted order.
    let sampled_idx = sample_indices(candidates.len(), n_iter, partition_random_seed);
    let mut sampled: Vec<CatBoostBuilder> = Vec::with_capacity(sampled_idx.len());
    for &i in &sampled_idx {
        sampled.push(
            candidates
                .get(i)
                .ok_or_else(|| degenerate("randomized_search: sampled index out of range"))?
                .clone(),
        );
    }

    // Evaluate the sampled subset; we refit the ORIGINAL winner below so
    // `best_model` matches `best_index` in the original slice. `failures` here is
    // indexed by SAMPLED position — mapped back to original indices below.
    let (best_pos, cv_results, sampled_failures) = run_over(
        pool,
        &sampled,
        metrics,
        fold_count,
        shuffle,
        partition_random_seed,
        inverted,
        folds,
        error_score,
    )?;

    // Map every failure's sampled position back to its ORIGINAL candidate index
    // (checked access; a stray out-of-range position is dropped, never panics).
    let failures: Vec<(usize, String)> = sampled_failures
        .into_iter()
        .filter_map(|(pos, msg)| sampled_idx.get(pos).map(|&orig| (orig, msg)))
        .collect();

    // Map the sampled position back to the ORIGINAL candidate index.
    let best_index = *sampled_idx
        .get(best_pos)
        .ok_or_else(|| degenerate("randomized_search: mapped index out of range"))?;
    let best_builder = candidates
        .get(best_index)
        .ok_or_else(|| degenerate("randomized_search: best index out of range"))?
        .clone();
    let best_model = if refit {
        Some(best_builder.fit(pool)?)
    } else {
        None
    };
    Ok(SearchResult {
        best_index,
        best_builder,
        cv_results,
        best_model,
        failures,
    })
}
