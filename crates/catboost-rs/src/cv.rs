//! k-fold cross-validation surface (ORCH-01) — the facade `cv()` glue plus its
//! pure partitioning / aggregation helpers.
//!
//! This module is orchestration-layer glue: it CALLS the already-oracle-locked
//! seams (`CatBoostBuilder::fit`, `Model::staged_predict`, `eval_metric`,
//! `fisher_yates_permutation`, `sum_f64`) N times and never re-implements them
//! (D-04 no-regression). It lives in the `catboost-rs` facade because only the
//! facade can reach `CatBoostBuilder::fit` / `Model::staged_predict`.
//!
//! - [`make_cv_folds`] (ORCH-01-S1): map `(n, group_id, fold_count, shuffle,
//!   seed, cv_type)` to disjoint [`CvFold`]s (plain or grouped k-fold).
//!
//! Every reduction routes through [`cb_core::sum_f64`] (D-08); no `unwrap` /
//! `expect` / `panic` / raw indexing appears on any path (the four workspace
//! restriction lints are denied in prod).

use std::collections::BTreeMap;
use std::ops::Range;

use cb_core::sum_f64;
use cb_data::Pool;
use cb_train::fisher_yates_permutation;
// Per-fold parallelism runs under rayon ONLY when the `cpu` feature is compiled;
// under any GPU feature fold execution is serial (see `cv`), so the prelude is
// imported only where it is used.
#[cfg(feature = "cpu")]
use rayon::prelude::*;

use crate::builder::CatBoostBuilder;
use crate::error::CatBoostError;
use crate::metrics::eval_metric;

/// One CV fold: the row indices used to TRAIN and to TEST. Disjoint; their
/// union covers every partitioned row exactly once (Classical) — for the
/// grouped variant, boundaries fall only between whole groups.
#[derive(Debug, Clone, PartialEq)]
pub struct CvFold {
    /// Row indices used to train the fold's model.
    pub train: Vec<usize>,
    /// Row indices held out to evaluate the fold's model.
    pub test: Vec<usize>,
}

/// How folds are formed. `Classical` tests on one block, trains on the rest;
/// `Inverted` trains on one block, tests on the rest (upstream `inverted=True`
/// / `type='Inverted'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvType {
    /// Test on one block, train on the complement (the default).
    Classical,
    /// Train on one block, test on the complement (`inverted=True`).
    Inverted,
}

/// Build a typed [`CatBoostError::Train`] wrapping a
/// [`cb_core::CbError::Degenerate`] with `msg` — the single degenerate-error
/// front door for the orchestration modules (no panic path). Shared with
/// `grid_search.rs` so the degenerate-error shape is single-sourced.
pub(crate) fn degenerate(msg: impl Into<String>) -> CatBoostError {
    CatBoostError::Train(cb_core::CbError::Degenerate(msg.into()))
}

/// Split `len` items into `fold_count` contiguous, near-equal blocks, returning
/// each block's `(start, end)` half-open bounds. The first `len % fold_count`
/// blocks are one larger than the rest, so every block is non-empty when
/// `fold_count <= len`. No allocation-time indexing / panic.
fn block_bounds(len: usize, fold_count: usize) -> Vec<(usize, usize)> {
    // `fold_count >= 1` is guaranteed by the caller's validation; guard anyway
    // so a zero divisor can never occur.
    if fold_count == 0 {
        return Vec::new();
    }
    let base = len / fold_count;
    let rem = len % fold_count;
    let mut bounds = Vec::with_capacity(fold_count);
    let mut start = 0usize;
    for k in 0..fold_count {
        let size = base + usize::from(k < rem);
        let end = start + size;
        bounds.push((start, end));
        start = end;
    }
    bounds
}

/// Verify each distinct `group_id` occupies exactly one contiguous run and
/// return the per-group row spans, in original order. A non-contiguous id
/// (a group re-appearing after a different group) is a
/// [`cb_core::CbError::Degenerate`], because the downstream `eval_grouped`
/// requires per-group contiguity.
fn contiguous_group_spans(group_id: &[u64]) -> Result<Vec<Range<usize>>, CatBoostError> {
    let mut spans: Vec<Range<usize>> = Vec::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut i = 0usize;
    while let Some(&id) = group_id.get(i) {
        let start = i;
        let mut j = i.saturating_add(1);
        while let Some(&next) = group_id.get(j) {
            if next != id {
                break;
            }
            j = j.saturating_add(1);
        }
        if !seen.insert(id) {
            return Err(degenerate(format!(
                "group_id is non-contiguous: id {id} appears in more than one run"
            )));
        }
        spans.push(start..j);
        i = j;
    }
    Ok(spans)
}

/// Assemble one plain-mode fold from the ordered index list and a test block's
/// bounds. `test` = `order[start..end]`; `train` = the rest of `order` in order.
fn plain_fold(order: &[usize], start: usize, end: usize, cv_type: CvType) -> CvFold {
    let test: Vec<usize> = order.get(start..end).unwrap_or(&[]).to_vec();
    let mut train: Vec<usize> = Vec::with_capacity(order.len().saturating_sub(test.len()));
    train.extend_from_slice(order.get(..start).unwrap_or(&[]));
    train.extend_from_slice(order.get(end..).unwrap_or(&[]));
    orient(train, test, cv_type)
}

/// Apply the [`CvType`] role assignment: `Classical` keeps `(train, test)`;
/// `Inverted` swaps them (train on the one block).
fn orient(train: Vec<usize>, test: Vec<usize>, cv_type: CvType) -> CvFold {
    match cv_type {
        CvType::Classical => CvFold { train, test },
        CvType::Inverted => CvFold {
            train: test,
            test: train,
        },
    }
}

/// (ORCH-01-S1) Partition `n` rows into `fold_count` folds. `group_id` empty ⇒
/// plain k-fold; non-empty ⇒ grouped k-fold (whole groups per fold, contiguity
/// respected). `shuffle` seeds a deterministic
/// [`cb_train::fisher_yates_permutation`] by `seed` — in plain mode over the row
/// order, in grouped mode over the GROUP-SPAN order only (never the rows within
/// a group); `false` uses the identity order. `cv_type` selects Classical vs
/// Inverted.
///
/// # Errors
/// [`CatBoostError::Train`] wrapping [`cb_core::CbError::Degenerate`] when
/// `fold_count < 2`, `fold_count > n` (or `> group_count` when grouped),
/// `n == 0`, or `group_id` is non-empty and non-contiguous.
pub fn make_cv_folds(
    n: usize,
    group_id: &[u64],
    fold_count: usize,
    shuffle: bool,
    seed: u64,
    cv_type: CvType,
) -> Result<Vec<CvFold>, CatBoostError> {
    if n == 0 {
        return Err(degenerate("make_cv_folds: n must be > 0"));
    }
    if fold_count < 2 {
        return Err(degenerate(format!(
            "make_cv_folds: fold_count must be >= 2 (got {fold_count})"
        )));
    }

    if group_id.is_empty() {
        // ---- Plain k-fold over row indices ----
        if fold_count > n {
            return Err(degenerate(format!(
                "make_cv_folds: fold_count {fold_count} exceeds n {n}"
            )));
        }
        let order: Vec<usize> = if shuffle {
            fisher_yates_permutation(n, seed)
                .into_iter()
                .map(|i| i as usize)
                .collect()
        } else {
            (0..n).collect()
        };
        let folds = block_bounds(order.len(), fold_count)
            .into_iter()
            .map(|(start, end)| plain_fold(&order, start, end, cv_type))
            .collect();
        Ok(folds)
    } else {
        // ---- Grouped k-fold: whole groups per fold ----
        let spans = contiguous_group_spans(group_id)?;
        let group_count = spans.len();
        if fold_count > group_count {
            return Err(degenerate(format!(
                "make_cv_folds: fold_count {fold_count} exceeds group_count {group_count}"
            )));
        }

        // Shuffle permutes which whole group is assigned to which fold — the
        // GROUP-SPAN order only, never rows within a group.
        let group_order: Vec<usize> = if shuffle {
            fisher_yates_permutation(group_count, seed)
                .into_iter()
                .map(|i| i as usize)
                .collect()
        } else {
            (0..group_count).collect()
        };

        let mut folds = Vec::with_capacity(fold_count);
        for (start, end) in block_bounds(group_count, fold_count) {
            // The set of group indices assigned to this fold's TEST block.
            let test_groups: std::collections::HashSet<usize> = group_order
                .get(start..end)
                .unwrap_or(&[])
                .iter()
                .copied()
                .collect();

            // Emit rows in ORIGINAL group order (spans in index order) so each
            // fold's rows stay ascending and every group stays contiguous —
            // a hard precondition of `eval_grouped` downstream.
            let mut test: Vec<usize> = Vec::new();
            let mut train: Vec<usize> = Vec::new();
            for (gi, span) in spans.iter().enumerate() {
                let target = if test_groups.contains(&gi) {
                    &mut test
                } else {
                    &mut train
                };
                target.extend(span.clone());
            }
            folds.push(orient(train, test, cv_type));
        }
        Ok(folds)
    }
}

/// One fold's per-metric TEST and TRAIN metric curves. Each inner `Vec<f64>` is
/// the metric's value at every training iteration (a "curve"); the outer `Vec`
/// is parallel to the requested `metrics` slice (`test[mi]` / `train[mi]` is
/// metric `mi`'s curve). Internal to [`aggregate_folds`] / the per-fold loop —
/// consumed by the public [`cv`] (ORCH-01-S5).
#[derive(Debug, Clone, PartialEq)]
struct FoldCurves {
    /// Per-metric TEST curve over iterations (`test[metric][iteration]`).
    test: Vec<Vec<f64>>,
    /// Per-metric TRAIN curve over iterations (`train[metric][iteration]`).
    train: Vec<Vec<f64>>,
}

/// The result of a `cv()` run: the `iterations` index plus one column per
/// `<dataset>-<metric>-<stat>` (e.g. `test-RMSE-mean`), each a per-iteration
/// vector of length `iterations`. Mirrors the upstream cv DataFrame columns.
#[derive(Debug, Clone, PartialEq)]
pub struct CvResult {
    /// The per-iteration index `[0, 1, …, iterations-1]`.
    pub iterations: Vec<usize>,
    /// One column per `<dataset>-<metric>-<stat>` name, sorted by key.
    pub columns: BTreeMap<String, Vec<f64>>,
}

/// Mean and SAMPLE (ddof=1) standard deviation of the per-fold `values` at one
/// iteration. The caller guarantees `values.len() >= 2` (the `F >= 2` gate in
/// [`aggregate_folds`]), so the `F - 1` denominator is never zero. Every sum
/// routes through [`cb_core::sum_f64`] (D-08).
fn mean_std_over_folds(values: &[f64]) -> (f64, f64) {
    let f = values.len();
    let mean = sum_f64(values) / f as f64;
    let sq: Vec<f64> = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .collect();
    let denom = f.saturating_sub(1) as f64;
    let std = (sum_f64(&sq) / denom).sqrt();
    (mean, std)
}

/// (ORCH-01-S4) Aggregate per-fold, per-metric TEST/TRAIN curves into the
/// upstream `<dataset>-<metric>-<stat>` columns: per iteration, the mean and
/// SAMPLE std (ddof=1) across folds.
///
/// # Errors
/// [`CatBoostError::Train`] wrapping [`cb_core::CbError::Degenerate`] when there
/// are fewer than 2 folds (`F == 0` — never `sum_f64(&[])/0 == NaN` — or
/// `F == 1`, which has no defined sample std), when a fold does not carry one
/// curve per metric, or when any curve's length differs from `iterations`
/// (ragged aggregation).
fn aggregate_folds(
    per_fold: &[FoldCurves],
    metrics: &[&str],
    iterations: usize,
) -> Result<CvResult, CatBoostError> {
    let fold_count = per_fold.len();
    // Defense-in-depth: F == 0 (e.g. an empty explicit `folds` list) would make
    // `sum_f64(&[]) / 0 == NaN`; F == 1 has no defined sample std. Both are
    // typed errors, never a NaN-bearing Ok.
    if fold_count < 2 {
        return Err(degenerate(format!(
            "cv aggregate: need >= 2 folds for a defined mean/sample-std (got {fold_count})"
        )));
    }

    // Every fold must carry exactly one curve per metric, each of length
    // `iterations`. Reject ragged input rather than aggregating over holes.
    for fold in per_fold {
        if fold.test.len() != metrics.len() || fold.train.len() != metrics.len() {
            return Err(degenerate(
                "cv aggregate: a fold does not carry one curve per metric",
            ));
        }
        for curve in fold.test.iter().chain(fold.train.iter()) {
            if curve.len() != iterations {
                return Err(degenerate(
                    "cv aggregate: ragged per-fold curve length (mismatched iterations)",
                ));
            }
        }
    }

    let mut columns: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (mi, &metric) in metrics.iter().enumerate() {
        for is_train in [false, true] {
            let dataset = if is_train { "train" } else { "test" };
            let mut means = Vec::with_capacity(iterations);
            let mut stds = Vec::with_capacity(iterations);
            for i in 0..iterations {
                let mut vals = Vec::with_capacity(fold_count);
                for fold in per_fold {
                    let curves = if is_train { &fold.train } else { &fold.test };
                    let Some(curve) = curves.get(mi) else {
                        return Err(degenerate("cv aggregate: missing metric curve"));
                    };
                    let Some(&v) = curve.get(i) else {
                        return Err(degenerate("cv aggregate: missing iteration value"));
                    };
                    vals.push(v);
                }
                let (mean, std) = mean_std_over_folds(&vals);
                means.push(mean);
                stds.push(std);
            }
            columns.insert(format!("{dataset}-{metric}-mean"), means);
            columns.insert(format!("{dataset}-{metric}-std"), stds);
        }
    }

    Ok(CvResult {
        iterations: (0..iterations).collect(),
        columns,
    })
}

/// Evaluate every requested `metric` at every staged prediction, yielding one
/// per-iteration curve per metric (`curves[metric][stage]`). `label`/`weight`/
/// `group_id` come from the sub-Pool the `stages` were predicted on. Weight and
/// group ids are forwarded as `Option` (an empty slice is fine — uniform weight
/// / a single group).
///
/// # Errors
/// [`CatBoostError`] propagated from [`eval_metric`] (unknown metric, empty
/// approx on a 0-row fold, non-contiguous `group_id`, …).
fn curve_over_stages(
    stages: &[Vec<f64>],
    sub_pool: &Pool,
    metrics: &[&str],
) -> Result<Vec<Vec<f64>>, CatBoostError> {
    let label = sub_pool.label();
    let weight = sub_pool.weights();
    let group_id = sub_pool.group_id();

    let mut curves: Vec<Vec<f64>> = Vec::with_capacity(metrics.len());
    for &metric in metrics {
        let mut curve = Vec::with_capacity(stages.len());
        for approx in stages {
            let value = eval_metric(label, approx, metric, Some(weight), Some(group_id))?;
            curve.push(value);
        }
        curves.push(curve);
    }
    Ok(curves)
}

/// (ORCH-01-S3) Run one CV fold: fit a fresh model on the `fold.train` sub-Pool,
/// then produce per-metric per-iteration TEST and TRAIN metric curves by
/// staged-evaluating on both sub-Pools. Delegates train / predict / eval
/// entirely to the reused facade seams (`CatBoostBuilder::fit`,
/// `Model::staged_predict`, [`eval_metric`]).
///
/// # Errors
/// [`CatBoostError`] propagated from any seam: a degenerate (0-row) train fold
/// errors inside `fit` (empty-target guard); a 0-row test fold errors inside the
/// subsequent [`eval_metric`] (empty-approx guard); a non-scalar/oblivious/
/// float-only model errors inside `staged_predict`
/// ([`CatBoostError::UnsupportedModel`]). A mismatch between the test and train
/// stage counts is a typed [`cb_core::CbError::Degenerate`]. Never panics, never
/// returns a `NaN`-bearing curve.
fn run_fold(
    pool: &Pool,
    fold: &CvFold,
    builder: &CatBoostBuilder,
    metrics: &[&str],
) -> Result<FoldCurves, CatBoostError> {
    let train_pool = pool.select_rows(&fold.train);
    let test_pool = pool.select_rows(&fold.test);

    let model = builder.fit(&train_pool)?;
    let test_stages = model.staged_predict(&test_pool, None, None, None)?;
    let train_stages = model.staged_predict(&train_pool, None, None, None)?;

    if test_stages.len() != train_stages.len() {
        return Err(degenerate(format!(
            "cv run_fold: test stage count {} != train stage count {}",
            test_stages.len(),
            train_stages.len()
        )));
    }

    let test = curve_over_stages(&test_stages, &test_pool, metrics)?;
    let train = curve_over_stages(&train_stages, &train_pool, metrics)?;
    Ok(FoldCurves { test, train })
}

/// Build the per-fold [`CvFold`] list from an explicit caller-supplied `folds`
/// argument: each entry is a fold's TEST-row index list, and `train` is that
/// fold's complement over `0..n_rows` in ascending order. An explicit but EMPTY
/// list is a typed error (never forwarded to zero-fold aggregation, which would
/// produce `sum_f64(&[])/0 == NaN`).
fn folds_from_explicit(list: &[Vec<usize>], n_rows: usize) -> Result<Vec<CvFold>, CatBoostError> {
    if list.is_empty() {
        return Err(degenerate(
            "cv: an explicit `folds` list must be non-empty (>= 2 folds)",
        ));
    }
    // Reject any out-of-range test index up-front: `Pool::select_rows` silently
    // skips OOB indices, so without this guard a typo'd explicit fold index
    // would quietly evaluate on a smaller test set (and a wrong train
    // complement) instead of erroring — diverging from upstream `catboost.cv`,
    // which rejects an out-of-range fold index.
    if let Some(&bad) = list.iter().flatten().find(|&&r| r >= n_rows) {
        return Err(degenerate(format!(
            "cv: explicit `folds` test index {bad} is out of range for a Pool with {n_rows} rows"
        )));
    }
    Ok(list
        .iter()
        .map(|test_idx| {
            let test_set: std::collections::HashSet<usize> = test_idx.iter().copied().collect();
            let train: Vec<usize> = (0..n_rows).filter(|r| !test_set.contains(r)).collect();
            CvFold {
                train,
                test: test_idx.clone(),
            }
        })
        .collect())
}

/// (ORCH-01-S5) k-fold cross-validate `builder`'s parameter set on `pool`,
/// returning per-iteration aggregate metric columns. Composes the S1 partitioner
/// ([`make_cv_folds`] or an explicit `folds` list), the S3 per-fold train +
/// staged-eval loop ([`run_fold`]), and the S4 cross-fold aggregation
/// ([`aggregate_folds`]).
///
/// - `metrics`: metric descriptor strings (ORCH-04 grammar). Empty ⇒ typed error.
/// - `folds`: explicit per-fold TEST-row index sets (`train` = complement);
///   `Some` overrides `fold_count`/`shuffle`/`partition_random_seed`.
///   `Some(&[])` (explicit but EMPTY) is a typed error, validated at entry.
/// - `inverted`: selects [`CvType::Inverted`] (train on the one block) when the
///   fold list is derived from [`make_cv_folds`].
///
/// Per-fold work runs under `rayon` (`par_iter`, order-preserving) ONLY when the
/// `cpu` feature is compiled; under any GPU feature (`wgpu`/`cuda`/`rocm`) fold
/// execution is SERIAL, sidestepping concurrent per-fold `GpuBackend` /
/// `ComputeClient` construction against a single device. The aggregated result
/// is byte-identical either way (each fold's curves are keyed by fold index).
///
/// # Errors
/// [`CatBoostError::UnsupportedModel`] if a fold's model is not scalar/oblivious/
/// float-only (propagated from `staged_predict`); [`CatBoostError::Train`]
/// wrapping [`cb_core::CbError::Degenerate`] for empty `metrics`, an empty
/// explicit `folds` list, a bad partition, fewer than 2 resulting folds,
/// mismatched per-fold iteration counts, or any training/eval failure. Never
/// panics, never returns a `NaN`-bearing `Ok`.
#[allow(clippy::too_many_arguments)]
pub fn cv(
    pool: &Pool,
    builder: &CatBoostBuilder,
    metrics: &[&str],
    fold_count: usize,
    shuffle: bool,
    partition_random_seed: u64,
    inverted: bool,
    folds: Option<&[Vec<usize>]>,
) -> Result<CvResult, CatBoostError> {
    if metrics.is_empty() {
        return Err(degenerate("cv: `metrics` must be non-empty"));
    }

    // Ranking `pairs` cannot survive the per-fold row re-indexing: each fold
    // trains on a `Pool::select_rows` sub-Pool, which DROPS pairs (row re-index
    // would invalidate their object-index ids), so every fold would silently
    // train pair-free — a wrong, not-attempted computation. Reject up-front.
    if !pool.pairs().is_empty() {
        return Err(degenerate(
            "cv: ranking `pairs` are not supported in this numeric/float-only cross-validation slice (Pool::select_rows drops pairs, so each fold would train pair-free); use a group_id-based ranking metric instead",
        ));
    }

    // Build the fold list: an explicit list (train = complement) overrides the
    // partition knobs; otherwise partition via `make_cv_folds`.
    let folds_vec: Vec<CvFold> = match folds {
        Some(list) => folds_from_explicit(list, pool.n_rows())?,
        None => make_cv_folds(
            pool.n_rows(),
            pool.group_id(),
            fold_count,
            shuffle,
            partition_random_seed,
            if inverted {
                CvType::Inverted
            } else {
                CvType::Classical
            },
        )?,
    };

    // Run each fold. `collect::<Result<Vec<_>, _>>()` preserves fold order and
    // short-circuits on the first typed error.
    #[cfg(feature = "cpu")]
    let per_fold: Vec<FoldCurves> = folds_vec
        .par_iter()
        .map(|f| run_fold(pool, f, builder, metrics))
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "cpu"))]
    let per_fold: Vec<FoldCurves> = folds_vec
        .iter()
        .map(|f| run_fold(pool, f, builder, metrics))
        .collect::<Result<Vec<_>, _>>()?;

    // The iteration count is the first fold's first curve length (every fold
    // trains the same fixed `iterations`; `aggregate_folds` rejects any ragged
    // fold). A defined `metrics` slice guarantees at least one curve per fold.
    let iterations = per_fold
        .first()
        .and_then(|f| f.test.first())
        .map_or(0, Vec::len);

    aggregate_folds(&per_fold, metrics, iterations)
}

#[cfg(test)]
#[path = "cv_test.rs"]
mod cv_test;
