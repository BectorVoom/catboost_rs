//! Unit tests for the hyperparameter-search module (ORCH-02). Mounted at the
//! crate root via `#[cfg(test)] mod grid_search_test;` (the facade's root-level
//! test-mount idiom, cf. `mod metrics_test;`). The module-internal items are
//! reached through their `crate::grid_search::` path (the `pub` / `pub(crate)`
//! surface); the crate-root `#![cfg_attr(test, allow(clippy::unwrap_used, …))]`
//! covers the restriction lints for this file.

use std::collections::BTreeMap;
use std::sync::Arc;

use cb_compute::{CustomMetric, CustomMetricHandle};
use cb_core::CbResult;
use cb_train::{fisher_yates_permutation, AucType, DcgDenominator, DcgMetricType, EvalMetric};

use crate::cv::CvResult;
use crate::grid_search::{
    metric_is_max_optimal, sample_indices, score_candidate, select_best, select_index,
};

/// Build a hand-made [`CvResult`] from `(column_key, per_iteration_values)`
/// pairs (the `iterations` index is irrelevant to scoring).
fn cvres(pairs: &[(&str, Vec<f64>)]) -> CvResult {
    let mut columns = BTreeMap::new();
    for (k, v) in pairs {
        columns.insert((*k).to_owned(), v.clone());
    }
    CvResult {
        iterations: Vec::new(),
        columns,
    }
}

/// A tiny custom-metric stub whose `is_max_optimal()` returns `true`, to prove
/// [`metric_is_max_optimal`] delegates to the trait for `EvalMetric::Custom`.
struct MaxStub;

impl CustomMetric for MaxStub {
    fn evaluate(&self, _approxes: &[f64], _target: &[f64], _weight: &[f64]) -> CbResult<(f64, f64)> {
        Ok((0.0, 0.0))
    }

    fn get_final_error(&self, _error: f64, _weight: f64) -> f64 {
        0.0
    }

    fn is_max_optimal(&self) -> bool {
        true
    }
}

// -------------------------------------------------------------------------
// ORCH-02-S1 — metric_is_max_optimal
// -------------------------------------------------------------------------

#[test]
fn direction_min_metrics() {
    for m in [
        EvalMetric::Rmse,
        EvalMetric::Logloss,
        EvalMetric::Msle,
        EvalMetric::Mae,
        EvalMetric::Mape,
        EvalMetric::Quantile { alpha: 0.5 },
    ] {
        assert!(
            !metric_is_max_optimal(&m),
            "{m:?} is Min-optimized (smaller is better)"
        );
    }
}

#[test]
fn direction_max_metrics() {
    for m in [
        EvalMetric::Ndcg {
            top: -1,
            dcg_type: DcgMetricType::Base,
            denominator: DcgDenominator::LogPosition,
        },
        EvalMetric::Map {
            top: -1,
            border: 0.5,
        },
        EvalMetric::Mrr {
            top: -1,
            border: 0.5,
        },
        EvalMetric::QueryAuc {
            auc_type: AucType::Classic,
        },
    ] {
        assert!(
            metric_is_max_optimal(&m),
            "{m:?} is Max-optimized (larger is better)"
        );
    }
}

#[test]
fn direction_custom_delegates() {
    let m = EvalMetric::Custom(CustomMetricHandle::new(Arc::new(MaxStub)));
    assert!(
        metric_is_max_optimal(&m),
        "Custom must delegate to CustomMetric::is_max_optimal (== true here)"
    );
}

// -------------------------------------------------------------------------
// ORCH-02-S2 — score_candidate + select_best
// -------------------------------------------------------------------------

#[test]
fn score_min_metric() {
    let cv = cvres(&[("test-RMSE-mean", vec![0.9, 0.5, 0.6])]);
    let s = score_candidate(&cv, "RMSE").unwrap();
    assert!((s - 0.5).abs() < 1e-12, "RMSE minimizes over iterations; got {s}");
}

#[test]
fn score_max_metric() {
    let cv = cvres(&[("test-NDCG-mean", vec![0.2, 0.8, 0.6])]);
    let s = score_candidate(&cv, "NDCG").unwrap();
    assert!((s - 0.8).abs() < 1e-12, "NDCG maximizes over iterations; got {s}");
}

#[test]
fn select_best_argmin_and_tiebreak() {
    let v = [
        cvres(&[("test-RMSE-mean", vec![0.5])]),
        cvres(&[("test-RMSE-mean", vec![0.4])]),
        cvres(&[("test-RMSE-mean", vec![0.7])]),
    ];
    assert_eq!(select_best(&v, "RMSE").unwrap(), 1, "argmin over candidates");

    let tie = [
        cvres(&[("test-RMSE-mean", vec![0.4])]),
        cvres(&[("test-RMSE-mean", vec![0.4])]),
        cvres(&[("test-RMSE-mean", vec![0.7])]),
    ];
    assert_eq!(
        select_best(&tie, "RMSE").unwrap(),
        0,
        "ties resolve to the LOWEST index"
    );
}

#[test]
fn score_all_non_finite_column_errs() {
    // A primary-metric column that is non-finite at every iteration must NOT
    // reduce to a finite sentinel (regression: `fold(f64::MIN, f64::max)`
    // returned `Some(f64::MIN)` for an all-NaN column, defeating the guard and
    // letting a garbage "best" model be selected/refit).
    let nan = cvres(&[("test-RMSE-mean", vec![f64::NAN, f64::NAN])]);
    assert!(
        score_candidate(&nan, "RMSE").is_err(),
        "an all-non-finite column must be a typed error, not a f64::MIN sentinel"
    );
    // Max-optimal direction has the mirror bug (f64::MAX seed) — cover it too.
    let nan_max = cvres(&[("test-NDCG-mean", vec![f64::NAN])]);
    assert!(
        score_candidate(&nan_max, "NDCG").is_err(),
        "an all-non-finite max-optimal column must also be a typed error"
    );
    // A partially-NaN column still returns the best FINITE value (isolated
    // non-finite entries are skipped, not propagated).
    let mixed = cvres(&[("test-RMSE-mean", vec![f64::NAN, 0.3, f64::INFINITY])]);
    let s = score_candidate(&mixed, "RMSE").unwrap();
    assert!(
        (s - 0.3).abs() < 1e-12,
        "isolated non-finite entries are skipped; got {s}"
    );
}

#[test]
fn score_missing_column_errs() {
    // A CvResult lacking the `test-RMSE-mean` column.
    let cv = cvres(&[("test-NDCG-mean", vec![0.4])]);
    assert!(
        score_candidate(&cv, "RMSE").is_err(),
        "a missing column must be a typed error"
    );

    let empty: [CvResult; 0] = [];
    assert!(
        select_best(&empty, "RMSE").is_err(),
        "an empty cv_results slice must be a typed error"
    );
}

// -------------------------------------------------------------------------
// error_score — select_index (tolerant selection ignoring non-finite scores)
// -------------------------------------------------------------------------

#[test]
fn select_index_ignores_nan_lowest_tie() {
    // A NaN entry (a `NaN` error_score) is skipped; the first finite argmin wins
    // ties on the LOWEST index.
    assert_eq!(
        select_index(&[f64::NAN, 0.5, 0.5], false),
        Some(1),
        "NaN is skipped; the first (lowest-index) finite min wins ties"
    );
    // All entries non-finite ⇒ no scoreable candidate.
    assert_eq!(
        select_index(&[f64::NAN, f64::NAN], true),
        None,
        "an all-non-finite score slice yields no selection"
    );
    // An empty slice is likewise `None` (never a panic).
    assert_eq!(select_index(&[], false), None, "empty scores ⇒ None");
    // A finite value competes normally against a NaN and wins.
    assert_eq!(
        select_index(&[f64::NAN, -1.0, 0.7], false),
        Some(1),
        "a finite (numeric error_score) value competes and wins over NaN"
    );
}

// -------------------------------------------------------------------------
// ORCH-02-S4 — sample_indices (deterministic subsampling; pure, no training)
// -------------------------------------------------------------------------

/// `fisher_yates_permutation(m, seed)` cast to `usize`.
fn perm(m: usize, seed: u64) -> Vec<usize> {
    fisher_yates_permutation(m, seed)
        .into_iter()
        .map(|i| i as usize)
        .collect()
}

/// A sorted copy (compare index SETS irrespective of order).
fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

#[test]
fn sample_is_first_n_of_permutation() {
    let full = perm(6, 0);
    let expected: Vec<usize> = full.into_iter().take(3).collect();
    assert_eq!(
        sample_indices(6, 3, 0),
        expected,
        "the sample is the first n_iter of fisher_yates_permutation(m, seed)"
    );
}

#[test]
fn sample_deterministic() {
    assert_eq!(
        sample_indices(6, 3, 0),
        sample_indices(6, 3, 0),
        "same (m, n_iter, seed) ⇒ identical sample"
    );
    assert_ne!(
        sorted(sample_indices(6, 3, 0)),
        sorted(sample_indices(6, 3, 7)),
        "a different seed generally yields a different subset"
    );
}

#[test]
fn sample_full_when_niter_ge_m() {
    assert_eq!(
        sorted(sample_indices(6, 6, 0)),
        (0..6).collect::<Vec<_>>(),
        "n_iter == m ⇒ the sample is all m indices"
    );
    assert_eq!(
        sorted(sample_indices(6, 10, 0)),
        (0..6).collect::<Vec<_>>(),
        "n_iter > m ⇒ still all m indices (min(n_iter, m))"
    );
}

#[test]
fn sample_zero_or_empty_errs() {
    // The pure-helper guard signal the facade turns into a typed error:
    // n_iter == 0 or m == 0 ⇒ an empty sample (never a panic / OOB).
    assert!(
        sample_indices(6, 0, 0).is_empty(),
        "n_iter == 0 ⇒ empty sample"
    );
    assert!(sample_indices(0, 3, 0).is_empty(), "m == 0 ⇒ empty sample");
}

/// F14 test fn 2 (SPEC-CATF-Δ6) — `grid_search` on a categorical pool FAILS
/// FAST rather than degrading to an all-NaN `SearchResult`.
///
/// The hazard this closes is not merely an error-vs-ok difference. With
/// `ErrorScore::Value(NaN)` (the sklearn default), EVERY candidate would fail
/// from inside its fold, `warn_fit_failed` would emit a warning, and a
/// `SearchResult` with all-NaN scores and an arbitrary `best_index` would be
/// RETURNED — a silent degradation, not an error. A categorical pool is a
/// CONFIGURATION failure, not a candidate failure, so converting it into
/// `error_score` is category confusion.
#[test]
fn grid_search_on_a_categorical_pool_fails_fast_not_as_all_nan_scores() {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let cat: Vec<String> = (0..n).map(|i| format!("c{}", i % 9)).collect();
    let floats = vec![(0..n).map(|i| (i % 7) as f64).collect::<Vec<f64>>()];
    use crate::IngestSource as _;
    let pool = crate::OwnedColumns::new(floats, label)
        .with_cat_features(vec![cat])
        .into_pool()
        .expect("categorical pool builds");

    let candidates = vec![
        crate::CatBoostBuilder::new().iterations(3).depth(2),
        crate::CatBoostBuilder::new().iterations(4).depth(3),
    ];

    let result = crate::grid_search::grid_search(
        &pool,
        &candidates,
        &["RMSE"],
        3,
        false,
        0,
        false,
        None,
        false,
        crate::grid_search::ErrorScore::Value(f64::NAN),
    );

    match result {
        Err(crate::CatBoostError::UnsupportedModel(msg)) => {
            assert!(
                msg.contains("categorical"),
                "the error must name the categorical columns, got: {msg}"
            );
        }
        Err(other) => panic!("expected UnsupportedModel, got {other:?}"),
        Ok(_) => panic!(
            "grid_search SILENTLY DEGRADED a categorical pool to an all-NaN SearchResult"
        ),
    }
}
