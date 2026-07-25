//! Unit tests for the cross-validation module (ORCH-01). Mounted as a CHILD of
//! `crate::cv` (`#[cfg(test)] #[path = "cv_test.rs"] mod cv_test;` inside
//! `cv.rs`) so the tests can reach the module's private helpers
//! (`aggregate_folds`, `FoldCurves`, `run_fold`, …) via `super::`, not only the
//! public `make_cv_folds` surface. The crate-root
//! `#![cfg_attr(test, allow(clippy::unwrap_used, …))]` covers the restriction
//! lints for this file.

use super::{aggregate_folds, cv, make_cv_folds, run_fold, CvFold, CvType, FoldCurves};
use crate::{eval_metric, CatBoostBuilder, EBootstrapType, IngestSource, OwnedColumns, Pool};

/// Sorted concatenation of every fold's `test` block.
fn all_test_sorted(folds: &[CvFold]) -> Vec<usize> {
    let mut v: Vec<usize> = folds.iter().flat_map(|f| f.test.iter().copied()).collect();
    v.sort_unstable();
    v
}

// -------------------------------------------------------------------------
// ORCH-01-S1 — make_cv_folds
// -------------------------------------------------------------------------

#[test]
fn plain_folds_partition() {
    let folds = make_cv_folds(6, &[], 3, false, 0, CvType::Classical).unwrap();
    assert_eq!(folds.len(), 3);

    // Concatenated + sorted test sets cover 0..6 exactly once.
    assert_eq!(all_test_sorted(&folds), (0..6).collect::<Vec<_>>());

    // Each pair of test sets is disjoint, and each train == complement of test.
    for (i, fi) in folds.iter().enumerate() {
        for (j, fj) in folds.iter().enumerate() {
            if i != j {
                assert!(
                    fi.test.iter().all(|x| !fj.test.contains(x)),
                    "test sets {i} and {j} overlap"
                );
            }
        }
        let mut complement: Vec<usize> = (0..6).filter(|x| !fi.test.contains(x)).collect();
        complement.sort_unstable();
        let mut train = fi.train.clone();
        train.sort_unstable();
        assert_eq!(train, complement, "fold {i} train != complement of test");
    }
}

#[test]
fn shuffle_determinism() {
    let a = make_cv_folds(10, &[], 5, true, 42, CvType::Classical).unwrap();
    let b = make_cv_folds(10, &[], 5, true, 42, CvType::Classical).unwrap();
    assert_eq!(a, b, "same (n, seed) must yield identical folds");

    let c = make_cv_folds(10, &[], 5, true, 43, CvType::Classical).unwrap();
    assert_ne!(a, c, "a different seed must yield a different partition");

    // A shuffled partition still covers 0..10 exactly once.
    assert_eq!(all_test_sorted(&a), (0..10).collect::<Vec<_>>());
}

#[test]
fn grouped_whole_group() {
    // 3 groups of 2 rows each, 3 folds => one whole group per fold.
    let group_id = [0u64, 0, 1, 1, 2, 2];
    let folds = make_cv_folds(6, &group_id, 3, false, 0, CvType::Classical).unwrap();
    assert_eq!(folds.len(), 3);

    let group_of = |row: usize| -> u64 { *group_id.get(row).unwrap() };
    for f in &folds {
        // Every test set is exactly the rows of a single group (no split).
        let groups: std::collections::HashSet<u64> = f.test.iter().map(|&r| group_of(r)).collect();
        assert_eq!(groups.len(), 1, "a fold's test spans more than one group");
        assert_eq!(f.test.len(), 2, "each group has 2 rows");
    }
    assert_eq!(all_test_sorted(&folds), (0..6).collect::<Vec<_>>());
}

#[test]
fn grouped_multi_group_per_fold() {
    // 6 groups of 2 rows, 2 folds => each fold's test spans 3 whole groups.
    let group_id: Vec<u64> = (0..6u64).flat_map(|g| [g, g]).collect();
    let n = group_id.len();
    let folds = make_cv_folds(n, &group_id, 2, false, 0, CvType::Classical).unwrap();
    assert_eq!(folds.len(), 2);

    let group_of = |row: usize| -> u64 { *group_id.get(row).unwrap() };
    for f in &folds {
        // Multiple whole groups per fold; each group's rows contiguous+unsplit.
        let groups: std::collections::HashSet<u64> = f.test.iter().map(|&r| group_of(r)).collect();
        assert!(groups.len() >= 2, "expected multiple groups per fold");
        // For each group present in the fold, ALL of its rows are present.
        for g in groups {
            let want: Vec<usize> = (0..n).filter(|&r| group_of(r) == g).collect();
            let got: Vec<usize> = f.test.iter().copied().filter(|&r| group_of(r) == g).collect();
            assert_eq!(got, want, "group {g} was split across folds");
        }
    }
    assert_eq!(all_test_sorted(&folds), (0..n).collect::<Vec<_>>());
}

#[test]
fn shuffle_grouped_permutes_spans_not_rows() {
    // 6 groups of 2 rows, 3 folds.
    let group_id: Vec<u64> = (0..6u64).flat_map(|g| [g, g]).collect();
    let n = group_id.len();

    let plain = make_cv_folds(n, &group_id, 3, false, 0, CvType::Classical).unwrap();
    let shuffled = make_cv_folds(n, &group_id, 3, true, 7, CvType::Classical).unwrap();

    // (a) group ASSIGNMENT to folds differs (spans permuted).
    let assignment = |folds: &[CvFold]| -> Vec<std::collections::BTreeSet<u64>> {
        folds
            .iter()
            .map(|f| f.test.iter().map(|&r| *group_id.get(r).unwrap()).collect())
            .collect()
    };
    assert_ne!(
        assignment(&plain),
        assignment(&shuffled),
        "shuffle should permute which groups land in which fold"
    );

    // (b) within-group row order preserved: each group's rows appear ascending.
    for f in &shuffled {
        for g in 0..6u64 {
            let rows: Vec<usize> = f
                .test
                .iter()
                .copied()
                .filter(|&r| *group_id.get(r).unwrap() == g)
                .collect();
            let mut sorted = rows.clone();
            sorted.sort_unstable();
            assert_eq!(rows, sorted, "group {g} rows were reordered");
        }
    }

    // (c) determinism holds for grouped mode too.
    let again = make_cv_folds(n, &group_id, 3, true, 7, CvType::Classical).unwrap();
    assert_eq!(shuffled, again, "grouped shuffle must be deterministic");

    // Coverage is preserved regardless.
    assert_eq!(all_test_sorted(&shuffled), (0..n).collect::<Vec<_>>());
}

#[test]
fn inverted_swaps_roles() {
    let classical = make_cv_folds(6, &[], 3, false, 0, CvType::Classical).unwrap();
    let inverted = make_cv_folds(6, &[], 3, false, 0, CvType::Inverted).unwrap();
    assert_eq!(classical.len(), inverted.len());
    for (c, i) in classical.iter().zip(inverted.iter()) {
        // Inverted fold k's TRAIN == Classical fold k's TEST.
        assert_eq!(i.train, c.test, "inverted train != classical test");
        assert_eq!(i.test, c.train, "inverted test != classical train");
    }
}

// -------------------------------------------------------------------------
// ORCH-01-S4 — aggregate_folds
// -------------------------------------------------------------------------

/// A [`FoldCurves`] with a single metric's `test` curve (train zeroed to the
/// same length), for aggregation tests that only inspect the `test-` columns.
fn test_only_fold(test_curve: Vec<f64>) -> FoldCurves {
    let train = vec![0.0_f64; test_curve.len()];
    FoldCurves {
        test: vec![test_curve],
        train: vec![train],
    }
}

#[test]
fn aggregate_two_folds() {
    // Two folds, one metric, two iterations: [[1,2],[3,4]].
    let per_fold = [test_only_fold(vec![1.0, 2.0]), test_only_fold(vec![3.0, 4.0])];
    let result = aggregate_folds(&per_fold, &["RMSE"], 2).unwrap();

    assert_eq!(result.iterations, vec![0, 1]);

    let mean = result.columns.get("test-RMSE-mean").unwrap();
    let std = result.columns.get("test-RMSE-std").unwrap();
    // mean: (1+3)/2 = 2, (2+4)/2 = 3.
    assert!((mean.first().copied().unwrap() - 2.0).abs() < 1e-12);
    assert!((mean.get(1).copied().unwrap() - 3.0).abs() < 1e-12);
    // sample std (ddof=1): sqrt((1+1)/1) = sqrt(2) at both iterations.
    let want = 2.0_f64.sqrt();
    assert!((std.first().copied().unwrap() - want).abs() < 1e-12);
    assert!((std.get(1).copied().unwrap() - want).abs() < 1e-12);

    // The train columns exist and are the correct shape.
    assert!(result.columns.contains_key("train-RMSE-mean"));
    assert!(result.columns.contains_key("train-RMSE-std"));
}

#[test]
fn aggregate_single_fold_errs() {
    // F == 1 has no defined sample std.
    let per_fold = [test_only_fold(vec![1.0, 2.0])];
    assert!(aggregate_folds(&per_fold, &["RMSE"], 2).is_err());
}

#[test]
fn aggregate_zero_folds_errs() {
    // F == 0 must be a typed error, NEVER a NaN-bearing Ok.
    let per_fold: [FoldCurves; 0] = [];
    let res = aggregate_folds(&per_fold, &["RMSE"], 2);
    assert!(res.is_err(), "empty per_fold slice must error, not produce NaN");
}

#[test]
fn aggregate_ragged_errs() {
    // Folds with mismatched curve lengths cannot be aggregated.
    let per_fold = [test_only_fold(vec![1.0, 2.0]), test_only_fold(vec![3.0])];
    assert!(aggregate_folds(&per_fold, &["RMSE"], 2).is_err());
}

#[test]
fn errors() {
    // fold_count = 1 (< 2).
    assert!(make_cv_folds(6, &[], 1, false, 0, CvType::Classical).is_err());
    // fold_count > n.
    assert!(make_cv_folds(6, &[], 7, false, 0, CvType::Classical).is_err());
    // n == 0.
    assert!(make_cv_folds(0, &[], 2, false, 0, CvType::Classical).is_err());
    // non-contiguous group_id = [0, 1, 0].
    assert!(make_cv_folds(3, &[0u64, 1, 0], 2, false, 0, CvType::Classical).is_err());
}

// -------------------------------------------------------------------------
// ORCH-01-S3 — run_fold
// -------------------------------------------------------------------------

/// A small numeric-regression Pool: 2 float features over 8 rows, plus a label.
fn small_regression_pool() -> Pool {
    let f0: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let f1: Vec<f64> = vec![7.0, 5.0, 6.0, 2.0, 3.0, 1.0, 4.0, 0.0];
    let label: Vec<f64> = vec![1.0, 2.0, 2.5, 3.0, 3.5, 5.0, 6.0, 6.5];
    OwnedColumns::new(vec![f0, f1], label)
        .into_pool()
        .expect("small regression pool builds")
}

#[test]
fn run_fold_matches_manual() {
    let pool = small_regression_pool();
    let builder = CatBoostBuilder::new().iterations(5).depth(2);
    let fold = CvFold {
        train: vec![0, 1, 2, 3, 4, 5],
        test: vec![6, 7],
    };

    let curves = run_fold(&pool, &fold, &builder, &["RMSE"]).unwrap();

    // Manual composition through the exact same seams.
    let train_pool = pool.select_rows(&fold.train);
    let test_pool = pool.select_rows(&fold.test);
    let model = builder.fit(&train_pool).unwrap();
    let test_stages = model
        .staged_predict(&test_pool, None, None, None)
        .unwrap();

    let manual: Vec<f64> = test_stages
        .iter()
        .map(|approx| {
            eval_metric(
                test_pool.label(),
                approx,
                "RMSE",
                Some(test_pool.weights()),
                Some(test_pool.group_id()),
            )
            .unwrap()
        })
        .collect();

    let got = curves.test.first().expect("one metric curve");
    assert_eq!(got.len(), manual.len(), "stage count matches");
    for (g, m) in got.iter().zip(manual.iter()) {
        assert!((g - m).abs() < 1e-5, "run_fold test curve diverged: {g} vs {m}");
    }
}

#[test]
fn run_fold_degenerate_zero_row_errs() {
    let pool = small_regression_pool();
    let builder = CatBoostBuilder::new().iterations(5).depth(2);

    // Empty TRAIN => errors inside fit (empty-target guard).
    let empty_train = CvFold {
        train: vec![],
        test: vec![0, 1],
    };
    assert!(
        run_fold(&pool, &empty_train, &builder, &["RMSE"]).is_err(),
        "empty train fold must be a typed error, never NaN"
    );

    // Empty TEST => fit succeeds, but eval_metric rejects the empty approx.
    let empty_test = CvFold {
        train: vec![0, 1, 2, 3],
        test: vec![],
    };
    assert!(
        run_fold(&pool, &empty_test, &builder, &["RMSE"]).is_err(),
        "empty test fold must be a typed error, never NaN"
    );
}

// -------------------------------------------------------------------------
// ORCH-01-S5 — cv (facade orchestration)
// -------------------------------------------------------------------------

#[test]
fn cv_empty_metrics_errs() {
    let pool = small_regression_pool();
    let builder = CatBoostBuilder::new().iterations(3).depth(2);
    let folds: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
    assert!(
        cv(&pool, &builder, &[], 3, false, 0, false, Some(&folds)).is_err(),
        "empty metrics must be a typed error"
    );
}

#[test]
fn cv_empty_explicit_folds_errs() {
    let pool = small_regression_pool();
    let builder = CatBoostBuilder::new().iterations(3).depth(2);
    let empty: [Vec<usize>; 0] = [];
    let res = cv(&pool, &builder, &["RMSE"], 3, false, 0, false, Some(&empty));
    assert!(
        res.is_err(),
        "an explicit but empty folds list must error, never a NaN-bearing Ok"
    );
}

#[test]
fn cv_pool_with_pairs_errs() {
    // A Pool carrying at least one ranking Pair must be rejected by cv() up-front:
    // `Pool::select_rows` DROPS pairs, so each fold would silently train
    // pair-free. Build the pool via the public ingest seam
    // (`OwnedColumns::with_pairs`).
    use cb_data::Pair;
    let f0: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let f1: Vec<f64> = vec![7.0, 5.0, 6.0, 2.0, 3.0, 1.0, 4.0, 0.0];
    let label: Vec<f64> = vec![1.0, 2.0, 2.5, 3.0, 3.5, 5.0, 6.0, 6.5];
    let pool = OwnedColumns::new(vec![f0, f1], label)
        .with_pairs(vec![Pair {
            winner_id: 0,
            loser_id: 1,
        }])
        .into_pool()
        .expect("in-range pairs build a Pool");
    assert!(
        !pool.pairs().is_empty(),
        "sanity: the constructed Pool carries a ranking pair"
    );

    let builder = CatBoostBuilder::new().iterations(3).depth(2);
    let folds: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
    let res = cv(&pool, &builder, &["RMSE"], 3, false, 0, false, Some(&folds));
    assert!(
        res.is_err(),
        "a Pool with ranking pairs must be a typed error (pairs unsupported in the numeric cv slice)"
    );
}

#[test]
fn cv_explicit_oob_fold_index_errs() {
    // An explicit folds list with a test index >= n_rows must error up-front,
    // not silently evaluate on a smaller test set (regression: `select_rows`
    // skips OOB indices, so cv() would quietly report wrong per-fold metrics).
    let pool = small_regression_pool();
    let n = pool.n_rows();
    let builder = CatBoostBuilder::new().iterations(3).depth(2);
    let folds = [vec![0_usize, 1], vec![2, n + 5]]; // n+5 is out of range
    let res = cv(&pool, &builder, &["RMSE"], 2, false, 0, false, Some(&folds));
    assert!(
        res.is_err(),
        "an out-of-range explicit fold index must be a typed error"
    );
}

/// AT-S5b: serial and `cpu`-parallel (rayon) fold execution produce a
/// byte-identical [`super::CvResult`] (`PartialEq`), not merely ≤1e-5-close. The
/// parallel path is the public [`cv`]; the serial path recomputes the same
/// private `run_fold` → `aggregate_folds` chain directly (reachable via the
/// child-module `super::` mount).
#[cfg(feature = "cpu")]
#[test]
fn cv_serial_vs_parallel_byte_identical() {
    let pool = small_regression_pool();
    let builder = CatBoostBuilder::new()
        .iterations(5)
        .depth(2)
        .boost_from_average(true)
        .bootstrap_type(EBootstrapType::No);
    let folds_list: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3], vec![4, 5]];

    // Parallel path (public cv, rayon under `cpu`).
    let parallel = cv(&pool, &builder, &["RMSE"], 3, false, 0, false, Some(&folds_list)).unwrap();

    // Serial recomputation through the identical private seams.
    let n = pool.n_rows();
    let cv_folds: Vec<CvFold> = folds_list
        .iter()
        .map(|test_idx| {
            let set: std::collections::HashSet<usize> = test_idx.iter().copied().collect();
            CvFold {
                train: (0..n).filter(|r| !set.contains(r)).collect(),
                test: test_idx.clone(),
            }
        })
        .collect();
    let per_fold: Vec<FoldCurves> = cv_folds
        .iter()
        .map(|f| run_fold(&pool, f, &builder, &["RMSE"]).unwrap())
        .collect();
    let iterations = per_fold.first().and_then(|f| f.test.first()).map(Vec::len).unwrap();
    let serial = aggregate_folds(&per_fold, &["RMSE"], iterations).unwrap();

    assert_eq!(
        parallel, serial,
        "serial and cpu-parallel CvResult must be byte-identical"
    );
}
