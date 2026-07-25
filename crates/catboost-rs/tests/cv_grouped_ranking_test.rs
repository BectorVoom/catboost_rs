//! ORCH-01-S5 (AT-S5c): the FULL `catboost_rs::cv` pipeline runs on GROUPED
//! folds with a RANKING metric (NDCG) without tripping `eval_grouped`'s
//! non-contiguous-`group_id` rejection, and returns correctly-shaped
//! `test-NDCG-mean`/`-std` + `train-NDCG-mean`/`-std` columns.
//!
//! This is a SHAPE / no-error / contiguity test, NOT a ≤1e-5 oracle: it exists
//! to prove that `make_cv_folds`'s grouped partition keeps every group's rows
//! contiguous through `select_rows`, so a ranking metric's `eval_grouped`
//! precondition holds end-to-end (a prior MAJOR gap: grouped k-fold was in-scope
//! but had zero end-to-end pipeline coverage). It is therefore unaffected by the
//! separate training-parity blocker on `cv_oracle_test`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use catboost_rs::{cv, CatBoostBuilder, IngestSource, OwnedColumns, Pool};

/// A small grouped ranking Pool: 4 groups × 3 rows (contiguous `group_id`),
/// 2 float features, graded relevance labels in {0,1,2}.
fn grouped_pool() -> Pool {
    // 12 rows, 4 contiguous groups of 3.
    let group_id: Vec<u64> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let f0: Vec<f64> = vec![0.1, 0.9, 0.5, 0.2, 0.8, 0.4, 0.3, 0.7, 0.6, 0.15, 0.85, 0.45];
    let f1: Vec<f64> = vec![0.8, 0.2, 0.5, 0.7, 0.3, 0.6, 0.9, 0.1, 0.4, 0.75, 0.25, 0.55];
    // Graded relevance within each group (0..2).
    let label: Vec<f64> = vec![0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0];
    OwnedColumns::new(vec![f0, f1], label)
        .with_group_id(group_id)
        .into_pool()
        .expect("grouped pool builds")
}

/// AT-S5c: grouped folds + the NDCG ranking metric run through the whole cv()
/// pipeline with no error and correctly-shaped columns.
///
/// SUBSTITUTION NOTE (documented, per the plan's fallback clause): the model is
/// trained with the DEFAULT RMSE loss (a scalar/oblivious/float-only model that
/// `staged_predict` accepts), while the EVALUATED metric is the RANKING metric
/// `"NDCG"`. This is a deliberate choice, not a workaround around an unsupported
/// path: the thing under test is that grouped CV folds preserve each group's row
/// contiguity so that NDCG's `eval_grouped` (which REQUIRES contiguous
/// `group_id`) does not error end-to-end. The metric — not the training loss — is
/// what routes through `eval_grouped`, so a ranking metric over grouped folds
/// exercises exactly the contiguity precondition regardless of the training loss.
/// (A ranking training loss such as `QueryRmse` would add training-path surface
/// without adding coverage of the grouped-contiguity path this test targets.)
#[test]
fn cv_grouped_ranking_metric_no_contiguity_error() {
    let pool = grouped_pool();
    let builder = CatBoostBuilder::new().iterations(4).depth(2);

    // group_id is non-empty ⇒ make_cv_folds takes the grouped path; 4 groups,
    // fold_count=2 ⇒ each fold tests 2 whole groups.
    let result = cv(&pool, &builder, &["NDCG"], 2, false, 0, false, None)
        .expect("grouped cv with a ranking metric must not trip eval_grouped");

    assert_eq!(result.iterations.len(), 4, "one row per training iteration");
    for column in [
        "test-NDCG-mean",
        "test-NDCG-std",
        "train-NDCG-mean",
        "train-NDCG-std",
    ] {
        let col = result
            .columns
            .get(column)
            .unwrap_or_else(|| panic!("missing column `{column}`"));
        assert_eq!(col.len(), 4, "column `{column}` has one value per iteration");
        assert!(
            col.iter().all(|v| v.is_finite()),
            "column `{column}` must be finite (no NaN from a contiguity failure)"
        );
    }
}
