//! Unit tests for the owned-`Vec` ingestion primitive
//! ([`crate::ingest::OwnedColumns`]).
//!
//! Focused on the validation seam: consistent columns build a `Pool`,
//! inconsistent columns return a typed [`cb_core::CbError`] (never a panic /
//! out-of-bounds — threats T-02-04 / T-02-05). Kept in a dedicated `*_test.rs`
//! file per the source/test separation rule (D-17).

use cb_core::CbError;

use crate::ingest::{IngestSource, OwnedColumns};
use crate::Pair;

/// All column kinds at matching lengths materialize a `Pool` with the right
/// shape.
#[test]
fn all_column_kinds_consistent_lengths_build_pool() {
    let pool = OwnedColumns::new(vec![vec![1.0, 2.0]], vec![0.0, 1.0])
        .with_cat_features(vec![vec!["a".to_owned(), "b".to_owned()]])
        .with_text_features(vec![vec!["hi".to_owned(), "yo".to_owned()]])
        .with_embedding_features(vec![vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]]])
        .with_subgroup_id(vec![1, 1])
        .with_baseline(vec![0.1, 0.2])
        .into_pool()
        .expect("all-consistent columns must build");

    assert_eq!(pool.n_rows(), 2);
    assert_eq!(pool.n_cat_features(), 1);
    assert_eq!(pool.n_text_features(), 1);
    assert_eq!(pool.n_embedding_features(), 1);
    assert_eq!(pool.subgroup_id(), &[1, 1]);
    assert_eq!(pool.baseline(), &[0.1, 0.2]);
}

/// A categorical column shorter than `n_rows` is a typed error, not a panic.
#[test]
fn cat_feature_length_mismatch_is_error() {
    let result = OwnedColumns::new(vec![vec![1.0, 2.0, 3.0]], vec![0.0, 1.0, 0.0])
        .with_cat_features(vec![vec!["a".to_owned()]])
        .into_pool();

    match result {
        Err(CbError::OutOfRange(msg)) => assert!(msg.contains("cat_feature")),
        other => panic!("expected OutOfRange for short cat column, got {other:?}"),
    }
}

/// An embedding column shorter than `n_rows` is a typed error.
#[test]
fn embedding_feature_length_mismatch_is_error() {
    let result = OwnedColumns::new(vec![vec![1.0, 2.0]], vec![0.0, 1.0])
        .with_embedding_features(vec![vec![vec![1.0_f32]]])
        .into_pool();

    assert!(matches!(result, Err(CbError::OutOfRange(_))));
}

/// An all-empty source (no features, no label) is a legitimate zero-row Pool,
/// not an error.
#[test]
fn empty_source_builds_zero_row_pool() {
    let pool = OwnedColumns::default()
        .into_pool()
        .expect("empty source is a valid zero-row Pool");

    assert_eq!(pool.n_rows(), 0);
    assert_eq!(pool.n_float_features(), 0);
}

/// WR-02: ranking pairs whose ids are all within `n_rows` build a `Pool`.
#[test]
fn pairs_within_n_rows_build_pool() {
    let pool = OwnedColumns::new(vec![vec![1.0, 2.0, 3.0]], vec![0.0, 1.0, 0.0])
        .with_pairs(vec![
            Pair { winner_id: 0, loser_id: 2 },
            Pair { winner_id: 1, loser_id: 0 },
        ])
        .into_pool()
        .expect("in-range pairs must build");

    assert_eq!(pool.n_rows(), 3);
}

/// WR-02: a pair referencing a non-existent row (`id >= n_rows`) is a typed
/// `OutOfRange` error, not a latent out-of-bounds.
#[test]
fn pair_winner_id_out_of_range_is_rejected() {
    let result = OwnedColumns::new(vec![vec![1.0, 2.0, 3.0]], vec![0.0, 1.0, 0.0])
        .with_pairs(vec![Pair { winner_id: 5, loser_id: 1 }])
        .into_pool();

    match result {
        Err(CbError::OutOfRange(msg)) => {
            assert!(msg.contains("winner_id"), "msg: {msg}");
            assert!(msg.contains('5'), "msg: {msg}");
        }
        other => panic!("expected OutOfRange for OOB winner_id, got {other:?}"),
    }
}

/// WR-02: the `loser_id` is bounds-checked symmetrically with `winner_id`.
#[test]
fn pair_loser_id_out_of_range_is_rejected() {
    let result = OwnedColumns::new(vec![vec![1.0, 2.0]], vec![0.0, 1.0])
        .with_pairs(vec![Pair { winner_id: 0, loser_id: 2 }])
        .into_pool();

    match result {
        Err(CbError::OutOfRange(msg)) => assert!(msg.contains("loser_id"), "msg: {msg}"),
        other => panic!("expected OutOfRange for OOB loser_id, got {other:?}"),
    }
}

/// `n_rows` is taken from the float features when present, and the label is
/// validated against it.
#[test]
fn n_rows_derives_from_float_features() {
    let result = OwnedColumns::new(vec![vec![1.0, 2.0, 3.0, 4.0]], vec![0.0])
        .into_pool();

    match result {
        Err(CbError::OutOfRange(msg)) => assert!(msg.contains("label")),
        other => panic!("expected label length error, got {other:?}"),
    }
}
