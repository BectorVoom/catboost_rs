//! Unit tests for [`crate::Pool`] — built through the owned-`Vec` ingestion
//! seam ([`crate::ingest::OwnedColumns`]).
//!
//! Kept in a dedicated `*_test.rs` file per the source/test separation rule
//! (D-17); no `#[cfg(test)] mod` lives in `pool.rs`.

use cb_core::CbError;

use crate::ingest::{IngestSource, OwnedColumns};
use crate::pool::Pair;

/// A Pool built from owned numeric columns + label exposes the expected
/// per-column lengths and metadata accessors.
#[test]
fn pool_exposes_column_lengths_and_metadata() {
    let f0 = vec![1.0, 2.0, 3.0];
    let f1 = vec![10.0, 20.0, 30.0];
    let label = vec![0.0, 1.0, 0.0];

    let pool = OwnedColumns::new(vec![f0, f1], label)
        .with_weights(vec![1.0, 0.5, 2.0])
        .with_group_id(vec![7, 7, 8])
        .into_pool()
        .expect("consistent-length columns must build a Pool");

    assert_eq!(pool.n_rows(), 3);
    assert_eq!(pool.n_float_features(), 2);
    assert_eq!(pool.float_feature(0), Some(&[1.0, 2.0, 3.0][..]));
    assert_eq!(pool.float_feature(1), Some(&[10.0, 20.0, 30.0][..]));
    assert_eq!(pool.float_feature(2), None);
    assert_eq!(pool.label(), &[0.0, 1.0, 0.0]);
    assert_eq!(pool.weights(), &[1.0, 0.5, 2.0]);
    assert_eq!(pool.group_id(), &[7, 7, 8]);
}

/// Categorical / text / embedding / group_id / subgroup_id / pairs / baseline
/// fields are present and default to empty when not supplied.
#[test]
fn unsupplied_columns_default_to_empty() {
    let pool = OwnedColumns::new(vec![vec![1.0, 2.0]], vec![0.0, 1.0])
        .into_pool()
        .expect("Pool with only floats + label must build");

    assert_eq!(pool.n_cat_features(), 0);
    assert_eq!(pool.n_text_features(), 0);
    assert_eq!(pool.n_embedding_features(), 0);
    assert!(pool.cat_features().is_empty());
    assert!(pool.text_features().is_empty());
    assert!(pool.embedding_features().is_empty());
    assert!(pool.group_id().is_empty());
    assert!(pool.subgroup_id().is_empty());
    assert!(pool.pairs().is_empty());
    assert!(pool.baseline().is_empty());
}

/// Building a Pool with mismatched column lengths returns a typed `CbResult`
/// error (no panic).
#[test]
fn mismatched_label_length_is_typed_error() {
    let result = OwnedColumns::new(vec![vec![1.0, 2.0, 3.0]], vec![0.0, 1.0]).into_pool();

    match result {
        Err(CbError::LengthMismatch { column, .. }) => assert!(
            column.contains("label"),
            "error should name the offending column, got: {column}"
        ),
        other => panic!("expected LengthMismatch for mismatched label, got {other:?}"),
    }
}

/// A mismatched float-feature column (a feature shorter than the rest) is a
/// typed error, not an out-of-bounds index.
#[test]
fn mismatched_float_feature_length_is_typed_error() {
    let result =
        OwnedColumns::new(vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0]], vec![0.0, 1.0, 0.0])
            .into_pool();

    assert!(matches!(result, Err(CbError::LengthMismatch { .. })));
}

/// Optional metadata columns (here, weights) are also length-validated.
#[test]
fn mismatched_weights_length_is_typed_error() {
    let result = OwnedColumns::new(vec![vec![1.0, 2.0]], vec![0.0, 1.0])
        .with_weights(vec![1.0])
        .into_pool();

    match result {
        Err(CbError::LengthMismatch { column, .. }) => assert!(column.contains("weights")),
        other => panic!("expected LengthMismatch for mismatched weights, got {other:?}"),
    }
}

/// Pairs whose ids are in range are carried through unchanged. Pairs are
/// object-index references, not per-row columns, so they are validated by id
/// bounds (`id < n_rows`, WR-02) rather than by column length.
#[test]
fn pairs_are_carried_through() {
    let pairs = vec![Pair { winner_id: 0, loser_id: 1 }];
    let pool = OwnedColumns::new(vec![vec![1.0, 2.0]], vec![0.0, 1.0])
        .with_pairs(pairs.clone())
        .into_pool()
        .expect("Pool with pairs must build");

    assert_eq!(pool.pairs(), pairs.as_slice());
}
