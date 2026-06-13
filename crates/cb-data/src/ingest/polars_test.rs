//! Unit tests for [`crate::ingest::polars`]. Dedicated `*_test.rs` file per the
//! source/test separation rule; no `#[cfg(test)] mod` lives in `polars.rs`.
//!
//! The central assertion is Arrow-vs-Polars column equality: a Polars `f64`
//! column rechunked through the shared Arrow path yields the SAME owned columns
//! as the direct Arrow path (D-05).

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array};
use cb_core::CbError;
use polars::prelude::*;

use crate::ingest::arrow::ArrowColumns;
use crate::ingest::polars::PolarsColumns;
use crate::ingest::IngestSource;

fn f64_array(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec())) as ArrayRef
}

fn sample_frame() -> DataFrame {
    df!(
        "f0" => &[1.0_f64, 2.0, 3.0],
        "f1" => &[10.0_f64, 20.0, 30.0],
        "y" => &[0.0_f64, 1.0, 0.0],
    )
    .unwrap()
}

#[test]
fn polars_column_equals_arrow_column() {
    // Arrow path.
    let arrow_pool = ArrowColumns::new(
        vec![f64_array(&[1.0, 2.0, 3.0]), f64_array(&[10.0, 20.0, 30.0])],
        vec![false, false],
        f64_array(&[0.0, 1.0, 0.0]),
    )
    .into_pool()
    .unwrap();

    // Polars path (rechunk -> shared Arrow validation).
    let polars_pool = PolarsColumns::new(
        sample_frame(),
        vec!["f0".to_string(), "f1".to_string()],
        vec![false, false],
        "y",
    )
    .into_pool()
    .unwrap();

    assert_eq!(polars_pool.n_rows(), arrow_pool.n_rows());
    assert_eq!(
        polars_pool.float_feature(0).unwrap(),
        arrow_pool.float_feature(0).unwrap()
    );
    assert_eq!(
        polars_pool.float_feature(1).unwrap(),
        arrow_pool.float_feature(1).unwrap()
    );
    assert_eq!(polars_pool.label(), arrow_pool.label());
}

#[test]
fn polars_non_f64_column_is_rejected() {
    let frame = df!(
        "f0" => &[1_i64, 2, 3],
        "y" => &[0.0_f64, 1.0, 0.0],
    )
    .unwrap();

    let err = PolarsColumns::new(frame, vec!["f0".to_string()], vec![false], "y")
        .into_pool()
        .unwrap_err();
    assert!(matches!(err, CbError::Ingestion { .. }));
}

#[test]
fn polars_nan_in_categorical_column_is_rejected() {
    let frame = df!(
        "f0" => &[1.0_f64, f64::NAN, 3.0],
        "y" => &[0.0_f64, 1.0, 0.0],
    )
    .unwrap();

    let err = PolarsColumns::new(frame, vec!["f0".to_string()], vec![true], "y")
        .into_pool()
        .unwrap_err();
    assert_eq!(err, CbError::NanInCategorical { column: 0 });
}

#[test]
fn polars_numeric_null_becomes_nan() {
    // CR-02 / WR-03: a nullable Polars f64 feature column must reach the Pool
    // with its `null` materialized as `f64::NAN` (not stripped to 0.0 by
    // `cont_slice`). Build a column with an explicit null via Option values.
    let f0 = Column::new("f0".into(), &[Some(1.0_f64), None, Some(3.0)]);
    let y = Column::new("y".into(), &[0.0_f64, 1.0, 0.0]);
    let frame = DataFrame::new(3, vec![f0, y]).unwrap();

    let pool = PolarsColumns::new(frame, vec!["f0".to_string()], vec![false], "y")
        .into_pool()
        .unwrap();

    let col = pool.float_feature(0).unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col[0], 1.0);
    assert!(col[1].is_nan(), "Polars null must become NaN, not 0.0");
    assert_eq!(col[2], 3.0);
}

#[test]
fn polars_null_in_categorical_column_is_rejected() {
    // CR-02 / threat T-02-14: a Polars `null` in a categorical column must be
    // rejected exactly like a NaN, even though it lives in the validity bitmap.
    let f0 = Column::new("f0".into(), &[Some(1.0_f64), None, Some(3.0)]);
    let y = Column::new("y".into(), &[0.0_f64, 1.0, 0.0]);
    let frame = DataFrame::new(3, vec![f0, y]).unwrap();

    let err = PolarsColumns::new(frame, vec!["f0".to_string()], vec![true], "y")
        .into_pool()
        .unwrap_err();
    assert_eq!(err, CbError::NanInCategorical { column: 0 });
}

#[test]
fn polars_missing_column_is_a_typed_error() {
    let err = PolarsColumns::new(
        sample_frame(),
        vec!["does_not_exist".to_string()],
        vec![false],
        "y",
    )
    .into_pool()
    .unwrap_err();
    assert!(matches!(err, CbError::Ingestion { .. }));
}
