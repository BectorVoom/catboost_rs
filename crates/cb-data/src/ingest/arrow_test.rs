//! Unit tests for [`crate::ingest::arrow`]. Dedicated `*_test.rs` file per the
//! source/test separation rule; no `#[cfg(test)] mod` lives in `arrow.rs`.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array};
use cb_core::CbError;

use crate::ingest::arrow::{arrow_f64_column, ArrowColumns};
use crate::ingest::{IngestSource, OwnedColumns};

fn f64_array(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec())) as ArrayRef
}

#[test]
fn arrow_float64_column_matches_owned_vec_path() {
    let raw = vec![1.0_f64, 2.5, -3.0, 4.25];

    // Owned-Vec path.
    let owned = OwnedColumns::new(vec![raw.clone()], vec![0.0, 1.0, 0.0, 1.0])
        .into_pool()
        .unwrap();

    // Arrow path.
    let arrow_col = f64_array(&raw);
    let read = arrow_f64_column(&arrow_col, 0, false).unwrap();

    assert_eq!(read, raw);
    assert_eq!(owned.float_feature(0).unwrap(), read.as_slice());
}

#[test]
fn arrow_columns_into_pool_produces_owned_columns() {
    let f0 = f64_array(&[1.0, 2.0, 3.0]);
    let f1 = f64_array(&[10.0, 20.0, 30.0]);
    let label = f64_array(&[0.0, 1.0, 0.0]);

    let pool = ArrowColumns::new(vec![f0, f1], vec![false, false], label)
        .into_pool()
        .unwrap();

    assert_eq!(pool.n_rows(), 3);
    assert_eq!(pool.n_float_features(), 2);
    assert_eq!(pool.float_feature(0).unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(pool.float_feature(1).unwrap(), &[10.0, 20.0, 30.0]);
    assert_eq!(pool.label(), &[0.0, 1.0, 0.0]);
}

#[test]
fn non_float64_dtype_is_rejected() {
    let int_col: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef;
    let err = arrow_f64_column(&int_col, 0, false).unwrap_err();
    assert!(matches!(err, CbError::Dtype { expected: "Float64", .. }));
}

#[test]
fn nan_in_categorical_column_is_rejected() {
    let col = f64_array(&[1.0, f64::NAN, 3.0]);
    let err = arrow_f64_column(&col, 2, true).unwrap_err();
    assert_eq!(err, CbError::NanInCategorical { column: 2 });
}

#[test]
fn nan_in_non_categorical_column_is_allowed() {
    let col = f64_array(&[1.0, f64::NAN, 3.0]);
    let read = arrow_f64_column(&col, 0, false).unwrap();
    assert_eq!(read.len(), 3);
    assert!(read[1].is_nan());
}

#[test]
fn float_feature_length_mismatch_is_rejected() {
    let f0 = f64_array(&[1.0, 2.0, 3.0]); // 3 rows
    let label = f64_array(&[0.0, 1.0]); // n_rows = 2
    let err = ArrowColumns::new(vec![f0], vec![false], label)
        .into_pool()
        .unwrap_err();
    assert!(matches!(
        err,
        CbError::LengthMismatch {
            expected: 2,
            actual: 3,
            ..
        }
    ));
}
