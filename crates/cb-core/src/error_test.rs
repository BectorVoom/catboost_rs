//! Unit tests for [`crate::error`]. Kept in a dedicated `*_test.rs` file per the
//! source/test separation rule (D-17); no `#[cfg(test)] mod` lives in `error.rs`.

use crate::error::{CbError, CbResult};

#[test]
fn invalid_bound_display_message_is_stable() {
    let err = CbError::InvalidBound { bound: 0 };
    assert_eq!(err.to_string(), "uniform bound must be > 0, got 0");
}

#[test]
fn out_of_range_display_message_is_stable() {
    let err = CbError::OutOfRange("seed".to_string());
    assert_eq!(err.to_string(), "value out of range: seed");
}

#[test]
fn dtype_display_message_is_stable() {
    let err = CbError::Dtype {
        expected: "Float64",
        got: "Int64".to_string(),
    };
    assert_eq!(err.to_string(), "unsupported dtype: expected Float64, got Int64");
}

#[test]
fn length_mismatch_display_message_is_stable() {
    let err = CbError::LengthMismatch {
        column: "label".to_string(),
        expected: 40,
        actual: 39,
    };
    assert_eq!(
        err.to_string(),
        "column `label` has length 39, expected 40 (n_rows)"
    );
}

#[test]
fn nan_in_categorical_display_message_is_stable() {
    let err = CbError::NanInCategorical { column: 2 };
    assert_eq!(err.to_string(), "NaN in categorical column 2");
}

#[test]
fn ingestion_display_message_is_stable() {
    let err = CbError::Ingestion {
        message: "non-contiguous chunk".to_string(),
    };
    assert_eq!(err.to_string(), "ingestion error: non-contiguous chunk");
}

#[test]
fn new_variants_preserve_clone_and_eq() {
    // The ingestion variants must keep CbError `Clone + PartialEq + Eq` (no
    // `#[from]` of a non-Eq external error). This will not compile otherwise.
    let a = CbError::Dtype {
        expected: "Float64",
        got: "Utf8".to_string(),
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn cb_result_ok_path_round_trips() {
    let ok: CbResult<u32> = Ok(42);
    assert_eq!(ok.unwrap(), 42);
}

#[test]
fn cb_result_err_path_carries_variant() {
    let err: CbResult<u32> = Err(CbError::InvalidBound { bound: 0 });
    assert!(matches!(err, Err(CbError::InvalidBound { bound: 0 })));
}
