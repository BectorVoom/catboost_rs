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
fn cb_result_ok_path_round_trips() {
    let ok: CbResult<u32> = Ok(42);
    assert_eq!(ok.unwrap(), 42);
}

#[test]
fn cb_result_err_path_carries_variant() {
    let err: CbResult<u32> = Err(CbError::InvalidBound { bound: 0 });
    assert!(matches!(err, Err(CbError::InvalidBound { bound: 0 })));
}
