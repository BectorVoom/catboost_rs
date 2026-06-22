//! Unit tests for the ingest adapters — strict dtype / contiguity rejection
//! (D-12) and the own-before-detach property (D-11 / threat T-08-08).
//!
//! Source/test separation (CLAUDE.md): declared `#[cfg(test)] mod ingest_py_test;`
//! at the crate root (`lib.rs`). Test code may freely `unwrap`/`panic` (crate-level
//! `#![cfg_attr(test, allow(...))]`).
//!
//! The Python-facing multi-source paths (Pandas/Arrow/Polars) are exercised in
//! `tests/test_ingestion.py`; here we cover the NumPy contract and the ownership
//! invariant at the Rust seam.

use catboost_rs::IngestSource;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::errors::CatBoostValueError;
use crate::ingest_py::{ingest_to_owned, numpy_to_owned};

/// A C-contiguous float32 2-D X with a matching float32 1-D y is accepted, and
/// the OWNED columns round-trip into a Pool with no live Python borrow (D-11):
/// the returned `OwnedColumns` is moved into `into_pool()` after the function
/// returns, proving it holds no `PyReadonlyArray` borrow.
#[test]
fn accepts_contiguous_f32_and_owns() {
    Python::attach(|py| {
        // 3 rows x 2 features, C-contiguous float32.
        let x = PyArray2::<f32>::zeros(py, [3, 2], false);
        let y = PyArray1::<f32>::zeros(py, 3, false);
        let owned = ingest_to_owned(py, x.as_any(), Some(y.as_any()), py_none(py))
            .expect("contiguous float32 input must be accepted");
        // `owned` is a value with no lifetime tied to `x`/`y`; moving it into
        // into_pool() compiles iff it borrows nothing from Python (own-before-detach).
        let pool = owned.into_pool().expect("owned columns must build a Pool");
        assert_eq!(pool.float_features().len(), 2);
        assert_eq!(pool.label().len(), 3);
    });
}

/// `numpy_to_owned` returns fully-owned columns for the bare NumPy path too.
#[test]
fn numpy_path_owns() {
    Python::attach(|py| {
        let x = PyArray2::<f32>::zeros(py, [4, 3], false);
        let owned = numpy_to_owned(x.as_any(), None).expect("float32 X accepted");
        let pool = owned.into_pool().expect("Pool builds");
        assert_eq!(pool.float_features().len(), 3);
        assert_eq!(pool.float_features()[0].len(), 4);
    });
}

/// A float64 X is rejected with the typed `CatBoostValueError` (no silent
/// precision coercion, D-12 / threat T-08-09) and the message names float32.
#[test]
fn rejects_float64_with_actionable_message() {
    Python::attach(|py| {
        let x = PyArray2::<f64>::zeros(py, [3, 2], false);
        let err = ingest_to_owned(py, x.as_any(), None, py_none(py))
            .expect_err("float64 X must be rejected (no silent coercion)");
        assert!(err.is_instance_of::<CatBoostValueError>(py));
        assert!(err.value(py).to_string().contains("float32"));
    });
}

/// A non-contiguous (Fortran-order) float32 X is rejected with the typed error
/// pointing at `ascontiguousarray`.
#[test]
fn rejects_non_contiguous_with_actionable_message() {
    Python::attach(|py| {
        // `is_fortran = true` => column-major => NOT C-contiguous.
        let x = PyArray2::<f32>::zeros(py, [3, 2], true);
        let err = ingest_to_owned(py, x.as_any(), None, py_none(py))
            .expect_err("non-C-contiguous X must be rejected");
        assert!(err.is_instance_of::<CatBoostValueError>(py));
        let msg = err.value(py).to_string();
        assert!(msg.contains("contiguous") || msg.contains("ascontiguousarray"));
    });
}

/// A y whose length differs from X's row count is rejected as a value error.
#[test]
fn rejects_length_mismatch() {
    Python::attach(|py| {
        let x = PyArray2::<f32>::zeros(py, [3, 2], false);
        let y = PyArray1::<f32>::zeros(py, 5, false);
        let err = ingest_to_owned(py, x.as_any(), Some(y.as_any()), py_none(py))
            .expect_err("mismatched y length must be rejected");
        assert!(err.is_instance_of::<CatBoostValueError>(py));
    });
}

/// Helper: a typed `None` for the `cat_features` slice argument.
fn py_none<'a>(_py: Python<'_>) -> Option<&'a [usize]> {
    None
}
