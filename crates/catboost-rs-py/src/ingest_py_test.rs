//! Unit tests for `numpy_to_owned` — strict dtype / contiguity rejection (D-12).
//!
//! Source/test separation (CLAUDE.md): declared `#[cfg(test)] mod ingest_py_test;`
//! in `ingest_py.rs`. Test code may freely `unwrap`/`panic` (crate-level
//! `#![cfg_attr(test, allow(...))]`).

use catboost_rs::IngestSource;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

use crate::ingest_py::numpy_to_owned;

/// A C-contiguous float32 2-D X with a matching float32 1-D y is accepted, and
/// the owned columns round-trip into a Pool (the happy path the smoke rides).
#[test]
fn accepts_contiguous_f32() {
    Python::attach(|py| {
        // 3 rows x 2 features, C-contiguous float32.
        let x = PyArray2::<f32>::zeros(py, [3, 2], false);
        let y = PyArray1::<f32>::zeros(py, 3, false);
        let owned = numpy_to_owned(x.as_any(), Some(y.as_any()))
            .expect("contiguous float32 input must be accepted");
        let pool = owned.into_pool().expect("owned columns must build a Pool");
        assert_eq!(pool.float_features().len(), 2);
        assert_eq!(pool.label().len(), 3);
    });
}

/// A float64 X is rejected with a ValueError (no silent precision coercion, D-12).
#[test]
fn rejects_float64() {
    Python::attach(|py| {
        let x = PyArray2::<f64>::zeros(py, [3, 2], false);
        let err = numpy_to_owned(x.as_any(), None)
            .expect_err("float64 X must be rejected (no silent coercion)");
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
}

/// A non-contiguous (Fortran-order) float32 X is rejected with a ValueError.
#[test]
fn rejects_non_contiguous() {
    Python::attach(|py| {
        // `is_fortran = true` => column-major => NOT C-contiguous.
        let x = PyArray2::<f32>::zeros(py, [3, 2], true);
        let err = numpy_to_owned(x.as_any(), None)
            .expect_err("non-C-contiguous X must be rejected");
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
}

/// A y whose length differs from X's row count is rejected.
#[test]
fn rejects_length_mismatch() {
    Python::attach(|py| {
        let x = PyArray2::<f32>::zeros(py, [3, 2], false);
        let y = PyArray1::<f32>::zeros(py, 5, false);
        let err = numpy_to_owned(x.as_any(), Some(y.as_any()))
            .expect_err("mismatched y length must be rejected");
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
}
