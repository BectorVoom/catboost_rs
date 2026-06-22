//! Unit asserts for the PYAPI-05 typed-exception taxonomy: each of the six
//! facade [`catboost_rs::CatBoostError`] variants converts (via
//! [`crate::errors::to_pyerr`]) to its expected Python exception type. Mirrors the
//! facade's `error_test.rs` variant-conversion assertion style, but for the
//! `CatBoostError -> PyErr` direction.

use catboost_rs::CatBoostError as FacadeError;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

use crate::errors::{
    to_pyerr, CatBoostError as PyCatBoostError, CatBoostValueError, PyCbError,
};

/// `FeatureMismatch` -> `CatBoostValueError`, message preserved.
#[test]
fn feature_mismatch_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::FeatureMismatch("k=3 != 5".to_owned()));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
        assert!(err.value(py).to_string().contains("k=3 != 5"));
    });
}

/// `Deserialize` -> `CatBoostValueError` (malformed model = value error).
#[test]
fn deserialize_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::Deserialize("bad magic".to_owned()));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
        assert!(err.value(py).to_string().contains("bad magic"));
    });
}

/// `SchemaVersion` -> `CatBoostValueError` (unsupported schema = value error).
#[test]
fn schema_version_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::SchemaVersion("v999".to_owned()));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
        assert!(err.value(py).to_string().contains("v999"));
    });
}

/// `Io` -> `PyIOError` (a file-I/O failure surfaces as the stdlib I/O error).
#[test]
fn io_maps_to_io_error() {
    Python::attach(|py| {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing.cbm");
        let err = to_pyerr(&FacadeError::Io(io));
        assert!(err.is_instance_of::<PyIOError>(py));
        assert!(err.value(py).to_string().contains("missing.cbm"));
    });
}

/// `Train` -> base `CatBoostError` (internal training error).
#[test]
fn train_maps_to_base_error() {
    Python::attach(|py| {
        let core = cb_core::CbError::Degenerate("empty target".to_owned());
        let err = to_pyerr(&FacadeError::Train(core));
        assert!(err.is_instance_of::<PyCatBoostError>(py));
        // NOT a value error: a training failure is the base category, not D-12.
        assert!(!err.is_instance_of::<CatBoostValueError>(py));
        assert!(err.value(py).to_string().contains("empty target"));
    });
}

/// `Model` -> base `CatBoostError` (internal model (de)serialization/apply error).
#[test]
fn model_maps_to_base_error() {
    Python::attach(|py| {
        let me = cb_model::ModelError::Deserialize("corrupt fbs".to_owned());
        let err = to_pyerr(&FacadeError::Model(me));
        assert!(err.is_instance_of::<PyCatBoostError>(py));
        assert!(err.value(py).to_string().contains("corrupt fbs"));
    });
}

/// The local newtype `From<PyCbError> for PyErr` routes through `to_pyerr` (the
/// orphan-legal call-site conversion used by `.map_err(PyCbError)?`).
#[test]
fn pycberror_newtype_routes_through_to_pyerr() {
    Python::attach(|py| {
        let err: PyErr = PyCbError(FacadeError::FeatureMismatch("z".to_owned())).into();
        assert!(err.is_instance_of::<CatBoostValueError>(py));
    });
}
