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

// ── EXPORT-01f (AT-01f-5): `Export` sub-variant -> Python exception mapping ──

/// `CategoricalFeaturesUnsupported` / `NonObliviousTreesUnsupported` /
/// `RegionTreesUnsupported` -> `CatBoostValueError` (the model itself is the
/// "bad input" to the export operation, like `PartialDependence`'s mapping).
#[test]
fn export_categorical_unsupported_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::Export(
            cb_model::OnnxExportError::CategoricalFeaturesUnsupported,
        ));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
    });
}

#[test]
fn export_non_oblivious_unsupported_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::Export(
            cb_model::OnnxExportError::NonObliviousTreesUnsupported,
        ));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
    });
}

#[test]
fn export_region_unsupported_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::Export(
            cb_model::OnnxExportError::RegionTreesUnsupported,
        ));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
    });
}

/// `NonIntegerClassLabelsUnsupported` -> `CatBoostValueError` (same "bad
/// input model" category as the other three guard-rejection variants).
#[test]
fn export_non_integer_class_labels_unsupported_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::Export(
            cb_model::OnnxExportError::NonIntegerClassLabelsUnsupported,
        ));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
    });
}

/// `Io` -> `PyIOError` (mirrors the top-level `CatBoostError::Io` arm's own
/// mapping exactly).
#[test]
fn export_io_maps_to_io_error() {
    Python::attach(|py| {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no parent dir");
        let err = to_pyerr(&FacadeError::Export(cb_model::OnnxExportError::Io(io)));
        assert!(err.is_instance_of::<PyIOError>(py));
        assert!(err.value(py).to_string().contains("no parent dir"));
    });
}

/// A minimal, non-empty local message — `prost::EncodeError` has no public
/// constructor, so the only way to obtain a REAL one (rather than fabricate a
/// variant that can never occur) is to force an actual encode failure: a
/// too-small `BufMut` target for a message with at least one non-default field.
#[derive(Clone, PartialEq, ::prost::Message)]
struct EncodeErrorProbe {
    #[prost(int64, tag = "1")]
    value: i64,
}

/// `Encode` -> base `CatBoostError` (an internal/unexpected failure, mirrors
/// `Train`/`Model`'s mapping — not user-input-driven).
#[test]
fn export_encode_maps_to_base_error() {
    let probe = EncodeErrorProbe { value: 42 };
    let mut small_buf = [0_u8; 0];
    let mut writer: &mut [u8] = &mut small_buf;
    let encode_err = prost::Message::encode(&probe, &mut writer)
        .expect_err("a zero-capacity buffer must fail to encode a non-empty message");

    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::Export(cb_model::OnnxExportError::Encode(
            encode_err,
        )));
        assert!(err.is_instance_of::<PyCatBoostError>(py));
        assert!(!err.is_instance_of::<CatBoostValueError>(py));
    });
}

/// `UnsupportedLoss` -> `CatBoostValueError` (an out-of-scope
/// LossFunctionChange loss name is a bad-input value error, like
/// `FeatureMismatch` / `PartialDependence`; FSTR-02 FL-03).
#[test]
fn unsupported_loss_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::UnsupportedLoss("AUC".to_owned()));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
        assert!(err.value(py).to_string().contains("AUC"));
    });
}

/// `UnsupportedModel` -> `CatBoostValueError` (a non-scalar / non-oblivious / CTR
/// model handed to `staged_predict` is a bad-input value error, like
/// `UnsupportedLoss` / `PartialDependence`; SP-03).
#[test]
fn unsupported_model_maps_to_value_error() {
    Python::attach(|py| {
        let err = to_pyerr(&FacadeError::UnsupportedModel("non-symmetric trees".to_owned()));
        assert!(err.is_instance_of::<CatBoostValueError>(py));
        assert!(err.value(py).to_string().contains("non-symmetric trees"));
    });
}
