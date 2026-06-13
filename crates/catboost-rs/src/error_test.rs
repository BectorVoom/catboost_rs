//! Unit asserts for the public [`CatBoostError`] (RAPI-02): the typed variants
//! exist and the `#[from]` conversions wire `?`-propagation from the internal
//! crates onto the public surface.

use crate::error::CatBoostError;
use cb_core::CbError;

/// `#[from] cb_core::CbError` converts a training/core error into the public
/// `Train` variant (the RAPI-02 conversion lock).
#[test]
fn cb_error_converts_into_train_variant() {
    let core = CbError::Degenerate("empty target".to_owned());
    let public: CatBoostError = core.into();
    match public {
        CatBoostError::Train(CbError::Degenerate(msg)) => {
            assert_eq!(msg, "empty target");
        }
        other => panic!("expected Train(Degenerate), got {other:?}"),
    }
}

/// `?` on a `CbError`-returning call yields a `CatBoostError` (the ergonomic
/// propagation RAPI-02 promises).
#[test]
fn question_mark_propagates_cb_error() {
    fn inner() -> Result<(), CbError> {
        Err(CbError::OutOfRange("x".to_owned()))
    }
    fn outer() -> Result<(), CatBoostError> {
        inner()?;
        Ok(())
    }
    assert!(matches!(
        outer(),
        Err(CatBoostError::Train(CbError::OutOfRange(_)))
    ));
}

/// `#[from] std::io::Error` converts a file error into the public `Io` variant.
#[test]
fn io_error_converts_into_io_variant() {
    let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
    let public: CatBoostError = io.into();
    assert!(matches!(public, CatBoostError::Io(_)));
}

/// `#[from] cb_model::ModelError` converts a (de)serialization error into the
/// public `Model` variant.
#[test]
fn model_error_converts_into_model_variant() {
    let me = cb_model::ModelError::Deserialize("bad magic".to_owned());
    let public: CatBoostError = me.into();
    assert!(matches!(public, CatBoostError::Model(_)));
}

/// The facade's own boundary variants exist and carry a human-readable message
/// (no panic path).
#[test]
fn boundary_variants_exist() {
    let d = CatBoostError::Deserialize("d".to_owned());
    let s = CatBoostError::SchemaVersion("s".to_owned());
    let f = CatBoostError::FeatureMismatch("f".to_owned());
    assert!(format!("{d}").contains('d'));
    assert!(format!("{s}").contains('s'));
    assert!(format!("{f}").contains('f'));
}
