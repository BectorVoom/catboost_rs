//! EXPORT-02 (CM-03) facade integration: `catboost_rs::Model::save_coreml` on a
//! guard-passing model loaded via the already-`pub` `Model::load_cbm` writes a
//! well-formed `.mlmodel` file, and rejects a categorical/CTR model with the
//! typed `CatBoostError::CoreMlExport`. Reuses committed `cb-oracle` fixtures
//! (no new fixture needed for a purely structural facade test), mirroring
//! `onnx_facade_test.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostError, Model};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn unique_tmp(tag: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "catboost_rs_coreml_{tag}_{}_{nonce}.mlmodel",
        std::process::id()
    ))
}

/// CM-03: `save_coreml` on the committed float-only regression fixture succeeds
/// and writes non-empty `.mlmodel` bytes (a valid protobuf-framed message).
#[test]
fn save_coreml_delegates_and_succeeds_regressor() {
    let model = Model::load_cbm(&fixture("model_serde/regression/model.cbm"))
        .expect("model_serde/regression/model.cbm loads");
    let path = unique_tmp("facade_regressor");

    model.save_coreml(&path).expect("save_coreml must succeed");
    assert!(path.exists());

    let bytes = std::fs::read(&path).expect("read back the written file");
    let _ = std::fs::remove_file(&path);
    assert!(!bytes.is_empty());
}

/// CM-03: `save_coreml` on a categorical/CTR model returns the typed
/// `CatBoostError::CoreMlExport(..)` and writes no file.
#[test]
fn save_coreml_rejects_ctr() {
    let model = Model::load_cbm(&fixture("ctr_load/simple.cbm"))
        .expect("ctr_load/simple.cbm loads");
    let path = unique_tmp("facade_reject_ctr");

    match model.save_coreml(&path) {
        Err(CatBoostError::CoreMlExport(_)) => {}
        other => panic!("expected CoreMlExport error, got {other:?}"),
    }
    assert!(!path.exists());
}
