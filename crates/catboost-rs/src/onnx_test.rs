//! Unit asserts for the [`crate::Model::save_onnx`] facade method (EXPORT-01f
//! / AT-01f-1a/1b). Mirrors `error_test.rs`'s internal `#[cfg(test)]`-mounted
//! module precedent (`crates/catboost-rs/src/lib.rs`).
//!
//! AT-01f-1b (the CTR-rejection facade test) MUST live here rather than under
//! `crates/catboost-rs/tests/`: `crates/cb-model`'s `.cbm`/`model.json`
//! deserializers unconditionally set `ctr_data: None` and never construct
//! `ModelSplit::Ctr` (CTR-model *loading* is separate, not-yet-merged work on
//! `feat/23-ctr-model-loading`), so no currently-loadable fixture can
//! exercise the CTR-rejection path — this test hand-constructs a
//! `cb_model::Model` containing a literal `ModelSplit::Ctr` split (the same
//! technique `cb-model`'s own EXPORT-01a guard tests use) and wraps it via
//! [`crate::Model::from_canonical`], which is `pub(crate)` and therefore only
//! reachable from an INTERNAL test module compiled as part of this crate, not
//! from the external `tests/` integration-test binary.

use crate::{CatBoostError, Model};

/// A minimal all-oblivious model with a single [`cb_model::ModelSplit::Ctr`]
/// split — guard-failing (categorical/CTR unsupported) but otherwise
/// structurally valid.
fn ctr_model() -> cb_model::Model {
    let ctr_split = cb_model::CtrSplit {
        projection: cb_train::TProjection::single(0),
        ctr_type: cb_model::ECtrType::Borders,
        prior: cb_model::Prior { num: 0.0, denom: 1.0 },
        target_border_idx: 0,
        border: 0.0,
        shift: 0.0,
        scale: 1.0,
    };
    cb_model::Model {
        oblivious_trees: vec![cb_model::ObliviousTree {
            splits: vec![cb_model::ModelSplit::Ctr(ctr_split)],
            leaf_values: vec![0.1, 0.2],
            leaf_weights: vec![1.0, 1.0],
        }],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// AT-01f-1b: `save_onnx` on a CTR model maps the guard rejection through
/// `CatBoostError::Export`, never a panic.
#[test]
fn save_onnx_maps_guard_error() {
    let model = Model::from_canonical(ctr_model());
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "catboost_rs_onnx_facade_guard_{}_{nonce}.onnx",
        std::process::id()
    ));
    let result = model.save_onnx(&path, false);
    assert!(matches!(result, Err(CatBoostError::Export(_))));
    assert!(!path.exists());
}
