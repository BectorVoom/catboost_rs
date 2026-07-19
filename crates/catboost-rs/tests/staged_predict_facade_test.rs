//! SP-03 facade integration: `catboost_rs::Model::staged_predict` narrows a
//! `Pool` to float columns, delegates to `cb_model::predict_raw_staged`, applies
//! `None` defaults, guards non-scalar/non-oblivious/CTR models, and maps a
//! wrong-width pool to `FeatureMismatch`. Exercised through the PUBLISHED facade.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostError, IngestSource, Model, OwnedColumns, Pool};
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The pinned float-only oblivious regressor (4 float features), through the facade.
fn staged_model() -> Model {
    Model::load_cbm(&fixture("staged_predict/model.cbm")).expect("staged_predict/model.cbm loads")
}

/// A `Pool` built from `numeric_tiny/X.npy` as SoA `f64` float columns.
fn numeric_tiny_pool() -> Pool {
    let x: Array2<f64> =
        read_npy(fixture("inputs/numeric_tiny/X.npy")).expect("numeric_tiny/X.npy loads");
    let float_features: Vec<Vec<f64>> = (0..x.ncols())
        .map(|fi| x.column(fi).iter().copied().collect())
        .collect();
    let label = vec![0.0_f64; x.nrows()];
    OwnedColumns::new(float_features, label)
        .into_pool()
        .expect("numeric_tiny pool builds")
}

/// SP-03 / Scenario 1: with `None` defaults (all trees, step 1), the final stage
/// equals `predict(pool)` exactly (byte-for-byte — both are the full-ensemble
/// RawFormulaVal over the same order-locked accumulation).
#[test]
fn staged_predict_facade_last_equals_predict() {
    let model = staged_model();
    let pool = numeric_tiny_pool();

    let stages = model
        .staged_predict(&pool, None, None, None)
        .expect("staged_predict ok");
    assert!(!stages.is_empty(), "at least one stage");

    let full = model.predict(&pool).expect("predict ok");
    assert_eq!(
        stages.last().unwrap(),
        &full,
        "final stage == predict(pool) for the default schedule"
    );
    // Every stage row is one value per object.
    for row in &stages {
        assert_eq!(row.len(), full.len(), "stage row has one value per object");
    }
}

/// SP-03 / Scenario 4: a wrong-width pool surfaces as `FeatureMismatch` (the
/// model expects 4 float features).
#[test]
fn staged_predict_feature_mismatch() {
    let model = staged_model();
    // 2 float columns, model expects 4.
    let narrow = OwnedColumns::new(vec![vec![0.0_f64; 3], vec![1.0_f64; 3]], vec![0.0; 3])
        .into_pool()
        .expect("narrow pool builds");
    match model.staged_predict(&narrow, None, None, None) {
        Err(CatBoostError::FeatureMismatch(_)) => {}
        other => panic!("expected FeatureMismatch, got {other:?}"),
    }
}

/// SP-03 / Scenario 3: a non-oblivious (non-symmetric) model is rejected with the
/// typed `UnsupportedModel` guard error, BEFORE any pool narrowing.
#[test]
fn staged_predict_rejects_non_scalar_oblivious() {
    let model = Model::load_cbm(&fixture("fstr_loss_change/non_symmetric_model.cbm"))
        .expect("non_symmetric_model.cbm loads");
    let pool = numeric_tiny_pool();
    match model.staged_predict(&pool, None, None, None) {
        Err(CatBoostError::UnsupportedModel(_)) => {}
        other => panic!("expected UnsupportedModel, got {other:?}"),
    }
}
