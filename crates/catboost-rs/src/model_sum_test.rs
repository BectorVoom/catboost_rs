//! Facade-level tests for `Model::sum_models` (SPEC `sum_models`, SM-05).
//! Mounted at the crate root via `#[cfg(test)] mod model_sum_test;`, mirroring
//! `error_test.rs` / `metrics_test.rs` / `onnx_test.rs`'s internal
//! `#[cfg(test)]`-mounted-module precedent (`crates/catboost-rs/src/lib.rs`).
//! Uses [`crate::Model::from_canonical`] (`pub(crate)`), so this MUST live
//! inside the crate rather than under `tests/` (same rationale as
//! `onnx_test.rs`).

use cb_data::ingest::IngestSource;

use crate::{CatBoostError, Model, OwnedColumns};

/// A tiny float-only oblivious canonical model, 1 split / 2 leaves, matching
/// the `numeric_tiny`-style single-feature pool built below.
fn tiny_model(bias: f64, leaf_values: [f64; 2]) -> cb_model::Model {
    cb_model::Model {
        oblivious_trees: vec![cb_model::ObliviousTree {
            splits: vec![cb_model::ModelSplit::Float(cb_model::Split {
                feature: 0,
                border: 0.5,
            })],
            leaf_values: leaf_values.to_vec(),
            leaf_weights: vec![1.0, 1.0],
        }],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

fn tiny_pool() -> crate::Pool {
    let float_features = vec![vec![0.1_f64, 0.9, 0.3]];
    let label = vec![0.0_f64; 3];
    OwnedColumns::new(float_features, label)
        .into_pool()
        .expect("tiny pool builds")
}

/// SM-05: `Model::sum_models(&[&a,&b], None)` predicts the weighted sum
/// (weights default to all-ones) of the inputs' raw predictions.
#[test]
fn sum_models_facade_roundtrip() {
    let a = Model::from_canonical(tiny_model(0.5, [1.0, 2.0]));
    let b = Model::from_canonical(tiny_model(-0.25, [0.5, -1.5]));
    let pool = tiny_pool();

    let merged = Model::sum_models(&[&a, &b], None).expect("compatible facade sum must succeed");

    let a_pred = a.predict(&pool).expect("a predicts");
    let b_pred = b.predict(&pool).expect("b predicts");
    let merged_pred = merged.predict(&pool).expect("merged predicts");

    for i in 0..merged_pred.len() {
        let expected = a_pred.get(i).copied().unwrap_or(f64::NAN)
            + b_pred.get(i).copied().unwrap_or(f64::NAN);
        let actual = merged_pred.get(i).copied().unwrap_or(f64::NAN);
        assert!((actual - expected).abs() <= 1e-9, "object {i}: {actual} vs {expected}");
    }
}

/// SM-05: an unmergeable pair (weight/model count mismatch) surfaces a
/// [`CatBoostError::Model`], not a panic or a silent wrong merge.
#[test]
fn sum_models_facade_maps_error() {
    let a = Model::from_canonical(tiny_model(0.0, [1.0, 2.0]));
    let b = Model::from_canonical(tiny_model(0.0, [0.5, -1.5]));

    let err = Model::sum_models(&[&a, &b], Some(&[1.0]))
        .expect_err("mismatched weight count must be rejected");
    assert!(matches!(err, CatBoostError::Model(_)), "expected Model, got {err:?}");
}
