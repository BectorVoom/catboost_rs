//! Feature-importance oracle (MODEL-03 partial): build the canonical
//! [`cb_model::Model`] from the committed upstream `model_serde/binclf/model.json`
//! (the SAME `CatBoostClassifier(boost_from_average=False, **ISOLATING_PARAMS)`
//! model on `numeric_tiny` that produced the `feature_importance/*.npy` fixtures —
//! identical seed / params / inputs, so the trained trees, leaf values, and leaf
//! weights coincide) and assert:
//!
//!   1. `prediction_values_change` reproduces upstream
//!      `feature_importance/prediction_values_change.npy` (length `n_features`,
//!      percentages summing to 100) at <= 1e-5, and that the result sums to 100
//!      (the in-env normalization gate, runs regardless of fixture availability),
//!   2. `interaction` reproduces `feature_importance/interaction.npy` (flattened
//!      `[feature_i, feature_j, score]` triples) at <= 1e-5.
//!
//! The loss-change importance is deliberately NOT implemented (D-12, out of
//! scope).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{interaction, prediction_values_change, Model, ModelSplit, ObliviousTree, Split};
use cb_oracle::{load_f64_vec, load_model_json, ModelJson};

const TOL: f64 = 1e-5;

/// Resolve a path under `cb-oracle/fixtures/` from cb-model's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Build the canonical [`Model`] from an upstream [`ModelJson`], carrying splits,
/// per-tree `leaf_values` AND `leaf_weights` (both importances weight on the leaf
/// weights), the model `bias`, and the per-float-feature borders.
fn model_from_json(mj: &ModelJson) -> Model {
    let oblivious_trees = mj
        .oblivious_trees
        .iter()
        .map(|t| {
            let splits = t
                .splits
                .iter()
                .map(|s| ModelSplit::Float(Split {
                    feature: usize::try_from(s.float_feature_index).expect("non-negative feature"),
                    border: s.border,
                }))
                .collect();
            ObliviousTree {
                splits,
                leaf_values: t.leaf_values.clone(),
                leaf_weights: t.leaf_weights.clone(),
            }
        })
        .collect();
    Model {
        oblivious_trees,
        bias: mj.bias().expect("bias must parse"),
        float_feature_borders: mj.float_feature_borders(),
        ctr_data: None,
    }
}

fn upstream_model() -> Model {
    let mj = load_model_json(&fixture("model_serde/binclf/model.json"))
        .expect("binclf model.json must load");
    model_from_json(&mj)
}

#[test]
fn prediction_values_change_matches_upstream_within_tol() {
    let model = upstream_model();
    let pvc = prediction_values_change(&model);

    let expected = load_f64_vec(&fixture("feature_importance/prediction_values_change.npy"))
        .expect("prediction_values_change.npy must load");

    assert_eq!(
        pvc.len(),
        expected.len(),
        "PVC length {} != fixture length {}",
        pvc.len(),
        expected.len()
    );
    for (i, (&got, &want)) in pvc.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "PVC[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }
}

/// In-env normalization gate (needs NO fixture): PredictionValuesChange must sum
/// to 100 (ConvertToPercents) within tolerance.
#[test]
fn prediction_values_change_sums_to_100() {
    let model = upstream_model();
    let pvc = prediction_values_change(&model);
    let total = cb_core::sum_f64(&pvc);
    assert!(
        (total - 100.0).abs() <= TOL,
        "PVC must sum to 100, got {total}"
    );
}

#[test]
fn interaction_matches_upstream_within_tol() {
    let model = upstream_model();
    let pairs = interaction(&model);

    // Flatten [feature_i, feature_j, score] triples row-major to match the
    // upstream `get_feature_importance(type="Interaction")` npy layout.
    let flat: Vec<f64> = pairs
        .iter()
        .flat_map(|&(i, j, score)| [i as f64, j as f64, score])
        .collect();

    let expected = load_f64_vec(&fixture("feature_importance/interaction.npy"))
        .expect("interaction.npy must load");

    assert_eq!(
        flat.len(),
        expected.len(),
        "interaction length {} != fixture length {}",
        flat.len(),
        expected.len()
    );
    for (i, (&got, &want)) in flat.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "interaction[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }
}
