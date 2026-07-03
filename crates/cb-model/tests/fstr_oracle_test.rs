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
//!      `[feature_i, feature_j, score]` triples) at <= 1e-5,
//!   3. `loss_function_change` (MODEL-03 / D-12; D-6.6-09) reproduces the
//!      committed `fstr_loss_change/oblivious_loss_function_change.npy`
//!      (`get_feature_importance(type='LossFunctionChange', data=pool)`) at
//!      <= 1e-5 for the binclf oblivious model, and
//!   4. the generalized non-symmetric `prediction_values_change` /
//!      `interaction` recursion (D-6.6-10) reproduces the upstream PVC /
//!      Interaction of a non-symmetric (Depthwise) model at <= 1e-5.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{
    interaction, load_cbm, loss_function_change, prediction_values_change, Model, ModelSplit,
    ObliviousTree, Split,
};
use cb_oracle::{load_f64_vec, load_model_json, ModelJson};
use ndarray::Array2;
use ndarray_npy::read_npy;

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
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: mj.bias().expect("bias must parse"),
        float_feature_borders: mj.float_feature_borders(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
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

// ── LossFunctionChange (MODEL-03 / D-12; D-6.6-09) ──────────────────────────

/// Load an `npy` matrix as per-feature `f32` SoA columns (object-major rows →
/// feature columns) for the apply / SHAP paths.
fn load_columns(rel: &str) -> Vec<Vec<f32>> {
    let x: Array2<f64> =
        read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

#[test]
fn loss_function_change_matches_upstream_within_tol() {
    // The canonical binclf oblivious model (same params/seed that produced the
    // committed `model_serde/binclf/model.json` AND the LossFunctionChange
    // fixture).
    let model = upstream_model();
    let cols = load_columns("fstr_loss_change/binclf_X.npy");
    let labels = load_f64_vec(&fixture("fstr_loss_change/binclf_y.npy"))
        .expect("binclf_y.npy must load");

    let n_features = model.float_feature_borders.len().max(4);
    let lfc = loss_function_change(&model, &cols, &labels, n_features);

    let expected =
        load_f64_vec(&fixture("fstr_loss_change/oblivious_loss_function_change.npy"))
            .expect("oblivious_loss_function_change.npy must load");

    assert_eq!(
        lfc.len(),
        expected.len(),
        "LFC length {} != fixture length {}",
        lfc.len(),
        expected.len()
    );
    for (i, (&got, &want)) in lfc.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "LFC[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }
}

// ── Non-symmetric PVC / Interaction (D-6.6-10) ──────────────────────────────

/// Load the committed non-symmetric (Depthwise) model under test from its
/// `.cbm` (carries the node graph + leaf values + leaf weights).
fn non_symmetric_model() -> Model {
    let model = load_cbm(&fixture("fstr_loss_change/non_symmetric_model.cbm"))
        .expect("non_symmetric_model.cbm must load");
    assert!(
        !model.non_symmetric_trees.is_empty(),
        "non_symmetric_model.cbm must decode into non_symmetric_trees"
    );
    assert!(
        model.oblivious_trees.is_empty(),
        "non_symmetric_model.cbm must have NO oblivious trees (pure non-symmetric)"
    );
    model
}

#[test]
fn non_symmetric_prediction_values_change_matches_upstream_within_tol() {
    let model = non_symmetric_model();
    let pvc = prediction_values_change(&model);

    let expected = load_f64_vec(&fixture("fstr_loss_change/non_symmetric_pvc.npy"))
        .expect("non_symmetric_pvc.npy must load");

    assert_eq!(
        pvc.len(),
        expected.len(),
        "non-symmetric PVC length {} != fixture length {}",
        pvc.len(),
        expected.len()
    );
    for (i, (&got, &want)) in pvc.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "non-symmetric PVC[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }
    // The non-symmetric PVC must still normalize to 100 (ConvertToPercents).
    let total = cb_core::sum_f64(&pvc);
    assert!(
        (total - 100.0).abs() <= TOL,
        "non-symmetric PVC must sum to 100, got {total}"
    );
}

#[test]
fn non_symmetric_interaction_matches_upstream_within_tol() {
    let model = non_symmetric_model();
    let pairs = interaction(&model);

    // Flatten [feature_i, feature_j, score] triples row-major to match the
    // upstream `get_feature_importance(type="Interaction")` npy layout.
    let flat: Vec<f64> = pairs
        .iter()
        .flat_map(|&(i, j, score)| [i as f64, j as f64, score])
        .collect();

    let expected = load_f64_vec(&fixture("fstr_loss_change/non_symmetric_interaction.npy"))
        .expect("non_symmetric_interaction.npy must load");

    assert_eq!(
        flat.len(),
        expected.len(),
        "non-symmetric interaction length {} != fixture length {}",
        flat.len(),
        expected.len()
    );
    for (i, (&got, &want)) in flat.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "non-symmetric interaction[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }
}
