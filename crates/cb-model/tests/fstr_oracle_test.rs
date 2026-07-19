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
    interaction, load_cbm, loss_function_change, loss_function_change_logloss,
    prediction_values_change, Model, ModelSplit,
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
    // FL-02: the Logloss-defaulted wrapper reproduces the pre-FL-01 behavior
    // byte-for-byte (binary model → Logloss `GetFinalError`).
    let lfc = loss_function_change_logloss(&model, &cols, &labels, n_features);

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

// ── FSTR-02: per-numeric-loss LossFunctionChange (FL-04a / FL-04b) ──────────
//
// Each oblivious REGRESSOR trained with a distinct Min-optimized loss
// (RMSE / MAE / MAPE / Quantile:alpha=0.5) is reconstructed from its committed
// `.cbm` and its upstream `get_feature_importance('LossFunctionChange')` vector
// reproduced <=1e-5 by feeding an INDEPENDENT hand-written final-error closure
// (a stronger oracle than routing through the very `cb_train::EvalMetric` the
// facade uses) into the generalized `loss_function_change`.

/// Reproduce the `{tag}_loss_function_change.npy` fixture for the regressor in
/// `{tag}_model.cbm` using the supplied final-error closure; assert <= TOL.
fn assert_regression_lfc_matches<F: Fn(&[f64], &[f64]) -> f64>(tag: &str, final_error: F) {
    let model = load_cbm(&fixture(&format!("fstr_loss_change/{tag}_model.cbm")))
        .unwrap_or_else(|e| panic!("{tag}_model.cbm must load: {e:?}"));
    assert!(
        model.non_symmetric_trees.is_empty() && !model.oblivious_trees.is_empty(),
        "{tag}_model.cbm must be a pure oblivious model"
    );
    let cols = load_columns(&format!("fstr_loss_change/{tag}_X.npy"));
    let labels = load_f64_vec(&fixture(&format!("fstr_loss_change/{tag}_y.npy")))
        .unwrap_or_else(|e| panic!("{tag}_y.npy must load: {e:?}"));
    let expected =
        load_f64_vec(&fixture(&format!("fstr_loss_change/{tag}_loss_function_change.npy")))
            .unwrap_or_else(|e| panic!("{tag}_loss_function_change.npy must load: {e:?}"));

    let n_features = expected.len();
    let lfc = loss_function_change(&model, &cols, &labels, n_features, final_error);
    assert_eq!(lfc.len(), expected.len(), "{tag} LFC length mismatch");
    for (i, (&got, &want)) in lfc.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "{tag} LFC[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }
}

/// RMSE `GetFinalError` = `sqrt(mean((a − t)^2))`.
fn rmse_final_error(approx: &[f64], target: &[f64]) -> f64 {
    let n = approx.len();
    if n == 0 {
        return 0.0;
    }
    let sq: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| (a - t) * (a - t))
        .collect();
    (cb_core::sum_f64(&sq) / n as f64).sqrt()
}

/// MAE `GetFinalError` = `mean(|a − t|)`.
fn mae_final_error(approx: &[f64], target: &[f64]) -> f64 {
    let n = approx.len();
    if n == 0 {
        return 0.0;
    }
    let ad: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| (a - t).abs())
        .collect();
    cb_core::sum_f64(&ad) / n as f64
}

/// MAPE `GetFinalError` = `mean(|a − t| / max(1, |t|))` — the upstream
/// `TMAPEMetric` divisor convention pinned by the eval-metric-extension oracle.
fn mape_final_error(approx: &[f64], target: &[f64]) -> f64 {
    let n = approx.len();
    if n == 0 {
        return 0.0;
    }
    let ape: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| (a - t).abs() / t.abs().max(1.0))
        .collect();
    cb_core::sum_f64(&ape) / n as f64
}

/// Quantile(alpha) `GetFinalError` = `mean(pinball(a, t, alpha))`,
/// `pinball = t >= a ? alpha·(t − a) : (1 − alpha)·(a − t)`.
fn quantile_final_error(approx: &[f64], target: &[f64], alpha: f64) -> f64 {
    let n = approx.len();
    if n == 0 {
        return 0.0;
    }
    let pin: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| {
            let d = t - a;
            if d >= 0.0 {
                alpha * d
            } else {
                (1.0 - alpha) * -d
            }
        })
        .collect();
    cb_core::sum_f64(&pin) / n as f64
}

#[test]
fn loss_function_change_rmse_matches_upstream_within_tol() {
    assert_regression_lfc_matches("rmse", rmse_final_error);
}

#[test]
fn loss_function_change_mae_matches_upstream_within_tol() {
    assert_regression_lfc_matches("mae", mae_final_error);
}

#[test]
fn loss_function_change_mape_matches_upstream_within_tol() {
    assert_regression_lfc_matches("mape", mape_final_error);
}

#[test]
fn loss_function_change_quantile_matches_upstream_within_tol() {
    assert_regression_lfc_matches("quantile", |a, t| quantile_final_error(a, t, 0.5));
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
