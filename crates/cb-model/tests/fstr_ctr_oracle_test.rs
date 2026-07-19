//! FSTR-01 oracle: `interaction()` / `prediction_values_change()` CTR-aware
//! attribution (FIC-02 AT-FIC02d, FIC-03 AT-FIC03d) on a MIXED float +
//! categorical model.
//!
//! Loads upstream's own `crates/cb-oracle/fixtures/fstr_ctr/model.cbm`
//! directly via the Phase-23 CTR-capable `load_cbm` (CTRLOAD, merged from
//! `feat/23-ctr-model-loading`; `decode_cbm` reconstructs `ctr_data` + CTR
//! splits with oracle max|diff| = 0 on the `ctr_load` fixtures). This makes
//! the test a faithful oracle of the ATTRIBUTION code alone: the model bytes
//! are upstream ground truth, so any interaction/PVC divergence is an
//! FIC-02/FIC-03 bug, never a training-parity artifact. (An earlier revision
//! re-trained the same model in-process via `train_cat`, which is blocked by
//! the paused ORD-07 training-time residual —
//! `.planning/phases/24-ctr-split-search-correctness/simple-ctr-cat-feature-weight/`.)
//!
//! Asserts:
//!
//!   0. sanity gate — the loaded model's `predict_raw_cat` predictions match
//!      upstream's own `predictions.npy` (if this fails, an interaction/PVC
//!      mismatch would be a model-loading problem, not an FIC-02/FIC-03
//!      algorithm bug),
//!   1. HARD GATE (re-asserted independently here, not just at fixture
//!      generation time, per PLAN.md T4/T5/T6): the loaded model contains
//!      >= 1 `ModelSplit::Ctr` split whose `projection.cat_features().len()
//!      >= 2` (a genuine combination CTR actually present),
//!   2. `interaction(&model)` matches `interaction.npy` (flattened
//!      `[feature_i, feature_j, score]` triples) at <= 1e-5,
//!   3. `prediction_values_change(&model)` matches
//!      `prediction_values_change.npy` at <= 1e-5, and sums to 100.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::stringify_int_category;
use cb_model::{
    interaction, load_cbm, prediction_values_change_with_data, predict_raw_cat, Model as CbModel,
    ModelSplit,
};
use cb_oracle::load_f64_vec;
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

/// Load `fstr_ctr/X_float.npy` ([N, 2] float64) as per-feature `f32` SoA
/// columns (feature-major, matching `predict_raw_cat`'s
/// `feature_values: &[Vec<f32>]` convention).
fn load_float_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("fstr_ctr/X_float.npy"))
        .unwrap_or_else(|e| panic!("fstr_ctr/X_float.npy must load as float64 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Load `fstr_ctr/X_cat.npy` ([N, 2] int32) as per-feature `Vec<String>` SoA
/// columns, stringified via `cb_data::stringify_int_category` (A4 — the same
/// convention `gen_fixtures.py` fed to upstream's `Pool`).
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture("fstr_ctr/X_cat.npy"))
        .unwrap_or_else(|e| panic!("fstr_ctr/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// Load upstream's `fstr_ctr/model.cbm` (the exact bytes `gen_fixtures.py`
/// saved from the model whose `interaction.npy` / `prediction_values_change.npy`
/// / `predictions.npy` fixtures were exported) via the CTR-capable `load_cbm`.
fn loaded_model() -> CbModel {
    let model = load_cbm(&fixture("fstr_ctr/model.cbm"))
        .unwrap_or_else(|e| panic!("fstr_ctr/model.cbm must load via CTR-capable load_cbm: {e:?}"));
    assert!(
        model.ctr_data.is_some(),
        "a categorical model must decode with ctr_data: Some(..)"
    );
    model
}

/// HARD GATE (SPEC §7 / PLAN.md T4/T5/T6 — re-asserted independently here, not
/// relying on `gen_fixtures.py`'s own Python-side check): the loaded model
/// contains at least one `ModelSplit::Ctr` split whose
/// `projection.cat_features().len()` is at least 2 (a genuine combination
/// CTR), across EITHER tree kind, so a future accidental fixture
/// regeneration that loses the combination split fails THIS Rust test
/// loudly rather than silently degrading coverage.
fn assert_combination_ctr_present(model: &CbModel) {
    let has_combination = model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .chain(model.non_symmetric_trees.iter().flat_map(|t| t.tree_splits.iter()))
        .any(|s| matches!(s, ModelSplit::Ctr(c) if c.projection.cat_features().len() >= 2));
    assert!(
        has_combination,
        "HARD GATE: fstr_ctr model must contain >= 1 combination-CTR split \
         (projection.cat_features().len() >= 2) — a future fixture \
         regeneration lost the combination split"
    );
}

/// Sanity gate (PLAN.md T5 Risk note): the loaded model's predictions match
/// upstream's own `predictions.npy` BEFORE trusting an interaction/PVC
/// mismatch as an FIC-02/FIC-03 algorithm bug rather than a model-loading
/// divergence.
#[test]
fn fstr_ctr_predictions_sanity_gate() {
    let model = loaded_model();
    let float_cols = load_float_columns();
    let cat_cols = load_cat_columns();
    let actual = predict_raw_cat(&model, &float_cols, &cat_cols);
    let expected =
        load_f64_vec(&fixture("fstr_ctr/predictions.npy")).expect("predictions.npy must load");
    assert_eq!(actual.len(), expected.len(), "prediction count must match upstream");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "prediction[{i}] diverges: got {got}, want {want} (sanity gate — model-loading, not FIC-02/03)"
        );
    }
}

/// AT-FIC02d: `interaction()` on the mixed fstr_ctr model matches upstream
/// `get_feature_importance(type='Interaction')` within `1e-5`.
#[test]
fn interaction_matches_upstream_on_mixed_ctr_model() {
    let model = loaded_model();
    assert_combination_ctr_present(&model);

    let pairs = interaction(&model);
    let flat: Vec<f64> = pairs
        .iter()
        .flat_map(|&(i, j, score)| [i as f64, j as f64, score])
        .collect();

    let expected =
        load_f64_vec(&fixture("fstr_ctr/interaction.npy")).expect("interaction.npy must load");

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

/// AT-FIC03d: `prediction_values_change_with_data()` on the SAME mixed
/// fstr_ctr model matches upstream
/// `get_feature_importance(type='PredictionValuesChange', data=pool)` within
/// `1e-5`, and sums to 100. The `_with_data` mode is REQUIRED for parity with
/// this fixture: `data=pool` makes upstream recompute per-leaf weights from
/// the pool via the apply path (`CollectLeavesStatistics`), which genuinely
/// differs from the stored training-time `leaf_weights` for online-CTR models.
#[test]
fn pvc_matches_upstream_on_mixed_ctr_model() {
    let model = loaded_model();
    // Re-asserted independently (do not rely on another test function having
    // already run it — test functions may run in any order).
    assert_combination_ctr_present(&model);

    let float_cols = load_float_columns();
    let cat_cols = load_cat_columns();
    let pvc = prediction_values_change_with_data(&model, &float_cols, &cat_cols);
    let expected = load_f64_vec(&fixture("fstr_ctr/prediction_values_change.npy"))
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

    let total = cb_core::sum_f64(&pvc);
    assert!((total - 100.0).abs() <= TOL, "PVC must sum to 100, got {total}");
}
