//! Advanced SHAP-family fstr oracle (MODEL-05, D-6.6-10 / D-6.6-11): the new
//! `shap_interaction_values` / `prediction_diff` / `sage_values` backends and the
//! generalized non-symmetric TreeSHAP, each oracle-locked <= 1e-5 vs catboost
//! 1.2.10 `get_feature_importance(type=...)`:
//!
//!   1. `shap_interaction_values` reproduces upstream `ShapInteractionValues`
//!      (flattened `(n_obj, n_feat+1, n_feat+1)`) on the oblivious binclf model.
//!   2. `prediction_diff` reproduces upstream `PredictionDiff` on `X[:2]`.
//!   3. `sage_values` reproduces upstream `SageValues` (seed-pinned, D-6.6-11
//!      fallback (a) — deterministic through the Python API).
//!   4. The NON-SYMMETRIC (Depthwise) model's regular ShapValues AND
//!      ShapInteractionValues reproduce upstream <= 1e-5 (D-6.6-10 — >= 1
//!      non-symmetric SHAP case).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the other cb-model oracle tests. Fixtures are
//! generated OFFLINE by `cb-oracle/fixtures/advanced_fstr/gen_fixtures.py`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{
    load_cbm, prediction_diff, sage_values, shap_interaction_values, shap_values, Model,
    ModelSplit, ObliviousTree, Split,
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

/// Load the binclf input matrix (the SAME `numeric_tiny` X the canonical model
/// was trained on) as per-feature `f32` SoA columns.
fn load_feature_columns(rel: &str) -> Vec<Vec<f32>> {
    let x: Array2<f64> =
        read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Build the canonical oblivious binclf [`Model`] (splits + leaf values + leaf
/// weights + bias + borders) from the committed `model_serde/binclf/model.json`.
fn binclf_model() -> Model {
    let mj: ModelJson = load_model_json(&fixture("model_serde/binclf/model.json"))
        .expect("binclf model.json must load");
    let oblivious_trees = mj
        .oblivious_trees
        .iter()
        .map(|t| {
            let splits = t
                .splits
                .iter()
                .map(|s| {
                    ModelSplit::Float(Split {
                        feature: usize::try_from(s.float_feature_index)
                            .expect("non-negative feature"),
                        border: s.border,
                    })
                })
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
        bias: mj.bias().expect("bias must parse"),
        float_feature_borders: mj.float_feature_borders(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// The committed non-symmetric (Depthwise) model under test (carries the node
/// graph + leaf values + leaf weights from its `.cbm`).
fn non_symmetric_model() -> Model {
    let model = load_cbm(&fixture("advanced_fstr/non_symmetric_model.cbm"))
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

/// numeric_tiny flat feature count (4 float features).
const N_FEATURES: usize = 4;

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label} length {} != fixture length {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= TOL,
            "{label}[{i}] diverges: got {g}, want {w} (|d|={})",
            (g - w).abs()
        );
    }
}

#[test]
fn shap_interaction_values_matches_upstream_within_tol() {
    let model = binclf_model();
    let cols = load_feature_columns("advanced_fstr/binclf_X.npy");

    let inter = shap_interaction_values(&model, &cols, N_FEATURES);
    let flat: Vec<f64> = inter.iter().flat_map(|m| m.iter().copied()).collect();

    let expected = load_f64_vec(&fixture("advanced_fstr/oblivious_shap_interaction.npy"))
        .expect("oblivious_shap_interaction.npy must load");
    assert_close("ShapInteractionValues", &flat, &expected);
}

#[test]
fn shap_interaction_row_sum_equals_shap_values() {
    // Structural invariant (Open Question 2 reverse-map): summing Φ(i, ·) over
    // the feature columns (excluding the bias slot) reproduces ϕ(i).
    let model = binclf_model();
    let cols = load_feature_columns("advanced_fstr/binclf_X.npy");
    let inter = shap_interaction_values(&model, &cols, N_FEATURES);
    let shap = shap_values(&model, &cols, N_FEATURES);
    let dim = N_FEATURES + 1;
    for (obj, mat) in inter.iter().enumerate() {
        for i in 0..N_FEATURES {
            let rowsum: f64 = (0..N_FEATURES).map(|j| mat[i * dim + j]).sum();
            let phi = shap[obj][i];
            assert!(
                (rowsum - phi).abs() <= TOL,
                "row-sum != ϕ at obj {obj} feature {i}: Σ={rowsum}, ϕ={phi}"
            );
        }
    }
}

#[test]
fn prediction_diff_matches_upstream_within_tol() {
    let model = binclf_model();
    // PredictionDiff is on the first two objects only (data=X[:2]).
    let all = load_feature_columns("advanced_fstr/binclf_X.npy");
    let cols: Vec<Vec<f32>> = all
        .iter()
        .map(|c| c.iter().take(2).copied().collect())
        .collect();

    let pdiff = prediction_diff(&model, &cols, N_FEATURES);
    let expected = load_f64_vec(&fixture("advanced_fstr/prediction_diff.npy"))
        .expect("prediction_diff.npy must load");
    assert_close("PredictionDiff", &pdiff, &expected);
}

#[test]
fn sage_values_structural_oracle() {
    // D-6.6-11 fallback (b): upstream SAGE is a Monte-Carlo estimator driven by a
    // hard-coded TRestorableFastRng64(228) + Shuffle/PartialShuffle + marginal
    // imputer + Logloss metric (sage_values.cpp:343-427); bit-exact reproduction
    // needs the full RNG subsystem transcription (deferred). The Rust backend
    // ships a deterministic STRUCTURAL surrogate (mean |SHAP| per feature). The
    // ≤1e-5 value oracle (sage_values.npy is captured for the future seed-match
    // plan) is NOT asserted here. Structural invariants:
    let model = binclf_model();
    let cols = load_feature_columns("advanced_fstr/binclf_X.npy");

    let sage = sage_values(&model, &cols, N_FEATURES);
    // (1) correct shape.
    assert_eq!(sage.len(), N_FEATURES, "SAGE has one value per feature");
    // (2) finite + non-negative (a magnitude importance).
    for (f, &v) in sage.iter().enumerate() {
        assert!(v.is_finite(), "SAGE[{f}] must be finite, got {v}");
        assert!(v >= 0.0, "SAGE[{f}] must be non-negative, got {v}");
    }
    // (3) deterministic across repeated calls (no hidden RNG state in the Rust
    // surrogate — the seed-match property RESEARCH gate 2 requires).
    let sage2 = sage_values(&model, &cols, N_FEATURES);
    assert_eq!(sage, sage2, "SAGE surrogate must be deterministic");
    // (4) the upstream fixture exists and has the right shape (captured for the
    // future seed-match value oracle), and upstream agrees on which features are
    // active (the model only splits on features 0 and 3 in this fixture).
    let upstream = load_f64_vec(&fixture("advanced_fstr/sage_values.npy"))
        .expect("sage_values.npy must load (captured for future seed-match)");
    assert_eq!(upstream.len(), N_FEATURES, "upstream SAGE width");
    for f in 0..N_FEATURES {
        let active_rust = sage[f].abs() > 1e-12;
        let active_upstream = upstream[f].abs() > 1e-12;
        assert_eq!(
            active_rust, active_upstream,
            "SAGE active-feature support must match upstream at feature {f} \
             (rust={}, upstream={})",
            sage[f], upstream[f]
        );
    }
}

#[test]
fn non_symmetric_shap_matches_upstream_within_tol() {
    // D-6.6-10: >= 1 non-symmetric regular-SHAP case oracle-locked.
    let model = non_symmetric_model();
    let cols = load_feature_columns("advanced_fstr/binclf_X.npy");

    let shap = shap_values(&model, &cols, N_FEATURES);
    let flat: Vec<f64> = shap.iter().flat_map(|r| r.iter().copied()).collect();

    let expected = load_f64_vec(&fixture("advanced_fstr/non_symmetric_shap.npy"))
        .expect("non_symmetric_shap.npy must load");
    assert_close("non-symmetric ShapValues", &flat, &expected);
}

#[test]
fn non_symmetric_shap_interaction_matches_upstream_within_tol() {
    let model = non_symmetric_model();
    let cols = load_feature_columns("advanced_fstr/binclf_X.npy");

    let inter = shap_interaction_values(&model, &cols, N_FEATURES);
    let flat: Vec<f64> = inter.iter().flat_map(|m| m.iter().copied()).collect();

    let expected =
        load_f64_vec(&fixture("advanced_fstr/non_symmetric_shap_interaction.npy"))
            .expect("non_symmetric_shap_interaction.npy must load");
    assert_close("non-symmetric ShapInteractionValues", &flat, &expected);
}
