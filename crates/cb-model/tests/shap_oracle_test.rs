//! Regular TreeSHAP oracle (MODEL-04, D-11): build the canonical
//! [`cb_model::Model`] from the committed upstream `model_serde/binclf/model.json`
//! (the SAME `CatBoostClassifier(boost_from_average=False, **ISOLATING_PARAMS)`
//! model on `numeric_tiny` that produced the `feature_importance/shap_values.npy`
//! fixture — identical seed / params / inputs, so the trained trees, leaf values,
//! AND leaf weights coincide) and assert two things:
//!
//!   1. the per-object `[n_features + 1]` SHAP matrix reproduces the upstream
//!      `feature_importance/shap_values.npy` (n_rows × (n_features+1), flattened
//!      row-major; trailing column = Σ_trees meanValue + bias) at <= 1e-5, and
//!   2. the **local-accuracy invariant** (D-11 — the strongest check): for every
//!      object, `Σ_columns shap[obj] == predict_raw[obj]` at <= 1e-5.
//!
//! The local-accuracy invariant is asserted IN-ENV against a Rust-built model too
//! (it needs no fixture), so the correctness gate runs regardless of fixture
//! availability.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the other cb-model oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{load_cbm, predict_raw, shap_values, Model, ModelSplit, ObliviousTree, Split};
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

/// Load the `numeric_tiny` input matrix as per-feature `f32` SoA columns.
fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Build the canonical [`Model`] from an upstream [`ModelJson`], carrying splits,
/// per-tree `leaf_values` AND `leaf_weights` (SHAP needs the weights), the model
/// `bias`, and the per-float-feature borders.
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
    Model::new(
        oblivious_trees,
        mj.bias().expect("bias must parse"),
        mj.float_feature_borders(),
    )
}

/// The number of float features = max split feature index + 1 (the fixture's
/// `n_features` is the flat feature count, which for numeric_tiny equals the
/// float-feature count = 4).
const N_FEATURES: usize = 4;

#[test]
fn shap_matrix_matches_upstream_within_tol() {
    let mj = load_model_json(&fixture("model_serde/binclf/model.json"))
        .expect("binclf model.json must load");
    let model = model_from_json(&mj);
    let cols = load_feature_columns();

    let shap = shap_values(&model, &cols, N_FEATURES).expect("the fixture model is float-only");
    let flat: Vec<f64> = shap.iter().flat_map(|row| row.iter().copied()).collect();

    let expected = load_f64_vec(&fixture("feature_importance/shap_values.npy"))
        .expect("shap_values.npy must load");

    assert_eq!(
        flat.len(),
        expected.len(),
        "SHAP matrix length {} != fixture length {}",
        flat.len(),
        expected.len()
    );
    for (i, (&got, &want)) in flat.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "SHAP[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }
}

#[test]
fn shap_local_accuracy_holds_on_upstream_model() {
    let mj = load_model_json(&fixture("model_serde/binclf/model.json"))
        .expect("binclf model.json must load");
    let model = model_from_json(&mj);
    let cols = load_feature_columns();

    let shap = shap_values(&model, &cols, N_FEATURES).expect("the fixture model is float-only");
    let preds = predict_raw(&model, &cols);
    assert_eq!(shap.len(), preds.len(), "one SHAP row per object");

    for (obj, (row, &pred)) in shap.iter().zip(preds.iter()).enumerate() {
        let total = cb_core::sum_f64(row);
        assert!(
            (total - pred).abs() <= TOL,
            "local accuracy fails at obj {obj}: Σshap={total}, predict_raw={pred} (|d|={})",
            (total - pred).abs()
        );
    }
}

/// Local accuracy on a small hand-built Rust model (needs NO fixture): the
/// strongest SHAP invariant (D-11) — `Σ_columns shap[obj] == predict_raw[obj]` —
/// must hold for every object regardless of fixture availability in-env.
#[test]
fn shap_local_accuracy_holds_in_env_no_fixture() {
    // Two depth-2 oblivious trees over 2 float features, with nonzero bias and
    // realistic per-leaf weights.
    let tree0 = ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 0.5 }),
            ModelSplit::Float(Split { feature: 1, border: 1.5 }),
        ],
        leaf_values: vec![0.10, -0.20, 0.30, -0.05],
        leaf_weights: vec![10.0, 5.0, 8.0, 7.0],
    };
    let tree1 = ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 1, border: 0.5 }),
            ModelSplit::Float(Split { feature: 0, border: 2.5 }),
        ],
        leaf_values: vec![-0.01, 0.04, 0.02, 0.07],
        leaf_weights: vec![6.0, 9.0, 4.0, 11.0],
    };
    let model = Model::new(
        vec![tree0, tree1],
        0.123,
        vec![vec![0.5, 2.5], vec![0.5, 1.5]],
    );

    // A spread of objects covering every leaf.
    let cols: Vec<Vec<f32>> = vec![
        vec![0.0, 1.0, 3.0, 0.2, 2.0],
        vec![0.0, 2.0, 0.0, 1.0, 1.0],
    ];

    let shap = shap_values(&model, &cols, 2).expect("the fixture model is float-only");
    let preds = predict_raw(&model, &cols);
    assert_eq!(shap.len(), preds.len());
    for (obj, (row, &pred)) in shap.iter().zip(preds.iter()).enumerate() {
        assert_eq!(row.len(), 3, "row = [f0, f1, bias-term]");
        let total = cb_core::sum_f64(row);
        assert!(
            (total - pred).abs() <= TOL,
            "in-env local accuracy fails at obj {obj}: Σshap={total}, pred={pred}"
        );
    }
}

/// D-6.6-10: regular TreeSHAP on a NON-SYMMETRIC (Depthwise) model must
/// reproduce upstream `get_feature_importance(type='ShapValues')` ≤1e-5, AND the
/// local-accuracy invariant (`Σshap + bias == predict_raw`) must hold — the same
/// strongest check the oblivious path satisfies. Uses the committed
/// `advanced_fstr/non_symmetric_model.cbm` (carries the node graph + leaf
/// weights). The existing oblivious cases above are UNCHANGED.
#[test]
fn non_symmetric_shap_matches_upstream_and_local_accuracy() {
    let model = load_cbm(&fixture("advanced_fstr/non_symmetric_model.cbm"))
        .expect("non_symmetric_model.cbm must load");
    assert!(
        !model.non_symmetric_trees.is_empty() && model.oblivious_trees.is_empty(),
        "fixture must be a pure non-symmetric model"
    );
    let cols = load_feature_columns(); // numeric_tiny X (same inputs the model trained on)

    let shap = shap_values(&model, &cols, N_FEATURES).expect("the fixture model is float-only");
    let flat: Vec<f64> = shap.iter().flat_map(|row| row.iter().copied()).collect();

    let expected = load_f64_vec(&fixture("advanced_fstr/non_symmetric_shap.npy"))
        .expect("non_symmetric_shap.npy must load");
    assert_eq!(flat.len(), expected.len(), "non-symmetric SHAP length");
    for (i, (&got, &want)) in flat.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "non-symmetric SHAP[{i}] diverges: got {got}, want {want} (|d|={})",
            (got - want).abs()
        );
    }

    // Local accuracy: Σ_columns row == predict_raw[obj].
    let preds = predict_raw(&model, &cols);
    assert_eq!(shap.len(), preds.len(), "one SHAP row per object");
    for (obj, (row, &pred)) in shap.iter().zip(preds.iter()).enumerate() {
        let total = cb_core::sum_f64(row);
        assert!(
            (total - pred).abs() <= TOL,
            "non-symmetric local accuracy fails at obj {obj}: Σshap={total}, pred={pred} (|d|={})",
            (total - pred).abs()
        );
    }
}

// ── SPEC-OH-15 — no SHAP surface silently drops a non-float split ────────────

/// A depth-2 model whose level 0 is a float split and level 1 a one-hot split.
/// Under the pre-SPEC-OH-15 code `float_splits_of` dropped the one-hot level, so
/// every surface computed at `tree_depth == 1`: leaves `30.0`/`40.0` were
/// unreachable and `shap_values` summed to `2.0` instead of the real prediction
/// — a SILENT local-accuracy violation. Every surface must now refuse.
fn one_hot_depth2_model() -> Model {
    Model::new(
        vec![ObliviousTree {
                splits: vec![
                    ModelSplit::Float(Split { feature: 0, border: 0.5 }),
                    ModelSplit::OneHot(cb_model::OneHotModelSplit {
                        cat_feature: 0,
                        value_hash: 1_296_865_003,
                    }),
                ],
                leaf_values: vec![1.0, 2.0, 30.0, 40.0],
                leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
            }],
        0.0,
        vec![vec![0.5]],
    )
}

#[test]
fn shap_values_rejects_one_hot_models_instead_of_shrinking_tree_depth() {
    let model = one_hot_depth2_model();
    let cols = vec![vec![0.9_f32]];
    assert!(matches!(
        cb_model::shap_values(&model, &cols, 1),
        Err(cb_model::ShapUnsupported::OneHotSplits)
    ));
}

#[test]
fn shap_interaction_values_rejects_one_hot_models() {
    let model = one_hot_depth2_model();
    let cols = vec![vec![0.9_f32]];
    assert!(matches!(
        cb_model::shap_interaction_values(&model, &cols, 1),
        Err(cb_model::ShapUnsupported::OneHotSplits)
    ));
}

#[test]
fn prediction_diff_rejects_one_hot_models() {
    let model = one_hot_depth2_model();
    let cols = vec![vec![0.9_f32, 0.1]];
    assert!(matches!(
        cb_model::prediction_diff(&model, &cols, 1),
        Err(cb_model::ShapUnsupported::OneHotSplits)
    ));
}

#[test]
fn sage_values_rejects_one_hot_models() {
    let model = one_hot_depth2_model();
    let cols = vec![vec![0.9_f32, 0.1]];
    assert!(matches!(
        cb_model::sage_values(&model, &cols, 1),
        Err(cb_model::ShapUnsupported::OneHotSplits)
    ));
}

/// The SAME silent drop applied to CTR splits before SPEC-OH-15 — the guard is
/// structural (`float_splits_of` is fallible), so both non-float kinds are
/// refused, not just the newly added one.
#[test]
fn shap_values_rejects_ctr_models_too() {
    let mut model = one_hot_depth2_model();
    model.oblivious_trees[0].splits[1] = ModelSplit::Ctr(cb_model::CtrSplit {
        projection: cb_train::TProjection::from_features(&[0]),
        ctr_type: cb_model::ECtrType::Borders,
        prior: cb_model::Prior { num: 0.0, denom: 1.0 },
        target_border_idx: 0,
        border: 0.5,
        shift: 0.0,
        scale: 1.0,
    });
    let cols = vec![vec![0.9_f32]];
    assert!(matches!(
        cb_model::shap_values(&model, &cols, 1),
        Err(cb_model::ShapUnsupported::CtrSplits)
    ));
}
