//! Leaf-weights capture oracle (Phase-4 Plan 01, RESEARCH Pitfall 1): every
//! trained oblivious tree must carry per-leaf summed training-document weights
//! (`leaf_weights`, length `2^depth`). SHAP / PredictionValuesChange / Interaction
//! all weight on these; without them every downstream fstr/SHAP path silently
//! returns zeros.
//!
//! Two gates:
//!  1. IN-ENV invariant (RESEARCH A4): for the UNWEIGHTED `regression_skeleton`
//!     fixture each tree's leaf weights sum (per tree) to the training object
//!     count (an unweighted leaf weight == its document count).
//!  2. ORACLE lock (≤1e-5): when the upstream `regression_skeleton/model.json`
//!     carries `leaf_weights`, the Rust-captured per-tree leaf weights match the
//!     upstream `leaf_weights` flattened in tree order. If the fixture predates
//!     the leaf_weights regeneration (empty `leaf_weights()`), the oracle gate is
//!     skipped with a logged reason and only the in-env invariant runs (the
//!     offline fixture follow-up regenerates the fixture).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
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

/// Load the raw `numeric_tiny` target (regression `y`).
fn load_regression_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

/// Train the RMSE `regression_skeleton` model (the first-slice isolating params)
/// and return the trained model plus the training object count.
fn train_regression_skeleton() -> (Model, usize) {
    let columns = load_feature_columns();
    let model_json = load_model_json(&fixture("regression_skeleton/model.json"))
        .unwrap_or_else(|e| panic!("regression_skeleton/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_regression_target();
    let n = target.len();

    let params = BoostParams {
        loss: Loss::Rmse,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
    };

    let model = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .unwrap_or_else(|e| panic!("regression_skeleton: training failed: {e:?}"));
    (model, n)
}

#[test]
fn leaf_weights_per_tree_sum_equals_doc_count_unweighted() {
    let (model, n) = train_regression_skeleton();
    assert!(!model.oblivious_trees.is_empty(), "model must have trees");

    for (tree_idx, tree) in model.oblivious_trees.iter().enumerate() {
        assert_eq!(
            tree.leaf_weights.len(),
            tree.leaf_values.len(),
            "tree {tree_idx}: leaf_weights length must equal leaf count (2^depth)"
        );
        // RESEARCH A4: for unweighted training, each leaf weight is its document
        // count, so the per-tree sum equals the training object count.
        let sum: f64 = cb_oracle_sum(&tree.leaf_weights);
        assert!(
            (sum - n as f64).abs() <= 1e-9,
            "tree {tree_idx}: leaf_weights sum {sum} != doc count {n}"
        );
    }
}

#[test]
fn leaf_weights_oracle_lock_vs_upstream_model_json() {
    let (model, _n) = train_regression_skeleton();
    let model_json = load_model_json(&fixture("regression_skeleton/model.json")).unwrap();

    let expected: Vec<f64> = model_json
        .leaf_weights()
        .into_iter()
        .flatten()
        .collect();
    if expected.is_empty() {
        // Fixture predates the leaf_weights regeneration (offline follow-up).
        eprintln!(
            "SKIP oracle lock: regression_skeleton/model.json has no leaf_weights \
             (regenerate the fixture with the updated gen_fixtures.py); \
             in-env invariant test still gates leaf-weight capture."
        );
        return;
    }

    let actual: Vec<f64> = model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.leaf_weights.iter().copied())
        .collect();
    compare_stage(Stage::LeafValues, &expected, &actual)
        .unwrap_or_else(|e| panic!("leaf_weights diverged from upstream: {e:?}"));
}

/// Tiny local left-to-right sum (test-only; production sums route through
/// `cb_core::sum_f64`). Kept here so the test file pulls no extra dep.
fn cb_oracle_sum(values: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for &v in values {
        acc += v;
    }
    acc
}
