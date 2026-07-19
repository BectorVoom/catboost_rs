//! `staged_predict` oracle (SP-04 / R1): [`cb_model::predict_raw_staged`]
//! reproduces upstream `catboost==1.2.10`
//! `model.staged_predict(X, prediction_type='RawFormulaVal', eval_period=k)`
//! within `<= 1e-5`, for a float-only oblivious `CatBoostRegressor`.
//!
//! Fixtures are generated OFFLINE by
//! `crates/cb-oracle/fixtures/staged_predict/gen_fixtures.py` (pinned seed,
//! `thread_count=1`, `bootstrap_type="No"`, `iterations=10`). The upstream stage
//! tree-counts were empirically confirmed (R1) and recorded in `config.json`:
//! `eval_period=1 -> {1..10}` (10 stages), `eval_period=3 -> {3,6,9,10}` (4
//! stages, always including the full ensemble as the final stage). Each expected
//! `.npy` is shaped `[n_stages, n_objects]`, row `j` = the cumulative RawFormulaVal
//! after `stage_tree_counts[j]` trees.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the other cb-model oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{load_cbm, predict_raw_staged, Model};
use cb_oracle::assert_abs_close;
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

/// The pinned float-only oblivious regressor, loaded from the committed `.cbm`.
fn staged_model() -> Model {
    load_cbm(&fixture("staged_predict/model.cbm")).expect("staged_predict/model.cbm loads")
}

/// `numeric_tiny` X as per-feature `f32` SoA columns — the layout
/// `predict_raw_staged` consumes.
fn numeric_tiny_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> =
        read_npy(fixture("inputs/numeric_tiny/X.npy")).expect("numeric_tiny/X.npy loads");
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Assert the Rust staged matrix reproduces the upstream `[n_stages, n_objects]`
/// fixture for a given `eval_period`, stage-count and per-stage value within TOL.
fn assert_period_matches(period: usize, expected_stages: usize) {
    let model = staged_model();
    let cols = numeric_tiny_columns();

    let expected: Array2<f64> = read_npy(fixture(&format!(
        "staged_predict/staged_period{period}.npy"
    )))
    .expect("staged expected matrix loads");
    assert_eq!(
        expected.nrows(),
        expected_stages,
        "fixture has the confirmed stage count for period {period}"
    );

    // ntree_start=0, ntree_end=0 (all trees), eval_period=period.
    let staged = predict_raw_staged(&model, &cols, 0, 0, period);
    assert_eq!(
        staged.len(),
        expected_stages,
        "Rust produced the confirmed stage count for period {period}"
    );

    for (j, actual_row) in staged.iter().enumerate() {
        let exp_row: Vec<f64> = expected.row(j).to_vec();
        assert_abs_close(&exp_row, actual_row, TOL)
            .unwrap_or_else(|e| panic!("period {period} stage {j} within TOL: {e}"));
    }
}

/// SP-04: `eval_period=1` reproduces the 10-stage upstream matrix (`{1..10}`).
#[test]
fn staged_predict_matches_upstream_period1() {
    assert_period_matches(1, 10);
}

/// SP-04: `eval_period=3` reproduces the 4-stage upstream matrix (`{3,6,9,10}`).
#[test]
fn staged_predict_matches_upstream_period3() {
    assert_period_matches(3, 4);
}

/// SP-04 (partial start): `ntree_start=2, eval_period=3` reproduces upstream's
/// 3-stage matrix at tree-counts `{5, 8, 10}`, where each stage sums ONLY trees
/// `[2, count)` and omits the model bias (upstream `[ntree_start, ntree_end)`
/// window). Oracle cover for the previously-untested `ntree_start > 0` path.
#[test]
fn staged_predict_matches_upstream_partial_start() {
    let model = staged_model();
    let cols = numeric_tiny_columns();

    let expected: Array2<f64> = read_npy(fixture("staged_predict/staged_start2_period3.npy"))
        .expect("partial-start expected matrix loads");
    assert_eq!(expected.nrows(), 3, "partial-start fixture has 3 stages ({{5,8,10}})");

    // ntree_start=2, ntree_end=0 (all), eval_period=3.
    let staged = predict_raw_staged(&model, &cols, 2, 0, 3);
    assert_eq!(staged.len(), 3, "Rust produced 3 partial-start stages");

    for (j, actual_row) in staged.iter().enumerate() {
        let exp_row: Vec<f64> = expected.row(j).to_vec();
        assert_abs_close(&exp_row, actual_row, TOL)
            .unwrap_or_else(|e| panic!("partial-start stage {j} within TOL: {e}"));
    }
}
