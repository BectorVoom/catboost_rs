//! Unit tests for `partial_dependence.rs` (FSTR-03 / PDP-01, PDP-02, PDP-05).
//! Sibling `#[path]` mount (source/test separation, CLAUDE.md), mirroring
//! `ctr_data_test.rs`.
//!
//! PDP-01 (averaging engine), PDP-02 (per-bin grid derivation) and PDP-05 (typed
//! validation) are unit-tested here. The PDP-03/PDP-04 upstream-parity oracles
//! live in `tests/partial_dependence_oracle_test.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use super::*;
use crate::model::{ModelSplit, ObliviousTree, Split};
use crate::predict_raw;
use cb_core::sum_f64;

/// A depth-1 oblivious tree splitting `feature` at `border`, with the
/// supplied (forward-bit-order) `leaf_values` (and matching-length zero
/// weights). Mirrors `predict_raw_multi_test.rs::one_split_tree`.
fn one_split_tree(feature: usize, border: f64, leaf_values: Vec<f64>) -> ObliviousTree {
    let n = leaf_values.len();
    ObliviousTree {
        splits: vec![ModelSplit::Float(Split { feature, border })],
        leaf_values,
        leaf_weights: vec![0.0; n],
    }
}

/// A small 2-float-feature model: one depth-1 tree splitting feature 0 at
/// `0.5`, two leaves `[lo, hi]`, `bias`. Feature 1 has no split (unused by the
/// tree) but is present in `float_feature_borders` so `n_float == 2`.
fn two_feature_model(lo: f64, hi: f64, bias: f64) -> Model {
    Model {
        oblivious_trees: vec![one_split_tree(0, 0.5, vec![lo, hi])],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![0.5], vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// AT-01a: for a 1-point grid `[v]`, the engine result equals the mean of
/// `predict_raw` computed independently in the test over columns whose target
/// feature column is entirely `v` — proves the override+average wiring.
#[test]
fn engine_averages_constant_grid_equals_direct_mean() {
    let model = two_feature_model(-1.25, 3.5, 0.42);
    let feature = 0;
    let grid = [0.7_f64];
    // Original columns: feature 0 varies, feature 1 is a decoy that must NOT
    // be overridden.
    let columns = vec![vec![0.1_f32, 0.9, 0.3], vec![10.0_f32, 20.0, 30.0]];

    // Independently-computed expected value: override column 0 to 0.7 for all
    // objects, keep column 1 untouched, predict, mean.
    let overridden = vec![vec![0.7_f32, 0.7, 0.7], columns[1].clone()];
    let expected_preds = predict_raw(&model, &overridden);
    let expected_mean = sum_f64(&expected_preds) / (expected_preds.len() as f64);

    let result = pdp_curve_single(&model, &columns, feature, &grid);

    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].to_bits(),
        expected_mean.to_bits(),
        "engine mean must be BIT-IDENTICAL to the independently-computed direct mean"
    );
}

/// AT-01b: output length == grid length; grid order preserved (verified by
/// constructing a grid whose per-point means are monotonically related to the
/// grid value, given the single split at 0.5).
#[test]
fn engine_output_length_and_order() {
    let model = two_feature_model(-10.0, 10.0, 0.0);
    let feature = 0;
    // Three grid points: two below the 0.5 border (both leaf lo == -10.0) and
    // one above (leaf hi == 10.0) — so the sequence is [-10.0, -10.0, 10.0],
    // NOT re-ordered.
    let grid = [0.1_f64, 0.2, 0.9];
    let columns = vec![vec![0.0_f32, 0.0, 0.0], vec![0.0_f32, 0.0, 0.0]];

    let result = pdp_curve_single(&model, &columns, feature, &grid);

    assert_eq!(result.len(), grid.len());
    assert_eq!(result[0].to_bits(), (-10.0_f64).to_bits());
    assert_eq!(result[1].to_bits(), (-10.0_f64).to_bits());
    assert_eq!(result[2].to_bits(), (10.0_f64).to_bits());
}

/// AT-01c: the `columns` argument passed to the engine is unchanged after the
/// call (no mutation of the caller's data — the override happens on a working
/// copy).
#[test]
fn engine_does_not_mutate_columns() {
    let model = two_feature_model(-1.0, 1.0, 0.0);
    let feature = 0;
    let grid = [0.1_f64, 0.9];
    let columns = vec![vec![0.2_f32, 0.6, 0.4], vec![5.0_f32, 6.0, 7.0]];
    let before = columns.clone();

    let _ = pdp_curve_single(&model, &columns, feature, &grid);

    assert_eq!(columns, before, "engine must not mutate the caller's columns");
}

// ---------------------------------------------------------------------------
// PDP-02 — per-bin grid derivation (T4). `grid_for_feature` is a deterministic
// construction from the model's borders (upstream exposes only bin indices, so
// there is no upstream numeric x-grid to oracle against — the values it feeds
// are what the PDP-03/04 oracles lock). n_borders borders -> n_borders+1 bins.
// ---------------------------------------------------------------------------

/// A model whose float feature 0 has the given `borders` (feature 1 present but
/// unused), so `grid_for_feature(&model, 0)` exercises the transform.
fn borders_model(borders_f0: Vec<f64>) -> Model {
    Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: vec![borders_f0, Vec::new()],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// AT-02: grid = `[b0-1, midpoints..., b_last+1]`, length `n_borders+1`, strictly
/// ascending; single-border and empty-borders edge cases.
#[test]
fn grid_for_feature_is_per_bin_representatives() {
    // 3 borders -> 4 bins: [1-1, (1+3)/2, (3+7)/2, 7+1] = [0, 2, 5, 8].
    let model = borders_model(vec![1.0, 3.0, 7.0]);
    let grid = grid_for_feature(&model, 0);
    assert_eq!(grid, vec![0.0, 2.0, 5.0, 8.0]);
    assert_eq!(grid.len(), 3 + 1);
    assert!(grid.windows(2).all(|w| w[0] < w[1]), "grid must be ascending");

    // 1 border -> 2 bins: [4-1, 4+1] = [3, 5].
    assert_eq!(grid_for_feature(&borders_model(vec![4.0]), 0), vec![3.0, 5.0]);

    // 0 borders (never split) -> benign single-point grid [0.0] (upstream would
    // reject; we do not error).
    assert_eq!(grid_for_feature(&borders_model(Vec::new()), 0), vec![0.0]);
}

// ---------------------------------------------------------------------------
// PDP-05 — typed input validation (T2). A T3-independent in-code Model with
// `n_float == 2` (a couple of non-empty border vecs, empty trees) so these
// arms stay genuinely parallel with the (skipped, oracle-blocked) T3 fixture
// task.
// ---------------------------------------------------------------------------

/// `n_float == 2`, no trees (validation arms only read
/// `float_feature_borders.len()`).
fn validation_model() -> Model {
    Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: vec![vec![0.1, 0.5], vec![0.2, 0.6]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// Valid rectangular non-empty columns matching `validation_model`'s
/// `n_float == 2` (used by AT-05d/AT-05e so checks 1-3 pass and the
/// range/duplicate check under test is isolated).
fn valid_columns() -> Vec<Vec<f32>> {
    vec![vec![0.0_f32, 1.0], vec![0.0_f32, 1.0]]
}

/// AT-05a: `features.len()` not in `{1, 2}` → `UnsupportedFeatureArity`.
#[test]
fn rejects_bad_arity() {
    let model = validation_model();
    let columns = valid_columns();

    match validate(&model, &columns, &[]) {
        Err(PdpError::UnsupportedFeatureArity { requested }) => assert_eq!(requested, 0),
        other => panic!("expected UnsupportedFeatureArity{{requested: 0}}, got {other:?}"),
    }

    match validate(&model, &columns, &[0, 1, 0]) {
        Err(PdpError::UnsupportedFeatureArity { requested }) => assert_eq!(requested, 3),
        other => panic!("expected UnsupportedFeatureArity{{requested: 3}}, got {other:?}"),
    }
}

/// AT-05b: malformed columns — `columns == []`, wrong width, and ragged
/// (unequal-length) columns — each → `MalformedColumns`. A valid-arity
/// `features` (`&[0]`) is passed so check 1 does not preempt this check.
#[test]
fn rejects_malformed_columns() {
    let model = validation_model();
    let n_float = model.float_feature_borders.len();

    // `columns == []`: malformed (0 != n_float), NOT EmptyDataset.
    match validate(&model, &[], &[0]) {
        Err(PdpError::MalformedColumns {
            expected_float_features,
            ..
        }) => assert_eq!(expected_float_features, n_float),
        other => panic!("expected MalformedColumns for columns==[], got {other:?}"),
    }

    // Wrong width: only 1 column when n_float == 2.
    let wrong_width = vec![vec![0.0_f32, 1.0]];
    match validate(&model, &wrong_width, &[0]) {
        Err(PdpError::MalformedColumns {
            expected_float_features,
            ..
        }) => assert_eq!(expected_float_features, n_float),
        other => panic!("expected MalformedColumns for wrong width, got {other:?}"),
    }

    // Ragged: correct width, unequal-length columns.
    let ragged = vec![vec![0.0_f32, 1.0, 2.0], vec![0.0_f32, 1.0]];
    match validate(&model, &ragged, &[0]) {
        Err(PdpError::MalformedColumns {
            expected_float_features,
            ..
        }) => assert_eq!(expected_float_features, n_float),
        other => panic!("expected MalformedColumns for ragged columns, got {other:?}"),
    }
}

/// AT-05c: correct width (`n_float` columns) but every column length 0 →
/// `EmptyDataset`. A valid-arity `features` (`&[0]`) is passed.
#[test]
fn rejects_empty_dataset() {
    let model = validation_model();
    let empty_but_right_width: Vec<Vec<f32>> = vec![Vec::new(), Vec::new()];

    match validate(&model, &empty_but_right_width, &[0]) {
        Err(PdpError::EmptyDataset) => {}
        other => panic!("expected EmptyDataset, got {other:?}"),
    }
}

/// AT-05d: valid rectangular non-empty columns, `features = [i]` with
/// `i >= n_float` → `FeatureIndexOutOfRange { index: i, n_float }`. Checks
/// 1-3 pass so the range check under test is isolated.
#[test]
fn rejects_out_of_range_feature() {
    let model = validation_model();
    let n_float = model.float_feature_borders.len();
    let columns = valid_columns();

    match validate(&model, &columns, &[n_float]) {
        Err(PdpError::FeatureIndexOutOfRange { index, n_float: nf }) => {
            assert_eq!(index, n_float);
            assert_eq!(nf, n_float);
        }
        other => panic!("expected FeatureIndexOutOfRange, got {other:?}"),
    }
}

/// AT-05e: valid rectangular non-empty columns, `features = [f, f]` (both in
/// range) → `DuplicateFeature { index: f }`. Checks 1-4 pass so the duplicate
/// check under test is isolated.
#[test]
fn rejects_duplicate_feature_pair() {
    let model = validation_model();
    let columns = valid_columns();

    match validate(&model, &columns, &[0, 0]) {
        Err(PdpError::DuplicateFeature { index }) => assert_eq!(index, 0),
        other => panic!("expected DuplicateFeature{{index: 0}}, got {other:?}"),
    }
}
