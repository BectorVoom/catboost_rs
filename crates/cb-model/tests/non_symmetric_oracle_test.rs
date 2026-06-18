//! Non-symmetric (Lossguide / Depthwise) tree oracle harness (FEAT-06, Gate-B
//! Wave-0, D-6.6-02). Locks the per-stage parity targets for the leaf-wise
//! grower (06.6-04) and the pointer-walk apply (06.6-05) against the committed
//! catboost 1.2.10 fixtures.
//!
//! # Splits-first contract (RESEARCH Open Question 1)
//!
//! Each test asserts the per-stage gate in the LOCKED order: SPLITS FIRST (the
//! tree STRUCTURE — the leaf-wise grower's first hard lock), THEN LeafValues,
//! THEN the per-iteration StagedApprox, THEN the final Predictions, and finally
//! a `.cbm` / json round-trip equality check. A single split mismatch must be
//! rejected before any value stage runs.
//!
//! # Wave-0 failing-test-first status (EXPECTED TO FAIL until 04 + 05)
//!
//! At the end of THIS plan (06.6-03) the non-symmetric GROWER (06.6-04) and the
//! non-symmetric APPLY pointer-walk (06.6-05) are NOT yet wired:
//! [`cb_model::predict_raw`] walks only `oblivious_trees`, which are EMPTY for a
//! loaded non-symmetric model, so it returns the constant `bias` and the
//! Predictions stage DIVERGES from the upstream reference. These tests are
//! therefore EXPECTED TO FAIL now — that is the Nyquist Wave-0 contract. They are
//! deliberately NOT `#[ignore]`d; plans 06.6-04 (structure → splits/leaves) and
//! 06.6-05 (apply → staged/predictions) turn them green. The plan's automated
//! verify asserts only that this harness COMPILES (`--no-run`).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the other cb-model oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_model::{load_cbm, predict_raw, save_cbm, Model};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, ModelJson, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from cb-model's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load a non-symmetric fixture's `X.npy` as per-feature `f32` SoA columns.
fn load_feature_columns(scenario: &str) -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture(&format!("non_symmetric/{scenario}/X.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Path to a temporary `.cbm` for the round-trip stage.
fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cb_model_06603_{name}.cbm"));
    p
}

/// The shared splits-first oracle gate for one non-symmetric scenario.
///
/// Loads the committed upstream `model.cbm` (the canonical non-symmetric model),
/// asserts the per-stage parity ladder against the upstream `model.json` (splits
/// / leaves) and the reference `staged.npy` / `predictions.npy`, and finally the
/// `.cbm` round-trip equality.
fn assert_non_symmetric_oracle(scenario: &str) {
    // The upstream model.json carries the SPLIT borders + leaf values the
    // structure (06.6-04) and leaf-value (06.6-05) stages lock against.
    let mj: ModelJson = load_model_json(&fixture(&format!("non_symmetric/{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    assert!(
        mj.is_non_symmetric(),
        "{scenario} fixture must be a non-symmetric (`trees`) model (Pitfall 3)"
    );

    // The canonical model under test: load the committed non-symmetric `.cbm`.
    let model = load_cbm(&fixture(&format!("non_symmetric/{scenario}/model.cbm")))
        .unwrap_or_else(|e| panic!("{scenario}/model.cbm must load: {e:?}"));
    assert!(
        !model.non_symmetric_trees.is_empty(),
        "{scenario} .cbm must decode into non_symmetric_trees"
    );

    let columns = load_feature_columns(scenario);

    // ── Stage 1: SPLITS (the first hard lock, Open Question 1) ──────────────
    // The Rust model's per-node interior split borders must match the upstream
    // model.json split borders, tree-for-tree, BEFORE any value stage. (06.6-04
    // populates the grower; until then the loaded `.cbm` already carries the
    // upstream structure, so this stage is the structural contract.)
    let expected_splits = mj
        .non_symmetric_split_borders()
        .unwrap_or_else(|e| panic!("{scenario} split borders must extract: {e:?}"));
    let actual_splits: Vec<f64> = model
        .non_symmetric_trees
        .iter()
        .flat_map(|t| {
            t.tree_splits
                .iter()
                .zip(t.step_nodes.iter())
                .filter(|(_, &(l, r))| !(l == 0 && r == 0))
                .filter_map(|(s, _)| s.as_float().map(|f| f.border))
        })
        .collect();
    compare_stage(Stage::Splits, &expected_splits, &actual_splits)
        .unwrap_or_else(|e| panic!("{scenario} SPLITS stage diverged: {e:?}"));

    // ── Stage 2: LEAF VALUES ────────────────────────────────────────────────
    let expected_leaves = mj
        .non_symmetric_leaf_values()
        .unwrap_or_else(|e| panic!("{scenario} leaf values must extract: {e:?}"));
    let actual_leaves: Vec<f64> = model
        .non_symmetric_trees
        .iter()
        .flat_map(|t| t.leaf_values.iter().copied())
        .collect();
    compare_stage(Stage::LeafValues, &expected_leaves, &actual_leaves)
        .unwrap_or_else(|e| panic!("{scenario} LEAF VALUES stage diverged: {e:?}"));

    // ── Stage 3: STAGED APPROX (per-iteration) ─────────────────────────────
    // EXPECTED TO FAIL until 06.6-05 wires the non-symmetric apply pointer-walk:
    // `predict_raw` walks only oblivious trees today, so the staged sums diverge.
    let staged: Array2<f64> = read_npy(fixture(&format!("non_symmetric/{scenario}/staged.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/staged.npy must load: {e:?}"));
    let final_stage_expected: Vec<f64> = {
        let last = staged.nrows().saturating_sub(1);
        staged.row(last).iter().copied().collect()
    };
    let actual_predictions = predict_raw(&model, &columns);
    compare_stage(Stage::StagedApprox, &final_stage_expected, &actual_predictions)
        .unwrap_or_else(|e| panic!("{scenario} STAGED APPROX stage diverged: {e:?}"));

    // ── Stage 4: PREDICTIONS (final) ────────────────────────────────────────
    let expected_predictions =
        load_f64_vec(&fixture(&format!("non_symmetric/{scenario}/predictions.npy")))
            .unwrap_or_else(|e| panic!("{scenario}/predictions.npy must load: {e:?}"));
    compare_stage(Stage::Predictions, &expected_predictions, &actual_predictions)
        .unwrap_or_else(|e| panic!("{scenario} PREDICTIONS stage diverged: {e:?}"));

    // ── Stage 5: .cbm round-trip equality ───────────────────────────────────
    // Our save → load reproduces an identical model (bit-exact our-serialization
    // round-trip, D-6.6-05).
    let rt = tmp(scenario);
    save_cbm(&model, &rt).unwrap_or_else(|e| panic!("{scenario} save_cbm: {e:?}"));
    let reloaded: Model = load_cbm(&rt).unwrap_or_else(|e| panic!("{scenario} reload: {e:?}"));
    assert_eq!(
        model, reloaded,
        "{scenario} .cbm round-trip must reproduce an identical model"
    );
}

/// SPLITS-only preflight for the SIMPLEST possible Depthwise fixture
/// (06.6-04 Task 1, RESEARCH §"Open Questions (RESOLVED)" Q1).
///
/// This is the FIRST hard non-symmetric gate and is deliberately authored BEFORE
/// the leaf-wise grower (06.6-04 Task 2) lands, so any draw-stream divergence is
/// caught the instant the grower is wired — not mid-grower. The fixture
/// (`non_symmetric/depthwise_simplest/`, generated offline from catboost 1.2.10 by
/// `gen_depthwise_simplest_fixture.py`) pins EVERY confound OFF — `random_strength=0`,
/// `bootstrap_type='No'`, NO categorical features (no CTR), `thread_count=1`,
/// `boost_from_average=False`, a pinned seed, and the SMALLEST non-trivial
/// `max_depth=2` — so the ONLY thing that can make our SPLITS differ from upstream
/// is the Depthwise level-order expansion / candidate-enumeration draw stream
/// (RESEARCH Open Question 1).
///
/// Until the grower lands the loaded `.cbm` carries the upstream structure, so this
/// asserts the structural decode contract; once 06.6-04 Task 2/3 wire the grower
/// and the train→lift path, the SAME assertion locks our grower's SPLITS.
///
/// ESCALATION FALLBACK (D-6.6-11, escalate-don't-weaken): if this preflight diverges
/// once the grower lands, the non-symmetric draw stream differs from upstream. The
/// resolution is to ESCALATE to the persistent instrumented trainer
/// (`/tmp/cb_build313` + clang-18; see RESEARCH §Environment Availability and the
/// memory note "instrumented trainer toolchain persists") to capture the exact
/// upstream draw order. NEVER loosen the tolerance, NEVER `#[ignore]` this test,
/// NEVER fabricate splits. This empirical preflight is the early-warning realization
/// of RESEARCH Open Question 1.
#[test]
fn depthwise_simplest_splits() {
    let scenario = "depthwise_simplest";
    let mj: ModelJson = load_model_json(&fixture(&format!("non_symmetric/{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    assert!(
        mj.is_non_symmetric(),
        "{scenario} fixture must be a non-symmetric (`trees`) model (Pitfall 3)"
    );

    let model = load_cbm(&fixture(&format!("non_symmetric/{scenario}/model.cbm")))
        .unwrap_or_else(|e| panic!("{scenario}/model.cbm must load: {e:?}"));
    assert!(
        !model.non_symmetric_trees.is_empty(),
        "{scenario} .cbm must decode into non_symmetric_trees"
    );

    // SPLITS FIRST (the first hard lock, Open Question 1): the interior-node split
    // borders must match the upstream model.json borders, tree-for-tree.
    let expected_splits = mj
        .non_symmetric_split_borders()
        .unwrap_or_else(|e| panic!("{scenario} split borders must extract: {e:?}"));
    let actual_splits: Vec<f64> = model
        .non_symmetric_trees
        .iter()
        .flat_map(|t| {
            t.tree_splits
                .iter()
                .zip(t.step_nodes.iter())
                .filter(|(_, &(l, r))| !(l == 0 && r == 0))
                .filter_map(|(s, _)| s.as_float().map(|f| f.border))
        })
        .collect();
    compare_stage(Stage::Splits, &expected_splits, &actual_splits)
        .unwrap_or_else(|e| panic!("{scenario} SPLITS stage diverged: {e:?}"));
}

#[test]
fn depthwise_non_symmetric_oracle_splits_first() {
    assert_non_symmetric_oracle("depthwise");
}

#[test]
fn lossguide_non_symmetric_oracle_splits_first() {
    assert_non_symmetric_oracle("lossguide");
}
