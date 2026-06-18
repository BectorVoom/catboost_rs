//! FEAT-06 / SC-3 grower → save_cbm → load_cbm → predict ≤1e-5 oracle (06.6-09).
//!
//! Closes the CR-02 gap: the pre-existing `non_symmetric_oracle_test` round-trip
//! only exercises an UPSTREAM-LOADED non-symmetric model (whose interior slots are
//! decoded as `u32::MAX` by cbm.rs:813) — it NEVER round-trips a model the Rust
//! leaf-wise grower produced in-memory. Before the 06.6-09 tree.rs fix, the grower
//! initialized `node_id_to_leaf_id` to `vec![0; node_count]`, so interior nodes were
//! mis-counted as leaves by the cbm serializer's `distinct_leaves` filter
//! (cbm.rs:200) and `save_cbm` failed with `ModelError::SchemaVersion`.
//!
//! This test trains a non-symmetric model via the Rust grower (`grow_policy=
//! Depthwise`), lifts it (`from_trained`), saves it (`save_cbm` — the CR-02 fix),
//! reloads it (`load_cbm`), and asserts the reloaded predictions match the in-memory
//! model ≤1e-5 (the headline SC-3 truth for the GROWER-produced path; D-6.6-02 — the
//! round-trip is part of the non-symmetric engine gate, not a follow-on).
//!
//! The model.json round-trip leg (WR-02, .cbm→json non-identity) is OUT of scope.
//! Do NOT `#[ignore]`, do NOT weaken the 1e-5 tolerance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_model::{load_cbm, predict_raw, save_cbm, Model as CbModel};
use cb_oracle::{load_model_json, ModelJson};
use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};
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

/// Load a non-symmetric fixture's `y.npy` regression target.
fn load_target(scenario: &str) -> Vec<f64> {
    let y: Array2<f64> = read_npy(fixture(&format!("non_symmetric/{scenario}/y.npy")))
        .map(|a: Array2<f64>| a)
        .or_else(|_| {
            // y may be saved 1-D; fall back to a 1-column read.
            read_npy::<_, ndarray::Array1<f64>>(fixture(&format!("non_symmetric/{scenario}/y.npy")))
                .map(|a| a.insert_axis(ndarray::Axis(1)))
        })
        .unwrap_or_else(|e| panic!("{scenario}/y.npy must load: {e:?}"));
    y.column(0).to_vec()
}

/// Build the simplest-Depthwise isolating [`BoostParams`] (mirrors the grower
/// oracle's pinned params: every confound OFF, `grow_policy=Depthwise`,
/// `max_depth=2`).
fn simplest_depthwise_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 2,
        depth: 2,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 42,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: EGrowPolicy::Depthwise,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

/// Path to a temporary `.cbm` for the grower round-trip (unique `06609` prefix so
/// it never collides with the 06603 round-trip test's temp files).
fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cb_model_06609_{name}.cbm"));
    p
}

/// SC-3 / FEAT-06 (CR-02): a non-symmetric model trained by the Rust leaf-wise
/// grower saves to `.cbm`, reloads, and re-predicts within 1e-5 of the in-memory
/// model. This is the grower→save→load→predict path the verifier found broken.
#[test]
fn non_symmetric_grower_save_load_predict_roundtrip() {
    let scenario = "depthwise_simplest";
    let columns = load_feature_columns(scenario);
    let target = load_target(scenario);
    let model_json: ModelJson =
        load_model_json(&fixture(&format!("non_symmetric/{scenario}/model.json")))
            .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    assert!(
        model_json.is_non_symmetric(),
        "{scenario} fixture must be a non-symmetric (`trees`) model"
    );
    let borders = model_json.float_feature_borders();

    // ── Train via the Rust leaf-wise grower (grow_policy=Depthwise) ───────────
    let params = simplest_depthwise_params();
    let model = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .unwrap_or_else(|e| panic!("{scenario}: non-symmetric training failed: {e:?}"));
    assert!(
        model.oblivious_trees.is_empty() && !model.non_symmetric_trees.is_empty(),
        "{scenario}: grow_policy=Depthwise must produce non_symmetric_trees \
         (oblivious={}, non_symmetric={})",
        model.oblivious_trees.len(),
        model.non_symmetric_trees.len()
    );

    // ── Lift into the canonical model ─────────────────────────────────────────
    let cb_model = CbModel::from_trained(&model, borders);
    assert!(
        cb_model.oblivious_trees.is_empty() && !cb_model.non_symmetric_trees.is_empty(),
        "{scenario}: from_trained must lift into non_symmetric_trees"
    );

    // ── Baseline: in-memory predictions the round-trip must reproduce ─────────
    let in_memory = predict_raw(&cb_model, &columns);

    // ── save_cbm — THE CR-02 fix. Before 06.6-09 this returned
    //    ModelError::SchemaVersion because interior nodes were mis-counted as
    //    leaves (node_id_to_leaf_id init was vec![0; …] instead of u32::MAX). ──
    let rt = tmp(scenario);
    save_cbm(&cb_model, &rt).unwrap_or_else(|e| {
        panic!(
            "{scenario}: save_cbm on a Rust-grower-trained non-symmetric model must \
             return Ok (this is the CR-02 fix — pre-fix it failed with \
             ModelError::SchemaVersion): {e:?}"
        )
    });

    // ── load_cbm → reload ─────────────────────────────────────────────────────
    let reloaded: CbModel = load_cbm(&rt)
        .unwrap_or_else(|e| panic!("{scenario}: load_cbm of the saved grower model: {e:?}"));
    assert!(
        !reloaded.non_symmetric_trees.is_empty(),
        "{scenario}: reloaded model must carry non_symmetric_trees"
    );

    // ── predict_raw on the reloaded model and compare ≤1e-5 (headline SC-3) ───
    let rt_predictions = predict_raw(&reloaded, &columns);
    assert_eq!(
        in_memory.len(),
        rt_predictions.len(),
        "{scenario}: prediction COUNT diverged (in_memory={}, reloaded={})",
        in_memory.len(),
        rt_predictions.len()
    );
    for (i, (a, b)) in in_memory.iter().zip(rt_predictions.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5,
            "{scenario}: grower→save→load→predict diverged at index {i}: \
             in_memory={a} reloaded={b} (>1e-5). The .cbm round-trip of a \
             Rust-grower-trained non-symmetric model must reproduce predictions \
             within 1e-5; do NOT weaken the tolerance, do NOT #[ignore]."
        );
    }
}
