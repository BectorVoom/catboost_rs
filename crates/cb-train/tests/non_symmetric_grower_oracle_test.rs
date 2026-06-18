//! FEAT-06 non-symmetric leaf-wise grower SPLITS oracle (Phase 06.6 plan 04, Task 3):
//! train a non-symmetric tree (Depthwise) with the leaf-wise grower
//! ([`cb_train::leaf_wise_grower`] via `grow_policy=Depthwise`), lift it into the
//! canonical [`cb_model::Model`] (`TreeVariant::NonSymmetric`-shaped
//! `NonSymmetricTree`), and lock the chosen SPLITS against the committed catboost
//! 1.2.10 simplest-Depthwise fixture (RESEARCH §"Open Questions (RESOLVED)" Q1 —
//! splits locked FIRST, before leaf values).
//!
//! # Splits-first contract & the draw-stream question
//!
//! This is the realization of the Task-1 `depthwise_simplest_splits` preflight on
//! the GROWER side: the simplest-Depthwise fixture pins every confound OFF
//! (`random_strength=0`, `bootstrap_type='No'`, NO categorical features, single
//! thread, `boost_from_average=False`, pinned seed, `max_depth=2`), so the ONLY
//! thing that can make our SPLITS differ from upstream is the leaf-wise
//! candidate-enumeration / expansion draw stream (RESEARCH Open Question 1). The
//! comparison is on the SORTED multiset of split borders: a different node
//! visitation order between the upstream nested-`trees` JSON (pre-order DFS) and our
//! `AddSplit` node-id order is a representation detail, NOT a divergence — what the
//! gate catches is a WRONG border/feature being chosen (a genuine draw-stream
//! divergence).
//!
//! # ESCALATION FALLBACK (D-6.6-11, escalate-don't-weaken)
//!
//! If the chosen SPLITS diverge from upstream, the leaf-wise draw stream differs
//! from catboost 1.2.10. The resolution is to ESCALATE to the persistent
//! instrumented trainer (`/tmp/cb_build313` + clang-18; RESEARCH §Environment
//! Availability, memory note "instrumented trainer toolchain persists") to capture
//! the exact upstream candidate / expansion order. NEVER loosen the tolerance,
//! NEVER `#[ignore]`, NEVER fabricate splits.
//!
//! Leaf VALUES + the apply pointer-walk round-trip are locked in 06.6-05; the HARD
//! gate for THIS plan is SPLITS.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle` / `cb-model`;
//! lives in `cb-train/tests/` because only cb-train can TRAIN (the cb-model
//! `non_symmetric_oracle_test.rs` locks the `.cbm` decode representation; this locks
//! the grower's chosen splits). The top-line `#![allow(...)]` mirrors
//! `monotone_oracle_test.rs:38`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_model::Model as CbModel;
use cb_oracle::{load_model_json, ModelJson};
use cb_train::{
    train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType,
};
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

/// Build the simplest-Depthwise isolating [`BoostParams`] (mirrors the generator's
/// pinned params: every confound OFF, `grow_policy=Depthwise`, `max_depth=2`).
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

/// The chosen split borders of a non-symmetric tree, as a SORTED multiset (so the
/// comparison is independent of the upstream nested-DFS vs our AddSplit node order).
fn sorted_borders(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v
}

/// FEAT-06 SC-3 (structure): the leaf-wise grower's chosen SPLITS reproduce
/// catboost 1.2.10 on the simplest-Depthwise fixture (Open Question 1 — splits
/// locked first). Trains via `grow_policy=Depthwise`, lifts into the canonical
/// model, and compares the lifted non-symmetric split borders against the upstream
/// `model.json` borders as a sorted multiset.
#[test]
fn non_symmetric_depthwise_grower_splits_match_upstream() {
    let scenario = "depthwise_simplest";
    let columns = load_feature_columns(scenario);
    let target = load_target(scenario);
    let model_json: ModelJson =
        load_model_json(&fixture(&format!("non_symmetric/{scenario}/model.json")))
            .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    assert!(
        model_json.is_non_symmetric(),
        "{scenario} fixture must be a non-symmetric (`trees`) model (Pitfall 3)"
    );
    let borders = model_json.float_feature_borders();

    let params = simplest_depthwise_params();
    let model = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .unwrap_or_else(|e| panic!("{scenario}: non-symmetric training failed: {e:?}"));

    // The leaf-wise grower must have produced NON-symmetric trees, not oblivious.
    assert!(
        model.oblivious_trees.is_empty() && !model.non_symmetric_trees.is_empty(),
        "{scenario}: grow_policy=Depthwise must produce non_symmetric_trees \
         (oblivious={}, non_symmetric={})",
        model.oblivious_trees.len(),
        model.non_symmetric_trees.len()
    );

    // Lift into the canonical model (TreeVariant::NonSymmetric-shaped) — the
    // oblivious lift path stays byte-identical (D-6.6-05).
    let cb_model = CbModel::from_trained(&model, borders);
    assert!(
        cb_model.oblivious_trees.is_empty() && !cb_model.non_symmetric_trees.is_empty(),
        "{scenario}: from_trained must lift into non_symmetric_trees"
    );

    // Our chosen interior-node split borders (filter terminal (0,0) nodes).
    let actual: Vec<f64> = cb_model
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

    let expected = model_json
        .non_symmetric_split_borders()
        .unwrap_or_else(|e| panic!("{scenario} upstream split borders must extract: {e:?}"));

    assert_eq!(
        actual.len(),
        expected.len(),
        "{scenario}: split COUNT diverged (ours={}, upstream={}) — a draw-stream \
         divergence; ESCALATE to the instrumented trainer (D-6.6-11), do NOT weaken",
        actual.len(),
        expected.len()
    );

    let a = sorted_borders(actual);
    let e = sorted_borders(expected);
    for (i, (ai, ei)) in a.iter().zip(e.iter()).enumerate() {
        assert!(
            (ai - ei).abs() <= 1e-5,
            "{scenario}: split border {i} diverged: ours={ai} upstream={ei} (>1e-5). \
             This is a leaf-wise draw-stream divergence (RESEARCH Open Question 1); \
             ESCALATE to /tmp/cb_build313 (D-6.6-11), do NOT loosen the tolerance, \
             do NOT #[ignore]."
        );
    }
}
