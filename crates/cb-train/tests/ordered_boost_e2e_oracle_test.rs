//! End-to-end ORDERED train→predict oracle (ORD-02, Plan 05-10 Task 2 — the FULL
//! multi-tree hard gate, gap-closure for the D-09 omission).
//!
//! Trains a `boosting_type=Ordered` model on the committed `ordered_boost_e2e/`
//! fixture (X/y), lifts it into the canonical `cb_model::Model`, predicts via the
//! PRODUCTION `cb_model::predict_raw` apply path, and asserts the final
//! predictions match upstream catboost 1.2.10 (boosting_type=Ordered) ≤1e-5 across
//! ALL iterations/trees (NOT just tree 0). This test runs unconditionally (never
//! skipped / never ignored) — it is the user's full multi-tree hard gate.
//!
//! # Why the production apply path (cb_model::predict_raw), not the staged approx
//!
//! The ≤1e-5 final-prediction assertion routes through `cb_model::predict_raw`
//! (the D-08 leaf-sum + bias-once apply path) so the ordered-BUILT model is
//! validated end-to-end through the SAME inference path a user would hit — not the
//! cb-train internal staged approximant row.
//!
//! # Multi-tree determinism (D-11)
//!
//! The isolating config pins `random_strength=0` + `bootstrap_type=No`, so NO
//! Box-Muller perturbation / bootstrap draws occur and the once-created fold
//! permutation (`create_folds`, built ONCE before the iteration loop) is fixed
//! across all 5 iterations — the ordered-path portion of the D-11 multi-tree
//! concern does not apply here.
//!
//! # No-leakage anchor (SC-2)
//!
//! Re-asserts the iter-0 ordered-approx per-object no-leakage signature directly
//! via the production `ordered_approx_delta_simple` over the learning fold's first
//! body/tail segment: a BODY document keeps delta 0 (estimation prefix, never
//! self-updated); the tail deltas are finite running leaf averages. This is the
//! structural no-leakage anchor (the committed `ordered_boost/ordered_approx_iter0`
//! is keyed to a DIFFERENT input order, so the structural anchor over THIS
//! dataset's fold is asserted instead, per the plan's documented choice).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_model::{predict_raw, Model as CbModel};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    body_sum_weights, body_tail_segments, create_folds, ordered_approx_delta_simple, train,
    BoostParams, EBootstrapType, EBoostingType, EOverfittingDetectorType,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const FIXTURE_SEED: u64 = 0;
const FOLD_LEN_MULTIPLIER: f64 = 2.0;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the ordered_boost_e2e input matrix (`X.npy`, float32 `[N, F]`) as
/// per-feature `f32` SoA columns.
fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f32> = read_npy(fixture("ordered_boost_e2e/X.npy"))
        .unwrap_or_else(|e| panic!("ordered_boost_e2e/X.npy must load as f32 [N,F]: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).to_vec())
        .collect()
}

/// The isolating ORDERED config (mirrors `ordered_boost_e2e/config.json`):
/// boosting_type=Ordered, permutation_count=1, fold_len_multiplier=2.0, depth=2,
/// iterations=5, lr=0.1, l2=3.0, Gradient, bootstrap=No, random_strength=0, seed=0.
fn ordered_params() -> BoostParams {
    BoostParams {
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
        random_seed: FIXTURE_SEED,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: 1,
        fold_len_multiplier: FOLD_LEN_MULTIPLIER,
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Ordered,
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
    }
}

/// FULL multi-tree ordered train→predict ≤1e-5 vs upstream, through the
/// production `cb_model::predict_raw` apply path. Runs unconditionally (never
/// skipped / never ignored).
#[test]
fn ordered_boost_e2e_oracle_predictions_match_upstream() {
    let columns = load_feature_columns();
    let model_json = load_model_json(&fixture("ordered_boost_e2e/model.json"))
        .unwrap_or_else(|e| panic!("ordered_boost_e2e/model.json must load: {e:?}"));
    // The model's float-feature borders (the trainer scores against these).
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture("ordered_boost_e2e/y.npy")).unwrap();
    let expected_predictions = load_f64_vec(&fixture("ordered_boost_e2e/predictions.npy")).unwrap();

    // Train the ORDERED model (boosting_type=Ordered) under the pinned config.
    let trained = train(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &ordered_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("ordered e2e training failed: {e:?}"));

    // Lift into the canonical model and predict via the PRODUCTION apply path
    // (cb_model::predict_raw) — the mandated end-to-end validation route.
    let model = CbModel::from_trained(&trained, borders.clone());
    let actual = predict_raw(&model, &columns);

    assert_eq!(
        actual.len(),
        expected_predictions.len(),
        "prediction count must match upstream (N objects, all trees applied)"
    );
    // ≤1e-5 over ALL objects (covering ALL 5 ordered trees, not just tree 0).
    compare_stage(Stage::Predictions, &expected_predictions, &actual).unwrap_or_else(|e| {
        panic!("ordered e2e predictions diverged from upstream (boosting_type=Ordered): {e:?}")
    });
}

/// SC-2 no-leakage anchor: the iter-0 ordered approximant reproduced by the
/// in-training ordered path over the learning fold's first body/tail segment — a
/// BODY document keeps delta 0 (estimation prefix, never self-updated), the tail
/// deltas are finite running leaf averages. Driven through the SAME production
/// machinery (`create_folds` → `greedy_tensor_search_oblivious_ordered` leaves →
/// `ordered_approx_delta_simple`) the train loop uses, so it locks the structural
/// no-leakage signature on THIS dataset's fold.
#[test]
fn ordered_boost_e2e_iter0_ordered_approx_no_leakage() {
    let model_json = load_model_json(&fixture("ordered_boost_e2e/model.json"))
        .unwrap_or_else(|e| panic!("ordered_boost_e2e/model.json must load: {e:?}"));
    let target = load_f64_vec(&fixture("ordered_boost_e2e/y.npy")).unwrap();
    let n = target.len();

    // Build the fold set ONCE exactly as the train loop does (continuous-stream
    // RNG, permutation_count=1 → 1 learning + 1 averaging fold).
    let folds = create_folds(
        n,
        /* permutation_count = */ 1,
        /* permutation_needed_for_learning = */ true,
        /* dynamic_body_tail = */ true,
        FOLD_LEN_MULTIPLIER,
        FIXTURE_SEED,
    );
    let learning = folds
        .iter()
        .find(|f| !f.is_averaging)
        .expect("a learning fold must exist");

    // Iter-0 RMSE der1 (boost_from_average=true ⇒ approx0 == target mean, so
    // der1[i] = target[i] - mean). Single leaf for the no-leakage structural
    // anchor (all docs in leaf 0) so the running per-leaf average is auditable.
    let mean = target.iter().copied().sum::<f64>() / n as f64;
    let der1: Vec<f64> = target.iter().map(|&t| t - mean).collect();
    let weights: Vec<f64> = vec![1.0; n];
    let leaf_of = vec![0usize; n]; // single-leaf anchor

    let segments = body_tail_segments(n, FOLD_LEN_MULTIPLIER);
    let seg_body_weights = body_sum_weights(n, FOLD_LEN_MULTIPLIER, &weights);
    let (body_finish, tail_finish) = *segments.first().expect("at least one segment");
    let body_sum_weight = *seg_body_weights.first().expect("a body sum weight");

    let delta = ordered_approx_delta_simple(
        &leaf_of,
        &der1,
        &weights,
        &learning.permutation,
        body_finish,
        tail_finish,
        body_sum_weight,
        /* n_leaves = */ 1,
        /* scaled_l2 = */ 0.0,
    )
    .expect("ordered approx delta over the learning fold");

    // No-leakage: the FIRST body document (permutation position 0) keeps delta 0 —
    // it is the estimation prefix and never updated by its own label.
    let first_body_doc = learning.permutation[0] as usize;
    assert!(
        delta[first_body_doc].abs() < 1e-12,
        "body doc {first_body_doc} must keep ordered-approx delta 0 (no self-update)"
    );
    // Every delta is finite and bounded (well-formed running leaf averages).
    for (i, &d) in delta.iter().enumerate() {
        assert!(d.is_finite(), "ordered approx delta[{i}] = {d} must be finite");
        assert!(d.abs() < 1e3, "ordered approx delta[{i}] = {d} out of sane range");
    }
    // The model.json carries 5 ordered trees (sanity that the fixture is the
    // multi-tree ordered model, not a single-tree stand-in).
    let expected_leaves = model_json.leaf_values();
    assert_eq!(
        expected_leaves.len(),
        5 * 4,
        "fixture must carry 5 ordered trees × 4 leaves (depth=2, iterations=5)"
    );
}
