//! PairLogit per-stage training oracle (LOSS-04 / Plan 06.3-09 Wave-5 gap
//! closure): train a plain boosted RANKING model under `Loss::PairLogit` over the
//! shared frozen ranking corpus (5 contiguous query groups over 12 objects,
//! `group_id` / `subgroup_id` / explicit pairs) and gate per-tree splits, per-tree
//! leaf values, per-iteration staged approximants, and final predictions against
//! the committed upstream catboost 1.2.10 `ranking_corpus/PairLogit` fixture at
//! <= 1e-5.
//!
//! PairLogit is the pairwise logistic loss over explicit `Pool.pairs` (EXP-approx;
//! cb-train stores the RAW approx and computes `exp()` INLINE in the der). It rides
//! the POINTWISE leaf estimator (NOT the Cholesky pairwise path —
//! `IsPairwiseScoring` is false for the non-`Pairwise` variant). The pair weight
//! enters the der/leaf via `Competitor.weight`, which `build_query_info` populates
//! with the upstream pair-weight normalization (06.3-09 Task 1).
//!
//! Pinned params mirror the fixture `model.json`: depth 2, 5 iterations,
//! learning_rate 0.3, l2_leaf_reg 3, leaf_estimation_iterations 1, boosting_type
//! Plain, bootstrap_type No, random_strength 0, thread_count 1, score_function
//! Cosine, leaf method Newton (the upstream PairLogit default). Predictions are RAW
//! (identity — the ranking score; no link transform on apply).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`. The fixture
//! is produced OFFLINE / frozen by upstream catboost 1.2.10 (RUN-ONCE/COMMIT,
//! D-08); this plan does NOT regenerate it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::Pair;
use cb_model::{predict_raw, Model as CbModel};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train_ranking, BoostParams, EBootstrapType, EOverfittingDetectorType, RankingData};
use ndarray::Array2;
use ndarray_npy::read_npy;

const N_ROWS: usize = 12;
const ITERATIONS: usize = 5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("ranking_corpus/inputs/X.npy"))
        .unwrap_or_else(|e| panic!("ranking_corpus/inputs/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

fn load_u64_vec(rel: &str) -> Vec<u64> {
    let arr: ndarray::Array1<u64> =
        read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    arr.to_vec()
}

fn load_pairs() -> Vec<Pair> {
    let arr: Array2<u32> = read_npy(fixture("ranking_corpus/inputs/pairs.npy"))
        .unwrap_or_else(|e| panic!("pairs.npy must load: {e:?}"));
    (0..arr.nrows())
        .map(|r| Pair {
            winner_id: arr[(r, 0)],
            loser_id: arr[(r, 1)],
        })
        .collect()
}

fn load_target() -> Vec<f64> {
    load_f64_vec(&fixture("ranking_corpus/inputs/y.npy")).unwrap()
}

fn base_params(loss: Loss) -> BoostParams {
    BoostParams {
        loss,
        iterations: ITERATIONS,
        depth: 2,
        learning_rate: 0.3,
        // PairLogit fixture model.json pins l2_leaf_reg = 3.
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        // PairLogit rides the pointwise Newton leaf (model.json leaf_estimation_method).
        leaf_method: LeafMethod::Newton,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 20_260_617,
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
        score_function: cb_compute::EScoreFunction::Cosine,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
    }
}

// CLOSED (06.3-13, GAP 1 / Truth #4): the PairLogit per-stage ≤1e-5 oracle now
// runs the FULL gate (Splits|LeafValues|StagedApprox|Predictions) with NO
// `#[ignore]` and NO weakened tolerance. The 06.3-10 instrumented catboost 1.2.10
// trainer (GO) captured the per-leaf SumDer/SumDer2 ground truth
// (`ranking_corpus/PairLogit/per_leaf_der_log.jsonl`), which proved TWO upstream
// facts the prior diagnosis missed:
//   1. The Newton denom is `-SumDer2 + l2*(sumAllWeights/docCount)` with
//      `sumAllWeights == docCount == 12` (the per-OBJECT document weight sum, NOT
//      the pairwise-weight total) — the 06.3-09 `sum_eff_weights` pairwise scaling
//      diverged Splits at index 6; `sum_all_weights` fixes it.
//   2. The "~6x" / "~23-denominator" anomaly was the MISSING `NormalizeLeafValues`
//      (`approx_updater_helpers.cpp:8-21`): for a pairwise loss upstream subtracts
//      the DOCUMENT-WEIGHTED mean leaf value (empty leaves forced to 0) BEFORE the
//      learning_rate scale. The raw per-leaf deltas were correct all along
//      (`leaf3` raw delta 0.1538 = 0.5/3.25); the centering is what makes the
//      stored values match model.json (verified ≤1e-9 against the frozen fixture).
#[test]
fn pairlogit_oracle_per_stage() {
    let scenario = "ranking_corpus/PairLogit";
    let columns = load_feature_columns();
    let target = load_target();
    let group_id = load_u64_vec("ranking_corpus/inputs/group_id.npy");
    let subgroup_id = load_u64_vec("ranking_corpus/inputs/subgroup_id.npy");
    let pairs = load_pairs();
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let ranking = RankingData {
        group_id: &group_id,
        subgroup_id: &subgroup_id,
        pairs: &pairs,
    };

    let mut staged = Vec::new();
    let model = train_ranking(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &base_params(Loss::PairLogit),
        Some(&mut staged),
        ranking,
    )
    .unwrap_or_else(|e| panic!("PairLogit training failed: {e:?}"));

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("PairLogit: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("PairLogit: leaf values diverged: {e:?}"));

    // REVIEW IN-01 (06.3-13): the model's per-tree `leaf_weights` are upstream's
    // `SumLeafWeights(GetWeights(TargetData))` — the per-OBJECT document weight sum
    // (counts here, all sample weight 1.0), NOT the pairwise-weight total. The
    // frozen PairLogit fixture stores integer document counts (tree0 `[8,3,0,1]`);
    // gate them ≤1e-5 now that this oracle executes (this is the SAME doc-weight
    // vector that feeds the `NormalizeLeafValues` weighted-mean centering).
    let expected_leaf_weights: Vec<f64> = model_json.leaf_weights().into_iter().flatten().collect();
    compare_stage(Stage::LeafValues, &expected_leaf_weights, &model.leaf_weights())
        .unwrap_or_else(|e| panic!("PairLogit: leaf weights (document counts) diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("PairLogit: staged approx diverged: {e:?}"));

    let cb_model = CbModel::from_trained(&model, borders.clone());
    let predictions = predict_raw(&cb_model, &columns);
    assert_eq!(predictions.len(), N_ROWS);
    let expected_predictions = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("PairLogit: raw predictions diverged: {e:?}"));
}
