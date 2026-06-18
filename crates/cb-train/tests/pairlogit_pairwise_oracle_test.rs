//! PairLogitPairwise per-stage training oracle (LOSS-04 / Plan 06.3-09 Wave-5 gap
//! closure): train a plain boosted RANKING model under `Loss::PairLogitPairwise`
//! over the shared frozen ranking corpus and gate per-tree splits, per-tree leaf
//! values, per-iteration staged approximants, and final predictions against the
//! committed upstream catboost 1.2.10 `ranking_corpus/PairLogitPairwise` fixture at
//! <= 1e-5.
//!
//! PairLogitPairwise uses the SAME pairwise-logit der as `Loss::PairLogit` (it maps
//! to the same upstream `TPairLogitError`), but `IsPairwiseScoring` — so the leaf
//! VALUES are solved via the dedicated Cholesky pairwise-leaf system
//! (`pairwise_leaves.rs`) over the per-leaf pairwise weight sums + der sums, NOT the
//! pointwise Gradient/Newton estimators. `*Pairwise` losses force
//! `boosting_type = Plain` (`IsPlainOnlyModeLoss`). EXP-approx (same as PairLogit);
//! predictions are RAW.
//!
//! Pinned params mirror the fixture `model.json`: depth 2, 5 iterations,
//! learning_rate 0.3, l2_leaf_reg 5, leaf_estimation_iterations 1, boosting_type
//! Plain, bootstrap_type No, random_strength 0, thread_count 1, score_function
//! Cosine, leaf method Gradient (the upstream PairLogitPairwise default; the leaf
//! VALUES actually flow through the Cholesky pairwise path because
//! `is_pairwise_scoring` is true).
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
        // PairLogitPairwise fixture model.json pins l2_leaf_reg = 5.
        l2_leaf_reg: 5.0,
        random_strength: 0.0,
        boost_from_average: false,
        // PairLogitPairwise fixture leaf_estimation_method = Gradient; the leaf
        // VALUES actually flow through the Cholesky pairwise path (is_pairwise_scoring).
        leaf_method: LeafMethod::Gradient,
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
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

// CLOSED (gap #1, Plan 06.3-16): the pairwise split-scorer is now wired into the
// greedy oblivious tree search behind `is_pairwise_scoring` (Plan 06.3-15 built
// `calculate_pairwise_score`; commit 6aaa769 routed candidate scoring through it).
// This resolves the tree-0 split-1 SPLIT-SELECTION divergence (upstream f0@1.628 vs
// the prior pointwise-histogram f1@1.816) that was deferred under 06.3-13. The
// `#[ignore]` is removed and the full four-stage ≤1e-5 gate (Splits | LeafValues |
// StagedApprox | Predictions) runs against the genuine catboost 1.2.10
// `ranking_corpus/PairLogitPairwise` fixture (model_guid 7a8f259-…, tags/v1.2.10).
// No tolerance was weakened and the fixture is upstream training output, not
// hand-authored.
#[test]
fn pairlogit_pairwise_oracle_per_stage() {
    let scenario = "ranking_corpus/PairLogitPairwise";
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
        &base_params(Loss::PairLogitPairwise),
        Some(&mut staged),
        ranking,
    )
    .unwrap_or_else(|e| panic!("PairLogitPairwise training failed: {e:?}"));

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("PairLogitPairwise: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("PairLogitPairwise: leaf values diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("PairLogitPairwise: staged approx diverged: {e:?}"));

    let cb_model = CbModel::from_trained(&model, borders.clone());
    let predictions = predict_raw(&cb_model, &columns);
    assert_eq!(predictions.len(), N_ROWS);
    let expected_predictions = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("PairLogitPairwise: raw predictions diverged: {e:?}"));
}
