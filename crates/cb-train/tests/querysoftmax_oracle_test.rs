//! QuerySoftMax per-stage training oracle (LOSS-04 / Plan 06.3-02 Wave A): train a
//! plain boosted RANKING model under `Loss::QuerySoftMax { lambda, beta }` over the
//! shared frozen ranking corpus (5 contiguous query groups of varied size over 12
//! objects, `group_id` / `subgroup_id` / explicit pairs) and gate per-tree splits,
//! per-tree leaf values, per-iteration staged approximants, and final predictions
//! against the committed upstream catboost 1.2.10 `ranking_corpus/QuerySoftMax`
//! fixture at <= 1e-5.
//!
//! QuerySoftMax is QUERYWISE (raw approx, `isExpApprox == false`): the der is a
//! per-group softmax cross-entropy over `Beta·approx`, MAX-SHIFTED before exp
//! (the NaN guard). It rides the EXISTING pointwise Newton leaf estimator (no
//! pairwise Cholesky path) through the grouped seam (`compute_gradients_grouped`).
//! Predictions are RAW (identity), so the Predictions stage gates the
//! final-iteration raw staged approx via the production `cb_model::predict_raw`.
//!
//! Pinned params mirror the corpus (`ranking_corpus/inputs/meta.json`): depth 2,
//! 5 iterations, learning_rate 0.3, leaf_estimation_iterations 1, boosting_type
//! Plain, bootstrap_type No, random_strength 0, thread_count 1. `lambda` / `beta`
//! are pinned to their upstream defaults (`0.01` / `1.0`); the generator passes no
//! override, so the defaults flow into the fixture.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`. The fixture
//! is produced OFFLINE / frozen via
//! `gen_ranking_fixtures.py --loss QuerySoftMax` (RUN-ONCE/COMMIT, D-08).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss, QUERYSOFTMAX_BETA_DEFAULT, QUERYSOFTMAX_LAMBDA_DEFAULT};
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
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        // QuerySoftMax's upstream default leaf-estimation method is Gradient (NOT
        // Newton — confirmed in the fixture model.json `tree_learner_options`,
        // RESEARCH A4). QueryRMSE uses Newton; QuerySoftMax uses Gradient.
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
        // catboost CPU default split score is Cosine.
        score_function: cb_compute::EScoreFunction::Cosine,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
    }
}

#[test]
fn querysoftmax_oracle_per_stage() {
    let scenario = "ranking_corpus/QuerySoftMax";
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

    let loss = Loss::QuerySoftMax {
        lambda: QUERYSOFTMAX_LAMBDA_DEFAULT,
        beta: QUERYSOFTMAX_BETA_DEFAULT,
    };

    let mut staged = Vec::new();
    let model = train_ranking(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &base_params(loss),
        Some(&mut staged),
        ranking,
    )
    .unwrap_or_else(|e| panic!("QuerySoftMax training failed: {e:?}"));

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("QuerySoftMax: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("QuerySoftMax: leaf values diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("QuerySoftMax: staged approx diverged: {e:?}"));

    let cb_model = CbModel::from_trained(&model, borders.clone());
    let predictions = predict_raw(&cb_model, &columns);
    assert_eq!(predictions.len(), N_ROWS);
    let expected_predictions = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("QuerySoftMax: raw predictions diverged: {e:?}"));
}
