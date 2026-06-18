//! YetiRankPairwise per-stage training oracle (LOSS-04 / Plan 06.3-04 Wave C).
//!
//! YetiRankPairwise shares the EXACT SAME sampled-pair RNG stream as
//! [`cb_compute::Loss::YetiRank`] (identical 2-level seed + Gumbel noise + Classic
//! weights) — the ONLY difference is the LEAF path: YetiRankPairwise solves leaf
//! values via the Cholesky pairwise-leaf system
//! (`is_pairwise_scoring` ⇒ `cb_train::pairwise_leaves`) and forces
//! `boosting_type = Plain`, where YetiRank rides the pointwise estimators. So the
//! RNG-draw ground truth (the parity crux) is the SAME frozen log
//! (`ranking_corpus/yetirank_pairwise/yetirank_rng_groundtruth.jsonl`, a copy of
//! the YetiRank stream), gated LIVE here against the Rust sampler.
//!
//! CLOSED (06.3-17): the end-to-end per-stage compare over the trained
//! YetiRankPairwise `model.json` through the Cholesky leaf path now passes all four
//! per-stage gates (Splits|LeafValues|StagedApprox|Predictions) at ≤1e-5. The two
//! gaps that previously deferred it are both resolved:
//!   1. The per-tree RNG seed-plumbing (`YetiRankTreeSeeder`) is calibrated
//!      draw-for-draw against the instrumented trainer's per-tree call-count
//!      fences (see `yetirank_pairwise_tree_rng_oracle_test.rs`). The crux was
//!      WR-02: the candidate-feature count must include EVERY quantized float
//!      feature (4), not just those with selected borders in the final model (3) —
//!      an unused-but-quantized feature still consumed an Rsm + normal draw per
//!      level.
//!   2. The pairwise SPLIT-scorer (`TPairwiseScoreCalcer`) landed in 06.3-15/16
//!      (the same scorer PairLogitPairwise uses).
//! Per D-6.3-03b: NO `#[ignore]`, NO weakened tolerance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{is_pairwise_scoring, is_plain_only, LeafMethod, Loss, RankingCompetitor as Competitor};
use cb_data::Pair;
use cb_model::{predict_raw, Model as CbModel};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    derive_query_seeds, train_ranking, yetirank_sample_pairs, BoostParams, EBootstrapType,
    EOverfittingDetectorType, RankingData,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

const N_ROWS: usize = 12;

fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("ranking_corpus/inputs/X.npy"))
        .unwrap_or_else(|e| panic!("X.npy must load: {e:?}"));
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
        .map(|r| Pair { winner_id: arr[(r, 0)], loser_id: arr[(r, 1)] })
        .collect()
}

fn json_raw(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat).unwrap_or_else(|| panic!("key {key} not in {line}")) + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

fn load_competitor_ground_truth(rel: &str) -> Vec<(usize, usize, f64)> {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    text.lines()
        .filter(|l| l.contains("\"event\":\"competitor\""))
        .map(|l| {
            (
                json_raw(l, "winner").parse().unwrap(),
                json_raw(l, "loser").parse().unwrap(),
                json_raw(l, "weight").parse().unwrap(),
            )
        })
        .collect()
}

/// YetiRankPairwise is routed to the PAIRWISE leaf path and forces Plain — the
/// loss-trait predicates must classify it correctly (a mis-route diverges leaf
/// values, RESEARCH Pitfall 2).
#[test]
fn yetirank_pairwise_uses_cholesky_leaf_and_plain() {
    let loss = Loss::YetiRankPairwise { permutations: 10, decay: 0.85 };
    assert!(is_pairwise_scoring(&loss), "YetiRankPairwise must use the Cholesky pairwise leaf path");
    assert!(is_plain_only(&loss), "YetiRankPairwise must force boosting_type = Plain");
    // YetiRank (non-pairwise) must NOT route to the Cholesky leaf.
    let plain = Loss::YetiRank { permutations: 10, decay: 0.85 };
    assert!(!is_pairwise_scoring(&plain), "YetiRank (pointwise) must NOT use the Cholesky leaf");
}

/// LIVE RNG-draw oracle: the sampler reproduces the frozen ground-truth sampled
/// competitor weights (the SAME stream as YetiRank, since the pair sampling is
/// identical — only the leaf differs).
#[test]
fn yetirank_pairwise_rng_draw_log_oracle() {
    let gt_rel = "ranking_corpus/yetirank_pairwise/yetirank_rng_groundtruth.jsonl";
    let raw_approx = [0.5_f64, -0.3, 0.1];
    let relevs = [2.0_f64, 0.0, 1.0];
    let seeds = derive_query_seeds(0, 1);
    let competitors: Vec<Vec<Competitor>> =
        yetirank_sample_pairs(&raw_approx, &relevs, 1.0, 10, 0.85, seeds[0]);

    let n = raw_approx.len();
    let mut got = vec![0.0_f64; n * n];
    for (winner, row) in competitors.iter().enumerate() {
        for c in row {
            got[winner * n + c.id] = c.weight;
        }
    }
    let mut expected = vec![0.0_f64; n * n];
    for (w, l, weight) in load_competitor_ground_truth(gt_rel) {
        expected[w * n + l] = weight;
    }
    compare_stage(Stage::LeafValues, &expected, &got).unwrap_or_else(|e| {
        panic!("YetiRankPairwise: sampled competitor weights diverged from ground truth: {e:?}")
    });
}

/// END-TO-END per-stage YetiRankPairwise compare (CLOSED, 06.3-17). Trains under
/// `Loss::YetiRankPairwise` with the Cholesky leaf path + the pairwise split-scorer
/// (06.3-15/16) over the per-tree-calibrated `YetiRankTreeSeeder` stream, and gates
/// all four stages (Splits|LeafValues|StagedApprox|Predictions) at ≤1e-5 against the
/// committed catboost 1.2.10 `model.json` / `staged.npy` / `predictions.npy`.
///
/// The final root cause was WR-02 (the candidate-feature undercount, fixed in
/// `boosting.rs`): the seeder drew an Rsm + normal per BORDERED feature (3) instead
/// of per QUANTIZED feature (4), short-changing the per-tree GTS draw count and
/// desyncing the learnfold/leafval recalc seeds from tree 1 onward. With the count
/// corrected the per-tree draw stream matches the instrumented trainer bit-exact
/// (`yetirank_pairwise_tree_rng_oracle_test.rs`) and the structure + leaf values +
/// staged approx + predictions all land within ≤1e-5. Per D-6.3-03b: NO `#[ignore]`,
/// NO weakened tolerance.
#[test]
fn yetirank_pairwise_end_to_end_per_stage() {
    let model_json = fixture("ranking_corpus/yetirank_pairwise/model.json");
    if model_json.exists() {
        // Closure landed (seed plumbing + pairwise split-scorer): wire the full
        // per-stage compare here (load model.json/staged/predictions, train_ranking
        // under Loss::YetiRankPairwise with the Gradient/Cholesky leaf, compare_stage
        // all four stages <= 1e-5) and remove this guard.
        let columns = load_feature_columns();
        let target = load_f64_vec(&fixture("ranking_corpus/inputs/y.npy")).unwrap();
        let group_id = load_u64_vec("ranking_corpus/inputs/group_id.npy");
        let subgroup_id = load_u64_vec("ranking_corpus/inputs/subgroup_id.npy");
        let pairs = load_pairs();
        let mj = load_model_json(&model_json).unwrap();
        let borders = mj.float_feature_borders();
        let ranking = RankingData { group_id: &group_id, subgroup_id: &subgroup_id, pairs: &pairs };
        let loss = Loss::YetiRankPairwise { permutations: 10, decay: 0.85 };
        let params = BoostParams {
            loss,
            iterations: 5,
            depth: 2,
            learning_rate: 0.3,
            l2_leaf_reg: 1e-20,
            random_strength: 0.0,
            boost_from_average: false,
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
        };
        let mut staged = Vec::new();
        let model = train_ranking(
            &CpuBackend, &columns, &borders, &target, &[], &params, Some(&mut staged), ranking,
        )
        .unwrap_or_else(|e| panic!("YetiRankPairwise training failed: {e:?}"));
        compare_stage(Stage::Splits, &mj.split_borders(), &model.split_borders())
            .unwrap_or_else(|e| panic!("YetiRankPairwise: splits diverged: {e:?}"));
        compare_stage(Stage::LeafValues, &mj.leaf_values(), &model.leaf_values())
            .unwrap_or_else(|e| panic!("YetiRankPairwise: leaf values diverged: {e:?}"));
        let expected_staged =
            load_f64_vec(&fixture("ranking_corpus/yetirank_pairwise/staged.npy")).unwrap();
        compare_stage(Stage::StagedApprox, &expected_staged, &staged)
            .unwrap_or_else(|e| panic!("YetiRankPairwise: staged approx diverged: {e:?}"));
        let cb_model = CbModel::from_trained(&model, borders.clone());
        let predictions = predict_raw(&cb_model, &columns);
        assert_eq!(predictions.len(), N_ROWS);
        let expected_predictions =
            load_f64_vec(&fixture("ranking_corpus/yetirank_pairwise/predictions.npy")).unwrap();
        compare_stage(Stage::Predictions, &expected_predictions, &predictions)
            .unwrap_or_else(|e| panic!("YetiRankPairwise: predictions diverged: {e:?}"));
    } else {
        // CLOSED (06.3-17): the YetiRankPairwise fixture is committed and the
        // present-fixture branch above runs the full 4-stage ≤1e-5 gate. A missing
        // model.json is now a regression (a removed/renamed fixture), not a deferred
        // state — fail loudly rather than silently pass.
        panic!(
            "YetiRankPairwise model.json fixture is missing — gap #2 is CLOSED and the \
             fixture must remain committed (escalate-don't-weaken, D-6.3-03b)"
        );
    }
}
