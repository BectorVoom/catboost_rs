//! YetiRank per-stage training oracle (LOSS-04 / Plan 06.3-04 Wave C): the
//! RNG-stream parity gate for the randomized YetiRank loss.
//!
//! # What this oracle gates (no weakening, no #[ignore])
//!
//! YetiRank has NO closed-form gradient — its pairwise weights are SAMPLED from a
//! `TFastRng64` stream whose exact draw COUNT + ORDER is the parity crux. The
//! ground truth is the OFFLINE instrumented generator
//! (`crates/cb-oracle/generator/yetirank_oracle.cpp`), which transcribes the
//! upstream RNG units verbatim and SELF-ORACLES bit-for-bit against the
//! oracle-locked `cb-core::TFastRng64` (see
//! `crates/cb-oracle/generator/instrument_ranking_rng_README.md`). Its captured
//! draw log is frozen at
//! `ranking_corpus/yetirank/yetirank_rng_groundtruth.jsonl`.
//!
//! `compare_stage` here gates the Rust `yetirank_sample_pairs` sampler's SAMPLED
//! COMPETITOR WEIGHTS against that frozen ground truth at <= 1e-5 — the integer/
//! f64-exact RNG-draw-log compare that gates the randomized stream INDEPENDENTLY
//! of the der (RESEARCH Pitfall 1: the der can match for permutations=1 yet
//! diverge at the default if the seed re-derivation is wrong). This assertion is
//! LIVE — it fails if the sampler's RNG draw order regresses.
//!
//! # End-to-end per-stage closure (06.3-14 ext — D-07 trainer-level RNG CLOSED)
//!
//! The FULL per-stage `compare_stage(Splits|LeafValues|StagedApprox|Predictions)`
//! over the frozen catboost 1.2.10 YetiRank `model.json` now PASSES at <= 1e-5
//! ([`yetirank_end_to_end_per_stage`]). The closure required reproducing the
//! trainer's per-tree RNG seed PLUMBING draw-for-draw
//! ([`cb_train::YetiRankTreeSeeder`]), the precise root cause the prior deferral
//! isolated. The actual model (verified against the instrumented trainer) is:
//!   * Per tree the persistent context RNG (`LearnProgress->Rand(random_seed)`)
//!     draws, IN ORDER: the structure-fold selection (1); the DERIVATIVE recalc
//!     seed (drives gradient + splits, learning fold); the per-level split-search
//!     draws (Rsm selection + `CalcScores` + `SelectBestCandidate` Box-Muller
//!     normals, consumed via `cb_core::std_normal` to advance the phase exactly);
//!     the LEARNING-fold approx-update recalc seed; and the AVERAGING-fold
//!     LEAF-VALUE recalc seed — THREE distinct YetiRank competitor re-samples per
//!     tree, NOT one. (The earlier "3 permutation folds" reading was the 3 RECALCS
//!     per tree; `fold_count == 1`.)
//!   * Each recalc partitions the query range into BLOCKS
//!     (`SetBlockCount(CB_THREAD_LIMIT=128)` ⇒ `block_count == n_groups` for the
//!     small corpus), drawing a per-block seed via
//!     `GenRandUI64Vector(n_groups, recalc_seed)` then one query seed per block
//!     ([`cb_train::derive_per_tree_query_seeds`]) — NOT the single shared block_rng
//!     the standalone self-oracle uses.
//!   * YetiRank is NOT `UseAveragingFoldAsFoldZero` (usePairs is true), so the
//!     LEARNING fold (gradient/structure) and AVERAGING fold (stored leaf values)
//!     carry SEPARATE approxes; the learning-fold approx is updated by the
//!     learnfold recalc WITHOUT `NormalizeLeafValues` (only `learning_rate`).
//!   * The sampler transcribes upstream's f32 bit-width (uniform cast to f32, f32
//!     Gumbel ratio, `TVector<TVector<float>>` competitor weights) — load-bearing
//!     for the end-to-end <= 1e-5 (an f64 sampler drifts ~1e-8 and flips a close
//!     split by tree 2).
//! The standalone full-precision RNG-draw oracle
//! ([`yetirank_rng_draw_log_oracle`]) stays GREEN throughout (it gates the
//! per-query Gumbel sampler against the single-block standalone ground truth via
//! [`cb_train::derive_query_seeds`], which is the correct chain for THAT generator).
//! The trainer fixture is OFFLINE/RUN-ONCE frozen (the 06.3-10 GO trainer); CI only
//! reads it. NO `#[ignore]`, NO weakened tolerance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::RankingCompetitor as Competitor;
use cb_compute::{LeafMethod, Loss};
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
const ITERATIONS: usize = 5;

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

fn yetirank_params(loss: Loss, leaf_method: LeafMethod) -> BoostParams {
    BoostParams {
        loss,
        iterations: ITERATIONS,
        depth: 2,
        learning_rate: 0.3,
        // YetiRank's upstream l2_leaf_reg default is ~0 (1e-20 in the fixture).
        l2_leaf_reg: 1e-20,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method,
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
    }
}

/// Parse the frozen `competitor` events from the instrumented ground-truth JSONL
/// into a (winner, loser, weight) list. The generator's smallest unit is one
/// group, 3 docs, permutations=10, decay=0.85, random_seed=0 — the SAME config
/// reproduced below.
fn load_competitor_ground_truth(rel: &str) -> Vec<(usize, usize, f64)> {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load (frozen RNG ground truth): {e:?}"));
    let mut out = Vec::new();
    for line in text.lines() {
        if !line.contains("\"event\":\"competitor\"") {
            continue;
        }
        let w = json_usize(line, "winner");
        let l = json_usize(line, "loser");
        let weight = json_f64(line, "weight");
        out.push((w, l, weight));
    }
    out
}

/// Parse the frozen `query_seed` event (the 2-level-derived per-query seed).
fn load_query_seed_ground_truth(rel: &str) -> u64 {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    for line in text.lines() {
        if line.contains("\"event\":\"query_seed\"") {
            // Parse the seed as a u64 DIRECTLY (not through f64 — a 64-bit seed
            // exceeds f64's 53-bit mantissa and would truncate, T-06.3-04-01).
            return json_raw(line, "seed").parse().unwrap();
        }
    }
    panic!("{rel}: no query_seed event in the ground-truth log");
}

// Minimal JSON field extractors (the log is a flat one-object-per-line schema; a
// full serde dependency is unnecessary for these fixed-shape lines).
fn json_usize(line: &str, key: &str) -> usize {
    json_raw(line, key).parse().unwrap()
}
fn json_f64(line: &str, key: &str) -> f64 {
    json_raw(line, key).parse().unwrap()
}
fn json_raw(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat).unwrap_or_else(|| panic!("key {key} not in {line}")) + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

/// LIVE RNG-draw oracle: the Rust `yetirank_sample_pairs` sampler must reproduce
/// the instrumented ground-truth sampled competitor weights AND the 2-level query
/// seed EXACTLY (<= 1e-5). This gates the randomized RNG stream independently of
/// the der — a desynced draw order fails here from the first sample.
#[test]
fn yetirank_rng_draw_log_oracle() {
    let gt_rel = "ranking_corpus/yetirank/yetirank_rng_groundtruth.jsonl";

    // The generator's smallest unit (yetirank_oracle.cpp main): one group, 3 docs,
    // rawApprox [0.5, -0.3, 0.1], relevs [2, 0, 1], permutations 10, decay 0.85,
    // random_seed 0. Reproduce the per-query seed via the 2-level chain.
    let raw_approx = [0.5_f64, -0.3, 0.1];
    let relevs = [2.0_f64, 0.0, 1.0];
    let permutations = 10_u32;
    let decay = 0.85_f64;
    let random_seed = 0_u64;

    // 2-level seed derivation (single group): the derived query seed MUST match the
    // instrumented log's query_seed event.
    let seeds = derive_query_seeds(random_seed, 1);
    let expected_seed = load_query_seed_ground_truth(gt_rel);
    assert_eq!(
        seeds[0], expected_seed,
        "YetiRank 2-level query seed must match the instrumented ground truth"
    );

    let competitors: Vec<Vec<Competitor>> =
        yetirank_sample_pairs(&raw_approx, &relevs, 1.0, permutations, decay, seeds[0]);

    // Flatten the Rust sampled adjacency into a (winner, loser, weight) matrix and
    // compare against the frozen ground truth (every nonzero edge, <= 1e-5).
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
    compare_stage(Stage::LeafValues, &expected, &got)
        .unwrap_or_else(|e| panic!("YetiRank: sampled competitor weights diverged from instrumented ground truth: {e:?}"));
}

/// END-TO-END per-stage YetiRank oracle (D-07 closure, 06.3-14 ext): train a Plain
/// boosted YetiRank model over the shared ranking corpus and gate per-tree splits,
/// per-tree leaf values, per-iteration staged approximants, and final predictions
/// against the frozen catboost 1.2.10 `ranking_corpus/yetirank` fixture at <= 1e-5.
///
/// This exercises the [`cb_train::YetiRankTreeSeeder`] per-tree multi-block seed
/// plumbing: the gradient/split competitor sample uses the derivative recalc seeds,
/// the leaf-value estimation uses a DISTINCT leaf-value recalc seed set, both
/// advanced off the persistent context RNG draw-for-draw with the trainer. YetiRank
/// rides the pointwise Newton leaf (model.json `leaf_estimation_method`). The
/// fixture is OFFLINE/RUN-ONCE-frozen (the instrumented 06.3-10 trainer); CI only
/// reads it. NO `#[ignore]`, NO weakened tolerance.
#[test]
fn yetirank_end_to_end_per_stage() {
    let scenario = "ranking_corpus/yetirank";
    let columns = load_feature_columns();
    let target = load_f64_vec(&fixture("ranking_corpus/inputs/y.npy")).unwrap();
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
    let loss = Loss::YetiRank { permutations: 10, decay: 0.85 };

    let mut staged = Vec::new();
    let model = train_ranking(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &yetirank_params(loss, LeafMethod::Newton),
        Some(&mut staged),
        ranking,
    )
    .unwrap_or_else(|e| panic!("YetiRank training failed: {e:?}"));

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("YetiRank: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("YetiRank: leaf values diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("YetiRank: staged approx diverged: {e:?}"));

    let cb_model = CbModel::from_trained(&model, borders.clone());
    let predictions = predict_raw(&cb_model, &columns);
    assert_eq!(predictions.len(), N_ROWS);
    let expected_predictions = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("YetiRank: raw predictions diverged: {e:?}"));
}
