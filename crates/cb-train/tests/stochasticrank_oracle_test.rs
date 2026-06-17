//! StochasticRank per-stage training oracle (LOSS-04 / Plan 06.3-04 Wave C;
//! gap #3 CLOSED in 06.3-18).
//!
//! StochasticRank is the OTHER RNG-stream ranking loss: a Monte-Carlo gradient
//! estimator whose per-doc Gaussian NOISE stream (drawn via `cb_core::std_normal`
//! from `TFastRng64(recalc_seed + group_index)`, `error_functions.h:1257`) is the
//! parity crux. Unlike YetiRank it has NO competitors — the only randomized input
//! is the per-group noise seed.
//!
//! # Three gates (no weakening, no #[ignore])
//!
//! 1. [`stochasticrank_rng_draw_log_oracle`] — the STANDALONE single-group
//!    self-oracle: the Rust `cb_core::std_normal` stream reproduces the frozen
//!    instrumented per-draw Gaussian noise
//!    (`stochasticrank_rng_groundtruth.jsonl`) at <= 1e-5. This gates the Gaussian
//!    SAMPLER independently of the trainer's seed plumbing. LIVE.
//!
//! 2. [`stochasticrank_pertree_noise_oracle`] — the D-07 TRAINER-LEVEL gate
//!    (06.3-18): the Rust per-tree per-group noise stream reproduces the
//!    instrumented catboost 1.2.10 trainer's per-tree draw log
//!    (`stochasticrank_pertree_noise_groundtruth.jsonl`, 110 events / 40 streams /
//!    10 base recalc seeds across 5 trees) bit-for-bit. This validates the per-tree
//!    context-RNG seed PLUMBING: each tree the persistent `LearnProgress->Rand`
//!    advances through the structure draw, the DERIVATIVE recalc base, the per-level
//!    split-search draws, the learning-fold base and the LEAF-VALUE recalc base —
//!    yielding TWO fresh per-tree recalc seeds (deriv + leafval). The per-tree
//!    main-RNG consumption is IDENTICAL to YetiRank's, so the SAME
//!    [`cb_train::YetiRankTreeSeeder`] drives both losses;
//!    StochasticRank consumes the raw `recalc_seeds[0]` (deriv) and `recalc_seeds[2]`
//!    (leafval) bases, then re-seeds each group's noise with `base + group_index`.
//!
//! 3. [`stochasticrank_end_to_end_per_stage`] — the FULL per-stage
//!    `compare_stage(Splits|LeafValues|StagedApprox|Predictions)` over the frozen
//!    catboost 1.2.10 StochasticRank `model.json` (metric=DCG, Gradient leaf), now
//!    PASSING at <= 1e-5. StochasticRank rides the querywise POINTWISE Gradient leaf
//!    (model.json `leaf_estimation_method`), single Monte-Carlo sample
//!    (`num_estimations = 1`), `sigma = 1`, `mu = 0`.
//!
//! ## Root causes closed (06.3-18, D-07)
//!
//! TWO distinct bugs, both isolated against the instrumented catboost 1.2.10 trainer:
//!
//! 1. **Per-tree noise seeding.** The PRIOR Rust path passed the FIXED
//!    `params.random_seed` to `compute_gradients_grouped` for EVERY tree — matching
//!    the standalone self-oracle but DIVERGING from the live trainer, whose noise
//!    re-seed advances PER TREE off the persistent context RNG and re-seeds the
//!    gradient/split recalc and the leaf-value recalc with DISTINCT bases. The
//!    `YetiRankTreeSeeder` per-tree advance + a leaf-value der re-compute in
//!    `boosting.rs` close that (the D-07 noise oracle proves it bit-exact).
//!
//! 2. **Per-query approx centering.** The StochasticRank der `mean` and SFA approx
//!    projection both read the per-query `approxes`, which the catboost trainer feeds
//!    GROUP-MEAN-CENTERED (the AveragingFold approx is maintained zero-mean per query;
//!    a ranking loss is shift-invariant within a query). The instrumented
//!    `srank_rawder.score`/`mean` fences showed catboost's `approxes[docId]` is the
//!    centered value, not the raw accumulated approx. Feeding the un-centered approx
//!    shifted the gradient by the per-query mean — invisible at tree 0 (uniform approx)
//!    but amplified by the ~1/0.0036 SFA projection into a >1e-5 leaf-value divergence
//!    from tree 1 on (count > 2) groups. `stochastic_rank_group_der` now centers the
//!    per-query approx at entry (`ranking_der.rs`).
//!
//! Together these close gap #3 / truth #5 + truth #7 StochasticRank.
//!
//! The trainer fixture is OFFLINE/RUN-ONCE frozen (the instrumented catboost 1.2.10
//! trainer); CI only reads it. NO `#[ignore]`, NO weakened tolerance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss, StochasticRankMetric};
use cb_compute::{
    STOCHASTIC_RANK_MU_DEFAULT, STOCHASTIC_RANK_NUM_ESTIMATIONS_DEFAULT, STOCHASTIC_RANK_SIGMA_DEFAULT,
};
use cb_core::{std_normal, TFastRng64};
use cb_data::Pair;
use cb_model::{predict_raw, Model as CbModel};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    train_ranking, BoostParams, EBootstrapType, EOverfittingDetectorType, RankingData,
    YetiRankTreeSeeder,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const N_ROWS: usize = 12;
const ITERATIONS: usize = 5;
const DEPTH: usize = 2;
const RANDOM_SEED: u64 = 20_260_617;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn json_raw(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat).unwrap_or_else(|| panic!("key {key} not in {line}")) + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

fn json_u64(line: &str, key: &str) -> u64 {
    json_raw(line, key).parse().unwrap()
}
fn json_usize(line: &str, key: &str) -> usize {
    json_raw(line, key).parse().unwrap()
}
fn json_f64(line: &str, key: &str) -> f64 {
    json_raw(line, key).parse().unwrap()
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

/// Per-group sizes in group order (group 0..5), derived from `group_id.npy`.
fn group_sizes() -> Vec<usize> {
    let group_id = load_u64_vec("ranking_corpus/inputs/group_id.npy");
    let mut sizes: Vec<usize> = Vec::new();
    let mut last: Option<u64> = None;
    for g in group_id {
        if last == Some(g) {
            *sizes.last_mut().unwrap() += 1;
        } else {
            sizes.push(1);
            last = Some(g);
        }
    }
    sizes
}

/// The StochasticRank loss the fixture was trained with (config.json
/// `StochasticRank:metric=DCG`; defaults `sigma = 1`, `mu = 0`,
/// `num_estimations = 1`).
fn stochasticrank_loss() -> Loss {
    Loss::StochasticRank {
        metric: StochasticRankMetric::Dcg,
        sigma: STOCHASTIC_RANK_SIGMA_DEFAULT,
        mu: STOCHASTIC_RANK_MU_DEFAULT,
        num_estimations: STOCHASTIC_RANK_NUM_ESTIMATIONS_DEFAULT,
    }
}

fn stochasticrank_params() -> BoostParams {
    BoostParams {
        loss: stochasticrank_loss(),
        iterations: ITERATIONS,
        depth: DEPTH,
        // The fixture's learning_rate is the f32-rounded 0.3 catboost stored
        // (`model.json` `learning_rate == 0.30000001192092896` == `0.3_f32 as f64`).
        // Using the exact f64 `0.3` introduces a ~4e-8 relative scale on every leaf
        // value, which the StochasticRank SFA approx-projection (`k = dot /
        // (sqrt(Σz²)+Nu)²`, a ~1/0.0036 amplifier) magnifies into a >1e-5 leaf-value
        // divergence by tree 1. Pin the f32 value the trainer actually used.
        learning_rate: f64::from(0.3_f32),
        // StochasticRank was trained with the catboost DEFAULT l2_leaf_reg (3.0,
        // model.json `tree_learner_options.l2_leaf_reg`) — NOT the ~0 the YetiRank /
        // LambdaMart fixtures pin. With uniform weights `scale_l2_reg(3, 12, 12) == 3`,
        // so each Gradient leaf denominator is `sum_weight + 3` (the per-leaf factor
        // that separates the prior 1e-20 run from the fixture).
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        // StochasticRank rides the pointwise Gradient leaf (model.json
        // `leaf_estimation_method` == Gradient, `der2 == 0`).
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: RANDOM_SEED,
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
    }
}

/// Parse the frozen `gauss_draw` noise values (doc order) from the standalone
/// single-group ground truth.
fn load_noise_ground_truth(rel: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load (frozen RNG ground truth): {e:?}"));
    text.lines()
        .filter(|l| l.contains("\"event\":\"gauss_draw\""))
        .map(|l| json_f64(l, "noise"))
        .collect()
}

/// One instrumented per-tree per-group noise event.
struct NoiseEvent {
    seed: u64,
    sample: usize,
    doc: usize,
    count: usize,
    noise: f64,
}

/// Parse the D-07 per-tree noise ground truth (`srank_noise` events at %.17g).
fn load_pertree_noise_ground_truth(rel: &str) -> Vec<NoiseEvent> {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load (frozen per-tree noise GT): {e:?}"));
    text.lines()
        .filter(|l| l.contains("\"event\":\"srank_noise\""))
        .map(|l| NoiseEvent {
            seed: json_u64(l, "seed"),
            sample: json_usize(l, "sample"),
            doc: json_usize(l, "doc"),
            count: json_usize(l, "count"),
            noise: json_f64(l, "noise"),
        })
        .collect()
}

/// LIVE RNG-draw oracle (standalone): the Rust `cb_core::std_normal` stream (the
/// SAME the StochasticRank der consumes) reproduces the instrumented single-group
/// ground-truth Gaussian noise draws EXACTLY (<= 1e-5). The generator's unit is one
/// group, num_estimations=1, group-0 seed = random_seed(5) + 0; one std_normal per
/// doc in ascending order.
#[test]
fn stochasticrank_rng_draw_log_oracle() {
    let gt_rel = "ranking_corpus/stochasticrank/stochasticrank_rng_groundtruth.jsonl";
    let group_seed = 5_u64; // random_seed 5 + group_index 0.

    let expected = load_noise_ground_truth(gt_rel);
    let count = expected.len();

    let mut rng = TFastRng64::from_seed(group_seed);
    let got: Vec<f64> = (0..count).map(|_| std_normal(&mut rng)).collect();
    assert_eq!(
        got.len(),
        expected.len(),
        "StochasticRank noise draw COUNT must match the instrumented ground truth \
         (one std_normal per doc per sample)"
    );
    compare_stage(Stage::StagedApprox, &expected, &got).unwrap_or_else(|e| {
        panic!("StochasticRank: Gaussian noise stream diverged from instrumented ground truth: {e:?}")
    });
}

/// D-07 TRAINER-LEVEL per-tree noise oracle (06.3-18): the Rust per-tree per-group
/// Gaussian noise stream reproduces the instrumented catboost 1.2.10 trainer's
/// per-tree draw log bit-for-bit (<= 1e-5). This validates the per-tree context-RNG
/// seed PLUMBING — the precise root cause the prior deferral isolated.
///
/// Model (verified bit-exact against the GT cluster bases): per tree the persistent
/// context RNG (`LearnProgress->Rand(random_seed)`) advances through the SAME draw
/// sequence as YetiRank (structure draw → derivative recalc base → per-level
/// split-search Rsm/score/normal draws → learning-fold base → leaf-value recalc
/// base). The DERIVATIVE recalc base is `recalc_seeds[0]`, the LEAF-VALUE recalc
/// base is `recalc_seeds[2]`. Each group's noise stream is re-seeded with
/// `recalc_base + group_index` (`error_functions.h:1257`); singleton groups
/// (`count <= 1`, e.g. group 3) draw NO noise.
#[test]
fn stochasticrank_pertree_noise_oracle() {
    let gt_rel = "ranking_corpus/stochasticrank/stochasticrank_pertree_noise_groundtruth.jsonl";
    let events = load_pertree_noise_ground_truth(gt_rel);
    assert!(
        !events.is_empty(),
        "per-tree noise ground truth must contain srank_noise events"
    );

    // Reproduce the trainer's per-tree main-RNG advance to derive the two per-tree
    // recalc bases (deriv + leafval) for all 5 trees. `n_features` is the count of
    // float features the trainer quantized as split candidates (WR-02: ALL listed
    // float features, including the unused-but-quantized one); `group_count` does
    // not affect the raw recalc bases the StochasticRank noise re-seed consumes.
    let sizes = group_sizes();
    let group_count = sizes.len();
    let model_json = load_model_json(&fixture("ranking_corpus/stochasticrank/model.json"))
        .unwrap_or_else(|e| panic!("stochasticrank/model.json must load: {e:?}"));
    let n_features = model_json.float_feature_borders().len();

    let mut seeder = YetiRankTreeSeeder::new(RANDOM_SEED, group_count, n_features, DEPTH);
    // Per tree: [deriv_base, leafval_base] in the trainer's recalc order.
    let mut recalc_bases: Vec<u64> = Vec::with_capacity(ITERATIONS * 2);
    for _tree in 0..ITERATIONS {
        let seeds = seeder.next_tree();
        recalc_bases.push(seeds.recalc_seeds[0]); // derivative recalc base
        recalc_bases.push(seeds.recalc_seeds[2]); // leaf-value recalc base
    }

    // For each recalc base, regenerate the per-group noise stream
    // (`TFastRng64(base + group_index)`, one std_normal per doc per sample) and
    // compare to the instrumented events bit-exact. The GT groups its events by the
    // per-group seed; we drive the SAME seeds from our reproduced bases.
    let num_estimations = STOCHASTIC_RANK_NUM_ESTIMATIONS_DEFAULT as usize;
    let mut expected: Vec<f64> = Vec::with_capacity(events.len());
    let mut got: Vec<f64> = Vec::with_capacity(events.len());

    for &base in &recalc_bases {
        for (group_index, &count) in sizes.iter().enumerate() {
            // Singleton groups draw no noise (the der is trivially zero); they emit
            // no events and consume no draws (the offset is still skipped upstream).
            if count <= 1 {
                continue;
            }
            let group_seed = base.wrapping_add(group_index as u64);
            let mut rng = TFastRng64::from_seed(group_seed);
            for sample in 0..num_estimations {
                for doc in 0..count {
                    let n = std_normal(&mut rng);
                    // Match the corresponding GT event (seed, sample, doc).
                    let ev = events
                        .iter()
                        .find(|e| e.seed == group_seed && e.sample == sample && e.doc == doc)
                        .unwrap_or_else(|| {
                            panic!(
                                "no GT srank_noise event for seed {group_seed} sample {sample} \
                                 doc {doc} (count {count})"
                            )
                        });
                    assert_eq!(
                        ev.count, count,
                        "GT event count {} != reproduced group size {count} for seed {group_seed}",
                        ev.count
                    );
                    expected.push(ev.noise);
                    got.push(n);
                }
            }
        }
    }

    assert_eq!(
        got.len(),
        events.len(),
        "reproduced per-tree noise draw COUNT ({}) must match the instrumented GT event count ({})",
        got.len(),
        events.len()
    );
    compare_stage(Stage::StagedApprox, &expected, &got).unwrap_or_else(|e| {
        panic!(
            "StochasticRank: per-tree per-group noise stream diverged from the instrumented \
             trainer ground truth (D-07): {e:?}"
        )
    });
}

/// END-TO-END per-stage StochasticRank oracle (D-07 closure, 06.3-18): train a Plain
/// boosted StochasticRank model (metric=DCG, Gradient leaf) over the shared ranking
/// corpus and gate per-tree splits, per-tree leaf values, per-iteration staged
/// approximants, and final predictions against the frozen catboost 1.2.10
/// `ranking_corpus/stochasticrank` fixture at <= 1e-5.
///
/// This exercises the per-tree StochasticRank noise seeding: the gradient/split der
/// uses the per-tree DERIVATIVE recalc base, the leaf-value der re-computes with a
/// DISTINCT LEAF-VALUE recalc base, both advanced off the persistent context RNG
/// draw-for-draw with the trainer (`boosting.rs` StochasticRank dispatch). The
/// fixture is OFFLINE/RUN-ONCE-frozen; CI only reads it. NO `#[ignore]`, NO weakened
/// tolerance.
#[test]
fn stochasticrank_end_to_end_per_stage() {
    let scenario = "ranking_corpus/stochasticrank";
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

    let mut staged = Vec::new();
    let model = train_ranking(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &stochasticrank_params(),
        Some(&mut staged),
        ranking,
    )
    .unwrap_or_else(|e| panic!("StochasticRank training failed: {e:?}"));

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("StochasticRank: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("StochasticRank: leaf values diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("StochasticRank: staged approx diverged: {e:?}"));

    let cb_model = CbModel::from_trained(&model, borders.clone());
    let predictions = predict_raw(&cb_model, &columns);
    assert_eq!(predictions.len(), N_ROWS);
    let expected_predictions =
        load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("StochasticRank: raw predictions diverged: {e:?}"));
}
