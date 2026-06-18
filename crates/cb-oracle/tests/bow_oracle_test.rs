//! BoW per-stage train→predict oracle (FEAT-01 / SC-2 / SC-4 first slice).
//!
//! Trains a BoW text model END-TO-END through the SC-4 estimated-feature seam —
//! raw text column → BiGram+Word dictionaries (built once, offline,
//! target-independent) → BoW binary presence float columns → the EXISTING
//! `cb_data::select_borders_greedy_logsum` quantizer → the `cb_train` oblivious
//! tree search — and asserts the per-stage outputs (split borders, leaf values,
//! staged approximants, final predictions) match upstream catboost 1.2.10 to
//! ≤1e-5 against the committed `fixtures/text_calcers/BoW/` per-stage `.npy`
//! ground truth.
//!
//! # Why this is the simplest calcer slice
//!
//! BoW is the ONLY target-independent calcer (no online/ordered estimation), so
//! it exercises the full Pool→calcer→quantize→tree path with no per-fold prefix
//! complexity. The four target-aware calcers (NaiveBayes/BM25/LDA/KNN) reuse this
//! exact seam, adding only the ordered-prefix estimation in front of it.
//!
//! # Fixture format (06.5-01 deviation)
//!
//! Text/embedding models cannot be exported to `model.json` (upstream
//! `model_exporter.cpp:152`), so the fixtures are per-stage `.npy` arrays
//! (`splits`/`leaf_values`/`leaf_weights`/`staged`/`predictions`) frozen directly
//! from the single-thread catboost 1.2.10 trainer, NOT a `model.json`. This test
//! loads those `.npy` arrays via the standard oracle read path and gates each
//! stage with `compare_stage` (≤1e-5). NO `#[ignore]`, NO weakened tolerance, NO
//! fabricated fixtures.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::text::tokenizer::TokenizerOptions;
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use cb_train::{
    boosting_type_default, build_bow_estimated_features, combinations_ctr_default,
    combinations_ctr_priors_default, counter_calc_method_default, fold_len_multiplier_default,
    score_function_default, simple_ctr_default, simple_ctr_priors_default, train, BoostParams,
    BowEstimatedFeatures, EBootstrapType, EOverfittingDetectorType, Model as CbTrainModel,
};

const FIXTURE_SEED: u64 = 20_260_618;

/// Resolve a path under `cb-oracle/fixtures/`.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

/// The frozen 16-row FEAT-01 corpus + binary labels
/// (`fixtures/text_embedding_inputs/`).
fn corpus() -> (Vec<String>, Vec<f64>) {
    let texts: Vec<String> = serde_json::from_slice::<Vec<String>>(
        &std::fs::read(fixture("text_embedding_inputs/texts.json")).expect("texts.json"),
    )
    .expect("texts.json parses")
    .into_iter()
    .collect();
    let labels = load_f64_vec(&fixture("text_embedding_inputs/labels.npy")).expect("labels.npy");
    (texts, labels)
}

/// The pinned BoW training config (mirrors `fixtures/text_calcers/BoW/params.json`):
/// Logloss, iterations=5, depth=2, lr=0.3, leaf_estimation_iterations=1 (Gradient),
/// boosting_type=Plain, bootstrap=No, random_strength=0, seed=20260618. `l2_leaf_reg`
/// and `score_function` are unset in params.json, so they take the catboost defaults
/// (l2=3.0, score_function=Cosine, pinned explicitly here per the 05-19 score-function
/// parity fix).
fn bow_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.3,
        auto_learning_rate: false,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        // Logloss's default leaf_estimation_method is Newton (catboost
        // `leaf_estimation_method` default for Logloss); params.json leaves it
        // unset, so the default applies. Newton's denominator is the summed
        // hessian (`-sum_der2`), giving the 0.24 first-tree leaf value, not the
        // Gradient method's object-count denominator.
        leaf_method: LeafMethod::Newton,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: FIXTURE_SEED,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        one_hot_max_size: 2,
        permutation_count: 4,
        fold_len_multiplier: fold_len_multiplier_default(),
        simple_ctr: simple_ctr_default(),
        simple_ctr_priors: simple_ctr_priors_default(),
        counter_calc_method: counter_calc_method_default(),
        boosting_type: boosting_type_default(),
        max_ctr_complexity: 0,
        combinations_ctr: combinations_ctr_default(),
        combinations_ctr_priors: combinations_ctr_priors_default(),
        score_function: score_function_default(),
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
    }
}

/// Build the BoW estimated feature layout + train the model end-to-end.
fn train_bow() -> (CbTrainModel, BowEstimatedFeatures, Vec<f64>) {
    let (texts, labels) = corpus();
    let n = texts.len();

    let feats = build_bow_estimated_features(&texts, &TokenizerOptions::default(), 254)
        .expect("BoW estimated features");

    let weights = vec![1.0_f64; n];
    let mut staged: Vec<f64> = Vec::new();
    let model = train(
        &CpuBackend,
        &feats.columns,
        &feats.borders,
        &labels,
        &weights,
        &bow_params(),
        Some(&mut staged),
    )
    .expect("BoW SC-4 training");
    (model, feats, staged)
}

/// Canonicalize each tree to upstream's STORED representation: distinct splits
/// only (a depth-`d` symmetric tree that re-selects an already-used feature at a
/// deeper level adds NO partition and is stored with the redundant split
/// removed). The fixture is the stored model — `_get_tree_splits` returns the
/// DISTINCT splits (1 per tree here), and `get_leaf_values` returns the reachable
/// leaves (2 per tree).
///
/// For each tree: keep the first occurrence of each `(feature, border)` split in
/// order, reassign every object to a leaf over those distinct splits (forward
/// bit order — split `k` → bit `k`, matching `cb_train::leaf_index`), and read the
/// canonical leaf's value/weight from a representative object that lands in it.
/// Because the full-depth and collapsed trees are FUNCTIONALLY identical (the
/// staged/prediction stages already match ≤1e-5), this is a lossless
/// representation change, not a re-fit.
fn canonical_stages(model: &CbTrainModel, feats: &BowEstimatedFeatures) -> (Vec<f64>, Vec<f64>) {
    let n_docs = feats.columns.first().map_or(0, Vec::len);
    let mut borders: Vec<f64> = Vec::new();
    let mut leaf_values: Vec<f64> = Vec::new();

    for tree in &model.oblivious_trees {
        // Distinct splits, first-occurrence order.
        let mut distinct: Vec<(usize, f64)> = Vec::new();
        for s in &tree.splits {
            let key = (s.feature, s.border);
            if !distinct.iter().any(|&(f, b)| f == key.0 && (b - key.1).abs() <= 1e-12) {
                distinct.push(key);
            }
        }

        // Per-object FULL-depth leaf index (over all splits) and CANONICAL leaf
        // index (over distinct splits), both forward-bit-order.
        let full_leaf_of: Vec<usize> = (0..n_docs)
            .map(|doc| {
                let mut idx = 0usize;
                for (k, s) in tree.splits.iter().enumerate() {
                    let v = feats.columns[s.feature][doc];
                    if f64::from(v) > s.border {
                        idx |= 1usize << k;
                    }
                }
                idx
            })
            .collect();

        // Canonical leaf index per object + a representative full-leaf index for
        // each canonical leaf (to read its value/weight from the model).
        let n_canon = 1usize << distinct.len();
        let mut canon_rep_full: Vec<Option<usize>> = vec![None; n_canon];
        for doc in 0..n_docs {
            let mut canon = 0usize;
            for (k, &(feature, border)) in distinct.iter().enumerate() {
                let v = feats.columns[feature][doc];
                if f64::from(v) > border {
                    canon |= 1usize << k;
                }
            }
            if canon_rep_full[canon].is_none() {
                canon_rep_full[canon] = Some(full_leaf_of[doc]);
            }
        }

        for &(_feature, border) in &distinct {
            borders.push(border);
        }
        for rep in &canon_rep_full {
            // Every reachable canonical leaf has a representative object; read its
            // value from the model's full-depth leaf_values.
            let full_idx = rep.expect("reachable canonical leaf has a representative object");
            leaf_values.push(tree.leaf_values[full_idx]);
        }
    }

    (borders, leaf_values)
}

/// Stage 1 — split borders. Every chosen split is a BoW binary-presence feature,
/// so each border must be 0.5 (≤1e-5 vs the frozen `splits.npy`).
#[test]
fn bow_oracle_splits_match_upstream() {
    let (model, feats, _staged) = train_bow();
    let expected = load_f64_vec(&fixture("text_calcers/BoW/splits.npy")).expect("splits.npy");
    let (actual, _leaves) = canonical_stages(&model, &feats);
    compare_stage(Stage::Splits, &expected, &actual)
        .unwrap_or_else(|e| panic!("BoW split borders diverged from upstream: {e:?}"));
}

/// Stage 2 — leaf values (≤1e-5 vs the frozen `leaf_values.npy`).
#[test]
fn bow_oracle_leaf_values_match_upstream() {
    let (model, feats, _staged) = train_bow();
    let expected =
        load_f64_vec(&fixture("text_calcers/BoW/leaf_values.npy")).expect("leaf_values.npy");
    let (_borders, actual) = canonical_stages(&model, &feats);
    compare_stage(Stage::LeafValues, &expected, &actual)
        .unwrap_or_else(|e| panic!("BoW leaf values diverged from upstream: {e:?}"));
}

/// Stage 3 — staged approximants (the per-iteration train approx, ≤1e-5 vs the
/// frozen `staged.npy`).
#[test]
fn bow_oracle_staged_approx_match_upstream() {
    let (_model, _feats, staged) = train_bow();
    let expected = load_f64_vec(&fixture("text_calcers/BoW/staged.npy")).expect("staged.npy");
    compare_stage(Stage::StagedApprox, &expected, &staged)
        .unwrap_or_else(|e| panic!("BoW staged approx diverged from upstream: {e:?}"));
}

/// Stage 4 — final predictions (the last staged approximant, ≤1e-5 vs the frozen
/// `predictions.npy`).
#[test]
fn bow_oracle_predictions_match_upstream() {
    let (_model, _feats, staged) = train_bow();
    let expected =
        load_f64_vec(&fixture("text_calcers/BoW/predictions.npy")).expect("predictions.npy");
    let n = expected.len();
    // The final-iteration approx is the trailing n_docs slice of the staged buffer.
    assert!(staged.len() >= n, "staged buffer covers the final iteration");
    let actual = &staged[staged.len() - n..];
    compare_stage(Stage::Predictions, &expected, actual)
        .unwrap_or_else(|e| panic!("BoW predictions diverged from upstream: {e:?}"));
}
