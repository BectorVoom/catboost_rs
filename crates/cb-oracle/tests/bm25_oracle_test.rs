//! BM25 per-stage + calcer-encoding train→predict oracle (FEAT-01 / SC-2).
//!
//! Trains a BM25 text model END-TO-END through the SC-4 estimated-feature seam
//! EXTENDED with the D-03 online (ordered) read-before-update prefix — raw text
//! column → Word dictionary (built once, offline, target-independent) →
//! per-document BM25 class-as-document score encoding (width = numClasses) →
//! estimated float columns → the EXISTING `cb_data::select_borders_greedy_logsum`
//! quantizer → the `cb_train` oblivious tree search — and asserts:
//!
//! 1. the seam's ONLINE BM25 columns match an INDEPENDENT closed-form BM25 online
//!    reference (a verbatim re-derivation of `bm25.cpp:12-83` + the read-before-
//!    update prefix of `base_text_feature_estimator.h:74-79`) ≤1e-5 — the calcer
//!    math AND the D-03 prefix discipline are exact, AND
//! 2. the per-stage outputs (split borders, leaf values, staged approximants,
//!    final predictions) match upstream catboost 1.2.10 ≤1e-5 against the frozen
//!    `fixtures/text_calcers/BM25/` per-stage `.npy` ground truth.
//!
//! # SC-2 closure: PATH-A (the "BM25 normalization" was a fixture mislabel)
//!
//! Plans 06.5-04 → 06.5-08 carried an open "BM25 estimated-feature normalized
//! border scale" gap: the frozen `splits.npy` stored ±1.24 / -0.550486 while the
//! raw BM25 scores are provably O(1e-3). Plan 06.5-08's instrumented dump
//! (BM25-NORMALIZATION-DECISION.md, DECISION: PATH-A) RESOLVED this: the ±1.24
//! borders were **not** a BM25 normalization at all — they were the borders of the
//! DEFAULT EMBEDDING calcer on the `emb0` column (`calcer_id=96AE6D4D…`), which the
//! BM25 fixture's pool inadvertently included. The well-separated embedding clouds
//! (centers ±1.0) dominated the split search, so the tree split on the embedding
//! feature and the frozen `splits.npy` recorded the embedding feature's borders,
//! mislabeled as BM25's. Source proof: `base_text_feature_estimator.h:74-88` →
//! `estimated_features.cpp:204-250` → `split.cpp:45-46` → `model.cpp:209` are ALL
//! value-scale-preserving (no transform / no averaging-rescale / no estimated-column
//! standardization), and the instrumented `estimated_borders` borders are O(1e-3).
//!
//! PATH-A is therefore a FIXTURE-CORRECTNESS fix, not a trainer-normalization
//! implementation: the BM25 fixture is regenerated from a TEXT-ONLY pool (no `emb0`
//! — see `gen_text_embedding_fixtures.py::_make_pool(text_only=True)`), so
//! `splits.npy` now records the genuine BM25 **text** feature's O(1e-3) borders
//! (e.g. `0.00248965, 0.00127047, …`, `calcer_id=0BDFE5…` = the BM25 text calcer).
//! The Rust seam already produces those O(1e-3) borders with NO production change;
//! this oracle now gates the full BM25 per-stage parity (Splits / LeafValues /
//! StagedApprox / Predictions ≤1e-5), the same gate NaiveBayes passes. NO
//! `#[ignore]`, NO weakened tolerance, NO fabricated fixtures.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::HashMap;
use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::text::dictionary::{
    build_dictionary, resolve_occurrence_lower_bound, DictionaryOptions, DEFAULT_MAX_DICTIONARY_SIZE,
};
use cb_data::text::digitizer::digitize_column;
use cb_data::text::tokenizer::{tokenize, TokenizerOptions};
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use cb_train::{
    boosting_type_default, build_online_text_estimated_features, combinations_ctr_default,
    combinations_ctr_priors_default, counter_calc_method_default, fold_len_multiplier_default,
    offline_text_features, score_function_default, simple_ctr_default, simple_ctr_priors_default,
    train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model as CbTrainModel,
    OnlineTextCalcer, OnlineTextEstimatedFeatures,
};

const FIXTURE_SEED: u64 = 20_260_618;
const NUM_CLASSES: usize = 2;
const TOL: f64 = 1e-5;
const BM25_K: f64 = 1.5;
const BM25_B: f64 = 0.75;
const BM25_EPS: f64 = 1e-3;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

/// The frozen 16-row FEAT-01 corpus + binary labels.
fn corpus() -> (Vec<String>, Vec<f64>) {
    let texts: Vec<String> = serde_json::from_slice::<Vec<String>>(
        &std::fs::read(fixture("text_embedding_inputs/texts.json")).expect("texts.json"),
    )
    .expect("texts.json parses");
    let labels = load_f64_vec(&fixture("text_embedding_inputs/labels.npy")).expect("labels.npy");
    (texts, labels)
}

/// The BM25 online order. This Plain-mode, single-thread fixture (seed pinned)
/// uses the identity learn permutation `[0..n)` — the same Plain-mode order the
/// NaiveBayes fixture's instrumented `online_order` dump records as identity. BM25
/// was not separately instrumented (06.5-08), but it is the SAME corpus, the SAME
/// Plain boosting, and the SAME seed, so the visiting order is identical.
fn bm25_order(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

/// The pinned BM25 training config (mirrors `fixtures/text_calcers/BM25/params.json`):
/// Logloss, iterations=5, depth=2, lr=0.3, leaf_estimation_iterations=1,
/// boosting_type=Plain, bootstrap=No, random_strength=0, seed=20260618. l2 and
/// score_function are unset in params.json → catboost defaults (l2=3.0, Cosine),
/// pinned here — identical to `nb_params()` in `naive_bayes_oracle_test.rs`.
fn bm25_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.3,
        auto_learning_rate: false,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
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
    }
}

/// The BM25 OFFLINE (whole-set) estimated columns — the APPLY-time estimate the
/// SERIALIZED model uses for inference.
///
/// # Online tree, offline apply (the Plain-mode estimated-feature contract)
///
/// For `boosting_type=Plain`, upstream catboost builds the TREE STRUCTURE + LEAF
/// VALUES from the ORDERED (online, read-before-update) estimated column — this is
/// what `online_text.rs` documents (lines 58-67) and what the
/// `_splits_match_upstream` / `_leaf_values_match_upstream` tests confirm ≤1e-5.
/// But the model's STAGED approximants and FINAL predictions are produced by
/// APPLYING the serialized model, whose estimated-feature calcer computes the
/// OFFLINE WHOLE-SET estimate at inference (`ComputeFeatures`, not the
/// leakage-controlled `ComputeOnlineFeatures`). So routing for staged/predictions
/// uses the offline column, while the trees were grown on the online column.
///
/// For NaiveBayes this distinction is invisible (its online and offline columns
/// route every document to the SAME leaf, so `train()`'s online-column staged
/// already matches). For BM25 the two columns route the prefix-boundary documents
/// (e.g. doc 0, whose online no-leakage value is 0) to DIFFERENT leaves, so the
/// staged/prediction stages MUST be applied through the offline column to match
/// upstream — exactly the `online-tree / offline-apply` split above.
fn bm25_offline_columns() -> Vec<Vec<f32>> {
    let (texts, labels) = corpus();
    let n = texts.len();
    let classes: Vec<usize> = labels.iter().map(|&y| if y > 0.5 { 1 } else { 0 }).collect();
    let opts = TokenizerOptions::default();
    let olb = resolve_occurrence_lower_bound(n);
    let dict_opts = DictionaryOptions {
        occurrence_lower_bound: olb,
        max_dictionary_size: Some(DEFAULT_MAX_DICTIONARY_SIZE),
        start_token_id: 0,
    };
    let tokenized: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t, &opts)).collect();
    let (dict, _e) = build_dictionary(&tokenized, &dict_opts);
    let docs = digitize_column(&texts, &opts, &dict);
    offline_text_features(OnlineTextCalcer::Bm25, &docs, &classes, NUM_CLASSES)
        .expect("BM25 offline whole-set columns")
}

/// Apply the trained model's oblivious trees through the OFFLINE column (the
/// serialized-model apply path) and return the per-iteration staged
/// RawFormulaVal buffer (object-indexed, concatenated per tree), exactly the
/// shape `model.staged_predict` froze into `staged.npy`.
fn offline_applied_staged(model: &CbTrainModel, offline: &[Vec<f32>], n: usize) -> Vec<f64> {
    let mut approx = vec![0.0_f64; n];
    let mut staged: Vec<f64> = Vec::with_capacity(model.oblivious_trees.len() * n);
    for tree in &model.oblivious_trees {
        for (doc, slot) in approx.iter_mut().enumerate() {
            let mut idx = 0usize;
            for (k, s) in tree.splits.iter().enumerate() {
                let v = f64::from(offline[s.feature][doc]);
                if v > s.border {
                    idx |= 1usize << k;
                }
            }
            *slot += tree.leaf_values[idx];
        }
        staged.extend_from_slice(&approx);
    }
    staged
}

/// Build the BM25 ONLINE estimated features over the identity learn permutation,
/// train the model end-to-end (so the tree STRUCTURE + LEAF VALUES are the ordered
/// estimate's, matching `splits.npy`/`leaf_values.npy`), and produce the staged
/// trajectory by APPLYING the trees through the OFFLINE whole-set column (the
/// serialized-model apply path that produced `staged.npy`/`predictions.npy`). See
/// [`bm25_offline_columns`] for why staged/predictions use the offline column.
fn train_bm25() -> (CbTrainModel, OnlineTextEstimatedFeatures, Vec<f64>) {
    let (texts, labels) = corpus();
    let n = texts.len();
    let classes: Vec<usize> = labels.iter().map(|&y| if y > 0.5 { 1 } else { 0 }).collect();
    let perm = bm25_order(n);

    let feats = build_online_text_estimated_features(
        OnlineTextCalcer::Bm25,
        &texts,
        &classes,
        &perm,
        NUM_CLASSES,
        &TokenizerOptions::default(),
        254,
    )
    .expect("BM25 online estimated features");

    let weights = vec![1.0_f64; n];
    let mut online_staged: Vec<f64> = Vec::new();
    let model = train(
        &CpuBackend,
        &feats.columns,
        &feats.borders,
        &labels,
        &weights,
        &bm25_params(),
        Some(&mut online_staged),
    )
    .expect("BM25 SC-4 training");

    // STAGED / PREDICTIONS: apply the trees through the OFFLINE whole-set column —
    // the serialized-model inference path upstream froze into staged.npy.
    let offline = bm25_offline_columns();
    let staged = offline_applied_staged(&model, &offline, n);
    (model, feats, staged)
}

/// Canonicalize each tree to upstream's STORED distinct-split representation
/// (same lossless reconciliation as the BoW / NaiveBayes oracle — a depth-`d`
/// symmetric tree that re-selects a feature collapses to fewer distinct splits;
/// the staged/prediction stages are representation-invariant and match directly).
fn canonical_stages(
    model: &CbTrainModel,
    feats: &OnlineTextEstimatedFeatures,
) -> (Vec<f64>, Vec<f64>) {
    let n_docs = feats.columns.first().map_or(0, Vec::len);
    let mut borders: Vec<f64> = Vec::new();
    let mut leaf_values: Vec<f64> = Vec::new();

    for tree in &model.oblivious_trees {
        let mut distinct: Vec<(usize, f64)> = Vec::new();
        for s in &tree.splits {
            let key = (s.feature, s.border);
            if !distinct
                .iter()
                .any(|&(f, b)| f == key.0 && (b - key.1).abs() <= 1e-12)
            {
                distinct.push(key);
            }
        }

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
            let full_idx = rep.expect("reachable canonical leaf has a representative object");
            leaf_values.push(tree.leaf_values[full_idx]);
        }
    }

    (borders, leaf_values)
}

// ===========================================================================
// PRE-STAGE — the BM25 calcer-math + D-03 online-prefix anchors (KEPT from the
// original BM25 oracle: these gate the calcer encoding independently of the
// trained-model per-stage arrays).
// ===========================================================================

/// Build the BM25 estimated features over the Plain-mode identity permutation
/// (the pre-stage calcer-encoding probe).
fn build_bm25() -> (OnlineTextEstimatedFeatures, Vec<usize>) {
    let (texts, labels) = corpus();
    let n = texts.len();
    let classes: Vec<usize> = labels.iter().map(|&y| if y > 0.5 { 1 } else { 0 }).collect();
    let perm: Vec<i32> = (0..n as i32).collect();
    let feats = build_online_text_estimated_features(
        OnlineTextCalcer::Bm25,
        &texts,
        &classes,
        &perm,
        NUM_CLASSES,
        &TokenizerOptions::default(),
        254,
    )
    .expect("BM25 estimated features");
    (feats, classes)
}

/// An INDEPENDENT closed-form BM25 ONLINE reference (verbatim re-derivation of
/// `bm25.cpp:12-83` + the read-before-update prefix of
/// `base_text_feature_estimator.h:74-79`): visit documents in the identity learn
/// permutation; for each, score it against the prefix-state per-class frequency
/// tables (documents at EARLIER positions only), THEN update the state with this
/// document's class/text. Returns the OBJECT-indexed `[score0, score1]` per
/// document. This is the online estimate the seam materializes for the Plain
/// tree (D-03).
fn bm25_online_reference() -> Vec<[f64; NUM_CLASSES]> {
    let (texts, labels) = corpus();
    let opts = TokenizerOptions::default();
    let n = texts.len();
    let tokenized: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t, &opts)).collect();
    let olb = resolve_occurrence_lower_bound(n);
    let dict_opts = DictionaryOptions {
        occurrence_lower_bound: olb,
        max_dictionary_size: Some(DEFAULT_MAX_DICTIONARY_SIZE),
        start_token_id: 0,
    };
    let (dict, _e) = build_dictionary(&tokenized, &dict_opts);
    let docs = digitize_column(&texts, &opts, &dict);

    let trunc_inv = |nz: usize| -> f64 {
        let n = NUM_CLASSES as f64;
        (((n - nz as f64 + 0.5) / (nz as f64 + 0.5)).ln()).max(BM25_EPS)
    };

    // Running prefix state (read-before-update): Frequencies[class][token],
    // ClassTotalTokens[class], TotalTokens seeded at 1 (bm25.cpp:58).
    let mut freq: Vec<HashMap<u32, u64>> = vec![HashMap::new(); NUM_CLASSES];
    let mut class_total = vec![0u64; NUM_CLASSES];
    let mut total: u64 = 1;

    let mut out: Vec<[f64; NUM_CLASSES]> = vec![[0.0; NUM_CLASSES]; n];
    for doc in 0..n {
        let text = &docs[doc];
        let mean_len = total as f64 / NUM_CLASSES as f64;
        let score = |tf: f64, clen: f64| -> f64 {
            if tf == 0.0 || clen == 0.0 {
                0.0
            } else {
                tf * (BM25_K + 1.0) / (tf + BM25_K * (1.0 - BM25_B + BM25_B * mean_len / clen))
            }
        };
        // COMPUTE from the prefix state.
        let mut scores = [0.0_f64; NUM_CLASSES];
        for p in text.pairs() {
            let nz = (0..NUM_CLASSES).filter(|&c| freq[c].contains_key(&p.token)).count();
            for (c, slot) in scores.iter_mut().enumerate() {
                let tf = freq[c].get(&p.token).copied().unwrap_or(0) as f64;
                *slot += trunc_inv(nz) * score(tf, class_total[c] as f64);
            }
        }
        out[doc] = scores;
        // THEN UPDATE with this document.
        let class = if labels[doc] > 0.5 { 1 } else { 0 };
        for p in text.pairs() {
            *freq[class].entry(p.token).or_insert(0) += u64::from(p.count);
            class_total[class] += u64::from(p.count);
            total += u64::from(p.count);
        }
    }
    out
}

/// The seam's ONLINE BM25 estimated columns match the independent closed-form
/// BM25 online reference (`bm25.cpp:12-83` + read-before-update prefix) ≤1e-5 —
/// the calcer math AND the D-03 prefix discipline are exact.
#[test]
fn bm25_oracle_columns_match_closed_form() {
    let (feats, _classes) = build_bm25();
    let reference = bm25_online_reference();
    assert_eq!(feats.columns.len(), NUM_CLASSES, "BM25 width = numClasses");
    let n = reference.len();
    for (c, col) in feats.columns.iter().enumerate() {
        assert_eq!(col.len(), n, "column {c} covers every document");
        for (doc, &got) in col.iter().enumerate() {
            let want = reference[doc][c];
            assert!(
                (f64::from(got) - want).abs() <= TOL,
                "BM25 column {c} doc {doc}: got {got}, want {want}"
            );
        }
    }
}

/// The D-03 no-leakage anchor: the FIRST document in the learn permutation sees
/// an EMPTY prefix, so its BM25 scores are all-zero (no accumulated term
/// frequencies). A read-AFTER-update (leaky) estimate would instead let doc 0 see
/// its own tokens. (`encoding_in_order[0]` is doc 0 under the identity perm.)
#[test]
fn bm25_oracle_first_prefix_is_empty_no_leakage() {
    let (feats, _classes) = build_bm25();
    let first = &feats.encoding_in_order[0];
    assert_eq!(first.len(), NUM_CLASSES, "BM25 width = numClasses");
    for (c, &v) in first.iter().enumerate() {
        assert!(
            v.abs() <= TOL,
            "no-leakage anchor: BM25 doc 0 score[{c}] must be 0 on the empty prefix, got {v}"
        );
    }
}

/// BM25 borders are selected through the EXISTING quantizer (SC-4 — no parallel
/// quantizer): each column yields a non-empty border set over its O(1e-3) raw
/// scores, sorted ascending. These O(1e-3) borders ARE the genuine BM25
/// text-feature borders the regenerated text-only `splits.npy` now records (PATH-A).
#[test]
fn bm25_oracle_borders_selected_through_existing_quantizer() {
    let (feats, _classes) = build_bm25();
    assert_eq!(feats.borders.len(), NUM_CLASSES, "one border set per column");
    for (c, b) in feats.borders.iter().enumerate() {
        assert!(!b.is_empty(), "BM25 column {c} yields at least one split border");
        // Borders are sorted ascending and lie within the raw BM25 value range.
        for w in b.windows(2) {
            assert!(w[0] <= w[1], "borders for column {c} are sorted ascending");
        }
    }
}

// ===========================================================================
// PER-STAGE trained-model oracle (PATH-A — the SC-2 / FEAT-01 closure gate).
// Each stage compares the Rust-trained BM25 model against the regenerated
// TEXT-ONLY `fixtures/text_calcers/BM25/*.npy` ground truth ≤1e-5.
// ===========================================================================

/// Stage 1 — split borders (≤1e-5 vs the frozen text-only `splits.npy`,
/// O(1e-3) BM25 text-feature borders, NOT the embedding feature's ±1.24).
#[test]
fn bm25_oracle_splits_match_upstream() {
    let (model, feats, _staged) = train_bm25();
    let expected = load_f64_vec(&fixture("text_calcers/BM25/splits.npy")).expect("splits.npy");
    let (actual, _leaves) = canonical_stages(&model, &feats);
    compare_stage(Stage::Splits, &expected, &actual)
        .unwrap_or_else(|e| panic!("BM25 split borders diverged from upstream: {e:?}"));
}

/// Stage 2 — leaf values (≤1e-5 vs the frozen `leaf_values.npy`).
#[test]
fn bm25_oracle_leaf_values_match_upstream() {
    let (model, feats, _staged) = train_bm25();
    let expected =
        load_f64_vec(&fixture("text_calcers/BM25/leaf_values.npy")).expect("leaf_values.npy");
    let (_borders, actual) = canonical_stages(&model, &feats);
    compare_stage(Stage::LeafValues, &expected, &actual)
        .unwrap_or_else(|e| panic!("BM25 leaf values diverged from upstream: {e:?}"));
}

/// Stage 3 — staged approximants (≤1e-5 vs the frozen `staged.npy`).
#[test]
fn bm25_oracle_staged_approx_match_upstream() {
    let (_model, _feats, staged) = train_bm25();
    let expected = load_f64_vec(&fixture("text_calcers/BM25/staged.npy")).expect("staged.npy");
    compare_stage(Stage::StagedApprox, &expected, &staged)
        .unwrap_or_else(|e| panic!("BM25 staged approx diverged from upstream: {e:?}"));
}

/// Stage 4 — final predictions (≤1e-5 vs the frozen `predictions.npy`).
#[test]
fn bm25_oracle_predictions_match_upstream() {
    let (_model, _feats, staged) = train_bm25();
    let expected =
        load_f64_vec(&fixture("text_calcers/BM25/predictions.npy")).expect("predictions.npy");
    let n = expected.len();
    assert!(staged.len() >= n, "staged buffer covers the final iteration");
    let actual = &staged[staged.len() - n..];
    compare_stage(Stage::Predictions, &expected, actual)
        .unwrap_or_else(|e| panic!("BM25 predictions diverged from upstream: {e:?}"));
}

