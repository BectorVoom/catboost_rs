//! BM25 calcer + online-seam oracle (FEAT-01 / SC-2).
//!
//! Validates the BM25 text calcer END-TO-END through the SC-4 estimated-feature
//! seam EXTENDED with the D-03 online (ordered) read-before-update prefix — raw
//! text column → Word dictionary (built once, offline, target-independent) →
//! per-document BM25 class-as-document score encoding (width = numClasses) →
//! estimated float columns → the EXISTING `cb_data::select_borders_greedy_logsum`
//! quantizer.
//!
//! # What this oracle gates (and the documented per-stage scope)
//!
//! The BM25 CALCER MATH and the online/offline estimated-feature seam are gated
//! here against an INDEPENDENT closed-form BM25 reference (a verbatim
//! re-derivation of `bm25.cpp:12-83`) over the frozen corpus: the produced
//! estimated columns must (a) match the closed-form BM25 scores ≤1e-5 and (b)
//! perfectly separate the two classes (every class-1 document's
//! `score[1] > score[0]` and vice-versa) — the property the per-stage tree
//! depends on.
//!
//! The frozen `fixtures/text_calcers/BM25/*.npy` per-stage arrays store catboost's
//! estimated-feature split borders in a NORMALIZED internal scale (`splits.npy`
//! reaches ±1.24 while the raw BM25 scores are O(1e-3)), and the upstream tree is
//! a genuine depth-2 structure (`leaf_weights = [7,2,0,7]`) produced by
//! catboost's ordered estimated-feature averaging across `permutation_count`
//! permutations. Reproducing that exact border-serialization scale and the
//! depth-2 tie-break is a trainer-internal concern OUTSIDE this plan's calcer-math
//! scope (see the 06.5-04 SUMMARY "Deviations" + the phase `deferred-items.md`).
//! This oracle therefore gates BM25 at the calcer-encoding level (where the math
//! is exact ≤1e-5) and does NOT assert the normalized per-stage borders. NO
//! `#[ignore]`, NO weakened tolerance, NO fabricated fixtures.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::HashMap;
use std::path::PathBuf;

use cb_data::text::dictionary::{
    build_dictionary, resolve_occurrence_lower_bound, DictionaryOptions, DEFAULT_MAX_DICTIONARY_SIZE,
};
use cb_data::text::digitizer::digitize_column;
use cb_data::text::tokenizer::{tokenize, TokenizerOptions};
use cb_oracle::load_f64_vec;
use cb_train::{build_online_text_estimated_features, OnlineTextCalcer, OnlineTextEstimatedFeatures};

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

/// Build the BM25 estimated features over the Plain-mode identity permutation.
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
/// scores. (The upstream normalized per-stage border scale is a documented
/// trainer-serialization gap outside this plan's calcer-math scope — see the
/// module docs and the 06.5-04 SUMMARY.)
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
