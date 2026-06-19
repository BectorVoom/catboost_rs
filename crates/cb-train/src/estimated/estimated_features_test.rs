//! Unit tests for the BoW estimated-feature seam
//! ([`super::estimated_features`]).
//!
//! Validates the SC-4 append+quantize wiring on the frozen 16-row FEAT-01
//! corpus: feature ordering (BiGram block then Word block), feature count
//! (25 + 8 = 33), binary presence semantics, the single-border-at-0.5 quantizer
//! output, and the inert-when-absent (D-04) non-regression property.

use cb_data::text::tokenizer::TokenizerOptions;

use super::estimated_features::{build_bow_estimated_features, build_mixed_estimated_features};

/// The frozen FEAT-01 corpus (mirrors
/// `fixtures/text_embedding_inputs/texts.json`).
fn corpus() -> Vec<String> {
    [
        "good great movie",
        "bad awful film",
        "good film great",
        "awful movie bad",
        "great good film",
        "bad bad awful",
        "great great good",
        "awful awful bad",
        "good good great",
        "bad film awful",
        "good movie great film",
        "awful bad movie film",
        "great wonderful good",
        "bad terrible awful",
        "good great great",
        "awful bad bad",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Inert when absent (D-04): an empty text column yields no estimated columns
/// and no borders — the non-text training path is unchanged.
#[test]
fn empty_text_yields_no_estimated_features() {
    let feats =
        build_bow_estimated_features(&[], &TokenizerOptions::default(), 254).expect("empty ok");
    assert!(feats.columns.is_empty());
    assert!(feats.borders.is_empty());
    assert_eq!(feats.bigram_feature_count, 0);
    assert_eq!(feats.word_feature_count, 0);
}

/// Feature count: 25 BiGram + 8 Word = 33 columns, BiGram block first.
#[test]
fn bow_feature_count_is_bigram_then_word() {
    let feats = build_bow_estimated_features(&corpus(), &TokenizerOptions::default(), 254)
        .expect("bow features");
    assert_eq!(feats.bigram_feature_count, 25);
    assert_eq!(feats.word_feature_count, 8);
    assert_eq!(feats.columns.len(), 33);
    assert_eq!(feats.borders.len(), 33);
}

/// Every column has length n_docs and every cell is binary 0.0/1.0 (presence).
#[test]
fn bow_columns_are_binary_presence_per_doc() {
    let n = corpus().len();
    let feats = build_bow_estimated_features(&corpus(), &TokenizerOptions::default(), 254)
        .expect("bow features");
    for col in &feats.columns {
        assert_eq!(col.len(), n, "each column spans all documents");
        for &v in col {
            assert!(v == 0.0 || v == 1.0, "BoW cell is binary presence, got {v}");
        }
    }
}

/// Each non-degenerate binary column quantizes to a single border at 0.5 through
/// the EXISTING quantizer (SC-4). A column that is all-1 or all-0 is degenerate
/// and yields no border.
#[test]
fn bow_binary_columns_border_at_half() {
    let feats = build_bow_estimated_features(&corpus(), &TokenizerOptions::default(), 254)
        .expect("bow features");
    for (f, (col, border)) in feats.columns.iter().zip(feats.borders.iter()).enumerate() {
        let has_zero = col.iter().any(|&v| v == 0.0);
        let has_one = col.iter().any(|&v| v == 1.0);
        if has_zero && has_one {
            assert_eq!(border.len(), 1, "feature {f}: mixed binary column -> 1 border");
            assert!(
                (border[0] - 0.5).abs() <= 1e-9,
                "feature {f}: binary border must be 0.5, got {}",
                border[0]
            );
        } else {
            assert!(border.is_empty(), "feature {f}: degenerate column -> no border");
        }
    }
}

/// The Word block reproduces the unigram presence: the corpus's word dictionary
/// is `bad=0, great=1, awful=2, good=3, film=4, movie=5, terrible=6, wonderful=7`
/// (the frozen dict_ids). Doc0 "good great movie" -> words good(3),great(1),
/// movie(5) present. The Word block starts at column index 25.
#[test]
fn bow_word_block_presence_matches_unigram_dictionary() {
    let feats = build_bow_estimated_features(&corpus(), &TokenizerOptions::default(), 254)
        .expect("bow features");
    let word_base = feats.bigram_feature_count; // 25
    // doc 0 = "good great movie": great(1), good(3), movie(5) present; others 0.
    let doc = 0usize;
    let present_word_ids = [1usize, 3, 5];
    for word_id in 0..feats.word_feature_count {
        let col = &feats.columns[word_base + word_id];
        let expected = if present_word_ids.contains(&word_id) { 1.0 } else { 0.0 };
        assert_eq!(
            col[doc], expected,
            "word feature {word_id} presence for doc 0 mismatch"
        );
    }
}

// ---------------------------------------------------------------------------
// SC-4 mixed text + embedding + numeric orchestration (06.5-07).
// ---------------------------------------------------------------------------

/// The frozen FEAT-01/02 binary labels (object order, mirrors `labels.npy`).
fn labels() -> Vec<f32> {
    // Interleaved class 1 / class 0 (pos/neg) over the 16-row corpus.
    (0..16).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect()
}

/// A clean class-separating numeric column (mirrors `numeric.npy`): +1 / -1.
fn numeric_col() -> Vec<f32> {
    labels().iter().map(|&y| if y > 0.5 { 1.0 } else { -1.0 }).collect()
}

/// Two well-separated embedding clouds, one per class (signal for KNN votes).
fn embeddings() -> Vec<Vec<f32>> {
    labels()
        .iter()
        .map(|&y| {
            if y > 0.5 {
                vec![1.0, 1.0, -1.0, -1.0]
            } else {
                vec![-1.0, -1.0, 1.0, 1.0]
            }
        })
        .collect()
}

/// Mixed inert when absent (D-04): no numeric, no text, no embeddings -> empty.
#[test]
fn mixed_empty_yields_no_features() {
    let feats = build_mixed_estimated_features(
        &[],
        &[],
        &[],
        &[],
        2,
        3,
        false,
        &TokenizerOptions::default(),
        254,
    )
    .expect("empty mixed ok");
    assert!(feats.columns.is_empty());
    assert!(feats.borders.is_empty());
    assert_eq!(feats.numeric_feature_count, 0);
    assert_eq!(feats.text_feature_count, 0);
    assert_eq!(feats.embedding_feature_count, 0);
}

/// Numeric-only pool: the estimated path is inert; only the numeric block exists,
/// joined directly + quantized (D-04 — the existing numeric path is unchanged).
#[test]
fn mixed_numeric_only_is_inert_estimated_path() {
    let num = numeric_col();
    let feats = build_mixed_estimated_features(
        &[num.clone()],
        &[],
        &[],
        &labels(),
        2,
        3,
        false,
        &TokenizerOptions::default(),
        254,
    )
    .expect("numeric-only mixed ok");
    assert_eq!(feats.numeric_feature_count, 1);
    assert_eq!(feats.text_feature_count, 0);
    assert_eq!(feats.embedding_feature_count, 0);
    assert_eq!(feats.columns.len(), 1);
    // The numeric column is passed through verbatim.
    assert_eq!(feats.columns[0], num);
    // A clean +1/-1 column quantizes to a single border at 0.0.
    assert_eq!(feats.borders[0].len(), 1);
    assert!((feats.borders[0][0] - 0.0).abs() <= 1e-9);
}

/// Mixed block layout: numeric block first, then BoW text block, then KNN
/// embedding block — counts and total width add up, all columns span n_docs.
#[test]
fn mixed_block_layout_order_and_counts() {
    let num = numeric_col();
    let feats = build_mixed_estimated_features(
        &[num],
        &corpus(),
        &embeddings(),
        &labels(),
        2,
        3,
        false,
        &TokenizerOptions::default(),
        254,
    )
    .expect("mixed ok");
    // 1 numeric + (25 BiGram + 8 Word = 33) BoW + 2 KNN class-vote columns.
    assert_eq!(feats.numeric_feature_count, 1);
    assert_eq!(feats.text_feature_count, 33);
    assert_eq!(feats.embedding_feature_count, 2);
    let total = feats.numeric_feature_count + feats.text_feature_count + feats.embedding_feature_count;
    assert_eq!(feats.columns.len(), total);
    assert_eq!(feats.borders.len(), total);

    let n = corpus().len();
    for col in &feats.columns {
        assert_eq!(col.len(), n, "every mixed column spans all documents");
    }

    // Block 0 is the numeric column (verbatim, ±1).
    assert_eq!(feats.columns[0], numeric_col());

    // The KNN block (last 2 columns) holds integer per-class vote counts summing
    // to k (=3) per document.
    let knn_base = feats.numeric_feature_count + feats.text_feature_count;
    for doc in 0..n {
        let v0 = feats.columns[knn_base][doc];
        let v1 = feats.columns[knn_base + 1][doc];
        assert!((v0 + v1 - 3.0).abs() < 1e-6, "doc{doc} KNN votes sum to k");
        assert_eq!(v0.fract(), 0.0, "doc{doc} KNN class0 vote is an integer count");
    }
}

/// Length-mismatch robustness: a numeric column shorter than the text column is
/// rejected with a typed error (no panic — V5/INFRA-02).
#[test]
fn mixed_rejects_length_mismatch() {
    let short_num = vec![1.0_f32; 3];
    let res = build_mixed_estimated_features(
        &[short_num],
        &corpus(),
        &embeddings(),
        &labels(),
        2,
        3,
        false,
        &TokenizerOptions::default(),
        254,
    );
    assert!(res.is_err(), "mismatched numeric column must be rejected");
}

// ---------------------------------------------------------------------------
// FEAT-01 residual: the KNN estimated-feature stored-border-VALUE root cause.
//
// The stored-border divergence (upstream 0.5 vs Rust-offline 1.5) is a column-
// VALUE divergence, NOT a border-algorithm divergence: `select_borders_greedy_
// logsum` is UNCHANGED. Upstream's KNN is an `IOnlineFeatureEstimator`, so the
// border-computing visitor is fed the ONLINE read-before-update prefix estimate
// (`estimated_features.cpp:472-478 ComputeOnlineFeatures(*learnPermutation,...)`),
// whose vote-count column has distinct values {0,1,…,k} -> first greedy-logsum
// border 0.5. The OFFLINE whole-set estimate inserts every doc first (each doc is
// its own neighbor), so a perfectly-separated k=3 cloud yields distinct values
// {0,k} -> border k/2 (1.5). These tests pin BOTH distributions through the SAME
// unchanged quantizer.
// ---------------------------------------------------------------------------

/// Distinct values of a column, sorted ascending (helper).
fn distinct_sorted(col: &[f32]) -> Vec<f32> {
    let mut v: Vec<f32> = col.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    v.dedup_by(|a, b| (*a - *b).abs() <= 1e-9);
    v
}

/// OFFLINE KNN block: a perfectly-separated k=3 cloud gives each document 3
/// same-class neighbors (the doc is its own neighbor at distance 0), so the
/// per-class vote column's distinct values are {0, 3} and the UNCHANGED quantizer
/// emits the single border 1.5 (= 3/2). This is the column-VALUE distribution that
/// does NOT match upstream's stored border.
#[test]
fn offline_knn_block_distinct_is_zero_k_border_is_half_k() {
    let feats = build_mixed_estimated_features(
        &[],
        &[],
        &embeddings(),
        &labels(),
        2,
        3,
        false, // OFFLINE whole-set estimate
        &TokenizerOptions::default(),
        254,
    )
    .expect("offline mixed ok");
    assert_eq!(feats.embedding_feature_count, 2, "two class-vote columns");
    let knn_base = feats.numeric_feature_count + feats.text_feature_count;
    for f in knn_base..feats.columns.len() {
        let dv = distinct_sorted(&feats.columns[f]);
        assert_eq!(dv, vec![0.0_f32, 3.0_f32], "offline KNN col {f} distinct = {{0, k}}");
        // The UNCHANGED greedy-logsum quantizer over {0,3} -> single border 1.5.
        assert_eq!(feats.borders[f].len(), 1, "offline KNN col {f}: one border");
        assert!(
            (feats.borders[f][0] - 1.5).abs() <= 1e-9,
            "offline KNN col {f} border must be 1.5 (= k/2), got {}",
            feats.borders[f][0]
        );
    }
}

/// ONLINE KNN block (upstream border source): the read-before-update prefix
/// estimate over the identity learn permutation gives early-prefix documents
/// `< k` (or mixed-class) neighbors, so the per-class vote column's distinct
/// values are {0, 1, 2, 3} and the UNCHANGED greedy-logsum quantizer's FIRST
/// border is 0.5 — EXACTLY upstream's stored KNN border. The fix moves the stored
/// border VALUE from 1.5 to 0.5 by changing the COLUMN VALUES only; the binarizer
/// is byte-identical to the offline path.
#[test]
fn online_knn_block_distinct_is_zero_to_k_first_border_is_half() {
    let feats = build_mixed_estimated_features(
        &[],
        &[],
        &embeddings(),
        &labels(),
        2,
        3,
        true, // ONLINE read-before-update estimate (upstream border source)
        &TokenizerOptions::default(),
        254,
    )
    .expect("online mixed ok");
    assert_eq!(feats.embedding_feature_count, 2, "two class-vote columns");
    let knn_base = feats.numeric_feature_count + feats.text_feature_count;
    for f in knn_base..feats.columns.len() {
        let dv = distinct_sorted(&feats.columns[f]);
        // The separated cloud's online distinct set is {0,1,2,3} (permutation-
        // invariant in its SET); at minimum it contains 0 and 1 so the lowest
        // border is 0.5.
        assert!(
            dv.contains(&0.0) && dv.contains(&1.0),
            "online KNN col {f} distinct {dv:?} contains 0 and 1 (-> border 0.5)"
        );
        assert!(
            !feats.borders[f].is_empty(),
            "online KNN col {f}: a non-degenerate column gets a border"
        );
        // The FIRST greedy-logsum border is the {0,1} midpoint = 0.5, which is
        // upstream's stored KNN border (estimated_features.cpp ComputeOnlineFeatures).
        assert!(
            (feats.borders[f][0] - 0.5).abs() <= 1e-9,
            "online KNN col {f} first border must be 0.5 (upstream stored), got {}",
            feats.borders[f][0]
        );
    }
}

/// The binarizer is byte-identical across the two modes — ONLY the column VALUES
/// differ (the column-VALUE root cause). The BoW text block (identical in both
/// modes) borders at 0.5 regardless, confirming the quantizer itself is unchanged
/// and the KNN border move is purely a column-value effect.
#[test]
fn online_vs_offline_differ_only_in_embedding_column_values() {
    let off = build_mixed_estimated_features(
        &[numeric_col()], &corpus(), &embeddings(), &labels(), 2, 3, false,
        &TokenizerOptions::default(), 254,
    )
    .expect("offline mixed ok");
    let on = build_mixed_estimated_features(
        &[numeric_col()], &corpus(), &embeddings(), &labels(), 2, 3, true,
        &TokenizerOptions::default(), 254,
    )
    .expect("online mixed ok");
    // Numeric + text blocks (and their borders) are byte-identical across modes.
    assert_eq!(off.numeric_feature_count, on.numeric_feature_count);
    assert_eq!(off.text_feature_count, on.text_feature_count);
    let shared = off.numeric_feature_count + off.text_feature_count;
    for f in 0..shared {
        assert_eq!(off.columns[f], on.columns[f], "shared col {f} identical");
        assert_eq!(off.borders[f], on.borders[f], "shared border {f} identical");
    }
    // Only the embedding block's COLUMN VALUES move (offline {0,k} vs online {0..k}).
    let knn_base = shared;
    let mut any_diff = false;
    for f in knn_base..off.columns.len() {
        if off.columns[f] != on.columns[f] {
            any_diff = true;
        }
    }
    assert!(any_diff, "the KNN block column values differ between online and offline");
}
