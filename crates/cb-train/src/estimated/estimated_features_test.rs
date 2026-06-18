//! Unit tests for the BoW estimated-feature seam
//! ([`super::estimated_features`]).
//!
//! Validates the SC-4 append+quantize wiring on the frozen 16-row FEAT-01
//! corpus: feature ordering (BiGram block then Word block), feature count
//! (25 + 8 = 33), binary presence semantics, the single-border-at-0.5 quantizer
//! output, and the inert-when-absent (D-04) non-regression property.

use cb_data::text::tokenizer::TokenizerOptions;

use super::estimated_features::build_bow_estimated_features;

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
