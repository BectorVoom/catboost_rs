//! Unit tests for the digitizer (`digitizer.rs`).
//!
//! Covers tokenize → dictionary-apply(Skip) → sorted-RLE `TText`
//! (`text_column_builder.cpp:6-11`). Source/test separation (INFRA-06).

use super::dictionary::{build_dictionary, resolve_occurrence_lower_bound, DictionaryOptions};
use super::digitizer::{digitize_column, digitize_document};
use super::tokenizer::TokenizerOptions;

fn fixture_corpus_raw() -> Vec<String> {
    vec![
        "good great movie".into(),
        "bad awful film".into(),
        "good film great".into(),
        "awful movie bad".into(),
        "great good film".into(),
        "bad bad awful".into(),
        "great great good".into(),
        "awful awful bad".into(),
        "good good great".into(),
        "bad film awful".into(),
        "good movie great film".into(),
        "awful bad movie film".into(),
        "great wonderful good".into(),
        "bad terrible awful".into(),
        "good great great".into(),
        "awful bad bad".into(),
    ]
}

fn build_word_dict() -> super::dictionary::Dictionary {
    let tokenizer = TokenizerOptions::default();
    let raw = fixture_corpus_raw();
    let tokenized: Vec<Vec<String>> = raw
        .iter()
        .map(|t| super::tokenizer::tokenize(t, &tokenizer))
        .collect();
    let olb = resolve_occurrence_lower_bound(raw.len());
    let opts = DictionaryOptions {
        occurrence_lower_bound: olb,
        ..DictionaryOptions::default()
    };
    let (dict, _entries) = build_dictionary(&tokenized, &opts);
    dict
}

fn ttext_pairs(t: &super::text::TText) -> Vec<(u32, u32)> {
    t.pairs().iter().map(|p| (p.token, p.count)).collect()
}

#[test]
fn digitize_document_matches_fixture_ttext_word_dict() {
    // Word-dict ids: bad=0 great=1 awful=2 good=3 film=4 movie=5 terrible=6
    // wonderful=7. From fixtures/text_tokenizer/ttext.json (NaiveBayes path):
    //   "good great movie" -> ids [3,1,5] -> TText [(1,1),(3,1),(5,1)]
    //   "bad bad awful"    -> ids [0,0,2] -> TText [(0,2),(2,1)]
    let dict = build_word_dict();
    let tok = TokenizerOptions::default();

    let t0 = digitize_document("good great movie", &tok, &dict);
    assert_eq!(ttext_pairs(&t0), vec![(1, 1), (3, 1), (5, 1)]);

    let t5 = digitize_document("bad bad awful", &tok, &dict);
    assert_eq!(ttext_pairs(&t5), vec![(0, 2), (2, 1)]);
}

#[test]
fn digitize_drops_unknown_tokens() {
    let dict = build_word_dict();
    let tok = TokenizerOptions::default();
    // "zzz" is not in the dictionary -> dropped (Skip); only "good"(3) survives.
    let t = digitize_document("zzz good zzz", &tok, &dict);
    assert_eq!(ttext_pairs(&t), vec![(3, 1)]);
}

#[test]
fn digitize_empty_document_yields_empty_ttext() {
    let dict = build_word_dict();
    let tok = TokenizerOptions::default();
    let t = digitize_document("", &tok, &dict);
    assert!(t.is_empty());
}

#[test]
fn digitize_column_processes_each_document() {
    let dict = build_word_dict();
    let tok = TokenizerOptions::default();
    let raw = fixture_corpus_raw();
    let column = digitize_column(&raw, &tok, &dict);
    assert_eq!(column.len(), raw.len());
    // First row "good great movie" -> [(1,1),(3,1),(5,1)].
    assert_eq!(ttext_pairs(&column[0]), vec![(1, 1), (3, 1), (5, 1)]);
}

#[test]
fn digitize_column_empty_input_yields_empty_column() {
    let dict = build_word_dict();
    let tok = TokenizerOptions::default();
    let column = digitize_column(&[], &tok, &dict);
    assert!(column.is_empty());
}
