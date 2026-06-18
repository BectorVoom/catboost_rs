//! Unit tests for the BiGram dictionary ([`super::bigram_dictionary`]).
//!
//! Pins the consecutive-pair counting, the `(count DESC, ngram ASC)` tie-break,
//! the `StartTokenId++` id assignment, and the `Apply(Skip)` path against the
//! frozen 16-row FEAT-01 corpus — the SAME corpus whose BoW `ttext` dump
//! (`fixtures/text_tokenizer/ttext.json`) the BiGram ids reproduce bit-exact.

use super::bigram_dictionary::{build_bigram_dictionary, BigramDictionary};
use super::dictionary::{DictionaryOptions, UnknownTokenPolicy};

/// The frozen FEAT-01 corpus, lowercased + space-split (the D-02 ByDelimiter
/// tokenization). Mirrors `fixtures/text_embedding_inputs/texts.json`.
fn corpus() -> Vec<Vec<String>> {
    let texts = [
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
    ];
    texts
        .iter()
        .map(|t| t.split(' ').map(str::to_string).collect())
        .collect()
}

fn opts() -> DictionaryOptions {
    DictionaryOptions {
        occurrence_lower_bound: 1,
        max_dictionary_size: Some(50_000),
        start_token_id: 0,
    }
}

/// The corpus produces a 25-bigram dictionary (ids 0..=24), matching the BoW
/// `ttext` ids that run up to 24.
#[test]
fn bigram_dictionary_has_25_entries() {
    let (dict, entries) = build_bigram_dictionary(&corpus(), &opts());
    assert_eq!(dict.len(), 25);
    assert_eq!(entries.len(), 25);
}

/// The `(count DESC, ngram ASC)` tie-break: the two count-3 bigrams come first,
/// `(awful,bad)` before `(good,great)` (count tie -> ngram-string ASC: "awful" <
/// "good"). Then the count-2 group sorted ASC.
#[test]
fn bigram_dictionary_orders_by_count_then_ngram() {
    let (_dict, entries) = build_bigram_dictionary(&corpus(), &opts());
    // id 0,1: the two count-3 bigrams, ngram ASC.
    assert_eq!((entries[0].first.as_str(), entries[0].second.as_str()), ("awful", "bad"));
    assert_eq!(entries[0].count, 3);
    assert_eq!((entries[1].first.as_str(), entries[1].second.as_str()), ("good", "great"));
    assert_eq!(entries[1].count, 3);
    // id 2: first of the count-2 group, ngram ASC -> (bad, awful).
    assert_eq!((entries[2].first.as_str(), entries[2].second.as_str()), ("bad", "awful"));
    assert_eq!(entries[2].count, 2);
}

/// Apply reproduces the frozen BoW BiGram `ttext` digitization: doc0
/// "good great movie" -> bigrams (good,great)=id1, (great,movie)=id18 -> sorted
/// ids [1,18] (the fixture's first BoW ttext event `[[1,1],[18,1]]`).
#[test]
fn bigram_apply_matches_frozen_ttext_doc0() {
    let (dict, _entries) = build_bigram_dictionary(&corpus(), &opts());
    let doc0: Vec<String> = "good great movie".split(' ').map(str::to_string).collect();
    let mut ids = dict.apply(&doc0, UnknownTokenPolicy::Skip);
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 18]);
}

/// Apply on doc10 "good movie great film" -> (good,movie)=16, (movie,great)=22,
/// (great,film)=17 -> sorted [16,17,22] (fixture BoW ttext `[[16,1],[17,1],[22,1]]`).
#[test]
fn bigram_apply_matches_frozen_ttext_doc10() {
    let (dict, _entries) = build_bigram_dictionary(&corpus(), &opts());
    let doc10: Vec<String> = "good movie great film".split(' ').map(str::to_string).collect();
    let mut ids = dict.apply(&doc10, UnknownTokenPolicy::Skip);
    ids.sort_unstable();
    assert_eq!(ids, vec![16, 17, 22]);
}

/// A document shorter than 2 tokens contributes no bigram and digitizes to an
/// empty id list (no panic, V5).
#[test]
fn bigram_short_document_contributes_nothing() {
    let (dict, _entries) = build_bigram_dictionary(&corpus(), &opts());
    let one: Vec<String> = vec!["good".to_string()];
    assert!(dict.apply(&one, UnknownTokenPolicy::Skip).is_empty());
    assert!(dict.apply(&[], UnknownTokenPolicy::Skip).is_empty());
}

/// An unknown bigram (a pair never seen in the learn corpus) is dropped under
/// Skip — here a reversed pair the corpus never contains.
#[test]
fn bigram_apply_skips_unknown_pair() {
    let (dict, _entries) = build_bigram_dictionary(&corpus(), &opts());
    // "movie movie" is not a learn bigram -> dropped.
    let doc: Vec<String> = "movie movie".split(' ').map(str::to_string).collect();
    assert!(dict.apply(&doc, UnknownTokenPolicy::Skip).is_empty());
}

/// Empty corpus -> empty dictionary (no panic).
#[test]
fn bigram_empty_corpus_is_empty() {
    let (dict, entries) = build_bigram_dictionary(&[], &opts());
    assert!(BigramDictionary::is_empty(&dict));
    assert!(entries.is_empty());
}
