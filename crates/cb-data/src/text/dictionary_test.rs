//! Unit tests for the frequency dictionary build + `Apply` (`dictionary.rs`).
//!
//! Covers count / strict-filter / (count DESC, token ASC) sort / truncate /
//! id-assign / Skip-unknown (`dictionary_builder.cpp:149-199`,
//! `frequency_based_dictionary_impl.cpp:13-25`). Source/test separation
//! (INFRA-06).

use super::dictionary::{
    build_dictionary, resolve_occurrence_lower_bound, DictionaryEntry, DictionaryOptions,
    UnknownTokenPolicy, DEFAULT_MAX_DICTIONARY_SIZE,
};

fn docs(rows: &[&[&str]]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect()
}

/// The 16-row FEAT-01 corpus (fixtures/text_embedding_inputs/texts.json),
/// already lowercase + space-split.
fn fixture_corpus() -> Vec<Vec<String>> {
    docs(&[
        &["good", "great", "movie"],
        &["bad", "awful", "film"],
        &["good", "film", "great"],
        &["awful", "movie", "bad"],
        &["great", "good", "film"],
        &["bad", "bad", "awful"],
        &["great", "great", "good"],
        &["awful", "awful", "bad"],
        &["good", "good", "great"],
        &["bad", "film", "awful"],
        &["good", "movie", "great", "film"],
        &["awful", "bad", "movie", "film"],
        &["great", "wonderful", "good"],
        &["bad", "terrible", "awful"],
        &["good", "great", "great"],
        &["awful", "bad", "bad"],
    ])
}

#[test]
fn resolve_olb_is_data_dependent() {
    // options_helper.cpp:394-401: poolSize < 1000 -> 1, else 5. Boundary strict.
    assert_eq!(resolve_occurrence_lower_bound(16), 1);
    assert_eq!(resolve_occurrence_lower_bound(999), 1);
    assert_eq!(resolve_occurrence_lower_bound(1000), 5);
    assert_eq!(resolve_occurrence_lower_bound(5000), 5);
}

#[test]
fn default_options_match_upstream() {
    let o = DictionaryOptions::default();
    assert_eq!(o.occurrence_lower_bound, 1);
    assert_eq!(o.max_dictionary_size, Some(DEFAULT_MAX_DICTIONARY_SIZE));
    assert_eq!(o.start_token_id, 0);
    assert_eq!(DEFAULT_MAX_DICTIONARY_SIZE, 50_000);
}

#[test]
fn build_matches_fixture_dict_ids() {
    // Expected from fixtures/text_tokenizer/dict_ids.json (Word unigram):
    // bad=10/0, great=10/1, awful=9/2, good=9/3, film=6/4, movie=4/5,
    // terrible=1/6, wonderful=1/7. Tie (bad,great both 10) broken by token ASC
    // (bad < great); (awful,good both 9) -> awful < good.
    let olb = resolve_occurrence_lower_bound(16);
    let opts = DictionaryOptions {
        occurrence_lower_bound: olb,
        ..DictionaryOptions::default()
    };
    let (_dict, entries) = build_dictionary(&fixture_corpus(), &opts);

    let expected = vec![
        DictionaryEntry { token: "bad".into(), id: 0, count: 10 },
        DictionaryEntry { token: "great".into(), id: 1, count: 10 },
        DictionaryEntry { token: "awful".into(), id: 2, count: 9 },
        DictionaryEntry { token: "good".into(), id: 3, count: 9 },
        DictionaryEntry { token: "film".into(), id: 4, count: 6 },
        DictionaryEntry { token: "movie".into(), id: 5, count: 4 },
        DictionaryEntry { token: "terrible".into(), id: 6, count: 1 },
        DictionaryEntry { token: "wonderful".into(), id: 7, count: 1 },
    ];
    assert_eq!(entries, expected);
}

#[test]
fn strict_filter_drops_below_lower_bound() {
    // OLB=2: tokens with count < 2 are dropped (strict <). "rare" appears once.
    let corpus = docs(&[&["a", "a", "rare"], &["a", "b"], &["b"]]);
    let opts = DictionaryOptions {
        occurrence_lower_bound: 2,
        ..DictionaryOptions::default()
    };
    let (dict, entries) = build_dictionary(&corpus, &opts);
    // a=3, b=2 survive; rare=1 dropped.
    let tokens: Vec<&str> = entries.iter().map(|e| e.token.as_str()).collect();
    assert_eq!(tokens, vec!["a", "b"]);
    assert!(dict.token_id("rare").is_none());
}

#[test]
fn tie_break_is_token_ascending() {
    // All count 1: ids assigned in token-ASC order (count tie).
    let corpus = docs(&[&["zebra", "apple", "mango"]]);
    let opts = DictionaryOptions {
        occurrence_lower_bound: 1,
        ..DictionaryOptions::default()
    };
    let (_d, entries) = build_dictionary(&corpus, &opts);
    let tokens: Vec<&str> = entries.iter().map(|e| e.token.as_str()).collect();
    assert_eq!(tokens, vec!["apple", "mango", "zebra"]);
    assert_eq!(entries.iter().map(|e| e.id).collect::<Vec<_>>(), vec![0, 1, 2]);
}

#[test]
fn max_dictionary_size_truncates_after_sort() {
    let corpus = docs(&[&["a", "a", "a", "b", "b", "c"]]);
    let opts = DictionaryOptions {
        occurrence_lower_bound: 1,
        max_dictionary_size: Some(2),
        start_token_id: 0,
    };
    let (_d, entries) = build_dictionary(&corpus, &opts);
    // a=3, b=2 kept (top-2 by count DESC); c=1 truncated.
    let tokens: Vec<&str> = entries.iter().map(|e| e.token.as_str()).collect();
    assert_eq!(tokens, vec!["a", "b"]);
}

#[test]
fn start_token_id_offsets_ids() {
    let corpus = docs(&[&["a", "b"]]);
    let opts = DictionaryOptions {
        occurrence_lower_bound: 1,
        max_dictionary_size: Some(DEFAULT_MAX_DICTIONARY_SIZE),
        start_token_id: 100,
    };
    let (_d, entries) = build_dictionary(&corpus, &opts);
    assert_eq!(entries.iter().map(|e| e.id).collect::<Vec<_>>(), vec![100, 101]);
}

#[test]
fn apply_skip_drops_unknown_tokens() {
    let corpus = docs(&[&["a", "b"]]);
    let opts = DictionaryOptions {
        occurrence_lower_bound: 1,
        ..DictionaryOptions::default()
    };
    let (dict, _e) = build_dictionary(&corpus, &opts);
    let tokens = vec!["a".to_string(), "zzz".to_string(), "b".to_string()];
    let ids = dict.apply(&tokens, UnknownTokenPolicy::Skip);
    // zzz unknown -> dropped; a->0, b->1.
    assert_eq!(ids, vec![dict.token_id("a").unwrap(), dict.token_id("b").unwrap()]);
    assert_eq!(ids.len(), 2);
}

#[test]
fn apply_insert_emits_unknown_id() {
    let corpus = docs(&[&["a", "b"]]);
    let opts = DictionaryOptions {
        occurrence_lower_bound: 1,
        ..DictionaryOptions::default()
    };
    let (dict, _e) = build_dictionary(&corpus, &opts);
    let unk = dict.unknown_token_id();
    assert_eq!(unk, 2); // size(2) + start(0)
    let tokens = vec!["zzz".to_string()];
    let ids = dict.apply(&tokens, UnknownTokenPolicy::Insert);
    assert_eq!(ids, vec![unk]);
}

#[test]
fn empty_corpus_yields_empty_dictionary() {
    let opts = DictionaryOptions::default();
    let (dict, entries) = build_dictionary(&[], &opts);
    assert!(dict.is_empty());
    assert!(entries.is_empty());
}
