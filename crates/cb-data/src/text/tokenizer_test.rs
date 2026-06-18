//! Unit tests for the `ByDelimiter` tokenizer (`tokenizer.rs`).
//!
//! Covers the upstream split + lowercasing + `SkipEmpty` behavior
//! (`tokenizer.cpp:188-254`). Source/test separation (INFRA-06): no inline
//! `#[cfg(test)]` in the production module.

use super::tokenizer::{tokenize, tokenize_default, TokenizerOptions, DEFAULT_DELIMITER};

#[test]
fn default_delimiter_is_space() {
    assert_eq!(DEFAULT_DELIMITER, " ");
    let opts = TokenizerOptions::default();
    assert_eq!(opts.delimiter, " ");
    assert!(opts.lowercasing);
    assert!(opts.skip_empty);
}

#[test]
fn splits_on_space_and_lowercases() {
    // "Hello World" -> ["hello", "world"] (behavior contract).
    assert_eq!(tokenize_default("Hello World"), vec!["hello", "world"]);
}

#[test]
fn fixture_corpus_documents_tokenize_as_expected() {
    // From fixtures/text_embedding_inputs/texts.json — the raw corpus is already
    // lowercase, single-space delimited; the tokenizer must reproduce the
    // post-split token list exactly.
    assert_eq!(
        tokenize_default("good great movie"),
        vec!["good", "great", "movie"]
    );
    assert_eq!(
        tokenize_default("good movie great film"),
        vec!["good", "movie", "great", "film"]
    );
}

#[test]
fn skip_empty_drops_tokens_between_consecutive_delimiters() {
    // Leading, trailing, and doubled delimiters all produce empty tokens that
    // SkipEmpty must drop (tokenizer.cpp:203-204).
    assert_eq!(
        tokenize_default("  hello   world  "),
        vec!["hello", "world"]
    );
}

#[test]
fn skip_empty_false_keeps_empty_tokens() {
    let opts = TokenizerOptions {
        delimiter: " ".to_string(),
        lowercasing: true,
        skip_empty: false,
    };
    // "a  b" split on " " with SkipEmpty=false -> ["a", "", "b"].
    assert_eq!(tokenize("a  b", &opts), vec!["a", "", "b"]);
}

#[test]
fn lowercasing_disabled_preserves_case() {
    let opts = TokenizerOptions {
        delimiter: " ".to_string(),
        lowercasing: false,
        skip_empty: true,
    };
    assert_eq!(tokenize("Hello World", &opts), vec!["Hello", "World"]);
}

#[test]
fn empty_input_yields_empty_token_vec() {
    assert!(tokenize_default("").is_empty());
    // A string of only delimiters with SkipEmpty -> no tokens.
    assert!(tokenize_default("     ").is_empty());
}

#[test]
fn split_by_string_uses_whole_delimiter_not_char_set() {
    // SplitBySet=false: the delimiter is the whole string "::", not the set
    // {':'}. "a::b:c" -> ["a", "b:c"].
    let opts = TokenizerOptions {
        delimiter: "::".to_string(),
        lowercasing: false,
        skip_empty: true,
    };
    assert_eq!(tokenize("a::b:c", &opts), vec!["a", "b:c"]);
}

#[test]
fn empty_delimiter_returns_whole_input_as_single_token() {
    // Panic-free total-function edge: empty delimiter -> no split.
    let opts = TokenizerOptions {
        delimiter: String::new(),
        lowercasing: false,
        skip_empty: true,
    };
    assert_eq!(tokenize("abc", &opts), vec!["abc"]);
    assert!(tokenize("", &opts).is_empty());
}
