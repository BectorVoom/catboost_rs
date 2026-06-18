//! SC-1 / D-01 LOAD-BEARING GATE — the tokenizer + dictionary + digitizer
//! token stream reproduced BIT-EXACT vs the D-07 instrumented catboost 1.2.10
//! dump (`crates/cb-oracle/fixtures/text_tokenizer/`).
//!
//! This is the first gate of Phase 6.5 (CONTEXT D-01): every text/embedding
//! calcer plan (06.5-03..07) is blocked on it. The fixtures are the instrumented
//! ground-truth JSON dumps frozen by Plan 01 (the trainer's `CB_INSTRUMENT_LOG`
//! sink over the 16-row corpus), NOT a `model.json` (upstream forbids JSON
//! export for text/embedding models — 06.5-01 deviation). We oracle the Rust
//! tokenizer/dictionary/digitizer against those JSON dumps, integer-exact.
//!
//! Three assertions (RESEARCH SC-1 observable signals):
//!
//! 1. token_stream — per-document post-split, post-lowercase token list.
//! 2. dict_ids — the Word unigram (token-string -> id, count) table, including
//!    the (count DESC, token ASC) deterministic sort.
//! 3. ttext — per-document `(tokenId, count)` sorted-RLE for the NaiveBayes path
//!    (the Word unigram dictionary). The BoW `ttext` uses the BiGram dictionary
//!    (token ids up to 24), which is a follow-on slice (RESEARCH Pitfall 4 —
//!    unigram first; bigram deferred); the unigram gate is asserted here against
//!    the NaiveBayes ttext events.
//!
//! Escalate-don't-weaken (D-07): NO `#[ignore]`, NO weakened tolerance — these
//! are exact integer comparisons.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::text::dictionary::{
    build_dictionary, resolve_occurrence_lower_bound, DictionaryOptions,
};
use cb_data::text::digitizer::digitize_document;
use cb_data::text::tokenizer::{tokenize, TokenizerOptions};
use serde_json::Value;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("text_tokenizer")
        .join(rel)
}

fn inputs(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("text_embedding_inputs")
        .join(rel)
}

fn load_json(path: &PathBuf) -> Value {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("fixture must exist: {} ({e})", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("fixture must be valid JSON: {} ({e})", path.display()))
}

/// The raw (pre-tokenization) corpus the trainer fitted on.
fn raw_corpus() -> Vec<String> {
    let v = load_json(&inputs("texts.json"));
    v.as_array()
        .expect("texts.json is a JSON array")
        .iter()
        .map(|s| s.as_str().expect("each text is a string").to_string())
        .collect()
}

/// All token_stream events for a given calcer, in dump order, as token lists.
fn token_stream_for(calcer: &str) -> Vec<Vec<String>> {
    let v = load_json(&fixture("token_stream.json"));
    v.as_array()
        .expect("token_stream.json is an array")
        .iter()
        .filter(|e| e["_calcer"].as_str() == Some(calcer))
        .map(|e| {
            e["tokens"]
                .as_array()
                .expect("tokens is an array")
                .iter()
                .map(|t| t.as_str().expect("token is a string").to_string())
                .collect()
        })
        .collect()
}

/// The Word unigram dict_ids entries for a calcer as (token, id, count) rows in
/// dump (assigned-id) order.
fn dict_ids_for(calcer: &str) -> Vec<(String, u32, u64)> {
    let v = load_json(&fixture("dict_ids.json"));
    let event = v
        .as_array()
        .expect("dict_ids.json is an array")
        .iter()
        .find(|e| e["_calcer"].as_str() == Some(calcer) && e["gram_order"].as_i64() == Some(1))
        .expect("a gram_order=1 (Word) dict_ids event for the calcer");
    event["entries"]
        .as_array()
        .expect("entries is an array")
        .iter()
        .map(|en| {
            (
                en["token"].as_str().expect("token string").to_string(),
                en["id"].as_u64().expect("id") as u32,
                en["count"].as_u64().expect("count"),
            )
        })
        .collect()
}

/// The ttext events for a calcer as (tokenId, count) pair lists, in dump order.
fn ttext_for(calcer: &str) -> Vec<Vec<(u32, u32)>> {
    let v = load_json(&fixture("ttext.json"));
    v.as_array()
        .expect("ttext.json is an array")
        .iter()
        .filter(|e| e["_calcer"].as_str() == Some(calcer))
        .map(|e| {
            e["pairs"]
                .as_array()
                .expect("pairs is an array")
                .iter()
                .map(|p| {
                    let a = p.as_array().expect("pair is a 2-array");
                    (
                        a[0].as_u64().expect("tokenId") as u32,
                        a[1].as_u64().expect("count") as u32,
                    )
                })
                .collect()
        })
        .collect()
}

/// 1. TOKENIZER bit-exact: the Rust ByDelimiter(space)+lowercase split
///    reproduces the instrumented token_stream for every document, for both the
///    BoW and NaiveBayes fits (the trainer fires the tokenizer once per
///    dictionary build pass; each pass replays the same 16-doc corpus).
#[test]
fn tokenizer_token_stream_bit_exact_vs_instrumented_dump() {
    let opts = TokenizerOptions::default();
    let raw = raw_corpus();

    for calcer in ["BoW", "NaiveBayes"] {
        let dumped = token_stream_for(calcer);
        assert!(
            !dumped.is_empty(),
            "instrumented token_stream must be non-empty for {calcer}"
        );
        // The dump replays the corpus N times (one pass per dictionary build);
        // it must be an exact concatenation of tokenize(raw[i]) cycled over the
        // corpus.
        assert_eq!(
            dumped.len() % raw.len(),
            0,
            "{calcer} token_stream length {} must be a whole multiple of corpus size {}",
            dumped.len(),
            raw.len()
        );
        for (i, dumped_tokens) in dumped.iter().enumerate() {
            let doc = &raw[i % raw.len()];
            let rust_tokens = tokenize(doc, &opts);
            assert_eq!(
                &rust_tokens, dumped_tokens,
                "{calcer} token_stream mismatch at event {i} (doc {:?})",
                doc
            );
        }
    }
}

/// 2. DICTIONARY bit-exact: the Rust frequency dictionary build reproduces the
///    instrumented Word unigram (token-string -> id, count) table EXACTLY,
///    including the (count DESC, token ASC) deterministic sort and the
///    StartTokenId++ id assignment.
#[test]
fn dictionary_token_ids_bit_exact_vs_instrumented_dump() {
    let raw = raw_corpus();
    let tok = TokenizerOptions::default();
    let tokenized: Vec<Vec<String>> = raw.iter().map(|t| tokenize(t, &tok)).collect();

    // OLB resolved from the actual learn-pool size (A4) — not hard-coded.
    let olb = resolve_occurrence_lower_bound(raw.len());
    assert_eq!(olb, 1, "16-row corpus < 1000 -> OLB = 1 (options_helper.cpp)");
    let opts = DictionaryOptions {
        occurrence_lower_bound: olb,
        ..DictionaryOptions::default()
    };
    let (_dict, entries) = build_dictionary(&tokenized, &opts);

    let rust: Vec<(String, u32, u64)> = entries
        .iter()
        .map(|e| (e.token.clone(), e.id, e.count))
        .collect();

    // Both BoW and NaiveBayes dump the SAME Word unigram dictionary; assert
    // against both to prove determinism across fits.
    for calcer in ["BoW", "NaiveBayes"] {
        let expected = dict_ids_for(calcer);
        assert_eq!(
            rust, expected,
            "Word unigram dict_ids mismatch vs {calcer} instrumented dump"
        );
    }
}

/// 3. DIGITIZER / TText bit-exact: the Rust digitizer (tokenize -> Word-dict
///    apply(Skip) -> sorted-RLE TText) reproduces the instrumented NaiveBayes
///    ttext for every document, integer-exact. The NaiveBayes path uses the
///    Word unigram dictionary (BoW's ttext is the BiGram dictionary — deferred).
#[test]
fn digitizer_ttext_bit_exact_vs_instrumented_naive_bayes_dump() {
    let raw = raw_corpus();
    let tok = TokenizerOptions::default();
    let tokenized: Vec<Vec<String>> = raw.iter().map(|t| tokenize(t, &tok)).collect();
    let olb = resolve_occurrence_lower_bound(raw.len());
    let opts = DictionaryOptions {
        occurrence_lower_bound: olb,
        ..DictionaryOptions::default()
    };
    let (dict, _entries) = build_dictionary(&tokenized, &opts);

    let dumped = ttext_for("NaiveBayes");
    assert!(
        !dumped.is_empty(),
        "instrumented NaiveBayes ttext must be non-empty"
    );
    assert_eq!(
        dumped.len() % raw.len(),
        0,
        "NaiveBayes ttext length {} must be a whole multiple of corpus size {}",
        dumped.len(),
        raw.len()
    );

    for (i, dumped_pairs) in dumped.iter().enumerate() {
        let doc = &raw[i % raw.len()];
        let ttext = digitize_document(doc, &tok, &dict);
        let rust_pairs: Vec<(u32, u32)> =
            ttext.pairs().iter().map(|p| (p.token, p.count)).collect();
        assert_eq!(
            &rust_pairs, dumped_pairs,
            "NaiveBayes ttext mismatch at event {i} (doc {:?})",
            doc
        );
    }
}
