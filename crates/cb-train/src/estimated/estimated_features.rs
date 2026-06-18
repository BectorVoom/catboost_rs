//! BoW estimated-feature materialization (SC-4 first slice) — raw text columns →
//! BoW binary presence float columns → EXISTING quantizer → tree-search inputs.
//!
//! This is the SC-4 integration seam for the target-INDEPENDENT BoW calcer. It
//! takes a raw text column, builds the BiGram + Word dictionaries ONCE (offline,
//! target-independent — RESEARCH Anti-Pattern "recompute dictionary per fold"),
//! digitizes every document against each dictionary, runs the BoW calcer
//! ([`cb_compute::bag_of_words_compute`]) to produce one binary presence column
//! per active token id, appends those columns to the float-feature layout, and
//! selects each column's split borders through the UNCHANGED
//! [`cb_data::select_borders_greedy_logsum`] quantizer (NO parallel quantizer,
//! SC-4). The result feeds straight into [`crate::train`].
//!
//! # Feature ordering (load-bearing, RESEARCH Pitfall 4)
//!
//! The default classification BoW applies over the BiGram dictionary FIRST, then
//! the Word dictionary (`BoW.dicts = ['Bigram','Word']`,
//! `text_processing_options.cpp:184-202`). The BoW `ttext` D-07 dump confirms
//! this order (BiGram ids first, then Word ids). The estimated columns are
//! therefore emitted BiGram-block then Word-block, each block in ascending
//! active-token-id order (the BoW lockstep order). A different block order would
//! shift every estimated feature index.
//!
//! # Inert when absent (D-04 byte-identical)
//!
//! An empty text column yields no estimated columns and no borders, so the
//! existing numeric/categorical training path is byte-for-byte unchanged.
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Empty documents digitize to empty `TText`s (all-zero presence); a degenerate
//! single-value column yields no border (the quantizer drops it). No
//! `unwrap`/`expect`/`panic`/raw-index in this module.

use cb_compute::bag_of_words_compute;
use cb_core::{CbError, CbResult};
use cb_data::select_borders_greedy_logsum;
use cb_data::text::bigram_dictionary::build_bigram_dictionary;
use cb_data::text::dictionary::{
    build_dictionary, resolve_occurrence_lower_bound, DictionaryOptions, DEFAULT_MAX_DICTIONARY_SIZE,
};
use cb_data::text::digitizer::{digitize_column, digitize_column_bigram};
use cb_data::text::tokenizer::{tokenize, TokenizerOptions};

/// The BoW estimated feature columns plus their selected split borders, in the
/// canonical BiGram-block-then-Word-block order, ready to append to the
/// float-feature layout and hand to [`crate::train`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BowEstimatedFeatures {
    /// One binary presence column per active token id, column-major
    /// (`columns[f][doc]`), `f32`-valued (estimated features are `f32`,
    /// RESEARCH Pitfall 6). BiGram block first, then Word block.
    pub columns: Vec<Vec<f32>>,
    /// The split borders for each column in [`Self::columns`], in the SAME order
    /// (selected via the existing `select_borders_greedy_logsum`). A binary
    /// presence column yields a single border at `0.5`.
    pub borders: Vec<Vec<f64>>,
    /// Number of BiGram-dictionary feature columns (the size of the BiGram
    /// block, for downstream feature-index bookkeeping / oracle introspection).
    pub bigram_feature_count: usize,
    /// Number of Word-dictionary feature columns (the size of the Word block).
    pub word_feature_count: usize,
}

/// Build the BoW estimated feature columns for a single raw text column.
///
/// `texts` is the per-document raw text (length `n_docs`). `tokenizer_options`
/// pins the D-02 ByDelimiter(space)+lowercase tokenization. `max_borders` is the
/// per-feature border budget passed to the existing quantizer (binary columns
/// need only 1, but the budget is plumbed so the seam matches the numeric path).
///
/// The `OccurrenceLowerBound` is resolved data-dependently from `texts.len()`
/// (`learn_pool_size < 1000 ? 1 : 5`, A4) — never hard-coded. `MaxDictionarySize`
/// is pinned to the catboost default (50000). Both dictionaries use
/// `StartTokenId = 0` (each dictionary's BoW block is indexed from 0 over its own
/// active ids, exactly as upstream `ActiveFeatureIndices = Iota(0..NumTokens)`).
///
/// # Errors
///
/// Returns [`CbError::Degenerate`] if a BoW compute step fails (it is currently
/// infallible, but the error is plumbed so the online calcers share the
/// signature).
pub fn build_bow_estimated_features(
    texts: &[String],
    tokenizer_options: &TokenizerOptions,
    max_borders: usize,
) -> CbResult<BowEstimatedFeatures> {
    // Inert when absent: no documents -> no estimated features (D-04).
    if texts.is_empty() {
        return Ok(BowEstimatedFeatures::default());
    }

    let n_docs = texts.len();

    // Build the two dictionaries ONCE over the learn corpus (offline,
    // target-independent). Tokenize each document once for the dictionary build.
    let tokenized: Vec<Vec<String>> = texts
        .iter()
        .map(|t| tokenize(t, tokenizer_options))
        .collect();

    let olb = resolve_occurrence_lower_bound(n_docs);
    let dict_options = DictionaryOptions {
        occurrence_lower_bound: olb,
        max_dictionary_size: Some(DEFAULT_MAX_DICTIONARY_SIZE),
        start_token_id: 0,
    };

    let (bigram_dict, bigram_entries) = build_bigram_dictionary(&tokenized, &dict_options);
    let (word_dict, word_entries) = build_dictionary(&tokenized, &dict_options);

    let bigram_feature_count = bigram_entries.len();
    let word_feature_count = word_entries.len();

    // Digitize every document against each dictionary (one TText per doc per
    // dictionary).
    let bigram_texts = digitize_column_bigram(texts, tokenizer_options, &bigram_dict);
    let word_texts = digitize_column(texts, tokenizer_options, &word_dict);

    // Active feature ids = Iota(0..NumTokens) for each untrimmed BoW calcer
    // (upstream `ActiveFeatureIndices`). Build them ascending.
    let bigram_active: Vec<u32> = (0..bigram_feature_count as u32).collect();
    let word_active: Vec<u32> = (0..word_feature_count as u32).collect();

    // Per document, compute the BoW presence row for each dictionary. The row is
    // the binary presence vector over that dictionary's active ids. We then
    // transpose to column-major for the trainer (`columns[f][doc]`).
    let total_features = bigram_feature_count + word_feature_count;
    let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n_docs); total_features];

    for doc in 0..n_docs {
        // BiGram block first (RESEARCH Pitfall 4 ordering).
        let bigram_text = bigram_texts
            .get(doc)
            .ok_or_else(|| CbError::Degenerate(format!("missing bigram TText for doc {doc}")))?;
        let bigram_row = bag_of_words_compute(bigram_text, &bigram_active)?;

        let word_text = word_texts
            .get(doc)
            .ok_or_else(|| CbError::Degenerate(format!("missing word TText for doc {doc}")))?;
        let word_row = bag_of_words_compute(word_text, &word_active)?;

        // Scatter the document's presence cells into the column-major layout:
        // BiGram block at [0, bigram_feature_count), Word block after it.
        for (f, &v) in bigram_row.iter().enumerate() {
            if let Some(col) = columns.get_mut(f) {
                col.push(v as f32);
            }
        }
        for (f, &v) in word_row.iter().enumerate() {
            if let Some(col) = columns.get_mut(bigram_feature_count + f) {
                col.push(v as f32);
            }
        }
    }

    // Select borders for each estimated column through the EXISTING quantizer
    // (SC-4 — NO parallel quantizer). A binary presence column yields a single
    // border at 0.5.
    let borders: Vec<Vec<f64>> = columns
        .iter()
        .map(|col| {
            let as_f64: Vec<f64> = col.iter().map(|&v| f64::from(v)).collect();
            select_borders_greedy_logsum(&as_f64, max_borders, false)
        })
        .collect();

    Ok(BowEstimatedFeatures {
        columns,
        borders,
        bigram_feature_count,
        word_feature_count,
    })
}
