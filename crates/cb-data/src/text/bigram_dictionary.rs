//! BiGram (`gram_order = 2`) frequency dictionary build + `Apply` — the second
//! dictionary the default classification BoW calcer applies (RESEARCH Pitfall 4).
//!
//! Verbatim Rust transcription of upstream CatBoost's multigram dictionary
//! builder (`catboost-master/library/cpp/text_processing/dictionary/`
//! `dictionary_builder.cpp:204-291`, `TMultigramDictionaryBuilderImpl::AddImpl`
//! + `Filter`) and its `Apply` path
//! (`frequency_based_dictionary_impl.h:424-477`,
//! `TMultigramDictionaryImpl::ApplyImpl`), 1.2.10 snapshot, pinned to
//! `GramOrder = 2`, `SkipStep = 0`, `EndOfSentenceTokenPolicy = Skip` (the D-02
//! default text-processing configuration).
//!
//! # Why BoW needs TWO dictionaries (RESEARCH Pitfall 4, SC-1 follow-on)
//!
//! The default classification text processing applies BoW over BOTH a BiGram
//! (`gram_order=2`) and a Word (unigram) dictionary
//! (`text_processing_options.cpp:184-202`; `BoW.dicts = ['Bigram','Word']`). The
//! unigram path landed in [`super::dictionary`] (SC-1). This module adds the
//! BiGram path so a document digitizes to two `TText`s — one per dictionary — and
//! the BoW calcer emits `NumBigramTokens + NumWordTokens` binary features. The
//! BoW `ttext` D-07 dump carries the BiGram ids first (up to 24 for the 16-row
//! corpus, `fixtures/text_tokenizer/ttext.json`), then the Word ids.
//!
//! # BiGram key & the `CompareNGram` tie-break (load-bearing, like Pattern 3)
//!
//! A bigram is the ordered consecutive token PAIR `(t[i], t[i+1])`
//! (`dictionary_builder.cpp:228-238`, `skipStep=0`, `tokenCount < GramOrder`
//! documents contribute nothing). Counts are made reproducible exactly like the
//! unigram path: sort surviving bigrams by `(count DESC, ngram ASC)` where the
//! ngram comparison is `CompareNGram` — compare the constituent token STRINGS
//! gram-by-gram ascending (`dictionary_builder.cpp:252-265`) — then assign ids
//! `StartTokenId++` in that order.
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Documents shorter than 2 tokens contribute no bigram; an unknown bigram under
//! Skip is dropped; empty corpus → empty dictionary. No
//! `unwrap`/`expect`/`panic`/raw-index in this module.

use std::collections::HashMap;

use super::dictionary::{DictionaryOptions, UnknownTokenPolicy};

/// A built BiGram (`gram_order=2`) frequency dictionary: an ordered
/// token-pair → global-token-id map plus the carried `StartTokenId` for the
/// unknown-token id computation. Mirrors upstream `TMultigramDictionaryImpl<2>`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BigramDictionary {
    pair_to_id: HashMap<(String, String), u32>,
    /// `StartTokenId` carried for the unknown-token id computation.
    start_token_id: u32,
}

/// One surviving BiGram dictionary entry, in assigned-id order. Mirrors a row of
/// the multigram `dict_ids` dump (the constituent tokens + id + count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigramEntry {
    /// The first token of the consecutive pair.
    pub first: String,
    /// The second token of the consecutive pair.
    pub second: String,
    /// The assigned global token id.
    pub id: u32,
    /// The bigram's occurrence count over the learn corpus.
    pub count: u64,
}

impl BigramDictionary {
    /// Number of bigrams in the dictionary (upstream `Size()`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.pair_to_id.len()
    }

    /// Whether the dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pair_to_id.is_empty()
    }

    /// The reserved unknown-token id: `dictionary_size + StartTokenId`
    /// (`frequency_based_dictionary_impl.h:419-420`).
    ///
    /// NOTE (IN-05): for an empty dictionary this is `0`, colliding with the id
    /// a single-entry dictionary would assign its first pair. The digitizer
    /// default `Skip` policy never emits this id; `Insert` over an empty
    /// dictionary is undefined and not used.
    #[must_use]
    pub fn unknown_token_id(&self) -> u32 {
        (self.pair_to_id.len() as u32).saturating_add(self.start_token_id)
    }

    /// Apply the BiGram dictionary to a token list, mirroring upstream
    /// `TMultigramDictionaryImpl<2>::ApplyImpl`
    /// (`frequency_based_dictionary_impl.h:424-477`) at `GramOrder=2`,
    /// `SkipStep=0`, `EndOfSentence=Skip`: form each consecutive token pair
    /// `(tokens[i], tokens[i+1])` for `i in 0..endTokenIndex` where
    /// `endTokenIndex = tokenCount >= 2 ? tokenCount - 1 : 0`
    /// (`multigram_dictionary_helpers.h:74-77`), and emit that bigram's id if
    /// present. Under `Skip` an unknown bigram (either constituent token unknown,
    /// or the pair absent from the dictionary) is dropped.
    #[must_use]
    pub fn apply(&self, tokens: &[String], policy: UnknownTokenPolicy) -> Vec<u32> {
        // endTokenIndex = lastGramTokenIndex < tokenCount ? tokenCount - 1 : 0
        // with lastGramTokenIndex = (GramOrder-1)*(skipStep+1) = 1.
        let end_token_index = tokens.len().saturating_sub(1);
        let mut token_ids: Vec<u32> = Vec::with_capacity(end_token_index);

        for i in 0..end_token_index {
            // Form the consecutive pair (tokens[i], tokens[i+1]) via checked
            // access; the no-panic library contract is preserved even though the
            // indices are in-bounds by construction.
            let (first, second) = match (tokens.get(i), tokens.get(i + 1)) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let key = (first.clone(), second.clone());
            if let Some(id) = self.pair_to_id.get(&key) {
                token_ids.push(*id);
            } else if policy == UnknownTokenPolicy::Insert {
                token_ids.push(self.unknown_token_id());
            }
            // Skip: drop the unknown bigram (no push).
        }
        token_ids
    }
}

/// Build a BiGram (`gram_order=2`) frequency dictionary from a corpus of
/// already-tokenized documents, transcribed from
/// `TMultigramDictionaryBuilderImpl<2>::AddImpl` + `Filter`
/// (`dictionary_builder.cpp:204-291`), pinned to `SkipStep=0`,
/// `EndOfSentence=Skip`.
///
/// Steps (in upstream order):
/// 1. Count each consecutive token pair `(t[i], t[i+1])` over all documents;
///    documents with `< 2` tokens contribute nothing
///    (`dictionary_builder.cpp:226-238`).
/// 2. Filter: keep a bigram iff `count >= occurrence_lower_bound` (strict `<`
///    drop, `dictionary_builder.cpp:276-278`).
/// 3. Sort surviving bigrams by `(count DESC, ngram ASC)` where the ngram
///    comparison compares the constituent token STRINGS gram-by-gram ascending
///    (`CompareNGram`, `dictionary_builder.cpp:252-265, 288-293`).
/// 4. Truncate to `min(size, MaxDictionarySize)`
///    (`dictionary_builder.cpp:294-295`).
/// 5. Assign ids `StartTokenId, StartTokenId+1, …` in sorted order.
///
/// Returns the [`BigramDictionary`] plus the surviving [`BigramEntry`] rows in
/// assigned-id order.
#[must_use]
pub fn build_bigram_dictionary(
    documents: &[Vec<String>],
    options: &DictionaryOptions,
) -> (BigramDictionary, Vec<BigramEntry>) {
    // Step 1: count consecutive token pairs (AddImpl, GramOrder=2, skipStep=0).
    let mut pair_to_count: HashMap<(String, String), u64> = HashMap::new();
    for doc in documents {
        // tokenCount < GramOrder (=2) -> contributes nothing.
        if doc.len() < 2 {
            continue;
        }
        for i in 0..doc.len() - 1 {
            if let (Some(a), Some(b)) = (doc.get(i), doc.get(i + 1)) {
                *pair_to_count.entry((a.clone(), b.clone())).or_insert(0) += 1;
            }
        }
    }

    // Step 2: filter strict `< occurrence_lower_bound`. Collect surviving
    // (count, pair) into parallel vectors the sort indexes into.
    let mut counts: Vec<u64> = Vec::with_capacity(pair_to_count.len());
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(pair_to_count.len());
    for (pair, count) in &pair_to_count {
        if *count < options.occurrence_lower_bound {
            continue;
        }
        counts.push(*count);
        pairs.push(pair.clone());
    }

    let dictionary_size = pairs.len();

    // Step 3: sort indices by (count DESC, ngram ASC). CompareNGram compares the
    // constituent token strings gram-by-gram ascending: first tokens, then (on a
    // first-token tie) second tokens — exactly tuple-ordering on (first, second).
    let mut indices: Vec<usize> = (0..dictionary_size).collect();
    indices.sort_by(|&l, &r| {
        // l/r come from 0..dictionary_size, so both gets are provably Some;
        // assert it in debug so a future bug panics instead of silently
        // collapsing an out-of-bounds None into Ordering::Equal/Less and
        // corrupting the deterministic (count DESC, ngram ASC) order (WR-08).
        debug_assert!(
            counts.get(l).is_some()
                && counts.get(r).is_some()
                && pairs.get(l).is_some()
                && pairs.get(r).is_some(),
            "build_bigram_dictionary comparator index out of bounds"
        );
        match counts.get(r).cmp(&counts.get(l)) {
            std::cmp::Ordering::Equal => pairs.get(l).cmp(&pairs.get(r)),
            non_eq => non_eq,
        }
    });

    // Step 4: truncate to min(size, MaxDictionarySize).
    let max_size = match options.max_dictionary_size {
        Some(m) => m as usize,
        None => usize::MAX,
    };
    let final_size = dictionary_size.min(max_size);

    // Step 5: assign globalTokenId = StartTokenId++ in sorted order.
    let mut pair_to_id: HashMap<(String, String), u32> = HashMap::with_capacity(final_size);
    let mut entries: Vec<BigramEntry> = Vec::with_capacity(final_size);
    let mut global_token_id = options.start_token_id;
    for &idx in indices.iter().take(final_size) {
        let (pair, count) = match (pairs.get(idx), counts.get(idx)) {
            (Some(p), Some(c)) => (p.clone(), *c),
            _ => continue,
        };
        pair_to_id.insert(pair.clone(), global_token_id);
        entries.push(BigramEntry {
            first: pair.0,
            second: pair.1,
            id: global_token_id,
            count,
        });
        global_token_id = global_token_id.saturating_add(1);
    }

    (
        BigramDictionary {
            pair_to_id,
            start_token_id: options.start_token_id,
        },
        entries,
    )
}
