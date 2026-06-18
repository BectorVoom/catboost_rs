//! Frequency-based dictionary build + `Apply` — the deterministic token→id
//! assignment that makes the digitized text stream reproducible.
//!
//! Verbatim Rust transcription of upstream CatBoost's unigram dictionary
//! builder (`catboost-master/library/cpp/text_processing/dictionary/`
//! `dictionary_builder.cpp:149-199`, `TUnigramDictionaryBuilderImpl::`
//! `FinishBuilding`) and its `Apply` path
//! (`frequency_based_dictionary_impl.cpp:13-25`,
//! `TUnigramDictionaryImpl::ApplyImpl`), 1.2.10 snapshot.
//!
//! # Why this is deterministic despite a different hash map (Pattern 3, SC-1)
//!
//! Upstream counts tokens in a flat hash map (`TokenToCount`,
//! `dictionary_builder.cpp:82`) whose iteration order is unspecified. It is made
//! reproducible by sorting the surviving tokens on `(count DESC, token-string
//! ASC)` (`dictionary_builder.cpp:167-172`) and assigning ids
//! `StartTokenId, StartTokenId+1, …` in that sorted order
//! (`dictionary_builder.cpp:179-183`). The Rust `HashMap` has a *different*
//! iteration order, but the same deterministic comparator yields the same
//! token→id table — bit-exact vs the D-07 `dict_ids.json` dump (SC-1).
//!
//! # Pinned defaults (D-02, RESEARCH Pitfall 3)
//!
//! - `OccurrenceLowerBound` is **data-dependent**:
//!   `learnPoolSize < 1000 ? 1 : 5` (`options_helper.cpp:394-401`). The 16-row
//!   FEAT-01 corpus is `< 1000`, so it resolves to **1**
//!   (`fixtures/text_embedding_inputs/meta.json`,
//!   `occurrence_lower_bound_pinned: 1`). The caller passes the resolved value;
//!   [`resolve_occurrence_lower_bound`] computes it from the pool size so the
//!   boundary is never hard-coded (A4).
//! - The filter is **strict**: a token is kept iff `count >=
//!   OccurrenceLowerBound`, i.e. dropped iff `value < OccurrenceLowerBound`
//!   (`dictionary_builder.cpp:156-158`).
//! - `MaxDictionarySize = 50000`; `-1` means "no limit" (upstream
//!   `GetMaxDictionarySize`, `util.h:12-17`).
//! - `StartTokenId = 0` (upstream default for the Word dictionary in this
//!   single-dictionary path).
//! - Unknown-token policy on `Apply` defaults to **Skip**
//!   (`types.h:13-16`): tokens absent from the dictionary are dropped.
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Empty corpus → empty dictionary; unknown token under Skip → dropped; no
//! `unwrap`/`expect`/`panic`/raw-index.

use std::collections::HashMap;

/// Upstream catboost `DEFAULT_DICTIONARY_BUILDER_OPTIONS` max dictionary size
/// (`text_processing_options.h:43-49`).
pub const DEFAULT_MAX_DICTIONARY_SIZE: u32 = 50_000;

/// Pool-size boundary below which `OccurrenceLowerBound` is 1, else 5
/// (`options_helper.cpp:394-401`).
pub const OCCURRENCE_LOWER_BOUND_POOL_BOUNDARY: usize = 1000;

/// Unknown-token policy for [`Dictionary::apply`] (upstream
/// `EUnknownTokenPolicy`, `types.h:13-16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownTokenPolicy {
    /// Drop tokens absent from the dictionary (upstream `Skip`, the default).
    Skip,
    /// Emit the reserved unknown-token id for absent tokens (upstream
    /// `Insert`). The unknown id is `dictionary_size + StartTokenId`
    /// (`frequency_based_dictionary_impl.h:126`).
    Insert,
}

/// Options pinned to the catboost default unigram (Word) dictionary
/// (`text_processing_options.h` / `options_helper.cpp`). Mirrors the subset of
/// `TDictionaryBuilderOptions` + `TDictionaryOptions` the FEAT-01 fixtures
/// exercise (D-02).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DictionaryOptions {
    /// Tokens with `count < occurrence_lower_bound` are dropped (strict `<`,
    /// `dictionary_builder.cpp:156`).
    pub occurrence_lower_bound: u64,
    /// Maximum surviving tokens after the sort (upstream `MaxDictionarySize`).
    /// `None` means "no limit" (upstream `-1`).
    pub max_dictionary_size: Option<u32>,
    /// First assigned token id (upstream `StartTokenId`, default `0` here).
    pub start_token_id: u32,
}

impl Default for DictionaryOptions {
    fn default() -> Self {
        Self {
            // Caller normally overrides via resolve_occurrence_lower_bound; the
            // struct default uses the small-pool value (1) to match the fixtures.
            occurrence_lower_bound: 1,
            max_dictionary_size: Some(DEFAULT_MAX_DICTIONARY_SIZE),
            start_token_id: 0,
        }
    }
}

/// Resolve the catboost data-dependent `OccurrenceLowerBound`:
/// `learn_pool_size < 1000 ? 1 : 5` (`options_helper.cpp:394-401`,
/// `SetDefaultMinTokenOccurrence`). The boundary is computed here, never
/// hard-coded at a call site (RESEARCH A4).
#[must_use]
pub fn resolve_occurrence_lower_bound(learn_pool_size: usize) -> u64 {
    if learn_pool_size < OCCURRENCE_LOWER_BOUND_POOL_BOUNDARY {
        1
    } else {
        5
    }
}

/// A built frequency dictionary: a token-string → global-token-id map plus the
/// surviving size, mirroring upstream `TUnigramDictionaryImpl`
/// (`frequency_based_dictionary_impl.h:137-141`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dictionary {
    token_to_id: HashMap<String, u32>,
    /// `StartTokenId` carried for the unknown-token id computation.
    start_token_id: u32,
}

/// One surviving dictionary entry, in assigned-id order. Mirrors a row of the
/// D-07 `dict_ids.json` dump (`dictionary_builder.cpp:184`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryEntry {
    /// The token string.
    pub token: String,
    /// The assigned global token id.
    pub id: u32,
    /// The token's occurrence count over the learn corpus.
    pub count: u64,
}

impl Dictionary {
    /// Number of tokens in the dictionary (upstream `Size()`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Whether the dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// The reserved unknown-token id: `dictionary_size + StartTokenId`
    /// (`frequency_based_dictionary_impl.h:126`).
    #[must_use]
    pub fn unknown_token_id(&self) -> u32 {
        // saturating to preserve the no-panic contract on the (unreachable for
        // the 50k-cap dictionary) u32 overflow.
        (self.token_to_id.len() as u32).saturating_add(self.start_token_id)
    }

    /// Look up a token's id (upstream `TokenToId.find`).
    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Apply the dictionary to a token list, mirroring upstream
    /// `TUnigramDictionaryImpl::ApplyImpl`
    /// (`frequency_based_dictionary_impl.cpp:13-25`): for each token, push its
    /// id if present; otherwise push the unknown id under `Insert`, or drop it
    /// under `Skip`.
    #[must_use]
    pub fn apply(&self, tokens: &[String], policy: UnknownTokenPolicy) -> Vec<u32> {
        let mut token_ids: Vec<u32> = Vec::with_capacity(tokens.len());
        for token in tokens {
            if let Some(id) = self.token_to_id.get(token) {
                token_ids.push(*id);
            } else if policy == UnknownTokenPolicy::Insert {
                token_ids.push(self.unknown_token_id());
            }
            // Skip: drop unknown token (no push).
        }
        token_ids
    }
}

/// Build a unigram (Word) frequency dictionary from a corpus of already-
/// tokenized documents, transcribed verbatim from
/// `TUnigramDictionaryBuilderImpl::AddImpl` + `FinishBuilding`
/// (`dictionary_builder.cpp:131-199`).
///
/// Steps (in upstream order):
/// 1. Count each token's total occurrences over all documents
///    (`AddImpl`: `TokenToCount[token] += weight`, weight = 1 per occurrence).
/// 2. Filter: keep token iff `count >= occurrence_lower_bound` (strict `<`
///    drop, `dictionary_builder.cpp:156-158`).
/// 3. Sort surviving tokens by `(count DESC, token-string ASC)`
///    (`dictionary_builder.cpp:167-172`).
/// 4. Truncate to `min(size, MaxDictionarySize)`
///    (`dictionary_builder.cpp:174-175`).
/// 5. Assign ids `StartTokenId, StartTokenId+1, …` in sorted order
///    (`dictionary_builder.cpp:179-183`).
///
/// Returns the [`Dictionary`] plus the surviving [`DictionaryEntry`] rows in
/// assigned-id order (the latter mirrors the D-07 `dict_ids.json` dump for the
/// SC-1 oracle).
#[must_use]
pub fn build_dictionary(
    documents: &[Vec<String>],
    options: &DictionaryOptions,
) -> (Dictionary, Vec<DictionaryEntry>) {
    // Step 1: count tokens (AddImpl, Word level: TokenToCount[token] += 1).
    let mut token_to_count: HashMap<String, u64> = HashMap::new();
    for doc in documents {
        for token in doc {
            *token_to_count.entry(token.clone()).or_insert(0) += 1;
        }
    }

    // Step 2: filter strict `< occurrence_lower_bound` (dictionary_builder.cpp:
    // 155-161). Collect surviving (count, token) into parallel vectors that the
    // sort indexes into, exactly like upstream `counts`/`tokens`.
    let mut counts: Vec<u64> = Vec::with_capacity(token_to_count.len());
    let mut tokens: Vec<String> = Vec::with_capacity(token_to_count.len());
    for (token, count) in &token_to_count {
        if *count < options.occurrence_lower_bound {
            continue;
        }
        counts.push(*count);
        tokens.push(token.clone());
    }

    let dictionary_size = tokens.len();

    // Step 3: sort indices by (count DESC, token ASC) — the deterministic
    // comparator (dictionary_builder.cpp:167-172). This is what makes the
    // assignment independent of the hash-map iteration order.
    let mut indices: Vec<usize> = (0..dictionary_size).collect();
    indices.sort_by(|&l, &r| {
        // counts[l] > counts[r]  -> l first (DESC by count)
        // tie -> tokens[l] < tokens[r] -> l first (ASC by token)
        // l/r come from 0..dictionary_size, so both gets are provably Some;
        // assert it in debug so a future bug panics instead of silently
        // collapsing an out-of-bounds None into Ordering::Equal/Less and
        // corrupting the deterministic (count DESC, token ASC) order (WR-08).
        debug_assert!(
            counts.get(l).is_some()
                && counts.get(r).is_some()
                && tokens.get(l).is_some()
                && tokens.get(r).is_some(),
            "build_dictionary comparator index out of bounds"
        );
        match counts.get(r).cmp(&counts.get(l)) {
            std::cmp::Ordering::Equal => tokens.get(l).cmp(&tokens.get(r)),
            non_eq => non_eq,
        }
    });

    // Step 4: truncate to min(size, MaxDictionarySize) (util.h GetMaxDictionarySize:
    // -1 / None => no limit).
    let max_size = match options.max_dictionary_size {
        Some(m) => m as usize,
        None => usize::MAX,
    };
    let final_size = dictionary_size.min(max_size);

    // Step 5: assign globalTokenId = StartTokenId++ in sorted order
    // (dictionary_builder.cpp:179-183).
    let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(final_size);
    let mut entries: Vec<DictionaryEntry> = Vec::with_capacity(final_size);
    let mut global_token_id = options.start_token_id;
    for &idx in indices.iter().take(final_size) {
        // `.get` keeps the library panic-free; idx is in-bounds by construction.
        let (token, count) = match (tokens.get(idx), counts.get(idx)) {
            (Some(t), Some(c)) => (t.clone(), *c),
            _ => continue,
        };
        token_to_id.insert(token.clone(), global_token_id);
        entries.push(DictionaryEntry {
            token,
            id: global_token_id,
            count,
        });
        global_token_id = global_token_id.saturating_add(1);
    }

    (
        Dictionary {
            token_to_id,
            start_token_id: options.start_token_id,
        },
        entries,
    )
}
