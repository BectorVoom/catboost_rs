//! `TText` — the digitized-text data type: a run-length-encoded list of
//! `(tokenId, count)` pairs **sorted ascending by tokenId**.
//!
//! This is a verbatim Rust transcription of upstream CatBoost's `NCB::TText`
//! (`catboost-master/catboost/private/libs/data_types/text.h:77-213`,
//! 1.2.10 snapshot). A digitized document is built by sorting the raw token-id
//! vector ascending, then collapsing runs of equal ids into `(tokenId, count)`
//! pairs (`text.h:169-179`, the `TText(TVector<ui32>&&)` constructor).
//!
//! # Why the sort-then-RLE order is load-bearing (D-01 / SC-1)
//!
//! Downstream calcers (notably BoW, `bow.cpp:7-21`) walk a `TText` and the
//! active feature-id set **in lockstep assuming both are sorted ascending by
//! tokenId**. A different ordering silently shifts every downstream feature
//! value. The pairs are therefore always kept sorted ascending and never
//! re-ordered after construction (RESEARCH Pattern 2). The token ids dumped by
//! the D-07 instrumented trainer (`text.h:178` `cb_instr_ttext` hook) are
//! exactly the `(Token(), Count())` pairs produced here, so this type is
//! oracle-gated bit-exact against `fixtures/text_tokenizer/ttext.json`.
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Empty input yields an empty `TText` (never a panic). All access is via
//! checked iteration / `.get(..)`; there is no `unwrap`/`expect`/`panic`/raw
//! index in this module (CLAUDE.md library discipline).

/// A `(tokenId, count)` pair, mirroring upstream `TText::TTokenToCountPair`
/// (`text.h:79-115`). `token` is the global dictionary token id; `count` is the
/// number of times that token appeared in the document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCount {
    /// Global dictionary token id (upstream `TTokenId::Id`, `text.h:28-29`).
    pub token: u32,
    /// Occurrence count of `token` in the document (upstream `Counter`, a
    /// `ui32`; `text.h:81`).
    pub count: u32,
}

/// A digitized text: `(tokenId, count)` pairs **sorted ascending by tokenId**,
/// run-length-encoded. Mirrors upstream `NCB::TText` (`text.h:77-213`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TText {
    /// Sorted-ascending, RLE-collapsed `(tokenId, count)` storage
    /// (upstream `TokenToCount`, `text.h:212`).
    token_to_count: Vec<TokenCount>,
}

impl TText {
    /// Build a [`TText`] from a vector of raw token ids, exactly mirroring the
    /// upstream `TText(TVector<ui32>&& tokenIds)` constructor
    /// (`text.h:169-179`): sort ascending, then collapse equal-id runs into
    /// `(tokenId, count)` pairs.
    ///
    /// Empty input yields an empty `TText`.
    #[must_use]
    pub fn from_token_ids(mut token_ids: Vec<u32>) -> Self {
        // `Sort(tokenIds);` (text.h:170) — ascending numeric sort.
        token_ids.sort_unstable();

        // RLE-collapse: `if (back().Token() != tokenId) push {id,1}; else
        // IncreaseCount();` (text.h:171-177).
        let mut token_to_count: Vec<TokenCount> = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            match token_to_count.last_mut() {
                Some(last) if last.token == token_id => {
                    // `IncreaseCount()` — `Counter++` (text.h:101-103). Saturate
                    // rather than wrap on the (practically unreachable) u32
                    // overflow to preserve the no-panic library contract.
                    last.count = last.count.saturating_add(1);
                }
                _ => token_to_count.push(TokenCount {
                    token: token_id,
                    count: 1,
                }),
            }
        }

        Self { token_to_count }
    }

    /// The sorted-ascending `(tokenId, count)` pairs (upstream iteration over
    /// `TokenToCount`, `text.h:181-187`).
    #[must_use]
    pub fn pairs(&self) -> &[TokenCount] {
        &self.token_to_count
    }

    /// Number of distinct tokens (RLE pairs) in the document (upstream
    /// `TokenToCount.size()`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_to_count.len()
    }

    /// Whether the document has no tokens (upstream `TokenToCount.empty()`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_to_count.is_empty()
    }
}
