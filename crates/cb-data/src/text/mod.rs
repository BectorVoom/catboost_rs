//! Text feature processing: the load-bearing D-01 / SC-1 tokenizer →
//! dictionary → digitizer → `TText` pipeline.
//!
//! This module is a verbatim Rust transcription of upstream CatBoost's
//! text-processing front end (1.2.10 snapshot). It is the **first gate** of
//! Phase 6.5 (CONTEXT D-01): the upstream token / dictionary-id / `TText`
//! stream must be reproduced bit-identically before any text/embedding calcer
//! oracle is attempted, because a different split, a wrong
//! `OccurrenceLowerBound`, or a missed lowercasing step silently shifts every
//! downstream feature index (RESEARCH Pitfall 3, SC-1).
//!
//! # Pipeline (RESEARCH architecture diagram)
//!
//! ```text
//!   raw text  ──tokenizer──►  tokens  ──dictionary.apply──►  token ids
//!                                                    └──TText::from_token_ids──►  (tokenId,count) RLE
//! ```
//!
//! - [`tokenizer`] — `ByDelimiter` split + lowercasing + `SkipEmpty`.
//! - [`dictionary`] — frequency-count dictionary build (strict
//!   `OccurrenceLowerBound` filter, `(count DESC, token ASC)` sort,
//!   `MaxDictionarySize` truncate, `StartTokenId++` ids) + `Apply` (Skip
//!   unknown tokens).
//! - [`digitizer`] — per text column: tokenize each document, dictionary-apply,
//!   build the sorted-RLE [`text::TText`].
//! - [`text`] — the [`text::TText`] sorted-RLE `(tokenId, count)` data type.
//!
//! Layout mirrors `cb-train::ctr/mod.rs` (`#[path = ...]` submodules, sibling
//! `_test.rs` files; INFRA-06 source/test separation).

// `text::text` is the `TText` data-type module, deliberately named to mirror
// upstream `data_types/text.h`; the inner-same-name is intentional (the plan's
// `files_modified` pins `text/text.rs`).
#[allow(clippy::module_inception)]
#[path = "text.rs"]
pub mod text;
#[path = "tokenizer.rs"]
pub mod tokenizer;
#[path = "dictionary.rs"]
pub mod dictionary;
#[path = "digitizer.rs"]
pub mod digitizer;

#[cfg(test)]
#[path = "text_test.rs"]
mod text_test;
#[cfg(test)]
#[path = "tokenizer_test.rs"]
mod tokenizer_test;
#[cfg(test)]
#[path = "dictionary_test.rs"]
mod dictionary_test;
#[cfg(test)]
#[path = "digitizer_test.rs"]
mod digitizer_test;
