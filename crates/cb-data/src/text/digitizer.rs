//! Digitizer — turns a raw text column into a column of [`TText`] documents by
//! tokenizing each document and applying a [`Dictionary`].
//!
//! Verbatim Rust transcription of upstream CatBoost's per-document digitization
//! (`catboost-master/catboost/private/libs/text_processing/`
//! `text_column_builder.cpp:6-11`, `TTextColumnBuilder::AddText`, and
//! `dictionary.cpp:13-17`, `TDictionaryProxy::Apply`), 1.2.10 snapshot.
//!
//! # Pipeline per document (text_column_builder.cpp:6-11)
//!
//! ```text
//!   text  ──Tokenizer->Tokenize──►  tokens  ──Dictionary->Apply(Skip)──►  tokenIds
//!                                                       └──TText{tokenIds}──►  sorted-RLE TText
//! ```
//!
//! The dictionary `Apply` here uses the **default unknown-token policy = Skip**
//! (`dictionary.h:32-41`), so tokens absent from the dictionary are dropped
//! before the `TText` is built. The `TText` constructor then sorts the surviving
//! ids ascending and RLE-collapses them (`text.h:169-179`).
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Empty column → empty `Vec`; empty document → empty `TText`; unknown tokens
//! dropped. No `unwrap`/`expect`/`panic`/raw-index in this module.

use super::dictionary::{Dictionary, UnknownTokenPolicy};
use super::text::TText;
use super::tokenizer::{tokenize, TokenizerOptions};

/// Digitize a single raw document: tokenize, dictionary-apply (Skip unknown),
/// then build the sorted-RLE [`TText`]. Mirrors
/// `TTextColumnBuilder::AddText` (`text_column_builder.cpp:6-11`).
#[must_use]
pub fn digitize_document(
    text: &str,
    tokenizer_options: &TokenizerOptions,
    dictionary: &Dictionary,
) -> TText {
    let tokens = tokenize(text, tokenizer_options);
    // Default policy Skip (dictionary.h:32-41 / dictionary.cpp:13-17): unknown
    // tokens are dropped.
    let token_ids = dictionary.apply(&tokens, UnknownTokenPolicy::Skip);
    TText::from_token_ids(token_ids)
}

/// Digitize a whole raw text column into a column of [`TText`] documents,
/// mirroring the per-document loop driven by
/// `TTextColumnBuilder::Build` (`text_column_builder.cpp:13-17`).
#[must_use]
pub fn digitize_column(
    texts: &[String],
    tokenizer_options: &TokenizerOptions,
    dictionary: &Dictionary,
) -> Vec<TText> {
    texts
        .iter()
        .map(|text| digitize_document(text, tokenizer_options, dictionary))
        .collect()
}
