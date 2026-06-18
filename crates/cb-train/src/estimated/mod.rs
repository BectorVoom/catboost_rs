//! Estimated-feature integration seam (SC-4) — calcer outputs join the EXISTING
//! quantization → tree path.
//!
//! Text/embedding calcers (BoW here; NaiveBayes/BM25/LDA/KNN in later plans)
//! produce numeric float feature columns. Those columns are NOT a new feature
//! kind and do NOT get a parallel quantizer (RESEARCH "Don't Hand-Roll", SC-4):
//! they are appended to the float-feature layout and flow through the UNCHANGED
//! `cb_data::select_borders_greedy_logsum` → tree search. This module is the
//! single seam where that append + border-selection happens, so the four online
//! calcers (Plans 04-07) extend it additively.
//!
//! # Inert when absent (D-04 byte-identical)
//!
//! With no text columns the seam emits no estimated features and the existing
//! numeric/categorical training path is byte-for-byte unchanged. This is the
//! load-bearing non-regression property the no-text suite gates.
//!
//! # Module layout
//!
//! Mirrors `cb-train::ctr/mod.rs` (`#[path = ...]` submodules + sibling
//! `_test.rs`; INFRA-06 source/test separation).

#[path = "online_text.rs"]
pub mod online_text;

#[path = "online_embedding.rs"]
pub mod online_embedding;

#[path = "estimated_features.rs"]
pub mod estimated_features;

#[cfg(test)]
#[path = "estimated_features_test.rs"]
mod estimated_features_test;

#[cfg(test)]
#[path = "online_text_test.rs"]
mod online_text_test;

#[cfg(test)]
#[path = "online_embedding_test.rs"]
mod online_embedding_test;
