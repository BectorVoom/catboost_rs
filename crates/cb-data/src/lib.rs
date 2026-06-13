//! `cb-data` — Pool, dataset, and quantization layer for catboost-rs.
//!
//! This crate owns the in-memory dataset representation ([`Pool`]) and the
//! parity-critical quantization primitives ([`select_borders_greedy_logsum`]).
//! Datasets are built through the [`IngestSource`] trait seam (D-04): Phase 2
//! ships the owned-`Vec` primitive ([`ingest::OwnedColumns`]); a borrowed /
//! zero-copy view can plug into the same seam at Phase 8 without reshaping
//! [`Pool`] (D-02 — owned now, no lifetime generic).
//!
//! # Test-lint exemption
//!
//! The restriction lints denied workspace-wide (`unwrap_used`, `expect_used`,
//! `panic`, `indexing_slicing`) also fire inside `#[test]` code, where
//! `unwrap()` / indexing are idiomatic. Because `lints.workspace = true`
//! forbids per-crate manifest lint overrides, the test-only exemption must live
//! in-code (RESEARCH Pitfall 1).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

mod borders;
pub mod ingest;
mod nan_mode;
mod pool;
mod quantize;
mod quantized_pool;

pub use borders::{penalty_maxsumlog, select_borders_greedy_logsum};
pub use nan_mode::{bin_of, insert_sentinel, nan_bin, NanMode};
pub use pool::{Pair, Pool};
pub use quantize::QuantizeParams;
pub use quantized_pool::{
    pack_bins, select_bin_width, ColumnBins, FeatureKind, QuantizedPool,
};

#[cfg(test)]
mod borders_test;
#[cfg(test)]
mod nan_mode_test;
#[cfg(test)]
mod pool_test;
#[cfg(test)]
mod quantized_pool_test;
