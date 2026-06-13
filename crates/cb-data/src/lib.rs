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

pub mod ingest;
mod pool;

pub use pool::{Pair, Pool};

#[cfg(test)]
mod pool_test;
