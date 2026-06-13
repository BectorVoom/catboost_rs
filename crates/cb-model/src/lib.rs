#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-model` — the canonical serializable model plus serialization and SHAP.
//!
//! Phase 4 re-homes the canonical [`Model`] here (RESEARCH Primary
//! Recommendation): it carries the boosting-order [`ObliviousTree`]s (each with
//! `leaf_values` AND `leaf_weights` — RESEARCH Pitfall 1), the model bias, and
//! the per-float-feature borders, so apply / SHAP / serialize need no training
//! pool. [`Split`] is REUSED from `cb-train` (not redefined).
//!
//! The `generated` module holds the `flatc --rust` FlatBuffers bindings for the
//! vendored upstream `model` / `features` / `ctr_data` schema (D-01), consumed by
//! the later `.cbm` (de)serializer plan.

mod model;

pub use model::{Model, ObliviousTree, Split};
