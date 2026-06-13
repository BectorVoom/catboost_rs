//! `cb-core` — shared error types and core primitives for catboost-rs.
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

mod error;
mod reduction;
mod rng;

pub use error::{CbError, CbResult};
pub use reduction::{sum_f32_in_f64, sum_f64};
pub use rng::TFastRng64;

#[cfg(test)]
mod error_test;
#[cfg(test)]
mod reduction_test;
#[cfg(test)]
mod rng_test;
