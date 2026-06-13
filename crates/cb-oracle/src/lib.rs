//! `cb-oracle` — the parity harness. Reads committed frozen fixtures (the
//! hybrid `.npy` + `config.json` format, D-09) and compares them to computed
//! actuals at absolute error <= 1e-5 (D-12) via a single audited comparator
//! primitive.
//!
//! Restriction lints fire inside test code; `lints.workspace = true` forbids
//! per-crate manifest overrides, so the test-only exemption lives in-code
//! (RESEARCH Pitfall 1).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

mod compare;
mod error;
mod fixture;
mod model_json;

pub use compare::{assert_abs_close, compare_stage, Stage};
pub use error::OracleError;
pub use fixture::{load_config, load_f64_vec, FixtureConfig};
pub use model_json::{load_model_json, ModelJson, ObliviousTree, SplitJson};

#[cfg(test)]
mod compare_test;
#[cfg(test)]
mod fixture_test;
#[cfg(test)]
mod model_json_test;
