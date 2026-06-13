#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-train` — the plain gradient-boosting training core (TRAIN-01/02/03).
//!
//! Grows symmetric oblivious trees ([`tree`]) over the generic `cb-compute`
//! `Runtime` boundary and drives the plain boosting loop ([`boosting`]) so a user
//! can train RMSE + Logloss models on the CPU whose splits, leaf values, and
//! staged approximants match upstream catboost 1.2.10 to <= 1e-5.
//!
//! # Parity discipline
//!
//! Every parity-critical float SUM routes through `cb_core::sum_f64` (via
//! `cb-compute`); the split tie-break is the strict `>` first-wins rule
//! (Pitfall 1); depth is capped against `2^depth` overflow (T-03-01-02). No
//! `unwrap`/`expect`/raw float fold in production (deny-lints + D-08).

mod tree;

pub use tree::{
    check_depth, greedy_tensor_search_oblivious, leaf_index, select_best_candidate, Candidate,
    FeatureMatrix, GrownTree, Split, MAX_DEPTH,
};
