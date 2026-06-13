//! Automatic learning-rate selection (TRAIN-08) — RED stub.
//!
//! Will port upstream `TAutoLRParamsGuesser` (`options_helper.cpp:116-262`).

use cb_core::{CbError, CbResult};

/// Auto-LR target classification (upstream `ETargetType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetType {
    /// RMSE regression.
    Rmse,
    /// Logloss / cross-entropy binary classification.
    Logloss,
    /// Anything not in the auto-LR table.
    Unknown,
}

/// Look up the `{A,B,C,D}` coefficient row (CPU). RED stub: always `None`.
#[must_use]
pub fn coefficients(
    _target: TargetType,
    _use_best_model: bool,
    _boost_from_average: bool,
) -> Option<[f64; 4]> {
    None
}

/// Guess the learning rate. RED stub: returns a wrong constant.
pub fn guess(
    _target: TargetType,
    _use_best_model: bool,
    _boost_from_average: bool,
    _learn_object_count: usize,
    _iter_count: usize,
) -> CbResult<f64> {
    Err(CbError::Degenerate("autolr not yet implemented".to_string()))
}

#[cfg(test)]
#[path = "autolr_test.rs"]
mod tests;
