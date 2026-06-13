//! Overfitting detection / early stopping (TRAIN-06) — RED skeleton.
//!
//! Filled in by the GREEN step; this skeleton only establishes the public API so
//! the unit tests in `overfit_test.rs` compile and FAIL first (TDD RED).

use cb_core::{CbError, CbResult};

#[cfg(test)]
#[path = "overfit_test.rs"]
mod tests;

/// The overfitting-detector type (`EOverfittingDetectorType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EOverfittingDetectorType {
    /// No detection.
    None,
    /// IncToDec (the default).
    IncToDec,
    /// Iter (IncToDec with threshold forced to 1.0).
    Iter,
    /// Wilcoxon signed-rank over post-local-max deltas.
    Wilcoxon,
}

/// The overfitting-detection state machine.
#[derive(Debug)]
pub struct OverfittingDetector;

impl OverfittingDetector {
    /// Construct a detector. RED stub.
    ///
    /// # Errors
    /// Stub always errors so the GREEN step replaces it.
    pub fn new(
        _detector_type: EOverfittingDetectorType,
        _threshold: f64,
        _iterations_wait: usize,
        _has_test: bool,
    ) -> CbResult<Self> {
        Err(CbError::Degenerate("overfit RED stub".to_owned()))
    }

    /// Whether the detector is active. RED stub.
    #[must_use]
    pub fn is_active(&self) -> bool {
        false
    }

    /// Feed one eval-metric value. RED stub.
    pub fn add_error(&mut self, _err: f64) {}

    /// Whether training should stop. RED stub.
    #[must_use]
    pub fn is_need_stop(&self) -> bool {
        false
    }
}

/// Tracks the best (lowest-loss) iteration for `use_best_model`. RED stub.
#[derive(Debug)]
pub struct BestModelTracker;

impl BestModelTracker {
    /// New tracker. RED stub.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Feed one eval-metric value. RED stub.
    pub fn add_error(&mut self, _err: f64) {}

    /// The best iteration so far. RED stub.
    #[must_use]
    pub fn best_iteration(&self) -> Option<usize> {
        None
    }
}

impl Default for BestModelTracker {
    fn default() -> Self {
        Self::new()
    }
}
