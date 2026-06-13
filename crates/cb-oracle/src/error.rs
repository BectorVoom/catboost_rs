//! Error type for the oracle harness, derived with [`thiserror`].
//!
//! Read/parse errors from `ndarray-npy`, `serde_json`, and `std::io` are wrapped
//! via `#[from]` so loaders can propagate with `?` and never panic (mitigates
//! T-01-01: malformed fixtures must error, not abort).

/// Errors surfaced by the oracle fixture loaders and comparator.
#[derive(Debug, thiserror::Error)]
pub enum OracleError {
    /// Expected and actual slices had different lengths.
    #[error("length mismatch: expected {expected} values, got {actual}")]
    LengthMismatch {
        /// Length of the expected (oracle) slice.
        expected: usize,
        /// Length of the actual (computed) slice.
        actual: usize,
    },

    /// A paired value diverged beyond the tolerance at `index`.
    #[error("diverged at index {index}: expected {expected}, actual {actual}, |diff| = {diff}")]
    Diverged {
        /// First index whose absolute difference exceeded the tolerance.
        index: usize,
        /// Expected (oracle) value at `index`.
        expected: f64,
        /// Actual (computed) value at `index`.
        actual: f64,
        /// Absolute difference `|expected - actual|`.
        diff: f64,
    },

    /// A stage-tagged length mismatch (`compare_stage`): the expected and actual
    /// slices for `stage` had different lengths.
    #[error("stage {stage:?}: length mismatch: expected {expected} values, got {actual}")]
    StageLengthMismatch {
        /// Oracle stage the mismatch occurred in.
        stage: crate::compare::Stage,
        /// Length of the expected (oracle) slice.
        expected: usize,
        /// Length of the actual (computed) slice.
        actual: usize,
    },

    /// A stage-tagged divergence (`compare_stage`): a paired value for `stage`
    /// diverged beyond `1e-5` at `index`.
    #[error("stage {stage:?}: diverged at index {index}: expected {expected}, actual {actual}, |diff| = {diff}")]
    StageDiverged {
        /// Oracle stage the divergence occurred in.
        stage: crate::compare::Stage,
        /// First index whose absolute difference exceeded the tolerance.
        index: usize,
        /// Expected (oracle) value at `index`.
        expected: f64,
        /// Actual (computed) value at `index`.
        actual: f64,
        /// Absolute difference `|expected - actual|`.
        diff: f64,
    },

    /// Failed to read a `.npy` fixture (bad header, dtype mismatch, etc.).
    #[error("failed to read .npy fixture: {0}")]
    Npy(#[from] ndarray_npy::ReadNpyError),

    /// Failed to parse a `config.json` fixture.
    #[error("failed to parse config.json: {0}")]
    Json(#[from] serde_json::Error),

    /// Underlying I/O error while reading a fixture file.
    #[error("fixture I/O error: {0}")]
    Io(#[from] std::io::Error),
}
