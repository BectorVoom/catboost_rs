//! Per-stage absolute-error comparator — the single audited parity primitive
//! every later phase reuses (RESEARCH Pattern 3, D-12). Returns `Result`, never
//! panics, and avoids indexing (mitigates T-01-02).

use crate::error::OracleError;

/// Oracle comparison stages. Each later phase compares one or more of these
/// against the pinned reference at absolute error <= 1e-5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Quantization borders.
    Borders,
    /// Per-tree split definitions.
    Splits,
    /// Per-tree leaf values.
    LeafValues,
    /// Per-iteration (staged) approximants.
    StagedApprox,
    /// Final model predictions.
    Predictions,
}

/// Asserts that every paired value in `expected` and `actual` is within `tol`
/// (absolute error).
///
/// Call sites default `tol` to `1e-5` (D-12).
///
/// # Errors
/// - [`OracleError::LengthMismatch`] if the slices differ in length.
/// - [`OracleError::Diverged`] at the first index whose absolute difference
///   exceeds `tol`.
pub fn assert_abs_close(expected: &[f64], actual: &[f64], tol: f64) -> Result<(), OracleError> {
    if expected.len() != actual.len() {
        return Err(OracleError::LengthMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (index, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > tol {
            return Err(OracleError::Diverged {
                index,
                expected: *e,
                actual: *a,
                diff,
            });
        }
    }
    Ok(())
}
