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
    /// Per-fold object permutation indices (Phase-5 Wave-0, D-02/D-03).
    ///
    /// Unlike every other stage this is compared **integer-exact** (`!=`), NOT
    /// at the `1e-5` tolerance: the permutation is the D-03 linchpin and any
    /// single-index mismatch must be rejected BEFORE any value stage runs. Use
    /// [`compare_permutation`] for this stage, not [`compare_stage`].
    Permutation,
    /// Per-object online (ordered) CTR values (Phase-5 Wave-0, D-02).
    ///
    /// Routed through the existing `≤1e-5` [`compare_stage`] path.
    OnlineCtr,
    /// Per-object per-iteration ordered-boosting approximants (Phase-5 Wave-0,
    /// D-02). Routed through the existing `≤1e-5` [`compare_stage`] path.
    OrderedApprox,
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
        // `!(diff <= tol)` rather than `diff > tol`: a non-finite `diff` — NaN
        // from a NaN/Inf `actual`, or Inf−Inf — must count as divergence, not
        // silently pass the gate (`NaN > tol` is always false, `NaN <= tol` too).
        if !(diff <= tol) {
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

/// Per-stage convenience wrapper over [`assert_abs_close`] at the fixed `1e-5`
/// parity tolerance (D-12), tagging any failure with the [`Stage`] it occurred
/// in so callers know which oracle stage drifted (INFRA-04).
///
/// This is the per-stage API surface the later phases call with Rust-computed
/// `actual` values: cb-train (P3) supplies `Stage::StagedApprox`/`Predictions`
/// actuals, cb-model (P4) supplies `Stage::Splits`/`Stage::LeafValues`. Phase 1
/// has no Rust algorithm yet, so it only proves the API gates falsifiably on
/// real oracle fixtures.
///
/// # Errors
/// - [`OracleError::StageLengthMismatch`] if the slices differ in length.
/// - [`OracleError::StageDiverged`] at the first index whose absolute difference
///   exceeds `1e-5`, carrying the offending `stage` and `index`.
pub fn compare_stage(stage: Stage, expected: &[f64], actual: &[f64]) -> Result<(), OracleError> {
    match assert_abs_close(expected, actual, 1e-5) {
        Ok(()) => Ok(()),
        Err(OracleError::LengthMismatch { expected, actual }) => {
            Err(OracleError::StageLengthMismatch {
                stage,
                expected,
                actual,
            })
        }
        Err(OracleError::Diverged {
            index,
            expected,
            actual,
            diff,
        }) => Err(OracleError::StageDiverged {
            stage,
            index,
            expected,
            actual,
            diff,
        }),
        // `assert_abs_close` only ever yields LengthMismatch / Diverged; any
        // other variant would be a future addition — propagate it untouched.
        Err(other) => Err(other),
    }
}

/// Integer-exact comparator for the [`Stage::Permutation`] linchpin (Phase-5
/// Wave-0, D-03).
///
/// Permutation indices are integers (`int32`/`i64` on the upstream side), so
/// they are compared with `==`, NOT at the `1e-5` float tolerance. Any single
/// index mismatch is rejected at the FIRST offending position — this is what
/// lets the D-03 ordering hold: a permutation that fails to reproduce upstream
/// exactly must short-circuit before any value stage (OnlineCtr / OrderedApprox)
/// is allowed to run.
///
/// This function never panics and never indexes (mitigates T-01-02): it walks
/// the two slices with a zipped iterator and returns a typed [`OracleError`].
///
/// # Errors
/// - [`OracleError::PermutationLengthMismatch`] if the slices differ in length.
/// - [`OracleError::PermutationDiverged`] at the first index whose value differs.
pub fn compare_permutation(expected: &[i64], actual: &[i64]) -> Result<(), OracleError> {
    if expected.len() != actual.len() {
        return Err(OracleError::PermutationLengthMismatch {
            stage: Stage::Permutation,
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (index, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e != a {
            return Err(OracleError::PermutationDiverged {
                stage: Stage::Permutation,
                index,
                expected: *e,
                actual: *a,
            });
        }
    }
    Ok(())
}
