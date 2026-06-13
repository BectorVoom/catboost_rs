//! Ingestion seam (D-04): the boundary through which a [`crate::Pool`] is built.
//!
//! [`IngestSource`] is the trait every dataset source implements to yield a
//! validated [`crate::Pool`]. Phase 2 ships exactly one implementation,
//! [`OwnedColumns`] (owned `Vec` inputs — the trivial primitive used by the
//! Builder API and `.npy` fixtures). At Phase 8 a borrowed / zero-copy view
//! plugs into the same seam by adding another `impl IngestSource`, without
//! touching `Pool` (D-02).
//!
//! All validation (column-length consistency) happens here and returns a typed
//! [`cb_core::CbResult`] — never a panic, never an out-of-bounds index
//! (threats T-02-04 / T-02-05).

use cb_core::CbResult;

use crate::Pool;

mod owned;

pub use owned::OwnedColumns;

#[cfg(test)]
mod owned_test;

/// A source of dataset columns that can be validated and materialized into a
/// [`Pool`].
///
/// Implementors are responsible for checking that every supplied column shares
/// the same object count before constructing the `Pool`; on any mismatch they
/// return a typed error rather than panicking.
pub trait IngestSource {
    /// Validate the source's columns and materialize them into an owned
    /// [`Pool`].
    ///
    /// # Errors
    ///
    /// Returns [`cb_core::CbError::OutOfRange`] when the supplied columns are not
    /// all the same length (a shape mismatch).
    fn into_pool(self) -> CbResult<Pool>;
}
