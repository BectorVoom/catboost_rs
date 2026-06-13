//! Shared library error type for catboost-rs, derived with [`thiserror`].
//!
//! Modernizes the vendored result-alias idiom
//! (`catboost-master/catboost/rust-package/src/error.rs`) to a `thiserror`-
//! derived enum: no hand-rolled `impl Display`/`impl Error`, no `unwrap()`
//! (D-15 / CLAUDE.md). Variants here are deliberately minimal; later plans
//! extend the enum (e.g. Plan 02's RNG returns these from fallible APIs).

/// Convenient `Result` alias used across catboost-rs library crates.
pub type CbResult<T> = std::result::Result<T, CbError>;

/// Errors surfaced by catboost-rs core primitives.
///
/// New variants may be added as later plans land; downstream `match`es should
/// remain robust to additional variants.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CbError {
    /// A uniform-sampling bound (or similar exclusive upper bound) was not
    /// strictly positive. Reserved for Plan 02's `TFastRng64::uniform`.
    #[error("uniform bound must be > 0, got {bound}")]
    InvalidBound {
        /// The offending bound value.
        bound: u64,
    },

    /// A value violated a precondition / fell outside its valid range.
    #[error("value out of range: {0}")]
    OutOfRange(String),
}
