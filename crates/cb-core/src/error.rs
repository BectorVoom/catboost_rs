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

    /// An external (Arrow / Polars) column had an unsupported logical type.
    ///
    /// Raised at the ingestion boundary when a column's `data_type()` is not one
    /// of the supported dtypes (e.g. a non-`Float64` numeric feature column).
    #[error("unsupported dtype: expected {expected}, got {got}")]
    Dtype {
        /// The dtype the ingestion path requires (e.g. `"Float64"`).
        expected: &'static str,
        /// The dtype actually present on the column (external dtype, stringified).
        got: String,
    },

    /// Two ingested columns disagreed on object count (`n_rows`).
    ///
    /// Raised at the ingestion boundary when a named column's length does not
    /// match the dataset's reference object count.
    #[error("column `{column}` has length {actual}, expected {expected} (n_rows)")]
    LengthMismatch {
        /// Name of the offending column.
        column: String,
        /// The reference object count every column must match.
        expected: usize,
        /// The offending column's actual length.
        actual: usize,
    },

    /// A `NaN` value appeared in a column declared categorical.
    ///
    /// Categorical columns must never carry `NaN`; smuggling one in is rejected
    /// at the ingestion boundary rather than being silently hashed (threat
    /// T-02-14).
    #[error("NaN in categorical column {column}")]
    NanInCategorical {
        /// Index of the offending categorical column.
        column: usize,
    },

    /// A requested tree `depth` exceeded the supported cap (upstream
    /// `MaxDepth`). `2^depth` leaf counts are allocated up front, so an
    /// oversized depth is rejected before allocation rather than overflowing
    /// (Phase-3 T-03-01-02 mitigation).
    #[error("tree depth {depth} exceeds the maximum supported depth {max}")]
    DepthExceeded {
        /// The requested depth.
        depth: usize,
        /// The maximum supported depth (upstream `MaxDepth`, 16).
        max: usize,
    },

    /// A training step hit a degenerate condition that cannot produce a valid
    /// tree (e.g. no candidate split improves the score, or an input dimension
    /// is empty). Surfaced as an error rather than a panic (Phase-3
    /// T-03-01-01 mitigation).
    #[error("degenerate training input: {0}")]
    Degenerate(String),

    /// An external (Arrow / Polars) source failed to yield a usable column.
    ///
    /// The external error is STRINGIFIED into `message` rather than wrapped via
    /// `#[from]`, so [`CbError`] keeps its `Clone` / `PartialEq` / `Eq` derives
    /// (02-PATTERNS.md Shared Pattern C). Covers non-contiguous chunked data,
    /// `to_arrow` / `rechunk` failures, and similar boundary conditions.
    #[error("ingestion error: {message}")]
    Ingestion {
        /// The stringified external (Arrow / Polars) error.
        message: String,
    },
}
