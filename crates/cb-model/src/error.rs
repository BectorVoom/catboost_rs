//! The `cb-model` (de)serialization error type, derived with [`thiserror`].
//!
//! Mirrors the `cb-oracle` error tradeoff (`crates/cb-oracle/src/error.rs`): an
//! `#[from] std::io::Error` arm makes the loaders propagate file-read failures
//! with `?`, which drops the auto-derivable `Clone`/`PartialEq`/`Eq` (an
//! `io::Error` is neither) — accepted, because the (de)serializers are I/O paths,
//! not value types that need comparison.
//!
//! Every malformed-input failure surfaces a typed variant — bad magic / corrupt
//! FlatBuffers / out-of-range offset map to [`ModelError::Deserialize`]; a wrong
//! `FormatVersion` (or a > 4 GiB core) maps to [`ModelError::SchemaVersion`] — so
//! `load_cbm` / `load_json` NEVER panic on hostile input (Security V5,
//! T-04-03-01..05). No `unwrap`/`expect`/raw-index lives in the production path;
//! the workspace deny-lints stay satisfied.

/// Errors surfaced by the `cb-model` `.cbm` / `model.json` (de)serializers.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The input could not be decoded into a [`crate::Model`]: bad `.cbm` magic,
    /// a declared size that overruns the file, a corrupt/truncated FlatBuffers
    /// buffer, a missing required table/field, or malformed JSON. Carries a
    /// human-readable description; surfaced instead of panicking on hostile
    /// input (Security V5, T-04-03-01/02/05).
    #[error("malformed model: {0}")]
    Deserialize(String),

    /// The input parsed structurally but declares an unsupported schema: a
    /// `FormatVersion` other than `FlabuffersModel_v1`, or a core blob larger
    /// than the 4 GiB the ui32 framing size can address (T-04-03-03).
    #[error("unsupported model schema: {0}")]
    SchemaVersion(String),

    /// The in-memory [`crate::Model`] cannot be represented in the target wire
    /// schema without data loss: e.g. a Region tree carrying a non-float split
    /// level, which the numeric `model.json` schema cannot round-trip (WR-03).
    /// Surfaced loudly instead of emitting a document whose level count desyncs
    /// from `leaf_values`.
    #[error("model cannot be serialized without data loss: {0}")]
    Serialize(String),

    /// Failed to (de)serialize JSON via `serde_json`.
    #[error("model.json (de)serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// A `cb-core` primitive error propagated through model construction.
    #[error(transparent)]
    Core(#[from] cb_core::CbError),

    /// Underlying I/O error while reading or writing a model file.
    #[error("model I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Two or more models cannot be merged: empty input, weight/model count
    /// mismatch, an unsupported model kind (CTR / non-oblivious), or
    /// incompatible feature/output structure. Surfaced loudly instead of
    /// emitting a wrong-valued merged model.
    #[error("models cannot be merged: {0}")]
    Merge(String),
}
