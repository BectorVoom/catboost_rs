//! The public, typed error for the `catboost-rs` facade, derived with
//! [`thiserror`] (RAPI-02 / D-08).
//!
//! # Why no `Clone` / `PartialEq` (D-08 tradeoff)
//!
//! [`CatBoostError`] carries an `#[from] std::io::Error` arm (so `save_*`/`load_*`
//! propagate file failures with `?`) and an `#[from] cb_model::ModelError` arm
//! (whose own `Io`/`Json` arms wrap `io::Error`/`serde_json::Error`). Because
//! `std::io::Error` is neither `Clone` nor `PartialEq`, this enum cannot derive
//! them — exactly the tradeoff `cb-oracle`/`cb-model` already accepted for their
//! I/O error types (`crates/cb-oracle/src/error.rs`, `crates/cb-model/src/error.rs`).
//! It is WHY the internal [`cb_core::CbError`] keeps its `Clone`/`PartialEq`/`Eq`
//! derives: callers that need a comparable error stay on `CbError`; the public
//! surface that must speak file I/O accepts the heavier, non-comparable type.
//!
//! # No panics across the public boundary (Security V5)
//!
//! Every fallible facade method returns `Result<_, CatBoostError>`; training
//! errors arrive via [`CatBoostError::Train`] (`#[from] cb_core::CbError`),
//! (de)serialization errors via [`CatBoostError::Model`]
//! (`#[from] cb_model::ModelError`), file I/O via [`CatBoostError::Io`], and the
//! facade's own boundary checks via the explicit
//! [`CatBoostError::Deserialize`] / [`CatBoostError::SchemaVersion`] /
//! [`CatBoostError::FeatureMismatch`] variants. No `unwrap`/`expect`/`panic`
//! appears on any error path.

/// The public error surfaced by the `catboost-rs` facade
/// ([`crate::CatBoostBuilder`] and [`crate::Model`]).
///
/// New variants may be added as later phases extend the surface; downstream
/// `match`es should remain robust to additional variants.
#[derive(Debug, thiserror::Error)]
pub enum CatBoostError {
    /// A training / core-primitive error from the internal boosting loop
    /// (`cb-train` -> `cb-core`). Converted with `?` via `#[from]`.
    #[error("training error: {0}")]
    Train(#[from] cb_core::CbError),

    /// A model (de)serialization or apply error from `cb-model`
    /// (`load_cbm`/`load_json`/`save_*`). Carries the typed `cb-model` error so
    /// malformed-file failures (bad magic, corrupt FlatBuffers, wrong schema,
    /// bad JSON) surface — never panic — across the public boundary
    /// (T-04-05-01, Security V5). Converted with `?` via `#[from]`.
    #[error("model error: {0}")]
    Model(#[from] cb_model::ModelError),

    /// Underlying file I/O error while saving or loading a model (the facade's
    /// own I/O, distinct from a `cb-model`-wrapped I/O error). Converted with `?`
    /// via `#[from]`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The facade could not decode an input into a model (a boundary failure the
    /// facade detects itself, before delegating). Carries a human-readable
    /// description; surfaced instead of panicking on malformed input
    /// (T-04-05-01, Security V5).
    #[error("malformed model: {0}")]
    Deserialize(String),

    /// The input parsed structurally but declares an unsupported schema /
    /// version (a boundary failure the facade detects itself).
    #[error("unsupported model schema: {0}")]
    SchemaVersion(String),

    /// A prediction / explain call supplied a [`cb_data::Pool`] whose float
    /// feature count does not match what the model expects, which would
    /// otherwise read out-of-range columns (T-04-05-02). Returned as a typed
    /// error so no out-of-bounds access crosses the public boundary.
    #[error("feature mismatch: {0}")]
    FeatureMismatch(String),

    /// A [`crate::Model::partial_dependence`] request was invalid — bad arity
    /// (not 1 or 2 features), an out-of-range or duplicate feature index, or an
    /// empty dataset. Carries the typed `cb-model` [`cb_model::PdpError`].
    /// Converted with `?` via `#[from]`.
    #[error("partial dependence error: {0}")]
    PartialDependence(#[from] cb_model::PdpError),
}
