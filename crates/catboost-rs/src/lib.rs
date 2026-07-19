#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `catboost-rs` — the published Builder-pattern facade (D-04 naming: the single
//! published crate; the five internal `cb-` crates are wrapped here).
//!
//! This crate composes the internal slice — `cb-train` (boosting), `cb-model`
//! (apply / serialize / SHAP / feature importance), `cb-core`/`cb-data`/
//! `cb-compute`/`cb-backend` — into one ergonomic, Rust-native surface:
//!
//! - [`CatBoostBuilder`] (D-05): `new()` + chained setters +
//!   `fit(&pool) -> Result<Model, CatBoostError>`; the `loss` selects
//!   classification vs regression.
//! - [`Model`] (D-06/D-07): `predict` / `predict_proba` / `predict_with`
//!   (enum core), `save_cbm`/`load_cbm`/`save_json`/`load_json`, `shap_values`,
//!   `feature_importance`.
//! - [`CatBoostError`] (D-08 / RAPI-02): the public typed error (`thiserror`,
//!   `#[from] cb_core::CbError`, `#[from] cb_model::ModelError`,
//!   `#[from] std::io::Error`).
//!
//! `anyhow` is intentionally absent (D-14/D-15 structural ban): the published
//! facade is a `thiserror`-only library.

mod builder;
mod error;
mod metrics;
mod model;

pub use builder::CatBoostBuilder;
pub use error::CatBoostError;
pub use metrics::{eval_metric, eval_metrics};
pub use model::Model;

// Re-export the prediction / importance enums so callers drive the facade
// without reaching into the internal crates.
pub use cb_model::{FeatureImportanceType, PredictionType};

// Re-export the partial-dependence result + error types (FSTR-03) so callers
// consume `Model::partial_dependence` entirely through the published crate.
pub use cb_model::{PartialDependence, PdpError};

// Re-export the ONNX export error type (EXPORT-01) so callers can match on
// `catboost_rs::OnnxExportError` sub-variants (via `CatBoostError::Export`)
// entirely through the published crate, mirroring the `PdpError` precedent.
pub use cb_model::OnnxExportError;

// Re-export the CoreML export error type (EXPORT-02) so callers can match on
// `catboost_rs::CoreMlExportError` sub-variants (via `CatBoostError::CoreMlExport`)
// entirely through the published crate, mirroring the `OnnxExportError` precedent.
pub use cb_model::CoreMlExportError;

// Re-export the loss / leaf-method / score-function / bootstrap knobs the
// Builder consumes, so a caller configures a run entirely through the published
// crate. `EScoreFunction` drives `.score_function()` (Cosine = catboost CPU
// default, L2 = variance-reduction alternative).
pub use cb_compute::{EScoreFunction, LeafMethod, Loss};
pub use cb_train::EBootstrapType;

// Re-export the Pool ingestion surface (the `fit`/predict input) from the
// published crate.
pub use cb_data::ingest::{IngestSource, OwnedColumns};
pub use cb_data::Pool;

#[cfg(test)]
mod error_test;
#[cfg(test)]
mod metrics_test;
#[cfg(test)]
mod model_sum_test;
#[cfg(test)]
mod onnx_test;
