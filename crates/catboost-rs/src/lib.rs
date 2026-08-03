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
mod cv;
mod error;
mod grid_search;
mod metrics;
mod model;

pub use builder::{CatBoostBuilder, FitResult};
pub use cv::{cv, make_cv_folds, CvFold, CvResult, CvType};
pub use error::CatBoostError;
pub use grid_search::{
    grid_search, metric_is_max_optimal, randomized_search, ErrorScore, SearchResult,
};
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
// The categorical / CTR knobs `CatBoostBuilder::simple_ctr`,
// `.combinations_ctr` and `.counter_calc_method` take (F07). Without these
// re-exports an external crate could name the setters but not their argument
// types, so a caller could not configure a categorical run through the
// published crate alone.
pub use cb_train::{CounterCalcMethod, ECtrType};

// PARAM-01: the training knobs the Builder's new setters take. Without these
// re-exports an external crate could name `grow_policy` / `od_type` /
// `boosting_type` / `eval_metric` but not their argument types, so the params
// would be un-configurable through the published crate alone (the same reasoning
// that motivated the CTR re-exports above).
//
// `parse_metric` is re-exported alongside `EvalMetric` because the string form
// (`"AUC"`, `"Quantile:alpha=0.9"`) is how upstream names a metric — without it a
// caller would have to reconstruct the parametric variants by hand.
pub use cb_train::{
    parse_metric, EBoostingType, EGrowPolicy, EOverfittingDetectorType, EvalMetric,
};

// Re-export the Pool ingestion surface (the `fit`/predict input) from the
// published crate.
// PARAM-03: the class-weight scheme selector `CatBoostBuilder::auto_class_weights`
// takes. Re-exported for the same reason as the CTR enums: without it a caller
// could name the setter but not its argument type.
pub use cb_data::AutoClassWeights;
pub use cb_data::ingest::{IngestSource, OwnedColumns};
pub use cb_data::Pool;

#[cfg(test)]
mod error_test;
#[cfg(test)]
mod grid_search_test;
#[cfg(test)]
mod metrics_test;
#[cfg(test)]
mod model_device_test;
#[cfg(test)]
mod model_sum_test;
#[cfg(test)]
mod onnx_test;
