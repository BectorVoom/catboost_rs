#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-model` — the canonical serializable model plus serialization and SHAP.
//!
//! Phase 4 re-homes the canonical [`Model`] here (RESEARCH Primary
//! Recommendation): it carries the boosting-order [`ObliviousTree`]s (each with
//! `leaf_values` AND `leaf_weights` — RESEARCH Pitfall 1), the model bias, and
//! the per-float-feature borders, so apply / SHAP / serialize need no training
//! pool. [`Split`] is REUSED from `cb-train` (not redefined).
//!
//! The `*_generated` modules hold the `flatc --rust` FlatBuffers bindings for the
//! vendored upstream `model` / `features` / `ctr_data` schema (D-01), consumed by
//! the later `.cbm` (de)serializer plan.

mod apply;
mod cbm;
mod ctr_data;
mod error;
mod export;
mod fstr;
mod json;
mod model;
mod model_sum;
mod partial_dependence;
mod predict;
mod shap;

pub use apply::{
    apply_virtual_ensembles, binarize_feature, collect_leaves_statistics,
    ctr_value_for_combined_projection, ctr_value_for_projection, predict_raw, predict_raw_cat,
    predict_raw_multi, predict_raw_multi_biased, predict_raw_staged,
};
pub use cbm::{decode_cbm, load_cbm, save_cbm, CBM1, FLATBUFFERS_MODEL_V1};
pub use ctr_data::{
    calc_inference, ctr_base_key, decode_ctr_data, decode_ctr_model_parts, encode_ctr_data,
    CtrData, CtrTableJson, CtrValueTable, ECtrType, Prior,
};
pub use error::ModelError;
pub use export::{export_onnx, OnnxExportError};
pub use fstr::{
    interaction, loss_function_change, loss_function_change_logloss, prediction_values_change,
    prediction_values_change_with_data, FeatureImportanceType,
};
pub use json::{decode_json, load_json, save_json};
pub use model::{
    CtrSplit, Model, ModelSplit, NonSymmetricTree, ObliviousTree, RegionLevel, RegionTree, Split,
    TreeVariant,
};
pub use model_sum::sum_models;
pub use partial_dependence::{partial_dependence, PartialDependence, PdpError};
pub use predict::{
    apply_multiclass_prediction, apply_prediction_type, apply_ve_prediction_type, MultiClassKind,
    PredictionType,
};
pub use shap::{prediction_diff, sage_values, shap_interaction_values, shap_values};

// Generated-bindings mount (D-01): a committed, `#[path]`-mounted module with a
// blanket lint-exemption, reused for EVERY vendored code-generator this crate
// consumes — `flatc --rust` (FlatBuffers, the original D-01 case) AND
// `prost-build`/`protox` (ONNX protobuf, EXPORT-01 T0). Both generators emit
// self-contained files (no cross-file `use crate::*_generated::*` references),
// so one macro shape covers both: a plain `#[path]` module with the SAME
// lint-exemption need (non-idiomatic generated names, clippy-restricted
// constructs intrinsic to generator output, not project code). The committed
// files under `src/generated/` are UNMODIFIED generator output (NEVER
// hand-edited, D-01):
//   * `model_generated.rs`    — TModelCore / TModelTrees (LeafValues / LeafWeights
//     / Bias) plus the inlined features + ctr_data + guid schema (flatc). This is
//     the authoritative binding the `.cbm` (de)serializer consumes.
//   * `features_generated.rs` — features schema (+ guid), self-contained (flatc).
//   * `ctr_data_generated.rs` — ctr_data schema (+ features + guid), self-contained
//     (flatc).
//   * `onnx_generated.rs`     — `prost`-generated bindings for the OFFICIAL ONNX
//     project's `onnx.proto3` schema, pinned to a tagged release (see the file's
//     own header comment for the exact tag/commit). Consumed by `export::onnx`
//     (EXPORT-01).
macro_rules! generated_module {
    ($modname:ident, $file:literal) => {
        #[allow(
            clippy::all,
            clippy::pedantic,
            clippy::nursery,
            clippy::restriction,
            non_snake_case,
            non_camel_case_types,
            non_upper_case_globals,
            unused_imports,
            dead_code,
            missing_docs,
            unsafe_op_in_unsafe_fn
        )]
        #[path = $file]
        pub mod $modname;
    };
}

generated_module!(model_generated, "generated/model_generated.rs");
generated_module!(features_generated, "generated/features_generated.rs");
generated_module!(ctr_data_generated, "generated/ctr_data_generated.rs");
generated_module!(onnx_generated, "generated/onnx_generated.rs");
