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
mod fstr;
mod json;
mod model;
mod predict;
mod shap;

pub use apply::{
    apply_virtual_ensembles, binarize_feature, ctr_value_for_combined_projection,
    ctr_value_for_projection, predict_raw, predict_raw_cat, predict_raw_multi,
    predict_raw_multi_biased,
};
pub use cbm::{decode_cbm, load_cbm, save_cbm, CBM1, FLATBUFFERS_MODEL_V1};
pub use ctr_data::{
    calc_inference, ctr_base_key, decode_ctr_data, encode_ctr_data, CtrData, CtrTableJson,
    CtrValueTable, ECtrType, Prior,
};
pub use error::ModelError;
pub use fstr::{interaction, prediction_values_change, FeatureImportanceType};
pub use json::{decode_json, load_json, save_json};
pub use model::{CtrSplit, Model, ModelSplit, ObliviousTree, Split};
pub use predict::{
    apply_multiclass_prediction, apply_prediction_type, apply_ve_prediction_type, MultiClassKind,
    PredictionType,
};
pub use shap::shap_values;

// flatc --rust generated FlatBuffers bindings for the vendored upstream schema
// (D-01). Generated with `flatc --rust --gen-all` so each committed file is
// SELF-CONTAINED (its schema includes are inlined under the file's own
// `pub mod ncat_boost_fbs` — the `NCatBoostFbs` namespace), with NO cross-file
// `use crate::*_generated::*` references. That makes each file mount as a plain
// `#[path]` module with no extra wiring. The committed files under
// `src/generated/` are unmodified flatc output (NEVER hand-edited, D-01):
//   * `model_generated.rs`    — TModelCore / TModelTrees (LeafValues / LeafWeights
//     / Bias) plus the inlined features + ctr_data + guid schema. This is the
//     authoritative binding the later `.cbm` (de)serializer plan consumes.
//   * `features_generated.rs` — features schema (+ guid), self-contained.
//   * `ctr_data_generated.rs` — ctr_data schema (+ features + guid), self-contained.
//
// `#[allow(...)]` mirrors the generated-FFI exemption pattern in CLAUDE.md: the
// bindings carry non-snake_case fields, unused helpers, and clippy-restricted
// constructs intrinsic to flatc output, not project code.
macro_rules! flatc_module {
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

flatc_module!(model_generated, "generated/model_generated.rs");
flatc_module!(features_generated, "generated/features_generated.rs");
flatc_module!(ctr_data_generated, "generated/ctr_data_generated.rs");
