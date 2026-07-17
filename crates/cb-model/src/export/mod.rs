//! Model export formats (Phase 17: Model Export).
//!
//! A sibling submodule to a future `coreml.rs` (EXPORT-02), mirroring
//! upstream's `catboost/libs/model/model_export/` directory, which co-locates
//! `onnx_helpers.cpp` and `coreml_helpers.cpp` as siblings in one library
//! rather than splitting them across crates (research.md Crate-Placement
//! Decision).

mod onnx;

pub use onnx::{export_onnx, OnnxExportError};
