// Hand-authored prost bindings for the SUBSET of Apple CoreML's
// `Model.proto` / `TreeEnsemble.proto` / `FeatureTypes.proto`
// (`CoreML.Specification`) that the float-only oblivious `TreeEnsembleRegressor`
// exporter (EXPORT-02) needs. Field numbers, enum values, and message shapes
// were extracted LIVE from `coremltools==9.0`'s compiled descriptors and
// confirmed by re-parsing Rust-encoded bytes through coremltools — see
// `.planning/plans/coreml-export/IMPLEMENTATION_NOTES.md` §1.
//
// This mirrors the committed-generated convention used for
// `onnx_generated.rs` / `model_generated.rs` (NOT part of `cargo build`; only
// `prost`'s derive macros are needed at compile time). Protobuf message nesting
// is irrelevant to the wire format — only the field tags matter — so the CoreML
// messages that are technically nested upstream (`TreeEnsembleParameters.TreeNode`,
// `…::EvaluationInfo`) are defined flat here. The subset is regressor-only,
// scalar, float-only; classifier / categorical / multi-array-input messages are
// deliberately omitted.
//
// To extend or re-verify: dump the field tags from
// `coremltools.proto.{Model_pb2,TreeEnsemble_pb2,FeatureTypes_pb2}` and re-run
// the `gen_fixtures.py` cross-parse check.

/// `CoreML.Specification.Model` (subset: only the `treeEnsembleRegressor` arm of
/// the `Type` oneof).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Model {
    #[prost(int32, tag = "1")]
    pub specification_version: i32,
    #[prost(message, optional, tag = "2")]
    pub description: ::core::option::Option<ModelDescription>,
    #[prost(bool, tag = "10")]
    pub is_updatable: bool,
    #[prost(oneof = "model::Type", tags = "302")]
    pub r#type: ::core::option::Option<model::Type>,
}
/// Nested oneof for [`Model`].
pub mod model {
    /// The `Type` oneof — only the regressor arm this exporter emits.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Type {
        #[prost(message, tag = "302")]
        TreeEnsembleRegressor(super::TreeEnsembleRegressor),
    }
}

/// `CoreML.Specification.ModelDescription` (subset).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelDescription {
    #[prost(message, repeated, tag = "1")]
    pub input: ::prost::alloc::vec::Vec<FeatureDescription>,
    #[prost(message, repeated, tag = "10")]
    pub output: ::prost::alloc::vec::Vec<FeatureDescription>,
    #[prost(string, tag = "11")]
    pub predicted_feature_name: ::prost::alloc::string::String,
}

/// `CoreML.Specification.FeatureDescription` (subset).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FeatureDescription {
    #[prost(string, tag = "1")]
    pub name: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub short_description: ::prost::alloc::string::String,
    #[prost(message, optional, tag = "3")]
    pub r#type: ::core::option::Option<FeatureType>,
}

/// `CoreML.Specification.FeatureType` (subset: `doubleType` / `multiArrayType`).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FeatureType {
    #[prost(oneof = "feature_type::Type", tags = "2, 5")]
    pub r#type: ::core::option::Option<feature_type::Type>,
}
/// Nested oneof for [`FeatureType`].
pub mod feature_type {
    /// The `Type` oneof — the two feature kinds this exporter emits.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Type {
        #[prost(message, tag = "2")]
        DoubleType(super::DoubleFeatureType),
        #[prost(message, tag = "5")]
        MultiArrayType(super::ArrayFeatureType),
    }
}

/// `CoreML.Specification.DoubleFeatureType` (empty message).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DoubleFeatureType {}

/// `CoreML.Specification.ArrayFeatureType` (subset: `shape` + `dataType`).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ArrayFeatureType {
    #[prost(int64, repeated, tag = "1")]
    pub shape: ::prost::alloc::vec::Vec<i64>,
    #[prost(enumeration = "ArrayDataType", tag = "2")]
    pub data_type: i32,
}

/// `CoreML.Specification.ArrayFeatureType.ArrayDataType` (subset of values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ArrayDataType {
    InvalidArrayDataType = 0,
    Float32 = 65568,
    Double = 65600,
    Int32 = 131104,
    Float16 = 65552,
}

/// `CoreML.Specification.TreeEnsembleRegressor` (subset).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TreeEnsembleRegressor {
    #[prost(message, optional, tag = "1")]
    pub tree_ensemble: ::core::option::Option<TreeEnsembleParameters>,
    #[prost(enumeration = "TreeEnsemblePostEvaluationTransform", tag = "2")]
    pub post_evaluation_transform: i32,
}

/// `CoreML.Specification.TreeEnsemblePostEvaluationTransform`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum TreeEnsemblePostEvaluationTransform {
    NoTransform = 0,
    ClassificationSoftMax = 1,
    RegressionLogistic = 2,
    ClassificationSoftMaxWithZeroClassReference = 3,
}

/// `CoreML.Specification.TreeEnsembleParameters` (subset).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TreeEnsembleParameters {
    #[prost(message, repeated, tag = "1")]
    pub nodes: ::prost::alloc::vec::Vec<TreeNode>,
    #[prost(uint64, tag = "2")]
    pub num_prediction_dimensions: u64,
    #[prost(double, repeated, tag = "3")]
    pub base_prediction_value: ::prost::alloc::vec::Vec<f64>,
}

/// `CoreML.Specification.TreeEnsembleParameters.TreeNode` (subset). Defined flat
/// (nesting does not affect the wire format).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TreeNode {
    #[prost(uint64, tag = "1")]
    pub tree_id: u64,
    #[prost(uint64, tag = "2")]
    pub node_id: u64,
    #[prost(enumeration = "TreeNodeBehavior", tag = "3")]
    pub node_behavior: i32,
    #[prost(uint64, tag = "10")]
    pub branch_feature_index: u64,
    #[prost(double, tag = "11")]
    pub branch_feature_value: f64,
    #[prost(uint64, tag = "12")]
    pub true_child_node_id: u64,
    #[prost(uint64, tag = "13")]
    pub false_child_node_id: u64,
    #[prost(bool, tag = "14")]
    pub missing_value_tracks_true_child: bool,
    #[prost(message, repeated, tag = "20")]
    pub evaluation_info: ::prost::alloc::vec::Vec<EvaluationInfo>,
}

/// `CoreML.Specification.TreeEnsembleParameters.TreeNode.TreeNodeBehavior`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum TreeNodeBehavior {
    BranchOnValueLessThanEqual = 0,
    BranchOnValueLessThan = 1,
    BranchOnValueGreaterThanEqual = 2,
    BranchOnValueGreaterThan = 3,
    BranchOnValueEqual = 4,
    BranchOnValueNotEqual = 5,
    LeafNode = 6,
}

/// `CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EvaluationInfo {
    #[prost(uint64, tag = "1")]
    pub evaluation_index: u64,
    #[prost(double, tag = "2")]
    pub evaluation_value: f64,
}
