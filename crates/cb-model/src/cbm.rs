//! Native `.cbm` (FlatBuffers `TModelCore`) save/load with validated, panic-free
//! deserialization (MODEL-01, RESEARCH Pattern 1, Security V5).
//!
//! # `.cbm` blob framing (`[VERIFIED: model.cpp:1113-1228]`)
//!
//! ```text
//! offset 0:  4 bytes  = b"CBM1"  (the model-file descriptor, LE POD)
//! offset 4:  4 bytes  = ui32 LE  = byte length of the FlatBuffers core blob
//! offset 8:  N bytes  = FlatBuffers TModelCore buffer (root_type TModelCore)
//! offset 8+N:          = optional model parts (CTR / text / embedding) — OUT OF SCOPE
//! ```
//!
//! The size field is a FIXED ui32 LE (`util/ysaveload.h:277-296` — NOT a varint);
//! Phase-4 numeric-only models are tiny so the 4-byte form always applies. The
//! `FormatVersion` string inside the `TModelCore` MUST equal `"FlabuffersModel_v1"`
//! EXACTLY — the upstream typo is canonical (`model.cpp:53,1167`, RESEARCH
//! Pitfall 5). The writer emits it; the reader rejects anything else.
//!
//! # Split-index encoding (`[VERIFIED: model.cpp:471-493 CalcBinFeatures]`)
//!
//! Upstream stores tree splits as GLOBAL binary-feature indices into a flat
//! `BinFeatures` list built by iterating `FloatFeatures` in order and pushing one
//! `(featureIndex, border)` entry per border. We mirror that exactly: `save_cbm`
//! maps each canonical `Split { feature, border }` to its global index in that
//! flat list; `load_cbm` rebuilds the flat list from the loaded `FloatFeatures`
//! borders and decodes each `TreeSplits[i]` back into a `Split`. This is the
//! standard upstream wire layout, so upstream `.cbm` files decode unchanged and
//! our own files round-trip.
//!
//! # Leaf layout (RESEARCH Pitfall 2)
//!
//! `LeafValues` / `LeafWeights` are a SINGLE flat `[f64]` array across all trees;
//! each tree's slice starts at `TreeStartOffsets`-derived leaf offset. We compute
//! the per-tree leaf offset as the running sum of `2^treeSize` (the leaf count of
//! every preceding tree), matching `TreeFirstLeafOffsets`.
//!
//! # Validation (Security V5, T-04-03-01..05)
//!
//! `load_cbm` checks the magic with `.get(0..4)` (never an index), BOUNDS the
//! declared ui32 size against the actual remaining file length BEFORE slicing (no
//! huge alloc / OOB), parses with the VERIFYING `root_as_tmodel_core` (never the
//! `_unchecked` accessor — the `flatbuffers` verifier rejects truncated/corrupt
//! buffers and caps depth), and uses checked `u32::try_from` / `.get` throughout.
//! Every failure maps to a typed [`ModelError`]; nothing panics.

use std::path::Path;

use flatbuffers::FlatBufferBuilder;

use crate::error::ModelError;
use crate::model_generated::ncat_boost_fbs::{
    root_as_tmodel_core, TFloatFeature, TFloatFeatureArgs, TModelCore, TModelCoreArgs, TModelTrees,
    TModelTreesArgs,
};
use crate::{Model, ObliviousTree, Split};

/// The `.cbm` model-file descriptor magic (`model.cpp:41,49-51`).
pub const CBM1: &[u8; 4] = b"CBM1";

/// The canonical (typo-preserving) `FormatVersion` upstream writes into every
/// `TModelCore` (`model.cpp:53` — the doubled-`b` typo is authoritative).
pub const FLATBUFFERS_MODEL_V1: &str = "FlabuffersModel_v1";

/// A `(featureIndex, border)` entry of the flat `BinFeatures` list — the wire
/// meaning of one global tree-split index (`CalcBinFeatures`, model.cpp:471-493).
struct BinFeature {
    feature: usize,
    border: f64,
}

/// Build the flat `BinFeatures` list from per-float-feature borders, iterating
/// features in order and pushing one entry per border (upstream
/// `CalcBinFeatures` order). The global split index indexes into this list.
fn build_bin_features(float_feature_borders: &[Vec<f64>]) -> Vec<BinFeature> {
    float_feature_borders
        .iter()
        .enumerate()
        .flat_map(|(feature, borders)| {
            borders.iter().map(move |&border| BinFeature { feature, border })
        })
        .collect()
}

/// Map a canonical [`Split`] to its global bin-feature index against `bins`.
///
/// Matches on the exact `(feature, border)` pair an entry was built from (the
/// borders came from the same `float_feature_borders`, so they compare bit-for-
/// bit). Returns a typed error if the split references a feature/border not in
/// the model's border pool (a malformed in-memory model).
fn split_to_global_index(split: &Split, bins: &[BinFeature]) -> Result<i32, ModelError> {
    let pos = bins
        .iter()
        .position(|b| b.feature == split.feature && b.border.to_bits() == split.border.to_bits())
        .ok_or_else(|| {
            ModelError::Deserialize(format!(
                "split on feature {} border {} has no matching border in float_feature_borders",
                split.feature, split.border
            ))
        })?;
    i32::try_from(pos).map_err(|_| {
        ModelError::SchemaVersion(format!("global split index {pos} exceeds i32 range"))
    })
}

/// Serialize `model` to the native `.cbm` format at `path` (MODEL-01).
///
/// Emits the `CBM1` magic, the ui32 LE core size, and a FlatBuffers `TModelCore`
/// carrying `FormatVersion = "FlabuffersModel_v1"`, the global `TreeSplits`,
/// per-tree `TreeSizes` / `TreeStartOffsets`, the flat `LeafValues` /
/// `LeafWeights` arrays, the `FloatFeatures` borders (as f32, the schema type),
/// `ApproxDimension = 1`, and the single `Bias` (bias-free leaf values, Open Q3 /
/// Pitfall 6).
///
/// # Errors
/// [`ModelError::SchemaVersion`] if the model is too large to address with the
/// ui32 framing (or a split index overflows `i32`); [`ModelError::Deserialize`]
/// if a split references a border absent from the model's border pool;
/// [`ModelError::Io`] on a write failure.
pub fn save_cbm(model: &Model, path: &Path) -> Result<(), ModelError> {
    let core = build_core_blob(model)?;

    let core_len = u32::try_from(core.len())
        .map_err(|_| ModelError::SchemaVersion("core blob exceeds 4 GiB (ui32 size)".to_owned()))?;

    let mut out = Vec::with_capacity(8usize.saturating_add(core.len()));
    out.extend_from_slice(CBM1);
    out.extend_from_slice(&core_len.to_le_bytes());
    out.extend_from_slice(&core);

    std::fs::write(path, out)?;
    Ok(())
}

/// Build the FlatBuffers `TModelCore` payload (the bytes after the 8-byte frame).
fn build_core_blob(model: &Model) -> Result<Vec<u8>, ModelError> {
    let bins = build_bin_features(&model.float_feature_borders);

    // Global tree splits + per-tree sizes/offsets, and the flat leaf arrays.
    let mut tree_splits: Vec<i32> = Vec::new();
    let mut tree_sizes: Vec<i32> = Vec::new();
    let mut tree_start_offsets: Vec<i32> = Vec::new();
    let mut leaf_values: Vec<f64> = Vec::new();
    let mut leaf_weights: Vec<f64> = Vec::new();

    let mut split_offset: i32 = 0;
    for tree in &model.oblivious_trees {
        let size = i32::try_from(tree.splits.len()).map_err(|_| {
            ModelError::SchemaVersion("tree depth exceeds i32 range".to_owned())
        })?;
        tree_start_offsets.push(split_offset);
        tree_sizes.push(size);
        split_offset = split_offset.checked_add(size).ok_or_else(|| {
            ModelError::SchemaVersion("cumulative tree-split offset overflow".to_owned())
        })?;
        for split in &tree.splits {
            tree_splits.push(split_to_global_index(split, &bins)?);
        }
        leaf_values.extend_from_slice(&tree.leaf_values);
        leaf_weights.extend_from_slice(&tree.leaf_weights);
    }

    let mut fbb = FlatBufferBuilder::new();

    // FloatFeatures: one table per float feature carrying its f32 borders. Index
    // and FlatIndex both equal the float-feature index (numeric-only models).
    let mut float_feature_offsets = Vec::with_capacity(model.float_feature_borders.len());
    for (idx, borders) in model.float_feature_borders.iter().enumerate() {
        let borders_f32: Vec<f32> = borders.iter().map(|&b| b as f32).collect();
        let borders_vec = fbb.create_vector(&borders_f32);
        let feature_idx = i32::try_from(idx).map_err(|_| {
            ModelError::SchemaVersion("float-feature index exceeds i32 range".to_owned())
        })?;
        let ff = TFloatFeature::create(
            &mut fbb,
            &TFloatFeatureArgs {
                Index: feature_idx,
                FlatIndex: feature_idx,
                Borders: Some(borders_vec),
                ..TFloatFeatureArgs::default()
            },
        );
        float_feature_offsets.push(ff);
    }
    let float_features = fbb.create_vector(&float_feature_offsets);

    let tree_splits_vec = fbb.create_vector(&tree_splits);
    let tree_sizes_vec = fbb.create_vector(&tree_sizes);
    let tree_start_offsets_vec = fbb.create_vector(&tree_start_offsets);
    let leaf_values_vec = fbb.create_vector(&leaf_values);
    let leaf_weights_vec = fbb.create_vector(&leaf_weights);

    let model_trees = TModelTrees::create(
        &mut fbb,
        &TModelTreesArgs {
            ApproxDimension: 1,
            TreeSplits: Some(tree_splits_vec),
            TreeSizes: Some(tree_sizes_vec),
            TreeStartOffsets: Some(tree_start_offsets_vec),
            FloatFeatures: Some(float_features),
            LeafValues: Some(leaf_values_vec),
            LeafWeights: Some(leaf_weights_vec),
            Scale: 1.0,
            Bias: model.bias,
            ..TModelTreesArgs::default()
        },
    );

    let format_version = fbb.create_string(FLATBUFFERS_MODEL_V1);
    let core = TModelCore::create(
        &mut fbb,
        &TModelCoreArgs {
            FormatVersion: Some(format_version),
            ModelTrees: Some(model_trees),
            ..TModelCoreArgs::default()
        },
    );
    fbb.finish(core, None);

    Ok(fbb.finished_data().to_vec())
}

/// Deserialize a native `.cbm` file at `path` into the canonical [`Model`]
/// (MODEL-01), validating every byte of untrusted input (Security V5).
///
/// # Errors
/// [`ModelError::Deserialize`] on a bad magic, an oversized/short declared size,
/// a corrupt/truncated FlatBuffers buffer, or a missing required table/field;
/// [`ModelError::SchemaVersion`] on a `FormatVersion` other than
/// `FlabuffersModel_v1`; [`ModelError::Io`] if the file cannot be read.
pub fn load_cbm(path: &Path) -> Result<Model, ModelError> {
    let buf = std::fs::read(path)?;
    decode_cbm(&buf)
}

/// Decode an in-memory `.cbm` byte buffer (the validated core of [`load_cbm`],
/// split out so the malformed-input unit tests exercise it directly).
///
/// # Errors
/// As [`load_cbm`] (minus the I/O arm).
pub fn decode_cbm(buf: &[u8]) -> Result<Model, ModelError> {
    // Magic — checked slice, never an index (T-04-03-03).
    if buf.get(0..4) != Some(CBM1.as_slice()) {
        return Err(ModelError::Deserialize(
            "bad .cbm magic (expected CBM1)".to_owned(),
        ));
    }

    // ui32 LE declared core size — checked slice + checked array conversion.
    let size_bytes: [u8; 4] = buf
        .get(4..8)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .ok_or_else(|| ModelError::Deserialize("truncated .cbm size field".to_owned()))?;
    let declared = u32::from_le_bytes(size_bytes) as usize;

    // BOUND the declared size against the actual remaining bytes BEFORE slicing —
    // a declared size larger than the file is rejected, never allocated/over-read
    // (Security V5 / T-04-03-01).
    let core = buf
        .get(8..8usize.saturating_add(declared))
        .ok_or_else(|| {
            ModelError::Deserialize(format!(
                "declared core size {declared} exceeds available {} bytes",
                buf.len().saturating_sub(8)
            ))
        })?;

    // VERIFYING accessor — the flatbuffers verifier rejects truncated/corrupt
    // buffers and caps table depth (T-04-03-02); never the `_unchecked` variant.
    let model_core = root_as_tmodel_core(core)
        .map_err(|e| ModelError::Deserialize(format!("corrupt FlatBuffers TModelCore: {e}")))?;

    // FormatVersion must be the canonical typo'd literal (T-04-03-03).
    match model_core.FormatVersion() {
        Some(FLATBUFFERS_MODEL_V1) => {}
        Some(other) => {
            return Err(ModelError::SchemaVersion(format!(
                "unexpected FormatVersion {other:?} (expected {FLATBUFFERS_MODEL_V1:?})"
            )))
        }
        None => {
            return Err(ModelError::SchemaVersion(
                "missing FormatVersion".to_owned(),
            ))
        }
    }

    let trees = model_core
        .ModelTrees()
        .ok_or_else(|| ModelError::Deserialize("missing ModelTrees".to_owned()))?;

    reconstruct_model(&trees)
}

/// Reconstruct the canonical [`Model`] from a verified `TModelTrees`.
fn reconstruct_model(trees: &TModelTrees) -> Result<Model, ModelError> {
    // Float-feature borders (f32 on the wire -> f64 canonical), in feature order.
    let float_feature_borders = read_float_feature_borders(trees)?;
    let bins = build_bin_features(&float_feature_borders);

    let tree_splits = trees
        .TreeSplits()
        .ok_or_else(|| ModelError::Deserialize("missing TreeSplits".to_owned()))?;
    let tree_sizes = trees
        .TreeSizes()
        .ok_or_else(|| ModelError::Deserialize("missing TreeSizes".to_owned()))?;
    let leaf_values = trees
        .LeafValues()
        .ok_or_else(|| ModelError::Deserialize("missing LeafValues".to_owned()))?;
    // LeafWeights may be absent in older models; default to zeros per leaf.
    let leaf_weights = trees.LeafWeights();

    let mut oblivious_trees = Vec::with_capacity(tree_sizes.len());
    let mut split_cursor: usize = 0;
    let mut leaf_cursor: usize = 0;

    for ti in 0..tree_sizes.len() {
        let size = usize::try_from(tree_sizes.get(ti)).map_err(|_| {
            ModelError::Deserialize("negative tree size".to_owned())
        })?;
        // Decode this tree's global split indices into canonical splits.
        let mut splits = Vec::with_capacity(size);
        for off in 0..size {
            let global = split_cursor.checked_add(off).ok_or_else(|| {
                ModelError::Deserialize("tree-split cursor overflow".to_owned())
            })?;
            if global >= tree_splits.len() {
                return Err(ModelError::Deserialize(
                    "TreeSplits shorter than declared tree sizes".to_owned(),
                ));
            }
            let gidx = usize::try_from(tree_splits.get(global)).map_err(|_| {
                ModelError::Deserialize("negative global split index".to_owned())
            })?;
            let bin = bins.get(gidx).ok_or_else(|| {
                ModelError::Deserialize(format!(
                    "global split index {gidx} out of range (bin features: {})",
                    bins.len()
                ))
            })?;
            splits.push(Split {
                feature: bin.feature,
                border: bin.border,
            });
        }
        split_cursor = split_cursor.saturating_add(size);

        // Leaf slice for this tree: 2^size leaves, flat-array offset per tree.
        let leaf_count = 1usize.checked_shl(u32::try_from(size).map_err(|_| {
            ModelError::Deserialize("tree size exceeds u32".to_owned())
        })?)
        .ok_or_else(|| ModelError::Deserialize("2^depth overflowed usize".to_owned()))?;

        let (tree_values, tree_weights) =
            read_tree_leaves(&leaf_values, leaf_weights.as_ref(), leaf_cursor, leaf_count)?;
        leaf_cursor = leaf_cursor.saturating_add(leaf_count);

        oblivious_trees.push(ObliviousTree {
            splits,
            leaf_values: tree_values,
            leaf_weights: tree_weights,
        });
    }

    Ok(Model {
        oblivious_trees,
        bias: read_bias(trees),
        float_feature_borders,
    })
}

/// Read the 1-dimensional model bias.
///
/// Upstream catboost 1.2.10 stores the single-target bias in the `MultiBias`
/// VECTOR (length `ApproxDimension`), leaving the scalar `Bias` field at its 0.0
/// default — so a regression `.cbm` carries its `boost_from_average` start value
/// only in `MultiBias[0]`. We therefore prefer `MultiBias[0]` when present and
/// fall back to the scalar `Bias` (which is what `save_cbm` writes for the 1-dim
/// case, Open Q3). For multi-dimensional models (Phase 5/6) this returns the
/// first dimension; higher dimensions are added when multiclass lands.
fn read_bias(trees: &TModelTrees) -> f64 {
    trees
        .MultiBias()
        .filter(|mb| !mb.is_empty())
        .map_or_else(|| trees.Bias(), |mb| mb.get(0))
}

/// Read the per-float-feature borders (f32 wire -> f64), preserving feature
/// order and empty inner vectors. Features are placed at their declared `Index`.
fn read_float_feature_borders(trees: &TModelTrees) -> Result<Vec<Vec<f64>>, ModelError> {
    let Some(features) = trees.FloatFeatures() else {
        return Ok(Vec::new());
    };
    // Place each feature at its declared Index so the canonical vector index
    // lines up with the float-feature index even if features are reordered.
    let mut max_index: usize = 0;
    let mut parsed: Vec<(usize, Vec<f64>)> = Vec::with_capacity(features.len());
    for fi in 0..features.len() {
        let ff: TFloatFeature = features.get(fi);
        let index = usize::try_from(ff.Index()).map_err(|_| {
            ModelError::Deserialize("negative float-feature Index".to_owned())
        })?;
        max_index = max_index.max(index);
        let borders: Vec<f64> = ff
            .Borders()
            .map(|b| b.iter().map(f64::from).collect())
            .unwrap_or_default();
        parsed.push((index, borders));
    }
    let mut out: Vec<Vec<f64>> = vec![Vec::new(); max_index.saturating_add(1)];
    for (index, borders) in parsed {
        if let Some(slot) = out.get_mut(index) {
            *slot = borders;
        }
    }
    Ok(out)
}

/// Slice one tree's `leaf_count` values (and weights, zero-filled if absent) out
/// of the flat leaf arrays starting at `offset`, with bounds checks.
fn read_tree_leaves(
    leaf_values: &flatbuffers::Vector<f64>,
    leaf_weights: Option<&flatbuffers::Vector<f64>>,
    offset: usize,
    leaf_count: usize,
) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
    let end = offset.checked_add(leaf_count).ok_or_else(|| {
        ModelError::Deserialize("leaf offset + count overflow".to_owned())
    })?;
    if end > leaf_values.len() {
        return Err(ModelError::Deserialize(
            "LeafValues shorter than declared tree leaves".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(leaf_count);
    let mut weights = Vec::with_capacity(leaf_count);
    for i in offset..end {
        values.push(leaf_values.get(i));
        // Weights are optional: zero-fill when absent or short.
        let w = leaf_weights
            .filter(|lw| i < lw.len())
            .map_or(0.0, |lw| lw.get(i));
        weights.push(w);
    }
    Ok((values, weights))
}
