//! GINF-01-S1/S2 — pure, backend-free GPU-apply guard + flattener.
//!
//! **MODEL-02 boundary:** this module imports NOTHING from the GPU compute
//! backend crate and NOTHING from the kernel DSL — it is pure `Model` inspection
//! (mirroring `export/onnx.rs`), so it compiles and runs on a machine with no GPU
//! dependency present. The device edge (kernel + launch host helper) lives in the
//! backend crate; the marshalling `predict_raw_on_device` lives in the
//! `catboost-rs` facade.

use cb_core::{CbError, CbResult};

use crate::model::{Model, ModelSplit};

/// Typed failure at the GPU-apply boundary (mirrors `OnnxExportError`): the
/// first disqualifying property of a model the device apply first slice cannot
/// evaluate.
#[derive(Debug, thiserror::Error)]
pub enum GpuApplyUnsupported {
    /// The model uses categorical / CTR features (a `ModelSplit::Ctr` split or
    /// baked `ctr_data`), which need a variable-size hash-table lookup.
    #[error("model uses categorical/CTR features, unsupported by the GPU apply first slice")]
    CategoricalFeatures,
    /// The model contains at least one one-hot categorical split (SPEC-OH-17).
    /// A DISTINCT variant from [`GpuApplyUnsupported::CategoricalFeatures`]: a
    /// one-hot split needs the RAW categorical column, which the device apply
    /// path (float bins only) never uploads. Named explicitly so the failure
    /// does not fall through into `flatten_oblivious_f64`'s
    /// "unexpected … in a guard-passed model" message.
    #[error("model contains one-hot categorical splits, unsupported by the GPU apply first slice")]
    OneHotSplits,
    /// The model contains non-symmetric (Lossguide / Depthwise) trees, which use
    /// a separate pointer-walk apply path.
    #[error("model contains non-symmetric (Lossguide/Depthwise) trees, unsupported")]
    NonObliviousTrees,
    /// The model contains region-path trees, which use a walk-until-diverge apply
    /// path.
    #[error("model contains region-path trees, unsupported")]
    RegionTrees,
    /// The model is multi-dimensional (`approx_dimension > 1`), which needs a
    /// dimension-major leaf gather.
    #[error("model is multi-dimensional (approx_dimension > 1), unsupported")]
    MultiDimensional,
}

/// GINF-01-S1: admit ONLY a float-only, oblivious, scalar model. Deterministic
/// check order (mirroring [`crate::export`]'s `is_onnx_exportable`):
/// non-symmetric → region → one-hot → CTR → multi-dim → `Ok(())`. Pure; no I/O;
/// total.
///
/// # Errors
/// A [`GpuApplyUnsupported`] variant naming the first disqualifying property.
pub fn check_gpu_apply_supported(model: &Model) -> Result<(), GpuApplyUnsupported> {
    if !model.non_symmetric_trees.is_empty() {
        return Err(GpuApplyUnsupported::NonObliviousTrees);
    }
    if !model.region_trees.is_empty() {
        return Err(GpuApplyUnsupported::RegionTrees);
    }
    // SPEC-OH-17: BEFORE the generic categorical arm, so the message is specific.
    let has_one_hot_split = model
        .oblivious_trees
        .iter()
        .flat_map(|tree| tree.splits.iter())
        .any(|split| matches!(split, ModelSplit::OneHot(_)));
    if has_one_hot_split {
        return Err(GpuApplyUnsupported::OneHotSplits);
    }
    let has_ctr_split = model
        .oblivious_trees
        .iter()
        .flat_map(|tree| tree.splits.iter())
        .any(|split| matches!(split, ModelSplit::Ctr(_)));
    if model.ctr_data.is_some() || has_ctr_split {
        return Err(GpuApplyUnsupported::CategoricalFeatures);
    }
    if model.approx_dimension > 1 {
        return Err(GpuApplyUnsupported::MultiDimensional);
    }
    Ok(())
}

/// The device-ready flat lowering of a float-only oblivious scalar model. All
/// per-object-invariant model state, uploaded ONCE per apply call.
///
/// The number of trees is `tree_split_offsets.len() - 1` (equivalently
/// `tree_leaf_offsets.len() - 1`); both offset arrays are CSR-style and have
/// length `n_trees + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatObliviousF64 {
    /// Concatenated split FLOAT-feature indices across all trees (u32 device index).
    pub split_features: Vec<u32>,
    /// Concatenated split borders (f64), 1:1 with `split_features`.
    pub split_borders: Vec<f64>,
    /// Per-tree start offset into `split_features` / `split_borders`; length
    /// `n_trees + 1` (CSR-style; tree `t` owns
    /// `[tree_split_offsets[t], tree_split_offsets[t + 1])`).
    pub tree_split_offsets: Vec<u32>,
    /// Concatenated leaf values across all trees (f64), already
    /// `learning_rate`-scaled.
    pub leaf_values: Vec<f64>,
    /// Per-tree start offset into `leaf_values`; length `n_trees + 1`.
    pub tree_leaf_offsets: Vec<u32>,
    /// The model bias, added exactly once per object.
    pub bias: f64,
}

/// GINF-01-S2: lower a supported model into [`FlatObliviousF64`]. Calls
/// [`check_gpu_apply_supported`] first, then concatenates each tree's float
/// splits + leaf values with CSR-style per-tree offsets, checked-casting indices
/// / offsets to `u32`.
///
/// # Errors
/// - [`CbError::Unsupported`] carrying the guard rejection (a
///   [`GpuApplyUnsupported`] converted via its `Display`) — NOT `OutOfRange`,
///   which is reserved for the index-overflow case.
/// - [`CbError::OutOfRange`] if a concatenated feature index or CSR offset
///   exceeds `u32::MAX` (mirroring `PackedCindex::device_arrays`).
///
/// Never panics.
pub fn flatten_oblivious_f64(model: &Model) -> CbResult<FlatObliviousF64> {
    check_gpu_apply_supported(model).map_err(|e| CbError::Unsupported(e.to_string()))?;

    let n_trees = model.oblivious_trees.len();
    let mut split_features: Vec<u32> = Vec::new();
    let mut split_borders: Vec<f64> = Vec::new();
    let mut leaf_values: Vec<f64> = Vec::new();
    let mut tree_split_offsets: Vec<u32> = Vec::with_capacity(n_trees + 1);
    let mut tree_leaf_offsets: Vec<u32> = Vec::with_capacity(n_trees + 1);

    // CSR base offset for the first tree.
    tree_split_offsets.push(0);
    tree_leaf_offsets.push(0);

    for tree in &model.oblivious_trees {
        for split in &tree.splits {
            match split {
                ModelSplit::Float(s) => {
                    let feature = u32::try_from(s.feature).map_err(|_| {
                        CbError::OutOfRange(format!(
                            "split feature index {} exceeds u32::MAX",
                            s.feature
                        ))
                    })?;
                    split_features.push(feature);
                    split_borders.push(s.border);
                }
                // Unreachable in practice: the guard above rejects any CTR or
                // one-hot split. A typed error (never a panic) keeps this total
                // if the guard and this match ever drift.
                ModelSplit::Ctr(_) => {
                    return Err(CbError::Unsupported(
                        "unexpected CTR split in a guard-passed model".to_string(),
                    ));
                }
                ModelSplit::OneHot(_) => {
                    return Err(CbError::Unsupported(
                        "unexpected one-hot split in a guard-passed model".to_string(),
                    ));
                }
            }
        }
        leaf_values.extend_from_slice(&tree.leaf_values);

        // Monotonic CSR offsets: the running concatenated lengths after this tree.
        let split_off = u32::try_from(split_features.len()).map_err(|_| {
            CbError::OutOfRange(format!(
                "concatenated split count {} exceeds u32::MAX",
                split_features.len()
            ))
        })?;
        let leaf_off = u32::try_from(leaf_values.len()).map_err(|_| {
            CbError::OutOfRange(format!(
                "concatenated leaf count {} exceeds u32::MAX",
                leaf_values.len()
            ))
        })?;
        tree_split_offsets.push(split_off);
        tree_leaf_offsets.push(leaf_off);
    }

    Ok(FlatObliviousF64 {
        split_features,
        split_borders,
        tree_split_offsets,
        leaf_values,
        tree_leaf_offsets,
        bias: model.bias,
    })
}

#[cfg(test)]
#[path = "gpu_apply_test.rs"]
mod tests;
