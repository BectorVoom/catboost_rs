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

use std::collections::BTreeMap;
use std::path::Path;

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::ctr_data::{ctr_base_key, decode_ctr_model_parts, encode_ctr_model_parts, ECtrType, Prior};
use crate::error::ModelError;
use crate::model_generated::ncat_boost_fbs::{
    root_as_tmodel_core, ECtrType as CoreECtrType, TCtrFeature, TCtrFeatureArgs,
    TFeatureCombination, TFeatureCombinationArgs, TFloatFeature, TFloatFeatureArgs, TKeyValue,
    TKeyValueArgs, TModelCore, TModelCoreArgs, TModelCtr, TModelCtrArgs, TModelCtrBase,
    TModelCtrBaseArgs, TModelTrees, TModelTreesArgs, TNonSymmetricTreeStepNode,
};
use crate::{CtrSplit, Model, ModelSplit, NonSymmetricTree, ObliviousTree, Split};

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file), mirroring the
// `ctr_data.rs` mount discipline.
#[cfg(test)]
#[path = "cbm_test.rs"]
mod tests;

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

/// A unique CTR feature IDENTITY (the SAVE-side inverse of the load path's
/// per-`CtrSplit` grouping, [`ctr_split_from`] cbm.rs:199): everything a
/// `TCtrFeature` carries EXCEPT the border — `(projection, ctr_type, prior,
/// target_border_idx, shift, scale)` — plus the SORTED-ASCENDING distinct border
/// set the trees test against it. The load path rebuilds one [`CtrSplit`] per
/// `(ctr_feature c, border_index k)`, so one identity + its `k`-th border
/// reconstructs exactly one `CtrSplit`.
#[derive(Debug, Clone, PartialEq)]
struct CtrIdentity {
    /// The combined categorical projection (sorted cat-feature member set).
    projection: cb_train::TProjection,
    /// The CTR type this feature computes.
    ctr_type: ECtrType,
    /// The apply prior `(num, denom)`.
    prior: Prior,
    /// The Buckets per-class numerator selector.
    target_border_idx: usize,
    /// The inference `Shift`.
    shift: f64,
    /// The inference `Scale`.
    scale: f64,
    /// The distinct borders SORTED ASCENDING (`f64::total_cmp`, deduped by bits)
    /// — the wire `Borders` order the load path reads `Borders[k]` back from.
    borders: Vec<f64>,
}

/// A CTR identity's grouping key — everything a `TCtrFeature` carries with the
/// `border` folded OUT (fact 2). `f64` fields compare by their raw bits so a
/// grouping is exact (never an epsilon collapse). A `BTreeMap` over this key
/// gives a deterministic identity order for reproducible round-trips.
type CtrIdentityKey = (String, u64, u64, usize, u64, u64);

/// The stable grouping key of a [`CtrSplit`]: `(ctr_base_key, prior.num bits,
/// prior.denom bits, target_border_idx, shift bits, scale bits)`.
fn ctr_identity_key(split: &CtrSplit) -> CtrIdentityKey {
    (
        ctr_base_key(split.ctr_type, split.projection.cat_features()),
        split.prior.num.to_bits(),
        split.prior.denom.to_bits(),
        split.target_border_idx,
        split.shift.to_bits(),
        split.scale.to_bits(),
    )
}

/// The ordered `CtrFeatures` plan a save emits: the distinct CTR identities in a
/// deterministic order, a `(CtrSplit identity) -> ctr_feature` lookup, and the
/// cumulative preceding-border offsets used to map a `CtrSplit` to its combined
/// global split index (`ctr_split_to_global_index`).
struct CtrFeaturePlan {
    /// The distinct CTR identities, in `index_by_key` (BTreeMap) order.
    identities: Vec<CtrIdentity>,
    /// Identity key -> its index into `identities` (the `ctr_feature` c).
    index_by_key: BTreeMap<CtrIdentityKey, usize>,
    /// `offsets[c]` = Σ `borders.len()` of `identities[0..c]` — the preceding CTR
    /// bins ahead of identity `c` in the combined `CtrFeatures` index range.
    offsets: Vec<usize>,
}

/// Group a model's tree `ModelSplit::Ctr` splits into ordered CTR identities
/// (T1, the inverse of [`build_combined_bins`]'s CTR walk). Each distinct
/// `(projection, ctr_type, prior, target_border_idx, shift, scale)` becomes one
/// identity carrying the SORTED-ASCENDING distinct set of borders the trees test
/// against it; the returned plan maps each tree `CtrSplit` back to its
/// `(ctr_feature, border_index)`. v1 groups OBLIVIOUS trees only (CTR splits on
/// non-symmetric trees are rejected in the save loop).
fn build_ctr_features(model: &Model) -> Result<CtrFeaturePlan, ModelError> {
    // Collect one identity per distinct key, folding `border` OUT of the key and
    // accumulating each split's border under it. `BTreeMap` keys give the
    // deterministic identity order.
    let mut grouped: BTreeMap<CtrIdentityKey, CtrIdentity> = BTreeMap::new();
    for tree in &model.oblivious_trees {
        for split in &tree.splits {
            if let ModelSplit::Ctr(cs) = split {
                grouped
                    .entry(ctr_identity_key(cs))
                    .or_insert_with(|| CtrIdentity {
                        projection: cs.projection.clone(),
                        ctr_type: cs.ctr_type,
                        prior: cs.prior,
                        target_border_idx: cs.target_border_idx,
                        shift: cs.shift,
                        scale: cs.scale,
                        borders: Vec::new(),
                    })
                    .borders
                    .push(cs.border);
            }
        }
    }

    let mut identities = Vec::with_capacity(grouped.len());
    let mut index_by_key = BTreeMap::new();
    for (i, (key, mut identity)) in grouped.into_iter().enumerate() {
        // Sort ascending + dedup by bits: the wire `Borders` the load path reads
        // `Borders[k]` from, matching `build_combined_bins`'s `border_index` walk.
        identity.borders.sort_by(f64::total_cmp);
        identity.borders.dedup_by_key(|b| b.to_bits());
        index_by_key.insert(key, i);
        identities.push(identity);
    }

    // Cumulative preceding-border offsets within the CTR range (v1 has no one-hot
    // bins, so the CTR range starts at `n_float_bins`, fact 1).
    let mut offsets = Vec::with_capacity(identities.len());
    let mut acc: usize = 0;
    for identity in &identities {
        offsets.push(acc);
        acc = acc.saturating_add(identity.borders.len());
    }

    Ok(CtrFeaturePlan {
        identities,
        index_by_key,
        offsets,
    })
}

/// Map a tree [`CtrSplit`] to its combined GLOBAL split index (T1):
/// `n_float_bins + Σ(preceding Borders.len()) + border_index` (fact 1). The
/// identity is located via its grouping key; the border index is its position in
/// that identity's sorted `borders`. A split absent from the plan (or a
/// border/offset out of range) is a typed error rather than a silent mis-index.
fn ctr_split_to_global_index(
    split: &CtrSplit,
    n_float_bins: usize,
    plan: &CtrFeaturePlan,
) -> Result<i32, ModelError> {
    let &c = plan.index_by_key.get(&ctr_identity_key(split)).ok_or_else(|| {
        ModelError::Serialize("CTR split identity missing from the CtrFeatures plan".to_owned())
    })?;
    let identity = plan
        .identities
        .get(c)
        .ok_or_else(|| ModelError::Serialize("CtrFeatures plan index out of range".to_owned()))?;
    let border_index = identity
        .borders
        .iter()
        .position(|b| b.to_bits() == split.border.to_bits())
        .ok_or_else(|| {
            ModelError::Serialize("CTR split border missing from its identity".to_owned())
        })?;
    let offset = plan
        .offsets
        .get(c)
        .copied()
        .ok_or_else(|| ModelError::Serialize("CtrFeatures plan offset out of range".to_owned()))?;
    let global = n_float_bins
        .checked_add(offset)
        .and_then(|g| g.checked_add(border_index))
        .ok_or_else(|| ModelError::Serialize("CTR global split index overflow".to_owned()))?;
    i32::try_from(global).map_err(|_| {
        ModelError::SchemaVersion(format!("CTR global split index {global} exceeds i32 range"))
    })
}

/// Build one `TCtrFeature` FlatBuffers table from a [`CtrIdentity`] (T2, the exact
/// inverse of [`ctr_split_from`]): `Ctr = TModelCtr{ Base = TModelCtrBase{
/// FeatureCombination.CatFeatures = projection, CtrType }, TargetBorderIdx,
/// PriorNum/PriorDenom, Shift, Scale }`, `Borders = sorted borders as f32`. The
/// `f32` casts on the value fields mirror the existing `FloatFeatures` border
/// write (`build_core_blob`) — the load path reads them back via `f64::from`.
fn build_tctr_feature<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    identity: &CtrIdentity,
) -> Result<WIPOffset<TCtrFeature<'a>>, ModelError> {
    let cats: Vec<i32> = identity
        .projection
        .cat_features()
        .iter()
        .map(|&f| {
            i32::try_from(f).map_err(|_| {
                ModelError::SchemaVersion(format!("cat-feature index {f} exceeds i32 range"))
            })
        })
        .collect::<Result<Vec<i32>, ModelError>>()?;
    let cat_vec = fbb.create_vector(&cats);
    let combination = TFeatureCombination::create(
        fbb,
        &TFeatureCombinationArgs {
            CatFeatures: Some(cat_vec),
            ..TFeatureCombinationArgs::default()
        },
    );
    // `model_generated`'s transparent `ECtrType(i8)` (a SEPARATE schema module
    // from `ctr_data_generated`'s copy) — the SAME discriminant `ctr_split_from`
    // reads back via `.0` (MINOR-1).
    let base = TModelCtrBase::create(
        fbb,
        &TModelCtrBaseArgs {
            FeatureCombination: Some(combination),
            CtrType: CoreECtrType(identity.ctr_type.as_i8()),
            TargetBorderClassifierIdx: 0,
        },
    );
    let target_border_idx = i32::try_from(identity.target_border_idx).map_err(|_| {
        ModelError::SchemaVersion("TargetBorderIdx exceeds i32 range".to_owned())
    })?;
    let ctr = TModelCtr::create(
        fbb,
        &TModelCtrArgs {
            Base: Some(base),
            TargetBorderIdx: target_border_idx,
            PriorNum: identity.prior.num as f32,
            PriorDenom: identity.prior.denom as f32,
            Shift: identity.shift as f32,
            Scale: identity.scale as f32,
        },
    );
    let borders_f32: Vec<f32> = identity.borders.iter().map(|&b| b as f32).collect();
    let borders_vec = fbb.create_vector(&borders_f32);
    Ok(TCtrFeature::create(
        fbb,
        &TCtrFeatureArgs {
            Ctr: Some(ctr),
            Borders: Some(borders_vec),
        },
    ))
}

/// One classified entry of the FULL combined `FloatFeatures -> OneHotFeatures
/// -> CtrFeatures` bin table (CTR-01) — the wire meaning of one GLOBAL
/// tree-split index once CTR/one-hot features are accounted for, extending
/// the float-only [`BinFeature`] the SAVE path still uses.
#[derive(Debug, Clone, PartialEq)]
enum BinKind {
    /// A float `value > border` threshold bin (byte-identical to the
    /// pre-CTR-load decode).
    Float {
        /// The float-feature index.
        feature: usize,
        /// The border threshold.
        border: f64,
    },
    /// A one-hot equality bin. COUNTED for the offset (upstream always
    /// includes `OneHotFeatures` bins in the combined index space) but not
    /// representable as a [`crate::ModelSplit`] in v1 (SPEC §2/CTR-05) — a
    /// *tree split* referencing this range is a typed error.
    OneHot,
    /// A CTR bin: the `border_index`-th border of `CtrFeatures[ctr_feature]`.
    Ctr {
        /// Index into `TModelTrees.CtrFeatures`.
        ctr_feature: usize,
        /// Index into that feature's `Borders` vector.
        border_index: usize,
    },
}

/// Build the FULL combined bin-feature table in upstream order —
/// `FloatFeatures` bins, then `OneHotFeatures` bins, then `CtrFeatures` bins
/// (CTR-01, `model.cpp:471-493 CalcBinFeatures`). Every `TreeSplits[i]` GLOBAL
/// index classifies against this table by range. The float prefix is
/// BYTE-IDENTICAL to [`build_bin_features`] (same `read_float_feature_borders`
/// order) so a numeric-only model (`CtrFeatures`/`OneHotFeatures` both empty)
/// classifies exactly as before.
///
/// # Errors
/// [`ModelError::Deserialize`] on a malformed `FloatFeatures` table (the same
/// failure [`read_float_feature_borders`] surfaces).
fn build_combined_bins(trees: &TModelTrees) -> Result<Vec<BinKind>, ModelError> {
    let float_feature_borders = read_float_feature_borders(trees)?;
    let mut out: Vec<BinKind> = build_bin_features(&float_feature_borders)
        .into_iter()
        .map(|b| BinKind::Float {
            feature: b.feature,
            border: b.border,
        })
        .collect();

    if let Some(one_hot_features) = trees.OneHotFeatures() {
        for i in 0..one_hot_features.len() {
            let n_values = one_hot_features.get(i).Values().map_or(0, |v| v.len());
            for _ in 0..n_values {
                out.push(BinKind::OneHot);
            }
        }
    }

    if let Some(ctr_features) = trees.CtrFeatures() {
        for c in 0..ctr_features.len() {
            let n_borders = ctr_features.get(c).Borders().map_or(0, |b| b.len());
            for k in 0..n_borders {
                out.push(BinKind::Ctr {
                    ctr_feature: c,
                    border_index: k,
                });
            }
        }
    }

    Ok(out)
}

/// Build a [`CtrSplit`] from a `TCtrFeature` + its `border_index`-th border
/// (CTR-02): `projection` from `Ctr.Base.FeatureCombination.CatFeatures`
/// (sorted/deduped via [`cb_train::TProjection::from_features`]), `ctr_type`
/// from `Ctr.Base.CtrType`, `prior` from `(PriorNum, PriorDenom)`,
/// `target_border_idx` from `Ctr.TargetBorderIdx`, `border` from
/// `Borders[border_index]`, `(shift, scale)` from `(Ctr.Shift, Ctr.Scale)` —
/// every field a checked f32->f64 cast or a bounds-checked accessor.
///
/// # Errors
/// [`ModelError::Deserialize`] if `Ctr`/`Base`/`FeatureCombination`/`Borders`
/// is missing, `border_index` is out of range, `CatFeatures` holds a negative
/// index, `TargetBorderIdx` is negative, or `CtrType` is an unknown
/// discriminant.
fn ctr_split_from(feature: TCtrFeature<'_>, border_index: usize) -> Result<CtrSplit, ModelError> {
    let ctr = feature
        .Ctr()
        .ok_or_else(|| ModelError::Deserialize("TCtrFeature missing Ctr".to_owned()))?;
    let base = ctr
        .Base()
        .ok_or_else(|| ModelError::Deserialize("TModelCtr missing Base".to_owned()))?;
    let combination = base.FeatureCombination().ok_or_else(|| {
        ModelError::Deserialize("TModelCtrBase missing FeatureCombination".to_owned())
    })?;

    let mut cat_features: Vec<usize> = Vec::new();
    if let Some(cats) = combination.CatFeatures() {
        for i in 0..cats.len() {
            let raw = cats.get(i);
            let idx = usize::try_from(raw).map_err(|_| {
                ModelError::Deserialize(format!("negative CatFeatures index {raw}"))
            })?;
            cat_features.push(idx);
        }
    }
    let projection = cb_train::TProjection::from_features(&cat_features);

    // The generated `ECtrType` here is `model_generated`'s transparent
    // `ECtrType(pub i8)` tuple (a SEPARATE self-contained schema module from
    // `ctr_data_generated`'s copy) — convert via `.0` (MINOR-1).
    let ctr_type = ECtrType::from_i8(base.CtrType().0)?;

    let target_border_idx = usize::try_from(ctr.TargetBorderIdx())
        .map_err(|_| ModelError::Deserialize("negative TargetBorderIdx".to_owned()))?;

    let borders = feature
        .Borders()
        .ok_or_else(|| ModelError::Deserialize("TCtrFeature missing Borders".to_owned()))?;
    if border_index >= borders.len() {
        return Err(ModelError::Deserialize(format!(
            "ctr border_index {border_index} out of range ({} borders)",
            borders.len()
        )));
    }
    let border = f64::from(borders.get(border_index));

    Ok(CtrSplit {
        projection,
        ctr_type,
        prior: Prior {
            num: f64::from(ctr.PriorNum()),
            denom: f64::from(ctr.PriorDenom()),
        },
        target_border_idx,
        border,
        shift: f64::from(ctr.Shift()),
        scale: f64::from(ctr.Scale()),
    })
}

/// Serialize `model` to the native `.cbm` format at `path` (MODEL-01).
///
/// Emits the `CBM1` magic, the ui32 LE core size, and a FlatBuffers `TModelCore`
/// carrying `FormatVersion = "FlabuffersModel_v1"`, the global `TreeSplits`,
/// per-tree `TreeSizes` / `TreeStartOffsets`, the flat `LeafValues` /
/// `LeafWeights` arrays, the `FloatFeatures` borders (as f32, the schema type),
/// `ApproxDimension = 1`, and the single `Bias` (bias-free leaf values, Open Q3 /
/// Pitfall 6). A categorical model additionally emits the `CtrFeatures` core
/// section (T4, grouped from the trees' `ModelSplit::Ctr` splits) and appends the
/// `ctr_data` model-parts tail after the frame + core; a numeric-only model
/// (`ctr_data: None`) emits neither and stays byte-identical.
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

    // Append the CTR model-parts tail AFTER the 8-byte frame + core (the tail
    // lives at `buf[8 + core_len..]`, exactly where `decode_cbm` reads it). The
    // `core_len` field still counts ONLY the FlatBuffers core, NOT the tail. A
    // numeric-only model (`ctr_data: None`) appends nothing → byte-identical.
    if let Some(ctr_data) = &model.ctr_data {
        out.extend_from_slice(&encode_ctr_model_parts(ctr_data)?);
    }

    std::fs::write(path, out)?;
    Ok(())
}

/// Build the FlatBuffers `TModelCore` payload (the bytes after the 8-byte frame).
fn build_core_blob(model: &Model) -> Result<Vec<u8>, ModelError> {
    let bins = build_bin_features(&model.float_feature_borders);

    // CTR save plan (T1/T4): group the trees' `ModelSplit::Ctr` splits into the
    // ordered `CtrFeatures` identities and their combined global-index math. v1
    // has no one-hot bins, so the CTR range begins right after the float bins.
    let plan = build_ctr_features(model)?;
    let n_float_bins = bins.len();

    // A CTR split must have a matching baked table in `model.ctr_data` (its
    // apply-time lookup key) — never emit a CtrFeature whose table would miss.
    if !plan.identities.is_empty() {
        let ctr_data = model.ctr_data.as_ref().ok_or_else(|| {
            ModelError::Serialize(
                "model carries CTR splits but ctr_data is None (nothing to save)".to_owned(),
            )
        })?;
        for identity in &plan.identities {
            let key = ctr_base_key(identity.ctr_type, identity.projection.cat_features());
            if !ctr_data.tables.contains_key(&key) {
                return Err(ModelError::Serialize(format!(
                    "CTR split table {key:?} missing from ctr_data"
                )));
            }
        }
    }

    // Number of output dimensions (D-6.2-01 / Plan 06.2-02). A model carries its
    // training `approx_dimension`; `0` is meaningless, so treat it as the scalar
    // default `1` (older models / construction paths that left it unset).
    let dim = model.approx_dimension.max(1);

    // Global tree splits + per-tree sizes/offsets, and the flat leaf arrays.
    let mut tree_splits: Vec<i32> = Vec::new();
    let mut tree_sizes: Vec<i32> = Vec::new();
    let mut tree_start_offsets: Vec<i32> = Vec::new();
    let mut leaf_values: Vec<f64> = Vec::new();
    let mut leaf_weights: Vec<f64> = Vec::new();
    // Non-symmetric flat-node vectors (FEAT-06 / D-6.6-05); EMPTY for an oblivious
    // model so the symmetric wire bytes stay byte-identical (the FlatBuffers
    // builder omits an empty `None` vector → no `NonSymmetric*` table fields).
    let mut non_symmetric_step_nodes: Vec<TNonSymmetricTreeStepNode> = Vec::new();
    let mut non_symmetric_node_ids: Vec<u32> = Vec::new();

    // Non-symmetric model: serialize the per-NODE flat triple (TreeSplits per
    // node, NonSymmetricStepNodes, NonSymmetricNodeIdToLeafId) + the flat leaf
    // arrays. TreeSizes counts NODES (not splits) here. A model is EITHER
    // all-oblivious or all-non-symmetric (upstream never mixes policies).
    if !model.non_symmetric_trees.is_empty() {
        let mut node_offset: i32 = 0;
        // Running flat-LeafValues cursor (value units) — the GLOBAL base each
        // tree's LOCAL leaf ids re-globalize against (the inverse of the decode
        // localization). `value_base` advances by `distinct_leaves * dim` per tree.
        let mut value_base: usize = 0;
        for tree in &model.non_symmetric_trees {
            let node_count = i32::try_from(tree.tree_splits.len()).map_err(|_| {
                ModelError::SchemaVersion("non-symmetric node count exceeds i32".to_owned())
            })?;
            tree_start_offsets.push(node_offset);
            tree_sizes.push(node_count);
            node_offset = node_offset.checked_add(node_count).ok_or_else(|| {
                ModelError::SchemaVersion("cumulative non-symmetric node offset overflow".to_owned())
            })?;

            for (idx, split) in tree.tree_splits.iter().enumerate() {
                let is_pure_leaf = matches!(tree.step_nodes.get(idx), Some(&(0, 0)) | None);
                if is_pure_leaf {
                    // Pure leaf nodes serialize a filler global split index `0`
                    // (verified upstream layout); the apply walk never reads it.
                    tree_splits.push(0);
                } else if let Some(float_split) = split.as_float() {
                    tree_splits.push(split_to_global_index(float_split, &bins)?);
                } else {
                    // A CTR split at a non-symmetric INTERIOR node: v1 supports CTR
                    // splits on OBLIVIOUS trees only (SPEC §2). Reject loudly rather
                    // than silently write a filler `0` (which would mis-apply).
                    return Err(ModelError::Serialize(
                        "non-symmetric CTR save unsupported (v1)".to_owned(),
                    ));
                }
            }
            for &(left_diff, right_diff) in &tree.step_nodes {
                non_symmetric_step_nodes
                    .push(TNonSymmetricTreeStepNode::new(left_diff, right_diff));
            }

            // Distinct leaves = count of nodes carrying a VALID (non-`u32::MAX`)
            // leaf id (halt points, including one-sided `(d, 0)` / `(0, d)` nodes —
            // `evaluator_impl.cpp:738` halts whenever the chosen diff is 0). The
            // decode stores these ids LOCAL (0-based per tree); re-globalize each to
            // the flat `LeafValues` index `value_base + local_leaf * dim`.
            let distinct_leaves = tree
                .node_id_to_leaf_id
                .iter()
                .filter(|&&id| id != u32::MAX)
                .count();
            for &local_id in &tree.node_id_to_leaf_id {
                if local_id == u32::MAX {
                    non_symmetric_node_ids.push(u32::MAX);
                } else {
                    let global = value_base
                        + (local_id as usize).saturating_mul(dim.max(1));
                    non_symmetric_node_ids.push(u32::try_from(global).map_err(|_| {
                        ModelError::SchemaVersion(
                            "non-symmetric global leaf id exceeds u32".to_owned(),
                        )
                    })?);
                }
            }

            let n_leaves = if dim == 0 { 0 } else { distinct_leaves };
            for l in 0..n_leaves {
                for d in 0..dim {
                    let v = tree.leaf_values.get(d * n_leaves + l).copied().ok_or_else(|| {
                        ModelError::SchemaVersion(
                            "non-symmetric leaf_values length is not a multiple of approx_dimension"
                                .to_owned(),
                        )
                    })?;
                    leaf_values.push(v);
                }
            }
            leaf_weights.extend_from_slice(&tree.leaf_weights);
            value_base = value_base.saturating_add(n_leaves.saturating_mul(dim.max(1)));
        }
    }

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
            // A FLOAT split maps to its float-feature/border global index; a CTR
            // split maps to its combined `CtrFeatures` global index (T4). Both push
            // EXACTLY one `tree_splits` entry, so `TreeSizes`/offsets are unchanged.
            match split {
                ModelSplit::Float(float_split) => {
                    tree_splits.push(split_to_global_index(float_split, &bins)?);
                }
                ModelSplit::Ctr(ctr_split) => {
                    tree_splits.push(ctr_split_to_global_index(ctr_split, n_float_bins, &plan)?);
                }
            }
        }
        // LEAF-MAJOR transpose (Pitfall 6): the training buffer is DIMENSION-MAJOR
        // (`leaf_values[d * n_leaves + l]`), but the `.cbm` / json `LeafValues` are
        // LEAF-MAJOR (`leaf0_d0, leaf0_d1, …, leaf1_d0, …` = `leaf_values[l * dim +
        // d]`). At `dim == 1` the two orders coincide (a single dimension), so the
        // wire bytes are byte-identical to the pre-6.2 scalar model. `leaf_weights`
        // stays one-per-leaf (the document partition is shared across dimensions),
        // so it is emitted unchanged at any dimension.
        let n_leaves = if dim == 0 { 0 } else { tree.leaf_values.len() / dim };
        for l in 0..n_leaves {
            for d in 0..dim {
                // Source dim-major index `d * n_leaves + l`; checked `.get` only.
                let v = tree.leaf_values.get(d * n_leaves + l).copied().ok_or_else(|| {
                    ModelError::SchemaVersion(
                        "leaf_values length is not a multiple of approx_dimension"
                            .to_owned(),
                    )
                })?;
                leaf_values.push(v);
            }
        }
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

    // CtrFeatures (T4): one `TCtrFeature` per grouped identity. `None` when the
    // model carries no CTR splits, so a numeric-only model omits the field
    // entirely and its `.cbm` bytes stay byte-identical (regression lock).
    let ctr_features_vec = if plan.identities.is_empty() {
        None
    } else {
        let mut ctr_offsets = Vec::with_capacity(plan.identities.len());
        for identity in &plan.identities {
            ctr_offsets.push(build_tctr_feature(&mut fbb, identity)?);
        }
        Some(fbb.create_vector(&ctr_offsets))
    };

    let tree_splits_vec = fbb.create_vector(&tree_splits);
    let tree_sizes_vec = fbb.create_vector(&tree_sizes);
    let tree_start_offsets_vec = fbb.create_vector(&tree_start_offsets);
    let leaf_values_vec = fbb.create_vector(&leaf_values);
    let leaf_weights_vec = fbb.create_vector(&leaf_weights);
    // Non-symmetric vectors — `None` for an oblivious model so the wire bytes
    // stay byte-identical (the table fields are simply absent, D-6.6-05).
    let non_symmetric_step_nodes_vec = if non_symmetric_step_nodes.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&non_symmetric_step_nodes))
    };
    let non_symmetric_node_ids_vec = if non_symmetric_node_ids.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&non_symmetric_node_ids))
    };

    let approx_dimension = i32::try_from(dim).map_err(|_| {
        ModelError::SchemaVersion("approx_dimension exceeds i32 range".to_owned())
    })?;
    let model_trees = TModelTrees::create(
        &mut fbb,
        &TModelTreesArgs {
            ApproxDimension: approx_dimension,
            TreeSplits: Some(tree_splits_vec),
            TreeSizes: Some(tree_sizes_vec),
            TreeStartOffsets: Some(tree_start_offsets_vec),
            FloatFeatures: Some(float_features),
            CtrFeatures: ctr_features_vec,
            LeafValues: Some(leaf_values_vec),
            LeafWeights: Some(leaf_weights_vec),
            NonSymmetricStepNodes: non_symmetric_step_nodes_vec,
            NonSymmetricNodeIdToLeafId: non_symmetric_node_ids_vec,
            Scale: 1.0,
            Bias: model.bias,
            ..TModelTreesArgs::default()
        },
    );

    let format_version = fbb.create_string(FLATBUFFERS_MODEL_V1);

    // InfoMap: emit the multiclass `class_params` (the SORTED distinct class labels
    // in `class_to_label`, the generic key upstream reads first, `model.cpp:1431`)
    // as a JSON STRING value, ONLY when the model carries class labels. A scalar /
    // regression model emits NO InfoMap, so the `.cbm` bytes stay byte-identical to
    // the pre-6.2 scalar model (D-04). CR-01 / LOSS-02.
    let info_map = if model.class_to_label.is_empty() {
        None
    } else {
        let class_params_json = serde_json::json!({
            "class_to_label": model.class_to_label,
        })
        .to_string();
        let key = fbb.create_string("class_params");
        let value = fbb.create_string(&class_params_json);
        let kv = TKeyValue::create(
            &mut fbb,
            &TKeyValueArgs {
                Key: Some(key),
                Value: Some(value),
            },
        );
        Some(fbb.create_vector(&[kv]))
    };

    let core = TModelCore::create(
        &mut fbb,
        &TModelCoreArgs {
            FormatVersion: Some(format_version),
            ModelTrees: Some(model_trees),
            InfoMap: info_map,
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

    // Recover the multiclass class labels from the InfoMap `class_params` /
    // `multiclass_params` JSON-string value (CR-01 / LOSS-02). Absent for a scalar
    // model (empty vector).
    let class_to_label = read_class_to_label(&model_core)?;

    // The optional model-parts tail (CTR-03/CTR-04): bytes after the 8-byte
    // frame + declared core. A numeric-only `.cbm` (no `CtrFeatures`) never
    // reads this; an ABSENT tail resolves to an empty slice here (never a
    // panic) so a `CtrFeatures`-present model with a missing/truncated tail
    // surfaces as a typed `ModelError` from `decode_ctr_model_parts`, not an
    // `Option` short-circuit.
    let tail = buf.get(8usize.saturating_add(declared)..).unwrap_or(&[]);

    reconstruct_model(&trees, class_to_label, tail)
}

/// Parse the SORTED distinct class labels from the `TModelCore` InfoMap: find the
/// `class_params` key (the generic key upstream reads first, `model.cpp:1431`),
/// falling back to `multiclass_params`, whose VALUE is a JSON string carrying
/// `class_to_label`. Each label coerces to `f64` (the canonical
/// [`Model::class_to_label`] type). Absent InfoMap / key yields an empty vector (a
/// scalar model carries no labels). A malformed JSON value is a typed
/// [`ModelError::Deserialize`] — never a panic (T-6.2-06-02).
fn read_class_to_label(core: &TModelCore) -> Result<Vec<f64>, ModelError> {
    let Some(info_map) = core.InfoMap() else {
        return Ok(Vec::new());
    };
    for key in ["class_params", "multiclass_params"] {
        for i in 0..info_map.len() {
            let kv = info_map.get(i);
            if kv.Key() == key {
                let parsed: serde_json::Value =
                    serde_json::from_str(kv.Value()).map_err(|e| {
                        ModelError::Deserialize(format!(
                            "malformed InfoMap {key} JSON value: {e}"
                        ))
                    })?;
                let labels = parsed
                    .get("class_to_label")
                    .and_then(serde_json::Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(serde_json::Value::as_f64)
                            .collect::<Vec<f64>>()
                    })
                    .unwrap_or_default();
                return Ok(labels);
            }
        }
    }
    Ok(Vec::new())
}

/// Reconstruct the canonical [`Model`] from a verified `TModelTrees`, carrying the
/// `class_to_label` already parsed from the enclosing `TModelCore` InfoMap and
/// the raw model-parts `tail` bytes (CTR-03/CTR-04) — read ONLY when
/// `CtrFeatures` is non-empty, so a numeric-only model never touches `tail`
/// (byte-identical to the pre-CTR-load decode).
fn reconstruct_model(
    trees: &TModelTrees,
    class_to_label: Vec<f64>,
    tail: &[u8],
) -> Result<Model, ModelError> {
    // Float-feature borders (f32 on the wire -> f64 canonical), in feature order.
    let float_feature_borders = read_float_feature_borders(trees)?;
    let bins = build_bin_features(&float_feature_borders);
    let has_ctr_features = trees.CtrFeatures().is_some_and(|v| !v.is_empty());

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

    // Number of output dimensions (D-6.2-01 / Plan 06.2-02); `<= 0` (older models
    // / unset) means the scalar default `1`.
    let dim = usize::try_from(trees.ApproxDimension()).unwrap_or(1).max(1);

    // Non-symmetric model (FEAT-06 / D-6.6-05): when `NonSymmetricStepNodes` is
    // present AND NON-EMPTY, `TreeSizes` counts NODES (not splits), so the
    // oblivious `2^size` leaf decode below would be WRONG — take the dedicated
    // flat-node decode and return early. An oblivious upstream `.cbm` carries an
    // EMPTY `NonSymmetricStepNodes` vector (`Some([])`, NOT `None` — verified
    // catboost 1.2.10), so it falls through to the byte-identical symmetric path.
    if trees
        .NonSymmetricStepNodes()
        .is_some_and(|v| !v.is_empty())
    {
        // v1 supports CTR splits on OBLIVIOUS trees only (SPEC §2 / research
        // §9.4 scope). A non-symmetric model carrying `CtrFeatures` is a typed
        // error rather than a silent `ctr_data: None` mis-load.
        if has_ctr_features {
            return Err(ModelError::Deserialize(
                "non-symmetric CTR unsupported (v1)".to_owned(),
            ));
        }
        let non_symmetric_trees =
            reconstruct_non_symmetric(trees, &bins, &leaf_values, leaf_weights.as_ref(), dim)?;
        return Ok(Model {
            oblivious_trees: Vec::new(),
            non_symmetric_trees,
            region_trees: Vec::new(),
            bias: read_bias(trees),
            float_feature_borders,
            ctr_data: None,
            approx_dimension: dim,
            class_to_label,
        });
    }

    // The FULL combined `Float -> OneHot -> Ctr` bin table (CTR-01); a
    // numeric-only model's prefix is byte-identical to `bins` above (same
    // `float_feature_borders`), so this replaces `bins.get(gidx)` in the split
    // loop below without changing the numeric decode.
    let combined_bins = build_combined_bins(trees)?;

    let mut oblivious_trees = Vec::with_capacity(tree_sizes.len());
    let mut split_cursor: usize = 0;
    // Separate cursors: `value_cursor` strides the LEAF-MAJOR `LeafValues` by
    // `leaf_count * dim`; `weight_cursor` strides the one-per-leaf `LeafWeights`
    // by `leaf_count`. At `dim == 1` they advance in lockstep (byte-identical).
    let mut value_cursor: usize = 0;
    let mut weight_cursor: usize = 0;

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
            let bin_kind = combined_bins.get(gidx).ok_or_else(|| {
                ModelError::Deserialize(format!(
                    "global split index {gidx} out of range (bin features: {})",
                    combined_bins.len()
                ))
            })?;
            let model_split = match bin_kind {
                BinKind::Float { feature, border } => crate::ModelSplit::Float(Split {
                    feature: *feature,
                    border: *border,
                }),
                // One-hot FEATURE tables are counted for the bin offset (CTR-01)
                // but no `ModelSplit::OneHot` variant exists; a *tree split*
                // referencing this range is a typed error (CTR-05), never a
                // silent drop.
                BinKind::OneHot => {
                    return Err(ModelError::Deserialize(
                        "one-hot split unsupported (v1)".to_owned(),
                    ))
                }
                BinKind::Ctr {
                    ctr_feature,
                    border_index,
                } => {
                    let ctr_features = trees.CtrFeatures().ok_or_else(|| {
                        ModelError::Deserialize(
                            "split references CtrFeatures but model has none".to_owned(),
                        )
                    })?;
                    if *ctr_feature >= ctr_features.len() {
                        return Err(ModelError::Deserialize(format!(
                            "ctr_feature index {ctr_feature} out of range ({} CtrFeatures)",
                            ctr_features.len()
                        )));
                    }
                    let tcf = ctr_features.get(*ctr_feature);
                    crate::ModelSplit::Ctr(ctr_split_from(tcf, *border_index)?)
                }
            };
            splits.push(model_split);
        }
        split_cursor = split_cursor.saturating_add(size);

        // Leaf slice for this tree: 2^size leaves, flat-array offset per tree.
        let leaf_count = 1usize.checked_shl(u32::try_from(size).map_err(|_| {
            ModelError::Deserialize("tree size exceeds u32".to_owned())
        })?)
        .ok_or_else(|| ModelError::Deserialize("2^depth overflowed usize".to_owned()))?;

        let (tree_values, tree_weights) = read_tree_leaves(
            &leaf_values,
            leaf_weights.as_ref(),
            value_cursor,
            weight_cursor,
            leaf_count,
            dim,
        )?;
        value_cursor = value_cursor.saturating_add(leaf_count.saturating_mul(dim));
        weight_cursor = weight_cursor.saturating_add(leaf_count);

        oblivious_trees.push(ObliviousTree {
            splits,
            leaf_values: tree_values,
            leaf_weights: tree_weights,
        });
    }

    // Parse the model-parts tail into `ctr_data` ONLY when the model actually
    // carries `CtrFeatures` (CTR-03/CTR-04) — a numeric-only model's `tail` is
    // never touched, so it stays byte-identical to the pre-CTR-load decode
    // (`ctr_data: None`). A `CtrFeatures`-present model with a missing/empty
    // tail surfaces as a typed error from `decode_ctr_model_parts` (never a
    // silent `None`).
    let ctr_data = if has_ctr_features {
        Some(decode_ctr_model_parts(tail)?)
    } else {
        None
    };

    Ok(Model {
        oblivious_trees,
        // Oblivious models carry no non-symmetric trees (the non-symmetric `.cbm`
        // is handled by the early return above, D-6.6-05).
        non_symmetric_trees: Vec::new(),
        // Region models are not produced by the `.cbm` decode path (GPUT-18 lands
        // via the json round-trip); an oblivious `.cbm` carries no region trees.
        region_trees: Vec::new(),
        bias: read_bias(trees),
        float_feature_borders,
        ctr_data,
        approx_dimension: dim,
        // Recovered from the `TModelCore` InfoMap `class_params` / `multiclass_params`
        // JSON value (CR-01 / LOSS-02); empty for a scalar model.
        class_to_label,
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

/// Reconstruct the non-symmetric (Lossguide / Depthwise) trees from a verified
/// `TModelTrees` (FEAT-06 / D-6.6-05). Returns an EMPTY vector for an oblivious
/// model (the `NonSymmetricStepNodes` / `NonSymmetricNodeIdToLeafId` vectors are
/// absent), so the symmetric read path is byte-identical (Pitfall — branch on
/// presence). A model is EITHER all-oblivious or all-non-symmetric.
///
/// Per-tree NODE count comes from `TreeSizes` / `TreeStartOffsets` (which, for a
/// non-symmetric model, count NODES not splits); per-node global split indices
/// come from `TreeSplits`; `(left_diff, right_diff)` from `NonSymmetricStepNodes`;
/// per-node leaf ids from `NonSymmetricNodeIdToLeafId`. The flat `LeafValues` /
/// `LeafWeights` arrays are sliced per tree by the distinct leaf count (the
/// number of non-`u32::MAX` node-id entries), un-transposed LEAF-MAJOR →
/// DIMENSION-MAJOR identically to the oblivious path (Pitfall 6).
///
/// # Errors
/// [`ModelError::Deserialize`] on any out-of-bounds index / ragged length
/// (T-06.6-06 — every read is checked; nothing panics, no OOB).
fn reconstruct_non_symmetric(
    trees: &TModelTrees,
    bins: &[BinFeature],
    leaf_values: &flatbuffers::Vector<f64>,
    leaf_weights: Option<&flatbuffers::Vector<f64>>,
    dim: usize,
) -> Result<Vec<NonSymmetricTree>, ModelError> {
    // An oblivious `.cbm` carries an EMPTY `NonSymmetricStepNodes` (`Some([])`),
    // not `None`; treat both as "no non-symmetric trees" (byte-identical path).
    let step_nodes = match trees.NonSymmetricStepNodes() {
        Some(v) if !v.is_empty() => v,
        _ => return Ok(Vec::new()),
    };
    let node_id_to_leaf_id = trees.NonSymmetricNodeIdToLeafId().ok_or_else(|| {
        ModelError::Deserialize(
            "NonSymmetricStepNodes present but NonSymmetricNodeIdToLeafId missing".to_owned(),
        )
    })?;
    let tree_splits = trees
        .TreeSplits()
        .ok_or_else(|| ModelError::Deserialize("missing TreeSplits".to_owned()))?;
    let tree_sizes = trees
        .TreeSizes()
        .ok_or_else(|| ModelError::Deserialize("missing TreeSizes".to_owned()))?;

    let mut out = Vec::with_capacity(tree_sizes.len());
    let mut node_cursor: usize = 0;
    // The non-symmetric `NonSymmetricNodeIdToLeafId` entries are GLOBAL indices
    // into the FLAT, LEAF-MAJOR `LeafValues` array (`evaluator_impl.cpp:742-751`:
    // `firstValueIdx = NodeIdToLeafId[index]; result += LeafValues[firstValueIdx
    // + classId]`), so they are value-unit indices (`leaf_index * dim`). We slice
    // each tree's leaves out of that flat array and LOCALIZE the per-node leaf id
    // to a 0-based LEAF index within the tree (`(global - value_base) / dim`), so
    // the per-tree `node_id_to_leaf_id` indexes the per-tree un-transposed
    // DIMENSION-MAJOR `leaf_values` directly at apply time.
    let mut value_cursor: usize = 0; // start of this tree in flat LeafValues (value units)
    let mut weight_cursor: usize = 0; // start of this tree in LeafWeights (leaf units)

    for ti in 0..tree_sizes.len() {
        let node_count = usize::try_from(tree_sizes.get(ti)).map_err(|_| {
            ModelError::Deserialize("negative non-symmetric tree node count".to_owned())
        })?;

        let mut node_splits: Vec<crate::ModelSplit> = Vec::with_capacity(node_count);
        let mut node_steps: Vec<(u16, u16)> = Vec::with_capacity(node_count);
        let mut node_leaf_ids: Vec<u32> = Vec::with_capacity(node_count);
        // A node is a HALT point (carries a real leaf value) when EITHER subtree
        // diff is 0 (`evaluator_impl.cpp:738` halts when the chosen `diff == 0`),
        // i.e. a node with a one-sided `(d, 0)` / `(0, d)` step still terminates
        // the walk on the zero side. Such a node has a VALID (non-`u32::MAX`)
        // `NonSymmetricNodeIdToLeafId` slot. The number of distinct leaf slots in
        // this tree is the count of valid entries.
        let mut distinct_leaves: usize = 0;
        // Track the tree's leaf-id span so we can validate contiguity and derive
        // the local 0-based leaf index.
        let mut min_global_value: Option<usize> = None;

        for off in 0..node_count {
            let global_node = node_cursor.checked_add(off).ok_or_else(|| {
                ModelError::Deserialize("non-symmetric node cursor overflow".to_owned())
            })?;
            if global_node >= step_nodes.len()
                || global_node >= node_id_to_leaf_id.len()
                || global_node >= tree_splits.len()
            {
                return Err(ModelError::Deserialize(
                    "non-symmetric node arrays shorter than declared tree sizes".to_owned(),
                ));
            }
            let sn = step_nodes.get(global_node);
            let (left_diff, right_diff) = (sn.LeftSubtreeDiff(), sn.RightSubtreeDiff());
            node_steps.push((left_diff, right_diff));

            let raw_leaf_id = node_id_to_leaf_id.get(global_node);
            // A `u32::MAX` slot marks a pure-interior node (both subtrees non-zero):
            // the walk can never halt there, so it carries no leaf value. Any other
            // value is a GLOBAL flat-LeafValues index (value units).
            let is_halt_node = raw_leaf_id != u32::MAX;
            if is_halt_node {
                distinct_leaves = distinct_leaves.saturating_add(1);
                let gv = usize::try_from(raw_leaf_id).map_err(|_| {
                    ModelError::Deserialize("non-symmetric leaf id exceeds usize".to_owned())
                })?;
                min_global_value =
                    Some(min_global_value.map_or(gv, |m: usize| m.min(gv)));
            }

            // Every node carries a split slot. A halt node MAY also be interior on
            // its non-zero side (a `(d, 0)` node both halts-right and branches-left),
            // so we decode the split for ANY node whose declared split index is a
            // valid bin; a pure leaf (no real split) keeps a sentinel placeholder.
            let is_pure_leaf = left_diff == 0 && right_diff == 0;
            if is_pure_leaf {
                node_splits.push(crate::ModelSplit::Float(Split {
                    feature: 0,
                    border: f64::NEG_INFINITY,
                }));
            } else {
                let gidx = usize::try_from(tree_splits.get(global_node)).map_err(|_| {
                    ModelError::Deserialize("negative non-symmetric split index".to_owned())
                })?;
                let bin = bins.get(gidx).ok_or_else(|| {
                    ModelError::Deserialize(format!(
                        "non-symmetric global split index {gidx} out of range (bins: {})",
                        bins.len()
                    ))
                })?;
                node_splits.push(crate::ModelSplit::Float(Split {
                    feature: bin.feature,
                    border: bin.border,
                }));
            }
        }
        node_cursor = node_cursor.saturating_add(node_count);

        // The tree's leaf values occupy `distinct_leaves * dim` flat slots starting
        // at the running `value_cursor` (the leaf-major flat layout). Localize each
        // node's GLOBAL value index to a 0-based LEAF index within this tree:
        // `local_leaf = (global_value - value_base) / dim`. The flat array is
        // contiguous per tree, so the minimum observed global value MUST equal the
        // running cursor (validated below).
        let value_base = min_global_value.unwrap_or(value_cursor);
        if value_base != value_cursor {
            return Err(ModelError::Deserialize(format!(
                "non-symmetric tree {ti} leaf-id base {value_base} does not match \
                 expected flat-LeafValues cursor {value_cursor}"
            )));
        }
        // Build the LOCAL per-node leaf ids (0-based leaf index within the tree;
        // `u32::MAX` for pure-interior nodes that never halt).
        for off in 0..node_count {
            let global_node = node_cursor.saturating_sub(node_count).saturating_add(off);
            let raw_leaf_id = node_id_to_leaf_id.get(global_node);
            if raw_leaf_id == u32::MAX {
                node_leaf_ids.push(u32::MAX);
            } else {
                let gv = usize::try_from(raw_leaf_id).unwrap_or(0);
                let local_leaf = gv.saturating_sub(value_base) / dim.max(1);
                node_leaf_ids.push(u32::try_from(local_leaf).map_err(|_| {
                    ModelError::Deserialize("local non-symmetric leaf id exceeds u32".to_owned())
                })?);
            }
        }

        // Slice this tree's leaves (un-transpose LEAF-MAJOR → DIMENSION-MAJOR).
        let (tree_values, tree_weights) = read_tree_leaves(
            leaf_values,
            leaf_weights,
            value_cursor,
            weight_cursor,
            distinct_leaves,
            dim,
        )?;
        value_cursor = value_cursor.saturating_add(distinct_leaves.saturating_mul(dim));
        weight_cursor = weight_cursor.saturating_add(distinct_leaves);

        out.push(NonSymmetricTree {
            tree_splits: node_splits,
            step_nodes: node_steps,
            node_id_to_leaf_id: node_leaf_ids,
            leaf_values: tree_values,
            leaf_weights: tree_weights,
        });
    }

    Ok(out)
}

/// Slice one tree's `leaf_count` values (and weights, zero-filled if absent) out
/// of the flat leaf arrays starting at `offset`, with bounds checks.
/// Read one tree's leaves, un-transposing the wire LEAF-MAJOR `LeafValues`
/// (`leaf_values[l * dim + d]`) back into the canonical DIMENSION-MAJOR buffer
/// (`leaf_values[d * leaf_count + l]`, Pitfall 6 / Plan 06.2-02). `value_offset`
/// is this tree's start in the flat `LeafValues` array (`leaf_count * dim`
/// values consumed); `weight_offset` is its start in `LeafWeights` (`leaf_count`
/// weights consumed — weights are one-per-leaf, NOT per-dimension). At `dim == 1`
/// leaf-major == dim-major, so the returned `values` are byte-identical to the
/// pre-6.2 scalar read.
fn read_tree_leaves(
    leaf_values: &flatbuffers::Vector<f64>,
    leaf_weights: Option<&flatbuffers::Vector<f64>>,
    value_offset: usize,
    weight_offset: usize,
    leaf_count: usize,
    dim: usize,
) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
    let value_span = leaf_count.checked_mul(dim).ok_or_else(|| {
        ModelError::Deserialize("leaf_count * approx_dimension overflow".to_owned())
    })?;
    let value_end = value_offset.checked_add(value_span).ok_or_else(|| {
        ModelError::Deserialize("leaf value offset + span overflow".to_owned())
    })?;
    if value_end > leaf_values.len() {
        return Err(ModelError::Deserialize(
            "LeafValues shorter than declared tree leaves".to_owned(),
        ));
    }
    // Un-transpose leaf-major -> dim-major: dst[d * leaf_count + l] = src[l*dim+d].
    let mut values = vec![0.0_f64; value_span];
    for l in 0..leaf_count {
        for d in 0..dim {
            let src = value_offset + l * dim + d;
            if let Some(slot) = values.get_mut(d * leaf_count + l) {
                *slot = leaf_values.get(src);
            }
        }
    }
    // Weights are one-per-leaf (shared across dimensions). Optional: zero-fill
    // when absent or short.
    let weight_end = weight_offset.checked_add(leaf_count).ok_or_else(|| {
        ModelError::Deserialize("leaf weight offset + count overflow".to_owned())
    })?;
    let mut weights = Vec::with_capacity(leaf_count);
    for i in weight_offset..weight_end {
        let w = leaf_weights
            .filter(|lw| i < lw.len())
            .map_or(0.0, |lw| lw.get(i));
        weights.push(w);
    }
    Ok((values, weights))
}
