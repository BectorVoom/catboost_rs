//! Bit-packed device-resident compressed index (cindex) — GPUT-15, Plan 10-06.
//!
//! This module builds the upstream grouped `WriteCompressedIndex` layout addressed by a
//! per-feature [`TCFeature`] `{Offset, Shift, Mask, ...}` descriptor: the plain
//! one-`u32`-per-cell quantized-bin matrix (`cindex[feature * n + obj]`) is repacked so
//! that SEVERAL features whose combined bit-width fits a 32-bit word share ONE word per
//! object, and each feature's bin is extracted by the ONE device accessor
//! [`crate::kernels::read_bin`] as `(cindex[Offset + obj] >> Shift) & Mask`. The bin
//! VALUE is byte-identical to the plain layout — only its STORAGE/EXTRACTION changes, so
//! the bin->border join every histogram/partition consumer performs is unchanged
//! (T-10-15). Memory efficiency is a first-class constraint (CLAUDE.md): packing e.g.
//! four 8-bit features into one word quarters the cindex footprint every histogram kernel
//! streams.
//!
//! # Open Q1 resolution (host-pack-then-upload-once, RESEARCH A2)
//!
//! GPUT-15 requires a DEVICE-RESIDENT bit-packed cindex; it does NOT require the packing
//! itself to run on the device. The borders / quantization are the CPU ≤1e-5 reference
//! and stay host-side, so the packing is a pure host transform of the already-quantized
//! bins. We therefore HOST-PACK the grouped layout once (this module's [`pack_cindex`])
//! and upload the packed words + the [`TCFeature`] table ONCE per fill — the packed
//! buffer is then fully device-resident and every kernel reads it in place. This is the
//! A2 interpretation: "device-resident cindex" is satisfied by a host-packed,
//! upload-once buffer; the on-device `binarize.cu` `WriteCompressedIndex` kernel
//! (§6.6a, `blockSize = 256`) is an equivalent PACKING location, reserved as a follow-up
//! only if a later phase needs the bins packed without a host round-trip. The extraction
//! math (`read_bin`) is byte-identical to what the on-device packer would produce, so the
//! choice is invisible to every consumer. Documented here per the plan's acceptance
//! criterion; the SUMMARY records the same decision.
//!
//! # Bit sizing vs. `bit_pack_layout` (10-05)
//!
//! 10-05's [`crate::kernels::bit_pack_layout`] packs MANY OBJECTS of ONE feature into a
//! word (`keys_per_word` objects per word) and sizes from a BORDER count with the
//! `n_bins + 1` convention. The cindex packs MANY FEATURES of ONE object into a word (one
//! word per object per group) and sizes from a BUCKET count (bin values `0..n_buckets`):
//! [`feature_bits`] = `ceil(log2(n_buckets))`. The two share only the `ceil(log2(..))`
//! sizing idea; the LAYOUT (grouped `TCFeature` Offset/Shift) is this module's job (the
//! forward hand-off 10-05's SUMMARY names). All Offset / word-count / bit-width
//! arithmetic is `checked_*` → [`CbError::OutOfRange`] (T-10-16); length disagreements →
//! [`CbError::LengthMismatch`].

use cb_core::{CbError, CbResult};
use rayon::prelude::*;

/// Per-feature bit-packed cindex descriptor (upstream `TCFeature`). `offset` is the WORD
/// base of the feature's group ([`crate::kernels::read_bin`] indexes `cindex[offset +
/// obj]`); `shift`/`mask` extract the feature's field from the shared word. `first_fold_index`
/// / `folds` describe the feature's border-fold span (the bin->border join is unchanged);
/// `one_hot_feature` selects EQUALITY (`== value`) vs THRESHOLD (`> bin`) split semantics
/// downstream — the extracted bin VALUE is identical either way, only the split test
/// differs (routed by the consumer, not here).
//
// `first_fold_index` and `folds` are CONTRACT-ONLY fields: they belong to the plan's
// frozen `TCFeature` field set but nothing in the lib target reads them (only the
// two writes below and this doc). `#[allow(dead_code)]` on each keeps the default
// build warning-free. `one_hot_feature` IS read — [`PackedCindex::device_arrays`]
// exports it as the fourth device array (SPEC-OH-21 / T24).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TCFeature {
    /// Word base of this feature's group (`read_bin` reads `cindex[offset + obj]`).
    pub offset: u64,
    /// Extraction mask `(1 << bits) - 1` for this feature's field width.
    pub mask: u32,
    /// Bit offset of this feature's field within the shared word.
    pub shift: u32,
    /// First border-fold index of this feature (the bin->border join base; 0 for the
    /// single-feature-group MVP, reserved for the multi-group fold offset).
    #[allow(dead_code)]
    pub first_fold_index: u32,
    /// Number of border folds (buckets) this feature spans — **as the CALLER supplied
    /// it in `n_buckets`**, which on the production path is the PADDED UNIFORM LINE
    /// WIDTH (`session.rs` packs `vec![n_bins_line; eff_n_features]`), not the
    /// feature's real cardinality.
    ///
    /// **Never use this as a one-hot candidate bound** (SPEC-OH-22 / [C16]): with
    /// `folds[f] == n_bins_line` for every feature, `border < folds[feature]` IS the
    /// loop bound, so it excludes nothing and a cardinality-2 column contributes 30
    /// phantom candidates in a 32-wide line. The real per-feature cardinality travels
    /// separately as `real_folds` (built by the host quantizer, carried on
    /// `DeviceTrainConfig`) and is deliberately NOT part of
    /// [`PackedCindex::device_arrays`]. Passing true cardinalities into `n_buckets`
    /// instead is also forbidden — it would change [`feature_bits`] and hence the
    /// packed words for EVERY pool, including float-only.
    #[allow(dead_code)]
    pub folds: u32,
    /// Whether this feature uses one-hot (equality) split semantics downstream.
    pub one_hot_feature: bool,
}

/// The packed cindex: the grouped bit-packed `words` + the per-feature [`TCFeature`]
/// table. `words` has length `num_groups * n` (one word per object per group);
/// `features` has length `n_features`.
#[derive(Debug, Clone)]
pub(crate) struct PackedCindex {
    /// The grouped bit-packed words (feature groups share words; `read_bin`-addressed).
    pub words: Vec<u32>,
    /// The per-feature `TCFeature` descriptor table (Offset/Shift/Mask/...).
    pub features: Vec<TCFeature>,
}

impl PackedCindex {
    /// Device-ready per-feature `(offsets, shifts, masks, one_hot_flags)` `u32` arrays
    /// for [`crate::kernels::read_bin`] and the split-semantics selection.
    /// `TCFeature.offset` is checked-cast to `u32` (the device array index type); an
    /// offset that overflows `u32` surfaces [`CbError::OutOfRange`] (T-10-16 — no
    /// unguarded index reaches the device). `one_hot_flags` is `0`/`1` rather than
    /// `bool` because the device index type is `u32`.
    ///
    /// `TCFeature.folds` is deliberately NOT exported here (SPEC-OH-22 / [C16]): on
    /// the production path it is the padded uniform line width and must never bound a
    /// one-hot candidate. The real per-feature cardinality travels as `real_folds` on
    /// `DeviceTrainConfig`.
    pub fn device_arrays(&self) -> CbResult<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let mut offsets = Vec::with_capacity(self.features.len());
        let mut shifts = Vec::with_capacity(self.features.len());
        let mut masks = Vec::with_capacity(self.features.len());
        let mut one_hot_flags = Vec::with_capacity(self.features.len());
        for f in &self.features {
            let off = u32::try_from(f.offset).map_err(|_| {
                CbError::OutOfRange(format!(
                    "cindex offset {} exceeds u32 device index range",
                    f.offset
                ))
            })?;
            offsets.push(off);
            shifts.push(f.shift);
            masks.push(f.mask);
            one_hot_flags.push(u32::from(f.one_hot_feature));
        }
        Ok((offsets, shifts, masks, one_hot_flags))
    }
}

/// Bits needed to represent bin values `0..n_buckets` (i.e. `ceil(log2(n_buckets))`,
/// clamped to `1..=32`). `n_buckets` is the per-feature BUCKET count (the quantized bin
/// takes a value in `0..n_buckets`); a single-bucket feature still needs one bit so the
/// packing geometry is well-defined. Overflow-guarded (T-10-16): a feature needing more
/// than 32 bits cannot share a 32-bit word and surfaces [`CbError::OutOfRange`].
pub(crate) fn feature_bits(n_buckets: usize) -> CbResult<u32> {
    if n_buckets <= 1 {
        return Ok(1);
    }
    // max bin value = n_buckets - 1; bits = floor(log2(max)) + 1 = ceil(log2(n_buckets)).
    let max_val = (n_buckets - 1) as u64;
    let bits = max_val.ilog2() + 1;
    if bits == 0 || bits > 32 {
        return Err(CbError::OutOfRange(format!(
            "cindex feature needs {bits} bits (n_buckets {n_buckets}); a packed field is at most 32 bits"
        )));
    }
    Ok(bits)
}

/// The bins-independent half of [`pack_cindex`]: the per-feature word-group placement
/// `(group, shift, mask)` and the [`TCFeature`] descriptor table, both pure functions of
/// the bucket counts. Shared by the host packer and the device quantize+pack fast path
/// (QPACK-01), so the two paths cannot disagree on geometry.
pub(crate) struct CindexPlan {
    /// Per-feature `TCFeature` descriptors (length `n_features`).
    pub features: Vec<TCFeature>,
    /// Per-feature `(group, shift, mask)` placement (length `n_features`; groups are
    /// assigned in monotone non-decreasing order).
    pub placed: Vec<(usize, u32, u32)>,
    /// Number of 32-bit word groups (`words.len() == num_groups * n`).
    pub num_groups: usize,
}

/// Compute the [`CindexPlan`] for `n_buckets`/`one_hot` over `n` objects — the "first
/// pass" of the packer, extracted verbatim so [`pack_cindex`] and the device fill share
/// it. Uses iterator folding — no slice indexing (D-13 / indexing_slicing).
pub(crate) fn plan_cindex(
    n_buckets: &[usize],
    one_hot: &[bool],
    n: usize,
) -> CbResult<CindexPlan> {
    let n_features = n_buckets.len();
    if one_hot.len() != n_features {
        return Err(CbError::LengthMismatch {
            column: "cindex one_hot flags".to_owned(),
            expected: n_features,
            actual: one_hot.len(),
        });
    }
    let mut placed: Vec<(usize, u32, u32)> = Vec::with_capacity(n_features);
    let mut group_index: usize = 0;
    let mut used_bits: u32 = 0;
    for &nb in n_buckets {
        let bits = feature_bits(nb)?;
        // Start a new group when this feature would not fit the current word. `used_bits`
        // and `bits` are each <= 32, so `used_bits + bits` <= 64 — no u32 overflow.
        if used_bits + bits > 32 {
            group_index = group_index.checked_add(1).ok_or_else(|| {
                CbError::OutOfRange("plan_cindex: group index overflows usize".to_owned())
            })?;
            used_bits = 0;
        }
        let shift = used_bits;
        let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
        placed.push((group_index, shift, mask));
        used_bits += bits;
    }
    let num_groups = group_index + 1;

    // The TCFeature descriptor table (serial — `n_features` entries).
    let mut features: Vec<TCFeature> = Vec::with_capacity(n_features);
    for ((&nb, &(group, shift, mask)), &is_one_hot) in
        n_buckets.iter().zip(placed.iter()).zip(one_hot.iter())
    {
        let offset = (group as u64).checked_mul(n as u64).ok_or_else(|| {
            CbError::OutOfRange(format!("plan_cindex: group offset {group} * n ({n}) overflows u64"))
        })?;
        let folds = u32::try_from(nb).map_err(|_| {
            CbError::OutOfRange(format!("plan_cindex: n_buckets ({nb}) exceeds u32 fold count"))
        })?;
        features.push(TCFeature {
            offset,
            mask,
            shift,
            first_fold_index: 0,
            folds,
            one_hot_feature: is_one_hot,
        });
    }
    Ok(CindexPlan {
        features,
        placed,
        num_groups,
    })
}

/// Host replica of [`crate::kernels::read_bin`] — `(words[offset + obj] >> shift) &
/// mask`. Used by the histogram host-reference and the bit-exact oracle to extract a
/// packed bin exactly as the device accessor does (the reference and device then agree
/// cell-for-cell). Pure integer, no device. An out-of-range index yields `0` (the caller
/// guarantees `offset + obj < words.len()`; the `.get()` avoids an indexing panic).
#[allow(dead_code)] // consumed by the `#[cfg(test)]` histogram reference + cindex oracle.
pub(crate) fn read_bin_host(words: &[u32], offset: u64, obj: usize, shift: u32, mask: u32) -> u32 {
    let idx = offset as usize + obj;
    words.get(idx).map(|&w| (w >> shift) & mask).unwrap_or(0)
}

/// Pack the plain feature-major quantized bins `bins` (`bins[feature * n + obj]`, values
/// in `0..n_buckets[feature]`) into the grouped bit-packed cindex + [`TCFeature`] table.
/// Features are grouped GREEDILY: a running word accumulates features until the next
/// feature's bit-width would overflow 32 bits, at which point a new group (a new word
/// column of `n` words) starts. Feature `f` in group `g` gets `offset = g * n` (word
/// base), `shift = cumulative bits of prior features in `g``, `mask = (1 << bits) - 1`.
///
/// `one_hot[f]` records whether feature `f` uses EQUALITY split semantics downstream; it
/// is copied verbatim into `TCFeature.one_hot_feature` and does NOT affect the packing
/// (the stored bin VALUE is identical either way — only the later split TEST differs).
/// Every non-one-hot caller passes an all-`false` slice, which reproduces the pre-SPEC-OH-21
/// bytes exactly. `one_hot.len() != n_features` → [`CbError::LengthMismatch`].
///
/// Every product / word-count that can overflow is `checked_*` → [`CbError::OutOfRange`]
/// (T-10-16); `bins.len() != n_features * n` → [`CbError::LengthMismatch`]. An out-of-range
/// bin (`>= n_buckets[feature]`) surfaces [`CbError::OutOfRange`] BEFORE it is masked into
/// a word (so a malformed bin can never silently truncate into another feature's field).
pub(crate) fn pack_cindex(
    bins: &[u32],
    n_buckets: &[usize],
    one_hot: &[bool],
    n: usize,
) -> CbResult<PackedCindex> {
    let n_features = n_buckets.len();
    if one_hot.len() != n_features {
        return Err(CbError::LengthMismatch {
            column: "cindex one_hot flags".to_owned(),
            expected: n_features,
            actual: one_hot.len(),
        });
    }

    // Length guard: the plain layout is exactly n_features * n cells.
    let stride = n_features.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "pack_cindex: n_features ({n_features}) * n ({n}) overflows usize"
        ))
    })?;
    if bins.len() != stride {
        return Err(CbError::LengthMismatch {
            column: "cindex".to_owned(),
            expected: stride,
            actual: bins.len(),
        });
    }
    if n_features == 0 || n == 0 {
        return Ok(PackedCindex {
            words: Vec::new(),
            features: Vec::new(),
        });
    }

    // Placement + descriptors are pure functions of the bucket counts (the bins never
    // influence the geometry) — shared with the device quantize+pack fast path, which
    // has NO host bins at all (QPACK-01).
    let plan = plan_cindex(n_buckets, one_hot, n)?;
    let CindexPlan {
        features,
        placed,
        num_groups,
    } = plan;

    // Words: one word per object per group.
    let num_words = num_groups.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "pack_cindex: num_groups ({num_groups}) * n ({n}) overflows usize"
        ))
    })?;
    let mut words = vec![0u32; num_words];

    // Second pass (b): OR each feature's masked field into its group's word column, in
    // PARALLEL over the groups. Group columns are DISJOINT `n`-word slices
    // (`par_chunks_mut(n)` yields exactly `num_groups` of them, aligned with
    // `group_ranges` below), and a feature writes ONLY its own group's column, so there
    // is no aliasing; within a group each feature owns a disjoint bit FIELD of the
    // shared word, so the packed words are BIT-IDENTICAL to the former serial loop.
    // The per-element value-range guard is preserved (T-10-16): a bin >= n_buckets
    // would corrupt an adjacent field once masked/shifted — reject, never truncate.
    //
    // `placed` assigns groups in monotone non-decreasing order, so each group's
    // features form one contiguous index range.
    let mut group_ranges: Vec<std::ops::Range<usize>> = vec![0..0; num_groups];
    for (fi, &(group, _, _)) in placed.iter().enumerate() {
        if let Some(r) = group_ranges.get_mut(group) {
            if r.start == r.end {
                *r = fi..fi + 1;
            } else {
                r.end = fi + 1;
            }
        }
    }
    words
        .par_chunks_mut(n)
        .zip(group_ranges.par_iter())
        .try_for_each(|(word_col, range)| -> CbResult<()> {
            for fi in range.clone() {
                let (&nb, &(_, shift, mask)) = match (n_buckets.get(fi), placed.get(fi)) {
                    (Some(nb), Some(p)) => (nb, p),
                    _ => {
                        return Err(CbError::OutOfRange(format!(
                            "pack_cindex: feature index {fi} out of the placement table (internal)"
                        )))
                    }
                };
                let col_start = fi.checked_mul(n).ok_or_else(|| {
                    CbError::OutOfRange(format!(
                        "pack_cindex: feature {fi} * n ({n}) overflows usize"
                    ))
                })?;
                let col_end = col_start.checked_add(n).ok_or_else(|| {
                    CbError::OutOfRange(format!(
                        "pack_cindex: column start ({col_start}) + n ({n}) overflows usize"
                    ))
                })?;
                let bins_chunk = bins.get(col_start..col_end).ok_or_else(|| {
                    CbError::OutOfRange(format!(
                        "pack_cindex: bin column {col_start}..{col_end} out of the {}-cell buffer",
                        bins.len()
                    ))
                })?;
                for (&raw, w) in bins_chunk.iter().zip(word_col.iter_mut()) {
                    if (raw as usize) >= nb {
                        return Err(CbError::OutOfRange(format!(
                            "pack_cindex: bin value {raw} >= n_buckets ({nb})"
                        )));
                    }
                    *w |= (raw & mask) << shift;
                }
            }
            Ok(())
        })?;

    Ok(PackedCindex { words, features })
}

/// QPACK-01: build the packed cindex ON DEVICE from the raw f32 float columns + the
/// per-feature borders — the fused device replacement for host
/// `quantize_feature_major` → [`pack_cindex`] → upload on the float-only fast path.
/// Returns the device handle of the packed word buffer (length
/// `plan.num_groups * n` u32 words), bit-identical to the host pipeline's upload.
///
/// Per feature: upload the f32 column + the f32-narrowed borders, then launch
/// [`crate::kernels::quantize_pack_feature_kernel`] to merge the feature's bit-field
/// into its group's word column. The FIRST feature of each group STOREs (initializing
/// the column — the buffer needs no zero-fill), later features OR into their disjoint
/// fields; the session's single stream orders the launches. The per-column upload
/// handles become garbage after the last launch retires and are dropped here.
///
/// # Preconditions (validated by the caller — `GpuTrainSession::begin_raw`)
/// - `columns.len() == borders.len() == plan.features.len()`, each column length `n`;
/// - every border round-trips f64→f32→f64 exactly (they are f32 midpoints by
///   construction), so the kernel's f32 compare is bit-equivalent to the host's f64
///   compare;
/// - every `borders[f].len() + 1 <= n_buckets[f]` (bin values fit their field).
///
/// # Errors
/// [`CbError::LengthMismatch`] on a column/borders arity mismatch (defense in depth —
/// the caller validates first).
pub(crate) fn fill_packed_cindex_on_device(
    client: &cubecl::client::ComputeClient<crate::SelectedRuntime>,
    columns: &[Vec<f32>],
    borders: &[Vec<f64>],
    plan: &CindexPlan,
    n: usize,
) -> CbResult<cubecl::server::Handle> {
    use cubecl::prelude::*;

    let n_features = plan.features.len();
    if columns.len() != n_features || borders.len() != n_features {
        return Err(CbError::LengthMismatch {
            column: "device quantize+pack columns/borders".to_owned(),
            expected: n_features,
            actual: columns.len().min(borders.len()),
        });
    }
    let num_words = plan.num_groups.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "fill_packed_cindex_on_device: num_groups ({}) * n ({n}) overflows usize",
            plan.num_groups
        ))
    })?;
    let words_h = client.empty(num_words * std::mem::size_of::<u32>());

    // 256-lane cubes, one thread per object (the kernel is a single bounds-guarded
    // pass — no CUBE_COUNT grid-stride, so it also runs on cubecl-cpu).
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
    let num_cubes = u32::try_from(n.div_ceil(256).max(1)).map_err(|_| {
        CbError::OutOfRange(format!(
            "fill_packed_cindex_on_device: cube count for n = {n} exceeds u32"
        ))
    })?;

    let mut prev_group: Option<usize> = None;
    for ((col, bord), &(group, shift, _mask)) in
        columns.iter().zip(borders.iter()).zip(plan.placed.iter())
    {
        if col.len() != n {
            return Err(CbError::LengthMismatch {
                column: "device quantize+pack float column".to_owned(),
                expected: n,
                actual: col.len(),
            });
        }
        // First feature of its group initializes the word column (groups are assigned
        // in monotone non-decreasing order, so "first" is exactly a group transition).
        let init_word = u32::from(prev_group != Some(group));
        prev_group = Some(group);

        let col_h = client.create(cubecl::bytes::Bytes::from_elems(col.clone()));
        let borders_f32: Vec<f32> = bord.iter().map(|&b| b as f32).collect();
        let n_borders = borders_f32.len();
        // A zero-length device read is never issued: an empty border list still
        // launches (bin is constantly 0 and the field must be stored/OR-merged), so
        // give the kernel a 1-element dummy border buffer it will never loop over.
        let borders_h = if n_borders == 0 {
            client.create(cubecl::bytes::Bytes::from_elems(vec![0.0_f32]))
        } else {
            client.create(cubecl::bytes::Bytes::from_elems(borders_f32))
        };
        let group_offset = (group as u64).checked_mul(n as u64).ok_or_else(|| {
            CbError::OutOfRange(format!(
                "fill_packed_cindex_on_device: group offset {group} * n ({n}) overflows u64"
            ))
        })?;
        let group_offset = u32::try_from(group_offset).map_err(|_| {
            CbError::OutOfRange(format!(
                "fill_packed_cindex_on_device: group offset {group_offset} exceeds u32"
            ))
        })?;

        crate::kernels::quantize_pack_feature_kernel::launch::<f32, crate::SelectedRuntime>(
            client,
            CubeCount::Static(num_cubes.max(1), 1, 1),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts(col_h, n) },
            unsafe { ArrayArg::from_raw_parts(borders_h, n_borders.max(1)) },
            unsafe { ArrayArg::from_raw_parts(words_h.clone(), num_words) },
            u32::try_from(n_borders).map_err(|_| {
                CbError::OutOfRange(format!(
                    "fill_packed_cindex_on_device: {n_borders} borders exceed u32"
                ))
            })?,
            group_offset,
            shift,
            init_word,
        );
    }
    Ok(words_h)
}
