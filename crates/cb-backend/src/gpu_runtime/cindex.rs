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

/// Per-feature bit-packed cindex descriptor (upstream `TCFeature`). `offset` is the WORD
/// base of the feature's group ([`crate::kernels::read_bin`] indexes `cindex[offset +
/// obj]`); `shift`/`mask` extract the feature's field from the shared word. `first_fold_index`
/// / `folds` describe the feature's border-fold span (the bin->border join is unchanged);
/// `one_hot_feature` selects EQUALITY (`== value`) vs THRESHOLD (`> bin`) split semantics
/// downstream — the extracted bin VALUE is identical either way, only the split test
/// differs (routed by the consumer, not here).
//
// `first_fold_index` / `folds` / `one_hot_feature` are the FROZEN descriptor contract
// (the plan's `TCFeature` field set) — carried for the multi-group fold offset and the
// one-hot split routing later phases consume; `#[allow(dead_code)]` keeps the default
// build warning-free until those consumers land (the read_bin consumers use only
// offset/shift/mask via [`PackedCindex::device_arrays`]).
#[allow(dead_code)]
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
    pub first_fold_index: u32,
    /// Number of border folds (buckets) this feature spans.
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
    /// Device-ready per-feature `(offsets, shifts, masks)` `u32` arrays for
    /// [`crate::kernels::read_bin`]. `TCFeature.offset` is checked-cast to `u32` (the
    /// device array index type); an offset that overflows `u32` surfaces
    /// [`CbError::OutOfRange`] (T-10-16 — no unguarded index reaches the device).
    pub fn device_arrays(&self) -> CbResult<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let mut offsets = Vec::with_capacity(self.features.len());
        let mut shifts = Vec::with_capacity(self.features.len());
        let mut masks = Vec::with_capacity(self.features.len());
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
        }
        Ok((offsets, shifts, masks))
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
/// Every product / word-count that can overflow is `checked_*` → [`CbError::OutOfRange`]
/// (T-10-16); `bins.len() != n_features * n` → [`CbError::LengthMismatch`]. An out-of-range
/// bin (`>= n_buckets[feature]`) surfaces [`CbError::OutOfRange`] BEFORE it is masked into
/// a word (so a malformed bin can never silently truncate into another feature's field).
pub(crate) fn pack_cindex(
    bins: &[u32],
    n_buckets: &[usize],
    n: usize,
) -> CbResult<PackedCindex> {
    let n_features = n_buckets.len();

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

    // First pass: place each feature into a group and record (group, shift, mask). Uses
    // iterator folding — no slice indexing (D-13 / indexing_slicing).
    let mut placed: Vec<(usize, u32, u32)> = Vec::with_capacity(n_features);
    let mut group_index: usize = 0;
    let mut used_bits: u32 = 0;
    for &nb in n_buckets {
        let bits = feature_bits(nb)?;
        // Start a new group when this feature would not fit the current word. `used_bits`
        // and `bits` are each <= 32, so `used_bits + bits` <= 64 — no u32 overflow.
        if used_bits + bits > 32 {
            group_index = group_index.checked_add(1).ok_or_else(|| {
                CbError::OutOfRange("pack_cindex: group index overflows usize".to_owned())
            })?;
            used_bits = 0;
        }
        let shift = used_bits;
        let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
        placed.push((group_index, shift, mask));
        used_bits += bits;
    }
    let num_groups = group_index + 1;

    // Words: one word per object per group.
    let num_words = num_groups.checked_mul(n).ok_or_else(|| {
        CbError::OutOfRange(format!(
            "pack_cindex: num_groups ({num_groups}) * n ({n}) overflows usize"
        ))
    })?;
    let mut words = vec![0u32; num_words];

    // Second pass: OR each feature's masked field into its group's word column, and emit
    // the TCFeature descriptor. `chunks_exact(n)` walks each feature's plain bin column
    // (bins.len() == n_features * n, n > 0, so exactly n_features chunks, no remainder);
    // `zip` keeps the bucket count + placement aligned — no slice indexing.
    let mut features: Vec<TCFeature> = Vec::with_capacity(n_features);
    for ((&nb, &(group, shift, mask)), bins_chunk) in
        n_buckets.iter().zip(placed.iter()).zip(bins.chunks_exact(n))
    {
        let group_base = group.checked_mul(n).ok_or_else(|| {
            CbError::OutOfRange(format!("pack_cindex: group {group} * n ({n}) overflows usize"))
        })?;
        let group_end = group_base.checked_add(n).ok_or_else(|| {
            CbError::OutOfRange(format!(
                "pack_cindex: group_base ({group_base}) + n ({n}) overflows usize"
            ))
        })?;
        let word_col = words.get_mut(group_base..group_end).ok_or_else(|| {
            CbError::OutOfRange(format!(
                "pack_cindex: word column {group_base}..{group_end} out of the {num_words}-word buffer"
            ))
        })?;
        for (&raw, w) in bins_chunk.iter().zip(word_col.iter_mut()) {
            // Value-range guard (T-10-16): a bin >= n_buckets would corrupt an adjacent
            // field once masked/shifted — reject it here rather than silently truncate.
            if (raw as usize) >= nb {
                return Err(CbError::OutOfRange(format!(
                    "pack_cindex: bin value {raw} >= n_buckets ({nb})"
                )));
            }
            *w |= (raw & mask) << shift;
        }
        let offset = (group as u64).checked_mul(n as u64).ok_or_else(|| {
            CbError::OutOfRange(format!("pack_cindex: group offset {group} * n ({n}) overflows u64"))
        })?;
        let folds = u32::try_from(nb).map_err(|_| {
            CbError::OutOfRange(format!("pack_cindex: n_buckets ({nb}) exceeds u32 fold count"))
        })?;
        features.push(TCFeature {
            offset,
            mask,
            shift,
            first_fold_index: 0,
            folds,
            one_hot_feature: false,
        });
    }

    Ok(PackedCindex { words, features })
}
