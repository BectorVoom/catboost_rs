//! [`QuantizedPool`] — the immutable, columnar (SoA) binned dataset produced by
//! [`crate::quantize`] (DATA-02), plus the zero-widening per-column width enum
//! [`ColumnBins`].
//!
//! # Memory efficiency (first-class constraint)
//!
//! Each binned column stores the narrowest unsigned integer that can hold its
//! bin indices: `u8` for `< 256` borders, `u16` for `< 65536`. Float features are
//! NEVER `u32` (a float feature can have at most `border_count <= 65535` bins);
//! `u32` is reserved for categorical perfect-hash columns whose unique-value
//! count can exceed `65535`. This mirrors upstream `CalcHistogramWidthForBorders`
//! (`catboost/libs/helpers/.../utils.h:175-181`).
//!
//! # Immutable after build (D-03)
//!
//! A [`QuantizedPool`] hands out only read accessors — no mutable scratch. The
//! trainer reuses its OWN scratch buffers; the quantized dataset is a frozen
//! parity artifact once built.
//!
//! # Per-feature SoA (D-11 / D-12)
//!
//! Bin columns, borders, and `NanMode` are stored one entry per feature in
//! parallel `Vec`s, matching the column-by-column layout the binarizer consumes.

use crate::nan_mode::NanMode;

/// The maximum border count representable by a `u8` bin index. A feature with
/// `< 256` borders has at most `256` bins (`0..=255`), so `u8` suffices.
const U8_BORDER_LIMIT: usize = 256;

/// The maximum border count representable by a `u16` bin index, and the hard
/// upper bound for FLOAT features (`utils.h:175-181` asserts `< 65536`).
const U16_BORDER_LIMIT: usize = 65_536;

/// One feature's binned column, stored in the narrowest unsigned integer width
/// that holds every bin index (the zero-widening memory optimization). The width
/// is chosen by [`select_bin_width`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnBins {
    /// `< 256` borders → 8-bit bin indices (float or small categorical).
    U8(Vec<u8>),
    /// `< 65536` borders → 16-bit bin indices (float or mid categorical).
    U16(Vec<u16>),
    /// `>= 65536` uniques → 32-bit bin indices. Reserved for categorical
    /// perfect-hash columns; a float feature can never reach this width.
    U32(Vec<u32>),
}

impl ColumnBins {
    /// Number of bin entries (objects) in this column.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            ColumnBins::U8(v) => v.len(),
            ColumnBins::U16(v) => v.len(),
            ColumnBins::U32(v) => v.len(),
        }
    }

    /// Whether this column holds zero objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bin index at object `row`, widened to `u32` for uniform read-back, or
    /// `None` if `row` is out of range. Read-only (D-03 immutability).
    #[must_use]
    pub fn get(&self, row: usize) -> Option<u32> {
        match self {
            ColumnBins::U8(v) => v.get(row).map(|&b| u32::from(b)),
            ColumnBins::U16(v) => v.get(row).map(|&b| u32::from(b)),
            ColumnBins::U32(v) => v.get(row).copied(),
        }
    }

    /// All bin indices widened to `u32`, for round-trip / comparison.
    #[must_use]
    pub fn to_u32_vec(&self) -> Vec<u32> {
        match self {
            ColumnBins::U8(v) => v.iter().map(|&b| u32::from(b)).collect(),
            ColumnBins::U16(v) => v.iter().map(|&b| u32::from(b)).collect(),
            ColumnBins::U32(v) => v.clone(),
        }
    }
}

/// Whether a column with the given border/bin width belongs to a float feature
/// (which is hard-capped at `u16`) or a categorical feature (which may need
/// `u32`). Used by [`select_bin_width`] to enforce the upstream invariant that a
/// float feature is NEVER `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// A float (numeric) feature — capped at `u16` (`< 65536` borders).
    Float,
    /// A categorical perfect-hash feature — may use `u32` for `>= 65536` uniques.
    Categorical,
}

/// Choose the narrowest [`ColumnBins`] width that can hold a bin index for a
/// feature with `border_count` borders of kind `kind`, transcribing
/// `CalcHistogramWidthForBorders` (`utils.h:175-181`):
///
/// - `< 256` borders → `U8`
/// - `< 65536` borders → `U16`
/// - `>= 65536` → `U32` (categorical only)
///
/// A FLOAT feature with `>= 65536` borders is rejected with `None` (upstream
/// `Y_ASSERT(borderCount < 65536)`): a float feature's bin index can never need
/// `u32`. Returning `None` lets the fallible quantize driver surface a
/// `CbError` rather than panic (threat T-02-09).
#[must_use]
pub fn select_bin_width(border_count: usize, kind: FeatureKind) -> Option<ColumnBins> {
    if border_count < U8_BORDER_LIMIT {
        Some(ColumnBins::U8(Vec::new()))
    } else if border_count < U16_BORDER_LIMIT {
        Some(ColumnBins::U16(Vec::new()))
    } else {
        match kind {
            // Float is hard-capped at u16 (utils.h asserts < 65536).
            FeatureKind::Float => None,
            // Categorical perfect-hash may exceed u16.
            FeatureKind::Categorical => Some(ColumnBins::U32(Vec::new())),
        }
    }
}

/// Build a [`ColumnBins`] of the width selected for `border_count`/`kind`, filled
/// from the `u32` bin indices in `bins`. Returns `None` if a float feature would
/// need `> u16` (mirrors [`select_bin_width`]). Truncation is impossible: every
/// `bin` is `<= border_count`, which the chosen width can hold.
#[must_use]
pub fn pack_bins(border_count: usize, kind: FeatureKind, bins: &[u32]) -> Option<ColumnBins> {
    match select_bin_width(border_count, kind)? {
        ColumnBins::U8(_) => Some(ColumnBins::U8(bins.iter().map(|&b| b as u8).collect())),
        ColumnBins::U16(_) => Some(ColumnBins::U16(bins.iter().map(|&b| b as u16).collect())),
        ColumnBins::U32(_) => Some(ColumnBins::U32(bins.to_vec())),
    }
}

/// The immutable quantized dataset: per-float-feature binned columns + the
/// borders and [`NanMode`] each was quantized with (DATA-02). Built by
/// [`crate::quantize`]; read-only thereafter (D-03).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedPool {
    /// Number of objects (rows).
    n_rows: usize,
    /// One binned column per float feature, in feature order (SoA, D-12).
    float_bins: Vec<ColumnBins>,
    /// The ascending borders (including any NanMode sentinel) used to bin each
    /// float feature, parallel to `float_bins`.
    float_borders: Vec<Vec<f32>>,
    /// The [`NanMode`] each float feature was quantized under, parallel to
    /// `float_bins`.
    float_nan_modes: Vec<NanMode>,
}

impl QuantizedPool {
    /// Construct an immutable [`QuantizedPool`] from per-feature binned columns,
    /// borders, and modes. The three `Vec`s MUST be the same length (one entry
    /// per float feature); the caller ([`crate::quantize`]) guarantees this.
    #[must_use]
    pub(crate) fn new(
        n_rows: usize,
        float_bins: Vec<ColumnBins>,
        float_borders: Vec<Vec<f32>>,
        float_nan_modes: Vec<NanMode>,
    ) -> Self {
        Self {
            n_rows,
            float_bins,
            float_borders,
            float_nan_modes,
        }
    }

    /// Number of objects (rows).
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of float feature columns.
    #[must_use]
    pub fn n_float_features(&self) -> usize {
        self.float_bins.len()
    }

    /// The binned column for float feature `index`, or `None` if out of range.
    #[must_use]
    pub fn float_bins(&self, index: usize) -> Option<&ColumnBins> {
        self.float_bins.get(index)
    }

    /// The borders (including any sentinel) for float feature `index`, or `None`.
    #[must_use]
    pub fn float_borders(&self, index: usize) -> Option<&[f32]> {
        self.float_borders.get(index).map(Vec::as_slice)
    }

    /// The [`NanMode`] float feature `index` was quantized under, or `None`.
    #[must_use]
    pub fn float_nan_mode(&self, index: usize) -> Option<NanMode> {
        self.float_nan_modes.get(index).copied()
    }
}
