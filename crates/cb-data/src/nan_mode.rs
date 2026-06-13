//! `NanMode` sentinel handling + strict `value > border` bin assignment (DATA-04)
//! — the parity-critical "which bin does a value land in" primitive, transcribed
//! from upstream CatBoost.
//!
//! # Source of truth
//!
//! - `catboost/private/libs/options/enums.h:107-111` — `ENanMode { Min, Max,
//!   Forbidden }` (variant order preserved here).
//! - `library/cpp/grid_creator/binarization.cpp` / `quantization.cpp:322-345` —
//!   NanMode sentinel insertion: `Min` prepends `numeric_limits<float>::lowest()`
//!   (== Rust [`f32::MIN`]), `Max` appends `numeric_limits<float>::max()`
//!   (== Rust [`f32::MAX`]); each consumes one border from the budget
//!   (`borderCount - 1`).
//! - `catboost/libs/helpers/.../utils.h:28-49` — bin assignment is the count of
//!   borders **strictly less than** the value (`value > border`); a value that
//!   equals a border lands in the LOWER bin. For `<= 64` borders this is a linear
//!   count; for `> 64` it is a `lowerBound`/`partitionPoint` binary search.
//! - `utils.h:51-66` — a NaN value quantizes to bin `0` under `Min`/`Forbidden`
//!   and to the TOP bin (`borders.len()`) under `Max`.
//!
//! # f32 discipline
//!
//! Feature values and borders are `f32` (CatBoost's storage type). The sentinels
//! are the exact `f32` limits so they round-trip bit-identically against the
//! oracle's stored borders (matching the 02-02 sentinel resolution, A1/A3).

/// The number of borders above which bin assignment switches from a linear
/// count to a binary search (`partition_point`). Mirrors upstream's `<= 64`
/// linear / `> 64` lower-bound threshold (`utils.h:28-49`).
const LINEAR_SCAN_THRESHOLD: usize = 64;

/// How missing (`NaN`) float values are handled during quantization, mirroring
/// upstream `ENanMode` (`enums.h:107-111`). The variant order is preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanMode {
    /// `NaN`s are treated as the smallest value: a [`f32::MIN`] sentinel border is
    /// prepended at index 0, and a `NaN` value quantizes to bin `0`.
    Min,
    /// `NaN`s are treated as the largest value: a [`f32::MAX`] sentinel border is
    /// appended, and a `NaN` value quantizes to the top bin.
    Max,
    /// `NaN`s are forbidden in the learn set; no sentinel is inserted. A `NaN`
    /// encountered at quantization time quantizes to bin `0` (defensive — the
    /// learn-set check that forbids `NaN` lives upstream of this primitive).
    Forbidden,
}

impl NanMode {
    /// Whether this mode prepends the [`f32::MIN`] sentinel at index 0.
    #[must_use]
    pub fn prepends_min_sentinel(self) -> bool {
        matches!(self, NanMode::Min)
    }

    /// Whether this mode appends the [`f32::MAX`] sentinel.
    #[must_use]
    pub fn appends_max_sentinel(self) -> bool {
        matches!(self, NanMode::Max)
    }
}

/// Assign `value` to a bin index over the ascending `borders` slice, using the
/// strict `value > border` rule (`utils.h:28-49`): the bin index is the count of
/// borders strictly less than `value`. A value that exactly equals a border lands
/// in the LOWER bin (the equal border does NOT increment the count). For more
/// than [`LINEAR_SCAN_THRESHOLD`] borders this is computed with a binary search
/// ([`slice::partition_point`]), otherwise with a linear count — both yield the
/// identical index.
///
/// `borders` MUST be sorted ascending (the binarizer guarantees this).
///
/// # Budget invariant (WR-06)
///
/// A bin index is at most `borders.len()`, returned as a `u32`. Float callers
/// guarantee a border set bounded by the per-feature budget (`< 65536`, the
/// `u16` bin cap enforced by the width selector, `utils.h:175-181`), so the cast
/// never truncates a real bin index. A `debug_assert!` documents and checks this
/// at the public boundary so a future caller passing an oversized border set
/// (e.g. a categorical reuse with `> u32::MAX` borders) trips in debug builds
/// rather than silently truncating.
#[must_use]
pub fn bin_of(borders: &[f32], value: f32) -> u32 {
    debug_assert!(
        borders.len() <= u32::MAX as usize,
        "bin_of: border count {} exceeds u32::MAX; the u32 bin index would truncate",
        borders.len()
    );
    // ui32 GetBinFromBorders(borders, value): count of `value > border`.
    let index = if borders.len() <= LINEAR_SCAN_THRESHOLD {
        // for (border : borders) bin += (value > border);  -- equal -> lower bin.
        borders.iter().filter(|&&border| value > border).count()
    } else {
        // LowerBound: first index whose border is NOT < value, i.e. the count of
        // borders strictly less than value (`partition_point(border < value)`).
        borders.partition_point(|&border| border < value)
    };
    // Border counts are bounded by the budget (<= 65535 for float, asserted by
    // the width selector and the debug_assert above), so this cast cannot
    // truncate a real bin index.
    index as u32
}

/// Apply this mode's sentinel to an already-selected ascending border set,
/// returning the borders the quantizer stores. `Min` prepends [`f32::MIN`] at
/// index 0; `Max` appends [`f32::MAX`]; `Forbidden` returns the borders
/// unchanged.
///
/// Each inserted sentinel consumes one border from the budget upstream
/// (`borderCount - 1`); this function only performs the insertion — the caller
/// reserves the budget via [`NanMode::reserved_border_budget`].
#[must_use]
pub fn insert_sentinel(mode: NanMode, borders: &[f32]) -> Vec<f32> {
    match mode {
        NanMode::Min => {
            // Prepend numeric_limits<float>::lowest() == f32::MIN at index 0.
            let mut out = Vec::with_capacity(borders.len() + 1);
            out.push(f32::MIN);
            out.extend_from_slice(borders);
            out
        }
        NanMode::Max => {
            // Append numeric_limits<float>::max() == f32::MAX.
            let mut out = Vec::with_capacity(borders.len() + 1);
            out.extend_from_slice(borders);
            out.push(f32::MAX);
            out
        }
        // Forbidden: no sentinel.
        NanMode::Forbidden => borders.to_vec(),
    }
}

impl NanMode {
    /// The border budget remaining for real (non-sentinel) borders given a total
    /// `border_count`. `Min`/`Max` reserve exactly one for the sentinel
    /// (`borderCount - 1`, saturating at 0); `Forbidden` reserves none.
    #[must_use]
    pub fn reserved_border_budget(self, border_count: usize) -> usize {
        match self {
            NanMode::Min | NanMode::Max => border_count.saturating_sub(1),
            NanMode::Forbidden => border_count,
        }
    }
}

/// The bin a `NaN` value quantizes to over `borders` (already including any
/// sentinel): `0` under `Min`/`Forbidden`, the TOP bin (`borders.len()`) under
/// `Max` (`utils.h:51-66`).
#[must_use]
pub fn nan_bin(mode: NanMode, borders: &[f32]) -> u32 {
    match mode {
        // NaN -> 0 (smallest); the Min sentinel at borders[0] makes 0 the
        // dedicated NaN bin.
        NanMode::Min | NanMode::Forbidden => 0,
        // NaN -> top bin; the Max sentinel at the end makes the last bin the
        // dedicated NaN bin. Same budget invariant as `bin_of` (WR-06 / IN-01):
        // the float border budget keeps the count well below `u32::MAX`.
        NanMode::Max => {
            debug_assert!(
                borders.len() <= u32::MAX as usize,
                "nan_bin: border count {} exceeds u32::MAX; the u32 top-bin index would truncate",
                borders.len()
            );
            borders.len() as u32
        }
    }
}
