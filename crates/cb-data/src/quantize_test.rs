//! Unit tests for the quantization driver ([`crate::quantize`]).
//!
//! Source/test separation (CLAUDE.md): these live in a dedicated `*_test.rs`
//! file declared as `mod quantize_test;` in `lib.rs`, never a `#[cfg(test)]`
//! block inside `quantize.rs`.

use crate::ingest::{IngestSource, OwnedColumns};
use crate::{ColumnBins, NanMode, QuantizeParams};

/// Extract a column's bins as `Vec<u32>` regardless of the packed width.
fn bins_u32(column: &ColumnBins) -> Vec<u32> {
    match column {
        ColumnBins::U8(v) => v.iter().map(|&b| u32::from(b)).collect(),
        ColumnBins::U16(v) => v.iter().map(|&b| u32::from(b)).collect(),
        ColumnBins::U32(v) => v.clone(),
    }
}

/// CR-01 regression: under `nan_mode = Max`, a `NaN`-bearing float column must
/// give `NaN` a DEDICATED top bin (the `f32::MAX` sentinel must be appended), so
/// the largest finite values do NOT collide with `NaN` in the same bin.
#[test]
fn max_nan_mode_gives_nan_its_own_top_bin() {
    // A single feature column with several distinct finite values plus a NaN.
    // The finite values exercise multiple real bins; the NaN must land strictly
    // above the largest finite value's bin.
    let column = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, f64::NAN, 6.0, 7.0, 8.0];
    let label = vec![0.0_f64; column.len()];
    let pool = OwnedColumns::new(vec![column.clone()], label)
        .into_pool()
        .expect("equal-length columns ingest");

    let params = QuantizeParams {
        border_count: 254,
        nan_mode: NanMode::Max,
    };
    let qp = pool.quantize(&params).expect("Max-mode quantization succeeds");

    // The feature contains a NaN, so the resolved per-feature mode is Max.
    assert_eq!(qp.float_nan_mode(0), Some(NanMode::Max));

    // The stored borders must end with the f32::MAX sentinel (CR-01: previously
    // never appended).
    let borders = qp.float_borders(0).expect("feature 0 borders present");
    assert_eq!(
        borders.last().copied(),
        Some(f32::MAX),
        "Max mode must append the f32::MAX sentinel as the final border"
    );

    let bins = bins_u32(qp.float_bins(0).expect("feature 0 bins present"));
    let nan_bin = bins[5]; // row index of the NaN value above.

    // The NaN must occupy the TOP bin == borders.len() (one above the highest
    // real border), and crucially NOT share a bin with any finite value.
    let top_bin = borders.len() as u32;
    assert_eq!(
        nan_bin, top_bin,
        "NaN must land in the dedicated top bin {top_bin}, got {nan_bin}"
    );

    // No finite value may reach the NaN's bin: every non-NaN row must be in a
    // strictly lower bin than the NaN. This is the exact collision CR-01 fixes.
    for (row, &value) in column.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        assert!(
            bins[row] < nan_bin,
            "finite value at row {row} (={value}) landed in bin {} but must be \
             strictly below the NaN top bin {nan_bin}",
            bins[row]
        );
    }
}

/// The default (`Min`) path is unchanged by the CR-01 fix: NaN -> bin 0, the
/// Min sentinel is at index 0, and no Max sentinel is appended.
#[test]
fn min_nan_mode_unchanged_after_max_wiring() {
    let column = vec![1.0_f64, 2.0, f64::NAN, 3.0, 4.0];
    let label = vec![0.0_f64; column.len()];
    let pool = OwnedColumns::new(vec![column], label)
        .into_pool()
        .expect("equal-length columns ingest");

    let qp = pool
        .quantize(&QuantizeParams::default())
        .expect("Min-mode quantization succeeds");

    assert_eq!(qp.float_nan_mode(0), Some(NanMode::Min));
    let borders = qp.float_borders(0).expect("feature 0 borders present");
    assert_eq!(borders.first().copied(), Some(f32::MIN), "Min sentinel at 0");
    assert_ne!(
        borders.last().copied(),
        Some(f32::MAX),
        "Min mode must NOT append the Max sentinel"
    );

    let bins = bins_u32(qp.float_bins(0).expect("feature 0 bins present"));
    assert_eq!(bins[2], 0, "NaN row quantizes to bin 0 under Min");
}
