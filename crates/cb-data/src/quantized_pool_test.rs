//! Unit tests for [`crate::quantized_pool`] — width selection (`u8`/`u16`/`u32`
//! + float cap) and lossless build→read round-trip per width arm (DATA-02), plus
//! a small end-to-end [`crate::Pool::quantize`] sanity check. Dedicated sibling
//! test file (CLAUDE.md source/test separation).

use crate::ingest::{IngestSource, OwnedColumns};
use crate::nan_mode::NanMode;
use crate::quantize::QuantizeParams;
use crate::quantized_pool::{pack_bins, select_bin_width, ColumnBins, FeatureKind};

/// `< 256` borders → U8; `< 65536` → U16; categorical `>= 65536` → U32.
#[test]
fn width_selection_picks_narrowest_arm() {
    // float, < 256 -> U8
    assert!(matches!(
        select_bin_width(255, FeatureKind::Float),
        Some(ColumnBins::U8(_))
    ));
    // float, exactly 256 -> U16 (>= 256)
    assert!(matches!(
        select_bin_width(256, FeatureKind::Float),
        Some(ColumnBins::U16(_))
    ));
    // float, < 65536 -> U16
    assert!(matches!(
        select_bin_width(65_535, FeatureKind::Float),
        Some(ColumnBins::U16(_))
    ));
    // categorical, >= 65536 -> U32
    assert!(matches!(
        select_bin_width(65_536, FeatureKind::Categorical),
        Some(ColumnBins::U32(_))
    ));
    assert!(matches!(
        select_bin_width(200_000, FeatureKind::Categorical),
        Some(ColumnBins::U32(_))
    ));
}

/// A FLOAT feature is NEVER u32: `>= 65536` borders on a float feature is
/// rejected (`None`), so the fallible driver can surface a CbError instead of
/// overflowing (threat T-02-09).
#[test]
fn float_feature_rejects_u32_width() {
    assert!(select_bin_width(65_536, FeatureKind::Float).is_none());
    assert!(select_bin_width(1_000_000, FeatureKind::Float).is_none());
}

/// build→read round-trip is lossless for each width arm.
#[test]
fn pack_bins_round_trips_losslessly_per_arm() {
    // U8 arm.
    let u8_bins = [0u32, 1, 255];
    let packed = pack_bins(255, FeatureKind::Float, &u8_bins).unwrap();
    assert!(matches!(packed, ColumnBins::U8(_)));
    assert_eq!(packed.to_u32_vec(), u8_bins);
    assert_eq!(packed.get(2), Some(255));

    // U16 arm.
    let u16_bins = [0u32, 256, 65_535];
    let packed = pack_bins(65_535, FeatureKind::Float, &u16_bins).unwrap();
    assert!(matches!(packed, ColumnBins::U16(_)));
    assert_eq!(packed.to_u32_vec(), u16_bins);
    assert_eq!(packed.get(2), Some(65_535));

    // U32 arm (categorical).
    let u32_bins = [0u32, 65_536, 1_000_000];
    let packed = pack_bins(70_000, FeatureKind::Categorical, &u32_bins).unwrap();
    assert!(matches!(packed, ColumnBins::U32(_)));
    assert_eq!(packed.to_u32_vec(), u32_bins);
    assert_eq!(packed.get(2), Some(1_000_000));
}

/// `ColumnBins::get` returns `None` past the end (no panic / no indexing).
#[test]
fn column_bins_get_out_of_range_is_none() {
    let packed = ColumnBins::U8(vec![1, 2, 3]);
    assert_eq!(packed.len(), 3);
    assert!(!packed.is_empty());
    assert_eq!(packed.get(3), None);
    assert!(ColumnBins::U16(Vec::new()).is_empty());
}

/// End-to-end: a 2-feature float Pool quantizes; bins respect strict `value >
/// border` and stay within `[0, n_borders]`, stored in a u8 column.
#[test]
fn quantize_small_pool_produces_strict_gt_bins() {
    // Feature 0: monotone column 0..5 (NaN-free -> Forbidden, no sentinel).
    let f0 = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let pool = OwnedColumns::new(vec![f0], vec![0.0, 0.0, 0.0, 0.0, 0.0])
        .into_pool()
        .unwrap();

    let qp = pool.quantize(&QuantizeParams::default()).unwrap();
    assert_eq!(qp.n_rows(), 5);
    assert_eq!(qp.n_float_features(), 1);
    assert_eq!(qp.float_nan_mode(0), Some(NanMode::Forbidden));

    let borders = qp.float_borders(0).unwrap();
    let bins = qp.float_bins(0).unwrap();
    // u8 width for a tiny feature.
    assert!(matches!(bins, ColumnBins::U8(_)));
    // Every bin in [0, n_borders]; strictly nondecreasing for a sorted column.
    let read = bins.to_u32_vec();
    assert_eq!(read.len(), 5);
    for &b in &read {
        assert!(b as usize <= borders.len());
    }
    for pair in read.windows(2) {
        assert!(pair[1] >= pair[0], "monotone column -> nondecreasing bins");
    }
}

/// A NaN-containing column under Min: the NaN object lands in bin 0 and the
/// borders carry the f32::MIN sentinel at index 0.
#[test]
fn quantize_nan_column_min_places_nan_in_bin_zero() {
    let f0 = vec![1.0_f64, f64::NAN, 2.0, 3.0, f64::NAN, 4.0];
    let pool = OwnedColumns::new(vec![f0], vec![0.0; 6]).into_pool().unwrap();
    let qp = pool.quantize(&QuantizeParams::default()).unwrap();

    assert_eq!(qp.float_nan_mode(0), Some(NanMode::Min));
    let borders = qp.float_borders(0).unwrap();
    assert_eq!(borders[0], f32::MIN, "Min sentinel at index 0");

    let read = qp.float_bins(0).unwrap().to_u32_vec();
    // NaN objects are at indices 1 and 4 -> bin 0.
    assert_eq!(read[1], 0);
    assert_eq!(read[4], 0);
}
