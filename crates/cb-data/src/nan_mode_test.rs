//! Unit tests for [`crate::nan_mode`] — strict `value > border` bin assignment,
//! NanMode sentinel insertion, and NaN bin placement (DATA-04). Source/test
//! separation: this is the dedicated sibling test file (CLAUDE.md).

use crate::nan_mode::{bin_of, insert_sentinel, nan_bin, NanMode};

/// The strict `>` rule: a value exactly equal to a border lands in the LOWER bin
/// (the equal border is NOT counted), and a value just above a border crosses
/// into the next bin.
#[test]
fn value_equal_to_border_lands_in_lower_bin() {
    let borders = [1.0_f32, 2.0, 3.0];
    // value == border[0] -> NOT counted -> bin 0 (lower bin).
    assert_eq!(bin_of(&borders, 1.0), 0);
    // value == border[1] -> bins below it: only border[0]=1.0 < 2.0 -> bin 1.
    assert_eq!(bin_of(&borders, 2.0), 1);
    // value == border[2] -> border[0],border[1] < 3.0 -> bin 2.
    assert_eq!(bin_of(&borders, 3.0), 2);
    // just above border[0] -> bin 1.
    assert_eq!(bin_of(&borders, 1.000001), 1);
    // below all borders -> bin 0.
    assert_eq!(bin_of(&borders, 0.5), 0);
    // above all borders -> top bin (== len).
    assert_eq!(bin_of(&borders, 3.5), 3);
}

/// Bin assignment must agree between the linear path (`<= 64` borders) and the
/// binary-search path (`> 64` borders): build a 100-border ramp and probe an
/// exact-equality and a between-borders value.
#[test]
fn binary_search_path_matches_strict_gt() {
    let borders: Vec<f32> = (0..100).map(|i| i as f32).collect();
    assert!(borders.len() > 64);
    // value == borders[50] -> lower bin -> 50.
    assert_eq!(bin_of(&borders, 50.0), 50);
    // value between borders[50] and borders[51] -> 51.
    assert_eq!(bin_of(&borders, 50.5), 51);
    // below all -> 0.
    assert_eq!(bin_of(&borders, -1.0), 0);
}

/// Min prepends `f32::MIN` at index 0; the original borders follow.
#[test]
fn min_prepends_f32_min_sentinel() {
    let borders = [1.0_f32, 2.0];
    let out = insert_sentinel(NanMode::Min, &borders);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0], f32::MIN);
    assert_eq!(&out[1..], &[1.0_f32, 2.0]);
}

/// Max appends `f32::MAX` at the end; the original borders precede it.
#[test]
fn max_appends_f32_max_sentinel() {
    let borders = [1.0_f32, 2.0];
    let out = insert_sentinel(NanMode::Max, &borders);
    assert_eq!(out.len(), 3);
    assert_eq!(&out[..2], &[1.0_f32, 2.0]);
    assert_eq!(out[2], f32::MAX);
}

/// Forbidden inserts no sentinel.
#[test]
fn forbidden_inserts_no_sentinel() {
    let borders = [1.0_f32, 2.0];
    let out = insert_sentinel(NanMode::Forbidden, &borders);
    assert_eq!(out, vec![1.0_f32, 2.0]);
}

/// A NaN quantizes to bin 0 under Min and Forbidden, and to the top bin under
/// Max.
#[test]
fn nan_goes_to_bin_zero_for_min_and_forbidden_top_for_max() {
    // Borders already carry the sentinel for Min/Max.
    let min_borders = insert_sentinel(NanMode::Min, &[1.0_f32, 2.0]);
    let max_borders = insert_sentinel(NanMode::Max, &[1.0_f32, 2.0]);
    let forbidden_borders = [1.0_f32, 2.0];

    assert_eq!(nan_bin(NanMode::Min, &min_borders), 0);
    assert_eq!(nan_bin(NanMode::Forbidden, &forbidden_borders), 0);
    // top bin == borders.len() (3 borders -> bin 3).
    assert_eq!(nan_bin(NanMode::Max, &max_borders), max_borders.len() as u32);
}

/// Min/Max reserve one border for the sentinel; Forbidden reserves none.
#[test]
fn reserved_budget_subtracts_one_for_sentinel_modes() {
    assert_eq!(NanMode::Min.reserved_border_budget(254), 253);
    assert_eq!(NanMode::Max.reserved_border_budget(254), 253);
    assert_eq!(NanMode::Forbidden.reserved_border_budget(254), 254);
    // saturating at 0.
    assert_eq!(NanMode::Min.reserved_border_budget(0), 0);
}
