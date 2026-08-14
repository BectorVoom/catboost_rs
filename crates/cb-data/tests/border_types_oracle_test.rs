//! Per-feature border oracle for EVERY `feature_border_type`
//! (`EBorderSelectionType`): proves [`cb_data::select_borders`] reproduces
//! upstream CatBoost's standalone quantization borders for all seven binarizers.
//!
//! The expected borders are the RAW standalone quantization output
//! (`Pool.quantize(border_count, feature_border_type, nan_mode)` then
//! `save_quantization_borders()`), frozen under
//! `cb-oracle/fixtures/border_types/` by
//! `generator/gen_border_type_fixtures.py`.
//!
//! # Why the corpora look the way they do
//!
//! Every cell runs UNDER BUDGET (`border_count` below the column's unique-value
//! count). At or above saturation all seven binarizers collapse onto the same
//! answer, so an at-budget fixture would pass for a wrong implementation.
//!
//! The `borders_runs` corpus (uneven duplicate runs) is the only one where
//! `GreedyMinEntropy` differs from `GreedyLogSum` and `MinEntropy` from
//! `MaxLogSum` — on evenly-spread data those pairs are provably byte-identical
//! (see [`cb_data::PenaltyType`]). Without it the fixture would be vacuous for
//! the two MinEntropy-penalty binarizers.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::{select_borders, EBorderSelectionType};
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from cb-data's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_x(dataset: &str) -> Array2<f64> {
    // The SPEC-OH-31 byte-identity corpus owns its own fixture directory rather
    // than living under inputs/ (mirrors the generator's EXTERNAL_INPUTS).
    let rel = if dataset == "float_only_byte_identity" {
        format!("{dataset}/inputs/X.npy")
    } else {
        format!("inputs/{dataset}/X.npy")
    };
    read_npy(fixture(&rel))
        .unwrap_or_else(|e| panic!("{dataset}/X.npy must load as 2-D f64: {e:?}"))
}

/// The `(dataset, border_count)` cells frozen by the generator. Every cell is
/// run for all seven border types.
const CELLS: &[(&str, usize)] = &[
    ("numeric_tiny", 8),
    ("numeric_tiny", 32),
    ("numeric_nan", 8),
    ("borders_dense", 16),
    ("borders_dense", 128),
    ("borders_runs", 8),
    ("borders_runs", 16),
    ("borders_runs", 32),
    // The SPEC-OH-31 byte-identity corpus at the exact budget its own fixture
    // quantizes with (512 unique values -> 32 borders). This cell is the reason
    // that frozen `.cbm` baseline had to be re-captured: its budget BINDS, so it
    // was serialized with the buggy greedy tie-break. Pinning catboost's borders
    // for it makes the re-baseline a move TOWARD upstream, not a silent drift.
    ("float_only_byte_identity", 32),
];

/// Feature indices whose oracle borders begin with the NanMode(`Min`)
/// `f32::MIN` sentinel, per dataset. Only `numeric_nan` has a NaN column, and
/// only its feature 0.
fn nan_sentinel_features(dataset: &str) -> &'static [usize] {
    match dataset {
        "numeric_nan" => &[0],
        _ => &[],
    }
}

/// Compare one (dataset, border_count, border_type) cell against its fixture,
/// APPENDING a one-line description of every diverging feature to `failures`
/// rather than panicking. Reporting the whole matrix at once is what makes a
/// systematic divergence (one algorithm, or one regime) legible; failing on the
/// first cell hides the pattern.
fn check_cell(
    dataset: &str,
    border_count: usize,
    border_type: EBorderSelectionType,
    failures: &mut Vec<String>,
) {
    let stem = format!("{dataset}.bc{border_count}.{}", border_type.as_str());
    let x = load_x(dataset);
    let expected_flat =
        load_f64_vec(&fixture(&format!("border_types/{stem}.borders.npy"))).unwrap();
    let per_feature =
        load_f64_vec(&fixture(&format!("border_types/{stem}.borders_per_feature.npy"))).unwrap();

    let n_features = x.ncols();
    assert_eq!(
        per_feature.len(),
        n_features,
        "{stem}: per-feature count vector must have one entry per feature"
    );

    let sentinel_features = nan_sentinel_features(dataset);
    let mut offset = 0usize;
    for (fi, &count_f64) in per_feature.iter().enumerate() {
        let count = count_f64 as usize;
        let expected = &expected_flat[offset..offset + count];
        offset += count;

        let column: Vec<f64> = x.column(fi).to_vec();
        let nan_sentinel = sentinel_features.contains(&fi);
        let actual = select_borders(&column, border_count, border_type, nan_sentinel);

        if actual.len() != expected.len() {
            failures.push(format!(
                "{stem} f{fi}: border COUNT {} != oracle {}",
                actual.len(),
                expected.len()
            ));
            continue;
        }

        // `save_quantization_borders()` writes the fixture as TEXT with ~10
        // significant digits. For ordinary borders that rounding is far inside
        // the parity tolerance, but the NanMode sentinel is `f32::MIN`
        // (-3.4028235e38), where the last printed digit is worth ~1e29 in
        // ABSOLUTE terms — so an absolute-tolerance comparison can never pass on
        // it. The sentinel is an exact known constant rather than a computed
        // quantity, so assert it as one and gate the remaining borders normally.
        let (expected_rest, actual_rest) = if nan_sentinel {
            let sentinel = f64::from(f32::MIN);
            match (expected.first(), actual.first()) {
                (Some(&e), Some(&a)) => {
                    if a != sentinel {
                        failures.push(format!(
                            "{stem} f{fi}: border 0 must be the exact f32::MIN NanMode \
                             sentinel {sentinel:e}, got {a:e}"
                        ));
                    }
                    // The oracle's own sentinel must round-trip to the same f32.
                    if (e as f32) != f32::MIN {
                        failures.push(format!(
                            "{stem} f{fi}: oracle border 0 {e:e} is not the f32::MIN sentinel"
                        ));
                    }
                }
                _ => failures.push(format!("{stem} f{fi}: expected a sentinel border")),
            }
            (&expected[1..], &actual[1..])
        } else {
            (expected, &actual[..])
        };

        if let Err(e) = compare_stage(Stage::Borders, expected_rest, actual_rest) {
            failures.push(format!("{stem} f{fi}: {e:?}"));
        }
    }

    assert_eq!(
        offset,
        expected_flat.len(),
        "{stem}: consumed every oracle border"
    );
}

/// Run every frozen cell for one border type, reporting the full failure matrix.
fn check_type(border_type: EBorderSelectionType) {
    let mut failures: Vec<String> = Vec::new();
    for &(dataset, border_count) in CELLS {
        check_cell(dataset, border_count, border_type, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "{} diverged from the catboost 1.2.10 oracle in {} cell(s):\n  {}",
        border_type.as_str(),
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn median_borders_match_oracle() {
    check_type(EBorderSelectionType::Median);
}

#[test]
fn greedy_log_sum_borders_match_oracle() {
    check_type(EBorderSelectionType::GreedyLogSum);
}

#[test]
fn uniform_and_quantiles_borders_match_oracle() {
    check_type(EBorderSelectionType::UniformAndQuantiles);
}

#[test]
fn min_entropy_borders_match_oracle() {
    check_type(EBorderSelectionType::MinEntropy);
}

#[test]
fn max_log_sum_borders_match_oracle() {
    check_type(EBorderSelectionType::MaxLogSum);
}

#[test]
fn uniform_borders_match_oracle() {
    check_type(EBorderSelectionType::Uniform);
}

#[test]
fn greedy_min_entropy_borders_match_oracle() {
    check_type(EBorderSelectionType::GreedyMinEntropy);
}

/// The `GreedyLogSum` dispatch through the new entry point must be BYTE-identical
/// to the pre-existing dedicated entry, so adding `feature_border_type` cannot
/// have perturbed the default fit path.
#[test]
fn greedy_log_sum_dispatch_is_byte_identical_to_the_dedicated_entry() {
    for &(dataset, border_count) in CELLS {
        let x = load_x(dataset);
        let sentinel_features = nan_sentinel_features(dataset);
        for fi in 0..x.ncols() {
            let column: Vec<f64> = x.column(fi).to_vec();
            let nan_sentinel = sentinel_features.contains(&fi);
            let via_dispatch = select_borders(
                &column,
                border_count,
                EBorderSelectionType::GreedyLogSum,
                nan_sentinel,
            );
            let direct =
                cb_data::select_borders_greedy_logsum(&column, border_count, nan_sentinel);
            assert_eq!(
                via_dispatch, direct,
                "{dataset} bc{border_count} feature {fi}: GreedyLogSum dispatch diverged \
                 from the dedicated entry"
            );
        }
    }
}

/// Every legal token round-trips through parse/as_str, and an unknown token is
/// rejected rather than silently defaulting.
#[test]
fn border_selection_type_parses_every_legal_token_and_rejects_others() {
    for ty in EBorderSelectionType::all() {
        assert_eq!(
            EBorderSelectionType::parse(ty.as_str()),
            Some(ty),
            "{} must round-trip",
            ty.as_str()
        );
    }
    assert_eq!(EBorderSelectionType::parse("greedylogsum"), None);
    assert_eq!(EBorderSelectionType::parse("ZzBogusValue"), None);
}
