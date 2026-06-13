//! Per-feature border oracle: proves the Rust GreedyLogSum binarizer
//! ([`cb_data::select_borders_greedy_logsum`]) reproduces upstream CatBoost's
//! standalone quantization borders to <= 1e-5 on the frozen `numeric_tiny` and
//! `numeric_nan` corpora (DATA-03).
//!
//! The expected borders are the RAW standalone GreedyLogSum quantization output
//! (`Pool.quantize().save_quantization_borders()`), committed under
//! `cb-oracle/fixtures/borders_quant/`. The flat `<dataset>.borders.npy` is
//! split per feature by `<dataset>.borders_per_feature.npy`, and each per-feature
//! slice is gated with `compare_stage(Stage::Borders, ...)`.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors `per_stage_oracle_test.rs:9`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::select_borders_greedy_logsum;
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

/// Load a 2-D `X.npy` input matrix (`n_rows x n_features`, f64).
fn load_x(dataset: &str) -> Array2<f64> {
    read_npy(fixture(&format!("inputs/{dataset}/X.npy")))
        .unwrap_or_else(|e| panic!("{dataset}/X.npy must load as 2-D f64: {e:?}"))
}

/// The border count budget recorded in `borders_quant/config.json` (A2): the
/// catboost 1.2.10 default `border_count=254`.
const BORDER_COUNT: usize = 254;

/// Compare per-feature Rust borders against the committed standalone oracle for
/// one dataset. `nan_sentinel_features` lists the feature indices whose oracle
/// borders begin with the NanMode `f32::MIN` sentinel (so the Rust call must
/// prepend it for those features).
fn check_dataset(dataset: &str, nan_sentinel_features: &[usize]) {
    let x = load_x(dataset);
    let expected_flat =
        load_f64_vec(&fixture(&format!("borders_quant/{dataset}.borders.npy"))).unwrap();
    let per_feature =
        load_f64_vec(&fixture(&format!("borders_quant/{dataset}.borders_per_feature.npy"))).unwrap();

    let n_features = x.ncols();
    assert_eq!(
        per_feature.len(),
        n_features,
        "{dataset}: per-feature count vector must have one entry per feature"
    );

    let mut offset = 0usize;
    for (fi, &count_f64) in per_feature.iter().enumerate() {
        let count = count_f64 as usize;
        let expected = &expected_flat[offset..offset + count];
        offset += count;

        // Extract feature column fi (SoA) from the row-major X matrix.
        let column: Vec<f64> = x.column(fi).to_vec();
        let nan_sentinel = nan_sentinel_features.contains(&fi);
        let actual = select_borders_greedy_logsum(&column, BORDER_COUNT, nan_sentinel);

        assert_eq!(
            actual.len(),
            expected.len(),
            "{dataset} feature {fi}: border count {} != oracle {}",
            actual.len(),
            expected.len()
        );
        compare_stage(Stage::Borders, expected, &actual).unwrap_or_else(|e| {
            panic!("{dataset} feature {fi}: borders diverged from oracle: {e:?}")
        });
    }

    assert_eq!(
        offset,
        expected_flat.len(),
        "{dataset}: consumed every oracle border"
    );
}

/// numeric_tiny: NaN-free, 4 numeric features, no sentinel on any feature.
#[test]
fn numeric_tiny_borders_match_oracle() {
    check_dataset("numeric_tiny", &[]);
}

/// numeric_nan: feature 0 is a NaN feature under nan_mode=Min, so its oracle
/// borders begin with the f32::MIN sentinel; features 1 and 2 are NaN-free.
#[test]
fn numeric_nan_borders_match_oracle() {
    check_dataset("numeric_nan", &[0]);
}
