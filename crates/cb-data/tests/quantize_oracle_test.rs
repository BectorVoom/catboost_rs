//! Quantization oracle: proves [`cb_data::Pool::quantize`] reproduces upstream
//! CatBoost's standalone quantization on the NaN-containing `numeric_nan` corpus
//! (DATA-02 / DATA-04) — per-feature borders match `borders_quant/` to <= 1e-5
//! (honoring the feature-0 `f32::MIN` sentinel, A1/A3), and NaN objects land in
//! the expected bin per the dataset's `nan_mode` (Min -> bin 0).
//!
//! The expected borders are the RAW standalone GreedyLogSum quantization output
//! (`Pool.quantize().save_quantization_borders()`), split per feature by
//! `<dataset>.borders_per_feature.npy`. Integration test (under `tests/`) so it
//! can depend on `cb-oracle`; the top-line allow mirrors `borders_oracle_test.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::ingest::{IngestSource, OwnedColumns};
use cb_data::{ColumnBins, NanMode, QuantizeParams};
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

/// Build a [`cb_data::Pool`] from the SoA columns of a fixture's `X.npy`.
fn pool_from_x(x: &Array2<f64>) -> cb_data::Pool {
    let n_features = x.ncols();
    let float_features: Vec<Vec<f64>> = (0..n_features).map(|fi| x.column(fi).to_vec()).collect();
    let label = vec![0.0_f64; x.nrows()];
    OwnedColumns::new(float_features, label)
        .into_pool()
        .expect("fixture columns are equal-length")
}

/// numeric_nan: feature 0 contains NaN under nan_mode=Min (sentinel-bearing
/// borders, NaN -> bin 0); features 1 and 2 are NaN-free. Quantize the whole
/// Pool and gate each feature's borders against the standalone oracle, then
/// assert the NaN rows land in bin 0.
#[test]
fn numeric_nan_quantization_matches_oracle() {
    let x = load_x("numeric_nan");
    let pool = pool_from_x(&x);

    // catboost 1.2.10 defaults (border_count=254, GreedyLogSum, nan_mode=Min).
    let qp = pool
        .quantize(&QuantizeParams::default())
        .expect("numeric_nan quantization must succeed");

    // Expected borders: flat f64, split per feature.
    let expected_flat =
        load_f64_vec(&fixture("borders_quant/numeric_nan.borders.npy")).unwrap();
    let per_feature =
        load_f64_vec(&fixture("borders_quant/numeric_nan.borders_per_feature.npy")).unwrap();

    let n_features = x.ncols();
    assert_eq!(qp.n_float_features(), n_features);
    assert_eq!(per_feature.len(), n_features);

    let mut offset = 0usize;
    for (fi, &count_f64) in per_feature.iter().enumerate() {
        let count = count_f64 as usize;
        let expected = &expected_flat[offset..offset + count];
        offset += count;

        // Widen the Rust f32 borders to f64 for the oracle comparator.
        let actual: Vec<f64> = qp
            .float_borders(fi)
            .unwrap_or_else(|| panic!("feature {fi} borders present"))
            .iter()
            .map(|&b| f64::from(b))
            .collect();

        assert_eq!(
            actual.len(),
            expected.len(),
            "numeric_nan feature {fi}: border count {} != oracle {}",
            actual.len(),
            expected.len()
        );
        compare_stage(Stage::Borders, expected, &actual).unwrap_or_else(|e| {
            panic!("numeric_nan feature {fi}: borders diverged from oracle: {e:?}")
        });
    }
    assert_eq!(offset, expected_flat.len(), "consumed every oracle border");

    // Feature 0 is the NaN feature under Min: borders[0] is the f32::MIN sentinel
    // and every NaN row quantizes to bin 0.
    assert_eq!(qp.float_nan_mode(0), Some(NanMode::Min));
    let f0_borders = qp.float_borders(0).unwrap();
    assert_eq!(f0_borders[0], f32::MIN, "feature 0 Min sentinel at index 0");

    let f0_bins = match qp.float_bins(0).unwrap() {
        ColumnBins::U8(v) => v.iter().map(|&b| u32::from(b)).collect::<Vec<_>>(),
        ColumnBins::U16(v) => v.iter().map(|&b| u32::from(b)).collect::<Vec<_>>(),
        ColumnBins::U32(v) => v.clone(),
    };
    // NaN row indices recorded in inputs/numeric_nan/config.json.
    for &row in &[3usize, 7, 11, 19, 28, 41] {
        assert_eq!(
            f0_bins[row], 0,
            "numeric_nan feature 0 NaN row {row} must quantize to bin 0 (Min)"
        );
    }

    // NaN-free features 1 and 2 carry no sentinel (first border != f32::MIN).
    for fi in [1usize, 2] {
        assert_eq!(qp.float_nan_mode(fi), Some(NanMode::Forbidden));
        let b = qp.float_borders(fi).unwrap();
        assert_ne!(b[0], f32::MIN, "feature {fi} must not carry a sentinel");
    }
}
