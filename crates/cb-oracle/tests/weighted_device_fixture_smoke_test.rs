//! GDC-06 (T03) smoke gate: the two frozen weighted-training fixtures load, are
//! genuinely non-uniform in weight, and carry the FULL 15-border quantization set
//! (`borders.npy` — model.json keeps only the pruned USED borders, which is too
//! few for the device `n_bins` arithmetic).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::load_f64_vec;
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

fn assert_scenario(scenario: &str) {
    let x: Array2<f32> = read_npy(fixture(&format!("{scenario}/X.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/X.npy must load as f32 [N,F]: {e:?}"));
    assert_eq!(x.nrows(), 64, "{scenario}: 64 rows");
    assert_eq!(x.ncols(), 2, "{scenario}: 2 float columns");

    let y = load_f64_vec(&fixture(&format!("{scenario}/y.npy"))).unwrap();
    assert_eq!(y.len(), 64);
    assert!(
        y.iter().all(|v| v.abs() < 10.0),
        "{scenario}: |y| < 10 backs the 2^33 fixed-point margin arithmetic"
    );

    let weights = load_f64_vec(&fixture(&format!("{scenario}/weights.npy"))).unwrap();
    assert_eq!(weights.len(), 64);
    assert!(
        weights.iter().any(|&w| (w - 1.0).abs() > 1e-12),
        "{scenario}: weights must be non-uniform or the fixture is vacuous"
    );

    let borders: Array2<f64> = read_npy(fixture(&format!("{scenario}/borders.npy")))
        .unwrap_or_else(|e| panic!("{scenario}/borders.npy must load as f64 [2,15]: {e:?}"));
    assert_eq!(borders.nrows(), 2, "{scenario}: one border row per float column");
    assert_eq!(borders.ncols(), 15, "{scenario}: the FULL 15-border set (16 bins)");
    for row in borders.rows() {
        assert!(
            row.windows(2).into_iter().all(|w| w[0] < w[1]),
            "{scenario}: borders must be strictly ascending"
        );
    }

    let predictions = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    assert_eq!(predictions.len(), 64);
    let mean = predictions.iter().sum::<f64>() / 64.0;
    let var = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / 64.0;
    assert!(var.sqrt() > 1e-6, "{scenario}: degenerate constant predictions");
}

#[test]
fn weighted_device_fixtures_load() {
    assert_scenario("weighted_device_sym");
    assert_scenario("weighted_device_nonsym");
}
