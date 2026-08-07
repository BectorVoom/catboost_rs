//! FPP-07 (T02) smoke gate: the two frozen EXACT-leaf device fixtures load, both pin
//! `leaf_estimation_method="Exact"` and `border_count=15`, and their predictions
//! genuinely differ — which proves the quantile α is load-bearing. A device path that
//! silently ignored `quantile_alpha` would pass a single-fixture test.
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

fn json_at(rel: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture(rel)).unwrap())
        .unwrap_or_else(|e| panic!("{rel} must parse as JSON: {e:?}"))
}

fn assert_scenario(scenario: &str, expected_loss: &str) -> Vec<f64> {
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

    let config = json_at(&format!("{scenario}/config.json"));
    assert_eq!(
        config["params"]["leaf_estimation_method"],
        serde_json::Value::String("Exact".to_owned()),
        "{scenario}: leaf_estimation_method=Exact is the entire point of this fixture"
    );
    assert_eq!(
        config["params"]["border_count"],
        serde_json::json!(15),
        "{scenario}: border_count=15 (16 bins) is the only device-admitted width here"
    );
    assert_eq!(
        config["params"]["loss_function"],
        serde_json::Value::String(expected_loss.to_owned()),
        "{scenario}: loss_function"
    );
    assert_eq!(
        config["params"]["boost_from_average"],
        serde_json::Value::Bool(false),
        "{scenario}: bias 0 isolates the exact-leaf axis from Track A"
    );

    let predictions = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    assert_eq!(predictions.len(), 64);
    let mean = predictions.iter().sum::<f64>() / 64.0;
    let var = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / 64.0;
    assert!(var.sqrt() > 1e-6, "{scenario}: degenerate constant predictions");
    predictions
}

#[test]
fn exact_leaf_device_fixtures_load_and_alpha_is_load_bearing() {
    let mae = assert_scenario("exact_leaf_device/mae", "MAE");
    let q07 = assert_scenario("exact_leaf_device/quantile07", "Quantile:alpha=0.7");

    let max_delta = mae
        .iter()
        .zip(q07.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_delta > 1e-6,
        "MAE and Quantile:alpha=0.7 predictions agree (max|Δ|={max_delta:.3e}) — a device \
         path that ignored quantile_alpha would pass, so the pair is vacuous"
    );
}
