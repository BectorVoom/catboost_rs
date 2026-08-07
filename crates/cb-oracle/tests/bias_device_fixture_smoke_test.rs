//! FPP-03 (T01) smoke gate: the frozen NON-ZERO-BIAS device fixture loads, genuinely
//! carries `boost_from_average=True`, has a target mean far enough from zero to
//! discriminate the bias fix, and ships the FULL 15-border quantization set
//! (`borders.npy` — `model.json` keeps only the pruned USED borders, which is too few
//! for the device `n_bins` arithmetic).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::load_f64_vec;
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "bias_device_sym";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

#[test]
fn bias_device_fixture_loads_with_nonzero_bias() {
    let x: Array2<f32> = read_npy(fixture(&format!("{SCENARIO}/X.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X.npy must load as f32 [N,F]: {e:?}"));
    assert_eq!(x.nrows(), 64, "{SCENARIO}: 64 rows");
    assert_eq!(x.ncols(), 2, "{SCENARIO}: 2 float columns");

    let y = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();
    assert_eq!(y.len(), 64);
    assert!(
        y.iter().all(|v| v.abs() < 10.0),
        "{SCENARIO}: |y| < 10 backs the 2^33 fixed-point margin arithmetic"
    );

    // THE POINT OF THIS FIXTURE: a near-zero target mean cannot discriminate a
    // starting approximant of `mean(y)` from the device's former hardcoded 0.0 seed.
    let mean_y = y.iter().sum::<f64>() / (y.len() as f64);
    assert!(
        mean_y.abs() > 0.5,
        "{SCENARIO}: |mean(y)| = {mean_y:.6} must exceed 0.5 or the bias axis is vacuous"
    );

    let borders: Array2<f64> = read_npy(fixture(&format!("{SCENARIO}/borders.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/borders.npy must load as f64 [2,15]: {e:?}"));
    assert_eq!(borders.nrows(), 2, "{SCENARIO}: one border row per float column");
    assert_eq!(borders.ncols(), 15, "{SCENARIO}: the FULL 15-border set (16 bins)");
    for row in borders.rows() {
        assert!(
            row.windows(2).into_iter().all(|w| w[0] < w[1]),
            "{SCENARIO}: borders must be strictly ascending"
        );
    }

    let predictions = load_f64_vec(&fixture(&format!("{SCENARIO}/predictions.npy"))).unwrap();
    assert_eq!(predictions.len(), 64);
    let mean_p = predictions.iter().sum::<f64>() / 64.0;
    let var = predictions.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>() / 64.0;
    assert!(var.sqrt() > 1e-6, "{SCENARIO}: degenerate constant predictions");

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture(&format!("{SCENARIO}/config.json"))).unwrap())
            .unwrap();
    assert_eq!(
        config["params"]["boost_from_average"], serde_json::Value::Bool(true),
        "{SCENARIO}: boost_from_average=True is the entire point of this fixture"
    );

    // `model.json` must expose 2 float features; its border list is the PRUNED subset
    // upstream actually used, so only membership in the frozen full set is checked.
    let model_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture(&format!("{SCENARIO}/model.json"))).unwrap())
            .unwrap();
    let float_features = model_json["features_info"]["float_features"]
        .as_array()
        .expect("model.json float_features array");
    assert_eq!(float_features.len(), 2, "{SCENARIO}: 2 float features in model.json");
    for (fi, f) in float_features.iter().enumerate() {
        for b in f["borders"].as_array().into_iter().flatten() {
            let b = b.as_f64().unwrap();
            assert!(
                borders.row(fi).iter().any(|fb| (fb - b).abs() < 1e-9),
                "{SCENARIO}: model border {b} (feature {fi}) missing from the frozen set"
            );
        }
    }
}
