//! DCTR-10 (T06) smoke gate: the frozen COUNTER-CTR device fixture loads, is
//! device-shaped (2 float columns with the FULL 15-border set — 16 bins matching the 16
//! CTR buckets `ctr_covered` requires, 1 CTR-routed cat column, 64 rows), and carries the
//! Counter prior pinned **explicitly**.
//!
//! THE COUNTER TRAP (R-11 sibling): upstream's default Counter prior is `0/1`, **not**
//! `0.5` (`cb-train/src/ctr/mod.rs` `default_priors()`). The prior must be pinned on BOTH
//! sides — `"Counter:Prior=0.5"` in this fixture's params and `simple_ctr_priors =
//! vec![0.5]` in the T12 `BoostParams`. A mismatch produces a silent, plausible-looking
//! divergence with no compile or shape error, so the params pin below is the
//! discriminating assertion of this file, alongside the presence of a real `Counter`
//! descriptor in the trained model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::load_f64_vec;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_device_counter";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(SCENARIO)
        .join(rel)
}

#[test]
fn ctr_device_counter_fixture_loads_with_pinned_prior() {
    let x: Array2<f32> = read_npy(fixture("X.npy"))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X.npy must load as f32 [N,F]: {e:?}"));
    assert_eq!(x.nrows(), 64, "{SCENARIO}: 64 rows");
    assert_eq!(
        x.ncols(),
        2,
        "{SCENARIO}: 2 float columns — cat-only pools can never reach the device"
    );

    let cat: Array1<i32> = read_npy(fixture("X_cat.npy"))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as i32 [N]: {e:?}"));
    assert_eq!(cat.len(), 64);
    let card = cat.iter().collect::<std::collections::BTreeSet<_>>().len();
    assert!(
        card > 2,
        "{SCENARIO}: the cat column must be CTR-routed, not one-hot (cardinality {card} > 2)"
    );

    let y = load_f64_vec(&fixture("y.npy")).unwrap();
    assert_eq!(y.len(), 64);
    assert!(
        y.iter().all(|&v| v == 0.0 || v == 1.0),
        "{SCENARIO}: binclf labels"
    );

    let borders: Array2<f64> = read_npy(fixture("borders.npy"))
        .unwrap_or_else(|e| panic!("{SCENARIO}/borders.npy must load as f64 [2,15]: {e:?}"));
    assert_eq!(borders.nrows(), 2);
    assert_eq!(
        borders.ncols(),
        15,
        "{SCENARIO}: border_count=15 — ctr_covered needs borders.len()+1 == n_bins"
    );
    for row in borders.rows() {
        assert!(
            row.windows(2).into_iter().all(|w| w[0] < w[1]),
            "{SCENARIO}: borders must be strictly ascending"
        );
    }

    let predictions = load_f64_vec(&fixture("predictions.npy")).unwrap();
    assert_eq!(predictions.len(), 64);
    let mean = predictions.iter().sum::<f64>() / 64.0;
    let var = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / 64.0;
    assert!(
        var.sqrt() > 1e-6,
        "{SCENARIO}: degenerate constant predictions"
    );

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("config.json")).unwrap()).unwrap();
    // THE explicit-prior pin (see the module docstring): upstream's Counter default is
    // 0/1, so a bare "Counter" here would silently train against a different prior than
    // the Rust side's `simple_ctr_priors = vec![0.5]`.
    assert_eq!(
        config["params"]["simple_ctr"],
        serde_json::json!(["Counter:Prior=0.5"]),
        "{SCENARIO}: the Counter prior must be pinned EXPLICITLY on the fixture side"
    );
    assert_eq!(
        config["params"]["max_ctr_complexity"],
        serde_json::json!(1),
        "{SCENARIO}: simple projections only — Track C scope"
    );
    assert_eq!(
        config["params"]["combinations_ctr"],
        serde_json::json!([]),
        "{SCENARIO}: no combination descriptor — Track C scope"
    );
    assert!(
        config["note"]
            .as_str()
            .is_some_and(|n| n.starts_with("FROZEN")),
        "{SCENARIO}: config.json must carry the FROZEN marker (R-12)"
    );

    let model_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("model.json")).unwrap()).unwrap();
    let ctrs = model_json["features_info"]["ctrs"]
        .as_array()
        .expect("model.json must carry a ctrs array");
    let counters = ctrs
        .iter()
        .filter(|c| c["ctr_type"] == serde_json::json!("Counter"))
        .count();
    assert!(
        counters >= 1,
        "{SCENARIO}: model.json has {} CTR descriptor(s) but none of type Counter — upstream \
         chose no Counter split, so the device Counter path would be unexercised; re-seed the \
         generator, do not accept this fixture",
        ctrs.len()
    );

    let float_features = model_json["features_info"]["float_features"]
        .as_array()
        .expect("model.json must carry float_features");
    assert!(
        float_features.iter().any(|f| f
            .get("borders")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|b| !b.is_empty())),
        "{SCENARIO}: no float split in the model — the float axis is decorative"
    );
}
