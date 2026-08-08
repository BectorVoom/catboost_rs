//! DCTR-14 (T07) smoke gate: the frozen BinarizedTargetMeanValue CTR device fixture
//! loads, is device-shaped, and carries **exactly one** CTR descriptor.
//!
//! The single-descriptor assertion is the discriminating one.
//! `ECtrType::target_border_count(BinarizedTargetMeanValue, _) == 1`
//! (`cb-train/src/ctr/mod.rs:137-146`) — BTMV does not binarize the target at all — so
//! one `(projection, prior)` yields exactly ONE CTR column, unlike `Buckets`, which
//! yields one per target class. A fixture carrying two descriptors would mean upstream
//! emitted something other than the intended simple BTMV column (a second prior, a
//! second projection, or a different type) and would silently retarget the DCTR-14
//! end-to-end oracle.
//!
//! The prior is pinned to `0.5` on both sides: BTMV's quantizer applies
//! `calc_normalization(prior)`, which is the identity `(0.0, 1.0)` only for priors in
//! `[0, 1]` (DCTR-04/DCTR-05). A prior outside that range would make this fixture
//! depend on Track E's correction rather than being inert under it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::load_f64_vec;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_device_btmv";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(SCENARIO)
        .join(rel)
}

fn read_json(rel: &str) -> serde_json::Value {
    let raw = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{SCENARIO}/{rel} must be readable: {e:?}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{SCENARIO}/{rel} must parse: {e:?}"))
}

#[test]
fn ctr_device_btmv_fixture_loads_with_single_target_border() {
    // --- device-shaped data (same shape contract as ctr_device_mixed / T05) --------
    let x: Array2<f32> = read_npy(fixture("X.npy"))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X.npy must load as f32 [N,F]: {e:?}"));
    assert_eq!(x.nrows(), 64, "{SCENARIO}: 64 rows");
    assert_eq!(
        x.ncols(),
        2,
        "{SCENARIO}: 2 float columns — a cat-only pool can never reach the device"
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

    // --- THE discriminating assertion: exactly ONE BTMV descriptor -----------------
    let model_json = read_json("model.json");
    let ctrs = model_json["features_info"]["ctrs"]
        .as_array()
        .expect("model.json must carry a ctrs array");
    assert_eq!(
        ctrs.len(),
        1,
        "{SCENARIO}: expected exactly ONE CTR descriptor — \
         target_border_count(BinarizedTargetMeanValue) == 1 means one column per \
         (projection, prior); got {}",
        ctrs.len()
    );
    let ctr = &ctrs[0];
    assert_eq!(
        ctr["ctr_type"],
        serde_json::json!("BinarizedTargetMeanValue"),
        "{SCENARIO}: the single descriptor must be BinarizedTargetMeanValue"
    );
    assert_eq!(
        ctr["target_border_idx"],
        serde_json::json!(0),
        "{SCENARIO}: BTMV never binarizes the target — its only selector is b = 0"
    );
    assert_eq!(
        ctr["prior_numerator"],
        serde_json::json!(0.5),
        "{SCENARIO}: prior numerator must be 0.5"
    );
    assert_eq!(
        ctr["prior_denomerator"],
        serde_json::json!(1),
        "{SCENARIO}: prior denominator must be 1 (DCTR-02: prior_denom is always 1.0)"
    );
    assert_eq!(
        ctr["elements"].as_array().map_or(0, Vec::len),
        1,
        "{SCENARIO}: max_ctr_complexity=1 ⇒ a simple, single-member projection"
    );

    assert!(
        model_json["features_info"]["float_features"]
            .as_array()
            .expect("model.json must carry float_features")
            .iter()
            .any(|f| f["borders"].as_array().is_some_and(|b| !b.is_empty())),
        "{SCENARIO}: no float split in the model — the float axis is decorative"
    );

    // --- config pins ---------------------------------------------------------------
    let config = read_json("config.json");
    assert_eq!(
        config["params"]["simple_ctr"],
        serde_json::json!(["BinarizedTargetMeanValue:Prior=0.5"]),
        "{SCENARIO}: the BTMV prior is pinned explicitly on the fixture side"
    );
    assert_eq!(
        config["params"]["combinations_ctr"],
        serde_json::json!([]),
        "{SCENARIO}: no combination descriptor — this is a SIMPLE BTMV fixture"
    );
    assert_eq!(
        config["params"]["max_ctr_complexity"],
        serde_json::json!(1),
        "{SCENARIO}: max_ctr_complexity=1"
    );
    assert_eq!(
        config["params"]["border_count"],
        serde_json::json!(15),
        "{SCENARIO}: border_count=15 is gate-load-bearing (R-11)"
    );
    assert_eq!(
        config["n_ctr_descriptors"],
        serde_json::json!(1),
        "{SCENARIO}: the generator must record the single-descriptor observation"
    );
    assert_eq!(
        config["observed_target_border_idxs"],
        serde_json::json!([0]),
        "{SCENARIO}: BTMV exposes only b = 0"
    );
    assert_eq!(
        config["requirement"],
        serde_json::json!("DCTR-14"),
        "{SCENARIO}: traceability to the specification"
    );
    assert!(
        config["note"]
            .as_str()
            .is_some_and(|n| n.starts_with("FROZEN")),
        "{SCENARIO}: config.json must carry the FROZEN marker (plan §2.6 / R-12)"
    );
}
