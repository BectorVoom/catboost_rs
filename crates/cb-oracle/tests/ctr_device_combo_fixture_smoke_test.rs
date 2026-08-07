//! FPP-10 (T03) smoke gate: the frozen COMBINATION-CTR device fixture loads and its
//! trained model genuinely contains at least one CTR split over a projection with >=2
//! members. That assertion is the discriminating one: a fixture where upstream happened
//! to pick only simple (1-member) projections cannot exercise the device combination-CTR
//! column builder (FPP-09) and must be re-seeded, not accepted.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::load_f64_vec;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_device_combo";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

/// Number of members in a model.json CTR descriptor's projection. Upstream serialises a
/// CTR's projection as the `elements` array — one entry per member, each carrying a
/// `combination_element` kind (`"cat_feature_value"` for a plain cat member, a binarized
/// float member otherwise). A simple projection has exactly one element; a combination
/// has two or more. Verified against `tensor_ctr_e2e/model.json`.
fn projection_len(ctr: &serde_json::Value) -> usize {
    ctr.get("elements")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len)
}

#[test]
fn ctr_device_combo_fixture_has_a_multi_member_projection() {
    let x: Array2<f32> = read_npy(fixture(&format!("{SCENARIO}/X.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X.npy must load as f32 [N,F]: {e:?}"));
    assert_eq!(x.nrows(), 64, "{SCENARIO}: 64 rows");
    assert_eq!(x.ncols(), 2, "{SCENARIO}: 2 float columns — device_n_float must be > 0");

    let cat: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as i32 [N,2]: {e:?}"));
    assert_eq!(cat.nrows(), 64);
    assert_eq!(cat.ncols(), 2, "{SCENARIO}: 2 cat columns are required for a 2-member combination");

    let y = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();
    assert_eq!(y.len(), 64);

    let borders: Array2<f64> = read_npy(fixture(&format!("{SCENARIO}/borders.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/borders.npy must load as f64 [2,15]: {e:?}"));
    assert_eq!(borders.nrows(), 2);
    assert_eq!(
        borders.ncols(),
        15,
        "{SCENARIO}: border_count=15 — ctr_covered needs borders.len()+1 == n_bins"
    );

    let predictions: Array1<f64> = read_npy(fixture(&format!("{SCENARIO}/predictions.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/predictions.npy must load: {e:?}"));
    assert_eq!(predictions.len(), 64);

    let model_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture(&format!("{SCENARIO}/model.json"))).unwrap())
            .unwrap();
    let ctrs = model_json["features_info"]["ctrs"]
        .as_array()
        .expect("model.json must carry a ctrs array");
    assert!(!ctrs.is_empty(), "{SCENARIO}: no CTR descriptors at all");

    let max_members = ctrs.iter().map(projection_len).max().unwrap_or(0);
    assert!(
        max_members >= 2,
        "{SCENARIO}: the widest CTR projection has {max_members} member(s); upstream chose \
         only simple projections, so the combination-CTR column builder is unexercised — \
         re-seed the generator, do not accept this fixture"
    );

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture(&format!("{SCENARIO}/config.json"))).unwrap())
            .unwrap();
    assert_eq!(
        config["params"]["max_ctr_complexity"],
        serde_json::json!(2),
        "{SCENARIO}: max_ctr_complexity=2 admits the 2-member projection"
    );
    assert!(
        config["combo_vs_complexity1_max_delta"]
            .as_f64()
            .expect("generator must record the complexity-1 discrimination delta")
            > 1e-6,
        "{SCENARIO}: a max_ctr_complexity=1 sibling predicts identically — vacuous"
    );
}
