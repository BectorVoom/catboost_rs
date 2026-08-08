//! DCTR-08 (T05) smoke gate: the frozen BUCKETS-CTR device fixture loads and its trained
//! model genuinely carries CTR splits at BOTH `target_border_idx` values 0 and 1.
//!
//! That last assertion is the discriminating one. A Buckets CTR at binclf emits one
//! candidate column per `target_border_idx` in `0..targetClassesCount`
//! (`GetTargetBorderCount`, `ctr_helper.h:34-42`), so a fixture in which upstream happened
//! to select only the `b = 0` numerator would leave the device kernel's `Buckets@1`
//! numerator (`good = counts[1]`, SPEC §4.2) completely unexercised — and every downstream
//! `<=1e-5` comparison would pass without ever touching it. If this assertion fires the
//! fixture must be re-generated at a higher escalation rung (see `gen_fixtures.py`), NEVER
//! by weakening the guard.
//!
//! One smoke file per fixture is the shipped convention (PLAN C-15).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::BTreeSet;
use std::path::PathBuf;

use cb_oracle::load_f64_vec;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_device_buckets";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

fn read_json(rel: &str) -> serde_json::Value {
    let path = fixture(rel);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e:?}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} must be JSON: {e:?}", path.display()))
}

#[test]
fn ctr_device_buckets_fixture_loads_with_both_target_borders() {
    let x: Array2<f32> = read_npy(fixture(&format!("{SCENARIO}/X.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X.npy must load as f32 [N,F]: {e:?}"));
    assert_eq!(x.nrows(), 64, "{SCENARIO}: 64 rows");
    assert_eq!(
        x.ncols(),
        2,
        "{SCENARIO}: 2 float columns — a cat-only pool can never reach the device \
         (`has_any_scorable_feature` needs n_float > 0)"
    );

    let cat: Array1<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as i32 [N]: {e:?}"));
    assert_eq!(cat.len(), 64, "{SCENARIO}: one cat column, 64 rows");

    let y = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();
    assert_eq!(y.len(), 64);

    let borders: Array2<f64> = read_npy(fixture(&format!("{SCENARIO}/borders.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/borders.npy must load as f64 [F,15]: {e:?}"));
    assert_eq!(borders.nrows(), 2, "{SCENARIO}: one border row per float feature");
    for (fi, row) in borders.rows().into_iter().enumerate() {
        assert_eq!(
            row.len(),
            15,
            "{SCENARIO}: float feature {fi} must carry exactly 15 borders — `ctr_covered` \
             needs borders.len() + 1 == n_bins (R-11)"
        );
    }

    let predictions: Array1<f64> = read_npy(fixture(&format!("{SCENARIO}/predictions.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/predictions.npy must load: {e:?}"));
    assert_eq!(predictions.len(), 64);

    let config = read_json(&format!("{SCENARIO}/config.json"));
    assert_eq!(
        config["observed_target_border_idxs"],
        serde_json::json!([0, 1]),
        "{SCENARIO}: the generator must have observed BOTH Buckets numerators"
    );
    assert!(
        config["note"]
            .as_str()
            .is_some_and(|n| n.starts_with("FROZEN")),
        "{SCENARIO}: config.json must carry the FROZEN marker (plan §2.6 / R-12)"
    );

    let model_json = read_json(&format!("{SCENARIO}/model.json"));
    let ctrs = model_json["features_info"]["ctrs"]
        .as_array()
        .expect("model.json must carry a ctrs array");
    assert!(!ctrs.is_empty(), "{SCENARIO}: no CTR descriptors at all");
    for c in ctrs {
        assert_eq!(
            c["ctr_type"],
            serde_json::json!("Buckets"),
            "{SCENARIO}: every descriptor must be Buckets, got {}",
            c["ctr_type"]
        );
    }
    // `target_border_idx` is a TOP-LEVEL key of each `features_info.ctrs[i]` descriptor
    // (verified against `ctr_buckets_simple/model.json`, whose key set is exactly
    // borders, ctr_type, elements, identifier, prior_denomerator, prior_numerator, scale,
    // shift, target_border_idx). Never read it through a defaulting `.get(..)`: a default
    // would make this guard silently vacuous.
    let idxs: BTreeSet<i64> = ctrs
        .iter()
        .map(|c| {
            c["target_border_idx"]
                .as_i64()
                .expect("every CTR descriptor must carry a top-level target_border_idx")
        })
        .collect();
    assert_eq!(
        idxs.into_iter().collect::<Vec<_>>(),
        vec![0, 1],
        "{SCENARIO}: the model must carry Buckets splits at BOTH target_border_idx 0 and 1 \
         — otherwise the device Buckets@1 numerator is unexercised. Re-generate at a higher \
         escalation rung; do NOT weaken this assertion"
    );
}
