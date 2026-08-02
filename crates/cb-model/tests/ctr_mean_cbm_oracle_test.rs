//! E19 / SPEC-CTRT-15 (decode half), acceptance A8 — an UPSTREAM-produced
//! `.cbm` carrying a `BinarizedTargetMeanValue` CTR table loads and predicts
//! within ≤1e-5, replacing the v1 `ModelError::Deserialize("mean/target-mean
//! CTR unsupported")` rejection.
//!
//! The fixture is `ctr_btmv_simple/model.cbm` — emitted by the SAME generator
//! invocation that froze `model.json` / `predictions.npy` (E13/E18), so the
//! `.cbm` and the reference predictions describe the identical upstream model,
//! and its generator's anti-false-pass guard proved the model genuinely carries
//! a BTMV descriptor. This is a REAL upstream-format gate, not a
//! self-comparison (risk R7).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::stringify_int_category;
use cb_model::ECtrType;
use cb_oracle::load_f64_vec;
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_btmv_simple";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

#[test]
fn upstream_btmv_cbm_loads_without_a_mean_rejection() {
    let model = cb_model::load_cbm(&fixture(&format!("{SCENARIO}/model.cbm")))
        .unwrap_or_else(|e| panic!("upstream BTMV .cbm must load (E19 lifts the v1 mean rejection): {e:?}"));

    let ctr_data = model
        .ctr_data
        .as_ref()
        .expect("the loaded BTMV model must carry ctr_data");
    let t = ctr_data
        .tables
        .values()
        .find(|t| t.ctr_type == ECtrType::BinarizedTargetMeanValue)
        .expect("no BTMV table in the loaded .cbm");

    assert_eq!(t.mean.len(), t.hashes.len(), "one (Sum, Count) pair per bucket");
    assert!(t.int_counts.is_empty(), "a mean table carries no int_counts");
    // ANTI-VACUITY: a decoder returning an all-zero `mean` of the right length
    // would otherwise pass the two shape assertions.
    assert!(
        t.mean.iter().any(|&(s, c)| s != 0.0 && c != 0),
        "all-zero mean table — the blob was not actually decoded"
    );
}

#[test]
fn upstream_btmv_cbm_predicts_within_1e_minus_5() {
    let model = cb_model::load_cbm(&fixture(&format!("{SCENARIO}/model.cbm")))
        .unwrap_or_else(|e| panic!("upstream BTMV .cbm must load: {e:?}"));
    let expected = load_f64_vec(&fixture(&format!("{SCENARIO}/predictions.npy"))).unwrap();
    let cat_cols = load_cat_columns();

    let ours = cb_model::predict_raw_cat(&model, &[], &cat_cols);

    assert_eq!(ours.len(), expected.len(), "prediction count must match upstream");
    assert!(
        ours.iter().any(|v| *v != ours[0]),
        "predictions are constant — the gate would be vacuous"
    );

    let max_div = ours
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_div <= 1e-5,
        "upstream-loaded BTMV .cbm diverged from upstream's own predictions: \
         max |diff| = {max_div:e}"
    );
    println!("upstream BTMV .cbm predict max |diff| = {max_div:e}");
}
