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

#[test]
fn mean_ctr_cbm_save_load_save_is_byte_identical() {
    // E20 / SPEC-CTRT-15 (round-trip half): save -> load -> save byte identity
    // for a model whose ctr_data carries a mean table, seeded from the
    // UPSTREAM-produced .cbm (never a self-comparison alone).
    let once_loaded = cb_model::load_cbm(&fixture(&format!("{SCENARIO}/model.cbm")))
        .unwrap_or_else(|e| panic!("upstream BTMV .cbm must load: {e:?}"));

    let dir = std::env::temp_dir();
    let first_path = dir.join(format!("btmv_rt1_{}.cbm", std::process::id()));
    let second_path = dir.join(format!("btmv_rt2_{}.cbm", std::process::id()));

    cb_model::save_cbm(&once_loaded, &first_path)
        .unwrap_or_else(|e| panic!("save_cbm must accept a mean model after E20: {e:?}"));
    let reloaded = cb_model::load_cbm(&first_path)
        .unwrap_or_else(|e| panic!("our own mean .cbm must reload: {e:?}"));

    // The mean vectors must survive exactly. (f32, i64) compares bitwise under
    // PartialEq for non-NaN f32 — assert no NaN first so equality is meaningful.
    for (key, t) in &once_loaded.ctr_data.as_ref().expect("ctr_data").tables {
        assert!(
            t.mean.iter().all(|(s, _)| !s.is_nan()),
            "NaN Sum in table {key} — bitwise comparison would be vacuous"
        );
    }
    assert_eq!(
        reloaded.ctr_data, once_loaded.ctr_data,
        "the mean ctr_data must survive save -> load exactly"
    );

    cb_model::save_cbm(&reloaded, &second_path)
        .unwrap_or_else(|e| panic!("second save must succeed: {e:?}"));
    let first = std::fs::read(&first_path).expect("read first save");
    let second = std::fs::read(&second_path).expect("read second save");
    let _ = std::fs::remove_file(&first_path);
    let _ = std::fs::remove_file(&second_path);
    assert_eq!(first, second, "save -> load -> save must be byte-identical");
}

#[test]
fn mean_blob_is_eight_bytes_per_bucket() {
    // Pins the wire stride (SPEC-CTRT-14): a future stride change is a test
    // failure here, not a silent incompatibility with upstream.
    let model = cb_model::load_cbm(&fixture(&format!("{SCENARIO}/model.cbm")))
        .unwrap_or_else(|e| panic!("upstream BTMV .cbm must load: {e:?}"));
    let ctr_data = model.ctr_data.as_ref().expect("ctr_data");
    let t = ctr_data
        .tables
        .values()
        .find(|t| t.ctr_type == ECtrType::BinarizedTargetMeanValue)
        .expect("BTMV table");

    let dir = std::env::temp_dir();
    let path = dir.join(format!("btmv_stride_{}.cbm", std::process::id()));
    cb_model::save_cbm(&model, &path).expect("save");

    // The re-encoded file must reload with the same bucket count — and because
    // the decoder's stride-8 length gate (`blob.len() / 8 == bucket_count`) is
    // the ONLY accepted mean layout (a 12-byte or mismatched blob is a typed
    // rejection), a reload success IS the 8-bytes-per-bucket pin on the writer.
    let reloaded = cb_model::load_cbm(&path)
        .unwrap_or_else(|e| panic!("our mean .cbm must reload through the stride-8 gate: {e:?}"));
    let _ = std::fs::remove_file(&path);
    let rt = reloaded
        .ctr_data
        .as_ref()
        .expect("ctr_data")
        .tables
        .values()
        .find(|x| x.ctr_type == ECtrType::BinarizedTargetMeanValue)
        .expect("BTMV table after re-encode");
    assert_eq!(rt.mean.len(), t.mean.len(), "bucket count must survive the 8-byte stride");
    assert!(!rt.mean.is_empty(), "anti-vacuity");
}

#[test]
fn saving_a_mean_table_whose_count_exceeds_i32_is_a_typed_error() {
    let mut model = cb_model::load_cbm(&fixture(&format!("{SCENARIO}/model.cbm")))
        .unwrap_or_else(|e| panic!("upstream BTMV .cbm must load: {e:?}"));
    // Corrupt one bucket's Count past the i32 wire range.
    if let Some(data) = model.ctr_data.as_mut() {
        if let Some(t) = data.tables.values_mut().next() {
            if let Some(pair) = t.mean.first_mut() {
                pair.1 = i64::from(i32::MAX) + 1;
            }
        }
    }
    let path = std::env::temp_dir().join(format!("btmv_overflow_{}.cbm", std::process::id()));
    let err = cb_model::save_cbm(&model, &path)
        .expect_err("a Count beyond i32 must be a typed Serialize rejection");
    let _ = std::fs::remove_file(&path);
    let msg = format!("{err:?}");
    assert!(
        msg.contains("bucket") && msg.contains("i32"),
        "the rejection must name the bucket and the i32 wire range: {msg}"
    );
}
