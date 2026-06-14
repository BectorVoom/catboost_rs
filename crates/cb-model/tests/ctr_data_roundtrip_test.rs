//! `ctr_data` round-trip + inference-apply oracle (ORD-03, Security V5).
//!
//! # What this locks
//!
//! 1. **Semantic round-trip:** a `ctr_data` section built from the trainer-side
//!    whole-set final-CTR table (`cb_train::build_final_ctr`) round-trips through
//!    BOTH the `.cbm` binary blob AND the `model.json` flat-array shape, yielding
//!    identical bucket counts.
//! 2. **Inference apply ≤1e-5:** the model-side `ctr_value_for_projection`
//!    reproduces the per-type `Calc(cic, tot)` value — anchored against the
//!    trainer's whole-set counts (the transcribe-then-self-oracle anchor, the
//!    D-04 / 05-02 precedent: no upstream categorical `model.json` fixture is
//!    committed — the plain_ctr fixture carries only the per-object `.npy`
//!    online vectors — so the model-side apply is locked against the
//!    independently-computed trainer-side whole-set table).
//! 3. **Malformed blob → typed Err, never panic** (T-05-04-V5).
//!
//! All comparisons go through `cb_oracle::assert_abs_close` (the ≤1e-5 gate) for
//! the float CTR values; the integer counts are compared exactly.

use std::collections::BTreeMap;

use cb_model::{
    ctr_value_for_projection, decode_ctr_data, encode_ctr_data, CtrData, CtrValueTable, ECtrType,
    Prior,
};
use cb_oracle::assert_abs_close;
use cb_train::{accumulate_online, build_final_ctr};

/// The same small categorical column the Task-1 unit tests use, as the
/// trainer-side anchor. `a` 3x (classes 1,1,0), `b` 2x (0,1), `c` 1x (1).
fn anchor_column() -> (Vec<&'static str>, Vec<usize>, Vec<f64>) {
    (
        vec!["a", "a", "b", "a", "b", "c"],
        vec![1, 1, 0, 0, 1, 1],
        vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    )
}

/// Lift a trainer-side `FinalCtrTable` (whole-set counts) into a model-side
/// `CtrValueTable`, hashing each bucket's representative value via the SAME
/// categorical hash source the inference path uses (`calc_cat_feature_hash`,
/// NEVER a model ctr_data hash_map — D Carried-Forward). The bucket order is the
/// first-seen perfect-hash order, so the representative values are the distinct
/// column values in first-seen order.
fn lift(
    ctr_type: ECtrType,
    final_table: &cb_train::FinalCtrTable,
    distinct_values_first_seen: &[&str],
) -> CtrValueTable {
    let hashes: Vec<u64> = distinct_values_first_seen
        .iter()
        .map(|v| u64::from(cb_data_hash(v)))
        .collect();
    let classes = final_table.target_classes_count.max(1);
    let (int_counts, mean) = if ctr_type == ECtrType::BinarizedTargetMeanValue
        || ctr_type == ECtrType::FloatTargetMeanValue
    {
        let mean: Vec<(f32, i64)> = final_table
            .mean_sum
            .iter()
            .zip(final_table.mean_count.iter())
            .map(|(&s, &c)| (s, c))
            .collect();
        (Vec::new(), mean)
    } else if ctr_type == ECtrType::Counter || ctr_type == ECtrType::FeatureFreq {
        let int_counts: Vec<Vec<i64>> = final_table.int_counts.iter().map(|&c| vec![c]).collect();
        (int_counts, Vec::new())
    } else {
        // Borders/Buckets: bucket-major class counts -> per-bucket [N0, N1].
        let mut int_counts = Vec::new();
        for chunk in final_table.int_counts.chunks(classes) {
            int_counts.push(chunk.to_vec());
        }
        (int_counts, Vec::new())
    };
    CtrValueTable {
        ctr_type,
        target_classes_count: classes,
        hashes,
        int_counts,
        mean,
        counter_denominator: final_table.counter_denominator,
    }
}

/// Bridge to the categorical hash (re-exported through cb-data via cb-model's
/// dependency; the test crate hashes the same way the apply path does).
fn cb_data_hash(s: &str) -> u32 {
    // cb-data is a transitive dependency of cb-model; the test exercises the same
    // hash by going through the public apply path (ctr_value_for_projection),
    // but for building the table we need the raw hash. Reuse cb-train's
    // accumulate path indirectly is overkill; instead hash via cb_data directly.
    cb_data::calc_cat_feature_hash(s)
}

#[test]
fn borders_ctr_data_round_trips_and_applies() {
    let (col, tc, t) = anchor_column();
    let acc = accumulate_online(&col, &tc, &t, 2, 2).expect("accumulate");
    let final_table = build_final_ctr(&acc, cb_train::ECtrType::Borders);
    let table = lift(ECtrType::Borders, &final_table, &["a", "b", "c"]);

    // Build the ctr_data section and round-trip through BOTH wire forms.
    let mut tables = BTreeMap::new();
    tables.insert("borders_simple".to_owned(), table.clone());
    let data = CtrData { tables };

    // .cbm binary blob round-trip: identical tables.
    let blob = encode_ctr_data(&data);
    let from_blob = decode_ctr_data(&blob).expect("decode blob");
    assert_eq!(from_blob, data, "binary blob round-trip must be identical");

    // model.json flat-array round-trip: identical table.
    let json = table.to_json();
    let from_json = CtrValueTable::from_json(&json).expect("from_json");
    assert_eq!(from_json, table, "model.json round-trip must be identical");

    // Inference apply ≤1e-5: prior 0.5, unit denom -> coincides with online +1.
    // bucket "a": N1=2, total=3 -> (2+0.5)/(3+1) = 0.625.
    let v_a = ctr_value_for_projection(&table, "a", Prior::unit(0.5), 0.0, 1.0, 0);
    assert_abs_close(&[0.625], &[v_a], 1e-5).expect("borders 'a' CTR");
    // bucket "b": N1=1, total=2 -> (1+0.5)/(2+1) = 0.5.
    let v_b = ctr_value_for_projection(&table, "b", Prior::unit(0.5), 0.0, 1.0, 0);
    assert_abs_close(&[0.5], &[v_b], 1e-5).expect("borders 'b' CTR");
    // missing value "z" -> empty path Calc(0,0): (0+0.5)/(0+1) = 0.5.
    let v_z = ctr_value_for_projection(&table, "z", Prior::unit(0.5), 0.0, 1.0, 0);
    assert_abs_close(&[0.5], &[v_z], 1e-5).expect("borders missing CTR");
}

#[test]
fn counter_vs_feature_freq_denominators_apply_distinctly() {
    let (col, tc, t) = anchor_column();
    let acc = accumulate_online(&col, &tc, &t, 2, 2).expect("accumulate");

    let counter = lift(
        ECtrType::Counter,
        &build_final_ctr(&acc, cb_train::ECtrType::Counter),
        &["a", "b", "c"],
    );
    let freq = lift(
        ECtrType::FeatureFreq,
        &build_final_ctr(&acc, cb_train::ECtrType::FeatureFreq),
        &["a", "b", "c"],
    );

    // Counter denom = max bucket total = 3; bucket "a" total 3 -> (3+0)/(3+1)=0.75.
    let c_a = ctr_value_for_projection(&counter, "a", Prior::unit(0.0), 0.0, 1.0, 0);
    assert_abs_close(&[0.75], &[c_a], 1e-5).expect("counter 'a'");
    // FeatureFreq denom = total sample count = 6; bucket "a" -> (3+0)/(6+1)=0.428571.
    let f_a = ctr_value_for_projection(&freq, "a", Prior::unit(0.0), 0.0, 1.0, 0);
    assert_abs_close(&[3.0 / 7.0], &[f_a], 1e-5).expect("feature_freq 'a'");
    // The two DIFFER (Pitfall 4): distinct denominators.
    assert!((c_a - f_a).abs() > 1e-3, "Counter vs FeatureFreq must differ");
}

#[test]
fn float_target_mean_round_trips_and_applies() {
    let (col, tc, t) = anchor_column();
    let acc = accumulate_online(&col, &tc, &t, 2, 2).expect("accumulate");
    let table = lift(
        ECtrType::FloatTargetMeanValue,
        &build_final_ctr(&acc, cb_train::ECtrType::FloatTargetMeanValue),
        &["a", "b", "c"],
    );

    // Round-trip through the binary blob (mean histories).
    let mut tables = BTreeMap::new();
    tables.insert("float_mean".to_owned(), table.clone());
    let data = CtrData { tables };
    let from_blob = decode_ctr_data(&encode_ctr_data(&data)).expect("decode");
    assert_eq!(from_blob, data);

    // bucket "a": raw targets 1+1+0 = 2.0 over count 3 -> Calc(2,3) prior 0/1:
    // (2+0)/(3+1) = 0.5.
    let v_a = ctr_value_for_projection(&table, "a", Prior::unit(0.0), 0.0, 1.0, 0);
    assert_abs_close(&[0.5], &[v_a], 1e-5).expect("float-mean 'a'");
}

#[test]
fn malformed_blob_is_typed_error_not_panic() {
    // A random short buffer must Err (never panic) — Security V5, T-05-04-V5.
    assert!(decode_ctr_data(&[0xff, 0x00, 0x13]).is_err());
    // An oversized declared table_count is rejected by the length cap.
    let mut blob = Vec::new();
    blob.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_ctr_data(&blob).is_err());
}
