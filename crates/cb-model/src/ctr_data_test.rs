//! Unit tests for [`crate::ctr_data`] — bounds rejection, malformed-blob typed
//! errors, and per-type Calc (Security V5, T-05-04-V5 / T-05-04-01).

use crate::ctr_data::{
    calc_inference, ctr_base_key, decode_ctr_data, decode_ctr_model_parts, encode_ctr_data,
    encode_ctr_model_parts, CtrData, CtrTableJson, CtrValueTable, ECtrType, Prior,
};
use std::collections::BTreeMap;

fn borders_table() -> CtrValueTable {
    CtrValueTable {
        ctr_type: ECtrType::Borders,
        target_classes_count: 2,
        hashes: vec![10, 20, 30],
        // bucket0 N=[1,2], bucket1 N=[1,1], bucket2 N=[0,1].
        int_counts: vec![vec![1, 2], vec![1, 1], vec![0, 1]],
        mean: Vec::new(),
        counter_denominator: 0,
    }
}

fn counter_table() -> CtrValueTable {
    CtrValueTable {
        ctr_type: ECtrType::Counter,
        target_classes_count: 2,
        hashes: vec![10, 20, 30],
        int_counts: vec![vec![3], vec![2], vec![1]],
        mean: Vec::new(),
        counter_denominator: 3,
    }
}

#[test]
fn ectr_type_unknown_discriminant_is_typed_error() {
    assert!(ECtrType::from_i8(9).is_err());
    for v in 0..=5i8 {
        assert_eq!(ECtrType::from_i8(v).map(ECtrType::as_i8).ok(), Some(v));
    }
}

#[test]
fn borders_calc_uses_history1_over_total() {
    let t = borders_table();
    // bucket for hash 10: N1=2, total=3; prior 0.5 -> (2+0.5)/(3+1)=0.625.
    let v = t.calc_for_hash(10, Prior::unit(0.5), 0.0, 1.0, 0);
    assert!((v - 0.625).abs() < 1e-9, "got {v}");
}

#[test]
fn missing_bucket_returns_empty_value_not_panic() {
    let t = borders_table();
    // hash 999 not present -> Calc(0,0): (0+0.5)/(0+1)=0.5 (others empty path).
    let v = t.calc_for_hash(999, Prior::unit(0.5), 0.0, 1.0, 0);
    assert!((v - 0.5).abs() < 1e-9, "got {v}");
}

#[test]
fn counter_missing_bucket_uses_denominator_empty_path() {
    let t = counter_table();
    // missing bucket -> Calc(0, denom=3): (0+0)/(3+1)=0.0.
    let v = t.calc_for_hash(999, Prior::unit(0.0), 0.0, 1.0, 0);
    assert!((v - 0.0).abs() < 1e-9, "got {v}");
    // present bucket hash 10: total=3 -> (3+0)/(3+1)=0.75.
    let v = t.calc_for_hash(10, Prior::unit(0.0), 0.0, 1.0, 0);
    assert!((v - 0.75).abs() < 1e-9, "got {v}");
}

#[test]
fn calc_inference_guards_zero_denominator() {
    let v = calc_inference(0.0, 0.0, Prior { num: 0.0, denom: 0.0 }, 0.0, 1.0);
    assert!(v.is_finite() && (v - 0.0).abs() < 1e-12);
}

#[test]
fn json_round_trip_preserves_counts() {
    let t = borders_table();
    let json = t.to_json();
    let back = CtrValueTable::from_json(&json).expect("from_json");
    assert_eq!(back, t);
}

#[test]
fn json_round_trip_mean_table() {
    let t = CtrValueTable {
        ctr_type: ECtrType::FloatTargetMeanValue,
        target_classes_count: 2,
        hashes: vec![5, 6],
        int_counts: Vec::new(),
        mean: vec![(6.0, 2), (10.0, 1)],
        counter_denominator: 0,
    };
    let json = t.to_json();
    let back = CtrValueTable::from_json(&json).expect("from_json mean");
    assert_eq!(back, t);
}

#[test]
fn json_ragged_blob_is_typed_error() {
    // stride 3 but only 2 elements -> ragged.
    let json = CtrTableJson {
        hash_map: vec![serde_json::json!("10"), serde_json::json!(1)],
        hash_stride: 3,
        counter_denominator: 0,
        ctr_type: 0,
        target_classes_count: 2,
    };
    assert!(CtrValueTable::from_json(&json).is_err());
}

#[test]
fn json_non_integer_count_is_typed_error() {
    let json = CtrTableJson {
        hash_map: vec![serde_json::json!("10"), serde_json::json!("not-a-number")],
        hash_stride: 2,
        counter_denominator: 0,
        ctr_type: 0,
        target_classes_count: 2,
    };
    assert!(CtrValueTable::from_json(&json).is_err());
}

#[test]
fn json_unknown_ctr_type_is_typed_error() {
    let json = CtrTableJson {
        hash_map: Vec::new(),
        hash_stride: 0,
        counter_denominator: 0,
        ctr_type: 42,
        target_classes_count: 2,
    };
    assert!(CtrValueTable::from_json(&json).is_err());
}

#[test]
fn blob_round_trip_preserves_tables() {
    let mut tables = BTreeMap::new();
    tables.insert("ctr_a".to_owned(), borders_table());
    tables.insert("ctr_b".to_owned(), counter_table());
    let data = CtrData { tables };
    let blob = encode_ctr_data(&data);
    let back = decode_ctr_data(&blob).expect("decode");
    assert_eq!(back, data);
}

#[test]
fn blob_truncated_is_typed_error_not_panic() {
    let mut tables = BTreeMap::new();
    tables.insert("ctr_a".to_owned(), borders_table());
    let data = CtrData { tables };
    let blob = encode_ctr_data(&data);
    // Truncate mid-blob: must Err, never panic (T-05-04-V5).
    let truncated = &blob[..blob.len() / 2];
    assert!(decode_ctr_data(truncated).is_err());
}

#[test]
fn blob_empty_is_typed_error() {
    // An empty buffer cannot even read the table_count u32.
    assert!(decode_ctr_data(&[]).is_err());
}

#[test]
fn blob_oversized_declared_length_is_rejected() {
    // table_count = u32::MAX -> exceeds the declared-length cap, typed Err
    // (DoS guard T-05-04-02), not a huge alloc / panic.
    let mut blob = Vec::new();
    blob.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_ctr_data(&blob).is_err());
}

#[test]
fn empty_ctr_data_round_trips() {
    let data = CtrData::default();
    let blob = encode_ctr_data(&data);
    let back = decode_ctr_data(&blob).expect("decode empty");
    assert_eq!(back, data);
    assert!(back.tables.is_empty());
}

// ── T3 — upstream model-parts tail ENCODER round-trips via the decoder ────────

#[test]
fn encode_ctr_model_parts_roundtrips_via_decode() {
    // A Borders table (width 2) keyed by proj=[0] and a Counter table (width 1,
    // counter_denominator set) keyed by proj=[1] — the projection is recovered
    // from the KEY on encode, so the keys MUST be the canonical `ctr_base_key`.
    let mut tables = BTreeMap::new();
    tables.insert(ctr_base_key(ECtrType::Borders, &[0]), borders_table());
    tables.insert(ctr_base_key(ECtrType::Counter, &[1]), counter_table());
    let data = CtrData { tables };

    let bytes = encode_ctr_model_parts(&data).expect("encode_ctr_model_parts");
    let back = decode_ctr_model_parts(&bytes).expect("decode_ctr_model_parts");
    assert_eq!(back, data, "encode -> decode must reproduce the CtrData");
}

#[test]
fn encode_ctr_model_parts_rejects_mean_table() {
    let mut tables = BTreeMap::new();
    tables.insert(
        ctr_base_key(ECtrType::FloatTargetMeanValue, &[0]),
        CtrValueTable {
            ctr_type: ECtrType::FloatTargetMeanValue,
            target_classes_count: 2,
            hashes: vec![5],
            int_counts: Vec::new(),
            mean: vec![(6.0, 2)],
            counter_denominator: 0,
        },
    );
    assert!(encode_ctr_model_parts(&CtrData { tables }).is_err());
}

#[test]
fn encode_ctr_model_parts_rejects_marker_hash() {
    // A bucket hash == 0xFFFF_FFFF_FFFF_FFFF would read back as an empty slot and
    // corrupt bucket_count — the encoder must reject it (never a silent mis-save).
    let mut t = borders_table();
    t.hashes = vec![0xFFFF_FFFF_FFFF_FFFF, 20, 30];
    let mut tables = BTreeMap::new();
    tables.insert(ctr_base_key(ECtrType::Borders, &[0]), t);
    assert!(encode_ctr_model_parts(&CtrData { tables }).is_err());
}

#[test]
fn encode_ctr_model_parts_rejects_count_exceeding_i32() {
    // A CTRBlob value that overflows i32 is a typed error, never `as` truncation.
    let mut t = borders_table();
    t.int_counts = vec![vec![i64::from(i32::MAX) + 1, 0], vec![1, 1], vec![0, 1]];
    let mut tables = BTreeMap::new();
    tables.insert(ctr_base_key(ECtrType::Borders, &[0]), t);
    assert!(encode_ctr_model_parts(&CtrData { tables }).is_err());
}
