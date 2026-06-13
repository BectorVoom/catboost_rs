//! Categorical perfect-hash oracle: proves the Rust CityHash64 port +
//! `CalcCatFeatureHash` + first-seen perfect-hash remap reproduce upstream
//! per-object hashes and bins on the explicit-categorical corpus (DATA-05).
//!
//! Oracle target: `cb-oracle/fixtures/cat_hash/cat_hashes.npy` (per-object
//! `CalcCatFeatureHash`, f64-encoded ui32) and `perfect_hash_bins.npy`
//! (per-object first-seen bins). Both are flat: column c0 (n_rows) then column c1
//! (n_rows). The corpus strings are reconstructed by tiling the per-column
//! first-seen orders from `cat_hash/config.json` to `n_rows` (the corpus is an
//! exact cycle of its first-seen order, verified at generation time), then hashed
//! with [`cb_data::calc_cat_feature_hash`] and remapped with
//! [`cb_data::perfect_hash_bins`].
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors `per_stage_oracle_test.rs:9`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_data::{calc_cat_feature_hash, perfect_hash_bins};
use cb_oracle::load_f64_vec;

/// Resolve a path under `cb-oracle/fixtures/` from cb-data's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Read the cat_hash config and return the per-column first-seen orders.
fn first_seen_orders() -> (Vec<String>, Vec<String>) {
    let raw = std::fs::read_to_string(fixture("cat_hash/config.json"))
        .expect("cat_hash/config.json must exist");
    let cfg: serde_json::Value = serde_json::from_str(&raw).expect("config.json must parse");
    let to_vec = |key: &str| -> Vec<String> {
        cfg[key]
            .as_array()
            .unwrap_or_else(|| panic!("config.{key} must be an array"))
            .iter()
            .map(|v| v.as_str().expect("first-seen entries are strings").to_owned())
            .collect()
    };
    (to_vec("c0_first_seen_order"), to_vec("c1_first_seen_order"))
}

/// Tile a per-column first-seen order to `n_rows` (the corpus is an exact cycle).
fn tile(order: &[String], n_rows: usize) -> Vec<String> {
    (0..n_rows).map(|i| order[i % order.len()].clone()).collect()
}

/// Per-object `CalcCatFeatureHash` matches `cat_hashes.npy` for both columns,
/// and the first-seen perfect-hash remap matches `perfect_hash_bins.npy`.
#[test]
fn cat_hashes_and_perfect_hash_bins_match_oracle() {
    let hashes_flat = load_f64_vec(&fixture("cat_hash/cat_hashes.npy")).unwrap();
    let bins_flat = load_f64_vec(&fixture("cat_hash/perfect_hash_bins.npy")).unwrap();
    assert_eq!(
        hashes_flat.len(),
        bins_flat.len(),
        "cat_hashes and perfect_hash_bins must have the same length"
    );
    // Flat layout is two equal-length columns (c0 then c1).
    assert_eq!(hashes_flat.len() % 2, 0, "flat length must be 2 * n_rows");
    let n_rows = hashes_flat.len() / 2;

    let (o0, o1) = first_seen_orders();
    let c0 = tile(&o0, n_rows);
    let c1 = tile(&o1, n_rows);

    for (col_idx, column) in [&c0, &c1].into_iter().enumerate() {
        let offset = col_idx * n_rows;

        // 1) Per-object CalcCatFeatureHash matches the oracle hashes (bit-exact).
        let actual_hashes: Vec<f64> = column
            .iter()
            .map(|s| calc_cat_feature_hash(s) as f64)
            .collect();
        for (row, &actual) in actual_hashes.iter().enumerate() {
            let expected = hashes_flat[offset + row];
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "column {col_idx} row {row}: CalcCatFeatureHash {actual} != oracle {expected}"
            );
        }

        // 2) First-seen perfect-hash bins match the oracle bins (integer-exact).
        let column_refs: Vec<&str> = column.iter().map(String::as_str).collect();
        let actual_bins = perfect_hash_bins(&column_refs).expect("within uniq bound");
        for (row, &bin) in actual_bins.iter().enumerate() {
            let expected = bins_flat[offset + row];
            assert_eq!(
                bin as f64,
                expected,
                "column {col_idx} row {row}: bin {bin} != oracle {expected}"
            );
        }
    }
}
