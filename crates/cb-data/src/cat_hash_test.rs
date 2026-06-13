//! Bit-exact unit vectors for the CityHash64 port and `CalcCatFeatureHash`
//! reduction. Like the PRNG bitstream vectors in `cb-core/src/rng_test.rs`, these
//! are INTEGER-exact (`assert_eq!`), NOT `<= 1e-5` — a one-bit divergence in the
//! hash silently breaks categorical parity downstream.
//!
//! Ground truth comes from the vendored CityHash 1.0 algorithm
//! (`catboost-master/util/digest/city.cpp`), computed by the standalone oracle
//! `cb-oracle/generator/cityhash_oracle.cpp` (the same algorithm the live
//! catboost library compiles for `CalcCatFeatureHash`). The corpus vectors are
//! mirrored into `cb-oracle/fixtures/cat_hash/config.json` `string_to_ui32` /
//! `string_to_ui64_precursor`.

use super::cat_hash::{
    calc_cat_feature_hash, city_hash_64, perfect_hash_bins, stringify_int_category, PerfectHash,
};
use cb_core::CbError;

/// `(input, expected ui64, expected ui32)` vectors transcribed verbatim from the
/// vendored-source oracle. Coverage spans every CityHash64 length path:
/// - `""` (len 0, the `k2` constant path),
/// - `"3"` / `"3.0"` (len 1 / 3, the `len < 4` byte-mix path; A4 demonstrator),
/// - `"alpha"` (len 5, the `4 <= len <= 8` path),
/// - `"hello world"` (len 11, the `8 < len <= 16` path),
/// - `"aaaaaaaaaaaaaaaa"` (len 16, the 16-byte boundary),
/// - `"aaaaaaaaaaaaaaaaa"` (len 17, the `HashLen17to32` path),
/// - `"this_is_a_long_category_value_over_16_bytes"` (len 43, `HashLen33to64`),
/// - the 70-byte input (len 70, the >64 multi-block loop path).
const VECTORS: &[(&str, u64, u32)] = &[
    ("", 11160318154034397263, 797982799),
    ("3", 11275350073939794026, 593172586),
    ("3.0", 510719357545682165, 3194819829),
    ("alpha", 1772952377847748331, 1296865003),
    ("hello world", 12386028635079221413, 1807130789),
    ("aaaaaaaaaaaaaaaa", 1737540773398541810, 2771219954),
    ("aaaaaaaaaaaaaaaaa", 343169547249257593, 2441481337),
    (
        "this_is_a_long_category_value_over_16_bytes",
        12863576434675650867,
        4193095987,
    ),
    (
        "0123456789012345678901234567890123456789012345678901234567890123456789",
        7040186705704256002,
        569246210,
    ),
];

/// `city_hash_64` reproduces the vendored-source ui64 bit-exactly across every
/// length path (empty, sub-16, 16-byte boundary, 17-32, 33-64, and >64 block).
#[test]
fn city_hash_64_matches_upstream_ui64_vectors() {
    for &(input, expected_ui64, _) in VECTORS {
        assert_eq!(
            city_hash_64(input.as_bytes()),
            expected_ui64,
            "city_hash_64({input:?}) ui64 mismatch"
        );
    }
}

/// `calc_cat_feature_hash(s) == (city_hash_64(s) & 0xffffffff)` and equals the
/// upstream ui32 vectors (`cat_feature.cpp:6-8`).
#[test]
fn calc_cat_feature_hash_matches_upstream_ui32_vectors() {
    for &(input, expected_ui64, expected_ui32) in VECTORS {
        let h32 = calc_cat_feature_hash(input);
        assert_eq!(h32, expected_ui32, "calc_cat_feature_hash({input:?}) mismatch");
        assert_eq!(
            h32 as u64,
            expected_ui64 & 0xffff_ffff,
            "calc_cat_feature_hash({input:?}) != city_hash_64 & 0xffffffff"
        );
    }
}

/// The empty string hashes to the `k2` constant path, distinctly from any
/// non-empty input.
#[test]
fn empty_string_hashes_to_k2_constant() {
    assert_eq!(city_hash_64(b""), 11160318154034397263);
    assert_eq!(calc_cat_feature_hash(""), 797982799);
}

/// A4: integer categories stringify as PLAIN integers (no decimal point), so
/// `'3'` and `'3.0'` hash differently. `stringify_int_category(3)` produces the
/// `'3'` form, never `'3.0'`.
#[test]
fn integer_category_stringifies_to_plain_integer() {
    assert_eq!(stringify_int_category(3), "3");
    assert_eq!(stringify_int_category(-2), "-2");
    assert_eq!(stringify_int_category(10), "10");
    // The plain-integer hash differs from the float-form hash (A4).
    assert_eq!(calc_cat_feature_hash(&stringify_int_category(3)), 593172586);
    assert_ne!(
        calc_cat_feature_hash("3"),
        calc_cat_feature_hash("3.0"),
        "'3' and '3.0' must hash differently (A4)"
    );
}

/// First-seen remap: bin 0 to the first-seen hash, 1 to the next NEW hash, and
/// repeats reuse the prior bin (`cat_feature_perfect_hash_helper.cpp:120` /
/// `:127`). Driven over a column with a repeated value.
#[test]
fn perfect_hash_first_seen_assignment() {
    let column = ["alpha", "beta", "alpha", "gamma", "beta", "alpha"];
    let bins = perfect_hash_bins(&column).expect("within uniq bound");
    assert_eq!(bins, vec![0, 1, 0, 2, 1, 0]);
}

/// `PerfectHash::remap` reuses the assigned bin on repeats and advances
/// `len()` only for new hashes.
#[test]
fn perfect_hash_remap_reuses_bins() {
    let mut ph = PerfectHash::new();
    assert!(ph.is_empty());
    assert_eq!(ph.remap(100).unwrap(), 0);
    assert_eq!(ph.remap(200).unwrap(), 1);
    assert_eq!(ph.remap(100).unwrap(), 0); // repeat -> same bin
    assert_eq!(ph.remap(300).unwrap(), 2);
    assert_eq!(ph.len(), 3);
}

/// The uniq-count bound returns a typed [`CbError::OutOfRange`] (NOT a panic)
/// when a new hash would exceed the cap. Exercised with a tiny cap so the bound
/// is reachable without materializing `u32::MAX` distinct hashes; the production
/// `remap` uses the real `MAX_UNIQ_CAT_VALUES = u32::MAX` cap
/// (`cat_feature_perfect_hash_helper.cpp:53-54`, Security V5 / T-02-11).
#[test]
fn perfect_hash_uniq_bound_returns_error_not_panic() {
    let mut ph = PerfectHash::new();
    // Fill the map to a cap of 2 distinct hashes.
    assert_eq!(ph.remap_bounded(10, 2).unwrap(), 0);
    assert_eq!(ph.remap_bounded(20, 2).unwrap(), 1);
    // A repeat of an existing hash is always fine (no new bin needed).
    assert_eq!(ph.remap_bounded(10, 2).unwrap(), 0);
    // A THIRD distinct hash would exceed the cap -> typed error, no panic.
    match ph.remap_bounded(30, 2) {
        Err(CbError::OutOfRange(msg)) => {
            assert!(msg.contains("unique values"), "got: {msg}");
        }
        other => panic!("expected OutOfRange on overflow, got {other:?}"),
    }
}
