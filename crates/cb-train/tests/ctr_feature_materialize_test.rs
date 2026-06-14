//! Materialization oracle for the combined-projection ONLINE CTR-feature column
//! (ORD-05, Plan 05-11 Task 1 — the per-fold online-CTR-during-growth path of
//! upstream `greedy_tensor_search.cpp` AddTreeCtrs).
//!
//! `materialize_ctr_feature` produces a per-document CTR-feature column from a
//! categorical projection: the combined-projection key per document
//! (`TProjection::combined_hash` over each member's `calc_cat_feature_hash`),
//! remapped to dense first-seen bins, run through the EXISTING read-before-
//! increment online prefix (`online_ctr_prefix_binclf`), then quantized to
//! integer CTR bins against the Borders quantizer (`calc_ctr_online_bin`). These
//! tests lock the four behaviors the materialization must satisfy:
//!
//! 1. NO LEAKAGE — each document's materialized online CTR value equals the
//!    read-before-increment prefix `(good + prior) / (total + 1)` computed BEFORE
//!    its own label (a document never sees its own label).
//! 2. COMBINED KEY — a 2-feature projection {0,1} keys on the combined hash; a
//!    single-feature projection {0} degenerates to the one-fold combined key
//!    (shares the keyspace).
//! 3. QUANTIZATION RANGE — every materialized CTR bin is a finite non-negative
//!    integer in [0, border_count], monotone-consistent with the online value.
//! 4. PRIOR PAIR — the column carries `prior_num`/`prior_denom` as a PAIR; for the
//!    fixture prior `0.5/1` the scalar fed to the online prefix equals
//!    `prior_num / prior_denom == 0.5` exactly.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_data::{calc_cat_feature_hash, stringify_int_category};
use cb_train::{materialize_ctr_feature, ctr_border_count_default, TProjection};

const PRIOR_NUM: f64 = 0.5;
const PRIOR_DENOM: f64 = 1.0;

/// A small two-feature categorical dataset + binclf target. cat0/cat1 are integer
/// category codes stringified via `stringify_int_category` (A4 plain-integer form).
fn small_dataset() -> (Vec<Vec<String>>, Vec<usize>) {
    let cat0_codes: [i64; 8] = [0, 1, 0, 1, 0, 1, 0, 1];
    let cat1_codes: [i64; 8] = [0, 0, 1, 1, 0, 0, 1, 1];
    let target_class: Vec<usize> = vec![1, 0, 1, 0, 1, 1, 0, 0];
    let cat0: Vec<String> = cat0_codes.iter().map(|&c| stringify_int_category(c)).collect();
    let cat1: Vec<String> = cat1_codes.iter().map(|&c| stringify_int_category(c)).collect();
    (vec![cat0, cat1], target_class)
}

/// The reference combined-projection bins for a projection over `cat_columns`
/// using the SAME fold + first-seen remap the production materialization uses.
fn reference_combined_bins(cat_columns: &[Vec<String>], proj: &TProjection) -> Vec<u32> {
    let n = cat_columns.first().map_or(0, Vec::len);
    let keys: Vec<u64> = (0..n)
        .map(|i| {
            let hashes: Vec<u32> = cat_columns
                .iter()
                .map(|col| calc_cat_feature_hash(&col[i]))
                .collect();
            proj.combined_hash(&hashes)
        })
        .collect();
    let mut map: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    keys.iter()
        .map(|&k| {
            let next = map.len() as u32;
            *map.entry(k).or_insert(next)
        })
        .collect()
}

#[test]
fn materialize_no_leakage_under_identity_permutation() {
    let (cat_columns, target_class) = small_dataset();
    let n = target_class.len();
    let proj = TProjection::from_features(&[0, 1]);
    let identity: Vec<i32> = (0..n as i32).collect();
    let border_count = ctr_border_count_default();

    let column = materialize_ctr_feature(
        &cat_columns,
        &proj,
        &identity,
        &target_class,
        PRIOR_NUM,
        PRIOR_DENOM,
        border_count,
    )
    .expect("materialize the combined-projection online CTR feature");

    assert_eq!(column.ctr_value.len(), n, "one CTR value per document");
    assert_eq!(column.bins.len(), n, "one CTR bin per document");

    // Reconstruct the no-leakage prefix INDEPENDENTLY: for each document i (in
    // identity order), the value seen is (good + prior) / (total + 1) computed
    // over only the strictly-earlier documents sharing its combined bin.
    let ref_bins = reference_combined_bins(&cat_columns, &proj);
    let bucket_count = ref_bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut good = vec![0i64; bucket_count];
    let mut total = vec![0i64; bucket_count];
    for i in 0..n {
        let bucket = ref_bins[i] as usize;
        let expected = (good[bucket] as f64 + PRIOR_NUM) / (total[bucket] as f64 + 1.0);
        assert!(
            (column.ctr_value[i] - expected).abs() <= 1e-9,
            "doc {i}: materialized CTR value must be the read-BEFORE-increment prefix (no leakage)"
        );
        // Increment AFTER reading (the document's own label never feeds its value).
        if target_class[i] == 1 {
            good[bucket] += 1;
        }
        total[bucket] += 1;
    }
}

/// Reconstruct the no-leakage online value trajectory for `proj` over the
/// identity permutation, INDEPENDENTLY of the production code, and assert the
/// production column reproduces it. This simultaneously validates the combined
/// key (the values are keyed on `reference_combined_bins`, which folds the
/// projection members via `combined_hash`).
fn assert_matches_reference_prefix(
    cat_columns: &[Vec<String>],
    proj: &TProjection,
    target_class: &[usize],
) {
    let n = target_class.len();
    let identity: Vec<i32> = (0..n as i32).collect();
    let border_count = ctr_border_count_default();
    let column = materialize_ctr_feature(
        cat_columns,
        proj,
        &identity,
        target_class,
        PRIOR_NUM,
        PRIOR_DENOM,
        border_count,
    )
    .expect("materialize the projection");

    let ref_bins = reference_combined_bins(cat_columns, proj);
    let bucket_count = ref_bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut good = vec![0i64; bucket_count];
    let mut total = vec![0i64; bucket_count];
    for i in 0..n {
        let bucket = ref_bins[i] as usize;
        let expected = (good[bucket] as f64 + PRIOR_NUM) / (total[bucket] as f64 + 1.0);
        assert!(
            (column.ctr_value[i] - expected).abs() <= 1e-9,
            "doc {i}: combined-key prefix value must match the reference (keyed on combined_hash)"
        );
        if target_class[i] == 1 {
            good[bucket] += 1;
        }
        total[bucket] += 1;
    }
}

#[test]
fn materialize_combined_key_and_single_feature_degeneration() {
    let (cat_columns, target_class) = small_dataset();

    // 2-feature projection {0,1}: bins keyed on the COMBINED hash (the fold of
    // both per-document cat hashes). The reconstruction over reference_combined_bins
    // (which itself uses combined_hash) matching the production values proves the
    // production keyed on the combined key.
    let proj_pair = TProjection::from_features(&[0, 1]);
    assert_matches_reference_prefix(&cat_columns, &proj_pair, &target_class);

    // Single-feature projection {0}: degenerates to the ONE-FOLD combined key
    // (combined_hash folds exactly one member); shares the same combined keyspace.
    let proj_single = TProjection::single(0);
    assert_matches_reference_prefix(&cat_columns, &proj_single, &target_class);

    // The two projections must NOT produce the same keyspace on this dataset
    // (cat0 alone has 2 distinct values; {0,1} has 4 distinct combined keys), so a
    // single-feature projection is genuinely distinct from the pair.
    let ref_single = reference_combined_bins(&cat_columns, &proj_single);
    let ref_pair = reference_combined_bins(&cat_columns, &proj_pair);
    let distinct_single = ref_single.iter().copied().max().map_or(0, |m| m + 1);
    let distinct_pair = ref_pair.iter().copied().max().map_or(0, |m| m + 1);
    assert!(
        distinct_pair > distinct_single,
        "combined projection {{0,1}} must have a richer keyspace than single {{0}}"
    );
}

#[test]
fn materialize_quantization_range_and_prior_pair() {
    let (cat_columns, target_class) = small_dataset();
    let n = target_class.len();
    let proj = TProjection::from_features(&[0, 1]);
    let identity: Vec<i32> = (0..n as i32).collect();
    let border_count = ctr_border_count_default();
    assert_eq!(border_count, 15, "Borders CTR border count is the upstream default 15");

    let column = materialize_ctr_feature(
        &cat_columns,
        &proj,
        &identity,
        &target_class,
        PRIOR_NUM,
        PRIOR_DENOM,
        border_count,
    )
    .expect("materialize for quantization-range check");

    // Prior PAIR carried verbatim; the scalar fed to the online prefix is
    // prior_num / prior_denom == 0.5 exactly (denom == 1 ⇒ scalar == num).
    assert_eq!(column.prior_num, PRIOR_NUM, "prior numerator carried as a pair");
    assert_eq!(column.prior_denom, PRIOR_DENOM, "prior denominator carried as a pair");
    assert!(
        (column.prior_num / column.prior_denom - 0.5).abs() <= f64::EPSILON,
        "scalar prior = prior_num / prior_denom == 0.5 exactly for the 0.5/1 fixture"
    );

    // Every bin a finite non-negative integer in [0, border_count]; monotone with
    // the online value (a larger online CTR maps to a >= bin).
    for i in 0..n {
        let bin = column.bins[i];
        assert!(
            (bin as usize) <= border_count,
            "doc {i}: CTR bin {bin} must be in [0, {border_count}]"
        );
        assert!(column.ctr_value[i].is_finite(), "doc {i}: online value finite");
        assert!(column.ctr_value[i] >= 0.0, "doc {i}: online value non-negative");
    }
    // Monotone-consistency: sort by online value, bins must be non-decreasing.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| column.ctr_value[a].partial_cmp(&column.ctr_value[b]).unwrap());
    for w in idx.windows(2) {
        let (a, b) = (w[0], w[1]);
        assert!(
            column.bins[a] <= column.bins[b],
            "CTR bin must be monotone non-decreasing in the online value"
        );
    }
}
