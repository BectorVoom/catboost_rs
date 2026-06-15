//! DE-RISK GATE (plan 05-19, ORD-01 / bar (c)): feeding S-ordered cat data through the
//! UNMODIFIED production `materialize_ctr_feature` reproduces the self-consistent averaging-fold
//! online-CTR bins bit-exact, for pc=4 AND pc=1.
//!
//! This proves the bar-(c) leaf-value gap is purely an ORDER problem (the missing initial
//! learn-set shuffle `S`), NOT a CTR-math problem — closing the stale 05-17 "deeper CTR-subsystem
//! divergence" worry (which rested on the internally-inconsistent blocker fixture, superseded by
//! `live_trainer_self_consistent.json`). The original-object averaging CTR order is
//! `Q = [S[p] for p in P_avg]`, where `S = create_shuffled_indices(n, random_seed)` (the upstream
//! initial learn-set shuffle = `fisher_yates_permutation`) and `P_avg` is the averaging fold
//! permutation over the S-shuffled data (captured in the self-consistent JSON).
//!
//! MUST pass before train_inner is touched (Task 3). All constants below trace to
//! `crates/cb-train/tests/fixtures/multi_permutation_fold/live_trainer_self_consistent.json`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use cb_data::stringify_int_category;
use cb_train::{create_shuffled_indices, materialize_ctr_feature, TProjection};
use ndarray::Array2;
use ndarray_npy::read_npy;

const PRIOR_NUM: f64 = 0.5;
const PRIOR_DENOM: f64 = 1.0;
const CTR_BORDER_COUNT: usize = 15;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cb-oracle/fixtures")
        .join(rel)
}

/// X_cat.npy ([N,2] int32) -> per-feature Vec<String> (PLAIN-integer stringify, A4).
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture("tensor_ctr_e2e/X_cat.npy")).unwrap();
    (0..x.ncols())
        .map(|c| {
            x.column(c)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

fn load_target_class() -> Vec<usize> {
    let y: ndarray::Array1<f64> = read_npy(fixture("tensor_ctr_e2e/y.npy")).unwrap();
    y.iter().map(|&t| usize::from(t > 0.5)).collect()
}

/// Drive the UNMODIFIED materialize_ctr_feature with Q = S∘P_avg and assert that, read back in
/// Q order, the per-object bins equal the captured `avg_bins_ctr_order` (the UNAMBIGUOUS CTR-order
/// ground truth) bit-exact. (Comparing in CTR order avoids the within-(cat,y)-group object-assignment
/// ambiguity: many object orders share a bucket's bin multiset; the captured CTR-order sequence is
/// the single source of truth.)
fn assert_s_order_reproduces_bins(p_avg: &[i32], avg_bins_ctr_order: &[u32]) {
    let cat_columns = load_cat_columns();
    let target_class = load_target_class();
    let n = target_class.len();
    assert_eq!(n, 30, "tensor_ctr_e2e fixture is N=30");

    // S = the upstream initial learn-set shuffle (random_seed=0); object S[k] at shuffled pos k.
    let s = create_shuffled_indices(n, 0);
    // Q (original-object averaging CTR order) = S composed with P_avg (averaging perm over shuffled).
    let q: Vec<i32> = p_avg.iter().map(|&p| s[p as usize]).collect();

    // single cat feature 0 (Borders), prior 0.5/1.0, border_count 15 — the upstream config.
    let proj = TProjection::from_features(&[0]);
    let column = materialize_ctr_feature(
        &cat_columns,
        &proj,
        &q,
        &target_class,
        PRIOR_NUM,
        PRIOR_DENOM,
        CTR_BORDER_COUNT,
    )
    .expect("materialize_ctr_feature over S∘P_avg must succeed");

    // materialize returns OBJECT-order bins; read them back in Q (CTR) order and compare to the
    // captured CTR-order ground truth.
    let bins_ctr_order: Vec<u32> = q.iter().map(|&o| column.bins[o as usize]).collect();
    assert_eq!(
        bins_ctr_order, avg_bins_ctr_order,
        "S-order (Q = S∘P_avg) through the UNMODIFIED materialize_ctr_feature must reproduce the \
         captured averaging CTR bins (CTR order) bit-exact"
    );
}

#[test]
fn s_order_reproduces_pc4_averaging_ctr_bins() {
    // live_trainer_self_consistent.json -> pc4.averaging_fold
    let p_avg: [i32; 30] = [
        11, 18, 15, 29, 16, 12, 0, 7, 19, 27, 4, 3, 5, 17, 14, 25, 9, 20, 8, 23, 6, 28, 26, 24, 2,
        13, 21, 22, 10, 1,
    ];
    // avg_bins_ctr_order (CTR materialization order = P_avg over shuffled data)
    let avg_bins: [u32; 30] = [
        7, 7, 7, 11, 3, 7, 2, 11, 7, 12, 9, 12, 11, 10, 3, 2, 11, 1, 7, 13, 5, 7, 8, 11, 1, 9, 1,
        10, 1, 13,
    ];
    assert_s_order_reproduces_bins(&p_avg, &avg_bins);
}

#[test]
fn s_order_reproduces_pc1_averaging_ctr_bins() {
    // live_trainer_self_consistent.json -> pc1.averaging_fold
    let p_avg: [i32; 30] = [
        10, 17, 25, 3, 6, 23, 7, 28, 4, 5, 1, 14, 0, 22, 21, 8, 12, 26, 11, 18, 2, 20, 24, 13, 29,
        19, 15, 9, 27, 16,
    ];
    // avg_bins_ctr_order (CTR materialization order = P_avg over shuffled data)
    let avg_bins: [u32; 30] = [
        7, 7, 7, 7, 7, 11, 12, 11, 11, 12, 13, 3, 3, 9, 2, 10, 13, 8, 1, 2, 1, 1, 12, 9, 13, 10,
        13, 13, 13, 1,
    ];
    assert_s_order_reproduces_bins(&p_avg, &avg_bins);
}
