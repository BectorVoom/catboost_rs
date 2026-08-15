//! The ordered approximant advances BOTH halves of each body/tail — body included.
//!
//! # The evidence was already committed
//!
//! `cb_train::ordered_approx_delta_simple` freezes the body rows at `0`, documenting
//! them as "the estimation prefix, not updated here". The committed UPSTREAM dump
//! `cb-oracle/fixtures/ordered_boost/ordered_approx_iter0.npy` says otherwise: all 30
//! of its per-object entries are NON-ZERO, body prefix included. That contradiction
//! sat in the repository unnoticed because the only test touching that fixture
//! (`ordered_boost_committed_approx_is_well_formed`) asserts finiteness, length and
//! boundedness — never a value — since the raw inputs behind it are uncommitted.
//!
//! [`cb_train::ordered_approx_delta_with_body`] implements the full delta:
//!   * BODY rows take the leaf average over the body prefix;
//!   * TAIL rows take the running add-then-read leaf average.
//!
//! Measured effect of using it for the per-body/tail scoring approximants, at
//! `permutation_count = 1` over a 60-cell corpus × iteration grid: 27/60 → 44/60
//! cells matching catboost 1.2.10, with ZERO cells regressed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_train::{ordered_approx_delta_simple, ordered_approx_delta_with_body};
use ndarray::Array1;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// A small hand-auditable scenario: 8 objects, identity order, body `[0, 3)`,
/// tail `[3, 8)`, two leaves alternating.
fn scenario() -> (Vec<usize>, Vec<f64>, Vec<f64>, Vec<i32>) {
    let leaf_of = vec![0, 1, 0, 1, 0, 1, 0, 1];
    let der = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let weights = vec![1.0; 8];
    let perm: Vec<i32> = (0..8).collect();
    (leaf_of, der, weights, perm)
}

/// THE claim: body rows are non-zero under the full delta, and zero under the
/// tail-only one. Both are asserted so the difference is explicit rather than
/// implied.
#[test]
fn the_full_delta_advances_body_rows_and_the_tail_only_delta_does_not() {
    let (leaf_of, der, weights, perm) = scenario();
    let (body_finish, tail_finish, n_leaves, scaled_l2) = (3usize, 8usize, 2usize, 1.0);

    let full = ordered_approx_delta_with_body(
        &leaf_of, &der, &weights, &perm, body_finish, tail_finish, n_leaves, scaled_l2,
    )
    .expect("full delta");
    let tail_only = ordered_approx_delta_simple(
        &leaf_of, &der, &weights, &perm, body_finish, tail_finish, 0.0, n_leaves, scaled_l2,
    )
    .expect("tail-only delta");

    for p in 0..body_finish {
        assert!(
            full[p] != 0.0,
            "body row {p} must ADVANCE under the full delta, got 0"
        );
        assert_eq!(
            tail_only[p], 0.0,
            "body row {p} must stay 0 under the tail-only delta (its documented contract)"
        );
    }

    // The TAIL halves must agree exactly — the fix changes the body treatment only,
    // so a difference there would mean the running walk was perturbed too.
    for p in body_finish..tail_finish {
        assert!(
            (full[p] - tail_only[p]).abs() < 1e-12,
            "tail row {p}: full {} vs tail-only {} — the two must share the running walk",
            full[p],
            tail_only[p]
        );
    }
}

/// Body rows in one leaf share ONE value (the prefix leaf average), while tail rows
/// vary — the structural signature that distinguishes "estimated on the prefix" from
/// "running".
#[test]
fn body_rows_of_a_leaf_share_the_prefix_average() {
    let (leaf_of, der, weights, perm) = scenario();
    let full = ordered_approx_delta_with_body(&leaf_of, &der, &weights, &perm, 4, 8, 2, 1.0)
        .expect("full delta");
    // Body = positions 0..4 → leaves [0,1,0,1]; the two leaf-0 rows must agree.
    assert!(
        (full[0] - full[2]).abs() < 1e-12,
        "both leaf-0 body rows must carry the SAME prefix average: {} vs {}",
        full[0],
        full[2]
    );
    assert!(
        (full[1] - full[3]).abs() < 1e-12,
        "both leaf-1 body rows must carry the SAME prefix average"
    );
    // And the tail rows of a leaf must NOT all be equal (they are a running average).
    assert!(
        (full[4] - full[6]).abs() > 1e-12,
        "tail rows of the same leaf must differ — they are a RUNNING average, not a \
         single prefix estimate"
    );
}

/// The committed upstream dump has NO zero entries. This is the external evidence for
/// the body-advances rule, asserted here so it is a live fact rather than a note.
#[test]
fn the_committed_upstream_ordered_approx_has_no_zero_body_entries() {
    let approx: Array1<f64> = read_npy(fixture("ordered_boost/ordered_approx_iter0.npy"))
        .expect("committed upstream ordered approx must load");
    assert!(!approx.is_empty(), "fixture must be non-empty");
    let zeros = approx.iter().filter(|v| **v == 0.0).count();
    assert_eq!(
        zeros,
        0,
        "the committed UPSTREAM ordered approx has {zeros} zero entries of {}; a \
         tail-only delta would leave the whole body prefix at 0, so this fixture is \
         direct evidence that upstream advances body rows too",
        approx.len()
    );
}
