//! Unit tests for the online-HNSW port ([`crate::hnsw`]).
//!
//! These are STRUCTURAL / invariant checks (naive-vs-exact agreement on small
//! prefixes, distance-order sanity, insertion bookkeeping). The bit-for-bit
//! agreement with upstream's instrumented `knn_neighbors` dump is asserted by the
//! `cb-oracle` integration test `hnsw_neighbor_oracle_test.rs` (SC-1), which is the
//! authoritative FEAT-07 gate.

use crate::hnsw::{l2_sqr_f32, HnswKnnCloud, OnlineHnswIndex};

/// Brute-force-exact k-NN (ascending distance, ascending-id tie-break) used to
/// cross-check the HNSW cloud on prefixes where the graph degenerates to exact.
fn exact_knn(points: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = points
        .iter()
        .enumerate()
        .map(|(id, p)| (l2_sqr_f32(query, p), id))
        .collect();
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

#[test]
fn l2_sqr_matches_scalar_reference() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [0.0_f32, 0.0, 0.0, 0.0];
    // ((1+4)+9)+16 == 30 in the SSE lane reduction order.
    assert_eq!(l2_sqr_f32(&a, &b), 30.0);
    assert_eq!(l2_sqr_f32(&a, &a), 0.0);
    // Non-multiple-of-4 length: scalar tail folded into lane 0.
    let c = [1.0_f32, 1.0, 1.0, 1.0, 2.0];
    let d = [0.0_f32; 5];
    assert_eq!(l2_sqr_f32(&c, &d), 8.0);
}

#[test]
fn empty_index_returns_no_neighbors() {
    let idx = OnlineHnswIndex::new(4, 3, 300).expect("opts valid");
    assert!(idx.is_empty());
    assert!(idx.get_nearest_neighbors(&[0.0, 0.0, 0.0, 0.0], 3).is_empty());
}

#[test]
fn small_prefixes_agree_with_exact() {
    // With MaxNeighbors=3, prefixes of size <= 4 (== MaxNeighbors + 1) use the
    // naive-exact path, so the HNSW cloud must equal brute-force-exact there.
    // Points are chosen with strictly-distinct pairwise distances so the ordering
    // is unambiguous (distance ties are resolved by the libc++ heap, NOT by an
    // ascending-id rule — that tie behavior is gated bit-for-bit by the upstream
    // `knn_neighbors` dump oracle, not here).
    let pts: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![3.0, 0.0, 0.0, 0.0],
        vec![7.0, 0.0, 0.0, 0.0],
        vec![15.0, 0.0, 0.0, 0.0],
    ];
    let mut cloud = HnswKnnCloud::new(4, 3).expect("cloud");
    for (i, p) in pts.iter().enumerate() {
        // Read-before-update: query the prefix, compare to exact over the prefix.
        let prefix = &pts[..i];
        let got = cloud.nearest_neighbors(p, 3).expect("query");
        if i <= 4 {
            let want = exact_knn(prefix, p, 3);
            assert_eq!(got, want, "prefix {i}: naive path must equal exact");
        }
        cloud.add_vector(p).expect("insert");
    }
    assert_eq!(cloud.len(), pts.len());
}

#[test]
fn query_after_full_insert_finds_self_nearest() {
    // Insert a well-separated set; querying an inserted point over the full cloud
    // returns it as the nearest (distance 0), even on the approximate path.
    let pts: Vec<Vec<f32>> = (0..10)
        .map(|i| vec![i as f32 * 10.0, 0.0, 0.0, 0.0])
        .collect();
    let mut cloud = HnswKnnCloud::new(4, 3).expect("cloud");
    for p in &pts {
        cloud.add_vector(p).expect("insert");
    }
    for (i, p) in pts.iter().enumerate() {
        let got = cloud.nearest_neighbors(p, 3).expect("query");
        assert_eq!(got.first().copied(), Some(i), "self is nearest for point {i}");
    }
}

#[test]
fn dim_mismatch_is_typed_error() {
    let mut cloud = HnswKnnCloud::new(4, 3).expect("cloud");
    assert!(cloud.add_vector(&[0.0, 0.0, 0.0]).is_err());
    assert!(cloud.nearest_neighbors(&[0.0, 0.0], 3).is_err());
}
