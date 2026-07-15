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

// ---------------------------------------------------------------------------
// Randomized property / differential oracle.
//
// Upstream ground truth exists only for the frozen 16-row `knn_neighbors` dump
// (`hnsw_neighbor_oracle_test.rs`), so random inputs are checked against the port's
// PROVABLE invariants instead of a per-input upstream reference:
//   1. Naive-path prefixes (size <= MaxNeighbors + 1) MUST equal brute-force-exact
//      — upstream's own `GetNearestNeighborsNaive` is an exact scan, so the port is
//      a strict differential oracle there. Random continuous embeddings make
//      distance ties measure-zero, so the ordering (not just the set) is pinned.
//   2. Approximate-path results are still hard-constrained: ids valid + distinct,
//      returned distances non-decreasing, cardinality <= min(k, prefix), non-empty
//      for a non-empty prefix. (Self-nearest over the full set is asserted ONLY in
//      the naive regime — approximate HNSW can genuinely miss the exact nearest,
//      e.g. at MaxNeighbors=1 the graph is too sparse to always reach self; that is
//      a faithful property of upstream's algorithm, NOT a port defect.)
//   3. The port is deterministic: an identical insertion stream yields an identical
//      neighbor stream (no RNG; the `visited` set is used only for membership).
// ---------------------------------------------------------------------------

/// A tiny self-contained xorshift64* PRNG so the randomized oracle is reproducible
/// and dependency-free (library crates ban `rand`; the test needs no crypto).
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the fixed point at 0.
        XorShift64(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// A uniform `f32` in `[0, bound)`.
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
    /// A `f32` in `[-1, 1)` with ~24 bits of mantissa entropy.
    fn unit(&mut self) -> f32 {
        let m = (self.next_u64() >> 40) as f32; // 24-bit
        (m / f32::from(1u16 << 12) / f32::from(1u16 << 12)) * 2.0 - 1.0
    }
    fn embeddings(&mut self, n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| (0..dim).map(|_| self.unit()).collect())
            .collect()
    }
}

/// Build the read-before-update online prefix stream (each point queried over the
/// prefix of already-inserted points, then inserted) — the exact shape the calcer
/// seam drives. Returns the per-prefix neighbor lists.
fn online_prefix_stream(pts: &[Vec<f32>], dim: usize, k: usize) -> Vec<Vec<usize>> {
    let mut cloud = HnswKnnCloud::new(dim, k).expect("cloud");
    let mut stream = Vec::with_capacity(pts.len());
    for p in pts {
        stream.push(cloud.nearest_neighbors(p, k).expect("query"));
        cloud.add_vector(p).expect("insert");
    }
    stream
}

#[test]
fn random_online_prefix_invariants_and_naive_path_equals_exact() {
    for seed in 1u64..=400 {
        let mut rng = XorShift64::new(seed);
        let n = 2 + rng.below(30) as usize; // 2..=31 points
        let dim = 1 + rng.below(8) as usize; // 1..=8 dims
        let k = 1 + rng.below(6) as usize; // 1..=6 neighbors (MaxNeighbors)
        let pts = rng.embeddings(n, dim);

        let stream = online_prefix_stream(&pts, dim, k);
        assert_eq!(stream.len(), n);

        for (i, got) in stream.iter().enumerate() {
            let prefix = &pts[..i];
            let query = &pts[i];

            // Cardinality: never more than the prefix or k; non-empty once a prefix
            // exists.
            assert!(got.len() <= k.min(i), "seed {seed} prefix {i}: len {} > min(k,i)", got.len());
            if i > 0 {
                assert!(!got.is_empty(), "seed {seed} prefix {i}: empty over non-empty prefix");
            }

            // Ids valid + distinct.
            let mut seen = std::collections::HashSet::new();
            for &id in got {
                assert!(id < i, "seed {seed} prefix {i}: id {id} out of range");
                assert!(seen.insert(id), "seed {seed} prefix {i}: duplicate id {id}");
            }

            // Returned distances non-decreasing (nearest first).
            let dists: Vec<f32> = got.iter().map(|&id| l2_sqr_f32(query, &prefix[id])).collect();
            for w in dists.windows(2) {
                assert!(w[0] <= w[1], "seed {seed} prefix {i}: distances not ascending: {dists:?}");
            }

            // Naive-exact regime: the port is a strict differential oracle vs exact.
            // Random continuous embeddings ⇒ ties are measure-zero, so the ORDER
            // (not just the set) must match brute-force-exact.
            if i <= k + 1 {
                let want = exact_knn(prefix, query, k);
                assert_eq!(
                    *got, want,
                    "seed {seed} prefix {i} (naive path): HNSW != exact\n  got  {got:?}\n  want {want:?}"
                );
            }
        }

        // Full-set apply invariants. Self-nearest is EXACT only in the naive regime
        // (MaxNeighbors + 1 >= n); on the approximate path HNSW may legitimately miss
        // self, so there we assert only the structural constraints.
        let mut full = HnswKnnCloud::new(dim, k).expect("cloud");
        for p in &pts {
            full.add_vector(p).expect("insert");
        }
        let naive_full = k + 1 >= n;
        for (i, p) in pts.iter().enumerate() {
            let got = full.nearest_neighbors(p, k).expect("query");
            assert!(got.len() <= k.min(n), "seed {seed}: full len {} > min(k,n)", got.len());
            assert!(!got.is_empty(), "seed {seed}: full query empty for {i}");
            let mut seen = std::collections::HashSet::new();
            for &id in &got {
                assert!(id < n, "seed {seed}: full id {id} out of range");
                assert!(seen.insert(id), "seed {seed}: full duplicate id {id}");
            }
            let dists: Vec<f32> = got.iter().map(|&id| l2_sqr_f32(p, &pts[id])).collect();
            for w in dists.windows(2) {
                assert!(w[0] <= w[1], "seed {seed}: full distances not ascending: {dists:?}");
            }
            if naive_full {
                assert_eq!(got.first().copied(), Some(i), "seed {seed}: self not nearest for {i}");
            }
        }

        // Determinism: an identical insertion stream reproduces the neighbor stream.
        let stream2 = online_prefix_stream(&pts, dim, k);
        assert_eq!(stream, stream2, "seed {seed}: non-deterministic neighbor stream");
    }
}
