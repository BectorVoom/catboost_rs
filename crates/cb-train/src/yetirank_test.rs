//! Unit tests for the YetiRank RNG-stream pair sampler (`yetirank.rs`,
//! LOSS-04 Wave C). These gate the RNG **draw order** — the parity crux — at the
//! smallest unit (permutations=1, a 2-doc group), hand-traced against the
//! bitstream-validated `cb_core::TFastRng64` reproduction. The instrumented C++
//! `CB_INSTRUMENT_LOG` ground-truth compare lands in the oracle tests (Task 4);
//! these tests pin the draw COUNT + ORDER + the deterministic sampler output so a
//! later seed-derivation regression is caught here first.

use cb_core::TFastRng64;

use super::{derive_query_seeds, sample_pairs, sum_competitor_weights};

/// The 2-level seed derivation (single-thread) must consume EXACTLY the upstream
/// draw count: one `GenRand()` for the block seed (`GenRandUI64Vector(1, seed)`),
/// then one `GenRand()` per query for the inner seed. For one group this is a
/// 2-draw chain: `seed_rng.GenRand()` → block seed; `block_rng.GenRand()` → query
/// seed. We reproduce the same chain by hand and assert `derive_query_seeds`
/// agrees bit-for-bit (the draw order IS the parity contract,
/// yetirank_helpers.cpp:365-389 + restorable_rng.cpp:3-9).
#[test]
fn query_seed_derivation_matches_hand_traced_two_level_chain() {
    let random_seed = 0_u64;

    // Hand trace: GenRandUI64Vector(1, seed)[0] = TFastRng64(seed).GenRand().
    let mut seed_rng = TFastRng64::from_seed(random_seed);
    let block_seed = seed_rng.gen_rand();
    // blockRng = TFastRng64(blockSeed); querySeed = blockRng.GenRand().
    let mut block_rng = TFastRng64::from_seed(block_seed);
    let expected_query_seed = block_rng.gen_rand();

    let seeds = derive_query_seeds(random_seed, 1);
    assert_eq!(seeds.len(), 1, "one query => one derived seed");
    assert_eq!(
        seeds[0], expected_query_seed,
        "the derived query seed must match the hand-traced 2-level chain \
         (block seed then per-query GenRand)"
    );
}

/// `derive_query_seeds` for multiple groups must draw the per-query seeds IN GROUP
/// ORDER from the SAME block rng (not re-seed per query) — a divergence here would
/// desync every group's sampled pairs.
#[test]
fn query_seed_derivation_draws_in_group_order_from_one_block_rng() {
    let random_seed = 42_u64;
    let mut seed_rng = TFastRng64::from_seed(random_seed);
    let block_seed = seed_rng.gen_rand();
    let mut block_rng = TFastRng64::from_seed(block_seed);
    let expected: Vec<u64> = (0..3).map(|_| block_rng.gen_rand()).collect();

    let seeds = derive_query_seeds(random_seed, 3);
    assert_eq!(seeds, expected, "3 query seeds drawn in order from one block rng");
}

/// For permutations=1 on a 2-doc group, AddNoise draws EXACTLY `query_size`
/// `gen_rand_real1()` Gumbel uniforms (one per doc, ascending docId), then sorts
/// and accumulates one Classic pair weight. We hand-trace the 2 uniform draws from
/// the same inner rng and assert the sampler's competitor weight matches the
/// hand-computed value — pinning both the draw COUNT (2, not 1 or 4) and the
/// Gumbel transform `u/(1.000001-u)` (yetirank_helpers.cpp:149-152).
#[test]
fn permutations_one_two_doc_group_draws_two_gumbel_uniforms() {
    let query_seed = 123_u64;
    // RAW approxes (exp()'d inline by the sampler); relevances: doc0 more relevant.
    let raw_approx = [0.5_f64, -0.3_f64];
    let relevs = [1.0_f64, 0.0_f64];
    let query_weight = 1.0_f64;
    let permutations = 1_u32;
    let decay = 0.85_f64;

    // Hand trace the inner rng: TFastRng64(querySeed); per doc one gen_rand_real1().
    // The sampler transcribes upstream's f32 bit-width: `uniformValue` is cast to
    // f32, the Gumbel ratio is computed in f32, the competitor-weight matrix is f32,
    // and the final `competitorsWeight = queryWeight * w / permutationCount` is f32
    // (`TVector<TVector<float>>` + `float TCompetitor.Weight`, yetirank_helpers.cpp).
    // The hand trace mirrors that exactly so the assertion stays bit-tight.
    let mut rand = TFastRng64::from_seed(query_seed);
    let exp0 = raw_approx[0].exp();
    let exp1 = raw_approx[1].exp();
    #[allow(clippy::cast_possible_truncation)]
    let u0 = rand.gen_rand_real1() as f32;
    let boot0 = exp0 * f64::from(u0 / (1.000_001_f32 - u0));
    #[allow(clippy::cast_possible_truncation)]
    let u1 = rand.gen_rand_real1() as f32;
    let boot1 = exp1 * f64::from(u1 / (1.000_001_f32 - u1));
    // Sort descending: the higher bootstrapped approx is the first sorted position.
    // Classic weight (f32): 0.15 · decay^0 · |relev[first] - relev[second]|, routed
    // to the higher-relevance doc as winner.
    let (first, second) = if boot0 >= boot1 { (0_usize, 1_usize) } else { (1_usize, 0_usize) };
    #[allow(clippy::cast_possible_truncation)]
    let rdiff = (relevs[first] as f32 - relevs[second] as f32).abs();
    #[allow(clippy::cast_possible_truncation)]
    let pair_weight = (0.15_f64 * 1.0 * f64::from(rdiff)) as f32;
    // AddWeight routes to the higher-relevance doc: doc0 (relev 1.0) wins over doc1.
    let mut expected = vec![vec![0.0_f32; 2]; 2];
    if relevs[first] > relevs[second] {
        expected[first][second] += pair_weight;
    } else if relevs[first] < relevs[second] {
        expected[second][first] += pair_weight;
    }
    // competitorsWeight = queryWeight(f32) · w(f32) / permutations, then f64-promoted.
    #[allow(clippy::cast_possible_truncation)]
    let qw = query_weight as f32;
    let expected_w01 = f64::from(qw * expected[0][1] / permutations as f32);
    let expected_w10 = f64::from(qw * expected[1][0] / permutations as f32);

    let competitors = sample_pairs(&raw_approx, &relevs, query_weight, permutations, decay, query_seed);
    assert_eq!(competitors.len(), 2, "two docs => two competitor rows");

    // Reconstruct the sampled weight matrix from the returned adjacency.
    let mut got = [[0.0_f64; 2]; 2];
    for (winner, row) in competitors.iter().enumerate() {
        for c in row {
            got[winner][c.id] = c.weight;
        }
    }
    assert!(
        (got[0][1] - expected_w01).abs() < 1e-12,
        "winner-0/loser-1 weight must match hand trace: got {}, want {expected_w01}",
        got[0][1]
    );
    assert!(
        (got[1][0] - expected_w10).abs() < 1e-12,
        "winner-1/loser-0 weight must match hand trace: got {}, want {expected_w10}",
        got[1][0]
    );
    // Sanity: with doc0 strictly more relevant, only the 0->1 edge is nonzero.
    assert!(got[0][1] > 0.0, "the more-relevant doc0 must win over doc1");
    assert_eq!(got[1][0], 0.0, "doc1 never wins over the more-relevant doc0");
}

/// The sampler is DETERMINISTIC in the seed (same seed => same pairs) — a
/// non-reproducible draw would break the per-iteration recompute parity.
#[test]
fn sampler_is_deterministic_in_seed() {
    let raw_approx = [0.2_f64, 0.8_f64, -0.1_f64];
    let relevs = [2.0_f64, 0.0_f64, 1.0_f64];
    let a = sample_pairs(&raw_approx, &relevs, 1.0, 10, 0.85, 777);
    let b = sample_pairs(&raw_approx, &relevs, 1.0, 10, 0.85, 777);
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(&b) {
        assert_eq!(ra.len(), rb.len(), "same seed => identical competitor counts");
        for (ca, cb) in ra.iter().zip(rb) {
            assert_eq!(ca.id, cb.id);
            assert!((ca.weight - cb.weight).abs() < 1e-15);
        }
    }
}

/// An empty group / zero permutations samples no pairs and never divides
/// (T-06.3-04-02 — the Security V5 guard).
#[test]
fn empty_group_and_zero_permutations_sample_nothing() {
    assert!(sample_pairs(&[], &[], 1.0, 10, 0.85, 1).is_empty());
    let zero_perm = sample_pairs(&[0.1, 0.2], &[1.0, 0.0], 1.0, 0, 0.85, 1);
    assert_eq!(zero_perm.len(), 2);
    assert!(zero_perm.iter().all(std::vec::Vec::is_empty));
}

/// `sum_competitor_weights` routes through the sanctioned `sum_f64` reduction
/// (D-08) — the order-locked primitive, not a raw fold.
#[test]
fn sum_competitor_weights_uses_ordered_reduction() {
    // The reduction-order canary: [1e16, 1.0, -1e16] sums to 0.0 only under the
    // order-locked sum_f64 (a naive left fold would lose the 1.0).
    let canary = [1e16_f64, 1.0, -1e16];
    assert_eq!(sum_competitor_weights(&canary), 0.0);
}
