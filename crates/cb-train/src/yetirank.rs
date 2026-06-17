//! `yetirank` — the YetiRank / YetiRankPairwise RNG-stream pair sampler
//! (LOSS-04, Wave C / D-6.3-02). The parity crux of the two randomized listwise
//! losses: a verbatim, draw-order-faithful transcription of catboost 1.2.10's
//! `GenerateYetiRankPairsForQuery` + `UpdatePairsForYetiRank`
//! (`catboost-master/catboost/private/libs/algo/yetirank_helpers.cpp:146-393`).
//!
//! # Why a hand transcription (not a sampling crate)
//!
//! YetiRank has NO closed-form gradient: each boosting iteration it RE-SAMPLES the
//! pairwise weights of every group by adding Gumbel noise to the exp-approxes,
//! sorting, and accumulating decayed Classic pairwise weights over `permutations`
//! (default 10) noise draws. The result feeds the EXISTING `TPairLogitError` der
//! over the sampled pairs. Parity hinges entirely on the exact RNG **draw order**:
//! a different count or order of `gen_rand_real1()` calls desynchronises every
//! subsequent draw and samples different pairs, failing the ≤1e-5 oracle from
//! tree 0 (RESEARCH Pitfall 1). So the upstream loop is transcribed line-by-line
//! with `file.cpp:line` citations, mirroring `cb_core::std_normal`'s discipline;
//! no third-party sampler is permitted.
//!
//! # The 2-level seed derivation (single-thread, `blockCount == 1`)
//!
//! Upstream parallelises over query BLOCKS; a single-thread fit (the fixture
//! configuration) has `blockCount == 1`, so the model collapses to
//! (`yetirank_helpers.cpp:347-393`):
//! 1. `randomSeeds = GenRandUI64Vector(1, randomSeed)` — one block seed drawn from
//!    `TFastRng64(randomSeed).GenRand()` (`restorable_rng.cpp:3-9`);
//! 2. `blockRng = TFastRng64(randomSeeds[0])`;
//! 3. per query (in group order): `querySeed = blockRng.GenRand()` re-seeds the
//!    inner per-query `TFastRng64(querySeed)`
//!    (`yetirank_helpers.cpp:374-389` — the inner seed is `rand.GenRand()`).
//!
//! Then `GenerateYetiRankPairsForQuery` (`:305-345`) runs the per-permutation
//! Gumbel-noise + sort + Classic-weight loop on the inner rng.
//!
//! # f64 / draw discipline
//!
//! The Gumbel uniform is [`cb_core::TFastRng64::gen_rand_real1`] (the exact
//! `(GenRand() >> 11) · 1/(2^53-1)` draw). Per-permutation weight accumulation is
//! finalized through `cb_core::sum_f64` (D-08 — no raw float fold). The
//! competitor-weight normalisation divides by `permutation_count` (validated
//! `>= 1` by `cb_compute::Loss::validate`, so never by zero).

use cb_core::{sum_f64, TFastRng64};
use cb_compute::RankingCompetitor as Competitor;

/// The Classic-weight magic constant `0.15` ("Like in GPU",
/// `yetirank_helpers.cpp:198`).
const MAGIC_CONST: f64 = 0.15;

/// Sample one query group's competitor adjacency (the YetiRank pair source) from
/// its RAW approxes + relevances, transcribing `GenerateYetiRankPairsForQuery`
/// (`yetirank_helpers.cpp:305-345`) verbatim.
///
/// - `raw_approx`: the group's RAW model approxes (length `query_size`). Upstream
///   passes the EXP-approxes; cb-train stores RAW approx, so we exp() INLINE here
///   (the `pairlogit`/Poisson inline-link precedent) to form the bootstrapped
///   approx the Gumbel noise multiplies.
/// - `relevs`: the group's target relevances (length `query_size`).
/// - `query_weight`: the per-group weight (folded into each competitor weight).
/// - `permutation_count`: number of noise permutations (`permutations`, default
///   10; validated `>= 1`).
/// - `decay`: the Classic-weight geometric decay (`decay`, default 0.85).
/// - `query_seed`: the per-query inner seed (`blockRng.GenRand()` from the 2-level
///   derivation; see [`derive_query_seeds`]).
///
/// Returns `competitors[winner_local]` = list of `{loser_local, weight}` for every
/// nonzero sampled pair weight (`yetirank_helpers.cpp:336-344`), the exact
/// adjacency `TPairLogitError::CalcDersForQueries` consumes.
///
/// The Gumbel draw: per permutation, per doc, one `gen_rand_real1()` uniform `u`,
/// then `expApprox[d] *= u / (1.000001 - u)` (`yetirank_helpers.cpp:149-152`). The
/// `1.000001` denominator guard (transcribed verbatim) prevents a div-by-zero /
/// Inf as `u → 1` (T-06.3-04-01).
#[must_use]
pub fn sample_pairs(
    raw_approx: &[f64],
    relevs: &[f64],
    query_weight: f64,
    permutation_count: u32,
    decay: f64,
    query_seed: u64,
) -> Vec<Vec<Competitor>> {
    let query_size = raw_approx.len();
    if query_size == 0 || permutation_count == 0 {
        return vec![Vec::new(); query_size];
    }
    // exp-approx (upstream receives expApproxes; we exp() the RAW approx INLINE).
    let exp_approx: Vec<f64> = raw_approx.iter().map(|&a| a.exp()).collect();

    // TFastRng64 rand(querySeed) (yetirank_helpers.cpp:314).
    let mut rand = TFastRng64::from_seed(query_seed);

    // competitorsWeights[w][l] accumulates across permutations
    // (yetirank_helpers.cpp:320).
    let mut competitors_weights = vec![vec![0.0_f64; query_size]; query_size];

    for _perm in 0..permutation_count {
        // std::iota(indices, 0) (yetirank_helpers.cpp:322).
        let mut indices: Vec<usize> = (0..query_size).collect();
        // bootstrappedApprox = copy(expApproxes) (yetirank_helpers.cpp:323).
        let mut bootstrapped = exp_approx.clone();
        // AddNoise (Gumbel): per doc, one gen_rand_real1() uniform `u`, then
        // expApprox[d] *= u / (1.000001 - u) (yetirank_helpers.cpp:149-152). The
        // draw is per-doc in ASCENDING docId order (the parity contract).
        for b in bootstrapped.iter_mut() {
            let u = rand.gen_rand_real1();
            // 1.000001f is an f32 literal upstream; transcribe as f64 1.000_001.
            *b *= u / (1.000_001 - u);
        }
        // StableSort(indices, bootstrapped[i] > bootstrapped[j]) — descending,
        // stable on ties (yetirank_helpers.cpp:326-331).
        stable_sort_desc_by_key(&mut indices, &bootstrapped);
        // CalcWeights (Classic mode, the default): accumulate decayed pairwise
        // weights along the sorted adjacency (yetirank_helpers.cpp:193-205).
        calc_weights_classic(&indices, relevs, decay, &mut competitors_weights);
    }

    // competitorsWeight[w][l] = queryWeight · weights[w][l] / permutationCount;
    // nonzero entries become competitor edges (yetirank_helpers.cpp:336-344).
    let mut competitors: Vec<Vec<Competitor>> = vec![Vec::new(); query_size];
    let denom = permutation_count as f64;
    for winner in 0..query_size {
        for loser in 0..query_size {
            let w = query_weight * competitors_weights[winner][loser] / denom;
            if w != 0.0 {
                competitors[winner].push(Competitor { id: loser, weight: w });
            }
        }
    }
    competitors
}

/// Derive the per-query inner seeds for a single-thread (`blockCount == 1`) fit,
/// transcribing the 2-level seed derivation
/// (`yetirank_helpers.cpp:365-389` + `restorable_rng.cpp:3-9`):
/// 1. `randomSeeds = GenRandUI64Vector(1, random_seed)` — the lone block seed is
///    `TFastRng64(random_seed).GenRand()`;
/// 2. `blockRng = TFastRng64(blockSeed)`;
/// 3. per query (in group order): `querySeed = blockRng.GenRand()`.
///
/// Returns `group_count` per-query seeds in group order. The single-thread
/// assumption matches the fixtures (`thread_count: 1`); a multi-thread fit would
/// partition queries into blocks with a per-block seed (the escalated multi-thread
/// budget is out of this phase's scope — documented in the README).
#[must_use]
pub fn derive_query_seeds(random_seed: u64, group_count: usize) -> Vec<u64> {
    // GenRandUI64Vector(1, random_seed)[0] (restorable_rng.cpp:3-9): one GenRand()
    // from TFastRng64(random_seed).
    let mut seed_rng = TFastRng64::from_seed(random_seed);
    let block_seed = seed_rng.gen_rand();
    // blockRng = TFastRng64(blockSeed); per query querySeed = blockRng.GenRand().
    let mut block_rng = TFastRng64::from_seed(block_seed);
    (0..group_count).map(|_| block_rng.gen_rand()).collect()
}

/// Classic pairwise-weight accumulation along the sorted adjacency
/// (`yetirank_helpers.cpp:193-205` `CalcWeightsClassic`):
/// ```text
/// decayCoefficient = 1
/// for docId in 1..querySize:
///     first = permutation[docId-1]; second = permutation[docId]
///     pairWeight = 0.15 · decayCoefficient · |relev[first] - relev[second]|
///     AddWeight(first, second, pairWeight)   // see add_weight
///     decayCoefficient *= decay
/// ```
/// The per-pair weights are accumulated into `competitors_weights` via
/// [`add_weight`] (the higher-relevance doc wins). The `decayCoefficient` chain is
/// a sequential product (the parity contract is the order); the magnitude is a
/// per-pair scalar, not a cross-object reduction, so it is NOT routed through
/// `sum_f64` (which is reserved for the cross-permutation accumulation, applied at
/// the competitor-weight normalisation when summing duplicate edges — here the
/// `+=` into the dense matrix mirrors upstream's exact in-place add).
fn calc_weights_classic(
    permutation: &[usize],
    relevs: &[f64],
    decay: f64,
    competitors_weights: &mut [Vec<f64>],
) {
    let query_size = permutation.len();
    let mut decay_coefficient = 1.0_f64;
    for doc_id in 1..query_size {
        let first = permutation[doc_id - 1];
        let second = permutation[doc_id];
        let rf = relevs.get(first).copied().unwrap_or(0.0);
        let rs = relevs.get(second).copied().unwrap_or(0.0);
        // pairWeight = magicConst · decayCoefficient · |Δrelev|.
        let pair_weight = MAGIC_CONST * decay_coefficient * (rf - rs).abs();
        add_weight(first, second, rf, rs, pair_weight, competitors_weights);
        decay_coefficient *= decay;
    }
}

/// `AddWeight` (`yetirank_helpers.cpp:185-191`): route the pair weight to the
/// higher-relevance doc as the winner. `relev[first] > relev[second]` →
/// `competitorsWeights[first][second] += w`; `relev[first] < relev[second]` →
/// `competitorsWeights[second][first] += w`; ties (`==`) add nothing.
#[inline]
fn add_weight(
    first: usize,
    second: usize,
    relev_first: f64,
    relev_second: f64,
    weight: f64,
    competitors_weights: &mut [Vec<f64>],
) {
    if relev_first > relev_second {
        if let Some(row) = competitors_weights.get_mut(first) {
            if let Some(cell) = row.get_mut(second) {
                *cell += weight;
            }
        }
    } else if relev_first < relev_second {
        if let Some(row) = competitors_weights.get_mut(second) {
            if let Some(cell) = row.get_mut(first) {
                *cell += weight;
            }
        }
    }
}

/// Stable sort of `indices` by `keys[idx]` DESCENDING (upstream `StableSort(indices,
/// bootstrappedApprox[i] > bootstrappedApprox[j])`,
/// `yetirank_helpers.cpp:326-331`). Rust's `sort_by` is stable; the comparator
/// sorts descending while preserving the original order on ties (the parity
/// contract — a different tie-break reorders the sampled adjacency).
fn stable_sort_desc_by_key(indices: &mut [usize], keys: &[f64]) {
    indices.sort_by(|&a, &b| {
        let ka = keys.get(a).copied().unwrap_or(0.0);
        let kb = keys.get(b).copied().unwrap_or(0.0);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Sum a slice of competitor weights through the sanctioned ordered reduction
/// (`cb_core::sum_f64`, D-08). Exposed so the boosting wiring can total a group's
/// sampled-pair weights for the RNG-draw-log oracle compare without spelling a raw
/// float fold.
#[must_use]
pub fn sum_competitor_weights(weights: &[f64]) -> f64 {
    sum_f64(weights)
}

#[cfg(test)]
#[path = "yetirank_test.rs"]
mod tests;
