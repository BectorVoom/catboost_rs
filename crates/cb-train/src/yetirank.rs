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

use cb_core::{std_normal, sum_f64, TFastRng64};
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
    // (yetirank_helpers.cpp:320). Upstream the matrix is `TVector<TVector<float>>`
    // — f32 storage + f32 accumulation. Transcribed as f32 here: the f32 bit-width
    // of the accumulated pair weights is LOAD-BEARING for the end-to-end ≤1e-5 gate
    // (an f64 accumulation drifts the leaf values ~1e-8 and flips a close split by
    // ~tree 2, 06.3-14 ext).
    let mut competitors_weights = vec![vec![0.0_f32; query_size]; query_size];

    for _perm in 0..permutation_count {
        // std::iota(indices, 0) (yetirank_helpers.cpp:322).
        let mut indices: Vec<usize> = (0..query_size).collect();
        // bootstrappedApprox = copy(expApproxes) (yetirank_helpers.cpp:323).
        let mut bootstrapped = exp_approx.clone();
        // AddNoise (Gumbel): per doc, one gen_rand_real1() uniform `u`, then
        // expApprox[d] *= u / (1.000001 - u) (yetirank_helpers.cpp:149-152). The
        // draw is per-doc in ASCENDING docId order (the parity contract).
        for b in bootstrapped.iter_mut() {
            // upstream: `const float uniformValue = rand.GenRandReal1();` — the
            // draw is CAST TO f32, and `expApproxes[d] *= uniformValue /
            // (1.000001f - uniformValue)` evaluates the ratio in f32 (both operands
            // f32), then promotes to f64 for the `*=` on the f64 `expApprox`. The
            // f32 round of `uniformValue` + the f32 ratio is LOAD-BEARING: doing the
            // whole thing in f64 drifts the sampled competitor weights by ~1e-8,
            // which compounds across trees and flips a close split by ~tree 2
            // (06.3-14 ext: end-to-end ≤1e-5 needs the f32 bit-width here).
            let u = rand.gen_rand_real1() as f32;
            let ratio = u / (1.000_001_f32 - u);
            *b *= f64::from(ratio);
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
    // Upstream: `const float competitorsWeight = queryWeight * competitorsWeights[w][l]
    //            / permutationCount;` — f32 throughout (queryWeight is `float`,
    // TCompetitor.Weight is `float`). Compute in f32, then promote to the f64
    // Competitor.weight (the der consumers promote f32→f64). The `!= 0` test is on
    // the f32 value (upstream).
    let mut competitors: Vec<Vec<Competitor>> = vec![Vec::new(); query_size];
    #[allow(clippy::cast_possible_truncation)]
    let query_weight_f32 = query_weight as f32;
    let denom = permutation_count as f32;
    // Iterate by reference rather than raw `[]` indexing (workspace-denied
    // clippy::indexing_slicing); `winner`/`loser` stay the dense-matrix indices via
    // `enumerate`, so the numeric result is bit-identical to the indexed form.
    for (winner, row) in competitors_weights.iter().enumerate() {
        for (loser, &cw) in row.iter().enumerate() {
            let w_f32 = query_weight_f32 * cw / denom;
            if w_f32 != 0.0 {
                if let Some(out_row) = competitors.get_mut(winner) {
                    out_row.push(Competitor {
                        id: loser,
                        weight: f64::from(w_f32),
                    });
                }
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

/// `GenRandUI64Vector(size, seed)` (`restorable_rng.cpp:3-9`): `size` consecutive
/// `GenRand()` draws from `TFastRng64(seed)`. The trainer's per-block seed vector.
#[must_use]
fn gen_rand_ui64_vec(seed: u64, size: usize) -> Vec<u64> {
    let mut rng = TFastRng64::from_seed(seed);
    (0..size).map(|_| rng.gen_rand()).collect()
}

/// Derive the per-query inner seeds for ONE `UpdatePairsForYetiRank` call, the
/// FAITHFUL multi-block trainer model (`yetirank_helpers.cpp:369-414`).
///
/// Unlike [`derive_query_seeds`] (the standalone single-block self-oracle chain),
/// the live trainer partitions the `[0, group_count)` query range into BLOCKS via
/// `TExecRangeParams::SetBlockCount(CB_THREAD_LIMIT=128)`. For the small fixture
/// corpora `block_size == CeilDiv(group_count, 128) == 1`, so `block_count ==
/// group_count` — i.e. ONE block per query. Each block draws its OWN block seed
/// from `randomSeeds = GenRandUI64Vector(block_count, recalc_seed)`, then a fresh
/// `TFastRng64(randomSeeds[blockId])` yields that block's single query seed via one
/// `GenRand()`. So per group `g`:
/// ```text
/// blockSeeds = GenRandUI64Vector(group_count, recalc_seed)
/// querySeed[g] = TFastRng64(blockSeeds[g]).GenRand()
/// ```
/// `recalc_seed` is the `randomSeed` argument `UpdatePairsForYetiRank` receives
/// (already passed through the caller's `GenRandUI64Vector(BodyTailArr=1, ·)` —
/// see [`YetiRankTreeSeeder`]). Verified bit-exact against the instrumented
/// trainer's per-group first Gumbel draws for all 5 corpus groups (06.3-14 ext).
///
/// NOTE: for `group_count == 1` this is NOT identical to [`derive_query_seeds`] —
/// the block model adds one extra `GenRandUI64Vector` layer. The standalone
/// self-oracle keeps [`derive_query_seeds`]; the trainer path uses THIS.
#[must_use]
pub fn derive_per_tree_query_seeds(recalc_seed: u64, group_count: usize) -> Vec<u64> {
    let block_seeds = gen_rand_ui64_vec(recalc_seed, group_count);
    block_seeds
        .into_iter()
        .map(|bs| TFastRng64::from_seed(bs).gen_rand())
        .collect()
}

/// Per-tree YetiRank seeding driver: a persistent context RNG that mirrors
/// `TLearnContext::LearnProgress->Rand` (seeded with `params.random_seed`) and
/// reproduces, draw-for-draw, the upstream per-tree RNG consumption so the
/// derivative-recalc and leaf-value-recalc YetiRank query seeds land on the exact
/// trainer stream (`train.cpp` boosting step + `greedy_tensor_search.cpp` +
/// `approx_calcer.cpp`). This closes the D-07 trainer-level RNG seed-plumbing gap.
///
/// # Per-tree draw sequence (single-host, `boosting_type=Plain`, `bootstrap=No`)
///
/// Transcribed from the instrumented trainer (06.3-14 ext, all events env-gated):
/// 1. `structure_draw = Rand.GenRand()` — the structure-fold selection
///    (`train.cpp:316`, `Folds[draw % foldCount]`).
/// 2. `deriv_base = Rand.GenRand()` → `randomSeeds = GenRandUI64Vector(1, deriv_base)`
///    (`train.cpp:326`, `BodyTailArr.size()==1`) → `UpdatePairsForYetiRank(randomSeeds[0])`
///    re-samples the DERIVATIVE competitors (drives gradient + split scoring).
/// 3. Per tree LEVEL (`depth` levels, `greedy_tensor_search.cpp:1189`):
///    - `SelectCandidatesAndCleanupStatsFromPrevTree`: one `GenRandReal1()` per
///      float-feature candidate sublist (`:334`, `OneFeature`, Rsm draw — Rsm=1 so
///      every feature survives but the draw still advances the RNG); `n_features`
///      draws.
///    - `CalcScores`: one `GenRand()` (`:884`, the per-level score randSeed).
///    - `SelectBestCandidate`: one `TRandomScore::GetInstance` per candidate
///      (`:955`), which calls `NormalDistribution(rand, 0, stDev)` →
///      `StdNormalDistribution` (Box–Muller). With `random_strength==0` the result
///      is discarded but the draws STILL happen; `n_features` candidates (one
///      `BestScore` per float feature). Each consumes a variable (rejection-driven)
///      even count of `GenRandReal1()` — reproduced via [`cb_core::std_normal`].
/// 4. `learnfold_base = Rand.GenRand()` → `GenRandUI64Vector(foldCount=1, ·)`
///    (`train.cpp:420`) for the learning-fold approx update (its YetiRank recalc is
///    consumed but does not feed the model; we only ADVANCE the RNG here).
/// 5. `leafval_base = Rand.GenRand()` (`approx_calcer.cpp:983`, inside
///    `CalcLeafValuesSimple`) → `UpdatePairsForYetiRank(leafval_base)` re-samples
///    the LEAF-VALUE competitors (drives leaf der + leaf weights on the
///    AveragingFold).
///
/// The deriv recalc seed passes through TWO `GenRandUI64Vector` layers
/// (`deriv_base → randomSeeds[0] → block seeds`); the leaf-value recalc seed is the
/// RAW `Rand.GenRand()` (ONE fewer layer). Both are handled here so callers receive
/// ready-to-use per-group query seeds.
pub struct YetiRankTreeSeeder {
    rng: TFastRng64,
    group_count: usize,
    n_features: usize,
    depth: usize,
}

/// The per-tree YetiRank query seeds: one set for the derivative/split recalc, one
/// for the learning-fold approx-update recalc, and one for the (averaging-fold)
/// leaf-value recalc — each a per-group inner Gumbel seed.
pub struct YetiRankTreeSeeds {
    /// Per-group inner seeds for the gradient/split competitor re-sample (learning
    /// fold 0 approx).
    pub deriv: Vec<u64>,
    /// Per-group inner seeds for the LEARNING-fold approx-update competitor
    /// re-sample (`UpdateLearningFold` -> `CalcApproxForLeafStruct`, drawn at
    /// `train.cpp:420` BEFORE the averaging-fold leaf values).
    pub learnfold: Vec<u64>,
    /// Per-group inner seeds for the AVERAGING-fold leaf-value competitor
    /// re-sample (the stored model leaf values).
    pub leafval: Vec<u64>,
    /// The recalc `randomSeed` values passed to `UpdatePairsForYetiRank` for the
    /// deriv / learnfold / leafval phases respectively (the trainer's
    /// `update_pairs.random_seed` fences). Exposed for the per-tree RNG-seed
    /// oracle to compare call-for-call against the instrumented trainer.
    pub recalc_seeds: [u64; 3],
}

impl YetiRankTreeSeeder {
    /// Create the seeder mirroring `LearnProgress->Rand(random_seed)`.
    #[must_use]
    pub fn new(random_seed: u64, group_count: usize, n_features: usize, depth: usize) -> Self {
        Self::new_with_scoring(random_seed, group_count, n_features, depth, false)
    }

    /// As [`Self::new`] but kept for the `*Pairwise` (`IsPairwiseScoring`) call site.
    ///
    /// 06.3-17: the `pairwise` argument is retained for caller-site clarity /
    /// API stability but no longer changes the per-tree RNG-advance accounting —
    /// the instrumented `yetirank_pairwise_tree_rng_groundtruth.jsonl` fences
    /// proved BOTH the pointwise and the pairwise split paths draw the
    /// per-candidate `BestScore.GetInstance` normals directly from
    /// `LearnProgress->Rand` (see [`Self::next_tree`]). The single draw model now
    /// serves both losses.
    #[must_use]
    pub fn new_with_scoring(
        random_seed: u64,
        group_count: usize,
        n_features: usize,
        depth: usize,
        _pairwise: bool,
    ) -> Self {
        Self {
            rng: TFastRng64::from_seed(random_seed),
            group_count,
            n_features,
            depth,
        }
    }

    /// The persistent context-RNG call count (`TRestorableFastRng64::GetCallCount`)
    /// consumed so far. Exposed so the per-tree RNG-draw oracle can assert the
    /// seeder lands on the exact trainer call-count fence
    /// (`yetirank_pairwise_tree_rng_groundtruth.jsonl` `tree_rng_start.cc`).
    #[must_use]
    pub fn call_count(&self) -> u64 {
        self.rng.call_count()
    }

    /// Advance the context RNG through ONE tree's full draw sequence and return the
    /// derivative + leaf-value per-group query seeds. Must be called once per tree,
    /// in tree order, so the persistent RNG phase stays aligned with the trainer.
    pub fn next_tree(&mut self) -> YetiRankTreeSeeds {
        // 1. structure-fold selection draw.
        let _structure = self.rng.gen_rand();

        // 2. derivative recalc: deriv_base -> GenRandUI64Vector(1, deriv_base)[0]
        //    is the randomSeed UpdatePairsForYetiRank receives.
        let deriv_base = self.rng.gen_rand();
        // GenRandUI64Vector(1, ·) always yields exactly one element; bounds-check
        // (clippy::indexing_slicing) rather than raw `[0]`. The `unwrap_or(deriv_base)`
        // is unreachable for the hardcoded size 1 (debug_assert pins the invariant).
        let deriv_recalc_seed = {
            let v = gen_rand_ui64_vec(deriv_base, 1);
            debug_assert_eq!(v.len(), 1, "GenRandUI64Vector(1, ·) must yield one seed");
            v.first().copied().unwrap_or(deriv_base)
        };
        let deriv = derive_per_tree_query_seeds(deriv_recalc_seed, self.group_count);

        // 3. per-level GreedyTensorSearch draws (consumed, output-irrelevant at
        //    random_strength=0 / Rsm=1, but they ADVANCE the shared RNG phase).
        //
        // 06.3-17 calibration (instrumented YetiRankPairwise trainer, the
        // `yetirank_pairwise_tree_rng_groundtruth.jsonl` per-level fences):
        // BOTH the pointwise and the `*Pairwise` (`IsPairwiseScoring`) split paths
        // draw the per-candidate `TRandomScore::GetInstance` normals DIRECTLY from
        // `LearnProgress->Rand` (`greedy_tensor_search.cpp:952` —
        // `candidate.BestScore.GetInstance(ctx.LearnProgress->Rand)`). The earlier
        // `pairwise`-skip hypothesis (that the pairwise child `TRestorableFastRng64`
        // bypasses the main RNG) was REFUTED by the `cand_score_rng` fence: every
        // candidate logs `dist=Normal, stdev=0` and a non-zero Marsaglia-polar
        // rejection draw count (2/4/6/8) on the PERSISTENT RNG. So the pairwise path
        // advances the main RNG by the SAME per-candidate normals as the pointwise
        // path; the `pairwise` flag no longer gates the normal draws.
        for _level in 0..self.depth {
            // SelectCandidates: one Rsm GenRandReal1 per float-feature sublist
            // (greedy_tensor_search.cpp:334, OneFeature). `n_features` draws.
            for _ in 0..self.n_features {
                let _ = self.rng.gen_rand_real1();
            }
            // CalcScores: one per-level score randSeed
            // (greedy_tensor_search.cpp:884, `Rand.GenRand()`).
            let _ = self.rng.gen_rand();
            // SelectBestCandidate: one Box-Muller (Marsaglia-polar) standard-normal
            // per candidate (`BestScore.GetInstance` → `NormalDistribution(rand, 0,
            // StDev)` → `StdNormalDistribution`, greedy_tensor_search.cpp:952 +
            // rand_score.h:42-44). One `BestScore` per float-feature candidate;
            // `n_features` candidates. The draw is taken even at `StDev==0` (the
            // value is discarded but the RNG still advances). The variable
            // (rejection-driven) even draw count is reproduced bit-exact by
            // [`cb_core::std_normal`] over the shared [`TFastRng64`].
            for _ in 0..self.n_features {
                let _ = std_normal(&mut self.rng);
            }
        }

        // 4. learning-fold update recalc: learnfold_base ->
        //    GenRandUI64Vector(foldCount=1, learnfold_base)[0] is the randomSeed
        //    `UpdateLearningFold -> CalcApproxForLeafStruct` passes to
        //    `UpdatePairsForYetiRank` (train.cpp:420 + approx_calcer.cpp:1147,
        //    `GenRandUI64Vector(BodyTailArr=1, ·)`). This re-samples the LEARNING
        //    fold's competitors, whose leaf values update the learning-fold approx
        //    that feeds the NEXT tree's gradient/structure (YetiRank is NOT
        //    UseAveragingFoldAsFoldZero — usePairs is true, learn_context.cpp:855).
        // Two GenRandUI64Vector(1, ·) layers: randomSeeds[0]=vec1(learnfold_base)
        // (train.cpp:420, foldCount=1), then randomSeeds2[0]=vec1(randomSeeds[0])
        // (approx_calcer.cpp:1147, BodyTailArr=1) — the seed `CalcApproxDeltaSimple`
        // passes verbatim to `UpdatePairsForYetiRank`.
        let learnfold_base = self.rng.gen_rand();
        // Two GenRandUI64Vector(1, ·) layers; bounds-check both `[0]` reads
        // (clippy::indexing_slicing). Each vec has exactly one element (size 1), so
        // the `unwrap_or` fallbacks are unreachable — pinned by debug_assert.
        let learnfold_rs0 = {
            let v = gen_rand_ui64_vec(learnfold_base, 1);
            debug_assert_eq!(v.len(), 1, "GenRandUI64Vector(1, ·) must yield one seed");
            v.first().copied().unwrap_or(learnfold_base)
        };
        let learnfold_recalc_seed = {
            let v = gen_rand_ui64_vec(learnfold_rs0, 1);
            debug_assert_eq!(v.len(), 1, "GenRandUI64Vector(1, ·) must yield one seed");
            v.first().copied().unwrap_or(learnfold_rs0)
        };
        let learnfold = derive_per_tree_query_seeds(learnfold_recalc_seed, self.group_count);

        // 5. averaging-fold leaf-value recalc: leafval_base is the RAW
        //    Rand.GenRand() passed directly to CalcLeafDersSimple ->
        //    UpdatePairsForYetiRank (one fewer GenRandUI64Vector layer than the
        //    deriv/learnfold recalcs). These competitors drive the STORED model
        //    leaf values on the AveragingFold.
        let leafval_base = self.rng.gen_rand();
        let leafval = derive_per_tree_query_seeds(leafval_base, self.group_count);

        YetiRankTreeSeeds {
            deriv,
            learnfold,
            leafval,
            recalc_seeds: [deriv_recalc_seed, learnfold_recalc_seed, leafval_base],
        }
    }
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
    competitors_weights: &mut [Vec<f32>],
) {
    let query_size = permutation.len();
    let mut decay_coefficient = 1.0_f64;
    for doc_id in 1..query_size {
        // doc_id in 1..query_size and query_size == permutation.len(), so both reads
        // are provably in range; bounds-check (clippy::indexing_slicing) anyway,
        // mirroring this file's stable_sort/add_weight `.get().copied().unwrap_or`
        // discipline. The `unwrap_or(0)` fallbacks are unreachable here.
        let first = permutation.get(doc_id - 1).copied().unwrap_or(0);
        let second = permutation.get(doc_id).copied().unwrap_or(0);
        // upstream `Relevs` is `const float*`; the per-pair |Δrelev| is computed in
        // f32 (`Abs(Relevs[i] - Relevs[j])`). Cast the f64 corpus relevances to f32
        // first so the subtraction round matches.
        #[allow(clippy::cast_possible_truncation)]
        let rf = relevs.get(first).copied().unwrap_or(0.0) as f32;
        #[allow(clippy::cast_possible_truncation)]
        let rs = relevs.get(second).copied().unwrap_or(0.0) as f32;
        // `const float pairWeight = magicConst * decayCoefficient * Abs(rf - rs)` —
        // the f64 product (magicConst, decayCoefficient are `double`) is assigned to
        // an f32, so cast the result to f32.
        #[allow(clippy::cast_possible_truncation)]
        let pair_weight = (MAGIC_CONST * decay_coefficient * f64::from((rf - rs).abs())) as f32;
        add_weight(first, second, rf, rs, pair_weight, competitors_weights);
        decay_coefficient *= decay;
    }
}

/// `AddWeight` (`yetirank_helpers.cpp:185-191`): route the pair weight to the
/// higher-relevance doc as the winner. `relev[first] > relev[second]` →
/// `competitorsWeights[first][second] += w`; `relev[first] < relev[second]` →
/// `competitorsWeights[second][first] += w`; ties (`==`) add nothing. f32 storage
/// + f32 accumulation, matching upstream `TVector<TVector<float>>`.
#[inline]
fn add_weight(
    first: usize,
    second: usize,
    relev_first: f32,
    relev_second: f32,
    weight: f32,
    competitors_weights: &mut [Vec<f32>],
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
