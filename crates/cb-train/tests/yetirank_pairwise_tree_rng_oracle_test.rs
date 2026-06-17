//! YetiRankPairwise PER-TREE RNG draw-count oracle (LOSS-04 / Plan 06.3-17).
//!
//! The end-to-end per-stage oracle (`yetirank_pairwise_oracle_test.rs`) diverged
//! at TREE 2's structure because the [`cb_train::YetiRankTreeSeeder`] advanced the
//! persistent context RNG by the WRONG number of draws per tree for trees 2+. The
//! root cause: an earlier hypothesis that the `*Pairwise` (`IsPairwiseScoring`)
//! split path draws its per-candidate `BestScore.GetInstance` normals from a CHILD
//! `TRestorableFastRng64` that does NOT advance `LearnProgress->Rand`.
//!
//! That hypothesis was REFUTED by the instrumented catboost 1.2.10 trainer. The
//! committed ground truth
//! `ranking_corpus/yetirank_pairwise/yetirank_pairwise_tree_rng_groundtruth.jsonl`
//! (captured via CB_INSTRUMENT_LOG per-tree `tree_rng_start/pre_gts/post_gts/
//! pre_leaf/end` call-count fences + per-candidate `cand_score_rng` fences) shows
//! every candidate logs `dist=Normal, stdev=0` with a non-zero Marsaglia-polar
//! draw count (2/4/6/8) ON THE PERSISTENT RNG. So the pairwise path advances the
//! main RNG by the SAME per-candidate normals as the pointwise path.
//!
//! This oracle pins the seeder's persistent-RNG call count after each of the 5
//! trees to the trainer's `tree_rng_start.cc` fence (0, 34, 76, 108, 146) and the
//! final `tree_rng_end.cc` (186), bit-exact. NO tolerance — this is an integer
//! draw count (escalate-don't-weaken, D-6.3-03b).

use std::path::PathBuf;

use cb_train::YetiRankTreeSeeder;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Parse the per-tree `tree_rng_start.cc` (and final `tree_rng_end.cc`) call-count
/// fences from the frozen instrumented draw-log. Returns the cumulative persistent
/// RNG call count at the START of each tree, followed by the count at the END of
/// the last tree.
fn load_tree_rng_fences(rel: &str) -> (Vec<u64>, u64) {
    let text =
        std::fs::read_to_string(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    let mut starts: Vec<(u64, u64)> = Vec::new(); // (iter, cc)
    let mut last_end: u64 = 0;
    for line in text.lines() {
        if line.contains("\"event\":\"tree_rng_start\"") {
            starts.push((json_u64(line, "iter"), json_u64(line, "cc")));
        } else if line.contains("\"event\":\"tree_rng_end\"") {
            last_end = json_u64(line, "cc");
        }
    }
    starts.sort_by_key(|&(iter, _)| iter);
    (starts.into_iter().map(|(_, cc)| cc).collect(), last_end)
}

fn json_u64(line: &str, key: &str) -> u64 {
    let pat = format!("\"{key}\":");
    let start = line
        .find(&pat)
        .unwrap_or_else(|| panic!("key {key} not in {line}"))
        + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().unwrap_or_else(|e| panic!("parse {key} from {line}: {e:?}"))
}

/// The `update_pairs.random_seed` recalc seeds, grouped 3-per-tree in trainer
/// order (deriv, learnfold, leafval). These are the exact `randomSeed` values the
/// trainer passes to `UpdatePairsForYetiRank` for each phase.
fn load_recalc_seeds(rel: &str) -> Vec<[u64; 3]> {
    let text =
        std::fs::read_to_string(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    let flat: Vec<u64> = text
        .lines()
        .filter(|l| l.contains("\"event\":\"update_pairs\""))
        .map(|l| json_u64(l, "random_seed"))
        .collect();
    assert!(flat.len() % 3 == 0, "update_pairs seeds must group 3-per-tree, got {}", flat.len());
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// The YetiRankPairwise fixture config (mirrors
/// `yetirank_pairwise_oracle_test.rs` + the committed `config.json`).
const RANDOM_SEED: u64 = 20_260_617;
const GROUP_COUNT: usize = 5; // GROUP_SIZES = [3,2,4,1,2]
// WR-02 (06.3-17): the candidate-feature count is ALL listed float features, NOT
// just those with selected borders in the final model. The corpus has 4 float
// features; feature 2 ends UNUSED (0 selected borders) but was a TRAINING
// candidate that consumed an Rsm draw + a normal per level. Under-counting it as 3
// short-changed the per-tree GTS draw count and desynced trees 1+.
const N_FEATURES: usize = 4;
const DEPTH: usize = 2;
const ITERATIONS: usize = 5;

#[test]
fn yetirank_pairwise_per_tree_rng_call_count_oracle() {
    let gt_rel = "ranking_corpus/yetirank_pairwise/yetirank_pairwise_tree_rng_groundtruth.jsonl";
    let (starts, last_end) = load_tree_rng_fences(gt_rel);
    assert_eq!(
        starts.len(),
        ITERATIONS,
        "ground truth must carry one tree_rng_start fence per iteration"
    );
    assert_eq!(starts[0], 0, "tree 0 must start at call count 0");

    // The pairwise flag is retained for API parity but no longer changes the
    // accounting (06.3-17): both losses draw the per-candidate normals.
    let mut seeder =
        YetiRankTreeSeeder::new_with_scoring(RANDOM_SEED, GROUP_COUNT, N_FEATURES, DEPTH, true);

    assert_eq!(
        seeder.call_count(),
        starts[0],
        "seeder must start at the trainer's tree-0 call-count fence"
    );

    let recalc = load_recalc_seeds(gt_rel);
    assert_eq!(recalc.len(), ITERATIONS, "one recalc-seed triple per iteration");

    for (tree, &expected_start) in starts.iter().enumerate() {
        assert_eq!(
            seeder.call_count(),
            expected_start,
            "before tree {tree}: seeder call count {} != trainer fence {expected_start} \
             (per-tree pairwise draw stream desynced)",
            seeder.call_count()
        );
        let seeds = seeder.next_tree();
        // The seeder's recalc seeds are stored [deriv, learnfold, leafval] (the DRAW
        // order: learnfold_base is drawn at train.cpp:449 BEFORE CalcLeafValues
        // draws leafval_base). The trainer LOGS UpdatePairsForYetiRank in CALL order
        // [deriv, leafval, learnfold] — CalcLeafValues (leafval) runs before
        // UpdateLearningFold (learnfold). So compare against the reordered triple.
        let trainer_call_order =
            [seeds.recalc_seeds[0], seeds.recalc_seeds[2], seeds.recalc_seeds[1]];
        assert_eq!(
            trainer_call_order, recalc[tree],
            "tree {tree}: recalc seeds (call order deriv/leafval/learnfold) {:?} != \
             trainer update_pairs seeds {:?} (seed derivation desynced)",
            trainer_call_order, recalc[tree]
        );
    }

    assert_eq!(
        seeder.call_count(),
        last_end,
        "after all {ITERATIONS} trees: seeder call count {} != trainer tree_rng_end fence \
         {last_end}",
        seeder.call_count()
    );
}
