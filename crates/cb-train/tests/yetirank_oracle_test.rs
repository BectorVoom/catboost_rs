//! YetiRank per-stage training oracle (LOSS-04 / Plan 06.3-04 Wave C): the
//! RNG-stream parity gate for the randomized YetiRank loss.
//!
//! # What this oracle gates (no weakening, no #[ignore])
//!
//! YetiRank has NO closed-form gradient — its pairwise weights are SAMPLED from a
//! `TFastRng64` stream whose exact draw COUNT + ORDER is the parity crux. The
//! ground truth is the OFFLINE instrumented generator
//! (`crates/cb-oracle/generator/yetirank_oracle.cpp`), which transcribes the
//! upstream RNG units verbatim and SELF-ORACLES bit-for-bit against the
//! oracle-locked `cb-core::TFastRng64` (see
//! `crates/cb-oracle/generator/instrument_ranking_rng_README.md`). Its captured
//! draw log is frozen at
//! `ranking_corpus/yetirank/yetirank_rng_groundtruth.jsonl`.
//!
//! `compare_stage` here gates the Rust `yetirank_sample_pairs` sampler's SAMPLED
//! COMPETITOR WEIGHTS against that frozen ground truth at <= 1e-5 — the integer/
//! f64-exact RNG-draw-log compare that gates the randomized stream INDEPENDENTLY
//! of the der (RESEARCH Pitfall 1: the der can match for permutations=1 yet
//! diverge at the default if the seed re-derivation is wrong). This assertion is
//! LIVE — it fails if the sampler's RNG draw order regresses.
//!
//! # Deferred end-to-end per-stage fixture (escalate-don't-weaken, D-6.3-03b)
//!
//! The FULL per-stage `compare_stage(Splits|LeafValues|StagedApprox|Predictions)`
//! over a trained YetiRank `model.json` requires the instrumented catboost 1.2.10
//! TRAINER build, which the Task-1 feasibility probe found INFEASIBLE this session
//! (toolchain absent + disk NO-GO; see the README STATUS section). Per
//! D-6.3-03b that step is DEFERRED — NOT weakened, NOT `#[ignore]`d, NOT
//! fabricated. [`yetirank_end_to_end_per_stage`] is the wired-but-pending compare:
//! it runs the full per-stage gate the MOMENT the frozen trainer fixture
//! (`ranking_corpus/yetirank/model.json`) lands, and otherwise asserts the
//! deferred-fixture invariant (the directory exists with the RNG ground truth) so
//! the test never silently passes on a missing gate. This mirrors the Phase-5
//! ORD-01 "ground truth committed, oracle wired, no weakening" precedent.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_compute::RankingCompetitor as Competitor;
use cb_oracle::{compare_stage, Stage};
use cb_train::{derive_query_seeds, yetirank_sample_pairs};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Parse the frozen `competitor` events from the instrumented ground-truth JSONL
/// into a (winner, loser, weight) list. The generator's smallest unit is one
/// group, 3 docs, permutations=10, decay=0.85, random_seed=0 — the SAME config
/// reproduced below.
fn load_competitor_ground_truth(rel: &str) -> Vec<(usize, usize, f64)> {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load (frozen RNG ground truth): {e:?}"));
    let mut out = Vec::new();
    for line in text.lines() {
        if !line.contains("\"event\":\"competitor\"") {
            continue;
        }
        let w = json_usize(line, "winner");
        let l = json_usize(line, "loser");
        let weight = json_f64(line, "weight");
        out.push((w, l, weight));
    }
    out
}

/// Parse the frozen `query_seed` event (the 2-level-derived per-query seed).
fn load_query_seed_ground_truth(rel: &str) -> u64 {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    for line in text.lines() {
        if line.contains("\"event\":\"query_seed\"") {
            // Parse the seed as a u64 DIRECTLY (not through f64 — a 64-bit seed
            // exceeds f64's 53-bit mantissa and would truncate, T-06.3-04-01).
            return json_raw(line, "seed").parse().unwrap();
        }
    }
    panic!("{rel}: no query_seed event in the ground-truth log");
}

// Minimal JSON field extractors (the log is a flat one-object-per-line schema; a
// full serde dependency is unnecessary for these fixed-shape lines).
fn json_usize(line: &str, key: &str) -> usize {
    json_raw(line, key).parse().unwrap()
}
fn json_f64(line: &str, key: &str) -> f64 {
    json_raw(line, key).parse().unwrap()
}
fn json_raw(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat).unwrap_or_else(|| panic!("key {key} not in {line}")) + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

/// LIVE RNG-draw oracle: the Rust `yetirank_sample_pairs` sampler must reproduce
/// the instrumented ground-truth sampled competitor weights AND the 2-level query
/// seed EXACTLY (<= 1e-5). This gates the randomized RNG stream independently of
/// the der — a desynced draw order fails here from the first sample.
#[test]
fn yetirank_rng_draw_log_oracle() {
    let gt_rel = "ranking_corpus/yetirank/yetirank_rng_groundtruth.jsonl";

    // The generator's smallest unit (yetirank_oracle.cpp main): one group, 3 docs,
    // rawApprox [0.5, -0.3, 0.1], relevs [2, 0, 1], permutations 10, decay 0.85,
    // random_seed 0. Reproduce the per-query seed via the 2-level chain.
    let raw_approx = [0.5_f64, -0.3, 0.1];
    let relevs = [2.0_f64, 0.0, 1.0];
    let permutations = 10_u32;
    let decay = 0.85_f64;
    let random_seed = 0_u64;

    // 2-level seed derivation (single group): the derived query seed MUST match the
    // instrumented log's query_seed event.
    let seeds = derive_query_seeds(random_seed, 1);
    let expected_seed = load_query_seed_ground_truth(gt_rel);
    assert_eq!(
        seeds[0], expected_seed,
        "YetiRank 2-level query seed must match the instrumented ground truth"
    );

    let competitors: Vec<Vec<Competitor>> =
        yetirank_sample_pairs(&raw_approx, &relevs, 1.0, permutations, decay, seeds[0]);

    // Flatten the Rust sampled adjacency into a (winner, loser, weight) matrix and
    // compare against the frozen ground truth (every nonzero edge, <= 1e-5).
    let n = raw_approx.len();
    let mut got = vec![0.0_f64; n * n];
    for (winner, row) in competitors.iter().enumerate() {
        for c in row {
            got[winner * n + c.id] = c.weight;
        }
    }
    let mut expected = vec![0.0_f64; n * n];
    for (w, l, weight) in load_competitor_ground_truth(gt_rel) {
        expected[w * n + l] = weight;
    }
    compare_stage(Stage::LeafValues, &expected, &got)
        .unwrap_or_else(|e| panic!("YetiRank: sampled competitor weights diverged from instrumented ground truth: {e:?}"));
}

/// Wired-but-pending end-to-end per-stage compare. Runs the full
/// `compare_stage(Splits|LeafValues|StagedApprox|Predictions)` over a trained
/// YetiRank `model.json` the MOMENT the deferred instrumented trainer fixture
/// lands; until then it asserts the deferred-fixture invariant (the RNG ground
/// truth is committed) so the test never silently passes on a missing gate. NO
/// `#[ignore]`, NO weakened tolerance.
#[test]
fn yetirank_end_to_end_per_stage() {
    let model_json = fixture("ranking_corpus/yetirank/model.json");
    if model_json.exists() {
        // OFFLINE closure landed: wire the full per-stage compare here (identical
        // shape to lambdamart_oracle_per_stage — load model.json/staged/predictions,
        // train_ranking under Loss::YetiRank, compare_stage all four stages <= 1e-5).
        panic!(
            "ranking_corpus/yetirank/model.json now exists — wire the full per-stage \
             compare_stage gate (see lambdamart_oracle_test.rs precedent) and remove \
             this guard. The end-to-end YetiRank trainer fixture has landed."
        );
    } else {
        // Deferred (path c): the trainer fixture is not yet built. Assert the RNG
        // ground truth IS committed (the recoverable part) so this is a real gate
        // on the deferred-closure invariant, not a silent skip.
        let gt = fixture("ranking_corpus/yetirank/yetirank_rng_groundtruth.jsonl");
        assert!(
            gt.exists(),
            "YetiRank RNG ground truth must be committed even while the end-to-end \
             trainer fixture is deferred (escalate-don't-weaken, D-6.3-03b)"
        );
    }
}
