//! YetiRankPairwise per-stage training oracle (LOSS-04 / Plan 06.3-04 Wave C).
//!
//! YetiRankPairwise shares the EXACT SAME sampled-pair RNG stream as
//! [`cb_compute::Loss::YetiRank`] (identical 2-level seed + Gumbel noise + Classic
//! weights) — the ONLY difference is the LEAF path: YetiRankPairwise solves leaf
//! values via the Cholesky pairwise-leaf system
//! (`is_pairwise_scoring` ⇒ `cb_train::pairwise_leaves`) and forces
//! `boosting_type = Plain`, where YetiRank rides the pointwise estimators. So the
//! RNG-draw ground truth (the parity crux) is the SAME frozen log
//! (`ranking_corpus/yetirank_pairwise/yetirank_rng_groundtruth.jsonl`, a copy of
//! the YetiRank stream), gated LIVE here against the Rust sampler.
//!
//! The end-to-end per-stage compare (over a trained YetiRankPairwise `model.json`
//! through the Cholesky leaf path) remains DEFERRED on TWO newly isolated
//! architectural gaps (06.3-14), NOT the prior toolchain/disk NO-GO (the 06.3-10
//! trainer is now BUILT/GO and was RUN this plan):
//!   1. The YetiRank multi-fold / per-tree RNG seed-plumbing gap isolated in
//!      `yetirank_oracle_test.rs` (the trainer samples over 3 permutation folds
//!      with a per-tree-advanced seed; the Rust sampler uses 1 fold + a fixed
//!      2-level chain) — a NEW seeding subsystem (Rule 4 scope).
//!   2. `*Pairwise` (`is_pairwise_scoring`) losses additionally need the pairwise
//!      SPLIT-scorer (`TPairwiseScoreCalcer`) isolated in 06.3-13 for
//!      PairLogitPairwise — also Rule 4 scope, not yet in cb-train.
//! Per D-6.3-03b: NO `#[ignore]`, NO weakened tolerance; see
//! `deferred-items.md [06.3-13]` (split-scorer) and `[06.3-14]` (seed plumbing).
//! The standalone full-precision RNG-draw oracle stays GREEN.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_compute::{is_pairwise_scoring, is_plain_only, Loss, RankingCompetitor as Competitor};
use cb_oracle::{compare_stage, Stage};
use cb_train::{derive_query_seeds, yetirank_sample_pairs};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn json_raw(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat).unwrap_or_else(|| panic!("key {key} not in {line}")) + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

fn load_competitor_ground_truth(rel: &str) -> Vec<(usize, usize, f64)> {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    text.lines()
        .filter(|l| l.contains("\"event\":\"competitor\""))
        .map(|l| {
            (
                json_raw(l, "winner").parse().unwrap(),
                json_raw(l, "loser").parse().unwrap(),
                json_raw(l, "weight").parse().unwrap(),
            )
        })
        .collect()
}

/// YetiRankPairwise is routed to the PAIRWISE leaf path and forces Plain — the
/// loss-trait predicates must classify it correctly (a mis-route diverges leaf
/// values, RESEARCH Pitfall 2).
#[test]
fn yetirank_pairwise_uses_cholesky_leaf_and_plain() {
    let loss = Loss::YetiRankPairwise { permutations: 10, decay: 0.85 };
    assert!(is_pairwise_scoring(&loss), "YetiRankPairwise must use the Cholesky pairwise leaf path");
    assert!(is_plain_only(&loss), "YetiRankPairwise must force boosting_type = Plain");
    // YetiRank (non-pairwise) must NOT route to the Cholesky leaf.
    let plain = Loss::YetiRank { permutations: 10, decay: 0.85 };
    assert!(!is_pairwise_scoring(&plain), "YetiRank (pointwise) must NOT use the Cholesky leaf");
}

/// LIVE RNG-draw oracle: the sampler reproduces the frozen ground-truth sampled
/// competitor weights (the SAME stream as YetiRank, since the pair sampling is
/// identical — only the leaf differs).
#[test]
fn yetirank_pairwise_rng_draw_log_oracle() {
    let gt_rel = "ranking_corpus/yetirank_pairwise/yetirank_rng_groundtruth.jsonl";
    let raw_approx = [0.5_f64, -0.3, 0.1];
    let relevs = [2.0_f64, 0.0, 1.0];
    let seeds = derive_query_seeds(0, 1);
    let competitors: Vec<Vec<Competitor>> =
        yetirank_sample_pairs(&raw_approx, &relevs, 1.0, 10, 0.85, seeds[0]);

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
    compare_stage(Stage::LeafValues, &expected, &got).unwrap_or_else(|e| {
        panic!("YetiRankPairwise: sampled competitor weights diverged from ground truth: {e:?}")
    });
}

/// Wired-but-pending end-to-end per-stage compare (deferred trainer fixture). NO
/// `#[ignore]`, NO weakened tolerance.
#[test]
fn yetirank_pairwise_end_to_end_per_stage() {
    let model_json = fixture("ranking_corpus/yetirank_pairwise/model.json");
    if model_json.exists() {
        panic!(
            "ranking_corpus/yetirank_pairwise/model.json now exists — wire the full \
             per-stage compare_stage gate (Cholesky leaf path) and remove this guard."
        );
    } else {
        let gt = fixture("ranking_corpus/yetirank_pairwise/yetirank_rng_groundtruth.jsonl");
        assert!(
            gt.exists(),
            "YetiRankPairwise RNG ground truth must be committed while the end-to-end \
             trainer fixture is deferred (D-6.3-03b)"
        );
    }
}
