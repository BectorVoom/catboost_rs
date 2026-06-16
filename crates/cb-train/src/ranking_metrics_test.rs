//! Unit tests for the per-group ranking-metric formulas ([`crate::ranking_metrics`]).
//!
//! Locks each per-group metric on HAND-COMPUTED tiny fixtures (independent of any
//! oracle), with the load-bearing edge cases: the `compare_docs` tie-break (equal
//! predictions → target ascending → stable index), `top=-1` (full group), and
//! `IDCG==0 → 1` (NDCG). Dedicated test file (CLAUDE.md source/test separation —
//! no inline `#[cfg(test)]`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use crate::ranking_metrics::{
    clamp_top, compare_docs, dcg_group, err_group, map_at_group, mrr_group, ndcg_group,
    pfound_group, precision_at_group, query_auc_group, recall_at_group, AucType, DcgDenominator,
    DcgMetricType,
};

const TOL: f64 = 1e-12;

/// `compare_docs`: predicted desc, ties broken by target ascending.
#[test]
fn compare_docs_tie_break() {
    // Distinct predictions: higher approx sorts first.
    assert!(compare_docs(2.0, 0.0, 1.0, 5.0));
    assert!(!compare_docs(1.0, 5.0, 2.0, 0.0));
    // Equal predictions: lower target sorts first.
    assert!(compare_docs(1.0, 0.0, 1.0, 3.0));
    assert!(!compare_docs(1.0, 3.0, 1.0, 0.0));
    // Equal predictions AND equal target: neither precedes the other.
    assert!(!compare_docs(1.0, 2.0, 1.0, 2.0));
}

/// `clamp_top`: top=-1 → full group; top > size → size; else top.
#[test]
fn clamp_top_semantics() {
    assert_eq!(clamp_top(-1, 4), 4);
    assert_eq!(clamp_top(10, 4), 4);
    assert_eq!(clamp_top(2, 4), 2);
    assert_eq!(clamp_top(0, 4), 0);
}

/// DCG (Base / LogPosition) hand-computed over 3 docs with DISTINCT predictions.
/// approx [3,1,2] target [2,0,1] → sorted order by approx desc = [doc0, doc2, doc1]
/// sorted targets = [2, 1, 0]; decay = [1, 1/log2(3), 1/log2(4)=0.5].
/// DCG = 2*1 + 1/log2(3) + 0 = 2 + 0.6309297535714574 = 2.6309297535714574.
#[test]
fn dcg_base_logposition_hand_computed() {
    let approx = [3.0, 1.0, 2.0];
    let target = [2.0, 0.0, 1.0];
    let got = dcg_group(&approx, &target, -1, DcgMetricType::Base, DcgDenominator::LogPosition);
    let expected = 2.0 + 1.0 / 3f64.log2();
    assert!((got - expected).abs() < TOL, "dcg = {got}, want {expected}");
}

/// NDCG = DCG / IDCG. Ideal target order = [2,1,0] (already ideal here), so
/// IDCG == DCG and NDCG == 1.
#[test]
fn ndcg_perfect_order_is_one() {
    let approx = [3.0, 1.0, 2.0];
    let target = [2.0, 0.0, 1.0];
    let got = ndcg_group(&approx, &target, -1, DcgMetricType::Base, DcgDenominator::LogPosition);
    assert!((got - 1.0).abs() < TOL, "ndcg = {got}");
}

/// NDCG with all-zero targets → IDCG == 0 → returns 1 (upstream `dcg.cpp:147`).
#[test]
fn ndcg_idcg_zero_returns_one() {
    let approx = [3.0, 1.0, 2.0];
    let target = [0.0, 0.0, 0.0];
    let got = ndcg_group(&approx, &target, -1, DcgMetricType::Base, DcgDenominator::LogPosition);
    assert!((got - 1.0).abs() < TOL, "ndcg idcg==0 = {got}");
}

/// The `compare_docs` tie-break is load-bearing: two docs with EQUAL predictions
/// must order the lower-target one first. approx [1,1] target [3,0] → sorted =
/// [doc1(t=0), doc0(t=3)]; DCG = 0*1 + 3*(1/log2(3)) = 3/log2(3).
#[test]
fn dcg_tie_break_orders_lower_target_first() {
    let approx = [1.0, 1.0];
    let target = [3.0, 0.0];
    let got = dcg_group(&approx, &target, -1, DcgMetricType::Base, DcgDenominator::LogPosition);
    let expected = 3.0 / 3f64.log2();
    assert!((got - expected).abs() < TOL, "dcg tie = {got}, want {expected}");
}

/// PFound cascade (decay 0.85): approx [3,2,1] target [1,0,1] sorted = [d0,d1,d2]
/// pLook=1; +1*1=1, pLook=(1-1)*.85=0; +0*0; pLook stays 0; +1*0=0 → PFound=1.
#[test]
fn pfound_cascade_hand_computed() {
    let approx = [3.0, 2.0, 1.0];
    let target = [1.0, 0.0, 1.0];
    let got = pfound_group(&approx, &target, -1, 0.85);
    assert!((got - 1.0).abs() < TOL, "pfound = {got}");
}

/// PFound with a non-saturating first doc: target [0.5, 1.0] approx [2,1] sorted
/// [d0(0.5), d1(1.0)]. pLook=1: +0.5; pLook=(1-0.5)*.85=0.425; +1.0*0.425=0.425.
/// PFound = 0.925.
#[test]
fn pfound_partial_relevance() {
    let approx = [2.0, 1.0];
    let target = [0.5, 1.0];
    let got = pfound_group(&approx, &target, -1, 0.85);
    assert!((got - 0.925).abs() < TOL, "pfound = {got}");
}

/// ERR cascade: approx [3,2,1] target [1,0,1] sorted [d0,d1,d2].
/// pLook=1: +1*1/1=1; pLook*=(1-1)=0; +0; +1*0/3=0 → ERR=1.
#[test]
fn err_cascade_hand_computed() {
    let approx = [3.0, 2.0, 1.0];
    let target = [1.0, 0.0, 1.0];
    let got = err_group(&approx, &target, -1);
    assert!((got - 1.0).abs() < TOL, "err = {got}");
}

/// MRR: first relevant (target > 0.5) is at rank 2 in approx-sorted order.
/// approx [3,2,1] target [0,1,0]: max relevant approx = 2 (doc1). pos counts
/// docs beating 2 → doc0 (approx 3>2) ⇒ pos=2 ⇒ MRR = 1/2.
#[test]
fn mrr_first_relevant_rank_two() {
    let approx = [3.0, 2.0, 1.0];
    let target = [0.0, 1.0, 0.0];
    let got = mrr_group(&approx, &target, -1, 0.5);
    assert!((got - 0.5).abs() < TOL, "mrr = {got}");
}

/// MRR with no relevant doc → 0.
#[test]
fn mrr_no_relevant_is_zero() {
    let approx = [3.0, 2.0, 1.0];
    let target = [0.0, 0.0, 0.0];
    let got = mrr_group(&approx, &target, -1, 0.5);
    assert!(got.abs() < TOL, "mrr = {got}");
}

/// PrecisionAt@2 (border 0.5): approx [3,2,1,0] target [1,0,1,1] sorted
/// [d0,d1,d2,d3]; top 2 = [d0(rel),d1(not)] → 1 relevant / 2 = 0.5.
#[test]
fn precision_at_k_hand_computed() {
    let approx = [3.0, 2.0, 1.0, 0.0];
    let target = [1.0, 0.0, 1.0, 1.0];
    let got = precision_at_group(&approx, &target, 2, 0.5);
    assert!((got - 0.5).abs() < TOL, "prec@2 = {got}");
}

/// RecallAt@2 (border 0.5): same set; top 2 has 1 relevant; total relevant = 3
/// → 1/3.
#[test]
fn recall_at_k_hand_computed() {
    let approx = [3.0, 2.0, 1.0, 0.0];
    let target = [1.0, 0.0, 1.0, 1.0];
    let got = recall_at_group(&approx, &target, 2, 0.5);
    assert!((got - 1.0 / 3.0).abs() < TOL, "rec@2 = {got}");
}

/// RecallAt with no relevant doc → 1 (upstream `precision_recall_at_k.cpp:59`).
#[test]
fn recall_at_no_relevant_is_one() {
    let approx = [3.0, 2.0];
    let target = [0.0, 0.0];
    let got = recall_at_group(&approx, &target, -1, 0.5);
    assert!((got - 1.0).abs() < TOL, "rec = {got}");
}

/// MAP@k full group (border 0.5): approx [4,3,2,1] target [1,0,1,0] sorted
/// [d0,d1,d2,d3]. hits at pos0 (1), pos2 (2). score = 1/1 + 2/3 = 1.6666...;
/// hits=2 → AP = 1.6666.../2 = 0.8333...
#[test]
fn map_at_k_hand_computed() {
    let approx = [4.0, 3.0, 2.0, 1.0];
    let target = [1.0, 0.0, 1.0, 0.0];
    let got = map_at_group(&approx, &target, -1, 0.5);
    let expected = (1.0 + 2.0 / 3.0) / 2.0;
    assert!((got - expected).abs() < TOL, "map = {got}, want {expected}");
}

/// MAP with no relevant doc → 0.
#[test]
fn map_no_relevant_is_zero() {
    let approx = [2.0, 1.0];
    let target = [0.0, 0.0];
    let got = map_at_group(&approx, &target, -1, 0.5);
    assert!(got.abs() < TOL, "map = {got}");
}

/// Ranking QueryAUC: perfectly ordered group → AUC 1. approx [3,2,1] target
/// [2,1,0] (prediction order matches target order, no inversions).
#[test]
fn query_auc_ranking_perfect_is_one() {
    let approx = [3.0, 2.0, 1.0];
    let target = [2.0, 1.0, 0.0];
    let got = query_auc_group(&approx, &target, AucType::Ranking);
    assert!((got - 1.0).abs() < TOL, "ranking auc = {got}");
}

/// Classic QueryAUC: binary target, perfect separation → AUC 1. approx [3,2,1,0]
/// target [1,1,0,0] → all positives outrank all negatives.
#[test]
fn query_auc_classic_perfect_is_one() {
    let approx = [3.0, 2.0, 1.0, 0.0];
    let target = [1.0, 1.0, 0.0, 0.0];
    let got = query_auc_group(&approx, &target, AucType::Classic);
    assert!((got - 1.0).abs() < TOL, "classic auc = {got}");
}

/// Classic QueryAUC: one swapped pair. approx [2,3,1,0] target [1,1,0,0]: both
/// positives (pred 2,3) still outrank both negatives (pred 1,0) → AUC 1.
/// Now make a real inversion: approx [1,2,3,0] target [1,0,0,1].
/// positives: doc0(pred1,w1), doc3(pred0,w1); negatives: doc1(pred2,w1),
/// doc2(pred3,w1). pairs: (p=1 vs n=2)→0, (p=1 vs n=3)→0, (p=0 vs n=2)→0,
/// (p=0 vs n=3)→0 → numerator 0 → AUC 0.
#[test]
fn query_auc_classic_worst_is_zero() {
    let approx = [1.0, 2.0, 3.0, 0.0];
    let target = [1.0, 0.0, 0.0, 1.0];
    let got = query_auc_group(&approx, &target, AucType::Classic);
    assert!(got.abs() < TOL, "classic auc worst = {got}");
}
