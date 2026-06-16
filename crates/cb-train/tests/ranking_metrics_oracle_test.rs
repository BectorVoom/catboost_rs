//! Per-metric ≤1e-5 oracle for all nine ranking metrics (LOSS-05, Plan 06.3-05)
//! against frozen catboost 1.2.10 reference values.
//!
//! Ranking metrics are eval-only, so the Python-reachable ground truth is
//! `catboost.utils.eval_metric(label, approx, metric, group_id=...)` over a FIXED,
//! KNOWN approx vector (NOT a trained-model prediction). The Rust
//! [`cb_train::EvalMetric::eval_grouped`] is fed the SAME approx/target/group_id
//! and gated `compare_stage(Stage::Predictions, expected, actual)` ≤1e-5 per
//! metric — the metric scalar is the "prediction" being compared.
//!
//! Fixtures were generated OFFLINE (RUN-ONCE/COMMIT, D-08) by:
//!     .venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --metrics-eval
//! CI only READS the committed `ranking_corpus/ranking_metrics/*.npy`.
//!
//! Coverage: every metric at its DEFAULT params, plus an explicit `top=2` case for
//! each @k metric (exercising the nth_element / tie path), plus QueryAUC in BOTH
//! singleclass modes (Ranking over graded relevance, Classic over a binarized
//! target).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::{compare_stage, load_f64_vec, Stage};
use cb_train::{AucType, DcgDenominator, DcgMetricType, EvalMetric};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("ranking_corpus")
        .join("ranking_metrics")
        .join(rel)
}

/// Shared frozen metric inputs: `(target, approx, group_id)`.
fn metric_inputs() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    let target = load_f64_vec(&fixture("target.npy")).unwrap();
    let approx = load_f64_vec(&fixture("approx.npy")).unwrap();
    let group_id_f = load_f64_vec(&fixture("group_id.npy")).unwrap();
    let group_id: Vec<u64> = group_id_f.iter().map(|&g| g as u64).collect();
    (target, approx, group_id)
}

/// Evaluate one metric over the frozen inputs and gate it against its committed
/// upstream scalar at ≤1e-5.
fn gate(metric: EvalMetric, fixture_name: &str, target: &[f64], approx: &[f64], group_id: &[u64]) {
    let expected = load_f64_vec(&fixture(fixture_name)).unwrap();
    assert_eq!(expected.len(), 1, "{fixture_name}: one scalar metric value");
    let actual = metric
        .eval_grouped(approx, target, &[], group_id, &[])
        .unwrap_or_else(|e| panic!("{fixture_name}: eval_grouped failed: {e:?}"));
    compare_stage(Stage::Predictions, &expected, &[actual])
        .unwrap_or_else(|e| panic!("{fixture_name}: diverged from upstream: {e:?}"));
}

// --- default-param metrics (top=-1, border=0.5, decay=0.85) -----------------

#[test]
fn ndcg_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Ndcg {
            top: -1,
            dcg_type: DcgMetricType::Base,
            denominator: DcgDenominator::LogPosition,
        },
        "ndcg.npy",
        &t,
        &a,
        &g,
    );
}

#[test]
fn dcg_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Dcg {
            top: -1,
            dcg_type: DcgMetricType::Base,
            denominator: DcgDenominator::LogPosition,
        },
        "dcg.npy",
        &t,
        &a,
        &g,
    );
}

#[test]
fn map_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Map { top: -1, border: 0.5 }, "map.npy", &t, &a, &g);
}

#[test]
fn mrr_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Mrr { top: -1, border: 0.5 }, "mrr.npy", &t, &a, &g);
}

#[test]
fn err_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Err { top: -1 }, "err.npy", &t, &a, &g);
}

#[test]
fn pfound_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::PFound { top: -1, decay: 0.85 }, "pfound.npy", &t, &a, &g);
}

#[test]
fn precision_at_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::PrecisionAt { top: -1, border: 0.5 },
        "precision_at.npy",
        &t,
        &a,
        &g,
    );
}

#[test]
fn recall_at_default_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::RecallAt { top: -1, border: 0.5 },
        "recall_at.npy",
        &t,
        &a,
        &g,
    );
}

#[test]
fn queryauc_ranking_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::QueryAuc {
            auc_type: AucType::Ranking,
        },
        "queryauc_ranking.npy",
        &t,
        &a,
        &g,
    );
}

/// QueryAUC Classic (singleclass default) over a binarized target (`target > 1.5
/// → 1`), the form Classic AUC requires (`target ∈ [0,1]`).
#[test]
fn queryauc_classic_matches_upstream() {
    let (_, a, g) = metric_inputs();
    let binary_target = load_f64_vec(&fixture("binary_target.npy")).unwrap();
    gate(
        EvalMetric::QueryAuc {
            auc_type: AucType::Classic,
        },
        "queryauc_classic.npy",
        &binary_target,
        &a,
        &g,
    );
}

// --- explicit top=2 cases (exercise the nth_element / tie path) -------------

#[test]
fn ndcg_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Ndcg {
            top: 2,
            dcg_type: DcgMetricType::Base,
            denominator: DcgDenominator::LogPosition,
        },
        "ndcg_top2.npy",
        &t,
        &a,
        &g,
    );
}

#[test]
fn dcg_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Dcg {
            top: 2,
            dcg_type: DcgMetricType::Base,
            denominator: DcgDenominator::LogPosition,
        },
        "dcg_top2.npy",
        &t,
        &a,
        &g,
    );
}

#[test]
fn map_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Map { top: 2, border: 0.5 }, "map_top2.npy", &t, &a, &g);
}

#[test]
fn mrr_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Mrr { top: 2, border: 0.5 }, "mrr_top2.npy", &t, &a, &g);
}

#[test]
fn err_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Err { top: 2 }, "err_top2.npy", &t, &a, &g);
}

#[test]
fn pfound_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::PFound { top: 2, decay: 0.85 }, "pfound_top2.npy", &t, &a, &g);
}

#[test]
fn precision_at_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::PrecisionAt { top: 2, border: 0.5 },
        "precision_at_top2.npy",
        &t,
        &a,
        &g,
    );
}

#[test]
fn recall_at_top2_matches_upstream() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::RecallAt { top: 2, border: 0.5 },
        "recall_at_top2.npy",
        &t,
        &a,
        &g,
    );
}
