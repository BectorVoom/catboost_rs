//! Grouped ranking `calc_metric` ≤1e-5 oracle (ORCH-04-S3) reusing the EXISTING
//! `ranking_corpus/ranking_metrics/*.npy` fixtures (NO new fixtures).
//!
//! This locks the ranking routing of the standalone `calc_metric` surface: it
//! drives `calc_metric(&metric, &target, &approx, &[], &group_id)` and gates the
//! scalar against the same committed catboost 1.2.10 reference the direct
//! `eval_grouped` oracle uses (`ranking_metrics_oracle_test.rs`). Coverage: every
//! metric at DEFAULT params, each @k metric at `top=2`, QueryAUC Ranking +
//! Classic, plus the empty-group (single group) and non-contiguous-group edge
//! cases.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::{compare_stage, load_f64_vec, Stage};
use cb_train::{calc_metric, AucType, DcgDenominator, DcgMetricType, EvalMetric};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("ranking_corpus")
        .join("ranking_metrics")
        .join(rel)
}

/// Shared frozen metric inputs `(target, approx, group_id)`.
fn metric_inputs() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    let target = load_f64_vec(&fixture("target.npy")).unwrap();
    let approx = load_f64_vec(&fixture("approx.npy")).unwrap();
    let group_id_f = load_f64_vec(&fixture("group_id.npy")).unwrap();
    let group_id: Vec<u64> = group_id_f.iter().map(|&g| g as u64).collect();
    (target, approx, group_id)
}

/// Evaluate one ranking metric via `calc_metric` over the frozen inputs and gate
/// it against its committed upstream scalar at ≤1e-5.
fn gate(metric: EvalMetric, fixture_name: &str, target: &[f64], approx: &[f64], group_id: &[u64]) {
    let expected = load_f64_vec(&fixture(fixture_name)).unwrap();
    assert_eq!(expected.len(), 1, "{fixture_name}: one scalar metric value");
    let actual = calc_metric(&metric, target, approx, &[], group_id)
        .unwrap_or_else(|e| panic!("{fixture_name}: calc_metric failed: {e:?}"));
    compare_stage(Stage::Predictions, &expected, &[actual])
        .unwrap_or_else(|e| panic!("{fixture_name}: diverged from upstream: {e:?}"));
}

// --- default-param metrics --------------------------------------------------

#[test]
fn ndcg_default() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Ndcg { top: -1, dcg_type: DcgMetricType::Base, denominator: DcgDenominator::LogPosition },
        "ndcg.npy", &t, &a, &g,
    );
}

#[test]
fn dcg_default() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Dcg { top: -1, dcg_type: DcgMetricType::Base, denominator: DcgDenominator::LogPosition },
        "dcg.npy", &t, &a, &g,
    );
}

#[test]
fn map_default() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Map { top: -1, border: 0.5 }, "map.npy", &t, &a, &g);
}

#[test]
fn mrr_default() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Mrr { top: -1, border: 0.5 }, "mrr.npy", &t, &a, &g);
}

#[test]
fn err_default() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Err { top: -1 }, "err.npy", &t, &a, &g);
}

#[test]
fn pfound_default() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::PFound { top: -1, decay: 0.85 }, "pfound.npy", &t, &a, &g);
}

#[test]
fn precision_at_default() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::PrecisionAt { top: -1, border: 0.5 }, "precision_at.npy", &t, &a, &g);
}

#[test]
fn recall_at_default() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::RecallAt { top: -1, border: 0.5 }, "recall_at.npy", &t, &a, &g);
}

#[test]
fn queryauc_ranking() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::QueryAuc { auc_type: AucType::Ranking }, "queryauc_ranking.npy", &t, &a, &g);
}

#[test]
fn queryauc_classic() {
    let (_, a, g) = metric_inputs();
    let binary_target = load_f64_vec(&fixture("binary_target.npy")).unwrap();
    gate(EvalMetric::QueryAuc { auc_type: AucType::Classic }, "queryauc_classic.npy", &binary_target, &a, &g);
}

// --- explicit top=2 cases ---------------------------------------------------

#[test]
fn ndcg_top2() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Ndcg { top: 2, dcg_type: DcgMetricType::Base, denominator: DcgDenominator::LogPosition },
        "ndcg_top2.npy", &t, &a, &g,
    );
}

#[test]
fn dcg_top2() {
    let (t, a, g) = metric_inputs();
    gate(
        EvalMetric::Dcg { top: 2, dcg_type: DcgMetricType::Base, denominator: DcgDenominator::LogPosition },
        "dcg_top2.npy", &t, &a, &g,
    );
}

#[test]
fn map_top2() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Map { top: 2, border: 0.5 }, "map_top2.npy", &t, &a, &g);
}

#[test]
fn mrr_top2() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Mrr { top: 2, border: 0.5 }, "mrr_top2.npy", &t, &a, &g);
}

#[test]
fn err_top2() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::Err { top: 2 }, "err_top2.npy", &t, &a, &g);
}

#[test]
fn pfound_top2() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::PFound { top: 2, decay: 0.85 }, "pfound_top2.npy", &t, &a, &g);
}

#[test]
fn precision_at_top2() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::PrecisionAt { top: 2, border: 0.5 }, "precision_at_top2.npy", &t, &a, &g);
}

#[test]
fn recall_at_top2() {
    let (t, a, g) = metric_inputs();
    gate(EvalMetric::RecallAt { top: 2, border: 0.5 }, "recall_at_top2.npy", &t, &a, &g);
}

// --- group edge cases -------------------------------------------------------

/// An empty `group_id` treats the whole eval set as a single group (Ok).
#[test]
fn empty_group_is_single_group() {
    let (t, a, _) = metric_inputs();
    let got = calc_metric(
        &EvalMetric::Ndcg { top: -1, dcg_type: DcgMetricType::Base, denominator: DcgDenominator::LogPosition },
        &t, &a, &[], &[],
    );
    assert!(got.is_ok(), "empty group_id should evaluate as one group: {got:?}");
    assert!(got.unwrap().is_finite());
}

/// A non-contiguous `group_id` (an id reappearing after another intervened) is a
/// typed error, not a panic.
#[test]
fn non_contiguous_group_id_errs() {
    let target = [3.0_f64, 2.0, 1.0];
    let approx = [0.9_f64, 0.1, 0.5];
    let group_id = [0_u64, 1, 0]; // 0 reappears after 1 => non-contiguous
    let got = calc_metric(
        &EvalMetric::Ndcg { top: -1, dcg_type: DcgMetricType::Base, denominator: DcgDenominator::LogPosition },
        &target, &approx, &[], &group_id,
    );
    assert!(got.is_err(), "non-contiguous group_id should error: {got:?}");
}
