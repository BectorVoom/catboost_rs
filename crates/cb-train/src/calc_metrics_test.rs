//! Unit tests for the standalone `calc_metrics` surface (ORCH-04).
//!
//! Mounted into `calc_metrics.rs` via `#[cfg(test)] #[path = ...] mod tests;`
//! (source/test separation, CLAUDE.md). The crate-level
//! `#![cfg_attr(test, allow(clippy::unwrap_used, ...))]` (lib.rs:1) lets these
//! tests `unwrap`/assert freely.

use super::{calc_metric, eval_metric, parse_metric};
use crate::ranking_metrics::{AucType, DcgDenominator, DcgMetricType};
use crate::EvalMetric;

// --- ORCH-04-S1: metric-descriptor parser -----------------------------------

#[test]
fn parse_ndcg_with_params() {
    let got = parse_metric("NDCG:top=2:type=Exp:denominator=Position").unwrap();
    assert_eq!(
        got,
        EvalMetric::Ndcg {
            top: 2,
            dcg_type: DcgMetricType::Exp,
            denominator: DcgDenominator::Position,
        }
    );
}

#[test]
fn parse_defaults() {
    // NDCG with no params takes the upstream defaults.
    assert_eq!(
        parse_metric("NDCG").unwrap(),
        EvalMetric::Ndcg {
            top: -1,
            dcg_type: DcgMetricType::Base,
            denominator: DcgDenominator::LogPosition,
        }
    );
    // Case-insensitive metric name.
    assert_eq!(parse_metric("rmse").unwrap(), EvalMetric::Rmse);
    assert_eq!(parse_metric("Logloss").unwrap(), EvalMetric::Logloss);
    assert_eq!(parse_metric("MSLE").unwrap(), EvalMetric::Msle);
}

#[test]
fn parse_enum_param_tolerates_whitespace() {
    // Enum params are trimmed before matching, consistent with numeric params
    // (`top= 2` is already tolerated). A stray space must not reject the parse.
    assert_eq!(
        parse_metric("NDCG:type= Exp :denominator= Position").unwrap(),
        EvalMetric::Ndcg {
            top: -1,
            dcg_type: DcgMetricType::Exp,
            denominator: DcgDenominator::Position,
        }
    );
    assert_eq!(
        parse_metric("QueryAUC:type= Ranking").unwrap(),
        EvalMetric::QueryAuc { auc_type: AucType::Ranking }
    );
}

#[test]
fn parse_ranking_param_defaults() {
    assert_eq!(parse_metric("MAP").unwrap(), EvalMetric::Map { top: -1, border: 0.5 });
    assert_eq!(parse_metric("MRR").unwrap(), EvalMetric::Mrr { top: -1, border: 0.5 });
    assert_eq!(parse_metric("ERR").unwrap(), EvalMetric::Err { top: -1 });
    assert_eq!(
        parse_metric("PFound").unwrap(),
        EvalMetric::PFound { top: -1, decay: 0.85 }
    );
    assert_eq!(
        parse_metric("PrecisionAt").unwrap(),
        EvalMetric::PrecisionAt { top: -1, border: 0.5 }
    );
    assert_eq!(
        parse_metric("RecallAt").unwrap(),
        EvalMetric::RecallAt { top: -1, border: 0.5 }
    );
    assert_eq!(
        parse_metric("DCG:top=3:type=Base").unwrap(),
        EvalMetric::Dcg {
            top: 3,
            dcg_type: DcgMetricType::Base,
            denominator: DcgDenominator::LogPosition,
        }
    );
    assert_eq!(
        parse_metric("MAP:top=5:border=0.25").unwrap(),
        EvalMetric::Map { top: 5, border: 0.25 }
    );
    assert_eq!(
        parse_metric("PFound:decay=0.5").unwrap(),
        EvalMetric::PFound { top: -1, decay: 0.5 }
    );
}

#[test]
fn parse_queryauc() {
    assert_eq!(
        parse_metric("QueryAUC:type=Ranking").unwrap(),
        EvalMetric::QueryAuc { auc_type: AucType::Ranking }
    );
    assert_eq!(
        parse_metric("QueryAUC").unwrap(),
        EvalMetric::QueryAuc { auc_type: AucType::Classic }
    );
    assert_eq!(
        parse_metric("queryauc:type=classic").unwrap(),
        EvalMetric::QueryAuc { auc_type: AucType::Classic }
    );
}

#[test]
fn parse_rejects_unknown() {
    // Unknown metric name.
    assert!(parse_metric("NoSuchMetric").is_err());
    // Unknown param key for a known metric.
    assert!(parse_metric("NDCG:bogus=1").is_err());
    // Unparseable value.
    assert!(parse_metric("NDCG:top=notanint").is_err());
    // A key not valid for this metric (border is not an NDCG key).
    assert!(parse_metric("NDCG:border=0.5").is_err());
    // Wrong enum value for type.
    assert!(parse_metric("QueryAUC:type=Base").is_err());
    // Malformed token (no '=').
    assert!(parse_metric("NDCG:top").is_err());
    // Duplicate key.
    assert!(parse_metric("NDCG:top=1:top=2").is_err());
}

// --- ORCH-04-S2 (weighting): deterministic hand-computed weighted RMSE -------
// A closed-form check that the weight column threads through `calc_metric`
// (belt-and-suspenders alongside the `rmse_weighted` oracle fixture).

#[test]
fn calc_metric_weighted_rmse_hand() {
    // approx-target diffs: [1, -1, 2]; weights [1, 2, 3].
    // weighted sq err = 1*1 + 2*1 + 3*4 = 15; sum_w = 6; sqrt(15/6) = sqrt(2.5).
    let approx = [2.0_f64, 1.0, 5.0];
    let label = [1.0_f64, 2.0, 3.0];
    let weight = [1.0_f64, 2.0, 3.0];
    let got = calc_metric(&EvalMetric::Rmse, &label, &approx, &weight, &[]).unwrap();
    assert!((got - 2.5_f64.sqrt()).abs() < 1e-12, "got {got}");
}

// --- ORCH-04-S4: dispatch + validation --------------------------------------

#[test]
fn dispatch_two_metrics() {
    // label in {0,1}, approx > -1 so RMSE + Logloss both valid.
    let label = [0.0_f64, 1.0, 1.0, 0.0];
    let approx = [0.2_f64, 0.9, -0.3, 0.5];
    let v = eval_metric(&label, &approx, &["RMSE", "Logloss"], &[], &[]).unwrap();
    assert_eq!(v.len(), 2);
    let rmse = calc_metric(&EvalMetric::Rmse, &label, &approx, &[], &[]).unwrap();
    let logloss = calc_metric(&EvalMetric::Logloss, &label, &approx, &[], &[]).unwrap();
    assert!((v[0] - rmse).abs() < 1e-12);
    assert!((v[1] - logloss).abs() < 1e-12);
}

#[test]
fn dispatch_ranking_empty_group() {
    // A ranking metric with empty group_id evaluates as a single group (Ok).
    let label = [3.0_f64, 2.0, 1.0, 0.0];
    let approx = [0.9_f64, 0.1, 0.5, 0.2];
    let v = eval_metric(&label, &approx, &["NDCG"], &[], &[]).unwrap();
    assert_eq!(v.len(), 1);
    assert!(v[0].is_finite());
}

#[test]
fn dispatch_length_mismatch_errs() {
    let label = [0.0_f64, 1.0, 1.0];
    let approx = [0.2_f64, 0.9]; // shorter
    assert!(eval_metric(&label, &approx, &["RMSE"], &[], &[]).is_err());
}

#[test]
fn dispatch_unknown_metric_errs() {
    let label = [0.0_f64, 1.0];
    let approx = [0.2_f64, 0.9];
    assert!(eval_metric(&label, &approx, &["bogus"], &[], &[]).is_err());
}
