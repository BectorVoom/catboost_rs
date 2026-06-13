//! Unit tests for the overfitting-detection state machine (TRAIN-06): each
//! detector type's stop decision on a synthetic eval-metric sequence, plus
//! `use_best_model` best-iteration tracking. These pin the ported
//! `overfitting_detector.cpp` semantics independently of the training loop /
//! oracle (the oracle test locks the end-to-end stop iteration).

use super::*;

/// A monotonically DECREASING loss (eval keeps improving) must NEVER stop: the
/// local max keeps advancing so `IterationsFromLocalMax` stays 0.
#[test]
fn inctodec_never_stops_on_improving_loss() {
    let mut det = OverfittingDetector::new(EOverfittingDetectorType::IncToDec, 0.5, 5, true)
        .expect("detector constructs");
    // maxIsOptimal=false (loss): decreasing loss is improving.
    for k in 0..50 {
        let loss = 10.0 - 0.1 * f64::from(k);
        det.add_error(loss);
        assert!(!det.is_need_stop(), "improving loss must not stop (iter {k})");
    }
}

/// A loss that bottoms out then climbs must eventually trigger IncToDec once the
/// p-value (built from the increasing-then-decreasing pattern) drops below the
/// threshold after `od_wait` iterations past the local optimum.
#[test]
fn inctodec_stops_after_loss_climbs() {
    let mut det = OverfittingDetector::new(EOverfittingDetectorType::IncToDec, 0.99, 5, true)
        .expect("detector constructs");
    let mut stopped_at = None;
    // 10 improving iterations (loss 10 -> 1), then 30 worsening (loss climbs).
    let mut losses = Vec::new();
    for k in 0..10 {
        losses.push(10.0 - f64::from(k));
    }
    for k in 0..30 {
        losses.push(1.0 + 0.5 * f64::from(k));
    }
    for (i, &loss) in losses.iter().enumerate() {
        det.add_error(loss);
        if det.is_need_stop() {
            stopped_at = Some(i);
            break;
        }
    }
    let at = stopped_at.expect("IncToDec must stop once the loss climbs");
    // Best (lowest loss) is iteration 9; the stop must come strictly after it,
    // and at least `od_wait` iterations past the local max.
    assert!(at > 9, "stop must be after the best iteration (got {at})");
    assert!(at >= 9 + 5, "stop must be >= od_wait past the local max (got {at})");
}

/// IncToDec is INACTIVE when the threshold is 0 (the upstream default `od_pval`):
/// `IsActive()` is false and `IsNeedStop()` never fires regardless of the curve.
#[test]
fn inctodec_inactive_when_threshold_zero() {
    let mut det = OverfittingDetector::new(EOverfittingDetectorType::IncToDec, 0.0, 5, true)
        .expect("detector constructs");
    assert!(!det.is_active(), "threshold 0 => inactive");
    for k in 0..40 {
        // A clearly-overfitting curve.
        let loss = if k < 5 { 10.0 - f64::from(k) } else { 5.0 + f64::from(k) };
        det.add_error(loss);
        assert!(!det.is_need_stop(), "inactive detector must never stop");
    }
}

/// Iter == IncToDec with the threshold forced to 1.0: it stops EXACTLY `od_wait`
/// iterations after the best (local-max) iteration, because the p-value is < 1.0
/// the moment the wait elapses past the max.
#[test]
fn iter_stops_exactly_od_wait_after_best() {
    let od_wait = 7;
    let mut det = OverfittingDetector::new(EOverfittingDetectorType::Iter, 0.0, od_wait, true)
        .expect("detector constructs");
    // Best at iteration 4 (loss minimum), then strictly worsening.
    let mut losses: Vec<f64> = (0..5).map(|k| 10.0 - f64::from(k)).collect();
    losses.extend((0..20).map(|k| 6.0 + f64::from(k)));
    let mut stopped_at = None;
    for (i, &loss) in losses.iter().enumerate() {
        det.add_error(loss);
        if det.is_need_stop() {
            stopped_at = Some(i);
            break;
        }
    }
    // Local max (best) is iteration 4; Iter stops when IterationsFromLocalMax
    // reaches od_wait, i.e. at iteration 4 + od_wait.
    assert_eq!(
        stopped_at,
        Some(4 + od_wait),
        "Iter must stop exactly od_wait iterations after the best"
    );
}

/// Wilcoxon must NOT stop while fewer than `od_wait` post-local-max deltas have
/// accumulated (the p-value stays at the neutral 1.0 until the wait elapses).
#[test]
fn wilcoxon_holds_until_wait_deltas_accumulate() {
    let mut det = OverfittingDetector::new(EOverfittingDetectorType::Wilcoxon, 0.5, 10, true)
        .expect("detector constructs");
    // Five improving steps then four worsening: only 4 post-max deltas < wait=10.
    let losses = [10.0, 9.0, 8.0, 7.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    for &loss in &losses {
        det.add_error(loss);
        assert!(
            !det.is_need_stop(),
            "Wilcoxon must hold until od_wait deltas accumulate"
        );
    }
}

/// Wilcoxon fires once enough consistently-worsening post-local-max deltas
/// accumulate that the signed-rank p-value drops below the threshold.
#[test]
fn wilcoxon_stops_on_consistent_worsening() {
    let mut det = OverfittingDetector::new(EOverfittingDetectorType::Wilcoxon, 0.5, 10, true)
        .expect("detector constructs");
    let mut losses = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    // Many consistently-worsening steps after the minimum (loss climbs steadily).
    for k in 0..40 {
        losses.push(2.0 + 0.5 * f64::from(k + 1));
    }
    let mut stopped = false;
    for &loss in &losses {
        det.add_error(loss);
        if det.is_need_stop() {
            stopped = true;
            break;
        }
    }
    assert!(stopped, "Wilcoxon must stop on consistent worsening");
}

/// `use_best_model` tracks the iteration with the BEST (lowest, for a loss) eval
/// metric; ties keep the FIRST (earliest) best, matching upstream
/// `best_model_min_trees` first-wins behaviour.
#[test]
fn best_model_tracks_lowest_loss_iteration() {
    let mut best = BestModelTracker::new();
    let losses = [5.0, 4.0, 3.0, 3.5, 2.0, 2.5, 2.0];
    for &loss in &losses {
        best.add_error(loss);
    }
    // Minimum 2.0 first occurs at index 4 (later equal 2.0 at index 6 must NOT win).
    assert_eq!(best.best_iteration(), Some(4));
}

/// An empty error stream has no best iteration.
#[test]
fn best_model_empty_has_no_best() {
    let best = BestModelTracker::new();
    assert_eq!(best.best_iteration(), None);
}

/// The Wilcoxon detector requires a test set when the threshold is positive
/// (upstream `CB_ENSURE(hasTest || threshold == 0)`); a positive threshold with
/// no eval set is a `CbError`, never a panic.
#[test]
fn wilcoxon_without_test_and_positive_threshold_errors() {
    let res = OverfittingDetector::new(EOverfittingDetectorType::Wilcoxon, 0.5, 10, false);
    assert!(res.is_err(), "Wilcoxon + positive threshold + no test must error");
}

/// The `None` detector type is always inactive and never stops.
#[test]
fn none_detector_never_stops() {
    let mut det = OverfittingDetector::new(EOverfittingDetectorType::None, 0.99, 1, true)
        .expect("detector constructs");
    assert!(!det.is_active());
    for k in 0..30 {
        det.add_error(100.0 + f64::from(k));
        assert!(!det.is_need_stop());
    }
}
