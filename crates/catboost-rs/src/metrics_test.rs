//! Facade-level tests for the standalone `eval_metric` / `eval_metrics` surface
//! (ORCH-04-S5). Mounted at the crate root via `#[cfg(test)] mod metrics_test;`
//! (the facade's root-level test-mount idiom, cf. `mod error_test;`).

use crate::{eval_metric, eval_metrics, CatBoostError};

#[test]
fn facade_rmse_matches() {
    // label=[1,0], approx=[0,1] => diffs [-1,1]; RMSE = sqrt((1+1)/2) = 1.0.
    let label = [1.0_f64, 0.0];
    let approx = [0.0_f64, 1.0];
    let got = eval_metric(&label, &approx, "RMSE", None, None).unwrap();
    assert!((got - 1.0).abs() < 1e-5, "got {got}");
}

#[test]
fn facade_list() {
    let label = [1.0_f64, 0.0];
    let approx = [0.0_f64, 1.0];
    let v = eval_metrics(&label, &approx, &["RMSE", "MSLE"], None, None).unwrap();
    assert_eq!(v.len(), 2);
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn facade_unknown_metric_errs() {
    let label = [1.0_f64, 0.0];
    let approx = [0.0_f64, 1.0];
    let err = eval_metric(&label, &approx, "bogus", None, None).unwrap_err();
    // The underlying CbError maps to CatBoostError::Train via `#[from]`.
    assert!(matches!(err, CatBoostError::Train(_)), "unexpected variant: {err:?}");
}
