//! ORCH-01-S5 (AT-S5): the published facade `catboost_rs::cv` reproduces
//! upstream `catboost.cv(pool, params, folds=<fixed>, shuffle=False)`'s
//! per-iteration `test-`/`train-` RMSE mean/std columns ≤1e-5 on the frozen
//! `cb-oracle/fixtures/cv/` fixtures.
//!
//! The fixture's `params` dict pins `boost_from_average=True`,
//! `bootstrap_type="No"` + `random_strength=0` (SPEC §1.1); this test pins the
//! SAME three values EXPLICITLY on the `CatBoostBuilder` so the oracle's
//! dependency on them is visible and self-documenting rather than riding
//! `new()`'s current defaults.
//!
//! Root cause (RESOLVED): the earlier BLOCKED fixture was generated with
//! catboost's raw `cv()` default `random_strength=1.0`, while the facade default
//! is `0.0` (`builder.rs:107`). `random_strength` perturbs split SCORES — not the
//! feature borders or scale/bias — so the fixture generator's per-fold
//! border/scale-bias parity self-check reported 0 mismatches and could not catch
//! it. With `random_strength=0` pinned on BOTH sides, the Rust facade reproduces
//! catboost 1.2.10 to ~2e-8 across all folds (every tree bit-identical).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{cv, CatBoostBuilder, EBootstrapType, IngestSource, OwnedColumns, Pool};
use cb_oracle::{compare_stage, load_f64_vec, Stage};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

fn cv_fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("cv")
        .join(rel)
}

/// The frozen 30x4 numeric-regression Pool from `cv/X.npy` + `cv/y.npy`.
fn cv_pool() -> Pool {
    let x: Array2<f64> = read_npy(cv_fixture("X.npy")).expect("cv/X.npy loads");
    let y: Array1<f64> = read_npy(cv_fixture("y.npy")).expect("cv/y.npy loads");
    let float_features: Vec<Vec<f64>> = (0..x.ncols())
        .map(|fi| x.column(fi).iter().copied().collect())
        .collect();
    OwnedColumns::new(float_features, y.to_vec())
        .into_pool()
        .expect("cv pool builds")
}

/// The explicit fixed folds (`cv/folds.json`): a list of TEST-row index lists.
fn cv_folds() -> Vec<Vec<usize>> {
    let text = std::fs::read_to_string(cv_fixture("folds.json")).expect("cv/folds.json reads");
    serde_json::from_str(&text).expect("cv/folds.json parses")
}

/// The pinned builder from `cv/params.json`, with `boost_from_average` +
/// `bootstrap_type` set EXPLICITLY (see the module doc).
fn cv_builder() -> CatBoostBuilder {
    let text = std::fs::read_to_string(cv_fixture("params.json")).expect("cv/params.json reads");
    let params: serde_json::Value = serde_json::from_str(&text).expect("cv/params.json parses");

    let iterations = params["iterations"].as_u64().expect("iterations") as usize;
    let learning_rate = params["learning_rate"].as_f64().expect("learning_rate");
    let depth = params["depth"].as_u64().expect("depth") as usize;
    let border_count = params["border_count"].as_u64().expect("border_count") as usize;
    assert_eq!(
        params["loss_function"].as_str(),
        Some("RMSE"),
        "fixture loss_function must be RMSE"
    );

    CatBoostBuilder::new()
        .iterations(iterations)
        .learning_rate(learning_rate)
        .depth(depth)
        .border_count(border_count)
        // Pinned explicitly (matching the fixture's params dict), not inherited.
        // random_strength=0 (facade default, builder.rs:107) is pinned here too:
        // catboost's raw cv default is 1.0, which perturbs split SCORES (not
        // borders/bias) and diverges the RMSE curve; pinning it on both sides is
        // what makes the oracle reproduce upstream to ~1e-8.
        .boost_from_average(true)
        .bootstrap_type(EBootstrapType::No)
        .random_strength(0.0)
}

/// Largest absolute per-element difference (for reporting alongside the assert).
fn max_abs_diff(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(e, a)| (e - a).abs())
        .fold(0.0_f64, f64::max)
}

/// AT-S5: per-iteration `test-`/`train-` RMSE mean/std ≤1e-5 vs the committed
/// `catboost.cv` fixtures. Passes once `random_strength=0` is pinned on both the
/// fixture and this builder (see the module doc); the tolerance is NOT loosened.
#[test]
fn cv_oracle_matches_upstream_columns() {
    let pool = cv_pool();
    let builder = cv_builder();
    let folds = cv_folds();

    let result = cv(&pool, &builder, &["RMSE"], 3, false, 0, false, Some(&folds))
        .expect("cv runs on the frozen fixture");

    let cases = [
        ("test-RMSE-mean", "test_rmse_mean.npy"),
        ("test-RMSE-std", "test_rmse_std.npy"),
        ("train-RMSE-mean", "train_rmse_mean.npy"),
        ("train-RMSE-std", "train_rmse_std.npy"),
    ];

    for (column, npy) in cases {
        let expected = load_f64_vec(&cv_fixture(npy)).expect("expected column loads");
        let actual = result
            .columns
            .get(column)
            .unwrap_or_else(|| panic!("cv result is missing column `{column}`"));
        assert_eq!(
            actual.len(),
            expected.len(),
            "column `{column}` length mismatch"
        );
        let mad = max_abs_diff(&expected, actual);
        println!("cv oracle column {column}: max|diff| = {mad:e}");
        compare_stage(Stage::Predictions, &expected, actual)
            .unwrap_or_else(|e| panic!("column `{column}` diverged > 1e-5: {e}"));
    }
}

#[test]
fn cv_empty_metrics_errs() {
    let pool = cv_pool();
    let builder = cv_builder();
    let folds = cv_folds();
    assert!(
        cv(&pool, &builder, &[], 3, false, 0, false, Some(&folds)).is_err(),
        "empty metrics must be a typed error, not a panic"
    );
}

#[test]
fn cv_empty_explicit_folds_errs() {
    let pool = cv_pool();
    let builder = cv_builder();
    let empty: [Vec<usize>; 0] = [];
    let res = cv(&pool, &builder, &["RMSE"], 3, false, 0, false, Some(&empty));
    assert!(
        res.is_err(),
        "an explicit but empty folds list must error, never a NaN-bearing Ok"
    );
}
