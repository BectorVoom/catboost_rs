//! ORCH-02-S3 (AT-S3a/b): the published facade `catboost_rs::grid_search`
//! decomposes into `catboost_rs::cv` per candidate + the documented argbest
//! selection — a SELF-CONSISTENCY oracle (same seams ⇒ machine precision) on
//! the frozen `cb-oracle/fixtures/cv/` corpus. No NEW fixtures: the three
//! candidates differ only in `depth ∈ {2,4,6}`, cross-validated over the same
//! fixed folds the ORCH-01 cv oracle uses.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::PathBuf;

use catboost_rs::{
    cv, grid_search, randomized_search, CatBoostBuilder, EBootstrapType, ErrorScore, IngestSource,
    OwnedColumns, Pool,
};
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

/// The pinned non-depth params from `cv/params.json` (`iterations`,
/// `learning_rate`, `border_count`; loss must be RMSE).
fn base_params() -> (usize, f64, usize) {
    let text = std::fs::read_to_string(cv_fixture("params.json")).expect("cv/params.json reads");
    let params: serde_json::Value = serde_json::from_str(&text).expect("cv/params.json parses");
    assert_eq!(
        params["loss_function"].as_str(),
        Some("RMSE"),
        "fixture loss_function must be RMSE"
    );
    (
        params["iterations"].as_u64().expect("iterations") as usize,
        params["learning_rate"].as_f64().expect("learning_rate"),
        params["border_count"].as_u64().expect("border_count") as usize,
    )
}

/// One candidate built from the fixture params, overriding `depth`, pinning the
/// same three oracle-critical knobs the cv oracle pins.
fn candidate(depth: usize) -> CatBoostBuilder {
    let (iterations, learning_rate, border_count) = base_params();
    CatBoostBuilder::new()
        .iterations(iterations)
        .learning_rate(learning_rate)
        .depth(depth)
        .border_count(border_count)
        .boost_from_average(true)
        .bootstrap_type(EBootstrapType::No)
        .random_strength(0.0)
}

/// The three-candidate grid (`depth ∈ {2,4,6}`).
fn candidates() -> Vec<CatBoostBuilder> {
    vec![candidate(2), candidate(4), candidate(6)]
}

/// A DEGENERATE candidate: `iterations(0)` ⇒ `cv()` returns Ok with an EMPTY
/// `test-RMSE-mean` column (empirically verified), so `score_candidate` fails
/// (empty/non-finite column) — the "cv Ok but unscoreable" `error_score` path.
/// (If a future build instead makes `iterations(0)` error inside `fit`, that is
/// the cv-errored `error_score` path, and both are treated as failures.)
fn degenerate_candidate() -> CatBoostBuilder {
    let (_iterations, learning_rate, border_count) = base_params();
    CatBoostBuilder::new()
        .iterations(0)
        .learning_rate(learning_rate)
        .depth(2)
        .border_count(border_count)
        .boost_from_average(true)
        .bootstrap_type(EBootstrapType::No)
        .random_strength(0.0)
}

/// The four-candidate grid: the three VALID depths plus ONE degenerate candidate
/// at the LAST index (`DEGEN_INDEX == 3`).
const DEGEN_INDEX: usize = 3;
fn candidates_with_degenerate() -> Vec<CatBoostBuilder> {
    vec![candidate(2), candidate(4), candidate(6), degenerate_candidate()]
}

/// Best value over iterations of a `test-RMSE-mean` column (min; RMSE minimizes).
fn best_rmse(col: &[f64]) -> f64 {
    col.iter().copied().fold(f64::MAX, f64::min)
}

/// AT-S3a: `grid_search`'s `cv_results` for the winner is byte-identical to a
/// direct `cv(...)` on that same candidate (same seams), and `best_index` is the
/// independent manual argmin of per-candidate best RMSE.
#[test]
fn grid_search_self_consistency_and_argbest() {
    let pool = cv_pool();
    let cands = candidates();
    let folds = cv_folds();

    let res = grid_search(&pool, &cands, &["RMSE"], 3, false, 0, false, Some(&folds), false, ErrorScore::Raise)
        .expect("grid_search runs on the frozen fixture");

    // SELF-CONSISTENCY: the returned cv_results equal a direct cv on the winner.
    let direct = cv(
        &pool,
        &cands[res.best_index],
        &["RMSE"],
        3,
        false,
        0,
        false,
        Some(&folds),
    )
    .expect("direct cv on the winner runs");
    assert_eq!(
        res.cv_results.columns, direct.columns,
        "grid_search cv_results must equal a direct cv on the winning candidate (same seams)"
    );
    assert_eq!(res.cv_results.iterations, direct.iterations);

    // ARGBEST (independent): manual argmin of per-candidate best RMSE.
    let mut scores = Vec::with_capacity(cands.len());
    for c in &cands {
        let r = cv(&pool, c, &["RMSE"], 3, false, 0, false, Some(&folds)).expect("per-candidate cv");
        scores.push(best_rmse(r.columns.get("test-RMSE-mean").expect("column present")));
    }
    let manual_argmin = scores
        .iter()
        .enumerate()
        .fold((0usize, f64::MAX), |(bi, bv), (i, &v)| {
            if v < bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0;
    println!(
        "grid_search argbest: per-candidate best RMSE = {scores:?}, argmin = {manual_argmin}, best_index = {}",
        res.best_index
    );
    assert_eq!(
        res.best_index, manual_argmin,
        "best_index must be the manual argmin of per-candidate best RMSE"
    );
    // The reported best builder is the depth that won.
    assert_eq!(res.best_builder, cands[res.best_index]);
    assert!(res.best_model.is_none(), "refit=false ⇒ no best_model");
}

/// AT-S3b: `refit=true` ⇒ `best_model` predictions equal a fresh
/// `best_builder.fit(pool)` prediction, ≤1e-9.
#[test]
fn grid_search_refit_matches_direct_fit() {
    let pool = cv_pool();
    let cands = candidates();
    let folds = cv_folds();

    let res = grid_search(&pool, &cands, &["RMSE"], 3, false, 0, false, Some(&folds), true, ErrorScore::Raise)
        .expect("grid_search with refit runs");
    let model = res.best_model.as_ref().expect("refit=true ⇒ Some(model)");

    let direct = cands[res.best_index]
        .fit(&pool)
        .expect("direct fit on the winner");

    let got = model.predict(&pool).expect("refit model predicts");
    let expected = direct.predict(&pool).expect("direct model predicts");
    assert_eq!(got.len(), expected.len());
    let mad = got
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("grid_search refit: max|diff| = {mad:e}");
    assert!(mad <= 1e-9, "refit predictions diverged: max|diff| = {mad:e}");
}

/// AT-S3b (errors): empty candidates / empty metrics are typed errors.
#[test]
fn grid_search_empty_inputs_err() {
    let pool = cv_pool();
    let cands = candidates();
    let folds = cv_folds();

    let empty: [CatBoostBuilder; 0] = [];
    assert!(
        grid_search(&pool, &empty, &["RMSE"], 3, false, 0, false, Some(&folds), false, ErrorScore::Raise).is_err(),
        "empty candidates must be a typed error"
    );
    assert!(
        grid_search(&pool, &cands, &[], 3, false, 0, false, Some(&folds), false, ErrorScore::Raise).is_err(),
        "empty metrics must be a typed error"
    );
}

/// AT-S4: `randomized_search` with `n_iter < M` evaluates exactly the sampled
/// subset, maps `best_index` back to the ORIGINAL slice, and is deterministic
/// across identical seeds; `n_iter >= M` matches `grid_search`; misuse is typed.
#[test]
fn randomized_search_determinism_and_mapping() {
    let pool = cv_pool();
    let cands = candidates();
    let folds = cv_folds();

    // n_iter=2 over 3 candidates: same seed ⇒ identical best_index + cv_results.
    let r1 = randomized_search(&pool, &cands, &["RMSE"], 2, 3, false, 0, false, Some(&folds), false, ErrorScore::Raise)
        .expect("randomized_search runs");
    let r2 = randomized_search(&pool, &cands, &["RMSE"], 2, 3, false, 0, false, Some(&folds), false, ErrorScore::Raise)
        .expect("randomized_search runs again");
    assert_eq!(r1.best_index, r2.best_index, "same seed ⇒ same best_index");
    assert_eq!(
        r1.cv_results.columns, r2.cv_results.columns,
        "same seed ⇒ identical cv_results"
    );
    assert!(
        r1.best_index < cands.len(),
        "best_index maps back into the ORIGINAL candidates slice"
    );
    assert_eq!(
        r1.best_builder,
        cands[r1.best_index],
        "best_builder is the ORIGINAL winning candidate"
    );

    // n_iter >= M ⇒ evaluates all candidates ⇒ same winner as grid_search.
    let full = randomized_search(&pool, &cands, &["RMSE"], 10, 3, false, 0, false, Some(&folds), false, ErrorScore::Raise)
        .expect("randomized_search full runs");
    let grid = grid_search(&pool, &cands, &["RMSE"], 3, false, 0, false, Some(&folds), false, ErrorScore::Raise)
        .expect("grid_search runs");
    assert_eq!(
        full.best_index, grid.best_index,
        "n_iter >= M is equivalent to grid_search"
    );

    // Misuse: n_iter == 0 / empty candidates are typed errors.
    assert!(
        randomized_search(&pool, &cands, &["RMSE"], 0, 3, false, 0, false, Some(&folds), false, ErrorScore::Raise)
            .is_err(),
        "n_iter == 0 must be a typed error"
    );
    let empty: [CatBoostBuilder; 0] = [];
    assert!(
        randomized_search(&pool, &empty, &["RMSE"], 2, 3, false, 0, false, Some(&folds), false, ErrorScore::Raise)
            .is_err(),
        "empty candidates must be a typed error"
    );
}

// -------------------------------------------------------------------------
// error_score — sklearn `error_score` semantics over a grid with one failure
// -------------------------------------------------------------------------

/// `error_score = Value(NaN)` (sklearn default): the degenerate candidate is
/// ranked WORST (never chosen over a finite-scored one), `grid_search` returns
/// `Ok`, `best_index` is one of the VALID depths, and `failures` records the
/// degenerate candidate's index.
#[test]
fn error_score_nan_ranks_failed_worst() {
    let pool = cv_pool();
    let cands = candidates_with_degenerate();
    let folds = cv_folds();

    let res = grid_search(
        &pool,
        &cands,
        &["RMSE"],
        3,
        false,
        0,
        false,
        Some(&folds),
        false,
        ErrorScore::Value(f64::NAN),
    )
    .expect("NaN error_score must NOT abort the search");

    assert!(
        res.best_index < DEGEN_INDEX,
        "a NaN-scored degenerate candidate must never win; best_index = {} (DEGEN = {DEGEN_INDEX})",
        res.best_index
    );
    assert!(
        res.failures.iter().any(|(i, _)| *i == DEGEN_INDEX),
        "failures must record the degenerate candidate's index; got {:?}",
        res.failures
    );
    // The winner still carries a real cv_results (non-empty column).
    let col = res
        .cv_results
        .columns
        .get("test-RMSE-mean")
        .expect("winner has a test-RMSE-mean column");
    assert!(!col.is_empty(), "winner's RMSE column must be non-empty");
}

/// `error_score = Raise`: the same grid ABORTS on the degenerate candidate.
#[test]
fn error_score_raise_aborts() {
    let pool = cv_pool();
    let cands = candidates_with_degenerate();
    let folds = cv_folds();

    let res = grid_search(
        &pool,
        &cands,
        &["RMSE"],
        3,
        false,
        0,
        false,
        Some(&folds),
        false,
        ErrorScore::Raise,
    );
    assert!(
        res.is_err(),
        "error_score='raise' must re-raise the degenerate candidate's failure"
    );
}

/// `error_score = Value(-1.0)` on the min-optimal RMSE: the assigned `-1.0` is
/// LOWER than any real RMSE, so the degenerate candidate WINS (a finite
/// error_score competes normally). It has a Some(empty) cv (iterations(0) ⇒ Ok
/// with an empty column), so the search returns `Ok` with `best_index` at the
/// degenerate index and `failures` recording it.
#[test]
fn error_score_numeric_competes() {
    let pool = cv_pool();
    let cands = candidates_with_degenerate();
    let folds = cv_folds();

    let res = grid_search(
        &pool,
        &cands,
        &["RMSE"],
        3,
        false,
        0,
        false,
        Some(&folds),
        false,
        ErrorScore::Value(-1.0),
    )
    .expect("iterations(0) yields Ok-with-empty-column ⇒ a finite error_score winner returns Ok");

    assert_eq!(
        res.best_index, DEGEN_INDEX,
        "a finite error_score of -1.0 beats every real RMSE ⇒ the degenerate candidate wins"
    );
    assert!(
        res.failures.iter().any(|(i, _)| *i == DEGEN_INDEX),
        "the degenerate candidate is still recorded as a failure even though it won: {:?}",
        res.failures
    );
}
