//! F10–F13 (SPEC-CATF-10/11/12, SPEC-CATF-Δ4/Δ7) — predict-side categorical
//! routing, and the silent-wrongness paths it closes.
//!
//! The class of defect under test: `predict_raw(m, fv)` is literally
//! `predict_raw_cat(m, fv, &[])` (`crates/cb-model/src/apply.rs`), so scoring a
//! categorical model through the float-only entry makes `cat_values.get(i)`
//! return `None` for every object. A CTR split then reads a missing table and a
//! ONE-HOT split evaluates to `false` — **every** categorical level fails, and
//! the caller receives plausible numbers with no error at all.

use std::path::{Path, PathBuf};

use catboost_rs::{
    CatBoostBuilder, CatBoostError, ECtrType, IngestSource, Loss, Model, OwnedColumns, Pool,
};
use cb_data::stringify_int_category;
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cb-oracle/fixtures")
        .join(rel)
}

/// The frozen `tensor_ctr_e2e` corpus as a 2-cat-column pool.
fn ctr_pool() -> Pool {
    let cats: Array2<i32> =
        read_npy(fixture("tensor_ctr_e2e/X_cat.npy")).expect("X_cat.npy must load as int32 [N,2]");
    let y = cb_oracle::load_f64_vec(&fixture("tensor_ctr_e2e/y.npy")).expect("y.npy must load");
    let cat_columns: Vec<Vec<String>> = (0..cats.ncols())
        .map(|c| {
            cats.column(c)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect();
    OwnedColumns::new(vec![vec![0.0_f64; y.len()]], y)
        .with_cat_features(cat_columns)
        .into_pool()
        .expect("ctr pool builds")
}

/// The SAME rows and float columns, with the categorical columns REMOVED — the
/// pool a caller accidentally hands to `predict` after fitting on the real one.
fn ctr_pool_without_cat_columns() -> Pool {
    let y = cb_oracle::load_f64_vec(&fixture("tensor_ctr_e2e/y.npy")).expect("y.npy must load");
    OwnedColumns::new(vec![vec![0.0_f64; y.len()]], y)
        .into_pool()
        .expect("cat-free pool builds")
}

fn fit_ctr_model() -> Model {
    CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(5)
        .depth(2)
        .learning_rate(0.1)
        .boost_from_average(false)
        .random_strength(0.0)
        .one_hot_max_size(1)
        .max_ctr_complexity(1)
        .simple_ctr(ECtrType::Borders)
        .simple_ctr_priors(vec![0.5])
        .fit(&ctr_pool())
        .expect("ctr fit must succeed")
}

/// A pool whose single cat column has cardinality 2 (routes to ONE-HOT), plus a
/// float column so the float-only path is not degenerate.
fn one_hot_pool() -> Pool {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let cat: Vec<String> = (0..n)
        .map(|i| if i % 2 == 0 { "alpha" } else { "beta" }.to_owned())
        .collect();
    let floats = vec![(0..n).map(|i| (i % 7) as f64).collect::<Vec<f64>>()];
    OwnedColumns::new(floats, label)
        .with_cat_features(vec![cat])
        .into_pool()
        .expect("one-hot pool builds")
}

fn one_hot_pool_without_cat_columns() -> Pool {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let floats = vec![(0..n).map(|i| (i % 7) as f64).collect::<Vec<f64>>()];
    OwnedColumns::new(floats, label)
        .into_pool()
        .expect("cat-free one-hot pool builds")
}

fn fit_one_hot_model() -> Model {
    CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(3)
        .depth(3)
        .learning_rate(0.1)
        .boost_from_average(false)
        .random_strength(0.0)
        .one_hot_max_size(2)
        .max_ctr_complexity(1)
        .fit(&one_hot_pool())
        .expect("one-hot fit must succeed")
}

// ---------------------------------------------------------------------------
// F10 — width validation against the DECLARED width
// ---------------------------------------------------------------------------

/// F10 — THE CRITICAL-3 COUNTEREXAMPLE, as a POSITIVE test.
///
/// `fit(pool) -> predict(the SAME pool)` must be `Ok`. T09's original design
/// compared the pool width against `max(projection member) + 1` — a width
/// DERIVED from the chosen splits. Here the second categorical column is
/// constant, so no split ever references it and the derived width is 1 while
/// the declared width is 2; the derived form would REJECT this fit→predict.
#[test]
fn fit_then_predict_the_same_pool_is_ok_even_when_a_cat_column_is_never_split_on() {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let informative: Vec<String> = (0..n).map(|i| format!("c{}", i % 9)).collect();
    let never_split_on: Vec<String> = vec!["same".to_owned(); n];
    let pool = OwnedColumns::new(vec![vec![0.0_f64; n]], label)
        .with_cat_features(vec![informative, never_split_on])
        .into_pool()
        .expect("pool builds");

    let model = CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(3)
        .depth(2)
        .learning_rate(0.1)
        .boost_from_average(false)
        .random_strength(0.0)
        .one_hot_max_size(1)
        .max_ctr_complexity(1)
        .fit(&pool)
        .expect("fit must succeed");

    let preds = model.predict(&pool).unwrap_or_else(|e| {
        panic!(
            "fit(pool) -> predict(SAME pool) must be Ok; a width derived from the \
             chosen splits would wrongly reject it: {e:?}"
        )
    });
    assert_eq!(preds.len(), n);
}

/// F10 — a pool with the WRONG declared cat width is a typed error, not a
/// silently-wrong score.
#[test]
fn predict_with_a_narrower_cat_width_is_a_typed_feature_mismatch() {
    let model = fit_ctr_model();
    let y = cb_oracle::load_f64_vec(&fixture("tensor_ctr_e2e/y.npy")).expect("y.npy must load");
    let cats: Array2<i32> =
        read_npy(fixture("tensor_ctr_e2e/X_cat.npy")).expect("X_cat.npy must load");
    // ONE cat column where the model was trained on two.
    let one_column: Vec<String> = cats
        .column(0)
        .iter()
        .map(|&code| stringify_int_category(i64::from(code)))
        .collect();
    let narrow = OwnedColumns::new(vec![vec![0.0_f64; y.len()]], y)
        .with_cat_features(vec![one_column])
        .into_pool()
        .expect("narrow pool builds");

    match model.predict(&narrow) {
        Err(CatBoostError::FeatureMismatch(msg)) => {
            assert!(
                msg.contains("categorical"),
                "the error must name the categorical width, got: {msg}"
            );
        }
        Err(other) => panic!("expected FeatureMismatch, got {other:?}"),
        Ok(_) => panic!(
            "predict SILENTLY SCORED a 2-cat-column model against a 1-cat-column pool"
        ),
    }
}

// ---------------------------------------------------------------------------
// F11 — predict routes through the categorical apply path
// ---------------------------------------------------------------------------

/// F11 — the facade's `predict` must equal a DIRECT `predict_raw_cat` call on
/// the same columns, to 1e-12, with a non-degeneracy guard so an all-identical
/// prediction vector cannot pass vacuously.
#[test]
fn predict_matches_a_direct_predict_raw_cat_call() {
    let model = fit_ctr_model();
    let pool = ctr_pool();

    let facade = model.predict(&pool).expect("facade predict must succeed");

    let columns: Vec<Vec<f32>> = pool
        .float_features()
        .iter()
        .map(|c| c.iter().map(|&v| v as f32).collect())
        .collect();
    let direct = cb_model::predict_raw_cat(model.as_canonical(), &columns, pool.cat_features());

    assert_eq!(facade.len(), direct.len());
    for (i, (&f, &d)) in facade.iter().zip(direct.iter()).enumerate() {
        assert!(
            (f - d).abs() < 1e-12,
            "object {i}: facade {f} vs direct predict_raw_cat {d}"
        );
    }
    // Non-degeneracy: if every prediction were identical, dropping the cat
    // columns would also "match" and this test would prove nothing.
    let min = direct.iter().copied().fold(f64::INFINITY, f64::min);
    let max = direct.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max - min > 1e-6,
        "predictions are degenerate (spread {}), so the comparison is vacuous",
        max - min
    );
}

/// F11 (Δ7) — `ensure_scalar_oblivious` must reject a ONE-HOT model.
///
/// A one-hot model has `ctr_data == None`, so it passed every existing arm of
/// that guard; `predict_raw_staged` is float-only and never reads cat columns,
/// so `staged_predict` scored it as though EVERY one-hot split failed.
#[test]
fn staged_predict_rejects_a_one_hot_model_instead_of_silently_scoring_it() {
    let model = fit_one_hot_model();
    match model.staged_predict(&one_hot_pool(), None, None, None) {
        Err(CatBoostError::UnsupportedModel(msg)) => {
            assert!(
                msg.contains("one-hot") || msg.contains("categorical"),
                "the error must name the one-hot limitation, got: {msg}"
            );
        }
        Err(other) => panic!("expected UnsupportedModel, got {other:?}"),
        Ok(_) => panic!(
            "staged_predict SILENTLY SCORED a one-hot model as though every one-hot split failed"
        ),
    }
}

// ---------------------------------------------------------------------------
// F12 — the zero-cat-column guard on all FOUR entrypoints
// ---------------------------------------------------------------------------

/// F12 — `predict` on a CTR model with a cat-FREE pool is a typed error.
#[test]
fn predict_on_a_ctr_model_with_no_cat_columns_is_a_typed_error() {
    let model = fit_ctr_model();
    match model.predict(&ctr_pool_without_cat_columns()) {
        Err(CatBoostError::FeatureMismatch(msg)) => assert!(msg.contains("categorical")),
        Err(other) => panic!("expected FeatureMismatch, got {other:?}"),
        Ok(_) => panic!("predict SILENTLY SCORED a CTR model against a cat-free pool"),
    }
}

/// F12 — `predict_with` too.
#[test]
fn predict_with_on_a_ctr_model_with_no_cat_columns_is_a_typed_error() {
    let model = fit_ctr_model();
    match model.predict_with(
        &ctr_pool_without_cat_columns(),
        cb_model::PredictionType::RawFormulaVal,
    ) {
        Err(CatBoostError::FeatureMismatch(msg)) => assert!(msg.contains("categorical")),
        Err(other) => panic!("expected FeatureMismatch, got {other:?}"),
        Ok(_) => panic!("predict_with SILENTLY SCORED a CTR model against a cat-free pool"),
    }
}

/// F12 — `predict_proba` too.
#[test]
fn predict_proba_on_a_ctr_model_with_no_cat_columns_is_a_typed_error() {
    let model = fit_ctr_model();
    match model.predict_proba(&ctr_pool_without_cat_columns()) {
        Err(CatBoostError::FeatureMismatch(msg)) => assert!(msg.contains("categorical")),
        Err(other) => panic!("expected FeatureMismatch, got {other:?}"),
        Ok(_) => panic!("predict_proba SILENTLY SCORED a CTR model against a cat-free pool"),
    }
}

/// F12 (Δ7) — a ONE-HOT model reaches the guard too, even though it has no
/// `ctr_data`: `needs_cat_columns()` is the predicate, not `ctr_data.is_some()`.
#[test]
fn predict_on_a_one_hot_model_with_no_cat_columns_is_a_typed_error() {
    let model = fit_one_hot_model();
    match model.predict(&one_hot_pool_without_cat_columns()) {
        Err(CatBoostError::FeatureMismatch(msg)) => assert!(msg.contains("categorical")),
        Err(other) => panic!("expected FeatureMismatch, got {other:?}"),
        Ok(_) => panic!(
            "predict SILENTLY SCORED a one-hot model against a cat-free pool — every \
             one-hot split evaluated false"
        ),
    }
}

// ---------------------------------------------------------------------------
// F13 — the remaining fstr / PDP paths
// ---------------------------------------------------------------------------

/// F13 — `partial_dependence` is defined over float features only, so a model
/// that needs cat columns must be REFUSED rather than swept with every
/// categorical level failing.
#[test]
fn partial_dependence_rejects_a_model_that_needs_cat_columns() {
    for model in [fit_ctr_model(), fit_one_hot_model()] {
        let pool = if model.as_canonical().ctr_data.is_some() {
            ctr_pool()
        } else {
            one_hot_pool()
        };
        match model.partial_dependence(&pool, &[0]) {
            Err(CatBoostError::UnsupportedModel(msg)) => {
                assert!(
                    msg.contains("partial_dependence"),
                    "the error must name the surface, got: {msg}"
                );
            }
            Err(other) => panic!("expected UnsupportedModel, got {other:?}"),
            Ok(_) => panic!(
                "partial_dependence SILENTLY SWEPT a categorical model over float \
                 features while every categorical split failed"
            ),
        }
    }
}

/// F13 — `feature_importance_with_data(PredictionValuesChange)` passes the
/// pool's cat columns straight through with NO width check, so a mismatched
/// pool produced silently-wrong importances.
#[test]
fn feature_importance_with_data_checks_the_cat_width() {
    let model = fit_ctr_model();
    match model.feature_importance_with_data(
        cb_model::FeatureImportanceType::PredictionValuesChange,
        &ctr_pool_without_cat_columns(),
        "RMSE",
    ) {
        Err(CatBoostError::FeatureMismatch(msg)) => assert!(msg.contains("categorical")),
        Err(other) => panic!("expected FeatureMismatch, got {other:?}"),
        Ok(_) => panic!(
            "feature_importance_with_data SILENTLY COMPUTED importances for a CTR \
             model against a cat-free pool"
        ),
    }
}
