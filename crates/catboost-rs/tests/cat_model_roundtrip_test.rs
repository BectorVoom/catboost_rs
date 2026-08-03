//! Regression tests for the categorical-model plumbing the fit-param surface
//! made reachable end to end: `.cbm` save -> load -> predict, `sum_models` /
//! `save_json` refusing what they cannot represent, and `ignored_features`
//! naming a CATEGORICAL column.
//!
//! The class of defect under test: state that exists at FIT time but is lost at
//! a serialization or composition boundary. Each of these produced a *silent*
//! wrong answer or an inexplicable error on the very pool the model was fit on,
//! rather than a typed failure at the boundary that lost the state.

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, CatBoostError, IngestSource, Loss, Model, OwnedColumns, Pool};

/// A scratch path for a save/load round-trip, unique per test name.
fn scratch(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("catboost_rs_cat_roundtrip_{name}"));
    p
}

/// A pool whose single categorical column has cardinality 2, so at
/// `one_hot_max_size(2)` it routes ONE-HOT (and bakes no CTR table). A second,
/// CONSTANT categorical column is declared but never selected, so the model's
/// DECLARED width (2) exceeds any width derivable from its splits (1) — the
/// exact gap between "what the pool declared" and "what the decoder can
/// recover".
fn one_hot_pool() -> Pool {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let informative: Vec<String> = (0..n)
        .map(|i| if i % 2 == 0 { "alpha" } else { "beta" }.to_owned())
        .collect();
    let constant: Vec<String> = vec!["same".to_owned(); n];
    let floats = vec![(0..n).map(|i| (i % 7) as f64).collect::<Vec<f64>>()];
    OwnedColumns::new(floats, label)
        .with_cat_features(vec![informative, constant])
        .into_pool()
        .expect("one-hot pool builds")
}

fn one_hot_model(pool: &Pool) -> Model {
    CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(4)
        .depth(3)
        .learning_rate(0.1)
        .boost_from_average(false)
        .random_strength(0.0)
        .one_hot_max_size(2)
        .max_ctr_complexity(1)
        .fit(pool)
        .expect("one-hot fit must succeed")
}

// ── Finding 1: `.cbm` decode hard-coded `cat_feature_count: 0` ───────────────

/// A categorical model must survive `save_cbm` -> `load_cbm` -> `predict` on the
/// pool it was fit on, and predict IDENTICALLY.
///
/// Before the fix, `reconstruct_model` hard-coded `cat_feature_count: 0` on both
/// return paths. The reloaded model still carried its one-hot splits, so
/// `needs_cat_columns()` was true, but it claimed to expect ZERO categorical
/// columns — so `Model::cat_columns` rejected the very pool the model was fit
/// on with "pool has 2 categorical feature(s), model expects 0". Every
/// save/load round-trip of a categorical model, and every upstream categorical
/// `.cbm`, was unusable through the facade.
#[test]
fn a_categorical_model_survives_a_cbm_round_trip_and_predicts_identically() {
    let pool = one_hot_pool();
    let model = one_hot_model(&pool);
    let before = model.predict(&pool).expect("the fresh model must predict");

    let path = scratch("cbm_predict.cbm");
    model.save_cbm(&path).expect("save_cbm must succeed");
    let reloaded = Model::load_cbm(&path).expect("load_cbm must succeed");
    let _ = std::fs::remove_file(&path);

    assert!(
        reloaded.as_canonical().cat_feature_count() >= 1,
        "a decoded model carrying categorical splits must recover a NON-ZERO \
         categorical width — 0 makes it claim to need cat columns while expecting \
         none, which no pool can satisfy"
    );

    let after = reloaded
        .predict(&pool)
        .expect("a reloaded categorical model must predict on the pool it was fit on");
    assert_eq!(
        before, after,
        "a .cbm round-trip must not change any prediction"
    );
}

/// The width check is a MINIMUM, not an equality: a decoded model's recovered
/// width is a lower bound (`cat_feature_count` is deliberately not written to
/// the `.cbm` bytes), so requiring equality rejected legitimate round-trips.
/// Supplying FEWER categorical columns than the model can reference must still
/// be a typed error — that is the silent-failure case the guard exists for.
#[test]
fn scoring_a_categorical_model_with_no_cat_columns_is_still_rejected() {
    let pool = one_hot_pool();
    let model = one_hot_model(&pool);

    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let floats = vec![(0..n).map(|i| (i % 7) as f64).collect::<Vec<f64>>()];
    let cat_free = OwnedColumns::new(floats, label)
        .into_pool()
        .expect("cat-free pool builds");

    let err = model
        .predict(&cat_free)
        .expect_err("a categorical model scored without cat columns must be rejected");
    assert!(
        matches!(err, CatBoostError::FeatureMismatch(_)),
        "expected a typed FeatureMismatch, got: {err}"
    );
}

// ── Finding 3: `save_json` silently dropped CTR splits ──────────────────────

/// A CTR-routed pool: the categorical column's cardinality far exceeds
/// `one_hot_max_size(1)`, so it takes the CTR path and the model carries
/// `ModelSplit::Ctr` splits (which `to_doc` used to DROP) rather than
/// `ModelSplit::OneHot` splits (which it already rejected).
fn ctr_pool() -> Pool {
    let n = 60_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let high_cardinality: Vec<String> = (0..n).map(|i| format!("c{}", i % 11)).collect();
    let floats = vec![vec![0.0_f64; n]];
    OwnedColumns::new(floats, label)
        .with_cat_features(vec![high_cardinality])
        .into_pool()
        .expect("CTR pool builds")
}

/// `save_json` must REFUSE a model carrying CTR splits rather than emitting a
/// document whose split count disagrees with its leaf count.
///
/// One-hot was already rejected; CTR was not. The oblivious arm used to
/// `filter_map(as_float)`, so a depth-`d` tree with a CTR level emitted `d - 1`
/// splits next to its full `2^d` leaf values. On reload, `leaf_index_for` could
/// only reach the lower half of the leaves, so the round-tripped model predicted
/// differently from the original — with no error at either end. `from_doc`
/// hard-codes `ctr_data: None` anyway, so a CTR model could never have
/// round-tripped even if the counts had lined up.
#[test]
fn save_json_refuses_a_ctr_model_instead_of_dropping_its_splits() {
    let pool = ctr_pool();
    let model = CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(4)
        .depth(3)
        .learning_rate(0.1)
        .boost_from_average(false)
        .random_strength(0.0)
        .one_hot_max_size(1)
        .max_ctr_complexity(1)
        .fit(&pool)
        .expect("CTR fit must succeed");

    let n_ctr = model
        .as_canonical()
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter(|s| matches!(s, cb_model::ModelSplit::Ctr(_)))
        .count();
    assert!(
        n_ctr >= 1,
        "the fixture must actually produce CTR splits, or this test cannot detect \
         them being dropped"
    );

    let path = scratch("reject_ctr.json");
    let err = model
        .save_json(&path)
        .expect_err("the numeric json schema cannot represent CTR splits");
    let _ = std::fs::remove_file(&path);
    let msg = err.to_string();
    assert!(
        msg.contains("CTR"),
        "the error must name the split kind it cannot represent, got: {msg}"
    );
    assert!(
        msg.contains(".cbm"),
        "the error must point at the format that CAN carry it, got: {msg}"
    );
}

/// The pre-existing one-hot refusal must stay intact (it is the same guard now).
#[test]
fn save_json_still_refuses_a_one_hot_model() {
    let pool = one_hot_pool();
    let model = one_hot_model(&pool);
    let path = scratch("reject_one_hot.json");
    let err = model
        .save_json(&path)
        .expect_err("the numeric json schema cannot represent one-hot splits");
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("one-hot"),
        "the error must still name one-hot, got: {err}"
    );
}

// ── Finding 8: `ignored_features` naming a categorical column ───────────────

/// `ignored_features` indexes upstream's FLAT (float-then-categorical) space, so
/// an index naming a categorical column must IGNORE that column — not abort the
/// fit as out-of-range.
///
/// The pool below has 1 float + 2 categorical columns, so flat index 1 is the
/// first categorical column. Before the fix, `apply_ignored_features` indexed
/// the float-border vector alone and rejected anything `>= n_float`, so
/// `ignored_features=[1]` failed with "out of range for a pool with 1 float
/// feature(s)" — and there was no way to ignore a categorical feature at all.
#[test]
fn ignored_features_accepts_a_categorical_flat_index_and_drops_that_column() {
    let pool = one_hot_pool();

    let baseline = one_hot_model(&pool);
    let baseline_one_hot = baseline
        .as_canonical()
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter(|s| matches!(s, cb_model::ModelSplit::OneHot(_)))
        .count();
    assert!(
        baseline_one_hot >= 1,
        "the baseline fit must select the informative categorical column, or this \
         test cannot detect it being ignored"
    );

    // Flat index 1 == categorical column 0 (1 float column precedes it).
    let ignored = CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(4)
        .depth(3)
        .learning_rate(0.1)
        .boost_from_average(false)
        .random_strength(0.0)
        .one_hot_max_size(2)
        .max_ctr_complexity(1)
        .ignored_features(vec![1])
        .fit(&pool)
        .expect("ignoring a CATEGORICAL flat index must be accepted, not rejected");

    let ignored_one_hot = ignored
        .as_canonical()
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter(|s| matches!(s, cb_model::ModelSplit::OneHot(_)))
        .count();
    assert_eq!(
        ignored_one_hot, 0,
        "an ignored categorical column must contribute no split at any level"
    );

    // The column keeps its INDEX — predict still takes the full-width pool.
    assert!(
        ignored.predict(&pool).is_ok(),
        "ignoring a feature must not change the pool width predict accepts"
    );
}

/// An index past the END of the flat space is still rejected — a typo that
/// silently ignores nothing is the failure this parameter is used to prevent.
#[test]
fn ignored_features_still_rejects_an_index_past_the_flat_width() {
    let pool = one_hot_pool();
    // 1 float + 2 cat = flat indices 0..3; 3 is out of range.
    let err = CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(2)
        .depth(2)
        .one_hot_max_size(2)
        .ignored_features(vec![3])
        .fit(&pool)
        .expect_err("an index past the flat width must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("out of range") && msg.contains("categorical"),
        "the error must report the FLAT width (both kinds), got: {msg}"
    );
}
