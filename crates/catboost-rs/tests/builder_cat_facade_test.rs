//! F09 (SPEC-CATF-08 / SPEC-CATF-Δ1 / Δ4) — `CatBoostBuilder::fit` routes a
//! CATEGORICAL pool through `train_cat` and bakes its CTR tables.
//!
//! Before F09, `fit()` called the float-only `train` and `pool.cat_features()`
//! was never read, so every categorical column was SILENTLY DROPPED: the fit
//! succeeded, returned a model with no CTR and no one-hot splits, and predicted
//! as though the column did not exist.

use std::path::{Path, PathBuf};

use catboost_rs::{CatBoostBuilder, ECtrType, IngestSource, Loss, OwnedColumns, Pool};
use cb_data::stringify_int_category;
use cb_model::ModelSplit;
use ndarray::Array2;
use ndarray_npy::read_npy;

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cb-oracle/fixtures")
        .join(rel)
}

/// The frozen `tensor_ctr_e2e` corpus: `X_cat.npy` `[30, 2] int32`,
/// `y.npy` `[30] float64`. Categorical values reach the pool as the A4
/// stringified form `calc_cat_feature_hash` hashes.
fn tensor_ctr_pool() -> Pool {
    let cats: Array2<i32> =
        read_npy(fixture("tensor_ctr_e2e/X_cat.npy")).expect("X_cat.npy must load as int32 [N,2]");
    let y = cb_oracle::load_f64_vec(&fixture("tensor_ctr_e2e/y.npy")).expect("y.npy must load");

    // A4: the PLAIN decimal stringification is what `calc_cat_feature_hash`
    // hashes, so the pool must carry that exact form.
    let cat_columns: Vec<Vec<String>> = (0..cats.ncols())
        .map(|c| {
            cats.column(c)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect();

    // One constant float column: the pool needs a float feature to quantize,
    // and a constant one contributes no float split, so every split the model
    // chooses must come from the categorical side.
    let float_features = vec![vec![0.0_f64; y.len()]];
    OwnedColumns::new(float_features, y)
        .with_cat_features(cat_columns)
        .into_pool()
        .expect("tensor_ctr_e2e pool builds")
}

/// A pool whose single categorical column has cardinality 2, so at
/// `one_hot_max_size(2)` it routes to the ONE-HOT path, not the CTR path.
fn cardinality_two_pool() -> Pool {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let cat: Vec<String> = (0..n)
        .map(|i| if i % 2 == 0 { "alpha" } else { "beta" }.to_owned())
        .collect();
    let floats = vec![(0..n).map(|i| (i % 7) as f64).collect::<Vec<f64>>()];
    OwnedColumns::new(floats, label)
        .with_cat_features(vec![cat])
        .into_pool()
        .expect("cardinality-2 pool builds")
}

/// F09 test fn 1 — a categorical pool bakes CTR data and emits a CTR split.
#[test]
fn fit_categorical_pool_bakes_ctr_data_and_emits_a_ctr_split() {
    let pool = tensor_ctr_pool();
    let model = CatBoostBuilder::new()
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
        .fit(&pool)
        .expect("categorical fit must succeed");

    let inner = model.as_canonical();
    assert!(
        inner.ctr_data.is_some(),
        "fit() on a categorical pool must bake the CTR tables — a None here means \
         fit() still calls the float-only train and SILENTLY DROPPED the cat columns"
    );
    let n_ctr_splits: usize = inner
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter(|s| matches!(s, ModelSplit::Ctr(_)))
        .count();
    assert!(
        n_ctr_splits >= 1,
        "a high-cardinality categorical pool must contribute at least one CTR split"
    );
}

/// F09 test fn 2 (Δ1) — the ONE-HOT arm of `fit()` is live.
///
/// This is the SOLE home of the cardinality-2 one-hot-contribution assertion:
/// F01's duplicate was deleted because it could not pass before `fit()` routed.
/// It proves PLAN-CHECK's CRITICAL-4 hazard ("one-hot columns silently dropped")
/// is genuinely closed rather than merely re-planned.
#[test]
fn fit_low_cardinality_pool_emits_one_hot_splits_not_ctr() {
    let pool = cardinality_two_pool();
    let model = CatBoostBuilder::new()
        .loss(Loss::Logloss)
        .iterations(3)
        .depth(3)
        .learning_rate(0.1)
        .boost_from_average(false)
        .random_strength(0.0)
        .one_hot_max_size(2)
        .max_ctr_complexity(1)
        .fit(&pool)
        .expect("one-hot fit must succeed");

    let inner = model.as_canonical();
    let has_one_hot_split = inner
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .any(|s| matches!(s, ModelSplit::OneHot(_)));
    assert!(
        has_one_hot_split,
        "a cardinality-2 categorical column at one_hot_max_size(2) must contribute \
         a ModelSplit::OneHot"
    );
    assert!(
        inner.ctr_data.is_none(),
        "a one-hot-routed column bakes no CTR table"
    );
}

/// F09 test fn 3 (Δ4) — the model records the pool's DECLARED cat width.
///
/// The declared width, never one derived from the chosen splits: this pool's
/// second categorical column is constant, so it is never selected, and a derived
/// width would report 1 where the truth is 2 (PLAN-CHECK CRITICAL-3).
#[test]
fn fit_categorical_pool_records_the_declared_cat_width() {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let informative: Vec<String> = (0..n).map(|i| format!("c{}", i % 9)).collect();
    let constant: Vec<String> = vec!["same".to_owned(); n];
    let pool = OwnedColumns::new(vec![vec![0.0_f64; n]], label)
        .with_cat_features(vec![informative, constant])
        .into_pool()
        .expect("two-cat-column pool builds");

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

    assert_eq!(
        model.as_canonical().cat_feature_count(),
        2,
        "the model must record the pool's DECLARED cat width (2), not a width \
         derived from the splits it happened to choose"
    );
}

/// F21 (SPEC-CATF-07, acceptance A1) — a CAT-FREE pool is untouched by the F09
/// branch.
///
/// The cat-free arm is byte-for-byte the pre-F09 call (same function, same
/// arguments, same order), so a numeric fit must be bit-identical.
///
/// **On the mandated mutation.** Inverting the branch predicate
/// (`pool.cat_features().is_empty()` -> `!...`) IS detected — by all three
/// tests above, from the CATEGORICAL side: a categorical pool routed to the
/// float-only `train` cannot find a split at all. The cat-FREE side of the
/// inversion is, by contrast, provably unobservable: `train_cat(rt, fv, fb,
/// &[], ..)` delegates to the same `train_inner` with an EMPTY `cat_columns`
/// slice, so it computes exactly what `train` computes. That is not a gap in
/// the test — it IS the no-regression property this task asserts, and it is
/// why `builder_oracle_test` stays green under the mutation.
#[test]
fn a_cat_free_pool_produces_a_float_only_model() {
    let n = 40_usize;
    let label: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let floats = vec![
        (0..n).map(|i| (i % 7) as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| (i % 3) as f64).collect::<Vec<f64>>(),
    ];
    let pool = OwnedColumns::new(floats, label)
        .into_pool()
        .expect("float-only pool builds");

    let model = CatBoostBuilder::new()
        .iterations(5)
        .depth(3)
        .learning_rate(0.1)
        .fit(&pool)
        .expect("float-only fit must succeed");

    let inner = model.as_canonical();
    assert!(
        inner.ctr_data.is_none(),
        "a cat-free pool must bake no CTR table"
    );
    assert_eq!(
        inner.cat_feature_count(),
        0,
        "a cat-free pool declares zero categorical columns"
    );
    let categorical_splits = inner
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter())
        .filter(|s| !matches!(s, ModelSplit::Float(_)))
        .count();
    assert_eq!(
        categorical_splits, 0,
        "a cat-free pool must produce a purely float-split model"
    );
    // And it still scores: the F10 width check must not fire on a float model.
    assert_eq!(model.predict(&pool).expect("predict must succeed").len(), n);
}
