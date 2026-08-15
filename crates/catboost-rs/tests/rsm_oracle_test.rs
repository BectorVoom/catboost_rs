//! `rsm` (`colsample_bylevel`) parity oracle against catboost 1.2.10.
//!
//! `rsm` offers only a fraction of the features to the split search at each tree
//! LEVEL. What makes it delicate is not the filtering — it is that the keep/drop
//! decision comes from the SHARED persistent learn RNG, one `GenRandReal1()` per
//! listed candidate, ahead of the bootstrap and `random_strength` draws. A wrong
//! draw count or order still trains, still looks plausible, and silently
//! desynchronises every later tree.
//!
//! So these tests pin three separable things:
//!
//! 1. `rsm = 1.0` is INERT (the default fast path must not move);
//! 2. each subsampling fraction reproduces upstream's predictions;
//! 3. a tree ENDS EARLY when a level's candidate list comes back empty
//!    (upstream `break`s the depth loop rather than skipping the level), which is
//!    the rule most likely to be got wrong and is invisible in the prediction
//!    vector alone.
//!
//! Fixtures: `crates/cb-oracle/fixtures/rsm/`, written by
//! `crates/cb-oracle/generator/gen_rsm_fixtures.py` (which refuses to emit a
//! vacuous cell).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{
    CatBoostBuilder, EBootstrapType, EGrowPolicy, IngestSource, OwnedColumns, Pool,
};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("rsm")
        .join(rel)
}

fn load_x(name: &str) -> Vec<Vec<f64>> {
    let x: Array2<f64> = read_npy(fixture(name)).expect("fixture matrix");
    (0..x.ncols()).map(|f| x.column(f).to_vec()).collect()
}

fn load_y(name: &str) -> Vec<f64> {
    let y: Array1<f64> = read_npy(fixture(name)).expect("fixture vector");
    y.to_vec()
}

fn pool_of(cols: Vec<Vec<f64>>, target: Vec<f64>) -> Pool {
    OwnedColumns::new(cols, target).into_pool().expect("pool must build")
}

fn learn_pool() -> Pool {
    pool_of(load_x("X.npy"), load_y("y.npy"))
}

fn eval_pool() -> Pool {
    let cols = load_x("X_eval.npy");
    let n = cols.first().map_or(0, Vec::len);
    pool_of(cols, vec![0.0; n])
}

/// The pinned fit from `gen_rsm_fixtures.py::BASE`.
fn builder() -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(false)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
}

fn preds_with(b: CatBoostBuilder) -> Vec<f64> {
    let model = b.fit(&learn_pool()).expect("fit must succeed");
    model.predict(&eval_pool()).expect("predict must succeed")
}

fn assert_matches(actual: &[f64], fixture_name: &str, label: &str) {
    let expected = load_y(fixture_name);
    assert_eq!(actual.len(), expected.len(), "{label}: prediction count");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{label} row {i}: predicted {a} but catboost 1.2.10 says {e} (|diff| = {})",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// 1. The default is inert
// ===========================================================================

/// `rsm = 1.0` must be BIT-identical to never setting it. The engine gates the
/// whole per-level draw machinery on `rsm < 1.0`, so if this drifts, every
/// default fit in the repository has silently changed.
#[test]
fn rsm_one_is_bit_identical_to_the_default() {
    let unset = preds_with(builder());
    let one = preds_with(builder().rsm(1.0));
    assert_eq!(
        unset, one,
        "rsm=1.0 must be BIT-identical to the default (upstream measures max|diff| = 0); \
         a difference means the rsm_active gate is consuming draws it should not"
    );
}

/// The same inertness under a DRAWING sampler, where the shared RNG is genuinely
/// being consumed — the case where a spurious extra draw would actually show up.
#[test]
fn rsm_one_is_inert_under_a_drawing_bootstrap() {
    let unset = preds_with(builder().bootstrap_type(EBootstrapType::Bernoulli).subsample(0.7));
    let one = preds_with(
        builder().bootstrap_type(EBootstrapType::Bernoulli).subsample(0.7).rsm(1.0),
    );
    assert_eq!(
        unset, one,
        "rsm=1.0 under Bernoulli must be BIT-identical to the default"
    );
}

// ===========================================================================
// 2. Parity targets
// ===========================================================================

#[test]
fn rsm_0p75_matches_catboost() {
    assert_matches(&preds_with(builder().rsm(0.75)), "preds_rsm_0p75.npy", "rsm=0.75");
}

#[test]
fn rsm_0p5_matches_catboost() {
    assert_matches(&preds_with(builder().rsm(0.5)), "preds_rsm_0p5.npy", "rsm=0.5");
}

#[test]
fn rsm_0p25_matches_catboost() {
    assert_matches(&preds_with(builder().rsm(0.25)), "preds_rsm_0p25.npy", "rsm=0.25");
}

/// Each fraction must give a DIFFERENT model. Without this the parity tests
/// above could all be passing against the same numbers.
#[test]
fn distinct_rsm_values_give_distinct_models() {
    let a = preds_with(builder().rsm(0.75));
    let b = preds_with(builder().rsm(0.5));
    let c = preds_with(builder().rsm(0.25));
    assert_ne!(a, b, "rsm=0.75 and rsm=0.5 must differ");
    assert_ne!(b, c, "rsm=0.5 and rsm=0.25 must differ");
}

// ===========================================================================
// 3. Early stop — the rule that is invisible in a prediction vector
// ===========================================================================

/// At `rsm = 0.1` upstream's fixture records per-tree split counts
/// `[3, 0, 0, 1, 0]` for a depth-3 request: four of the five trees ran out of
/// candidates and STOPPED, two of them before making any split at all. The
/// engine must reproduce that, not silently grow a full-depth tree from whatever
/// features happened to survive.
#[test]
fn a_level_with_no_selected_feature_ends_the_tree() {
    let model = builder().rsm(0.1).fit(&learn_pool()).expect("fit must succeed");
    let actual: Vec<usize> = model
        .as_canonical()
        .oblivious_trees
        .iter()
        .map(|t| t.splits.len())
        .collect();
    assert_eq!(
        actual,
        vec![3, 0, 0, 1, 0],
        "per-tree split counts must match catboost 1.2.10's early-stop behaviour \
         (greedy_tensor_search.cpp:1209 breaks the depth loop when the candidate list \
         is empty); got {actual:?}"
    );
}

/// The predictions from that same early-stopping fit, so the short trees are
/// checked numerically and not only structurally.
#[test]
fn early_stopping_fit_matches_catboost() {
    assert_matches(
        &preds_with(builder().rsm(0.1)),
        "preds_rsm_early_stop.npy",
        "rsm=0.1 (early stop)",
    );
}

/// The contrast: at `rsm = 1.0` every tree grows to the full depth. This is what
/// makes the assertion above a real detector rather than a coincidence.
#[test]
fn rsm_one_grows_every_tree_to_full_depth() {
    let model = builder().rsm(1.0).fit(&learn_pool()).expect("fit must succeed");
    for (i, tree) in model.as_canonical().oblivious_trees.iter().enumerate() {
        assert_eq!(tree.splits.len(), 3, "tree {i} must reach the requested depth");
    }
}

// ===========================================================================
// 4. Composition with the other consumers of the same RNG stream
// ===========================================================================

#[test]
fn rsm_composes_with_bernoulli_bootstrap() {
    assert_matches(
        &preds_with(
            builder().bootstrap_type(EBootstrapType::Bernoulli).subsample(0.7).rsm(0.5),
        ),
        "preds_rsm_0p5_bernoulli.npy",
        "rsm=0.5 + Bernoulli",
    );
}

#[test]
fn rsm_composes_with_bayesian_bootstrap() {
    assert_matches(
        &preds_with(
            builder()
                .bootstrap_type(EBootstrapType::Bayesian)
                .bagging_temperature(1.0)
                .rsm(0.5),
        ),
        "preds_rsm_0p5_bayesian.npy",
        "rsm=0.5 + Bayesian",
    );
}

#[test]
fn rsm_composes_with_random_strength() {
    assert_matches(
        &preds_with(builder().random_strength(1.0).rsm(0.5)),
        "preds_rsm_0p5_random_strength.npy",
        "rsm=0.5 + random_strength=1",
    );
}

// ===========================================================================
// 5. Refusals
// ===========================================================================

/// Upstream's range is `(0, 1]` (`oblivious_tree_options.cpp:125`). `0` is
/// excluded because it would select nothing at any level.
#[test]
fn out_of_range_rsm_is_refused() {
    for bad in [0.0, -0.1, 1.5, f64::NAN] {
        let err = builder().rsm(bad).fit(&learn_pool()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rsm should be in (0, 1]"),
            "rsm={bad} must be refused with upstream's range message, got: {msg}"
        );
    }
}

/// The leaf-wise / region growers call `SelectFeaturesForScoring` from their own
/// loop, so their draw accounting is different and unverified. A non-default
/// `rsm` there is REFUSED, never silently ignored.
#[test]
fn rsm_below_one_is_refused_for_non_symmetric_grow_policies() {
    for policy in [EGrowPolicy::Depthwise, EGrowPolicy::Lossguide, EGrowPolicy::Region] {
        let err = builder()
            .grow_policy(policy)
            .rsm(0.5)
            .fit(&learn_pool())
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rsm < 1 is only implemented for grow_policy=SymmetricTree"),
            "{policy:?} + rsm=0.5 must be refused, got: {msg}"
        );
    }
}

/// ...but the DEFAULT `rsm` must still train under every grow policy — the
/// refusal above must not have made `rsm` a de-facto SymmetricTree-only gate.
#[test]
fn default_rsm_still_trains_under_every_grow_policy() {
    for policy in [EGrowPolicy::Depthwise, EGrowPolicy::Lossguide, EGrowPolicy::Region] {
        builder()
            .grow_policy(policy)
            .rsm(1.0)
            .fit(&learn_pool())
            .unwrap_or_else(|e| panic!("{policy:?} + rsm=1.0 must train: {e}"));
    }
}
