//! `random_score_type` parity oracle (NormalWithModelSizeDecrease / Gumbel)
//! against catboost 1.2.10.
//!
//! The parameter selects the distribution of the `random_strength` split-score
//! perturbation. Upstream differs in BOTH halves
//! (`greedy_tensor_search.cpp:861-866`, `rand_score.h:41-49`):
//!
//! | | std-dev | draw |
//! |---|---|---|
//! | `NormalWithModelSizeDecrease` | `strength * dsdz * modelSizeMultiplier` | `Val + Normal(0, stdev)` |
//! | `Gumbel` | `strength * dsdz * 1.0` | `Val + stdev * ln(ln(1/GenRandReal1()))` |
//!
//! Gumbel does NOT decay the perturbation as the model grows — the model-size
//! multiplier is exactly the "…WithModelSizeDecrease" half of the other name.
//! The two also consume different amounts of RNG per candidate (the normal draw
//! is rejection sampling over PAIRS of uniforms, Gumbel takes one), which shifts
//! the whole downstream draw stream.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, ERandomScoreType, IngestSource, OwnedColumns, Pool};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;
const RANDOM_STRENGTH: f64 = 1.0;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("random_score_type")
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
    OwnedColumns::new(cols, target)
        .into_pool()
        .expect("pool must build")
}

fn eval_pool() -> Pool {
    let cols = load_x("X_eval.npy");
    let n = cols.first().map_or(0, Vec::len);
    pool_of(cols, vec![0.0; n])
}

/// The pinned fit from `gen_random_score_type_fixtures.py::PARAMS`.
fn builder(score_type: ERandomScoreType, strength: f64) -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(strength)
        .boost_from_average(true)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
        .random_score_type(score_type)
}

fn predict_with(score_type: ERandomScoreType, strength: f64) -> Vec<f64> {
    let pool = pool_of(load_x("X.npy"), load_y("y.npy"));
    let model = builder(score_type, strength)
        .fit(&pool)
        .expect("the fit must succeed");
    model.predict(&eval_pool()).expect("predict must succeed")
}

fn check(score_type: ERandomScoreType, file: &str) {
    let actual = predict_with(score_type, RANDOM_STRENGTH);
    let expected = load_y(file);
    assert_eq!(actual.len(), expected.len(), "{score_type:?}: count");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{score_type:?}: row {i} predicted {a} but catboost 1.2.10 says {e} \
             (|diff| {})",
            (a - e).abs()
        );
    }
}

#[test]
fn normal_with_model_size_decrease_matches_catboost() {
    check(
        ERandomScoreType::NormalWithModelSizeDecrease,
        "preds_NormalWithModelSizeDecrease.npy",
    );
}

#[test]
fn gumbel_matches_catboost() {
    check(ERandomScoreType::Gumbel, "preds_Gumbel.npy");
}

/// INERT at `random_strength = 0`: no perturbation is drawn at all, so the two
/// settings must be byte-identical to each other AND to the frozen unperturbed
/// baseline. This is what proves the wave cannot touch a default fit.
#[test]
fn random_score_type_is_inert_without_random_strength() {
    let expected = load_y("preds_strength0.npy");
    let mut seen: Vec<Vec<f64>> = Vec::new();
    for score_type in ERandomScoreType::all() {
        let actual = predict_with(score_type, 0.0);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= TOL,
                "{score_type:?} at random_strength=0: row {i} moved to {a} from {e}"
            );
        }
        seen.push(actual);
    }
    assert_eq!(
        seen.first(),
        seen.last(),
        "the two score types must be byte-identical without random_strength"
    );
}

/// The frozen fixture must separate the two distributions, else the parity
/// tests would pass for an implementation that ignores the parameter.
#[test]
fn the_frozen_fixture_separates_the_two_distributions() {
    let normal = load_y("preds_NormalWithModelSizeDecrease.npy");
    let gumbel = load_y("preds_Gumbel.npy");
    let sep = normal
        .iter()
        .zip(gumbel.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        sep > TOL,
        "the two distributions differ by only {sep}; the fixture cannot detect an \
         implementation that ignores random_score_type"
    );
}

/// Only the default type applies the model-size-decrease multiplier — the
/// structural difference behind the two names.
#[test]
fn only_the_default_type_decays_with_model_size() {
    assert!(ERandomScoreType::NormalWithModelSizeDecrease.uses_model_size_decrease());
    assert!(!ERandomScoreType::Gumbel.uses_model_size_decrease());
}

/// Every legal token round-trips; an unknown one is rejected.
#[test]
fn random_score_type_parses_every_legal_token() {
    for t in ERandomScoreType::all() {
        assert_eq!(ERandomScoreType::parse(t.as_str()), Some(t));
    }
    assert_eq!(ERandomScoreType::parse("gumbel"), None);
    assert_eq!(ERandomScoreType::parse("ZzBogusValue"), None);
}
