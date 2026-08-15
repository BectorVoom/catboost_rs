//! `thread_count` parity oracle against catboost 1.2.10.
//!
//! `thread_count` is the one parameter in this wave whose correct behaviour is to
//! change NOTHING. Upstream measures max|diff| = 0 across
//! `thread_count` ∈ {1, 2, 4, 8, 16, -1}, and that invariance is structural, not
//! incidental: every parallelised loop blocks on CONSTANT block sizes
//! (`CB_THREAD_LIMIT = 128`), never on the runtime thread count, so neither the
//! RNG stream nor any reduction order can depend on it.
//!
//! So the test is an INVARIANCE test, and a demanding one — it is run under the
//! configurations that actually consume randomness and reduce in parallel
//! (bootstrap, `random_strength`, `rsm`), because those are where a
//! thread-dependent implementation would show up. A per-thread RNG or a
//! rayon-order-dependent float reduction would pass a single-threaded check and
//! fail here.
//!
//! It also anchors the numbers to upstream: matching ourselves at every thread
//! count is worthless if the shared value is wrong, so one cell is checked
//! against the frozen `rsm` fixtures (`thread_count` is pinned to 1 there, and we
//! must reproduce it at every count).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, EBootstrapType, IngestSource, OwnedColumns, Pool};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

/// The thread counts swept. `0` is "all cores" (upstream's `-1`); the rest bracket
/// it from below, including counts above and below this machine's core count.
const THREAD_COUNTS: [usize; 6] = [1, 2, 3, 4, 8, 0];

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

/// Matches `gen_rsm_fixtures.py::BASE` (which pins `thread_count=1` upstream).
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

/// Sweep `THREAD_COUNTS` over a configuration and require BIT-identical output.
fn assert_invariant(label: &str, make: impl Fn(CatBoostBuilder) -> CatBoostBuilder) {
    let reference = preds_with(make(builder()).thread_count(1));
    for tc in THREAD_COUNTS {
        let got = preds_with(make(builder()).thread_count(tc));
        assert_eq!(
            got, reference,
            "{label}: thread_count={tc} changed the model. thread_count is a resource \
             knob only — upstream measures max|diff| = 0 across every count, so any \
             difference means a reduction or an RNG stream has become \
             thread-order dependent"
        );
    }
}

// ===========================================================================
// Invariance — under every configuration that actually uses randomness
// ===========================================================================

#[test]
fn the_plain_fit_is_thread_count_invariant() {
    assert_invariant("plain", |b| b);
}

/// Bernoulli draws per object, so a per-thread RNG would diverge here.
#[test]
fn a_bernoulli_bootstrap_fit_is_thread_count_invariant() {
    assert_invariant("bernoulli", |b| {
        b.bootstrap_type(EBootstrapType::Bernoulli).subsample(0.7)
    });
}

#[test]
fn a_bayesian_bootstrap_fit_is_thread_count_invariant() {
    assert_invariant("bayesian", |b| {
        b.bootstrap_type(EBootstrapType::Bayesian).bagging_temperature(1.0)
    });
}

/// MVS both draws AND reduces over the per-object gradient norms.
#[test]
fn an_mvs_fit_is_thread_count_invariant() {
    assert_invariant("mvs", |b| b.bootstrap_type(EBootstrapType::Mvs).subsample(0.7));
}

/// `random_strength` draws once per candidate inside the parallel-over-features
/// level pass — the likeliest place for a thread-order dependency to appear.
#[test]
fn a_random_strength_fit_is_thread_count_invariant() {
    assert_invariant("random_strength", |b| b.random_strength(1.0));
}

/// `rsm` decides candidates from the shared RNG per level.
#[test]
fn an_rsm_fit_is_thread_count_invariant() {
    assert_invariant("rsm", |b| b.rsm(0.5));
}

/// All of them at once.
#[test]
fn a_combined_fit_is_thread_count_invariant() {
    assert_invariant("combined", |b| {
        b.bootstrap_type(EBootstrapType::Bernoulli)
            .subsample(0.7)
            .random_strength(1.0)
            .rsm(0.5)
    });
}

// ===========================================================================
// ...and the invariant value is the one upstream computes
// ===========================================================================

/// Agreeing with ourselves at every thread count proves nothing if the shared
/// value is wrong. The `rsm` fixtures were captured at upstream `thread_count=1`,
/// so every count here must reproduce them.
#[test]
fn every_thread_count_reproduces_the_upstream_numbers() {
    let expected = load_y("preds_rsm_0p5.npy");
    for tc in THREAD_COUNTS {
        let actual = preds_with(builder().rsm(0.5).thread_count(tc));
        assert_eq!(actual.len(), expected.len(), "thread_count={tc}: prediction count");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= TOL,
                "thread_count={tc} row {i}: predicted {a} but catboost 1.2.10 says {e}"
            );
        }
    }
}

/// A thread count far above the core count must still train (rayon oversubscribes
/// rather than failing), so the knob cannot be turned into an accidental limit.
#[test]
fn an_oversubscribed_thread_count_still_trains() {
    let reference = preds_with(builder().thread_count(1));
    let got = preds_with(builder().thread_count(256));
    assert_eq!(got, reference, "thread_count=256 must train and stay invariant");
}
