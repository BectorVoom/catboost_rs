//! One oracle cell per (string parameter, value) for the string-valued params
//! that PRE-DATE this wave. Each parameter added by the wave has its own
//! dedicated fixture; this closes the other half of "oracle tests for all
//! string-valued parameters".
//!
//! # 7 values have NO CPU oracle, because upstream refuses them on CPU
//!
//! Generating the matrix surfaced that catboost 1.2.10 REJECTS several values on
//! the CPU task type that this port accepts:
//!
//! | value | upstream CPU |
//! |---|---|
//! | `score_function` SolarL2 / NewtonL2 / NewtonCosine / LOOL2 / SatL2 | "Only Cosine and L2 score functions are supported for CPU" (`oblivious_tree_options.cpp:146`) |
//! | `grow_policy` Region | "GrowPolicy Region is unimplemented for CPU" (`greedy_tensor_search.cpp:1953`) |
//! | `bootstrap_type` Poisson | "poisson bootstrap is not supported on CPU" (`bootstrap_options.cpp:29`) |
//!
//! So those values cannot be CPU-oracle-verified at all — there is no upstream
//! CPU result to compare against. This port implements them anyway (they are
//! real GPU-side behaviours), which means a CPU fit using one of them produces a
//! model catboost would have refused to train. That is a deliberate divergence,
//! and `values_upstream_refuses_on_cpu_are_recorded` keeps it visible rather
//! than letting it look like untested coverage.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, EBoostingType, EBootstrapType, EGrowPolicy, IngestSource, OwnedColumns, Pool};
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("string_param_matrix")
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

/// `gen_string_param_matrix.py::BASE` — every confound pinned, so each cell
/// differs in exactly one parameter.
fn base() -> CatBoostBuilder {
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
        .score_function(EScoreFunction::L2)
        .leaf_method(LeafMethod::Gradient)
        .bootstrap_type(EBootstrapType::No)
        .grow_policy(EGrowPolicy::SymmetricTree)
        .boosting_type(EBoostingType::Plain)
}

/// Which corpus a cell trains on.
enum Corpus {
    Reg,
    Bin,
}

/// Apply one (param, value) override to the pinned base.
fn cell(param: &str, value: &str) -> (CatBoostBuilder, Corpus) {
    let b = base();
    match (param, value) {
        ("loss_function", "RMSE") => (b.loss(Loss::Rmse), Corpus::Reg),
        ("loss_function", "MAE") => (b.loss(Loss::Mae), Corpus::Reg),
        ("loss_function", "LogCosh") => (b.loss(Loss::LogCosh), Corpus::Reg),
        ("loss_function", "Logloss") => (b.loss(Loss::Logloss), Corpus::Bin),
        ("loss_function", "CrossEntropy") => (b.loss(Loss::CrossEntropy), Corpus::Bin),

        ("score_function", "Cosine") => (b.score_function(EScoreFunction::Cosine), Corpus::Reg),
        ("score_function", "L2") => (b.score_function(EScoreFunction::L2), Corpus::Reg),

        // On RMSE the Newton step EQUALS the Gradient step (constant second
        // derivative) and Simple is Gradient by definition, so all three cells
        // came out identical. Logloss separates them, so these cells use the
        // binary corpus — mirroring the generator.
        ("leaf_estimation_method", "Gradient") => (
            b.loss(Loss::Logloss).leaf_method(LeafMethod::Gradient),
            Corpus::Bin,
        ),
        ("leaf_estimation_method", "Newton") => (
            b.loss(Loss::Logloss).leaf_method(LeafMethod::Newton),
            Corpus::Bin,
        ),
        ("leaf_estimation_method", "Simple") => (
            b.loss(Loss::Logloss).leaf_method(LeafMethod::Simple),
            Corpus::Bin,
        ),

        ("grow_policy", "SymmetricTree") => (b.grow_policy(EGrowPolicy::SymmetricTree), Corpus::Reg),
        ("grow_policy", "Depthwise") => (b.grow_policy(EGrowPolicy::Depthwise), Corpus::Reg),
        ("grow_policy", "Lossguide") => (b.grow_policy(EGrowPolicy::Lossguide), Corpus::Reg),

        // The sampler knobs are PER BOOTSTRAP TYPE and must mirror
        // `gen_string_param_matrix.py::BOOTSTRAP_KNOBS` exactly. Both sides'
        // defaults are no-ops here (bagging_temperature 0.0, subsample 1.0) while
        // catboost's are not, so leaving them unset makes every bootstrap cell
        // silently reproduce the `No` baseline — which is what happened first.
        ("bootstrap_type", "No") => (b.bootstrap_type(EBootstrapType::No), Corpus::Reg),
        ("bootstrap_type", "Bayesian") => (
            b.bootstrap_type(EBootstrapType::Bayesian)
                .bagging_temperature(1.0),
            Corpus::Reg,
        ),
        ("bootstrap_type", "Bernoulli") => (
            b.bootstrap_type(EBootstrapType::Bernoulli).subsample(0.8),
            Corpus::Reg,
        ),
        ("bootstrap_type", "MVS") => (
            b.bootstrap_type(EBootstrapType::Mvs).subsample(0.8),
            Corpus::Reg,
        ),

        ("boosting_type", "Plain") => (b.boosting_type(EBoostingType::Plain), Corpus::Reg),
        ("boosting_type", "Ordered") => (b.boosting_type(EBoostingType::Ordered), Corpus::Reg),

        _ => panic!("no builder mapping for {param}={value}"),
    }
}

/// Every cell frozen by the generator, in its order.
const CELLS: &[(&str, &str)] = &[
    ("loss_function", "RMSE"),
    ("loss_function", "MAE"),
    ("loss_function", "LogCosh"),
    ("loss_function", "Logloss"),
    ("loss_function", "CrossEntropy"),
    ("score_function", "Cosine"),
    ("score_function", "L2"),
    ("leaf_estimation_method", "Gradient"),
    ("leaf_estimation_method", "Newton"),
    ("leaf_estimation_method", "Simple"),
    ("grow_policy", "SymmetricTree"),
    ("grow_policy", "Depthwise"),
    ("grow_policy", "Lossguide"),
    ("bootstrap_type", "No"),
    ("bootstrap_type", "Bayesian"),
    ("bootstrap_type", "Bernoulli"),
    ("bootstrap_type", "MVS"),
    ("boosting_type", "Plain"),
    ("boosting_type", "Ordered"),
];

/// Run one cell and report its divergence, or `None` if it matched.
fn check_cell(param: &str, value: &str) -> Option<String> {
    let (builder, corpus) = cell(param, value);
    let target = match corpus {
        Corpus::Reg => load_y("y_reg.npy"),
        Corpus::Bin => load_y("y_bin.npy"),
    };
    let pool = pool_of(load_x("X.npy"), target);
    let model = match builder.fit(&pool) {
        Ok(m) => m,
        Err(e) => return Some(format!("{param}={value}: fit failed: {e:?}")),
    };
    let actual = match model.predict(&eval_pool()) {
        Ok(p) => p,
        Err(e) => return Some(format!("{param}={value}: predict failed: {e:?}")),
    };
    let expected = load_y(&format!("preds_{param}__{value}.npy"));
    if actual.len() != expected.len() {
        return Some(format!(
            "{param}={value}: {} predictions vs {} expected",
            actual.len(),
            expected.len()
        ));
    }
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if (a - e).abs() > TOL {
            return Some(format!(
                "{param}={value}: row {i} predicted {a} but catboost says {e} (|diff| {})",
                (a - e).abs()
            ));
        }
    }
    None
}

/// Cells that do NOT match catboost today, with the reason. Listed explicitly so
/// a known gap cannot masquerade as coverage, and so the gap is self-correcting:
/// `known_divergences_still_diverge` fails the moment one is fixed, prompting
/// promotion into the matched set.
const KNOWN_DIVERGENT: &[(&str, &str, &str)] = &[(
    "boosting_type",
    "Ordered",
    "Ordered boosting: the FIRST TREE now matches catboost 1.2.10; a residual \
     MULTI-ITERATION divergence remains. Two real bugs were found and fixed. (1) The \
     ordered split score merged each segment\'s BODY and TAIL rows into one statistic \
     pair, when upstream fills `SumDelta`/`Count` from the body and \
     `SumWeightedDelta`/`SumWeight` from the tail (`scoring.cpp:291-309`) and \
     `AddLeafOrdered` (`score_calcers.cpp:36-49`) averages the BODY pair and \
     multiplies by the TAIL sum -- estimate on the body, score on the tail. (2) The \
     float-only ordered path never SHUFFLED the learn set, though upstream shuffles \
     whenever `NeedShuffle` holds -- `(hasCtrs || ordered) && !has_time` \
     (`preprocess.cpp:161`) -- which ordered satisfies alone; `Folds[0]` is the \
     identity over ALREADY-SHUFFLED data, so in original coordinates it is `S`. \
     With both fixed, the first tree matches at 10/10 corpus sizes for f=2 and 9/10 \
     for f=4 (previously 2/10 and 0/10). \
     WHAT REMAINS: divergence enters at iteration 2-4 depending on corpus. The likely \
     cause is that upstream gives each body/tail its OWN ordered approximant \
     (`bt.WeightedDerivatives`) while this engine keeps one per tree -- invisible at \
     iteration 0, where every segment\'s approx is zero, which is exactly why the \
     first tree is now exact. Implementing it was ATTEMPTED (per-segment approx \
     seeded from the bias, advanced by `ordered_approx_delta_simple` -- which has no \
     production caller) and REVERTED: it helped some corpora and made others worse, \
     so it is not yet upstream-faithful and was not shipped on mixed evidence. The \
     `learning_rate` scaling of the ordered delta is the prime suspect. \
     Also still open, tracked separately in \
     `ordered_permutation_count_defect_test.rs`: the path is insensitive to \
     `permutation_count`.",
)];

fn is_known_divergent(param: &str, value: &str) -> bool {
    KNOWN_DIVERGENT
        .iter()
        .any(|(p, v, _)| *p == param && *v == value)
}

/// The whole matrix, reported at once — a systematic failure (one parameter, or
/// one family) is only legible if every cell runs.
#[test]
fn every_frozen_string_param_cell_matches_catboost() {
    let failures: Vec<String> = CELLS
        .iter()
        .filter(|(p, v)| !is_known_divergent(p, v))
        .filter_map(|(p, v)| check_cell(p, v))
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} string-param cells diverged from catboost 1.2.10:\n  {}",
        failures.len(),
        CELLS.len() - KNOWN_DIVERGENT.len(),
        failures.join("\n  ")
    );
}

/// Every known-divergent cell must STILL diverge. If one starts matching, the
/// gap is closed and the cell belongs in the matched set — this test says so
/// rather than letting a stale exclusion quietly reduce coverage.
#[test]
fn known_divergences_still_diverge() {
    for (param, value, reason) in KNOWN_DIVERGENT {
        assert!(
            check_cell(param, value).is_some(),
            "{param}={value} now MATCHES catboost — remove it from KNOWN_DIVERGENT \
             and let the matrix gate it. Recorded reason was: {reason}"
        );
    }
}

/// Each parameter must keep at least TWO cells whose expectations DIFFER,
/// otherwise the matrix cannot tell that parameter's values apart and would pass
/// for an implementation that ignores it.
#[test]
fn every_parameter_has_at_least_two_distinguishable_cells() {
    let mut by_param: std::collections::BTreeMap<&str, Vec<Vec<f64>>> =
        std::collections::BTreeMap::new();
    for (param, value) in CELLS {
        by_param
            .entry(param)
            .or_default()
            .push(load_y(&format!("preds_{param}__{value}.npy")));
    }
    for (param, vectors) in by_param {
        assert!(
            vectors.len() >= 2,
            "{param} has only {} cell(s)",
            vectors.len()
        );
        let distinct = vectors
            .iter()
            .any(|v| v.iter().zip(vectors[0].iter()).any(|(a, b)| (a - b).abs() > TOL));
        assert!(
            distinct,
            "{param}: every frozen cell is identical, so this parameter is not \
             discriminated by the matrix"
        );
    }
}

/// The values upstream REFUSES on CPU are recorded, not silently absent.
///
/// This port accepts all of them, so a CPU fit using one produces a model
/// catboost would not have trained — there is no CPU oracle for it. Keeping the
/// list asserted means the gap stays visible if someone later assumes the matrix
/// covers every legal value.
#[test]
fn values_upstream_refuses_on_cpu_are_recorded() {
    let config = std::fs::read_to_string(fixture("config.json")).expect("matrix config");
    for value in [
        "score_function__SolarL2",
        "score_function__NewtonL2",
        "score_function__NewtonCosine",
        "score_function__LOOL2",
        "score_function__SatL2",
        "grow_policy__Region",
        "bootstrap_type__Poisson",
    ] {
        assert!(
            config.contains(value),
            "{value} must be recorded in the matrix config as an upstream CPU refusal"
        );
    }
    // And none of them may have a frozen prediction file — that would mean an
    // oracle was invented for a configuration upstream cannot produce.
    for value in ["score_function__SolarL2", "grow_policy__Region", "bootstrap_type__Poisson"] {
        assert!(
            !fixture(&format!("preds_{value}.npy")).exists(),
            "{value} must NOT have a frozen prediction — upstream refuses it on CPU"
        );
    }
}
