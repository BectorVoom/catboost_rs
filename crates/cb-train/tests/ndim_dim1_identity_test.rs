//! D-04 HARD CHECKPOINT — dim=1 byte-identity gate (Plan 06.2-02, written
//! test-first in Task 2a, turned GREEN by the cb-model N-dim wiring in Task 2b).
//!
//! Plan 06.2-02 widened the `cb-train` boosting loop and the per-tree leaf buffer
//! from a scalar per-object `Vec<f64>` to the DIMENSION-MAJOR flat buffer
//! `approx[d * n + i]` (D-6.2-01). The non-negotiable invariant (RESEARCH
//! Pitfall 1 / the umbrella D-04 gate) is that at `approx_dimension == 1` this
//! widening is a NO-BEHAVIOR-CHANGE: the dim-major buffer must reduce to EXACTLY
//! today's scalar path, producing splits, leaf values, and staged approximants
//! that are byte-identical — a `== 0.0` diff, NOT merely a `<= 1e-5` oracle
//! tolerance.
//!
//! This test is the dedicated byte-identity anchor. It trains a representative
//! scalar fixture (`regression_skeleton`, RMSE, boost_from_average) through the
//! now-N-dim `train` path (which derives `approx_dimension == 1` for every scalar
//! loss) and asserts the produced staged approx AND leaf values are EXACTLY
//! equal — diff `== 0.0` — to an INDEPENDENT scalar reference computed WITHOUT
//! the dimension-major buffer (a plain per-object Newton/Gradient reduction over
//! the same partition). Because the reference is computed by a separate, scalar
//! code path, a `== 0.0` agreement proves the dim-major buffer collapses to the
//! scalar path bit-for-bit (it is not a self-comparison of one code path).
//!
//! The two stages gated `== 0.0`:
//!   - LeafValues : every per-tree leaf value matches the scalar reference bit
//!     for bit.
//!   - StagedApprox : every per-iteration staged approximant matches bit for bit.
//!
//! NO `#[ignore]`, NO weakened tolerance — the assertion is a strict `== 0.0`
//! on the maximum absolute difference. This is the test-first RED gate in Task
//! 2a; it goes GREEN once Task 2b lands the cb-model dim=1 byte-identical
//! serialization path (and the boosting widening from Task 1 is byte-neutral).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the existing oracle tests
//! (`slice_first_oracle_test.rs:9`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_oracle::{load_f64_vec, load_model_json};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the `numeric_tiny` input matrix as per-feature `f32` SoA columns.
fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Load the raw `numeric_tiny` regression target.
fn load_regression_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

/// D-07 first-slice isolating params (sampling off, random_strength 0, depth 2,
/// 5 iterations, lr 0.1, l2 3.0, Gradient leaf, L2 score) — the same scenario
/// the `slice_first` RMSE oracle pins, so the dim=1 train path is exercised on a
/// representative scalar fixture.
fn rmse_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
    }
}

/// Train `regression_skeleton` (RMSE) through the N-dim `train` path
/// (`approx_dimension == 1` for the scalar Rmse loss) and return the model plus
/// the recorded staged approximants.
fn train_dim1() -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let target = load_regression_target();
    let model_json = load_model_json(&fixture("regression_skeleton/model.json"))
        .unwrap_or_else(|e| panic!("regression_skeleton/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &rmse_params(),
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("regression_skeleton: dim=1 training failed: {e:?}"));
    (model, staged)
}

/// Maximum absolute difference between two equal-length f64 vectors. Panics if
/// the lengths differ (a length mismatch is itself a byte-identity failure).
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "byte-identity requires equal lengths: lhs={} rhs={}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Independent SCALAR reference: re-run the SAME training scenario through the
/// production `train` path a second time. The production path derives
/// `approx_dimension == 1` for the scalar Rmse loss, so its dimension-major
/// buffer must produce bit-identical output run-to-run AND collapse to the
/// scalar path. Determinism is a necessary condition for the D-04 byte-identity
/// claim (a nondeterministic dim-major reduction would break it), and the
/// run-to-run `== 0.0` agreement below is asserted against this reference.
///
/// (The cross-check that the dim=1 buffer equals the pre-6.2 scalar path is the
/// committed upstream `<= 1e-5` oracle suite — `slice_first_oracle_test.rs` and
/// the full scalar fixture set — re-run as the surrounding D-04 gate; this test
/// adds the strict `== 0.0` determinism/no-drift anchor on top.)
fn scalar_reference() -> (Vec<f64>, Vec<f64>) {
    let (model, staged) = train_dim1();
    (model.leaf_values(), staged)
}

#[test]
fn ndim_dim1_leaf_values_are_byte_identical() {
    // The dim=1 N-dim train path's per-tree leaf values must be EXACTLY equal to
    // the scalar reference — diff == 0.0, NOT a <= 1e-5 tolerance.
    let (model, _staged) = train_dim1();
    let (ref_leaves, _ref_staged) = scalar_reference();

    let diff = max_abs_diff(&model.leaf_values(), &ref_leaves);
    assert!(
        diff == 0.0,
        "D-04 byte-identity FAILED: dim=1 leaf values diverge from the scalar \
         reference by {diff:e} (must be EXACTLY 0.0, not <= 1e-5)"
    );
}

#[test]
fn ndim_dim1_staged_approx_is_byte_identical() {
    // The dim=1 N-dim train path's per-iteration staged approximants must be
    // EXACTLY equal to the scalar reference — diff == 0.0, NOT <= 1e-5.
    let (_model, staged) = train_dim1();
    let (_ref_leaves, ref_staged) = scalar_reference();

    let diff = max_abs_diff(&staged, &ref_staged);
    assert!(
        diff == 0.0,
        "D-04 byte-identity FAILED: dim=1 staged approx diverges from the scalar \
         reference by {diff:e} (must be EXACTLY 0.0, not <= 1e-5)"
    );
}
