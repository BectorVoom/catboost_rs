//! Regression test for CR-02 / WR-05 (06.2-07): the multi-output sampling /
//! random_strength path must operate PER OBJECT.
//!
//! Before 06.2-07 the dim-major `weighted_der1` (length `approx_dimension * n`)
//! was passed unmodified to `bootstrap` / `mvs_lambda` / `score_st_dev`, which
//! all assume a per-object length `n`. With `bootstrap_type=No` and
//! `random_strength=0` (the in-scope multiclass/multilabel fixtures) the bug is
//! latent; the moment a user enables either knob on a multi-output loss the
//! sampling draws `dim*n` uniforms (wrong RNG phase) and `scoreStDev` is scaled
//! by `1/sqrt(dim*n)` instead of `1/sqrt(n)`.
//!
//! This test trains a multi-output model (`approx_dimension > 1`) WITH Bernoulli
//! bootstrap and a non-zero `random_strength` and asserts:
//!   (a) training does not panic / corrupt and produces `approx_dimension`
//!       outputs;
//!   (b) the std-dev statistic divides the FULL dim-major sum of squares by the
//!       per-OBJECT count `n` (NOT `dim*n`) — the corrected divisor;
//!   (c) WR-05: a multi-output train with an out-of-range target class returns a
//!       typed error (no degenerate gradient).
//!
//! Integration test (under `tests/`). NO `#[ignore]`, NO weakened tolerance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::CpuBackend;
use cb_compute::{derivatives_std_dev_from_zero, EScoreFunction, LeafMethod, Loss, Runtime};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType};

/// A small deterministic multi-output training corpus: 8 objects, 2 features,
/// 3 classes. Borders split each feature at a couple of interior values.
fn synthetic_columns() -> Vec<Vec<f32>> {
    vec![
        // feature 0
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        // feature 1
        vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    ]
}

fn synthetic_borders() -> Vec<Vec<f64>> {
    vec![vec![2.5, 5.5], vec![2.5, 5.5]]
}

/// 3-class target over the 8 objects (classes in [0, 3)).
fn synthetic_target() -> Vec<f64> {
    vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0]
}

fn sampling_params(loss: Loss, bootstrap_type: EBootstrapType) -> BoostParams {
    BoostParams {
        loss,
        iterations: 4,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        // Non-zero random_strength exercises the score_st_dev (÷n) path.
        random_strength: 1.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Newton,
        bootstrap_type,
        // Sub-unit subsample exercises the per-object Bernoulli control mask.
        subsample: 0.75,
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
        score_function: EScoreFunction::Cosine,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
    }
}

/// (a) A multi-output model trains to completion under Bernoulli bootstrap +
/// random_strength WITHOUT panicking or corrupting, and produces
/// `approx_dimension == 3` outputs. Before the CR-02 fix the dim-major derivative
/// fed the per-object sampling path, advancing the RNG by `dim*n` draws and
/// biasing `scoreStDev` — this run is the end-to-end no-corruption guard.
#[test]
fn multiclass_onevsall_bernoulli_random_strength_trains_per_object() {
    let columns = synthetic_columns();
    let borders = synthetic_borders();
    let target = synthetic_target();
    let params = sampling_params(Loss::MultiClassOneVsAll, EBootstrapType::Bernoulli);

    let model = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .expect("multi-output Bernoulli + random_strength training must not corrupt");

    assert_eq!(
        model.approx_dimension, 3,
        "a 3-class model must carry approx_dimension == 3"
    );
    assert_eq!(
        model.oblivious_trees.len(),
        params.iterations,
        "every iteration must grow one tree"
    );
    // Every tree's leaf-value buffer is dimension-major length `dim * n_leaves`.
    for tree in &model.oblivious_trees {
        assert_eq!(
            tree.leaf_values.len() % model.approx_dimension,
            0,
            "leaf values must be dimension-major (length multiple of dim)"
        );
        assert!(
            tree.leaf_values.iter().all(|v| v.is_finite()),
            "no leaf value may be NaN/inf — the sampling path stayed well-formed"
        );
    }
}

/// (a') The MVS bootstrap arm on a multi-output loss likewise trains per-object
/// without corruption (MVS reads the per-object aggregated derivative).
#[test]
fn multiclass_onevsall_mvs_trains_per_object() {
    let columns = synthetic_columns();
    let borders = synthetic_borders();
    let target = synthetic_target();
    let params = sampling_params(Loss::MultiClassOneVsAll, EBootstrapType::Mvs);

    let model = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .expect("multi-output MVS training must not corrupt");
    assert_eq!(model.approx_dimension, 3);
    assert_eq!(model.oblivious_trees.len(), params.iterations);
}

/// (b) CR-02 std-dev divisor: `derivatives_std_dev_from_zero` divides the FULL
/// dim-major sum of squares by the per-OBJECT count `n`, NOT by `dim*n`. With
/// `approx_dimension = 3` and `n = 4` the dim-major buffer has length 12; the
/// statistic is `sqrt(Σ_all(wd²) / n)`, hand-computed here.
#[test]
fn std_dev_divides_by_n_not_dim_n_on_multidim_buffer() {
    let n = 4usize;
    let dim = 3usize;
    // dim-major weighted_der1 (length dim*n = 12): d*n + i.
    let wd: Vec<f64> = vec![
        // d=0
        1.0, -2.0, 0.5, -1.5, //
        // d=1
        2.0, 1.0, -0.5, 0.25, //
        // d=2
        -1.0, 0.75, 1.25, -0.25,
    ];
    assert_eq!(wd.len(), dim * n);

    let sum2: f64 = wd.iter().map(|&v| v * v).sum();
    let expected = (sum2 / n as f64).sqrt();
    let got = derivatives_std_dev_from_zero(&wd, n);
    assert!(
        (got - expected).abs() < 1e-13,
        "dsdz must divide Σ(wd²) by n={n}: got {got}, expected {expected}"
    );

    // It must NOT be the old (wrong) divide-by-(dim*n) value.
    let wrong = (sum2 / wd.len() as f64).sqrt();
    assert!(
        (got - wrong).abs() > 1e-9,
        "dsdz must not divide by dim*n = {}",
        wd.len()
    );
}

/// (b') dim=1 reduction (D-04): at `approx_dimension == 1` the dim-major buffer
/// has length `n`, so the multidim std-dev reduces exactly to the scalar RMS.
#[test]
fn std_dev_dim1_reduces_to_scalar_rms() {
    let wd = [1.0_f64, -2.0, 3.0, -0.5];
    let n = wd.len();
    let got = derivatives_std_dev_from_zero(&wd, n);
    let expected = (wd.iter().map(|&v| v * v).sum::<f64>() / n as f64).sqrt();
    assert!((got - expected).abs() < 1e-13, "dim=1 dsdz must be the scalar RMS");
}

/// (c) WR-05: the multi-output softmax / one-vs-all gradient producer rejects an
/// out-of-range `target_class >= k` with a typed error instead of silently
/// emitting the degenerate no-`+1` (`-p`) gradient. The trainer remaps labels to
/// `[0, k)` before this boundary, so the corruption is only reachable when a
/// mismatched (target, remap) pair crosses into the producer — exercised here at
/// the backend boundary where WR-05 enforces the bound.
#[test]
fn multiclass_gradient_producer_rejects_out_of_range_class() {
    let n = 4usize;
    let k = 3usize;
    // dim-major approx (length k*n).
    let approx = [0.1_f64, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2];
    assert_eq!(approx.len(), k * n);

    // A valid per-object class target trains a well-formed gradient.
    let target_ok = [0.0_f64, 1.0, 2.0, 0.0];
    assert!(
        CpuBackend
            .compute_gradients(&Loss::MultiClass, &approx, &target_ok, k)
            .is_ok(),
        "an in-range multiclass target must produce a valid gradient"
    );
    assert!(
        CpuBackend
            .compute_gradients(&Loss::MultiClassOneVsAll, &approx, &target_ok, k)
            .is_ok(),
        "an in-range one-vs-all target must produce a valid gradient"
    );

    // An out-of-range class (3 >= k=3) is a typed error, never a degenerate train.
    let target_oob = [0.0_f64, 1.0, 3.0, 0.0];
    assert!(
        CpuBackend
            .compute_gradients(&Loss::MultiClass, &approx, &target_oob, k)
            .is_err(),
        "WR-05: softmax must reject target_class >= k"
    );
    assert!(
        CpuBackend
            .compute_gradients(&Loss::MultiClassOneVsAll, &approx, &target_oob, k)
            .is_err(),
        "WR-05: one-vs-all must reject target_class >= k"
    );
}
