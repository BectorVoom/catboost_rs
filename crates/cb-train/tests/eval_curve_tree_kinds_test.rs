//! Regression lock: the eval-set approximant must advance for EVERY tree kind,
//! and a multi-dimensional loss must not silently get a dimension-0-only curve.
//!
//! The class of defect under test: the per-iteration eval update used to read
//! `trees.last()` — the OBLIVIOUS ensemble — alone. Under `grow_policy =
//! Lossguide` / `Depthwise` every tree is pushed to `non_symmetric_trees` and
//! `trees` stays EMPTY, so `eval_approx` never moved off `bias`. The metric
//! returned the same constant for all N iterations, which meant:
//!
//!   * `BestModelTracker` never saw a strictly-smaller value, so
//!     `best_iteration() == Some(0)` and `use_best_model` truncated the returned
//!     model to ONE tree;
//!   * the `Iter` detector saw no improvement and stopped after `od_wait`
//!     iterations regardless of `iterations`;
//!   * `eval_history` was a flat line.
//!
//! The device grow path is explicitly excluded when `eval_sets` is non-empty, so
//! nothing masked this.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_train::{
    train_with_eval, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType, EvalSet,
};

/// A small, learnable regression corpus: `y` is a clean linear function of two
/// informative float features, so ANY working boosting run drives the eval RMSE
/// down monotonically for the first several iterations.
fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f32>>, Vec<f64>) {
    let n = 120_usize;
    let f0: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    let f1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 2.0 * f64::from(f0[i]) - 3.0 * f64::from(f1[i]))
        .collect();

    let m = 40_usize;
    let e0: Vec<f32> = (0..m).map(|i| ((i + 3) % 13) as f32).collect();
    let e1: Vec<f32> = (0..m).map(|i| ((i + 5) % 7) as f32).collect();
    let ey: Vec<f64> = (0..m)
        .map(|i| 2.0 * f64::from(e0[i]) - 3.0 * f64::from(e1[i]))
        .collect();

    let borders = vec![
        (1..13).map(|k| f64::from(k) - 0.5).collect::<Vec<f64>>(),
        (1..7).map(|k| f64::from(k) - 0.5).collect::<Vec<f64>>(),
    ];
    (vec![f0, f1], borders, y, vec![e0, e1], ey)
}

fn params(grow_policy: EGrowPolicy, iterations: usize, use_best_model: bool) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth: 3,
        learning_rate: 0.3,
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
        od_wait: 20,
        use_best_model,
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
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

/// Train with an eval set under `grow_policy`, returning the model and the
/// per-iteration eval-metric curve.
fn run(grow_policy: EGrowPolicy, iterations: usize, use_best_model: bool) -> (usize, Vec<f64>) {
    let (x, borders, y, ex, ey) = corpus();
    let eval = EvalSet {
        feature_values: &ex,
        target: &ey,
        cat_columns: &[],
    };
    let mut curve = Vec::new();
    let model = train_with_eval(
        &CpuBackend,
        &x,
        &borders,
        &y,
        &[],
        &params(grow_policy, iterations, use_best_model),
        None,
        Some(&eval),
        Some(&mut curve),
    )
    .expect("training with an eval set must succeed");
    let tree_count = model.oblivious_trees.len()
        + model.non_symmetric_trees.len()
        + model.region_trees.len();
    (tree_count, curve)
}

/// The eval curve must MOVE under a non-symmetric grow policy.
///
/// A flat curve is the precise signature of the defect: `trees.last()` was
/// `None` every iteration, so nothing was ever added to `eval_approx`.
#[test]
fn the_eval_curve_advances_under_a_non_symmetric_grow_policy() {
    for policy in [EGrowPolicy::Lossguide, EGrowPolicy::Depthwise] {
        let (_, curve) = run(policy, 20, false);
        assert_eq!(curve.len(), 20, "one metric value per iteration ({policy:?})");
        let first = curve[0];
        assert!(
            curve.iter().any(|v| (v - first).abs() > 1e-9),
            "the eval metric under {policy:?} never changed across 20 iterations \
             (constant {first}) — the eval approximant is not being advanced by the \
             trees this policy actually produces"
        );
        // And it must IMPROVE: this corpus is exactly learnable.
        let last = curve[curve.len() - 1];
        assert!(
            last < first,
            "the eval metric under {policy:?} must improve ({first} -> {last})"
        );
    }
}

/// The oblivious path is unchanged — the same assertion must hold there, so a
/// regression that broke oblivious while fixing non-symmetric is caught too.
#[test]
fn the_eval_curve_advances_under_the_symmetric_grow_policy() {
    let (_, curve) = run(EGrowPolicy::SymmetricTree, 20, false);
    let first = curve[0];
    let last = curve[curve.len() - 1];
    assert!(
        last < first,
        "the oblivious eval metric must improve ({first} -> {last})"
    );
}

/// A MULTI-DIMENSIONAL loss combined with an eval set must be REFUSED, not
/// silently scored on output dimension 0.
///
/// `eval_approx` holds one `f64` per eval object with no `approx_dimension`
/// factor, the per-tree contribution reads `leaf_values[leaf]` (dimension 0 of a
/// dimension-major buffer), and `EvalMetric::eval` requires
/// `approx.len() == target.len()`. So a multiclass fit with an eval set produced
/// a curve computed from dimension 0's raw scores alone — and that curve drives
/// `use_best_model`'s truncation and the detector's stop. A silently-wrong
/// stopping decision is worse than a refused one.
#[test]
fn a_multi_dimensional_loss_with_an_eval_set_is_refused() {
    let (x, borders, _y, ex, _ey) = corpus();
    // A 3-class target -> approx_dimension 3.
    let y: Vec<f64> = (0..x[0].len()).map(|i| (i % 3) as f64).collect();
    let ey: Vec<f64> = (0..ex[0].len()).map(|i| (i % 3) as f64).collect();
    let eval = EvalSet {
        feature_values: &ex,
        target: &ey,
        cat_columns: &[],
    };
    let mut p = params(EGrowPolicy::SymmetricTree, 5, false);
    p.loss = Loss::MultiClass;
    // MultiClass has no Gradient leaf optimizer; Newton is upstream's default
    // and the only accepted method, so it must be set or the loss is rejected
    // before the eval-dimension guard is reached.
    p.leaf_method = LeafMethod::Newton;

    let err = train_with_eval(
        &CpuBackend,
        &x,
        &borders,
        &y,
        &[],
        &p,
        None,
        Some(&eval),
        None,
    )
    .expect_err("a multi-dimensional loss with an eval set must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("dimension"),
        "the error must explain that the eval surface is single-dimension, got: {msg}"
    );

    // ...and the SAME configuration without an eval set still trains.
    train_with_eval(&CpuBackend, &x, &borders, &y, &[], &p, None, None, None)
        .expect("a multi-dimensional loss without an eval set is unaffected");
}

/// `use_best_model` must not collapse a non-symmetric model to a single tree.
///
/// With a frozen eval curve, `best_iteration()` was `Some(0)` and the truncation
/// at the end of the boosting loop cut `non_symmetric_trees` down to ONE tree —
/// the user got back a near-constant predictor from a 20-iteration fit.
#[test]
fn use_best_model_does_not_truncate_a_non_symmetric_model_to_one_tree() {
    let (tree_count, _) = run(EGrowPolicy::Lossguide, 20, true);
    assert!(
        tree_count > 1,
        "use_best_model truncated a 20-iteration Lossguide fit to {tree_count} tree(s) \
         — the best-iteration tracker was fed a constant eval curve"
    );
}
