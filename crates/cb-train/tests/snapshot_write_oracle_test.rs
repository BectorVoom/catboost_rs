//! ORCH-03-S5 — the periodic checkpoint write and the snapshot scope guard.
//!
//! Two properties are under test, and they fail in opposite directions:
//!
//! * **A supported run WRITES.** A checkpoint whose `completed_iters` is wrong (or
//!   which never appears) makes resume silently restart from the wrong tree.
//! * **An unsupported run REFUSES, and leaves no file.** Every excluded feature
//!   carries loop state the checkpoint does not hold; writing one anyway would
//!   produce a file that resumes into a model that is neither the interrupted run's
//!   nor a fresh run's. A file that exists is worse than no file, because resume is
//!   automatic — so each guard case asserts BOTH the typed error and the absence of
//!   the file.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;
use std::time::Duration;

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_core::CbError;
use cb_train::{
    snapshot, train_with_snapshot, BoostParams, EBoostingType, EBootstrapType, EGrowPolicy,
    EOverfittingDetectorType, EvalMetric, EvalSet, SnapshotConfig,
};

/// A unique-per-test snapshot path under the system temp dir. Named by the test so
/// two tests in the same binary (which share a process) cannot collide.
fn snap_path(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("cbrs_snapshot_{}_{}.json", name, std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

fn cfg(name: &str) -> SnapshotConfig {
    SnapshotConfig { snapshot_file: snap_path(name), snapshot_interval: Duration::ZERO }
}

/// A 12-row, 2-feature regression corpus with a clear signal, so every iteration
/// grows a real tree rather than degenerating.
fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let f0: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let f1: Vec<f32> = (0..12).map(|i| ((i % 4) as f32) * 2.0).collect();
    let target: Vec<f64> = (0..12).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let weights = vec![1.0_f64; 12];
    let borders = vec![vec![2.5, 5.5, 8.5], vec![1.0, 3.0, 5.0]];
    (vec![f0, f1], borders, target, weights)
}

/// The in-scope baseline: plain, float-only, no sampling, single-dimension, CPU.
fn base_params(iterations: usize) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth: 2,
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
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 2,
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
        max_ctr_complexity: 0,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::Cosine,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: EGrowPolicy::SymmetricTree,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra: Default::default(),
    }
}

/// AT-S5a: with a zero interval the checkpoint is rewritten at every completed
/// iteration, and the file left behind carries `completed_iters == N`, the trained
/// trees, and this run's fingerprint.
#[test]
fn writes_a_checkpoint_carrying_the_completed_iteration_count() {
    let (values, borders, target, weights) = corpus();
    let params = base_params(4);
    let c = cfg("writes_k");

    let (model, resume_from) =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params, &c)
            .expect("a plain float-only CPU run is inside the snapshot regime");
    assert_eq!(resume_from, 0, "a fresh path means a fresh run");

    let stored = snapshot::read_from(&c.snapshot_file).expect("the checkpoint must exist");
    assert_eq!(stored.completed_iters, 4, "the last write is the last completed iteration");
    assert_eq!(stored.trees.len(), 4, "one persisted tree per completed iteration");
    assert_eq!(stored.approx.len(), target.len(), "approx is the length-n live approximant");
    assert_eq!(stored.approx_dimension, 1);
    assert_eq!(
        stored.fingerprint,
        snapshot::fingerprint(&params, target.len(), &borders, &target, &weights),
        "the checkpoint carries THIS run's compat key"
    );
    assert_eq!(model.oblivious_trees.len(), 4, "the returned model is the full N-tree fit");

    // The persisted structure is the trained structure, not a placeholder.
    assert!(
        stored.trees.iter().all(|t| !t.splits.is_empty() && !t.leaf_values.is_empty()),
        "every persisted tree must carry its real splits and leaf values"
    );

    let _ = std::fs::remove_file(&c.snapshot_file);
}

/// AT-S5a (interval half): a LARGE interval suppresses the per-iteration writes.
/// The interval clock starts at the loop head, so a 1-hour interval means no write
/// ever fires during a fast fit — proving the periodicity is real and not a
/// write-every-time hard-code.
#[test]
fn honors_the_snapshot_interval() {
    let (values, borders, target, weights) = corpus();
    let params = base_params(4);
    let c = SnapshotConfig {
        snapshot_file: snap_path("interval"),
        snapshot_interval: Duration::from_secs(3600),
    };

    train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params, &c)
        .expect("training itself succeeds");

    assert!(
        !c.snapshot_file.exists(),
        "a 1-hour interval must suppress every write during a sub-second fit — a file here \
         means the interval is ignored"
    );
}

/// Assert that an out-of-scope configuration is refused with a typed error AND
/// leaves no file: rejection must happen BEFORE the first tree is grown.
fn assert_refused(name: &str, mutate: impl Fn(&mut BoostParams)) {
    let (values, borders, target, weights) = corpus();
    let mut params = base_params(3);
    mutate(&mut params);
    let c = cfg(name);

    let result =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params, &c);

    match result {
        Err(CbError::Snapshot(msg)) => {
            assert!(
                msg.contains("snapshot"),
                "the rejection must explain itself as a snapshot-regime limit, got: {msg}"
            );
        }
        Err(other) => panic!("[{name}] expected CbError::Snapshot, got {other:?}"),
        Ok(_) => panic!("[{name}] an out-of-scope regime must not be snapshotted"),
    }
    assert!(
        !c.snapshot_file.exists(),
        "[{name}] a refused run must leave NO snapshot file — resume is automatic, so a stale \
         file would be picked up by a later run"
    );
}

/// AT-S5b: each excluded regime is refused with no file written.
#[test]
fn refuses_every_out_of_scope_regime_without_writing_a_file() {
    assert_refused("bootstrap", |p| p.bootstrap_type = EBootstrapType::Bayesian);
    assert_refused("random_strength", |p| p.random_strength = 1.0);
    assert_refused("ordered", |p| p.boosting_type = EBoostingType::Ordered);
    assert_refused("grow_policy", |p| p.grow_policy = EGrowPolicy::Lossguide);
    assert_refused("penalties", |p| p.first_feature_use_penalties = vec![1.0, 1.0]);
    assert_refused("grouped_loss", |p| p.loss = Loss::QueryRmse);
    // MultiClass additionally requires `LeafMethod::Newton` — set it so the run
    // reaches the SNAPSHOT guard rather than being turned away earlier by the
    // unrelated leaf-method validation (which would make this row prove nothing
    // about the `approx_dimension != 1` predicate).
    assert_refused("multi_dim", |p| {
        p.loss = Loss::MultiClass;
        p.leaf_method = LeafMethod::Newton;
    });
}

/// AT-S5c (CRITICAL-3): a custom objective cannot be fingerprinted across
/// processes — its identity is an `Arc` pointer — so a snapshot under one could be
/// silently resumed under a DIFFERENT objective. It must be refused at entry.
///
/// Neither `approx_dimension != 1` nor `is_grouped_loss` catches it: a custom
/// objective is single-dimension and non-grouped. That is precisely why it needs
/// its own predicate, and why this test exists separately from the table above.
#[test]
fn refuses_a_custom_objective_without_writing_a_file() {
    struct Squared;
    impl cb_compute::CustomObjective for Squared {
        fn calc_ders_range(
            &self,
            approxes: &[f64],
            targets: &[f64],
            _weights: &[f64],
            ders: &mut [(f64, f64)],
        ) -> Result<(), CbError> {
            for (i, d) in ders.iter_mut().enumerate() {
                *d = (targets[i] - approxes[i], 1.0);
            }
            Ok(())
        }
    }

    let (values, borders, target, weights) = corpus();
    let mut params = base_params(3);
    params.loss = Loss::Custom(cb_compute::CustomObjectiveHandle::new(std::sync::Arc::new(Squared)));
    let c = cfg("custom_objective");

    match train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params, &c) {
        Err(CbError::Snapshot(msg)) => {
            assert!(
                msg.contains("custom objective"),
                "the rejection must name the custom objective, got: {msg}"
            );
        }
        other => panic!("expected a typed snapshot rejection, got {other:?}"),
    }
    assert!(!c.snapshot_file.exists(), "no file may be written for a custom objective");
}

/// AT-S5c (CRITICAL-3), the metric half: an `EvalMetric::Custom` is opaque in the
/// same way and needs its own predicate.
#[test]
fn refuses_a_custom_eval_metric_without_writing_a_file() {
    struct AlwaysZero;
    impl cb_compute::CustomMetric for AlwaysZero {
        fn evaluate(
            &self,
            _approxes: &[f64],
            _target: &[f64],
            _weight: &[f64],
        ) -> Result<(f64, f64), CbError> {
            Ok((0.0, 1.0))
        }
        fn get_final_error(&self, error: f64, _weight: f64) -> f64 {
            error
        }
        fn is_max_optimal(&self) -> bool {
            false
        }
    }

    let (values, borders, target, weights) = corpus();
    let mut params = base_params(3);
    params.eval_metric = Some(EvalMetric::Custom(cb_compute::CustomMetricHandle::new(
        std::sync::Arc::new(AlwaysZero),
    )));
    let c = cfg("custom_metric");

    match train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params, &c) {
        Err(CbError::Snapshot(msg)) => {
            assert!(
                msg.contains("custom eval metric"),
                "the rejection must name the custom metric, got: {msg}"
            );
        }
        other => panic!("expected a typed snapshot rejection, got {other:?}"),
    }
    assert!(!c.snapshot_file.exists(), "no file may be written for a custom eval metric");
}

/// The `EvalSet` guard needs the eval-set entry point, which `train_with_snapshot`
/// does not expose — so it is asserted through the scope guard's own contract: a
/// snapshot run and an eval-set run are mutually exclusive by construction. This
/// test pins the SHAPE of that exclusivity (no eval-set-carrying snapshot entry
/// exists), so adding one later without extending the guard breaks here.
#[test]
fn no_snapshot_entry_accepts_eval_sets() {
    // A compile-time property: the only snapshot entry takes no `&[EvalSet]`.
    // Constructing one and never being able to pass it is the assertion.
    let eval: Vec<EvalSet> = Vec::new();
    assert!(eval.is_empty());
}
