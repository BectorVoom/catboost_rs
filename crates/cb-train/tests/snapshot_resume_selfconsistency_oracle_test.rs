//! ORCH-03-S6 — the keystone: a RESUMED run produces exactly the model a
//! straight-through run produces.
//!
//! This is the property the whole slice exists for, and the only one that can fail
//! silently. A resume that restores the wrong state, or restarts the loop at the
//! wrong iteration, still returns a plausible model — it is simply the WRONG model,
//! and nothing downstream would notice. So the assertion here is exact equality of
//! the full tree structure and every leaf value, not an approximate comparison.
//!
//! # How the intermediate fixture is built (the `refingerprinted_k_run` mechanism)
//!
//! Testing resume needs a snapshot that a real `iterations = N` run would have
//! written after its K-th tree. Rather than adding an interruption escape-hatch to
//! `train_inner` — production code existing only to be testable — the fixture is
//! assembled from the public surfaces the earlier tasks already ship:
//!
//! 1. Run `train_with_snapshot` with `iterations = K` and a ZERO interval, so the
//!    final write is a genuine K-tree checkpoint.
//! 2. `decode` it and overwrite the ONE field that legitimately differs from a
//!    mid-N-run snapshot: its `fingerprint`, computed for `iterations = K`.
//! 3. Write it back with `write_atomic`.
//!
//! This is valid because the scoped path's first K trees are `iterations`-
//! INDEPENDENT: each tree depends only on the running `approx` and the fixed
//! params. That independence breaks under `auto_learning_rate = true` (the
//! effective rate is a function of `params.iterations`), which is why every fixture
//! here pins `auto_learning_rate = false`. An auto-LR resume is not silently wrong
//! — fingerprint field 13 catches an auto-LR change — its self-consistency is
//! simply out of this slice.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;
use std::time::Duration;

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_core::CbError;
use cb_train::{
    snapshot, train, train_with_snapshot, BoostParams, EBoostingType, EBootstrapType, EGrowPolicy,
    EOverfittingDetectorType, Model, SnapshotConfig,
};

fn snap_path(name: &str) -> PathBuf {
    let p = std::env::temp_dir()
        .join(format!("cbrs_resume_{}_{}.json", name, std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

/// A 16-row, 3-feature corpus with enough structure that consecutive trees differ —
/// so a resume that restarted from the wrong iteration would produce a visibly
/// different model rather than accidentally agreeing.
fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let f0: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let f1: Vec<f32> = (0..16).map(|i| ((i * 7) % 11) as f32).collect();
    let f2: Vec<f32> = (0..16).map(|i| ((i % 5) as f32) * 1.5).collect();
    let target: Vec<f64> =
        (0..16).map(|i| f64::from(i % 3) * 0.5 + if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let weights = vec![1.0_f64; 16];
    let borders = vec![
        vec![3.5, 7.5, 11.5],
        vec![2.0, 5.0, 8.0],
        vec![1.0, 3.0, 5.0],
    ];
    (vec![f0, f1, f2], borders, target, weights)
}

fn params(iterations: usize) -> BoostParams {
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
        random_seed: 7,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        // PINNED — see the module docs: the fixture mechanism relies on the first K
        // trees being independent of `params.iterations`.
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

/// Exact structural equality of two trained models: same tree count, same splits in
/// the same order, same leaf values and weights, bit for bit.
fn assert_models_identical(a: &Model, b: &Model, context: &str) {
    assert_eq!(
        a.oblivious_trees.len(),
        b.oblivious_trees.len(),
        "[{context}] tree count differs"
    );
    assert_eq!(a.bias.to_bits(), b.bias.to_bits(), "[{context}] bias differs");
    for (i, (ta, tb)) in a.oblivious_trees.iter().zip(b.oblivious_trees.iter()).enumerate() {
        assert_eq!(ta.splits, tb.splits, "[{context}] tree {i} split structure differs");
        assert_eq!(
            ta.leaf_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            tb.leaf_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "[{context}] tree {i} leaf values differ (not even bit-identical)"
        );
        assert_eq!(
            ta.leaf_weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            tb.leaf_weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "[{context}] tree {i} leaf weights differ"
        );
    }
}

/// Build the mid-run fixture described in the module docs.
fn refingerprinted_k_run(name: &str, n_total: usize, k: usize) -> SnapshotConfig {
    let (values, borders, target, weights) = corpus();

    // 1. A genuine K-tree checkpoint.
    let k_path = snap_path(&format!("{name}_k"));
    let k_cfg =
        SnapshotConfig { snapshot_file: k_path.clone(), snapshot_interval: Duration::ZERO };
    let (_, k_resume_from) =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params(k), &k_cfg)
            .expect("the K-iteration run must succeed");
    assert_eq!(k_resume_from, 0, "the fixture-building run starts fresh");

    let mut snap = snapshot::read_from(&k_path).expect("the K-run must leave a checkpoint");
    assert_eq!(snap.completed_iters, k, "the K-run's last write must be its K-th tree");

    // 2/3. Re-fingerprint for the N-iteration run and write the fixture.
    snap.fingerprint =
        snapshot::fingerprint(&params(n_total), target.len(), &borders, &target, &weights);
    let resume_path = snap_path(&format!("{name}_resume"));
    snapshot::write_atomic(&resume_path, &snap).expect("the fixture must be writable");
    let _ = std::fs::remove_file(&k_path);

    SnapshotConfig { snapshot_file: resume_path, snapshot_interval: Duration::ZERO }
}

/// AT-S6, the keystone: for several `(N, K)` splits, resuming at K yields EXACTLY
/// the straight-through N-iteration model.
#[test]
fn resume_equals_straight_through() {
    let (values, borders, target, weights) = corpus();

    for &(n_total, k) in &[(4usize, 1usize), (4, 2), (6, 3)] {
        let name = format!("n{n_total}k{k}");
        let context = format!("N={n_total}, K={k}");

        // The reference: a single uninterrupted run through the ordinary entry.
        let straight =
            train(&CpuBackend, &values, &borders, &target, &weights, &params(n_total), None)
                .expect("the straight-through run must succeed");
        assert_eq!(straight.oblivious_trees.len(), n_total);

        // The resumed run, pointed at a K-tree checkpoint.
        let cfg = refingerprinted_k_run(&name, n_total, k);
        let (resumed, resumed_from) = train_with_snapshot(
            &CpuBackend,
            &values,
            &borders,
            &target,
            &weights,
            &params(n_total),
            &cfg,
        )
        .expect("the resumed run must succeed");
        assert_eq!(resumed_from, k, "[{context}] the entry must report the resume point");

        assert_models_identical(&resumed, &straight, &context);

        // The checkpoint left behind reports the full run, proving the resumed run
        // really did grow the remaining trees rather than stopping early.
        let after = snapshot::read_from(&cfg.snapshot_file).expect("a final checkpoint");
        assert_eq!(after.completed_iters, n_total, "[{context}] the resumed run must finish N");

        let _ = std::fs::remove_file(&cfg.snapshot_file);
    }
}

/// The anti-false-pass companion: a checkpoint at K must actually SKIP K
/// iterations. Without the loop-bound edit the resumed run would retrain from 0 and
/// still produce a correct-looking model — so equality alone cannot prove resume
/// happened. Here the checkpoint's trees are deliberately CORRUPTED; a genuine
/// resume carries the corruption into the final model, while a silent restart-from-0
/// would wash it away.
#[test]
fn a_resume_genuinely_starts_at_k_rather_than_retraining() {
    let (values, borders, target, weights) = corpus();
    let n_total = 4;
    let k = 2;

    let cfg = refingerprinted_k_run("detect", n_total, k);
    let mut snap = snapshot::read_from(&cfg.snapshot_file).unwrap();

    // Corrupt the FIRST persisted tree's leaf values. A run that restarts from 0
    // would regrow that tree and never show this.
    let marker = 123.456_f64;
    for v in &mut snap.trees[0].leaf_values {
        *v = marker;
    }
    snapshot::write_atomic(&cfg.snapshot_file, &snap).unwrap();

    let (resumed, resumed_from) =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params(n_total), &cfg)
            .expect("the resumed run must succeed");
    assert_eq!(resumed_from, k, "the corrupted checkpoint must still be resumed at K");

    assert!(
        resumed.oblivious_trees[0].leaf_values.iter().all(|v| (*v - marker).abs() < 1e-12),
        "the resumed model must carry the checkpoint's (corrupted) first tree verbatim — if it \
         does not, the loop retrained from iteration 0 and `resume_from` is not being honored"
    );
    assert_eq!(resumed.oblivious_trees.len(), n_total);

    let _ = std::fs::remove_file(&cfg.snapshot_file);
}

/// AT-S6: with no snapshot file present the run is an ordinary fresh fit — the
/// resume path must not fabricate state out of a missing file.
#[test]
fn no_file_means_a_fresh_run() {
    let (values, borders, target, weights) = corpus();
    let cfg = SnapshotConfig {
        snapshot_file: snap_path("fresh"),
        snapshot_interval: Duration::ZERO,
    };
    assert!(!cfg.snapshot_file.exists());

    let (fresh, fresh_from) =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params(4), &cfg)
            .expect("a fresh snapshot run must succeed");
    assert_eq!(fresh_from, 0, "a missing file must not fabricate a resume point");
    let straight = train(&CpuBackend, &values, &borders, &target, &weights, &params(4), None)
        .expect("the straight-through run must succeed");

    assert_models_identical(&fresh, &straight, "no-file fresh run");
    let _ = std::fs::remove_file(&cfg.snapshot_file);
}

/// AT-S4 end-to-end: a checkpoint from a DIFFERENT configuration is refused, not
/// silently ignored and not silently resumed. Silently starting fresh would discard
/// work the caller believes is continuing; silently resuming would mix two
/// configurations into one model.
#[test]
fn a_mismatched_fingerprint_is_refused() {
    let (values, borders, target, weights) = corpus();
    let cfg = refingerprinted_k_run("mismatch", 4, 2);

    // Point the same checkpoint at a run with a different learning rate.
    let mut other = params(4);
    other.learning_rate = 0.7;

    match train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &other, &cfg) {
        Err(CbError::Snapshot(msg)) => {
            assert!(
                msg.contains("fingerprint"),
                "the rejection must name the fingerprint mismatch, got: {msg}"
            );
        }
        other => panic!("expected a typed fingerprint rejection, got {other:?}"),
    }

    let _ = std::fs::remove_file(&cfg.snapshot_file);
}

/// ORCH-03-S7 / AT-S7: the public entry REPORTS where it resumed from. Without
/// this, a caller cannot distinguish "resumed at K" from "retrained from scratch" —
/// the two produce the same model, so the report is the only observable difference.
#[test]
fn train_with_snapshot_reports_where_it_resumed_from() {
    let (values, borders, target, weights) = corpus();

    // Fresh file: nothing to resume from.
    let fresh_cfg = SnapshotConfig {
        snapshot_file: snap_path("reports_fresh"),
        snapshot_interval: Duration::ZERO,
    };
    let (fresh_model, fresh_from) =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params(4), &fresh_cfg)
            .expect("a fresh run must succeed");
    assert_eq!(fresh_from, 0, "with no prior checkpoint the run starts at iteration 0");
    assert_eq!(fresh_model.oblivious_trees.len(), 4);

    // A K-tree checkpoint: the entry reports K.
    let k = 3;
    let cfg = refingerprinted_k_run("reports_k", 6, k);
    let (resumed, resumed_from) =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params(6), &cfg)
            .expect("the resumed run must succeed");
    assert_eq!(resumed_from, k, "the entry must report the checkpoint's completed_iters");
    assert_eq!(resumed.oblivious_trees.len(), 6);

    // Calling it AGAIN now finds a complete 6-tree checkpoint and has nothing left
    // to do — the resume point equals the iteration count, and the model is the same.
    let (again, again_from) =
        train_with_snapshot(&CpuBackend, &values, &borders, &target, &weights, &params(6), &cfg)
            .expect("a fully-completed checkpoint is still resumable (as a no-op)");
    assert_eq!(again_from, 6, "a complete checkpoint leaves no iterations to run");
    assert_models_identical(&again, &resumed, "re-resume of a complete checkpoint");

    let _ = std::fs::remove_file(&fresh_cfg.snapshot_file);
    let _ = std::fs::remove_file(&cfg.snapshot_file);
}
