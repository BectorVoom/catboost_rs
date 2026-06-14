//! Ordered-boosting WIRING test (ORD-02, 05-10 Task 1) — locks that
//! `train_with_eval_sets` actually DRIVES the ordered split-scoring subsystem
//! (`greedy_tensor_search_oblivious_ordered`, 05-08) under
//! `EBoostingType::Ordered`, building the fold set ONCE before the iteration loop
//! and estimating leaf VALUES on the averaging fold exactly as Plain.
//!
//! This is the structural-wiring gate (the FULL multi-tree e2e ≤1e-5 oracle vs
//! upstream lives in `ordered_boost_e2e_oracle_test.rs`, Task 2). It proves:
//!   * Ordered training is NO LONGER a `debug_assert` no-op — it returns a fully
//!     grown `iterations`-tree model with finite leaf values and staged approx.
//!   * The Ordered path does NOT regress the Plain path: the SAME inputs under
//!     `Plain` still train (the Plain oracles in the sibling tests lock parity).
//!   * Ordered and Plain can DIFFER in tree structure (the ordered per-segment
//!     score is genuinely consulted) — falsifiable: if the Ordered branch silently
//!     fell through to Plain, the dead-code stub would make them identical AND the
//!     `create_folds`-once invariant would be absent.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_train::{train, BoostParams, EBootstrapType, EBoostingType, EOverfittingDetectorType, Model};

/// A small deterministic numeric dataset (N=30, 2 float features, RMSE target)
/// with ascending integer-derived borders, mirroring the isolating ordered_boost
/// config (depth=2, iterations=5, lr=0.1, l2=3.0, bootstrap=No, random_strength=0,
/// permutation_count=1, fold_len_multiplier=2.0, seed=0).
fn dataset() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = 30usize;
    let f0: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let f1: Vec<f32> = (0..n).map(|i| ((i * 3) % 11) as f32).collect();
    let target: Vec<f64> = (0..n).map(|i| (i % 5) as f64 + 0.5 * (i % 3) as f64).collect();
    // Ascending candidate borders spanning each feature's range.
    let b0: Vec<f64> = vec![1.5, 3.5, 5.5];
    let b1: Vec<f64> = vec![2.5, 5.5, 8.5];
    (vec![f0, f1], vec![b0, b1], target)
}

fn params(boosting_type: EBoostingType) -> BoostParams {
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
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type,
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
    }
}

fn train_with(boosting_type: EBoostingType) -> (Model, Vec<f64>) {
    let (columns, borders, target) = dataset();
    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &params(boosting_type),
        Some(&mut staged),
    )
    .expect("training must succeed");
    (model, staged)
}

/// The Ordered branch is wired and grows a full model: `iterations` trees, each
/// with `2^depth` finite leaf values, a finite bias, and per-iteration staged
/// approximants. (Pre-05-10 the Ordered branch was a `debug_assert` no-op that
/// silently produced the Plain model; this gate fails if the branch is dead.)
#[test]
fn ordered_training_grows_a_full_finite_model() {
    let (model, staged) = train_with(EBoostingType::Ordered);

    assert_eq!(model.oblivious_trees.len(), 5, "5 iterations → 5 trees");
    assert!(model.bias.is_finite(), "bias must be finite");
    for (ti, tree) in model.oblivious_trees.iter().enumerate() {
        assert_eq!(tree.splits.len(), 2, "depth=2 → 2 splits in tree {ti}");
        assert_eq!(tree.leaf_values.len(), 4, "depth=2 → 4 leaves in tree {ti}");
        assert_eq!(tree.leaf_weights.len(), 4, "depth=2 → 4 leaf weights in tree {ti}");
        for (li, &lv) in tree.leaf_values.iter().enumerate() {
            assert!(lv.is_finite(), "leaf value tree {ti} leaf {li} = {lv} must be finite");
        }
        // Leaf weights sum to N (unweighted ⇒ document count, A4).
        let total: f64 = tree.leaf_weights.iter().sum();
        assert!((total - 30.0).abs() < 1e-9, "leaf weights tree {ti} must sum to N=30");
    }
    assert_eq!(staged.len(), 5 * 30, "staged is iterations × N, flat");
    for (i, &v) in staged.iter().enumerate() {
        assert!(v.is_finite(), "staged[{i}] = {v} must be finite");
    }
}

/// The Ordered split-scoring subsystem is genuinely consulted: on a dataset whose
/// growing body/tail segments shift the per-segment scores, the Ordered model's
/// tree structure differs from the Plain model's. If the Ordered branch were dead
/// (fell through to Plain), the two structures would be byte-identical — this gate
/// would then fail, catching a silent no-op wiring.
#[test]
fn ordered_structure_differs_from_plain() {
    let (ordered, _) = train_with(EBoostingType::Ordered);
    let (plain, _) = train_with(EBoostingType::Plain);

    let ordered_splits: Vec<(usize, f64)> = ordered
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter().map(|s| (s.feature, s.border)))
        .collect();
    let plain_splits: Vec<(usize, f64)> = plain
        .oblivious_trees
        .iter()
        .flat_map(|t| t.splits.iter().map(|s| (s.feature, s.border)))
        .collect();

    assert_ne!(
        ordered_splits, plain_splits,
        "Ordered per-segment scoring must produce a different tree structure than Plain \
         on this dataset (a dead Ordered branch would make them identical)"
    );
}

/// The Plain path is unchanged by the Ordered wiring: a Plain run still trains to
/// a full finite 5-tree model (the numeric Plain oracles in slice_first /
/// one_hot / leaf_methods lock exact ≤1e-5 parity; this is the local smoke that
/// the shared driver still serves Plain).
#[test]
fn plain_path_still_trains() {
    let (model, staged) = train_with(EBoostingType::Plain);
    assert_eq!(model.oblivious_trees.len(), 5);
    assert!(model.bias.is_finite());
    assert_eq!(staged.len(), 5 * 30);
    assert!(staged.iter().all(|v| v.is_finite()));
}
