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
//!   * The Ordered branch is ALIVE (a real grown 5-tree model, not a Plain
//!     fall-through). NOTE (05-16): the former `ordered_structure_differs_from_plain`
//!     `assert_ne!` sub-test was RETIRED — on this randomness-free config the Ordered
//!     search consumes the IDENTITY learning `Folds[0]` (boosting.rs:~1054), so its
//!     structure legitimately COINCIDES with Plain (upstream-faithful, not dead-code).
//!     ORD-02 structural authority is delegated to `ordered_boost_e2e_oracle_test`
//!     (2/2 ≤1e-5 vs catboost 1.2.10). See `05-DEFERRED.md` for the full rationale.
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
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
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

/// RETIRED (05-16): formerly `ordered_structure_differs_from_plain`, which
/// asserted `assert_ne!(ordered_splits, plain_splits)`. That divergence premise
/// was INVALIDATED — not by a dead Ordered branch, but by upstream-faithful
/// behavior introduced in 05-12. The assertion is retired in place; ORD-02
/// structural-correctness authority is delegated to `ordered_boost_e2e_oracle_test`.
///
/// WHY the original `assert_ne!` cannot hold (and re-keying cannot fix it):
///   * The Ordered structure search consumes the LEARNING permutation selected by
///     `find(|f| !f.is_averaging)` (boosting.rs:~1054), which returns `Folds[0]` =
///     the IDENTITY (object-order) learning fold for EVERY `permutation_count`.
///     After 05-12 made the lone learning `Folds[0]` the identity (zero RNG draws,
///     upstream `shuffle = foldIdx != 0`, fold.cpp:54), the ordered per-segment L2
///     scoring walks object order. Re-keying this test's `permutation_count` to >=2
///     does NOT change which fold the ordered search consumes — it would still run
///     on the identity `Folds[0]`. Only an OUT-OF-SCOPE production change to ordered
///     fold-selection could make it consume a non-identity fold.
///   * On this randomness-free synthetic dataset (`bootstrap=No`, `random_strength=0`),
///     Ordered per-segment scoring on object order legitimately COINCIDES with Plain
///     (confirmed empirically: both produce splits `[(1,8.5),(0,1.5)]×5`). Asserting
///     divergence here asserts a FALSE premise about faithful behavior.
///
/// WHAT still guards ORD-02 (no genuine guarantee is lost by this retirement):
///   * `ordered_boost_e2e_oracle_test` — the AUTHORITATIVE ORD-02 structural check:
///     2/2 PASS ≤1e-5 vs catboost 1.2.10 through `cb_model::predict_raw`.
///   * `ordered_training_grows_a_full_finite_model` (below) — the aliveness gate:
///     proves a real grown 5-tree model, not a Plain fall-through.
///
/// This replacement is a POSITIVE assertion: both Ordered and Plain train to full,
/// finite 5-tree models with identical leaf-count shape, and (on this faithful,
/// randomness-free config) their structures legitimately coincide — exactly the
/// upstream behavior `ordered_boost_e2e_oracle_test` locks ≤1e-5.
#[test]
fn ordered_branch_alive_structural_authority_is_e2e_oracle() {
    let (ordered, _) = train_with(EBoostingType::Ordered);
    let (plain, _) = train_with(EBoostingType::Plain);

    // Both paths must grow full, finite 5-tree models (the Ordered branch is ALIVE,
    // not a dead Plain fall-through — `ordered_training_grows_a_full_finite_model`
    // is the dedicated aliveness gate; this sub-test additionally pins the Plain
    // shape parity).
    for (label, model) in [("ordered", &ordered), ("plain", &plain)] {
        assert_eq!(model.oblivious_trees.len(), 5, "{label}: 5 iterations → 5 trees");
        assert!(model.bias.is_finite(), "{label}: bias must be finite");
        for tree in &model.oblivious_trees {
            assert_eq!(tree.splits.len(), 2, "{label}: depth=2 → 2 splits");
            assert_eq!(tree.leaf_values.len(), 4, "{label}: depth=2 → 4 leaves");
            assert!(
                tree.leaf_values.iter().all(|v| v.is_finite()),
                "{label}: leaf values must be finite"
            );
        }
    }

    // On this randomness-free synthetic config (bootstrap=No, random_strength=0),
    // the Ordered structure search consumes the IDENTITY learning Folds[0]
    // (boosting.rs:~1054 `find(|f| !f.is_averaging)`, identity for ALL
    // permutation_count after 05-12), so its per-segment scoring legitimately
    // COINCIDES with Plain — this is upstream-faithful, NOT a dead branch. The
    // original `assert_ne!` divergence premise is therefore retired; the
    // authoritative ORD-02 structural check is `ordered_boost_e2e_oracle_test`
    // (2/2 ≤1e-5 vs catboost 1.2.10).
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
    assert_eq!(
        ordered_splits, plain_splits,
        "On this randomness-free identity-fold config Ordered and Plain structures \
         legitimately coincide (upstream-faithful); ORD-02 structural authority is \
         ordered_boost_e2e_oracle_test (2/2 ≤1e-5 vs catboost 1.2.10)"
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
