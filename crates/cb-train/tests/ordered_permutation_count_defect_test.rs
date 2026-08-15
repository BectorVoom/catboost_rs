//! The float-only `boosting_type=Ordered` defect, pinned so it cannot be forgotten.
//!
//! # What is wrong
//!
//! Upstream ordered boosting carries `learning_fold_count(permutation_count, true)`
//! LEARNING permutations and selects a structure fold per iteration
//! (`takenFold[iter] = Folds[Rand.GenRand() % learning_folds]`, `train.cpp:208`).
//! Its predictions therefore CHANGE with `permutation_count` — measured against
//! catboost 1.2.10 on a 300x4 float corpus, `pc=1` and `pc=2` agree (both resolve to
//! ONE learning fold) while `pc=4` (three folds) differs.
//!
//! This engine's float-only ordered path is INSENSITIVE to `permutation_count`:
//! [`cb_train::fold::create_folds`] gives every learning fold the IDENTITY
//! permutation (correct only relative to upstream's already-shuffled learn data),
//! and `train_inner` always reads the FIRST non-averaging fold. So changing
//! `permutation_count` changes the fold COUNT and nothing else, and the model is
//! byte-identical across values.
//!
//! # Why this test asserts the WRONG behaviour on purpose
//!
//! It follows the `known_divergences_still_diverge` pattern already used by
//! `string_param_matrix_test`: the defect is recorded as an executable fact, so the
//! moment someone makes the ordered path honour `permutation_count` this test FAILS
//! and forces the fix to be completed (drop this file, add a parity oracle). A
//! comment alone would rot; a self-correcting test cannot.
//!
//! # What is already known about the root cause (so the next session does not re-derive it)
//!
//! * The PLAIN path is exact (max |diff| = 2.2e-16 vs catboost 1.2.10) — the defect is
//!   ordered-specific.
//! * The divergence is present in the FIRST tree (`iterations = 1`), so it is not
//!   accumulated drift.
//! * When our answer happens to match upstream it matches EXACTLY (0.0 / 1.1e-16) —
//!   the arithmetic and the apply path are right; only the STRUCTURE selection differs.
//! * `has_time` does NOT separate the cases (it matches at some shapes with the flag
//!   and at others without), so "upstream shuffles and we do not" is too simple.
//! * Composing the learning permutation with `create_shuffled_indices(n, seed)` was
//!   TRIED and makes parity WORSE (it turns previously-exact shapes into
//!   divergences), so upstream's learn-set shuffle for this path is NOT that stream.
//! * The committed `ordered_boost_e2e` oracle passes because it pins
//!   `permutation_count = 1` on a 30-row corpus where the identity permutation
//!   coincides with upstream's — it is NOT evidence that the path is right.
//!
//! # What has been tried, and measured (so it is not re-derived)
//!
//! The cycling itself has been WIRED and measured, twice, and reverted both times.
//! The mechanism is not the hard part — the per-fold PERMUTATIONS are:
//!
//! * `structure_fold_cycle` (instrument-derived, pc=4/seed=0 -> `[0,2,0,2,2]`) wires
//!   in cleanly, and the invariant it must satisfy HOLDS: at `iterations = 1` the
//!   cycle takes fold 0, and pc=4 then reproduces pc=1 EXACTLY (max |diff| = 0).
//! * Per-fold, per-body/tail approximants were added alongside it, with every fold's
//!   trajectory advanced each iteration over its own permutation (a fold left stale
//!   would score a later tree from an out-of-date trajectory).
//! * Two candidate rules for fold `j >= 1`'s object order were measured over a 60-cell
//!   corpus x iteration grid at pc=4, against the 21/60 baseline of NOT cycling:
//!     - `[S[p] for p in stream[j]]` (the rule the CTR path documents): **19/60**
//!     - `stream[j]` uncomposed:                                        **12/60**
//!   Both are WORSE than not cycling at all, so neither is upstream's rule and
//!   neither was shipped. pc=1 stayed at 44/60 throughout — the cycling work does not
//!   regress the single-fold path, it just does not improve the multi-fold one.
//!
//! So the missing piece is specifically **what permutation each learning fold beyond
//! the first carries**, which needs upstream's exact fold-permutation RNG stream via
//! the instrumented-CLI draw-accounting workflow.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{
    train, BoostParams, EBootstrapType, EBoostingType, EGrowPolicy, EOverfittingDetectorType,
};

fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = 300;
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let mut cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n); 4];
    let mut target = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<f64> = (0..4).map(|_| next()).collect();
        for (f, c) in cols.iter_mut().enumerate() {
            c.push(v[f] as f32);
        }
        target.push(3.0 * v[0] + 2.0 * v[1] - v[2] + 0.5 * v[3]);
    }
    let borders: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| cb_data::select_borders_greedy_logsum_f32(c, 32, false))
        .collect();
    let weights = vec![1.0_f64; n];
    (cols, borders, target, weights)
}

fn params(boosting_type: EBoostingType, permutation_count: usize) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 5,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
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
        permutation_count,
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type,
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
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

fn leaves(p: &BoostParams) -> Vec<f64> {
    let (cols, borders, target, weights) = corpus();
    let model =
        train(&CpuBackend, &cols, &borders, &target, &weights, p, None).expect("fit must succeed");
    model
        .oblivious_trees
        .iter()
        .flat_map(|t| t.leaf_values.iter().copied())
        .collect()
}

/// KNOWN DEFECT — asserts the WRONG behaviour on purpose.
///
/// Upstream's float-only ordered predictions change between `permutation_count = 2`
/// and `4`; ours do not. When this test starts FAILING, the ordered path has begun
/// honouring `permutation_count` — finish the job: delete this file and add a real
/// parity oracle against catboost 1.2.10 at several permutation counts.
#[test]
fn known_defect_float_only_ordered_ignores_permutation_count() {
    let one = leaves(&params(EBoostingType::Ordered, 1));
    let four = leaves(&params(EBoostingType::Ordered, 4));
    assert!(
        !one.is_empty(),
        "vacuous: the ordered fit produced no leaves at all"
    );
    assert_eq!(
        one, four,
        "float-only ordered boosting is now SENSITIVE to permutation_count — the known \
         defect this test pins has been fixed. Delete this file and replace it with a \
         parity oracle against catboost 1.2.10 (which gives DIFFERENT predictions at \
         pc=1/2 versus pc=4 on a 300x4 float corpus)."
    );
}

/// The control that keeps the test above honest: `permutation_count` must be a knob
/// that CAN matter. It is genuinely consumed on the CTR path (verified through the
/// public Python surface: pc=1/2 agree and pc=4 differs on a categorical pool), so
/// the insensitivity above is specific to the float-only ORDERED path rather than
/// the parameter being unwired everywhere.
///
/// Asserted here as the PLAIN baseline being permutation-invariant BY DESIGN — Plain
/// boosting has no learning permutation at all, so this is the intended behaviour and
/// not a defect.
#[test]
fn plain_boosting_is_permutation_count_invariant_by_design() {
    let one = leaves(&params(EBoostingType::Plain, 1));
    let four = leaves(&params(EBoostingType::Plain, 4));
    assert_eq!(
        one, four,
        "Plain boosting has no learning permutation, so permutation_count must be inert"
    );
}
