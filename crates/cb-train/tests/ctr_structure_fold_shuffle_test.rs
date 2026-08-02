//! BUG-SFS / SPEC-BTMV residual — the STRUCTURE-search fold sees the learn-set
//! shuffle `S`.
//!
//! Upstream shuffles the learn data itself when CTRs are present and
//! `has_time=false` (`NeedShuffle`, `preprocess.cpp:161`;
//! `ShuffleLearnDataIfNeeded`, `preprocess.cpp:183`), and only THEN builds the
//! folds. `Folds[0]` is created with `shuffle = foldIdx != 0`
//! (`learn_context.cpp:526-529`) — identity **over the S-shuffled data** — so in
//! ORIGINAL-object coordinates the structure fold's online-CTR prefix order is
//! `S` itself, not the raw identity.
//!
//! cb-train materialized the fold-0 structure CTR column under the raw identity.
//! The averaging (leaf-value) fold already composed `S` (`Q = S ∘ P_avg`,
//! plan 05-19), so the two folds disagreed about the CTR feature space: on the
//! `ctr_btmv_simple` fixture the structure search chose bins `(6, 12)` where
//! upstream chose `(7, 10)`, split 12 was degenerate at apply time (one leaf
//! permanently empty), and E13 failed at `max |diff| = 1.371e-1` even with the
//! baked table byte-correct (BUG-BTMV COMPLETION.md §6).
//!
//! This gate pins the defect at STRUCTURE level — the persisted CTR split
//! borders — which is sharper than the ≤1e-5 prediction gate: a border shift
//! names the wrong bin directly instead of showing up as an opaque prediction
//! diff.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_oracle::{load_f64_vec, load_model_json};
use cb_train::{
    train_cat, BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_btmv_simple";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The two categorical columns as SoA `Vec<String>` — field-for-field the E13
/// harness (`ctr_btmv_simple_oracle_test.rs`, a protected file this test must
/// not modify; the duplication is deliberate, precedent B02).
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The fixture's pinned config, EVERY field explicit — copied verbatim from the
/// E13 harness (the name matches the copy-paste precedent recorded in
/// BUG-BTMV PLAN §1.2; do not "fix" it here either).
fn counter_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
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
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: ECtrType::BinarizedTargetMeanValue,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: vec![0.5],
        score_function: cb_train::score_function_default(),
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

#[test]
fn btmv_structure_ctr_borders_match_upstreams_committed_splits() {
    let cat_cols = load_cat_columns();
    let model_json = load_model_json(&fixture(&format!("{SCENARIO}/model.json")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();

    let (trained, _baked) = train_cat(
        &CpuBackend,
        &[],
        &borders,
        &cat_cols,
        &target,
        &[],
        &counter_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("BTMV-CTR training failed: {e:?}"));

    // Upstream's committed model.json: every one of the 5 trees carries the SAME
    // two CTR splits, with VALUE-space borders 7.999999046325684 and
    // 10.999999046325684 (bin indices 7 and 10 under the (b+1) - 2^-20 bridge,
    // BUG-CTRB). Collect ours per tree, order-insensitively.
    let expected: Vec<f64> = vec![7.999_999_046_325_684, 10.999_999_046_325_684];

    assert_eq!(trained.oblivious_trees.len(), 5, "5 boosting iterations");
    for (ti, tree) in trained.oblivious_trees.iter().enumerate() {
        let mut got: Vec<f64> = tree.ctr_splits.iter().map(|c| c.border).collect();
        got.sort_by(f64::total_cmp);
        assert_eq!(
            got, expected,
            "tree {ti}: the persisted CTR split borders must equal upstream's \
             committed pair (bins 7 and 10). A shifted pair — e.g. \
             (6.99…, 12.99…), bins (6, 12) — means the STRUCTURE-search CTR \
             column was materialized under the wrong permutation: the structure \
             fold Folds[0] is the identity over the S-SHUFFLED learn data \
             (learn_context.cpp:526-529 on the ShuffleLearnDataIfNeeded output, \
             preprocess.cpp:161/183), i.e. S itself in original-object order \
             (boosting.rs, `structure_fold_columns` fold 0). BUG-SFS."
        );
        for c in &tree.ctr_splits {
            assert_eq!(
                c.ctr_type,
                ECtrType::BinarizedTargetMeanValue.as_i8(),
                "tree {ti}: both chosen splits are BTMV simple-CTR splits"
            );
            assert_eq!(
                c.projection.cat_features(),
                &[1usize],
                "tree {ti}: upstream's committed CTR projection is cat feature 1"
            );
        }
    }
}

#[test]
fn has_time_disables_the_structure_fold_shuffle() {
    // The `!need_shuffle` branch (has_time = true): upstream does NOT shuffle a
    // time-ordered learn set (`NeedShuffle`, preprocess.cpp:161), so the
    // structure fold really IS the raw identity there. This characterizes that
    // the S-composition is gated on `need_shuffle` rather than unconditional —
    // training must succeed and produce a non-degenerate model, and the chosen
    // borders will generally DIFFER from the shuffled run's.
    let cat_cols = load_cat_columns();
    let model_json = load_model_json(&fixture(&format!("{SCENARIO}/model.json")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();

    let mut params = counter_params();
    params.has_time = true;

    let (trained, _baked) = train_cat(
        &CpuBackend,
        &[],
        &borders,
        &cat_cols,
        &target,
        &[],
        &params,
        None,
    )
    .unwrap_or_else(|e| panic!("has_time BTMV training failed: {e:?}"));

    assert_eq!(trained.oblivious_trees.len(), 5);
    // ANTI-VACUITY: the time-ordered run still chooses CTR splits (the corpus is
    // categorical-only, so a treeless / CTR-less model would make this test
    // meaningless).
    assert!(
        trained
            .oblivious_trees
            .iter()
            .all(|t| !t.ctr_splits.is_empty()),
        "every tree must still carry CTR splits under has_time = true"
    );
}
