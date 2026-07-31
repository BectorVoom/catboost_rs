//! T04 / SPEC-OH-02 + SPEC-OH-03 — mixed-kind splits must apply in LEVEL order.
//!
//! # The defect this pins
//!
//! `Model::from_trained` built each tree's `Vec<ModelSplit>` kind-grouped — every
//! float split first, then every CTR split — discarding the trainer's
//! `GrownTree.level_kinds`, which records the true per-level interleaving.
//!
//! Meanwhile `leaf_index_for` (`apply.rs:208-215`) walks the STORED order and
//! treats split `i` as leaf-index bit `i`, and `.cbm` save/load preserve that order
//! 1:1. So for a tree whose level 0 is a CTR split and level 1 a float split, the
//! trainer computes leaf `0b10 = 2` while the model applies leaf `0b01 = 1` —
//! leaves 1 and 2 transposed. That is a genuine mis-prediction, not a cosmetic
//! ordering nit.
//!
//! The defect is LATENT rather than active: the committed CTR fixtures happen not
//! to put a CTR split at level 0, so `tensor_ctr_e2e_oracle_test` passes 3/3 both
//! before and after this fix. This test constructs the case the fixtures miss.
//!
//! # Why it is fixed here rather than deferred
//!
//! One-hot training must store mixed float/one-hot levels through the SAME lift.
//! Left unfixed, one-hot would inherit the identical transposition on day one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_model::{predict_raw_cat, Model as CbModel};
use cb_train::{CtrSplitSpec, LevelKind, Split};
use cb_train::{Model as TrainModel, ObliviousTree, TProjection};

/// The CTR-value border. With NO baked table the apply path takes the
/// not-found→empty branch: `calc_inference(0.0, 0.0, prior{0.5, 1.0}, 0.0, 1.0)`
/// = `(0.0 + 0.5) / (0.0 + 1.0) = 0.5`. A border of `0.9` therefore makes the CTR
/// split FAIL deterministically, with no fixture dependency.
const CTR_BORDER: f64 = 0.9;

/// Build a depth-2 tree whose level 0 is a CTR split and level 1 a float split.
///
/// `leaf_values` are distinct per leaf so a transposition is unambiguous:
/// leaf 0 → 10.0, leaf 1 → 20.0, leaf 2 → 30.0, leaf 3 → 40.0.
fn ctr_at_level_zero_tree() -> TrainModel {
    let tree = ObliviousTree {
        // Kind-grouped storage (unchanged): one float split, one CTR split.
        splits: vec![Split {
            feature: 0,
            border: 0.0,
        }],
        ctr_splits: vec![CtrSplitSpec {
            projection: TProjection::single(0),
            ctr_type: 0,
            prior_num: 0.5,
            prior_denom: 1.0,
            target_border_idx: 0,
            border: CTR_BORDER,
            shift: 0.0,
            scale: 1.0,
        }],
        one_hot_splits: Vec::new(),
        // The TRUE level order: CTR is level 0 (bit 0), float is level 1 (bit 1).
        level_kinds: vec![
            LevelKind::Ctr {
                ctr_idx: 0,
                border: CTR_BORDER,
            },
            LevelKind::Float(0),
        ],
        leaf_values: vec![10.0, 20.0, 30.0, 40.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    };

    TrainModel {
        oblivious_trees: vec![tree],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// SPEC-OH-03: the applied leaf must equal the trainer's `leaf_of` for the same
/// object.
///
/// The object FAILS the CTR split (level 0 → bit 0 = 0) and PASSES the float split
/// (level 1 → bit 1 = 1), so the trainer's leaf is `0b10 = 2` → `30.0`.
///
/// Under the pre-fix kind-grouped order the model stored `[Float, Ctr]`, making the
/// float split bit 0 and the CTR split bit 1 → leaf `0b01 = 1` → `20.0`. Observing
/// `20.0` here IS the defect.
#[test]
fn ctr_at_level_zero_float_at_level_one_applies_to_the_trainer_leaf() {
    let trained = ctr_at_level_zero_tree();
    let model = CbModel::from_trained(&trained, vec![vec![0.0]]);

    // One object: float feature 0 = 1.0 (> border 0.0 → PASSES the float split).
    // Its categorical value is irrelevant — with no baked ctr_data the CTR split
    // takes the deterministic not-found branch and FAILS against CTR_BORDER.
    let feature_values = vec![vec![1.0f32]];
    let cat_columns = vec![vec!["a".to_owned()]];

    let preds = predict_raw_cat(&model, &feature_values, &cat_columns);
    assert_eq!(preds.len(), 1, "one object in, one prediction out");

    assert!(
        (preds[0] - 30.0).abs() < 1e-12,
        "mixed-kind splits must apply in LEVEL order. Expected leaf 0b10 = 2 \
         (CTR fails at level 0, float passes at level 1) -> 30.0, got {}. \
         A value of 20.0 means the splits were stored kind-grouped as \
         [Float, Ctr], transposing leaves 1 and 2 — the SPEC-OH-02 defect.",
        preds[0]
    );
}

/// The stored `Vec<ModelSplit>` itself must be in level order — the property the
/// apply assertion above depends on, checked directly so a failure localizes to the
/// lift rather than the apply path.
#[test]
fn from_trained_stores_mixed_kind_splits_in_level_order() {
    let trained = ctr_at_level_zero_tree();
    let model = CbModel::from_trained(&trained, vec![vec![0.0]]);

    let splits = &model.oblivious_trees[0].splits;
    assert_eq!(splits.len(), 2, "depth-2 tree stores two splits");
    assert!(
        matches!(splits[0], cb_model::ModelSplit::Ctr(_)),
        "level 0 is a CTR split, so it must be STORED first (bit 0); got {:?}",
        splits[0]
    );
    assert!(
        matches!(splits[1], cb_model::ModelSplit::Float(_)),
        "level 1 is a float split, so it must be STORED second (bit 1); got {:?}",
        splits[1]
    );
}

/// SPEC-OH-02 invariant: a tree with EMPTY `level_kinds` (every single-kind tree,
/// which is every tree the float and device paths produce) keeps the legacy
/// kind-grouped assembly, byte-identical to pre-change. This is the structural
/// guard for SPEC-OH-31 at the lift.
#[test]
fn empty_level_kinds_keeps_the_byte_identical_legacy_order() {
    let tree = ObliviousTree {
        splits: vec![
            Split {
                feature: 0,
                border: 0.5,
            },
            Split {
                feature: 1,
                border: 1.5,
            },
        ],
        ctr_splits: Vec::new(),
        one_hot_splits: Vec::new(),
        // EMPTY: the legacy path must be taken.
        level_kinds: Vec::new(),
        leaf_values: vec![1.0, 2.0, 3.0, 4.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    };
    let trained = TrainModel {
        oblivious_trees: vec![tree],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    };

    let model = CbModel::from_trained(&trained, vec![vec![0.5], vec![1.5]]);
    let splits = &model.oblivious_trees[0].splits;
    assert_eq!(splits.len(), 2);
    for (i, split) in splits.iter().enumerate() {
        match split {
            cb_model::ModelSplit::Float(s) => {
                assert_eq!(s.feature, i, "float splits keep their original order");
            }
            other => panic!("expected a float split at index {i}, got {other:?}"),
        }
    }
}
