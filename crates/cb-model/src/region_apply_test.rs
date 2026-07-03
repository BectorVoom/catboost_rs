//! Region path-model apply + json round-trip tests (GPUT-18 / D-03a, Plan 12-02
//! Task 1). Locks the walk-until-diverge semantics of [`crate::TreeVariant::Region`]:
//! a depth-`d` region has EXACTLY `d + 1` leaves (NOT `2^d`), the walk breaks at the
//! FIRST direction mismatch and returns `leaf == bin`, a malformed region contributes
//! `0.0` (no panic), and save→load→apply is bit-identical.
//!
//! Sibling `#[path]` mount (source/test separation, CLAUDE.md) of `apply.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use crate::model::{ModelSplit, RegionLevel, RegionTree, Split};
use crate::{predict_raw, Model};

/// A depth-3 region on features {0,1,2}, each level `value > 0.5` with CONTINUE
/// direction `true` (descend while the object passes). Leaf values are the bin
/// index scaled: bin `k` → `10 * (k + 1)`.
fn depth3_region() -> RegionTree {
    let levels = (0..3)
        .map(|f| RegionLevel {
            split: ModelSplit::Float(Split {
                feature: f,
                border: 0.5,
            }),
            expected_direction: true,
            one_hot: false,
        })
        .collect();
    RegionTree {
        levels,
        // bin 0,1,2,3 → 10,20,30,40 (d + 1 == 4 leaves).
        leaf_values: vec![10.0, 20.0, 30.0, 40.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    }
}

fn region_model(tree: RegionTree) -> Model {
    Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: Vec::new(),
        region_trees: vec![tree],
        bias: 0.0,
        float_feature_borders: vec![vec![0.5], vec![0.5], vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// Feature columns for four probe objects (object-major intent, SoA columns):
/// - o0 diverges at level 0 (f0 <= 0.5)            → bin 0
/// - o1 diverges at level 1 (f0 > 0.5, f1 <= 0.5)  → bin 1
/// - o2 diverges at level 2 (f0,f1 > 0.5, f2 <= .5)→ bin 2
/// - o3 matches every direction                     → bin 3
fn probe_columns() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0, 1.0, 1.0], // feature 0
        vec![0.0, 0.0, 1.0, 1.0], // feature 1
        vec![0.0, 0.0, 0.0, 1.0], // feature 2
    ]
}

/// A depth-`d` region has EXACTLY `d + 1` leaf values (NOT `2^d`), and the walk
/// maps each probe object to `leaf == bin` (depth reached at divergence).
#[test]
fn region_depth_d_has_d_plus_one_leaves_and_walks_to_bin() {
    let tree = depth3_region();
    assert_eq!(tree.levels.len(), 3, "depth 3 region has 3 levels");
    assert_eq!(
        tree.leaf_values.len(),
        tree.levels.len() + 1,
        "depth-d region has d+1 leaves, never 2^d"
    );

    let model = region_model(tree);
    let preds = predict_raw(&model, &probe_columns());
    // bin 0,1,2,3 → 10,20,30,40 (bias 0).
    assert_eq!(preds, vec![10.0, 20.0, 30.0, 40.0]);
}

/// The walk BREAKS at the first direction mismatch: flipping level 0's expected
/// direction to `false` makes the "all-pass" object o3 diverge immediately (bin 0)
/// while the "f0<=0.5" object o0 now CONTINUES past level 0.
#[test]
fn region_walk_breaks_at_first_direction_mismatch() {
    let mut tree = depth3_region();
    tree.levels[0].expected_direction = false; // continue when f0 <= 0.5
    let model = region_model(tree);
    let preds = predict_raw(&model, &probe_columns());

    // o0 (f0<=0.5): level0 matches (continue) → bin1; level1 f1<=0.5 vs dir true
    //   → mismatch → break → bin 1 → 20.
    assert_eq!(preds[0], 20.0);
    // o3 (all >0.5): level0 f0>0.5 (true) != expected false → break → bin 0 → 10.
    assert_eq!(preds[3], 10.0);
}

/// A malformed region (a `leaf_values` too SHORT for the reachable bin) contributes
/// `0.0`, never panics (T-12-03 — checked `.get`, bounded walk).
#[test]
fn region_malformed_short_leaf_values_contributes_zero_no_panic() {
    let mut tree = depth3_region();
    tree.leaf_values = Vec::new(); // no leaves at all — every bin reads None → 0.0
    let model = region_model(tree);
    let preds = predict_raw(&model, &probe_columns());
    assert_eq!(preds, vec![0.0, 0.0, 0.0, 0.0], "malformed region → 0.0, no panic");
}

/// save → load → apply is bit-identical (the region json round-trip).
#[test]
fn region_json_round_trip_is_identical() {
    let model = region_model(depth3_region());
    let before = predict_raw(&model, &probe_columns());

    let path = std::env::temp_dir().join(format!(
        "region_apply_rt_{}_{}.json",
        std::process::id(),
        line!()
    ));
    crate::save_json(&model, &path).expect("save region json");
    let loaded = crate::load_json(&path).expect("load region json");
    let _ = std::fs::remove_file(&path);

    // The region tree survived the round-trip (structure + leaves).
    assert_eq!(loaded.region_trees.len(), 1);
    assert_eq!(loaded.region_trees[0].levels.len(), 3);
    assert_eq!(loaded.region_trees[0].leaf_values, vec![10.0, 20.0, 30.0, 40.0]);
    // Oblivious / non-symmetric paths stay empty (a model is all-region here).
    assert!(loaded.oblivious_trees.is_empty());
    assert!(loaded.non_symmetric_trees.is_empty());

    let after = predict_raw(&loaded, &probe_columns());
    assert_eq!(before, after, "save→load→apply is bit-identical");
}
