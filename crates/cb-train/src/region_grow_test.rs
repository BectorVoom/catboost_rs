//! Region CPU grower oracle (GPUT-18 / D-03a, Plan 12-02 Task 2). Region has NO
//! upstream CPU reference (Pitfall 1) — THIS grower establishes the frozen ≤1e-5
//! reference the device Region path (Plan 04) reproduces. Locks: a depth-`d` region
//! grows exactly `d + 1` leaves (NOT `2^d`); the frozen path
//! `(feature, border, direction)` + `leaf_of` reproduce bit-for-bit on re-run
//! (determinism); a degenerate root is a typed error (no panic); and the grown
//! `leaf_of` agrees bit-for-bit with the `cb_model` walk-until-diverge apply.
//!
//! Sibling `#[path]` mount (source/test separation, CLAUDE.md) of `tree.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::EScoreFunction;

use crate::tree::{region_grower, FeatureMatrix, GrownTree, Split};

/// A pinned two-level-separable fixture. `f0 > 1.5` peels off the `+3` gradient
/// pair first (they diverge), the surviving `{0,1,2,3}` frontier then splits on
/// `f0 > 0.5`, and the final `{2,3}` frontier is uniform (path terminates at
/// depth 2). Deterministic + reproducible.
fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let feature_values = vec![
        vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0], // f0
        vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0], // f1 (irrelevant to the grown path)
    ];
    let feature_borders = vec![vec![0.5_f64, 1.5], vec![0.5_f64]];
    let der1 = vec![-2.0_f64, -2.0, 0.0, 0.0, 3.0, 3.0];
    let weight = vec![1.0_f64; 6];
    (feature_values, feature_borders, der1, weight)
}

fn grow() -> GrownTree {
    let (fv, fb, der1, weight) = fixture();
    let matrix = FeatureMatrix::new(&fv, &fb);
    region_grower(&matrix, &der1, &weight, 0.0, 3, 1, 6, EScoreFunction::Cosine)
        .expect("region grows on the pinned fixture")
}

/// FROZEN Region structure oracle: a depth-2 region path with EXACTLY 3 leaves
/// (`depth + 1`, NEVER `2^depth == 4`), the pinned `(feature, border, direction)`
/// per level, and the per-object terminal bin.
#[test]
fn region_grower_frozen_structure_has_depth_plus_one_leaves() {
    let g = grow();

    // depth == 2 (two path levels).
    assert_eq!(g.splits.len(), 2, "depth-2 region has 2 levels");
    assert_eq!(g.region_directions.len(), 2);
    assert_eq!(g.region_one_hot, vec![false, false], "CPU grower emits no one-hot");

    // FROZEN path: level 0 peels the `f0 > 1.5` (+3) pair off (continue=false, i.e.
    // the NOT-passes child continues); level 1 splits `f0 > 0.5` (continue=true).
    assert_eq!(g.splits[0], Split { feature: 0, border: 1.5 });
    assert_eq!(g.splits[1], Split { feature: 0, border: 0.5 });
    assert_eq!(g.region_directions, vec![false, true]);

    // FROZEN leaf_of (bin per object): o4,o5 diverge at level 0 → bin 0; o0,o1
    // diverge at level 1 → bin 1; o2,o3 survive both → bin 2 (== depth).
    assert_eq!(g.leaf_of, vec![1, 1, 2, 2, 0, 0]);

    // A depth-d region has EXACTLY d+1 distinct leaves (the failure signal for the
    // "region is a node graph" bug is `2^depth == 4`).
    let leaf_count = g.leaf_of.iter().copied().max().unwrap() + 1;
    assert_eq!(leaf_count, g.splits.len() + 1, "d+1 leaves, never 2^d");
    assert_eq!(leaf_count, 3);
}

/// The frozen path reproduces BIT-FOR-BIT on re-run (determinism — the device
/// Region path must reproduce a stable CPU reference).
#[test]
fn region_grower_is_deterministic() {
    let a = grow();
    let b = grow();
    assert_eq!(a.splits, b.splits);
    assert_eq!(a.region_directions, b.region_directions);
    assert_eq!(a.region_one_hot, b.region_one_hot);
    assert_eq!(a.leaf_of, b.leaf_of);
}

/// A degenerate root (uniform gradient — no beneficial split) is a typed
/// [`crate::CbError::Degenerate`], never a panic (mirrors the leaf-wise contract).
#[test]
fn region_grower_degenerate_root_is_typed_error_not_panic() {
    let fv = vec![vec![0.0_f32, 1.0, 2.0, 3.0]];
    let fb = vec![vec![0.5_f64, 1.5, 2.5]];
    let der1 = vec![1.0_f64; 4]; // uniform → every split gain below the 1e-9 cutoff
    let weight = vec![1.0_f64; 4];
    let matrix = FeatureMatrix::new(&fv, &fb);
    let result = region_grower(&matrix, &der1, &weight, 0.0, 3, 1, 4, EScoreFunction::Cosine);
    assert!(result.is_err(), "uniform-gradient root yields no region path");
}

/// The grown `leaf_of` is EXACTLY the walk-until-diverge bin an in-crate replica of
/// the `AddRegionImpl` walk produces (the same walk `cb_model::apply::region_leaf`
/// runs). Lock the CPU grower ↔ apply agreement here without the dev-dep `cb_model`
/// diamond (the full lift + `cb_model::predict_raw` path is exercised by the
/// `tests/region_e2e_test.rs` integration oracle).
#[test]
fn region_grown_leaf_of_matches_the_walk() {
    let g = grow();
    let (fv, _fb, _der1, _weight) = fixture();
    let n = fv[0].len();

    // Replica walk: bin = 0; for each level, continue while `(value > border) ==
    // direction`, else break. Leaf = bin. (Identical to `region_leaf`.)
    for obj in 0..n {
        let mut bin = 0usize;
        for (level, split) in g.splits.iter().enumerate() {
            let value = f64::from(fv[split.feature][obj]);
            let passes = value > split.border;
            if passes == g.region_directions[level] {
                bin += 1;
            } else {
                break;
            }
        }
        assert_eq!(bin, g.leaf_of[obj], "obj {obj}: walk bin != grown leaf_of");
    }
}
