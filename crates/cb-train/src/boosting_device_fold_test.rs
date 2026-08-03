//! Unit self-oracle for the Phase 12 Plan 03 (GPUT-18) device-fold NON-SYMMETRIC arm:
//! the transcribed `device_leaf_of_nonsym` pointer-walk (the load-bearing piece of the
//! `boosting.rs` `!dev_tree.step_nodes.is_empty()` fold arm) must assign each object to the
//! SAME distinct leaf as an independent replica of `cb_model::apply::leaf_index_nonsym` over
//! the identical hand-built node graph.
//!
//! Mounted as a sibling `#[path]` submodule of `boosting` (source/test separation, CLAUDE.md),
//! so it reaches the private `super::device_leaf_of_nonsym` + `super::Split` directly. The
//! END-TO-END "folds into `non_symmetric_trees`, `trees` stays empty" assertion runs through
//! `cb_train::train()` in `device_nonsym_fit_test.rs` (Task 3) — a src-mounted unit test
//! cannot instantiate `cb_model` (the `cb_train` dev-dep diamond, 12-02 SUMMARY), so this
//! file replicates the walk inline as its reference rather than importing `cb_model`.

use super::{device_leaf_of_nonsym, Split};

/// An independent replica of the non-symmetric walk (`leaf_index_nonsym`): a bounded
/// flat-node walk over `step_nodes`, halting on the zero side and reading the distinct
/// leaf id. Written FRESH here (not calling `device_leaf_of_nonsym`) so the test is a real
/// cross-check, not a tautology.
fn replica_walk(
    obj: usize,
    splits: &[Split],
    step_nodes: &[(u16, u16)],
    node_id_to_leaf_id: &[u32],
    features: &[Vec<f32>],
) -> Option<usize> {
    let mut index: i64 = 0;
    for _ in 0..=step_nodes.len() {
        let idx = usize::try_from(index).ok()?;
        let &(left, right) = step_nodes.get(idx)?;
        let split = splits.get(idx)?;
        let v = *features.get(split.feature).and_then(|c| c.get(obj))?;
        let passes = f64::from(v) > split.border;
        let diff: i64 = if passes { i64::from(right) } else { i64::from(left) };
        index += diff;
        if diff == 0 {
            let leaf = *node_id_to_leaf_id.get(idx)?;
            if leaf == u32::MAX {
                return None;
            }
            return usize::try_from(leaf).ok();
        }
    }
    None
}

/// Build a fixed 5-node non-symmetric graph:
/// - node 0 (interior): `f0 > 0.5`, children (1, 2)  → step (1, 2)
/// - node 1 (interior): `f1 > 0.5`, children (3, 4)  → step (2, 3)
/// - node 2 (leaf, id 0), node 3 (leaf, id 1), node 4 (leaf, id 2)
///
/// Routing: `f0 > 0.5` → leaf 0; `f0 <= 0.5 & f1 <= 0.5` → leaf 1;
/// `f0 <= 0.5 & f1 > 0.5` → leaf 2.
fn fixture() -> (Vec<Split>, Vec<(u16, u16)>, Vec<u32>) {
    let splits = vec![
        Split { feature: 0, border: 0.5 },
        Split { feature: 1, border: 0.5 },
        Split { feature: 0, border: 0.0 }, // inert leaf placeholder
        Split { feature: 0, border: 0.0 },
        Split { feature: 0, border: 0.0 },
    ];
    let step_nodes = vec![(1u16, 2u16), (2, 3), (0, 0), (0, 0), (0, 0)];
    let node_id_to_leaf_id = vec![u32::MAX, u32::MAX, 0, 1, 2];
    (splits, step_nodes, node_id_to_leaf_id)
}

#[test]
fn device_leaf_of_nonsym_matches_replica_walk() {
    let (splits, step_nodes, node_id_to_leaf_id) = fixture();
    // 6 objects spanning all three leaves.
    let f0 = vec![1.0_f32, 2.0, 0.0, 0.0, 0.2, 0.9];
    let f1 = vec![0.0_f32, 9.0, 0.0, 1.0, 0.7, 0.1];
    let features = vec![f0, f1];
    let n = 6usize;

    let mut seen = [false; 3];
    for obj in 0..n {
        let got = device_leaf_of_nonsym(obj, &splits, &step_nodes, &node_id_to_leaf_id, &features);
        let want = replica_walk(obj, &splits, &step_nodes, &node_id_to_leaf_id, &features);
        assert_eq!(got, want, "walk disagreement at obj {obj}");
        let leaf = got.expect("every object must reach a valid leaf");
        assert!(leaf < 3, "leaf id {leaf} out of range at obj {obj}");
        if let Some(s) = seen.get_mut(leaf) {
            *s = true;
        }
    }
    // The fixture objects exercise all three distinct leaves.
    assert_eq!(seen, [true, true, true], "all three leaves must be reachable");

    // Spot-check the routing semantics explicitly (independent of the replica).
    // obj 0: f0=1.0 > 0.5 → leaf 0.
    assert_eq!(
        device_leaf_of_nonsym(0, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(0)
    );
    // obj 3: f0=0.0 <= 0.5, f1=1.0 > 0.5 → leaf 2.
    assert_eq!(
        device_leaf_of_nonsym(3, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(2)
    );
    // obj 4: f0=0.2 <= 0.5, f1=0.7 > 0.5 → leaf 2.
    assert_eq!(
        device_leaf_of_nonsym(4, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(2)
    );
    // obj 2: f0=0.0 <= 0.5, f1=0.0 <= 0.5 → leaf 1.
    assert_eq!(
        device_leaf_of_nonsym(2, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(1)
    );
}

#[test]
fn malformed_graph_yields_none_not_panic() {
    // A cyclic graph: node 0's left diff is 0 but its leaf slot is the interior sentinel
    // (`u32::MAX`) — the walk halts on the zero side but finds no real leaf id → None
    // (the caller substitutes a checked leaf-0 fallback, never a panic, T-12-05).
    let splits = vec![Split { feature: 0, border: 0.5 }];
    let step_nodes = vec![(0u16, 0u16)];
    let node_id_to_leaf_id = vec![u32::MAX];
    let features = vec![vec![0.0_f32]];
    assert_eq!(
        device_leaf_of_nonsym(0, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        None,
        "an interior-sentinel halt point must yield None, not a fabricated leaf"
    );

    // A self-loop (node 0 points back to itself with a non-zero diff on both sides) must be
    // rejected by the visit cap, not spin forever.
    let splits = vec![Split { feature: 0, border: 0.5 }, Split { feature: 0, border: 0.5 }];
    let step_nodes = vec![(1u16, 1u16), (u16::MAX, u16::MAX)];
    let node_id_to_leaf_id = vec![u32::MAX, u32::MAX];
    let features = vec![vec![0.0_f32]];
    assert_eq!(
        device_leaf_of_nonsym(0, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        None,
        "a walk exceeding the node-count cap must terminate as None"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// T23 / SPEC-OH-20 — a cat-only pool is device-eligible.
//
// `device_host_eligible` (`boosting.rs`) used to require `matrix.n_features() > 0`,
// i.e. at least one FLOAT column. A pool routed entirely one-hot has zero float
// columns and would therefore never reach the device grower, making SPEC-OH-20's
// 0-float target unreachable. Clause 11 is extracted into
// `has_any_scorable_feature` so it can be asserted directly — the full
// `device_host_eligible` expression needs a whole fit context.
// ─────────────────────────────────────────────────────────────────────────────

use super::has_any_scorable_feature;
use crate::tree::FeatureMatrix;

#[test]
fn cat_only_pool_with_one_hot_columns_is_device_eligible() {
    // (a) 0 float + 1 one-hot cat column — the SPEC-OH-20 target.
    let no_float: Vec<Vec<f32>> = Vec::new();
    let no_borders: Vec<Vec<f64>> = Vec::new();
    let cat_bins = vec![vec![0u32, 1, 0, 1]];
    let cat_only = FeatureMatrix {
        feature_values: &no_float,
        feature_borders: &no_borders,
        cat_bins: &cat_bins,
    };
    assert!(
        has_any_scorable_feature(&cat_only),
        "a pool with 0 float columns and 1 one-hot cat column must be scorable \
         (SPEC-OH-20); the old `n_features() > 0` clause made it unreachable"
    );

    // (b) 2 float + 0 cat — today's float-only path, unchanged.
    let floats = vec![vec![0.0_f32, 1.0], vec![2.0_f32, 3.0]];
    let borders = vec![vec![0.5_f64], vec![2.5_f64]];
    let float_only = FeatureMatrix::new(&floats, &borders);
    assert!(
        has_any_scorable_feature(&float_only),
        "the float-only path must be byte-unchanged (SPEC-OH-31)"
    );

    // (c) 0 float + 0 cat — a genuinely feature-less pool stays ineligible.
    let empty_cats: Vec<Vec<u32>> = Vec::new();
    let featureless = FeatureMatrix {
        feature_values: &no_float,
        feature_borders: &no_borders,
        cat_bins: &empty_cats,
    };
    assert!(
        !has_any_scorable_feature(&featureless),
        "a pool with no float AND no cat columns has nothing to score"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// T24 / SPEC-OH-21 — device quantization emits one-hot bin columns, and a
// SEPARATE per-feature real-cardinality array `real_folds`.
//
// `real_folds` is NOT `TCFeature.folds`: on the production path the latter is the
// padded uniform line width (`session.rs` packs `vec![n_bins_line; ..]`), which
// bounds nothing. Without a true per-feature cardinality the device scorer sweeps
// the whole padded line and a cardinality-2 column contributes 30 phantom
// "all-objects-right" candidates that can tie or beat a real one.
// ─────────────────────────────────────────────────────────────────────────────

use super::{quantize_feature_major, quantize_feature_major_with_one_hot};

#[test]
fn quantize_emits_one_hot_bin_columns_in_the_shared_bin_line() {
    let n = 4usize;
    let floats = vec![
        vec![0.0_f32, 1.0, 2.0, 3.0],
        vec![10.0_f32, 11.0, 12.0, 13.0],
    ];
    // 1 border -> 2 bins; 2 borders -> 3 bins.
    let borders = vec![vec![1.5_f64], vec![10.5_f64, 12.5]];
    // Two one-hot columns, cardinalities 4 and 2.
    let cat_bins = vec![vec![0u32, 1, 2, 3], vec![1u32, 0, 1, 0]];

    let (bins, n_bins, real_folds) =
        quantize_feature_major_with_one_hot(&floats, &borders, &cat_bins, n);

    assert_eq!(
        bins.len(),
        (floats.len() + cat_bins.len()) * n,
        "the device axis is the CONCATENATION of float then one-hot stripes"
    );
    assert_eq!(
        n_bins, 4,
        "n_bins is max(float n_bins = 3, max cat cardinality = 4)"
    );
    // The cat stripes are the PerfectHash bin columns VERBATIM — no re-binning.
    let cat0 = bins
        .get(floats.len() * n..(floats.len() + 1) * n)
        .expect("cat stripe 0");
    let cat1 = bins
        .get((floats.len() + 1) * n..(floats.len() + 2) * n)
        .expect("cat stripe 1");
    assert_eq!(cat0, &[0u32, 1, 2, 3][..], "one-hot column 0 copied verbatim");
    assert_eq!(cat1, &[1u32, 0, 1, 0][..], "one-hot column 1 copied verbatim");
    // Device feature index `n_float + c` IS one-hot column `c` — the contiguous
    // range the two-pass scorer bounds with `feature_lo = n_float`.
    assert_eq!(
        real_folds,
        vec![2u32, 3, 4, 2],
        "real_folds = [borders+1 per float, then each one-hot column's cardinality]"
    );
}

#[test]
fn cat_only_pool_yields_a_nonzero_n_bins_and_a_legal_padded_line() {
    let n = 4usize;
    let no_float: Vec<Vec<f32>> = Vec::new();
    let no_borders: Vec<Vec<f64>> = Vec::new();
    let cat_bins = vec![vec![0u32, 1, 1, 0]];

    let (bins, n_bins, real_folds) =
        quantize_feature_major_with_one_hot(&no_float, &no_borders, &cat_bins, n);

    assert_eq!(bins.len(), n, "one stripe for the single one-hot column");
    assert_eq!(
        n_bins, 2,
        "a 0-float pool must NOT report n_bins == 0 — the backend session declines \
         on `n_features == 0 || n_bins == 0`, which would make SPEC-OH-20's 0-float \
         target unreachable"
    );
    assert_eq!(real_folds, vec![2u32]);
    // `pad_hist_line_bins(2)` is 32, inside the legal {32,64,128,256} family.
    assert!(
        n_bins <= 32,
        "a cardinality-2 column pads to the legal 32-wide histogram line"
    );
}

#[test]
fn real_folds_is_the_true_cardinality_and_float_only_is_unchanged() {
    let n = 6usize;
    let floats = vec![
        vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0_f32, 4.0, 3.0, 2.0, 1.0, 0.0],
    ];
    // 3 borders -> 4 bins; 5 borders -> 6 bins.
    let borders = vec![
        vec![0.5_f64, 2.5, 4.5],
        vec![0.5_f64, 1.5, 2.5, 3.5, 4.5],
    ];
    let cat_bins = vec![vec![0u32, 1, 0, 1, 1, 0]];

    let (_bins, _n_bins, real_folds) =
        quantize_feature_major_with_one_hot(&floats, &borders, &cat_bins, n);
    assert_eq!(
        real_folds,
        vec![4u32, 6, 2],
        "real_folds carries the TRUE per-feature cardinality — the whole reason it \
         exists separately from `TCFeature.folds`"
    );

    // FLOAT-ONLY invariance (SPEC-OH-31): production ALWAYS calls the one-hot
    // quantizer, passing an empty `cat_bins` when the pool has no one-hot columns.
    // The float bins and n_bins must then be element-wise identical to the plain
    // `quantize_feature_major`, and `real_folds` must still be fully populated —
    // T27b's unconditional `real_folds.len() == eff_n_features` assertion depends
    // on it.
    let no_cats: Vec<Vec<u32>> = Vec::new();
    let (oh_bins, oh_n_bins, oh_folds) =
        quantize_feature_major_with_one_hot(&floats, &borders, &no_cats, n);
    let (plain_bins, plain_n_bins) = quantize_feature_major(&floats, &borders, n);
    assert_eq!(
        oh_bins, plain_bins,
        "a float-only pool must produce byte-identical bins through either entry"
    );
    assert_eq!(oh_n_bins, plain_n_bins, "and an identical n_bins");
    assert_eq!(
        oh_folds,
        vec![4u32, 6],
        "real_folds is NEVER empty on a device-eligible fit, float-only included"
    );
}

#[test]
fn one_hot_cardinality_above_the_device_bound_falls_back_to_the_cpu_grower() {
    // SPEC §9 R10 says "bound the cardinality on the device OR FALL BACK". Falling
    // back is the correct reading: returning an error from the quantizer would ABORT
    // an otherwise valid fit. The bound is expressed as a `device_host_eligible`
    // clause, so an over-wide column silently (and correctly) trains on the CPU.
    assert!(
        super::one_hot_cardinalities_fit_the_device(&[2, 4, super::DEVICE_ONE_HOT_MAX_CARDINALITY]),
        "cardinalities at or below the bound stay device-eligible"
    );
    assert!(
        !super::one_hot_cardinalities_fit_the_device(&[
            2,
            super::DEVICE_ONE_HOT_MAX_CARDINALITY + 1
        ]),
        "a column wider than the device bound must make the fit host-INeligible \
         (i.e. fall back to the CPU grower), never raise an error"
    );
    assert!(
        super::one_hot_cardinalities_fit_the_device(&[]),
        "a pool with no one-hot columns is unaffected (SPEC-OH-31)"
    );
}
