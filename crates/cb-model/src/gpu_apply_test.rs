//! Unit tests for `gpu_apply.rs` (GINF-01-S1 guard, GINF-01-S2 flattener).
//! Sibling `#[path]` mount (source/test separation, CLAUDE.md), mirroring
//! `export/onnx_test.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use super::{check_gpu_apply_supported, flatten_oblivious_f64, FlatObliviousF64, GpuApplyUnsupported};
use crate::ctr_data::{CtrData, ECtrType, Prior};
use crate::model::{CtrSplit, Model, ModelSplit, NonSymmetricTree, ObliviousTree, RegionTree, Split};
use crate::predict_raw;

// ── Shared fixtures ─────────────────────────────────────────────────────────

/// An all-empty, all-oblivious, float-only, `ctr_data: None`, scalar model — the
/// baseline every disqualifying-condition test overrides one field of.
fn empty_model() -> Model {
    Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// A depth-2 oblivious tree on features {`f0`, `f1`} (`> border`), with the four
/// forward-bit-order `leaf_values`.
fn depth2_tree(f0: usize, b0: f64, f1: usize, b1: f64, leaves: [f64; 4]) -> ObliviousTree {
    ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: f0, border: b0 }),
            ModelSplit::Float(Split { feature: f1, border: b1 }),
        ],
        leaf_values: leaves.to_vec(),
        leaf_weights: vec![1.0; 4],
    }
}

/// A minimal single-leaf (depth-0) non-symmetric tree — enough to make
/// `model.non_symmetric_trees` non-empty.
fn minimal_non_symmetric_tree() -> NonSymmetricTree {
    NonSymmetricTree {
        tree_splits: Vec::new(),
        step_nodes: vec![(0, 0)],
        node_id_to_leaf_id: vec![0],
        leaf_values: vec![0.0],
        leaf_weights: vec![0.0],
    }
}

/// A minimal depth-0 region tree (one leaf, no levels).
fn minimal_region_tree() -> RegionTree {
    RegionTree {
        levels: Vec::new(),
        leaf_values: vec![0.0],
        leaf_weights: vec![0.0],
    }
}

/// A minimal `CtrSplit` over a single categorical feature.
fn minimal_ctr_split() -> CtrSplit {
    CtrSplit {
        projection: cb_train::TProjection::single(0),
        ctr_type: ECtrType::Borders,
        prior: Prior { num: 0.0, denom: 1.0 },
        target_border_idx: 0,
        border: 0.0,
        shift: 0.0,
        scale: 1.0,
    }
}

// ── GINF-01-S1: guard ───────────────────────────────────────────────────────

#[test]
fn guard_accepts_float_oblivious_scalar() {
    let mut model = empty_model();
    model.oblivious_trees = vec![depth2_tree(0, 0.5, 1, 0.5, [1.0, 2.0, 3.0, 4.0])];
    model.float_feature_borders = vec![vec![0.5], vec![0.5]];
    model.bias = 0.25;
    assert!(check_gpu_apply_supported(&model).is_ok());
}

#[test]
fn guard_rejects_ctr() {
    // (a) via a ModelSplit::Ctr split.
    let mut via_split = empty_model();
    via_split.oblivious_trees = vec![ObliviousTree {
        splits: vec![ModelSplit::Ctr(minimal_ctr_split())],
        leaf_values: vec![0.1, 0.2],
        leaf_weights: vec![1.0, 1.0],
    }];
    assert!(matches!(
        check_gpu_apply_supported(&via_split),
        Err(GpuApplyUnsupported::CategoricalFeatures)
    ));

    // (b) via baked ctr_data.
    let mut via_data = empty_model();
    via_data.ctr_data = Some(CtrData { tables: std::collections::BTreeMap::new() });
    assert!(matches!(
        check_gpu_apply_supported(&via_data),
        Err(GpuApplyUnsupported::CategoricalFeatures)
    ));
}

#[test]
fn guard_rejects_non_symmetric() {
    let mut model = empty_model();
    model.non_symmetric_trees = vec![minimal_non_symmetric_tree()];
    assert!(matches!(
        check_gpu_apply_supported(&model),
        Err(GpuApplyUnsupported::NonObliviousTrees)
    ));
}

#[test]
fn guard_rejects_region() {
    let mut model = empty_model();
    model.region_trees = vec![minimal_region_tree()];
    assert!(matches!(
        check_gpu_apply_supported(&model),
        Err(GpuApplyUnsupported::RegionTrees)
    ));
}

#[test]
fn guard_rejects_multidim() {
    let mut model = empty_model();
    model.approx_dimension = 2;
    assert!(matches!(
        check_gpu_apply_supported(&model),
        Err(GpuApplyUnsupported::MultiDimensional)
    ));
}

// ── GINF-01-S2: flattener ───────────────────────────────────────────────────

/// A known 2-tree, depth-2, float-only oblivious scalar model with distinct
/// borders + distinct leaf values + a nonzero bias.
fn two_tree_model() -> Model {
    let mut model = empty_model();
    model.oblivious_trees = vec![
        // Tree 0: splits (f0 > 0.5), (f1 > 1.5).
        depth2_tree(0, 0.5, 1, 1.5, [1.0, 2.0, 3.0, 4.0]),
        // Tree 1: splits (f1 > 0.0), (f0 > 2.5); distinct features/borders/leaves.
        depth2_tree(1, 0.0, 0, 2.5, [10.0, 20.0, 30.0, 40.0]),
    ];
    model.float_feature_borders = vec![vec![0.5, 2.5], vec![0.0, 1.5]];
    model.bias = 0.25;
    model
}

/// Four probe objects (columns f0, f1) spanning distinct leaves across both trees.
fn probe_columns() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0, 3.0, 2.0], // f0
        vec![0.0, 2.0, 1.0, 3.0], // f1
    ]
}

/// Host reconstruction of raw predictions from the flat arrays, mirroring
/// `leaf_index` (forward bit order) + per-tree leaf gather + `bias`. The leaf
/// sum is a left-to-right fold seeded at `0.0`, then `bias + sum` — mirroring
/// `predict_raw_one`'s `model.bias + sum_f64(&oblivious)` association exactly.
fn reconstruct(flat: &FlatObliviousF64, features: &[Vec<f32>]) -> Vec<f64> {
    let n_objects = features.first().map_or(0, Vec::len);
    let n_trees = flat.tree_split_offsets.len() - 1;
    (0..n_objects)
        .map(|obj| {
            let mut leaf_sum = 0.0f64;
            for t in 0..n_trees {
                let s0 = flat.tree_split_offsets[t] as usize;
                let s1 = flat.tree_split_offsets[t + 1] as usize;
                let mut leaf = 0usize;
                for (bit, s) in (s0..s1).enumerate() {
                    let f = flat.split_features[s] as usize;
                    let border = flat.split_borders[s];
                    let v = features[f].get(obj).copied().unwrap_or(f32::NAN);
                    if f64::from(v) > border {
                        leaf |= 1usize << bit;
                    }
                }
                let li = flat.tree_leaf_offsets[t] as usize + leaf;
                leaf_sum += flat.leaf_values.get(li).copied().unwrap_or(0.0);
            }
            flat.bias + leaf_sum
        })
        .collect()
}

#[test]
fn flatten_roundtrip_matches_cpu() {
    let model = two_tree_model();
    let cols = probe_columns();
    let flat = flatten_oblivious_f64(&model).unwrap();

    let host = reconstruct(&flat, &cols);
    let cpu = predict_raw(&model, &cols);

    assert_eq!(host.len(), cpu.len());
    for (h, c) in host.iter().zip(cpu.iter()) {
        assert_eq!(h, c, "host reconstruction must equal predict_raw exactly");
    }
}

#[test]
fn flatten_offsets_invariants() {
    let model = two_tree_model();
    let flat = flatten_oblivious_f64(&model).unwrap();
    let n_trees = model.oblivious_trees.len();

    assert_eq!(flat.tree_split_offsets.len(), n_trees + 1);
    assert_eq!(flat.tree_leaf_offsets.len(), n_trees + 1);
    assert_eq!(flat.split_features.len(), flat.split_borders.len());

    // Offsets start at 0, end at the concatenated lengths, and are monotonic.
    assert_eq!(flat.tree_split_offsets[0], 0);
    assert_eq!(flat.tree_leaf_offsets[0], 0);
    assert_eq!(*flat.tree_split_offsets.last().unwrap() as usize, flat.split_features.len());
    assert_eq!(*flat.tree_leaf_offsets.last().unwrap() as usize, flat.leaf_values.len());
    for w in flat.tree_split_offsets.windows(2) {
        assert!(w[1] >= w[0]);
    }
    for w in flat.tree_leaf_offsets.windows(2) {
        assert!(w[1] >= w[0]);
    }

    // Each tree's CSR leaf span equals the source tree's leaf-value count.
    for (t, tree) in model.oblivious_trees.iter().enumerate() {
        let span = (flat.tree_leaf_offsets[t + 1] - flat.tree_leaf_offsets[t]) as usize;
        assert_eq!(span, tree.leaf_values.len());
        let split_span = (flat.tree_split_offsets[t + 1] - flat.tree_split_offsets[t]) as usize;
        assert_eq!(split_span, tree.splits.len());
    }
}

#[test]
fn flatten_rejects_unsupported() {
    // CTR model → Err (the guard rejection surfaced as CbError).
    let mut model = empty_model();
    model.oblivious_trees = vec![ObliviousTree {
        splits: vec![ModelSplit::Ctr(minimal_ctr_split())],
        leaf_values: vec![0.1, 0.2],
        leaf_weights: vec![1.0, 1.0],
    }];
    assert!(flatten_oblivious_f64(&model).is_err());
}


// ── SPEC-OH-17 — the guard names one-hot explicitly ─────────────────────────

#[test]
fn gpu_apply_guard_names_one_hot_splits_explicitly() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![ModelSplit::OneHot(crate::OneHotModelSplit {
            cat_feature: 0,
            value_hash: 1_296_865_003,
        })],
        leaf_values: vec![1.0, 2.0],
        leaf_weights: vec![1.0, 1.0],
    });
    model.float_feature_borders = vec![vec![0.5]];
    assert!(matches!(
        check_gpu_apply_supported(&model),
        Err(GpuApplyUnsupported::OneHotSplits)
    ));
    // …and the flattener inherits the SAME rejection (it calls the guard first),
    // never the "unexpected one-hot split in a guard-passed model" fallback.
    let flat = flatten_oblivious_f64(&model);
    assert!(flat.is_err(), "the flattener must inherit the guard rejection");
}
