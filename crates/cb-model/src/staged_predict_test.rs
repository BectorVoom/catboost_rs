//! Unit tests for [`crate::predict_raw_staged`] — per-tree-prefix cumulative raw
//! approx for SCALAR oblivious float-only models (SP-01 / SP-02).
//!
//! Sibling `#[path]` mount (source/test separation, CLAUDE.md) of `apply.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use crate::model::{ModelSplit, Split};
use crate::{predict_raw, predict_raw_staged, Model, ObliviousTree};

/// A depth-2 oblivious tree on features {0, 1} (`> 0.5`), with leaf values
/// `scale * {1, 2, 3, 4}` in forward-bit order (4 leaves).
fn scaled_tree(scale: f64) -> ObliviousTree {
    ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split {
                feature: 0,
                border: 0.5,
            }),
            ModelSplit::Float(Split {
                feature: 1,
                border: 0.5,
            }),
        ],
        leaf_values: vec![scale, scale * 2.0, scale * 3.0, scale * 4.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    }
}

/// A scalar oblivious model built from `scales` (one tree per scale), bias `bias`.
fn model_from_scales(scales: &[f64], bias: f64) -> Model {
    Model {
        oblivious_trees: scales.iter().map(|&s| scaled_tree(s)).collect(),
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![0.5], vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// A model holding only the first `k` trees of `model` (same bias) — the
/// hand-rolled "truncated apply" reference for a prefix of length `k`.
fn truncate(model: &Model, k: usize) -> Model {
    Model {
        oblivious_trees: model.oblivious_trees.iter().take(k).cloned().collect(),
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: model.bias,
        float_feature_borders: model.float_feature_borders.clone(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// Four probe objects spanning the four leaves of each depth-2 tree.
fn probe_columns() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0, 0.0, 1.0], // feature 0
        vec![0.0, 0.0, 1.0, 1.0], // feature 1
    ]
}

/// SP-01: a single-stage prefix over the first `k` trees equals the full-apply
/// accumulation truncated to `k`; at `k == T` it equals `predict_raw` exactly.
#[test]
fn staged_prefix_matches_truncated_apply() {
    let model = model_from_scales(&[1.0, 10.0, 100.0], 0.5);
    let cols = probe_columns();
    let n_trees = model.oblivious_trees.len();

    // k = T - 1: one stage covering the first two trees, compared to the
    // truncated-model apply (the hand-rolled prefix reference).
    let k = n_trees - 1;
    let staged = predict_raw_staged(&model, &cols, 0, k, k);
    assert_eq!(staged.len(), 1, "single stage yields exactly one row");
    let expected_prefix = predict_raw(&truncate(&model, k), &cols);
    assert_eq!(staged[0], expected_prefix, "prefix row == truncated apply at k");

    // k = T: the full prefix equals `predict_raw` on the whole model.
    let staged_full = predict_raw_staged(&model, &cols, 0, n_trees, n_trees);
    assert_eq!(staged_full.len(), 1);
    let full = predict_raw(&model, &cols);
    assert_eq!(staged_full[0], full, "prefix row at k==T == predict_raw");
}

/// SP-02: the stage schedule (start / end / period) produces exactly the upstream
/// stage tree-counts, always including `ntree_end`, one cumulative row per stage.
#[test]
fn staged_schedule_boundaries() {
    // T = 10 trees, distinct scales so each stage row differs.
    let scales: Vec<f64> = (0..10).map(|i| 10f64.powi(i)).collect();
    let model = model_from_scales(&scales, 0.25);
    let cols = probe_columns();

    // Sub-case 1: start=0, end=0 (all), period=3 => counts {3, 6, 9, 10}.
    let staged = predict_raw_staged(&model, &cols, 0, 0, 3);
    assert_eq!(staged.len(), 4, "period-3 over 10 trees => 4 stages");
    for (row, &count) in staged.iter().zip([3usize, 6, 9, 10].iter()) {
        let expected = predict_raw(&truncate(&model, count), &cols);
        assert_eq!(row, &expected, "stage row == prefix apply at count {count}");
    }

    // Sub-case 2: period=1 => T rows; final row == predict_raw(full).
    let step1 = predict_raw_staged(&model, &cols, 0, 0, 1);
    assert_eq!(step1.len(), 10, "period-1 => one stage per tree");
    let full = predict_raw(&model, &cols);
    assert_eq!(step1.last().unwrap(), &full, "final stage == full apply");

    // Sub-case 3: start >= end => empty Vec.
    let empty = predict_raw_staged(&model, &cols, 10, 5, 1);
    assert!(empty.is_empty(), "start >= end yields no stages");
}
