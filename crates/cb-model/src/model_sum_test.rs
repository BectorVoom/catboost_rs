//! Unit tests for `sum_models` (SPEC `sum_models`, SM-01..SM-04). Mounted from
//! `model_sum.rs` (source/test separation, CLAUDE.md).
use std::collections::BTreeMap;

use super::*;
use crate::apply::predict_raw;
use crate::ctr_data::CtrData;
use crate::model::{ModelSplit, NonSymmetricTree, Split};

/// A tiny float-only oblivious model: 2 trees, 1 split each (2 leaves/tree),
/// `approx_dimension = 1`, `ctr_data = None`, non-symmetric/region empty.
fn tiny_model(bias: f64) -> Model {
    let split = ModelSplit::Float(Split {
        feature: 0,
        border: 0.5,
    });
    Model {
        oblivious_trees: vec![
            ObliviousTree {
                splits: vec![split.clone()],
                leaf_values: vec![1.0, 2.0],
                leaf_weights: vec![3.0, 4.0],
            },
            ObliviousTree {
                splits: vec![split],
                leaf_values: vec![-1.0, 5.0],
                leaf_weights: vec![1.0, 1.0],
            },
        ],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

#[test]
fn sum_models_rejects_empty() {
    let err = sum_models(&[], &[]).expect_err("empty models must be rejected");
    assert!(
        matches!(err, ModelError::Merge(_)),
        "expected ModelError::Merge, got {err:?}"
    );
}

#[test]
fn sum_models_single_scales_leaves() {
    let m = tiny_model(0.5);
    let result = sum_models(&[&m], &[2.0]).expect("single-model sum must succeed");

    assert_eq!(result.oblivious_trees.len(), m.oblivious_trees.len());
    for (rt, mt) in result.oblivious_trees.iter().zip(m.oblivious_trees.iter()) {
        assert_eq!(rt.splits, mt.splits);
        for (rv, mv) in rt.leaf_values.iter().zip(mt.leaf_values.iter()) {
            assert!((rv - 2.0 * mv).abs() <= 1e-12, "leaf {rv} vs {mv}");
        }
    }
    assert!((result.bias - 1.0).abs() <= 1e-12, "bias {}", result.bias);
}

/// A second tiny float-only oblivious model with a DIFFERENT tree count (3
/// trees) but the SAME structural fields (`float_feature_borders`,
/// `approx_dimension`, `class_to_label`) as `tiny_model`, so the two are
/// mergeable (SM-02).
fn tiny_model_b(bias: f64) -> Model {
    let split = ModelSplit::Float(Split {
        feature: 0,
        border: 0.5,
    });
    Model {
        oblivious_trees: vec![
            ObliviousTree {
                splits: vec![split.clone()],
                leaf_values: vec![0.5, -0.5],
                leaf_weights: vec![1.0, 1.0],
            },
            ObliviousTree {
                splits: vec![split.clone()],
                leaf_values: vec![2.0, 2.0],
                leaf_weights: vec![1.0, 1.0],
            },
            ObliviousTree {
                splits: vec![split],
                leaf_values: vec![-3.0, 1.0],
                leaf_weights: vec![1.0, 1.0],
            },
        ],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

#[test]
fn sum_models_concats_trees() {
    // Bias kept at 0.0 on both inputs so this assertion is isolated from SM-03
    // (bias summation lands in TASK-04).
    let m0 = tiny_model(0.0);
    let m1 = tiny_model_b(0.0);
    let a = m0.oblivious_trees.len();
    let b = m1.oblivious_trees.len();

    let result = sum_models(&[&m0, &m1], &[0.5, 1.5]).expect("compatible sum must succeed");
    assert_eq!(result.oblivious_trees.len(), a + b);

    for (rt, mt) in result
        .oblivious_trees
        .iter()
        .take(a)
        .zip(m0.oblivious_trees.iter())
    {
        assert_eq!(rt.splits, mt.splits);
        for (rv, mv) in rt.leaf_values.iter().zip(mt.leaf_values.iter()) {
            assert!((rv - 0.5 * mv).abs() <= 1e-12, "leaf {rv} vs {mv}");
        }
    }
    for (rt, mt) in result
        .oblivious_trees
        .iter()
        .skip(a)
        .zip(m1.oblivious_trees.iter())
    {
        assert_eq!(rt.splits, mt.splits);
        for (rv, mv) in rt.leaf_values.iter().zip(mt.leaf_values.iter()) {
            assert!((rv - 1.5 * mv).abs() <= 1e-12, "leaf {rv} vs {mv}");
        }
    }

    let columns: Vec<Vec<f32>> = vec![vec![0.1, 0.9, 0.3]];
    let merged_pred = predict_raw(&result, &columns);
    let m0_pred = predict_raw(&m0, &columns);
    let m1_pred = predict_raw(&m1, &columns);
    for i in 0..merged_pred.len() {
        let expected = 0.5 * m0_pred.get(i).copied().unwrap_or(f64::NAN)
            + 1.5 * m1_pred.get(i).copied().unwrap_or(f64::NAN);
        let actual = merged_pred.get(i).copied().unwrap_or(f64::NAN);
        assert!((actual - expected).abs() <= 1e-9, "object {i}: {actual} vs {expected}");
    }
}

#[test]
fn sum_models_sums_bias() {
    let m0 = tiny_model(0.25);
    let m1 = tiny_model_b(-0.75);
    let weights = [2.0, 4.0];

    let result = sum_models(&[&m0, &m1], &weights).expect("compatible sum must succeed");
    let expected = cb_core::sum_f64(&[weights[0] * m0.bias, weights[1] * m1.bias]);
    assert_eq!(result.bias, expected, "bias must be a bit-exact sum_f64 reduction");
}

#[test]
fn sum_models_empty_weights_defaults_ones() {
    let m = tiny_model(0.5);
    let default_weight = sum_models(&[&m], &[]).expect("empty weights must default to ones");
    let explicit_weight = sum_models(&[&m], &[1.0]).expect("explicit weight 1.0 must succeed");
    assert_eq!(default_weight, explicit_weight);
}

#[test]
fn sum_models_rejects_weight_count_mismatch() {
    let m0 = tiny_model(0.0);
    let m1 = tiny_model_b(0.0);
    // Non-empty weights whose length != models.len().
    let err = sum_models(&[&m0, &m1], &[1.0]).expect_err("mismatched weight count must be rejected");
    assert!(matches!(err, ModelError::Merge(_)), "expected Merge, got {err:?}");
}

#[test]
fn sum_models_rejects_non_oblivious() {
    let m0 = tiny_model(0.0);
    let mut m1 = tiny_model_b(0.0);
    m1.non_symmetric_trees.push(NonSymmetricTree {
        tree_splits: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        leaf_values: Vec::new(),
        leaf_weights: Vec::new(),
    });
    let err = sum_models(&[&m0, &m1], &[1.0, 1.0])
        .expect_err("a non-symmetric model must be rejected");
    assert!(matches!(err, ModelError::Merge(_)), "expected Merge, got {err:?}");
}

#[test]
fn sum_models_rejects_ctr_model() {
    let m0 = tiny_model(0.0);
    let m1 = tiny_model_b(0.0).with_ctr_data(CtrData {
        tables: BTreeMap::new(),
    });
    let err = sum_models(&[&m0, &m1], &[1.0, 1.0]).expect_err("a CTR model must be rejected");
    assert!(matches!(err, ModelError::Merge(_)), "expected Merge, got {err:?}");
}

#[test]
fn sum_models_rejects_border_mismatch() {
    let m0 = tiny_model(0.0);
    let mut m1 = tiny_model_b(0.0);
    m1.float_feature_borders = vec![vec![0.25, 0.75]];
    let err =
        sum_models(&[&m0, &m1], &[1.0, 1.0]).expect_err("mismatched borders must be rejected");
    assert!(matches!(err, ModelError::Merge(_)), "expected Merge, got {err:?}");
}

#[test]
fn sum_models_rejects_approx_dim_mismatch() {
    let m0 = tiny_model(0.0);
    let mut m1 = tiny_model_b(0.0);
    m1.approx_dimension = 2;
    let err = sum_models(&[&m0, &m1], &[1.0, 1.0])
        .expect_err("mismatched approx_dimension must be rejected");
    assert!(matches!(err, ModelError::Merge(_)), "expected Merge, got {err:?}");
}

#[test]
fn sum_models_rejects_class_to_label_mismatch() {
    let m0 = tiny_model(0.0);
    let mut m1 = tiny_model_b(0.0);
    m1.class_to_label = vec![0.0, 1.0];
    let err = sum_models(&[&m0, &m1], &[1.0, 1.0])
        .expect_err("mismatched class_to_label must be rejected");
    assert!(matches!(err, ModelError::Merge(_)), "expected Merge, got {err:?}");
}
