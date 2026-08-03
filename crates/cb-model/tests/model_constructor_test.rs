//! F08 — `#[non_exhaustive]` left no `Model` shape unexpressible from outside.
//!
//! This is an INTEGRATION target, so it compiles as a SEPARATE crate and
//! `#[non_exhaustive]` applies to it exactly as it does to `catboost-rs` and
//! `cb-train`. Without this test the `#[non_exhaustive]` decision is unverified
//! from the outside: the migration could have silently lost a shape (an explicit
//! `ctr_data`, `approx_dimension != 1`, a non-empty `class_to_label`, the
//! non-oblivious tree vectors) and nothing would notice until a downstream crate
//! needed it.

use std::collections::BTreeMap;

use cb_model::{
    CtrData, Model, ModelSplit, NonSymmetricTree, ObliviousTree, RegionLevel, RegionTree, Split,
};

fn tree() -> ObliviousTree {
    ObliviousTree {
        splits: vec![ModelSplit::Float(Split {
            feature: 0,
            border: 0.5,
        })],
        leaf_values: vec![-1.0, 1.0],
        leaf_weights: vec![2.0, 3.0],
    }
}

/// The base shape: trees + bias + borders, everything else at its zero value.
#[test]
fn new_leaves_every_other_field_at_its_zero_value() {
    let m = Model::new(vec![tree()], 0.25, vec![vec![0.5]]);

    assert_eq!(m.oblivious_trees.len(), 1);
    assert!((m.bias - 0.25).abs() < f64::EPSILON);
    assert_eq!(m.float_feature_borders, vec![vec![0.5]]);
    assert!(m.non_symmetric_trees.is_empty());
    assert!(m.region_trees.is_empty());
    assert_eq!(m.ctr_data, None);
    assert_eq!(m.approx_dimension, 1);
    assert!(m.class_to_label.is_empty());
    assert_eq!(m.cat_feature_count(), 0);
}

/// Every shape the migrated sites needed is reachable through the builders, and
/// each builder sets EXACTLY its own field — a builder that clobbered a
/// neighbour would make a migrated literal silently non-identical.
#[test]
fn an_external_crate_can_build_every_model_shape_without_struct_literal_syntax() {
    let ctr_data = CtrData {
        tables: BTreeMap::new(),
    };
    let non_sym = NonSymmetricTree {
        tree_splits: vec![ModelSplit::Float(Split {
            feature: 0,
            border: 1.5,
        })],
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        leaf_values: vec![0.5],
        leaf_weights: vec![1.0],
    };
    let region = RegionTree {
        levels: vec![RegionLevel {
            split: ModelSplit::Float(Split {
                feature: 0,
                border: 2.5,
            }),
            expected_direction: true,
            one_hot: false,
        }],
        leaf_values: vec![0.1, 0.2],
        leaf_weights: vec![1.0, 1.0],
    };

    let m = Model::new(vec![tree()], 0.25, vec![vec![0.5]])
        .with_ctr_data(ctr_data.clone())
        .with_non_symmetric_trees(vec![non_sym.clone()])
        .with_region_trees(vec![region.clone()])
        .with_approx_dimension(3)
        .with_class_to_label(vec![0.0, 1.0, 2.0])
        .with_cat_feature_count(4);

    // Each builder set exactly its own field, and none disturbed the base.
    assert_eq!(m.ctr_data, Some(ctr_data));
    assert_eq!(m.non_symmetric_trees, vec![non_sym]);
    assert_eq!(m.region_trees, vec![region]);
    assert_eq!(m.approx_dimension, 3);
    assert_eq!(m.class_to_label, vec![0.0, 1.0, 2.0]);
    assert_eq!(m.cat_feature_count(), 4);
    assert_eq!(m.oblivious_trees, vec![tree()]);
    assert!((m.bias - 0.25).abs() < f64::EPSILON);
    assert_eq!(m.float_feature_borders, vec![vec![0.5]]);
}

/// Builder calls are order-independent and idempotent per field, so a migrated
/// site may chain them in any order without changing the resulting model.
#[test]
fn builder_order_does_not_change_the_resulting_model() {
    let a = Model::new(vec![tree()], 1.0, vec![vec![0.5]])
        .with_approx_dimension(2)
        .with_cat_feature_count(5)
        .with_class_to_label(vec![7.0]);
    let b = Model::new(vec![tree()], 1.0, vec![vec![0.5]])
        .with_class_to_label(vec![7.0])
        .with_cat_feature_count(5)
        .with_approx_dimension(2);

    assert_eq!(a, b);
}
