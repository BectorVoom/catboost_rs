//! One-hot split apply tests (SPEC-OH-10).
//!
//! An object passes a [`crate::ModelSplit::OneHot`] split iff the RAW categorical
//! value in its `cat_feature` column hashes — via `cb_data::calc_cat_feature_hash`,
//! the same function the trainer and the `.cbm` writer use — to exactly the
//! split's `value_hash`. There is no border, no ordering, and no `ctr_data`
//! lookup: it is a pure equality test on the upstream raw `i32` hash space.
//!
//! Sibling `#[path]` mount (source/test separation, CLAUDE.md) of `apply.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use crate::model::{ModelSplit, ObliviousTree, OneHotModelSplit};
use crate::{predict_raw_cat, Model};

/// A depth-1 oblivious model whose only split is `cat_feature 0 == hash("b")`.
/// Leaf 0 (split fails) is `1.0`, leaf 1 (split passes) is `2.0`.
fn one_hot_depth1_model() -> Model {
    let value_hash = cb_data::calc_cat_feature_hash("b") as i32;
    Model {
        oblivious_trees: vec![ObliviousTree {
            splits: vec![ModelSplit::OneHot(OneHotModelSplit {
                cat_feature: 0,
                value_hash,
            })],
            leaf_values: vec![1.0, 2.0],
            leaf_weights: vec![1.0, 1.0],
        }],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// SPEC-OH-10 — only the object whose raw category equals the split's category
/// takes the pass branch. Under T05's `false` stub every object lands in leaf 0
/// (`[1.0, 1.0, 1.0]`), which is the expected Red.
#[test]
fn one_hot_split_passes_only_on_the_matching_raw_category() {
    let model = one_hot_depth1_model();
    let cat_columns = vec![vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]];

    let got = predict_raw_cat(&model, &[], &cat_columns);

    assert_eq!(
        got,
        vec![1.0, 2.0, 1.0],
        "only the raw value hashing to the split's value_hash passes"
    );
}

/// SPEC-OH-10 — a one-hot split whose `cat_feature` column is absent from the
/// supplied `cat_columns` fails defensively (`false`), matching
/// `passes_float_split`'s short-column NaN behaviour: no panic, no index out of
/// bounds, and the object lands in the fail leaf.
#[test]
fn one_hot_split_on_a_missing_cat_column_is_defensively_false() {
    let value_hash = cb_data::calc_cat_feature_hash("b") as i32;
    let mut model = one_hot_depth1_model();
    model.oblivious_trees[0].splits = vec![ModelSplit::OneHot(OneHotModelSplit {
        // Column 7 does not exist in the single-column input below.
        cat_feature: 7,
        value_hash,
    })];

    let cat_columns = vec![vec!["b".to_owned()]];
    let got = predict_raw_cat(&model, &[], &cat_columns);

    assert_eq!(got, vec![1.0], "a missing cat column never passes the split");
}
