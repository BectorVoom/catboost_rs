//! Tests for the `model.json` parser: load the committed `regression_skeleton`
//! oracle and assert the oblivious-tree invariants the Wave-1 slice relies on.
//!
//! Source/test separation is mandatory (CLAUDE.md): the parser lives in
//! `model_json.rs`; all assertions live here.

use std::path::PathBuf;

use crate::model_json::load_model_json;

/// Absolute path to a committed fixture file under `crates/cb-oracle/fixtures/`.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

#[test]
fn parses_regression_skeleton_model_json() {
    let model = load_model_json(&fixture("regression_skeleton/model.json"))
        .expect("regression_skeleton/model.json parses");

    // Non-empty: a trained skeleton has one oblivious tree per boosting
    // iteration.
    assert!(
        !model.oblivious_trees.is_empty(),
        "expected at least one oblivious tree"
    );

    for (t, tree) in model.oblivious_trees.iter().enumerate() {
        // Oblivious (symmetric) tree invariant: `depth` splits => `2^depth`
        // leaves. Asserting the relationship (rather than a hard-coded depth)
        // keeps the test valid across param changes.
        let depth = tree.splits.len();
        let expected_leaves = 1usize << depth;
        assert_eq!(
            tree.leaf_values.len(),
            expected_leaves,
            "tree {t}: {} leaves != 2^{depth}",
            tree.leaf_values.len()
        );
        // Every split this phase is a numeric FloatFeature split.
        for split in &tree.splits {
            assert_eq!(split.split_type, "FloatFeature");
            assert!(split.float_feature_index >= 0);
        }
    }
}

#[test]
fn split_and_leaf_extractors_have_expected_lengths() {
    let model = load_model_json(&fixture("regression_skeleton/model.json"))
        .expect("regression_skeleton/model.json parses");

    let total_splits: usize = model.oblivious_trees.iter().map(|t| t.splits.len()).sum();
    let total_leaves: usize = model
        .oblivious_trees
        .iter()
        .map(|t| t.leaf_values.len())
        .sum();

    // Extractors flatten in tree order; the flattened lengths must match the
    // per-tree sums so `compare_stage` lines up with the oracle vectors.
    assert_eq!(model.split_borders().len(), total_splits);
    assert_eq!(model.leaf_values().len(), total_leaves);
    assert!(total_splits > 0 && total_leaves > 0);
}

#[test]
fn bias_reads_scale_and_bias() {
    let model = load_model_json(&fixture("regression_skeleton/model.json"))
        .expect("regression_skeleton/model.json parses");

    // scale_and_bias = [1, [bias]]; bias() must read the inner [0] value and be
    // finite for a trained regression skeleton.
    let bias = model.bias().expect("bias present in scale_and_bias[1][0]");
    assert!(bias.is_finite(), "bias should be finite, got {bias}");
}
