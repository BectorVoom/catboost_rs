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

// --- Phase-5 Wave-0: ctr_data parsing (backward-compatible #[serde(default)]) -

#[test]
fn borders_only_model_has_empty_ctr_data() {
    // The pre-Phase-5 regression skeleton has NO ctr_data block. The new
    // #[serde(default)] field must keep it parsing with an empty map (RESEARCH
    // A5 — the parser was borders-only before).
    let model = load_model_json(&fixture("regression_skeleton/model.json"))
        .expect("regression_skeleton/model.json parses without ctr_data");
    assert!(
        model.ctr_data().is_empty(),
        "borders-only model should have an empty ctr_data map"
    );
}

#[test]
fn ctr_data_block_round_trips_bucket_counts() {
    use crate::model_json::ModelJson;

    // A minimal upstream-shaped model.json carrying one Borders CTR table.
    // hash_stride = 3 => each bucket is [hash_string, count0, count1]; two
    // buckets here. This mirrors json_model_helpers.cpp:475-482 exactly.
    let json = r#"{
        "features_info": { "float_features": [] },
        "oblivious_trees": [],
        "scale_and_bias": [1, [0.0]],
        "ctr_data": {
            "0/3/Borders": {
                "hash_map": ["12345", 4, 1, "67890", 0, 3],
                "hash_stride": 3,
                "counter_denominator": 0
            }
        }
    }"#;

    let model: ModelJson = serde_json::from_str(json).expect("ctr_data model parses");
    let ctr = model.ctr_data();
    assert_eq!(ctr.len(), 1, "exactly one CTR table");

    let table = ctr.get("0/3/Borders").expect("the Borders table is keyed");
    assert_eq!(table.hash_stride, 3);
    assert_eq!(table.counter_denominator, 0);

    let counts = table.bucket_counts().expect("counts parse");
    // Two buckets, each with hash_stride-1 == 2 integer counts, hash stripped.
    assert_eq!(counts, vec![vec![4_i64, 1], vec![0, 3]]);
}

#[test]
fn ctr_data_counter_table_exposes_single_count_and_denominator() {
    use crate::model_json::ModelJson;

    // Counter/FeatureFreq tables carry a single ctrTotal per bucket
    // (hash_stride = 2) and a non-zero counter_denominator
    // (static_ctr_provider.cpp: CTR = ctrTotal[bucket] / CounterDenominator).
    let json = r#"{
        "features_info": { "float_features": [] },
        "oblivious_trees": [],
        "scale_and_bias": [1, [0.0]],
        "ctr_data": {
            "0/Counter": {
                "hash_map": ["111", 7, "222", 2],
                "hash_stride": 2,
                "counter_denominator": 9
            }
        }
    }"#;

    let model: ModelJson = serde_json::from_str(json).expect("counter ctr_data parses");
    let table = model.ctr_data().get("0/Counter").expect("counter table keyed");
    assert_eq!(table.counter_denominator, 9);
    let counts = table.bucket_counts().expect("counter counts parse");
    assert_eq!(counts, vec![vec![7_i64], vec![2]]);
}

#[test]
fn ragged_ctr_hash_map_is_a_typed_error_not_a_panic() {
    use crate::model_json::ModelJson;

    // hash_map length (5) is not a multiple of hash_stride (3): malformed blob
    // must surface as MalformedModel, never panic (T-05-01-01).
    let json = r#"{
        "features_info": { "float_features": [] },
        "oblivious_trees": [],
        "scale_and_bias": [1, [0.0]],
        "ctr_data": {
            "0/Borders": {
                "hash_map": ["1", 2, 3, "4", 5],
                "hash_stride": 3,
                "counter_denominator": 0
            }
        }
    }"#;

    let model: ModelJson = serde_json::from_str(json).expect("model parses");
    let table = model.ctr_data().get("0/Borders").expect("table keyed");
    assert!(
        table.bucket_counts().is_err(),
        "ragged hash_map must error, not panic"
    );
}

// ── Non-symmetric "trees" nested-node parser (FEAT-06, RESEARCH Pitfall 3) ──

#[test]
fn non_symmetric_depthwise_parses_into_non_empty_flat_triple() {
    // The committed Depthwise fixture is a real catboost 1.2.10 non-symmetric
    // model.json (top-level "trees", nested {split,left,right}/{value,weight}).
    let model = load_model_json(&fixture("non_symmetric/depthwise/model.json"))
        .expect("depthwise model.json must load");
    assert!(
        model.is_non_symmetric(),
        "Depthwise model must populate `trees`, not `oblivious_trees`"
    );
    assert!(
        model.oblivious_trees.is_empty(),
        "a non-symmetric model must NOT route through oblivious_trees (Pitfall 3)"
    );

    let flat = model
        .non_symmetric_flat_trees()
        .expect("flatten must succeed");
    assert!(!flat.is_empty(), "expected at least one non-symmetric tree");
    // Pitfall-3 warning sign: a zero-length triple means the nested schema was
    // mis-parsed as oblivious. Every tree must have nodes + distinct leaves.
    for (i, t) in flat.iter().enumerate() {
        assert!(!t.step_nodes.is_empty(), "tree {i}: empty step_nodes (Pitfall 3)");
        assert_eq!(
            t.split_borders.len(),
            t.step_nodes.len(),
            "tree {i}: per-node split/step length mismatch"
        );
        assert_eq!(
            t.node_id_to_leaf_id.len(),
            t.step_nodes.len(),
            "tree {i}: per-node leaf-id length mismatch"
        );
        assert!(!t.leaf_values.is_empty(), "tree {i}: zero distinct leaves (Pitfall 3)");
        assert_eq!(
            t.leaf_values.len(),
            t.leaf_weights.len(),
            "tree {i}: leaf value/weight length mismatch"
        );
        // Leaf count == number of terminal (0,0) step nodes.
        let terminal = t.step_nodes.iter().filter(|&&(l, r)| l == 0 && r == 0).count();
        assert_eq!(terminal, t.leaf_values.len(), "tree {i}: leaf count mismatch");
    }
}

#[test]
fn non_symmetric_lossguide_split_borders_are_non_empty() {
    let model = load_model_json(&fixture("non_symmetric/lossguide/model.json"))
        .expect("lossguide model.json must load");
    assert!(model.is_non_symmetric());
    let borders = model
        .non_symmetric_split_borders()
        .expect("split borders must extract");
    assert!(
        !borders.is_empty(),
        "Lossguide split borders must be non-empty (the splits-first lock, Open Question 1)"
    );
    assert!(
        borders.iter().all(|b| b.is_finite()),
        "interior split borders must be finite (leaf placeholders excluded)"
    );
}

#[test]
fn deeply_nested_non_symmetric_tree_is_a_typed_error_not_a_stack_overflow() {
    use crate::model_json::ModelJson;

    // Build a left-spine deeper than MAX_NON_SYMMETRIC_DEPTH (64) — the converter
    // must reject it (T-06.6-07), never recurse unbounded.
    let mut node = String::from(r#"{"value": 0.0, "weight": 1}"#);
    for _ in 0..80 {
        node = format!(
            r#"{{"split":{{"border":0.5,"float_feature_index":0,"split_index":0,"split_type":"FloatFeature"}},"left":{node},"right":{{"value":0.0,"weight":1}}}}"#
        );
    }
    let json = format!(
        r#"{{"features_info":{{"float_features":[]}},"trees":[{node}],"scale_and_bias":[1,[0.0]]}}"#
    );
    let parsed: Result<ModelJson, _> = serde_json::from_str(&json);
    // Either serde rejects the over-deep nesting OR our depth-bounded converter
    // does — both are typed errors, never an unbounded-recursion stack overflow.
    let rejected = match parsed {
        Err(_) => true,
        Ok(model) => model.non_symmetric_flat_trees().is_err(),
    };
    assert!(
        rejected,
        "an over-deep tree must be a typed error, not unbounded recursion"
    );
}
