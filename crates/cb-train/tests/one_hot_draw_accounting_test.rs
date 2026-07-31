//! T01a / SPEC-OH-27 — the one-hot RSM draw-order ground-truth artifact.
//!
//! This is the SCAFFOLDING half: it asserts only that the ground-truth artifact
//! exists and states a machine-readable verdict. The behavioral draw-count
//! assertion lives in T01b, where a production consumer exists.
//!
//! Why this matters: a one-hot-routed categorical column changes the number of
//! candidate sub-lists at each tree level, and upstream charges one unconditional
//! RNG draw per sub-list. Getting the count wrong desynchronises every subsequent
//! tree's bootstrap sample — the same defect class as the two fabricated MVS draws
//! fixed in `d7676b5`, which passed every non-bootstrap test.

use std::path::PathBuf;

fn ground_truth_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.planning/plans/one-hot-categorical-training")
        .join("instrumented-ground-truth/ONE_HOT_GROUND_TRUTH.md")
}

/// The artifact exists and states exactly one of the three permitted verdicts.
/// A missing artifact, or one that hedges without committing, is a failure —
/// T01b consumes this verdict to decide between enforcing the rule and
/// typed-rejecting one-hot × bootstrap.
#[test]
fn one_hot_ground_truth_artifact_is_present_and_states_a_verdict() {
    let path = ground_truth_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("ground-truth artifact missing at {}: {e}", path.display()));

    let verdicts = [
        "RSM_RULE: n_float + n_one_hot",
        "RSM_RULE: n_float",
        "STATUS: NOT-ESTABLISHED",
    ];
    let found: Vec<&str> = verdicts
        .iter()
        .copied()
        .filter(|v| text.contains(v))
        .collect();

    assert!(
        !found.is_empty(),
        "ONE_HOT_GROUND_TRUTH.md must state one of {verdicts:?}"
    );
    // `RSM_RULE: n_float` is a prefix of `RSM_RULE: n_float + n_one_hot`, so the
    // stricter reading wins; assert the artifact is not ambiguous BETWEEN a rule
    // and a non-establishment.
    assert!(
        !(text.contains("STATUS: NOT-ESTABLISHED") && text.contains("RSM_RULE:")),
        "the artifact must not claim BOTH a derived rule and NOT-ESTABLISHED"
    );
}

/// If a rule IS claimed, the artifact must also carry its evidence grade and the
/// compression caveat — a bare verdict line with no provenance is exactly what
/// SPEC-OH-27 forbids ("do NOT guess a draw count").
#[test]
fn a_claimed_rule_carries_its_evidence_grade_and_caveats() {
    let text = std::fs::read_to_string(ground_truth_path()).expect("ground-truth artifact");
    if !text.contains("RSM_RULE:") {
        return; // NOT-ESTABLISHED: nothing further to require.
    }

    assert!(
        text.contains("SOURCE-DERIVED") || text.contains("instrumented"),
        "a claimed rule must state how it was established"
    );
    assert!(
        text.contains("CompressCandidates"),
        "a claimed rule must address the CompressCandidates re-bundling path, \
         which has different draw arithmetic"
    );
    assert!(
        text.contains("greedy_tensor_search.cpp"),
        "a claimed rule must cite the upstream source it was derived from"
    );
}
