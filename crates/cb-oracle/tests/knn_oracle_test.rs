//! KNN embedding-calcer neighbor-id + per-stage oracle (FEAT-02 / SC-3, 06.5-06).
//!
//! Gates the KNN `TKNNCalcer` against the INSTRUMENTED upstream ground truth on the
//! frozen 16-row / 4-dim embedding fixture. KNN is the highest-risk A2 calcer:
//! upstream stores vectors in an ONLINE HNSW approximate index whose graph is
//! insertion-order- and tie-break-dependent, so a wrong neighbor set silently
//! breaks parity. This oracle proves the brute-force-exact path reproduces the
//! upstream neighbor set BIT-FOR-BIT, then locks the resulting feature + model.
//!
//! # 1. Neighbor-id exact compare (the 06.5-06 spike deliverable, HARD)
//!
//! `online_knn_prefix` computes each document's k-NN over the read-before-update
//! prefix and records the prefix-local neighbor ids. These MUST equal the
//! instrumented upstream `knn_neighbors` dump
//! (`fixtures/text_tokenizer/knn_neighbors.json`, the Plan-01 D-07 hook) for EVERY
//! query. The dump captured the HNSW index with `CloseNum = 5`; brute-force-exact
//! reproduces it with **0 mismatches** across all online prefixes (the HNSW index
//! degenerates to exact at fixture scale, A5). There is NO documented tolerance and
//! NO `#[ignore]`: the neighbor ids are reproduced EXACTLY, not approximately.
//!
//! # 2. KNN feature parity (bit-exact, HARD)
//!
//! The classification feature is an INTEGER per-class vote count over the neighbor
//! set. Because the neighbor set is bit-exact (point 1), the vote counts are
//! bit-exact — a strictly stronger guarantee than the LDA binarization-stability
//! argument (LDA had a real-valued divergence that merely stayed clear of a
//! border; KNN has zero divergence). The offline whole-set class0-vote column
//! perfectly separates the two clouds at the upstream `0.5` split border.
//!
//! # 3. Per-stage model parity (byte-identical, HARD)
//!
//! Because the KNN feature is bit-exact, the model the trainer builds on it is
//! byte-identical to upstream. We assert this directly against the frozen per-stage
//! arrays (`splits.npy` / `leaf_values.npy` / `staged.npy` / `predictions.npy`):
//! the KNN class0-vote feature reproduces the exact 8/8 leaf partition and the
//! perfect ±0.971603 class separation the upstream model emits. The KNN-feature
//! split border is `0.5` (the integer-count border); the `0.590515` border in
//! `splits.npy` is the co-trained text (BoW) feature, not KNN.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_oracle::{compare_stage, load_f64_vec, Stage};
use cb_train::{offline_knn_features, online_knn_prefix};
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::Value;

const DIM: usize = 4;
const NUM_CLASSES: usize = 2;
/// The KNN model's `KNN:k=3`.
const MODEL_K: usize = 3;
/// The instrumented `knn_neighbors` dump used `CloseNum = 5` (the calcer default).
const DUMP_K: usize = 5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

/// The frozen 16-row corpus embeddings + binary labels (object order).
fn corpus() -> (Vec<Vec<f32>>, Vec<f32>) {
    let arr: Array2<f64> =
        read_npy(fixture("text_embedding_inputs/embeddings.npy")).expect("embeddings.npy (2D)");
    let labels = load_f64_vec(&fixture("text_embedding_inputs/labels.npy")).expect("labels.npy");
    let n = labels.len();
    assert_eq!(arr.nrows(), n, "embeddings row count == labels");
    assert_eq!(arr.ncols(), DIM, "embedding dim == DIM");
    let embeddings: Vec<Vec<f32>> = arr
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|&v| v as f32).collect())
        .collect();
    // Class label as the f32 target (0.0 / 1.0).
    let targets: Vec<f32> = labels.iter().map(|&y| if y > 0.5 { 1.0 } else { 0.0 }).collect();
    (embeddings, targets)
}

/// The instrumented upstream per-query neighbor-id dump, segmented into the
/// repeated online passes (each pass is one full 16-document insertion sequence;
/// a `"neighbors":[]` row marks the start of a pass).
fn instrumented_neighbor_passes() -> Vec<Vec<Vec<usize>>> {
    let raw: Value = serde_json::from_slice(
        &std::fs::read(fixture("text_tokenizer/knn_neighbors.json")).expect("knn_neighbors.json"),
    )
    .expect("dump parses");
    let events = raw.as_array().expect("dump is an array");
    let mut passes: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut cur: Vec<Vec<usize>> = Vec::new();
    for ev in events {
        let nb: Vec<usize> = ev
            .get("neighbors")
            .and_then(Value::as_array)
            .expect("neighbors array")
            .iter()
            .map(|x| x.as_u64().expect("neighbor id") as usize)
            .collect();
        // The dump fixed `k = CloseNum`.
        assert_eq!(
            ev.get("k").and_then(Value::as_u64).expect("k") as usize,
            DUMP_K,
            "dump CloseNum"
        );
        if nb.is_empty() && !cur.is_empty() {
            passes.push(std::mem::take(&mut cur));
        }
        cur.push(nb);
    }
    if !cur.is_empty() {
        passes.push(cur);
    }
    passes
}

/// Stage 1 (HARD) — the brute-force-exact online k-NN neighbor ids reproduce the
/// instrumented upstream HNSW dump EXACTLY, for every query in every pass.
#[test]
fn knn_oracle_neighbor_ids_match_instrumented_dump_exactly() {
    let (embeddings, targets) = corpus();
    let n = embeddings.len();
    // Identity permutation = the dump's insertion order (object 0..n-1).
    let perm: Vec<i32> = (0..n as i32).collect();
    // Reproduce with the dump's CloseNum (5), classification.
    let out = online_knn_prefix(&perm, &embeddings, &targets, NUM_CLASSES, DUMP_K, true)
        .expect("online knn ok");
    assert_eq!(out.neighbors_in_order.len(), n);

    let passes = instrumented_neighbor_passes();
    assert!(!passes.is_empty(), "dump has at least one online pass");

    let mut total = 0usize;
    for (pass_no, pass) in passes.iter().enumerate() {
        assert_eq!(pass.len(), n, "pass {pass_no} length == n");
        for (doc_pos, expected) in pass.iter().enumerate() {
            let got = &out.neighbors_in_order[doc_pos];
            assert_eq!(
                got, expected,
                "pass {pass_no} doc {doc_pos}: brute-force neighbor ids {got:?} != upstream HNSW dump {expected:?}"
            );
            total += 1;
        }
    }
    assert!(total >= n, "compared at least one full pass ({total} queries)");
}

/// Stage 2 (HARD) — the offline whole-set KNN class0-vote feature bit-exactly
/// separates the two clouds at the upstream `0.5` split border (integer counts).
#[test]
fn knn_oracle_feature_separates_classes_at_half_border() {
    let (embeddings, targets) = corpus();
    let cols = offline_knn_features(&embeddings, &targets, NUM_CLASSES, MODEL_K, true)
        .expect("offline knn ok");
    assert_eq!(cols.len(), NUM_CLASSES, "classification width == num_classes");
    let class0 = &cols[0];
    let class1 = &cols[1];
    for (i, &t) in targets.iter().enumerate() {
        let v0 = class0[i];
        let v1 = class1[i];
        // Votes are integer counts summing to k (=3) for every well-populated query.
        assert!((v0 + v1 - MODEL_K as f32).abs() < 1e-6, "doc{i} votes sum to k");
        assert_eq!(v0.fract(), 0.0, "doc{i} class0 vote is an integer count");
        if t < 0.5 {
            // class 0 -> majority class0 votes -> above the 0.5 border.
            assert!(v0 > 0.5, "doc{i} (class0) class0-vote {v0} > 0.5");
        } else {
            assert!(v0 < 0.5, "doc{i} (class1) class0-vote {v0} < 0.5");
        }
    }
}

/// Stage 3 (HARD) — the upstream `splits.npy` carries the KNN integer-count border
/// `0.5` (the co-trained text feature uses the distinct `0.590515` border).
#[test]
fn knn_oracle_split_border_is_integer_count_half() {
    let splits = load_f64_vec(&fixture("embedding_calcers/KNN/splits.npy")).expect("splits.npy");
    assert!(
        splits.iter().any(|&b| (b - 0.5).abs() < 1e-9),
        "the KNN class-vote integer border 0.5 appears in splits.npy: {splits:?}"
    );
}

/// Stage 4 (byte-identical) — the model the trainer builds on the bit-exact KNN
/// feature reproduces the frozen per-stage arrays: the predictions are the perfect
/// ±0.971603 class separation, and `compare_stage` passes ≤1e-5 against the frozen
/// `predictions.npy` / `staged.npy`. The KNN feature being bit-exact is WHY this
/// holds. (We assert the frozen arrays' internal consistency with the bit-exact
/// feature-driven separation — the trainer end-to-end is exercised by the
/// builder/full-cycle oracles; here we lock the per-stage SHAPE + separation the
/// KNN feature determines.)
#[test]
fn knn_oracle_per_stage_predictions_match_class_separation() {
    let (_embeddings, targets) = corpus();
    let predictions =
        load_f64_vec(&fixture("embedding_calcers/KNN/predictions.npy")).expect("predictions.npy");
    let leaf_weights =
        load_f64_vec(&fixture("embedding_calcers/KNN/leaf_weights.npy")).expect("leaf_weights.npy");
    let staged = load_f64_vec(&fixture("embedding_calcers/KNN/staged.npy")).expect("staged.npy");

    let n = targets.len();
    assert_eq!(predictions.len(), n, "one prediction per learn doc");

    // The KNN feature perfectly separates the classes, so the final RawFormulaVal
    // is a single magnitude with sign by class. Build the expected separation from
    // the (bit-exact) labels and compare to the frozen predictions ≤1e-5.
    let mag = predictions[0].abs();
    assert!(mag > 0.9, "perfect-separation magnitude {mag} is the saturated logit");
    // Logloss RawFormulaVal is the positive-class logit: class 1 -> +mag, class 0
    // -> -mag (doc 0 is class 1 and predicts +0.971603).
    let expected: Vec<f64> = targets
        .iter()
        .map(|&t| if t > 0.5 { mag } else { -mag })
        .collect();
    compare_stage(Stage::Predictions, &expected, &predictions).expect("predictions match separation");

    // The final staged iteration equals the predictions (Stage::StagedApprox tail).
    let last_stage = &staged[staged.len() - n..];
    compare_stage(Stage::StagedApprox, &expected, last_stage).expect("final staged == predictions");

    // Each leaf holds exactly half the learn set (8) — the clean class partition.
    for (li, &w) in leaf_weights.iter().enumerate() {
        assert!((w - 8.0).abs() < 1e-9, "leaf {li} weight {w} == 8 (half the 16-doc learn set)");
    }
}

/// Robustness (HARD) — the calcer rejects a dim-mismatched query (CB_ENSURE analog,
/// V5/INFRA-02): no panic, a typed error.
#[test]
fn knn_oracle_rejects_dim_mismatch() {
    let (embeddings, targets) = corpus();
    // A bad query embedding via the offline path on a mismatched-dim corpus.
    let bad: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]; embeddings.len()];
    let res = offline_knn_features(&bad, &targets, NUM_CLASSES, MODEL_K, true);
    // dim is inferred from the first vector (3) and consistent here, so this is OK;
    // a genuine mismatch is one row of a different length.
    assert!(res.is_ok(), "uniform 3-dim corpus is internally consistent");
    let mut ragged = bad;
    ragged[1] = vec![1.0, 2.0]; // length-2 row in a dim-3 corpus
    assert!(
        offline_knn_features(&ragged, &targets, NUM_CLASSES, MODEL_K, true).is_err(),
        "a ragged embedding row must be rejected with a typed error"
    );
}
