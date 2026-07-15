//! FEAT-07 SC-1 — online-HNSW neighbor-SET bit-for-bit oracle.
//!
//! Gates the [`cb_compute::HnswKnnCloud`] port against upstream CatBoost 1.2.10's
//! INSTRUMENTED `knn_neighbors` dump (`fixtures/text_tokenizer/knn_neighbors.json`,
//! the Plan-01 D-07 hook) over the frozen 16-row / 4-dim embedding corpus
//! (`fixtures/text_embedding_inputs/embeddings.npy`), `KNN:k=5`.
//!
//! The dump records, per calcer, 32 `knn_neighbors` events: the first 16 are the
//! ONLINE read-before-update prefix queries (`Compute` over the learn set in
//! identity order, each object queried against the prefix of already-inserted
//! objects BEFORE its own insertion). Prefixes of size <= `MaxNeighbors + 1` (== 6)
//! use upstream's naive-exact path; prefixes of size >= 7 exercise the APPROXIMATE
//! HNSW search + the libc++ heap tie-break. Reproducing the dump index-for-index —
//! including those approximate-path queries — is the bit-for-bit fidelity gate the
//! prior brute-force-exact calcer could not carry (FEAT-07).
//!
//! This is the authoritative neighbor-SET gate; the end-to-end XOR per-stage gate
//! lives in `text_embedding_end_to_end_oracle_test.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_compute::HnswKnnCloud;
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::Value;

const DIM: usize = 4;
const K: usize = 5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(rel)
}

/// The frozen 16-row / 4-dim embedding corpus (cast f64 -> f32 exactly as the
/// calcer ingests it).
fn embeddings() -> Vec<Vec<f32>> {
    let arr: Array2<f64> =
        read_npy(fixture("text_embedding_inputs/embeddings.npy")).expect("embeddings.npy (2D)");
    assert_eq!(arr.ncols(), DIM, "embedding dim == DIM");
    arr.rows()
        .into_iter()
        .map(|row| row.iter().map(|&v| v as f32).collect())
        .collect()
}

/// The upstream instrumented neighbor-id dump, grouped by calcer, in event order.
fn upstream_neighbor_dump() -> Vec<(String, Vec<usize>)> {
    let raw = std::fs::read(fixture("text_tokenizer/knn_neighbors.json")).expect("knn dump");
    let events: Vec<Value> = serde_json::from_slice(&raw).expect("knn dump parses");
    events
        .into_iter()
        .map(|e| {
            let calcer = e["_calcer"].as_str().unwrap_or("").to_owned();
            let neighbors = e["neighbors"]
                .as_array()
                .expect("neighbors array")
                .iter()
                .map(|v| v.as_u64().expect("neighbor id") as usize)
                .collect::<Vec<usize>>();
            (calcer, neighbors)
        })
        .collect()
}

/// SC-1: the online-HNSW cloud reproduces upstream's neighbor SET index-for-index
/// on the ONLINE read-before-update prefix (the first 16 events of each calcer
/// block), including the approximate-path prefixes (size >= 7).
#[test]
fn online_hnsw_prefix_neighbors_match_upstream_dump() {
    let embeds = embeddings();
    let n = embeds.len();
    assert_eq!(n, 16, "frozen 16-row corpus");

    // The dump interleaves calcer blocks (BoW then NaiveBayes), each 32 events; the
    // first 16 of a block are the online prefix queries. Take the first block's
    // online prefix (identical across calcers — same corpus, same identity order).
    let dump = upstream_neighbor_dump();
    let first_calcer = dump.first().map(|(c, _)| c.clone()).expect("dump non-empty");
    let block: Vec<Vec<usize>> = dump
        .iter()
        .filter(|(c, _)| *c == first_calcer)
        .map(|(_, ne)| ne.clone())
        .collect();
    assert!(block.len() >= n, "at least one online prefix per object");
    let expected_online = &block[..n];

    // Replay the ONLINE read-before-update prefix (identity order) through the port.
    let mut cloud = HnswKnnCloud::new(DIM, K).expect("cloud");
    let mut got_online: Vec<Vec<usize>> = Vec::with_capacity(n);
    for embed in &embeds {
        got_online.push(cloud.nearest_neighbors(embed, K).expect("query"));
        cloud.add_vector(embed).expect("insert");
    }

    // Index-for-index equality — NO set relaxation, NO tolerance.
    let mut approx_path_queries = 0usize;
    for (i, (got, want)) in got_online.iter().zip(expected_online.iter()).enumerate() {
        if i >= K + 2 {
            approx_path_queries += 1;
        }
        assert_eq!(
            got, want,
            "online prefix {i}: HNSW neighbor set diverges from upstream dump\n  got  {got:?}\n  want {want:?}"
        );
    }
    assert!(
        approx_path_queries > 0,
        "the gate must exercise the approximate HNSW search path (prefix >= 7)"
    );
}
