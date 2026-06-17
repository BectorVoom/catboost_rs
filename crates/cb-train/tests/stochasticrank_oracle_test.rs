//! StochasticRank per-stage training oracle (LOSS-04 / Plan 06.3-04 Wave C).
//!
//! StochasticRank is the OTHER RNG-stream loss: a Monte-Carlo gradient estimator
//! whose per-doc Gaussian NOISE stream (drawn via `cb_core::std_normal` from
//! `TFastRng64(random_seed + group_index)`) is the parity crux. The ground truth
//! is the OFFLINE instrumented generator
//! (`crates/cb-oracle/generator/stochasticrank_oracle.cpp`), self-oracled
//! bit-for-bit against `cb-core::std_normal`, with its noise/score/order stream
//! frozen at `ranking_corpus/stochasticrank/stochasticrank_rng_groundtruth.jsonl`.
//!
//! [`stochasticrank_rng_draw_log_oracle`] gates the Rust Gaussian noise stream
//! against that frozen ground truth at <= 1e-5 — the integer/f64-exact RNG-draw
//! compare that gates the randomized stream INDEPENDENTLY of the der. This is LIVE.
//!
//! The end-to-end per-stage compare over a trained StochasticRank `model.json` is
//! DEFERRED on the instrumented trainer build (path c — toolchain absent + disk
//! NO-GO; escalate-don't-weaken, D-6.3-03b). NO `#[ignore]`, NO weakened
//! tolerance — see the README STATUS section.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_core::{std_normal, TFastRng64};
use cb_oracle::{compare_stage, Stage};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn json_raw(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat).unwrap_or_else(|| panic!("key {key} not in {line}")) + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

/// Parse the frozen `gauss_draw` noise values (doc order) from the ground truth.
fn load_noise_ground_truth(rel: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(fixture(rel))
        .unwrap_or_else(|e| panic!("{rel} must load (frozen RNG ground truth): {e:?}"));
    text.lines()
        .filter(|l| l.contains("\"event\":\"gauss_draw\""))
        .map(|l| json_raw(l, "noise").parse().unwrap())
        .collect()
}

/// LIVE RNG-draw oracle: the Rust `cb_core::std_normal` stream (the SAME the
/// StochasticRank der consumes) must reproduce the instrumented ground-truth
/// Gaussian noise draws EXACTLY (<= 1e-5). The generator's smallest unit is one
/// group, 3 docs, num_estimations=1, group-0 seed = random_seed(5) + 0; the noise
/// is one std_normal per doc in ascending order.
#[test]
fn stochasticrank_rng_draw_log_oracle() {
    let gt_rel = "ranking_corpus/stochasticrank/stochasticrank_rng_groundtruth.jsonl";
    let count = 3_usize;
    let group_seed = 5_u64; // random_seed 5 + group_index 0.

    let mut rng = TFastRng64::from_seed(group_seed);
    let got: Vec<f64> = (0..count).map(|_| std_normal(&mut rng)).collect();

    let expected = load_noise_ground_truth(gt_rel);
    assert_eq!(
        got.len(),
        expected.len(),
        "StochasticRank noise draw COUNT must match the instrumented ground truth \
         (one std_normal per doc per sample)"
    );
    compare_stage(Stage::StagedApprox, &expected, &got).unwrap_or_else(|e| {
        panic!("StochasticRank: Gaussian noise stream diverged from instrumented ground truth: {e:?}")
    });
}

/// Wired-but-pending end-to-end per-stage compare (deferred trainer fixture). NO
/// `#[ignore]`, NO weakened tolerance.
#[test]
fn stochasticrank_end_to_end_per_stage() {
    let model_json = fixture("ranking_corpus/stochasticrank/model.json");
    if model_json.exists() {
        panic!(
            "ranking_corpus/stochasticrank/model.json now exists — wire the full \
             per-stage compare_stage gate (querywise pointwise leaf) and remove this guard."
        );
    } else {
        let gt = fixture("ranking_corpus/stochasticrank/stochasticrank_rng_groundtruth.jsonl");
        assert!(
            gt.exists(),
            "StochasticRank RNG ground truth must be committed while the end-to-end \
             trainer fixture is deferred (D-6.3-03b)"
        );
    }
}
