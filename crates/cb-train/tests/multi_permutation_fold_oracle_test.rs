//! Integer-exact `permutation_count>=2` AveragingFold draw-order oracle (ORD-01 /
//! WR-01, Plan 05-15) — ANCHORED on a committed catboost 1.2.10 dump.
//!
//! # Why this gate exists
//!
//! The production-default `permutation_count=4` AveragingFold draw was, until Plan
//! 05-15, NEVER validated against upstream catboost (WR-01). The pre-averaging
//! `GenRand` draw in `create_folds` used to fire before the FIRST learning shuffle
//! (`idx == 1`), which is correct ONLY at `permutation_count=1` (where
//! `learning_folds == 1` so `idx == 1` IS the averaging fold). For
//! `permutation_count > 1` the pre-draw fired too early and the averaging fold's
//! permutation landed at an RNG call-count never checked against catboost. Plan
//! 05-15 moved the pre-draw to `idx == learning_folds` (the averaging-fold
//! position) for ALL `permutation_count`; this oracle PROVES the corrected draw
//! order against UPSTREAM — not against cb-core's own `TFastRng64`.
//!
//! # The upstream anchor (the MANDATORY authority)
//!
//! catboost's Python API does not expose the internal
//! `AveragingFold->LearnPermutation` array, but the trained-model tree-0
//! `leaf_weights` ARE the AveragingFold partition counts
//! (05-CTR-LEAF-VALUE-RESEARCH.md Open-Q5: "model.json leaf_weights are the
//! AveragingFold (shuffled) partition counts"). A WRONG pre-averaging advance
//! count yields a DIFFERENT AveragingFold permutation, hence a DIFFERENT
//! partition, hence DIFFERENT leaf_weights — so asserting that the partition the
//! cb-train AveragingFold permutation produces over the online-prefix CTR column
//! equals catboost's COMMITTED leaf_weights validates the advance count against
//! catboost itself. This is observable UPSTREAM output (committed under
//! `fixtures/multi_permutation_fold/` by `gen_multi_permutation_fold.py`), the
//! mandatory authority that catches the exact WR-01 risk.
//!
//! The empirical anchor (catboost 1.2.10, tensor_ctr_e2e config family):
//!   pc=1 -> tree0 leaf_weights [6, 0, 7, 17]   (learning_folds == 1)
//!   pc=2 -> tree0 leaf_weights [6, 0, 7, 17]   (learning_folds == 1; same draw stream as pc=1)
//!   pc=4 -> tree0 leaf_weights [6, 0, 10, 14]  (learning_folds == 3)
//! The pc=2 == pc=1 equality is the smoking gun for the WR-01 fix: with the
//! CORRECT `idx == learning_folds` guard the averaging shuffle is preceded by
//! zero learning shuffles at both pc=1 and pc=2 (`learning_folds == 1`), so the
//! partition matches catboost; the OLD (first-learning-shuffle) guard would have
//! diverged pc=2. The pc=2 anchor is the MANDATORY upstream authority this gate
//! enforces.
//!
//! # pc=4: CLOSED integer-exact via the instrumented draw accounting (Plan 05-17)
//!
//! At `permutation_count=4` (`learning_folds == 3`) the cb-train AveragingFold
//! permutation now reproduces catboost 1.2.10's committed partition `[6,0,10,14]`
//! integer-exact. The earlier divergence (`[6,0,8,16]`) was resolved by the Plan
//! 05-17 instrumented C++ harness (`instrument_fold_rng.cpp`), a deliberate,
//! user-approved C++ instrumentation deviation authorized by the 2026-06-15
//! CONTEXT decision revision (scoped to this gap only). The harness logged
//! catboost's per-fold `TRestorableFastRng64::GetCallCount()` and DISCOVERED the
//! ground-truth rule the empirical 05-15 partition sweep could not reach: the
//! AveragingFold shuffle starts at RNG call-count `== learning_folds` (each of
//! the `learning_folds` fold positions consumes exactly ONE non-shuffle
//! pre-averaging GenRand — the per-fold upstream RNG consumption:
//! InitOnlineEstimatedFeatures / target-classifier / per-fold CTR-grid). This ONE
//! consistent rule reproduces BOTH the e2e-bit-exact pc=1/pc=2 partition
//! `[6,0,7,17]` (learning_folds == 1, C == 1) AND the pc=4 partition `[6,0,10,14]`
//! (learning_folds == 3, C == 3), reducing to the prior single pre-averaging
//! GenRand at `learning_folds == 1` (no regression on pc=1/pc=2). The discovered
//! per-fold accounting is committed as
//! `fixtures/multi_permutation_fold/rng_draw_accounting.json`. The pc=4 test below
//! is now a HARD integer-exact equality against the committed catboost
//! `[6,0,10,14]` — no pin, no assert_ne, no #[ignore].
//!
//! # Secondary cross-check (NOT the authority)
//!
//! A self-derived `TFastRng64` draw sequence (identity Folds[0] -> learning folds
//! 1..lf-1 each a Fisher-Yates pass -> one pre-averaging GenRand -> the averaging
//! Fisher-Yates pass) is asserted to AGREE with `create_folds`. Per WR-01 this is
//! ONLY a self-consistency cross-check — a self-oracle bakes in the same
//! advance-count assumption the implementation makes and cannot catch a wrong
//! advance count. The committed catboost leaf_weights are the validating
//! authority.
//!
//! Integer-exact comparison only (`compare_permutation` / `Stage::Permutation`,
//! the D-03 comparator — NOT a 1e-5 value check). Every test runs
//! unconditionally (none are ignore-attributed / skipped).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::HashMap;
use std::path::PathBuf;

use cb_core::TFastRng64;
use cb_data::{calc_cat_feature_hash, stringify_int_category};
use cb_oracle::{compare_permutation, Stage};
use cb_train::{
    calc_ctr_online_bin, create_folds, online_ctr_prefix_binclf, Fold,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const FIXTURE_N: usize = 30;
const FIXTURE_SEED: u64 = 0;
const FOLD_LEN_MULTIPLIER: f64 = 2.0;
const PRIOR: f64 = 0.5;
const CTR_BORDER_COUNT: usize = 15;
/// Tree-0 CTR split borders for the winning single-feature {0} CTR (read from the
/// committed upstream model_pc{N}.json `ctrs[0].borders` = [2.999, 7.999]). bit0
/// (forward order, first split listed) tests the HIGH border; bit1 the LOW border.
const HIGH_BORDER: f64 = 7.999_999_046_325_684;
const LOW_BORDER: f64 = 2.999_999_046_325_684;

/// Resolve a path under `cb-train/tests/fixtures/` from cb-train's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(rel)
}

/// Load the catboost-1.2.10 committed tree-0 leaf_weights (the AveragingFold
/// partition counts) for a given permutation_count, parsed from
/// `multi_permutation_fold/leaf_weights.json`.
fn upstream_leaf_weights(pc: usize) -> Vec<i64> {
    let raw = std::fs::read_to_string(fixture("multi_permutation_fold/leaf_weights.json"))
        .expect("multi_permutation_fold/leaf_weights.json must exist (committed catboost 1.2.10 dump)");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("leaf_weights.json is valid JSON");
    let lw = v
        .get(pc.to_string())
        .and_then(|e| e.get("tree0_leaf_weights"))
        .and_then(|a| a.as_array())
        .unwrap_or_else(|| panic!("leaf_weights.json must record pc={pc} tree0_leaf_weights"));
    lw.iter()
        .map(|x| x.as_i64().expect("leaf_weights are integers"))
        .collect()
}

/// Build the tensor_ctr_e2e fold set at `permutation_count`: hasCtrs => a learning
/// permutation is needed; Plain => `dynamic_body_tail=false`.
fn folds(pc: usize) -> Vec<Fold> {
    create_folds(
        FIXTURE_N,
        pc,
        /* permutation_needed_for_learning = */ true,
        /* dynamic_body_tail = */ false,
        FOLD_LEN_MULTIPLIER,
        FIXTURE_SEED,
    )
}

/// Single-feature {0} perfect-hash bins (first-seen dense remap of
/// `calc_cat_feature_hash(stringify_int_category(code))`) and binarized class —
/// the inputs the online-prefix CTR + tree-0 partition consume.
fn feature0_bins_and_classes() -> (Vec<u32>, Vec<usize>) {
    // The cat columns / labels live with the tensor_ctr_e2e fixture under cb-oracle.
    let tensor = |name: &str| -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("tensor_ctr_e2e")
            .join(name)
    };
    let x: Array2<i32> =
        read_npy(tensor("X_cat.npy")).expect("tensor_ctr_e2e/X_cat.npy must load as int32 [N,2]");
    let y: Vec<f64> = {
        let arr: ndarray::Array1<f64> =
            read_npy(tensor("y.npy")).expect("tensor_ctr_e2e/y.npy must load as float64 [N]");
        arr.to_vec()
    };
    let mut map: HashMap<u32, u32> = HashMap::new();
    let bins: Vec<u32> = (0..x.nrows())
        .map(|i| {
            let s = stringify_int_category(i64::from(x[(i, 0)]));
            let key = calc_cat_feature_hash(&s);
            let next = map.len() as u32;
            *map.entry(key).or_insert(next)
        })
        .collect();
    let classes: Vec<usize> = y.iter().map(|&t| usize::from(t > 0.5)).collect();
    (bins, classes)
}

/// Reproduce catboost's tree-0 AveragingFold partition: the online read-before-
/// increment CTR over `permutation`, quantized via the production
/// `calc_ctr_online_bin` (Borders, prior 0.5, scale 15, shift 0), then split into
/// the 4 oblivious leaves by the tree-0 borders {2.999, 7.999}. Returns the 4
/// leaf counts (the leaf_weights catboost commits).
fn averaging_partition(permutation: &[i32]) -> Vec<i64> {
    let (bins, classes) = feature0_bins_and_classes();
    let prefix = online_ctr_prefix_binclf(permutation, &bins, &classes, PRIOR)
        .expect("online prefix over the averaging permutation");
    let mut part = [0i64; 4];
    for doc in 0..FIXTURE_N {
        let bin = calc_ctr_online_bin(prefix.good[doc] as f64, prefix.total[doc], PRIOR, CTR_BORDER_COUNT);
        // Forward-bit leaf index: bit0 = bin > HIGH_BORDER, bit1 = bin > LOW_BORDER.
        let bit0 = usize::from(bin > HIGH_BORDER);
        let bit1 = usize::from(bin > LOW_BORDER);
        let leaf = bit0 | (bit1 << 1);
        part[leaf] += 1;
    }
    part.to_vec()
}

/// Self-derived (SECONDARY cross-check, NOT the authority) AveragingFold
/// permutation under the Plan 05-17 instrumented draw accounting: drive a
/// persistent `TFastRng64(seed)` with EXACTLY `learning_folds` pre-averaging
/// GenRand draws (one per fold position — the discovered
/// `averaging_shuffle_start_callcount == learning_folds` rule from
/// `rng_draw_accounting.json`), then the AveragingFold Fisher-Yates pass. Per
/// WR-01 this bakes in the same advance-count assumption the implementation makes,
/// so it can only CROSS-CHECK `create_folds`, never validate it against upstream
/// (that is the committed catboost leaf_weights' job). At `learning_folds == 1`
/// this is the single pre-averaging GenRand the pc=1/pc=2 e2e stream locked.
fn self_derived_averaging_permutation(n: usize, seed: u64, learning_folds: usize) -> Vec<i32> {
    let mut rng = TFastRng64::from_seed(seed);
    // One pre-averaging GenRand per fold position: the averaging shuffle begins at
    // RNG call-count == learning_folds.
    for _ in 0..learning_folds {
        rng.gen_rand();
    }
    // AveragingFold Fisher-Yates pass.
    let mut v: Vec<i32> = (0..n as i32).collect();
    for i in 1..n {
        let j = rng.uniform((i as u64) + 1) as usize;
        v.swap(i, j);
    }
    v
}

/// PRIMARY (upstream-anchored, MANDATORY): the partition the cb-train pc=2
/// AveragingFold permutation produces over the online-prefix CTR equals catboost
/// 1.2.10's committed tree-0 leaf_weights integer-exact. This would FAIL if the
/// pre-averaging advance count were wrong (the WR-01 risk).
#[test]
fn multi_permutation_count_two_averaging_matches_catboost_1_2_10() {
    let fs = folds(2);
    let averaging = fs.iter().find(|f| f.is_averaging).expect("averaging fold for pc=2");
    let actual = averaging_partition(&averaging.permutation);
    let expected = upstream_leaf_weights(2);
    compare_permutation(&expected, &actual).unwrap_or_else(|e| {
        panic!(
            "pc=2 AveragingFold partition {actual:?} diverged from committed catboost 1.2.10 leaf_weights {expected:?} [{:?}]: {e}",
            Stage::Permutation
        )
    });
}

/// pc=4 (production default), HARD integer-exact equality (Plan 05-17 closure):
/// the partition the cb-train pc=4 AveragingFold permutation produces over the
/// online-prefix CTR equals catboost 1.2.10's committed tree-0 leaf_weights
/// `[6,0,10,14]` integer-exact via `compare_permutation`. This is the closure of
/// the SC-1 / ORD-01 blocking gap at the production default: `create_folds` now
/// consumes the instrumented per-fold draw accounting (averaging shuffle starts at
/// call-count == learning_folds; see `rng_draw_accounting.json`). NO pin, NO
/// assert_ne, NO #[ignore].
#[test]
fn multi_permutation_count_four_averaging_matches_catboost_1_2_10() {
    let fs = folds(4);
    let averaging = fs.iter().find(|f| f.is_averaging).expect("averaging fold for pc=4");
    let actual = averaging_partition(&averaging.permutation);
    let expected = upstream_leaf_weights(4);
    compare_permutation(&expected, &actual).unwrap_or_else(|e| {
        panic!(
            "pc=4 AveragingFold partition {actual:?} diverged from committed catboost 1.2.10 leaf_weights {expected:?} [{:?}]: {e}",
            Stage::Permutation
        )
    });
}

/// The averaging shuffle begins at RNG call-count == learning_folds (Plan 05-17):
/// the SECONDARY self-derived sequence (learning_folds pre-averaging GenRand draws,
/// then the averaging Fisher-Yates pass) AGREES with `create_folds`'s averaging
/// permutation integer-exact, for pc=2 and pc=4. A self-consistency cross-check
/// ONLY (per WR-01 — the committed catboost leaf_weights above are the validating
/// authority).
#[test]
fn multi_permutation_averaging_starts_at_learning_folds_callcount() {
    for pc in [2usize, 4usize] {
        let learning_folds = usize::max(1, pc - 1);
        let fs = folds(pc);
        let averaging = fs.iter().find(|f| f.is_averaging).expect("averaging fold");
        let expected: Vec<i64> = self_derived_averaging_permutation(FIXTURE_N, FIXTURE_SEED, learning_folds)
            .into_iter()
            .map(i64::from)
            .collect();
        let actual: Vec<i64> = averaging.permutation.iter().map(|&v| i64::from(v)).collect();
        compare_permutation(&expected, &actual).unwrap_or_else(|e| {
            panic!(
                "pc={pc} AveragingFold permutation diverged from the self-derived post-learning-shuffle draw [{:?}]: {e}",
                Stage::Permutation
            )
        });
    }
}

/// The lone learning Folds[0] is the IDENTITY (zero draws) for pc>=2 — upstream's
/// `shuffle = foldIdx != 0`. (The other learning folds 1..lf-1 ARE shuffled; this
/// asserts the FIRST learning fold specifically is the identity.)
#[test]
fn multi_permutation_learning_fold_zero_is_identity() {
    for pc in [2usize, 4usize] {
        let fs = folds(pc);
        let fold0 = fs.first().expect("at least one fold");
        assert!(!fold0.is_averaging, "Folds[0] is a learning fold");
        let identity: Vec<i64> = (0..FIXTURE_N as i64).collect();
        let actual: Vec<i64> = fold0.permutation.iter().map(|&v| i64::from(v)).collect();
        compare_permutation(&identity, &actual).unwrap_or_else(|e| {
            panic!(
                "pc={pc} learning Folds[0] is not the identity [0..30] [{:?}]: {e}",
                Stage::Permutation
            )
        });
    }
}
