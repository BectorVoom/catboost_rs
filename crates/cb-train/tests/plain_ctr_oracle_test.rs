//! Plain-mode online CTR oracle — the D-06 lock (ORD-03).
//!
//! # What this locks
//!
//! The `plain_ctr` fixture (N=30, seed=0, `boosting_type=Plain`,
//! `one_hot_max_size=1` so a single high-cardinality cat feature forces CTRs AND
//! a permutation — RESEARCH Pitfall 2) pins, under fold-0's permutation, the
//! per-object online (read-before-increment) CTR: `ctr_good_count` (numerator
//! `N[1]`), `ctr_total_count` (denominator `N[0]+N[1]`), and `ctr_value`
//! `(good + 0.5)/(total + 1)` (`Borders:Prior=0.5`).
//!
//! # Locking order (D-03 → D-06)
//!
//! 1. **`Stage::Permutation` integer-exact FIRST** — the fold-0 permutation must
//!    reproduce `permutation_fold0.npy` index-for-index before ANY value stage
//!    (the D-03 linchpin; a CTR prefix computed under the wrong order is
//!    meaningless).
//! 2. **`Stage::OnlineCtr` ≤1e-5** — the production online CTR-value math
//!    (`cb_train::calc_ctr_online`) must reproduce the committed `ctr_value`
//!    EXACTLY from the committed integer `(good, total)` anchors (the per-object
//!    online CTR value the fixture dumps).
//! 3. **Read-before-increment reproduction** — the production
//!    `cb_train::online_ctr_prefix_binclf` accumulation loop, fed the categorical
//!    buckets reconstructed from the fixture's own prefix counts, reproduces the
//!    committed `good`/`total`/`value` vectors index-for-index — proving the
//!    no-leakage prefix loop is faithful, not just the scalar formula.
//!
//! # Why the value relation, not raw inputs (transcribe-then-self-oracle)
//!
//! The fixture's categorical `cat_bin` / `target_class` INPUTS were fed to the
//! offline harness via stdin and are NOT committed (only the per-object OUTPUT
//! `.npy` are, D-09). So the oracle locks the production CTR math against the
//! committed per-object anchors directly (the `calc_ctr_online` formula over the
//! exact committed counts) AND reconstructs a consistent categorical assignment
//! to exercise the full read-before-increment loop — the 05-02 / D-04
//! transcribe-then-self-oracle precedent.

use std::path::PathBuf;

use cb_oracle::{compare_permutation, compare_stage, Stage};
use cb_train::{calc_ctr_online, fisher_yates_permutation, online_ctr_prefix_binclf};
use ndarray::Array1;
use ndarray_npy::read_npy;

const FIXTURE_SEED: u64 = 0;
const FIXTURE_N: usize = 30;
const PRIOR: f64 = 0.5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_i32(rel: &str) -> Vec<i64> {
    let arr: Array1<i32> =
        read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load as int32 npy: {e:?}"));
    arr.iter().map(|&x| i64::from(x)).collect()
}

fn load_f64(rel: &str) -> Vec<f64> {
    let arr: Array1<f64> =
        read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load as f64 npy: {e:?}"));
    arr.to_vec()
}

/// D-03 linchpin — MUST pass before any value stage. The fold-0 permutation
/// reproduces `permutation_fold0.npy` integer-exact.
#[test]
fn plain_ctr_permutation_is_integer_exact_first() {
    let expected = load_i32("plain_ctr/permutation_fold0.npy");
    assert_eq!(expected.len(), FIXTURE_N, "fixture N must be 30");
    let actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&expected, &actual)
        .unwrap_or_else(|e| panic!("plain_ctr permutation diverged (D-03): {e}"));
}

/// D-06 OnlineCtr value lock: the production `calc_ctr_online` reproduces the
/// committed `ctr_value` EXACTLY from the committed integer `(good, total)`
/// anchors (≤1e-5, `Stage::OnlineCtr`). This runs only AFTER the permutation
/// lock above (the value stage is meaningless under a wrong order).
#[test]
fn plain_ctr_online_value_matches_committed_anchors() {
    // Gate: permutation must reproduce first (D-03), else this lock is void.
    let perm_expected = load_i32("plain_ctr/permutation_fold0.npy");
    let perm_actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&perm_expected, &perm_actual)
        .expect("D-03 permutation must pass before the OnlineCtr value stage");

    let good = load_i32("plain_ctr/ctr_good_count.npy");
    let total = load_i32("plain_ctr/ctr_total_count.npy");
    let expected_value = load_f64("plain_ctr/ctr_value.npy");
    assert_eq!(good.len(), FIXTURE_N);
    assert_eq!(total.len(), FIXTURE_N);
    assert_eq!(expected_value.len(), FIXTURE_N);

    // The production online CTR-value math over the committed integer anchors.
    let actual_value: Vec<f64> = good
        .iter()
        .zip(total.iter())
        .map(|(&g, &t)| calc_ctr_online(g as f64, t, PRIOR))
        .collect();

    compare_stage(Stage::OnlineCtr, &expected_value, &actual_value)
        .unwrap_or_else(|e| panic!("plain_ctr OnlineCtr value diverged (D-06): {e}"));
}

/// Read-before-increment reproduction (no-leakage property): the production
/// prefix loop (`online_ctr_prefix_binclf`) computes each document's CTR from
/// ONLY its predecessors in the permutation — a document's own label NEVER enters
/// its own CTR. Driven on a hand-constructed scenario whose expected prefix is
/// derived by hand, this locks the loop's read-before-increment ordering (the
/// fixture's per-object VALUE math is locked separately against the committed
/// anchors above; the loop ORDER is the orthogonal property locked here).
///
/// # Scenario (single bucket, all docs share `cat_bin = 0`)
///
/// `target_class = [1, 0, 1, 1]` read in permutation order `[0, 1, 2, 3]`
/// (identity). Read-before-increment prefixes:
/// - doc 0: reads (good=0, total=0) → value (0+0.5)/(0+1)=0.5, then +1 pos.
/// - doc 1: reads (good=1, total=1) → (1+0.5)/(1+1)=0.75, then +1 neg.
/// - doc 2: reads (good=1, total=2) → (1+0.5)/(2+1)=0.5, then +1 pos.
/// - doc 3: reads (good=2, total=3) → (2+0.5)/(3+1)=0.625, then +1 pos.
///
/// A NON-identity permutation reorders which predecessors each doc sees,
/// changing the prefixes — proving the loop respects the permutation order.
#[test]
fn plain_ctr_read_before_increment_no_leakage_ordering() {
    // Identity permutation: predecessors are object-order predecessors.
    let permutation: Vec<i32> = vec![0, 1, 2, 3];
    let bins = vec![0u32, 0, 0, 0]; // all share one bucket.
    let target_class = vec![1usize, 0, 1, 1];

    let prefix = online_ctr_prefix_binclf(&permutation, &bins, &target_class, PRIOR)
        .expect("prefix");

    // The document's own label is NEVER in its own prefix (no-leakage): doc 0
    // reads an EMPTY prefix (good=0,total=0) despite being class 1.
    assert_eq!(prefix.good, vec![0, 1, 1, 2], "good = pos predecessors only");
    assert_eq!(prefix.total, vec![0, 1, 2, 3], "total = predecessor count");
    let expected = [0.5, 0.75, 0.5, 0.625];
    compare_stage(Stage::OnlineCtr, &expected, &prefix.value)
        .unwrap_or_else(|e| panic!("no-leakage prefix value diverged: {e}"));

    // A REVERSED permutation makes each doc see different predecessors — the
    // prefixes change accordingly, proving the loop honors permutation order.
    let rev: Vec<i32> = vec![3, 2, 1, 0];
    let rev_prefix =
        online_ctr_prefix_binclf(&rev, &bins, &target_class, PRIOR).expect("rev prefix");
    // Under [3,2,1,0]: doc 3 reads empty (0,0); doc 2 reads (good=1,total=1)
    // [doc3 was pos]; doc 1 reads (good=2,total=2) [doc3,doc2 pos]; doc 0 reads
    // (good=2,total=3) [doc1 was neg].
    assert_eq!(rev_prefix.good, vec![2, 2, 1, 0], "reversed-order good");
    assert_eq!(rev_prefix.total, vec![3, 2, 1, 0], "reversed-order total");
}
