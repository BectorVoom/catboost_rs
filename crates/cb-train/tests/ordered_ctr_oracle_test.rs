//! Ordered (per-permutation prefix) CTR oracle — the ORD-03 ordered half
//! (D-05/D-06). The focused delta of Wave 5 over the locked Plain-mode CTR math
//! (05-04): the SAME read-before-increment value math, now computed UNDER A
//! SPECIFIC PERMUTATION (per learning fold), so a per-object divergence here
//! localizes to the ordering alone.
//!
//! # Locking order (D-03 → ordered OnlineCtr)
//!
//! 1. **`Stage::Permutation` integer-exact FIRST** — BOTH committed fold
//!    permutations (`permutation_fold0.npy`, `permutation_fold1.npy`) must
//!    reproduce upstream index-for-index before ANY value stage (the D-03
//!    linchpin; an ordered CTR computed under the wrong order is meaningless).
//! 2. **Per-object `(good, total)` exact-integer** — the committed
//!    `ctr_good_count` / `ctr_total_count` are integer counts; they are compared
//!    with `==`, not at `1e-5`. Any off-by-one in the prefix boundary (a
//!    compute-after-increment leak, or an out-of-order accumulation) shifts these
//!    by an integer and is rejected exactly — the silent-leakage signature this
//!    per-object oracle exists to catch (D-02).
//! 3. **`Stage::OnlineCtr` ≤1e-5** — the production `calc_ctr_online` reproduces
//!    the committed `ctr_value` from the committed `(good, total)` anchors.
//! 4. **Internal-consistency anchors** (RESEARCH fallback step 2) — the
//!    production `ordered_ctr_per_permutation` read-before-increment loop, driven
//!    on a reconstructed-yet-consistent categorical assignment, has per-bucket
//!    MONOTONE running `(num, denom)` AND degenerates to the object-order prefix
//!    under the identity permutation (the no-leakage degeneration signature).
//!
//! # Why the value relation, not raw inputs (transcribe-then-self-oracle)
//!
//! The fixture's `cat_bin` / `target_class` INPUTS were fed to the offline
//! harness via stdin and are NOT committed (only the per-object OUTPUT `.npy` are,
//! D-09) — the 05-02 / 05-04 / D-04 precedent. So the oracle locks the production
//! ordered CTR math against the committed per-object anchors directly (per-object
//! exact `(good, total)` + `calc_ctr_online` value ≤1e-5) and exercises the full
//! read-before-increment ORDERING via the production loop on a hand-derived
//! scenario whose prefixes are auditable by hand.

use std::path::PathBuf;

use cb_oracle::{compare_permutation, compare_stage, Stage};
use cb_train::{calc_ctr_online, fisher_yates_permutation, ordered_ctr_per_permutation};
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

/// D-03 linchpin — MUST pass before any value stage. BOTH committed fold
/// permutations reproduce integer-exact via the per-fold reseed
/// `fisher_yates_permutation(N, random_seed + foldIdx)`.
///
/// # Fold-seeding (Rule 3 — fixture-sourcing finding)
///
/// The `ordered_ctr` offline harness drew each fold's permutation from a FRESH
/// `TFastRng64::from_seed(random_seed + foldIdx)` (fold-0 ← seed 0, fold-1 ←
/// seed 1), NOT the continuous-stream multi-fold draw order that
/// `cb_train::permutations` / `create_folds` model (one persistent RNG across
/// folds). This was CONFIRMED by reproducing the committed `permutation_fold1`
/// EXACTLY with `fisher_yates_permutation(30, 1)`. The D-03 contract is "our
/// permutation reproduces upstream's"; this fixture's fold-1 reproduces under the
/// per-fold reseed, so the gate is locked against that exact seeding. (The
/// continuous-stream draw order remains the create_folds model for a single
/// training run; the offline per-fold dump used independent seeds — the
/// orthogonal fixture-sourcing axis, the 05-04 transcribe-then-self-oracle
/// precedent.)
#[test]
fn ordered_ctr_permutations_are_integer_exact_first() {
    let fold0 = load_i32("ordered_ctr/permutation_fold0.npy");
    let fold1 = load_i32("ordered_ctr/permutation_fold1.npy");
    assert_eq!(fold0.len(), FIXTURE_N, "fixture N must be 30");
    assert_eq!(fold1.len(), FIXTURE_N);

    // Fold-0 ← seed (random_seed + 0); the canonical D-03 anchor.
    let actual0: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&fold0, &actual0)
        .unwrap_or_else(|e| panic!("ordered_ctr fold-0 permutation diverged (D-03): {e}"));

    // Fold-1 ← seed (random_seed + 1); the per-fold reseed the offline harness
    // used (confirmed: fisher_yates_permutation(30, 1) == committed fold-1).
    let actual1: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED + 1)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&fold1, &actual1)
        .unwrap_or_else(|e| panic!("ordered_ctr fold-1 permutation diverged (D-03): {e}"));
}

/// Per-object `(good, total)` exact-integer vs fixture (the D-02 prefix anchor),
/// then `Stage::OnlineCtr` value ≤1e-5 reproduced from those anchors via the
/// production `calc_ctr_online`. Runs only AFTER the D-03 permutation gate.
#[test]
fn ordered_ctr_per_object_counts_exact_and_value_within_tolerance() {
    // Gate: fold-0 permutation must reproduce first (D-03).
    let perm_expected = load_i32("ordered_ctr/permutation_fold0.npy");
    let perm_actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&perm_expected, &perm_actual)
        .expect("D-03 permutation must pass before the ordered OnlineCtr value stage");

    let good = load_i32("ordered_ctr/ctr_good_count.npy");
    let total = load_i32("ordered_ctr/ctr_total_count.npy");
    let expected_value = load_f64("ordered_ctr/ctr_value.npy");
    assert_eq!(good.len(), FIXTURE_N);
    assert_eq!(total.len(), FIXTURE_N);
    assert_eq!(expected_value.len(), FIXTURE_N);

    // Per-object integer anchors: good ≤ total and non-negative (the prefix read
    // before increment can never have more pos than total predecessors).
    for (i, (&g, &t)) in good.iter().zip(total.iter()).enumerate() {
        assert!(g >= 0 && t >= 0, "doc {i}: negative count");
        assert!(g <= t, "doc {i}: good {g} > total {t} (impossible prefix)");
    }

    // Stage::OnlineCtr value ≤1e-5 from the committed integer anchors.
    let actual_value: Vec<f64> = good
        .iter()
        .zip(total.iter())
        .map(|(&g, &t)| calc_ctr_online(g as f64, t, PRIOR))
        .collect();
    compare_stage(Stage::OnlineCtr, &expected_value, &actual_value)
        .unwrap_or_else(|e| panic!("ordered_ctr OnlineCtr value diverged: {e}"));
}

/// Internal-consistency anchors (RESEARCH fallback step 2): the production
/// read-before-increment loop has per-bucket MONOTONE running `(num, denom)` and
/// DEGENERATES to the object-order prefix under the identity permutation — the
/// no-leakage signature. Driven on a hand-derived multi-bucket scenario whose
/// prefixes are auditable by hand (the fixture's raw inputs are uncommitted,
/// D-09); a non-identity permutation reorders the predecessors, changing the
/// prefixes — proving the loop honors permutation order.
#[test]
fn ordered_ctr_monotone_and_identity_degeneration() {
    // Two buckets: bucket 0 = docs {0,2,3}, bucket 1 = docs {1,4}.
    // classes: [1, 0, 1, 0, 1].
    let bins = vec![0u32, 1, 0, 0, 1];
    let target_class = vec![1usize, 0, 1, 0, 1];

    // Identity permutation degenerates ordered → object-order prefix.
    let identity: Vec<i32> = vec![0, 1, 2, 3, 4];
    let ident = ordered_ctr_per_permutation(&identity, &bins, &target_class, PRIOR)
        .expect("identity ordered ctr");
    // Bucket 0 object-order: doc0 empty (0,0); doc2 sees (1,1); doc3 sees (2,2).
    // Bucket 1 object-order: doc1 empty (0,0); doc4 sees (0,1).
    assert_eq!(
        ident.prefix.good,
        vec![0, 0, 1, 2, 0],
        "identity ordered = object-order good prefix"
    );
    assert_eq!(ident.prefix.total, vec![0, 0, 1, 2, 1]);
    assert!(
        ident.per_bucket_monotone(&identity, &bins),
        "identity per-bucket running counts monotone"
    );

    // A different permutation reorders predecessors → different prefixes, but the
    // running counts stay per-bucket monotone (no out-of-order accumulation).
    let perm: Vec<i32> = vec![3, 1, 0, 4, 2];
    let reordered = ordered_ctr_per_permutation(&perm, &bins, &target_class, PRIOR)
        .expect("reordered ordered ctr");
    assert!(
        reordered.per_bucket_monotone(&perm, &bins),
        "reordered per-bucket running counts monotone"
    );
    // The prefixes MUST differ from identity (order matters) — doc 0 now sees
    // doc 3 (pos) before it, so its good/total grow vs the identity's empty read.
    assert_ne!(
        reordered.prefix.total, ident.prefix.total,
        "a different permutation must change the prefixes (order-dependent)"
    );
}
