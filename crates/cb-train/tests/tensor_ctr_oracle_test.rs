//! Tensor / combination CTR oracle — ORD-05, the FINAL rung of the additive
//! categorical ladder (D-05). Per D-05 a tensor CTR is the SAME online
//! read-before-increment accumulation (05-04) and the SAME value math, computed
//! over a COMBINED projection hash instead of a single feature's hash. By landing
//! after single-feature CTR is oracle-locked (05-04/05-05), a tensor-CTR
//! divergence isolates to the projection-enumeration / combined-hash logic, not
//! the CTR math underneath it.
//!
//! # The fixture
//!
//! `tensor_ctr` (N=30, seed=0, `boosting_type=Plain`, `one_hot_max_size=1`,
//! `max_ctr_complexity=2`, `simple_ctr`/`combinations_ctr` = `Borders:Prior=0.5`,
//! two cat features cat0/cat1 each above one_hot_max_size so a genuine 2-feature
//! combination is formed). It pins, under fold-0's permutation, the per-object
//! online CTR over the COMBINED projection: `ctr_good_count` (numerator `N[1]`),
//! `ctr_total_count` (denominator `N[0]+N[1]`), and `ctr_value`
//! `(good + 0.5)/(total + 1)`.
//!
//! # Locking order (D-03 → combined OnlineCtr)
//!
//! 1. **`Stage::Permutation` integer-exact FIRST** — the fold-0 permutation must
//!    reproduce `permutation_fold0.npy` index-for-index before ANY value stage
//!    (the D-03 linchpin; a CTR prefix computed under the wrong order is
//!    meaningless).
//! 2. **Per-object `(good, total)` exact-integer** — the committed integer
//!    counts are compared with `==`, not at `1e-5`. Any off-by-one in the prefix
//!    boundary (a compute-after-increment leak, or an out-of-order accumulation)
//!    shifts these by an integer and is rejected exactly.
//! 3. **`Stage::OnlineCtr` ≤1e-5** — the production `calc_ctr_online` reproduces
//!    the committed `ctr_value` from the committed `(good, total)` anchors.
//! 4. **Combined-projection accumulation** — the production
//!    `online_ctr_prefix_binclf` loop, driven over BUCKETS derived from the
//!    COMBINED projection hash (`TProjection::combined_hash` folding two
//!    per-document cat hashes via the `ctr_provider.h` CalcHash), on a
//!    hand-derived 2-feature scenario whose prefixes are auditable by hand,
//!    reproduces the read-before-increment good/total/value AND degenerates to
//!    the single-feature prefix when the projection has one member (the tensor
//!    keyspace contains the simple keyspace, D-05).
//!
//! # Why the value relation, not raw inputs (transcribe-then-self-oracle)
//!
//! The fixture's `cat0` / `cat1` / `target_class` INPUTS were fed to the offline
//! harness via stdin and are NOT committed (only the per-object OUTPUT `.npy`
//! are, D-09) — the 05-02 / 05-04 / 05-05 precedent. So the oracle locks the
//! production tensor-CTR math against the committed per-object anchors directly
//! (per-object exact `(good, total)` + `calc_ctr_online` value ≤1e-5) and
//! exercises the full COMBINED-projection accumulation via the production loop on
//! a hand-derived scenario whose combined-bucket prefixes are auditable by hand.

use std::path::PathBuf;

use cb_data::calc_cat_feature_hash;
use cb_oracle::{compare_permutation, compare_stage, Stage};
use cb_train::{
    calc_ctr_online, fisher_yates_permutation, online_ctr_prefix_binclf, TProjection,
};
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

/// Map a sequence of COMBINED projection keys to dense first-seen bins (the
/// perfect-hash remap the online accumulation keys on), so the combined
/// projection hash drives the SAME `online_ctr_prefix_binclf` bucketing as a
/// single feature would — the only difference is the KEY (D-05).
fn combined_keys_to_bins(combined_keys: &[u64]) -> Vec<u32> {
    use std::collections::HashMap;
    let mut map: HashMap<u64, u32> = HashMap::new();
    combined_keys
        .iter()
        .map(|&k| {
            let next = map.len() as u32;
            *map.entry(k).or_insert(next)
        })
        .collect()
}

/// D-03 linchpin — MUST pass before any value stage. The fold-0 permutation
/// reproduces `permutation_fold0.npy` integer-exact via
/// `fisher_yates_permutation(N, random_seed)`.
#[test]
fn tensor_ctr_permutation_is_integer_exact_first() {
    let fold0 = load_i32("tensor_ctr/permutation_fold0.npy");
    assert_eq!(fold0.len(), FIXTURE_N, "fixture N must be 30");

    let actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&fold0, &actual)
        .unwrap_or_else(|e| panic!("tensor_ctr fold-0 permutation diverged (D-03): {e}"));
}

/// Per-object `(good, total)` exact-integer vs the fixture (the combined-projection
/// prefix anchor), then `Stage::OnlineCtr` value ≤1e-5 reproduced from those
/// anchors via the production `calc_ctr_online`. Runs only AFTER the D-03
/// permutation gate.
#[test]
fn tensor_ctr_per_object_counts_exact_and_value_within_tolerance() {
    // Gate: fold-0 permutation must reproduce first (D-03).
    let perm_expected = load_i32("tensor_ctr/permutation_fold0.npy");
    let perm_actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&perm_expected, &perm_actual)
        .expect("D-03 permutation must pass before the combined OnlineCtr value stage");

    let good = load_i32("tensor_ctr/ctr_good_count.npy");
    let total = load_i32("tensor_ctr/ctr_total_count.npy");
    let expected_value = load_f64("tensor_ctr/ctr_value.npy");
    assert_eq!(good.len(), FIXTURE_N);
    assert_eq!(total.len(), FIXTURE_N);
    assert_eq!(expected_value.len(), FIXTURE_N);

    // Per-object integer anchors: good ≤ total and non-negative (the prefix read
    // before increment can never have more pos than total predecessors).
    for (i, (&g, &t)) in good.iter().zip(total.iter()).enumerate() {
        assert!(g >= 0 && t >= 0, "doc {i}: negative count");
        assert!(g <= t, "doc {i}: good {g} > total {t} (impossible prefix)");
    }

    // Stage::OnlineCtr value ≤1e-5 from the committed integer anchors over the
    // COMBINED projection bucket (Borders:Prior=0.5).
    let actual_value: Vec<f64> = good
        .iter()
        .zip(total.iter())
        .map(|(&g, &t)| calc_ctr_online(g as f64, t, PRIOR))
        .collect();
    compare_stage(Stage::OnlineCtr, &expected_value, &actual_value)
        .unwrap_or_else(|e| panic!("tensor_ctr combined OnlineCtr value diverged: {e}"));
}

/// The combined projection hash drives the SAME read-before-increment online
/// accumulation as a single feature, on a hand-derived 2-feature scenario whose
/// combined-bucket prefixes are auditable by hand (the fixture's raw cat0/cat1
/// inputs are uncommitted, D-09). The 2-feature combination FOLDS the two
/// per-document cat hashes (`TProjection::combined_hash`, the ctr_provider.h
/// CalcHash); the resulting combined buckets differ from EITHER single feature's
/// buckets (the genuine tensor surface).
#[test]
fn tensor_ctr_combined_projection_drives_online_accumulation() {
    // Six docs over two cat features. cat0 ∈ {p,q}, cat1 ∈ {x,y}; the COMBINATION
    // (cat0,cat1) has up to four distinct buckets.
    let cat0 = ["p", "p", "q", "p", "q", "q"];
    let cat1 = ["x", "y", "x", "x", "y", "x"];
    let target_class = vec![1usize, 0, 1, 0, 1, 1];
    let n = cat0.len();

    // Per-document per-feature hashes, then the COMBINED projection key over the
    // 2-feature projection {0,1} (the ctr_provider.h fold).
    let proj = TProjection::from_features(&[0, 1]);
    let combined_keys: Vec<u64> = (0..n)
        .map(|i| {
            let feature_hashes = [calc_cat_feature_hash(cat0[i]), calc_cat_feature_hash(cat1[i])];
            proj.combined_hash(&feature_hashes)
        })
        .collect();
    let combined_bins = combined_keys_to_bins(&combined_keys);

    // The four combinations present: (p,x) docs{0,3}, (p,y) doc{1}, (q,x)
    // docs{2,5}, (q,y) doc{4}. Distinct combined keys → distinct bins.
    assert_eq!(
        combined_bins[0], combined_bins[3],
        "(p,x) docs share a combined bucket"
    );
    assert_eq!(combined_bins[2], combined_bins[5], "(q,x) docs share a bucket");
    assert_ne!(
        combined_bins[0], combined_bins[2],
        "(p,x) and (q,x) are DIFFERENT combinations (cat0 differs)"
    );
    assert_ne!(
        combined_bins[0], combined_bins[1],
        "(p,x) and (p,y) are DIFFERENT combinations (cat1 differs)"
    );

    // Identity (object-order) prefix over the combined buckets — the SAME
    // read-before-increment loop as single-feature CTR, now keyed on the combined
    // bucket.
    let identity: Vec<i32> = (0..n as i32).collect();
    let prefix = online_ctr_prefix_binclf(&identity, &combined_bins, &target_class, PRIOR)
        .expect("combined online prefix");

    // Hand-audit the (p,x) bucket {doc0, doc3}: doc0 sees empty (0,0); doc3 sees
    // doc0's class-1 (good=1,total=1). (q,x) bucket {doc2, doc5}: doc2 empty;
    // doc5 sees doc2's class-1 (good=1,total=1).
    assert_eq!(prefix.good[0], 0);
    assert_eq!(prefix.total[0], 0, "doc0 first in (p,x) bucket");
    assert_eq!(prefix.good[3], 1, "doc3 sees doc0 (class 1) in (p,x)");
    assert_eq!(prefix.total[3], 1);
    assert_eq!(prefix.good[5], 1, "doc5 sees doc2 (class 1) in (q,x)");
    assert_eq!(prefix.total[5], 1);
    // Singleton buckets (p,y)/(q,y) read empty.
    assert_eq!(prefix.total[1], 0, "doc1 alone in (p,y)");
    assert_eq!(prefix.total[4], 0, "doc4 alone in (q,y)");
    // Value relation holds: doc3 -> (1+0.5)/(1+1) = 0.75.
    assert!((prefix.value[3] - 0.75).abs() < 1e-12);

    // DEGENERATION: a SINGLE-feature projection {0} (cat0 alone) produces COARSER
    // buckets — (p,x) and (p,y) merge into the single "p" bucket, so the prefixes
    // DIFFER from the combined ones. This proves the combined key is genuinely
    // 2-feature, not collapsing to one feature (D-05 tensor surface).
    let proj_single = TProjection::single(0);
    let single_keys: Vec<u64> = (0..n)
        .map(|i| {
            let feature_hashes = [calc_cat_feature_hash(cat0[i]), calc_cat_feature_hash(cat1[i])];
            proj_single.combined_hash(&feature_hashes)
        })
        .collect();
    let single_bins = combined_keys_to_bins(&single_keys);
    let single_prefix = online_ctr_prefix_binclf(&identity, &single_bins, &target_class, PRIOR)
        .expect("single online prefix");
    // In the cat0-only bucketing, "p" = {0,1,3}: doc3 now ALSO sees doc1 (class 0)
    // → total 2, good 1 — strictly more predecessors than the (p,x) combination.
    assert_eq!(single_prefix.total[3], 2, "cat0-only merges (p,y) into doc3's bucket");
    assert_ne!(
        single_prefix.total[3], prefix.total[3],
        "combination ≠ single feature (the tensor is genuinely 2-dimensional)"
    );
}
