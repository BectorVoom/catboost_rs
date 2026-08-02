//! Unit tests for the whole-set (Plain-mode) CTR accumulation
//! ([`crate::ctr::online`]).

use crate::ctr::online::{
    accumulate_online, ordered_ctr_per_permutation, TCtrHistory, TCtrMeanHistory,
};

/// A small binclf categorical column: three distinct values, mixed classes.
/// `a` appears 3x (classes 1,1,0), `b` 2x (classes 0,1), `c` 1x (class 1).
fn small_column() -> (Vec<&'static str>, Vec<usize>, Vec<f64>) {
    let column = vec!["a", "a", "b", "a", "b", "c"];
    let target_class = vec![1, 1, 0, 0, 1, 1];
    let target = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    (column, target_class, target)
}

#[test]
fn whole_set_class_counts_are_exact_integers() {
    let (col, tc, t) = small_column();
    let acc = accumulate_online(&col, &tc, &t, 2, 1).expect("accumulate");
    assert_eq!(acc.bucket_count, 3, "three distinct categories -> 3 buckets");
    // Bucket 0 = "a" (first-seen): classes [1,1,0] -> N[0]=1, N[1]=2.
    assert_eq!(acc.class_histories[0].n, vec![1, 2]);
    // Bucket 1 = "b": classes [0,1] -> N[0]=1, N[1]=1.
    assert_eq!(acc.class_histories[1].n, vec![1, 1]);
    // Bucket 2 = "c": class [1] -> N[0]=0, N[1]=1.
    assert_eq!(acc.class_histories[2].n, vec![0, 1]);
}

#[test]
fn totals_match_per_bucket_document_counts() {
    let (col, tc, t) = small_column();
    let acc = accumulate_online(&col, &tc, &t, 2, 1).expect("accumulate");
    // total_counts is the Counter/FeatureFreq numerator: per-bucket doc count.
    assert_eq!(acc.total_counts, vec![3, 2, 1]);
    // TCtrHistory::total agrees with the explicit total.
    assert_eq!(acc.class_histories[0].total(), 3);
    assert_eq!(acc.class_histories[1].total(), 2);
    assert_eq!(acc.class_histories[2].total(), 1);
}

#[test]
fn binarized_mean_divides_by_target_border_count() {
    let (col, tc, t) = small_column();
    // target_border_count = 2: BinarizedTargetMeanValue adds class/2.
    let acc = accumulate_online(&col, &tc, &t, 2, 2).expect("accumulate");
    // Bucket "a": classes 1,1,0 -> (0.5 + 0.5 + 0.0) = 1.0 over count 3.
    assert!((acc.binarized_mean[0].sum - 1.0).abs() < 1e-6);
    assert_eq!(acc.binarized_mean[0].count, 3);
    // Bucket "b": classes 0,1 -> (0.0 + 0.5) = 0.5 over count 2.
    assert!((acc.binarized_mean[1].sum - 0.5).abs() < 1e-6);
    assert_eq!(acc.binarized_mean[1].count, 2);
}

#[test]
fn float_mean_adds_raw_target() {
    let column = vec!["x", "x", "y"];
    let target_class = vec![0, 1, 1];
    let target = vec![2.5, 3.5, 10.0];
    let acc = accumulate_online(&column, &target_class, &target, 2, 1).expect("accumulate");
    // Bucket "x": raw targets 2.5 + 3.5 = 6.0 over count 2.
    assert!((acc.float_mean[0].sum - 6.0).abs() < 1e-6);
    assert_eq!(acc.float_mean[0].count, 2);
    // Bucket "y": raw target 10.0 over count 1.
    assert!((acc.float_mean[1].sum - 10.0).abs() < 1e-6);
    assert_eq!(acc.float_mean[1].count, 1);
}

#[test]
fn length_mismatch_is_typed_error_not_panic() {
    let column = vec!["a", "b"];
    let target_class = vec![0]; // wrong length
    let target = vec![0.0, 1.0];
    assert!(accumulate_online(&column, &target_class, &target, 2, 1).is_err());
}

#[test]
fn zero_target_border_count_is_typed_error() {
    let (col, tc, t) = small_column();
    assert!(accumulate_online(&col, &tc, &t, 2, 0).is_err());
}

#[test]
fn empty_column_yields_zero_buckets() {
    let acc = accumulate_online(&[], &[], &[], 2, 1).expect("empty accumulate");
    assert_eq!(acc.bucket_count, 0);
    assert!(acc.class_histories.is_empty());
    assert!(acc.total_counts.is_empty());
}

#[test]
fn ctr_history_increment_is_bounds_checked() {
    let mut h = TCtrHistory::new(2);
    h.increment(1);
    h.increment(1);
    h.increment(0);
    assert_eq!(h.n, vec![1, 2]);
    // An out-of-range class is ignored (no panic), leaving counts unchanged.
    h.increment(5);
    assert_eq!(h.n, vec![1, 2]);
}

#[test]
fn mean_history_add_accumulates_sum_and_count() {
    let mut m = TCtrMeanHistory::default();
    m.add(1.0);
    m.add(0.5);
    assert!((m.sum - 1.5).abs() < 1e-6);
    assert_eq!(m.count, 2);
}

/// A 3-doc hand-auditable ordered (per-permutation) prefix. Two buckets:
/// bucket 0 = docs {0, 2}, bucket 1 = doc {1}; classes `[1, 0, 1]`. Under the
/// permutation `[2, 0, 1]` (doc 2 first, then doc 0, then doc 1):
/// - step 0: doc 2 (bucket 0) reads EMPTY (good=0,total=0) → value (0+0.5)/1=0.5,
///   then +1 pos in bucket 0.
/// - step 1: doc 0 (bucket 0) reads (good=1,total=1) [doc 2 was pos] →
///   (1+0.5)/2=0.75, then +1 pos.
/// - step 2: doc 1 (bucket 1) reads EMPTY (good=0,total=0) → 0.5, then +1 neg.
/// The OBJECT-order vectors must therefore be good=[1,0,0], total=[1,0,0],
/// value=[0.75,0.5,0.5]; the running per-step (num,denom) read in LEARN order is
/// (0,0),(1,1),(0,0) — per-bucket monotone (bucket 0: (0,0)→(1,1); bucket 1: (0,0)).
#[test]
fn ordered_ctr_three_doc_hand_auditable_prefix() {
    let permutation: Vec<i32> = vec![2, 0, 1];
    let bins = vec![0u32, 1, 0];
    let target_class = vec![1usize, 0, 1];

    let out = ordered_ctr_per_permutation(&permutation, &bins, &target_class, 0.5)
        .expect("ordered ctr");

    // OBJECT order (indexed by doc).
    assert_eq!(out.prefix.good, vec![1, 0, 0], "doc0 reads doc2's pos; doc1/doc2 empty");
    assert_eq!(out.prefix.total, vec![1, 0, 0]);
    let expected_value = [0.75, 0.5, 0.5];
    for (i, (&v, &e)) in out.prefix.value.iter().zip(expected_value.iter()).enumerate() {
        assert!((v - e).abs() < 1e-6, "doc {i} value {v} != {e}");
    }

    // PERMUTATION-order running (num, denom) read at each learn step.
    assert_eq!(out.step_num, vec![0, 1, 0], "step reads: doc2 empty, doc0 sees 1, doc1 empty");
    assert_eq!(out.step_denom, vec![0, 1, 0]);

    // Per-bucket monotone internal-consistency anchor.
    assert!(out.per_bucket_monotone(&permutation, &bins), "per-bucket running counts monotone");
}

/// Identity-permutation degeneration: ordered prefix under the identity
/// permutation equals the object-order read-before-increment prefix (the prefix
/// each doc sees is exactly its object-order predecessors). For a single bucket
/// with classes `[1, 0, 1, 1]` the running good/total are the pure object-order
/// prefix sums — the degeneration anchor (identity ordered == plain prefix).
#[test]
fn ordered_ctr_identity_permutation_degenerates_to_object_order_prefix() {
    let permutation: Vec<i32> = vec![0, 1, 2, 3];
    let bins = vec![0u32, 0, 0, 0];
    let target_class = vec![1usize, 0, 1, 1];

    let out = ordered_ctr_per_permutation(&permutation, &bins, &target_class, 0.5)
        .expect("ordered ctr");
    // Object-order prefix: doc0 empty, doc1 sees (1,1), doc2 sees (1,2), doc3 (2,3).
    assert_eq!(out.prefix.good, vec![0, 1, 1, 2]);
    assert_eq!(out.prefix.total, vec![0, 1, 2, 3]);
    assert!(out.per_bucket_monotone(&permutation, &bins));
}

#[test]
fn ordered_ctr_length_mismatch_is_typed_error() {
    let permutation: Vec<i32> = vec![0, 1];
    let bins = vec![0u32]; // wrong length
    let target_class = vec![1usize, 0];
    assert!(ordered_ctr_per_permutation(&permutation, &bins, &target_class, 0.5).is_err());
}

// ---------------------------------------------------------------------------
// E04 / SPEC-CTRT-04 — `online_class_prefix`, the ONE generic classes-prefix
// producer.
//
// Upstream `UpdateGoodCount` (online_ctr.cpp:115-121):
//   if (Buckets) *goodCount = curCount; else *goodCount -= curCount;
// applied cumulatively over border = 0..targetBorderCount starting from
// goodCount = Total. For a single border `b` that collapses to:
//   Buckets -> N[b]
//   Borders -> Total - sum(N[0..=b])
// ---------------------------------------------------------------------------

#[test]
fn online_class_prefix_matches_hand_computed_upstream_vectors() {
    use crate::ctr::online::online_class_prefix;
    use crate::ctr::ECtrType;

    // (counts, b, ctr_type, expected_num, expected_denom) — every value
    // hand-computed from the upstream formula above, not from this code.
    let cases: Vec<(Vec<i64>, usize, ECtrType, f64, i64)> = vec![
        // Buckets selects the b-th class count directly.
        (vec![3, 7], 0, ECtrType::Buckets, 3.0, 10),
        (vec![3, 7], 1, ECtrType::Buckets, 7.0, 10),
        // Borders subtracts the cumulative head: 10 - 3 = 7.
        (vec![3, 7], 0, ECtrType::Borders, 7.0, 10),
        (vec![2, 5, 4], 0, ECtrType::Buckets, 2.0, 11),
        (vec![2, 5, 4], 1, ECtrType::Buckets, 5.0, 11),
        // 11 - 2 = 9.
        (vec![2, 5, 4], 0, ECtrType::Borders, 9.0, 11),
        // 11 - (2 + 5) = 4.
        (vec![2, 5, 4], 1, ECtrType::Borders, 4.0, 11),
        // An empty bucket must produce the empty value, never a division setup.
        (vec![0, 0], 0, ECtrType::Borders, 0.0, 0),
        // Degenerate: no classes at all. Must not panic.
        (vec![], 0, ECtrType::Buckets, 0.0, 0),
        // Out-of-range border index: checked `.get`, so 0 rather than a panic.
        (vec![3, 7], 9, ECtrType::Buckets, 0.0, 10),
    ];

    for (counts, b, ctr_type, want_num, want_denom) in cases {
        let (num, denom) = online_class_prefix(&counts, b, ctr_type);
        // Integers exactly representable in f64 — exact equality, no tolerance.
        assert_eq!(
            num, want_num,
            "numerator for counts={counts:?} b={b} type={ctr_type:?}"
        );
        assert_eq!(
            denom, want_denom,
            "denominator for counts={counts:?} b={b} type={ctr_type:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// E05 / SPEC-CTRT-05 (acceptance A7) — THE REGRESSION FIREWALL.
//
// These three tests are the single most important risk control in Part 1. They
// prove that today's Borders-binclf prefix is BIT-FOR-BIT the (classes = 2,
// b = 0) special case of the generic `online_class_prefix` producer, so the
// W2/W3 per-type dispatch refactor cannot silently move any of the 11 existing
// CTR oracles.
//
// Test fns 2 and 3 carry literals TRANSCRIBED FROM A PRE-CHANGE RUN (captured
// before `online_ctr_prefix_binclf` was re-routed through the generic producer),
// which is what makes them a frozen characterization rather than a
// self-comparison.
// ---------------------------------------------------------------------------

/// The fixed 12-document scenario shared by test fns 2 and 3.
fn e05_scenario() -> (Vec<i32>, Vec<u32>, Vec<usize>, f64) {
    let permutation = vec![3, 0, 7, 1, 9, 4, 11, 2, 6, 5, 10, 8];
    let bins = vec![0, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1, 2];
    let target_class = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1];
    (permutation, bins, target_class, 0.5)
}

#[test]
fn borders_binclf_is_bit_identical_to_the_generic_class_prefix_at_b0() {
    use crate::ctr::online::online_class_prefix;
    use crate::ctr::ECtrType;

    // Exhaustive over a small grid: 13 x 13 = 169 cases. Bit equality on the
    // raw f64, NOT a tolerance — the firewall's whole value is that it admits
    // no drift at all.
    for n0 in 0..=12i64 {
        for n1 in 0..=12i64 {
            let (num, denom) = online_class_prefix(&[n0, n1], 0, ECtrType::Borders);
            assert_eq!(
                num.to_bits(),
                (n1 as f64).to_bits(),
                "generic Borders numerator must be BIT-identical to N[1] at n0={n0} n1={n1}"
            );
            assert_eq!(
                denom,
                n0 + n1,
                "generic Borders denominator must be the bucket total at n0={n0} n1={n1}"
            );
        }
    }
}

#[test]
fn online_ctr_prefix_binclf_output_is_unchanged_by_the_generic_reroute() {
    use crate::ctr::online::online_ctr_prefix_binclf;

    let (perm, bins, tc, prior) = e05_scenario();
    let got = online_ctr_prefix_binclf(&perm, &bins, &tc, prior).expect("prefix must succeed");

    // FROZEN literals, transcribed from the run taken BEFORE the re-route.
    let want_good: Vec<i64> = vec![0, 0, 2, 0, 0, 3, 2, 1, 0, 1, 0, 1];
    let want_total: Vec<i64> = vec![0, 0, 2, 0, 1, 3, 3, 1, 3, 1, 2, 2];
    // `value` is compared BIT-FOR-BIT, not with a tolerance: the re-route must
    // not perturb the f64 result by even one ulp.
    let want_value_bits: Vec<u64> = vec![
        4602678819172646912,
        4602678819172646912,
        4605681218924227243,
        4602678819172646912,
        4598175219545276416,
        4606056518893174784,
        4603804719079489536,
        4604930618986332160,
        4593671619917905920,
        4604930618986332160,
        4595172819793696085,
        4602678819172646912,
    ];

    assert_eq!(got.good, want_good, "good vector moved under the re-route");
    assert_eq!(got.total, want_total, "total vector moved under the re-route");
    let got_bits: Vec<u64> = got.value.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        got_bits, want_value_bits,
        "CTR value is not BIT-identical to the pre-re-route output"
    );
}

#[test]
fn ordered_ctr_per_permutation_step_counts_match_the_prefix_reroute() {
    use crate::ctr::online::ordered_ctr_per_permutation;

    let (perm, bins, tc, prior) = e05_scenario();
    let got = ordered_ctr_per_permutation(&perm, &bins, &tc, prior).expect("ordered must succeed");

    // The no-out-of-order anchor must survive the re-route.
    assert!(
        got.per_bucket_monotone(&perm, &bins),
        "per-bucket prefixes must stay monotone along the permutation"
    );

    // FROZEN literals from the same pre-change run. `ordered_ctr_per_permutation`
    // RE-DERIVES the read-before-increment loop separately from
    // `online_ctr_prefix_binclf`; if only one of the two is re-routed they
    // silently diverge, which is exactly what this test catches.
    let want_step_num: Vec<i64> = vec![0, 0, 1, 0, 1, 0, 1, 2, 2, 3, 0, 0];
    let want_step_denom: Vec<i64> = vec![0, 0, 1, 0, 1, 1, 2, 2, 3, 3, 2, 3];

    assert_eq!(got.step_num, want_step_num, "step_num moved under the re-route");
    assert_eq!(
        got.step_denom, want_step_denom,
        "step_denom moved under the re-route"
    );
}
