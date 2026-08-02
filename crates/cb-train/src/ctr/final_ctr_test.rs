//! Unit tests for the whole-set final-CTR table build
//! ([`crate::ctr::final_ctr`]) — all six types, Counter vs FeatureFreq
//! denominator distinction (Pitfall 4), FloatTargetMeanValue final-path-only
//! (Pitfall 5).

use crate::ctr::final_ctr::build_final_ctr;
use crate::ctr::online::accumulate_online;
use crate::ctr::ECtrType;

/// `a` 3x (classes 1,1,0), `b` 2x (0,1), `c` 1x (1) — bucket totals [3,2,1].
fn acc() -> crate::ctr::online::OnlineCtrAccumulator {
    let column = vec!["a", "a", "b", "a", "b", "c"];
    let target_class = vec![1, 1, 0, 0, 1, 1];
    let target = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    accumulate_online(&column, &target_class, &target, 2, 2).expect("accumulate")
}

#[test]
fn borders_table_flattens_per_class_counts() {
    let table = build_final_ctr(&acc(), ECtrType::Borders, true);
    assert_eq!(table.target_classes_count, 2);
    // bucket-major: [a.N0, a.N1, b.N0, b.N1, c.N0, c.N1] = [1,2, 1,1, 0,1].
    assert_eq!(table.int_counts, vec![1, 2, 1, 1, 0, 1]);
    assert_eq!(table.counter_denominator, 0);
}

#[test]
fn buckets_table_shares_class_count_layout() {
    let table = build_final_ctr(&acc(), ECtrType::Buckets, true);
    assert_eq!(table.int_counts, vec![1, 2, 1, 1, 0, 1]);
}

#[test]
fn counter_denominator_is_max_bucket_total() {
    // Counter: counts = bucket totals [3,2,1]; CounterDenominator = max = 3.
    let table = build_final_ctr(&acc(), ECtrType::Counter, true);
    assert_eq!(table.int_counts, vec![3, 2, 1]);
    assert_eq!(table.counter_denominator, 3, "Counter denom = MAX bucket total");
}

#[test]
fn feature_freq_denominator_is_total_sample_count() {
    // FeatureFreq: SAME counts [3,2,1] but CounterDenominator = total = 6.
    let table = build_final_ctr(&acc(), ECtrType::FeatureFreq, true);
    assert_eq!(table.int_counts, vec![3, 2, 1]);
    assert_eq!(
        table.counter_denominator, 6,
        "FeatureFreq denom = total sample count"
    );
}

#[test]
fn counter_and_feature_freq_differ_only_in_denominator() {
    // Pitfall 4: same numerator counts, DIFFERENT denominators (3 vs 6).
    let counter = build_final_ctr(&acc(), ECtrType::Counter, true);
    let freq = build_final_ctr(&acc(), ECtrType::FeatureFreq, true);
    assert_eq!(counter.int_counts, freq.int_counts);
    assert_ne!(counter.counter_denominator, freq.counter_denominator);
}

#[test]
fn binarized_target_mean_uses_class_over_border_count() {
    let table = build_final_ctr(&acc(), ECtrType::BinarizedTargetMeanValue, true);
    // target_border_count=2: bucket "a" classes 1,1,0 -> (0.5+0.5+0)=1.0/count3.
    assert!((table.mean_sum[0] - 1.0).abs() < 1e-6);
    assert_eq!(table.mean_count[0], 3);
    assert!(table.int_counts.is_empty(), "mean type carries no int counts");
}

#[test]
fn float_target_mean_uses_raw_target() {
    let table = build_final_ctr(&acc(), ECtrType::FloatTargetMeanValue, true);
    // raw targets for "a": 1.0 + 1.0 + 0.0 = 2.0 over count 3.
    assert!((table.mean_sum[0] - 2.0).abs() < 1e-6);
    assert_eq!(table.mean_count[0], 3);
}

#[test]
fn ctr_type_default_priors_match_upstream_counts() {
    // Borders/Buckets/BinarizedTargetMean: THREE priors {0,0.5,1}.
    assert_eq!(ECtrType::Borders.default_priors().len(), 3);
    assert_eq!(ECtrType::Buckets.default_priors().len(), 3);
    assert_eq!(ECtrType::BinarizedTargetMeanValue.default_priors().len(), 3);
    // Counter/FeatureFreq/FloatTargetMean: a SINGLE prior {0}.
    assert_eq!(ECtrType::Counter.default_priors().len(), 1);
    assert_eq!(ECtrType::FeatureFreq.default_priors().len(), 1);
    assert_eq!(ECtrType::FloatTargetMeanValue.default_priors().len(), 1);
}

#[test]
fn ctr_type_i8_discriminants_match_upstream() {
    // Mirror the upstream ECtrType discriminants bit-for-bit.
    assert_eq!(ECtrType::Borders.as_i8(), 0);
    assert_eq!(ECtrType::Buckets.as_i8(), 1);
    assert_eq!(ECtrType::BinarizedTargetMeanValue.as_i8(), 2);
    assert_eq!(ECtrType::FloatTargetMeanValue.as_i8(), 3);
    assert_eq!(ECtrType::Counter.as_i8(), 4);
    assert_eq!(ECtrType::FeatureFreq.as_i8(), 5);
    // Round-trip from_i8.
    for v in 0..=5i8 {
        assert_eq!(ECtrType::from_i8(v).map(ECtrType::as_i8), Some(v));
    }
    assert_eq!(ECtrType::from_i8(7), None);
}

// ---------------------------------------------------------------------------
// E11 / SPEC-CTRT-13 — per-type final tables in the bake path.
//
// `bake_ctr_table` hard-coded build_final_ctr(&acc, ECtrType::Borders, true),
// ctr_type: Borders.as_i8() and counter_denominator: 0, so every baked inference
// table was a Borders table no matter what the trainer chose.
// ---------------------------------------------------------------------------

/// A 12-document, 3-bucket categorical column with a mixed binclf target.
fn e11_fixture() -> (Vec<Vec<String>>, Vec<usize>, crate::projection::TProjection) {
    let col: Vec<String> = (0..12)
        .map(|i| cb_data::stringify_int_category(i % 3))
        .collect();
    let target_class: Vec<usize> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1];
    let proj = crate::projection::TProjection::from_features(&[0]);
    (vec![col], target_class, proj)
}

#[test]
fn bake_emits_the_requested_type_and_denominator() {
    use crate::ctr::bake::bake_ctr_table;

    let (cats, tc, proj) = e11_fixture();
    let bake = |ty: ECtrType| {
        bake_ctr_table(&cats, &proj, &tc, 2, 15, 0.5, 1.0, ty).expect("bake must succeed")
    };

    // --- Borders: per-bucket [N0, N1], no counter denominator, no mean. ------
    let borders = bake(ECtrType::Borders);
    assert_eq!(borders.ctr_type, 0);
    assert!(borders.int_counts.iter().all(|c| c.len() == 2));
    assert_eq!(borders.counter_denominator, 0);
    assert!(borders.mean.is_empty());

    // --- Buckets: SAME shape as Borders. The two differ only at apply time,
    //     via target_border_idx — not in the baked table. -------------------
    let buckets = bake(ECtrType::Buckets);
    assert_eq!(buckets.ctr_type, 1);
    assert!(buckets.int_counts.iter().all(|c| c.len() == 2));
    assert_eq!(buckets.counter_denominator, 0);
    assert!(buckets.mean.is_empty());
    assert_eq!(
        buckets.int_counts, borders.int_counts,
        "Buckets and Borders bake the SAME counts; only apply-time selection differs"
    );

    // --- Counter: ONE value per bucket and a real denominator. --------------
    // The column is i % 3 over 12 documents, so every bucket holds 4 documents
    // and the MAX bucket total is 4.
    let counter = bake(ECtrType::Counter);
    assert_eq!(counter.ctr_type, 4);
    assert!(
        counter.int_counts.iter().all(|c| c.len() == 1),
        "Counter's wire TargetClassesCount is 0 and the decoder forces width 1, \
         so the bake must emit one value per bucket"
    );
    assert_eq!(counter.int_counts, vec![vec![4], vec![4], vec![4]]);
    assert_eq!(
        counter.counter_denominator, 4,
        "CounterDenominator is the MAX bucket total (online_ctr.cpp:934-936)"
    );
    assert!(counter.mean.is_empty());

    // --- BinarizedTargetMeanValue: mean pairs, no int counts. ---------------
    let btmv = bake(ECtrType::BinarizedTargetMeanValue);
    assert_eq!(btmv.ctr_type, 2);
    assert!(
        btmv.int_counts.is_empty(),
        "the mean types carry (Sum, Count) pairs, not class counts"
    );
    assert_eq!(btmv.mean.len(), 3, "one (Sum, Count) pair per bucket");
    // Anti-vacuity: at least one bucket must carry a non-zero sum, else an
    // all-zero mean table would satisfy the shape assertions trivially.
    assert!(
        btmv.mean.iter().any(|&(sum, _)| sum != 0.0),
        "a bucket with positive-class documents must carry a non-zero Sum"
    );
    // The Sum half is f32-typed (upstream TCtrMeanHistory::Sum, online_ctr.h:373).
    let _: f32 = btmv.mean[0].0;
}

#[test]
fn borders_bake_bytes_are_unchanged() {
    use crate::ctr::bake::bake_ctr_table;

    // FROZEN literals transcribed from a PRE-CHANGE run: the default Borders path
    // must be byte-identical after the per-type reshape. This is what keeps the
    // 11 CTR oracles green.
    let (cats, tc, proj) = e11_fixture();
    let t = bake_ctr_table(&cats, &proj, &tc, 2, 15, 0.5, 1.0, ECtrType::Borders)
        .expect("bake must succeed");

    assert_eq!(
        t.hashes,
        vec![
            14_096_670_708_071_601_218,
            10_650_234_391_120_027_977,
            6_692_239_851_685_836_511
        ]
    );
    assert_eq!(t.int_counts, vec![vec![0, 4], vec![4, 0], vec![1, 3]]);
    // (Shift, Scale) is derived type-agnostically from the prior and must not move.
    assert_eq!(t.shift.to_bits(), 9_223_372_036_854_775_808);
    assert_eq!(t.scale.to_bits(), 4_624_633_867_356_078_080);
    assert_eq!(t.counter_denominator, 0);
}
