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
    let table = build_final_ctr(&acc(), ECtrType::Borders);
    assert_eq!(table.target_classes_count, 2);
    // bucket-major: [a.N0, a.N1, b.N0, b.N1, c.N0, c.N1] = [1,2, 1,1, 0,1].
    assert_eq!(table.int_counts, vec![1, 2, 1, 1, 0, 1]);
    assert_eq!(table.counter_denominator, 0);
}

#[test]
fn buckets_table_shares_class_count_layout() {
    let table = build_final_ctr(&acc(), ECtrType::Buckets);
    assert_eq!(table.int_counts, vec![1, 2, 1, 1, 0, 1]);
}

#[test]
fn counter_denominator_is_max_bucket_total() {
    // Counter: counts = bucket totals [3,2,1]; CounterDenominator = max = 3.
    let table = build_final_ctr(&acc(), ECtrType::Counter);
    assert_eq!(table.int_counts, vec![3, 2, 1]);
    assert_eq!(table.counter_denominator, 3, "Counter denom = MAX bucket total");
}

#[test]
fn feature_freq_denominator_is_total_sample_count() {
    // FeatureFreq: SAME counts [3,2,1] but CounterDenominator = total = 6.
    let table = build_final_ctr(&acc(), ECtrType::FeatureFreq);
    assert_eq!(table.int_counts, vec![3, 2, 1]);
    assert_eq!(
        table.counter_denominator, 6,
        "FeatureFreq denom = total sample count"
    );
}

#[test]
fn counter_and_feature_freq_differ_only_in_denominator() {
    // Pitfall 4: same numerator counts, DIFFERENT denominators (3 vs 6).
    let counter = build_final_ctr(&acc(), ECtrType::Counter);
    let freq = build_final_ctr(&acc(), ECtrType::FeatureFreq);
    assert_eq!(counter.int_counts, freq.int_counts);
    assert_ne!(counter.counter_denominator, freq.counter_denominator);
}

#[test]
fn binarized_target_mean_uses_class_over_border_count() {
    let table = build_final_ctr(&acc(), ECtrType::BinarizedTargetMeanValue);
    // target_border_count=2: bucket "a" classes 1,1,0 -> (0.5+0.5+0)=1.0/count3.
    assert!((table.mean_sum[0] - 1.0).abs() < 1e-6);
    assert_eq!(table.mean_count[0], 3);
    assert!(table.int_counts.is_empty(), "mean type carries no int counts");
}

#[test]
fn float_target_mean_uses_raw_target() {
    let table = build_final_ctr(&acc(), ECtrType::FloatTargetMeanValue);
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
