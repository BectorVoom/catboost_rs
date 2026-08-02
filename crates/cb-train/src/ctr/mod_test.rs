//! Unit tests for the [`ECtrType`](super::ECtrType) capability helpers
//! (SPEC-CTRT-01, SPEC-CTRT-02).
//!
//! Each test pins one helper against its upstream anchor:
//! - `target_border_count` → `ctr_helper.h:34-42` (`GetTargetBorderCount`)
//! - `is_cpu_supported`    → `restrictions.h:18-48` (`IsSupportedCtrType(CPU, …)`)
//! - `is_online_prefix`    → `ctr_type.cpp:43-56` (`IsPermutationDependentCtrType`)

use super::ECtrType;

/// Every variant, in discriminant order. Asserting over this array (rather than a
/// hand-picked subset) is what stops a future seventh variant from silently
/// inheriting a default answer.
const ALL_TYPES: [ECtrType; 6] = [
    ECtrType::Borders,
    ECtrType::Buckets,
    ECtrType::BinarizedTargetMeanValue,
    ECtrType::FloatTargetMeanValue,
    ECtrType::Counter,
    ECtrType::FeatureFreq,
];

#[test]
fn target_border_count_is_two_for_buckets_and_one_for_the_rest() {
    // `GetTargetBorderCount(ctr, targetClassesCount)` (ctr_helper.h:34-42):
    //   Buckets                        -> targetClassesCount
    //   BinarizedTargetMeanValue/Counter -> 1
    //   otherwise                      -> targetClassesCount - 1
    // With targetClassesCount = 2 (binary classification) every non-Buckets
    // CPU-legal type collapses to 1.
    assert_eq!(
        ECtrType::Buckets.target_border_count(2),
        2,
        "Buckets keeps the full class count as its border count"
    );

    for t in [
        ECtrType::Borders,
        ECtrType::BinarizedTargetMeanValue,
        ECtrType::Counter,
    ] {
        assert_eq!(
            t.target_border_count(2),
            1,
            "{t:?} must collapse to a single target border at targetClassesCount = 2"
        );
    }
}

#[test]
fn is_cpu_supported_rejects_exactly_float_target_mean_and_feature_freq() {
    // restrictions.h:18-48 — IsSupportedCtrType(ETaskType::CPU, …) is true for
    // exactly {Borders, Buckets, BinarizedTargetMeanValue, Counter}.
    let expected = [
        (ECtrType::Borders, true),
        (ECtrType::Buckets, true),
        (ECtrType::BinarizedTargetMeanValue, true),
        (ECtrType::FloatTargetMeanValue, false),
        (ECtrType::Counter, true),
        (ECtrType::FeatureFreq, false),
    ];

    // Asserted over ALL_TYPES so a newly added variant cannot escape the check.
    assert_eq!(expected.len(), ALL_TYPES.len());
    for (t, want) in expected {
        assert_eq!(
            t.is_cpu_supported(),
            want,
            "{t:?}: CPU support must match restrictions.h:18-48"
        );
    }
}

#[test]
fn is_online_prefix_is_false_only_for_counter() {
    // ctr_type.cpp:43-56 — Counter is NOT permutation-dependent, so it is
    // accumulated as a whole-set tally rather than a read-before-increment
    // online prefix. Every other online-path type is a prefix type.
    for t in [
        ECtrType::Borders,
        ECtrType::Buckets,
        ECtrType::BinarizedTargetMeanValue,
    ] {
        assert!(
            t.is_online_prefix(),
            "{t:?} is permutation-dependent and must use the online prefix path"
        );
    }

    assert!(
        !ECtrType::Counter.is_online_prefix(),
        "Counter is permutation-INdependent (ctr_type.cpp:43-56)"
    );
}
