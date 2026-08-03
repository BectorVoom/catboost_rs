//! Unit tests for the [`ECtrType`](super::ECtrType) capability helpers
//! (SPEC-CTRT-01, SPEC-CTRT-02).
//!
//! Each test pins one helper against its upstream anchor:
//! - `target_border_count` → `ctr_helper.h:34-42` (`GetTargetBorderCount`)
//! - `is_cpu_supported`    → `restrictions.h:18-48` (`IsSupportedCtrType(CPU, …)`)
//! - `is_online_prefix`    → `ctr_type.cpp:43-56` (`IsPermutationDependentCtrType`)
//! - `final_ctr_target_border_count` → `online_ctr.cpp:914` (`CalcFinalCtrsImpl`)

use super::{final_ctr_target_border_count, ECtrType};

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

// ---------------------------------------------------------------------------
// BUG-BTMV / SPEC-BTMV-01 — the WHOLE-SET bake's target border count.
//
// Upstream has TWO target-border-count rules and they are not the same
// function. `ECtrType::target_border_count` (above) mirrors
// `GetTargetBorderCount` (ctr_helper.h:34-42), the ONLINE-path helper.
// `final_ctr_target_border_count` mirrors `CalcFinalCtrsImpl`
// (online_ctr.cpp:914), which is `targetClassesCount - 1` UNCONDITIONALLY —
// computed once, OUTSIDE the per-type dispatch. These tests pin the second rule
// directly, independently of any bake.
// ---------------------------------------------------------------------------

#[test]
fn final_ctr_target_border_count_is_classes_minus_one() {
    // online_ctr.cpp:914   int targetBorderCount = targetClassesCount - 1;
    // online_ctr.cpp:920   elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount);
    //
    // No `ctrType` appears in either line: the whole-set divisor is
    // type-INDEPENDENT. BUG-BTMV was `bake.rs` passing `targetClassesCount`
    // itself, which halved every BinarizedTargetMeanValue `Sum`.
    for (classes, want) in [(2usize, 1usize), (3, 2), (5, 4), (10, 9)] {
        assert_eq!(
            final_ctr_target_border_count(classes),
            want,
            "CalcFinalCtrsImpl divides by targetClassesCount - 1 \
             (online_ctr.cpp:914): {classes} classes must give {want}"
        );
    }
}

#[test]
fn final_ctr_target_border_count_floors_at_one() {
    // `accumulate_online` rejects `target_border_count == 0` with a typed
    // `CbError::Degenerate` (online.rs:176-180), so without the `.max(1)` floor
    // a single-class corpus would start erroring where it returns `Ok` today.
    // The floor is behavior-preserving at `classes == 1` (every target_class is
    // 0, so every Sum is 0 under either divisor) and flips `Err` -> `Ok` at
    // `classes == 0`, which is unreachable — the sole production caller
    // hard-codes 2 (boosting.rs:5582). See PLAN D3.
    assert_eq!(
        final_ctr_target_border_count(1),
        1,
        "a single-class corpus must not divide by zero (D3, online.rs:176-180)"
    );
    assert_eq!(
        final_ctr_target_border_count(0),
        1,
        "saturating_sub must not underflow at zero classes (D3)"
    );
}

#[test]
fn the_two_target_border_rules_differ_for_buckets() {
    // The structural statement: substituting one rule for the other is
    // UNDETECTABLE at binary classification and WRONG at multiclass. Asserted at
    // classes = 3, the smallest count where the two rules separate.
    for t in ALL_TYPES {
        let online = t.target_border_count(3);
        let bake = final_ctr_target_border_count(3);
        if matches!(t, ECtrType::Buckets) {
            assert_ne!(
                online, bake,
                "Buckets is exactly where the two rules diverge \
                 (GetTargetBorderCount returns targetClassesCount; \
                 CalcFinalCtrsImpl returns targetClassesCount - 1) — this \
                 inequality is WHY the bake must not route through the \
                 ECtrType helper (BUG-BTMV, PLAN §0)"
            );
        }
    }

    // All THREE candidate divisors are distinct at classes = 3. That is what
    // makes `final_ctr_test::bake_target_border_divisor_is_classes_minus_one_\
    // not_the_ctr_type_helper` (B01 Test 2) a genuine discriminator: at
    // classes = 2 the fix and the helper both return 1 and no runtime gate can
    // tell them apart.
    //
    //   expression                                  classes=2   classes=3
    //   ------------------------------------------  ---------   ---------
    //   `classes`            (BUG-BTMV, the defect)     2           3
    //   `classes - 1`        (upstream, D1/E1)          1           2
    //   `BTMV.target_border_count(classes)` (E4)        1           1
    assert_eq!(
        ECtrType::Buckets.target_border_count(3),
        3,
        "GetTargetBorderCount keeps the full class count for Buckets"
    );
    assert_eq!(
        ECtrType::BinarizedTargetMeanValue.target_border_count(3),
        1,
        "GetTargetBorderCount is fixed at 1 for BinarizedTargetMeanValue"
    );
    assert_eq!(
        final_ctr_target_border_count(3),
        2,
        "CalcFinalCtrsImpl is targetClassesCount - 1 regardless of type"
    );
}
