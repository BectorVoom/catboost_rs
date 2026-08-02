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

// ---------------------------------------------------------------------------
// BUG-BTMV / SPEC-BTMV-01 — the whole-set bake divides by
// `targetClassesCount - 1` (online_ctr.cpp:914), NOT by `targetClassesCount`
// and NOT by GetTargetBorderCount (ctr_helper.h:34-42, the ONLINE-path helper).
// ---------------------------------------------------------------------------

#[test]
fn btmv_bake_sums_class_one_documents_per_bucket() {
    use crate::ctr::bake::bake_ctr_table;

    // `e11_fixture` is reused deliberately: it is already cross-checked against
    // the frozen Borders bake, which is what lets these expectations be DERIVED
    // rather than asserted by fiat.
    //
    // Buckets are `i % 3` in first-seen order 0, 1, 2. Class-1 counts are 4, 0, 3.
    // With the upstream whole-set divisor `targetClassesCount - 1 == 1` for
    // binclf, each bucket's Sum is EXACTLY its class-1 count.
    let (cats, tc, proj) = e11_fixture();
    let t = bake_ctr_table(
        &cats, &proj, &tc, 2, 15, 0.5, 1.0,
        ECtrType::BinarizedTargetMeanValue,
    )
    .expect("bake must succeed");

    // THE NON-FIAT STEP: tie the mean table to the INDEPENDENTLY FROZEN Borders
    // payload rather than to a literal chosen by the author.
    let borders = bake_ctr_table(&cats, &proj, &tc, 2, 15, 0.5, 1.0, ECtrType::Borders)
        .expect("borders bake must succeed");

    for b in 0..t.mean.len() {
        assert_eq!(
            f64::from(t.mean[b].0),
            borders.int_counts[b][1] as f64,
            "bucket {b}: the BTMV Sum must equal the Borders N1 (class-1 count) — \
             the whole-set divisor is targetClassesCount - 1 == 1 for binclf \
             (online_ctr.cpp:914). Observed Sum {} vs N1 {}: the ratio names the \
             wrong divisor.",
            t.mean[b].0,
            borders.int_counts[b][1]
        );
        assert_eq!(
            t.mean[b].1,
            borders.int_counts[b][0] + borders.int_counts[b][1],
            "bucket {b}: Count must equal N0 + N1"
        );
    }

    // Literal pin as well — belt and braces, and the human-readable failure.
    assert_eq!(
        t.mean,
        vec![(4.0f32, 4i64), (0.0, 4), (3.0, 4)],
        "hand-computed from e11_fixture. Halved Sums => bake.rs:196 passed \
         `classes` (2) as target_border_count instead of `classes - 1` (1)."
    );

    // ANTI-VACUITY, both clauses.
    assert!(
        t.mean.iter().any(|&(s, c)| f64::from(s) != c as f64 && s != 0.0),
        "every bucket has Sum == Count or Sum == 0 — a degenerate corpus would \
         satisfy this test trivially; the corpus must contain a bucket with a \
         MIXED target (bucket 2: 3 of 4)"
    );

    assert!(
        t.int_counts.is_empty(),
        "the mean types carry (Sum, Count) pairs"
    );
}

#[test]
fn bake_target_border_divisor_is_classes_minus_one_not_the_ctr_type_helper() {
    use crate::ctr::bake::bake_ctr_table;

    // A 3-class corpus. Production is binclf-only (boosting.rs:5582 hard-codes 2),
    // so this is a CHARACTERIZATION of `bake_ctr_table`'s public contract at a
    // configuration production does not reach. It exists because at classes == 2
    // the FIX and the HELPER are indistinguishable (both yield 1); the BUG yields
    // 2 and is distinguishable, but only in the Sum MAGNITUDE, not in
    // which-candidate-is-which — so classes = 3 is required to tell the fix apart
    // from the helper:
    //
    //   expression                                  classes=2   classes=3
    //   ------------------------------------------  ---------   ---------
    //   `classes`            (TODAY'S BUG)              2           3
    //   `classes - 1`        (upstream)                 1           2
    //   `BTMV.target_border_count(classes)`             1           1
    //
    // STOP CONDITION: if this assertion ever needs to change, either upstream's
    // CalcFinalCtrsImpl changed (re-read online_ctr.cpp:914) or someone routed
    // the bake through GetTargetBorderCount. Do NOT adjust the expected value.

    // ANTI-VACUITY: this is a discriminator ONLY while the three candidates differ.
    assert_ne!(
        3usize - 1,
        ECtrType::BinarizedTargetMeanValue.target_border_count(3),
        "this test is only a discriminator while these differ; if the helper ever \
         returns `classes - 1` for BTMV, re-derive the discriminator before \
         touching the expectations"
    );
    assert_ne!(3usize, 3usize - 1);

    let col: Vec<String> = ["a", "a", "a", "b", "b", "b"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    let tc: Vec<usize> = vec![0, 1, 2, 2, 2, 0];
    let proj = crate::projection::TProjection::from_features(&[0]);

    let t = bake_ctr_table(
        &[col], &proj, &tc, 3, 15, 0.5, 1.0,
        ECtrType::BinarizedTargetMeanValue,
    )
    .expect("bake must succeed");

    // Under `classes - 1 == 2`: "a" = (0+1+2)/2 = 1.5 over 3; "b" = (2+2+0)/2 = 2.0 over 3.
    assert_eq!(
        t.mean,
        vec![(1.5f32, 3i64), (2.0, 3)],
        "the whole-set divisor must be `classes - 1` == 2. Observed {:?}. \
         `classes` (3, today's bug) would give [(1.0, 3), (1.333.., 3)]; \
         the ctr-type helper (1) would give [(3.0, 3), (4.0, 3)].",
        t.mean
    );
}

#[test]
fn btmv_bake_at_one_class_is_unchanged_and_does_not_error() {
    use crate::ctr::bake::bake_ctr_table;

    let col: Vec<String> = (0..6).map(|i| cb_data::stringify_int_category(i % 2)).collect();
    let tc: Vec<usize> = vec![0; 6];
    let proj = crate::projection::TProjection::from_features(&[0]);

    let t = bake_ctr_table(
        &[col], &proj, &tc, 1, 15, 0.5, 1.0,
        ECtrType::BinarizedTargetMeanValue,
    );

    assert!(
        t.is_ok(),
        "`accumulate_online` rejects `target_border_count == 0` \
         (online.rs:176-180), so the bake floors the divisor at 1 \
         (`saturating_sub(1).max(1)`, the same idiom as `online_mean_prefix`, \
         online.rs:321). Without the floor this bake would start returning \
         CbError::Degenerate where it returns Ok today. Upstream's \
         CalcFinalCtrsImpl divides by 0 here and is undefined; the floor is a \
         deliberate, behavior-preserving divergence."
    );
    let t = t.expect("checked above");
    assert!(
        t.mean.iter().all(|&(s, _)| s == 0.0),
        "every target_class is 0, so every Sum must be 0 under any divisor"
    );
}

// ---------------------------------------------------------------------------
// BUG-BTMV / SPEC-BTMV-04 — the non-mean baked payloads are byte-identical.
//
// These are GUARDS, not Reds: they pass both before and after the divisor fix,
// because the whole-set target border count never reached a non-mean payload in
// the first place. Their falsifiability comes from the mutation check M-B05,
// whose expectation is INVERTED — mutating the divisor must leave these GREEN,
// with the BTMV test as the control proving the mutation was live.
//
// `borders_bake_bytes_are_unchanged` (above) already froze the Borders bake.
// These extend the freeze to Buckets and Counter, which had no byte-level pin.
// ---------------------------------------------------------------------------

#[test]
fn buckets_and_counter_bake_bytes_are_unchanged() {
    use crate::ctr::bake::bake_ctr_table;

    // The isolation these assertions encode: the whole-set target border count
    // reaches exactly one accumulator field, `binarized_mean`
    // (online.rs:212-214), which exactly one arm of `build_final_ctr` reads
    // (BinarizedTargetMeanValue, final_ctr.rs). Borders/Buckets read
    // `class_histories`; Counter/FeatureFreq read `total_counts`. No non-mean
    // payload can move when the divisor changes. If either of these fails, that
    // isolation has been broken — STOP AND REPORT.
    let (cats, tc, proj) = e11_fixture();
    let bake = |ty: ECtrType| {
        bake_ctr_table(&cats, &proj, &tc, 2, 15, 0.5, 1.0, ty).expect("bake must succeed")
    };

    // The same frozen triple `borders_bake_bytes_are_unchanged` pins: the hash
    // set and (Shift, Scale) are derived type-agnostically from the prior
    // (bake.rs:263-270) and must not move with the CTR type either.
    let frozen_hashes = vec![
        14_096_670_708_071_601_218_u64,
        10_650_234_391_120_027_977,
        6_692_239_851_685_836_511,
    ];

    // --- Buckets ------------------------------------------------------------
    let buckets = bake(ECtrType::Buckets);
    assert_eq!(buckets.ctr_type, 1, "SPEC-BTMV-04: Buckets discriminant");
    assert_eq!(
        buckets.hashes, frozen_hashes,
        "SPEC-BTMV-04: the divisor cannot reach the bucket hash set"
    );
    assert_eq!(
        buckets.int_counts,
        vec![vec![0, 4], vec![4, 0], vec![1, 3]],
        "SPEC-BTMV-04: Buckets reads `class_histories`, never `binarized_mean` \
         — its per-bucket [N0, N1] payload is identical to the frozen Borders \
         one and must not move when the whole-set divisor changes"
    );
    assert_eq!(buckets.counter_denominator, 0);
    assert!(
        buckets.mean.is_empty(),
        "SPEC-BTMV-04: the Buckets arm leaves `mean` empty (bake.rs:226-238)"
    );
    assert_eq!(buckets.shift.to_bits(), 9_223_372_036_854_775_808);
    assert_eq!(buckets.scale.to_bits(), 4_624_633_867_356_078_080);

    // --- Counter ------------------------------------------------------------
    let counter = bake(ECtrType::Counter);
    assert_eq!(counter.ctr_type, 4, "SPEC-BTMV-04: Counter discriminant");
    assert_eq!(
        counter.hashes, frozen_hashes,
        "SPEC-BTMV-04: the divisor cannot reach the bucket hash set"
    );
    assert_eq!(
        counter.int_counts,
        vec![vec![4], vec![4], vec![4]],
        "SPEC-BTMV-04: Counter reads `total_counts`, never `binarized_mean` — \
         the column is i % 3 over 12 documents, so every bucket holds 4"
    );
    assert_eq!(
        counter.counter_denominator, 4,
        "SPEC-BTMV-04: CounterDenominator is the MAX bucket total \
         (online_ctr.cpp:934-936) and is divisor-independent"
    );
    assert!(
        counter.mean.is_empty(),
        "SPEC-BTMV-04: the Counter arm leaves `mean` empty (bake.rs:239-249)"
    );
    assert_eq!(counter.shift.to_bits(), 9_223_372_036_854_775_808);
    assert_eq!(counter.scale.to_bits(), 4_624_633_867_356_078_080);
}

#[test]
fn the_divisor_is_unreachable_from_every_non_mean_payload() {
    use crate::ctr::bake::bake_ctr_table;

    // The explicit isolation statement, over every CPU-legal type at once.
    // `bake_ctr_table`'s per-type reshape (bake.rs:226-260) populates `mean`
    // ONLY in the BinarizedTargetMeanValue | FloatTargetMeanValue arm; every
    // other arm fills `int_counts` and leaves `mean` empty. Since the whole-set
    // target border count is consumed exclusively by `binarized_mean`, which
    // only the mean arm reads, it is structurally unreachable from the payload
    // of any other type.
    let (cats, tc, proj) = e11_fixture();
    let bake = |ty: ECtrType| {
        bake_ctr_table(&cats, &proj, &tc, 2, 15, 0.5, 1.0, ty).expect("bake must succeed")
    };

    for ty in [ECtrType::Borders, ECtrType::Buckets, ECtrType::Counter] {
        let t = bake(ty);
        assert!(
            t.mean.is_empty(),
            "{ty:?} must carry NO mean payload (bake.rs:226-249), so the \
             whole-set divisor cannot reach it — SPEC-BTMV-04"
        );
        assert!(
            !t.int_counts.is_empty(),
            "{ty:?} carries its payload in `int_counts`, not `mean`"
        );
    }

    let btmv = bake(ECtrType::BinarizedTargetMeanValue);
    assert!(
        !btmv.mean.is_empty(),
        "BinarizedTargetMeanValue is the ONLY CPU-legal type whose payload the \
         divisor reaches (bake.rs:250-260) — if this is empty the isolation \
         statement is vacuous"
    );
    assert!(
        btmv.int_counts.is_empty(),
        "the mean types carry (Sum, Count) pairs INSTEAD of class counts, so \
         `int_counts` must be empty exactly for BinarizedTargetMeanValue"
    );
}
