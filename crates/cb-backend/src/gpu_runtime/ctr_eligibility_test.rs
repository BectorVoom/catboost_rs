//! T17 / DCTR-15 — the covering test for [`super::resident_combination_eligible`], the device
//! transcription of the CPU's per-level combination-CTR candidate gate;
//! **and T18 / DCTR-16** — the covering test for
//! [`super::resident_eligible_max_bucket_count`], the cat-feature-weight `maxCount` input that
//! reads that SAME predicate (§"T18 / DCTR-16" below).
//!
//! # What is under test
//!
//! Upstream's `AddTreeCtrs` (`greedy_tensor_search.cpp:491-551`, v1.2.10) builds combination
//! projections only out of `seenProj = {binAndOneHotFeaturesTree} ∪ currentTree.GetUsedCtrs()`
//! and `continue`s on `baseProj.IsEmpty()`, so **at level 0 of every tree no combination CTR
//! candidate exists at all**. `AddSimpleCtrs` is unconditional, so a single-feature projection
//! is always a candidate.
//!
//! The CPU mirrors this in `cb_train::tree::combination_ctr_eligible` (`tree.rs:2896-2906`),
//! whose callers guard on `projection.is_combination()` first. The device predicate folds that
//! caller-side guard into the function (`members.len() < 2` is unconditionally eligible),
//! because pass C enumerates a flat column list with no projection type to dispatch on.
//!
//! # Why the case list is a transcription, not a design
//!
//! Every case below is one of `crates/cb-train/src/tree_test.rs`'s seven
//! `combination_ctr_eligible` cases (`:296-361`), re-expressed in the device's
//! `&[u32]` / `&[Vec<u32>]` member-list space, **plus** the simple-projection case the CPU
//! predicate never sees because its callers filter it out first. A divergence in any expected
//! value would be a defect in the transcription, not a device design choice.
//!
//! # Source/test separation (CLAUDE.md / AGENTS.md)
//!
//! `gpu_runtime/mod.rs` is production code; every `#[test]` for the predicate lives here.
//!
//! # Backend
//!
//! [`super::resident_combination_eligible`] is a pure host predicate — it touches no
//! `ComputeClient` and launches no kernel — so this module runs under **every** backend,
//! including the default `cpu` one and `wgpu`. No GPU is required. The same is true of
//! [`super::resident_eligible_max_bucket_count`], which is a fold over host slices.
//!
//! # T18 / DCTR-16 — the filtered `maxCount`
//!
//! `CalcMaxFeatureValueCount` (`greedy_tensor_search.cpp:1070-1088`, v1.2.10) is recomputed
//! **per level** over `candidatesContexts` — the current level's already-`AddTreeCtrs`-gated
//! candidate list — so an INELIGIBLE combination's (larger) bucket count never enters the max.
//! The CPU mirror is `cb_train::tree::eligible_max_bucket_count` (`tree.rs:2920-2933`), covered
//! by `tree_test.rs`'s AT-ORD06-04a/b/c (`:388-429`); the three cases below are those three,
//! re-expressed in the device's flat `(bucket_counts, projection_members)` column space.
//!
//! **⚠ C-16 scope invariant.** The eligibility filter scopes the INNER max ONLY. The phantom
//! mixed float-partition count is folded in by the caller, OUTSIDE the filter
//! (`eligible_max.max(phantom_max).max(1)`), exactly as the CPU's
//! `max_bucket_count_with_phantom` (`tree.rs:3033`) folds it in outside its already-filtered
//! `eligible_max` input (`tree.rs:3116-3125`) — upstream's `binAndOneHotFeaturesTree` base is
//! unconditional in `AddTreeCtrs`. `phantom_max_is_folded_in_outside_the_eligibility_filter`
//! pins that.
//!
//! **⚠ R-20 is OPEN and these tests do not close it.** `ctr_device_combo` provably does not
//! discriminate D-2 (D-1 alone already passes it at `2.082e-17`). Until T19 that was
//! structural — with the cb-train gate's projection-arity conjunct in place EVERY device CTR
//! column had exactly one member, so the filter was the identity on every reachable input.
//! **T19 dropped that conjunct and re-measured**: ≥2-member columns now reach pass C, yet
//! reverting D-2's call site to the unfiltered `.max()` leaves `device_ctr_combo_fit_test`
//! byte-identical (`2.082e-17`, `grown = 5`, 8 CTR splits / 3 combinations). The filter is no
//! longer inert by construction, but **still no end-to-end test moves under it**. What is
//! proved here is (a) that the helper
//! filters, (b) — at the source level only — that pass C calls it and folds the phantom in
//! outside it. Whether reverting D-2 changes anything a fit can observe is **UNMEASURED**;
//! `SPEC.md` R-20 names **T22's device-vs-CPU split-sequence differential (DCTR-20)** as the
//! primary evidence, and if T22 measures that reverting D-2 does not move the split sequence,
//! R-20 stays open and must be recorded as such.

use super::{resident_combination_eligible, resident_eligible_max_bucket_count};

/// AT-DCTR15-a (device-only case; the CPU predicate never sees it because
/// `select_level_ctr_aware` guards on `projection.is_combination()` first): a SIMPLE
/// (single-member) projection is ALWAYS eligible, even at level 0 with nothing chosen —
/// `AddSimpleCtrs` is unconditional at every level of every tree.
#[test]
fn simple_projection_is_always_eligible() {
    assert!(
        resident_combination_eligible(&[3], &[]),
        "a single-member projection must be eligible with an empty chosen list \
         (`AddSimpleCtrs` is unconditional)"
    );
    assert!(
        resident_combination_eligible(&[3], &[vec![7], vec![1, 2]]),
        "a single-member projection must be eligible regardless of what is already chosen"
    );
}

/// AT-DCTR15-b — the device transcription of the CPU's seven `combination_ctr_eligible`
/// cases (`tree_test.rs:296-361`), one assertion each, in the same order.
///
/// | # | CPU test | members | chosen | expected |
/// |---|---|---|---|---|
/// | 1 | `combination_ineligible_when_no_ctr_used_empty` | `[1,3]` | `[]` | `false` |
/// | 2 | `combination_ineligible_when_used_is_unrelated` | `[1,3]` | `[[2]]` | `false` |
/// | 3 | `combination_eligible_extends_simple_ctr` | `[1,3]` | `[[1]]` | `true` |
/// | 4 | `combination_ineligible_length_gap_two` | `[1,2,3]` | `[[1]]` | `false` |
/// | 5 | `combination_eligible_via_any_of_multiple_used` | `[1,3]` | `[[1],[3]]` | `true` |
/// | 6 | `combination_eligible_extends_existing_combination` | `[1,2,3]` | `[[1,2]]` | `true` |
/// | 7 | `combination_ineligible_against_itself` | `[1,3]` | `[[1,3]]` | `false` |
#[test]
fn resident_combination_eligible_matches_cpu_rule() {
    // 1. The level-0 / `baseProj.IsEmpty()` skip: NOTHING chosen yet in this tree, so no
    //    combination candidate can exist. This is the invariant DCTR-15 names explicitly and
    //    the one a fit-lifetime (rather than tree-lifetime) chosen list would violate at
    //    tree 1 level 0.
    assert!(
        !resident_combination_eligible(&[1, 3], &[]),
        "a combination must be INELIGIBLE with an empty chosen list (level 0 of every tree)"
    );

    // 2. A chosen projection unrelated to the candidate (not a subset) licenses nothing.
    assert!(
        !resident_combination_eligible(&[1, 3], &[vec![2]]),
        "`[2]` is not a subset of `[1,3]` — the combination stays ineligible"
    );

    // 3. The canonical one-feature extension of an already-chosen SIMPLE CTR split.
    assert!(
        resident_combination_eligible(&[1, 3], &[vec![1]]),
        "`[1]` has one fewer member and is a subset of `[1,3]` — eligible"
    );

    // 4. A length gap of TWO is not a legitimate one-at-a-time `proj.AddCatFeature(...)`
    //    extension, even though `[1] ⊆ [1,2,3]`.
    assert!(
        !resident_combination_eligible(&[1, 2, 3], &[vec![1]]),
        "`|q| + 1 != |p|` (gap of two) must be ineligible even though `q` is a subset"
    );

    // 5. `any(..)` semantics: eligible via EITHER of two chosen simple projections.
    assert!(
        resident_combination_eligible(&[1, 3], &[vec![1], vec![3]]),
        "eligibility via ANY chosen projection, not only the first"
    );

    // 6. Extending an already-chosen COMBINATION by one more feature is legitimate.
    assert!(
        resident_combination_eligible(&[1, 2, 3], &[vec![1, 2]]),
        "`[1,2]` extended by one feature is `[1,2,3]` — eligible"
    );

    // 7. A projection is never eligible against ITSELF (length gap 0, not 1).
    assert!(
        !resident_combination_eligible(&[1, 3], &[vec![1, 3]]),
        "a projection must never license itself (`|q| + 1 != |p|` at gap 0)"
    );

    // 8. PARTIAL OVERLAP — a case the CPU's own seven tests do NOT cover, added here
    //    because §2.5's mutation of the subset conjunct (`all` → `any`) left the seven
    //    transcribed cases GREEN. Every one of them pairs a candidate with a `q` that is
    //    either a full subset or wholly disjoint, and in all seven `q` is short enough
    //    that `all` and `any` coincide (`|q| == 1`, or the arity conjunct rejects first).
    //    `[1,9]` overlaps `[1,2,3]` in exactly one member and has the right arity, so it
    //    is the shape that separates "SUBSET of `p`" from "touches `p`": upstream's
    //    `proj.AddCatFeature(...)` extension can never reach `[1,2,3]` from `[1,9]`.
    assert!(
        !resident_combination_eligible(&[1, 2, 3], &[vec![1, 9]]),
        "a PARTIALLY overlapping `q` of the right arity must still be ineligible — \
         `q` must be a SUBSET of the candidate, not merely intersect it"
    );
}

/// AT-DCTR16-a — the device transcription of the CPU's three `eligible_max_bucket_count`
/// cases (`cb-train/src/tree_test.rs:388-429`), plus the two guard cases the CPU covers
/// only implicitly.
///
/// | # | CPU test | `bucket_counts` | `projection_members` | `chosen` | expected |
/// |---|---|---|---|---|---|
/// | 1 | `max_bucket_count_excludes_ineligible_combination_at_root` | `[4,6,40]` | `[[0],[1],[0,1]]` | `[]` | `6` |
/// | 2 | `max_bucket_count_includes_combination_once_eligible` | `[4,6,40]` | `[[0],[1],[0,1]]` | `[[0]]` | `40` |
/// | 3 | `max_bucket_count_unchanged_for_all_simple_columns` | `[4,6,40]` | `[[0],[1],[2]]` | `[]` | `40` |
/// | 4 | — (the preserved `.unwrap_or(1).max(1)` guard) | `[]` | `[]` | `[]` | `1` |
/// | 5 | — (the guard's `None` arm, newly reachable) | `[40,12]` | `[[0,1],[2,3]]` | `[]` | `1` |
/// | 6 | — (T17's `is_none_or` convention) | `[4,6,40]` | `[[0],[1]]` | `[]` | `40` |
///
/// Case 1 is the whole point of D-2: the 2-member combination's bucket count **dominates**
/// (`40` vs the simple columns' `4` / `6`), and `(1 + count/maxCount)^(-model_size_reg)` is
/// INCREASING in `maxCount`, so letting it in would raise the cat-feature weight of every
/// unused simple candidate — an independent way to flip the greedy winner away from the CPU's.
/// Case 3 is the D-04 regression lock: with every column simple the filter is the identity, so
/// the value is byte-identical to the pre-T18 unfiltered `.max()`.
#[test]
fn eligible_max_excludes_ineligible_combinations() {
    // Two SIMPLE columns and one 2-member COMBINATION whose bucket count dominates.
    let bucket_counts = [4usize, 6, 40];
    let members = vec![vec![0u32], vec![1], vec![0, 1]];

    // 1. `chosen` empty — level 0 of every tree (`baseProj.IsEmpty()`), so `[0,1]` is
    //    INELIGIBLE and its 40 must not enter the max.
    assert_eq!(
        resident_eligible_max_bucket_count(&bucket_counts, &members, &[]),
        6,
        "an INELIGIBLE combination's bucket count must not enter `maxCount` — expected the \
         max over the two simple columns"
    );

    // 2. The same columns, but this tree has already chosen a CTR split on `[0]`, so
    //    `[0,1]` is a legitimate one-at-a-time extension and IS eligible.
    assert_eq!(
        resident_eligible_max_bucket_count(&bucket_counts, &members, &[vec![0]]),
        40,
        "once `[0]` is chosen, `[0,1]` is eligible and its bucket count MUST enter `maxCount`"
    );

    // 3. Regression lock (the regime every device fit with `max_ctr_complexity == 1` is in,
    //    and the regime EVERY device fit was in before T19 dropped the cb-train gate's arity
    //    conjunct): all columns simple ⇒ the filter is the identity.
    let all_simple = vec![vec![0u32], vec![1], vec![2]];
    assert_eq!(
        resident_eligible_max_bucket_count(&bucket_counts, &all_simple, &[]),
        40,
        "with every column SIMPLE the filter must be the identity — byte-unchanged from the \
         pre-T18 unfiltered `.max()`"
    );

    // 4. The preserved `.unwrap_or(1).max(1)` guard: no columns at all.
    assert_eq!(
        resident_eligible_max_bucket_count(&[], &[], &[]),
        1,
        "an empty column list must fall back to 1 (the preserved `.unwrap_or(1).max(1)` guard)"
    );

    // 5. The guard's `None` arm is reachable with a NON-empty column list for the first
    //    time once the filter exists: every column is an ineligible combination.
    assert_eq!(
        resident_eligible_max_bucket_count(&[40, 12], &[vec![0, 1], vec![2, 3]], &[]),
        1,
        "when the filter empties a non-empty column list the fallback is still 1, matching \
         the CPU's `eligible_max_bucket_count`"
    );

    // 6. An ABSENT member list counts as SIMPLE — the same `is_none_or` convention pass C's
    //    own gate uses, so the two gates cannot disagree about a degenerate column. Here
    //    `projection_members` is SHORTER than `bucket_counts`, so column 2 has no entry and
    //    its 40 is kept. Production cannot produce this (`build_device_ctr_config` rejects an
    //    empty member list with `CbError::Degenerate`); the convention exists so that a
    //    degenerate column is scored (today's behaviour) rather than silently dropped.
    let short = vec![vec![0u32], vec![1]];
    assert_eq!(
        resident_eligible_max_bucket_count(&bucket_counts, &short, &[]),
        40,
        "a column with NO member-list entry must be treated as SIMPLE (eligible), mirroring \
         pass C's `is_none_or`"
    );
}

/// AT-DCTR16-b (C-16 / checker MINOR-12d) — the phantom mixed float-partition count is folded
/// in **OUTSIDE** the eligibility filter.
///
/// This asserts the **composed** expression, not the helper alone: the helper's signature
/// cannot even see a phantom count, and the composition is the caller's
/// `eligible_max.max(phantom_max).max(1)` (`gpu_runtime/mod.rs`, pass C), mirroring
/// `max_bucket_count_with_phantom` (`tree.rs:3033`). Both directions are driven so the test is
/// not degenerate: with a DOMINATING phantom the answer is the phantom **regardless of the
/// chosen list** (it is not filtered), and with a SMALL phantom the filtered max still governs
/// (so the composition is not a constant, and the filter is still doing its job underneath).
///
/// Filtering the phantom too would diverge from the CPU in the opposite direction — it would
/// UNDER-count `maxCount` and depress every cat-feature weight.
#[test]
fn phantom_max_is_folded_in_outside_the_eligibility_filter() {
    let bucket_counts = [4usize, 6, 40];
    let members = vec![vec![0u32], vec![1], vec![0, 1]];
    let nothing_chosen: Vec<Vec<u32>> = Vec::new();
    let extended: Vec<Vec<u32>> = vec![vec![0u32]];

    // A DOMINATING phantom: 100 for BOTH chosen states — the phantom is never filtered.
    for (chosen, label) in [
        (&nothing_chosen, "chosen = [] (combination INELIGIBLE)"),
        (&extended, "chosen = [[0]] (combination ELIGIBLE)"),
    ] {
        let eligible_max = resident_eligible_max_bucket_count(&bucket_counts, &members, chosen);
        let max_bucket_count = eligible_max.max(100usize).max(1);
        assert_eq!(
            max_bucket_count, 100,
            "a DOMINATING phantom count must survive the eligibility filter unchanged \
             ({label}) — C-16: the filter scopes the INNER max only"
        );
    }

    // A SMALL phantom: the filtered max governs, and it still differs between the two
    // chosen states — i.e. the composition above is not vacuously constant.
    let small_phantom = 5usize;
    assert_eq!(
        resident_eligible_max_bucket_count(&bucket_counts, &members, &nothing_chosen)
            .max(small_phantom)
            .max(1),
        6,
        "with a small phantom the FILTERED max governs (the ineligible 40 stays out)"
    );
    assert_eq!(
        resident_eligible_max_bucket_count(&bucket_counts, &members, &extended)
            .max(small_phantom)
            .max(1),
        40,
        "with a small phantom and the combination now eligible, its 40 governs"
    );
}

/// AT-DCTR16-c — a SOURCE-LEVEL pin that pass C actually calls the filtered helper, and that
/// it folds the phantom in **outside** the call (C-16).
///
/// # What this can and cannot say
///
/// This is a **textual** pin over `gpu_runtime/mod.rs`, in the same style as
/// `cb-train`'s `boosting_ctr_gate_test.rs` source scans. It proves the call site EXISTS and
/// has the C-16 shape. It is **not** behavioural evidence: while the cb-train gate still
/// carries its projection-arity conjunct, every device CTR column has exactly one member, the
/// filter is the identity, and **no fit can observe D-2 at all** (R-20). The behavioural
/// detector is T22's device-vs-CPU split-sequence differential (DCTR-20), and it is
/// UNMEASURED as of T18.
///
/// It is fail-loud on rename/reformat by design: if the call site moves or is rewritten, this
/// test must be re-read and updated deliberately, exactly like the gate scans in cb-train.
#[test]
fn pass_c_calls_the_filtered_max_and_folds_the_phantom_outside_it() {
    let src = include_str!("mod.rs");

    const CALL: &str = "let eligible_max = resident_eligible_max_bucket_count(";
    const COMPOSE: &str = "let max_bucket_count = eligible_max.max(phantom_max).max(1);";

    assert!(
        src.contains(CALL),
        "pass C must compute `eligible_max` through `resident_eligible_max_bucket_count` \
         (DCTR-16 / D-2); the unfiltered `cs.bucket_counts.iter().copied().max()` must not \
         come back"
    );
    assert!(
        src.contains(COMPOSE),
        "C-16: the phantom count must be folded in OUTSIDE the eligibility filter, exactly as \
         `max_bucket_count_with_phantom` (`tree.rs:3033`) does"
    );

    // The helper's ARGUMENT LIST must not mention the phantom — filtering the phantom would
    // diverge from the CPU in the opposite direction (C-16 / checker MINOR-12d).
    let after_call = src.split(CALL).nth(1).unwrap_or("");
    let call_args = after_call.split(");").next().unwrap_or("");
    assert!(
        !call_args.is_empty(),
        "the `resident_eligible_max_bucket_count(` call site was found but its argument list \
         could not be delimited — the scan needs updating"
    );
    assert!(
        !call_args.contains("phantom"),
        "C-16: `phantom_max` must NOT be passed through the eligibility filter; it is folded \
         in outside it. Found call arguments: {call_args}"
    );
}
