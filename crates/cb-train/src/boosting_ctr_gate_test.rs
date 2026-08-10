//! Characterization pins for the device CTR admission gate
//! ([`super::ctr_types_are_device_covered`]) and for the constants that make each of its
//! conjunct removals a provable no-op.
//!
//! Mounted as a sibling `#[path]` submodule of `boosting` (source/test separation,
//! CLAUDE.md; the `device_ctr_combo_config_test.rs` precedent), so it reaches the private
//! `super::{CTR_PRIOR_DENOM, ctr_types_are_device_covered}` directly.
//!
//! # What lives here, and what does NOT
//!
//! `device_ctr_combo_config_test.rs` owns the **behavioural** GATE STATE TABLE — for every
//! `(arity, ctr_type, target_border_idx, prior_denom)` shape, the coverage the gate reports
//! *right now*. That table answers "what does the gate admit"; it cannot answer "why is
//! deleting this conjunct safe", because a routing table is silent about which shapes a real
//! fit can actually produce.
//!
//! This file answers the second question. Each P1 conjunct removal is a no-op **only**
//! because the attribute it tested is structurally constant at the single production
//! materialization site, and that structural constancy is exactly what is pinned here.
//!
//! # Extension contract (T10 / T12 / T16 / T19 used this file; **CLOSED at T23**)
//!
//! Every task in the serial gate chain added ONE section below, in gate-conjunct order,
//! consisting of:
//!
//! 1. a `#[test]` named `<attribute>_is_structurally_<invariant>` pinning the production
//!    constant / call-site that makes its deletion safe (green-on-write ⇒ the PLAN §2.5
//!    mutation check is mandatory, and its verbatim failure goes in the task note); and
//! 2. a `#[test]` named `the_device_gate_no_longer_reads_<attribute>`, using [`gate_body`],
//!    which is genuinely RED before the deletion and green after.
//!
//! Use the [`production_source`] / [`code_lines_mentioning`] / [`gate_body`] helpers rather
//! than re-deriving a source scan; they already strip comments, so a doc-comment mention of
//! a symbol never counts as a use.
//!
//! # The source-scan pins and what DCTR-18 (T23) did to each of them
//!
//! That contract produced a file of pins written against an *accumulating `matches!` list*.
//! T23 replaced the list with a delegation to
//! [`crate::ctr::ECtrType::is_cpu_supported`], so every pin had to be re-examined rather
//! than assumed. The disposition, in file order — recorded here because a pin that
//! silently stops discriminating is exactly the failure this file exists to prevent:
//!
//! | pin | disposition |
//! |---|---|
//! | `ctr_prior_denom_is_structurally_unit` (T01) | **KEPT** — scans [`production_source`], not the gate |
//! | `the_device_gate_no_longer_reads_prior_denom` (T01) | **KEPT** — still the DCTR-02 conjunct pin |
//! | `borders_and_buckets_are_the_cpu_legal_ordered_class_prefix_types` (T10) | **KEPT** — pure [`crate::ctr::ECtrType`] classification |
//! | `buckets_is_the_only_type_with_a_nonzero_target_border` (T10) | **KEPT** — classification + a `production_source` scan |
//! | `the_device_gate_no_longer_pins_the_ctr_type_to_borders` (T10) | **KEPT**, doc updated — both halves still hold and still bite |
//! | `counter_is_a_cpu_legal_whole_set_tally_not_a_class_prefix` (T12) | **KEPT** — classification + the prior-default trap |
//! | `the_device_gate_admits_counter_in_its_type_list` (T12) | **RETIRED** — unsatisfiable by the delegated form; replaced by the behavioural case |
//! | `btmv_is_a_cpu_legal_online_prefix_over_a_float_mean_not_a_class_count` (T16) | **KEPT** |
//! | `the_device_gate_admits_binarized_target_mean_value_in_its_type_list` (T16) | **RETIRED** — same reason |
//! | `the_admitted_set_is_exactly_the_cpu_supported_types` (T16) | **REWRITTEN** as [`the_gate_delegates_to_the_cpu_supported_partition`] — same claim, structural evidence |
//! | `the_device_gate_no_longer_reads_the_buckets_numerator_selector` (T10) | **KEPT** |
//! | `combination_arity_is_structurally_bounded_and_carried_whole` (T19) | **KEPT** — scans `build_device_ctr_config`'s shape, never the gate |
//! | `the_device_gate_no_longer_reads_the_projection_arity` (T19) | **KEPT** — still the DCTR-17 conjunct pin |
//!
//! ⚠ **The four raw-text pins are a trap for anyone editing the gate's inline comments.**
//! [`gate_body`] returns the function's source INCLUDING its `//` comments, and four tests
//! assert on it with `contains`: the body must never spell `prior_denom`,
//! `target_border_idx`, `ECtrType::Borders.as_i8()` or `is_simple` — not even in prose.
//! (The `ECtrType::<Variant>` scans run over the comment-STRIPPED body instead, so the
//! gate's comment may discuss the types by name.)

/// The production module this file characterizes, read at COMPILE time.
///
/// A source scan is the only way to assert "there is exactly one production call site" — a
/// value assertion cannot see a second, differently-parameterised call that a future edit
/// adds. `include_str!` resolves relative to THIS file, which is `crates/cb-train/src/`, the
/// directory `boosting.rs` lives in.
fn production_source() -> &'static str {
    include_str!("boosting.rs")
}

/// Lines of `src` that mention `needle` in CODE — line comments (`//`, `///`, `//!`) are
/// stripped first, so the many prose references to a symbol never inflate the count.
///
/// Block comments are not stripped; `boosting.rs` uses none, and a future one would show up
/// as an extra hit rather than as a silent miss (fail-loud, not fail-open).
fn code_lines_mentioning(src: &str, needle: &str) -> Vec<String> {
    src.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .filter(|line| line.contains(needle))
        .map(|line| line.trim().to_string())
        .collect()
}

/// The source text of `ctr_types_are_device_covered`, signature through closing brace, with
/// its doc comment excluded — so a conjunct's name surviving only in the prose above the
/// function does not read as a surviving conjunct.
///
/// Extraction is by the top-level `\n}` that follows the signature, which is unambiguous in
/// rustfmt'd Rust (no nested brace can sit at column 0).
fn gate_body() -> String {
    let src = production_source();
    const SIGNATURE: &str = "fn ctr_types_are_device_covered(";
    let start = src.find(SIGNATURE).unwrap_or_else(|| {
        panic!("`{SIGNATURE}` not found in boosting.rs — the gate was renamed; update this test")
    });
    let tail = src.get(start..).unwrap_or_default();
    let end = tail.find("\n}").unwrap_or_else(|| {
        panic!("no top-level closing brace after `{SIGNATURE}` — extraction assumption broken")
    });
    tail.get(..end).unwrap_or_default().to_string()
}

// ---------------------------------------------------------------------------------------
// DCTR-02 (T01) — the prior DENOMINATOR conjunct
// ---------------------------------------------------------------------------------------

/// The structural reason deleting `col.prior_denom == 1.0` from the device gate cannot
/// change any fit's routing.
///
/// Upstream forbids a non-unit CTR prior denominator on the CPU task type outright —
/// `catboost/private/libs/algo/ctr_helper.cpp:50` (v1.2.10):
///
/// ```text
/// CB_ENSURE(denom == 1.0f, "Error: CPU could use only 1 as denom for ctrs currently");
/// ```
///
/// so there is no parity surface to gain or lose. This crate mirrors that: the denominator
/// is not a parameter anywhere, it is the constant [`super::CTR_PRIOR_DENOM`], and it
/// reaches `materialize_ctr_feature` from exactly ONE production call site. A column with
/// `prior_denom != 1.0` is therefore unreachable from any real fit, and the conjunct that
/// rejected it was testing a value that can never occur.
///
/// PLAN §2.5: this test is green on write, so its discriminating power was proved by
/// mutation (const → `2.0`); the verbatim failure is recorded in `notes/T01.md`.
#[test]
fn ctr_prior_denom_is_structurally_unit() {
    assert_eq!(
        super::CTR_PRIOR_DENOM,
        1.0,
        "upstream ctr_helper.cpp:50 forbids denom != 1 on the CPU task type; if this \
         constant ever becomes non-unit, the device gate's deleted `prior_denom == 1.0` \
         conjunct must be RESTORED (see DCTR-02)"
    );

    // Exactly two code mentions: the definition, and the single materialization argument.
    // Source ORDER is deliberately not asserted — the use (`:2237`, inside
    // `materialize_ctr_columns_for_perm`) precedes the definition (`:2257`) today, and
    // that is an incidental layout fact, not a contract.
    let mentions = code_lines_mentioning(production_source(), "CTR_PRIOR_DENOM");
    assert_eq!(
        mentions.len(),
        2,
        "expected exactly the definition + ONE production use of CTR_PRIOR_DENOM, found: \
         {mentions:?}"
    );
    assert!(
        mentions
            .iter()
            .any(|line| line == "const CTR_PRIOR_DENOM: f64 = 1.0;"),
        "the denominator must stay a hard-coded constant, never a plumbed parameter; found: \
         {mentions:?}"
    );
    assert!(
        mentions.iter().any(|line| line == "CTR_PRIOR_DENOM,"),
        "the sole production materialization call (`materialize_ctr_columns_for_perm`) must \
         pass the constant BARE — an expression here would mean the denominator became \
         fit-dependent and DCTR-02's no-op proof would no longer hold; found: {mentions:?}"
    );
}

/// DCTR-02's observable completion: the gate expression itself no longer reads
/// `prior_denom`. RED before T01's deletion, green after.
#[test]
fn the_device_gate_no_longer_reads_prior_denom() {
    let body = gate_body();
    assert!(
        !body.contains("prior_denom"),
        "`ctr_types_are_device_covered` still reads `prior_denom`; DCTR-02 deletes that \
         conjunct because `CTR_PRIOR_DENOM` makes it unreachable. Body was:\n{body}"
    );
}

// ---------------------------------------------------------------------------------------
// DCTR-08 (T10) — the CTR TYPE conjunct (widened) and the TARGET BORDER conjunct (deleted)
// ---------------------------------------------------------------------------------------

/// The structural reason admitting `Buckets` is a *type-list* change and nothing more.
///
/// `Borders` and `Buckets` are exactly the CPU-legal ORDERED CLASS-PREFIX types — the class
/// the device kernel implements, selecting its numerator from `(ctr_type,
/// target_border_idx)` (SPEC §4.2, proved against `online_class_prefix` by DCTR-06's
/// `cb-backend` self-oracle). The two types the gate must NEVER admit are the GPU-only pair
/// (`restrictions.h:20-32`): they have no CPU parity surface at all, so no device fixture
/// could ever pin them.
///
/// This test pins that classification at its source ([`crate::ctr::ECtrType`]) rather than
/// restating the discriminants, which is why the gate is allowed to enumerate them via
/// `from_i8` instead of hand-rolling a second list (C-3).
///
/// PLAN §2.5: green on write ⇒ its discriminating power was proved by mutation; the verbatim
/// failure is recorded in `notes/T10.md`.
#[test]
fn borders_and_buckets_are_the_cpu_legal_ordered_class_prefix_types() {
    use crate::ctr::ECtrType;

    for admitted in [ECtrType::Borders, ECtrType::Buckets] {
        assert!(
            admitted.is_cpu_supported(),
            "{admitted:?} must be CPU-legal (`restrictions.h:18-48`) — the device gate may \
             only admit types that have a CPU parity surface to be measured against"
        );
        assert!(
            admitted.is_online_prefix(),
            "{admitted:?} must be a permutation-dependent ONLINE PREFIX type — that is the \
             accumulation the device `ordered_ctr_prefix_kernel` implements. A whole-set \
             type routed through it would get the wrong statistic (Counter is T12's own \
             kernel, BTMV is T14's)"
        );
    }
    for gpu_only in [ECtrType::FloatTargetMeanValue, ECtrType::FeatureFreq] {
        assert!(
            !gpu_only.is_cpu_supported(),
            "{gpu_only:?} is GPU-only upstream (`restrictions.h:20-32`) and must stay \
             permanently outside the device gate's admission list — there is no CPU oracle \
             to prove parity against"
        );
    }
}

/// The structural reason deleting `col.target_border_idx == 0` widens the admitted set by
/// exactly the `Buckets@1` column and nothing else.
///
/// `GetTargetBorderCount` (`ctr_helper.h:34-42`, mirrored by
/// [`crate::ctr::ECtrType::target_border_count`]) is the ONLY producer of a non-zero
/// `target_border_idx` in this crate: [`super::materialize_ctr_columns_for_perm`] loops
/// `0..ctr_type.target_border_count(TARGET_CLASSES)` at the single production
/// materialization site, and `ctr_splits_for_tree`'s `!has_ctr` fallback hard-codes `0`. At
/// binclf that count is `2` for `Buckets` and `1` for every other CPU-legal type, so a
/// `Borders` column structurally cannot carry a selector the deleted conjunct would have
/// rejected.
///
/// **If `target_border_count` ever returns more than 1 for a non-Buckets admitted type, this
/// test fails and the deleted conjunct must be reconsidered** — the device numerator
/// selector would then be reachable on a type whose kernel arm is pinned unreachable
/// (T08: `Borders@1` is the `let mut g = 0u32` initializer's dead arm).
///
/// PLAN §2.5: green on write ⇒ mutation-proved; see `notes/T10.md`.
#[test]
fn buckets_is_the_only_type_with_a_nonzero_target_border() {
    use crate::ctr::ECtrType;

    // The binclf target-class count — the same `TARGET_CLASSES` constant
    // `materialize_ctr_columns_for_perm` passes.
    const CLASSES: usize = 2;
    assert_eq!(
        ECtrType::Buckets.target_border_count(CLASSES),
        2,
        "Buckets keeps one column per class at binclf; that second column IS the b = 1 \
         numerator this device wave admits"
    );
    for single in [
        ECtrType::Borders,
        ECtrType::BinarizedTargetMeanValue,
        ECtrType::Counter,
    ] {
        assert_eq!(
            single.target_border_count(CLASSES),
            1,
            "{single:?} emits exactly ONE column per (projection, prior) at binclf, so its \
             columns can only ever carry selector 0 — which is what makes the device gate's \
             deleted selector conjunct a no-op for every type except Buckets"
        );
    }

    // Exactly one code mention: the emission loop. A second call site would mean a second
    // producer of the selector, and the argument above would no longer be exhaustive.
    let mentions = code_lines_mentioning(production_source(), "target_border_count");
    assert_eq!(
        mentions.len(),
        1,
        "expected exactly ONE production use of `target_border_count` (the emission loop in \
         `materialize_ctr_columns_for_perm`), found: {mentions:?}"
    );
    assert!(
        mentions
            .iter()
            .any(|line| line.contains("for target_border_idx in 0..ctr_type.target_border_count(")),
        "the sole use must remain the per-candidate emission loop; found: {mentions:?}"
    );
}

/// DCTR-08's first observable completion: the gate no longer pins the CTR type to `Borders`
/// by equality, and reuses the in-crate `from_i8` reconstruction instead of a second
/// hand-rolled discriminant list (C-3). RED before T10's edit, green after.
///
/// **KEPT VERBATIM THROUGH DCTR-18 (T23)**, unlike its two per-type siblings, because both
/// halves survive the delegation with their meaning intact: the equality is still forbidden
/// (a regression to `ctr_type == <one type>` is exactly the shape P1 removed) and `from_i8`
/// is still the reconstruction the final form uses. Only the failure message needed
/// updating — the `{Borders, Buckets}` admission list it named no longer exists, having
/// been superseded by the `is_cpu_supported` delegation.
#[test]
fn the_device_gate_no_longer_pins_the_ctr_type_to_borders() {
    let body = gate_body();
    assert!(
        !body.contains("ECtrType::Borders.as_i8()"),
        "`ctr_types_are_device_covered` still tests `ctr_type == Borders.as_i8()`; DCTR-08 \
         replaced that equality with a type LIST, and DCTR-18 replaced the list in turn with \
         the `is_cpu_supported` delegation. Body was:\n{body}"
    );
    // Comment-STRIPPED, and hardened to an exact count by T23: the final gate's inline
    // comment explains the delegation and therefore names its helpers, so a raw
    // `contains` here would be satisfied by prose alone (T23 MUT-B measured that).
    assert_eq!(
        code_lines_mentioning(&body, "ECtrType::from_i8").len(),
        1,
        "the admission decision must go through `crate::ctr::ECtrType::from_i8`, in CODE and \
         exactly once — it is in-crate here, so re-transcribing the discriminants is \
         forbidden (C-3), and `from_i8` is what makes an UNKNOWN discriminant decline by \
         default. Body was:\n{body}"
    );
}

// ---------------------------------------------------------------------------------------
// DCTR-10 (T12) — the CTR TYPE conjunct, widened again to admit `Counter`
// ---------------------------------------------------------------------------------------

/// The structural reason admitting `Counter` is a *type-list* change and nothing more — and
/// the reason it could NOT be admitted by widening the class-prefix arm.
///
/// Three facts, each load-bearing:
///
/// 1. **`Counter` is CPU-legal** (`restrictions.h:18-48`), so it has a parity surface a
///    fixture can be measured against — unlike the GPU-only pair.
/// 2. **`Counter` is NOT an online prefix** (`IsPermutationDependentCtrType(Counter) ==
///    false`, `ctr_type.cpp:43-56`): its numerator is the whole-learn-set bucket total and
///    its denominator the CONSTANT max bucket total. Routing it through the ordered
///    class-prefix kernel would return a silently wrong statistic, which is why `cb-backend`
///    gives it a separate permutation-free entry point and keeps the class-prefix launcher's
///    host guard rejecting the discriminant. **If this ever flips to `true`, the device
///    dispatch's Counter arm is reading the wrong upstream contract.**
/// 3. **`Counter` emits exactly ONE column per `(projection, prior)`**
///    (`target_border_count(Counter, 2) == 1`, `ctr_helper.h:34-42`), so admitting it cannot
///    interact with the numerator-selector conjunct T10 deleted.
///
/// It also pins the trap that no oracle on the frozen `ctr_device_counter` fixture can catch
/// (T06's finding): **`Counter`'s DEFAULT prior is the single `0/1`**, not the
/// `{0/1, 0.5/1, 1/1}` triple the class-count types get (`cat_feature_options.cpp:118-138`).
/// Upstream compensates a Counter prior change by shifting the CTR border, so on that fixture
/// `Prior=0.5`, `Prior=0` and `Prior=3` produce bit-identical predictions — the ≤1e-5 e2e
/// cannot police a prior mismatch. The guard is the explicit pin on both sides
/// (`"simple_ctr": ["Counter:Prior=0.5"]` in the frozen `config.json`,
/// `simple_ctr_priors = vec![0.5]` in `device_ctr_counter_fit_test`), and THIS assertion is
/// why that pin may never be dropped in favour of the default.
///
/// PLAN §2.5: green on write ⇒ its discriminating power was proved by mutation; the verbatim
/// failures are recorded in `notes/T12.md`.
#[test]
fn counter_is_a_cpu_legal_whole_set_tally_not_a_class_prefix() {
    use crate::ctr::ECtrType;

    assert!(
        ECtrType::Counter.is_cpu_supported(),
        "Counter must be CPU-legal (`restrictions.h:18-48`) — the device gate may only admit \
         types that have a CPU parity surface to be measured against"
    );
    assert!(
        !ECtrType::Counter.is_online_prefix(),
        "Counter must be permutation INDEPENDENT (`ctr_type.cpp:43-56`): a whole-set bucket \
         tally over a CONSTANT max-bucket denominator, not a read-before-increment prefix. \
         The device admits it through its OWN permutation-free entry point precisely because \
         of this; if this classification flips, the class-prefix kernel would be the right \
         home and the DCTR-10 dispatch arm is wrong"
    );

    // The binclf target-class count — the same `TARGET_CLASSES` constant
    // `materialize_ctr_columns_for_perm` passes.
    const CLASSES: usize = 2;
    assert_eq!(
        ECtrType::Counter.target_border_count(CLASSES),
        1,
        "Counter does not binarize the target at all, so it emits ONE column per \
         (projection, prior) and its columns can only ever carry selector 0"
    );

    // The prior trap (T06). Counter gets a SINGLE zero prior by default; the class-count
    // types get three. A `BoostParams` that omits `simple_ctr_priors` therefore does NOT
    // reproduce the frozen `Counter:Prior=0.5` fixture.
    let counter_defaults = ECtrType::Counter.default_priors();
    assert_eq!(
        counter_defaults.len(),
        1,
        "Counter's default prior set is the single `0/1` (`cat_feature_options.cpp:118-138`), \
         not the class-count triple; found {counter_defaults:?}"
    );
    assert!(
        counter_defaults
            .first()
            .is_some_and(|p| p.num == 0.0 && p.denom == 1.0),
        "Counter's ONE default prior must be `0/1`; found {counter_defaults:?}. The DCTR-10 \
         e2e pins `simple_ctr_priors = vec![0.5]` explicitly BECAUSE the default differs from \
         the frozen fixture's `Counter:Prior=0.5`, and upstream's compensating CTR-border \
         shift makes that mismatch invisible to the ≤1e-5 bar"
    );
    for triple in [ECtrType::Borders, ECtrType::Buckets] {
        assert_eq!(
            triple.default_priors().len(),
            3,
            "{triple:?} gets the `{{0/1, 0.5/1, 1/1}}` default triple — the contrast that \
             makes Counter's single-prior default a genuine trap rather than a uniform rule"
        );
    }
}

// RETIRED by DCTR-18 (T23): `the_device_gate_admits_counter_in_its_type_list` asserted that
// the gate body named `ECtrType::Counter` exactly once. The final gate names NO type at all
// — it delegates to `is_cpu_supported` — so that assertion is not merely false, it is
// unsatisfiable by the required form, and "keeping" it would mean forbidding the delegation
// DCTR-18 mandates. Its coverage is REPLACED, and strengthened, by
// `gate_admits_exactly_the_cpu_supported_ctr_types`, which CALLS the predicate with
// `ctr_type = 4` and asserts `true` — the thing the name scan only ever approximated.

// ---------------------------------------------------------------------------------------
// DCTR-14 (T16) — the CTR TYPE conjunct, widened a final time to admit
// `BinarizedTargetMeanValue`
// ---------------------------------------------------------------------------------------

/// The structural reason admitting `BinarizedTargetMeanValue` is a *type-list* change and
/// nothing more — and the reason it could NOT be admitted by widening the class-prefix arm.
///
/// Four facts, each load-bearing:
///
/// 1. **BTMV is CPU-legal** (`restrictions.h:18-48`), so it has a parity surface a fixture can
///    be measured against — unlike the GPU-only pair.
/// 2. **BTMV IS an online prefix** (`IsPermutationDependentCtrType`, `ctr_type.cpp:43-56`),
///    which is what separates it from `Counter`: the device must feed it the learn permutation
///    and read each bucket's history strictly BEFORE folding the document's own target in. A
///    permutation-free entry point would be the wrong home for it.
/// 3. **…but its numerator is not a class COUNT.** Its history is `TCtrMeanHistory`, a running
///    FLOAT `Sum` plus an integer `Count` (`online_ctr.h:373`), which cannot be derived from
///    one bucket's `[N0, N1]` — hence its own device accumulator, and hence the class-prefix
///    launcher's host guard rejecting the discriminant. `is_online_prefix()` alone does NOT
///    imply "routes through the class-prefix kernel"; BTMV is the counterexample, and this
///    assertion pair is what records that.
/// 4. **BTMV emits exactly ONE column per `(projection, prior)`**
///    (`target_border_count(BinarizedTargetMeanValue, 2) == 1`, `ctr_helper.h:34-42`), so
///    admitting it cannot interact with the numerator-selector conjunct T10 deleted.
///
/// It also pins the prior trap, whose shape is the OPPOSITE of Counter's: BTMV gets the
/// `{0/1, 0.5/1, 1/1}` TRIPLE by default (`cat_feature_options.cpp:118-138`), while the frozen
/// `ctr_device_btmv` fixture carries the single `Prior=0.5`. Omitting `simple_ctr_priors` would
/// therefore materialize THREE CTR columns against a one-descriptor model, which is why
/// `device_ctr_btmv_fit_test` pins `simple_ctr_priors = vec![0.5]` and asserts it before the
/// fit.
///
/// PLAN §2.5: green on write ⇒ its discriminating power was proved by mutation; the verbatim
/// failures are recorded in `notes/T16.md`.
#[test]
fn btmv_is_a_cpu_legal_online_prefix_over_a_float_mean_not_a_class_count() {
    use crate::ctr::ECtrType;

    assert!(
        ECtrType::BinarizedTargetMeanValue.is_cpu_supported(),
        "BinarizedTargetMeanValue must be CPU-legal (`restrictions.h:18-48`) — the device gate \
         may only admit types that have a CPU parity surface to be measured against"
    );
    assert!(
        ECtrType::BinarizedTargetMeanValue.is_online_prefix(),
        "BinarizedTargetMeanValue must be permutation DEPENDENT (`ctr_type.cpp:43-56`): the \
         device accumulator reads each bucket's (Sum, Count) history strictly BEFORE folding \
         the document's own target in, walking the LEARN PERMUTATION. If this classification \
         flips, the permutation-free entry point (Counter's) would be the right home and the \
         DCTR-14 dispatch arm is wrong"
    );

    // The binclf target-class count — the same `TARGET_CLASSES` constant
    // `materialize_ctr_columns_for_perm` passes.
    const CLASSES: usize = 2;
    assert_eq!(
        ECtrType::BinarizedTargetMeanValue.target_border_count(CLASSES),
        1,
        "BinarizedTargetMeanValue does not binarize the target into classes at all, so it \
         emits ONE column per (projection, prior) and its columns can only ever carry \
         selector 0"
    );

    // The prior trap, inverted relative to Counter's: BTMV gets the class-count TRIPLE, so an
    // omitted `simple_ctr_priors` materializes three columns, not one.
    let btmv_defaults = ECtrType::BinarizedTargetMeanValue.default_priors();
    assert_eq!(
        btmv_defaults.len(),
        3,
        "BinarizedTargetMeanValue's default prior set is the `{{0/1, 0.5/1, 1/1}}` triple \
         (`cat_feature_options.cpp:118-138`); found {btmv_defaults:?}. `device_ctr_btmv_fit_test` \
         pins `simple_ctr_priors = vec![0.5]` explicitly BECAUSE the default would materialize \
         THREE CTR columns against a fixture whose model.json carries exactly one descriptor"
    );
    assert_eq!(
        ECtrType::Counter.default_priors().len(),
        1,
        "Counter's single-prior default is the contrast that makes the per-type prior default a \
         genuine trap rather than a uniform rule — a task copying one e2e's params into another \
         must re-check this arm"
    );
}

// RETIRED by DCTR-18 (T23): `the_device_gate_admits_binarized_target_mean_value_in_its_type_list`,
// the exact sibling of the Counter scan retired above and retired for the same reason. Its
// coverage is REPLACED by `gate_admits_exactly_the_cpu_supported_ctr_types`' `ctr_type = 2`
// case.
//
// `the_admitted_set_is_exactly_the_cpu_supported_types` is NOT retired — it is REWRITTEN
// below (`the_gate_delegates_to_the_cpu_supported_partition`), because its *statement* is
// exactly DCTR-18's and only its *evidence* had to change: it used to prove "admitted ==
// CPU-supported" by counting each type's name in the gate text, which the delegated form
// cannot satisfy; it now proves the same thing by pinning the delegation itself.

/// The admitted set is CLOSED at the four CPU-legal types, and stays closed because the gate
/// does not carry its own list at all.
///
/// This is the rewritten form of T16's `the_admitted_set_is_exactly_the_cpu_supported_types`
/// (DCTR-14), whose assertions counted `ECtrType::<Variant>` mentions inside the gate body.
/// The claim is unchanged — *the device gate admits exactly `is_cpu_supported`* — but under
/// DCTR-18 it is established structurally rather than by enumeration, which is strictly
/// stronger in the direction that matters: a hand-rolled `matches!` list naming the same four
/// types today would satisfy the behavioural test
/// [`gate_admits_exactly_the_cpu_supported_ctr_types`] perfectly, and then drift the moment
/// [`crate::ctr::ECtrType::is_cpu_supported`] changed. **This test is the only thing that
/// forbids that list from coming back** (C-3 — `from_i8` / `is_cpu_supported` are in-crate
/// here, so re-transcribing them is a duplicate, not a decoupling).
///
/// The partition assertions are kept from the retired version: they are what make
/// "delegates to `is_cpu_supported`" equivalent to "admits exactly `{Borders, Buckets,
/// BinarizedTargetMeanValue, Counter}`", so a later change to that helper cannot silently
/// widen the device gate.
///
/// PLAN §2.5: green on write ⇒ mutation-proved; the verbatim failures are in `notes/T23.md`.
#[test]
fn the_gate_delegates_to_the_cpu_supported_partition() {
    use crate::ctr::ECtrType;

    let body = gate_body();

    // (1) The delegation, spelled in the gate body: reconstruct through `from_i8`, decide
    //     through `is_cpu_supported`. `the_device_gate_no_longer_pins_the_ctr_type_to_borders`
    //     asserts the `from_i8` half too; it is repeated here so this test stands alone as
    //     DCTR-18's statement.
    //
    //     BOTH scans run over the COMMENT-STRIPPED body, and that is not decoration: the
    //     gate's own inline comment explains the delegation and therefore SPELLS both
    //     helpers. A raw `body.contains(..)` here is satisfied by that prose and stays green
    //     through a complete un-wiring — measured, under T23's MUT-B, which is what
    //     upgraded these two assertions from `contains` to an exact code-mention count.
    assert_eq!(
        code_lines_mentioning(&body, "ECtrType::from_i8").len(),
        1,
        "the gate must reconstruct the discriminant through `crate::ctr::ECtrType::from_i8`, \
         in CODE and exactly once — that is what makes an UNKNOWN discriminant decline BY \
         DEFAULT rather than by a range test. Body was:\n{body}"
    );
    assert_eq!(
        code_lines_mentioning(&body, "is_cpu_supported").len(),
        1,
        "the gate must DELEGATE its admission decision to \
         `crate::ctr::ECtrType::is_cpu_supported` (DCTR-18), in CODE and exactly once, not \
         carry its own type list. Body was:\n{body}"
    );

    // (2) No hand-rolled list may come back. NOT ONE `ECtrType` variant may be named in the
    //     gate's code — neither an admitted one (that would be the duplicate C-3 forbids) nor
    //     a GPU-only one (there is no CPU oracle to prove parity against). The scan runs over
    //     the COMMENT-STRIPPED body, so the prose inside the gate may still discuss types.
    for variant in [
        ECtrType::Borders,
        ECtrType::Buckets,
        ECtrType::BinarizedTargetMeanValue,
        ECtrType::FloatTargetMeanValue,
        ECtrType::Counter,
        ECtrType::FeatureFreq,
    ] {
        assert!(
            code_lines_mentioning(&body, &format!("ECtrType::{variant:?}")).is_empty(),
            "the final gate must not name `ECtrType::{variant:?}` in code: an enumeration \
             here is a SECOND type list that can drift from `is_cpu_supported`, \
             `validate_ctr_types` and `materialize_ctr_feature` silently (C-3). Body \
             was:\n{body}"
        );
    }

    // (3) …and the partition it delegates to is the one DCTR-18 specifies, so (1)+(2) really
    //     do mean "admits exactly these four".
    for admitted in [
        ECtrType::Borders,
        ECtrType::Buckets,
        ECtrType::BinarizedTargetMeanValue,
        ECtrType::Counter,
    ] {
        assert!(
            admitted.is_cpu_supported(),
            "{admitted:?} is admitted by the device gate through `is_cpu_supported`, so it \
             MUST be CPU-legal (`restrictions.h:18-48`); if this flips, the device gate just \
             narrowed without anyone deciding to"
        );
    }
    for gpu_only in [ECtrType::FloatTargetMeanValue, ECtrType::FeatureFreq] {
        assert!(
            !gpu_only.is_cpu_supported(),
            "{gpu_only:?} must stay CPU-illegal (`restrictions.h:20-32`); if this flips, the \
             device gate WIDENS onto a type with no CPU parity surface, silently"
        );
    }
}

/// DCTR-08's second observable completion: the gate expression itself no longer reads the
/// Buckets numerator selector. RED before T10's deletion, green after.
///
/// (The identifier is spelled only in this assertion's `contains` argument and in prose
/// OUTSIDE the gate body — [`gate_body`] returns the raw source of the function including
/// its inline comments, so a comment mentioning the field inside the body would make this
/// test fail. That is deliberate: a "removed" conjunct that survives in a comment is still
/// a rename hazard.)
#[test]
fn the_device_gate_no_longer_reads_the_buckets_numerator_selector() {
    let body = gate_body();
    assert!(
        !body.contains("target_border_idx"),
        "`ctr_types_are_device_covered` still reads the Buckets numerator selector; DCTR-08 \
         deletes that conjunct because `target_border_count` makes a non-zero selector \
         reachable ONLY on Buckets, whose device numerator DCTR-06 implements. Body \
         was:\n{body}"
    );
}

// ---------------------------------------------------------------------------------------
// DCTR-17 (T19) — the projection ARITY conjunct (deleted; the FPP-11 escalation resolved)
// ---------------------------------------------------------------------------------------

/// The structural reasons deleting `col.projection.is_simple()` admits exactly the
/// projections the CPU itself enumerates, and hands the device all of each one.
///
/// Unlike the other three conjunct removals, this one does NOT rest on the tested attribute
/// being constant — a combination column is perfectly reachable, and admitting it genuinely
/// changes which candidates the device scores. What makes it a *coverage* change rather than
/// a semantic one is three separate structural facts, pinned here:
///
/// 1. **Arity is bounded by a fit parameter, at ONE producer.**
///    [`crate::tensor_ctr_candidates`] is the sole source of CTR candidates in this crate
///    (`AddTreeCtrs`, `greedy_tensor_search.cpp:491-551`, gated `GetFullProjectionLength <=
///    max_ctr_complexity` at `:532-533`), and `max_ctr_complexity == 1` emits SimpleCtrs
///    only. So the device can never be handed an unbounded projection family, and a user who
///    does not ask for combinations still gets exactly today's admitted set.
/// 2. **A projection's member list is SORTED and de-duplicated by construction**
///    ([`crate::TProjection::from_features`]), which is the order the CPU's `combined_hash`
///    folds in AND the order [`super::build_device_ctr_config`] emits both `member_bins` and
///    `projection_members` in. The device therefore folds the same members in the same order;
///    `SPEC` §4.1's "SORTED" requirement needs no re-sort and no runtime check.
/// 3. **Both the structure and the averaging column lists come from ONE producer.** There is
///    exactly one `DeviceCtrColumn` construction in this file, inside `build_columns`, an
///    order-preserving `map` applied to each list. That is what makes the device's
///    POSITION-indexed leaf gather pair with the host's FULL-IDENTITY gather (T10 §1); a
///    task that filters, reorders or de-duplicates one list without the other breaks the
///    pairing silently (measured at `|Δ| = 2.506e-1` on `device_ctr_buckets_fit_test`).
///    Note the DCTR-15 eligibility gate deliberately skips a column from SCORING inside
///    `cb-backend`'s pass C — it never removes one from either list, which is why it is
///    compatible with this invariant.
///
/// What this test canNOT say is that the device's per-level *candidate* semantics match the
/// CPU's — that is `cb-backend`'s `resident_combination_eligible` / DCTR-15, whose covering
/// tests live in `crates/cb-backend/src/gpu_runtime/ctr_eligibility_test.rs`, and whose
/// end-to-end evidence is `tests/device_ctr_combo_fit_test.rs` (`grown == iterations`,
/// `max |Δpred| = 2.082e-17`).
///
/// PLAN §2.5: green on write ⇒ mutation-proved; see `notes/T19.md`.
#[test]
fn combination_arity_is_structurally_bounded_and_carried_whole() {
    use crate::{tensor_ctr_candidates, TProjection};

    // (1) The arity bound. Three CTR-eligible cat features (cardinality above
    // `one_hot_max_size`), so the enumeration is not degenerate.
    let cards = [10u32, 10, 10];
    let simple_only = tensor_ctr_candidates(&cards, 1, 1);
    assert!(
        !simple_only.is_empty() && simple_only.iter().all(|c| c.is_simple),
        "`max_ctr_complexity == 1` must emit SimpleCtrs ONLY — that is what keeps the \
         device's admitted set unchanged for a fit that never asked for combinations; got \
         {simple_only:?}"
    );
    for max_complexity in [2usize, 3] {
        let candidates = tensor_ctr_candidates(&cards, 1, max_complexity);
        assert!(
            candidates.iter().any(|c| !c.is_simple),
            "`max_ctr_complexity == {max_complexity}` must emit at least one COMBINATION, \
             or the deleted arity conjunct would have been unreachable and this whole track \
             vacuous"
        );
        assert!(
            candidates
                .iter()
                .all(|c| c.projection.full_projection_length() <= max_complexity),
            "`tensor_ctr_candidates` must honour the `GetFullProjectionLength <= \
             max_ctr_complexity` gate (`greedy_tensor_search.cpp:532-533`) — the device gate \
             no longer bounds the arity itself, so this is the ONLY bound"
        );
    }

    // (2) Sorted, de-duplicated members — the fold order shared by `combined_hash` and the
    // device seam.
    let projection = TProjection::from_features(&[2, 0, 2, 1]);
    assert_eq!(
        projection.cat_features(),
        [0usize, 1, 2],
        "`TProjection` must keep its members sorted and de-duplicated (`AddCatFeature` / \
         `IsRedundant`); `build_device_ctr_config` deliberately does NOT re-sort, so a \
         regression here would silently reorder the device's `member_bins` fold"
    );

    // (3) One seam producer, applied to both lists. A source scan is the only way to assert
    // "there is exactly one construction site" — a value assertion cannot see a second one a
    // future edit adds.
    let src = production_source();
    let constructions = code_lines_mentioning(src, "DeviceCtrColumn {");
    assert_eq!(
        constructions.len(),
        1,
        "expected exactly ONE production `DeviceCtrColumn` construction (inside \
         `build_device_ctr_config`'s `build_columns`); a second producer would break the \
         single-producer pairing the device's position-indexed leaf gather depends on. \
         Found: {constructions:?}"
    );
    let build_calls = code_lines_mentioning(src, "build_columns(");
    assert_eq!(
        build_calls.len(),
        2,
        "expected EXACTLY TWO `build_columns` calls — the structure list and the averaging \
         list — so both halves are the same order-preserving map over the same specs. \
         (The closure's own `let build_columns = |…|` binding carries no `(`, so it is not \
         counted here.) Found: {build_calls:?}"
    );

    // And the single candidate producer this crate feeds that seam from.
    let producer = code_lines_mentioning(src, "tensor_ctr_candidates(");
    assert_eq!(
        producer.len(),
        1,
        "expected exactly ONE production call to `tensor_ctr_candidates` — the sole CTR \
         candidate enumeration, and therefore the sole place `max_ctr_complexity` bounds the \
         arity the device can see. Found: {producer:?}"
    );
}

/// DCTR-17's observable completion: the gate expression itself no longer reads the
/// projection arity. RED before T19's deletion, green after.
///
/// (As with the `target_border_idx` pin above, [`gate_body`] returns the RAW source of the
/// function including its inline comments, so spelling `is_simple` in a comment inside the
/// body would fail this test. That is deliberate — a "removed" conjunct surviving in a
/// comment is still a rename hazard — and it is why the body's DCTR-17 comment says
/// "projection-arity conjunct" in prose instead.)
#[test]
fn the_device_gate_no_longer_reads_the_projection_arity() {
    let body = gate_body();
    assert!(
        !body.contains("is_simple"),
        "`ctr_types_are_device_covered` still tests the projection arity; DCTR-17 deletes \
         that conjunct — the FPP-11 escalation it was restored under is resolved (the primary \
         cause was the device's MISSING per-level combination-eligibility gate, DCTR-15, not \
         the arity itself), and `device_ctr_combo_fit_test` now passes UN-IGNORED at \
         `max |Δpred| = 2.082e-17` with `CountingGpu.grown == iterations`. Body was:\n{body}"
    );
}

// ---------------------------------------------------------------------------------------
// DCTR-18 (T23) — the FINAL gate form: delegation to `from_i8` / `is_cpu_supported`
// ---------------------------------------------------------------------------------------
//
// Two tests replace three retired source-scan pins (see `the_gate_names_no_ctr_type_variant`
// for the accounting). The first is BEHAVIOURAL — it calls the predicate over every
// discriminant instead of scanning its text — and is therefore the durable statement of the
// admitted set; the second pins the DELEGATION itself, which is the only thing a behavioural
// test cannot see (a hand-rolled `matches!` list that happens to enumerate the same four
// types satisfies the first test exactly).

/// A CTR column carrying `ctr_type`, deliberately shaped so that **every conjunct P1
/// removed would reject it**: a two-member COMBINATION projection (the arity conjunct T19
/// deleted) and `target_border_idx = 1` (the numerator-selector conjunct T10 deleted).
///
/// So a `true` from the gate on such a column is only possible when both conjuncts are
/// really gone — the admitted-type rows below double as arity/selector regression pins.
/// `prior_denom` is the production `1.0`; the non-unit case is covered separately by
/// `gate_ignores_a_non_unit_prior_denominator` and by the gate-state table's row 6.
fn combination_column_with_type(ctr_type: i8) -> crate::ctr::CtrFeatureColumn {
    const N: usize = 4;
    crate::ctr::CtrFeatureColumn {
        projection: crate::TProjection::from_features(&[0, 1]),
        ctr_type,
        target_border_idx: 1,
        prior_num: 0.5,
        prior_denom: 1.0,
        bins: vec![0; N],
        ctr_value: vec![0.0; N],
        bucket_count: 4,
    }
}

/// DCTR-18's observable completion, stated BEHAVIOURALLY: the gate admits exactly the
/// CPU-supported CTR types, on a projection of any arity and any numerator selector, and
/// declines everything else — including discriminants that are not `ECtrType` values at all.
///
/// This is the durable replacement for the two retired per-type source scans
/// (`the_device_gate_admits_counter_in_its_type_list`,
/// `the_device_gate_admits_binarized_target_mean_value_in_its_type_list`), and it is
/// strictly stronger than they were: they asserted that a type's NAME appeared in the gate's
/// text, which the delegated form cannot satisfy and which never proved the predicate
/// actually returned `true` for it.
///
/// Every case is evaluated before reporting, mirroring the gate-state table's contract —
/// a fail-fast loop would hide the rest of the admitted-set diff.
///
/// PLAN §2.5: this test is GREEN ON WRITE (the pre-T23 `matches!` list already admits
/// exactly these four types and `from_i8` already declines unknown discriminants), so its
/// discriminating power was proved by mutation; the verbatim failures are in `notes/T23.md`.
#[test]
fn gate_admits_exactly_the_cpu_supported_ctr_types() {
    use crate::ctr::ECtrType;

    // A free `fn` rather than a closure: the loop below also pushes to `mismatches`
    // directly, which a `FnMut` capture would forbid (E0499).
    fn check(
        mismatches: &mut Vec<String>,
        label: String,
        cols: &[crate::ctr::CtrFeatureColumn],
        expected: bool,
    ) {
        let actual = super::ctr_types_are_device_covered(cols);
        if actual != expected {
            mismatches.push(format!("{label}: expected {expected}, got {actual}"));
        }
    }

    let mut mismatches: Vec<String> = Vec::new();

    // Every ECtrType discriminant, admitted iff CPU-legal (`restrictions.h:18-48`). The
    // expectation is DERIVED from `is_cpu_supported`, not restated — a second hand-written
    // list here would reintroduce exactly the drift C-3 forbids in the production gate.
    for discriminant in 0_i8..=5 {
        let Some(ctr_type) = ECtrType::from_i8(discriminant) else {
            mismatches.push(format!(
                "discriminant {discriminant} must be a known `ECtrType` — `from_i8` covers 0..=5"
            ));
            continue;
        };
        check(
            &mut mismatches,
            format!("{ctr_type:?} (discriminant {discriminant})"),
            &[combination_column_with_type(discriminant)],
            ctr_type.is_cpu_supported(),
        );
    }

    // Unknown discriminants decline by DEFAULT, because reconstruction goes through
    // `from_i8` (which returns `None`) rather than through a range test. `i8::MIN` /
    // `i8::MAX` are the boundary strays; `6` is the first value past the enum and `-1` the
    // first below it.
    for stray in [6_i8, 7, -1, i8::MIN, i8::MAX] {
        check(
            &mut mismatches,
            format!("unknown discriminant {stray}"),
            &[combination_column_with_type(stray)],
            false,
        );
    }

    // A MIXED set: one admitted column plus one GPU-only column must decline as a whole —
    // the `.all(..)` fold, which no single-column case can reach.
    check(
        &mut mismatches,
        "mixed {Borders, FeatureFreq}".to_string(),
        &[
            combination_column_with_type(ECtrType::Borders.as_i8()),
            combination_column_with_type(ECtrType::FeatureFreq.as_i8()),
        ],
        false,
    );

    // The EMPTY set declines — the caller's `is_empty()` arm owns that, and `.all(..)` over
    // an empty slice is vacuously `true`, so without the leading `!cols.is_empty()` a fit
    // with no CTR columns at all would be reported as CTR-covered.
    check(
        &mut mismatches,
        "the empty column set".to_string(),
        &[],
        false,
    );

    assert!(
        mismatches.is_empty(),
        "the FINAL device CTR gate does not admit exactly the CPU-supported type set:\n{}\n\
         DCTR-18 fixes this set permanently for P1: the four CPU-legal types are admitted on \
         ANY arity and ANY numerator selector, and the two GPU-only types \
         (`restrictions.h:20-32`) plus every unknown discriminant decline. Widening it past \
         `is_cpu_supported` claims a CPU parity surface upstream does not provide.",
        mismatches.join("\n")
    );
}

/// The prior DENOMINATOR is not read by the final gate either — the behavioural counterpart
/// of `the_device_gate_no_longer_reads_prior_denom`'s source scan.
///
/// Kept separate from the loop above so the loop's columns stay at the production
/// `prior_denom = 1.0` and a failure here names the denominator rather than the type.
#[test]
fn gate_ignores_a_non_unit_prior_denominator() {
    let mut col = combination_column_with_type(crate::ctr::ECtrType::Borders.as_i8());
    col.prior_denom = 2.0;
    assert!(
        super::ctr_types_are_device_covered(&[col]),
        "the final gate must ignore `prior_denom` entirely (DCTR-02): the conjunct was \
         deleted as provably dead, since `CTR_PRIOR_DENOM` is the only denominator any real \
         fit can produce (`ctr_prior_denom_is_structurally_unit`)"
    );
}
