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
//! # Extension contract (T10 / T12 / T16 / T19 / T22 reuse this file)
//!
//! Every later task in the serial gate chain adds ONE section below, in gate-conjunct order,
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
#[test]
fn the_device_gate_no_longer_pins_the_ctr_type_to_borders() {
    let body = gate_body();
    assert!(
        !body.contains("ECtrType::Borders.as_i8()"),
        "`ctr_types_are_device_covered` still tests `ctr_type == Borders.as_i8()`; DCTR-08 \
         replaces that equality with the `{{Borders, Buckets}}` admission list. Body \
         was:\n{body}"
    );
    assert!(
        body.contains("ECtrType::from_i8"),
        "the admission list must go through `crate::ctr::ECtrType::from_i8` — it is in-crate \
         here, so re-transcribing the discriminants is forbidden (C-3), and `from_i8` is \
         what makes an UNKNOWN discriminant decline by default. Body was:\n{body}"
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

/// DCTR-10's observable completion: the gate expression itself admits `Counter`. RED before
/// T12's widening, green after.
///
/// The scan runs over the COMMENT-STRIPPED body ([`code_lines_mentioning`]), so the prose
/// inside the gate that explains why Counter is admitted cannot satisfy this on its own — the
/// enumeration entry must really be there. Exactly ONE code mention is expected: a second
/// would mean the type is tested in two places, and the admission list would no longer be a
/// single enumeration.
#[test]
fn the_device_gate_admits_counter_in_its_type_list() {
    let body = gate_body();
    let mentions = code_lines_mentioning(&body, "ECtrType::Counter");
    assert_eq!(
        mentions.len(),
        1,
        "`ctr_types_are_device_covered` must name `ECtrType::Counter` exactly once, in its \
         `from_i8` admission list (DCTR-10): Counter's device statistic is implemented \
         (`launch_counter_ctr_resident`, DCTR-09) and pinned end to end by \
         `device_ctr_counter_fit_test`. Found: {mentions:?}. Body was:\n{body}"
    );
}

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

/// DCTR-14's observable completion: the gate expression itself admits
/// `BinarizedTargetMeanValue`. RED before T16's widening, green after.
///
/// The scan runs over the COMMENT-STRIPPED body ([`code_lines_mentioning`]), so the prose
/// inside the gate that explains why BTMV is admitted cannot satisfy this on its own — the
/// enumeration entry must really be there. Exactly ONE code mention is expected: a second would
/// mean the type is tested in two places, and the admission list would no longer be a single
/// enumeration.
#[test]
fn the_device_gate_admits_binarized_target_mean_value_in_its_type_list() {
    let body = gate_body();
    let mentions = code_lines_mentioning(&body, "ECtrType::BinarizedTargetMeanValue");
    assert_eq!(
        mentions.len(),
        1,
        "`ctr_types_are_device_covered` must name `ECtrType::BinarizedTargetMeanValue` exactly \
         once, in its `from_i8` admission list (DCTR-14): BTMV's device statistic is \
         implemented (`launch_btmv_ctr_resident`, DCTR-12) and pinned end to end by \
         `device_ctr_btmv_fit_test`. Found: {mentions:?}. Body was:\n{body}"
    );
}

/// The admitted set is now CLOSED at the four CPU-legal types, and this is the executable
/// statement of that.
///
/// P1's type track ends here: `is_cpu_supported()` partitions `ECtrType` into exactly the four
/// types the device now implements and the two GPU-only ones that can never be measured against
/// a CPU oracle. A later task that widens the gate past this partition is claiming a parity
/// surface upstream does not provide, and this test is what stops that from being an accident.
#[test]
fn the_admitted_set_is_exactly_the_cpu_supported_types() {
    use crate::ctr::ECtrType;

    let body = gate_body();
    for admitted in [
        ECtrType::Borders,
        ECtrType::Buckets,
        ECtrType::BinarizedTargetMeanValue,
        ECtrType::Counter,
    ] {
        assert!(
            admitted.is_cpu_supported(),
            "{admitted:?} is named in the gate's admission list, so it MUST be CPU-legal"
        );
        assert_eq!(
            code_lines_mentioning(&body, &format!("ECtrType::{admitted:?}")).len(),
            1,
            "the gate must name `ECtrType::{admitted:?}` exactly once in its admission list; \
             body was:\n{body}"
        );
    }
    for gpu_only in [ECtrType::FloatTargetMeanValue, ECtrType::FeatureFreq] {
        assert!(
            !gpu_only.is_cpu_supported(),
            "{gpu_only:?} must stay CPU-illegal (`restrictions.h:20-32`)"
        );
        assert!(
            code_lines_mentioning(&body, &format!("ECtrType::{gpu_only:?}")).is_empty(),
            "the gate must NEVER name the GPU-only `ECtrType::{gpu_only:?}` — there is no CPU \
             oracle to prove parity against. Body was:\n{body}"
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
