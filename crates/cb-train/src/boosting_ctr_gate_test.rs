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
