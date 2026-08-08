//! FPP-09 (T07) unit self-oracle: COMBINATION (tensor) CTR projections are device-covered
//! and every projection member reaches the device as its own `member_bins` entry.
//!
//! Mounted as a sibling `#[path]` submodule of `boosting` (source/test separation,
//! CLAUDE.md; the `boosting_device_fold_test.rs` precedent), so it reaches the private
//! `super::{build_device_ctr_config, ctr_types_are_device_covered}` directly. PLAN blocker
//! **B-2 is resolved here the same way as T06 — option (a)-equivalent**: the function is
//! already free-standing, so it is tested in place rather than widened to `pub(crate)` for
//! an integration test that would then also need a device-free construction path.
//!
//! # The discriminating assertion
//!
//! Before this task, `build_device_ctr_config` extracted only
//! `projection.cat_features().first()` and emitted `member_bins: vec![member]`. A 2-member
//! combination therefore arrived at the device as ONE member's raw bucket column, and the
//! device would have scored a combination split from the wrong bins — WRONG, not merely
//! worse. `combination_projection_carries_every_member` is the test that fails on that.
//!
//! # The conscious re-opening of the gate (P1 / T00 / R-21 / DCTR-18)
//!
//! Until P1 this file closed with two functions —
//! `a_combination_column_set_is_NOT_device_covered_yet` and
//! `a_non_borders_column_still_declines` — that pinned `ctr_types_are_device_covered` in its
//! CLOSED form. Their contract said the gate's four conjuncts (projection arity, `ctr_type ==
//! Borders`, `target_border_idx == 0`, `prior_denom == 1.0`) may be re-opened only as
//! *"a conscious act accompanied by a passing `device_ctr_combo_fit_test` (currently
//! `#[ignore]`d), not an accident."*
//!
//! **Phase "Device CTR Coverage P1" is that conscious act, and this paragraph is the written
//! record the contract asked for.**
//!
//! (a) P1 removes all four conjuncts deliberately, one task per conjunct, each with its own
//!     kernel self-oracle and its own frozen upstream fixture — it is not a side effect of
//!     unrelated work.
//! (b) The condition the original comment attached is honoured: `device_ctr_combo_fit_test`
//!     is **un-ignored in T19** and rewritten with a `CountingGpu` device-commit assertion
//!     (`grown.get() == params.iterations`), because `oblivious_trees.len() == iterations`
//!     is satisfied by the CPU grower too and is therefore not evidence of device
//!     commitment (R-8). The stale "(currently `#[ignore]`d)" clause above is preserved only
//!     as a quotation of the superseded contract; T19 removes the attribute itself.
//! (c) The measured evidence that closed the FPP-11 escalation: on `ctr_device_combo/`,
//!     the control arm (today's device behaviour with the arity gate simply opened) misses
//!     the bar at `max|Δpred| = 2.746e-2`, while the D-1 arm (a per-level combination
//!     eligibility gate mirroring upstream's `AddTreeCtrs` `baseProj.IsEmpty()` skip) is
//!     exact at `2.082e-17`. `2.082e-17 != 1.388e-17`, the CPU-fallback value, which is
//!     independent evidence the device path actually ran.
//! (d) Those two functions are replaced by the single table-driven
//!     `gate_admits_exactly_the_current_wave` **because per-conjunct negative tests go
//!     vacuous rather than red**: all three sub-assertions of
//!     `a_non_borders_column_still_declines` used `TProjection::from_features(&[0, 1])`, a
//!     COMBINATION, so each declined via the arity conjunct as well as via its named one.
//!     Deleting the `prior_denom` conjunct would have left the `prior_denom = 2.0`
//!     assertion green — no longer a prior-denominator pin, just a second arity pin. The
//!     table uses SIMPLE projections for exactly those rows so each one isolates the
//!     conjunct named in its `flips_at` note, and it asserts the gate's admitted set in
//!     EVERY intermediate wave rather than only in its closed state.
//!
//! Every task that mutates the gate expression runs
//! `cargo test -p cb-train --lib device_ctr_combo_config_tests`.

use cb_compute::DeviceCtrAveraging;

use super::{build_device_ctr_config, ctr_types_are_device_covered};
use crate::ctr::{CtrFeatureColumn, ECtrType};
use crate::TProjection;

const N: usize = 8;
const CTR_BORDER_COUNT: usize = 15;

/// A device-covered CTR column over `projection`: Borders, target border 0, unit prior
/// denominator. `bins`/`ctr_value` are not read by `build_device_ctr_config` (it rebuilds
/// the border table from the prior), so they stay minimal.
fn covered_column(projection: TProjection, bucket_count: usize) -> CtrFeatureColumn {
    CtrFeatureColumn {
        projection,
        ctr_type: ECtrType::Borders.as_i8(),
        target_border_idx: 0,
        prior_num: 0.5,
        prior_denom: 1.0,
        bins: vec![0; N],
        ctr_value: vec![0.0; N],
        bucket_count,
    }
}

/// Raw bucket columns for three CTR-eligible cat features, each DISTINCT so a builder that
/// repeated member 0 is detectable.
fn buckets() -> Vec<Vec<u32>> {
    vec![
        (0..N).map(|i| (i % 2) as u32).collect(),
        (0..N).map(|i| (i % 3) as u32).collect(),
        (0..N).map(|i| (i % 4) as u32).collect(),
    ]
}

fn build(cols: &[CtrFeatureColumn]) -> cb_compute::DeviceCtrConfig {
    let perm: Vec<i32> = (0..N as i32).collect();
    let eligible_absolute = vec![0usize, 1, 2];
    build_device_ctr_config(
        cols,
        cols,
        &perm,
        &perm,
        &vec![0usize; N],
        &buckets(),
        &eligible_absolute,
        CTR_BORDER_COUNT,
    )
    .expect("a device-covered CTR column set must build a config")
}

#[test]
fn combination_projection_carries_every_member() {
    let projection = TProjection::from_features(&[0, 2]);
    assert!(projection.is_combination(), "the fixture must be a real 2-member projection");
    let cfg = build(&[covered_column(projection, 8)]);

    let col = cfg
        .columns
        .first()
        .expect("one structure column per input column");
    assert_eq!(
        col.member_bins.len(),
        2,
        "a 2-member combination projection must contribute TWO member_bins entries; \
         got {} — the device would score the combination split from one member's bins",
        col.member_bins.len()
    );
    assert_ne!(
        col.member_bins[0], col.member_bins[1],
        "the two members must be DISTINCT raw bucket columns, not a repeat of the first"
    );

    // Members arrive in projection-sorted order — the SAME order `combined_hash` folds in,
    // which is what makes the device's combined bin identical to the CPU's (PLAN V-4).
    let expected = buckets();
    assert_eq!(col.member_bins[0], expected[0], "member 0 == cat feature 0's buckets");
    assert_eq!(col.member_bins[1], expected[2], "member 1 == cat feature 2's buckets");
}

#[test]
fn simple_projection_is_byte_unchanged() {
    // D-04 regression: a simple projection must produce exactly what the pre-change
    // single-member extraction produced.
    let cfg = build(&[covered_column(TProjection::from_features(&[1]), 3)]);
    let col = cfg.columns.first().expect("one column");
    assert_eq!(col.member_bins.len(), 1, "a simple projection stays single-member");
    assert_eq!(col.member_bins[0], buckets()[1], "the single member is cat feature 1");
    assert_eq!(col.bucket_count, 3);
    assert!((col.prior - 0.5).abs() < f64::EPSILON, "prior = prior_num / prior_denom");
    assert_eq!(
        col.borders.len(),
        CTR_BORDER_COUNT,
        "the border table is unchanged for a simple projection"
    );
}

/// One row of the P1 GATE STATE TABLE: a single-column set, the coverage
/// `ctr_types_are_device_covered` must report for it **right now**, and the ONE task allowed
/// to change that expectation.
struct GateRow {
    /// Human-readable shape, used verbatim in the failure message.
    label: &'static str,
    column: CtrFeatureColumn,
    expected: bool,
    /// The single task permitted to flip `expected`, and the conjunct whose deletion does it.
    flips_at: &'static str,
}

fn with_type(mut col: CtrFeatureColumn, ctr_type: ECtrType) -> CtrFeatureColumn {
    col.ctr_type = ctr_type.as_i8();
    col
}

fn with_target_border(mut col: CtrFeatureColumn, target_border_idx: usize) -> CtrFeatureColumn {
    col.target_border_idx = target_border_idx;
    col
}

fn with_prior_denom(mut col: CtrFeatureColumn, prior_denom: f64) -> CtrFeatureColumn {
    col.prior_denom = prior_denom;
    col
}

/// GATE STATE TABLE (P1 / R-21 / DCTR-18). Each row is
/// `(arity, ctr_type, target_border_idx, prior_denom) -> expected coverage`. The `flips_at`
/// note names the ONE task allowed to change that row, and that task MUST change it in the
/// SAME commit as its conjunct deletion. See this module's "conscious re-opening" doc
/// section for the justification the superseded `:124-126` contract demanded.
///
/// ```text
///   row 1  simple   Borders      0  1.0  => true    flips_at: never (D-04 pin)
///   row 2  combo    Borders      0  1.0  => false   flips_at: T19 (arity)
///   row 3  simple   Counter      0  1.0  => false   flips_at: T12 (type)
///   row 4  simple   Buckets      1  1.0  => false   flips_at: T10 (type + target border)
///   row 5  simple   BTMV         0  1.0  => false   flips_at: T16 (type)
///   row 6  simple   Borders      0  2.0  => true    FLIPPED by T01 (prior denominator)
///   row 7  simple   FloatTMV     0  1.0  => false   flips_at: never (no parity surface)
///   row 8  simple   FeatureFreq  0  1.0  => false   flips_at: never
/// ```
///
/// Rows 3–6 use a **SIMPLE** projection on purpose. The functions this table replaced used
/// `TProjection::from_features(&[0, 1])` for the same three cases, so each of them declined
/// via the ARITY conjunct as well as via its named one — meaning they would have gone
/// silently VACUOUS (still green, no longer pinning anything) at T01/T10/T12, and only
/// turned red at T19, all four at once. A simple projection makes each row isolate exactly
/// the conjunct in its `flips_at` note.
///
/// **Honest limitation of row 4**: it varies two attributes against row 1 (type AND target
/// border), so it cannot distinguish "T10 removed the `target_border_idx == 0` conjunct
/// globally" from "T10 made it conditional on Buckets". Adding `simple/Buckets/b=0` and
/// `simple/Borders/b=1` rows (both `flips_at: T10`) would draw that distinction; the plan
/// rates it low-impact either way, since T23 collapses the predicate and retires this table,
/// and the extra rows would renumber the ones later tasks are specified against.
fn gate_state_table() -> Vec<GateRow> {
    let simple = || TProjection::from_features(&[0]);
    let combo = || TProjection::from_features(&[0, 1]);

    vec![
        GateRow {
            label: "simple/Borders/b=0/denom=1.0",
            column: covered_column(simple(), 2),
            expected: true,
            flips_at: "never — D-04 pin, no P1 task may flip this row",
        },
        GateRow {
            label: "combo[0,1]/Borders/b=0/denom=1.0",
            column: covered_column(combo(), 6),
            expected: false,
            flips_at: "T19 — deletes `col.projection.is_simple()`",
        },
        GateRow {
            label: "simple/Counter/b=0/denom=1.0",
            column: with_type(covered_column(simple(), 2), ECtrType::Counter),
            expected: false,
            flips_at: "T12 — admits Counter in the `ctr_type` conjunct",
        },
        GateRow {
            label: "simple/Buckets/b=1/denom=1.0",
            column: with_target_border(
                with_type(covered_column(simple(), 2), ECtrType::Buckets),
                1,
            ),
            expected: false,
            flips_at: "T10 — admits Buckets AND deletes `col.target_border_idx == 0`",
        },
        GateRow {
            label: "simple/BTMV/b=0/denom=1.0",
            column: with_type(
                covered_column(simple(), 2),
                ECtrType::BinarizedTargetMeanValue,
            ),
            expected: false,
            flips_at: "T16 — admits BinarizedTargetMeanValue in the `ctr_type` conjunct",
        },
        GateRow {
            label: "simple/Borders/b=0/denom=2.0",
            column: with_prior_denom(covered_column(simple(), 2), 2.0),
            expected: true,
            flips_at: "flipped by T01 — deleted `col.prior_denom == 1.0` (DCTR-02). The \
                       deletion is a proven no-op: the only production materialization site \
                       (`boosting.rs:2237`) passes `CTR_PRIOR_DENOM = 1.0`, pinned by \
                       `boosting_ctr_gate_tests::ctr_prior_denom_is_structurally_unit`, so \
                       this row is unreachable from any real fit",
        },
        GateRow {
            label: "simple/FloatTMV/b=0/denom=1.0",
            column: with_type(covered_column(simple(), 2), ECtrType::FloatTargetMeanValue),
            expected: false,
            flips_at: "never — GPU-only upstream (`restrictions.h:20-32`), no parity surface",
        },
        GateRow {
            label: "simple/FeatureFreq/b=0/denom=1.0",
            column: with_type(covered_column(simple(), 2), ECtrType::FeatureFreq),
            expected: false,
            flips_at: "never — GPU-only upstream (`restrictions.h:20-32`), no parity surface",
        },
    ]
}

#[test]
fn gate_admits_exactly_the_current_wave() {
    for (idx, row) in gate_state_table().into_iter().enumerate() {
        let n = idx + 1;
        let expected = row.expected;
        let actual = ctr_types_are_device_covered(&[row.column]);
        assert_eq!(
            actual, expected,
            "row {n} ({}): expected {expected}, got {actual}\n   flips_at: {}\n   \
             If you deleted that conjunct, update THIS row in the SAME commit. If you did \
             not, the device gate just moved by accident — which is exactly what the \
             conscious-act contract in this module's doc forbids.",
            row.label, row.flips_at
        );
    }
}

#[test]
fn an_all_simple_borders_set_stays_covered() {
    // D-04 pin, carried over unchanged from the retired
    // `a_combination_column_set_is_NOT_device_covered_yet` (`:139-146` at `a0a67ec`): the
    // shipped CTR arm is untouched by the escalation AND by P1. No task in P1 may flip
    // this. It is deliberately a MULTI-column set, so it also pins the `.all(..)` fold,
    // which the single-column table rows cannot reach.
    let simple = vec![
        covered_column(TProjection::from_features(&[0]), 2),
        covered_column(TProjection::from_features(&[1]), 3),
    ];
    assert!(
        ctr_types_are_device_covered(&simple),
        "an all-simple Borders set must remain device-covered"
    );
}

#[test]
fn an_unknown_projection_member_is_a_typed_error() {
    // The per-member `eligible_absolute` lookup and its typed error must survive the
    // rewrite — a combination naming a non-CTR-eligible feature is a caller bug, not a
    // silent drop.
    let perm: Vec<i32> = (0..N as i32).collect();
    let cols = vec![covered_column(TProjection::from_features(&[0, 9]), 6)];
    let err = build_device_ctr_config(
        &cols,
        &cols,
        &perm,
        &perm,
        &vec![0usize; N],
        &buckets(),
        &[0usize, 1, 2],
        CTR_BORDER_COUNT,
    )
    .expect_err("member 9 is not CTR-eligible");
    assert!(
        format!("{err}").contains('9'),
        "the typed error must name the offending member, got: {err}"
    );
}

#[test]
fn averaging_columns_get_the_same_member_treatment() {
    // The averaging permutation's columns run through the SAME closure; a fix applied to
    // only one call would leave the leaf-value gather reading one member's bins.
    let cfg = build(&[covered_column(TProjection::from_features(&[0, 2]), 8)]);
    let averaging: &DeviceCtrAveraging = cfg
        .averaging
        .as_ref()
        .expect("a covered CTR fit always populates the averaging arm");
    let col = averaging
        .columns
        .first()
        .expect("one averaging column per input column");
    assert_eq!(
        col.member_bins.len(),
        2,
        "the averaging arm must carry both members too"
    );
}

#[test]
fn device_ctr_column_carries_type_border_and_sorted_members() {
    // DCTR-01 (T02): the seam must carry the owning `CtrFeatureColumn`'s CTR IDENTITY —
    // its `ctr_type`, its Buckets numerator selector `target_border_idx`, and the FULL
    // sorted projection member set — not just the raw bins. Tracks A/B/C read `ctr_type` +
    // `target_border_idx`; Track D's per-level eligibility gate (T17) reads
    // `projection_members`. Nothing reads them yet, so this is a pure plumbing pin: the
    // device's numeric output must be byte-identical after it.
    let sources = vec![
        // A two-member COMBINATION, Borders / b = 0.
        covered_column(TProjection::from_features(&[0, 2]), 8),
        // A SIMPLE Buckets column selecting the b = 1 numerator — the only type that ever
        // carries `target_border_idx > 0` (`GetTargetBorderCount`, `ctr_helper.h:35-42`).
        with_target_border(
            with_type(covered_column(TProjection::from_features(&[1]), 3), ECtrType::Buckets),
            1,
        ),
    ];
    let cfg = build(&sources);

    // Both the structure half and the averaging half run through the same closure; a
    // population applied to only one would leave the leaf-value gather blind to the
    // identity.
    let averaging: &DeviceCtrAveraging = cfg
        .averaging
        .as_ref()
        .expect("a covered CTR fit always populates the averaging arm");
    for (half, emitted) in [("structure", &cfg.columns), ("averaging", &averaging.columns)] {
        assert_eq!(
            emitted.len(),
            sources.len(),
            "{half}: one emitted column per source column"
        );
        for (idx, (col, source)) in emitted.iter().zip(sources.iter()).enumerate() {
            assert_eq!(
                col.ctr_type, source.ctr_type,
                "{half} column {idx}: the seam must carry the source CTR type \
                 (got {}, want {})",
                col.ctr_type, source.ctr_type
            );
            let want_border = u32::try_from(source.target_border_idx)
                .expect("the fixture's target border index fits u32");
            assert_eq!(
                col.target_border_idx, want_border,
                "{half} column {idx}: the seam must carry the Buckets numerator selector \
                 (got {}, want {want_border}) — a silent 0 would select the wrong numerator",
                col.target_border_idx
            );

            let want_members: Vec<u32> = source
                .projection
                .cat_features()
                .iter()
                .map(|&m| u32::try_from(m).expect("the fixture's members fit u32"))
                .collect();
            assert_eq!(
                col.projection_members, want_members,
                "{half} column {idx}: the seam must carry EVERY projection member \
                 (got {:?}, want {want_members:?})",
                col.projection_members
            );
            // The production guardrail mirrors `build_device_ctr_config`'s empty-projection
            // rejection: an empty member list would make T17's subset test treat a
            // combination as simple.
            assert!(
                !col.projection_members.is_empty(),
                "{half} column {idx}: projection_members must have len() >= 1"
            );
            // Sorted + deduped BY CONSTRUCTION (`TProjection::from_features` sorts and
            // dedups, `projection.rs:121-126`) — the builder must NOT re-sort, and must not
            // reorder either.
            assert!(
                col.projection_members.windows(2).all(|w| w[0] < w[1]),
                "{half} column {idx}: projection_members must be STRICTLY increasing \
                 (sorted + deduped), got {:?}",
                col.projection_members
            );
        }
    }

    // The combination row is a real 2-member projection, so the member assertion above is
    // not vacuously satisfied by a single-member column.
    assert_eq!(
        cfg.columns
            .first()
            .expect("the combination column")
            .projection_members
            .len(),
        2,
        "the combination fixture must exercise a multi-member member list"
    );
}
