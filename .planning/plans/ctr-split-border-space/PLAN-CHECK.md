---
title: Plan check — BUG-CTRB (CTR split border space)
verdict: PASS
pass: 2
checked_at: 2026-08-02
plan: .planning/plans/ctr-split-border-space/PLAN.md (v2, 1502 lines)
spec: .planning/plans/ctr-split-border-space/SPEC.md (spec_version 1, revised)
head: fed705f
---

## Plan Check Result

**Verdict:** PASS
**Goal:** Make the trainer persist `CtrSplitSpec.border` as a VALUE-space threshold
matching upstream's `(bin+1) − 2⁻²⁰`, so training (`bin > b`) and apply
(`ctr_value > border`) agree, without breaking any existing consumer.
**Plan:** `.planning/plans/ctr-split-border-space/PLAN.md` v2 (6 tasks, C01–C06, 5 waves)

### Summary

- **All three pass-1 findings are genuinely FIXED, not merely acknowledged.** Each
  fix was re-verified against the working tree, not accepted from the plan's prose.
- The CRITICAL is fully repaired **in both directions**: I independently re-derived
  the complete `LevelKind::Ctr` border-payload consumer set (one site) and confirmed
  that converting `:3289` leaves the E12 gate, all 11 CTR oracles and all 3 one-hot
  targets GREEN — so C04's restated M2 enumerates outcomes that will actually occur,
  and its "required green" half is not a trap.
- **JOB 2 adjudication: KEEP M2 as written — option (a), sound.** It is not merely
  harmless-but-redundant; under the inherited §3.1 protocol it is *required*, because
  C01 step 6 / C04 Test D are otherwise unfalsified guard assertions. One SPEC
  sentence needs rewording so it stops literally forbidding what the plan now does.
- **JOB 3: no new CRITICAL or MAJOR defect was introduced by the v2 rewrite.** Five
  MINOR prose/staleness items, one of them newly created by the rewrite (C04 still
  tells the executor to preserve a warning comment that v2 deleted from C03).
- **JOB 4: executable end to end.** Every Red can fail for its stated reason, every
  Green is reachable, both mutation checks are satisfiable, and the §3.2 command
  blocks are faithful to ctr-type `PLAN.md` §3.2.

---

### Pass-1 disposition table

| # | Pass-1 finding | Severity | Disposition at pass 2 |
|---|---|---|---|
| 1 | "Principal hazard" false; C04 M2 unsatisfiable; C03 mandated a false production comment | CRITICAL | **FIXED — verified.** §0 rewritten + retraction; D3/D3a added; C03's `:3287-3290` comment replaced with an arithmetically correct units-contract text; M2 restated to four enumerated outcomes (2 fail, 2 required-green); C01 step 6, C04 criteria and §6 realigned. `grep "Do NOT apply"`/`"silently corrupt"` → the false claim appears **only** inside retraction blocks, never as mandated source. |
| 2 | C04 ∥ C05 not disjoint (same file, C04-only helper) | MAJOR | **FIXED — verified.** `C04 → C05` serialized; §5 edges, the wave diagram (W0..W4, longest path 5), C05's `Depends on: C04`, the exclusive-resource declaration on `crates/cb-train/src/tree.rs`, and pre-flight **and** post-flight `git diff` gates in both tasks all agree. No residual "disjoint/parallel" claim about C04/C05 anywhere. |
| 3 | E6's parity conclusion inferred; f32 formula not general; `0..=254` sweep implies generality | MAJOR | **FIXED — verified.** E6 split into `[VERIFIED]` arithmetic / `[INFERRED]` generality; **D2 re-based on the `.cbm` f32 fixed point (E14)**, explicitly "NOT parity generality"; Tests B/C bounded to `0..ctr_border_count_default()`; `0..=254` survives only as a prohibition; **new Test E** (`b = 16` characterization) exists, is numerically correct, and carries the STOP condition + `boosting.rs:3238` reachability guard; new OQ-D records the open upstream question. |
| 4 | C01 step 3 bit-exact only at `norm == 1.0` | MINOR | **FIXED.** Prior pinned **non-tunable** at `0.5/1.0` with `assert_eq!((shift, norm), (0.0, 1.0))` before the premise check; corpus tuning scoped to rows/categories only (E22). |
| 5 | C05 had no `save_cbm` precedent and only "STOP" | MINOR | **FIXED.** E23 added; C05 step 5 is a mandatory 3-part in-memory fallback gate **plus its own `+0.1` mutation**, with "STOP alone is not an acceptable outcome". |
| 6 | Plan dir missing from §1 protected untracked list | MINOR | **FIXED.** §1 now lists **three** paths; `git status --short` at `fed705f` returns exactly those three. |
| 7 | `<mod>` placeholder in C04 step 2 | MINOR | **FIXED.** `tree::general`, matching the verified mount `crates/cb-train/src/tree.rs:92-94`. |

---

### JOB 1 — verification of the three fixes

#### 1a. The retraction in §0 is accurate (re-derived, not accepted)

`[VERIFIED: CODEGRAPH assign_leaf_of_averaging + LOCAL reads]`

- `crates/cb-train/src/tree.rs:3287-3290` = `LevelKind::Ctr { ctr_idx, border: *border }`
  (`border: *border` on **:3289**); `crates/cb-train/src/tree.rs:3291-3303` =
  `CtrSplitSpec { … }` (`border: *border` on **:3297**). Both line anchors exact.
- **Complete consumer set of the `LevelKind::Ctr` border payload, workspace-wide**
  (`grep -rn "LevelKind::Ctr"` cross-checked against CodeGraph):
  - `crates/cb-train/src/boosting.rs:1926-1938` — **the only read**:
    `.and_then(|col| col.bins.get(obj)).is_some_and(|&bin| f64::from(bin) > *border)`.
  - `crates/cb-model/src/model.rs:450` — `{ ctr_idx, .. }`, border ignored.
  - `crates/cb-train/src/boosting.rs:4109` — `{ .. }`, ignored.
  - `crates/cb-train/src/tree_one_hot_fused_test.rs:105,262` — `{ .. }`, ignored.
  - `crates/cb-train/tests/ctr_split_scoring_test.rs:125` —
    `assert!(matches!(grown.level_kinds[0], LevelKind::Ctr { .. }))`: **the only oracle
    that touches `level_kinds` at all, and it does not read the border.**
  - `crates/cb-train/src/boosting_test.rs:325` and
    `crates/cb-model/tests/mixed_kind_split_order_test.rs:60` — hand-built literals,
    not trainer output.
- Operand type confirmed: `pub bins: Vec<u32>`
  (`crates/cb-train/src/ctr/ctr_feature.rs`, `CtrFeatureColumn` at `:71`). `f64::from(u32)`
  is exact, so for integer `bin` and integer `b`: `bin > b ⟺ bin ≥ b+1 ⟺ bin > (b+1) − 2⁻²⁰`.
- Reachable domain confirmed: `for border_idx in 0..ctr_border_count` at
  `crates/cb-train/src/tree.rs:3162`, `let border = border_idx as f64;` at `:3163`;
  `ctr_border_count_default() -> 15` at `crates/cb-train/src/boosting.rs:529-531`;
  **the only production call site is `crates/cb-train/src/boosting.rs:3238`** — every
  other reference is a test or the `lib.rs:95` re-export
  `[VERIFIED: LOCAL grep -rn ctr_border_count_default --include=*.rs crates/]`.
  So `b ∈ 0..=14`, and `bins` is clamped to `[0, 15]` (`ctr_feature.rs` step 4).

**⇒ The §0 retraction is correct. C03's replacement comment is arithmetically CORRECT**
("Because that operand is an INTEGER, `bin > b` and `bin > (b+1) - 2^-20` are
arithmetically EQUIVALENT here"), and it no longer asserts corruption anywhere.

#### 1b. C04's restated M2 enumerates outcomes that will actually occur

The pass-2 prompt asks specifically whether the claim "converting `:3289` leaves the
E12 gate and all 11 oracles + 3 one-hot targets GREEN" is right — because if it were
wrong, M2 would be broken in the other direction.

It is **right**, on two independent grounds:

1. **Arithmetic:** the only consumer's left operand is an exact-integer `u32`
   (above), and the mutated threshold `(b+1) − 2⁻²⁰` lies strictly inside `(b, b+1)`
   for every reachable `b ≤ 14`, so the boolean is identical for every possible `bin`
   value (including the clamp ceiling `bin = 15`).
2. **Assertion survey:** no test in the eleven oracles, the three one-hot targets or
   the E12 target asserts on a `LevelKind::Ctr` **border value** — the single site that
   inspects `level_kinds` (`ctr_split_scoring_test.rs:125`) matches `{ .. }`, and
   `GrownTree.level_kinds` never reaches `.cbm`/json (the only bridge,
   `Model::from_trained`'s mixed-kind arm at `model.rs:450`, discards the border).

M2's outcomes 1 and 2 (Test D and C01 step 6 fail) are equally certain: for
`b ∈ 0..=14`, `(b+1) − 2⁻²⁰` is non-integral, so `assert_eq!(bin_border, bin_border.trunc())`
fails. **M2 is satisfiable in both directions and is correctly specified.**

#### 1c. C04 ∥ C05 serialization is real

`§5` diagram (`C01|C02 → C03 → C04 → C05 → C06`, "Acyclic. Longest path … (5 waves)"),
the wave headers (`W0`…`W4`), C04's `Order: 4 · Depends on: C03` +
`Exclusive resource: crates/cb-train/src/tree.rs`, C05's `Order: 5 · Depends on: **C04**
(not merely C03)` + `Exclusive resource … taken over from C04`, and the pre-flight /
post-flight `git diff crates/cb-train/src/tree.rs` gates in **both** tasks are mutually
consistent. C06's part-4 table carries a dedicated **"no residual mutation"** row.
No surviving statement anywhere claims C04 and C05 are disjoint or parallel.

#### 1d. The f32 formula is bounded to the reachable domain

I recomputed the sweep independently (numpy float32/float64):

| b | f32 form | in `(b, b+1)`? | f32 fixed point? | f64 form | f32 fixed point? |
|---|---|---|---|---|---|
| 0–15 | `b.999999046325684` | yes | yes | identical | yes |
| 16 | **`17.0`** | **no** | yes | `16.999999046325684` | **no** |
| 17 / 31 / 127 / 254 | `18.0 / 32.0 / 128.0 / 255.0` | no | yes | `…999999…` | no |

- **Test B / Test C are bounded** to `0..ctr_border_count_default()` (= `0..=14`), with
  the loop-bound rationale stated as the reachability guarantee. ✔
- **The `0..=254` sweep is gone** — `grep -n 254 PLAN.md` returns only E6's
  characterization row, the revision log, C04's prohibition and the completion checkbox. ✔
- **Test E exists and is correct.** `ctr_bin_border_to_value_space(16.0) == 17.0` ✔;
  it is framed as a CHARACTERIZATION with a STOP condition; it names
  `boosting.rs:3238`; it asserts the reachability guard itself
  (`ctr_border_count_default() == 15`); and it carries the honesty note that upstream's
  epsilon above `b = 15` is `[UNVERIFIED]` (the `(b+1) − 1e-6f` alternative rounds the
  other way). It also correctly becomes **D2's discriminator**: an f64 helper body
  yields `16.999999046325684 ≠ 17.0`, so M1 falsifies it. ✔
- **The STOP threshold is arithmetically right.** "if `ctr_border_count` … exceeds 16"
  = `≥ 17` ⇒ `b` can reach 16. At `ctr_border_count == 16`, `b ≤ 15`, still safe. ✔
- **D2's rationale is re-based** on the `.cbm` fixed point, verified end to end:
  save narrows at `crates/cb-model/src/cbm.rs:437`
  (`identity.borders.iter().map(|&b| b as f32)`), load widens at `cbm.rs:601`
  (`let border = f64::from(borders.get(border_index));`), and the encode-side lookup is
  bit-exact at `cbm.rs:366` against borders collected from the model's own splits
  (`build_ctr_features`, `cbm.rs:291`, sorted `:321`, bit-deduped `:322`) — so it cannot
  miss when the border value changes. D2 now says explicitly "The rationale is NOT
  'parity generality'". ✔

#### 1e. The three MINORs

- **C01 prior pinning:** the PIN block is present and non-tunable, with
  `assert_eq!((shift, norm), (0.0, 1.0), …)` before the premise check. E22's derivation
  matches `calc_normalization` (`crates/cb-train/src/ctr/calc_ctr.rs:60-66`), and
  `materialize_ctr_feature` does take `prior_num, prior_denom` separately and derives the
  scalar internally, so C01's `calc_normalization(prior_num / prior_denom)` is the right
  reconstruction. ✔
- **C05 fallback gate:** step 5 is mandatory, has three concrete in-memory assertions,
  and applies the `+0.1` mutation to Fallback 1 so the fallback is itself falsifiable.
  Fallback 2 correctly anticipates that `build_ctr_features` / `ctr_split_to_global_index`
  are private (they are — both `fn`, not `pub`, in `cbm.rs`) and offers the equivalent
  direct property. ✔
- **`tree::general`:** mount verified verbatim at `crates/cb-train/src/tree.rs:92-94`
  (`#[cfg(test)] #[path = "tree_test.rs"] mod general;`); `tree_test.rs:1-7` has no
  file-level `#![allow]` and uses `use crate::tree::{…}` + `use crate::TProjection;`,
  exactly as §3.0 states. ✔
- **§1 protected paths:** `git status --short` at `fed705f` returns exactly
  `?? .planning/plans/ctr-split-border-space/`,
  `?? crates/cb-oracle/fixtures/ctr_counter_simple/`,
  `?? crates/cb-train/tests/ctr_counter_simple_oracle_test.rs` — the three the plan lists. ✔

---

### JOB 2 — adjudication: keep M2 (option **(a)**, sound and consistent)

**Recommendation: KEEP mutation M2 exactly as v2 writes it. Do not drop it.**

1. **It is required, not optional.** C01 step 6 and C04 Test D are *guard* assertions:
   they pass on first write. The inherited protocol (ctr-type `PLAN.md` §3.1, restated
   in BUG-CTRB §3.1) says a guard that cannot be made to fail by a named mutation "is
   not a guard and the task is not complete". Converting `:3289` is the **only**
   mutation in existence that falsifies those two assertions. Dropping M2 would leave
   the sole detector of the wrong-site edit unproven — a strictly worse plan.
2. **It is a well-formed §3.1 mutation check.** Named, single-expression, with a named
   failing message and a manual revert. Outcomes 1 and 2 are the guard proof; outcomes
   3 and 4 are *evidence*, recorded as required results.
3. **Risk (c) — executor escalation on a "partially green" check — is adequately
   contained.** Five independent brakes: D3a; the §0 enforcement box; the four
   enumerated outcomes with "Outcomes 3 and 4 are REQUIRED results, not incidental";
   the explicit "Do NOT strengthen a test, alter production code, or hunt for a further
   failure in order to make an oracle regress under M2"; and a dedicated completion
   checkbox recording that nothing was strengthened. C04's risk section names the trap
   by its pass-1 label. I judge residual escalation risk **low**.
4. **One wording fix is needed — in SPEC, not in PLAN.** SPEC-CTRB-03's sentence *"no
   mutation check can be built around `:3289`"* is now over-broad and, read literally,
   forbids exactly what M2 is. The true statement is narrower: *no ORACLE-BASED
   expectation can be built around `:3289`; the mutation check that CAN be built is M2,
   whose detectors are the integrality assertions.* Since C06 must verify SPEC↔E20/D3a
   consistency, leaving the absolute phrasing invites a spurious STOP. See MINOR-2.

---

### JOB 3 — did the v2 rewrite introduce new defects?

No CRITICAL or MAJOR. Checks performed and their results:

- **§0 / D3 / D3a vs task bodies:** consistent. C03's risk section, C04's M2 block and
  risk section, C06's SPEC-CTRB-03 note, §6's SPEC-CTRB-03 row and §8's closing bullet
  all state the same no-op framing with the same detector set. No contradiction found.
- **Did serialization break an assumption C05 made about running in parallel?** No.
  C05's only parallelism-dependent content was its mutation text; it now correctly
  targets the post-C04 helper call, its pre-flight expects "C03 + C04 hunks", and its
  risk row says "serialized (§5)". Its `+0.1` mutation is still valid post-extraction
  and still falsifies Test 1 (verified: `f32(x + 0.1) ≠ x + 0.1` at these magnitudes,
  ulp = 2⁻²⁰ in `[8,16)`), and it cannot break `save_cbm`'s lookup because the border
  pool is built from the model's own splits (`cbm.rs:291/321-322/366`).
- **Does Test E contradict the reachability guard it asserts?** No. Test E calls the
  helper directly with `16.0`, outside Tests B/C's `0..=14` sweep and outside the doc
  block's `≤ 15` DOMAIN; it labels the result a characterization of an unreachable input
  and asserts the guard (`ctr_border_count_default() == 15`) in the same test. Coherent.
- **Stale cross-references / task IDs / line numbers / wave numbers:** wave numbering,
  task orders and dependency edges are internally consistent. Two staleness items found,
  both MINOR (MINOR-1 and MINOR-4 below).
- **Completion evidence vs changed steps:** re-read task by task. C01 (6 boxes),
  C02 (4), C03 (10), C04 (10, including the new Test E, the bounded-sweep box, the
  four-outcome M2 box, the "no test strengthened" box and the post-flight diff box),
  C05 (7, including the fallback-either-way box), C06 (7, including the residual-mutation
  row). Every box maps to a step that still exists in v2. No orphan or stale criterion.
- **SPEC↔PLAN numeric agreement:** SPEC-CTRB-01's BOUNDED DOMAIN table, its Reachability
  paragraph and its STOP CONDITION match E6/E21 and my independent sweep exactly.

---

### CodeGraph / local evidence (pass 2, re-derived)

- `greedy_tensor_search_oblivious_with_ctr` — `crates/cb-train/src/tree.rs:3228`
  - Sites: `LevelKind::Ctr` `:3287-3290` (`border: *border` at `:3289`), `CtrSplitSpec`
    `:3291-3303` (`border: *border` at `:3297`), both fed by the same
    `CtrAwareSplit::Ctr { col, border }` binding at `:3277`.
  - Impact: the sole producer of persisted CTR borders (E13 re-confirmed: the other five
    `GrownTree` producers set `ctr_splits: Vec::new()`).
- `assign_leaf_of_averaging` — `crates/cb-train/src/boosting.rs:1887`, CTR arm `:1926-1938`
  - Callers: 1 (in `boosting.rs`); no covering tests.
  - Consumes `LevelKind::Ctr.border` against `CtrFeatureColumn::bins` (`Vec<u32>`,
    `crates/cb-train/src/ctr/ctr_feature.rs`, struct at `:71`). **Impact: M2 is a no-op.**
- `passes_ctr_aware` — `crates/cb-train/src/tree.rs:2589-2602`, bin-space test at `:2600`.
  Unchanged by D4. Verified byte-for-byte.
- `passes_ctr_split` — `crates/cb-model/src/apply.rs:157-189`, `ctr_value > split.border`
  at `:189`. 1 caller. Unchanged by D6.
- `Model::from_trained` / `lift_ctr` — `crates/cb-model/src/model.rs:359`, `:366-383`,
  `border: c.border` at `:376`; mixed-kind arm `LevelKind::Ctr { ctr_idx, .. }` at `:450`
  (border discarded). 31 callers. The only producer→consumer bridge.
- `.cbm` codec — `ctr_split_to_global_index` `cbm.rs:351` (bit-exact lookup `:366`,
  typed error `:368`), `build_tctr_feature` `:390` (f32 narrowing `:437`),
  decode `:601`, `build_ctr_features` `:291` (`:310-311` collect, `:321-322` sort+dedup),
  `save_cbm` `:634` (CTR tail `:649-651`), `load_cbm` `:1028`, `decode_cbm` `:1038`.
  All plan citations exact.
- `ctr_splits_for_tree` — `crates/cb-train/src/boosting.rs:1988-2021`, `border: 0.0` at
  `:2015`, called only on the `else` branch of `if has_ctr` at `:5419-5429` with
  `has_ctr = !materialized_ctr_features.is_empty()` at `:4657`; the retention comment is
  at `:5417-5418`. E9/D5 confirmed verbatim.
- `ctr_border_count_default` — `crates/cb-train/src/boosting.rs:529-531` (`15`); the only
  production call site is `:3238`; `lib.rs:95` is a re-export; all other hits are tests.
  E21 confirmed.
- **Fixture borders re-dumped this pass** (`json.features_info.ctrs[*].borders`):
  `ctr_counter_simple` `[8.999999046325684, 10.999999046325684]`;
  `tensor_ctr_e2e` `[2.9999990463256836, 7.999999046325684]`;
  `fstr_ctr` `[3.9999990463256836, 6.999999046325684, 11.999999046325684]` +
  `[3.9999990463256836]`. Every value reproduced bit-for-bit by
  `f64::from((b as f32 + 1.0) − f32::powi(2.0, −20))` for `b ∈ {2,3,6,7,8,10,11}`.
  **C04 Test A's table is correct.**
- **Harness anchors re-verified:** `tensor_ctr_e2e_oracle_test.rs` `fixture` `:55`,
  `load_cat_columns` `:68`, `tensor_ctr_params` `:86`, `train_cat(` `:232`;
  `ctr_counter_simple_oracle_test.rs` — **4** `#[test]` fns (`:131`, `:164`, `:190`,
  `:236`), `counter_params()` `:59-100`, non-degeneracy guard `:147-150`;
  `multi_permutation_fold_oracle_test.rs` `HIGH_BORDER`/`LOW_BORDER` `:107-108`, used on
  the untruncated quantizer output `:186-200` (E8 real);
  `boosting_test.rs` `CtrSplitSpec` `:337`, guard-not-Red precedent wording `:502-504`,
  E03 tests `:508/:560/:739/:792` with `spec.border == 0.0` at `:549`/`:823`;
  `mixed_kind_split_order_test.rs` `CTR_BORDER` doc `:29-33`, `CtrSplitSpec` `:47`;
  `ctr_split_scoring_test.rs` `uninformative_float_matrix` `:56`, `border` assertion
  `:122-123`, `ctr_splits.len() == 2` `:268`; `cbm_oracle_test.rs` temp path `:128`;
  `boosting_test.rs` mount at `boosting.rs:5630-5632`.
- **API existence for C05:** `CtrData::from_baked` (`crates/cb-model/src/ctr_data.rs:313`),
  `Model::with_ctr_data` (`model.rs:530`), `predict_raw_cat` (`apply.rs:409`),
  `non_symmetric_grower_roundtrip_oracle_test.rs` present. `cb-train` is a normal
  dependency of `cb-model`; `cb-oracle`/`cb-backend` are dev-deps. All 20 referenced test
  targets exist on disk.
- **C01's imports:** `calc_normalization`/`materialize_ctr_feature` at
  `crates/cb-train/src/lib.rs:48`, `ECtrType`/`CtrFeatureColumn` at `:50`,
  `ctr_border_count_default` at `:95`, `greedy_tensor_search_oblivious_with_ctr` at `:109`,
  `CtrSplitSpec`/`FeatureMatrix` at `:110`, `LevelKind` at `:111`. `cb_compute` is a normal
  dependency of `cb-train` (precedent: `ctr_split_scoring_test.rs:24,262`).
  C01 step 4's 12-argument call matches the real signature positionally.

---

### Issues (all MINOR — none blocks execution)

#### [MINOR-1] C04 tells the executor to preserve a warning comment that the v2 rewrite deleted (NEW in v2)

- **Plan location:** C04, "The extraction": *"…keeping the `*** Do NOT apply this to
  LevelKind::Ctr.border ***` warning comment in place."*
- **Evidence:** `grep -n "Do NOT apply this to" PLAN.md` → **no match**. v2 replaced
  C03's `:3297` comment wholesale; the surviving text contains no such marker.
- **Impact:** cosmetic. The executor either invents the marker or ignores the clause;
  neither changes behavior or the diff gate.
- **Required revision:** change the clause to *"keeping the C03 comment block above it
  unchanged"*, or add the `***` marker line to C03's mandated `:3297` comment so C04's
  instruction has a referent.

#### [MINOR-2] SPEC-CTRB-03's absolute wording literally forbids the M2 the plan keeps

- **Plan location:** `SPEC.md` SPEC-CTRB-03, "Consequence for the plan": *"no mutation
  check can be built around `:3289`"* — vs PLAN C04's mutation M2 and §0's Enforcement
  item 3.
- **Impact:** C06's completion criterion requires verifying SPEC↔E20/D3a consistency; an
  executor reading the absolute sentence next to M2 may raise a false STOP.
- **Required revision (assign to C06's part-3 "still outstanding" list):** narrow the
  sentence to *"No **oracle-based** expectation can be built around `:3289` — the E12
  gate and the eleven oracles stay green either way. The mutation check that CAN be
  built is PLAN C04's M2, whose detectors are the integrality assertions (C01 step 6,
  C04 Test D)."*

#### [MINOR-3] Two code blocks are pseudo-code that will not compile verbatim

- **Plan location:** (a) C04 Test E: ``assert_eq!(cb_train-side `ctr_border_count_default()`, 15, …)``;
  (b) C01 step 2: `let col = materialize_ctr_feature(...)?;` inside a `#[test] fn` returning `()`.
- **Evidence:** `ctr_border_count_default` is re-exported at `crates/cb-train/src/lib.rs:95`,
  so inside `tree_test.rs` the callable form is `crate::ctr_border_count_default()`.
  `materialize_ctr_feature` returns `CbResult<CtrFeatureColumn>`; the file-level allow list
  C01 mandates permits `.expect(...)` (precedent `ctr_split_scoring_test.rs:115`).
- **Impact:** momentary compile errors, resolved in seconds; no wrong code.
- **Required revision:** write `crate::ctr_border_count_default()` in Test E and
  `.expect("materialize ctr feature")` in C01 step 2.

#### [MINOR-4] Post-C03 references to "line 3297" are stale by construction

- **Plan location:** C04 ("Replace the line-3297 expression…"), C05 step 4 ("change the
  persisted border expression at `:3297`"), C03 completion box ("changes exactly one
  expression (line 3297)").
- **Evidence:** C03 inserts ~14 comment lines above the expression and C04 inserts a
  ~43-line helper above `greedy_tensor_search_oblivious_with_ctr`, so the expression is no
  longer at 3297 once those land.
- **Impact:** cosmetic — each instruction also names the expression by content
  (`border: *border` inside the `ctr_splits.push(CtrSplitSpec {` literal), which is unique
  in the file; a blind line-3297 edit fails to compile immediately.
- **Required revision:** append "(at `HEAD = fed705f`; the line shifts after C03/C04 —
  locate by expression, not by number)" to those three references.

#### [MINOR-5] SPEC-CTRB-03 cites the call site where it means the definition

- **Plan location:** `SPEC.md` SPEC-CTRB-03: *"`ctr_border_count_default()` is `15`
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:3238]`"*.
- **Evidence:** `:3238` is `let ctr_border_count = ctr_border_count_default();`; the
  definition returning `15` is `:529-531`. PLAN E21 states both correctly.
- **Required revision:** cite `boosting.rs:529-531` (definition) and `:3238` (sole
  production call site), matching E21.

---

### Implementation Order Review (confirmed)

1. **W0 — C01 ∥ C02.** Two new disjoint integration-test files, no production edits.
   Both genuinely Red today (C01: `bin_border == value_border`; C02: an integer border
   `x` gives `x + 2⁻²⁰` non-integral → `assert_eq!(k, k.round())` fails). **Valid.**
2. **W1 — C03.** Correctly gated on both Reds; one expression + comment-only hunks;
   diff gate plus C01's integrality assertion are the entire (and, per E20, the only
   possible) safety net. **Valid.**
3. **W2 — C04.** Behavior-preserving extraction; exclusive holder of `tree.rs`;
   pre-flight and post-flight diff gates. M1 and M2 run in sequence with manual reverts.
   **Valid.**
4. **W3 — C05.** Now correctly after C04 (its mutation names C04's helper). Guard-not-Red
   is declared, falsifiability supplied by the `+0.1` mutation, fallback keeps
   SPEC-CTRB-05 gated even if `save_cbm` rejects the shape. **Valid.**
5. **W4 — C06.** Documentation + confirmation runs, including the residual-mutation diff
   row that only means anything because C04/C05 are serialized. **Valid.**

No cycles. No intermediate state leaves the tree unbuildable: W0 adds test targets only;
C03 is a one-expression change; C04's extraction is behavior-preserving; C05 permanently
adds only a test file.

---

### Potential Bugs (adversarial pass, all covered or unreachable)

- **The 2⁻²⁰ sliver.** A document whose apply-space value lands in
  `(b+1 − 2⁻²⁰, b+1)` would have `bin = b` (training false) but `v > border` (apply true)
  — the defect surviving in miniature. **Unreachable for C01's mandated corpus:** with
  `prior = 0.5/1.0`, `v = 15·(2·good+1) / (2·(total+1))`, a rational whose denominator is
  ≤ ~50 for a 12–24 row corpus, so its distance to any integer is either 0 or ≥ 0.02 ≫ 9.5e-7.
  It is also upstream's own convention, so it is parity-faithful, not a Rust-side defect.
  No plan change required.
- **Clamping vs the step-3 premise.** `materialize_ctr_feature` clamps `trunc(bin_f)` into
  `[0, 15]`; a clamped document would break `v[i].trunc() as u32 == col.bins[i]`. For
  Borders CTR with `prior = 0.5`, `ctr < 1` strictly, so `v < 15` and no clamp fires.
  C01's step-3 assertion would surface any violation as a named failure anyway. Covered.
- **C02 Test 2 (upstream border membership) failing after C03.** Handled explicitly by
  OQ-B as a STOP-AND-REPORT structural parity finding. Probability low: SPEC §10 records
  that the fix was applied experimentally this session and the E12 gate passed at ≤1e-5,
  which requires the same chosen bin thresholds.
- **`ctr_split_scoring_test.rs:122-123`** (`assert!(border >= 0.0 && border < 10.0)`):
  re-confirmed safe — first-wins at `tree.rs:3175-3182` picks `border_idx = 0`, so the
  persisted value moves `0.0 → 0.9999990463256836`, still inside the assertion.
- **Un-reverted mutation in `tree.rs`:** now triple-netted (C04 post-flight, C05
  pre/post-flight, C06's dedicated diff row) with `git checkout --`/`stash`/`clean`
  forbidden throughout.

---

### Verification-strategy coverage

| Spec | Falsifiable gate | Verified adequate? |
|---|---|---|
| SPEC-CTRB-01 | C02 Tests 1–2 (Red today), C04 Test A (bit-exact fixture pins), Test E + M1 (f32-vs-f64 discriminator) | Yes — Test A's seven values reproduced this pass |
| SPEC-CTRB-02 | C01 Tests 1–2 (Red today, names doc/bin/both borders/both booleans) | Yes — non-tautological; C01 reimplements both decisions from `tree.rs:2600` / `apply.rs:189` verbatim, so an apply-side "fix" cannot green it |
| SPEC-CTRB-03 | C01 step 6 + C04 Test D, falsified by M2 | Yes — and correctly *not* claimed to be gated by any oracle |
| SPEC-CTRB-04 | C06 decision record + `boosting::tests` green with zero diff | Yes — E03's four tests call `ctr_splits_for_tree` directly with a non-empty list and are unaffected by D5 |
| SPEC-CTRB-05 | C05 bitwise round-trip, falsified by `+0.1`; fallback gate if `save_cbm` rejects | Yes — mutation verified to break f32 narrowing |

§3.2's command blocks are **faithful** to ctr-type `PLAN.md` §3.2: the nine cb-train CTR
oracle targets, the two cb-model ones, the three one-hot targets, the `.cbm`/serde block
and the clippy line are reproduced verbatim and in the same order. BUG-CTRB tightens the
diff gate from ctr-type's "four may be mechanically edited" to "all eleven zero-diff",
which is correct here because no task in this plan edits any of them, and adds
`ctr_nonmean_byte_identity_test` plus the two `--no-fail-fast` sweeps. The §3.1 protocol
is reproduced with §1's stricter manual-revert requirement.

---

### Required Plan Revisions (all MINOR; none gates execution)

1. Fix C04's dangling reference to the deleted `*** Do NOT apply this to
   LevelKind::Ctr.border ***` marker (MINOR-1).
2. Narrow SPEC-CTRB-03's *"no mutation check can be built around `:3289`"* to
   *"no oracle-based expectation…"*, and add it to C06's part-3 outstanding list (MINOR-2).
3. Make C04 Test E's guard assertion and C01 step 2 compilable
   (`crate::ctr_border_count_default()`, `.expect(...)`) (MINOR-3).
4. Qualify the post-C03 `:3297` line references as "at HEAD; locate by expression" (MINOR-4).
5. Correct SPEC-CTRB-03's `boosting.rs:3238` citation to `:529-531` (+ `:3238` as the call
   site) (MINOR-5).

---

### Unverified Items

- **Upstream's exact epsilon above `b = 15`.** Still `[UNVERIFIED]`, correctly, and
  correctly labelled as such in E6/OQ-D/Test E. Unreachable at
  `ctr_border_count_default() == 15`; `catboost-master/` is prohibited (D9) and no fixture
  reaches that range. Not blocking.
- **`save_cbm` on a `CtrData::from_baked` CTR model.** No in-repo precedent (E23).
  Now covered either way by C05's fallback gate.
- **No test suite was executed by this check** (no build run). The eleven oracles, the
  three one-hot targets and the E12 gate are asserted from the plan's recorded run plus
  the reproduced failure text in §1; the plan mandates running them at C03, C04 and C06.
