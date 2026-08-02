---
title: Plan check — BUG-BTMV (BTMV target-border divisor)
verdict: ISSUES_FOUND
pass: 2
checked_at: 2026-08-02
plan: .planning/plans/btmv-target-border-divisor/PLAN.md (v1 + pass-1 fixes, 1595 lines, 7 tasks B01–B07)
spec: .planning/plans/btmv-target-border-divisor/SPEC.md (spec_version 1, unchanged)
head: c21f44a
---

## Plan Check Result

**Verdict:** ISSUES_FOUND — **2 residual MAJOR, 3 MINOR. No CRITICAL.**
**Goal:** Make the whole-set CTR bake pass `targetClassesCount - 1` as
`accumulate_online`'s `target_border_count`, so every baked
`BinarizedTargetMeanValue` `Sum` stops being halved, without moving any non-mean payload.
**Plan:** `.planning/plans/btmv-target-border-divisor/PLAN.md`

### Summary

- **All four applied fixes are substantively CORRECT.** Every decision the coordinator
  made is the right one, and the D3 arithmetic and the B01 Test 2 comment are now
  accurate. Nothing cleared in pass 1 was disturbed.
- **But two of the four are incompletely propagated.** Each MAJOR fix landed in one place
  and left its contradiction standing in another — **four stale lines in total**:
  - MAJOR-1 (serialize B06 → B02): the §5 **wave diagram** and B02's **header** are fixed,
    but the §5 **dependency-edge block** (`:416-420`), the §5 **"parallel by FILE"** bullet
    (`:425-427`) and B02's own prose (`:655`, *"keeps B02 parallel with B01/B06"*) still say
    the opposite. B02's `Order:` field also still precedes B06's.
  - MAJOR-2 (scope B06's command): the new callout is at B06 **step 1** (`:898-904`), but
    the unsatisfiable gate pass 1 actually named — B06 **step 5**, `:927`,
    `cargo test -p cb-train --lib ctr::` *(all ctr test modules) — green* — is untouched,
    so the task now contradicts itself.
- **The D3 rewrite is arithmetically correct** and consistent with B01 Test 3, B04 Test 2
  and B06 Test 3 — but the two **production comments derived from it** still carry the
  blanket `classes <= 1` claim D3 just retracted (`:998-999`, `:1169-1170`).
- **No new defect was introduced** beyond those propagation gaps. Wave numbering, task IDs,
  completion criteria, mutation outcomes and every §3.2 command remain internally consistent.
- **The six unapplied MINORs are all still genuinely non-blocking**; none should have been
  applied as a blocker.
- **All five remaining items are mechanical text edits**, specified verbatim below. Applying
  them needs no re-verification of any claim — **a pass 3 is not warranted.**

---

### Pass-1 disposition table

| # | Pass-1 finding | Sev | Disposition at pass 2 |
|---|---|---|---|
| 1 | B06's live `online.rs:321` mutation can corrupt B02's Red; hand-off gate covered only B03 | MAJOR | **DECISION FIXED, PROPAGATION INCOMPLETE.** §5 diagram `:389-402` now reads `B01 ∥ B06` + `B02 SERIALIZED AFTER B06` with a correct, well-argued rationale; B02's header `:639` reads `Depends on: **B06**`. **Stale:** §5 edge block `:416-420`, §5 bullet `:425-427`, B02 prose `:655`, B02 `Order: 2` vs B06 `Order: 3`. → **RESIDUAL-1** |
| 2 | B06's `cargo test … --lib ctr::` gate unsatisfiable once B01's Reds land | MAJOR | **DECISION FIXED, WRONG LINE PATCHED.** Correct callout added at step 1 `:898-904` (names `ctr/mod.rs:49-50`, forbids weakening B01's Reds). **Stale:** step 5 `:927` still mandates the wide sweep. → **RESIDUAL-2** |
| 3 | D3's "`classes <= 1` byte-identical" false at `classes == 0` | MINOR | **FIXED in D3 (`:220`) — arithmetic verified correct.** Not propagated to the two mandated source comments (`:998-999`, `:1169-1170`). → **MINOR-A** |
| 4 | B01 Test 2's comment contradicted its own table | MINOR | **FIXED (`:534-552`).** Now states the FIX and HELPER both yield 1 at `classes == 2` while the BUG yields 2, and that `classes = 3` is still needed to separate fix from helper — matches the table exactly. One duplicated word. → **MINOR-C** |
| 5 | B02's `ours.len() == 1` / projection guards are `[INFERRED]`, failure routed to STOP | MINOR | Not applied. Still non-blocking (fails safe, and the bake is post-training so table composition is fix-invariant). |
| 6 | Upstream line drift (`:917`→`:918`, `:930-934`→`:927-930`, `ctr_helper.h:34-42`→`:35-42`) | MINOR | Not applied. Non-blocking; the quoted code is verbatim correct and `ctr_helper.h:34-42` already matches the pre-existing doc at `ctr/mod.rs:109`. |
| 7 | `private_intra_doc_links` from a public doc to the private helper | MINOR | Not applied. Non-blocking — `cargo clippy --all-targets` does not run rustdoc. |
| 8 | B07's "three production/test files" miscount | MINOR | Not applied. Cosmetic. |
| 9 | B01 Tests 2/3 don't name `proj` / `tc` | MINOR | Not applied. Non-blocking; `e11_fixture` (`final_ctr_test.rs:115-122`) is the obvious template. |
| 10 | `const fn` + `usize::max` | MINOR | Already handled inline at `:1183-1185`; no change needed. |

---

### Issues

#### [RESIDUAL-1 · MAJOR] The B06 → B02 serialization is contradicted in three places, and B02's `Order` still precedes B06's

- **Plan location:** §5 `:416-420` (dependency edges), §5 `:425-427` (parallelism bullet),
  B02 `:655`, B02 header `:639` vs B06 header `:805`.
- **Evidence (verbatim, current file):**
  - `:416-420` — `B01 ─┐ / B02 ─┼─> B03 ──> B04 ──> B05 ──> B07 / B06 ─┘` and
    *"Longest path: (B01|B02|B06) -> B03 -> B04 -> B05 -> B07 (5 waves)"*. The new
    `B06 → B02` edge is absent and B02 is still drawn in the parallel fan-in.
  - `:425-427` — *"**W0 is genuinely parallel by FILE.** B01 edits `src/ctr/final_ctr_test.rs`;
    B02 creates a brand-new integration target; B06 edits `src/ctr/online_test.rs`.
    No two of them write the same file."* This is the exact sentence pass 1 asked to be
    restated, and it still asserts B02 ∥ B06 on the very ground (file disjointness) that
    `:401-402` now says is insufficient.
  - `:655` — *"A new target also keeps B02 parallel with B01/B06."*
  - `:639` `Order: 2 · Depends on: **B06**` vs `:805` `Order: 3` — B02 is numbered ahead of
    the task it now depends on.
- **Failure scenario:** an executor (or a parallel dispatcher) reading `:416-420`/`:425-427`
  starts B02 while M-B06 is applied at `online.rs:321`. `[VERIFIED pass 1: CODEGRAPH
  online_mean_prefix → sole production caller crates/cb-train/src/ctr/ctr_feature.rs:234,
  the BTMV arm of materialize_ctr_feature, on the training path B02 drives through
  train_cat.]` B02's structural guards then fail against an in-flight mutation, which the
  plan routes to STOP-AND-REPORT.
- **Impact:** a false structural-divergence alarm that halts W0 — precisely the hazard the
  §5 diagram was edited to prevent. (Fails safe: no false green.)
- **Required revision (3 edits + 1 renumber):**
  1. `:416-420` → `B01 ─┐` / `B06 ──> B02 ─┼─> B03 ──> B04 ──> B05 ──> B07`, and
     *"Longest path: B06 -> B02 -> B03 -> B04 -> B05 -> B07 (6 tasks)"*.
  2. `:425-427` → *"**W0 is parallel only for B01 ∥ B06.** They are disjoint by file **and**
     neither touches the other's production path — B06's mutation reaches
     `online_mean_prefix`, which `final_ctr_test`'s bake-only tests never call. **B02 is NOT
     parallel with B06** (see the diagram): file disjointness is not the criterion; a live
     production edit is."*
  3. `:655` → *"A new target also keeps B02 independent of B01's file."*
  4. Renumber: B06 `Order: 2`, B02 `Order: 3` (B03–B07 unchanged).

#### [RESIDUAL-2 · MAJOR] B06 step 5 still mandates the wide `ctr::` sweep the new step-1 callout forbids

- **Plan location:** B06 TDD step 5, `:927`: `- ` + `cargo test -p cb-train --lib ctr::` +
  ` (all ctr test modules) — green.` — against the new callout at `:899-904`: *"Scope this
  command to `ctr::online_test` and NOTHING wider … That criterion would be unsatisfiable,
  and the tempting 'fix' is to weaken B01's Reds — which is FORBIDDEN."*
- **Evidence:** `crates/cb-train/src/ctr/final_ctr_test.rs` and `online_test.rs` are mounted
  into the **same `--lib` target** (`crates/cb-train/src/ctr/mod.rs:49-50` and `:43-44`)
  `[VERIFIED: CODEGRAPH ctr/mod.rs verbatim]`. §5 still declares `B01 ∥ B06`, so B01's two
  deliberate Reds may already be present when B06 reaches step 5.
- **Failure scenario:** B06 cannot close: step 5 fails with B01's two expected failures while
  step 1 forbids widening the filter to see them. The escalation path is to weaken B01.
- **Impact:** execution-blocking self-contradiction inside one task; the exact trap pass 1
  named, moved rather than removed.
- **Required revision (1 line):** `:927` → ``- `cargo test -p cb-train --lib ctr::online_test`
  — green (see step 1: do NOT widen to `ctr::`; B01's two Reds in `ctr::final_ctr_test` are
  expected and are not B06's to fix).`` Optionally add `ctr::calc_ctr_test ctr::mod_test`,
  which are unaffected by B01; `ctr::final_ctr_test` must stay out until B03.
- **Cross-check performed:** `grep -n "\-\-lib ctr::"` shows `:927` is the **only** remaining
  unscoped sweep. B01 (`:587`), B03 (`:1027`), B04 (`:1255`, `:1321`), B05 (`:1419`) and
  §3.2 (`:302-305`) all use the narrow filters. **No other task depends on a wide sweep.** ✔

#### [MINOR-A] D3's correction did not propagate to the two comments it is mandated into

- **Plan location:** B03's production comment `:998-999`: *"Behavior at `classes <= 1` is
  unchanged either way: every target_class is 0, so every Sum is 0."*; B04's helper doc
  `:1169-1170`: *"Behavior at `target_classes_count <= 1` is identical either way (every
  `target_class` is 0, so every `Sum` is 0)."*
- **Evidence:** D3 (`:220`) now correctly states that `classes == 0` moves the other way
  (`Err(Degenerate)` → `Ok`). Both comments are mandated **verbatim into committed source**,
  so the plan would ship the retracted claim into `bake.rs` and `ctr/mod.rs`.
- **Required revision:** in both, replace `<= 1` with `== 1` and append: *"At
  `target_classes_count == 0` the floor changes the outcome — today the zero-divisor
  rejection returns `CbError::Degenerate`, with the floor the bake returns `Ok` on a
  degenerate table. Unreachable: the sole production caller hard-codes 2
  (`boosting.rs:5582`)."*

#### [MINOR-B] B02 has no `online.rs` empty-diff pre-flight (belt-and-braces only)

- With B06 serialized ahead of it, B06's own completion criteria (`:938-939`) already require
  `git diff crates/cb-train/src/ctr/online.rs` to be EMPTY, so B02 is protected by
  construction. Adding the same one-line pre-flight to B02 (mirroring B03's `:966-967`) would
  make the protection local and self-evident. **Not required.**

#### [MINOR-C] Duplicated word in B01 Test 2's corrected comment

- `:537-538`: *"It exists because at classes == 2 the / the FIX and the HELPER are
  indistinguishable"*. The content is correct and matches the table; drop one `the`.

---

### Confirmation of the four applied fixes

- **§5 diagram `:389-402`** — correct and well-argued: names M-B06 as a *live* edit to
  `online.rs:321`, names `ctr_feature.rs:234` as the training-path consumer, states that
  file-level disjointness is not the criterion, and predicts the exact failure mode
  (structural guards → false STOP). Matches my pass-1 evidence exactly.
- **B02 header `:639`** — `Depends on: **B06**` with the reason inline. ✔
- **B06 step-1 callout `:898-904`** — correct, cites `ctr/mod.rs:49-50`, and explicitly
  forbids the weaken-B01 escalation. ✔ (Wrong line patched — RESIDUAL-2.)
- **D3 `:220`** — arithmetically correct and consistent with everything that exercises the
  floor:
  - `classes == 1` byte-identical: today `divisor = classes = 1`; after the fix
    `1.saturating_sub(1).max(1) = 1`. ✔ → **B01 Test 3** (`:571-582`) passes before and after. ✔
  - `classes == 0`: today the 5th argument is `0` → `accumulate_online` rejects at
    `online.rs:176-180`; after the fix it is `1` → `Ok`. ✔ Unreachable: all eight
    `bake_ctr_table` call sites pass `2` `[VERIFIED pass 1: LOCAL grep]`. ✔
  - **B04 Test 2** (`:1224-1225`, `(1 → 1)`, `(0 → 1)`) pins the *helper's* value, not bake
    behavior — **no contradiction**. ✔
  - **B06 Test 3** (`:891-894`) exercises `online_mean_prefix`, which has **no** zero-divisor
    rejection at all (`crates/cb-train/src/ctr/online.rs:298-356`), so `classes ∈ {0,1}` are
    `Ok` there both before and after and B06 changes no production code — **no contradiction**. ✔
  - **B04 Test 3** operates only at `classes = 3` — unaffected. ✔
- **B01 Test 2 comment `:534-552`** — now matches its own table row-for-row
  (bug 2/3, fix 1/2, helper 1/1) and correctly identifies `classes = 3` as required only to
  separate the fix from the helper. ✔

### Did the edits introduce anything new?

Checked and clean, apart from the propagation gaps above:

- **Wave numbering / task IDs / spec tags:** W0–W4 and B01–B07 unchanged and consistent;
  §6's coverage table and §7's OQ list still name the same tasks and gates.
- **Completion criteria vs changed steps:** B06's six boxes (`:935-941`) all still map to
  existing steps; none references the wide sweep, so RESIDUAL-2 is confined to `:927`.
  B02's criteria (`:779-788`) are unaffected by the new dependency.
- **B03's pre-flight** (`:966-967`, `git diff online.rs` EMPTY) is unchanged and still
  correct — now redundant with B06→B02 but harmless.
- **Mutation outcome sets** (M-B06's four, M1's six, M2's five, M-B05's four) are untouched
  and still correct — re-checked M-B06 outcome 4 against `online_test.rs:416-444` (expects
  `[0,1,1,2]` at `classes = 2`; under `classes.max(1)` it fails) and outcome 3 against
  Test 3's floor. ✔
- **§3.2 command block** unchanged and still faithful; the three narrow `--lib` filters
  (`:302-305`) already match the scoping decision.
- **No new claim about upstream, the fixtures, or any symbol was introduced**, so nothing
  cleared in pass 1 needed re-verification.

### Unapplied pass-1 MINORs — still non-blocking

All six remain correctly non-blocking; **none should have been applied as a blocker**:
the B02 structural guards fail safe and the bake's table composition is fix-invariant
(the bake runs post-training over `tree.ctr_splits`, `boosting.rs:5560-5588`); the upstream
line drift never reaches an assertion and `ctr_helper.h:34-42` already matches the
pre-existing doc at `ctr/mod.rs:109`; `private_intra_doc_links` is invisible to
`cargo clippy --all-targets`; B07's file count and the unnamed `proj`/`tc` are cosmetic;
the `const fn` fallback is already written into the task. If any one were picked up
opportunistically, MINOR-A's sibling (the `ctr_helper.h` anchor) is the only other one that
reaches committed source, and consistency with the existing doc argues for leaving it.

---

### Implementation Order Review (post-fix)

`B06 → B02`, `B01` ∥ both, then `B03 → B04 → B05 → B07`. Acyclic; longest path 6 tasks.
Exclusive resources: `online.rs`/`online_test.rs` (B06, released with a verified empty diff),
`final_ctr_test.rs` (B01 → B05), `bake.rs` (B03 → B04 → B05) with pre-flight and post-flight
diff gates. `git checkout --` / `git stash` / `git clean` remain forbidden in §1, §3.2 and
every mutation revert; the three protected untracked paths still match `git status --short`
at `c21f44a`. **Once RESIDUAL-1 and RESIDUAL-2 are applied, the order is valid and every
task's prerequisites are available when it begins.**

### Unverified Items (unchanged from pass 1)

- **OQ-A** — that B03 alone turns the E13 gate green remains `[INFERRED]`; correctly gated by
  B03's STOP-AND-REPORT branch, which forbids weakening the gate, touching the fixture or
  regenerating anything.
- **B02's `ours.len() == 1` / `projection == [1]`** at HEAD — inferred, fails safe.
- **OQ-B** — the `u64::MAX` sentinel is confirmed in both committed CTR fixtures with the
  stated payloads and totals, not proven in general; B02's two structural guards catch a change.
- **No test suite was executed in pass 2.** Pass 1 executed
  `cargo test -p cb-train --test ctr_btmv_simple_oracle_test` and reproduced §1.1 verbatim.
