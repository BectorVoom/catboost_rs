# Plan Check — Device CTR Coverage P1

**Verdict (pass 1): ISSUES_FOUND**
**Checker:** `plan-checker` agent (independent). Review returned inline because the
agent's `Write` tool was disabled; transcribed here by the orchestrator without
softening. Findings independently re-verified where marked.
**Reviewed at:** worktree `…/gpu-borders-shared-sample`, branch
`worktree-gpu-borders-shared-sample`, HEAD `a0a67ec`, clean.
**Artifacts:** `SPEC.md` (DCTR-01…19), `PLAN.md` (23 tasks, 7 waves), `research.md`.

## Summary

Strong, well-evidenced plan. Its central hazard — the R-8 false-pass class — is
genuinely closed: every e2e asserts `CountingGpu.grown.get() == params.iterations`
plus a `Σ ctr_splits >= 1` vacuity guard. The checker independently verified a
safety property the plan relies on but never states: a cb-train/cb-backend gate
mismatch cannot silently drop CTR columns, because
`crates/cb-backend/src/gpu_runtime/session.rs:1498-1505` returns `Ok(None)` for any
`ctr.is_some()` config that `ctr_covered` rejects — the whole fit declines and
`grown == 0` fires loudly.

**One CRITICAL blocks approval**, plus 3 MAJOR and 13 MINOR.

## CRITICAL-1 — the gate already has five covering assertions that four tasks silently break

The plan's load-bearing CodeGraph claim that `ctr_types_are_device_covered` has
"⚠️ no covering tests found" is **false**.

**Independently re-verified by the orchestrator** (`grep -rn
"ctr_types_are_device_covered" crates/ --include=*.rs`, plus reading
`crates/cb-train/src/device_ctr_combo_config_test.rs:115-173`):

- `:21` — `use super::{build_device_ctr_config, ctr_types_are_device_covered};`
- `:132` — combination projection **must decline**
- `:144` — all-simple Borders **must remain covered**
- `:156` — `ctr_type = Counter` **must decline**
- `:163` — `prior_denom = 2.0` **must decline**
- `:170` — `target_border_idx = 1` **must decline**

These pin **exactly the four conjuncts P1 removes**. The file's own doc at
`:124-126` states: *"This test pins the CLOSED gate deliberately: re-opening it
must be a conscious act accompanied by a passing `device_ctr_combo_fit_test`
(currently `#[ignore]`d), not an accident."* Mounted at `boosting.rs:7406-7408`,
so it runs under `cargo test -p cb-train --lib`.

**Root cause of the miss:** CodeGraph's test attribution is unreliable for
`#[path]`-mounted in-crate sibling modules. The plan treated one such "⚠️" as
verified fact.

**Failure scenario:** T01 deletes `&& col.prior_denom == 1.0` ⇒ `:163` fails, and
T01's Validation block omits that binary, so T01 is marked done with
`cargo test -p cb-train --lib` red. T10 (`:170`), T12 (`:156`) and T19 (`:132`)
each break another assertion the same way; the failure first surfaces at whichever
task next runs a workspace validation, attributed to the wrong task.

**Impact:** four gate-chain tasks ship a red in-crate suite with no owner; the
phase's own "conscious act" guardrail is defeated by accident; T22's completion
criterion is unachievable because the gap never existed; and V-0 — the plan's
justification for trusting CodeGraph — is falsified on its most load-bearing symbol.

**Required revision:** correct §1 V-0; add a Wave-0 task **T00** before T01 that
consciously retires/rewrites those two test functions with a written justification
citing `:124-126`; add `cargo test -p cb-train --lib device_ctr_combo_config_tests`
to T01/T10/T12/T16/T19/T22; rewrite T22's completion evidence.

## MAJOR-1 — combination × {Buckets, Counter, BTMV} admitted with zero coverage

After T22 the predicate admits any CPU-supported type on any-arity projection, but
coverage delivered is: simple Borders/Buckets/Counter/BTMV, and combination
**Borders only** (`ctr_device_combo/gen_fixtures.py:87`). No task exercises ≥2
members with a non-Borders type. A composition defect — e.g. the `bucket_counts`
fallback at `session.rs:1749-1758` reading `member_bins.first()` only, whose
comment still reads *"the gate admits only simple projections"* — yields a wrong
tree with no failing test.

**Required revision:** preferred = a device-vs-CPU-grower split-sequence
differential task over `ctr_device_combo` for `combinations_ctr ∈ {Buckets,
Counter, BTMV}` (no upstream fixture needed, strictly stronger than
≤1e-5-vs-upstream, and it also gives R-20 a real detector). Also fix the false
comment.

## MAJOR-2 — T09's `ctr_covered` unit test cannot compile as specified

The plan says to mount next to "`session_depth_gt1_test.rs`'s existing mount in
`gpu_runtime/session.rs`", but that grep returns nothing — the mount is
`crates/cb-backend/src/gpu_runtime/mod.rs:774-775`. And `ctr_covered` is private
to `mod session` (`session.rs:134`), so a module mounted in `mod.rs` is a sibling
and cannot name it ⇒ `E0603`. Risk: implementer field-fixes by widening visibility.

**Required revision:** mount inside `session.rs`; keep `ctr_covered` private; fix
the grep hint.

## MAJOR-4 — T05's fixture recipe can deadlock

T05 pins `CARDS=(6,)`, `N_ROWS=64`, keeps `iterations=5` as "load-bearing", yet
requires both `target_border_idx ∈ {0,1}` in `model.json`. The only in-repo
precedent achieving that guard (`ctr_buckets_simple/`) used two cat columns, 10
iterations, 60 rows. The escalation rule allows only widening seeds/cardinality, so
pressure falls on weakening the anti-false-pass guard — the only thing exercising
T08's `Buckets@1` numerator.

**Required revision:** ordered ladder (widen seeds → raise cardinality → add a
second cat column → raise iterations); mark only the genuinely gate-load-bearing
params as unchangeable.

## Judgements requested

- **§2.5 mutation protocol: real verification, not ceremony.** T15 is an unlisted
  eighth candidate; its `≥2 distinct bins` guard is a reasonable substitute but
  should say so.
- **R-20 / D-2 unit-test-only: adequate but improvable** — the unit test proves the
  helper filters, not that it is wired into pass C. MAJOR-1 option (b) fixes both.
- **DCTR-12's substitution pins the width via detector B, not A.** Checker measured
  22–41 mismatches per 96 documents at `divisor = 3` — the numeric detector is real
  and discriminating. Detector A pins the output buffer, not the bucket accumulator;
  the "`n*8` bytes" claim rests on an unstated assumption.
- **Track E ordering enforced; no-op proof genuine.**
- **DCTR-03's retention is right.**
- **Command correctness: clean** — no bare `--features rocm` anywhere.
- **Wrong Red predictions: only two** — T09 (cannot reach Red) and T01 (Red is
  incomplete; also turns `:163` red, unpredicted).
- **P2/P3 foreclosure: none.** T21's and T13's boundary pins must be *inverted* by
  P2/P3 — annotate them.

## Remaining MINOR findings (all carried into the revision)

MINOR-1 `ctr_covered` has two callers (`:1471`, `:1701`) — record that `:1471` is
the safety property. MINOR-2 T05's accessor is top-level `c["target_border_idx"]`.
MINOR-3 reframe T14 detector A. MINOR-4 8 literal sites, only 5 need edits.
MINOR-5 line drift: pass C loop is `mod.rs:4907` not `:4919`; `#[ignore]` at
`device_ctr_combo_fit_test.rs:179`; `ordered_ctr_prefix_kernel` at
`ctr_device.rs:135`. MINOR-6 T17 must list `session.rs:2593`. MINOR-7
`one_hot_x_ctr_still_declines` already exists at
`crates/cb-train/tests/device_fpp_composition_test.rs:255`. MINOR-8 reconcile
SPEC §6 scenario 7 (23/23) with the plan's 27/27 DoD. MINOR-9 serialise T20 after
T01; forbid live mutations during workspace validations. MINOR-10 fix T18/T19 wave
labels and the T05/T06/T07 parallelism contradiction. MINOR-11 transcribe T17/T18
predicate cases from `tree_test.rs`. MINOR-12 four bug mitigations (bucket_counts
fallback; `projection_members` id space undocumented; T04's dead `norm == 0.0`
guard; T18 must not drop `phantom_max` — `mod.rs:4906`, CPU counterpart applies
phantom *outside* the filter). MINOR-13 annotate P2/P3 boundary inversions.

## Unverified by the checker

The FPP-11 three-arm spike and the Q2 BTMV spike (both reverted, not reproducible
at `a0a67ec`); `run_device_tests.sh` = 23/23 (needs the rig); whether a seed in
`range(24)` satisfies T05's guard; CubeCL authorability of the three new kernels;
whether the device leaf-value gather uses the host `assign_leaf_over_ctr_columns`
path or the SPD-03 device `leaf_of` consumption for the new types (worth one
sentence in T10).

---

## Pass 2 — verdict ISSUES_FOUND (2 MAJOR + 9 MINOR, all text-level; no BLOCKER, no CRITICAL)

**All four pass-1 blocking findings RESOLVED**, each remedy re-verified against the
worktree. Command discipline and the `CountingGpu` false-pass discipline survived intact
(`grep -n -- "--features rocm" PLAN.md | grep -v no-default-features` → **0**). Nothing
pass 1 validated was damaged.

### V-0c — verified CORRECT, and it strengthens the CRITICAL-1 remedy

The planner's correction (which pass 1 missed) is right. `covered_column(...)` builds
Borders/`b=0`/`denom=1.0` over the caller's projection, and all four negative assertions
pass `from_features(&[0,1])`, so the **arity** conjunct alone rejects them:

| assertion | rejected today by | actually flips at |
|---|---|---|
| `:132` combo set | arity only | **T19** |
| `:144` all-simple → expect covered | none (positive pin) | never |
| `:156` `ctr_type=Counter` | arity **+** type | **T19** |
| `:163` `prior_denom=2.0` | arity **+** prior_denom | **T19** |
| `:170` `target_border_idx=1` | arity **+** target_border | **T19** |

Pass 1's failure schedule (T01 breaks `:163`, T10 breaks `:170`, T12 breaks `:156`) was
**wrong** — nothing goes red until T19, where all four flip at once. The hazard is worse
than pass 1 described: an implementer told to expect red at T01 would find green and
reasonably skip the migration. T00 is therefore *more* necessary, as claimed.

### Two NEW MAJORs (introduced by the revision itself)

1. **T12/T16 omit the mandatory gate-state row flip; T19 misstates which rows it flips.**
   T00's contract requires the named task to flip its row *in the same commit* as its
   conjunct deletion. T01 and T10 do; T12 (row 3) and T16 (row 5) do not, and T19's
   validation comment claimed "flips rows 2/3/4/6" when under the new simple-projection
   table it flips **row 2 only**.
2. **T22 has no disposition for a failing vacuity guard 4** (no ≥2-member CTR split chosen
   for a type — a benign fixture outcome, not a defect). T22 gates T23 gates the phase DoD,
   and the only in-plan action would be weakening the guard that prevents a vacuous pass.

### Judgements

T22 **can** discriminate what it claims (compares `splits`/`ctr_splits`/`one_hot_splits`/
`level_kinds` per tree on the full `CtrSplitSpec` identity; CPU arm asserts `grown == 0`,
closing the device-vs-device tautology). T00 preserves the guardrail's intent — it chose
*rewrite over retire*, correctly, since retiring leaves the gate unpinned across five
conjunct edits. T10's leaf-gather ambiguity is the highest residual risk and is correctly
dispositioned as "escalate, do not patch inside an e2e task".

---

## Revision round 2 — all 11 required revisions applied by the orchestrator

Applied directly (each was a precise text fix; the two MAJORs first):

1. **[MAJOR]** Added the row-flip instruction to T12 (**row 3**) and T16 (**row 5**), each
   "in the same commit as the gate widening" with the failure mode spelled out; corrected
   T19's validation comment to "flips row 2 (arity) ONLY — rows 3/4/6 were already
   discharged by T12/T10/T01".
2. **[MAJOR]** Added a mandatory **guard-4 escalation ladder** to T22 (raise `iterations`
   → raise `depth` → sweep priors → record the type as structurally unreachable / escalate),
   plus the explicit statement that params are tunable here **because T22 is a self-oracle,
   not an upstream comparison** (R-12 freezes `X`/`y`/`borders.npy`, not the params), and an
   absolute prohibition on weakening or deleting assertion 4.
3. MINOR-A — softened V-0c's "each row tests exactly one conjunct" and documented row 4's
   two-attribute exception, with the two optional isolating rows named.
4. MINOR-B — fixed T05's and T06's copy-paste-wrong Red `Run:` binaries (T05 → buckets,
   T06 → counter).
5. MINOR-C — T02 Green/Rollback now say **5 edited literals of 8 sites**, matching C-12.
6. MINOR-D — T05 rung 3 corrected: `ctr_buckets_simple` is **cat-only**, which
   `N_FLOAT >= 1` forbids, so the guard is *strictly harder* here; expect rung 4.
7. MINOR-E — T17 disk-evidence `mod.rs:4919` → `:4907`. (The one surviving `:4919` string
   is the C-13 correction table recording the fix — correct as-is.)
8. MINOR-F — T24 Rollback "four" → "five appended lines".
9. MINOR-G — removed stale old-T22 cross-refs in T13 and T21; T22 owns
   `device_ctr_combo_types_diff_test.rs`, not `boosting_ctr_gate_test.rs`.
10. MINOR-H — SPEC DCTR-20 citations corrected against disk: false comment at
    `session.rs:1748`, fallback `:1749-1765`.
11. MINOR-I — recorded the §2.5 isolation rule for the `{T21, T22}` parallel set (both
    mutate production and both run `run_device_tests.sh`; file disjointness is not
    sufficient — the same reasoning that serialised T20 after T01).

**Post-revision verification**: 0 bare `--features rocm`; all stale strings gone; the three
fixture smoke-test names are internally consistent across File / Run / Validation / DoD.

---

## Pass 3 — **VERDICT: PASS**

Scoped verification of the eleven revisions. **All eleven RESOLVED**, each with a
line-number citation; both MAJORs confirmed properly closed:

- **MAJOR-1**: T00's table (`:732-737`) is matched exactly by T01 (row 6), T10 (row 4),
  T12 (row 3, `:1813-1819`), T16 (row 5, `:2128-2130`) and T19's corrected comment
  (`:2464`). Row numbering verified consistent across all four instructions.
- **MAJOR-2**: T22's ladder (`:2721-2745`) is coherent and causally ordered; the
  self-oracle / params-tunable distinction at `:2727-2731` explicitly pre-empts the wrong
  inference ("do not assume R-12 freezes the params too"); the prohibition on weakening
  assertion 4 is absolute and reasoned; guard-4 is correctly scoped as the only failure
  routed to the ladder, with defect failures still going to their owning task.

**Regression checks all clean**: zero bare `--features rocm`; the three fixture smoke-test
names consistent across File / Run / Validation / DoD; no task body contradicts §5.3; the
`CountingGpu` discipline untouched (six e2e assertions plus the two intentional `== 0`
decline pins). **No unverified items.**

Checker correction to the orchestrator's brief: `:4919` survives at four sites, not one —
`:383`, `:2240`, `:3085`, `:3243` — but all four are provenance records ("corrected from
v1's `:4919`, C-13"), not stale citations.

### Four residual non-blocking nits — all applied after the PASS

1. **T19 lacked an explicit §Files row-flip line**, and its Green said "delete the arity
   conjunct. **Nothing else.**" — which read as forbidding the row-2 edit that three other
   places require. This was the only real internal contradiction; T19 now has a §Files
   block instructing the row-2 flip in the same commit, and its Green says so.
2. T22's Parallel line now carries the `{T21, T22}` §2.5 isolation cross-reference.
3. §2.5's mutation roster now includes T22 (×2), making the isolation rule's scope
   self-evident.
4. T22's rationale citation corrected to `session.rs:1746-1765` (comment `:1748`,
   fallback `:1749-1765`), matching the already-corrected SPEC.

---

## Final status

**Gate satisfied on pass 3 of the permitted three.** Pass 1 ISSUES_FOUND (1 CRITICAL,
3 MAJOR, 13 MINOR) → revision → pass 2 ISSUES_FOUND (2 MAJOR, 9 MINOR, all text-level) →
revision → **pass 3 PASS**. The plan is ready for implementation.

