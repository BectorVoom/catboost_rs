---
title: "MVS sampler upstream-parity — TDD implementation plan"
phase: mvs-tree2-parity
branch: fix/bootstrap-rng-draw-accounting
base_commit: 2c14d7f
plan_version: 3
status: draft
updated_at: 2026-07-31T00:00:00Z
revision_note: >
  v2 responded to PLAN-CHECK pass 1 (3 CRITICAL, 5 MAJOR, 7 MINOR): Wave B DISSOLVED (phase
  fully serial), the two controlled-revert Reds moved into a throwaway `git worktree` at
  2c14d7f, every "clippy clean" / "gate scripts pass" criterion made DIFFERENTIAL, TASK-06
  gained a non-device-gated mirror test, TASK-02 an early device probe, TASK-08 moved before
  TASK-07.
  v3 responds to PLAN-CHECK pass 2 (2 CRITICAL, 3 MAJOR, 5 MINOR), every fix EMPIRICALLY
  MEASURED before being claimed. (1) C2-1: v2's replacement clippy gate grepped `-->`
  location lines, which clippy emits for WARNINGS too, so it was itself RED at HEAD on two
  unfixable warnings (`bootstrap.rs:134` excessive_precision, `mvs_device.rs:80`
  manual_rotate) — every clippy check is now ERROR-attributed via `--message-format=json` +
  jq (verified EMPTY for all four phase files, twice, and again after a `touch` to rule out
  caching), with a verified awk fallback. (2) C2-2: `B4` corrected to the measured 100
  errors / 10 files, `B5` to 4 errors / 3 files, v2's set-equality fallback WITHDRAWN (none
  of its commands could match its own targets), and CI's real gate
  `cargo clippy --workspace --lib -- -D warnings` added as `B11` with its measured RED
  result. (3) C2-3: the worktree moved off the 16 GB RAM-backed tmpfs onto btrfs /home with
  a shared disk-backed CARGO_TARGET_DIR — actually created, the Red actually reproduced
  (59 s cold / 6.6 GB / 4 s warm), then removed. (4) C2-4: `plan6.md` gains a MANDATORY
  step 1a seam extraction so its Red cannot be vacuous. (5) C2-5: plan3's headline binds
  >=5, not >=7. (6) C2-6: a hard constraint forbidding `.sum()` in `bootstrap.rs` prose.
  (7) C2-7..C2-10: scoped worktree cleanup, the two stale `:442-486` citations fixed,
  line-break-tolerant MVS-S7 greps proven to hit all three phrases, and the plan5 wave /
  MVS-S5 `:323` bookkeeping.
source_spec: .planning/plans/mvs-tree2-parity/SPEC.md
source_research: .planning/plans/mvs-tree2-parity/research.md
task_files: [plan1.md, plan2.md, plan3.md, plan4.md, plan5.md, plan6.md, plan7.md, plan8.md, plan9.md]
progress: progress.md
---

# MVS sampler upstream-parity — TDD implementation plan

Plan-only artifact. **No production code and no test body is authored here.**
Every path, symbol, line number, command and baseline below was re-verified at
HEAD `2c14d7f` this session by reading the file or executing the command; the
evidence is inline and repeated per task in each `planN.md`.

Read `SPEC.md` first — it holds the ten specifications (`MVS-S1` … `MVS-S10`),
the ten acceptance criteria (`AC-1` … `AC-10`), the impact table and the risk
register. This file adds only: the goal-backward derivation, the wave/dependency
ordering with an explicit write-conflict check, the coverage tables, and the
cross-cutting guardrails.

`planning/settings.json` (v2) was read: `implementation.use_worktree = false`, so
every task edits the working tree in place on branch
`fix/bootstrap-rng-draw-accounting`. No naming or destination rule in that file
conflicts with this phase's layout.

---

## 0. Goal-backward derivation

The phase's observable end state is `AC-2` + `AC-3`: **MVS matches upstream
CatBoost 1.2.10 at ≤1e-5 over ALL trees, for every seed and both
`boost_from_average` settings, with no reduced-tree carve-out anywhere.** Working
backwards from each acceptance criterion to the artifact that makes it true:

| To claim … | you must first have … | task |
|---|---|---|
| `AC-2` `bootstrap_dev/mvs` 3/3 trees, no carve-out | the fabricated draws gone AND `MVS_GATED_TREES` deleted | TASK-02 |
| `AC-3` 10/10 `(seed, bias)` scenarios ≤1e-5 | a committed multi-seed × bias upstream fixture family + its oracle | TASK-03 |
| either of those | the MVS arm advancing the RNG by exactly ONE draw | **TASK-01** |
| `AC-1` a unit-level draw contract that the oracle cannot express | `TFastRng64::{call_count, raw_state}` asserted against a one-draw probe | TASK-01 |
| `AC-4` block threshold target `1200.0` exactly at `(0.8, 1500)` | a named, testable seam for `SampleRate * blockSize` | TASK-04 |
| `AC-5` every MVS weight `f32`-round-trips | the store narrowed like upstream's `TVector<float>` | TASK-05 |
| `AC-6` the device MVS self-oracle still green | `AC-4`/`AC-5` mirrored into `cb-backend`'s deliberate inline copies | TASK-06 |
| `AC-7` no superseded tree-2 causal claim survives | the carve-out gone (TASK-02) before the grep can pass | TASK-07 |
| `AC-10` the four residual deviations recorded as deviations | the final numerics settled, so the doc is not written twice | TASK-07 |
| `AC-8` device-vs-CPU MVS still ≤1e-5 and 3/3 trees | the host sampler AND the `cb-backend` mirror both final | TASK-08 |
| `AC-9` frozen fixtures byte-unchanged, shared accounting untouched, 503/1 + new tests | everything above, measured once at the end | TASK-09 |

Note the deliberate inversion versus a file-layout plan: **TASK-01 is the whole
bug** and is a 10-line *deletion*; the other eight tasks exist to (a) make the
defect impossible to reintroduce silently (TASK-02, TASK-03), (b) land the two
user-approved f32 transcription fixes without breaking the device self-oracle
(TASK-04, TASK-05, TASK-06), and (c) remove the wrong diagnosis and prove nothing
else moved (TASK-07, TASK-08, TASK-09).

**The root cause is PROVEN — do not re-investigate.** `bootstrap.rs:413-423`
fabricates 2 draws per tree; upstream takes exactly 1
(`mvs.cpp:174 randSeed = rand->GenRand()`, and `performRandomChoice == false`
routes `TCalcScoreFold::Sample` down the draw-free `SetControlNoZeroWeighted`
branch, `calc_score_cache.cpp:742-748`). A reverted spike measured 3/5 → 5/5
(`bias=true`) and 0/5 → 5/5 (`bias=false`)
`[VERIFIED: research.md §0, §8.2, §8.3]`. Any task that reopens the diagnosis is
out of scope.

---

## 1. Execution order — FULLY SERIAL (revised, PLAN-CHECK CRITICAL-3)

**v1 proposed a parallel Wave B and it was not parallel-safe.** TASK-02's and
TASK-03's controlled-revert Red procedures both mutated
`crates/cb-train/src/bootstrap.rs` — the file TASK-04 owns in the same wave — in one
shared working tree (`planning/settings.json` → `"use_worktree": false`, re-verified).
Two interleaved `git stash push`/`pop` on that path could have landed the file with the
two fabricated draws still present while every CPU oracle still passed (they passed
*with* the defect for 3 of 5 bias-true seeds). That is a plausible path to shipping the
phase with the defect partially reintroduced, so v2 removes the shared-file mutation
**entirely** and drops the parallelism:

```
1  plan1  TASK-01  MVS one-draw contract + delete the 2 fabricated draws  [cb-train/src]
2  plan2  TASK-02  remove the MVS_GATED_TREES carve-out (3/3 trees)       [cb-train/tests]  + early device probe
3  plan3  TASK-03  multi-seed × bias fixture family + oracle              [cb-oracle, cb-train/tests]
4  plan4  TASK-04  f32 `SampleRate * blockSize` target                    [cb-train/src]
5  plan5  TASK-05  f32-narrowed MVS weight store                          [cb-train/src]
6  plan6  TASK-06  mirror S4/S5 into cb-backend + a non-device-gated mirror test  [cb-backend]
7  plan8  TASK-08  ROCm device re-verification                            [writes nothing]
8  plan7  TASK-07  documentation debt + MVS-S10 deviations                [docs, .planning]
9  plan9  TASK-09  frozen-fixture invariance + full regression sign-off
```

**Both fixes are applied, not just one.** The checker offered "(a) serialize" or
"(b) isolated worktree"; the coordinator asked for whichever removes shared-file
mutation entirely. Serialization alone only removes the *concurrency*, leaving the
in-place mutate-and-restore fidelity risk (MINOR-4). So:

- the phase is **serial** (no `parallelizable: true` anywhere); AND
- TASK-02's and TASK-03's controlled-revert Reds are captured in a **throwaway
  `git worktree add` at `2c14d7f`**, never by mutating the main tree.
  `git stash` on `crates/cb-train/src/bootstrap.rs` is **FORBIDDEN** in every task.

A worktree at `2c14d7f` also satisfies **MAJOR-1** for free: that commit has the defect
present *and* the f32 target absent, which is exactly the state both Reds must be
measured against.

### 1.0 Worktree location and target dir — MEASURED, not assumed (C2-3)

v2 put the worktree at `/tmp/mvs-red-task0N`. **`/tmp` is a 16 GB RAM-backed tmpfs**
(`df -h /tmp` → `tmpfs 16G`) with only ~3.3 GiB RAM free, and no `CARGO_TARGET_DIR` is
set anywhere, so `cargo test --manifest-path "$W/Cargo.toml"` would have built into
`$W/target` **on tmpfs** — risking ENOSPC or swap thrash mid-Red, whose documented
fallback was the forbidden stash. Use the repo's existing on-disk worktree convention
and an explicit disk-backed target dir:

```bash
WT=/home/user/Documents/workspace/catboost_rs-worktrees/mvs-red-task0N   # N = 2 or 3
TD=/home/user/Documents/workspace/catboost_rs-worktrees/.target-mvs-red  # SHARED by both Reds

df -BG --output=avail,fstype /home | tail -1     # pre-check: need >= 15G, fstype must NOT be tmpfs
git worktree add --detach "$WT" 2c14d7f
mkdir -p "$TD"
CARGO_TARGET_DIR="$TD" cargo test --manifest-path "$WT/Cargo.toml" -p cb-train --test <target> -- --nocapture
```

**Measured end to end in the v3 revision** `[VERIFIED: RUN — worktree actually created,
Red actually produced, then removed]`:

| measurement | value |
|---|---|
| `df -h /tmp` (what v2 would have used) | `tmpfs 16G` — RAM-backed |
| `df -h /home` (what v3 uses) | `btrfs`, **209 GB avail** |
| worktree + target dir filesystem | `btrfs /home` for both ✔ (never tmpfs) |
| **cold** build + run of `--test bootstrap_dev_oracle_test` in the worktree | **59 s wall clock** |
| `du -sh $TD` after that cold build | **6.6 GB** (`/home` avail 209 G → 206 G) |
| **warm** re-run after a one-line test edit | **4 s wall clock** |

6.6 GB is 41 % of the entire tmpfs v2 would have used, on a machine with 3.3 GiB free
RAM. On disk it is 3 % of free space. Because `$TD` is **shared** by both Reds, TASK-03
pays the warm cost, not the cold one. 59 s is not a hang — do not interrupt it.

**Cleanup gate, scoped (C2-7).** `git worktree list` already reports **6 entries** at
HEAD (the main tree, `catboost_rs-worktrees/23-ctr-model-loading`, and four
`.claude/worktrees/agent-*`) `[VERIFIED: RUN]`, so "no leftover entry" is false as
written. The criterion is: **`git worktree list` contains no entry under
`/home/user/Documents/workspace/catboost_rs-worktrees/mvs-red-*`**, and `$TD` is removed:

```bash
git worktree remove --force "$WT" && rm -rf "$TD"
git worktree list | grep -c 'mvs-red'   # must print 0
git stash list                          # must be EMPTY
```

`[VERIFIED: RUN — after teardown, 6 entries remain and `grep -c mvs-red` printed 0]`

### 1.1 Ordering rationale for the two moved tasks

- **TASK-02 and TASK-03 precede TASK-04** (MAJOR-1). Their Reds bind pre-fix diagnostic
  values that were measured with `bootstrap.rs:312` still in `f64`. Capturing them before
  TASK-04 exists keeps the main tree's history aligned with the worktree baseline.
- **TASK-08 precedes TASK-07** (MINOR-3). TASK-08 writes nothing; TASK-07 writes doc
  comments in `crates/cb-train/src/bootstrap.rs`, which TASK-08's seven
  `cargo test -p cb-train --features rocm` invocations compile. Running 08 first removes
  the target-dir lock contention and the mid-edit rebuild at zero cost. (The *correctness*
  risk was nil — a doc-only edit cannot move numerics — so v1's real error was claiming
  "disjoint by construction" rather than acknowledging build contention.)

**Plan-file numbering note.** `planN.md` filenames stay bound to their TASK IDs
(`plan7.md` = TASK-07, `plan8.md` = TASK-08) so that every cross-reference in
`PLAN-CHECK.md`, `progress.md` and this file keeps resolving after the reorder. The
**`order:` frontmatter field is the execution rank** and is authoritative: TASK-08 is
`order: 7`, TASK-07 is `order: 8`. This is the single place where filename order differs
from execution order, and it is deliberate — renumbering two just-reviewed artifacts by
name would silently invalidate the review's citations.

### 1.2 Dependency graph

```text
TASK-01 -> TASK-02 -> TASK-03 -> TASK-04 -> TASK-05 -> TASK-06 -> TASK-08 -> TASK-07 -> TASK-09
```

Acyclic (a total order). Explicit `depends_on` per task, as in each `planN.md`
frontmatter — the *semantic* prerequisites are retained alongside the serial chain so a
reader can see which dependencies are real and which are only sequencing:

| Task | order | depends_on (semantic) | serial predecessor |
|---|---|---|---|
| TASK-01 | 1 | — | — |
| TASK-02 | 2 | TASK-01 | TASK-01 |
| TASK-03 | 3 | TASK-01 | TASK-02 |
| TASK-04 | 4 | TASK-01; **must not precede TASK-02/03's Red capture** | TASK-03 |
| TASK-05 | 5 | TASK-04 | TASK-04 |
| TASK-06 | 6 | TASK-04, TASK-05 | TASK-05 |
| TASK-08 | 7 | TASK-06 | TASK-06 |
| TASK-07 | 8 | TASK-02, TASK-05; **sequenced after TASK-08** (MINOR-3) | TASK-08 |
| TASK-09 | 9 | TASK-02, TASK-03, TASK-06, TASK-07, TASK-08 | TASK-07 |

### 1.3 Write-conflict check for the revised (serial) structure

With a total order there is no concurrent write by construction, so the check below is
about **which files each task touches at all** — including the temporary writes v1's
table wrongly omitted (CRITICAL-3):

| Task | Files it writes (permanent) | Files it writes TEMPORARILY | Where |
|---|---|---|---|
| TASK-01 | `cb-train/src/bootstrap.rs`, `cb-train/src/bootstrap_test.rs` | — | main tree |
| TASK-02 | `cb-train/tests/bootstrap_dev_oracle_test.rs` | `cb-train/src/bootstrap.rs` (defect re-inserted for the Red) | **throwaway worktree at `2c14d7f`** |
| TASK-03 | `cb-oracle/generator/gen_fixtures.py`, `cb-oracle/fixtures/mvs_seeds/**` (new), `cb-train/tests/mvs_seeds_oracle_test.rs` (new) | `cb-train/src/bootstrap.rs` (defect re-inserted for the Red) | **throwaway worktree at `2c14d7f`** |
| TASK-04 | `cb-train/src/bootstrap.rs`, `cb-train/src/bootstrap_test.rs` | — | main tree |
| TASK-05 | `cb-train/src/bootstrap.rs`, `cb-train/src/bootstrap_test.rs` | — | main tree |
| TASK-06 | `cb-backend/src/kernels/mvs_device_test.rs`, `cb-backend/src/kernels/mvs_device.rs` (**comment lines only**, MINOR-1) | — | main tree |
| TASK-08 | **none** | — | — |
| TASK-07 | `cb-train/src/bootstrap.rs` (doc comments only), `.planning/plans/device-bootstrap-parity/{progress.md,SPEC.md}` | — | main tree |
| TASK-09 | `.planning/plans/mvs-tree2-parity/{progress.md,SPEC.md}` | — | — |

The main tree therefore only ever sees `crates/cb-train/src/bootstrap.rs` written by
TASK-01 → TASK-04 → TASK-05 → TASK-07, strictly in that order.

Ordering independence facts, checked against CodeGraph rather than file names:

- `bootstrap()` has exactly three call sites — `boosting.rs:3262` (device branch),
  `boosting.rs:3833` (CPU branch), and `bootstrap_test.rs`
  `[VERIFIED: grep "bootstrap(" crates/cb-train/src/boosting.rs; research.md §6.1]`.
- **The `A + B` spike claim, correctly scoped (MAJOR-1 — v1 over-claimed here).**
  Research §8.5/§8.6 measured only the **post-fix** combinations (`A + B` and
  `A + B + C`), reporting 5/5 + 5/5 with byte-identical residuals. It therefore supports
  only this: *TASK-04 cannot change the post-fix pass/fail verdict of TASK-02 or
  TASK-03.* It says **nothing** about the **defect-present + f32-target** state (`B`
  without `A`), which was never measured — and the +1.788e-5 target shift at
  `block_size = 1500` is precisely the kind of perturbation that can move a near-tied
  argmax, so the pre-fix `StageDiverged` values and the pre-fix failing set **may
  differ** with TASK-04 present. v1's claim that TASK-04 "provably cannot invalidate
  TASK-03" is **withdrawn**. Consequence, now enforced: both Reds are captured at
  `2c14d7f` in the isolated worktree, where the f32 target is absent by construction.
- `crates/cb-oracle/generator/gen_fixtures.py` is touched only by TASK-03, and only
  by APPENDING a new function plus a new `--mvs-seeds-only` argv arm — the existing
  `gen_bootstrap()` (`:710`) and `gen_bootstrap_dev()` (`:858`) bodies are not
  edited `[VERIFIED: read `:700-956`, `:3346-3385`]`.

---

## 2. Specification → task coverage

| Spec | Behaviour | Primary task | Also verified by |
|---|---|---|---|
| `MVS-S1` | MVS `bootstrap()` consumes exactly ONE main-stream draw | **TASK-01** | TASK-02, TASK-03, TASK-08 |
| `MVS-S2` | `bootstrap_dev/mvs` matches upstream 3/3 trees, no carve-out | **TASK-02** | TASK-08 (device arm), TASK-09 |
| `MVS-S3` | multi-seed × bias fixture family gates the sampler | **TASK-03** | TASK-09 |
| `MVS-S4` | block threshold target reproduces upstream's `float` expression | **TASK-04** | TASK-06, TASK-09 |
| `MVS-S5` | stored MVS weights are narrowed through `f32` | **TASK-05** | TASK-06, TASK-09 |
| `MVS-S6` | `cb-backend` inline CPU transcription stays consistent | **TASK-06** | TASK-08 |
| `MVS-S7` | the superseded tree-2 diagnosis is removed from the tree | **TASK-07** | TASK-09 (grep re-run) |
| `MVS-S8` | device-vs-CPU MVS parity survives the fix | **TASK-08** | — |
| `MVS-S9` | frozen fixtures + shared draw accounting stay invariant | **TASK-09** | every task's Verify step |
| `MVS-S10` | remaining known deviations are documented, not silently carried | **TASK-07** | TASK-09 |

Every specification maps to ≥1 task; every task references ≥1 specification.

## 3. Acceptance criterion → task coverage

| AC | Observable success condition | Task that makes it true | Evidence artifact |
|---|---|---|---|
| `AC-1` | MVS `bootstrap()` advances the RNG by exactly one `gen_rand`, asserted by `raw_state()` against a one-draw probe | TASK-01 | `cargo test -p cb-train --lib bootstrap` |
| `AC-2` | `bootstrap_dev/mvs` ≤1e-5 over **3/3** trees; no reduced-tree carve-out anywhere in the file | TASK-02 | `--test bootstrap_dev_oracle_test` printout `3/3 trees` |
| `AC-3` | all 10 `(seed, bias)` scenarios ≤1e-5, and — binding — **≥5** (incl. ≥1 per bias setting) demonstrably fail without the fix; 7/10 is expected-and-recorded, not a gate (MAJOR-5) | TASK-03 | `--test mvs_seeds_oracle_test` + the recorded worktree revert run |
| `AC-4` | the block threshold target is exactly `1200.0` at `(0.8, 1500)` | TASK-04 | `cargo test -p cb-train --lib mvs_block_sample_size` |
| `AC-5` | every MVS sample weight round-trips through `f32` losslessly | TASK-05 | `cargo test -p cb-train --lib mvs_sample_weights_are_f32` |
| `AC-6` | the device MVS self-oracle is green on the device backend **AND a non-device-gated test enforces the mirror** (MAJOR-3) | TASK-06 | `cargo test -p cb-backend --lib mvs` (cpu, the new mirror test) + `--no-default-features --features rocm --lib mvs` |
| `AC-7` | no superseded tree-2 causal claim survives **outside `.planning/plans/mvs-tree2-parity/`** (MINOR-2) | TASK-07 | the three-string grep, scoped |
| `AC-8` | device-vs-CPU MVS ≤1e-5 and the device arm reports 3/3 trees — **projected**, first measured on ROCm; probed early by TASK-02 (MAJOR-2) | TASK-08 (probe in TASK-02) | rocm `device_bootstrap_parity_test` + `bootstrap_dev_oracle_test` |
| `AC-9` | three frozen roots byte-unchanged; shared constants + replay untouched; `cb-train` **503 passed / 1 failed / 4 ignored** + the new tests; every lint/script gate DIFFERENTIAL vs §4.11 | TASK-09 | `git status --short` on the roots + `--no-fail-fast` tally + the §4.12 diff-scoped greps |
| `AC-10` | the four remaining deviations appear as documented deviations | TASK-07 | the doc diff on `bootstrap.rs` |

---

## 4. Cross-cutting guardrails (apply to EVERY task)

1. **Source/test separation is MANDATORY** (`CLAUDE.md`). No `#[cfg(test)] mod tests { … }`
   body inside a production file. In `src/`, tests live in a sibling `*_test.rs`
   mounted with `#[cfg(test)] #[path = "x_test.rs"] mod tests;` — the exact
   precedent is `crates/cb-train/src/bootstrap.rs:55-57`
   `[VERIFIED: read]`. `crates/cb-backend/src/kernels.rs:2955-2963` is the
   cb-backend precedent for `kernels/mvs_device_test.rs` `[VERIFIED: grep]`.
   Integration tests go under `crates/<crate>/tests/`. The CI backstop is
   `bash scripts/check-source-test-separation.sh` (it flags only the *brace* form;
   the `mod x;` declaration form is explicitly allowed) `[VERIFIED: read the script]`.
2. **No `unwrap` / `expect` / `panic` / raw indexing in production.** The workspace
   denies all four (`Cargo.toml [workspace.lints.clippy]`: `unwrap_used`,
   `expect_used`, `panic`, `indexing_slicing` = `"deny"`) `[VERIFIED: read]`. Test
   files opt out with a file-level
   `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]`
   — precedent `crates/cb-train/src/bootstrap_test.rs:10` and
   `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:24` `[VERIFIED: read]`. **Every
   new test file this phase adds MUST carry that opt-out** — the 14 pre-existing files in
   §4.11's clippy baseline are red precisely because they lack it.
3. **Typed errors only** — `thiserror`-derived `CbError` / `CbResult` from
   `cb-core`. No new error variant is needed in this phase: `bootstrap`'s public
   signature and `BootstrapResult` are unchanged (SPEC §4), and `Poisson` remains
   the only erroring arm.
4. **All draws through `cb_core::TFastRng64`; all float sums through
   `cb_core::sum_f64`.** This is a **design rule about the code this phase writes**, and
   it is trivially satisfied: the phase adds no float sum and no `anyhow`. The two
   backstop scripts (`check-no-raw-float-sum.sh`, `check-no-anyhow.sh`) are **red at
   HEAD** and are therefore DIFFERENTIAL gates, not absolute ones — see §4.12.
5. **Backend build lines.** Device runs are ALWAYS
   `--no-default-features --features rocm` (never bare `--features rocm` — that
   unifies `cpu` in) and ALWAYS `--test <target>` for `cb-train` (a blanket rocm
   test build fails on ~37 files importing `CpuBackend`). `cb-backend --lib` under
   rocm DOES build as a whole `[PROJECT: .planning/plans/device-bootstrap-parity/plan3.md:167,
   plan10.md:84-117; research.md §8.8]`. The local rig is present and is
   **gfx1151** (`AMD Radeon 860M`, `/home/user/rocm/opt/rocm/bin/rocminfo`)
   `[VERIFIED: RUN rocminfo]`.
6. **Device test skip convention.** Any new test that cannot run on the default
   `cpu` feature must gate with
   `#[cfg(not(any(feature = "rocm", feature = "cuda")))]` for the real body and a
   `#[cfg(any(feature = "rocm", feature = "cuda"))]` arm that PRINTS
   `SKIP <name>: …` — copy the shape of
   `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:225-227` /
   `device_bootstrap_parity_test.rs:534-540` `[VERIFIED: read]`. Import
   `cb_backend::CpuBackend` **inside** the gated fn, never at file top level, so
   the file still compiles under rocm.
7. **ε = 1e-5** for every upstream-parity assertion — `cb_oracle::compare_stage`
   hard-codes `1e-5` `[VERIFIED: crates/cb-oracle/src/compare.rs:85`
   `assert_abs_close(expected, actual, 1e-5)`]`. Do not introduce a looser bar.
   The pre-existing device-vs-CPU self-oracle bar in
   `kernels/mvs_device_test.rs:27` is `TOL = 1e-4`; leave it alone, do not copy it.
8. **Do NOT hand-roll or re-tune** (each independently verified; changing any
   re-breaks Bayesian): `calculate_threshold`'s algorithm
   (`bootstrap.rs:209-273`), `PRE_TREE_DRAWS` (`boosting.rs:59`),
   `POST_TREE_EXTRA_DRAWS` (`boosting.rs:69`), `replay_grow_draws`
   (`device_draw_replay.rs:64-85`), `select_level_perturbed`'s draw shape
   (`tree.rs`), `last_iter_mean_leaf_value` (`bootstrap.rs:363-369`),
   `mvs_lambda`, `single_probability`, `mean_grad_value`'s formula
   `[VERIFIED: grep + read; research.md §5.5]`.
9. **Frozen fixtures.** `crates/cb-oracle/fixtures/bootstrap/**`,
   `crates/cb-oracle/fixtures/bootstrap_dev/**` and
   `crates/cb-oracle/fixtures/inputs/**` must be **byte-unchanged** at the end of
   the phase. Never regenerate one to make a test pass — that destroys the
   independent evidence that made the root cause provable (SPEC R4). The only
   generator entrypoint this phase may add is a NEW `--mvs-seeds-only` writing a
   NEW root.
10. **CubeCL rule.** This phase needs **no new kernel and no executable kernel edit**
    (`MVS-S6` non-goal). The ONE permitted kernel-file change is a **doc-only comment**
    at `crates/cb-backend/src/kernels/mvs_device.rs:145-146` recording the deliberate
    `f64`-target deviation (MINOR-1 — the gate is "no executable change", proven by a
    `git diff` showing comment lines only). If any task nonetheless reaches for
    `#[cube]`, STOP: read
    `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` first,
    and on any CubeCL build error read `cubecl_error_guideline.md` BEFORE
    attempting a fix. Blind fixes are prohibited (`CLAUDE.md`).
11. **Known pre-existing reds — record, never chase.** Extended in v2 with the
    lint/script baselines v1 omitted (PLAN-CHECK CRITICAL-1, CRITICAL-2). Every row
    `[VERIFIED: RUN at HEAD 2c14d7f during the plan-check revision]`:

    Every row below was **re-measured in the v3 revision** with the exact command shown,
    and B4/B5 were run **twice consecutively** to confirm reproducibility.

    | # | gate | HEAD baseline (measured) |
    |---|---|---|
    | B1 | `cargo test -p cb-train --no-fail-fast` | **503 passed / 1 failed / 4 ignored** — the failure is `monotone_non_symmetric_and_region_are_typed_errors` (`crates/cb-train/tests/monotone_oracle_test.rs:286`) |
    | B2 | `cargo test -p cb-backend --lib --no-fail-fast` (cpu) | **173 passed / 60 failed** (CubeCL-CPU MLIR `plane_inclusive_sum` unsupported) |
    | B3 | `cargo clippy -p cb-train --all-targets` — **the command v1 wrote** | **RED — the build ABORTS** at `crates/cb-oracle/src/model_json.rs:161:17` ("indexing may panic") *before* `cb-train`'s own test targets are reached, because clippy lints path dependencies. **Never use it.** |
    | **B4** | `clippy_error_files -p cb-train --all-targets` (the §4.12 helper) | **RED — exactly 100 errors across exactly 10 files** (measured twice, identical): `tensor_ctr_oracle_test.rs` 31, `device_seam_test.rs` 22, `yetirank_pairwise_tree_rng_oracle_test.rs` 11, `ordered_ctr_oracle_test.rs` 11, `plain_ctr_oracle_test.rs` 8, `ordered_boost_oracle_test.rs` 8, `permutation_oracle_test.rs` 3, `structure_fold_cycle_oracle_test.rs` 2, `s_order_ctr_bins_oracle_test.rs` 2, `learn_set_shuffle_oracle_test.rs` 2 — **sums to exactly 100**. All ten lack the file-level `#![allow(...)]`. *(v2 wrongly said 14 files / a list summing to 104: it counted `-->` location lines, which clippy also emits for warnings, so four warning-only files — `tensor_ctr_e2e_oracle_test.rs`, `multilabel_oracle_test.rs`, `multiclass_oracle_test.rs`, `ctr_split_scoring_test.rs` — were miscounted as error files. Corrected per C2-2.)* |
    | **B5** | `clippy_error_files -p cb-backend --lib` | **RED — exactly 4 errors across 3 files** (measured twice, identical): `cpu_runtime.rs` **2** (`:696:13`, `:1025:29`), `kernels/bootstrap_device.rs` 1 (`:230:28`), `kernels/exact_quantile.rs` 1 (`:178:8`) |
    | B6 | `cargo clippy -p cb-backend --all-targets` | additionally 2 lib-**test**-target errors (`kernels/gradient.rs:18`, `kernels/score_split.rs:374`); restrict to `--lib` for exactly 4 |
    | **B7** | `bash scripts/check-source-test-separation.sh` | **exit 0** — `OK: no inline #[cfg(test)] module bodies in production source`. The one script that IS an absolute gate. |
    | B8 | `bash scripts/check-no-raw-float-sum.sh` | **exit 1** — 15 `D-08 violation:` headers, 36 output lines. Includes `crates/cb-train/src/boosting.rs:1649`. `crates/cb-train/src/bootstrap.rs` is **clean** of its `SUM_PATTERN` (verified: `grep -nE '\.sum\(\)\|\.fold\(0\.0\|\.fold\(0_f\|\.fold\(0f' crates/cb-train/src/bootstrap.rs` → exit 1) — see §4.13. |
    | B9 | `bash scripts/check-no-anyhow.sh` | **exit 1** — 12 `D-14 violation:` headers, 25 output lines; every hit is a doc comment reading "no `anyhow`" |
    | **B11** | **`cargo clippy --workspace --lib -- -D warnings`** — the gate CI actually runs (`.github/workflows/ci.yml:47`) | **RED at HEAD**, exit **101**, aborting at `error: could not compile 'cb-data' (lib) due to 4 previous errors`. With `--keep-going --message-format=json`: **5 errors** (`cb-backend/src/cpu_runtime.rs` ×2, `kernels/bootstrap_device.rs`, `kernels/exact_quantile.rs`, `cb-oracle/src/model_json.rs`) and **100 warnings**, which `-D warnings` promotes. Of those 100, **exactly one names a file this phase touches**: `crates/cb-backend/src/kernels/mvs_device.rs:80` `clippy::manual_rotate`. `crates/cb-train/src/bootstrap.rs` does **NOT** appear in this gate's set (measured before and after `touch`). Added per C2-2/C2-10 because this is the invocation that will judge the phase's new **lib** code. |
    | B12 | pre-existing **warnings** inside the phase's own edit surface | `crates/cb-train/src/bootstrap.rs:134` `clippy::excessive_precision` (×2 under `--all-targets`: once for the lib unit, once for the lib-test unit) and `crates/cb-backend/src/kernels/mvs_device.rs:80` `clippy::manual_rotate`. **Neither is fixable by this phase**: `bootstrap.rs:110-116` states that trimming the verbatim `0.693_147_18_f32` literal "would change the exact f32 bit pattern and break Bayesian parity at the ~1e-5 oracle bound". These two entries are why the gate must select **errors**, not any diagnostic (C2-1). |
    | B10 | `cb-backend` build warnings | `kernels.rs:645:30`, `:673:38` (`float_literal_f32_fallback`, future-incompat, not errors) |

    **`--keep-going` is MANDATORY for B4/B5/B11.** Without it cargo aborts targets in
    parallel and the surfaced subset varies run to run — measured: the same B4 command
    reported 8 errors in `ordered_boost_oracle_test.rs` + 2 in
    `structure_fold_cycle_oracle_test.rs` and stopped, versus 100 across 10 files with
    `--keep-going`. A non-`--keep-going` baseline is not comparable between runs.

12. **Every lint/script gate in this phase is DIFFERENTIAL, never absolute**
    (CRITICAL-1, CRITICAL-2). v1 wrote "clippy clean" in six tasks and "the three gate
    scripts pass" in the Definition of Done. Both are **unsatisfiable at HEAD**, and
    worse, B8 reports `crates/cb-train/src/boosting.rs:1649` — a file `MVS-S9`/TASK-09
    requires to be **byte-unchanged** — so "make the script pass" directly contradicts
    another plan requirement. Satisfying these gates literally would mean editing ~25
    files across five crates inside a bug-fix phase: exactly the scope leakage SPEC `R5`
    forbids. The two acceptable forms are:

    This phase touches exactly four source files —
    `crates/cb-train/src/bootstrap.rs`, `crates/cb-train/src/bootstrap_test.rs`,
    `crates/cb-train/tests/mvs_seeds_oracle_test.rs` (new),
    `crates/cb-backend/src/kernels/mvs_device_test.rs` — plus comment-only lines in
    `crates/cb-backend/src/kernels/mvs_device.rs`. The gate is: **no gate output line
    names any of those paths.**

    **The clippy gate MUST select ERRORS, not any diagnostic (C2-1).** v2 grepped
    clippy's `-->` location lines, which clippy emits **for warnings too** — so the gate
    was RED at HEAD on two pre-existing warnings the phase cannot fix
    (`bootstrap.rs:134` `excessive_precision`, `mvs_device.rs:80` `manual_rotate`; see
    `B12`), reproducing the exact defect CRITICAL-1 was raised about. Use the
    severity-filtered form:

    ```bash
    # jq is present: /usr/bin/jq, jq-1.8.1 [VERIFIED: RUN command -v jq]
    clippy_error_files() {          # $@ = cargo clippy target selection
      cargo clippy "$@" --no-deps --keep-going --message-format=json 2>/dev/null \
        | jq -r 'select(.reason=="compiler-message") | .message
                 | select(.level=="error") | .spans[]? | select(.is_primary) | .file_name'
    }
    PHASE='src/bootstrap\.rs|bootstrap_test\.rs|mvs_seeds_oracle_test\.rs|mvs_device'

    clippy_error_files -p cb-train  --all-targets | grep -E "$PHASE"   # must be EMPTY
    clippy_error_files -p cb-backend --lib        | grep -E "$PHASE"   # must be EMPTY
    bash scripts/check-no-raw-float-sum.sh 2>&1   | grep -E "$PHASE"   # must be EMPTY
    bash scripts/check-no-anyhow.sh       2>&1    | grep -E "$PHASE"   # must be EMPTY
    ```

    **Measured at HEAD `2c14d7f` in the v3 revision** `[VERIFIED: RUN]`: all four are
    EMPTY (`grep` exit 1), and the two clippy invocations reproduce `B4` (100 errors / 10
    files) and `B5` (4 errors / 3 files) **identically on two consecutive runs**. The
    `cb-train` gate was additionally re-run after `touch crates/cb-train/src/bootstrap.rs`
    to prove it is not silently empty from cargo caching — still EMPTY, while the same run
    does emit the two `bootstrap.rs:134` warnings, confirming the file IS being linted.

    **If `jq` is unavailable**, the equivalent pairs each `^error` with its following
    `-->` and drops the `could not compile` aggregates:

    ```bash
    clippy_error_files() {
      cargo clippy "$@" --no-deps --keep-going 2>&1 \
        | awk '/^error/ {if ($0 !~ /could not compile/) want=1; next}
               want && /^[[:space:]]+--> / {print $2; want=0}' | cut -d: -f1
    }
    ```

    **Verified byte-equivalent to the `jq` form** `[VERIFIED: RUN]`: same 100 errors /
    same 10 files, same EMPTY phase-file filter.

    **There is NO set-equality fallback (C2-2).** v2 offered one and *none* of its four
    commands could match its own targets (`^error` counts are 110 and 5, not 100 and 4,
    because of the `could not compile …` aggregate lines; and the `uniq -c` over `-->`
    lines yields ~40 warning-polluted files, not 14). With the severity-filtered per-file
    gate above it is unnecessary: `B4`/`B5` are recorded for reference, and the gate that
    must pass is the per-file one. If a genuine cross-crate regression check is ever
    wanted, compare `clippy_error_files … | sort | uniq -c` against `B4`/`B5` — those two
    tables are the measured, reproducible ones.

    B7 (`check-source-test-separation.sh`) remains an **absolute** gate — it passes at
    HEAD (exit 0) and must keep passing. **Never** restate B3, B8, B9 or B11 as "must
    pass".

13. **Do not write a D-08 pattern into `bootstrap.rs` prose (C2-6).**
    `scripts/check-no-raw-float-sum.sh` applies `SUM_PATTERN`
    (`\.sum\(\)|\.fold\(0\.0|\.fold\(0_f|\.fold\(0f`) with `grep -RIlE` to every
    non-`*_test.rs` source file — **comments included**. That is why 12 of `B8`'s 15
    baseline violations are doc comments *describing* the ban.
    `crates/cb-train/src/bootstrap.rs` is currently **clean** of the pattern
    `[VERIFIED: RUN — grep exit 1]`, which is the only reason its diff-scoped `B8` gate is
    empty. So any doc text this phase adds to `bootstrap.rs` (TASK-01, TASK-04, TASK-05
    and especially TASK-07's deviation (a), which is *about* summation order) must refer to
    the reduction as `sum_f64`, "a raw iterator summation" or "a naive iterator sum" —
    **never** the literal `.sum()` or `.fold(0.0`. Re-run the `B8` diff-scoped grep after
    every doc edit.
12. **The instrumented upstream build is already compiled and up to date** at
    `/home/user/cb_instrumented_build/catboost-src` (+ binary at
    `/home/user/cb_instrumented_build/build/catboost/app/catboost`) and reproduces
    its committed traces byte-for-byte. `mvs.cpp` itself has **zero** trace points,
    and research §4.4 concludes adding them is unnecessary for this fix. **No task
    in this plan performs instrumentation work.** If a future finding genuinely
    needs an `mvs.cpp` trace point, the recipe is research §4.4 steps 1-5 and the
    task must cite why the existing `tree_rng_*` / `gts_level_rng` fences are
    insufficient.
13. **Every measured number goes into `progress.md`.** A task is not done until its
    row in the measurements table is filled with a real observed value, not
    "expected".

---

## 4.14 Post-check correction and the residual non-blocking findings

Plan-check pass 3 (the last of the three allowed passes) returned `ISSUES_FOUND` with
exactly ONE execution-blocking issue, `[C3-1]`, plus four non-blocking ones. `[C3-1]` was
corrected AFTER that pass, so **the artifacts as they now stand have not themselves been
re-checked by an independent pass** — the executor carries that.

**`[C3-1]` (CRITICAL, corrected here).** The mandated step-1a seam body was the bare
`sample_rate * block_size as f64`, but the unit tests call the helper with a raw `0.8`
literal and `0.8_f64 * 1500.0` is exactly `1200.0` — so assertion 1 would have PASSED on
its first run, the documented Red `left: 1200.0000178813934` would have been unreachable,
and the `MVS-S4` half of the mirror would have shipped untested. Fixed in `plan4.md` and
`plan6.md`: the seam narrows the rate itself
(`f64::from(sample_rate as f32) * block_size as f64`), which is idempotent at the real
call site because `bootstrap.rs:294` already narrows. Independently re-derived:

| `(rate, block)` | bare `f64` | in-situ (step 1a) | upstream target (Green) |
|---|---|---|---|
| `(0.8, 1500)` | 1200.0 | 1200.0000178813934 | **1200.0** |
| `(0.8, 8192)` | 6553.6 | 6553.60009765625 | 6553.60009765625 |
| `(0.8, 3616)` | 2892.8 | 2892.800043106079 | **2892.800048828125** |

Assertion 2 is the no-regression leg (already exact); assertions 1 and 3 now genuinely
fail before the Green and pass after.

**Non-blocking, for the executor to handle in flight:**

- `[C3-2]` `plan6.md`'s completion criterion "`cpu_block_threshold` uses …" contradicts
  its own Green step, which moves the arithmetic into the helper. Treat the Green step as
  authoritative.
- `[C3-3]` `plan3.md`'s "warm 4 s" is optimistic: `plan2.md` deletes `$TD` as a completion
  criterion, so TASK-03 pays the cold ~60 s build. Budget for it.
- `[C3-4]` No gate catches a NEW clippy *warning* in a phase file — §4.12 selects errors
  only, and `B11` (`--workspace --lib`) aborts in `cb-backend` before reaching `cb-train`.
  If a task adds a warning to `bootstrap.rs`, nothing here fails. Eyeball the clippy output
  for the four phase files rather than trusting the gate alone.
- `[C3-5]` `grep -c … # MUST print 0` also exits 1 (guard the call site); the cold ROCm
  build is ~12 m 45 s, unstated in the estimates; `plan8.md`'s `depends_on` omits TASK-03.

**What pass 3 verified by EXECUTION** (the strongest evidence in this phase): in a
throwaway worktree the checker applied TASK-01 + TASK-04 + TASK-05 and measured
`[cpu] bootstrap_dev/mvs: … within 1e-5 of upstream over 3/3 trees` with
`bootstrap_oracle_test` 5/5 green. TASK-01 alone suffices for the goal; the two f32
changes are transcription fidelity, not load-bearing. It also reproduced the pre-fix Red
byte-for-byte (`StageDiverged { Splits, index: 5, expected: -0.025514747947454453,
actual: -0.2692405581474304 }`).

## 5. Hardware and tooling routing

| Task | order | No GPU needed | Local ROCm gfx1151 | Python + `catboost==1.2.10` |
|---|---|---|---|---|
| TASK-01 | 1 | **yes** | — | — |
| TASK-02 | 2 | yes (CPU arm + the worktree Red) | **required** for the early device probe (MAJOR-2) | — |
| TASK-03 | 3 | yes (the Rust oracle) | — | **required** (fixture generation) |
| TASK-04 | 4 | **yes** | — | — |
| TASK-05 | 5 | **yes** | — | — |
| TASK-06 | 6 | yes for the NEW mirror test (host arithmetic, default `cpu`) | **required** for the device self-oracle | — |
| TASK-08 | 7 | — | **required** | — |
| TASK-07 | 8 | **yes** | — | — |
| TASK-09 | 9 | yes (CPU tally + invariance) | recommended (re-confirm 06/08) | — |

TASK-02 gains a ROCm requirement in v2 (MAJOR-2): it edits the rocm-gated device arm of
`bootstrap_dev_oracle_test.rs` at `:347`, and v1 left that new claim unmeasured for five
tasks. The probe is cheap and has **no dependency on TASK-06** — the `cb-backend` mirror
is a change to a *test* file in a *different* crate and cannot affect
`bootstrap_dev_oracle_test`.

`catboost==1.2.10` is importable from BOTH `.venv/bin/python` and the system
`python3` `[VERIFIED: RUN — `.venv` reports catboost 1.2.10 / numpy 1.26.4; system
python3 reports 1.2.10]`. Use `.venv/bin/python`, which is the convention recorded
in `gen_fixtures.py`'s own header.

Tasks 1, 3, 4, 5, 7 and the CPU half of 9 are backend-independent. TASK-02 (probe),
TASK-06 and TASK-08 need the ROCm rig. No CUDA/Kaggle sign-off is required this phase:
the fix is host-side and `boosting.rs:3262` proves the device branch consumes the *same*
host sampler, so a ROCm run is a strictly stronger gate than "it compiles under cuda".

---

## 6. New risks found while planning (beyond SPEC §9)

| # | risk | why it is real | mitigation / owner |
|---|---|---|---|
| **P1** | After `MVS-S4`, `kernels/mvs_device_test.rs`'s `cpu_block_threshold` computes the target in `f32` while the **kernel** `mvs_sample_kernel` still computes `sample_size = rate * f64::cast_from(u64::cast_from(bs))` in `f64` `[VERIFIED: crates/cb-backend/src/kernels/mvs_device.rs:146]`. **Now sized exactly**: of the five existing `cpu_block_threshold` call sites only `(rate 0.3, n 200)` moves at all — by 1.431e-6 absolute / **2.384e-8 relative**; `(0.5,48)`, `(0.7,64)`, `(0.6,96)`, `(0.5,8192)` and `(0.5,24)` are **bit-identical** `[VERIFIED: RUN — numpy f32-vs-f64 table]`. | `mvs_device_test.rs` asserts `assert_eq!(kept_dev, kept_cpu)` — an EXACT keep-count equality (`:224-227`). A flip needs an object's pinned `r` inside a 2.4e-8·p window around `p`; over 200 objects that is ~5e-6 likely. The weight bar (`TOL = 1e-4`) absorbs the shift with ~4 orders to spare. | TASK-06 owns it. If the keep-count equality flips on ROCm: do NOT loosen `TOL`, and do NOT edit `mvs_sample_kernel`'s executable behaviour (SPEC `MVS-S6` non-goal). STOP and escalate with the observed object index. Blast radius is the self-oracle ONLY — `launch_mvs_weights_resident` is dead on the live path (`cb-compute/src/runtime.rs:1131` defaults `mvs_lambda: None`; no `Some` anywhere in `cb-train`) `[VERIFIED: research.md §6.2 + PLAN-CHECK CodeGraph]`. |
| **P2** | `MVS-S7`'s postcondition is repo-wide ("no occurrence … remains anywhere"), but two *stale cross-references* live in a COMPLETED sibling phase's spec: `.planning/plans/device-bootstrap-parity/SPEC.md:746` ("MVS gated to trees 0–1, see progress.md R-1") and `:793` ("mvs over trees 0–1 (R-1)") `[VERIFIED: grep + read]`. | They are not one of the three banned claims, but they become false the moment TASK-02 lands. | TASK-07 **annotates** them (appends "superseded — RESOLVED by `mvs-tree2-parity`; MVS is now 3/3") rather than rewriting them. Completed evidence is preserved, never overwritten. |
| **P3** | `ISOLATING_PARAMS` (`gen_fixtures.py:151-164`) pins `random_seed: SEED` = 0 and does NOT contain `boost_from_average` (it is set per call site) `[VERIFIED: read]`. A new generator that splats `ISOLATING_PARAMS` without overriding BOTH knobs would silently emit ten identical seed-0 fixtures. | The family would then have zero discriminating power and would pass both before and after the fix — the exact failure mode `MVS-S3` exists to prevent. | TASK-03's Red is defined as "**≥5** of 10 fail with the fix reverted, including ≥1 per bias setting" (MAJOR-5); a family that passes 10/10 pre-fix FAILS the task, and two `config.json` files must provably differ. |
| **P4** | `mvs_sample_weights` narrows `sample_rate` at `bootstrap.rs:294` (`f64::from(sample_rate as f32)`) BEFORE the target is computed. `MVS-S4`'s expression `(sample_rate as f32) * block_size as f32` therefore applies a second, idempotent `as f32` to an already-f32-exact value. | Harmless (idempotent), but a reader may "simplify" it away and re-break the transcription faithfulness for callers that bypass `:294`. | TASK-04 must document the double narrowing as deliberate in the helper's doc comment. |
| **P5** (v2, from CRITICAL-1/2) | Six v1 tasks gated on "clippy clean" and the DoD on "the three gate scripts pass". Measured: `cargo clippy -p cb-train --all-targets` **aborts** on a dev-dependency (`cb-oracle/src/model_json.rs:161`), `--no-deps --keep-going` yields **100 errors across 14** pre-existing test files, and 2 of 3 scripts **exit 1** `[VERIFIED: RUN]`. | An implementer facing a red gate either declares the task blocked, or "fixes" ~25 unrelated files across five crates — including `boosting.rs`, which `MVS-S9` freezes — or silently drops the only lint check on the phase's own new code. | §4.11 records every baseline; §4.12 makes all of them differential/diff-scoped. `check-source-test-separation.sh` stays absolute. |
| **P6** (v2, from CRITICAL-3) | v1's Wave B had TASK-02 and TASK-03 `git stash push` `crates/cb-train/src/bootstrap.rs` — the file TASK-04 owned in the same wave — in ONE shared working tree (`use_worktree: false`). | Interleaved pops could land the file with the two fabricated draws still present. Every CPU oracle would still pass (they passed *with* the defect for 3 of 5 bias-true seeds), so the phase could ship partially un-fixed. | Phase is now fully **serial**, and both controlled-revert Reds move to a throwaway `git worktree add` at `2c14d7f`. `git stash` on that path is FORBIDDEN. §1.3 lists the temporary writes v1 omitted. |
| **P7** (v2, from MAJOR-3) | After v1's TASK-06 the `cb-backend` mirror had **no test**: 4 of 5 call sites are bit-identical under `MVS-S4` and the 5th moves 2.384e-8 relative against a `TOL = 1e-4` bar `[VERIFIED: RUN]`. | A future phase edits `mvs_block_sample_size` or the weight store, forgets the copy, and every CPU and rocm test still passes — risk `R2` recurs in exactly the form this phase set out to close. | TASK-06 now adds a **non-device-gated** host-arithmetic test pinning the mirrored target values and the reference weights' f32-representability. It runs on the default `cpu` feature, closing the "invisible on default CI" hole `MVS-S6` itself names. |

---

## 7. Definition of done for the phase

- [ ] `AC-1` … `AC-10` (SPEC §6) all hold, each with a recorded measurement in
      `progress.md`.
- [ ] `cargo test -p cb-train --test bootstrap_oracle_test` green — 5 tests,
      including the value-sensitive 3-tree `bootstrap_oracle_bayesian`
      `[VERIFIED green at HEAD this session]`.
- [ ] `cargo test -p cb-train --test bootstrap_dev_oracle_test` green and printing
      `3/3 trees` for **all four** scenarios.
- [ ] `git status --short crates/cb-oracle/fixtures/bootstrap crates/cb-oracle/fixtures/bootstrap_dev crates/cb-oracle/fixtures/inputs`
      is EMPTY.
- [ ] `git diff` touches none of `PRE_TREE_DRAWS`, `POST_TREE_EXTRA_DRAWS`,
      `replay_grow_draws`, `select_level_perturbed`'s draw shape, or
      `calculate_threshold`'s algorithm.
- [ ] `cargo test -p cb-train --no-fail-fast` → **507 passed / 1 failed / 4 ignored**
      (503 + 4 new CPU-visible tests; the 4 ignored are pre-existing and unchanged; the
      single failure is `monotone_non_symmetric_and_region_are_typed_errors`)
      `[baseline 503/1/4 VERIFIED: RUN]`. Re-derive the tally, by name, if a task adds a
      test beyond the five planned (the four above plus TASK-06's mirror test, which
      lives in `cb-backend`, not `cb-train`).
- [ ] **Differential lint gates (§4.12), NOT "clean"** — every one of these is red at
      HEAD for pre-existing reasons. The gate is the **ERROR-attributed, diff-scoped** set
      of four commands in §4.12; all four must return EMPTY. Each was measured EMPTY at
      HEAD, so a hit is genuinely this phase's. **Never** assert that clippy is clean, that
      `B8`/`B9` pass, or that any output "matches the 14-file set" (there are 10 error
      files, and v2's set-equality fallback is withdrawn — C2-2).
- [ ] `bash scripts/check-source-test-separation.sh` → **exit 0** (the one absolute
      script gate; it passes at HEAD).
- [ ] No doc text added to `crates/cb-train/src/bootstrap.rs` contains the literal
      `.sum()` or `.fold(0.0` (§4.13 / C2-6) — the D-08 backstop greps comments, and
      `bootstrap.rs` is currently clean, which is the only reason its `B8` gate is empty.
- [ ] ROCm: `device_bootstrap_parity_test`, the device arm of
      `bootstrap_dev_oracle_test` (MVS 3/3 — **projected**, and probed early in TASK-02;
      if it cannot hold while the CPU arm does, `MVS-S8`'s device-only-residual
      escalation applies), `cb-backend --lib mvs`, and the four other device suites all
      green; the refreshed device-vs-CPU MVS figure recorded (was `4.703e-11` at
      `2c14d7f`).
- [ ] `cargo test -p cb-backend --lib mvs` on the **default `cpu` feature** → the new
      non-device-gated mirror test passes (AC-6's second half; MAJOR-3).
- [ ] No production `unwrap`/`expect`/`panic`/raw index added; no new `#[cube]`
      kernel; no executable change to `mvs_device.rs` (comment lines only); no `anyhow`.
- [ ] `git worktree list | grep -c 'mvs-red'` prints **0** and `$TD` is removed (the
      6 pre-existing worktree entries are baseline — C2-7); `git stash list` is empty
      (MINOR-4).
- [ ] `progress.md` reflects the final state with per-task evidence, and
      `SPEC.md`'s ten `implementation_state` fields are flipped to `implemented`
      with their measured evidence inline.
