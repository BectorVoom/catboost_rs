---
plan: 9
task_id: TASK-09
phase: mvs-tree2-parity
status: pending
order: 9
wave: serial (v2)
hardware: none (CPU) + ROCm re-confirmation recommended
depends_on: [TASK-02, TASK-03, TASK-06, TASK-07, TASK-08]
blocks: []
specifications: [MVS-S9]
parallelizable: false
revision_note: >
  v2: the tally states the 4 ignored tests (MINOR-6); every "clippy clean" / "gate scripts
  pass" criterion becomes differential against the measured HEAD baselines (CRITICAL-1,
  CRITICAL-2) — note `check-no-raw-float-sum.sh` names `boosting.rs:1649`, a file this very
  task requires byte-unchanged, so the v1 criterion was self-contradictory; the expected
  changed-file set gains `mvs_device.rs` (comment-only, MINOR-1) and the worktree/stash
  restore proof (MINOR-4).
---

# Task 9: Frozen fixtures and the shared draw accounting stay invariant — phase sign-off

## Objective

Prove that the phase achieved its result by fixing the sampler and **nothing else**:
the three frozen fixture roots are byte-unchanged, the four shared draw-accounting
sites are untouched, `calculate_threshold`'s algorithm is unchanged, and the whole
regression surface is at its known baseline plus exactly the four new tests.

**Observable completion condition:**
`git status --short crates/cb-oracle/fixtures/bootstrap crates/cb-oracle/fixtures/bootstrap_dev crates/cb-oracle/fixtures/inputs`
is EMPTY, and `cargo test -p cb-train --no-fail-fast` reports
**507 passed / 1 failed / 4 ignored** with the single failure being the pre-existing
`monotone_non_symmetric_and_region_are_typed_errors`. The **4 ignored** are pre-existing and
must be stated (MINOR-6): the HEAD tally is `503 passed / 1 failed / 4 ignored`
`[VERIFIED: RUN]`, and omitting the ignored count invites a matching run being misread as a
discrepancy.

## Specification references

- `MVS-S9` — frozen fixtures and the shared draw accounting stay invariant.
  Principal failure reason: *the fix is "achieved" by regenerating a frozen fixture or
  by re-tuning shared draw constants, destroying the independent evidence that made
  this root-cause provable.*

## Prerequisites and blocking

- Prerequisites: **TASK-02, TASK-03, TASK-06, TASK-07, TASK-08** — every task that
  writes a file. TASK-01, TASK-04, TASK-05 are transitive prerequisites.
- Blocks nothing; this is the phase close-out.
- **Not parallelisable** — it measures the final state.

## Context and evidence

### The frozen roots

| root | status | why |
|---|---|---|
| `crates/cb-oracle/fixtures/bootstrap/**` | FROZEN | the `boost_from_average=True` family; `bootstrap_oracle_test`'s 5 tests are the blocking gate, incl. the value-sensitive 3-tree `bootstrap_oracle_bayesian` |
| `crates/cb-oracle/fixtures/bootstrap_dev/**` | FROZEN | the bias-0 family; `bootstrap_dev/mvs` is the worst case that proves `MVS-S2` **without regeneration** |
| `crates/cb-oracle/fixtures/inputs/**` | FROZEN | the shared `bootstrap_multiblock` 1500×4 dataset both families and the new `mvs_seeds` family load |

`[VERIFIED: research.md §5.5, §8.3; SPEC MVS-S9]`. `crates/cb-oracle/fixtures/mvs_seeds/**`
is the ONLY new fixture root this phase may add (TASK-03).

### The four shared draw-accounting sites — must be byte-unchanged

| site | location at HEAD | independently verified by |
|---|---|---|
| `PRE_TREE_DRAWS = 2` | `crates/cb-train/src/boosting.rs:59` `[VERIFIED: grep]` | the instrumented trace (`train.cpp:208,211`) |
| `POST_TREE_EXTRA_DRAWS = 2` | `crates/cb-train/src/boosting.rs:69` `[VERIFIED: grep]` | `tree_rng_end.cc − tree_rng_pre_leaf.cc == 2`, 12/12 confirmations across all 4 scenarios × 3 trees |
| `replay_grow_draws` | `crates/cb-train/src/device_draw_replay.rs:64-85` `[VERIFIED: CodeGraph — 7 callers in boosting.rs, tests in device_draw_replay_test.rs]` | `WR01-S7`: matched the REAL grower's `raw_state()` across 5 shapes, border-less features, 4 consecutive trees |
| `select_level_perturbed`'s draw shape | `crates/cb-train/src/tree.rs` | the instrumented RSM / `SelectBestCandidate` trace |

Plus `calculate_threshold`'s algorithm (`crates/cb-train/src/bootstrap.rs:209-273`) —
`MVS-S4`'s non-goal. Changing any of the four re-breaks Bayesian (SPEC `R1`).

### Known pre-existing reds — record, never chase

The authoritative table is `PLAN.md` §4.11 (`B1` … `B10`), all re-measured at HEAD
`2c14d7f` `[VERIFIED: RUN]`. The load-bearing ones for this task:

1. `cb-train`: `monotone_non_symmetric_and_region_are_typed_errors`
   (`crates/cb-train/tests/monotone_oracle_test.rs:286`). Baseline **503 passed /
   1 failed / 4 ignored**.
2. `cb-backend --lib` under the default `cpu` feature: **173 passed / 60 failed**
   (~338 s) — the CubeCL-CPU MLIR `plane_inclusive_sum` limitation.
3. **`cargo clippy -p cb-train --all-targets` is RED and ABORTS** at
   `crates/cb-oracle/src/model_json.rs:161:17` before reaching cb-train's own test targets
   (clippy lints path dependencies). Error-attributed with `--no-deps --keep-going`:
   **100 errors across exactly 10** pre-existing `cb-train` integration test files that lack
   the file-level `#![allow(...)]` (`B4`; v2 said 14 — it miscounted warning-only files).
   **Never** use the bare command, and never require cleanliness.
4. `clippy_error_files -p cb-backend --lib`: exactly **4 errors across 3 files**
   (`cpu_runtime.rs` ×2 at `:696:13`/`:1025:29`, `kernels/bootstrap_device.rs:230:28`,
   `kernels/exact_quantile.rs:178:8`); `--all-targets` adds 2 lib-test errors
   (`kernels/gradient.rs:18`, `kernels/score_split.rs:374`).
4b. **`cargo clippy --workspace --lib -- -D warnings`** — the gate CI runs
   (`.github/workflows/ci.yml:47`) — is **RED at HEAD**, exit 101, aborting at
   `could not compile 'cb-data' (lib) due to 4 previous errors`; error-attributed it has
   **5 errors** and **100 warnings** that `-D warnings` promotes, exactly one of which names
   a phase file (`kernels/mvs_device.rs:80` `manual_rotate`) (`B11`).
4c. **Two pre-existing warnings sit inside the phase's own edit surface** and are NOT
   fixable here: `crates/cb-train/src/bootstrap.rs:134` `excessive_precision` (trimming the
   literal would break Bayesian parity — `bootstrap.rs:110-116`) and
   `crates/cb-backend/src/kernels/mvs_device.rs:80` `manual_rotate` (`B12`). This is why
   every clippy check in this phase selects **errors**, never `-->` lines.
5. **`bash scripts/check-no-raw-float-sum.sh` exits 1** (15 files / 36 lines) and
   **`check-no-anyhow.sh` exits 1** (12 files / 25 lines). `check-source-test-separation.sh`
   exits 0 and IS an absolute gate.
   **Critical interaction:** the float-sum script names
   `crates/cb-train/src/boosting.rs:1649` — a file THIS task requires byte-unchanged — so
   v1's "the three gate scripts pass" criterion **directly contradicted `MVS-S9`**. It has
   been replaced by the diff-scoped form.
6. `cb-backend` build warnings `kernels.rs:645`, `:673`
   (`float_literal_f32_fallback`, future-incompat, not errors).

### The expected new-test arithmetic

| task | new CPU-visible test | count |
|---|---|---|
| TASK-01 | `mvs_bootstrap_consumes_exactly_one_main_stream_draw` (lib) | +1 |
| TASK-03 | `mvs_seeds_cpu_matches_upstream_across_seeds_and_bias` (integration) | +1 |
| TASK-04 | `mvs_block_sample_size_reproduces_upstream_float_expression` (lib) | +1 |
| TASK-05 | `mvs_sample_weights_are_f32_representable` (lib) | +1 |

⇒ **503 + 4 = 507 passed / 1 failed / 4 ignored** for `cb-train`. TASK-03's
`#[cfg(any(rocm, cuda))]` skip companion does not count on the default feature, and the
**4 ignored are pre-existing and unchanged** (MINOR-6).

TASK-06's `cpu_reference_mirrors_cb_train_mvs_transcription` is a **fifth** new test but
lives in `cb-backend`, so it does NOT enter this tally — it raises the `cb-backend --lib`
count instead (baseline 173 passed / 60 failed → **174 passed / 60 failed**).

If a task added a test beyond these five, re-derive both tallies by name and record the
derivation rather than adjusting the target silently.

`cargo test -p cb-train --lib bootstrap` should show **10** tests (7 baseline
`[VERIFIED: RUN --list]` + 3 new lib tests).

## Files

**None** — this is a verification and record-keeping task. Its only writes are
`.planning/plans/mvs-tree2-parity/progress.md` and the `implementation_state` /
evidence lines of `.planning/plans/mvs-tree2-parity/SPEC.md`.

## TDD sequence

### 1. Red

The falsifying checks come first, before any tally is claimed. Run and capture:

```
git status --short crates/cb-oracle/fixtures/bootstrap \
                   crates/cb-oracle/fixtures/bootstrap_dev \
                   crates/cb-oracle/fixtures/inputs
```

- **Expected: EMPTY.** Any output is an immediate phase failure (SPEC `R4`): a frozen
  fixture was regenerated, destroying the independent evidence. Remedy is
  `git checkout -- <path>` followed by re-running the oracle that "needed" the
  regeneration — and if it then fails, that is a genuine finding to escalate, not a
  reason to keep the regenerated fixture.

Then the shared-accounting checks:

```
git diff crates/cb-train/src/device_draw_replay.rs
git diff crates/cb-train/src/tree.rs
git diff -U0 crates/cb-train/src/boosting.rs
grep -n "PRE_TREE_DRAWS: usize = 2\|POST_TREE_EXTRA_DRAWS: usize = 2" crates/cb-train/src/boosting.rs
git diff crates/cb-train/src/bootstrap.rs | grep -n "calculate_threshold" 
```

- **Expected:** the first two EMPTY; `boosting.rs` EMPTY (no task in this plan edits
  it); both constants still `2`; and the `bootstrap.rs` diff shows changes to
  `calculate_threshold`'s CALL SITE arguments only — never to its body
  (`:209-273`). Record each.

### 2. Green

Run the full regression surface and reconcile every number against the recorded
baseline.

CPU (default `cpu` feature):

- `cargo test -p cb-train --no-fail-fast`
  → expect **507 passed / 1 failed / 4 ignored**; the single failure must be
  `monotone_non_symmetric_and_region_are_typed_errors`. If the count differs,
  enumerate the delta test-by-test before accepting it. Note `cargo` prints one
  `test result:` line per target, so sum them — e.g.
  `cargo test -p cb-train --no-fail-fast 2>&1 | grep -E "^test result"` and add the
  passed/failed/ignored columns (that is how the 503/1/4 baseline was measured).
- `cargo test -p cb-train --lib bootstrap` → 10 passed.
- `cargo test -p cb-train --test bootstrap_oracle_test` → 5 passed (the frozen
  blocking gate, incl. `bootstrap_oracle_bayesian`).
- `cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture` → 1 passed,
  four `3/3 trees` lines.
- `cargo test -p cb-train --test mvs_seeds_oracle_test -- --nocapture` → 1 passed, ten
  scenario lines.
- `cargo test -p cb-train --test regularization_oracle_test` → green (the
  Bayesian / `random_strength` draw path must not have moved).
- `cargo test -p cb-train --test yetirank_pairwise_tree_rng_oracle_test` → green (the
  other RNG-phase oracle; SPEC `R1`).
- `cargo test -p cb-train --test multidim_sampling_regression_test` → green (the MVS
  multi-dim smoke).
- `cargo test -p cb-backend --lib --no-fail-fast` → **174 passed / 60 failed** (baseline
  173/60 plus TASK-06's new non-device-gated mirror test). Confirm the FAILING set is
  IDENTICAL to the baseline (no new name) — the pass count is expected to rise by exactly 1.
- `cargo test -p cb-backend --lib cpu_reference_mirrors` → 1 passed. `AC-6`'s GPU-free half.

Lints and gates — **DIFFERENTIAL, per `PLAN.md` §4.12. Do NOT assert cleanliness or
"passes"; four of these are red at HEAD for pre-existing reasons.**

```bash
# PLAN.md §4.12's helper — ERROR-attributed. jq is present (/usr/bin/jq, jq-1.8.1).
clippy_error_files() {
  cargo clippy "$@" --no-deps --keep-going --message-format=json 2>/dev/null \
    | jq -r 'select(.reason=="compiler-message") | .message
             | select(.level=="error") | .spans[]? | select(.is_primary) | .file_name'
}
PHASE='src/bootstrap\.rs|bootstrap_test\.rs|mvs_seeds_oracle_test\.rs|mvs_device'

# THE GATE — all four must be EMPTY. Each measured EMPTY at HEAD, so a hit is this phase's.
clippy_error_files -p cb-train  --all-targets | grep -E "$PHASE"
clippy_error_files -p cb-backend --lib        | grep -E "$PHASE"
bash scripts/check-no-raw-float-sum.sh 2>&1   | grep -E "$PHASE"
bash scripts/check-no-anyhow.sh       2>&1    | grep -E "$PHASE"

# Reference distributions (PLAN.md §4.11 B4/B5) — for attributing a hit, not a pass/fail gate:
clippy_error_files -p cb-train  --all-targets | sort | uniq -c   # 100 errors / 10 files
clippy_error_files -p cb-backend --lib        | sort | uniq -c   #   4 errors /  3 files
bash scripts/check-no-raw-float-sum.sh 2>&1 | grep -c "^D-08 violation:"   # 15
bash scripts/check-no-anyhow.sh        2>&1 | grep -c "^D-14 violation:"   # 12

# The ONE absolute script gate.
bash scripts/check-source-test-separation.sh    # exit 0: "OK: no inline #[cfg(test)] module bodies…"
```

**v2's four set-equality commands are withdrawn (C2-2): none of them could match its own
recorded targets** — `grep -cE "^error"` yields 110 and 5 (not 100 and 4) because of the
`error: could not compile …` aggregate lines, and the `uniq -c` over `-->` lines yields
~40 warning-polluted files (not 14). The two `uniq -c` lines above are the corrected,
twice-reproduced distributions and exist only to help attribute a hit.

Never run bare `cargo clippy -p cb-train --all-targets` (it aborts on the `cb-oracle`
dev-dependency), never omit `--keep-going`, and never grep `-->` lines (they include
warnings — `bootstrap.rs:134` and `mvs_device.rs:80` both warn pre-existingly, `B12`).

Device re-confirmation (recommended; TASK-08 already ran the full set — re-run the two
cheapest as a final state check):

- `cargo test -p cb-backend --no-default-features --features rocm --lib mvs -- --test-threads 1`
- `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1`

### 3. Refactor

No code. Instead, close out the records:

1. **`progress.md`** — fill every measurement row with the OBSERVED value (not
   "expected"), flip all nine task rows to DONE with their evidence one-liners, set
   `status: implemented`, and update the blockers section.
2. **`SPEC.md`** — flip each of the ten `implementation_state:` fields from
   `unimplemented` to `implemented` and append the measured evidence inline, matching
   the sibling phase's convention (`.planning/plans/device-bootstrap-parity/SPEC.md:714,
   746, 793` show the shape: `implemented — <measured figure>`). Leave
   `document_state: draft` alone unless the user approves promotion.
3. Confirm the new-test arithmetic derivation is written down, so a future reader can
   re-derive 507 from 503.

### 4. Verify

- Run: `git status --short crates/cb-oracle/fixtures/` → shows ONLY `mvs_seeds/**`.
- Run: `git status --short` → the changed set is exactly:
  `crates/cb-train/src/bootstrap.rs`, `crates/cb-train/src/bootstrap_test.rs`,
  `crates/cb-train/tests/bootstrap_dev_oracle_test.rs`,
  `crates/cb-train/tests/mvs_seeds_oracle_test.rs` (new),
  `crates/cb-oracle/generator/gen_fixtures.py`,
  `crates/cb-oracle/fixtures/mvs_seeds/**` (new),
  `crates/cb-backend/src/kernels/mvs_device_test.rs`,
  `crates/cb-backend/src/kernels/mvs_device.rs` (**comment lines only**, MINOR-1),
  `.planning/plans/device-bootstrap-parity/{progress.md,SPEC.md}`,
  `.planning/plans/mvs-tree2-parity/**`.
  **Anything else is scope leakage** — investigate and report it.
- Run: `git diff --stat` and confirm no file outside that list appears.
- Run: the three `MVS-S7` greps, **scoped** as the amended spec requires:

  ```
  for p in "never trees 0 or 1" "divergence enters when tree 2" "MVS tree-2 sampling gap"; do
    grep -rn "$p" crates/ .planning/ | grep -v "^\.planning/plans/mvs-tree2-parity/"
  done
  ```

  → must be EMPTY (MINOR-2: hits inside this phase's own artifacts are refuted quotations
  and are explicitly permitted by `MVS-S7`/`AC-7` v2).
- Run: `grep -rn "MVS_GATED_TREES\|MVS_SCENARIO\|gated_trees" crates/` → no output.
- Confirm: every `AC-1` … `AC-10` row in `PLAN.md` §3 has a recorded measurement in
  `progress.md`.
- Confirm: no production `unwrap`/`expect`/`panic`/raw index was added —
  `git diff crates/cb-train/src/bootstrap.rs | grep -nE "unwrap\(|expect\(|panic!|\[[a-z_]+\]"`
  reviewed by eye (the `if let Some(slot)` guards must still be the access form).
- Confirm: no new `#[cube]` kernel and **no EXECUTABLE change** to
  `crates/cb-backend/src/kernels/mvs_device.rs` (MINOR-1 replaces v1's "must be EMPTY"):

  ```
  git diff -U0 crates/cb-backend/src/kernels/mvs_device.rs \
    | grep -E "^[+-]" | grep -v "^[+-][+-]" | grep -vE "^[+-]\s*(//|///|//!)"    # must be EMPTY
  ```

- Run: `git worktree list | grep -c 'mvs-red'` → **0**, and
  `ls -d /home/user/Documents/workspace/catboost_rs-worktrees/.target-mvs-red` → absent.
  The **6** pre-existing worktree entries (main tree,
  `catboost_rs-worktrees/23-ctr-model-loading`, four `.claude/worktrees/agent-*`) are
  baseline `[VERIFIED: RUN]` — do not require "no leftover entry" (C2-7).
  `git stash list` → EMPTY (MINOR-4).

## Completion criteria

- [ ] The three frozen fixture roots are byte-unchanged (`git status` EMPTY).
- [ ] `device_draw_replay.rs`, `tree.rs` and `boosting.rs` are byte-unchanged; both
      draw constants still `2`; `calculate_threshold`'s BODY unchanged.
- [ ] `cargo test -p cb-train --no-fail-fast` → **507 passed / 1 failed / 4 ignored**
      (MINOR-6 — state the ignored count), the failure being the known `monotone…` red,
      with the 503 → 507 derivation written down.
- [ ] `bootstrap_oracle_test` 5/5; `bootstrap_dev_oracle_test` four × 3/3;
      `mvs_seeds_oracle_test` 10/10; `regularization_oracle_test`,
      `yetirank_pairwise_tree_rng_oracle_test`, `multidim_sampling_regression_test`
      green.
- [ ] `cb-backend --lib` (cpu) → **174 passed / 60 failed**, the failing set IDENTICAL to
      the 60-name baseline; the +1 pass is TASK-06's mirror test.
- [ ] **Differential lint gates** (CRITICAL-1/2, C2-1/C2-2): all four **error-attributed**
      diff-scoped checks EMPTY; `check-source-test-separation.sh` exits 0. Reference
      distributions for attribution: `PLAN.md` §4.11 `B4` (**100 errors / 10 files**), `B5`
      (**4 errors / 3 files**), `B8` 15 headers, `B9` 12 headers. **Do NOT** assert clippy
      cleanliness, do NOT assert `check-no-raw-float-sum.sh` / `check-no-anyhow.sh` pass
      (the former names `boosting.rs:1649`, which this task requires byte-unchanged), and do
      NOT grep `-->` lines (they include the two unfixable warnings in `B12`).
- [ ] No `.sum()` / `.fold(0.0` literal was added to `crates/cb-train/src/bootstrap.rs`
      doc text (C2-6) — verified by the `B8` diff-scoped grep still being EMPTY.
- [ ] The rocm re-confirmation pair is green.
- [ ] The changed-file set matches the expected list exactly (now including the
      comment-only `mvs_device.rs`) — no scope leakage; no leftover worktree; empty stash.
- [ ] `progress.md` fully filled with OBSERVED values; `SPEC.md`'s ten
      `implementation_state` fields flipped to `implemented` with evidence.

## Completion evidence to record in `progress.md`

- The `git status`/`git diff --stat` transcripts for the frozen roots, the four
  shared-accounting sites, and the whole changed-file set.
- The `cargo test -p cb-train --no-fail-fast` tally (**passed / failed / ignored**) with
  the 503 → 507 derivation.
- The `cb-backend --lib` cpu tally (174/60) and confirmation that the 60 failing names are
  unchanged.
- All four diff-scoped grep results and the four set-equality counts.
- The `check-source-test-separation.sh` output.
- The comment-only `git diff` proof for `mvs_device.rs`, the `git worktree list` and
  `git stash list` outputs.
- The rocm re-confirmation results.

## Risks and guardrails

- **SPEC `R4` — regenerating a frozen fixture.** The single most damaging failure mode
  in this phase: it would make the oracle agree with a wrong sampler and erase the
  evidence that proved the root cause. Guard: the FIRST thing this task runs.
- **SPEC `R1` — re-tuning shared draw accounting.** Guard: the `git diff` checks plus
  the two RNG-phase oracles.
- **SPEC `R5` — chasing pre-existing reds.** The `monotone…` failure, the 60
  `cb-backend` cpu MLIR failures, the 100 `cb-train` + 4 `cb-backend` clippy errors, and the
  two red gate scripts are ALL baseline. Record them; never "fix" them inside this phase.
  Satisfying `check-no-raw-float-sum.sh` and `check-no-anyhow.sh` literally would mean
  editing ~25 files across five crates — including `boosting.rs`, which this very task
  freezes. If a cleanup is genuinely wanted, it is a separate phase.
- **A self-contradictory Definition of Done (CRITICAL-2).** v1 required both "the three
  gate scripts pass" AND `boosting.rs` byte-unchanged; those cannot both hold. If any
  criterion in this phase ever looks like it needs a frozen file edited to satisfy a lint,
  the criterion is wrong — check `PLAN.md` §4.12 for the differential form rather than
  editing the file.
- **A drifting tally accepted without derivation.** If the count is not 507, the
  temptation is to update the target. Instead enumerate the delta by name — an
  unexplained extra pass is as suspicious as an extra failure (e.g. a test that was
  supposed to be `#[cfg]`-gated running on the wrong feature).
- **Scope leakage.** The expected changed-file list is deliberately exhaustive. A file
  outside it means some task exceeded its brief; report it rather than folding it in.
- **Declaring the phase done without the device evidence.** `AC-6` and `AC-8` are
  rocm-only. A CPU-only sign-off would leave the two f32 mirrors and the device parity
  claim unverified — exactly the invisible-on-default-CI failure mode `MVS-S6` exists
  to prevent.
