---
plan: 2
task_id: TASK-02
phase: mvs-tree2-parity
status: pending
order: 2
wave: serial (v2 — Wave B dissolved, PLAN-CHECK CRITICAL-3)
hardware: CPU + local ROCm gfx1151 (for the early device probe, MAJOR-2)
depends_on: [TASK-01]
blocks: [TASK-07, TASK-09]
specifications: [MVS-S2]
parallelizable: false
parallel_with: []
revision_note: >
  v2: the controlled-revert Red moves into a throwaway `git worktree` at 2c14d7f
  (CRITICAL-3 — v1 mutated `bootstrap.rs`, the file TASK-04 owned in the same wave);
  `git stash` on that path is forbidden; a byte-identity restore gate is added (MINOR-4);
  an early ROCm probe of the device arm is added (MAJOR-2); the clippy gate becomes
  diff-scoped (CRITICAL-1).
---

# Task 2: MVS is gated over ALL trees — remove the `MVS_GATED_TREES` carve-out

## Objective

After this task `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` gates **all
four** `bootstrap_dev` scenarios — including `mvs` — over the fixture's full
**3/3** trees at ≤1e-5, and the `gated_trees` concept does not exist in the file
any more.

**Observable completion condition:**
`cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture` prints
`over 3/3 trees` for `no`, `bayesian`, `bernoulli` AND `mvs`, and
`grep -n "MVS_GATED_TREES\|MVS_SCENARIO\|gated_trees" crates/cb-train/tests/bootstrap_dev_oracle_test.rs`
returns nothing.

## Specification references

- `MVS-S2` — MVS matches upstream over ALL trees on the bias-0 family.
  Principal failure reason: *the MVS sample for trees ≥ 1 is drawn from the wrong
  RNG phase, so a split argmax flips.*

## Prerequisites and blocking

- Prerequisite: **TASK-01** (the deletion). Without it this task's claim is false —
  research measured the exact failure it produces.
- Blocks TASK-07 (the `MVS-S7` grep cannot pass while the superseded 30-line
  comment is still in the file) and TASK-09.
- **NOT parallelisable (revised, CRITICAL-3).** v1 marked this task
  `parallelizable: true` alongside TASK-03 and TASK-04 on the strength of an ownership
  table that listed only `bootstrap_dev_oracle_test.rs`. That table was **wrong**: this
  task's Red procedure also WRITES `crates/cb-train/src/bootstrap.rs` — the file TASK-04
  owns — and `planning/settings.json` has `"use_worktree": false`, so all tasks share one
  working tree. v2 dissolves the wave; the phase is serial and this task's Red runs in an
  isolated throwaway worktree (step 1 below).
- **This task must complete BEFORE TASK-04 (MAJOR-1).** Its Red binds pre-fix diagnostic
  float values that were measured with `bootstrap.rs:312` still in `f64`. v1 claimed
  TASK-04 "provably cannot invalidate" them; that claim is **withdrawn** — research
  §8.5/§8.6 measured only the POST-fix combinations, never defect-present + f32-target,
  and the +1.788e-5 target shift at `block_size = 1500` is exactly the kind of
  perturbation that can move a near-tied argmax. Capturing the Red at `2c14d7f` (where
  the f32 target is absent by construction) removes the exposure entirely.

## Context and evidence

All line numbers re-read at HEAD `2c14d7f` this session.

- `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` is 383 lines. The carve-out
  machinery to remove `[VERIFIED: read in full]`:
  - `:108-116` — `SCENARIOS`, whose doc says "MVS is deliberately NOT here";
    contains `no`, `bayesian`, `bernoulli`.
  - `:118-119` — `MVS_SCENARIO: (&str, EBootstrapType, f64, f32) = ("mvs", EBootstrapType::Mvs, 0.8, 0.0)`.
  - `:121-155` — the **30-line doc comment** encoding the WRONG diagnosis, and
    `const MVS_GATED_TREES: usize = 2;` at `:155`.
  - `:157-169` — `gate_against_upstream`'s doc + signature, whose fifth parameter
    is `gated_trees: usize` (`:168`).
  - `:174-186` — the truncation arithmetic driven by `gated_trees`
    (`split_end`, `leaf_end`, `staged_end`, plus the
    `assert!(gated_trees <= n_trees, …)` guard at `:175-178`).
  - `:216-219` — the printout `"… over {gated_trees}/{n_trees} trees"`.
  - `:234-237` and `:344-347` — the two identical
    `SCENARIOS.iter().map(|&s| (s, 3usize)).chain(std::iter::once((MVS_SCENARIO, MVS_GATED_TREES)))`
    constructions, in the CPU test (`bootstrap_dev_cpu_matches_upstream`, `:225-255`)
    and the device test (`bootstrap_dev_device_matches_upstream`, `:259-382`).
- **Only these six symbols reference the carve-out repo-wide**
  `[VERIFIED: grep -rn "MVS_GATED_TREES\|MVS_SCENARIO" --include="*.rs" crates/` →
  `bootstrap_dev_oracle_test.rs:111, 119, 155, 161, 237, 347` and nothing else]`.
  No other test, no production file, imports them.
- **The recorded Red** `[VERIFIED: research.md §8.1]`, obtained at HEAD by setting
  `MVS_GATED_TREES = 3`:

  ```
  [cpu] bootstrap_dev/no|bayesian|bernoulli: … 3/3 trees   (ok)
  panicked … [cpu] bootstrap_dev/mvs: splits diverged from upstream:
    StageDiverged { stage: Splits, index: 5,
                    expected: -0.025514747947454453,
                    actual: -0.2692405581474304, diff: 0.2437258101999760 }
  ```

  The panic text comes from `gate_against_upstream`'s
  `.unwrap_or_else(|e| panic!("[{who}] {dir}: splits diverged from upstream: {e:?}"))`
  at `:190-191`, and `StageDiverged` is the real
  `cb_oracle::OracleError` variant (`crates/cb-oracle/src/error.rs:46-58`) produced
  by `compare_stage`'s `1e-5` wrapper (`crates/cb-oracle/src/compare.rs:85`)
  `[VERIFIED: read]`. The trailing `diff` digits may print one ulp differently — the
  `index: 5` and the two values are the load-bearing part.
- **No fixture work is required.** `crates/cb-oracle/fixtures/bootstrap_dev/mvs/`
  is already committed at `2c14d7f` with `config.json`, `model.json`,
  `predictions.npy`, `staged.npy` `[VERIFIED: ls]`, and is already the bias-0
  worst case (0/5 seeds passed pre-fix). It is a **FROZEN** root — this task must
  not regenerate it.
- **Each scenario already loads its OWN borders** at `:239-241` (CPU) and
  `:349-351` (device) via `load_model_json(...).float_feature_borders()` — the
  invariant SPEC `R3` demands. Preserve that; do not hoist a shared border set out
  of the loop.
- **Post-fix residuals** across every configuration: `max|Δleaf| ∈ [5.9e-9, 6.9e-9]`,
  `max|Δstaged| ∈ [1.6e-8, 2.4e-8]` — three orders inside the ≤1e-5 bar
  `[VERIFIED: research.md §8.2]`. So the 3/3 claim has ~3 orders of margin, not a
  hairline pass.
- **Baseline at HEAD**: `bootstrap_dev_cpu_matches_upstream ok` in 0.21 s
  `[VERIFIED: RUN this session]`.

## Files

- Modify: `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` — the only file.
- Do NOT touch: `crates/cb-oracle/fixtures/bootstrap_dev/**` (frozen), any
  production source, or `crates/cb-train/tests/device_bootstrap_parity_test.rs`.

## TDD sequence

### 1. Red — by controlled revert in an ISOLATED WORKTREE (revised, CRITICAL-3 / MINOR-4)

The production change is a prerequisite here, so the falsifying run must be produced
deliberately. **`git stash` on `crates/cb-train/src/bootstrap.rs` is FORBIDDEN** — v1's
instruction to stash that path could destroy TASK-04's in-flight seam extraction and, on
an interleaved pop, silently leave the two fabricated draws in place while every CPU
oracle still passed (they passed *with* the defect for 3 of 5 bias-true seeds). Instead
the Red is captured in a throwaway worktree at `2c14d7f`, which has the defect present
**and** TASK-04's f32 target absent — exactly the state the recorded values were measured
in (MAJOR-1).

1. Make the test change first (step 2 below) in the MAIN tree, leaving `bootstrap.rs` as
   TASK-01 left it. Run the suite — it should PASS. That alone proves nothing.
2. Create the isolated baseline **on disk, never `/tmp`** (C2-3 — `/tmp` is a 16 GB
   RAM-backed tmpfs and no `CARGO_TARGET_DIR` is set, so v2's `W=/tmp/…` would have built
   6.6 GB into tmpfs), and carry ONLY this task's test change into it:

   ```bash
   WT=/home/user/Documents/workspace/catboost_rs-worktrees/mvs-red-task02
   TD=/home/user/Documents/workspace/catboost_rs-worktrees/.target-mvs-red

   df -BG --output=avail,fstype /home | tail -1     # need >= 15G; fstype must NOT be tmpfs
   git worktree add --detach "$WT" 2c14d7f
   mkdir -p "$TD"
   git diff -- crates/cb-train/tests/bootstrap_dev_oracle_test.rs > /tmp/task02-test.patch
   git -C "$WT" apply /tmp/task02-test.patch
   ```

   (If TASK-01 is already committed, produce the patch with
   `git diff 2c14d7f -- crates/cb-train/tests/bootstrap_dev_oracle_test.rs` instead.)
   The worktree now holds: upstream fixtures as committed, the carve-out removed, and the
   **defect present**. Nothing in the main tree is touched.
3. Run, inside the worktree, with the disk-backed shared target dir:

   ```bash
   CARGO_TARGET_DIR="$TD" cargo test --manifest-path "$WT/Cargo.toml" \
     -p cb-train --test bootstrap_dev_oracle_test -- --nocapture
   ```

   **Cost, measured** `[VERIFIED: RUN in the v3 plan revision — this exact worktree +
   command was created, executed and removed]`: **59 s** cold (6.6 GB into `$TD`, `/home`
   209 G → 206 G), **4 s** warm on a re-run after a one-line test edit. 59 s is not a hang.
   `$TD` is shared with TASK-03, which therefore pays only the warm cost.
4. **Expected failure — the discriminating-power proof. This exact output was REPRODUCED
   during the v3 plan revision** `[VERIFIED: RUN — worktree at 2c14d7f with
   `MVS_GATED_TREES` set to 3, i.e. the same test-side effect this task's edit has]`:

   ```
   [cpu] bootstrap_dev/no: splits + leaf values + staged within 1e-5 of upstream over 3/3 trees
   [cpu] bootstrap_dev/bayesian: … over 3/3 trees
   [cpu] bootstrap_dev/bernoulli: … over 3/3 trees
   thread 'bootstrap_dev_cpu_matches_upstream' panicked at crates/cb-train/tests/bootstrap_dev_oracle_test.rs:191:29:
   [cpu] bootstrap_dev/mvs: splits diverged from upstream: StageDiverged { stage: Splits,
     index: 5, expected: -0.025514747947454453, actual: -0.2692405581474304,
     diff: 0.24372581019997597 }
   test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
   ```

   The failure must be MVS-only and at split index **5**. The panic comes from
   `gate_against_upstream`'s `.unwrap_or_else(|e| panic!(…))` at `:190-191`. Note the
   measured `diff` is `0.24372581019997597` (research §8.1 recorded
   `0.2437258101999760`, one digit shorter); `index: 5` and the two values are the
   load-bearing part.
5. **Tear down and prove the main tree is untouched** (MINOR-4; scoped per C2-7):

   ```bash
   git worktree remove --force "$WT" && rm -rf "$TD"
   git worktree list | grep -c 'mvs-red'               # MUST print 0 (6 unrelated entries are baseline)
   git stash list                                      # MUST be empty
   git diff --stat -- crates/cb-train/src/bootstrap.rs  # MUST show ONLY TASK-01's change
   ```

   `git worktree list` reports **6 pre-existing entries** at HEAD (main tree,
   `catboost_rs-worktrees/23-ctr-model-loading`, four `.claude/worktrees/agent-*`)
   `[VERIFIED: RUN]`, so "no leftover entry" would be false — scope the check to
   `mvs-red`. The teardown above was executed and verified to leave 6 entries and
   `grep -c mvs-red` = 0.

   Then re-run the suite in the main tree and confirm it is green. **Record every output
   in `progress.md`.**

If step 4 does not fail, the test is vacuous — most likely the `gated_trees`
truncation was not fully removed and MVS is still being compared over 2 trees.
Fix the test, do not accept the pass.

### 2. Green — remove the carve-out

Minimum change, in this order:

1. Fold `mvs` back into `SCENARIOS` (`:112-116`) as a fourth entry
   `("mvs", EBootstrapType::Mvs, 0.8, 0.0)`, and delete the "MVS is deliberately NOT
   here" sentence from its doc.
2. Delete `MVS_SCENARIO` (`:118-119`) and the entire `:121-155` block —
   the 30-line doc comment **and** `const MVS_GATED_TREES`. Do not merely bump the
   constant to `3`: SPEC `MVS-S7` requires the superseded diagnosis to be *removed*,
   not re-tuned.
3. Drop `gate_against_upstream`'s `gated_trees` parameter (`:168`) and the doc
   paragraph that explains it (`:160-162`). Inside the body, replace the truncated
   slices with the full ones: compare `up_splits` vs `our_splits`,
   `up_leaves` vs `our_leaves`, and `expected_staged` vs `staged` in full, and
   delete `split_end` / `leaf_end` / `staged_end` / the `depth` and `n_rows` locals
   (`:179-186`) and the `assert!(gated_trees <= n_trees, …)` guard (`:175-178`)
   **only if** they become unused. Keep `n_trees` if the printout still uses it.
4. Update the printout (`:216-219`) to state the full tree count, e.g.
   `"… within 1e-5 of upstream over {n_trees}/{n_trees} trees"` or simply
   `"over all {n_trees} trees"`. The string must still make a 3/3 claim visible with
   `--nocapture`, because that printout is `AC-2`'s evidence.
5. Simplify both iteration sites (`:234-237`, `:344-347`) to plain
   `for &(scenario, bt, subsample, temp) in SCENARIOS` loops, and update the two
   `gate_against_upstream(...)` call sites (`:253`, `:380`) to drop the last
   argument.
6. **Keep** everything else in the device test untouched: the `CountingGpu`
   anti-false-pass wrapper (`:270-340`), the `gpu.grown.get() == p.iterations`
   assertion (`:364-370`), the `expect_sampled` assertion (`:371-378`), and the
   bias-0 assertion in `gate_against_upstream` (`:210-215`).
7. Update the file's module doc (`:1-23`) only where it is now false. It currently
   makes no MVS carve-out claim, so a light touch is enough; do not rewrite the
   `WR01-S14`/`WR01-S15` framing.

- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture`

### 3. Refactor

- With `gated_trees` gone, `gate_against_upstream` should read as a flat
  four-stage comparison. Remove any now-dead local, but do NOT change what is
  compared or the ε (`compare_stage` is fixed at `1e-5`).
- Keep the per-scenario `float_feature_borders()` load inside the loop — SPEC `R3`.
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test`
- **Clippy — ERROR-attributed and diff-scoped, not "clean"** (CRITICAL-1 / C2-1). Two
  traps: `cargo clippy -p cb-train --all-targets` ABORTS on a dev-dependency
  (`crates/cb-oracle/src/model_json.rs:161`) before reaching cb-train's own test targets,
  **and** grepping `-->` location lines catches warnings too, which is why v2's form was
  red at HEAD `[VERIFIED: RUN]`. Use the severity-filtered helper from `PLAN.md` §4.12:

  ```bash
  clippy_error_files -p cb-train --all-targets | grep "bootstrap_dev_oracle_test"
  ```

  → must be EMPTY. `bootstrap_dev_oracle_test.rs` is NOT among the **10** error files in
  baseline `B4` — it carries the file-level `#![allow(...)]` at `:24` — so any hit naming
  it is genuinely this task's, most likely an unused-const or unused-variable left by an
  incomplete removal.

### 4. Verify

- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture`
  → 1 passed, four `3/3 trees` lines.
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` → 5 passed (frozen
  family unaffected).
- Run: `grep -n "MVS_GATED_TREES\|MVS_SCENARIO\|gated_trees" crates/cb-train/tests/bootstrap_dev_oracle_test.rs`
  → no output.
- Run: `grep -rn "MVS_GATED_TREES\|MVS_SCENARIO" crates/` → no output.
- Run: `git status --short crates/cb-oracle/fixtures/` → EMPTY.
- Confirm: the device test still compiles. Because it is behind
  `#[cfg(any(feature = "rocm", feature = "cuda"))]`, a `cpu` run does not type-check
  it — so run
  `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test --no-run`
  to compile the device arm. **This is mandatory**: both `.chain(...)` sites were
  edited and only the rocm build sees the second one.

### 4b. EARLY DEVICE PROBE — mandatory (new in v2, MAJOR-2)

This task edits the **rocm-gated device arm** at `:347`, creating a new claim —
"the device arm reports 3/3 trees for MVS" (`AC-8`) — that v1 left **unmeasured for five
subsequent tasks**, all of them CPU-only. If it is red, the repo would carry a red device
suite until TASK-08 discovered it, with the failure no longer attributable to this change.

The probe has **no dependency on TASK-06**: the `cb-backend` mirror is a change to a
*test* file (`kernels/mvs_device_test.rs`) in a *different* crate and cannot affect
`bootstrap_dev_oracle_test`. So run it here, while this change is still the only one in
flight:

- Run: `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1`
- **Expected:** four scenarios, each printing `3/3 trees`, each with
  `gpu.grown == 3` (`:364-370`) and the sampled-seam guard satisfied (`:371-378`).
- **Record** the device-arm transcript in `progress.md` regardless of outcome. The
  `2c14d7f` baseline was `no`/`bayesian`/`bernoulli` 3/3 and **`mvs` 2/3**
  `[PROJECT: device-bootstrap-parity/progress.md:166]`.
- **If the MVS device row fails while the CPU arm passes**, this is `MVS-S8`'s SECOND
  failure mode — device/CPU split-argmax disagreement, not draw accounting. It is a live
  mode: the sibling phase measured `split_mismatched_trees = 4/20` on the BASE grower
  `[VERIFIED: device-bootstrap-parity/progress.md:162]`, and
  `compare_stage(Stage::Splits, …)` treats a different split as a **hard failure**, not a
  bounded delta (unlike `device_bootstrap_parity_test.rs:411-417`, which bounds the
  divergent tree's *contribution*). Apply `MVS-S8`'s **pre-defined escalation**: record a
  **device-only documented residual** with the divergent tree index and its bounded
  contribution, and escalate. Do **NOT** loosen a tolerance, do **NOT** touch
  `replay_grow_draws` / `PRE_TREE_DRAWS` / `POST_TREE_EXTRA_DRAWS` (SPEC `R1`), and do
  **NOT** re-introduce a reduced-tree carve-out on the CPU arm — `MVS-S2` is a CPU claim
  and stands on its own.

## Completion criteria

- [ ] The controlled-revert Red was captured **in a throwaway worktree at `2c14d7f`**,
      never by `git stash` on `crates/cb-train/src/bootstrap.rs`, and reproduced
      `StageDiverged { stage: Splits, index: 5, … }` for MVS only, with the other three
      scenarios at 3/3.
- [ ] The worktree was on **disk** (`catboost_rs-worktrees/mvs-red-task02`) with a
      disk-backed `CARGO_TARGET_DIR`, never `/tmp` (C2-3).
- [ ] Restore gate (MINOR-4, scoped per C2-7): `git worktree list | grep -c 'mvs-red'`
      prints **0** (6 unrelated entries are baseline), `$TD` removed, `git stash list` is
      EMPTY, and `git diff --stat -- crates/cb-train/src/bootstrap.rs` shows ONLY
      TASK-01's change.
- [ ] `MVS_GATED_TREES`, `MVS_SCENARIO` and `gated_trees` no longer exist anywhere.
- [ ] `mvs` is a normal member of `SCENARIOS`.
- [ ] The 30-line superseded diagnosis comment is deleted (not edited, not bumped).
- [ ] All four scenarios print a 3/3-tree claim under `--nocapture` (CPU).
- [ ] The device arm compiles under `--no-default-features --features rocm`
      (`--no-run`) **and the early device probe (§4b) was RUN**, its transcript recorded,
      and any MVS-row failure escalated per `MVS-S8` rather than absorbed.
- [ ] The **error-attributed** diff-scoped clippy check for `bootstrap_dev_oracle_test`
      is EMPTY (not "clippy clean", and not a `-->`-line grep — `PLAN.md` §4.12).
- [ ] `bootstrap_oracle_test` still 5/5; fixtures byte-unchanged.

## Completion evidence to record in `progress.md`

- Both CPU `--nocapture` transcripts (worktree-reverted → failing; main tree → four `3/3`).
- The worktree teardown / stash-list / `git diff --stat` restore proof.
- The `grep` proof that the three symbols are gone.
- The rocm `--no-run` compile result **and** the §4b device-arm transcript, with the
  MVS row compared against the `2c14d7f` baseline of 2/3.

## Risks and guardrails

- **A vacuous green.** The single largest risk: if the truncation slices are left
  in place with `gated_trees` hard-coded to `3`, the test passes but the *concept*
  survives, violating `MVS-S2`'s postcondition ("the `gated_trees` concept is gone
  from the file"). Guard: the `grep` in Verify.
- **Editing only the CPU `.chain(...)` site.** There are TWO (`:237`, `:347`) and
  the second is rocm-gated. Guard: the mandatory rocm `--no-run` compile.
- **SPEC R3 — a shared border set.** Do not hoist `float_feature_borders()` out of
  the per-scenario loop as a "simplification"; CatBoost quantization borders are not
  stable across configurations.
- **SPEC R4 — regenerating `bootstrap_dev/mvs`.** Forbidden. The committed fixture
  already proves the claim; a regeneration would destroy the independent evidence.
- **Do not touch the device test's anti-false-pass guards.** `CountingGpu` and the
  `grown`/`sampled_trees` assertions are what stop a silent CPU fallback from making
  the device claim vacuous.
- **`git stash` on `crates/cb-train/src/bootstrap.rs` is forbidden** (CRITICAL-3). One
  shared working tree (`use_worktree: false`) plus a stash on the phase's most important
  production file is how the defect gets silently reintroduced while the oracles stay
  green. Use the worktree.
- **A "clippy clean" criterion is unachievable** (CRITICAL-1) — six v1 tasks carried one.
  Use the diff-scoped grep in Refactor; never assert cleanliness.
