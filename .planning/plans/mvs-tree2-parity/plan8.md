---
plan: 8
task_id: TASK-08
phase: mvs-tree2-parity
status: pending
order: 7
wave: serial (v2 — runs BEFORE TASK-07, PLAN-CHECK MINOR-3)
hardware: local ROCm gfx1151 (required)
depends_on: [TASK-06]
blocks: [TASK-07, TASK-09]
specifications: [MVS-S8]
parallelizable: false
parallel_with: []
plan_file_note: >
  This file stays `plan8.md` (bound to TASK-08) while its execution rank is 7; it runs
  before `plan7.md`/TASK-07. See PLAN.md §1's "Plan-file numbering note".
revision_note: >
  v2: moved ahead of TASK-07 (MINOR-3 — TASK-07 writes `bootstrap.rs` while this task
  compiles `cb-train` seven times under rocm in the same working tree; this task writes
  nothing, so running it first is free). The MVS 3/3 device claim is now explicitly a
  PROJECTED outcome first probed in TASK-02 (MAJOR-2), and `MVS-S8`'s second failure mode
  (device/CPU split-argmax disagreement) plus its pre-defined device-only-residual
  escalation are spelled out.
---

# Task 8: Device-vs-CPU MVS parity survives the fix (ROCm re-verification)

## Objective

Prove, on the local ROCm rig, that the host-side fix has moved the DEVICE numbers
identically to the CPU numbers — device-vs-CPU MVS `max|Δpred|` stays ≤1e-5 and the
device arm of `bootstrap_dev_oracle_test` now reports **3/3** trees for MVS — and that
no other device suite regressed.

**Observable completion condition:** all eight device commands in Verify are green,
the refreshed device-vs-CPU MVS figure is recorded (was `4.703e-11` at `2c14d7f`), and
`git status --short` shows **no source change made by this task**.

## Specification references

- `MVS-S8` — device-vs-CPU MVS parity survives the fix. Principal failure reason:
  *the device branch shares this exact host sampler (`boosting.rs:3262`), so the fix
  moves the device numbers; if any device-side draw accounting silently depended on
  the old 3-draw count, device and CPU would diverge.*
  Scope is **verification only — no edit expected.**

## Prerequisites and blocking

- Prerequisite: **TASK-06** (the `cb-backend` mirror must be in place, or the rocm
  `--lib` run asserts against a stale reference). Transitively also TASK-01, TASK-04,
  TASK-05.
- Blocks **TASK-07** (new in v2 — see below) and TASK-09.
- **Runs BEFORE TASK-07 (revised, MINOR-3).** v1 put the two in a parallel Wave E and
  claimed they were "disjoint by construction" because this task writes nothing. The write
  set was indeed disjoint, but TASK-07 edits `crates/cb-train/src/bootstrap.rs` while this
  task runs seven `cargo test -p cb-train --features rocm --test …` invocations that
  COMPILE that crate in the same working tree — target-dir lock contention and mid-edit
  rebuilds. Since this task writes nothing, sequencing it first costs nothing and removes
  the contention entirely.
- **The MVS 3/3 device claim should already have been probed in TASK-02** (MAJOR-2). If
  TASK-02's §4b probe was skipped, run it as the FIRST thing here and note the attribution
  loss: five tasks will have landed since the change that created the claim.

## Context and evidence

### Why the device is affected at all

`boosting.rs:3218-3291` (the `device_active` branch) does `PRE_TREE_DRAWS` →
**`bootstrap(...)` at `:3262` — the very same host function the CPU branch calls at
`:3833`** → `grow_tree_on_device` → `replay_grow_draws` (`:3607`) →
`POST_TREE_EXTRA_DRAWS`. The comment at `:3222` states it: "The device branch keeps
the ENTIRE sampler on the host"
`[VERIFIED: grep "bootstrap(" crates/cb-train/src/boosting.rs → 3262, 3833;
grep PRE_TREE_DRAWS/POST_TREE_EXTRA_DRAWS → 3233, 3608, 3802, 4715; research.md §6.2]`.
So the fix moves device and CPU numbers **identically by construction** — which is the
desired outcome, but it means every device suite must be re-run.

### What must NOT change

- `replay_grow_draws` (`crates/cb-train/src/device_draw_replay.rs:64-85`) — its own
  doc says `PRE_TREE_DRAWS` / `POST_TREE_EXTRA_DRAWS` and the `bootstrap()` call are
  deliberately NOT replayed because the device branch shares those code paths with
  the CPU branch (`device_draw_replay.rs:38-40`) `[VERIFIED: read via CodeGraph]`. It
  has 7 callers in `boosting.rs` and is covered by
  `crates/cb-train/src/device_draw_replay_test.rs`. **Do not touch it.**
- `PRE_TREE_DRAWS = 2` (`boosting.rs:59`), `POST_TREE_EXTRA_DRAWS = 2` (`:69`). The
  latter's doc records `tree_rng_end.cc − tree_rng_pre_leaf.cc == 2` confirmed 12/12
  across all four bootstrap scenarios and all three trees `[VERIFIED: read
  boosting.rs:61-69]`. **Do not touch either.**

### The device suites and what they lock

| target | test fns | what it asserts |
|---|---|---|
| `device_bootstrap_parity_test` | `wr01_base_device_grower_holds_1e5_vs_cpu`, `wr01_device_sampled_bootstrap_holds_1e5_vs_cpu`, `wr01_device_run_to_run_jitter_within_budget`, `wr01_poisson_is_rejected_identically_on_every_backend` (`:511-533`), plus a `#[cfg(not(...))]` skip printer at `:535-540` | the MVS row lives in `sampled_types_hold_1e5` (`:370-431`) at `(n, nf, depth, iters) = (20000, 16, 6, 20)` with `EPS = 1e-5`; it also carries the two anti-false-pass guards — `sample_len == n` (`:393-396`) and `max|Δpred(sampled, unsampled)| > EPS` (`:422-429`) `[VERIFIED: read]` |
| `bootstrap_dev_oracle_test` (device arm) | `bootstrap_dev_device_matches_upstream` (`:259-382`) | device-vs-UPSTREAM at ≤1e-5; with TASK-02 landed it must now report `3/3 trees` for MVS. Guards: `gpu.grown.get() == p.iterations` (`:364-370`) and `expect_sampled` (`:371-378`) |
| `device_oblivious_parity_probe_test`, `device_seam_test`, `device_nonsym_fit_test`, `device_region_fit_test`, `device_bootstrap_speed_test` | — | the other device paths that must not regress `[VERIFIED: ls crates/cb-train/tests/]` |
| `cb-backend --lib` (rocm) | incl. the three `mvs` self-oracles | TASK-06's mirror |

### Build-line convention (non-negotiable)

Always `--no-default-features --features rocm`; always `--test <target>` for
`cb-train` (a blanket rocm test build fails on ~37 files importing `CpuBackend`).
`cb-backend --lib` under rocm builds as a whole.
`[PROJECT: .planning/plans/device-bootstrap-parity/plan10.md:84-117, plan2.md:98-127,
plan6.md:116-149; research.md §8.8]`

Local rig present: gfx1151 / AMD Radeon 860M, `rocminfo` at
`/home/user/rocm/opt/rocm/bin/rocminfo` `[VERIFIED: RUN]`.

### Baselines to beat / match

- device-vs-CPU `max|Δpred|` at `20000×16 d6`: Bernoulli `5.589e-11`, Bayesian
  `5.477e-11`, **MVS `4.703e-11`** at `2c14d7f`
  `[PROJECT: .planning/plans/device-bootstrap-parity/progress.md:167]`.
- device-vs-upstream on the bias-0 family at `2c14d7f`: `no`/`bayesian`/`bernoulli`
  3/3 trees, **`mvs` 2/3** (the carve-out). After this phase it must be **4 × 3/3**
  `[PROJECT: progress.md:166]`.
- run-to-run jitter budget `≤1e-7`, measured `0.000e0` on both ROCm and T4
  `[PROJECT: progress.md:170]`.
- `research.md §12 MEDIUM #4` flags that the rocm run was **NOT executed** in the
  research session — this task is the first real device evidence for the fix. Treat a
  surprise here as a genuine finding, not a flaky run.

## Files

**None.** `MVS-S8`'s scope is "verification only — no edit expected". If a device test
genuinely needs an edit, that is a NEW finding: stop, record it, and escalate rather
than absorbing a source change into a verification task.

## TDD sequence

### 1. Red

There is no new test to write — the device suites already exist and already encode the
claims. The falsifying baseline is the **`2c14d7f` device record**, which this task
compares against:

- MVS device-vs-upstream was **2/3 trees** (the carve-out) and MVS device-vs-CPU was
  `4.703e-11`.
- The Red-equivalent evidence is therefore: run
  `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1`
  and confirm the device arm now prints **3/3** for MVS where the recorded baseline
  said 2/3. If it still prints 2/3, TASK-02's second `.chain(...)` site (`:347`, the
  rocm-gated one) was not edited — go fix TASK-02.
- **If any device test FAILS**, that is the informative outcome and the task's real
  content: capture the failing target, test fn, printed deltas, and (for
  `sampled_types_hold_1e5`) the `first_mismatched_trees` list from `:400-404`. Do NOT
  loosen `EPS`, do NOT touch `replay_grow_draws`, and do NOT re-tune
  `POST_TREE_EXTRA_DRAWS` — that is SPEC `R1` and would re-break Bayesian.

  **Distinguish the two failure modes before escalating** (`MVS-S8` v2):

  1. **Draw-accounting** — device-vs-CPU `max|Δpred|` breaks 1e-5 in
     `device_bootstrap_parity_test`. Since both sides call the SAME host `bootstrap()`
     (`boosting.rs:3262` / `:3833`), this would mean some device-side accounting silently
     depended on the old 3-draw count. Genuinely surprising; escalate with the numbers.
  2. **Device/CPU split-argmax disagreement** — the device arm of
     `bootstrap_dev_oracle_test` fails on `Stage::Splits` while the **CPU arm of the same
     test passes**. This is a live mode, not a hypothetical: the sibling phase measured
     `split_mismatched_trees = 4/20` on the BASE grower
     `[VERIFIED: PROJECT device-bootstrap-parity/progress.md:162]`, and
     `compare_stage(Stage::Splits, …)` treats a different border as a **hard failure**,
     unlike `device_bootstrap_parity_test.rs:411-417` which bounds the divergent tree's
     *contribution*. MVS's heterogeneous keep-probabilities make near-ties likelier.

  For mode 2 apply `MVS-S8`'s **pre-defined escalation**: record a **device-only documented
  residual** — a new specification entry naming the divergent tree index, the differing
  border values, and the divergent tree's bounded contribution (measure it the way
  `sampled_types_hold_1e5` does). It is **NEVER** a tolerance loosening, **NEVER** a
  `replay_grow_draws` / `PRE_TREE_DRAWS` / `POST_TREE_EXTRA_DRAWS` change, and **NEVER** a
  re-introduced reduced-tree carve-out on the CPU arm — `MVS-S2` is a CPU claim and stands
  on its own evidence.

### 2. Green

No production change is expected. "Green" is the full device suite passing with the
recorded figures. Run, in this order (each on the ROCm rig):

1. `cargo test -p cb-backend --no-default-features --features rocm --lib mvs -- --nocapture --test-threads 1`
   (cheapest; confirms TASK-06's mirror before anything expensive)
2. `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1`
   → **the headline**: four scenarios, each `3/3 trees`, each device-grown
   (`gpu.grown == 3`) and each sampled scenario carrying a real sample
3. `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_parity_test -- --nocapture --test-threads 1`
   → 4 tests; record the MVS row's `max|Δpred(device,cpu)|`,
   `split_mismatched_trees`, and `max|Δpred(sampled,unsampled)|`

### 3. Refactor

Nothing to refactor — no code is written. Instead, do the **attribution pass**: for
every number that moved relative to the `2c14d7f` baseline, state in `progress.md`
whether the movement is expected (the host sampler changed, so the MVS fit is a
different — correct — model) or unexplained (which would be a blocker). Specifically:

- MVS device-vs-CPU `max|Δpred|` should stay the same ORDER (~5e-11): both sides moved
  together, so the DIFFERENCE should not grow.
- MVS device-vs-upstream should IMPROVE from 2/3 gated trees to 3/3 real trees.
- Bernoulli / Bayesian / No figures should be **unchanged** — the fix touches only the
  MVS arm. A movement in a non-MVS row is a red flag worth escalating.

### 4. Verify

All eight commands, all `--no-default-features --features rocm`:

- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_parity_test -- --nocapture --test-threads 1`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_probe_test -- --nocapture`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_seam_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_speed_test -- --nocapture`
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib -- --test-threads 1`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test mvs_seeds_oracle_test`
  → prints its SKIP line (TASK-03's new test is CPU-gated; this proves it compiles
  under rocm and does not silently claim device evidence).
- Run the **error-attributed** rocm clippy check (`PLAN.md` §4.12 helper):
  `clippy_error_files -p cb-backend --no-default-features --features rocm --all-targets | grep "mvs_device"`
  → must be EMPTY, and no NEW error file versus TASK-06's recorded rocm baseline.
  **`--keep-going` is mandatory** (CRITICAL-1): without it cargo aborts targets in parallel
  and the surfaced subset varies run to run. **Do not grep `-->` lines** (C2-1):
  `mvs_device.rs:80` carries a pre-existing `clippy::manual_rotate` warning, so that form is
  red before any work. Never assert "clippy clean" — the `cb-backend --lib` baseline is
  4 errors across 3 files at HEAD.
- Run: `git status --short` → shows NO change under `crates/` from this task.
- Confirm: `git diff crates/cb-train/src/device_draw_replay.rs` → EMPTY.
- Confirm: `grep -n "PRE_TREE_DRAWS: usize = 2\|POST_TREE_EXTRA_DRAWS: usize = 2" crates/cb-train/src/boosting.rs`
  → both constants still `2`.

## Completion criteria

- [ ] The device arm of `bootstrap_dev_oracle_test` reports **3/3 trees for MVS** (up
      from the recorded 2/3), with `gpu.grown == iterations` and the sampled-seam
      guard intact for all four scenarios. This was a **projected** outcome until this run
      (research §12 MEDIUM #4: no GPU run in the research session), so the transcript is
      the phase's first evidence for `AC-8`. If it fails while the CPU arm passes, the
      `MVS-S8` mode-2 device-only-residual escalation applies and the task completes with
      that residual RECORDED — not with a loosened tolerance.
- [ ] `device_bootstrap_parity_test` green (4 tests); the MVS device-vs-CPU
      `max|Δpred|` recorded and ≤1e-5, same order as the `4.703e-11` baseline.
- [ ] The `max|Δpred(sampled, unsampled)| > EPS` anti-false-pass guard still fires
      (i.e. the sample still reaches the split histogram).
- [ ] Bernoulli / Bayesian / No device figures unchanged relative to `2c14d7f`.
- [ ] `device_oblivious_parity_probe_test`, `device_seam_test`,
      `device_nonsym_fit_test`, `device_region_fit_test`,
      `device_bootstrap_speed_test` all green.
- [ ] `cb-backend --lib` under rocm shows no new failure; rocm clippy no new error.
- [ ] `mvs_seeds_oracle_test` compiles under rocm and prints its SKIP line.
- [ ] **No file under `crates/` was modified by this task**;
      `device_draw_replay.rs` byte-unchanged; both draw constants still `2`.

## Completion evidence to record in `progress.md`

- The four-scenario device transcript with the `3/3 trees` lines.
- The MVS / Bernoulli / Bayesian device-vs-CPU `max|Δpred|` values and their
  `2c14d7f` baselines side by side.
- `split_mismatched_trees` counts and the largest divergent-tree
  `max|Δcontribution|`.
- The `cb-backend --lib` rocm pass/fail tally and the rocm clippy error set.
- An explicit statement that no source file was changed.

## Risks and guardrails

- **SPEC `R1` — "fixing" a device failure by re-tuning shared accounting.** Forbidden.
  If device and CPU diverge, the cause is NOT `POST_TREE_EXTRA_DRAWS` or
  `replay_grow_draws` (both independently verified against the instrumented trace and
  the value-sensitive 3-tree `bootstrap_oracle_bayesian`). Escalate with the numbers.
- **Absorbing a source change into a verification task.** `MVS-S8` says "no edit
  expected". Any needed edit is a new finding requiring its own spec/task.
- **Wrong build line.** Bare `--features rocm` unifies `cpu` in; a blanket
  `cargo test -p cb-train --features rocm` (no `--test`) fails to build on ~37 files.
  Always both flags plus `--test <target>`.
- **Treating a `cpu`-feature run as device evidence.** Every device test is behind
  `#[cfg(any(feature = "rocm", feature = "cuda"))]` and prints an explicit SKIP
  otherwise. A green default-feature run proves nothing here.
- **Flaky-run misattribution.** `research.md §12 MEDIUM #4` marks the rocm outcome as
  reasoned-but-unexecuted. If a device test fails, re-run it once with
  `--test-threads 1` before concluding — the run-to-run jitter budget was measured at
  `0.000e0`, so a genuinely different result on a re-run is itself a finding.
- **No CUDA sign-off is required this phase.** The fix is host-side and the device
  branch provably shares the same sampler, so a ROCm run is strictly stronger than a
  cuda compile. Do not open a Kaggle/Colab loop for this.
