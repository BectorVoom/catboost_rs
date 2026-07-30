---
plan: 8
task_id: TASK-08
phase: device-bootstrap-parity
status: pending
order: 8
wave: E
hardware: local ROCm gfx1151 (required)
depends_on: [TASK-02, TASK-06, TASK-07]
specifications: [WR01-S15]
---

# Task 8: The parity oracles — device vs upstream and device vs CPU at ≤1e-5

## Objective

After this task each of Bernoulli, Bayesian and MVS has a committed, running oracle
proving the **device-grown** model matches upstream CatBoost 1.2.10 to ≤1e-5 on the
bias-0 fixtures, and matches the in-repo CPU grower to ≤1e-5 on a multi-iteration
synthetic fixture. This is the phase's sign-off gate and the point at which the
"never a fabricated device result" rule (D-04 / T-10-05, restated at
`boosting.rs:3044-3053`) is satisfied for the newly enabled types.

## Specification references

- `WR01-S15` — the device reproduces upstream at ≤1e-5 for Bernoulli, Bayesian and
  MVS. Primary failure reason: an enabled bootstrap type does not meet the parity
  bar.

## Context and evidence

- **The fixtures** land in TASK-07: `crates/cb-oracle/fixtures/bootstrap_dev/…`,
  1500×4, depth 2, 3 iterations, `boost_from_average=False`, `random_strength=0`,
  `score_function=L2`, `leaf_estimation_method=Gradient`, `random_seed=0`,
  `thread_count=1`.
- **The CPU harness to reuse**: `crates/cb-train/tests/common/bootstrap_harness.rs`
  (created in TASK-07's refactor) plus `cb_oracle::{compare_stage, load_f64_vec,
  load_model_json, Stage}` `[VERIFIED: bootstrap_oracle_test.rs:1-130]`.
- **The device test shape to reuse**: TASK-01/02's
  `crates/cb-train/tests/common/device_fixture.rs` (the `CpuRefRuntime` + skip
  pattern derived from `device_nonsym_fit_test.rs:92-210`).
- **Device-signature assertion precedent**: `device_nonsym_fit_test.rs:136-149`.
- **Split-structure equality is NOT assertable** — TASK-02 established why
  (fixed-point histogram, `kernels.rs:2335`, `:3889-3893`). This test therefore
  gates on prediction / staged / leaf deltas and REPORTS structure differences,
  reusing TASK-02's attribution helper to fail loudly if any mismatch exceeds the
  fixed-point floor.
- **Tolerance**: ε = **1e-5** everywhere. Do not copy the ε=1e-4 precedents.
- **The `bootstrap_dev/no` scenario is the control**: at `bootstrap_type = No` the
  device path was already proven in TASK-01, so a failure there means the fixture or
  the harness is wrong, not the sampling.

## Files

- Modify: `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` — add the
  `#[cfg(any(feature = "rocm", feature = "cuda"))] mod device { … }` arm that runs
  the SAME four scenarios through `GpuBackend` and gates them against the SAME
  upstream artifacts, plus the `#[cfg(not(...))]` SKIP arms.
- Create: `crates/cb-train/tests/device_bootstrap_fit_test.rs` — device-vs-CPU on a
  synthetic ≥3-iteration bias-0 fixture (a bigger, harder shape than the 1500×4
  upstream one, to catch accumulation).
- Modify: `crates/cb-train/tests/common/device_fixture.rs` — expose the shared
  `CpuRefRuntime` and the split-mismatch attribution helper from TASK-02.

## TDD sequence

### 1. Red

**A. Device vs upstream** (`bootstrap_dev_oracle_test.rs::device`), one test per
scenario so a failure names the sampler:

- `device_bernoulli_matches_upstream`
- `device_bayesian_matches_upstream`
- `device_mvs_matches_upstream`
- `device_no_matches_upstream` (control)

Each asserts, in this order (cheapest and most diagnostic first):
1. **device signature** — the fit is device-grown (non-empty `oblivious_trees`,
   empty `non_symmetric_trees` / `region_trees`, plus the same positive device-arm
   invariant `device_nonsym_fit_test.rs:136-149` uses). A silent CPU fallback must
   fail HERE, not pass silently.
2. `max|Δleaf|` vs `model.json` leaf values ≤ 1e-5 — this is the assertion that
   fails FIRST if the WR01-S2 channel split were reverted, which is why it precedes
   predictions.
3. `max|Δstaged|` vs `staged.npy` ≤ 1e-5 (per-iteration, so tree-1 RNG drift and
   tree-1 MVS λ drift surface as an early-stage failure rather than a final-value
   one).
4. `max|Δpred|` vs `predictions.npy` ≤ 1e-5.
5. split structure vs upstream: REPORTED, and any mismatch run through TASK-02's
   fixed-point-floor attribution (a mismatch above the floor is a hard failure).

**B. Device vs CPU** (`device_bootstrap_fit_test.rs`), one test per type on a
synthetic fixture: 20000×16, depth 6, 10 iterations, bias 0, unit weights,
`random_strength = 0`, `score_function = L2`, `LeafMethod::Gradient`,
`subsample = 0.8` / `bagging_temperature = 1.0`. Assert `max|Δpred| ≤ 1e-5` against
`CpuRefRuntime`, plus the same device signature and the same structure attribution.

- Run:
  `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1`
  `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_fit_test -- --nocapture --test-threads 1`

### 2. Green

No production change should be needed — TASK-03..07 supply the mechanism. If a
scenario fails, diagnose in this order (each step isolates one earlier task):

| Symptom | Likely cause | Task to revisit |
|---|---|---|
| tree 0 leaves wrong, splits right | leaf channel sampled | TASK-03 (WR01-S2) |
| tree 0 splits wrong | score channel unsampled or wrong `s` | TASK-03 (WR01-S1) / TASK-06 (WR01-S6) |
| tree 0 right, tree 1+ wrong (all types) | RNG phase drift | TASK-05 (WR01-S7) |
| tree 0 right, tree 1+ wrong (MVS only) | λ not carried | TASK-05 (WR01-S8) |
| device signature assertion fails | gate declined | TASK-06 (WR01-S9) |
| `OutOfRange` from the guard | fixed-point range (MVS) | TASK-03 (WR01-S10) |
| structure mismatch above the floor | real divergence | TASK-02 (WR01-S12) — STOP |

Fix in the owning task's file, re-run that task's own tests, then return here.

### 3. Refactor

- Collapse the four upstream scenarios into one table-driven runner
  `(name, EBootstrapType, subsample, bagging_temperature)` so adding a type later is
  one row. Keep the four `#[test]` entry points (so failures name the sampler) that
  each call the runner.
- Move the "assert device-grown" block into `common/device_fixture.rs` as
  `assert_device_grown(&model, iterations, label)` and use it in all device tests
  (TASK-01, 02, 06, 08) — one definition of "the device really ran".
- Run: all device test targets touched.

### 4. Verify

- Run: `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test -- --nocapture --test-threads 1`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_fit_test -- --nocapture --test-threads 1`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_split_tiebreak_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_gate_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_seam_test`
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test` (cpu → CPU arm green,
  device arm SKIP)
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (**blocking**)
- Run: `cargo test -p cb-train` (CPU suite; one known pre-existing red)
- Confirm: record every measured `max|Δleaf|`, `max|Δstaged|`, `max|Δpred|` per
  scenario in `progress.md` — these numbers are the phase's evidence.

## Non-gating measurement (do, record, do not act on)

Run one Bernoulli device fit at the 20000×16 shape with `CB_GPU_PROF=1` and record
the per-stage timings plus the added per-tree `8·n`-byte upload cost. This is the
input to the Design B′ decision (R14); it is explicitly NOT a gate in this phase.
`bench/bootstrap_gpu/bootstrap_bench.py` exists as a harness (untracked); its
`only_No_is_gpu_eligible` caveat becomes false after TASK-06 and should be updated
or deleted.

## Completion criteria

- [ ] All four upstream scenarios pass at ≤1e-5 on leaves, staged, and predictions,
      under `--features rocm`.
- [ ] All three device-vs-CPU scenarios pass at ≤1e-5 at 20000×16 depth 6 ×10.
- [ ] Every device test asserts the fit was device-grown before asserting numbers.
- [ ] Any structure mismatch is attributed below the fixed-point floor.
- [ ] The cpu build runs the CPU arm and SKIPs the device arm.
- [ ] `bootstrap_oracle_test` green and unchanged.
- [ ] Measured deltas and `CB_GPU_PROF` timings recorded in `progress.md`.

## Risks and guardrails

- **R6 silent CPU fallback** — assertion 1 in every test.
- **R9 tolerance creep** — grep the new files for `1e-4` before committing; there
  must be none.
- **R10 tie flipped on a non-degenerate split** — the structure attribution turns
  this from an invisible risk into a loud failure.
- **R14 per-tree upload cost** — measured, recorded, not acted on.
- **Ambiguous failures** — the diagnosis table in Green step exists precisely so a
  failure here routes back to exactly one earlier task instead of triggering a
  shotgun debug across the whole phase.
- **MVS at 20000×16 may trip the fixed-point range guard** (F-D: the weight channel
  `threshold / sqrt(λ + der²)` is unbounded as λ → 0). If it does, that is the guard
  working — record the observed maximum and decide in TASK-10 whether the bound
  needs a per-bin rather than per-fit estimate. Do NOT relax the guard to make a
  test pass.
