---
title: "WR-01 — Device bootstrap parity — TDD implementation plan"
phase: device-bootstrap-parity
branch: fix/bootstrap-rng-draw-accounting
base_commit: 5a5068a
plan_version: 1
status: draft
updated_at: 2026-07-30T00:00:00Z
source_spec: .planning/plans/device-bootstrap-parity/SPEC.md
source_research: .planning/plans/device-bootstrap-parity/research.md
task_files: [plan1.md, plan2.md, plan3.md, plan4.md, plan5.md, plan6.md, plan7.md, plan8.md, plan9.md, plan10.md]
progress: progress.md
---

# WR-01 — TDD implementation plan

Plan-only artifact. **No production code is authored here.** Every path, symbol,
line reference and command below was read from disk or executed this session
(evidence is inline, and repeated per task in each `planN.md`).

Read `SPEC.md` first — it holds the sixteen specifications (WR01-S1 … WR01-S16),
the four new session findings (F-A … F-F), and the Poisson decision (§8).

---

## 0. Goal-backward derivation

The phase's observable end state is AC-4 in SPEC §6: *a Bernoulli / Bayesian / MVS
fit runs on the device and matches upstream CatBoost 1.2.10 to ≤1e-5.* Working
backwards:

| To claim … | you must first have … | task |
|---|---|---|
| device == upstream ≤1e-5 (AC-4) | a bias-0 upstream fixture (AC-3) AND a correct device sampling mechanism | TASK-08 |
| a bias-0 upstream fixture | `gen_bootstrap_dev()` + a CPU-side oracle proving the fixture itself | TASK-07 |
| a correct sampling mechanism | the gate open + the host multiplier built | TASK-06 |
| the gate open | the RNG stream phase-exact + MVS λ carried | TASK-05 |
| the gate open | the sample crossing the seam, exclusively host-side | TASK-04 |
| the sample reaching the histogram | the score/leaf channel split + a range guard | TASK-03 |
| any ≤1e-5 device claim at all | proof the BASE grower holds ≤1e-5 | TASK-01 |
| trust in split agreement | the tie-break rule characterised and locked | TASK-02 |
| a defensible sign-off | a leaf-reduce jitter budget + CUDA confirmation | TASK-10 |
| an honest Poisson story | one backend-independent typed rejection | TASK-09 |

Note the deliberate inversion versus a naive plan: **TASK-01 and TASK-02 come
first** even though they add no capability. They are the two places where the phase
can be proven impossible cheaply (base-grower tolerance, tie-break rule), and both
are pure test work on the existing tree.

---

## 1. Execution order and waves

```
Wave A (parallel, disjoint crates/files):
    plan1  TASK-01  base-grower ≤1e-5 oracle        [cb-train/tests]
    plan5  TASK-05  host draw replay + λ carry      [cb-train/src]
    plan7  TASK-07  bias-0 fixtures + CPU oracle    [cb-oracle, cb-train/tests]

Wave B (after TASK-01):
    plan2  TASK-02  split tie-break characterisation [cb-train/tests]
    plan3  TASK-03  device score/leaf channel split  [cb-backend]

Wave C:  plan4  TASK-04  seam widening + sample_from_host   (after 03)
Wave D:  plan6  TASK-06  gate relax + host sample build     (after 04, 05)
Wave E:  plan8  TASK-08  device parity oracles              (after 06, 07, 02)
         plan9  TASK-09  Poisson contract                   (after 06)
Wave F:  plan10 TASK-10  determinism budget + CUDA sign-off (after 08, 09)
```

Dependency graph (acyclic):

```text
TASK-01 ─┬─> TASK-02 ──────────────────────┐
         └─> TASK-03 ─> TASK-04 ─┐         │
TASK-05 ────────────────────────┬┴> TASK-06 ┼─> TASK-08 ─┐
TASK-07 ────────────────────────┘           │            ├─> TASK-10
                                 TASK-06 ───┴─> TASK-09 ─┘
```

Write-conflict check for the parallel sets:

- **Wave A.** TASK-01 creates `crates/cb-train/tests/device_oblivious_parity_test.rs`;
  TASK-05 creates `crates/cb-train/src/device_draw_replay{,_test}.rs` and touches
  `crates/cb-train/src/lib.rs`; TASK-07 touches
  `crates/cb-oracle/generator/gen_fixtures.py` and creates
  `crates/cb-train/tests/bootstrap_dev_oracle_test.rs`. Disjoint files. The only
  shared file is `crates/cb-train/src/lib.rs` (TASK-05 only) — no conflict.
- **Wave B.** TASK-02 is `cb-train/tests` only; TASK-03 is `cb-backend` only.
- **Wave E.** TASK-08 is tests only; TASK-09 touches `boosting.rs`, `bootstrap.rs`,
  `session.rs`, `kernels/bootstrap_device_test.rs`. Disjoint.

---

## 2. Specification → task coverage

| Spec | Behaviour | Primary task | Also verified by |
|---|---|---|---|
| WR01-S1 | sampled pair → split histogram | TASK-03 | TASK-08 |
| WR01-S2 | unsampled pair → leaf estimate | TASK-03 | TASK-08 |
| WR01-S3 | empty sample ⇒ byte-identical | TASK-03 | TASK-01, TASK-04 |
| WR01-S4 | sample crosses the seam, length-validated | TASK-04 | TASK-08 |
| WR01-S5 | host vs device sampler mutual exclusion | TASK-04 | TASK-06 |
| WR01-S6 | host multiplier from `bootstrap()` | TASK-06 | TASK-08 |
| WR01-S7 | RNG draw replay | TASK-05 | TASK-08 |
| WR01-S8 | `prev_leaf_mean_l2` carry | TASK-05 | TASK-08 (MVS) |
| WR01-S9 | gate allow-list + config threading | TASK-06 | TASK-09 |
| WR01-S10 | fixed-point range guard | TASK-03 | TASK-08 |
| WR01-S11 | base grower ≤1e-5 | TASK-01 | TASK-10 |
| WR01-S12 | tie-break rule locked | TASK-02 | TASK-08 |
| WR01-S13 | leaf-reduce jitter budget | TASK-10 | — |
| WR01-S14 | bias-0 fixture + CPU oracle | TASK-07 | TASK-08 |
| WR01-S15 | device == upstream ≤1e-5 | TASK-08 | TASK-10 (CUDA) |
| WR01-S16 | Poisson contract | TASK-09 | — |

Every spec maps to ≥1 task; every task references ≥1 spec.

---

## 3. Cross-cutting guardrails (apply to EVERY task)

1. **Source/test separation is mandatory** (CLAUDE.md). No `#[cfg(test)] mod tests`
   inside a production file. In `cb-backend`/`cb-train` `src/`, tests live in a
   sibling `*_test.rs` mounted with
   `#[cfg(test)] #[path = "x_test.rs"] mod tests;` (precedents:
   `crates/cb-train/src/bootstrap.rs:55-57`, `crates/cb-backend/src/kernels.rs:2941-2946`).
   Integration tests live under `crates/<crate>/tests/`.
2. **No `unwrap` / `expect` / `panic` / raw indexing in production** — workspace
   lints deny them. Test files opt out with a file-level
   `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]`
   (precedent `device_nonsym_fit_test.rs:19`).
3. **Typed errors only**: `thiserror`-derived `CbError` + `CbResult`
   (`crates/cb-core/src/error.rs`).
4. **Backend build lines**: always
   `--no-default-features --features rocm` (never bare `--features rocm`), and
   always `--test <target>` (a blanket rocm test build fails on 37 files importing
   `CpuBackend`).
5. **Device test skip convention**: `#[cfg(any(feature = "rocm", feature = "cuda"))]
   mod device { … }` with a local `CpuRefRuntime` implementing `Runtime` and
   inheriting the seam defaults, plus a `#[cfg(not(...))]` arm that prints
   `SKIP <test name>: needs rocm/cuda`. Copy `device_nonsym_fit_test.rs:92-210`
   verbatim in shape; do not invent a new pattern.
6. **ε = 1e-5** in every sign-off assertion. Never copy the shipped ε=1e-4.
7. **Do not hand-roll**: `bootstrap()`, `fast_log2f`/`fast_logf`, `std_normal`,
   `PRE_TREE_DRAWS`/`POST_TREE_EXTRA_DRAWS`, `fold_weights_resident`,
   `calc_average`/`scale_l2_reg`, `accumulate_leaf_weights`/`normalize_leaf_values`,
   the device samplers, the skip/gate test patterns.
8. **`cb-backend` must never depend on `cb-train`; `cb-compute` must never see a
   `cubecl` type.**
9. **CubeCL rule** — Design A needs **no new kernel**. If a task nonetheless reaches
   for `#[cube]`, STOP: read
   `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` first, and
   on any CubeCL build error read `cubecl_error_guideline.md` before attempting a
   fix. Blind fixes are prohibited.
10. **Known pre-existing red, not ours**:
    `monotone_non_symmetric_and_region_are_typed_errors` in `cb-train`. Record it,
    never "fix" it inside this phase.
11. **Do not modify** `crates/cb-oracle/fixtures/bootstrap/**` — the bias≠0 family is
    frozen and `bootstrap_oracle_test` is the blocking gate.

---

## 4. Hardware routing

| Task | Local ROCm gfx1151 | Kaggle CUDA P100 | No GPU needed |
|---|---|---|---|
| TASK-01 | **required** | confirm in TASK-10 | — |
| TASK-02 | **required** | not required | — |
| TASK-03 | **required** (unit + e2e) | confirm in TASK-10 | compile check only |
| TASK-04 | **required** | confirm in TASK-10 | CPU compile of all 3 impls |
| TASK-05 | — | — | **yes** (pure host) |
| TASK-06 | **required** | confirm in TASK-10 | CPU no-regression |
| TASK-07 | — | — | **yes** (python + CPU oracle) |
| TASK-08 | **required** | confirm in TASK-10 | — |
| TASK-09 | **required** (sampler oracle) | not required | CPU error contract |
| TASK-10 | **required** | **required** (sign-off) | — |

CUDA sign-off is deliberately batched into TASK-10 rather than spread across tasks:
Kaggle is a slow, manual loop, and every earlier task has a local ROCm gate that is
strictly stronger than "it compiles under cuda".

---

## 5. Definition of done for the phase

- [ ] AC-1 … AC-9 (SPEC §6) all hold.
- [ ] `cargo test -p cb-train --test bootstrap_oracle_test` green (blocking).
- [ ] `crates/cb-oracle/fixtures/bootstrap/` byte-unchanged.
- [ ] No new bootstrap type is device-eligible without a passing parity oracle.
- [ ] ε=1e-5 in every new sign-off assertion; no ε=1e-4 copied.
- [ ] The WR-01 note at `boosting.rs:3143-3159` is replaced with an accurate
      statement of what is now wired and what remains deferred (Design B′, Poisson,
      weighted pools, `random_strength != 0`).
- [ ] `progress.md` reflects the final state with evidence per task.
