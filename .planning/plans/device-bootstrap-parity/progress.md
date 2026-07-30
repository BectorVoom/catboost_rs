---
phase: device-bootstrap-parity
branch: fix/bootstrap-rng-draw-accounting
base_commit: 5a5068a
status: planned
updated_at: 2026-07-30T00:00:00Z
spec: SPEC.md
plan: PLAN.md
---

# Phase progress: WR-01 device bootstrap parity

## Summary

- Total tasks: 10
- Pending: 10
- In progress: 0
- Blocked: 0
- Completed: 0

## Execution order

| # | Task | File | Wave | Hardware | Status | Gated by |
|---|---|---|---|---|---|---|
| 1 | `TASK-01` base-grower ≤1e-5 oracle | `plan1.md` | A | local ROCm | pending | — |
| 2 | `TASK-02` split tie-break characterisation | `plan2.md` | B | local ROCm | pending | `TASK-01` |
| 3 | `TASK-03` device score/leaf channel split + range guard | `plan3.md` | B | local ROCm | pending | `TASK-01` |
| 4 | `TASK-04` seam widening + `sample_from_host` | `plan4.md` | C | local ROCm + 4-backend compile | pending | `TASK-03` |
| 5 | `TASK-05` host draw replay + λ carry | `plan5.md` | A | none | pending | — |
| 6 | `TASK-06` gate relax + host multiplier | `plan6.md` | D | local ROCm | pending | `TASK-04`, `TASK-05` |
| 7 | `TASK-07` bias-0 fixtures + CPU oracle | `plan7.md` | A | none | pending | — |
| 8 | `TASK-08` device parity oracles (≤1e-5) | `plan8.md` | E | local ROCm | pending | `TASK-02`, `TASK-06`, `TASK-07` |
| 9 | `TASK-09` Poisson contract | `plan9.md` | E | ROCm + CPU | pending | `TASK-06` |
| 10 | `TASK-10` determinism budget + CUDA sign-off | `plan10.md` | F | ROCm + Kaggle CUDA | pending | `TASK-08`, `TASK-09` |

Wave A (`TASK-01`, `TASK-05`, `TASK-07`) can start immediately and in parallel —
disjoint files, verified in `PLAN.md` §1.

## Task checklist

- [ ] `TASK-01` — committed ≤1e-5 base-grower oracle replaces the uncommitted probe — specs: `WR01-S11`
- [ ] `TASK-02` — device/CPU tie-break rule proven identical; mismatches attributed below the fixed-point floor — specs: `WR01-S12`
- [ ] `TASK-03` — split histogram ← sampled pair, leaves ← unsampled pair, empty sample byte-identical, range guard — specs: `WR01-S1`, `WR01-S2`, `WR01-S3`, `WR01-S10`
- [ ] `TASK-04` — `grow_tree_on_device(approx, target, sample)` across 3 impls; `sample_from_host` exclusivity — specs: `WR01-S4`, `WR01-S5`
- [ ] `TASK-05` — `replay_grow_draws` matches the real grower's `raw_state()`; `prev_leaf_mean_l2` carried — specs: `WR01-S7`, `WR01-S8`
- [ ] `TASK-06` — gate allow-list `{No, Bayesian, Bernoulli, Mvs}`; host multiplier built from `bootstrap()` — specs: `WR01-S6`, `WR01-S9`
- [ ] `TASK-07` — `bootstrap_dev/` bias-0 fixtures generated; CPU oracle ≤1e-5 — specs: `WR01-S14`
- [ ] `TASK-08` — device == upstream ≤1e-5 and device == CPU ≤1e-5 for all three types — specs: `WR01-S15`
- [ ] `TASK-09` — Poisson rejected identically on every backend; sampler capability oracle — specs: `WR01-S16`
- [ ] `TASK-10` — ≤1e-7 run-to-run budget; CUDA sign-off; close-out — specs: `WR01-S13`

## Dependency graph

```text
TASK-01 ─┬─> TASK-02 ──────────────────────┐
         └─> TASK-03 ─> TASK-04 ─┐         │
TASK-05 ────────────────────────┬┴> TASK-06 ┼─> TASK-08 ─┐
TASK-07 ────────────────────────┘           │            ├─> TASK-10
                                 TASK-06 ───┴─> TASK-09 ─┘
```

Acyclic. Plan numbering is a valid topological order.

## Specification coverage

| Spec | Task | Spec store |
|---|---|---|
| `WR01-S1` | `TASK-03` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S2` | `TASK-03` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S3` | `TASK-03` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S4` | `TASK-04` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S5` | `TASK-04` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S6` | `TASK-06` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S7` | `TASK-05` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S8` | `TASK-05` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S9` | `TASK-06` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S10` | `TASK-03` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S11` | `TASK-01` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S12` | `TASK-02` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S13` | `TASK-10` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S14` | `TASK-07` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S15` | `TASK-08` | `SPEC.md` §5 (draft, unimplemented) |
| `WR01-S16` | `TASK-09` | `SPEC.md` §5 (draft, unimplemented) |

All sixteen specifications are `implementation_state: unimplemented` and
`document_state: draft`.

## Measurements to record (fill in during implementation)

| Measurement | Task | Baseline (this session) | Observed |
|---|---|---|---|
| base grower `max\|Δpred\|` 512×4 d3 ×5 | 01 | 2.22e-16 | |
| base grower `max\|Δpred\|` 2048×8 d6 ×10 | 01 | 2.22e-16 | |
| base grower `max\|Δpred\|` 20000×16 d6 ×20 | 01 | 4.44e-16 | |
| base grower `max\|Δpred\|` 1500×4 d2 ×3 | 01 | not measured | |
| split-mismatched trees @20000×16 | 01/02 | 3/20 | |
| largest split-mismatch gain gap vs fixed-point floor | 02 | not measured | |
| `raw_state()` / `call_count()` per replay row | 05 | not measured | |
| `bootstrap_dev` CPU oracle max deltas ×4 scenarios | 07 | not measured | |
| device-vs-upstream `max\|Δleaf\|` / `Δstaged` / `Δpred` ×4 | 08 | not measured | |
| device-vs-CPU `max\|Δpred\|` ×3 @20000×16 | 08 | not measured | |
| `CB_GPU_PROF` per-stage timings + upload cost | 08 | not measured | |
| MVS fixed-point headroom vs 2^33 | 10 | not measured | |
| run-to-run `max\|Δpred\|` ×5 (ROCm / CUDA) | 10 | not measured | |

## Blockers

- **None blocking start.** Wave A can begin now.

### Decisions — RESOLVED 2026-07-30

1. **D5 (Poisson) — CONFIRMED "reject up front".** The user initially chose "include
   all four types", then reversed to the planner's recommendation once the evidence
   was presented: reject Poisson uniformly on every backend and keep the device
   sampler as a documented capability oracle with no parity claim. `plan9.md`
   implements this; its "Alternative A" is now dead and may be ignored.
   Empirical backing from the Kaggle P100 run (`bench/bootstrap_gpu`, kernel v2):
   official CatBoost **CPU** rejects Poisson with
   `bootstrap_options.cpp:29: Error: poisson bootstrap is not supported on CPU`,
   while official CatBoost **GPU** trains it fine (1.33 s on 300k×50) — confirming
   Poisson is GPU-only upstream and no CPU-semantics parity target exists.
2. **`CbError::Degenerate → Unsupported` — REJECTED; keep `Degenerate`.** The user
   chose the lower-blast-radius option: no caller-visible variant change. Fix ONLY the
   misleading message. `plan9.md` and `SPEC.md` §8 have been updated accordingly.
3. **`WR01-S13` escalation** — STILL CONDITIONAL (not a decision, an outcome): if the
   ≤1e-7 run-to-run budget fails, converting `partition_update_kernel` to fixed-point
   `Atomic<u64>` becomes a new `plan11.md`. Scoped in `plan10.md` §Escalation.

## Known pre-existing red (NOT this phase)

- `monotone_non_symmetric_and_region_are_typed_errors` (`cb-train`). Recorded in
  `5a5068a`'s commit message (493 passed / 1 failed on a clean tree). Do not fix here.

## Specification-store synchronization

TreeFinder MCP is available, but this repository keeps its specification store as
plain `.planning/plans/<slug>/SPEC.md` files (twelve sibling phases, none registered
in TreeFinder). No TreeFinder document was added, updated, or left stale;
`SPEC.md` is the draft spec of record. See `SPEC.md` §12 for the opt-in migration
path if the corpus is ever moved into TreeFinder.

| Document | Action | State |
|---|---|---|
| `.planning/plans/device-bootstrap-parity/SPEC.md` | added | draft |
| `.planning/plans/device-bootstrap-parity/PLAN.md` | added | draft |
| `.planning/plans/device-bootstrap-parity/plan1.md` … `plan10.md` | added | pending |
| `.planning/plans/device-bootstrap-parity/progress.md` | added | planned |
| TreeFinder corpus | **not synchronized** (by design, §12) | n/a |
