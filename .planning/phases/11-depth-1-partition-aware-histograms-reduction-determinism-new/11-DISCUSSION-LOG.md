# Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-03
**Phase:** 11-depth-1-partition-aware-histograms-reduction-determinism-newton-der2
**Areas discussed:** Newton refinement loop, Depth-6 test workload, Subtraction-trick residency, How ε bar is proven

---

## Newton der2 refinement loop (GPUT-07)

| Option | Description | Selected |
|--------|-------------|----------|
| Fully device-resident | Refinement loop on device: reuse Phase 7.2 der1/der2 handles + apply_leaf_delta, recompute ders per step, no per-iteration readback; pin leaf_estimation_iterations in the fixture | ✓ |
| Host-orchestrated readback | Host drives the loop, reads der/approx back each iteration; simpler but breaks residency | |

**User's choice:** Fully device-resident
**Notes:** Preserves residency (the milestone's whole point); per-iteration readback × hundreds of trees would undercut the speed goal. leaf_estimation_iterations frozen in the CPU-reference fixture so device matches CPU exactly at ε=1e-4. → D-01, D-02.

---

## Depth-6 correctness fixture + CUDA speed workload

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse synthetic generator | Extend Phase-10 seeded synthetic generator to depth-6 RMSE + Logloss; one generator for both correctness fixture and speed workload | ✓ |
| Real named dataset now | Introduce Higgs/Epsilon this phase; pulls Phase-14 scope forward, adds non-reproducible external dependency | |

**User's choice:** Reuse synthetic generator
**Notes:** Fully reproducible, no external download. Real datasets stay deferred to Phase 14 (BENCH-03). → D-03.

---

## Subtraction trick + histogram memory residency (GPUT-05)

| Option | Description | Selected |
|--------|-------------|----------|
| Smaller-sibling + parent-resident | Compute the smaller partition directly, derive the larger by subtracting from the parent's resident histogram; keep only parent-level histograms resident (upstream §6.4, memory-lean) | ✓ |
| Always materialize both | Build both siblings directly every split; ~2× histogram work, more resident memory, gives up the subtraction-trick speed win | |

**User's choice:** Smaller-sibling + parent-resident
**Notes:** Memory efficiency is a first-class constraint at depth 6 (64 leaves × features × bins × channels); this is the speed lever that approaches parity. → D-04.

---

## How the ε=1e-4 bar is proven across the boosting run (GPUT-06 / SC-3)

| Option | Description | Selected |
|--------|-------------|----------|
| Final ε + per-tree diagnostic | Gate on final-prediction ε=1e-4 (blocking) AND instrument a per-tree split-agreement + run-to-run spread diagnostic in the Kaggle oracle | ✓ |
| Final ε only | Gate on final-prediction ε only; a mid-run flip that washes out by the end passes silently, no debugging locus | |

**User's choice:** Final ε + per-tree diagnostic
**Notes:** Directly evidences SC-3's "no split flips compounding over the boosting run" and gives a locus for debugging if the bar is missed. → D-05.

---

## Claude's Discretion

- Sub-wave decomposition/ordering (ROADMAP suggests: depth>1 histograms → reduction determinism → Newton der2) — planner refines.
- Newton leaf-estimation backtracking (upstream `AnyImprovement` line-search) — research flag: confirm whether the CPU reference uses it at the pinned config; mirror on device if so.
- Exact channel layout of partition-aware `pointwise_hist2`, `2^level` slot addressing, contiguous `TDataPartition` reorder — resolved by research/planning against §6.3/§6.4 + Phase-10 primitives.

## Deferred Ideas

- Real named datasets (Higgs/Epsilon) → Phase 14 (BENCH-03).
- Non-symmetric grow policies, Exact leaf, sampling/MVS, CTR/categoricals → Phase 12.
- Pairwise/ranking/multiclass/ordered/Langevin device families → Phase 13.
- On-device border/quantile computation → out of scope milestone-wide (host CPU quantization stays the ≤1e-5 reference).
