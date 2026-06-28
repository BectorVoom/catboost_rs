# Phase 10: Coarse Runtime Grow-Tree Seam + GpuTrainSession Residency + Wire Depth-1 + Kaggle CUDA Oracle & Speed Harness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-29
**Phase:** 10-coarse-runtime-grow-tree-seam-gputrainsession-residency-wire
**Areas discussed:** Fallback granularity, Kaggle harness form, Reduction spike depth, Depth-1 oracle/baseline

---

## Fallback granularity (Ok(None) → CPU)

| Option | Description | Selected |
|--------|-------------|----------|
| Per-fit, all-or-nothing | Decide once at begin_device_training; all-device or all-CPU for the whole fit, no mid-run mixing | ✓ |
| Per-tree, may mix | Each grow_tree_on_device may fall back to CPU for that tree only | |
| Per-tree gate, error on mix | Per-tree signature, but None mid-run on a covered session is a hard error | |

**User's choice:** Per-fit, all-or-nothing.
**Notes:** Chosen for parity safety (no device/CPU drift compounding across boosting) and a clean speed measurement. Gate lives at session creation. A covered session that can't grow mid-run → CbError, not a silent CPU graft.

---

## Kaggle CUDA harness form (BENCH-01 / BENCH-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Thin .ipynb + diffable .py | Thin notebook calls committed bench/oracle .py; fixtures committed; RESULTS.md | |
| Self-contained .ipynb | Everything inline in one notebook | |
| Script + README only | Committed runnable script(s) + README; user assembles the notebook each phase | ✓ |

**User's choice:** Script + README only (`bench/cuda_oracle.py` + `bench/README.md`).
**Notes:** Most flexible, keeps all logic diffable. Correctness is a blocking gate before any speed number; human-gated external step; ROCm in-env is smoke-only, not a gate.

---

## Reduction-determinism spike depth (SC5, feeds Phase 11)

| Option | Description | Selected |
|--------|-------------|----------|
| Runnable micro-benchmark | Prototype all 3 candidates on-device; measure err + ms → SPIKE-REDUCTION.md | ✓ |
| Paper analysis doc only | Reasoned recommendation, no running code | |
| Doc + defer proto to P11 | Recommendation now; runnable comparison in Phase 11 step 0 | |

**User's choice:** Runnable micro-benchmark of i64 atomics vs private-histogram merge vs two-pass segmented reduce.
**Notes:** Real err+ms table backs the Phase 11 recommendation. gfx1100 lacks f64 atomic-add — spike records per-backend viability.

---

## Depth-1 oracle configs + speed framing (GPUT-04 / BENCH-02)

| Option | Description | Selected |
|--------|-------------|----------|
| RMSE+Logloss; speed = honest baseline | Both configs ≤1e-5; report depth-1 speed as-is, accept device may be ≈/slower (launch-bound) | |
| RMSE only first; honest baseline | Land RMSE first, Logloss follows; honest speed | |
| RMSE+Logloss; require device ≥ CPU | Both configs ≤1e-5; depth-1 device MUST beat CPU wall-clock | ✓ |

**User's choice:** RMSE+Logloss both, and depth-1 device MUST beat CPU.
**Notes:** Deliberate, informed bar above the written success criteria. Implication captured in CONTEXT (D-10-09): Phase 10 carries a launch-overhead-reduction obligation (fused/batched launches, persistent kernel, full residency). Research flag: if depth-1 device > CPU is genuinely infeasible on CUDA, ESCALATE to the user — do not silently relax.

---

## Claude's Discretion

- Exact `DeviceGrownTree` struct fields + precise host-typed seam method signatures (within GPUT-01's named shape).
- `GpuTrainSession` internal handle layout / lifetime mechanics.
- Fixture file format + RESULTS sign-off log structure.
- Micro-benchmark kernel sizes / problem shapes for the reduction spike.

## Deferred Ideas

- depth>1 histograms, Newton der2, Cosine GPU score, production reduction kernel → Phase 11.
- CTR / pairwise / multiclass / ordered-boosting device paths → Phase 12.
- Comprehensive final speed-parity sign-off (BENCH-03) → Phase 13.
- Estimated-feature quantization-grid parity todo — reviewed, NOT folded (FEAT-07 backlog, unrelated to GPU seam).
