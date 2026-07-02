# Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-03
**Phase:** 10-gpu-foundations-runtime-seam-session-residency-device-primitive-library-compressed-index-depth-1-kaggle-cuda-oracle-speed-harness
**Areas discussed:** Primitive-library oracle depth, Reduction-determinism spike depth, Kaggle CUDA harness shape, cindex packing fidelity

**Note:** This phase was unusually well-specified by ROADMAP.md/REQUIREMENTS.md. The structural choices (seam signatures, `GpuTrainSession` residency, `Ok(None)` fallback, ε bars, Cosine default, depth-1 MVP, landmines, large-n speed bar) were already LOCKED and were NOT re-asked. Discussion targeted only the remaining open implementation choices.

---

## Primitive-library oracle depth (GPUT-16)

### How to oracle-test the from-scratch device-primitive library on Kaggle CUDA

| Option | Description | Selected |
|--------|-------------|----------|
| Per-primitive standalone | Each primitive gets its own Kaggle CUDA oracle vs a CPU reference (≤1e-4), independent of the tree — max confidence, more harness work | |
| End-to-end only | Oracle primitives transitively through depth-1 tree + cindex — leaner, but subtle bugs hide until a later phase | |
| Hybrid: tier by risk | Standalone oracles for high-risk primitives (scan, segmented-scan, radix sort + 1-bit reorder, reduce-by-key, stat-agg); trivial ones (fill/transform, simple reduce) folded into end-to-end | ✓ |

**User's choice:** Hybrid — tier by risk.
**Notes:** Balances substrate confidence against harness volume; high-risk/hard-to-isolate primitives get isolation, trivial ones ride the depth-1 end-to-end path.

### Ground-truth reference for the standalone primitive oracles

| Option | Description | Selected |
|--------|-------------|----------|
| Serial CPU/numpy reference | Expected values from a trivial serial computation of the same generic primitive on the same random input — self-contained | ✓ |
| Upstream CatBoost/CUB capture | Capture actual CUB outputs from an instrumented upstream run as fixtures — higher fidelity but heavy, and equal to serial for generic primitives | |
| Rust CPU primitives reused | Reuse existing cb-compute CPU helpers (transcribe inline per no-cb-train-dep landmine) | |

**User's choice:** Serial CPU/numpy reference.
**Notes:** Generic building blocks (a prefix-scan is a prefix-scan); a dead-simple serial reference is easy to trust and keeps the harness self-contained. Transcribe any CPU reference inline (no crate-boundary reach).

---

## Reduction-determinism spike depth (feeds Phase 11)

### How deep should the reduction-determinism spike go

| Option | Description | Selected |
|--------|-------------|----------|
| Prototype + measure | Implement top 2-3 candidate deterministic strategies, measure correctness variance AND speed on Kaggle CUDA, then recommend | ✓ |
| Paper survey only | Survey candidates, document tradeoffs + a pick, defer implementation to Phase 11 | |
| You decide | Let research/planning choose based on primitive-lib needs | |

**User's choice:** Prototype + measure.
**Notes:** The CUDA harness stands up this phase and a deterministic reduce is on SC-1's critical path anyway; measuring now hard-de-risks Phase 11's ε=1e-4 gate.

### Relationship of the spike winner to the shipped reduce primitive

| Option | Description | Selected |
|--------|-------------|----------|
| Winner becomes the primitive | The measured-best deterministic strategy IS the reduce/segmented-reduce/reduce-by-key that lands in the Phase 10 primitive library | ✓ |
| Spike stays exploratory | Ship a provisional simple reduce now; spike only writes a recommendation; Phase 11 swaps in the winner | |

**User's choice:** Winner becomes the primitive.
**Notes:** Avoids building a throwaway reduce; the depth-1 tree + stat-agg exercise it end-to-end immediately, giving the winner real validation on landing.

---

## Kaggle CUDA harness shape (BENCH-01)

### Harness form factor + where oracle fixtures live

| Option | Description | Selected |
|--------|-------------|----------|
| Committed .ipynb + repo fixtures | Notebook builds the --features cuda wheel, loads committed seeded fixtures + CPU expected values, runs correctness (blocking) then warm-run speed, prints structured report | ✓ |
| Python script + markdown runbook | harness.py + a markdown runbook — more portable, less native to Kaggle click-to-run | |
| Generate fixtures in-notebook | Notebook generates inputs from a fixed seed and computes CPU expected-values live — fewer artifacts, but CPU reference runs on Kaggle rather than pinned in-repo | |

**User's choice:** Committed .ipynb + repo fixtures.
**Notes:** Notebook is the native, diffable Kaggle artifact; committed fixtures keep the CPU ≤1e-5 reference the pinned in-repo authority and make the human-gated run push-button.

### Large-n dataset backing the depth-1 speed bar (D-10-09)

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic generator | Seeded generator (configurable n_rows/n_features) feeds both the ≤1e-5 correctness fixture and the large-n speed workload — reproducible, no download, tunable above break-even | ✓ |
| Real named dataset (Higgs/Epsilon) | Standard large public set — realistic/comparable, but adds a Kaggle data dependency and splits the correctness fixture from the speed workload | |
| Both: synthetic + one real | Synthetic pinned bar + one real set as informal cross-check — more coverage, more work | |

**User's choice:** Synthetic generator.
**Notes:** No external download; fully reproducible; correctness and speed share one generator; n tunable above the launch-overhead break-even.

---

## cindex packing fidelity (GPUT-15)

### How faithfully to replicate upstream's bit-packing in Phase 10

| Option | Description | Selected |
|--------|-------------|----------|
| Exact upstream 32-bit packing | Replicate WriteCompressedIndex — TCFeature Offset/Shift/Mask/OneHot packing multiple features per 32-bit word — from the start | ✓ |
| Simple one-value-per-slot first | Simpler resident layout first, tighten to real packing later — faster to first-correct, but rewrites histogram address arithmetic later | |
| You decide | Let research confirm tractability from §6.6a and pick | |

**User's choice:** Exact upstream 32-bit packing.
**Notes:** Memory efficiency is a first-class project constraint; every later histogram kernel consumes this as THE input so its address arithmetic should be right once; the CPU quantized layout is the ≤1e-4 oracle so correctness is checkable immediately regardless of packing complexity.

---

## Claude's Discretion

- Wave decomposition/ordering (ROADMAP suggests: primitive library → cindex → seam+residency → depth-1+Cosine → Kaggle harness → reduction spike).
- Seam module placement in cb-compute (`runtime.rs`, mirroring `compute_gradients_grouped`), `apply_leaf_delta` kernel scope, per-fit session lifecycle, and the bin→border join.

## Deferred Ideas

- Real named large datasets (Higgs/Epsilon) as a realistic speed cross-check — revisit at Phase 14 (BENCH-03).
- On-device border/quantile computation (`FastGpuBorders`/`ComputeQuantileBorders`) — out of scope; host quantization is the ≤1e-5 reference.
- Newton der2, depth>1 partition-aware histograms + subtraction trick, GPUT-14 operative gate — Phase 11.
- Stale `.planning/spikes/MANIFEST.md` (Spike 001, online-CTR / Phase 5) is unrelated to Phase 10.
