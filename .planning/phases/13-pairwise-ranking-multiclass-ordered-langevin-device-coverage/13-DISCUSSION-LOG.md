# Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-04
**Phase:** 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
**Areas discussed:** Scope & structure, Multi-output leaves, Ordered residency, Cholesky + ranking

---

## Scope & Structure

### Phase structure

| Option | Description | Selected |
|--------|-------------|----------|
| One phase, waves | Same as Phase 12 (D-01). Planner decomposes into internal waves; each family flips independently behind Ok(None). No sub-phase ceremony. | ✓ |
| Formal sub-split | 13.1–13.5, each with own discuss/plan/verify/ship cycle. Heavier process. | |

**User's choice:** One phase, waves

### Ambition

| Option | Description | Selected |
|--------|-------------|----------|
| All 5, ordered can slip | Target all five; ordered boosting may stay on CPU fallback if it over-runs, not blocking the phase. | |
| All 5, hard commit | Commit to landing all five device families including ordered boosting, no slip. | ✓ |
| Defer ordered now | Scope ordered boosting OUT up front. | |

**User's choice:** All 5, hard commit
**Notes:** Hard commit to all five including ordered boosting raises the stakes on the ordered-residency approach (see Ordered residency below) — resolved with full device residency (D-05).

---

## Multi-output Leaves

### Leaf shape

| Option | Description | Selected |
|--------|-------------|----------|
| Single tree, block leaves | Mirrors upstream multilogit; extend DeviceGrownTree.leaf_values to leaf_count × approx_dim + approx_dim. Correct for coupled softmax. | ✓ |
| K separate scalar trees | K independent scalar trees, reuse scalar DeviceGrownTree. Diverges from upstream; wrong for coupled softmax MultiClass. | |

**User's choice:** Single tree, block leaves

### Der2 block

| Option | Description | Selected |
|--------|-------------|----------|
| Full multi-row der2 | True multi-row der2 block per leaf — coupled (softmax) + diagonal (separable), matching CPU Newton. | ✓ |
| Diagonal-only first | Separable losses first, coupled softmax as follow-up. Incomplete GPUT-12. | |

**User's choice:** Full multi-row der2

---

## Ordered Residency

### Residency mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Full device residency | Permutation approx state device-resident across iterations; only O(1) descriptors cross the seam. Heaviest, true no-readback. | ✓ |
| Resident approx, host permute | Device-resident approx buffers but host-computed permutation/prefix ordering (fixed per fold). Lighter kernels. | |
| You decide (research) | Defer mechanics to research; lock only device-resident at ε=1e-4. | |

**User's choice:** Full device residency
**Notes:** Consistent with the hard-commit ambition and the milestone's no-readback speed goal.

### Fixture determinism

| Option | Description | Selected |
|--------|-------------|----------|
| Pin seed, freeze permutation | Pin seed/permutation config, freeze CPU-reference permutation + approx trajectory, reproduce bit-for-bit. Same as Phase 12 D-07. | ✓ |
| You decide (research) | Research resolves fixture pinning; lock only deterministic ε=1e-4 frozen check. | |

**User's choice:** Pin seed, freeze permutation

---

## Cholesky + Ranking

### Cholesky solver

| Option | Description | Selected |
|--------|-------------|----------|
| f64 solve + ridge | Batched Cholesky decomposition + forward/back subst in f64 + ridge, matching CalcScoresCholesky. Holds ε=1e-4. | ✓ |
| You decide (research) | Lock only ε=1e-4 match; research resolves precision/pivoting/ridge against §6.3 linear_solver + pairwise_oracle.h. | |

**User's choice:** f64 solve + ridge

### Ranking objective set

| Option | Description | Selected |
|--------|-------------|----------|
| All 5 | QueryRMSE, QuerySoftMax, QueryCrossEntropy, YetiRank, PFound-F on device, incl. stochastic pair (pinned-seed). Complete GPUT-22. | ✓ |
| Deterministic 3 first | Deterministic trio first; YetiRank + PFound-F later wave. Incomplete if stochastic pair slips. | |

**User's choice:** All 5

---

## Claude's Discretion

- Internal wave decomposition/ordering beyond the pinned roadmap sub-order (query-grouping infra and multi-output leaf-block extension likely shared sub-waves).
- Query-grouping infra mechanics (group-bias removal, in-query sampling radix-sort layout, taken-docs masks).
- Cholesky pivoting/ordering + exact ridge/l2 placement (against §6.3 linear_solver + pairwise_oracle.h).
- Langevin Gaussian RNG stream layout + per-element seeding.
- Which specific multiclass fixtures (class counts, coupled vs diagonal) get device oracles.

## Deferred Ideas

- Comprehensive aggregate speed benchmark + real named datasets + >20× gap sign-off — Phase 14 (BENCH-03).
- On-device border/quantile computation (FastGpuBorders) — out of scope milestone-wide.
- Formal 13.1–13.5 sub-phase split — declined (D-01).
- Deterministic-only ranking subset — declined (D-08); lower-risk fallback noted.
- K separate scalar trees for multiclass — declined (D-03).
- Reviewed todo `estimated-feature-grid-parity.md` (KNN/HNSW, FEAT-07/Phase 9) — not folded, out of scope.
