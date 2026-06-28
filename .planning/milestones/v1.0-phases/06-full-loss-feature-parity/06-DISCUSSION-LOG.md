# Phase 6: Full Loss & Feature Parity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-15
**Phase:** 6-full-loss-feature-parity
**Areas discussed:** Decomposition & sequencing, Multi-dim approx refactor, Oracle depth & completeness, Cross-phase scope tensions

---

## Decomposition & Sequencing

### Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Split into sub-phases | Restructure ROADMAP into formal sub-phases, each its own discuss→plan→execute→verify cycle + oracle gate | ✓ |
| One phase, many waves | Single phase, many additive plans/waves grouped by subsystem; one verification at the end | |
| Hybrid: phase with checkpoint gates | One phase with explicit mid-phase oracle milestone gates per subsystem | |

**User's choice:** Split into sub-phases (Recommended)

### Order

| Option | Description | Selected |
|--------|-------------|----------|
| Scalar losses → multidim → rest | 6.1 regression → 6.2 multiclass (N-dim refactor) → 6.3 ranking → 6.4 score/uncertainty/custom → 6.5 text/embedding → 6.6 advanced | ✓ |
| Multidim refactor first | N-dim refactor before everything; front-loads the highest-risk architectural change | |
| Group by risk, defer advanced | All losses+metrics first, then text/embedding, advanced-features cluster last | |

**User's choice:** Scalar losses → multidim → rest (Recommended)
**Notes:** D-01 makes the split a roadmap restructure — `/gsd-phase` is the immediate next step before planning.

---

## Multi-dim approx refactor

### Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Refactor core to N-dim, 1 = degenerate | Generalize the loop so approx is always a vector; scalar = dim=1; single code path | ✓ |
| Parallel multi-dim path | Keep scalar path untouched, add a separate multiclass path | |
| You decide / research-driven | Let research recommend the refactor shape | |

**User's choice:** Refactor core to N-dim, 1 = degenerate case (Recommended)

### Refactor gate

| Option | Description | Selected |
|--------|-------------|----------|
| No-behavior-change checkpoint first | Pure mechanical refactor re-greens ALL existing scalar oracles at dim=1 BEFORE any multiclass math | ✓ |
| Refactor + multiclass together | Do both in one slice, rely on existing oracles to catch regressions | |
| You decide | Planner picks checkpoint granularity | |

**User's choice:** No-behavior-change checkpoint first (Recommended)

---

## Oracle depth & completeness

### Completeness

| Option | Description | Selected |
|--------|-------------|----------|
| Every named loss/metric, oracle-locked | Implement+lock every named loss/metric; "etc." = deferred-to-v2 | ✓ |
| Representative set, defer the long tail | Lock a subset per category, defer the rest | |
| Every loss CatBoost supports | Enumerate the complete upstream registry, lock all | |

**User's choice:** Every named loss/metric, oracle-locked (Recommended)

### Oracle depth

| Option | Description | Selected |
|--------|-------------|----------|
| Python-reachable floor, escalate per-area | Default per-stage Python-reachable parity; C++ instrumentation only on escalation | |
| Final-prediction parity only | Just final predictions ≤1e-5, skip per-stage | |
| C++ instrumentation where helpful | Proactively build instrumented harnesses for trickier categories | ✓ |

**User's choice:** C++ instrumentation where helpful
**Notes:** Deliberately goes beyond the Phase-5 "escalate-only" rule. Selective (not universal) — simple regression leaf-math still rides the Python-reachable per-stage oracle. Disk-pressure feasibility (root ~100% full) is a first-class risk; the Phase-5 `/tmp` clang-18 + `_catboost` toolchain is reused (incremental rebuild).

---

## Cross-phase scope tensions

### Custom objective (LOSS-07)

| Option | Description | Selected |
|--------|-------------|----------|
| Rust trait now, defer Python bridge to Phase 8 | Build+test the Rust trait in 6.4; the PyO3 callback wraps it in Phase 8 | ✓ |
| Build the Python bridge in Phase 6 | Pull a minimal PyO3 seam forward into 6.4 | |
| Move LOSS-07 entirely to Phase 8 | Defer all of custom objectives to Phase 8 | |

**User's choice:** Rust trait now, defer Python bridge to Phase 8 (Recommended)
**Notes:** Python callback bridge captured as a Phase-8 dependency.

### Grow policy / non-symmetric trees (FEAT-06)

| Option | Description | Selected |
|--------|-------------|----------|
| Full parity: non-symmetric train + apply + serialize | Implement Lossguide/Depthwise/Region, apply path, .cbm/json round-trip, oracle-locked | ✓ |
| Train symmetric-equivalent only | Support params only where they stay symmetric; defer true non-symmetric | |
| Split FEAT-06 into its own phase | Pull grow policies out of Phase 6 entirely | |

**User's choice:** Full parity: non-symmetric train + apply + serialize (Recommended)
**Notes:** Effectively a second tree engine; touches cb-model (apply + serialize), likely its own multi-wave structure within 6.6.

### Text/embedding calcer depth (FEAT-01/02)

| Option | Description | Selected |
|--------|-------------|----------|
| All named calcers, oracle-locked | text=BoW/NaiveBayes/BM25, embedding=LDA/KNN; all six ≤1e-5 | ✓ |
| Text first, embedding deferred | Lock text calcers, defer LDA/KNN | |
| You decide / research-driven | Let research recommend the split | |

**User's choice:** All named calcers, oracle-locked (Recommended)
**Notes:** Tokenizer parity is the first risk to nail before scoring the text calcers.

---

## Claude's Discretion

- Exact sub-phase requirement-to-plan mapping and wave structure within each 6.x.
- Exact der1/der2 formulas + parameter defaults per named loss/metric (transcribe from upstream).
- Precise N-dim approx data layout and which kernels/host-reductions change.
- Score-function math for the 5 new EScoreFunction variants.
- Whether 6.6's non-symmetric tree engine needs further internal sub-splitting.
- Which specific categories actually get C++ instrumentation (decided per-category in research, gated by disk feasibility).

## Deferred Ideas

- **LOSS-07 Python callback bridge** → Phase 8 (Rust trait ships 6.4; PyO3 wrapper is a Phase-8 dependency).
- **Any loss/metric not explicitly named in the Phase-6 success criteria** ("etc.") → v2.
