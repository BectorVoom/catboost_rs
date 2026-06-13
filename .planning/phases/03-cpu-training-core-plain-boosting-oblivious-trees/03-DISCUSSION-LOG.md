# Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 3-CPU Training Core — Plain Boosting & Oblivious Trees
**Areas discussed:** Compute boundary design, Threading & determinism, First oracle-locked slice, Intermediate-oracle depth

---

## Compute boundary design

### CPU compute path

| Option | Description | Selected |
|--------|-------------|----------|
| Plain Rust, CubeCL-shaped trait | Abstract trait; CPU impl plain scalar Rust, no cubecl; designed CubeCL-implementable for P7. Exact f64 control. | |
| CubeCL CPU runtime now | Use CubeCL's CPU runtime as the P3 backend; one CPU→GPU code path; proves the seam early. | ✓ |
| Plain Rust, CPU-native trait | Trait around CPU needs; let P7 adapt CubeCL. Simplest now, risks P7 re-design. | |

**User's choice:** CubeCL CPU runtime now.
**Notes:** Deliberately more ambitious — validate the CubeCL seam in Phase 3. Surfaced an immediate conflict with the P2 D-08 reduction invariant, resolved below.

### Reductions vs the order-locked cb-core::sum_f64 invariant

| Option | Description | Selected |
|--------|-------------|----------|
| Kernels elementwise, host-side ordered reduce | Kernels do order-independent work; parity-critical sums finalized host-side via cb-core::sum_f64 in frozen order. | ✓ |
| Reproduce sum order inside kernels | Deterministic sequential reduction kernels; defeats parallelism; codegen may not guarantee f64 order. | |
| Redefine the invariant around CubeCL | Treat CubeCL CPU reduction as canonical; re-derive sum_f64; re-opens locked P2 decision. | |

**User's choice:** Kernels elementwise, host-side ordered reduce.
**Notes:** Preserves P2 D-07/D-08 exactly; the parity safety net for D-01.

### Kernel location vs cb-compute's role (P1 D-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Kernels in cb-backend, cb-compute abstract | cb-compute = own abstract R: Runtime/F: Float traits + host orchestration, no cubecl; #[cube] kernels + cubecl only in cb-backend. | ✓ |
| Kernels in cb-compute, relax D-03 | cb-compute holds #[cube] kernels over cubecl's Runtime/Float; depends on cubecl; re-opens P1 boundary. | |
| Split: traits in cb-compute, kernels in cb-backend, share cubecl::Float | Hybrid; cubecl as types-only dep in cb-compute. | |

**User's choice:** Kernels in cb-backend, cb-compute abstract.
**Notes:** Honors P1 D-03 literally.

### Trait granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Coarse domain ops | ML-level ops (build_histograms, compute_gradients, eval_splits); backend owns decomposition; host finalizes ordered sums. | ✓ |
| Fine primitives | Generic map/scan/reduce/scatter; cb-compute composes; pushes reduction shape into generic layer. | |
| You decide | Leave to planning. | |

**User's choice:** Coarse domain ops.

---

## Threading & determinism

### Histogram per-bin accumulation determinism

| Option | Description | Selected |
|--------|-------------|----------|
| Ordered single-pass f64 bins | f64 per-bin accumulators in canonical object order; bit-matches upstream thread_count=1; kernel does parallel per-object gradients. | ✓ |
| Parallel partial-histograms + ordered merge | Parallel partials merged in fixed order; only matches if upstream merge order reproduced; higher drift risk. | |
| You decide | Leave to research. | |

**User's choice:** Ordered single-pass f64 bins.

### Phase 3 threading target

| Option | Description | Selected |
|--------|-------------|----------|
| Single-thread end-to-end now | CubeCL CPU runtime single-threaded; smallest validation surface; parallelism a later perf pass. | |
| Parallel elementwise, ordered reductions | Parallel order-independent kernels now; reductions ordered; exercises CubeCL parallelism early, de-risks P7. | ✓ |
| You decide | Leave to planning. | |

**User's choice:** Parallel elementwise, ordered reductions.
**Notes:** Consistent with the CubeCL-first stance (D-01); determinism comes solely from the host-side ordered reductions.

---

## First oracle-locked slice

### Parameter posture

| Option | Description | Selected |
|--------|-------------|----------|
| Simplified isolating params first | bootstrap_type=No, random_strength=0, fixed l2/depth/lr; each knob added as its own oracle. | ✓ |
| Full upstream defaults immediately | Bayesian bootstrap + random_strength=1 + l2=3 engaged at once; harder to localize drift. | |
| You decide | Leave to planning. | |

**User's choice:** Simplified isolating params first.

### Anchor loss

| Option | Description | Selected |
|--------|-------------|----------|
| RMSE regression | Simplest math; isolates machinery; Logloss follows additively. | |
| Logloss binary classification | Headline classification; sigmoid+Newton on unproven core. | |
| Both RMSE + Logloss | Regression + binary classification together; broadest first proof. | ✓ |

**User's choice:** Both RMSE + Logloss.
**Notes:** Lines up with Phase 4's binary-clf + regression lock; user accepted the broader first-slice surface.

### Leaf-method rollout

| Option | Description | Selected |
|--------|-------------|----------|
| Default method first, others additive | Upstream-default method (likely Newton) first, then add Gradient/Exact/Simple each as its own oracle. | |
| All four up front | Implement Gradient/Newton/Exact/Simple together; complete TRAIN-03 in one wave. | ✓ |
| You decide | Leave to planning. | |

**User's choice:** All four up front.

---

## Intermediate-oracle depth

| Option | Description | Selected |
|--------|-------------|----------|
| Python-reachable floor | Per-tree splits + leaf values + per-iteration approximants from save_model(json) + staged_predict; no C++ build; drift localized by simplified first slice. | ✓ |
| Add C++ instrumentation now | Also extract per-bin histograms/per-object gradients via instrumenting catboost-master; pulls deferred C++ build forward. | |
| You decide | Leave to research. | |

**User's choice:** Python-reachable floor.
**Notes:** Honors P1 D-08 deferral of C++ instrumentation.

---

## Claude's Discretion

Parity-dictated specifics flowed to research/planning (see CONTEXT.md `<decisions>` → Claude's Discretion): exact split-score/gain formula and oblivious tree growth, leaf-value math for the four methods, per-loss gradient/hessian (RMSE/Logloss), split tie-break determinism (known landmine), bootstrap/sampling draw order, overfitting-detector statistics, auto-LR coefficient tables, eval-metric set/logging shape, concrete `cubecl` version.

## Deferred Ideas

- Parallel/multi-threaded performance tuning (Phase 3 judged on parity, not perf).
- C++ instrumentation for iteration-internal oracles (deferred per P1 D-08, primarily Phase 5).
- GPU kernels — Phase 7, additively on the cb-compute seam.
- Ordered boosting / ordered CTR / categoricals (Phase 5); serialization / SHAP / Builder API (Phase 4).
