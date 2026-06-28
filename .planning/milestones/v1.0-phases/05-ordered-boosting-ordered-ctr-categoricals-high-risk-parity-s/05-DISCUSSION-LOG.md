# Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-14
**Phase:** 5-Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice)
**Areas discussed:** Per-object oracle strategy, First slice & additive order, CTR type coverage breadth, Fixtures & isolating params

---

## Per-object oracle strategy — source

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone C++ micro-harness | Reuse the Phase-2 cityhash_oracle.cpp pattern: link ONLY the needed upstream TUs (online_ctr.cpp, approx_calcer ordered path), dump per-object CTR/approx to .npy. Disk-safe, true per-object ground truth. | ✓ |
| Python-reachable proxy only | No C++ build; lock final-model ctr_data + staged_predict + metric leakage signature; re-derive per-object CTR offline in Python. Cheapest, per-object ordered-approx remains indirect. | |
| Full catboost C++ build + instrumentation | Build all of catboost-master with hooks. Highest fidelity but MLIR-scale disk footprint — infeasible given disk at ~100%. | |

**User's choice:** Standalone C++ micro-harness
**Notes:** Actions the Phase-3 D-11 "C++ instrumentation deferred primarily to Phase 5" via the lightest viable mechanism; disk constraint (STATE.md) ruled out a full build. → D-01.

## Per-object oracle strategy — depth

| Option | Description | Selected |
|--------|-------------|----------|
| Full per-object stack | Dump permutation indices + per-object running CTR num/denom + per-object ordered approximant per iteration. Strongest leakage detection. | ✓ |
| Permutation + CTR values | Permutation + per-object CTR feature values, no per-iteration ordered approx. | |
| Layered, decide per-slice | Lock permutation first, add depth per additive slice. | |

**User's choice:** Full per-object stack
**Notes:** A prefix-boundary off-by-one is the bug final-prediction parity would mask. Permutation locked as its own stage first (both sides TFastRng64-seeded). → D-02, D-03.

---

## First slice & additive order

| Option | Description | Selected |
|--------|-------------|----------|
| One-hot first (ORD-04) | No permutation, no CTR; rides existing plain boosting. Isolates one_hot_max_size path selection. | ✓ |
| Permutation machinery first (ORD-01) | Fold generation locked by permutation-only oracle; pure-infrastructure first slice, no trainable model. | |
| You decide (research-driven) | Researcher derives dependency-minimal order. | |

**User's choice:** One-hot first (ORD-04)

## First slice & additive order — CTR sequencing

| Option | Description | Selected |
|--------|-------------|----------|
| Plain CTR → Ordered CTR → Ordered boosting | Lock Plain-mode CTR first, then ordered CTR, then ordered boosting. Each ordering layer separable. | ✓ |
| Straight to Ordered (signature path) | Skip a Plain-CTR lock; go directly to ordered CTR + ordered boosting. Fewer fixtures, harder to localize. | |
| You decide (research-driven) | Research determines whether Plain CTR is separable upstream. | |

**User's choice:** Plain CTR → Ordered CTR → Ordered boosting
**Notes:** Tensor/combination CTRs (ORD-05) sequenced last (build on single-feature CTR). → D-04, D-05, D-06.

---

## CTR type coverage breadth

| Option | Description | Selected |
|--------|-------------|----------|
| All 6, each its own oracle | Implement + lock Borders/Buckets/BinarizedTargetMeanValue/FloatTargetMeanValue/Counter/FeatureFreq, each a separate additive fixture. Fully closes ORD-03. | ✓ |
| Defaults first, rest partial | Lock Borders + Counter; mark ORD-03 PARTIAL, defer the other four to Phase 6 (D-12 precedent). | |
| You decide (research-driven) | Research recommends breadth after reading ctr enums. | |

**User's choice:** All 6, each its own oracle
**Notes:** Continues the Phase 3–4 breadth pattern. Research may confirm shared math (Counter/FeatureFreq) to right-size fixtures. → D-07.

---

## Fixtures & isolating params — corpus

| Option | Description | Selected |
|--------|-------------|----------|
| New purpose-built categorical fixtures | Controlled cardinality (low-card forces one-hot, high-card forces CTR), small N for auditable per-object prefixes. Reuse existing corpora only where they fit. | ✓ |
| Reuse numeric_categorical | Drive from the existing corpus, extend with new expected-output fixtures. | |
| You decide (research-driven) | Research inspects existing corpus cardinality and generates new only where needed. | |

**User's choice:** New purpose-built categorical fixtures
**Notes:** → D-08, D-09.

## Fixtures & isolating params — config pin

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit isolating params | Pin boosting_type, simple_ctr/combinations_ctr + prior, fold_count, one_hot_max_size, max_ctr_complexity, thread_count=1 per fixture. | |
| Upstream auto-selection | Let upstream auto-pick boosting_type + default CTR config. | |
| You decide (research-driven) | Capture the isolating-params principle; research pins exact values after reading defaults. | ✓ |

**User's choice:** You decide (research-driven)
**Notes:** Captured as Claude's discretion BUT bound to the D-07 isolating-params principle (explicit per-slice pins, reject auto-selection); exact values set after research reads upstream defaults. → Claude's Discretion in CONTEXT.md.

---

## Claude's Discretion

- Exact training-config pin per fixture (research-driven, bound to D-07 isolating-params principle).
- Permutation-generation draw order + TFold body/tail boundaries; ordered-CTR online accumulation order + per-type priors; ordered-boosting approximant update; one-hot vs CTR threshold semantics; TProjection combination enumeration + max_ctr_complexity; populating/applying the .cbm/model.json ctr_data section; whether new work needs a cb-backend kernel.
- **Build feasibility of isolated TU linking in the micro-harness — research must validate early; escalate before planning if impractical (the oracle strategy rests on D-01).**

## Deferred Ideas

- Full catboost C++ build with in-tree instrumentation (rejected; micro-harness supersedes).
- Multiclass/ranking-aware ordered statistics, text/embedding, advanced fstr, uncertainty, monotone constraints, feature selection, alternative grow policies → Phase 6.
- LossFunctionChange importance (Phase-4 D-12, still deferred).
- GPU CTR/permutation kernels → Phase 7.
- Broader .cbm cross-version load tolerance for CTR models → later hardening.
