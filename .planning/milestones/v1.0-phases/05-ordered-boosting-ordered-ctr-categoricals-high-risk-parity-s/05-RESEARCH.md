# Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice) - Research

**Researched:** 2026-06-14
**Domain:** CatBoost anti-leakage training internals — multi-permutation folds, ordered boosting, ordered/online target statistics (CTR), native categorical handling (one-hot + tensor CTRs), and model-side `ctr_data` populate+apply.
**Confidence:** HIGH on algorithm transcription (every parity claim cited to `catboost-master/` 1.2.10 file:line); HIGH (negative) on the D-01 micro-harness-by-TU-isolation feasibility verdict — **it is NOT feasible the cityhash way; see ESCALATION below**; MEDIUM on the proposed fallback oracle strategy (needs user sign-off before planning).

> **⚠️ PLANNING BLOCKER — read first.** Research-priority #1 (D-01 TU-isolation feasibility) resolves **NEGATIVE**. The Phase-2 `cityhash_oracle.cpp` precedent works because CityHash is a *leaf utility with zero catboost includes* — it was transcribed, not linked. `online_ctr.cpp` and the ordered path in `approx_calcer.cpp` transitively pull in the entire training-context object graph (`TLearnContext`, `TTrainingDataProviders`, `NPar::ILocalExecutor`, `TScratchCache`, `IDerCalcer`, the full options system, `metrics/metric.h`). You **cannot** link "only the needed TUs" — they have no isolatable boundary. The whole D-01 oracle strategy as written does not stand. A concrete fallback (transcribe-then-self-oracle, mirroring exactly how cityhash actually worked) is proposed in **§ Per-Object Oracle Strategy**. This needs a discuss-phase decision before task planning. The rest of this document is written so planning can proceed *the moment the fallback is accepted* — the algorithm transcription is complete and unaffected by the oracle-sourcing decision.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Per-object ground truth comes from a standalone C++ micro-harness, NOT a full catboost build and NOT Python alone. Reuse the Phase-2 precedent (`cityhash_oracle.cpp`): a small C++ tool that links ONLY the specific upstream translation units needed (`online_ctr.cpp`, the ordered path in `approx_calcer.cpp`, fold/permutation generation) and dumps per-object internals to `.npy`. *(Research finds the "link only needed TUs" mechanism infeasible — see ESCALATION; the intent (offline standalone C++ ground truth committed as `.npy`) is preserved by the fallback.)*
- **D-02:** Per-object oracle schema is the FULL per-object stack: (1) permutation indices per fold; (2) per-object running CTR numerator/denominator under the prior formula `(sumTarget + prior)/(sumCount + priorWeight)`; (3) per-object ordered approximant per iteration. Rust asserts each ≤1e-5.
- **D-03:** Permutation is the linchpin, locked as its own stage. Both impls are seeded by the bit-exact `TFastRng64` port. Rust asserts (a) permutation reproduces upstream exactly, THEN (b) per-object CTR/approx values match *under that permutation*. Permutation reproduction locked before any value comparison.
- **D-04:** One-hot first (ORD-04) is the narrowest first slice — no permutation, no CTR; rides existing plain boosting + oblivious trees.
- **D-05:** Additive spine: ORD-01 permutation/fold machinery → Plain CTR → Ordered CTR → Ordered boosting (ORD-02) → tensor/combination CTRs (ORD-05) last.
- **D-06:** Plain CTR locked before Ordered CTR (the key isolation). Lock Plain-mode CTR (whole-dataset target stat, no per-object prefix) first, THEN ordered-permutation CTR, THEN ordered boosting.
- **D-07:** All six CTR types implemented and oracle-locked this phase, each as its own additive fixture. Research confirms which share math to right-size fixtures, but every type gets its own oracle.
- **D-08:** New purpose-built categorical fixtures: a low-cardinality column (forces one-hot below `one_hot_max_size`), a high-cardinality column (forces CTR), small N for human-auditable prefixes. Reuse `numeric_categorical`/`explicit_categorical` only where they already fit.
- **D-09:** Frozen-committed-fixtures convention holds. Generators run OFFLINE; `.npy`/config committed under `crates/cb-oracle/fixtures/`; neither runs in CI. Pinned `catboost==1.2.10`, `thread_count=1`.

### Claude's Discretion
- Exact training-config pin per fixture — research-driven, MUST follow D-07 isolating-params: pin every knob explicitly (`boosting_type` Plain/Ordered **not** auto, explicit `simple_ctr`/`combinations_ctr` type + prior, small fixed `fold_count`/`permutation_count`, explicit `one_hot_max_size`, `max_ctr_complexity`, `thread_count=1`). Reject upstream auto-selection.
- Exact permutation-generation draw order and `TFold` bookkeeping — `fold.cpp`, `learn_context.cpp`.
- Exact ordered-CTR online accumulation order and prior/`priorWeight` defaults per type — `online_ctr.cpp/.h`, `cat_feature_options.*`, `ctr_config.h`.
- Exact ordered-boosting approximant update — `approx_calcer.cpp` ordered path.
- One-hot vs CTR path-selection threshold semantics — `cat_feature_options.*`, `greedy_tensor_search.cpp`.
- Tensor/projection enumeration and `max_ctr_complexity` — `projection.h`, `greedy_tensor_search.cpp`.
- Populating + applying the `.cbm`/`model.json` `ctr_data` section — `ctr_provider.h`, `online_ctr.h` (model-side), `ctr_data.fbs`.
- Whether new CTR/permutation work needs a `cb-backend` kernel or is pure host orchestration.
- Build feasibility of linking `online_ctr.cpp`/`approx_calcer.cpp` in isolation — **research must validate early; if impractical, escalate.** → **Done: impractical. Escalated below.**

### Deferred Ideas (OUT OF SCOPE)
- Full catboost C++ build with in-tree instrumentation (disk infeasibility, D-01).
- Multiclass/ranking-aware ordered statistics, text/embedding features, advanced fstr, uncertainty, monotone constraints, feature selection, alternative grow policies — Phase 6.
- LossFunctionChange feature importance — still deferred (Phase-4 D-12).
- GPU CTR/permutation kernels — Phase 7.
- Broader `.cbm` cross-version CTR-model load tolerance (beyond 1.2.10) — later.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ORD-01 | Multi-permutation fold machinery (`fold_count`/`permutation_count` permutations, `TFold`-equivalent bookkeeping) | § Permutation/Fold Machinery — `Shuffle` Fisher-Yates draw order (`util/random/shuffle.h:24-32`), `CountLearningFolds = max(1, permutation_count-1)` (`learn_context.cpp:48-49`), `TFold::BuildDynamicFold` body/tail prefixes (`fold.cpp:99-211`), `cb-core::TFastRng64::uniform` already bit-exact |
| ORD-02 | Ordered boosting (`EBoostingType::Ordered`), exact prefix boundaries, per-object intermediate oracle | § Ordered Boosting Approximant Update — `CalcApproxDeltaSimple` (`approx_calcer.cpp:706`), `UpdateApproxDeltasHistoricallyImpl` (`:566-600`), body/tail prefix split (`fold.cpp:156-198`) |
| ORD-03 | Ordered target statistics / CTR — six types with priors | § The Six CTR Types — `online_ctr.cpp` dispatch (`:732-800`), `CalcCTR` (`online_ctr.h:128-131`), per-type math, default priors (`cat_feature_options.cpp:118-138`) |
| ORD-04 | One-hot encoding (`one_hot_max_size`) | § One-Hot Path Selection — `AddOneHotFeatures` (`greedy_tensor_search.cpp:171-197`); used when `1 < count <= one_hot_max_size` |
| ORD-05 | Feature combinations (tensor CTRs), `max_ctr_complexity` | § Tensor / Combination CTRs — `AddTreeCtrs` (`greedy_tensor_search.cpp:491-551`), `TProjection.GetFullProjectionLength` (`projection.h:138-144`), `max_ctr_complexity` gate (`:532-533`) |
</phase_requirements>

## Summary

CatBoost's anti-leakage core is two cooperating mechanisms, both keyed on a per-fold random permutation seeded by `TFastRng64`: **ordered CTR** (target statistics computed only from a document's *predecessors* in the permutation) and **ordered boosting** (gradient/approx for a document computed from a model trained only on its predecessors). Both reduce to "accumulate-then-read in permutation order, never including self." The algorithm transcription is fully captured here from `catboost-master/` 1.2.10 with file:line citations.

The single highest-risk planning item is the **oracle-sourcing strategy (D-01), which research finds broken as written**: the two research-flag TUs cannot be linked in isolation (they transitively include the full training context). The viable fallback — and what the cityhash precedent *actually* did — is **transcribe the algorithm into a standalone C++ harness with zero catboost includes, then self-oracle** the transcription against a small set of Python-reachable anchors plus the final-model `ctr_data` (which the model.json/.cbm *does* expose, and which `static_ctr_provider.cpp` shows exactly how to interpret). This preserves D-02's per-object depth (the harness still dumps per-object internals to `.npy`) while removing the impossible link step.

A second, equally load-bearing finding: **on CPU, `boosting_type` defaults to `Plain`, not Ordered** (`boosting_options.cpp:16`); the famous "small dataset → Ordered" auto-selection is **GPU-only** (`catboost_options.cpp:802-806`). And **permutation is needed in Plain mode too** whenever any cat feature exceeds `one_hot_max_size` (because CTRs are permutation-dependent) — `learn_context.cpp:42-45,71-73`. These two facts directly enable the D-04/D-06 isolation ladder and must be pinned per fixture.

**Primary recommendation:** Before any task planning, run a discuss-phase to accept the **transcribe-then-self-oracle** fallback for D-01 (the link-in-isolation mechanism is dead). Then plan the D-05 additive spine exactly as locked, using the transcribed algorithm in this document; gate the `ctr_data` `.cbm`/`model.json` round-trip on the existing Phase-4 substrate (which currently parses only borders, not `ctr_data`).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Permutation generation (`TFastRng64` Fisher-Yates) | `cb-train` (host orchestration) | `cb-core` (RNG primitive — already exists) | Order-dependent; pure host; reuses `cb-core::TFastRng64::uniform` |
| Fold body/tail prefix bookkeeping (`TFold`) | `cb-train` | — | Pure host state machine over the permutation |
| Online CTR accumulation (prefix counts) | `cb-train` (host, order-locked) | `cb-core::sum_f64` for any float sum | Inherently sequential per permutation; **not** kernel-eligible (order-dependent) |
| CTR quantization (`CalcCTR` shift/norm/border) | `cb-train` | `cb-backend` (optional elementwise kernel — order-independent) | Per-doc CTR→bin is elementwise; could be a kernel, but trivial — keep host first |
| Ordered approximant update | `cb-train` | `cb-core::sum_f64` | Sequential prefix accumulation; host-only |
| One-hot encoding path selection | `cb-train` (candidate generation) | — | Pure threshold logic on cardinality |
| Tensor/projection enumeration | `cb-train` | — | Combinatorial host logic |
| Final `ctr_data` table build (`CalcFinalCtrsImpl`) | `cb-train` → `cb-model` | `cb-core::sum_f64` | Whole-learn-set counts baked into the model |
| Model-side `ctr_data` (de)serialize | `cb-model` (`.cbm`/`json`) | existing `ctr_data_generated.rs` bindings | Extends Phase-4 serde substrate |
| Model-side CTR apply at inference | `cb-model` (`apply.rs`) | `cb-data::calc_cat_feature_hash` + projection `CalcHash` | Extends Phase-4 `predict_raw` |
| Categorical hashing for CTR consumers | `cb-data::calc_cat_feature_hash` | — | Carried-forward D; CTRs consume this, never a model's `ctr_data` hash_map |

**Kernel verdict (Claude's-discretion item):** Ordered/online CTR accumulation and ordered approximant updates are **order-dependent and must stay host-side** (consistent with Phase-3 D-02/D-05 host-ordered-reduce invariant). The only kernel-eligible slice is the per-doc `CalcCTR` quantization (elementwise), and even that is trivial enough to defer to `cb-backend` only if a measured need arises. **Recommend: pure host orchestration for Phase 5; no new `cb-backend` kernel.** [VERIFIED: cross-checked against Phase-3 D-02/D-03 carried-forward + `online_ctr.cpp` sequential accumulation structure]

## Standard Stack

This phase adds **no new external crates** — it is pure-Rust algorithm transcription against the existing workspace. The "stack" is the vendored upstream source (the oracle) plus the existing crates.

### Core (existing crates, extended)
| Crate | Role this phase | Why |
|-------|-----------------|-----|
| `cb-core` | `TFastRng64` (permutation seed), `sum_f64`/`sum_f32_in_f64` (all CTR/approx sums) | Already bit-exact (RNG `uniform` verified); D-08 CI-grep ban applies to all new accumulation |
| `cb-data` | `calc_cat_feature_hash`, `Pool` categorical columns | CTR consumers hash via this; projection hashing builds on it |
| `cb-train` | permutation/fold machinery, online CTR, ordered approx, one-hot, tensor CTR — the bulk of the phase | Extends `boosting.rs`/`tree.rs` |
| `cb-model` | `ctr_data` (de)serialize + apply | Extends Phase-4 `.cbm`/`json`/`apply.rs`; bindings already committed |
| `cb-oracle` | `compare_stage` ≤1e-5 gate, `.npy` fixtures, `model_json.rs` | New `Stage` variants + `ctr_data` parsing needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Transcribe-then-self-oracle (fallback) | Link upstream TUs in isolation (original D-01) | **Rejected — infeasible** (transitive header weight, see ESCALATION) |
| Transcribe-then-self-oracle | Full catboost C++ build + instrumentation | Rejected by D-01 (disk: 21G free, MLIR-scale build won't fit reliably) |
| Reuse `numeric_categorical`/`explicit_categorical` corpora | Purpose-built tiny categorical fixtures | Per D-08, build purpose-built where existing corpora fail to isolate; reuse where they fit |

**Installation:** None. (No new packages — confirmed against CLAUDE.md "always use the latest crate versions" rule: nothing to add.)

## Package Legitimacy Audit

> Not applicable — this phase installs **no external packages**. All work is pure-Rust transcription against existing workspace crates and the vendored `catboost-master/` oracle. The Python generator pins `catboost==1.2.10` (already established, D-09) and runs offline only.

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                        ┌─────────────────────────────────────────────────────────┐
  Pool (cat columns) ──▶│ cb-data: calc_cat_feature_hash (CityHash64 & 0xffffffff) │
                        └───────────────────────┬─────────────────────────────────┘
                                                │ hashed cat values (ui32)
                                                ▼
                  ┌──────────────────────────────────────────────────────┐
                  │ cardinality check: uniqCount vs one_hot_max_size       │
                  └───────┬───────────────────────────────┬───────────────┘
            1<count<=ohms │                                │ count>ohms
                          ▼                                ▼
           ┌──────────────────────┐        ┌────────────────────────────────────────┐
           │ ONE-HOT path (ORD-04)│        │ CTR path (ORD-03/05)                     │
           │ ESplitType::OneHot   │        │  permutation (TFastRng64 Fisher-Yates)   │◀── seed
           │ no permutation,no CTR│        │              │                            │
           └──────────┬───────────┘        │              ▼                            │
                      │                     │  TFold body/tail prefixes (fold.cpp)     │
                      │                     │              │                            │
                      │                     │   ┌──────────┴───────────┐               │
                      │                     │   ▼                      ▼               │
                      │                     │ PLAIN CTR            ORDERED CTR          │
                      │                     │ whole-set stat       per-doc PREFIX stat  │
                      │                     │ (CalcFinalCtrsImpl)  (calcGoodCounts:     │
                      │                     │                       read-before-incr)   │
                      │                     │   │                      │               │
                      │                     │   └──────────┬───────────┘               │
                      │                     │              ▼                            │
                      │                     │   CalcCTR shift/norm/borderCount → ui8 bin│
                      │                     └──────────────┬───────────────────────────┘
                      ▼                                    ▼
           ┌────────────────────────────────────────────────────────────────┐
           │ cb-train: candidate splits → oblivious tree growth → leaf values │
           │   Plain boosting (CalcLeafValuesSimple, whole fold)              │
           │   Ordered boosting (CalcApproxDeltaSimple, body/tail prefix)     │
           └───────────────────────────┬────────────────────────────────────┘
                                        ▼
           ┌────────────────────────────────────────────────────────────────┐
           │ cb-model: bake ctr_data (TCtrValueTable per TModelCtrBase) into  │
           │   .cbm / model.json  (PriorNum/PriorDenom/Shift/Scale + blob)    │
           └───────────────────────────┬────────────────────────────────────┘
                                        ▼ inference
           ┌────────────────────────────────────────────────────────────────┐
           │ cb-model apply: CalcHashes(projection) → bucket → TModelCtr.Calc │
           │   = (countInClass+PriorNum)/(totalCount+PriorDenom) ; +Shift;*Scale│
           └────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure (within existing crates)
```
crates/cb-train/src/
├── permutation.rs     # TFastRng64 Fisher-Yates shuffle, block-aware (fold.cpp:43-96 + shuffle.h)
├── fold.rs            # TFold body/tail prefix state (fold.cpp:99-211)
├── ctr/
│   ├── online.rs      # ordered/plain online accumulation (online_ctr.cpp)
│   ├── calc_ctr.rs    # CalcCTR shift/norm/borderCount (online_ctr.h:102-131)
│   └── final_ctr.rs   # CalcFinalCtrsImpl whole-set table build (online_ctr.cpp:875-940)
├── projection.rs      # TProjection + GetFullProjectionLength (projection.h)
├── candidates.rs      # one-hot vs CTR path selection (greedy_tensor_search.cpp:171-551)
├── boosting.rs        # (extend) ordered vs plain leaf-value/approx path
└── tree.rs            # (extend) categorical splits
crates/cb-model/src/
├── ctr_data.rs        # (new) TCtrValueTable serde over generated bindings
└── apply.rs           # (extend) model-side CTR Calc at inference
crates/cb-oracle/src/
├── compare.rs         # (extend) Stage::Permutation, Stage::OnlineCtr, Stage::OrderedApprox
└── model_json.rs      # (extend) parse ctr_data section (currently borders-only)
```

### Pattern 1: Online (ordered) CTR — read-before-increment
**What:** For each document in permutation order, read the accumulated class counts for its bucket, compute the CTR, THEN increment the bucket with this document's own class. The "read before increment" *is* the no-leakage property.
**When to use:** Borders/Buckets/BinarizedTargetMeanValue/FloatTargetMeanValue (the permutation-dependent types).
**Example:**
```cpp
// Source: catboost-master/catboost/private/libs/algo/online_ctr.cpp:168-184 (CalcOnlineCTRClasses)
auto calcGoodCounts = [&](int blockStart, int nextBlockStart, int datasetIdx) {
    auto docOffset = datasetIdx ? testOffsets[datasetIdx - 1] : 0;
    for (int docId = blockStart; docId < nextBlockStart; ++docId) {
        const auto elemId = enumeratedCatFeatures[docOffset + docId];
        int goodCount = totalCountByDoc[docId - blockStart] = bv.GetTotal(elemId); // READ prefix
        auto bordersData = bv.GetBorders(elemId);
        for (int border = 0; border < targetBorderCount; ++border) {
            UpdateGoodCount(bordersData[border], ctrType, &goodCount);
            goodCountByBorderByDoc[border][docId - blockStart] = goodCount;
        }
        if (datasetIdx == 0) {              // learn set only:
            ++bordersData[permutedTargetClass[docId]];  // INCREMENT after read
            ++bv.GetTotal(elemId);
        }
    }
};
```
For the simple binclf path (`CalcQuantizedCtrs`, `online_ctr.cpp:300-307`): `goodCount=elem[1]; totalCount=elem[0]+elem[1]; ++elem[permutedTargetClass[...]]` — same read-before-increment, `N[0]`/`N[1]` = neg/pos class counts.

### Pattern 2: CTR value computation (`CalcCTR`)
**What:** Quantize the (countInClass, totalCount, prior) triple into a `ui8` bin via shift/norm/borderCount.
**Example:**
```cpp
// Source: catboost-master/catboost/private/libs/algo/online_ctr.h:128-131
inline ui8 CalcCTR(float countInClass, int totalCount, float prior, float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);   // NOTE: online denom is hard +1
    return (ctr + shift) / norm * borderCount;
}
// shift/norm from CalcNormalization (online_ctr.cpp:102-111):
//   left = min(0, prior); right = max(1, prior); shift = -left; norm = right - left;
```
**Critical distinction (document for the planner):** the **online (training)** `CalcCTR` uses denominator `totalCount + 1` (online_ctr.h:129). The **model-side (inference)** `TModelCtr::Calc` uses `(countInClass + PriorNum) / (totalCount + PriorDenom)` then `(ctr + Shift) * Scale` (online_ctr.h:289-292). These are *different normalizations* — the success-criteria phrase `(sumTarget + prior)/(sumCount + priorWeight)` matches the **inference** form (`priorWeight` = `PriorDenom`). For default priors like `0/1`, `0.5/1`, `1/1`, `PriorDenom = 1`, so they coincide numerically with the online `+1`; but the planner must not assume they are the same code path.

### Pattern 3: Body/tail ordered-boosting prefix
**What:** The fold is split into a growing sequence of body/tail segments; leaf deltas are estimated on the body and applied to the tail, so a tail document's approximant never depends on itself.
**Example:**
```cpp
// Source: catboost-master/catboost/private/libs/algo/fold.cpp:35-41 + 156-198 (BuildDynamicFold)
static ui32 SelectMinBatchSize(ui32 n)        { return n > 500 ? Min<ui32>(100, n/50) : 1; }
static double SelectTailSize(ui32 old, double mult) { return ceil(old * mult); } // mult = fold_len_multiplier (default 2.0)
// loop: bodyFinish = leftPartLen; tailFinish = ceil(leftPartLen * mult) (capped at n);
//        leftPartLen = tailFinish;  // until leftPartLen >= n
// Plain boosting uses a single body/tail spanning the whole fold instead (BuildPlainFold).
```

### Anti-Patterns to Avoid
- **Using a model's `ctr_data` hash_map as the categorical hash source.** It stores CTR-*projection* hashes (`CalcHash`/`MultiHash` folding, `ctr_provider.h:65-78`), NOT `CalcCatFeatureHash`. CTR consumers MUST use `cb-data::calc_cat_feature_hash`. [VERIFIED: STATE.md Plan 02-04 Rule-1 fix + `ctr_provider.h` CalcHash]
- **Computing CTR after incrementing the bucket.** Leaks the document's own label — the exact bug this phase's per-object oracle exists to catch.
- **Assuming Ordered boosting is the CPU default.** It is `Plain` (`boosting_options.cpp:16`); pin `boosting_type` explicitly per fixture.
- **Hand-rolling float sums.** D-08 CI-grep bans any accumulation outside `cb-core::sum_f64`/`sum_f32_in_f64` — applies to CTR running sums, priors, leaf stats, ordered approximants. **Caveat:** the *integer* class counts (`TCtrHistory::N[2]` are `int`) are exact integer accumulation, not floats — they do not route through `sum_f64`. Only float sums (`TCtrMeanHistory::Sum` is `float`, leaf ders) do.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Permutation RNG draw | A new uniform sampler | `cb-core::TFastRng64::uniform(bound)` | Already bit-exact port of `NPrivate::GenUniform` (common_ops.h:48-60); verified in `rng_test.rs` |
| Categorical hashing | CityHash re-port | `cb-data::calc_cat_feature_hash` | Already bit-exact (Plan 02-04); ui32 = CityHash64 & 0xffffffff, first-seen perfect-hash bins |
| Float reduction | Manual `+=` loops | `cb-core::sum_f64` | D-08 CI-grep ban; order-locked accumulator |
| FlatBuffers `ctr_data` schema | Re-derive `.fbs` bindings | `cb-model::generated::ctr_data_generated` | Already committed (Phase-4 04-01); covers `TModelCtr`/`TCtrFeature`/`TCtrValueTable`/`ECtrType` |
| Per-stage ≤1e-5 comparison | New assert helper | `cb-oracle::compare_stage` | Existing 1e-5 gate; just add `Stage` variants |

**Key insight:** ~80% of the primitives this phase needs already exist and are oracle-locked from Phases 1–4. The genuinely new work is (a) the permutation/fold state machine, (b) the online CTR accumulation loops, (c) the ordered approximant prefix, (d) `ctr_data` serde+apply, and (e) the per-object oracle harness. Do not re-port primitives.

## Per-Object Oracle Strategy (D-01 ESCALATION + Fallback)

### ESCALATION — the locked D-01 mechanism is infeasible

**Claim:** `online_ctr.cpp` and the ordered path in `approx_calcer.cpp` **cannot be compiled/linked in isolation** the way `cityhash_oracle.cpp` was. [VERIFIED: include analysis of both TUs]

**Evidence:**
- `cityhash_oracle.cpp` includes only `<cstdint>`, `<cstring>`, `<string>`, `<iostream>` — **zero catboost headers**. It is a *transcription*, not a link against the upstream TU. The README and file header even say so ("a standalone, dependency-free transcription").
- `online_ctr.cpp` includes (online_ctr.cpp:1-26): `fold.h`, `index_hash_calcer.h`, `learn_context.h`, `scoring.h`, `tree_print.h`, `catboost/libs/model/model.h`, `local_executor.h`, plus the `TTrainingDataProviders`/`TScratchCache`/`TLearnContext`/`TCtrHelper` object graph. There is no compilation unit boundary that excludes the trainer.
- `approx_calcer.cpp` is worse — 35 includes (approx_calcer.cpp:1-39) including `metrics/metric.h`, `optimal_const_for_loss.h`, the full `options/catboost_options.h`, `error_functions.h`, `gradient_walker.h`, and the multi-approx machinery. It depends on `IDerCalcer`, `TLearnContext`, `IMetric`.
- Linking these "alone" transitively requires building essentially all of `catboost/private/libs/algo` + `libs/model` + `libs/metrics` + `libs/data` + the YT `util`/`library` runtime — i.e. the full build D-01 explicitly rejected on disk grounds (21G free; MLIR-scale build already filled disk during Plan 03-01, per STATE.md Blockers).

**Conclusion:** "link ONLY the specific upstream translation units needed" has no isolatable target. The cityhash precedent does not generalize to these TUs because cityhash was a leaf utility and these are mid-graph trainer internals.

### Proposed fallback (needs discuss-phase sign-off before planning)

**Transcribe-then-self-oracle** — exactly what cityhash *actually* did, generalized to per-object internals:

1. **Transcribe** the four small algorithms in this document into a standalone C++ harness (`crates/cb-oracle/generator/ordered_oracle.cpp`) with **zero catboost includes**, copying verbatim:
   - the `Shuffle` Fisher-Yates loop (`shuffle.h:24-32`) over a transcribed `TFastRng64` (or reuse the already-validated `cityhash`-style transcription pattern),
   - the online CTR `read-before-increment` + `CalcCTR` (online_ctr.cpp:168-184/300-307 + online_ctr.h:102-131),
   - the body/tail prefix (`fold.cpp:35-41,156-198`),
   - the ordered approximant prefix update (`approx_calcer.cpp:566-600`).
   The harness reads a committed fixture config + input `.npy`, and dumps per-object `.npy`: permutation indices, per-doc `(goodCount,totalCount)` numerator/denominator, per-iteration ordered approx.
2. **Self-oracle the transcription** against independently-derived anchors so the transcription itself is trustworthy (this is the crux — a transcription is only as good as its validation):
   - **Permutation (D-03 linchpin):** validate the harness permutation against the **already-oracle-locked `cb-core::TFastRng64`** by reproducing the same Fisher-Yates draw in Rust and asserting equality. Since `TFastRng64::uniform` is bitstream-verified (`rng_test.rs`), a Rust reproduction of `Shuffle` is itself ground truth for the permutation — the C++ harness becomes a *cross-check*, not the sole source.
   - **Final CTR values:** validate the harness's whole-set counts against the **upstream model's `ctr_data`** — the trained `.cbm`/`model.json` *does* expose `TCtrValueTable` blobs (counts per bucket), and `static_ctr_provider.cpp:14-126` documents exactly how to interpret them. This is Python-reachable today.
   - **Ordered approx & per-object running CTR:** these are the genuinely un-exposed values. Validate them indirectly: assert that the harness's *final* prediction (sum of ordered approx over iterations) matches upstream `predict` ≤1e-5, AND that per-object intermediates are internally consistent (running numerator/denominator monotone, prefix == final when permutation is identity).
3. **Frozen committed fixtures (D-09):** the harness runs offline; only `.npy` outputs land in `crates/cb-oracle/fixtures/`. Neither generator runs in CI.

**Why this preserves D-02/D-03 intent:** the per-object stack (permutation, running num/denom, per-iteration approx) is still dumped and asserted ≤1e-5 — only the *source* of permutation/final-CTR ground truth shifts from "link the trainer" (impossible) to "the oracle-locked Rust RNG + the model's own `ctr_data`" (available). The per-object leakage signature is still the comparison point.

**Residual risk (flag for the planner):** the per-iteration ordered approximant has no fully-independent external anchor (upstream never dumps it). The mitigation above (final-prediction parity + internal-consistency + identity-permutation degeneration) is strong but not a direct per-value external oracle. If the user wants a *direct* external anchor, the only remaining option is the full catboost build with instrumentation (deferred per D-01) — surface this tradeoff in discuss-phase.

### Per-Object Oracle Schema (D-02) — concrete `.npy` layout

| `.npy` file | Shape | dtype | Comparison stage | Source |
|-------------|-------|-------|------------------|--------|
| `permutation_fold{k}.npy` | `[N]` | `int32` | `Stage::Permutation` (exact, integer) | Fisher-Yates over `TFastRng64`, per fold k=0..fold_count-1 |
| `ctr_good_count.npy` | `[N]` | `int32` | exact integer (numerator) | `goodCount` per doc (online_ctr.cpp:177/303) |
| `ctr_total_count.npy` | `[N]` | `int32` | exact integer (denominator) | `totalCount` per doc (online_ctr.cpp:173/304) |
| `ctr_value.npy` | `[N]` (or `[N, n_prior]`) | `float64` | `Stage::OnlineCtr` ≤1e-5 | `(good+prior)/(total+1)` per doc/prior |
| `ordered_approx_iter{t}.npy` | `[N]` | `float64` | `Stage::OrderedApprox` ≤1e-5 | per-iteration ordered approx (approx_calcer.cpp:590) |

**Locking order (D-03):** assert `Stage::Permutation` (exact) FIRST; only if it passes do `Stage::OnlineCtr`/`Stage::OrderedApprox` run. Add these three variants to `cb-oracle::compare.rs::Stage`. Numerator/denominator are integers — compare exactly, not at 1e-5 (they are `int` in upstream, `TCtrHistory::N[2]`).

## Common Pitfalls

### Pitfall 1: Online vs inference CTR denominator divergence
**What goes wrong:** Using `(count+prior)/(total+PriorDenom)` in the training online path, or `/(total+1)` in the inference path.
**Why it happens:** Two different formulas (online_ctr.h:129 vs :290) that coincide only when `PriorDenom==1`.
**How to avoid:** Online accumulation uses literal `+1`; model-side `Calc` uses `+PriorDenom`. Keep them as separate functions. Pin priors with denom=1 in fixtures so they coincide, but implement both correctly.
**Warning signs:** Final-prediction parity passes but per-object online CTR diverges (or vice versa) when a non-unit `PriorDenom` is introduced.

### Pitfall 2: Permutation needed even in Plain mode
**What goes wrong:** Skipping permutation generation for `boosting_type=Plain` and getting wrong CTRs.
**Why it happens:** `IsPermutationNeeded` returns true if `hasCtrs` regardless of boosting type (learn_context.cpp:42-43). `hasCtrs = maxCatUniqueCount > one_hot_max_size` (learn_context.cpp:71-73).
**How to avoid:** Generate permutation whenever any cat feature exceeds `one_hot_max_size`, OR boosting is Ordered. For the D-04 one-hot-only first slice (all cats ≤ `one_hot_max_size`), there is genuinely NO permutation — that is the isolation.
**Warning signs:** `fold_count`/permutation present in a one-hot-only fixture (should be absent), or absent in a Plain+CTR fixture (should be present).

### Pitfall 3: One-hot vs CTR threshold off-by-one
**What goes wrong:** Wrong feature routed to one-hot vs CTR at the boundary `count == one_hot_max_size`.
**Why it happens:** One-hot used when `1 < count <= one_hot_max_size` (greedy_tensor_search.cpp:182: skip if `>` or `<=1`); CTR used when `count > one_hot_max_size` (greedy_tensor_search.cpp:469: skip if `<=`). Boundary is **inclusive for one-hot**, exclusive for CTR. Also note: cardinality is `OnLearnOnly` unique count (learn-set only).
**How to avoid:** Build a fixture with a column of cardinality exactly `one_hot_max_size` (→ one-hot) and one of `one_hot_max_size+1` (→ CTR).
**Warning signs:** A `count==one_hot_max_size` feature generating a CTR table.

### Pitfall 4: Counter/FeatureFreq denominator semantics
**What goes wrong:** Wrong `CounterDenominator` for Counter (max bucket count) vs FeatureFreq (total sample count).
**Why it happens:** `Counter: denom = MaxElement(counts)` (online_ctr.cpp:935 / :728); `FeatureFreq: denom = totalSampleCount` (online_ctr.cpp:938). Also `CounterCalcMethod` default is `SkipTest` (cat_feature_options.cpp:234) — Full would include test docs.
**How to avoid:** Implement the two denominators distinctly; pin `counter_calc_method` explicitly.
**Warning signs:** Counter and FeatureFreq producing identical values (they share the `++ctrIntArray[elemId]` accumulation but differ only in denominator).

### Pitfall 5: BinarizedTargetMeanValue vs FloatTargetMeanValue accumulation
**What goes wrong:** Adding `targetClass/targetBorderCount` vs raw `target`.
**Why it happens:** `BinarizedTargetMeanValue: elem.Add(targetClass[z]/targetBorderCount)` (online_ctr.cpp:920 / :467); `FloatTargetMeanValue: elem.Add(targets[z])` raw (online_ctr.cpp:925). Both use `TCtrMeanHistory{Sum:float, Count:int}`.
**How to avoid:** Two distinct accumulators; `FloatTargetMeanValue` is **not** in the online (train-time) dispatch (online_ctr.cpp:732-800 handles Borders/Buckets/BinarizedTargetMeanValue/Counter only) — it appears only in `CalcFinalCtrsImpl` (online_ctr.cpp:923). Plan FloatTargetMeanValue via the final-CTR path, not the online path.
**Warning signs:** FloatTargetMeanValue appearing in per-object online dumps (it shouldn't — only as a final-model CTR).

### Pitfall 6: `boosting_type` and CTR-type auto-selection
**What goes wrong:** Relying on upstream defaults that auto-pick Ordered (GPU-only) or default CTR sets (Borders+Counter).
**Why it happens:** Default `simple_ctr`/`combinations_ctr` = `[Borders(priors 0/1,0.5/1,1/1), Counter(prior 0/1)]` (catboost_options.cpp:450-451 + cat_feature_options.cpp:118-138). To oracle-lock a *single* CTR type per D-07, you must override `simple_ctr` explicitly.
**How to avoid:** Pin `boosting_type`, `simple_ctr`, `combinations_ctr`, all priors, `one_hot_max_size`, `max_ctr_complexity`, `permutation_count`, `fold_len_multiplier`, `counter_calc_method`, `thread_count=1` explicitly per fixture. Reject auto-selection (Claude's-discretion item).

## Runtime State Inventory

> Not a rename/refactor/migration phase — greenfield algorithm implementation extending existing crates. Section included for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — phase produces new fixtures, mutates no existing datastore. | none |
| Live service config | None. | none |
| OS-registered state | None. | none |
| Secrets/env vars | None. | none |
| Build artifacts | `cb-oracle/generator/__pycache__`, `catboost_info/` (offline generator scratch) — not committed; no stale-artifact risk. New `ordered_oracle.cpp` compiles to a throwaway binary, not linked into the workspace. | none |

**Nothing found in every category — verified by inspection of the phase scope (pure-Rust + offline generator).**

## Code Examples

### CTR type dispatch (which math for which type)
```cpp
// Source: catboost-master/catboost/private/libs/algo/online_ctr.cpp:732-800 (online/train-time) — abbreviated
if (ctrType == ECtrType::Borders && targetClassesCount == SIMPLE_CLASSES_COUNT /*==2*/) {
    CalcOnlineCTRSimple(...);     // binclf Borders: N[0]/N[1], read-before-incr
} else if (ctrType == ECtrType::BinarizedTargetMeanValue) {
    CalcOnlineCTRMean(...);       // Add(targetClass / targetBorderCount)
} else if (ctrType == ECtrType::Buckets ||
          (ctrType == ECtrType::Borders && targetClassesCount > 2)) {
    CalcOnlineCTRClasses(...);    // multi-class borders/buckets; UpdateGoodCount differs by type
} else { /* ECtrType::Counter */
    CalcOnlineCTRCounter(...);    // denom = max bucket count; counts, no target
}
// FloatTargetMeanValue & FeatureFreq: NOT in online dispatch — only final-CTR (CalcFinalCtrsImpl).
```

### Final CTR table build (whole-set, baked into model)
```cpp
// Source: catboost-master/catboost/private/libs/algo/online_ctr.cpp:916-939 (CalcFinalCtrsImpl)
for (ui32 z = 0; z < totalSampleCount; ++z) {
    const ui64 elemId = hashArr[z];
    if (ctrType == ECtrType::BinarizedTargetMeanValue) ctrMean[elemId].Add((float)targetClass[z] / targetBorderCount);
    else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) ++ctrIntArray[elemId];
    else if (ctrType == ECtrType::FloatTargetMeanValue) ctrMean[elemId].Add(targets[z]);
    else ++ctrIntArray[targetClassesCount * elemId + targetClass[z]];  // Borders/Buckets multi-class
}
if (ctrType == ECtrType::Counter)     result->CounterDenominator = *MaxElement(...);  // max bucket
if (ctrType == ECtrType::FeatureFreq) result->CounterDenominator = totalSampleCount;  // N
```

### Model-side inference apply (per type)
```cpp
// Source: catboost-master/catboost/libs/model/static_ctr_provider.cpp:52-122 + online_ctr.h:289-292
// Borders (binclf): ctr->Calc(ctrHistory[1], ctrHistory[0] + ctrHistory[1])
// Mean types:       ctr->Calc(ctrMeanHistory.Sum, ctrMeanHistory.Count)
// Counter/FeatFreq: ctr->Calc(ctrTotal[bucket], CounterDenominator)
// Buckets:          ctr->Calc(ctrHistory[TargetBorderIdx], sum over classes)
// where Calc(cic, tot) = (cic + PriorNum) / (tot + PriorDenom); return (ctr + Shift) * Scale;
// missing bucket → emptyVal = ctr->Calc(0, denom) (Counter) or ctr->Calc(0,0) (others)
```

### Default priors per CTR type
```cpp
// Source: catboost-master/catboost/private/libs/options/cat_feature_options.cpp:118-138 (TPrior = {num, denom})
// Borders / Buckets / BinarizedTargetMeanValue: {0,1}, {0.5,1}, {1,1}   (THREE priors → 3 CTR columns)
// FeatureFreq / Counter:                        {0.0,1}                  (one prior)
// FloatTargetMeanValue:                         {0,1}                    (one prior)
// Defaults: one_hot_max_size=2, max_ctr_complexity=4 (cat_feature_options.cpp:231-232),
//           permutation_count=4, fold_len_multiplier=2.0, boosting_type=Plain (boosting_options.cpp:11-16)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase-2 cat-hash fixtures from model `ctr_data` hash_map | `cb-data::calc_cat_feature_hash` (standalone CityHash transcription) | Plan 02-04 Rule-1 fix | CTR consumers must use `calc_cat_feature_hash`; `ctr_data` hash_map is projection hashes |
| (this phase) link upstream TUs for ground truth | transcribe-then-self-oracle | Phase 5 research (this doc) | The locked D-01 link mechanism is dead; fallback proposed |

**Deprecated/outdated:**
- The notion that the cityhash precedent "links upstream TUs" — it transcribes them. Any plan that assumes `online_ctr.cpp` can be `#include`d/linked standalone is wrong.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The transcribe-then-self-oracle fallback is acceptable to the user as the D-01 replacement | Per-Object Oracle Strategy | Whole phase oracle strategy stalls; must re-discuss before planning |
| A2 | Per-iteration ordered approximant has no external upstream anchor (validated indirectly via final-prediction parity + internal consistency) | Per-Object Oracle Strategy | If a direct anchor is required, only the rejected full-build path remains |
| A3 | Disk has enough headroom (21G free now) for the offline transcribed-harness build (tiny, zero catboost deps) — but NOT for a full catboost build | ESCALATION | If even the tiny harness can't build, escalate; but transcription needs only g++ + the input `.npy`, ~no footprint |
| A4 | The committed `ctr_data_generated.rs` bindings match the 1.2.10 `ctr_data.fbs` exactly (they were generated in Phase-4) | Standard Stack | `ctr_data` serde mismatch; re-generate from the vendored `.fbs` |
| A5 | `model_json.rs` does NOT yet parse `ctr_data` (only borders) — confirmed by grep; the plan must add it | Recommended Structure | If it already parses ctr_data, the parsing task is redundant (low risk) |
| A6 | For default unit-denominator priors, online `+1` and inference `+PriorDenom` coincide; fixtures should pin denom=1 to keep them aligned | Pitfall 1 | Non-unit priors would expose the divergence; pin priors explicitly |

**This table is non-empty: A1 and A2 (oracle strategy) require user confirmation in a discuss-phase before task planning.**

## Open Questions

1. **Does upstream expose per-iteration ordered approx anywhere Python-reachable?**
   - What we know: `static_ctr_provider`/model exposes final `ctr_data`; `predict` exposes final approx; staged `predict` (`prediction_type`, `ntree_start/end`) exposes per-tree *cumulative* approx on the *plain* path.
   - What's unclear: staged predict reflects the *final model's* additive trees, not the *training-time ordered* per-object approx (which is fold/permutation-specific and discarded after training).
   - Recommendation: treat per-iteration ordered approx as un-exposed; validate via the indirect anchors in the fallback (A2). Raise in discuss-phase.

2. **`fold_count` vs `permutation_count` terminology.** Success criteria say `fold_count`; upstream option is `permutation_count` (default 4), and learning folds = `max(1, permutation_count-1)` for ordered (learn_context.cpp:48-49), plus one averaging fold.
   - Recommendation: pin `permutation_count` explicitly per fixture; document the `LearningFoldCount = max(1, permutation_count-1)` relationship so the per-fold permutation dumps line up (e.g. `permutation_count=2` → 1 learning fold + 1 averaging fold).

3. **`fold_permutation_block` for small N.** `DefaultFoldPermutationBlockSize = min(256, docCount/1000 + 1)` (defaults_helper.h) = **1** for N<1000.
   - Recommendation: keep fixtures small (N<1000) so block size is 1 (per-object shuffle, simplest to reproduce). Pin `fold_permutation_block` if any ambiguity.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `g++` / C++17 | offline transcribed oracle harness (`ordered_oracle.cpp`) | ✓ (cityhash harness built with it) | system | — |
| Python 3.12 + `catboost==1.2.10` | offline fixture generator (final-CTR anchor via model.json) | ✓ (`.venv` present, gen_fixtures.py runs) | 1.2.10 pinned | — |
| `numpy` | `.npy` fixture I/O | ✓ | — | — |
| Rust stable workspace | all impl + oracle tests | ✓ | latest | — |
| Disk headroom | builds | ⚠ 21G free | — | Per-crate test profile (cb-compute full link blocked by MLIR — verify per-crate per STATE.md) |
| Full catboost C++ build | (rejected D-01) | ✗ | — | transcribe-then-self-oracle |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** full catboost build (✗) → transcribe-then-self-oracle (the phase's central design choice). Disk pressure → per-crate verification (carried-forward Phase-3/4 practice).

## Validation Architecture

> `workflow.nyquist_validation: true` (config.json:24) — section required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cb-oracle::compare_stage` ≤1e-5 gate |
| Config file | none (Cargo workspace); fixtures under `crates/cb-oracle/fixtures/` |
| Quick run command | `cargo test -p cb-train` (and `-p cb-model`, `-p cb-oracle`) |
| Full suite command | `cargo test --workspace` (⚠ MLIR/disk — prefer per-crate per STATE.md Blockers) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ORD-01 | Permutation reproduces upstream Fisher-Yates exactly (per fold) | unit (exact int) | `cargo test -p cb-train permutation` | ❌ Wave 0 |
| ORD-01 | TFold body/tail prefix boundaries match `SelectMinBatchSize`/`SelectTailSize` | unit | `cargo test -p cb-train fold_prefix` | ❌ Wave 0 |
| ORD-02 | Per-object ordered approx per iteration ≤1e-5 (`Stage::OrderedApprox`) | oracle | `cargo test -p cb-train ordered_boost_oracle` | ❌ Wave 0 |
| ORD-02 | Ordered final prediction ≤1e-5 vs upstream | oracle | `cargo test -p cb-model ordered_predict_oracle` | ❌ Wave 0 |
| ORD-03 | Each of 6 CTR types: per-object online num/denom (exact) + value ≤1e-5 | oracle (×6) | `cargo test -p cb-train ctr_<type>_oracle` | ❌ Wave 0 |
| ORD-03 | Plain-mode CTR (whole-set) ≤1e-5 — locked BEFORE ordered (D-06) | oracle | `cargo test -p cb-train plain_ctr_oracle` | ❌ Wave 0 |
| ORD-04 | One-hot path selection at `count==one_hot_max_size` (incl) and `+1` (CTR) | unit | `cargo test -p cb-train one_hot_threshold` | ❌ Wave 0 |
| ORD-04 | One-hot-only model trains+predicts ≤1e-5 (no permutation present) | oracle | `cargo test -p cb-model one_hot_predict_oracle` | ❌ Wave 0 |
| ORD-05 | Tensor CTR (`max_ctr_complexity`) projection enumeration + ≤1e-5 | oracle | `cargo test -p cb-model tensor_ctr_oracle` | ❌ Wave 0 |
| (model) | `ctr_data` `.cbm`/`model.json` round-trip + upstream load ≤1e-5 | oracle | `cargo test -p cb-model ctr_data_roundtrip` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** the single owning-stage test (e.g. `cargo test -p cb-train permutation`) — < 30s.
- **Per wave merge:** `cargo test -p cb-train -p cb-model -p cb-oracle`.
- **Phase gate:** all ORD-01..05 oracles green before `/gsd-verify-work`; per-crate (not `--workspace`) due to MLIR/disk.

### Wave 0 Gaps
- [ ] `crates/cb-oracle/src/compare.rs` — add `Stage::Permutation`, `Stage::OnlineCtr`, `Stage::OrderedApprox`.
- [ ] `crates/cb-oracle/src/model_json.rs` — add `ctr_data` parsing (currently borders-only).
- [ ] `crates/cb-oracle/generator/ordered_oracle.cpp` — transcribed standalone harness (per fallback).
- [ ] `crates/cb-oracle/fixtures/` — purpose-built categorical fixtures (D-08): low-card (one-hot), high-card (CTR), tiny N; per-fold permutation `.npy`; per-object num/denom/approx `.npy`; per-CTR-type config (D-07).
- [ ] D-03 ordering harness: assert `Stage::Permutation` exact before value stages run.
- [ ] Framework install: none — existing `#[test]` + `compare_stage` suffice.

## Security Domain

> `security_enforcement: true` (config.json:46) — section required.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a (library, no auth surface) |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes | `.cbm`/`model.json` `ctr_data` parsing MUST validate bounds before slicing blobs; malformed `IndexHashRaw`/`CTRBlob`/`CounterDenominator`/`TargetClassesCount` → typed `ModelError`, never panic (carries forward Phase-4 04-03 panic-free serde discipline). No `unwrap`/raw-index on untrusted model bytes. |
| V6 Cryptography | no | CityHash is a non-cryptographic hash (correctness, not security) — no change |

### Known Threat Patterns for Rust model-load + training
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed `ctr_data` FlatBuffer (oversized blob, bad bucket index) | Tampering / DoS | `VERIFYING` flatbuffers accessor + bounds-check bucket index `< leafCount` before indexing (mirror Phase-4 `root_as_tmodel_core` discipline); return `ModelError` |
| Bucket index out of range at inference (`ptrBuckets[doc]`) | Tampering | Upstream guards with `NotFoundIndex` → `emptyVal` (static_ctr_provider.cpp:115-119); Rust must replicate the not-found→empty path, no OOB index |
| Integer overflow in `CounterDenominator`/count accumulation | DoS | counts bounded; `u32::MAX` uniq-count guard already established (Plan 02-04); reuse typed-error pattern |
| `unwrap`/`panic` in new CTR/permutation code | Availability | INFRA-02 lint gate (`deny unwrap/expect/panic/indexing_slicing` in libs); `thiserror`, `anyhow` CI-grep banned |

## Project Constraints (from CLAUDE.md + AGENTS.md)

- **Source/test separation MANDATORY** — no inline `#[cfg(test)]`; tests in `*_test.rs`/`tests/`. (CLAUDE.md, AGENTS.md §2)
- **`thiserror` in libraries; `anyhow` structurally banned** (CI grep). New errors → typed enum variants.
- **`unwrap()` strictly prohibited in production**; INFRA-02 denies `unwrap`/`expect`/`panic`/`indexing_slicing` in libs.
- **All parity-critical float sums via `cb-core::sum_f64`/`sum_f32_in_f64`** (D-08 CI grep). Integer class counts are exact-int, exempt.
- **CubeCL kernels ONLY in `cb-backend`, only after reading `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`; generics-float; load `cubecl_error_guideline.md` on any build error before fixing.** (Research recommends NO new kernel this phase — pure host.)
- **Latest crate versions** — but this phase adds no crates.
- **`cb-compute` stays cubecl-free; abstract `R: Runtime`/`F: Float` seam** (Phase-3 D-03).
- **Categorical hashing via `cb-data::calc_cat_feature_hash`** — never a model's `ctr_data` hash_map.

## Sources

### Primary (HIGH confidence — vendored 1.2.10 source, read line-by-line)
- `catboost/private/libs/algo/online_ctr.cpp` (:102-940) + `online_ctr.h` (:128-185) — six CTR types, online read-before-increment, `CalcCTR`, dispatch, final-CTR build.
- `catboost/private/libs/algo/approx_calcer.cpp` (:566-600, :706-839, :900-1007) — ordered (`CalcApproxDeltaSimple`/`UpdateApproxDeltasHistorically`) vs plain (`CalcLeafValuesSimple`).
- `catboost/private/libs/algo/fold.{h,cpp}` (fold.h:31-234, fold.cpp:35-211) — `TFold`, body/tail prefixes, `BuildDynamicFold`/`BuildPlainFold`, `InitPermutationData`.
- `catboost/private/libs/algo/learn_context.cpp` (:38-152, :494-587) — `IsPermutationNeeded`, `CountLearningFolds`, fold creation loop, `Rand(randomSeed)`.
- `catboost/private/libs/algo/greedy_tensor_search.cpp` (:171-197, :457-551) — one-hot path, simple/tree CTR enumeration, `max_ctr_complexity` gate.
- `catboost/private/libs/algo/projection.h` (:61-145) — `TProjection`, `GetFullProjectionLength`.
- `catboost/private/libs/ctr_description/ctr_type.{h,cpp}` — `ECtrType`, `NeedTarget`/`NeedTargetClassifier`/`IsPermutationDependentCtrType`.
- `catboost/private/libs/options/cat_feature_options.cpp` (:118-138, :226-237) — default priors, `one_hot_max_size=2`, `max_ctr_complexity=4`.
- `catboost/private/libs/options/catboost_options.cpp` (:429-453, :778-816) — CTR defaults (Borders+Counter), boosting-type auto (GPU-only Ordered).
- `catboost/private/libs/options/boosting_options.cpp` (:11-16) — `boosting_type=Plain` default, `permutation_count=4`, `fold_len_multiplier=2.0`.
- `util/random/shuffle.h` (:24-32) + `util/random/common_ops.h` (:48-91) — Fisher-Yates `Shuffle`, `GenUniform`/`Uniform`.
- `catboost/libs/model/online_ctr.h` (:260-315, :369-401) — `TModelCtr::Calc`, `TCtrHistory`, `TCtrMeanHistory`.
- `catboost/libs/model/static_ctr_provider.cpp` (:14-126) — model-side per-type CTR apply at inference.
- `catboost/libs/model/ctr_provider.h` (:65-122) — projection `CalcHash`/`CalcHashes`.
- `catboost/libs/model/flatbuffers/ctr_data.fbs` — `ctr_data` schema (Rust bindings committed).

### Secondary (HIGH — existing project assets, grepped/read this session)
- `crates/cb-core/src/rng.rs` (:208-230) — `try_uniform`/`uniform` (bit-exact `GenUniform`).
- `crates/cb-oracle/generator/cityhash_oracle.cpp` — the *actual* precedent (transcription, zero catboost includes).
- `crates/cb-oracle/src/compare.rs` — `Stage` enum + `compare_stage` 1e-5 gate.
- `crates/cb-oracle/src/model_json.rs` — borders-only parser (no `ctr_data` yet).
- `crates/cb-model/src/generated/ctr_data_generated.rs` — committed FlatBuffers bindings.
- `.planning/STATE.md` — disk pressure (MLIR), cat-hash provenance, host-ordered-reduce invariant.

### Tertiary (LOW)
- none — every claim is cited to vendored source or a session-grepped project file.

## Metadata

**Confidence breakdown:**
- Algorithm transcription (CTR types, ordered boosting, permutation, one-hot, tensor): **HIGH** — every formula cited to 1.2.10 file:line, cross-checked across online/final/inference paths.
- D-01 TU-isolation feasibility verdict: **HIGH (negative)** — include analysis is definitive; the precedent transcribes, not links.
- Fallback oracle strategy: **MEDIUM** — design is sound and preserves D-02/D-03 intent, but per-iteration ordered approx lacks a direct external anchor (A2); needs user sign-off (A1).
- Defaults/config pins: **HIGH** — read directly from options source.

**Research date:** 2026-06-14
**Valid until:** 2026-07-14 (vendored source is frozen at 1.2.10 — stable; 30 days).
