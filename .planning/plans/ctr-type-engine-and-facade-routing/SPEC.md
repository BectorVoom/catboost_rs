---
title: CTR-type engine honesty + cat_features/CTR facade routing
status: draft
format: markdown
spec_version: 1
updated_at: 2026-08-01T00:00:00Z
source_requirements:
  - "User decision (2026-08-01): 'Fix the engine first, then route everything' — make the inert CTR params genuinely honored, then wire cat_features through the facade + Python bindings"
  - "User decision (2026-08-01): lift the .cbm v1 mean-CTR restriction so BinarizedTargetMeanValue models round-trip"
  - "User decision (2026-08-01): honor CTR type + FULL prior list, keep the scalar BoostParams field; record the multi-description default as a known parity gap"
  - "User decision (2026-08-01): reuse the committed one_hot_train/ fixture for the public-API oracle; run NO corpus-wide fixture generation"
  - ".planning/plans/ctr-type-engine-and-facade-routing/research.md"
  - ".planning/plans/catboost-builder-cat-features-routing/{SPEC.md,PLAN.md,PLAN-CHECK.md} (existing, ISSUES_FOUND)"
---

# CTR-type engine honesty + `cat_features`/CTR facade routing

> Draft specification. NOT approved, accepted, final, or implemented.
> Evidence tags: `[VERIFIED: WEB …]`, `[VERIFIED: LOCAL <path:line>]`,
> `[VERIFIED: CODEGRAPH <symbol>]`, `[VERIFIED: EXPERIMENT …]`, `[INFERRED: …]`,
> `[UNVERIFIED: …]`.

---

## 1. Context

`catboost-rs` carries a complete, oracle-verified CTR training engine and a complete,
CTR-aware inference path. Four `BoostParams` fields that describe CTR behavior are
nevertheless **never read** by `train_inner`, and the categorical surface is unreachable
from the public facade. This specification covers both, in that order.

### 1.1 The four inert parameters

`params.simple_ctr`, `params.simple_ctr_priors`, `params.combinations_ctr` (the CTR
**type**; `combinations_ctr_priors` IS read) and `params.counter_calc_method` have zero
reads in `train_inner`
`[VERIFIED: LOCAL grep 'params\.<field>' crates/cb-train/src/boosting.rs]`.
Instead:

- `ctr_splits_for_tree` hard-codes `ctr_type: ECtrType::Borders.as_i8()`
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:1940]`;
- the prior numerator is taken from `combinations_ctr_priors.first()` for **every**
  candidate, simple or combination
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:3155]`;
- the bake path hard-codes Borders
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/bake.rs:192,232; crates/cb-train/src/ctr/ctr_feature.rs:232]`;
- `counter_calc_skip_test` is documented as "reserved"
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/final_ctr.rs:70-73]`.

### 1.2 What is already correct (do NOT rebuild)

- `build_final_ctr` implements **all six** `ECtrType` variants
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/final_ctr.rs:75-130]`.
- The online accumulator populates `class_histories` / `binarized_mean` / `float_mean` /
  `total_counts` unconditionally
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:192-229]`.
- Inference-side `CtrValueTable::calc_for_hash` dispatches all six
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:226-258]`.
- `calc_ctr_online`, `calc_ctr_online_bin` and `calc_normalization` are **type-agnostic
  and correct**: upstream uses ONE `CalcCTR` and ONE `CalcNormalization` for every online
  type `[VERIFIED: WEB online_ctr.h:128-131]`. They need per-type **inputs**, not per-type
  variants.

### 1.3 Only three online arms are missing

CatBoost's CPU backend **rejects** `FloatTargetMeanValue` and `FeatureFreq` at options
validation `[VERIFIED: WEB restrictions.h:18-48]`, reproduced empirically — both raise
"not implemented on CPU yet" `[VERIFIED: EXPERIMENT probe1.py]`. The parity target is the
CPU engine, so the missing online arms are exactly **Buckets**,
**BinarizedTargetMeanValue** and **Counter**.

### 1.4 The facade gap (Part 2)

`CatBoostBuilder::fit` reads only `pool.float_features()` and unconditionally calls the
float-only `cb_train::train`; `pool.cat_features()` is never read
`[VERIFIED: LOCAL crates/catboost-rs/src/builder.rs:383]`, and `boost_params()` pins every
CTR field to its default with an "inert here" comment
`[VERIFIED: LOCAL crates/catboost-rs/src/builder.rs:292-303]` — the full pinned block,
including `counter_calc_method` (`:296`), `max_ctr_complexity` (`:301`) and
`combinations_ctr` / `combinations_ctr_priors` (`:302-303`), all of which the
previously-cited `:284-295` range omitted. `Model::predict_with` calls
the float-only `predict_raw`. Because `predict_raw(model, fv) == predict_raw_cat(model, fv, &[])`
and an absent categorical value resolves to the empty string
`[VERIFIED: LOCAL crates/cb-model/src/apply.rs:370-372,410-414]`, a CTR model scored through
today's `predict()` would return **numerically wrong predictions with no error**. Wiring
`fit()` without `predict()` would ship exactly that footgun.

---

## 2. Scope and non-goals

### In scope — Part 1 (engine; lands first)

1. `ECtrType` capability helpers and a CPU-legality guard.
2. A generic classes-prefix online producer, with Borders-binclf proven to be its
   `(classes = 2, b = 0)` special case.
3. Online arms for Buckets, BinarizedTargetMeanValue and Counter.
4. `ctr_type` and prior selection routed by `CtrCandidate.is_simple`.
5. Candidate expansion over the **full prior list** and over `target_border_idx`.
6. Per-type final tables in the bake path.
7. **`.cbm` mean-CTR codec** (encode + decode) so `BinarizedTargetMeanValue` models
   round-trip — locked user decision.
8. `counter_calc_method` threading (Counter-only).

### In scope — Part 2 (facade; after Part 1)

Revision **and execution** of `.planning/plans/catboost-builder-cat-features-routing/`
(SPEC-CATF-01..17, 18 tasks, verdict `ISSUES_FOUND`), with all its blocking findings
resolved. See §8.

### Non-goals

- **Multi-description CTR configs** (`simple_ctr = [Borders(...), Counter(...)]`
  simultaneously). The `BoostParams` field stays scalar `ECtrType`. Recorded as an explicit
  parity gap — SPEC-CTRT-19. Locked user decision.
- `FloatTargetMeanValue` / `FeatureFreq` online arms — CPU-illegal upstream (§1.3). They
  must be **rejected**, not implemented.
- Non-unit CTR prior denominators (CPU forbids them
  `[VERIFIED: WEB ctr_helper.cpp:50]` — this vindicates the repo's `prior_denom: 1.0` pin
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:1942]`).
- Non-Uniform CTR binarization (CPU forbids it
  `[VERIFIED: WEB catboost_options.cpp:517-520]`).
- Varying `ctr_target_border_count` (default 1 ⇒ `targetClassesCount == 2` for binclf
  `[VERIFIED: EXPERIMENT probe2.py]`); changing it multiplies the candidate set for every
  type.
- Multiclass Borders apply correctness (`target_border_idx` ignored at inference) —
  pre-existing, recorded as a known limitation with the
  `static_ctr_provider.cpp:90-110` anchor.
- Any change to the uncommitted one-hot device wave. It must stay green.

---

## 3. Dependencies

| Dependency | Interface | Evidence |
|---|---|---|
| `ECtrType` | `Borders=0, Buckets=1, BinarizedTargetMeanValue=2, FloatTargetMeanValue=3, Counter=4, FeatureFreq=5`; `as_i8`, `from_i8` | `[VERIFIED: LOCAL crates/cb-train/src/ctr/mod.rs:67-95]` |
| `CounterCalcMethod` | `Full \| SkipTest`, default `SkipTest` | `[VERIFIED: LOCAL crates/cb-train/src/ctr/mod.rs:124-134]` |
| `OnlineCtrAccumulator` | `class_histories: Vec<TCtrHistory>`, `binarized_mean`/`float_mean: Vec<TCtrMeanHistory>`, `total_counts: Vec<i64>`, `classes`, `bucket_count` | `[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:126-135]` |
| `calc_ctr_online` | `(count_in_class: f64, total_count: i64, prior: f64) -> f64` = `(cic + prior)/(tot + 1)` | `[VERIFIED: LOCAL crates/cb-train/src/ctr/calc_ctr.rs:76-79]` |
| `calc_ctr_online_bin` | `(cic, tot, prior, border_count) -> f64` | `[VERIFIED: LOCAL crates/cb-train/src/ctr/calc_ctr.rs:89-101]` |
| `CtrCandidate.is_simple` | `bool`, from `projection.is_simple()` | `[VERIFIED: LOCAL crates/cb-train/src/candidates.rs:156,194]` |
| `CtrValueTable` | `.cbm`/JSON baked table; mean tables rejected on encode AND decode (v1) | `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:584-588,756-758]` |
| upstream v1.2.10 | `online_ctr.{h,cpp}`, `ctr_type.h`, `catboost_options.cpp`, `restrictions.h` | `[VERIFIED: WEB github.com/catboost/catboost tag v1.2.10]` |

**The vendored `catboost-master/` is NOT a valid parity reference.** It contains three
files and its `greedy_tensor_search.cpp` is 1997 lines vs v1.2.10's 1955 — a *different
revision* `[VERIFIED: LOCAL wc -l]`. CLAUDE.md's description of a full vendored tree is
stale. All upstream citations here come from the live `v1.2.10` tag.

---

## 4. Typed contracts

```rust
// SPEC-CTRT-01/02 — cb-train/src/ctr/mod.rs
impl ECtrType {
    /// targetBorderCount for binary classification (targetClassesCount == 2).
    /// Borders/BTMV/Counter → 1; Buckets → 2.
    pub fn target_border_count(self, target_classes_count: usize) -> usize;
    /// Whether CatBoost's CPU backend accepts this type at all.
    /// FloatTargetMeanValue and FeatureFreq → false.
    pub fn is_cpu_supported(self) -> bool;
    /// Whether the online statistic is a permutation PREFIX (Borders/Buckets/BTMV)
    /// or a whole-set constant (Counter).
    pub fn is_online_prefix(self) -> bool;
}

// SPEC-CTRT-04 — the ONE generic classes-prefix producer.
// Returns (numerator, denominator) for `calc_ctr_online`; NOT a new value function.
pub(crate) fn online_class_prefix(
    counts: &[i64],            // this bucket's prefix per-class counts N[0..classes]
    target_border_idx: usize,  // `b`
    ctr_type: ECtrType,
) -> (f64, i64);

// SPEC-CTRT-17 — cb-train/src/ctr/final_ctr.rs
pub fn build_final_ctr(
    acc: &OnlineCtrAccumulator,
    ctr_type: ECtrType,
    counter_calc_skip_test: bool,   // NEW: threaded, no longer "reserved"
) -> FinalCtrTable;
```

Typed errors (all `cb_core::CbError`, never panic, never silent):

| Condition | Error |
|---|---|
| `simple_ctr`/`combinations_ctr` is CPU-illegal | `CbError::Unsupported` carrying upstream's own wording |
| baked table length disagreement | `CbError::LengthMismatch` |
| `.cbm` mean blob malformed | `ModelError::Deserialize` |

---

## 5. Failure-isolated behavioral specifications — Part 1

Each specification below has ONE primary reason to fail.

### SPEC-CTRT-01 — `ECtrType::target_border_count`
**Given** a CTR type and `target_classes_count = 2` (binclf),
**when** `target_border_count` is called,
**then** it returns `2` for `Buckets` and `1` for `Borders` / `BinarizedTargetMeanValue` /
`Counter`.
*Rationale:* Buckets emits one candidate column per `target_border_idx`, and binclf has two
`[VERIFIED: WEB online_ctr.cpp:300-307; confirmed in a real model.json — both idx 0 and 1 appear]`.
*Acceptance:* unit test over all four CPU-legal types.

### SPEC-CTRT-02 — `ECtrType::is_cpu_supported`
**Given** `FloatTargetMeanValue` or `FeatureFreq`, **when** queried, **then** `false`;
all four others `true` `[VERIFIED: WEB restrictions.h:18-48]`.

### SPEC-CTRT-03 — CPU-illegal CTR types are typed-rejected before training
**Given** `BoostParams` whose `simple_ctr` or `combinations_ctr` is CPU-illegal,
**when** `train_cat` is called,
**then** it returns `CbError::Unsupported` naming the type and mirroring upstream's
"not implemented on CPU yet" wording, **before** any accumulation.
*Non-goal:* implementing those arms.
*Acceptance:* one test per illegal type; a legal type is unaffected.

### SPEC-CTRT-04 — the generic classes-prefix producer
**Given** a bucket's prefix per-class counts, a `target_border_idx` `b`, and a class-count
CTR type, **when** `online_class_prefix` is called, **then** it returns
`(numerator, denominator)` where
`Buckets → (N[b], Total)` and `Borders → (Total − Σ_{c ≤ b} N[c], Total)`
`[VERIFIED: WEB online_ctr.cpp:115-121,168-185]`.
*Acceptance:* table-driven unit test over hand-computed vectors.

### SPEC-CTRT-05 — Borders-binclf is the generic producer's special case (**regression firewall**)
**Given** `classes = 2`, `target_border_idx = 0`, `ctr_type = Borders`,
**when** `online_class_prefix` runs, **then** its output is **bit-for-bit identical** to
today's `good = N[1]`, `total = N[0] + N[1]`
`[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:309]`.
*Rationale:* this single test protects every existing CTR oracle from the refactor.
*Acceptance:* bit-equality assertion (`==` on raw values), not a tolerance.

### SPEC-CTRT-06 — Buckets online arm
**Given** a Buckets CTR feature, **when** the training-time column is materialized,
**then** each document's value uses the prefix `N[b]` numerator and prefix `Total`
denominator through the **unchanged** `calc_ctr_online` / `calc_ctr_online_bin`, preserving
read-before-increment.
*Acceptance:* `buckets_simple` fixture, ≤1e-5.

### SPEC-CTRT-07 — BinarizedTargetMeanValue online arm
**Given** a BTMV CTR feature, **when** materialized, **then** the read-before-increment
prefix `(Sum, Count)` is used, with the added value `targetClass / (targetClassesCount − 1)`
`[VERIFIED: WEB online_ctr.cpp CalcOnlineCTRMean]`.
**`Sum` MUST be accumulated in `f32`**, matching upstream's `TCtrMeanHistory::Sum`
— an f64 accumulation diverges only at boundary bins, which is precisely the
hard-to-localize failure class `LEARNING_RATE_F32.md` documents.
*Acceptance:* `btmv_simple` fixture ≤1e-5, plus a differential test isolating the f32 factor.

### SPEC-CTRT-08 — Counter online arm is NOT a prefix
**Given** a Counter CTR feature, **when** materialized, **then** every document's value uses
the **whole-set** bucket count over a **constant** `MAX`-bucket denominator — permutation
independent, no read-before-increment `[VERIFIED: WEB online_ctr.cpp CalcOnlineCTRCounter]`.
*Acceptance:* `counter_simple` fixture ≤1e-5, **plus** an assertion that two different
permutations yield the identical column (the property that distinguishes it from a prefix).

### SPEC-CTRT-09 — `ctr_splits_for_tree` emits the candidate's CTR type
**Given** a chosen CTR split, **when** the split spec is emitted, **then** `ctr_type` is
`simple_ctr` when `candidate.is_simple` and `combinations_ctr` otherwise — never the
hard-coded `Borders`.
*Precondition:* `ctr_splits_for_tree` currently has **no covering test**
`[VERIFIED: CODEGRAPH]`; one must exist before it is modified.
*Acceptance:* `mixed_simple_vs_combo` fixture proves the routing end to end.

### SPEC-CTRT-10 — prior selection follows `is_simple`
**Given** a CTR candidate, **when** its prior is resolved, **then** simple candidates read
`simple_ctr_priors` and combination candidates read `combinations_ctr_priors`.
*Acceptance:* a fixture with deliberately different priors per route.

### SPEC-CTRT-11 — candidate expansion over the FULL prior list
**Given** `simple_ctr_priors = [0.0, 0.5, 1.0]`, **when** CTR candidates are enumerated,
**then** ONE candidate column is emitted per prior, in upstream's order
`[VERIFIED: WEB greedy_tensor_search.cpp:414-427]` — not `.first()` only.
*Risk:* this changes the scored candidate set and therefore tie-breaks. It is the
**highest-risk sub-scope** and MUST land behind its own fixture, after SPEC-CTRT-04/05.
*Acceptance:* `borders_multiprior` fixture ≤1e-5; every single-prior CTR oracle unchanged.

### SPEC-CTRT-12 — candidate expansion over `target_border_idx`
**Given** a Buckets CTR at binclf, **when** candidates are enumerated, **then** both
`target_border_idx = 0` and `= 1` are emitted.
*Acceptance:* the `buckets_simple` generator asserts both indices appear in
`model.json → features_info.ctrs`.

### SPEC-CTRT-13 — the bake path builds per-type final tables
**Given** a trained model with a non-Borders CTR, **when** the final tables are baked,
**then** `build_final_ctr` is called with **that** type, and the emitted `ctr_type`
discriminant matches.
*Acceptance:* per-type bake test; Borders bytes unchanged.

### SPEC-CTRT-14 — `.cbm` ENCODES mean-type CTR tables
**Given** a model carrying a `BinarizedTargetMeanValue` table, **when** `save_cbm` runs,
**then** the mean table is encoded as `f32 Sum` + `i32 Count` pairs into the raw `CTRBlob`
byte array — replacing today's "mean tables rejected (v1)" path
`[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:756-758]`.
*Locked user decision.*

### SPEC-CTRT-15 — `.cbm` DECODES mean-type CTR tables, round-trip exact
**Given** a `.cbm` carrying a mean CTR table, **when** it is loaded, **then** the table
reconstructs exactly, replacing the `Deserialize("mean/target-mean CTR unsupported (v1, MAJOR-2)")`
rejection `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:584-588]`.
*Acceptance:* save→load→save byte-identity, **and** an upstream-produced `.cbm` with a BTMV
table loads and predicts within 1e-5.

### SPEC-CTRT-16 — non-mean `.cbm` bytes are unchanged (**regression gate**)
**Given** any model with no mean CTR table, **when** saved, **then** the bytes are identical
to the pre-change output.
*Acceptance:* byte comparison against a frozen baseline captured **before** SPEC-CTRT-14.

### SPEC-CTRT-17 — `counter_calc_method` is threaded, and its testability is stated honestly
**Given** `counter_calc_method`, **when** Counter CTRs are computed, **then** `SkipTest`
counts learn documents only and `Full` counts learn + every eval set
`[VERIFIED: WEB online_ctr.cpp:716-729]`.
**It is unobservable without an eval set** — measured `maxdiff = 0.000e+00` learn-only vs
`4.010e-01` with an eval set `[VERIFIED: EXPERIMENT]`. Therefore its acceptance test MUST
use an eval set; a learn-only test would pass trivially and prove nothing.
*Acceptance:* `counter_full_eval` fixture, or an explicit recorded deferral — never a
learn-only "pass".

### SPEC-CTRT-18 — every existing CTR oracle stays green at the default config
**Given** the unchanged default (`Borders`, priors `[0.5]`), **when** the full CTR suite
runs, **then** all **ELEVEN** existing CTR oracle targets pass unchanged — nine in
`cb-train` (`plain_ctr_oracle_test`, `ordered_ctr_oracle_test`, `tensor_ctr_oracle_test`,
`tensor_ctr_e2e_oracle_test`, `s_order_ctr_bins_oracle_test`, `ctr_split_scoring_test`,
`ctr_feature_materialize_test`, `multi_permutation_e2e_oracle_test`,
`multi_permutation_fold_oracle_test`) and two in `cb-model`
(`ctr_data_roundtrip_test`, `fstr_ctr_oracle_test`). This is exactly the target list
`PLAN.md` §3.2 runs; the two documents name the SAME eleven targets.
*Rationale:* all pin `Borders:Prior=0.5`, so a correctly-defaulted change is a no-op for
them — that is the regression gate.

### SPEC-CTRT-19 — the multi-description default gap is DOCUMENTED, not silently shipped
**Given** upstream's CPU default `[Borders(0/1, 0.5/1, 1/1), Counter(0/1)]`
`[VERIFIED: WEB catboost_options.cpp:439-453]` versus this repo's scalar single-type model,
**when** Part 2 exposes the default to users, **then** the divergence is recorded in the
parameter documentation and in `params.rs`, with the upstream anchor.
*Non-goal:* closing it (locked user decision).

### 5.1 AMENDMENT (2026-08-03, during E15 execution) — iteration-scale structure-parity semantics

E14's 20-iteration fixture exposed a divergence that is **independent of the
prior expansion**: the SAME corpus diverged from upstream at ~1.1e-1 with a
SINGLE prior once the iteration count exceeded ~8, while every committed
5-iteration CTR oracle stayed green. Localization against an **instrumented
v1.2.10 CLI build** (fold-approx/leaf-delta dumps at `%.17g`; the repo's
`logging_level="Debug"` winner-score technique; `/home/user/cb_instrumented_build`)
identified THREE missing upstream semantics, now specified as SPEC-CTRT-20..22.
All three are invisible at the 5-iteration scale of the pre-existing oracles —
each becomes structure-flipping at iteration scale.

### SPEC-CTRT-20 — per-learning-fold approx; the structure search reads the TAKEN fold
**Given** the Plain cat-CTR path, **when** a tree's structure is searched, **then** the
derivatives come from the TAKEN learning fold's OWN approx — advanced each iteration over
that fold's OWN CTR-bin leaf assignment by re-estimated fold deltas
(`UpdateLearningFold`/`CalcApproxForLeafStruct`, `train.cpp:585`,
`approx_calcer.cpp:706-800`) — never from the averaging fold's approx. The averaging
fold's approx (leaf-VALUE derivatives) and the linear output approx (`AvrgApprox`;
metrics/OD/staged) remain distinct streams. `UseAveragingFoldAsFoldZero` is FALSE here
because CTRs force `IsAveragingFoldPermuted` `[VERIFIED: WEB learn_context.cpp v1.2.10]`.
*Detection:* iteration-0 fold leaf `(SumDer, SumWeights, delta)` match the instrumented
upstream bit-for-bit; a single-approx engine's search derivatives drift from iteration 1.
*Acceptance:* the 20-iteration `ctr_borders_multiprior` fixture ≤1e-5; all 11 CTR oracles
unchanged (fold and averaging partitions coincide only when no CTR is present, so every
non-CTR path is structurally byte-identical).

### SPEC-CTRT-21 — EXP-domain approx storage with upstream's APPROXIMATE transcendentals
**Given** an `IsStoreExpApprox` loss (Logloss/CrossEntropy on this path,
`approx_updater_helpers.h:60-72`), **when** any training-fold approx (learning folds AND
the averaging fold) is stored or advanced, **then** the upstream approximate pipeline is
reproduced BIT-FOR-BIT: per-leaf delta exp-ification via `fmath::expd_v`
(`ExpApproxIf` → `FastExpInplaceAvx2`), per-document learning-rate application via
`fast_exp(FastLogf(δ)·lr)` (`ApplyLearningRate<true>`), and derivatives via
`p = 1 − 1/(1+e)` (`CalcCrossEntropyDerRangeImpl`, `error_functions.cpp:304-334` —
that rounding order, not `e/(1+e)`). The approximation error (~1e-6 absolute in log
space per application) is LOAD-BEARING: it feeds the next iteration's derivatives and
moves greedy scores across tie-break boundaries by ~10-20 iterations.
*Scope:* the cat-CTR path. The float-only / one-hot Logloss paths keep exact `exp`
derivative streams — their committed oracles prove the divergence stays under 1e-5 at
their iteration scale; widening is a recorded follow-up, not a silent change.
*Acceptance:* `crates/cb-train/src/fast_approx.rs` ports `fast_exp` (65536-entry table,
`library/cpp/fast_exp`), `fmath::expd` (`contrib/libs/fmath`), and `FastLogf`
(`library/cpp/fast_log`), pinned bit-exact by committed reference vectors generated from
the REAL upstream objects (`tests/fixtures/fast_approx_ref{.txt,_generator.cpp.txt}`);
the composed pipeline reproduces the instrumented fold approx (`0.99903645…`, NOT the
exact-exp `0.99903892…`).

### SPEC-CTRT-22 — `UsedCtrSplits` is MODEL-LIFETIME, and the weight key is `(ctr_type, projection)`
**Given** a CTR split some ALREADY-GROWN tree chose, **when** any later tree's search
scores a candidate with the same `(ctr_type, projection)`, **then** the `model_size_reg`
cat-feature weight is `1.0` — the penalty
`(1 + count/maxCount)^(−model_size_reg)` applies only while the pair has never been
used by the model (`GetCatFeatureWeight` + `TLearnProgress::UsedCtrSplits` +
`ProcessCtrSplit`, `greedy_tensor_search.cpp:926-950, :1126`; `learn_context.h:108`).
A per-tree-only lift (the pre-fix behavior) keeps an already-baked projection's score
~`weight×` too low from tree 1 on.
*Detection:* upstream's iteration-1 level-0 winner score jumps 1.192119137 → 1.647348574
on identical derivatives — the ratio is exactly the lifted penalty.
*Acceptance:* the 20-iteration fixtures above; the within-tree lift (already present)
remains, now keyed on `(ctr_type, projection)`.

---

## 6. Failure-isolated behavioral specifications — Part 2 (facade)

Part 2 **amends** the existing `SPEC-CATF-01..17` at
`.planning/plans/catboost-builder-cat-features-routing/SPEC.md` rather than replacing it.
Only the deltas are specified here.

### SPEC-CATF-Δ1 — supersedes the CRITICAL-4 precondition
`CRITICAL-4` ("one-hot columns silently dropped") is **OBSOLETE**: `ModelSplit::OneHot`,
`ObliviousTree.one_hot_splits` and `level_kinds` exist and the device grower emits one-hot
levels `[VERIFIED: CODEGRAPH; LOCAL crates/cb-train/src/boosting.rs:3908-4077]`.
Part 2 must **verify** this, not re-plan it.

### SPEC-CATF-Δ2 — supersedes BLOCKER-2
The four inert params become real in Part 1, so promoting them to `IMPLEMENTED` in
`crates/catboost-rs-py/src/params.rs` is honest — **except** that the promoted semantics are
the scalar single-description model (SPEC-CTRT-19). The promotion must carry that caveat.

### SPEC-CATF-Δ3 — supersedes BLOCKER-1 (locked user decision)
The public-API oracle reuses the **already-committed** `crates/cb-oracle/fixtures/one_hot_train/`
fixture. **No corpus-wide generation runs.** `crates/cb-oracle/generator/gen_fixtures.py`
falls through to `main()` on unrecognised argv and would regenerate the entire frozen corpus
`[VERIFIED: LOCAL gen_fixtures.py __main__]`. New Part-1 fixtures use the **fixture-local**
`fixtures/<name>/gen_fixtures.py` pattern (9 in-repo precedents), which structurally cannot
dirty the corpus.

### SPEC-CATF-Δ4 — supersedes CRITICAL-3
The categorical width check uses the pool's **DECLARED** cat width, never a width derived
from the splits the model happened to choose. A derived width is a data-dependent lower
bound and would reject legitimate `fit(pool) → predict(same pool)` round-trips whenever a
trailing cat column is one-hot-routed, constant, or simply unused.

### SPEC-CATF-Δ5 — resolves MAJOR-5
`data_to_pool` has **exactly 14** call sites; the four the plan omits are
`search.rs:304`, `search.rs:389`, `cv.rs:61`, `cv.rs:64`
`[VERIFIED: LOCAL grep -rn data_to_pool crates/catboost-rs-py/src/]`. All 14 must be updated
together or the crate does not build.

### SPEC-CATF-Δ6 — resolves MAJOR-6
`cv()` and `grid_search()` on a categorical pool must have defined behavior — either full
support or a typed rejection — with a test. Silent degradation is not acceptable.

### SPEC-CATF-Δ7 — resolves MAJOR-7
`predict_with` is not the only path that can silently mis-score a CTR model. Every public
scoring path that reaches a CTR-bearing model must either route categorical values or raise
a typed error: `predict`, `predict_proba`, `staged_predict`, and the fstr/SHAP entry points.

### SPEC-CATF-Δ8 — the motivating end-to-end gate
`bench/one_hot_gpu_speed/one_hot_bench_colab.py` currently bails `BLOCKED-FACADE-ROUTING`
via a preflight that greps for `train_cat` in `crates/catboost-rs/src/builder.rs`
`[VERIFIED: LOCAL]`. When Part 2 lands, that preflight must pass **without editing the
bench**.

---

## 7. Acceptance scenarios

| # | Scenario | Gate |
|---|---|---|
| A1 | Buckets CTR trains and predicts | `buckets_simple` ≤1e-5, both `target_border_idx` present |
| A2 | BTMV CTR trains and predicts | `btmv_simple` ≤1e-5, **plus** the accumulator-level f32 differential (E07 test fn 2) — see the A2 note below |
| A3 | Counter CTR trains and predicts | `counter_simple` ≤1e-5 + permutation-invariance |
| A4 | Multi-prior expansion | `borders_multiprior` ≤1e-5 |
| A5 | `is_simple` routing | `mixed_simple_vs_combo` ≤1e-5 |
| A6 | `counter_calc_method` | `counter_full_eval` (eval set) or recorded deferral |
| A7 | Borders regression firewall | SPEC-CTRT-05 bit-equality + all 11 existing CTR oracles |
| A8 | `.cbm` mean round-trip | save→load→save byte-identical; upstream BTMV `.cbm` loads |
| A9 | `.cbm` non-mean unchanged | byte-identical vs pre-change baseline |
| A10 | CPU-illegal types rejected | typed error, both types |
| A11 | `fit(X, y, cat_features=[...])` end to end | public Python API ≤1e-5 vs catboost 1.2.10 |
| A12 | One-hot wave still green | `one_hot_oracle`, `one_hot_draw_accounting`, `device_one_hot_parity` |
| A13 | Bench preflight passes | `one_hot_bench_colab.py` no longer bails, unedited |

**Note on A2 — which f32 differential is the GATE.** For binary classification the
BTMV added value is exactly `targetClass ∈ {0.0f, 1.0f}` (upstream passes
`targetClassesCount - 1 == 1`, `online_ctr.cpp:762`), and f32 and f64 accumulation of
small integers are **bit-identical below `2^24`**. Therefore the **fixture-scale**
differential (E13 test fn 3, 30 rows) **CANNOT discriminate f32 from f64 for binclf**;
it is a **REPORTING step, not a gate**, and it must print its measured
`REPORTED: f32/f64 indistinguishable at this scale (maxdiff = …)` line rather than
claim a pass. The gate for the f32 requirement is E07 test fn 2, an allocation-free
accumulator test that seeds `TCtrMeanHistory { sum: 16_777_216.0f32, count: 16_777_216 }`
and asserts the next `add(1.0)` does not move `sum`.

---

## 8. Impact scope

**Part 1 — MUST CHANGE:** `cb-train/src/ctr/{online,calc_ctr,final_ctr,bake,ctr_feature,mod}.rs`,
`cb-train/src/boosting.rs` (:1940, :3155), `cb-model/src/ctr_data.rs` (mean codec).
Classification: **cross-module** (`cb-train` → `cb-model` via the serialized `ctr_type`).

**Part 2:** `catboost-rs/src/{builder,model,lib}.rs`,
`catboost-rs-py/src/{params,estimator,classifier,regressor,ranker,search,cv,pool}.rs`.
Classification: **external/public** (it changes the public Rust and Python surface).

**Blocking constraints discovered:**

- `ctr_splits_for_tree` has **no covering test** `[VERIFIED: CODEGRAPH]` — add one first.
- **62 files pin `simple_ctr:` in a `BoostParams` literal** `[VERIFIED: LOCAL grep]`. Adding
  a field or changing a field type is a 62-file edit. **Adding no field and changing no type
  is free** — design accordingly.
- The **uncommitted one-hot device wave** touches `boosting.rs` in the same regions Part 1
  edits. Land Part 1 **on top of** it, not beside it, and re-run its oracles.
- The stale doc comment at `crates/cb-train/src/boosting.rs:815-816` ("EMPTY … until T19
  populates it") is now false; fix opportunistically.

---

## 9. Compatibility and migration

- `.cbm` v1 currently **rejects** mean tables. SPEC-CTRT-14/15 make them writable and
  readable. Existing non-mean `.cbm` files must remain byte-identical (SPEC-CTRT-16), so the
  change is additive and no migration is required for existing models.
- No `BoostParams` field is added or retyped, so all 62 construction sites are untouched.
- Default behavior is unchanged (`Borders`, `[0.5]`), so every existing oracle is a no-op
  gate.

---

## 10. Risks and open questions

| # | Risk | Prevention |
|---|---|---|
| R1 | Multi-prior expansion changes tie-breaks corpus-wide | land SPEC-CTRT-11 behind its own fixture, after 04/05; run the 11 existing CTR oracles every wave |
| R2 | f64-vs-f32 BTMV divergence at boundary bins | accumulate `Sum` in f32; differential test |
| R3 | `gen_fixtures.py` corpus wipe | fixture-local generators only; never invoke the corpus generator |
| R4 | `counter_calc_method` "verified" without an eval set | eval-set fixture mandatory, or explicit deferral |
| R5 | Merge friction with the one-hot wave | land on top; re-run its oracles |
| R6 | `ctr_splits_for_tree` change unguarded | add a covering test first |
| R7 | `.cbm` mean codec byte-format wrong | round-trip + upstream-produced `.cbm` load, not self-comparison alone |

**Open questions (none blocking):**

1. `[UNVERIFIED]` Whether `PrepareScratchBlob<int>` zero-initializes `TBucketsView` — the
   repo allocates zeroed vectors regardless.
2. `[UNVERIFIED]` Online-vs-final bucket space under a non-default `ctr_leaf_count_limit`
   (defaults to `u64::MAX`).
3. `[UNVERIFIED]` `PerfectHash` vs upstream `ComputeReindexHash` bin ordering — lookups are
   by hash, and Counter's MAX and Buckets' numerator are order-independent.
4. `[UNVERIFIED]` Whether the proposed 30-row isolating configs actually yield a winning
   split of each intended type — must be asserted **at generation time** (the mandatory
   anti-false-pass guard).

---

## 11. Traceability and sources

- `.planning/plans/ctr-type-engine-and-facade-routing/research.md` (932 lines, §A–§L).
- `.planning/plans/catboost-builder-cat-features-routing/{SPEC.md,PLAN.md,PLAN-CHECK.md}`.
- Upstream catboost **v1.2.10** tag: `online_ctr.{h,cpp}`, `ctr_type.h`, `restrictions.h`,
  `catboost_options.cpp`, `cat_feature_options.cpp`, `ctr_helper.{h,cpp}`,
  `greedy_tensor_search.cpp`.
- **Pending TreeFinder registration:** neither this document nor
  `catboost-builder-cat-features-routing/SPEC.md` is indexed
  `[VERIFIED: mcp__tree_finder__search_hierarchy returned only snapshot-resume and
  xgboost-rust-rewrite]`. Both should be registered under the `.planning/plans/**/SPEC.md`
  convention already used by those two documents.
