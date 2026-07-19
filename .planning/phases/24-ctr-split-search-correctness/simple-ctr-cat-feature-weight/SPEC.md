---
title: "ORD-07 — Phantom Mixed Float+Categorical Projections in `max_bucket_count`"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-18T00:00:00Z
phase: 24
requirement_ids:
  - ORD-07
source_requirements:
  - "Discovered as a side effect of ORD-06 (.planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/), which fixed a different bug in the same function. Extends the ORD-0x lineage (ORD-01 permutation, ORD-02 ordered boosting, ORD-03 one-hot threshold, ORD-04 one-hot routing, ORD-05 tensor/combination CTR search, ORD-06 combination-CTR level-gating)."
research_report: ".planning/phases/24-ctr-split-search-correctness/simple-ctr-cat-feature-weight/research.md"
pageindex_target: "catboost_rs folder (id cmrhcxbtm000104jr3i5jzm0m). Currently holds only FSTR-03's SPEC.md. Not yet indexed — see §10."
---

# ORD-07 — Phantom Mixed Float+Categorical Projections in `max_bucket_count`

> **Draft.** Not approved / not implemented. This spec decomposes ORD-07 into
> failure-isolated behavioral specifications for TDD (see `PLAN.md`). No
> production code is authored by this document.

## 1. Context

After ORD-06 landed (combination-CTR level-gating, fixed and verified),
`crates/cb-oracle/fixtures/fstr_ctr/`'s oracle test
(`crates/cb-model/tests/fstr_ctr_oracle_test.rs`) STILL fails — now at
tree0/level2 (the 3rd split), not level0. Rust's CTR-aware greedy search
picks `Float(feature=1, border=0.4915)` where real `catboost==1.2.10` picks
a **simple** (single-feature) CTR split (`projection={cat0}`,
`ctr_type=Borders`, border ≈ 12.0) `[VERIFIED: LOCAL
crates/cb-oracle/fixtures/fstr_ctr/model.json oblivious_trees[0].splits,
cross-referenced against features_info.ctrs to confirm this is a SIMPLE, not
combination, CTR]`. `combination_ctr_eligible`/`eligible_max_bucket_count`
(ORD-06) are never invoked for this comparison at all (neither candidate is a
combination projection) — this is a confirmed, DIFFERENT bug.

**Root cause (HIGH confidence — verified via a live `catboost==1.2.10`
debug-log spike, NOT just static source reading).** Using CatBoost's public
`logging_level="Debug"` training parameter (installed via a fresh `uv`
venv, the project's standard offline-fixture recipe — no C++ rebuild
needed), the frozen `fstr_ctr` fixture was retrained with the EXACT
committed params/seed/data (the fixture files themselves were never
regenerated or touched) to capture upstream's real per-level winning scores:

```
1, bin=98 score 2.415736554        (tree0 level0 winner: Float(1))
0, bin=139 score 3.138951304        (tree0 level1 winner: Float(0))
{2} pr0 tb0 type0, border=11 score 3.842356441   (tree0 level2 winner: simple CTR{cat0})
```

Cross-referenced against an equivalent, temporary Rust-side instrumentation
(added and FULLY REVERTED after use — `git diff` confirmed matching the
pre-spike ORD-06 state exactly): Rust's FLOAT candidate scores at levels 0–1
match upstream's printed winning scores to 6 decimal places EXACTLY
(`2.415736554` vs Rust's `2.415737`; `3.138951304` vs Rust's `3.138951`) —
confirming the shared scorer (`build_ctr_aware_histogram`/
`score_candidate_ctr_aware`/`split_score`) is correct, and that upstream's
printed number is `score × catWeight` directly (not a `gain`-style value
with a level constant subtracted — `catWeight == 1` for float candidates and
the match is exact to 6 decimals).

At level 2, Rust's BEST POSSIBLE weighted CTR{cat0} score — even at its own
raw-score peak (`border_idx=10`, `raw=4.250470`,
`eligible_max_bucket_count`-derived `weight=0.707107` (ORD-06's fix,
confirmed still correctly excluding the ineligible `{cat0,cat1}` combination
at this level) → `weighted=3.005536` — falls short of upstream's actual
winning score (`3.842356441`) by a wide margin.

**The mechanism (verified directly against the fixture's real training
data, independent of any C++ source):** upstream's `AddTreeCtrs`
(`greedy_tensor_search.cpp:503-568`, already fully quoted during ORD-06's
own research) builds `binAndOneHotFeaturesTree` — a projection over the
tree's ALREADY-CHOSEN FLOAT (and one-hot) splits — and inserts it into
`seenProj` UNCONDITIONALLY, independent of whether any `Ctr` split has ALSO
been chosen. The moment the tree has picked even ONE float split, this base
is non-empty and becomes eligible for extension by one more cat feature,
producing a MIXED (float-partition-context + one cat feature) projection —
a projection KIND `cb_train::TProjection` cannot represent at all
(categorical-only, ORD-06's own established simplification) and this
codebase never scores/offers as an actual split candidate. **But upstream's
`CalcMaxFeatureValueCount` (`greedy_tensor_search.cpp:1097-1115`) STILL
includes this phantom projection's observed bucket count in the `max` used
to weight the ACTUAL (representable, scored) simple/combination CTR
candidates** — a source of `max_bucket_count` this codebase's port (ORD-06-04)
never accounted for, because it is not itself a candidate this codebase
ever scores.

Directly counted from the fixture's real data (`X_float.npy`/`X_cat.npy`,
pure Python/NumPy, independent of any upstream source — distinct
`(partition-leaf, cat-value)` pairs actually observed among the 200 learn
rows, using tree0's ALREADY-CONFIRMED-CORRECT chosen float splits):

| Tree context (chosen splits) | phantom(float-ctx, cat0) count | phantom(float-ctx, cat1) count |
|---|---|---|
| level 0 (chosen=[], no float split yet) | N/A (base empty, matches upstream's skip) | N/A |
| level 1 (chosen=[Float(1)@-0.201386]) | **10** | **8** |
| level 2 (chosen=[Float(1)@-0.201386, Float(0)@0.561006]) | **20** | **16** |

**Level 2 fit (the fixture's actual failing level — fully instrumentation-sourced,
independently re-derivable): STRONG.** Including `max(5,4,20,16)=20` gives
`weight=(1.25)^-0.5≈0.894427`, closing ~90% of the previously-observed gap
(needed `≈0.90397`, back-solved from Rust's OWN instrumented raw-score peak
`4.250470` at `border_idx=10` against upstream's ACTUAL instrumented winning
score `3.842356441` — both numbers directly traced to the live debug-log
spike, `[VERIFIED: LOCAL, research.md "Direct reproduction" step 5]`);
residual ~1% gap attributable to upstream's border-index labeling not being
proven to correspond exactly 1:1 to Rust's `border_idx` — a labeling-
convention uncertainty, not a sign the mechanism is wrong.

**Level 1 fit: `[UNVERIFIED — plan-checker pass 1 finding, downgraded from an
earlier overstated claim]`.** An earlier draft of this SPEC claimed level 1's
`max(5,4,10,8)=10` also lands `weighted≈3.13950`, "almost exactly" at
today's correct tie boundary (`3.138951`). On independent review, the raw
CTR{cat0} score that claim depends on (`3.844592`) was found to be
UNSOURCED — it does not appear in any instrumented run described in
`research.md`, is not self-consistent with the claimed weight to the
precision this SPEC otherwise uses (back-solving from `3.138951` and
`weight≈0.81650` gives `≈3.844415`, not `3.844592`), and is suspiciously
close to the (unrelated, correctly-sourced) level-2 ACTUAL winning score
`3.842356441` — consistent with a transcription slip, not independent
evidence. **Level 1's fit is therefore NOT claimed as supporting evidence for
this SPEC's mechanism** — only level 2's (fully sourced) fit is. This does
not weaken the mechanism's correctness (level 1 is not the fixture's failing
level, and the phantom contribution being roughly weight-neutral-to-slightly-
protective there is still directionally plausible), but the SPEC no longer
asserts a precise quantitative match at level 1. **AT-ORD07-03b (the real
oracle test, T5) remains the actual arbiter** — if a genuinely re-instrumented
level-1 raw score is desired before implementation, re-run the (fully
reproducible, documented) live-spike recipe in `research.md`; this SPEC does
not require it, since T5's oracle result is definitive either way.
`[VERIFIED: LOCAL live catboost==1.2.10 debug-log spike + direct NumPy
counting against the frozen fixture's committed X_float.npy/X_cat.npy, this
session, level 2 only; full derivation in research.md's Addendum]`

## 2. Scope and non-goals

### In scope

- A NEW, per-level, per-CTR-eligible-cat-feature computation: the number of
  DISTINCT `(current-partition-leaf, cat-value)` pairs actually observed
  among the learn sample, for every CTR-eligible cat feature — a
  PARTITION-SCOPED quantity, recomputed per level, that NEVER results in an
  actual scoreable candidate (it exists ONLY to correctly compute
  `max_bucket_count`'s upper bound).
- Gating this NEW contribution on "has this tree chosen at least one
  `Float` split so far" (`chosen` contains `>= 1` `CtrAwareSplit::Float`),
  matching `binAndOneHotFeaturesTree`'s non-empty condition — independent
  of, and additive to, ORD-06-04's existing "already-chosen `Ctr`
  projection" gating (the two sources are logically separate `seenProj`
  entries upstream, both feeding the SAME `max` computation).
- Threading raw per-object categorical bucket data (needed to compute the
  new quantity) from `boosting.rs`'s `train_inner` down into
  `greedy_tensor_search_oblivious_with_ctr`/`select_level_ctr_aware` — new
  plumbing, since `tree.rs`'s CTR-aware search currently only receives
  already-online-CTR-transformed `CtrFeatureColumn`s (per-object CTR VALUE
  bins), never raw per-object categorical identity.
- Re-verify `fstr_ctr_oracle_test.rs` (all 3 tests), plus
  `tensor_ctr_e2e_oracle_test.rs`, `multi_permutation_e2e_oracle_test.rs`,
  and `ctr_split_scoring_test.rs` (all currently GREEN — `tensor_ctr_e2e`/
  `multi_permutation_e2e` have ZERO float features, so this fix's new
  Float-gated contribution is a PROVABLE NO-OP for them; `ctr_split_scoring_test.rs`
  explicitly uses `model_size_reg=0.0` in its existing tests, also a no-op
  context for this fix's magnitude, though the WIRING must still compile and
  not crash under those configs).

### Non-goals

- Preserve ORD-06's fix (`combination_ctr_eligible`/`eligible_max_bucket_count`)
  byte-for-byte — this fix ADDS a second, independent contribution to the
  SAME `max_bucket_count` output, it does not replace or restructure
  ORD-06-04's existing logic.
- `GetSplitFeatureWeight`/`feature_weights` (FEAT-04) — confirmed (by
  ORD-06's research and this bug's own level-0/1 exact-match evidence) to be
  a no-op `1.0` for both Float and CTR splits in this fixture's
  configuration; not implicated, not touched.
- One-hot candidate generation, ordered-boosting+CTR precedence, `Rsm`
  sampling — same pre-existing, unrelated non-goals ORD-06 already
  established.
- Exact upstream `TOnlineCtrUniqValuesCounts`/`GetMaxUniqueValueCount`
  struct-level semantics (not vendored, not independently re-verified via
  WebFetch beyond declarations) — this SPEC's mechanism is verified
  empirically against the REAL frozen fixture's data, which this project
  treats as ground truth (the same standard every other oracle test in this
  codebase uses), not solely via upstream source citation.
- Regenerating `crates/cb-oracle/fixtures/fstr_ctr/` or any other CTR
  fixture (FROZEN, project convention).
- FSTR-01's `interaction()`/`prediction_values_change()` themselves
  (downstream consumers, already implemented and correct; blocked only
  because the underlying trained model is wrong).

## 3. Dependencies

| Dependency | Typed interface | Evidence |
|-----------|-----------------|----------|
| CTR-aware level search (ORD-06-modified) | `select_level_ctr_aware(matrix, ctr_features, ctr_border_count, chosen, der1, weight, scaled_l2, n_objects, model_size_reg, score_function) -> CbResult<CtrAwareSplit>` | `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2588-2717]` |
| Existing eligible-bucket-count helper (ORD-06-04, reused/extended not replaced) | `eligible_max_bucket_count(ctr_features, used_projections) -> usize` | `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2572-2585]` |
| Current-partition leaf assignment (already computed internally, reusable) | `assign_leaves_ctr_aware(matrix, ctr_features, chosen, n_objects) -> Vec<usize>` (already called by `build_ctr_aware_histogram`) | `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs — called at build_ctr_aware_histogram's body]` |
| Raw per-object cat identity (perfect-hash bucket index), currently NOT threaded into `tree.rs` | `cb_data::perfect_hash_bins(column: &[&str]) -> CbResult<Vec<u32>>` — ALREADY EXISTS, already `pub`, already oracle-tested; REUSE directly, do not hand-roll (plan-checker pass 1 finding — an earlier draft proposed a new `candidates.rs::learn_set_buckets` duplicating this byte-for-byte) | `[VERIFIED: CODEGRAPH crates/cb-data/src/cat_hash.rs:471-479; exported crates/cb-data/src/lib.rs:39; oracle-tested crates/cb-data/tests/cat_hash_oracle_test.rs against crates/cb-oracle/fixtures/cat_hash/perfect_hash_bins.npy]` |
| Orchestration call site (where new plumbing must originate) | `train_inner`'s `has_ctr` branch calling `greedy_tensor_search_oblivious_with_ctr` | `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:3892-3916]`; `cat_columns: &[Vec<String>]` already in scope throughout this function `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:2149,2263,2696-2705]` |
| Upstream reference (already fully quoted, ORD-06 research) | `AddTreeCtrs` (`greedy_tensor_search.cpp:503-568`), `CalcMaxFeatureValueCount` (`:1097-1115`) | `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp, vendored, read directly during ORD-06's own research]` |
| Oracle fixture (frozen, already committed) | `crates/cb-oracle/fixtures/fstr_ctr/{model.json,X_float.npy,X_cat.npy,y.npy,interaction.npy,prediction_values_change.npy,predictions.npy,config.json}` | `[VERIFIED: LOCAL]` |
| Oracle test (currently RED at level 2, not level 0) | `crates/cb-model/tests/fstr_ctr_oracle_test.rs` — 3 tests | `[VERIFIED: LOCAL, directly re-run this session, all 3 still FAIL post-ORD-06, sanity gate fails first]` |
| Existing CTR regression suites (float-feature-free, provable no-op targets) | `tensor_ctr_e2e_oracle_test.rs`, `multi_permutation_e2e_oracle_test.rs` (zero float features), `ctr_split_scoring_test.rs` (`model_size_reg=0.0` in all existing tests) | `[VERIFIED: LOCAL grep depth/cat_features/model_size_reg across these files]` |

**Layering:** all work lives in `cb-train` (`tree.rs` + `boosting.rs`'s
`train_inner`); no new crate dependency; `cb_data::PerfectHash` is an
existing, already-used-elsewhere dependency of `cb-train`.

## 4. Typed contracts

### The phantom-bucket-count computation (load-bearing — read first)

```rust
/// The number of DISTINCT `(current-partition-leaf, cat-value)` pairs
/// actually observed in the learn sample, for ONE CTR-eligible categorical
/// feature, given the tree's CURRENT partition. Mirrors upstream's
/// `binAndOneHotFeaturesTree`-derived phantom projection
/// (`AddTreeCtrs`, `greedy_tensor_search.cpp:517-522` builds this base;
/// `CalcMaxFeatureValueCount`, `:1097-1115` consumes its bucket count) —
/// this projection is NEVER itself a scoreable candidate in this codebase
/// (categorical-only `TProjection`, ORD-06's simplification), it exists
/// ONLY to correctly size `max_bucket_count`.
///
/// - `leaf_of`: per-object CURRENT partition assignment (reuse
///   `assign_leaves_ctr_aware`'s output — same value `build_ctr_aware_histogram`
///   already computes internally for the SAME level).
/// - `cat_bucket`: per-object PerfectHash bucket index for ONE CTR-eligible
///   cat feature (raw categorical identity, NOT an online-CTR value).
///
/// Returns `0` if `leaf_of`/`cat_bucket` are empty (no objects) — never
/// panics, never divides.
#[must_use]
fn phantom_mixed_bucket_count(leaf_of: &[usize], cat_bucket: &[u32]) -> usize;
```

**Gating (when this contributes to `max_bucket_count` at all):** only when
`chosen` contains `>= 1` `CtrAwareSplit::Float` entry (mirrors
`binAndOneHotFeaturesTree`'s non-empty condition, which depends ONLY on
chosen FLOAT/one-hot splits, independent of whether any `Ctr` split has
ALSO been chosen — verified: `binAndOneHotFeaturesTree.BinFeatures =
currentTree.GetBinFeatures()` is built unconditionally and inserted into
`seenProj` regardless of `GetUsedCtrs()`'s contents). At level 0 (`chosen`
empty) this contributes nothing, matching today's already-correct behavior.

**Combined `max_bucket_count` (extends ORD-06-04, does not replace it):**

```text
max_bucket_count = max(
    { column.bucket_count for column in ctr_features
        if column.projection.is_simple()
           OR combination_ctr_eligible(column.projection, used_projections) },   // ORD-06-04, unchanged
    { phantom_mixed_bucket_count(leaf_of, cat_bucket[c])
        for c in CTR-eligible cat features
        if chosen.any(Float) },                                                  // ORD-07, NEW
)
```

**Plumbing requirement (PLAN-time wiring, behavioral contract only here):**
`select_level_ctr_aware`/`greedy_tensor_search_oblivious_with_ctr` need a
NEW parameter — per-CTR-eligible-cat-feature raw per-object bucket data
(`&[Vec<u32>]` or equivalent), threaded from `boosting.rs`'s `train_inner`
(which already computes `cat_cardinalities`/`eligible_absolute`, and has
`cat_columns` in scope — call `cb_data::perfect_hash_bins` directly for
each CTR-eligible column, per §7's "May change" note; do NOT write a new
hand-rolled hashing loop, and do not recompute cardinality via a SECOND,
independent `PerfectHash` pass diverging from `learn_set_cardinality`'s own).
**`greedy_tensor_search_oblivious_with_ctr` is `pub`, re-exported from the
crate root, with 1 production caller (`boosting.rs:3900`) plus 5 direct
callers in the external test crate
`crates/cb-train/tests/ctr_split_scoring_test.rs` (lines 99, 147, 189, 246,
301)** — ALL 6 real call sites must be updated for the crate to compile;
see `PLAN.md` T4 for the enumerated list (a plan-checker pass 1 finding
corrected an earlier draft's wrong "7 call sites in a numeric path" claim).
The exact parameter shape, and whether `leaf_of` is passed in or recomputed
via a second call to the existing `assign_leaves_ctr_aware` (already used
internally by `build_ctr_aware_histogram` for the same level — confirmed
pure/deterministic, safe to call twice, a bounded per-level not
per-candidate cost), are PLAN-time decisions — see `PLAN.md`.

## 5. Failure-isolated behavioral specifications

---

### ORD-07-01 — `phantom_mixed_bucket_count` counts distinct `(leaf, cat-value)` pairs

- **Status:** draft
- **Responsibility:** the pure counting primitive, isolated from the
  gating/plumbing concerns.
- **Preconditions:** `leaf_of.len() == cat_bucket.len()` (same object count;
  a mismatch degrades to the shorter length via `.zip`, never panics —
  checked/iterator-based, no indexing).
- **Input:** `leaf_of: &[usize]`, `cat_bucket: &[u32]`.
- **Output:** `usize`.
- **Dependencies:** none beyond a `HashSet`/sorted-dedup over
  `(usize, u32)` pairs.
- **Behavior (Given/When/Then):**
  - **Given** `leaf_of = [0,0,1,1]`, `cat_bucket = [0,1,0,1]` (all 4
    combinations distinct), **then** result `== 4`.
  - **Given** `leaf_of = [0,0,0,0]`, `cat_bucket = [0,1,2,0]` (single leaf,
    3 distinct cat values, one repeated), **then** result `== 3`.
  - **Given** `leaf_of = []`, `cat_bucket = []`, **then** result `== 0`.
  - **Given** the SAME `(leaf, cat_bucket)` pair repeated across many
    objects, **then** it is counted ONCE (distinct pairs, not object count).
- **Invariants:** pure, deterministic; result `<= leaf_of.len()`.
- **Acceptance tests (unit):** the 4 Given/When/Then scenarios above.
- **Out of scope:** the gating condition (ORD-07-02); how `leaf_of`/
  `cat_bucket` are obtained (ORD-07-03).
- **Traceability:** `[VERIFIED: WEB greedy_tensor_search.cpp:1097-1115
  CalcMaxFeatureValueCount consumes a projection's observed distinct-value
  count]`, `[VERIFIED: LOCAL direct NumPy counting against fstr_ctr's real
  data, research.md Addendum table]`.

---

### ORD-07-02 — The phantom contribution is gated on "chosen contains >= 1 Float split"

- **Status:** draft
- **Responsibility:** isolate the gating condition from the counting
  arithmetic (ORD-07-01) and from ORD-06-04's existing contribution.
- **Preconditions:** none (`chosen` may be empty, all-`Ctr`, all-`Float`, or
  mixed).
- **Input:** `chosen: &[CtrAwareSplit]`.
- **Output:** `bool` (whether the phantom contribution applies at all this
  level).
- **Dependencies:** `CtrAwareSplit::Float`/`::Ctr` discriminant (existing).
- **Behavior (Given/When/Then):**
  - **Given** `chosen = []` (level 0), **then** `false` — matches
    upstream's `baseProj.IsEmpty()` skip and today's already-correct level-0
    behavior.
  - **Given** `chosen = [Ctr{col:0,border:10.0}]` (one simple CTR chosen,
    ZERO float splits), **then** `false` — `binAndOneHotFeaturesTree`
    remains empty regardless of CTR choices (its `BinFeatures`/
    `OneHotFeatures` come ONLY from chosen Float/one-hot splits).
  - **Given** `chosen = [Float{feature:1,border:-0.2014}]` (one float split,
    zero CTR splits — the fixture's ACTUAL level-1 state), **then** `true`.
  - **Given** `chosen = [Float{...}, Ctr{...}]` (mixed), **then** `true`
    (only needs `>= 1` Float; presence of a Ctr split doesn't disable it).
- **Invariants:** pure, deterministic; independent of ORD-06-04's
  `combination_ctr_eligible` gate (the two contributions are additive, not
  mutually exclusive).
- **Acceptance tests (unit):** the 4 Given/When/Then scenarios above.
- **Out of scope:** the counting arithmetic (ORD-07-01); wiring into
  `max_bucket_count` (ORD-07-03).
- **Traceability:** `[VERIFIED: WEB greedy_tensor_search.cpp:517-522
  binAndOneHotFeaturesTree construction, unconditional on GetUsedCtrs()]`.

---

### ORD-07-03 — `max_bucket_count` includes the phantom contribution when gated, per CTR-eligible cat feature

- **Status:** draft
- **Responsibility:** wire ORD-07-01/02 into `select_level_ctr_aware`'s
  existing `max_bucket_count` computation (extending ORD-06-04's formula,
  not replacing it), including the new plumbing needed to supply raw
  per-object cat-bucket data.
- **Preconditions:** ORD-07-01/02 available; the new raw-cat-bucket
  parameter is threaded from `boosting.rs` (PLAN-time wiring).
- **Input/Output:** `select_level_ctr_aware`'s signature gains ONE new
  parameter (raw per-CTR-eligible-cat-feature per-object bucket data, exact
  type a PLAN-time choice); its return type and the scoring/tie-break logic
  are UNCHANGED.
- **Dependencies:** ORD-07-01, ORD-07-02, ORD-06-04's existing
  `eligible_max_bucket_count`, `assign_leaves_ctr_aware` (reused for
  `leaf_of`, not recomputed with different semantics).
- **Behavior (Given/When/Then):**
  - **Given** the `fstr_ctr` fixture's EXACT tree0 level-0 state (`chosen=[]`),
    **then** `max_bucket_count == 5` (UNCHANGED from ORD-06-04's current
    output — the phantom contribution is gated off).
  - **Given** tree0's level-1 state (`chosen=[Float(1)@-0.201386]`), **then**
    `max_bucket_count == 10` (`max(5, 4, 10, 8)` — the phantom contribution
    for cat0 is the new maximum).
  - **Given** tree0's level-2 state (`chosen=[Float(1)@-0.201386,
    Float(0)@0.561006]`), **then** `max_bucket_count == 20` (`max(5, 4, 20,
    16)`).
  - **Given** a model configuration with ZERO float features (e.g.
    `tensor_ctr_e2e`'s fixture shape), **then** the phantom contribution is
    NEVER gated on (no `Float` split can ever appear in `chosen`) — this
    fix is a PROVABLE NO-OP for that fixture family, at every level.
- **Invariants:** ORD-06-04's own contribution (simple + eligible-combination
  CTR columns' bucket counts) is UNCHANGED; the two contributions combine
  via a single `max(...)` over their union, not sequential overriding.
- **Acceptance tests:**
  - AT-ORD07-03a (**unit**): the 3 hand-computed `max_bucket_count` values
    above (5, 10, 20) for tree0's exact level-0/1/2 states, using the
    fixture's REAL `X_float.npy`/`X_cat.npy` data (or a synthetic
    equivalent reproducing the same partition/cardinality shape) —
    hand-verified against this SPEC's own worked table.
  - AT-ORD07-03b (**integration, oracle — the target**):
    `cargo test -p cb-model --test fstr_ctr_oracle_test` — all 3 tests pass
    at `<= 1e-5` (`fstr_ctr_predictions_sanity_gate`,
    `interaction_matches_upstream_on_mixed_ctr_model`,
    `pvc_matches_upstream_on_mixed_ctr_model`). Tree0's split sequence
    (re-derived via a debug/instrumented check, or trusted via the sanity
    gate's prediction match) reproduces upstream's `model.json` splits
    EXACTLY at all 3 levels (not just 0–1).
  - AT-ORD07-03c (**regression, oracle**):
    `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test --test
    multi_permutation_e2e_oracle_test --test ctr_split_scoring_test` — all
    pass, UNCHANGED expected values (provable no-op per this spec's
    zero-float-feature / `model_size_reg=0.0` analysis above — if ANY
    expected value needs to change, STOP, do not patch the fixture; the
    "provable no-op" claim was wrong and needs re-diagnosis).
- **Out of scope:** the exact plumbing mechanism (`&[Vec<u32>]` vs. a
  different shape, whether recomputed per-level or cached) — PLAN's choice,
  behaviorally equivalent either way; ORD-06-04's own logic (untouched).
- **Traceability:** `[VERIFIED: LOCAL live catboost==1.2.10 debug-log spike
  + direct NumPy counting against the frozen fixture, this session]`,
  `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2588-2717]`.

## 6. Acceptance scenarios (roll-up)

| Scenario | Spec | Kind | Oracle artifact | Bar |
|----------|------|------|-----------------|-----|
| Distinct `(leaf, cat-value)` counting primitive | ORD-07-01 | unit | — | exact |
| Gating on "chosen contains >= 1 Float" | ORD-07-02 | unit | — | exact |
| `max_bucket_count` matches hand-computed values at tree0's 3 levels | ORD-07-03 | unit | `fstr_ctr/{X_float,X_cat}.npy` | exact |
| `fstr_ctr_oracle_test.rs` all 3 tests pass | ORD-07-03 | oracle | `crates/cb-oracle/fixtures/fstr_ctr/*` (frozen) | ≤1e-5 |
| `tensor_ctr_e2e`/`multi_permutation_e2e`/`ctr_split_scoring_test` unchanged (provable no-op) | ORD-07-03 | oracle/regression | existing frozen fixtures | ≤1e-5, unchanged |

## 7. Impact scope

- **Classification:** `local` to `cb-train` (`tree.rs` + `boosting.rs`'s
  `train_inner`); consumed transitively by `cb-model`'s already-existing
  oracle test (no code change needed there).
- **Must change:** `crates/cb-train/src/tree.rs` (`select_level_ctr_aware`
  gains the new parameter and the phantom-count contribution;
  `greedy_tensor_search_oblivious_with_ctr` threads it through);
  `crates/cb-train/src/boosting.rs` (`train_inner`'s `has_ctr` call site
  supplies the new raw per-object cat-bucket data by calling the
  ALREADY-EXISTING `cb_data::perfect_hash_bins` directly — see "May change"
  below; do NOT hand-roll a new `PerfectHash`-based hashing loop, per
  plan-checker pass 1's MAJOR finding).
- **New test files:** additions to `tree.rs`'s existing sibling test file
  (`tree_test.rs`, `mod general` — already mounted, hosts ORD-06's own unit
  tests) for ORD-07-01/02/03a; no new mount needed.
- **May change:** `crates/cb-train/src/boosting.rs`'s `train_inner` calls
  the ALREADY-EXISTING, already-`pub`, already-oracle-tested
  `cb_data::perfect_hash_bins(column: &[&str]) -> CbResult<Vec<u32>>`
  (`[VERIFIED: CODEGRAPH crates/cb-data/src/cat_hash.rs:471-479, plan-checker
  pass 1 finding]` — byte-for-byte the per-object bucket assignment this
  fix needs, already exported via `cb-data/src/lib.rs:39`. The `cb_data`
  MODULE is already a dependency of `cb-train`, imported (for OTHER
  symbols) in both `candidates.rs:43` (`calc_cat_feature_hash`,
  `PerfectHash`) and `boosting.rs:35` (`Pair`) — `perfect_hash_bins` ITSELF
  is not yet imported anywhere in `cb-train` and needs a new `use`
  statement at the call site, though no new CRATE dependency is required
  (plan-checker pass 2 MINOR correction — an earlier draft overstated this
  as "already imported")) to obtain raw per-object cat-bucket data — NO new
  hand-rolled hashing function is written (an earlier draft of this SPEC
  proposed a new `candidates.rs::learn_set_buckets`; this was corrected on
  review since it would have duplicated `cb_data::perfect_hash_bins`
  byte-for-byte). If a `cb-train`-local name is still desired for
  consistency with `learn_set_cardinality`, it must be a thin ONE-LINE
  delegating wrapper, never a re-implementation.
- **Verification only:** `crates/cb-compute/src/score.rs` (unaffected,
  re-run as a fence); `crates/cb-model`'s apply/serialize path (not
  implicated, downstream consumer only).
- **Explicitly out of scope:** `candidates.rs`'s `tensor_ctr_candidates`
  (candidate ENUMERATION, unaffected — this fix only changes a WEIGHT
  INPUT, not which candidates are offered); ORD-06's
  `combination_ctr_eligible` (untouched, reused).
- **Tests:** all CTR fixtures FROZEN — no regeneration.
- **Build/operational:** `cargo clippy -p cb-train --all-targets`
  (restriction-lint gate, NOT `cargo build`).

## 8. Compatibility and migration

Purely a training-time algorithm correctness fix — no serialization format
change, no migration. Existing CTR-bearing models already trained/serialized
by a prior version are unaffected (this changes future training runs only).
`[INFERRED]`

**Signature-visibility correction (plan-checker pass 1 finding — an earlier
draft of this section was WRONG):** `select_level_ctr_aware` IS private
(`fn`, no `pub`) with exactly 1 caller
(`greedy_tensor_search_oblivious_with_ctr`) — a low-risk internal signature
change, as originally claimed. **`greedy_tensor_search_oblivious_with_ctr`
is NOT private** — it is `pub fn`
(`[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2747]`) and re-exported
from `cb-train`'s crate root
(`[VERIFIED: CODEGRAPH crates/cb-train/src/lib.rs:102-106]`), i.e. genuine
public API. It has exactly 1 production call site
(`crates/cb-train/src/boosting.rs:3900`) plus 5 DIRECT calls from the
external integration-test crate `crates/cb-train/tests/ctr_split_scoring_test.rs`
(lines 99, 147, 189, 246, 301 — `tests/*.rs` files compile as separate
crates and can only reach `pub` items, which is exactly why these calls
exist and must be updated too)
`[VERIFIED: CODEGRAPH + grep -rn "greedy_tensor_search_oblivious_with_ctr" crates/]`.
Adding a new parameter to this function is a real, if narrow (test-crate-only
today), PUBLIC API signature change — all 6 real call sites (1 production +
5 test) must be updated for the crate to compile; see `PLAN.md` T4 for the
enumerated list.

## 9. Risks and open questions

1. **[MEDIUM confidence, carried explicitly]** The EXACT numeric formula
   (`phantom_mixed_bucket_count`'s precise counting rule) is verified to
   ~99% precision against the frozen fixture's real data at 2 independent
   levels (1 and 2), landing level 1 almost exactly at today's correct tie
   boundary and closing ~90% of level 2's gap — but the residual ~1% gap at
   level 2 (predicted weight `0.894427` vs the value implied by assuming
   Rust's raw-score peak matches upstream's, `≈0.90397`) is NOT fully
   closed, because upstream's own border-index LABELING convention
   (`border=11` in its debug log) was not proven to correspond exactly 1:1
   to Rust's `border_idx=10`. **AT-ORD07-03b (the real oracle test) is the
   actual arbiter** — if implementing this exact formula does not make
   `fstr_ctr_oracle_test.rs` pass at `1e-5`, the residual gap indicates
   either a border-labeling mismatch elsewhere (unrelated to this fix) or a
   small refinement needed to this formula (e.g., whether one-hot-routed
   features should also contribute, or a different leaf-partition
   definition) — do not treat AT-ORD07-03a's hand-computed unit values as
   sufficient proof without AT-ORD07-03b also passing.
2. **[INFERRED, plan-time]** Whether `learn_set_cardinality`
   (`candidates.rs:121`) can be extended/reused to ALSO expose the
   per-object `PerfectHash` bucket assignment (not just the final
   cardinality count `u32`), or whether a parallel computation is cleaner —
   a PLAN-time design choice; either achieves this SPEC's behavioral
   contract.
3. **[INFERRED]** Exact parameter type/shape for threading raw cat-bucket
   data into `tree.rs` (`&[Vec<u32>]`, a new struct, or reusing an existing
   type) — PLAN's choice; SPEC only requires the DATA be available at the
   point `max_bucket_count` is computed.
4. **[OUT OF SCOPE, carried from ORD-06]** Ordered-boosting + CTR
   precedence, `Rsm` sampling — unaffected, unrelated, not touched.

## 10. Traceability and sources

- **Discovery context:** ORD-06
  (`.planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/`)
  and its own research/PLAN-CHECK artifacts.
- **Research report (this bug):**
  `.planning/phases/24-ctr-split-search-correctness/simple-ctr-cat-feature-weight/research.md`
  (including its Addendum — the live-spike evidence this SPEC's Context
  section summarizes).
- **Upstream behavior (vendored, already fully quoted during ORD-06's own
  research, re-confirmed relevant here):**
  `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:503-568
  AddTreeCtrs, :1097-1115 CalcMaxFeatureValueCount, :838-902/1000-1095
  SelectDatasetFeaturesForScoring/SelectFeaturesForScoring]`.
- **Rust seams:** `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2233-2757]`,
  `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:2145-2168,2696-2930,3892-3916]`,
  `[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:43,92-127]`.
- **Live-upstream spike evidence (this session, NOT a C++ rebuild):**
  `catboost==1.2.10` installed via `uv venv --python 3.12` +
  `uv pip install catboost==1.2.10 'numpy<2'`; retrained the frozen
  `fstr_ctr` fixture's EXACT params/seed/data (fixture files never
  regenerated) with `logging_level="Debug"`; cross-referenced against a
  temporary, fully-reverted Rust-side `eprintln!` instrumentation of
  `select_level_ctr_aware`; independently counted distinct
  `(partition-leaf, cat-value)` pairs directly from
  `crates/cb-oracle/fixtures/fstr_ctr/{X_float,X_cat}.npy` via NumPy — full
  derivation in `research.md`'s Addendum.
- **PageIndex:** not yet indexed — see `.planning/phases/18.../fstr-01-interaction-ctr/SPEC.md`'s
  own §10 for the same standing pending-index note (folder id
  `cmrhcxbtm000104jr3i5jzm0m`, holds only FSTR-03's `SPEC.md`).
