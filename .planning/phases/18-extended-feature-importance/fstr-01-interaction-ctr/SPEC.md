---
title: "FSTR-01 — Interaction and PredictionValuesChange Feature Importance with CTR (categorical) Support"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-17T00:00:00Z
phase: 18
requirement_ids:
  - FSTR-01
source_requirements:
  - ".planning/REQUIREMENTS.md (FSTR-01) — git-recovered (commit a82289c); NOT in the working tree. Confirm the canonical revision before flipping the requirement checkbox."
  - ".planning/ROADMAP.md (Phase 18) — git-recovered (commit a82289c); not in the working tree."
research_report: ".planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/research.md"
pageindex_target: "catboost_rs folder (id cmrhcxbtm000104jr3i5jzm0m). Currently holds only the FSTR-03 SPEC.md. This document is NOT YET indexed — the MCP's process_document ingests files as NEW documents with no in-place Markdown upsert, so indexing this SPEC would sit alongside (not replace) the FSTR-03 one. Human owner should index it as a second document in the same folder. See §10."
---

# FSTR-01 — Interaction and PredictionValuesChange with CTR Support

> **Draft.** Not approved / not implemented. This spec decomposes FSTR-01 into
> failure-isolated behavioral specifications for TDD (see `PLAN.md`). No
> production code is authored by this document.

## 1. Context

catboost-rs computes three feature-importance modes in `cb-model/src/fstr.rs`:
`prediction_values_change`, `interaction`, `loss_function_change`
`[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:35-37 pub use]`. The first two
are **structurally CTR-blind today**: both the OBLIVIOUS bit-indexed arm and the
NON-SYMMETRIC DFS/recursion arm treat any `ModelSplit::Ctr` split as invisible —
every consumer calls `ModelSplit::float_feature()`, which returns `None` for a
CTR split, and `continue`/`return`s past it
`[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:149-151 (PVC oblivious),
225-233 (PVC non-symmetric), 316-321 (Interaction oblivious), 448-455
(Interaction non-symmetric)]`. A model containing categorical (CTR) splits
therefore silently produces an **incomplete** `PredictionValuesChange` /
`Interaction` importance — CTR-split effect is computed and skipped, not
redistributed to the categorical features it depends on.

This is not a missing feature so much as an **unfinished** one: CTR structural
support (`ModelSplit::Ctr`, `CtrSplit`, `apply.rs::predict_raw_cat`,
`Model::from_trained`'s CTR-split construction) already exists and is
oracle-tested for catboost-rs's own trained categorical models
`[VERIFIED: CODEGRAPH crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs;
crates/cb-model/src/apply.rs:37,170-199,386]`. `fstr.rs` is the one place still
projecting only onto the float-feature space.

**Upstream algorithm (the target to match, verified against the PINNED oracle
version, not `master` HEAD).** Fetched directly from the `v1.2.10` git tag
(matching this project's pinned oracle floor `catboost==1.2.10` exactly)
`[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/{calc_fstr.cpp,feature_str.cpp}, accessed 2026-07-17]`:

- **`Interaction`** (`CalcFeatureInteraction`): each internally-accumulated
  `(FirstFeature, SecondFeature, Score)` triple has EACH side expanded to its
  list of external (original, flat) feature indices — a float/one-hot split
  expands to a 1-element list; an `OnlineCtr` split expands to the list of its
  projection's constituent original feature indices (`BinFeatures` → float,
  `CatFeatures` → categorical; `OneHotFeatures` is NOT included in this
  expansion, unlike PVC below). `Score` is distributed over the FULL
  CROSS-PRODUCT of `(f0 ∈ side0) × (f1 ∈ side1)`, each cell getting
  `Score / (side0.len() * side1.len())`; self-pairs (`f0 == f1`) are skipped;
  `(f0, f1)` is order-normalized before accumulating.
- **`PredictionValuesChange`** (`CalcRegularFeatureEffect`): an `OnlineCtr`
  split's effect is divided EQUALLY (no cross-product — one projection, not a
  pair) across `proj.BinFeatures.len() + proj.CatFeatures.len() +
  proj.OneHotFeatures.len()` and added into each constituent original feature's
  slot (`floatFeatureEffect[...]` / `catFeatureEffect[...]`).

**Codebase simplification (verified, not upstream-general).** This project's
`cb_train::TProjection` is **categorical-only** — it has no `BinFeatures` /
`OneHotFeatures` member at all
`[VERIFIED: LOCAL crates/cb-train/src/projection.rs:93-106 "Bin / one-hot
projection members ... are out of scope ... and are not modeled here"]`, and
`ModelSplit` has no separate one-hot variant (only `Float` and `Ctr`)
`[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:70-76]`. So for THIS
codebase, a CTR split's expansion/redistribution set is simply
`projection.cat_features()` — the `BinFeatures`/`OneHotFeatures` terms in the
upstream formulas above are vacuous here and are NOT implemented (there is
nothing to implement; a future float-in-CTR or one-hot-split feature would
need to extend this, out of scope).

## 2. Scope and non-goals

### In scope (this slice, per 2026-07-17 scoping decision)

- Extend `interaction()` so a model containing `ModelSplit::Ctr` splits (in
  either oblivious or non-symmetric trees) attributes CTR-split pairwise effect
  to the underlying original categorical feature(s), per the cross-product rule
  above.
- Extend `prediction_values_change()` so a `ModelSplit::Ctr` split's effect is
  redistributed equally across its projection's constituent categorical
  feature(s), per the equal-split rule above. **(Folded into this slice by
  explicit user decision — supersedes the research report's Open Question #1,
  which had flagged this as possibly deferrable.)**
- A new shared **combined flat feature-index** convention and resolver (§4) so
  both functions place categorical-feature contributions into the SAME output
  index space as float features, without requiring a new field on `Model`.
- New oracle fixture(s) with a trained model containing BOTH float and
  categorical features (at least one simple/single-feature CTR and one
  combination/tensor CTR), comparing both `interaction()` and
  `prediction_values_change()` output against upstream
  `get_feature_importance(type='Interaction'|'PredictionValuesChange')` at
  `≤1e-5`.
- `interaction()` and `prediction_values_change()`'s public signatures are
  UNCHANGED (`&Model -> Vec<(usize,usize,f64)>` / `&Model -> Vec<f64>`) — both
  stay dataset-free, additive changes only.

### Non-goals

- **`LossFunctionChange` + CTR (FSTR-02)** — a separate, larger slice: it is
  built on `shap_values`, which is itself CTR-blind and sizes its output vector
  to `model.float_feature_borders.len()` only
  `[VERIFIED: CODEGRAPH crates/cb-model/src/shap.rs:728,1096-1097]`. Untouched
  by this slice.
- **SHAP / SHAP-interaction CTR support** (`shap_values`,
  `shap_interaction_values`, `prediction_diff`, `sage_values`) — untouched,
  FSTR-02's dependency.
- **Upstream `.cbm` CTR model loading** (the unmerged `feat/23-ctr-model-loading`
  branch) — confirmed NOT required for this slice; the oracle fixture is built
  via `cb_train::train_cat` → `Model::from_trained`, the SAME pattern
  `tensor_ctr_e2e_oracle_test.rs` already uses, entirely independent of that
  branch `[VERIFIED: LOCAL git log/branch inspection, research.md Constraints]`.
- **Arbitrary original-column interleaving of float and categorical features.**
  The combined flat index this slice defines (§4) is verified correct ONLY
  when the oracle fixture places all float feature columns before all
  categorical feature columns in the original training data — see §4's
  explicit invariant. General support for arbitrary interleaving (matching
  upstream's `TFeaturesLayout::GetExternalFeatureIdx` for any column order) is
  **deferred**; this codebase does not track original column order at all
  today `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313 — no such
  field]`.
- One-hot cat splits and float-in-CTR (`BinFeatures`) — not modeled by this
  codebase's `TProjection` at all (see Context); not addressed here.
- Python / facade surfacing beyond what already exists. The Rust facade
  `catboost-rs::Model::feature_importance` already calls `interaction()` /
  `prediction_values_change()` for all models
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:139-149]` — no NEW
  facade wiring is required; this slice's contribution is making the existing
  call path CORRECT for CTR models. `catboost-rs-py` exposes no
  `get_feature_importance` surface at all yet (0 hits) — wiring it is a
  separate later DX task, per the FSTR-03 core/facade split precedent.

## 3. Dependencies

| Dependency | Typed interface | Evidence |
|-----------|-----------------|----------|
| `ModelSplit` / `CtrSplit` | `ModelSplit::{Float(Split), Ctr(CtrSplit)}`; `CtrSplit.projection: cb_train::TProjection` | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:43-76]` |
| Cat-feature membership | `TProjection::cat_features(&self) -> &[usize]` — sorted, de-duplicated LOCAL cat-feature indices (0-based among cat features only, NOT interleaved with float indices) | `[VERIFIED: CODEGRAPH crates/cb-train/src/projection.rs:102-132]` |
| Existing float-index resolver | `ModelSplit::float_feature(&self) -> Option<usize>` — `None` for `Ctr` | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:82-88]` |
| Existing float-feature width | `feature_count(model) -> usize` (private, `fstr.rs:80-100`) — `max(float split index) + 1` across BOTH tree kinds; a CTR split does not widen it | `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:75-100]` |
| Pair accumulator | `interaction_add(pairs: &mut Vec<(usize,usize)>, sums: &mut Vec<f64>, a: usize, b: usize, contribution: f64)` — insertion-order, deterministic (no `HashMap`) | `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:262-274]` |
| Deterministic float fold | `cb_core::sum_f64(&[f64]) -> f64` (D-08) | `[VERIFIED: CODEGRAPH crates/cb-core/src/reduction.rs:32]` |
| Oracle comparator | `cb_oracle::compare::assert_abs_close(expected, actual, tol) -> Result<(), OracleError>` | `[VERIFIED: CODEGRAPH crates/cb-oracle/src/compare.rs:46]` |
| Fixture-building pattern (no `.cbm` decode needed) | `cb_train::train_cat(...) -> cb_train::Model` then `cb_model::Model::from_trained(&trained, float_feature_borders).with_ctr_data(ctr_data)` | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:326-432; crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs]` |
| Existing fixture families (float+cat mix check) | `plain_ctr`, `ordered_ctr`, `tensor_ctr`, `tensor_ctr_e2e`, `one_hot_cat` — all CTR-value/approx isolation fixtures for prior ORD-03/04/05 requirements; **NONE dump `get_feature_importance` ground truth, and none mix float+categorical columns** (verified by reading each `config.json`: all are cat-only). A NEW fixture is required. | `[VERIFIED: LOCAL crates/cb-oracle/fixtures/{plain_ctr,ordered_ctr,tensor_ctr,tensor_ctr_e2e,one_hot_cat}/config.json]` |
| Upstream ground truth | `catboost==1.2.10` (pinned), offline `uv`-managed venv, same recipe as FSTR-03: `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'` | `[PROJECT: fstr-03-partial-dependence/PLAN.md T3; memory fstr03-partial-dependence-plan.md]` |

**Layering:** all work lives in `cb-model`, which already depends on
`cb-train` (`TProjection`) as a normal (non-dev) dependency
`[VERIFIED: LOCAL crates/cb-model/Cargo.toml:24]`; no `cb-backend`/CubeCL edge
is introduced (MODEL-02 boundary preserved).

## 4. Typed contracts

### Combined flat feature-index space (load-bearing — read first)

`cb_model::Model` exposes **no original-column-order feature map** — only
`float_feature_borders: Vec<Vec<f64>>` (float-local index) and
`ctr_data: Option<CtrData>`
`[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313]`. This slice does
**not** add such a field. Instead, both `interaction()` and
`prediction_values_change()` derive a **combined flat index** purely from what
appears in the model's own splits, mirroring the EXISTING `feature_count()`
convention (widen based on observed usage, not a stored config):

```text
n_float      = feature_count(model)                     // existing, unchanged
n_cat_used   = 1 + max over every ModelSplit::Ctr split in either tree kind
                    of ( max(projection.cat_features()) )
                 (0 if the model has no CTR splits)

flat_index(Float(f))          = f                 // f ∈ [0, n_float)
flat_index(Ctr, cat_local(c)) = n_float + c        // c ∈ [0, n_cat_used)
```

i.e. **float features occupy `[0, n_float)`; categorical features occupy
`[n_float, n_float + n_cat_used)`**, both counts derived additively from the
model's own splits (no `Model` struct change; consistent with `feature_count`'s
existing "no-flat-feature-map" constraint noted in the FSTR-03 SPEC §4).

**Explicit invariant this convention relies on (scoping limitation, §2
non-goal):** this flat index equals upstream's true `GetExternalFeatureIdx`
(original dataset column order) **only when the oracle fixture is constructed
so all float feature columns precede all categorical feature columns** in the
training data (i.e., `cat_features=[n_float, n_float+1, ...]` at train time).
The Planner/fixture generator MUST honor this when building `gen_fixtures.py`
for §7's new fixture, and the SPEC's acceptance tests are only meaningful
under this construction. `[INFERRED, per the FSTR-03 precedent of scoping the
index-space problem rather than solving general column interleaving; VERIFIED:
CODEGRAPH crates/cb-model/src/model.rs:272-313 confirms no alternative source
of original order exists]`

```rust
/// The number of DISTINCT categorical feature indices referenced by any
/// `ModelSplit::Ctr` split's projection, across both tree kinds — the
/// categorical analogue of the existing (private) `feature_count`. `0` if the
/// model has no CTR splits.
fn cat_feature_count(model: &Model) -> usize; // FIC-01

/// The combined flat index for a categorical feature's LOCAL index `c`
/// (as it appears in `TProjection::cat_features()`), given the model's float
/// width. Always `n_float + c` — see the load-bearing note above.
fn flat_cat_index(n_float: usize, local_cat_index: usize) -> usize; // FIC-01
```

`interaction()`'s and `prediction_values_change()`'s **public return types are
UNCHANGED** (`Vec<(usize, usize, f64)>` / `Vec<f64>`); the `usize`s now range
over `[0, n_float + n_cat_used)` instead of `[0, n_float)` whenever the model
has CTR splits. For a float-only model (`n_cat_used == 0`), behavior is
**byte-identical** to today (regression lock — see FIC-02/FIC-03 acceptance
criteria).

## 5. Failure-isolated behavioral specifications

Each specification below has one behavioral responsibility, one trigger, an
explicit dependency boundary, and one primary cause of acceptance-test failure.

---

### FIC-01 — Combined flat feature-index resolver

- **Status:** draft
- **Responsibility:** compute `n_cat_used` (the categorical analogue of
  `feature_count`) and the `flat_cat_index` mapping from a model's CTR splits
  alone. *Isolates index-space arithmetic from both importance algorithms that
  consume it.*
- **Preconditions:** none (must handle zero CTR splits, one CTR split, many).
- **Input:** `model: &Model`.
- **Output:** `usize` (`cat_feature_count`); `flat_cat_index` is a pure
  `(usize, usize) -> usize` function (`n_float + c`), not model-dependent.
- **Dependencies:** `ModelSplit::Ctr(CtrSplit)` pattern match,
  `CtrSplit.projection.cat_features()`
  `[VERIFIED: CODEGRAPH crates/cb-train/src/projection.rs:130-132]`.
- **Behavior (Given/When/Then):**
  - **Given** a model with no `ModelSplit::Ctr` splits in either tree kind,
    **then** `cat_feature_count == 0`.
  - **Given** a model whose only CTR split has `projection.cat_features() ==
    [2]`, **then** `cat_feature_count == 3` (`max + 1`).
  - **Given** a model with CTR splits across BOTH oblivious and non-symmetric
    trees, **then** `cat_feature_count` is the max across ALL of them (mirrors
    `feature_count`'s `oblivious_max.max(non_symmetric_max)` pattern).
  - **Given** a combination-CTR split with `projection.cat_features() == [0,
    3]`, **then** `cat_feature_count == 4` (max member `+1`, not member count).
- **Invariants:** pure, no allocation beyond the existing split iteration;
  `flat_cat_index(n_float, c) == n_float + c` always (no overflow guard needed
  beyond Rust's own `usize` arithmetic — `n_float`/`c` come from in-memory
  `Vec` lengths/indices, never attacker-controlled raw bytes at this layer).
- **Acceptance tests (unit):**
  - AT-FIC01a: no-CTR model → `0`.
  - AT-FIC01b: single simple-CTR split → `max_cat_index + 1`.
  - AT-FIC01c: combination-CTR split (2+ cat features) → count reflects the
    MAX member index, not `projection.cat_features().len()`.
  - AT-FIC01d: CTR splits in both oblivious and non-symmetric trees → overall
    max.
- **Out of scope:** float feature counting (existing `feature_count`,
  unchanged); the importance math itself.
- **Traceability:** `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:75-100
  feature_count precedent]`, `[VERIFIED: CODEGRAPH crates/cb-train/src/projection.rs:130-132]`.

---

### FIC-02 — `interaction()` CTR-aware pairwise attribution

- **Status:** draft
- **Responsibility:** extend `interaction()` so a tree-split pair involving one
  or two `ModelSplit::Ctr` splits contributes cross-product-distributed effect
  to the correct combined-flat-index pairs, in BOTH the oblivious and
  non-symmetric arms, without perturbing float-only output.
- **Preconditions:** FIC-01's `cat_feature_count`/`flat_cat_index` available.
- **Input:** `model: &Model` (unchanged signature).
- **Output:** `Vec<(usize, usize, f64)>` (unchanged type; index range widens
  per §4 when CTR splits are present).
- **Dependencies:** FIC-01; existing `interaction_add`; existing oblivious
  bit-indexed loop (`fstr.rs:292-329`) and non-symmetric DFS
  (`fstr.rs:369-468`) — **the delta/sign computation is UNCHANGED** (it is
  split-kind-agnostic, per research finding — only the "which flat index/indices
  does this split's contribution attribute to" step changes).
- **Behavior (Given/When/Then):**
  - **Given** a tree-split pair `(s0, s1)` where BOTH are `Float`, **then**
    behavior is BYTE-IDENTICAL to today (single-element expansion each side,
    same as the current `float_feature()` path) — regression lock.
  - **Given** a pair where `s0` is `Float(f)` and `s1` is `Ctr` with
    `projection.cat_features() == [c0, c1, ...]`, **then** the pair's computed
    `delta` (unchanged math) is distributed as
    `delta.abs() / (1 * cat_features.len())` into EVERY
    `(f, flat_cat_index(n_float, c_i))` pair (order-normalized, skipping any
    `f == flat_cat_index(..)` collision — structurally impossible here since
    ranges are disjoint, but keep the existing equality guard for defense).
  - **Given** a pair where BOTH `s0` and `s1` are `Ctr` (e.g. a combination CTR
    at one tree level paired with a simple CTR at another), **then** the
    pair's `delta` is distributed over the FULL cross product of
    `s0.cat_features() × s1.cat_features()`, each cell getting
    `delta.abs() / (side0.len() * side1.len())`, self-pairs
    (`flat_cat_index(..) == flat_cat_index(..)`, i.e. same original cat
    feature on both sides) skipped.
  - **Given** the SAME accumulation happening across MULTIPLE tree-split pairs
    that resolve to the same combined-flat-index pair (e.g. two different
    split-pairs both touching `(cat0, cat1)`), **then** contributions
    ACCUMULATE (via the existing `interaction_add`, unchanged) before the
    final `score = sum / total_effect * 100` normalization.
- **Invariants / side effects:** pure; every float fold via `cb_core::sum_f64`
  (D-08, unchanged — the per-pair `delta` computation is untouched, only the
  post-delta attribution step is new); the existing oblivious/non-symmetric
  loop bodies are NOT edited in place — the CTR expansion is added as
  additional logic around the existing `src1`/`src2` resolution step, per the
  "add a parallel arm, never inline into the middle" pitfall (research
  Pitfall 1).
- **Acceptance tests:**
  - AT-FIC02a (**regression, unit**): re-run the EXISTING
    `fstr_oracle_test.rs` items 2 and 4 (float-only `interaction` oracle
    assertions) UNCHANGED — must stay green, proving byte-identical float-only
    behavior.
  - AT-FIC02b (**unit**): a hand-built tiny oblivious `Model` with one
    `Float` split and one `Ctr` split (single cat feature) at the two split
    levels → asserts the resulting pair is `(float_idx, flat_cat_index(..),
    delta.abs())` (100% of total, single pair) — proves the basic float×cat
    cross-product with `side1.len()==1` (no division needed).
  - AT-FIC02c (**unit**): a hand-built tiny oblivious `Model` with a
    combination-CTR split (`cat_features() == [0, 1]`) at one level and a
    DIFFERENT plain `Float` split at another level → asserts the pair's
    `delta` is split EQUALLY (`/2`) into `(float_idx, flat_cat(0))` and
    `(float_idx, flat_cat(1))` — proves the cross-product division.
  - AT-FIC02d (**oracle**): new fixture (§7) with mixed float+categorical
    features (including at least one combination CTR) → `interaction()`
    output matches upstream `get_feature_importance(type='Interaction')`
    within `1e-5`, under the §4 float-columns-before-cat-columns fixture
    invariant. The fixture's loaded model MUST assert `>= 1` `CtrSplit` with
    `projection.cat_features().len() >= 2` (a combination CTR actually
    present), so a future fixture regeneration that accidentally loses the
    combination split fails this test loudly instead of silently degrading
    to float×simple-CTR-only coverage.
  - AT-FIC02e (**unit, non-symmetric arm — MANDATORY, not deferrable to the
    oracle test**): a hand-built `NonSymmetricTree` with TWO
    `ModelSplit::Ctr` splits at different node depths on the SAME
    root-to-leaf path, where the two projections PARTIALLY OVERLAP (e.g.
    `cat_features() == [0, 1]` at one depth and `== [1, 2]` at a deeper
    node on the same path) → asserts BY HAND COMPUTATION the exact resulting
    pair set: the cross-product cell where both sides resolve to the SAME
    flat cat index (`flat_cat(1)` vs `flat_cat(1)`) is skipped (self-pair),
    while the three non-colliding cells (`(flat_cat(0),flat_cat(1))`,
    `(flat_cat(0),flat_cat(2))`, `(flat_cat(1),flat_cat(2))`) each receive
    their `delta`-derived share, each divided by `2*2=4` (both sides have
    2 members). **Leaf-value construction requirement (plan-checker pass #3
    CRITICAL-1):** the two leaves reachable below the deeper split MUST have
    the SAME sign and unequal magnitude (e.g. `L=+3.0` at the deeper split's
    left child, `R=+1.0` at its right child) — an OPPOSITE-signed pair
    (e.g. `+3.0`/`-1.0`) makes the correct signed-accumulate-then-abs-once
    result (`|R-L|`) numerically COINCIDE with the buggy
    sign-dropping/abs-per-leaf-then-sum result (`|L|+|R|`), so it would NOT
    actually distinguish a correct implementation from the exact regression
    this test exists to catch (per `interaction_dfs`'s sign convention,
    `fstr.rs:459-466`: left child sign `-1`, right child sign `+1`). With
    `L=+3.0`, `R=+1.0`: correct `=|R-L|=2.0`, buggy `=|L|+|R|=4.0` —
    genuinely distinct. This is the ONLY test in the slice that exercises
    `interaction_dfs`'s path-pair cross-product with a `Vec<usize>`-valued
    path entry (as opposed to AT-FIC02b/c, which are both oblivious-arm-only)
    — required because the DFS `path` element type change
    (`Vec<(usize,i32)>` → `Vec<(Vec<usize>,i32)>`) is the single most
    structurally invasive change in this slice (plan-checker MAJOR-1
    finding), and the self-pair skip must operate PER CROSS-PRODUCT CELL,
    not once per path-entry-pair (a coarser check would silently drop valid
    non-colliding cells alongside the one true collision).
- **Out of scope:** `prediction_values_change` (FIC-03); the delta/sign math
  itself (unchanged, out of this spec's responsibility).
- **Traceability:** `[VERIFIED: WEB calc_fstr.cpp CalcFeatureInteraction,
  v1.2.10]`, `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:288-355,
  369-468]`.

---

### FIC-03 — `prediction_values_change()` CTR-aware redistribution

- **Status:** draft
- **Responsibility:** extend `prediction_values_change()` so a `ModelSplit::Ctr`
  split's per-node `dif` contribution (unchanged math) is redistributed
  equally across its projection's constituent categorical features' combined
  flat indices, in BOTH tree-kind arms, without perturbing float-only output.
- **Preconditions:** FIC-01 available; output vector width widens to `n_float +
  n_cat_used` (via `feature_count`-analogous sizing, see Impact below).
- **Input:** `model: &Model` (unchanged signature).
- **Output:** `Vec<f64>` (unchanged type; length widens per §4 when CTR splits
  present; still sums to 100 via unchanged `convert_to_percents`).
- **Dependencies:** FIC-01; existing `pvc_accumulate_oblivious` /
  `pvc_accumulate_non_symmetric` (`fstr.rs:143-174`, `198-260`) — the
  `count1`/`count2`/`avrg`/`dif` computation is UNCHANGED (split-kind-agnostic);
  only the "which slot(s) does `dif` add to" step changes for a `Ctr` split.
- **Behavior (Given/When/Then):**
  - **Given** a `Float` split, **then** behavior is BYTE-IDENTICAL to today
    (`dif` added to `res[src_idx]` alone) — regression lock.
  - **Given** a `Ctr` split with `projection.cat_features() == [c0, c1, ...]`
    (length `k`), **then** `dif / k` is added into `res[flat_cat_index(n_float,
    c_i)]` for EVERY `c_i` in the projection (equal-split redistribution, no
    cross-product — single projection, not a pair).
  - **Given** `k == 1` (simple/single-feature CTR), **then** the FULL `dif` is
    added to the single categorical feature's slot (no fractional loss).
- **Invariants / side effects:** the existing `count1 == 0.0 || count2 == 0.0`
  short-circuit (oblivious) and the `sum_count`/`denom` guard (non-symmetric)
  are UNCHANGED — the CTR redistribution only affects the "attribute `dif` to
  slot(s)" step, never the div-by-zero guards (T-04-04-03 preserved).
  `convert_to_percents`'s `total == 0.0` guard is unchanged. Every fold via
  `cb_core::sum_f64` where new summation is introduced (there is none new here
  beyond the existing scalar `+=`, since redistribution is per-split not a
  fold over a collection — no new `.sum()`/`fold` call is added).
- **Acceptance tests:**
  - AT-FIC03a (**regression, unit**): re-run the EXISTING
    `fstr_oracle_test.rs` items 1 and 4 (float-only `prediction_values_change`
    oracle assertions) UNCHANGED — must stay green.
  - AT-FIC03b (**unit**): a hand-built tiny oblivious `Model` with ONE `Ctr`
    split (simple, single cat feature) and non-zero leaf weights on both sides
    → asserts the FULL `dif` lands in `res[flat_cat_index(n_float, c)]`
    and all other slots are `0.0`, and the result sums to 100 after
    `convert_to_percents`.
  - AT-FIC03c (**unit**): a hand-built tiny `Model` with a combination-CTR
    split (`cat_features() == [0, 1]`) → asserts `dif/2` lands in EACH of
    `res[flat_cat_index(n_float,0)]` and `res[flat_cat_index(n_float,1)]`.
  - AT-FIC03d (**oracle**): the SAME new fixture as AT-FIC02d →
    `prediction_values_change()` output matches upstream
    `get_feature_importance(type='PredictionValuesChange')` within `1e-5`.
- **Out of scope:** `interaction` (FIC-02); the `dif`/`avrg` math itself
  (unchanged).
- **Traceability:** `[VERIFIED: WEB calc_fstr.cpp CalcRegularFeatureEffect,
  v1.2.10, full function quoted in research.md Addendum]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:116-138, 143-260]`.

## 6. Acceptance scenarios (roll-up)

| Scenario | Spec | Kind | Oracle artifact | Bar |
|----------|------|------|-----------------|-----|
| `cat_feature_count`/`flat_cat_index` arithmetic (no-CTR, simple, combination, both tree kinds) | FIC-01 | unit | — | exact |
| Existing float-only `interaction` assertions unchanged | FIC-02 | unit (regression) | `feature_importance/interaction.npy` (existing) | ≤1e-5, unchanged |
| Hand-built float×simple-CTR pair attribution | FIC-02 | unit | — | exact |
| Hand-built combination-CTR cross-product division | FIC-02 | unit | — | exact |
| Hand-built non-symmetric DFS: two Ctr splits, partial overlap self-pair skip (AT-FIC02e) | FIC-02 | unit | — | exact |
| `interaction` on mixed float+CTR fixture == upstream, combination-CTR presence asserted | FIC-02 | oracle | new fixture (§7) | ≤1e-5 |
| Existing float-only `prediction_values_change` assertions unchanged | FIC-03 | unit (regression) | `feature_importance/prediction_values_change.npy` (existing) | ≤1e-5, unchanged |
| Hand-built simple-CTR full redistribution | FIC-03 | unit | — | exact |
| Hand-built combination-CTR equal-split redistribution | FIC-03 | unit | — | exact |
| `prediction_values_change` on mixed float+CTR fixture == upstream | FIC-03 | oracle | new fixture (§7) | ≤1e-5 |

## 7. Impact scope

- **Classification:** `local` (single crate `cb-model`). `[VERIFIED: CODEGRAPH deps]`
- **Must change:** `crates/cb-model/src/fstr.rs` — `feature_count()` widens (or
  a sibling helper is introduced) to include `n_cat_used`; `interaction()` and
  `prediction_values_change()` (and their private accumulation helpers) gain
  CTR-aware attribution steps. New private helpers for FIC-01
  (`cat_feature_count`, `flat_cat_index`) live in the same module.
- **New test file:** `crates/cb-model/src/fstr_test.rs` (sibling unit-test
  file — confirmed NO existing `fstr_test.rs` today
  `[VERIFIED: LOCAL find crates/cb-model/src -iname 'fstr_test.rs' → no result]`),
  mounted via `#[cfg(test)] #[path = "fstr_test.rs"] mod tests;` in `fstr.rs`
  per the mandatory source/test separation rule
  `[VERIFIED: LOCAL CLAUDE.md "Source/Test Separation"; precedent
  crates/cb-model/src/ctr_data.rs:58-61]`.
- **New oracle fixture:** `crates/cb-oracle/fixtures/fstr_ctr/` (name a
  plan-time choice) — `gen_fixtures.py`, `config.json`, a trained model with
  float features BEFORE all categorical features (§4 invariant), including at
  least one simple and one combination CTR, plus committed
  `interaction.npy` (flattened `[feature_i, feature_j, score]` triples — the
  SAME format the existing `fstr_oracle_test.rs` fixtures already use, per
  its own docstring; not a new/undecided convention) /
  `prediction_values_change.npy` ground truth. **Hard gate (not a soft note):**
  the generated model MUST contain at least one combination CTR split
  (`projection.cat_features().len() >= 2`) — verified both at generation time
  (T4) AND re-asserted directly in the Rust oracle test (T5/T6) so a future
  fixture regeneration that accidentally loses the combination split fails
  loudly instead of silently degrading test coverage to float×simple-CTR-only.
- **New oracle test:** `crates/cb-model/tests/fstr_ctr_oracle_test.rs` (new
  file, or an addition to the existing `fstr_oracle_test.rs` — plan-time
  choice; the existing file's own docstring already enumerates 4 numbered
  scenarios, so a 5th/6th could extend it, or a dedicated CTR-focused file
  keeps concerns separate — Planner's call).
- **Verification only (read, not modified):** `crates/cb-model/src/apply.rs`,
  `crates/cb-model/src/model.rs`, `crates/cb-model/src/ctr_data.rs`,
  `crates/cb-train/src/projection.rs`,
  `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` (pattern source),
  `crates/catboost-rs/src/model.rs::feature_importance` (must keep compiling
  against the signature-preserving change; not expected to need edits).
- **Explicitly out of scope:** `crates/cb-model/src/shap.rs` (all of it),
  `feat/23-ctr-model-loading` branch content, `catboost-rs-py`.
- **Tests:** existing `fstr_oracle_test.rs` / `advanced_fstr_oracle_test.rs`
  float-only assertions must remain green UNCHANGED (regression gate); no
  shipped fixture's ground truth changes.
- **Build/operational:** none new; validates under
  `cargo test -p cb-model -p cb-oracle` and `cargo test -p cb-train` (confirm
  `tensor_ctr_e2e_oracle_test.rs` / `multi_permutation_e2e_oracle_test.rs`
  unaffected, since this slice reuses but does not modify their pattern).
  **Restriction-lint gate:** `cargo clippy -p cb-model --all-targets` (NOT
  `cargo build`, which does not enforce `unwrap_used`/`expect_used`/`panic`/
  `indexing_slicing` — a documented gotcha from the FSTR-03 plan check)
  `[PROJECT: fstr-03-partial-dependence/PLAN-CHECK.md MAJOR #2]`.

## 8. Compatibility and migration

Additive only for float-only models (`n_cat_used == 0` ⇒ byte-identical
output, regression-locked by AT-FIC02a/AT-FIC03a). For models WITH CTR splits,
this is a **behavior fix**, not a breaking API change: the public function
signatures and return TYPES are unchanged, but the returned index range widens
and previously-silent CTR contributions now appear. No serialization format,
no `Model` struct field, no existing fixture is touched. `[INFERRED]`

**Caller-visible semantic note (non-breaking but worth flagging):** any
downstream caller of `interaction()`/`prediction_values_change()` that
previously assumed "every index is a float-feature index" (true when
`n_cat_used == 0`) must now know the range widens for CTR-bearing models. The
Rust facade `catboost-rs::Model::feature_importance` calls these functions
directly and needs no code change, but its documentation should be checked for
this assumption once this slice lands (flagged, not required by this SPEC —
see §2 non-goals on facade/Python surfacing).

## 9. Risks and open questions

1. **[RESOLVED] Exact upstream CTR attribution rule for BOTH `Interaction`
   and `PredictionValuesChange`**, verified against the pinned `v1.2.10` tag
   specifically (not `master` HEAD, avoiding the version-drift risk the
   research report flagged as LOW confidence). See §1 and the research
   report's Addendum for full verbatim C++.
2. **[RESOLVED] Scope: does this slice include PVC-CTR?** Yes — user-confirmed
   2026-07-17, superseding the research report's Open Question #1.
3. **[OPEN, fixture-design, non-blocking for architecture] Which existing
   fixture (if any) can be extended vs. a wholly new one.** No existing
   fixture mixes float + categorical columns (verified: all of `plain_ctr`,
   `ordered_ctr`, `tensor_ctr`, `tensor_ctr_e2e`, `one_hot_cat` are
   categorical-only). §7 requires a NEW fixture; the Planner should decide the
   exact `gen_fixtures.py` recipe (feature counts, cardinalities, whether to
   reuse `tensor_ctr`'s cardinality/param choices with float columns added).
4. **[OPEN, minor] Whether `feature_count()` should be widened in place, or a
   new combined-width helper introduced alongside it.** Either satisfies FIC-01
   through FIC-03; a plan-time implementation choice, not a behavioral
   difference (the SPEC's Given/When/Then does not depend on which).
5. **[INFERRED] `catboost` not installed in this dev container** — the new
   fixture's ground truth must be generated offline via the same `uv`-managed
   venv recipe FSTR-03 used (`uv venv --python 3.12 && uv pip install
   catboost==1.2.10 'numpy<2'`). `[PROJECT: fstr03-partial-dependence-plan.md]`
6. **[INFERRED] `Model` widening via `n_cat_used` derived from splits alone
   (not a stored field) means a model with a CTR split whose baked
   `ctr_data` table exists but is never actually reached by any split in the
   trees it ships (a pathological/hand-crafted case) would NOT widen the
   output.** This mirrors the existing `feature_count()` semantics exactly
   (also split-derived, not `ctr_data`-derived) and is considered correct
   behavior, not a gap — `[INFERRED, consistent with existing precedent]`.
7. **[OPEN, per plan-checker review 2026-07-17] The verbatim `v1.2.10` C++
   quoted in research.md's Addendum (`CalcFeatureInteraction`,
   `CalcRegularFeatureEffect`) was obtained via an agent-driven `WebFetch`
   summarization, not read character-by-character by a human, and the
   independent Plan Checker pass had no web-fetch capability to re-verify it.
   This is the single highest-risk claim in this SPEC — the entire
   attribution algorithm rests on it.** Carried as an OPEN risk, not settled
   fact, until AT-FIC02d/AT-FIC03d's oracle comparison against real
   `catboost==1.2.10` output empirically confirms or refutes it. If the
   oracle comparison fails and the fixture/model-loading sanity gates (T5/T6
   predictions-first check) pass, the FIRST hypothesis to revisit is that
   this quoted algorithm is subtly wrong (e.g., an omitted normalization
   step, a different tie-break, or a wrapper-level reordering — see risk 9),
   not that the Rust translation of it is wrong.
8. **[RESOLVED, per plan-checker pass #2 (2026-07-17) — deliberate, recorded
   scope decision, not an oversight] The new oracle fixture (§7) trains ONLY
   oblivious (symmetric) trees.** `cb_train::grow_policy_default()` is
   `EGrowPolicy::SymmetricTree`
   `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:128-135]`, and T4's
   fixture adapts `tensor_ctr_e2e`'s param set (which uses this default), so
   AT-FIC02d/AT-FIC03d's oracle comparison exercises the OBLIVIOUS arm's CTR
   logic only. **The non-symmetric arm's CTR logic has NO oracle-level
   backstop in this slice** — verification rests entirely on PLAN.md T2's
   hand-built, discriminating unit test (AT-FIC02e, specifically
   strengthened by plan-checker pass #2 to catch a sign-dropping/
   premature-`abs()` regression) for FIC-02, and T3's oblivious-only unit
   tests for FIC-03 (lower risk, since PVC's `dif` redistribution has no
   sign-cancellation concern). Building a second `grow_policy=Lossguide`
   fixture purely to oracle-cover this path is explicitly deferred as
   disproportionate scope growth — a candidate follow-up hardening task, not
   a blocker for this SPEC's acceptance criteria.
9. **[OPEN, per plan-checker review 2026-07-17] Whether upstream's
   Python-facing `get_feature_importance(type='PredictionValuesChange')`
   array is genuinely ordered by original/external flat feature index, versus
   reflecting `CalcRegularFeatureEffect`'s own internal `Sort(...)` by score
   (which research.md's Addendum quotes as part of that C++ function's own
   return value) is INFERRED, not cited to a specific wrapper source.** This
   SPEC assumes a Python-facing layer reorders the C++ function's
   score-sorted `TVector<TFeatureEffect>` back into original-feature-index
   order before it reaches `get_feature_importance` (consistent with every
   OTHER importance type in this project already being consumed in
   feature-index order, e.g. the existing float-only oracle fixtures) — but
   this specific reordering step's source was not cited. AT-FIC03d's oracle
   comparison is the actual gate that proves or disproves this ordering
   assumption; if AT-FIC03d fails on ordering alone (values present but
   permuted), re-sort the fixture's ground truth by feature index at
   generation time as the fix, rather than treating it as a Rust-side bug.

## 10. Traceability and sources

- **Requirement:** git-recovered `.planning/REQUIREMENTS.md`/`ROADMAP.md`
  (commit `a82289c`) — not in the working tree; re-verify canonical revision
  before flipping any requirement checkbox.
- **Research report:** `.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/research.md`
  (this SPEC's evidence base; its Addendum holds the full verbatim v1.2.10
  C++ this SPEC's algorithm section summarizes).
- **Upstream behavior (pinned tag, not `master`):**
  `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/calc_fstr.cpp
  CalcFeatureInteraction, CalcRegularFeatureEffect]`,
  `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/feature_str.cpp
  CalcMostInteractingFeatures featureToIdx overload]`.
- **Rust seams:** `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs (full file)]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:43-98,272-313]`,
  `[VERIFIED: CODEGRAPH crates/cb-train/src/projection.rs:93-187]`.
- **Fixture-family absence proof:** `[VERIFIED: LOCAL
  crates/cb-oracle/fixtures/{plain_ctr,ordered_ctr,tensor_ctr,tensor_ctr_e2e,one_hot_cat}/config.json
  — all categorical-only, none dump fstr ground truth]`.
- **Constraints:** `[VERIFIED: LOCAL Cargo.toml:10-14 restriction lints]`,
  `[VERIFIED: LOCAL CLAUDE.md source/test separation + no-unwrap]`.
- **PageIndex:** the `catboost_rs` folder is indexed (folder id
  `cmrhcxbtm000104jr3i5jzm0m`) and currently holds only the FSTR-03 `SPEC.md`
  `[VERIFIED: PAGEINDEX browse_documents(folder_id=cmrhcxbtm000104jr3i5jzm0m) →
  1 document, "SPEC.md" (FSTR-03), status=completed]`. This SPEC is **not yet
  indexed** — `process_document` has no in-place Markdown upsert / doc-id
  overwrite (file/URL ingestion only), so indexing this document requires the
  human owner to add it as a second document in the same folder out-of-band.
  No indexing action was taken by the planner to avoid creating an
  unreconciled duplicate.
