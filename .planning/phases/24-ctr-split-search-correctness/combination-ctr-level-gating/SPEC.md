---
title: "ORD-06 — Tree-Structure-Scoped Combination-CTR Candidate Eligibility"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-18T00:00:00Z
phase: 24
requirement_ids:
  - ORD-06
source_requirements:
  - "Discovered as a side effect of FSTR-01 (.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/), not a pre-existing tracked requirement. Extends the ORD-0x lineage (ORD-01 permutation, ORD-02 ordered boosting, ORD-03 one-hot threshold, ORD-04 one-hot routing, ORD-05 tensor/combination CTR search) that this bug lives inside and corrects."
research_report: ".planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/research.md"
pageindex_target: "catboost_rs folder (id cmrhcxbtm000104jr3i5jzm0m). Currently holds only FSTR-03's SPEC.md. Not yet indexed — process_document has no in-place Markdown upsert; human owner should add this as a new document. See §10."
---

# ORD-06 — Tree-Structure-Scoped Combination-CTR Candidate Eligibility

> **Draft.** Not approved / not implemented. This spec decomposes ORD-06 into
> failure-isolated behavioral specifications for TDD (see `PLAN.md`). No
> production code is authored by this document.

## 1. Context

While implementing FSTR-01, a new oracle fixture mixing float and categorical
features (`crates/cb-oracle/fixtures/fstr_ctr/`) exposed a genuine
training-time divergence: `cb_train::train_cat` produces a **different tree
structure** than real `catboost==1.2.10` for identical data/params/seed —
diverging at the very root split. This is confirmed by direct reproduction
(`cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture` fails
even at the model-PREDICTION sanity gate, not just at FSTR-01's
interaction/PVC comparison) `[VERIFIED: LOCAL research.md "directly
reproduced the RED test"]`.

**Root cause (CONFIRMED, not the executor's original hypothesis).** The
scoring math itself (`cat_feature_weight`, `build_ctr_aware_histogram`,
`select_level_ctr_aware` — `crates/cb-train/src/tree.rs:2416,2306,2527`)
is byte-for-byte consistent with upstream's `GetCatFeatureWeight`
(`greedy_tensor_search.cpp:926-950`)
`[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2416; WEB
github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/greedy_tensor_search.cpp:926-950]`.
The actual defect is in **candidate-set generation**, upstream of scoring:
`tensor_ctr_candidates` (`crates/cb-train/src/candidates.rs:179`) is called
**once per tree, outside any level loop** (`crates/cb-train/src/boosting.rs:2705`),
enumerating ALL simple + combination CTR projections from cat-feature
cardinalities alone — with **no notion of which splits the tree has already
chosen**. Its full static result is fed identically into
`greedy_tensor_search_oblivious_with_ctr` (`crates/cb-train/src/tree.rs:2669`)
at **every** level, including the root
`[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:159-201;
crates/cb-train/src/boosting.rs:2680-2930]`.

**Upstream's actual rule (verified against the pinned `v1.2.10` tag).**
`AddTreeCtrs` (`greedy_tensor_search.cpp:503-568`) only makes a
**combination** (multi-feature) CTR eligible once the tree already has at
least one split chosen to extend. At the root (`currentTree` empty),
`binAndOneHotFeaturesTree` is empty and the only `seenProj` entry
(`baseProj.IsEmpty()`) is explicitly **skipped** — zero combination
candidates are ever offered at the root
`[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/greedy_tensor_search.cpp:503-568,
verbatim quote in research.md]`. `AddSimpleCtrs` (single-feature CTRs), by
contrast, is called **unconditionally at every level** regardless of tree
structure — simple CTRs are NOT part of this bug
`[VERIFIED: WEB same file, SelectDatasetFeaturesForScoring call order,
lines ~860-871]`.

**Architecture correction (found while preparing this SPEC's typed contracts
— narrows the fix considerably from research.md's Recommended Architecture).**
Reading `select_level_ctr_aware` (`crates/cb-train/src/tree.rs:2527-2632`) in
full shows `ctr_features: &[CtrFeatureColumn]` is the tree-wide,
ALREADY-MATERIALIZED static column list (unchanged by this fix — no
materialization-timing change is needed), and the function ALREADY computes
`used_projections: Vec<&TProjection>` from `chosen` (`tree.rs:2578-2586`,
today used ONLY to exempt an already-used projection from the
`cat_feature_weight` penalty). **This existing `used_projections` value IS
exactly the "seen CTR base projections" set this fix needs** — no new
plumbing from `boosting.rs`/`candidates.rs` is required. The correct,
minimal fix is an ELIGIBILITY FILTER added to the existing "CTR candidates
next" loop (`tree.rs:2589-2610`, `for col in 0..ctr_features.len()`): for a
column whose `projection.is_combination()`, additionally require that some
member of `used_projections` is that projection with exactly one member
removed (a legitimate one-feature extension); if not, `continue` — skip
scoring this column at this level entirely, exactly as if it were not
materialized. **`tensor_ctr_candidates` (`candidates.rs`) and
`boosting.rs`'s up-front materialization are UNCHANGED** — they may keep
generating/materializing every combinatorially-valid column once per tree,
tree-wide, exactly as today; the bug and its fix live entirely in
`select_level_ctr_aware`'s per-level SCORING loop, which decides which
already-materialized columns are considered THIS level. (research.md's
Recommended Architecture proposed changing `candidates.rs`'s enumeration and
`boosting.rs`'s materialization timing — a viable but unnecessarily invasive
alternative; this SPEC adopts the smaller, more surgical fix, since it
achieves the identical observable behavior with a much smaller blast
radius.)

**Codebase-specific simplification (resolves the research's one open
question — no further spike needed).** Upstream's `seenProj` base set is
`{binAndOneHotFeaturesTree}` (built from `TSplitTree::GetBinFeatures()` +
`GetOneHotFeatures()` — the tree's chosen FLOAT and one-hot splits) `∪`
`{p.Projection for p in currentTree.GetUsedCtrs()}` (the tree's chosen CTR
splits) `[VERIFIED: WEB v1.2.10 split.h:469-494, quoted verbatim: GetBinFeatures
filters ESplitType::FloatFeature, GetOneHotFeatures filters
ESplitType::OneHotFeature, GetUsedCtrs filters ESplitType::OnlineCtr]`.
Extending `binAndOneHotFeaturesTree` by one more cat feature ALWAYS yields a
**mixed** float-bin+cat (or one-hot+cat) projection, since `AddCatFeature`
appends to the existing base without removing its non-cat members
`[VERIFIED: WEB v1.2.10 projection.h:57-59,109-111, quoted verbatim: TProjection
has separate CatFeatures/BinFeatures/OneHotFeatures fields; AddCatFeature only
pushes onto CatFeatures]`. **`cb_train::TProjection` is deliberately
categorical-only — it has no `BinFeatures`/`OneHotFeatures` member at all**,
a PRE-EXISTING design decision unrelated to this bug
`[VERIFIED: LOCAL crates/cb-train/src/projection.rs:93-106, "Bin / one-hot
projection members... are out of scope... and are not modeled here"]`. So
`binAndOneHotFeaturesTree`-derived candidates (mixed float+cat) are
**structurally unrepresentable** in this codebase regardless of this bug, and
were already out of scope before this bug was discovered. **Therefore the
correct, scope-appropriate port for THIS codebase is: a combination CTR is
eligible starting from a base projection ONLY when it extends an
ALREADY-CHOSEN `ModelSplit`/`AnySplit::Ctr` split's projection** (the
`GetUsedCtrs()` half of upstream's `seenProj` — the ONLY half whose result
type, pure-categorical, this codebase's `TProjection` can represent) — **a
tree that has chosen ONLY float splits so far (no CTR yet) offers ZERO
combination-CTR candidates**, exactly matching upstream's real behavior for
the subset of projections this codebase's `TProjection` is capable of
producing (upstream would only unlock a MIXED projection there, which this
codebase cannot and need not represent).

## 2. Scope and non-goals

### In scope

- Make **combination** (multi-feature, `is_combination()`/`cat_features().len()
  >= 2`) CTR candidate generation **level- and tree-structure-aware**: at each
  level, a combination candidate is eligible only if it extends the
  projection of an **already-chosen** `Ctr` split (simple or combination)
  earlier in the SAME tree, by exactly one more CTR-eligible cat feature not
  already a member, capped by `max_ctr_complexity`, deduplicated within the
  level.
- At level 0 (no splits chosen yet) and at any level where the tree has
  chosen ONLY `Float` splits so far (zero `Ctr` splits chosen), combination
  candidates are **empty**.
- Preserve **simple** (single-feature) CTR candidates as unconditionally
  available at every level, unchanged (matches `AddSimpleCtrs`; confirmed NOT
  part of this bug).
- Preserve the existing scoring formulas (`cat_feature_weight`,
  `build_ctr_aware_histogram`, `select_level_ctr_aware`'s score comparison,
  `l2_split_score`/`cosine_split_score`) **byte-identical** — this is a
  candidate-set-generation fix, not a scoring fix.
- Preserve the strict first-wins (`> best`, never `>=`) tie-break discipline
  already established in this module.
- Re-verify (not just re-run) `tensor_ctr_e2e_oracle_test.rs`,
  `multi_permutation_e2e_oracle_test.rs`, and `ctr_split_scoring_test.rs`
  after the fix.

### Non-goals

- **Mixed float+cat (or one-hot+cat) combination CTR projections** — upstream
  supports these (`binAndOneHotFeaturesTree`-derived candidates); this
  codebase's `TProjection` structurally does not and this is NOT changed by
  this fix (pre-existing, unrelated scope boundary — see §1).
- **`Rsm` (random subspace method) feature sampling** — upstream's
  `AddTreeCtrs` inner loop includes a
  `Rand.GenRandReal1() > ObliviousTreeOptions->Rsm` gate; this codebase's
  existing fixtures/params use the default `Rsm=1.0` (always-include), and
  no evidence was found that this codebase implements or exposes `Rsm` at
  all — treating it as out of scope; if a future `Rsm < 1.0` config is added,
  it is a SEPARATE requirement.
- **One-hot candidate generation** (`route_categorical`/`EncodingPath::OneHot`)
  — confirmed NOT tree-structure-dependent upstream (`AddOneHotFeatures` runs
  unconditionally every level); not touched.
- **Ordered boosting + CTR interaction** (`boosting.rs`'s `has_ctr` branch
  precedence relative to `ordered_learning_perm`) — a separate, pre-existing
  design question, not part of this bugfix's diagnosis or fix. Must not be
  accidentally altered while touching the same branch (see §9 risk).
- **FSTR-01's `interaction()`/`prediction_values_change()`** — unaffected;
  they fail only because the model fed to them is wrong. This fix, once
  landed, is what unblocks FSTR-01's T5/T6 (out of this SPEC's own acceptance
  criteria, but the shared oracle test is the integration proof for both).
- **Regenerating any existing fixture** — `crates/cb-oracle/fixtures/fstr_ctr/`
  and all other CTR fixtures are FROZEN (upstream quantization is
  run-to-run nondeterministic, a documented project convention); this fix
  must make the ALREADY-COMMITTED fixture pass, never regenerate it.

## 3. Dependencies

| Dependency | Typed interface | Evidence |
|-----------|-----------------|----------|
| Combination-CTR static enumerator (current, buggy) | `tensor_ctr_candidates(...) -> Vec<TProjection>` (or similar) — enumerates ALL simple+combination projections from cardinalities alone | `[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:179]` |
| CTR-aware level search | `select_level_ctr_aware(..., chosen: &[CtrAwareSplit], ...)` — ALREADY threads the tree's chosen-so-far splits through | `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2527]` |
| CTR-aware tree search driver | `greedy_tensor_search_oblivious_with_ctr(...)` — the `depth`-level loop | `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2669]` |
| Orchestration / materialization | `train_cat` → `train_inner` → (candidate gen once) → (level loop) — `crates/cb-train/src/boosting.rs:2145,2259,2705,2843-2929,3892-3916` | `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs]` |
| Projection type | `TProjection::{cat_features(), from_features(), with_added(), is_simple(), is_combination(), full_projection_length()}` — categorical-only, already sorted/deduped | `[VERIFIED: CODEGRAPH crates/cb-train/src/projection.rs:99-187]` |
| CTR value materialization | `materialize_ctr_feature` / `CtrFeatureColumn` — unchanged contract, called at new timing/granularity | `[VERIFIED: CODEGRAPH crates/cb-train/src/ctr/ctr_feature.rs (per research.md)]` |
| Scoring (confirmed correct, reused unchanged) | `cat_feature_weight` (`tree.rs:2416`), `l2_split_score`/`cosine_split_score` (`cb-compute/src/score.rs:49,73`) | `[VERIFIED: CODEGRAPH + WEB v1.2.10 greedy_tensor_search.cpp:926-950]` |
| Upstream reference algorithm | `AddTreeCtrs` (`greedy_tensor_search.cpp:503-568`), `SelectDatasetFeaturesForScoring`'s call order (`~838-902`) | `[VERIFIED: WEB v1.2.10 tag, verbatim quotes in this SPEC/research.md]` |
| Upstream `TSplitTree`/`TProjection` semantics | `GetBinFeatures`/`GetOneHotFeatures`/`GetUsedCtrs` (`split.h:469-494`), `IsRedundant`/`AddCatFeature`/`GetFullProjectionLength` (`projection.h:57-130`) | `[VERIFIED: WEB v1.2.10 tag, verbatim quotes — resolves research.md's one open blocker]` |
| Oracle fixture (frozen, already committed) | `crates/cb-oracle/fixtures/fstr_ctr/{model.cbm,model.json,X_float.npy,X_cat.npy,y.npy,interaction.npy,prediction_values_change.npy,predictions.npy,config.json}` | `[VERIFIED: LOCAL, generated by a prior session via real catboost==1.2.10]` |
| Oracle test (currently RED) | `crates/cb-model/tests/fstr_ctr_oracle_test.rs` — 3 tests: `fstr_ctr_predictions_sanity_gate`, `interaction_matches_upstream_on_mixed_ctr_model`, `pvc_matches_upstream_on_mixed_ctr_model` | `[VERIFIED: LOCAL, directly re-run, all 3 FAIL, sanity gate fails first]` |
| Existing CTR regression suites | `tensor_ctr_e2e_oracle_test.rs`, `multi_permutation_e2e_oracle_test.rs`, `ctr_split_scoring_test.rs` (all currently GREEN, categorical-only, `depth=2, max_ctr_complexity=2, boosting_type=Plain`) | `[VERIFIED: LOCAL, grep'd params, all Plain, none Ordered+CTR]` |

**Layering:** all work lives in `cb-train` (`candidates.rs`, `tree.rs`,
`boosting.rs`); no new crate dependency; `cb-compute`'s scoring functions are
consumed unchanged.

## 4. Typed contracts

### The eligibility predicate (load-bearing — read first)

`select_level_ctr_aware` (`crates/cb-train/src/tree.rs:2527-2632`) ALREADY
computes, once per level:

```rust
let used_projections: Vec<&crate::TProjection> = chosen
    .iter()
    .filter_map(|s| match s {
        CtrAwareSplit::Ctr { col, .. } => ctr_features.get(*col).map(|c| &c.projection),
        CtrAwareSplit::Float(_) => None,
    })
    .collect();
```

`[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2578-2586]` — this IS the
`seen_ctr_bases` set from §1's discussion; no new derivation of "already
chosen CTR projections" is needed. The fix is a NEW pure predicate, plus one
new `continue` guard where the existing "CTR candidates next" loop
(`tree.rs:2589-2610`, `for col in 0..ctr_features.len()`) currently scores
every materialized column unconditionally:

```rust
/// Whether a combination (`>= 2`-feature) CTR projection is eligible to be
/// scored at the CURRENT level, given the tree's already-chosen CTR
/// projections. Mirrors upstream `AddTreeCtrs`'s `seenProj`/`baseProj.IsEmpty()`
/// gate (`greedy_tensor_search.cpp:503-568`), restricted to this codebase's
/// categorical-only `TProjection` (§1 "Codebase-specific simplification" —
/// only `GetUsedCtrs()`-derived bases are representable/relevant here).
///
/// A SIMPLE (single-feature) projection is ALWAYS eligible — this predicate
/// is never called for one (see ORD-06-03's call-site guard, `is_combination()`
/// first). A COMBINATION projection `p` is eligible iff `used_projections`
/// contains some projection `q` such that `q`'s cat-feature set is a SUBSET of
/// `p`'s with exactly one FEWER member (`p` is `q` extended by exactly one
/// feature) — mirroring `AddTreeCtrs`'s `proj.AddCatFeature(...)` extension.
#[must_use]
fn combination_ctr_eligible(
    projection: &TProjection,
    used_projections: &[&TProjection],
) -> bool {
    let members = projection.cat_features();
    used_projections.iter().any(|q| {
        let q_members = q.cat_features();
        q_members.len() + 1 == members.len()
            && q_members.iter().all(|m| members.contains(m))
    })
}
```

`combination_ctr_eligible` returns `false` for EVERY combination projection
whenever `used_projections` is empty (whether `chosen` is entirely empty, OR
contains only `Float` splits) — this exactly mirrors upstream's
`baseProj.IsEmpty()` root-level skip, generalized correctly to "no `Ctr`
split chosen yet" for this codebase's categorical-only projections (§1).

**Call-site change (`tree.rs`'s existing "CTR candidates next" loop,
`ORD-06-03`):**

```rust
for col in 0..ctr_features.len() {
    let Some(column) = ctr_features.get(col) else { continue };
    if column.projection.is_combination()
        && !combination_ctr_eligible(&column.projection, &used_projections)
    {
        continue; // NEW: skip an illegitimate-at-this-level combination candidate.
    }
    // ... existing cat_weight / score / push(scored) logic, UNCHANGED ...
}
```

**No change to `candidates.rs`'s `tensor_ctr_candidates`, and no change to
`boosting.rs`'s materialization timing** — both continue to generate/materialize
every combinatorially-valid column once per tree exactly as today (§1
Architecture correction). `max_ctr_complexity` capping and cross-column
deduplication are ALREADY handled once, correctly, by the EXISTING
`tensor_ctr_candidates`/`enumerate_projections` — this fix does not need to
re-derive either.

## 5. Failure-isolated behavioral specifications

---

### ORD-06-01 — `combination_ctr_eligible` is false whenever no `Ctr` split has been chosen yet

- **Status:** draft
- **Responsibility:** the root-level (and any float-only-so-far level) gate.
  *Isolates the "is a combination CTR eligible AT ALL yet" question from the
  extension-membership check.*
- **Preconditions:** none (must handle `used_projections` empty and
  non-empty).
- **Input:** `projection: &TProjection` (a combination, `cat_features().len()
  >= 2`), `used_projections: &[&TProjection]`.
- **Output:** `bool`.
- **Dependencies:** `TProjection::cat_features()` (existing).
- **Behavior (Given/When/Then):**
  - **Given** `used_projections` is empty (tree root, OR a tree so far
    containing only `Float` splits — §1's codebase-specific simplification:
    a float-only history yields no `Ctr`-derived entries in
    `used_projections` at all), **then** `combination_ctr_eligible` returns
    `false` for ANY combination `projection` (matches upstream's
    `baseProj.IsEmpty()` skip, generalized correctly for this codebase's
    categorical-only projections).
  - **Given** `used_projections` is non-empty, **then** the result depends on
    ORD-06-02's extension-membership check.
- **Invariants:** pure function; no I/O; deterministic given the same inputs.
- **Acceptance tests (unit):**
  - AT-ORD06-01a: `used_projections = []`, `projection = {0,1}` → `false`.
  - AT-ORD06-01b: `used_projections = []` (representing a tree with only
    `Float` splits chosen, i.e. zero `Ctr` entries reached this filter-map),
    `projection = {2,3}` → `false`.
- **Out of scope:** the extension-membership arithmetic itself (ORD-06-02);
  wiring into `select_level_ctr_aware`'s loop (ORD-06-03).
- **Traceability:** `[VERIFIED: WEB greedy_tensor_search.cpp:503-568
  baseProj.IsEmpty() skip]`, `[VERIFIED: LOCAL projection.rs:93-106
  categorical-only]`.

---

### ORD-06-02 — `combination_ctr_eligible` is true iff some used projection is the candidate minus exactly one member

- **Status:** draft
- **Responsibility:** given a non-empty `used_projections`, decide whether
  `projection` is a legitimate one-feature extension of some member of it.
- **Preconditions:** `used_projections` may be empty or non-empty (ORD-06-01
  covers the empty case as a degenerate `false`; this spec covers the
  membership arithmetic for the general case, including non-empty
  `used_projections` that still yield `false`).
- **Input/Output:** same as ORD-06-01 (`combination_ctr_eligible`).
- **Dependencies:** `TProjection::cat_features()` only — no new type, no
  hand-rolled projection extension (the predicate compares MEMBER SETS, it
  does not construct a new `TProjection`).
- **Behavior (Given/When/Then):**
  - **Given** `used_projections = [{0}]` (one simple CTR chosen), `projection
    = {0,1}`, **then** `true` (`{0}` has 1 member, `{0,1}` has 2, and `{0} ⊆
    {0,1}`).
  - **Given** `used_projections = [{0}]`, `projection = {1,2}`, **then**
    `false` (`{0}` is NOT a subset of `{1,2}` — this projection does not
    extend the used one; it is unrelated).
  - **Given** `used_projections = [{0}]`, `projection = {0,1,2}` (length 3,
    2 more members than the used projection's 1), **then** `false` (the
    length gap is 2, not exactly 1 — this candidate would need an
    intermediate `{0,x}` to already be used first; a single already-used
    length-1 projection cannot license a length-3 jump in one step, matching
    upstream's strict one-at-a-time `AddCatFeature` extension).
  - **Given** `used_projections = [{0}, {1}]` (two distinct simple CTRs
    chosen), `projection = {0,1}`, **then** `true` (matches via EITHER `{0}`
    or `{1}` — the predicate only needs ONE match, `any(...)`).
  - **Given** `used_projections = [{0,1}]` (an already-chosen COMBINATION
    CTR), `projection = {0,1,2}`, **then** `true` (extending a combination by
    one more feature is legitimate, same rule, length gap exactly 1).
  - **Given** `used_projections = [{0,1}]`, `projection = {0,1}` (identical,
    not an extension), **then** `false` (length gap is 0, not 1 — a
    projection is never eligible against itself; also structurally
    impossible for this predicate to be called this way since callers only
    invoke it for `projection.is_combination()` columns not already in
    `used_projections`, but the predicate itself must not accidentally
    return `true` for equal-length inputs).
- **Invariants:** pure, deterministic; symmetric in neither direction (this
  is NOT a symmetric relation — `combination_ctr_eligible(P, [Q])` and
  `combination_ctr_eligible(Q, [P])` can differ when `|P| != |Q|`).
- **Acceptance tests (unit):** the 6 Given/When/Then scenarios above, each an
  independent test in a new sibling test file (see §7).
- **Out of scope:** the "used_projections empty" degenerate case (ORD-06-01);
  wiring into the search loop (ORD-06-03); `max_ctr_complexity` capping and
  cross-column dedup (ALREADY handled by the existing, unmodified
  `tensor_ctr_candidates`/`enumerate_projections` — this predicate only
  filters ELIGIBILITY among already-generated, already-capped, already-deduped
  columns).
- **Traceability:** `[VERIFIED: WEB greedy_tensor_search.cpp:541-554 the
  extend-by-one-feature loop, verbatim]`, `[VERIFIED: CODEGRAPH
  crates/cb-train/src/projection.rs:130-132 cat_features()]`.

---

### ORD-06-03 — `select_level_ctr_aware` skips ineligible combination columns at scoring time

- **Status:** draft
- **Responsibility:** wire ORD-06-01/02 into the EXISTING "CTR candidates
  next" loop (`tree.rs:2589-2610`), replacing the current
  "score every materialized column unconditionally" behavior for
  COMBINATION columns only, WITHOUT changing simple-CTR handling, scoring
  formulas, `cat_feature_weight`, or the tie-break rule.
- **Preconditions:** ORD-06-01/02 available (`combination_ctr_eligible`).
- **Input/Output:** NO public signature change anywhere — `tree.rs`,
  `boosting.rs`, and `candidates.rs`'s existing public functions are
  UNCHANGED; this is a purely internal, private-function-level change inside
  `select_level_ctr_aware`'s existing loop body.
- **Dependencies:** ORD-06-01/02 (`combination_ctr_eligible`), the EXISTING
  `used_projections` computation already present in `select_level_ctr_aware`
  (`tree.rs:2578-2586`, unchanged), `TProjection::is_combination()`
  (existing).
- **Behavior (Given/When/Then):**
  - **Given** a tree at level 0 (`chosen` empty) with a model configuration
    that has BOTH float and CTR-eligible cat features, **then** the level's
    scored candidates include float candidates (unchanged), simple CTR
    candidates (unchanged, every simple column unconditionally scored), and
    ZERO combination CTR candidates (every combination column's
    `combination_ctr_eligible` check returns `false`, so it is `continue`'d
    before scoring — never pushed to `scored`).
  - **Given** the SAME tree at level 1, having chosen a `Ctr` split on a
    simple projection `{k}` at level 0, **then** level 1's scored candidates
    additionally include every materialized combination column whose
    projection is `{k}` extended by exactly one more feature (already
    materialized, now simply no longer filtered out).
  - **Given** a tree where the FIRST chosen split is `Float`, **then** every
    subsequent level's combination-CTR candidates remain filtered out
    (`used_projections` stays empty of `Ctr` entries) until a `Ctr` split is
    eventually chosen at some later level (§1 simplification).
  - **Given** the existing `tensor_ctr_e2e`/`multi_permutation_e2e` fixtures'
    configs (`depth=2`, `max_ctr_complexity=2`, 2 cat features, NO float
    features), **then** the fix must be re-verified (not assumed) to be a
    provable no-op for their exact tree structures/leaf values.
- **Invariants:** scoring formulas (`cat_feature_weight`,
  `score_candidate_ctr_aware`, `split_score`), the strict `> best` tie-break,
  and the FLOAT-then-simple-CTR-then-combination-CTR enumeration ORDER within
  the loop are UNCHANGED; only WHICH combination columns reach the scoring
  step changes, via one added `continue` guard.
- **Acceptance tests:**
  - AT-ORD06-03a (**integration, oracle**):
    `cargo test -p cb-model --test fstr_ctr_oracle_test` — all 3 tests
    (`fstr_ctr_predictions_sanity_gate`,
    `interaction_matches_upstream_on_mixed_ctr_model`,
    `pvc_matches_upstream_on_mixed_ctr_model`) pass at `<= 1e-5`.
  - AT-ORD06-03b (**regression, oracle**):
    `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test --test
    multi_permutation_e2e_oracle_test --test ctr_split_scoring_test` — all
    pass, UNCHANGED expected values (if any expected value needed to change,
    STOP and re-derive from real upstream output, not patch the fixture to
    match Rust's new output).
  - AT-ORD06-03c (**unit, new regression fence for exactly this bug**): a
    hand-built/synthetic scenario in `ctr_split_scoring_test.rs` (or a new
    sibling) with a genuine multi-feature `CtrFeatureColumn` materialized
    alongside a float candidate, `chosen` empty (level 0) — asserts the
    combination column is NEVER pushed to `scored`/cannot win, distinct from
    the existing single-feature-only tests there.
- **Out of scope:** the eligibility-predicate arithmetic itself (ORD-06-01/02,
  already specified); one-hot routing; ordered-boosting+CTR precedence;
  `candidates.rs`/`boosting.rs` (explicitly unchanged, §1/§4).
- **Traceability:** `[VERIFIED: WEB greedy_tensor_search.cpp:838-902
  SelectDatasetFeaturesForScoring call order]`, `[VERIFIED: CODEGRAPH
  crates/cb-train/src/tree.rs:2527-2632]`.

---

### ORD-06-04 — `max_bucket_count` is scoped to the per-level ELIGIBLE candidate set (plan-checker CRITICAL finding)

- **Status:** draft
- **Responsibility:** fix a SECOND, independent scoring-INPUT bug discovered
  by the Plan Checker: `max_bucket_count` (`tree.rs:2576-2581`, the
  `GetCatFeatureWeight`/`CalcMaxFeatureValueCount` penalty input) is computed
  over **ALL** materialized `ctr_features` unconditionally — it is NOT
  narrowed by ORD-06-03's new eligibility guard, since that guard only
  filters `scored`'s membership, computed AFTER `max_bucket_count` in the
  existing code.
- **Why this matters (verified against the vendored, NOT-absent
  `greedy_tensor_search.cpp:1097-1115` `CalcMaxFeatureValueCount`):** upstream
  computes this max over `candidatesContexts` — literally the CURRENT
  LEVEL's `SelectFeaturesForScoring` return value, i.e. the ALREADY
  `AddTreeCtrs`-gated list. Upstream NEVER includes an ineligible
  combination's bucket count in this max. Rust's unscoped
  `ctr_features.iter().map(|c| c.bucket_count).max()` DOES include it,
  inflating `max_bucket_count` whenever any combination column exists,
  REGARDLESS of ORD-06-03's fix. With the default `model_size_reg = 0.5`
  (non-zero, not overridden by the target fixture's `config.json`), this
  measurably changes `cat_feature_weight`'s multiplier for every simple-CTR
  candidate competing at the SAME level as an (now ineligible, but still
  bucket-counted) combination — for the target fixture (`cat_cardinalities:
  [5,4]`, so a `{0,1}` combination has `bucket_count` up to `5*4=20`),
  correct scoped `max_bucket_count = max(5,4) = 5` vs Rust's unscoped `=
  max(5,4,20) = 20` is a ~26% relative difference in the weight multiplier
  (`(1+1)^-0.5 ≈ 0.707` vs `(1+0.25)^-0.5 ≈ 0.894`) — enough to
  independently flip which candidate wins at the root, which is EXACTLY
  where the originally-observed divergence occurs. **Without this fix,
  AT-ORD06-03a is at material risk of still failing after ORD-06-01/02/03
  are correctly implemented.**
- **Preconditions:** ORD-06-01/02 (`combination_ctr_eligible`) available.
- **Input/Output:** no signature change — `max_bucket_count`'s computation
  inside `select_level_ctr_aware` changes from an unconditional `.iter()` to
  a `.filter(...)` over the SAME `ctr_features` slice, gated by the SAME
  predicate ORD-06-03 uses for `scored`.
- **Dependencies:** `combination_ctr_eligible` (ORD-06-01/02);
  `used_projections` must be computed BEFORE `max_bucket_count` (a
  reordering of the two existing `let` bindings — `used_projections`
  currently follows `max_bucket_count` in source order; this fix requires
  the reverse).
- **Behavior (Given/When/Then):**
  - **Given** `ctr_features` contains one combination column (`{0,1}`,
    `bucket_count=20`) and two simple columns (`{0}` bucket_count=5, `{1}`
    bucket_count=4), and `chosen` is empty (level 0, so the combination is
    INELIGIBLE per ORD-06-01), **then** `max_bucket_count == 5` (the max over
    ELIGIBLE columns only: the two simple ones), NOT `20`.
  - **Given** the SAME `ctr_features`, but `chosen = [Ctr(single(0))]` (so
    `{0,1}` IS now eligible per ORD-06-02), **then** `max_bucket_count ==
    20` (the combination is now correctly included).
  - **Given** `ctr_features` contains ONLY simple columns (no combination at
    all), **then** `max_bucket_count` is UNCHANGED from today's computation
    (the filter is a no-op when every column is simple — regression lock for
    the categorical-only-fixture case, e.g. `tensor_ctr_e2e` at any level
    where only simple candidates are materialized).
- **Invariants:** `max_bucket_count >= 1` always (existing `.max(1)` guard
  preserved); the filter predicate is IDENTICAL to the one ORD-06-03 applies
  to `scored` (a single shared predicate, not two independently-maintained
  copies — avoids future drift between the two gates).
- **Acceptance tests (unit):**
  - AT-ORD06-04a: synthetic `ctr_features` (1 combination + 2 simple, as
    above), `chosen = []` → `max_bucket_count == 5`.
  - AT-ORD06-04b: same `ctr_features`, `chosen = [Ctr(single(0))]` →
    `max_bucket_count == 20`.
  - AT-ORD06-04c: `ctr_features` all-simple → `max_bucket_count` unchanged
    from the pre-fix formula (regression lock).
- **Out of scope:** the eligibility predicate's own arithmetic (already
  specified, ORD-06-01/02); any OTHER scoring formula input (the Plan
  Checker's review found no other per-level-scoped input besides
  `max_bucket_count` — `cat_weight`'s `already_used` check, `hist_feature`
  indexing, and `score_candidate_ctr_aware` are all confirmed unaffected).
- **Traceability:** `[VERIFIED: WEB
  catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:1097-1115
  CalcMaxFeatureValueCount — VENDORED, directly read, not a v1.2.10-tag
  fetch]`, `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2576-2581
  max_bucket_count]`, `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:529
  model_size_reg default 0.5]`, `[VERIFIED: LOCAL
  crates/cb-oracle/fixtures/fstr_ctr/config.json — no model_size_reg
  override]`, `[PLAN-CHECK: pass 1 CRITICAL finding, 2026-07-18]`.

## 6. Acceptance scenarios (roll-up)

| Scenario | Spec | Kind | Oracle artifact | Bar |
|----------|------|------|-----------------|-----|
| `used_projections` empty → predicate always false | ORD-06-01 | unit | — | exact |
| Extension-membership arithmetic (subset + length-gap-1) | ORD-06-02 | unit | — | exact |
| `fstr_ctr_oracle_test.rs` all 3 tests pass | ORD-06-03 | oracle | `crates/cb-oracle/fixtures/fstr_ctr/*` (frozen, already committed) | ≤1e-5 |
| `tensor_ctr_e2e`/`multi_permutation_e2e`/`ctr_split_scoring_test` unchanged | ORD-06-03 | oracle/regression | existing frozen fixtures | ≤1e-5, unchanged |
| New level-0-combination-cannot-win regression fence | ORD-06-03 | unit | — | exact |
| `max_bucket_count` scoped to eligible candidates only | ORD-06-04 | unit | — | exact |

## 7. Impact scope

- **Classification:** `local` to `cb-train` (single crate); consumed
  transitively by `cb-model`'s oracle test (already exists, no code change
  needed there) and `catboost-rs`'s facade (no signature change expected).
- **Must change:** `crates/cb-train/src/tree.rs` ONLY — add the new private
  `combination_ctr_eligible` predicate (ORD-06-01/02), one `continue` guard
  inside `select_level_ctr_aware`'s existing "CTR candidates next" loop
  (ORD-06-03), AND reorder+filter the existing `max_bucket_count` computation
  (ORD-06-04, a plan-checker CRITICAL finding — a second, independent
  scoring-input bug in the SAME function, without which the primary oracle
  acceptance test is at material risk of still failing). All three reuse the
  ALREADY-COMPUTED `used_projections`. **`candidates.rs` and `boosting.rs`
  are UNCHANGED** (§1 Architecture correction) — no new materialization
  timing, no new candidate-generation function, no signature changes
  anywhere.
- **New test files:** `tree.rs` already mounts SEVERAL sibling test files
  (`#[path = "tree_test.rs"]`, `tree_tie_break_test.rs`,
  `tree_ordered_test.rs`, `tree_pairwise_test.rs`, `region_grow_test.rs` —
  `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:92,96,100,104,3155]`); the new
  `combination_ctr_eligible` unit tests (ORD-06-01/02) go in whichever
  existing sibling file already covers CTR-aware search internals (likely
  `tree_test.rs` — Planner to confirm by reading its contents), per the
  mandatory source/test separation rule — no embedded `mod tests` body, no
  new mount needed if an appropriate sibling already exists. Additions to
  `crates/cb-train/tests/ctr_split_scoring_test.rs` (or a new sibling
  integration-test file) for AT-ORD06-03c.
- **Verification only:** `crates/cb-compute/src/score.rs` (confirmed
  correct, re-run `score_test.rs` as a fence); `crates/cb-model`'s apply path
  (not implicated — the bug is upstream of a built model; the sanity-gate
  test in `fstr_ctr_oracle_test.rs` is the downstream verification signal).
- **Explicitly out of scope:** `catboost-master/` (read-only oracle
  reference, never modified); FSTR-01's `fstr.rs` (unaffected, already
  implemented, blocked only by this bug); ONNX export, CTR model loading
  (phase 23) — both consume an already-built `Model`, not training-time
  search.
- **Tests:** `crates/cb-oracle/fixtures/fstr_ctr/` and ALL other CTR
  fixtures are FROZEN — this fix must make the ALREADY-COMMITTED fixture
  pass; no fixture regeneration.
- **Build/operational:** none new; validated via
  `cargo test -p cb-train`, `cargo test -p cb-model`, full regression run.
  Restriction-lint gate: `cargo clippy -p cb-train --all-targets` (clippy,
  not `cargo build`, enforces `unwrap_used`/`expect_used`/`panic`/
  `indexing_slicing` — the same recurring project gotcha as every prior
  slice).

## 8. Compatibility and migration

Purely a training-time algorithm correctness fix — no serialization format
change, no public API signature change expected (internal `cb-train` wiring
only), no migration. Existing CTR-bearing models ALREADY TRAINED and
SERIALIZED by a prior (buggy) version of `cb_train::train_cat` are NOT
affected by this fix (the fix changes future training runs only, not stored
`.cbm`/`Model` data) — no re-training/re-serialization requirement is
introduced by this SPEC. `[INFERRED]`

## 9. Risks and open questions

1. **[RESOLVED]** The exact `TSplitTree`/`TProjection` upstream semantics
   (`GetBinFeatures`/`GetOneHotFeatures`/`GetUsedCtrs`/`IsRedundant`),
   flagged as a hard blocker by research.md, resolved via a direct `v1.2.10`
   tag `WebFetch` (see §1's "Codebase-specific simplification" — the mixed
   float+cat base is structurally irrelevant to this codebase's
   categorical-only `TProjection`).
2. **[OPEN, non-blocking, explicitly out of scope]** `boosting.rs`'s
   `has_ctr` branch is checked BEFORE the `ordered_learning_perm` branch,
   meaning `boosting_type=Ordered` + CTR features would currently take the
   non-ordered CTR-aware path unconditionally. This is a PRE-EXISTING,
   SEPARATE question (not this bug, not touched by this fix) — the Planner
   must verify no test exercises Ordered+CTR (confirmed absent: all existing
   CTR e2e fixtures use `Plain`) and must NOT accidentally alter this
   precedence while wiring ORD-06-03 into the same branch region.
3. **[INFERRED, MEDIUM confidence per research.md]** `tensor_ctr_e2e`'s and
   `multi_permutation_e2e`'s currently-passing status is because the bug is
   LATENT there (the illegitimate early combination candidate doesn't
   happen to out-score the legitimate winner in that specific toy data), not
   because those fixtures are immune to it. AT-ORD06-03b requires empirical
   re-verification (not assumption) that the fix is a no-op for them.
4. **[OUT OF SCOPE, explicitly excluded]** Upstream's `Rsm` (random
   subspace method) sampling gate inside `AddTreeCtrs`'s inner loop — no
   evidence this codebase implements/exposes `Rsm` at all; if it does exist
   under another name, the Planner should confirm it's fixed at the
   always-include default (`Rsm=1.0` equivalent) for all currently-relevant
   fixtures, or flag a new open question if not.

## 10. Traceability and sources

- **Discovery context:** `.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/` (FSTR-01, the feature whose oracle fixture surfaced this bug; unaffected by it beyond being blocked).
- **Research report:** `.planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/research.md`.
- **Upstream behavior (pinned `v1.2.10` tag):**
  `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/greedy_tensor_search.cpp
  AddTreeCtrs:503-568, SelectDatasetFeaturesForScoring call order ~838-902,
  GetCatFeatureWeight:926-950]`,
  `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/split.h
  GetBinFeatures/GetOneHotFeatures/GetUsedCtrs:469-494]`,
  `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/projection.h
  fields:57-59, IsRedundant:87-89, AddCatFeature:109-111,
  GetFullProjectionLength:125-130]`.
- **Rust seams:** `[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:159-201]`,
  `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2233-2757]`,
  `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:2680-2930,3890-3916]`,
  `[VERIFIED: CODEGRAPH crates/cb-train/src/projection.rs:93-187]`.
- **Absence proof (vendored tree):** `[VERIFIED: LOCAL grep -rln "class
  TSplitTree" catboost-master → empty]` — this is WHY the `v1.2.10` WebFetch
  was necessary (resolved risk 1).
- **Oracle fixture/test (already committed, frozen):**
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/fstr_ctr/*]`,
  `[VERIFIED: LOCAL crates/cb-model/tests/fstr_ctr_oracle_test.rs, directly
  re-run, all 3 currently RED]`.
- **Constraints:** `[VERIFIED: LOCAL Cargo.toml:10-14 restriction lints]`,
  `[VERIFIED: LOCAL CLAUDE.md source/test separation]`,
  `[PROJECT: memory ctr-model-loading.md, "CTR fixtures are frozen"]`.
- **PageIndex:** not yet indexed — `catboost_rs` folder (id
  `cmrhcxbtm000104jr3i5jzm0m`) currently holds only FSTR-03's `SPEC.md`
  `[VERIFIED: PAGEINDEX browse_documents]`. Human owner should add this as a
  new document; `process_document` has no in-place upsert.
