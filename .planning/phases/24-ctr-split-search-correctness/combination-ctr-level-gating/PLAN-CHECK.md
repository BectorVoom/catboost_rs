## Plan Check Result

**Verdict:** PASS
**Goal:** ORD-06 — make combination (multi-feature) CTR candidates ineligible
at any tree level until the tree has already chosen at least one `Ctr` split
to extend, matching upstream `AddTreeCtrs`'s `baseProj.IsEmpty()` gate,
without perturbing simple-CTR handling, scoring formulas, or the
already-committed `tensor_ctr_e2e`/`multi_permutation_e2e` fixtures — AND
(added in the revision this pass reviews) scoping `max_bucket_count` to the
same per-level eligible candidate set, matching upstream
`CalcMaxFeatureValueCount`.
**Plan:** `.planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/PLAN.md`
(spec: `SPEC.md`, evidence: `SOURCES.md`)

---

### Pass #1 summary (for context — superseded by this pass's findings below)

Pass #1 (verdict `ISSUES_FOUND`) found:
1. **[CRITICAL]** `max_bucket_count` (`tree.rs:2576-2581`, a `cat_feature_weight`
   scoring INPUT) was computed over ALL materialized `ctr_features`
   unconditionally, unaffected by the then-planned eligibility guard, which
   only filtered `scored`'s membership. Verified against vendored
   `CalcMaxFeatureValueCount` (`greedy_tensor_search.cpp:1097-1115`), which
   iterates only the current level's already-`AddTreeCtrs`-gated
   `candidatesContexts`. This threatened the plan's own primary acceptance
   criterion (`fstr_ctr_oracle_test.rs`'s sanity gate).
2. **[MAJOR]** T3 had no contingency branch for "the sanity gate itself still
   fails after T1+T2 land" — only a branch for "sanity gate passes,
   interaction/PVC still fail."
3. **[minor]** `SPEC.md`'s `pub(crate) fn combination_ctr_eligible` vs
   `PLAN.md`'s `fn combination_ctr_eligible` (no `pub(crate)`) inconsistency.
4. **[minor, no action needed]** T0's re-verify-source step already covered
   the line-number drift pass 1 found.

**Resolution applied** (per the task prompt, independently re-verified below,
not merely trusted): a new SPEC section `ORD-06-04` and a new PLAN task `T2.5`
were added, reordering `used_projections` before `max_bucket_count` and
filtering `max_bucket_count`'s input by the SAME `combination_ctr_eligible`
predicate T2 uses for `scored`; T3 gained an explicit second contingency
branch; `combination_ctr_eligible`'s visibility was made consistently
private (plain `fn`) in both documents.

---

### Pass #2 findings (this review)

#### (a) Numeric walkthrough of the ORD-06-04 fix — CONFIRMED CORRECT

Independently re-derived (not reused from pass 1) against the CURRENT
`crates/cb-train/src/tree.rs` source (re-read directly, not assumed):

- `max_bucket_count` (pre-fix): `tree.rs:2576-2581` —
  `ctr_features.iter().map(|c| c.bucket_count).max().unwrap_or(1).max(1)`,
  computed BEFORE `used_projections` (`tree.rs:2582-2590`) in the CURRENT
  source (confirms the reorder is real and necessary, not hypothetical).
- Corrected code (T2.5's Green step):
  ```rust
  let used_projections: Vec<&crate::TProjection> = /* unchanged, now first */;
  let max_bucket_count = ctr_features
      .iter()
      .filter(|c| c.projection.is_simple() || combination_ctr_eligible(&c.projection, &used_projections))
      .map(|c| c.bucket_count)
      .max()
      .unwrap_or(1)
      .max(1);
  ```
- **Case `chosen = []`** (`used_projections = []`): filter keeps `{0}`
  (`is_simple`, bucket_count 5) and `{1}` (`is_simple`, bucket_count 4);
  excludes `{0,1}` (`is_combination`, `combination_ctr_eligible({0,1}, [])`
  → `false`, empty `used_projections`). `max(5,4) = 5`. **Matches SPEC's
  expected `5`.**
- **Case `chosen = [Ctr(single(0))]`** (`used_projections = [&{0}]`): filter
  keeps `{0}`, `{1}` (simple, unconditional), AND `{0,1}` — `combination_ctr_eligible({0,1}, [{0}])`:
  `q_members = {0}` (len 1), `members = {0,1}` (len 2), `1+1==2` and
  `0 ∈ {0,1}` → `true`. `max(5,4,20) = 20`. **Matches SPEC's expected `20`.**
- Independently confirmed upstream's actual scoping via the VENDORED
  `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:1097-1115`
  (`CalcMaxFeatureValueCount`, directly read this pass, not taken on faith):
  it iterates `candidatesContexts` — the return of `SelectFeaturesForScoring`,
  itself built by `AddFloatFeatures → AddOneHotFeatures → AddSimpleCtrs
  (unconditional) → AddTreeCtrs (only if `currentSplitTree.Defined()`, and
  only adds EXTENSIONS of already-used bases, `greedy_tensor_search.cpp:503-568`,
  directly read this pass)`. This is exactly the "eligible-only" set T2.5
  reconstructs. **The fix is a correct, verified port.**
- `model_size_reg` default confirmed `0.5` directly at
  `crates/cb-train/src/boosting.rs:536-538`
  (`pub fn model_size_reg_default() -> f64 { 0.5 }`), and the target fixture's
  `crates/cb-oracle/fixtures/fstr_ctr/config.json` confirmed to carry no
  `model_size_reg` override (re-read this pass) — SPEC's ~26% weight-multiplier
  divergence claim is arithmetically sound: `(1+1)^-0.5 ≈ 0.707` (correct,
  `max=5`) vs `(1+0.25)^-0.5 ≈ 0.894` (buggy, `max=20`).

#### (b) Sweep for a THIRD per-level-scoped-input gap — NONE FOUND (thorough re-check)

Per the task's explicit instruction to be extra-careful here, `build_ctr_aware_histogram`
(`tree.rs:2306-2375`) and the rest of `select_level_ctr_aware` (`tree.rs:2527-2639`)
were re-read in full this pass, checking every place `ctr_features` is iterated
unfiltered:

- `n_ctr = ctr_features.len()` / `n_features = n_float + n_ctr` (`tree.rs:2316-2317`)
  — sizes the combined histogram's feature axis over ALL materialized columns,
  including now-ineligible combinations. **Not a scoring-input bug**: this only
  reserves array CAPACITY (a `(leaf, feature, bin)` flat layout — `cb-compute/src/histogram.rs:235-241`
  `cell_base`); an ineligible column's histogram slice is built but never READ
  (T2's `continue` skips it before `score_candidate_ctr_aware`/`hist_feature`
  use), so it cannot influence any OTHER column's score. Wasted compute only,
  explicitly documented as such in the existing code comment
  (`tree.rs:2325-2326`, "Extra empty upper bins contribute `0.0` and are inert").
- `ctr_actual = ctr_features.iter().flat_map(|c| c.bins.iter().copied()).max()...`
  (`tree.rs:2331-2335`) — sizes `n_bins` (histogram WIDTH) over the max observed
  bin value across ALL columns, including ineligible ones. Same conclusion:
  inert capacity sizing, not a per-column scoring input — a wider-than-needed
  `n_bins` does not change `scan_border_to_leaf_stats`'s per-feature prefix sum
  for any OTHER feature (feature-major layout, independent per-feature blocks;
  confirmed via `cb-compute/src/histogram.rs:714-771` `scan_border_to_leaf_stats`,
  re-read this pass).
- `already_used` check (`tree.rs:2601`, `used_projections.iter().any(|p| **p == column.projection)`)
  — unaffected: it is EQUALITY-based (not eligibility-based) and only reached
  for columns that already passed T2's eligibility `continue` guard (simple
  columns unconditionally, combinations only if eligible) — no unfiltered read.
- `hist_feature = n_float + col` (`tree.rs:2612`) — a plain positional index
  into the (over-sized but inert) histogram; not itself a scoring VALUE.
- **A genuinely subtle case independently traced and confirmed CORRECT (not a
  bug)**: does an ALREADY-CHOSEN combination projection (e.g. `{0,1}` split at
  level `k`) remain eligible for re-scoring at level `k+1`? Traced through
  `combination_ctr_eligible({0,1}, used_projections=[{0,1}])`: `q_members.len()+1
  == members.len()` → `2+1 == 2` → `false` — **correctly ineligible**, matching
  upstream's actual behavior: `AddTreeCtrs`'s erasure block
  (`greedy_tensor_search.cpp:570-582`, directly read this pass) removes an
  already-used combination projection's cached candidate unless it is ALSO
  re-derived as a fresh one-feature EXTENSION of some (shorter) `baseProj` this
  level — the un-extended projection itself is never re-added to `candList`.
  This exact scenario is independently covered by SPEC's own ORD-06-02
  scenario 6 ("`used_projections = [{0,1}]`, `projection = {0,1}` → `false`,
  length gap 0"), so the plan already specifies the behavior this review
  independently confirmed matches upstream. (By contrast, SIMPLE CTRs remain
  reofferable at every level via `AddSimpleCtrs`, unconditionally, matching
  the plan's unchanged simple-CTR handling.) No revision needed — flagged here
  only because this is precisely the class of subtlety the task asked to hunt
  for, and it resolved in the plan's favor.
- **Conclusion: `max_bucket_count` was the ONLY other per-level-scoped scoring
  input with a real correctness defect.** No third gap found.

#### (c) T2.5 ordering and T3 contingency — CONFIRMED SOUND

- T2.5's stated dependency ("depends on T1... must follow T2 to avoid
  conflicting edits to the same function... T3 requires BOTH T2 and T2.5")
  is accurate: T1 provides `combination_ctr_eligible`; T2 and T2.5 both edit
  adjacent lines of the SAME `select_level_ctr_aware` function body, so
  serializing them (rather than "parallelizing" against the same function) is
  the correct call, and T3's oracle acceptance test genuinely requires both
  landed (per (a)'s numeric proof) — not merely prose caution.
  Re-verified against the CURRENT file that `max_bucket_count` (`2576-2581`)
  really does precede `used_projections` (`2582-2590`) in source order today,
  confirming the reorder T2.5 describes is real and necessary, not a
  hypothetical concern.
- T3's new contingency branch ("if the sanity gate itself still fails after
  T1+T2+T2.5... re-check whether ANY OTHER per-level-scoped scoring input
  still reads unfiltered `ctr_features`... before concluding the predicate's
  own arithmetic is at fault") is genuinely actionable: it names concrete
  functions to re-read (`select_level_ctr_aware`, `build_ctr_aware_histogram`)
  and a concrete class of defect (a scoring INPUT, not the candidate list
  itself) — exactly the shape of bug this pass's own (b) sweep exercised. Not
  padding; it correctly encodes this review's own methodology for a future
  executor to reuse if the (now unlikely, but not logically impossible) case
  arises.

#### (d) Other inconsistencies scanned — one minor, non-blocking gap found

- `SOURCES.md` (the evidence ledger) was NOT updated alongside this revision:
  it still reflects only the pre-pass-1 research and contains no entry for
  the vendored `CalcMaxFeatureValueCount` evidence, `model_size_reg_default`,
  or the `fstr_ctr` fixture's `config.json` non-override that SPEC.md's own
  §5/§10 now correctly cite. This is a documentation-completeness gap in the
  designated "evidence ledger" file, not a plan-correctness defect — SPEC.md
  itself carries the full, independently-reverified citations, so no
  implementation risk follows from it. Recommended (non-blocking) revision:
  append an "ORD-06-04 (plan-checker pass 1→2)" section to `SOURCES.md`
  mirroring SPEC.md §5's traceability line, for consistency with the rest of
  the ledger's style.
- No other new inconsistency found between SPEC.md, PLAN.md, and the current
  `tree.rs`/`greedy_tensor_search.cpp` sources.

---

### Specification Coverage

- [x] Combination CTR ineligible at root / float-only history (ORD-06-01):
  T1's Red tests map directly; predicate re-verified correct against current
  `projection.rs` (`cat_features`, `is_simple`, `is_combination`, `from_features`,
  `single`, all confirmed present with the exact signatures the predicate
  assumes).
- [x] Extension-membership arithmetic (ORD-06-02): T1's 6 scenarios verified
  by direct execution-by-hand against the predicate's exact code, including
  the subtle "already-used combination cannot re-win against itself"
  scenario 6, independently cross-checked against upstream's erasure logic
  (see (b) above).
- [x] Simple CTR unconditionally available at every level (ORD-06-03):
  confirmed the guard only fires for `is_combination()`; `AddSimpleCtrs`
  upstream counterpart re-read, confirmed unconditional every level.
- [x] Wiring into `select_level_ctr_aware` (ORD-06-03): guard placement
  (before `cat_weight`/`hist_feature` use) confirmed to avoid wasted
  scoring compute, matches "never even considered" upstream semantics.
- [x] **`max_bucket_count` scoped to the per-level eligible set (ORD-06-04)**:
  T2.5 maps directly; numeric example independently re-derived and confirmed
  correct against both the corrected Rust code and the vendored
  `CalcMaxFeatureValueCount` (see (a)).
- [x] `candidates.rs`/`boosting.rs` unchanged: re-confirmed via CodeGraph —
  `select_level_ctr_aware` has exactly 1 caller
  (`greedy_tensor_search_oblivious_with_ctr`), no other dependent relies on
  "every materialized column scored every level."
- [x] T3 contingency for both failure modes (sanity gate fails outright, and
  sanity gate passes but interaction/PVC fail): both branches now present
  and actionable.

### CodeGraph Evidence

- `select_level_ctr_aware` in `crates/cb-train/src/tree.rs:2527-2639`
  - Definition: re-read in full this pass (not merely diffed). `max_bucket_count`
    at `2576-2581` (precedes `used_projections` in CURRENT source, confirming
    T2.5's reorder claim is accurate today, not stale). `used_projections` at
    `2582-2590`. The "CTR candidates next" loop at `2596-2622`.
  - Callers/dependents: 1 caller, `greedy_tensor_search_oblivious_with_ctr`
    (`tree.rs:2669`), consumed by 7 sites in `boosting.rs`; test coverage via
    `crates/cb-train/tests/ctr_split_scoring_test.rs` (confirmed this pass:
    `greedy_tensor_search_oblivious_with_ctr` is `pub`, reachable from the
    integration-test crate boundary with an explicit `model_size_reg`
    parameter — AT-ORD06-03c is feasible as specified).
  - Callees: `build_ctr_aware_histogram` (`tree.rs:2306-2375`, re-read in full
    this pass), `score_candidate_ctr_aware` (`tree.rs:2389-2399`),
    `cat_feature_weight` (`tree.rs:2416-2422`), `scan_border_to_leaf_stats`
    (`crates/cb-compute/src/histogram.rs:714-771`, re-read this pass).
  - Impact assessment: T2's `continue` guard and T2.5's `max_bucket_count`
    filter are both pure narrowings of what gets SCORED/WEIGHTED; neither
    touches `ctr_features` itself (the tree-wide materialized list) or any
    positional index (`col` still ranges `0..ctr_features.len()`,
    `ctr_features.get(*col)` reconstruction elsewhere in `boosting.rs`
    unaffected). **No index-corruption risk, confirmed.**
- `CalcMaxFeatureValueCount` in
  `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:1097-1115`
  (vendored, directly read this pass, not a web fetch) — iterates
  `candidatesContexts` (the current level's already-`AddTreeCtrs`-gated
  candidate list), confirming T2.5's filter is the correct port.
- `AddTreeCtrs` in
  `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:509-583`
  (vendored, directly read this pass) — confirms `seenProj` construction,
  the `baseProj.IsEmpty()` skip, the extend-by-one-feature loop, AND (newly
  traced this pass) the erasure block (`570-582`) that drops an already-used,
  non-re-extended combination projection from later-level candidacy —
  independently corroborating ORD-06-02 scenario 6's "against itself → false"
  rule.
- `TProjection` in `crates/cb-train/src/projection.rs:99-187` — `cat_features()`,
  `is_simple()`, `is_combination()`, `from_features()`, `single()`, `with_added()`
  all confirmed present with the exact signatures SPEC.md §4/PLAN.md T1 assume;
  `#[derive(..., PartialEq, Eq, Hash, PartialOrd, Ord)]` confirms `==`/`contains`
  usage in the predicate and the existing `already_used` check both compile.
- `model_size_reg_default` in `crates/cb-train/src/boosting.rs:536-538` —
  returns `0.5`; `crates/cb-oracle/fixtures/fstr_ctr/config.json` re-read this
  pass, confirmed no override — the ~26% weight-multiplier divergence claim in
  SPEC §5 ORD-06-04 is arithmetically reproduced independently in (a) above.

### Issues

None at CRITICAL, MAJOR, or BLOCKER severity. Pass 1's CRITICAL and MAJOR
items are both resolved and independently re-verified correct (see (a)-(c)
above). One MINOR, non-blocking item remains:

#### [MINOR] `SOURCES.md` evidence ledger not updated to reflect the ORD-06-04 revision
- **Plan location:** `SOURCES.md` (entire file — no ORD-06-04 section exists).
- **Evidence:** `SPEC.md` §5 ORD-06-04 and §10 cite
  `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:1097-1115`
  (`CalcMaxFeatureValueCount`), `crates/cb-train/src/boosting.rs:529`
  (`model_size_reg` default), and the `fstr_ctr` fixture's `config.json`
  non-override; none of these appear in `SOURCES.md`, which otherwise mirrors
  every other SPEC claim with a corresponding ledger entry.
- **Failure scenario:** none for implementation correctness (SPEC.md's own
  citations are complete and independently re-verified this pass); this is a
  documentation-consistency gap only — a future reader consulting SOURCES.md
  alone would not find the ORD-06-04 evidence trail.
- **Impact:** cosmetic / documentation hygiene only.
- **Required revision (non-blocking):** append an ORD-06-04 evidence section
  to `SOURCES.md` mirroring SPEC.md §5/§10's citations, for consistency with
  the ledger's existing style. Does not block implementation.

### Implementation Order Review

1. T0 (re-verify current source) — correctly first; this pass re-confirmed
   the file still matches SPEC.md §4's description closely enough for T0's
   "if drifted, STOP and re-derive" instruction to remain sufficient (the
   same few-line citation drift pass 1 found persists, already anticipated).
2. T1 (`combination_ctr_eligible`) before T2/T2.5 — correct; both depend on
   the predicate existing.
3. T2 (wire the guard into `scored`) before T2.5 (reorder + filter
   `max_bucket_count`) — correct; both touch the same function, avoiding
   concurrent edits to the same lines.
4. T2.5 before T3 — correct and NECESSARY: (a)'s numeric proof confirms T3's
   primary oracle acceptance test requires T2.5 to have landed; T2 alone is
   provably insufficient.
5. T3 (oracle verification, both fixture success AND regression-fixture
   no-op) before T4 (full sweep + gate) — correct.
6. No circular task dependencies; strictly serial single-file fix, confirmed
   no two tasks mutate the same lines concurrently.

### Potential Bugs

- **`max_bucket_count` scope** — RESOLVED by T2.5, independently re-verified
  correct via direct numeric walkthrough and vendored-source re-read (see (a)).
- **A third per-level-scoped-input gap** — searched for exhaustively this
  pass (see (b)); none found. `n_bins`/`ctr_actual` histogram-width sizing
  reads unfiltered `ctr_features` but is provably inert (capacity only, no
  score derived from it for any other column).
- **Already-used combination re-eligibility at a later level** — traced and
  confirmed CORRECT (predicate returns `false` against itself, matching
  upstream's erasure behavior) — not a bug, already correctly specified by
  ORD-06-02 scenario 6.
- **Degenerate-candidate risk** (all columns filtered out) — re-confirmed:
  `enumerate_projections`/`tensor_ctr_candidates` always emits every
  CTR-eligible feature's simple projection whenever `max_ctr_complexity >= 1`,
  and simple projections are never eligibility-filtered, so `max_bucket_count`'s
  filtered set can only be empty when `ctr_features` itself is empty — already
  handled by the pre-existing `.unwrap_or(1).max(1)` guard, preserved
  unchanged by T2.5.
- **Ordered boosting + CTR precedence** — still correctly out of scope
  (unchanged from pass 1's assessment); `config.json` confirms `Plain`.

### Required Plan Revisions

1. (Non-blocking, minor) Append an ORD-06-04 evidence section to `SOURCES.md`
   mirroring SPEC.md §5/§10's citations (vendored `CalcMaxFeatureValueCount`,
   `model_size_reg_default`, fixture `config.json` non-override) for
   documentation consistency. Does not block implementation start.

### Unverified Items

None material. All symbols, line ranges, and upstream citations cited in this
review were independently re-read this pass (not reused from pass 1's
findings or SPEC.md's own claims without cross-checking): `tree.rs:2306-2639`,
`crates/cb-compute/src/histogram.rs:185-241,714-771`,
`crates/cb-train/src/projection.rs:99-187`, `crates/cb-train/src/boosting.rs:525-538`,
`crates/cb-oracle/fixtures/fstr_ctr/config.json`, and
`catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:460-583,895-1115`
(vendored, present in the repository, directly read — no WebFetch needed for
this pass's re-verification since the relevant file is not absent from the
vendored tree, unlike `split.h`/`projection.h` which pass 1 could not
independently re-fetch and which remain taken on SPEC.md's own evidence
ledger for the `v1.2.10`-tag-specific quotes of `TSplitTree`/`TProjection`
semantics — this residual (pre-existing, not newly introduced) reliance on
SPEC's own WebFetch citations for `split.h`/`projection.h` is unchanged from
pass 1 and does not affect this pass's verdict, since this pass's own
CalcMaxFeatureValueCount/AddTreeCtrs verification uses the VENDORED,
directly-read file instead).
