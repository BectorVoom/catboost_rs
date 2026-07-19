# Plan Check Result — PASS #3 (final allowed pass)

> **Post-pass-3 status: fixes applied, NOT independently re-verified.** Per
> the spec-tdd-planner-skill's cap of 3 checker passes, no pass #4 was run.
> The planner applied all 3 required revisions below directly to
> `SPEC.md`/`PLAN.md`:
> 1. **(CRITICAL)** AT-FIC02e's worked example changed from the flawed
>    opposite-signed `+3.0`/`-1.0` pair to a same-signed `L=+3.0`/`R=+1.0`
>    pair, with the arithmetic re-derived by hand against
>    `interaction_dfs`'s actual sign convention (confirmed: correct
>    `=|R-L|=2.0`, buggy sign-dropping `=|L|+|R|=4.0` — genuinely distinct,
>    whereas the retracted pair gave `4.0` under both). Fixed in both
>    SPEC.md's AT-FIC02e bullet and PLAN.md's T2 Red step.
> 2. **(MAJOR)** PLAN.md T3 now has an explicit, mandatory step widening
>    `prediction_values_change()`'s own `res` allocation
>    (`fstr.rs:123-138`) from `vec![0.0_f64; n_features]` to
>    `vec![0.0_f64; n_features + cat_feature_count(model)]`, and T3's Files
>    list now names `prediction_values_change()` itself, not just its two
>    accumulate helpers.
> 3. **(MINOR)** SPEC §9 risk item 7's dangling "see risk 8" cross-reference
>    corrected to "see risk 9."
>
> These fixes were derived by careful hand re-verification against the
> checker's own cited evidence (re-confirmed independently by re-deriving
> the `interaction_dfs` sign arithmetic against `fstr.rs:459-466` before
> writing the corrected worked example), but **have not passed an
> independent 4th automated check**. Per skill guardrails, this plan is
> presented as **ISSUES_FOUND → revisions applied, unverified by a final
> independent pass** — not as checker-approved `PASS`. A human reviewer (or
> a 4th ad-hoc check, if the user requests one outside the skill's normal
> 3-pass flow) should confirm these specific 3 fixes before treating the
> plan as fully ready.


**Verdict:** ISSUES_FOUND
**Goal:** FSTR-01 — extend `cb_model::fstr::interaction()` and
`prediction_values_change()` to attribute `ModelSplit::Ctr` (categorical)
split effect to a combined float+cat flat-index space, matching upstream
CatBoost `v1.2.10`'s `CalcFeatureInteraction` / `CalcRegularFeatureEffect`,
without perturbing existing float-only output.
**Plan:** `.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/PLAN.md`
(specs: `SPEC.md` FIC-01/FIC-02/FIC-03; evidence: `SOURCES.md`)

---

## Pass #1 and pass #2 summary (for context only — do not re-litigate; see
below for what pass #3 independently re-confirmed vs. newly found)

**Pass #1** (`ISSUES_FOUND`) found 1 MAJOR (non-symmetric DFS arm had zero
unit-test coverage) + 4 MINOR (T5 hedging on a settled fact; T2/T3 "parallel"
framing ambiguity; `interaction.npy` format left open; T4's combination-CTR
gate was a soft note) + 1 recommended-not-blocking item (agent-fetched
verbatim `v1.2.10` C++ / PVC ordering assumption stated with more confidence
than evidence supported). All 5 blocking items were closed in the pass-#2
revision (new `AT-FIC02e` test added; T5's settled-fact language; explicit
serialization guidance; pinned `.npy` format; hard gate re-asserted in Rust
tests); the recommended item became SPEC §9 risks 7 and 8 (now 7 and 9 after
pass #2's own risk-8 insertion — see MINOR-1 below).

**Pass #2** (`ISSUES_FOUND`) confirmed all 5 pass-#1 items closed, but found
1 new MAJOR while re-deriving the arithmetic via CodeGraph: T2's Green step
gave ONE shared formula (`delta.abs() / (side0.len()*side1.len())`) for both
arms, which is correct for the oblivious arm (where `delta` is already a
fully-aggregated, pre-abs() scalar) but **wrong** for the non-symmetric DFS
arm (where the per-terminal contribution is a single leaf's signed value,
accumulated **signed** across a whole tree, with `abs()` deferred to a
separate, later call site, `fstr.rs:384`) — applying the oblivious formula
verbatim there would drop the `sign` multiplication and take `abs()` too
early, defeating cross-leaf sign cancellation. Pass #2 required three things:
(1) an arm-specific, signed, non-abs()'d formula for the DFS arm
(`sign * delta / (side0.len()*side1.len())`), (2) a strengthened AT-FIC02e
construction guaranteed to numerically distinguish the correct
(signed-then-abs-once) result from the incorrect (abs-per-leaf-then-sum)
result, and (3) an explicit, recorded decision on whether T4's fixture
exercises non-symmetric trees at all.

**This pass's independent re-verification of the pass-#2 resolution
(re-derived from the actual current `fstr.rs` source via CodeGraph, not
trusted from the requester's summary):**

- **Revision #1 (arm-specific formula) — CONFIRMED CORRECT.** SPEC.md §5
  FIC-02 and PLAN.md T2's Green step now give two distinct formulas. I
  re-derived this independently against the verbatim current bodies of
  `interaction()` (`crates/cb-model/src/fstr.rs:288-355`),
  `interaction_accumulate_non_symmetric`/`interaction_dfs`
  (`crates/cb-model/src/fstr.rs:369-468`), and `interaction_add`
  (`crates/cb-model/src/fstr.rs:265-274`). Confirmed: dividing by the
  per-occurrence constant `side0.len()*side1.len()` **before** the
  cross-leaf signed sum (at the modified `fstr.rs:434` call site) is
  mathematically identical, by linearity, to aggregating the full signed sum
  for that specific ancestor-pair occurrence first and dividing once
  afterward — because every leaf sharing a given path-entry-pair occurrence
  shares the *same* two split projections (hence the same divisor), and
  `interaction_add`'s `(pairs, sums)` accumulator (`fstr.rs:265-274`) keys
  strictly by `(a, b)`, so different cross-product cells accumulate into
  independent slots without interference. Leaving `fstr.rs:384`'s
  `signed.abs()` as the sole, unmodified point where magnitude is taken
  remains valid after this generalization. **This part of item (a) in the
  requester's checklist is verified sound — no issue.**
- **Revision #2 (strengthened AT-FIC02e construction) — NOT actually
  achieved; see Issue CRITICAL-1 below.** The literal worked-example
  arithmetic added to satisfy this revision is wrong: hand-deriving it
  against the real `interaction_dfs` sign convention (`fstr.rs:457-466`,
  left child sign `-1`, right child sign `+1`) shows the chosen leaf values
  (`+3.0`/`-1.0`) produce an **identical** result under the correct and the
  buggy (sign-dropping) implementation, not a different one as claimed. This
  is a new, material finding for this pass — see below.
- **Revision #3 (grow_policy scope decision) — CONFIRMED CLOSED and
  internally consistent.** `crates/cb-train/src/boosting.rs:128-135`
  (`grow_policy_default() -> EGrowPolicy::SymmetricTree`) is cited correctly
  in both SPEC §9 risk 8 and PLAN.md T4; SPEC §6's acceptance-scenario
  roll-up table does not claim non-symmetric oracle coverage anywhere (its
  FIC-02/FIC-03 oracle rows read "mixed float+CTR fixture," with no
  arm-specificity claimed) — SPEC §9 risk 8 explicitly notes this
  consistency itself. **Item (c) of the requester's checklist is verified —
  no overclaim found.**

---

## Pass #3 findings (new)

### Specification Coverage

- [x] FIC-01 arithmetic (AT-FIC01a-d): unit tasks map 1:1 to SPEC
      Given/When/Then; sound, unchanged from pass #2.
- [x] FIC-02 oblivious arm (AT-FIC02a/b/c): sound, unchanged from pass #2.
- [ ] **FIC-02 non-symmetric arm's sole safety net (AT-FIC02e) does not
      reliably catch the regression class it was strengthened to catch** —
      see Issue CRITICAL-1. The requirement this must satisfy ("verification
      rests entirely on ... AT-FIC02e" per SPEC §9 risk 8) is therefore
      **not met as literally specified**.
- [ ] **FIC-03's output-vector widening is not an explicit task step** — see
      Issue MAJOR-1. `feature_count()` is explicitly locked unchanged by
      T1's own regression discipline, but T3's Files/Green step never
      instructs widening `prediction_values_change()`'s own
      `let mut res = vec![0.0_f64; n_features]` allocation
      (`crates/cb-model/src/fstr.rs:125`) to `n_features +
      cat_feature_count(model)`.
- [x] Grow-policy/non-symmetric oracle-coverage scope decision: resolved,
      consistent between SPEC.md and PLAN.md (see above).

### CodeGraph Evidence

- `interaction_dfs` in `crates/cb-model/src/fstr.rs:392-468` (verbatim,
  re-read this pass) — confirms the exact sign convention: `let mut sign:
  i32 = -1; for &child_idx in &[left_child, right_child] { ... path.push
  ((feature_idx, sign)); ...; sign = -sign; }` (`fstr.rs:459-466`) — left
  child gets `-1`, right child gets `+1`, freshly assigned per node (not
  accumulated across depths beyond the product taken at the terminal). The
  terminal accumulation loop (`fstr.rs:421-436`) computes, for each pair of
  path entries `(f1,s1)`,`(f2,s2)`: `sign = s1*s2; interaction_add(...,
  sign*delta)` where `delta` is the **single leaf's own raw value**
  (`fstr.rs:419`). Used to hand-derive the AT-FIC02e worked example
  independently (see CRITICAL-1).
- `interaction_accumulate_non_symmetric` in `crates/cb-model/src/fstr.rs:369-386`
  — confirms `signed.abs()` (`fstr.rs:384`) is taken exactly once per
  `(a,b)` pair, after the full per-tree DFS completes; unchanged by this
  slice, correctly left alone by the plan.
- `interaction_add` in `crates/cb-model/src/fstr.rs:265-274` — confirms the
  `(pairs, sums)` accumulator is keyed strictly by `(a,b)` with independent
  slots; supports the linearity argument in item (a) above.
- `feature_count` / `prediction_values_change` in
  `crates/cb-model/src/fstr.rs:75-138` (verbatim, re-read this pass) —
  confirms `prediction_values_change()`'s own body (not the two accumulate
  helpers) is where `res`'s length is fixed: `let n_features =
  feature_count(model); let mut res = vec![0.0_f64; n_features];`
  (`fstr.rs:124-125`), and `feature_count()` itself only ever considers
  `ModelSplit::float_feature()` (`fstr.rs:87,95`) — it is NOT touched by
  FIC-01 per PLAN.md T1's own explicit regression lock ("do NOT touch
  `feature_count`'s existing body/behavior"). Used to establish Issue
  MAJOR-1.
- `pvc_accumulate_oblivious` in `crates/cb-model/src/fstr.rs:143-174`
  (verbatim, re-read this pass) — confirms the checked
  `res.get_mut(src_idx)` write-back pattern T3 is meant to extend; supports
  the "silent no-op, not a panic" failure mode described in MAJOR-1.
- `Model::feature_importance` in `crates/catboost-rs/src/model.rs:139-149` —
  unchanged from pass #2's verification; no facade break, confirmed again.

### Issues

#### [CRITICAL] AT-FIC02e's strengthened worked example does not actually distinguish the correct implementation from the sign-dropping bug it was added to catch

- **Plan location:** SPEC.md §5 FIC-02, AT-FIC02e bullet (the `+3.0`/`-1.0`
  worked example); PLAN.md T2's Red step, `interaction_non_symmetric_two_ctr_splits_partial_overlap_self_pair`
  (identical worked example repeated verbatim).
- **Requirement:** Pass #2's Required Revision #2 (verbatim): "leaf values
  chosen so the signed-cancellation-then-abs result differs numerically from
  the abs()-per-leaf-then-sum result... not merely 'a known non-zero delta
  `d`'" — SPEC §9 risk 8 additionally states AT-FIC02e is "the SOLE
  verification of this arm's correctness in the entire slice," since T4's
  fixture is oblivious-only by deliberate scope decision.
- **Evidence (hand-derived against the actual current code, per the
  requester's explicit instruction to re-derive this, not trust the
  document):** Both leaves described ("reaching the two children of the
  deeper split") share the identical ancestor path entry down to-and-
  including the shallower CTR split — call its path sign `s_A` (identical
  for both leaves, since they are both descendants of the same edge). The
  deeper split's own local child sign differs per `fstr.rs:459-466`'s fixed
  convention: `-1` for its left child, `+1` for its right child. So, writing
  `L` for the leaf reached via the deeper split's left child and `R` for the
  leaf reached via its right child, the *correct* per-pair signed
  contribution (matching the unmodified `fstr.rs:433-434` and `384`) is:
  `contribution_L = s_A * (-1) * L`, `contribution_R = s_A * (+1) * R`, so
  the correct final magnitude for the shared pair (after the single, deferred
  `.abs()` at `fstr.rs:384`) is `|s_A * (R - L)| = |R - L|`. The claimed
  *incorrect* (sign-dropping, abs-per-leaf) implementation instead computes
  `|L| + |R|`. These two expressions, `|R - L|` and `|L| + |R|`, are
  **equal** exactly when `L` and `R` have opposite (or zero) sign, and
  **differ** (with `|R - L| < |L| + |R|`, genuine cancellation) only when `L`
  and `R` share the **same** sign. The chosen values, `L = +3.0` (reaching
  the deeper split's left child) and `R = -1.0` (its right child), have
  **opposite** signs: `|R - L| = |(-1.0) - 3.0| = 4.0`, and `|L| + |R| =
  3.0 + 1.0 = 4.0` — **identical**, not "a DIFFERENT number by construction"
  as the document literally claims. (The division by the constant
  `side0.len()*side1.len()` present in the actual CTR formula scales both
  sides by the same factor and does not change this equality.) The
  document's own stated arithmetic, `|3.0 - 1.0| = 2.0`, is simply
  arithmetically wrong for a leaf pair of `+3.0` and `-1.0` under the actual
  `sign*delta` computation — it silently drops the minus sign on the second
  leaf's own value in the subtraction (treating `R` as `+1.0` instead of
  `-1.0`). A pair of **same-signed** leaf values (e.g. `+3.0` and `+1.0`)
  would actually reproduce the claimed `2.0` (correct) vs `4.0` (incorrect)
  distinction — but that is not what either document specifies.
- **Failure scenario:** An executor implements AT-FIC02e literally as
  written (leaf values `+3.0`/`-1.0` at the two children of the deeper
  split) and separately implements the DFS arm's Green step *incorrectly*
  (drops `sign`, takes `abs()` per leaf before summing — precisely the
  regression class this test exists to catch). Both the buggy and the
  correct implementation produce `4.0` (pre-division-and-normalization) for
  every one of the three surviving cross-product cells, so AT-FIC02e is
  **green under the bug**. Since T4's oracle fixture is oblivious-only by
  explicit, accepted scope decision (SPEC §9 risk 8), there is **no other
  test in the entire slice** that would ever catch this — the sign-dropping
  bug ships silently.
- **Impact:** The non-symmetric arm's `interaction()` output for any real
  model with CTR splits sharing a root-to-leaf path (the exact scenario this
  slice singles out as its highest-risk code change) could be systematically
  wrong (inflated, non-cancelling) with zero test signal, in production, for
  as long as the codebase exists — defeating the explicit purpose pass #2
  added this test for.
- **Required revision:** Change AT-FIC02e's chosen leaf values (in both
  SPEC.md and PLAN.md, which currently repeat the identical flawed example)
  to a **same-signed** pair — e.g. `+3.0` and `+1.0` reaching the deeper
  split's two children — and recompute the worked numbers from the actual
  `sign*delta` mechanics (`s_A` common factor, deeper-split local sign `-1`/
  `+1`) rather than a naive literal subtraction of the two raw values. With
  `L=+3.0`, `R=+1.0`: correct `= |R-L| = |1.0-3.0| = 2.0` (matches the
  document's already-claimed "2.0"), incorrect (abs-per-leaf) `= |3.0| +
  |1.0| = 4.0` — genuinely distinct, as originally intended. Re-verify the
  final per-cell numbers (divided by `2*2=4`, then percent-normalized) using
  this corrected pair before finalizing the test's hard-coded assertion.
  This must be fixed in **both** SPEC.md's FIC-02 AT-FIC02e bullet and
  PLAN.md's T2 Red step (currently byte-identical text in both places).

#### [MAJOR] T3's task text never instructs widening `prediction_values_change()`'s own output-vector allocation, though `feature_count()` is explicitly locked unchanged

- **Plan location:** PLAN.md T3 ("Files" list and "Green" step); SPEC.md §5
  FIC-03 Preconditions ("output vector width widens to `n_float +
  n_cat_used`... see Impact below" — no concrete "Impact" subsection within
  FIC-03 actually restates this as an action item; §7's document-level
  "Must change" bullet is the only place it is even implied, generically,
  for the whole file).
- **Requirement:** SPEC §4 ("`interaction()`'s and `prediction_values_change()`'s
  public return types are UNCHANGED... the `usize`s now range over `[0,
  n_float + n_cat_used)`... whenever the model has CTR splits") and FIC-03's
  own Output clause ("length widens per §4 when CTR splits present").
- **Evidence:** `crates/cb-model/src/fstr.rs:123-138` (`prediction_values_change`,
  verbatim, re-read this pass): `let n_features = feature_count(model); let
  mut res = vec![0.0_f64; n_features];` — this is the **only** place `res`'s
  length is fixed; `pvc_accumulate_oblivious`/`pvc_accumulate_non_symmetric`
  (the two functions T3's "Files" list actually names for modification)
  receive `res: &mut [f64]` as an already-allocated **slice**, so they
  cannot themselves widen it. `feature_count()` (`fstr.rs:80-100`) only ever
  calls `ModelSplit::float_feature()` (`fstr.rs:87,95`) and is explicitly
  **not** to be touched — PLAN.md T1's own Refactor step states "do NOT
  touch `feature_count`'s existing body/behavior (regression lock)." Given
  this lock, nothing in T1/T2/T3 as literally written ever widens the vec
  `prediction_values_change()` allocates.
- **Failure scenario:** An executor follows T3's Green step exactly as
  written (modify only the two accumulate helpers) and calls
  `res.get_mut(flat_cat_index(n_float, c))` for a slot at or beyond
  `n_features` — this is a **checked** access (matches the project's
  no-raw-indexing discipline), so it returns `None` and the `if let Some(s)
  = ...` guard silently does nothing: the entire CTR redistribution is
  dropped with no panic, no error, no log. For AT-FIC03b's hand-built model
  (a single `Ctr` split, no float splits at all — `n_features =
  feature_count(model) == 0`), `res` would be allocated as a **zero-length**
  vector, and `dif` would be silently discarded entirely; `res` would stay
  empty; `convert_to_percents` (guarded on `total == 0.0`) would leave it
  unchanged; `prediction_values_change()` would return `Vec::new()`.
- **Impact:** FIC-03 as specified cannot pass its own unit tests
  (AT-FIC03b/c) or oracle test (AT-FIC03d) without this additional,
  unstated code change. In TDD practice this is very likely to be
  self-revealing (the Red step for AT-FIC03b would fail loudly — an index
  that silently doesn't exist rather than an incorrect value — prompting the
  executor to notice and fix the allocation site), so this is not rated
  CRITICAL, but it is a genuine, concrete gap in the plan's own stated task
  scope for an artifact (`prediction_values_change`'s own body) that SPEC §4
  explicitly requires to widen and that T3's "Files" list omits.
- **Required revision:** Add an explicit T3 step (or a shared T1/T0 step,
  since both FIC-02 and FIC-03 need it) stating: "in `prediction_values_change()`
  (`fstr.rs:123-138`), change `let mut res = vec![0.0_f64; n_features];` to
  `let mut res = vec![0.0_f64; n_features + cat_feature_count(model)];`" —
  and add `crates/cb-model/src/fstr.rs`'s `prediction_values_change()`
  function itself to T3's "Files" list (currently only the two accumulate
  helpers are named).

#### [MINOR] SPEC §9 risk item 7's cross-reference ("see risk 8") points to the wrong risk after pass #2's insertion

- **Plan location:** SPEC.md §9, risk item 7 (the agent-fetched-verbatim-C++
  risk), final sentence: "...the FIRST hypothesis to revisit is that this
  quoted algorithm is subtly wrong (e.g., an omitted normalization step, a
  different tie-break, or a wrapper-level reordering — **see risk 8**)."
- **Evidence:** Risk 8 (as pass #2 inserted it) is the `[RESOLVED]`
  grow_policy/non-symmetric-oracle-coverage scope decision — it has nothing
  to do with "wrapper-level reordering." Risk **9** ("Whether upstream's
  Python-facing `get_feature_importance(type='PredictionValuesChange')`
  array is genuinely ordered by original/external flat feature index, versus
  ... internal `Sort(...)` by score") is the risk item that actually
  discusses wrapper-level reordering. This is a renumbering artifact from
  pass #2 inserting the new risk-8 item into the middle of a previously
  differently-numbered list.
- **Failure scenario:** Low — this only misdirects a human debugging an
  oracle mismatch to the wrong risk item; it does not affect any executable
  task or test.
- **Required revision:** Change "see risk 8" to "see risk 9" in SPEC §9 risk
  item 7.

### Implementation Order Review

Task order (`T0 → T1 → {T2,T3 serialized} → {T5 needs T2+T4, T6 needs T3+T4}
→ T7`) remains dependency-valid. The MAJOR-1 fix (widening
`prediction_values_change`'s allocation) belongs inside T3 (it is a
prerequisite for T3's own Green step to make AT-FIC03b/c pass, not a
separate downstream task) — no task reordering required, only a text
addition at T3. The CRITICAL-1 fix (AT-FIC02e's leaf values) is a text
correction inside T2's existing Red step — no reordering required either.

### Potential Bugs

- **Sign-dropping / premature-`abs()` regression in the non-symmetric DFS
  arm, undetected by the current AT-FIC02e text** (CRITICAL-1). Trigger:
  implementing the DFS arm's per-cell contribution as `delta.abs() /
  (n0*n1)` (dropping `sign`) instead of `sign * delta / (n0*n1)`. Failure
  mode: with the specific leaf values currently written into both documents,
  this bug produces byte-identical output to the correct implementation for
  the given test, so it ships undetected. Required mitigation: same-signed
  leaf-value pair, per the Required Revision above.
- **Silent redistribution loss in `prediction_values_change()` for any
  CTR-only or CTR-heavy hand-built test model** (MAJOR-1). Trigger: T3
  implemented literally as written, without independently noticing that
  `feature_count(model)` alone under-sizes `res` whenever the model has any
  `Ctr` splits. Failure mode: checked `.get_mut` silently no-ops instead of
  panicking, so the symptom is "unexpectedly all-zero output" rather than a
  crash — exactly the kind of swallowed-error class this review is tasked
  to search for. Required mitigation: explicit allocation-widening step,
  per the Required Revision above.

### Verification Strategy Assessment

- FIC-01 unit coverage: sound, unchanged from pass #2.
- FIC-02 oblivious-arm coverage (AT-FIC02a/b/c): sound, unchanged.
- FIC-02 non-symmetric-arm coverage (AT-FIC02e): **present but not
  discriminating as literally specified** — a verification-strategy defect,
  not merely a spec-coverage gap (same category pass #2 flagged, not yet
  actually closed). This is the single highest-severity open item in this
  slice, because SPEC §9 risk 8 itself states this test is the *only*
  verification for this code path.
- FIC-03 unit coverage (AT-FIC03b/c) is well-specified at the level of
  *expected values*, but implicitly depends on an allocation-widening change
  that no task explicitly assigns — a verification-strategy gap in the
  sense that the tests, as specified, cannot pass without an unstated
  supporting code change.
- Oracle-level backstop (T5/T6): confirmed oblivious-only by deliberate,
  consistently-documented scope decision (SPEC §9 risk 8) — acceptable as an
  explicit, bounded scope decision, not a defect in itself.

### Required Plan Revisions

1. **(CRITICAL)** Replace AT-FIC02e's worked leaf-value example in **both**
   SPEC.md §5 FIC-02 and PLAN.md T2 with a same-signed pair (e.g. `+3.0` and
   `+1.0` reaching the deeper split's two children), and recompute the
   documented "correct vs incorrect" numbers from the actual `sign*delta`
   mechanics (common ancestor sign factor, deeper-split's own `-1`/`+1`
   local child sign) rather than a literal subtraction of the raw leaf
   values. Re-verify that the corrected pair yields numerically distinct
   correct/incorrect results before finalizing the hard-coded test
   assertion.
2. **(MAJOR)** Add an explicit step to PLAN.md T3 (and its "Files" list)
   instructing that `prediction_values_change()`'s own body
   (`crates/cb-model/src/fstr.rs:123-138`) widen its `res` allocation from
   `vec![0.0_f64; n_features]` to `vec![0.0_f64; n_features +
   cat_feature_count(model)]`, since `feature_count()` itself is explicitly
   locked unchanged by T1.
3. **(MINOR)** Fix SPEC §9 risk item 7's dangling cross-reference from "see
   risk 8" to "see risk 9."

### Unverified Items

- **The exact verbatim `v1.2.10` C++ source for `CalcFeatureInteraction` /
  `CalcRegularFeatureEffect`** — unchanged from passes #1/#2: this review has
  no web-fetch tool and cannot independently re-verify the quoted upstream
  source. Carried as SPEC §9 risk 7 (open, not settled), pending
  AT-FIC02d/AT-FIC03d's oracle comparison.
- **Whether upstream's Python-facing PVC array is ordered by original
  feature index vs. internal score-sort** — unchanged from passes #1/#2,
  carried as SPEC §9 risk 9 (mislabeled "risk 8" at one dangling
  cross-reference site — see MINOR finding above), with fallback guidance.
  Same status: acknowledged, not independently verified, gated on T6's
  oracle comparison.
