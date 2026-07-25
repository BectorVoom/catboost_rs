# Verification pass 6 (post-cap, third follow-up)

This is a sixth review of `cv-cross-validation` (ORCH-01), continuing beyond
the spec-tdd-planner-skill's normal 3-pass cap per explicit, repeated user
request. Pass 5 (preserved in full below) returned **ISSUES_FOUND** with one
MAJOR: `SPEC.md` had not been updated to reflect the empty-explicit-`folds`
invariant that `PLAN.md` had already fixed (missing from §5-S4's Behavior,
§5-S5's Behavior, §4's `cv` doc-comment `# Errors` list, §6's Acceptance
Scenarios table, and §9's Risks table; `PLAN.md`'s own §0 Acceptance→Task
table was stale for the same reason). This pass (a) independently
re-verifies every one of those five required `SPEC.md` additions plus the
`PLAN.md` §0 fix are genuinely present and accurate, (b) greps `SPEC.md` for
every remaining "population" mention and confirms none is a live, un-corrected
claim, (c) re-derives every load-bearing CodeGraph claim fresh from source
(not the documents' own citations) for the symbols underpinning this
specific fix, and (d) re-runs the full checker workflow from scratch,
hunting adversarially for anything new.

## Verification of the five (six, counting the `PLAN.md` companion) requested fixes

All are **PRESENT** and **accurate**, independently re-read this pass:

1. **`SPEC.md` §5-S4 "Behavior"** now states, immediately after the existing
   `F == 1` sentence: *"`F == 0` (an empty `per_fold` slice — e.g. from an
   explicit but empty caller-supplied `folds` list) is ALSO a typed error,
   not `sum_f64(&[])/0 == NaN`... a defense-in-depth guard independent of the
   TASK-06 entry-point check"* (`SPEC.md` lines 464-467). This matches
   `PLAN.md` TASK-04's `aggregate_zero_folds_errs` Red test exactly (lines
   387-394): `F == 0` ⇒ `.is_err()`, never `NaN`.
2. **`SPEC.md` §5-S5 "Behavior"** now has a dedicated bullet: *"`folds =
   Some(&[])` (explicit but EMPTY) ⇒ `Err(CatBoostError::Train(Degenerate))`,
   validated at entry BEFORE `aggregate_folds` is reached — never a
   `NaN`-bearing `Ok(CvResult)`"* (lines 494-496). Matches `PLAN.md` TASK-06's
   `cv_empty_explicit_folds_errs` Red test exactly (lines 564-569).
3. **`SPEC.md` §4's `cv` doc-comment** now explicitly separates the two
   triggers instead of leaving the caller to infer it from "a bad partition":
   the `folds` parameter doc states `Some(&[])` "is ALSO a typed error,
   validated at entry" (lines 309-313), and the `# Errors` list itself now
   reads *"a bad partition, empty `metrics`, **an empty explicit `folds`
   list**, fewer than 2 resulting folds, mismatched per-fold iteration
   counts, or any training/eval failure"* (lines 321-323) — closing the exact
   ambiguity pass 5 flagged (previously "a bad partition" was the only
   phrase a reader could map this case onto).
4. **`SPEC.md` §6's Acceptance Scenarios table** gained two new rows,
   `AT-S4b` (*"`F==0` (empty per-fold slice) and `F==1` ⇒ typed error, never
   `NaN`"*, line 557) and `AT-S5d` (*"`cv(..., folds=Some(&[]))` (explicit
   empty folds) ⇒ typed error, never a `NaN`-bearing `Ok`"*, line 558). Both
   descriptions match the corresponding `PLAN.md` Red tests verbatim
   (`aggregate_zero_folds_errs` / `cv_empty_explicit_folds_errs`).
5. **`SPEC.md` §9's Risks table** gained a new row (line 626) documenting the
   empty-explicit-`folds` risk, its discovery point ("found at post-cap
   verification pass 4"), and its two-guard mitigation (entry-point check in
   `cv()` + defense-in-depth check in `aggregate_folds()`), citing `AT-S4b`/
   `AT-S5d` and the fresh `sum_f64([])==0.0` CodeGraph evidence. This closes
   the "sole missing risk row" gap pass 5 identified (every other
   multi-pass-discovered risk already had a tracked row; this was the one
   exception, and no longer is).
6. **`PLAN.md` §0's "Goal-backward derivation" table** — the `AT-S4` row
   now reads *"cross-fold mean/sample-std (ddof=1) over hand curves match
   closed form; `F==0`/`F==1` ⇒ typed error, never NaN"* and the `AT-S5` row
   now reads *"...; `folds=Some(&[])` ⇒ typed error, never NaN"* (`PLAN.md`
   lines 31-32). This closes the specific staleness pass 5 flagged. Note:
   `PLAN.md` chose the "extend the existing AT-S4/AT-S5 rows" option pass
   5's required revision explicitly offered as an alternative to "add two new
   rows" — so `PLAN.md` never cites the literal `AT-S4b`/`AT-S5d` labels
   `SPEC.md` §6 introduces (confirmed via `grep -n "AT-S4b\|AT-S5d"
   PLAN.md`, zero hits). This is a **cosmetic ID-labeling asymmetry, not a
   content gap** — the underlying invariant IS described accurately in both
   documents' tables, and pass 5's own revision text pre-authorized this
   exact alternative ("extend the TASK-04 and TASK-06 rows, **or** add two
   new rows"). Not re-raised as a new issue.

All six fixes are genuine, accurate, and independently re-derivable — not
narrative alone.

## Verification: no remaining stale "population" language

Per the user's specific instruction, `grep -n -i "population" SPEC.md` was
re-run this pass. Three hits, all at the lines pass 5's fix summary named
(§1.1 line 126, §5-S4 line 461, §9 line 621) — each read in full context this
pass, and each is **corrective/historical framing, not a live claim**:

- Line 126: *"...to ddof=1, not 'population std' as an earlier draft of this
  section stated."* — explicitly flags the OLD assumption as wrong.
- Line 461: *"...(empirically confirmed against `catboost.cv`'s reported
  `-std` columns; **NOT population/ddof=0 as an earlier draft assumed** — see
  §1.1)..."* — same pattern, embedded directly in the live formula sentence
  as a parenthetical correction, not asserting population std is used.
- Line 621 (Risks table): *"Empirically confirmed SAMPLE std, ddof=1
  (`/(F-1)`), **not population**. §5-S4 pinned."* — again explicitly stating
  the resolution, not a live claim of population std.

No hit asserts or implies that `cv()`'s `-std` columns are population std.
This confirms pass 5's own finding (the ddof fix itself, from an earlier
pass) remains clean; this was a re-verification, not a new discovery.

## Fresh CodeGraph/Read re-derivation of the load-bearing symbols underpinning this fix (this pass, not trusted from the documents' own citations)

- `cb_core::sum_f64` — DEFINITION re-read verbatim at
  `crates/cb-core/src/reduction.rs:32-38`: `pub fn sum_f64(values: &[f64]) ->
  f64 { let mut acc = 0.0_f64; for &v in values { acc += v; } acc }`. Confirms
  `sum_f64(&[])` returns `0.0` (loop body never executes), so
  `sum_f64(&[])/0usize as f64 == 0.0/0.0 == NaN` — the exact defect the new
  SPEC/PLAN language describes. 131 workspace-wide callers (CodeGraph
  blast-radius), consistent with prior passes.
- `EvalMetric::eval` — re-read verbatim at `crates/cb-train/src/metrics.rs:
  261-269`: `if approx.is_empty() { return
  Err(CbError::Degenerate("eval metric: empty eval set".to_owned())); }` —
  confirms the citation TASK-05's corrected attribution relies on, exact line
  numbers match.
- `cb_train::boosting` empty-target guard — re-read verbatim at
  `crates/cb-train/src/boosting.rs:2336-2338`: `if n == 0 { return
  Err(CbError::Degenerate("empty target".to_owned())); }`, with `n =
  target.len()` at line 2334 for the non-multilabel path — confirms TASK-05's
  citation exactly.
- `cb_model::apply::predict_raw_staged` — re-read verbatim at
  `crates/cb-model/src/apply.rs:465-532`: `n_objects =
  feature_values.first().map_or(0, Vec::len)` at line 492; with
  `n_objects == 0` the `rows` vec is empty, the per-row loop (line 511) never
  executes, and `stages` (pre-sized via `Vec::with_capacity(n_objects)` per
  stage) stays at length 0 for every stage — confirms this function does NOT
  error on 0 objects, matching the corrected TASK-05 prose exactly.
- `Model::ensure_scalar_oblivious` / `staged_predict` — re-read verbatim at
  `crates/catboost-rs/src/model.rs:144-171` / `:173+` (`staged_predict`
  header at 173, matching the `189-208` range cited for the body/defaults).
  Guard rejects `approx_dimension > 1`, non-empty `non_symmetric_trees`/
  `region_trees`, and `Some(ctr_data)`, each with a typed
  `CatBoostError::UnsupportedModel` — unchanged from prior passes.
- `CatBoostBuilder::new()` — re-read verbatim at `crates/catboost-rs/src/
  builder.rs:98-117`: `boost_from_average: true,` at line 108,
  `bootstrap_type: EBootstrapType::No,` at line 110. Setters
  `boost_from_average` (line 197) and `bootstrap_type` (line 211) confirmed
  `#[must_use]`, consuming-`self`. Unchanged from prior passes; re-confirmed
  because TASK-01/TASK-06 both still depend on this pair.
- Filesystem check (this pass): `crates/catboost-rs/src/cv.rs`,
  `crates/cb-oracle/fixtures/cv/`, and any `select_rows`/`make_cv_folds`/
  `aggregate_folds`/`cv(` symbol are confirmed **absent** from the tree (`ls
  crates/catboost-rs/src/`, `ls crates/cb-oracle/fixtures/` show no `cv`
  artifacts) — consistent with the plan's own `status: draft` /
  `implementation_state: unimplemented` claim; no drift between "what the
  plan describes" and "what exists on disk" has crept in since pass 5.

No stale citation, incorrect signature, or misattributed ownership was found
in any load-bearing claim touched by this fix.

## New finding from this pass

**None.** No new BLOCKER, CRITICAL, MAJOR, or MINOR issue was found. The one
asymmetry noted above (the `AT-S4b`/`AT-S5d` IDs existing in `SPEC.md` §6 but
not literally cited by that ID inside `PLAN.md` §0, which instead folds the
same content into the existing `AT-S4`/`AT-S5` rows) is not raised as an
issue: the content is accurate and complete in both documents, and pass 5's
own required-revision text explicitly pre-authorized this exact alternative
("extend the TASK-04 and TASK-06 rows, **or** add two new rows"). Raising it
again here would be re-litigating an already-resolved, already-optional
choice, not identifying a new defect.

## Verdict rationale (this pass)

All five specifically-requested `SPEC.md` additions (plus the `PLAN.md` §0
companion fix) are genuinely present, technically accurate, and now fully
consistent with `PLAN.md`'s actual Red tests — verified by a full,
independent re-read of both documents in their entirety (not spot-checked),
fresh CodeGraph/Read re-derivation of every load-bearing symbol underpinning
the fix, and an explicit re-grep confirming no stale "population std"
language survives anywhere in `SPEC.md`. No new functional, structural,
ordering, or documentation-consistency defect was found in this pass.

Per the user's own framing: this plan has now had 5 real, distinct findings
across 5 passes, each genuinely fixed and independently re-verified in the
following pass. This sixth, adversarial, from-scratch pass — checking
specifically for the fix requested AND hunting independently for anything
else — found **nothing**. The plan is genuinely complete, internally
consistent between `SPEC.md` and `PLAN.md`, and every acceptance criterion
traces to a concrete, TDD-executable task with objective verification.

**Verdict: PASS.**

Per the user's explicit instruction, automated fix-scope iteration on this
plan should be considered exhausted going forward: six passes have now run
(three past the normal 3-pass cap), the last one found a genuine
documentation-only gap which is now closed, and this seventh-in-sequence
check (pass 6) found nothing further. Any subsequent review of this artifact
pair should be a human read-through rather than another automated
adversarial pass, per the user's own stated ceiling.

---

## Plan Check Result (pass 6)

**Verdict:** PASS
**Goal:** ORCH-01 — `cv()` k-fold cross-validation surface: partition a `Pool` into disjoint train/test folds, train+staged-evaluate each fold via existing seams, and aggregate per-iteration `test-<M>-mean/std` / `train-<M>-mean/std` columns, oracle-locked ≤1e-5 against `catboost.cv(..., folds=<fixed>, shuffle=False)` for a scalar/oblivious/float-only (numeric-regression) first slice.
**Plan:** `/home/user/Documents/workspace/catboost_rs/.planning/plans/cv-cross-validation/PLAN.md`, `/home/user/Documents/workspace/catboost_rs/.planning/plans/cv-cross-validation/SPEC.md`

### Summary
- Pass 5's sole MAJOR (SPEC.md/PLAN.md §0 documentation drift on the
  empty-explicit-`folds` invariant) is genuinely and accurately fixed in
  both documents: SPEC.md §5-S4 Behavior, §5-S5 Behavior, §4's `cv`
  doc-comment `# Errors` list, §6's Acceptance Scenarios table (new
  `AT-S4b`/`AT-S5d` rows), and §9's Risks table all now document the
  invariant; PLAN.md §0's Acceptance→Task table is no longer stale.
- No remaining stale "population std" language exists anywhere in SPEC.md —
  all three "population" hits are explicit historical/corrective framing
  ("not population... as an earlier draft assumed/stated"), never a live
  claim.
- Every load-bearing CodeGraph/Read citation underpinning this specific fix
  (`sum_f64`, `EvalMetric::eval`'s empty-approx guard, `boosting.rs`'s
  empty-target guard, `predict_raw_staged`'s zero-object silent-success
  behavior, `ensure_scalar_oblivious`/`staged_predict`,
  `CatBoostBuilder::new()`'s `boost_from_average`/`bootstrap_type` defaults)
  was independently re-derived fresh from source this pass and matches the
  documents' citations exactly, including line numbers.
- The new SPEC §6/§9/PLAN §0 rows accurately describe what TASK-04's
  `aggregate_zero_folds_errs` and TASK-06's `cv_empty_explicit_folds_errs`
  Red tests actually assert — verified by reading both the SPEC prose and
  the PLAN task bodies side by side.
- One cosmetic asymmetry (PLAN.md §0 does not cite the literal
  `AT-S4b`/`AT-S5d` labels SPEC.md §6 introduces, instead folding the same
  content into the existing `AT-S4`/`AT-S5` rows) is not raised as an issue:
  pass 5's own required-revision text explicitly pre-authorized this exact
  alternative, and the content itself is accurate in both places.
- No new functional, structural, sequencing, or documentation defect was
  found in this from-scratch adversarial pass. This plan reaches a genuine
  PASS on this sixth pass.

### Specification Coverage
- [x] ORCH-01-S1 fold-index partitioning (`make_cv_folds`) incl. grouped +
  shuffle interaction: TASK-03. Unchanged from pass 3/4/5 verification.
- [x] ORCH-01-S2 `Pool::select_rows`: TASK-02. Unchanged.
- [x] ORCH-01-S3 per-fold train + staged eval, incl. degenerate 0-row fold
  (correct per-seam attribution): TASK-05. Unchanged, re-confirmed this pass
  (`boosting.rs:2336-2338`, `apply.rs:465-532`, `metrics.rs:267-269` all
  re-read verbatim, exact match).
- [x] ORCH-01-S4 cross-fold mean/std aggregation, incl. `F==0`/`F==1`
  degenerate guards: TASK-04. `F==0` guard now documented in SPEC §5-S4 AND
  §6 (`AT-S4b`) AND §9 Risks — closes pass 5's MAJOR.
- [x] ORCH-01-S5 Rust facade `cv` + oracle, incl. empty-explicit-`folds`
  entry-point guard: TASK-01 (fixtures) + TASK-06. Now documented in SPEC
  §4/§5-S5/§6 (`AT-S5d`)/§9 AND PLAN §0 — closes pass 5's MAJOR.
- [x] ORCH-01-S6 Python `catboost_rs.cv`: TASK-07. Unchanged.
- [x] Non-goals correctly scoped. Unchanged.
- [x] SPEC/PLAN traceability for the empty-explicit-`folds` invariant: now
  present in both documents, at every location pass 5 required (§5-S4,
  §5-S5, §4 doc-comment, §6 acceptance table, §9 risk table, PLAN §0 table).

### CodeGraph Evidence
- `cb_core::sum_f64` — `crates/cb-core/src/reduction.rs:32-38`, re-read
  verbatim this pass: `pub fn sum_f64(values: &[f64]) -> f64 { let mut acc =
  0.0_f64; for &v in values { acc += v; } acc }`. `sum_f64(&[])` returns
  `0.0` (loop never executes). Impact assessment: confirms
  `sum_f64(&[])/0usize as f64 == NaN`, the exact defect the new SPEC/PLAN
  language and TASK-04/06 guards close.
- `EvalMetric::eval` — `crates/cb-train/src/metrics.rs:261-269`, re-read
  verbatim: `if approx.is_empty() { return Err(CbError::Degenerate("eval
  metric: empty eval set".to_owned())); }`. Impact assessment: matches
  TASK-05's corrected attribution exactly (this is the seam that raises the
  typed error for an empty TEST fold, not `staged_predict`).
- `cb_train::boosting` (`train_inner`) — `crates/cb-train/src/boosting.rs:
  2320-2338`, re-read verbatim: `n = target.len()` (non-multilabel path,
  line 2334), `if n == 0 { return Err(CbError::Degenerate("empty
  target".to_owned())); }` (lines 2336-2338). Impact assessment: matches
  TASK-05's citation for the empty-TRAIN-fold case exactly.
- `cb_model::apply::predict_raw_staged` — `crates/cb-model/src/apply.rs:
  465-532`, re-read verbatim: `n_objects = feature_values.first().map_or(0,
  Vec::len)` (line 492); with `n_objects == 0`, `rows` is empty, the
  per-row loop (line 511) never executes, every stage's `Vec<f64>` stays at
  length 0. Impact assessment: confirms this function does NOT error on a
  zero-row pool — it silently returns empty per-stage vectors, exactly as
  TASK-05's corrected prose states.
- `Model::ensure_scalar_oblivious` / `staged_predict` —
  `crates/catboost-rs/src/model.rs:144-171` / `:173+`, re-read verbatim.
  Guard rejects `approx_dimension>1`, non-empty `non_symmetric_trees`/
  `region_trees`, `Some(ctr_data)`, each typed `UnsupportedModel`. Unchanged
  from prior passes.
- `CatBoostBuilder::new()` — `crates/catboost-rs/src/builder.rs:98-117`,
  re-read verbatim: `boost_from_average: true,` (line 108), `bootstrap_type:
  EBootstrapType::No,` (line 110); setters at lines 197/211,
  `#[must_use]`, consuming-`self`. Impact assessment: TASK-01/TASK-06's
  fixture-parity fix remains correctly grounded.
- Filesystem check (this pass): `crates/catboost-rs/src/cv.rs`,
  `crates/cb-data/src/select_rows` (as a symbol), and
  `crates/cb-oracle/fixtures/cv/` are confirmed ABSENT from the working
  tree — consistent with the plan's `status: draft` /
  `implementation_state: unimplemented` claims; no implementation has
  silently begun that the plan text doesn't already describe.

### Issues

None found in this pass.

### Implementation Order Review
1. `TASK-01 ∥ TASK-02 ∥ TASK-03 → TASK-04 → TASK-05 → TASK-06 → TASK-07`
   remains acyclic; unchanged from pass 3/4/5 — this pass's fix is
   documentation-only and does not touch task sequencing.
2. The two guards (TASK-04's `aggregate_zero_folds_errs` defense-in-depth
   check, TASK-06's `cv_empty_explicit_folds_errs` entry-point check)
   remain correctly placed: TASK-04 (aggregation) before TASK-06 (facade
   `cv`, which depends on TASK-05 which depends on TASK-04) — no reordering
   needed.

### Potential Bugs
- No new potential bug surfaced this pass. All bugs identified across
  passes 1-5 (quantization sharing, `boost_from_average`/`bootstrap_type`
  mismatch, ddof convention, rayon/GPU concurrency, grouped+shuffle
  interaction, degenerate 0-row fold, empty explicit `folds`) remain closed
  with concrete guards/tests, now also fully documented in both SPEC.md and
  PLAN.md.

### Required Plan Revisions
None. The plan is ready for implementation as written.

### Unverified Items
- Whether `catboost.cv(..., folds=<explicit fixed list>, return_models=True)`
  actually returns per-fold models when `folds=` is passed explicitly — this
  is assumed by TASK-01's systematic parity self-check. Not independently
  re-run in this session (no live `catboost` install available this pass
  either); carried forward unchanged from pass 4/5 as a residual
  runtime-behavior assumption, not a plan defect (the same API usage
  pattern was already used successfully in the pass-2/pass-3 empirical
  investigation).
- Whether concurrent multi-thread use of `cubecl::client::ComputeClient<
  SelectedRuntime>` (`wgpu`/`cuda`/`rocm`) is actually unsafe, or merely
  unverified — no GPU hardware was exercised this session either; the
  plan's conservative `#[cfg(feature="cpu")]`-only parallelism scoping
  remains a reasonable mitigation regardless of the answer, carried forward
  unchanged from pass 4.

---

# Verification pass 5 (post-cap, second follow-up)

This is a fifth review of `cv-cross-validation` (ORCH-01), continuing beyond
the spec-tdd-planner-skill's normal 3-pass cap per explicit, repeated user
request. The prior recorded verdict (pass 4, "Extra verification pass
(post-cap)", preserved in full below) was **ISSUES_FOUND**, with one
residual MAJOR (an empty explicit `folds` argument could reach
`aggregate_folds` with zero folds and silently produce `NaN`) and one MINOR
(TASK-05's degenerate-zero-row test description mis-attributed which seam
raises the typed error for an empty TEST fold). The orchestrator applied
three fixes to `PLAN.md` afterward. This pass (a) independently re-derives
every load-bearing runtime-behavior claim from source — not the documents'
own citations — for all three fixes, (b) re-runs the full checker workflow
against the current `SPEC.md`/`PLAN.md` pair from scratch, and (c) hunts
adversarially for anything new.

## Verification of the three requested fixes

All three are **PRESENT** in `PLAN.md` and **technically correct**, verified
by independently re-deriving the underlying runtime behavior from source
this pass (not trusted from the documents' own citations):

1. **TASK-04 gained `aggregate_zero_folds_errs`.** Confirmed present at
   `PLAN.md` lines 387-394 (Red-test list), requiring `F == 0` (empty
   `per_fold` slice) to error, "NEVER `NaN`". Independently re-derived:
   `cb_core::sum_f64` (`crates/cb-core/src/reduction.rs:32-38`) folds an
   empty slice through zero loop iterations and returns the untouched
   accumulator `0.0_f64` — confirmed by fresh `Read` this pass, not merely
   re-cited. `sum_f64(&[]) / 0usize as f64` therefore evaluates to
   `0.0 / 0.0`, which is IEEE-754 `NaN`, exactly as the finding claimed. A
   dedicated `per_fold.is_empty()` guard inside `aggregate_folds` (as the
   Red test requires) is the correct, necessary fix, and is now specified.
2. **TASK-06's `pub fn cv` gained an entry-point non-empty check on the
   caller-supplied `folds` argument.** Confirmed present at `PLAN.md` lines
   489-494 (task description: "validate an explicit `folds` argument is
   non-empty too when `Some`... BEFORE calling `aggregate_folds`") and a new
   Red test `cv_empty_explicit_folds_errs` at lines 564-569
   (`cv(&pool, &builder, &["RMSE"], 3, false, 0, false, Some(&[]))` must be
   `.is_err()`). This closes the actual reachability path: `make_cv_folds`'s
   own `fold_count >= 2` guard is bypassed entirely whenever `folds = Some(_)`
   is supplied, so only an entry-point check on the caller-supplied list
   itself can prevent `Some(&[])` from reaching `run_fold`/`aggregate_folds`.
   Correctly sequenced BEFORE the `#[cfg(feature="cpu")]`/serial fold-running
   split and the `aggregate_folds` call, per the task's own prose.
3. **TASK-05's degenerate-zero-row test description was corrected** to
   distinguish the two cases precisely (lines 444-460). Independently
   re-derived, fresh `Read` this pass (not trusted from prior citations):
   - Empty TRAIN fold: `CatBoostBuilder::fit` (`crates/catboost-rs/src/
     builder.rs:334-402`) passes `pool.label()` straight through to
     `cb_train::train` → `train_with_eval_sets` → `train_inner`
     (`crates/cb-train/src/boosting.rs:2259`); when not multilabel,
     `n = target.len()` (line 2334), and `crates/cb-train/src/
     boosting.rs:2336-2338` reads exactly `if n == 0 { return
     Err(CbError::Degenerate("empty target".to_owned())); }` — confirmed
     verbatim. Also checked, as an additional adversarial step not
     previously verified: `fit`'s border-computation step
     (`select_borders_greedy_logsum`, `crates/cb-data/src/borders.rs:209-251`)
     runs BEFORE this guard and is called on the (possibly empty) float
     columns; independently traced that it does **not** panic on an empty
     column — `greedy_split` (`borders.rs:268-276`) has an explicit
     `if values.len() < 2 { return Vec::new(); }` guard, so the empty-train
     path reaches the `n == 0` check safely rather than panicking earlier in
     the pipeline. This closes a residual concern this pass raised and
     resolved itself: the "typed error, never panic" claim depends on more
     than just the cited guard, and the earlier stage was verified clean.
   - Empty TEST fold: `predict_raw_staged` (`crates/cb-model/src/
     apply.rs:465-532`), read verbatim again this pass: `n_objects =
     feature_values.first().map_or(0, Vec::len)` (line 492); with
     `n_objects == 0`, the per-object `rows` vec is empty, the `for row in
     &rows` loop (line 511) never executes, and every stage's `Vec<f64>`
     (pre-sized via `Vec::with_capacity(n_objects)`, line 510) stays at
     length 0 — confirmed this does NOT error, matching the corrected
     prose. The typed error instead fires one seam later, in
     `EvalMetric::eval`'s `approx.is_empty()` check
     (`crates/cb-train/src/metrics.rs:267-269`, read verbatim: `if
     approx.is_empty() { return Err(CbError::Degenerate("eval metric: empty
     eval set".to_owned())); }`), triggered by `run_fold`'s subsequent
     `eval_metric` call on the empty `test_stages[s]`/`test_pool.label()`
     pair. Both facts match the corrected TASK-05 prose exactly.

All three fixes are genuine, technically correct, and independently
re-derivable from the current on-disk source — not narrative alone.

## New finding from this pass (not previously flagged)

### [MAJOR] `SPEC.md` was not updated to reflect the empty-explicit-`folds` invariant that `PLAN.md` just fixed — a real SPEC/PLAN traceability gap, and `PLAN.md`'s own §0 acceptance-mapping table is now stale too

- **Plan location:** `SPEC.md` §5-S4 (ORCH-01-S4 Behavior), §5-S5 (ORCH-01-S5
  Behavior), §4 (the `cv` function's `# Errors` doc-comment), §6 (Acceptance
  Scenarios table), §9 (Risks and Open Questions table); `PLAN.md` §0
  ("Goal-backward derivation" Acceptance→Task table).
- **Requirement:** The mandatory operating principles require comparing the
  plan against "the specification, acceptance criteria... current
  implementation, and tests" and require every meaningful risk/requirement to
  have an objective, spec-traceable verification method (workflow §7). The
  pass-4 MAJOR finding this pass verifies as fixed was framed explicitly as a
  violation of "the plan's own stated invariant" drawn from `SPEC.md`
  §5-S4/§9 — but the actual textual fix was applied ONLY to `PLAN.md`.
- **Evidence:** A full re-read of the current `SPEC.md` (this pass) confirms
  the empty-explicit-`folds` invariant is **absent** from every place it
  should appear:
  - §5-S4 "Behavior" states only *"`F == 1` (a single fold) is a degenerate
    case with no defined sample std — typed error, not division by zero or
    NaN."* — no sentence anywhere addresses `F == 0`.
  - §5-S5 "Behavior" enumerates `cv()`'s typed-error triggers (`metrics`
    empty; unsupported model; `folds = Some(...)` overrides
    `fold_count`/`shuffle`/`seed`) but never states that an explicit `folds`
    argument must itself be non-empty.
  - §4's `pub fn cv` doc-comment `# Errors` section lists `[CatBoostError::
    Train]` triggers as *"a bad partition, empty `metrics`, mismatched
    per-fold iteration counts, or any training/eval failure"* — "a bad
    partition" naturally reads as `make_cv_folds`'s own `fold_count`
    validation (the INTERNALLY-generated path), not the caller-supplied
    explicit `folds` list; there is no separate, explicit mention of
    `Some(&[])` being rejected.
  - §6's Acceptance Scenarios table (the document's own testable-ID registry:
    AT-S1a/b/c/d, AT-S2, AT-S3/b, AT-S4, AT-S5/b/c, AT-S6) has **no entry**
    for this new invariant — no `AT-S4b` or `AT-S5d`.
  - §9's Risks table lists five explicitly `**RESOLVED**` rows (each tracking
    a prior finding closed with a citation) but has no row — resolved or
    open — for the empty-explicit-`folds` NaN risk, even though every other
    finding from every prior pass (quantization sharing, `boost_from_average`/
    `bootstrap_type`, ddof, rayon/GPU concurrency, grouped+shuffle) DOES have
    a tracked row there. This one is the sole exception.
  - `PLAN.md` §0's own "Goal-backward derivation" table (Acceptance ID →
    Task) was not extended either: it still lists only the original AT-S1
    through AT-S6 rows, so the two NEW Red tests (`aggregate_zero_folds_errs`
    in TASK-04, `cv_empty_explicit_folds_errs` in TASK-06) have no
    Acceptance-ID anchor anywhere in either document — they exist only as
    named Red-test bullets inside the task bodies.
  - As a smaller, related editorial gap: TASK-04's own "Formula (implement
    exactly)" section — the passage that is supposed to be the definitive,
    implement-verbatim algorithm description for the Green step — restates
    the `F == 1` guard explicitly but does not restate the `F == 0` guard
    that the adjacent Red-test bullet requires; only the Red-test list
    carries it.
- **Failure scenario:** None functionally for THIS implementation pass — the
  concrete Red tests and task prose in `PLAN.md` are sufficient for a
  TDD-disciplined implementer to build the guard correctly regardless of
  `SPEC.md`'s gap (the Red test forces it either way). The risk is
  structural/process, not a runtime bug: (a) a future maintainer who
  consults `SPEC.md` alone (e.g., to re-derive a plan after this one ships,
  or to audit "what does ORCH-01 guarantee") would not learn that
  `cv(..., folds=Some(&[]), ...)` is a documented, typed-error case — they
  would have to discover it by reading `PLAN.md`'s task-level prose instead
  of the acceptance-criteria document; (b) `SPEC.md`'s own Acceptance
  Scenarios table (§6) is the document's canonical list of "the testable
  things that must be true" — it silently under-counts by (at least) one
  real, shipped acceptance criterion, breaking the bidirectional
  spec-scenario-ID ↔ test traceability the rest of the document maintains
  scrupulously (every other test in every other task cites an AT-ID).
- **Impact:** Low immediate functional risk (the code-level fix is fully and
  correctly specified in `PLAN.md`); moderate process/maintainability risk
  (spec-plan drift, incomplete acceptance-criteria registry, weakened future
  audit/re-derivation reliability). This is the same category of
  documentation-integrity issue this multi-pass review has been tracking
  throughout (e.g., pass 3's citation fixes, pass 4's MINOR attribution fix)
  — those were fixed by updating the relevant document; this one was fixed
  in only one of the two documents that jointly constitute "the plan."
- **Required revision:**
  1. Add a sentence to `SPEC.md` §5-S4 "Behavior": *"`F == 0` (no folds at
     all — e.g. an empty explicit `folds` list) is likewise a degenerate
     case: typed error, never `NaN` (`sum_f64(&[])` is `0.0`, and `0.0/0`
     would otherwise silently evaluate to `NaN`)."*
  2. Add a bullet to `SPEC.md` §5-S5 "Behavior": *"An explicit `folds =
     Some(&[])` (empty list) is a typed error, validated before
     `make_cv_folds`/`run_fold`/`aggregate_folds` ever run."*
  3. Extend `SPEC.md` §4's `cv` doc-comment `# Errors` list to explicitly
     name "an empty explicit `folds` list" as one of the `CatBoostError::
     Train` triggers (distinct from "a bad partition").
  4. Add a new row to `SPEC.md` §6's Acceptance Scenarios table (e.g.
     `AT-S4b` for the `aggregate_folds` unit guard, `AT-S5d` for the `cv()`
     entry-point guard), and reference these new IDs from `PLAN.md` §0's
     Acceptance→Task table (extend the TASK-04 and TASK-06 rows, or add two
     new rows).
  5. Add a `**RESOLVED**`-style row to `SPEC.md` §9's Risks table for this
     finding, matching the pattern already used for every other
     multi-pass-discovered risk in that table, so the risk registry stays
     complete.
  6. (Minor, optional) Restate the `F == 0` guard in TASK-04's "Formula
     (implement exactly)" prose itself, not only in the Red-test bullet, so
     the algorithm description and the test list agree without relying on
     the reader cross-referencing both.

## Verdict rationale (this pass)

The three specifically-requested fixes are genuinely present, technically
correct, and independently re-verified against fresh source reads this pass
(not trusted from the documents' own citations) — including one additional
adversarial check this pass performed unprompted (`select_borders_greedy_
logsum`/`greedy_split`'s empty-input safety, upstream of the cited `n == 0`
guard) which also came back clean. No new *functional* defect was found in
the fixed behavior itself. However, an independent, from-scratch,
adversarial pass surfaced one new MAJOR: the fix was applied to `PLAN.md`
only, while `SPEC.md` — the acceptance-criteria document this whole review
chain has been holding to a "every requirement/risk is spec-traceable"
standard — was never updated to register the invariant, its acceptance
scenario, its risk-table entry, or its typed-error doc-comment. Per the
mandatory operating principles, a MAJOR issue found during this pass must be
reported even though it is a documentation-consistency issue rather than a
runtime-behavior defect, and the plan cannot be certified `PASS` while it
stands.

**Verdict: ISSUES_FOUND.**

---

# Extra verification pass (post-cap)

This is a fourth, explicitly-requested review of `cv-cross-validation`
(ORCH-01), beyond the spec-tdd-planner-skill's normal 3-pass cap. The prior
recorded verdict (pass 3, preserved in full below the line) was
**ISSUES_FOUND**, with one residual MAJOR and one MINOR. The orchestrator
applied two fixes directly to `PLAN.md` afterward, without an independent
4th pass. This pass (a) verifies those two fixes are genuinely present and
concretely specified, (b) independently re-verifies every load-bearing
CodeGraph claim in `SPEC.md`/`PLAN.md` from scratch (not trusting the
document's own citations), and (c) hunts adversarially for anything new.

## Verification of the two requested fixes

1. **TASK-01 systematic per-fold parity self-check.** PRESENT and concretely
   specified. `PLAN.md` TASK-01 gained a dedicated subsection ("Systematic
   parity self-check", lines 192-206) that requires, for EACH fold: calling
   `catboost.cv(..., return_models=True)` to obtain that fold's actual
   trained model, separately fitting a standalone
   `catboost.CatBoostRegressor(**params).fit(Pool(X[train_idx], y[train_idx]))`
   on the identical row subset, and asserting `model.get_borders()` (per
   feature) AND `model.get_scale_and_bias()` are IDENTICAL between the two —
   for every fold, not just the two already-known keys — with an explicit
   instruction that any newly found mismatch must be root-caused and folded
   into the pinned `params` dict BEFORE the fixture is frozen. The
   Validation section (lines 207-213) adds a concrete run-once command
   (`.venv/bin/python -c "<per-fold get_borders/get_scale_and_bias comparison
   script>"`) with an explicit pass condition ("must print zero mismatches
   across every fold"), and Completion Evidence (lines 214-217) requires the
   result to be recorded in `summary.json`. This is a concrete, executable,
   objectively-checkable task — not prose — and it is correctly sequenced
   BEFORE TASK-06 (which only consumes the frozen fixture).
2. **TASK-06 explicit builder pinning.** PRESENT. `PLAN.md` TASK-06's Red
   step (lines 524-532) now explicitly states the test calls
   `.boost_from_average(true).bootstrap_type(EBootstrapType::No)` on the
   `CatBoostBuilder` under test, with an explicit rationale citing this exact
   pass-3 MINOR finding by name. CodeGraph-confirmed (this pass, fresh read)
   that `CatBoostBuilder::bootstrap_type` (`crates/catboost-rs/src/builder.rs:211`)
   and `CatBoostBuilder::boost_from_average` (`builder.rs:197`) are real,
   `#[must_use]`, consuming-`self` builder methods with exactly this
   signature, so the cited calls will compile as written.

Both fixes are genuine, not just narrative — verdict on these two items
alone would be PASS.

## Fresh, from-scratch CodeGraph re-verification (this pass)

Every load-bearing symbol citation in `SPEC.md`/`PLAN.md` was re-queried
directly (not trusted from the documents' own citations):

- `CatBoostBuilder::new()` — `crates/catboost-rs/src/builder.rs:98-117`.
  Confirmed verbatim: `boost_from_average: true,` at line 108,
  `bootstrap_type: EBootstrapType::No,` at line 110. Setters `boost_from_average`
  (line 197) and `bootstrap_type` (line 211) exist, `#[must_use]`,
  consuming-`self`. Matches SPEC §1.1 and PLAN TASK-06 exactly.
- `cb_core::sum_f64` — DEFINED at `crates/cb-core/src/reduction.rs:32`
  (`pub fn sum_f64(values: &[f64]) -> f64`), a strict left-to-right fold, no
  compensation. 131 workspace-wide callers (CodeGraph blast-radius). Matches
  the "corrected citation" both files now use.
- `Model::ensure_scalar_oblivious` / `staged_predict` —
  `crates/catboost-rs/src/model.rs:144-171` / `:189-208`, verbatim re-read.
  Confirmed the guard rejects `approx_dimension > 1`, non-empty
  `non_symmetric_trees`/`region_trees`, and `Some(ctr_data)`, each with a typed
  `CatBoostError::UnsupportedModel`. `staged_predict` defaults
  `ntree_start`/`ntree_end`/`eval_period` to `0`/`0`/`1` exactly as SPEC/PLAN
  state, and calls `ensure_scalar_oblivious()` BEFORE touching `pool`.
- `Model::feature_columns` (`model.rs:94-107`) — checks only FLOAT-FEATURE
  COLUMN COUNT against the model (`FeatureMismatch` on mismatch), never row
  count. Confirms a load-bearing NEW finding below (zero-row pools do not
  error here).
- `cb_model::apply::predict_raw_staged` — `crates/cb-model/src/apply.rs:465-532`,
  read verbatim. `n_objects = feature_values.first().map_or(0, Vec::len)`;
  with `n_objects == 0` the per-row loop never executes and every stage's
  output `Vec<f64>` stays empty — **this function does NOT error on a
  zero-row input; it silently returns empty per-stage vectors.**
- `EvalMetric::eval` (flat, non-grouped path) — `crates/cb-train/src/metrics.rs:261-299`,
  read verbatim. Confirmed it DOES check `approx.is_empty()` and returns
  `Err(CbError::Degenerate("eval metric: empty eval set"))`. This is the
  ACTUAL source of the typed error TASK-05's `run_fold_degenerate_zero_row_errs`
  test relies on when the TEST fold is empty — not `staged_predict` itself,
  which succeeds silently on 0 rows (see above). The typed error still fires
  (via `run_fold`'s subsequent `eval_metric` call), so TASK-05's test
  will pass as designed, but the plan's own prose ("propagated from the real
  `fit`/`staged_predict` error path") mis-attributes which seam actually
  raises it — a documentation-accuracy MINOR, not a functional gap.
- `cb_train::boosting` empty-train-set guard — `crates/cb-train/src/boosting.rs:2332-2338`,
  read verbatim: `if n == 0 { return Err(CbError::Degenerate("empty target".to_owned())); }`.
  Confirms an empty TRAIN fold does error inside `fit`, as TASK-05 assumes.
- `EvalMetric::eval_grouped` / `group_spans` — `crates/cb-train/src/metrics.rs:487-574`
  / `:633-661`, read verbatim. Confirmed the non-contiguous-group-id rejection
  at lines 655-658 and the empty-approx rejection at line 501-503. Matches
  SPEC §5-S1's stated invariant for `make_cv_folds`'s grouped output.
- `cb_train::fisher_yates_permutation` (`permutation.rs:109`) and
  `cb_train::create_folds` (`fold.rs:256-330`) — both read verbatim; confirmed
  `fisher_yates_permutation` is the plain Fisher-Yates draw the plan reuses,
  and `create_folds` builds boosting learning/averaging RNG-permutation folds
  — structurally unrelated to CV train/test partitioning. The "contrast, not
  reused" framing holds.
- `Pool` / `Pool::from_validated_columns` — `crates/cb-data/src/pool.rs:37-205`,
  read verbatim. Private (`pub(crate)`) constructor at line 82, confirming
  `select_rows` (TASK-02) must live inside `cb-data`. All ten accessor names
  SPEC/PLAN cite (`n_rows`, `float_features`, `cat_features`, `text_features`,
  `embedding_features`, `label`, `weights`, `group_id`, `subgroup_id`,
  `pairs`, `baseline`) exist exactly as named.
- `catboost_rs::eval_metric` / `eval_metrics` — `crates/catboost-rs/src/metrics.rs:23-59`,
  read verbatim. `eval_metric` (singular, line 44) is a thin wrapper over
  `eval_metrics` (plural, line 23), itself a thin wrapper over
  `cb_train::calc_metrics::eval_metric`. Matches SPEC/PLAN's cited signature
  and location exactly.
- `GpuBackend` — `crates/cb-backend/src/gpu_backend.rs:58-63`, read verbatim:
  `pub struct GpuBackend { session: RefCell<Option<GpuTrainSession>> }`,
  `#[derive(Default)]`. `GpuTrainSession` (`gpu_runtime/session.rs:634-756`)
  holds `client: cubecl::client::ComputeClient<SelectedRuntime>`. Matches the
  `#[cfg(feature="cpu")]` rayon-gating rationale in SPEC §5-S5/PLAN TASK-06
  exactly.
- `rayon` dependency + `cpu` default feature — CONFIRMED:
  `crates/catboost-rs/Cargo.toml:39` (`rayon.workspace = true`), root
  `Cargo.toml:49` (`rayon = "1.12.0"`), `crates/catboost-rs/Cargo.toml:18`
  (`cpu = [...]` feature), and `default = ["cpu"]` (line 17, per pass-3's
  citation, re-confirmed present in this pass's grep). `cb-data`'s
  `Cargo.toml:12` confirms `cb-data -> cb-core` only (no reverse-dependency
  risk for `select_rows`'s placement).
- `catboost-rs-py` binding precedents — `EstimatorBase` (`estimator.rs:24-61`),
  `fit_pool` (`estimator.rs:204-206`), `data_to_pool` (`estimator.rs:236-250`),
  `make_builder` (`params.rs:451-513`) all read verbatim and match the exact
  signatures SPEC §3/PLAN TASK-07 cite, including `make_builder`'s existing
  `boost_from_average`/`bootstrap_type` param wiring (lines 497-499, 506-508).

No stale citation, incorrect signature, or misattributed ownership was found
in any load-bearing claim.

## New finding from this pass (not previously flagged)

### [MAJOR] An empty (or otherwise unvalidated) caller-supplied `folds` argument is never guarded, and can reach `aggregate_folds` with zero folds, silently producing NaN — violating the plan's own stated "never NaN" invariant

- **Plan location:** `PLAN.md` TASK-06 ("build folds from `folds` if `Some`
  else `make_cv_folds`") and TASK-04 (`aggregate_folds`'s formula and its
  only stated degenerate case, `F == 1`).
- **Requirement:** SPEC ORCH-01-S4's own invariant: *"all reductions go
  through `cb_core::sum_f64`; no panic"* and its explicit worked handling of
  `F == 1`: *"`F == 1` (a single fold) is a degenerate case with no defined
  sample std — typed error, not division by zero or NaN."* The spirit of this
  sentence is that a degenerate fold count must error, never silently emit
  NaN.
- **Evidence:**
  - `make_cv_folds` (TASK-03) validates `fold_count >= 2` — but this
    validation applies ONLY to the internally-generated-fold path.
  - `cv()`'s contract (SPEC §4 / PLAN TASK-06) accepts
    `folds: Option<&[Vec<usize>]>`, and TASK-06 states plainly: *"build folds
    from `folds` if `Some` else `make_cv_folds`"* — i.e., when the caller
    supplies `Some(&[])` (an empty `folds` list — a plausible caller
    mistake, e.g. from a dynamically computed KFold split that degenerated
    to zero splits on a tiny dataset, or an empty Python `folds=[]`), the
    internal partitioner and its `fold_count >= 2` guard are **never
    invoked**. Nothing in TASK-03/04/05/06/07 validates that the
    caller-supplied `folds` list itself is non-empty before it is run
    through `run_fold`/`aggregate_folds`.
  - TASK-04's own formula, if implemented literally as specified — `mean[i] =
    sum_f64(&[curve_f[i] for f]) / F` — evaluates to `sum_f64(&[]) / 0 =
    0.0_f64 / 0.0_f64`, which in IEEE-754 is **`NaN`, not a panic and not an
    `Err`**. TASK-04's Red step only tests `F == 1`
    (`aggregate_single_fold_errs`) and ragged lengths
    (`aggregate_ragged_errs`) — **`F == 0` is never tested, and the plan's
    own prose singles out `F == 1` as "the" degenerate case**, which is
    exactly the kind of phrasing likely to lead an implementer to add an
    `if per_fold.len() == 1 { return Err(...) }` guard and stop there,
    missing the `F == 0` case entirely.
  - CodeGraph/Read-confirmed this pass: `cb_core::sum_f64(&[])` returns
    `0.0` (the loop over an empty slice never executes, `acc` stays `0.0`) —
    `crates/cb-core/src/reduction.rs:32-38`. There is no existing
    "non-empty" guard anywhere upstream of `aggregate_folds` in the cited
    call chain for the explicit-`folds` path.
- **Failure scenario:** A Rust caller (or, via TASK-07, a Python caller of
  `catboost_rs.cv(pool, params, folds=[])`) supplies an empty explicit
  `folds` list. `cv()` skips `make_cv_folds` entirely, `run_fold` is mapped
  over zero elements (Rayon's `par_iter().map(...).collect::<Result<Vec<_>,
  _>>()` succeeds trivially on an empty iterator, producing `Ok(vec![])`),
  and `aggregate_folds(&[], metrics, iterations)` is invoked. If
  implemented per the plan's literal formula, every `test-<M>-mean` /
  `train-<M>-mean` value is `NaN` (from `0.0/0.0`), and — separately — the
  `iterations` count itself has no well-defined source (no fold curve exists
  to read a stage count from), so the very shape of `CvResult.iterations`
  is undefined for this input. Neither failure is a panic, so the workspace's
  clippy `indexing_slicing`/`unwrap_used` gates would NOT catch it; it is a
  silent-wrong-result bug, exactly the class of defect this plan's own
  Non-Goals section says the project avoids elsewhere ("never a wrong
  result").
- **Impact:** Both the Rust facade (`catboost_rs::cv`) and the Python
  surface (`catboost_rs.cv`) are affected identically (TASK-07 passes
  `folds` straight through with no additional validation). A caller who
  passes an empty explicit fold list — plausible from an upstream bug or an
  edge-case dataset — gets a `CvResult` of NaN columns / undefined shape
  instead of a typed, actionable error. Because the oracle fixture (TASK-01)
  always uses a fixed, non-empty `folds.json`, the ≤1e-5 oracle test in
  TASK-06 would NEVER exercise this path and would not catch the omission.
- **Required revision:** Add an explicit validation step to `cv()` (TASK-06):
  when `folds = Some(f)` and `f.is_empty()`, return
  `Err(CatBoostError::Train(CbError::Degenerate("cv: folds must be
  non-empty")))` BEFORE calling `run_fold`/`aggregate_folds` — mirroring the
  existing "empty `metrics`" guard already planned. Add a corresponding Red
  test (e.g. `cv_empty_explicit_folds_errs`) to TASK-06's test list.
  Additionally, `aggregate_folds` (TASK-04) should defensively guard
  `per_fold.is_empty()` as its own `Err(...)` (defense in depth, since
  `aggregate_folds` is also unit-tested directly in `cv_test.rs`
  independent of `cv()`'s own validation) with a new Red test
  (`aggregate_zero_folds_errs`) alongside the existing `aggregate_single_fold_errs`.
  Because `aggregate_folds` needs an `iterations: usize` value that, per
  TASK-04/05, is derived from an actual fold's curve length, this guard also
  resolves the otherwise-undefined "what is `iterations` when there are no
  folds" question structurally (by making the zero-fold case an error before
  that value is ever needed).

### [MINOR] TASK-05's degenerate-zero-row test description mis-attributes which seam raises the typed error

- **Plan location:** `PLAN.md` TASK-05 Red step,
  `run_fold_degenerate_zero_row_errs` bullet ("propagated from the real
  `fit`/`staged_predict` error path").
- **Evidence:** Read/CodeGraph-confirmed this pass: for an empty TEST
  sub-pool, `Model::staged_predict` does **not** error (`predict_raw_staged`,
  `cb-model/src/apply.rs:465-532`, silently returns empty per-stage vectors
  when `n_objects == 0`); the actual typed error surfaces one seam later, in
  `EvalMetric::eval`'s `approx.is_empty()` check
  (`cb-train/src/metrics.rs:267-269`), which `run_fold`'s per-stage
  `eval_metric` call triggers. For an empty TRAIN sub-pool, `fit()` does
  correctly error at `cb-train/src/boosting.rs:2336-2338`
  (`CbError::Degenerate("empty target")`), matching the plan's claim for
  that half of the test.
- **Failure scenario:** None functionally — the test as designed will still
  pass, since SOME typed error is still raised in both the empty-train and
  empty-test sub-cases. This is a documentation-accuracy issue only: an
  implementer reading "propagated from the real `fit`/`staged_predict` error
  path" might expect `staged_predict` itself to reject a zero-row pool and
  be surprised to discover (only at debug time) that it silently succeeds
  and the error actually comes from `eval_metric`.
- **Impact:** Low — cosmetic/documentation only, does not change required
  code behavior or test outcome.
- **Required revision:** Correct TASK-05's prose to say the empty-test-fold
  error is raised by the subsequent `eval_metric` call (not `staged_predict`
  itself), to avoid confusing a future implementer who inspects
  `staged_predict`'s behavior on 0 rows and finds no guard there.

## Verdict rationale

The two specifically-requested fixes are genuinely present, correctly
wired, and adequately concrete — verified independently this pass, not
merely re-read from the file's own claims. However, an independent
from-scratch adversarial pass surfaced one new MAJOR gap (unguarded
zero-length explicit `folds`, capable of silently emitting NaN in
production-facing output columns, in violation of the plan's own stated
invariant) plus one MINOR documentation-accuracy nit. Per the mandatory
operating principles, a MAJOR issue found during this pass — even though it
predates and is orthogonal to the two specific fixes requested — must be
reported, and the plan cannot be certified PASS while it stands.

**Verdict: ISSUES_FOUND.**

---

# Pass 3 (preserved verbatim from before this extra pass) — final revision verification

**Files actually changed this time: CONFIRMED.** Both artifacts contain
substantial, concrete new content relative to pass 1/pass 2 (which were
byte-identical to pass 1 and contained none of this):

- `SPEC.md` now has a new **§1.1 "Empirical resolution of the per-fold
  quantization question (closes a prior CRITICAL finding)"** subsection
  (lines 82-127) documenting the quantization-independence finding, the
  `boost_from_average`/`bootstrap_type` default-mismatch root cause, and the
  ddof=1 resolution, each with concrete mitigation text.
- `PLAN.md` TASK-01's Green step (lines 163-190) now contains a concrete
  Python params-dict code sketch with
  `"boost_from_average":True,"bootstrap_type":"No"` literally present, plus a
  narrative "Empirical finding" paragraph explaining why.
- `PLAN.md` TASK-04's formula/Red-test sections now say **sample std,
  ddof=1** (`/(F-1)`) throughout, with a worked numeric example.
- `PLAN.md` TASK-06 now has explicit `#[cfg(feature = "cpu")]` /
  `#[cfg(feature = "wgpu"/"cuda"/"rocm")]` gating text and a byte-identical
  serial-vs-parallel test.
- `SPEC.md`/`PLAN.md` cite `crates/cb-core/src/reduction.rs:32` for
  `sum_f64` consistently, with an explicit "corrected citation" note.

This is a real, substantive revision — not a repeat of pass 2's
files-unchanged failure.

**Per-item verification of the 7 flagged fixes:**

1. **`boost_from_average`/`bootstrap_type` fixture pinning** — PRESENT and
   CORRECTLY WIRED. SPEC §1.1 states the rationale (raw-dict `catboost.cv()`
   does not get the `CatBoostRegressor` Python-estimator-class default
   injection). PLAN TASK-01's Green step literally embeds
   `"boost_from_average":True,"bootstrap_type":"No"` in the concrete Python
   `params` dict passed to `catboost.cv(...)` — not merely prose. PLAN
   TASK-06's Red step explicitly notes the fixture's `params.json` now
   carries both keys and that the Rust `CatBoostBuilder`'s own defaults
   already match them (`builder.rs:108,110`), so no Rust-side override is
   needed. CodeGraph-confirmed: `crates/catboost-rs/src/builder.rs:108`
   (`boost_from_average: true,`) and `:110` (`bootstrap_type:
   EBootstrapType::No,`) inside `CatBoostBuilder::new()` — exactly as cited.
2. **ddof = sample std (ddof=1)** — PRESENT and arithmetically correct. SPEC
   §5-S4 and PLAN TASK-04 both state `sqrt(sum_f((curve_f[i]-mean)^2) /
   (F-1))`. The worked example (two folds `[1.0,2.0]`/`[3.0,4.0]`): mean =
   `[2.0, 3.0]`; at `i=0`, deviations from mean 2.0 are `(1-2)^2=1` and
   `(3-2)^2=1`, sum = 2.0, divided by `(F-1)=1` → `2.0`, `sqrt(2.0)` —
   matches the plan's stated expected value exactly. (The OLD, wrong ddof=0
   answer would have been `sqrt(2.0/2)=1.0`; the plan does NOT state that.)
3. **Rayon/GPU `#[cfg(feature="cpu")]` gating** — PRESENT. PLAN TASK-06 now
   explicitly splits `folds.par_iter().map(run_fold).collect()` under
   `#[cfg(feature = "cpu")]` from a serial `folds.iter()...` path under any
   GPU feature, with an explicit rationale citing `GpuBackend`/
   `cubecl::client::ComputeClient` concurrency being unverified
   (`gpu_backend.rs:57-63`, CodeGraph-confirmed: `GpuBackend` at
   `gpu_backend.rs:58` holds `session: RefCell<Option<GpuTrainSession>>`).
   A new `cv_serial_vs_parallel_byte_identical` test (under `cpu`) and a
   `cargo check -p catboost-rs --no-default-features --features wgpu`
   compile-check are both added to TASK-06's Validation list.
4. **Grouped k-fold + shuffle interaction** — PRESENT. SPEC §5-S1 explicitly
   states shuffle permutes GROUP-SPAN order, never within-group row order,
   with the preserved-contiguity invariant tied to `eval_grouped`'s
   `group_spans` contiguity requirement (CodeGraph-confirmed:
   `crates/cb-train/src/metrics.rs:633-659`, the non-contiguous-id rejection
   is at lines 655-658). PLAN TASK-03 adds two new Red tests
   (`grouped_multi_group_per_fold`, `shuffle_grouped_permutes_spans_not_rows`)
   beyond the original 5. PLAN TASK-06 adds a new
   `cv_grouped_ranking_test.rs` integration test exercising a ranking
   loss/metric through the full `cv()` pipeline with grouped folds.
5. **Degenerate 0-row fold test** — PRESENT. PLAN TASK-05 adds
   `run_fold_degenerate_zero_row_errs`, explicitly described as exercising
   the REAL `fit`/`staged_predict` error path (not a hand-asserted
   invariant). SPEC's acceptance-scenario table adds `AT-S3b` for the same.
6. **`fold_count`+`folds` divergence documentation** — PRESENT. SPEC §8 adds
   an explicit "Intentional divergence from upstream's combined-argument
   error behavior" paragraph citing `core.py:7164-7167`.
7. **`sum_f64` citation fix** — PRESENT and consistent. Both files now cite
   `crates/cb-core/src/reduction.rs:32` throughout (SPEC §1/§10, PLAN §0/
   TASK-04). CodeGraph-confirmed this pass: `sum_f64` is DEFINED at
   `crates/cb-core/src/reduction.rs:32` (`pub fn sum_f64(values: &[f64]) ->
   f64`); the old `crates/cb-train/src/metrics.rs:292` citation is gone from
   both files.

All 7 flagged items are confirmed present, correctly wired (not just prose),
and independently verified via fresh CodeGraph queries this pass (not trusted
from the files' own citations). One residual gap survives from pass 2's
CRITICAL #2 (see Issues below) that was not among the 7 explicitly flagged
items but was part of pass 2's required revisions and remains open.

---

## Plan Check Result (pass 3, historical)

**Verdict:** ISSUES_FOUND
**Goal:** ORCH-01 — `cv()` k-fold cross-validation surface: partition a `Pool` into disjoint train/test folds, train+staged-evaluate each fold via existing seams, and aggregate per-iteration `test-<M>-mean/std` / `train-<M>-mean/std` columns, oracle-locked ≤1e-5 against `catboost.cv(..., folds=<fixed>, shuffle=False)` for a scalar/oblivious/float-only (numeric-regression) first slice.
**Plan:** `/home/user/Documents/workspace/catboost_rs/.planning/plans/cv-cross-validation/PLAN.md`, `/home/user/Documents/workspace/catboost_rs/.planning/plans/cv-cross-validation/SPEC.md`

### Summary
- The pass-1 BLOCKER/CRITICAL (`boost_from_average`/`bootstrap_type` default mismatch, discovered by pass 2) is now genuinely fixed: SPEC §1.1 documents the root cause and mitigation, and PLAN TASK-01's Green step embeds the fix as concrete Python code (`"boost_from_average":True,"bootstrap_type":"No"` in the literal `params` dict), not merely prose. `CatBoostBuilder::new()`'s defaults (`builder.rs:108,110`) were re-confirmed via CodeGraph to already match, so TASK-06 correctly does not need a Rust-side override.
- ddof=1 (sample std) is correctly pinned and the worked arithmetic example in TASK-04 is correct.
- Both pass-1 MAJOR gaps (rayon/GPU-backend concurrency safety; grouped-k-fold + shuffle interaction and missing ranking end-to-end test) are now closed with concrete `#[cfg(feature="cpu")]` gating + a byte-identity test + a `--features wgpu` compile check, and new grouped/shuffle unit tests + a `cv_grouped_ranking_test.rs` integration test, respectively.
- Both pass-1 MINOR gaps (fold_count+folds divergence undocumented; `sum_f64` mis-cited) are closed.
- One residual gap remains from pass 2's CRITICAL #2: pass 2 required BOTH (a) writing the quantization-independence finding into SPEC (done, §1.1) AND (b) adding a concrete Green-step task in TASK-01 that diffs `get_scale_and_bias()`/`get_borders()` (or otherwise cross-checks) between a `return_models=True` cv fold model and a standalone-equivalent fit on the same row subset, to catch any FURTHER hidden `CatBoostBuilder`-default divergence beyond the one now-confirmed `boost_from_average`/`bootstrap_type` pair. Only (a) was done; (b) is still absent from TASK-01's concrete Green/Validation steps (it exists only as narrative about the ONE mismatch already found, not as a forward-looking self-check task). This is downgraded from pass 2's CRITICAL to MAJOR here because TASK-06's own ≤1e-5 oracle assertion still provides a comprehensive (if late) safety net for any further hidden mismatch — the risk is inefficient/late discovery, not a confirmed defect, but it was explicitly required by the prior pass and is not yet closed.
- Given one MAJOR issue remains open (an explicit prior-pass required revision only half-implemented), this plan cannot yet return PASS, though it has made substantial, verified progress on the BLOCKER and all four other MAJOR/MINOR items from pass 1/2.
- **[Superseded by the extra verification pass above]** Both the MAJOR (TASK-01 forward-looking self-check) and MINOR (TASK-06 explicit builder pinning) items below were subsequently fixed and independently re-verified as genuinely present in the "Extra verification pass (post-cap)" section at the top of this document. A NEW MAJOR issue (unguarded empty explicit `folds`) was found in that pass instead. Treat the section at the top of this document, not this historical block, as authoritative for current status.

### Specification Coverage
- [x] ORCH-01-S1 fold-index partitioning (`make_cv_folds`): TASK-03, `cv_test.rs`. `fisher_yates_permutation` CodeGraph-confirmed at `crates/cb-train/src/permutation.rs:109`; `create_folds` CodeGraph-confirmed at `crates/cb-train/src/fold.rs:256-330` as a structurally distinct boosting learning/averaging-fold builder (NOT the CV partitioner) — SPEC's contrast framing is accurate.
- [x] ORCH-01-S1 grouped variant + `shuffle` interaction: SPEC §5-S1 now pins the shuffle×group semantics; TASK-03 adds `grouped_multi_group_per_fold` and `shuffle_grouped_permutes_spans_not_rows`. Resolved.
- [x] ORCH-01-S2 `Pool::select_rows`: TASK-02, `crates/cb-data/src/pool.rs`. CodeGraph-confirmed `Pool`'s private `from_validated_columns` ctor and accessors at `pool.rs:69-205`.
- [x] ORCH-01-S3 per-fold train + staged eval AND bias-term parity with upstream: TASK-05 (`run_fold`) + TASK-01's fixture pinning (the parity fix). CodeGraph-confirmed `Model::staged_predict`/`ensure_scalar_oblivious` at `crates/catboost-rs/src/model.rs:144,189` and `CatBoostBuilder::fit`/`boost_params` construction at `builder.rs:65-320+`.
- [x] ORCH-01-S4 cross-fold mean/std aggregation: TASK-04, routes through `cb_core::sum_f64`, CodeGraph-confirmed DEFINED at `crates/cb-core/src/reduction.rs:32`; citation now correct in both files. **[Extra pass: F==0 explicit-folds case still unguarded — see new MAJOR above.]**
- [x] ORCH-01-S5 Rust facade `cv` + oracle: TASK-01 (fixtures) + TASK-06.
- [x] ORCH-01-S5 rayon-parallel fold safety: TASK-06 now gates to `#[cfg(feature="cpu")]` with a documented rationale, a byte-identity test, and a `--features wgpu` compile check. Resolved.
- [x] ORCH-01-S6 Python `catboost_rs.cv`: TASK-07.
- [x] Non-goals correctly scoped.
- [x] Full parameter-by-parameter `CatBoostBuilder`-default-vs-raw-`cv()`-default parity beyond `boost_from_average`/`bootstrap_type`: now verified by TASK-01's systematic per-fold self-check (see extra pass above).

### CodeGraph Evidence
- `CatBoostBuilder::new()` — `crates/catboost-rs/src/builder.rs:93-117` (fresh Read this pass). `boost_from_average: true,` at line 108; `bootstrap_type: EBootstrapType::No,` at line 110. Both CONFIRM the SPEC/PLAN's citations exactly. `boost_from_average` setter at line 197, `bootstrap_type` setter at line 211 (both `#[must_use]` consuming builder methods).
  - Impact assessment: the fixture's `params` dict now matches these defaults explicitly, closing the confirmed ~0.17-scale divergence. No further Rust-side change needed for this specific pair.
- `cb_core::sum_f64` — `crates/cb-core/src/reduction.rs:32` (`pub fn sum_f64(values: &[f64]) -> f64`), 131 callers workspace-wide (CodeGraph blast-radius). Both SPEC and PLAN now cite this location consistently; the stale `metrics.rs:292` citation is gone.
- `GpuBackend` — `crates/cb-backend/src/gpu_backend.rs:57-63` (`#[derive(Default)] pub struct GpuBackend { session: RefCell<Option<GpuTrainSession>> }`). CodeGraph-confirmed this pass. Matches SPEC/PLAN's citation and rationale for the `#[cfg(feature="cpu")]` gate.
- `EvalMetric::eval_grouped` / `group_spans` — `crates/cb-train/src/metrics.rs:487-574` (`eval_grouped`) and `:628-661` (`group_spans`). Contiguity check confirmed at lines 654-658 (`seen.windows(2).any(...) => Err(CbError::Degenerate("... not contiguous ..."))`). Matches the invariant SPEC §5-S1/PLAN TASK-03 now state explicitly for `make_cv_folds`'s grouped-mode output.
- `cb_train::create_folds` — `crates/cb-train/src/fold.rs:256-330`, confirmed to build boosting learning/averaging-fold RNG-permutation machinery for ordered boosting/CTR — structurally distinct from disjoint CV train/test partitioning. SPEC's "contrast, not reused" framing remains accurate.
- `Model::staged_predict` / `ensure_scalar_oblivious` — `crates/catboost-rs/src/model.rs:144-171` (guard rejecting `approx_dimension>1`, non-symmetric/region trees, CTR data with typed `CatBoostError::UnsupportedModel`), `:173+` (`staged_predict`, confirmed present per blast-radius listing with 3 callers and a covering test `staged_predict_facade_test.rs`).
- `catboost_rs::eval_metric` / `eval_metrics` — `crates/catboost-rs/src/metrics.rs:23-59`, confirmed as thin non-panicking wrappers over `cb_train::calc_metrics::eval_metric` (`crates/cb-train/src/calc_metrics.rs:296-332`). Matches SPEC/PLAN's citations exactly.
- `rayon` dependency — CONFIRMED: `crates/catboost-rs/Cargo.toml:39` (`rayon.workspace = true`) and root `Cargo.toml:49` (`rayon = "1.12.0"`). Also confirmed `crates/catboost-rs/Cargo.toml:17` (`default = ["cpu"]`), i.e. `cpu` is the DEFAULT feature — so an ordinary `cargo test` exercises the new parallel path, and the `--features wgpu` compile-check exercises the serial path, exactly as PLAN TASK-06 validation intends.
- `Pool` — `crates/cb-data/src/pool.rs:37-205` (struct + private `from_validated_columns` ctor at line 82 + accessors through line 205). Matches SPEC/PLAN's `pool.rs:82-205` citation exactly.

### Issues (pass 3, historical — see extra pass above for current status)

#### [MAJOR — RESOLVED per extra pass] Pass-2 CRITICAL #2's forward-looking self-check task is still missing from TASK-01 — only the retrospective narrative was written in
- **Plan location:** PLAN.md TASK-01 ("Green (generation intent)" and "Validation (offline, run-once/commit)" sections).
- **Requirement:** Pass 2's CRITICAL #2 required revision (verbatim from `PLAN-CHECK.md` pass 2): *"Write the quantization-independence finding into SPEC §1.1/§9 explicitly... **and add a Green-step check in TASK-01 that diffs `get_scale_and_bias()`/`get_borders()` between a `return_models=True` fold model and a standalone fit on the same row subset, to catch any further hidden default mismatch beyond `boost_from_average`.**"*
- **Evidence:** SPEC §1.1 now documents the quantization-independence finding (first half of the requirement — DONE). PLAN TASK-01's Green step, however, only narrates what was ALREADY discovered about `boost_from_average`/`bootstrap_type` (the ONE mismatch already found and now pinned); it contains no concrete generator step, script assertion, or Validation-list item that diffs `get_scale_and_bias()`/`get_borders()` (or any other parameter) between a `return_models=True` cv fold model and a standalone-equivalent fit for the frozen fixture. TASK-01's "Validation (offline, run-once/commit)" list is: venv setup, run `gen_cv_fixtures.py`, and a shape/finiteness sanity check on the `.npy` outputs — no default-parity diff-check is present. This matches pass 2's own "Unverified Items" entry, which is still unverified: *"Whether any OTHER `CatBoostBuilder` default beyond `boost_from_average`... also diverges between the raw `catboost.cv()` native-call default and the Python-estimator-class convenience default — only `boost_from_average` was isolated and confirmed in this session; a full parameter-by-parameter diff... has not been performed."*
- **Failure scenario:** If some other `CatBoostBuilder` default (e.g. `score_function`, `leaf_method`, `l2_leaf_reg`, `random_strength`) also happens to mirror the Python-estimator-class convenience default rather than the raw native `_cv()` default — the exact same class of bug `boost_from_average` was — TASK-06's oracle test will fail with an unexplained divergence, and the implementer will need to re-run the same forensic investigation (return_models=True + get_scale_and_bias/get_borders comparison) that took two full plan-check passes to complete for `boost_from_average`, this time discovered only after TASK-02 through TASK-05 are already implemented.
- **Impact:** Repeats the exact "late discovery" risk pattern this plan has already hit twice (quantization sharing, then bootstrap/boost_from_average) for any remaining unverified parameter. Not a confirmed defect — TASK-06's ≤1e-5 oracle assertion will still catch it if it exists — but the explicit process safeguard the prior pass required to avoid a THIRD round of this exact failure mode was not implemented.
- **Required revision:** Add a concrete step to TASK-01's Green/Validation flow: after generating the fixture, run `catboost.cv(pool, params, folds=<fixed>, shuffle=False, return_models=True)`, take fold 0's returned model, separately fit a standalone model with the SAME `params` dict on the identical fold-0 train-row subset (mirroring exactly what `CatBoostBuilder`/`run_fold` will do), and assert `get_scale_and_bias()` matches (and, ideally, that `get_borders()` / a raw-prediction comparison on the test subset also matches to a tight tolerance). Record the result in `summary.json`. This closes the parity-verification gap BEFORE TASK-06 is written, rather than relying solely on TASK-06's oracle test to surface it late.
- **Status: RESOLVED.** Independently re-verified present and concrete in the extra verification pass above.

#### [MINOR — RESOLVED per extra pass] TASK-06's oracle test relies on `CatBoostBuilder::new()`'s implicit defaults for `boost_from_average`/`bootstrap_type` rather than setting them explicitly
- **Plan location:** PLAN.md TASK-06 Red step ("the builder's own `boost_from_average`/`bootstrap_type` DEFAULTS already match... so no override is needed on the Rust side").
- **Evidence:** CodeGraph-confirmed `CatBoostBuilder::new()` sets `boost_from_average: true` (`builder.rs:108`) and `bootstrap_type: EBootstrapType::No` (`builder.rs:110`) as of today. The oracle test's correctness is silently coupled to these two literals never changing.
- **Failure scenario:** A future, unrelated change to `CatBoostBuilder::new()`'s defaults (e.g. to track a new upstream catboost version) would break the cv oracle test in a way that looks like a cv-specific regression, when it is actually a builder-default change with no connection to cv's own logic.
- **Impact:** Low — the test would fail loudly (not silently pass with wrong values), so this is a debuggability/attribution nicety, not a correctness risk.
- **Required revision:** In TASK-06's oracle test, explicitly call `.boost_from_average(true).bootstrap_type(EBootstrapType::No)` on the `CatBoostBuilder` under test (even though currently redundant with the defaults), with a comment noting this pins the fixture-parity assumption independent of `CatBoostBuilder::new()`'s future defaults.
- **Status: RESOLVED.** Independently re-verified present in the extra verification pass above.

### Implementation Order Review
1. `TASK-01 ∥ TASK-02 ∥ TASK-03 → TASK-04 → TASK-05 → TASK-06 → TASK-07` remains acyclic and each task's prerequisites are produced before consumption. The BLOCKER that previously invalidated this ordering (TASK-01/TASK-05/06 disagreeing on `boost_from_average`) is resolved — the ordering is now sound with respect to the confirmed defect.
2. The new MAJOR (empty-`folds` guard) fits naturally as an additional Red/Green step inside TASK-06 (`cv()`'s own validation) plus a defensive guard inside TASK-04 (`aggregate_folds`) — it does not require reshaping the wave structure or introducing a new task.
3. TASK-03/04/05 sequentially modifying `cv.rs`, and TASK-06 depending on TASK-01+TASK-05, TASK-07 depending on TASK-06, all remain correctly sequenced.

### Potential Bugs
- **Unguarded empty explicit `folds` (NEW, this pass)** — trigger: caller passes `folds=Some(&[])` (Rust) or `folds=[]` (Python); failure mode: `aggregate_folds` silently computes `0.0/0.0 = NaN` for every output column, and `CvResult.iterations`'s length is undefined since no fold curve exists to derive it from; path: `cv()` → (skips `make_cv_folds`) → `run_fold` mapped over zero elements → `aggregate_folds(&[], ...)`; mitigation: the required TASK-06/TASK-04 guards above.
- **Hidden `CatBoostBuilder`-default divergence beyond `boost_from_average`/`bootstrap_type`** — CLOSED (TASK-01's systematic per-fold `get_borders`/`get_scale_and_bias` self-check, independently re-verified present this pass).
- **Grouped-mode + `shuffle=true`** — CLOSED (SPEC §5-S1 pinned; TASK-03/06 new tests added).
- **Concurrent GPU-backend invocation under rayon-parallel folds** — CLOSED (TASK-06 `#[cfg(feature="cpu")]` gate + byte-identity test + `wgpu` compile check).
- **Empty-fold (0-row train/test block, from an internally-generated fold)** — CLOSED (TASK-05 `run_fold_degenerate_zero_row_errs` exercises the real error path; this pass additionally confirmed via CodeGraph exactly WHICH seam raises the error in each half — see the MINOR documentation-accuracy note above).

### Required Plan Revisions
1. **(NEW, this pass, MAJOR)** Add an explicit non-empty validation for the caller-supplied `folds` argument in `cv()` (TASK-06), plus a defensive `per_fold.is_empty()` guard in `aggregate_folds` (TASK-04), each returning a typed `CatBoostError::Train(CbError::Degenerate(...))` rather than silently computing NaN. Add `cv_empty_explicit_folds_errs` (TASK-06) and `aggregate_zero_folds_errs` (TASK-04) Red tests.
2. **(NEW, this pass, MINOR)** Correct TASK-05's prose describing which seam (`eval_metric`, not `staged_predict`) actually raises the typed error for an empty TEST sub-fold.
3. ~~Add the `get_scale_and_bias()`/`get_borders()` parity self-check to TASK-01~~ — DONE, independently re-verified this pass.
4. ~~Make TASK-06's `CatBoostBuilder` construction explicitly pin `boost_from_average`/`bootstrap_type`~~ — DONE, independently re-verified this pass.

### Unverified Items
- Whether `catboost.cv(..., folds=<explicit fixed list>, return_models=True)` actually returns per-fold models when `folds=` is passed explicitly (as opposed to `fold_count`-driven auto-partitioning) — this is assumed by TASK-01's new systematic parity self-check. Not independently re-run in this session (no live `catboost` install available); flagged here as a residual runtime-behavior assumption rather than a plan defect, since the same API usage pattern was already used successfully in the pass-2/pass-3 empirical investigation for the two already-confirmed defaults.
- Whether concurrent multi-thread use of `cubecl::client::ComputeClient<SelectedRuntime>` (`wgpu`/`cuda`/`rocm`) is actually unsafe, or merely unverified — no GPU hardware was exercised this session; the plan's decision to scope parallel fold execution to `cpu` only is a reasonable, conservative mitigation regardless of the answer, so this does not block the plan, only documents a design choice.
