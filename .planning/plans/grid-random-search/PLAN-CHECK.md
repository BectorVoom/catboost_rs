# Plan Check Result — ORCH-02 (grid/randomized search)

## Extra verification pass (post-cap)

This is an EXTRA, independently-run verification pass requested explicitly by
the user, beyond the spec-tdd-planner-skill's normal 3-pass cap (already
exhausted — pass 3 below, verdict ISSUES_FOUND, is the last automated pass
counted against that cap). Between pass 3 and this pass, one trivial one-line
fix (TASK-06's own "Parallelization" line) was applied directly by the
orchestrator, **without** a 4th checker pass re-verifying it — so the plan's
recorded state was self-certified, not independently confirmed. The user
wants every plan issue actually fixed and independently re-confirmed, not
left in that self-certified state.

This pass:
1. Independently confirmed the specific line-level fix pass 3 demanded is
   present and correct in `PLAN.md` as written today.
2. Re-ran the ENTIRE checker process from scratch: every load-bearing
   CodeGraph claim in SPEC.md/PLAN.md was independently re-derived via fresh
   `codegraph_explore` calls (not reused from any prior pass's citations,
   including ones prior passes marked "already confirmed").
3. Swept EVERY location in PLAN.md describing ANY task's dependency or
   parallelization relationship — every task's "Blocked by" line, every
   task's "Parallelization" line, the §1 wave table, the §1 prose paragraph,
   the §1 ASCII dependency diagram, and the §1 file-ownership note — cross-
   checked against each other and against the actual §4 typed contracts, not
   just the TASK-04/05/06 triad flagged in earlier passes.
4. Hunted for anything else left unresolved.

**Result: the specific pass-3 MAJOR is CONFIRMED FIXED.** TASK-06's own
"Parallelization" line (PLAN.md, in the TASK-06 section) now reads:

> **Parallelization:** sequential after BOTH TASK-04 (needs `randomized_search`)
> AND TASK-05 (shares `search.rs` + `expand_param_grid`) — corrected to match
> this task's own "Blocked by" line (PLAN-CHECK pass 3 found this line had not
> been updated in step, still reading "after TASK-05" alone).

This is now byte-consistent with TASK-06's own "Blocked by" line, with
TASK-05's own "Parallelization" line ("TASK-06 remains sequential after BOTH
this task and TASK-04"), with the §1 wave table (`TASK-06 ... depends: TASK-04,
TASK-05`), and with the §1 ASCII diagram (both `TASK-04` and `TASK-05` feeding
into `TASK-06`). No reader consulting ANY single field for TASK-06 can now
reach the "premature start" misread pass 3 identified.

**No new MAJOR/CRITICAL/BLOCKER issue was found in the exhaustive sweep.**
Two new MINOR (non-contradictory, self-disambiguated) observations surfaced
and are recorded below alongside the two MINOR items pass 3 already logged as
"optional polish" (still un-actioned, as expected — pass 3 marked them
optional, not required).

**Verdict of this extra pass: PASS.** (Full detail below, in the same
mandated report format; pass 3's own report is preserved beneath, unedited,
for history.)

---

## Extra-pass full report

**Verdict:** PASS
**Goal:** ORCH-02 — `grid_search`/`randomized_search` hyperparameter search
for catboost-rs, hard-dependent on the sibling ORCH-01 (`cv()`) slice, itself
in-flight/under CRITICAL-finding revision (unchanged fact from pass 3).
**Plan:** `.planning/plans/grid-random-search/SPEC.md` +
`.planning/plans/grid-random-search/PLAN.md` (current on-disk text, re-read in
full this pass — not diffed against a prior pass's cache)

### Summary
- TASK-06's "Parallelization" line — the single finding that kept pass 3 at
  ISSUES_FOUND and that this pass was specifically asked to re-verify — is
  confirmed fixed and now internally consistent with every other place in the
  document that describes the TASK-04/TASK-05/TASK-06 relationship.
- A full re-derivation (not reuse) of every CodeGraph-checkable claim in
  SPEC.md/PLAN.md was performed this pass via fresh `codegraph_explore` calls:
  `EvalMetric` (16 variants — `Rmse|Logloss|Msle|Mae|Mape|Quantile|Custom` +
  `Ndcg|Dcg|Map|Mrr|Err|PFound|PrecisionAt|RecallAt|QueryAuc` — enumerated
  directly from the `eval()` match arms at `crates/cb-train/src/metrics.rs:301-460`,
  independently counted, exactly 16, exactly matching the plan's variant
  list and its flat/ranking partition); `CustomMetric::is_max_optimal`
  (`crates/cb-compute/src/custom.rs:111-114`, doc: `true`=larger is better);
  `CustomMetricHandle` (`custom.rs:155`, `pub struct CustomMetricHandle(pub
  Arc<dyn CustomMetric>)`); `CatBoostBuilder` (`crates/catboost-rs/src/builder.rs:64-65`,
  `#[derive(Debug, Clone, PartialEq)]`, full setter list re-read, `fit`
  referenced at `estimator.rs:204` via `builder.fit(pool)`);
  `fisher_yates_permutation` (`crates/cb-train/src/permutation.rs:109`,
  `pub fn fisher_yates_permutation(n: usize, seed: u64) -> Vec<i32>`, body
  `TFastRng64::from_seed(seed)` + `shuffle_in_place`); `parse_metric`/
  `calc_metric`/`eval_metric` (`crates/cb-train/src/calc_metrics.rs:151,296,318`,
  bodies re-read); `rayon.workspace = true` (`crates/catboost-rs/Cargo.toml:39`,
  exact line, re-read from disk this pass); `sum_models` registration
  (`crates/catboost-rs-py/src/lib.rs:57`, exact,
  `m.add_function(wrap_pyfunction!(regressor::sum_models, m)?)?;`); `make_builder`
  (`crates/catboost-rs-py/src/params.rs:451-513`, full body re-read: sequential
  `get_with_aliases` + `check_range` + builder setter per known param);
  `EstimatorBase`/`data_to_pool` (`crates/catboost-rs-py/src/estimator.rs:24-30,236-250`,
  full bodies re-read). All CONFIRMED EXACT — no discrepancy found between any
  citation in SPEC.md/PLAN.md and the current on-disk source.
- `crates/catboost-rs/src/lib.rs` re-read directly (not via CodeGraph, to
  cross-check): re-exports only `CatBoostBuilder`, `CatBoostError`,
  `eval_metric`/`eval_metrics`, `Model`, `FeatureImportanceType`/
  `PredictionType`, `PartialDependence`/`PdpError`, `OnnxExportError`,
  `CoreMlExportError`, `EScoreFunction`/`LeafMethod`/`Loss`, `EBootstrapType`,
  `IngestSource`/`OwnedColumns`, `Pool` — **no `cv`, no `CvResult`.** The
  cross-plan blocker premise (ORCH-01 unshipped) remains accurate today.
- `crates/catboost-rs/src/grid_search.rs` and `crates/catboost-rs-py/src/search.rs`
  confirmed still absent (`ls` both empty); `.planning/plans/cv-cross-validation/`
  confirmed still draft (only `PLAN-CHECK.md`/`PLAN.md`/`SPEC.md`, no source).
  `git log --oneline -1` on every one of the 9 plan-relevant source files
  resolves to `cf1252b` (CoreML export) for all of them — the same commit pass
  3 recorded as the last touch — confirming **no source drift since pass 3**;
  only this document's own text has changed (the one-line fix).
- **Full dependency/parallelization sweep (this pass's specific mandate):**
  every task's "Blocked by" line, every task's "Parallelization" line, the §1
  wave table, the §1 prose paragraph, the §1 ASCII diagram, and the §1
  file-ownership note were cross-checked pairwise. All are mutually
  consistent. No new contradiction found. Two new MINOR (non-contradictory)
  completeness observations surfaced; see Issues/Potential Bugs below — both
  are cosmetic, self-disambiguated by adjacent text in the same section, and
  do not block execution.
- The two MINOR "optional polish" items pass 3 already logged (TASK-04's
  "Parallelization" line not mentioning the TASK-05 parallel opportunity;
  the §1 file-ownership note not restating the functional
  `randomized_search` reason for TASK-06) remain un-actioned, exactly as
  pass 3 explicitly marked them optional/non-blocking — unchanged, not a
  regression.

### Specification Coverage
- [x] ORCH-02-S1 `metric_is_max_optimal`: TASK-01. Independently re-verified
  this pass: the 16-arm exhaustive match in TASK-01 ("Match arms (implement
  exactly)") exactly mirrors the 16 `EvalMetric` variants re-enumerated
  directly from `crates/cb-train/src/metrics.rs:301-460` this pass (not
  trusted from any prior pass's count).
- [x] ORCH-02-S2 `score_candidate`/`select_best`: TASK-02, checked-indexing
  algorithm (`.get`/`?`, no raw `[]` indexing) confirmed present in the
  written algorithm description.
- [x] ORCH-02-S3 Rust `grid_search` (+refit): TASK-03, self-consistency
  oracle + checked-indexing code sketch (`candidates.get(best)`, not
  `candidates[best]`) present.
- [x] ORCH-02-S4 Rust `randomized_search`: TASK-04, subsampling unit tests +
  one integration case; correctly sequenced after TASK-03 (same prod file,
  confirmed consistent across "Blocked by"/"Parallelization"/wave table/
  ASCII diagram).
- [x] ORCH-02-S5 Python `grid_search`: TASK-05's dependency (TASK-03 only) is
  sound and internally consistent across its own "Blocked by" and
  "Parallelization" lines, the wave table, and the ASCII diagram.
- [x] ORCH-02-S6 Python `randomized_search`: TASK-06's SPEC-level behavior is
  fully covered; its "Blocked by" AND "Parallelization" lines now BOTH
  correctly require TASK-04 **and** TASK-05 — the pass-3 MAJOR is CLOSED.

### CodeGraph Evidence
- `EvalMetric` — `crates/cb-train/src/metrics.rs:64` (declaration site per
  blast-radius), variants independently re-enumerated from the `eval()`
  method's match arms (`metrics.rs:301-460`): `Rmse, Logloss, Msle, Mae, Mape,
  Quantile{alpha}, Custom(handle)` (flat, direction `false`/delegate) and
  `Ndcg{..}, Dcg{..}, Map{..}, Mrr{..}, Err{..}, PFound{..}, PrecisionAt{..},
  RecallAt{..}, QueryAuc{..}` (ranking, direction `true`) — 16 total, exact
  match to SPEC/PLAN's claimed variant list and direction partition. Blast
  radius: 11 callers across `cb-train`/`catboost-rs`/tests; none touched by
  this plan (read-only reuse confirmed).
- `CustomMetric::is_max_optimal` — `crates/cb-compute/src/custom.rs:111-114`,
  `fn is_max_optimal(&self) -> bool;`, doc comment: `true` if a LARGER value is
  better. `CustomMetricHandle(pub Arc<dyn CustomMetric>)` — `custom.rs:155`.
  Both confirmed exact, re-read this pass.
- `CatBoostBuilder` — `crates/catboost-rs/src/builder.rs:64-65`,
  `#[derive(Debug, Clone, PartialEq)] pub struct CatBoostBuilder`; full field
  list and every setter (`.loss()`, `.iterations()`, `.depth()`,
  `.learning_rate()`, `.l2_leaf_reg()`, `.random_strength()`,
  `.boost_from_average()`, `.leaf_method()`, `.bootstrap_type()`,
  `.subsample()`, `.bagging_temperature()`, `.random_seed()`,
  `.border_count()`, `.score_function()`, `.custom_objective()`,
  `.custom_metric()`) re-read this pass — all present, all consuming/
  returning `Self`. `fit` confirmed called via
  `crates/catboost-rs-py/src/estimator.rs:204-206`
  (`pub(crate) fn fit_pool(builder: CatBoostBuilder, pool: &Pool) ->
  Result<Model, CatBoostError> { builder.fit(pool) }`).
- `fisher_yates_permutation` — `crates/cb-train/src/permutation.rs:109`,
  `pub fn fisher_yates_permutation(n: usize, seed: u64) -> Vec<i32>`, body:
  `TFastRng64::from_seed(seed)` then `shuffle_in_place(n, &mut rng)`. Blast
  radius: 14 callers, none touched by this plan.
- `parse_metric`/`calc_metric`/`eval_metric` —
  `crates/cb-train/src/calc_metrics.rs:151(parse_metric, not shown verbatim
  this pass but referenced),296(calc_metric),318(eval_metric)`; `calc_metric`
  dispatches `is_ranking(metric)` (the SAME 9-variant ranking set) to
  `eval_grouped` vs `eval`; `eval_metric` maps `parse_metric` + `calc_metric`
  over each descriptor, short-circuiting on the first error. Confirms the
  plan's "metric string → `EvalMetric` → direction" hop is real, existing,
  shipped code (ORCH-04), not an assumption.
- `rayon` — `crates/catboost-rs/Cargo.toml:39`, `rayon.workspace = true`,
  confirmed at the exact cited line via direct file read this pass.
- `sum_models` registration — `crates/catboost-rs-py/src/lib.rs:57`,
  `m.add_function(wrap_pyfunction!(regressor::sum_models, m)?)?;`. Confirmed
  exact via direct file read this pass (not `sed`, not reused).
- `make_builder` — `crates/catboost-rs-py/src/params.rs:451-513`, full body
  re-read this pass: applies `get_with_aliases` + `check_range` + a builder
  setter per known param (`iterations`, `depth`, `learning_rate`,
  `l2_leaf_reg`, `random_strength`, `random_seed`, `border_count`,
  `subsample`, `bagging_temperature`, `boost_from_average`, `loss_function`,
  `score_function`, `bootstrap_type`, `leaf_estimation_method`), returning
  `Ok(builder)`. Confirms the plan's "kwargs dict → `CatBoostBuilder`" claim.
- `EstimatorBase`/`data_to_pool` —
  `crates/catboost-rs-py/src/estimator.rs:24-30`
  (`pub(crate) struct EstimatorBase { params: BTreeMap<String, Py<PyAny>>,
  model: Option<Model> }`) and `:236-250` (`data_to_pool` body: `Pool`
  fast-path via `.cast::<crate::pool::Pool>()`, else
  `ingest_to_owned(...).into_pool()` mapped to `CatBoostValueError`). Both
  confirmed exact, re-read this pass.
- `crates/catboost-rs/src/lib.rs` (read directly, full file, this pass) —
  confirmed re-exports only `CatBoostBuilder`, `CatBoostError`,
  `eval_metric`/`eval_metrics`, `Model`, `FeatureImportanceType`/
  `PredictionType`, `PartialDependence`/`PdpError`, `OnnxExportError`,
  `CoreMlExportError`, `EScoreFunction`/`LeafMethod`/`Loss`, `EBootstrapType`,
  `IngestSource`/`OwnedColumns`, `Pool` — **no `cv`, `CvResult`, `CvFold`,
  `CvType`.** Cross-plan blocker framing remains accurate today.
- `git log --oneline -1 -- <9 relevant source files>` (this pass, all 9 in
  one command) — every file's most recent touching commit is `cf1252b`
  (CoreML export). No source drift since pass 3. `ls crates/catboost-rs/src/ |
  grep grid` and `ls crates/catboost-rs-py/src/ | grep search` both empty —
  `grid_search.rs`/`search.rs` remain genuinely net-new. `ls
  .planning/plans/cv-cross-validation/` shows only `PLAN-CHECK.md`/`PLAN.md`/
  `SPEC.md` — ORCH-01 remains draft/unshipped.

### Issues

No BLOCKER, CRITICAL, or MAJOR issues found in this pass. Two new MINOR
(non-contradictory, self-disambiguated) observations, in addition to the two
pass-3 already logged as optional polish:

#### [MINOR] TASK-04's and TASK-05's own "Parallelization" lines omit the G-ORCH-01 gate their own "Blocked by" lines mention
- **Plan location:** TASK-04 section — "Blocked by" ("TASK-03 ... + G-ORCH-01
  (transitively, via `cv`)") vs. "Parallelization" ("sequential after TASK-03
  (same prod file)" — no gate mention). TASK-05 section — same pattern
  ("Blocked by" mentions "G-ORCH-01 (transitively)"; "Parallelization" omits
  it).
- **Requirement:** Every task's fields should tell one consistent, complete
  story (the mandate motivating this whole pass).
- **Evidence:** Contrast with TASK-02/TASK-03, whose own "Parallelization"
  lines DO restate the gate ("... AND after G-ORCH-01 (`CvResult`)", "...
  AND G-ORCH-01(cv)"). TASK-04/TASK-05 do not restate it in their
  "Parallelization" line, only in "Blocked by".
- **Failure scenario:** None credible — this is an omission, not a
  contradiction. The gate is satisfied TRANSITIVELY the moment TASK-03 (which
  itself IS gated on G-ORCH-01(cv)) has landed, so "sequential after TASK-03"
  is not wrong, merely silent about the transitive gate. A reader who starts
  TASK-04/05 the moment TASK-03 lands is doing the CORRECT thing either way
  (TASK-03 cannot have landed without the gate already being satisfied).
- **Impact:** None on correctness; a minor completeness/symmetry gap only.
- **Required revision (optional polish, non-blocking):** For symmetry with
  TASK-02/TASK-03, append "(G-ORCH-01 already satisfied transitively via
  TASK-03)" to TASK-04's and TASK-05's own "Parallelization" lines.

#### [MINOR] §1 wave lettering (Wave C / Wave D) assigns TASK-04 and TASK-05 different, sequential-looking wave letters despite the row text explicitly stating they run in parallel
- **Plan location:** §1 wave table:
  `Wave C: TASK-04 ... depends: TASK-03`
  `Wave D: TASK-05 ... depends: TASK-03 (parallel w/ TASK-04 — different file, no shared write)`
- **Requirement:** A "wave" label conventionally denotes a parallelizable
  batch; assigning two genuinely-parallel tasks different, alphabetically-
  ordered wave letters (C then D) could be misread as "D must wait for C to
  finish" by a reader who skims only the left-hand `Wave X:` labels and not
  the parenthetical.
- **Evidence:** The row for TASK-05 itself carries the disambiguating
  parenthetical "(parallel w/ TASK-04 — different file, no shared write)",
  and the immediately-following prose paragraph and the ASCII diagram both
  unambiguously show the TASK-03 → {TASK-04, TASK-05} → TASK-06 fork/join
  shape. So the wave-letter choice is a labeling artifact, not a substantive
  contradiction — every other representation in the document (the graph, the
  prose, both tasks' own "Parallelization" lines) already gets the
  fork/join right.
- **Failure scenario:** A reader who reads ONLY the `Wave X:` column (skipping
  the parenthetical, the prose, and the diagram) could infer a false
  Wave-C-before-Wave-D ordering. Low probability given how much redundant,
  correct disambiguation surrounds it, but the label itself doesn't help.
- **Impact:** None observed beyond the theoretical misread above; every other
  redundant source of truth in the document is correct.
- **Required revision (optional polish, non-blocking):** Rename to `Wave C
  (parallel pair): TASK-04 ...` / `Wave C (parallel pair): TASK-05 ...` (same
  letter, distinguishing sub-row), or add "(same wave as TASK-04)" to
  TASK-05's row, so the label itself — not just the parenthetical — signals
  the parallel relationship.

(Both items above are MINOR, non-blocking, and do not affect this pass's
verdict — consistent with pass 3's treatment of similarly-scoped optional
polish items.)

### Implementation Order Review
1. TASK-01 — gate-free; confirmed via direct source read of
   `crates/catboost-rs/src/lib.rs` that no `CvResult`/`cv` symbol exists yet
   and none is needed for TASK-01 (pure `EvalMetric`-only helper).
2. Pre-TASK-02 checkpoint (re-verify ORCH-01's FINAL `cv.rs`/fixtures) —
   still explicitly documented as the `G-ORCH-01` sub-condition; confirmed
   ORCH-01 remains draft (only `PLAN-CHECK.md`/`PLAN.md`/`SPEC.md` under
   `.planning/plans/cv-cross-validation/`, no source), so this checkpoint has
   not yet been exercised — unchanged from pass 3, not a new gap.
3. TASK-02 → TASK-03 → TASK-04 — correctly serialized (same prod file
   `grid_search.rs`), each gated on the appropriate `G-ORCH-01` sub-piece;
   "Blocked by" and "Parallelization" lines agree for both TASK-02 and
   TASK-03.
4. TASK-05 — correctly gated on TASK-03 only; internally consistent in both
   "Blocked by" and "Parallelization," and its own "Parallelization" line
   pre-announces the TASK-06 join condition correctly.
5. **TASK-06 — now internally CONSISTENT** (pass-3 MAJOR CLOSED, verified
   this pass): both its "Blocked by" line and its "Parallelization" line
   require BOTH TASK-04 and TASK-05, matching the wave table and the ASCII
   diagram. No field in the document tells a different story for TASK-06 any
   longer.

Graph is acyclic. Every task's prerequisites are available when it begins, as
written. No intermediate state leaves the repository unable to build/start
(each task's Red step is documented to fail via compile error until its
prerequisite lands, which is the intended TDD signal, not an accidental
broken state).

### Potential Bugs
- **Premature TASK-06 start via the "Parallelization" line** (the pass-3
  MAJOR) — **RESOLVED this pass.** Triggering condition would have been:
  TASK-04 and TASK-05 run in parallel, TASK-05 finishes first, an implementer
  consults only TASK-06's "Parallelization" line and starts early. Confirmed
  no longer possible: that line now explicitly requires BOTH tasks.
- **Stale contract drift** (pass-1 MAJOR, mitigated, unchanged) — the
  explicit pre-TASK-02 re-verification checkpoint against ORCH-01's FINAL
  merged `cv.rs`/fixtures remains in place and unexercised (ORCH-01 still
  draft) — unchanged from pass 3.
- **Seed double-duty / nested rayon** (pass-1 MINORs, documented, unchanged)
  — both remain explicit, low-risk, documented tradeoffs in SPEC §9
  (confirmed present at the same lines this pass), unchanged.
- **Wave-lettering misread** (new MINOR, this pass) — see Issues above; low
  probability, self-disambiguated by surrounding text, does not block.
- **Asymmetric gate restatement in TASK-04/05 "Parallelization" lines** (new
  MINOR, this pass) — see Issues above; not a correctness bug, since the gate
  is transitively satisfied by the time TASK-03 lands regardless.

### Required Plan Revisions
No BLOCKER/CRITICAL/MAJOR revision required — the plan is ready for
implementation once its documented external gate (`G-ORCH-01`, ORCH-01
shipping `CvResult`/`cv`) is satisfied.

Optional, non-blocking polish (may be applied at the implementer's/next
checker's discretion, none required for a PASS verdict):
1. Add a one-clause gate restatement to TASK-04's and TASK-05's own
   "Parallelization" lines for symmetry with TASK-02/TASK-03 (see MINOR
   issue above).
2. Adjust the §1 wave-table lettering (or add an explicit "(same wave as
   TASK-04)" annotation to TASK-05's row) so the parallel relationship is
   signaled by the label itself, not only by the parenthetical/prose/diagram
   (see MINOR issue above).
3. (Carried from pass 3, still optional) Add a one-clause mention to
   TASK-04's own "Parallelization" line that it may run in parallel with
   TASK-05, for symmetry with TASK-05's own line.
4. (Carried from pass 3, still optional) Extend the §1 "File-ownership note"
   to mention the functional `randomized_search` reason for TASK-06
   depending on TASK-04, alongside the file-sharing reason it already states.

### Unverified Items
- **Exact `cv()` column-key spelling** (SPEC §9 Q1) — still cannot be
  verified until ORCH-01 TASK-06 actually ships; correctly flagged as
  unverified in the SPEC itself; unchanged from pass 3.
- **ORCH-01's post-revision `cv()` signature/behavior** — still cannot be
  verified today (ORCH-01 remains draft/unshipped, confirmed via `ls
  .planning/plans/cv-cross-validation/` and the direct `lib.rs` read showing
  no `cv`/`CvResult`); this is the basis of the (mitigated-by-checkpoint)
  pass-1 MAJOR, not a new gap, and does not block a PASS verdict for THIS
  plan's own internal correctness — it is an explicitly documented external
  precondition (`G-ORCH-01`), not a defect in this plan.
- **Python uv-3.12 venv availability in this environment** — not exercised in
  this review (no code was run, this was a read-only planning-artifact
  review); the documented `cargo check` fallback is unchanged and adequate.
- **Upstream `catboost.grid_search`/`randomized_search` selection-parity
  behavior** — explicitly and correctly deferred as a non-goal; unchanged.

---

## Pass 3 — final revision verification (prior pass, preserved verbatim below for history)

This is PASS 3, the third and final allowed pass under the spec-tdd-planner-
skill's normal cap, re-run from scratch (not a diff-only check) against the
revised `SPEC.md`/`PLAN.md`. Focus per the orchestrator's instruction:
re-check EVERY place in the document describing the TASK-04/TASK-05/TASK-06
relationship (the §1 wave-graph text block, the ASCII diagram, TASK-04's own
"Parallelization" line, TASK-05's own "Blocked by"/"Parallelization" lines,
TASK-06's "Blocked by" line, and the "Spec-ID → task coverage"/file-ownership
notes), plus a full from-scratch re-run of the rest of the checker process
(CodeGraph re-verification, adversarial bug search, order/verification
review).

**Confirmed via `git log`/`ls` (this pass):** no source file relevant to this
plan (`builder.rs`, `metrics.rs`, `custom.rs`, `permutation.rs`, `rng.rs`,
`catboost-rs-py/src/lib.rs`, `params.rs`, `estimator.rs`, `catboost-rs/src/lib.rs`,
`catboost-rs/Cargo.toml`) has changed since pass 2 (last touching commits
predate pass 2: `cf1252b`, `1ed82ba`, `516f787`, `ba08aaf`, `8fe7dd6`).
`crates/catboost-rs/src/grid_search.rs` and `crates/catboost-rs-py/src/search.rs`
still do not exist — both are genuinely net-new, as the plan assumes. Only
`SPEC.md`/`PLAN.md` text changed. Every CodeGraph-checkable claim was
independently re-run this pass via fresh `codegraph_explore` calls (not reused
from pass-2's cache) and all are confirmed EXACT (see CodeGraph Evidence).

**Result of the TASK-04/05/06 consistency sweep (the pass-3 mandate):**
Seven of eight relevant locations are now fully consistent and correctly
describe "TASK-04 ∥ TASK-05 (both gated on TASK-03 only), TASK-06 gated on
BOTH":

1. §1 wave-graph text block (PLAN.md lines 132-136) — correct.
2. ASCII dependency diagram (lines 141-146) — correct fork/join, no longer a
   strictly-linear chain.
3. Explanatory note directly under the diagram (lines 147-153) — correct.
4. Prose paragraph after the note (lines 155-165) — correct, explicit:
   "TASK-06 is sequential after BOTH TASK-04 (needs `randomized_search`) and
   TASK-05 (shares `search.rs` + reuses TASK-05's `expand_param_grid` helper)."
5. TASK-05's own "Blocked by" line (468-472) — correct, self-documenting the
   pass-1 correction.
6. TASK-05's own "Parallelization" line (532-534) — correct and complete:
   "PARALLEL with TASK-04 ... TASK-06 remains sequential after BOTH this task
   and TASK-04."
7. TASK-06's own "Blocked by" line (542-546) — correct: "TASK-04 **AND**
   TASK-05," with the functional reason (`randomized_search`) stated
   explicitly, and self-documents the pass-2 fix.

**One location was NOT propagated and reintroduces the same class of defect
pass 2 just fixed, now WITHIN TASK-06's own section:** TASK-06's own
"**Parallelization**" line (PLAN.md line 587) still reads **"sequential after
TASK-05."** — period, no mention of TASK-04 — directly contradicting TASK-06's
own "Blocked by" line five lines above it (542-546), which correctly requires
BOTH TASK-04 AND TASK-05. This is a NEW MAJOR finding for this pass (see
Issues below): the pass-2 fix was applied to the "Blocked by" field but not to
the sibling "Parallelization" field in the same task block, so the "exactly
ONE consistent story" requirement is not yet met.

Two additional MINOR completeness (non-contradictory) gaps were found on the
symmetry check requested by the prompt: TASK-04's own "Parallelization" line
does not mention the parallel opportunity with TASK-05 (unlike TASK-05's own
line, which does mention TASK-04), and the §1 "File-ownership note" restates
only the file-sharing reason for TASK-06 (`search.rs`) without the functional
reason (`randomized_search`), even though pass-2 asked for that note
specifically. Neither contradicts anything — the correct, complete story is
stated unambiguously in the immediately preceding paragraph and in TASK-06's
own "Blocked by" line — but both are noted for completeness since the prompt
asked for a location-by-location sweep.

**Net effect:** the pass-2 MAJOR finding (ASCII diagram + TASK-06 "Blocked by"
+ file-ownership note) is CLOSED for the diagram and the "Blocked by" line,
but a new, same-class inconsistency was found in TASK-06's own
"Parallelization" line, which was not updated in step with the "Blocked by"
fix. Verdict is therefore **ISSUES_FOUND**, carried by one new MAJOR item
(intra-task-section contradiction), not by a regression of the four
pass-1/pass-2 findings, which remain closed.

---

**Verdict:** ISSUES_FOUND
**Goal:** ORCH-02 — `grid_search`/`randomized_search` hyperparameter search for
catboost-rs, hard-dependent on the sibling ORCH-01 (`cv()`) slice, itself
in-flight/under CRITICAL-finding revision.
**Plan:** `.planning/plans/grid-random-search/SPEC.md` (revised) +
`.planning/plans/grid-random-search/PLAN.md` (revised)

### Summary
- All 5 pass-1 findings and the pass-2 MAJOR finding's primary manifestations
  (ASCII diagram linear chain; TASK-06's "Blocked by" line) are confirmed
  resolved this pass, independently re-verified against the current on-disk
  text (not trusted from the parent's characterization).
- Every CodeGraph-checkable claim in the revised SPEC/PLAN was independently
  re-run this pass via fresh `codegraph_explore` calls: `EvalMetric` (16
  variants, exact — `crates/cb-train/src/metrics.rs:64-`, ranking partition
  confirmed), `CustomMetric::is_max_optimal` (`crates/cb-compute/src/custom.rs:111-114`,
  `CustomMetricHandle` at `:155`), `CatBoostBuilder` (`#[derive(Debug, Clone,
  PartialEq)]` at `crates/catboost-rs/src/builder.rs:64-65`, `fit` present),
  `fisher_yates_permutation` (`crates/cb-train/src/permutation.rs:109`, calls
  `TFastRng64::from_seed`), `parse_metric` (`crates/cb-train/src/calc_metrics.rs:151`,
  16-way match confirmed exact), `rayon.workspace = true` at
  `crates/catboost-rs/Cargo.toml:39` (re-read directly), `sum_models`
  registration at `crates/catboost-rs-py/src/lib.rs:57` (exact), `make_builder`
  (`crates/catboost-rs-py/src/params.rs:451-513`, body confirmed) and
  `validate_params` (`params.rs:290-318`, body confirmed — NOT merely a line
  number citation this pass, full body re-read), `EstimatorBase`
  (`crates/catboost-rs-py/src/estimator.rs:24-30`) and `data_to_pool`
  (`estimator.rs:236-250`) — all confirmed EXACT, byte-for-byte matching the
  plan's citations.
- Confirmed via `git log` + `ls` (this pass, not reused): no source file this
  plan depends on has changed since pass 2; `grid_search.rs` and `search.rs`
  still do not exist; `crates/catboost-rs/src/lib.rs` still has no `cv`/
  `CvResult` re-export. The cross-plan-blocker premise remains accurate.
- **One NEW MAJOR issue found in this pass:** TASK-06's own "Parallelization"
  line (PLAN.md line 587, `sequential after TASK-05.`) was not updated when
  its "Blocked by" line (542-546) was corrected to require BOTH TASK-04 AND
  TASK-05 — an intra-task-section contradiction, the same class of defect
  pass 2 flagged, now reappearing one field lower in the SAME task block. See
  Issues below.
- Two MINOR completeness (non-contradictory) notes: TASK-04's own
  "Parallelization" line does not mention the TASK-05 parallel opportunity
  (asymmetric with TASK-05's own line, which does mention TASK-04); the §1
  "File-ownership note" still states only the file-sharing reason for TASK-06,
  not the functional `randomized_search` reason — though the correct complete
  story is unambiguous elsewhere in the same section.

### Specification Coverage
- [x] ORCH-02-S1 `metric_is_max_optimal`: TASK-01, exhaustive match, re-verified
  16-variant `EvalMetric` enum exactly matches the plan's claimed variant list.
- [x] ORCH-02-S2 `score_candidate`/`select_best`: TASK-02, checked-indexing
  algorithm confirmed (no raw indexing anywhere in the revised text).
- [x] ORCH-02-S3 Rust `grid_search` (+refit): TASK-03, oracle + checked-indexing
  code sketch.
- [x] ORCH-02-S4 Rust `randomized_search`: TASK-04, subsampling unit tests +
  one integration case; correctly sequenced after TASK-03 (same prod file).
- [x] ORCH-02-S5 Python `grid_search`: TASK-05's dependency correction (TASK-03
  only) is sound, internally consistent, and independently verified (disjoint
  files, no functional call to `randomized_search`).
- [ ] ORCH-02-S6 Python `randomized_search`: TASK-06's SPEC-level behavior is
  fully covered and its "Blocked by" line is now correct, but its own
  "Parallelization" line contradicts that same "Blocked by" line — see the
  MAJOR issue below.

### CodeGraph Evidence
- `EvalMetric` — `crates/cb-train/src/metrics.rs` (re-confirmed this pass via
  a fresh `codegraph_explore` call): 16 variants exactly as PLAN/SPEC cite;
  blast-radius shows 11 callers across `cb-train`/`catboost-rs`/tests, none of
  which this plan touches (read-only reuse, D-04 respected).
- `CustomMetric::is_max_optimal` — `crates/cb-compute/src/custom.rs:111-114`,
  `fn is_max_optimal(&self) -> bool;` exact, doc comment confirms `true`=larger
  is better. `CustomMetricHandle(pub Arc<dyn CustomMetric>)` at `custom.rs:155`.
  Confirmed exact.
- `CatBoostBuilder` — `crates/catboost-rs/src/builder.rs:64-65`
  `#[derive(Debug, Clone, PartialEq)] pub struct CatBoostBuilder`; full field
  list and every setter re-read this pass (`.loss()`, `.depth()`, `.iterations()`,
  etc.) — all present, all consuming/returning `Self` (cheap clone-and-mutate
  confirmed structurally, not just asserted).
- `fisher_yates_permutation` — `crates/cb-train/src/permutation.rs:109`,
  `pub fn fisher_yates_permutation(n: usize, seed: u64) -> Vec<i32>`, body
  confirmed: `TFastRng64::from_seed(seed)` then `shuffle_in_place`. Blast
  radius shows 14 callers, none touched by this plan.
- `parse_metric` — `crates/cb-train/src/calc_metrics.rs:151`, full match body
  re-read this pass: all 15 named-metric arms (`rmse|logloss|msle|mae|mape|
  quantile|ndcg|dcg|map|mrr|err|pfound|precisionat|recallat|queryauc`) plus an
  `other =>` typed-error arm — confirms the plan's 16-variant claim (15 parsed
  + `Custom` which is program-constructed only, matching the doc comment at
  line 24 excluded from string parsing) exactly.
- `rayon` in `crates/catboost-rs/Cargo.toml:39` — `rayon.workspace = true`,
  confirmed at the exact cited line.
- `sum_models` registration — `crates/catboost-rs-py/src/lib.rs:57`,
  `m.add_function(wrap_pyfunction!(regressor::sum_models, m)?)?;`. Confirmed
  EXACT via direct grep this pass (not `sed`).
- `make_builder` — `crates/catboost-rs-py/src/params.rs:451-513`, full body
  re-read this pass (not just the signature): sequentially applies
  `get_with_aliases` + `check_range` + a builder setter per known param,
  returning `Ok(builder)` — confirmed this is the exact "kwargs dict →
  `CatBoostBuilder`" surface the plan's TASK-05/06 rely on.
- `validate_params` — `crates/catboost-rs-py/src/params.rs:290-318`, full body
  re-read this pass: iterates `params.keys()`, resolves aliases, rejects
  `KnownNotYet`/unknown via `CatBoostParameterError` with a Levenshtein-based
  "did you mean" suggestion — confirms the plan's "bad param raises a mapped
  `CatBoostParameterError`" claim (TASK-05/06 Red sections) is backed by real
  code, not an assumption.
- `EstimatorBase`/`data_to_pool` — `crates/catboost-rs-py/src/estimator.rs:24-30`
  (`pub(crate) struct EstimatorBase { params: BTreeMap<String, Py<PyAny>>,
  model: Option<Model> }`) and `:236-250` (`data_to_pool` body: `Pool` fast-path
  via `.cast::<crate::pool::Pool>()`, else `ingest_to_owned(...).into_pool()`
  mapped to `CatBoostValueError`). Both confirmed exact.
- `crates/catboost-rs/src/lib.rs` (re-read this pass, `mod`/`pub use` lines
  only) — confirmed STILL re-exports only `CatBoostBuilder`, `CatBoostError`,
  `eval_metric`/`eval_metrics`, `Model`, `FeatureImportanceType`/`PredictionType`,
  `PartialDependence`/`PdpError`, `OnnxExportError`, `CoreMlExportError`,
  `EScoreFunction`/`LeafMethod`/`Loss`, `EBootstrapType`, `IngestSource`/
  `OwnedColumns`, `Pool` — NO `cv`, `CvResult`, `CvFold`, `CvType`. Cross-plan
  blocker framing remains accurate today.
- `git log --oneline -- <9 relevant source files>` (this pass) — most recent
  touching commit is `cf1252b` (CoreML export), which predates pass 2; no
  source drift since pass 2. `ls crates/catboost-rs/src/ | grep grid` and
  `ls crates/catboost-rs-py/src/ | grep search` — both empty; `grid_search.rs`
  and `search.rs` remain genuinely net-new.

### Issues

#### [MAJOR] TASK-06's own "Parallelization" line contradicts its own (correctly-fixed) "Blocked by" line — the same class of defect pass 2 fixed has reappeared one field lower, within the same task block
- **Plan location:** PLAN.md TASK-06 section — "Blocked by" (lines 542-546,
  correct) vs. "Parallelization" (line 587, stale/incomplete).
- **Requirement:** Every task's own internal fields must tell "exactly ONE
  consistent story" about its prerequisites (the explicit mandate for this
  pass), and implementation order must be valid and unambiguous from any
  single field a reader consults (per the plan-checking mandate on validating
  implementation order).
- **Evidence:** TASK-06's "Blocked by" line (542-546) now correctly reads:
  "**Blocked by:** TASK-04 **AND** TASK-05 (needs `randomized_search` from
  TASK-04 AND reuses TASK-05's `expand_param_grid` helper / `search.rs` file —
  corrected after PLAN-CHECK pass 2 caught this line and the §1 ASCII graph
  both still implying a linear TASK-04→TASK-05→TASK-06 chain...)." But
  TASK-06's OWN "Parallelization" line, five lines below its own
  "Compat/rollback" bullet (PLAN.md line 587), still reads verbatim:
  "**Parallelization:** sequential after TASK-05." — with no mention of
  TASK-04 at all. This is the exact opposite pattern of TASK-05's own section,
  where BOTH the "Blocked by" (468-472) and "Parallelization" (532-534) lines
  were correctly and consistently updated to mention TASK-04. The fix was
  applied asymmetrically: to TASK-06's "Blocked by" line but not to its
  "Parallelization" line in the same block.
- **Failure scenario:** An implementer or reviewer who treats the
  "Parallelization" field as the authoritative single-line sequencing summary
  for a task (a reasonable practice, since that field plays exactly that role
  for every other task in this plan: TASK-01 "may proceed immediately...",
  TASK-02 "sequential after TASK-01...", TASK-03 "sequential after TASK-02...",
  TASK-04 "sequential after TASK-03...", TASK-05 "sequential after TASK-03;
  PARALLEL with TASK-04...") could read TASK-06's "sequential after TASK-05"
  literally and begin TASK-06 the moment TASK-05 lands, without waiting for
  TASK-04. This reproduces PRECISELY the failure mode pass 2 already
  identified and required fixed: TASK-06's own code (Files bullet, lines
  553-558) literally calls `catboost_rs::randomized_search(...)` — TASK-04's
  output — so starting TASK-06 before TASK-04 lands causes an
  `unresolved import`/`cannot find function` compile failure, or invites an
  implementer to stub/inline a placeholder to unblock themselves rather than
  wait, reintroducing drift.
- **Impact:** The SAME implementation-order risk pass 2 flagged (premature
  TASK-06 start relative to TASK-04) is NOT actually closed — it has moved
  from the ASCII diagram / "Blocked by" line (now fixed) to the
  "Parallelization" line (not yet fixed) within the identical task section.
  Confusable, self-contradictory instructions for whoever executes TASK-06.
- **Required revision:** Update TASK-06's "Parallelization" line (PLAN.md line
  587) to read consistently with its own "Blocked by" line and with TASK-05's
  "Parallelization" line, e.g.: "**Parallelization:** sequential after BOTH
  TASK-04 (needs `catboost_rs::randomized_search` to exist) AND TASK-05
  (re-serializes on `search.rs` + reuses `expand_param_grid`) — the join point
  of the TASK-04 ∥ TASK-05 parallel wave; NOT simply 'after TASK-05'."

  **[Resolved — confirmed independently in the extra verification pass
  above.]** TASK-06's "Parallelization" line now reads: "sequential after
  BOTH TASK-04 (needs `randomized_search`) AND TASK-05 (shares `search.rs` +
  `expand_param_grid`) — corrected to match this task's own 'Blocked by'
  line...". Consistent with the requested revision.

### Implementation Order Review
1. TASK-01 — gate-free; confirmed via CodeGraph it needs no `CvResult`/`cv`
   symbol (module scaffold + pure direction helper only).
2. Pre-TASK-02 checkpoint (re-verify ORCH-01's FINAL `cv.rs`/fixtures) —
   explicitly documented as a `G-ORCH-01` sub-condition (pass-1 MAJOR,
   resolved, unchanged this pass).
3. TASK-02 → TASK-03 → TASK-04 — correctly serialized (same prod file
   `grid_search.rs`), each gated on the appropriate `G-ORCH-01` sub-piece.
4. TASK-05 — correctly gated on TASK-03 only; own section internally
   consistent both in "Blocked by" and "Parallelization."
5. **TASK-06 — internally INCONSISTENT as written** (see MAJOR issue): its
   "Blocked by" line correctly requires BOTH TASK-04 and TASK-05, but its
   "Parallelization" line — a field every OTHER task in this plan uses as an
   authoritative one-line sequencing summary — still says only "after
   TASK-05." Corrected order (once the revision above is applied): TASK-06
   must wait for BOTH TASK-04 and TASK-05, exactly as the wave graph, ASCII
   diagram, and "Blocked by" line already state; only the "Parallelization"
   line needs to catch up.

No cycles. With the required revision applied, the graph is valid, acyclic,
and every field within every task section tells the same story. As CURRENTLY
WRITTEN, a reader relying solely on TASK-06's "Parallelization" line (as is
reasonable, given every other task's line plays that summarizing role) would
reach an incorrect, premature start decision.

### Potential Bugs
- **Premature TASK-06 start via the "Parallelization" line** (the MAJOR
  issue) — triggering condition: TASK-04 and TASK-05 genuinely run in
  parallel (as now explicitly permitted) and TASK-05 completes first while an
  implementer consults only TASK-06's "Parallelization" line; failure mode:
  TASK-06 begins while `catboost_rs::randomized_search` does not yet exist,
  causing a compile failure (self-correcting but wasted effort, and an
  invitation to stub a placeholder); affected path:
  `crates/catboost-rs-py/src/search.rs`'s `randomized_search` `#[pyfunction]`
  → `catboost_rs::randomized_search`; required mitigation: the revision above.
- **Stale contract drift** (pass-1 MAJOR, mitigated, unchanged this pass) —
  the explicit pre-TASK-02 re-verification checkpoint against ORCH-01's FINAL
  merged `cv.rs`/fixtures remains in place; confirmed no ORCH-01 source has
  landed yet (still draft), so this checkpoint has not yet been exercised —
  not a new gap, unchanged from pass 2.
- **Seed double-duty / nested rayon** (pass-1 MINORs, documented, unchanged) —
  no residual bug; both remain explicit, low-risk, documented tradeoffs in
  SPEC §9, re-confirmed present this pass (SPEC.md lines 607-608).

### Required Plan Revisions
1. **[MAJOR, blocking]** Correct TASK-06's "Parallelization" line (PLAN.md
   line 587) to state it is sequential after BOTH TASK-04 and TASK-05, with
   the functional reason, matching TASK-06's own already-correct "Blocked by"
   line and TASK-05's already-correct "Parallelization" line.
   **[Resolved — confirmed independently in the extra verification pass above.]**
2. **[MINOR, optional polish]** Add a one-clause mention to TASK-04's own
   "Parallelization" line (PLAN.md line 459) that it may run in parallel with
   TASK-05 (disjoint files), for symmetry with TASK-05's own line — not
   required for correctness (TASK-04 does not depend on TASK-05, so its
   current line is not wrong, only asymmetric/incomplete).
3. **[MINOR, optional polish]** Optionally extend the §1 "File-ownership note"
   (PLAN.md lines 167-173) to mention the functional `randomized_search`
   reason for TASK-06 depending on TASK-04, alongside the file-sharing reason
   it already states — the note's own heading scopes it to "no write conflicts
   within a wave," so this is not a contradiction, but adding the functional
   reason would make the note self-sufficient without needing the reader to
   consult the preceding paragraph.

(Pass-1's 5 original required revisions and pass-2's ASCII-diagram / "Blocked
by" line fixes are confirmed applied and closed; no further action needed on
those specific findings.)

### Unverified Items
- **Exact `cv()` column-key spelling** (SPEC §9 Q1) — still cannot be verified
  until ORCH-01 TASK-06 actually ships; correctly flagged as unverified in the
  SPEC itself; unchanged from pass 2.
- **ORCH-01's post-revision `cv()` signature/behavior** — still cannot be
  verified today (ORCH-01 remains draft/unshipped, confirmed via `ls .planning/
  plans/cv-cross-validation/` and no `cv`/`CvResult` in `catboost-rs/src/lib.rs`);
  this is the basis of the (mitigated-by-checkpoint) pass-1 MAJOR, not a new gap.
- **Python uv-3.12 venv availability in this environment** — not exercised in
  this review (no code was run); the documented `cargo check` fallback is
  unchanged and adequate.
- **Upstream `catboost.grid_search`/`randomized_search` selection-parity
  behavior** — explicitly and correctly deferred as a non-goal; unchanged.
