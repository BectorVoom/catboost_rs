# Plan Check Result

**VERDICT: ISSUES_FOUND**

**Goal:** Wire categorical-feature (`cat_features`/CTR) routing through the `CatBoostBuilder`
facade (train + predict side) and the Python estimators, at ≤1e-5 upstream parity.
**Plan:** `.planning/plans/catboost-builder-cat-features-routing/PLAN.md` (18 tasks, T01–T18)
**Spec:** `.planning/plans/catboost-builder-cat-features-routing/SPEC.md` (SPEC-CATF-01..17)

---

## Summary

The plan is unusually well-researched — most of its `[VERIFIED]` anchors check out, its
open-question resolutions (OQ-1/2/3) are correct and evidence-backed, its dependency graph is
acyclic, and its serialization claims on `builder.rs` / `model.rs` / `params.rs` are right.

It nevertheless **must not be executed as written**. Four defects are load-bearing:

1. **T16's fixture-generation command regenerates the ENTIRE frozen oracle corpus.**
   `gen_fixtures.py` has no positional-scenario dispatch — an unrecognised argv falls through
   to `main()`. This directly destroys the invariant SPEC-CATF-17 exists to protect.
2. **Three of the five promoted Python CTR kwargs are inert in the engine.**
   `params.simple_ctr`, `params.simple_ctr_priors`, `params.combinations_ctr` and
   `params.counter_calc_method` are **never read** by `train_inner`; the CTR type is
   hard-coded to `Borders` and all priors come from `combinations_ctr_priors`. Promoting
   them to `IMPLEMENTED` manufactures exactly the silently-ignored-kwarg failure the
   registry's honesty policy — quoted by the SPEC itself to justify SPEC-CATF-12 — forbids.
3. **T09's derived categorical width is a data-dependent lower bound enforced as strict
   equality.** It will reject legitimate `fit(pool) → predict(same pool)` round-trips whenever
   a trailing cat column is one-hot-routed, constant, or simply unused by any chosen split.
4. **One-hot-routed categorical columns are silently dropped from training.** `train_inner`
   never calls `grow_one_hot_tree`; `cb_train::ObliviousTree` has no one-hot split field. With
   the builder's default `one_hot_max_size = 2`, every binary categorical column silently
   contributes nothing — while this plan advertises `one_hot_max_size` as a first-class setter
   and Python kwarg.

Plus a build-breaking omission (four un-listed `data_to_pool` call sites), an uncovered
behaviour change in `cv`/`grid_search`, and three invalid validation commands.

One finding cuts the *other* way: the plan's self-declared "single largest schedule risk"
(A1 / F5 / R9, `permutation_count=4`) is **already closed by an existing passing oracle** the
plan did not cite.

---

## Specification Coverage (independently re-derived, not taken from the plan's table)

| Spec | Task | Verdict |
|---|---|---|
| CATF-01 `one_hot_max_size` setter | T01 | Mechanically covered; **semantically hollow** — see [CRITICAL-4] |
| CATF-02 `max_ctr_complexity` setter | T02 | Covered (`params.max_ctr_complexity` IS consumed, `boosting.rs:2711`) |
| CATF-03 `simple_ctr` + priors | T03 | Field reaches `BoostParams`; **engine ignores both** — see [BLOCKER-2] |
| CATF-04 `combinations_ctr` + priors | T04 | Type ignored; only `combinations_ctr_priors` is live — [BLOCKER-2] |
| CATF-05 `counter_calc_method` | T05 | Field reaches `BoostParams`; **never read** — [BLOCKER-2] |
| CATF-06 default-equivalence | T06 | Covered; mutation check is weak but adequate — [MINOR-9] |
| CATF-07 float-only no-regression | T18 | Covered for the float path; **does not cover the cv/grid_search categorical path** — [MAJOR-6] |
| CATF-08 categorical `fit()` → `train_cat` | T08 | Covered, with a genuine anti-false-pass guard |
| CATF-09 nameable types | T07 | Covered; `ECtrType::as_i8` / `CounterCalcMethod::default()` verified to exist |
| CATF-10 cat width validation | T09 | **Not implementable as specified** — [CRITICAL-3] |
| CATF-11 CTR-aware `predict_with` | T10 | Covered; the `predict_raw == predict_raw_cat(.., &[])` identity is verified |
| CATF-12 never silently mis-scored | T11 | Covered for `predict`/`predict_with`/`predict_proba` **only** — [MAJOR-7] |
| CATF-13 five CTR kwargs from Python | T12,T13 | Accept/reject plumbing covered; **semantics unsound** — [BLOCKER-2] |
| CATF-14 `cat_features` fit-kwarg | T14 | Covered, but **misses 4 call sites** — [MAJOR-5]; `int|str` narrowed to `int` — [MINOR-10] |
| CATF-15 registry truthful | T15 | Covered, and the SKIP-repair is a real improvement; verified the fallback will pass (see evidence) |
| CATF-16 Python-API oracle ≤1e-5 | T17 | Covered; at risk from [CRITICAL-3] with no ladder step for it |
| CATF-17 fixture frozen | T16 | **Command as written unfreezes everything** — [BLOCKER-1] |

Acceptance scenarios: A1, A2, A6, A7, A9, A10 covered. **A3 and A5 are at risk** from
CRITICAL-3. **A4 is covered but narrower than its own title** (MAJOR-7). A8 covered but
build-broken as specified (MAJOR-5).

---

## CodeGraph Evidence (load-bearing verifications)

- `CatBoostBuilder::fit` — `crates/catboost-rs/src/builder.rs:321+`
  - Callers (blast radius): `cv.rs:407` (`builder.fit(&train_pool)`), `grid_search.rs:405`
    (`best_builder.fit(pool)`), `grid_search.rs` `run_over` path, `catboost-rs-py`
    `estimator.rs:204 fit_pool(builder, pool) → builder.fit(pool)`.
    `CatBoostBuilder` itself: 16 callers across `cv.rs`, `lib.rs`, `params.rs`, `estimator.rs`.
    Covering tests: `builder_oracle_test.rs`, `cv_oracle_test.rs`, `grid_search_oracle_test.rs`.
  - Impact: all inherit categorical routing with no signature change — confirmed. But see
    MAJOR-6: `cv.rs:407` is immediately followed by `model.staged_predict(...)` at `:408-409`.
- `Model::predict_with` — `crates/catboost-rs/src/model.rs:120-128`
  - Callers: `model.rs:135` (`predict`), `model.rs:280` (`predict_proba`),
    `catboost-rs-py/src/classifier.rs:102, 131, 282`; tests `builder_oracle_test.rs:128,139,153`.
  - Callees: `feature_columns` (`model.rs:94-107`, float width only), `predict_raw`
    (`cb-model/src/apply.rs:370`).
  - Impact: the plan's 3-entry-point guard (T11) covers `predict`, `predict_with`,
    `predict_proba` and therefore transitively every Python `predict`/`predict_proba` adapter.
    Verified complete for `predict_with`. See MAJOR-7 for the paths that do **not** go
    through `predict_with`.
- `cb_model::predict_raw` — `crates/cb-model/src/apply.rs:370-372` is literally
  `predict_raw_cat(model, feature_values, &[])`. Inside `predict_raw_cat`
  (`apply.rs:386-416`), line **411** is `col.get(obj).cloned().unwrap_or_default()` — the
  empty-string fallback. 45 callers of `predict_raw` repo-wide. **The plan's byte-identity
  argument for the float path is correct.**
- `cb_train::train_cat` — `crates/cb-train/src/boosting.rs:2149-2172`. Signature matches the
  plan verbatim. Delegates to `train_inner` (`:2263`). Existing test callers:
  `tensor_ctr_e2e_oracle_test.rs`, `multi_permutation_e2e_oracle_test.rs`.
- `cb_model::Model::from_trained` — `crates/cb-model/src/model.rs:326-426`. Lifts
  `t.splits → ModelSplit::Float` and `t.ctr_splits → ModelSplit::Ctr`; sets `ctr_data: None`.
  `with_ctr_data` (`:431-434`) sets `Some(..)` unconditionally.
- `ModelSplit` (`cb-model/src/model.rs:71`) has exactly **two** variants (`Float`, `Ctr`) —
  confirmed by the exhaustive 2-arm match in `passes_split` (`apply.rs:196-201`).
  **There is no one-hot ModelSplit.**
- `cb_train::ObliviousTree` (`boosting.rs:775-793`) has `splits: Vec<Split>` (float) +
  `ctr_splits: Vec<CtrSplitSpec>`. **No one-hot field.** Its own doc says "the numeric /
  one-hot / ordered boosting paths produce ONLY float splits here".
- `grow_one_hot_tree` (`cb-train/src/tree.rs:3141`) — callers: 3 in `cb-train/src/lib.rs`
  (re-exports only) + tests `one_hot_oracle_test.rs`, `tree_test.rs`. `one_hot_oracle_test.rs:90`
  builds "a minimal one-hot-only plain-boosting driver" **in the test**. **No production caller
  in `boosting.rs`.**
- `params.one_hot_max_size` in `train_inner` — read only at `boosting.rs:2711` and `:2725`,
  both purely for CTR *eligibility* (`tensor_ctr_candidates` / `route_categorical` →
  `eligible_absolute`).
- CTR type / priors in `train_inner`:
  - `boosting.rs:1853` — `ctr_type: crate::ctr::ECtrType::Borders.as_i8()` **hard-coded**.
  - `boosting.rs:2821` — `let ctr_prior_num = params.combinations_ctr_priors.first()...`
  - `boosting.rs:4789` — `ctr_splits_for_tree(&ctr_candidates, &params.combinations_ctr_priors)`
  - `grep` of `boosting.rs` finds **zero** reads of `params.simple_ctr`,
    `params.simple_ctr_priors`, `params.combinations_ctr`, `params.counter_calc_method`.
- Projection index space: `boosting.rs:2715-2728` builds `eligible_absolute` (CTR-eligible cat
  indices only); `:2829-2840` re-indexes candidate projections back to **absolute** cat indices.
  `CtrSplitSpec.projection` therefore holds absolute indices — the plan's F4 is **correct**, and
  `passes_ctr_split` (`apply.rs:157-166`) gathers by those absolute indices. This validates the
  *indexing* half of T09 and invalidates the *completeness* half (CRITICAL-3).
- `Pool::select_rows` — `crates/cb-data/src/pool.rs:219-259` gathers `cat_features` (`:232-236`).
  cv folds therefore carry categorical columns.
- `ensure_scalar_oblivious` — `crates/catboost-rs/src/model.rs:144-171`; `:164-169` rejects
  `ctr_data.is_some()` with `UnsupportedModel`. Consumed by `staged_predict`.
- `Model::predict_raw_on_device` — `model.rs:249-272` calls `cb_model::flatten_oblivious_f64`,
  which its doc states rejects categorical/CTR models. **This entry point is safe** (contrary to
  a plausible worry).
- `Model::shap_values` (`model.rs:290-293`) → `cb_model::shap_values`, which projects splits via
  `float_splits_of` (`cb-model/src/shap.rs:550-552`) — CTR splits are silently **dropped**.
  `Model::partial_dependence` (`model.rs:424-431`) and `feature_importance_with_data`
  (`model.rs:369-378`, passes `pool.cat_features()` with **no width check**) are in the same class.
- `data_to_pool` — `crates/catboost-rs-py/src/estimator.rs:236-250`. Call sites (grep, full):
  `classifier.rs:68,100,126,280`; `regressor.rs:68,149,254`; `ranker.rs:65,100`;
  `estimator.rs:270`; **`search.rs:304,389`**; **`cv.rs:61,64`** — 14 total.
- `ingest_to_owned` — `ingest_py.rs:52-67`; branch 3 is `numpy_to_owned(x, y)`
  (`:99-107`, no cat parameter). F2 confirmed. `pandas_to_owned` (`:210-305`) handles a
  zero-numeric-column frame correctly (`:272-274` → `Vec::new()`), so T16/T17's
  categorical-only DataFrame is ingestible.
- `OwnedColumns::reference_n_rows` (`cb-data/src/ingest/owned.rs:133-137`) derives `n_rows` from
  float column 0, else the **label**. T08/T11's zero-float pools are therefore valid *because*
  they carry `y`. Verified — the plan's constructions work.
- `test_params.py:20-29` — `_CORE_PY` points at `catboost-master/catboost/python-package/
  catboost/core.py`, which **does not exist** (verified by `ls`). The plan's T15 finding is
  correct. I additionally verified the fix will work: extracting the 119
  `CatBoostClassifier.__init__` kwargs from the installed `catboost==1.2.10` `core.py` and
  diffing against `params.rs` yields **zero** missing names, so un-skipping the coverage test
  will not cascade into out-of-scope VOCABULARY work.
- Environment: `.venv/bin/python` → `catboost 1.2.10`; `.venv/bin/maturin` present.
  Workspace lints (`Cargo.toml:10-14`): `unwrap_used`/`expect_used`/`panic`/`indexing_slicing`
  = `deny`. `crates/catboost-rs/src/lib.rs:1` has the `cfg_attr(test, allow(...))` exemption.
  `cv.rs:552-553` is the `#[path]` child-mount precedent. All plan claims confirmed.

---

## Issues

### [BLOCKER-1] T16's fixture command regenerates the entire frozen oracle corpus

- **Plan location:** T16 "Red"/"Validation"; §3 "Repository-verified commands"
  (`gen_fixtures.py <scenario>`).
- **Requirement:** SPEC-CATF-17 (fixture frozen, not regenerated); memory `ctr-model-loading`
  (upstream quantization is run-to-run nondeterministic).
- **Evidence:** `crates/cb-oracle/generator/gen_fixtures.py` `__main__` block accepts **only**
  the flags `--wave1-only`, `--wave2-only`, `--wave3-only`, `--multiclass-only`,
  `--multilabel-only`, `--mvs-seeds-only`, `--bootstrap-dev-only`, `--multiquantile-only`;
  **every other argv falls through to `else: main()`**, and `main()` regenerates every
  committed fixture family. There is no positional-scenario dispatch. The plan's claim that
  `gen_fixtures.py <scenario>` is repository-verified is false.
- **Failure scenario:** the implementer runs
  `.venv/bin/python crates/cb-oracle/generator/gen_fixtures.py catf_python_ctr` (twice, per the
  double-generation freeze check). Every fixture under `crates/cb-oracle/fixtures/` is rewritten
  with a fresh, nondeterministically-quantized upstream run.
- **Impact:** mass spurious oracle failures across `cb-train`, `cb-model`, `cb-data`,
  `catboost-rs` and the Python suite; the frozen CTR corpus (`plain_ctr`, `tensor_ctr_e2e`,
  `ordered_ctr`, `ctr_load`, `fstr_ctr`, `one_hot_cat`, …) is destroyed. Recoverable only by
  `git checkout`, and only if the implementer notices.
- **Required revision:** T16 must add a dedicated `--catf-python-ctr-only` flag (mirroring
  `gen_bootstrap_dev_only`) and a `gen_catf_python_ctr_only()` wrapper, and every command in
  T16 must use that flag. Add an explicit guard step: `git status --short crates/cb-oracle/fixtures`
  must show **only** the five new files, and the task must abort if any other fixture is dirty.

### [BLOCKER-2] `simple_ctr`, `combinations_ctr` (type), `counter_calc_method` and `simple_ctr_priors` are dead in the engine — promoting them to `IMPLEMENTED` creates silently-ignored kwargs

- **Plan location:** T03, T04, T05 (Rust setters); T12 (`counter_calc_method` → `IMPLEMENTED`);
  T13 (`simple_ctr`/`combinations_ctr` → `IMPLEMENTED`, `parse_ctr_type` over all 6 variants).
- **Requirement:** SPEC-CATF-13; and the `params.rs:1-18` honesty policy (Implemented /
  KnownNotYet(rejected) / Unknown(rejected) — **never silently ignored**), which SPEC §1 itself
  cites as the justification for SPEC-CATF-12.
- **Evidence (CodeGraph + grep of the single production consumer `train_inner`):**
  - `crates/cb-train/src/boosting.rs:1853` — `ctr_type: crate::ctr::ECtrType::Borders.as_i8()`,
    hard-coded for every emitted `CtrSplitSpec`.
  - `crates/cb-train/src/boosting.rs:2821` — `params.combinations_ctr_priors.first()` is the
    prior for **all** CTRs, simple and combination alike.
  - `crates/cb-train/src/boosting.rs:4789` — `ctr_splits_for_tree(&ctr_candidates,
    &params.combinations_ctr_priors)`.
  - Zero reads of `params.simple_ctr`, `params.simple_ctr_priors`, `params.combinations_ctr`,
    `params.counter_calc_method` anywhere in `boosting.rs` (only the `BoostParams` field
    declarations at `:255-272`, `:298-304`).
- **Failure scenario:** `CatBoostRegressor(simple_ctr="Counter", counter_calc_method="Full")
  .fit(df, y, cat_features=[2])` is accepted, `_param_status` reports `"IMPLEMENTED"`, and the
  model is trained with **Borders** CTRs and `SkipTest` counting. The user gets numbers that
  silently disagree with upstream CatBoost under the same configuration, with no error.
  Symmetrically, `builder.simple_ctr_priors(vec![0.0])` has no effect while
  `combinations_ctr_priors` silently governs the simple-CTR prior.
- **Impact:** the plan's headline deliverable ("CTR config is now configurable") is false for
  3 of the 5 promoted kwargs; it introduces a class of silent wrongness strictly worse than the
  one SPEC-CATF-12 removes, because the registry now actively asserts these kwargs are honoured.
  It also invalidates SPEC §9 R4's "lockstep" rationale, which is backwards relative to the
  engine (`combinations_ctr_priors`, not `simple_ctr_priors`, is the live knob).
- **Required revision (pick one, do not leave ambiguous):**
  1. **Preferred:** keep `simple_ctr`, `combinations_ctr`, `counter_calc_method` in
     `KNOWN_NOT_YET` (rejected as parity gaps) for this plan, and shrink SPEC-CATF-13 to the two
     kwargs that are genuinely live (`one_hot_max_size` — subject to CRITICAL-4 — and
     `max_ctr_complexity`); **or**
  2. Promote them but make the accepted domain exactly the implemented domain: `parse_ctr_type`
     accepts **only** `"Borders"` and rejects the other five with an explicit
     "not yet implemented in catboost-rs" `CatBoostParameterError`; `counter_calc_method` accepts
     only `"SkipTest"`; and `simple_ctr`'s `Prior=` must either equal `combinations_ctr`'s or be
     rejected (since the engine uses one prior for both).
  In **either** case, T03/T04/T05's Rust setters must carry a doc-comment stating that the
  `ECtrType` / `CounterCalcMethod` / `simple_ctr_priors` values are currently recorded on
  `BoostParams` but not consumed by `train_inner`, with the `boosting.rs:1853` / `:2821`
  anchors — otherwise the setters are a footgun for Rust callers too.

### [CRITICAL-3] T09's `expected_cat_features()` is a data-dependent lower bound enforced as strict equality — it will reject legitimate `fit → predict` round-trips

- **Plan location:** T09 Green (`expected_cat_features()` = `max(projection member)+1` over
  every `ModelSplit::Ctr`, floored at 1; `cat_columns()` errors on `actual != expected`);
  finding F3.
- **Requirement:** SPEC-CATF-10 ("Given `m == n`, the pool's categorical columns are returned
  unchanged"); acceptance A3, A5.
- **Evidence:**
  - `cb-model/src/model.rs:272-313` — the canonical `Model` has no cat-feature-count field
    (F3 correct).
  - `crates/cb-train/src/boosting.rs:2715-2728` — `eligible_absolute` contains **only** the
    CTR-eligible cat indices (`route_categorical(card, one_hot_max_size) == Ctr`).
  - `boosting.rs:2829-2840` — candidate projections are re-indexed to **absolute** cat indices
    over that eligible subset.
  - `boosting.rs:4890-4930` + `tree.rs:2960-2990` — only the projections that actually **won a
    level** become `CtrSplitSpec`s and hence `ModelSplit::Ctr`.
  Therefore `max(member)+1` equals the true training cat width **only if** the highest-indexed
  cat column is both CTR-eligible and chosen by at least one split.
- **Failure scenarios (both reachable with default parameters):**
  1. Cat columns `[A: cardinality 20, B: cardinality 2]`, default `one_hot_max_size = 2`.
     `route_categorical(2, 2) == OneHot` ⇒ `eligible_absolute = [0]` ⇒ every projection is `{0}`
     ⇒ `expected = 1`. `model.predict(&the_very_pool_it_was_fitted_on)` (2 cat columns) returns
     `FeatureMismatch("pool has 2 categorical features, model expects 1")`.
  2. Two CTR-eligible columns, but across 5 trees × depth 2 the greedy search never selects a
     split on column 1 (e.g. it is uninformative). `expected = 1`, pool has 2 ⇒ same false
     rejection. This is *exactly* the shape of T17's oracle fixture, and the T17 localization
     ladder has **no step** that would identify this cause (step 4 is "any other divergence →
     report").
- **Impact:** the core user journey `fit(cat_pool) → predict(cat_pool)` fails with a typed error
  on data-dependent inputs. A3 and A5 become flaky. The guard also *under*-detects (it cannot
  notice a pool with the right count but the wrong column order), so it trades a real false-
  negative rate for a fabricated false-positive rate.
- **Required revision:** the model must carry the trained categorical width explicitly rather
  than deriving it. Options, in order of preference:
  1. Add `n_cat_features: usize` (or `cat_feature_count`) to `cb_model::Model`, set by
     `Model::with_ctr_data`/a new `with_cat_feature_count(pool.n_cat_features())` in T08's
     `fit()` arm. This is additive and does not touch `cb-train` — but note SPEC §7 lists
     `cb-model` as verification-only, so **the spec must be amended** (a one-field additive
     change with a `Default`-preserving serde path).
  2. If (1) is refused, `cat_columns()` must enforce `actual >= expected` (a *sufficiency*
     check, not equality) and the SPEC-CATF-10 wording ("naming both counts", `m != n`) must be
     rewritten to match. SPEC-CATF-12 is still satisfied (a zero-column pool fails `0 >= 1`).
     Document explicitly that a wider-but-correctly-ordered pool is accepted.
  In either case T09's Red test must be rewritten: today it asserts the *equality* semantics.
  Add a positive regression test "fit on N cat columns, predict on the SAME pool, must be Ok"
  parameterised over a configuration where one cat column is one-hot-routed.

### [CRITICAL-4] `one_hot_max_size` is exposed as a first-class setter/kwarg, but one-hot-routed categorical columns are silently dropped from training

- **Plan location:** T01 (setter), T12 (Python kwarg, `IMPLEMENTED`, range `[0,255]`);
  SPEC-CATF-01.
- **Requirement:** the `params.rs:1-18` honesty policy; the SPEC's own "never silently wrong"
  framing (§1).
- **Evidence:**
  - `cb_train::ObliviousTree` (`boosting.rs:775-793`) carries `splits: Vec<Split>` (float only)
    and `ctr_splits`; there is **no** one-hot split field.
  - `cb_model::ModelSplit` (`cb-model/src/model.rs:71`) has exactly two variants, confirmed by
    the exhaustive match in `apply.rs:196-201`. `from_trained` (`model.rs:326-363`) lifts only
    `Float` and `Ctr`.
  - `grow_one_hot_tree` (`cb-train/src/tree.rs:3141`) has **no production caller** — its only
    non-`lib.rs` callers are `one_hot_oracle_test.rs` (which at `:90` builds its own
    "minimal one-hot-only plain-boosting driver") and `tree_test.rs`.
  - In `train_inner`, `params.one_hot_max_size` is read only at `boosting.rs:2711` and `:2725`,
    both to *exclude* low-cardinality columns from `tensor_ctr_candidates` /
    `eligible_absolute`. Excluded columns are then referenced by nothing.
- **Failure scenario:** `CatBoostClassifier().fit(df, y, cat_features=[2])` where column 2 is a
  binary category (`"yes"/"no"`, cardinality 2). With the builder default `one_hot_max_size = 2`,
  `route_categorical(2, 2) == OneHot` ⇒ the column enters no CTR projection and no one-hot split
  ⇒ **the feature contributes nothing to the model**, with no warning. Upstream CatBoost would
  one-hot encode it. Predictions diverge arbitrarily from upstream; the ≤1e-5 parity bar is
  violated for a very common data shape.
- **Impact:** the plan's headline capability ("categorical features are now usable through the
  facade") is false for low-cardinality categoricals — arguably the most common categorical kind.
  Every test in the plan (T08, T16, T17) pins `one_hot_max_size(1)`, which structurally excludes
  this case, so nothing in the plan would detect it.
- **Required revision:** add a task that makes the gap explicit rather than silent. Minimum
  acceptable: in `CatBoostBuilder::fit`'s categorical arm, compute each cat column's
  `learn_set_cardinality` (already public: `cb_train::learn_set_cardinality`) and return a typed
  `CatBoostError` naming any column that routes to `EncodingPath::OneHot`, with a message stating
  that one-hot categorical training is not yet implemented and suggesting `one_hot_max_size=1`.
  Add the matching spec entry and a dedicated test (cardinality-2 column + default
  `one_hot_max_size` ⇒ typed error, not a silently degraded model). `EncodingPath::Skip`
  (constant column) may be accepted with no error since upstream also skips it — assert that
  distinction in the test.

### [MAJOR-5] T14's `data_to_pool` arity change misses four call sites; two files are absent from the task's file list

- **Plan location:** T14 Green step 2 ("Update ALL seven existing `data_to_pool(py, x, None)`
  call sites … (verified list above)"); T14 "Files".
- **Evidence:** grep of `crates/catboost-rs-py/src/` finds 14 `data_to_pool` call sites. The
  plan lists 10 (3 `fit` sites + 7 "must keep compiling"). **Missing:**
  `crates/catboost-rs-py/src/search.rs:304`, `search.rs:389`,
  `crates/catboost-rs-py/src/cv.rs:61`, `cv.rs:64`.
- **Failure scenario:** T14 compiles `estimator.rs`, `regressor.rs`, `classifier.rs`,
  `ranker.rs` and stops with `error[E0061]: this function takes 4 arguments but 3 were supplied`
  in `search.rs` and `cv.rs`. The implementer must then edit two files the task declares
  out of its scope, without guidance on what to pass.
- **Impact:** build break mid-task; the plan's "verified list" claim is false, which erodes
  trust in the rest of the anchors. Secondarily, it means Python `catboost_rs.cv(...)` and
  `catboost_rs.grid_search(...)` gain no `cat_features` kwarg — a coverage nuance worth an
  explicit decision.
- **Required revision:** add `crates/catboost-rs-py/src/search.rs` and
  `crates/catboost-rs-py/src/cv.rs` to T14's file list, enumerate all four sites, and state
  explicitly that they pass `None` (and record "`cat_features` on `cv`/`grid_search`" as a
  follow-up, or add it).

### [MAJOR-6] `cv()` and `grid_search()` on a categorical pool become a failure path, with no spec entry, no test, and a silent-degradation mode

- **Plan location:** SPEC §7 "Callers inheriting the new behavior with no signature change";
  T18 (which gates only the float-only path).
- **Evidence:**
  - `crates/catboost-rs/src/cv.rs:398-409` — `run_fold` does `pool.select_rows(...)` →
    `builder.fit(&train_pool)?` → `model.staged_predict(&test_pool, ...)?`.
  - `crates/cb-data/src/pool.rs:232-236` — `select_rows` **preserves** `cat_features`.
  - `crates/catboost-rs/src/model.rs:164-169` — `ensure_scalar_oblivious`, called by
    `staged_predict`, rejects `ctr_data.is_some()` with `UnsupportedModel`.
  - SPEC §2 explicitly requires that rejection be **preserved, not extended**.
  - `crates/catboost-rs-py/tests/test_cv.py:114-135` documents today's behaviour and is written
    tolerantly (`try/except CatBoostError: return`), so it will *pass either way* and will not
    surface the change. (Note it uses a NumPy `x`, so the cat columns are dropped at ingestion
    — it will in fact keep taking the clean-completion path.)
- **Failure scenario:** a user calls `catboost_rs.cv(Pool(df, y, cat_features=[2]), ...)`.
  Before: returns curves (ignoring the cat column). After: `UnsupportedModel`. Worse, in
  `grid_search` with `ErrorScore::Value(nan)` (the sklearn default), **every** candidate fails,
  `warn_fit_failed` emits a warning, and a `SearchResult` with all-NaN scores and an arbitrary
  `best_index` is returned — a silent degradation, not an error.
- **Impact:** a user-visible behaviour change on two shipped public APIs, uncovered by any
  regression gate in the plan and unlisted in SPEC §8 "Compatibility and migration" (which
  claims exactly one deliberate behaviour change).
- **Required revision:** add a spec entry and a task. Concretely: (a) document the change in
  SPEC §8; (b) add a Rust test `cv_on_categorical_pool_is_typed_unsupported_model` asserting
  the typed error rather than a silent NaN; (c) decide and record whether `grid_search` should
  fail fast on a categorical pool rather than converting the failure into `error_score` (a
  candidate-independent failure is not a candidate failure).

### [MAJOR-7] SPEC-CATF-12's "never silently mis-scored" is enforced only on `predict_with`; four other public paths remain silently wrong on the newly-reachable CTR models

- **Plan location:** T11; SPEC-CATF-12 and SPEC §2 non-goals.
- **Evidence:**
  - `Model::shap_values` (`catboost-rs/src/model.rs:290-293`) → `cb_model::shap_values`, which
    projects splits via `float_splits_of` (`cb-model/src/shap.rs:550-552`) — CTR splits are
    dropped, no error.
  - `Model::partial_dependence` (`model.rs:424-431`) — `feature_columns` only.
  - `Model::feature_importance_with_data(PredictionValuesChange)` (`model.rs:369-378`) passes
    `pool.cat_features().to_vec()` with **no width validation** — a zero-cat pool reproduces the
    exact empty-string-category defect SPEC §1 describes.
  - `Model::feature_importance` (structure-only) likewise.
  - **Verified safe:** `Model::predict_raw_on_device` (`model.rs:249-253`) guards via
    `cb_model::flatten_oblivious_f64`, which rejects categorical/CTR models.
- **Failure scenario:** a user fits a CTR model through the newly-wired facade and calls
  `shap_values(&pool)` — receives a full, plausible SHAP matrix computed from the float splits
  alone (i.e. from an essentially empty tree structure for a categorical-only model), with no
  error and a violated local-accuracy invariant.
- **Impact:** before this plan these paths were unreachable with a CTR model through the facade
  (you could only get one by `load_cbm`); this plan makes them routinely reachable. SPEC §2's
  "preserve, do not extend" instruction, read literally, forbids fixing this — which makes the
  spec internally inconsistent with SPEC-CATF-12's stated principle.
- **Required revision:** amend SPEC §2 and add a task extending the existing
  `ensure_scalar_oblivious`-style typed rejection (or at minimum a `cat_columns()` width check)
  to `shap_values`, `partial_dependence`, and `feature_importance_with_data`. If the maintainer
  insists on deferring, the plan must add an explicit, tested `UnsupportedModel` rejection
  rather than leaving silently-wrong numbers — a rejection is cheap, additive, and precisely
  the precedent `staged_predict` already sets.

### [MAJOR-8] Three validation commands use invalid `cargo test` syntax

- **Plan location:** T09 (`cargo test -p catboost-rs --test model_cat_predict_test::predict_on_ctr_model_with_wrong_cat_width`), T10 (`--test model_cat_predict_test::ctr_model_predict_matches`), T11 (`--test model_cat_predict_test::ctr_model_predict_on_cat_free_pool`).
- **Evidence:** `--test <NAME>` selects a **test target**; a name filter is a separate
  positional argument after `--`. `crates/catboost-rs/tests/` contains no target named
  `model_cat_predict_test::…`.
- **Failure scenario:** `error: no test target named 'model_cat_predict_test::…'` — the Red step
  cannot even be executed, and an implementer may misread it as the expected Red.
- **Impact:** blocks the TDD loop for the entire Wave-2 (`model.rs`) lane.
- **Required revision:** `cargo test -p catboost-rs --test model_cat_predict_test -- <filter>`.

### [MINOR-9] T06's mutation check does not exercise the `boost_params()` wiring, and cannot detect a `simple_ctr`/`combinations_ctr` cross-wire

- **Evidence:** the mandated mutation edits a `new()` default
  (`simple_ctr_priors: vec![0.0]`). `simple_ctr_default()` and `combinations_ctr_default()` both
  return `ECtrType::Borders` (`boosting.rs:464-466`, `:509-511`), so a swapped read in
  `boost_params()` is invisible to T06. (T03/T04 *do* catch it because they set distinct
  non-default values — so the plan is safe overall, but T06's stated guarantee is overstated.)
- **Required revision:** add a second mandated mutation on a `boost_params()` **read** (e.g.
  `one_hot_max_size: self.max_ctr_complexity as u32`), and record both failure texts.

### [MINOR-10] SPEC-CATF-14 promises `list[int | str]`; T14 ships `list[int]` only

- T14 correctly verifies `Pool.__new__` takes `Option<Vec<usize>>` and correctly refuses to
  introduce name-based selection. But the divergence is recorded only as a plan follow-up.
- **Required revision:** amend SPEC-CATF-14's Input line to `Optional[list[int]]` with a
  one-line note, so the spec and the deliverable agree.

### [MINOR-11] T14's post-ingestion width guard misfires on duplicate indices

- `cat_features=[2,2]` yields `requested.len() == 2` but one ingested cat column, producing the
  "declared 2 … carries 1" message which misattributes the cause.
- **Required revision:** de-duplicate (and range-check) `cat_features` before the comparison, or
  reject duplicates explicitly.

### [MINOR-12] `CatBoostBuilder` derives `PartialEq`/`Debug`; adding seven fields changes both

- `builder.rs:64` `#[derive(Debug, Clone, PartialEq)]`. Two builders that were equal may now
  compare unequal, and `Debug` output widens. No semantic consumer found
  (`grid_search::SearchResult` documents that builders are excluded from equality), so this is
  informational — but T04/T05 should re-run `cargo test -p catboost-rs -p catboost-rs-py` for
  any `assert_eq!` on a builder or on a `Debug` string.

---

## Plan-accuracy note in the plan's favour (not an issue)

The plan calls Assumption A1 / finding F5 / risk R9 (`permutation_count = 4` on the CTR path)
"the single largest schedule risk in the plan". **It is already closed.**
`crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs` trains the *same*
`tensor_ctr_e2e` corpus through `train_cat` at `permutation_count: 4`, `max_ctr_complexity: 2`,
`one_hot_max_size: 1`, `boosting_type: Plain`, `fold_len_multiplier: 2.0`,
`random_strength: 0`, seed 0, Logloss — and asserts ≤1e-5 against a committed upstream
reference (`crates/cb-train/tests/fixtures/multi_permutation_fold/predictions_pc4.npy`),
predicting through the production `cb_model::predict_raw_cat` path. This simultaneously
de-risks Assumption A2 (the ORD-06/07 gating bug does not bite this corpus even at
`max_ctr_complexity = 2`). The plan's *decision* (generate the fixture at 4) is correct; only
its risk framing is wrong. T17's localization ladder step 1 should simply cite this test
instead of proposing a scratch Rust run.

---

## Implementation Order Review

The dependency graph is acyclic and the edit-conflict serializations are correct:
`builder.rs` (T01→T02→T03→T04→T05→T06→T08), `model.rs` (T09→T10→T11), `params.rs` (T12→T13)
each serialize on a single file; `T07` (`lib.rs` + a new `tests/` file) and `T16`
(`gen_fixtures.py` + `fixtures/`) genuinely do not overlap and may run concurrently.
T14's parallelism claim with T12/T13 is **invalid as written** because of MAJOR-5: once
`search.rs` and `cv.rs` are added to T14, there is still no overlap with `params.rs`, so the
claim survives the fix — but T14 must be re-checked after the file list is corrected.

Required ordering changes:

1. **Insert a new T00 before T01**: resolve BLOCKER-2 — decide whether `simple_ctr`,
   `combinations_ctr`, `counter_calc_method` are promoted-with-narrowed-domain or stay
   `KNOWN_NOT_YET`. This decision changes T03/T04/T05's doc-comments and T12/T13's entire
   content, so it cannot be deferred to implementation time.
2. **Insert a task before T09** that gives `cb_model::Model` an explicit trained cat-feature
   count (CRITICAL-3 option 1), or amend SPEC-CATF-10 to a sufficiency check (option 2).
   T09 as written cannot be implemented against the current spec text.
3. **Insert a task after T08** for the one-hot rejection (CRITICAL-4). It must land with T08,
   not after T17, because T14's Python surface makes the bad configuration trivially reachable.
4. **Move T16 earlier only after adding the `--catf-python-ctr-only` flag** (BLOCKER-1); until
   then T16 is unsafe to run at all.
5. **Add a cv/grid_search categorical regression task between T08 and T18** (MAJOR-6).

Everything else in the ordering is sound: contracts (setters) before consumers (`fit`), the
model-side validation helper before its consumer (`predict_with`), the re-exports before the
PyO3 crate that names them, and the frozen fixture before the oracle that reads it.

---

## Potential Bugs (beyond the issues above)

- **`combinations_ctr=[]` is unrepresentable.** T13 rejects a list whose length ≠ 1, but the
  committed `plain_ctr` fixture uses `"combinations_ctr": []`. Any future fixture or user
  configuration that disables combination CTRs has no Python expression. Low impact under
  `max_ctr_complexity=1` (where combinations do not arise), but the grammar contract should
  state what an empty list means (accept and leave the builder default, or reject with a
  message that says why).
- **`Prior=<num>/<den>` rejection is right but under-specified.** T13 rejects the fractional
  form because `simple_ctr_priors: Vec<f64>` cannot carry a denominator. Note that the engine
  itself pins `prior_denom: 1.0` (`boosting.rs:1855`), so the rejection is correct and should
  cite that anchor.
- **T12's `check_range("one_hot_max_size", .., 0.0, 255.0)`** is Assumption A4 and unverified
  (`catboost-master` does not contain `libs/options/cat_feature_options.cpp` in this checkout).
  Given CRITICAL-4 the practical usable range is `{0, 1}`; whatever bound ships must not
  contradict the one-hot rejection.
- **`max_ctr_complexity` upper bound `[1, 8]`** is likewise unverified. `enumerate_projections`
  is combinatorial in the eligible-feature count; a large value on a wide categorical frame is a
  performance cliff. Worth a comment, not a blocker.
- **T17's non-degeneracy guard** (`proba[:,1].std() > 1e-6`) is good. Add the same guard to the
  fit-kwarg twin, and add an assertion that the fitted model actually carries ≥1
  `ModelSplit::Ctr` (reachable from Python only indirectly — otherwise a configuration that
  silently degenerates to "no CTR splits at all" could still pass at 1e-5 if upstream also
  degenerates).

---

## Required Plan Revisions (checklist)

1. T16: add a `--catf-python-ctr-only` flag to `gen_fixtures.py` and use it in every command;
   add a "no other fixture is dirty" guard. **[BLOCKER-1]**
2. New T00: decide and record the honest domain for `simple_ctr` / `combinations_ctr` /
   `counter_calc_method` (reject, or narrow to the implemented `Borders` / `SkipTest` domain);
   update T03/T04/T05 doc-comments with the `boosting.rs:1853` / `:2821` anchors.
   **[BLOCKER-2]**
3. Replace T09's derived width with an explicitly stored trained cat-feature count (amend
   SPEC §7's "cb-model is verification-only"), or change SPEC-CATF-10 to a `actual >= expected`
   sufficiency check; add a positive `fit → predict same pool` regression test parameterised
   over a one-hot-routed trailing column. **[CRITICAL-3]**
4. Add a task rejecting `EncodingPath::OneHot` categorical columns in `fit()`'s categorical arm
   with a typed error, plus a spec entry and a cardinality-2 + default-`one_hot_max_size` test.
   **[CRITICAL-4]**
5. T14: add `crates/catboost-rs-py/src/search.rs` and `crates/catboost-rs-py/src/cv.rs` to the
   file list and enumerate all 14 `data_to_pool` sites. **[MAJOR-5]**
6. Add a cv/grid_search categorical-pool spec entry, SPEC §8 compatibility note, and a test;
   decide whether `grid_search` should fail fast rather than converting the failure into
   `error_score`. **[MAJOR-6]**
7. Amend SPEC §2 and add typed CTR rejections (or width checks) for `shap_values`,
   `partial_dependence`, `feature_importance_with_data`. **[MAJOR-7]**
8. Fix the three `cargo test --test <target>::<filter>` commands to
   `--test <target> -- <filter>`. **[MAJOR-8]**
9. T06: add a second mandated mutation on a `boost_params()` read. **[MINOR-9]**
10. SPEC-CATF-14: narrow `list[int | str]` to `list[int]`. **[MINOR-10]**
11. T14: de-duplicate / range-check `cat_features` before the width guard. **[MINOR-11]**
12. T17: cite `multi_permutation_e2e_oracle_test.rs` in ladder step 1 and downgrade the A1/R9
    risk framing.

---

## Unverified Items

- **`ctr_border_count_default()` parity with upstream `CtrBorderCount` (default 15).** Not read
  in this pass; inferred to be correct because the existing CTR fixtures pass without pinning
  it. T16's fixture likewise does not pin it — acceptable but unproven for the new
  configuration.
- **Whether the T16/T17 configuration actually yields ≥1 CTR split on both cat columns.** Data-
  dependent; this is precisely the exposure described in CRITICAL-3 and cannot be settled
  without running the fixture generation.
- **The upstream domains for `one_hot_max_size` and `max_ctr_complexity`** (plan Assumption A4).
  `catboost-master/` in this checkout contains only `catboost/private/libs/algo`, so
  `libs/options/cat_feature_options.cpp` could not be read.
- **`cb_train::ObliviousTree.ctr_splits` ordering vs. `level_kinds`** — the plan does not depend
  on it, and `from_trained` appends CTR splits after float splits with the leaf index assigned
  by `leaf_index_for`; the existing oracles cover this, so it was not re-derived here.
- **Whether any currently-green test asserts on `CatBoostBuilder`'s `Debug` or `PartialEq`.**
  Not exhaustively searched (MINOR-12); the regression command in T04/T05 would surface it.
