---
title: TDD implementation plan — categorical-feature (cat_features/CTR) routing through the CatBoostBuilder facade
spec: ./SPEC.md
research: ./research.md
status: ready-for-implementation
plan_version: 1
created_at: 2026-07-31
task_count: 18
spec_ids_covered: SPEC-CATF-01 .. SPEC-CATF-17 (all 17)
---

# TDD Implementation Plan

Companion to the LOCKED `SPEC.md` in this directory. The spec is authoritative and
is **not** re-derived here. This document is the executable task list.

Every `[VERIFIED]` claim below was re-checked against the working tree on
2026-07-31 via CodeGraph MCP (`codegraph_explore`), direct reads, or a executed
shell command. Anchors quoted from `research.md` were re-verified, not trusted.

---

## 0. Resolutions to the three open questions (SPEC.md §9)

### OQ-1 — Are `simple_ctr_priors` / `combinations_ctr_priors` standalone upstream Python kwargs?

**Resolution: NO. They do not exist as standalone kwargs. Priors ride the
`simple_ctr` / `combinations_ctr` CTR-description string grammar.**

Evidence chain:

1. `test_params.py` resolves its authoritative upstream vocabulary from
   `_CORE_PY = <repo>/catboost-master/catboost/python-package/catboost/core.py`
   `[VERIFIED: crates/catboost-rs-py/tests/test_params.py:20-29]`.
2. **That path does not exist in this checkout.** `catboost-master/` contains only
   `catboost/private/libs/algo` — no `python-package`
   `[VERIFIED: shell — find catboost-master -name "*.py" returns nothing;
   ls catboost-master/catboost/python-package/catboost/ → No such file]`.
   Consequently `test_every_upstream_param_is_in_registry` **currently SKIPS**
   `[VERIFIED: shell — .venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q -rs
   → "SKIPPED [1] ... vendored core.py not found"; 4 passed, 1 skipped]`.
   This is a **material correction** to SPEC-CATF-15's premise and is handled by
   Task T15.
3. The equivalent authoritative source that *does* exist is the installed
   `catboost==1.2.10` package
   `[VERIFIED: .venv/lib/python3.12/site-packages/catboost/core.py]`. Grepping it:
   - `simple_ctr=None` at `core.py:5366`, `combinations_ctr=None` at `:5367`
     (CatBoostClassifier `__init__`), plus the Regressor (`:6001-6002`) and
     Ranker (`:6409-6410`) mirrors.
   - **Zero** occurrences of `simple_ctr_priors` or `combinations_ctr_priors`
     anywhere in `core.py`.
   - The documented type is `list of strings`, with the grammar
     `'Borders:CtrBorderCount=5:Prior=0:Prior=0.5'`
     `[VERIFIED: core.py:4957-4972]`.

**Planning consequence:**

- `simple_ctr_priors` / `combinations_ctr_priors` stay **Rust-only** builder
  setters (still shipped in lockstep per the locked user decision — Tasks T03/T04).
- They are **NOT** added to `VOCABULARY` in `params.rs`. Adding a non-upstream
  name to `VOCABULARY` would corrupt the registry's stated contract ("the
  AUTHORITATIVE list transcribed from upstream `core.py`",
  `[VERIFIED: crates/catboost-rs-py/src/params.rs:59-67]`).
- The Python `simple_ctr` / `combinations_ctr` surface therefore accepts the
  upstream-shaped **CTR-description grammar**, from which the type AND the priors
  are parsed (Task T13). Accepted forms: a bare type string (`"Borders"`), a
  `"<Type>:Prior=<f>[:Prior=<f>…]"` descriptor, or a 1-element list of either
  (upstream passes a list). Any other grammar component
  (`CtrBorderCount=`, `TargetBorderCount=`, `TargetBorderType=`) is REJECTED with a
  `CatBoostParameterError` naming it — never silently dropped (honesty policy,
  `[VERIFIED: params.rs:1-18]`).

### OQ-2 — Which estimators get the `cat_features` fit-kwarg?

**Resolution: ALL of them (all three this repo exposes).**

Upstream has `cat_features` on all four `fit` signatures
`[VERIFIED: .venv/.../catboost/core.py — CatBoost.fit:2758, CatBoostClassifier.fit:5435,
CatBoostRegressor.fit:6068, CatBoostRanker.fit:6476]`.

This repo exposes exactly three estimator classes — there is no bare `CatBoost`
class `[VERIFIED: crates/catboost-rs-py/src/lib.rs:47-50 registers
CatBoostRegressor, CatBoostClassifier, CatBoostRanker, Pool]`. All three currently
have `#[pyo3(signature = (x, y = None))]`
`[VERIFIED: regressor.rs:51-52, classifier.rs:60-61, ranker.rs:57-58]`.

**Planning consequence:** all three gain `cat_features = None` as the third
parameter (Task T14).

*Recorded divergence:* upstream `CatBoostRanker.fit` places `group_id` third and
`cat_features` fourth. This repo's Ranker `fit` has no `group_id` kwarg (grouping
comes only from a `Pool`), so `cat_features` takes the third slot on all three
classes for internal consistency. Documented in the Ranker docstring by T14.

### OQ-3 — `Pool` + `cat_features` fit-kwarg collision: ignore or error?

**Resolution: EXPLICIT TYPED ERROR.**

This overrides the "documented ignore, consistent with `y`" option, on verified
upstream evidence:

```python
# .venv/lib/python3.12/site-packages/catboost/core.py:1522-1533  [VERIFIED]
def _build_train_pool(X, y, cat_features, ...):
    if isinstance(X, Pool):
        train_pool = X
        if any(v is not None for v in [cat_features, text_features, ...]):
            raise CatBoostError(
                "cat_features, text_features, embedding_features, sample_weight, ... "
                "should have the None type when X has catboost.Pool type."
            )
        ...
        if y is not None:
            raise CatBoostError("Incorrect value of y: X is catboost.Pool object, ...")
```

Upstream raises for **both** `cat_features` and `y` on the `Pool` fast-path.
Choosing "ignore" would diverge from upstream **and** from the crate's own
never-silently-ignore-a-kwarg doctrine (`params.rs:1-18`). Choosing "error" is
upstream-exact for `cat_features`.

**Justification for the asymmetry with `y`:** the existing `y`-is-ignored behavior
on the `Pool` fast-path (`estimator.rs:230-232`, WR-04) is a **pre-existing
upstream divergence**. It is NOT changed here — flipping it would alter the
observable behavior of an already-shipped, already-tested surface with no spec
coverage in `SPEC.md`. It is recorded below as a follow-up item. The new parameter
is shipped upstream-correct from day one rather than being made bug-compatible
with an old divergence.

**Follow-up recorded (out of this plan's scope):** align the `Pool` + `y` fast-path
with upstream's `CatBoostError` in a later plan; it needs its own spec entry and a
compatibility note.

---

## 0b. Additional verified findings that shape the plan

These were discovered during verification and are NOT in `SPEC.md` or
`research.md`. Each changes a task's design.

| # | Finding | Evidence | Plan consequence |
|---|---|---|---|
| F1 | `crates/catboost-rs/src/builder_test.rs` does not exist. `boost_params()` is a **private** `fn` (`builder.rs:259`), so a unit test must be mounted as a **CHILD** module of `builder`, not registered in `lib.rs`. A `mod builder_test;` line in `lib.rs` would make it a SIBLING of `builder` and it could not see `boost_params`. | `[VERIFIED: ls crates/catboost-rs/src/ — no builder_test.rs; builder.rs:259 `fn boost_params(&self)` has no `pub`; lib.rs:68-79 registers 6 sibling test mods; the child-mount precedent is cv.rs:551-553 `#[cfg(test)] #[path = "cv_test.rs"] mod cv_test;`]` | T01 creates `builder_test.rs` AND mounts it with the `#[path]` child pattern at the bottom of `builder.rs`. Test filter becomes `builder::builder_test::…`. |
| F2 | `numpy_to_owned(x, y)` takes **no** `cat_features` argument — the NumPy branch of `ingest_to_owned` **silently drops** a supplied `cat_features`. | `[VERIFIED: ingest_py.rs:52-67 — branch 3 calls `numpy_to_owned(x, y)`; `pub(crate) fn numpy_to_owned(x, y)` at :99-107 has no cat param]` | T14 must not let `fit(np_array, y, cat_features=[…])` silently ignore the kwarg. Because `SPEC.md §7` lists `ingest_py.rs` as **verification-only / must not be modified**, the guard is implemented in `estimator.rs::data_to_pool` as a **post-ingestion width assertion** (`pool.n_cat_features() == cat_features.len()`), which covers the NumPy drop and any future silent-drop path without touching `ingest_py.rs`. |
| F3 | `cb_model::Model` carries **no** field recording the trained categorical-feature count. | `[VERIFIED: cb-model/src/model.rs:272-313 — fields are oblivious_trees, non_symmetric_trees, region_trees, bias, float_feature_borders, ctr_data, approx_dimension, class_to_label. No cat count.]` | T09 derives the expected cat width from the CTR splits' projections: `max(projection.cat_features()) + 1` over every `ModelSplit::Ctr`, floored at `1` when `ctr_data.is_some()`. Documented as a derived lower bound. |
| F4 | `CtrSplit` carries `projection: cb_train::TProjection`; `apply` gathers member values by `projection.cat_features()` indices into the full per-object cat row. | `[VERIFIED: cb-model/src/model.rs:43-63; cb-model/src/apply.rs:157-168]` | T09's derivation is sound against the same indexing the apply path uses. |
| F5 | The builder pins `permutation_count: permutation_count_default()` = **4**, but every committed CTR fixture pins `permutation_count: 1`. There is no builder setter for it and it is out of this spec's scope. | `[VERIFIED: builder.rs:290 pins the default; cb-train/src/boosting.rs:448-450 `permutation_count_default() -> 4`; fixtures/plain_ctr/config.json and fixtures/tensor_ctr_e2e/config.json both set `"permutation_count": 1`]` | **The new CTR oracle fixture (T16) MUST be generated at `permutation_count=4`** so both sides agree. If the ≤1e-5 gate then fails, T17 runs the localization probe and **STOPS AND REPORTS** a permutation_count parity gap — it does not add a setter (scope leak). |
| F6 | `cb-train` is single-leaf-estimation-step; upstream defaults `leaf_estimation_iterations` above 1 for several losses. The builder does not surface it. | `[VERIFIED: gen_fixtures.py:150 comment "leaf_estimation_iterations=1 -> single Newton/Gradient leaf step"; boost_params() at builder.rs:260-317 sets no leaf_estimation_iterations field]` | T16's generator MUST pin `leaf_estimation_iterations=1` on the upstream side. |
| F7 | `score_function` default: builder = `score_function_default()` (Cosine, the catboost CPU default); the CTR fixtures do not pin it (so upstream also uses Cosine). | `[VERIFIED: builder.rs:115 + :304-308 comment; fixtures/tensor_ctr_e2e/config.json has no score_function key]` | T16 pins `score_function="Cosine"` explicitly on BOTH sides rather than relying on matching defaults. |
| F8 | `random_strength`: builder default is `0.0`; the sklearn-facing upstream default is `1`, the raw dict-API default is `0`. | `[VERIFIED: builder.rs:107 `random_strength: 0.0`; gen_fixtures.py:156-170 ISOLATING_PARAMS pins `"random_strength": 0`; memory cv-orch01-random-strength-fixture]` | T16 pins `random_strength=0` explicitly on BOTH sides (the known trap). |
| F9 | The real Python test environment is the **repo-root `.venv`** (Python 3.12.13, `catboost==1.2.10`, `pytest 9.1.1`, `maturin` present). A `crates/catboost-rs-py/.venv-py8` directory DOES exist but has **no** `catboost` and no `pytest` — it is a dead shell. | `[VERIFIED: shell — .venv/bin/python -m pytest --version → pytest 9.1.1; .venv/bin/python -c "import catboost" → 1.2.10; crates/catboost-rs-py/.venv-py8/bin/python -c "import catboost" → ModuleNotFoundError]` | Every Python command in this plan uses `.venv`. |
| F10 | `cb-model`, `cb-train`, `cb-data` are **regular** `[dependencies]` of `catboost-rs`, so `crates/catboost-rs/tests/*.rs` may `use cb_model::…` directly. | `[VERIFIED: crates/catboost-rs/Cargo.toml:23-40]` | T08's anti-false-pass guard can name `cb_model::ModelSplit` from an integration test. |
| F11 | Pandas categorical ingestion requires categorical columns to be **TRAILING**; an interleaved cat column is rejected. Arrow/Polars rejects non-empty `cat_features` outright. | `[VERIFIED: ingest_py.rs:196-263 (trailing rule + error text); :333-342 (Arrow rejection)]` | T16's fixture DataFrame is categorical-only (all cat columns trailing by construction). T14's docstrings state the Pandas/NumPy-object-only limitation. |
| F12 | `CatBoostBuilder` derives `Debug, Clone, PartialEq` (not `Copy`). | `[VERIFIED: builder.rs:60-65]` | Adding `Vec<f64>` priors fields is derive-compatible. `ECtrType` and `CounterCalcMethod` both derive `Debug, Clone, Copy, PartialEq, Eq` `[VERIFIED: cb-train/src/ctr/mod.rs:65-67, :129-130]`, so no derive breaks. |

---

## 1. Goal-backward derivation

SPEC.md §6 acceptance scenarios A1–A10 become the following observable end states.
Tasks are derived from these, not from file layout.

| Acceptance | Observable end state | Delivered by |
|---|---|---|
| A10 | `CatBoostBuilder::new(..)` with no CTR setter called produces `boost_params()` CTR fields byte-equal to today's `*_default()` values | T01–T06 |
| A2 | `fit(&cat_pool)` returns a model with `ctr_data.is_some()` and ≥1 `ModelSplit::Ctr` | T08 |
| A5 | `predict` on a CTR model with wrong cat width → `FeatureMismatch` naming both counts | T09 |
| A3 | `fit` → `predict` on the same categorical pool matches upstream ≤1e-5 | T10, T16, T17 |
| A4 | `predict`/`predict_proba`/`predict_with` on a CTR model with **zero** cat columns → typed `FeatureMismatch` | T11 |
| A6 | `CatBoostRegressor(max_ctr_complexity=2).fit(...)` accepted; value reaches `BoostParams` | T12 |
| A7 | `simple_ctr="Bogus"` → `CatBoostParameterError` listing valid variants | T13 |
| A8 | `fit(df, y, cat_features=[2])` ≡ `Pool(df, y, cat_features=[2])` | T14 |
| A9 | `_param_status` returns `"IMPLEMENTED"` for the promoted set; `nan_mode` stays `KNOWN_NOT_YET` | T15 |
| A1 | The three existing facade oracle tests pass **unmodified** | T18 |

Nameability of `catboost_rs::ECtrType` / `catboost_rs::CounterCalcMethod` (SPEC-CATF-09)
is a compile-time prerequisite of T12/T13 → T07.
The frozen fixture (SPEC-CATF-17) is a data prerequisite of T17 → T16.

---

## 2. Execution waves and dependency graph

```text
WAVE 1 (two independent lanes, run concurrently)

  lane A — builder.rs (STRICTLY SERIAL: single-file edit conflict)
    T01 -> T02 -> T03 -> T04 -> T05 -> T06 -> T08

  lane B — independent files
    T07  (crates/catboost-rs/src/lib.rs + new tests/ file)
    T16  (crates/cb-oracle/generator/gen_fixtures.py + fixtures/)

WAVE 2 — model.rs (STRICTLY SERIAL: single-file edit conflict); needs T08
    T09 -> T10 -> T11

WAVE 3 — python bindings
    T12 -> T13        (both edit params.rs: SERIAL)   needs T05, T07
    T14               (estimator.rs + regressor/classifier/ranker.rs)  needs T08, T10
    T12,T13,T14 -> T15

WAVE 4 — oracle gate
    T15, T16, T11 -> T17

WAVE 5 — final regression gate
    T17 -> T18
```

Edge list (acyclic):

```text
T01->T02  T02->T03  T03->T04  T04->T05  T05->T06  T06->T08
T08->T09  T09->T10  T10->T11
T05->T12  T07->T12  T12->T13
T08->T14  T10->T14
T12->T15  T13->T15  T14->T15
T15->T17  T16->T17  T11->T17
T17->T18
```

Parallelizable pairs (verified no file overlap, no contract dependency):
`{T07, T16}` with the whole of lane A; `T14` with `{T12,T13}` *only after* both
T08 and T10 land (different files: `estimator.rs`/`regressor.rs`/`classifier.rs`/
`ranker.rs` vs `params.rs`).

Everything in lane A serializes on `crates/catboost-rs/src/builder.rs`; everything
in Wave 2 serializes on `crates/catboost-rs/src/model.rs`; T12/T13 serialize on
`crates/catboost-rs-py/src/params.rs`. These are edit-conflict serializations, not
logical dependencies — noted per task.

---

## 3. Shared conventions every task must honor

- **No `unwrap()`/`expect()`/`panic!()`/raw indexing in production code.** Workspace
  lints deny all four `[VERIFIED: Cargo.toml:9-14 `[workspace.lints.clippy]`
  unwrap_used/expect_used/panic/indexing_slicing = "deny"]`. Test code is exempted
  in-code via `#![cfg_attr(test, allow(...))]` at the crate root
  `[VERIFIED: crates/catboost-rs/src/lib.rs:1]`, and integration tests via a
  file-level `#![allow(...)]` `[VERIFIED: tests/builder_oracle_test.rs:34]`.
- **Source/test separation.** No `#[cfg(test)] mod tests` in a production file.
  Unit tests go in a sibling `*_test.rs`; integration tests in `crates/<crate>/tests/`.
- **Builder setters** are `#[must_use] pub fn name(mut self, name: T) -> Self`
  `[VERIFIED: the shape of every setter in builder.rs:119-253]`.
- **`boost_params()` comment discipline:** when a field gains a setter, DELETE its
  "pinned / inert here" comment and replace it with a setter reference
  `[VERIFIED: builder.rs:284-308 shows both the pinned form and the already-migrated
  `score_function` form]`.
- **Do NOT touch** `cb-train`, `cb-model`, `cb-data`, or
  `crates/catboost-rs-py/src/ingest_py.rs` (SPEC §7 verification-only).
- **Do NOT** attempt any fix to the ORD-06/ORD-07 combination-CTR gating bug
  (`.planning/phases/24-ctr-split-search-correctness/`). If an oracle fails ONLY for
  `max_ctr_complexity > 1`, **STOP AND REPORT**.

### Guard-test falsifiability protocol (used by T06 and T18)

SPEC-CATF-06 and SPEC-CATF-07 are **regression guards**: a correctly-implemented
predecessor makes them pass on first write, so a classic Red is unavailable. For
these two tasks only, the Red step is replaced by an explicit **mutation check**:

1. Write the test. Run it. Record the result.
2. Apply the named single-line mutation to production code.
3. Re-run; the test MUST fail with the named message. Record the failure text.
4. **Revert the mutation** (`git checkout -- <file>` or manual revert) and re-run
   to confirm green.

A guard test that cannot be made to fail by its named mutation is not a guard and
the task is not complete.

### Repository-verified commands

```bash
# Rust unit tests for the builder's child test module (T01-T06, T08 partly)
cargo test -p catboost-rs --lib builder::builder_test

# A single Rust integration test file
cargo test -p catboost-rs --test builder_cat_facade_test
cargo test -p catboost-rs --test model_cat_predict_test
cargo test -p catboost-rs --test reexport_ctr_types_test

# The three existing float-only oracle regression gates (MUST stay unmodified)
cargo test -p catboost-rs --test builder_oracle_test \
                          --test cv_oracle_test \
                          --test grid_search_oracle_test

# Lint gate (workspace restriction lints are clippy-only, not rustc)
cargo clippy -p catboost-rs -p catboost-rs-py --all-targets

# Rebuild + install the Python extension into the repo-root .venv
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop \
    -m crates/catboost-rs-py/Cargo.toml --features cpu

# Python tests
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_categorical.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q

# Fixture generation (OFFLINE, RUN ONCE, then COMMIT — never in CI)
.venv/bin/python crates/cb-oracle/generator/gen_fixtures.py <scenario>
```

`maturin develop` flags verified via `.venv/bin/maturin develop --help`
(`-F/--features`, `-m/--manifest-path` both present).

---

## 4. Tasks

---

### T01 — `one_hot_max_size` builder setter (+ create the builder unit-test harness)

- **Specs:** SPEC-CATF-01
- **Blocked by:** none
- **Blocks:** T02
- **Parallelizable:** NO with lane A (owns `builder.rs`). YES with T07, T16.

**Goal / completion condition:** `CatBoostBuilder::new().one_hot_max_size(7)` yields
`boost_params().one_hot_max_size == 7`, and an unset builder still yields
`one_hot_max_size_default()`.

**Files**
- Create: `crates/catboost-rs/src/builder_test.rs`
- Modify: `crates/catboost-rs/src/builder.rs`

**Verified symbols**
- `CatBoostBuilder` struct fields end at `score_function: EScoreFunction,`
  `[VERIFIED: builder.rs:84-85]`; `new()` initializer block `[VERIFIED: builder.rs:98-117]`.
- `boost_params()` currently assigns `one_hot_max_size: one_hot_max_size_default(),`
  with a three-line "pinned … never exercises the one-hot branch" comment
  `[VERIFIED: builder.rs:284-287]`.
- `one_hot_max_size_default() -> u32` is a `pub const fn`
  `[VERIFIED: cb-train/src/candidates.rs:78-80]`; `BoostParams.one_hot_max_size: u32`
  `[VERIFIED: cb-train/src/boosting.rs:239]`. It is already imported into `builder.rs`
  `[VERIFIED: builder.rs:34-41 use-list]`.
- Child-module mount precedent: `crates/catboost-rs/src/cv.rs:551-553`.

**Red**
- File: `crates/catboost-rs/src/builder_test.rs` (new)
- Test fn: `one_hot_max_size_setter_reaches_boost_params`
- Header of the new file:
  ```rust
  //! Unit tests for `CatBoostBuilder`'s CTR/categorical configuration surface.
  //! Mounted as a CHILD module of `builder` so the private `boost_params()` is
  //! visible (source/test separation, CLAUDE.md).
  use super::CatBoostBuilder;
  ```
- Setup/input: `let p = CatBoostBuilder::new().one_hot_max_size(7).boost_params();`
- Expected: `assert_eq!(p.one_hot_max_size, 7);`
- **Expected initial failure mode:** compile error
  `error[E0599]: no method named 'one_hot_max_size' found for struct 'CatBoostBuilder'`.
  (If the `#[path]` mount is added in the same step, the *first* error may instead be
  `error[E0583]: file not found for module` — add the file first, then the mount.)
- Run: `cargo test -p catboost-rs --lib builder::builder_test`

**Green (minimum change)**
1. Add field `one_hot_max_size: u32,` to `CatBoostBuilder` (after `score_function`).
2. Initialize it in `new()` as `one_hot_max_size: one_hot_max_size_default(),`.
3. Add the setter immediately after the existing `score_function` setter:
   ```rust
   /// The one-hot vs CTR encoding threshold for a categorical column
   /// (`cat_feature_options.cpp:231-232`). A column with learn-set cardinality
   /// `1 < c <= one_hot_max_size` is one-hot encoded; above it, CTR.
   #[must_use]
   pub fn one_hot_max_size(mut self, one_hot_max_size: u32) -> Self {
       self.one_hot_max_size = one_hot_max_size;
       self
   }
   ```
4. In `boost_params()`, replace `one_hot_max_size: one_hot_max_size_default(),`
   with `one_hot_max_size: self.one_hot_max_size,` and REPLACE the "pinned…"
   comment with a `/// surfaced via [`CatBoostBuilder::one_hot_max_size`]` note.
5. Append to the bottom of `builder.rs`:
   ```rust
   #[cfg(test)]
   #[path = "builder_test.rs"]
   mod builder_test;
   ```
- Preserve: the `use cb_train::{… one_hot_max_size_default …}` import (still used by
  `new()`); every other `boost_params()` field verbatim.
- Do NOT implement: any other setter, any `fit()` change.
- Run: `cargo test -p catboost-rs --lib builder::builder_test`

**Refactor**
- Constraint: no behavior change; only doc-comment wording. Do not reorder any
  `boost_params()` field.
- Regression scope: `cargo test -p catboost-rs`

**Validation**
```bash
cargo test -p catboost-rs --lib builder::builder_test
cargo test -p catboost-rs
cargo clippy -p catboost-rs --all-targets
```

**Completion evidence:** the named test passes; `cargo test -p catboost-rs` is green
including the three float-only oracle tests, unmodified.

---

### T02 — `max_ctr_complexity` builder setter

- **Specs:** SPEC-CATF-02
- **Blocked by:** T01
- **Blocks:** T03
- **Parallelizable:** NO (owns `builder.rs`).

**Goal:** `.max_ctr_complexity(k)` → `boost_params().max_ctr_complexity == k`;
unset → `max_ctr_complexity_default()`.

**Files:** Modify `crates/catboost-rs/src/builder.rs`, `crates/catboost-rs/src/builder_test.rs`.

**Verified symbols:** `boost_params()` line `max_ctr_complexity: max_ctr_complexity_default(),`
under the "Tensor-CTR config pinned … inert here" comment
`[VERIFIED: builder.rs:298-301]`; `max_ctr_complexity_default() -> usize`
`[VERIFIED: cb-train/src/boosting.rs:499-501, delegating to
cb-train/src/projection.rs:58-60 `pub const fn max_ctr_complexity_default() -> usize { 4 }`]`;
`BoostParams.max_ctr_complexity: usize` `[VERIFIED: cb-train/src/boosting.rs:290]`.

**Red**
- File: `crates/catboost-rs/src/builder_test.rs`
- Test fn: `max_ctr_complexity_setter_reaches_boost_params`
- Input: `CatBoostBuilder::new().max_ctr_complexity(3).boost_params()`
- Expected: `assert_eq!(p.max_ctr_complexity, 3);`
- **Expected initial failure:** `error[E0599]: no method named 'max_ctr_complexity'
  found for struct 'CatBoostBuilder'`.
- Run: `cargo test -p catboost-rs --lib builder::builder_test::max_ctr_complexity`

**Green:** field `max_ctr_complexity: usize`, `new()` default
`max_ctr_complexity_default()`, `#[must_use]` setter, `boost_params()` reads
`self.max_ctr_complexity`. Leave the sibling `combinations_ctr*` lines still pinned
(T04's job) and trim the shared comment so it no longer claims `max_ctr_complexity`
is inert.

**Refactor:** none beyond comment accuracy. Regression: `cargo test -p catboost-rs`.

**Validation:** as T01, with the `max_ctr_complexity` filter.

**Completion evidence:** both T01 and T02 unit tests pass; full crate test green.

---

### T03 — `simple_ctr` + `simple_ctr_priors` builder setters (lockstep)

- **Specs:** SPEC-CATF-03
- **Blocked by:** T02
- **Blocks:** T04
- **Parallelizable:** NO (owns `builder.rs`).

**Goal:** `.simple_ctr(t)` and `.simple_ctr_priors(v)` each reach `BoostParams`;
unset → `simple_ctr_default()` / `simple_ctr_priors_default()`.

**Files:** Modify `builder.rs`, `builder_test.rs`.

**Verified symbols:** `boost_params()` lines
`simple_ctr: simple_ctr_default(),` and `simple_ctr_priors: simple_ctr_priors_default(),`
under the "CTR config pinned … inert here" comment `[VERIFIED: builder.rs:292-295]`;
`simple_ctr_default() -> ECtrType` returns `ECtrType::Borders`
`[VERIFIED: cb-train/src/boosting.rs:464-466]`; `simple_ctr_priors_default() -> Vec<f64>`
returns `vec![0.5]` `[VERIFIED: cb-train/src/boosting.rs:473-475]`;
`BoostParams.simple_ctr: ECtrType` / `.simple_ctr_priors: Vec<f64>`
`[VERIFIED: cb-train/src/boosting.rs:261, :267]`;
`ECtrType` enum with 6 variants, `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`
`[VERIFIED: cb-train/src/ctr/mod.rs:65-80]`.

`builder.rs` must add `ECtrType` to its existing `use cb_train::{...}` list
`[VERIFIED: builder.rs:34-41 currently imports the `*_default` fns but NOT `ECtrType`]`.

**Red**
- File: `builder_test.rs`
- Test fn: `simple_ctr_and_priors_setters_reach_boost_params`
- Input:
  ```rust
  let p = CatBoostBuilder::new()
      .simple_ctr(cb_train::ECtrType::Counter)
      .simple_ctr_priors(vec![0.0, 1.0])
      .boost_params();
  ```
  (the test crate may name `cb_train` directly — it is a regular dependency,
  `[VERIFIED: crates/catboost-rs/Cargo.toml:34]`)
- Expected: `assert_eq!(p.simple_ctr, cb_train::ECtrType::Counter);`
  `assert_eq!(p.simple_ctr_priors, vec![0.0, 1.0]);`
- **Expected initial failure:** `error[E0599]: no method named 'simple_ctr' found for
  struct 'CatBoostBuilder'`.
- Run: `cargo test -p catboost-rs --lib builder::builder_test::simple_ctr`

**Green:** add both fields + both `new()` defaults + both `#[must_use]` setters;
`boost_params()` reads `self.simple_ctr` and `self.simple_ctr_priors.clone()`
(the field is `Vec<f64>` and `boost_params(&self)` takes `&self`, so a `.clone()` is
required — mirrors the existing `loss: self.loss.clone()` precedent at
`builder.rs:261-264`). Doc-comment the priors setter with the lockstep rationale
(SPEC §5 CATF-03 / R4): a non-default `ECtrType` has different default priors
(`ECtrType::default_priors()` at `cb-train/src/ctr/mod.rs:110-121`), so the caller
must set both.

**Refactor:** none. Regression: `cargo test -p catboost-rs`.

**Validation:** as T01, with the `simple_ctr` filter.

---

### T04 — `combinations_ctr` + `combinations_ctr_priors` builder setters (lockstep)

- **Specs:** SPEC-CATF-04
- **Blocked by:** T03
- **Blocks:** T05
- **Parallelizable:** NO (owns `builder.rs`).

Identical shape to T03 over the combination fields.

**Verified symbols:** `boost_params()` lines `combinations_ctr: combinations_ctr_default(),`
and `combinations_ctr_priors: combinations_ctr_priors_default(),`
`[VERIFIED: builder.rs:302-303]`; `combinations_ctr_default() -> ECtrType`
= `Borders` `[VERIFIED: cb-train/src/boosting.rs:509-511]`;
`combinations_ctr_priors_default() -> Vec<f64>` = `vec![0.5]`
`[VERIFIED: cb-train/src/boosting.rs:518-520]`; `BoostParams.combinations_ctr` /
`.combinations_ctr_priors` `[VERIFIED: cb-train/src/boosting.rs:298, :304]`.

**Red**
- Test fn: `combinations_ctr_and_priors_setters_reach_boost_params`
- Input: `.combinations_ctr(ECtrType::Buckets).combinations_ctr_priors(vec![0.25])`
- Expected: both `BoostParams` fields reflect the values.
- **Expected initial failure:** `error[E0599]: no method named 'combinations_ctr'`.
- Run: `cargo test -p catboost-rs --lib builder::builder_test::combinations_ctr`

**Green:** two fields, two defaults, two setters, two `boost_params()` reads
(priors `.clone()`). Delete the now-false "the numeric facade path forms no
combination, so these are inert here" comment.

**Refactor / Validation / Evidence:** as T03.

---

### T05 — `counter_calc_method` builder setter

- **Specs:** SPEC-CATF-05
- **Blocked by:** T04
- **Blocks:** T06, T12
- **Parallelizable:** NO (owns `builder.rs`).

**Verified symbols:** `boost_params()` line `counter_calc_method: counter_calc_method_default(),`
`[VERIFIED: builder.rs:296]`; `counter_calc_method_default() -> CounterCalcMethod`
= `SkipTest` `[VERIFIED: cb-train/src/boosting.rs:480-482]`;
`CounterCalcMethod { SkipTest, Full }`, `#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]`
`[VERIFIED: cb-train/src/ctr/mod.rs:129-136]`; `BoostParams.counter_calc_method`
`[VERIFIED: cb-train/src/boosting.rs:272]`.

**Red**
- Test fn: `counter_calc_method_setter_reaches_boost_params`
- Input: `CatBoostBuilder::new().counter_calc_method(CounterCalcMethod::Full).boost_params()`
- Expected: `assert_eq!(p.counter_calc_method, CounterCalcMethod::Full);`
- **Expected initial failure:** `error[E0599]: no method named 'counter_calc_method'`.
- Run: `cargo test -p catboost-rs --lib builder::builder_test::counter_calc_method`

**Green:** field + `new()` default + setter + `boost_params()` read. Add
`CounterCalcMethod` to the `use cb_train::{...}` list.

**Refactor:** after this task ALL seven CTR fields are field-backed — sweep
`boost_params()` and delete every remaining "pinned … inert here" CTR comment that
is now false. Do NOT touch the `permutation_count` / `fold_len_multiplier` /
`has_time` / `boosting_type` comments (still genuinely pinned).

**Validation:** `cargo test -p catboost-rs`; `cargo clippy -p catboost-rs --all-targets`.

---

### T06 — `boost_params()` default-equivalence guard

- **Specs:** SPEC-CATF-06
- **Blocked by:** T05
- **Blocks:** T08
- **Parallelizable:** YES with T07 (different files) — but NO with lane A.

**Goal:** a single test proves the inline-default → builder-field refactor changed
no default. This is the one guard against R1.

**Files:** Modify `crates/catboost-rs/src/builder_test.rs` only.

**Red — MUTATION-CHECK protocol (see §3)**
- Test fn: `unset_builder_ctr_defaults_equal_pinned_upstream_defaults`
- Body asserts all SEVEN fields at once against the `*_default()` functions:
  ```rust
  let p = CatBoostBuilder::new().boost_params();
  assert_eq!(p.one_hot_max_size,        cb_train::one_hot_max_size_default());
  assert_eq!(p.max_ctr_complexity,      cb_train::max_ctr_complexity_default());
  assert_eq!(p.simple_ctr,              cb_train::simple_ctr_default());
  assert_eq!(p.simple_ctr_priors,       cb_train::simple_ctr_priors_default());
  assert_eq!(p.combinations_ctr,        cb_train::combinations_ctr_default());
  assert_eq!(p.combinations_ctr_priors, cb_train::combinations_ctr_priors_default());
  assert_eq!(p.counter_calc_method,     cb_train::counter_calc_method_default());
  ```
- Expected first run: **PASS** (T01–T05 were done correctly).
- **Mandatory mutation check:** in `builder.rs::new()`, temporarily change
  `simple_ctr_priors: simple_ctr_priors_default(),` to
  `simple_ctr_priors: vec![0.0],`. Re-run.
  **Expected failure:** ``assertion `left == right` failed`` with
  `left: [0.0]`, `right: [0.5]`.
  Then REVERT and confirm green. Record both outputs as completion evidence.
- Run: `cargo test -p catboost-rs --lib builder::builder_test::unset_builder_ctr_defaults`

**Green:** none — the assertion set IS the deliverable. If the first run fails, fix
the offending `new()` default (that is the whole point of the guard).

**Refactor:** none.

**Validation**
```bash
cargo test -p catboost-rs --lib builder::builder_test
cargo test -p catboost-rs
```

**Completion evidence:** the recorded pass → mutated-fail → reverted-pass triple.

---

### T07 — Re-export `ECtrType` and `CounterCalcMethod` from `catboost_rs`

- **Specs:** SPEC-CATF-09
- **Blocked by:** none
- **Blocks:** T12
- **Parallelizable:** YES — touches only `lib.rs` + a new `tests/` file; no overlap
  with lane A's `builder.rs` or Wave 2's `model.rs`.

**Goal:** an external crate depending only on `catboost_rs` can name
`catboost_rs::ECtrType` and `catboost_rs::CounterCalcMethod`.

**Files**
- Modify: `crates/catboost-rs/src/lib.rs`
- Create: `crates/catboost-rs/tests/reexport_ctr_types_test.rs`

**Verified symbols:** `lib.rs:61` currently has exactly `pub use cb_train::EBootstrapType;`
— the stated precedent — and there is **no** `ECtrType`/`CounterCalcMethod` re-export
`[VERIFIED: crates/catboost-rs/src/lib.rs:56-66 read in full]`. Both types are
`pub use`d from `cb_train` `[VERIFIED: cb-train/src/ctr/mod.rs:65, :130 are `pub enum`;
research.md records the `cb-train/src/lib.rs:46,49` re-exports]`.

An integration test under `tests/` compiles as a **separate crate** that sees only
`catboost_rs`'s public API — the exact "downstream crate" proxy SPEC-CATF-09 asks for.

**Red**
- File: `crates/catboost-rs/tests/reexport_ctr_types_test.rs`
- Test fn: `ctr_setter_parameter_types_are_nameable_downstream`
  ```rust
  //! SPEC-CATF-09: the CTR setter parameter types must be nameable through the
  //! PUBLISHED crate alone (this integration test is a separate crate and may
  //! import nothing but `catboost_rs`).
  use catboost_rs::{CounterCalcMethod, ECtrType};

  #[test]
  fn ctr_setter_parameter_types_are_nameable_downstream() {
      assert_eq!(ECtrType::Borders.as_i8(), 0);
      assert_eq!(CounterCalcMethod::default(), CounterCalcMethod::SkipTest);
  }
  ```
  Do NOT add `cb-train` to this file's imports — that would defeat the test.
- **Expected initial failure:** `error[E0432]: unresolved imports
  'catboost_rs::CounterCalcMethod', 'catboost_rs::ECtrType'`.
- Run: `cargo test -p catboost-rs --test reexport_ctr_types_test`

**Green:** add one line next to `lib.rs:61`, with a comment mirroring the existing
re-export blocks:
```rust
// Re-export the CTR configuration enums the Builder's categorical setters
// consume, so a caller (and the PyO3 binding crate) configures a categorical run
// entirely through the published crate. `EBootstrapType` above is the precedent.
pub use cb_train::{CounterCalcMethod, ECtrType};
```

**Refactor:** none. Do not re-export `TProjection`/`BakedCtrData` — not required by
any spec and would widen the public surface without a contract.

**Validation**
```bash
cargo test -p catboost-rs --test reexport_ctr_types_test
cargo doc -p catboost-rs --no-deps          # confirms no rustdoc link breakage
cargo clippy -p catboost-rs --all-targets
```

---

### T08 — `fit()` routes a categorical pool to `train_cat` and bakes CTR data

- **Specs:** SPEC-CATF-08
- **Blocked by:** T06
- **Blocks:** T09, T14
- **Parallelizable:** NO (owns `builder.rs`).

**Goal:** `CatBoostBuilder::fit(&pool)` on a pool with non-empty `cat_features()`
returns a model whose canonical form has `ctr_data.is_some()` **and** at least one
`ModelSplit::Ctr` in some tree.

**Files**
- Modify: `crates/catboost-rs/src/builder.rs`
- Create: `crates/catboost-rs/tests/builder_cat_facade_test.rs`

**Verified symbols and exact call shape**
- `fit` today: `pub fn fit(&self, pool: &Pool) -> Result<Model, CatBoostError>` at
  `builder.rs:334`, ending with `train(&backend, &feature_values, &feature_borders,
  pool.label(), pool.weights(), &params, None)?` at `:383-391` and
  `cb_model::Model::from_trained(&trained, feature_borders)` at `:400`
  `[VERIFIED: builder.rs:334-402 read in full]`.
- `train_cat` signature `[VERIFIED: cb-train/src/boosting.rs:2149-2158]`:
  ```rust
  pub fn train_cat<R: Runtime>(
      runtime: &R, feature_values: &[Vec<f32>], feature_borders: &[Vec<f64>],
      cat_columns: &[Vec<String>], target: &[f64], weights: &[f64],
      params: &BoostParams, staged_out: Option<&mut Vec<f64>>,
  ) -> CbResult<(Model, BakedCtrData)>
  ```
- Proven attach pattern `[VERIFIED: cb-train/tests/tensor_ctr_e2e_oracle_test.rs:232-251]`:
  `CbModel::from_trained(&trained, borders.clone()).with_ctr_data(cb_model::CtrData::from_baked(&baked))`.
- `Pool::cat_features() -> &[Vec<String>]` `[VERIFIED: cb-data/src/pool.rs:152-156]`;
  `Pool::n_cat_features()` `[VERIFIED: cb-data/src/pool.rs:123-126]`.
- `CtrData::from_baked(&cb_train::BakedCtrData) -> Self`
  `[VERIFIED: cb-model/src/ctr_data.rs:305-313]`.
- Error coverage needs no new variant: `train_cat` returns the same `CbResult`
  shape, covered by `CatBoostError::Train(#[from] cb_core::CbError)`
  `[VERIFIED: crates/catboost-rs/src/error.rs:36-37]`.
- `ModelSplit::{Float, Ctr}` `[VERIFIED: cb-model/src/model.rs:70-76]`; usable from a
  `tests/` file because `cb-model` is a regular dependency
  `[VERIFIED: crates/catboost-rs/Cargo.toml:34]`.
- `OwnedColumns` and `Pool` are re-exported from the facade
  `[VERIFIED: crates/catboost-rs/src/lib.rs:65-66]`; `OwnedColumns::with_cat_features`
  is the ingestion-side setter `[VERIFIED: referenced at ingest_py.rs:302]`.

**Red**
- File: `crates/catboost-rs/tests/builder_cat_facade_test.rs`
- Test fn: `fit_categorical_pool_bakes_ctr_data_and_emits_a_ctr_split`
- Setup: load the frozen `tensor_ctr_e2e` corpus (already committed, no new fixture
  needed for this task):
  - `crates/cb-oracle/fixtures/tensor_ctr_e2e/X_cat.npy` — `[30, 2] int32`
  - `crates/cb-oracle/fixtures/tensor_ctr_e2e/y.npy` — `[30] float64`
  `[VERIFIED: ls of that directory + its config.json `npy_schema`]`
  Stringify each integer code with the plain-integer form (`format!("{v}")`, the A4
  convention the fixture's own note describes), build
  `OwnedColumns::new(Vec::new(), y).with_cat_features(cat_cols).into_pool()?`.
  Reuse the `fixture()` path helper shape from
  `crates/catboost-rs/tests/builder_oracle_test.rs:46-53`.
- Builder config: `Loss::Logloss`, `iterations(5)`, `depth(2)`, `learning_rate(0.1)`,
  `l2_leaf_reg(3.0)`, `boost_from_average(false)`, `leaf_method(LeafMethod::Gradient)`,
  `bootstrap_type(EBootstrapType::No)`, `random_strength(0.0)`, `random_seed(0)`,
  `one_hot_max_size(1)`, `max_ctr_complexity(1)`.
  (`max_ctr_complexity(1)` deliberately keeps this task clear of the ORD-06/07
  combination-gating bug — this test is a routing assertion, not an oracle.)
- Assertions (**both are the mandated anti-false-pass guard, R2**):
  ```rust
  let inner = model.as_canonical();
  assert!(inner.ctr_data.is_some(), "categorical fit must bake CTR data");
  let has_ctr_split = inner.oblivious_trees.iter().any(|t| {
      t.splits.iter().any(|s| matches!(s, cb_model::ModelSplit::Ctr(_)))
  });
  assert!(has_ctr_split, "at least one tree must carry a ModelSplit::Ctr");
  ```
- **Expected initial failure:** ``assertion failed: inner.ctr_data.is_some()`` —
  because `fit()` calls the float-only `train`, which never sees the cat columns and
  never returns `BakedCtrData`.
- Run: `cargo test -p catboost-rs --test builder_cat_facade_test`

**Green (minimum change)**
In `fit()`, keep everything up to and including `let params = self.boost_params();`
and the `backend` binding **byte-identical**. Then branch:

```rust
let canonical = if pool.cat_features().is_empty() {
    // Float-only path — BYTE-IDENTICAL to the pre-change code (D-04 / SPEC-CATF-07).
    let trained = train(
        &backend, &feature_values, &feature_borders,
        pool.label(), pool.weights(), &params, None,
    )?;
    cb_model::Model::from_trained(&trained, feature_borders)
} else {
    // Categorical path (SPEC-CATF-08): the cat-aware entrypoint returns the model
    // PLUS the baked whole-set inference tables, attached for the CTR apply path.
    let (trained, baked) = train_cat(
        &backend, &feature_values, &feature_borders, pool.cat_features(),
        pool.label(), pool.weights(), &params, None,
    )?;
    cb_model::Model::from_trained(&trained, feature_borders)
        .with_ctr_data(cb_model::CtrData::from_baked(&baked))
};
Ok(Model::from_canonical(canonical))
```
Add `train_cat` to the `use cb_train::{...}` list. Keep the `CB_GPU_PROF`
`prof_train_t` timing block covering **both** arms.

- Do NOT implement: any predict-side change (T09–T11), any new error variant.

**Refactor**
- Constraint: the `rayon::join` border/value computation must stay above the branch
  and unchanged; the `feature_borders` binding is moved into the branch, so use
  `feature_borders` by value in each arm (both arms consume it exactly once — no
  clone needed).
- Regression scope: `cargo test -p catboost-rs` — the three float-only oracle tests
  are the D-04 gate here (they must pass **unmodified**).

**Validation**
```bash
cargo test -p catboost-rs --test builder_cat_facade_test
cargo test -p catboost-rs --test builder_oracle_test --test cv_oracle_test --test grid_search_oracle_test
cargo test -p catboost-rs
cargo clippy -p catboost-rs --all-targets
```

**Completion evidence:** both assertions pass; the three pre-existing oracle tests
pass with zero diff in their files (`git diff --stat crates/catboost-rs/tests/` shows
only the NEW file).

---

### T09 — `Model::cat_columns()` — categorical width validation on the predict path

- **Specs:** SPEC-CATF-10
- **Blocked by:** T08
- **Blocks:** T10
- **Parallelizable:** NO (owns `model.rs`).

**Goal:** `Ok(cat columns)` when the pool's categorical width matches what the model
expects; `Err(CatBoostError::FeatureMismatch)` naming both counts otherwise.

**Files**
- Modify: `crates/catboost-rs/src/model.rs`
- Create: `crates/catboost-rs/tests/model_cat_predict_test.rs`

**Verified symbols**
- `Model { inner: cb_model::Model }`, `pub fn as_canonical(&self) -> &cb_model::Model`
  `[VERIFIED: crates/catboost-rs/src/model.rs:63-81]`.
- The float precedent to mirror exactly (message shape + strict equality):
  `fn feature_columns(&self, pool: &Pool) -> Result<Vec<Vec<f32>>, CatBoostError>`
  returning `FeatureMismatch(format!("pool has {actual} float features, model expects {expected}"))`
  `[VERIFIED: crates/catboost-rs/src/model.rs:94-107]`.
- `CatBoostError::FeatureMismatch(String)` `[VERIFIED: crates/catboost-rs/src/error.rs:69-70]`.
- `ctr_data: Option<CtrData>` on the canonical model
  `[VERIFIED: cb-model/src/model.rs:294-298]`; the canonical model has **no** cat-count
  field (finding F3), so the expected width is DERIVED.
- `CtrSplit.projection: cb_train::TProjection` and the apply path's member gather
  `split.projection.cat_features()` `[VERIFIED: cb-model/src/model.rs:43-45;
  cb-model/src/apply.rs:157-168]`.
- `Pool::n_cat_features()` / `cat_features()` `[VERIFIED: cb-data/src/pool.rs:123-126, :152-156]`.

**Red**
- File: `crates/catboost-rs/tests/model_cat_predict_test.rs`
- Test fn: `predict_on_ctr_model_with_wrong_cat_width_is_feature_mismatch`
- Setup: reuse the T08 helper shape — fit a CTR model from the frozen
  `tensor_ctr_e2e` corpus (2 cat features). Then build a **second** pool from only
  the FIRST cat column (1 cat feature, same labels).
- Action: `model.predict(&narrow_pool)`
- Expected:
  ```rust
  match model.predict(&narrow_pool) {
      Err(CatBoostError::FeatureMismatch(msg)) => {
          assert!(msg.contains('1') && msg.contains('2'), "must name both counts: {msg}");
          assert!(msg.contains("categorical"), "must say categorical: {msg}");
      }
      other => panic!("expected FeatureMismatch, got {other:?}"),
  }
  ```
- **Expected initial failure:** the `other =>` arm fires with
  `expected FeatureMismatch, got Ok([...])` — today `predict_with` calls the float-only
  `predict_raw` and the float width (0 == 0) check passes, so a wrong-width cat pool is
  silently scored.
- Run: `cargo test -p catboost-rs --test model_cat_predict_test::predict_on_ctr_model_with_wrong_cat_width`

**Green (minimum change)** — add two private helpers to `impl Model` in `model.rs`,
directly beneath `feature_columns`:

```rust
/// True when the canonical model carries baked CTR tables (SPEC-CATF-11/12).
fn is_ctr_model(&self) -> bool {
    self.inner.ctr_data.is_some()
}

/// The number of categorical columns a CTR model needs, DERIVED from its CTR
/// splits' projections (the canonical model stores no cat-feature count). It is
/// `max(member index) + 1` over every `ModelSplit::Ctr` — the same indexing
/// `cb_model::apply::passes_ctr_split` uses to gather member values — floored at
/// `1` for a CTR model whose splits reference no member, so a cat-free pool can
/// never be silently accepted (SPEC-CATF-12).
fn expected_cat_features(&self) -> usize { /* checked iteration; no indexing */ }

/// `Ok(cat columns)` when the pool's categorical width matches; otherwise
/// `Err(CatBoostError::FeatureMismatch)` naming both counts. A non-CTR model
/// expects zero categorical columns and returns an empty vector, so the float-only
/// predict path is byte-identical (D-04).
fn cat_columns(&self, pool: &Pool) -> Result<Vec<Vec<String>>, CatBoostError> { ... }
```

`cat_columns` body contract:
- `if !self.is_ctr_model() { return Ok(Vec::new()); }` — preserves D-04.
- `let expected = self.expected_cat_features(); let actual = pool.n_cat_features();`
- `if actual != expected { return Err(CatBoostError::FeatureMismatch(format!(
     "pool has {actual} categorical features, model expects {expected}"))); }`
- `Ok(pool.cat_features().to_vec())`

At this task the helper is called from `predict_with` **only** as a validation step
(the actual `predict_raw_cat` routing is T10's Green):
insert `let cat_columns = self.cat_columns(pool)?;` after
`let columns = self.feature_columns(pool)?;` and bind it with `let _ = &cat_columns;`
plus a `// consumed by SPEC-CATF-11 in the next task` marker. This keeps T09's change
minimal AND makes its Red go green.

- Use checked iteration only (`.iter()`, `.max()`, `.copied()`) — the workspace denies
  `indexing_slicing`.
- Do NOT implement: the `predict_raw_cat` call (T10).

**Refactor**
- Constraint: `feature_columns` must not change. Deduplicate the two mismatch message
  builders only if it does not change either string.
- Regression scope: `cargo test -p catboost-rs`.

**Validation**
```bash
cargo test -p catboost-rs --test model_cat_predict_test
cargo test -p catboost-rs --test builder_oracle_test --test cv_oracle_test --test grid_search_oracle_test
cargo clippy -p catboost-rs --all-targets
```

---

### T10 — `predict_with` is CTR-aware

- **Specs:** SPEC-CATF-11
- **Blocked by:** T09
- **Blocks:** T11, T14
- **Parallelizable:** NO (owns `model.rs`).

**Goal:** a CTR model + a width-matching pool routes through
`cb_model::predict_raw_cat` with the pool's cat columns; a non-CTR model keeps the
existing `predict_raw` call byte-identically.

**Files:** Modify `crates/catboost-rs/src/model.rs`, `crates/catboost-rs/tests/model_cat_predict_test.rs`.

**Verified symbols**
- Current body `[VERIFIED: crates/catboost-rs/src/model.rs:120-128]`:
  ```rust
  pub fn predict_with(&self, pool, prediction_type) -> Result<Vec<f64>, CatBoostError> {
      let columns = self.feature_columns(pool)?;
      let raw = predict_raw(&self.inner, &columns);
      Ok(apply_prediction_type(prediction_type, &raw))
  }
  ```
- `predict_raw(model, fv) == predict_raw_cat(model, fv, &[])`
  `[VERIFIED: cb-model/src/apply.rs:369-372]` — so routing everything through
  `predict_raw_cat` with the T09 `cat_columns()` result (empty for a non-CTR model)
  is provably byte-identical on the float path.
- `pub fn predict_raw_cat(model, feature_values, cat_columns) -> Vec<f64>`
  `[VERIFIED: cb-model/src/apply.rs:386-390]`.
- `predict` and `predict_proba` both funnel through `predict_with`
  `[VERIFIED: crates/catboost-rs/src/model.rs:135-137 for `predict`]`.
- `predict_raw` is currently imported at `model.rs:20`; `predict_raw_cat` is NOT —
  it must be added to the `use cb_model::{...}` list.

**Red**
- Test fn: `ctr_model_predict_matches_the_direct_cb_model_ctr_apply`
- Setup: fit the CTR model exactly as in T08 (frozen `tensor_ctr_e2e` corpus,
  `max_ctr_complexity(1)`), keeping the same pool.
- Reference: compute `cb_model::predict_raw_cat(model.as_canonical(), &[], &cat_cols)`
  directly (the already-oracle-verified apply path).
- Action: `model.predict(&pool)?`
- Expected: element-wise equality to the reference within `1e-12`
  (Rust↔Rust identity, not an oracle — the ≤1e-5 upstream gate is T17), AND a
  non-degeneracy guard: `assert!(reference.iter().any(|v| *v != reference[0]))`
  so a constant-zero vector cannot make the test vacuous.
- **Expected initial failure:** the values differ — the facade's `predict` evaluates
  every `ModelSplit::Ctr` against an EMPTY-STRING category (SPEC §1's silent-wrongness
  hazard), so it returns a different, wrong vector. Failure message: the
  `assert!((a - b).abs() < 1e-12` line with a concrete numeric gap.
- Run: `cargo test -p catboost-rs --test model_cat_predict_test::ctr_model_predict_matches`

**Green (minimum change)**
```rust
pub fn predict_with(&self, pool, prediction_type) -> Result<Vec<f64>, CatBoostError> {
    let columns = self.feature_columns(pool)?;
    // SPEC-CATF-11: a CTR model needs the pool's categorical columns. For a
    // non-CTR model `cat_columns` is empty and `predict_raw_cat(m, fv, &[])` IS
    // `predict_raw(m, fv)` (cb-model/src/apply.rs:369-372), so the float-only
    // output is byte-identical (D-04 / SPEC-CATF-07).
    let cat_columns = self.cat_columns(pool)?;
    let raw = predict_raw_cat(&self.inner, &columns, &cat_columns);
    Ok(apply_prediction_type(prediction_type, &raw))
}
```
Add `predict_raw_cat` to the `use cb_model::{...}` list. Keep `predict_raw` imported
only if still used elsewhere in the file; if it becomes unused, remove it from the
import (an unused import is a warning, and clippy runs with `--all-targets`).

- Do NOT change: `staged_predict`, `shap_values`, `save_onnx`, `save_coreml`,
  `feature_importance*`. `ensure_scalar_oblivious` at `model.rs:144-171` already
  rejects CTR models for `staged_predict` with `UnsupportedModel`
  `[VERIFIED: model.rs:164-169]` — that rejection is the CORRECT behavior per SPEC §2
  and must be preserved, not extended.
- Update `predict_with`'s `# Errors` doc block to mention the categorical-width
  mismatch alongside the float one.

**Refactor**
- Constraint: no reordering of the two validation calls (float check first, matching
  the existing error precedence).
- Regression scope: `cargo test -p catboost-rs` plus, explicitly, the staged-predict
  and ONNX/CoreML facade tests (`staged_predict_facade_test`, `onnx_facade_test`,
  `coreml_facade_test`) to prove the CTR rejections still fire.

**Validation**
```bash
cargo test -p catboost-rs --test model_cat_predict_test
cargo test -p catboost-rs --test staged_predict_facade_test --test onnx_facade_test --test coreml_facade_test
cargo test -p catboost-rs --test builder_oracle_test --test cv_oracle_test --test grid_search_oracle_test
cargo test -p catboost-rs
cargo clippy -p catboost-rs --all-targets
```

---

### T11 — A CTR model may never be silently mis-scored (zero-cat-column guard)

- **Specs:** SPEC-CATF-12
- **Blocked by:** T10
- **Blocks:** T17
- **Parallelizable:** NO with Wave 2 (test file shared with T09/T10; production change
  expected to be zero).

**Goal:** `predict` / `predict_with` / `predict_proba` on a CTR model with a pool
carrying **zero** categorical columns returns a typed `CatBoostError::FeatureMismatch`
— never a number. This is the spec's dedicated anti-silent-wrongness gate and its
assertion must be on the ERROR, not on "some result".

**Files:** Modify `crates/catboost-rs/tests/model_cat_predict_test.rs`. Production
change expected to be **none** (T09's `cat_columns()` already floors the expected
width at 1 for a CTR model). If the test fails, the fix belongs in
`expected_cat_features()`.

**Red**
- Test fn: `ctr_model_predict_on_cat_free_pool_is_typed_error_on_all_three_entrypoints`
- Setup: fit the CTR model (as T08/T10). Build a cat-free pool with the SAME labels
  and zero cat columns and zero float columns:
  `OwnedColumns::new(Vec::new(), y.clone()).into_pool()?`.
- Action + expectation — assert on all THREE public entrypoints, each individually:
  ```rust
  for (name, result) in [
      ("predict",      model.predict(&cat_free)),
      ("predict_with", model.predict_with(&cat_free, PredictionType::RawFormulaVal)),
      ("predict_proba",model.predict_proba(&cat_free)),
  ] {
      match result {
          Err(CatBoostError::FeatureMismatch(msg)) =>
              assert!(msg.contains("categorical"), "{name}: {msg}"),
          Err(other) => panic!("{name}: wrong error variant: {other:?}"),
          Ok(v) => panic!("{name}: SILENTLY SCORED a CTR model with no categorical \
                           columns — returned {} values", v.len()),
      }
  }
  ```
- **Expected initial state:** with T09+T10 landed this SHOULD already pass.
  Therefore apply the **mutation check** (§3): temporarily change
  `expected_cat_features()`'s floor from `.max(1)` to no floor, re-run, and confirm
  the `Ok(v) =>` arm fires with `predict: SILENTLY SCORED a CTR model ...`.
  Revert and confirm green. Record both outputs.
- Run: `cargo test -p catboost-rs --test model_cat_predict_test::ctr_model_predict_on_cat_free_pool`

**Green:** none expected. If the mutation-check reveals the floor is genuinely
missing, add it in `expected_cat_features()` only.

**Refactor:** none.

**Validation**
```bash
cargo test -p catboost-rs --test model_cat_predict_test
cargo test -p catboost-rs
```

**Completion evidence:** the three-entrypoint assertion passes; the recorded
mutated-fail output proves the guard is falsifiable.

---

### T12 — Python: `one_hot_max_size`, `max_ctr_complexity`, `counter_calc_method` kwargs

- **Specs:** SPEC-CATF-13 (scalar/enum half)
- **Blocked by:** T05 (setters), T07 (re-exports)
- **Blocks:** T13, T15
- **Parallelizable:** NO with T13 (both own `params.rs`). YES with T14.

**Goal:** `CatBoostRegressor(max_ctr_complexity=2).fit(...)` is accepted (no parity-gap
error) and the value reaches `BoostParams`; a bad enum string or out-of-range number
raises `CatBoostParameterError`.

**Files:** Modify `crates/catboost-rs-py/src/params.rs`; add tests to
`crates/catboost-rs-py/src/params_test.rs` (exists) and/or
`crates/catboost-rs-py/tests/test_params.py`.

**Verified symbols**
- `IMPLEMENTED` is a 14-entry `const &[&str]` `[VERIFIED: params.rs:42-57]`;
  `"one_hot_max_size"` (:107), `"max_ctr_complexity"` (:99),
  `"counter_calc_method"` (:86), `"simple_ctr"` (:128), `"combinations_ctr"` (:129),
  `"cat_features"` (:159) are all in `VOCABULARY` but NOT in `IMPLEMENTED`
  `[VERIFIED: params.rs:68-188]`, so `validate_params` rejects each as a parity gap
  `[VERIFIED: params.rs:290-318]`.
- Shape precedents: `parse_bootstrap_type` (:390-401), `parse_leaf_method` (:404-414),
  `check_range` (:425-435), the `get_with_aliases` + `check_range` + setter pattern
  (:456-511).
- `params.rs` imports from the facade at `:22`:
  `use catboost_rs::{CatBoostBuilder, EBootstrapType, EScoreFunction, LeafMethod, Loss};`
  — T07's re-exports let `CounterCalcMethod` (and T13's `ECtrType`) be added there.

**Red**
- File: `crates/catboost-rs-py/tests/test_params.py`
- Test fn: `test_ctr_scalar_kwargs_are_accepted_and_reach_the_builder`
  ```python
  def test_ctr_scalar_kwargs_are_accepted_and_reach_the_builder():
      x, y = _toy_xy()
      model = CatBoostRegressor(
          iterations=5, max_ctr_complexity=2,
          one_hot_max_size=3, counter_calc_method="Full",
      )
      model.fit(x, y)                       # must NOT raise
      assert catboost_rs._param_status("max_ctr_complexity") == "IMPLEMENTED"
      assert catboost_rs._param_status("one_hot_max_size") == "IMPLEMENTED"
      assert catboost_rs._param_status("counter_calc_method") == "IMPLEMENTED"

  def test_bad_counter_calc_method_is_parameter_error():
      x, y = _toy_xy()
      with pytest.raises(CatBoostParameterError) as e:
          CatBoostRegressor(counter_calc_method="Nope").fit(x, y)
      assert "Nope" in str(e.value)
      assert "SkipTest" in str(e.value) and "Full" in str(e.value)
  ```
- **Expected initial failure:**
  `CatBoostParameterError: parameter 'max_ctr_complexity' is a known CatBoost
  parameter not yet implemented in catboost-rs (parity gap)`.
- Run:
  ```bash
  VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
  .venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q -k ctr_scalar
  ```

**Green (minimum change)**
1. Add `"one_hot_max_size"`, `"max_ctr_complexity"`, `"counter_calc_method"` to
   `IMPLEMENTED`.
2. Add, next to `parse_leaf_method`:
   ```rust
   /// Map a `counter_calc_method` string onto a [`CounterCalcMethod`]
   /// (`cat_feature_options.cpp:234`).
   fn parse_counter_calc_method(name: &str) -> PyResult<CounterCalcMethod> {
       match name {
           "SkipTest" => Ok(CounterCalcMethod::SkipTest),
           "Full"     => Ok(CounterCalcMethod::Full),
           other => Err(CatBoostParameterError::new_err(format!(
               "unknown counter_calc_method `{other}` (supported: SkipTest, Full)"
           ))),
       }
   }
   ```
3. In `make_builder`, after the `leaf_estimation_method` block:
   ```rust
   if let Some(v) = get_with_aliases::<u32>(params, py, "one_hot_max_size")? {
       // one_hot_max_size in [0, 255] (upstream's cat-feature one-hot cap).
       check_range("one_hot_max_size", f64::from(v), 0.0, 255.0, true)?;
       builder = builder.one_hot_max_size(v);
   }
   if let Some(v) = get_with_aliases::<usize>(params, py, "max_ctr_complexity")? {
       // max_ctr_complexity in [1, 8] (upstream MaxTensorComplexity domain).
       check_range("max_ctr_complexity", v as f64, 1.0, 8.0, true)?;
       builder = builder.max_ctr_complexity(v);
   }
   if let Some(v) = get_with_aliases::<String>(params, py, "counter_calc_method")? {
       builder = builder.counter_calc_method(parse_counter_calc_method(&v)?);
   }
   ```
4. Add `CounterCalcMethod` to the `use catboost_rs::{...}` list at `params.rs:22`.

- Do NOT implement: `simple_ctr` / `combinations_ctr` (T13) or `cat_features` (T14).
  Leave all three out of `IMPLEMENTED` in this task so T13/T14's Reds stay genuine.

**Refactor**
- Constraint: reuse `get_with_aliases` / `check_range` verbatim — no hand-rolled
  validation (SPEC-CATF-13 constraint).
- Regression scope: full `pytest crates/catboost-rs-py/tests/` +
  `cargo test -p catboost-rs-py`.

**Validation**
```bash
cargo test -p catboost-rs-py
cargo clippy -p catboost-rs-py --all-targets
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
```

---

### T13 — Python: `simple_ctr` / `combinations_ctr` CTR-description grammar

- **Specs:** SPEC-CATF-13 (enum + priors half); resolves OQ-1
- **Blocked by:** T12
- **Blocks:** T15
- **Parallelizable:** NO (owns `params.rs`).

**Goal:** `simple_ctr` / `combinations_ctr` accept the upstream-shaped CTR-description
value and drive BOTH the type setter and the priors setter; an unknown type or an
unsupported grammar component raises `CatBoostParameterError` naming it and listing
the accepted CTR types.

**Accepted input forms (the OQ-1 resolution):**
- `"Borders"` — type only; the builder's `simple_ctr_priors` default is untouched.
- `"Borders:Prior=0.5"` / `"Borders:Prior=0:Prior=0.5:Prior=1"` — type plus one or
  more `Prior=<float>` components, which become the priors `Vec<f64>` in order.
- A `list[str]` of exactly one such string (upstream's declared type is
  `list of strings`, `[VERIFIED: catboost core.py:4957]`).

**Rejected (typed error, never silently dropped):** a list of length ≠ 1; any other
`Key=Value` component (`CtrBorderCount`, `TargetBorderCount`, `TargetBorderType`,
`CtrHistoryUnit`); a malformed float after `Prior=`; a `Prior=<num>/<den>` fractional
form (the Rust `simple_ctr_priors` field is a flat `Vec<f64>`
`[VERIFIED: cb-train/src/boosting.rs:267]`, so a non-unit denominator has no
representation — reject explicitly rather than silently drop the denominator).

**Files:** Modify `crates/catboost-rs-py/src/params.rs`; add tests to
`crates/catboost-rs-py/tests/test_params.py` and `crates/catboost-rs-py/src/params_test.rs`.

**Red**
- File: `crates/catboost-rs-py/tests/test_params.py`
- Test fn: `test_bad_simple_ctr_lists_accepted_ctr_types` (SPEC A7 verbatim)
  ```python
  def test_bad_simple_ctr_lists_accepted_ctr_types():
      x, y = _toy_xy()
      with pytest.raises(CatBoostParameterError) as e:
          CatBoostRegressor(simple_ctr="NotACtrType").fit(x, y)
      msg = str(e.value)
      assert "NotACtrType" in msg
      for variant in ("Borders", "Buckets", "Counter", "FeatureFreq"):
          assert variant in msg
  ```
- **Expected initial failure:** the raised error is the parity-gap message
  (`"simple_ctr" is a known CatBoost parameter not yet implemented ... (parity gap)`),
  so `assert "NotACtrType" in msg` fails.
- Run: `.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q -k simple_ctr`

**Green (minimum change)**
1. Add `"simple_ctr"`, `"combinations_ctr"` to `IMPLEMENTED`.
2. Add `ECtrType` to the `use catboost_rs::{...}` list.
3. Add two helpers:
   ```rust
   /// Map a bare CTR-type name onto an [`ECtrType`] (`ctr_type.h`).
   fn parse_ctr_type(name: &str) -> PyResult<ECtrType> { /* 6 variants + error listing all 6 */ }

   /// Parse ONE upstream CTR-description string (`"<Type>[:Prior=<f>]*"`,
   /// core.py:4957-4970) into `(ECtrType, Option<Vec<f64>>)`. Only the `Prior=`
   /// component is supported; every other component is REJECTED by name rather
   /// than silently ignored (honesty policy, params.rs:1-18). Upstream has NO
   /// standalone `simple_ctr_priors` kwarg — priors ride this grammar (OQ-1).
   fn parse_ctr_description(param: &str, value: &str)
       -> PyResult<(ECtrType, Option<Vec<f64>>)> { ... }
   ```
4. Add a small extractor that accepts `String` OR a 1-element `Vec<String>` for the
   kwarg value (upstream's declared list shape), erroring on any other length.
5. Wire into `make_builder`:
   ```rust
   if let Some(v) = ctr_description_arg(params, py, "simple_ctr")? {
       let (t, priors) = parse_ctr_description("simple_ctr", &v)?;
       builder = builder.simple_ctr(t);
       if let Some(p) = priors { builder = builder.simple_ctr_priors(p); }
   }
   // …identical block for "combinations_ctr" -> .combinations_ctr / .combinations_ctr_priors
   ```

- Do NOT add `simple_ctr_priors` / `combinations_ctr_priors` to `VOCABULARY` or
  `IMPLEMENTED` — they are NOT upstream kwargs (OQ-1). Doc-comment that decision at
  the helper.
- Do NOT implement `cat_features` (T14).

**Refactor**
- Constraint: keep the grammar parser total and panic-free (no `unwrap`, no slicing
  — use `split(':')`, `strip_prefix("Prior=")`, `str::parse::<f64>()`).
- Regression scope: full `pytest` + `cargo test -p catboost-rs-py`.

**Validation**
```bash
cargo test -p catboost-rs-py
cargo clippy -p catboost-rs-py --all-targets
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q
```

**Completion evidence:** A7 test green; a positive test proving
`simple_ctr="Borders:Prior=0.25"` sets both type and priors (asserted via a
`fit` that succeeds plus a Rust-side unit test in `params_test.rs` calling
`parse_ctr_description` directly).

---

### T14 — `cat_features` as a `fit(X, y, cat_features=...)` kwarg

- **Specs:** SPEC-CATF-14; resolves OQ-2, OQ-3
- **Blocked by:** T08 (categorical training must exist), T10 (predict must work)
- **Blocks:** T15
- **Parallelizable:** YES with T12/T13 (disjoint files) once T08+T10 have landed.

**Goal:** `CatBoostRegressor().fit(df, y, cat_features=[2])` produces the same `Pool`
categorical content as `Pool(df, y, cat_features=[2])`; `fit(pool_obj, cat_features=…)`
raises; a source that cannot ingest categoricals raises rather than silently dropping.

**Files**
- Modify: `crates/catboost-rs-py/src/estimator.rs` (`data_to_pool`)
- Modify: `crates/catboost-rs-py/src/regressor.rs`, `classifier.rs`, `ranker.rs`
  (`fit` signatures)
- Modify: `crates/catboost-rs-py/src/params.rs` (`IMPLEMENTED` += `"cat_features"`)
- Create: `crates/catboost-rs-py/tests/test_categorical.py`
- **Do NOT modify** `crates/catboost-rs-py/src/ingest_py.rs` (SPEC §7 verification-only).

**Verified symbols**
- `pub(crate) fn data_to_pool(py, x: &Bound<PyAny>, y: Option<&Bound<PyAny>>) -> PyResult<Pool>`
  with the Pool fast-path at `:241-246` and `ingest_to_owned(py, x, y, None)?` at `:247`
  `[VERIFIED: estimator.rs:236-250]` — the `None` is exactly SPEC-CATF-14's threading
  point.
- `ingest_to_owned(py, x, y, cat_features: Option<&[usize]>)` already takes the
  parameter `[VERIFIED: ingest_py.rs:52-57]`.
- `Pool.__new__` accepts `cat_features: Option<Vec<usize>>` and passes
  `cat_features.as_deref()` `[VERIFIED: crates/catboost-rs-py/src/pool.rs:56-98]` —
  **indices only, no string column names**. The fit-kwarg MUST use the same
  `Option<Vec<usize>>` type so the two surfaces agree; a `str` element raises the same
  PyO3 `TypeError` `Pool` raises today. (SPEC-CATF-14 writes
  `Optional[list[int | str]]`; name-based selection is a pre-existing `Pool`-level gap
  and is explicitly NOT introduced here — recorded as a follow-up.)
- Current signatures `#[pyo3(signature = (x, y = None))]`
  `[VERIFIED: regressor.rs:51-52, classifier.rs:60-61, ranker.rs:57-58]`; all three
  bodies call `data_to_pool(py, x, y)?`
  `[VERIFIED: regressor.rs:68, classifier.rs:68, ranker.rs:65]`.
- Other `data_to_pool(py, x, None)` call sites that must keep compiling:
  `regressor.rs:149,254`, `classifier.rs:100,126,280`, `ranker.rs:100`,
  `estimator.rs:270` `[VERIFIED: grep]`.
- Silent-drop hazard on the NumPy branch: finding F2.

**Red**
- File: `crates/catboost-rs-py/tests/test_categorical.py` (new)
- Test fn: `test_fit_cat_features_kwarg_matches_pool_cat_features` (SPEC A8)
  ```python
  import numpy as np, pandas as pd, pytest
  import catboost_rs
  from catboost_rs import CatBoostRegressor, Pool

  def _cat_frame():
      rng = np.random.default_rng(0)
      df = pd.DataFrame({
          "f0": rng.standard_normal(40).astype(np.float32),
          "f1": rng.standard_normal(40).astype(np.float32),
          "c0": [f"g{i % 5}" for i in range(40)],   # trailing categorical (F11)
      })
      y = rng.standard_normal(40).astype(np.float32)
      return df, y

  def test_fit_cat_features_kwarg_matches_pool_cat_features():
      df, y = _cat_frame()
      via_pool = CatBoostRegressor(iterations=5, one_hot_max_size=1)
      via_pool.fit(Pool(df, y, cat_features=[2]))
      via_kwarg = CatBoostRegressor(iterations=5, one_hot_max_size=1)
      via_kwarg.fit(df, y, cat_features=[2])       # <-- the new kwarg
      np.testing.assert_allclose(
          via_kwarg.predict(Pool(df, y, cat_features=[2])),
          via_pool.predict(Pool(df, y, cat_features=[2])),
          atol=0, rtol=0,
      )
  ```
- **Expected initial failure:**
  `TypeError: CatBoostRegressor.fit() got an unexpected keyword argument 'cat_features'`
  (PyO3 rejects the unknown kwarg against `signature = (x, y = None)`).
- Run:
  ```bash
  VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
  .venv/bin/python -m pytest crates/catboost-rs-py/tests/test_categorical.py -q -k cat_features_kwarg
  ```

**Green (minimum change)**
1. `estimator.rs`: widen `data_to_pool` to
   ```rust
   pub(crate) fn data_to_pool(
       py: Python<'_>,
       x: &Bound<'_, PyAny>,
       y: Option<&Bound<'_, PyAny>>,
       cat_features: Option<&[usize]>,
   ) -> PyResult<Pool>
   ```
   - **Pool fast-path (OQ-3 resolution — upstream-exact ERROR, `core.py:1522-1529`):**
     ```rust
     if let Ok(pool_ref) = x.cast::<crate::pool::Pool>() {
         if cat_features.is_some() {
             return Err(CatBoostValueError::new_err(
                 "cat_features should have the None type when X has a catboost_rs.Pool \
                  type: the Pool already declares its categorical columns and is the \
                  single source of truth",
             ));
         }
         return pool_ref.borrow().to_pool();
     }
     ```
     (Leave the existing `y`-is-ignored behavior and its WR-04 doc paragraph UNCHANGED
     — see OQ-3's recorded follow-up.)
   - Thread the value: `ingest_to_owned(py, x, y, cat_features)?`.
   - **Post-ingestion silent-drop guard (finding F2), inside `data_to_pool`, after
     `.into_pool()`:**
     ```rust
     if let Some(requested) = cat_features {
         if pool.n_cat_features() != requested.len() {
             return Err(CatBoostValueError::new_err(format!(
                 "`cat_features` declared {} categorical column(s) but the ingested \
                  dataset carries {}; categorical ingestion is currently supported \
                  only for a Pandas DataFrame with TRAILING object columns (a plain \
                  NumPy array carries no categorical block, and the Arrow/Polars path \
                  rejects cat_features outright)",
                 requested.len(), pool.n_cat_features(),
             )));
         }
     }
     ```
     This keeps `ingest_py.rs` untouched while making the NumPy drop impossible.
2. Update ALL seven existing `data_to_pool(py, x, None)` call sites to
   `data_to_pool(py, x, None, None)` (verified list above).
3. `regressor.rs` / `classifier.rs` / `ranker.rs`: change each `fit` to
   ```rust
   #[pyo3(signature = (x, y = None, cat_features = None))]
   fn fit(
       mut slf: PyRefMut<'_, Self>,
       py: Python<'_>,
       x: &Bound<'_, PyAny>,
       y: Option<&Bound<'_, PyAny>>,
       cat_features: Option<Vec<usize>>,
   ) -> PyResult<Py<Self>> {
       validate_params(&slf.base.params)?;
       let pool = data_to_pool(py, x, y, cat_features.as_deref())?;
       ...
   ```
   Docstring addition on each: the Pandas-trailing-object-column requirement (F11),
   the Pool-collision error, and (Ranker only) the upstream positional divergence.
4. `params.rs`: add `"cat_features"` to `IMPLEMENTED`. Do **not** add any
   `make_builder` branch for it — it is consumed at ingestion, not as a `BoostParams`
   scalar (research.md pitfall 2). Add a code comment saying exactly that, so a later
   reader does not "fix" the apparent omission.

**Refactor**
- Constraint: `data_to_pool`'s existing WR-04 doc block stays; append the new
  `cat_features` paragraphs rather than rewriting it.
- Regression scope: `cargo test -p catboost-rs-py`, full pytest, and specifically
  `test_ingestion.py` (the Pool/ingest contract suite) and `test_check_estimator.py`
  (sklearn contract — a changed `fit` arity can break `check_estimator`).

**Validation**
```bash
cargo test -p catboost-rs-py
cargo clippy -p catboost-rs-py --all-targets
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_categorical.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_ingestion.py \
                          crates/catboost-rs-py/tests/test_check_estimator.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
```

**Additional required tests in the same file (each a separate assertion, all must be
present before the task is complete):**
- `test_pool_plus_cat_features_kwarg_is_rejected` — `fit(Pool(...), cat_features=[2])`
  raises with `"single source of truth"` in the message (OQ-3).
- `test_numpy_plus_cat_features_kwarg_is_rejected_not_ignored` — `fit(np_x, y,
  cat_features=[2])` raises (finding F2); asserts the message names the count
  mismatch. **This must never pass by returning a fitted model.**
- `test_cat_features_kwarg_on_all_three_estimators` — the same kwarg is accepted by
  `CatBoostRegressor`, `CatBoostClassifier`, and (with a grouped `Pool`-free frame,
  expecting the existing group-less rejection) `CatBoostRanker`'s signature (OQ-2).

---

### T15 — Parameter-registry introspection stays truthful (+ resolve the vendored `core.py` gap)

- **Specs:** SPEC-CATF-15
- **Blocked by:** T12, T13, T14
- **Blocks:** T17
- **Parallelizable:** NO (needs all three preceding promotions landed).

**Goal:** `_param_status` returns `"IMPLEMENTED"` for all six promoted kwargs;
`nan_mode` stays `KNOWN_NOT_YET` and stays rejected at `fit()`; and the coverage test
`test_every_upstream_param_is_in_registry` **actually executes** instead of skipping.

**The gap this task must close (see OQ-1 evidence):**
`test_every_upstream_param_is_in_registry` reads
`<repo>/catboost-master/catboost/python-package/catboost/core.py`, which does not
exist — the test SKIPS today `[VERIFIED: shell pytest run]`. SPEC-CATF-15 requires it
to "still pass". A skipped test is not a gate. Resolution: extend the test's source
resolution to fall back to the **installed** `catboost` package's `core.py`
(`importlib.util.find_spec("catboost")` → `<pkg>/core.py`) when the vendored path is
absent, keeping the vendored path as the first choice. The installed package is the
identical 1.2.10 source `[VERIFIED: .venv catboost.__version__ == 1.2.10, matching
gen_fixtures.py's pinned CATBOOST_VERSION]`. The `pytest.skip` remains only if NEITHER
source is available (a true no-oracle environment).

**Files:** Modify `crates/catboost-rs-py/tests/test_params.py`. Production change
expected: none (T12–T14 already moved the six names).

**Red**
- Test fn: `test_promoted_ctr_kwargs_report_implemented`
  ```python
  def test_promoted_ctr_kwargs_report_implemented():
      for name in ("cat_features", "one_hot_max_size", "max_ctr_complexity",
                   "simple_ctr", "combinations_ctr", "counter_calc_method"):
          assert catboost_rs._param_status(name) == "IMPLEMENTED", name
      # The parity-gap sentinel must be untouched by this change.
      assert catboost_rs._param_status("nan_mode") == "KNOWN_NOT_YET"
  ```
- Second Red (the real one for this task) — make the coverage test executable:
  modify `_CORE_PY` resolution, then assert the test no longer skips:
  ```python
  def test_upstream_vocabulary_source_is_available():
      assert _core_py_path() is not None and _core_py_path().exists()
  ```
- **Expected initial failure:** `test_upstream_vocabulary_source_is_available` fails
  (`assert None is not None`) because only the missing vendored path is consulted.
  `test_promoted_ctr_kwargs_report_implemented` should already pass from T12–T14 — if
  any name fails, the corresponding earlier task is incomplete.
- Run: `.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q -rs`
  (`-rs` so a lingering SKIP is visible, not hidden).

**Green**
- Replace the module-level `_CORE_PY` constant with a `_core_py_path()` function:
  vendored path first, then `importlib.util.find_spec("catboost")`'s package dir,
  then `None`. Update both `_upstream_classifier_init_kwargs()` and the skip guard.
- Do NOT weaken `assert len(upstream) > 100`.

**Refactor**
- Constraint: `test_known_not_yet_param_rejected_as_parity_gap` (which uses
  `nan_mode="Min"`) and `test_typo_param_suggests_closest_match` must remain
  byte-unchanged — they are the R8 guard that the promotion did not over-reach
  `[VERIFIED: test_params.py:68-86]`.

**Validation**
```bash
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q -rs
# MUST report "0 skipped"
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
```

**Completion evidence:** pytest output showing `test_params.py` with **0 skipped**,
`test_every_upstream_param_is_in_registry` PASSED, and the six `IMPLEMENTED` +
`nan_mode` `KNOWN_NOT_YET` assertions green.

---

### T16 — Generate and FREEZE the categorical Python-API oracle fixture

- **Specs:** SPEC-CATF-17
- **Blocked by:** none
- **Blocks:** T17
- **Parallelizable:** YES — from the very start, concurrently with all of Wave 1–3.
  Touches only `crates/cb-oracle/generator/gen_fixtures.py` and a new
  `crates/cb-oracle/fixtures/catf_python_ctr/` directory.

**Goal:** a committed, never-regenerated upstream `catboost==1.2.10` reference for a
categorical training+prediction run whose configuration is **exactly reproducible
through the catboost-rs Python surface**.

**Files**
- Modify: `crates/cb-oracle/generator/gen_fixtures.py` (add `gen_catf_python_ctr()`
  plus its dispatch entry; mirror `gen_tensor_ctr_e2e` at `:2104-2240`
  `[VERIFIED: read in full]`)
- Create + COMMIT: `crates/cb-oracle/fixtures/catf_python_ctr/{X_cat.npy,y.npy,model.json,predictions.npy,config.json}`

**Fixture design (each choice is a verified risk mitigation)**

| Choice | Value | Why |
|---|---|---|
| Feature layout | **categorical-only**: 2 object columns, 0 float columns | Removes the quantization-border variable entirely, so SPEC-CATF-16's "use the model's own `float_feature_borders()`" constraint is satisfied vacuously and R5 cannot fire. Precedent: `tensor_ctr_e2e` trains with `feature_values = &[]` `[VERIFIED: tensor_ctr_e2e_oracle_test.rs:233-234]`. |
| Categorical values | **strings** (`"g0".."g4"` / `"h0".."h3"`), stored as `[N,2]` int codes in `X_cat.npy` + a documented `f"g{code}"` / `f"h{code}"` mapping in `config.json` | Both sides hash the identical string; no `stringify_int_category` divergence risk. `.npy` stays int32-loadable by `ndarray-npy` (the same trick `tensor_ctr_e2e` uses `[VERIFIED: its config.json npy_schema]`). |
| Cardinalities | 5 and 4, `one_hot_max_size=1` | Both columns are CTR-eligible (cardinality > one_hot_max_size), matching the routing rule at `cb-train/src/candidates.rs:92-104`. |
| `max_ctr_complexity` | **1** | Simple CTRs only — deliberately clear of the ORD-06/ORD-07 combination-gating bug (SPEC §2 non-goal, R7). `plain_ctr` uses the same value and passes `[VERIFIED: fixtures/plain_ctr/config.json]`. |
| `permutation_count` | **4** (upstream default; do NOT pin 1) | Finding F5: the builder hardcodes `permutation_count_default()` = 4 and has no setter. Pinning 1 would guarantee an unreproducible fixture. |
| `leaf_estimation_iterations` | **1** | Finding F6: `cb-train` is single-step. |
| `score_function` | `"Cosine"` explicitly on both sides | Finding F7. |
| `random_strength` | **0** explicitly on both sides | Finding F8, the known trap. |
| `loss_function` | `"Logloss"`, `boost_from_average=False` | Borders CTR needs target classes; matches every existing CTR fixture and lets the Python oracle compare `predict_proba[:,1]` against `sigmoid(raw)` exactly like `test_oracle_parity.py` does `[VERIFIED: test_oracle_parity.py:52-58]`. |
| Determinism | `thread_count=1`, `random_seed=0`, `verbose=False` | `gen_fixtures.py` ISOLATING discipline `[VERIFIED: :156-172]`. |
| Remaining | `depth=2, iterations=5, learning_rate=0.1, l2_leaf_reg=3.0, leaf_estimation_method="Gradient", bootstrap_type="No", counter_calc_method="SkipTest", simple_ctr=["Borders:Prior=0.5"], combinations_ctr=["Borders:Prior=0.5"], boosting_type="Plain"` | Every value is either the builder's pinned default or reachable through a T12/T13 kwarg. |

Route the whole config through the low-level `CatBoost(params)` API (not
`CatBoostClassifier(**kwargs)`) so non-sklearn keys are honored — the exact reason
`gen_tensor_ctr_e2e` does so `[VERIFIED: gen_fixtures.py comment at ~:2196]`.

**Red**
- There is no code under test here. The falsifiability requirement is the **freeze
  check** (R6): generate the fixture TWICE into separate directories with the identical
  seed and assert byte-identity of `predictions.npy` before committing.
  ```bash
  .venv/bin/python crates/cb-oracle/generator/gen_fixtures.py catf_python_ctr
  cp -r crates/cb-oracle/fixtures/catf_python_ctr /tmp/catf_run1
  .venv/bin/python crates/cb-oracle/generator/gen_fixtures.py catf_python_ctr
  diff -r /tmp/catf_run1 crates/cb-oracle/fixtures/catf_python_ctr
  ```
  **Expected:** `diff` is empty. If `model.json` differs (quantization
  nondeterminism, memory `ctr-model-loading`), that CONFIRMS the freeze requirement:
  commit run 1 and record the observed instability in `config.json`'s `note`.
  If `predictions.npy` differs, **STOP AND REPORT** — a nondeterministic reference
  cannot be an oracle.

**Green:** write `gen_catf_python_ctr()`; run it once; **commit** all five artifacts.

**Refactor**
- `config.json` MUST carry the standard keys (`scenario`, `requirement`, `seed`,
  `catboost_version`, `thread_count`, `n_rows`, `n_cat_features`, `params`,
  `npy_schema`, `note`) matching the `tensor_ctr_e2e` shape, plus the explicit
  `cat_value_map` documenting the int-code → string mapping, plus a
  `"NEVER regenerate in CI"` note (SPEC-CATF-17).

**Validation**
```bash
.venv/bin/python -c "import catboost; assert catboost.__version__ == '1.2.10'"
.venv/bin/python crates/cb-oracle/generator/gen_fixtures.py catf_python_ctr
git status --short crates/cb-oracle/fixtures/catf_python_ctr    # 5 new files
.venv/bin/python -c "import json,pathlib; \
  c=json.loads(pathlib.Path('crates/cb-oracle/fixtures/catf_python_ctr/config.json').read_text()); \
  assert c['params']['max_ctr_complexity']==1 and c['params']['permutation_count']==4 \
     and c['params']['random_strength']==0 and c['params']['thread_count']==1"
```

**Completion evidence:** five committed files; the double-generation diff output;
the assertion command above exiting 0.

---

### T17 — Public-Python-API categorical oracle at ≤1e-5

- **Specs:** SPEC-CATF-16
- **Blocked by:** T11, T15, T16
- **Blocks:** T18
- **Parallelizable:** NO (final integration gate).

**Goal:** `catboost_rs.Pool(df, y, cat_features=[0,1])` → `CatBoostClassifier(...).fit()`
→ `.predict_proba()` matches the frozen upstream reference within 1e-5, driven
entirely through the PUBLIC Python API.

**Files:** Modify `crates/catboost-rs-py/tests/test_categorical.py`;
modify `crates/catboost-rs-py/tests/conftest.py` (add an `oracle_catf_ctr` fixture
mirroring `oracle_binclf` at `conftest.py:~92-115` `[VERIFIED: read in full]`).

**Red**
- Test fn: `test_python_api_categorical_oracle_parity` (SPEC A3 / CATF-16)
  ```python
  def test_python_api_categorical_oracle_parity(oracle_catf_ctr):
      df   = oracle_catf_ctr["df"]          # 2 object cols, built from the int codes
      y    = oracle_catf_ctr["y"]
      ref  = oracle_catf_ctr["ref_raw"]     # upstream RawFormulaVal
      est = catboost_rs.CatBoostClassifier(
          loss_function="Logloss", iterations=5, depth=2, learning_rate=0.1,
          l2_leaf_reg=3.0, boost_from_average=False,
          leaf_estimation_method="Gradient", bootstrap_type="No",
          score_function="Cosine", random_strength=0, random_seed=0,
          one_hot_max_size=1, max_ctr_complexity=1,
          counter_calc_method="SkipTest",
          simple_ctr="Borders:Prior=0.5", combinations_ctr="Borders:Prior=0.5",
      )
      est.fit(catboost_rs.Pool(df, y, cat_features=[0, 1]))
      proba = est.predict_proba(catboost_rs.Pool(df, y, cat_features=[0, 1]))
      # Non-degeneracy: a constant prediction vector would make the comparison vacuous.
      assert proba[:, 1].std() > 1e-6
      np.testing.assert_allclose(proba[:, 1], _sigmoid(ref), atol=1e-5, rtol=0)
  ```
- Second assertion in the same file (SPEC A3's fit-kwarg twin):
  `test_python_api_categorical_oracle_parity_via_fit_kwarg` — identical, but
  `est.fit(df, y, cat_features=[0, 1])`, proving the T14 path reaches the same numbers.
- **Expected initial failure (before T12–T16):**
  `CatBoostParameterError: parameter 'max_ctr_complexity' ... (parity gap)`.
  After T12–T16 the meaningful failure mode is a numeric
  `Not equal to tolerance atol=1e-05` mismatch.
- Run:
  ```bash
  VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
  .venv/bin/python -m pytest crates/catboost-rs-py/tests/test_categorical.py -q -k oracle
  ```

**Green:** none expected — the production behavior is delivered by T08–T14. This task
adds the conftest fixture and the assertions. If the ≤1e-5 gate fails, run the
localization ladder below; **do not** patch `cb-train` or `cb-model`.

**Localization ladder (run in order; STOP AND REPORT at the first hit)**
1. Compare `est.predict_proba` to a direct Rust
   `cb_train::train_cat` + `cb_model::predict_raw_cat` run on the same corpus
   (a scratch Rust integration test, deleted afterward). If the Rust-direct run also
   diverges, the gap is inside the already-shipped engine, not this plan's wiring →
   **STOP AND REPORT**.
2. Re-generate the fixture at `permutation_count=1` and re-compare. If it passes at 1
   and fails at 4 → **permutation_count facade-parity gap (finding F5). STOP AND
   REPORT.** Do not add a `permutation_count` setter — it is a separate facade-wiring
   plan (SPEC §2 non-goals).
3. Re-generate at `max_ctr_complexity=2` and compare. If it passes at 1 and fails
   at 2 → **this is the ORD-06/ORD-07 combination-CTR gating bug
   (`.planning/phases/24-ctr-split-search-correctness/`). STOP AND REPORT. DO NOT
   FIX IT HERE** (SPEC §2, R7).
4. Any other divergence → report the localized stage with numbers; do not proceed.

**Refactor**
- Constraint: the comparison must read the model's own borders. With a
  categorical-only fixture there are none, so also add the explicit assertion
  `assert est._model_float_feature_count() == 0` — or, if no such accessor exists,
  assert `df.select_dtypes(include="number").shape[1] == 0` in the conftest fixture,
  documenting that R5 is structurally excluded.

**Validation**
```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_categorical.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
```

**Completion evidence:** both oracle assertions green with `atol=1e-5, rtol=0`, plus
the non-degeneracy guard, plus the full pytest suite green.

---

### T18 — Float-only `fit()` no-regression gate

- **Specs:** SPEC-CATF-07
- **Blocked by:** T17
- **Blocks:** none (final task)
- **Parallelizable:** NO.

**Goal:** the float-only path is provably unchanged: `cb_train::train` (not
`train_cat`) is the entrypoint, and the three pre-existing facade oracle tests pass
**with zero diff to their files**.

**Files:** Modify `crates/catboost-rs/src/builder_test.rs` (one added test). The three
oracle test files are **read-only** for this task.

**Red — MUTATION-CHECK protocol (§3)**
- Test fn: `float_only_fit_is_never_ctr_routed`
  ```rust
  // A pool with zero categorical columns must produce a model with NO CTR data —
  // the D-04 / SPEC-CATF-07 byte-identity invariant, asserted structurally.
  let pool = /* small numeric-only OwnedColumns -> Pool */;
  let model = CatBoostBuilder::new().iterations(3).fit(&pool).expect("fit");
  assert!(model.as_canonical().ctr_data.is_none(),
          "a cat-free pool must never route through train_cat");
  ```
  (`builder_test.rs` is a child of `builder`, and the crate root sets
  `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, ...))]`
  `[VERIFIED: crates/catboost-rs/src/lib.rs:1]`, so `expect` is permitted in test code.)
- Expected first run: PASS.
- **Mandatory mutation:** in `fit()`, invert the branch predicate
  (`pool.cat_features().is_empty()` → `!pool.cat_features().is_empty()`). Re-run.
  **Expected failure:** ``a cat-free pool must never route through train_cat``.
  Revert; confirm green. Record both outputs.

**Green:** none.

**Refactor:** none.

**Validation — the full gate**
```bash
# 1. The three pre-existing oracle tests, UNMODIFIED (prove it):
git diff --stat crates/catboost-rs/tests/builder_oracle_test.rs \
                crates/catboost-rs/tests/cv_oracle_test.rs \
                crates/catboost-rs/tests/grid_search_oracle_test.rs
#    -> MUST print nothing.
cargo test -p catboost-rs --test builder_oracle_test \
                          --test cv_oracle_test \
                          --test grid_search_oracle_test

# 2. Everything else.
cargo test --workspace
cargo clippy --workspace --all-targets
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
```

`cv.rs:407` and `grid_search.rs:405,515` call `builder.fit(&pool)` with an unmodified
`&Pool` `[VERIFIED: grep of both files]`, so they inherit the routing automatically and
`cv_oracle_test` / `grid_search_oracle_test` are the correct inheritance gates.

**Completion evidence:** the empty `git diff --stat`; the three oracle tests green;
`cargo test --workspace` green; the recorded mutated-fail output.

---

## 5. SPEC-ID → task coverage table

| Spec ID | Title | Task(s) | Primary assertion |
|---|---|---|---|
| SPEC-CATF-01 | `one_hot_max_size` setter | **T01** | `boost_params().one_hot_max_size == 7` |
| SPEC-CATF-02 | `max_ctr_complexity` setter | **T02** | `boost_params().max_ctr_complexity == 3` |
| SPEC-CATF-03 | `simple_ctr` + priors (lockstep) | **T03** | both `BoostParams` fields reflect the setters |
| SPEC-CATF-04 | `combinations_ctr` + priors (lockstep) | **T04** | both `BoostParams` fields reflect the setters |
| SPEC-CATF-05 | `counter_calc_method` setter | **T05** | `boost_params().counter_calc_method == Full` |
| SPEC-CATF-06 | `boost_params()` default-equivalence | **T06** | all 7 fields == `*_default()`; mutation-checked |
| SPEC-CATF-07 | float-only `fit()` no-regression | **T18** | 3 oracle tests unmodified + `ctr_data.is_none()`; mutation-checked |
| SPEC-CATF-08 | categorical `fit()` → `train_cat` + bake | **T08** | `ctr_data.is_some()` AND ≥1 `ModelSplit::Ctr` |
| SPEC-CATF-09 | setter types nameable downstream | **T07** | `use catboost_rs::{CounterCalcMethod, ECtrType};` compiles in a separate crate |
| SPEC-CATF-10 | categorical width validation | **T09** | typed `FeatureMismatch` naming both counts |
| SPEC-CATF-11 | `predict_with` is CTR-aware | **T10** | facade `predict` == direct `predict_raw_cat` |
| SPEC-CATF-12 | never silently mis-scored | **T11** | typed error on all 3 entrypoints w/ zero cat cols; mutation-checked |
| SPEC-CATF-13 | five CTR kwargs from Python | **T12** (scalar/enum), **T13** (ctr-description grammar) | fit accepted + `CatBoostParameterError` naming the bad value and listing variants |
| SPEC-CATF-14 | `cat_features` fit-kwarg | **T14** | kwarg path ≡ Pool path; Pool-collision error; NumPy-drop error |
| SPEC-CATF-15 | registry stays truthful | **T15** | six `IMPLEMENTED`, `nan_mode` `KNOWN_NOT_YET`, coverage test **0 skipped** |
| SPEC-CATF-16 | public-Python-API oracle ≤1e-5 | **T17** | `predict_proba[:,1]` vs `sigmoid(ref)` at `atol=1e-5, rtol=0` |
| SPEC-CATF-17 | fixture frozen, not regenerated | **T16** | 5 committed files + double-generation diff |

**All 17 specifications covered. Every task references ≥1 spec ID.**

---

## 6. Risk register (mapped from SPEC §9, with the plan's concrete mitigation)

| # | Risk | Mitigation in this plan |
|---|---|---|
| R1 | Inline-default → field refactor silently changes a default | T06, with a mandatory mutation check |
| R2 | Categorical `fit()` "passes" because the float path ignored the cat columns | T08 asserts BOTH `ctr_data.is_some()` and a `ModelSplit::Ctr` |
| R3 | CTR model silently mis-scored on predict | T11 asserts a typed error on all three entrypoints; the `Ok(_)` arm's panic message says "SILENTLY SCORED" |
| R4 | Non-default `ECtrType` paired with mismatched priors | T03/T04 ship priors in lockstep; T13's grammar parses both together |
| R5 | Oracle uses shared quantization borders | T16's fixture is categorical-only → zero float features → structurally impossible |
| R6 | Fixture regenerated in CI | T16 commits the artifacts and runs a double-generation determinism check; `config.json` carries the "NEVER regenerate" note |
| R7 | Scope leak into ORD-06/07 | `max_ctr_complexity=1` everywhere; T17's localization ladder step 3 mandates **STOP AND REPORT** |
| R8 | `test_params.py` coverage test goes stale | T15 updates it AND fixes its silent SKIP |
| **R9 (new)** | `permutation_count` 4-vs-1 facade gap breaks the oracle (finding F5) | T16 generates at 4; T17 ladder step 2 mandates **STOP AND REPORT**, not a setter |
| **R10 (new)** | `fit(np_array, y, cat_features=…)` silently drops the kwarg (finding F2) | T14's post-ingestion width guard in `estimator.rs` (no `ingest_py.rs` edit) + a dedicated test |

---

## 7. Unresolved blockers and unverified assumptions

**Blockers: none.** Every symbol, signature, path, and command cited above was
verified against the working tree.

**Assumptions carried into implementation (each with its falsification point):**

1. **A1 — `cb-train` honors `permutation_count=4` on the CTR path equivalently to
   upstream.** Not verifiable without running the oracle. Falsified by T17 ladder
   step 2 → STOP AND REPORT (R9). This is the single largest schedule risk in the plan.
2. **A2 — `max_ctr_complexity=1` with 2 CTR-eligible columns is outside the ORD-06/07
   bug's blast radius.** Supported by `plain_ctr` (1 cat feature, complexity 1) and
   `tensor_ctr_e2e` (2 cat features, complexity 2) both passing today, but not proven
   for the exact new configuration. Falsified by T17 ladder step 3 → STOP AND REPORT.
3. **A3 — the installed `catboost` 1.2.10 `core.py` is an acceptable substitute for the
   absent vendored `catboost-master/.../core.py` for T15's vocabulary source.** The
   version matches `gen_fixtures.py`'s pin exactly. If the maintainer intends the
   vendored tree to be restored instead, T15's fallback ordering (vendored first)
   already accommodates that with no further change.
4. **A4 — `check_range` bounds `one_hot_max_size ∈ [0,255]` and
   `max_ctr_complexity ∈ [1,8]`.** These are plausible upstream domains but were NOT
   read out of upstream C++ source in this pass (`catboost-master` contains only
   `catboost/private/libs/algo`, not `libs/options/cat_feature_options.cpp`). T12's
   implementer should confirm against upstream docs or widen the bound rather than
   invent a tighter one; a wrong bound only over-rejects, and is caught by any test
   using a legitimate value.
5. **A5 — `.venv`'s installed `catboost_rs` is rebuilt by `maturin develop` before each
   Python test run.** Every Python validation block in this plan includes the
   `maturin develop` line for exactly this reason.

**Recorded follow-ups (explicitly OUT of this plan's scope):**
- Align the `Pool` + `y` fast-path with upstream's `CatBoostError` (OQ-3 asymmetry).
- `cat_features` by column NAME (upstream accepts `list[int | str]`; this repo's
  `Pool` and therefore the new fit-kwarg accept indices only).
- A `permutation_count` builder setter (only if A1 is falsified).
- CTR-awareness for `shap_values` / `staged_predict` / ONNX / CoreML — SPEC §2
  confirms the existing typed rejections are correct and must be preserved.

---

## 8. Process attestation

**No GSD skill, command, workflow, or agent was used at any point in producing this
plan.** No `planning/settings.json` was read (none exists at that path in this
repository; this plan lives under `.planning/plans/`, the directory the invoking
message specified). No `spec-tdd-planner-skill`, `planning-settings-skill`, or any
other Skill-tool invocation occurred. Tools used: `Read`, `Bash` (read-only
inspection commands plus one read-only `pytest` run), `codegraph_explore` (MCP), and
`Write` for this file only. No production source file was created or modified.
