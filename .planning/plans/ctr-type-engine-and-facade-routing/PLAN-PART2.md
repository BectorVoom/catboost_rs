---
title: TDD implementation plan — Part 2, cat_features/CTR facade routing (tasks F00–F23)
parent: ./PLAN.md
spec: ./SPEC.md
amends: ../catboost-builder-cat-features-routing/PLAN.md
status: ready-for-implementation
---

# Part 2 — `cat_features` / CTR facade routing

Part 2 **amends and executes** `../catboost-builder-cat-features-routing/PLAN.md`
(18 tasks T01–T18, PLAN-CHECK verdict `ISSUES_FOUND`: 2 BLOCKER, 2 CRITICAL,
4 MAJOR, 4 MINOR). Where a T-task body survives, this document **cites it by
anchor** rather than duplicating it; where it changes, the delta is stated in full.

§3 / §3.1 / §3.2 / §4 of `./PLAN.md` apply verbatim.

**Part 2 starts only after E23 is green.** Every one of PLAN-CHECK's blocking
findings is resolved below, and the resolution is stated per task.

---

## P2.0 — Disposition of PLAN-CHECK's 12 required revisions

| # | PLAN-CHECK finding | Disposition here | Task |
|---|---|---|---|
| 1 | **BLOCKER-1** `gen_fixtures.py <scenario>` unfreezes the whole corpus | **DISSOLVED, not mitigated.** Per the locked user decision, Part 2 runs **NO corpus-wide generation at all** and reuses the already-committed `crates/cb-oracle/fixtures/one_hot_train/`. `generator/gen_fixtures.py` is never invoked. (SPEC-CATF-Δ3) | F19, F20 |
| 2 | **BLOCKER-2** four params inert ⇒ promoting them manufactures silently-ignored kwargs | **RESOLVED BY PART 1.** After E10/E11/E22 the type, the prior lists and `counter_calc_method` are genuinely read. Promotion is now honest — **except** the single-description limit, which F00 documents as an explicit parity gap. (SPEC-CATF-Δ2, SPEC-CTRT-19) | F00, F03, F04, F05, F16 |
| 3 | **CRITICAL-3** derived cat width is a data-dependent lower bound enforced as equality | **RESOLVED by option 1**: `cb_model::Model` carries an explicit trained cat width; the check uses the pool's **DECLARED** width, never a width derived from chosen splits. (SPEC-CATF-Δ4) | F08, F10 |
| 4 | **CRITICAL-4** one-hot columns silently dropped | **OBSOLETE.** `ModelSplit::OneHot`, `ObliviousTree.one_hot_splits`, `level_kinds` and `passes_one_hot_split` all exist and the device grower emits one-hot levels. Part 2 **verifies**, does not re-plan. The verifying assertion lives in **F09 test fn 2** only (F01's duplicate is deleted — it cannot pass before `fit()` routes). (SPEC-CATF-Δ1) | F09 |
| 5 | **MAJOR-5** 4 un-listed `data_to_pool` sites | **RESOLVED.** All **14** call sites enumerated below, verified by grep this session. (SPEC-CATF-Δ5) | F17 |
| 6 | **MAJOR-6** `cv()`/`grid_search()` undefined on a categorical pool | **RESOLVED**: defined behavior + a test. (SPEC-CATF-Δ6) | F14 |
| 7 | **MAJOR-7** other scoring paths silently wrong | **HALF ALREADY DONE** by the uncommitted one-hot wave (`shap_values` now returns `Result<_, ShapUnsupported>` and rejects one-hot **and** CTR splits). F13 closes `partial_dependence` + `feature_importance_with_data`. **The residual hole — `staged_predict` on a ONE-HOT model — is closed by F11 (the `ensure_scalar_oblivious` one-hot arm) and F12 (the `staged_predict_facade_test` case + mutation check).** (SPEC-CATF-Δ7) | F11, F12, F13 |
| 8 | **MAJOR-8** three invalid `cargo test --test <target>::<filter>` commands | **FIXED** — every command here uses `--test <target> -- <filter>` (§3.2). | all |
| 9 | **MINOR-9** T06 mutation cannot detect a `boost_params()` cross-wire | **FIXED** — F06 mandates a **second** mutation on a `boost_params()` read. | F06 |
| 10 | **MINOR-10** `list[int | str]` narrowed to `list[int]` | **RECORDED** as a spec amendment + a follow-up. | F17 |
| 11 | **MINOR-11** width guard misfires on duplicate indices | **FIXED** — de-duplicate + range-check before comparing. | F17 |
| 12 | T17 should cite `multi_permutation_e2e_oracle_test.rs`; downgrade A1/R9 | **APPLIED** — the `permutation_count=4` risk is already closed by that passing oracle. | F20 |

---

## WAVE W6 — Honesty first, then the builder setters

### F00 — Document the multi-description default parity gap (SPEC-CTRT-19)

- **Specs:** SPEC-CTRT-19; SPEC-CATF-Δ2
- **Blocked by:** E23. **Blocks:** F03, F05, F16.
- **Parallelizable:** **YES** with F01/F02 (F00 owns doc comments in `cb-train` and
  `catboost-rs-py/src/params.rs`'s header; F01/F02 own `catboost-rs/src/builder.rs`).

**Goal / observable completion condition.** The divergence between upstream's CPU
default `simple_ctr = [Borders(0/1, 0.5/1, 1/1), Counter(0/1)]` and this repo's
**scalar single-description** model is recorded, with the upstream anchor, in every
place a user or implementer will look — and a test asserts the documentation exists
so it cannot silently rot.

**Files**
- Modify: `crates/cb-train/src/boosting.rs` (`simple_ctr_default` doc at `:461-466`,
  `simple_ctr_priors_default` at `:468-475`, `combinations_ctr_default` at
  `:506-511`, `combinations_ctr_priors_default` at `:513-520`)
- Modify: `crates/catboost-rs-py/src/params.rs` (the honesty-policy header block at
  `:1-18`)
- Modify: `.planning/plans/ctr-type-engine-and-facade-routing/SPEC.md` (§10 open
  question 6 → resolved)

**Exact verified files/symbols to touch**
- `pub fn simple_ctr_default() -> ECtrType { ECtrType::Borders }` at
  `crates/cb-train/src/boosting.rs:464-466`, whose current doc already says
  "the numeric/one-hot slices leave it here and never exercise the CTR path" —
  **now false after Part 1** `[VERIFIED: CODEGRAPH, read verbatim]`.
- `combinations_ctr_default()` at `:509-511`, same stale framing
  `[VERIFIED: CODEGRAPH]`.
- `simple_ctr_priors_default()` (`:473`) and `combinations_ctr_priors_default()`
  (`:518`) both return `vec![0.5]`; upstream's Borders default is
  `[0.0, 0.5, 1.0]` `[VERIFIED: research §D.2]`.
- `crates/catboost-rs-py/src/params.rs:1-18` is the stated honesty policy
  (Implemented / KnownNotYet(rejected) / Unknown(rejected) — **never silently
  ignored**) `[VERIFIED: Part-2 PLAN.md T12 anchor]`.
- Upstream anchor to cite: `catboost_options.cpp:439-453` (`SetCtrDefaults`),
  measured via `get_all_params()` as
  `['Borders:…:Prior=0/1:Prior=0.5/1:Prior=1/1', 'Counter:…:Prior=0/1']`
  `[VERIFIED: research §D.1, EXPERIMENT probe2.py]`.

**Red**
- File: `crates/catboost-rs-py/tests/test_params.py`
- Test fn: `test_ctr_default_parity_gap_is_documented`
  ```python
  def test_ctr_default_parity_gap_is_documented():
      src = (REPO / "crates/catboost-rs-py/src/params.rs").read_text()
      assert "catboost_options.cpp:439-453" in src, \
          "the multi-description default parity gap must cite its upstream anchor"
      assert "single-description" in src or "one CTR description" in src
  ```
- **EXPECTED INITIAL FAILURE:**
  `AssertionError: the multi-description default parity gap must cite its upstream anchor`.
- Run: `.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q -k parity_gap`

**Green (minimal implementation intent).** Add to each of the four `*_default()`
doc comments, and to the `params.rs` header, a paragraph of exactly this content:
*"KNOWN PARITY GAP: upstream's CPU default is a LIST of two CTR descriptions
(`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`,
`catboost_options.cpp:439-453`). This crate models ONE description with a prior
LIST. The type and the full prior list ARE honored (SPEC-CTRT-09/10/11); a
simultaneous `[Borders, Counter]` configuration is NOT representable. Deliberate:
`simple_ctr: ECtrType` is pinned at 62 construction sites and retyping it has zero
behavioral benefit to any of them."*

**Refactor constraints + required regression scope**
- Constraint: **doc comments only** — no signature, no default value changes. In
  particular do **not** change `simple_ctr_priors_default()` from `vec![0.5]` to
  upstream's `[0.0, 0.5, 1.0]`: that would move every existing CTR oracle.
- Constraint: delete the now-false "inert here" / "never exercise the CTR path"
  sentences rather than leaving them beside the new text.
- Regression scope: `cargo test -p cb-train`, full pytest.

**Validation**
```bash
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q
```

**Completion evidence.** The doc-presence test green; `grep -n "inert here" crates/cb-train/src/boosting.rs`
returns nothing for the four CTR defaults.

---

### F01–F07 — Builder setters, default guard, re-exports

These execute the surviving bodies of the existing Part-2 plan with the deltas
listed. **Read the cited task body in
`../catboost-builder-cat-features-routing/PLAN.md` before implementing; it carries
the verbatim Green code and the verified `builder.rs` anchors.**

| Task | Source | Specs | Blocked by | Blocks | Parallel | Delta this plan imposes |
|---|---|---|---|---|---|---|
| **F01** `one_hot_max_size` setter + create `builder_test.rs` | T01 (PLAN.md §4 T01, incl. finding **F1**: `boost_params()` is a **private** `fn` at `builder.rs:259`, so `builder_test.rs` MUST be mounted as a **CHILD** module via `#[cfg(test)] #[path = "builder_test.rs"] mod builder_test;` at the bottom of `builder.rs` — precedent `crates/catboost-rs/src/cv.rs:551-553`; a `mod builder_test;` line in `lib.rs` would make it a SIBLING and it could not see `boost_params`) | SPEC-CATF-01, Δ1 | E23 | F02 | NO (owns `builder.rs`); YES vs F07/F19 | **UPGRADED (Δ1):** drop T01's "semantically hollow" caveat — one-hot columns now genuinely train. **The second test `cardinality_two_column_contributes_at_default_one_hot_max_size` is DELETED from F01.** It cannot pass here: at F01 time `CatBoostBuilder::fit` still calls the float-only `train` (`crates/catboost-rs/src/builder.rs:383-391` `[VERIFIED]`) and `pool.cat_features()` is never read, so no `ModelSplit::OneHot` can exist. The identical assertion already lives in **F09 test fn 2** (`fit_low_cardinality_pool_emits_one_hot_splits_not_ctr`), which is where `fit()` routing lands — keep it **only** there. F01's deliverable is the setter plus the `builder_test.rs` mount. |
| **F02** `max_ctr_complexity` setter | T02 verbatim | SPEC-CATF-02 | F01 | F03 | NO | none (already live in the engine, `boosting.rs`) |
| **F03** `simple_ctr` + `simple_ctr_priors` setters (lockstep) | T03 | SPEC-CATF-03, Δ2 | F00, F02 | F04 | NO | **REVISED:** delete T03's BLOCKER-2 "not consumed by `train_inner`" doc requirement — after E10 it IS consumed. **REPLACE** with F00's parity-gap paragraph + a note that `Prior=<n>/<d>` with `d != 1` is illegal on CPU anyway (`ctr_helper.cpp:50`), which vindicates the engine's `prior_denom: 1.0` pin at `crates/cb-train/src/boosting.rs:1942`. |
| **F04** `combinations_ctr` + priors (lockstep) | T04 | SPEC-CATF-04, Δ2 | F03 | F05 | NO | same as F03 |
| **F05** `counter_calc_method` setter | T05 | SPEC-CATF-05, Δ2 | F04 | F06, F15 | NO | **REVISED:** the doc comment MUST state that the flag is **observable only with an eval set** (measured `0.000e+00` learn-only vs `4.010e-01` with eval) and cite E23's outcome (fixture-verified or recorded deferral). Do **not** claim blanket parity coverage. |
| **F06** `boost_params()` default-equivalence guard | T06 | SPEC-CATF-06 | F05 | **F08** | NO | **MINOR-9 FIX:** T06's single mutation (`new()`'s `simple_ctr_priors: vec![0.0]`) cannot detect a `boost_params()` **read** cross-wire, because `simple_ctr_default()` and `combinations_ctr_default()` both return `Borders`. **ADD a second mandated mutation on a read** — e.g. `one_hot_max_size: self.max_ctr_complexity as u32` — and record BOTH failure texts. |
| **F07** re-export `ECtrType` / `CounterCalcMethod` | T07 verbatim | SPEC-CATF-09 | E23 | F15 | **YES** — owns `crates/catboost-rs/src/lib.rs` + a new `tests/reexport_ctr_types_test.rs`; no overlap with `builder.rs` or `model.rs` | none |

**Shared Red/Green/Validation shape for F01–F06** (from T01, verified):
- Red: `cargo test -p catboost-rs --lib builder::builder_test -- <name>`;
  **EXPECTED INITIAL FAILURE**
  `error[E0599]: no method named '<setter>' found for struct 'CatBoostBuilder'`
  (or, if the `#[path]` mount is added first,
  `error[E0583]: file not found for module` — create the file, then the mount).
- Green: add field → `new()` default → `#[must_use] pub fn name(mut self, …) -> Self`
  → `boost_params()` reads `self.<field>` (with `.clone()` for the two `Vec<f64>`
  priors, mirroring the existing `loss: self.loss.clone()` precedent at
  `builder.rs:261-264`); DELETE the field's "pinned / inert here" comment.
- Refactor constraint: never reorder a `boost_params()` field.
- Regression scope for every one of F01–F06:
  ```bash
  cargo test -p catboost-rs --lib builder::builder_test
  cargo test -p catboost-rs --test builder_oracle_test --test cv_oracle_test \
                            --test grid_search_oracle_test
  cargo test -p catboost-rs
  cargo clippy -p catboost-rs --all-targets
  ```
- **MINOR-12 note:** `CatBoostBuilder` derives `Debug, Clone, PartialEq`
  (`builder.rs:60-65`); adding seven fields widens both. `ECtrType` and
  `CounterCalcMethod` both derive `Debug, Clone, Copy, PartialEq, Eq`
  `[VERIFIED: crates/cb-train/src/ctr/mod.rs:65-67, :129-130]`, so no derive
  breaks — but F04/F05 must re-run `cargo test -p catboost-rs -p catboost-rs-py`
  for any `assert_eq!` on a builder or a `Debug` string.

**Completion evidence for F01–F07.** Each setter's unit test green; **F01 carries
NO one-hot-contribution test** — that assertion belongs to F09 test fn 2 and is
evidenced there; F06's recorded pass → 2× mutated-fail → reverted-pass; F07's
separate-crate import compiling; the three pre-existing facade oracle tests passing
**with zero diff to their files**.

---

### F09 — `fit()` routes a categorical pool to `train_cat` and bakes CTR data

> **ORDERING NOTE (revised).** F09 now runs **AFTER F08**, whose section appears
> further down under "WAVE W7" in this document. The previous order
> (`F06 -> F09 -> F08`) was an **inverted producer/consumer edge**: F09's Green step 2
> calls `Model::with_cat_feature_count`, which **F08 creates**. Implement F08 first,
> then this task. The §4 edge list now reads `F06->F08  F08->F09  F09->F10`.

- **Specs:** SPEC-CATF-08; SPEC-CATF-Δ1; acceptance **A11** precondition
- **Blocked by:** **F08** (which is itself blocked by F06). **Blocks:** F10, F17.
- **Parallelizable:** **NO** — owns `crates/catboost-rs/src/builder.rs`.

**Source body:** `../catboost-builder-cat-features-routing/PLAN.md` §4 **T08** —
its verified `train_cat` call shape, the `rayon::join` constraint, the
`CB_GPU_PROF` timing block and the exact branch code are correct and are reused
verbatim. `train_cat`'s signature is re-verified this session at
`crates/cb-train/src/boosting.rs:2236-2245` `[VERIFIED: LOCAL]`.

**Deltas this plan imposes**
1. **Δ1 assertion (new).** T08 asserts `ctr_data.is_some()` **and** `≥1
   ModelSplit::Ctr`. **ADD** a sibling test using a pool whose cat column has
   cardinality 2 at `one_hot_max_size(2)`, asserting `≥1 ModelSplit::OneHot` and
   `ctr_data.is_none()` — proving the one-hot arm of `fit()` is live and that the
   CRITICAL-4 hazard is genuinely closed. `ModelSplit::{Float, OneHot, Ctr}` is
   verified to have exactly three variants `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:101, OneHotModelSplit at :80]`.
2. **Store the trained cat width (Δ4 precondition).** In the categorical arm, after
   `CtrData::from_baked`, call F08's
   `Model::with_cat_feature_count(pool.n_cat_features())` — **which already exists,
   because F08 lands first.** This is what makes F10's check use the DECLARED width
   rather than a derived lower bound.
3. **No `EncodingPath::OneHot` rejection.** T08's CRITICAL-4 companion task from
   PLAN-CHECK (revision #4) is **DROPPED** — one-hot columns now train, so a
   rejection would be wrong. Record that explicitly in the task notes so a reader
   of PLAN-CHECK does not re-add it.

**Red**
- File: `crates/catboost-rs/tests/builder_cat_facade_test.rs` (new)
- Test fn 1: `fit_categorical_pool_bakes_ctr_data_and_emits_a_ctr_split`
  (T08's body verbatim, using the frozen `crates/cb-oracle/fixtures/tensor_ctr_e2e/`
  corpus — `X_cat.npy` `[30,2] int32`, `y.npy` `[30] float64` — with
  `one_hot_max_size(1)`, `max_ctr_complexity(1)`).
  **EXPECTED INITIAL FAILURE:** ``assertion failed: inner.ctr_data.is_some()`` —
  `fit()` calls the float-only `train`, which never sees the cat columns.
- Test fn 2: `fit_low_cardinality_pool_emits_one_hot_splits_not_ctr` (new, Δ1).
  Fits a pool whose cat column has cardinality 2 at `one_hot_max_size(2)` and
  asserts `≥1 ModelSplit::OneHot` and `ctr_data.is_none()`. **This is the SOLE home
  of the cardinality-2 one-hot-contribution assertion** — F01's duplicate has been
  deleted, because it cannot pass before `fit()` routes.
  **EXPECTED INITIAL FAILURE:** ``assertion failed: has_one_hot_split``.
- Test fn 3: `fit_categorical_pool_records_the_declared_cat_width` (new, Δ4).
  **EXPECTED INITIAL FAILURE:** ``assertion `left == right` failed`` on the recorded
  width — **not** a compile error. `Model::with_cat_feature_count` and
  `cat_feature_count()` already exist, because **F08 lands before F09**. The earlier
  "sequence F09 → F08 … with test fn 3 red until F08" concession is **RETRACTED**;
  it contradicted this task's own Completion evidence ("Three tests green").
- Run: `cargo test -p catboost-rs --test builder_cat_facade_test`

**Green / Refactor / Validation.** As T08, plus the two additions. Regression scope
is T08's (`builder_oracle_test`, `cv_oracle_test`, `grid_search_oracle_test`
unmodified — `git diff --stat` over those three files must print nothing).

**Completion evidence.** Three tests green; the three pre-existing oracle files
byte-unchanged.

---

## WAVE W7 — Model-side cat width and predict routing

> **F08's section is printed here for continuity with the original document layout,
> but it EXECUTES AT THE END OF W6, BEFORE F09** — it is the producer of
> `Model::with_cat_feature_count`, which F09 calls. §4's edge list
> (`F06->F08  F08->F09  F09->F10`) is authoritative for ordering.

### F08 — `cb_model::Model` carries the trained categorical width

- **Specs:** SPEC-CATF-Δ4 (resolves CRITICAL-3); precondition of SPEC-CATF-10
- **Blocked by:** **F06** (not F09 — the previous `F09->F08` edge was inverted).
  **Blocks:** **F09**, and transitively F10.
- **Parallelizable:** **NO** — owns `crates/cb-model/src/model.rs`.

**Goal / observable completion condition.** `cb_model::Model` gains an explicit
`cat_feature_count: usize` (default `0`, serde-default-preserving) set from the
pool's **DECLARED** cat width at fit time, so no consumer ever derives a width from
the splits the model happened to choose. In the SAME task `Model` becomes
`#[non_exhaustive]` and gains a constructor + builder surface, so that this field —
and every future `Model` field — costs external crates **nothing**.

**LOCKED DECISION (do not re-litigate): `#[non_exhaustive]` + a constructor.**
`crates/cb-model/src/model.rs:305` is `#[derive(Debug, Clone, PartialEq)] pub
struct Model`, with **no `Default`** and **no `#[non_exhaustive]`**, and a
workspace-wide search for `..Default::default()` inside a `Model {` literal returns
**0** `[VERIFIED: LOCAL grep]`. Every construction is therefore an exhaustive struct
literal and would gain `error[E0063]: missing field cat_feature_count`. The recorded
decision is to pay a **one-time** migration now and permanently prevent external
breakage for every future `Model` field: `#[non_exhaustive]` makes struct-literal
syntax illegal from any *other* crate, so from F08 onward a new `Model` field is a
`cb-model`-internal change only.

**Verified blast radius** (re-derive with the greps below; do NOT trust a
hard-coded line list — line numbers drift):
- **INSIDE the `cb-model` crate** (`crates/cb-model/src/**`, including the
  `mod`-mounted unit-test files such as `cbm_test.rs`, `model_test.rs`,
  `partial_dependence_test.rs`, …): `#[non_exhaustive]` does **NOT** affect
  intra-crate literals. These need **only the new field added** —
  `cat_feature_count: 0` — as a mechanical edit. Roughly **37** literals plus the
  production sites `json.rs`, `cbm.rs` (×2), `model_sum.rs`, `model.rs`
  (`from_trained`).
- **OUTSIDE the `cb-model` crate**: `#[non_exhaustive]` **forbids struct-literal
  syntax entirely**, so these MUST migrate to the constructor/builder form.
  Two groups, and **both** are external crates:
  1. `crates/catboost-rs` — **4** literals: `src/model_sum_test.rs:16`,
     `src/onnx_test.rs:32`, `src/model_device_test.rs:25`, `:68`
     `[VERIFIED: LOCAL grep -rnE '(^|[^a-zA-Z_:])Model \{' crates/catboost-rs]`;
     and `crates/cb-train` — **1** literal:
     `crates/cb-train/tests/ctr_split_scoring_test.rs:518`
     `[VERIFIED: LOCAL]`. (PLAN-CHECK P4-4 says "2 in `cb-train`"; a fresh
     `grep -rnE '(^|[^a-zA-Z_:])(cb_model::Model|CbModel) *\{' crates/cb-train`
     finds exactly **one**. **The grep at execution time is authoritative.**)
  2. **`crates/cb-model/tests/*.rs` — integration tests are SEPARATE CRATES**, so
     `#[non_exhaustive]` applies to them exactly as it does to `catboost-rs` and
     `cb-train`. This group was NOT in PLAN-CHECK P4-4's count and is the largest
     part of the migration.
     **No per-file breakdown is given here on purpose.** Five successive attempts
     to enumerate this group by grep produced five different answers (36/12 files,
     18/11, 18/9, 20/10, …), because the counts move with the regex — see the
     warning below. The compiler produces the authoritative list; a plan-time
     tally would only invite an executor to trust it.

**Expected magnitude — NOT a work list.** External migration set is roughly
**20–25** sites: the bulk in `crates/cb-model/tests`, **4 in `crates/catboost-rs`**
(`src/model_sum_test.rs:16`, `src/onnx_test.rs:32`, `src/model_device_test.rs:25`,
`:68` — all written `cb_model::Model {`, and twice mis-recorded as ZERO), and **1**
in `crates/cb-train` (`tests/ctr_split_scoring_test.rs:518`). Intra-crate, roughly
**20** sites in `cb-model/src`, including 4 written `crate::Model {` in
`cbm_test.rs` that a `:`-excluding regex cannot see.

Treat these as an order-of-magnitude sanity check on the compiler's output. If the
compiler names far fewer or far more, STOP AND REPORT — that gap is the signal, not
a number to reconcile by hand.

> **⚠️ THIS COUNT HAS BEEN GOT WRONG FOUR TIMES, EACH BY A DIFFERENT GREP.**
> Three distinct types are named `Model` — `cb_model::Model`,
> `catboost_rs::Model` (`crates/catboost-rs/src/model.rs:64`) and
> `cb_train::Model` (`crates/cb-train/src/boosting.rs:905`) — plus
> `cm::Model` in `crates/cb-model/src/generated/coreml_generated.rs:25`. A regex
> anchored on `Model {` conflates all four and also matches `-> Model {` return
> types; a regex excluding `:` (e.g. `[^a-zA-Z_:]`) silently misses every
> `cb_model::Model {` and `crate::Model {` construction, which is how
> `crates/catboost-rs` was twice recorded as ZERO when it has four.
>
> **DO NOT trust any grep here, including the ones below. THE COMPILER IS THE
> AUTHORITY.** See the locating procedure in the two bullets that follow.

**No ZERO-DIFF oracle is affected.**
`crates/cb-model/tests/fstr_ctr_oracle_test.rs` — one of the eleven SPEC-CTRT-18
targets and **ZERO DIFF REQUIRED** — contains **no** `cb_model::Model` struct
literal at all; its only `Model` occurrence is `fn loaded_model() -> CbModel {` at
`:84`, a return type, and the model itself is loaded rather than constructed
`[VERIFIED: LOCAL grep -nE 'Model *\{|Model::' crates/cb-model/tests/fstr_ctr_oracle_test.rs
→ one hit, line 84]`. An earlier revision of this task claimed one literal here and
raised a STOP-AND-REPORT open item on that basis; **that count was wrong (it
matched the `-> CbModel {` return type) and the open item is withdrawn.**
`ctr_data_roundtrip_test.rs`, the other cb-model oracle in the eleven, likewise
contains no `Model` literal.

`cbm_oracle_test.rs` (4) and `json_oracle_test.rs` (3) DO carry literals, but
neither is among the eleven SPEC-CTRT-18 targets. They are float-only oracles the
**one-hot** plan's SPEC-OH-31 lists as "must be edited by no task" — that plan has
shipped, and this is a mechanical, assertion-preserving constructor migration, so
record the diff in F08's completion evidence and state plainly that no assertion
changed.

**Why this and not the derived bound.** PLAN-CHECK CRITICAL-3, re-verified:
`max(projection member) + 1` equals the true training width **only if** the
highest-indexed cat column is both CTR-eligible **and** chosen by some split. Two
reachable counterexamples at default parameters: (a) a trailing cardinality-2
column routes to `EncodingPath::OneHot` and never enters `eligible_absolute`
(`route_categorical(2, 2) == OneHot` `[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:92-104]`);
(b) an uninformative column is never selected. In both, `fit(pool) → predict(same
pool)` would be rejected. **A derived width is a data-dependent lower bound and
must not be enforced as equality.**

**Files**
- Modify: `crates/cb-model/src/model.rs` — add `#[non_exhaustive]` to `pub struct
  Model`, add the `cat_feature_count: usize` field, add the constructor + builder
  surface (Green step 2), and set the field in `from_trained` (`:359`).
- Modify: `crates/cb-model/src/model_test.rs` (exists)
- Modify: `crates/cb-model/src/{cbm,json}.rs` — **REQUIRED, mechanical.** These
  hold **compile-forced production literals** (`json.rs:825`, `cbm.rs:1193`,
  `cbm.rs:1318`) which cannot compile without the new field; the same applies to
  `crates/cb-model/src/model_sum.rs:120`. Add `cat_feature_count: 0` at each (the
  decoders have no cat width to report). **The earlier prohibition "modify
  `{cbm,json}.rs` only if the field must survive serialization — it must NOT" was
  SELF-CONTRADICTORY and is DELETED.** The accurate constraint is: do **not** touch
  the FlatBuffers schema, the `.cbm` encoder's **byte output**, or `json.rs`'s
  **serde shape** — `cat_feature_count` is **runtime-only** and is neither written
  nor read by either codec, so E00's / `float_only_byte_identity`'s frozen
  baselines stay valid (test fn 2 proves it).
- Modify: **every remaining `cb_model::Model` struct literal inside the `cb-model`
  crate** — **mechanical, forced by the new non-`Default` field; CHANGE NO
  ASSERTION.** Add `cat_feature_count: 0`.
  **Locating procedure (authoritative):** add the field, run
  `cargo build -p cb-model --all-targets`, and fix exactly the sites the compiler
  names with `error[E0063]: missing field cat_feature_count`. Repeat until clean.
  Do NOT work from a hard-coded line list and do NOT trust a grep — a grep
  excluding `:` misses the 4 `crate::Model {` literals in `cbm_test.rs`
  (`:322`, `:879`, `:900`, `:1030`). Expect **21** sites.
- Modify: **every `cb_model::Model` struct literal OUTSIDE the `cb-model` crate** —
  **mechanical, forced by `#[non_exhaustive]`; CHANGE NO ASSERTION.** These must be
  **migrated to constructor/builder form** (struct-literal syntax is no longer
  legal for them).
  **Locating procedure (authoritative):** add `#[non_exhaustive]`, run
  `cargo build --workspace --all-targets`, and migrate exactly the sites the
  compiler names with `error[E0639]` (cannot create a non-exhaustive struct using
  a struct literal). Repeat until clean. Expect **23** sites. A grep may be used
  as a pre-flight estimate ONLY, never as the work list — see the warning above.
  The estimate below was enumerated at plan time and is the expected shape, not
  the source of truth:
  - `crates/catboost-rs/src/model_sum_test.rs:16`,
    `crates/catboost-rs/src/onnx_test.rs:32`,
    `crates/catboost-rs/src/model_device_test.rs:25`, `:68` — **4**;
  - `crates/cb-train/tests/ctr_split_scoring_test.rs:518` — **1**. This file is
    **one of the eleven SPEC-CTRT-18 oracle targets**; it sits in the
    **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES** row of the per-file diff gate
    (PLAN.md §3.2 and the tables in E15/E16 Completion evidence), with **F08 named
    as an owning task**. **Weakening or deleting any assertion in it is FORBIDDEN.**
  - `crates/cb-model/tests/*.rs` — **18** across 11 files (see the blast-radius
    block above). Two of these — `cbm_oracle_test.rs` and `json_oracle_test.rs` —
    are float-only oracles the SHIPPED one-hot plan's SPEC-OH-31 listed as
    "edited by no task"; the migration is mechanical and assertion-preserving, so
    record the diff and state plainly that no assertion changed. **No ZERO-DIFF
    oracle of THIS plan is affected** — `fstr_ctr_oracle_test.rs` and
    `ctr_data_roundtrip_test.rs` contain no `cb_model::Model` struct literal.
- **NOT** `crates/catboost-rs/src/builder.rs` — the call site belongs to **F09**,
  which lands after F08. F08 ships the field, the builder-style setter and the
  getter only; nothing calls them yet, which is exactly why F08 can precede F09.

**Exact verified files/symbols to touch**
- `cb_model::Model` fields: `oblivious_trees`, `non_symmetric_trees`,
  `region_trees`, `bias`, `float_feature_borders`, `ctr_data`, `approx_dimension`,
  `class_to_label` — **no cat count** (Part-2 PLAN finding **F3**, re-confirmed:
  the canonical model has no such field) `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs]`.
- `Model::with_ctr_data(mut self, ctr_data: CtrData) -> Self` at
  **`crates/cb-model/src/model.rs:530`** is the builder-style precedent to mirror
  exactly for `with_cat_feature_count`
  `[VERIFIED: LOCAL crates/cb-model/src/model.rs:527-533, read verbatim —
  `#[must_use]`, `mut self`, one field assignment, `self`]`.
- `Model::from_trained(trained: &cb_train::Model, float_feature_borders:
  Vec<Vec<f64>>) -> Self` at **`crates/cb-model/src/model.rs:359`** is the
  production constructor `[VERIFIED: LOCAL, read verbatim]`; it must set
  `cat_feature_count: 0` (it has no pool, so the DECLARED width arrives later via
  the builder — F09's job).
- **`from_trained` alone cannot replace the external literals**: they build models
  `from_trained` cannot express (hand-written `ObliviousTree`s, an explicit
  `ctr_data`, `approx_dimension != 1`, a non-empty `class_to_label`). The migration
  therefore needs a **base constructor plus per-field builders** — Green step 2.
- `Pool::n_cat_features()` `[VERIFIED: Part-2 PLAN T08/T09 anchors, cb-data/src/pool.rs]`.

**Red**
- File: `crates/cb-model/src/model_test.rs`
- Test fn 1: `with_cat_feature_count_is_readable_and_defaults_to_zero`
  Expected: a freshly built `Model` reports `0`; after
  `.with_cat_feature_count(3)` it reports `3`.
- Test fn 2 (**the byte-identity guard**):
  `adding_the_cat_feature_count_does_not_change_cbm_bytes`
  Expected: `save_cbm` on a model with `cat_feature_count = 7` produces bytes
  **identical** to the same model with `cat_feature_count = 0`. This is what keeps
  E00's and `float_only_byte_identity`'s frozen baselines valid.
- Test fn 3 (**the external-construction guard — it must live in an INTEGRATION
  target so it compiles as a separate crate**, e.g.
  `crates/cb-model/tests/model_constructor_test.rs`):
  `an_external_crate_can_build_every_model_shape_without_struct_literal_syntax`
  Expected: the constructor + builders reproduce, by `assert_eq!` against a model
  built inside `cb-model`, each shape the migrated sites need — trees + bias +
  borders, `.with_ctr_data(..)`, `.with_approx_dimension(..)`,
  `.with_class_to_label(..)`, `.with_cat_feature_count(..)`. Without this test the
  `#[non_exhaustive]` decision is unverified from the outside.
- **EXPECTED INITIAL FAILURE:**
  1. `error[E0599]: no method named 'with_cat_feature_count'` (test fn 1).
  2. `error[E0063]: missing field cat_feature_count in initializer of Model` —
     emitted **once per intra-crate struct literal** the moment the field is added
     (~37 in `crates/cb-model/src/**`, including the production sites `json.rs:825`,
     `cbm.rs:1193`, `cbm.rs:1318`, `model_sum.rs:120`). `cargo test -p cb-model`
     does not build until every one carries `cat_feature_count: 0`.
  3. `error[E0639]: cannot create non-exhaustive struct using functional record
     update syntax / struct expression` (the `#[non_exhaustive]` diagnostic) —
     emitted **once per EXTERNAL struct literal**: 36 in
     `crates/cb-model/tests/*.rs`, 4 in `crates/catboost-rs`, 1 in
     `crates/cb-train/tests/ctr_split_scoring_test.rs`. Those targets do not build
     until each is migrated to the constructor/builder form.
- Run: `cargo test -p cb-model --lib model_test -- cat_feature_count`

**Green.**
1. Add `#[non_exhaustive]` to `pub struct Model` (`crates/cb-model/src/model.rs:305`,
   keeping `#[derive(Debug, Clone, PartialEq)]`) and the field
   `pub cat_feature_count: usize` (logical default `0`).
2. Add the **constructor + builder surface**, every method `#[must_use]` and shaped
   exactly like `with_ctr_data` (`:530`) — `mut self`, one assignment, `self`:
   - `pub fn new(oblivious_trees: Vec<ObliviousTree>, bias: f64,
     float_feature_borders: Vec<Vec<f64>>) -> Self` — every other field at its
     zero/empty value (`approx_dimension: 1`, `cat_feature_count: 0`,
     `ctr_data: None`, the two non-oblivious tree vectors empty);
   - `with_cat_feature_count(mut self, n: usize) -> Self` (the field this task
     exists for) and `pub fn cat_feature_count(&self) -> usize`;
   - the builders the migration needs so external crates can still express every
     shape: `with_non_symmetric_trees`, `with_region_trees`,
     `with_approx_dimension`, `with_class_to_label` (`with_ctr_data` already
     exists). Add **only** what a migrated site actually sets — no speculative API.
3. Set `cat_feature_count: 0` in `from_trained` (`:359`) and in the four production
   literals (`json.rs:825`, `cbm.rs:1193`, `cbm.rs:1318`, `model_sum.rs:120`).
4. Add `cat_feature_count: 0` to every remaining **intra-crate** literal —
   mechanical, **CHANGE NO ASSERTION**.
5. Migrate every **external** literal to `Model::new(..)` + builders — mechanical,
   **CHANGE NO ASSERTION**; the resulting `Model` must be field-for-field identical
   to the literal it replaces.
**Do not** touch the FlatBuffers schema, the `.cbm` encoder's **byte output**, or
`json.rs`'s **serde shape**: `cat_feature_count` is runtime-only, neither written
nor read by either codec.

**Rationale, recorded so it is not re-litigated.** `#[non_exhaustive]` costs a
one-time migration of the external literals **now** (5 outside `cb-model` + 36 in
`cb-model`'s own integration targets) and **permanently prevents external breakage
for every future `Model` field** — this exact defect class (a shared struct grows a
field while consumers sit in no task's Files list) has now been found four times in
this plan. Without it, every later field repeats the whole blast radius.

**Refactor constraints + required regression scope**
- Constraint: **SPEC-CATF §7 lists `cb-model` as verification-only.** This task is
  an explicit, recorded amendment to that section (PLAN-CHECK revision #3,
  option 1). State the amendment in `../catboost-builder-cat-features-routing/SPEC.md`
  §7 in the same change.
- Constraint: **the `#[non_exhaustive]` migration is MECHANICAL in every file it
  touches.** No assertion may be added, removed, weakened or reworded anywhere —
  in particular not in `cbm_oracle_test.rs`, `json_oracle_test.rs`,
  `fstr_ctr_oracle_test.rs` or `crates/cb-train/tests/ctr_split_scoring_test.rs`,
  all of which carry frozen assertions gated elsewhere in this plan. **A migrated
  site must produce a `Model` field-for-field identical to the literal it replaced.**
- Regression scope: `cargo test -p cb-model`, especially
  `ctr_nonmean_byte_identity_test` (E00), `float_only_byte_identity_test`,
  `cbm_oracle_test`, `json_oracle_test`, `model_sum_test`; **plus
  `cargo test -p catboost-rs` and `cargo test -p cb-train --test
  ctr_split_scoring_test`, the two other crates the migration reaches.**

**Validation**
```bash
cargo test -p cb-model --lib model_test
cargo test -p cb-model --test model_constructor_test
cargo test -p cb-model --test ctr_nonmean_byte_identity_test \
  --test float_only_byte_identity_test --test cbm_oracle_test --test json_oracle_test
cargo test -p cb-model -p catboost-rs
cargo test -p cb-train --test ctr_split_scoring_test --test fstr_ctr_oracle_test
cargo clippy -p cb-model -p catboost-rs --all-targets
```

**Completion evidence.** All three tests green — in particular the `.cbm`
byte-identity guard, which proves the field is runtime-only, and the external
construction guard, which proves `#[non_exhaustive]` left no shape unexpressible.
**Migration accounting:** `git diff --stat` shows the intra-crate literals gaining
exactly one `cat_feature_count: 0` line each, and every external literal replaced
by a `Model::new(..)` + builder chain — **no assertion touched in any file**.
`grep -rnE '(^|[^a-zA-Z_:>-])Model \{' crates/cb-train crates/cb-model/tests`
(excluding `-> Model {` return types) returns **zero** `cb_model::Model` struct
literals afterwards — all 19 have moved to constructor form. No ZERO-DIFF oracle of
this plan is touched.

---

### F10–F13 — Predict-side routing and the remaining silent-wrongness paths

All four own `crates/catboost-rs/src/model.rs` and are **STRICTLY SERIAL**.

| Task | Source | Specs | Blocked by | Blocks | Delta |
|---|---|---|---|---|---|
| **F10** `Model::cat_columns()` width validation | T09 | SPEC-CATF-10, Δ4 | F09 | F11 | **REWRITTEN.** Delete T09's `expected_cat_features()` derivation entirely. `cat_columns()` now compares `pool.n_cat_features()` against `self.inner.cat_feature_count()`. **Rename the predicate** `is_ctr_model()` → **`needs_cat_columns()`** = `ctr_data.is_some() || any tree carries a ModelSplit::OneHot` (Δ7: a one-hot-only model has **no** `ctr_data`, but `predict_raw = predict_raw_cat(m, fv, &[])` makes `cat_values.get(i) → None → false`, so **every one-hot split fails** — a silent-wrongness class the original plan did not know about). **ADD a positive regression test**: `fit(pool with 2 cat columns, one one-hot-routed) → predict(the SAME pool)` must be `Ok` — the case T09's equality-on-a-derived-bound would have rejected. |
| **F11** `predict_with` is CTR-aware **and one-hot-aware** | T10 | SPEC-CATF-11, Δ7 | F10 | F12, F14, F17 | **WIDENED.** T10's byte-identity argument (`predict_raw(m, fv) == predict_raw_cat(m, fv, &[])`, re-verified at `crates/cb-model/src/apply.rs:393-395` `[VERIFIED: CODEGRAPH]`) still holds. Route through `predict_raw_cat` using `needs_cat_columns()`. **EXTEND `ensure_scalar_oblivious` (`crates/catboost-rs/src/model.rs:144-171`) with a `needs_cat_columns()` ONE-HOT arm returning `CatBoostError::UnsupportedModel`.** The earlier "preserve, do not extend" instruction is **RETRACTED where it blocks this**: that function today rejects `approx_dimension > 1`, non-symmetric trees, region trees and `ctr_data.is_some()`, but **NOT `ModelSplit::OneHot`** `[VERIFIED: CODEGRAPH — 1 caller, `staged_predict` at `:196`]`, and `predict_raw_staged` (`crates/cb-model/src/apply.rs:488-526`) is **float-only** and never reads cat columns. So after F09 a one-hot model — which has `ctr_data == None` — passes the guard and is scored as though **every one-hot split failed**: a NEW silent-wrongness surface on a documented public API, exactly the class F10/F11 close for `predict_raw`. The pre-existing CTR rejection is preserved unchanged; only the one-hot arm is added. |
| **F12** zero-cat-column guard on all **four** entrypoints (`predict`, `predict_with`, `predict_proba`, **`staged_predict`**) | T11 | SPEC-CATF-12, Δ7 | F11 | F20 | **WIDENED** to the `needs_cat_columns()` predicate, **and widened again to cover `staged_predict`** (MAJOR-7 residue). **Red — ADD a one-hot case to `crates/catboost-rs/tests/staged_predict_facade_test.rs`** (already in F10–F13's shared regression scope): fit a pool whose cat column has cardinality 2 at `one_hot_max_size(2)`, call `model.staged_predict(&pool, …)`, and expect `Err(CatBoostError::UnsupportedModel)`; the `Ok(_)` arm must `panic!("staged_predict SILENTLY SCORED a one-hot model as though every one-hot split failed")`. **Mutation check (§3.1), RECORDED:** temporarily make `needs_cat_columns()` return `ctr_data.is_some()` only, re-run, and assert BOTH the `predict` arm's `Ok(v) => panic!("SILENTLY SCORED …")` and the new `staged_predict` arm's panic fire; record both failure texts; revert **manually** (never `git checkout --`). |
| **F13** typed rejection for the remaining fstr/PDP paths | **NEW** (MAJOR-7 residue) | SPEC-CATF-Δ7 | F12 | — | `shap_values` is **already fixed** by the uncommitted one-hot wave (`crates/cb-model/src/shap.rs:552-608` `float_splits_of` returns `Err(ShapUnsupported::OneHotSplits / ::CtrSplits)`; `crates/catboost-rs/src/model.rs` propagates with `?`) `[VERIFIED: LOCAL git diff + grep]`. **Remaining:** `Model::partial_dependence` (`crates/catboost-rs/src/model.rs:427-434`) passes only `feature_columns` `[VERIFIED: CODEGRAPH]`, and `Model::feature_importance_with_data(PredictionValuesChange)` (`:365-382`) passes `pool.cat_features().to_vec()` with **no width check** `[VERIFIED: CODEGRAPH]`. Add `self.cat_columns(pool)?` to the latter and a typed `UnsupportedModel` rejection to the former when `needs_cat_columns()`. **Verified already safe (do not touch):** `Model::predict_raw_on_device` guards via `cb_model::flatten_oblivious_f64`, which rejects categorical/CTR models. |

**Shared Red shape.** New file
`crates/catboost-rs/tests/model_cat_predict_test.rs`; commands use
`cargo test -p catboost-rs --test model_cat_predict_test -- <filter>`
(**MAJOR-8 fix** — never `--test <target>::<filter>`).

**Shared regression scope for F10–F13**
```bash
cargo test -p catboost-rs --test model_cat_predict_test
cargo test -p catboost-rs --test staged_predict_facade_test --test onnx_facade_test \
                          --test coreml_facade_test
cargo test -p catboost-rs --test builder_oracle_test --test cv_oracle_test \
                          --test grid_search_oracle_test
cargo test -p catboost-rs -p cb-model
cargo clippy -p catboost-rs --all-targets
```

**Completion evidence.** F10's positive `fit → predict same pool` test green (the
CRITICAL-3 counterexample); F11's facade-vs-direct `predict_raw_cat` identity at
1e-12 with a non-degeneracy guard, **plus `ensure_scalar_oblivious`'s new one-hot
arm**; F12's **four**-entrypoint typed errors — including the one-hot
`staged_predict` case in `crates/catboost-rs/tests/staged_predict_facade_test.rs` —
plus the recorded mutation failure texts for **both** the `predict` and the
`staged_predict` arms; F13's two typed rejections with tests, and an explicit
note that `shap_values` needed no work.

---

## WAVE W8 — `cv()` / `grid_search()` on a categorical pool

### F14 — Defined behavior for `cv` and `grid_search` with categorical pools

- **Specs:** SPEC-CATF-Δ6 (resolves MAJOR-6); amends SPEC-CATF §8
- **Blocked by:** F11. **Blocks:** F20.
- **Parallelizable:** **YES** with W9 — owns `crates/catboost-rs/src/{cv,grid_search}.rs`
  and their `_test.rs` siblings; no overlap with `params.rs` or the estimator files.

**Goal / observable completion condition.** `cv(categorical_pool, …)` and
`grid_search(categorical_pool, …)` have **defined, tested** behavior — never a
silent all-NaN `SearchResult`.

**The verified hazard.** `crates/catboost-rs/src/cv.rs:398-409` does
`pool.select_rows(...)` → `builder.fit(&train_pool)?` → `model.staged_predict(&test_pool, ...)?`;
`Pool::select_rows` **preserves** `cat_features` (`crates/cb-data/src/pool.rs:232-236`);
and `ensure_scalar_oblivious` (`crates/catboost-rs/src/model.rs:144-171`) rejects
`ctr_data.is_some()` with `UnsupportedModel`. So after F09, `cv` on a categorical
pool flips from "returns curves, ignoring the cat column" to `UnsupportedModel`.
Worse, in `grid_search` with `ErrorScore::Value(nan)` (the sklearn default),
**every** candidate fails, `warn_fit_failed` emits a warning, and a `SearchResult`
with all-NaN scores and an arbitrary `best_index` is returned — **a silent
degradation, not an error** `[VERIFIED: PLAN-CHECK MAJOR-6, re-derived]`.
`crates/catboost-rs-py/tests/test_cv.py:114-135` is written tolerantly
(`try/except CatBoostError: return`) and will pass either way — it is **not** a gate.

**Decision this plan takes (record it; do not leave ambiguous).**
**Fail fast, at the top, with a typed error.** A categorical pool is a
*configuration* failure, not a *candidate* failure, so converting it into
`error_score` is category-confusion. `cv` and `grid_search` both return
`CatBoostError::UnsupportedModel` naming the count of categorical columns and
pointing at `staged_predict`'s CTR limitation, **before** any fold is fitted.

**Files**
- Modify: `crates/catboost-rs/src/cv.rs`, `crates/catboost-rs/src/grid_search.rs`
- Modify: `crates/catboost-rs/src/cv_test.rs`, `crates/catboost-rs/src/grid_search_test.rs`
- Modify: `../catboost-builder-cat-features-routing/SPEC.md` §8 (add the second
  deliberate behavior change — §8 currently claims exactly one)

**Red**
- Test fn 1 (`cv_test.rs`): `cv_on_a_categorical_pool_is_a_typed_unsupported_model`
  Expected: `Err(CatBoostError::UnsupportedModel(msg))` with
  `msg.contains("categorical")`; **and** an assertion that it fails **before**
  fitting (e.g. an `iterations(10_000)` builder still returns in well under a
  second, or a fold-counter side channel remains 0).
- Test fn 2 (`grid_search_test.rs`):
  `grid_search_on_a_categorical_pool_fails_fast_not_as_all_nan_scores`
  Expected: `Err(..)`, and explicitly **NOT** `Ok(SearchResult)` whose scores are
  all NaN. The `Ok(_)` arm's panic message must read
  `"grid_search SILENTLY DEGRADED a categorical pool to an all-NaN SearchResult"`.
- Test fn 3: `cv_on_a_float_only_pool_is_unchanged` — the D-04 no-op proof.
- **EXPECTED INITIAL FAILURE:** test fn 1 returns `Ok(curves)` today (the float
  path silently ignores cat columns); after F09 lands it returns
  `Err(UnsupportedModel)` **from inside the fold**, so the "fails before fitting"
  assertion is what stays red until the Green.
- Run: `cargo test -p catboost-rs --lib cv::cv_test -- categorical` and
  `cargo test -p catboost-rs --lib grid_search::grid_search_test -- categorical`

**Green.** A single guard at the top of `cv()` and of `grid_search()`'s entry:
`if pool.n_cat_features() > 0 { return Err(CatBoostError::UnsupportedModel(format!(
"cv/grid_search does not support categorical pools ({n} categorical column(s)): \
cross-validation scores via staged_predict, which rejects CTR models \
(crates/catboost-rs/src/model.rs ensure_scalar_oblivious). Fit directly with \
CatBoostBuilder::fit and score with predict instead."))) }`

**Refactor constraints + required regression scope**
- Constraint: do NOT **weaken** `ensure_scalar_oblivious` — SPEC-CATF §2 requires
  its existing CTR rejection be preserved. **Note:** F11 deliberately **extends** it
  with a `needs_cat_columns()` one-hot arm (Δ7 / MAJOR-7); extending is required,
  weakening is forbidden. F14's guard runs before any fold either way.
- Constraint: record the change in `SPEC.md` §8 as a **second** deliberate,
  observable behavior change.
- Regression scope: `cv_oracle_test`, `grid_search_oracle_test` (unmodified),
  `crates/catboost-rs-py/tests/test_cv.py`.

**Validation**
```bash
cargo test -p catboost-rs --lib cv::cv_test --lib grid_search::grid_search_test
cargo test -p catboost-rs --test cv_oracle_test --test grid_search_oracle_test
cargo test -p catboost-rs
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_cv.py -q
```

**Completion evidence.** Three tests green, including the explicit
"SILENTLY DEGRADED" anti-NaN assertion; the two float-only oracle targets green
with zero diff to their files; the `SPEC.md` §8 amendment present.

---

## WAVE W9 — Python bindings

| Task | Source | Specs | Blocked by | Blocks | Parallel | Delta |
|---|---|---|---|---|---|---|
| **F15** Python `one_hot_max_size` / `max_ctr_complexity` / `counter_calc_method` | T12 | SPEC-CATF-13 (scalar half), Δ2 | F05, F07 | F16, F18 | NO with F16 (both own `params.rs`); **YES** with F17 | **REVISED:** the ranges are now VERIFIED, not assumed (Part-2 assumption **A4** is closed): `one_hot_max_size` default **2**, upper bound `OneHotMaxSizeLimit = GetMaxBinCount()` on CPU (`cat_feature_options.cpp:232-233,267-268`); `max_ctr_complexity` default **4**, `CB_ENSURE(MaxTensorComplexity < GetMaxTreeDepth())` (`:231,269-271`) `[VERIFIED: research §G.1]`. `counter_calc_method` doc must carry F05's eval-set-only honesty note. |
| **F16** Python `simple_ctr` / `combinations_ctr` CTR-description grammar | T13 | SPEC-CATF-13 (grammar half), Δ2 | F00, F15 | F18 | NO (owns `params.rs`) | **REVISED MATERIALLY.** The accepted domain is now **exactly the four CPU-legal types** `{Borders, Buckets, BinarizedTargetMeanValue, Counter}`; `FloatTargetMeanValue` and `FeatureFreq` must be **REJECTED** with upstream's own wording (`catboost_options.cpp:504-509`, mirroring E02's engine-side guard) — **T13's test expecting `"FeatureFreq"` in the accepted-variants list is therefore WRONG and must be inverted**. The single-description limit and the `Prior=<n>/<d>` (`d != 1`) rejection remain, the latter now citing `ctr_helper.cpp:50` **and** the engine's `prior_denom: 1.0` pin at `crates/cb-train/src/boosting.rs:1942`. **`combinations_ctr=[]` — THE MAPPING IS MANDATED HERE, NOT LEFT TO THE IMPLEMENTER.** The engine field is a **scalar `ECtrType`** with **no "disabled" representation**; `max_ctr_complexity = 1` is the **only** in-engine way to suppress combination CTRs. F16 must therefore pick **exactly one** of these two, and record which: **(a)** map `combinations_ctr=[]` to `max_ctr_complexity = 1` (leaving `combinations_ctr` at its default `ECtrType`), documenting that mapping in `params.rs` and asserting it in a test — the committed `crates/cb-oracle/fixtures/plain_ctr/config.json` uses exactly the `[]` form `[VERIFIED: LOCAL]`; **or (b)** reject `combinations_ctr=[]` with a typed `CatBoostParameterError` whose message says the engine has no disabled representation and directs the user to `max_ctr_complexity=1`. **Leaving it undefined is forbidden** — a silently-ignored kwarg is precisely what BLOCKER-2 was about, and re-introducing one here would undo P2.0 row 2. If (a) is chosen, the test must assert that `combinations_ctr=[]` **actually changes** `max_ctr_complexity` in the resulting `BoostParams`, not merely that `fit()` does not raise. |
| **F17** `cat_features` as a `fit()` kwarg | T14 | SPEC-CATF-14, Δ5 | F09, F11 | F18 | **YES** with F15/F16 (disjoint files) | **Δ5 FIX — all 14 `data_to_pool` call sites, enumerated and re-verified this session** `[VERIFIED: LOCAL grep -rn "data_to_pool(" crates/catboost-rs-py/src/]`: `classifier.rs:68,100,126,280`; `ranker.rs:65,100`; `regressor.rs:68,149,254`; `estimator.rs:270`; **`cv.rs:61,64`**; **`search.rs:304,389`**. T14's file list omits `cv.rs` and `search.rs` — **add both**; all four pass `None`. **MINOR-11 FIX:** de-duplicate and range-check `cat_features` before the post-ingestion width guard (`cat_features=[2,2]` currently mis-reports "declared 2 … carries 1"). **MINOR-10:** ship `Optional[list[int]]` and amend SPEC-CATF-14's Input line; record name-based selection as a follow-up. **OQ-3 (upstream-exact):** `fit(Pool, cat_features=…)` raises, per `core.py:1522-1533`. **Finding F2:** the NumPy branch of `ingest_to_owned` silently drops `cat_features`; the guard lives in `estimator.rs::data_to_pool`, NOT in `ingest_py.rs` (SPEC §7 verification-only). |
| **F18** registry truthfulness + the vendored `core.py` gap | T15 | SPEC-CATF-15 | F15, F16, F17 | F20 | NO | **SURVIVES.** T15's finding is re-confirmed: `test_params.py`'s `_CORE_PY` points at `catboost-master/catboost/python-package/catboost/core.py`, which **does not exist** in this checkout (the vendored tree is a 3-file stub), so `test_every_upstream_param_is_in_registry` currently **SKIPS**. Fall back to the installed `catboost==1.2.10` package's `core.py`. **A skipped test is not a gate.** Validation MUST report `0 skipped`. |

**Shared Red/Validation shape for W9**
```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_params.py -q -rs   # MUST show 0 skipped after F18
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_categorical.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/test_ingestion.py \
                           crates/catboost-rs-py/tests/test_check_estimator.py -q
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
cargo test -p catboost-rs-py
cargo clippy -p catboost-rs-py --all-targets
```

**F16's inverted Red (the one materially new assertion):**
```python
def test_cpu_illegal_ctr_types_are_rejected_by_name():
    for bad in ("FloatTargetMeanValue", "FeatureFreq"):
        with pytest.raises(CatBoostParameterError) as e:
            CatBoostRegressor(simple_ctr=bad).fit(*_toy_xy())
        msg = str(e.value)
        assert bad in msg
        assert "not implemented on CPU" in msg
        for ok in ("Borders", "Buckets", "BinarizedTargetMeanValue", "Counter"):
            assert ok in msg
```
**EXPECTED INITIAL FAILURE:** the raised error is the parity-gap message
(`"simple_ctr" is a known CatBoost parameter not yet implemented … (parity gap)`),
so `assert bad in msg` fails.

**Verified starting state.** `IMPLEMENTED` is a **14-entry** `const &[&str]` at
`crates/catboost-rs-py/src/params.rs:42-57`
(`iterations, learning_rate, depth, l2_leaf_reg, loss_function, border_count,
random_seed, random_strength, bagging_temperature, bootstrap_type, subsample,
score_function, boost_from_average, leaf_estimation_method`)
`[VERIFIED: LOCAL, read verbatim]`. The six promoted names —
`cat_features`, `one_hot_max_size`, `max_ctr_complexity`, `simple_ctr`,
`combinations_ctr`, `counter_calc_method` — are in `VOCABULARY` but **not** in
`IMPLEMENTED`, so `validate_params` rejects each as a parity gap.

**Completion evidence for W9.** `_param_status` returns `"IMPLEMENTED"` for all six;
`nan_mode` stays `KNOWN_NOT_YET` and stays rejected at `fit()`;
`test_params.py` reports **0 skipped**; F16's inverted rejection test green;
F17's four required extra tests green (Pool-collision, NumPy-drop, duplicate-index,
all-three-estimators).

---

## WAVE W10 — Gates

| Task | Source | Specs | Blocked by | Blocks | Delta |
|---|---|---|---|---|---|
| **F19** Reuse the committed fixture; run NO corpus generation | T16 **REPLACED** | SPEC-CATF-17, Δ3 | E23 | F20 | **T16 IS DELETED, not fixed.** Per the locked user decision the public-API oracle reuses the already-committed `crates/cb-oracle/fixtures/one_hot_train/{default_binary,multi}/` family — each carrying `X_float.npy`, `y.npy`, `cat_cols.json`, `model.cbm`, `model.json`, `predictions.npy`, `config.json` `[VERIFIED: LOCAL ls]`. F19 is a **read-only audit** task: confirm the artifacts load, confirm `config.json`'s pinned params are all reachable through the Python surface after F15–F17, and confirm `git status --porcelain crates/cb-oracle/fixtures` is clean. **`crates/cb-oracle/generator/gen_fixtures.py` is NEVER invoked.** |
| **F20** Public-Python-API categorical oracle ≤1e-5 | T17 | SPEC-CATF-16; acceptance **A11** | F12, F14, F18, F19 | F21 | **REVISED.** Drive `one_hot_train/default_binary` (3 float + 1 cat column, cardinality 2, `one_hot_max_size=2`, `max_ctr_complexity=0`, `score_function="L2"`, `learning_rate=0.1`, `depth=3`, `iterations=3`) end to end through `catboost_rs.Pool(df, y, cat_features=[3])` → `fit()` → `predict()` and compare against the committed `predictions.npy` at `atol=1e-5, rtol=0`. **Localization ladder step 1 now cites `crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs`** (which already proves `permutation_count=4` + `max_ctr_complexity=2` at ≤1e-5 through the production `predict_raw_cat` path) instead of proposing a scratch Rust run — so the A1/F5/R9 `permutation_count` risk is **already closed** and must be downgraded, not re-litigated. **Non-degeneracy guard mandatory** (`predictions.std() > 1e-6`) plus an assertion that the fitted model carries ≥1 `ModelSplit::OneHot`, so a configuration that degenerates to no categorical split cannot pass. |
| **F21** Float-only `fit()` no-regression | T18 | SPEC-CATF-07; acceptance **A1** | F20 | F22 | **SURVIVES.** Mutation check: invert `fit()`'s branch predicate (`pool.cat_features().is_empty()` → `!…`), expect ``a cat-free pool must never route through train_cat``, revert **manually** (never `git checkout --`). `git diff --stat` over `builder_oracle_test.rs`, `cv_oracle_test.rs`, `grid_search_oracle_test.rs` MUST print nothing. |
| **F22** Bench preflight passes without editing the bench | **NEW** | SPEC-CATF-Δ8; acceptance **A13** | F21 | F23 | `bench/one_hot_gpu_speed/one_hot_bench_colab.py` bails `BLOCKED-FACADE-ROUTING` via `FACADE_ROUTING_MARKER = ("train_cat", "crates/catboost-rs/src/builder.rs")` at **line 117**, checked at **:210-227** with `grep -c '{needle}' {REPO}/{rel}` `[VERIFIED: LOCAL, read verbatim]`. After F09 the marker is present. **Red — STRENGTHENED.** A bare `grep -c "train_cat"` is satisfied by a mere **comment** containing the string, so it does not prove the facade routes. The test MUST assert the marker appears in a **CALL position** — a regex such as `\btrain_cat\s*\(` (or `cb_train::train_cat(` / `train_cat(&`) on a line that is not a `//` comment — **and** it must be paired with F09's `ctr_data.is_some()` behavioral assertion (`crates/catboost-rs/tests/builder_cat_facade_test.rs` test fn 1) as the real proof of routing. It must also assert `bench/one_hot_gpu_speed/one_hot_bench_colab.py` is byte-unchanged (`git diff --stat` prints nothing) — the bench must pass **unedited**. |
| **F23** One-hot wave + workspace baseline gate | **NEW** | acceptance **A12** | F22 | — | Final regression gate. Re-run `crates/cb-train/tests/{one_hot_oracle,one_hot_draw_accounting,device_one_hot_parity}_test.rs`, then `cargo test --workspace --no-fail-fast` and diff the failure set against `.planning/plans/one-hot-categorical-training/baseline/workspace-test-baseline.txt`. **Gate: no target that passes there may fail here.** Known-accepted: the 59 `cb_backend` cpu-backend failures; `exact_quantile_weighted_matches_cpu` is FLAKY (~2/5 runs); `cb-train`'s `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors` is a verified pre-existing failure absent from the (self-documented incomplete) transcript. |

**F20's Red (verbatim shape)**
- File: `crates/catboost-rs-py/tests/test_categorical.py`
- Test fn: `test_public_api_one_hot_oracle_parity`
- **EXPECTED INITIAL FAILURE (before F15–F17):**
  `CatBoostParameterError: parameter 'one_hot_max_size' is a known CatBoost
  parameter not yet implemented in catboost-rs (parity gap)`. After F15–F17 the
  meaningful failure mode is a numeric `Not equal to tolerance atol=1e-05`.
- Localization ladder (STOP AND REPORT at the first hit):
  1. Compare against a direct Rust `train_cat` + `predict_raw_cat` run on the same
     corpus — the already-passing `crates/cb-train/tests/one_hot_oracle_test.rs`
     **is** that comparison; if it passes and the Python path fails, the gap is in
     the facade/binding wiring, which is this plan's scope.
  2. If the Rust-direct path also diverges, the gap is inside the shipped engine —
     **STOP AND REPORT**.
  3. If it passes at `max_ctr_complexity=0` and fails at `>1`, that is the
     ORD-06/ORD-07 combination-CTR gating bug
     (`.planning/phases/24-ctr-split-search-correctness/`) — **STOP AND REPORT. DO
     NOT FIX IT HERE.**

**F22's Red (verbatim shape) — strengthened beyond the bare `grep -c`**
```python
import re

def test_bench_preflight_facade_routing_marker_is_present():
    builder = (REPO / "crates/catboost-rs/src/builder.rs").read_text()

    # 1. The bench's own preflight (one_hot_bench_colab.py:117, checked at :210-227)
    #    only does `grep -c "train_cat"` — which a COMMENT would satisfy. Assert the
    #    marker is in a CALL position on a non-comment line.
    call_sites = [
        ln for ln in builder.splitlines()
        if re.search(r"\btrain_cat\s*\(", ln) and not ln.lstrip().startswith("//")
    ]
    assert call_sites, (
        "`train_cat` must appear as an actual CALL in builder.rs, not merely as a "
        "comment: bench/one_hot_gpu_speed/one_hot_bench_colab.py:117 greps for the "
        "bare string, so a comment would make the bench pass while fit() still "
        "routes float-only"
    )

    # 2. Paired behavioral proof — the marker means nothing without it.
    #    F09 test fn 1 (crates/catboost-rs/tests/builder_cat_facade_test.rs)
    #    asserts `ctr_data.is_some()` after fitting a categorical pool.
    #    Run it as part of this task's validation; a green marker test with a red
    #    F09 test fn 1 is NOT a pass.
```
**EXPECTED INITIAL FAILURE (before F09):** `AssertionError` with that message.
**Validation addition:**
`cargo test -p catboost-rs --test builder_cat_facade_test -- fit_categorical_pool_bakes_ctr_data`

**Validation for W10**
```bash
cargo test -p catboost-rs --test builder_oracle_test --test cv_oracle_test \
                          --test grid_search_oracle_test
git diff --stat crates/catboost-rs/tests/builder_oracle_test.rs \
                crates/catboost-rs/tests/cv_oracle_test.rs \
                crates/catboost-rs/tests/grid_search_oracle_test.rs   # MUST be empty
git diff --stat bench/one_hot_gpu_speed/one_hot_bench_colab.py        # MUST be empty
git status --porcelain crates/cb-oracle/fixtures                      # MUST be clean
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test \
                       --test device_one_hot_parity_test
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
cargo test --workspace --no-fail-fast 2>&1 | tail -60
cargo clippy --workspace --all-targets
```

**Completion evidence for W10.** F20's two oracle assertions green at
`atol=1e-5, rtol=0` with the non-degeneracy and `ModelSplit::OneHot` guards;
F21's recorded mutated-fail/revert plus the three empty `git diff --stat`s;
F22's marker test green **asserting a CALL-position `train_cat(` on a non-comment
line** and paired with F09 test fn 1's `ctr_data.is_some()`, with the bench
byte-unchanged; F23's workspace transcript
diffed against the accepted baseline with an explicit statement that no
previously-passing target regressed.

---

> **Coverage tables, risk register, unresolved blockers and the process
> attestation are in `./PLAN-COVERAGE.md`.**
