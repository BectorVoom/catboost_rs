## Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Make four inert CTR params (`simple_ctr`, `simple_ctr_priors`, `combinations_ctr` TYPE, `counter_calc_method`) genuinely honored by the training engine, then route `cat_features`/CTR through the `CatBoostBuilder` facade and the Python bindings, at ≤1e-5 vs `catboost==1.2.10`.
**Plan:** `.planning/plans/ctr-type-engine-and-facade-routing/{PLAN.md, PLAN-W2-W3.md, PLAN-W4-W5.md, PLAN-PART2.md, PLAN-COVERAGE.md}` — 48 tasks (E00–E23, F00–F23)
**Pass:** 5 of 5 (confirming pass, scoped to Jobs 1–3). Pass 1 = ISSUES_FOUND (5C/5M/5m); pass 2 = ISSUES_FOUND (1C/2M/6m); pass 3 = ISSUES_FOUND (1C/1M/5m); pass 4 = ISSUES_FOUND (4C/1M/5m)

---

### Summary

- **JOB 1 — all five pass-4 fixes are VERIFIED CORRECT and, in four of five cases, over-delivered.**
  P4-1, P4-2, P4-3, P4-5 are fully applied with the exact wording demanded. P4-4 is applied in
  **form** (`#[non_exhaustive]` + constructor/builders per the locked decision, the
  self-contradictory `cbm.rs`/`json.rs` prohibition deleted, the rationale recorded) — but its
  **numbers are wrong**, see Job 2.
- **JOB 2 — the counts were re-derived from scratch. Three of the five headline numbers are right;
  two are wrong, and the mandated locating grep is broken in a way that reproduces the exact
  same-name-different-type trap that caused all three prior miscounts.**
  - `crates/cb-model/tests` = **18 literals** ✓ **plan is CORRECT** (but across **9** files, not 11).
  - `crates/cb-train` = **1** (`ctr_split_scoring_test.rs:518`) ✓ **plan is CORRECT**.
  - The **withdrawal of the `fstr_ctr_oracle_test.rs` STOP-AND-REPORT open item is CORRECT** ✓ — the
    file's only `Model` occurrence is `fn loaded_model() -> CbModel {` at `:84`, a return type; the
    model is produced by `load_cbm`. `ctr_data_roundtrip_test.rs` contains **zero** `Model` tokens.
    **No ZERO-DIFF oracle of this plan is affected** ✓.
  - `crates/catboost-rs` = **4**, **NOT 0** ✗ — `src/model_sum_test.rs:16`, `src/onnx_test.rs:32`,
    `src/model_device_test.rs:25`, `:68`, all `cb_model::Model {` in a **separate crate**.
  - **Total external migration set = 23, not 19** ✗.
  - `crates/cb-model/src` = **21** literals (+ one `Self {`), not 17 and not "roughly 37" ✗.
- **JOB 3 — no unowned consumer found among the three symbols whose ownership changed.** The two new
  `pub(crate) fn`s collide with nothing and have no pre-existing caller; `Model` has no
  pattern-match / destructuring / FRU consumer anywhere in the workspace; `decode_ctr_model_parts` /
  `decode_ctr_blob` have no mean-rejection consumer besides the one `cbm_test.rs` test E19 now owns.
- **Locked decisions (1)–(5) all still honored**; not reconsidered.
- **Execution-blocking?** ONE MAJOR remains: F08 states, with a `[VERIFIED]` tag, that
  "`crates/catboost-rs` contributes **ZERO**" and "Total external migration set: **19**", while the
  SAME task's Files list, Rationale ("5 outside `cb-model`") and Regression scope
  (`cargo test -p catboost-rs`) all say **4 in `catboost-rs`**. That is a self-contradiction inside
  one task. Compounding it, the grep the task mandates ("Locate them with, **and only with**")
  returns **none** of the five external literals and **does** return
  `crates/cb-train/src/boosting.rs:5476` — a genuine struct literal of the **different**
  `cb_train::Model`. Everything else remaining is MINOR/cosmetic and must not block.

---

### Pass 1–4 disposition (brief)

| Pass | Finding | Status at pass 5 |
|---|---|---|
| 1 | CRITICAL-1..5, MAJOR-6..10, 12 MINORs | VERIFIED at pass 3/4; not re-litigated |
| 2 | NEW-1..NEW-9 | VERIFIED at pass 3/4; not re-litigated |
| 3 | P3-1, P3-2 | VERIFIED at pass 4; P3-7 retracted (pass 3 was wrong) |
| 4 | **P4-1** s_order oracle unowned + mis-gated (CRITICAL) | **FIXED — VERIFIED** |
| 4 | **P4-2** E09 leaves `cb-train` un-buildable (CRITICAL) | **FIXED — VERIFIED** |
| 4 | **P4-3** E19 breaks unowned `cbm_test.rs` decode test (CRITICAL) | **FIXED — VERIFIED** |
| 4 | **P4-4** `cb_model::Model` blast radius (CRITICAL) | **FIXED IN FORM — numbers wrong (P5-1)** |
| 4 | **P4-5** no observation channel for `train_inner` locals (MAJOR) | **FIXED — VERIFIED** |
| 4 | P4-6 `pub(crate) fn` + `pub use` = `E0365` (MINOR) | **UNAPPLIED** — still MINOR |
| 4 | P4-7 E22 test fn 4 denominator channel (MINOR) | **PARTLY APPLIED** (file-choice justified; channel + field-creep prohibition still absent) — MINOR |
| 4 | P4-8 E11 `counter_calc_skip_test` `false` vs `true` (MINOR) | **UNAPPLIED** — still MINOR |
| 4 | P4-9 third mean-rejection doc block (MINOR) | **UNAPPLIED** — MINOR/doc |
| 4 | P4-10 three under-listed edge cells (MINOR, cosmetic) | **UNAPPLIED** — cosmetic |

---

### JOB 1 — verification of the five pass-4 fixes

**P4-1 — VERIFIED, complete.**

| Requirement | Evidence |
|---|---|
| owned by E09 (7→9) | `PLAN-W2-W3.md:355-367` ("mechanical, forced by the widened signature", the two new args at `:70`) |
| owned by E22 (9→10) | `PLAN-W4-W5.md:670-680` (`extra_cat_columns`, empty slice) |
| "CHANGE NO ASSERTION" both | `PLAN-W2-W3.md:363-367`; `PLAN-W4-W5.md:677-680` |
| `E0061` in E09's expected-failure list | `PLAN-W2-W3.md:456-460` (item 1c, "mirror `error[E0061]`… 9 arguments but 7 were supplied") |
| `E0061` in E22's expected-failure list | `PLAN-W4-W5.md:768` |
| MECHANICAL group = **4**, ZERO-DIFF = **7**, in all five places | `PLAN.md:229-253` (7 / 4, `s_order` listed under MECHANICAL at `:250-253`); E15 table `PLAN-W2-W3.md:1440-1450` (row 8 = MECHANICAL); E16 table `PLAN-W2-W3.md:1733-1743` (row 8 = MECHANICAL); `PLAN-COVERAGE.md:37`; `PLAN-COVERAGE.md:105` |
| E16 STOP-AND-REPORT exempts the new file | `PLAN-W2-W3.md:1508` names `s_order_ctr_bins_oracle_test.rs` in the exemption text |
| E09 completion evidence names the diff | `PLAN-W2-W3.md:518-524` ("exactly four added `target_border_idx: 0` … PLUS the three widened call sites … and NOTHING else") |

**P4-2 — VERIFIED, complete.** `PLAN-W2-W3.md:317-324` now reads "**E09 MUST leave `cb-train`
compiling** … E09 makes them compile with the pre-change constants … **E10 makes them
per-candidate**. (An earlier revision … claimed 'its two call sites in `boosting.rs` are E10's' —
that was WRONG and is deleted…)". Files item at `:345-354` adds `boosting.rs` with the
behavior-preserving `ECtrType::Borders` / `target_border_idx: 0` at `:3238` and `:3274` and the
explicit "Per-candidate type/prior resolution is E10's and MUST NOT be done here". EXPECTED INITIAL
FAILURE item **1d** (`:461-465`) carries the two production `E0061`s and states "not deferrable to
E10". Green (`:489-495`) ends "**`cb-train` MUST compile at the end of E09**". Completion evidence
(`:525-528`) gates `git diff crates/cb-train/src/boosting.rs` to exactly those two sites and requires
`cargo build -p cb-train` to succeed. The `E09->E10` edge already serializes the file.

**P4-3 — VERIFIED, complete.** `PLAN-W4-W5.md:119-139` adds `crates/cb-model/src/cbm_test.rs` to
E19's Files with "**INVERT the existing green test** `decode_ctr_parts_mean_ctr_type_is_typed_error`
at `:731-746` … **Do NOT delete it.** Rename it to
`decode_ctr_parts_mean_ctr_type_round_trips_sum_and_count` … assert the single decoded bucket is
`(7.0f32, 3i64)` … `assert!(!mean.is_empty())` … Keep a **sibling negative case** … neither stride 8
nor stride 12 … **No other test in `cbm_test.rs` may be touched.**" The wording explicitly mirrors
E20's encode-side inversion. Accounting block at `:299-302`; `PLAN-COVERAGE.md:34` (SPEC-CTRT-15)
carries the same. Verified against the on-disk test at `crates/cb-model/src/cbm_test.rs:731-746`
(1 bucket, 8-byte `f32`+`i32` blob — precisely what E19's stride-8 probe accepts).

**P4-5 — VERIFIED, complete and stronger than demanded.** `PLAN-W2-W3.md:1336-1354` — E15 **Green
step 0**, executed *first*, extracts two `pub(crate) fn`s in `crates/cb-train/src/boosting.rs`:
`materialize_ctr_columns_for_perm(..) -> CbResult<Vec<CtrFeatureColumn>>` (the body of both
`train_inner` loops at `:3237-3247` / `:3273-3282`) and
`cat_eligible_buckets_for(cat_columns, eligible_absolute) -> Vec<Vec<u32>>` (the `:3074` build), both
declared behavior-preserving. E15 test fn 2 (`:1278-1305`) and E16 test fn 1 (`:1587-1624`) both
assert **through** those functions, and both carry an explicit
"**FORBIDDEN: a test that re-derives the expansion itself** … Such a test is a **tautological
guard**". The `cat_eligible_buckets` pin is made falsifiable exactly as demanded — same fixture,
`cat_eligible_buckets_for(..).len() == 1` **while** `cols.len() == 2` — plus a byte-unchanged
cross-check against the Borders configuration, and the retracted lockstep invariant is explicitly
restated as retracted (`:1620-1624`).

---

### JOB 2 — independent re-derivation of the `Model` blast radius

**Method.** Three distinct greps, because no single regex is sufficient: (a) bare `Model {`, (b)
qualified `cb_model::Model {` / `crate::Model {` / `CbModel {` (the plan's own char class
`[^a-zA-Z_:]` **excludes `:` and therefore cannot see any qualified path**), (c) `Self {` inside
`impl Model`. Every hit was then classified by reading the line: a hit is counted **only** if it is a
struct-literal construction of `cb_model::Model`. Excluded: `-> Model {` return types,
`pub struct Model {`, `impl Model {`, the CoreML proto `cm::Model`
(`crates/cb-model/src/generated/coreml_generated.rs:25`), `catboost_rs::Model`
(`crates/catboost-rs/src/model.rs:64`) and `cb_train::Model`
(`crates/cb-train/src/boosting.rs:905`).

**Struct shape confirmed** `[CODEGRAPH crates/cb-model/src/model.rs:304-346]`:
`#[derive(Debug, Clone, PartialEq)] pub struct Model` — 8 fields, **no `Default`**, **no
`#[non_exhaustive]`**, **no `Serialize`/`Deserialize` derive**. Workspace-wide `..Default::default()`
hits are **all** in `export/onnx.rs` / `export/coreml.rs` proto structs — **zero** inside a `Model`
literal ✓ (the plan's claim is correct).

#### A. `crates/cb-model/src` — **21 literals** (+ one `Self {`). Plan says 17 (parent brief) / "roughly 37" (plan text). **Both wrong.**

Bare `Model {` — **17**:
`model_sum.rs:115`, `json.rs:825`, `cbm.rs:1193`, `cbm.rs:1318` (the four production sites),
`region_apply_test.rs:42`, `fstr_test.rs:21`, `gpu_apply_test.rs:22`,
`partial_dependence_test.rs:36`, `:123`, `:164`, `model_sum_test.rs:17`, `:73`,
`export/coreml_test.rs:25`, `staged_predict_test.rs:37`, `:52`, `export/onnx_test.rs:23`,
`apply_one_hot_test.rs:25`.

Qualified `crate::Model {` — **4**, **invisible to the plan's grep**:
`cbm_test.rs:322`, `:879`, `:900`, `:1030`.

Plus `Self {` at `model.rs:515` inside `from_trained` — the plan handles this one by name
(Green step 3), so it is not a gap.

*Where the plan's "roughly 37" comes from:* the plan's grep
`grep -rnE '(^|[^a-zA-Z_:])Model *\{$' crates/cb-model/src` returns **37 lines**, of which **20 are
not literals** (16 `-> Model {` return types, `pub struct Model {` ×2, `impl Model {`, and the CoreML
proto struct). 37 − 20 = 17; the plan then says "37 literals **plus** the production sites
`json.rs`, `cbm.rs` (×2), `model_sum.rs`", double-counting four sites that are already inside the 37.
**Non-blocking**: the plan says "do NOT trust a hard-coded line list", every intra-crate site is
compile-forced by `error[E0063]`, and the four `crate::Model {` sites the grep misses will still
break the build and be fixed. MINOR.

#### B. `crates/cb-model/tests` — **18 literals**. **Plan is CORRECT.** Across **9** files, not 11.

`apply_oracle_test.rs:66`, `:110`; `cbm_oracle_test.rs:78`, `:94`;
`advanced_fstr_oracle_test.rs:78`; `class_params_roundtrip_test.rs:25`, `:47`;
`predict_oracle_test.rs:59`; `json_oracle_test.rs:53`, `:206`;
`predict_raw_multi_test.rs:33`, `:73`, `:99`, `:141`;
`shap_oracle_test.rs:72`, `:162`, `:243`; `fstr_oracle_test.rs:70`.

The plan's grep returns 35 lines (18 literals + 17 `-> Model {` return types); its per-file
breakdown ("advanced_fstr 3, apply 3, cbm 4, class_params 4, fstr 4, json 3,
partial_dependence_oracle 1, predict 2, predict_raw_multi 5, shap 5, staged_predict_oracle 1") sums
to **35**, i.e. it is the raw line count, not the literal count — even though the headline **18** is
right. `partial_dependence_oracle_test.rs:35` and `staged_predict_oracle_test.rs:38` are
`-> Model { load_cbm(..) }` and contain **no literal at all**, so the correct file count is **9**.
MINOR.

#### C. `crates/cb-train` — **1**. **Plan is CORRECT.**
`crates/cb-train/tests/ctr_split_scoring_test.rs:518` (`cb_model::Model {`, read verbatim; `:501` is
the `-> cb_model::Model {` return type). `crates/cb-train/src/boosting.rs:5476` is a literal of
`cb_train::Model` — a **different type** — and must **not** be touched.

#### D. `crates/catboost-rs` — **4**, NOT 0. **Plan's headline is WRONG.**
```
crates/catboost-rs/src/model_sum_test.rs:16     cb_model::Model {
crates/catboost-rs/src/onnx_test.rs:32          cb_model::Model {
crates/catboost-rs/src/model_device_test.rs:25  cb_model::Model {
crates/catboost-rs/src/model_device_test.rs:68  cb_model::Model {
```
All four are `#[cfg(test)] mod`-mounted inside `crates/catboost-rs/src` — a **separate crate** from
`cb-model`, so `#[non_exhaustive]` forbids the literal syntax at every one of them.
`crates/catboost-rs/src/model.rs:65` holds `inner: cb_model::Model` as a **field**, not a literal —
unaffected. The remaining `-> Model {` hits in `crates/catboost-rs/tests/*` are `catboost_rs::Model`
return types — correctly excluded.

#### E. All other crates — **0**.
`catboost-rs-py`, `cb-oracle`, `cb-data`, `cb-compute`, `cb-core`: no `cb_model::Model` literal.
`cb-backend`'s `FlatModel` / `GrownModel` are unrelated types.

#### Totals

| Group | Independently derived | Plan asserts | Verdict |
|---|---|---|---|
| `crates/cb-model/src` (intra-crate, field only) | **21** (+ `Self {` at `model.rs:515`) | 17 / "roughly 37" | ✗ MINOR (compile-forced anyway) |
| `crates/cb-model/tests` (external, migrate) | **18** across **9** files | 18 across 11 files | **✓ count correct**, file count MINOR |
| `crates/cb-train` (external, migrate) | **1** | 1 | **✓ correct** |
| `crates/catboost-rs` (external, migrate) | **4** | 0 (headline) / 4 (Files, Rationale, Regression scope) | **✗ MAJOR — self-contradictory** |
| **Total external migration set** | **23** | 19 | **✗** |
| `fstr_ctr_oracle_test.rs` withdrawal | **CORRECT** | withdrawn | **✓** |
| `ctr_data_roundtrip_test.rs` | zero literals | zero | **✓** |

#### The mandated locating grep is broken (this is the substantive defect)

`PLAN-PART2.md:367-371` says "Locate them with, **and only with**:"
```
grep -rnE '(^|[^a-zA-Z_:])Model \{' crates/catboost-rs crates/cb-train
```
Run verbatim, that returns **10 lines**, of which:
- **0** are `cb_model::Model` literals — the char class excludes `:`, so every `cb_model::Model {` is
  invisible. All five true external non-`cb-model` sites are missed.
- **1** IS a genuine struct literal — `crates/cb-train/src/boosting.rs:5476`, of the **different**
  `cb_train::Model`. This is precisely the same-name-different-type trap that caused all three prior
  miscounts, now embedded in a mandated command.
- the rest are `-> Model {` return types, `pub struct`, `impl`.

The corrected regex the plan itself uses one paragraph earlier
(`'(^|[^a-zA-Z_:])(cb_model::Model|CbModel) *\{'`) works, and applied to `crates/catboost-rs` it
surfaces exactly the four sites the "ZERO" paragraph denies.

#### Other F08 checks requested

- **F08 no longer forbids touching `cbm.rs`/`json.rs`** ✓ — `PLAN-PART2.md:348-358`: "**REQUIRED,
  mechanical** … **The earlier prohibition … was SELF-CONTRADICTORY and is DELETED.**" The accurate
  replacement constraint (do not touch the FlatBuffers schema, the `.cbm` encoder's **byte output**,
  or `json.rs`'s **serde shape**) is stated in both Files and Green, and is backed by test fn 2
  (`adding_the_cat_feature_count_does_not_change_cbm_bytes`), which keeps E00's and
  `float_only_byte_identity`'s frozen baselines valid.
- **The `#[non_exhaustive]` rationale is recorded** ✓ — `PLAN-PART2.md:267-276` ("LOCKED DECISION (do
  not re-litigate)") and `:474-480` ("permanently prevents external breakage for every future `Model`
  field — this exact defect class … has now been found four times in this plan").
- **`cbm_oracle_test.rs` (2 literals) / `json_oracle_test.rs` (2 literals) disposition is ADEQUATE**
  ✓ — they are correctly identified as NOT among the eleven SPEC-CTRT-18 targets; the plan names them
  as SPEC-OH-31 "edited by no task" files from the SHIPPED one-hot plan, requires the migration be
  mechanical and assertion-preserving, adds them by name to F08's regression scope
  (`PLAN-PART2.md:489-490`), adds an explicit "**A migrated site must produce a `Model`
  field-for-field identical to the literal it replaced**" constraint (`:484-486`), and requires the
  diff be recorded with a plain statement that no assertion changed. Given the migration is forced by
  a locked user decision and the byte-identity guard (test fn 2) is separate and explicit, this is a
  coherent and sufficient disposition. (The per-file counts are 2 and 2, not the plan's 4 and 3 —
  same raw-grep inflation; MINOR.)
- **F08's regression scope already reaches all three crates** ✓ — `PLAN-PART2.md:487-492` runs
  `cargo test -p cb-model`, `cargo test -p catboost-rs`, and
  `cargo test -p cb-train --test ctr_split_scoring_test`, naming catboost-rs as one of "the two other
  crates the migration reaches". This is what makes P5-1 a MAJOR rather than a CRITICAL: the wrong
  headline is contradicted by three other statements in the same task, and the compiler plus this
  regression scope will surface the four sites.

---

### JOB 3 — the recurring defect class, restricted to symbols whose ownership changed

**(a) The two new `pub(crate) fn`s in `boosting.rs` (E15 Green step 0).** No collision, no existing
caller affected.
- **No pre-existing caller.** Both bodies are lifted from `let` bindings **inside `train_inner`**
  (`crates/cb-train/src/boosting.rs:2555`) — `cat_eligible_buckets` at `:3074`, the two
  materialization loops at `:3237-3247` / `:3273-3282`. Function locals have no callers by
  construction; after extraction the only callers are the two `train_inner` sites the same task
  rewrites, plus the two tests. The extraction is declared behavior-preserving with the D-04
  single-prior byte-identity proof required to still hold after step 0 alone
  (`PLAN-W2-W3.md:1352-1354`). ✓
- **No owned-region collision.** `boosting.rs` is claimed by E02, E09, E10, E15, E16, E21, E22 — all
  declare **Parallelizable: NO**, so none can run concurrently. Region-wise they are disjoint or
  strictly sequenced: E09 (`:3238`/`:3274` arity) → E10 (same sites, values) → E15 (extract those
  loops + `:3074` + the bake block `:5437-5473`) → E16 (`:4662` + the helper body) → E21 (`EvalSet`)
  → E22 (`counter_calc_method` + the helper). E16 explicitly says "**Do not re-inline the helper** and
  do not add a second expansion site; `cat_eligible_buckets_for` is untouched by this task"
  (`PLAN-W2-W3.md:1648-1650`). ✓
- **No `E0365` risk for these two.** E15/E16's tests live in `crates/cb-train/src/boosting_test.rs`,
  a `#[cfg(test)] mod`-mounted sibling **inside** the crate, so `pub(crate)` is visible; and unlike
  the E04/E06/E07/E08 producers, no task orders these two into a `pub use` re-export list. ✓
- One drift: E22's Files (`PLAN-W4-W5.md:647-648`) still says "thread the eval cat columns from
  **BOTH** `materialize_ctr_feature` call sites" in `boosting.rs`. After E15's extraction there is
  **one** such call site (inside `materialize_ctr_columns_for_perm`) plus two *helper* call sites.
  Descriptive drift only — the compiler forces the correct shape. **MINOR.**

**(b) `Model` + `#[non_exhaustive]` + constructor — consumers beyond the literal sites.** None.
- **No pattern match / destructuring anywhere.** A workspace-wide search for
  `(let|match|if let) … Model {` returns 9 hits, all of which are `let model = Model { … }`
  *assignments* already inside the counted literal set — **zero** destructuring patterns, zero
  `match … { Model { .. } => }`. ✓
- **No functional-record-update consumer.** Zero `..Default::default()` / `..other` inside a `Model`
  literal workspace-wide. ✓ (`#[non_exhaustive]`'s `E0639` would otherwise fire on FRU too.)
- **No derived serde.** `Model` derives only `Debug, Clone, PartialEq`
  `[CODEGRAPH crates/cb-model/src/model.rs:304]`; `json.rs` builds it by hand, so `#[non_exhaustive]`
  cannot break a derive. ✓
- **The facade holds it as a private field**, not a literal — `crates/catboost-rs/src/model.rs:65
  inner: cb_model::Model`, constructed via `Model::from_canonical(inner)`
  `[CODEGRAPH crates/catboost-rs/src/model.rs:61-88]`. Unaffected. ✓
- F08's **test fn 3** correctly lives in an **integration** target
  (`crates/cb-model/tests/model_constructor_test.rs`) so the `#[non_exhaustive]` decision is verified
  from **outside** the crate, and Green step 2 requires builders for exactly the shapes the migrated
  sites need (`with_non_symmetric_trees`, `with_region_trees`, `with_approx_dimension`,
  `with_class_to_label`, `with_ctr_data`, `with_cat_feature_count`) with "Add **only** what a migrated
  site actually sets". Cross-checked against the four `catboost-rs` and one `cb-train` literals: they
  set `oblivious_trees`, `bias`, `float_feature_borders`, `ctr_data`, `approx_dimension`,
  `class_to_label` — all covered. ✓

**(c) `decode_ctr_model_parts` / `decode_ctr_blob` after E19's inversion.** No unowned consumer.
- `[CODEGRAPH decode_ctr_model_parts (crates/cb-model/src/ctr_data.rs:531) — callers in
  crates/cb-model/src/cbm.rs, crates/cb-model/src/lib.rs; tests: ctr_data_test.rs, cbm_test.rs]`.
- The **only** test pinning the mean rejection is `cbm_test.rs:731-746`, now owned by E19 with the
  INVERT instruction. Its four sibling negative tests that also call `decode_ctr_model_parts`
  (`:695` bucket-index gap, `:712` duplicate index, `:725` blob width mismatch, `:766` duplicate
  table key) all build **`TailECtrType::Borders` / `Counter`** parts — read verbatim — so none is
  affected by removing the `is_mean()` guard. ✓
- `decode_ctr_blob` (`:711`) has **exactly 1 caller** (`decode_one_ctr_value_table`) and
  ⚠️ no covering tests `[CODEGRAPH]`; E19 adds a **sibling** decoder rather than modifying it
  (`PLAN-W4-W5.md:153-157`), so that single caller is E19's own edit. ✓
- The other four `is_mean()` sites are untouched by E19 and remain correct:
  `ctr_data.rs:369` (`to_json`), `:443` (`from_json` — pinned green by
  `ctr_data_test.rs:83 json_round_trip_mean_table`, which already passes and does not go through the
  `.cbm` decoder), `:812` (the **encode** rejection, E20's), `:1055` (the self-describing encoder).
  ✓ — but see P5-4 (the decode-side module doc block at `ctr_data.rs:495-505` still becomes a
  documentation lie; P4-9, unapplied, MINOR).

---

### Issues

#### [MAJOR] P5-1 — F08's external blast-radius headline is self-contradictory and understates the migration by 4 sites; the mandated locating grep finds none of them and surfaces a wrong-type false positive

- **Plan location:** `PLAN-PART2.md:287-312` (blast-radius group 1 + "Total external migration set:
  19" + the "`crates/catboost-rs` contributes **ZERO**" paragraph); `:364-380` (the "Locate them with,
  and only with" grep block); `:439-444` (EXPECTED INITIAL FAILURE item 3)
- **Requirement:** SPEC-CATF-Δ4 / CATF-10; the plan's own "no previously-passing target may fail" gate
- **Evidence:**
  - `PLAN-PART2.md:309-312` — "**Total external migration set: 19** (18 in `cb-model/tests` + 1 in
    `cb-train`). `crates/catboost-rs` contributes **ZERO** — its `Model` hits are `-> Model {` return
    types of its OWN `catboost_rs::Model`, a different type `[VERIFIED: LOCAL
    crates/catboost-rs/src/model.rs:64]`."
  - Contradicted three ways inside the SAME task: Files `:373-375` lists the four
    `crates/catboost-rs` sites to migrate; the Rationale `:474-476` says "**5** outside `cb-model`";
    Regression scope `:489-491` runs `cargo test -p catboost-rs` and calls it one of "the two other
    crates the migration reaches".
  - Independently re-derived this session: **4** `cb_model::Model` struct literals in
    `crates/catboost-rs/src/{model_sum_test.rs:16, onnx_test.rs:32, model_device_test.rs:25, :68}`,
    each read verbatim, each in a crate distinct from `cb-model`. True total = **23**.
  - The mandated grep `grep -rnE '(^|[^a-zA-Z_:])Model \{' crates/catboost-rs crates/cb-train`, run
    verbatim, yields 10 lines: **zero** `cb_model::Model` literals (the char class excludes `:`), and
    **one genuine literal of the wrong type** at `crates/cb-train/src/boosting.rs:5476`
    (`cb_train::Model`).
  - EXPECTED INITIAL FAILURE item 3 says "36 in `crates/cb-model/tests/*.rs`" where the blast-radius
    block says 18; the true number is **18**.
- **Failure scenario:** An executor that trusts the headline ("ZERO", "19") skips `crates/catboost-rs`
  and reports F08 complete at 19/23; `cargo test -p catboost-rs` then fails with 4 × `error[E0639]`
  in a crate the executor believes is out of scope. An executor that instead obeys "Locate them with,
  **and only with**" gets a hit list containing **no** real target and **one** `cb_train::Model`
  literal at `boosting.rs:5476`, and attempts to migrate the trainer's own model type to a
  `Model::new(..)` that does not exist on it — a wrong edit inside E22's serialized file, in a wave
  that runs after E22.
- **Impact:** F08 blocks the `F08->F09->F10` producer chain (itself a pass-1 CRITICAL fix). A wrong
  edit at `boosting.rs:5476` lands in the trainer's model assembly. The completion-evidence count
  (19) cannot match the real diff (23), so the task's own gate is un-satisfiable as written.
- **Required revision:**
  1. Delete the paragraph at `PLAN-PART2.md:310-312` ("`crates/catboost-rs` contributes **ZERO** …")
     — it is false and carries a `[VERIFIED]` tag it has not earned. Replace with:
     "`crates/catboost-rs` contributes **4** — `src/model_sum_test.rs:16`, `src/onnx_test.rs:32`,
     `src/model_device_test.rs:25`, `:68`, all written `cb_model::Model {` and all in a crate
     separate from `cb-model`. Its `-> Model {` hits under `tests/` are `catboost_rs::Model`
     (`crates/catboost-rs/src/model.rs:64`), a different type, and are NOT literals."
  2. `:309` — "Total external migration set: **19**" → "**23** (18 in `cb-model/tests` + 4 in
     `catboost-rs` + 1 in `cb-train`)". Same in the Rationale (`:474-476` already says 5 outside
     `cb-model` — keep it and make the two consistent) and in EXPECTED INITIAL FAILURE item 3, whose
     "36 in `crates/cb-model/tests/*.rs`" must become **18**.
  3. Replace the mandated grep block at `:368-371` with regexes that can see a qualified path AND
     exclude the two same-named types, and add the explicit exclusion note:
     ```bash
     # external literals — MUST be migrated to Model::new(..) + builders
     grep -rnE '(^|[^a-zA-Z_:])(cb_model::Model|CbModel|Model) *\{$' \
          crates/catboost-rs crates/cb-train crates/cb-model/tests \
       | grep -v -- '->'
     ```
     followed by: "**EXCLUDE by hand, they are DIFFERENT TYPES that merely share the name:**
     `crates/catboost-rs/src/model.rs:64` (`catboost_rs::Model`) and every literal of it;
     `crates/cb-train/src/boosting.rs:905` / the literal at `:5476` (`cb_train::Model`);
     `crates/cb-model/src/generated/coreml_generated.rs:25` (`cm::Model`). This same-name confusion
     has produced a wrong count four times — verify the type of every hit by reading the line before
     editing it."
  4. Correct the intra-crate grep at `:362` the same way — as written it misses the four
     `crate::Model {` literals in `crates/cb-model/src/cbm_test.rs:322`, `:879`, `:900`, `:1030`.

#### [MINOR] P5-2 — F08's literal counts are raw grep line counts, not literal counts

- **Plan location:** `PLAN-PART2.md:284-286` ("Roughly **37**"), `:298-307` (the per-file breakdown
  summing to 35), `:436-438` ("~37 in `crates/cb-model/src/**`"), `:474-476` ("36 in `cb-model`'s own
  integration targets")
- **Evidence:** The intra-crate grep returns 37 lines of which 20 are return types / `pub struct` /
  `impl` / the CoreML proto struct; true literal count is **21** (17 bare + 4 `crate::`-qualified).
  The `cb-model/tests` grep returns 35 lines of which 17 are return types; true count is **18** across
  **9** files, not 11 — `partial_dependence_oracle_test.rs` and `staged_predict_oracle_test.rs` are
  `-> Model { load_cbm(..) }` with no literal.
- **Impact:** Prose only. The plan explicitly says "do NOT trust a hard-coded line list", and every
  site is compile-forced (`E0063` intra-crate, `E0639` external). No edit changes.
- **Required revision:** State 21 / 18-across-9, and drop the double-counting of the four production
  sites ("37 literals **plus** the production sites `json.rs`, `cbm.rs` ×2, `model_sum.rs`" — those
  four are already inside the 37).

#### [MINOR] P5-3 — (P4-6, unapplied) `pub(crate) fn` producers are simultaneously ordered into a `pub use` re-export list

- **Plan location:** E04 (`PLAN.md:883` + Files `:853-854`), E06 (`PLAN-W2-W3.md:81` + Files `:39-40`),
  E07 (`:196`), E08 (`:280`)
- **Evidence:** `crates/cb-train/src/ctr/mod.rs:144-148` re-exports `pub` items; `pub use` of a
  `pub(crate)` item is `error[E0365]`.
- **Impact:** A compile error on E04's first Green step, trivially self-correcting. Cannot produce
  wrong code or a false pass. Note this does **not** touch E15/E16's two new `pub(crate) fn`s — those
  are not re-exported (Job 3a).
- **Required revision:** "`pub(crate) fn`" → "`pub fn`" in E04/E06/E07/E08.

#### [MINOR] P5-4 — (P4-7 / P4-8 / P4-9 / P4-10, unapplied) four residual pass-4 minors

- **P4-7** (`PLAN-W4-W5.md:749-750`): E22 test fn 4's `denominator_full` / `denominator_skiptest`
  still have no stated observation channel — `CtrFeatureColumn` has no denominator field and E22 adds
  none. The file-placement rationale was added (`:728-736`) but the channel was not. The
  field-creep prohibition ("Do NOT add a denominator field to `CtrFeatureColumn`") is still missing;
  adding one would break `tree_test.rs:374`, `:662`, which E22 does not own. The behavior is already
  gated by E22 test fn 1 in `ctr/online_test.rs` (`Full` → totals `[2,4]`, denominator 4) and by E23.
- **P4-8** (`PLAN-W2-W3.md:699` vs `:781-782`): E11 still says "threaded as a constant **`false`**
  here" in one paragraph and "every caller passes **`true`** (the `SkipTest` default)" in Green step
  3. `counter_calc_method_default()` returns `SkipTest`, so **`true`** is the behavior-preserving
  value. Delete the `false`.
- **P4-9**: `crates/cb-model/src/ctr_data.rs:495-505` (the decode-side module doc block stating that
  mean-type CTRs "are **rejected** (v1, SPEC §2/MAJOR-2)") still becomes a documentation lie after
  E19 and is still unnamed in E19's list of blocks to update, unlike E20's two. The file **is** in
  E19's Files, so the edit is authorized — only unnamed.
- **P4-10** (cosmetic): E18 "Blocked by: E13" omits `E11->E18`; E23 "Blocks: F00"
  (`PLAN-W4-W5.md:875`) omits `E23->F01/F07/F19`; F12 "Blocks: F20" (`PLAN-PART2.md:565`) omits
  `F12->F13`. Every omission is an under-statement and §4's edge list is declared authoritative.

#### [MINOR] P5-5 — two stale descriptions created by E15's extraction and E09's new file

- E22's Files (`PLAN-W4-W5.md:647-648`) says "thread the eval cat columns from **BOTH**
  `materialize_ctr_feature` call sites" in `boosting.rs`; after E15 Green step 0 there is **one** such
  call site (inside `materialize_ctr_columns_for_perm`) plus two *helper* call sites.
- E09's "Parallelizable: **NO** — owns `crates/cb-train/src/ctr/ctr_feature.rs`"
  (`PLAN-W2-W3.md:316`) no longer names `crates/cb-train/src/boosting.rs`, which E09 now also edits
  (the very next sentences explain it, so an executor is not misled). No hazard: E02, which also owns
  `boosting.rs` and has no edge to/from E09, is likewise **Parallelizable: NO**, so the two cannot run
  concurrently.
- Both are wording; neither changes what gets edited.

---

### Implementation Order Review

Edge list re-checked against `PLAN.md:361-376` — **unchanged from pass 4 and still acyclic**; the only
ID-descending edge remains `E03->E02`, and `E03` has no in-edges.

- **The pass-4 fixes moved no edge.** P4-1 and P4-3 are Files-list additions; P4-2 is a Files-list
  addition inside the already-serialized `E09->E10` chain; P4-4 was resolved by `#[non_exhaustive]`
  rather than a relocation, so `F06->F08->F09->F10` stands as the corrected producer chain; P4-5's
  extraction is E15's own Green step 0 and precedes both consumers (E15 test fn 2, E16 test fn 1).
- **File-ownership serialization, re-derived:** `boosting.rs` E02 ‖ E09 → E10 → E15 → E16 → E21 → E22
  (E02 and E09 are unordered but both **Parallelizable: NO**, and their regions — the `train_inner`
  rejection vs `:3238`/`:3274` — are disjoint); `online.rs` E04 → E05 → E06 → E07 → E08;
  `ctr_feature.rs` E09 → E22; `cb-model/src/ctr_data.rs` E11 → E19 → E20; `cb-model/src/model.rs` F08
  only; `catboost-rs/src/model.rs` F10 → F11 → F12 → F13 (F08 touches only
  `catboost-rs/src/*_test.rs`, not `model.rs` — no collision); `params.rs` F15 → F16. All intact.
- **No intermediate un-buildable state remains.** The two pass-4 order defects are closed: E09 now
  ends with `cargo build -p cb-train` succeeding, and every one of the eleven SPEC-CTRT-18 targets
  builds at every task boundary.
- **P5-1 requires no ordering change** — it is a text correction plus a corrected grep inside F08.

---

### Potential Bugs

- **Under-migration at F08 (P5-1).** Trigger: trusting "catboost-rs contributes ZERO / total 19".
  Failure: 4 × `error[E0639]` in `crates/catboost-rs`. Caught by F08's own regression scope
  (`cargo test -p catboost-rs`), so it surfaces rather than shipping — but the executor is told the
  crate is out of scope, so the likely response is confusion or an unauthorized edit.
- **Wrong-type edit at `crates/cb-train/src/boosting.rs:5476` (P5-1).** Trigger: obeying "Locate them
  with, **and only with**" the broken grep. Failure: an attempt to migrate a `cb_train::Model` literal
  to a `Model::new(..)` that does not exist on that type; self-correcting at compile time, but it is
  an edit to E22's serialized file made during W7.
- **Missed intra-crate literals in `cbm_test.rs` (P5-2).** Trigger: the `:362` grep excluding `crate::`
  paths. Failure: 4 × `error[E0063]`. Compile-forced, self-correcting.
- **Vacuous denominator assertions at E22 test fn 4 (P5-4/P4-7).** Trigger: no channel for
  `denominator_full`. Worst branch: an executor adds a denominator field to `CtrFeatureColumn`,
  re-breaking `tree_test.rs:374`, `:662` in a file E22 does not own. Mitigation: state the channel or
  drop the two lines, and add the explicit prohibition.
- **`E0365` at E04's Green (P5-3).** Trigger: `pub use` of a `pub(crate) fn`. Self-correcting.
- Closed at earlier passes and NOT re-litigated: bake copy-back per-split normalization; the
  `(projection, ctr_type)` de-dup key; `target_border_idx` excluded from `ctr_base_key`; `is_mean()`
  covering both mean types; the single-remap rule for Counter bins vs totals; the f32-vs-f64 BTMV
  accumulator differential; the eval-set bucket-space rule; the `.cbm` mean stride-8/12 probe.

---

### Compliance Checks (re-confirmed, not re-litigated)

- **Locked decisions, all still honored:** (1) engine before facade — `E23->F00/F01/F07/F19` plus
  "Part 2 starts only after E23 is green" ✓; (2) the `.cbm` mean lift gated by E00's hand-constructed
  **non-mean** byte-identity baseline plus E20's stride-mutation falsifiability ✓; (3) scalar
  `simple_ctr: ECtrType` untouched, the full prior list honored by E15, the multi-description default
  documented as a gap by F00 ✓; (4) `crates/cb-oracle/fixtures/one_hot_train/` reused read-only by
  F19/F20 with no corpus-wide generation and `gen_fixtures.py` never invoked ✓; (5) `Model` gets
  `#[non_exhaustive]` + a constructor — F08 Green steps 1–2, with the rationale recorded at
  `PLAN-PART2.md:267-276` and an external-crate verification test (test fn 3) in an integration
  target ✓.
- **Source/test separation**, **no `unwrap`/`expect`/`panic!` in production**, **no bare
  `--features rocm`**, **`catboost-master/` never cited as a parity reference**, **the accepted
  failing-test baseline used as a "no previously-passing target may fail" gate** (now honored on both
  the encode and decode sides after P4-3), **the uncommitted one-hot wave never reverted and its three
  oracles in mandatory regression scope for every `boosting.rs`/`tree.rs` task** — all ✓.

---

### Required Plan Revisions

1. **F08 (`PLAN-PART2.md`):** delete the false "`crates/catboost-rs` contributes **ZERO**" paragraph
   (`:310-312`); correct "Total external migration set" 19 → **23**; correct EXPECTED INITIAL FAILURE
   item 3's "36 in `crates/cb-model/tests`" → **18**; replace the "and only with" grep block with
   regexes that see qualified paths, exclude `->` lines, and carry an explicit
   "these three same-named types are NOT `cb_model::Model`" exclusion list naming
   `crates/catboost-rs/src/model.rs:64`, `crates/cb-train/src/boosting.rs:905`/`:5476`, and
   `crates/cb-model/src/generated/coreml_generated.rs:25`. **(MAJOR, P5-1)**
2. **F08:** correct the intra-crate grep so it sees `crate::Model {`
   (`crates/cb-model/src/cbm_test.rs:322`, `:879`, `:900`, `:1030`), and restate the counts as
   **21 intra-crate literals + `from_trained`'s `Self {`** and **18 external across 9 files**,
   dropping the double-count of the four production sites. **(MINOR, P5-2)**
3. **E04/E06/E07/E08:** `pub(crate) fn` → `pub fn`. **(MINOR, P5-3)**
4. **E22 test fn 4:** state the denominator observation channel or drop the two lines, and add
   "Do NOT add a denominator field to `CtrFeatureColumn`". **(MINOR, P5-4)**
5. **E11:** delete "threaded as a constant `false` here" at `PLAN-W2-W3.md:699`; `true` everywhere.
   **(MINOR, P5-4)**
6. **E19:** name `crates/cb-model/src/ctr_data.rs:495-505` as a third doc block to update in the same
   edit. **(MINOR, P5-4)**
7. **E18 / E23 / F12:** append the missing IDs to the three "Blocked by:"/"Blocks:" cells.
   **(MINOR, cosmetic, P5-4)**
8. **E22 Files / E09 Parallelizable:** "BOTH `materialize_ctr_feature` call sites" → "the single call
   site inside `materialize_ctr_columns_for_perm`, plus that helper's two call sites"; name
   `boosting.rs` in E09's "owns" clause. **(MINOR, P5-5)**

---

### Unverified Items

1. Whether the 30-row isolating configs yield a winning split of each intended type — data-dependent;
   falsified at generation time by each generator's anti-false-pass assertion. Correctly recorded.
2. The upstream `CTRBlob` mean stride — `[INFERRED]`, with the mandatory 8 → 12 → STOP-AND-REPORT
   probe and E18's upstream-produced `.cbm` as the falsifier. Resolved as far as a plan can.
3. Whether E23's fixture can discriminate `Full` from `SkipTest` while both land within 1e-5 — a
   pre-written deferral path exists.
4. Whether `catboost==1.2.10` is byte-deterministic for categorical-only fixtures — falsified by each
   fixture task's double-generation `diff -r`.
5. `bench/one_hot_gpu_speed/one_hot_bench_colab.py` end-to-end behavior after F09 — only the
   source-level preflight is testable locally.
6. Pass-4 item 6 (whether the facade-wrapper alternative to F08 is compatible with F10/F13) is
   **CLOSED** — the locked decision selected `#[non_exhaustive]` + a constructor on
   `cb_model::Model`, so the alternative is out of scope and no longer needs verification.
