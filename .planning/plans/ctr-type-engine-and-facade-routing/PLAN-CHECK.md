## Plan Check Result

**Verdict:** PASSED-with-minors (zero CRITICAL, zero MAJOR)
**Goal:** Make four inert CTR params (`simple_ctr`, `simple_ctr_priors`, `combinations_ctr` TYPE, `counter_calc_method`) genuinely honored by the training engine, then route `cat_features`/CTR through the `CatBoostBuilder` facade and the Python bindings, at ≤1e-5 vs `catboost==1.2.10`.
**Plan:** `.planning/plans/ctr-type-engine-and-facade-routing/{PLAN.md, PLAN-W2-W3.md, PLAN-W4-W5.md, PLAN-PART2.md, PLAN-COVERAGE.md}` — 48 tasks (E00–E23, F00–F23)
**Pass:** 6 of 6 (confirming pass, scoped to Jobs 1–5 on commit `33d8068`). Pass 1 = 5C/5M/5m; pass 2 = 1C/2M/6m; pass 3 = 1C/1M/5m; pass 4 = 4C/1M/5m; pass 5 = 0C/1M/4m. **Pass 6 = 0 CRITICAL, 0 MAJOR, 7 MINOR.**

---

### Summary

- **JOB 1 — pass-5's MAJOR (P5-1) is FIXED.** Every actionable defect it named is gone: the false
  "`crates/catboost-rs` contributes **ZERO**" paragraph is deleted and replaced by the correct **4**
  with the four line numbers; the "Total external migration set: **19**" headline is gone; the
  mandated *work-list* grep ("Locate them with, **and only with**") — which returned no real target
  and one wrong-type false positive — is **deleted outright** and replaced by a compiler-driven
  procedure; the intra-crate grep that could not see `crate::Model {` is likewise deleted, with the
  four `cbm_test.rs` sites now named explicitly. A `⚠️` block names all four same-named types
  (`cb_model::Model`, `catboost_rs::Model`, `cb_train::Model`, `cm::Model`) and declares the compiler
  authoritative. **One demanded sub-revision did not land:** EXPECTED INITIAL FAILURE item 3 still
  says "36 in `crates/cb-model/tests/*.rs`" where the true count is **18**.
- **JOB 1 (counts, independently re-derived this session, third time).** `crates/cb-model/src` =
  **21** literals (+ `from_trained`'s `Self {`); `crates/cb-model/tests` = **18** across **9** files;
  `crates/catboost-rs` = **4**; `crates/cb-train` = **1**; **external total = 23**. F08's two
  *authoritative* figures — "Expect **21** sites" (intra) and "Expect **23** sites" (external) — are
  **both exactly right**, as is the "roughly **20–25**" magnitude band and PLAN.md §4's independent
  restatement (18 / 4 / 1, ~20 intra). The stale figures are confined to prose: **37** (×2), **36**
  (×2), **19** (×1), "11 files" (×1), "(4)/(3)" for `cbm_oracle_test`/`json_oracle_test` (true 2/2).
- **JOB 2 — compiler-driven enumeration is SOUND as a plan instruction.** (a) The field and
  `#[non_exhaustive]` are added first (Green step 1) and both locating procedures say *add → build →
  fix exactly what rustc names → **repeat until clean***, using `cargo build -p cb-model
  --all-targets` and `cargo build --workspace --all-targets`. (b) The STOP-AND-REPORT trigger is
  **not falsifiable** as worded ("far fewer or far more") — MINOR. (c) The crate-boundary claim is
  **VERIFIED**: `catboost-rs` depends on `cb-model` by path, its three offending files are
  `#[cfg(test)] mod`-mounted in `src/lib.rs:74-79`, and `--all-targets` builds that unit-test target,
  so `E0639` fires at all four; `crates/cb-model/tests/*.rs` are separate integration crates and fire
  too. (d) The **completion-evidence grep did not get rewritten** and is now the weakest link — MINOR,
  because it is dominated by mandatory build/test commands in the same task.
- **JOB 3 — P5-2…P5-5 all confirmed still present, all still MINOR, none upgraded by the rewrite.**
  P5-3's `E0365` (E04/E06/E07/E08 declare `pub(crate) fn` while their Files add the same name to a
  `pub use` block) is re-verified against source and is still a self-correcting compile error.
  P5-4's P4-7 is re-verified against source: `CtrFeatureColumn` has **no** denominator field
  (`projection, ctr_type, prior_num, prior_denom, bins, ctr_value, bucket_count`), `denominator_full`
  still has no channel, and the "do NOT add a denominator field" prohibition is still absent — but
  F08 does not touch that type, so the hazard is unchanged.
- **JOB 4 — no unowned consumer.** `impl Model` in `cb-model` contains exactly four methods
  (`from_trained`, `with_ctr_data`, `leaf_values`, `leaf_weights`), so F08's `new` + six builders
  collide with nothing; the struct has exactly 8 fields, all expressible by `new(3) + 6 builders`,
  so no migrated shape is unexpressible; no FRU and no destructuring of `Model` exists anywhere
  outside `cb-model`; `from_trained`'s 30 callers see a body-only change. The two new
  `pub(crate) fn` names do not exist yet anywhere in the workspace.
- **JOB 5 — order intact.** The §4 edge list is byte-unchanged by `33d8068` and machine-verified
  acyclic (61 edges, 48 nodes, 0 cycles). No file-ownership serialization moved; no un-buildable
  intermediate state was introduced.
- **Execution-blocking? NO.** Every remaining finding is prose or a redundant secondary check whose
  authoritative counterpart in the same task is correct. **The plan is PASSED (with minors).**

---

### Pass 1–5 disposition

| Pass | Finding | Status at pass 6 |
|---|---|---|
| 1 | CRITICAL-1..5, MAJOR-6..10, 12 MINORs | VERIFIED at pass 3/4; not re-litigated |
| 2 | NEW-1..NEW-9 | VERIFIED at pass 3/4; not re-litigated |
| 3 | P3-1, P3-2 | VERIFIED at pass 4; P3-7 retracted (pass 3 was wrong) |
| 4 | P4-1..P4-5 (4 CRITICAL + 1 MAJOR) | VERIFIED FIXED at pass 5; not re-litigated |
| 4 | P4-6..P4-10 (5 MINOR) | carried into P5-3/P5-4; see below |
| 5 | **P5-1** F08 blast-radius headline self-contradictory; mandated grep broken (MAJOR) | **FIXED — VERIFIED** (compiler-driven rewrite; 1 sub-item unlanded → P6-1) |
| 5 | P5-2 counts are raw grep line counts (MINOR) | **PARTLY APPLIED** — "Expect 21"/"Expect 23" added and correct; "37"/"36"/"19"/"11 files" residue remains → **P6-1 / P6-2** |
| 5 | P5-3 (=P4-6) `pub(crate) fn` + `pub use` = `E0365` (MINOR) | **UNAPPLIED** — re-verified, still MINOR → **P6-5** |
| 5 | P5-4 (=P4-7/8/9/10) four residual minors (MINOR) | **UNAPPLIED** — all four re-verified, none upgraded → **P6-6** |
| 5 | P5-5 two stale descriptions (E22 "BOTH", E09 "owns") (MINOR) | **UNAPPLIED** — still wording-only → **P6-7** |
| 5 | — (new at pass 6) completion-evidence grep not rewritten | **NEW MINOR** → **P6-3** |
| 5 | — (new at pass 6) STOP-AND-REPORT trigger not falsifiable | **NEW MINOR** → **P6-4** |

---

### JOB 1 — verification of the P5-1 fix in F08

Diff reviewed: `git show 33d8068`, `git diff 95856b3..33d8068 -- .planning/plans/…`. Only three plan
files postdate pass 5's report (mtimes 11:52 vs PLAN-CHECK.md 11:49): `PLAN-PART2.md`, `PLAN.md`,
`PLAN-COVERAGE.md`. `PLAN-W2-W3.md` (11:36) and `PLAN-W4-W5.md` (11:30) were already in front of pass 5
and are **not** re-litigated here.

**Pass-5 Required Revision 1 — four sub-items.**

| Sub-item demanded | Landed? | Evidence |
|---|---|---|
| Delete the false "`crates/catboost-rs` contributes **ZERO**" paragraph | **YES** | The paragraph is gone. `PLAN-PART2.md:308-314` now states "**4 in `crates/catboost-rs`** (`src/model_sum_test.rs:16`, `src/onnx_test.rs:32`, `src/model_device_test.rs:25`, `:68` — all written `cb_model::Model {`, and twice mis-recorded as ZERO)". Repeated at `:290-292`. |
| "Total external migration set: **19**" → **23** | **YES, in the load-bearing places** | The "19" headline is deleted; `:308` gives "roughly **20–25**", `:394` gives "Expect **23** sites". Both correct. Residual "19" survives only in Completion evidence (`:542`) → **P6-3**. |
| EXPECTED INITIAL FAILURE item 3's "36 in `crates/cb-model/tests`" → **18** | **NO** | `PLAN-PART2.md:466-468` still reads "36 in `crates/cb-model/tests/*.rs`, 4 in `crates/catboost-rs`, 1 in `crates/cb-train/…`" (= 41, vs the correct 23). → **P6-1** |
| Replace the "Locate them with, **and only with**" grep block | **YES — superseded** | The whole block is deleted. `:381-386` and `:391-397` are now compiler-driven procedures. The `⚠️` block at `:320-331` names all four same-named types with file:line and ends "**DO NOT trust any grep here, including the ones below. THE COMPILER IS THE AUTHORITY.**" |

**Pass-5 Required Revision 2 — two sub-items.**

| Sub-item demanded | Landed? | Evidence |
|---|---|---|
| Correct the intra-crate grep so it sees `crate::Model {` | **YES — superseded** | The grep is deleted; `:384-386` now says "do NOT trust a grep — a grep excluding `:` misses the 4 `crate::Model {` literals in `cbm_test.rs` (`:322`, `:879`, `:900`, `:1030`). Expect **21** sites." All four line numbers re-verified correct this session. |
| Restate counts as **21 intra** / **18 across 9** and drop the double-count | **PARTLY** | "Expect **21** sites" (`:386`) ✓ and "roughly **20** sites in `cb-model/src`" (`:312-313`) ✓; but "Roughly **37** literals **plus** the production sites…" (`:284-286`) and "~37" (`:461`) survive with the double-count intact, and "**18** across **11** files" (`:406`) keeps the wrong file count. → **P6-1 / P6-2** |

**NEW self-contradictions introduced by the rewrite** (all MINOR, all captured as P6-1/P6-2/P6-3):

1. `:406` — "`crates/cb-model/tests/*.rs` — **18** across 11 files **(see the blast-radius block
   above)**" is a **dangling cross-reference**: the blast-radius block it points at now says at
   `:302-306` "**No per-file breakdown is given here on purpose.**" The pointer survived the deletion
   of its referent, and it carries the wrong file count (9, not 11).
2. The external total is now stated **four ways in one task**: "roughly 20–25" (`:308`) ✓,
   "Expect **23**" (`:394`) ✓, "36 + 4 + 1 = 41" (`:466-468`) ✗, "all **19**" (`:542`) ✗.
3. The intra-crate total is stated **four ways**: "Roughly 37" (`:284`) ✗, "roughly 20" (`:313`) ✓,
   "Expect 21" (`:386`) ✓, "~37" (`:461`) ✗.
4. `:345` — "`cbm_oracle_test.rs` (**4**) and `json_oracle_test.rs` (**3**)"; ground truth is **2**
   and **2** (`cbm_oracle_test.rs:78, :94`; `json_oracle_test.rs:53, :206`).
5. `:500` — "5 outside `cb-model` + **36** in `cb-model`'s own integration targets" repeats the 36.

**What the rewrite got RIGHT and must not be touched:**
- The withdrawal of the `fstr_ctr_oracle_test.rs` STOP-AND-REPORT open item is **correct** — re-verified:
  the file's only `Model` token is `fn loaded_model() -> CbModel {` at `:84`, a return type
  `[VERIFIED: LOCAL, read verbatim]`. `ctr_data_roundtrip_test.rs` has zero. **F08 reaches none of the
  seven ZERO-DIFF oracles** — a workspace-wide literal enumeration finds `cb_model::Model` literals in
  `crates/cb-train` at exactly two places: `tests/ctr_split_scoring_test.rs:518` (MECHANICAL group,
  F08 now named as an owning task in PLAN.md §3.2 and PLAN-COVERAGE SPEC-CTRT-18) and
  `src/boosting.rs:5476` — which is a **`cb_train::Model`** literal and must never be touched.
- PLAN.md §4's lane-A restatement (`18` in cb-model/tests, `4` in catboost-rs, `1` in cb-train,
  `~20-25` total, `~20` intra) is **entirely correct** — the authoritative ordering document carries
  the right numbers.
- PLAN-COVERAGE's new R15, the Δ4 amendment and the SPEC-CTRT-18/R1 regrouping (ZERO-DIFF 8→7,
  MECHANICAL 3→4) are internally consistent with PLAN.md §3.2.

---

### JOB 2 — is compiler-driven enumeration sound as a plan instruction?

**(a) Ordering permits enumeration — YES.** `Green` step 1 adds `#[non_exhaustive]` **and** the field
before any migration step (steps 3–5). Both Files bullets state the procedure explicitly:
- intra-crate (`:381-383`): "add the field, run `cargo build -p cb-model --all-targets`, and fix
  exactly the sites the compiler names with `error[E0063]: missing field cat_feature_count`.
  **Repeat until clean.**"
- external (`:391-395`): "add `#[non_exhaustive]`, run `cargo build --workspace --all-targets`, and
  migrate exactly the sites the compiler names with `error[E0639]` … **Repeat until clean.**"

The Red phase is coherent with this: the plan states outright that once the field lands
"`cargo test -p cb-model` does not build until every one carries `cat_feature_count: 0`" (`:462-463`),
so the Red state is a *compile* failure, not a test failure, and is declared as such. The iteration
instruction is present in both directions, which is what makes the procedure closed. ✓

**(b) STOP-AND-REPORT trigger — NOT falsifiable as worded.** `:316-318`: "If the compiler names **far
fewer or far more**, STOP AND REPORT." Against a band of "roughly 20–25" this has no decision
boundary: 18 vs 23 vs 30 are all arguable. The consequence is bounded — a spurious stop wastes a
round-trip; a missed stop costs nothing, because the compiler list is the work list and the build must
go clean regardless. MINOR (**P6-4**), with a concrete numeric band proposed below.

**(c) `E0639` fires everywhere the plan expects — VERIFIED.**
- `crates/catboost-rs/Cargo.toml:34` — `cb-model = { path = "../cb-model", default-features = false }`
  `[VERIFIED: LOCAL]`. Separate crate ⇒ `#[non_exhaustive]` applies.
- `crates/catboost-rs/src/lib.rs:74-79` mounts `#[cfg(test)] mod model_device_test;`,
  `#[cfg(test)] mod model_sum_test;`, `#[cfg(test)] mod onnx_test;` `[VERIFIED: LOCAL, read verbatim]`.
  These compile into the **lib unit-test target**, which `--all-targets` builds — so all four literals
  (`model_sum_test.rs:16`, `onnx_test.rs:32`, `model_device_test.rs:25`, `:68`, each read verbatim and
  each written `cb_model::Model {`) are reached. ✓
- `crates/cb-model/tests/*.rs` are per-file integration crates ⇒ external ⇒ `E0639`. ✓
- FRU is not a factor: a workspace-wide search finds **zero** `..expr` inside any `Model` literal
  outside `cb-model` (the only external FRU hits are `..device_params()` on `BoostParams` in
  `crates/cb-train/tests/device_seam_test.rs:312,:358`) `[VERIFIED: LOCAL]`. No `match`/`let`
  destructuring of `Model` exists outside `cb-model` either.
- `Model` derives only `Debug, Clone, PartialEq` `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:304-346]`
  — no serde derive that `#[non_exhaustive]` could disturb.

**(d) Completion-evidence gate — degraded, but not load-bearing.** `:540-543` still runs
```
grep -rnE '(^|[^a-zA-Z_:>-])Model \{' crates/cb-train crates/cb-model/tests
```
Three defects survive the rewrite: the char class still excludes `:`, so it cannot see
`cb_model::Model {` — i.e. it is blind to the single `cb-train` target it scans
(`ctr_split_scoring_test.rs:518`); `crates/catboost-rs` is **not in the path list at all**, so 4 of the
23 sites are outside the gate; and it asserts "all **19** have moved". Run verbatim today it returns
**39 lines**, and after a perfect migration it will still return the `-> Model {` return types plus
`crates/cb-train/src/boosting.rs:{905,956,5476}` — including a genuine bare `Model {` literal of
`cb_train::Model` at `:5476` `[VERIFIED: LOCAL, executed]`. **Why this is MINOR, not MAJOR:** the same
task mandates `cargo build --workspace --all-targets` clean, plus `cargo test -p catboost-rs` and
`cargo test -p cb-train --test ctr_split_scoring_test` in **both** the Refactor regression scope
(`:517-521`) and Validation (`:529-530`) — none of which can pass with an unmigrated site, so the grep
cannot license an incomplete migration. And the `⚠️` block explicitly names `boosting.rs:905` /
`cb_train::Model` as a different type, so the wrong-type edit pass 5 feared is now warned against
in-task. Fix is still required for hygiene (**P6-3**).

---

### JOB 1/2 evidence — independent re-derivation of the `Model` blast radius (third derivation)

**Method.** One regex covering *both* bare and qualified forms —
`(^|[^A-Za-z_])(Model|CbModel)[[:space:]]*\{` — then every hit classified by reading the line.
Counted only struct-literal constructions of `cb_model::Model`. Excluded by type:
`-> Model {` / `-> CbModel {` return types, `pub struct Model {`, `impl Model {`,
`cm::Model` (`crates/cb-model/src/generated/coreml_generated.rs:25`, constructed at
`export/coreml.rs:293`), `catboost_rs::Model` (`crates/catboost-rs/src/model.rs:64`),
`cb_train::Model` (`crates/cb-train/src/boosting.rs:905`, literal at `:5476`).

| Group | Derived (pass 6) | Pass 5 | F08 says | Verdict |
|---|---|---|---|---|
| `crates/cb-model/src` (intra, add field) | **21** + `Self {` at `model.rs:515` | 21 | "Expect **21**" ✓ / "roughly 20" ✓ / "**37**" ✗ ×2 | count correct where it matters |
| `crates/cb-model/tests` (external, migrate) | **18** across **9** files | 18 / 9 | **18** ✓, "11 files" ✗, "**36**" ✗ ×2 | count correct, file count wrong |
| `crates/catboost-rs` (external, migrate) | **4** | 4 | **4** ✓ | ✓ corrected |
| `crates/cb-train` (external, migrate) | **1** (`tests/ctr_split_scoring_test.rs:518`) | 1 | **1** ✓ | ✓ |
| all other crates + repo outside `crates/` | **0** | 0 | 0 | ✓ |
| **external total** | **23** | 23 | "Expect **23**" ✓ / "20–25" ✓ / "41" ✗ / "19" ✗ | correct where authoritative |

Intra-crate 21: `gpu_apply_test.rs:22`, `json.rs:825`, `partial_dependence_test.rs:36,:123,:164`,
`model_sum.rs:115`, `model_sum_test.rs:17,:73`, `export/coreml_test.rs:25`, `region_apply_test.rs:42`,
`staged_predict_test.rs:37,:52`, `cbm_test.rs:322,:879,:900,:1030`, `export/onnx_test.rs:23`,
`cbm.rs:1193,:1318`, `fstr_test.rs:21`, `apply_one_hot_test.rs:25` `[VERIFIED: LOCAL, each read]`.
Note `model_sum.rs:115`, not `:120` as F08's Green step 3 and Files say — a one-line drift the
compiler corrects (`E0063` names the real line); folded into P6-1.

External 18 across 9: `apply_oracle_test.rs:66,:110`; `cbm_oracle_test.rs:78,:94`;
`predict_oracle_test.rs:59`; `predict_raw_multi_test.rs:33,:73,:99,:141`;
`class_params_roundtrip_test.rs:25,:47`; `shap_oracle_test.rs:72,:162,:243`;
`advanced_fstr_oracle_test.rs:78`; `json_oracle_test.rs:53,:206`; `fstr_oracle_test.rs:70`.
`partial_dependence_oracle_test.rs:35` and `staged_predict_oracle_test.rs:38` are
`-> Model { load_cbm(..) }` and contain **no literal** — hence 9 files, not 11.

---

### JOB 3 — status of P5-2 … P5-5

- **P5-2** (counts) — **partly applied**; residue re-scoped as **P6-1 / P6-2**. Still MINOR: the plan
  says three times not to trust a count, both authoritative "Expect N" figures are right, and every
  site is compile-forced.
- **P5-3** (= P4-6, `E0365`) — **UNAPPLIED, re-verified, still MINOR, NOT upgraded.**
  `PLAN.md:900` (E04), `PLAN-W2-W3.md:81` (E06), `:196` (E07), `:280` (E08) all still say
  `pub(crate) fn`, while `PLAN.md:853-856` (E04) says "add `online_class_prefix` to the existing
  `pub use online::{…}` re-export block at `:144-148`" and `PLAN-W2-W3.md:39-40` (E06) repeats it.
  Source confirms the re-export is a `pub use`:
  `crates/cb-train/src/ctr/mod.rs:144-148` — `pub use online::{ accumulate_online,
  online_ctr_prefix_binclf, ordered_ctr_per_permutation, OnlineCtrAccumulator, … };`
  `[VERIFIED: LOCAL, read verbatim]`. `pub use` of a `pub(crate)` item is `error[E0365]` — a compile
  error at E04's first Green step, self-correcting, no wrong code and no false pass.
  **Not touched by the F08 rewrite.**
- **P5-4** (= P4-7/8/9/10) — **UNAPPLIED, all four re-verified, none upgraded.**
  - **P4-7 (the one with teeth): confirmed real, confirmed still MINOR.** `PLAN-W4-W5.md:749-750`
    still asserts `denominator_full == 3` / `denominator_skiptest == 2`, and
    `crates/cb-train/src/ctr/ctr_feature.rs:69-95` shows `CtrFeatureColumn`'s complete field set —
    `projection, ctr_type, prior_num, prior_denom, bins, ctr_value, bucket_count`
    `[VERIFIED: LOCAL, read verbatim]`. There is **no denominator field**, and E22 adds none, so two
    of test fn 4's five assertions have no observation channel (the other three — `bucket_count_full`,
    `bucket_count_skiptest`, `column_full.bins.len()` — are observable via `column.bucket_count` /
    `column.bins`). The "**Do NOT add a denominator field to `CtrFeatureColumn`**" prohibition is
    still absent; the hazard it guards is real — `crates/cb-train/src/tree_test.rs:374` and `:662`
    construct exhaustive `CtrFeatureColumn { … }` literals in a file E22 does not own
    `[VERIFIED: LOCAL, read verbatim]`, so a new field is `E0063` there. Still MINOR because the
    MAX-denominator behavior is already pinned at bins level by E22 test fn 1 in `ctr/online_test.rs`
    (`Full` → totals `[2,4]`) and again by E23. **F08 does not touch `CtrFeatureColumn`, so this is
    not upgraded.**
  - **P4-8:** `PLAN-W2-W3.md:697-701` still says the new `counter_calc_skip_test` parameter is
    "threaded as a constant **`false`** here", while Green step 3 (`:780-781`) says "every caller
    passes **`true`** (the `SkipTest` default)". `true` is the behavior-preserving value. UNAPPLIED.
  - **P4-9:** `crates/cb-model/src/ctr_data.rs:495-505` still documents that mean-type CTRs
    "are **rejected** (v1, SPEC §2/MAJOR-2 — their `TCtrMeanHistory` byte layout is not empirically
    dissected and no fixture exercises it)" `[VERIFIED: LOCAL, read verbatim]`. E19's Files
    (`PLAN-W4-W5.md:115`) lists the file but still names only the two other blocks. UNAPPLIED, doc-only.
  - **P4-10:** re-verified verbatim — `PLAN-W4-W5.md:36` "Blocked by: E13" omits `E11->E18`;
    `PLAN-W4-W5.md:875` "Blocks: F00 (start of Part 2)" omits `E23->F01/F07/F19`;
    `PLAN-PART2.md:590` "Blocked by: F11. **Blocks:** F20" omits `F12->F13`. §4's edge list is
    declared authoritative and carries all of them. Cosmetic.
- **P5-5** (stale descriptions) — **UNAPPLIED.** `PLAN-W4-W5.md:647-648` still says "**BOTH**
  `materialize_ctr_feature` call sites"; after E15 Green step 0 there is one such call site inside
  `materialize_ctr_columns_for_perm` plus that helper's two call sites. `PLAN-W2-W3.md:316` still
  omits `boosting.rs` from E09's "owns" clause. Wording only; the compiler forces the correct shape
  and E02 (the only other unordered `boosting.rs` owner) is likewise `Parallelizable: NO`.

---

### JOB 4 — the recurring unowned-consumer sweep (restricted to symbols the pass-4/5 revisions changed, plus everything `33d8068` newly touches)

**(a) F08's `Model` constructor + builder surface (NEW at `33d8068`).** No unowned consumer, no
collision, no unexpressible shape.
- `impl Model` at `crates/cb-model/src/model.rs:348` contains **exactly four** methods —
  `from_trained` (`:359`), `with_ctr_data` (`:530`), `leaf_values` (`:537`), `leaf_weights` (`:546`)
  `[VERIFIED: CODEGRAPH + LOCAL, read verbatim]`. There is **no existing `Model::new`**, no
  `with_non_symmetric_trees`, `with_region_trees`, `with_approx_dimension`, `with_class_to_label` or
  `cat_feature_count` — F08's six new methods collide with nothing. `ModelSplit` has its own separate
  `impl` at `:110`; there is no second `impl Model` block in the crate.
- `Model` has **exactly 8 pub fields** (`oblivious_trees`, `non_symmetric_trees`, `region_trees`,
  `bias`, `float_feature_borders`, `ctr_data`, `approx_dimension`, `class_to_label`)
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:304-346]`. F08's `new(oblivious_trees, bias,
  float_feature_borders)` plus `with_ctr_data` (already exists) and the five new builders cover all
  **nine** post-F08 fields, so every one of the 23 migrated sites is expressible. ✓
- `from_trained` — **30 callers** `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:359; callers incl.
  crates/cb-model/tests/non_symmetric_grower_roundtrip_oracle_test.rs, crates/cb-train/tests/device_nonsym_fit_test.rs,
  device_region_fit_test.rs, lambdamart_oracle_test.rs +22]`. F08 changes only its **body** (adds
  `cat_feature_count: 0` to the `Self {` at `:515`), not its signature — all 30 callers unaffected. ✓
- `with_ctr_data` — 1 caller + `crates/cb-model/src/model_sum_test.rs`
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:530]`. Signature untouched. ✓
- The facade holds `inner: cb_model::Model` as a **field**, exposed by
  `as_canonical(&self) -> &cb_model::Model` (`crates/catboost-rs/src/model.rs:79`)
  `[VERIFIED: CODEGRAPH]` — a borrow, not a literal. Unaffected. ✓
- `catboost-rs-py`, `cb-oracle`, `cb-data`, `cb-compute`, `cb-core`, `cb-backend` and everything
  outside `crates/` contain **zero** `cb_model::Model` literals `[VERIFIED: LOCAL, workspace-wide]`. ✓

**(b) E15's two extracted `pub(crate) fn`s (unchanged since pass 5, re-confirmed).**
- Neither name exists anywhere in the workspace today (`fn materialize_ctr_columns_for_perm` /
  `fn cat_eligible_buckets_for` → no match) `[VERIFIED: LOCAL]`, so extraction cannot shadow or
  collide.
- Their bodies are `train_inner` locals: `cat_eligible_buckets` is built at
  `crates/cb-train/src/boosting.rs:3074` (`let cat_eligible_buckets: Vec<Vec<u32>> =
  eligible_absolute.iter().map(…)`) `[VERIFIED: LOCAL, read verbatim]`, and the two materialization
  loops call `crate::ctr::materialize_ctr_feature(` at `:3238` and `:3274` `[VERIFIED: LOCAL]` —
  exactly the two sites E09/E10/E22 own. Locals have no callers by construction. ✓
- `materialize_ctr_feature` blast radius `[VERIFIED: CODEGRAPH crates/cb-train/src/ctr/ctr_feature.rs:124
  — tests: crates/cb-train/tests/ctr_feature_materialize_test.rs, s_order_ctr_bins_oracle_test.rs,
  ctr_split_scoring_test.rs]` — all three are named as owned by E09/E22 (the `s_order` one being
  pass-4's P4-1 fix, already verified). Test call sites re-enumerated:
  `ctr_feature_materialize_test.rs:73,:122,:189`, `ctr_split_scoring_test.rs:384,:394`,
  `s_order_ctr_bins_oracle_test.rs:70` `[VERIFIED: LOCAL]` — no unowned consumer. ✓
- The one name `cat_eligible_buckets` also appears in `crates/cb-train/src/tree_test.rs:573,:601,:634,
  :684` as **local test variables** feeding `max_bucket_count_with_phantom` `[VERIFIED: LOCAL]` — not
  callers of the extracted function; unaffected by the extraction.

**(c) E19/E20 mean-CTR decode/encode.** Unchanged since pass 5 (`PLAN-W4-W5.md` mtime 11:30 predates
pass 5's report at 11:49) and cleared there: the only test pinning the decode-side mean rejection is
`crates/cb-model/src/cbm_test.rs:731-746`, now explicitly owned and INVERTED by E19
(`PLAN-W4-W5.md:119-139`, re-read this session). Not re-litigated.

**(d) Anything else `33d8068` newly touched.** PLAN.md §3.2 (diff-gate regrouping) and PLAN-COVERAGE
(R15 / Δ4 / SPEC-CTRT-18 / R1) add **F08** as an owning task for
`crates/cb-train/tests/ctr_split_scoring_test.rs` in the MECHANICAL row. Cross-checked: that file is
indeed the sole `cb-train` external site (`:518`), and it is now owned by E09, E11, E16, E22 **and**
F08 — all of which declare mechanical, assertion-preserving edits, and all of which are strictly
ordered (`E09→E10→…→E22→E23→F00…→F08`). No new unowned consumer. ✓

---

### Issues

#### [MINOR] P6-1 — F08 states each of its two counts four ways; two of the eight figures are stale raw-grep numbers

- **Plan location:** `PLAN-PART2.md:284-286` ("Roughly **37** … **plus** the production sites"),
  `:461` ("~37"), `:466-468` (EXPECTED INITIAL FAILURE item 3: "36 … 4 … 1"), `:500`
  ("5 outside `cb-model` + **36** in `cb-model`'s own integration targets"), `:542` ("all **19**"),
  `:345` ("`cbm_oracle_test.rs` (4) and `json_oracle_test.rs` (3)"), `:370`/`:489`
  (`model_sum.rs:120`).
- **Requirement:** none directly; this is the sanity-check input to F08's own STOP-AND-REPORT gate.
- **Evidence:** re-derived this session `[VERIFIED: LOCAL, every hit read]` — intra-crate literals
  **21** (not 37; the 37 is the raw grep line count, of which 20 are return types / `pub struct` /
  `impl` / the `cm::Model` proto, and the "plus the production sites" clause double-counts four sites
  already inside it); `cb-model/tests` literals **18** (not 36; the 36 is again the line count);
  external total **23** (not 41, not 19); `cbm_oracle_test.rs` **2** and `json_oracle_test.rs` **2**;
  the `model_sum.rs` literal is at **`:115`**, not `:120`. The same task states the correct figures at
  `:312-313` ("roughly 20"), `:386` ("Expect **21**"), `:308` ("roughly 20–25") and `:394`
  ("Expect **23**"), and PLAN.md §4's lane-A block states 18/4/1 correctly.
- **Failure scenario:** an executor adds the field, sees rustc name **21** `E0063` sites against a Red
  section predicting "~37", or **23** `E0639` sites against a Red section predicting 41, and raises a
  spurious STOP-AND-REPORT. Wasted round-trip only.
- **Impact:** no wrong code, no false pass, no un-buildable state. The compiler list is the work list;
  the two "Expect N" figures attached to the authoritative procedures are both exactly right; the
  build must go clean either way.
- **Why not MAJOR:** the plan states the resolution rule three separate times ("THE COMPILER IS THE
  AUTHORITY", "do NOT trust a grep", "a plan-time tally would only invite an executor to trust it"),
  and the STOP-AND-REPORT trigger is anchored to the *correct* magnitude block, not to these figures.
- **Required revision:** `:284` → "**21** literals, including the production sites `json.rs:825`,
  `cbm.rs:1193`, `cbm.rs:1318`, `model_sum.rs:115` and `model.rs:515` (`from_trained`'s `Self {`)"
  (drop the "plus" double-count); `:461` "~37" → "**21**"; `:466-468` "36 in
  `crates/cb-model/tests/*.rs`" → "**18**"; `:500` "36" → "**18**"; `:542` "all 19" → "all **23**";
  `:345` "(4)"/"(3)" → "(**2**)"/"(**2**)"; `:370`/`:489` `model_sum.rs:120` → `:115`.

#### [MINOR] P6-2 — a dangling cross-reference and a wrong file count survive the deleted per-file breakdown

- **Plan location:** `PLAN-PART2.md:406-407`
- **Evidence:** the bullet reads "`crates/cb-model/tests/*.rs` — **18** across 11 files
  **(see the blast-radius block above)**", but the blast-radius block now reads at `:302-306`
  "**No per-file breakdown is given here on purpose.** Five successive attempts to enumerate this
  group by grep produced five different answers …". The referent was deleted; the pointer was not.
  The file count is also wrong: **9**, not 11 — `partial_dependence_oracle_test.rs:35` and
  `staged_predict_oracle_test.rs:38` are `-> Model { load_cbm(..) }` with no literal
  `[VERIFIED: LOCAL, read verbatim]`.
- **Failure scenario:** an executor follows the pointer, finds an explicit refusal to provide what it
  points at, and loses confidence in the surrounding (correct) instructions.
- **Impact:** prose only.
- **Required revision:** replace the parenthetical with "(the compiler enumerates them — no per-file
  breakdown is given, deliberately)" and drop "across 11 files", or state "across **9** files".

#### [MINOR] P6-3 — the completion-evidence grep was not rewritten with the rest of F08 and is now the weakest check in the task

- **Plan location:** `PLAN-PART2.md:540-543`
- **Evidence:** the block still runs
  `grep -rnE '(^|[^a-zA-Z_:>-])Model \{' crates/cb-train crates/cb-model/tests` and asserts it
  "returns **zero** `cb_model::Model` struct literals afterwards — all 19 have moved". Three defects,
  each re-verified by running it: (i) the char class still excludes `:`, so the one `cb-train` target
  it scans (`tests/ctr_split_scoring_test.rs:518`, written `cb_model::Model {`) is **invisible to it
  before and after** migration; (ii) `crates/catboost-rs` — 4 of the 23 sites — is **not in the path
  list**; (iii) run verbatim today it returns **39 lines**, and after a perfect migration it will
  still return the `-> Model {` return types plus `crates/cb-train/src/boosting.rs:905` /`:956` /
  `:5476`, the last being a genuine bare `Model {` literal of `cb_train::Model`
  `[VERIFIED: LOCAL, executed]`. The stale "19" is folded into P6-1.
- **Failure scenario:** the gate reports "zero" even if `catboost-rs` and `cb-train` were never
  migrated (vacuous pass); or an executor reading it loosely treats `boosting.rs:5476` as an
  unmigrated site and edits the trainer's own model type.
- **Impact:** none that survives the task's other gates — `cargo build --workspace --all-targets`
  clean, `cargo test -p catboost-rs` and `cargo test -p cb-train --test ctr_split_scoring_test` are
  all mandated in the same task (`:517-521`, `:529-530`) and none can pass with an unmigrated site.
  The `⚠️` block at `:320-331` already names `crates/cb-train/src/boosting.rs:905` /`cb_train::Model`
  as a different type.
- **Why not MAJOR:** the gate is redundant to a strictly stronger build gate in the same task, and the
  wrong-type edit it could invite is explicitly warned against three paragraphs earlier and is
  self-correcting (`cb_train::Model` has no `new`, so any such edit is `error[E0599]`).
- **Required revision:** replace the block with a compiler-based statement plus a corrected grep:
  ```bash
  # the authoritative check: a clean build IS the completion evidence
  cargo build --workspace --all-targets            # zero E0063, zero E0639
  # secondary hygiene scan (all three external crates; sees qualified paths)
  grep -rnE '(^|[^a-zA-Z_:])(cb_model::Model|CbModel|Model) *\{' \
       crates/cb-model/tests crates/catboost-rs crates/cb-train | grep -v -- '->'
  ```
  followed by: "**Expected residue: exactly one line —
  `crates/cb-train/src/boosting.rs:5476`, a literal of `cb_train::Model`, a DIFFERENT TYPE that must
  NOT be migrated.** Anything else is an unmigrated site." Replace "all 19" with "all **23**".

#### [MINOR] P6-4 — the STOP-AND-REPORT trigger has no decision boundary

- **Plan location:** `PLAN-PART2.md:316-318`
- **Evidence:** "If the compiler names **far fewer or far more**, STOP AND REPORT — that gap is the
  signal, not a number to reconcile by hand", stated against "roughly **20–25**" (`:308`) and
  "roughly **20**" intra (`:313`). "Far" is undefined; the same task elsewhere gives exact figures
  (21 / 23) and, in the Red section, contradictory ones (P6-1).
- **Failure scenario:** the compiler names 23 external and 21 intra (the true values); an executor
  comparing against the Red section's 41 and "~37" reads that as "far fewer" and stops.
- **Impact:** a wasted round-trip. A missed stop costs nothing — the migration is compiler-driven.
- **Required revision:** make it numeric and falsifiable: "**STOP AND REPORT if `cargo build
  --workspace --all-targets` names fewer than 18 or more than 28 `E0639` sites, or fewer than 18 or
  more than 25 `E0063` sites.** Expected: 23 and 21 respectively."

#### [MINOR] P6-5 — (P5-3 / P4-6, still unapplied) `pub(crate) fn` producers are ordered into a `pub use` re-export list

- **Plan location:** E04 (`PLAN.md:900` Green + Files `:853-856`), E06 (`PLAN-W2-W3.md:81` + Files
  `:39-40`), E07 (`:196`), E08 (`:280`)
- **Evidence:** `crates/cb-train/src/ctr/mod.rs:144-148` is `pub use online::{ … }`
  `[VERIFIED: LOCAL, read verbatim]`; `pub use` of a `pub(crate)` item is `error[E0365]`.
- **Impact:** a compile error on E04's first Green step, trivially self-correcting. Does not touch
  E15/E16's two new `pub(crate) fn`s — those are not re-exported (Job 4b).
- **Required revision:** `pub(crate) fn` → `pub fn` in E04/E06/E07/E08.

#### [MINOR] P6-6 — (P5-4 = P4-7/8/9/10, all still unapplied, none upgraded)

- **P4-7** (`PLAN-W4-W5.md:749-750`): `denominator_full` / `denominator_skiptest` still have no stated
  observation channel — `CtrFeatureColumn` (`crates/cb-train/src/ctr/ctr_feature.rs:69-95`) carries no
  denominator field and E22 adds none. The **field-creep prohibition is still missing**; adding such a
  field is `error[E0063]` at `crates/cb-train/src/tree_test.rs:374` and `:662`, a file E22 does not
  own. Already gated at bins level by E22 test fn 1 and by E23. **Revision:** state the channel or
  drop the two assertions, and add "**Do NOT add a denominator field to `CtrFeatureColumn`** — it
  would break `tree_test.rs:374`, `:662`, which this task does not own."
- **P4-8** (`PLAN-W2-W3.md:699` vs `:780-781`): "threaded as a constant **`false`** here" contradicts
  "every caller passes **`true`** (the `SkipTest` default)". `true` is behavior-preserving.
  **Revision:** delete the `false`.
- **P4-9**: `crates/cb-model/src/ctr_data.rs:495-505` still states mean CTRs "are **rejected**" and
  becomes a documentation lie after E19; the file is in E19's Files so the edit is authorized, only
  unnamed. **Revision:** name it as a third doc block to update.
- **P4-10** (cosmetic): `PLAN-W4-W5.md:36` omits `E11->E18`; `:875` omits `E23->F01/F07/F19`;
  `PLAN-PART2.md:590` omits `F12->F13`. §4's edge list is authoritative and complete.
  **Revision:** append the missing IDs.

#### [MINOR] P6-7 — (P5-5, still unapplied) two stale descriptions from E15's extraction and E09's new file

- `PLAN-W4-W5.md:647-648` — "thread the eval cat columns from **BOTH** `materialize_ctr_feature` call
  sites"; after E15 Green step 0 there is one such call site (inside
  `materialize_ctr_columns_for_perm`) plus that helper's two call sites.
- `PLAN-W2-W3.md:316` — E09's "Parallelizable: **NO** — owns `crates/cb-train/src/ctr/ctr_feature.rs`"
  no longer names `crates/cb-train/src/boosting.rs`, which E09 now also edits. No hazard: E02, the only
  other unordered `boosting.rs` owner, is likewise `Parallelizable: NO`.
- **Impact:** wording; neither changes what gets edited.

---

### Implementation Order Review

- **The §4 edge list is byte-unchanged by `33d8068`** — `git diff 95856b3..33d8068 -- PLAN.md` touches
  only §3.2's diff-gate categories (the pass-4 `s_order` regrouping, verified at pass 5) and the W6
  lane-A prose block. The edge list itself is not in the diff `[VERIFIED: LOCAL]`.
- **Acyclicity re-verified mechanically** on the current list (`PLAN.md:376-395`): **61 edges,
  48 nodes, 0 cycles** by DFS `[VERIFIED: LOCAL, executed]`. The only ID-descending edge remains
  `E03->E02`, and `E03` has no in-edges.
- **No file-ownership serialization moved.** `boosting.rs`: E02 ‖ E09 → E10 → E15 → E16 → E21 → E22
  (E02/E09 unordered but both `Parallelizable: NO`, disjoint regions); `online.rs`:
  E04→E05→E06→E07→E08; `ctr_feature.rs`: E09→E22; `cb-model/src/ctr_data.rs`: E11→E19→E20;
  `cb-model/src/model.rs`: **F08 only**; `catboost-rs/src/model.rs`: F10→F11→F12→F13 — F08 touches
  only `catboost-rs/src/*_test.rs`, never `model.rs`, so no collision with the F10–F13 chain;
  `params.rs`: F15→F16. All intact.
- **F08's new cross-crate reach introduces no ordering hazard.** F08 now edits files in three crates
  (`cb-model`, `catboost-rs`, `cb-train`). The `catboost-rs` files it touches
  (`src/{model_sum,onnx,model_device}_test.rs`) are owned by no other task in this plan
  `[VERIFIED: LOCAL, grep over all five plan documents]`. The one `cb-train` file
  (`tests/ctr_split_scoring_test.rs`) is co-owned with E09/E11/E16/E22, all of which complete before
  F00 (`E23->F00`), so F08's edit is strictly last on that file. ✓
- **No un-buildable intermediate state introduced.** F08 is a single task that adds the field, the
  attribute and every migration in one Green sequence, and its own Validation requires
  `cargo test -p cb-model -p catboost-rs` and `cargo test -p cb-train --test ctr_split_scoring_test`
  to pass at the task boundary. The two pass-4 order defects remain closed (E09 ends with
  `cargo build -p cb-train` succeeding).
- **P6-1 … P6-7 require no ordering change** — all are text corrections.

---

### Potential Bugs

- **Spurious STOP-AND-REPORT at F08 (P6-1/P6-4).** Trigger: the compiler names 21/23 sites against a
  Red section predicting ~37/41. Failure mode: an unnecessary halt. No wrong code.
- **Vacuous completion evidence at F08 (P6-3).** Trigger: running the stale grep, which omits
  `crates/catboost-rs` and cannot see `cb_model::Model {`. Failure mode: the grep reports "zero" for
  work not done. Fully caught by the mandatory `cargo test -p catboost-rs` and
  `cargo test -p cb-train --test ctr_split_scoring_test` in the same task.
- **Wrong-type edit at `crates/cb-train/src/boosting.rs:5476` (P6-3).** Trigger: reading the
  completion grep's output without applying the `cb_model::Model` qualifier. Failure mode: an attempt
  to migrate a `cb_train::Model` literal to a `Model::new(..)` that does not exist on it —
  `error[E0599]`, self-correcting. Now warned against by name in F08's `⚠️` block.
- **`E0365` at E04's Green (P6-5).** Trigger: `pub use` of a `pub(crate) fn`. Self-correcting.
- **Vacuous denominator assertions at E22 test fn 4 (P6-6/P4-7).** Worst branch: an executor adds a
  denominator field to `CtrFeatureColumn`, breaking `tree_test.rs:374`, `:662` in a file E22 does not
  own. Mitigation: state the channel or drop the lines, and add the prohibition.
- Closed at earlier passes and NOT re-litigated: bake copy-back per-split normalization; the
  `(projection, ctr_type)` de-dup key; `target_border_idx` excluded from `ctr_base_key`; `is_mean()`
  covering both mean types; the single-remap rule for Counter bins vs totals; the f32-vs-f64 BTMV
  accumulator differential; the eval-set bucket-space rule; the `.cbm` mean stride-8/12 probe; the
  four pass-4 CRITICALs; the `fstr_ctr_oracle_test.rs` STOP-AND-REPORT open item (withdrawn, and the
  withdrawal re-verified here).

---

### Compliance Checks (re-confirmed, not re-litigated)

- **Locked decisions (1)–(5) all still honored**, and (5) is now *strengthened*: `Model` gets
  `#[non_exhaustive]` + a constructor, the rationale is recorded as "LOCKED DECISION (do not
  re-litigate)" at `PLAN-PART2.md:267-276`, PLAN-COVERAGE Δ4 carries it, and F08 test fn 3 verifies it
  from an **integration** target (a genuinely separate crate). None reopened.
- **Source/test separation** — F08's new test fn 3 goes to `crates/cb-model/tests/model_constructor_test.rs`
  (a dedicated file), and every migrated file is an existing `*_test.rs` sibling or integration
  target. ✓
- **No `unwrap`/`expect`/`panic!` in production** — F08's Green adds only `#[must_use]` builders that
  assign one field and return `self`, mirroring `with_ctr_data` (`crates/cb-model/src/model.rs:530`,
  read verbatim). ✓
- **`catboost-master/` is never cited as a parity reference** — re-checked across all five plan
  documents; the only upstream citations are `catboost/libs/model/online_ctr.h`,
  `online_ctr.cpp`, `static_ctr_provider.cpp`, `catboost_options.cpp` and `json_model_helpers.cpp` as
  *upstream v1.2.10 line anchors*, plus `crates/cb-oracle/fixtures/` and the installed
  `catboost==1.2.10` as the empirical oracles. No task reads the stale 3-file stub. ✓
- **No bare `--features rocm`**; the accepted failing-test baseline
  (`.planning/plans/one-hot-categorical-training/baseline/`, 297 passed / 59 failed all in
  `cb_backend`) is used as a "no previously-passing target may fail" gate; the uncommitted one-hot wave
  is never reverted. ✓
- **The eleven SPEC-CTRT-18 oracles**: F08's `#[non_exhaustive]` reaches **none of the seven ZERO-DIFF
  files** and exactly one MECHANICAL file (`ctr_split_scoring_test.rs:518`), which now names F08 as an
  owning task in both PLAN.md §3.2 and PLAN-COVERAGE SPEC-CTRT-18 — re-verified by workspace-wide
  literal enumeration this session. ✓

---

### Required Plan Revisions

All MINOR. None blocks execution; apply as hygiene before or during F08.

1. **F08 (`PLAN-PART2.md`) — reconcile the eight count statements to the two correct ones:** `:284`
   "Roughly 37 … plus" → "**21**, including `json.rs:825`, `cbm.rs:1193`, `cbm.rs:1318`,
   `model_sum.rs:115`, `model.rs:515`"; `:461` "~37" → "**21**"; `:466-468` "36" → "**18**"; `:500`
   "36" → "**18**"; `:542` "all 19" → "all **23**"; `:345` "(4)"/"(3)" → "(**2**)"/"(**2**)";
   `:370`/`:489` `model_sum.rs:120` → `:115`. **(P6-1)**
2. **F08 `:406`:** drop the dangling "(see the blast-radius block above)" and either delete "across 11
   files" or correct it to "across **9** files". **(P6-2)**
3. **F08 `:540-543`:** replace the completion-evidence grep with the corrected three-crate,
   qualified-path form plus the "expected residue: exactly one line — `boosting.rs:5476`,
   `cb_train::Model`, DO NOT migrate" note, and make the clean `cargo build --workspace --all-targets`
   the primary evidence. **(P6-3)**
4. **F08 `:316-318`:** make the STOP-AND-REPORT trigger numeric — "fewer than 18 or more than 28
   `E0639`; fewer than 18 or more than 25 `E0063`". **(P6-4)**
5. **E04/E06/E07/E08:** `pub(crate) fn` → `pub fn`. **(P6-5)**
6. **E22 test fn 4:** state the denominator observation channel or drop the two lines, and add
   "Do NOT add a denominator field to `CtrFeatureColumn` (it breaks `tree_test.rs:374`, `:662`)".
   **(P6-6)**
7. **E11:** delete "threaded as a constant `false` here" (`PLAN-W2-W3.md:699`); `true` everywhere.
   **(P6-6)**
8. **E19:** name `crates/cb-model/src/ctr_data.rs:495-505` as a third doc block to update.
   **(P6-6)**
9. **E18 / E23 / F12:** append the missing IDs to the three "Blocked by:"/"Blocks:" cells.
   **(P6-6, cosmetic)**
10. **E22 Files / E09 Parallelizable:** "BOTH `materialize_ctr_feature` call sites" → "the single call
    site inside `materialize_ctr_columns_for_perm`, plus that helper's two call sites"; name
    `boosting.rs` in E09's "owns" clause. **(P6-7)**

---

### Unverified Items

1. `cargo build --workspace --all-targets` was **not executed** this session (a full workspace build
   against a recorded `target/`-disk-exhaustion hazard). The `E0063`/`E0639` site counts (21 / 23) are
   derived by exhaustive type-classified enumeration of every `Model {`-shaped token in the workspace,
   not by rustc. F08's procedure makes rustc authoritative, so this does not affect the verdict.
2. Whether the 30-row isolating configs yield a winning split of each intended type — data-dependent;
   falsified at generation time by each generator's anti-false-pass assertion.
3. The upstream `CTRBlob` mean stride — `[INFERRED]`, with the mandatory 8 → 12 → STOP-AND-REPORT
   probe and E18's upstream-produced `.cbm` as the falsifier.
4. Whether E23's fixture can discriminate `Full` from `SkipTest` while both land within 1e-5 — a
   pre-written deferral path exists.
5. Whether `catboost==1.2.10` is byte-deterministic for categorical-only fixtures — falsified by each
   fixture task's double-generation `diff -r`.
6. `bench/one_hot_gpu_speed/one_hot_bench_colab.py` end-to-end behavior after F09 — only the
   source-level preflight is testable locally.
7. Pass-4 item 6 (facade-wrapper alternative to F08) — **CLOSED** by the locked decision; out of scope.
