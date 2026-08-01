## Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Make four inert CTR params (`simple_ctr`, `simple_ctr_priors`, `combinations_ctr` TYPE, `counter_calc_method`) genuinely honored by the training engine, then route `cat_features`/CTR through the `CatBoostBuilder` facade and the Python bindings, at ≤1e-5 vs `catboost==1.2.10`.
**Plan:** `.planning/plans/ctr-type-engine-and-facade-routing/{PLAN.md, PLAN-W2-W3.md, PLAN-W4-W5.md, PLAN-PART2.md, PLAN-COVERAGE.md}` — 48 tasks (E00–E23, F00–F23)
**Pass:** 3 of 3 (final). Pass 1 = ISSUES_FOUND (5 CRITICAL, 5 MAJOR, 5 MINOR); pass 2 = ISSUES_FOUND (1 CRITICAL, 2 MAJOR, 6 MINOR)

---

### Summary

- **All 9 pass-2 required revisions were applied, and 8 of them are VERIFIED correct against the working tree this session** (anchors re-derived by direct read + grep, not taken from the plan's own claims). The `cat_eligible_buckets` correction, the `Full` bucket-space ownership move to E22, the `find`-as-existence-gate wording, the four new edges, the 10→11 reconciliation, the `:4669` / `:5464` / `HashMap<u64,u32>` anchor fixes, the defensive-not-reachable E11 rewording and the `is_mean()` codec constraint are all present and all factually correct.
- **Revision #1 (NEW-1, the CRITICAL) is PARTIAL, and the residue is itself a CRITICAL.** The fix enumerated **two** of the **four** compile-forced edit groups in `crates/cb-train/tests/ctr_split_scoring_test.rs`. Verified this session: that file **also** calls `materialize_ctr_feature` at `:384` and `:394` (7 args — E09 widens to 9, E22 widens again) and `bake_ctr_table` at `:542`, `:576`, `:645` (7 args — E11 widens to 8, E22 widens again). **E11 and E22 do not list the file at all**, and E09's instruction for it is "add one field initializer per literal **and nothing else**", while E16 says "if a call no longer compiles for any reason other than the dropped argument, **STOP AND REPORT**". As written, one of the eleven SPEC-CTRT-18 oracle targets cannot compile at E09, E11 or E22, with no authorized fix.
- **A second, related MAJOR:** the new zero-diff carve-out ("**ten** of these eleven files must show `git diff --stat` printing NOTHING; `ctr_split_scoring_test.rs` is the exception") is factually false for two more of the eleven: **`crates/cb-train/tests/ctr_feature_materialize_test.rs`** is an explicit Files entry of **E09 and E22**, and **`crates/cb-model/tests/ctr_data_roundtrip_test.rs`** is an explicit Files entry of **E11**. The gate contradicts three tasks' own Files lists under both the per-task and the cumulative reading.
- Everything else found this pass is MINOR: two unstated observation channels for assertions on `train_inner` locals, a `pub(crate)` vs `pub use` re-export inconsistency, and three "Blocks:" cells that under-list edges the edge list now carries.
- **Locked user decisions (1)–(4) and every project constraint remain honored.** Re-checked: engine-before-facade (`E23->F00/F01/F07/F19` + Part 2's prose gate); the `.cbm` mean lift with E00's pre-change hand-constructed non-mean baseline; the scalar `simple_ctr: ECtrType` untouched with the multi-description gap documented by F00; `one_hot_train/` reused read-only with no corpus generation; source/test separation; no `unwrap`/`expect`/`panic`/indexing in production; no `--features rocm` without `--no-default-features`; `crates/cb-oracle/generator/gen_fixtures.py` never invoked; `catboost-master/` never cited as a parity reference; the accepted failing-test baseline used as a "no previously-passing target may fail" gate; the uncommitted one-hot wave never reverted and its three oracles mandatory for every `boosting.rs`/`tree.rs` task.

---

### Pass 1 disposition

| # | Pass-1 finding | Status after pass 3 |
|---|---|---|
| 1 | CRITICAL-1 bake copy-back owned by no task | **VERIFIED** — E15 owns `boosting.rs:5437-5473`; copy-back re-keyed on `(projection, ctr_type)`, `spec.prior_num`/`prior_denom` assignments deleted, `shift`/`scale` derived per split; E15 test fn 3 pins distinct `prior_num` AND distinct `scale` |
| 2 | CRITICAL-2 wrong bake de-dup key | **VERIFIED** — `(projection, ctr_type)` only; explicit `target_border_idx` prohibition in `ctr_base_key`/bake key; E11 test fn 4's `tables.len() == 1` + `== 2` complement; rationale now correctly stated as defensive (pass-2 NEW-8 applied) |
| 3 | CRITICAL-3 E20 breaks an unlisted green test + a second doc block | **VERIFIED** — `ctr_data_test.rs:197-212` INVERT instruction; both doc blocks (`:756-758`, `# Errors` at `:801-806`) |
| 4 | CRITICAL-4 eval-set bucket space undefined | **VERIFIED** — E22 now owns `crates/cb-train/src/ctr/ctr_feature.rs` with the explicit `extra_cat_columns` widening; E21 step 5 is specification-only (pass-2 NEW-3 applied) |
| 5 | CRITICAL-5 inverted F09/F08 order | **VERIFIED** — `F06->F08->F09` in the edge list, the W6 lane, the W7 header and both task bodies |
| 6 | MAJOR-6 W0 graph vs edge list | **VERIFIED** — `E01->E02`, `E03->E02`, `E13->E18` present; graph acyclic (re-derived below) |
| 7 | MAJOR-7 `staged_predict` silently wrong on one-hot | **VERIFIED** — `ensure_scalar_oblivious` one-hot arm (F11), one-hot case in `staged_predict_facade_test` (F12), F14's "extending is required, weakening is forbidden" |
| 8 | MAJOR-8 vacuous E12 permutation test | **VERIFIED** — assertion on the materialized column + the fn-3b Borders anti-vacuity companion |
| 9 | MAJOR-9 600 MB `2^24` fixture | **VERIFIED** — seeded `TCtrMeanHistory` accumulator differential with the `assert_ne!` guard; SPEC §7 A2 note |
| 10 | MAJOR-10 `target_border_idx` route under-specified | **VERIFIED** — `tree.rs:3296` reads `column.target_border_idx`, `tree.rs:3237` param and `boosting.rs:4662` literal `0` deleted (all three re-confirmed verbatim this session); the erroneous `cat_eligible_buckets` lockstep constraint is retracted (pass-2 NEW-2 applied) |
| 11 | MINOR line numbers | **VERIFIED** (`:4662` confirmed to be the literal `0`; `:4669` the `&cat_eligible_buckets` argument) |
| 12 | MINOR 10→11 oracle reconciliation | **VERIFIED** — `PLAN-COVERAGE.md:30` and R1 now say 11 |
| 13 | MINOR Validation blocks vs stated scope | **VERIFIED** — E02's Validation now appends `cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test` |
| 14 | MINOR stale §6.1 blocker | **VERIFIED** |
| 15 | MINOR weak F22 marker test | **VERIFIED** |
| 16 | Extra: E19 stride probe | **VERIFIED** — 8 → 12 → STOP AND REPORT; layout `[INFERRED]` |
| 17 | Extra: F16 `combinations_ctr = []` | **VERIFIED** — binary resolution, "leaving it undefined is forbidden" |

---

### Pass 2 disposition

| # | Pass-2 required revision | Status | Evidence checked this session |
|---|---|---|---|
| **NEW-1** | CRITICAL — own the compile-forced edits in `ctr_split_scoring_test.rs` / `tree_test.rs`; carve the zero-diff gate | **PARTIAL** | E09 Files now name `tree_test.rs:374,:662` and `ctr_split_scoring_test.rs:41,:68` with "CHANGE NO ASSERTION" ✓ (literals confirmed at exactly those lines); E09's EXPECTED INITIAL FAILURE names both `E0061` and `E0063` ✓; E16 Files name the five `greedy_tensor_search_oblivious_with_ctr` sites `:99,:148,:191,:249,:305` ✓ (confirmed); E15/E16 carve the gate to TEN files ✓. **BUT** the enumeration misses `materialize_ctr_feature` at `:384,:394` (E09, E22) and `bake_ctr_table` at `:542,:576,:645` (E11, E22), and misses two more of the eleven files entirely — see **P3-1** and **P3-2** |
| **NEW-2** | MAJOR — replace the `cat_eligible_buckets` lockstep constraint | **VERIFIED** | E15 (`PLAN-W2-W3.md:1120-1128`), E16 (`:1358-1367`, `:1468-1475`) and R11 all carry the per-eligible-cat-feature / `.max()` / MUST-NOT-grow statement; the `assert_eq!(cat_eligible_buckets.len(), ctr_features.len())` is deleted and explicitly retracted (`:1406-1410`); R11 scoped to `structure_fold_columns` + `averaging_ctr_features` ✓. Code re-confirmed: built at `boosting.rs:3074` from `eligible_absolute`, passed at `:4669`, consumed by `.max()` at `tree.rs:3026` |
| **NEW-3** | MAJOR — give the `Full` remap an owner and a reachable test | **VERIFIED** (one MINOR residue, P3-4) | E22 Files include `crates/cb-train/src/ctr/ctr_feature.rs` with Green step 0: `extra_cat_columns: &[Vec<String>]`, folded into the `HashMap<u64,u32>` remap **after** the learn documents, output indexed by the learn slice, threaded from `boosting.rs:3238` and `:3274` (both confirmed as the only two production call sites); test fn 4 moved to `crates/cb-train/tests/ctr_feature_materialize_test.rs` with the Run/Validation commands pointing there; E21 step 5 marked "a SPECIFICATION, consumed by E22 … produces NO code in E21" ✓ |
| **NEW-4** | MINOR — state the `find`'s post-fix role | **VERIFIED** | E15 Green 2(a) `PLAN-W2-W3.md:1216-1229`: the `find` "SURVIVES, purely as an existence gate", written as `.any(...)`, with `BakedCtrTable.{shift,scale,prior_num,prior_denom}` recorded informational-only and "MUST stay" |
| **NEW-5** | MINOR — four missing edges | **VERIFIED** | `PLAN.md:352` `E23->F01 E23->F07 E23->F19`; `:359` `F14->F20`. Graph re-derived: still acyclic |
| **NEW-6** | MINOR — 10→11 and E02's Validation | **VERIFIED** | `PLAN-COVERAGE.md:30` "all 11 single-prior oracles unchanged"; R1 "all 11 single-prior oracles"; `PLAN.md:699` adds the two cb-model targets |
| **NEW-7** | MINOR — anchor corrections | **VERIFIED** | `:4668`/`:4670` → `:4669` everywhere; `find(...)` cited at `:5464` (confirmed: `:5459` is `for tree in &mut trees {`); "first-seen `HashMap<u64,u32>` remap" everywhere, no `PerfectHash` wording survives |
| **NEW-8** | MINOR — E11 rationale defensive, not reachable | **VERIFIED** | E11 Green step 5 and R12 both say "the multi-type case is not reachable today"; test fn 4 kept as the pin |
| **NEW-9** | MINOR (codec) — key on `ctr_type.is_mean()` | **VERIFIED** | E19 Refactor constraint `PLAN-W4-W5.md:233-242` and E20 `:398-408`, both mandatory, both naming the `FloatTargetMeanValue` test / `BinarizedTargetMeanValue` fixture asymmetry |

---

### CodeGraph / verbatim evidence (pass 3, re-derived)

- `CtrFeatureColumn` — `crates/cb-train/src/ctr/ctr_feature.rs:69`; external struct literals at `crates/cb-train/src/tree_test.rs:374`, `:662`, `crates/cb-train/tests/ctr_split_scoring_test.rs:41`, `:68`. **Exactly the four E09 now owns.** ✓
- `materialize_ctr_feature` — `crates/cb-train/src/ctr/ctr_feature.rs:124`, **7 parameters** (`cat_columns, projection, permutation, target_class, prior_num, prior_denom, ctr_border_count`); first-seen remap is the local `HashMap<u64,u32>` beginning at `:183`; `ctr_type: ECtrType::Borders.as_i8()` hard-code at `:232`; `bucket_count: remap.len()`.
  - Callers: production `crates/cb-train/src/boosting.rs:3238`, `:3274` ✓ (exactly the two E22 threads) — **plus `crates/cb-train/tests/ctr_split_scoring_test.rs:384`, `:394` and `crates/cb-train/tests/ctr_feature_materialize_test.rs:73`, `:122`, `:189`.** Load-bearing for **P3-1**.
- `bake_ctr_table` — `crates/cb-train/src/ctr/bake.rs:110`, 7 parameters. Test callers: **`crates/cb-train/tests/ctr_split_scoring_test.rs:542`, `:576`, `:645`** (verbatim: `bake_ctr_table(&cat_columns, &proj, &target_class, 2, ctr_border_count_default(), PRIOR_NUM, PRIOR_DENOM)`). Load-bearing for **P3-1**.
- `build_final_ctr` — test callers `crates/cb-model/tests/ctr_data_roundtrip_test.rs:101,138,143,163` and `crates/cb-train/src/ctr/final_ctr_test.rs:20,29,36,44,55,56,63,72`. **Both files ARE in E11's Files list** ✓ — but `ctr_data_roundtrip_test.rs` is one of the eleven oracles, which breaks the "ten files, zero diff" claim (**P3-2**).
- `greedy_tensor_search_oblivious_with_ctr` — `crates/cb-train/src/tree.rs:3228`; `target_border_idx: usize` at `:3237`; `CtrSplitSpec { … target_border_idx, … }` at `:3296`; production call `crates/cb-train/src/boosting.rs:4653` with the literal `0` at `:4662`; `&cat_eligible_buckets` at `:4669`; test calls at `ctr_split_scoring_test.rs:99,148,191,249,305`. **All exactly as the plan now states.** ✓
- `cat_eligible_buckets` — `crates/cb-train/src/boosting.rs:3074`, `eligible_absolute.iter().map(|&abs_idx| … perfect_hash_bins(&as_str))`; a **local inside `train_inner` (`:2555`)**; consumed at `crates/cb-train/src/tree.rs:3026` by an order-insensitive `.max()`. The plan's constraint is correct; its **test channel** is not stated (**P3-3**).
- `structure_fold_columns` (`boosting.rs:3201`), `materialized_ctr_features` (`:3258`), `averaging_ctr_features` (`:3270`) — **also `train_inner` locals**, not observable from `boosting_test.rs` without extracting a helper (**P3-3**).
- The bake block — `crates/cb-train/src/boosting.rs:5437-5473`, read verbatim: `let mut baked` `:5437`; `seen` de-dup `:5440-5443`; `bake_ctr_table(…)` `:5445-5453` with `ctr_prior_num` at `:5451`; copy-back comment `:5458`; `for tree in &mut trees` `:5459`; `.find(|t| t.projection == spec.projection)` **`:5464`**; `spec.prior_num = table.prior_num;` `:5468`. **All pass-2 anchor corrections confirmed.** ✓
- `crates/cb-train/src/lib.rs:45-52` and `crates/cb-train/src/ctr/mod.rs:138-148` re-export **`pub`** items (`online_ctr_prefix_binclf`, `materialize_ctr_feature`, …). E04/E06/E07/E08 specify `pub(crate) fn` **and** addition to those `pub use` lists — mutually exclusive (**P3-5**).

---

### Issues

#### [CRITICAL] P3-1 — The NEW-1 ownership fix enumerates 2 of 4 compile-forced edit groups; `ctr_split_scoring_test.rs` still cannot compile at E09, E11 and E22

- **Plan location:** PLAN-W2-W3.md E09 "Files" + "NOTE (both files)"; E11 "Files"; PLAN-W4-W5.md E22 "Files"; PLAN.md §3.2 carve-out; E15/E16 Completion evidence
- **Requirement:** SPEC-CTRT-06/-07/-08/-13/-17, SPEC-CTRT-18, acceptance A7
- **Evidence (all re-read verbatim this session):**
  - `crates/cb-train/tests/ctr_split_scoring_test.rs:26` imports **`bake_ctr_table, greedy_tensor_search_oblivious_with_ctr, materialize_ctr_feature`**.
  - It calls `materialize_ctr_feature` with **7 arguments** at `:384` and `:394`. **E09 widens that function to 9** (`+ ctr_type, + target_border_idx`) and **E22 widens it again** (`+ extra_cat_columns`). Neither task's instruction for this file mentions these sites; E09 says the permitted edits are "add one field initializer per literal **and nothing else**".
  - It calls `bake_ctr_table` with **7 arguments** at `:542`, `:576`, `:645`. **E11 widens that function to 8** (`+ ctr_type`; E11's own EXPECTED INITIAL FAILURE is `E0061: takes 7 arguments but 8 were supplied`) and **E22 widens it again** (`+ counter_calc_skip_test` + the eval cat columns). **Neither E11 nor E22 lists `ctr_split_scoring_test.rs` in Files at all.**
  - E16 additionally instructs: "If a call no longer compiles for any reason other than the dropped argument, **STOP AND REPORT** rather than adjusting the test."
  - E09's Parallelizable note asserts "no later task re-edits them except E16's argument drop" — **false**: E11 and E22 both do.
- **Failure scenario:** At E09 the executor adds two `target_border_idx: 0` initializers, then `cargo test -p cb-train --test ctr_split_scoring_test` fails to build with two `E0061`s at `:384`/`:394` that no instruction authorizes fixing. Same at E11 (three `E0061`s at `:542`/`:576`/`:645`, in a file E11 does not own) and again at E22. In each case the executor either stalls, or "fixes" the break by deleting the offending construction — silently removing the permutation-difference and bake assertions from a CTR regression oracle during the highest-risk waves — or edits it anyway, at which point E15/E16's completion gate ("only the mechanical field/argument edits, ZERO assertion changes … E09's two initializers and E16's five dropped arguments") is provably unsatisfiable because five further edit sites exist.
- **Impact:** One of the eleven SPEC-CTRT-18 targets — the most CTR-split-specific one — is un-buildable through W2–W5, so the A7 firewall runs at 10/11 exactly when tie-breaks are being changed. Risk of silent coverage loss.
- **Required revision:**
  1. Add to **E09's** `ctr_split_scoring_test.rs` entry: "…and add the two new arguments (`ctr_type: ECtrType::Borders`, `target_border_idx: 0`) to the `materialize_ctr_feature` calls at `:384` and `:394`, preserving today's behavior. **CHANGE NO ASSERTION.**"
  2. Add `crates/cb-train/tests/ctr_split_scoring_test.rs` to **E11's** Files: "pass the resolved `ECtrType::Borders` to the `bake_ctr_table` calls at `:542`, `:576`, `:645`. **CHANGE NO ASSERTION.**"
  3. Add `crates/cb-train/tests/ctr_split_scoring_test.rs` to **E22's** Files: "pass an empty `extra_cat_columns` slice at `:384`, `:394` and `counter_calc_skip_test = true` at `:542`, `:576`, `:645` — the `SkipTest` default, byte-identical behavior. **CHANGE NO ASSERTION.**"
  4. Restate the PLAN.md §3.2 carve-out and E15/E16's completion evidence over the **complete** permitted-edit set for that file: 2 struct-literal initializers (E09), 2 `materialize_ctr_feature` argument additions (E09) + 2 more (E22), 5 dropped `greedy_tensor_search_oblivious_with_ctr` arguments (E16), 3 `bake_ctr_table` argument additions (E11) + 3 more (E22). Gate: **only** signature-driven argument/field edits, **zero** assertion changes.
  5. Delete E09's false claim that no later task re-edits the file.

#### [MAJOR] P3-2 — The "TEN files, zero diff" claim is false for two more of the eleven oracles

- **Plan location:** PLAN.md §3.2 "ZERO-DIFF CARVE-OUT (one file only)"; E15 and E16 Completion evidence; PLAN-COVERAGE SPEC-CTRT-18 row and R1
- **Requirement:** SPEC-CTRT-18, acceptance A7
- **Evidence:** `crates/cb-train/tests/ctr_feature_materialize_test.rs` is an explicit **Files** entry of **E09** ("Modify … (existing target)", plus a whole new test fn) **and of E22** ("existing target; it is where test fn 4 lives"). `crates/cb-model/tests/ctr_data_roundtrip_test.rs` is an explicit **Files** entry of **E11** (test fns 2 and 4) — and its `build_final_ctr` calls at `:101,138,143,163` are compile-forced by E11's signature change. Both files are in the eleven (PLAN.md §3.2, SPEC-CTRT-18).
- **Failure scenario:** At E09/E11/E22 the task's own Files list orders an edit that §3.2 forbids; at E15/E16 the completion evidence demands `git diff --stat` over "the ten" print nothing, which cannot hold under either the per-task or the cumulative baseline. The executor must invent an interpretation of a gate that PLAN.md calls authoritative.
- **Impact:** No wrong production code, but the central regression gate of Part 1 is self-contradictory at three points; an executor may relocate legitimate new tests out of the reusable targets, or record a false "gate satisfied".
- **Required revision:** Restate the carve-out as: *"The gate is **no modification of any EXISTING assertion** in the eleven files. Three of them are legitimately edited by owning tasks — `ctr_split_scoring_test.rs` (E09/E11/E16/E22, signature-driven only), `ctr_feature_materialize_test.rs` (E09 and E22, ADDITIVE new test fns), `ctr_data_roundtrip_test.rs` (E11, ADDITIVE test fns 2/4 + the compile-forced `build_final_ctr` argument). For the remaining **eight**, `git diff --stat` must print nothing."* Update PLAN-COVERAGE's SPEC-CTRT-18 row and R1 identically.

#### [MINOR] P3-3 — E15 test fn 2 and E16 test fn 1 assert on `train_inner` locals with no stated observation channel; the `cat_eligible_buckets` pin is vacuous if re-derived

- **Plan location:** PLAN-W2-W3.md E15 Red test fn 2; E16 Red test fn 1 ("capture the `Vec<Vec<u32>>` as built at `crates/cb-train/src/boosting.rs:3074` and compare it element-for-element…")
- **Evidence:** `cat_eligible_buckets` (`:3074`), `structure_fold_columns` (`:3201`), `materialized_ctr_features` (`:3258`) and `averaging_ctr_features` (`:3270`) are all `let` bindings **inside `train_inner` (`:2555`)**. A child-module test in `boosting_test.rs` can reach private *items*, not function locals. If the test re-derives `cat_eligible_buckets` from `eligible_absolute` + `cat_columns`, the comparison is tautological (the expression does not depend on the prior/border expansion at all), so the "no-growth pin" can never fail and is not a guard under §3.1.
- **Impact:** A verification-mechanism gap on R11's "most fragile thing W3 can break". Not behavior-threatening: the no-growth constraint is stated four times in prose and no Green step touches `:3074`.
- **Required revision:** State the channel once, in E15: *"extract the expansion into a private helper in `boosting.rs` (E15 owns the file) — e.g. `fn materialize_ctr_columns(cat_columns, projections, perm, target_class, priors, borders, …) -> CbResult<Vec<CtrFeatureColumn>>` — and assert on its return value; `cat_eligible_buckets` stays where it is and is pinned by asserting `cat_eligible_buckets.len() == <count of CTR-eligible cat features>` **while** `ctr_features.len()` is the expanded product, which is the exact inverse of the retracted invariant."*

#### [MINOR] P3-4 — E22 test fn 4's two denominator assertions have no observation channel on `CtrFeatureColumn`

- **Plan location:** PLAN-W4-W5.md E22 Red test fn 4 (`assert_eq!(denominator_full, 3, …)`, `assert_eq!(denominator_skiptest, 2)`)
- **Evidence:** `CtrFeatureColumn` (`crates/cb-train/src/ctr/ctr_feature.rs:69-94`) carries `projection, ctr_type, prior_num, prior_denom, bins, ctr_value, bucket_count` (+ `target_border_idx` after E09). **No denominator field**, and E22's Green step 0 does not add one. The other three assertions (`bucket_count_full == 3`, `bucket_count_skiptest == 2`, `column_full.bins.len() == 3`) are expressible ✓.
- **Impact:** Low — the MAX-denominator behavior is already gated by E22 test fn 1 (`ctr/online_test.rs`, `denominator 2` vs `4`) and by E23 test fn 4 (`denominator_full > denominator_skiptest`), so no coverage is lost. The hazard is that an executor "fixes" the gap by adding a field to `CtrFeatureColumn`, which re-breaks the four external literals in files E22 does not own.
- **Required revision:** Either drop the two denominator lines from test fn 4 (pointing at test fn 1), or state the channel: *"assert the denominator by calling the re-exported `online_counter_column(&column_full.bins, &[2,2,2], column_full.bucket_count)` — the eval bins are deterministic because the remap appends extras after the learn documents."* **Add: "Do NOT add a denominator field to `CtrFeatureColumn`."**

#### [MINOR] P3-5 — `pub(crate) fn` producers are simultaneously ordered into `pub use` re-export lists

- **Plan location:** E04, E06, E07, E08 (Green: "One `#[must_use] pub(crate) fn`"; Files: "add … to the existing `pub use online::{…}` re-export block at `crates/cb-train/src/ctr/mod.rs:144-148`; and to `crates/cb-train/src/lib.rs:46-50`")
- **Evidence:** `crates/cb-train/src/ctr/mod.rs:144-148` and `crates/cb-train/src/lib.rs:45-52` re-export **`pub`** items. Re-exporting a `pub(crate)` fn with `pub use` is `error[E0364]`.
- **Impact:** A compile error on the first Green step of E04, trivially self-correcting (make the fn `pub`, matching `online_ctr_prefix_binclf`). Also note E22 test fn 4's suggested channel (P3-4) needs these public.
- **Required revision:** Change "`pub(crate) fn`" to "`pub fn` (matching `online_ctr_prefix_binclf`'s visibility, since the Files list re-exports it)" in E04/E06/E07/E08.

#### [MINOR] P3-6 — Three "Blocks:"/"Blocked by:" cells under-list edges the (authoritative) edge list carries

- **Plan location:** E18 ("Blocked by: E13" — the list also has `E11->E18`); E23 ("Blocks: F00" — the list also has `E23->F01/F07/F19`); PLAN-PART2 F12 row ("Blocks: F20" — the list also has `F12->F13`)
- **Impact:** None on ordering — every omission is an *under*-statement, and the edge list is declared authoritative; the W4 wave text already names E11. Cosmetic.
- **Required revision:** Append the missing IDs to the three cells.

#### [MINOR] P3-7 — Remap span cited as `ctr_feature.rs:183-196`; the loop actually ends at `:198`

- **Evidence:** the `// 2. Remap combined keys…` comment is at `:183`; `combined_bins.push(bin);` / closing brace land at `:197-198`.
- **Impact:** None — unambiguous in context. Cosmetic.

---

### Implementation Order Review

Edge list re-derived from `PLAN.md:345-359` and checked against all 48 task bodies:

```
E01->E02  E03->E02  E01->E04  E04->E05  E05->E06  E06->E07  E07->E08  E08->E09
E09->E10  E02->E10  E03->E10  E10->E11  E11->E12  E11->E13  E11->E15  E14->E15
E05->E15  E15->E16  E16->E17  E11->E18  E13->E18  E18->E19  E19->E20  E00->E20
E11->E21  E21->E22  E22->E23  E23->F00  E23->F01  E23->F07  E23->F19
F00->F03  F00->F05  F00->F16  F01->F02  F02->F03  F03->F04  F04->F05  F05->F06
F06->F08  F08->F09  F09->F10  F10->F11  F11->F12  F12->F13  F11->F14
F05->F15  F07->F15  F15->F16  F09->F17  F11->F17  F15->F18  F16->F18  F17->F18
F18->F20  F19->F20  F12->F20  F14->F20  F20->F21  F21->F22  F22->F23
```

- **Acyclic.** The only ID-descending edge is `E03->E02`; `E03` has no in-edges and `E02`'s sole out-edge is `E02->E10`, from which no path returns. The four edges added this round (`E23->F01`, `E23->F07`, `E23->F19`, `F14->F20`) all advance the wave order.
- **Blocked-by/Blocks reconciliation:** every task's Blocked-by set is a subset of its in-edges and every in-edge is justified by a body; the only mismatches are the three under-listed "Blocks" cells in P3-6.
- **File-ownership serialization spot-checks:** `boosting.rs` is serialized E02 → E10 → E15 → E16 → E21 → E22 ✓. `online.rs` E04 → E05 → E06 → E07 → E08 ✓. `ctr_feature.rs` E09 → (nine tasks) → E22 ✓ — **no collision**: E09 lands in W2, E22 in W5, connected by `E09->E10->E11->E21->E22`, and E22's own note states this correctly. `crates/cb-model/src/ctr_data.rs` E11 → E19 → E20 ✓. `catboost-rs/src/model.rs` F10 → F11 → F12 → F13 ✓. `params.rs` F15 → F16 ✓.
- **The only order-relevant correction required** is P3-1: three tasks (E09, E11, E22) must take ownership of the compile-forced edits in `crates/cb-train/tests/ctr_split_scoring_test.rs`. These are Files-list additions, not new tasks, and change no edge.

---

### Potential Bugs

- **Un-buildable oracle target (P3-1).** Trigger: E09's `materialize_ctr_feature` widening / E11's and E22's `bake_ctr_table` widening. Failure mode: `E0061` × 5 in `ctr_split_scoring_test.rs` with no authorized fix; worst branch is assertion deletion in a CTR regression oracle. Mitigation: the four Files-list additions above.
- **Vacuous no-growth guard (P3-3).** Trigger: re-deriving `cat_eligible_buckets` inside the test. Failure mode: a green test that cannot fail; the `model_size_reg` cat-weight regression it was added to catch would pass unnoticed. Mitigation: the helper-extraction channel, or the inverse-length assertion.
- **`CtrFeatureColumn` field creep (P3-4).** Trigger: an executor adding a denominator field to satisfy test fn 4. Failure mode: `E0063` again at `tree_test.rs:374,662` and `ctr_split_scoring_test.rs:41,68` in files E22 does not own. Mitigation: the explicit prohibition.
- **`is_mean()` covering both mean types** — now an explicit MANDATORY constraint in both E19 and E20, with the `FloatTargetMeanValue`-vs-`BinarizedTargetMeanValue` asymmetry spelled out. Closed.
- **Divergent remaps between Counter totals and Counter bins** — closed by E22 Green step 0's single-remap rule plus the explicit FORBIDDEN clause on a second bucket space inside `online_counter_column`.
- **f32-vs-f64 quantization for non-BTMV types** — carried, correctly recorded as a limitation, not a plan defect.

---

### Compliance Checks (all PASS)

- **Locked decisions:** (1) engine before facade — `E23->F00/F01/F07/F19` plus "Part 2 starts only after E23 is green" ✓; (2) `.cbm` mean lift with a non-mean byte gate — E00's hand-constructed pre-change baseline + E20's stride-mutation falsifiability ✓; (3) CTR type + full prior list honored, scalar `simple_ctr: ECtrType` unchanged (no `BoostParams` field added or retyped, 62 sites untouched), multi-description gap documented by F00 with a doc-presence test ✓; (4) `crates/cb-oracle/fixtures/one_hot_train/` reused read-only by F19/F20, no corpus-wide generation ✓.
- **Source/test separation:** every unit test is a sibling `<name>_test.rs` mounted via `#[cfg(test)] #[path = …] mod`; integration tests live in `crates/<crate>/tests/` ✓.
- **No `unwrap`/`expect`/`panic`/indexing in production:** every Green step mandates checked `.get`, `saturating_*`, typed `CbError`/`ModelError` ✓.
- **Backend trap:** no command passes `--features rocm` without `--no-default-features`; the only `--features` use is `maturin develop … --features cpu` ✓.
- **`crates/cb-oracle/generator/gen_fixtures.py`:** never invoked; all six new fixtures use the fixture-local pattern with the corpus-cleanliness guard ✓.
- **`catboost-master/`:** never cited as a parity reference; only as the reason F18's registry test currently SKIPs ✓.
- **Accepted failing-test baseline:** used as a "no previously-passing target may fail" gate by E23 and F23, with the flaky and pre-existing-failure caveats recorded ✓.
- **Uncommitted one-hot wave:** never reverted; `git checkout --` forbidden in every mutation revert; `one_hot_oracle_test`, `one_hot_draw_accounting_test`, `device_one_hot_parity_test` are mandatory regression scope for E02, E05, E10, E15, E16, E21, E22 and the F23 closing gate ✓.

---

### Required Plan Revisions

1. **E09 + E11 + E22 + PLAN.md §3.2 + E15/E16 completion evidence:** own the remaining compile-forced edits in `crates/cb-train/tests/ctr_split_scoring_test.rs` — `materialize_ctr_feature` at `:384`, `:394` (E09, again E22) and `bake_ctr_table` at `:542`, `:576`, `:645` (E11, again E22) — and restate the carve-out over the complete permitted-edit set. Delete E09's "no later task re-edits them" claim. **(CRITICAL, P3-1)**
2. **PLAN.md §3.2 + PLAN-COVERAGE SPEC-CTRT-18/R1 + E15/E16:** rewrite the zero-diff gate as "no modification of any EXISTING assertion", naming the three legitimately-edited oracle files and requiring an empty `git diff --stat` over the remaining **eight**. **(MAJOR, P3-2)**
3. **E15/E16:** state the observation channel for the expansion assertions (a private helper in `boosting.rs`) and make the `cat_eligible_buckets` pin falsifiable (length vs the CTR-eligible-feature count while `ctr_features` grows). **(MINOR, P3-3)**
4. **E22 test fn 4:** drop the two denominator assertions or state the `online_counter_column` channel; add "Do NOT add a denominator field to `CtrFeatureColumn`". **(MINOR, P3-4)**
5. **E04/E06/E07/E08:** `pub(crate) fn` → `pub fn` for the producers that the same tasks re-export. **(MINOR, P3-5)**
6. **E18 / E23 / F12:** append the missing IDs to the three "Blocked by:"/"Blocks:" cells. **(MINOR, P3-6)**
7. **E21/E22:** `ctr_feature.rs:183-196` → `:183-198`. **(MINOR, P3-7)**

---

### Unverified Items

1. Whether the 30-row isolating configs yield a winning split of each intended type — data-dependent; falsified at generation time by each generator's mandatory anti-false-pass assertion. Correctly recorded.
2. The upstream `CTRBlob` mean stride — `[INFERRED]`, with the mandatory 8 → 12 → STOP AND REPORT probe and an upstream-produced `.cbm` (E18) as the falsifier. Resolved as far as a plan can resolve it.
3. Whether E23's fixture can discriminate `Full` from `SkipTest` while both land within 1e-5 — pre-written deferral path exists; ladder step 0 now points at an implemented rule (E22 Green step 0).
4. Whether `catboost==1.2.10` is byte-deterministic for categorical-only fixtures — falsified by each fixture task's double-generation `diff -r`.
5. `bench/one_hot_gpu_speed/one_hot_bench_colab.py` end-to-end behavior after F09 — only the source-level preflight is testable locally; F22 gates a call-position marker paired with F09 test fn 1.
6. `ensure_scalar_oblivious` has no covering tests today; F11 extends it and F12 adds the first direct coverage. Acceptable — F12 should assert the pre-existing CTR arm alongside the new one-hot arm.
