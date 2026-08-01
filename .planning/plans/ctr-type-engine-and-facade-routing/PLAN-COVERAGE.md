---
title: TDD implementation plan — coverage tables, risks, blockers, attestation
parent: ./PLAN.md
spec: ./SPEC.md
status: ready-for-implementation
task_count: 48
---

# Coverage, risks, blockers, attestation

Companion to `./PLAN.md`, `./PLAN-W2-W3.md`, `./PLAN-W4-W5.md`, `./PLAN-PART2.md`.
**48 tasks total: E00–E23 (24, engine) + F00–F23 (24, facade).**

---

## 1. SPEC-CTRT-01..19 → task coverage

| Spec ID | Title | Task(s) | Primary assertion |
|---|---|---|---|
| **SPEC-CTRT-01** | `ECtrType::target_border_count` | **E01** | `Buckets.target_border_count(2) == 2`; the other three CPU-legal types `== 1`; total `match`, no `_ =>` arm |
| **SPEC-CTRT-02** | `ECtrType::is_cpu_supported` | **E01** | `false` for exactly `FloatTargetMeanValue`/`FeatureFreq`, asserted over an all-six array |
| **SPEC-CTRT-03** | CPU-illegal types typed-rejected before training | **E02** (engine), **F16** (Python surface) | `CbError::Unsupported` naming the type + "not implemented on CPU yet", **before** accumulation; a legal type unaffected |
| **SPEC-CTRT-04** | generic classes-prefix producer | **E04** | 10-row hand-computed table: `Buckets → (N[b], Total)`, `Borders → (Total − Σ_{c≤b} N[c], Total)`; exact `assert_eq!` |
| **SPEC-CTRT-05** | Borders-binclf is the `(classes=2, b=0)` special case — **regression firewall** | **E05** | 169-case exhaustive **bit equality** (`to_bits()`), not a tolerance; + 2 frozen characterization tests; + all 11 CTR oracles |
| **SPEC-CTRT-06** | Buckets online arm | **E08** (unit), **E09** (wiring), **E16** (`ctr_buckets_simple` ≤1e-5) | prefix `N[b]` / prefix `Total` through the UNCHANGED `calc_ctr_online*`; read-before-increment preserved |
| **SPEC-CTRT-07** | BTMV online arm, `Sum` in **f32** | **E07** (unit + the **allocation-free accumulator** differential — a seeded `TCtrMeanHistory { sum: 16_777_216.0f32, count: 16_777_216 }` + one `add(1.0)`, **NOT** a `2^24`-document fixture), **E09** (wiring), **E13** (`ctr_btmv_simple` ≤1e-5; its test fn 3 is a **reporting step, not a gate** — see SPEC §7's A2 note) | `assert_ne!(f32_ref, f64_ref)` anti-vacuity **then** `assert_eq!(hist.sum.to_bits(), 16_777_216.0f32.to_bits())`; distinct `OnlineMeanPrefix` type forbids i64 truncation |
| **SPEC-CTRT-08** | Counter is NOT a prefix | **E06** (unit + permutation invariance), **E09**, **E12** (`ctr_counter_simple` ≤1e-5 + E2E invariance **of the materialized column**) | whole-set bucket count over the constant MAX denominator; **no permutation parameter at all**. E12 test fn 3 asserts permutation-invariance of `CtrFeatureColumn.bins`/`.ctr_value`, **not** of the baked table (a baked-table comparison is VACUOUS — the bake is whole-set and permutation-independent for **every** type including Borders), and test fn 3b is the mandatory anti-vacuity companion proving the SAME comparison FAILS for a Borders column on the same corpus |
| **SPEC-CTRT-09** | `ctr_splits_for_tree` emits the candidate's type | **E03** (covering test FIRST), **E10** | `specs[i].ctr_type` follows `is_simple`; never the hard-coded `Borders` at `boosting.rs:1940` |
| **SPEC-CTRT-10** | prior selection follows `is_simple` | **E10** (unit), **E17** (`ctr_mixed_simple_vs_combo` ≤1e-5) | simple → `simple_ctr_priors`, combination → `combinations_ctr_priors`; fixes `boosting.rs:3155` |
| **SPEC-CTRT-11** | candidate expansion over the FULL prior list | **E14** (fixture first), **E15** | one column per prior in upstream order; `[0.0,0.5,1.0,0.0,0.5,1.0]`; structure/averaging alignment asserted; **all 11 single-prior oracles unchanged** |
| **SPEC-CTRT-12** | candidate expansion over `target_border_idx` | **E16** | both `b=0` and `b=1` emitted for Buckets at binclf; generator asserts `idxs == {0,1}` in `model.json`; `assert_ne!(bins_b0, bins_b1)`. The index reaches `CtrSplitSpec` via **`crates/cb-train/src/tree.rs:3296` reading `column.target_border_idx`**; the whole-tree parameter (`tree.rs:3237`) and its literal `0` argument (`boosting.rs:4662`) are **DELETED**; `ctr_splits_for_tree` keeps the constant `0` as a documented, E03-pinned no-CTR-candidate fallback (its signature `(candidates, priors)` has no column in scope) |
| **SPEC-CTRT-13** | bake path builds per-type final tables | **E11** | per-type `int_counts`/`mean`/`counter_denominator` shapes; `CtrData::from_baked` carries `mean`; frozen Borders bake literals unchanged; bake key = **`(projection, ctr_type)` only**, with E11 test fn 4 pinning `tables.len() == 1` for two Buckets splits at different `target_border_idx` |
| **SPEC-CTRT-14** | `.cbm` ENCODES mean tables | **E20** | `f32 Sum` + `i32 Count` pairs in `CTRBlob`; 8-byte stride pinned; `i32` overflow is a typed `Serialize` error; **BOTH** doc blocks updated (`ctr_data.rs:756-758` **and** the `# Errors` block at `:801-806`); **`ctr_data_test.rs:197-212 encode_ctr_model_parts_rejects_mean_table` is INVERTED into `encode_ctr_model_parts_round_trips_a_mean_table`, never deleted** |
| **SPEC-CTRT-15** | `.cbm` DECODES mean tables, round-trip exact | **E18** (fixture), **E19** (decode + upstream `.cbm` ≤1e-5), **E20** (save→load→save byte identity) | all-zero-`mean` anti-vacuity guard; malformed-blob typed rejection; **the stride-ambiguity branch: try 8, then 12, and STOP AND REPORT if 12 matches — never silently pick one** |
| **SPEC-CTRT-16** | non-mean `.cbm` bytes unchanged — **regression gate** | **E00** (baseline frozen FIRST, hand-constructed model), enforced by **E19/E20** | byte comparison vs a baseline captured **before** the codec change; stride-mutation check proves the two paths are independent |
| **SPEC-CTRT-17** | `counter_calc_method` threaded, testability stated honestly | **E21** (`EvalSet.cat_columns` + `train_cat_with_eval_sets` + **the `Full` bucket-space rule as a SPECIFICATION + doc comment only**), **E22** (**implements** the rule — owns `crates/cb-train/src/ctr/ctr_feature.rs`, widens `materialize_ctr_feature` with `extra_cat_columns` and threads it from `boosting.rs:3238` / `:3274` — plus the flag threading and the eval-only-unseen-category test, which lives in `crates/cb-train/tests/ctr_feature_materialize_test.rs`), **E23** (eval-set fixture **or** recorded deferral) | `assert_ne!(skip, full)` on totals; `denominator_full > denominator_skiptest`; **learn-only test FORBIDDEN**. Under `Full` the first-seen `HashMap<u64,u32>` remap (`crates/cb-train/src/ctr/ctr_feature.rs:183-196`, inside `materialize_ctr_feature`) is built over `learn ++ every eval set`, so an eval-only category gets its OWN bucket and **both `bucket_count` and the `MAX` denominator GROW**, while the learn-document output column stays indexed by the learn slice (E22 test fn 4). "eval-only unseen categorical value ⇒ bucket-space divergence" is **step 0** of E23's localization ladder |
| **SPEC-CTRT-18** | every existing CTR oracle stays green at defaults | **E05** primarily; **regression scope of E02, E04, E05, E06–E11, E15, E16, E22** | all 11 named oracles green, under a **PER-FILE diff gate in three categories** (PLAN.md §3.2; the per-file table is in E15's and E16's Completion evidence). **ZERO DIFF REQUIRED (8):** `plain_ctr_oracle_test`, `ordered_ctr_oracle_test`, `tensor_ctr_oracle_test`, `tensor_ctr_e2e_oracle_test`, `s_order_ctr_bins_oracle_test`, `multi_permutation_e2e_oracle_test`, `multi_permutation_fold_oracle_test`, `fstr_ctr_oracle_test` — `git diff --stat` over them prints nothing. **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES (3):** `ctr_split_scoring_test` (E09 `:41`/`:68` + `:384`/`:394`, E11 `:542`/`:576`/`:645`, E16 `:99,:148,:191,:249,:305`, E22 all five call sites again), `ctr_feature_materialize_test` (E09, E22 — ADDITIVE test fns + widened args), `ctr_data_roundtrip_test` (E11 — ADDITIVE test fns 2/4 + the compile-forced `build_final_ctr` arg at `:101,:138,:143,:163`). **A diff touching an EXISTING assertion in ANY of the eleven is a STOP-AND-REPORT condition** |
| **SPEC-CTRT-19** | multi-description default gap DOCUMENTED | **F00** | doc-presence test asserting the `catboost_options.cpp:439-453` anchor in `params.rs`; the four `*_default()` doc comments updated; "inert here" text deleted |

**All 19 covered. Every engine task references ≥1 SPEC-CTRT ID.**

---

## 2. SPEC-CATF-Δ1..Δ8 → task coverage

| Spec ID | Title | Task(s) | Primary assertion |
|---|---|---|---|
| **Δ1** | CRITICAL-4 is OBSOLETE — verify, do not re-plan | **F09 ONLY** (test fn 2: `fit()`'s one-hot arm emits `ModelSplit::OneHot` for a cardinality-2 column at `one_hot_max_size = 2`). **F01's duplicate `cardinality_two_column_contributes_at_default_one_hot_max_size` is DELETED** — at F01 time `fit()` still calls the float-only `train` and `pool.cat_features()` is never read, so it could not pass | `≥1 ModelSplit::OneHot` in a fitted tree **and** `ctr_data.is_none()`; the one-hot rejection task PLAN-CHECK asked for is explicitly **DROPPED** |
| **Δ2** | BLOCKER-2 superseded; promotion honest **with** the scalar caveat | **F00**, **F03**, **F04**, **F05**, **F16** | the promoted domain is exactly `{Borders, Buckets, BinarizedTargetMeanValue, Counter}`; the single-description limit documented, not hidden |
| **Δ3** | BLOCKER-1 dissolved — reuse `one_hot_train/`, NO corpus generation | **F19**, **F20** | `git status --porcelain crates/cb-oracle/fixtures` clean; `generator/gen_fixtures.py` never invoked (all new Part-1 fixtures use the 9-precedent fixture-local pattern) |
| **Δ4** | width check uses the pool's **DECLARED** cat width | **F08** (stored `cat_feature_count`) → **F09** (set at fit) → **F10** (compare against it) — **in that order**; the previous `F09 → F08` edge was an inverted producer/consumer edge | positive `fit(pool) → predict(same pool)` test with a one-hot-routed trailing column returns `Ok` — the CRITICAL-3 counterexample |
| **Δ5** | all 14 `data_to_pool` sites updated together | **F17** | the crate builds; the four omitted sites (`search.rs:304,389`, `cv.rs:61,64`) explicitly listed and passed `None` |
| **Δ6** | `cv()` / `grid_search()` have defined behavior | **F14** | typed `UnsupportedModel` **before** any fold is fitted; explicit anti-NaN assertion `"grid_search SILENTLY DEGRADED …"` |
| **Δ7** | every public scoring path routes or raises | **F11** (`predict_with`, `needs_cat_columns()`, **and the new one-hot arm in `ensure_scalar_oblivious`**), **F12** (`predict`/`predict_with`/`predict_proba`/**`staged_predict`**), **F13** (`partial_dependence`, `feature_importance_with_data`) | `shap_values` needed **no work** — already fixed by the uncommitted one-hot wave (`ShapUnsupported::{OneHotSplits,CtrSplits}`). **`staged_predict` DID need work:** `ensure_scalar_oblivious` (`crates/catboost-rs/src/model.rs:144-171`) rejects multi-dim / non-symmetric / region / `ctr_data.is_some()` but **NOT `ModelSplit::OneHot`**, and `predict_raw_staged` is float-only — so after F09 a one-hot model was silently mis-scored. F12 adds a one-hot case to `crates/catboost-rs/tests/staged_predict_facade_test.rs` and records the mutation check |
| **Δ8** | the bench preflight passes unedited | **F22** | the marker must appear in a **CALL position** (`\btrain_cat\s*\(` on a non-comment line) — a bare `grep -c "train_cat"` is satisfiable by a comment — **paired with F09 test fn 1's `ctr_data.is_some()`**, **and** `git diff --stat bench/one_hot_gpu_speed/one_hot_bench_colab.py` empty |

**All 8 covered.**

---

## 3. SPEC-CATF-01..17 → task coverage (Part 2's own spec set)

| Spec ID | Task | Spec ID | Task |
|---|---|---|---|
| CATF-01 `one_hot_max_size` setter | **F01** (setter only — the one-hot-contribution test moved to F09) | CATF-10 cat width validation | **F08 → F09 → F10** |
| CATF-02 `max_ctr_complexity` setter | **F02** | CATF-11 `predict_with` CTR-aware | **F11** |
| CATF-03 `simple_ctr` + priors | **F03** | CATF-12 never silently mis-scored (incl. `staged_predict`) | **F12** |
| CATF-04 `combinations_ctr` + priors | **F04** | CATF-13 five CTR kwargs from Python | **F15 + F16** |
| CATF-05 `counter_calc_method` setter | **F05** | CATF-14 `cat_features` fit-kwarg | **F17** |
| CATF-06 default-equivalence | **F06** | CATF-15 registry stays truthful | **F18** |
| CATF-07 float-only no-regression | **F21** | CATF-16 public-API oracle ≤1e-5 | **F20** |
| CATF-08 categorical `fit()` → `train_cat` | **F09** | CATF-17 fixture frozen, not regenerated | **F19** |
| CATF-09 types nameable downstream | **F07** | | |

**All 17 covered.**

---

## 4. Acceptance A1..A13 → task coverage

| # | Scenario | Gate | Delivered by |
|---|---|---|---|
| **A1** | Buckets CTR trains and predicts | `ctr_buckets_simple` ≤1e-5, **both** `target_border_idx` present | **E08, E09, E16** |
| **A2** | BTMV CTR trains and predicts | `ctr_btmv_simple` ≤1e-5 + the **accumulator-level** f32 differential (E07 test fn 2). E13 test fn 3's fixture-scale differential **cannot discriminate for binclf** (f32 and f64 are bit-identical below `2^24`; the fixture is 30 rows) and is a **reporting step, not a gate** — recorded in SPEC §7's A2 note | **E07, E09, E11, E13** |
| **A3** | Counter CTR trains and predicts | `ctr_counter_simple` ≤1e-5 **+ permutation invariance** | **E06, E09, E11, E12** |
| **A4** | Multi-prior expansion | `ctr_borders_multiprior` ≤1e-5 | **E14, E15** |
| **A5** | `is_simple` routing | `ctr_mixed_simple_vs_combo` ≤1e-5 | **E10, E17** |
| **A6** | `counter_calc_method` | `ctr_counter_full_eval` (**eval set mandatory**) **or** a recorded deferral | **E21, E22, E23** |
| **A7** | Borders regression firewall | SPEC-CTRT-05 bit equality + all 11 existing CTR oracles | **E05** (+ regression scope of E02/E04/E06–E11/E15/E16/E22) |
| **A8** | `.cbm` mean round-trip | save→load→save byte-identical; **upstream** BTMV `.cbm` loads and predicts ≤1e-5 | **E18, E19, E20** |
| **A9** | `.cbm` non-mean unchanged | byte-identical vs the pre-change frozen baseline | **E00** (capture) → **E20** (enforce) |
| **A10** | CPU-illegal types rejected | typed error, both types, engine **and** Python | **E02, F16** |
| **A11** | `fit(X, y, cat_features=[…])` end to end | public Python API ≤1e-5 vs catboost 1.2.10 | **F17, F19, F20** |
| **A12** | One-hot wave still green | `one_hot_oracle`, `one_hot_draw_accounting`, `device_one_hot_parity` | **F23** (final gate) + **mandatory regression scope of every `boosting.rs`/`tree.rs` task**: E02, E10, E15, E16, E21, E22 |
| **A13** | Bench preflight passes | `one_hot_bench_colab.py` no longer bails, **unedited** | **F22** |

**All 13 covered.**

---

## 5. Risk register (SPEC §10 + research §I, with this plan's concrete mitigation)

| # | Risk | Mitigation in this plan |
|---|---|---|
| **R1** | Multi-prior expansion changes tie-breaks corpus-wide | E14's fixture lands **before** E15's code; E15 sits **after** the E05 firewall; all 11 single-prior oracles are E15's mandatory regression scope, under the **PER-FILE diff gate in three categories** (PLAN.md §3.2). **ZERO DIFF REQUIRED (8):** `plain_ctr_oracle_test`, `ordered_ctr_oracle_test`, `tensor_ctr_oracle_test`, `tensor_ctr_e2e_oracle_test`, `s_order_ctr_bins_oracle_test`, `multi_permutation_e2e_oracle_test`, `multi_permutation_fold_oracle_test`, `fstr_ctr_oracle_test`. **MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES (3):** `ctr_split_scoring_test` (E09/E11/E16/E22 — signature-driven argument/field edits only), `ctr_feature_materialize_test` (E09/E22 — ADDITIVE test fns + widened args), `ctr_data_roundtrip_test` (E11 — ADDITIVE test fns + the compile-forced `build_final_ctr` argument). **A diff touching an EXISTING assertion in ANY of the eleven is a STOP-AND-REPORT condition**; weakening or deleting any assertion is **forbidden** |
| **R2** | f64-vs-f32 BTMV divergence at boundary bins | `Sum` accumulated in f32 (E07); a **distinct `OnlineMeanPrefix`** type structurally forbids i64 truncation; an **allocation-free accumulator differential** (E07 test fn 2: seed `TCtrMeanHistory { sum: 16_777_216.0f32, count: 16_777_216 }`, `add(1.0)`, assert the sum does not move) with the `assert_ne!` anti-vacuity guard **kept** — the earlier `2^24 + 1`-document fixture is **deleted** (≈600 MB resident for one `#[test]`, against a recorded `target/`-disk-exhaustion hazard); a fixture-scale differential in E13 that must **report** rather than silently pass, and which SPEC §7's A2 note records as **unable to discriminate for binclf** |
| **R3** | `gen_fixtures.py` corpus wipe | **Structurally impossible here.** Every new fixture uses the fixture-local `fixtures/<name>/gen_fixtures.py` pattern (9 precedents); every generator ends with a `git status --porcelain crates/cb-oracle/fixtures` guard that `sys.exit(1)`s on any foreign dirty path; the corpus generator is never invoked |
| **R4** | `counter_calc_method` "verified" without an eval set | E23's generator **asserts** `maxdiff(full, skiptest) > 1e-3` and refuses to emit a non-discriminating fixture; E23 test fn 3 re-asserts it on the Rust side; a learn-only test is explicitly **forbidden**; deferral text is pre-written |
| **R5** | Merge friction with the uncommitted one-hot wave | Land on top, never revert; **never `git checkout --`** in a mutation revert (§3.1 says manual revert); the 3 one-hot targets are mandatory regression scope for E02/E10/E15/E16/E21/E22 and F23 is the closing gate |
| **R6** | `ctr_splits_for_tree` change unguarded | **E03 adds the covering test BEFORE E10 modifies it**, with a 2× mutation check; CodeGraph confirms "⚠️ no covering tests found" today |
| **R7** | `.cbm` mean codec byte-format wrong | E19 loads an **upstream-produced** `.cbm` (E18) rather than self-comparing; E20 adds save→load→save byte identity **and** an 8-byte-stride pin; E00's independent non-mean baseline proves the paths do not cross |
| **R8** | Multiclass Borders apply ignores `target_border_idx` | Out of scope (binclf). Recorded as a known limitation with the `static_ctr_provider.cpp:90-110` anchor in E16's notes; the repo's `numerator_denominator` Borders arm (`crates/cb-model/src/ctr_data.rs:227-241`) is correct for binclf |
| **R9** | `permutation_count = 4` facade parity gap | **ALREADY CLOSED** — `crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs` trains the `tensor_ctr_e2e` corpus through `train_cat` at `permutation_count=4`, `max_ctr_complexity=2` and asserts ≤1e-5 through the production `predict_raw_cat` path. F20 cites it; the risk is **downgraded, not re-litigated** |
| **R10** | `fit(np_array, y, cat_features=…)` silently drops the kwarg | F17's post-ingestion width guard in `estimator.rs::data_to_pool` (**never** in `ingest_py.rs`, which SPEC §7 pins as verification-only) + a dedicated test that must never pass by returning a fitted model |
| **R11** *(new)* | Structure/averaging CTR column **misalignment** under expansion | E15 test fn 2 asserts both vectors carry the **same** `(projection, prior)` sequence and the same length; E16 extends it over `(projection, b, prior)`. **The alignment invariant is scoped to `structure_fold_columns` and `averaging_ctr_features` ONLY** — it is documented at `crates/cb-train/src/boosting.rs:3262-3267` and is the most fragile thing W3 can break. It does **NOT** extend to `cat_eligible_buckets`: that vector (`crates/cb-train/src/boosting.rs:3074`, passed at `:4669`) is one `perfect_hash_bins` column per CTR-**eligible categorical feature** (`eligible_absolute`), consumed by an order-insensitive `.max()` at `crates/cb-train/src/tree.rs:3026`. It is **NOT** index-aligned with `ctr_features` and **MUST NOT** grow with the expansion — E16 test fn 1 pins it **byte-unchanged** |
| **R12** *(revised)* | The bake de-duplicates by `projection` alone, **and its copy-back overwrites every split's prior/shift/scale** | E11 changes the `seen: Vec<TProjection>` de-dup at `crates/cb-train/src/boosting.rs:5440-5443` to key on **`(projection, ctr_type)` ONLY**. **Rationale, stated accurately:** under the locked scalar-field design a projection determines `is_simple` (`CtrCandidate.is_simple` comes from `TProjection::is_simple()`, `crates/cb-train/src/candidates.rs:151-157, 194`) and therefore determines its type, so **the multi-type case is not reachable today**; the key is nonetheless the **correct** one and is what E15's copy-back lookup keys on. **`target_border_idx` MUST NOT enter `ctr_base_key` or the bake key**: it is a per-split selector consumed by `CtrValueTable::numerator_denominator`, `ctr_base_key` (`crates/cb-model/src/ctr_data.rs:299`) carries no index, `apply.rs:126 ctr_table_key` reconstructs the identical form, and adding it would break the apply-side key reconstruction, every existing CTR oracle and every committed `.cbm` fixture. One Buckets table serves both `b=0` and `b=1` — pinned by E11 test fn 4. **E15/E16 additionally OWN the copy-back at `:5458-5472`**: it is re-keyed on `(projection, ctr_type)`, it **stops overwriting** `spec.prior_num`/`spec.prior_denom` (already correct from `tree.rs:3294-3295`), and `spec.shift`/`spec.scale` are derived **per split** from `calc_normalization(spec.prior_num)` — pinned by E15 test fn 3. The re-keyed lookup **survives purely as an existence gate** (nothing is read out of the table any more): only splits with a baked `(projection, ctr_type)` table get a derived `shift`/`scale`, a split with no table keeps `0.0`/`1.0`. Consequently `BakedCtrTable.{shift, scale, prior_num, prior_denom}` (`crates/cb-train/src/ctr/bake.rs:78-86`) become **informational-only in production** — `CtrData::from_baked` already ignores them — and **MUST stay** for the existing `ctr_split_scoring_test` / `final_ctr_test` assertions |
| **R13** *(new)* | `EvalSet` cannot carry cat columns ⇒ SPEC-CTRT-17 unimplementable | E21 adds the field (6 literal sites, 4 files, no derives) and a `train_cat_with_eval_sets` wrapper; `train_inner` already accepts `eval_sets` |
| **R14** *(new)* | `F08`'s new `Model` field changes `.cbm` bytes | F08's second test asserts `save_cbm` bytes are identical with `cat_feature_count` 0 vs 7; the field is runtime-only and touches neither the FlatBuffers schema nor `json.rs` |

---

## 6. Unresolved blockers and unverified assumptions

### 6.1 No unresolved blockers

All five plan documents (`PLAN.md`, `PLAN-W2-W3.md`, `PLAN-W4-W5.md`,
`PLAN-PART2.md`, `PLAN-COVERAGE.md`) **exist on disk** under
`/home/user/Documents/workspace/catboost_rs/.planning/plans/ctr-type-engine-and-facade-routing/`.
The earlier blocker recorded here — that the planning agent's read-only tool grant
had left them unwritten — is **STALE and has been deleted**; acting on it would send
an executor to materialize documents that are already present.

### 6.2 Unverified assumptions carried into implementation (each with its falsification point)

1. **[UNVERIFIED — research §J.4, the largest schedule risk]** That the 30-row
   isolating configs actually yield a **winning split of each intended type**.
   Data-dependent. *Falsified at generation time* by each generator's mandatory
   anti-false-pass assertion (E12/E13/E14/E16/E17/E23). If an assertion fires, widen
   the corpus or strengthen the categorical signal — **never weaken the assertion**.
2. **[UNVERIFIED — research §J.3]** That this repo's `PerfectHash` first-seen bin
   ordering matches upstream's `ComputeReindexHash`. Non-blocking: lookups are by
   hash, Counter's MAX denominator is order-independent, and Buckets' numerator is
   per-bucket. *Falsified by* E12's localization-ladder step 1.
3. **[UNVERIFIED — research §J.2]** That the online Counter bucket space equals the
   final-CTR bucket space. `ctr_leaf_count_limit` defaults to `u64::MAX`, so they
   coincide for every realistic fixture. Would matter only if that knob is exposed.
4. **[UNVERIFIED — research §J.1]** Whether `PrepareScratchBlob<int>`
   zero-initializes `TBucketsView`. Non-blocking: this repo allocates fresh zeroed
   vectors regardless.
5. **[UNVERIFIED]** The exact upstream `CTRBlob` byte layout for mean tables
   (`f32 Sum` + `i32 Count` pairs, **8-byte stride**) is **inferred** from
   `TCtrMeanHistory`'s C++ struct, not read from a hex dump. *Falsified by* E19 test
   fn 1 against the **upstream-produced** `model.cbm` from E18 — which is exactly why
   E18 precedes E19 and why self-comparison alone is insufficient (R7).
   **Explicit alternative recorded:** this repo's own self-describing CTR format uses
   `f32 LE Sum ; i64 LE Count` — a **12-byte** stride
   `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:947]`. E19 therefore carries a
   mandatory two-branch probe: try stride 8; if the length check fails, try stride
   12; **if 12 matches, STOP AND REPORT** (SPEC-CTRT-14 and E20's encoder must be
   re-specified) rather than silently adopting it; if neither matches, return a typed
   `Deserialize` naming both candidate strides.
6. **[UNVERIFIED]** That `max_ctr_complexity = 1`/`0` keeps every new fixture clear
   of the ORD-06/ORD-07 combination-CTR gating bug. Supported by `plain_ctr` and
   `tensor_ctr_e2e` passing today, not proven for the new configs. *Falsified by*
   E17's and F20's localization ladders → **STOP AND REPORT**, do not fix here.
7. **[UNVERIFIED]** That E23's fixture can be made to discriminate `Full` from
   `SkipTest` at ≤1e-5 on both sides. Research measured `4.010e-01` with a 40-row
   eval set, so the effect exists; whether both settings *also* land within 1e-5 of
   upstream is untested. *Falsified by* E23 → take the pre-written deferral path.
8. **[UNVERIFIED]** That `catboost==1.2.10` re-runs are byte-deterministic for the
   **categorical-only** fixtures. `ctr_load/gen_fixtures.py` records observed
   run-to-run nondeterminism for a fixture **with a float column**; removing float
   columns should remove it. *Falsified by* each fixture task's double-generation
   `diff -r` → if `predictions.npy` differs, **STOP AND REPORT**.

### 6.3 Recorded follow-ups (explicitly OUT of scope)

- Multi-description CTR configs (`simple_ctr = [Borders(...), Counter(...)]`) —
  documented as a parity gap by F00 (SPEC-CTRT-19), not closed.
- `per_feature_ctr` overrides (upstream checks `PerFeatureCtrs` **first** for a
  single-cat projection, `ctr_helper.h:52-62`) — unsupported; E17 adds the doc note.
- Widening `TProjection::is_simple()` if bin/one-hot projection members are ever
  added (`IsSingleCatFeature` also requires `BinFeatures.empty() && OneHotFeatures.empty()`,
  `projection.h:102-104`) — E17 adds the doc note.
- Multiclass Borders apply (`target_border_idx` ignored at inference,
  `static_ctr_provider.cpp:90-110`) — known limitation, R8.
- `cat_features` by column NAME (upstream accepts `list[int | str]`; this repo's
  `Pool` and the new fit-kwarg accept indices only) — MINOR-10.
- `cat_features` on Python `cv()` / `grid_search()` — F14 rejects categorical pools
  there entirely; exposing the kwarg is a later plan.
- Aligning the `Pool` + `y` fit fast-path with upstream's `CatBoostError` — the
  pre-existing WR-04 divergence, untouched by F17.
- Registering these documents in TreeFinder under the `.planning/plans/**/SPEC.md`
  convention alongside `snapshot-resume` and `xgboost-rust-rewrite` — partially
  done by this planning run.

---

## 7. Process attestation

**No GSD skill, command, workflow, or agent was invoked, imported, delegated to, or
depended upon at any point in producing this plan.** No `Skill` tool call was made —
in particular not `spec-tdd-planner-skill`, not `planning-settings-skill`, and not
any other listed skill. No `./planning/settings.json` was read or written (this plan
lives under `.planning/plans/`, the directory the invoking message specified).
Goal-backward reasoning informed only the *shape* of §4's wave graph (acceptance
scenarios → observable end states → tasks ordered by prerequisite, not file layout).

**Tools used:** `Read`; `Bash` (read-only inspection only — `git status`,
`git diff --stat`, `git rev-parse`, `ls`, `find`, `grep`, `wc`, `sed -n`, and one
read-only `.venv/bin/python -c "import catboost"` version probe); `ToolSearch`;
`mcp__codegraph__codegraph_explore`; `mcp__tree_finder__{document_create,
document_get,document_list,document_patch}` for these four planning documents only.

**No production source file was created, modified, moved, or deleted.** No fixture
was generated. `crates/cb-oracle/generator/gen_fixtures.py` was **not** executed.
The uncommitted one-hot device wave in the working tree was **not** touched.
