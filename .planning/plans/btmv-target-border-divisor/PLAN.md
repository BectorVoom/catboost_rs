---
title: TDD plan — BinarizedTargetMeanValue Sum is halved (the bake passes the wrong target_border_count)
plan_id: BUG-BTMV
status: draft
format: markdown
plan_version: 1
updated_at: 2026-08-02T00:00:00Z
source_spec: .planning/plans/btmv-target-border-divisor/SPEC.md
inherits_conventions_from: .planning/plans/ctr-type-engine-and-facade-routing/PLAN.md §3, §3.1, §3.2
sibling_plan: .planning/plans/ctr-split-border-space/PLAN.md (BUG-CTRB — structure and rigor matched)
specifications:
  - SPEC-BTMV-01
  - SPEC-BTMV-02
  - SPEC-BTMV-03
  - SPEC-BTMV-04
tasks: 7
---

# TDD plan — BUG-BTMV (BTMV target-border divisor)

Plan only. **No production code is written by the planner.** Every task below is
an executable prompt: exact file, exact line, exact command, exact assertion,
exact expected failure text.

This plan is the direct sibling of `.planning/plans/ctr-split-border-space/`
(BUG-CTRB), executed successfully earlier this session at HEAD `c21f44a`. It
matches that plan's structure deliberately: an explicit hazard section (§0), a
locked-decision table (§2), an evidence ledger (§4), waves with an exclusive-
resource discipline (§5), and mutation checks whose expected outcomes are
**enumerated**, including the outcomes that must stay **green**.

---

## 0. THE PRINCIPAL HAZARD — TWO DIFFERENT UPSTREAM DIVISORS, ONE NAME

> **This section resolves SPEC.md §7 R1, and RETRACTS the recommendation in
> SPEC.md §3.** SPEC.md §3 says *"The fix should route through
> [`ECtrType::target_border_count`] rather than hard-coding `classes - 1`, so
> the rule lives in exactly one place."* **That recommendation is WRONG and must
> not be implemented.** Upstream has **two distinct divisors** and the helper
> mirrors the *other* one.

CatBoost 1.2.10 computes a "target border count" in **two different places with
two different rules**, and only one of them is the whole-set bake:

```text
 ── THE ONLINE (per-object prefix) PATH ─────────────────────────────────────────
 online_ctr.cpp:738   const ui32 targetBorderCount =
                          GetTargetBorderCount(ctrInfo[ctrIdx], targetClassesCount);
 online_ctr.cpp:741   writer->AllocateCtrData(ctrIdx, targetBorderCount, priors.size());
                      ^^^^^ the ONLY consumer of the :738 value — an ALLOCATION SIZE

 online_ctr.cpp:762   CalcOnlineCTRMean(..., targetClassesCount - 1, ...);
                                             ^^^^^^^^^^^^^^^^^^^^^ the BTMV mean
                                             divisor, passed as a LITERAL, NOT via
                                             GetTargetBorderCount
 online_ctr.cpp:777   CalcOnlineCTRClasses(..., GetTargetBorderCount(...), ...);
                      ^^^^^ the helper IS used here — for the CLASS-prefix types

 ── THE WHOLE-SET BAKE (CalcFinalCtrsImpl) — WHAT THIS PLAN FIXES ──────────────
 online_ctr.cpp:914   int targetBorderCount = targetClassesCount - 1;
 online_ctr.cpp:920   elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount);
                      ^^^^^ UNCONDITIONAL. Computed ONCE, OUTSIDE the type switch,
                            for EVERY ctrType. GetTargetBorderCount is not called
                            anywhere in CalcFinalCtrsImpl.
```

`[VERIFIED: UPSTREAM catboost v1.2.10 tag, `catboost/private/libs/algo/online_ctr.cpp`,
fetched and read in full — see E1/E2/E3]`

### Consequences, all four of which every task must respect

1. **The correct expression at `bake.rs:196` is `classes - 1`, floored — NOT
   `ctr_type.target_border_count(classes)`.** `CalcFinalCtrsImpl` is
   type-**independent**: the divisor does not depend on `ctrType` at all.
2. **`ECtrType::target_border_count` (`crates/cb-train/src/ctr/mod.rs:123`) is
   correct and stays untouched.** It faithfully mirrors `GetTargetBorderCount`
   (`ctr_helper.h:34-42`, E4) — the **online**-path helper. It is simply not the
   bake's divisor. **Do not "unify" the two. Do not delete it.**
3. **`ECtrType::target_border_count` has ZERO production callers today**
   (`crates/cb-train/src/ctr/mod_test.rs:32,43` are its only call sites,
   E5). An executor who reads SPEC.md §3 will be tempted to wire it into the
   bake "so it is finally used". **That is the defect this section exists to
   prevent.** If the helper's disuse is to be fixed, that is a separate change on
   the online-allocation path, out of scope here.
4. **No runtime gate at binary classification can tell the two options apart**
   — for `classes = 2` the helper returns `1` for BTMV and `classes - 1` is also
   `1`. The discriminator therefore has to be constructed deliberately, at
   `classes = 3`, where the three candidate expressions produce **three distinct
   values**:

   | expression | `classes = 2` | `classes = 3` |
   |---|---|---|
   | today's bug (`classes`) | **2** | **3** |
   | **the fix** (`classes - 1`, E1) | 1 | **2** |
   | SPEC.md §3's suggestion (`BTMV.target_border_count(classes)`) | 1 | **1** |

   B01 Test 2 is that three-way discriminator and is the **only** gate that can
   catch a helper-based "fix". It is a **characterization** test at a
   configuration production never reaches (production hard-codes `classes = 2`,
   `boosting.rs:5582`, E6) and carries a STOP CONDITION accordingly.

### The choice also PROVABLY preserves today's non-mean baked payloads

SPEC.md §7 R1 asks, if ambiguous, to prefer the option that provably preserves
today's non-mean payloads. **It is not ambiguous** (E1 is unconditional), and
`classes - 1` preserves them anyway — by code path, not by luck:

- `target_border_count` (the 5th argument) reaches **exactly one** expression in
  `accumulate_online`: `let divisor = target_border_count as f32;`
  (`online.rs:198`), used at **exactly one** site: `binarized_mean[bucket]
  .add(class as f32 / divisor)` (`online.rs:212-214`) `[VERIFIED: E7]`.
- `OnlineCtrAccumulator::binarized_mean` is read by **exactly one** arm of
  `build_final_ctr` — `ECtrType::BinarizedTargetMeanValue`
  (`final_ctr.rs`, E8). Borders/Buckets read `class_histories`, Counter/
  FeatureFreq read `total_counts`, FloatTargetMeanValue reads `float_mean`.
- `bake_ctr_table` only populates `BakedCtrTable.mean` in the
  `BinarizedTargetMeanValue | FloatTargetMeanValue` arm (`bake.rs:250-260`);
  every other arm leaves `mean` empty `[VERIFIED: E9]`.

⇒ **The divisor is unreachable from any non-mean baked payload.** B05 turns that
proof into a frozen-bytes guard plus an enumerated mutation whose required
outcome is that the non-mean gates stay **GREEN**.

---

## 1. Working-tree state this plan lands on (verified, load-bearing)

`[VERIFIED: LOCAL git log/status + cargo test, executed 2026-08-02]`

- Branch `fix/bootstrap-rng-draw-accounting`; HEAD **`c21f44a`**
  *"fix(train): BUG-CTRB C04-C06 — name the conversion, gate the codec, close
  out"*, preceded by `06296de` (BUG-CTRB C01-C03) and `c1aecc0` (the BUG-CTRB
  plan). **BUG-CTRB is landed and committed.**
- **THREE UNTRACKED paths MUST NOT be lost** (`git status --short`, executed):
  - `crates/cb-oracle/fixtures/ctr_btmv_simple/` — the frozen E13 fixture
    (`config.json`, `gen_fixtures.py`, `model.cbm`, `model.json`,
    `predictions.npy`, `X_cat.npy`, `y.npy`)
  - `crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs` — E13's gate
  - `.planning/plans/btmv-target-border-divisor/` — **this plan and its SPEC**

  → **`git checkout --`, `git stash` and `git clean` are FORBIDDEN in every task
  of this plan, including every mutation revert.** All reverts are MANUAL edits.
  All three paths are unrecoverable if deleted.
- `.cargo/config.toml` exists, is deliberately uncommitted (`.git/info/exclude`),
  and only sets `debug = "line-tables-only"` / `split-debuginfo = "unpacked"`.
  It changes no codegen and no test semantics. No task may remove it.
  (Disk headroom checked: `/home` 629 G free — the documented `target/`
  exhaustion mode is not currently active.)

### 1.1 THE DEFECT IS LIVE — but the failure TEXT is NOT what SPEC.md describes

**⚠ CORRECTION TO THE PLAN REQUEST AND TO SPEC.md §1.** The request states the
E13 gate is *"failing at 1.283e-1"*, implying a `compare_stage` divergence
panic. At HEAD `c21f44a` the max divergence **is still `1.283e-1`**, but the
assertion that actually fires is the **anti-vacuity guard**, eight lines earlier:

```text
$ cargo test -p cb-train --test ctr_btmv_simple_oracle_test
running 4 tests
test btmv_baked_table_carries_a_non_empty_mean_vector ... ok
test btmv_save_cbm_is_a_typed_rejection_until_e20 ... ok
test btmv_simple_predictions_match_upstream_within_1e_minus_5 ... FAILED
test btmv_f64_sum_accumulation_diverges_from_upstream_on_this_fixture ... ok

---- btmv_simple_predictions_match_upstream_within_1e_minus_5 stdout ----
panicked at crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs:141:5:
predictions are constant — the gate would be vacuous

test result: FAILED. 3 passed; 1 failed
```

`[VERIFIED: LOCAL executed 2026-08-02 at c21f44a]`

Run with `-- --nocapture`, the still-passing reporting test prints the real
number `[VERIFIED: LOCAL executed]`:

```text
REPORTED: f32/f64 indistinguishable at this scale (maxdiff = 1.2830170735076996e-1)
```

**What this means, and why an executor must know it before starting.**

- Our model's predictions are **CONSTANT** across all 30 documents. Every
  document falls on the same side of every persisted CTR border, so the tree
  routes everything to one leaf.
- That is exactly what a halved `Sum` predicts. Apply-space CTR value is
  `((Sum + 0.5) / (Count + 1)) * 15`. With upstream's Sums the five buckets
  quantize to bins ≈ `10, 6, 9, 6, 11`; with our halved Sums they quantize to
  ≈ `5, 3, 5, 3, 6`. The structure search (which uses the **correct**
  `online_mean_prefix`, SPEC-BTMV-03) chose borders in the upper range, and no
  halved value reaches them. `[INFERRED from E10 + E11 + the committed table]`
- `1.2830170735076996e-1` is the max |diff| of a constant prediction vector
  against upstream's — the SAME number SPEC.md quotes, so **BUG-CTRB did not
  move BTMV predictions at all**, consistent with E20 of the sibling plan.
- **Expected Red text for B03's secondary gate is therefore
  `predictions are constant — the gate would be vacuous`, NOT
  `BTMV-CTR predictions diverged from upstream (max |diff| = …)`.** An executor
  who "cannot reproduce the reported failure" because the message differs must
  not go hunting; this is the reproduction.

### 1.2 Accepted pre-existing warning (do NOT fix it here)

`crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs:25-26` emits
`warning: unused imports: TProjection and materialize_ctr_feature`
`[VERIFIED: LOCAL, in the cargo test output above]`. It is a **warning**, not a
denied lint, and the file is one of the three protected untracked paths. **No
task in this plan edits that file for any reason**, including to silence this
warning or to rename the misnamed `counter_params()` helper (`:59`, a copy-paste
name in a BTMV fixture — B02 copies it verbatim rather than renaming it).

---

## 2. Locked decisions this plan encodes (non-negotiable)

| # | Decision | Basis |
|---|---|---|
| **D1** | The bake's divisor is **`target_classes_count - 1`, type-INDEPENDENT**. `CalcFinalCtrsImpl` computes it once, outside the type switch. | §0; E1 |
| **D2** | **`ECtrType::target_border_count` is NOT used at the bake.** SPEC.md §3's recommendation to route through it is **RETRACTED**. The helper mirrors `GetTargetBorderCount`, the ONLINE-path allocation/class helper. It stays byte-unchanged apart from a clarifying doc comment (B04). | §0; E1/E2/E4 |
| **D3** | The concrete expression is **`classes.saturating_sub(1).max(1)`** — the exact idiom already used by `online_mean_prefix` (`online.rs:321`, E10). The `.max(1)` floor exists because `accumulate_online` **rejects `target_border_count == 0` with a typed error** (`online.rs:176-180`, E7); without the floor, `bake_ctr_table(classes = 1, …)` would start returning `CbError::Degenerate` where it returns `Ok` today. With the floor, `classes == 1` is **byte-identical to today's** (a single-class corpus has every `target_class == 0`, so every `Sum` is `0` under either divisor). **`classes == 0` moves the OTHER way** (plan-check pass 1, MINOR-1): today it is `Err(Degenerate)` from the zero-divisor rejection, after the fix it is `Ok` with a degenerate table. Unreachable — the sole production caller hard-codes `2` (`boosting.rs:5582`) — and no test covers it, but the earlier blanket "`classes <= 1` is byte-identical" claim was FALSE and is corrected here. | E7; E10; B04 Test 3 |
| **D4** | `accumulate_online`'s **FOURTH** argument (`classes`) at `bake.rs:196` is **CORRECT and unchanged** (SPEC.md §9 OQ1 answered: only the fifth is wrong). Upstream uses `targetClassesCount` for exactly what our `classes` drives — the class-histogram width and `TargetClassesCount` (`online_ctr.cpp:909-912, 930-934`). | E3 |
| **D5** | `accumulate_online`'s signature and semantics are **unchanged**. It is correct; it is being called wrongly. | SPEC.md §2 |
| **D6** | `online_mean_prefix` (`online.rs:298-356`) is **unchanged**. It is upstream-exact — `online_ctr.cpp:762` passes `targetClassesCount - 1` as a LITERAL to `CalcOnlineCTRMean`, not via the helper (E2). B06 **verifies** this rather than assuming it. **SPEC-BTMV-03's "if it is also wrong the scope widens" branch is CLOSED: it is not wrong.** | E2; E10 |
| **D7** | `build_final_ctr`, `ctr_feature.rs`, `cb-model` and every apply-side file are **NOT changed**. | SPEC.md §2 |
| **D8** | **No fixture is regenerated.** `crates/cb-oracle/generator/gen_fixtures.py` is **NEVER** invoked, and neither is `crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py`. | ctr-type `PLAN.md` §3 |
| **D9** | `catboost-master/` is a stale 3-file stub of a **different revision** and is **never** cited. Upstream evidence in this plan comes from the **`v1.2.10` git tag**, fetched by URL and quoted with line numbers, plus the committed fixtures. | CLAUDE.md correction block |
| **D10** | The 11 CTR oracles + 3 one-hot targets are a **NON-REGRESSION gate only**. None of them exercises a mean-CTR bake (E12), so none can prove this defect fixed. **They must never be cited as this plan's falsifiable gate.** | E12 |
| **D11** | The helper introduced in B04 lives in **`crates/cb-train/src/ctr/mod.rs`, private, adjacent to `ECtrType::target_border_count`**, so the two divisors sit side by side with contrasting doc blocks. `bake` is a **child** module of `ctr`, so it can call a `ctr`-private item; `mod_test` is a **child** module of `ctr`, so it can test one. **No visibility is widened and no new `#[cfg(test)]` mount is added.** | E13; E14 |

---

## 3. Shared conventions

**Inherited verbatim** from
`.planning/plans/ctr-type-engine-and-facade-routing/PLAN.md`:

- **§3** — source/test separation, the no-`unwrap`/`expect`/`panic!`/raw-indexing
  production ban, typed errors, the never-`gen_fixtures.py` rule, the accepted
  failing-test baseline, the `≤1e-5` parity bar.
- **§3.1** — the guard-test falsifiability (MUTATION CHECK) protocol.
- **§3.2** — the repository-verified command blocks, **including the exact
  11-CTR-oracle block and the 3 one-hot targets**, reproduced verbatim in §3.2
  below so no task has to cross-reference mid-execution.

### 3.0 Deltas and clarifications specific to this plan

- **Source/test separation (CLAUDE.md, MANDATORY).** No `mod tests` and no
  `#[cfg(test)]` block inside any production file. This plan reuses **only
  existing mounts** and adds none. The four mounts in
  `crates/cb-train/src/ctr/mod.rs:42-53` `[VERIFIED: LOCAL]`:

  | mount | file | test filter | used by |
  |---|---|---|---|
  | `mod online_test;` (`:43-44`) | `crates/cb-train/src/ctr/online_test.rs` | `ctr::online_test` | **B06** |
  | `mod calc_ctr_test;` (`:46-47`) | `calc_ctr_test.rs` | `ctr::calc_ctr_test` | *not used* |
  | `mod final_ctr_test;` (`:49-50`) | `final_ctr_test.rs` | `ctr::final_ctr_test` | **B01, B05** (the E11 bake tests already live here) |
  | `mod mod_test;` (`:52-53`) | `mod_test.rs` | `ctr::mod_test` | **B04** |

  New **integration** tests go in `crates/<crate>/tests/` as their own targets
  (**B02**).
- **Visibility (D11).** `mod ctr;` is **private** in `crates/cb-train/src/lib.rs:21`
  `[VERIFIED: LOCAL]`. A private `fn` in `ctr/mod.rs` is visible to `ctr::bake`
  (descendants may read ancestors' private items) and to `ctr::mod_test`. **No
  `pub`, no `pub(crate)`, no new `pub use` in `lib.rs`.**
- **Test-code lint exemption.** `cb-train`'s crate root already carries
  `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used,
  clippy::panic, clippy::indexing_slicing))]` (`lib.rs:1`), which covers every
  `--lib` test module. **Do not add a file-level `#![allow(...)]` to
  `final_ctr_test.rs`, `mod_test.rs` or `online_test.rs`** — none of them has one
  today. A **new integration** target (B02) does need the established file-level
  form, precedent `crates/cb-train/tests/ctr_counter_simple_oracle_test.rs:15`:
  `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]`.
- **Dev-dependency edge.** `cb-model` is a dev-dependency of `cb-train`, and
  `serde_json`, `ndarray`, `ndarray-npy`, `cb-oracle`, `approx` are available to
  `cb-train` integration tests `[VERIFIED: LOCAL crates/cb-train/Cargo.toml]`.
- **The workspace enables NO cast lint** — only `unwrap_used`, `expect_used`,
  `panic`, `indexing_slicing` are denied (`Cargo.toml:10-14`)
  `[VERIFIED: LOCAL]`. `as f32` / `as usize` will not trip clippy; do not
  pre-emptively add an `#[allow]`.
- **NEVER `git checkout --` / `git stash` / `git clean`** — see §1.

### 3.1 Mutation-check protocol (reference)

As ctr-type `PLAN.md` §3.1, with §1's manual-revert requirement:

1. Write the test. Run it. **Record the result verbatim.**
2. Apply the task's **named single-expression mutation** to production code.
3. Re-run; the test MUST fail **with the named message**. **Record the failure
   text verbatim.**
4. **Manually revert** the mutation (never `git checkout --`) and re-run to
   confirm green.

A guard test that cannot be made to fail by its named mutation is not a guard and
the task is **not complete**. Where a mutation's expected outcome set includes
tests that must stay **GREEN**, those green results are **REQUIRED results**, not
incidental — a regression there is a STOP-AND-REPORT condition.

### 3.2 Repository-verified commands

```bash
# ── THIS PLAN'S PRIMARY FALSIFIABLE GATES ────────────────────────────────────
cargo test -p cb-train --lib ctr::final_ctr_test    # B01 (bake per-bucket), B05
cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test   # B02 (NEW target)
cargo test -p cb-train --lib ctr::mod_test          # B04 (the named divisor helper)
cargo test -p cb-train --lib ctr::online_test       # B06 (prefix producer)

# ── THE E13 INTEGRATION GATE (SECONDARY — it does NOT localize) ──────────────
cargo test -p cb-train --test ctr_btmv_simple_oracle_test        # must reach 4/4

# ── E12's Counter gate + the three BUG-CTRB gates (mandatory regression) ─────
cargo test -p cb-train --test ctr_counter_simple_oracle_test     # 4/4
cargo test -p cb-train --test ctr_border_space_test
cargo test -p cb-train --test ctr_border_upstream_anchor_test
cargo test -p cb-model --test ctr_border_cbm_roundtrip_test

# ── THE 11 EXISTING CTR ORACLES — run the WHOLE block, never a subset ────────
# 9 in cb-train + 2 in cb-model. DIFF GATE for this plan: ALL ELEVEN files must
# show ZERO diff (`git diff --stat` prints nothing for them). NO task in this
# plan edits any of the eleven. Editing, weakening, rewording or deleting ANY
# assertion in ANY of them is a STOP-AND-REPORT condition.
cargo test -p cb-train --test plain_ctr_oracle_test \
                       --test ordered_ctr_oracle_test \
                       --test tensor_ctr_oracle_test \
                       --test tensor_ctr_e2e_oracle_test \
                       --test s_order_ctr_bins_oracle_test \
                       --test ctr_split_scoring_test \
                       --test ctr_feature_materialize_test \
                       --test multi_permutation_e2e_oracle_test \
                       --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test

# ── THE ONE-HOT WAVE REGRESSION SCOPE (A12) ─────────────────────────────────
cargo test -p cb-train --test one_hot_oracle_test \
                       --test one_hot_draw_accounting_test \
                       --test device_one_hot_parity_test

# ── .cbm / model serde gates ────────────────────────────────────────────────
cargo test -p cb-model --test cbm_oracle_test --test json_oracle_test \
                       --test float_only_byte_identity_test
cargo test -p cb-model --test ctr_nonmean_byte_identity_test   # E00 frozen bytes

# ── Broad crate sweeps (baseline: only the pre-existing
#    monotone_non_symmetric_and_region_are_typed_errors may fail) ────────────
cargo test -p cb-train --no-fail-fast
cargo test -p cb-model --no-fail-fast

# ── Lints ───────────────────────────────────────────────────────────────────
cargo clippy --workspace --all-targets
# Accepted pre-existing WARNING (§1.2): unused imports at
# crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs:25-26. Do NOT fix it.

# ── FORBIDDEN ───────────────────────────────────────────────────────────────
# crates/cb-oracle/generator/gen_fixtures.py                     NEVER
# crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py      NEVER
# git checkout -- / git stash / git clean                        NEVER (§1)
```

---

## 4. Evidence ledger

Every claim this plan rests on, with its verification label. Tasks cite by tag.

| Tag | Claim | Evidence |
|---|---|---|
| **E1** | **`CalcFinalCtrsImpl` — the whole-set bake — divides by `targetClassesCount - 1`, UNCONDITIONALLY and type-INDEPENDENTLY.** `online_ctr.cpp:914` `int targetBorderCount = targetClassesCount - 1;` sits **outside** the per-type `if/else` chain that begins at `:917`; the BTMV arm at `:918-921` reads it: `elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount);`. **`GetTargetBorderCount` is not called anywhere inside `CalcFinalCtrsImpl` (`:875-939`).** | `[VERIFIED: UPSTREAM v1.2.10 catboost/private/libs/algo/online_ctr.cpp:875-939, fetched via raw.githubusercontent.com and read in full]` |
| **E2** | **The ONLINE mean path also uses `targetClassesCount - 1`, passed as a LITERAL — not via the helper.** `online_ctr.cpp:756-767`: `} else if (ctrType == ECtrType::BinarizedTargetMeanValue) { CalcOnlineCTRMean(testOffsets, hashArr, leafCount, foldLearnTargetClass[classifierId], targetClassesCount - 1, …); }` — the literal is at **`:762`**, exactly the line SPEC.md cites. `CalcOnlineCTRMean`'s parameter is `int targetBorderCount` (`:442`) and its only use is `elem.Add(static_cast<float>(permutedTargetClass[docId]) / targetBorderCount);` (`:467`). | `[VERIFIED: UPSTREAM v1.2.10 online_ctr.cpp:437-470, 720-790]` |
| **E3** | **The 4th argument (`classes`) IS correct (OQ1 answered: no).** `CalcFinalCtrsImpl` uses `targetClassesCount` for exactly what our `classes` drives: `result->TargetClassesCount = targetClassesCount;` and `ctrIntArray = result->AllocateBlobAndGetArrayRef<int>(leafCount * targetClassesCount);` (`:909-911`), then `TArrayRef<int> elem = MakeArrayRef(ctrIntArray.data() + targetClassesCount * elemId, targetClassesCount); ++elem[targetClass[z]];` (`:930-934`). Our mirror: `TCtrHistory::new(classes)` (`online.rs:192`) + `build_final_ctr`'s bucket-major `for c in 0..classes` flatten. | `[VERIFIED: UPSTREAM online_ctr.cpp:905-935]` + `[VERIFIED: CODEGRAPH accumulate_online, build_final_ctr]` |
| **E4** | `GetTargetBorderCount` (`ctr_helper.h:34-42`) is: `BTMV \|\| Counter → 1`; else `Buckets ? targetClassesCount : targetClassesCount - 1`. `ECtrType::target_border_count` (`crates/cb-train/src/ctr/mod.rs:123-131`) mirrors it exactly (with `saturating_sub` hardening). The helper is therefore **correct** — for the online path (`online_ctr.cpp:738`, `:777`). | `[VERIFIED: UPSTREAM v1.2.10 catboost/private/libs/algo/ctr_helper.h:34-42, fetched and read]` + `[VERIFIED: CODEGRAPH ECtrType::target_border_count]` |
| **E5** | **`ECtrType::target_border_count` has ZERO production callers.** Workspace-wide call sites are `crates/cb-train/src/ctr/mod_test.rs:32` and `:43` only. | `[VERIFIED: LOCAL grep -rn "target_border_count" --include=*.rs crates/ — the only other hits are the definition (`mod.rs:123`), `accumulate_online`'s parameter name, test names/comments in `online_test.rs`/`final_ctr_test.rs`, and the unrelated Python param string `"ctr_target_border_count"` at `crates/catboost-rs-py/src/params.rs:132`]` |
| **E6** | **`bake_ctr_table` has exactly ONE production caller**, `crates/cb-train/src/boosting.rs:5578-5586`, which passes a hard-coded `2, // binclf target-class count` (`:5582`). Other call sites are tests: `crates/cb-model/tests/ctr_data_roundtrip_test.rs:254`, `crates/cb-train/src/ctr/final_ctr_test.rs:130,195`, `crates/cb-train/tests/ctr_split_scoring_test.rs:548,583,653`. | `[VERIFIED: LOCAL grep -rn "bake_ctr_table(" --include=*.rs crates/; boosting.rs:5570-5590 read]` |
| **E7** | **`accumulate_online` has exactly ONE production caller** — `crates/cb-train/src/ctr/bake.rs:196` — and it is the wrong call: `accumulate_online(&key_refs, &target_class_n, &target_zero, classes, classes)?`. Test callers (unaffected, and all passing an explicit intentional divisor): `crates/cb-train/src/ctr/online_test.rs:20,33,46,60,74,80,85`; `crates/cb-train/src/ctr/final_ctr_test.rs:15`; `crates/cb-model/tests/ctr_data_roundtrip_test.rs:100,134,160`. Inside the function, `target_border_count` reaches exactly one expression — `let divisor = target_border_count as f32;` (`online.rs:198`) — used at exactly one site, `binarized_mean[bucket].add(class as f32 / divisor)` (`:212-214`); and `target_border_count == 0` is rejected with `CbError::Degenerate` at `:176-180`. **SPEC.md §9 OQ2 answered: no other caller passes a wrong divisor.** | `[VERIFIED: LOCAL grep -rn "accumulate_online" --include=*.rs crates/]` + `[VERIFIED: CODEGRAPH accumulate_online — verbatim body]` |
| **E8** | `binarized_mean` is read by **exactly one** arm of `build_final_ctr`: `ECtrType::BinarizedTargetMeanValue => { table.mean_sum = acc.binarized_mean.iter().map(\|m\| m.sum).collect(); … }`. `Borders \| Buckets` read `acc.class_histories`; `Counter`/`FeatureFreq` read `acc.total_counts`; `FloatTargetMeanValue` reads `acc.float_mean`. | `[VERIFIED: CODEGRAPH build_final_ctr; crates/cb-train/src/ctr/final_ctr.rs:83-143 read in full]` |
| **E9** | `bake_ctr_table` populates `BakedCtrTable.mean` **only** in the `ECtrType::BinarizedTargetMeanValue \| ECtrType::FloatTargetMeanValue` arm (`bake.rs:250-260`); `Borders \| Buckets` (`:226-238`) and `Counter \| FeatureFreq` (`:239-249`) leave `mean` empty and fill `int_counts`. `(shift, scale, prior)` are derived type-agnostically from the prior (`:263-270`) and are untouched by the divisor. | `[VERIFIED: CODEGRAPH bake_ctr_table — verbatim body :114-285]` |
| **E10** | **`online_mean_prefix` computes the divisor CORRECTLY**: `// targetBorderCount = targetClassesCount - 1 (online_ctr.cpp:762), floored at 1 so a degenerate classes can never divide by zero.` / `let divisor = classes.saturating_sub(1).max(1) as f32;` (`online.rs:319-321`), used at `h.add(class as f32 / divisor)` (`:351`). Its only production caller is `crates/cb-train/src/ctr/ctr_feature.rs:233-241`, the `ECtrType::BinarizedTargetMeanValue` arm, passing `SIMPLE_CLASSES_COUNT` (`= 2`, `online.rs:52`). ⇒ divisor `1`, matching E2 **at every class count, not just binclf**. | `[VERIFIED: CODEGRAPH online_mean_prefix]` + `[VERIFIED: LOCAL ctr_feature.rs:215-245; grep for callers]` |
| **E11** | **Upstream's committed BTMV table** (`crates/cb-oracle/fixtures/ctr_btmv_simple/model.json` → `ctr_data`, one key: `{"identifier":[{"cat_feature_index":1,"combination_element":"cat_feature_value"}],"type":"BinarizedTargetMeanValue"}`, `hash_stride: 3`, `counter_denominator: 0`): <br>`['18446744073709551615', 3, 7, '14096670708071601218', 5, 7, '10650234391120027977', 3, 7, '3644720124901778394', 4, 6, '15097791572046390990', 2, 5, '6692239851685836511', 4, 5]`. The five real buckets' Counts sum to **30 = n_rows**; every Sum is an **integer** (divisor `1`, so `Sum` = the count of class-1 documents). | `[VERIFIED: LOCAL .venv/bin/python json dump of the committed model.json, executed]` |
| **E12** | **`hash_map[0]` is a `u64::MAX` SENTINEL whose payload is STALE and must be SKIPPED.** `18446744073709551615` leads the array in **both** committed CTR fixtures: `ctr_btmv_simple` (payload `3, 7`, duplicating the real bucket `10650234391120027977`) and `ctr_counter_simple` (`stride 2`, payload `6`). Including it makes the counts sum to 37 (BTMV) / 36 (Counter) instead of 30. `cb_oracle::CtrTableJson::bucket_counts()` (`crates/cb-oracle/src/model_json.rs:323-350`) does **not** strip it and does **not** expose the hash strings — **B02 must read `CtrTableJson.hash_map` directly.** | `[VERIFIED: LOCAL python dump of both fixtures, executed]` + `[VERIFIED: LOCAL model_json.rs:295-351 read]` |
| **E13** | **Existing bake tests that must stay green and must NOT be edited.** `crates/cb-train/src/ctr/final_ctr_test.rs` (211 lines): `e11_fixture()` (fn at `:115`, body `:115-122`) = `stringify_int_category(i % 3)` over 12 rows with `target_class = [1,0,1,1,0,0,1,0,1,1,0,1]`; `bake_emits_the_requested_type_and_denominator` (`:124-185`, whose Buckets block is `:142-151` and Counter block `:155-167`); `borders_bake_bytes_are_unchanged` (`:187-211` — frozen `hashes` at `:199`, `int_counts = [[0,4],[4,0],[1,3]]`, `shift.to_bits()` `:208`, `scale.to_bits()` `:209`, `counter_denominator == 0` `:210`). Also `binarized_target_mean_uses_class_over_border_count` (`:61-70`) which calls `accumulate_online(…, 2, 2)` **directly** (not via the bake) and asserts `mean_sum[0] == 1.0` — **this test is CORRECT and unaffected; it tests `build_final_ctr`'s pass-through at an explicitly chosen divisor of 2.** | `[VERIFIED: LOCAL final_ctr_test.rs read in full; line numbers re-confirmed by grep]` |
| **E14** | **The mount topology that makes D11 work.** `crates/cb-train/src/lib.rs:21` declares `mod ctr;` (**private**). `crates/cb-train/src/ctr/mod.rs:31-40` declares `pub mod online/calc_ctr/final_ctr/ctr_feature/bake`; `:42-53` mount the four `#[cfg(test)]` test modules. `mod_test.rs:9` already does `use super::ECtrType;` — a child module reading `ctr`'s items. | `[VERIFIED: CODEGRAPH ctr/mod.rs verbatim]` + `[VERIFIED: LOCAL lib.rs:1-30, mod_test.rs:1-50]` |
| **E15** | **No existing assertion breaks when the Sums double.** `ctr_data_roundtrip_test::from_baked_carries_mean_tables_for_btmv` (`:259-281`) asserts only `mean.len() == 3`, `!mean.is_empty()` and the anti-vacuity `any(sum != 0.0)`. `final_ctr_test::bake_emits_the_requested_type_and_denominator`'s BTMV block asserts `mean.len() == 3` and `any(sum != 0.0)`. `ctr_btmv_simple_oracle_test::btmv_baked_table_carries_a_non_empty_mean_vector` asserts lengths + `any(s != 0.0)`. `ctr_split_scoring_test`'s three bake calls all pass `ECtrType::Borders`. **⇒ ZERO diff is required on all of these; none needs a mechanical edit.** | `[VERIFIED: LOCAL all four files read at the cited ranges]` |
| **E16** | `ctr_btmv_simple_oracle_test::btmv_f64_sum_accumulation_diverges_from_upstream_on_this_fixture` (`:204-243`) panics only if some baked `Sum` is **not** an f32 round-trip fixed point. Today's Sums are `{2.5, 1.5, 2, 1, 2}`; after the fix `{5, 3, 4, 2, 4}` — **all exactly representable in f32 in both cases**, so this reporting test stays green (and its printed `maxdiff` should drop to ≤1e-5). | `[VERIFIED: LOCAL test read in full + executed today]` |
| **E17** | **MEASURED per-bucket divergence** (SPEC.md §1's table): every `Count` matches upstream exactly; every `Sum` is exactly half. `(5,7)→(2.5,7)`, `(3,7)→(1.5,7)`, `(4,6)→(2,6)`, `(2,5)→(1,5)`, `(4,5)→(2,5)`. **That our five baked hashes equal upstream's five real hashes also proves our bake chose the SAME projection (cat feature 1) and the same bucket set.** | `[VERIFIED: SESSION-MEASURED during E13 execution — NOT re-executed by the planner. B02's Red must record the actually-observed values verbatim; if they differ from this table, STOP AND REPORT.]` |
| **E18** | Hand-computed BTMV expectation on `e11_fixture()` (E13): buckets are `i % 3`, each holding 4 documents in first-seen order `0, 1, 2`. Class-1 counts per bucket: bucket0 (`i = 0,3,6,9` → classes `1,1,1,1`) = **4**; bucket1 (`i = 1,4,7,10` → `0,0,0,0`) = **0**; bucket2 (`i = 2,5,8,11` → `1,0,1,1`) = **3**. ⇒ correct `mean = [(4.0, 4), (0.0, 4), (3.0, 4)]`; today's buggy `mean = [(2.0, 4), (0.0, 4), (1.5, 4)]`. Cross-checks against the frozen `int_counts = [[0,4],[4,0],[1,3]]` (E13) bucket-for-bucket. | `[VERIFIED: LOCAL arithmetic over the literals in final_ctr_test.rs:113-122, cross-checked against the frozen Borders bake]` |

---

## 5. Execution waves and dependency order

```text
W0  RED / VERIFY
      B01 ∥ B06   (disjoint files, no production edit overlap)
        B01  src/ctr/final_ctr_test.rs   (+ per-bucket bake Red, + R1 discriminator)  [BTMV-01]
        B06  src/ctr/online_test.rs      (prefix producer VERIFICATION + mutation)    [BTMV-03]
      B02  SERIALIZED AFTER B06                                                       [BTMV-02]
        B02  tests/ctr_btmv_bake_upstream_table_test.rs  (NEW target, by hash)
        * NOT parallel with B06 (plan-check pass 1, MAJOR-1). B06's M-B06 mutation
          is a LIVE edit to `online.rs:321`, inside `online_mean_prefix`, whose sole
          production caller is `ctr_feature.rs:234` on the TRAINING path — and B02
          trains through `train_cat`. Run concurrently, B02's structural guards fail
          against B06's in-flight mutation, and the plan routes that to
          STOP-AND-REPORT: a FALSE ALARM that halts W0. File-level disjointness is
          not sufficient here; the hazard is a production edit, not an edit conflict.

W1  GREEN — the one-expression fix
      B03  src/ctr/bake.rs:196                                                        [BTMV-01, -02]

W2  REFACTOR — name the divisor, pin it, and disarm the helper trap
      B04  src/ctr/mod.rs + src/ctr/bake.rs + src/ctr/mod_test.rs                     [BTMV-01]

W3  GUARD — the non-mean payloads are provably untouched
      B05  src/ctr/final_ctr_test.rs (frozen Buckets/Counter bytes + mutation)        [BTMV-04]

W4  CLOSE-OUT — decisions, blast radius, evidence, diff gates
      B07                                                                             [BTMV-03, -04]

Dependency edges
      B01 ───────────┐
      B06 ──> B02 ───┴─> B03 ──> B04 ──> B05 ──> B07
Acyclic. Longest path: B06 -> B02 -> B03 -> B04 -> B05 -> B07   (6 tasks).
B06 -> B02 is a PRODUCTION-EDIT serialization, not a file conflict (§5, MAJOR-1).
```

**Parallelism justification.**

- **W0 is parallel for `B01 ∥ B06` ONLY.** B01 edits
  `src/ctr/final_ctr_test.rs`; B06 edits `src/ctr/online_test.rs`. No two of them
  write the same file, and neither mutation reaches a path the other observes.
- **B02 is NOT parallel with B06** (plan-check pass 1, MAJOR-1). File
  disjointness is the WRONG test here: B06's M-B06 is a live edit to production
  `online.rs:321`, on the training path B02 trains through via `train_cat`. Run
  concurrently, B02's structural guards fail against B06's in-flight mutation and
  the plan routes that to STOP-AND-REPORT — a false alarm that halts W0.
- **B06 must nevertheless be FULLY COMPLETE — mutation reverted — before W1
  starts.** Its mutation edits `crates/cb-train/src/ctr/online.rs:321`, which
  feeds the *training-side* CTR column that B03's E13 measurement depends on. A
  live B06 mutation would silently corrupt B03's parity number. **Gate: `git diff
  crates/cb-train/src/ctr/online.rs` must be EMPTY before B03 begins.**
- **`crates/cb-train/src/ctr/bake.rs` is an EXCLUSIVE resource** held by B03,
  then B04, then B05, in that order. **B04 and B05 each begin with a pre-flight
  `git diff crates/cb-train/src/ctr/bake.rs`** that must show only the hunks owned
  by completed predecessors (B03 for B04; B03 + B04 for B05). **A stray hunk is a
  STOP-AND-REPORT condition — report it, do not clean it up.**
- **`crates/cb-train/src/ctr/final_ctr_test.rs` is an EXCLUSIVE resource** held by
  B01, then B05. B05 is **additive only**: it must not touch B01's or E13's
  existing functions.
- B07 writes no code at all.

---

# WAVE W0 — RED / VERIFY

## B01 — PRIMARY RED: the bake's per-bucket `(Sum, Count)`, and the three-way divisor discriminator

- **Specs:** SPEC-BTMV-01 (primary); resolves SPEC.md §7 R1
- **Order:** 1 · **Depends on:** none · **Status:** pending
- **Files:** Modify `crates/cb-train/src/ctr/final_ctr_test.rs` (**ADDITIVE ONLY**)
- **Touches production code:** NO
- **Exclusive resource:** `crates/cb-train/src/ctr/final_ctr_test.rs`

### Objective

Catch the divisor error at **TABLE level**, in a unit test, with hand-computed
per-bucket values — so a future regression fails here rather than only in a
30-row end-to-end fixture. And build the **only** gate that can distinguish
`classes - 1` from `ECtrType::target_border_count(classes)` (§0, D2).

### Why the E13 oracle is not acceptable as the primary Red

`ctr_btmv_simple_oracle_test` fails today with `predictions are constant — the
gate would be vacuous` (§1.1) — a message consistent with a dozen unrelated
defects (no CTR split chosen, a broken projection, a border-space error, a leaf
assignment bug). It is the SECONDARY gate (B03). The PRIMARY Red must name the
bucket, the expected `(Sum, Count)`, the observed `(Sum, Count)`, and the
divisor that explains the ratio.

### Test construction (exact)

Append to `crates/cb-train/src/ctr/final_ctr_test.rs`, **after** the existing
`borders_bake_bytes_are_unchanged` (`:187-211`, currently the last item in the
file), under a new banner comment:

```text
// ---------------------------------------------------------------------------
// BUG-BTMV / SPEC-BTMV-01 — the whole-set bake divides by
// `targetClassesCount - 1` (online_ctr.cpp:914), NOT by `targetClassesCount`
// and NOT by GetTargetBorderCount (ctr_helper.h:34-42, the ONLINE-path helper).
// ---------------------------------------------------------------------------
```

The file already has `use crate::ctr::bake::bake_ctr_table;` inside the two E11
test functions (`:126`, `:189`); follow that established local-`use` style rather
than adding a file-level import.

**Test 1 — `btmv_bake_sums_class_one_documents_per_bucket` (THE PRIMARY RED).**

1. `let (cats, tc, proj) = e11_fixture();` — reuse the existing helper (E13).
   **Do not invent a new corpus**: `e11_fixture` is already cross-checked against
   the frozen Borders bake, which is what lets Test 1's expectations be *derived*
   rather than *asserted by fiat*.
2. `let t = bake_ctr_table(&cats, &proj, &tc, 2, 15, 0.5, 1.0,
   ECtrType::BinarizedTargetMeanValue).expect("bake must succeed");`
3. **Derivation comment (mandatory, verbatim intent).** State in the test body
   that buckets are `i % 3` in first-seen order `0, 1, 2`; that class-1 counts are
   `4, 0, 3` (E18); and that with the upstream divisor
   `targetClassesCount - 1 == 1` the `Sum` is *exactly* the class-1 count.
4. **Cross-check against the frozen Borders bake (the non-fiat step).** Bake the
   same corpus as `ECtrType::Borders` and assert, per bucket `b`:
   `assert_eq!(f64::from(t.mean[b].0), borders.int_counts[b][1] as f64,
       "bucket {b}: the BTMV Sum must equal the Borders N1 (class-1 count) — the
        whole-set divisor is targetClassesCount - 1 == 1 for binclf
        (online_ctr.cpp:914). Observed Sum {} vs N1 {}: the ratio names the wrong
        divisor.", …)`
   and `assert_eq!(t.mean[b].1, borders.int_counts[b][0] + borders.int_counts[b][1],
       "bucket {b}: Count must equal N0 + N1")`.
   *This is what makes Test 1 non-tautological: it ties the mean table to an
   INDEPENDENTLY FROZEN payload (`[[0,4],[4,0],[1,3]]`, E13) rather than to a
   literal the author chose.*
5. **Literal pin as well** (belt and braces, and the human-readable failure):
   `assert_eq!(t.mean, vec![(4.0f32, 4i64), (0.0, 4), (3.0, 4)],
       "hand-computed from e11_fixture (E18). Halved Sums ⇒ bake.rs:196 passed
        `classes` (2) as target_border_count instead of `classes - 1` (1).")`.
6. **ANTI-VACUITY (mandatory, both clauses, as SPEC-BTMV-02 requires and B01
   adopts):** assert some bucket has `Sum != Count` **and** `Sum != 0`:
   ```text
   assert!(t.mean.iter().any(|&(s, c)| f64::from(s) != c as f64 && s != 0.0),
       "every bucket has Sum == Count or Sum == 0 — a degenerate corpus would
        satisfy this test trivially; the corpus must contain a bucket with a
        MIXED target (bucket 2: 3 of 4, E18)");
   ```
   *(Bucket 2 = `(3.0, 4)` satisfies it. Bucket 0 = `(4.0, 4)` alone would not.)*
7. `assert!(t.int_counts.is_empty(), "the mean types carry (Sum, Count) pairs");`

**Test 2 — `bake_target_border_divisor_is_classes_minus_one_not_the_ctr_type_helper`
(THE R1 THREE-WAY DISCRIMINATOR — the ONLY gate that can catch a helper-based
"fix").**

Bake at `classes = 3` and assert the divisor is `2`:

```text
// A 3-class corpus. Production is binclf-only (boosting.rs:5582 hard-codes 2,
// E6), so this is a CHARACTERIZATION of `bake_ctr_table`'s public contract at a
// configuration production does not reach. It exists because at classes == 2 the
// the FIX and the HELPER are indistinguishable (both yield 1); the BUG
// yields 2 and is distinguishable, but only in the Sum MAGNITUDE, not in
// which-candidate-is-which — so classes = 3 is still required to tell the
// fix apart from the helper:
//
//   expression                                  classes=2   classes=3
//   ------------------------------------------  ---------   ---------
//   `classes`            (TODAY'S BUG)              2           3
//   `classes - 1`        (upstream, D1/E1)          1           2
//   `BTMV.target_border_count(classes)` (E4)        1           1
//
// STOP CONDITION: if this assertion ever needs to change, either upstream's
// CalcFinalCtrsImpl changed (re-read online_ctr.cpp:914) or someone routed the
// bake through GetTargetBorderCount (§0, D2). Do NOT adjust the expected value.
```

1. Corpus: 6 rows over 2 categories, e.g. `col = ["a","a","a","b","b","b"]`
   (built with `cb_data::stringify_int_category` for consistency with
   `e11_fixture`, or plain `String`s — either is fine, both are hashed the same
   way), `target_class = vec![0usize, 1, 2, 2, 2, 0]`.
2. `bake_ctr_table(&[col], &proj, &tc, 3, 15, 0.5, 1.0,
   ECtrType::BinarizedTargetMeanValue)`.
3. Expected under `classes - 1 == 2`: bucket "a" `Sum = (0 + 1 + 2)/2 = 1.5`,
   `Count = 3`; bucket "b" `Sum = (2 + 2 + 0)/2 = 2.0`, `Count = 3`.
   Assert `t.mean == vec![(1.5f32, 3i64), (2.0, 3)]` with a message naming all
   three candidate divisors and the values each would produce
   (`classes` → `(1.0, 3), (1.333…, 3)`; helper → `(3.0, 3), (4.0, 3)`).
4. **ANTI-VACUITY:** assert the three candidates are actually distinct here —
   `assert_ne!(3usize - 1, ECtrType::BinarizedTargetMeanValue.target_border_count(3));`
   and `assert_ne!(3usize, 3usize - 1);` — with the message *"this test is only a
   discriminator while these differ; if the helper ever returns `classes - 1` for
   BTMV, re-derive the discriminator before touching the expectations"*.

**Test 3 — `btmv_bake_at_one_class_is_unchanged_and_does_not_error`
(D3's floor characterization; GREEN today, must stay GREEN).**

`bake_ctr_table(&[col], &proj, &tc, 1, 15, 0.5, 1.0, BTMV)` with every
`target_class == 0`. Assert `is_ok()` and that every `Sum` is `0.0`. Message:
*"`accumulate_online` rejects `target_border_count == 0` (online.rs:176-180), so
the bake floors the divisor at 1 (`saturating_sub(1).max(1)`, the same idiom as
`online_mean_prefix`, online.rs:321). Without the floor this bake would start
returning `CbError::Degenerate` where it returns `Ok` today. Upstream's
`CalcFinalCtrsImpl` divides by 0 here and is undefined; the floor is a
deliberate, behavior-preserving divergence (D3)."*
**This test passes BEFORE and AFTER the fix — it is the proof that the floor
changes nothing.**

### TDD sequence

1. **Red** — write Tests 1–3. Run `cargo test -p cb-train --lib ctr::final_ctr_test`.
   **Expected:**
   - **Test 1 FAILS** — observed `mean == [(2.0, 4), (0.0, 4), (1.5, 4)]`, and the
     step-4 cross-check fails first on bucket 0 (`2 != 4`).
   - **Test 2 FAILS** — observed `[(1.0, 3), (1.333…, 3)]` (divisor `3`).
   - **Test 3 PASSES.**
   - **Every pre-existing test in this file PASSES** (`borders_bake_bytes_are_unchanged`,
     `bake_emits_the_requested_type_and_denominator`,
     `binarized_target_mean_uses_class_over_border_count`, …).
   **Record all failure text verbatim, including the observed tuples.**
   **If Test 1 passes on first write, STOP** — the defect is not where this plan
   says it is; report before proceeding.
2. **Green** — none in this task. B01 is Red-only by construction; B03 turns it
   green.
3. **Refactor** — none.
4. **Verify** —
   - `cargo clippy --workspace --all-targets` — no NEW warning (§1.2's
     pre-existing unused-import warning is accepted).
   - `git diff --stat` shows **only** `crates/cb-train/src/ctr/final_ctr_test.rs`;
     **zero** diff on any of the 11 CTR oracles.
   - `git diff crates/cb-train/src/ctr/final_ctr_test.rs` is **purely additive** —
     no existing line removed or reworded (E13/E15).

### Completion criteria

- [ ] Tests 1 and 2 fail; each failure names the bucket, the observed tuple, the
      expected tuple and the divisor that explains the ratio.
- [ ] Test 1's step-4 cross-check against the frozen `borders.int_counts` is
      present (the non-fiat step).
- [ ] Test 2 enumerates all three candidate divisors and carries the STOP CONDITION.
- [ ] Test 3 passes.
- [ ] Both anti-vacuity guards present and passing.
- [ ] The diff on `final_ctr_test.rs` is purely additive.
- [ ] No production file modified.

### Risks and guardrails

- **R:** an executor "simplifies" Test 1 by deleting the Borders cross-check,
  leaving only literals. **M:** step 4 is a completion criterion; the literals
  alone would be an author-chosen fiat.
- **R:** Test 2 is dismissed as "testing an unreachable config". **M:** §0 states
  it is the *only possible* discriminator; D2 depends on it; it is a
  characterization with a STOP CONDITION, exactly like BUG-CTRB's Test E.
- **R:** temptation to also fix the divisor while here. **M:** B01 touches no
  production file; the split into Red (B01) and Green (B03) is what makes the
  failure attributable.

---

## B02 — RED: the baked BTMV table must match upstream's committed `ctr_data` bucket-for-bucket

- **Specs:** SPEC-BTMV-02
- **Order:** 3 · **Depends on:** **B06** (see §5 — B06's mutation is a live edit to the training path B02 trains through) · **Status:** pending
- **Files:** **Create** `crates/cb-train/tests/ctr_btmv_bake_upstream_table_test.rs`
- **Touches production code:** NO

### Objective

Anchor the baked table to **catboost 1.2.10's own committed bytes**, by hash, so
the gate is at TABLE level (SPEC-BTMV-02) rather than at prediction level. B01
proves *internal* correctness against a hand-derivation; B02 proves *upstream
agreement*. Both are needed — a systematically wrong-but-self-consistent
derivation would satisfy B01.

### Why a separate target rather than adding to the E13 file

`crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs` is one of the three
**protected untracked paths** (§1). This plan does not edit it. A new target also
keeps B02 free of edit conflicts with B01/B06 — but B02 still runs AFTER B06
(§5, MAJOR-1): the hazard there is B06's live production mutation, not a shared
file.

### Test construction (exact)

File header (§3.0):

```rust
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
```

Harness to copy **verbatim, field-for-field** from
`crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs`: `SCENARIO`, `fixture()`
(`:33`), `load_cat_columns()` (`:44`), the params builder (`:59`, named
`counter_params()` — **copy the name as-is; §1.2 forbids renaming it in the
source file, and diverging here would make the two harnesses hard to diff**), and
the `train_cat(...)` call shape (`:115-125`). Do **not** re-derive the params:
every field must be explicit and identical, or the trained model will differ from
the one E13 measures.

**Test 1 — `baked_btmv_table_matches_upstream_ctr_data_bucket_for_bucket`.**

1. Train through production `train_cat`, exactly as E13's `fit()` does, obtaining
   `(trained, baked)`.
2. Select our table:
   ```text
   let ours = baked.tables.iter()
       .filter(|t| t.ctr_type == ECtrType::BinarizedTargetMeanValue.as_i8())
       .collect::<Vec<_>>();
   assert_eq!(ours.len(), 1,
       "expected exactly ONE baked BTMV table; got {}. More than one means the \
        trainer chose a second BTMV projection — a STRUCTURAL divergence from \
        upstream (which committed exactly one, on cat feature 1). STOP AND REPORT.",
       ours.len());
   let ours = ours[0];
   assert_eq!(ours.projection.cat_features(), &[1usize],
       "upstream's committed CTR is on cat_feature_index 1 (E11); ours is on {:?}. \
        A different chosen projection is a STRUCTURAL parity finding, NOT a test \
        to weaken. STOP AND REPORT.", ours.projection.cat_features());
   ```
3. Read upstream's table with the **typed** oracle struct, but the **raw**
   `hash_map` field (E12 — `bucket_counts()` neither strips the sentinel nor
   exposes hashes):
   ```text
   let mj = cb_oracle::load_model_json(&fixture(&format!("{SCENARIO}/model.json")))…;
   let (key, table) = mj.ctr_data.iter()
       .find(|(k, _)| k.contains("\"type\":\"BinarizedTargetMeanValue\""))
       .expect("the committed model.json must carry a BinarizedTargetMeanValue ctr_data entry");
   assert_eq!(table.hash_stride, 3, "a mean CTR table is (hash, Sum, Count)");
   ```
   Then walk `table.hash_map.chunks_exact(3)`:
   - `hash = chunk[0].as_str()…parse::<u64>()` (upstream emits the hash as a
     **string**, E11);
   - **SKIP `hash == u64::MAX`** with an inline comment citing E12: *"upstream's
     dense hash map leads with a `u64::MAX` EMPTY-SLOT sentinel whose payload is
     stale memory (here `3, 7`, duplicating a real bucket). Including it inflates
     the count total from 30 to 37."*
   - `sum = chunk[1].as_f64()`, `count = chunk[2].as_i64()`.
4. **Structural guards before comparing:**
   ```text
   assert_eq!(upstream.len(), 5, "expected 5 real buckets after dropping the \
       u64::MAX sentinel (cat1 cardinality is 5, config.json)");
   assert_eq!(upstream.iter().map(|b| b.count).sum::<i64>(), 30,
       "the real buckets' Counts must sum to n_rows = 30; a different total means \
        the sentinel filter is wrong (E12)");
   assert_eq!(ours.mean.len(), ours.hashes.len(), "one (Sum, Count) per bucket");
   assert_eq!(ours.hashes.len(), upstream.len(), "bucket-count mismatch");
   ```
5. **THE COMPARISON — by hash, not by index** (our first-seen bucket order need
   not equal upstream's map order):
   ```text
   for (i, &h) in ours.hashes.iter().enumerate() {
       let u = upstream.iter().find(|b| b.hash == h).unwrap_or_else(|| panic!(
           "baked hash {h} is absent from upstream's committed ctr_data {:?}. \
            A hash-set mismatch is a STRUCTURAL finding (different projection or \
            different category hashing), NOT a divisor bug. STOP AND REPORT.",
           upstream.iter().map(|b| b.hash).collect::<Vec<_>>()));
       assert_eq!(ours.mean[i].1, u.count,
           "bucket {h}: Count {} != upstream {}", ours.mean[i].1, u.count);
       assert_eq!(f64::from(ours.mean[i].0), u.sum,
           "bucket {h}: Sum {} != upstream {}. Ratio {:.6} — a ratio of exactly 0.5 \
            means the bake divided by `classes` (2) instead of `classes - 1` (1); \
            see online_ctr.cpp:914 and PLAN §0.",
           ours.mean[i].0, u.sum, f64::from(ours.mean[i].0) / u.sum);
   }
   ```
   **Exact `==` on the Sums is legitimate**: with the correct divisor every
   upstream Sum is a small integer (`5, 3, 4, 2, 4`, E11), exactly representable
   in both `f32` and `f64`. State that in a comment so no one relaxes it to an
   epsilon compare.
6. **ANTI-VACUITY (mandatory, exactly as SPEC-BTMV-02 requires):**
   ```text
   assert!(upstream.iter().any(|b| b.sum != b.count as f64 && b.sum != 0.0),
       "no upstream bucket has Sum != Count and Sum != 0 — a degenerate corpus \
        would satisfy this comparison trivially");
   ```
   *(`3644720124901778394` = `(4, 6)` satisfies it, E11.)*

**Test 2 — `upstream_btmv_sums_are_integers_which_pins_the_divisor_at_one`
(characterization; GREEN today and after — it reads only the committed fixture).**

Assert every real upstream `Sum` equals its own `round()`, with the message:
*"for binclf `targetClass ∈ {0,1}`, so `Sum = Σ targetClass / targetBorderCount`
is an integer **iff** `targetBorderCount == 1`. Upstream's committed Sums
`{5,3,4,2,4}` are integers ⇒ upstream's whole-set divisor is 1
(`targetClassesCount - 1`, online_ctr.cpp:914), NOT 2. This is fixture-side
evidence for D1 that is independent of our implementation."*

### TDD sequence

1. **Red** — `cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test`.
   **Expected:** **Test 1 FAILS** at the first Sum comparison, reporting a ratio
   of **exactly 0.5**; **Test 2 PASSES**; the structural guards (steps 2 and 4)
   **PASS** — a failure there instead means E17's session measurement no longer
   holds and is a **STOP-AND-REPORT**. Record the observed `(hash, Sum, Count)`
   triples verbatim and compare them to E17's table.
2. **Green** — none here (B03).
3. **Refactor** — none.
4. **Verify** — `cargo clippy --workspace --all-targets`; `git diff --stat` shows
   only the one new untracked file; **zero** diff on the 11 oracles and on the
   three protected paths (`git status --short` must still list
   `crates/cb-oracle/fixtures/ctr_btmv_simple/` and
   `crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs` untouched).

### Completion criteria

- [ ] Test 1 fails at the Sum comparison with a reported ratio of exactly `0.5`.
- [ ] Test 2 passes.
- [ ] The `u64::MAX` sentinel is explicitly skipped with the E12 citation, and the
      `sum(count) == 30` guard passes.
- [ ] The projection guard asserts `[1]` and passes.
- [ ] The anti-vacuity guard passes.
- [ ] No fixture regenerated; neither `gen_fixtures.py` invoked.
- [ ] No production file modified; no protected path modified.

### Risks and guardrails

- **R:** the sentinel is silently included → a phantom sixth bucket and a
  confusing "hash absent" failure. **M:** step 4's `len() == 5` and
  `sum(count) == 30` guards fail loudly and first.
- **R:** an executor copies E13's params *approximately*. **M:** field-for-field
  copy is a completion criterion; a params drift changes the trained structure and
  would surface as a bogus "structural divergence" STOP.
- **R:** `serde_json` `as_f64()` on an integer JSON value. **M:** `as_f64()` is
  defined for JSON integers; step 5's comment pins why exact `==` is used.

---

## B06 — VERIFY (guard): the ONLINE prefix producer is NOT affected

- **Specs:** SPEC-BTMV-03
- **Order:** 2 · **Depends on:** none · **Status:** pending
- **Files:** Modify `crates/cb-train/src/ctr/online_test.rs` (**ADDITIVE ONLY**);
  **temporarily** mutates `crates/cb-train/src/ctr/online.rs` during step 4
- **Touches production code:** NO permanently; YES temporarily (mutation only)
- **Exclusive resource:** `crates/cb-train/src/ctr/online.rs` and
  `crates/cb-train/src/ctr/online_test.rs` — **released before W1 begins**

### Objective

SPEC-BTMV-03 says: *"believed correct — verify with a test; do not assume. If it
is also wrong, this spec's scope widens and the plan must say so."*
**The planner verified it and it is CORRECT (D6, E2, E10). B06 turns that
verification into a permanent gate and proves the gate is falsifiable.**

### The finding this task encodes

Upstream's ONLINE mean path passes `targetClassesCount - 1` **as a literal** at
`online_ctr.cpp:762`, **not** via `GetTargetBorderCount`
(which appears at `:738`/`:741` for *allocation sizing* and at `:777` for the
*class* prefix types) — E2. `online_mean_prefix` computes exactly
`classes.saturating_sub(1).max(1)` (`online.rs:321`) — E10.
**⇒ It is correct at every class count, not merely at binclf.**
**SPEC-BTMV-03's scope-widening branch is CLOSED.**

### Why this is a GUARD, not a Red

It passes before and after B03: `online_mean_prefix` is not being changed.
Falsifiability comes from the §3.1 mutation check. State that in a comment above
the new tests (precedent wording: `boosting_test.rs:502-504`).

### Test construction (exact)

Append **at the END** of `crates/cb-train/src/ctr/online_test.rs` (570 lines), under
a new `// BUG-BTMV / SPEC-BTMV-03` banner. The E07 block this task extends is
`:412-476` — banner `:412-414`,
`btmv_prefix_reads_sum_and_count_before_incrementing` (fn at `:417`),
`btmv_sum_is_accumulated_in_f32_not_f64` (fn at `:447`) — followed by an
unrelated E08 block (`:478-570`). **Appending at the end (not mid-file) keeps the
diff purely additive.** Reuse the E07 tests' local-`use` style
(`use crate::ctr::online::online_mean_prefix;` inside the fn). **Do not edit any
existing test.**

**Test 1 — `online_mean_prefix_divides_by_classes_minus_one_at_three_classes`.**
The binclf case is already covered by
`btmv_prefix_reads_sum_and_count_before_incrementing` (`:416-443`, divisor 1).
Test 1 adds the case that **discriminates** the two upstream rules:

```text
// classes = 3. The three candidate divisors DIFFER here:
//   `classes - 1`                              (upstream, online_ctr.cpp:762) -> 2
//   `BTMV.target_border_count(classes)`        (GetTargetBorderCount)         -> 1
//   `classes`                                                                -> 3
// This is the ONLINE analogue of B01 Test 2, and it is why `online_mean_prefix`
// is provably correct at ALL class counts rather than only at binclf.
let perm: Vec<i32> = vec![0, 1, 2, 3];
let bins: Vec<u32> = vec![0, 0, 0, 0];
let tc:   Vec<usize> = vec![2, 0, 1, 2];
let got = online_mean_prefix(&perm, &bins, &tc, 3, 0.5).expect("mean prefix");
// Read-before-increment, adds targetClass / 2:
//   doc 0 reads (0.0, 0) then adds 1.0
//   doc 1 reads (1.0, 1) then adds 0.0
//   doc 2 reads (1.0, 2) then adds 0.5
//   doc 3 reads (1.5, 3) then adds 1.0
assert_eq!(got.sum,   vec![0.0f32, 1.0, 1.0, 1.5]);
assert_eq!(got.count, vec![0i64, 1, 2, 3]);
```
Message on the `sum` assertion must name all three candidates and the values each
would produce, and cite `online_ctr.cpp:762`.

**Test 2 — `online_mean_prefix_and_the_whole_set_bake_agree_at_binclf`
(the cross-path consistency statement — the reason BUG-BTMV existed at all).**

Assert, in one place, that the ONLINE divisor and the WHOLE-SET divisor agree at
`classes = 2` and **also** at `classes = 3`, by comparing the two producers'
per-bucket totals on the same corpus:

- run `online_mean_prefix(perm, bins, tc, classes, prior)` over an identity
  permutation and take the **final** history — i.e. `sum.last() + tc.last()/d`
  is awkward, so instead assert the **prefix-sum invariant**: for every `i > 0`,
  `got.sum[i] - got.sum[i-1]` equals `tc[i-1] as f32 / (classes - 1) as f32`.
- Message: *"the ONLINE prefix (online_ctr.cpp:762) and the WHOLE-SET bake
  (online_ctr.cpp:914) use the SAME divisor `targetClassesCount - 1`. BUG-BTMV was
  precisely the two paths disagreeing: the prefix was right and the bake was
  wrong, so the trained structure and the baked table described different CTR
  values. If this ever fails, the two paths have diverged again."*

**Test 3 — `online_mean_prefix_floors_the_divisor_at_one`
(D3's floor, on the side that already has it).**
`online_mean_prefix(&[0], &[0], &[0], 1, 0.5)` and `(…, 0, 0.5)` must both be
`Ok` with `sum == [0.0]`. Message citing `online.rs:319-321`.

### TDD sequence

1. **Write and run** — `cargo test -p cb-train --lib ctr::online_test`.
   > **Scope this command to `ctr::online_test` and NOTHING wider** (plan-check
   > pass 1, MAJOR-2). A broader `cargo test -p cb-train --lib ctr::` sweeps in
   > `final_ctr_test`, mounted into the SAME `--lib` target (`ctr/mod.rs:49-50`),
   > where B01 deliberately leaves two tests RED until B03 lands. That criterion
   > would be unsatisfiable, and the tempting "fix" is to weaken B01's Reds —
   > which is FORBIDDEN.
   **Expected: ALL THREE PASS on first write** (this is a verification, not a
   Red). **Record the result verbatim.** If any fails, **STOP AND REPORT** —
   SPEC-BTMV-03's scope-widening branch has opened and this plan must be revised
   before B03 lands.
2. **Green** — none (nothing to implement).
3. **Refactor** — none.
4. **MUTATION CHECK (§3.1) — one named mutation.**

   **M-B06 — proves the tests are load-bearing.**
   At `crates/cb-train/src/ctr/online.rs:321`, change
   `let divisor = classes.saturating_sub(1).max(1) as f32;`
   to `let divisor = classes.max(1) as f32;`
   **Expected outcomes — record ALL of them:**
   1. **Test 1 FAILS** — observed `sum == [0.0, 0.666…, 0.666…, 1.0]` (divisor 3).
   2. **Test 2 FAILS** on the prefix-delta invariant.
   3. **Test 3 still PASSES** (at `classes = 1` both forms give 1) — *expected,
      record it, do not treat it as a weak guard.*
   4. **The pre-existing `btmv_prefix_reads_sum_and_count_before_incrementing`
      (fn at `:417`) FAILS** — at binclf `classes.max(1) == 2 != 1`. Record it;
      **do not edit that test.**
   Revert **manually** (never `git checkout --`); re-run; confirm green.
5. **Verify** —
   - `cargo test -p cb-train --lib ctr::online_test` — green. **Scoped
     deliberately** (plan-check pass 1, MAJOR-2): a wider `--lib ctr::` sweeps in
     `final_ctr_test` (same `--lib` target, `ctr/mod.rs:49-50`), where B01 leaves
     two tests RED until B03. Widening it makes this gate unsatisfiable and
     invites weakening B01's Reds, which is FORBIDDEN.
   - `git diff crates/cb-train/src/ctr/online.rs` must be **EMPTY**. This is the
     hand-off gate for W1 (§5): a residual mutation here would silently corrupt
     B03's E13 parity measurement.
   - `cargo clippy --workspace --all-targets`.

### Completion criteria

- [ ] Tests 1–3 pass on first write; the result recorded verbatim.
- [ ] Test 1 discriminates all three candidate divisors at `classes = 3`.
- [ ] **M-B06 recorded — all four outcomes**, including outcome 3 staying green.
- [ ] Mutation reverted **manually**; `git diff crates/cb-train/src/ctr/online.rs`
      is **EMPTY**.
- [ ] The diff on `online_test.rs` is purely additive; no existing test edited.
- [ ] SPEC-BTMV-03 recorded as **CLOSED — not affected** (feeds B07).

### Risks and guardrails

- **R:** an executor "harmonizes" `online_mean_prefix` with the new bake helper
  (B04) by making both call one function. **M:** D6 forbids changing
  `online_mean_prefix`; the two divisors are the same *expression* by coincidence
  of upstream's two independent code paths (E1 vs E2), and B04's helper doc must
  say so. Merging them would create exactly the coupling that would hide a future
  divergence.
- **R:** the mutation is left in place and poisons B03. **M:** the empty-diff
  hand-off gate is a completion criterion and is re-checked in B03's pre-flight.

---

# WAVE W1 — GREEN

## B03 — GREEN: pass the whole-set target border count (ONE expression)

- **Specs:** SPEC-BTMV-01, SPEC-BTMV-02
- **Order:** 4 · **Depends on:** B01, B02, B06 · **Status:** pending
- **Files:** Modify `crates/cb-train/src/ctr/bake.rs` (one expression + comments)
- **Touches production code:** YES — **this is the only task that changes behavior**
- **Exclusive resource:** `crates/cb-train/src/ctr/bake.rs` (taken first)

### Pre-flight (mandatory)

1. `git diff crates/cb-train/src/ctr/online.rs` must be **EMPTY** (B06's hand-off
   gate). A residual mutation would corrupt this task's parity measurement.
2. `git status --short` must still list the three protected untracked paths (§1).
3. Re-run B01 and B02 to confirm they are still red:
   `cargo test -p cb-train --lib ctr::final_ctr_test` and
   `cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test`.

### The change — exactly this, nowhere else

At `crates/cb-train/src/ctr/bake.rs:196`:

```text
-    let acc = accumulate_online(&key_refs, &target_class_n, &target_zero, classes, classes)?;
+    // The WHOLE-SET bake's target border count is `targetClassesCount - 1`,
+    // UNCONDITIONALLY and independently of `ctr_type`
+    // (`CalcFinalCtrsImpl`, online_ctr.cpp:914 — the expression sits OUTSIDE the
+    // per-type switch). It is consumed by exactly one accumulator field,
+    // `binarized_mean` (online.rs:212-214), which only the
+    // BinarizedTargetMeanValue arm of `build_final_ctr` reads.
+    //
+    // *** NOT `ctr_type.target_border_count(classes)`. *** That helper mirrors
+    // `GetTargetBorderCount` (ctr_helper.h:34-42), which upstream uses on the
+    // ONLINE path for allocation sizing (online_ctr.cpp:738/741) and for the
+    // CLASS prefix types (:777) — never in CalcFinalCtrsImpl. The two agree at
+    // binclf and DIFFER for Buckets, so routing through the helper would be
+    // undetectable here and wrong at multiclass. See PLAN §0 / D2.
+    //
+    // The `.max(1)` floor: `accumulate_online` rejects `target_border_count == 0`
+    // with a typed error (online.rs:176-180), so a single-class corpus would
+    // start erroring without it. Same idiom as `online_mean_prefix`
+    // (online.rs:321). Behavior at `classes == 1` is unchanged (`classes == 0` flips Err->Ok; unreachable, see D3) either way: every
+    // target_class is 0, so every Sum is 0.
+    let target_border_count = classes.saturating_sub(1).max(1);
+    // 4th arg `classes` = TargetClassesCount (the class-histogram WIDTH,
+    // online_ctr.cpp:909-911/930-934) — correct, and deliberately NOT the same
+    // quantity as the 5th.
+    let acc = accumulate_online(
+        &key_refs,
+        &target_class_n,
+        &target_zero,
+        classes,
+        target_border_count,
+    )?;
```

**Nothing else changes.** In particular: no change to `accumulate_online`
(`online.rs`), `online_mean_prefix`, `ECtrType::target_border_count`
(`ctr/mod.rs:123-131`), `build_final_ctr`, `ctr_feature.rs`, `boosting.rs:5582`
(the 4th-argument `2`), or anything in `cb-model`.

Also update `bake.rs`'s module doc (`:1-44`) where it says the accumulation is
`[N0, N1]` class counts, to note that the same accumulator carries the mean
history and that the whole-set divisor is `targetClassesCount - 1` — **comment
only**.

### TDD sequence

1. **Red** — established by B01 and B02 (re-confirmed in pre-flight).
2. **Green** — apply the change. Then, **in order**:
   - `cargo test -p cb-train --lib ctr::final_ctr_test` → **all green**,
     including B01 Tests 1–3 and every pre-existing test in the file
     (notably `borders_bake_bytes_are_unchanged` — SPEC-BTMV-04's first gate).
   - `cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test` → **all green**.
   - `cargo test -p cb-train --test ctr_btmv_simple_oracle_test` → **4/4 green**
     (the SECONDARY / integration Red). The `predictions are constant` guard must
     stop firing **and** `compare_stage` must pass at ≤1e-5.

   > **BRANCH — read before running the E13 gate.** The expectation that E13 goes
   > green rests on: the baked table becoming **bit-identical** to upstream's
   > committed `ctr_data` (B02, gated), the training-side prefix already being
   > upstream-exact (B06/E10), and the persisted border already being in value
   > space (BUG-CTRB, landed at `c21f44a`). That is a strong chain, **but it is an
   > inference, not a measurement** `[INFERRED]`.
   >
   > **If B01 and B02 are green and E13 is still red, STOP AND REPORT.** Record:
   > (a) whether the `predictions are constant` guard still fires or the failure
   > has moved to `compare_stage`; (b) the new `max |diff|` (via `-- --nocapture`,
   > §1.1); (c) the baked table (which B02 proves is now correct). A second,
   > independent defect is then present and needs its own SPEC. **Do NOT weaken
   > the E13 gate, do NOT touch the fixture, and do NOT regenerate anything.**
3. **Refactor** — **none in this task.** Extraction of the named helper is B04, so
   the behavior change and the structural change are two reviewable diffs.
4. **Verify** — the **mandatory regression scope**, run in full (§3.2):
   - the 11 CTR oracles block (9 cb-train + 2 cb-model) — **all green**, **zero
     diff on all eleven files** (E15 predicts no mechanical edit is needed;
     **confirm by running, do not assert from theory**)
   - the 3 one-hot targets — **all green**
   - `cargo test -p cb-train --test ctr_counter_simple_oracle_test` — **4/4**
   - the three BUG-CTRB gates — `ctr_border_space_test`,
     `ctr_border_upstream_anchor_test`, `ctr_border_cbm_roundtrip_test` — green
   - `cargo test -p cb-model --test ctr_nonmean_byte_identity_test` — green
   - `cargo test -p cb-model --test cbm_oracle_test --test json_oracle_test --test float_only_byte_identity_test`
   - `cargo test -p cb-train --no-fail-fast` and `cargo test -p cb-model --no-fail-fast`
     — the **only** permitted failure is the pre-existing
     `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors`
   - `cargo clippy --workspace --all-targets` (§1.2's warning excepted)
   - **Diff gate:** `git diff --stat` must show `crates/cb-train/src/ctr/bake.rs`
     and **nothing else**; `git diff crates/cb-train/src/ctr/bake.rs` must contain
     **exactly one changed call expression** (plus the new `let` binding) and
     comment-only hunks.

### Completion criteria

- [ ] Pre-flight: `git diff crates/cb-train/src/ctr/online.rs` was empty; the three
      protected paths intact.
- [ ] `git diff crates/cb-train/src/ctr/bake.rs` changes exactly one call
      expression; the 4th argument is still `classes`.
- [ ] B01 Tests 1–3 green; B02 Tests 1–2 green.
- [ ] `borders_bake_bytes_are_unchanged` green (SPEC-BTMV-04, first gate).
- [ ] `ctr_btmv_simple_oracle_test` **4/4**, `max |diff| ≤ 1e-5`, and the
      `predictions are constant` guard no longer fires — **or** the STOP-AND-REPORT
      branch executed with all three recorded facts.
- [ ] 11 CTR oracles green with **zero diff** on all eleven.
- [ ] 3 one-hot targets green; `ctr_counter_simple_oracle_test` 4/4; the three
      BUG-CTRB gates green; `ctr_nonmean_byte_identity_test` green.
- [ ] Sweeps show no new failure beyond the recorded pre-existing monotone one.
- [ ] `cargo clippy --workspace --all-targets` clean of new warnings.

### Risks and guardrails

- **R (principal):** an executor implements SPEC.md §3's retracted advice
  (`ctr_type.target_border_count(classes)`). **M:** §0 and D2 retract it in the
  plan; the production comment retracts it in the source; **B01 Test 2 is the only
  runtime detector and it FAILS under that implementation** (it would produce
  `(3.0, 3), (4.0, 3)` at `classes = 3` instead of `(1.5, 3), (2.0, 3)`).
- **R:** the 4th argument is "helpfully" changed too. **M:** D4/E3; the frozen
  `borders_bake_bytes_are_unchanged` (`int_counts = [[0,4],[4,0],[1,3]]`) fails
  immediately if the class-histogram width moves.
- **R:** E13 stays red and an executor starts changing other production code
  hunting for green. **M:** the explicit STOP-AND-REPORT branch in step 2.
- **R:** disk exhaustion during the full sweeps. **M:** `df -h /home` before the
  `--no-fail-fast` sweeps; 629 G free at plan time. If `target/` growth becomes a
  problem, run the targeted blocks and record that the sweeps were deferred —
  **do not** delete `.cargo/config.toml`.

---

# WAVE W2 — REFACTOR

## B04 — REFACTOR: name the whole-set divisor, pin it, and disarm the helper trap

- **Specs:** SPEC-BTMV-01
- **Order:** 5 · **Depends on:** B03 · **Status:** pending
- **Files:** Modify `crates/cb-train/src/ctr/mod.rs`;
  Modify `crates/cb-train/src/ctr/bake.rs`;
  Modify `crates/cb-train/src/ctr/mod_test.rs` (**ADDITIVE ONLY**)
- **Touches production code:** YES — behavior-preserving extraction + comments
- **Exclusive resource:** `crates/cb-train/src/ctr/bake.rs` (taken over from B03)

### Pre-flight (mandatory)

`git diff crates/cb-train/src/ctr/bake.rs` must show **only B03's hunks** (one
changed call expression, the new `let` binding, comment-only hunks). **A stray
hunk is a STOP-AND-REPORT condition** — report it, do not clean it up.

### Objective

Put the two divisors **side by side, named and documented**, so that proximity can
never again make them look interchangeable — the structural fix for the class of
defect BUG-BTMV belongs to. And give SPEC-BTMV-01 a direct unit test on the
divisor rule itself, independent of any bake.

### The extraction (D11)

Add to `crates/cb-train/src/ctr/mod.rs`, **immediately after** the `impl ECtrType`
block (i.e. after `:189`, before the `CounterCalcMethod` doc at `:191`), as a
**private free function** — `bake` is a child module of `ctr` and may call it;
`mod_test` is a child module of `ctr` and may test it. **No `pub`, no
`pub(crate)`, no new `pub use` in `lib.rs`.**

```text
/// The target border count used by the **WHOLE-SET** CTR bake
/// (`CalcFinalCtrsImpl`, `online_ctr.cpp:914`).
///
/// ```text
/// online_ctr.cpp:914   int targetBorderCount = targetClassesCount - 1;
/// online_ctr.cpp:920   elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount);
/// ```
///
/// # This is NOT [`ECtrType::target_border_count`]
///
/// Upstream has **two** target-border-count rules and they are not the same
/// function:
///
/// | path | rule | upstream site |
/// |---|---|---|
/// | whole-set bake | `targetClassesCount - 1`, **type-independent** | `online_ctr.cpp:914` (this fn) |
/// | online, mean | `targetClassesCount - 1`, passed as a literal | `online_ctr.cpp:762` (`online_mean_prefix`) |
/// | online, alloc + class types | `GetTargetBorderCount(ctrInfo, …)`, **type-dependent** | `online_ctr.cpp:738/741, :777` ([`ECtrType::target_border_count`]) |
///
/// `GetTargetBorderCount` is NEVER called inside `CalcFinalCtrsImpl`. The two
/// rules agree for Borders / BinarizedTargetMeanValue / Counter at binary
/// classification and **DIFFER for `Buckets`** (the helper returns
/// `target_classes_count`, this returns `target_classes_count - 1`), so
/// substituting one for the other is undetectable at binclf and wrong at
/// multiclass. **BUG-BTMV was the bake passing `target_classes_count` itself.**
///
/// # The `.max(1)` floor
///
/// [`online::accumulate_online`] rejects `target_border_count == 0` with a typed
/// error, so a single-class corpus would begin returning `CbError::Degenerate`
/// without the floor. Behavior at `target_classes_count == 1` is identical either
/// way (every `target_class` is 0, so every `Sum` is 0). At
/// `target_classes_count == 0` the floor flips `Err(Degenerate)` to `Ok` with a
/// degenerate table — unreachable, the sole production caller hard-codes 2. Same idiom as
/// [`online::online_mean_prefix`]. Upstream divides by 0 here and is undefined.
///
/// # Deliberately NOT shared with `online_mean_prefix`
///
/// The two expressions coincide because upstream's two independent code paths
/// happen to use the same rule (`:762` and `:914`), not because they are one
/// rule. Merging them would hide a future upstream divergence. See PLAN §0.
const fn final_ctr_target_border_count(target_classes_count: usize) -> usize {
    target_classes_count.saturating_sub(1).max(1)
}
```

> `usize::max` is not `const` on all toolchains; if `const fn` does not compile,
> drop `const` (a plain `fn`) rather than rewriting the expression — the
> expression form is what B01 Test 2 and B04 Test 1 pin.

Then in `crates/cb-train/src/ctr/bake.rs`, replace B03's inline binding with
`let target_border_count = super::final_ctr_target_border_count(classes);`,
**keeping** the `*** NOT ctr_type.target_border_count(classes) ***` warning
comment at the call site (shortened to a one-line pointer at the helper's doc).

### The counter-comment on the helper (comment-only, load-bearing)

Add to `ECtrType::target_border_count`'s doc block
(`crates/cb-train/src/ctr/mod.rs:108-121`), **without touching its body**:

```text
/// # Which path uses this
///
/// This mirrors `GetTargetBorderCount` (`ctr_helper.h:34-42`), which upstream
/// calls on the **ONLINE** path only: for CTR-data allocation sizing
/// (`online_ctr.cpp:738/741`) and for the CLASS prefix types (`:777`). It is
/// **NOT** the whole-set bake's divisor — see
/// [`final_ctr_target_border_count`], which is `targetClassesCount - 1`
/// unconditionally (`online_ctr.cpp:914`). The two DIFFER for
/// [`Buckets`](Self::Buckets). Do not substitute one for the other (BUG-BTMV).
///
/// NOTE: this helper currently has **no production caller** — the online
/// allocation path does not yet route through it. Do not "fix" that by wiring it
/// into the bake.
```

### Tests (source/test separation — existing `mod_test` mount, E14)

Append to `crates/cb-train/src/ctr/mod_test.rs`. It uses `use super::ECtrType;`
(`:9`); extend that with `use super::final_ctr_target_border_count;`. **Do not add
a file-level `#![allow(...)]`** (§3.0) and **do not edit
`target_border_count_is_two_for_buckets_and_one_for_the_rest`** (`:24-48`).

**Test 1 — `final_ctr_target_border_count_is_classes_minus_one`.**
Pin the rule directly: `(2 → 1), (3 → 2), (5 → 4), (10 → 9)`, message citing
`online_ctr.cpp:914`.

**Test 2 — `final_ctr_target_border_count_floors_at_one`.**
`(1 → 1)`, `(0 → 1)`, message citing `online.rs:176-180` and D3.

**Test 3 — `the_two_target_border_rules_differ_for_buckets`
(the structural statement, over `ALL_TYPES`).**
Using the existing `ALL_TYPES` array (`:14-21`):
```text
for t in ALL_TYPES {
    let online = t.target_border_count(3);
    let bake   = final_ctr_target_border_count(3);
    if matches!(t, ECtrType::Buckets) {
        assert_ne!(online, bake, "Buckets is exactly where the two rules diverge \
            (GetTargetBorderCount returns targetClassesCount; CalcFinalCtrsImpl \
             returns targetClassesCount - 1) — this inequality is WHY the bake \
             must not route through the ECtrType helper (BUG-BTMV, PLAN §0)");
    }
}
assert_eq!(ECtrType::Buckets.target_border_count(3), 3);
assert_eq!(ECtrType::BinarizedTargetMeanValue.target_border_count(3), 1);
assert_eq!(final_ctr_target_border_count(3), 2);
```
Message must state that all three values being distinct is what makes B01 Test 2
a discriminator.

### TDD sequence

1. **Red** — run the pre-flight diff check, then write Tests 1–3 **before** the
   extraction. They do not compile (`final_ctr_target_border_count` does not
   exist). **Record the compile error as the Red** — the function's absence is the
   missing behavior.
2. **Green** — add the function, the helper counter-comment, and the `bake.rs`
   call-site replacement. Run `cargo test -p cb-train --lib ctr::mod_test` →
   Tests 1–3 green, and the pre-existing `mod_test` tests green.
3. **Refactor** — confirm the `saturating_sub(1).max(1)` expression appears
   **exactly twice** in `crates/cb-train/src/ctr/`: once in
   `final_ctr_target_border_count` (`mod.rs`) and once in `online_mean_prefix`
   (`online.rs:321`) — and that the duplication is **deliberate and documented**
   (the helper's "Deliberately NOT shared" block). Command:
   `grep -rn "saturating_sub(1).max(1)" crates/cb-train/src/ctr/`.
   **Do not deduplicate them** (D6).
4. **Verify** + **MUTATION CHECK (§3.1)** — two named mutations, run **in
   sequence**, each manually reverted before the next:

   **M1 — proves B01/B02 catch the original defect.**
   Change the helper body to `target_classes_count.max(1)` (i.e. restore the bug).
   **Expected outcomes — record ALL of them:**
   1. **B04 Test 1 FAILS** (`2 != 1`).
   2. **B01 Test 1 FAILS** — `mean == [(2.0,4),(0.0,4),(1.5,4)]`.
   3. **B01 Test 2 FAILS** — `[(1.0,3),(1.333…,3)]`.
   4. **B02 Test 1 FAILS** — Sum ratio exactly `0.5`.
   5. **`ctr_btmv_simple_oracle_test` FAILS** — back to
      `predictions are constant` (§1.1).
   6. **`borders_bake_bytes_are_unchanged`, the 11 CTR oracles, the 3 one-hot
      targets, `ctr_counter_simple_oracle_test` and the three BUG-CTRB gates all
      stay GREEN** — *required results (SPEC-BTMV-04, §0's isolation proof).*
   Revert manually; confirm green.

   **M2 — proves the R1 discriminator is load-bearing (§0, D2).**
   Change the helper body to
   `ECtrType::BinarizedTargetMeanValue.target_border_count(target_classes_count)`
   — i.e. implement SPEC.md §3's retracted advice.
   **Expected outcomes — EXACTLY these, all five recorded:**
   1. **B04 Test 1 FAILS** at `3 → 1` (expected `2`).
   2. **B04 Test 3 FAILS** at `final_ctr_target_border_count(3) == 2`.
   3. **B01 Test 2 FAILS** — `[(3.0,3),(4.0,3)]` instead of `[(1.5,3),(2.0,3)]`.
   4. **B01 Test 1 stays GREEN** — at `classes = 2` the helper also returns 1.
   5. **B02, `ctr_btmv_simple_oracle_test`, the 11 oracles, the 3 one-hot targets
      and the BUG-CTRB gates all stay GREEN** — at binclf the two rules are
      indistinguishable.

   > Outcomes 4 and 5 are **REQUIRED results, not incidental**. They are the
   > empirical confirmation of §0's central claim: **no binclf runtime gate can
   > detect a helper-based "fix"**, which is exactly why B01 Test 2 and B04 Tests
   > 1/3 have to exist. **A regression in 4 or 5 is a STOP-AND-REPORT
   > condition** — it would mean §0's reasoning is wrong.
   >
   > **Do NOT** strengthen a test, alter production code, or hunt for a further
   > failure in order to make an oracle regress under M2.

   Revert manually; confirm green.

   Finally re-run the full mandatory regression scope (§3.2) to confirm the
   extraction was behavior-preserving, and re-run the pre-flight diff so
   `bake.rs` carries only the B03 + B04 hunks before B05 takes the file.

### Completion criteria

- [ ] Pre-flight: `git diff crates/cb-train/src/ctr/bake.rs` showed **only B03's hunks**.
- [ ] `final_ctr_target_border_count` exists in `crates/cb-train/src/ctr/mod.rs`,
      is **private** (no `pub`, no `pub(crate)`, no `lib.rs` re-export), and carries
      the two-rules table, the `.max(1)` rationale and the "Deliberately NOT
      shared" block.
- [ ] `ECtrType::target_border_count`'s **body is byte-unchanged**; only its doc
      block grew, and it now names the ONLINE path and the zero-production-caller
      fact (E5).
- [ ] `bake.rs` calls `super::final_ctr_target_border_count(classes)`; the 4th
      argument is still `classes`.
- [ ] B04 Tests 1–3 green under `cargo test -p cb-train --lib ctr::mod_test`.
- [ ] `grep -rn "saturating_sub(1).max(1)" crates/cb-train/src/ctr/` shows exactly
      **two** hits, both documented as deliberate.
- [ ] **M1 recorded — all six outcomes**, including outcome 6 staying green.
- [ ] **M2 recorded — all five outcomes**, including outcomes 4–5 staying green.
- [ ] **No test was strengthened and no production code altered in order to make an
      oracle regress under M2.**
- [ ] Both mutations reverted **manually** (no `git checkout --`/`stash`/`clean`).
- [ ] Full regression scope green; zero diff on the 11 oracles.
- [ ] Post-flight: `git diff crates/cb-train/src/ctr/bake.rs` shows only B03 + B04
      hunks before B05 starts.
- [ ] `cargo clippy --workspace --all-targets` clean of new warnings.

### Risks and guardrails

- **R:** the helper is made `pub`/`pub(crate)` "for the tests". **M:** D11 and
  E14 — `mod_test` is a child of `ctr` and can already see it. A visibility
  widening is a completion-criterion failure.
- **R (the M2 trap):** an executor treats M2's green oracles as "the mutation
  check failed" and escalates — strengthening tests or changing production code
  until something regresses, i.e. **introducing a real defect chasing a phantom
  one**. **M:** the five outcomes are enumerated; 4–5 are stated as required
  results; §0 explains why no binclf gate can see it.
- **R:** an executor deduplicates the two `saturating_sub(1).max(1)` sites. **M:**
  D6, step 3's grep, and the helper's "Deliberately NOT shared" block.
- **R:** `const fn` + `usize::max` toolchain friction. **M:** the inline note —
  drop `const`, never rewrite the expression.

---

# WAVE W3 — GUARD

## B05 — GUARD: the non-mean baked payloads are byte-identical

- **Specs:** SPEC-BTMV-04
- **Order:** 6 · **Depends on:** B04 · **Status:** pending
- **Files:** Modify `crates/cb-train/src/ctr/final_ctr_test.rs` (**ADDITIVE ONLY**);
  **temporarily** mutates `crates/cb-train/src/ctr/mod.rs` during step 4
- **Touches production code:** NO permanently; YES temporarily (mutation only)
- **Exclusive resources:** `crates/cb-train/src/ctr/final_ctr_test.rs` (from B01);
  `crates/cb-train/src/ctr/bake.rs` / `ctr/mod.rs` (from B04)

### Pre-flight (mandatory)

`git diff crates/cb-train/src/ctr/bake.rs` and
`git diff crates/cb-train/src/ctr/mod.rs` must show **only B03's and B04's
hunks** — no residual mutation from B04's M1/M2. **A stray hunk is a
STOP-AND-REPORT condition.**

### Objective

SPEC-BTMV-04 says the Borders/Buckets/Counter baked payloads must be
byte-identical before and after. E8/E9 prove that **by code path**. B05 turns the
proof into a runtime gate and — crucially — proves the gate's **direction**: the
mutation that breaks BTMV must leave these **green**.

E11's `borders_bake_bytes_are_unchanged` already froze the **Borders** bake. B05
extends the freeze to **Buckets** and **Counter**, which had no byte-level pin.

### Why this is a GUARD, not a Red

It passes before and after B03 — the divisor never reached these payloads. Its
falsifiability comes from the §3.1 mutation check plus the *inverted* expectation
below. State that in a comment above the new tests.

### Test construction (exact)

Append to `crates/cb-train/src/ctr/final_ctr_test.rs`, after B01's block.

**Test 1 — `buckets_and_counter_bake_bytes_are_unchanged`
(the frozen extension of E11's Borders pin).**

Using `e11_fixture()` (E13):
- Buckets: `int_counts == vec![vec![0,4], vec![4,0], vec![1,3]]` (identical to the
  frozen Borders payload, E13/E15), `counter_denominator == 0`,
  `mean.is_empty()`, `ctr_type == 1`, and the same frozen `hashes` and
  `shift`/`scale` bit patterns as `borders_bake_bytes_are_unchanged`
  (`:199-209`: `hashes` at `:199`, `shift.to_bits() == 9_223_372_036_854_775_808`
  at `:208`, `scale.to_bits() == 4_624_633_867_356_078_080` at `:209`).
- Counter: `int_counts == vec![vec![4], vec![4], vec![4]]`,
  `counter_denominator == 4`, `mean.is_empty()`, `ctr_type == 4`.

Message on each: *"SPEC-BTMV-04: the whole-set target border count reaches only
`OnlineCtrAccumulator::binarized_mean` (online.rs:212-214), which only
`build_final_ctr`'s BinarizedTargetMeanValue arm reads (final_ctr.rs). No
non-mean payload can move when the divisor changes. If this fails, that isolation
has been broken — STOP AND REPORT."*

**Test 2 — `the_divisor_is_unreachable_from_every_non_mean_payload`
(the explicit isolation statement).**

Bake the same corpus at **all four CPU-legal types** and assert
`mean.is_empty()` for `Borders`, `Buckets` and `Counter`, and `!mean.is_empty()`
for `BinarizedTargetMeanValue`; and assert `int_counts.is_empty()` exactly for
BTMV. Message naming `bake.rs:226-260` (the per-type reshape arms, E9).

### TDD sequence

1. **Write and run** — `cargo test -p cb-train --lib ctr::final_ctr_test`.
   **Expected: BOTH PASS on first write**, alongside B01's tests and every
   pre-existing test in the file. **Record verbatim.**
2. **Green / Refactor** — none.
3. **MUTATION CHECK (§3.1) — one named mutation, with an INVERTED expectation.**

   **M-B05 — proves the isolation direction.**
   In `crates/cb-train/src/ctr/mod.rs`, change
   `final_ctr_target_border_count`'s body to `target_classes_count.max(1)`
   (B04's M1 mutation, reused here for a different purpose).
   **Expected outcomes — EXACTLY these, all four recorded:**
   1. **B05 Test 1 stays GREEN.**
   2. **B05 Test 2 stays GREEN.**
   3. **`borders_bake_bytes_are_unchanged` stays GREEN.**
   4. **B01 Test 1 FAILS** (`mean` halves) — the control, proving the mutation is
      actually live and the build is not stale.

   > **This is a mutation check whose PASSING outcome is "the guard did NOT
   > fire".** That is the point: outcomes 1–3 staying green is the *empirical*
   > proof of SPEC-BTMV-04's isolation claim (E8/E9), and outcome 4 is the
   > control that stops a stale build from faking it. **If outcome 1, 2 or 3
   > regresses, the isolation is broken and SPEC-BTMV-04 is false — STOP AND
   > REPORT.** **If outcome 4 does NOT fail, the mutation did not take effect —
   > re-check the build before recording anything.**
   >
   > Under §3.1 this task's falsifiability requirement is satisfied by outcome 4
   > (the control), not by outcomes 1–3.

   Revert **manually**; re-run; confirm green.
4. **Verify** —
   - the 11 CTR oracles + 3 one-hot targets + `ctr_counter_simple_oracle_test` +
     `ctr_nonmean_byte_identity_test` + the three BUG-CTRB gates — green
   - `git diff crates/cb-train/src/ctr/mod.rs` shows only B04's hunks (no residual
     mutation)
   - `git diff crates/cb-train/src/ctr/final_ctr_test.rs` is purely additive
   - `cargo clippy --workspace --all-targets`

### Completion criteria

- [ ] Pre-flight diffs showed only B03 + B04 hunks.
- [ ] Tests 1 and 2 pass on first write; recorded verbatim.
- [ ] The Buckets and Counter frozen payloads are pinned at byte level, with the
      same `hashes`/`shift`/`scale` bit patterns as E11's Borders pin.
- [ ] **M-B05 recorded — all four outcomes**, including outcomes 1–3 staying green
      and outcome 4 (the control) failing.
- [ ] Mutation reverted **manually**; `git diff crates/cb-train/src/ctr/mod.rs`
      shows only B04's hunks.
- [ ] The diff on `final_ctr_test.rs` is purely additive; no existing test edited.
- [ ] Regression scope green.

### Risks and guardrails

- **R:** an executor reads "the mutation must make the test fail" from §3.1 and
  "fixes" B05 until something fails. **M:** the inverted expectation is stated
  three times (objective, outcome list, the block quote), and outcome 4 is named
  as the falsifiability control.
- **R:** a stale build makes M-B05 look like a pass. **M:** outcome 4 is a
  mandatory control with its own STOP condition.
- **R:** the frozen Buckets/Counter literals are copied wrong. **M:** they are
  derivable — Buckets' `int_counts` must equal Borders' (asserted by the existing
  `bake_emits_the_requested_type_and_denominator`, Buckets block `:142-151`), and
  Counter's `[[4],[4],[4]]` / `counter_denominator == 4` are already asserted
  there (Counter block `:155-167`); B05 re-pins them with the
  hashes/shift/scale triple that only `borders_bake_bytes_are_unchanged` carried.

---

# WAVE W4 — CLOSE-OUT

## B07 — CLOSE-OUT: decisions, blast radius, evidence

- **Specs:** SPEC-BTMV-03, SPEC-BTMV-04 (recording); all four (traceability)
- **Order:** 7 · **Depends on:** B05 · **Status:** pending
- **Files:** Modify `.planning/plans/btmv-target-border-divisor/SPEC.md`
  (status + retraction + resolved OQs); create
  `.planning/plans/btmv-target-border-divisor/COMPLETION.md`
- **Touches production code:** NO — **B07 writes no code at all**

### What B07 must record

1. **The SPEC.md §3 retraction.** SPEC.md's *"The fix should route through
   [`target_border_count`] rather than hard-coding `classes - 1`"* is **wrong**.
   Amend §3's Note in place with the §0 evidence (`online_ctr.cpp:914` vs `:738`/
   `:762`/`:777`, `ctr_helper.h:34-42`) and mark the original text as retracted —
   **do not silently delete it**; a future reader must see that the question was
   asked and answered.
2. **SPEC.md §7 R1 — RESOLVED.** `classes - 1` (floored). Not ambiguous. Also
   record that it preserves the non-mean payloads regardless (§0), so the
   fallback criterion is satisfied too.
3. **SPEC.md §7 OQ1 — ANSWERED: no.** The 4th argument (`classes`) is correct
   (D4, E3). Cite `online_ctr.cpp:909-912, 930-934`.
4. **SPEC.md §7 OQ2 — ANSWERED: no.** `accumulate_online` has exactly **one**
   production caller (`bake.rs:196`); every other call site is a test passing a
   deliberate explicit divisor (E7). Reproduce the full enumeration:
   - production: `crates/cb-train/src/ctr/bake.rs:196`
   - tests: `crates/cb-train/src/ctr/online_test.rs:20,33,46,60,74,80,85`;
     `crates/cb-train/src/ctr/final_ctr_test.rs:15`;
     `crates/cb-model/tests/ctr_data_roundtrip_test.rs:100,134,160`
5. **SPEC-BTMV-03 — CLOSED, not affected.** `online_mean_prefix` is
   upstream-exact at **every** class count (E2/E10), gated by B06. The
   scope-widening branch did not open.
6. **A new latent finding, out of scope, recorded for a future plan.**
   `ECtrType::target_border_count` has **zero production callers** (E5): the
   online CTR-data allocation path (`online_ctr.cpp:738/741`) is not yet
   implemented in this repository, so the helper is currently unused production
   code. **Do not "fix" this by wiring it into the bake.**
7. **The §1.1 correction to the reported symptom** — E13's Red text is
   `predictions are constant — the gate would be vacuous`, with
   `max |diff| = 1.2830170735076996e-1` visible only under `-- --nocapture`.
   Carry it into SPEC.md §1 so the SPEC and the plan agree.
8. **The final diff gate.** `git diff --stat` for the whole change must show
   **exactly three production/test files** plus the four test files:
   - `crates/cb-train/src/ctr/bake.rs` (B03, B04)
   - `crates/cb-train/src/ctr/mod.rs` (B04)
   - `crates/cb-train/src/ctr/mod_test.rs` (B04)
   - `crates/cb-train/src/ctr/final_ctr_test.rs` (B01, B05)
   - `crates/cb-train/src/ctr/online_test.rs` (B06)
   - `crates/cb-train/tests/ctr_btmv_bake_upstream_table_test.rs` (B02, new)
   - the plan documents
   **Zero diff on:** all 11 CTR oracles, the 3 one-hot targets,
   `ctr_counter_simple_oracle_test`, the three BUG-CTRB gates,
   `ctr_nonmean_byte_identity_test`, `crates/cb-train/src/ctr/online.rs`,
   `crates/cb-train/src/ctr/final_ctr.rs`, `crates/cb-train/src/ctr/ctr_feature.rs`,
   `crates/cb-train/src/boosting.rs`, everything in `crates/cb-model/src/`, and
   **all three protected untracked paths** (§1).
9. **Final full regression transcript** (§3.2, whole block) attached to
   `COMPLETION.md`, plus `cargo clippy --workspace --all-targets`.

### Completion criteria

- [ ] SPEC.md §3's Note is retracted in place with evidence, not deleted.
- [ ] R1, OQ1, OQ2 recorded as resolved, each with its upstream citation.
- [ ] SPEC-BTMV-03 recorded as CLOSED / not affected.
- [ ] The `target_border_count`-has-no-production-caller finding recorded as
      out of scope, with the explicit "do not wire it into the bake" warning.
- [ ] §1.1's symptom correction carried into SPEC.md.
- [ ] The final diff gate verified file-by-file; the three protected untracked
      paths intact (`git status --short`).
- [ ] Full regression transcript + clippy attached.
- [ ] No production code written in this task.

---

## 6. Specification coverage

| Spec | Contract | Red / gate | Green | Guard / pin |
|---|---|---|---|---|
| **SPEC-BTMV-01** — the bake divides by the CTR type's target border count · **REFINED by D1/D2: the divisor is `target_classes_count - 1`, type-INDEPENDENT** | per-bucket `(Sum, Count)` = hand-computed class-1 counts | **B01** Test 1 (unit, table level); **B01** Test 2 (the R1 three-way discriminator) | **B03** | **B04** (named helper + Tests 1–3 + M1/M2) |
| **SPEC-BTMV-02** — the baked table matches upstream bucket-for-bucket | every `(hash, Sum, Count)` equals upstream's committed stride-3 entry; anti-vacuity `Sum != Count && Sum != 0` | **B02** Test 1 (by hash, sentinel-skipped) | **B03** | **B02** Test 2 (upstream Sums are integers ⇒ divisor 1) |
| **SPEC-BTMV-03** — the prefix producer is unaffected | `online_mean_prefix` divides by `classes - 1` at every class count | **B06** (verification, green on write) | n/a — nothing to change | **B06** Tests 1–3 + mutation **M-B06** |
| **SPEC-BTMV-04** — the non-mean bake arms are unchanged | Borders/Buckets/Counter payloads byte-identical | n/a — guard | n/a | **B05** Tests 1–2 + inverted mutation **M-B05**; E11's `borders_bake_bytes_are_unchanged` re-run in **B03** |

Every task maps to at least one spec; every spec maps to at least one task.
SPEC.md §5's acceptance scenarios map as: **A-BTMV-1** → B03 step 2 (E13 gate);
**A-BTMV-2** → B02; **A-BTMV-3** → B03/B04/B05 verify blocks; **A-BTMV-4** → B03
verify block (Counter gate + the three BUG-CTRB gates).

---

## 7. Open questions and residual risk

- **OQ-A `[INFERRED]`** — that fixing the divisor alone turns E13 green. The chain
  is strong (B02 gates the table to bit-identity with upstream; B06 gates the
  training-side prefix; BUG-CTRB gated the border space) but unmeasured until B03.
  **B03 step 2 carries an explicit STOP-AND-REPORT branch.**
- **OQ-B `[UNVERIFIED]`** — whether upstream's `hash_map` leading `u64::MAX` entry
  is *always* an empty-slot sentinel, or only in these two fixtures. Both
  committed CTR fixtures show it in position 0 with a stale payload (E12), and
  dropping it makes the Counts sum to `n_rows` in both. B02's `len() == 5` and
  `sum(count) == 30` guards fail loudly if the assumption breaks on a future
  fixture.
- **OQ-C `[UNVERIFIED]`** — whether the whole-set bake should ever be reached with
  `classes != 2`. Production hard-codes `2` (E6). B01 Test 2 and B04 Test 1
  characterize `classes = 3` deliberately, as the only available discriminator,
  and carry STOP CONDITIONS.
- **Residual `[OUT OF SCOPE]`** — `ECtrType::target_border_count` has no
  production caller (E5). Recorded in B07; not fixed here.
