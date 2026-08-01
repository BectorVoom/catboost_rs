---
title: TDD implementation plan — CTR-type engine honesty + cat_features/CTR facade routing
spec: ./SPEC.md
research: ./research.md
part2_source: ../catboost-builder-cat-features-routing/{SPEC.md,PLAN.md,PLAN-CHECK.md}
status: ready-for-implementation
plan_version: 1
created_at: 2026-08-01
task_count: 48
spec_ids_covered: SPEC-CTRT-01..19 (all 19); SPEC-CATF-Δ1..Δ8 (all 8); SPEC-CATF-01..17 (all 17, via Part 2)
acceptance_covered: A1..A13 (all 13)
---

# TDD Implementation Plan

Companion to `./SPEC.md` (authoritative, not re-derived here) and `./research.md`.
Part 2 **amends and executes** `../catboost-builder-cat-features-routing/PLAN.md`
(18 tasks, verdict `ISSUES_FOUND`); its surviving task bodies are cited by anchor
rather than duplicated.

Every `[VERIFIED]` anchor below was re-checked against the working tree on
2026-08-01 at `HEAD = 41e7e9c86ff8f91b55cff694832637aeddecb34d` **plus the
uncommitted one-hot device wave**, via `codegraph_explore` (MCP), direct reads, or
an executed read-only shell command. Anchors quoted from `SPEC.md` / `research.md`
were re-verified, not trusted.

---

## 0. Working-tree state this plan lands on (verified, load-bearing)

`git status --short` shows a **complete, uncommitted one-hot device wave**
touching 57 tracked files including `crates/cb-train/src/{boosting,tree}.rs`,
`crates/cb-model/src/{apply,model,cbm,json,shap,fstr,partial_dependence}.rs`,
`crates/cb-backend/*`, `crates/catboost-rs/src/{model,error}.rs`,
`crates/catboost-rs-py/src/errors.rs`, and
`crates/cb-oracle/fixtures/float_only_byte_identity/{baseline.cbm,README.md}`
`[VERIFIED: LOCAL git status --short]`.

**Land on top of it. Never revert it. Never `git checkout --` a file it touched.**

Facts this wave already established, which change the Part-2 plan:

| Fact | Evidence | Consequence |
|---|---|---|
| `ModelSplit` has THREE variants `{Float, OneHot, Ctr}` | `[CODEGRAPH ModelSplit → crates/cb-model/src/model.rs:101]`; `OneHotModelSplit` at `:80` | SPEC-CATF-Δ1 confirmed: CRITICAL-4 is OBSOLETE |
| `passes_one_hot_split` is live in the apply dispatch | `[CODEGRAPH → crates/cb-model/src/apply.rs:205-224]` | one-hot models genuinely score |
| `ObliviousTree` carries `one_hot_splits` + `level_kinds` | `[LOCAL crates/cb-train/src/boosting.rs:808-826]` | one-hot trains end to end |
| `shap_values` now returns `Result<_, ShapUnsupported>` and REJECTS one-hot/CTR splits | `[LOCAL crates/cb-model/src/shap.rs:552-608; crates/catboost-rs/src/model.rs shap_values uses `?`]` | SPEC-CATF-Δ7 is **already half-done**; only `partial_dependence` + `feature_importance_with_data` remain |
| `crates/cb-train/src/boosting.rs:816-817` doc comment "which is all of them until T19 populates it" is now FALSE | `[LOCAL sed -n '805,825p']` | fix opportunistically (task E10) |

**`catboost-master/` is NOT a parity reference.** CLAUDE.md's claim of a full
vendored C++ tree is STALE; the tree is a 3-file stub of a different revision. All
upstream citations below come from the live `v1.2.10` tag via `research.md` §A/§K.

---

## 1. Locked decisions this plan encodes (non-negotiable)

1. **Engine first, then facade.** Part 1 (W0–W5) fully precedes Part 2 (W6–W10).
2. **Lift the `.cbm` v1 mean-CTR restriction** so BTMV models round-trip
   (SPEC-CTRT-14/15). Real serialization sub-scope with its own byte-identity
   oracle (SPEC-CTRT-16).
3. **Keep the scalar `simple_ctr: ECtrType` field shape.** Honor the CTR *type*
   and the FULL prior *list*; do NOT convert the field to a list.
   `grep -rn "simple_ctr:" crates/ --include=*.rs` → **65 occurrences across 62
   files** `[VERIFIED: LOCAL]`. **Adding no field and changing no type is free.**
   The multi-description default gap is a documented parity gap (SPEC-CTRT-19,
   task F00).
4. **Reuse the committed `crates/cb-oracle/fixtures/one_hot_train/` fixture** for
   the Part-2 public-API oracle. **Run NO corpus-wide fixture generation.**

---

## 2. Deviation from `research.md` §H's wave ordering (stated, with evidence)

§H is followed with **two** corrections, both forced by CodeGraph/read evidence:

**Correction 1 — the frozen non-mean `.cbm` byte baseline moves from W4 to W0.**
§H puts all `.cbm` work in W4. But SPEC-CTRT-16's gate is only meaningful against
bytes frozen **before** the codec change; capturing it later degenerates into a
self-comparison. Task **E00** therefore captures it first, over a
**hand-constructed** `cb_model::Model` (not a trained one) so it is immune to every
W1–W3 trainer change. Precedent and shape: `crates/cb-oracle/fixtures/float_only_byte_identity/`
+ `crates/cb-model/tests/float_only_byte_identity_test.rs`
(`capture_float_only_baseline` is `#[ignore]`d; README records `CAPTURED_AT_SHA`
and a forbidden-to-regenerate note) `[VERIFIED: LOCAL, read in full]`.

**Correction 2 — the per-type BAKE (SPEC-CTRT-13) moves from W4 to W2.**
§H schedules `counter_simple` / `btmv_simple` fixtures in W2 but `bake.rs` in W4.
That is **unsatisfiable**: an end-to-end `≤1e-5` prediction gate needs the baked
inference table, and

- `bake_ctr_table` hard-codes `build_final_ctr(&acc, ECtrType::Borders)` and
  `ctr_type: ECtrType::Borders.as_i8()`, and pins `counter_denominator: 0`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/bake.rs:192,232,236]`;
- `BakedCtrTable` has **no** mean fields at all
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/bake.rs:61-86]`;
- `CtrData::from_baked` hard-codes `mean: Vec::new()`
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs, from_baked body]`.

So W2 gains task **E11** (per-type bake + mean threading). W4 keeps ONLY the
`.cbm` mean codec. Everything else in §H is followed exactly.

**A third finding that reshapes W5 (not an ordering change, a scope discovery):**
`EvalSet` carries **only** `feature_values` and `target` — there is **no
categorical column field**
`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:1951-1959]`. `counter_calc_method
= Full` counts learn **+ every eval set** into the Counter bucket totals
(`online_ctr.cpp:716-729`), so SPEC-CTRT-17 is **structurally unimplementable**
until `EvalSet` can carry cat columns. Blast radius measured: `EvalSet` appears in
**4 files** with **6 struct-literal sites**
(`crates/cb-train/tests/eval_metrics_oracle_test.rs:120,124,228,232`;
`crates/cb-train/tests/overfit_oracle_test.rs:213`;
`crates/cb-train/src/boosting.rs:2093`), and the struct carries **no derives**
`[VERIFIED: LOCAL grep]`. Tractable — task **E21** does it. `train_inner` already
accepts `eval_sets: &[EvalSet]`
`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:2555-2567]`, so only a public
`train_cat_with_eval_sets` wrapper is missing.

---

## 3. Shared conventions every task must honor

- **Source/test separation is MANDATORY** (CLAUDE.md). No `mod tests` / no
  `#[cfg(test)]` block in a production file. Unit tests go in a sibling
  `<name>_test.rs` mounted via `#[cfg(test)] #[path = "<name>_test.rs"] mod …`;
  integration tests in `crates/<crate>/tests/`.
  Verified mounts to reuse: `crates/cb-train/src/boosting.rs:5493-5495`
  (`#[path = "boosting_test.rs"] mod tests;` → filter `boosting::tests::…`),
  `crates/cb-train/src/ctr/mod.rs:42-50` (`online_test`, `calc_ctr_test`,
  `final_ctr_test` → filter `ctr::online_test::…`),
  `crates/cb-train/src/candidates.rs:53-55`.
- **No `unwrap()`/`expect()`/`panic!()`/raw indexing in production.**
  `[VERIFIED: LOCAL Cargo.toml:10-14 — unwrap_used/expect_used/panic/indexing_slicing = "deny"]`.
  Test code is exempted in-code via `#![cfg_attr(test, allow(...))]` at the crate
  root or a file-level `#![allow(...)]` in an integration test
  (precedent: `crates/cb-model/tests/float_only_byte_identity_test.rs:15`).
- **Typed errors only.** `cb_core::CbError::{Unsupported(String), Degenerate(String),
  LengthMismatch{..}, OutOfRange(String)}` all exist
  `[VERIFIED: LOCAL crates/cb-core/src/error.rs:28,47,84,92]`.
  `cb_model::ModelError::{Serialize, Deserialize}` for the codec.
- **Never invoke `crates/cb-oracle/generator/gen_fixtures.py`.** Its `__main__`
  block dispatches only on the eight `--*-only` flags; **every other argv falls
  through to `else: main()`**, regenerating the ENTIRE frozen corpus
  `[VERIFIED: LOCAL tail -30 of that file]`. Every new fixture in this plan uses
  the **fixture-local** `crates/cb-oracle/fixtures/<name>/gen_fixtures.py` pattern
  — **9 in-repo precedents**: `advanced_fstr`, `coreml_export`, `ctr_load`,
  `feature_selection`, `fstr_ctr`, `fstr_loss_change`, `model_sum`,
  `partial_dependence`, `staged_predict` `[VERIFIED: LOCAL find]`. That pattern is
  structurally incapable of dirtying the corpus.
- **Every fixture generator carries the mandatory anti-false-pass guard**:
  assert `model.json → features_info.ctrs` contains **≥1 descriptor of the
  INTENDED `ctr_type`**, and (for `buckets_simple`) that **both**
  `target_border_idx: 0` **and** `1` appear. Without it, "the model trained" is
  satisfiable by a config that produced **zero** CTR splits, and both sides agree
  trivially.
- **Every fixture generator ends with a corpus-cleanliness guard**:
  `git status --porcelain crates/cb-oracle/fixtures` must list ONLY paths under
  the scenario's own directory; otherwise `sys.exit(1)` printing the offenders
  (shape copied from `gen_one_hot_only`'s dirty-path snapshot,
  `[VERIFIED: LOCAL crates/cb-oracle/generator/gen_fixtures.py:1398-1424]`).
- **Isolating fixture config (pin ALL of these on the upstream side):**
  `boosting_type="Plain"`, `one_hot_max_size=1`, `permutation_count=1`,
  `fold_len_multiplier=2.0`, `bootstrap_type="No"`, `random_strength=0`,
  `random_seed=0`, `thread_count=1`, `boost_from_average=False`, `verbose=False`,
  `leaf_estimation_method="Gradient"`, `leaf_estimation_iterations=1`,
  `loss_function="Logloss"`, `max_ctr_complexity=1` (2 for
  `mixed_simple_vs_combo`), `combinations_ctr=[]` for the simple-only scenarios.
  Shape verified against `crates/cb-oracle/fixtures/plain_ctr/config.json`
  `[VERIFIED: LOCAL, read in full]`. Fixtures are **categorical-only** (no float
  columns) so upstream float-quantization nondeterminism is structurally excluded
  (precedent: `tensor_ctr_e2e`).
- **Fixtures are generated ONCE, committed, and NEVER regenerated in CI.**
  Each `config.json` carries a `"note"` with the `NEVER regenerate` sentence
  (precedent: `crates/cb-oracle/fixtures/one_hot_train/default_binary/config.json`).
- **Backend trap.** `--features rocm` alone does **not** switch backends (`cpu` is
  a default feature that wins the cfg chain). Use
  `--no-default-features --features rocm`. Any test driving the resident GPU
  scorer/splitter cannot run on `cubecl-cpu` (CUBE_COUNT unsupported) and must be
  gated `#[cfg(all(test, not(feature = "cpu")))]`.
- **Accepted failing-test baseline.**
  `.planning/plans/one-hot-categorical-training/baseline/` — 297 passed / **59
  failed, all in the `cb_backend` lib target** under the CubeCL cpu backend
  `[VERIFIED: LOCAL baseline/README.md]`. The gate is **"no target that passes
  here may fail later"**, not full green. `exact_quantile_weighted_matches_cpu` is
  **FLAKY** (passes ~2 of 5 identical runs) — not a regression. The transcript is
  self-documented as INCOMPLETE (`cargo test --workspace` ran without
  `--no-fail-fast`), and `cb-train`'s
  `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors` is a
  verified pre-existing failure absent from it.
- **Parity bar** `≤1e-5` vs `catboost==1.2.10`, which IS installed and usable:
  `.venv/bin/python -c "import catboost"` → `1.2.10`, Python 3.12.13
  `[VERIFIED: LOCAL executed]`.

### 3.1 Guard-test falsifiability protocol (MUTATION CHECK)

Several tasks are **regression guards**: a correctly-implemented predecessor makes
them pass on first write, so a classic Red is unavailable. For those tasks the Red
step is replaced by an explicit mutation check:

1. Write the test. Run it. **Record the result.**
2. Apply the task's **named single-line mutation** to production code.
3. Re-run; the test MUST fail with the named message. **Record the failure text.**
4. **Revert the mutation** (manual revert, NEVER `git checkout --` — the working
   tree carries the uncommitted one-hot wave) and re-run to confirm green.

A guard test that cannot be made to fail by its named mutation is not a guard and
the task is **not complete**.

### 3.2 Repository-verified commands

```bash
# cb-train unit tests (child-module filters)
cargo test -p cb-train --lib ctr::online_test
cargo test -p cb-train --lib ctr::calc_ctr_test
cargo test -p cb-train --lib ctr::final_ctr_test
cargo test -p cb-train --lib boosting::tests
cargo test -p cb-train --lib candidates::tests

# A single integration-test TARGET (name filter goes after `--`, NOT `::`)
cargo test -p cb-train --test <target_name> -- <filter>
cargo test -p cb-model --test <target_name> -- <filter>

# THE 11 EXISTING CTR ORACLES — the SPEC-CTRT-18 regression gate (all verified to exist).
# These ELEVEN targets are exactly the list SPEC-CTRT-18 names; the two documents agree.
# 9 in cb-train + 2 in cb-model. Every task that states "the 11 CTR oracles" as its
# regression scope MUST run this whole block, not a subset.
#
# DIFF GATE — PER FILE, in three categories. THREE of the eleven are legitimately
# edited by owning tasks; the other EIGHT are untouched by every task.
#
#   ZERO DIFF REQUIRED (8) — `git diff --stat` over these must print NOTHING:
#     plain_ctr_oracle_test, ordered_ctr_oracle_test, tensor_ctr_oracle_test,
#     tensor_ctr_e2e_oracle_test, s_order_ctr_bins_oracle_test,
#     multi_permutation_e2e_oracle_test, multi_permutation_fold_oracle_test,
#     fstr_ctr_oracle_test
#
#   MECHANICAL EDITS ONLY, ZERO ASSERTION CHANGES (3) — signature-driven
#   argument/field edits and ADDITIVE new test functions, nothing else:
#     ctr_split_scoring_test      — E09 (`target_border_idx` at :41,:68 and
#                                   `materialize_ctr_feature` args at :384,:394),
#                                   E11 (`bake_ctr_table` args at :542,:576,:645),
#                                   E16 (five dropped args at :99,:148,:191,:249,:305),
#                                   E22 (all five call sites again)
#     ctr_feature_materialize_test — E09, E22 (ADDITIVE test fns + widened args)
#     ctr_data_roundtrip_test      — E11 (ADDITIVE test fns 2/4 + the compile-forced
#                                   `build_final_ctr` arg at :101,:138,:143,:163)
#
#   A diff that touches an EXISTING assertion in ANY of the eleven — added,
#   removed, weakened or reworded — is a STOP-AND-REPORT condition. Weakening or
#   deleting any assertion in any of them is FORBIDDEN.
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

# THE ONE-HOT WAVE REGRESSION SCOPE (A12) — mandatory for every boosting.rs/tree.rs task
cargo test -p cb-train --test one_hot_oracle_test \
                       --test one_hot_draw_accounting_test \
                       --test device_one_hot_parity_test

# .cbm / model serde gates
cargo test -p cb-model --test cbm_oracle_test --test json_oracle_test \
                       --test float_only_byte_identity_test

# Lints
cargo clippy --workspace --all-targets

# Fixture generation — FIXTURE-LOCAL ONLY. NEVER `generator/gen_fixtures.py`.
.venv/bin/python crates/cb-oracle/fixtures/<scenario>/gen_fixtures.py

# Python extension + tests (Part 2)
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop \
    -m crates/catboost-rs-py/Cargo.toml --features cpu
.venv/bin/python -m pytest crates/catboost-rs-py/tests/ -q
```

---

## 4. Execution waves and dependency order

```text
PART 1 — ENGINE

W0  foundations, ZERO behavior change
      E00 ∥ E01 ∥ E03            (genuinely parallel — disjoint files)
        E00  (fixtures/ctr_nonmean_byte_identity + cb-model test)
        E01  (cb-train/src/ctr/mod.rs)
        E03  (cb-train/src/boosting_test.rs)
      E02  SERIALIZED AFTER BOTH E01 AND E03
        E02  (cb-train/src/boosting.rs validation block + boosting_test.rs test)
        * E02 is NOT parallel. It consumes `is_cpu_supported` (E01) and it ALSO
          edits `crates/cb-train/src/boosting_test.rs`, the same file E03 owns —
          the two are NOT disjoint. Edges `E01->E02` and `E03->E02` below.

W1  regression firewall, ZERO behavior change     (needs E01)
      E04 -> E05                (both own cb-train/src/ctr/online.rs: SERIAL)

W2  type routing becomes real, ONE prior          (needs E05; E02 for the guard)
      E06 -> E07 -> E08         (all own online.rs: SERIAL)
      E08 -> E09                (ctr_feature.rs)
      E09 -> E10                (boosting.rs + tree.rs)
      E10 -> E11                (bake.rs + cb-model/ctr_data.rs)
      E11 -> E12, E13           (fixture gates; E12 ∥ E13, disjoint fixture dirs)

W3  candidate expansion — HIGHEST RISK            (needs E11; E05 firewall green)
      E14 (fixture, no code)  ─┐
      E15 (multi-prior)       ─┴-> E16 (target_border_idx + buckets fixture)
      E16 -> E17 (mixed_simple_vs_combo gate)
      E14 is parallel with ALL of W0–W2 (fixture dir only)

W4  .cbm mean codec           (needs E11 for a BTMV model; E13 for the generator; E00 baseline)
      E18 (fixture) -> E19 (DECODE) -> E20 (ENCODE + round-trip + E00 gate)
        * E18 modifies fixtures/ctr_btmv_simple/gen_fixtures.py, CREATED BY E13 —
          edge `E13->E18` below.

W5  counter_calc_method                           (needs E11)
      E21 (EvalSet.cat_columns + train_cat_with_eval_sets)
        -> E22 (thread counter_calc_method)
        -> E23 (counter_full_eval fixture + eval-set gate | recorded deferral)

PART 2 — FACADE  (starts only after W5 is green)

W6  honesty + builder setters
      F00 (SPEC-CTRT-19 doc gap; unblocks F03/F04/F05/F16)
      lane A (STRICTLY SERIAL on crates/catboost-rs/src/builder.rs):
        F01 -> F02 -> F03 -> F04 -> F05 -> F06 -> F08 -> F09
        (F08 owns cb-model/src/model.rs, not builder.rs, but it is the PRODUCER of
         `Model::with_cat_feature_count`, which F09 CALLS — so it precedes F09.)
      lane B (parallel with lane A, disjoint files):
        F07 (crates/catboost-rs/src/lib.rs + new tests/ file)
        F19 (fixture reuse audit — read-only, no generation)

W7  model-side cat width + predict routing        (needs F09)
      F08 already landed at the end of W6 (cb-model/src/model.rs: stored cat width)
      F09 -> F10 -> F11 -> F12 -> F13   (F10–F13 all own crates/catboost-rs/src/model.rs: SERIAL)

W8  cv / grid_search defined behavior             (needs F11)
      F14 (crates/catboost-rs/src/{cv,grid_search}.rs + tests)

W9  Python bindings                               (needs F05, F07, F09, F11)
      F15 -> F16                (both own catboost-rs-py/src/params.rs: SERIAL)
      F17                       (estimator/regressor/classifier/ranker/search/cv.rs)
      F15,F16,F17 -> F18

W10 gates
      F18, F19, F12 -> F20 (public-API oracle ≤1e-5, A11)
      F20 -> F21 (float-only no-regression, A1/CATF-07)
      F21 -> F22 (bench preflight, A13)
      F22 -> F23 (one-hot wave + workspace baseline gate, A12)
```

**Edge list (acyclic).**

```text
E01->E02  E03->E02
E01->E04  E04->E05  E05->E06  E06->E07  E07->E08  E08->E09  E09->E10
E02->E10  E03->E10  E10->E11  E11->E12  E11->E13
E11->E15  E14->E15  E05->E15  E15->E16  E16->E17
E11->E18  E13->E18  E18->E19  E19->E20  E00->E20
E11->E21  E21->E22  E22->E23
E23->F00  E23->F01  E23->F07  E23->F19
F00->F03  F00->F05  F00->F16
F01->F02  F02->F03  F03->F04  F04->F05  F05->F06  F06->F08
F08->F09  F09->F10  F10->F11  F11->F12  F12->F13
F11->F14
F05->F15  F07->F15  F15->F16  F09->F17  F11->F17
F15->F18  F16->F18  F17->F18
F18->F20  F19->F20  F12->F20  F14->F20  F20->F21  F21->F22  F22->F23
```

Four edges are corrections applied after the pass-1 plan review, each load-bearing:

- **`E01->E02`** — E02 consumes `ECtrType::is_cpu_supported`, which E01 creates.
- **`E03->E02`** — E02 and E03 **both** edit `crates/cb-train/src/boosting_test.rs`;
  they are an edit-conflict serialization, not a logical dependency.
- **`E13->E18`** — E18 modifies
  `crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py`, a file **created by
  E13**.
- **`F06->F08->F09`** replaces the previous inverted `F06->F09->F08`: F09 calls
  `Model::with_cat_feature_count`, which **F08 produces**. The producer must land
  first; F08 has no dependency on F09 other than that call site, which belongs to
  F09.

Four further edges were added in the pass-2 revision so the edge list — declared
**authoritative for ordering** — matches the task bodies it governs. All four go
forward in wave order and preserve acyclicity:

- **`E23->F01`**, **`E23->F07`**, **`E23->F19`** — F01, F07 and F19 each state
  "Blocked by: E23" in their own task bodies (locked decision 1: Part 1 fully
  precedes Part 2), but only `E23->F00` was listed. Without them an executor
  reading the edge list alone could start F01/F07/F19 before Part 1 is green.
- **`F14->F20`** — F14 states "**Blocks:** F20" and F20 states "**Blocked by:**
  F12, F14, F18, F19"; the edge was missing, so F20 could run before F14's
  `cv`/`grid_search` rejection lands.

**Parallelization is claimed ONLY where file ownership AND CodeGraph show no
conflict.** Every SERIAL note below is an edit-conflict serialization on a single
file, stated per task.

---

# PART 1 — ENGINE

## WAVE W0 — Foundations (zero behavior change)

---

### E00 — Freeze the non-mean `.cbm` CTR byte baseline (BEFORE any codec change)

- **Specs:** SPEC-CTRT-16; acceptance **A9**
- **Blocked by:** none. **Blocks:** E20.
- **Parallelizable:** **YES** with E01/E02/E03 — owns only a new fixture directory
  and a new `crates/cb-model/tests/` file; no overlap with `cb-train`.

**Goal / observable completion condition.** A committed, frozen
`crates/cb-oracle/fixtures/ctr_nonmean_byte_identity/baseline.cbm` produced from a
**hand-constructed** `cb_model::Model` carrying non-mean CTR tables, plus a
non-`#[ignore]`d test that compares `save_cbm` output against it byte-for-byte and
is proven falsifiable by a mutation check.

**Why hand-constructed, not trained.** W1–W3 change the trainer's chosen candidate
set (SPEC-CTRT-11 explicitly changes tie-breaks). A *trained* baseline would drift
and the gate would silently become vacuous. A hand-constructed model isolates the
**serializer**, which is exactly what SPEC-CTRT-16 gates.

**Files**
- Create: `crates/cb-oracle/fixtures/ctr_nonmean_byte_identity/README.md`
- Create (generated once, then frozen):
  `crates/cb-oracle/fixtures/ctr_nonmean_byte_identity/baseline.cbm`
- Create: `crates/cb-model/tests/ctr_nonmean_byte_identity_test.rs`

**Exact verified files/symbols to touch (read-only dependencies)**
- `cb_model::save_cbm` — the encoder under gate; the mean-rejection sites it will
  later lose are `crates/cb-model/src/ctr_data.rs`
  `build_tctr_value_table` (`if table.ctr_type.is_mean() { return Err(ModelError::Serialize("mean/target-mean CTR unsupported on save (v1, MAJOR-2)")) }`)
  and `decode_one_ctr_value_table`
  (`ModelError::Deserialize("mean/target-mean CTR unsupported (v1, MAJOR-2)")`)
  `[VERIFIED: LOCAL, both read verbatim]`.
- `cb_model::CtrValueTable { ctr_type, target_classes_count, hashes, int_counts,
  mean: Vec<(f32,i64)>, counter_denominator }`
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:157-174]`.
- `cb_model::ctr_base_key(ctr_type, cat_features) -> String` = `"ctr:type=<i8>:proj=<f0>,<f1>,…"`
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs]`.
- `EMPTY_HASH_MARKER = 0xFFFF_FFFF_FFFF_FFFF` — a bucket hash must never equal it
  `[VERIFIED: LOCAL build_tctr_value_table body]`.
- Existing construction precedent for `CtrValueTable` literals:
  `crates/cb-model/tests/ctr_data_roundtrip_test.rs` `[VERIFIED: LOCAL exists]`.
- Pattern to copy verbatim in **shape**:
  `crates/cb-model/tests/float_only_byte_identity_test.rs` — deterministic LCG
  input generator (`:38-50`), `baseline_dir()` (`:27-32`), the `#[ignore]`d
  `capture_float_only_baseline` (`:147-151`), the comparison test
  `float_only_cbm_bytes_match_the_frozen_plan_base_baseline` (`:249`), and
  `frozen_baseline_records_its_plan_base_sha` (`:296`)
  `[VERIFIED: LOCAL, read in full]`.

**CodeGraph evidence for ordering.** `build_tctr_value_table` and
`decode_one_ctr_value_table` are the ONLY two mean-rejection sites; E19/E20 are the
only tasks that edit them. Capturing here is therefore provably pre-change for the
codec.

**Red — MUTATION CHECK (§3.1); a classic Red is unavailable for a fresh baseline**
- File: `crates/cb-model/tests/ctr_nonmean_byte_identity_test.rs`
- Capture fn: `capture_ctr_nonmean_baseline` — `#[test] #[ignore]`, run once with
  `-- --ignored`, writes `baseline.cbm` **and** the README (mirroring
  `float_only_byte_identity_test.rs:147-200`).
- Gate fn: `nonmean_ctr_cbm_bytes_match_the_frozen_baseline`
- Setup / input (fully deterministic, no RNG, no trainer):
  a `cb_model::Model` with two oblivious trees of depth 1, one `ModelSplit::Ctr`
  each, and a `CtrData` carrying **two** tables:
  1. `ctr_type = ECtrType::Borders`, `target_classes_count = 2`,
     `hashes = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222, 0x3333_3333_3333_3333]`,
     `int_counts = [[3,7],[11,2],[0,5]]`, `mean = vec![]`, `counter_denominator = 0`;
  2. `ctr_type = ECtrType::Counter`, `target_classes_count = 0`,
     the same three hashes, `int_counts = [[10],[13],[5]]`, `mean = vec![]`,
     `counter_denominator = 13`.
  Both keyed by `ctr_base_key(ctr_type, &[0])`. Neither hash equals
  `EMPTY_HASH_MARKER`.
- Expected output: `save_cbm(&model)?` bytes `==` the committed `baseline.cbm`
  bytes, asserted with `assert_eq!(produced, frozen)` and, on failure, a message
  printing both lengths and the first differing offset.
- **EXPECTED INITIAL FAILURE (before the capture run):**
  `No such file or directory (os error 2)` opening
  `crates/cb-oracle/fixtures/ctr_nonmean_byte_identity/baseline.cbm`.
- **Mandatory mutation (proves the gate is falsifiable):** in
  `crates/cb-model/src/ctr_data.rs::build_tctr_value_table`, temporarily change the
  emitted `CounterDenominator` to `denominator + 1`. Re-run.
  **Expected failure:** `assertion `left == right` failed` with a differing byte at
  the counter-denominator offset. **Revert manually** (do NOT `git checkout --`)
  and confirm green. **Record both outputs.**
- Run: `cargo test -p cb-model --test ctr_nonmean_byte_identity_test -- --ignored`
  then `cargo test -p cb-model --test ctr_nonmean_byte_identity_test`

**Green (minimal implementation intent).** No production change. The deliverables
are the fixture bytes plus a README recording, verbatim in this shape:
`CAPTURED_AT_SHA: 41e7e9c86ff8f91b55cff694832637aeddecb34d` **plus the explicit
sentence "captured on top of the UNCOMMITTED one-hot device wave present in the
working tree at capture time"**; the exact hand-constructed model definition; and a
`## Do not regenerate` section stating that a later regeneration turns SPEC-CTRT-16
into a self-comparison that proves nothing (wording modelled on
`float_only_byte_identity/README.md`'s "Still frozen from here on").

**Refactor constraints + required regression scope**
- Do NOT touch `crates/cb-oracle/fixtures/float_only_byte_identity/` (already
  re-baselined once by the one-hot wave; its README documents the single
  deliberate exception).
- Regression scope: `cargo test -p cb-model`.

**Validation**
```bash
cargo test -p cb-model --test ctr_nonmean_byte_identity_test -- --ignored
cargo test -p cb-model --test ctr_nonmean_byte_identity_test
cargo test -p cb-model
git status --porcelain crates/cb-oracle/fixtures   # ONLY ctr_nonmean_byte_identity/*
cargo clippy -p cb-model --all-targets
```

**Completion evidence.** The two committed fixture files; the recorded
pass → mutated-fail → reverted-pass triple; the `git status --porcelain` output
listing only the new directory.

---

### E01 — `ECtrType` capability helpers

- **Specs:** SPEC-CTRT-01, SPEC-CTRT-02
- **Blocked by:** none. **Blocks:** E02, E04.
- **Parallelizable:** **YES** with E00 and E03 — owns
  `crates/cb-train/src/ctr/mod.rs` and a new `crates/cb-train/src/ctr/mod_test.rs`.
  **NO** with E02, which consumes `is_cpu_supported` (edge `E01->E02`).

**Goal / observable completion condition.** `ECtrType::target_border_count(2)`
returns `2` for `Buckets` and `1` for `Borders` / `BinarizedTargetMeanValue` /
`Counter`; `is_cpu_supported()` is `false` for exactly
`FloatTargetMeanValue` and `FeatureFreq`; `is_online_prefix()` is `true` for
`Borders`/`Buckets`/`BinarizedTargetMeanValue` and `false` for `Counter`.

**Files**
- Modify: `crates/cb-train/src/ctr/mod.rs`
- Create: `crates/cb-train/src/ctr/mod_test.rs` (mounted `#[cfg(test)]
  #[path = "mod_test.rs"] mod mod_test;` alongside the existing `online_test` /
  `calc_ctr_test` / `final_ctr_test` mounts at `mod.rs:42-50`)
- Modify: `crates/cb-train/src/lib.rs` — nothing to add; `ECtrType` is already
  re-exported `[VERIFIED: LOCAL crates/cb-train/src/lib.rs:49]`.

**Exact verified files/symbols to touch**
- `pub enum ECtrType { Borders=0, Buckets=1, BinarizedTargetMeanValue=2,
  FloatTargetMeanValue=3, Counter=4, FeatureFreq=5 }` with
  `#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(i8)]`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/mod.rs:65-80]`.
- Existing `impl ECtrType`: `as_i8` (`:86`), `from_i8 -> Option<Self>` (`:93`),
  `default_priors -> Vec<Prior>` (`:110`) `[VERIFIED: LOCAL]`. The new helpers go
  in the SAME `impl` block, after `from_i8`.
- Upstream anchors to cite in doc comments:
  `ctr_helper.h:34-42` (`GetTargetBorderCount`: BTMV/Counter → 1, Buckets →
  `targetClassesCount`, else `targetClassesCount - 1`); `restrictions.h:18-48`
  (`IsSupportedCtrType(CPU, …)` true only for
  `{Borders, Buckets, BinarizedTargetMeanValue, Counter}`);
  `ctr_type.cpp:43-56` (`IsPermutationDependentCtrType(Counter) == false`).

**CodeGraph evidence.** `ECtrType` (`crates/cb-train/src/ctr/mod.rs:67`) has **6
callers** in `bake.rs`, `ctr_feature.rs`, `final_ctr.rs`, `lib.rs` (+1), with
covering tests `crates/cb-train/src/ctr/final_ctr_test.rs`
`[VERIFIED: CODEGRAPH blast radius]`. Adding **inherent methods** is purely
additive — zero of those callers changes.

**Red**
- File: `crates/cb-train/src/ctr/mod_test.rs`
- Test fns (three, one per helper — separate primary causes):
  1. `target_border_count_is_two_for_buckets_and_one_for_the_rest`
     Input: each of the four CPU-legal types with `target_classes_count = 2`.
     Expected: `assert_eq!(ECtrType::Buckets.target_border_count(2), 2);` and
     `assert_eq!(t.target_border_count(2), 1)` for `Borders`,
     `BinarizedTargetMeanValue`, `Counter`.
  2. `is_cpu_supported_rejects_exactly_float_target_mean_and_feature_freq`
     Expected: `false` for `FloatTargetMeanValue` and `FeatureFreq`; `true` for the
     other four — asserted over an explicit all-six array so a new variant cannot
     silently escape.
  3. `is_online_prefix_is_false_only_for_counter`
     Expected: `true` for `Borders`/`Buckets`/`BinarizedTargetMeanValue`, `false`
     for `Counter`.
- **EXPECTED INITIAL FAILURE:**
  `error[E0599]: no method named `target_border_count` found for enum `ECtrType``
  (and the two analogues). If the `#[path]` mount is added in the same edit, the
  *first* error may instead be `error[E0583]: file not found for module` — create
  the file first, then add the mount.
- Run: `cargo test -p cb-train --lib ctr::mod_test`

**Green (minimal implementation intent).** Three `#[must_use] pub const fn` (or
`pub fn` where `match` on `self` forbids `const`) inherent methods on `ECtrType`,
each a total `match` over all six variants (no `_ =>` arm, so a future variant is a
compile error, not a silent default). No other file changes.

**Refactor constraints + required regression scope**
- Constraint: do NOT alter `as_i8` / `from_i8` / `default_priors`; do NOT reorder
  variants (the `#[repr(i8)]` discriminants are wire values shared with
  `cb-model`).
- Regression scope: `cargo test -p cb-train --lib ctr::` and `cargo test -p cb-train`.

**Validation**
```bash
cargo test -p cb-train --lib ctr::mod_test
cargo test -p cb-train --lib ctr::
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** Three named tests green; `cargo test -p cb-train` green
including all nine cb-train CTR oracle targets.

---

### E02 — CPU-illegal CTR types are typed-rejected before any accumulation

- **Specs:** SPEC-CTRT-03; acceptance **A10**
- **Blocked by:** E01 (`is_cpu_supported`) **and** E03 (both edit
  `crates/cb-train/src/boosting_test.rs`). **Blocks:** E10.
- **Parallelizable:** **NO.** E02 is **serialized after BOTH E01 and E03** — edges
  `E01->E02` and `E03->E02` in §4. It consumes `is_cpu_supported` (E01) and it
  edits `crates/cb-train/src/boosting_test.rs`, which E03 also owns; the earlier
  claim that E02/E03 touch disjoint files was **wrong**. It may run in parallel
  with E00 only. **NO** with E10/E15/E16/E22 (all own `boosting.rs`).

**Goal / observable completion condition.** `train_cat(..)` with
`params.simple_ctr` or `params.combinations_ctr` set to `FloatTargetMeanValue` or
`FeatureFreq` returns `Err(CbError::Unsupported(..))` naming the type and mirroring
upstream's wording, **before** any CTR accumulation or tree growth; a legal type is
unaffected.

**Files**
- Modify: `crates/cb-train/src/boosting.rs`
- Modify: `crates/cb-train/src/boosting_test.rs` (add the unit test) — **E03 also
  edits this file, so E02 lands AFTER E03 (edge `E03->E02`). This is a genuine
  edit-conflict serialization, not a logical dependency, and it is why W0's graph
  lists only `E00 ∥ E01 ∥ E03` as parallel.**

**Exact verified files/symbols to touch**
- `fn train_inner<R: Runtime>(runtime, feature_values, feature_borders,
  cat_columns, target, weights, params, staged_out, eval_sets, history, ranking)
  -> CbResult<(Model, BakedCtrData)>` at `crates/cb-train/src/boosting.rs:2555-2567`
  `[VERIFIED: LOCAL]`. It is the **single production consumer** — `train`
  (`:2037`), `train_with_eval_sets` (`:2139`) and `train_cat` (`:2236`) all
  delegate to it `[VERIFIED: LOCAL]`.
- Insertion point: immediately after the existing `params.loss.validate()?;` at
  `crates/cb-train/src/boosting.rs:2574`, and **before** the score-function
  rejection block at `:2576-2580` `[VERIFIED: LOCAL, read verbatim]`.
- `CbError::Unsupported(String)` `[VERIFIED: LOCAL crates/cb-core/src/error.rs:92]`.
- Upstream message to mirror: `catboost_options.cpp:504-509`
  `CB_ENSURE(IsSupportedCtrType(CPU, ctrType), "Ctr type " << ctrType << " is not implemented on CPU yet")`.

**CodeGraph / grep evidence that unconditional validation is SAFE.**
`grep -rn "simple_ctr: ECtrType::\|combinations_ctr: ECtrType::\|simple_ctr: cb_train::ECtrType::" crates/ --include=*.rs`
returns **zero matches** — **no test or production site anywhere pins a non-default
`ECtrType`**; every one of the 62 `BoostParams` construction sites uses
`simple_ctr_default()` / `combinations_ctr_default()`, both of which return
`ECtrType::Borders` `[VERIFIED: LOCAL grep; crates/cb-train/src/boosting.rs:464-466,509-511]`.
The guard therefore rejects **nothing** that exists today ⇒ zero behavior change.

**Red**
- File: `crates/cb-train/src/boosting_test.rs`
- Test fn: `cpu_illegal_ctr_types_are_typed_unsupported_before_training`
- Setup: a `BoostParams` literal pinning every field explicitly (Pitfall-6
  discipline), with `iterations: 1, depth: 1`, once with
  `simple_ctr: ECtrType::FloatTargetMeanValue` and once with
  `combinations_ctr: ECtrType::FeatureFreq`; a 4-row, 1-cat-column corpus.
- Action: `cb_train::train_cat(&CpuBackend::default(), &[], &[], &cat_cols, &y, &w, &params, None)`
- Expected output (typed error, per case):
  ```rust
  match result {
      Err(CbError::Unsupported(msg)) => {
          assert!(msg.contains("FloatTargetMeanValue"), "must name the type: {msg}");
          assert!(msg.contains("not implemented on CPU yet"), "must mirror upstream: {msg}");
      }
      other => panic!("expected CbError::Unsupported, got {other:?}"),
  }
  ```
  Plus a third case asserting `ECtrType::Borders` still returns `Ok`.
- **EXPECTED INITIAL FAILURE:** the `other =>` arm fires with
  `expected CbError::Unsupported, got Ok(..)` — today the type is never read
  (`grep 'params.simple_ctr' crates/cb-train/src/boosting.rs` → **zero reads**
  `[VERIFIED: LOCAL]`), so training silently succeeds with Borders CTRs.
- Run: `cargo test -p cb-train --lib boosting::tests -- cpu_illegal_ctr_types`

**Green (minimal implementation intent).** A private
`fn validate_ctr_types(params: &BoostParams) -> CbResult<()>` in `boosting.rs`
next to the existing `validate_score_function` (already imported by
`boosting_test.rs:24` `[VERIFIED: LOCAL]`), checking `params.simple_ctr` then
`params.combinations_ctr` via `ECtrType::is_cpu_supported`, returning
`CbError::Unsupported(format!("Ctr type {ty:?} is not implemented on CPU yet (upstream catboost_options.cpp:504-509)"))`.
Called once from `train_inner` at the verified insertion point.

**Refactor constraints + required regression scope**
- Constraint: place the call **after** `params.loss.validate()?` so the existing
  loss-validation error precedence is unchanged.
- Regression scope (MANDATORY, `boosting.rs` is touched): the 11 CTR oracles
  **and** the one-hot wave scope
  (`one_hot_oracle_test`, `one_hot_draw_accounting_test`,
  `device_one_hot_parity_test`) — §3.2.

**Validation**
```bash
cargo test -p cb-train --lib boosting::tests -- cpu_illegal_ctr_types
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test --test device_one_hot_parity_test
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** Both illegal-type cases return the typed error naming the
type; the Borders case returns `Ok`; **all 11 CTR oracles** (9 cb-train + 2
cb-model) and all 3 one-hot targets green.

---

### E03 — Covering test for `ctr_splits_for_tree` (MUST land before E10 modifies it)

- **Specs:** precondition of SPEC-CTRT-09 (SPEC §8 blocking constraint; research R7)
- **Blocked by:** none. **Blocks:** E02 (edit-conflict on `boosting_test.rs`), E10.
- **Parallelizable:** **YES** with E00 and E01 — owns
  `crates/cb-train/src/boosting_test.rs` only. **NO** with E02, which edits the
  same file (edge `E03->E02`).

**Goal / observable completion condition.** `ctr_splits_for_tree` — which
**CodeGraph confirms has ⚠️ NO covering tests** — gains a characterization test
pinning its current output, so E10's change to it is guarded rather than silent.

**Exact verified files/symbols to touch**
- `fn ctr_splits_for_tree(candidates: &[crate::candidates::CtrCandidate], priors: &[f64]) -> Vec<CtrSplitSpec>`
  at `crates/cb-train/src/boosting.rs:1929-1949` — **private**, so the test must
  live in the child module `boosting_test.rs`
  `[VERIFIED: LOCAL; mount at boosting.rs:5493-5495 `#[path = "boosting_test.rs"] mod tests;`]`.
- Its current body pins `prior_num = priors.first().copied().unwrap_or(0.5)`,
  `ctr_type: crate::ctr::ECtrType::Borders.as_i8()` (**:1940**),
  `prior_denom: 1.0`, `target_border_idx: 0`, `border: 0.0`, `shift: 0.0`,
  `scale: 1.0` `[VERIFIED: LOCAL, read verbatim]`.
- Its ONE caller: `crates/cb-train/src/boosting.rs:5318`
  `ctr_splits_for_tree(&ctr_candidates, &params.combinations_ctr_priors)`, reached
  only on the `!has_ctr` branch (the no-CTR-candidate fallback)
  `[VERIFIED: LOCAL sed -n '5305,5322p'; CODEGRAPH "1 caller in crates/cb-train/src/boosting.rs"]`.
- `crate::tree::CtrSplitSpec` fields `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:169-192]`;
  already imported by `boosting_test.rs:23` `[VERIFIED: LOCAL]`.
- `crate::candidates::CtrCandidate { projection: TProjection, is_simple: bool }`
  `[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:151-157]`.

**CodeGraph evidence for ordering.**
`ctr_splits_for_tree (crates/cb-train/src/boosting.rs:1929) — 1 caller in
crates/cb-train/src/boosting.rs; ⚠️ no covering tests found`
`[VERIFIED: CODEGRAPH]`. This is precisely the SPEC §8 blocking constraint and
research R7. **No task may modify this function until this test exists.**

**Red**
- File: `crates/cb-train/src/boosting_test.rs`
- Test fn: `ctr_splits_for_tree_emits_one_spec_per_candidate_with_the_head_prior`
- Setup / input:
  ```rust
  let cands = vec![
      CtrCandidate { projection: TProjection::from_features(&[0]),    is_simple: true },
      CtrCandidate { projection: TProjection::from_features(&[0, 1]), is_simple: false },
  ];
  let specs = super::ctr_splits_for_tree(&cands, &[0.25, 0.75]);
  ```
- Expected output (characterization — pins TODAY's behavior exactly):
  `specs.len() == 2`; for **both** specs `prior_num == 0.25` (head prior only),
  `prior_denom == 1.0`, `target_border_idx == 0`, `border == 0.0`,
  `shift == 0.0`, `scale == 1.0`, and
  `ctr_type == cb_train::ECtrType::Borders.as_i8()`; and
  `specs[i].projection == cands[i].projection` for both.
- A second test fn `ctr_splits_for_tree_empty_priors_defaults_to_half`:
  `super::ctr_splits_for_tree(&cands, &[])` ⇒ every `prior_num == 0.5`
  (pins the `unwrap_or(0.5)` fallback).
- **EXPECTED INITIAL FAILURE:** none from the assertions — this is a
  **characterization** test and passes on first write. Therefore the Red is the
  **MUTATION CHECK** (§3.1):
  **Mandatory mutation:** at `crates/cb-train/src/boosting.rs:1940` change
  `ECtrType::Borders.as_i8()` to `ECtrType::Counter.as_i8()`. Re-run.
  **Expected failure:** ``assertion `left == right` failed: left: 4, right: 0``.
  **Second mandatory mutation:** change `priors.first()` to `priors.last()`.
  **Expected failure:** `left: 0.75, right: 0.25`.
  Revert both manually; confirm green. **Record all three outputs.**
- Run: `cargo test -p cb-train --lib boosting::tests -- ctr_splits_for_tree`

**Green (minimal implementation intent).** **None** — the assertion set IS the
deliverable. No production change.

**Refactor constraints + required regression scope**
- Constraint: do NOT change `ctr_splits_for_tree` in this task. Its change is E10.
- Regression scope: `cargo test -p cb-train --lib boosting::tests`.

**Validation**
```bash
cargo test -p cb-train --lib boosting::tests -- ctr_splits_for_tree
cargo test -p cb-train --lib boosting::tests
cargo test -p cb-train
```

**Completion evidence.** Both characterization tests green; the recorded
pass → 2× mutated-fail → reverted-pass sequence.

---

## WAVE W1 — The regression firewall (zero behavior change)

> This wave is the single most important risk control in Part 1. It proves that
> today's Borders-binclf prefix is **bit-for-bit** the `(classes = 2, b = 0)`
> special case of the generic classes-prefix producer, so the W2/W3 refactor
> cannot silently move any of the 11 existing CTR oracles.

---

### E04 — `online_class_prefix`, the ONE generic classes-prefix producer

- **Specs:** SPEC-CTRT-04
- **Blocked by:** E01. **Blocks:** E05.
- **Parallelizable:** **NO** with E05/E06/E07/E08 — all own
  `crates/cb-train/src/ctr/online.rs` (edit-conflict serialization).

**Goal / observable completion condition.** A pure function
`online_class_prefix(counts: &[i64], target_border_idx: usize, ctr_type: ECtrType)
-> (f64, i64)` returning `Buckets → (N[b], Total)` and
`Borders → (Total − Σ_{c ≤ b} N[c], Total)`, table-driven-tested over
hand-computed vectors. **No caller yet** — E05 wires it.

**Files**
- Modify: `crates/cb-train/src/ctr/online.rs`
- Modify: `crates/cb-train/src/ctr/online_test.rs` (exists, mounted at
  `crates/cb-train/src/ctr/mod.rs:42-44` `[VERIFIED: LOCAL]`)
- Modify: `crates/cb-train/src/ctr/mod.rs` — add `online_class_prefix` to the
  existing `pub use online::{…}` re-export block at `:144-148`
  `[VERIFIED: LOCAL]`; and to `crates/cb-train/src/lib.rs:46-50`.

**Exact verified files/symbols to touch**
- `TCtrHistory { pub n: Vec<i64> }` with `total()` = `self.n.iter().sum()`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/online.rs:58-88]`.
- `SIMPLE_CLASSES_COUNT: usize = 2` `[VERIFIED: LOCAL online.rs:52]`.
- Upstream transcription targets (research §A.1):
  `online_ctr.cpp:115-121` —
  `UpdateGoodCount(curCount, ctrType, goodCount) { if (Buckets) *goodCount = curCount; else *goodCount -= curCount; }`
  applied cumulatively over `border = 0..targetBorderCount` starting from
  `goodCount = Total`; `online_ctr.cpp:168-185` — the read-THEN-increment order.

**CodeGraph evidence.** `crates/cb-train/src/ctr/online.rs` symbols
`accumulate_online` (`:163`) and `online_ctr_prefix_binclf` (`:263`) have **14** and
**(via calc_ctr_online) 10** callers with covering tests
`ctr_data_roundtrip_test.rs`, `final_ctr_test.rs`, `ctr/online_test.rs`,
`calc_ctr_test.rs`, `ordered_ctr_oracle_test.rs`, `plain_ctr_oracle_test.rs`,
`tensor_ctr_oracle_test.rs` `[VERIFIED: CODEGRAPH research §F.1]`. Adding a NEW
`pub(crate)` function changes none of them — this task is provably additive.

**Red**
- File: `crates/cb-train/src/ctr/online_test.rs`
- Test fn: `online_class_prefix_matches_hand_computed_upstream_vectors`
- Setup / input — a table of hand-computed cases:
  | `counts` | `b` | `ctr_type` | expected `(num, denom)` |
  |---|---|---|---|
  | `[3, 7]` | 0 | `Buckets` | `(3.0, 10)` |
  | `[3, 7]` | 1 | `Buckets` | `(7.0, 10)` |
  | `[3, 7]` | 0 | `Borders` | `(7.0, 10)`  — `10 − 3` |
  | `[2, 5, 4]` | 0 | `Buckets` | `(2.0, 11)` |
  | `[2, 5, 4]` | 1 | `Buckets` | `(5.0, 11)` |
  | `[2, 5, 4]` | 0 | `Borders` | `(9.0, 11)` — `11 − 2` |
  | `[2, 5, 4]` | 1 | `Borders` | `(4.0, 11)` — `11 − (2+5)` |
  | `[0, 0]` | 0 | `Borders` | `(0.0, 0)`   — empty bucket |
  | `[]` | 0 | `Buckets` | `(0.0, 0)`  — degenerate, no panic |
  | `[3, 7]` | 9 | `Buckets` | `(0.0, 10)` — out-of-range `b`, checked `.get` |
- Expected output: exact `assert_eq!` on both components (integers exactly
  representable in `f64`; **no tolerance**).
- **EXPECTED INITIAL FAILURE:**
  `error[E0425]: cannot find function `online_class_prefix` in module `super``
  (or `error[E0603]` if the re-export is added before the definition).
- Run: `cargo test -p cb-train --lib ctr::online_test -- online_class_prefix`

**Green (minimal implementation intent).** One `#[must_use] pub(crate) fn` in
`online.rs`:
```
total = counts.iter().sum::<i64>()
Buckets → (counts.get(b).copied().unwrap_or(0) as f64, total)
_       → (total − Σ_{c=0..=b} counts.get(c).copied().unwrap_or(0)) as f64, total)
```
Checked `.get` only; no indexing; no panic; `saturating_sub` on the cumulative
subtraction so a malformed `counts` can never underflow. Doc comment carries the
`online_ctr.cpp:115-121,168-185` anchors and states explicitly that
`Counter` is **not** a class-prefix type and must never be passed here (E06 owns
Counter).

**Refactor constraints + required regression scope**
- Constraint: do NOT modify `online_ctr_prefix_binclf` in this task (that is E05).
  Do NOT change `calc_ctr_online` / `calc_ctr_online_bin` / `calc_normalization` —
  they are type-agnostic and correct (research §A.0); each new arm supplies
  per-type **inputs**, never a new value function.
- Regression scope: `cargo test -p cb-train --lib ctr::` plus the 11 CTR oracles.

**Validation**
```bash
cargo test -p cb-train --lib ctr::online_test
cargo test -p cb-train --lib ctr::
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** The 10-row table test green; all 11 existing CTR oracles
green (they are untouched — the function has no caller yet).

---

### E05 — Borders-binclf IS the generic producer's special case (REGRESSION FIREWALL)

- **Specs:** SPEC-CTRT-05; acceptance **A7**; guards SPEC-CTRT-18
- **Blocked by:** E04. **Blocks:** E06, E15.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/ctr/online.rs`.

**Goal / observable completion condition.** A bit-equality test proves
`online_class_prefix(&[n0, n1], 0, ECtrType::Borders) == (n1 as f64, n0 + n1)` for
an exhaustive small grid, and `online_ctr_prefix_binclf` is then re-routed through
`online_class_prefix` with **all 11 existing CTR oracles still green**.

**Files**
- Modify: `crates/cb-train/src/ctr/online.rs`
- Modify: `crates/cb-train/src/ctr/online_test.rs`

**Exact verified files/symbols to touch**
- `online_ctr_prefix_binclf(permutation, bins, target_class, prior) -> CbResult<OnlineCtrPrefix>`
  at `crates/cb-train/src/ctr/online.rs:263-320`. Its read-before-increment core is
  lines **:297-310**:
  ```rust
  let elem = counts.get(bucket);
  let (n0, n1) = elem.map_or((0, 0), |e| (e[0], e[1]));
  let g = n1;          // good = N[1] (pos class)
  let t = n0 + n1;     // total = N[0] + N[1]
  ...
  *slot = calc_ctr_online(g as f64, t, prior);
  ```
  `[VERIFIED: LOCAL, read verbatim]`.
- `ordered_ctr_per_permutation` (`online.rs:361`) calls
  `online_ctr_prefix_binclf` then **re-derives the same loop** at `:377-401` for
  the step-wise `(num, denom)`. That second loop must be re-routed **identically**
  or the two will diverge — this is the subtle failure mode this task must close.
- `counts: Vec<[i64; SIMPLE_CLASSES_COUNT]>` is a fixed-size `[i64; 2]` array;
  `online_class_prefix` takes `&[i64]`, so the call site passes
  `elem.map_or(&[][..], |e| &e[..])` or reconstructs a 2-slot slice — **checked
  access only** (`indexing_slicing` is denied).

**CodeGraph evidence for ordering.** `online_ctr_prefix_binclf` is the shared
primitive behind `materialize_ctr_feature` (`ctr_feature.rs:204`) and
`ordered_ctr_per_permutation`, which together feed `plain_ctr`, `ordered_ctr`,
`tensor_ctr`, `tensor_ctr_e2e`, `s_order_ctr_bins`, `ctr_split_scoring`,
`ctr_feature_materialize`, `multi_permutation_e2e` and
`multi_permutation_fold` `[VERIFIED: CODEGRAPH + LOCAL research §F.1]`. Landing the
equivalence proof BEFORE the W2/W3 dispatch is what makes those 10 oracles a
meaningful gate rather than a coincidence.

**Red**
- File: `crates/cb-train/src/ctr/online_test.rs`
- Test fn 1 (the firewall):
  `borders_binclf_is_bit_identical_to_the_generic_class_prefix_at_b0`
  Setup / input: exhaustive `for n0 in 0..=12 { for n1 in 0..=12 {` (169 cases).
  Expected output — **bit equality on raw values, NOT a tolerance**:
  ```rust
  let (num, denom) = online_class_prefix(&[n0, n1], 0, ECtrType::Borders);
  assert_eq!(num.to_bits(), (n1 as f64).to_bits(),
      "generic Borders numerator must be BIT-identical to N[1] at n0={n0} n1={n1}");
  assert_eq!(denom, n0 + n1);
  ```
- Test fn 2 (proves the re-route did not change the public prefix):
  `online_ctr_prefix_binclf_output_is_unchanged_by_the_generic_reroute`
  Setup: a fixed 12-document permutation `[3,0,7,1,9,4,11,2,6,5,10,8]`, fixed
  `bins = [0,1,0,2,1,0,2,2,1,0,1,2]`, fixed
  `target_class = [1,0,1,1,0,0,1,0,1,1,0,1]`, `prior = 0.5`.
  Expected output: `good`, `total` and — crucially — `value` compared to a
  **hard-coded literal expected vector transcribed from the PRE-change run**, with
  `value` asserted by `to_bits()` equality (bit-for-bit, not `≤1e-5`).
  *Procedure:* run the test once **before** the Green edit, paste the observed
  vectors in as literals, then perform the Green edit. This makes the test a true
  frozen characterization rather than a self-comparison.
- Test fn 3: `ordered_ctr_per_permutation_step_counts_match_the_prefix_reroute`
  Expected: `OrderedCtrPrefix::per_bucket_monotone(&perm, &bins) == true` **and**
  `step_num` / `step_denom` equal the same hard-coded literals.
- **EXPECTED INITIAL FAILURE:** test fn 1 fails to compile with
  `error[E0425]: cannot find function `online_class_prefix`` **only if E04 has not
  landed**; with E04 landed, test fn 1 **passes on first write** — that is the
  point (it is the proof, not a driver). Test fns 2/3 also pass on first write.
  Therefore the Red for this task is the **MUTATION CHECK** (§3.1):
  **Mandatory mutation A:** in `online_class_prefix`, change the non-Buckets arm's
  cumulative bound from `c <= b` to `c < b`. Re-run.
  **Expected failure:** test fn 1 —
  ``generic Borders numerator must be BIT-identical to N[1] at n0=1 n1=0`` with
  `left: 4607182418800017408 (1.0), right: 0 (0.0)`.
  **Mandatory mutation B:** in the re-routed `online_ctr_prefix_binclf`, swap the
  returned numerator/denominator order. **Expected failure:** test fn 2's frozen
  `value` bit comparison. Revert both manually; confirm green.
  **Record all outputs.**
- Run: `cargo test -p cb-train --lib ctr::online_test`

**Green (minimal implementation intent).** Replace the two inline
`(n1, n0 + n1)` derivations — one in `online_ctr_prefix_binclf` (`online.rs:299-301`)
and one in `ordered_ctr_per_permutation` (`online.rs:393-395`) — with a single call
to `online_class_prefix(&slots, 0, ECtrType::Borders)`. **No other change.** The
public signatures, the return types, the read-before-increment ORDER and the
`calc_ctr_online` call all stay byte-identical.

**Refactor constraints + required regression scope**
- Constraint: the read-BEFORE-increment order is the no-leakage property. It must
  not move relative to the `++N[targetClass]` at `online.rs:311-316` / `:396-400`.
- Constraint: do NOT change `OnlineCtrPrefix` / `OrderedCtrPrefix` shapes.
- **Required regression scope: ALL 11 existing CTR oracles + the 3 one-hot
  targets.** This is the wave's entire purpose.

**Validation**
```bash
cargo test -p cb-train --lib ctr::
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test \
  --test device_one_hot_parity_test
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** The 169-case bit-equality test green; the two frozen
characterization tests green; **all 11 CTR oracles + 3 one-hot targets green**; the
recorded 2× mutated-fail → reverted-pass sequence.

---

> **This document continues.** The plan is FOUR documents forming ONE plan; task
> IDs are unique across them and §4's dependency edge list above is authoritative
> for ordering.
>
> | Document | Contents |
> |---|---|
> | `./PLAN.md` (this file) | front matter, working-tree state, locked decisions, wave-ordering deviations, shared conventions, verified commands, wave graph + edge list, **W0 (E00–E03)**, **W1 (E04–E05)** |
> | `./PLAN-W2-W3.md` | **W2 (E06–E13)** type routing becomes real; **W3 (E14–E17)** candidate expansion |
> | `./PLAN-W4-W5.md` | **W4 (E18–E20)** `.cbm` mean codec; **W5 (E21–E23)** `counter_calc_method` |
> | `./PLAN-PART2.md` | **Part 2 (F00–F23)** facade routing, amending `../catboost-builder-cat-features-routing/PLAN.md` |
