---
title: TDD plan — CTR split border is persisted in bin space but compared in value space
plan_id: BUG-CTRB
status: draft
format: markdown
plan_version: 2
updated_at: 2026-08-02T00:00:00Z
revision: "pass-1 plan-check revisions applied (1 CRITICAL, 2 MAJOR, 3 MINOR)"
source_spec: .planning/plans/ctr-split-border-space/SPEC.md
source_check: .planning/plans/ctr-split-border-space/PLAN-CHECK.md
inherits_conventions_from: .planning/plans/ctr-type-engine-and-facade-routing/PLAN.md §3, §3.1, §3.2
specifications:
  - SPEC-CTRB-01
  - SPEC-CTRB-02
  - SPEC-CTRB-03
  - SPEC-CTRB-04
  - SPEC-CTRB-05
tasks: 6
---

# TDD plan — BUG-CTRB (CTR split border space)

Plan only. **No production code is written by the planner.** Every task below is
an executable prompt: exact file, exact line, exact command, exact assertion.

## Revision log

**v2 — pass-1 plan-check revisions** (`PLAN-CHECK.md`, verdict ISSUES_FOUND:
1 CRITICAL, 2 MAJOR, 3 MINOR). All six required revisions applied; **none
rejected.**

| Finding | What changed |
|---|---|
| **CRITICAL** — "principal hazard" was factually wrong; C04's M2 was unsatisfiable; C03 mandated a false production comment | §0 rewritten (converting `:3289` is arithmetically a **no-op** on the `u32` operand — E20); new **D3**/**D3a**; C03's `:3287-3290` comment replaced with an accurate units-contract text; **C04 M2 restated to exactly the two failures that can occur, with the oracles' staying green recorded as a REQUIRED result**; C01 step 6, C04 completion criteria and §6's SPEC-CTRB-03 row realigned |
| **MAJOR** — C04 ∥ C05 not disjoint | **Serialized `C04 → C05`** (4 → 5 waves); C05's dependency corrected to C04; `crates/cb-train/src/tree.rs` declared an exclusive resource; pre-flight **and** post-flight `git diff crates/cb-train/src/tree.rs` checks added to both tasks |
| **MAJOR** — f32 formula is not general | **E6** relabelled (arithmetic `[VERIFIED]`, generality `[INFERRED]`); **D2**'s rationale re-based on the `.cbm` f32 fixed point (E14); helper doc gained a **DOMAIN / reachability / STOP CONDITION** block; **C04 Tests B and C bounded to `0..ctr_border_count_default()`** (no `0..=254`); **new Test E** — `b = 16` boundary characterization naming `boosting.rs:3238`, which also becomes D2's discriminator; new **OQ-D** |
| **MINOR-1** — C01's reconstruction is bit-exact only at `norm == 1.0` | Prior **pinned non-tunable** at `0.5/1.0` with `assert_eq!((shift, norm), (0.0, 1.0))` before the premise check (E22); corpus tuning scoped to rows/categories only |
| **MINOR-4** — C05 had no `save_cbm` precedent and no fallback | **E23** added; C05 gained a mandatory **step-5 fallback gate** (3 in-memory assertions + its own mutation) so SPEC-CTRB-05 is never left ungated |
| **MINOR-5** — untracked plan directory | `.planning/plans/ctr-split-border-space/` added to §1's protected list (now **three** paths) |
| **MINOR-6** — `<mod>` placeholder | Replaced with **`tree::general`** (verified mount, `tree.rs:92-94`) |

Also folded in: `ctr_border_count_default()` = `15` at
`crates/cb-train/src/boosting.rs:529-531`, called at exactly one production site
`:3238` (**E21**); the E12 target has **four** `#[test]` fns (§1); the workspace
enables no cast lint (**E24**).

---

## 0. THE TWO ADJACENT `border:` SITES — units, and the ONLY detector

> **CORRECTED at plan-check pass 1.** Plan v1 called this a "principal hazard"
> and claimed that converting `tree.rs:3289` "silently corrupts leaf-value
> assignment". **That claim was FALSE and is retracted.** The corrected framing
> below is what every task now encodes; the revised SPEC-CTRB-03 says the same.

`crates/cb-train/src/tree.rs` contains **TWO adjacent `border: *border` sites,
eight lines apart, both `f64`, both fed by the SAME `CtrAwareSplit::Ctr { col,
border }` binding.** They carry **different units**. Exactly ONE of them changes.

```text
crates/cb-train/src/tree.rs
────────────────────────────────────────────────────────────────────────────────
3287    level_kinds.push(LevelKind::Ctr {
3288        ctr_idx: ctr_splits.len(),
3289        border: *border,          ← BIN SPACE.   Leave it. (units contract)
3290    });
3291    ctr_splits.push(CtrSplitSpec {
....
3297        border: *border,          ← VALUE SPACE. *** THIS ONE. ***
....
3303    });
```

### `:3297` — `CtrSplitSpec.border` — VALUE SPACE — this is the defect

Persisted into the model, lifted by `cb_model::Model::from_trained`'s `lift_ctr`
closure (`crates/cb-model/src/model.rs:366-383`, `border: c.border` at `:376`)
into `CtrSplit.border`, and consumed by `passes_ctr_split` as
`ctr_value > split.border` where `ctr_value` is the SCALED inference value
`[VERIFIED: CODEGRAPH passes_ctr_split, crates/cb-model/src/apply.rs:157-189]`.

### `:3289` — `LevelKind::Ctr.border` — BIN SPACE — leave it, but know why

**Converting it too would be arithmetically a NO-OP over the reachable domain.
It is not a correctness trap. It is a units contract.**

- Its **only** consumer of the border payload workspace-wide is
  `assign_leaf_of_averaging` (`crates/cb-train/src/boosting.rs:1887`, CTR arm
  `:1926-1938`), which evaluates `f64::from(bin) > *border` against
  `CtrFeatureColumn::bins`, declared **`pub bins: Vec<u32>`**
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/ctr_feature.rs:85-87]`
  `[VERIFIED: CODEGRAPH assign_leaf_of_averaging]`.
  Every other `LevelKind::Ctr` match site ignores the border (`model.rs:450`,
  `boosting.rs:4109`, `tree_one_hot_fused_test.rs:105,262` all bind `{ .. }` /
  `{ ctr_idx, .. }`) `[VERIFIED: CODEGRAPH, plan-check pass 1]`.
- `f64::from(u32)` is exact, so for **integer** `bin` and **integer** `b`:

  ```text
  bin > b   ⟺   bin >= b+1   ⟺   bin > (b+1) − 2⁻²⁰
  ```

  The bin-space and value-space thresholds are **exactly equivalent** on this
  operand for every `b` where `(b+1) − 2⁻²⁰ ∈ (b, b+1)` — i.e. `b <= 15` (E6).
- The reachable domain is `b ∈ 0..=14`: candidates are enumerated
  `for border_idx in 0..ctr_border_count` (`tree.rs:3162`), and
  `ctr_border_count` is **not configurable** — `ctr_border_count_default()`
  returns `15` (`crates/cb-train/src/boosting.rs:529-531`) and is called at
  **exactly one** production site, `boosting.rs:3238`
  `[VERIFIED: LOCAL grep "ctr_border_count_default()" crates/cb-train/src/boosting.rs → :529, :3238 only]`.

**⇒ THE OPERATIONAL CONSEQUENCE, which every task must respect:**

> **No runtime gate can detect a wrong-site conversion.** The E12 gate, all 11
> CTR oracles and all 3 one-hot targets stay **GREEN** whether or not `:3289` is
> converted. The **only** detectors are the explicit integrality assertions —
> C01 step 6 and C04 Test D (`assert_eq!(bin_border, bin_border.trunc())`).
> **Any plan step, completion criterion or review expectation that an oracle
> will regress when `:3289` is converted is UNSATISFIABLE and must not be
> written.** (SPEC-CTRB-03, revised.)

A third `*border` read exists in the same neighbourhood and is **also bin space
and also unchanged**: `passes_ctr_aware`
(`crates/cb-train/src/tree.rs:2589-2602`, the training split test
`f64::from(bin) > *border`) `[VERIFIED: LOCAL tree.rs:2597-2600]`.

### Enforcement

1. **C03**'s diff gate: the `tree.rs` diff must contain **exactly one** changed
   expression plus comment-only hunks.
2. **C01 step 6** and **C04 Test D**: named integrality assertions — the only
   possible detector.
3. **C04 mutation M2**: converts `:3289` deliberately and records that the two
   integrality assertions fail **while the E12 gate and the 11 oracles stay
   green**. The green half is not a nuisance result — it is the *empirical
   confirmation* of the no-op arithmetic above, and an oracle regression under
   M2 would be a STOP-AND-REPORT (it would mean this reasoning is wrong).

---

## 1. Working-tree state this plan lands on (verified, load-bearing)

`[VERIFIED: LOCAL git log/status, executed 2026-08-02]`

- Branch `fix/bootstrap-rng-draw-accounting`; HEAD `fed705f`
  *"E11 — per-type final tables in the bake path + mean threading"*, preceded by
  `c4b82c4` (E10) and `c370d00` (build: move the debuginfo override out of the
  shared manifest).
- **THREE UNTRACKED paths MUST NOT be lost** (`git status --short` at `fed705f`)
  `[VERIFIED: LOCAL]`:
  - `crates/cb-oracle/fixtures/ctr_counter_simple/`
    (`config.json`, `gen_fixtures.py`, `model.json`, `predictions.npy`,
    `X_cat.npy`, `y.npy`) — E12's work-in-progress fixture
  - `crates/cb-train/tests/ctr_counter_simple_oracle_test.rs` — E12's gate
  - **`.planning/plans/ctr-split-border-space/`** — **this plan and its SPEC and
    PLAN-CHECK.** `git clean` would delete the documents an executor is working
    from. *(Added at plan-check pass 1, MINOR-5.)*

  → **`git checkout --`, `git stash`, `git clean` are FORBIDDEN in every task of
  this plan, including every mutation revert.** All reverts are MANUAL edits.
  This restates ctr-type `PLAN.md` §3.1 step 4 and is stricter here because all
  three paths are unrecoverable.
- `.cargo/config.toml` exists, is **deliberately uncommitted**
  (`.git/info/exclude`), and sets `debug = "line-tables-only"` +
  `split-debuginfo = "unpacked"` for `[profile.dev]` / `[profile.test]`.
  It changes **no** codegen, opt-level, overflow checks or debug assertions —
  **test semantics are identical**; it exists only to keep `target/` off the
  documented disk-exhaustion failure mode. No task may remove it, and no task may
  rely on full DWARF being present.
- **The defect is live and reproduced.** `[VERIFIED: LOCAL executed]`

  ```text
  $ cargo test -p cb-train --test ctr_counter_simple_oracle_test
  running 4 tests
  test counter_column_is_permutation_invariant_end_to_end ... ok
  test borders_column_is_not_permutation_invariant_on_the_same_corpus ... ok
  test counter_simple_model_actually_carries_a_counter_ctr_split ... ok
  test counter_simple_predictions_match_upstream_within_1e_minus_5 ... FAILED
    panicked at crates/cb-train/tests/ctr_counter_simple_oracle_test.rs:159:9:
    Counter-CTR predictions diverged from upstream (max |diff| = 2.6874900161694987e-1):
    StageDiverged { stage: Predictions, index: 0, expected: -0.12070300927021438,
                    actual: 0.1480459923467355, diff: 0.26874900161694987 }
  test result: FAILED. 3 passed; 1 failed
  ```

  The `max |diff|` matches SPEC.md §6 A-CTRB-1 exactly.

  **Minor correction to the plan request:** the target contains **4** `#[test]`
  functions, not 5 `[VERIFIED: LOCAL cargo test output above]`. "E12 passes 5/5"
  in the request is off by one; the gate this plan must satisfy is **4/4 in
  `ctr_counter_simple_oracle_test`**. Recorded so an executor does not go hunting
  for a missing fifth test.

---

## 2. Locked decisions this plan encodes (non-negotiable)

| # | Decision | Basis |
|---|---|---|
| D1 | The persisted-border conversion is applied at **`crates/cb-train/src/tree.rs:3297` ONLY**. | §0 |
| D2 | The conversion is **`f64::from((b as f32 + 1.0) - f32::powi(2.0, -20))`**, computed in **f32**, widened once. **Rationale (corrected at plan-check pass 1): the `.cbm` codec narrows `Borders` to f32 on save and widens via `f64::from` on load (E14), so the persisted value MUST be an f32 fixed point. The f32 formulation is one; the naive f64 formulation is not (E6). The rationale is NOT "parity generality" — neither formulation is correct above `b = 15` (E6, SPEC-CTRB-01 BOUNDED DOMAIN).** | SPEC-CTRB-01 Invariant 2; §4 E4/E5/E6/E14 |
| D3 | `LevelKind::Ctr.border` stays **bin space**, byte-unchanged — as a **units contract**, not as a correctness requirement. Converting it would be arithmetically a **no-op** (§0, E20). No runtime gate can detect a wrong-site conversion; only C01 step 6 / C04 Test D can. | SPEC-CTRB-03 (revised); §0 |
| D3a | **No task may expect an oracle, the E12 gate or any one-hot target to regress when `:3289` is converted.** Such an expectation is unsatisfiable. | SPEC-CTRB-03 "Consequence for the plan" |
| D4 | `passes_ctr_aware` (`tree.rs:2597-2600`) and the candidate enumeration (`tree.rs:3162-3163`) stay **bin space**, byte-unchanged. | SPEC.md §2 non-goals |
| D5 | `ctr_splits_for_tree`'s `border: 0.0` (`boosting.rs:2015`) is **NOT converted**. | SPEC-CTRB-04; §4 E9 |
| D6 | `crates/cb-model/src/apply.rs` is **NOT** changed. It is already correct and is what makes upstream `.cbm` models score correctly today. | SPEC.md §2 |
| D7 | **No fixture is regenerated.** `crates/cb-oracle/generator/gen_fixtures.py` is **never** invoked. | ctr-type `PLAN.md` §3 |
| D8 | The 11 CTR oracles are a **NON-REGRESSION gate only**. They cannot prove this defect fixed (§4 E10) and must never be cited as this plan's falsifiable gate. | SPEC.md §1 |
| D9 | `catboost-master/` is a stale 3-file stub of a **different revision** and is **never** cited as a parity reference. Parity evidence = the committed fixtures + the `v1.2.10` tag + the installed `catboost==1.2.10`. | CLAUDE.md correction block |

---

## 3. Shared conventions

**Inherited verbatim** from
`.planning/plans/ctr-type-engine-and-facade-routing/PLAN.md`:

- **§3** — source/test separation, the no-`unwrap`/`expect`/`panic!`/raw-indexing
  production ban, typed errors, the never-`gen_fixtures.py` rule, the accepted
  failing-test baseline, the `≤1e-5` parity bar.
- **§3.1** — the guard-test falsifiability (MUTATION CHECK) protocol.
- **§3.2** — the repository-verified command blocks, **including the exact
  11-CTR-oracle block and the 3 one-hot targets**, reproduced in §3.2 below
  verbatim so no task has to cross-reference mid-execution.

### 3.0 Deltas and clarifications specific to this plan

- **Source/test separation (CLAUDE.md, MANDATORY).** No `mod tests` and no
  `#[cfg(test)]` block inside any production file. Existing mounts this plan
  reuses / must not disturb:
  - `crates/cb-train/src/tree.rs:92-94` → `#[cfg(test)] #[path = "tree_test.rs"]
    **`mod general;`** → filter prefix **`tree::general`**; `cargo test -p cb-train
    --lib tree::` selects all five tree test modules (`general`, `tie_break`,
    `ordered`, `pairwise`, `tree_one_hot_fused_test`)
    `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:88-110]`.
    `tree_test.rs` carries **no** file-level `#![allow(...)]` — it relies on the
    crate-level `#![cfg_attr(test, allow(...))]`; do **not** add one
    `[VERIFIED: LOCAL crates/cb-train/src/tree_test.rs:1-7]`.
  - `crates/cb-train/src/boosting.rs:5631` → `boosting_test.rs`
    (filter `boosting::tests`) — **not modified by this plan**
  - `crates/cb-train/src/ctr/mod.rs:43,46,49,52` → `online_test`,
    `calc_ctr_test`, `final_ctr_test`, `mod_test` — **not modified by this plan**

  New **integration** tests go in `crates/<crate>/tests/` as their own targets.
- **Test-code lint exemption** uses the established file-level form (precedent
  `crates/cb-train/tests/ctr_counter_simple_oracle_test.rs:15`):
  `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]`.
- **Dev-dependency edge that makes the cross-space tests possible:** `cb-model`
  is a **dev-dependency of `cb-train`**, and `cb-train` is a normal dependency of
  `cb-model`. A `crates/cb-train/tests/*.rs` integration test may therefore use
  **both** the trainer and `cb_model`'s apply surface with no runtime cycle
  `[VERIFIED: LOCAL crates/cb-train/Cargo.toml [dev-dependencies]]`.
  `serde_json`, `ndarray`, `ndarray-npy`, `cb-oracle`, `approx` are likewise
  available to cb-train integration tests `[VERIFIED: LOCAL same file]`.
- **Temp files** use `std::env::temp_dir()` with a pid-tagged name — the
  established pattern (`crates/cb-model/tests/cbm_oracle_test.rs:128`,
  `float_only_byte_identity_test.rs:139`) `[VERIFIED: LOCAL grep]`.
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
the task is **not complete**.

### 3.2 Repository-verified commands

```bash
# ── THE PRIMARY DEFECT GATE (this plan's own falsifiable gate) ───────────────
cargo test -p cb-train --test ctr_border_space_test          # C01 (new target)
cargo test -p cb-train --test ctr_border_upstream_anchor_test # C02 (new target)
cargo test -p cb-model --test ctr_border_cbm_roundtrip_test   # C05 (new target)

# ── THE E12 INTEGRATION GATE (SECONDARY — it does NOT localize) ──────────────
cargo test -p cb-train --test ctr_counter_simple_oracle_test   # must be 4/4

# ── cb-train unit tests (child-module filters) ──────────────────────────────
cargo test -p cb-train --lib tree::general      # C04 — `mod general;` (tree.rs:92-94)
cargo test -p cb-train --lib tree::             # all five tree test modules
cargo test -p cb-train --lib boosting::tests

# ── THE 11 EXISTING CTR ORACLES — run the WHOLE block, never a subset ────────
# 9 in cb-train + 2 in cb-model. DIFF GATE for this plan: ALL ELEVEN files must
# show ZERO diff (`git diff --stat` prints nothing for them). No task in this
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

# ── THE ONE-HOT WAVE REGRESSION SCOPE (A12) — mandatory for every
#    boosting.rs / tree.rs task ───────────────────────────────────────────────
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

# ── FORBIDDEN ───────────────────────────────────────────────────────────────
# crates/cb-oracle/generator/gen_fixtures.py      NEVER
# git checkout -- / git stash / git clean         NEVER (untracked E12 artifacts)
```

---

## 4. Evidence ledger

Every claim this plan rests on, with its verification label. Tasks cite by tag.

| Tag | Claim | Evidence |
|---|---|---|
| **E1** | The CTR candidate enumeration produces **integer bin indices**: `for border_idx in 0..ctr_border_count { let border = border_idx as f64; … CtrAwareSplit::Ctr { col, border } }`. | `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:3162-3164]` |
| **E2** | The **training** split test is bin space: `passes_ctr_aware` → `.and_then(\|c\| c.bins.get(obj)).is_some_and(\|&bin\| f64::from(bin) > *border)`. | `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:2597-2600]` |
| **E3** | The **apply** split test is value space: `passes_ctr_split` ends `ctr_value > split.border`, where `ctr_value` is `ctr_value_for_combined_projection(...)` / `calc_inference(...)` — the **`(ctr + Shift) * Scale`** scaled value. | `[VERIFIED: CODEGRAPH passes_ctr_split; crates/cb-model/src/apply.rs:157-189; crates/cb-model/src/ctr_data.rs:274-282]` |
| **E4** | Upstream stores CTR borders as `(bin + 1) − 2⁻²⁰` in **f32**. All committed CTR fixture borders: `tensor_ctr_e2e` `[2.9999990463256836, 7.999999046325684]`; `ctr_counter_simple` `[8.999999046325684, 10.999999046325684]`; `fstr_ctr` `[3.9999990463256836, 6.999999046325684, 11.999999046325684]` and `[3.9999990463256836]`. | `[VERIFIED: LOCAL .venv/bin/python json dump of each fixture model.json → features_info.ctrs[*].borders]` |
| **E5** | `(b as f32 + 1.0) − f32::powi(2.0, −20)`, widened to f64, **reproduces every value in E4 bit-for-bit** for b ∈ {2,3,6,7,8,10,11}, and is an **f32 fixed point** (`f64::from(x as f32) == x`) for every b in `0..=15`. | `[VERIFIED: LOCAL numpy float32 recomputation, all 8 values exact, all fixed-point true]` |
| **E6** | **BOUNDED DOMAIN — corrected at plan-check pass 1.** The two invariants of SPEC-CTRB-01 hold **together only for `b <= 15`**. Measured: <br>`b = 0..15` — f32 form `b.999999046325684`, in `(b, b+1)` ✓, f32 fixed point ✓; f64 form identical, fixed point ✓. <br>`b = 16` — **f32 form `17.0`, NOT in `(b, b+1)` ✗**, fixed point ✓; f64 form `16.999999046325684`, **NOT an f32 fixed point ✗**. <br>`b = 17/31/127/254` — f32 form `18.0 / 32.0 / 128.0 / 255.0`, same collapse. <br>**Conclusion: NEITHER formulation is generally correct above 15.** The f32 form loses Invariant 1 (strict interval); the f64 form loses Invariant 2 (f32 fixed point). **D2 mandates f32 because Invariant 2 is the one the `.cbm` codec ENFORCES (E14), not because f32 is "parity-correct in general".** | Arithmetic: `[VERIFIED: LOCAL numpy float32 vs float64 sweep, independently reproduced at plan-check pass 1]`. The claim that upstream uses this expression **at all border counts**: `[INFERRED]` — no in-repo artifact reaches `b >= 16`, and `catboost-master/` may not be cited (D9). A different upstream epsilon (e.g. `(b+1) − 1e-6f`) reproduces every committed value identically for `b <= 15` but rounds the OPPOSITE way at `b = 16`, so the formula is **underdetermined above 15**. |
| **E20** | **Converting `LevelKind::Ctr.border` (`tree.rs:3289`) is arithmetically a NO-OP.** Its only border-consuming site is `assign_leaf_of_averaging` (`boosting.rs:1926-1938`), whose left operand is `CtrFeatureColumn::bins`, declared `pub bins: Vec<u32>`. `f64::from(u32)` is exact, so for integer `bin`/`b`: `bin > b ⟺ bin >= b+1 ⟺ bin > (b+1) − 2⁻²⁰` for every `b <= 15`. **⇒ No oracle, no one-hot target and not the E12 gate can detect a wrong-site conversion.** | `[VERIFIED: LOCAL crates/cb-train/src/ctr/ctr_feature.rs:85-87 `pub bins: Vec<u32>`; crates/cb-train/src/boosting.rs:1926-1938]` `[VERIFIED: CODEGRAPH assign_leaf_of_averaging — full border-payload consumer set is `boosting.rs:1938` only; `model.rs:450`, `boosting.rs:4109`, `tree_one_hot_fused_test.rs:105,262` bind `{ .. }` and ignore it]` |
| **E21** | **`b` is bounded to `0..=14`.** `ctr_border_count_default()` returns `15` (`crates/cb-train/src/boosting.rs:529-531`) and is called at **exactly one** production site, `crates/cb-train/src/boosting.rs:3238` (`let ctr_border_count = ctr_border_count_default();`). It is **not** plumbed through `BoostParams`. Candidates are `for border_idx in 0..ctr_border_count` (`tree.rs:3162`), so `b <= 14` — one border of margin inside E6's safe range. | `[VERIFIED: LOCAL grep -n "ctr_border_count_default()" crates/cb-train/src/boosting.rs → :529, :3238 only; both lines read]` |
| **E22** | `calc_normalization(0.5)` returns `(shift, norm) = (0.0, 1.0)`: `left = min(0, 0.5) = 0`, `right = max(1, 0.5) = 1`, `shift = -left = 0`, `norm = right − left = 1`. This is what makes C01 step 3's algebraic reconstruction **bit-exact**; at `norm != 1.0` it would not be (`(a/n)*b` vs `a*(b/n)`). | `[VERIFIED: LOCAL crates/cb-train/src/ctr/calc_ctr.rs:60-66]` |
| **E23** | **No in-repo precedent for `save_cbm` on a TRAINER-produced CTR model.** `save_cbm` is exercised by `non_symmetric_grower_roundtrip_oracle_test`, `class_params_roundtrip_test`, `cbm_oracle_test`, `non_symmetric_oracle_test`, `float_only_byte_identity_test` (all float / non-symmetric) and `ctr_nonmean_byte_identity_test` (CTR, but a **hand-built** `CtrData`, never `CtrData::from_baked`). C05 therefore needs an explicit fallback (MINOR-4). | `[VERIFIED: LOCAL grep -rln save_cbm crates/*/tests, plan-check pass 1]` |
| **E24** | The workspace lint set (`[workspace.lints.clippy]`) denies only `unwrap_used`, `expect_used`, `panic`, `indexing_slicing`. **No cast lint is enabled**, so C03's `as f32` will not trip clippy. | `[VERIFIED: LOCAL Cargo.toml:10-14; confirmed at plan-check pass 1]` |
| **E7** | `bin = trunc(v)`: `materialize_ctr_feature` computes `bin_f` (the apply-space value) then `bins[i] = clamp(bin_f.trunc())` into `[0, ctr_border_count]`. For non-mean types `bin_f = calc_ctr_online_bin(good, total, prior, bc) = (ctr + shift) / norm * bc`, i.e. exactly `(ctr + Shift) * Scale` with `Scale = bc/norm` — `calc_inference`'s tail verbatim. | `[VERIFIED: LOCAL crates/cb-train/src/ctr/ctr_feature.rs:263-292; crates/cb-train/src/ctr/calc_ctr.rs:89-101; crates/cb-model/src/ctr_data.rs:281]` |
| **E8** | **An existing GREEN oracle already treats the untruncated `calc_ctr_online_bin` output as living in the SAME space as upstream's persisted borders** — `multi_permutation_fold_oracle_test::averaging_partition` computes `bit0 = bin > 7.999_999_046_325_684`, `bit1 = bin > 2.999_999_046_325_684` on the raw quantizer output and reproduces catboost 1.2.10's committed tree-0 `leaf_weights`. This is independent, pre-existing, upstream-anchored confirmation of the `trunc(v) > b ⟺ v > (b+1)−2⁻²⁰` identity. | `[VERIFIED: LOCAL crates/cb-train/tests/multi_permutation_fold_oracle_test.rs:104-108, 186-200]` |
| **E9** | `ctr_splits_for_tree` (`boosting.rs:1988-2021`, `border: 0.0` at `:2015`) is called **only** on the `!has_ctr` branch (`boosting.rs:5419-5429`), where `has_ctr = !materialized_ctr_features.is_empty()` (`:4657`) — so the candidate list is empty and the function returns an **empty `Vec`**; the `border: 0.0` literal is never constructed in production. The code comment says so: *"`ctr_splits_for_tree` is retained for the no-CTR candidate path (it returns empty there)"* (`:5417-5418`). | `[VERIFIED: LOCAL all four line ranges read]` |
| **E10** | The 11 CTR oracles are green **by data-dependent coincidence**: a divergence needs a document whose CTR bin lands exactly on a chosen border, and none does in any pre-existing fixture. They are a **non-regression gate only**. | `[INFERRED from E2+E3+E7: for bin `b` and border `b`, training gives `b > b` = false, apply gives `v > b` = true for every `v ∈ (b, b+1)`]` + `[VERIFIED: LOCAL all 11 green at HEAD while `ctr_counter_simple` fails]` |
| **E11** | `CtrSplitSpec { … }` is constructed at exactly **four** sites workspace-wide: `crates/cb-train/src/tree.rs:3291` (production, oblivious CTR search), `crates/cb-train/src/boosting.rs:2007` (production, dead `!has_ctr` fallback per E9), `crates/cb-train/src/boosting_test.rs:337` (**test**, hand-built literal `0.5`), `crates/cb-model/tests/mixed_kind_split_order_test.rs:47` (**test**, hand-built literal `0.9`). | `[VERIFIED: LOCAL grep -rn "CtrSplitSpec" --include=*.rs crates/]` + `[VERIFIED: CODEGRAPH blast radius: 6 callers in lib.rs/boosting.rs/tree.rs; tests: boosting_test.rs]` |
| **E12** | **SPEC.md §9 OQ1 is incomplete and is corrected here.** It names three sites (`tree.rs:3291`, `boosting.rs:2007`, `boosting_test.rs:337`) and **misses `crates/cb-model/tests/mixed_kind_split_order_test.rs:47`**. That fourth site is a hand-constructed TEST model with a literal border `CTR_BORDER = 0.9` chosen so the not-found→empty apply value `0.5` deterministically fails the split — **it is unaffected by this change**. | `[VERIFIED: LOCAL crates/cb-model/tests/mixed_kind_split_order_test.rs:29-33, 47-58]` |
| **E13** | **No non-oblivious / region / leaf-wise / ordered / pairwise / perturbed grower persists CTR borders.** Every other `GrownTree` producer sets `ctr_splits: Vec::new()`: `greedy_tensor_search_oblivious_perturbed` (`tree.rs:774`), `leaf_wise_grower` (`:2089`), `region_grower` (`:2238`), `greedy_tensor_search_oblivious_ordered` (`:2541`), `greedy_tensor_search_oblivious_pairwise` (`:3708`). Only `greedy_tensor_search_oblivious_with_ctr` populates it (`:3291`). **OQ1 is CLOSED: no.** | `[VERIFIED: LOCAL grep + awk enclosing-function resolution over tree.rs]` |
| **E14** | `.cbm` narrows CTR borders to **f32** on save (`build_tctr_feature`: `identity.borders.iter().map(\|&b\| b as f32)`, `cbm.rs:437`) and widens on load (`let border = f64::from(borders.get(border_index));`, `cbm.rs:601`). The encode-side lookup `ctr_split_to_global_index` (`cbm.rs:351-382`) resolves the split by **exact bit equality** — `identity.borders.iter().position(\|b\| b.to_bits() == split.border.to_bits())` (`:366`) — against borders collected **from the model's own splits** (`build_ctr_features`, `cbm.rs:291`), so the lookup cannot miss regardless of the border's value. | `[VERIFIED: CODEGRAPH ctr_split_to_global_index, build_tctr_feature]` + `[VERIFIED: LOCAL cbm.rs:233-249, 291, 601]` |
| **E15** | `crates/cb-model/tests/ctr_nonmean_byte_identity_test.rs` (E00's frozen byte baseline) builds its model **by hand** from `cb_model::{CtrSplit, ObliviousTree, …}` with literal borders and pinned bucket hashes — *"Fully deterministic: no RNG, no trainer, no fixture inputs"* — so the trainer's border convention cannot reach it. | `[VERIFIED: LOCAL ctr_nonmean_byte_identity_test.rs:1-60]` |
| **E16** | The only existing assertion anywhere on a **trained** `CtrSplitSpec.border` value is `ctr_split_scoring_test.rs:122-123`: `assert!(border >= 0.0 && border < 10.0)`, on a column with `bins = [0,0,0,0,10,10,10,10]`. Under strict first-wins over the enumeration order `0..15`, **`border_idx = 0` wins**, so the persisted value goes `0.0 → 1.0 − 2⁻²⁰ ≈ 0.99999905`, which still satisfies `>= 0.0 && < 10.0`. **The assertion survives unchanged.** | `[VERIFIED: LOCAL ctr_split_scoring_test.rs:89-129 read in full; strict `>` first-wins at tree.rs:3175-3182]` |
| **E17** | `crates/cb-train/tests/ctr_split_scoring_test.rs:268` asserts only `grown.ctr_splits.len() == 2` (a **count**), and `:238-239` names the chosen bins in a **comment** (`borders 7 then 2`) with no value assertion. Unaffected. | `[VERIFIED: LOCAL ctr_split_scoring_test.rs:217-276]` |
| **E18** | `cb_oracle::FeaturesInfoJson` exposes **only** `float_features`; there is **no** typed accessor for `features_info.ctrs[*].borders`. C02 must parse `model.json` with `serde_json::Value` directly (precedent: `multi_permutation_fold_oracle_test.rs:121-133` parses a committed JSON with raw `serde_json`). | `[VERIFIED: LOCAL crates/cb-oracle/src/model_json.rs:265-270; multi_permutation_fold_oracle_test.rs:121-133]` |
| **E19** | `.cbm` round-trip API: `pub fn save_cbm(model: &Model, path: &Path)` (`cbm.rs:634`), `pub fn load_cbm(path: &Path) -> Result<Model, ModelError>` (`cbm.rs:1028`), `pub fn decode_cbm(buf: &[u8])` (`cbm.rs:1038`). `save_cbm` appends the CTR model-parts tail when `model.ctr_data.is_some()` (`:649-651`). | `[VERIFIED: LOCAL cbm.rs:634-654, 1028-1041]` |

---

## 5. Execution waves and dependency order

```text
W0  RED — two independent falsifying gates (disjoint NEW files → truly parallel)
      C01 ∥ C02
        C01  crates/cb-train/tests/ctr_border_space_test.rs           [CTRB-02, CTRB-03]
        C02  crates/cb-train/tests/ctr_border_upstream_anchor_test.rs [CTRB-01]

W1  GREEN — the one-expression fix
      C03  crates/cb-train/src/tree.rs:3297                            [CTRB-01, -02, -03]

W2  REFACTOR — extract and pin the conversion
      C04  tree.rs (named helper + units comments) + tree_test.rs      [CTRB-01, CTRB-03]

W3  ROUND-TRIP — the .cbm guard (SERIALIZED after C04, see below)
      C05  crates/cb-model/tests/ctr_border_cbm_roundtrip_test.rs      [CTRB-05]

W4  CLOSE-OUT — decisions, blast-radius confirmations, evidence
      C06                                                              [CTRB-03, CTRB-04]

Dependency edges
      C01 ─┐
           ├─> C03 ──> C04 ──> C05 ──> C06
      C02 ─┘
Acyclic. Longest path: C01|C02 -> C03 -> C04 -> C05 -> C06  (5 waves).
```

**Parallelism justification.**

- **W0 is genuinely parallel.** C01 and C02 create two *different* new files
  (`ctr_border_space_test.rs`, `ctr_border_upstream_anchor_test.rs`) and touch no
  production code → no write conflict.
- **C04 and C05 are NOT parallel** *(corrected at plan-check pass 1, MAJOR-1)*.
  Plan v1 declared them disjoint. They are not: **C05's named mutation edits
  `crates/cb-train/src/tree.rs`** — the same production file C04 edits — and it
  **names `ctr_bin_border_to_value_space`, which does not exist until C04's
  extraction lands.** Run in parallel, C04's M1/M2 reverts and C05's `+ 0.1`
  revert can clobber each other; run in the other order, C05's mutation text does
  not compile. With `git checkout --` forbidden (§1), recovery would be
  hand-editing under uncertainty, and C06's diff gate would attribute the orphan
  hunk to nobody.
  **Resolution: serialize `C04 → C05`** (option (a) of the check's required
  revision — one extra wave, no ambiguity).
- **`crates/cb-train/src/tree.rs` is an EXCLUSIVE resource** held by C03, then
  C04, then C05, in that order. **C04 and C05 each begin with a pre-flight
  check:** `git diff crates/cb-train/src/tree.rs` must show **only** the hunks
  owned by the already-completed predecessor tasks (C03 for C04; C03 + C04 for
  C05). **A stray hunk from a sibling task is a STOP-AND-REPORT condition** — do
  not "clean it up", report it.
- C06 writes no code at all.

---

# WAVE W0 — RED

## C01 — PRIMARY RED: training and apply must agree for a document on the border

- **Specs:** SPEC-CTRB-02 (primary), SPEC-CTRB-03 (positive integral assertion)
- **Order:** 1 · **Depends on:** none · **Status:** pending
- **Files:** Create `crates/cb-train/tests/ctr_border_space_test.rs`
- **Touches production code:** NO

### Objective

A single focused test that **fails today for the right reason** and localizes the
defect to `CtrSplitSpec.border` — not to "predictions differ". After C03 it is
this plan's permanent, fixture-distribution-independent regression gate (D8: the
11 oracles cannot serve that role).

### Why an end-to-end parity assertion is not acceptable here

`ctr_counter_simple_oracle_test` fails with `max |diff| = 2.687e-1` — a number
that is consistent with a dozen unrelated defects. It is the SECONDARY gate
(C03). The PRIMARY Red must name the disagreeing document, its bin, the chosen
bin-space border, the persisted value-space border, and both booleans.

### Contract under test (state it in the module doc comment)

Let `v` be a document's CTR value in **apply space** and `b` the chosen bin-space
border index. Then, for every `v` the quantizer can produce:

```text
  TRAINING decides:  trunc(v) > b            (E2, E7)
  APPLY    decides:  v > CtrSplitSpec.border  (E3)
  ⇒ REQUIRED:        CtrSplitSpec.border ∈ [b, b+1)  and specifically (b+1) − 2⁻²⁰
```

`bin = trunc(v)` is not an assumption: it is `materialize_ctr_feature`'s own
step 4 (E7), and the test **re-derives `v` from the column's own `ctr_value`**
and asserts `v.trunc() as u32 == bins[i]` before using it.

### Test construction (exact)

File header:

```rust
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
```

Imports available to a cb-train integration test (§3.0): `cb_train::{
materialize_ctr_feature, greedy_tensor_search_oblivious_with_ctr, calc_normalization,
ctr_border_count_default, ECtrType, FeatureMatrix, LevelKind, TProjection }`,
`cb_compute::EScoreFunction`. **All of these are already re-exported from
`cb_train`'s crate root** — `calc_normalization` and `materialize_ctr_feature` at
`crates/cb-train/src/lib.rs:47-48`, `greedy_tensor_search_oblivious_with_ctr` and
`LevelKind` at `:109-111`, `CtrSplitSpec`/`FeatureMatrix` at `:110`
`[VERIFIED: LOCAL grep of lib.rs]`. If anything else is needed it must be reached
via its existing public path — **never** by adding a `pub use` to production code
in this task.

**Test 1 — `train_and_apply_agree_for_a_document_whose_bin_equals_the_border`
(the primary Red).**

1. Build a small deterministic categorical corpus in-test (no fixture): e.g.
   `cat_columns = vec![vec![plain string categories]]` with ~12–24 rows over
   4–8 distinct category values, `target_class` a 0/1 vector, identity
   permutation `(0..n as i32)`, `ctr_border_count = ctr_border_count_default()`
   (`= 15`, E21), `ctr_type = ECtrType::Borders`, `target_border_idx = 0`.

   > **PIN — NON-TUNABLE (MINOR-1, plan-check pass 1):
   > `prior_num = 0.5`, `prior_denom = 1.0`.**
   > Step 8 invites corpus tuning; **that permission covers rows and category
   > values ONLY.** The prior is fixed because step 3's reconstruction is
   > bit-exact *only* at `norm == 1.0`: `calc_ctr_online_bin` computes
   > `(ctr + shift) / norm * border_count` (`calc_ctr.rs:89-101`) while step 3
   > computes `(ctr + shift) * (border_count / norm)`, and `(a/n)*b != a*(b/n)`
   > in general. `calc_normalization(0.5) = (0.0, 1.0)` (E22), so both reduce to
   > `ctr * border_count` and the premise holds exactly. At any other prior the
   > premise can fail by 1 ulp at an integer boundary and the task would be
   > misdiagnosed as "the bin↔value relation does not hold".
2. `let col = materialize_ctr_feature(...)?;`
3. **Re-derive apply-space values from the column itself** (E7):
   ```text
   let (shift, norm) = calc_normalization(prior_num / prior_denom);
   assert_eq!((shift, norm), (0.0, 1.0),
       "this reconstruction is bit-exact ONLY at norm == 1.0 ((a/n)*b != a*(b/n)); \
        the prior is pinned at 0.5/1.0 and is NOT tunable — see C01 step 1");
   let scale = ctr_border_count as f64 / norm;              // == Scale (E7)
   v[i] = (col.ctr_value[i] + shift) * scale;               // == calc_inference tail
   ```
   Then assert, for every `i`: `v[i].trunc() as u32 == col.bins[i]`
   — message: *"the bin↔value relation `bin = trunc(v)` (ctr_feature.rs:263-292)
   does not hold; this test's premise is invalid — investigate before touching
   any border"*. **Both assertions pass today and must keep passing**; they are
   what make the rest non-tautological.
4. Grow depth-1 with an **uninformative** float matrix (copy the shape of
   `ctr_split_scoring_test::uninformative_float_matrix`: one feature, one border
   `0.5`, constant values) so the CTR candidate wins:
   `greedy_tensor_search_oblivious_with_ctr(&matrix, &[col.clone()], ctr_border_count, &der1, &weight, 3.0, 1, n, 0, 0.0, EScoreFunction::Cosine, &[])`.
5. Assert `grown.ctr_splits.len() == 1` and `grown.splits.is_empty()`
   — anti-vacuity: *"no CTR split won; this gate would be vacuous"*.
6. Extract the bin-space border:
   ```text
   let LevelKind::Ctr { border: bin_border, .. } = grown.level_kinds[0] else { panic!(…) };
   ```
   **SPEC-CTRB-03 positive assertion — THE ONLY POSSIBLE DETECTOR (§0, D3a):**
   - `assert_eq!(bin_border, bin_border.trunc())` — message:
     *"LevelKind::Ctr.border must stay in BIN space (integral). Its consumer
     `assign_leaf_of_averaging` (boosting.rs:1938) compares it against the u32
     `col.bins`, so converting it here is arithmetically a NO-OP and NO oracle
     can catch it — this assertion is the only detector. Units contract, see
     PLAN §0."*
   - `assert!(bin_border >= 0.0 && bin_border < ctr_border_count as f64)`.
7. Let `value_border = grown.ctr_splits[0].border;`
8. **Anti-vacuity guard (mandatory).** Collect
   `on_border = { i : col.bins[i] as f64 == bin_border && v[i] > bin_border }`.
   `assert!(!on_border.is_empty(), "no document lands strictly inside the chosen
   border's bin — this corpus cannot exercise SPEC-CTRB-02; widen the corpus, do
   NOT weaken the assertion")`.
   *(If the chosen corpus trips this, the executor tunes the corpus — more rows /
   more category values — and records the final corpus in the completion evidence.
   It must NOT relax the guard or switch to a fixture.)*
9. **THE AGREEMENT ASSERTION**, over **every** document `i` (not just the
   on-border ones):
   ```text
   let training = f64::from(col.bins[i]) > bin_border;   // tree.rs:2600 / boosting.rs:1938
   let apply    = v[i] > value_border;                   // apply.rs:189
   assert_eq!(training, apply,
       "doc {i}: bin {} vs bin-space border {bin_border}: training={training}, \
        but apply value {} vs persisted border {value_border}: apply={apply}. \
        The persisted CtrSplitSpec.border is in BIN space; every consumer reads it \
        as a VALUE-space threshold (SPEC-CTRB-02).", col.bins[i], v[i]);
   ```

**Test 2 — `persisted_border_brackets_the_chosen_bin`.**
Same setup; assert
`bin_border < value_border && value_border < bin_border + 1.0`
with the message *"the persisted border must lie strictly inside `(b, b+1)` so
that `v > border` reproduces `trunc(v) > b`"*. **Red today** (`value_border ==
bin_border`). This is the cheapest possible statement of the whole defect.

**Test 3 — `every_candidate_border_is_a_valid_bin_index`** (characterization,
green today and after): assert the winning `bin_border` equals some
`border_idx as f64` for `border_idx in 0..ctr_border_count` (E1). Guards D4.

### TDD sequence

1. **Red** — write the file. Run
   `cargo test -p cb-train --test ctr_border_space_test`.
   **Expected:** Test 1 and Test 2 **FAIL**; Test 3 and the step-3 bin↔value
   assertion **PASS**. Record the verbatim failure text of Tests 1 and 2,
   including the concrete `doc`, `bin`, `bin_border`, `v[i]`, `value_border`.
   **If Test 1 passes on first write, STOP** — the corpus is not exercising the
   defect (step 8's guard should have caught it); fix the corpus, do not proceed.
2. **Green** — none in this task. C01 is Red-only by construction; C03 turns it
   green.
3. **Refactor** — none (new file, no duplication yet).
4. **Verify** —
   - `cargo clippy --workspace --all-targets` (the new target must be lint-clean).
   - `git diff --stat` shows **only** the one new untracked file; **zero** diff on
     any of the 11 CTR oracles.

### Completion criteria

- [ ] Tests 1 and 2 fail; the failure names the document, both borders and both booleans.
- [ ] The bin↔value premise assertion (step 3) passes.
- [ ] The SPEC-CTRB-03 integral assertion (step 6) passes.
- [ ] The anti-vacuity guard (step 8) passes with a non-empty `on_border` set.
- [ ] No production file is modified.
- [ ] `cargo clippy --workspace --all-targets` clean.

### Risks and guardrails

- **R:** the corpus produces a winning border with no on-border document →
  vacuous gate. **M:** step 8's guard fails loudly; tune the corpus, never the guard.
- **R:** temptation to reach into private `passes_ctr_aware` / `passes_ctr_split`.
  **M:** forbidden — both decisions are reproduced from their verbatim source
  expressions (cited inline), and the bin↔value premise is *proved* against the
  real column in step 3 rather than assumed.

---

## C02 — RED: the persisted border must match upstream's value-space convention

- **Specs:** SPEC-CTRB-01
- **Order:** 2 · **Depends on:** none · **Status:** pending
- **Files:** Create `crates/cb-train/tests/ctr_border_upstream_anchor_test.rs`
- **Touches production code:** NO

### Objective

Anchor the persisted border to **upstream catboost 1.2.10's own committed
values**, not to a formula this repository invented. C01 proves *internal
consistency*; C02 proves *upstream agreement*. Both are needed: a border of
`b + 0.5` would satisfy C01 and still be wrong on the wire.

### Test construction (exact)

Harness to copy: `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs`
(`fixture()` path helper `:55-66`, `load_cat_columns()` `:68`,
`tensor_ctr_params()` `:86`, the `train_cat(...)` call `:232`)
`[VERIFIED: LOCAL]`. Upstream borders are read with raw `serde_json` because
`cb_oracle` has no CTR-border accessor (E18).

```text
fn upstream_ctr_borders(scenario: &str) -> Vec<f64>
    // read fixtures/<scenario>/model.json as serde_json::Value
    // → ["features_info"]["ctrs"] as array
    // → flat_map each entry ["borders"] as array of f64
    // sorted by f64::total_cmp
```

**Test 1 — `trained_ctr_borders_follow_the_upstream_value_space_convention`.**
For each scenario in `["tensor_ctr_e2e", "ctr_counter_simple"]`:
1. Train through production `train_cat` with that fixture's pinned params
   (copy `tensor_ctr_params()` / `counter_params()` field-for-field — Pitfall-6:
   every field explicit).
2. Collect every `trained.oblivious_trees[*].ctr_splits[*].border`.
3. `assert!(!borders.is_empty(), "…no CTR split at all — gate would be vacuous")`.
4. For each border `x`:
   - **integrality of `x + 2⁻²⁰`:**
     ```text
     let k = x + f64::from(f32::powi(2.0, -20));
     assert_eq!(k, k.round(), "border {x} is not of upstream's form (b+1) − 2^-20 …");
     ```
   - **f32 fixed point:** `assert_eq!(f64::from(x as f32), x, "…the .cbm codec
     narrows Borders to f32 (cbm.rs:437); a non-fixed-point border shifts on
     save/load")`.
   - **strictly positive:** `assert!(x > 0.0)`.

   **Red today:** the trainer persists integers (`2.0`, `7.0`, `8.0`, `10.0`), so
   `k = 2.00000095…` and `k.round() = 2.0` → first assertion fails.

**Test 2 — `trained_ctr_borders_are_members_of_the_upstream_border_set`.**
For each scenario, assert every trained border is bit-present in
`upstream_ctr_borders(scenario)`:
```text
assert!(upstream.iter().any(|u| u.to_bits() == x.to_bits()),
    "trained CTR border {x} is not among upstream's {upstream:?} for {scenario}. \
     If the CONVENTION assertions above pass and this one fails, the trainer chose \
     a DIFFERENT bin threshold than catboost 1.2.10 — that is a structural parity \
     finding, NOT a test to weaken. STOP AND REPORT.");
```
Expected upstream sets (E4): `tensor_ctr_e2e` →
`[2.9999990463256836, 7.999999046325684]`; `ctr_counter_simple` →
`[8.999999046325684, 10.999999046325684]`.

> **Why membership is expected to hold.** `tensor_ctr_e2e_oracle_test` already
> passes at ≤1e-5 across all 5 trees, which requires the same tree structure as
> upstream, hence the same chosen bin thresholds. For `ctr_counter_simple` the
> same becomes true once C03 lands. If Test 2 fails **after** C03 while Test 1
> passes, it is a genuine finding — record it and stop (see §7 OQ-B).

### TDD sequence

1. **Red** — `cargo test -p cb-train --test ctr_border_upstream_anchor_test`.
   **Expected:** Test 1 FAILS for both scenarios naming the integer borders; Test 2
   FAILS. Record verbatim, including the exact border multisets observed.
2. **Green** — none here (C03).
3. **Refactor** — none.
4. **Verify** — `cargo clippy --workspace --all-targets`; `git diff --stat` shows
   only the new untracked file; zero diff on the 11 oracles.

### Completion criteria

- [ ] Tests 1 and 2 fail, naming the actual (integer) borders per scenario.
- [ ] The anti-vacuity `!borders.is_empty()` guard passes.
- [ ] No fixture regenerated; `gen_fixtures.py` not invoked.
- [ ] No production file modified.

### Risks and guardrails

- **R:** `ctr_counter_simple` may train slowly / need the exact `counter_params()`.
  **M:** copy that function verbatim from
  `crates/cb-train/tests/ctr_counter_simple_oracle_test.rs:59-100` rather than
  re-deriving it.
- **R:** upstream `model.json` shape drift. **M:** the accessor asserts
  `features_info.ctrs` exists and is a non-empty array before reading.

---

# WAVE W1 — GREEN

## C03 — GREEN: convert the persisted border to value space (ONE expression)

- **Specs:** SPEC-CTRB-01, SPEC-CTRB-02, SPEC-CTRB-03
- **Order:** 3 · **Depends on:** C01, C02 · **Status:** pending
- **Files:** Modify `crates/cb-train/src/tree.rs` (one expression + comments)
- **Touches production code:** YES — this is the only task that changes behavior

### ⚠ Read PLAN §0 before opening the file. Two adjacent `border: *border` sites.

**No runtime gate can catch a wrong-site edit here (§0, E20).** The one-expression
diff gate below and C01's integrality assertion are the entire safety net.

### The change — exactly this, nowhere else

At `crates/cb-train/src/tree.rs:3297`, inside the `ctr_splits.push(CtrSplitSpec {`
literal:

```text
-                    border: *border,
+                    // VALUE SPACE (SPEC-CTRB-01). The search chose the integer
+                    // bin index `*border`; every consumer of the PERSISTED border
+                    // — `cb_model::passes_ctr_split` (apply.rs:189), the `.cbm`
+                    // codec (cbm.rs:437/601) and upstream CatBoost — compares it
+                    // against the SCALED CTR value, where the bin is `trunc(v)`.
+                    // Upstream's convention is `(bin + 1) - 2^-20`; every committed
+                    // CTR fixture border matches it bit-for-bit.
+                    //
+                    // Computed in f32 and widened once because the `.cbm` codec
+                    // narrows `Borders` to f32 on save (cbm.rs:437) and widens via
+                    // `f64::from` on load (cbm.rs:601) — the value must therefore
+                    // be an f32 fixed point, which the f32 form is and the naive
+                    // f64 form is not.
+                    //
+                    // DOMAIN: `bin <= 15`. Above that the f32 subtraction rounds
+                    // to exactly `bin + 1` and the strict-interval property is
+                    // lost. Guaranteed reachable-safe by
+                    // `ctr_border_count_default() == 15` (boosting.rs:529-531),
+                    // consumed at boosting.rs:3238 — the single, non-configurable
+                    // source of `ctr_border_count`. See the `b = 16`
+                    // characterization test in tree_test.rs.
+                    border: f64::from((*border as f32 + 1.0) - f32::powi(2.0, -20)),
```

And at `crates/cb-train/src/tree.rs:3287-3290`, **add a comment only** (the
expression is byte-unchanged).

> **The comment text below is corrected at plan-check pass 1.** Plan v1 mandated
> writing *"Converting it here silently corrupts leaf-value assignment"* into
> this file. **That is FALSE (§0, E20) and must NOT be shipped in the source.**
> Use exactly the text below.

```text
                 level_kinds.push(LevelKind::Ctr {
                     ctr_idx: ctr_splits.len(),
+                    // BIN SPACE — deliberately NOT the value-space conversion
+                    // applied to the persisted `CtrSplitSpec.border` below.
+                    //
+                    // This border is training-only: its sole consumer is
+                    // `assign_leaf_of_averaging` (boosting.rs:1926-1938), which
+                    // tests `f64::from(bin) > *border` against
+                    // `CtrFeatureColumn::bins` (`Vec<u32>`). Because that operand
+                    // is an INTEGER, `bin > b` and `bin > (b+1) - 2^-20` are
+                    // arithmetically EQUIVALENT here, so converting this line
+                    // would change no behavior — it is kept in bin space as a
+                    // UNITS contract, matching the unit of the value it is
+                    // compared against. Mixing units across these two adjacent,
+                    // identically-typed fields is how BUG-CTRB arose.
+                    //
+                    // Consequence: no runtime test can detect a conversion here.
+                    // The guard is the explicit integrality assertion in
+                    // tests/ctr_border_space_test.rs and tree_test.rs
+                    // (SPEC-CTRB-03).
                     border: *border,
                 });
```

**Nothing else changes.** In particular: no change to `tree.rs:2597-2600`
(`passes_ctr_aware`), `tree.rs:3162-3164` (candidate enumeration),
`boosting.rs:2015` (`border: 0.0`, D5/E9), or anything in `cb-model`.

### TDD sequence

1. **Red** — already established by C01 and C02. Re-run both to confirm they are
   still red at the start of this task:
   `cargo test -p cb-train --test ctr_border_space_test --test ctr_border_upstream_anchor_test`
2. **Green** — apply the one-expression change above. Then, in order:
   - `cargo test -p cb-train --test ctr_border_space_test` → **all green**
   - `cargo test -p cb-train --test ctr_border_upstream_anchor_test` → **all green**
   - `cargo test -p cb-train --test ctr_counter_simple_oracle_test` → **4/4 green**
     (the SECONDARY/integration Red; `max |diff|` must drop from `2.687e-1` to
     ≤1e-5)
3. **Refactor** — **none in this task.** Extraction of the named helper is C04, so
   that the behavior change and the structural change are two reviewable diffs.
4. **Verify** — the **mandatory regression scope for any `tree.rs`/`boosting.rs`
   change**, run in full (§3.2):
   - the 11 CTR oracles block (9 cb-train + 2 cb-model) — **all green**
   - the 3 one-hot targets — **all green**
   - `cargo test -p cb-model --test ctr_nonmean_byte_identity_test` — **green**
     (E15 predicts unaffected; **confirm by running, do not assert from theory**)
   - `cargo test -p cb-model --test cbm_oracle_test --test json_oracle_test --test float_only_byte_identity_test`
   - `cargo test -p cb-train --no-fail-fast` and `cargo test -p cb-model --no-fail-fast`
     — the **only** permitted failure is the pre-existing
     `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors`
     (ctr-type `PLAN.md` §3 accepted baseline)
   - `cargo clippy --workspace --all-targets`
   - **Diff gate:** `git diff --stat` must show `crates/cb-train/src/tree.rs` and
     **nothing else**; `git diff crates/cb-train/src/tree.rs` must contain exactly
     **one** changed expression plus comment-only hunks.

### Completion criteria

- [ ] `git diff crates/cb-train/src/tree.rs` changes exactly one expression (line 3297).
- [ ] `LevelKind::Ctr`'s `border: *border` at line 3289 is **byte-identical**.
- [ ] C01 green (including the integral `LevelKind` assertion).
- [ ] C02 green (convention + upstream membership).
- [ ] `ctr_counter_simple_oracle_test` 4/4, `max |diff|` ≤ 1e-5.
- [ ] 11 CTR oracles green, **zero diff** on all eleven files.
- [ ] 3 one-hot targets green.
- [ ] `ctr_nonmean_byte_identity_test` green (**run, not assumed**).
- [ ] `cb-train` + `cb-model` sweeps show no new failure beyond the recorded
      pre-existing monotone one.
- [ ] `cargo clippy --workspace --all-targets` clean.

### Risks and guardrails

- **R (principal, per SPEC §9 R1'):** the fix is ONE LINE, so its tests are at
  high risk of being tautological. **M:** C01 reconstructs both decisions from
  their verbatim source expressions rather than calling the (private)
  `passes_ctr_aware` / `passes_ctr_split`, so an executor who "fixed"
  `crates/cb-model/src/apply.rs:189` instead would **not** turn C01 green; and
  C02 anchors the value against upstream's own committed borders.
- **R:** converting line 3289 as well. **M:** the one-expression diff gate and
  C01's integrality assertion — **and nothing else**, because no runtime gate can
  detect it (§0, E20, D3a). Do **not** expect an oracle to catch this.
- **R:** the `as f32` cast trips a workspace lint. **M:** it will not — the
  workspace denies only `unwrap_used`, `expect_used`, `panic`, `indexing_slicing`;
  no cast lint is enabled (E24). `f64 → f32` `as` casts already exist in
  production (`cbm.rs:431-437`). Recorded so the executor does not pre-emptively
  add an `#[allow]`.
- **R:** a reviewer later "simplifies" the f32 arithmetic to f64. **M:** the
  production comment states the codec fixed-point reason, and C04's Test C fails
  at `b = 16` if they do (E6).

---

# WAVE W2 — REFACTOR (exclusive holder of `crates/cb-train/src/tree.rs`)

## C04 — REFACTOR: name the conversion, and pin it against upstream

- **Specs:** SPEC-CTRB-01, SPEC-CTRB-03
- **Order:** 4 · **Depends on:** C03 · **Status:** pending
- **Files:** Modify `crates/cb-train/src/tree.rs`; Modify
  `crates/cb-train/src/tree_test.rs`
- **Touches production code:** YES — behavior-preserving extraction only
- **Exclusive resource:** `crates/cb-train/src/tree.rs` (held by C04 for the
  duration; C05 takes it next)

### Pre-flight (mandatory)

`git diff crates/cb-train/src/tree.rs` must show **only C03's hunks** (one changed
expression + comment-only hunks). **A stray hunk is a STOP-AND-REPORT condition** —
report it, do not clean it up.

### Objective

Turn the inline conversion into a **named, documented function** so the two
spaces can never again be confused by proximity, and give SPEC-CTRB-01 a direct
unit test over its **reachable domain**, plus a boundary characterization that
trips loudly if that domain is ever widened (E6, E21).

### The extraction

Add to `crates/cb-train/src/tree.rs`, immediately **above**
`greedy_tensor_search_oblivious_with_ctr` (private — no new public API):

```text
/// Convert a chosen CTR **bin index** into the **value-space** threshold the
/// model persists (SPEC-CTRB-01).
///
/// The structure search enumerates candidates as integer bin indices and tests
/// `bin > b` (`passes_ctr_aware`, this file). Every consumer of the PERSISTED
/// border instead tests `ctr_value > border` against the SCALED CTR value
/// (`cb_model::passes_ctr_split`, apply.rs:189). Upstream CatBoost bridges the
/// two with `(b + 1) - 2^-20`, so that `ctr_value > border` reproduces
/// `trunc(ctr_value) > b` for every value the quantizer can produce.
///
/// Computed in `f32` and widened once: the `.cbm` codec narrows `Borders` to
/// `f32` on save (cbm.rs:437) and widens via `f64::from` on load (cbm.rs:601), so
/// the value must be an `f32` fixed point. Computing in `f64` and narrowing on
/// save would not be. That codec requirement — NOT any claim of general
/// correctness — is why this is f32.
///
/// # DOMAIN: `bin_border <= 15`
///
/// Two properties must hold together:
///   1. strict interval:  `bin_border < result < bin_border + 1`
///   2. f32 fixed point:  `f64::from(result as f32) == result`
///
/// They hold together ONLY while `bin_border + 1` has an f32 ulp `<= 2^-20`,
/// i.e. `bin_border <= 15`. At `bin_border == 16` the f32 subtraction rounds to
/// exactly `17.0` and property 1 is LOST; the f64 formulation would instead lose
/// property 2. NEITHER form is correct above 15 — the correct value there is
/// simply not known from any in-repo artifact.
///
/// Reachability guarantee: `ctr_border_count` is not configurable —
/// `ctr_border_count_default()` returns 15 (boosting.rs:529-531) and is consumed
/// at exactly one site, boosting.rs:3238 — so candidates are
/// `border_idx in 0..15` and `bin_border <= 14`.
///
/// STOP CONDITION: if `ctr_border_count` ever becomes configurable or exceeds 16,
/// this contract breaks and BUG-CTRB reappears at the top of the range. The
/// `b = 16` characterization test in `tree_test.rs` pins that boundary.
///
/// This function is for the PERSISTED border ONLY. `LevelKind::Ctr.border` stays
/// in BIN space — see the comment at its construction site.
fn ctr_bin_border_to_value_space(bin_border: f64) -> f64 {
    f64::from((bin_border as f32 + 1.0) - f32::powi(2.0, -20))
}
```

Replace the line-3297 expression with
`border: ctr_bin_border_to_value_space(*border),`, keeping the `*** Do NOT apply
this to LevelKind::Ctr.border ***` warning comment in place.

### Test (source/test separation — sibling file, existing mount)

The mount is `crates/cb-train/src/tree.rs:92-94` → **`mod general;`**, so the
focused command is `cargo test -p cb-train --lib tree::general`
`[VERIFIED: LOCAL tree.rs:88-110]`.

Append to `crates/cb-train/src/tree_test.rs`. That file has **no** file-level
`#![allow(...)]` — it inherits the crate-level `#![cfg_attr(test, allow(...))]`.
**Do not add one.** Its existing import style is
`use crate::tree::{combination_ctr_eligible, select_best_candidate, Candidate};`
+ `use crate::TProjection;` — extend those `use` lists rather than adding
`use super::*` `[VERIFIED: LOCAL tree_test.rs:1-7]`.

**Test A — `ctr_bin_border_matches_every_committed_upstream_fixture_border`.**
Hard-code the eight upstream values (E4) and assert bit-equality:

| `b` | expected `ctr_bin_border_to_value_space(b as f64)` | source fixture |
|---|---|---|
| 2 | `2.9999990463256836` | `tensor_ctr_e2e` |
| 3 | `3.9999990463256836` | `fstr_ctr` |
| 6 | `6.999999046325684` | `fstr_ctr` |
| 7 | `7.999999046325684` | `tensor_ctr_e2e` |
| 8 | `8.999999046325684` | `ctr_counter_simple` |
| 10 | `10.999999046325684` | `ctr_counter_simple` |
| 11 | `11.999999046325684` | `fstr_ctr` |

Compare with `.to_bits()` equality and a message naming the fixture.

**Test B — `ctr_bin_border_brackets_its_bin_over_the_reachable_domain`.**
For `b in 0..ctr_border_count_default()` — i.e. **`b ∈ 0..=14`, the REACHABLE
domain and nothing more** (E21):
`assert!(x > b as f64 && x < (b + 1) as f64)` — Invariant 1, the property that
makes SPEC-CTRB-02 hold. A comment must state that the loop bound is
`ctr_border_count_default()` **precisely because that is the reachability
guarantee**, and that the property is **known false** above 15 (Test E).

**Test C — `ctr_bin_border_is_an_f32_fixed_point_over_the_reachable_domain`.**
For `b in 0..ctr_border_count_default()` — **`0..=14` ONLY; do NOT sweep
`0..=254`** *(corrected at plan-check pass 1, MAJOR-3: a `0..=254` sweep passes
and thereby implies a generality the formula does not have)*:
`assert_eq!(f64::from(x as f32), x)` with the message
*"the .cbm codec narrows Borders to f32 (cbm.rs:437) and widens via f64::from
(cbm.rs:601); a non-fixed-point border shifts on save/load (SPEC-CTRB-01
Invariant 2, SPEC-CTRB-05). This asserts the f32 fixed point over the reachable
domain ONLY — it does NOT assert the formula is general."*

> **Where D2's falsification went.** Because Tests B and C now stop at `b = 14`,
> neither discriminates f32 from f64 — the two formulations agree for `b <= 15`
> (E6). D2's discriminator therefore moves to **Test E**, the only point where
> they differ. **Do NOT re-widen B or C to recover a discriminator.**

**Test E (NEW — `b = 16` BOUNDARY CHARACTERIZATION, required by MAJOR-3).**
`#[test] fn ctr_bin_border_collapses_to_b_plus_one_at_sixteen_and_is_unreachable()`:

```text
let x = ctr_bin_border_to_value_space(16.0);
assert_eq!(x, 17.0,
    "CHARACTERIZATION, not a desired property: at b = 16 the f32 subtraction \
     rounds to exactly b+1, so `v > border` becomes `v >= next_after(b+1)` and \
     the strict-interval invariant (SPEC-CTRB-01 Invariant 1) is LOST — BUG-CTRB \
     would reappear at the top of the range. This is UNREACHABLE today: \
     `ctr_border_count` is not configurable — `ctr_border_count_default()` returns \
     15 (boosting.rs:529-531) and is consumed at exactly one site, \
     boosting.rs:3238 — so border_idx <= 14. \
     IF THIS TEST EVER NEEDS TO CHANGE, ctr_border_count has become configurable \
     or exceeds 16: STOP AND REPORT. Do not adjust this assertion; re-derive the \
     border formula for the widened domain first. The naive f64 form is not an \
     alternative — it yields 16.999999046325684, which is NOT an f32 fixed point \
     and would shift on the .cbm round trip.");

// The reachability guard itself, asserted rather than merely commented:
assert_eq!(cb_train::ctr_border_count_default(), 15,
    "the b = 16 case is unreachable ONLY while this is 15 (boosting.rs:529-531)");
```

Test E is also **D2's discriminator**: an f64 implementation
(`(b + 1.0) - 2f64.powi(-20)`) yields `16.999999046325684 != 17.0`, so Test E
fails under mutation M1 `[VERIFIED: E6]`.

> **Honesty note the test comment must carry:** the value `17.0` is what *this*
> formula produces, not what upstream produces. Upstream's exact epsilon
> expression above `b = 15` is `[UNVERIFIED]` — an alternative consistent with
> every committed fixture (`(b+1) − 1e-6f`) rounds the **opposite** way at
> `b = 16`. That is precisely why this is a characterization test with a STOP
> condition and not a correctness assertion.

**Test D — `persisted_and_level_kind_borders_are_in_different_spaces`.**
Grow the same depth-1 CTR tree as C01 (or reuse a minimal hand-built
`CtrFeatureColumn` — `tree_test.rs` can see private items) and assert, in ONE
test, both halves of §0:
- `level_kinds[0]`'s border `== bin_border.trunc()` (integral, bin space), and
- `ctr_splits[0].border == ctr_bin_border_to_value_space(bin_border)`.

Message: *"§0 hazard: the two adjacent `border:` sites must remain in DIFFERENT
spaces."*

### TDD sequence

1. **Red** — run the pre-flight diff check, then write Tests A–E **before** the
   extraction. They do not compile (`ctr_bin_border_to_value_space` does not
   exist). Record the compile error as the Red — the function's absence is the
   missing behavior.
   *(Test D's `ctr_splits[0].border` half would pass post-C03 even inline; that is
   expected — D is a guard, covered by M2 below.)*
2. **Green** — add the function, replace the call site. Run
   `cargo test -p cb-train --lib tree::general` → A, B, C, D, E green.
3. **Refactor** — verify no duplicated epsilon literal remains anywhere in
   `tree.rs`; `grep -n "powi(2.0, -20)\|2\^-20" crates/cb-train/src/tree.rs`
   must show the expression **once**, inside the helper (doc-comment mentions
   excepted).
4. **Verify** + **MUTATION CHECK (§3.1)** — two named mutations, run **in
   sequence**, each manually reverted before the next:

   **M1 — proves Test E / D2 is load-bearing.**
   Change the helper body to `(bin_border + 1.0) - 2.0_f64.powi(-20)` (compute in
   f64).
   **Expected outcomes — record ALL of them:**
   - **Test E FAILS**: `16.999999046325684 != 17.0`.
   - Tests A, B, C **still PASS** — the two formulations are identical for
     `b <= 15` (E5/E6). *This is expected and must be recorded, not treated as a
     weak guard.*
   - The 11 oracles and the E12 gate **stay GREEN** (no reachable `b >= 16`).
   Revert manually; confirm green.

   **M2 — proves the integrality assertions are load-bearing, AND empirically
   confirms E20.** *(Expectation corrected at plan-check pass 1, CRITICAL-1. Plan
   v1 demanded three failures including an E12-gate regression; that was
   **unsatisfiable**.)*
   Apply `ctr_bin_border_to_value_space(*border)` to the **`LevelKind::Ctr`** site
   at line 3289 as well.
   **Expected outcomes — EXACTLY these, all four recorded:**
   1. **Test D FAILS** on `assert_eq!(bin_border, bin_border.trunc())`.
   2. **C01's step-6 SPEC-CTRB-03 assertion FAILS** (run
      `cargo test -p cb-train --test ctr_border_space_test`).
   3. **`cargo test -p cb-train --test ctr_counter_simple_oracle_test` stays
      4/4 GREEN.**
   4. **The 11 CTR oracles and the 3 one-hot targets stay GREEN.**

   > Outcomes 3 and 4 are **REQUIRED results, not incidental**. They are the
   > empirical confirmation of E20 — that the conversion is arithmetically a
   > no-op on the `u32` operand — and therefore the reason the integrality
   > assertions have to exist at all. **A regression in 3 or 4 is a
   > STOP-AND-REPORT condition**: it would mean E20's reasoning is wrong and the
   > whole SPEC-CTRB-03 framing needs re-derivation.
   >
   > **Do NOT** strengthen a test, alter production code, or hunt for a further
   > failure in order to make an oracle regress under M2. Under §3.1 this task's
   > guard requirement is satisfied by outcomes 1 and 2 alone.

   Revert manually; confirm green.

   Finally re-run the full mandatory regression scope (11 CTR oracles + 3 one-hot
   + `ctr_counter_simple_oracle_test` + `cargo clippy --workspace --all-targets`)
   to confirm the extraction was behavior-preserving, and re-run the pre-flight
   diff so `tree.rs` carries only the C03 + C04 hunks before C05 takes the file.

### Completion criteria

- [ ] Pre-flight: `git diff crates/cb-train/src/tree.rs` showed **only C03's hunks**.
- [ ] `ctr_bin_border_to_value_space` exists, is private, carries the DOMAIN /
      reachability / STOP-CONDITION doc block, and is the **only** place the
      `2^-20` expression appears in `tree.rs`.
- [ ] Tests A, B, C, D, **E** green under `cargo test -p cb-train --lib tree::general`.
- [ ] Tests B and C sweep **`0..ctr_border_count_default()` only** — no `0..=254`
      sweep anywhere.
- [ ] **M1 recorded:** Test E fails (`16.999999046325684 != 17.0`); Tests A, B, C
      still pass; oracles stay green.
- [ ] **M2 recorded — all four outcomes:** (1) Test D fails, (2) C01's step-6
      assertion fails, (3) the E12 gate stays **4/4 GREEN**, (4) the 11 oracles +
      3 one-hot targets stay **GREEN**. Outcomes 3–4 green are **required**;
      a regression there is STOP-AND-REPORT.
- [ ] **No test was strengthened and no production code altered in order to make
      an oracle regress under M2.**
- [ ] Both mutations reverted **manually** (no `git checkout --` / `git stash` /
      `git clean`).
- [ ] 11 CTR oracles + 3 one-hot targets green; zero diff on the eleven.
- [ ] Post-flight: `git diff crates/cb-train/src/tree.rs` shows only C03 + C04
      hunks (no residual mutation) before C05 starts.
- [ ] `cargo clippy --workspace --all-targets` clean.

### Risks and guardrails

- **R:** `tree_test.rs` may need an explicit import. **M:** it uses
  `use crate::tree::{...}` + `use crate::TProjection;`; extend those lists, never
  add `use super::*` or a second `#![allow(...)]` (§3.0).
- **R (the CRITICAL-1 trap):** an executor treats M2's green oracles as "the
  mutation check failed" and escalates — strengthening tests or changing
  production code until something regresses, i.e. **introducing a real defect
  chasing a phantom one**. **M:** the four expected outcomes are enumerated
  above; outcomes 3–4 being green are stated as *required results*; and D3a
  forbids any step that expects an oracle regression from `:3289`.
- **R:** M1 leaves the helper in f64 form if the revert is sloppy. **M:** M1 is
  reverted and re-confirmed green before M2 is applied; the post-flight diff check
  is the second net.

---

# WAVE W3 — ROUND-TRIP (takes over `crates/cb-train/src/tree.rs` from C04)

## C05 — `.cbm` save→load→predict is bit-stable for a CTR model

- **Specs:** SPEC-CTRB-05
- **Order:** 5 · **Depends on:** **C04** (not merely C03) · **Status:** pending
- **Files:** Create `crates/cb-model/tests/ctr_border_cbm_roundtrip_test.rs`;
  **temporarily** mutates `crates/cb-train/src/tree.rs` during step 4
- **Touches production code:** NO permanently; YES temporarily (mutation only)
- **Exclusive resource:** `crates/cb-train/src/tree.rs` — taken over from C04

> **Why C05 depends on C04, not C03** *(corrected at plan-check pass 1,
> MAJOR-1)*: C05's named mutation edits `crates/cb-train/src/tree.rs` and
> **references `ctr_bin_border_to_value_space`, which does not exist until C04
> lands**. Plan v1 ran C04 ∥ C05 and declared them disjoint; they are not. See
> §5.

### Pre-flight (mandatory)

`git diff crates/cb-train/src/tree.rs` must show **only C03's and C04's hunks**
(one converted expression now routed through the helper, the helper definition,
and comment-only hunks) — i.e. **no residual mutation from C04's M1/M2**.
**A stray hunk is a STOP-AND-REPORT condition.**

### Objective

The new border is the first non-integer CTR border this trainer has ever written.
`.cbm` narrows `Borders` to **f32** on save and widens on load (E14). Prove the
round-trip is lossless *and* that the encode-side split→border lookup still
resolves.

### Why this is a GUARD, not a Red

It passes both before and after C03: an integer border is trivially an f32 fixed
point too. Its falsifiability therefore comes from the §3.1 mutation check, not
from an initial Red. State that in the module doc comment (precedent wording:
`boosting_test.rs:502-504`).

### Test construction (exact)

Harness precedents: `crates/cb-model/tests/non_symmetric_grower_roundtrip_oracle_test.rs`
(train → save → load → predict), `cbm_oracle_test.rs:128` (temp path).
`cb-model`'s dev-deps already include `cb-train`, `cb-backend`, `cb-oracle`,
`ndarray`, `ndarray-npy` `[VERIFIED: LOCAL crates/cb-model/Cargo.toml]`.

> **No precedent exists for this exact shape (E23).** `save_cbm` has never been
> called on a **trainer-produced** CTR model: every existing caller is float-only
> / non-symmetric, and the one CTR caller
> (`ctr_nonmean_byte_identity_test`) uses a **hand-built** `CtrData`, never
> `CtrData::from_baked`. Step 5 of the TDD sequence therefore carries a mandatory
> **fallback gate** so SPEC-CTRB-05 is never left ungated.

**Test 1 — `ctr_model_cbm_roundtrip_preserves_every_border_bitwise`.**
1. Train `ctr_counter_simple` (or `tensor_ctr_e2e`) via `train_cat`, copying the
   pinned params field-for-field from the corresponding oracle test.
2. `let model = CbModel::from_trained(&trained, borders).with_ctr_data(CtrData::from_baked(&baked));`
3. Collect the in-memory borders: every `ModelSplit::Ctr(c) => c.border` across
   `model.oblivious_trees`. `assert!(!in_memory.is_empty(), "no CTR split — the
   gate would be vacuous")`.
4. `save_cbm(&model, &tmp)?;` then `let reloaded = load_cbm(&tmp)?;`
   (`std::env::temp_dir()` + pid-tagged name; delete on success).
5. Collect the reloaded borders in the same order and assert
   **`to_bits()` equality, element by element**, with a message naming the index,
   both values and `cbm.rs:437`/`:601`.
6. **Positive lookup assertion (E14):** `save_cbm` must return `Ok`. A border
   absent from its identity's `borders` set would surface as
   `ModelError::Serialize("CTR split border missing from its identity")` from
   `ctr_split_to_global_index` (`cbm.rs:366-369`). Assert the `Ok` explicitly with
   that error name in the failure message, so a future regression in
   `build_ctr_features` is attributed correctly.

**Test 2 — `ctr_model_cbm_roundtrip_predictions_are_bit_equal`.**
Predict with `predict_raw_cat` on the fixture's cat columns using **both** the
original and the reloaded model; assert `a.to_bits() == b.to_bits()` per element,
plus the non-degeneracy guard `assert!(before.iter().any(|v| *v != before[0]))`
(copied from `ctr_counter_simple_oracle_test.rs:147-150`).

### TDD sequence

1. **Red** — unavailable by construction (guard test). Run the pre-flight diff
   check, write both tests, run
   `cargo test -p cb-model --test ctr_border_cbm_roundtrip_test`, **record that
   they pass on first write**, per §3.1 step 1.
2. **Green** — n/a.
3. **Refactor** — none.
4. **Verify** + **MUTATION CHECK (§3.1)** — named mutation:
   in `crates/cb-train/src/tree.rs`, change the persisted border expression at
   `:3297` to `ctr_bin_border_to_value_space(*border) + 0.1`
   *(chosen because `x + 0.1` is not representable in f32 at these magnitudes, so
   the save-side narrowing loses information — exactly what Test 1 exists to
   detect; adding a whole integer or dropping the epsilon would NOT falsify it,
   since integers are f32 fixed points too — state this reasoning in the
   completion evidence).*
   **Expected:** Test 1 fails with the bitwise-inequality message naming
   `cbm.rs:437`. Record verbatim. **Revert manually**, re-run green, and re-run
   the post-flight diff so `tree.rs` carries only C03 + C04 hunks.
   Then: `cargo test -p cb-model --test ctr_data_roundtrip_test --test cbm_oracle_test`
   and `cargo clippy --workspace --all-targets`.
5. **FALLBACK GATE — mandatory if `save_cbm` rejects the shape**
   *(added at plan-check pass 1, MINOR-4; E23: no precedent exists).*
   If `save_cbm` returns a typed `ModelError` on the `CtrData::from_baked` model:
   - **Do not stop with SPEC-CTRB-05 ungated.** Record the exact typed error and
     the model shape that produced it as a **separate finding** (a `.cbm` codec
     limitation, not a border-space defect), to be reported alongside this plan.
   - **Then gate SPEC-CTRB-05 in memory instead**, in the same new test file:
     - **Fallback 1:** for every persisted border `x` on the model,
       `assert_eq!(f64::from(x as f32), x)` — the f32 fixed point is the *entire*
       mechanism by which the round trip would have been lossless (SPEC-CTRB-01
       Invariant 2), so asserting it directly preserves the contract.
     - **Fallback 2:** exercise the encode-side lookup without the full save —
       call the `.cbm` path far enough to prove `build_ctr_features` +
       `ctr_split_to_global_index`'s **bit-exact** border lookup
       (`cbm.rs:366`) resolves; if that is not reachable from a test, assert the
       equivalent property directly: every tree `CtrSplit.border` is bit-present
       in the sorted, bit-deduped border set its identity would carry
       (`cbm.rs:310-311, :321-322`).
     - **Fallback 3:** keep Test 2's prediction check, comparing the model against
       itself round-tripped through `Model::from_trained` + `with_ctr_data`
       instead of through the codec, so a prediction-level bit-stability
       assertion still exists.
   - The **same `+ 0.1` mutation check applies to Fallback 1** and must be
     recorded: `x + 0.1` is not an f32 fixed point, so Fallback 1 fails under it.

### Completion criteria

- [ ] Pre-flight: `git diff crates/cb-train/src/tree.rs` showed **only C03 + C04
      hunks** (no residual C04 mutation).
- [ ] Both tests green; first-write pass recorded per §3.1 step 1.
- [ ] Mutation `+ 0.1` makes Test 1 (or Fallback 1) fail with the named bitwise
      message; recorded verbatim.
- [ ] Mutation reverted **manually**; green re-confirmed; post-flight diff clean.
- [ ] The rationale for choosing `+ 0.1` over `+ 1.0` / epsilon-drop is recorded.
- [ ] **Either** `save_cbm` returned `Ok` and the codec round trip is bit-equal,
      **or** the typed error was recorded as a separate finding **and** the
      step-5 fallback gate is in place. **SPEC-CTRB-05 is not left ungated
      either way.**
- [ ] `cargo clippy --workspace --all-targets` clean.

### Risks and guardrails

- **R (E23 — no precedent):** `.cbm` CTR save has never been exercised on a
  trainer-produced (`CtrData::from_baked`) model, and v1 groups OBLIVIOUS trees
  only (CTR splits on non-symmetric trees are rejected in the save loop,
  `cbm.rs:289-290`). **M:** both candidate fixtures train oblivious trees; and if
  `save_cbm` errors, step 5's fallback gate keeps SPEC-CTRB-05 covered while the
  codec limitation is reported separately — **"STOP" alone is not an acceptable
  outcome here.**
- **R:** C05's mutation collides with C04's. **M:** serialized (§5), plus the
  pre-flight and post-flight `git diff crates/cb-train/src/tree.rs` checks.
- **R:** temp-file leakage. **M:** pid-tagged name, removed on success (precedent
  `cbm_oracle_test.rs:128`).

---

# WAVE W4 — CLOSE-OUT

## C06 — Record the non-changes, and confirm the blast radius by RUNNING it

- **Specs:** SPEC-CTRB-03 (**decision record only** — see the note below),
  SPEC-CTRB-04

> **SPEC-CTRB-03 note (plan-check pass 1, CRITICAL-1).** C06's part-4 confirmation
> runs are a **non-regression** gate for the change as a whole. They are **NOT**
> acceptance for SPEC-CTRB-03: converting `:3289` is arithmetically a no-op
> (E20), so the oracles stay green either way and would prove nothing about it.
> SPEC-CTRB-03's acceptance is C01 step 6 + C04 Test D, and C06's contribution to
> it is the written decision record below.
- **Order:** 6 · **Depends on:** C03, C04, C05 · **Status:** pending
- **Files:** Modify `.planning/plans/ctr-split-border-space/SPEC.md`
  (§9 OQ1 correction + §5 SPEC-CTRB-04 decision record) — **planning Markdown only**
- **Touches production code:** NO

### Objective

Three things that are cheap now and expensive to reconstruct later:
1. **Record the decision NOT to convert `boosting.rs:2015`** with its evidence, so
   a future reader does not "finish the job".
2. **Determine and state** whether E03's `ctr_splits_for_tree_*` characterization
   tests must change. (Answer, pre-determined below — the task is to *confirm by
   running*, not to re-derive.)
3. **Confirm the "NOT affected" claims by executing them**, not by assertion.

### 1 — SPEC-CTRB-04: do NOT convert `ctr_splits_for_tree`'s `border: 0.0`

**Decision: leave `crates/cb-train/src/boosting.rs:2015` unchanged.**

Evidence (E9), to be written into SPEC.md §5 as a permanent decision record:

- `ctr_splits_for_tree` is invoked at exactly one site,
  `crates/cb-train/src/boosting.rs:5419-5429`, on the **`else` branch of
  `if has_ctr`**.
- `has_ctr = !materialized_ctr_features.is_empty()` (`boosting.rs:4657`).
- On that branch there are no materialized CTR features, hence no CTR candidates,
  hence `candidates.iter().map(...)` iterates zero times and the function returns
  an **empty `Vec`** — the `border: 0.0` literal is never constructed in
  production.
- The production comment already states it: *"`ctr_splits_for_tree` is retained
  for the no-CTR candidate path (it returns empty there) so the existing seam
  keeps compiling"* (`boosting.rs:5417-5418`).
- Converting it would produce `0.9999990463256836` for a spec that is never built,
  and would silently break the two E03 characterization tests (below) for no gain.

**Acceptance:** `cargo test -p cb-train --lib boosting::tests` green, unchanged.

### 2 — E03's characterization tests: **NO CHANGE REQUIRED**

`[VERIFIED: LOCAL crates/cb-train/src/boosting_test.rs:494-585, read in full]`

- `ctr_splits_for_tree_emits_one_spec_per_candidate_with_the_head_prior` (`:508`)
  asserts `spec.border == 0.0` (`:549`).
- `ctr_splits_for_tree_empty_priors_defaults_to_half` (`:560`) asserts priors only.
- `ctr_splits_for_tree_routes_type_and_prior_by_is_simple` (`:739`) and
  `ctr_splits_for_tree_defaults_are_byte_identical_to_the_e03_characterization`
  (`:792`) exercise the same function.

All four call `super::ctr_splits_for_tree(...)` **directly with a NON-empty
candidate list**, bypassing the `has_ctr` gate — a deliberate characterization
choice. Because D5 leaves `boosting.rs:2015` unchanged, `spec.border` is still
`0.0` and **every assertion in all four tests continues to hold verbatim.**

**State this explicitly in the completion evidence: E03's tests are UNAFFECTED and
must NOT be edited.** Any diff to `boosting_test.rs` in this plan is a
STOP-AND-REPORT condition.

Two further **test-only** `CtrSplitSpec` construction sites are likewise
unaffected because they hand-build literal borders (E11, E12):
`boosting_test.rs:337` (`border: 0.5`) and
`crates/cb-model/tests/mixed_kind_split_order_test.rs:47`
(`border: CTR_BORDER = 0.9`, chosen so the not-found→empty apply value `0.5`
deterministically fails the split).

### 3 — SPEC.md corrections

> **Status update (plan-check pass 1).** The coordinator has **already applied**
> the corrections listed below to `SPEC.md`. The executor's job here is
> **verification, not authorship**: re-read `SPEC.md` and confirm each item is
> present and consistent with the evidence ledger. Only the last item is
> outstanding.

**Already applied — confirm present:**

- **§2 non-goals + §4** — `LevelKind::Ctr.border` is described as a **units
  contract** whose conversion "would be arithmetically harmless". ✔
- **§5 SPEC-CTRB-01** — the **BOUNDED DOMAIN** table, the **Reachability**
  paragraph (`ctr_border_count_default()` = 15, `boosting.rs:529-531`, consumed at
  `:3238`) and the **STOP CONDITION**. ✔ Must match E6/E21 exactly.
- **§5 SPEC-CTRB-03** — retitled to a **UNITS contract**, carrying the explicit
  retraction of the "silently corrupts leaf assignment" claim, the integer-operand
  derivation, and *"Any plan step that expects an oracle to regress when `:3289`
  is converted is unsatisfiable and must not be written."* ✔ Must match E20/D3a.
- **§9 R1** — the old "principal risk" is **RETRACTED**; R1' (tautological
  one-line test) replaces it. ✔
- **§9 OQ1 — CLOSED**: four `CtrSplitSpec` construction sites, not three (the
  missing one is `crates/cb-model/tests/mixed_kind_split_order_test.rs:47`,
  test-only, literal `0.9`, unaffected); and **no** non-oblivious / region /
  leaf-wise / ordered / pairwise / perturbed grower persists CTR borders — all
  five other `GrownTree` producers set `ctr_splits: Vec::new()`
  (`tree.rs:774`, `:2089`, `:2238`, `:2541`, `:3708`). ✔ (E11, E12, E13)

**Still outstanding — write these:**

- **§10** — the E12 target has **4** `#[test]` functions, so the gate is **4/4**,
  not the "5/5" recorded there (§1) `[VERIFIED: LOCAL cargo test output]`.
- **§5 SPEC-CTRB-04** — add the decision record from part 1 above.
- **§1 (optional but recommended)** — add E8 as pre-existing, upstream-anchored
  confirmation of the `trunc(v) > b ⟺ v > (b+1)−2⁻²⁰` identity
  (`multi_permutation_fold_oracle_test.rs:104-108, 186-200`).

### 4 — Confirmation runs (RUN them; do not assert from theory)

| Claim | Command | Expected |
|---|---|---|
| E00 frozen byte baseline unaffected (E15) | `cargo test -p cb-model --test ctr_nonmean_byte_identity_test` | green |
| float-only bytes unaffected | `cargo test -p cb-model --test float_only_byte_identity_test` | green |
| E03 characterization unaffected | `cargo test -p cb-train --lib boosting::tests` | green, zero diff to `boosting_test.rs` |
| `ctr_split_scoring_test:122-123` survives (E16) | (in the 11-oracle block) | green, zero diff |
| 11 CTR oracles | the §3.2 block, in full | green, **zero diff on all eleven** |
| 3 one-hot targets | the §3.2 block | green |
| E12 integration gate | `cargo test -p cb-train --test ctr_counter_simple_oracle_test` | **4/4** (the target has exactly four `#[test]` fns, E-§1) |
| **no residual mutation** | `git diff crates/cb-train/src/tree.rs` | only C03 + C04 hunks — **no** `+ 0.1`, no f64 helper body, no converted `:3289` |
| crate sweeps | `cargo test -p cb-train --no-fail-fast`; `cargo test -p cb-model --no-fail-fast` | only the pre-existing `monotone_non_symmetric_and_region_are_typed_errors` fails |
| lints | `cargo clippy --workspace --all-targets` | clean |

### TDD sequence

1. **Red** — n/a (documentation + confirmation task; no new behavior).
2. **Green** — n/a.
3. **Refactor** — n/a.
4. **Verify** — execute the whole table in part 4 and paste the results into the
   completion evidence. Then apply the SPEC.md edits in part 3.

### Completion criteria

- [ ] SPEC-CTRB-04's decision (do NOT convert `boosting.rs:2015`) recorded in
      SPEC.md §5 with its E9 evidence chain.
- [ ] SPEC.md §9 OQ1 closed: four construction sites (not three), and **no**
      non-oblivious grower persists CTR borders.
- [ ] The "E03 tests are UNAFFECTED and must not be edited" determination recorded.
- [ ] SPEC.md's already-applied pass-1 corrections (§2/§4, SPEC-CTRB-01 BOUNDED
      DOMAIN + STOP CONDITION, SPEC-CTRB-03 units retitle, §9 R1 retraction)
      **verified present and consistent with E6/E20/E21**.
- [ ] SPEC.md §10 corrected to the **4/4** E12 gate.
- [ ] Every row of the part-4 table executed and its result pasted verbatim,
      **including the `git diff crates/cb-train/src/tree.rs` residual-mutation
      row**.
- [ ] `git diff --stat` shows **zero** diff on `crates/cb-train/src/boosting.rs`,
      `crates/cb-train/src/boosting_test.rs`, `crates/cb-model/src/apply.rs`, and
      all eleven CTR oracle files.
- [ ] No fixture regenerated; `gen_fixtures.py` never invoked.

### Risks and guardrails

- **R:** treating the 11-oracle green as proof the defect is fixed. **M:** D8 and
  E10 — this plan's proof is C01 + C02 + C04's mutation checks; the eleven are a
  non-regression gate only.
- **R:** treating the 11-oracle green as acceptance for **SPEC-CTRB-03**. **M:**
  the note under this task's spec list, D3a, and E20 — the oracles are green under
  a wrong-site conversion too, so they prove nothing about that spec.
- **R:** editing SPEC.md's already-approved sections. **M:** the remaining edits
  are **additive** (a decision record, a `4/4` correction, an evidence note); the
  pass-1 corrections were made by the coordinator and are only **verified** here,
  not rewritten.

---

## 6. Specification → task coverage

| Spec | Tasks | Falsifiable gate | Notes |
|---|---|---|---|
| **SPEC-CTRB-01** persisted border is a value-space threshold | **C02** (Red, upstream-anchored) · **C03** (Green) · **C04** (bounded sweep A/B/C + boundary Test E + mutation M1) | C02 Tests 1–2 fail today; **C04 Test E** fails (`16.999999046325684 != 17.0`) for any f64 formulation (E6) | Tests B/C sweep the **reachable domain only** (`0..=14`, E21) and assert **no** generality. The f32-vs-f64 discriminator is **Test E**, not B/C. D2's rationale is the `.cbm` f32 fixed point (E14), **not** parity generality |
| **SPEC-CTRB-02** training and apply agree on the border | **C01** (PRIMARY Red) · **C03** (Green) | C01 Tests 1–2 fail today, naming doc/bin/both borders/both booleans | Not covered by any existing oracle (E10) |
| **SPEC-CTRB-03** `LevelKind::Ctr.border` stays bin space — a **units contract** | **C01** (integral assertion) · **C03** (byte-unchanged expression at line 3289 + accurate comment) · **C04** (Test D + mutation M2) · **C06** (decision record) | **M2 fails Test D and C01's step-6 assertion — and ONLY those.** The E12 gate, the 11 CTR oracles and the 3 one-hot targets **stay GREEN** under M2, by E20's integer-operand arithmetic | **Corrected at plan-check pass 1.** Converting `:3289` is arithmetically a **no-op**; the integrality assertions are the **only** possible detector (D3, D3a). The oracles are **vacuous** as acceptance for this spec and must never be cited as such |
| **SPEC-CTRB-04** `!has_ctr` fallback unaffected | **C06** (decision record + `boosting::tests` run) | `boosting::tests` green with zero diff | E9: the literal is unreachable in production |
| **SPEC-CTRB-05** `.cbm` round-trip lossless | **C05** (guard + mutation) | mutation `+ 0.1` makes the bitwise assertion fail | Guard, not Red — passes before and after C03 |

**Every task maps back:** C01→{02,03}, C02→{01}, C03→{01,02,03}, C04→{01,03},
C05→{05}, C06→{03,04}. No orphan tasks, no uncovered specs.

---

## 7. Open questions (none blocking)

- **OQ-A (resolved, applied).** SPEC.md §9 OQ1 named three `CtrSplitSpec`
  construction sites; there are **four** (E11/E12), and **no** non-oblivious
  grower persists CTR borders (E13). **Both corrections are now IN SPEC.md**
  (applied by the coordinator at pass 1); C06 verifies rather than authors them.
- **OQ-D (open, deliberately unresolved — do NOT resolve by guessing).**
  Upstream's exact CTR-border epsilon expression **above `b = 15`** is
  `[UNVERIFIED]`. Every committed fixture is reproduced bit-for-bit by
  `(b+1) − 2⁻²⁰` computed in f32, but so is `(b+1) − 1e-6f`, and the two round
  **opposite ways** at `b = 16`. No in-repo artifact reaches that range and
  `catboost-master/` may not be cited (D9). The range is **unreachable** today
  (E21), so this is not blocking — it is pinned by C04 Test E's characterization
  + STOP condition. **If a future task makes `ctr_border_count` configurable, this
  question becomes blocking and must be settled against the `v1.2.10` tag before
  the widened domain ships.**
- **OQ-B (contingent, non-blocking).** If, **after** C03, C02's Test 2
  (upstream-border membership) fails while Test 1 (the convention) passes, the
  trainer chose a different bin threshold than catboost 1.2.10 on that fixture.
  That is a **new structural parity finding**, not a border-space defect —
  **STOP AND REPORT**; do not weaken Test 2 and do not proceed past C03.
  Probability assessed low: `tensor_ctr_e2e` already matches upstream at ≤1e-5
  across all 5 trees, which requires identical tree structure.
- **OQ-C (cosmetic).** The plan request states the E12 gate passes "5/5"; the
  target has **4** `#[test]` functions (§1). If a fifth is added between planning
  and execution, the gate is "all tests in
  `ctr_counter_simple_oracle_test` green", whatever the count.

## 8. Blockers

**None.** Every referenced path, symbol, line number, command and test target has
been verified against the working tree; the defect has been reproduced; and the
fix's numeric form has been verified bit-for-bit against **every committed CTR
fixture border** (E4/E5).

Two things are explicitly **not** claimed, and the plan is built around that:

- The formula is **not** verified above `b = 15` — it is *known* to lose the
  strict-interval invariant there (E6), the correct upstream value in that range
  is `[UNVERIFIED]`, and the range is **unreachable** at
  `ctr_border_count_default() == 15` (E21). C04 Test E pins the boundary with a
  STOP condition instead of asserting generality.
- Converting `tree.rs:3289` is **not** a correctness hazard (E20). The plan's
  detector for it is an explicit integrality assertion, not any oracle.
