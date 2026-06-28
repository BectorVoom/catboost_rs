---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 10
subsystem: cpu-training-ordered-boosting
tags: [ordered-boosting, ORD-02, train-driver, e2e-oracle, fold, parity]

requires:
  - phase: 05-08
    provides: "greedy_tensor_search_oblivious_ordered (ordered per-segment split-scoring subsystem) + re-export"
  - phase: 05-03
    provides: "create_folds / body_tail_segments / body_sum_weights (fold machinery)"
  - phase: 04-02
    provides: "cb_model::predict_raw production apply path"
  - phase: 04-01
    provides: "cb_model::Model::from_trained"
provides:
  - "EBoostingType::Ordered branch in train_with_eval_sets() driving greedy_tensor_search_oblivious_ordered over a once-created fold set (FOLDS-BUILT-ONCE)"
  - "gen_ordered_boost_e2e() offline generator for the full ordered train->predict fixture stack"
  - "ordered_boost_e2e_oracle_test.rs (FULL multi-tree ordered oracle through cb_model::predict_raw, no #[ignore])"
  - "ordered_boost_wiring_test.rs (Ordered-vs-Plain structural divergence + folds-built-once wiring gate)"
affects: [05-09]

tech-stack:
  added: []
  patterns:
    - "FOLDS-BUILT-ONCE: create_folds called exactly once (grep-enforced, non-comment count == 1), outside the iteration loop, continuous-stream RNG"
    - "Ordered branch grows STRUCTURE via the ordered search; leaf VALUES reuse the Plain compute_leaf_deltas/accumulate_leaf_weights on the averaging fold (Plain-identical, CalcLeafValuesSimple)"
    - "e2e oracle validates through the production cb_model::predict_raw apply path (not the cb-train staged approx)"

key-files:
  created:
    - crates/cb-train/tests/ordered_boost_wiring_test.rs
    - crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs
  modified:
    - crates/cb-train/src/boosting.rs
    - crates/cb-oracle/generator/gen_fixtures.py
    - crates/cb-train/Cargo.toml
    - Cargo.lock

key-decisions:
  - "greedy_tensor_search_oblivious_ordered's REAL 05-08 signature is 8 params (matrix, der1, weight, permutation, l2_leaf_reg, fold_len_multiplier, depth, n_objects) — it derives body_tail_segments + body_sum_weights INTERNALLY. The plan's placeholder arg list (passing &segments/&body_sum_weights/sum_all_weights) was wrong; the real signature was read from tree.rs and 05-08-SUMMARY before wiring."
  - "create_folds fully-qualified at the single call site (crate::fold::create_folds) and dropped from the `use` import so the grep-enforced non-comment count is exactly 1 (the call), satisfying FOLDS-BUILT-ONCE."
  - "cb-model added as a DEV-dependency of cb-train (not a runtime dep) so the e2e oracle can validate through cb_model::predict_raw; dev-deps are exempt from the cb-model->cb-train build-graph cycle."

patterns-established:
  - "Ordered branch passes the FULL un-masked weighted_der1/weights (random_strength=0 + bootstrap=No ⇒ no perturbation/bootstrap draws) to the ordered search; leaf_of is object-order so the Plain leaf-value path applies unchanged."

requirements-completed: []  # ORD-02 NOT yet closable — see Blocking Issue below (oracle fixtures pending offline catboost==1.2.10 generation).

duration: ~40min
completed: 2026-06-14
---

# Phase 5 Plan 10: Ordered-Boosting Train-Driver Wiring + FULL Multi-Tree E2E Oracle Summary

**Wired `greedy_tensor_search_oblivious_ordered` (05-08) into `train_with_eval_sets` under `EBoostingType::Ordered` with a once-created fold set, and authored the full multi-tree ordered train→predict oracle through `cb_model::predict_raw` — the oracle's binary fixtures are blocked on an offline `catboost==1.2.10` generation run.**

## Performance

- **Duration:** ~40 min
- **Tasks:** 1 of 2 fully complete + verified; Task 2 source complete, oracle blocked on external fixtures
- **Files modified:** 6

## Accomplishments

- **Task 1 (COMPLETE, verified):** `train_with_eval_sets` now branches on `params.boosting_type`. The Ordered path builds the fold set ONCE before the iteration loop (`crate::fold::create_folds`, continuous-stream RNG, permutation_count=1 → 1 learning + 1 averaging fold), grows each tree's STRUCTURE via `greedy_tensor_search_oblivious_ordered` over the learning fold permutation, and estimates leaf VALUES on the averaging fold via the existing Plain `compute_leaf_deltas`/`accumulate_leaf_weights`. The Plain branch is byte-identical. Ordered is no longer dead code.
- **FOLDS-BUILT-ONCE invariant holds and is grep-enforced:** `create_folds` appears exactly once (non-comment) in `boosting.rs`, outside the iteration loop.
- **Task 2 (source COMPLETE, oracle BLOCKED):** `gen_ordered_boost_e2e()` added to `gen_fixtures.py` (wired into `main()`); `ordered_boost_e2e_oracle_test.rs` trains the same Ordered config, lifts into `cb_model::Model`, predicts via the production `cb_model::predict_raw`, and asserts `compare_stage(Stage::Predictions)` ≤1e-5 across ALL trees with NO `#[ignore]`, plus the iter-0 no-leakage anchor. The test compiles and is correct but cannot PASS until the binary fixtures are generated (see Blocking Issue).

## Task Commits

1. **Task 1: wire ordered search into train()** — `eee112c` (feat) — boosting.rs Ordered branch + ordered_boost_wiring_test.rs
2. **Task 2: e2e generator + oracle (source)** — `018c633` (feat) — gen_fixtures.py gen_ordered_boost_e2e + ordered_boost_e2e_oracle_test.rs + cb-model dev-dep + Cargo.lock

_Note: Task 1 was executed TDD-style; the RED state (`ordered_structure_differs_from_plain` failing against the dead stub) was observed before the GREEN wiring, then committed together as the wiring feat._

## Files Created/Modified

- `crates/cb-train/src/boosting.rs` — replaced the `debug_assert`-only Ordered stub with a real `boosting_type` branch; fold set built once; Ordered tree-structure growth via `greedy_tensor_search_oblivious_ordered`; Plain path unchanged.
- `crates/cb-train/tests/ordered_boost_wiring_test.rs` (created) — Ordered grows a full finite 5-tree model; Ordered structure DIFFERS from Plain (falsifiable dead-branch catch); Plain still trains.
- `crates/cb-oracle/generator/gen_fixtures.py` — `gen_ordered_boost_e2e()` (OFFLINE, pinned catboost==1.2.10) + `ORDERED_BOOST_E2E` path + `main()` wiring.
- `crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs` (created) — FULL multi-tree ordered oracle through `cb_model::predict_raw` (no `#[ignore]`) + iter-0 no-leakage anchor.
- `crates/cb-train/Cargo.toml` / `Cargo.lock` — cb-model added as a dev-dependency only.

## Decisions Made

- **Real 05-08 signature used (NOT the plan's placeholder).** Per the project-specific instruction, the actual `greedy_tensor_search_oblivious_ordered` signature (8 params; segments/body-sum-weights derived internally) was read from `tree.rs` + `05-08-SUMMARY.md` before writing the call site. The plan's `<action>` placeholder (passing `&segments`, `&body_sum_weights`, `sum_all_weights`) would not have compiled; the call uses `(&matrix, &weighted_der1, &weights, learning_perm, params.l2_leaf_reg, params.fold_len_multiplier, params.depth, n)`.
- **FOLDS-BUILT-ONCE grep satisfied by fully-qualifying the call** (`crate::fold::create_folds`) and removing `create_folds` from the `use` import, so the only non-comment occurrence is the single call site.
- **cb-model is a dev-dependency** (test-only) of cb-train so the oracle can route through the production apply path without introducing a runtime dependency cycle.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan's ordered-search call signature was a placeholder and would not compile**
- **Found during:** Task 1
- **Issue:** The plan's `<action>` listed `greedy_tensor_search_oblivious_ordered(&matrix, &weighted_der1, &weights, &learning_fold.permutation, &segments, &body_sum_weights, params.l2_leaf_reg, sum_all_weights, params.depth, n)` — but the 05-08 function takes 8 params and derives segments/body-sum-weights internally.
- **Fix:** Called with the real signature `(&matrix, &weighted_der1, &weights, learning_perm, params.l2_leaf_reg, params.fold_len_multiplier, params.depth, n)`.
- **Files modified:** crates/cb-train/src/boosting.rs
- **Verification:** `cargo check --workspace --tests` clean; wiring test green.
- **Committed in:** eee112c

**2. [Rule 3 - Blocking] cb-model not available to cb-train tests**
- **Found during:** Task 2
- **Issue:** The e2e oracle must predict through `cb_model::predict_raw`, but cb-model was not a (dev-)dependency of cb-train (E0432 unresolved import).
- **Fix:** Added `cb-model = { path = "../cb-model" }` under `[dev-dependencies]` (test-only; exempt from the build-graph cycle).
- **Files modified:** crates/cb-train/Cargo.toml, Cargo.lock
- **Verification:** e2e test compiles (`--no-run` succeeds).
- **Committed in:** 018c633

---

**Total deviations:** 2 auto-fixed (both Rule 3 blocking). **Impact:** Necessary to compile against the real subsystem and the production apply path. No scope creep.

## Issues Encountered

### BLOCKING ISSUE (Rule 2 escalation — checkpoint:human-action): ordered_boost_e2e oracle fixtures cannot be generated in this environment

- **What:** Task 2's oracle (`ordered_boost_e2e_oracle_test.rs`) requires the OFFLINE-generated fixtures `crates/cb-oracle/fixtures/ordered_boost_e2e/{X.npy, y.npy, model.json, predictions.npy}`, produced by `gen_ordered_boost_e2e()` using pinned `catboost==1.2.10`.
- **Why blocked:** `catboost==1.2.10` is **not importable** in this environment (`ModuleNotFoundError: No module named 'catboost'` under every available Python; `pip show catboost` → not found; no venv carries it). Threat-register entry T-05-10-SC assumed catboost was already pinned/installed (D-09), but it is not present here.
- **What was NOT done (by explicit instruction):** the oracle was **NOT** weakened, **NOT** `#[ignore]`'d, **NOT** given a looser tolerance, and **NO** fixtures were fabricated. The plan's user-set FULL multi-tree hard gate forbids these.
- **Current state:** `ordered_boost_e2e_oracle_test.rs` compiles and is correct; running it FAILS on the missing-fixture file loads (NotFound), as expected until the fixtures exist.
- **Resolution required (offline, one-time):** on a machine with `catboost==1.2.10` (+ numpy), run the generator and commit the four binary fixtures:
  ```
  cd crates/cb-oracle/generator && python3 gen_fixtures.py
  # then commit crates/cb-oracle/fixtures/ordered_boost_e2e/{config.json,X.npy,y.npy,model.json,predictions.npy}
  ```
  After the fixtures land, run `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` — it must pass ≤1e-5 across all trees. If it does NOT pass at random_strength=0 (contrary to the D-11 analysis), escalate per the plan's Rule 2 rather than weakening the gate.

## Verification Results

Per-crate (disk-pressure constraint — no `cargo test --workspace`):

- `cargo test -p cb-train --test ordered_boost_wiring_test` — 3/3 (Ordered grows full model; Ordered structure ≠ Plain; Plain still trains).
- `cargo test -p cb-train --test ordered_boost_oracle_test` — 5/5 (standalone approximant unchanged).
- `cargo test -p cb-train --test slice_first_oracle_test` — 2/2 (Plain numeric unchanged).
- `cargo test -p cb-train --test one_hot_oracle_test` — 4/4 (Plain one-hot unchanged).
- `cargo test -p cb-train --test leaf_methods_oracle_test` — 3/3 (Plain leaf methods unchanged).
- `grep -v '^[[:space:]]*//' crates/cb-train/src/boosting.rs | grep -c 'create_folds'` — **1** (FOLDS-BUILT-ONCE).
- `grep -c "greedy_tensor_search_oblivious_ordered" crates/cb-train/src/boosting.rs` — 3 (≥1).
- `grep -c "this whole-set train driver stays the Plain path" crates/cb-train/src/boosting.rs` — 0 (stub removed).
- `grep -c "def gen_ordered_boost_e2e" crates/cb-oracle/generator/gen_fixtures.py` — 1.
- `grep -c "cb_model::predict_raw\|predict_raw(" crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs` — ≥1.
- `grep -c "#\[ignore" crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs` — 0.
- `cargo check --workspace --tests` — clean.
- No new `unwrap`/`expect`/`panic`/`anyhow` in production boosting.rs.

**NOT yet passing (blocked):** `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` — fails on missing fixtures (see Blocking Issue). ORD-02 cannot be marked complete until this passes.

## Known Stubs

None in production code. The Ordered branch is fully implemented and exercised by `ordered_boost_wiring_test.rs`. The e2e oracle is real (not a stub) but is data-blocked on the offline fixture generation.

## Threat Flags

None — no new security surface beyond the plan's `<threat_model>` (fixture loads go through the bounds-checked npy reader / cb_model JSON loader; the Ordered branch uses checked `.get` and `create_folds` Fisher-Yates `< n` indices).

## Next Phase Readiness

- Task 1 (ordered train-driver wiring) is production-ready and verified; 05-09 (tensor-CTR wiring that depended on the ordered subsystem being wired) is unblocked on the wiring axis.
- **ORD-02 closure is gated on the offline fixture generation** (Blocking Issue). Once `ordered_boost_e2e/` is committed and the oracle passes, ORD-02 (AUTH of the full multi-tree hard gate) closes.

## Self-Check: PASSED

- FOUND: crates/cb-train/src/boosting.rs
- FOUND: crates/cb-train/tests/ordered_boost_wiring_test.rs
- FOUND: crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs
- FOUND: crates/cb-oracle/generator/gen_fixtures.py
- FOUND commit eee112c (Task 1)
- FOUND commit 018c633 (Task 2 source)

_Note: this self-check verifies the committed source artifacts exist. The ORD-02
oracle PASS is separately gated on the offline `ordered_boost_e2e/` fixtures (see
Blocking Issue) and is intentionally NOT claimed here._

---
*Phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s*
*Completed (Task 1) / Blocked (Task 2 oracle): 2026-06-14*
