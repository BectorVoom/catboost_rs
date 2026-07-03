---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
plan: 02
subsystem: grow-policy / tree-model
status: complete
tags: [region, grow-policy, GPUT-18, tree-variant, path-model, cpu-oracle]
requires:
  - cb-train boosting loop (grow dispatch, leaf-value estimation)
  - cb-model apply / json / from_trained lift
provides:
  - cb_model::TreeVariant::Region + RegionTree/RegionLevel path model
  - cb_model::Model.region_trees + from_trained Region lift
  - cb_model walk-until-diverge region apply (region_leaf) + region json round-trip
  - cb_train::region_grower (d+1-leaf CPU path grower) + region_trees
  - grow_policy=Region CPU training (validate_grow_policy rejection LIFTED)
  - frozen ≤1e-5 CPU Region oracle (the reference Plan 04's device kernel reproduces)
affects:
  - crates/cb-model/src/{model,apply,json,cbm,lib}.rs
  - crates/cb-train/src/{tree,boosting,lib}.rs
tech-stack:
  added: []
  patterns:
    - "Region is a PATH model (walk-while-direction-matches), NOT a binary node graph"
    - "leaf_values indexed directly by walk bin (0..=depth); self-consistent, no re-sort"
    - "src-mounted unit tests cannot use cb_model (dev-dep cb_train diamond) → e2e in tests/"
key-files:
  created:
    - crates/cb-model/src/region_apply_test.rs
    - crates/cb-train/src/region_grow_test.rs
    - crates/cb-train/tests/region_e2e_test.rs
  modified:
    - crates/cb-model/src/model.rs
    - crates/cb-model/src/apply.rs
    - crates/cb-model/src/json.rs
    - crates/cb-model/src/cbm.rs
    - crates/cb-model/src/lib.rs
    - crates/cb-train/src/tree.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/lib.rs
decisions:
  - "Continue direction = the frontier child with higher subsequent best-split gain (upstream SelectLeavesToSplit picks the lower-Score == higher-gain child); ties prefer the passes-child (deterministic)."
  - "leaf_values indexed DIRECTLY by the AddRegionImpl walk bin (0..=depth), NOT upstream's depth-then-direction sorted storage — simpler + fully self-consistent, and this plan ESTABLISHES the oracle (no upstream numeric fixture to match this phase)."
  - "Region apply + e2e that touches cb_model must live in a tests/ integration test, not a src-mounted unit test, because the src test's cb_train instance differs from cb_model's cb_train dev-dep (diamond → E0308)."
metrics:
  duration: ~55 min
  completed: 2026-07-03
  tasks: 2
  files: 11
---

# Phase 12 Plan 02: Region CPU Grow-Policy Path Summary

Built the CPU Region grow-policy path (GPUT-18 / D-03a) — the single largest Phase-12 lift and the v1.0-escalated "Region OUT" gap. Region is now a first-class `TreeVariant::Region(RegionTree)` **path** model (upstream `TRegionModel`): an oblivious-like path walked while each level's split matches the stored direction, diverging into a terminal leaf otherwise. A depth-`d` region has exactly `d + 1` leaves. Trainable (`grow_policy=Region`), serializable (json round-trip), applicable (walk-until-diverge), and oracle-locked ≤1e-5 to a frozen CPU reference — the reference Plan 04's device Region kernel reproduces.

## What shipped

**Task 1 — `TreeVariant::Region` path model + apply + json (commit `00251d1`)**
- `RegionLevel { split, expected_direction, one_hot }` + `RegionTree { levels, leaf_values, leaf_weights }` (cb-model) and the cb-train `RegionTree { splits, directions, one_hot, leaf_values, leaf_weights }`.
- `TreeVariant::Region` variant with exhaustive match arms (`as_oblivious`/`as_non_symmetric`/new `as_region`/`leaf_values`/`leaf_weights`).
- `Model.region_trees` on both cb-model and cb-train; `from_trained` Region lift (float split → `ModelSplit::Float`, zipped with direction + one-hot).
- `region_leaf` walk in `predict_raw_one`: `bin = 0; for level { if passes == expected_direction { bin += 1 } else break } leaf = bin`. Checked `.get`, malformed → `0.0`, no panic (T-12-03). Transcribes `add_model_value.cu::AddRegionImpl`.
- Self-contained `region_trees` json round-trip (new top-level key, `skip_serializing_if` keeps oblivious/non-sym exports byte-identical).
- `region_apply_test.rs`: d+1 leaves (not 2^d), walk breaks at first mismatch, malformed → 0.0, json save→load→apply bit-identical.

**Task 2 — Region CPU grower + `validate_grow_policy` lift (commit `57259c5`)**
- `region_grower` (tree.rs): grows ONE path, reusing `check_depth` + `best_split_for_leaf`. At each level the frontier is split; the child with the higher subsequent best-split gain continues, the other diverges into a terminal bin. Emits `region_directions`/`region_one_hot` + bin-ordered `leaf_of` (NOT `step_nodes`). Degenerate root → typed `CbError::Degenerate`.
- `GrownTree` gained `region_directions`/`region_one_hot` (empty for oblivious/non-sym; 5 construction sites updated).
- `validate_grow_policy`: Region rejection LIFTED; explicit Region dispatch arm (removed the SymmetricTree fall-through); `n_leaves = depth+1` shadow for region; persistence pushes `RegionTree`; `use_best_model` truncates `region_trees`. Monotone-constraints + Region still rejected.
- `region_grow_test.rs`: frozen depth-2 structure (splits `[(f0,1.5),(f0,0.5)]`, directions `[false,true]`, `leaf_of [1,1,2,2,0,0]`, 3 leaves), determinism, degenerate→Err, grown `leaf_of` == replica walk bin.
- `region_e2e_test.rs`: `grow_policy=Region` trains `region_trees` (oblivious/non-sym empty), frozen structure + predictions `[2,2,0,0,-3,-3]` ≤1e-5 through `cb_model` apply, deterministic.

## Frozen Region oracle (the reference Plan 04 reproduces)

- Fixture: f0 `[0,0,1,1,2,2]`, f1 `[0,1,0,1,0,1]`, RMSE der1 `[-2,-2,0,0,3,3]` (target `[2,2,0,0,-3,-3]`), borders f0 `[0.5,1.5]` / f1 `[0.5]`, `scaled_l2 = 0`, score L2.
- Grown path: level 0 `f0 > 1.5` continue=`false` (peels the +3 pair → bin 0); level 1 `f0 > 0.5` continue=`true` (bin 1 = `{0,1}`); surviving `{2,3}` → bin 2. Depth 2, 3 leaves.
- Gradient-L2 leaf values `[-3, 2, 0]`; one lr=1 iteration fits the separable fixture exactly → predictions `[2,2,0,0,-3,-3]`.
- Frozen in `crates/cb-train/src/region_grow_test.rs` (structure) and `crates/cb-train/tests/region_e2e_test.rs` (end-to-end predictions).

## Deviations from Plan

### Auto-fixed / structural (Rules 2-3)

**1. [Rule 3 - Blocking] cb-train `Model.region_trees` field added in Task 1's commit**
- `cb_model::from_trained` (Task 1) reads `trained.region_trees`, which requires the cb-train field to exist for cb-model to compile. The cb-train `RegionTree` type + `Model.region_trees` field (empty, no grower) therefore ship in Task 1's commit; the grower that populates them ships in Task 2. Each commit compiles + tests independently.

**2. [Rule 3 - Blocking] e2e apply oracle moved to a `tests/` integration test**
- The plan lists only `region_grow_test.rs`. A src-mounted unit test's `cb_train` instance differs from `cb_model`'s `cb_train` dev-dep (feature-unification diamond → `E0308`, "multiple different versions of crate cb_train"). Added `crates/cb-train/tests/region_e2e_test.rs` (integration, `cb_train` external like `non_symmetric_grower_oracle_test.rs`) for the `grow_policy=Region` train → lift → `predict_raw` frozen oracle. The src `region_grow_test.rs` locks the grower structure + a replica-walk agreement (no `cb_model`).

**3. [Decision] Direct bin-indexed leaf storage (not upstream's sorted storage)**
- Upstream `BuildTreeLikeModel<TRegionModel>` sorts leaves by (depth, last-direction). This plan stores `leaf_values` indexed directly by the walk `bin` — simpler and fully self-consistent (grower `leaf_of` and apply walk agree by construction). Legitimate because this plan ESTABLISHES the oracle; no upstream numeric Region fixture is matched this phase (RESEARCH Pitfall 1).

## Deferred Issues

- Pre-existing `cargo clippy` debt in `cb-backend` / `cb-oracle` (`indexing_slicing`) blocks a full-graph `cargo clippy -p cb-model`; unrelated to Region (those crates untouched). `cargo build` / `cargo test -p cb-model` / `cargo test -p cb-train` all pass. Logged to `deferred-items.md`.
- Multi-output (`approx_dimension > 1`) region apply is out of scope (the in-scope Region oracle is scalar); `predict_raw_multi` does not iterate `region_trees`. No exhaustiveness break (the multi path iterates separate vecs, not a `TreeVariant` match).
- Region SHAP / fstr / `.cbm` round-trip are out of scope (json round-trip only, per plan); those consumers iterate `oblivious_trees` and silently ignore region trees.

## Verification

- `cargo test -p cb-model region_apply` — 4/4 green.
- `cargo test -p cb-model` — 85 passed, 0 failed (no oblivious/non-sym regression).
- `cargo test -p cb-train --lib` — 235 passed, 0 failed.
- `cargo test -p cb-train --test region_e2e_test` — 2/2 green (frozen structure + predictions ≤1e-5, deterministic).
- `cargo test -p cb-train --test non_symmetric_grower_oracle_test` — 1/1 green (dispatch change no-regression).

## Self-Check: PASSED

- Created files exist: `region_apply_test.rs`, `region_grow_test.rs`, `region_e2e_test.rs` — all FOUND.
- Commits exist: `00251d1`, `57259c5` — both FOUND.
- Region symbols present: `region_grower`, `TreeVariant::Region`, `region_leaf` — all OK.
