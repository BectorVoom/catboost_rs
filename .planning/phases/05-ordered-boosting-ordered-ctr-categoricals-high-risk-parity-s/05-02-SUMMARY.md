---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 02
subsystem: cb-train (categorical one-hot encoding path, ORD-04 / D-04)
tags: [one-hot, categorical, encoding-path, one_hot_max_size, oblivious-trees, d-04-isolation, self-oracle]
requires:
  - cb-train::train / greedy_tensor_search_oblivious (Phase 3 plain boosting + oblivious trees)
  - cb-train::slice_first_oracle (the upstream-locked ≤1e-5 float reference, Phase 3)
  - cb-data::calc_cat_feature_hash + PerfectHash (CityHash64 first-seen bins, Phase 2)
  - cb-compute::{reduce_leaf_stats, gradient_leaf_delta, scale_l2_reg, sum_f64} (D-08)
  - cb-oracle::compare_stage (Stage::StagedApprox / Predictions, ≤1e-5 gate)
provides:
  - cb-train::candidates (route_categorical / EncodingPath / learn_set_cardinality / route_column)
  - cb-train::BoostParams.one_hot_max_size (explicit pin, default 2)
  - cb-train::tree::{OneHotSplit, AnySplit, GrownOneHotTree, grow_one_hot_tree}
  - cb-train::FeatureMatrix.cat_bins + FeatureMatrix::new
  - crates/cb-train/tests/one_hot_oracle_test.rs (D-04 isolation oracle)
affects:
  - 05-03..05-06 (CTR path branches off route_categorical::Ctr; the one-hot split
    machinery is the categorical-tree substrate the CTR splits extend)
tech-stack:
  added: []
  patterns:
    - one-hot routing on learn-set cardinality (AddOneHotFeatures, inclusive/exclusive boundary)
    - separate OneHotSplit/AnySplit type to keep the shared float Split byte-for-byte (no cross-crate churn)
    - transcribe-then-self-oracle for D-04 (anchor to the oracle-locked float path, not a missing fixture)
key-files:
  created:
    - crates/cb-train/src/candidates.rs
    - crates/cb-train/src/candidates_test.rs
    - crates/cb-train/tests/one_hot_oracle_test.rs
  modified:
    - crates/cb-train/src/tree.rs
    - crates/cb-train/src/tree_test.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/lib.rs
    - crates/catboost-rs/src/builder.rs
    - crates/cb-train/tests/autolr_e2e_test.rs
    - crates/cb-train/tests/bootstrap_oracle_test.rs
    - crates/cb-train/tests/eval_metrics_oracle_test.rs
    - crates/cb-train/tests/leaf_methods_oracle_test.rs
    - crates/cb-train/tests/leaf_weights_oracle_test.rs
    - crates/cb-train/tests/loss_oracle_test.rs
    - crates/cb-train/tests/overfit_oracle_test.rs
    - crates/cb-train/tests/regularization_oracle_test.rs
    - crates/cb-train/tests/slice_first_oracle_test.rs
decisions:
  - "One-hot routing is inclusive at count==one_hot_max_size, exclusive above (CTR), skip at <=1 — reproducing AddOneHotFeatures skip predicate (count>max)||(count<=1) (greedy_tensor_search.cpp:171-197, RESEARCH Pitfall 3)."
  - "The shared float Split struct was left BYTE-FOR-BYTE unchanged; one-hot splits use a SEPARATE OneHotSplit/AnySplit type confined to the categorical growth path, avoiding breaking ~20 cb-model Split literals (Rule 3 scope containment)."
  - "D-04 oracle is transcribe-then-self-oracle: the one-hot-only model is anchored to the EXISTING upstream-oracle-locked float train on the equivalent one-hot binary columns (the committed one_hot_cat fixture is the CTR/permutation Wave-0 anchor — no one-hot-only model.json — and per its own config generates a permutation, so it cannot serve as a D-04 isolation oracle)."
metrics:
  duration_min: 17
  completed: "2026-06-13"
  tasks: 2
  files: 17
---

# Phase 5 Plan 02: One-Hot Encoding Path Selection (ORD-04 / D-04) Summary

Delivered the narrowest first slice of the high-risk categorical phase: one-hot
encoding for low-cardinality categoricals (`one_hot_max_size`), riding the
EXISTING plain boosting + oblivious trees with NO permutation and NO CTR math.
A categorical column routes to one-hot vs CTR purely on its learn-set
cardinality (inclusive boundary at `one_hot_max_size`), one-hot `cat_bin == k`
splits grow through the same L2 score + strict first-wins tie-break as float
splits, and a one-hot-only model trains and predicts bit-for-bit equal to the
upstream-oracle-locked float reference (≤1e-5) with NO permutation present.

## What Shipped

### Task 1 — One-hot vs CTR path selection on learn-set cardinality (`392fe65`)
- New `cb-train::candidates` module: `EncodingPath{OneHot, Ctr, Skip}` +
  `route_categorical(cardinality, one_hot_max_size)` reproducing the upstream
  `AddOneHotFeatures` skip predicate `(count > max) || (count <= 1)`
  (`greedy_tensor_search.cpp:171-197`) — INCLUSIVE one-hot at
  `count == one_hot_max_size`, EXCLUSIVE above (CTR), skip at `<= 1` (RESEARCH
  Pitfall 3, no off-by-one).
- `learn_set_cardinality` / `route_column`: learn-set-only distinct-value count
  via `cb_data::calc_cat_feature_hash` + `PerfectHash` first-seen bins (the
  `GetUniqueValuesCounts(...).OnLearnOnly` equivalent — NEVER a model `ctr_data`
  hash_map, D Carried-Forward / Anti-Pattern). Propagates the typed `OutOfRange`
  overflow guard from `PerfectHash::remap` (no panic, T-05-02-01).
- `BoostParams.one_hot_max_size` (default 2, `cat_feature_options.cpp:231-232`)
  pinned EXPLICITLY (never auto-selected, RESEARCH Pitfall 6);
  `one_hot_max_size_default()` exposes the canonical default.
- `candidates_test.rs`: all boundary cases (`one_hot_threshold*`) — `==max`
  inclusive one-hot, `==max+1` CTR, `==2` one-hot, `<=1` skip, default-2
  boundary, learn-set cardinality counting, end-to-end `route_column`.

### Task 2 — Categorical one-hot splits + one-hot-only predict oracle ≤1e-5 (`da4fb30`)
- `tree.rs`: `OneHotSplit{feature, value}` (a `cat_bin == value` equality,
  `IsTrueOneHotFeature`, `split.h:16-17`); `AnySplit{Float|OneHot}`;
  `FeatureMatrix.cat_bins` + `FeatureMatrix::new`; `grow_one_hot_tree` enumerates
  FLOAT candidates then ONE-HOT candidates (`AddFloatFeatures`/`AddOneHotFeatures`
  order, feature asc / border|bin asc), scores each via the SAME `l2_split_score`
  + ordered `reduce_leaf_stats`, and selects with the SAME strict first-wins
  tie-break (strict `>`, Pitfall 1), forward-bit leaf index.
- The shared float `Split` struct was left **byte-for-byte unchanged** — one-hot
  splits live in the separate `AnySplit` type confined to the categorical growth
  path, so the ~20 `cb-model` `Split{feature, border}` literals (re-export of
  `cb_train::Split`) keep compiling untouched.
- `one_hot_oracle_test.rs` (D-04 isolation oracle): a one-hot-only boosting driver
  (built on `grow_one_hot_tree`, Gradient/RMSE/`boost_from_average`, NO RNG)
  trains+predicts and is gated against the EXISTING upstream-oracle-locked float
  `train` on the equivalent one-hot binary columns — `Stage::StagedApprox` (every
  per-iteration approximant) and `Stage::Predictions` (final) match ≤1e-5 (in fact
  bit-exact). `no_permutation_in_one_hot_only_path` asserts the path is
  deterministic (no RNG / no permutation reordered the documents, RESEARCH Pitfall
  2). Path-selection boundary re-asserted at the oracle entry.
- `tree_test.rs`: single one-hot split leaf-assignment (locks the structure +
  `IsTrueOneHotFeature` leaf index, not a hand-picked winning bin) + degenerate
  no-candidate guard (typed `Degenerate`, never panic, T-05-02-02).
- All float sums via `cb_core::sum_f64` (D-08); checked access only, no
  `unwrap`/`expect`/`panic`/raw-index, no `anyhow` (INFRA-02).

## Verification

- `cargo test -p cb-train one_hot_threshold` — **9 green** (Task 1 boundary unit).
- `cargo test -p cb-train --lib one_hot` — **11 green** (candidates + tree one-hot units).
- `cargo test -p cb-train --test one_hot_oracle_test` — **3 green** (path boundary +
  predict ≤1e-5 vs the oracle-locked float reference + no-permutation determinism).
- `cargo test -p cb-train --lib` — **50 green** (no regression in tree/candidates/tie-break).
- `cargo test -p cb-train --test slice_first_oracle_test` — **2 green** (the float
  reference the one-hot oracle anchors to is itself still locked vs upstream ≤1e-5).
- `cargo clippy -p cb-train --lib` — clean for this plan's code (only PRE-EXISTING
  `enum_variant_names` in cb-backend + `excessive_precision` in bootstrap.rs:134,
  neither introduced here — out of scope per the scope boundary).

## Deviations from Plan

### [Rule 3 - Blocking] One-hot oracle re-anchored from the `one_hot_cat` model.json to the oracle-locked float reference

- **Found during:** Task 2 (and flagged while reading context for Task 1).
- **Issue:** Plan Task 2 says "train the `one_hot_cat` fixture ... and assert
  splits/leaf-values/staged-approx/final-prediction ≤1e-5 vs upstream
  `model.json`" with "NO permutation present (D-04 isolation)". The committed
  05-01 `one_hot_cat/` fixture is the **CTR/permutation Wave-0 ANCHOR**: it has
  NO input feature columns, NO one-hot-only trained `model.json`, only
  permutation / CTR / ordered-approx `.npy` files; and its own `config.json`
  documents that because `cat1` cardinality (4) exceeds `one_hot_max_size` (3) a
  **permutation IS generated** (`learn_context.cpp:71-73`). It therefore cannot
  serve as a one-hot-ONLY, no-permutation D-04 isolation oracle as literally
  written — the very property RESEARCH Pitfall 2 says must hold.
- **Fix:** Implemented the D-04 oracle as **transcribe-then-self-oracle** (the same
  philosophy the phase's D-01 mechanism was revised to, 05-CONTEXT.md DECISION
  REVISION): a one-hot split is structurally a binary feature, so the one-hot-only
  model is anchored to the EXISTING `cb_train::train` float path on the equivalent
  one-hot binary columns. That float path is itself locked ≤1e-5 vs upstream
  catboost 1.2.10 in `slice_first_oracle_test.rs` (TRAIN-01/02/03), so the
  one-hot path inherits the upstream lock transitively — with no missing fixture
  and the NO-permutation property asserted structurally. The reference's continued
  upstream lock is re-verified in this plan (slice_first 2/2 green).
- **Files modified:** `crates/cb-train/tests/one_hot_oracle_test.rs` (rationale
  documented in its module header).
- **Commit:** `da4fb30`.

### [Rule 3 - Blocking] `BoostParams.one_hot_max_size` field propagation

- **Found during:** Task 1.
- **Issue:** Adding the required `one_hot_max_size` field to `BoostParams` broke
  every `BoostParams { ... }` struct literal (1 production in `catboost-rs/builder.rs`
  + 11 test literals across 9 cb-train test files).
- **Fix:** Pinned the production builder literal to `one_hot_max_size_default()`
  (the facade doesn't yet surface categorical config; numeric training never
  exercises the one-hot branch) and added the field to each test literal. No
  behavior change to any existing test (all still green).
- **Files modified:** `crates/catboost-rs/src/builder.rs` + the 9 cb-train test files.
- **Commit:** `392fe65`.

## Notes / Environment

- **Disk / link pressure (plan-flagged).** A `cargo test -p cb-train` that builds
  ALL integration-test binaries in parallel triggered the flagged mold LINK
  failure (`mold: fatal: cannot open .mold-XXX`) when several heavy
  polars/MLIR/cubecl test binaries linked concurrently — a disk-thrash artifact,
  NOT a compile error (every crate compiled clean). Mitigated by clearing
  `target/debug/incremental` (freed ~40G headroom) and scoping verification to
  per-binary commands (`--lib`, `--test one_hot_oracle_test`) exactly as the plan
  advises. No full-workspace test link was required to verify this plan.
- **Sandbox tmpfs.** The harness's sandbox task-output tmpfs filled mid-run
  (ENOSPC on a tiny RAM disk, unrelated to the 40G-free main disk); build/test
  commands were run with the sandbox disabled (local trusted dev box, only cargo
  build/test on this repo) so output capture used the main filesystem.
- **REQUIREMENTS pre-marked ORD-04 complete** from the prior plan's update; this
  plan actually delivers it (path selection unit-locked + predict ≤1e-5 oracle).

## Known Stubs

None. The CTR path (`EncodingPath::Ctr`) is a deliberate, documented routing
target for later waves (05-03+), not a stub — `route_categorical` returns it
correctly and the one-hot path is fully wired and oracle-locked.

## Self-Check: PASSED

All 3 created files and 14 modified files exist on disk; both task commits
(`392fe65`, `da4fb30`) are present in git history. Verified below.
