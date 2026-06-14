---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 11
subsystem: training
tags: [categorical, ctr, tensor-ctr, ordered-boosting, online-ctr, projection, catboost]

# Dependency graph
requires:
  - phase: 05-06
    provides: TProjection / combined_hash / fold_cat_hash / enumerate_projections / tensor_ctr_candidates
  - phase: 05-04
    provides: online_ctr_prefix_binclf (read-before-increment prefix) + calc_ctr_online_bin (Borders quantizer)
  - phase: 05-09
    provides: cb-model ModelSplit::Ctr representation + CtrSplitSpec (prior num/denom pair) + predict_raw_cat apply path
  - phase: 05-02
    provides: learn_set_cardinality / route_categorical / EncodingPath (OnLearnOnly cardinality)
provides:
  - materialize_ctr_feature + CtrFeatureColumn (combined-projection online CTR-feature column, prior num/denom pair)
  - ctr_border_count_default() == 15 (Borders CTR border count, pinned explicitly)
  - train_cat entry point (cat-aware training: OnLearnOnly cardinalities + per-candidate CTR-feature materialization)
  - private train_inner sharing the boosting loop (train/train_with_eval_sets delegate with empty cat columns)
affects: [05-12]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Materialization-then-carry: per-candidate CTR-feature columns are computed and carried on the iteration before any scoring (05-12 scores them)"
    - "Prior carried as a num/denom PAIR end-to-end (never a pre-divided scalar) so the 05-12 bake receives the denominator"
    - "train_inner shared-body refactor keeps the public train/train_with_eval_sets signatures byte-identical while adding the cat-aware train_cat entry"

key-files:
  created:
    - crates/cb-train/src/ctr/ctr_feature.rs
    - crates/cb-train/tests/ctr_feature_materialize_test.rs
  modified:
    - crates/cb-train/src/ctr/mod.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/lib.rs

key-decisions:
  - "materialize_ctr_feature reuses online_ctr_prefix_binclf + calc_ctr_online_bin verbatim — no re-derived CTR math (D-05)"
  - "CTR-eligible-position projection members are re-indexed to ABSOLUTE cat-feature indices before materialization so the right columns are read"
  - "cat-CTR learn permutation built with dynamic_body_tail=false (a single learn order over the whole set is sufficient for the online prefix; the Ordered split-scoring fold set keeps dynamic_body_tail=true)"
  - "the cat learn permutation is built ONLY when there are CTR candidates, leaving the numeric path's RNG draw stream untouched (train byte-identical)"

patterns-established:
  - "Combined-projection materialization: fold per-feature hashes -> first-seen dense bins -> read-before-increment online prefix -> Borders quantization (truncate+clamp to [0, border_count])"
  - "FOLDS-BUILT-ONCE extended to two create_folds sites (Ordered split-scoring fold + cat-CTR learn permutation), grep-bounded <= 2"

requirements-completed: [ORD-05]

# Metrics
duration: 18min
completed: 2026-06-14
---

# Phase 05 Plan 11: Categorical-CTR Ingestion + CTR-Feature Materialization Summary

**A new cat-aware `train_cat` entry point threads categorical columns into training — computing OnLearnOnly per-feature cardinalities and materializing a per-candidate combined-projection online CTR feature column (read-before-increment, Borders-quantized) the tree search can split on — while `train()` stays byte-identical.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-14
- **Completed:** 2026-06-14
- **Tasks:** 2 (Task 1 TDD: RED + GREEN; Task 2)
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

### Task 1 — Combined-projection online CTR-feature materialization (TDD)

- **RED (commit 4fe07b3):** `ctr_feature_materialize_test.rs` locks four behaviors — no-leakage prefix, combined-key (+ single-feature degeneration), quantization range, and prior num/denom pair — against a stub returning `Degenerate`; all three test functions compiled and failed.
- **GREEN (commit fe09f25):** `materialize_ctr_feature` + `CtrFeatureColumn` in `crates/cb-train/src/ctr/ctr_feature.rs`:
  1. Folds each document's per-feature `calc_cat_feature_hash` into the combined projection key via `TProjection::combined_hash` (members in sorted order; absolute-feature indexing).
  2. Remaps combined keys to dense first-seen bins (insertion-order `HashMap<u64, u32>`, bounded to `u32::MAX` with a typed `CbError::OutOfRange`).
  3. Derives the scalar online prior `prior_num / prior_denom` and runs the EXISTING `online_ctr_prefix_binclf` over the combined bins (no re-derived prefix loop).
  4. Quantizes each document's online CTR value to a Borders bin via `calc_ctr_online_bin`, truncated toward zero and clamped to `[0, ctr_border_count]`.
  - Carries the prior as the `prior_num` / `prior_denom` PAIR (matching `CtrSplitSpec` / `cb_model::CtrSplit`), and the raw online value as `ctr_value` for 05-12 scoring.
  - `ctr_border_count_default()` returns 15 (pinned).

### Task 2 — `train_cat` entry point (commit 5f0b678)

- New `train_cat<R: Runtime>(runtime, feature_values, feature_borders, cat_columns, target, weights, params, staged_out) -> CbResult<Model>`.
- Factored a private `train_inner` carrying `cat_columns`; `train` / `train_with_eval` / `train_with_eval_sets` are unchanged in signature/behavior and delegate with empty `cat_columns`.
- Replaced the hardcoded `cat_cardinalities = &[]` on the cat path with per-feature OnLearnOnly cardinalities via `learn_set_cardinality`, fed the REAL cat set to `tensor_ctr_candidates`, re-indexed CTR-eligible-position members to absolute cat indices, built the single learn permutation once (`create_folds`), and materialized a combined-projection CTR-feature column per candidate carrying the prior num/denom pair.
- Materialized columns are carried for Plan 05-12 scoring; the tree search does not yet split on them, so the numeric path is unaffected.

## Verification

All verification commands run PER-CRATE (disk-pressure constraint — never `cargo test --workspace`):

- `cargo test -p cb-train --test ctr_feature_materialize_test` — 3/3 green (no-leakage + combined-key + range + prior-pair).
- `cargo test -p cb-train --test slice_first_oracle_test --test one_hot_oracle_test --test leaf_methods_oracle_test --test ordered_boost_e2e_oracle_test` — all green (numeric/one-hot/ordered regression guard; `train()` byte-identical).
- `cargo test -p cb-train --lib` — 128/128 green.
- `cargo check -p cb-train --tests` — clean (train_cat wiring compiles).
- `grep -E "unwrap\(|expect\(|panic!|anyhow" crates/cb-train/src/ctr/ctr_feature.rs` — empty (no banned production patterns).

### Acceptance-criteria source assertions (all satisfied)

- `grep -c "pub fn materialize_ctr_feature" ctr_feature.rs` == 1; `prior_num`/`prior_denom` present; signature takes both (`prior_num: f64, prior_denom: f64` on one line); scalar derived as `prior_num / prior_denom`.
- `online_ctr_prefix_binclf` and `calc_ctr_online_bin` reused in `ctr_feature.rs`; `ctr_data` count == 0 (no model-hash-map leakage).
- `pub fn train_cat` == 1 in boosting.rs; `train_cat` re-exported from lib.rs; `learn_set_cardinality` present; `materialize_ctr_feature` called with the prior pair; `ctr_border_count_default` present.
- **FOLDS-BUILT-ONCE:** non-comment `create_folds(` count == 2 (one Ordered split-scoring fold set + one cat-CTR learn permutation) — within the documented `<= 2` bound.
- Hardcoded `cat_cardinalities: &[] = &[]` count == 0.

## Deviations from Plan

None — plan executed as written. Two clarifications worth recording (not deviations):

- The cat-CTR learn permutation is built with `dynamic_body_tail=false` (a single full-span learn order is what the online read-before-increment prefix consumes; the dynamic body/tail segmentation belongs to the Ordered split-scoring fold set, which keeps `dynamic_body_tail=true`).
- The plan's artifact spec named the struct field `ctr_value: Vec<f64>`; implemented exactly as named (the raw online prefix value per document), alongside the quantized `bins: Vec<u32>`.

## Known Stubs

None that block the plan's goal. The materialized `Vec<CtrFeatureColumn>` is intentionally CARRIED but NOT YET SCORED into the tree — this is the documented hand-off to Plan 05-12 (ORD-05 Part 2/2), which scores the columns into the oblivious search, bakes `ctr_data`, and closes the e2e oracle. This is by design per the plan's objective, not an unwired data path.

## Self-Check: PASSED

- FOUND: crates/cb-train/src/ctr/ctr_feature.rs
- FOUND: crates/cb-train/tests/ctr_feature_materialize_test.rs
- FOUND commit 4fe07b3 (RED test)
- FOUND commit fe09f25 (GREEN impl)
- FOUND commit 5f0b678 (train_cat)
