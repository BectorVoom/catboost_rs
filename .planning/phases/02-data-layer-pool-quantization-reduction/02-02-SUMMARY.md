---
phase: 02-data-layer-pool-quantization-reduction
plan: 02
subsystem: data-layer
tags: [pool, ingest, borders, quantization, greedy-logsum, oracle, parity, f32-f64]

# Dependency graph
requires:
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "cb-core::sum_f64 reduction primitive (D-07), D-08 grep gate, Wave-0 borders_quant fixtures, arrow/polars wired into cb-data"
provides:
  - "cb_data::Pool — owned float/cat/text/embedding columns + label/weights/group_id/subgroup_id/pairs/baseline (no lifetime generic, D-02)"
  - "cb_data::ingest::IngestSource trait seam (D-04) + OwnedColumns owned-Vec impl with typed length validation"
  - "cb_data::select_borders_greedy_logsum — GreedyLogSum priority-queue greedy binarizer, oracle-locked <=1e-5 per feature"
  - "cb_data::penalty_maxsumlog — the MaxSumLog penalty -(count+1e-8).ln()"
  - "Corrected borders_quant fixtures: RAW standalone GreedyLogSum quantization borders (49/49/49/49 numeric_tiny; 44/49/49 numeric_nan with f32::MIN sentinel)"
affects: [02-03-cat-hash, 02-04-class-weights, 02-05, all downstream quantization + tree-split plans consuming Pool and borders]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ingestion trait seam (IngestSource): Pool built through a validation boundary; owned-Vec now, borrowed view pluggable at Phase 8 without reshaping Pool"
    - "f64 penalty/score accumulators, f32 border midpoints (RESEARCH Pitfall 2): split scoring in f64, LeftBorder 0.5f*a+0.5f*b in f32"
    - "Priority-queue greedy as a linear-scan max over tiny bin sets (avoids BinaryHeap<f64> NaN/Ord hazards), bit-matching std::priority_queue tie-break (lb favored on score ties)"
    - "Oracle target is the STANDALONE quantizer output, never a trained model's pruned get_borders()"

key-files:
  created:
    - crates/cb-data/src/pool.rs
    - crates/cb-data/src/pool_test.rs
    - crates/cb-data/src/ingest/mod.rs
    - crates/cb-data/src/ingest/owned.rs
    - crates/cb-data/src/ingest/owned_test.rs
    - crates/cb-data/src/borders.rs
    - crates/cb-data/src/borders_test.rs
    - crates/cb-data/tests/borders_oracle_test.rs
  modified:
    - crates/cb-data/src/lib.rs
    - crates/cb-data/Cargo.toml
    - crates/cb-oracle/generator/gen_fixtures.py
    - crates/cb-oracle/fixtures/borders_quant/config.json
    - crates/cb-oracle/fixtures/borders_quant/numeric_tiny.borders.npy
    - crates/cb-oracle/fixtures/borders_quant/numeric_tiny.borders_per_feature.npy
    - crates/cb-oracle/fixtures/borders_quant/numeric_nan.borders.npy
    - crates/cb-oracle/fixtures/borders_quant/numeric_nan.borders_per_feature.npy
    - .gitignore
    - Cargo.lock

key-decisions:
  - "Pool is lifetime-free owned Vecs (D-02); a borrowed view plugs into IngestSource at Phase 8 rather than reshaping Pool"
  - "Greedy bin queue implemented as a linear-scan arg-max over a Vec<Bin> (bin count <= max_borders+1, tiny), sidestepping BinaryHeap<f64> Ord/NaN problems while preserving std::priority_queue tie-break (lb wins on score ties)"
  - "The unweighted greedy uses exact integer object counts, so the only float fold is the unit-weight total — routed through cb_core::sum_f64 to honor D-07/D-08 even though it equals the object count"
  - "Rule 1 fixture fix: borders_quant oracle was generated from a trained model's get_borders() (a training-PRUNED subset, ~11/11/7/15) which no standalone binarizer can reproduce; regenerated from Pool.quantize().save_quantization_borders() (raw 49/49/49/49). f32 sentinel snapped to exact f32::MIN (TSV text lost precision)"

patterns-established:
  - "Source/test separation: every module has a sibling *_test.rs (semicolon-form #[cfg(test)] mod), zero #[cfg(test)] brace bodies in production files"
  - "Oracle fixtures pin the RAW standalone quantizer output (the Rust parity target), not post-training pruned borders"

requirements-completed: [DATA-01, DATA-03]

# Metrics
duration: ~30min
completed: 2026-06-13
---

# Phase 2 Plan 02: Pool + IngestSource Seam + GreedyLogSum Border Oracle Summary

**The first parity vertical slice: an owned `Pool` (all CatBoost column kinds + metadata) built through the `IngestSource` validation seam, fed into a bit-transcribed GreedyLogSum priority-queue binarizer whose per-feature borders match upstream's standalone quantization to <=1e-5 on the frozen `numeric_tiny` and `numeric_nan` corpora, with all summation routed through the audited `cb_core::sum_f64` primitive.**

## Performance
- **Duration:** ~30 min (includes empirical root-cause + Rule-1 fixture regeneration)
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `auto`, both committed atomically)
- **Files:** 18 changed (10 created, 8 modified)

## Accomplishments

### Task 1 — Pool + IngestSource owned-Vec seam (DATA-01) — `7f70392`
- `cb_data::Pool`: owned float (SoA `Vec<Vec<f64>>`), categorical, text, embedding columns plus `label`, `weights`, `group_id`, `subgroup_id`, `pairs`, `baseline`. No lifetime generic / no `Cow` (D-02). Accessors return slices; `Pair { winner_id, loser_id }` for pairwise data.
- `cb_data::ingest::IngestSource` trait (D-04 seam) + `OwnedColumns` builder impl. `into_pool()` validates that every present column matches `n_rows` (derived from float feature 0 then label) and returns `CbError::OutOfRange` naming the offending column — never a panic or out-of-bounds index (threats T-02-04/T-02-05). No `unwrap`/`expect`/`panic`/`[]`-indexing in the module.
- Dedicated `pool_test.rs` (8 cases) + `ingest/owned_test.rs` (5 cases): length accessors, empty-default columns, typed mismatch errors for label/float/weights/cat/embedding, zero-row empty Pool, pair pass-through.

### Task 2 — GreedyLogSum binarizer + oracle (DATA-03) — `23012af`
- `cb_data::select_borders_greedy_logsum(column, max_borders, nan_sentinel)`: narrows f64→f32, drops NaNs, sorts; runs the priority-queue greedy (`Bin` = `TFeatureBin`) transcribed line-by-line from `binarization.cpp` (1320-1520); collects `LeftBorder` (f32 `0.5*a+0.5*b`) of every non-first bin into a sorted-dedup set; normalizes `-0.0f`→`+0.0f`; optionally prepends the NanMode `f32::MIN` sentinel.
- `cb_data::penalty_maxsumlog(count) = -(count + 1e-8).ln()` (f64).
- Scoring in f64, border midpoints in f32 (RESEARCH Pitfall 2). The unit-weight total is summed through `cb_core::sum_f64` (D-07/D-08).
- `borders_test.rs` (5 cases): penalty value, duplicate-column collapse, `-0.0→+0.0` (sign-bit checked), 2-value/1-border midpoint, sentinel prepend.
- `tests/borders_oracle_test.rs`: per-feature `compare_stage(Stage::Borders, expected, actual)` for `numeric_tiny` (4 features, no sentinel) and `numeric_nan` (feature 0 sentinel) — both pass at <=1e-5.

## Verification (all green)
- `cargo test -p cb-data` — 16 unit + 2 oracle tests pass.
- `cargo test -p cb-data --test borders_oracle_test` — numeric_tiny + numeric_nan match oracle <=1e-5.
- `bash scripts/check-no-raw-float-sum.sh` — exits 0 (no raw float fold in cb-data).
- `cargo clippy -p cb-data --lib -- -D warnings` — clean (also clean `--all-targets` for cb-data).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] borders_quant oracle fixtures captured TRAINING-PRUNED borders, not raw quantization borders**
- **Found during:** Task 2 (oracle test brought the existing fixture under a real Rust comparison for the first time).
- **Issue:** Plan 02-01's `gen_fixtures.py` generated `numeric_tiny.borders.npy` from a *trained model's* `get_borders()`, which returns only the borders actually used by some tree split (a training-dependent PRUNED subset — 11/11/7/15 for numeric_tiny). No standalone GreedyLogSum binarizer can reproduce that subset; empirically the Rust port (bit-validated against `Pool.quantize().save_quantization_borders()` at maxdiff ~5e-10) produces the full 49/49/49/49 set. The fixture was therefore an unreachable oracle target.
- **Fix:** Rewrote `_extract_borders` + `gen_borders_quant` to source borders from the STANDALONE quantizer (`Pool.quantize(border_count=254, feature_border_type='GreedyLogSum', nan_mode='Min').save_quantization_borders()`), parsing the 2-3 column TSV (NaN features carry a trailing `Min` annotation). Regenerated `numeric_tiny` (49/49/49/49) and `numeric_nan` (44/49/49, feature-0 `f32::MIN` sentinel). Updated `config.json` A1/A3/A2 text + added `borders_source`.
- **Files modified:** `crates/cb-oracle/generator/gen_fixtures.py`, `crates/cb-oracle/fixtures/borders_quant/{numeric_tiny,numeric_nan}.borders*.npy`, `crates/cb-oracle/fixtures/borders_quant/config.json`
- **Commit:** `23012af`

**2. [Rule 1 - Bug] f32 NanMode sentinel serialized with truncated precision**
- **Found during:** Task 2 oracle test (numeric_nan feature 0 diverged 3.85e28 at index 0).
- **Issue:** `save_quantization_borders()` writes the sentinel as ~10-significant-figure text (`-3.402823466e+38`), which widens to a different f64 than the exact `f64::from(f32::MIN)` the Rust port emits (`-3.4028234663852886e+38`, bits `c7efffffe0000000`).
- **Fix:** In `_extract_borders`, snap any border at sentinel magnitude to the exact `np.finfo(np.float32).min` (verified bit-identical to Rust `f64::from(f32::MIN)`).
- **Files modified:** `crates/cb-oracle/generator/gen_fixtures.py` (+ regenerated npy).
- **Commit:** `23012af`

**3. [Rule 3 - Housekeeping] gitignore Python bytecode cache**
- **Issue:** Importing the generator emitted an untracked `crates/cb-oracle/generator/__pycache__/`.
- **Fix:** Added `__pycache__/` patterns to `.gitignore`.
- **Commit:** `23012af`

## Known Stubs
None. Pool wires real owned columns; borders.rs is the full greedy algorithm (no placeholder). Categorical/text features are stored as raw strings (hashing is the 02-03 cat-hash plan's scope, not a stub).

## Threat Flags
None. No new network/auth/file-access surface. The two trust boundaries in the plan's threat model (column-length mismatch → typed error; f32/f64 border arithmetic) are both mitigated and test-locked (T-02-04/05 by typed errors, T-02-06 by the f32 midpoints + oracle gate + `-0.0`/duplicate unit cases + the D-08 grep gate).

## Notes for Downstream Plans
- The corrected `borders_quant` fixtures are the RAW standalone quantizer output. Any future plan comparing against them must run the standalone GreedyLogSum path, NOT a trained model's `get_borders()`.
- `select_borders_greedy_logsum` currently implements the UNWEIGHTED path. A weighted variant (cumulative-weight bins, `TWeightedFeatureBin`) is a future extension; the `total_object_weight` reduction hook is already in place.
- `Pool`/`IngestSource` are ready for the cat-hash (02-03) and class-weights (02-04) plans to extend.

## Self-Check: PASSED
- Created files exist: pool.rs, pool_test.rs, ingest/{mod,owned,owned_test}.rs, borders.rs, borders_test.rs, tests/borders_oracle_test.rs — all present.
- Commits present: `7f70392` (Task 1), `23012af` (Task 2) — both in git history.
- All verification commands green (16+2 tests, D-08 gate, clippy).

---
*Phase: 02-data-layer-pool-quantization-reduction*
*Completed: 2026-06-13*
