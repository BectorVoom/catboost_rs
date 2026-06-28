---
phase: 02-data-layer-pool-quantization-reduction
plan: 01
subsystem: testing
tags: [reduction, f64-summation, oracle-fixtures, ci-gate, arrow, polars, cityhash, borders, class-weights]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: cb-core crate (error, rng), cb-oracle generator (gen_inputs.py / gen_fixtures.py), pinned catboost==1.2.10 venv, source/test-separation + clippy CI gates
provides:
  - "cb-core::sum_f64 / sum_f32_in_f64 — the single sanctioned sequential f64 reduction primitive (DATA-07/D-07), order-locked"
  - "scripts/check-no-raw-float-sum.sh — D-08 CI grep gate banning all raw float .sum()/.fold(0.0,+) in library crates"
  - "arrow 59.0.0 + polars 0.54.4 wired into the workspace; cb-data depends on cb-core/thiserror/arrow/polars (no anyhow, D-14)"
  - "Wave-0 oracle fixtures: numeric_nan + explicit_categorical inputs; borders_quant/, cat_hash/, class_weights/ expected outputs"
  - "Resolved Assumptions A1–A5 (NaN sentinel, default border_count, integer-cat stringification, CityHash64 source) recorded in per-fixture config.json"
affects: [02-02-borders-quantization, 02-03-cat-hash, 02-04-class-weights, all downstream Phase-2 parity slices that route sums through cb-core]

# Tech tracking
tech-stack:
  added: [arrow 59.0.0, polars 0.54.4 (dtype-full, default-features=false)]
  patterns:
    - "Single audited summation primitive: every library sum routes through cb-core::sum_f64; all other raw float summation is CI-banned"
    - "Order-lock property test: adversarial [1e16, 1.0, -1e16] == 0.0 proves naive-sequential (not Kahan/pairwise)"
    - "Per-fixture verbatim oracle baselines: config-dependent behavior (NaN sentinel presence) pinned per fixture rather than by a fixed rule"

key-files:
  created:
    - crates/cb-core/src/reduction.rs
    - crates/cb-core/src/reduction_test.rs
    - scripts/check-no-raw-float-sum.sh
    - crates/cb-oracle/fixtures/borders_quant/ (config.json + borders .npy)
    - crates/cb-oracle/fixtures/cat_hash/ (config.json + cat_hashes/perfect_hash_bins .npy)
    - crates/cb-oracle/fixtures/class_weights/ (config.json + balanced/sqrt_balanced .npy)
    - crates/cb-oracle/fixtures/inputs/numeric_nan/
    - crates/cb-oracle/fixtures/inputs/explicit_categorical/
  modified:
    - crates/cb-core/src/lib.rs
    - .github/workflows/ci.yml
    - Cargo.toml
    - crates/cb-data/Cargo.toml
    - Cargo.lock
    - crates/cb-oracle/generator/gen_inputs.py
    - crates/cb-oracle/generator/gen_fixtures.py
    - .gitignore

key-decisions:
  - "A1/A3: get_borders() DOES surface the f32::MIN NanMode sentinel for NaN(Min) at default border budget, but presence is config-dependent (omitted at small budgets / nan_mode=Max) — pinned per fixture; Rust must match each fixture verbatim"
  - "A2: catboost 1.2.10 defaults are border_count=254, feature_border_type=GreedyLogSum, nan_mode=Min"
  - "A4: integer cat features stringify as PLAIN integers before hashing ('3' ui32=2658984922 != '3.0' ui32=1187060909)"
  - "A5: (string->ui32) hash vectors extracted from upstream model.json ctr_data hash_map; Rust must port util/digest/city.cpp (CityHash64 & 0xffffffff) bit-exactly, NOT use a third-party cityhash crate"
  - "reduction.rs is the single sanctioned hand-written summation loop; excluded from the D-08 grep gate alongside *_test.rs"
  - "cb-data deliberately omits anyhow (D-14, library-crate rule)"

patterns-established:
  - "Single audited f64 reduction primitive in cb-core; D-08 CI grep gate enforces it is the only one"
  - "Oracle fixtures are build-time-only (generator on dev machine, thread_count=1, pinned seed); CI consumes frozen .npy/config.json"
  - "Config-dependent oracle behavior is pinned per-fixture and compared verbatim, never assumed by a fixed rule"

requirements-completed: [DATA-07]

# Metrics
duration: ~10min
completed: 2026-06-13
---

# Phase 2 Plan 01: Reduction Primitive, D-08 Summation Gate & Wave-0 Oracle Fixtures Summary

**The single sequential-f64 reduction primitive (`cb-core::sum_f64`, order-locked via `[1e16,1.0,-1e16]==0.0`), a D-08 CI grep gate banning all other raw float summation, arrow/polars wired into the workspace, and Wave-0 oracle fixtures (numeric_nan + borders/cat-hash/class-weight scenarios) committed with Assumptions A1–A5 empirically resolved against catboost 1.2.10.**

## Performance

- **Duration:** ~10 min (task commits 12:35–12:45 +09:00); plan spanned a blocking human-verify checkpoint
- **Started:** 2026-06-13T12:35:02+09:00
- **Completed:** 2026-06-13
- **Tasks:** 3 (2 auto + 1 blocking human-verify checkpoint, approved)
- **Files modified:** 22 (across 3 task commits)

## Accomplishments
- `cb-core::sum_f64` / `sum_f32_in_f64`: the only sanctioned summation primitive — a naive left-to-right f64 fold (no Kahan/pairwise), re-exported from cb-core, with a dedicated `reduction_test.rs` whose order-lock test asserts `sum_f64(&[1e16, 1.0, -1e16]) == 0.0`.
- `scripts/check-no-raw-float-sum.sh`: D-08 CI grep gate that fails the build on any raw `.sum()`/`.fold(0.0,+)` in library crates, excluding only `reduction.rs` (sanctioned) and `*_test.rs`; wired into `.github/workflows/ci.yml` alongside the existing gates.
- arrow 59.0.0 and polars 0.54.4 (dtype-full, default-features=false) added to the workspace; `cb-data` now depends on cb-core + thiserror + arrow + polars (anyhow intentionally absent, D-14); `Cargo.lock` regenerated and committed (supply-chain pin).
- Wave-0 oracle fixtures generated from the pinned catboost==1.2.10 venv: `numeric_nan` (NaN float column) and `explicit_categorical` input datasets, plus `borders_quant/`, `cat_hash/`, and `class_weights/` expected-output scenarios.
- Assumptions A1–A5 empirically resolved and recorded in per-fixture `config.json` (see below).

## Resolved Assumptions A1–A5 (for downstream Phase-2 plans)

- **A1 / A3 (NaN sentinel in `get_borders()`):** EMPIRICAL — `get_borders()` DOES surface the NanMode `f32::MIN` sentinel (`-3.4028234663852886e+38`, `numeric_limits<float>::lowest`) as `borders[0]` for a NaN feature under `nan_mode=Min` at the default border budget. **CAVEAT:** sentinel inclusion is config-dependent — it tracks the realized border budget, not `nan_mode` alone; at small budgets the same NaN(Min) feature can OMIT the sentinel, and `nan_mode=Max` never prepends it. The Rust border oracle MUST compare per-fixture borders verbatim (sentinel present iff present at index 0), not assume a fixed rule. This fixture pins the default-param baseline (`numeric_nan`: sentinel PRESENT; `numeric_tiny`: ABSENT). Recorded in `borders_quant/config.json`.
- **A2 (default `border_count`):** catboost 1.2.10 default `border_count=254`, `border_selection_type=GreedyLogSum`, `nan_mode=Min`. Recorded per scenario in `borders_quant/config.json`.
- **A4 (integer-cat stringification):** integer categorical features stringify as PLAIN integers before `CalcCatFeatureHash` — `'3'` → ui32 `2658984922`, distinct from `'3.0'` → ui32 `1187060909`. Rust must hash the integer string form. Recorded in `cat_hash/config.json` with full `string_to_ui32` / `string_to_ui64_precursor` tables.
- **A5 (CityHash64 vectors source):** `(string → ui32)` vectors EXTRACTED from upstream catboost 1.2.10 (`model.json` `ctr_data` hash_map), NOT from a third-party cityhash crate. `hash_definition = CalcCatFeatureHash(s) = CityHash64(s) & 0xffffffff`. Rust's port of `util/digest/city.cpp` must reproduce `string_to_ui32` bit-exactly. Recorded in `cat_hash/config.json`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Reduction primitive in cb-core** — `1f2b9f1` (feat)
2. **Task 2: D-08 CI-grep gate + arrow/polars workspace deps** — `d92ae65` (feat)
3. **Task 3: Wave-0 oracle fixtures + resolve A1–A5 (blocking human-verify checkpoint, approved)** — `025c381` (feat)

_Note: Task 1 is TDD-style but the order-lock test and implementation landed in a single commit; the property test exercises RED→GREEN within `reduction_test.rs`._

## Files Created/Modified
- `crates/cb-core/src/reduction.rs` - `sum_f64` / `sum_f32_in_f64` sequential f64 fold primitive (D-07), with `//!` doc citing binarization.cpp:803-815 + calc_class_weights.cpp:36-54
- `crates/cb-core/src/reduction_test.rs` - one-`#[test]`-per-property test file; order-lock + empty + f32-in-f64 accumulation
- `crates/cb-core/src/lib.rs` - `mod reduction;` + `pub use reduction::{sum_f64, sum_f32_in_f64};` + test wiring
- `scripts/check-no-raw-float-sum.sh` - D-08 grep gate (excludes reduction.rs + *_test.rs)
- `.github/workflows/ci.yml` - invokes the new gate alongside check-no-anyhow / check-source-test-separation
- `Cargo.toml` - workspace deps arrow 59.0.0, polars 0.54.4
- `crates/cb-data/Cargo.toml` - cb-core/thiserror/arrow/polars deps (no anyhow)
- `Cargo.lock` - regenerated for arrow/polars (supply-chain pin)
- `crates/cb-oracle/generator/gen_inputs.py` - numeric_nan + explicit_categorical datasets
- `crates/cb-oracle/generator/gen_fixtures.py` - borders_quant / cat_hash / class_weights scenario emitters
- `crates/cb-oracle/fixtures/borders_quant/` - borders .npy + config.json (A1/A2/A3)
- `crates/cb-oracle/fixtures/cat_hash/` - cat_hashes/perfect_hash_bins .npy + config.json (A4/A5)
- `crates/cb-oracle/fixtures/class_weights/` - balanced/sqrt_balanced .npy + config.json
- `crates/cb-oracle/fixtures/inputs/{numeric_nan,explicit_categorical}/` - new input datasets
- `.gitignore` - ignore cb-oracle catboost_info/ training logs

## Decisions Made
See key-decisions frontmatter and the A1–A5 section above. All A1–A5 resolutions are empirical against catboost 1.2.10 and pinned in per-fixture config.json so downstream Rust ports compare verbatim.

## Deviations from Plan

None - plan executed exactly as written. The plan anticipated A1–A5 might require empirical resolution; the config-dependent NaN-sentinel finding (A1/A3) was captured as a per-fixture caveat exactly as the plan's "EMPIRICALLY whether get_borders() includes the sentinel" instruction directed, not as an unplanned deviation.

## Issues Encountered
None. The single subtlety — NaN-sentinel presence in `get_borders()` being budget-dependent rather than a fixed `nan_mode` rule — was resolved by pinning per-fixture verbatim baselines and documenting the caveat in `borders_quant/config.json` for downstream consumers.

## Checkpoint Handling
Task 3 was a `checkpoint:human-verify` (gate=blocking-human). Tasks 1–2 and the Task-3 fixture work were committed before the pause; the human reviewed the generated fixtures and the A1–A5 resolutions and responded "approved" with no discrepancies. Finalization (this SUMMARY + state advance) proceeded on approval.

## User Setup Required
None - no external service configuration required. (The oracle generator's `.venv` with pinned catboost==1.2.10 is dev-machine-only and was established in Phase 1.)

## Next Phase Readiness
- DATA-07 reduction primitive is live and order-locked; every downstream Phase-2 sum must route through it (D-08 enforces this).
- Wave-0 oracle baselines for borders, cat-hash, and class-weights are frozen and ready for the 02-02 / 02-03 / 02-04 parity slices to compare against verbatim.
- A1–A5 are resolved, so the borders/quantization, cat-hash (CityHash64 port), and class-weight implementations can proceed without open assumptions.
- arrow/polars are available for the cb-data ingestion plans.

## Self-Check: PASSED

All claimed created files exist on disk (reduction.rs, reduction_test.rs, check-no-raw-float-sum.sh, borders_quant/config.json, cat_hash/cat_hashes.npy, class_weights/balanced.npy, inputs/numeric_nan/X.npy) and all task commits (1f2b9f1, d92ae65, 025c381) are present in git history.

---
*Phase: 02-data-layer-pool-quantization-reduction*
*Completed: 2026-06-13*
