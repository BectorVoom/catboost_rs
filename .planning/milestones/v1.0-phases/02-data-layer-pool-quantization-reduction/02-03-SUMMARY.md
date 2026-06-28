---
phase: 02-data-layer-pool-quantization-reduction
plan: 03
subsystem: data-layer
tags: [quantization, nan-mode, sentinel, bin-assignment, quantized-pool, soa, width-enum, oracle, parity]

# Dependency graph
requires:
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "cb_data::Pool (owned SoA float columns), select_borders_greedy_logsum (GreedyLogSum + optional f32::MIN sentinel), borders_quant numeric_nan fixtures (RAW standalone quantizer output), cb_core::sum_f64 (D-08 gate)"
provides:
  - "cb_data::NanMode { Min, Max, Forbidden } enum (enums.h:107-111 order) + prepends_min_sentinel/appends_max_sentinel/reserved_border_budget helpers"
  - "cb_data::bin_of(&[f32], f32) -> u32 — strict value>border bin assignment (utils.h:28-49), equal value -> lower bin; linear <=64 / partition_point otherwise"
  - "cb_data::insert_sentinel / nan_bin — Min prepends f32::MIN, Max appends f32::MAX; NaN -> 0 (Min/Forbidden) / top (Max)"
  - "cb_data::ColumnBins { U8/U16/U32 } zero-widening width enum + QuantizedPool immutable SoA (per-feature bins/borders/NanMode, read-only D-03)"
  - "cb_data::select_bin_width / pack_bins / FeatureKind — CalcHistogramWidthForBorders (utils.h:175-181): float capped at u16, u32 cat-only"
  - "cb_data::Pool::quantize(&QuantizeParams) -> CbResult<QuantizedPool> driver (D-01) composing borders:: + nan_mode::"
affects: [02-04-class-weights, 02-05, all downstream tree-split/training plans consuming QuantizedPool binned columns]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-feature NanMode resolution: a float column with any NaN quantizes under the requested nan_mode (Min); a NaN-free column is Forbidden (no sentinel) — matches the oracle's per-feature sentinel presence"
    - "Zero-widening bin storage: each binned column picks the narrowest unsigned int (u8/u16) that holds its index; float NEVER u32 (memory efficiency as a first-class constraint)"
    - "Fallible width selection: a float feature needing >u16 returns None -> CbError, never a panic/overflow (threat T-02-09)"
    - "quantize reuses select_borders_greedy_logsum verbatim so quantized borders match the standalone-quantizer oracle without re-deriving the sentinel"

key-files:
  created:
    - crates/cb-data/src/nan_mode.rs
    - crates/cb-data/src/nan_mode_test.rs
    - crates/cb-data/src/quantized_pool.rs
    - crates/cb-data/src/quantized_pool_test.rs
    - crates/cb-data/src/quantize.rs
    - crates/cb-data/tests/quantize_oracle_test.rs
  modified:
    - crates/cb-data/src/lib.rs

key-decisions:
  - "Per-feature NanMode: NaN-bearing column -> params.nan_mode (Min default); NaN-free column -> Forbidden (no sentinel). This reproduces the oracle's per-feature sentinel presence (numeric_nan feature 0 sentinel, features 1-2 none) without a global flag"
  - "bin_of dual path: linear count for <=64 borders, partition_point for >64 — both yield the identical strict-> index; a 100-border test asserts parity between the two paths"
  - "Float bin width is hard-capped at u16 (select_bin_width returns None for float >=65536), surfaced as CbError::OutOfRange by the quantize driver rather than a panic (T-02-09)"
  - "quantize narrows the f64-stored (f32-valued) borders back to f32 before bin_of so value and border are compared in the same width (f32), preventing boundary drift"

patterns-established:
  - "Source/test separation: nan_mode.rs/quantized_pool.rs each have a dedicated sibling *_test.rs; oracle parity lives in tests/quantize_oracle_test.rs"
  - "Quantization composes the audited sub-primitives (borders::, nan_mode::) rather than re-deriving them, so the oracle gate on borders transitively validates the quantize path"

requirements-completed: [DATA-02, DATA-04]

# Metrics
duration: ~5min
completed: 2026-06-13
---

# Phase 2 Plan 03: Float-Feature Quantization Vertical Slice (NanMode + QuantizedPool) Summary

**The float path from raw `Pool` to immutable binned `QuantizedPool`: a `NanMode` enum with strict `value > border` bin assignment (equal value -> lower bin) and `f32::MIN`/`f32::MAX` sentinel handling, the zero-widening `ColumnBins { U8/U16/U32 }` SoA (float capped at u16, u32 categorical-only), and a `pool.quantize(&params) -> CbResult<QuantizedPool>` driver that composes GreedyLogSum borders with NaN handling — oracle-locked on the `numeric_nan` corpus to <=1e-5 with NaN rows landing in bin 0.**

## Performance
- **Duration:** ~5 min (12:26–12:31 UTC; two atomic task commits)
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `auto`, both TDD-style, both committed atomically)
- **Files:** 7 changed (6 created, 1 modified)

## Accomplishments

### Task 1 — NanMode + strict bin assignment (DATA-04) — `0101717`
- `cb_data::NanMode { Min, Max, Forbidden }` (`#[derive(Debug, Clone, Copy, PartialEq, Eq)]`, per-variant docs) mirroring upstream `enums.h:107-111` variant order.
- `bin_of(borders: &[f32], value: f32) -> u32`: the strict `value > border` count (`utils.h:28-49`). A value exactly equal to a border lands in the LOWER bin (the equal border is not counted). Linear count for `<= 64` borders, `partition_point(|&b| b < value)` for `> 64` — a 100-border test asserts the two paths agree.
- `insert_sentinel`: `Min` prepends `f32::MIN`, `Max` appends `f32::MAX`, `Forbidden` none. `reserved_border_budget`: `Min`/`Max` reserve one (`borderCount - 1`, saturating), `Forbidden` none. `nan_bin`: NaN -> `0` for `Min`/`Forbidden`, `borders.len()` (top) for `Max` (`utils.h:51-66`).
- `nan_mode_test.rs` (7 cases): boundary-equality lower-bin, linear/binary-search parity, Min/Max/Forbidden sentinel insertion, NaN bin placement, reserved budget.
- No `unwrap`/`panic`/indexing in the production module.

### Task 2 — QuantizedPool SoA + ColumnBins + quantize driver + oracle (DATA-02) — `04a3841`
- `cb_data::ColumnBins { U8(Vec<u8>), U16(Vec<u16>), U32(Vec<u32>) }` zero-widening width enum with read-only `len`/`is_empty`/`get`/`to_u32_vec`.
- `cb_data::QuantizedPool`: immutable per-feature SoA (`Vec<ColumnBins>` bins + `Vec<Vec<f32>>` borders + `Vec<NanMode>`), read accessors only (no mutable scratch, D-03).
- `select_bin_width` / `pack_bins` / `FeatureKind`: `CalcHistogramWidthForBorders` (`utils.h:175-181`) — `<256` -> U8, `<65536` -> U16; a FLOAT feature with `>=65536` borders returns `None` (never u32); `u32` is categorical perfect-hash only.
- `Pool::quantize(&QuantizeParams) -> CbResult<QuantizedPool>` (D-01): per float feature determines the NanMode (NaN-bearing -> `params.nan_mode`; NaN-free -> `Forbidden`), selects borders via `select_borders_greedy_logsum` (`borders::`, sentinel prepended for `Min`), assigns each value via `bin_of`/`nan_bin` (`nan_mode::`) comparing in f32, and packs into the width-selected `ColumnBins`. A float feature exceeding u16 surfaces `CbError::OutOfRange` (T-02-09), never a panic.
- `QuantizeParams` (catboost 1.2.10 defaults: `border_count=254`, `nan_mode=Min`).
- `quantized_pool_test.rs` (6 cases): width selection per arm, float-u32 rejection, lossless round-trip per arm, out-of-range `get`, end-to-end strict-`>` bins, NaN-in-bin-0.
- `tests/quantize_oracle_test.rs`: quantizes `numeric_nan` end-to-end, gates each feature's borders with `compare_stage(Stage::Borders, ...)` against the standalone-quantizer oracle (feature 0 = 44 borders incl. `f32::MIN` sentinel, features 1-2 = 49 each), and asserts the 6 NaN rows quantize to bin 0 and NaN-free features carry no sentinel — green at <=1e-5.

## Verification (all green)
- `cargo test -p cb-data` — 29 unit (7 nan_mode + 6 quantized_pool + prior 16) + 2 borders-oracle + 1 quantize-oracle, all pass.
- `cargo test -p cb-data --test quantize_oracle_test` — numeric_nan borders match oracle <=1e-5, NaN bins placed.
- `bash scripts/check-no-raw-float-sum.sh` — exits 0 (the only float fold is the unit-weight total inside `borders.rs`, already routed through `cb_core::sum_f64`; quantize adds no raw fold).
- `cargo clippy -p cb-data --lib -- -D warnings` — clean.

## Deviations from Plan

None - plan executed exactly as written. The per-feature NanMode resolution (NaN-bearing -> Min, NaN-free -> Forbidden) was the plan's intended composition of the borders-sentinel decision (already config-dependent per the 02-01/02-02 A1/A3 caveat) and is not an unplanned deviation.

## Deferred Issues (out of scope)

- `cargo clippy -p cb-data --all-targets -- -D warnings` fails on a PRE-EXISTING lint in `cb-oracle` (`compare.rs:44`, `neg_cmp_op_on_partial_ord`), a dev-dependency dragged in by cb-data's integration tests. The lint was introduced by Phase-01 commit `902368d` (the deliberate NaN/Inf-as-divergence `!(diff <= tol)`) and only surfaces under rust-1.96.0. The plan's required gate is `--lib` (clean). Logged to `deferred-items.md`; recommend a cb-oracle housekeeping fix. Not caused by this plan's changes (SCOPE BOUNDARY).

## Known Stubs
None. `quantize` runs the full borders+NaN+pack pipeline (no placeholder). Categorical/text/embedding quantization is out of this plan's scope (float vertical slice); the cat perfect-hash path is the cat-hash plan's responsibility, and `FeatureKind::Categorical`/`ColumnBins::U32` are already in place for it.

## Threat Flags
None. No new network/auth/file-access surface. All three threat-register mitigations are implemented and test-locked: T-02-07 (strict `>` — boundary-equality lower-bin test), T-02-08 (NaN sentinel off-by-one — `reserved_border_budget` + oracle border-count match 44/49/49), T-02-09 (bin-index overflow / width arm — float-u32 rejection test + `CbError` path + `partition_point`/checked casts).

## Notes for Downstream Plans
- `QuantizedPool` is immutable after build (D-03): it exposes read accessors only. The trainer must allocate its own scratch.
- `quantize` currently handles FLOAT features only (the plan's vertical slice). Categorical perfect-hash binning (`FeatureKind::Categorical` -> `ColumnBins::U32`) is wired into the width enum but not yet driven by `quantize`; the cat-hash plan extends the driver.
- Per-feature NanMode is `Min` for NaN-bearing columns and `Forbidden` for NaN-free ones; a `Max` request is honored by `quantize` (appends `f32::MAX`, NaN -> top bin) but no oracle fixture exercises `Max` yet.

## Self-Check: PASSED
- Created files exist: nan_mode.rs, nan_mode_test.rs, quantized_pool.rs, quantized_pool_test.rs, quantize.rs, tests/quantize_oracle_test.rs — all present.
- Commits present: `0101717` (Task 1), `04a3841` (Task 2) — both in git history.
- All verification commands green (29+2+1 tests, D-08 gate, --lib clippy).

---
*Phase: 02-data-layer-pool-quantization-reduction*
*Completed: 2026-06-13*
