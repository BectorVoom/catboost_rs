---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Phase 3 context gathered
last_updated: "2026-06-13T07:04:44.265Z"
last_activity: 2026-06-13 -- Plan 02-04 complete (CityHash64 port + CalcCatFeatureHash + first-seen perfect-hash remap, oracle-locked; corrected cat_hash fixtures)
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 8
  completed_plans: 8
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-13)

**Core value:** A memory-efficient, Rust-native CatBoost implementation with verifiable feature parity (oracle-tested ≤1e-5), embeddable in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.
**Current focus:** Phase 02 — data-layer-pool-quantization-reduction

## Current Position

Phase: 02 (data-layer-pool-quantization-reduction) — EXECUTING
Plan: 5 of 5
Status: Ready to execute (02-04 complete)
Last activity: 2026-06-13 -- Plan 02-04 complete (CityHash64 port + CalcCatFeatureHash + first-seen perfect-hash remap, oracle-locked; corrected cat_hash fixtures)

Progress: [████████░░] 80% (4 of 5 phase-02 plans complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 01 P01 | 5 | 4 tasks | 21 files |
| Phase 01 P02 | 4min | 1 tasks | 4 files |
| Phase 01 P03 | 9min | 3 tasks | 42 files |
| Phase 02 P01 | 10min | 3 tasks | 22 files |
| Phase 02 P02 | 30min | 2 tasks | 18 files |
| Phase 02 P03 | 5 | 2 tasks | 7 files |
| Phase 02 P04 | 25min | 2 tasks | 11 files |
| Phase 02 P05 | ~25min | 2 tasks | 12 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Phased by oracle-passing vertical slices, narrowest-first (research-mandated); each phase must be oracle-passing ≤1e-5 vs upstream before the next begins.
- Roadmap: CPU path fully oracle-locked (through Phase 6) before GPU (Phase 7); GPU is additive on the generic `R: Runtime` boundary established in Phase 3.
- [Phase ?]: Plan 01-01: pinned approx to stable 0.5 (not 0.6.0-rc2 pre-release); test-only dev-dep
- [Phase ?]: Plan 01-01: committed Cargo.lock for supply-chain integrity (T-01-SC)
- [Phase ?]: Plan 01-01: uniform in-code test-lint exemption + --lib CI clippy gate (Pitfall 1)
- [Phase ?]: Plan 01-02: TFastRng64 ported bit-for-bit; two PCG streams deduped into shared Lcg32 (bitstream-identical, oracle-proven)
- [Phase ?]: Plan 01-02: derived Clone/PartialEq/Eq on CbError to enable Result equality assertions (backward-compatible)
- [Phase ?]: INFRA-04 compare_stage ships API + real-fixture read + 1e-5 gate in P1; comparison vs Rust-computed actuals deferred to P3/P4
- [Phase 02]: Plan 02-01: single sequential f64 reduction primitive (`cb-core::sum_f64`/`sum_f32_in_f64`) order-locked via `[1e16,1.0,-1e16]==0.0`; D-08 CI grep bans all other float summation
- [Phase 02]: Plan 02-01 A1/A3 (RESOLVED): `get_borders()` surfaces the `f32::MIN` NanMode sentinel for NaN(Min) features at the default border budget; presence is config-dependent (omitted at small budgets / nan_mode=Max), so pinned per-fixture in `borders_quant/config.json` — Rust must match each fixture verbatim
- [Phase 02]: Plan 02-01 A2 (RESOLVED): catboost 1.2.10 default `border_count=254`, `feature_border_type=GreedyLogSum`, `nan_mode=Min`
- [Phase 02]: Plan 02-01 A4 (RESOLVED): integer cat features stringify as PLAIN integers before `CalcCatFeatureHash` (`'3'` ui32=2658984922 ≠ `'3.0'` ui32=1187060909)
- [Phase 02]: Plan 02-01 A5 (RESOLVED): `(string→ui32)` hash vectors extracted from upstream model.json `ctr_data` hash_map; Rust must port `util/digest/city.cpp` (CityHash64 & 0xffffffff) to reproduce them bit-exactly (no third-party crate)
- [Phase ?]: [Phase 02]: Plan 02-01 COMPLETE — sum_f64/sum_f32_in_f64 reduction primitive shipped + order-locked; D-08 grep gate live; arrow 59 / polars 0.54 wired; Wave-0 borders/cat-hash/class-weight fixtures committed; A1-A5 resolved
- [Phase ?]: [Phase 02]: Plan 02-02: Pool is lifetime-free owned Vecs (D-02); IngestSource trait seam validates column lengths with typed CbError; borrowed view plugs in at Phase 8 without reshaping Pool
- [Phase ?]: [Phase 02]: Plan 02-02: GreedyLogSum binarizer bit-transcribed from binarization.cpp (f64 penalty/score, f32 border midpoints), oracle-locked <=1e-5 per feature; sums routed through cb_core::sum_f64
- [Phase ?]: [Phase 02]: Plan 02-02 (Rule 1 fix): borders_quant fixtures regenerated from STANDALONE Pool.quantize().save_quantization_borders() (raw 49/49/49/49) instead of training-pruned get_borders(); f32 sentinel snapped to exact f32::MIN
- [Phase ?]: Per-feature NanMode: NaN-bearing column -> Min sentinel, NaN-free -> Forbidden
- [Phase ?]: Float bin width hard-capped at u16 -> CbError not panic; u32 categorical-only (utils.h:175-181)
- [Phase 02]: Plan 02-04: CityHash64 ported bit-exact from vendored util/digest/city.cpp (Yandex CityHash 1.0, NOT mainline/third-party crate); CalcCatFeatureHash = city_hash_64 & 0xffffffff; first-seen perfect-hash bins (bin = map.size()), uniq count bounded to u32::MAX with typed CbError (no panic)
- [Phase 02]: Plan 02-04 (Rule 1 fix): cat_hash string_to_ui32 fixtures regenerated from a standalone C++ tool transcribing the vendored city.cpp (generator/cityhash_oracle.cpp) -- the Wave-0 vectors had been extracted from a trained model's ctr_data hash_map (CTR-projection hashes, NOT CalcCatFeatureHash). 'alpha'=1296865003 (was 3214079027); '3'=593172586 (was 2658984922). Downstream cat-hash consumers must use cb_data::calc_cat_feature_hash, never a model's ctr_data hash_map.
- [Phase 02]: Plan 02-05: Polars rides the shared Arrow validator (rechunk -> cont_slice -> arrow::Float64Array -> arrow_f64_column) to avoid polars/arrow-crate type incompatibility while honoring the rechunk->Arrow key_link (D-05)
- [Phase 02]: Plan 02-05: ingestion CbError variants (Dtype/LengthMismatch/NanInCategorical/Ingestion) stringify external arrow/polars errors (no #[from]) so the enum keeps Clone+PartialEq+Eq (Shared Pattern C / D-06); this is the taxonomy Phase 8 maps to Python exceptions (PYAPI-05)
- [Phase 02]: Plan 02-05: class weights computed in f32 to bit-match upstream float lambdas (SqrtBalanced fixture is f32 sqrt(3) widened, absorbed by <=1e-5, fixture unchanged); 1e-8 floor returns 1.0 on an empty/degenerate class (no div-by-zero); all summary sums via cb_core::sum_f64
- [Phase 02]: Plan 02-05 COMPLETE — DATA-06 (Arrow+Polars validated ingestion) + DATA-08 (Balanced/SqrtBalanced + per-object/per-class weights) shipped, oracle-locked; Phase 2 data layer complete

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- Phase 5 (Ordered Boosting/CTR), Phase 7 (GPU/CubeCL-ROCm), and Phase 8 (Python ABI/packaging) are flagged NEEDS DEEPER RESEARCH — run the per-phase research spike before planning each.
- GPU tolerance epsilon (Phase 7) is unspecified — must be set and signed off before Phase 7 planning.
- **Plan 02-01 COMPLETE (human approved Task-3 checkpoint).** Tasks 1–3 committed (1f2b9f1, d92ae65, 025c381); 02-01-SUMMARY.md written and self-checked; plan counter advanced to 02-02. No open blockers from 02-01.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-13T06:08:54.486Z
Stopped at: Phase 3 context gathered
Resume file: .planning/phases/03-cpu-training-core-plain-boosting-oblivious-trees/03-CONTEXT.md
