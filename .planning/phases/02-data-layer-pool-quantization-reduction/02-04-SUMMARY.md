---
phase: 02-data-layer-pool-quantization-reduction
plan: 04
subsystem: data-layer
tags: [cat-hash, cityhash64, calccatfeaturehash, perfect-hash, first-seen-bins, oracle, parity, integer-exact]

# Dependency graph
requires:
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "Wave-0 cat_hash fixtures (cat_hashes.npy / perfect_hash_bins.npy / config.json), explicit_categorical input corpus, cb-oracle load_f64_vec harness, A4 integer-cat stringification resolution"
provides:
  - "cb_data::city_hash_64 — bit-exact Rust port of vendored util/digest/city.cpp (Yandex CityHash 1.0)"
  - "cb_data::calc_cat_feature_hash — CalcCatFeatureHash(s) = city_hash_64(s) & 0xffffffff (cat_feature.cpp:6-8)"
  - "cb_data::stringify_int_category — A4 plain-integer stringification ('3' != '3.0')"
  - "cb_data::PerfectHash + perfect_hash_bins — first-seen perfect-hash remap (bin = map.size()), uniq-count bounded to u32::MAX with a typed CbError on overflow (no panic)"
  - "generator/cityhash_oracle.cpp — authoritative CalcCatFeatureHash oracle tool transcribed from vendored city.cpp; corrected cat_hash fixtures"
affects: [02-05, Phase-5-CTR (categorical statistics build on these bins), all downstream categorical-feature parity slices]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bit-exact C++-port discipline (mirrors cb-core::rng): wrapping_* arithmetic over explicit u64, // C++-line citations, non-cryptographic caveat doc block, integer-exact assert_eq! vectors (NOT <=1e-5)"
    - "Authoritative oracle = vendored source, not a trained-model artifact: CalcCatFeatureHash computed by a standalone C++ tool transcribed from util/digest/city.cpp (the algorithm the live catboost library compiles), not extracted from ctr_data hash_map (which stores CTR-projection hashes)"
    - "First-seen perfect-hash via HashMap<u32,u32> + insertion counter (bin = current map size); the upstream TMap sorted order matters only for the deferred most-frequent-to-0 tiebreak, out of Phase-2 scope"
    - "u32::MAX uniq-cat bound enforced with a typed CbError::OutOfRange, never a panic (Security V5 / T-02-11); overflow path tested via an explicit small cap"

key-files:
  created:
    - crates/cb-data/src/cat_hash.rs
    - crates/cb-data/src/cat_hash_test.rs
    - crates/cb-data/tests/cat_hash_oracle_test.rs
    - crates/cb-oracle/generator/cityhash_oracle.cpp
  modified:
    - crates/cb-data/src/lib.rs
    - crates/cb-data/Cargo.toml
    - crates/cb-oracle/generator/gen_fixtures.py
    - crates/cb-oracle/fixtures/cat_hash/cat_hashes.npy
    - crates/cb-oracle/fixtures/cat_hash/config.json
    - Cargo.lock

key-decisions:
  - "A5 CORRECTION (Rule 1 fixture fix): the Wave-0 cat_hash string_to_ui32 vectors were extracted from a trained model's ctr_data hash_map, which stores CTR-PROJECTION hashes (CalcHashes + MultiHash + priors, index_hash_calcer.h), NOT raw CalcCatFeatureHash. Those were the wrong oracle target for a CityHash64 port. Regenerated from a standalone C++ tool transcribing the vendored util/digest/city.cpp -- the authoritative algorithm the live catboost library compiles. 'alpha' is now 1296865003 (was 3214079027); '3' is 593172586 (was 2658984922)."
  - "CityHash variant is the Yandex CityHash 1.0 (city.h:8-9: results differ from mainline CityHash); ported from scratch, NO third-party cityhash crate (RESEARCH Pitfall 4 / T-02-10, T-02-SC)"
  - "Little-endian unaligned loads (load64/load32) mirror ReadUnaligned<uiN> on the LE targets catboost ships; clippy-clean via slice .get() (no panicking index) and u64::rotate_right (exact Rotate/RotateByAtLeast1 equivalent)"
  - "First-seen bins are plain-training order; the TMap sorted RB-tree order is only relevant to the deferred mapMostFrequentValueTo0 tiebreak (out of Phase-2 scope, noted in code)"

patterns-established:
  - "Oracle correctness: regenerate a mis-captured fixture from the authoritative vendored source (precedent set by 02-02's borders_quant fix)"
  - "Source/test separation held: cat_hash.rs has zero #[cfg(test)] bodies; a pub(crate) remap_bounded seam lets the overflow/CB_ENSURE path be tested without materializing u32::MAX hashes"

requirements-completed: [DATA-05]

# Metrics
duration: ~25min
completed: 2026-06-13
---

# Phase 2 Plan 04: CityHash64 Port, CalcCatFeatureHash & First-Seen Perfect-Hash Oracle Summary

**A bit-exact Rust port of Yandex CatBoost's CityHash 1.0 (`util/digest/city.cpp`), the `CalcCatFeatureHash = CityHash64(bytes) & 0xffffffff` reduction, and the first-seen perfect-hash remap (`bin = map.size()`) that turns category strings into dense bins — validated integer-exact against per-object hash and bin oracles on the explicit-categorical corpus, after correcting the Wave-0 cat_hash fixtures that had captured CTR-projection hashes instead of the raw `CalcCatFeatureHash`.**

## Performance
- **Duration:** ~25 min (includes empirical root-cause of the mislabeled oracle vectors + Rule-1 fixture regeneration)
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `auto` with `tdd="true"`, both committed atomically)
- **Files:** 11 changed (4 created, 7 modified) across 2 task commits

## Accomplishments

### Task 1 — CityHash64 + CalcCatFeatureHash bit-exact port (DATA-05) — `a791c89`
- `cb_data::city_hash_64`: a line-by-line transcription of the vendored `util/digest/city.cpp` (CityHash 1.0), covering every length path — `HashLen0to16`, `HashLen17to32`, `HashLen33to64`, and the `>64`-byte 56-byte-state block loop. All arithmetic uses `wrapping_*`; each step carries a `//` citation of the C++ line; the module opens with the non-cryptographic caveat doc block (mirroring `cb-core::rng`).
- `cb_data::calc_cat_feature_hash(s) = (city_hash_64(s.as_bytes()) & 0xffffffff) as u32` (`cat_feature.cpp:6-8`).
- `cb_data::stringify_int_category(i64)`: the A4 plain-integer form (`3 -> "3"`, distinct from `"3.0"`).
- `cat_hash_test.rs`: 9 bit-exact `(string -> ui64/ui32)` `assert_eq!` vectors spanning empty / `<16` / 16-byte boundary / 17-32 / 33-64 / `>64` multi-block, plus the A4 `'3' != '3.0'` proof.
- **[Rule 1 fixture fix]** Discovered the Wave-0 `cat_hash` `string_to_ui32` vectors were CTR-projection hashes (extracted from a trained model's `ctr_data` `hash_map`), not `CalcCatFeatureHash`. Added `generator/cityhash_oracle.cpp` (a standalone transcription of `city.cpp`, the authoritative algorithm the live library compiles), rewired `gen_cat_hash` to use it, and regenerated `cat_hashes.npy` + `config.json`.

### Task 2 — First-seen perfect-hash remap + categorical bin oracle (DATA-05) — `2b14d03`
- `cb_data::PerfectHash`: `HashMap<u32,u32>` lookup + insertion counter; `remap(hash)` assigns `bin = map.len()` for each new hash (`cat_feature_perfect_hash_helper.cpp:120`), repeats reuse their bin (`:127`).
- Uniq-count bounded to `MAX_UNIQ_CAT_VALUES = u32::MAX` (`:53-54`); the next distinct insert past the bound returns `CbError::OutOfRange` rather than panicking (Security V5 / T-02-11). A `pub(crate) remap_bounded(hash, cap)` seam lets the overflow path be exercised with a small cap.
- `cb_data::perfect_hash_bins(column)`: one-pass hash + first-seen remap returning per-object bins.
- `cat_hash_test.rs`: first-seen assignment (`[0,1,0,2,1,0]`), bin reuse, and overflow-returns-typed-error (no panic) unit tests.
- `tests/cat_hash_oracle_test.rs`: reconstructs the corpus by tiling the per-column first-seen orders from `config.json` to `n_rows` (the corpus is an exact cycle), hashes each value, and asserts per-object `CalcCatFeatureHash` matches `cat_hashes.npy` (bit-exact) AND first-seen bins match `perfect_hash_bins.npy` (integer-exact).

## Verification (all green)
- `cargo test -p cb-data cat_hash` — 7 unit tests pass.
- `cargo test -p cb-data --test cat_hash_oracle_test` — per-object hashes + bins match oracle.
- `cargo test -p cb-data` — 36 unit + 2 (borders) + 1 (cat-hash) + 1 (quantize) oracle tests pass.
- `cargo clippy -p cb-data --lib -- -D warnings` — clean.
- `bash scripts/check-no-raw-float-sum.sh` — exits 0 (D-08; no raw float summation introduced).
- No `cityhash` crate in `Cargo.toml` (port is from vendored source, T-02-SC accept-disposition honored).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Wave-0 cat_hash fixtures captured CTR-projection hashes, not CalcCatFeatureHash**
- **Found during:** Task 1 (the bit-exact unit vectors brought the fixture under a real Rust comparison for the first time).
- **Issue:** Plan 02-01's `gen_fixtures.py` `_isolate_cat_hash` extracted `(string -> ui64)` values from a *trained model's* `ctr_data` `hash_map`. That map stores CTR-PROJECTION hashes (`CalcHashes` over projections, `MultiHash`-combined with priors — `catboost/private/libs/algo/index_hash_calcer.h`), NOT the raw `CalcCatFeatureHash(string)`. A faithful CityHash64 port can never reproduce them. Empirically, `CalcCatFeatureHash("3") = CityHash64("3") & 0xffffffff = 593172586`, but the fixture claimed `2658984922`; `"alpha"` was `3214079027` vs the true `1296865003`. Verified the true values three ways: (a) a standalone C++ harness compiled from the vendored `city.cpp`, (b) the Rust port, and (c) cross-checking the model's stored values are unrelated to `CityHash64` of the strings.
- **Fix:** Added `crates/cb-oracle/generator/cityhash_oracle.cpp` — a dependency-free transcription of the vendored `util/digest/city.cpp` (the same algorithm the live catboost library compiles), reading strings on stdin and emitting `ui64\tui32`. Rewrote `gen_cat_hash` to hash each distinct string with this tool instead of mining `ctr_data`. Regenerated `cat_hashes.npy` and `config.json` (`string_to_ui32` / `string_to_ui64_precursor` now hold the true `CalcCatFeatureHash`; A5 text corrected; `borders_source` recorded). `perfect_hash_bins.npy` was unchanged (bins are first-seen order, independent of the hash value).
- **Files modified:** `crates/cb-oracle/generator/gen_fixtures.py`, `crates/cb-oracle/generator/cityhash_oracle.cpp` (new), `crates/cb-oracle/fixtures/cat_hash/cat_hashes.npy`, `crates/cb-oracle/fixtures/cat_hash/config.json`
- **Commit:** `a791c89`
- **Precedent:** Directly parallels 02-02's Rule-1 fix, where `borders_quant` fixtures had captured training-pruned `get_borders()` instead of the standalone quantizer output. Same principle: the oracle target must be the authoritative algorithm output, not a downstream trained-model artifact.

## Out-of-Scope Discoveries (deferred, not fixed)
- Pre-existing `clippy::neg_cmp_op_on_partial_ord` lint in `crates/cb-oracle/src/compare.rs:44` (`if !(diff <= tol)`, from Phase-1 commit `902368d`). Surfaces only under `--all-targets` clippy of cb-data's dependency graph; the plan's gate `clippy -p cb-data --lib` is clean. Logged to `deferred-items.md`; not in scope for the cat-hash plan (compare.rs untouched).

## Known Stubs
None. `city_hash_64` is the full algorithm (every length path), `perfect_hash_bins` performs the real first-seen remap, and the uniq-count bound is enforced. No placeholder data flows anywhere.

## Threat Flags
None. No new network/auth/file-access surface. The plan's two trust boundaries are both mitigated and test-locked: (T-02-10) CityHash-variant mismatch — ported from vendored source, bit-exact `assert_eq!` vectors including a `>64`-byte multi-block case; (T-02-11) uniq-cat overflow — `u32::MAX` bound returns a typed `CbError::OutOfRange`, exercised by a unit test, no panic. T-02-SC (no crate installs) honored — CityHash is ported, not crate-sourced.

## Notes for Downstream Plans
- The corrected `cat_hash` fixtures hold the TRUE `CalcCatFeatureHash`. Any future plan comparing against them must hash via `cb_data::calc_cat_feature_hash` (CityHash 1.0 from vendored source), never a third-party cityhash crate or a model's `ctr_data` hash_map.
- Phase 5 (CTR) categorical statistics build on these first-seen bins; the bin assignment is now oracle-locked per object.
- `perfect_hash_bins` currently implements the plain-training first-seen path. The `mapMostFrequentValueTo0` tiebreak (which uses the upstream `TMap` sorted order) is deferred and noted in `cat_hash.rs`.

## Self-Check: PASSED
- Created files exist: `cat_hash.rs`, `cat_hash_test.rs`, `tests/cat_hash_oracle_test.rs`, `generator/cityhash_oracle.cpp` — all present.
- Commits present: `a791c89` (Task 1), `2b14d03` (Task 2) — both in git history.
- All verification commands green (7 cat_hash unit + 1 cat-hash oracle + full cb-data suite, D-08 gate, lib clippy).

---
*Phase: 02-data-layer-pool-quantization-reduction*
*Completed: 2026-06-13*
