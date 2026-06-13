---
phase: 01-workspace-lint-discipline-oracle-harness
plan: 02
subsystem: infra
tags: [prng, rng, pcg, lcg, parity-oracle, cb-core, rust, tdd]

# Dependency graph
requires:
  - phase: 01-01
    provides: "cb-core scaffold (lib.rs, error.rs CbError::InvalidBound/OutOfRange + CbResult<T>), workspace clippy deny table, in-code test-lint exemption"
provides:
  - "cb-core::TFastRng64 — exact bit-for-bit port of CatBoost's PCG-XSH-RR fast PRNG (two 32-bit LCG streams concatenated to 64 bits)"
  - "from_seed(u64) one-arg ctor (TArgs derivation via a TReallyFastRng32) and new(seed1,seq1,seed2,seq2) four-arg ctor with FixSeq distinct-stream rule"
  - "gen_rand() -> u64, advance(delta) seekable jump (LcgAdvance binary exponentiation), try_uniform(bound) -> CbResult<u64> (rejection sampling, InvalidBound on 0), infallible uniform(bound)"
  - "Bitstream-oracle test suite (rng_test.rs) transcribed from vendored fast_ut.cpp: Test3, Test2 Uniform(100) sequence, TestAdvance parity, TestAdvanceBoundaries, InvalidBound"
  - "CbError now derives Clone + PartialEq + Eq (enables Result equality assertions in downstream tests)"
affects: [03-stub-crates-oracle-generator, all-later-phases-sampling-permutation, ordered-boosting, gpu-sampling]

# Tech tracking
tech-stack:
  added: []
  patterns: [exact-C++-bitstream-port-with-wrapping-arithmetic, two-streams-deduped-into-shared-Lcg32-type, fallible-precondition-via-Result-not-panic, transcribed-vendored-test-vectors-as-parity-oracle]

key-files:
  created:
    - crates/cb-core/src/rng.rs
    - crates/cb-core/src/rng_test.rs
  modified:
    - crates/cb-core/src/lib.rs
    - crates/cb-core/src/error.rs

key-decisions:
  - "Deduped the two 32-bit PCG streams into a single shared Lcg32 type (REFACTOR) — bitstream-identical, proven by the passing oracle vectors"
  - "from_seed derivation order: low 32 bits = first GenRand call, high = second (ToRand64 order in common_ops.h); validated by Test3 == 14895365814383052362"
  - "uniform() (infallible) returns 0 for the degenerate bound==0 case via unwrap_or(0) — clippy::panic-clean; callers needing the error use try_uniform"
  - "Derived Clone/PartialEq/Eq on CbError (Rule 3) so try_uniform(100) == Ok(37) compiles; backward-compatible, no existing test broke"

patterns-established:
  - "Exact C++ port pattern: every multiply/add uses wrapping_mul/wrapping_add (Pitfall 5) for bitstream parity AND debug-overflow-panic safety"
  - "Precondition-as-Result: port C++ Y_ABORT_UNLESS into try_* returning CbError, with an infallible wrapper that never panics (D-13)"
  - "Parity oracle as transcribed vendored vectors: no C++ build required; expected values copied verbatim from fast_ut.cpp with provenance comments"

requirements-completed: [INFRA-05]

# Metrics
duration: 4min
completed: 2026-06-13
---

# Phase 1 Plan 02: TFastRng64 Exact PRNG Port Summary

**Bit-for-bit Rust port of CatBoost's PCG-XSH-RR `TFastRng64` (two 32-bit LCG streams → 64 bits) in `cb-core`, validated against five vendored `fast_ut.cpp` oracle vectors including `from_seed(17).gen_rand() == 14895365814383052362` and the full 20-value `Uniform(100)` sequence, with a seekable `advance()` and a non-panicking `try_uniform` precondition.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-06-13T01:00:45Z
- **Completed:** 2026-06-13T01:04:44Z
- **Tasks:** 1 (TDD: RED → GREEN → REFACTOR)
- **Files modified:** 2 created, 2 modified

## Accomplishments
- `TFastRng64` ported bit-for-bit from `fast.h`/`lcg_engine.{h,cpp}`/`common_ops.h`/`fast.cpp`: PCG mixer (XSH-RR), LCG iterate (`x*A + C`), `from_seed` one-arg derivation via a `TReallyFastRng32`, `new` four-arg ctor with the `FixSeq` distinct-stream rule, `gen_rand` `(R1<<32)|R2`, rejection-sampling `uniform`, and `LcgAdvance` binary-exponentiation `advance`.
- All LCG/mixer/advance arithmetic uses `wrapping_mul`/`wrapping_add` (RESEARCH Pitfall 5; threat T-01-04) — parity-exact and debug-overflow-panic-free.
- `try_uniform(bound) -> CbResult<u64>` ports the C++ `Y_ABORT_UNLESS(max > 0)` precondition into `Err(CbError::InvalidBound)` — never panics (threat T-01-05, D-13).
- Module-level doc-comment marks `TFastRng64` non-cryptographic / parity-only (RESEARCH Security V6, threat T-01-06).
- Oracle suite (`rng_test.rs`) transcribed verbatim from `fast_ut.cpp` — all 6 tests pass: `Test3`, `Test2` 20-value `Uniform(100)` sequence, `TestAdvance` parity, `TestAdvanceBoundaries`, and two `InvalidBound`/`try_uniform` cases.

## Task Commits

Each task was committed atomically:

1. **Task 1: TFastRng64 exact bitstream port (RED → GREEN → REFACTOR)** - `697d811` (feat)

**Plan metadata:** committed separately (docs: complete plan).

_TDD note: RED was observed before GREEN — a stub `rng.rs` returning `0` was wired in and the transcribed vectors were run and seen to fail (Test3/Test2/Advance mismatches, plus a `PartialEq`-missing compile error) before the exact port was written and they turned GREEN. The REFACTOR (deduping the two streams into a shared `Lcg32`) was applied with the vectors staying green. RED stub and GREEN port were squashed into the single task `feat` commit, consistent with Plan 01-01's established convention (RED stubs not committed separately)._

## Files Created/Modified
- `crates/cb-core/src/rng.rs` - Production `TFastRng64`: `pcg_mix`, `lcg_advance`, shared `Lcg32` stream type, `fix_seq`, all ctors/`gen_rand`/`advance`/`uniform`/`try_uniform`; non-crypto module doc-comment; no inline `#[cfg(test)]`.
- `crates/cb-core/src/rng_test.rs` - Bitstream-oracle tests transcribed from vendored `fast_ut.cpp` (D-17 separate test file).
- `crates/cb-core/src/lib.rs` - Added `mod rng;`, `pub use rng::TFastRng64;`, `#[cfg(test)] mod rng_test;`.
- `crates/cb-core/src/error.rs` - Added `Clone, PartialEq, Eq` to `CbError`'s derive (the `InvalidBound` variant itself already existed from Plan 01-01).

## Decisions Made
- Deduped the two 32-bit PCG streams into one shared `Lcg32` type (REFACTOR) — bitstream-identical, proven by the green oracle vectors.
- `from_seed` derivation draws low-then-high 32-bit halves per `ToRand64` (`common_ops.h`); validated exactly by `Test3`.
- Infallible `uniform()` returns `0` for `bound == 0` via `unwrap_or(0)` (clippy::panic-clean); `try_uniform` is the fallible path callers should use.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Derived `Clone, PartialEq, Eq` on `CbError`**
- **Found during:** Task 1 (RED — transcribed `try_uniform(100) == Ok(37)` assertion)
- **Issue:** `assert_eq!(rng.try_uniform(100), Ok(37))` requires `Result<u64, CbError>: PartialEq`, which `CbError` did not implement; the test could not compile (`E0369`).
- **Fix:** Added `Clone, PartialEq, Eq` to `CbError`'s `#[derive(...)]`. Backward-compatible (additive); `thiserror` and existing `error_test.rs` (`matches!` / `to_string`) unaffected.
- **Files modified:** `crates/cb-core/src/error.rs`
- **Verification:** `cargo test --workspace` (all suites pass, including `error_test`); `cargo clippy --workspace --lib -- -D warnings` clean.
- **Committed in:** `697d811` (Task 1 commit)

**Note on `error.rs` plan instruction:** The plan asked to "add a reserved `CbError` variant (`InvalidBound`) in `error.rs`." That variant was already added by Plan 01-01 (it reserved `InvalidBound` + `OutOfRange` precisely for this plan), so no new variant was needed — `try_uniform` simply returns the existing one. This is not a deviation, just a pre-satisfied instruction.

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The `CbError` derive addition was required to compile the transcribed oracle assertion. No scope creep — the RNG algorithm, ctors, method signatures, error path, wrapping arithmetic, doc-comment, and source/test separation are exactly as planned.

## Issues Encountered
None beyond the `PartialEq` compile gap (resolved via Rule 3 above). The exact port reproduced every vendored vector on the first GREEN run (no algorithm debugging required).

## Known Stubs
None. `rng.rs` is a complete, working port; all oracle vectors pass.

## Threat Flags
None — no new security surface beyond the plan's threat model. `TFastRng64` introduces no network/file/auth surface; it is a pure deterministic compute primitive. Threats T-01-04 (overflow), T-01-05 (`uniform(0)`), T-01-06 (non-crypto misuse) are mitigated as planned (wrapping arithmetic, `try_uniform -> Result`, non-crypto doc-comment).

## TDD Gate Compliance
This is a `type: tdd` plan. RED (failing transcribed vectors against a `0`-returning stub + `PartialEq` compile error) was observed before GREEN (exact port). REFACTOR (dedupe to shared `Lcg32`) kept vectors green. Per Plan 01-01's established convention the RED stub and GREEN port are squashed into the single `feat(01-02)` task commit `697d811` rather than separate `test(...)`/`feat(...)` commits — so the git log shows one `feat` commit for this single-task plan, not a discrete `test`-then-`feat` pair.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `cb-core::TFastRng64` is exported and ready for every later phase's sampling/permutation/bootstrap logic (INFRA-05 satisfied).
- `CbError` now supports equality comparison, simplifying downstream `Result` assertions.
- Verified green: `cargo test -p cb-core rng` (6/6), `cargo clippy -p cb-core --lib -- -D warnings`, full `cargo test --workspace`, `cargo clippy --workspace --lib -- -D warnings`, `bash scripts/check-no-anyhow.sh`.
- Plan 03 (next, Wave 2): create the six stub crates + Python oracle generator + input corpus; extend ci.yml with the feature-stub build step.

## Self-Check: PASSED

- `crates/cb-core/src/rng.rs` exists on disk; `crates/cb-core/src/rng_test.rs` exists on disk.
- Commit `697d811` present in git history (`feat(01-02): exact TFastRng64 PRNG port...`).

---
*Phase: 01-workspace-lint-discipline-oracle-harness*
*Completed: 2026-06-13*
