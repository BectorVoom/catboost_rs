---
phase: 08-python-bindings-dual-api-packaging
plan: 06
subsystem: api
tags: [pyo3, free-threading, gil-safety, maturin, abi3, numpy]

# Dependency graph
requires:
  - phase: 08-03
    provides: own-before-detach (D-11) ingestion call sites that copy Python buffers into Rust-owned OwnedColumns before any GIL release
  - phase: 08-05
    provides: complete compiling catboost-rs-py crate (estimator trio + ingestion), full Rust+pytest suite green
provides:
  - "#[pymodule(gil_used = false)] declaration on the catboost_rs module (PyO3 0.29, free-threaded-aware)"
  - "tests/test_free_threaded.py: concurrent fit/predict buffer-safety test (>=8 threads) with a GIL-build skip-guard"
  - "FREE_THREADING.md: abi3-vs-free-threaded deferral rationale + custom-loss GIL-reentry caveat"
affects: [packaging, wheel-build, custom-loss-callback, gsd-verify-work]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "gil_used=false as a code property backed by own-before-detach (not new copying logic)"
    - "free-threaded skip-guard: tests skip cleanly on a GIL build, run only under python3.13t (Phase-7.5 cpu-skip lesson)"

key-files:
  created:
    - crates/catboost-rs-py/tests/test_free_threaded.py
    - crates/catboost-rs-py/FREE_THREADING.md
  modified:
    - crates/catboost-rs-py/src/lib.rs

key-decisions:
  - "PYAPI-06 satisfied as a CODE PROPERTY (own-before-detach + gil_used=false), free-threaded WHEEL deferred (abi3-py312 mutually exclusive with free-threading in PyO3 0.29)"
  - "No python3.13t available in-env: concurrent free-threaded run is a SCOPED DEFERRAL; PYAPI-06 stands code-property-validated; the GIL-build skip-guard test passes (gate met)"
  - "custom_loss callback GIL-reentry documented as accepted caveat (A6 / T-08-20), not a contradiction of gil_used=false"

patterns-established:
  - "Skip-guard via sys._is_gil_enabled() with pre-3.13 fallback (absent attr => GIL enabled => skip)"
  - "Free-threaded validation is a documented runtime test, gated on interpreter availability, never fabricated"

requirements-completed: [PYAPI-06]

# Metrics
duration: 18min
completed: 2026-06-23
---

# Phase 8 Plan 06: Free-Threaded Buffer-Safety (PYAPI-06) Summary

**`#[pymodule(gil_used = false)]` declared and backed by the 08-03 own-before-detach discipline, with a concurrent fit/predict buffer-safety test (GIL-build skip-guarded) and a FREE_THREADING.md documenting the abi3-vs-free-threaded wheel deferral and the custom-loss GIL-reentry caveat.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-23
- **Completed:** 2026-06-23
- **Tasks:** 2
- **Files modified:** 3 (1 modified, 2 created)

## Accomplishments
- Set `#[pymodule(gil_used = false)]` on the `catboost_rs` module (PyO3 0.29); the abi3-py312 cpu wheel rebuilds and the module imports cleanly with the flag.
- Added `tests/test_free_threaded.py`: an ≥8-thread concurrent `fit`/`predict` buffer-safety test over both per-thread-private inputs and a single shared, write-protected input array, asserting finite + cross-thread-equal results (T-08-18 / T-08-19). It SKIPs (never false-passes, never panics) on a GIL build.
- Wrote `FREE_THREADING.md` documenting (a) PYAPI-06 as a code property, (b) why the free-threaded wheel is deferred (abi3 ⊥ free-threaded in PyO3 0.29), (c) the validation command, and (d) the custom-loss callback GIL-reentry caveat (A6).
- Full regression suite green: 29 Rust tests pass; pytest 73 passed, 5 skipped (3 new free-threaded skips on the GIL venv), 79 xfailed.

## Task Commits

Each task was committed atomically:

1. **Task 1: gil_used=false flag + multi-thread buffer-safety test** - `733546f` (feat)
2. **Task 2: Document the abi3↔free-threaded deferral + custom-loss caveat** - `fedf1b3` (docs)

_Note: Task 1 is TDD-tagged. Because the buffer-safety test is meaningful only under a free-threaded interpreter (unavailable in-env), the test's "RED" on a GIL build is a clean skip rather than a run-to-fail; the test and the `gil_used=false` implementation were committed together in a single feat commit. See Deviations._

## Files Created/Modified
- `crates/catboost-rs-py/src/lib.rs` - Changed `#[pymodule]` to `#[pymodule(gil_used = false)]` with an explanatory doc comment tying the flag to own-before-detach and noting the custom-loss exception.
- `crates/catboost-rs-py/tests/test_free_threaded.py` - Concurrent fit/predict buffer-safety test; module-level skip-guard via `sys._is_gil_enabled()` (with a pre-3.13 fallback treating the missing attribute as GIL-enabled); a free-threaded import test; documented run command.
- `crates/catboost-rs-py/FREE_THREADING.md` - PYAPI-06 code-property rationale, abi3↔free-threaded deferral, validation steps, custom-loss reentry caveat.

## Decisions Made
- **PYAPI-06 = code property, free-threaded wheel deferred.** abi3-py312 and free-threading are mutually exclusive in PyO3 0.29 (RESEARCH Pitfall 1; CONTEXT Deferred Ideas). The shipped artifact is the abi3-py312 cpu wheel; PYAPI-06 stands on own-before-detach + `gil_used=false`.
- **Scoped deferral of the concurrent free-threaded RUN.** No `python3.13t` / `python3.14t` interpreter is installed in-env (only CPython 3.12.3, GIL build). Per the plan's human-check and the checkpoint guidance, this is recorded as a scoped deferral, not fabricated as a pass. The required-and-met gate is the **GIL-build skip-guard test passing** (it does); the free-threaded run is deferred-pending-interpreter (exact command below).
- **custom-loss reentry as accepted caveat (A6 / T-08-20).** The Python `custom_loss`/`custom_metric`/`eval_metric` callback re-enters the GIL per der1/der2/eval via `Python::attach` (serialized) — documented exception, not a default fixture-reachable path.

## Deferred / Pending-Interpreter Validation

**Free-threaded concurrent fit/predict run — DEFERRED (no free-threaded interpreter in-env).**

- **Detected:** `python3.13t` / `python3.14t` / `python3.13-nogil` not on PATH; the test venv is CPython 3.12.3 (GIL, `sys._is_gil_enabled` absent).
- **PYAPI-06 status:** CODE-PROPERTY-VALIDATED — own-before-detach call sites (08-03) + `#[pymodule(gil_used = false)]` + the GIL-build skip-guard test passing.
- **What is deferred:** only the *runtime* concurrent free-threaded execution of `test_free_threaded.py` (it currently skips, as designed, on the GIL build).
- **Exact command to discharge the deferral once a free-threaded interpreter exists in-env:**
  ```sh
  python3.13t -m venv .venv-ft
  .venv-ft/bin/pip install maturin pytest numpy
  cd crates/catboost-rs-py
  ../../.venv-ft/bin/maturin develop --features cpu
  ../../.venv-ft/bin/python -m pytest tests/test_free_threaded.py -q   # must RUN (not skip) and pass
  ```

## Deviations from Plan

### TDD flow adaptation (no auto-fix; structural note)

**1. [TDD adaptation] RED-as-skip for the free-threaded test on a GIL build**
- **Found during:** Task 1 (TDD-tagged).
- **Issue:** A `tdd="true"` task normally requires a failing (RED) test before implementation. The buffer-safety test is, by design (and per the plan's behavior spec), *meaningful only* under a free-threaded interpreter and must SKIP — never fail/panic — on a GIL build. With no free-threaded interpreter in-env, a genuine run-to-fail RED is impossible without violating the explicit skip-guard requirement.
- **Resolution:** Authored the test (it collects and skips cleanly = 3 skipped, proving validity), then set `gil_used=false`, rebuilt, and confirmed the module loads and the test still skips cleanly. Committed test + implementation together in one feat commit (`733546f`). No fabricated pass.
- **Verification:** `pytest tests/test_free_threaded.py -q` → 3 skipped; `pytest tests/test_smoke.py` → 5 passed (module loads with the flag); full suite green.

---

**Total deviations:** 1 (TDD flow adaptation; no code auto-fixes).
**Impact on plan:** None on scope. The adaptation is dictated by the plan's own skip-guard requirement plus the in-env interpreter gap, both anticipated by the plan and the checkpoint guidance.

## Issues Encountered
- The test venv path in the prompt (`crates/catboost-rs-py/.venv-py8`) resolved correctly once located (the repo-root `.venv-py8` does not exist; tooling lives under the crate). maturin/pytest/python3.12 all present. `maturin develop` emitted a benign pre-existing `patchelf` rpath warning (not a failure); the abi3-py312 wheel built and installed.

## User Setup Required
None - no external service configuration required. (A free-threaded interpreter is only needed to *discharge the deferred validation*; it is not required for the shipped abi3 cpu artifact.)

## Next Phase Readiness
- PYAPI-06 satisfied as a code property; ready for `/gsd-verify-work`. The phase gate "GIL-build skip-guard passes" is met. The free-threaded concurrent run is a scoped deferral with an exact command to discharge it if a `python3.13t` is provisioned in-env.
- No blockers introduced. The custom-loss caveat is documented for any future free-threaded callback work.

## Self-Check: PASSED

- FOUND: crates/catboost-rs-py/src/lib.rs (gil_used present, 2 occurrences)
- FOUND: crates/catboost-rs-py/tests/test_free_threaded.py
- FOUND: crates/catboost-rs-py/FREE_THREADING.md
- FOUND: .planning/phases/08-python-bindings-dual-api-packaging/08-06-SUMMARY.md
- FOUND commit: 733546f (Task 1)
- FOUND commit: fedf1b3 (Task 2)

---
*Phase: 08-python-bindings-dual-api-packaging*
*Completed: 2026-06-23*
