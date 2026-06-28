---
status: testing
phase: 08-python-bindings-dual-api-packaging
source: [08-VERIFICATION.md]
started: 2026-06-23
updated: 2026-06-23
---

## Current Test

number: 1
name: Concurrent free-threaded fit/predict under python3.13t (PYAPI-06 runtime)
expected: |
  On a free-threaded interpreter, all 3 tests in test_free_threaded.py RUN (not skipped)
  and pass — concurrent fit/predict across >=8 threads produces finite, cross-thread-equal
  results with no buffer corruption.
awaiting: user response

## Tests

### 1. Concurrent free-threaded fit/predict under python3.13t (PYAPI-06 runtime)
expected: |
  Install python3.13t, create a venv, `maturin develop --features cpu`, then run
  `python3.13t -m pytest crates/catboost-rs-py/tests/test_free_threaded.py -q`.
  All 3 tests RUN (not skipped) and pass: concurrent fit/predict across >=8 threads
  produces finite, cross-thread-equal results, no corruption. (Exact command also in
  crates/catboost-rs-py/FREE_THREADING.md.)
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
