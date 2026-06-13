---
status: testing
phase: 02-data-layer-pool-quantization-reduction
source: [02-VERIFICATION.md]
started: 2026-06-13
updated: 2026-06-13
---

## Current Test

number: 1
name: STL priority_queue heap tie-break port in GreedyLogSum greedy split (WR-01)
expected: |
  crates/cb-data/src/borders.rs (heap_push / heap_pop / adjust_heap, ~lines 305–400)
  reproduces libstdc++ __push_heap / __pop_heap / __adjust_heap behavior so that, on
  tied-score candidate bins, the greedy split selects the SAME bin upstream CatBoost's
  std::priority_queue would pop. The borders oracle tests pass (indirect evidence) and a
  permutation-invariance regression test passes, but a tie-case confirmation against a C++
  reference (or expert review of the array-layout path) is needed to be certain the port is
  bit-faithful on ties.
awaiting: user response

## Tests

### 1. STL priority_queue heap tie-break port (WR-01)
expected: heap_push/heap_pop/adjust_heap in borders.rs match libstdc++ priority_queue pop
  order on tied scores; greedy split picks the same bin as upstream. Confirm via a tie-case
  C++ reference run or expert review of borders.rs:305–400, OR accept the passing borders
  oracle + permutation-invariance tests as sufficient parity evidence.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
