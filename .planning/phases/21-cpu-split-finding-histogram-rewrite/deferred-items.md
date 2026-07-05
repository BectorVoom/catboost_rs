
## Pre-existing test failure (out of scope, discovered Plan 21-02)

- `crates/cb-train/tests/monotone_oracle_test.rs::monotone_non_symmetric_and_region_are_typed_errors`
  asserts `grow_policy=Region` is rejected with a typed error ("Region OUT",
  D-6.6-04). This assertion is STALE: `boosting.rs:1369` (`validate_grow_policy`)
  documents that the Region-OUT rejection was **LIFTED** by GPUT-18/D-03a — Region
  now grows on CPU via `region_grower`. The test fails on the pre-21-02 baseline
  (verified by stashing the 21-02 tree.rs changes and re-running) and is unrelated
  to the oblivious histogram rewrite. Region is explicitly out of scope for Phase 21
  (CONTEXT: GPU/device untouched; the CPU Region grower is not the oblivious path).
  Fix belongs to a monotone/Region test-maintenance sweep, not this plan.
