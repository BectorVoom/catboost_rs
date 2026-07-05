
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

## D-08 backstop script (`scripts/check-no-raw-float-sum.sh`) pre-existing red (Plan 21-06)

Discovered during 21-06 Task 1. `bash scripts/check-no-raw-float-sum.sh` exits 1,
but every flagged line pre-dates this session (present on HEAD `c640137`) and lives
in files 21-06 does NOT modify:

- `crates/cb-compute/src/score.rs:124,183,203,229` — `usize` sums
  (`per_dim_leaves.iter().map(Vec::len).sum()`), NOT float folds. The crude
  `\.sum\(\)` regex cannot distinguish `usize` from `f64`.
- `crates/cb-compute/src/leaf.rs:664` — the token `iter().sum()` inside a doc comment.
- `crates/cb-backend/src/kernels/*` (`pairwise_hist.rs`, `grow_loop.rs`,
  `score_split.rs`, `pointwise_hist.rs`, `reduce.rs`) — doc-comment mentions of
  `.sum()` (all say "NEVER a naive `.sum()`").
- `crates/cb-backend/src/kernels/update_part_props.rs:164,184` — `usize` sums.

The file 21-06 rewrites (`crates/cb-compute/src/histogram.rs`) is CLEAN of any raw
float fold (`grep -nE '\.sum\(\)|\.fold\(0\.0' histogram.rs` returns nothing); the
new `build_bucket_histogram` scatter-adds via `cb_core::scatter_add_f64` (the
sanctioned primitive). Task 1's intent — "no raw float fold introduced in cb-compute"
— is satisfied; the global script red is a pre-existing false-positive backlog, out
of scope for this gap-closure plan. Fix (deferred): tighten the regex to skip
`usize`-typed sums and comment lines. Do NOT route `usize` sums through the
float-only `cb_core::sum_f64`.
