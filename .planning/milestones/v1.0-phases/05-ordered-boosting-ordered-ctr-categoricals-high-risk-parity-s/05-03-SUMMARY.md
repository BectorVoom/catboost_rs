---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 03
subsystem: cb-train (multi-permutation fold machinery, ORD-01 / D-03 linchpin)
tags: [permutation, fisher-yates, fold, body-tail, ordered-boosting, tfastrng64, d-03, integer-exact]
requires:
  - cb-core::TFastRng64 (bit-exact PCG/LCG RNG — uniform/from_seed, Phase 1)
  - cb-train::bootstrap (the persistent-RNG draw-phase discipline to mirror, Phase 3)
  - cb-train::BoostParams (the plain-boosting param struct extended here, Phase 3)
  - cb-oracle::compare_permutation + Stage::Permutation (integer-exact, 05-01)
  - frozen fixtures plain_ctr/ordered_boost permutation_fold0.npy + body_tail_boundaries.npy (05-01)
provides:
  - cb-train::permutation (fisher_yates_permutation / permutations / fold_block_size)
  - cb-train::fold (body_tail_boundaries / body_tail_segments / plain_fold_body_tail /
    learning_fold_count / create_folds / Fold / select_min_batch_size / select_tail_size /
    body_sum_weights)
  - cb-train::BoostParams.{permutation_count (default 4), fold_len_multiplier (default 2.0)}
    + permutation_count_default() / fold_len_multiplier_default()
  - crates/cb-train/tests/permutation_oracle_test.rs (D-03 integer-exact lock)
affects:
  - 05-04..05-06 (ordered CTR / ordered approx value stages compute UNDER this
    permutation + body/tail prefix; the D-03 lock gates them)
tech-stack:
  added: []
  patterns:
    - integer-exact permutation reproduction over the bit-exact TFastRng64 (== not 1e-5), gating value stages (D-03)
    - persistent-RNG continuous-stream multi-fold draw order (mirrors bootstrap.rs PRE_TREE_DRAWS discipline)
    - explicit-pin permutation_count/fold_len_multiplier via *_default() helpers (RESEARCH Pitfall 6, no auto-select)
key-files:
  created:
    - crates/cb-train/src/permutation.rs
    - crates/cb-train/src/permutation_test.rs
    - crates/cb-train/src/fold.rs
    - crates/cb-train/src/fold_test.rs
    - crates/cb-train/tests/permutation_oracle_test.rs
  modified:
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/lib.rs
    - crates/catboost-rs/src/builder.rs
    - crates/cb-train/tests/autolr_e2e_test.rs
    - crates/cb-train/tests/bootstrap_oracle_test.rs
    - crates/cb-train/tests/eval_metrics_oracle_test.rs
    - crates/cb-train/tests/leaf_methods_oracle_test.rs
    - crates/cb-train/tests/leaf_weights_oracle_test.rs
    - crates/cb-train/tests/loss_oracle_test.rs
    - crates/cb-train/tests/one_hot_oracle_test.rs
    - crates/cb-train/tests/overfit_oracle_test.rs
    - crates/cb-train/tests/regularization_oracle_test.rs
    - crates/cb-train/tests/slice_first_oracle_test.rs
key-decisions:
  - "Permutation seed = the training random_seed (random_seed=0 for the in-scope fixtures); Fisher-Yates draw order is identity-init then for i in 1..n: j=uniform(i+1); swap(i,j) — transcribed verbatim from shuffle.h:28-30 and cross-locked to the committed permutation_fold0.npy."
  - "body_tail_boundaries.npy schema decoded as the leftPartLen sequence [initial SelectMinBatchSize(n), tail_0, tail_1, …, n] — for N=30/mult=2.0 that is [1 2 4 8 16 30]; body_tail_segments derives the (body,tail) pairs as windows-of-2."
  - "permutation_count/fold_len_multiplier added to the SHARED BoostParams (reuse, not a new struct) and propagated to all 13 BoostParams literals via *_default() helpers — the prior plan's one_hot_max_size pattern — keeping the whole workspace compiling (cargo check --workspace --tests green)."
  - "create_folds draws every fold's permutation IN ORDER from a single persistent TFastRng64 (continuous stream, never reseeded per fold), mirroring bootstrap.rs; averaging fold uses the plain single span, learning folds use the dynamic growing body/tail."
patterns-established:
  - "Pattern 1: integer-exact permutation oracle as the FIRST stage (D-03) — a single-index transposition is rejected before any value stage, asserted falsifiably."
  - "Pattern 2: monotone, capped body/tail growth (tailFinish=min(ceil(left*mult),n)) with a degenerate-multiplier progress guard so the loop always terminates (T-05-03-01)."
requirements-completed: [ORD-01]

duration: 12min
completed: 2026-06-14
---

# Phase 5 Plan 03: Multi-Permutation Fold Machinery (ORD-01 / D-03 linchpin) Summary

**Fisher-Yates per-fold permutation over the bit-exact `TFastRng64` (block 1 for N<1000) locked INTEGER-EXACT against the committed `permutation_fold0.npy`, plus the `TFold` body/tail prefix state machine (`[1 2 4 8 16 30]` for N=30) and `LearningFoldCount=max(1,permutation_count-1)`+averaging-fold creation — the D-03 ordering layer every downstream CTR/approx value depends on.**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-06-14
- **Tasks:** 2
- **Files modified:** 18 (5 created, 13 modified)

## Accomplishments

- `permutation.rs`: the modern Fisher-Yates `Shuffle` (`shuffle.h:24-32`) over
  `cb_core::TFastRng64::uniform`, block size 1 for N<1000 (`defaults_helper.h`),
  reproducing upstream catboost 1.2.10's permutation INDEX-FOR-INDEX. The exact
  draw order is documented with the upstream citation, mirroring the bootstrap.rs
  `PRE_TREE_DRAWS` draw-phase discipline; the RNG is CONSUMED, never re-ported.
- `permutation_oracle_test.rs`: the D-03 linchpin — generated fold-0 permutation
  asserted integer-exact (`==`, not 1e-5) vs the committed `plain_ctr` AND
  `ordered_boost` `permutation_fold0.npy` via `cb_oracle::compare_permutation`
  (`Stage::Permutation`), plus a single-transposition falsifiability guard.
- `fold.rs`: `SelectMinBatchSize`/`SelectTailSize` (`fold.cpp:35-41`), the dynamic
  growing body/tail boundary loop (`fold.cpp:148-198`), the `BuildPlainFold`
  single-span path kept intact (`fold.cpp:268-274`), `LearningFoldCount=max(1,
  permutation_count-1)` (`learn_context.cpp:48-49`), and `create_folds` (learning
  folds + one averaging fold, per-fold permutations drawn in continuous-stream
  order). `body_sum_weights` routes through `cb_core::sum_f64` (D-08).
- `BoostParams.permutation_count` (default 4) + `fold_len_multiplier` (default
  2.0), pinned EXPLICITLY via `*_default()` helpers (RESEARCH Pitfall 6) and
  propagated to every construction site with the whole workspace still compiling.

## Task Commits

1. **Task 1: Fisher-Yates permutation over TFastRng64 (D-03 linchpin)** — `f81da1c` (feat)
2. **Task 2: TFold body/tail prefix state machine + permutation_count fold creation** — `a015c8f` (feat)

_Note: both tasks were `tdd="true"`, but TDD_MODE is false for this phase (no RED-commit gate); each task is a single feat commit with its production module + sibling unit tests + the integration oracle._

## Files Created/Modified

- `crates/cb-train/src/permutation.rs` — Fisher-Yates over TFastRng64; `fisher_yates_permutation`, persistent-RNG `permutations`, `fold_block_size`.
- `crates/cb-train/src/permutation_test.rs` — identity edges (N=0,1), seed=42/N=5 anchor `[4 2 0 3 1]`, manual uniform replay, bijection invariant, block-size boundary, continuous-stream multi-fold draws.
- `crates/cb-train/tests/permutation_oracle_test.rs` — integer-exact D-03 lock vs `permutation_fold0.npy` (plain_ctr + ordered_boost) + transposition reject.
- `crates/cb-train/src/fold.rs` — body/tail prefix state machine, plain single-span, `learning_fold_count`, `create_folds`, `Fold`, `body_sum_weights`.
- `crates/cb-train/src/fold_test.rs` — batch/tail-size boundaries, `[1 2 4 8 16 30]` N=30 prefix lock, N>500 prefix, LearningFoldCount for pc∈{1,2,4}, averaging=plain/learning=dynamic, continuous fold draws, weighted/unweighted body sums.
- `crates/cb-train/src/boosting.rs` — `BoostParams.{permutation_count, fold_len_multiplier}` + `permutation_count_default()` / `fold_len_multiplier_default()`.
- `crates/cb-train/src/lib.rs` — module declarations + public re-exports.
- `crates/catboost-rs/src/builder.rs` + 11 cb-train test files — pinned the two new fields at every `BoostParams` literal (default helpers).

## Decisions Made

- **Permutation seed = training `random_seed`** (0 for the in-scope fixtures);
  the Fisher-Yates draw order was confirmed by reproducing the fixture
  `[8 12 5 18 14 …]` (N=30, seed=0) exactly in Rust before writing production
  code, and re-anchored to the 05-01 `[4 2 0 3 1]` (seed=42, N=5) self-oracle.
- **`body_tail_boundaries.npy` decoded as the `leftPartLen` sequence** — initial
  `SelectMinBatchSize(n)` then each segment's `tailFinish`, terminating at `n`.
  `body_tail_segments` derives the `(body,tail)` pairs as windows-of-2.
- **Reused the shared `BoostParams`** (not a new struct) for the two permutation
  knobs, propagating via `*_default()` helpers — the prior plan's
  `one_hot_max_size` pattern — so no cross-crate churn (workspace compiles).
- **`create_folds` continuous-stream draw order** — every fold's permutation is
  drawn in order from ONE persistent `TFastRng64`, never reseeded per fold,
  mirroring bootstrap.rs; averaging fold = plain span, learning folds = dynamic.

## Deviations from Plan

None — plan executed exactly as written. No Rule 1/2/3 auto-fixes were needed
and no architectural (Rule 4) decision arose. The two new `BoostParams` fields
breaking the existing literals was anticipated by the plan's notes (the prior
plan's `one_hot_max_size` propagation pattern) — handled with `*_default()`
helpers, not a deviation.

## Issues Encountered

None. The disk/link pressure flagged in the plan was avoided by scoping
verification to per-crate `cargo test -p cb-train` (lightweight; no MLIR/cubecl
test-profile link). `cargo check --workspace --tests` (the cross-crate integrity
gate the plan mandates) was run and passed cleanly — no sibling-crate breakage
from the two new shared `BoostParams` fields.

## Verification

- `cargo test -p cb-train permutation` — **2 integration + 9 unit green** (D-03 integer-exact lock FIRST, plus identity/anchor/bijection/stream units).
- `cargo test -p cb-train fold_prefix` — **6 green** (body/tail boundary lock, incl. `[1 2 4 8 16 30]` and N>500).
- `cargo test -p cb-train --lib` — **75 green** (was 50; +25 fold/permutation units, no regression).
- `cargo test -p cb-train --test one_hot_oracle_test` — **3 green**; `--test slice_first_oracle_test` — **2 green** (no regression in the touched test literals).
- `cargo check --workspace --tests` — **clean (exit 0)** — no cross-crate breakage from the new `BoostParams.{permutation_count, fold_len_multiplier}` fields.
- `cargo clippy -p cb-train --lib` — clean for this plan's code; the only warnings are PRE-EXISTING (`cb-backend` `enum_variant_names`, `bootstrap.rs` `excessive_precision`), out of scope.
- No `unwrap`/`expect`/`panic`/raw-index and no `anyhow` in `permutation.rs` / `fold.rs` production (checked `Vec::swap`, capped/monotone prefix arithmetic).

## Known Stubs

None. `fold_block_size(>1)` (the N≥1000 block-aware shuffle) and the dynamic
body/tail's consumption by ordered approx are deliberate, documented extension
points for later waves (05-04+), not stubs — the in-scope per-object (block=1)
path and the boundary math are fully wired and oracle-locked at N=30.

## Next Phase Readiness

- The D-03 permutation lock is GREEN and gates the downstream value stages —
  05-04+ (ordered CTR / ordered approx) can now compute under this exact
  permutation + body/tail prefix knowing the ordering is upstream-faithful.
- `create_folds` exposes the learning/averaging fold split and per-fold
  permutations the ordered slices consume; `body_sum_weights` is ready for the
  ordered-approx body-prefix normalization.

## Self-Check: PASSED

All 5 created files and 13 modified files exist on disk; both task commits
(`f81da1c`, `a015c8f`) are present in git history.

---
*Phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s*
*Completed: 2026-06-14*
