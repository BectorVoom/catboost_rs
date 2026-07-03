---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 08
subsystem: cb-train
tags: [gpu, runtime-seam, grow-tree, boosting-loop, GPUT-01, GPUT-04, D-10-01, D-04]

# Dependency graph
requires:
  - phase: 10-02 (GPUT-01 grow-tree runtime seam)
    provides: DeviceGrownTree host struct + begin/grow/end default-impl seam this plan calls from the loop
  - phase: 10-07 (device-resident depth-1 grow + session)
    provides: GpuBackend begin/grow/end seam impls + coverage gate this plan's per-fit decision consumes
provides:
  - "Device grow branch in cb_train::train_inner: begin once per fit, grow_tree_on_device per iteration, end at teardown"
  - "bin_id -> border join (border = feature_borders[feature][bin_id]) folding a DeviceGrownTree into an ObliviousTree identical to a CPU-grown tree"
  - "Per-fit all-or-nothing gate (D-10-01): begin Ok(false)/host-ineligible -> byte-unchanged CPU grower; Ok(None) mid-run -> typed error (no tree mixing)"
  - "quantize_feature_major (feature-major cindex + n_bins) for begin_device_training"
  - "DeviceSessionGuard: end_device_training on every exit path incl. the ? error path"
affects:
  - "10-09 Kaggle CUDA authoritative depth-1 end-to-end RMSE ~1e-9 / Logloss <=1e-5 (human-gated) — now reachable from fit()"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two composed coverage gates: a host-eligibility predicate in cb-train (excludes ranking/ordered/CTR/penalties/monotone/multi-dim/sampling/perturbation/eval — the configs the backend seam cannot see) AND the backend's begin_device_training gate (depth/loss/Plain/fold/score-fn). The decision is made ONCE per fit; the CpuBackend default Ok(false) keeps the CPU path byte-unchanged (D-04)."
    - "RAII DeviceSessionGuard for seam teardown: end_device_training fires on every train_inner exit path (Ok and ?-propagated Err), best-effort (Drop swallows the infallible teardown), inert when no session opened."
    - "Host-side leaf assignment from the resolved splits (value > border + forward-bit leaf_index) instead of consuming DeviceGrownTree.leaf_of (empty in the hot path, D-05) — so no n-length buffer crosses the seam per tree."

key-files:
  created:
    - crates/cb-train/tests/device_seam_test.rs
  modified:
    - crates/cb-train/src/boosting.rs

key-decisions:
  - "Per-fit all-or-nothing (D-10-01): the device-vs-CPU decision is made ONCE at begin. A later Ok(None) from a committed device fit is a typed CbError::Degenerate (T-10-23), NOT a silent CPU-tree backfill — mixing device- and CPU-grown trees in one model is forbidden."
  - "Host-eligibility predicate excludes every config the depth-1 device grower does not implement AND that the backend seam cannot observe (ranking group_spans, ordered perm, materialized/structure-fold CTR, penalties, monotone, approx_dimension>1, multiclass/multilabel, bootstrap != No, random_strength != 0, eval sets). begin's finer gate (depth==1 / RMSE|Logloss|CrossEntropy / Plain / fold==1 / supported score fn) then owns the rest. Both must pass for device_active."
  - "scaled_l2 is computed ONCE before the loop (constant across the fit for the device-eligible config: fixed weights/n, no per-tree sampling), matching the CPU per-tree scale_l2_reg(l2, sumAllWeights, n)."
  - "Leaf values fold through the EXISTING normalize_leaf_values with is_pairwise=false — for RMSE/Logloss that is exactly the CPU learning_rate*delta store (no centering), so a device tree is byte-identical in structure to a CPU-grown one (the device returns UN-scaled leaves, 10-02 contract)."
  - "n_bins = max_f(feature_borders[f].len()) + 1 (uniform per-feature bin-line size the session expects); bins[f*n+i] = count of ascending borders value strictly exceeds, so quantized_bin > bin_id <=> value > feature_borders[feature][bin_id] (the round-trip behind the bin_id->border join)."

requirements-completed: [GPUT-01, GPUT-04]

# Metrics
duration: ~35min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 08: Wire the GPUT-01 Device Grow Seam into cb-train's Boosting Loop Summary

**Made the device grow path reachable from the public `fit()`: `cb_train::train_inner` now decides ONCE per fit (D-10-01 all-or-nothing) whether the whole fit runs on the device grower — via a host-eligibility predicate composed with the backend's `begin_device_training` coverage gate — then, when committed, grows one oblivious tree per iteration through `grow_tree_on_device`, resolves each `(feature, bin_id)` split to `border = feature_borders[feature][bin_id]` (the one non-obvious correctness join, T-10-22 range-checked), assigns leaves on the host (D-05, no `n`-length `leaf_of` crossing), folds the UN-scaled device leaves through the existing `normalize_leaf_values` (learning_rate applied, no centering for RMSE/Logloss), updates `approx`, and pushes an `ObliviousTree` identical to a CPU-grown one. `Ok(None)` after a committed device fit is a typed error (no CPU/device tree mixing, T-10-23); `DeviceSessionGuard` calls `end_device_training` on every exit path including the `?` error path (T-10-24). On the default `CpuBackend` `begin` returns `Ok(false)`, so the CPU grower runs byte-unchanged (D-04) — cb-train's 231 lib tests + 53 integration binaries stay green, and a new 4-test CPU-only mock-`Runtime` suite locks the host-side fold.**

## Performance
- **Duration:** ~35 min
- **Completed:** 2026-07-03
- **Tasks:** 2
- **Files:** 2 (1 created, 1 modified)

## Accomplishments

- **Device grow branch in `train_inner` (Task 1, GPUT-01/04).** A `device_active` flag is decided once before the tree loop: `device_host_eligible` (a predicate over `group_spans` / `ordered_learning_perm` / CTR columns / penalties / monotone / `approx_dimension` / multiclass / multilabel / bootstrap / random_strength / eval sets / feature count) composed with `runtime.begin_device_training(...)`. Inside the loop, `if device_active` grows on device, folds the tree, and `continue`s — the CPU body is byte-unchanged when the branch is not taken (D-04).
- **bin_id -> border join + fold (Task 2, Pattern 4).** Each device split `(feature, bin_id)` resolves to `Split { feature, border = feature_borders[feature][bin_id] }` with a checked `.get` range guard (T-10-22 typed `CbError::OutOfRange`). Leaf partition is recomputed on the host from the resolved splits; leaf values fold through `accumulate_leaf_weights` + `normalize_leaf_values(is_pairwise=false, learning_rate, ...)`; `approx[i] += leaf_values[leaf(i)]`; staged approx recorded per iteration.
- **Per-fit all-or-nothing + teardown (D-10-01 / T-10-23 / T-10-24).** `Ok(None)` after `begin -> Ok(true)` is a typed `CbError::Degenerate` (no tree mixing). `DeviceSessionGuard` (RAII) calls `end_device_training` on every exit path, including `?`-propagated errors; inert when no session was opened.
- **`quantize_feature_major`** builds the feature-major cindex (`bins[f*n+i] = #borders exceeded`) + uniform `n_bins` for `begin_device_training`, with the round-trip guarantee `quantized_bin > bin_id <=> value > feature_borders[feature][bin_id]`.
- **CPU-only structural test suite** (`device_seam_test.rs`, 4 tests): the bin_id->border join + leaf-value scaling + staged contribution, the all-or-nothing rejection, the out-of-range bin_id typed error, and the `begin -> Ok(false)` CPU fallback — validated with a mock `Runtime` whose `compute_gradients` errors so the "device branch was taken" / "CPU fallback was taken" assertions are real.

## Task Commits
1. **Task 1: wire the device grow seam into the boosting loop** — `8f7280b` (feat)
2. **Task 2: structural device-seam fold test** — `d23c31d` (test)

## Deviations from Plan
None functional — the plan executed as written. Scope-clarifying notes (documented above as key-decisions, not correctness deviations):

1. **The full `(feature, bin_id) -> border` fold ships in the Task 1 commit** (one contiguous device branch), with Task 2 delivering the CPU-only structural validation (`device_seam_test.rs`). Task 1's action explicitly defers the fold to Task 2; keeping each commit compiling + green forced the fold into the Task-1 branch and the validation into the Task-2 test. The bin_id range-check (T-10-22), all-or-nothing (T-10-23), and teardown (T-10-24) mitigations are all present and each commit's `cargo test -p cb-train` is green.
2. **Host-eligibility predicate added beyond the backend gate.** The backend's `begin_device_training` cannot see cb-train-only complexity (CTR, ordered, ranking, penalties, monotone, multi-dim, sampling, perturbation, eval). A host predicate gates those OFF before `begin` is even called, so a config the depth-1 device grower does not implement can never be accepted (a correctness requirement, Rule 2 — without it a RMSE/depth-1/Plain fixture that ALSO uses e.g. ordered boosting could be mis-accepted).

## Known Stubs
None. The device path is fully wired end-to-end from `fit()` (host eligibility -> begin -> per-iteration grow -> bin_id->border fold -> approx update -> tree push -> end teardown). The uncovered configs are intentional `Ok(false)` / host-ineligible CPU-fallback gates (D-04), not stubs. The authoritative device leaf-value oracle (RMSE ~1e-9 / Logloss <=1e-5) is the human-gated Kaggle CUDA run in 10-09 — this plan makes it reachable.

## Threat Flags
None beyond the plan's `<threat_model>`. T-10-22 (out-of-range border index) mitigated: bin_id range-checked with checked `.get` -> typed `OutOfRange`, no panic/index (tested). T-10-23 (silent model corruption via tree mixing) mitigated: per-fit all-or-nothing, Ok(None) after commit is a typed error (tested). T-10-24 (leaked device session) mitigated: `DeviceSessionGuard` ends the session on every exit path incl. the error path.

## Next Phase Readiness
- 10-09 (Kaggle CUDA authoritative oracle) can now drive the depth-1 device path from the public `fit()`; the in-env proof here is CPU-only (mock Runtime) fold correctness + the 231-lib / 53-integration D-04 no-regression suite. The GPU end-to-end RMSE/Logloss sign-off remains human-gated.
- No cb-backend dependency was added to cb-compute or cb-train's device seam (the loop talks to the device only through the `cb_compute::Runtime` trait); the feature-unification landmine is not touched.

## Self-Check: PASSED
- Files: `crates/cb-train/src/boosting.rs` (modified — device branch + gate + guard + quantize helper), `crates/cb-train/tests/device_seam_test.rs` (created) — both FOUND.
- Commits: `8f7280b` (feat 10-08 wire seam), `d23c31d` (test 10-08 structural) — both FOUND in git log.
- `cargo test -p cb-train --lib` -> 231 passed / 0 failed; `--tests` -> 53 binaries all `test result: ok`; `--test device_seam_test` -> 4 passed / 0 failed. CPU path byte-unchanged (D-04).

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
