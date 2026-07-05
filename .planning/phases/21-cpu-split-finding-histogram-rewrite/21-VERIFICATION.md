---
phase: 21-cpu-split-finding-histogram-rewrite
verified: 2026-07-05T12:04:02Z
status: gaps_found
score: 2/3 must-haves verified
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "Per-tree CPU time is flat within noise across border_count 32→254 (PERF-01 roadmap Success Criterion #1, the histogram fingerprint)"
    status: failed
    reason: "Reproduced independently (cargo test --release CB_PERF=1 perf_baseline_test): per-tree ms goes 36.5 (nbins=32) → 155.98 (nbins=254), a ~4.3× increase for ~8× more bins — still clearly scaling with border_count, not flat. This matches the executors' own documented finding (21-02-SUMMARY.md and 21-05-SUMMARY.md both self-report this as PARTIAL, not met). Root cause per the SUMMARYs: `scan_borders_to_leaf_stats` is O(n_bins²) per feature per level (recomputes both prefix sums fresh per border) and `build_bucket_histogram` allocates fresh per-level scratch of size ∝ n_bins — the running-prefix O(n_bins) rewrite and scratch-buffer reuse that would collapse this were explicitly deferred by 21-01 → 21-02 → 21-05, each citing the same reason: the TRUE-side prefix reorder is a parity hazard (Pitfall 2 / summation-order risk) that must be gated by an atomic full-oracle-suite run, which disk pressure prevented running in one pass during 21-05."
    artifacts:
      - path: "crates/cb-compute/src/histogram.rs"
        issue: "scan_border_to_leaf_stats (line 426) recomputes false/true prefix sums fresh per border call — O(n_bins) per border, O(n_bins²) per feature per level — instead of a running-prefix single O(n_bins) pass"
      - path: "crates/cb-train/src/tree.rs"
        issue: "GrowScratch/best_split_for_leaf/CTR histogram builds allocate a fresh Vec<f64> of size ∝ n_leaves·n_features·n_bins per level/leaf rather than reusing a cleared scratch buffer across levels"
    missing:
      - "A running-prefix O(n_bins) rewrite of the border scan (with a parity-safe TRUE-side summation strategy, gated by an atomically-run full oracle suite so no tie-flip goes uncaught)"
      - "Scratch-buffer reuse in build_bucket_histogram/GrowScratch so per-level histogram allocation is eliminated, not just per-candidate allocation"
      - "A re-run of the CB_PERF n_bins sweep after the above, confirming flat (within-noise) timing 32→254 bins, matching official CatBoost's ~2.1× fingerprint for the same 8× bin range"
---

# Phase 21: CPU Split-Finding Histogram Rewrite Verification Report

**Phase Goal:** CPU training split-finding matches CatBoost's histogram/bucket-stats algorithm — per-feature bin histograms + subtraction trick + parallelism — collapsing the ~250–450× single-thread slowdown Spike 002 measured, while preserving the ≤10⁻⁵ CPU parity bar, across ALL CPU grow policies (oblivious `SymmetricTree`, non-symmetric `Depthwise`/`Lossguide`, and the online-CTR-feature scoring path).
**Verified:** 2026-07-05T12:04:02Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria — the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | (PERF-01) CPU oblivious split search builds per-feature histograms + subtraction trick, replacing the per-candidate full-dataset rescan; per-tree time no longer scales with `border_count` (flat within noise 32→254) | ✗ FAILED (partial) | The rescan replacement IS real and verified in code (`select_level_plain`/`select_level_perturbed` no longer call `score_candidate`; `GrowScratch` builds one histogram per level; subtraction trick present in `GrowScratch::advance`). **But the "flat within noise 32→254" bar is NOT met** — independently reproduced: `CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture` → nbins=32: 36.5ms, nbins=254: 155.98ms (~4.3× for ~8× bins, clearly linear-ish, not flat). Executors' own SUMMARYs (21-02, 21-05) self-report this as PARTIAL with the same root cause and numbers in the same ballpark (44.1→355.0 single-thread, 39.9→145.4 16-thread). |
| 2 | (PERF-02) ALL CPU grow policies (`SymmetricTree`, `Depthwise`, `Lossguide`) AND the online-CTR-feature scoring path use the histogram scorer; every shipped ≤1e-5 CPU oracle fixture stays bit-exact | ✓ VERIFIED | Code-verified: `select_level_plain`/`select_level_perturbed` (tree.rs:818,893), `best_split_for_leaf` (tree.rs:1046, shared by Depthwise/Lossguide/Region), `select_level_ctr_aware`/`build_ctr_aware_histogram`/`score_candidate_ctr_aware` (tree.rs:2152,2305) all call `build_bucket_histogram`/`scan_border(s)_to_leaf_stats`; `score_candidate` (the old per-candidate rescan) is defined but never called (`#[allow(dead_code)]`, only a WR-01 reference scorer). Ran `cargo test -p cb-compute --release` (all green, 208 tests) and `cargo test -p cb-train --release --no-fail-fast` (every integration binary + 240-test lib suite green) MYSELF — only failure is `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors`, confirmed (via git log on `boosting.rs`/`monotone_oracle_test.rs`) to be a stale pre-Phase-21 assertion (Region-OUT rejection was lifted in Phase 12/GPUT-18, this test predates Phase 21 by ~15 phases) — unrelated to histogram scoring, not a regression. |
| 3 | (PERF-03) split search parallelized over features/candidates (rayon) with reusable scratch buffers (no per-candidate allocation storm); documented end-to-end speedup + stated per-core efficiency factor vs official CatBoost 1-thread | ✓ VERIFIED | `grep -n 'into_par_iter\|par_chunks_mut' crates/cb-train/src/tree.rs` shows 3 real parallel sites (`select_level_plain` scoring, `GrowScratch::new` binning, `best_split_for_leaf` binning); `rayon = "1.12.0"` in root `Cargo.toml [workspace.dependencies]` + `cb-train/Cargo.toml`, confirmed absent from `cb-compute/Cargo.toml` (`CB-COMPUTE-CLEAN`). Ran `cargo test -p cb-train --release --test rayon_determinism_test` MYSELF → 2/2 pass (SymmetricTree + Depthwise byte-identical across two live-pool runs). Per-candidate rescan allocation storm is eliminated (histogram built once per level/leaf, not per candidate). Documented speedup numbers (21-05-SUMMARY.md) cross-checked against Spike-002's raw recorded artifacts (`catboost_results.txt`, `README.md`) — the cited official-CatBoost 1-thread figures (5.11, 6.34, 10.84, 6.53, 19.2, 18.5 ms) are real, not fabricated. |

**Score:** 2/3 roadmap Success Criteria verified; 1 explicitly failed (self-documented by the executors as PARTIAL, independently reproduced by this verifier).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-compute/src/histogram.rs` | `BucketHistogram` + build + remove + bin_of + prefix-scan primitives, pure host Rust, `sum_f64`-routed | ✓ VERIFIED | 490 lines; `BucketHistogram` (188), `build_bucket_histogram` (337), `remove` (268), `bin_of` (305), `scan_border_to_leaf_stats`/`scan_borders_to_leaf_stats` (426/481) all present. No `.unwrap()`, no raw indexing (only `.get()`-style / Vec macros / attributes), no `cb_backend`/`rayon`/`cubecl` reference. |
| `crates/cb-compute/src/histogram_test.rs` | bit-exact equivalence tests | ✓ VERIFIED | 318 lines; `cargo test -p cb-compute --release` all green. |
| `crates/cb-train/src/tree.rs` | histogram-backed `select_level_plain`/`_perturbed`, `best_split_for_leaf`, `select_level_ctr_aware`, `GrowScratch`, rayon parallel sections | ✓ VERIFIED | All symbols present and wired (see truths table); 2938 lines. |
| `crates/cb-train/src/leaf_wise_scorer_test.rs` | leaf-wise histogram-vs-rescan equivalence tests | ✓ VERIFIED | 157 lines; mounted via `#[path]` as `tree::leaf_wise_scorer` (not an embedded `mod tests`), 3 tests pass. |
| `crates/cb-train/tests/rayon_determinism_test.rs` | byte-identical-model determinism test under rayon | ✓ VERIFIED | 172 lines; 2/2 tests pass (verified by this verifier, not just SUMMARY claim). |
| `Cargo.toml` / `crates/cb-train/Cargo.toml` | `rayon = "1.12.0"` workspace dep, cb-train only | ✓ VERIFIED | Confirmed present; `cb-compute/Cargo.toml` confirmed clean of rayon/cubecl. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `tree.rs::select_level_plain/_perturbed` | `cb-compute::histogram.rs` | `build_bucket_histogram` + `scan_borders_to_leaf_stats` | ✓ WIRED | Confirmed by grep + successful test run; `score_candidate(` no longer called from these functions. |
| `tree.rs::best_split_for_leaf` | `cb-compute::histogram.rs` | per-leaf `build_bucket_histogram` + `scan_borders_to_leaf_stats` | ✓ WIRED | Confirmed; shared by Depthwise/Lossguide/Region. |
| `tree.rs::select_level_ctr_aware` | `cb-compute::histogram.rs` | `build_ctr_aware_histogram` (float ∪ CTR bin columns) + `scan_border_to_leaf_stats` | ✓ WIRED | Confirmed. |
| `tree.rs` (3 sites) | `rayon` | `into_par_iter`/`par_chunks_mut`, ordered collect | ✓ WIRED | Confirmed; determinism test proves the ordering is safe. |
| `tree.rs` | `cb-backend` | (must NOT exist) | ✓ CONFIRMED ABSENT | `grep -rnE 'use +cb_backend|cb_backend::' crates/cb-train/src/tree.rs` → NO-NEW-BACKEND-SEAM. |

### Behavioral Spot-Checks / Test Runs (performed by this verifier, not sourced from SUMMARY claims)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| cb-compute full suite | `cargo test -p cb-compute --release` | all green | ✓ PASS |
| cb-train full suite | `cargo test -p cb-train --release --no-fail-fast` | all green except 1 pre-existing unrelated failure | ✓ PASS (with documented, confirmed pre-existing exception) |
| Stale monotone test is pre-existing, not a regression | `cargo test -p cb-train --release --test monotone_oracle_test`; `git log` on `boosting.rs`/`monotone_oracle_test.rs` | Fails identically; Region-OUT rejection lifted in Phase 12 (GPUT-18), test predates Phase 21 | ✓ CONFIRMED non-regression |
| rayon determinism | `cargo test -p cb-train --release --test rayon_determinism_test` | 2/2 pass | ✓ PASS |
| clippy on new histogram code | `cargo clippy -p cb-compute --all-targets -- -D warnings` (filtered to non-`cb-data` output) | 0 warnings attributable to `histogram.rs` (4 errors, all in unrelated pre-existing `cb-data/src/text/bigram_dictionary.rs`) | ✓ PASS |
| PERF-01 n_bins sweep (independent reproduction) | `CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture` | nbins 16→254: 30.2 → 36.5 → 51.0 → 156.0 ms — NOT flat | ✗ FAIL (confirms the gap) |
| Cited official-CatBoost numbers are real, not fabricated | `grep` Spike-002's `catboost_results.txt`/`README.md` for the exact figures cited in 21-05-SUMMARY.md | All cited figures (5.11, 6.34, 10.84, 6.53, 19.2, 18.5) found verbatim in the spike artifacts | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PERF-01 | 21-01, 21-02 | Per-feature histograms + subtraction trick replace per-candidate rescan; flat n_bins scaling | ⚠️ PARTIAL | Rescan replacement done and verified; flatness NOT achieved (see truth #1). |
| PERF-02 | 21-01, 21-02, 21-03, 21-04 | All CPU grow policies + CTR path use histogram scorer; bit-exact parity | ✓ SATISFIED | Verified across all 5 plans, independently re-run by this verifier. |
| PERF-03 | 21-05 | rayon parallelism, reusable scratch, determinism, documented speedup | ✓ SATISFIED | Verified; determinism test re-run by this verifier; speedup numbers cross-checked against Spike-002 raw data. |

No orphaned requirements: PERF-01/02/03 are the only requirements mapped to Phase 21 in REQUIREMENTS.md, and all three are covered by at least one of the 5 plans' `requirements:` frontmatter.

### Anti-Patterns Found

None blocking. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers, no `.unwrap()`, no raw indexing, no embedded `#[cfg(test)] mod tests` blocks (all test mounts are `#[path]`-based sibling files per project convention) in any of the phase's modified files (`histogram.rs`, `histogram_test.rs`, `tree.rs`, `leaf_wise_scorer_test.rs`, `rayon_determinism_test.rs`).

The one known non-green test (`monotone_non_symmetric_and_region_are_typed_errors`) is a pre-existing, out-of-scope, already-documented (`deferred-items.md`) failure unrelated to this phase's work — confirmed independently via git history (the assertion predates Phase 21 by ~15 phases; the code path it tests against was changed in Phase 12/GPUT-18, not Phase 21).

## Gaps Summary

The phase delivers the great majority of its goal: **all five in-scope CPU scoring paths (oblivious plain/perturbed, Depthwise, Lossguide, Region, online-CTR) are demonstrably converted from the per-candidate full-dataset rescan to per-feature/per-leaf bin histograms**, the subtraction trick is real and exercised on the oblivious path, rayon parallelism is real and proven deterministic, and **the full CPU oracle suite (bit-exact ≤1e-5 parity) is preserved** — independently re-run and confirmed by this verifier, not merely trusted from SUMMARY claims.

However, the roadmap's **PERF-01 Success Criterion is explicit and quantifiable**: "per-tree CPU time no longer scales with `border_count` (flat within noise across 32→254 bins)". This is **not met**. Independently reproduced: per-tree time still increases roughly linearly with `border_count` (only the constant factor collapsed, ~27–30×), not flat. The executors themselves flagged this honestly and consistently across three SUMMARYs (21-01, 21-02, 21-05) as a deliberate, safety-motivated deferral — the remaining lever (a running-prefix O(n_bins) rewrite of the border scan) touches summation order on the parity-critical TRUE-side fold, and was judged too risky to land without an atomic full-suite run, which the environment's disk pressure prevented during execution.

Net effect on the phase goal ("collapsing the ~250–450× single-thread slowdown"): the per-core gap **is** substantially closed — from 105–454× (pre-rewrite) down to 8.6–33× (single-thread) — a genuine 13–30× improvement. But it is not fully "collapsed" to CatBoost's own ~2.1× n_bins fingerprint, and the specific PERF-01 acceptance bar as literally written in ROADMAP.md is not satisfied.

This looks like a deliberate, well-reasoned, thoroughly-documented engineering trade-off, not a hidden or careless gap — but per verification protocol a roadmap Success Criterion that fails is a BLOCKER-tier finding requiring a human decision, not a silent pass.

**This looks intentional.** To accept this deviation and proceed to Phase 22 with the current 13–30× (rather than fully-flat) CPU baseline, add to this file's frontmatter:

```yaml
overrides:
  - must_have: "Per-tree CPU time is flat within noise across border_count 32→254 (PERF-01)"
    reason: "Dominant per-candidate rescan pathology eliminated (13-30x per-tree speedup, 105-454x -> 8.6-33x per-core gap vs official CatBoost); residual n_bins-linear scaling requires a parity-risky running-prefix scan reorder deferred for a follow-up under disk-pressure-free conditions with an atomic full-suite gate."
    accepted_by: "<name>"
    accepted_at: "<ISO timestamp>"
```

Alternatively, a small follow-up plan (21-06) implementing the running-prefix O(n_bins) scan + `build_bucket_histogram` scratch reuse, gated by an atomic (single, uninterrupted) full-suite run, would close this gap outright before Phase 22.

---

_Verified: 2026-07-05T12:04:02Z_
_Verifier: Claude (gsd-verifier)_
