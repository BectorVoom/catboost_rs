---
phase: 21-cpu-split-finding-histogram-rewrite
verified: 2026-07-06T02:17:20Z
status: passed_with_override
score: 2/3 verified; PERF-01 accepted via human override (amended bar)
behavior_unverified: 0
overrides_applied: 1
overrides:
  - truth: "Per-tree CPU time is flat within noise across border_count 32→254 (PERF-01 roadmap Success Criterion #1)"
    decision: accept_and_amend
    accepted_by: echinops27@gmail.com
    accepted_at: 2026-07-06
    rationale: "'Flat within noise' proven algorithmically unachievable at the Spike-002 harness (n=10000, nf=20, depth=6): the residual n_bins scaling is the irreducible O(n_bins·n_leaves·n_features) split-scoring arithmetic (n-independent), which flatness would require n≫n_bins·n_leaves (n≥~100k) to dominate; official CatBoost is itself ~2.1× (not flat) here. Two parity-safe gap-closure cycles (21-06 algorithmic rewrite, 21-07 bit-identical scan+score fusion) collapsed the pre-Phase-21 ~105–454× per-core gap to a small constant (~3.5× 32→254) and exhausted the allocation lever; the only remaining lever cannot reach flat and re-incurs parity risk. Bar amended in ROADMAP.md + REQUIREMENTS.md to the demonstrated constant-factor / competitive-with-upstream target; PERF-02 (bit-identical parity) and PERF-03 (rayon + reusable scratch, ~100× fewer hot-path allocations) fully verified. Phase 21 accepted as delivered."
re_verification:
  previous_status: gaps_found
  previous_score: 2/3
  gaps_closed: []
  gaps_remaining: []
  regressions: []
gaps:
  - truth: "Per-tree CPU time is flat within noise across border_count 32→254 (PERF-01 roadmap Success Criterion #1, the histogram fingerprint)"
    status: accepted_with_override
    reason: "21-07 (the second dedicated gap-closure attempt) fused the O(n_bins) border scan with the split-score fold into ONE pass (scan_and_score_borders + multi_dim_split_score_into), eliminating the Vec<Vec<Vec<LeafStats>>> materialization and the per-candidate num/den score Vecs entirely from the hot path — a real, independently-verified, bit-identical (to_bits() exact) refactor. But the literal roadmap bar ('flat within noise 32→254') is STILL not met, and — critically — 21-07 itself did not even reach its own more modest self-imposed target (~2.3-2.6x). Independently reproduced (4 runs, this verifier, not sourced from SUMMARY): 32->254 per-tree-ms ratios of 3.57x, 3.64x, 2.50x, 3.53x (median ~3.55x), closely matching (in fact slightly higher than) the executor's own self-reported ~3.33x median. This EMPIRICALLY reconfirms, for the second consecutive gap-closure cycle, that allocation churn was never the dominant cost — the residual n_bins-linear term is the irreducible O(n_bins*n_leaves*n_features) split-scoring ARITHMETIC (a fresh sum_f64 fold over 2*n_parent leaves per border per feature per level), which is n-independent and therefore does not shrink as the row count grows relative to n_bins*n_leaves at this harness size (n=10000, nf=20, depth=6, n_bins*n_leaves≈16K>10K). The executor's own root-cause analysis (21-07-SUMMARY.md, Issues Encountered + key-decision D5) is candid and matches the code-level evidence found by this verifier."
    artifacts:
      - path: "crates/cb-compute/src/histogram.rs"
        issue: "scan_and_score_borders (line 869) is a genuine single-pass fusion — no full Vec<Vec<Vec<LeafStats>>> materialization, no per-candidate score Vec (confirmed absent from the hot path; the old scan_borders_to_leaf_stats at line 745 is retained only as the bit-identity reference in tests and doc comments, grep-confirmed zero production call sites) — but the O(n_bins) per-border loop still runs one fresh sum_f64 fold per border per feature per level; this arithmetic cost is unaffected by removing the allocation, so it remains linear in n_bins."
      - path: "crates/cb-train/src/tree.rs"
        issue: "select_level_plain (line 850), select_level_perturbed (line 942), and best_split_for_leaf (line 1160) are all confirmed rewired to scan_and_score_borders (grep count = 3 call sites, matching the plan's key_links), but none of this changes the fundamental per-border-per-feature-per-level scoring work, which remains the dominant, n-independent, non-flat cost at this harness size."
    missing:
      - "A human decision: either (a) accept the current ~3.5-3.6x (down from the pre-21-06 ~4.3x baseline, via two allocation-focused gap-closure cycles that have now demonstrably exhausted the allocation-side lever) as the terminal CPU baseline and amend/retire the roadmap's literal 'flat within noise' wording for PERF-01, or (b) commission a THIRD follow-up plan that changes the scoring ALGORITHM itself (not just allocation) — e.g. avoiding the full per-border-per-feature rescore by only rescoring the SAME candidates that changed, or restructuring to an n-dependent-only cost model — which is a materially larger, more invasive change than either 21-06 or 21-07 attempted, with its own fresh parity-reproof burden."
---

# Phase 21: CPU Split-Finding Histogram Rewrite Verification Report

**Phase Goal:** CPU training split-finding matches CatBoost's histogram/bucket-stats algorithm — per-feature bin histograms + subtraction trick + parallelism — collapsing the ~250–450× single-thread slowdown Spike 002 measured, while preserving the ≤10⁻⁵ CPU parity bar, across ALL CPU grow policies (oblivious `SymmetricTree`, non-symmetric `Depthwise`/`Lossguide`, and the online-CTR-feature scoring path).
**Verified:** 2026-07-06T02:17:20Z
**Status:** passed_with_override (PERF-01 accepted + amended 2026-07-06; PERF-02/03 verified)
**Re-verification:** Yes — after 21-07, the SECOND gap-closure plan targeting the sole prior failure (PERF-01 flatness), via a parity-safe scan+score fusion

## Goal Achievement

### Observable Truths (Roadmap Success Criteria — the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | (PERF-01) CPU oblivious split search builds per-feature histograms + subtraction trick, replacing the per-candidate full-dataset rescan; per-tree time no longer scales with `border_count` (flat within noise 32→254) | ✗ FAILED (partial, further improved, target still not reached) | 21-07 fused the O(n_bins) running-prefix border scan with the split-score fold into ONE pass (`scan_and_score_borders`, histogram.rs:869; `multi_dim_split_score_into`, score.rs:148), eliminating the `Vec<Vec<Vec<LeafStats>>>` materialization and the per-candidate num/den score `Vec`s entirely — verified in code (grep: zero production call sites remain for the materializing `scan_borders_to_leaf_stats`; all 3 hot callers `select_level_plain`/`select_level_perturbed`/`best_split_for_leaf` route through the fused path, confirmed at tree.rs:850,942,1160). The fusion is genuinely BIT-IDENTICAL, not merely ≤1e-5: both new equivalence tests (`fused_scan_score_bit_identical`, `multi_dim_split_score_into_matches_alloc`) assert exact `to_bits()` equality and both PASS (re-run by this verifier). **The flatness bar is still NOT met, and 21-07's own more modest self-target (~2.3–2.6×) is ALSO not met.** Independently reproduced 4× by this verifier (`CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture`, not sourced from the SUMMARY): 32→254 ratios of 3.57×, 3.64×, 2.50×, 3.53× (median ≈3.55×) — matching (in fact slightly higher than) the executor's self-reported ~3.33× median. This is the SECOND consecutive gap-closure cycle (21-06 then 21-07) that fails to close this roadmap Success Criterion; both cycles targeted allocation, and 21-07's own measured decomposition (flat floor ≈1.7ms + ~0.026 ms/bin) demonstrates the residual is now provably the irreducible `O(n_bins·n_leaves·n_features)` scoring arithmetic, not allocation churn — the allocation lever is now exhausted. |
| 2 | (PERF-02) ALL CPU grow policies (`SymmetricTree`, `Depthwise`, `Lossguide`) AND the online-CTR-feature scoring path use the histogram scorer; every shipped ≤1e-5 CPU oracle fixture stays bit-exact (or ≤1e-5 oracle-equivalent on the reordered TRUE-side scan) | ✓ VERIFIED (regression-checked, re-confirmed after fusion) | Re-ran the FULL atomic gate myself, independently: `CARGO_INCREMENTAL=0 cargo test -p cb-train --release --no-fail-fast` → **413 passed / 1 failed** (tallied via a `grep`+`python3` summation over the full raw log, not trusted from the SUMMARY), the sole failure being the documented pre-existing `monotone_non_symmetric_and_region_are_typed_errors` (confirmed unrelated — same test that failed identically pre-21-07); `cargo test -p cb-compute --release` → **212 passed / 0 failed** (197+5+1+9+0 across the 5 test binaries, summed independently), including the two NEW 21-07 bit-identity guards (`histogram_test::fused_scan_score_bit_identical` and `score_test::multi_dim_split_score_into_matches_alloc`, both confirmed present and passing in the raw log). Zero additional oracle fixtures flipped from the fusion — the atomic zero-flip insurance gate is real and reproducible, not merely trusted from the SUMMARY. |
| 3 | (PERF-03) split search parallelized over features/candidates (rayon) with reusable scratch buffers (no per-candidate allocation storm); documented end-to-end speedup + stated per-core efficiency factor vs official CatBoost 1-thread | ✓ VERIFIED (regression-checked, and further improved by 21-07) | Re-ran `cargo test -p cb-train --release --test rayon_determinism_test` myself → **2/2 pass** (SymmetricTree + Depthwise byte-identical), no regression from the 21-07 fusion. Code-level confirmation: the per-level `Vec<Vec<Vec<LeafStats>>>` materialization (`n_borders·dim·2·n_parent`) AND the per-candidate score-side `num_terms`/`den_terms` `Vec`s (allocated `n_bins·n_features×` per level in the old two-step path) are both GONE from the hot path — `scan_and_score_borders` allocates its `per_dim`/`num_scratch`/`den_scratch`/`total_*`/`acc_false_*`/`col` buffers ONCE per feature per level and reuses them across every candidate border (confirmed by direct code read of histogram.rs:869-993: all mutable scratch is declared once, before the border loop, and only `.clear()`/indexed-mutated inside it). Scratch is per-rayon-task-local (allocated inside the `select_level_plain` `.map` closure — confirmed at tree.rs:833-879), so no cross-thread mutation. `cb-compute/Cargo.toml` confirmed still rayon/cubecl-free. |

**Score:** 2/3 roadmap Success Criteria verified; PERF-01 remains FAILED against its literal quantitative bar for the SECOND consecutive verification cycle, despite two dedicated gap-closure plans (21-06, 21-07) that both delivered real, substantial, independently-verified engineering improvements on the allocation side.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-compute/src/score.rs::multi_dim_split_score_into` | Scratch-reusing sibling of `multi_dim_split_score`, byte-identical, zero per-call heap Vec | ✓ VERIFIED | Present (lines 148-198): takes caller-owned `num_scratch`/`den_scratch: &mut Vec<f64>`, does `.clear()` + the SAME push sequence + `sum_f64` reduction as the original. `multi_dim_split_score` is now a thin wrapper delegating to it (lines 110-130) — `grep -c 'multi_dim_split_score_into' crates/cb-compute/src/score.rs` = 5 (definition + delegation call + doc references), confirming the ≥2 acceptance bar. |
| `crates/cb-compute/src/histogram.rs::scan_and_score_borders` | Fused single-pass border scan + score, no `Vec<Vec<Vec<LeafStats>>>` materialization | ✓ VERIFIED | Present (lines 869-993). Read in full by this verifier: per-parent totals computed once outside the border loop; the ONLY cross-border accumulation is the FALSE-side running prefix (`acc_false_w`/`acc_false_d`, the already-gated 21-06 values); the per-border score fold is a FRESH call to `multi_dim_split_score_into` per border (no running num/den across borders) — confirmed NO forbidden cross-border score reorder was introduced. |
| `crates/cb-train/src/tree.rs` (3 callers) | `select_level_plain` / `select_level_perturbed` / `best_split_for_leaf` all routed through `scan_and_score_borders` | ✓ VERIFIED | `grep -c 'scan_and_score_borders' crates/cb-train/src/tree.rs` = 8 (3 call sites at lines 850, 942, 1160 + 5 doc-comment references); `scan_borders_to_leaf_stats` (the materializing function) has zero remaining call sites in `tree.rs` production code. |
| `crates/cb-compute/src/histogram_test.rs::fused_scan_score_bit_identical` | Bit-identity equivalence test, exact f64 (`to_bits()`) equality, Cosine+L2, dim=1+dim=2 | ✓ VERIFIED | Present and read in full; uses `assert_eq!(fused[b].to_bits(), ref_score.to_bits(), ...)` — genuinely exact, not approx. Re-run by this verifier: PASS. |
| `crates/cb-compute/src/score_test.rs::multi_dim_split_score_into_matches_alloc` | Scratch-reuse equivalence test including a third pass reusing the same scratch | ✓ VERIFIED | Present and read in full; asserts `.to_bits()` equality for dim=1/dim=2/Cosine/L2 AND a third call reusing the SAME (cleared) scratch buffer, proving clear-then-refill order-preservation. Re-run by this verifier: PASS. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `tree.rs::select_level_plain / select_level_perturbed / best_split_for_leaf` | `histogram.rs::scan_and_score_borders` | fused single-pass border scan scoring each border inline via reused scratch | ✓ WIRED | All 3 call sites confirmed by direct code read (lines 850, 942, 1160); per-task-local scratch (rayon `.map` closure) confirmed for `select_level_plain`. |
| `histogram.rs::scan_and_score_borders` | `score.rs::multi_dim_split_score_into` | per-border FRESH `sum_f64` fold in canonical dimension-then-leaf order, no cross-border reorder | ✓ WIRED | Confirmed by code read (histogram.rs:983-989) — one `multi_dim_split_score_into` call per border, inside the border loop, over the per-border-refreshed `per_dim` scratch. |
| `histogram.rs::scan_and_score_borders` | `cb-core::sum_f64` | the only sanctioned per-border fold | ✓ WIRED | Confirmed transitively via `multi_dim_split_score_into` → `sum_f64` (score.rs:163, 190). |
| `tree.rs` | `cb-backend` | (must NOT exist) | ✓ CONFIRMED ABSENT | `grep -rnE 'use +cb_backend|cb_backend::' crates/cb-train/src` → no matches. |
| `cb-compute/Cargo.toml` | `rayon`/`cubecl` | (must NOT exist) | ✓ CONFIRMED ABSENT | Only comment lines forbidding them; no dependency lines added by 21-07. |

### Behavioral Spot-Checks / Test Runs (performed independently by this verifier, not sourced from SUMMARY claims)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| cb-train full suite (atomic parity gate) | `CARGO_INCREMENTAL=0 cargo test -p cb-train --release --no-fail-fast` | 413 passed / 1 failed (independently tallied by summing every `X passed; Y failed` line in the raw log via a small python script) | ✓ PASS (matches SUMMARY exactly; sole failure is the documented pre-existing `monotone_non_symmetric_and_region_are_typed_errors`, unrelated to this phase) |
| cb-compute full suite | `cargo test -p cb-compute --release` | 212 passed / 0 failed (197+5+1+9+0 across 5 test binaries) | ✓ PASS (matches SUMMARY exactly; includes the 2 new 21-07 bit-identity guards, confirmed present in the log by name) |
| rayon determinism | `cargo test -p cb-train --release --test rayon_determinism_test` | 2/2 pass | ✓ PASS |
| new bit-identity guards present + passing | (embedded in the cb-compute run above) | `histogram_test::fused_scan_score_bit_identical ... ok`, `score_test::multi_dim_split_score_into_matches_alloc ... ok` | ✓ PASS |
| PERF-01 n_bins sweep (independent reproduction, 4 runs, this verifier) | `CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture` | 32→254 ratios: 3.57×, 3.64×, 2.50×, 3.53× (median ≈3.55×) | ✗ FAIL (confirms the gap remains open; matches/slightly exceeds the SUMMARY's self-reported ~3.33× median, well within measurement noise) |
| `Vec<Vec<Vec<LeafStats>>>` gone from production hot path | `grep -rn 'Vec<Vec<Vec<LeafStats>>>' crates/cb-compute/src/histogram.rs crates/cb-train/src/tree.rs` | only in `scan_borders_to_leaf_stats`'s signature/body (retained as the bit-identity test reference, zero production callers) + one doc comment | ✓ PASS (the materializing type is gone from the hot path, confirmed) |
| No `cb-backend` dependency introduced | `grep -rnE 'use +cb_backend|cb_backend::' crates/cb-train/src/tree.rs` | 0 matches | ✓ PASS |
| `cb-compute` stays rayon/cubecl-free | `grep -nE 'rayon|cubecl' crates/cb-compute/Cargo.toml` | only forbidding comments, no dep lines | ✓ PASS |
| No debt markers introduced | `grep -rn 'TBD\|FIXME\|XXX' crates/cb-compute/src/score.rs crates/cb-compute/src/histogram.rs crates/cb-train/src/tree.rs crates/cb-compute/src/histogram_test.rs crates/cb-compute/src/score_test.rs` | 0 matches | ✓ PASS |
| Git commits for 21-07 exist | `git show --stat 85c63cd`, `git show --stat 5aa1be6` | Both commits present, messages match the SUMMARY's claimed Task 1 (RED) / Task 2 (GREEN) content | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|-----------------|-------------|--------|----------|
| PERF-01 | 21-01, 21-02, 21-06, 21-07 | Per-feature histograms + subtraction trick replace per-candidate rescan; flat n_bins scaling | ⚠️ PARTIAL (unchanged verdict, despite a SECOND dedicated gap-closure attempt) | Two gap-closure cycles (21-06 scan/build/advance rewrite, 21-07 scan+score fusion) delivered real, independently-verified allocation-side optimizations; flatness bar NOT achieved (independently re-measured ~3.5-3.6× median vs the roadmap's "flat" bar and 21-07's own ~2.3-2.6× self-target), and the residual is now demonstrated (by the executor's own measured floor+slope decomposition, corroborated by this verifier's code read) to be irreducible split-scoring arithmetic, not allocation. |
| PERF-02 | 21-01, 21-02, 21-03, 21-04, 21-06, 21-07 | All CPU grow policies + CTR path use histogram scorer; bit-exact / ≤1e-5-equivalent parity | ✓ SATISFIED | Re-verified independently by this verifier after the 21-07 fusion; atomic full-suite gate re-run, matching the SUMMARY's exact tally (413/1, 212/0) with zero additional flips. The fusion is proven bit-identical (exact `to_bits()` equality) by two new equivalence tests, both re-run and passing. |
| PERF-03 | 21-05, 21-06, 21-07 | rayon parallelism, reusable scratch, determinism, documented speedup | ✓ SATISFIED (further improved) | Re-verified; determinism test re-run 2/2; the full per-candidate materialization (`Vec<Vec<Vec<LeafStats>>>`) and per-candidate score-side Vecs are now confirmed GONE from the hot path (not merely reduced, as after 21-06) — replaced by reused scratch, confirmed by direct code read. |

**No orphaned requirements**: PERF-01/02/03 are the only requirements REQUIREMENTS.md maps to Phase 21, and all three appear in at least one plan's `requirements:` frontmatter (21-01 through 21-07, including the new 21-07 which declares `[PERF-01, PERF-03]`). REQUIREMENTS.md itself still marks all three `[x]` Complete and "Phase 21 | Complete" in its traceability table — this verifier's independent finding, unchanged from the prior verification cycle, is that PERF-01's completion claim is **still not fully supported** by the roadmap's own literal acceptance bar (see Gaps Summary).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cb-train/src/tree.rs` | 711 | `fn hist_add` is dead code (compiler warning: "associated function ... never used", re-confirmed by this verifier's own build output during the atomic gate re-run) | ℹ️ Info | Same pre-existing orphaned leftover noted in the prior verification cycle (superseded by `add_relocated`/`relocate_sub`), NOT introduced or touched by 21-07. Non-blocking. |

No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in any of the 21-07-modified files (`score.rs`, `score_test.rs`, `histogram.rs`, `histogram_test.rs`, `tree.rs`) — independently grepped by this verifier, zero matches.

## Gaps Summary

**PERF-02 and PERF-03 remain fully delivered and show no regression** from the 21-07 fusion — independently re-run by this verifier (not trusted from SUMMARY claims): the atomic full-oracle-suite parity gate (413/1 cb-train, 212/0 cb-compute, matching the SUMMARY's tally exactly, with zero additional flips) and the rayon determinism test (2/2) both pass. The scan+score fusion is genuinely bit-identical — both new equivalence tests assert exact `to_bits()` equality (not approximate), and this verifier confirmed by direct code read that the per-border score fold remains a FRESH `sum_f64` with no cross-border num/den reorder introduced. PERF-03 is now delivered MORE completely than at the prior verification: the full `Vec<Vec<Vec<LeafStats>>>` materialization and the per-candidate score-side `Vec`s are confirmed entirely gone from the production hot path (not merely reduced).

**PERF-01 remains the sole open gap, unresolved for a SECOND consecutive gap-closure cycle.** 21-07 — the plan specifically dispatched to close the remaining allocation-side lever after 21-06 — delivered exactly what it set out to (a bit-identical scan+score fusion eliminating essentially all remaining per-candidate allocation in the hot path), and this verifier independently confirmed the fusion is real, correct, and non-regressive. But the roadmap's Success Criterion #1 ("flat within noise across 32→254 bins") is STILL not met, and — notably — 21-07's own more modest self-imposed target (~2.3–2.6×) was ALSO not met: this verifier's independent 4-run reproduction measured a 32→254 median ratio of ~3.55×, essentially unchanged from (if anything marginally worse than, within noise, than) the ~3.33× the executor self-reported and the ~3.3-3.6× measured at the prior (21-06) verification cycle.

This is a materially important finding: **two consecutive gap-closure plans (21-06 and 21-07), both targeting allocation/materialization overhead, have now demonstrably exhausted that lever** without closing the gap. The executor's own root-cause analysis — corroborated independently by this verifier's code read of `scan_and_score_borders` — is that the residual cost is the `O(n_bins·n_leaves·n_features)` split-scoring ARITHMETIC (a fresh `sum_f64` fold over `2·n_parent` leaves per candidate border per feature per level), which is n-INDEPENDENT (the histogram cell count does not depend on row count) and therefore does not shrink at this benchmark's scale (n=10000, nf=20, depth=6, where `n_bins·n_leaves≈16K > n=10K`). Flatness would require `n ≫ n_bins·n_leaves` (roughly `n≥100k`), a regime this harness does not test, and official CatBoost itself is measured (per the executors' own citation, not re-derived here) at ~2.1× at this same size — i.e., not flat either.

**This is the second consecutive verification cycle where PERF-01's literal flatness bar is not met, now after TWO dedicated gap-closure plans exhausting the identified allocation-side lever.** This remains a BLOCKER-tier finding per protocol (a roadmap Success Criterion has now failed to close across two remediation attempts) requiring an explicit human decision. The prior verification's suggested override (in the previous revision of this file) was never accepted — its `accepted_by`/`accepted_at` fields were left as unfilled placeholders (`"<name>"`, `"<ISO timestamp>"`), so no override is currently in effect; this re-verification does not fabricate acceptance on the human's behalf.

Two paths forward, unchanged in kind from the prior cycle but now backed by stronger evidence that the allocation lever is exhausted:

1. **Accept** the current ~3.5-3.6× CPU baseline (down from the pre-21-06 ~4.3× baseline, via two allocation-focused gap-closure cycles that have now demonstrably reached diminishing returns on that lever) and amend/retire the roadmap's literal "flat within noise" wording for PERF-01 — proceeding to Phase 22 with this as the documented, honest terminal state of the CPU split-finding allocation optimization.
2. **Commission a third, more invasive follow-up plan** that changes the underlying scoring ALGORITHM (not merely allocation) — a materially larger change than either 21-06 or 21-07 attempted, carrying its own fresh parity-reproof burden and genuine risk to the ≤1e-5 crux.

**This looks intentional and well-engineered across two cycles, not careless.** To accept the current ~3.5-3.6× (rather than fully-flat) CPU baseline and proceed to Phase 22, add to this file's frontmatter:

```yaml
overrides:
  - must_have: "Per-tree CPU time is flat within noise across border_count 32→254 (PERF-01)"
    reason: "Two dedicated gap-closure plans (21-06: O(n_bins^2)->O(n_bins) scan rewrite + flat scatter-add build + retained-parent advance; 21-07: bit-identical scan+score fusion eliminating the Vec<Vec<Vec<LeafStats>>> materialization + per-candidate score Vecs) have exhausted the allocation-side optimization lever, improving the 32->254 ratio from ~4.3x (pre-21-06) to ~3.5-3.6x (post-21-07), independently re-measured by two separate verification cycles. The residual is proven (by measured floor+slope decomposition, corroborated by code-level inspection of the fused scan+score pass) to be the irreducible O(n_bins*n_leaves*n_features) split-scoring arithmetic, which is n-independent and does not flatten at this benchmark scale (n_bins*n_leaves~16K > n=10K); official CatBoost is itself ~2.1x (not flat) at this same size. Reaching literal flatness requires either n>=100k (outside this harness's regime) or an invasive scoring-algorithm change carrying fresh parity-reproof risk, judged out of scope for a third gap-closure iteration at this time."
    accepted_by: "<name>"
    accepted_at: "<ISO timestamp>"
```

Alternatively, a THIRD follow-up plan implementing an actual scoring-algorithm change (not further allocation hoisting) — explicitly re-running the atomic full-oracle-suite gate before landing — would be required to close this gap outright before Phase 22.

Note also that `.planning/REQUIREMENTS.md` and `.planning/ROADMAP.md` currently mark PERF-01 as `[x]` Complete and Phase 21 as "(completed 2026-07-05)" — this predates both 21-06's and 21-07's honest partial outcomes and should be reconciled once the human decision below is made (either update to reflect an accepted override, or reopen for a third follow-up plan).

---

_Verified: 2026-07-06T02:17:20Z_
_Verifier: Claude (gsd-verifier)_
