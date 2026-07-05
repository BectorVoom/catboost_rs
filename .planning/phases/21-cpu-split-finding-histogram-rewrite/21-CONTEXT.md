# Phase 21: CPU Split-Finding Histogram Rewrite - Context

**Gathered:** 2026-07-05
**Status:** Ready for planning
**Source:** Spike 002/003/004 findings + user scope decisions (/gsd-plan-phase, 2026-07-05)

<domain>
## Phase Boundary

Rewrite the **CPU** training split-finding to use CatBoost's histogram/bucket-stats
algorithm instead of the current per-candidate full-dataset rescan, closing the
~250–450× single-thread CPU-training slowdown measured in Spike 002 — while keeping
every shipped ≤10⁻⁵ CPU oracle bit-exact.

**In scope (user decision: ALL CPU grow policies):**
- Oblivious `SymmetricTree` host split search (the default, dominant path).
- Non-symmetric leaf-wise growers: `Depthwise` and `Lossguide`
  (`tree.rs::leaf_wise_grower` / `best_split_for_leaf`).
- The online-CTR-feature scoring path
  (`greedy_tensor_search_oblivious_with_ctr`) — CTR-projection candidates scored
  through the same histogram scorer.
- Parallelism (rayon) over features/candidates + reusable scratch buffers.

**Out of scope:**
- The GPU/device grow path (already histogram-based via `pointwise_hist.rs`) —
  untouched. Do NOT regress the device path.
- The pairwise (`TPairwiseScoreCalcer`) scoring path may keep its dedicated scorer;
  only fold it into the histogram design if it falls out cleanly — otherwise leave
  it and note the exclusion.
- Any change to model format, leaf-value calculation, or inference.
</domain>

<decisions>
## Implementation Decisions (LOCKED)

### Root cause (Spike 003 — do not re-investigate)
- The CPU path `select_level_plain`/`score_candidate` (`crates/cb-train/src/tree.rs`)
  re-scans ALL n objects and re-reduces them from scratch for EVERY `(feature,border)`
  candidate → `O(n·n_features·n_bins·depth)` per tree. No `TBucketStats` histogram,
  no subtraction trick. This is the dominant slowdown and diverges from
  `docs/CATBOOST_CORE_DESIGN.md` § "The Tree-Growing Pipeline" step 3c.

### Algorithm (PERF-01)
- Build per-feature bin histograms once per level: a single `O(n)` pass accumulates
  `(feature, bin) → {Σder1, Σweight}` for the current leaf partition; score all of a
  feature's borders from its histogram in `O(n_bins)` — the `O(n)` factor is paid
  ONCE per level, not once per candidate.
- Implement the **subtraction trick**: derive the larger child's histogram by
  subtracting the smaller sibling's from the parent's (mirrors upstream
  `TStatsForSubtractionTrick` / `PrevTreeLevelStats`).
- **Mirror the existing device reference:** `crates/cb-backend/src/kernels/pointwise_hist.rs`
  (Phase 11 `pointwise_hist2` + subtraction trick) is the algorithmic template —
  transcribe its logic onto the host.

### Parity (PERF-02) — the crux
- EVERY shipped ≤10⁻⁵ CPU oracle fixture MUST stay bit-exact after the rewrite.
- The current slowness is a DIRECT consequence of the parity-first
  `reduce_leaf_stats` gather-and-sum (`cb-compute/src/histogram.rs`, `cb_core::sum_f64`
  in canonical object order, D-05/D-08). The rewrite MUST preserve that bit-exact
  summation while dropping the rescan — accumulate bins with a deterministic ordered
  sum (fixed-point-u64, already proven in Phase 10/11, OR per-bin ordered `sum_f64`).
  Parity is the acceptance gate, not an afterthought.

### Parallelism (PERF-03) — SECOND, after the algorithm
- Fix the algorithm FIRST (removes the n_bins/n_features blow-up + most allocations);
  parallelize SECOND.
- Parallelize over features/candidates with `rayon`; keep the final per-bin reduction
  deterministic (per-feature independent, deterministic merge) so parity holds.
- Introduce reusable scratch buffers (a `TLearnContext` analogue): one incrementally
  updated `leaf_of`, fixed-size histogram arrays — eliminate the per-candidate
  allocation storm (`Vec<bool>`/obj in `assign_leaves` + nested `Vec<Vec<f64>>` in
  `reduce_leaf_stats`, ~10⁸ allocs/tree — Spike 004 confirmed this OOM-killed the
  full benchmark grid).

### Hard constraints
- **NEVER add a `cb-train` dependency on `cb-backend`** (feature-unification landmine,
  repeated across v1.0/v1.1). Transcribe the histogram logic inline into
  `cb-train`/`cb-compute`; do NOT reach across the seam to reuse the device kernel.
- CPU-only: the `cubecl`-free `cb-compute` generic boundary (D-03) stays cubecl-free.
- Kernels/`#[cube]` code are NOT touched by this phase (pure host Rust).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Root-cause evidence (start here)
- `.planning/spikes/002-perf-baseline-and-scaling/README.md` — measured slowdown + scaling signature; harness.
- `.planning/spikes/003-split-finding-hotpath-audit/README.md` — exact hot path, design-doc divergence, recommended fix.
- `.planning/spikes/004-parallelism-and-allocation-audit/README.md` — single-thread + allocation causes.

### Code under change / reference
- `crates/cb-train/src/tree.rs` — `select_level_plain`, `select_level_perturbed`, `score_candidate`, `assign_leaves`, `greedy_tensor_search_oblivious_perturbed`, `greedy_tensor_search_oblivious_with_ctr`, `leaf_wise_grower`, `best_split_for_leaf`.
- `crates/cb-compute/src/histogram.rs` — `reduce_leaf_stats`, `LeafStats`, `reduce_leaf_der2`; `cb_core::sum_f64` summation contract.
- `crates/cb-backend/src/kernels/pointwise_hist.rs` — device histogram + subtraction trick to mirror (READ, do not depend on).
- `crates/cb-train/src/boosting.rs` — grow dispatch (`~3800/3854/3914`) and the boosting loop.

### Design source of truth
- `docs/CATBOOST_CORE_DESIGN.md` § "Core CPU Training Algorithm" (lines ~858–1023) — the histogram/`TBucketStats` + subtraction-trick + parallel-over-candidates pipeline this phase must match.

### Benchmark harness (measure before/after)
- `crates/cb-train/tests/perf_baseline_test.rs` (CB_PERF-gated) + `.planning/spikes/002-perf-baseline-and-scaling/catboost_grid.py`.
</canonical_refs>

<specifics>
## Specific Ideas

- Acceptance for PERF-01: re-run the Spike-002 `n_bins` sweep after the rewrite and
  show per-tree time is ~flat (within noise) across border_count 32→254 — the
  histogram fingerprint — vs the current ~linear-in-bins blow-up.
- Acceptance for PERF-02: the existing CPU oracle test suite (all shipped ≤10⁻⁵
  fixtures across losses/policies/CTR) passes bit-exact — this is the gate.
- Acceptance for PERF-03: documented end-to-end speedup vs the pre-rewrite baseline
  on the Spike-002 grid; state the target factor vs official CatBoost 1-thread.
- Suggested sequencing: (1) oblivious histogram + subtraction + parity; (2) extend to
  Depthwise/Lossguide + CTR path; (3) rayon + scratch-buffer reuse. Planner may split
  waves accordingly.
</specifics>

<deferred>
## Deferred Ideas

- CTR-path CPU-perf on categorical-heavy real datasets (Spike 002 was numeric-only) —
  a validation nuance for the Phase 22 benchmark, not a blocker here.
- Pairwise-scorer histogram unification (only if it does not fall out cleanly).
</deferred>

---

*Phase: 21-cpu-split-finding-histogram-rewrite*
*Context gathered: 2026-07-05 from Spike 002/003/004 + user scope decisions*
