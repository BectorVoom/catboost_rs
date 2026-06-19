---
title: Estimated-feature stored-border-VALUE quantization-grid parity
date: 2026-06-19
priority: medium
status: pending
origin: Phase 06.5 (deferred in 06.5-07, confirmed still-open after 06.5-08/09)
blocks: nothing (non-blocking — FEAT-01/FEAT-02/SC-1..SC-4 all closed ≤1e-5)
area: cb-train estimated-feature quantization / serialization
---

# Estimated-feature stored-border grid parity

## Progress — quick task 260619-cpr (2026-06-19)

**Partially closed; narrowed residual remains (this todo stays open).** See
`.planning/quick/260619-cpr-estimated-feature-stored-border-value-qu/260619-cpr-SUMMARY.md`.

- **Root cause found:** the `0.5` vs `1.5` divergence is a column-VALUE divergence, NOT a
  border-algorithm gap. Upstream KNN is an `IOnlineFeatureEstimator` fed the online
  read-before-update estimate → vote distribution `{0,1,…,k}` (first border `0.5`); Rust
  used the offline whole-set estimate → `{0,k}` (border `k/2 = 1.5`). `select_borders_greedy_logsum`
  is upstream-exact and was left unchanged.
- **Fixed:** KNN block now routes through `online_knn_prefix` via `embedding_online`; the
  unchanged quantizer stores `0.5`. KNN stored-border hard gate passes exactly; XOR fixture
  added with both estimated features load-bearing.
- **Residual (gate NOT relaxed — no `#[ignore]`, no weakened tolerance):** the XOR per-stage
  in-order parity is OPEN. The KNN stored-border VALUE (0.5) is closed; per-stage is not.

### CORRECTION (2026-06-19, second pass — the "thread the permutation" fix is DISPROVEN)

The earlier note guessed the residual was just "Rust uses the identity permutation; thread
the averaging-fold learn permutation." **An empirical search disproved that.** Building the
pre-baked online KNN column over every plausible learn permutation —
`identity`, `S = create_shuffled_indices(n,seed)`, `Q = averaging_ctr_permutation(n,lf,seed)`
for `lf∈{1,2,3}`, `permutations(n,4,seed)[0..3]`, the `S∘perm` compositions, and all their
inverses — and training, the BEST max|pred−upstream| was **~0.32** (need ≤1e-5). Re-applying
the online-trained trees to the OFFLINE columns (next hypothesis) also floored at ~0.32. So a
column/permutation swap **cannot** close per-stage parity.

**The true (deeper) root cause — three coupled gaps the degenerate SC-4 corpus masked:**
1. **Train-vs-apply feature-source split.** Upstream builds tree STRUCTURE + LEAF VALUES on the
   per-fold **ONLINE** estimated features (leakage-controlled), but the fixture's `staged.npy`
   /`predictions.npy` are `model.staged_predict`/`model.predict` on the pool — i.e. the trees
   **re-applied to the OFFLINE (application) estimated features** (`data.cpp:537`
   `EstimatedObjectsData`, `learnPermutation = Nothing()`). Rust's `train()` uses ONE pre-baked
   column for BOTH, so neither online nor offline alone (nor online-train+offline-apply) matches.
2. **Multi-permutation fold averaging.** `permutation_count = 4` → upstream AVERAGES the
   estimated-feature-driven leaf values over multiple permutation folds. The recurring IDENTICAL
   `0.324` across many distinct single permutations shows the single-fold Rust model sits a fixed
   structural distance from upstream regardless of which one permutation is chosen — the gap is the
   averaging, not the choice.
3. **Online features are per-iteration dynamic.** A single static pre-baked column cannot capture
   the ordered/averaging-fold dynamics upstream evolves across boosting iterations.

**Therefore closing per-stage parity is a CORE BOOSTING-LOOP change, not a `build_mixed` tweak:**
thread SEPARATE online (per-fold, structure+leaves) and offline (application, predictions)
estimated-feature column sets through `train()`/predict, AND reproduce the multi-permutation
fold averaging of estimated-feature leaf values. This is multi-day architectural work touching
the core training loop with regression risk to the entire oracle suite; the instrumented trainer
should confirm the exact per-fold online columns + averaging weights before implementation.
The honest residual test (`xor_oracle_per_stage_residual_…`) stays RED-on-success and is the gate.

---

The exact border **values** upstream catboost 1.2.10 stores for *estimated* (text /
embedding) feature columns do not bit-match the Rust-selected grid, even though
trained-model **predictions still match ≤1e-5**. Phase 06.5 closed the calcer math
(FEAT-01/02) and the SC-4 join, then explicitly deferred this grid question. After
06.5-08/09 proved the "BM25 ±1.24 border" was a *fixture mislabel* (not a real
normalization), this generalized grid concern is what remains, and it is **unowned**.

## Symptoms / evidence (from Phase 06.5)

- **KNN integer-vote border:** upstream stores `0.5` for the class-vote split; the
  Rust `select_borders_greedy_logsum` on the `{0, k}` vote distribution returns the
  midpoint (e.g. `1.5`). Both induce the SAME 8/8 partition, so predictions agree —
  only the stored border VALUE differs.
- **BoW digitization grid:** a deliberately non-degenerate XOR-structured mixed
  corpus (prototyped in 06.5-07, then REJECTED as the SC-4 fixture) forces exact KNN
  vote-count + BoW digitization-grid parity; under it the staged-approx / predictions
  did **not** match ≤1e-5 — confirming the grid is a distinct, still-open concern,
  not just a cosmetic stored-value difference.

## Why deferred / non-blocking

- This is a **trainer estimated-feature quantization/serialization** question (how
  upstream selects the stored border grid for estimated columns), NOT a calcer-math
  or SC-4-join defect.
- The 06.5-07 SC-4 oracle deliberately isolates the JOIN (closed ≤1e-5) from the GRID
  (open) by using a degenerate-separating corpus + structure-invariant Splits/
  LeafValues gating (per-tree leaf MULTISET ≤1e-5; magnitudes exact, only the
  ambiguous leaf ORDER freed).
- FEAT-01 (BoW/NaiveBayes/BM25 per-stage ≤1e-5) and FEAT-02 (LDA documented-tolerance
  + KNN bit-exact neighbor ids; SC-4 KNN end-to-end ≤1e-5) are CLOSED and do not
  depend on this.

## Scope when picked up

1. Reproduce upstream's estimated-feature border-grid selection: which algorithm
   catboost uses for estimated columns (vs the numeric `select_borders_greedy_logsum`
   path) — e.g. the integer-vote `0.5` border and the BoW digitization grid.
2. Wire a Rust estimated-feature grid path that reproduces the stored border VALUES
   bit-for-bit (not just the partition), so a serialized model's borders match.
3. Re-introduce the rejected non-degenerate XOR mixed corpus as the HARD oracle:
   StagedApprox + Predictions ≤1e-5 with exact stored borders (no structure-invariant
   leaf-order relaxation). This is the gate that 06.5-07 could not pass.

## Pointers

- `.planning/phases/06.5-text-and-embedding-features/deferred-items.md`
  ("General estimated-feature quantization-GRID parity (06.5-07)") — fullest writeup.
- `.planning/phases/06.5-text-and-embedding-features/06.5-07-SUMMARY.md` — the SC-4
  join closure + the rejected XOR-corpus prototype.
- Upstream chain (scale-preserving for BM25, per 06.5-08 dump):
  `base_text_feature_estimator.h:74-88` → `estimated_features.cpp:204-250` →
  `split.cpp:45-46` → `model.cpp:209`. The grid-selection divergence is in how the
  estimated column's borders are chosen/stored, not in the calcer scores.
- Rust seam: `cb-train/src/estimated/estimated_features.rs`, and the numeric border
  selector `select_borders_greedy_logsum` (the wrong tool for the integer-vote case).

## Done when

- A non-degenerate (XOR-style) text+embedding+numeric corpus trains a model whose
  serialized estimated-feature borders match upstream bit-for-bit, AND StagedApprox +
  Predictions match ≤1e-5 with NO structure-invariant leaf-order relaxation.
- KNN vote border serializes as `0.5` (not `1.5`); BoW digitization grid matches.
- No `#[ignore]`, no weakened tolerance.
