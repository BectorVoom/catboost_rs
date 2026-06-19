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
