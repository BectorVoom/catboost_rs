---
title: Estimated-feature stored-border-VALUE quantization-grid parity
date: 2026-06-19
priority: medium
status: pending
origin: Phase 06.5 (deferred in 06.5-07, confirmed still-open after 06.5-08/09)
blocks: nothing (non-blocking — FEAT-01/FEAT-02/SC-1..SC-4 all closed ≤1e-5)
area: cb-train estimated-feature KNN calcer (online HNSW port) — NOT quantization/serialization
---

# Estimated-feature stored-border grid parity

> **TL;DR (2026-06-19, instrumented-trainer verdict — read the "DEFINITIVE ROOT CAUSE" section
> below):** This is NOT a quantization/serialization or boosting-loop issue. The XOR per-stage
> residual is blocked by ONE thing: upstream's KNN estimated-feature calcer uses **online HNSW**
> (approximate NN), while the Rust calcer is brute-force-EXACT (A2/D-05 deferred the HNSW crate).
> They return different neighbors on the XOR corpus → columns diverge. Closing it = **port
> `library/cpp/online_hnsw` (~936 LOC) to Rust bit-for-bit**, its own focused phase. The
> stored-border VALUE (0.5) and the boosting-loop ordering are already understood/easy.

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
fold averaging of estimated-feature leaf values.

### DEFINITIVE ROOT CAUSE (2026-06-19, fourth pass — INSTRUMENTED TRAINER, this supersedes all above)

The earlier "structure-fold cycling + averaging fold `Q(lf=3)`" recipe was a hypothesis from
offline diagnostics and is **DISPROVEN** by the instrumented catboost 1.2.10 trainer (rebuilt this
session; faithful — reproduces the XOR predictions bit-identical, `max|diff| = 0.0`). The dump
(`/tmp/xor_instr.jsonl`, events `est_call`/`est_col`/`structure_fold`/`tree_struct`/`leaf_indices`/
`leaf_partition`/`knn_neighbors`) shows:

1. **NO structure-fold cycling.** `structure_fold` logs `fold_count=1, taken_fold=0` for EVERY
   iteration. Plain + no CTRs ⇒ `permutation_needed_for_learning=false` ⇒ `learning_fold_count=1`.
   The boosting-loop rework (per-fold cycling) was the WRONG premise — reverted.
2. **The online estimated column is a SINGLE static column over `S`** (not `Q`). Both online
   `est_call`s use the IDENTITY fold permutation over the S-shuffled learn data
   (`fold_cc … "before_averaging" … "is_permuted":0` — the averaging fold is NOT permuted without
   CTRs/ordered), i.e. original-object visiting order = `S = create_shuffled_indices(n, seed)`.
   Structure + leaves + approx all use this one column; predictions apply the OFFLINE column.
   So the boosting-loop part is SIMPLE (single online-over-`S` column for training + offline
   post-hoc apply — NO `train_inner` change, NO cycling, NO folded machinery).
3. **THE REAL BLOCKER — upstream's KNN calcer is ONLINE HNSW, not exact kNN.**
   `knn.cpp:42` / `knn.h:17,109`: `TOnlineHnswDenseVectorIndex<float, TL2SqrDistance<float>>`,
   `GetNearestNeighbors<…NOnlineHnsw…>(embed, knum, /*searchNeighborhoodSize*/300, …)`,
   built with `TOnlineHnswBuildOptions({CloseNum=k, 300})`. The incremental HNSW graph returns
   **APPROXIMATE** neighbors. Concrete evidence from `knn_neighbors`: for cloud-B query doc6 over
   prefix `{14,15,0,7,4}`, the exact 3-NN are `{0,2,4}` (all cloud-B = the Rust brute-force result),
   but upstream's HNSW returns `{1,3,4}` (two cloud-A — wrong-vs-exact). My/upstream prefix
   neighbors agree for p0–p4 (as sets) and DIVERGE from p5 on. The Rust KNN is brute-force-EXACT
   (Phase 6.5 **A2/D-05: "no third-party HNSW crate"**), so it CANNOT reproduce upstream's
   approximate-HNSW neighbors → the online + offline KNN columns diverge → per-stage parity fails.
   (There is also a class-vote-order flip: upstream feat0 = class-1 vote, Rust `[class0,class1]`.)

**Conclusion: closing the XOR per-stage ≤1e-5 gate requires porting upstream's online HNSW
(`library/cpp/online_hnsw`, ~936 LOC: dynamic dense graph + incremental insert + HNSW search,
`TL2SqrDistance`, RNG-driven graph construction) to Rust bit-for-bit.** This is a deliberately
deferred dependency (A2/D-05) and is realistically its OWN focused phase, NOT a continuation of
this todo. It is the dominant blocker; the boosting-loop / column-ordering parts are easy once the
KNN columns match upstream.

**Instrumented trainer (rebuild recipe for the HNSW work):** clang-18+lld-18 via `apt-get download`
+ `dpkg -x` to `/tmp/clang18_prefix`; conan/ninja/cython via `uv` (persist in `~/.local/bin`);
`build/build_native.py --targets _catboost --cmake-{target,build}-toolchain build/toolchains/clang.toolchain`
→ `lib_catboost.so`; copy to `/tmp/cb_instr_pkg/catboost/_catboost.so`, run with
`PYTHONPATH=/tmp/cb_instr_pkg CB_INSTRUMENT_LOG=… LD_LIBRARY_PATH=/tmp/clang18_prefix/usr/lib/...`.
`estimated_features.cpp` carries an added `est_col`/`est_call` dump (full per-object estimated
columns + per-call online/offline + fold perm); `/tmp/run_xor_instr.py` is the XOR fit script.

The honest residual test (`xor_oracle_per_stage_residual_…`) stays RED-on-success until the HNSW
port lands. The `text_embedding_xor/` fixture is frozen and ready; no regeneration needed.

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
