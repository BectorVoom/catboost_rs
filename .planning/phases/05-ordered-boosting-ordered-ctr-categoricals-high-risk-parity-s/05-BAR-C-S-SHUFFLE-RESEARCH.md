# Bar (c) closure research — the storage reorder `S` IS the initial learn-set shuffle (TRACTABLE)

**Status:** feasibility CONFIRMED 2026-06-15 (plan 05-18 investigation, live instrumented
catboost 1.2.10). This is the gap-source for the next bar-(c) plan.

## What `S` is (resolved)

`S` is NOT deep data-provider internals. It is catboost's **initial learn-set shuffle**,
applied once at training start, BEFORE fold creation:

- `preprocess.cpp:183 ShuffleLearnDataIfNeeded` → `NCB::Shuffle(learnData->ObjectsGrouping, /*permuteBlockSize*/ 1, rand)`.
- `NeedShuffle` (`preprocess.cpp:161`) returns **true whenever `catFeatureCount > 0`** (and
  not `HasTimeFlag`). Our configs have cat features → always shuffled.
- For trivial grouping (no groups) + `permuteBlockSize == 1`, `NCB::Shuffle`
  (`objects_grouping.cpp`) → `CreateShuffledIndices(objectCount, rand, &indices)` = a plain
  Fisher-Yates over the `n` objects, driven by the shared `TRestorableFastRng64` seeded by
  `random_seed`.

Therefore the averaging-fold CTR order `Q = S ∘ LearnPermutation`: the learn data is first
permuted by `S`, then fold permutations (LearnPermutation / AveragingFold) are drawn
RELATIVE to the shuffled data. cb-train currently skips `S` and operates on object-order
`X_cat`, so its fold permutations are relative to object order → wrong per-mixed-bucket
interleaving → wrong pc=4 leaf values (bar (c)).

Verified `S` for the `tensor_ctr_e2e` fixture (n=30, seed=0, captured unambiguously via
both cat columns + target from the identity learning fold `perm_subset == [0..29]`):
`S = [5,2,6,12,9,28,1,17,29,11,7,3,18,10,4,14,8,19,23,13,27,20,15,26,21,24,16,0,22,25]`.
(Note `S != fisher_yates_permutation(30,0) = [8,12,5,18,…]`, so either `CreateShuffledIndices`
uses a different shuffle direction than `shuffle_in_place`, or the shared RNG has consumed
some pre-shuffle draws — the plan's first task must byte-match `CreateShuffledIndices` +
the exact RNG state at the shuffle.)

## The port (next plan scope)

1. **Add the initial learn-set shuffle to `train_cat`** (gated like `NeedShuffle`:
   `catFeatureCount > 0 && !has_time`): draw `S` from the persistent `random_seed` RNG via a
   faithful `CreateShuffledIndices` port, BEFORE `create_folds`. Apply `S` to the learn
   data (X, y, and any per-object structures) so all downstream CTR materialization, fold
   permutations, and leaf indexing operate on `S`-shuffled data.
2. **Thread the RNG accounting**: `S` consumes the first draws; fold creation continues on
   the same stream. This likely SUBSUMES / corrects the current `create_folds` cc=29/87
   pre-draw hack (the must_haves' "learning_folds full FY passes" is really the `S` shuffle
   consuming the first draws) — re-derive create_folds against `S`-then-folds.
3. **Map predictions back to original object order** (`S` is internal; final RawFormulaVal
   is per-original-object — `GetSubset` then inverse at output).

## HARD no-regression constraints (carry from 05-18)

- pc=1 / `tensor_ctr_e2e_oracle_test` MUST stay green ≤1e-5 (it already exercises `S` for
  pc=1 implicitly — its borders just don't split mixed buckets; the explicit `S` port must
  keep it green).
- The numeric / one-hot / Plain-no-CTR paths must NOT shuffle (NeedShuffle is cat-only here;
  but ordered boosting with no cat features DOES shuffle — match `NeedShuffle` exactly).
- No oracle weakened; bar-(c) closure = `multi_permutation_e2e_oracle_test` pc=4 ≤1e-5
  (currently uncommitted) PLUS all existing green oracles.

## Source refs

- `catboost/private/libs/algo/preprocess.cpp:161 NeedShuffle`, `:183 ShuffleLearnDataIfNeeded`.
- `catboost/libs/data/objects_grouping.cpp NCB::Shuffle`, `CreateShuffledIndices`.
- `catboost/libs/data/order.cpp` (EObjectsOrder::RandomShuffled).
- cb-train primitive already present: `permutation.rs::shuffle_in_place` / `fisher_yates_permutation`.
- Self-consistent oracle: `live_trainer_self_consistent.json`; instrumentation:
  `crates/cb-oracle/generator/instrument_live_trainer_README.md`.
