# Live-trainer instrumentation — pc=4 structure-fold cycling ground truth (Plan 05-17, bar (c))

**Authorization.** This is a DELIBERATE, USER-APPROVED deviation from the
D-11 / Phase-1 D-08 "Python-reachable floor, no C++ instrumentation" rule. The
authorization is the **2026-06-15 gap-closure DECISION REVISION** in
`.planning/phases/05-.../05-CONTEXT.md`, plus the explicit user choice on
2026-06-15 to "Attempt toolchain provision + build" to close closure bar (c).
The instrumentation is scoped to THIS pc=4 boosting gap only.

**RUN-ONCE / COMMIT, OFFLINE ONLY, NEVER in CI (D-09 / D-12).** The instrumented
trainer is built and run by hand; CI reads only the committed
`crates/cb-train/tests/fixtures/multi_permutation_fold/live_trainer_structure_fold.json`,
never the trainer.

## Why a live build was required

The pc=4 odd-tree partition `[8,8,0,14]` is a property of the LIVE training-time
`ComputeOnlineCTRs(AveragingFold)` float CTR values feeding `CalcLeafValues`
(`approx_calcer.cpp:1082`), recomputed per iteration, combined with the
per-iteration STRUCTURE-fold selection `Folds[Rand.GenRand() % foldCount]`
(`train.cpp:208`). The exhaustive 05-15 offline sweep proved it is NOT
recoverable from the (correct) AveragingFold permutation + offline
online-prefix reconstruction (best offline max-diff 0.176). The live trainer is
the only ground truth.

## Toolchain provisioned (sudo-free)

- **Conan 2.29.1**, **Ninja 1.13.0**, **Cython 3.2.5 (+numpy)** via `uv tool install`.
- **clang-18 / lld-18** (Ubuntu 24.04 noble): catboost 1.2.10's vendored libc++
  (`contrib/libs/cxxsupp/libcxx`) needs clang>=16 builtins (`__remove_cv`,
  `__bf16`); clang-14 FAILS to compile it. The clang-18 + lld-18 debs were
  `apt-get download`ed (no sudo) and `dpkg -x`-extracted into `/tmp/clang18_prefix`.
- Build: `catboost-master/build/build_native.py --targets _catboost` against the
  project `.venv` Python 3.13 (`-DPython3_INCLUDE_DIR/.../_EXECUTABLE` overrides;
  FindPython otherwise defaults to system 3.12 → ABI mismatch with the 3.13 venv).
- Conan fetched openssl/ragel/swig/yasm from conancenter (network available).

## The instrumentation patch (catboost/private/libs/algo/train.cpp)

Env-gated by `CB_INSTRUMENT_LOG` (fully inert when unset). Per boosting
iteration `TrainOneIteration` logs three JSONL events:

1. `structure_fold`: `Rand.GetCallCount()` BEFORE the structure draw, the raw
   `Rand.GenRand()`, and the selected `taken_fold = draw % foldCount`
   (the `train.cpp:208` structure-fold selection, captured WITHOUT double-drawing).
2. `tree_struct`: the chosen oblivious tree's per-split `BinBorder`.
3. `leaf_partition`: the AveragingFold `sumLeafWeights` (the leaf partition) and
   the pre-normalize `treeValues` (leaf deltas) from `CalcLeafValues`
   (`approx_calcer.cpp:1082`, fixed AveragingFold).

Sink helper:

```cpp
static void CbInstrumentLog(const TString& line) {
    static std::mutex mtx;
    const char* path = std::getenv("CB_INSTRUMENT_LOG");
    if (path == nullptr) { return; }
    std::lock_guard<std::mutex> g(mtx);
    FILE* f = std::fopen(path, "a");
    if (f != nullptr) { std::fputs(line.c_str(), f); std::fputc('\n', f); std::fclose(f); }
}
```

## RUN-ONCE invocation

```sh
# Instrumented _catboost.so dropped into a copy of the venv catboost package:
cp /tmp/cb_build313/catboost/python-package/catboost/lib_catboost.so \
   <pkg>/catboost/_catboost.so
CB_INSTRUMENT_LOG=/tmp/instr_pc4.jsonl PC=4 \
  PYTHONPATH=<pkg> .venv/bin/python run_pc4_instr.py   # CatBoost(...).fit(Pool(X,y,cat_features=[0,1]))
```

The instrumented trainer reproduced upstream tree leaf_weights
`[6,0,10,14],[8,8,0,14],[6,0,10,14],[8,8,0,14],[8,8,0,14]` and predictions
BIT-IDENTICAL (max abs diff 0.0) to the committed `predictions_pc4.npy`.

## Discovered rule (ported into cb_train::boosting.rs)

Per boosting iteration the STRUCTURE fold is `Folds[Rand.GenRand() % foldCount]`
with `foldCount == LearningFoldCount`; leaf VALUES always use the FIXED
AveragingFold. Observed:

| pc | foldCount | per-iter taken_fold | per-tree partition |
|----|-----------|---------------------|--------------------|
| 1  | 1         | 0,0,0,0,0           | all [6,0,7,17]     |
| 2  | 1         | 0,0,0,0,0           | all [6,0,7,17]     |
| 4  | 3         | 0,2,0,2,2           | [6,0,10,14],[8,8,0,14],[6,0,10,14],[8,8,0,14],[8,8,0,14] |

At `foldCount==1` the modulo is always 0, so the structure fold is always
`Folds[0]` and the rule reduces to the prior fixed-fold behavior — pc=1/pc=2
stay byte-identical (no regression). Only at pc>=4 does the per-iteration
cycling produce per-tree structure variation. The committed ground truth is
`live_trainer_structure_fold.json`.

## Second instrumentation cycle (Plan 05-18, 2026-06-15) — Spike-001 resolution

**Authorization.** RE-AUTHORIZED by the user 2026-06-15 for a second cycle to
derive a SELF-CONSISTENT oracle, after Spike 001 proved the first-cycle
`live_trainer_ctr_bins_blocker.json` was internally inconsistent.

**Two added env-gated (`CB_INSTRUMENT_LOG`) events:**

1. `train.cpp` `self_consistent_ctr` (helper `CbLogSelfConsistentCtr`): for the
   AveragingFold AND the per-iteration structure (taken) fold, dumps the
   projection cat-features, `LearnPermutationFeaturesSubset` (via `ForEach`),
   `GetLearnPermutationArray()`, `LearnTargetClass`, and the `GetData` ui8 bins —
   all from the SAME fold object.
2. `online_ctr.cpp` `online_ctr_inputs`: right before `CalcOnlineCTRSimple`, dumps
   the LITERAL inputs the algorithm consumes — `perm_subset`
   (`foldLearnPermutationFeaturesSubset`), the reindexed `enumerated` cat features
   (`hashArr`), and every classifier's `target_classes` (`foldLearnTargetClass`).

**What it revealed (the Spike-001 inconsistency, explained).** The averaging
online-CTR ui8 bins (`GetData`/`Feature[docIdx]`) are stored in the CTR
materialization order **Q**, where `Q = S ∘ LearnPermutation` and `S` is the
catboost quantized data-provider's internal object STORAGE reorder (verified: the
identity learning fold's `perm_subset == [0..29]` yields a NON-natural object
cat-sequence). The first cycle paired those bins with
`GetLearnPermutationArray()` = `[11,18,15,29,…]` — the leaf-index iteration order,
a DIFFERENT order — and never logged `LearnTargetClass`. Hence inconsistent.

With the atomic capture: the bins ARE the single-cat-0 Borders online prefix
under order **Q** with target `LearnTargetClass[1]` (= binarized y;
`LearnTargetClass[0]` is all-zeros / unused). Q reproduces all five tree
partitions `[6,0,10,14],[8,8,0,14],[6,0,10,14],[8,8,0,14],[8,8,0,14]` bit-exact.
The self-consistent ground truth is `live_trainer_self_consistent.json`;
`live_trainer_ctr_bins_blocker.json` is annotated `superseded_by` it.

**Why bar-(c) is DEFERRED (the precise remaining blocker).** pc=4 tree-B borders
`[3,7]` SPLIT the mixed cat buckets (cat3, cat4), so the leaf VALUES depend on the
exact per-mixed-bucket bin→object assignment — which is fixed by `S`. cb-train
materializes CTRs on object-order `X_cat` WITHOUT `S`; the must_haves'
`[11,18,15,…]` reproduces the pc=4 PARTITION `[8,8,0,14]` but the wrong leaf
VALUES (leaf1 `sum_y=6` vs upstream `5`, ≫1e-5). Closing bar-(c) requires porting
`S` (the data-provider quantized-object storage order) into cb-train — a
research-grade subsystem out of this plan's scope. pc=1 stays green because its
borders do not split the mixed buckets (leaf composition is order-invariant
there). Per the authorized FALLBACK: cb-train production is UNTOUCHED, the pc=4
e2e oracle is UNCOMMITTED, no oracle weakened.

## 06.3-10 re-run note (2026-06-17) — instrumented trainer GO

The full instrumented `_catboost` trainer build was **re-attempted and SUCCEEDED**
this session (GO; see `instrumented_trainer_STATUS.md`). The disk NO-GO
precondition that forced the 06.3-03/04/09 deferrals no longer holds: `/` is now
**67 GB free / 72% used**, well above the 25 GB Release-C++-link safety floor
(the README documents linking failed only at ~8-12 GB / 95-97%).

The `/tmp` toolchain (`/tmp/clang18_prefix`, `/tmp/cb_build313`) from prior
sessions was **absent** and was re-fetched from scratch by the new re-runnable
driver `build_instrumented_trainer.sh`:

- clang-18 / lld-18 / llvm-18 (noble `1:18.1.3-1ubuntu1`) via sudo-free
  `apt-get download` + `dpkg -x` into `/tmp/clang18_prefix`;
- conan / ninja / cython reused from `~/.local/bin` (uv tool);
- `build_native.py --targets _catboost` against the `.venv` Python 3.13 with
  `-DPython3_INCLUDE_DIR` / `-DPython3_EXECUTABLE` overrides.

**Build-config fix discovered (recorded for re-runs):** catboost's
`build/toolchains/clang.toolchain` hardcodes bare `clang` / `clang++` and
re-exports `ENV{CC}`/`ENV{CXX}`, overriding the `-DCMAKE_*_COMPILER` cache
entries. The driver therefore creates `clang` → `clang-18` /
`clang++` → `clang++-18` (and a `clang-14` alias for the CUDA-host probe)
symlinks inside the prefix bin so the bare names resolve on PATH.

The 06.3-10 instrumentation adds two NEW env-gated surfaces beyond the pc=4
work above: per-leaf `Der1`/`Der2` in `approx_calcer_querywise.cpp`
(`leaf_der` event, PairLogit plan 13) and the YetiRank Gumbel RNG draw in
`yetirank_helpers.cpp` (`yeti_gumbel` event, plan 14) — both verified firing in
the smoke run. Artifact paths are in `instrumented_trainer_STATUS.md`.
