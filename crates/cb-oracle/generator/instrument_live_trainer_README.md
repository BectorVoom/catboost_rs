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
