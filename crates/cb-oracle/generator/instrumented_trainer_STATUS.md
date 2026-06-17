# Instrumented catboost 1.2.10 trainer — build STATUS (Plan 06.3-10)

## Outcome: **GO** ✅

The instrumented `_catboost.so` trainer (with env-gated `CB_INSTRUMENT_LOG`
per-leaf SumDer/SumDer2 + RNG-draw events) **LINKED, runs, and produces a
non-empty instrumentation log**. This GO **unblocks** downstream plans 06.3-13
(PairLogit per-leaf-der2 oracle) and 06.3-14 (YetiRank/StochasticRank
end-to-end per-stage trainer fixtures).

Recorded per the escalate-don't-weaken contract (D-6.3-03b): a definitive GO,
not a weakened tolerance or a fabricated fixture.

## Artifact paths (OFFLINE, RUN-ONCE; NOT committed — D-09 / D-12)

| Artifact | Path |
|----------|------|
| Built instrumented trainer (raw) | `/tmp/cb_build313/catboost/python-package/catboost/lib_catboost.so` |
| Staged venv-package copy (`_catboost.so`) | `/tmp/cb_build313/instr_pkg/catboost/_catboost.so` (39.7 MB) |
| clang-18 toolchain prefix | `/tmp/clang18_prefix` (`/usr/bin/clang-18`, `clang++-18`, `lld-18`) |
| CMake/Ninja build root | `/tmp/cb_build313` |
| Full build log | `/tmp/instr_build.log` |
| Smoke-run instrumentation log | `/tmp/instr_smoke.jsonl` (58 KB) |

These live under `/tmp` and are deliberately uncommitted. Re-create them by
re-running `crates/cb-oracle/generator/build_instrumented_trainer.sh`
(idempotent; reuses an existing `/tmp/clang18_prefix` + `/tmp/cb_build313`).

## Build environment (the precondition that flipped vs prior deferrals)

- **Disk:** 67 GB free / 72% used on `/` — well above the 25 GB Release-C++-link
  safety floor. This is the NEW FACT vs 06.3-03/04/09, which deferred under the
  ~8-12 GB / 95-97% link-failure regime.
- **Toolchain:** clang-18 / lld-18 / llvm-18 (Ubuntu noble `1:18.1.3-1ubuntu1`)
  fetched sudo-free via `apt-get download` (132 MB) + `dpkg -x` into
  `/tmp/clang18_prefix`; conan / ninja / cython already on PATH (`~/.local/bin`).
- **Python:** project `.venv` Python 3.13.13 (`-DPython3_INCLUDE_DIR` /
  `-DPython3_EXECUTABLE` overrides — FindPython otherwise picks system 3.12).

## Instrumentation surfaces compiled into the artifact (CB_INSTRUMENT_LOG-gated)

All four hooks are env-gated (strict no-op when `CB_INSTRUMENT_LOG` is unset),
verified present in the compiled sources:

| Hook | File | Event |
|------|------|-------|
| per-leaf Der1/Der2 (grouped/pairwise reduction) | `approx_calcer_querywise.cpp` (`AddLeafDersForQueries`) | `leaf_der` |
| per-leaf SumWeights (pointwise leaf path) | `approx_calcer.cpp` (`CalcLeafValues`) | `leaf_weight` |
| YetiRank Gumbel RNG draw | `yetirank_helpers.cpp` (`GenerateYetiRankPairsForQuery`) | `yeti_gumbel` |
| StochasticRank Gaussian noise | `algo_helpers/error_functions.cpp` | `srank_noise` |

## Smoke verification (CB_INSTRUMENT_LOG run)

A tiny `CatBoostRanker(loss_function="YetiRank", iterations=3, depth=2)` `.fit()`
on a 12-doc / 2-group corpus with the staged `_catboost.so` and
`CB_INSTRUMENT_LOG=/tmp/instr_smoke.jsonl` produced a 58 KB JSONL with:

```
    264 "event":"leaf_der"      <- 06.3-10 per-leaf SumDer/SumDer2 hook
   1080 "event":"yeti_gumbel"   <- 06.3-10 YetiRank RNG-draw hook
      3 "event":"structure_fold"
      3 "event":"tree_struct"
      3 "event":"leaf_partition"
      6 "event":"avg_ctr_bins"
      4 "event":"fold_cc"
      3 "event":"leaf_indices"
```

The two 06.3-10 surfaces (`leaf_der`, `yeti_gumbel`) fire as designed; the rest
are prior-session live-trainer events that survived in the vendored source.
`fit OK; trees: 3`.

## RUN-ONCE invocation recipe (for plans 13 / 14)

```sh
# 1. (re)build if /tmp artifacts are gone — idempotent, sudo-free:
bash crates/cb-oracle/generator/build_instrumented_trainer.sh

# 2. train a corpus config with the instrumented trainer, capturing the log:
CB_INSTRUMENT_LOG=/tmp/<config>.jsonl \
  PYTHONPATH=/tmp/cb_build313/instr_pkg \
  .venv/bin/python <fit_script.py>

# 3. parse the per-leaf SumDer2 (plan 13) / RNG-draw + per-stage model.json
#    (plan 14) ground truth from the JSONL, freeze as committed fixtures, and
#    drop the corresponding #[ignore] / absent-fixture guards. NO tolerance
#    weakening, NO fabrication.
```

## NO-GO branch (not taken)

Had the build failed at link / disk / toolchain, this file would record **NO-GO**
with the precise failing step, and plans 13/14 would keep the PairLogit oracles
`#[ignore]`'d and the YetiRank/StochasticRank `model.json` absent-fixture
invariants in force — with NO weakened tolerance and NO fabricated fixtures.
That branch did not occur; the build is GO.
