---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 17
subsystem: cb-train (multi-permutation AveragingFold + cat-CTR boosting)
tags: [ORD-01, SC-1, parity, ctr, fold, rng, instrumentation, live-trainer]
requires:
  - "catboost 1.2.10 in .venv (offline oracle)"
  - "Conan + Ninja + clang-18/lld-18 + Python 3.13 (build toolchain, provisioned)"
provides:
  - "Bars (a),(b),(d),(e): pc=4 AveragingFold partition [6,0,10,14] integer-exact, hard-equality oracle, pc=1/pc=2 locks, fold-creation ground truth"
  - "Live-trainer instrumented ground truth: structure-fold cycling + averaging permutation + averaging online-CTR bins"
affects:
  - "Bar (c) pc=4 e2e prediction oracle (DEFERRED — root blocker re-localized to online-CTR bins)"
tech-stack:
  added:
    - "uv tool: conan 2.29.1, ninja 1.13.0, cython 3.2.5(+numpy)"
    - "clang-18 + lld-18 (apt-get download + dpkg -x, sudo-free) for catboost vendored libc++"
  patterns:
    - "Live C++ trainer instrumentation (env-gated CB_INSTRUMENT_LOG), RUN-ONCE/COMMIT, never CI"
    - "FindPython override to project .venv Python 3.13 (avoid system 3.12 ABI mismatch)"
key-files:
  created:
    - crates/cb-oracle/generator/instrument_live_trainer_README.md
    - crates/cb-train/tests/fixtures/multi_permutation_fold/live_trainer_structure_fold.json
    - crates/cb-train/tests/fixtures/multi_permutation_fold/live_trainer_ctr_bins_blocker.json
  modified:
    - .planning/phases/05-.../deferred-items.md
  untouched_by_fallback:
    - crates/cb-train/src/fold.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs (uncommitted, anchor)
decisions:
  - "Provisioned the build toolchain sudo-free and built an instrumented catboost 1.2.10 trainer (the user-chosen 'Attempt toolchain provision + build' path)"
  - "Recovered the structure-fold cycling rule AND found the create_folds averaging permutation was wrong"
  - "Re-localized bar (c) to an online-CTR-bin parity gap; took the authorized FALLBACK (defer c, leave production untouched) because the fix is a CTR-subsystem change with blast radius across all CTR locks"
metrics:
  duration: "~1 session"
  completed: 2026-06-15
---

# Phase 5 Plan 17: pc=4 AveragingFold Parity — Live-Trainer Instrumentation Summary

Provisioned a sudo-free catboost 1.2.10 build toolchain and built an
instrumented trainer to close closure bar (c); the live ground truth recovered
the per-iteration structure-fold cycling rule and proved the `create_folds`
averaging permutation was wrong, then re-localized the true remaining blocker to
the AveragingFold online-CTR bin computation — an architectural CTR-parity gap
beyond porting the structure-fold rule, so bar (c) was cleanly deferred under the
authorized FALLBACK with bars (a),(b),(d),(e) green and no production regression.

## What this plan set out to do

Close bar (c): a pc=4 end-to-end train→predict oracle proving final predictions
match upstream catboost 1.2.10 ≤1e-5. Prior agents had bars (a),(b),(d),(e) green
but escalated (c) as blocked on a live-trainer build (Conan/Ninja missing). The
user chose **"Attempt toolchain provision + build."**

## What was done

### 1. Build toolchain provisioned (sudo-free)
- **Conan 2.29.1, Ninja 1.13.0, Cython 3.2.5(+numpy)** via `uv tool install`.
- **clang-18 + lld-18**: catboost 1.2.10's vendored libc++
  (`contrib/libs/cxxsupp/libcxx`) needs clang>=16 builtins (`__remove_cv`,
  `__bf16`); clang-14 FAILS to compile it. clang-18/lld-18 debs were
  `apt-get download`ed (no sudo) and `dpkg -x`-extracted into a local prefix.
- Network (conancenter) available; openssl/ragel/swig/yasm fetched OK.

### 2. Instrumented trainer built and run
- Patched `train.cpp` + `learn_context.cpp` with env-gated (`CB_INSTRUMENT_LOG`)
  logging of: per-iteration RNG `GetCallCount()`, the selected structure fold
  (`train.cpp:208`), the AveragingFold leaf partition + leaf values
  (`approx_calcer.cpp:1082`), per-object `BuildIndices(AveragingFold)` leaf
  assignment, the averaging online-CTR ui8 bins, and the fold-creation call-counts.
- Built `_catboost` against the project `.venv` Python 3.13 (FindPython override —
  the default picks system 3.12, an ABI mismatch). The instrumented trainer
  reproduced upstream tree leaf_weights `[A,B,A,B,B]` and predictions
  **bit-identical (max abs diff 0.0)** to the committed `predictions_pc4.npy` —
  proving the instrumentation is faithful. Disk stayed >47G free throughout.

### 3. Ground truth recovered (committed)
- **Structure-fold cycling** (`live_trainer_structure_fold.json`): per iteration
  `takenFold = Folds[Rand.GenRand() % foldCount]`; pc=4 sequence `[0,2,0,2,2]`;
  leaf values stay on the FIXED AveragingFold. Ported into `boosting.rs` during the
  session and verified it produces tree0 `[6,0,10,14]`.
- **The `create_folds` averaging permutation was WRONG.** The `fold_cc`
  `GetCallCount()` log proves the averaging Fisher-Yates shuffle starts at
  call-count **29 (pc=1) / 87 (pc=4)** = `learning_folds` FULL Fisher-Yates passes,
  NOT the committed "callcount == learning_folds" (=1/=3) rule. Our `uniform`
  shuffle advanced to cc=87 reproduces upstream `[11,18,15,29,16,...]` BIT-EXACT
  (cc=29 → `[10,17,25,3,6,...]` for pc=1). The OLD permutation `[23,19,25,...]`
  only coincidentally yielded the partition COUNTS `[6,0,10,14]`/`[6,0,7,17]`.

### 4. Root blocker re-localized (`live_trainer_ctr_bins_blocker.json`)
Even WITH the bit-exact averaging permutation, our online-CTR ui8 bins differ from
upstream's `ComputeOnlineCTRs(AveragingFold)`. Upstream avg-order bins
`[7,7,7,11,3,7,2,11,7,12,9,...]` are not reproduced by `materialize_ctr_feature`
(single-cat Borders, prior 0.5, border_count 15): the first 4 match, then diverge
at position 4. Correcting `create_folds` to the true permutation REGRESSES the
pc=1/pc=2 `[6,0,7,17]`→`[6,0,8,16]` partition locks and the pc=1 e2e — because
those locks are pinned to the *compensating* wrong-permutation + wrong-CTR-bins
combination.

## Why bar (c) is deferred (the authorized FALLBACK)

Closing (c) now requires a `cb_train` online-CTR materialization fix to match
`ComputeOnlineCTRs(AveragingFold)` bit-exact — a CTR-SUBSYSTEM change that ripples
through every committed CTR oracle (the pc=1/pc=2/pc=4 partition locks,
tensor_ctr_e2e, the leaf-value path), since those locks currently pass on the
compensating-error combination. This is beyond "port the recovered structure-fold
rule" and is the objective's explicit STOP-and-checkpoint condition. Per the
FALLBACK: production code left UNTOUCHED, the pc=4 e2e oracle left UNCOMMITTED,
and the live-trainer ground truth committed so a follow-up CTR-parity plan can
transcribe the exact `ComputeOnlineCTRs(AveragingFold)` bins directly.

## Bars status (no regression)

| Bar | Status | Evidence |
|-----|--------|----------|
| (a) pc=4 partition `[6,0,10,14]` | GREEN | `multi_permutation_fold_oracle_test` 4/4 |
| (b) hard-equality oracle | GREEN | (same) |
| (c) pc=4 e2e ≤1e-5 | **DEFERRED** | root blocker re-localized to online-CTR bins; production untouched |
| (d) pc=1/pc=2 locks + tensor_ctr_e2e (3/3) + ordered_boost_e2e (2/2) + cb-train lib (130/130) | GREEN | all pass |
| (e) fold-creation ground truth | GREEN | committed accounting + live-trainer ground truth |

## Deviations from Plan

### Authorized deviation taken: FALLBACK (defer bar (c))
- **Trigger:** After porting the structure-fold cycling AND correcting the
  averaging permutation to the bit-exact upstream value, the pc=4 e2e still
  diverged — root cause is the online-CTR bin computation
  (`ComputeOnlineCTRs(AveragingFold)`), a CTR-subsystem parity gap with blast
  radius across all committed CTR locks.
- **Action (per objective FALLBACK + Rule 4):** reverted all production changes
  (`fold.rs`/`boosting.rs` zero diff), left the pc=4 e2e oracle uncommitted, and
  committed the live-trainer ground truth + re-localized blocker evidence.
- **No weakening:** no `#[ignore]`, no `assert_ne`, no pinned-delta, no fabricated
  pass.

### Rule 1 finding (documented, not applied): create_folds averaging permutation
- The committed `create_folds` averaging permutation is provably wrong (only
  partition-count-invariant against the locks). Fixing it in isolation regresses
  the locks because they compensate via wrong CTR bins. A correct fix must change
  BOTH the permutation and the online-CTR materialization together — recorded for
  the follow-up CTR-parity plan, NOT applied here (would regress green bars).

## Known Stubs
None introduced. Production code unchanged.

## Threat Flags
None. The live-trainer instrumentation and generators are offline-only (never
CI-wired); CI reads only the committed frozen fixtures. No new untrusted-input,
network, auth, or deserialization surface in the training library.

## Self-Check: PASSED
- Created files exist: `instrument_live_trainer_README.md`,
  `live_trainer_structure_fold.json`, `live_trainer_ctr_bins_blocker.json`,
  `05-17-SUMMARY.md` — all FOUND.
- Commits exist: `3dbce77` (live-trainer ground truth), `ebb0e4d` (bar-(c)
  re-localization) — both FOUND.
- Baseline oracles re-verified GREEN post-revert: partition 4/4, tensor_ctr_e2e
  3/3, ordered_boost_e2e 2/2, cb-train lib 130/130. Production diff empty.
