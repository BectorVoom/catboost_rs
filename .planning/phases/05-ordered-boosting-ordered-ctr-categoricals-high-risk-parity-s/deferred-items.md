# Deferred Items — Phase 05 (05-17 bar (c) escalation)

## 05-17 bar (c): pc=4 e2e oracle blocked on live-trainer averaging-fold CTR

**Status:** BLOCKED — genuinely-new blocker surfaced during live-instrumentation
task; live catboost-trainer build infeasible in this environment.

### Disk re-measurement (hard NO-GO gate — NOT tripped)
- `df -h /` at start: 242G total, 164G used, **66G free** (72%).
- `target/debug` did not exist at start (no reclaim needed).
- After full per-crate test builds: 172G used, **58G free** (75%). Ample headroom
  throughout. Disk NO-GO condition was never reached.

### What was recovered offline (ground truth, reproducible)
1. **Upstream pc=4 per-tree structure** (`model_pc4.json`): the 5 trees use ONLY the
   single feature-0 Borders CTR (borders `[2.999, 3.999, 7.999]`, split_index 0/1/2).
   Per-tree leaf_weights (the `BuildIndices(AveragingFold)` partition):
   - tree0 `[6,0,10,14]` borders `{2.999,7.999}` (structure A) — **we match**
   - tree1 `[8,8,0,14]` borders `{3.999,7.999}` (structure B)
   - tree2 `[6,0,10,14]` (A); tree3 `[8,8,0,14]` (B); tree4 `[8,8,0,14]` (B)
   - Pattern `[A,B,A,B,B]`. OUR train_cat produces `[A,X,A,X,A]` with
     `X=[9,0,7,14]` — diverges at trees 1,3,4.
2. **Structure-fold cycling confirmed as the mechanism** (`train.cpp:208`
   `takenFold = Folds[Rand.GenRand() % foldCount]`, `foldCount == LearningFoldCount
   == 3` at pc=4). Our `boosting.rs` uses a FIXED `Folds[0]` (identity) structure
   column for every tree; upstream cycles the structure fold per iteration.
3. **Leaf values use the FIXED AveragingFold** (`approx_calcer.cpp:1082`
   `const TFold& fold = ctx->LearnProgress->AveragingFold`) — confirmed.
4. **The AveragingFold permutation is bit-exact correct.** Our `create_folds` /
   Task-1 instrumented harness both produce
   `[23,19,25,29,14,13,10,12,9,3,2,4,17,21,27,11,28,8,1,15,26,16,7,24,5,6,20,0,22,18]`,
   and its cross-check gives tree0's `[6,0,10,14]` exactly (bar (a)/(b) hold).
5. **Whole-set CTR counts match upstream's baked `ctr_data.hash_map`** exactly
   (per-value `(c0,c1)`: `(6,0),(4,0),(0,5),(4,0)+unknown,(2,6),(1,6)`).

### The genuinely-new blocker (why bar (c) cannot close offline)
Applying tree1's borders `{3.999, 7.999}` to the bit-exact-correct AveragingFold
online CTR yields `[9,7,0,14]`, **NOT** upstream's `[8,8,0,14]`. The difference is
exactly ONE object at the online-CTR boundary `15·ctr = 3.75` (docs 4/13/27,
first-occurrence of their feature value: `(0+0.5)/(1+1)·15 = 3.75 < 3.999`).
EXHAUSTIVELY ruled out offline:
- All 3 learning folds + AveragingFold + pre-draw variants 0..7: NO permutation
  produces `[8,8,0,14]` (probe `probe_each_fold_float_for_b`).
- read-before-increment, read-after-increment, `>=` vs `>` border test, quantized
  integer bins vs raw float CTR: none move a boundary object across 3.999.
- Brute-force over all 3^5 structure-fold sequences (real CTR-aware search + fixed
  averaging leaf values): best maxdiff vs `predictions_pc4.npy` is **0.176**, never
  ≤1e-5 — because no fold's structure ever reproduces the `[8,8,0,14]` partition.

The `[8,8,0,14]` partition is therefore a property of the **live training-time
AveragingFold online CTR values** (`ComputeOnlineCTRs(AveragingFold)`, recomputed
per iteration, `train.cpp:252-296`), whose exact boundary placement is NOT
recoverable from the (correct) permutation + offline online-prefix reconstruction.
Recovering it requires instrumenting the LIVE catboost 1.2.10 trainer to dump the
per-iteration averaging-fold CTR float values.

### Why the authorized live build is infeasible HERE (not a disk NO-GO)
- The full catboost trainer build (`build_native.py`) requires **Conan** and
  **Ninja**, neither installed; it also fetches all contrib deps over the network.
  Only `cmake` + `g++` are present. Installing Conan/Ninja + network contrib
  fetch + a multi-hour full-graph compile is a NEW large environment/architectural
  step well beyond "porting a recovered rule" — the objective's explicit
  STOP-and-checkpoint condition.
- The prebuilt `.venv` `_catboost.so` (1.2.10) cannot be patched to emit the
  averaging-fold CTR without rebuilding the same trainer graph.

### Recommended resolution (needs a decision)
Either (a) provision Conan+Ninja+network and build a minimal instrumented
`_catboost`/`catboost_train_interface` patching `CalcLeafValues`/`ComputeOnlineCTRs`
to dump the per-iteration averaging-fold CTR floats (the true ground truth), then
transcribe the structure-fold cycling + that exact boundary CTR into `boosting.rs`;
or (b) accept bar (c) deferred at pc=4 with bars (a),(b),(d),(e) green and the
mechanism + evidence documented here. Bars (a)/(b)/(d)/(e) are NOT regressed: no
production code was modified in this session.

---

## 05-17 bar (c) — UPDATE: live trainer BUILT; structure-fold + permutation recovered; root blocker re-localized to online-CTR bins

**Decision taken (2026-06-15):** "Attempt toolchain provision + build." DONE.

### Toolchain provisioned (sudo-free) and instrumented trainer built
- Conan 2.29.1, Ninja 1.13.0, Cython 3.2.5(+numpy) via `uv tool install`.
- **clang-18 + lld-18** (catboost 1.2.10's vendored libc++ needs clang>=16 builtins;
  clang-14 FAILS). `apt-get download` (no sudo) + `dpkg -x` into `/tmp/clang18_prefix`.
- Built instrumented `_catboost` against the project `.venv` Python 3.13 via
  `build_native.py` (FindPython overridden to 3.13; default picks system 3.12 → ABI
  mismatch). Conan fetched openssl/ragel/swig/yasm from conancenter (network OK).
- Instrumented `train.cpp` + `learn_context.cpp` (env-gated `CB_INSTRUMENT_LOG`).
  The trainer reproduced upstream tree leaf_weights `[A,B,A,B,B]` and predictions
  **bit-identical (max abs diff 0.0)** to `predictions_pc4.npy`. Disk stayed >47G free.

### What the live instrumentation RECOVERED (committed ground truth)
1. **Structure-fold cycling** (`live_trainer_structure_fold.json`): per iteration
   `takenFold = Folds[Rand.GenRand() % foldCount]` (`train.cpp:208`); pc=4 sequence
   `[0,2,0,2,2]`; leaf values stay on the FIXED AveragingFold. Ported & verified:
   the structure-fold cycling produces tree0 `[6,0,10,14]` exactly.
2. **The AveragingFold permutation in `create_folds` was WRONG.** The `fold_cc`
   `GetCallCount()` log proves the averaging Fisher-Yates shuffle begins at
   call-count **29 (pc=1) / 87 (pc=4)** = `learning_folds` FULL Fisher-Yates passes
   (one prefix pass + one per non-identity learning fold), NOT the committed
   "callcount == learning_folds" (=1/=3) rule. Our `uniform`-shuffle advanced to
   cc=87 reproduces upstream `[11,18,15,29,16,...]` BIT-EXACT (cc=29 →
   `[10,17,25,3,6,...]` for pc=1). The OLD permutation `[23,19,25,...]` only
   *coincidentally* yielded the partition COUNTS `[6,0,10,14]`/`[6,0,7,17]`.

### The NEWLY-LOCALIZED root blocker (`live_trainer_ctr_bins_blocker.json`)
Even WITH the bit-exact averaging permutation, **our online-CTR ui8 bins differ
from upstream's `ComputeOnlineCTRs(AveragingFold)`.** Upstream avg-order bins
`[7,7,7,11,3,7,2,11,7,12,9,...]` are not reproduced by `materialize_ctr_feature`
(single-cat Borders, prior 0.5, border_count 15): the first 4 match then diverge
at position 4 (ours 7, upstream 3). Neither class-0-as-good nor the {0,1}
combination projection reproduces the sequence. Correcting `create_folds` to the
true permutation therefore REGRESSES pc=1/pc=2 `[6,0,7,17]`→`[6,0,8,16]` and the
pc=1 e2e — because the current locks are pinned to the *compensating*
wrong-permutation + wrong-CTR-bins combination.

### Why bar (c) is still deferred (architectural blast radius)
Closing (c) now requires a `cb_train` online-CTR materialization fix to match
`ComputeOnlineCTRs(AveragingFold)` bit-exact — a CTR-SUBSYSTEM change that ripples
through EVERY committed CTR oracle (pc=1/pc=2/pc=4 partition locks, tensor_ctr_e2e,
the leaf-value path), since those locks currently pass on the compensating-error
combination. That is beyond "port the recovered structure-fold rule" and is the
objective's explicit STOP-and-checkpoint condition. Per the FALLBACK: production
code left UNTOUCHED (`fold.rs`/`boosting.rs` zero diff), the pc=4 e2e oracle left
UNCOMMITTED, bars (a),(b),(d),(e) green. The instrumented ground truth
(`live_trainer_structure_fold.json`, `live_trainer_ctr_bins_blocker.json`,
`instrument_live_trainer_README.md`) is committed so a follow-up CTR-parity plan
can transcribe the exact `ComputeOnlineCTRs(AveragingFold)` bins directly.

### Plan 05-18 execution finding (2026-06-15): the proposed reindex fix is INSUFFICIENT

Plan 05-18 attempted to close bar (c) via THREE coupled fixes. Two are now PROVEN
correct and bit-exact, but the third (the CTR-bins fix) is PROVEN insufficient, so
the wave could NOT reach a simultaneously-green state and was reverted to the green
baseline per the plan's hard invariant #3 (STOP-and-checkpoint, do not commit a
weakened/compensating/regressed state).

1. **create_folds averaging permutation (CORRECT, bit-exact).** Rewriting the
   learning-permutation-needed branch so each of `learning_folds` fold positions
   performs one FULL Fisher-Yates pass (29 draws for N=30) before the averaging
   pass reproduces `permutations(30, learning_folds+1, 0)[learning_folds]`:
   pc=4 → `[11,18,15,29,16,12,0,7,...]` == `upstream_avg_perm` BIT-EXACT;
   pc=1 → `[10,17,25,3,6,...]`. Verified by scratch assertion.

2. **Per-iteration structure-fold cycling (CORRECT, sequence bit-exact).** Driving
   the structure-fold pick from the persistent fold-creation RNG continued through
   the committed live-trainer call-count schedule (`[116,140,166,190,212]` for pc=4,
   from `live_trainer_structure_fold.json`) reproduces the pc=4 fold sequence
   `[0,2,0,2,2]` integer-exact. learning_folds==1 (pc=1/pc=2) is always fold 0
   (byte-identical, no RNG consumed). NOTE: the intermediate per-iteration draw
   counts (24/26/24/22) are config/data-coupled (`GreedyTensorSearch` Rsm +
   per-level `CalcScores` + leaf seed) and are NOT reproducible by the
   bootstrap=No/random_strength=0 trainer, so the call-count schedule had to be
   TRANSCRIBED from the committed fixture rather than derived.

3. **materialize_ctr_feature permutation-order bucket reindex (INSUFFICIENT — the
   blocker).** The plan hypothesized that assigning dense first-seen bucket IDs in
   PERMUTATION order (per upstream `ComputeReindexHash`, index_hash_calcer.cpp:171-227)
   would reproduce `upstream_avg_ctr_bins_avg_order`. It does NOT:
     - For a SINGLE-feature CTR ({0}), the bucket-ID assignment order is a **no-op**
       on the online prefix: the read-before-increment counts within a bucket depend
       only on which docs precede in permutation order, NOT on the bucket's integer
       ID. So the reindex order cannot change single-feature bins.
     - Production `materialize_ctr_feature` over feature-0 under the corrected pc=4
       averaging permutation yields avg-order bins
       `[7,7,7,11,7,11,11,7,7,3,7,3,2,12,9,...]` — matching upstream's first 4
       (`7,7,7,11`) then diverging at position 4 (ours 7, upstream 3), EXACTLY the
       divergence the blocker fixture localized.
     - Neither `good=N[1]` nor `good=N[0]`, nor the `{0,1}` combination projection,
       reproduces `upstream_avg_ctr_bins_avg_order` `[7,7,7,11,3,7,2,11,7,12,9,12,...]`.
   The upstream `ComputeOnlineCTRs(AveragingFold)` bins are a property of the live
   training-time online-CTR path NOT captured by the offline materialization (as the
   blocker fixture itself stated). This is a deeper CTR-subsystem divergence than a
   reindex-order swap.

**Consequence (the Rule-3 STOP).** With fix #1 correct, the pc=1 averaging
permutation changes from the old call-count-1 draw to the second-full-pass draw,
and fix #3 cannot compensate (it's a no-op for the single-feature CTR). Result:
**pc=1 `tensor_ctr_e2e_oracle_test` REGRESSED** (predictions diff ≈ 0.0056, was
GREEN ≤1e-5), and pc=4 e2e still diverged (diff ≈ 0.028). The plan's hard invariant
#3 ("if pc=1 REGRESSES the fix is WRONG — STOP, do not re-pin to a cb-train value")
fired. The wave was reverted to the green baseline (Task-1 commit `git reset`); no
oracle was weakened, no fixture touched, the pc=4 e2e oracle remains UNCOMMITTED.

**What a real closure needs (next plan).** A faithful re-implementation of
`ComputeOnlineCTRs(AveragingFold)` (online_ctr.cpp:600-790 / CalcOnlineCTRSimple /
CalcQuantizedCtrs) that reproduces the exact avg-order ui8 bins — NOT a reindex-order
tweak of the existing `online_ctr_prefix_binclf`. Only then can the corrected
permutation (#1) + structure-fold cycling (#2) + the true bins (#3) be re-pinned
together to a simultaneously-green pc=1 + pc=4 + partition-lock state. Until that
online-CTR fix exists, bar (c) stays deferred; bars (a),(b),(d),(e) remain green.

---

## 05-18 (2026-06-15) — Spike-001 RESOLVED; bar (c) blocker precisely re-localized to data-provider storage reorder S

**Re-instrumentation (second, re-authorized cycle) SUCCEEDED** and produced a
SELF-CONSISTENT oracle (`live_trainer_self_consistent.json`), superseding the
internally-inconsistent `live_trainer_ctr_bins_blocker.json`. Two atomic
`CB_INSTRUMENT_LOG` events were added (`train.cpp` `self_consistent_ctr`,
`online_ctr.cpp` `online_ctr_inputs`).

**The Spike-001 inconsistency, explained.** The averaging online-CTR ui8 bins
(`GetData`/`Feature[docIdx]`) are stored in the CTR materialization order **Q**,
where `Q = S ∘ LearnPermutation`. The first cycle paired those bins with
`GetLearnPermutationArray()` = `[11,18,15,29,…]` (the leaf-index iteration order, a
DIFFERENT order) and never logged `LearnTargetClass` — hence inconsistent. With the
atomic capture the bins ARE the single-cat-0 Borders online prefix under order Q
with target `LearnTargetClass[1]` (= binarized y; `[0]` is all-zeros/unused). Q
reproduces all 5 partitions `[6,0,10,14],[8,8,0,14],…` bit-exact.

**The REAL bar-(c) blocker (sharper than the prior "online-CTR bins" note).** The
online-CTR math in cb-train is ALREADY correct (Spike-001 proof). The gap is `S` —
the catboost quantized data-provider's internal object STORAGE reorder (verified:
the identity learning fold's `perm_subset == [0..29]` produces a NON-natural object
cat-sequence). pc=4 tree-B borders `[3,7]` SPLIT the mixed cat buckets, so leaf
VALUES depend on the exact per-mixed-bucket bin→object assignment fixed by `S`. The
must_haves' `[11,18,15,…]` reproduces the pc=4 PARTITION `[8,8,0,14]` but the WRONG
leaf VALUES (leaf1 `sum_y=6` vs upstream `5`, ≫1e-5). cb-train materializes on
object-order `X_cat` WITHOUT `S`; no `create_folds`-generable permutation reproduces
upstream's pc=4 leaf values, and re-pinning to a lucky cb-train permutation is
forbidden (invariant #2). pc=1 is green because its borders do not split the mixed
buckets (leaf composition order-invariant there) — NOT because cb-train reproduces
the CTR order.

**FALLBACK taken (user-chosen 2026-06-15).** cb-train production
(`fold.rs`/`boosting.rs`) UNTOUCHED; pc=4 e2e oracle UNCOMMITTED; no oracle
weakened. Committed: the self-consistent oracle + blocker annotation + README
update. Closing bar (c) requires porting `S` (the data-provider quantized-object
storage order) into cb-train — a research-grade subsystem, a new plan. Bars
(a),(b),(d),(e) remain green.
