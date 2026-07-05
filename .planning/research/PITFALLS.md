# Pitfalls Research

**Domain:** Parity-critical gradient-boosting library — v1.2 new surfaces (ONNX/CoreML export, GPU inference evaluator, extended fstr, CV/tuning/snapshot orchestration, online-HNSW KNN, benchmark + PyPI release)
**Researched:** 2026-07-05
**Confidence:** HIGH for ONNX/CoreML export limits, GPU-inference determinism, and HNSW feasibility (upstream source + instrumented-trainer evidence already in-repo); MEDIUM for CV fold-parity internals and PyPI free-threaded specifics.

> **Headline findings for roadmap risk-planning**
> 1. **ONNX/CoreML export of categorical/CTR/text/embedding models is IMPOSSIBLE upstream — do not scope it as a bug to fix.** CatBoost itself refuses these formats for any model with non-float features. The v1.2 requirement must be reframed to "float-feature models only, error clearly otherwise." (Pitfall 1, 2)
> 2. **FEAT-07 bit-exact HNSW parity IS achievable — but only by porting the graph-construction RNG and insertion order bit-for-bit, NOT by writing "a good approximate KNN."** HNSW is deterministic given identical seed + insertion order + distance. The requirement should read "reproduce upstream's `online_hnsw` graph + search exactly," and its oracle must be the per-object neighbor **set**, not the final prediction. (Pitfall 8)
> 3. **The GPU inference evaluator re-imports every v1.1 GPU determinism landmine** (double-accumulate atomic-add ordering, the `-inf`-literal HIP reject, `|=` A100 codegen bug) and adds upstream's own hard restrictions (oblivious-only, 1-dim-only, no cats, only 3 prediction types implemented). Match the restrictions exactly rather than "doing better." (Pitfall 5, 6, 7)

---

## Critical Pitfalls

### Pitfall 1: Trying to export categorical / CTR / text / embedding models to ONNX

**What goes wrong:**
A CatBoost model trained with categorical features (the library's flagship capability, via ordered target statistics / CTR), or with text/embedding features, **cannot be represented in ONNX-ML at all**. Upstream CatBoost's `save_model(format="onnx")` only supports datasets **without** categorical features; text and embedding features "will likely never be supported." If v1.2 scopes "ONNX export" as a blanket feature, it will pass on the toy float-only fixtures and then fail (or silently produce a wrong model) the moment a real CatBoost user — whose entire reason for using CatBoost is categorical handling — exports their model.

**Why it happens:**
The ONNX `TreeEnsemble`/`TreeEnsembleClassifier` operators have no node mode for set-membership tests; a CTR is a learned numeric transform of a category's target statistics computed *during training* with permutation/ordering — there is no ONNX primitive for it. Developers assume "export = serialize the trees" and miss that CatBoost's split conditions reference CTR feature values that don't exist outside the trainer.

**How to avoid:**
- **Reframe the requirement** to match upstream exactly: ONNX export is supported **only for models whose features are all float/numeric**. Enforce it at the export entry point with a typed `thiserror` variant (`ExportUnsupported { format: Onnx, reason: "categorical/text/embedding features present" }`) — mirror CatBoost's own error, never emit a silently-wrong graph.
- Document the upstream workaround (cast categoricals to float64 pre-training loses native handling) but do **not** implement CTR-in-ONNX — it is out of scope and upstream doesn't do it either.
- Support the three model types upstream supports: binary classification (`Logloss`), multiclass (`MultiClass`), regression. Use `ai.onnx.ml` TreeEnsemble operators.

**Warning signs:**
- Test fixtures are all numeric-only → false confidence. Real CatBoost adoption models are categorical-heavy.
- An export path that "succeeds" on a categorical model is a red flag — upstream *rejects* it, so you've diverged.

**Phase to address:** Model export (ONNX) phase — gate the scope in the SPEC as "float-only models."

---

### Pitfall 2: ONNX/CoreML float-precision drift silently breaking the ≤10⁻⁵ parity bar

**What goes wrong:**
The project's parity bar is **≤10⁻⁵ absolute vs official CatBoost, and CatBoost accumulates leaf values in `double`** (Bias is `double`, the CPU applier sums in double). ONNX `TreeEnsemble` and CoreML tree evaluators operate in **float32** internally (ONNX Runtime's tree ensemble accumulates in float; CoreML is float32 end-to-end). Over hundreds of trees, float32 accumulation drifts well beyond 10⁻⁵ from the double reference — so an exported model that is *structurally correct* still "fails parity" if you test it against the `.cbm` double predictions at the CPU bar. Teams then chase a non-bug or, worse, loosen the bar globally.

**Why it happens:**
Confusing "the exported model is faithful" with "the exported model reproduces double-precision predictions." The drift is inherent to the target runtime's numeric type, not an export bug.

**How to avoid:**
- **Do not apply the ≤10⁻⁵ double bar to exported-model outputs.** Define an export-specific tolerance and test methodology:
  - **Oracle = official CatBoost's own ONNX/CoreML export**, evaluated in the *same runtime* (ONNX Runtime / CoreML). Compare `our_onnx → ORT` against `catboost_onnx → ORT`. Both suffer the same float32 drift, so this comparison can legitimately be tight (target ≤10⁻⁵–10⁻⁶ on `RawFormulaVal`, because the numeric path is identical).
  - **Secondary check**: `our_onnx → ORT` vs our own `.cbm` predictions, at a **documented looser tolerance** (realistic: ~10⁻⁴ to 10⁻⁵ on RawFormulaVal for shallow/short ensembles; larger for deep/long ensembles) — this catches structural bugs but is explicitly *not* the parity gate.
- Test on **RawFormulaVal** first (identity post-processing) to isolate tree math from sigmoid/softmax. Add Probability/Class as separate checks.
- Use ONNX Runtime as the checker (it is the reference ONNX-ML tree evaluator); pin the opset and ORT version in the test harness.

**Warning signs:**
- Parity "failures" that grow with `iterations` / `depth` → precision drift, not a bug.
- Passing at 10⁻² but failing at 10⁻⁵ against `.cbm` double predictions → expected float32 behavior, re-baseline the oracle.

**Phase to address:** Model export phase — define export tolerance + ORT-as-checker in the SPEC before coding.

---

### Pitfall 3: ONNX opset / prediction-type / binary-label mismatches

**What goes wrong:**
- **Opset drift:** the `ai.onnx.ml` TreeEnsemble op semantics changed across opsets; exporting against one opset and validating with an ONNX Runtime built for another produces subtle score differences or load failures.
- **RawFormulaVal vs Probability confusion:** CatBoost's raw tree sum is `RawFormulaVal`; Probability requires the sigmoid/softmax post-transform. If the export bakes in the wrong prediction type (or none), consumers get logits when they expect probabilities.
- **The known onnxruntime binary-classification label bug:** ORT infers the *label* incorrectly for binary classification exported from CatBoost — the documented guidance is "ignore the label output for binary classification, use the probability." A parity test that asserts on the label will spuriously fail.

**Why it happens:**
Export parameters (`onnx_domain`, `onnx_model_version`, opset) are under-specified, and the ONNX-ML classifier output contract (label + probabilities map) is more complex than a single score.

**How to avoid:**
- Pin the export opset explicitly and match it in the ORT test environment; record both in the fixture manifest.
- Export and test **RawFormulaVal** and **Probability** as distinct, labeled cases; make the prediction type an explicit export parameter mirroring CatBoost's `export_parameters` / `prediction_type`.
- In binary-classification parity tests, **assert on the probability/score, not the ORT-inferred label** (document the upstream ORT bug in the test).

**Warning signs:** Load errors mentioning unsupported opset; label mismatches only in binary classification; probabilities off by a monotonic transform (missing sigmoid).

**Phase to address:** Model export phase.

---

### Pitfall 4: CoreML non-determinism, float32-only, and limited op support

**What goes wrong:**
CoreML export shares the categorical-feature prohibition (float-only), is **float32 end-to-end** (same drift as Pitfall 2, usually worse — no double anywhere), and its tree/activation op coverage is narrower than ONNX. Exotic post-transforms (RMSEWithUncertainty, MultiProbability, custom activations) have no CoreML representation. CoreML also carries metadata and can exhibit platform-dependent evaluation (only truly testable on macOS/Apple runtimes), so a "successful export" produced/validated on Linux CI may not be faithfully checkable at all in-env.

**Why it happens:**
CoreML is an Apple-ecosystem format; the toolchain (coremltools) and the evaluator are macOS-centric, and the format targets float32 mobile inference, not double-precision parity.

**How to avoid:**
- Scope CoreML to the **same float-only, 1-dim + multiclass, RawFormulaVal/Probability/Class** subset as upstream; reject everything else with a typed error.
- Oracle strategy = **official CatBoost's CoreML export as the reference**, compared in the same evaluator, at a float32-realistic tolerance (≥10⁻⁴). Do not hold CoreML to the ≤10⁻⁵ double bar.
- Decide the validation environment up front: if no macOS/CoreML runtime is available in CI, treat CoreML parity as **structural** (compare the exported CoreML tree structure/leaf values byte-or-field-wise against CatBoost's CoreML export) rather than execution-based, and document the gap — do **not** fabricate execution parity you can't run (mirrors the Kaggle-CUDA "no in-env oracle" discipline already used for GPU).

**Warning signs:** CoreML parity "passes" but was never executed on an Apple runtime; predictions differ from ONNX for the same model (float32 vs the ONNX path).

**Phase to address:** Model export (CoreML) phase — pair with ONNX but keep tolerance/validation separate.

---

### Pitfall 5: GPU inference evaluator — non-deterministic reductions breaking ε=1e-4

**What goes wrong:**
Upstream's GPU evaluator accumulates each document's prediction into a `double4` and folds tree-sub-blocks with a **`TAtomicAdd` into `results`** (§7.1 of `CATBOOST_CUDA_KERNELS_DESIGN.md`). Atomic-add ordering is non-deterministic across launches, so a naive CubeCL port gives run-to-run jitter. For a *single* model apply this jitter is tiny, but the same class of non-determinism is exactly what forced v1.1 to build a **fixed-point u64 deterministic reduction** to hold ε=1e-4 across hundreds of trees. An inference evaluator that sums many trees per document hits the same wall — and worse, the parity bar is now against CatBoost, not just self-consistency.

**Why it happens:**
Floating-point atomic-add is not associative; thread-scheduling order varies. Developers assume "inference is just a sum, order doesn't matter."

**How to avoid:**
- **Reuse the v1.1 deterministic-reduction machinery** (fixed-point u64 accumulation → convert to float) for the per-document tree-value fold. This is already proven on P100 and gfx1100 to hold ε=1e-4 across hundreds of trees (Key Decision, v1.1). Do not re-introduce a raw float `TAtomicAdd`.
- Accumulate leaf values in the widest precision the backend allows (upstream uses `double`; match that where the backend supports f64 atomics, else fixed-point u64 as in v1.1 — note gfx1100 lacks f64 atomic-add, so the fixed-point path is mandatory in-env).
- Oracle the GPU evaluator against the **CPU Rust predictor** at ε=1e-4 (the established GPU bar) AND against official CatBoost `EnableGPUEvaluation` output on Kaggle CUDA (the sole authoritative GPU oracle).

**Warning signs:** Predictions differ across repeated identical GPU applies (run twice, diff > 0); parity that passes on 10 trees and drifts at 500 trees; ε holds on wgpu/cpu facade but fails on real rocm/CUDA.

**Phase to address:** GPU inference evaluator phase.

---

### Pitfall 6: GPU inference evaluator — the `-inf`-literal HIP landmine (and other CubeCL codegen traps) re-biting

**What goes wrong:**
The feature accessor in upstream's evaluator returns `NegativeInfty()` for out-of-range features (§7.1). A direct CubeCL transcription — `F::new(f32::NEG_INFINITY)` inside a `#[cube]` kernel — emits a bare `double(-inf)` literal that the **HIP/comgr compiler on gfx1100 rejects** (`use of undeclared identifier 'inf'`), failing the whole kernel JIT at runtime. This is **invisible to `cargo check` and to the `cpu`/`wgpu` facades** — it only surfaces when the rocm suite runs on the real GPU. It cost v1.1 a 16/75 rocm failure that looked like a logic bug.

**Why it happens:**
CubeCL's cpu facade compiles infinity literals fine; the HIP backend does not. Verifier/fixer subagents cannot reach the GPU, so the trap escapes normal review.

**How to avoid:**
- **Never seed a `#[cube]` kernel reduction/accessor with an infinity literal.** Use the finite sentinel `F::new(f32::MIN)` (~-3.4e38) for the out-of-range/argmin seed — behaviorally identical to -inf for all realizable inputs. Host-side (oracle) code may keep `f64::NEG_INFINITY`.
- Also carry forward the **`|=` → `+=`/`<<` leaf-index construction** (upstream replaced `|=` with `+=`/`<<` to dodge the A100 codegen bug `MLTOOLS-6839`, noted in §7.1) — transcribe the shift form, not the OR form.
- **Run the rocm suite in-env yourself before any GPU-inference sign-off** (`cargo test -p cb-backend --no-default-features --features rocm`). `cargo check`/cpu tests cannot catch HIP codegen rejects. Do not delegate GPU validation to a subagent that can't reach the device.

**Warning signs:** rocm `ServerUnhealthy`/`Launch` compilation errors; tests green on cpu/wgpu but a fraction fail on rocm; failures that vanish when you remove an infinity/`|=` expression.

**Phase to address:** GPU inference evaluator phase (and any phase touching `#[cube]` code).

---

### Pitfall 7: GPU inference evaluator — silently exceeding upstream's supported subset

**What goes wrong:**
Upstream's `TGpuEvaluator` **enforces at construction**: oblivious (symmetric) trees only, **one output dimension only** (`CB_ENSURE(GetDimensionsCount() == 1)`), and **no categorical/text/embedding features**. Only `RawFormulaVal`, `Probability`, and `Class` post-transforms are implemented; `Exponent`, `RMSEWithUncertainty`, `MultiProbability`, multidim softmax, and `CalcLeafIndexes*` **`ythrow "Unimplemented on GPU"`**. A port that "helpfully" runs multiclass or non-symmetric trees on device will diverge from CatBoost (which falls back to CPU for those) — a parity failure, not a feature.

**Why it happens:**
The v1.2 CPU/training engine already supports non-symmetric trees, multiclass, CTR, uncertainty — so it's tempting to route all of them through the new GPU evaluator. But upstream's GPU *inference* path is deliberately narrow; matching CatBoost means matching the narrowness.

**How to avoid:**
- Mirror upstream's construction-time guards exactly: reject non-oblivious, `ApproxDimension != 1`, and any cat/text/embedding model with a typed error, and **fall back to the CPU evaluator** (the `Ok(None)`→CPU seam pattern from v1.1 is the right shape).
- Implement only `RawFormulaVal`/`Probability`/`Class`; explicitly error on the unimplemented prediction types (do not approximate them on device).
- Keep the warp-interleaved quantized-buffer layout and `FeatureVal = CPU SplitIdx` verbatim (no border transform, per §7.1) — transforming borders is a classic silent divergence.

**Warning signs:** GPU predict "works" on a multiclass or CatFeature model where CatBoost would have thrown/fallen back; predictions match CPU on `RawFormulaVal` but diverge on Probability (post-transform bug).

**Phase to address:** GPU inference evaluator phase.

---

### Pitfall 8: Online-HNSW — treating "approximate" as "non-deterministic" and declaring bit-exact parity infeasible

**What goes wrong:**
The Phase-6.5 estimated-feature KNN residual was diagnosed **definitively** (instrumented catboost 1.2.10, `max|diff|=0.0` on frozen predictions): upstream's KNN embedding calcer uses `NOnlineHnsw::TOnlineHnswDenseVectorIndex<float, TL2SqrDistance<float>>` (approximate NN, `searchNeighborhoodSize=300`), while the Rust `cb_compute::KnnCalcer` is **brute-force-exact**. The two return *different neighbor sets* (proof: query doc6 exact 3-NN `{0,2,4}` vs upstream HNSW `{1,3,4}`), so exact-KNN can never match. The trap is concluding "approximate ⇒ random ⇒ bit-exact parity is impossible, weaken the requirement." **That is wrong.** HNSW is *approximate* but **deterministic**: given the same seed, same insertion order, same distance function, and same graph-construction RNG, it builds an identical graph and returns identical (approximate) neighbors every time.

**Why it happens:**
"Approximate nearest neighbor" is conflated with "stochastic output." ANN indices are approximate w.r.t. *true* nearest neighbors, but their output is a pure deterministic function of (data, insertion order, RNG stream, parameters).

**How to avoid — and the feasibility answer the roadmap needs:**
- **Bit-exact parity IS achievable.** The requirement must be **reframed** from "implement a good approximate KNN" to: **"port `catboost-master/library/cpp/online_hnsw` bit-for-bit — the dynamic dense graph, incremental insert order, `TL2SqrDistance`, and the RNG-driven level/neighbor selection — so the constructed graph and search results match upstream's exactly."** This overrides the original Phase-6.5 decision "no third-party HNSW crate" (A2/D-05): you cannot use an off-the-shelf HNSW crate (hnswlib/`instant-distance`) because its RNG and construction order will differ — you must transcribe upstream's.
- **The oracle must be the per-object neighbor *set* (and vote order), not the final prediction.** Instrument `knn_neighbors` and assert the returned indices match upstream index-for-index over the shuffled prefix `S = create_shuffled_indices(n, seed)`. Also reproduce the **class-vote-order** convention (upstream feat0 = class-1 vote; the Rust `[class0,class1]` order was a separate bug). Getting the prediction to ≤10⁻⁵ *follows* from matching the neighbor set — testing only the prediction hides which half is wrong.
- Scope is bounded and known: ~936 LOC (dynamic dense graph + incremental insert + HNSW search + `TL2SqrDistance` + RNG). It is **its own focused phase** (FEAT-07) with the instrumented-trainer recipe already documented (`catboost-instrumented-trainer-build`).
- The boosting-loop integration is **easy** (already proven): a single static online-over-`S` column for train + offline post-hoc apply; `fold_count=1`, no cycling, no `train_inner` change. Do not re-open the disproven hypotheses (thread the learn permutation; structure-fold cycling/averaging) — both are DISPROVEN.

**Warning signs:**
- Anyone proposing to "just use a mature HNSW crate for speed" — guarantees a different RNG/graph → parity permanently unreachable.
- A KNN oracle that asserts only on the final XOR prediction (masks the neighbor-set divergence).
- Neighbor sets that agree on early prefixes but diverge from p5 onward (the exact signature of construction-order/RNG mismatch, already observed).

**Phase to address:** FEAT-07 online-HNSW port — a dedicated debt/hardening phase. **Recommend the roadmap state the requirement as "match upstream `online_hnsw` graph construction & search bit-for-bit," with per-object neighbor-set oracles, not "approximate KNN within tolerance."**

---

### Pitfall 9: CV / hyperparameter tuning — data leakage and fold-partition divergence from CatBoost's `cv()`

**What goes wrong:**
Two distinct failures:
1. **Non-parity folds:** CatBoost's `cv()` has specific, non-obvious partitioning semantics — `type` ∈ {Classical, Inverted, TimeSeries}; **stratified sampling is ON by default for `Logloss`/`MultiClass`/`MultiClassOneVsAll` and OFF otherwise**; optional `shuffle` before splitting; and **group identifiers force all objects of a group into the same fold**. A reimplementation that uses sklearn's `KFold`/`StratifiedKFold` or a naive contiguous split will produce different folds → different per-fold metrics → "parity failure" that is really a fold-assignment mismatch. Ranking data with query groups is especially easy to get wrong (splitting mid-group leaks and breaks group metrics).
2. **Leakage in tuning:** grid/random search that computes CTR/target-statistics or fits the quantization borders on the *full* dataset before CV, or that reuses a snapshot across folds, leaks target information into validation. CatBoost's ordered target statistics are precisely designed to avoid this — a careless orchestration layer can undo that guarantee.

**Why it happens:**
CV/tuning is treated as a thin wrapper, so the subtle CatBoost-specific defaults (stratification triggers, group-awareness, per-fold border computation) get skipped. Leakage is invisible — it *improves* CV scores, so it looks like success.

**How to avoid:**
- Reproduce `cv()` semantics exactly: implement all three `type`s, match the stratification default-by-loss-function rule, honor `shuffle`/`partition_random_seed`, and keep **groups intact within a fold**. Oracle the *fold assignment* (which object → which fold, given seed) against CatBoost, not just the final metric.
- Compute quantization borders, CTR statistics, and any target-dependent preprocessing **inside each fold's train split only** — never on the full pool. Add an explicit test that a target-permuted feature yields ~chance CV score (leakage canary).
- For hyperparameter tuning, reuse the CV harness per candidate; never carry a snapshot/warm-start across folds unless explicitly matching CatBoost's `randomized_search`/`grid_search` behavior.

**Warning signs:** CV scores noticeably better than held-out test; per-fold metrics that match CatBoost on regression but not on Logloss (stratification default missed); ranking CV where group metrics are unstable (split mid-group).

**Phase to address:** Orchestration phase (CV / tuning).

---

### Pitfall 10: Snapshot / resume — non-reproducible resumed training

**What goes wrong:**
`save_snapshot`/`snapshot_file`/`snapshot_interval` must let training **resume bit-identically** to an uninterrupted run. If the snapshot omits any piece of RNG/iteration state — the boosting iteration counter, the permutation/fold RNG state, the ordered-boosting fold permutations, the learning-rate schedule position, the overfitting-detector history, the current approx cursor — a resumed model diverges from a straight run, silently breaking the ≤10⁻⁵ parity bar for any user who relies on snapshotting.

**Why it happens:**
Snapshot is treated as "dump the trees so far," but reproducible resume requires serializing the *entire* trainer state, including the RNG streams that drive sampling and ordered boosting (the same permutation machinery that was hard-won in Phase 5).

**How to avoid:**
- Snapshot the **complete** trainer state: iteration index, all RNG/permutation states, ordered-boosting fold assignments, approx cursors, best-iteration/overfitting-detector state, metric history. Version the snapshot format.
- Oracle test: train N iterations straight vs train `k`, snapshot, resume to `N` → assert **bit-identical** model (≤10⁻⁵, ideally 0.0). Do this for plain *and* ordered boosting, and with sampling enabled (the RNG-sensitive cases).
- Reject resume across incompatible parameter/version changes with a typed error (CatBoost's "exclusive parameters" class of error).

**Warning signs:** Resumed model differs from straight run; divergence only appears with `bootstrap`/`subsample`/ordered boosting (RNG state not restored); resume works for 1 fold but not with CTR.

**Phase to address:** Orchestration phase (snapshot/resume).

---

### Pitfall 11: PyPI release — per-backend wheel confusion and abi3/free-threaded traps

**What goes wrong:**
- **Backend/wheel mismatch:** the project ships per-backend wheels (`cpu`/`rocm`/`wgpu`/`cuda`) selected at *compile time* via Cargo features. On PyPI there is one package name and one import (`catboost_rs`); a user `pip install`ing the wrong wheel for their hardware gets import/runtime failures. The rocm wheel additionally needs system ROCm libs via `ROCM_PATH` + `LD_PRELOAD` of `libhiprtc`/`libamdhip64` (the bundled patchelf-renamed `libhiprtc` **segfaults the HIP JIT**) — a landmine already hit in Phase 8.
- **abi3 vs free-threaded:** the wheels are `abi3-py312` (PyO3 0.29, `gil_used=false`). The concurrent free-threaded `fit`/`predict` UAT could not be validated in-env (needs `python3.13t`), so a free-threading regression could ship unverified. abi3 also constrains which PyO3 APIs are usable.

**Why it happens:**
GPU-backend selection is a compile-time axis that doesn't map cleanly onto PyPI's single-artifact-per-platform model; free-threaded Python is new and hard to test without a `t` build.

**How to avoid:**
- Decide and **document the distribution model** explicitly: either distinct PyPI package names per backend (`catboost-rs-cpu`, `catboost-rs-cuda`, …) or a CPU-default wheel + documented extra-index/local-build instructions for GPU wheels. Do not rely on users guessing.
- Ship the rocm wheel with its documented `ROCM_PATH`/`LD_PRELOAD` runtime requirement in `pyproject-rocm.toml`; never bundle the patchelf-renamed `libhiprtc`.
- Gate the free-threaded concurrency claim behind an actual `python3.13t` run (human-gated, mirroring the existing Phase-8 open UAT) before advertising free-threaded support; until then, document it as "abi3 built with `gil_used=false`, concurrency validated as a code property, not under a live `t` interpreter."
- Test the wheel install path on a clean environment per backend (import + a smoke predict) in CI.

**Warning signs:** ImportError/segfault on GPU wheels in a clean env; free-threaded claims with no `python3.13t` test evidence; users filing "wrong wheel" issues.

**Phase to address:** Adoption/DX (PyPI release readiness) phase.

---

### Pitfall 12: Benchmark vs official CatBoost — measuring the wrong thing / apples-to-oranges

**What goes wrong:**
The v1.2 benchmark must show accuracy + speed vs official CatBoost on real datasets. Easy ways to produce a misleading result: comparing against the **pre-Phase-10 host-light CPU baseline** (the 23.9–42.1× v1.1 numbers are vs *our own old baseline*, NOT vs official CatBoost — conflating them overstates competitiveness); benchmarking GPU on ROCm in-env (non-authoritative) instead of Kaggle CUDA; not pinning thread counts / `iterations` / hyperparameters identically; timing including JIT/first-launch kernel compilation; or comparing prediction accuracy without fixing the same borders/seed.

**Why it happens:**
Benchmark framing is subtle and the existing v1.1 speedup numbers (vs internal baseline) are seductive to reuse as if they were vs-official.

**How to avoid:**
- State the comparison baseline unambiguously: **official CatBoost** (matched version, e.g. 1.2.10) on the **same hardware**, same dataset, same hyperparameters, same thread count. GPU speed head-to-head runs on **Kaggle CUDA** (the sole authoritative GPU oracle), never on in-env ROCm.
- Separate accuracy parity (≤10⁻⁵ CPU / ε=1e-4 GPU vs CatBoost's *own* predictions) from speed. Report both.
- Exclude kernel-compile/JIT warmup from timed regions (warm up first); report medians over repeats.
- Discharge the standing v1.1 debt honestly in the benchmark: **GPUT-14 aggregate sign-off and the Phase-10/11 BENCH-02 rows were never run** — the v1.2 benchmark is the place to actually execute them, not to restate the partial BENCH-03 stitch as if complete.

**Warning signs:** Speedups quoted "vs CatBoost" that are actually vs the host-light baseline; GPU numbers from ROCm presented as authoritative; benchmark that omits the categorical-heavy real datasets CatBoost is optimized for.

**Phase to address:** Adoption/DX (benchmark) phase.

---

### Pitfall 13: Extended fstr — Interaction / LossFunctionChange / partial-dependence numeric divergence

**What goes wrong:**
The extended feature-importance methods each have exact CatBoost algorithms that are easy to approximate incorrectly:
- **LossFunctionChange** requires re-evaluating the loss with a feature's contribution removed — using the *training* loss vs a supplied dataset, and the correct handling of CTR/combination features, matters. Computing it against the wrong dataset or loss gives plausible-but-wrong numbers that pass a smell test but fail ≤10⁻⁵.
- **Interaction** strength is defined over CatBoost's specific pairwise split-cooccurrence accounting; a from-scratch definition won't match.
- **Partial dependence** must marginalize using CatBoost's exact grid/quantization and averaging convention.

**Why it happens:**
fstr methods "look like" standard model-interpretation formulas, so developers implement the textbook version rather than CatBoost's exact one. SHAP + basic fstr already shipped, creating false confidence that the rest is a small delta.

**How to avoid:**
- Oracle each fstr type independently against CatBoost's `get_feature_importance(type=...)` output at ≤10⁻⁵, on models **with categorical/CTR features** (where the accounting is hardest), not just numeric fixtures.
- Read the exact upstream definition in `CATBOOST_CORE_DESIGN.md` / vendored source for each type before implementing; do not substitute a generic interpretation-library formula.

**Warning signs:** fstr matches on numeric-only models but diverges with CTR/combinations; LossFunctionChange sign/magnitude off when the eval dataset differs from train.

**Phase to address:** Extended feature-importance phase.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use an off-the-shelf HNSW crate (hnswlib/instant-distance) for KNN | Fast to wire, well-tested | **Never matches upstream's RNG/graph → FEAT-07 bit-exact parity permanently unreachable** | Never (for parity). Only if KNN parity is explicitly dropped from scope. |
| Hold exported ONNX/CoreML models to the ≤10⁻⁵ double bar | One tolerance everywhere | Chasing phantom "failures" from float32 drift; risk of loosening the global bar | Never — define an export-specific tolerance instead |
| Route multiclass/non-symmetric/CTR models through the new GPU evaluator | "More features on GPU" | Diverges from CatBoost (which CPU-falls-back) → parity failures | Never — mirror upstream's oblivious/1-dim/float-only restriction + CPU fallback |
| Raw float `TAtomicAdd` in the GPU evaluator reduction | Simplest transcription of §7.1 | Non-deterministic, breaks ε=1e-4 over many trees | Never — reuse v1.1 fixed-point u64 reduction |
| Seed a `#[cube]` reduction with `-inf` | Faithful to upstream `NegativeInfty()` | HIP JIT reject on gfx1100, invisible to cpu/wgpu check | Never in kernel code — use `f32::MIN` sentinel |
| Snapshot only the trees built so far | Small serialized state | Non-reproducible resume with sampling/ordered boosting | Never if reproducible resume is promised |
| Benchmark vs the host-light CPU baseline | Reuses impressive v1.1 numbers | Misrepresents competitiveness vs official CatBoost | Only for internal regression tracking, clearly labeled |
| Validate CoreML parity structurally (no execution) | Works without macOS runtime | Doesn't prove runtime faithfulness | Acceptable if no Apple runtime in-env AND the gap is documented (like Kaggle-CUDA discipline) |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| ONNX Runtime (checker) | Testing against `.cbm` double predictions at 10⁻⁵ | Oracle = official CatBoost's own ONNX in the same ORT; export-specific tolerance; RawFormulaVal first |
| ONNX-ML TreeEnsemble | Exporting categorical/CTR/text models | Reject with typed error (upstream does); float-only models only |
| onnxruntime binary classifier | Asserting on the inferred label | Ignore label for binary classification; assert on probability |
| CoreML / coremltools | Assuming Linux-CI execution parity | Validate on Apple runtime or do structural parity + document gap |
| Kaggle CUDA (GPU oracle) | Signing off GPU inference on in-env ROCm | ROCm = non-gating smoke; authoritative correctness+speed on Kaggle CUDA |
| `python3.13t` free-threaded | Advertising free-threaded support untested | Gate the claim behind an actual `t`-interpreter run |
| rocm wheel runtime | Bundling patchelf-renamed `libhiprtc` | `ROCM_PATH` + `LD_PRELOAD` system libs; never bundle the renamed lib (segfaults HIP JIT) |
| CatBoost `cv()` | Using sklearn KFold semantics | Match `type`/stratification-default/group-awareness exactly; oracle the fold assignment |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| GPU evaluator per-call model re-upload | GPU predict slower than CPU on small batches | Build `TGPUModelData` **once** (upstream does in ctor), cache device buffers across calls (like v1.1 `GpuTrainSession`) | Any repeated predict / online serving |
| GPU predict on tiny batches | Kernel-launch overhead dominates | Batch documents; keep quantized buffer cached; fall back to CPU below a batch-size threshold | Single-row / low-latency inference |
| Deterministic fixed-point reduction overhead | GPU predict marginally slower than a raw-atomic version | Accept it — determinism is required for parity; optimize the fixed-point conversion, not by removing it | N/A (correctness constraint) |
| Brute-force-exact KNN as "the fast path" | O(n²) apply on large embedding sets | The HNSW port is also the perf answer (approximate = sub-linear search); but correctness-first | Large embedding datasets |
| Benchmark timing includes JIT warmup | First run wildly slower | Warm up kernels before timed region; report medians | Every GPU benchmark |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Loading untrusted `.cbm`/`.onnx`/`.coreml` models without validation | Malformed model → panic/UB in deserializer | Validate structure on load; return typed errors, never `unwrap()` (project rule); fuzz the deserializers |
| Snapshot files trusted across versions | Resuming from a tampered/incompatible snapshot corrupts training | Version + validate snapshot; reject incompatible params with a typed error |
| Data leakage in CV/tuning (integrity, not confidentiality) | Overstated model quality shipped to users | Per-fold border/CTR computation; leakage canary test |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| ONNX/CoreML export silently succeeds on categorical models | User ships a wrong model, discovers it in production | Reject at export with a clear message naming the offending feature type (mirror CatBoost) |
| Wrong per-backend wheel installed | ImportError/segfault, "it doesn't work" | Clear per-backend package naming/docs; CPU-default; smoke test on install |
| GPU predict diverges from CPU predict without explanation | User distrusts the whole library | Document the ε=1e-4 GPU bar and the CPU-fallback conditions explicitly |
| CV `type`/stratification defaults differ from CatBoost silently | Migrating users get different CV numbers than CatBoost | Match defaults exactly; document any deviation |

## "Looks Done But Isn't" Checklist

- [ ] **ONNX export:** Often missing the **categorical/text/embedding rejection** — verify export throws a typed error on a CTR model, and that a numeric model loads+scores in ONNX Runtime within the export tolerance (RawFormulaVal + Probability, binary label ignored).
- [ ] **CoreML export:** Often missing **execution validation** — verify it either runs on an Apple runtime OR has documented structural parity; confirm float32 tolerance is used, not the double bar.
- [ ] **GPU inference evaluator:** Often missing **determinism across repeated applies** and the **oblivious/1-dim/no-cat guards** — verify twice-run predictions are identical, and that a multiclass/CatFeature model falls back to CPU (matches CatBoost), and that the rocm suite passes in-env (no `-inf`/`|=` HIP reject).
- [ ] **Online-HNSW (FEAT-07):** Often missing the **per-object neighbor-set oracle** — verify the returned neighbor indices match upstream index-for-index over the shuffled prefix (not just the final XOR prediction), and the class-vote order matches.
- [ ] **CV/tuning:** Often missing **fold-assignment parity** and the **leakage canary** — verify object→fold matches CatBoost for a fixed seed across all three `type`s, groups stay intact, and a target-permuted feature scores ~chance.
- [ ] **Snapshot/resume:** Often missing **RNG/ordered-boosting state** — verify snapshot-then-resume is bit-identical to a straight run with sampling AND ordered boosting enabled.
- [ ] **PyPI release:** Often missing **clean-env install smoke per backend** and the **free-threaded run** — verify each wheel imports+predicts in a fresh env; verify (or explicitly defer with evidence) the `python3.13t` concurrency claim.
- [ ] **Benchmark:** Often missing the **vs-official-CatBoost baseline** — verify speed/accuracy are vs official CatBoost (matched version/hardware/params), GPU on Kaggle CUDA, not vs the host-light baseline.
- [ ] **Extended fstr:** Often missing **categorical/CTR-model oracles** — verify Interaction/LossFunctionChange/partial-dependence match `get_feature_importance` on models *with* CTR features, not just numeric.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Scoped ONNX/CoreML for categorical models | LOW | Reframe SPEC to float-only + typed rejection; delete the impossible path |
| Held export to the ≤10⁻⁵ double bar | LOW | Re-baseline oracle to CatBoost's own ONNX/CoreML in the same runtime; set export tolerance |
| Used off-the-shelf HNSW crate | HIGH | Rip out; port `online_hnsw` bit-for-bit with neighbor-set oracle (the only path to parity) |
| GPU evaluator with raw atomic reduction | MEDIUM | Swap in v1.1 fixed-point u64 reduction; re-validate ε=1e-4 on Kaggle CUDA |
| `-inf` in a `#[cube]` kernel | LOW | Replace with `f32::MIN` sentinel; re-run rocm suite in-env |
| Snapshot missing RNG state | MEDIUM | Extend snapshot format (versioned); add straight-vs-resume bit-identical test |
| Leaky CV/tuning | MEDIUM | Move border/CTR computation inside folds; add leakage canary; re-run benchmarks |
| Benchmark vs wrong baseline | LOW | Re-run against official CatBoost on matched hardware/params; relabel numbers |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Categorical ONNX export | Model export (ONNX) | Typed rejection on CTR model; numeric model scores in ORT |
| 2. Export float32 drift vs ≤10⁻⁵ | Model export (ONNX) | Export tolerance defined; oracle = CatBoost's own ONNX in ORT |
| 3. Opset / prediction-type / binary label | Model export (ONNX) | Pinned opset; RawFormulaVal + Probability cases; label ignored for binary |
| 4. CoreML determinism / float32 / ops | Model export (CoreML) | Float32 tolerance; execution-on-Apple or documented structural parity |
| 5. GPU reduction non-determinism | GPU inference evaluator | Twice-run predictions identical; ε=1e-4 vs CPU + Kaggle CUDA |
| 6. `-inf`/`\|=` HIP codegen traps | GPU inference evaluator | rocm suite passes in-env; `f32::MIN` sentinel; `+=`/`<<` leaf index |
| 7. Exceeding upstream GPU subset | GPU inference evaluator | Multiclass/CatFeature/non-sym → CPU fallback (matches CatBoost) |
| 8. HNSW "approximate = infeasible" | FEAT-07 online-HNSW port | Per-object neighbor-set matches upstream index-for-index; XOR ≤10⁻⁵ |
| 9. CV fold divergence / leakage | Orchestration (CV/tuning) | Fold assignment matches CatBoost per seed; leakage canary ~chance |
| 10. Non-reproducible resume | Orchestration (snapshot/resume) | Straight-vs-resume bit-identical with sampling + ordered boosting |
| 11. Wheel/abi3/free-threaded | Adoption/DX (PyPI) | Clean-env per-backend install smoke; free-threaded run or documented defer |
| 12. Benchmark baseline confusion | Adoption/DX (benchmark) | Vs official CatBoost, matched version/hardware; GPU on Kaggle CUDA |
| 13. Extended fstr divergence | Extended feature-importance | Matches `get_feature_importance` on CTR models at ≤10⁻⁵ |

## Sources

- CatBoost — ONNX export only supports datasets without categorical features; text/embedding "likely never supported": [ONNX | CatBoost docs](https://catboost.ai/docs/en/concepts/apply-onnx-ml); [ONNX export doesn't support categorical features · Issue #863](https://github.com/catboost/catboost/issues/863) (HIGH)
- CatBoost `save_model` ONNX supported types (binary/multiclass/regression) + `export_parameters` + onnxruntime binary-label bug: [save_model | CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_save_model); [sklearn-onnx CatBoost tutorial](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_catboost.html) (HIGH)
- CoreML export — datasets without categorical features only; `prediction_type` raw/probability: [Export a model to CoreML | CatBoost](https://catboost.ai/docs/en/features/export-model-to-core-ml); [CoreML | CatBoost](https://catboost.ai/docs/en/concepts/export-coreml) (HIGH)
- CatBoost `cv()` — `type` {Classical/Inverted/TimeSeries}, stratification default by loss, group-in-fold, snapshot params: [cv | CatBoost](https://catboost.ai/docs/en/concepts/python-reference_cv); [Cross-validation | CatBoost](https://catboost.ai/docs/en/concepts/cli-reference_cross-validation) (HIGH)
- GPU inference evaluator internals (oblivious/1-dim/no-cat restriction, double4 TAtomicAdd, unimplemented prediction types, MLTOOLS-6839 `|=`→`+=` codegen bug, warp-interleaved layout, `FeatureVal`=SplitIdx verbatim): `docs/CATBOOST_CUDA_KERNELS_DESIGN.md` §7.1 (in-repo, HIGH)
- HNSW root-cause + feasibility (upstream `NOnlineHnsw::TOnlineHnswDenseVectorIndex`, instrumented-trainer neighbor-set proof, ~936 LOC port scope, disproven hypotheses): project memory `knn-estimated-feature-is-online-hnsw.md`; note `phase65-text-embedding-outcome` (HIGH)
- GPU determinism + HIP `-inf`/`f32::MIN` landmine + fixed-point u64 reduction + rocm-only in-env validation: project memory `cubecl-hip-no-inf-literal.md`, `phase10-reduce-determinism-spike`, `phase76-gpu-tolerance-signoff-outcome`; `.planning/notes/gpu-training-host-light-root-cause.md` (HIGH)
- PyO3 abi3-py312 / `gil_used=false` / free-threaded UAT gap / rocm wheel `LD_PRELOAD` requirement: project memory `phase8-python-bindings-outcome.md` (HIGH)
- v1.1 speedup baseline caveat (23.9–42.1× vs host-light CPU baseline, NOT vs official CatBoost) + standing debt GPUT-14/BENCH-02: `.planning/PROJECT.md` Current State (HIGH)

---
*Pitfalls research for: catboost-rs v1.2 Parity Completion & Release Readiness*
*Researched: 2026-07-05*
