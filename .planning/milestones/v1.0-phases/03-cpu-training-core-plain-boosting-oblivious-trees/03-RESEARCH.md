# Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees - Research

**Researched:** 2026-06-13
**Domain:** Gradient-boosting training core — symmetric (oblivious) tree growth, split scoring, leaf estimation, bootstrap/sampling, overfitting detection, auto-LR; realized over a generic `R: Runtime`/`F: Float` compute boundary (`cb-compute`) with a CubeCL CPU backend (`cb-backend`) and a host-side ordered-reduction parity discipline.
**Confidence:** HIGH for upstream math/formulas (read directly from vendored `catboost-master/` 1.2.10 source); MEDIUM for CubeCL CPU-runtime wiring (manual + cargo-search confirmed, no build run this session); HIGH for the Python oracle schema (verified against a committed `model.json`).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** The CubeCL CPU runtime stands up in Phase 3 (not a plain-Rust placeholder). The `cpu` backend uses CubeCL's CPU runtime now, proving the CubeCL seam this phase. CubeCL enters the critical parity path now — D-02/D-03 contain the parity risk.
- **D-02:** Kernels do order-independent work; **every parity-critical SUM is finalized host-side via `cb-core::sum_f64`**. CubeCL kernels compute only order-independent per-element work (per-object gradient/hessian, per-bin scatter). All parity-critical reductions (histogram bin totals, leaf values, weight/border stats) are finalized host-side in the frozen sequential order. Kernels never do their own internal float reduction.
- **D-03:** `#[cube]` kernels + the `cubecl` dependency live **ONLY in `cb-backend`**; `cb-compute` stays abstract. `cb-compute` defines its own abstract `R: Runtime`/`F: Float` traits + host-side boosting orchestration and never depends on `cubecl`. `cb-backend` holds the `#[cube]` kernels and implements `cb-compute`'s traits.
- **D-04:** Coarse domain-level trait ops — `R: Runtime` exposes ML-level operations (`build_histograms`, `compute_gradients`, `eval_splits`); the backend owns internal kernel decomposition while `cb-compute`'s host orchestration finalizes ordered sums.
- **D-05:** Per-bin histogram accumulation is an **ordered single-pass over f64 accumulators in canonical object order** (`cb-core::sum_f64` semantics), bit-matching upstream's `thread_count=1` accumulation. The kernel computes per-object gradients in parallel; the histogram **reduction** is the ordered host-side pass.
- **D-06:** Parallel elementwise kernels now, ordered reductions host-side. Determinism comes exclusively from host-side ordered reductions; all summation routes through `cb-core::sum_f64` — never a parallel float-add into shared accumulators.
- **D-07:** The first slice locks against **simplified isolating params** (`bootstrap_type=No`, `random_strength=0`, fixed `l2_leaf_reg`, fixed `depth`/`learning_rate`/`iterations`) so any divergence can only be the tree/leaf math. Each additional knob is added as its OWN additive oracle.
- **D-08:** The first slice covers **BOTH RMSE (regression) and Logloss (binary classification)**.
- **D-09:** **All four leaf-estimation methods (Gradient, Newton, Exact, Simple)** are implemented up front (TRAIN-03 in one wave).
- **D-10:** Additive widening order after the first slice: sampling/bootstrap (TRAIN-04) → full regularization `random_strength`/`bagging_temperature` (TRAIN-05) → overfitting detection / early stopping (TRAIN-06) → per-iteration eval-set metric logging (TRAIN-07) → automatic learning-rate selection (TRAIN-08). Each gets its own oracle.
- **D-11:** **Python-reachable oracle floor — no C++ instrumentation in Phase 3.** Assert per-tree splits (feature+border), per-tree leaf values, and per-iteration approximants from pinned `catboost==1.2.10` (`save_model(format='json')` + `staged_predict`). Internal histograms/per-object gradients are NOT extracted. Drift is localized by the simplified-isolating-params first slice.

### Claude's Discretion (parity-dictated — research reads upstream and reproduces)
- Exact split-score / gain formula and the oblivious (symmetric) tree growth procedure — `greedy_tensor_search.cpp`, `score_calcers.cpp`.
- Exact leaf-value math for Gradient/Newton/Exact/Simple (incl. `leaf_estimation_iterations`, l2 application) — `approx_calcer.*`, `online_predictor.*`, `hessian.*`.
- Per-loss gradient/hessian for RMSE and Logloss — `error_functions.*`, `ders_holder.h`.
- Split tie-break determinism for equal-gain splits — known landmine; MUST be pinned.
- Bootstrap/sampling draw sequences seeded by `TFastRng64` + exact draw ORDER — `tensor_search_helpers.cpp`, `bootstrap_options.*`.
- Overfitting-detector statistics (Wilcoxon/IncToDec/Iter, `od_pval`/`od_wait`) + `use_best_model` — `catboost/libs/overfitting_detector/`.
- Auto learning-rate coefficient tables/formula keyed by dataset size / target / `use_best_model` / boost-from-average — `options_helper.cpp`.
- Concrete `cubecl` crate version — latest stable per CLAUDE.md.

### Deferred Ideas (OUT OF SCOPE)
- Parallel/multi-threaded performance tuning (parity, not perf, is judged; ordered-reduce seam established now, perf later).
- C++ instrumentation of `catboost-master` for iteration-internal oracle values (per-bin histograms, per-object gradients) — deferred to Phase 5 unless a divergence cannot be localized at tree granularity.
- GPU kernels (`wgpu`/`cuda`/`rocm`) — Phase 7, additively on this seam.
- Ordered boosting / ordered CTR / categoricals (Phase 5); serialization / SHAP / Builder API (Phase 4).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRAIN-01 | Plain GB train loop (`iterations`, `learning_rate`, `depth`) | Boosting driver flow (`train_model.cpp`), per-iteration approx update, `learning_rate` scaling of leaf deltas, starting-approx/`boost_from_average` (§Boosting Loop). |
| TRAIN-02 | Symmetric (oblivious) decision trees | `GreedyTensorSearchOblivious` — one split per depth level, applied across the whole level; split score = ΣAddLeafPlain over buckets; tie-break `gain > bestGain` strict (§Tree Growth, §Split Scoring, Pitfall 1). |
| TRAIN-03 | Leaf estimation — Gradient, Newton, Exact, Simple | `CalcMethodDelta`/`CalcLeafDeltasSimple`, `CalcDeltaNewtonBody`, `CalcAverage`, `CalcExactLeafDeltas`; `leaf_estimation_iterations` gradient loop (§Leaf Estimation). "Simple" maps to `ELeavesEstimation` choices via score function; see §Leaf Estimation note. |
| TRAIN-04 | Bootstrap/sampling — Poisson, Bayesian, Bernoulli, MVS, No; `subsample`; object/group units | `Bootstrap()` dispatch + per-block reseed draw order (§Bootstrap & Sampling, Pitfall 4). |
| TRAIN-05 | Regularization — `l2_leaf_reg`, `random_strength`, `bagging_temperature` | `ScaleL2Reg` (l2 scaling), `TRandomScore::GetInstance` + Box-Muller normal draw, Bayesian weight `powf(-log(u+1e-100), temp)` (§Regularization, Pitfall 3). |
| TRAIN-06 | Overfitting detection + early stopping | `overfitting_detector.cpp` (Wilcoxon / IncToDec / Iter), `use_best_model` (§Overfitting Detection). |
| TRAIN-07 | Eval-set validation metrics per iteration | Per-iteration metric logging shape; `eval_metric` default = objective; multiple eval sets (§Eval-Set Metric Logging). |
| TRAIN-08 | Automatic learning-rate selection | `TAutoLRParamsGuesser` coefficient table + formula, keyed by target/task/`use_best_model`/`boost_from_average` (§Auto Learning-Rate). |
</phase_requirements>

## Summary

The Phase-3 training core is a faithful port of CatBoost 1.2.10's CPU plain-boosting path. The whole algorithm is small in surface but exacting in numeric detail: every formula was read directly from the vendored C++ and is reproduced below with `double`-accumulator semantics. The defining structure is the **oblivious (symmetric) tree** — `GreedyTensorSearchOblivious` selects exactly **one** split per depth level and applies it across all nodes of that level, so a depth-`d` tree has `2^d` leaves and `d` splits. Split scoring uses per-bucket `TBucketStats` summed into a score; the leaf delta for the chosen split uses `CalcAverage`/`CalcDeltaNewton` with an L2 term scaled by `l2 * (sumWeight/docCount)`. Per-loss gradients are trivial: RMSE der1 = `target − approx`, der2 = `−1`; Logloss/CrossEntropy der1 = `target − p`, der2 = `−p(1−p)` where `p = sigmoid(approx)`.

The parity risk lives in three places, all of which are RNG- or reduction-order-sensitive: (1) the **split tie-break** is a strict `gain > bestGain` over a fixed candidate-iteration order — the first candidate achieving the max wins (Phase-2 WR-01 redux); (2) the **random-strength** perturbation draws a Box-Muller normal from `TFastRng64` per candidate via `TRandomScore::GetInstance`, consuming a variable number of `GenRandReal1()` draws (rejection loop) — exact draw order and the `GenRandReal1(ui64)` mantissa formula must match; (3) the **bootstrap** path re-seeds a fresh `TRestorableFastRng64(randSeed + blockIdx)` and `Advance(10)` per 1000-element block — the block structure is part of the contract even at `thread_count=1`. The D-07 simplified-isolating first slice (`bootstrap_type=No`, `random_strength=0`) deliberately sidesteps (2) and (3) so the first oracle isolates the deterministic tree/leaf math.

The compute boundary (D-01/D-02/D-03) is realized with CubeCL 0.10.0's `CpuRuntime` behind the `cpu` feature in `cb-backend`. Kernels do only order-independent elementwise work (per-object gradient/hessian, per-bin scatter); every parity-critical sum is finalized host-side through the existing `cb-core::sum_f64` in canonical object order. `cb-compute` stays `cubecl`-free with its own abstract `R: Runtime`/`F: Float` traits.

**Primary recommendation:** Build the thinnest end-to-end slice first — RMSE + Logloss, `bootstrap_type=No`, `random_strength=0`, fixed `depth`/`l2_leaf_reg`/`learning_rate`/`iterations`, Gradient leaf estimation, `boost_from_average` explicitly fixed — oracle-locked on splits + leaf_values + staged approximants from the existing Python generator. Stand up the `R: Runtime` trait + CubeCL `CpuRuntime` gradient/histogram kernels in that same slice. Then add: the other three leaf methods (D-09), then sampling/bootstrap, then `random_strength`/`bagging_temperature`, then overfitting detection, then eval logging, then auto-LR — each its own additive oracle (D-10).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-object gradient/hessian compute | `cb-backend` (CubeCL `#[cube]` kernel) | `cb-compute` (trait surface) | Order-independent elementwise (D-02/D-06); the kernel's natural home. |
| Per-bin histogram scatter | `cb-backend` (kernel) | — | Order-independent scatter; the **reduction** is NOT here. |
| Histogram bin-total reduction | `cb-compute` host orchestration | `cb-core::sum_f64` | Parity-critical ordered sum (D-02/D-05); MUST be host-side sequential. |
| Split score / gain computation | `cb-compute` host orchestration | — | Uses reduced bucket stats; ordered (`AddLeafPlain` accumulation). |
| Split selection / tie-break | `cb-compute` (`cb-train`) | — | Strict `gain > bestGain` over fixed candidate order (Pitfall 1). |
| Leaf-delta math (Gradient/Newton/Exact/Simple) | `cb-compute`/`cb-train` host | `cb-core::sum_f64` | Reductions over leaf members; ordered. |
| Boosting loop / approx update | `cb-train` | `cb-compute` | Orchestration; per-iteration state. |
| Bootstrap / sampling draws | `cb-train` | `cb-core::TFastRng64` | RNG-sequenced; host-only, deterministic. |
| Overfitting detection / early stop | `cb-train` | — | Pure host state machine over eval metrics. |
| Auto-LR / option defaults | `cb-train` (or `cb-compute` params module) | — | Pure scalar formula; pre-train. |
| Eval-set metric logging | `cb-train` | `cb-core::sum_f64` | Metric accumulation routes through the reduction primitive. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 | CPU compute runtime (`CpuRuntime`) + `#[cube]` kernels in `cb-backend` only (D-01/D-03). | Latest stable [VERIFIED: cargo search, 2026-06-13]; mandated by AGENTS.md/CLAUDE.md; matches the Phase-7 roadmap pin. Manual shows `features = ["cpu"]` for `CpuRuntime` [CITED: cubecl_manual/Cubecl_generics.md]. |
| `cb-core` | local | `sum_f64`/`sum_f32_in_f64` ordered reduction, `TFastRng64`, `CbError`. | Already complete; the process-wide ordered-sum and RNG invariant all training sums/draws route through (D-02/D-08, P2 D-07). [VERIFIED: crates/cb-core/src/reduction.rs, rng.rs] |
| `cb-data` | local | `QuantizedPool` typed-width SoA bins (`ColumnBins::{U8,U16,U32}`), borders, `NanMode`. | The histogram-kernel input (P2 D-11). [VERIFIED: crates/cb-data/src/quantized_pool.rs] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `bytemuck` | 1.x (latest) | `Pod` data transfer to/from CubeCL buffers (`Bytes::from_elems`). | Required by the CubeCL CPU runtime data-transfer pattern. [CITED: cubecl_manual/Cubecl_generics.md] |
| `cb-oracle` | local | `.npy` fixture read, `compare_stage(Stage::{Splits,LeafValues,StagedApprox,Predictions}, …)` at ≤1e-5. | Phase-3 adds expected-OUTPUT fixtures; harness already in place. [VERIFIED: crates/cb-oracle/src/compare.rs] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| CubeCL `CpuRuntime` kernels for the elementwise work | Plain-Rust loops (no CubeCL this phase) | Rejected by D-01 — the user deliberately stands up the CubeCL seam now to de-risk Phase 7. Plain Rust would defer the integration risk. |
| Per-candidate normal draw via `TFastRng64` Box-Muller | A faster/cleaner normal sampler | Rejected — parity requires the **exact** Marsaglia-polar draw sequence (Pitfall 3). No substitution permitted. |
| Parallel partial-histogram + merge | host-ordered single-pass reduction | Rejected by D-05 — merge order would have to reproduce upstream's; higher drift risk. |

**Installation:**
```bash
# Workspace deps (root Cargo.toml [workspace.dependencies]); cb-backend only:
cubecl = { version = "0.10.0", features = ["cpu"] }
bytemuck = { version = "1", features = ["extern_crate_std"] }
```

**Version verification:** `cargo search cubecl` → `cubecl = "0.10.0"` (2026-06-13). Offline this session for crates.io API; `cargo search` is the authoritative local registry index. Confirm `bytemuck` latest at plan time via `cargo search bytemuck`.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `cubecl` | crates.io | est. ~2 yr (Tracel-AI/Burn ecosystem) | high (Burn dependency) | github.com/tracel-ai/cubecl | OK | Approved — already roadmap-pinned, manual-documented, `cargo search` confirms 0.10.0. |
| `bytemuck` | crates.io | mature (>5 yr) | very high | github.com/Lokathor/bytemuck | OK | Approved — ubiquitous `Pod`/cast crate; required by CubeCL data transfer. |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

> Note: `gsd-tools query package-legitimacy check` was unavailable and crates.io API was offline this session. `cubecl 0.10.0` is confirmed via `cargo search` (local registry index) AND cited in the project CubeCL manual; `bytemuck` is cited in the same manual. If the planner can reach the network, run `gsd-tools query package-legitimacy check --ecosystem crates cubecl bytemuck` to upgrade these to a tool-confirmed verdict before the install task.

## Architecture Patterns

### System Architecture Diagram

```text
                 ┌─────────────────────────────────────────────────────────────┐
   QuantizedPool │ cb-train: PLAIN BOOSTING DRIVER (host, deterministic)         │
   (SoA bins) ──▶│                                                               │
   target,weights│  init approx ← StartingApprox (boost_from_average ? mean/logit│
                 │                                : 0)                            │
                 │  for iter in 0..iterations:                                   │
                 │    ┌──────────────────────────────────────────────────────┐  │
                 │    │ 1. compute_gradients(approx, target) ── der1,der2      │  │ ← cb-backend kernel (elementwise, D-02)
                 │    │ 2. Bootstrap()  ── SampleWeights via TFastRng64        │  │ ← host RNG (D-07: No → all 1.0)
                 │    │ 3. GreedyTensorSearchOblivious:                        │  │
                 │    │      for depth in 0..depth:                            │  │
                 │    │        build_histograms(bins, der) ───────────────────┼──┼─▶ cb-backend scatter kernel
                 │    │        bucket-total reduction ────────────────────────┼──┼─▶ cb-compute host sum_f64 (ORDERED, D-05)
                 │    │        CalcScores (AddLeafPlain per bucket)            │  │
                 │    │        score += GetInstance(rand)  (random_strength)   │  │ ← host RNG (D-07: 0 → no draw)
                 │    │        SelectBestCandidate: gain > bestGain (STRICT)   │  │ ← tie-break landmine (Pitfall 1)
                 │    │        AddSplit(bestSplit); partition objects→leaves   │  │
                 │    │ 4. CalcLeafValues (Gradient|Newton|Exact|Simple):      │  │
                 │    │      leafDers reduce (sum_f64) → CalcMethodDelta        │  │ ← cb-compute host (ORDERED)
                 │    │      × leaf_estimation_iterations                      │  │
                 │    │ 5. approx[i] += learning_rate * leafDelta[leaf(i)]     │  │
                 │    │ 6. eval-set metrics → OverfittingDetector.AddError     │  │ ← TRAIN-06/07
                 │    │    if IsNeedStop(): break                              │  │
                 │    └──────────────────────────────────────────────────────┘  │
                 │  model = oblivious_trees[] {splits, leaf_values} + bias        │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 ▼
                            ORACLE (Python catboost==1.2.10, frozen .npy)
            splits(feature+border) · leaf_values · staged approximants ── ≤1e-5
```

### Recommended Project Structure
```
crates/cb-compute/src/
├── lib.rs            # re-exports; NO cubecl dep (D-03)
├── runtime.rs        # abstract `Runtime` trait (build_histograms/compute_gradients/eval_splits) + `Float` (D-04)
├── histogram.rs      # host-side ordered bucket reduction (sum_f64), TBucketStats analogue
├── score.rs          # split score / gain (L2/Cosine AddLeafPlain), MINIMAL_SCORE
├── leaf.rs           # CalcMethodDelta Gradient/Newton/Exact/Simple, ScaleL2Reg, CalcAverage
├── loss.rs           # RMSE / Logloss(CrossEntropy) der1/der2, IsExpApprox, sigmoid
└── *_test.rs         # dedicated test files (no inline #[cfg(test)])
crates/cb-backend/src/
├── lib.rs            # SelectedRuntime = CpuRuntime under feature="cpu"
├── kernels.rs        # #[cube] gradient/hessian + histogram-scatter (generics-float)
└── cpu_runtime.rs    # impl cb-compute::Runtime for the CubeCL CPU runtime
crates/cb-train/src/
├── lib.rs
├── boosting.rs       # the iteration loop, approx update, learning_rate
├── tree.rs           # GreedyTensorSearchOblivious, SelectBestCandidate (tie-break)
├── bootstrap.rs      # Poisson/Bayesian/Bernoulli/MVS/No draw order (TFastRng64)
├── overfit.rs        # Wilcoxon/IncToDec/Iter detectors, use_best_model
├── autolr.rs         # TAutoLRParamsGuesser coefficient table + formula
└── *_test.rs
```

### Pattern 1: Oblivious (symmetric) tree growth — one split per level
**What:** A depth-`d` tree is `d` splits, each applied to EVERY node at its level; `2^d` leaves. The leaf index for an object is the `d`-bit number formed by its split outcomes.
**When to use:** Always — this is the only Phase-3 tree type.
**Example:**
```cpp
// Source: catboost/private/libs/algo/greedy_tensor_search.cpp:1189-1259
for (ui32 curDepth = 0; curDepth < MaxDepth; ++curDepth) {
    candidatesContexts = SelectFeaturesForScoring(...);
    CalcScores(data, currentSplitTree, scoreStDev, &candidatesContexts, fold, ctx);
    double bestScore = MINIMAL_SCORE;            // = std::numeric_limits<double>::lowest()
    const TCandidateInfo* bestSplitCandidate = nullptr;
    SelectBestCandidate(..., scoreBeforeSplit, &bestScore, &bestSplitCandidate);
    if (bestScore == MINIMAL_SCORE) break;
    scoreBeforeSplit = bestScore;
    const TSplit bestSplit = bestSplitCandidate->GetBestSplit(...);
    SetPermutedIndices(bestSplit, ...);          // re-partition objects across the level
    currentSplitTree.AddSplit(bestSplit);
}
```

### Pattern 2: Host-ordered reduction after an order-independent kernel (D-02/D-05)
**What:** Kernel writes per-object/per-bin contributions; host sums them in canonical order via `cb-core::sum_f64`.
**When to use:** Every parity-critical sum (histogram bin totals, leaf der/weight sums, metric accumulation).
**Example:**
```rust
// cb-backend kernel (order-independent scatter) — generics-float (AGENTS.md)
#[cube(launch)]
fn gradient_kernel<F: Float>(approx: &Array<F>, target: &Array<F>, der1: &mut Array<F>) {
    if ABSOLUTE_POS < approx.len() {
        der1[ABSOLUTE_POS] = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS]; // RMSE der1
    }
}
// cb-compute host: ordered finalize — NEVER sum inside the kernel
let bin_total = cb_core::sum_f64(&per_object_contributions_in_object_order);
```

### Pattern 3: Per-stage oracle lock per slice
**What:** Each MVP slice asserts `Stage::Splits`, `Stage::LeafValues`, `Stage::StagedApprox` against a frozen `.npy`/`model.json` fixture at ≤1e-5.
**When to use:** Every slice (D-10) — the gate is the slice's definition of done.

### Anti-Patterns to Avoid
- **Summing inside a CubeCL kernel for a parity-critical total** — violates D-02/D-05; breaks the 1e-5 gate non-deterministically. Kernels scatter; the host reduces.
- **`>=` tie-break in split selection** — upstream uses strict `>` (first-wins). Using `>=` flips ties and diverges (Pitfall 1).
- **Drawing the normal for `random_strength` with a different sampler** — must be the exact Box-Muller/`GenRandReal1` sequence (Pitfall 3).
- **Assuming `bootstrap_type=No` is the default** — CPU default for RMSE/Logloss object-sampling is **MVS** (subsample=0.8). The first slice MUST set `bootstrap_type=No` explicitly in the oracle params (Pitfall 4 / Assumption A1).
- **Forgetting `boost_from_average`** — defaults to TRUE for RMSE (false for Logloss); changes the starting approx / model bias (Pitfall 2).
- **Inline `#[cfg(test)]`** — CLAUDE.md/AGENTS.md forbid it; dedicated `*_test.rs` only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Float summation | A new accumulator / Kahan / `.sum()` | `cb-core::sum_f64` / `sum_f32_in_f64` | CI-grep banned (D-08); order is the parity contract. |
| PRNG | A new RNG or `rand` crate | `cb-core::TFastRng64` (+ `GenRandReal1` helper) | Bitstream-validated port; the draw stream is parity-critical. |
| Normal distribution | `rand_distr::Normal` | Port `StdNormalDistribution` (Box-Muller, Marsaglia polar) over `TFastRng64` | Upstream's exact rejection-loop draw sequence (Pitfall 3). |
| Wilcoxon p-value | A stats crate | Port `NStatistics::Wilcoxon` semantics | Must match upstream's exact statistic for `od_pval` parity. |
| Quantized bins | Re-binning | `cb-data::QuantizedPool` (P2) | Already oracle-locked; the kernel input. |
| GPU/CPU compute dispatch | Custom runtime abstraction in `cb-compute` | CubeCL `Runtime` in `cb-backend` behind `cb-compute`'s trait | D-01/D-03; Phase-7 attaches additively. |

**Key insight:** In this domain the "library" is upstream CatBoost itself — every formula must be transcribed, not reinvented. The only genuinely reusable engineering is the CubeCL runtime and the already-built `cb-core`/`cb-data` primitives.

## Runtime State Inventory

> Not a rename/refactor/migration phase. This is greenfield algorithm implementation filling three stub crates (`cb-compute`, `cb-backend`, `cb-train`). No stored data, live-service config, OS-registered state, secrets, or stale build artifacts are renamed or migrated.
>
> **Stored data:** None — Phase 3 produces an in-memory model; serialization is Phase 4.
> **Live service config:** None.
> **OS-registered state:** None.
> **Secrets/env vars:** None new (`DEBUG`/`OUT_DIR` belong to the vendored rust-package build, untouched).
> **Build artifacts:** Adding `cubecl`/`bytemuck` to `cb-backend` changes its build; no stale artifacts to clean (stub crate). Verified: `cb-backend/src/lib.rs` currently `SelectedRuntime = ()`.

## Common Pitfalls

### Pitfall 1: Split tie-break — strict `>` over a fixed candidate order
**What goes wrong:** Two candidate splits produce equal gain; the wrong one is chosen, and the whole tree (and every downstream tree) diverges.
**Why it happens:** `SelectBestCandidate` uses `if (gain > bestGain)` (strict) iterating candidates in a fixed nested order (`candidatesContexts` → `subList` → `candidate`). The **first** candidate reaching the max gain wins; later equal-gain candidates do NOT replace it. This is the Phase-2 WR-01 STL-tie-break situation again.
**How to avoid:** Reproduce upstream's candidate enumeration order EXACTLY (feature index ascending, bin/border ascending within feature) and use strict `>`. Do not sort by score; do not use `>=`.
**Warning signs:** First tree matches on a dataset with no ties but diverges on one with equal-gain features; divergence appears at a specific `split_index` in `model.json`.
**Source:** `greedy_tensor_search.cpp:948-966`.

### Pitfall 2: `boost_from_average` starting approx (RMSE only) → model bias
**What goes wrong:** RMSE predictions are off by a constant (the target mean); Logloss is fine, masking the bug.
**Why it happens:** `AdjustBoostFromAverageDefaultValue` sets `boost_from_average=true` by default for RMSE/MAE/Quantile/MAPE/MultiRMSE (NOT Logloss). The starting approx = `CalcOptimumConstApprox` (target mean for RMSE) and is stored in `model.json` `scale_and_bias = [1, [bias]]`. Verified in the committed fixture: `scale_and_bias: [1, [0.31519...]]`.
**How to avoid:** Implement `boost_from_average` and the starting-approx calc in the first slice; the staged approximants oracle already bakes it in. Pin `boost_from_average` explicitly in the oracle params so the slice and oracle agree.
**Source:** `options_helper.cpp:353-373`, `train_model.cpp:848-866`.

### Pitfall 3: `random_strength` normal draw — exact Box-Muller sequence from `TFastRng64`
**What goes wrong:** With `random_strength != 0`, split scores get a normal perturbation; a different RNG draw order picks different splits.
**Why it happens:** `score = candidate.BestScore.GetInstance(Rand)` adds `NormalDistribution<double>(rand, 0, StDev)`, which calls `StdNormalDistribution` — a **Marsaglia polar / Box-Muller rejection loop** that consumes a *variable* number of `GenRandReal1()` draws (`do { x=2u−1; y=2u−1; r=x²+y² } while (r>1 || r<=0)`), each `GenRandReal1()` = `(GenRand()>>11) * (1/9007199254740991.0)` for a `ui64` engine. The per-candidate `scoreStDev = random_strength * derivativesStDevFromZero * (NormalWithModelSizeDecrease ? mult : 1)`; the multiplier path is the default `random_score_type`.
**How to avoid:** Port `StdNormalDistribution` and `GenRandReal1(ui64)` exactly; draw per candidate in the same order `GetInstance` is called. **D-07's first slice sets `random_strength=0`, eliminating all normal draws** — defer this pitfall to the TRAIN-05 additive slice.
**Source:** `rand_score.h:GetInstance`, `util/random/normal.h:12-28`, `util/random/common_ops.h:19-21`, `greedy_tensor_search.cpp:855-867`.

### Pitfall 4: Bootstrap default is MVS, and draws are per-1000-element-block reseeded
**What goes wrong:** A default-params model can't be reproduced because the sampler is MVS (data/derivative-dependent), and even Bayesian/Bernoulli use a per-block RNG reseed.
**Why it happens:** (a) CPU default `bootstrap_type` for RMSE/Logloss object-sampling = **MVS**, `subsample` default 0.8 (`catboost_options.cpp:781-800`). (b) `GenerateRandomWeights` (Bayesian) does `randSeed = rand->GenRand()` once, then per block `TRestorableFastRng64 rand(randSeed + blockIdx); rand.Advance(10)` with block size 1000. (c) Bernoulli object subsample lives in `SetSampledControl` (`Control[i] = rand->GenRandReal1() < rate`). (d) Bayesian weight = `powf(-FastLogf(GenRandReal1()+1e-100), bagging_temperature)`.
**How to avoid:** First slice uses `bootstrap_type=No` (all weights 1.0, no draws). Implement MVS/Bayesian/Bernoulli/Poisson in the TRAIN-04 slice with the exact block-reseed order; MVS internal weights are NOT Python-observable (D-11) — validate MVS only end-to-end via splits/leaves, and consider it the highest sampling-parity risk.
**Source:** `tensor_search_helpers.cpp:322-560`, `catboost_options.cpp:779-800`, `calc_score_cache.cpp:1183-1198`, `mvs.cpp`.

### Pitfall 5: Leaf-value L2 scaling and the gradient-iterations loop
**What goes wrong:** Leaf values are off by a small factor; Newton vs Gradient confused.
**Why it happens:** Leaf delta = `sumDer / (sumWeight + scaledL2)` for Gradient, `sumDer / (−sumDer2 + scaledL2)` for Newton, where `scaledL2 = l2_leaf_reg * (sumAllWeights / allDocCount)` (`ScaleL2Reg`). `leaf_estimation_iterations` (default 1, auto-forced to 1 for <1000 iters and <20 features) runs the `leafUpdaterFunc`/`approxUpdaterFunc` loop; each iteration recomputes ders against the running approx delta.
**How to avoid:** Implement `ScaleL2Reg` and the gradient-iterations loop verbatim. For the first slice, the auto rule sets `leaf_estimation_iterations=1` (small dataset) — pin it explicitly in the oracle.
**Source:** `online_predictor.h:112-178`, `approx_calcer.cpp:482-525,712-800`, `options_helper.cpp:290-303`.

### Pitfall 6: Logloss `eval_metric`/objective uses exp-approx internally but JSON/staged are raw
**What goes wrong:** Sigmoid applied twice, or `IsExpApprox` mishandled.
**Why it happens:** `TCrossEntropyError::CalcDersRange` computes `e = exp(approx)` then `p = 1 − 1/(1+e) = sigmoid(approx)`; der1 = `target − p`, der2 = `−p(1−p)`. `IsExpApprox` controls whether the stored approx is already exponentiated. For CPU Logloss the staged approximants from `staged_predict` are RawFormulaVal (raw logits) — match those, not probabilities.
**How to avoid:** Implement der1/der2 from the raw approx via `sigmoid`; the staged oracle is raw logits. Probability prediction types are Phase 4.
**Source:** `error_functions.cpp:304-341`.

## Code Examples

### Per-loss gradient/hessian (the only two losses this phase)
```cpp
// Source: catboost/private/libs/algo_helpers/error_functions.h:391-402 (RMSE)
double TRMSEError::CalcDer (double approx, float target) const { return target - approx; } // der1
double TRMSEError::CalcDer2(double, float)               const { return -1.0; }           // der2
// weight applied after: ders[i].Der1 *= weights[i]; ders[i].Der2 *= weights[i];

// Source: catboost/private/libs/algo_helpers/error_functions.cpp:317-340 (Logloss/CrossEntropy)
const double e = exp(approx);            // (or approx itself if IsExpApprox)
const double p = 1 - 1 / (1 + e);        // = sigmoid(approx)
der1 = target - p;
der2 = -p * (1 - p);
der3 = -p * (1 - p) * (1 - 2 * p);       // only when calcThirdDer
```

### Leaf-delta math (Gradient / Newton) + L2 scaling
```cpp
// Source: catboost/private/libs/algo_helpers/online_predictor.h:112-178
inline double CalcAverage(double sumDelta, double count, double scaledL2) {
    double inv = count > 0 ? 1.0 / (count + scaledL2) : 0.0;
    return sumDelta * inv;
}
inline double ScaleL2Reg(float l2, double sumAllWeights, int allDocCount) {
    return l2 * (sumAllWeights / allDocCount);
}
// Gradient leaf delta: CalcAverage(SumDer, SumWeights, ScaleL2Reg(l2, sumAllW, docCount))
// Newton  leaf delta: SumDer / (-SumDer2 + ScaleL2Reg(l2, sumAllW, docCount))
//   == CalcDeltaNewtonBody(sumDer, sumDer2, l2, sumAllWeights, allDocCount)
```

### Split score accumulation (L2 score calcer — the parity-relevant CPU path)
```cpp
// Source: catboost/private/libs/algo/score_calcers.cpp:20-49 + online_predictor.h
// For each split bucket, AddLeafPlain folds left+right leaf contributions:
double rightAvrg = CalcAverage(rightStats.SumWeightedDelta, rightStats.SumWeight, L2Regularizer);
double leftAvrg  = CalcAverage(leftStats.SumWeightedDelta,  leftStats.SumWeight,  L2Regularizer);
Scores[splitIdx] += rightAvrg * rightStats.SumWeightedDelta;   // AddLeaf
Scores[splitIdx] += leftAvrg  * leftStats.SumWeightedDelta;
// Default score_function is Cosine (oblivious_tree_options.cpp:22), score = num/sqrt(den).
// L2ScoreCalcer is the simpler analogue; CONFIRM which the chosen score_function default
// uses for the first slice and pin score_function explicitly (Open Q1).
```

### Tie-break (strict, first-wins)
```cpp
// Source: greedy_tensor_search.cpp:948-966
double bestGain = -std::numeric_limits<double>::infinity();
for (candidatesContext : candidatesContexts)
  for (subList : candidatesContext.CandidateList)
    for (candidate : subList.Candidates) {
        double score = candidate.BestScore.GetInstance(Rand); // random_strength perturb
        double gain  = (score - scoreBeforeSplit) * featureWeight;
        if (gain > bestGain) { bestGain = gain; *bestSplitCandidate = &candidate; }
    }
```

### Overfitting detector (IncToDec — the default `type`)
```cpp
// Source: overfitting_detector.cpp:127-174  (default type = IncToDec, wait_iterations = 20, stop_pvalue = 0 → inactive)
// IsActive() iff Threshold > 0; IsNeedStop() iff !IsEmpty && CurrentPValue < Threshold.
// Iter detector == IncToDec with threshold forced to 1.0.
// Wilcoxon uses NStatistics::Wilcoxon over deltas after the local max.
```

### Auto learning-rate (TRAIN-08)
```cpp
// Source: options_helper.cpp:116-262
// Active only iff learning_rate, leaf_estimation_method, leaf_estimation_iterations, l2_leaf_reg ALL NotSet.
// key = (TargetType{RMSE|Logloss|MultiClass}, CPU, useBestModel, boostFromAverage)
// coeffs {A=DatasetSizeCoeff, B=DatasetSizeConst, C=IterCountCoeff, D=IterCountConst}
double custIter = exp(C*log(iterCount) + D);
double defIter  = exp(C*log(1000)     + D);
double defLR    = exp(A*log(learnObjectCount) + B);
lr = Round(min(defLR * custIter / defIter, 0.5), 6);
// Example CPU coeffs: RMSE,bestModel=F,bfa=T -> {0.158,-4.287,-0.813,2.571}
//                     Logloss,bestModel=F,bfa=F -> {0.427,-7.525,-0.917,2.63}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hand-rolled CPU compute loops | CubeCL `CpuRuntime` generic kernels | This phase (D-01) | Same kernels target GPU in Phase 7 with no `cb-compute` change. |
| `random_score_type=Normal` | `NormalWithModelSizeDecrease` (default) | upstream 1.2.x | Adds a model-length multiplier to `scoreStDev` — affects `random_strength` parity. |
| Default bootstrap Bayesian | **MVS** for CPU regression/binclf object-sampling | upstream | `subsample` default becomes 0.8; first-slice must override to `No`. |

**Deprecated/outdated:** Nothing relevant to Phase 3 — the vendored 1.2.10 source IS the spec.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | First-slice oracle must set `bootstrap_type=No`, `random_strength=0`, fixed `l2_leaf_reg`/`depth`/`learning_rate`/`iterations`, AND explicitly fix `boost_from_average` + `leaf_estimation_iterations` per loss; the default bootstrap (MVS) and default `boost_from_average` (true for RMSE) would otherwise leak into the lock. | Pitfall 2/4, Validation | High — a mismatch between slice params and oracle params makes the first slice unreproducible and mis-attributes the divergence. |
| A2 | The CPU default `score_function` is **Cosine** (`oblivious_tree_options.cpp:22`), so the first slice must either pin `score_function=L2` for the simplest math OR implement the Cosine calcer. | Code Examples, Open Q1 | Medium — using the wrong score calcer changes every split. |
| A3 | `cubecl 0.10.0`'s `CpuRuntime` (feature `cpu`) is build-compatible with the latest stable Rust toolchain and the deny-lint workspace policy, and `#[cube]` kernels using `generics-float` compile cleanly. Not built this session. | Standard Stack | Medium — if the CPU runtime needs extra features/toolchain, the `cb-backend` wave needs a spike (Open Q2). |
| A4 | `leaf_estimation_iterations` auto-forces to 1 for the first-slice dataset (<1000 iters, <20 features), and the existing `numeric_tiny` corpus satisfies this. | Pitfall 5 | Low — easily pinned explicitly in oracle params. |
| A5 | The staged_predict oracle for Logloss yields RawFormulaVal (raw logits), not probabilities, matching the internal approx. | Pitfall 6 | Medium — if `staged_predict` returns probabilities, the StagedApprox stage must transform before comparing. |
| A6 | "Simple" leaf estimation (TRAIN-03 term) corresponds to upstream's behavior where leaf delta = `CalcAverage` without Newton hessian — i.e. it maps to the Gradient/Exact family rather than a distinct fifth method; upstream `ELeavesEstimation` enum is `{Gradient, Newton, Exact, Simple}`. Need to confirm `Simple`'s exact formula. | Leaf Estimation, Open Q3 | Medium — one of the four required methods could be mis-implemented. |

**Note:** A1–A6 need confirmation in `/gsd-discuss-phase` or by extending the oracle generator and reading the `Simple`/`Exact` leaf paths before the relevant slice.

## Open Questions (RESOLVED)

> All four open questions are resolved below. Each resolution names the plan/task that owns the answer so executors can trace it. Resolutions reflect the locked first-slice params (D-07) and the Python-reachable oracle floor (D-11).

1. **Which `score_function` does the first slice pin?**
   - What we know: CPU default is Cosine; only Cosine and L2 are CPU-supported (`MakePointwiseScoreCalcer`). L2 is the simpler formula.
   - What's unclear: Whether locking against the *default* (Cosine) or pinning L2 gives the cleaner first slice.
   - Recommendation: Pin `score_function=L2` for the very first isolating slice (simplest score math), then add a Cosine-default slice. Implement both calcers (both are short).
   - **RESOLVED — `score_function=L2` for the first slice.** The first-slice oracle (Plan 00 Task 3) and the slice implementation (Plan 01) both pin `score_function='L2'` for the simplest, unambiguous split-score math. The Cosine-default calcer is implemented but locked in a later additive slice. This matches A2.

2. **CubeCL `CpuRuntime` build spike.**
   - What we know: cubecl 0.10.0, features `["cpu"]`, `bytemuck` for transfer; manual has a working `CpuRuntime` example.
   - What's unclear: Interaction with the workspace deny-lints (`indexing_slicing`, `panic`) inside generated `#[cube]` code, and whether `cb-backend` needs a test-lint exemption.
   - Recommendation: First `cb-backend` task is a minimal `#[cube]` gradient kernel that compiles + runs on `CpuRuntime`, BEFORE wiring it into the boosting loop. Read the CubeCL manual + error guideline first (AGENTS.md).
   - **RESOLVED — the CubeCL CpuRuntime build spike is Plan 00 Task 1.** Plan 00 Task 1 stands up the minimal `#[cube]` gradient kernel and proves it compiles + runs on `CpuRuntime` under the workspace deny-lints, before any boosting wiring (Plan 01). Any lint-exemption need surfaces there. This matches A3.

3. **Exact `Exact` and `Simple` leaf-estimation formulas (D-09).**
   - What we know: `CalcExactLeafDeltas` is called for `ELeavesEstimation::Exact` (`approx_calcer.cpp:734-744`); Gradient/Newton are in `CalcLeafDeltasSimple`.
   - What's unclear: The exact body of `CalcExactLeafDeltas` (quantile-style exact optimization) and how `Simple` differs.
   - Recommendation: Read `CalcExactLeafDeltas` and the `Simple` path fully in the TRAIN-03 wave (planner: add a focused read task) before implementing; the four methods land together (D-09) so all four must be pinned.
   - **RESOLVED — the Exact/Simple leaf bodies are read first in Plan 02 Task 1 (read-first) before implementation.** The TRAIN-03 leaf-methods wave (Plan 02) carries an explicit read-first of `CalcExactLeafDeltas` and the `Simple` path so all four methods (Gradient/Newton/Exact/Simple) are transcribed verbatim and oracle-locked together (D-09). This matches A6.

4. **`random_strength` normal-draw oracle observability.**
   - What we know: The draws affect split selection but the draw sequence isn't Python-observable.
   - What's unclear: Whether a `random_strength!=0` slice can be localized at tree granularity, or whether it needs the C++ instrumentation deferred to Phase 5.
   - Recommendation: Lock the TRAIN-05 slice end-to-end (splits+leaves) on a tiny dataset; if it diverges and can't be localized, escalate to instrumentation per D-11.
   - **RESOLVED — `random_strength != 0` is locked end-to-end at tree granularity.** The TRAIN-05 regularization slice locks `random_strength`/`bagging_temperature` end-to-end (splits + leaf_values + staged approximants) on a tiny dataset via the Python-reachable oracle (D-11). C++ instrumentation is escalated only if a divergence genuinely cannot be localized at tree granularity — deferred to Phase 5 by default (D-11 floor).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (stable) | all crates | ✓ (assumed; workspace builds) | latest stable | — |
| `cargo` registry index | `cubecl`/`bytemuck` resolution | ✓ (cargo search worked) | — | vendored offline cache if CI is air-gapped |
| `cubecl` 0.10.0 + `cpu` feature | `cb-backend` CubeCL CPU runtime | ✓ (in index) | 0.10.0 | plain-Rust path is FORBIDDEN by D-01 — no fallback |
| Python `catboost==1.2.10` + venv | oracle fixture generation (build-time only) | ✓ (`crates/cb-oracle/generator/.venv`) | 1.2.10 | none needed — fixtures are committed frozen; CI only reads `.npy` |
| C++ compiler (g++/clang++) | cityhash oracle ONLY (Phase 2) | ✓ | — | not needed in Phase 3 (no C++ build, D-11) |

**Missing dependencies with no fallback:** none identified (offline crates.io API this session is non-blocking — `cargo search` confirmed cubecl; a build spike is the Open-Q2 task).
**Missing dependencies with fallback:** none required.

## Validation Architecture

> Nyquist validation is ENABLED for this phase (`workflow.nyquist_validation: true`). Per-slice oracle locks are the sampling rate.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cb-oracle` per-stage comparator (`compare_stage`, ≤1e-5 abs) |
| Config file | none (cargo native); fixtures under `crates/cb-oracle/fixtures/` |
| Quick run command | `cargo test -p cb-train` (and `-p cb-compute -p cb-backend` for unit tests) |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01/02/03 (first slice) | RMSE+Logloss plain boosting, oblivious trees, Gradient leaf est — splits/leaf_values/staged match | oracle | `cargo test -p cb-train slice_first_oracle` | ❌ Wave 0 (fixture + test) |
| TRAIN-03 | Newton/Exact/Simple leaf values match | oracle | `cargo test -p cb-train leaf_methods_oracle` | ❌ Wave 0 |
| TRAIN-01 | per-loss der1/der2 unit (RMSE `t−a`,`−1`; Logloss `t−p`,`−p(1−p)`) | unit | `cargo test -p cb-compute loss::` | ❌ |
| TRAIN-02 | tie-break: strict `>` first-wins on equal-gain dataset | unit | `cargo test -p cb-train tree::tie_break` | ❌ |
| TRAIN-04 | Poisson/Bayesian/Bernoulli/MVS/No draw sequences reproduce upstream end-to-end | oracle | `cargo test -p cb-train bootstrap_oracle` | ❌ |
| TRAIN-05 | `random_strength`/`bagging_temperature`/`l2_leaf_reg` slices | oracle | `cargo test -p cb-train regularization_oracle` | ❌ |
| TRAIN-06 | Wilcoxon/IncToDec/Iter stop decision + `use_best_model` best-iter | unit+oracle | `cargo test -p cb-train overfit::` | ❌ |
| TRAIN-07 | per-iteration eval metric values (multiple eval sets) | oracle | `cargo test -p cb-train eval_metrics_oracle` | ❌ |
| TRAIN-08 | auto-LR formula matches `get_all_params()['learning_rate']` | unit | `cargo test -p cb-train autolr::` | ❌ |
| Boundary | CubeCL `CpuRuntime` gradient kernel runs | unit | `cargo test -p cb-backend kernels::gradient` | ❌ |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate-under-edit>` (quick).
- **Per wave merge:** `cargo test --workspace` (full).
- **Phase gate:** full workspace green + every slice's `compare_stage` oracle passing at ≤1e-5 before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] Extend `crates/cb-oracle/generator/gen_fixtures.py` to emit per-slice training oracles: `splits`/`leaf_values` from `model.json` (already produced for `regression_skeleton`), per-iteration `staged.npy`, and a **binclf_skeleton** (Logloss) scenario mirroring `regression_skeleton`. Pin `bootstrap_type=No`, `random_strength=0`, explicit `boost_from_average`, `leaf_estimation_iterations`, `score_function` (A1/A2).
- [ ] New oracle scenarios: `leaf_methods/{gradient,newton,exact,simple}`, `bootstrap/{poisson,bayesian,bernoulli,mvs,no}`, `regularization/{l2,random_strength,bagging_temp}`, `overfit/{wilcoxon,inctodec,iter,use_best_model}`, `eval_metrics`, `autolr`.
- [ ] A `model.json` parser in `cb-oracle` (or `cb-train` tests) extracting `oblivious_trees[i].splits` (float_feature_index, border) and `leaf_values` into `Vec<f64>` for `compare_stage(Stage::Splits/LeafValues, …)`.
- [ ] `cb-backend` build spike test: minimal `#[cube]` kernel on `CpuRuntime` (Open Q2) — must compile under deny-lints.
- [ ] Framework install: add `cubecl`/`bytemuck` to `[workspace.dependencies]` and `cb-backend`'s manifest.

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1`. This is a numeric ML training core with no network, auth, session, or external-input-trust surface; ASVS web categories are largely N/A. The relevant control is the existing PRNG misuse guard.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | Pool/QuantizedPool ingestion already validates (P2 D-06); Phase 3 consumes validated `QuantizedPool` + target/weights. Fallible APIs return `CbError`, never `unwrap` (CLAUDE.md). |
| V6 Cryptography | yes (negative) | `TFastRng64` is explicitly NON-cryptographic (documented in `rng.rs`). It must NEVER be used for secrets — only parity-reproducing sampling. No crypto is introduced this phase. |

### Known Threat Patterns for {Rust ML training core}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Panic via `unwrap`/indexing on degenerate input (empty leaf, count=0) | Denial of Service | Deny-lints (`unwrap_used`/`indexing_slicing`); `CalcAverage` already guards `count > 0`; return `CbError` not panic. |
| Integer/float overflow in wrapping RNG / `2^depth` leaf count | Tampering | `TFastRng64` uses `wrapping_*` (verified); cap `depth` (upstream `MaxDepth` ≤ 16) so `2^depth` fits. |
| Non-determinism breaking the oracle gate (a security-equivalent integrity property here) | Tampering | All sums via `cb-core::sum_f64`; `thread_count=1` equivalent ordered host reduction (D-02/D-05); CI-grep ban on raw float sum. |
| Mis-seeded sampler leaking unintended draw order | Information Disclosure (parity integrity) | Reproduce upstream's exact `randSeed + blockIdx` + `Advance(10)` block reseed (Pitfall 4). |

## Sources

### Primary (HIGH confidence)
- `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp` (vendored 1.2.10) — oblivious tree growth, `SelectBestCandidate` tie-break, `CalcScoreStDev`, `CalcScores`.
- `catboost-master/catboost/private/libs/algo/score_calcers.{h,cpp}` — L2/Cosine score calcers, `AddLeafPlain`.
- `catboost-master/catboost/private/libs/algo_helpers/online_predictor.h` — `CalcAverage`, `ScaleL2Reg`, `CalcDeltaGradient`, `CalcDeltaNewtonBody`, `TSum`.
- `catboost-master/catboost/private/libs/algo_helpers/error_functions.{h,cpp}` + `ders_holder.h` — RMSE/CrossEntropy der1/der2/der3, `IsExpApprox`.
- `catboost-master/catboost/private/libs/algo/approx_calcer.{h,cpp}` — `CalcLeafDeltasSimple`, gradient-iterations loop, `CalcExactLeafDeltas` dispatch.
- `catboost-master/catboost/private/libs/algo/tensor_search_helpers.cpp` — `Bootstrap`, `GenerateRandomWeights` (per-block reseed), Bayesian weight.
- `catboost-master/catboost/private/libs/algo/rand_score.h` — `TRandomScore::GetInstance` (normal/Gumbel).
- `catboost-master/util/random/normal.h`, `util/random/common_ops.h` — Box-Muller `StdNormalDistribution`, `GenRandReal1`/`ToRandReal1`.
- `catboost-master/catboost/libs/overfitting_detector/overfitting_detector.cpp` — Wilcoxon/IncToDec/Iter detectors.
- `catboost-master/catboost/libs/train_lib/options_helper.cpp` — `TAutoLRParamsGuesser` table+formula, `AdjustBoostFromAverageDefaultValue`, `UpdateLeavesEstimationIterations`.
- `catboost-master/catboost/private/libs/options/{boosting,oblivious_tree,overfitting_detector,bootstrap}_options.{h,cpp}`, `catboost_options.cpp` — defaults (depth=6, l2=3.0, lr=0.03, iterations=1000, leaf_estimation_method=Gradient, score_function=Cosine, bootstrap default MVS, OD type IncToDec/wait=20/pval=0).
- `catboost-master/catboost/libs/train_lib/train_model.cpp` — boosting driver, `CalcOptimumConstApprox` starting approx.
- `crates/cb-oracle/fixtures/regression_skeleton/model.json` — verified JSON schema: `oblivious_trees[].{splits[{border,float_feature_index,split_index,split_type}],leaf_values,leaf_weights}`, `scale_and_bias=[1,[bias]]`, `features_info.float_features[].borders`.
- `crates/cb-core/src/{reduction.rs,rng.rs}`, `crates/cb-data/src/quantized_pool.rs`, `crates/cb-oracle/src/compare.rs` — local primitives consumed.

### Secondary (MEDIUM confidence)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/{INDEX.md,Cubecl_generics.md}` — `CpuRuntime` usage, generic `#[cube]` kernel pattern, `cubecl = {version="0.10.0", features=["cpu"]}` + `bytemuck`.
- `cargo search cubecl` (2026-06-13) — `cubecl = "0.10.0"` latest stable.

### Tertiary (LOW confidence)
- None — all claims are sourced from vendored source or verified fixtures.

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM-HIGH — cubecl 0.10.0 confirmed via cargo search + manual, but no build run this session (Open Q2).
- Architecture / formulas: HIGH — every formula transcribed from vendored 1.2.10 source.
- Pitfalls: HIGH — tie-break, boost_from_average, MVS default, normal-draw sequence, L2 scaling all read directly from source and (for schema) verified against a committed fixture.

**Research date:** 2026-06-13
**Valid until:** 2026-07-13 (stable — vendored source is frozen at 1.2.10; only the cubecl crate version could drift, re-confirm at plan time).
