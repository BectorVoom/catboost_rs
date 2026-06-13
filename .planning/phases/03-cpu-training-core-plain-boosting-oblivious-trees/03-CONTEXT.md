# Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Realizes the **generic `R: Runtime` / `F: Float` compute boundary** (`cb-compute` + the `cpu` backend in `cb-backend`) and the **plain gradient-boosting training core** (`cb-train`) — so a user can train a plain-boosted model of symmetric oblivious trees on the CPU and have every per-tree split, leaf value, and per-iteration approximant match upstream CatBoost **1.2.10** to ≤1e-5. Deliverables (TRAIN-01 … TRAIN-08):

- **`cb-compute` generic boundary** — abstract `R: Runtime` / `F: Float` traits + host-side boosting orchestration; the `cpu` backend (`SelectedRuntime`) runs histogram / gradient / scan / reduction work.
- **Plain boosting loop** (`iterations`, `learning_rate`, `depth`) building **symmetric oblivious trees** with **leaf estimation** (Gradient, Newton, Exact, Simple), per-tree split + leaf-value intermediate oracles ≤1e-5.
- **Bootstrap / sampling** (Poisson, Bayesian, Bernoulli, MVS, No; `subsample`; object/group units) and **regularization** (`l2_leaf_reg`, `random_strength`, `bagging_temperature`) seeded by the Phase-1 `TFastRng64` port, reproducing upstream draws.
- **Overfitting detection / early stopping** (Wilcoxon/IncToDec/Iter, `od_pval`/`od_wait`, `use_best_model`) and **per-iteration eval-set metric logging** (multiple eval sets, `eval_metric`).
- **Automatic learning-rate selection** from dataset size, matching upstream; **first end-to-end CPU train→predict cycle**.

This phase is judged on parity + the establishment of the generic seam. It does NOT cover ordered boosting / ordered CTR / categoricals (Phase 5), serialization / SHAP / Builder API (Phase 4), or GPU backends (Phase 7).

</domain>

<decisions>
## Implementation Decisions

### Compute Boundary Design (TRAIN — success criterion 1)
- **D-01: CubeCL CPU runtime stands up in Phase 3 (not a plain-Rust placeholder).** The `cpu` backend uses CubeCL's CPU runtime now, proving the CubeCL seam in this phase rather than deferring it to Phase 7. This is a deliberate, more-ambitious reading of the Phase-1 discretion item that left "CubeCL CPU runtime vs plain CPU path" open. Consequence: CubeCL enters the critical parity path now — D-02/D-03 below contain the parity risk.
- **D-02: Kernels do order-independent work; every parity-critical SUM is finalized host-side via `cb-core::sum_f64`.** CubeCL kernels compute only order-independent per-element work (per-object gradient/hessian, per-bin scatter). All parity-critical reductions (histogram bin totals, leaf values, weight/border stats) are finalized **host-side** through `cb-core::sum_f64`/`sum_f32_in_f64` in the frozen sequential order. This preserves the Phase-2 D-07/D-08 reduction invariant exactly — kernels never do their own internal float reduction that could drift the 1e-5 bar.
- **D-03: `#[cube]` kernels + the `cubecl` dependency live ONLY in `cb-backend`; `cb-compute` stays abstract.** `cb-compute` defines its **own** abstract `R: Runtime` / `F: Float` traits and the host-side boosting orchestration (ordered reductions, leaf math) and never depends on `cubecl` — Phase-1 D-03 is honored literally. `cb-backend` holds the `#[cube]` kernels and implements `cb-compute`'s traits. Phase 7 adds GPU arms in the same place, additively.
- **D-04: Coarse domain-level trait ops.** `cb-compute`'s `R: Runtime` trait exposes ML-level operations (e.g. `build_histograms(pool, grads)`, `compute_gradients(approx, target)`, `eval_splits(...)`); the backend owns internal kernel decomposition while `cb-compute`'s host orchestration finalizes ordered sums. Fits the elementwise-kernel + host-ordered-reduce split (D-02) and lets each backend optimize independently.

### Threading & Determinism (success criterion 1–2; ties to D-02)
- **D-05: Per-bin histogram accumulation is an ordered single-pass over f64 accumulators in canonical object order** (`cb-core::sum_f64` semantics), bit-matching upstream's `thread_count=1` accumulation order. The kernel computes per-object gradients in parallel; the histogram **reduction** is the ordered host-side pass. (Chosen over parallel-partial-histogram-then-merge, which would require reproducing upstream's exact merge order — higher drift risk.)
- **D-06: Parallel elementwise kernels now, ordered reductions host-side.** Phase 3 allows parallel order-independent kernels (gradient/hessian per object, scatter) to exercise CubeCL's parallelism early and de-risk Phase 7; determinism comes exclusively from the host-side ordered reductions (D-02, D-05). All summation routes through `cb-core::sum_f64` — never a parallel float-add into shared accumulators.

### First Oracle-Locked Slice & Sequencing (success criteria 2, 5)
- **D-07: The first slice locks against simplified isolating params.** First train→predict oracle is generated with deliberately simplified upstream params (`bootstrap_type=No`, `random_strength=0`, fixed `l2_leaf_reg`, fixed `depth`/`learning_rate`/`iterations`) so any divergence can only be the tree/leaf math, not an interacting subsystem. Each additional knob is then added as its OWN additive oracle. Matches the narrowest-first roadmap philosophy.
- **D-08: The first slice covers BOTH RMSE (regression) and Logloss (binary classification).** The user chose to prove regression and binary classification together in the first slice (lining up with Phase 4's binary-clf + regression lock), accepting the broader first-slice surface.
- **D-09: All four leaf-estimation methods (Gradient, Newton, Exact, Simple) are implemented up front.** TRAIN-03 is completed in one wave rather than rolling methods out additively — the user chose breadth here.
- **D-10: Additive widening order after the first slice:** sampling/bootstrap (TRAIN-04) → full regularization `random_strength`/`bagging_temperature` (TRAIN-05) → overfitting detection / early stopping (TRAIN-06) → per-iteration eval-set metric logging (TRAIN-07) → automatic learning-rate selection (TRAIN-08). Each gets its own oracle.

### Intermediate-Oracle Depth (success criterion 2; INFRA-04)
- **D-11: Python-reachable oracle floor — no C++ instrumentation in Phase 3.** The intermediate oracle asserts per-tree **splits** (feature + border), per-tree **leaf values**, and **per-iteration approximants**, all extracted from the pinned Python `catboost==1.2.10` API (`save_model(format='json')` + `staged_predict`). Internal histograms / per-object gradients are NOT extracted — that would need C++ instrumentation, which Phase-1 D-08 deferred (primarily to Phase 5). Drift is localized instead by the simplified-isolating-params first slice (D-07). If a divergence genuinely cannot be localized at tree granularity, targeted instrumentation is reconsidered then.

### Claude's Discretion (parity-dictated — research reads upstream and reproduces)
- Exact split-score / gain formula and the **oblivious (symmetric) tree growth** procedure — `greedy_tensor_search.cpp`, `score_calcers.cpp`.
- Exact leaf-value math for Gradient/Newton/Exact/Simple (incl. `leaf_estimation_iterations`, l2 application) — `approx_calcer.*`, `online_predictor.*`, `hessian.*`.
- Per-loss gradient/hessian definitions for RMSE and Logloss — `error_functions.*`, `ders_holder.h`.
- **Split tie-break determinism** — equal-gain split selection must reproduce upstream's exact tie-break (cf. Phase-2 WR-01, where an STL `priority_queue` tie-break had to be reproduced). Research must pin this; it is a known parity landmine.
- Bootstrap/sampling draw sequences seeded by `TFastRng64` (Poisson/Bayesian/Bernoulli/MVS) and the exact draw ORDER — `tensor_search_helpers.cpp`, `bootstrap_options.*`.
- Overfitting-detector statistics (Wilcoxon/IncToDec/Iter, `od_pval`/`od_wait`) and `use_best_model` semantics — `catboost/libs/overfitting_detector/`.
- Auto learning-rate coefficient tables/formula keyed by dataset size / target / `use_best_model` / boost-from-average — `catboost/libs/train_lib/options_helper.cpp` (`TAutoLearningRateKey`).
- The exact eval-metric set and per-iteration logging shape (TRAIN-07).
- Concrete `cubecl` crate version — latest stable per CLAUDE.md; consult the CubeCL manual before writing kernels (AGENTS.md).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project & Roadmap
- `.planning/PROJECT.md` — core value, constraints (memory efficiency first-class, thiserror/anyhow, latest crate versions), oracle strategy.
- `.planning/ROADMAP.md` § "Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees" — goal + 5 success criteria this phase is judged against.
- `.planning/REQUIREMENTS.md` — TRAIN-01 … TRAIN-08 requirement text and traceability.
- `.planning/phases/01-workspace-lint-discipline-oracle-harness/01-CONTEXT.md` — crate map (D-01/D-05), `cb-compute` no-cubecl rule (D-03), `cb-backend` sole-runtime-owner (D-02), oracle pin 1.2.10 (D-07), fixture format/layout (D-09/D-10), frozen input corpus (D-11), determinism `thread_count=1` (D-12), deferred C++ instrumentation (D-08).
- `.planning/phases/02-data-layer-pool-quantization-reduction/02-CONTEXT.md` — `QuantizedPool` typed-width SoA bin contract (D-10/D-11) consumed by the histogram kernel; `cb-core` reduction primitive (D-07) + CI-grep ban (D-08) that all training sums route through.

### CubeCL (D-01/D-03 — kernels live in cb-backend)
- `AGENTS.md` (project root) — CubeCL kernel rules: kernels must use `generics-float`; read the manual before writing kernel code; on any build error load the error guideline before fixing.
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — CubeCL manual entry point (MUST read before kernel authoring).
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` — load on ANY CubeCL build error before attempting a fix.

### Vendored Reference & Oracle Source (catboost-master/, version 1.2.10)
- `catboost-master/catboost/libs/train_lib/train_model.cpp`, `train_model.h` — top-level CPU train loop / boosting driver.
- `catboost-master/catboost/libs/train_lib/options_helper.cpp` — automatic learning-rate selection (`TAutoLearningRateKey`, `TLearningRateCoefficients`) (TRAIN-08).
- `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp`, `.h` — oblivious (symmetric) tree growth + split search (TRAIN-02).
- `catboost-master/catboost/private/libs/algo/score_calcers.cpp`, `.h`, `rand_score.cpp` — split score / gain functions + `random_strength` (TRAIN-05).
- `catboost-master/catboost/private/libs/algo/tensor_search_helpers.cpp`, `.h` — bootstrap/sampling helpers seeded by the RNG (TRAIN-04).
- `catboost-master/catboost/private/libs/algo/approx_calcer.cpp`, `.h`, `approx_calcer_helpers.*`, `approx_updater_helpers.*` — leaf-value estimation Gradient/Newton/Exact/Simple (TRAIN-03).
- `catboost-master/catboost/private/libs/algo_helpers/error_functions.cpp`, `.h`, `ders_holder.h`, `hessian.cpp`, `.h`, `online_predictor.*` — per-loss gradient/hessian (RMSE, Logloss) + leaf statistics.
- `catboost-master/catboost/private/libs/algo/index_calcer.cpp`, `.h`, `calc_score_cache.*`, `fold.cpp`, `.h` — object→leaf indexing, histogram/score caches, fold bookkeeping (the `TFold` non-ordered subset for plain boosting).
- `catboost-master/catboost/private/libs/algo/learn_context.cpp`, `.h` — training context / state.
- `catboost-master/catboost/libs/overfitting_detector/overfitting_detector.cpp`, `.h` + `catboost-master/catboost/private/libs/options/overfitting_detector_options.*` — early stopping (Wilcoxon/IncToDec/Iter, `od_pval`/`od_wait`) (TRAIN-06).
- `catboost-master/catboost/private/libs/options/boosting_options.*`, `bootstrap_options.*`, `catboost_options.*`, `loss_description.*` — parameter defaults the simplified first slice (D-07) and later additive slices pin against.
- **Reduction order (D-02/D-05):** research must confirm upstream's `thread_count=1` histogram accumulation + leaf-sum order and reproduce it via `cb-core::sum_f64`.

### Process / Project Rules
- `CLAUDE.md` (project root) — constraints, naming, mandatory source/test separation, latest-crate-versions rule.
- `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — Rust lint/error/test conventions and the source/test-separation rule.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cb-compute/` — currently a stub (`lib.rs` doc comment only; manifest has `[lints] workspace = true` and intentionally NO `cubecl`/`anyhow` deps). Phase 3 fills it with the abstract `R: Runtime`/`F: Float` traits + host orchestration. **Keep `cubecl` out of this crate (D-03).**
- `crates/cb-backend/` — currently a stub with `SelectedRuntime = ()` placeholders per feature arm (`cpu`/`wgpu`/`cuda`/`rocm`), zero runtime dispatch (P1 D-02). Phase 3 replaces the `cpu` arm with the CubeCL CPU runtime and adds the `#[cube]` kernels here.
- `crates/cb-train/` — stub; Phase 3 fills it with the boosting loop, calling `cb-compute`.
- `crates/cb-core/` — exports `CbError`/`CbResult`, `TFastRng64`, and the order-locked `sum_f64`/`sum_f32_in_f64` reduction primitive. Training sums, sampling draws, and metric accumulation all reuse these (no new summation primitives — D-08 CI grep enforces it).
- `crates/cb-data/` — `Pool`/`QuantizedPool` complete; the typed-width SoA bin accessor (P2 D-11) is the histogram-kernel input.
- `crates/cb-oracle/` — live fixture infra (`.npy` read via `ndarray-npy`, ≤1e-5 abs-error gate, per-stage `compare_stage` API) + frozen input corpus at `fixtures/inputs/{numeric_tiny,numeric_categorical,grouped_ranking}`. Phase 3 adds expected-OUTPUT fixtures (splits/leaves/approximants) referencing those inputs; the corpus generator extends to emit training oracles via `save_model(json)` + `staged_predict` (D-11).

### Established Patterns
- Source/test separation mandatory: dedicated `*_test.rs` files, no inline `#[cfg(test)]`; test-lint exemption via `#![cfg_attr(test, allow(...))]` (already present atop each stub `lib.rs`).
- Deny-lints via workspace `[lints]`; `anyhow` structurally banned from library crates (P1 D-13/D-14).
- Enforcement-by-CI-grep idiom (anyhow ban, reduction-primitive D-08 ban) — Phase 3 keeps all float summation inside `cb-core::sum_f64`.
- Oracle determinism: pinned seed, `thread_count=1`, frozen committed fixtures (generator does not run in CI).

### Integration Points
- `cb-compute`'s abstract `R: Runtime`/`F: Float` traits (D-03/D-04) are the boundary on which `cb-backend` attaches the CubeCL CPU runtime now (D-01) and GPU runtimes additively in Phase 7 (P1 D-02/D-05) — `cb-compute`/`cb-train` stay unchanged across the GPU addition.
- The `QuantizedPool` typed-width bin accessor (P2 D-11) is consumed by the histogram kernel; the `cb-core` reduction primitive (P2 D-07) is the process-wide ordered-sum invariant the host orchestration finalizes through (D-02/D-05).
- The trained-model representation produced here (oblivious trees + leaf values) is the input to Phase 4's `.cbm` serialization / apply / SHAP and the Builder API.

</code_context>

<specifics>
## Specific Ideas

- The user deliberately chose the **more ambitious** option at the architectural fork: stand up the CubeCL CPU runtime *in Phase 3* (D-01) rather than a plain-Rust placeholder, to validate the CubeCL seam early. The parity safety net is that all order-sensitive arithmetic is pulled OUT of the kernels and finalized host-side via `cb-core::sum_f64` (D-02/D-05) — kernels only do order-independent work.
- The user chose **breadth in the first slice**: both RMSE and Logloss (D-08) and all four leaf-estimation methods up front (D-09), while keeping the *parameter* surface deliberately narrow (simplified isolating params, D-07). The split is intentional: prove the full tree/leaf math surface, but against params simple enough that a divergence is unambiguous.
- The user kept Phase 3 within **Python-reachable oracles** (D-11), honoring the Phase-1 deferral of C++ instrumentation — accepting tree-granularity drift localization in exchange for no C++ build this phase.

</specifics>

<deferred>
## Deferred Ideas

- **Parallel/multi-threaded performance tuning** — Phase 3 exercises parallel elementwise kernels (D-06) but is judged on parity, not perf. Any parallel-partial-histogram-merge or multi-thread reduction strategy that must reproduce upstream's merge order is deferred to a later perf pass on the established ordered-reduce seam.
- **C++ instrumentation of `catboost-master`** for iteration-internal oracle values (per-bin histograms, per-object gradients) — deferred per P1 D-08, primarily to Phase 5; reconsidered in Phase 3 only if a divergence cannot be localized at tree granularity (D-11).
- **GPU kernels (`wgpu`/`cuda`/`rocm`)** — added additively in Phase 7 on the `cb-compute` seam established here; `cb-backend` is where they attach.
- **Ordered boosting / ordered CTR / categoricals, serialization / SHAP / Builder API** — Phases 5 and 4 respectively; Phase 3 is plain boosting on numeric features only.

None of the above are scope creep — all are explicitly later-phase items surfaced while clarifying Phase 3 boundaries.

</deferred>

---

*Phase: 3-CPU Training Core — Plain Boosting & Oblivious Trees*
*Context gathered: 2026-06-13*
