---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 04-02-PLAN.md
last_updated: "2026-06-14T00:00:00.000Z"
last_activity: 2026-06-14 -- Plan 04-02 complete (apply path + prediction types + CrossEntropy/Focal)
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 22
  completed_plans: 20
  percent: 41
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-13)

**Core value:** A memory-efficient, Rust-native CatBoost implementation with verifiable feature parity (oracle-tested ≤1e-5), embeddable in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.
**Current focus:** Phase 04 — model-serialization-shap-rust-api-first-full-oracle-lock

## Current Position

Phase: 04 (model-serialization-shap-rust-api-first-full-oracle-lock) — EXECUTING
Plan: 3 of 5
Status: 04-02 complete; Wave 3 (04-03 .cbm serialize) next
Last activity: 2026-06-14 -- Plan 04-02 complete (apply path + prediction types + CrossEntropy/Focal)

Progress: [████░░░░░░] 40% (2 of 5 phase-04 plans complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 9
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 03 | 9 | - | - |
| 04 | 2 | ~90 min | ~45 min |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 01 P01 | 5 | 4 tasks | 21 files |
| Phase 01 P02 | 4min | 1 tasks | 4 files |
| Phase 01 P03 | 9min | 3 tasks | 42 files |
| Phase 02 P01 | 10min | 3 tasks | 22 files |
| Phase 02 P02 | 30min | 2 tasks | 18 files |
| Phase 02 P03 | 5 | 2 tasks | 7 files |
| Phase 02 P04 | 25min | 2 tasks | 11 files |
| Phase 02 P05 | ~25min | 2 tasks | 12 files |
| Phase 03 P00 | ~75min | 4 tasks | 16 files |
| Phase 03 P01 | ~20min | 4 tasks | 27 files |
| Phase 03 P02 | 12min | 2 tasks | 14 files |
| Phase 03 P03 | ~70min | 2 tasks | 28 files |
| Phase 03 P04 | 95min | 2 tasks | 14 files |
| Phase 03 P05 | 50min | 2 tasks | 11 files |
| Phase 03 P06 | 35min | 2 tasks | 30 files |
| Phase 03 P07 | 7min | 2 tasks | 5 files |
| Phase 03 P08 | 10min | 3 tasks | 8 files |
| Phase 04 P01 | ~40min | 3 tasks | 10 files |
| Phase 04 P02 | ~50min | 2 tasks | 15 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Phased by oracle-passing vertical slices, narrowest-first (research-mandated); each phase must be oracle-passing ≤1e-5 vs upstream before the next begins.
- Roadmap: CPU path fully oracle-locked (through Phase 6) before GPU (Phase 7); GPU is additive on the generic `R: Runtime` boundary established in Phase 3.
- [Phase ?]: Plan 01-01: pinned approx to stable 0.5 (not 0.6.0-rc2 pre-release); test-only dev-dep
- [Phase ?]: Plan 01-01: committed Cargo.lock for supply-chain integrity (T-01-SC)
- [Phase ?]: Plan 01-01: uniform in-code test-lint exemption + --lib CI clippy gate (Pitfall 1)
- [Phase ?]: Plan 01-02: TFastRng64 ported bit-for-bit; two PCG streams deduped into shared Lcg32 (bitstream-identical, oracle-proven)
- [Phase ?]: Plan 01-02: derived Clone/PartialEq/Eq on CbError to enable Result equality assertions (backward-compatible)
- [Phase ?]: INFRA-04 compare_stage ships API + real-fixture read + 1e-5 gate in P1; comparison vs Rust-computed actuals deferred to P3/P4
- [Phase 02]: Plan 02-01: single sequential f64 reduction primitive (`cb-core::sum_f64`/`sum_f32_in_f64`) order-locked via `[1e16,1.0,-1e16]==0.0`; D-08 CI grep bans all other float summation
- [Phase 02]: Plan 02-01 A1/A3 (RESOLVED): `get_borders()` surfaces the `f32::MIN` NanMode sentinel for NaN(Min) features at the default border budget; presence is config-dependent (omitted at small budgets / nan_mode=Max), so pinned per-fixture in `borders_quant/config.json` — Rust must match each fixture verbatim
- [Phase 02]: Plan 02-01 A2 (RESOLVED): catboost 1.2.10 default `border_count=254`, `feature_border_type=GreedyLogSum`, `nan_mode=Min`
- [Phase 02]: Plan 02-01 A4 (RESOLVED): integer cat features stringify as PLAIN integers before `CalcCatFeatureHash` (`'3'` ui32=2658984922 ≠ `'3.0'` ui32=1187060909)
- [Phase 02]: Plan 02-01 A5 (RESOLVED): `(string→ui32)` hash vectors extracted from upstream model.json `ctr_data` hash_map; Rust must port `util/digest/city.cpp` (CityHash64 & 0xffffffff) to reproduce them bit-exactly (no third-party crate)
- [Phase ?]: [Phase 02]: Plan 02-01 COMPLETE — sum_f64/sum_f32_in_f64 reduction primitive shipped + order-locked; D-08 grep gate live; arrow 59 / polars 0.54 wired; Wave-0 borders/cat-hash/class-weight fixtures committed; A1-A5 resolved
- [Phase ?]: [Phase 02]: Plan 02-02: Pool is lifetime-free owned Vecs (D-02); IngestSource trait seam validates column lengths with typed CbError; borrowed view plugs in at Phase 8 without reshaping Pool
- [Phase ?]: [Phase 02]: Plan 02-02: GreedyLogSum binarizer bit-transcribed from binarization.cpp (f64 penalty/score, f32 border midpoints), oracle-locked <=1e-5 per feature; sums routed through cb_core::sum_f64
- [Phase ?]: [Phase 02]: Plan 02-02 (Rule 1 fix): borders_quant fixtures regenerated from STANDALONE Pool.quantize().save_quantization_borders() (raw 49/49/49/49) instead of training-pruned get_borders(); f32 sentinel snapped to exact f32::MIN
- [Phase ?]: Per-feature NanMode: NaN-bearing column -> Min sentinel, NaN-free -> Forbidden
- [Phase ?]: Float bin width hard-capped at u16 -> CbError not panic; u32 categorical-only (utils.h:175-181)
- [Phase 02]: Plan 02-04: CityHash64 ported bit-exact from vendored util/digest/city.cpp (Yandex CityHash 1.0, NOT mainline/third-party crate); CalcCatFeatureHash = city_hash_64 & 0xffffffff; first-seen perfect-hash bins (bin = map.size()), uniq count bounded to u32::MAX with typed CbError (no panic)
- [Phase 02]: Plan 02-04 (Rule 1 fix): cat_hash string_to_ui32 fixtures regenerated from a standalone C++ tool transcribing the vendored city.cpp (generator/cityhash_oracle.cpp) -- the Wave-0 vectors had been extracted from a trained model's ctr_data hash_map (CTR-projection hashes, NOT CalcCatFeatureHash). 'alpha'=1296865003 (was 3214079027); '3'=593172586 (was 2658984922). Downstream cat-hash consumers must use cb_data::calc_cat_feature_hash, never a model's ctr_data hash_map.
- [Phase 02]: Plan 02-05: Polars rides the shared Arrow validator (rechunk -> cont_slice -> arrow::Float64Array -> arrow_f64_column) to avoid polars/arrow-crate type incompatibility while honoring the rechunk->Arrow key_link (D-05)
- [Phase 02]: Plan 02-05: ingestion CbError variants (Dtype/LengthMismatch/NanInCategorical/Ingestion) stringify external arrow/polars errors (no #[from]) so the enum keeps Clone+PartialEq+Eq (Shared Pattern C / D-06); this is the taxonomy Phase 8 maps to Python exceptions (PYAPI-05)
- [Phase 02]: Plan 02-05: class weights computed in f32 to bit-match upstream float lambdas (SqrtBalanced fixture is f32 sqrt(3) widened, absorbed by <=1e-5, fixture unchanged); 1e-8 floor returns 1.0 on an empty/degenerate class (no div-by-zero); all summary sums via cb_core::sum_f64
- [Phase 02]: Plan 02-05 COMPLETE — DATA-06 (Arrow+Polars validated ingestion) + DATA-08 (Balanced/SqrtBalanced + per-object/per-class weights) shipped, oracle-locked; Phase 2 data layer complete
- [Phase 03]: Plan 03-00: CubeCL CpuRuntime stood up now (D-01) — SelectedRuntime = cubecl::cpu::CpuRuntime; cubecl 0.10.0 + bytemuck wired into cb-backend ONLY (D-03); cb-compute stays cubecl-free
- [Phase 03]: Plan 03-00: first #[cube] gradient_kernel<F: Float> (generics-float, RMSE der1 = target-approx) runs on CpuRuntime — order-independent elementwise only, NO reduction (D-02/D-06); RESEARCH Open Q2 closed
- [Phase 03]: Plan 03-00: cubecl 0.10.0 launch API — ArrayArg::from_raw_parts(Handle, len) (2 args, by value, no turbofish); read_one(Handle)->Result<Bytes, ServerError>; clone output Handle for launch arg
- [Phase 03]: Plan 03-00: cb-oracle::model_json parses upstream model.json (scale_and_bias=[1,[bias]]); extractors return Vec<f64> for compare_stage(Stage::Splits|LeafValues); no unwrap, OracleError::MalformedModel
- [Phase 03]: Plan 03-00: Open Q1 RESOLVED — score_function=L2 (simplest first-slice split math); regression_skeleton + binclf_skeleton frozen with D-07 isolating params (bootstrap_type=No, random_strength=0, depth=2, iterations=5, leaf_estimation_iterations=1, thread_count=1, explicit boost_from_average); Logloss staged = RawFormulaVal logits (A5/Pitfall 6)
- [Phase 03]: Plan 03-00: Wave-0 Nyquist gate signed off (03-VALIDATION.md nyquist_compliant: true, wave_0_complete: true) — unblocks Plan 01 slice_first_oracle (gates TRAIN-01/02/03)
- [Phase 03]: Plan 03-01: cb-compute abstract Runtime/Float boundary stood up cubecl-free (D-03 verified via cargo tree); cb-backend CpuBackend impls it launching elementwise gradient/hessian/scatter #[cube] kernels (UN-reduced, D-02), host folds via cb_core::sum_f64
- [Phase 03]: Plan 03-01: oblivious leaf index = forward bit order (split i -> bit i); model.json leaf_values are ALREADY learning_rate-scaled — boosting stores lr*delta and adds directly to staged approx (verified vs regression_skeleton tree 0)
- [Phase 03]: Plan 03-01: Gradient leaf delta = CalcAverage(sumDer, sumWeight, scaledL2), scaledL2 = l2*(sumAllW/docCount) (== l2 unweighted); L2 split score = sum over level leaves of avg*sumDer, strict gain>bestGain first-wins tie-break (Pitfall 1)
- [Phase 03]: Plan 03-01: TRAIN-01/02 COMPLETE, TRAIN-03 Gradient done (Newton/Exact/Simple -> Plan 02); slice_first_oracle gates Splits/LeafValues/StagedApprox <=1e-5 for RMSE + Logloss; cargo test --workspace green
- [Phase 03]: Plan 03-01: added CbError::DepthExceeded (depth>16) + CbError::Degenerate (no candidate split/empty) — guards, never panic (T-03-01-01/02); extended cb-oracle::model_json with float_feature_borders() accessor for the oracle test
- [Phase ?]: Plan 03-02: TRAIN-03 complete — four leaf methods (Gradient/Newton/Exact/Simple) oracle-locked <=1e-5; Newton via Logloss (der2=-p(1-p) distinct), Exact via MAE weighted-median (Exact rejected for RMSE/Logloss upstream), Simple==Gradient (A6); added Loss::Mae + mae_gradient_kernel
- [Phase 03]: Plan 03-03: TRAIN-04 bootstrap/sampling — No/Bernoulli/MVS oracle-locked <=1e-5 end-to-end; added TFastRng64::gen_rand_real1; Bayesian per-1000-block reseed (from_seed(rand_seed+block_idx).advance(10)) + verbatim FastLogf (NOT ln); Bernoulli f32-subsample sequential control; MVS CalculateThreshold importance sampler (8192 block)
- [Phase 03]: Plan 03-03: Poisson REJECTED on CPU (CbError) mirroring upstream bootstrap_options.cpp — no Python CPU oracle exists; unit-locked dispatch rejection only (plan deviation, Rule 3)
- [Phase 03]: Plan 03-03: sample weights/control gate SPLIT SCORING ONLY; leaf VALUES use the full unsampled fold (verified vs upstream — Bayesian/MVS weights never enter CalcLeafValues). Per-tree RNG draw accounting: 2 pre (fold pick + derivative seed) + bootstrap-internal + (depth+1) per-level CalcScores + MVS full-doc +2
- [Phase 03]: Plan 03-03 RESIDUAL: Bayesian MULTI-TREE end-to-end lock #[ignore]d (first tree + draw sequence locked); tree-1+ diverges ~0.02 INSENSITIVE to RNG phase — structural Bayesian draw-stream issue, tracked in deferred-items.md
- [Phase ?]: TRAIN-05 random_strength: std_normal verbatim Marsaglia-polar port over TFastRng64; two-pass SetBestScore/SelectBestCandidate draw order; first-tree end-to-end lock, multi-tree RNG-phase residual escalated D-11
- [Phase 03]: Plan 03-08 (CR-01 closed): score_st_dev now reads the FULL un-sampled fold weighted_der1, NOT the control-masked score_weighted_der1 — matches upstream CalcDerivativesStDevFromZeroPlainBoosting (greedy_tensor_search.cpp:99 = fold.BodyTailArr.front().WeightedDerivatives) and the leaf path; histogram inputs to the perturbed search stay masked. Masked input biases scoreStDev low (zeroed entries, full-n denominator) whenever bootstrap_type!=No + random_strength!=0.
- [Phase 03]: Plan 03-08: CR-01 RED->GREEN locked at the cb-compute UNIT boundary (score_st_dev_masked_vector_biases_low_vs_full_fold_cr01), NOT first-tree end-to-end. Exhaustive sweep proved numeric_tiny's first tree cannot isolate the std-dev bias: tree-0 splits are robust to the masked-vs-full difference at small random_strength, and at large random_strength the variable-length Box-Muller draw-stream residual (D-11) dominates and the fix is not isolable. WR-06 (n-from-slice-length) deliberately NOT folded in (signature unchanged).
- [Phase ?]: [Phase 04]: Plan 04-01 COMPLETE — per-leaf weights in cb-train (sum_f64, 2^depth, unweighted==doc count); canonical cb-model::Model {oblivious_trees,bias,float_feature_borders}+per-tree leaf_weights reusing cb_train::Split; flatc 25.12.19 bindings committed (D-01, genuine flatc --rust --gen-all, user-approved deviation from per-file cmd); model_json leaf_weights #[serde(default)]; oracle lock 2/2.
- [Phase 04]: Plan 04-02 COMPLETE — pure-Rust cb-model::predict_raw apply path (strict-> binarize, forward-bit leaf index via cb_train::leaf_index, bias + sum_f64 over leaf values; NO backend/cubecl import — MODEL-02) oracle-locked ≤1e-5; PredictionType {RawFormulaVal/Probability/LogProbability/Class/Exponent} (two-column probs, f64::exp; Exponent absorbs FastExp gap A2) locked ≤1e-5 (LOSS-06, uncertainty types deferred to Phase 6 per D-10); Loss::CrossEntropy (delegates to logloss helper) + Loss::Focal{alpha,gamma} (error_functions.h:1684-1709, p-clamp [1e-13,1-1e-13] T-04-02-02) — binclf trains under all three losses oracle-locked ≤1e-5 (LOSS-01 complete, D-09).
- [Phase 04]: Plan 04-02 CubeCL pattern — a GENERIC #[cube(launch)] scalar arg requires F: ScalarArgType (CubeElement+Scalar+NumCast), incompatible with the generics-float rule; pass loss params (alpha/gamma) as length-1 Array<F> read at index 0 to keep F: Float. Math via associated-fn form (F::ln/F::powf/F::exp/F::clamp) per the cubecl error guideline; label branch via if-as-statement. Loss enum dropped Eq (Focal carries f64; no call site needed Eq).
- [Phase 04]: Plan 04-02 ENV — cargo test -p cb-compute loss and cargo test --workspace blocked by disk (<1GB free; polars-core test-profile rlib ~1.3GB). CrossEntropy/Focal der1/der2 fully exercised+passing via cb-train/tests/loss_oracle_test.rs instead; logged in deferred-items.md.

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- Phase 5 (Ordered Boosting/CTR), Phase 7 (GPU/CubeCL-ROCm), and Phase 8 (Python ABI/packaging) are flagged NEEDS DEEPER RESEARCH — run the per-phase research spike before planning each.
- GPU tolerance epsilon (Phase 7) is unspecified — must be set and signed off before Phase 7 planning.
- **Plan 02-01 COMPLETE (human approved Task-3 checkpoint).** Tasks 1–3 committed (1f2b9f1, d92ae65, 025c381); 02-01-SUMMARY.md written and self-checked; plan counter advanced to 02-02. No open blockers from 02-01.
- **Environment: disk pressure.** `cargo test --workspace` pulls in `cubecl-cpu`'s heavy `tracel-mlir-sys` (MLIR) transitive dep, which filled the disk (100%) during Plan 03-01 and corrupted incremental caches mid-build. Resolved by clearing `target/debug/incremental` + stale deps and rebuilding. CPU-only builds still compile the MLIR optimizer dep — keep an eye on disk headroom before full-workspace builds.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-13T19:58:05.723Z
Stopped at: Phase 4 context gathered
Resume file: .planning/phases/04-model-serialization-shap-rust-api-first-full-oracle-lock/04-CONTEXT.md
