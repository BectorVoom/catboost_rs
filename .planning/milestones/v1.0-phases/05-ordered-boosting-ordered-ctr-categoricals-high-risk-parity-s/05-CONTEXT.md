# Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice) - Context

**Gathered:** 2026-06-14
**Status:** Ready for planning

<domain>
## Phase Boundary

CatBoost's defining **anti-leakage** algorithms plus native categorical handling, oracle-locked ≤1e-5 vs upstream CatBoost **1.2.10**, with **per-object intermediate oracles** confirming no silent leakage. Deliverables (ORD-01 … ORD-05):

- **Multi-permutation fold machinery (ORD-01)** — `fold_count` permutations, `TFold`-equivalent bookkeeping, seeded by the Phase-1 `TFastRng64` port, reproducing upstream permutations exactly.
- **Ordered boosting (ORD-02)** — `EBoostingType::Ordered` with exact prefix boundaries and the exact prior formula `(sumTarget + prior) / (sumCount + priorWeight)`; per-object intermediate oracle, no leakage signature in train metrics.
- **Ordered target statistics / CTR (ORD-03)** — all six CTR types (`Borders`, `Buckets`, `BinarizedTargetMeanValue`, `FloatTargetMeanValue`, `Counter`, `FeatureFreq`) with priors, matching upstream.
- **One-hot encoding (ORD-04)** — low-cardinality categoricals (`one_hot_max_size` threshold) select the correct encoding path.
- **Feature combinations / tensor CTRs (ORD-05)** — `SimpleCtrs`/`CombinationCtrs`, `max_ctr_complexity` control, matching upstream on categorical datasets.

**NOT in this phase:** multiclass / multilabel / full regression-loss matrix / ranking losses+metrics / text+embedding features / uncertainty / advanced fstr (SHAP interaction, PredictionDiff, SAGE) / monotone constraints / feature selection / alternative grow policies (all Phase 6); GPU backends (Phase 7); Python bindings (Phase 8). LossFunctionChange importance remains deferred (Phase-4 D-12).

</domain>

<decisions>
## Implementation Decisions

### Per-Object Oracle Strategy (success criteria 1–2, INFRA-04; the flagged central risk)
- **D-01: Per-object ground truth comes from a standalone C++ micro-harness, NOT a full catboost build and NOT Python alone.** Reuse the Phase-2 precedent (`crates/cb-oracle/generator/cityhash_oracle.cpp`): a small C++ tool that links ONLY the specific upstream translation units needed (`online_ctr.cpp`, the ordered path in `approx_calcer.cpp`, fold/permutation generation) and dumps per-object internals to `.npy` fixtures. Rationale: the upstream Python API cannot expose per-object ordered approximants or the per-object running target-statistic *during* training, yet the success criteria demand exactly that ("per-object intermediate oracles confirming no silent leakage"). A full catboost C++ build is rejected as infeasible — disk is ~100% full and the MLIR-scale build footprint won't fit (see Blockers/Concerns in STATE.md). This finally actions the Phase-3 D-11 deferral ("C++ instrumentation deferred primarily to Phase 5").
- **D-02: The per-object oracle schema is the FULL per-object stack.** The micro-harness dumps, and Rust asserts ≤1e-5: (1) **permutation indices** per fold; (2) **per-object running CTR** numerator/denominator under the prior formula `(sumTarget + prior) / (sumCount + priorWeight)`; (3) **per-object ordered approximant per iteration**. Chosen over shallower schemas because a prefix-boundary off-by-one is exactly the bug that final-prediction parity would mask — this is a "high-risk" phase and the per-object signal is the point.
- **D-03: The permutation is the linchpin and is locked as its own stage.** Both our Rust impl and upstream are seeded by the bit-exact `TFastRng64` port (Phase 1), so the harness ALSO dumps the permutation(s) it generated. Rust asserts (a) our permutation reproduces upstream's exactly, THEN (b) per-object CTR/approx values match *under that permutation*. Permutation reproduction is locked before any value comparison.

> **⚠ DECISION REVISION (2026-06-14, post-research, user-approved) — supersedes the *mechanism* of D-01 (and the sourcing detail of D-02/D-03):**
> Research (`05-RESEARCH.md` § "Per-Object Oracle Strategy") proved the locked D-01 mechanism — a standalone C++ micro-harness that **links** the upstream `online_ctr.cpp` / ordered-path `approx_calcer.cpp` / fold-permutation TUs — is **infeasible**: those TUs are mid-graph trainer internals (no isolatable link boundary; transitively require the full build D-01 rejected on disk grounds). The `cityhash_oracle.cpp` precedent worked only because CityHash is a leaf utility that was *transcribed*, not linked.
> **Replacement mechanism (approved): transcribe-then-self-oracle.** A dependency-free `crates/cb-oracle/generator/ordered_oracle.cpp` (zero catboost includes) transcribes the four small algorithms verbatim and dumps the per-object `.npy` stack. Trust is established via independent anchors, NOT by linking the trainer:
>   - **Permutation (D-03 linchpin, unchanged intent):** anchored to the already-oracle-locked `cb-core::TFastRng64` (bitstream-verified) — a Rust reproduction of Fisher-Yates is itself ground truth; the C++ harness is a cross-check.
>   - **Final CTR values:** anchored to the trained model's own `ctr_data` blob (`TCtrValueTable`, Python-reachable today).
>   - **Per-iteration ordered approximant (A2 residual risk, accepted):** NO direct external anchor exists upstream. Validated **indirectly** — final-prediction parity ≤1e-5 + internal consistency (monotone running num/denom) + identity-permutation degeneration (prefix == final). A direct anchor would require the rejected full instrumented build.
> **The D-02 per-object schema intent is preserved** — permutation indices, per-object running CTR num/denom, per-iteration ordered approx are all still dumped and asserted (integers exact; floats ≤1e-5); only the *source of ground truth* shifts from "link the trainer" (impossible) to "oracle-locked RNG + model `ctr_data` + indirect anchors". Planner: follow this revised mechanism; the `.npy` layout is in `05-RESEARCH.md` § "Per-Object Oracle Schema (D-02)".

> **⚠ DECISION REVISION (2026-06-15, gap-closure, user-approved) — authorizes a C++ instrumentation deviation from the D-11 / Phase-1 D-08 "Python-reachable floor, no C++ instrumentation" rule, SCOPED to the single pc=4 blocking gap:**
> Re-verification (`05-VERIFICATION.md`, status `gaps_found`) left ONE blocking gap: at the production-default `permutation_count=4`, the `create_folds` AveragingFold partition is `[6,0,8,16]` while catboost 1.2.10 produces `[6,0,10,14]`. SC-1 ("reproduces upstream permutations exactly") has no permutation_count carve-out and pc=4 is the default, so the developer escalated it as a blocker (2026-06-15). 05-15's **empirical** draw-stream sweep (Fisher-Yates 0..7 × pre-averaging GenRands 0..7) found no clean per-fold rule reproducing BOTH the e2e-bit-exact pc=1/pc=2 stream AND pc=4 — pointing to extra upstream RNG consumption at `permutation_count>2` (multi-fold structure-fold selection / per-fold CTR-grid construction drawing on the same persistent `Rand`) that is not recoverable from the lossy partition observable alone.
> **Approved mechanism:** build a **minimal instrumented C++ harness** that logs catboost 1.2.10's per-fold RNG draw accounting at pc=4 (and ideally general `permutation_count>=4`) as ground truth, then transcribe the exact draw accounting into `create_folds` so the pc=4 partition equals `[6,0,10,14]` integer-exact. This is a **deliberate, user-approved deviation** from D-11/D-08 — recorded here so the planner and plan-checker treat C++ instrumentation as IN-SCOPE for this gap only. It does NOT re-open instrumentation for any other Phase 5 mechanism (the transcribe-then-self-oracle D-01-revision still governs everything else).
> **Hard feasibility constraint (first-class risk):** D-01 rejected a full instrumented catboost build on disk grounds (root disk ~100% full — see STATE.md Blockers and the `disk-pressure-and-full-suite-verification` memory). The plan MUST treat build/link feasibility under disk pressure as a first-class risk: prefer the **smallest** instrumented unit (a targeted RNG-logging hook over only the fold/permutation/learn-context translation units, or a localized source patch + minimal build), free disk before/after, and include an explicit **feasibility-probe task that escalates** (rather than silently expanding scope) if the minimal instrumented build cannot fit. The committed `fixtures/multi_permutation_fold/` pc=4 dump (`leaf_weights.json [6,0,10,14]`, `model_pc4.json`) remains the upstream anchor.
> **Closure bar:** (a) `create_folds` reproduces catboost 1.2.10 pc=4 `[6,0,10,14]` integer-exact (ideally general `permutation_count>=4`); (b) `multi_permutation_count_four_partition_pinned_and_upstream_delta_recorded` is upgraded from a pinned-delta record to an integer-exact equality assertion vs the committed pc=4 `leaf_weights`; (c) a pc=4 e2e prediction oracle proves final predictions match upstream `<=1e-5`; (d) no regression on the pc=1/pc=2 integer-exact locks or the existing e2e oracles.

### First Slice & Additive Order (success criteria 2–5; D-07 isolating-params philosophy)
- **D-04: One-hot first (ORD-04) is the narrowest first slice.** Low-cardinality one-hot encoding uses no permutation and no CTR — it rides the EXISTING plain boosting + oblivious trees, isolating the encoding-path selection (`one_hot_max_size`) before any ordering math enters.
- **D-05: Additive spine after one-hot:** ORD-01 permutation/fold machinery (locked by the permutation-only oracle, D-03) → **Plain CTR** → **Ordered CTR** → **Ordered boosting** (ORD-02) → tensor/combination CTRs (ORD-05) last (they build on single-feature CTR). Each layer is added as its own additive oracle so a divergence isolates to a single mechanism.
- **D-06: Plain CTR is locked before Ordered CTR (the key isolation).** Lock Plain-mode CTR first (simpler target statistic, no per-object prefix), THEN add the ordered-permutation CTR, THEN ordered boosting — so a per-object divergence localizes to the CTR math vs the ordering math vs the boosting ordering, never an entangled three-way break. Mirrors Phase-3 D-07.

### CTR Type Coverage (success criterion 3, ORD-03)
- **D-07: All six CTR types are implemented and oracle-locked this phase, each as its own additive fixture.** `Borders`, `Buckets`, `BinarizedTargetMeanValue`, `FloatTargetMeanValue`, `Counter`, `FeatureFreq` — ORD-03 is fully closed (no partial deferral). Consistent with the breadth pattern from Phases 3–4 (prove the full math surface against narrow isolating params). Research should confirm which types share math (e.g. `Counter`/`FeatureFreq`) to right-size the fixture set, but every implemented type gets its own oracle.

### Oracle Fixtures (success criteria 1–5; INFRA-03/04)
- **D-08: New purpose-built categorical fixtures drive Phase 5.** Generate small fixtures with controlled cardinality — a low-cardinality column (forces one-hot below `one_hot_max_size`), a high-cardinality column (forces the CTR path), and small N so per-object permutation prefixes stay human-auditable. Reuse the existing `numeric_categorical` / `explicit_categorical` input corpora only where they already fit; do not retrofit a corpus that fails to isolate the one-hot vs CTR vs ordering paths.
- **D-09: Frozen-committed-fixtures convention holds.** The micro-harness (D-01) and the Python generator run OFFLINE; their `.npy`/config outputs are committed under `crates/cb-oracle/fixtures/`; neither generator runs in CI. Pinned `catboost==1.2.10`, `thread_count=1`.

### Carried Forward (locked in prior phases — do not re-litigate)
- **All parity-critical float summation routes through `cb-core::sum_f64`/`sum_f32_in_f64`** (Phase-2 D-07, Phase-3 D-02); the D-08 CI-grep ban applies to all new CTR/ordered accumulation (running sums, priors, leaf stats).
- **CubeCL `#[cube]` kernels + the `cubecl` dep live ONLY in `cb-backend`; `cb-compute` stays abstract** (Phase-3 D-03). Any new order-independent CTR/permutation kernel work attaches in `cb-backend`; ordered reductions finalize host-side.
- **Categorical hashing is `cb-data::calc_cat_feature_hash`** (CityHash64 & 0xffffffff, first-seen perfect-hash bins; Phase-2 D Plan 02-04) — CTR consumers use this, NEVER a model's `ctr_data` hash_map.
- **Source/test separation mandatory** (no inline `#[cfg(test)]`); `thiserror` in libraries, `anyhow` structurally banned (CI grep).

### Claude's Discretion (parity-dictated — research reads upstream and reproduces)
- **Exact training-config pin per fixture** — user chose research-driven, but it MUST follow the D-07 isolating-params principle: pin every knob explicitly (`boosting_type` Plain/Ordered not auto, explicit `simple_ctr`/`combinations_ctr` type + prior, small fixed `fold_count`, explicit `one_hot_max_size`, `max_ctr_complexity`, `thread_count=1`), with exact values chosen after reading upstream defaults. Reject upstream auto-selection for the locks.
- Exact permutation-generation draw order and `TFold` bookkeeping (body/tail prefix boundaries) — `fold.cpp`, `learn_context.cpp`.
- Exact ordered-CTR online accumulation order and the prior/`priorWeight` defaults per CTR type — `online_ctr.cpp`/`.h`, `cat_feature_options.*`, `ctr_config.h`.
- Exact ordered-boosting approximant update (per-prefix model application) — `approx_calcer.cpp` ordered path.
- One-hot vs CTR path-selection threshold semantics (`one_hot_max_size`, inclusive/exclusive) — `cat_feature_options.*`, `greedy_tensor_search.cpp`.
- Tensor/projection (`TProjection`) combination enumeration and `max_ctr_complexity` control — `projection.h`, `greedy_tensor_search.cpp`.
- Populating + applying the `.cbm`/`model.json` `ctr_data` section at inference (CTR values baked into the model; bindings already committed in `cb-model::generated::ctr_data_generated`) — `ctr_provider.h`, `online_ctr.h` (model-side), `ctr_data.fbs`.
- Whether new CTR/permutation work needs a `cb-backend` kernel or is pure host orchestration — keep order-independent work kernel-eligible, all reductions host-side via `cb-core::sum_f64`.
- Build feasibility of linking `online_ctr.cpp`/`approx_calcer.cpp` in isolation in the micro-harness (transitive header weight) — **research must validate this early**; if isolation is impractical, escalate before planning (the whole oracle strategy rests on D-01).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project & Roadmap
- `.planning/PROJECT.md` — core value, constraints (memory-efficiency first-class, `thiserror`/`anyhow`, latest crate versions), oracle strategy.
- `.planning/ROADMAP.md` § "Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice)" — goal + 5 success criteria this phase is judged against, plus the **research flag** (line-by-line `approx_calcer.cpp` + `online_ctr.*`; design the per-object intermediate-oracle schema first).
- `.planning/REQUIREMENTS.md` — ORD-01 … ORD-05 requirement text + traceability.
- `.planning/phases/04-model-serialization-shap-rust-api-first-full-oracle-lock/04-CONTEXT.md` — `.cbm`/`model.json` framing + apply path this phase extends with the `ctr_data` section; canonical `cb-model::Model`; Python-reachable oracle pattern (D-13); D-12 LossFunctionChange deferral.
- `.planning/phases/03-cpu-training-core-plain-boosting-oblivious-trees/03-CONTEXT.md` — generic `R: Runtime`/`F: Float` compute seam (D-01/02/03/04), host-ordered-reduce invariant (D-02/D-05), the `TFold` non-ordered subset for plain boosting (this phase adds the ordered subset), C++-instrumentation deferral "primarily to Phase 5" (D-11).
- `.planning/phases/02-data-layer-pool-quantization-reduction/02-CONTEXT.md` — `Pool`/`QuantizedPool`, `calc_cat_feature_hash` (CityHash64, first-seen perfect-hash), `cb-core::sum_f64` reduction + CI-grep ban.
- `.planning/phases/01-workspace-lint-discipline-oracle-harness/01-CONTEXT.md` — crate map, `TFastRng64` port (the permutation seed), oracle pin 1.2.10, fixture format/layout, `thread_count=1` determinism, deferred-C++-instrumentation history.

### Vendored Reference & Oracle Source (catboost-master/, version 1.2.10)
- `catboost-master/catboost/private/libs/algo/online_ctr.cpp`, `online_ctr.h` — **ordered CTR computation** (ORD-03): all six CTR types, online accumulation order, priors. (Research flag — read line by line.)
- `catboost-master/catboost/private/libs/algo/approx_calcer.cpp` — **ordered boosting** approximant update / prefix application (ORD-02). (Research flag — read line by line.)
- `catboost-master/catboost/private/libs/algo/fold.cpp` — `TFold` bookkeeping, body/tail prefix boundaries, permutation storage (ORD-01).
- `catboost-master/catboost/private/libs/algo/learn_context.cpp` — permutation generation / training state (ORD-01, D-03).
- `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp` — tree growth with CTR features, one-hot vs CTR path selection, combination enumeration (ORD-04/ORD-05).
- `catboost-master/catboost/private/libs/algo/index_calcer.cpp` — object→leaf + CTR bin assignment at train time.
- `catboost-master/catboost/private/libs/algo/projection.h` — `TProjection` feature combinations / tensor CTRs (ORD-05).
- `catboost-master/catboost/private/libs/ctr_description/ctr_type.h`, `ctr_config.h` — `ECtrType` enum (the six types) + CTR config (ORD-03).
- `catboost-master/catboost/private/libs/options/cat_feature_options.cpp`, `cat_feature_options.h` — `simple_ctr`/`combinations_ctr`/`one_hot_max_size`/`max_ctr_complexity` defaults the isolating-params fixtures pin against (ORD-04/ORD-05, Claude's-discretion config pin).
- `catboost-master/catboost/private/libs/options/catboost_options.cpp`, `catboost_options.h` — `boosting_type` (Plain/Ordered) + auto-selection logic to deliberately override per fixture.
- `catboost-master/catboost/libs/model/ctr_provider.h`, `catboost-master/catboost/libs/model/online_ctr.h` — model-side CTR storage/application at inference (the `.cbm` `ctr_data` section to populate + apply).
- `catboost-master/catboost/libs/model/flatbuffers/ctr_data.fbs` — the CTR `.cbm` schema; Rust bindings already committed at `crates/cb-model/src/generated/ctr_data_generated.rs`.

### Oracle Harness (micro-harness precedent + fixtures, D-01/D-08/D-09)
- `crates/cb-oracle/generator/cityhash_oracle.cpp` — the Phase-2 standalone-C++-micro-harness PRECEDENT this phase's per-object harness follows (link only the needed upstream TUs, dump `.npy`).
- `crates/cb-oracle/fixtures/` — committed fixture root; new categorical/per-object fixtures land here. Existing input corpora: `crates/cb-oracle/fixtures/inputs/{numeric_categorical,explicit_categorical}`.
- `crates/cb-oracle/src/model_json.rs` — upstream `model.json` parser (already parses `ctr_data` hash_map per Phase-2 notes); reuse/extend for CTR round-trip + per-object comparison via the `compare_stage` ≤1e-5 API.

### CubeCL constraint (D-03 carried forward)
- `AGENTS.md` (project root) + `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — kernels (if any) use generics-float, live in `cb-backend`, read the manual before writing kernel code; load `cubecl_error_guideline.md` on any build error before fixing.

### Process / Project Rules
- `CLAUDE.md` (project root) — constraints, naming, mandatory source/test separation, latest-crate-versions rule.
- `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — Rust lint/error/test conventions and the source/test-separation rule.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cb-data/src/cat_hash.rs` — `calc_cat_feature_hash` (CityHash64 & 0xffffffff, first-seen perfect-hash bins) — the categorical hashing CTRs consume (D Carried-Forward). `Pool` already holds categorical columns.
- `crates/cb-train/src/{boosting.rs,tree.rs}` — plain boosting loop + oblivious trees; the one-hot first slice (D-04) extends this with categorical splits; ordered boosting (ORD-02) layers the permutation-prefix approximant on top.
- `crates/cb-compute/` — abstract `R: Runtime`/`F: Float` boundary + host-ordered reductions; new CTR/permutation orchestration finalizes sums here via `cb-core::sum_f64`.
- `crates/cb-backend/` — sole owner of `#[cube]` kernels (D-03); any order-independent CTR/permutation kernel attaches here.
- `crates/cb-model/src/generated/ctr_data_generated.rs` — **already-committed** FlatBuffers bindings for the `.cbm` CTR section; this phase populates + applies it (apply path in `cb-model/src/apply.rs`).
- `crates/cb-model/src/{cbm.rs,json.rs}` — `.cbm`/`model.json` save/load; extend to (de)serialize the `ctr_data` section.
- `crates/cb-core/` — `TFastRng64` (the permutation seed, D-03) + order-locked `sum_f64`/`sum_f32_in_f64`.
- `crates/cb-oracle/` — `.npy` fixture infra, `compare_stage` ≤1e-5 gate, frozen-corpus pattern, and the `cityhash_oracle.cpp` micro-harness precedent (D-01).

### Established Patterns
- D-07 isolating-params first slice + per-knob additive oracle (Phase 3) — the spine for D-04/D-05/D-06.
- Standalone C++ micro-harness for ground truth unreachable from Python (Phase-2 cityhash precedent) — generalized here to per-object internals (D-01).
- All float summation inside `cb-core::sum_f64` (D-08 CI grep) — applies to CTR running sums, priors, ordered approximants, leaf stats.
- Frozen committed fixtures; generators (Python + C++ micro-harness) run offline, never in CI (D-09).

### Integration Points
- `Pool` categorical columns → `cb-data` hashing → (one-hot | CTR) feature materialization → `cb-train` ordered/plain boosting → `cb-model` `.cbm`/`model.json` with `ctr_data` → `cb-model` apply path computes/looks-up CTRs at inference → `catboost-rs` Builder facade (categorical params surface through `CatBoostBuilder`).
- The `.cbm`/apply substrate from Phase 4 is extended (CTR section), not replaced — Phase 6 (multiclass leaf dims, advanced fstr) extends it further.
- The permutation/fold machinery (ORD-01) is the foundation Phase 6 ranking (group-aware) may reuse.

</code_context>

<specifics>
## Specific Ideas

- The user actioned the long-deferred **C++ instrumentation** decision decisively but via the **lightest viable mechanism** (standalone micro-harness, not a full build), explicitly weighing the documented disk constraint (D-01).
- The user chose the **deepest per-object oracle** (full stack: permutation + running CTR num/denom + per-iteration ordered approx, D-02) — consistent with treating this as the project's highest-risk parity slice, where the per-object leakage signature is the whole point.
- The user chose **maximum isolation in sequencing** (one-hot → Plain CTR → Ordered CTR → Ordered boosting, D-04/D-05/D-06) so a per-object break never lands in an entangled three-way interaction — directly mirroring the Phase-3 simplified-isolating-params discipline.
- The user chose **full breadth on CTR types** (all six, D-07), continuing the Phase 3–4 "prove the full math surface against narrow params" pattern.

</specifics>

<deferred>
## Deferred Ideas

- **Full catboost C++ build with in-tree instrumentation** — rejected for Phase 5 (disk infeasibility, D-01); the micro-harness supersedes it. Revisit only if isolated TU linking proves impractical.
- **Multiclass/ranking-aware ordered statistics, text/embedding features, advanced fstr, uncertainty, monotone constraints, feature selection, alternative grow policies** — Phase 6.
- **LossFunctionChange feature importance** — still deferred (Phase-4 D-12).
- **GPU CTR/permutation kernels** — Phase 7 (additive on the `cb-compute` seam).
- **Broader `.cbm` cross-version load tolerance for CTR models** (beyond 1.2.10) — later hardening pass.

None of the above are scope creep — all are explicitly later-phase items surfaced while bounding Phase 5.

</deferred>

---

*Phase: 5-Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice)*
*Context gathered: 2026-06-14*
