# Phase 9: Online HNSW Estimated-Feature Parity - Context

**Gathered:** 2026-06-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Port `catboost-master/library/cpp/online_hnsw/base` (plus the `NHnsw` routines it
calls) to Rust **bit-for-bit** so the KNN estimated-feature calcer returns
upstream-identical neighbor IDs, closing the XOR per-stage ≤1e-5 oracle gate that
the brute-force-exact calcer (Phase 6.5 A2/D-05) cannot. Both calcer flavors must
match upstream: the online incremental path (`TKNNUpdatableCloud`,
`AddItem`→`GetNearestNeighbors`) and the offline whole-set apply path (`TKNNCloud`).

**Locked by ROADMAP/REQUIREMENTS (do NOT re-derive):** FEAT-07 and SC-1..SC-4.
Distance is `TL2SqrDistance<float>` (squared L2, `float` vectors). Build options:
`MaxNeighbors=32`, `SearchNeighborhoodSize=300`, `LevelSizeDecay/NumVertices =
AUTO_SELECT(0)`; calcer constructs with `CloseNum=k` and search size `300`. Reference
ONLY the vendored C++ — no sklearn-ann / annoy / faiss / nmslib (different algorithms
cannot be bit-matched). The `text_embedding_xor/` fixture is **frozen** — no regeneration.

**Out of scope:** any new ANN algorithm, GPU HNSW, or changes to the boosting loop
(the online estimated column is a SINGLE static column over `S` for train + offline
post-hoc apply — proven in the `knn-estimated-feature-is-online-hnsw` memory; the
"core boosting-loop rework" hypothesis is DISPROVEN, do not revisit).

</domain>

<decisions>
## Implementation Decisions

### D-01 — Construction-order fidelity model: research gate FIRST (Area 1)
The ROADMAP and the `knn-estimated-feature-is-online-hnsw` memory both frame the
crux as "RNG-driven build order." A discussion-time grep of
`library/cpp/online_hnsw/base/` AND `library/cpp/hnsw/` found **no RNG** (no
`TFastRng`/`Shuffle`/`RandomNumber`; level growth is via `AddNewLevelIfLastIsFull`,
deterministic by fullness, not random level-draw). **However, the user chose to KEEP
the RNG framing as a research gate** rather than commit to the deterministic model on
the strength of a grep.

**Mandatory research gate (blocks the fidelity commitment):** the researcher must
conclusively prove or disprove RNG/seed involvement across the *full* call path —
`online_hnsw/base/`, every `NHnsw` routine it calls (`GetLevelSizes`,
`FindApproximateNeighbors`, neighbor-trim/`Retrim`), and any seed entering at the
`catboost/private/libs/embedding_features/knn.{h,cpp}` call site — BEFORE the planner
locks the bit-exact strategy.

- **If RNG is confirmed:** replicate the seed source + draw order first (the original crux).
- **If RNG is refuted (grep's indication):** the parity target becomes deterministic
  replication of (a) exact `AddItem` insertion order (online: per-fold estimate over
  permutation `S = create_shuffled_indices(n, seed)`; offline `TKNNCloud`: pool/document
  order, `learnPermutation = Nothing()`) and (b) byte-identical distance tie-breaking
  (`TDistanceLess`, search-heap pop order, neighbor-trim), plus f32 squared-L2
  accumulation order. Transcribe verbatim from the C++.

The instrumented trainer (D-04) is the tool that settles this conclusively.

### D-02 — Port scope: FULL transcription now (Area 2)
Port **all** listed files of `online_hnsw/base/` in one pass — including the
serialization surface (`index_reader.cpp`, `index_writer.cpp`, `index_snapshot_data.h`,
`index_data.h`, `ConstructIndexData`/`ExpectedSize`/`WriteIndex`) — even though no SC
exercises serialization. Plus the required `NHnsw` routines from
`library/cpp/hnsw/index_builder/build_routines.{h,cpp}` (`GetLevelSizes`,
`FindApproximateNeighbors`). Rationale: complete the `.cbm` save/load path for a trained
KNN cloud once so `online_hnsw` is never revisited. (~832 LOC base + the `NHnsw` routines.)

### D-03 — Existing exact calcer: KEEP behind a flag (Area 3)
HNSW becomes the **default** (parity) path for both online and offline calcers. The
current brute-force-exact `cb_compute::KnnCalcer` (A2/D-05) is retained, selectable via
a builder/param flag, for users who explicitly want exact NN.
- Default MUST be HNSW so the oracle gate (D-04, SC-3) exercises the parity path.
- This introduces a config surface upstream does NOT have — planner must name the flag
  clearly and document that exact-mode is non-parity (returns `{0,2,4}`-style exact
  neighbors, diverging from upstream `{1,3,4}`).
- The exact path doubles as a cross-check: the exact-vs-approximate divergence is itself
  evidence for SC-1.

### D-04 — Verification: REBUILD instrumented trainer UP FRONT (Area 4)
Reprovision clang-18 and rebuild the instrumented catboost-1.2.10 trainer at phase start
(recipe: `catboost-instrumented-trainer-build` memory + `crates/cb-oracle/generator/
build_instrumented_trainer.sh`; the trainer is NO LONGER in `/tmp` — full provision
needed; conan/ninja persist in `~/.local/bin`). This gives:
- Regeneration of the full `knn_neighbors` evidence corpus to diff any intermediate
  query/prefix during the bit-exact debug loop.
- The conclusive RNG prove/disprove for D-01's research gate.
The frozen `text_embedding_xor/` fixture remains the SC-3 oracle gate (no regeneration);
the instrumented trainer is for *evidence/debugging*, not for re-baking the fixture.

### Claude's Discretion
- **Crate/module placement of the port** — not asked; planner/researcher decide. Natural
  homes: a module under `crates/cb-compute/` (where `KnnCalcer`/`IncrementalCloud` already
  live) or a dedicated `cb-hnsw` crate. This is a pure-CPU compute port — the
  `phase75-grow-loop-outcome` landmine ("never add cb-train dep to cb-backend") does not
  apply, but keep the port free of backend/cubecl symbols (mirror MODEL-02's discipline).
- **f32 accumulation/comparator details** — fall out of D-01's research; transcribe to match.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### C++ port source (the parity target — transcribe verbatim)
- `catboost-master/library/cpp/online_hnsw/base/` — FULL port (D-02): `dynamic_dense_graph.{h,cpp}`, `index_base.{h,cpp}`, `item_storage_index.{h,cpp}`, `build_options.{h,cpp}`, `index_data.h`, `index_reader.{h,cpp}`, `index_writer.{h,cpp}`, `index_snapshot_data.h`
- `catboost-master/library/cpp/hnsw/index_builder/build_routines.{h,cpp}` — `GetLevelSizes` (line 4), `FindApproximateNeighbors` (line 430); the `NHnsw` routines `index_base.h` depends on (search at line 184, level sizing at lines 37–49)
- `catboost-master/catboost/private/libs/embedding_features/knn.{h,cpp}` — call site: `TKNNUpdatableCloud` / `TKNNCloud` `GetNearestNeighbors`, `AddItem` (insertion order), `TOnlineHnswBuildOptions({CloseNum=k, 300})`, serialization (`ConstructIndexData`/`WriteIndex`)

### Rust integration points (where the port wires in)
- `crates/cb-compute/src/embedding_calcers.rs` — existing brute-force-exact `KnnCalcer` + `IncrementalCloud` (D-03: keep behind flag; HNSW default)
- `crates/cb-train/src/estimated/online_embedding.rs` — KNN online/offline seam: `online_knn_prefix`, `offline_knn_features`, `knn_feature_count` (the `IOnlineFeatureEstimator` path; both flavors)
- `crates/cb-train/src/estimated/estimated_features.rs` — estimated-feature column layout
- `crates/cb-oracle/tests/text_embedding_end_to_end_oracle_test.rs` — the RED-on-success per-stage gate to FLIP to passing ≤1e-5 (SC-3/SC-4); honest oracle, no `#[ignore]`, no weakened tolerance, KNN vote border serializes `0.5`

### Evidence / trainer rebuild
- `crates/cb-oracle/generator/build_instrumented_trainer.sh` + `instrument_text_pipeline.cpp` + `instrument_text_pipeline_README.md` — instrumented trainer recipe (D-04)
- `.planning/todos/pending/estimated-feature-grid-parity.md` — full root-cause history (instrumented-trainer verdict, disproven hypotheses, the cloud-B doc6 → upstream `{1,3,4}` vs exact `{0,2,4}` proof)
- Memory `knn-estimated-feature-is-online-hnsw` and `catboost-instrumented-trainer-build` (toolchain provision)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cb_compute::KnnCalcer` / `IncrementalCloud` — current exact calcer; becomes the
  flagged fallback (D-03) and the structural anchor for where HNSW slots in.
- `online_knn_prefix` / `offline_knn_features` / `knn_feature_count`
  (`online_embedding.rs`) — the online (ordered-boosting leaf path) and offline
  (whole-set apply) seams already exist; the port swaps the neighbor source behind them.
- The frozen `text_embedding_xor/` fixture + the existing RED-on-success oracle test —
  reused as-is; the test flips green on success.

### Established Patterns
- KNN is wired as an `IOnlineFeatureEstimator` exactly like LDA (read-before-update
  online estimate → vote distribution `{0,1,…,k}`, first border `0.5`). The `0.5` border
  VALUE is already correct (06.5 quick task 260619-cpr); only the *neighbor set* diverges.
- MODEL-02 discipline: pure-CPU compute modules import no backend/cubecl symbol — apply
  the same to the HNSW port.

### Integration Points
- The neighbor source is the ONLY thing that changes behind the existing seam — the
  boosting loop, permutation handling, and border quantization stay (single static online
  column over `S`; offline post-hoc apply; `fold_count=1`, no cycling).

</code_context>

<specifics>
## Specific Ideas

- SC-1 concrete anchor: cloud-B query doc6 over prefix `{14,15,0,7,4}` must yield
  upstream's HNSW `{1,3,4}` (two cloud-A, wrong-vs-exact), NOT the brute-force-exact
  `{0,2,4}`. Prefix neighbors agree p0–p4 as sets, diverge from p5. Reproduce the
  divergence-from-exact, not merely "close."
- Class-vote ordering must match upstream: feat0 = class-1 vote (upstream), not Rust's
  `[class0, class1]`.
- KNN vote border serializes `0.5` (not `1.5`) — already achieved, must not regress.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Crate placement and f32/comparator details
are in-scope Claude's-discretion items, captured under Implementation Decisions, not
deferred.)

</deferred>

---

*Phase: 9-online-hnsw-estimated-feature-parity*
*Context gathered: 2026-06-23*
