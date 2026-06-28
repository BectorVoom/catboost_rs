# Phase 2: Data Layer — Pool, Quantization & Reduction - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Realizes the `cb-data` crate — the leaf data layer every downstream tree depends on — and oracle-locks it to upstream CatBoost **1.2.10** (≤1e-5). Deliverables (DATA-01…DATA-08):

- **`Pool`** — raw float / categorical / text / embedding columns plus label, weights, group_id, subgroup_id, pairs, baseline.
- **`QuantizedPool`** — columnar SoA `u8`/`u16`/`u32` bin storage, immutable after build, buffers reusable across rounds.
- **GreedyLogSum** border selection — per-feature border set matching upstream exactly (NaN/duplicate columns, `<`/`<=` semantics), intermediate-oracle-validated.
- **`NanMode`** (Min/Max/Forbidden) missing-value handling; **categorical feature hashing** — both matching upstream.
- **Single audited deterministic reduction utility** — matches the C++ `double` accumulator type + summation order; the only summation primitive in the codebase.
- **Ingestion** — Arrow/Polars (Rust crates) with dtype/contiguity validation; copy-in path for training. (Python/NumPy zero-copy is Phase 8.)
- **Weights** — per-object / per-class weights and auto class weights (Balanced/SqrtBalanced).

This phase is judged purely on parity + memory-efficiency; it adds no training/algorithm code (that is Phase 3). It must NOT depend on CubeCL.

</domain>

<decisions>
## Implementation Decisions

### Pool ↔ QuantizedPool Design (DATA-01, DATA-02)
- **D-01:** **Two distinct types.** `Pool` (raw columns) and `QuantizedPool` (binned) are separate structs; quantization is an explicit transition `pool.quantize(&params) -> QuantizedPool`. Mirrors upstream's `TRawObjectsDataProvider` vs `TQuantizedObjectsDataProvider` split and keeps the raw-vs-quantized boundary sharp.
- **D-02:** **Owned now, zero-copy seam later.** Phase 2 `Pool` owns its column buffers (`Vec`-backed). The constructor is built around an ingestion trait so a borrowed/zero-copy NumPy view can plug in at Phase 8 *without reshaping `Pool`*. No lifetime-generic / `Cow` `Pool<'a>` yet — that complexity is not paid until there is a Python caller to benefit.
- **D-03:** **Immutable bins + caller-owned scratch.** `QuantizedPool` is built once and immutable; bin columns are stable read-only SoA. "Buffers reused across rounds" (DATA-02) means the **Phase 3 trainer reuses its OWN** histogram/gradient scratch over the fixed bins — `QuantizedPool` itself does not hand out mutable scratch. (Keeps data-layer ownership clean; the exact training-scratch reuse boundary is a Phase 3 concern.)

### Ingestion Scope Boundary (DATA-06)
- **D-04:** **Rust-native ingestion + trait seam in Phase 2; PyO3 in Phase 8.** Phase 2 builds ingestion from Rust-side sources behind an ingestion trait. The PyO3/NumPy zero-copy binding is wired in Phase 8 reusing this seam. No PyO3 dependency enters Phase 2.
- **D-05:** **Validated external paths = Arrow (`arrow-rs`) + Polars.** These are the ingestion sources wired and unit/oracle-tested in Phase 2, proving the zero-copy/validation path before Python exists. Raw owned-`Vec` construction remains the trivial primitive used by the Rust Builder API and the `.npy` oracle fixtures. (Polars is Arrow-backed — it may ride on the Arrow path; planning to confirm.)
- **D-06:** **Validation at the ingestion boundary, typed `CbError`.** dtype/contiguity/NaN-in-categorical checks run in the ingestion trait impls and return typed `thiserror` `CbError` variants with actionable messages — the same error taxonomy Phase 8 maps to specific Python exceptions (forward-compatible with PYAPI-05).

### Reduction Primitive (DATA-07)
- **D-07:** **Owned by `cb-core`.** The single deterministic reduction/summation utility lives in `cb-core` (already home to `CbError` + `TFastRng64`). Every crate depends on `cb-core`, so the "only summation primitive" is reachable everywhere without new dependency edges (avoids `cb-compute`/`cb-train` depending on `cb-data` just for summation).
- **D-08:** **Enforced by CI-grep backstop + convention** — mirrors the Phase 1 `anyhow`-ban pattern (D-14). A `scripts/check-*` grep fails CI on raw float `.sum()` / `.fold(0.0, +)` (etc.) in library crates; the convention is that all summation routes through the utility.
- **D-09:** **Exact C++ accumulation behavior pinned by research.** The researcher MUST determine the exact accumulator type (`f64`) AND summation order/strategy (naive sequential vs Kahan vs pairwise/blocked) upstream uses for the values Phase 2 reproduces (border statistics, weight sums) and the utility replicates it precisely. Do NOT assume naive sequential without confirming.

### Quantized Bin Storage (DATA-02)
- **D-10:** **Bin widths `u8`/`u16`/`u32`, matching upstream exactly.** Per-feature width: `u8` when the border count fits a byte, else `u16`, else `u32` for very high-cardinality quantization where upstream uses it. (Goes one width beyond DATA-02's literal `u8`/`u16` text — chosen deliberately for exact parity; research to confirm where >`u16` is actually reached within our supported `max_bin` range.)
- **D-11:** **Typed per-column enum** — a feature's bins are exposed as an enum `{ U8(&[u8]), U16(&[u16]), U32(&[u32]) }`; the Phase 3 histogram kernel matches on width. Zero widening cost preserves the memory/bandwidth win (memory efficiency is a first-class constraint); kernels handle the width arms.
- **D-12:** **Per-feature `Vec` SoA layout** — each feature column is its own contiguous buffer. Cache-friendly for per-feature histogram passes and simple to build; consistent with the immutable-after-build decision (D-03).

### Claude's Discretion
- Exact GreedyLogSum border algorithm, `<`/`<=` boundary assignment, NaN/duplicate-column handling — parity-dictated; researcher reads upstream and reproduces.
- `NanMode` (Min/Max/Forbidden) semantics and the categorical hash function — parity-dictated; match upstream exactly.
- Auto class-weight formulas (Balanced / SqrtBalanced) and per-object/per-class weight handling — parity-dictated; match upstream.
- The intermediate-oracle fixture schema for borders/quantization (which values to extract and compare) — left to research/planning, drawing inputs from the frozen corpus (D-11 of Phase 1).
- Whether Polars ingestion rides the Arrow code path or is separate — planning to decide.
- Concrete crate versions for `arrow`/`polars` — latest stable per CLAUDE.md.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project & Roadmap
- `.planning/PROJECT.md` — core value, constraints (memory efficiency first-class, thiserror/anyhow, latest crate versions), oracle strategy.
- `.planning/ROADMAP.md` § "Phase 2: Data Layer — Pool, Quantization & Reduction" — goal + 5 success criteria this phase is judged against.
- `.planning/REQUIREMENTS.md` — DATA-01 … DATA-08 requirement text and traceability.
- `.planning/phases/01-workspace-lint-discipline-oracle-harness/01-CONTEXT.md` — Phase 1 locked decisions this phase builds on (crate map D-05, oracle pin 1.2.10 D-07, fixture format/layout D-09/D-10, frozen input corpus D-11, determinism D-12, anyhow-ban enforcement pattern D-14).

### Vendored Reference & Oracle Source (catboost-master/, version 1.2.10)
- `catboost-master/catboost/libs/data/data_provider.{h,cpp}`, `data_provider_builders.{h,cpp}` — raw vs quantized data-provider split that `Pool`/`QuantizedPool` mirror (D-01).
- `catboost-master/catboost/libs/data/columns.{h,cpp}`, `composite_columns.{h,cpp}` — column storage shapes.
- `catboost-master/catboost/libs/data/quantized_features_info.{h,cpp}` — quantized feature metadata, `ENanMode`.
- `catboost-master/catboost/libs/data/quantization.cpp` — quantization driver.
- `catboost-master/catboost/libs/data/borders_io.cpp` — border serialization + NaN-mode I/O semantics.
- `catboost-master/catboost/private/libs/quantization/grid_creator.{h,cpp}`, `utils.cpp` — GreedyLogSum / border-grid construction (DATA-03).
- `catboost-master/catboost/libs/data/cat_feature_perfect_hash.{h,cpp}`, `cat_feature_perfect_hash_helper.{h,cpp}` — categorical feature hashing (DATA-05).
- `catboost-master/catboost/libs/data/baseline.{h,cpp}`, `ctrs.{h,cpp}` — baseline + CTR column context (CTR itself is Phase 5; read for Pool surface only).
- **Reduction order (DATA-07, D-09):** researcher to locate the upstream summation/accumulator used for border stats and weight sums (search `catboost-master/catboost/private/libs/algo*` and `catboost-master/library/cpp/` for the canonical `double`-accumulator pattern) and pin the exact order.

### Process / Project Rules
- `CLAUDE.md` (project root) — constraints, naming, mandatory source/test separation, latest-crate-versions rule.
- `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — Rust lint/error/test conventions and the source/test-separation rule.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cb-data/` — currently a stub (`lib.rs` doc comment only); Phase 2 fills it. Manifest already has `[lints] workspace = true` and the structural `anyhow` ban (no `anyhow` in `[dependencies]` — keep it that way, D-14).
- `crates/cb-core/` — home for the reduction primitive (D-07); already exports `CbError`/`CbResult` and `TFastRng64`. Add the summation utility + its `*_test.rs` here.
- `crates/cb-oracle/` — fixture infra is live: `fixture.rs` (read `.npy` via `ndarray-npy`), `compare.rs` (≤1e-5 abs-error gate), per-stage comparison API, and a frozen input corpus at `fixtures/inputs/{numeric_tiny,numeric_categorical,grouped_ranking}`. Phase 2 adds border/quantization **expected-output** fixtures referencing those inputs; the corpus generator (`generator/gen_inputs.py`) currently emits inputs only.

### Established Patterns
- Workspace deps are centralized in root `Cargo.toml` `[workspace.dependencies]` (thiserror 2, ndarray 0.17, ndarray-npy 0.10, serde, approx). `arrow`/`polars` will be added there at latest stable.
- Source/test separation is mandatory: dedicated `*_test.rs` files, no inline `#[cfg(test)]` in production modules (Phase 1 D-17); test-lint exemption via in-code `#![cfg_attr(test, allow(...))]`.
- Enforcement-by-CI-grep is the established belt-and-suspenders idiom (the `anyhow` ban) — the reduction-primitive guard (D-08) follows the same shape.
- Oracle determinism: pinned seed, `thread_count=1`, fixtures committed frozen (generator does not run in CI) — Phase 2 border/weight oracles inherit this.

### Integration Points
- `QuantizedPool`'s immutable SoA bins + the typed-width column accessor (D-11) are the contract the **Phase 3** histogram/gradient kernels consume over the generic `R: Runtime`/`F: Float` seam in `cb-compute`.
- The ingestion trait seam (D-04) is the attach point for **Phase 8** PyO3/NumPy zero-copy ingestion; its `CbError` taxonomy (D-06) feeds Phase 8's Python-exception mapping (PYAPI-05).
- The single reduction primitive (D-07) becomes a process-wide invariant: every later phase (training sums, metric accumulation, SHAP) must route through it.

</code_context>

<specifics>
## Specific Ideas

- Deliberate deviation from the literal requirement text: DATA-02 says "`u8`/`u16`", but the user chose `u8`/`u16`/`u32` to match upstream exactly (D-10). Planning should treat exact upstream parity as the tiebreaker over the requirement's shorthand.
- The user wants the "only summation primitive" guarantee enforced structurally (CI grep), not just by convention — consistent with how Phase 1 made the `anyhow` ban structural rather than advisory.
- The user scoped Phase 2 ingestion to Arrow + Polars (Rust crates) explicitly, keeping NumPy/PyO3 out until Phase 8 — a clean separation of the Rust data path from the Python boundary.

</specifics>

<deferred>
## Deferred Ideas

- **PyO3 / NumPy zero-copy ingestion** — deferred to Phase 8 (PYAPI-04/PYAPI-06), reusing the Phase 2 ingestion trait seam (D-04). Not scope creep; an explicit later-phase attach point.
- **CTR / ordered target-statistic columns** — `Pool` carries the columns CTRs will use, but CTR computation itself is Phase 5 (ORD-03). Phase 2 only stores the raw categorical data + hashing.
- **GPU bin storage / kernels** — `QuantizedPool` is CubeCL-free; GPU consumption of bins is Phase 7, additively on the generic seam.

None of the above are scope creep — all are explicitly later-phase items surfaced while clarifying Phase 2 boundaries.

</deferred>

---

*Phase: 2-Data Layer — Pool, Quantization & Reduction*
*Context gathered: 2026-06-13*
