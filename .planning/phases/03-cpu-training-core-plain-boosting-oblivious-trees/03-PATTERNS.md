# Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 19 new + 3 modified (3 stub `lib.rs`, root `Cargo.toml`, generator)
**Analogs found:** 19 / 22 with an in-repo analog; 3 partial (first-of-kind CubeCL kernels + abstract trait surface)

> Reading note for the planner: in this domain the "library" is upstream CatBoost itself (`catboost-master/` 1.2.10). The analogs below cover the **engineering scaffolding** (crate layout, lint exemption, error/result types, reduction routing, oracle harness wiring, test layout). The **algorithm bodies** are transcribed from the vendored C++ named in 03-CONTEXT.md canonical_refs / 03-RESEARCH.md — those are not in-repo analogs.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cb-compute/src/lib.rs` (modify) | lib-root | re-export | `crates/cb-core/src/lib.rs` | exact |
| `crates/cb-compute/src/runtime.rs` | trait/boundary | request-response | *(none — first abstract `R`/`F` trait)* | partial (trait-bound style only) |
| `crates/cb-compute/src/histogram.rs` | service | transform (ordered reduce) | `crates/cb-data/src/borders.rs` | role+flow match |
| `crates/cb-compute/src/score.rs` | service | transform | `crates/cb-data/src/borders.rs` | role+flow match |
| `crates/cb-compute/src/leaf.rs` | service | transform (ordered reduce) | `crates/cb-data/src/borders.rs` | role+flow match |
| `crates/cb-compute/src/loss.rs` | utility | transform (elementwise) | `crates/cb-data/src/nan_mode.rs` (`bin_of`) | role match |
| `crates/cb-compute/src/*_test.rs` | test | — | `crates/cb-core/src/reduction_test.rs` (unit) | exact |
| `crates/cb-backend/src/lib.rs` (modify) | lib-root / feature dispatch | config | self (`SelectedRuntime` arms) | exact (extend in place) |
| `crates/cb-backend/src/kernels.rs` | kernel | streaming (elementwise/scatter) | *(none — first `#[cube]` kernel)* | partial (CubeCL manual only) |
| `crates/cb-backend/src/cpu_runtime.rs` | provider (trait impl) | request-response | `crates/cb-data/src/ingest/owned.rs` (trait impl) | role match |
| `crates/cb-backend/src/kernels_test.rs` | test | — | `crates/cb-core/src/reduction_test.rs` | exact |
| `crates/cb-train/src/lib.rs` (modify) | lib-root | re-export | `crates/cb-core/src/lib.rs` | exact |
| `crates/cb-train/src/boosting.rs` | service (orchestrator) | event-driven (iteration loop) | `crates/cb-data/src/quantize.rs` (driver) | role match |
| `crates/cb-train/src/tree.rs` | service | transform (greedy search) | `crates/cb-data/src/borders.rs` (greedy + tie-break) | exact (greedy+tie-break) |
| `crates/cb-train/src/leaf.rs` (or in cb-compute) | service | transform (ordered reduce) | `crates/cb-data/src/borders.rs` | role+flow match |
| `crates/cb-train/src/bootstrap.rs` | service | transform (RNG-sequenced) | `crates/cb-core/src/rng.rs` (consumer style) | flow match |
| `crates/cb-train/src/overfit.rs` | service (state machine) | event-driven | *(none — closest is greedy state in `borders.rs`)* | partial |
| `crates/cb-train/src/autolr.rs` | utility | transform (scalar formula) | `crates/cb-data/src/weights.rs` | role match |
| `crates/cb-train/src/*_test.rs` | test | — | `crates/cb-core/src/reduction_test.rs` | exact |
| `crates/cb-train/tests/*_oracle_test.rs` | test (integration oracle) | — | `crates/cb-data/tests/borders_oracle_test.rs` | exact |
| `crates/cb-oracle/src/model_json.rs` (new — `model.json` parser) | utility (parser) | file-I/O | `crates/cb-oracle/src/fixture.rs` | role+flow match |
| `crates/cb-oracle/generator/gen_fixtures.py` (extend) | config/generator | batch | `crates/cb-oracle/src/bin/write_skeleton.rs` (one-off generator) | role match |
| `*/Cargo.toml` (cb-backend add cubecl; root add workspace deps) | config | — | `crates/cb-data/Cargo.toml` | exact |

## Pattern Assignments

### `crates/cb-compute/src/lib.rs` (lib-root, re-export) — MODIFY

**Analog:** `crates/cb-core/src/lib.rs` (lines 1-32) — the canonical module-tree + lint-exemption + `pub use` + `#[cfg(test)] mod *_test;` layout.

The stub today is a single doc comment with the lint attr already on line 1. Extend it to the `cb-core` shape: a module-level doc, the test-lint exemption (already present), `mod runtime; mod histogram; mod score; mod leaf; mod loss;`, then `pub use` of the public trait/fn surface, then the gated `mod *_test;` block.

**Lint-exemption header (KEEP AS-IS — already atop the stub, matches `cb-core/src/lib.rs:8-16`):**
```rust
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
```

**Module/re-export/test-mod block (copy shape from `cb-core/src/lib.rs:18-31`):**
```rust
mod error;
mod reduction;
mod rng;

pub use error::{CbError, CbResult};
pub use reduction::{sum_f32_in_f64, sum_f64};
pub use rng::TFastRng64;

#[cfg(test)]
mod error_test;
```
> `cb-train/src/lib.rs` and `cb-backend/src/lib.rs` follow the identical shape — same header (already present), same `mod`/`pub use`/`#[cfg(test)] mod` ordering.

---

### `crates/cb-compute/src/runtime.rs` (abstract `R: Runtime` / `F: Float` traits) — NEW

**Analog:** None in-repo — this is the first abstract trait boundary. (D-03: `cb-compute` MUST NOT depend on `cubecl`; manifest at `crates/cb-compute/Cargo.toml:10-13` is intentionally dependency-free except future `cb-core`/`cb-data`.) Closest stylistic precedent is the `IngestSource` trait seam referenced in `crates/cb-data/src/lib.rs:5-8` (trait seam that a later backend plugs into) and the `FeatureKind`/`ColumnBins` enum-dispatch in `quantized_pool.rs`.

**Pattern to follow (from CONTEXT D-04 + RESEARCH "Recommended Project Structure"):** coarse domain-level trait ops — `build_histograms`, `compute_gradients`, `eval_splits` — generic over `F: Float`. `cb-backend` implements this trait; `cb-compute` holds the host orchestration that finalizes ordered sums. Return fallible work as `cb_core::CbResult<_>` (see Shared Pattern B). Do NOT name `cubecl` here.

---

### `crates/cb-compute/src/histogram.rs` / `score.rs` / `leaf.rs` (host-side ordered reductions) — NEW

**Analog:** `crates/cb-data/src/borders.rs` — the established "transcribe-upstream-C++-with-f64-discipline-and-route-every-float-sum-through-`cb_core::sum_f64`" service module.

**Summation-routing pattern (COPY EXACTLY — `borders.rs:36, 53-56`):** every parity-critical fold goes through `cb_core::sum_f64`; never a raw `.sum()` / `.fold(0.0, +)` (CI-grep banned, D-08).
```rust
use cb_core::sum_f64;

// Where this module folds floats, route through the sanctioned reduction
// primitive rather than any raw fold (D-07 / D-08).
fn total_object_weight(weights: &[f64]) -> f64 {
    sum_f64(weights)
}
```

**Source-of-truth doc-block pattern (COPY SHAPE — `borders.rs:1-35`):** open each module with a `# Source of truth` doc citing the exact upstream file + line range (e.g. `online_predictor.h:112-178` for `CalcAverage`/`ScaleL2Reg`, `score_calcers.cpp:20-49` for `AddLeafPlain`), an `# f64 vs f32 discipline` note, and a `# Summation routing (D-07 / D-08)` note. The formula bodies come from RESEARCH §"Code Examples" / canonical_refs, NOT from any in-repo analog.

**Guard-not-panic pattern:** `CalcAverage` already guards `count > 0` (RESEARCH Security Domain); mirror `borders.rs`' fallible style — return `0.0` / `CbResult` rather than dividing-by-zero or indexing unchecked (deny-lints `indexing_slicing`/`panic`).

---

### `crates/cb-compute/src/loss.rs` (RMSE/Logloss der1/der2) — NEW

**Analog:** `crates/cb-data/src/nan_mode.rs` — small pure elementwise transcription module (`bin_of`, `nan_bin`) with one upstream-cited formula per fn.

**Pattern:** one `#[must_use]` fn per derivative with the upstream cite inline (RESEARCH Code Examples: RMSE der1 = `target - approx`, der2 = `-1.0`; Logloss `p = sigmoid(approx)`, der1 = `target - p`, der2 = `-p*(1-p)` — `error_functions.{h,cpp}`). These are elementwise scalars; the per-object loop that calls them lives in the kernel (`cb-backend`), the parity SUM is host-side (D-02).

---

### `crates/cb-backend/src/lib.rs` (`SelectedRuntime` arms) — MODIFY

**Analog:** itself — `crates/cb-backend/src/lib.rs:7-22`. The `#[cfg(feature = "cpu")] pub type SelectedRuntime = ();` placeholder is replaced by the CubeCL `CpuRuntime` alias **in the existing cpu arm only**; the `wgpu`/`cuda`/`rocm` arms stay `= ()` (Phase 7, D-02 — compile-time selection, no runtime `match`).

**Existing cfg-arm structure to preserve (lines 9-22):**
```rust
#[cfg(feature = "cpu")]
pub type SelectedRuntime = ();   // → cubecl::cpu::CpuRuntime under feature="cpu"

#[cfg(all(feature = "wgpu", not(feature = "cpu")))]
pub type SelectedRuntime = ();   // unchanged (Phase 7)
// ... cuda, rocm arms unchanged ...
```

---

### `crates/cb-backend/src/kernels.rs` (`#[cube]` gradient/hessian + histogram-scatter) — NEW

**Analog:** None in-repo — first `#[cube]` kernels. Reference is the CubeCL manual (AGENTS.md mandate): `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` and `Cubecl_generics.md`; on ANY build error load `cubecl_error_guideline.md` BEFORE fixing (CLAUDE.md/AGENTS.md hard rule).

**Pattern (from RESEARCH Pattern 2, generics-float per AGENTS.md):**
```rust
#[cube(launch)]
fn gradient_kernel<F: Float>(approx: &Array<F>, target: &Array<F>, der1: &mut Array<F>) {
    if ABSOLUTE_POS < approx.len() {
        der1[ABSOLUTE_POS] = target[ABSOLUTE_POS] - approx[ABSOLUTE_POS]; // RMSE der1
    }
}
```
**Hard constraint (D-02/D-05, RESEARCH Anti-Patterns):** kernels do ONLY order-independent elementwise / scatter work. **Never** sum inside a kernel for a parity-critical total — the host reduces via `cb_core::sum_f64`. Kernels must use `generics-float` (no hard-coded `f32`/`f64`).

> The Open-Q2 build spike (a minimal `#[cube]` gradient kernel compiling + running on `CpuRuntime` under the deny-lints) should be the FIRST cb-backend task before wiring into the loop.

---

### `crates/cb-backend/src/cpu_runtime.rs` (`impl cb_compute::Runtime for CpuRuntime`) — NEW

**Analog:** `crates/cb-data/src/ingest/owned.rs` — the established "implement a `cb-*` trait for a concrete type" pattern (it implements the `IngestSource` seam). cpu_runtime.rs implements `cb-compute`'s `Runtime` trait by launching the `kernels.rs` kernels and returning per-object/per-bin buffers (NOT reduced) for the host to fold. Data transfer to/from CubeCL buffers uses `bytemuck::Pod` (`Bytes::from_elems`) per RESEARCH Standard Stack.

---

### `crates/cb-train/src/boosting.rs` (the iteration loop) — NEW

**Analog:** `crates/cb-data/src/quantize.rs` — the existing top-level driver that orchestrates per-feature work and returns a `QuantizedPool`. boosting.rs is the analogous driver: init approx (`boost_from_average`), loop `iterations` calling `compute_gradients` → bootstrap → `GreedyTensorSearchOblivious` → leaf estimation → `approx += learning_rate * delta` → eval/overfit, returning the trained oblivious-tree model.

**Fallible-driver pattern:** return `cb_core::CbResult<Model>`; surface degenerate inputs (empty leaf, `count==0`, `depth` cap) as `CbError`, never `unwrap`/panic (RESEARCH Security Domain threat table).

---

### `crates/cb-train/src/tree.rs` (`GreedyTensorSearchOblivious`, tie-break) — NEW

**Analog:** `crates/cb-data/src/borders.rs` — the closest existing greedy-search-with-exact-tie-break precedent. `borders.rs` already reproduced a max-heap greedy split (and the project already fought the WR-01 STL `priority_queue` tie-break in Phase 2). tree.rs reproduces the SAME class of landmine.

**Tie-break pattern (CONTEXT D-discretion / RESEARCH Pitfall 1, `greedy_tensor_search.cpp:948-966`):** strict `gain > bestGain`, first-wins, iterating candidates in the EXACT upstream nested order (feature index ascending, border ascending within feature). Do NOT sort by score; do NOT use `>=`.
```rust
// score accumulation routes bucket stats through cb_core::sum_f64 first;
// then strict first-wins selection:
let mut best_gain = f64::NEG_INFINITY;     // MINIMAL_SCORE
for candidate in candidates_in_upstream_order {
    let gain = (candidate.score - score_before_split) * feature_weight;
    if gain > best_gain { best_gain = gain; best = Some(candidate); }  // STRICT >
}
```

---

### `crates/cb-train/src/bootstrap.rs` (Poisson/Bayesian/Bernoulli/MVS/No) — NEW

**Analog:** `crates/cb-core/src/rng.rs` (the `TFastRng64` API + its `from_seed`/`advance`/`uniform` surface) — bootstrap is the first real consumer of the RNG. No existing consumer in-repo, so copy the *call discipline* from the RNG's own doc contract.

**Draw-order pattern (RESEARCH Pitfall 4, `tensor_search_helpers.cpp:322-560`):** per-1000-element-block reseed — `TFastRng64::from_seed(rand_seed + block_idx)` then `.advance(10)` per block; the block structure is part of the contract even at `thread_count=1`. First slice uses `bootstrap_type=No` (all weights 1.0, zero draws) — implement that arm first.

> The normal draw for `random_strength` (TRAIN-05) needs a `StdNormalDistribution` port (Box-Muller/Marsaglia-polar over `TFastRng64::gen_rand` → a `GenRandReal1` helper). No in-repo analog; transcribe from `util/random/normal.h`. This belongs to the TRAIN-05 additive slice, not the first slice.

---

### `crates/cb-train/src/overfit.rs` (Wilcoxon/IncToDec/Iter) — NEW

**Analog:** None strong — closest is the stateful greedy loop in `borders.rs`. It is a pure host state machine over eval-metric deltas (`overfitting_detector.cpp:127-174`). Use `CbError`/`CbResult` for fallible paths; route any metric folds through `cb_core::sum_f64`.

---

### `crates/cb-train/src/autolr.rs` (`TAutoLRParamsGuesser`) — NEW

**Analog:** `crates/cb-data/src/weights.rs` — a pure scalar-formula utility module (`balanced_class_weights` etc.) with named constants. autolr.rs is a coefficient-table + `exp/log` formula keyed by `(TargetType, CPU, useBestModel, boostFromAverage)` (RESEARCH Code Examples, `options_helper.cpp:116-262`). Mirror `weights.rs`' `#[must_use]` pure-fn + `const` table style.

---

### `crates/cb-oracle/src/model_json.rs` (`model.json` parser) — NEW

**Analog:** `crates/cb-oracle/src/fixture.rs` — the established serde-`Deserialize` + fallible loader pattern (`FixtureConfig`, `load_config`). model_json.rs adds `Deserialize` structs for the verified schema and extractor fns returning `Vec<f64>` for `compare_stage(Stage::Splits|LeafValues, …)`.

**Verified schema (from `crates/cb-oracle/fixtures/regression_skeleton/model.json`):** top-level keys `["features_info", "model_info", "oblivious_trees", "scale_and_bias"]`. Each `oblivious_trees[i]` has `leaf_values: [f64]`, `leaf_weights: [f64]`, `splits: [{border, float_feature_index, split_index, split_type}]`. `scale_and_bias = [1, [bias]]`. `features_info.float_features[].borders`.

**Loader pattern (COPY SHAPE — `fixture.rs:28-47`):**
```rust
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ObliviousTree {
    pub leaf_values: Vec<f64>,
    pub splits: Vec<SplitJson>,
}
pub fn load_model_json(path: &Path) -> Result<ModelJson, OracleError> {
    let contents = std::fs::read_to_string(path)?;   // OracleError::Io
    Ok(serde_json::from_str(&contents)?)             // OracleError::Json
}
```
> Add the new export to `crates/cb-oracle/src/lib.rs` `pub use` block (mirrors lines 23-25) and a `mod model_json;` + `#[cfg(test)] mod model_json_test;` (lines 19-30).

---

### `crates/cb-oracle/generator/gen_fixtures.py` (extend) — MODIFY

**Analog:** `crates/cb-oracle/src/bin/write_skeleton.rs` — the existing one-off "generate-and-commit, never run in CI" generator (its header documents the run-once-commit discipline + the file-scope lint allow for a generator). gen_fixtures.py extends to emit per-slice training oracles via the pinned Python `catboost==1.2.10`: `save_model(format='json')` → `model.json`, `staged_predict` → `staged.npy`, plus a `binclf_skeleton` (Logloss) scenario mirroring `regression_skeleton`.

**Pinned params per A1/A2 (must be explicit in generated configs):** `bootstrap_type=No`, `random_strength=0`, fixed `l2_leaf_reg`/`depth`/`learning_rate`/`iterations`, explicit `boost_from_average` (true for RMSE, false for Logloss), `leaf_estimation_iterations=1`, `score_function=L2` (first slice). Determinism: pinned seed, `thread_count=1` (matches `FixtureConfig` at `fixture.rs:28-36`).

---

### Test files (`*_test.rs` unit + `tests/*_oracle_test.rs` integration) — NEW

**Unit-test analog:** `crates/cb-core/src/reduction_test.rs` — dedicated sibling file, gated by `#[cfg(test)] mod reduction_test;` in `lib.rs`. NO inline `#[cfg(test)]` blocks (CLAUDE.md/AGENTS.md hard rule). The crate's `lib.rs` already carries the `#![cfg_attr(test, allow(...))]` exemption.

**Integration-oracle-test analog:** `crates/cb-data/tests/borders_oracle_test.rs` (lines 1-101) — the canonical "load committed fixture, compute Rust actual, gate via `compare_stage`" integration test.

**Fixture-path + gate pattern (COPY EXACTLY — `borders_oracle_test.rs:14, 24-30, 78-80`):**
```rust
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)] // mirrors per_stage_oracle_test.rs:9

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("cb-oracle").join("fixtures").join(rel)
}
// ... compute actual via cb-train ...
compare_stage(Stage::StagedApprox, &expected, &actual)
    .unwrap_or_else(|e| panic!("staged approx diverged: {e:?}"));
```
> Integration tests live under `tests/` (not `src/`) so they may depend on `cb-oracle` as a dev-dependency — same reasoning documented at `borders_oracle_test.rs:11-13`.

## Shared Patterns

### A. Crate manifest + dependency declaration
**Source:** `crates/cb-data/Cargo.toml:1-23` (full shape) and `crates/cb-oracle/Cargo.toml:18-21` (dev-dep edge).
**Apply to:** all three crate manifests + root `Cargo.toml`.
- Every library crate carries `[lints] workspace = true` (lines 7-8) — never a per-crate clippy override (forbidden when `lints.workspace = true`).
- `anyhow` is structurally banned from library crates (the `# NOTE: anyhow is intentionally absent (D-14)` comment is present in every manifest and must stay). `anyhow`/`approx` permitted ONLY in `[dev-dependencies]`.
- Workspace deps use `.workspace = true` (line 13); local crates use `{ path = "../cb-core" }` (line 12).
- **New for Phase 3:** add to root `[workspace.dependencies]` (root `Cargo.toml:18-29` block): `cubecl = { version = "0.10.0", features = ["cpu"] }` and `bytemuck = { version = "1", features = ["extern_crate_std"] }`. Wire `cubecl`/`bytemuck` ONLY into `crates/cb-backend/Cargo.toml` (D-03 — never `cb-compute`). `cb-backend`'s manifest comment at lines 11-14 currently says "NO cubecl dependency yet" — update that comment when adding it.

### B. Error / Result type
**Source:** `crates/cb-core/src/error.rs:1-78` (the `thiserror`-derived `CbError` enum + `CbResult<T>` alias).
**Apply to:** all fallible APIs in `cb-compute`, `cb-backend`, `cb-train`.
```rust
use cb_core::{CbError, CbResult};
```
- `thiserror`-derived, `Clone + PartialEq + Eq` (no `#[from]` of non-Clone externals — stringify instead, per the `Ingestion` variant doc at lines 67-77).
- Add new Phase-3 variants (e.g. degenerate-leaf, depth-cap) to this enum — downstream `match`es are documented to tolerate added variants (lines 13-15).
- `unwrap()`/`expect()`/`panic!`/unchecked indexing are deny-lints in production — return `CbError`, guard counts (cf. `CalcAverage`'s `count > 0` guard).

### C. Float-summation routing (parity contract)
**Source:** `crates/cb-core/src/reduction.rs:1-51` (`sum_f64`, `sum_f32_in_f64`); consumer pattern at `crates/cb-data/src/borders.rs:36, 53-56`.
**Apply to:** EVERY parity-critical sum — histogram bin totals, leaf der/weight sums, score accumulation, metric accumulation, bootstrap weight sums.
```rust
use cb_core::sum_f64;          // never `.sum()` / `.fold(0.0, +)` on floats
let bin_total = cb_core::sum_f64(&per_object_contributions_in_object_order);
```
- Sequential left-to-right `f64` fold, `thread_count==1`, NO Kahan/pairwise. Order is the contract (D-02/D-05).
- Enforced by CI grep `scripts/check-no-raw-float-sum.sh` (D-08) — a raw float fold anywhere in a library crate fails CI.

### D. Test-lint exemption + source/test separation
**Source:** `crates/cb-core/src/lib.rs:8-16` (lib-root exemption) and `crates/cb-oracle/tests/per_stage_oracle_test.rs:9` (integration-test file-scope allow).
**Apply to:** every crate `lib.rs` (already present atop all three Phase-3 stubs) and every test file.
- `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]` at `lib.rs` scope; bare `#![allow(...)]` at integration-test-file scope.
- NO inline `#[cfg(test)] mod tests {}` in production source — dedicated `<name>_test.rs` sibling, declared `#[cfg(test)] mod <name>_test;` in `lib.rs`.

### E. Per-stage oracle gate
**Source:** `crates/cb-oracle/src/compare.rs:9-96` (`Stage` enum + `compare_stage`); usage at `crates/cb-data/tests/borders_oracle_test.rs:78-80`.
**Apply to:** every cb-train oracle test.
- `Stage::{Splits, LeafValues, StagedApprox, Predictions}` already exist (lines 10-21) — no new stages needed for Phase 3.
- `compare_stage(stage, expected, actual)` gates at fixed `1e-5` abs error and returns a stage-tagged `OracleError` (never panics internally).

## No Analog Found

| File | Role | Data Flow | Reason | Use Instead |
|------|------|-----------|--------|-------------|
| `crates/cb-compute/src/runtime.rs` | trait/boundary | request-response | First abstract `R: Runtime`/`F: Float` trait in the repo; nothing defines a generic compute boundary yet | RESEARCH §"Recommended Project Structure" + CONTEXT D-04 trait-op list; trait-seam precedent = `IngestSource` (`cb-data/src/lib.rs:5-8`) |
| `crates/cb-backend/src/kernels.rs` | kernel | streaming | First `#[cube]` CubeCL kernels; no GPU/kernel code exists | CubeCL manual `INDEX.md` + `Cubecl_generics.md` (AGENTS.md); RESEARCH Pattern 2 |
| `crates/cb-train/src/overfit.rs` | service (state machine) | event-driven | No event-driven detector/state-machine exists in-repo | `overfitting_detector.cpp:127-174` (RESEARCH Code Examples) |

> For all three, the engineering scaffolding (error type B, summation routing C, lint/test layout D) still applies from the Shared Patterns; only the algorithm/trait *body* has no in-repo analog.

## Metadata

**Analog search scope:** `crates/cb-core/`, `crates/cb-data/`, `crates/cb-oracle/`, plus the three stub crates `crates/cb-compute/`, `crates/cb-backend/`, `crates/cb-train/`; root `Cargo.toml`; `crates/cb-oracle/fixtures/regression_skeleton/model.json`.
**Files scanned:** 16 Rust source files + 4 manifests + 1 model.json schema inspection.
**Pattern extraction date:** 2026-06-13
