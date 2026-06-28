# Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock) - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 18 new/modified Rust files (+ committed flatc-generated bindings + new fixtures)
**Analogs found:** 16 / 18 (2 have no in-repo Rust analog — see No Analog Found)

> Scope note: the C++ at `catboost-master/` is the parity SPEC (the algorithm to
> transcribe), NOT a code-pattern analog. Every "Analog" below is an existing
> **Rust** file in `crates/`; the C++ citations from RESEARCH.md tell the planner
> *what math to write*, while the Rust analog tells them *how the code should look*
> (error type, module shape, summation routing, test layout).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog (Rust) | Match Quality |
|-------------------|------|-----------|------------------------|---------------|
| `crates/cb-model/src/model.rs` | model | transform | `crates/cb-train/src/boosting.rs` (`Model`/`ObliviousTree`) | exact |
| `crates/cb-model/src/generated/*.rs` | generated/config | file-I/O | — (flatc output, no analog) | none |
| `crates/cb-model/src/cbm.rs` | service (serde) | file-I/O | `crates/cb-oracle/src/model_json.rs` (load/save framing) | role-match |
| `crates/cb-model/src/json.rs` | service (serde) | file-I/O | `crates/cb-oracle/src/model_json.rs` | exact |
| `crates/cb-model/src/apply.rs` | service | transform | `cb-train/src/tree.rs` `leaf_index` + `boosting.rs` `tree_eval_contribution` | exact |
| `crates/cb-model/src/predict.rs` | service | transform | `crates/cb-compute/src/loss.rs` (`sigmoid`, scalar transforms) | role-match |
| `crates/cb-model/src/shap.rs` | service | transform | `crates/cb-compute/src/loss.rs` + `cb-core::sum_f64` routing | partial |
| `crates/cb-model/src/fstr.rs` | service | transform | `crates/cb-compute/src/score.rs` / `loss.rs` (weighted reductions) | partial |
| `crates/cb-model/src/error.rs` *(or inline in lib.rs)* | utility | — | `crates/cb-oracle/src/error.rs` | exact |
| `crates/cb-model/src/lib.rs` (modified) | config | — | `crates/cb-train/src/lib.rs` (module wiring + re-exports) | exact |
| `crates/catboost-rs/src/builder.rs` | builder | request-response | `cb-train/src/boosting.rs` `BoostParams` (param surface) | role-match |
| `crates/catboost-rs/src/model.rs` | model (facade) | transform | `cb-train/src/boosting.rs` `Model` impl methods | role-match |
| `crates/catboost-rs/src/error.rs` | utility | — | `crates/cb-core/src/error.rs` | exact |
| `crates/catboost-rs/src/lib.rs` (modified) | config | — | `crates/cb-train/src/lib.rs` | exact |
| `crates/cb-train/src/boosting.rs` (modified) | model | transform | itself (add `leaf_weights` capture) | self |
| `crates/cb-oracle/src/model_json.rs` (modified) | service (serde) | file-I/O | itself (add `leaf_weights` field) | self |
| `crates/cb-model/src/*_test.rs` | test | — | `crates/cb-train/tests/leaf_methods_oracle_test.rs` | exact |
| `crates/cb-oracle/generator/gen_fixtures.py` (modified) | test fixture gen | batch | itself (existing scenario pattern) | self |

## Shared Patterns

These cross-cutting patterns apply to nearly every new file. The planner should
reference them once per plan rather than re-deriving.

### Source/Test Separation + Lint Exemption
**Source:** `crates/cb-model/src/lib.rs:1`, `crates/cb-oracle/src/lib.rs:9-17`
**Apply to:** EVERY new `cb-model` / `catboost-rs` file.
Every library crate root carries the test-only lint exemption (manifest overrides
are forbidden under `lints.workspace = true`). New source files contain ONLY
implementation; tests go in dedicated `*_test.rs` files, wired with
`#[cfg(test)] mod foo_test;` at the crate root.

```rust
// crate root (lib.rs) first line:
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
```

Module wiring + re-export pattern from `crates/cb-train/src/lib.rs:16-37`:
```rust
mod apply;
mod cbm;
mod model;
// ...
pub use model::{Model, ObliviousTree};
pub use apply::predict_raw;

#[cfg(test)]
mod apply_test;
```

### Float Summation Routing (D-08 grep gate)
**Source:** `crates/cb-core/src/reduction.rs:31-38` (`sum_f64`), used in
`crates/cb-data/src/borders.rs:36,54` and `crates/cb-train/src/boosting.rs:32,451`.
**Apply to:** `apply.rs`, `shap.rs`, `fstr.rs`, `predict.rs` (any float fold).
NEVER write `iter().sum()` / `.fold(0.0, +)` over floats. Import and call
`cb_core::sum_f64` / `sum_f32_in_f64`. CI script
`scripts/check-no-raw-float-sum.sh` bans raw folds in library crates.

### thiserror Error Enums (no anyhow, no unwrap)
**Source:** `crates/cb-core/src/error.rs:16-97`, `crates/cb-oracle/src/error.rs:8-80`.
**Apply to:** `cb-model/src/error.rs`, `catboost-rs/src/error.rs`.
- `#[derive(Debug, thiserror::Error)]`, one `#[error("…")]` per variant.
- For the PUBLIC `CatBoostError` (D-08): use `#[from] CbError` for the training/data
  arm AND `#[from] std::io::Error` for `Io` — note this makes `CatBoostError` NOT
  `Clone`/`PartialEq` (io::Error is neither), which is WHY internal crates keep the
  `Clone`/`PartialEq`/`Eq`-derivable `CbError` (see `cb-oracle/src/error.rs` which
  already mixes `#[from] std::io::Error` + `#[from] serde_json::Error` and so drops
  those derives — copy that exact tradeoff).
- The deserialization variants pattern (validate-then-typed-error, never panic) is
  modeled by `model_json.rs:123-131` `bias()` (`.ok_or_else(|| OracleError::…)`).

Both new crate stubs currently note "anyhow intentionally absent" — keep it:
`crates/cb-model/Cargo.toml:11`, `crates/catboost-rs/Cargo.toml:11-12`.

### Oracle Integration-Test Layout
**Source:** `crates/cb-train/tests/leaf_methods_oracle_test.rs:1-60`.
**Apply to:** all `*_test.rs` covering MODEL-0x / LOSS-0x oracle locks.
- Tests that need `cb-oracle` go under `tests/` (integration), not `src/*_test.rs`
  (unit) — `cb-model`'s oracle tests should be integration tests so they can depend
  on `cb-oracle`. Pure-math unit tests (e.g. der1/der2) go in `src/*_test.rs` like
  `crates/cb-compute/src/loss_test.rs:1-30`.
- Fixture path helper (copy verbatim, adjust crate hop):
```rust
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("cb-oracle").join("fixtures").join(rel)
}
```
- Gate with `cb_oracle::compare_stage(Stage::Predictions, &expected, &actual)?`
  (`crates/cb-oracle/src/compare.rs:70-96`); the ≤1e-5 tolerance is baked in.

### Fixture Generation (D-13, offline only)
**Source:** `crates/cb-oracle/generator/gen_fixtures.py` (scenario at lines
~585-640 shows the `(method, loss, target, prediction_type)` matrix;
`save_model(format="json")` at :612; `prediction_type=` at :618,640).
**Apply to:** new Phase-4 fixtures. Mirror the existing scenario function shape;
add `save_model(format="cbm")` for the `.cbm` load-parity fixture,
`get_feature_importance(type='ShapValues'|'PredictionValuesChange'|'Interaction', data=Pool)`,
and `predict(prediction_type=…)` per LOSS-06 type. Keep `thread_count=1`, fixed
`random_seed`. Generator does NOT run in CI; output `.npy`/`.cbm`/`model.json` are
committed.

## Pattern Assignments

### `crates/cb-model/src/model.rs` (model, transform)

**Analog:** `crates/cb-train/src/boosting.rs:121-161` (the `Model`/`ObliviousTree`
structs + accessor methods).

**Struct + accessor pattern** (`boosting.rs:123-161`):
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct ObliviousTree {
    pub splits: Vec<Split>,
    pub leaf_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub oblivious_trees: Vec<ObliviousTree>,
    pub bias: f64,
}

impl Model {
    #[must_use]
    pub fn leaf_values(&self) -> Vec<f64> {
        self.oblivious_trees.iter()
            .flat_map(|t| t.leaf_values.iter().copied())
            .collect()
    }
}
```

**Phase-4 delta (RESEARCH Pitfall 1, the FIRST task):** the canonical `cb-model::Model`
adds `leaf_weights: Vec<Vec<f64>>` (one inner vec per tree, length `2^depth`) and
per-feature `float_feature_borders: Vec<Vec<f64>>`. Re-home `cb-train::Model` →
`cb-model::Model` (Claude's discretion D / RESEARCH Architecture) to avoid a dep
cycle. The `Split { feature: usize, border: f64 }` shape is `cb-train/src/tree.rs:54-59`
— reuse or re-export it, do not redefine its semantics.

---

### `crates/cb-model/src/apply.rs` (service, transform)

**Analog:** `crates/cb-train/src/tree.rs:99-110` (`leaf_index`) +
`crates/cb-train/src/boosting.rs:268-282` (`tree_eval_contribution`).

**Leaf-walk + accumulate pattern** (`boosting.rs:268-282`) — the apply path is this
generalized over all trees, with `+bias`:
```rust
fn tree_eval_contribution(tree: &ObliviousTree, matrix: &FeatureMatrix, obj: usize) -> f64 {
    let passes: Vec<bool> = tree.splits.iter()
        .map(|s| matrix.feature_values.get(s.feature)
            .and_then(|col| col.get(obj))
            .is_some_and(|&v| f64::from(v) > s.border))   // strict > (Phase-2/3 border semantics)
        .collect();
    let leaf = leaf_index(&passes);                        // forward bit order
    tree.leaf_values.get(leaf).copied().unwrap_or(0.0)     // checked .get, no indexing
}
```

**Forward-bit-order leaf index** (`tree.rs:102-110`) — already matches upstream
`CalcIndexesBasic` (RESEARCH Pattern 2 Step B): `idx |= 1usize << i` per passing
split. Reuse `cb_train::leaf_index` directly rather than re-rolling.

**Phase-4 delta:** raw-float-feature input must binarize via strict `> border`
count first (RESEARCH Pattern 2 Step A) — the same `value > border` test the
trainer uses; the `cb-data` border lookup (`crates/cb-data/src/quantize.rs`, strict
`value > border` at quantize.rs:3) is the reference for the count. Accumulate leaf
sums through `cb_core::sum_f64`, then add `bias` (single source of truth; RESEARCH
Pitfall 6). MODEL-02: keep this file pure Rust — do NOT touch `cb-backend`/cubecl.

---

### `crates/cb-model/src/predict.rs` (service, transform)

**Analog:** `crates/cb-compute/src/loss.rs:44-75` (`sigmoid` and the scalar
transform style — small `#[must_use]` fns, upstream arithmetic transcribed
verbatim with a source-line doc comment).

**Scalar-transform pattern** (`loss.rs:50-54`):
```rust
/// `e = exp(approx); p = 1 - 1/(1+e)`. Transcribed in the upstream form so
/// rounding matches. (error_functions.cpp:317-340)
#[must_use]
pub fn sigmoid(approx: f64) -> f64 {
    let e = approx.exp();
    1.0 - 1.0 / (1.0 + e)
}
```

**Phase-4 delta:** add `PredictionType` enum (RawFormulaVal/Probability/
LogProbability/Class/Exponent) and per-type transforms (RESEARCH Pattern 3 table).
Use `f64::exp` for Probability/LogProbability (oracle uses `std::exp` there);
`Exponent` may need FastExp-awareness (RESEARCH Pitfall 3 / Assumption A2 —
verify ≤1e-5 holds with `f64::exp` when generating the fixture). `Class` threshold
defaults to 0 (RESEARCH Pitfall 4). `LogProbability` returns TWO columns.

---

### `crates/cb-model/src/cbm.rs` (service, file-I/O)

**Analog:** `crates/cb-oracle/src/model_json.rs:139-143` (`load_model_json` — the
read-then-parse, fallible-loader shape) for the I/O framing; the FlatBuffers body
uses the committed `generated/` bindings.

**Loader shape** (`model_json.rs:139-143`):
```rust
pub fn load_model_json(path: &Path) -> Result<ModelJson, OracleError> {
    let contents = std::fs::read_to_string(path)?;     // io::Error via #[from]
    let model = serde_json::from_str(&contents)?;
    Ok(model)
}
```

**Phase-4 delta:** `.cbm` is `b"CBM1"` magic + `u32::to_le_bytes(size)` + FlatBuffers
`TModelCore` (RESEARCH Pattern 1 / Code Example, `model.cpp:1113-1163`). Use
`u32::try_from(...).map_err(|_| CatBoostError::SchemaVersion(...))?` (checked
conversion — `indexing_slicing`/overflow are deny-lints, RESEARCH Security V5).
Emit `FormatVersion = "FlabuffersModel_v1"` exactly (RESEARCH Pitfall 5). On read,
use the VERIFYING flatbuffers accessor (`root_as_*`), map failure to
`CatBoostError::Deserialize` (RESEARCH Security V5).

---

### `crates/cb-model/src/json.rs` (service, file-I/O)

**Analog:** `crates/cb-oracle/src/model_json.rs` (WHOLE FILE — the upstream
`model.json` schema is already modeled here as serde `Deserialize` structs).

**Serde-struct pattern** (`model_json.rs:22-81`): `#[derive(Debug, Clone, PartialEq,
Deserialize)]` per node; `#[serde(default)]` on optional arrays; untyped
`serde_json::Value` for `scale_and_bias` with a typed accessor `bias()` that returns
`Result` (model_json.rs:71-131).

**Phase-4 delta (D-04, MODEL-06):** this file must SERIALIZE (`Serialize` derive +
`serde_json::to_string`) the upstream schema, not just parse it. Add `leaf_weights`
per tree (RESEARCH Pitfall 2 — JSON layout is per-tree nested, NOT the `.cbm` flat
array). The existing `cb-oracle::model_json` doubles as the round-trip oracle, so
field names must match (`oblivious_trees`, `leaf_values`, `scale_and_bias =
[1,[bias]]`, `features_info.float_features[].borders`). EXTEND
`cb-oracle/src/model_json.rs` to add a `leaf_weights` field on `ObliviousTree`
(self-modification).

---

### `crates/cb-model/src/shap.rs` (service, transform)

**Analog (closest in-repo):** `crates/cb-compute/src/loss.rs` (verbatim-transcription
+ `#[must_use]` scalar style) and `cb-core::sum_f64` routing
(`crates/cb-train/src/boosting.rs:451`). No existing Rust recursion analog — this is
the hardest new code.

**Spec:** RESEARCH Pattern 4 + Code Example (`extend` at `shap_values.cpp:44-64`),
prepared-trees `shap_prepared_trees.cpp:25-222`. Needs `leaf_weights`. Output shape
`[approxDim][featureCount+1]`; trailing column = `Σ meanValue + bias`. Assert the
local-accuracy invariant `Σ shap == RawFormulaVal prediction` (D-11) using
`cb_oracle::compare_stage`.

**Pattern caution:** the `indexing_slicing = deny` lint makes the index-heavy SHAP
recursion non-trivial — use `.get()` + checked access throughout (RESEARCH Project
Constraints), mirroring the `.and_then(|col| col.get(obj))` style in
`boosting.rs:274-277`. All feature-path weight folds route through `cb_core::sum_f64`.

---

### `crates/cb-model/src/fstr.rs` (service, transform)

**Analog:** `crates/cb-compute/src/score.rs` / `loss.rs` (weighted-reduction +
`sum_f64` style). No exact Rust analog for `CalcEffect`.

**Spec:** RESEARCH Pattern 5 (`CalcEffect`, `feature_str.h:233-270`) +
Pattern 6 (Interaction). Both need `leaf_weights`. PredictionValuesChange normalizes
to percent (Σ=100); Interaction returns `Vec<(usize, usize, f64)>`. LossFunctionChange
is OUT OF SCOPE (D-12). Skip leaves with `count==0` (the upstream short-circuit).

---

### `crates/cb-model/src/error.rs` (utility)

**Analog:** `crates/cb-oracle/src/error.rs:8-80` (thiserror enum mixing `#[from]`
arms that drop `Clone`/`PartialEq`). NOTE: per D-08 the PUBLIC `CatBoostError`
lives in `catboost-rs/src/error.rs`; `cb-model` may surface a narrower internal
error or reuse `cb-core::CbError`. Decide during planning — RESEARCH puts the
public enum in `catboost-rs`.

---

### `crates/catboost-rs/src/error.rs` (utility) — the public `CatBoostError` (D-08, RAPI-02)

**Analog:** `crates/cb-core/src/error.rs:16-97` (variant catalog + `#[error]` docs)
and `crates/cb-oracle/src/error.rs:69-79` (`#[from]` for `io`/`serde_json`).

**Public-error pattern (compose both):**
```rust
#[derive(Debug, thiserror::Error)]   // NOT Clone/PartialEq — io::Error blocks them (D-08 rationale)
pub enum CatBoostError {
    #[error(transparent)]
    Train(#[from] cb_core::CbError),               // training/data arm
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("deserialization failed: {0}")]
    Deserialize(String),
    #[error("unsupported model schema version: {0}")]
    SchemaVersion(String),
    #[error("feature mismatch: {0}")]
    FeatureMismatch(String),
}
```

---

### `crates/catboost-rs/src/builder.rs` (builder, request-response) — RAPI-01, D-05

**Analog:** `crates/cb-train/src/boosting.rs:57-119` (`BoostParams` — the full param
surface the Builder wraps) and the `train()` entry point (`boosting.rs:298-318`).

**Builder drives `cb-train::train`:** `CatBoostBuilder::new()...loss(...).iterations(...)
.depth(...).fit(&pool) -> Model`. The loss selects clf-vs-regression (D-05). `fit`
maps the builder fields onto `BoostParams` (`boosting.rs:57-119`) and calls
`cb_train::train`. Use the typestate/Builder convention from CLAUDE.md (Rust side =
Builder pattern). Loss enum source: `crates/cb-compute/src/runtime.rs:42-55` (extend
with CrossEntropy/Focal per D-09).

---

### `crates/catboost-rs/src/model.rs` (model facade, transform) — D-06/D-07

**Analog:** `crates/cb-train/src/boosting.rs:141-161` (methods-on-`Model` ergonomics).

**Methods-on-Model pattern (D-07):** `Model::{predict(pool, PredictionType),
predict_proba, predict, save_cbm, load_cbm, save_json, load_json, shap_values,
feature_importance}` — a cohesive object delegating to `cb-model`. The enum-core +
shorthand pattern (D-06): one `predict(pool, PredictionType)` plus
`predict_proba()`/`predict()` wrappers calling it with a fixed `PredictionType`.

---

## No Analog Found

| File | Role | Data Flow | Reason | Planner Guidance |
|------|------|-----------|--------|------------------|
| `crates/cb-model/src/generated/*.rs` | generated | file-I/O | flatc output; not hand-written, no in-repo analog | Generate via `flatc --rust` from vendored `.fbs` (RESEARCH Standard Stack install block), COMMIT (D-01). Add `flatbuffers = "25.12.19"` to `cb-model/Cargo.toml`. `flatc` NOT installed locally — RESEARCH Open Q2 / A5: add an install-or-`checkpoint:human-verify` task. |
| `crates/cb-model/src/shap.rs` recursion core | service | transform | No existing Rust tree-recursion in the repo | Transcribe line-for-line from `shap_values.cpp` (RESEARCH Pattern 4 + Code Example). The only in-repo guidance is the summation routing + checked-index discipline (Shared Patterns above). Budget extra time for `indexing_slicing`-clean recursion. |

## Metadata

**Analog search scope:** `crates/cb-model`, `crates/catboost-rs`, `crates/cb-train`
(`src/`, `tests/`), `crates/cb-oracle/src`, `crates/cb-core/src`,
`crates/cb-compute/src`, `crates/cb-data/src`, `crates/cb-oracle/generator`.
**Files scanned (read in full or targeted):** 14 Rust source files + 2 Cargo.toml +
workspace manifest + generator grep.
**C++ spec citations:** sourced from 04-RESEARCH.md (not re-derived here).
**Pattern extraction date:** 2026-06-13
