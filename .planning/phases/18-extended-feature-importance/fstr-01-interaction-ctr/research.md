# Phase 18 Research: FSTR-01 — Interaction Feature Importance with CTR (categorical) Support

## Research Summary

- **Phase goal:** Close the last named gap in Phase 18 (Extended Feature
  Importance) by extending `cb_model::interaction()` — currently correct only
  for FLOAT-split (numeric-only) models — to also attribute pairwise
  `Interaction` importance across models that contain CTR (categorical) splits,
  matching upstream CatBoost `get_feature_importance(type='Interaction')`
  within the project's `1e-5` oracle bar.
- **Recommended approach:** keep `interaction()`'s existing tree-walk
  (oblivious bit-indexed + non-symmetric DFS) untouched for the FLOAT-vs-FLOAT
  case (byte-identical, do not touch — `D-6.6-05`/`D-6.6-10` locked code), and
  add a second accumulation arm that resolves each `ModelSplit::Ctr`'s
  `CtrSplit.projection.cat_features` (already-available field,
  `[CODEGRAPH: crates/cb-train/src/projection.rs:130]`) into interaction pairs,
  mirroring upstream's `GetFeatureToIdxMap` / `CalcMostInteractingFeatures(model,
  featureToIdx)` / `CalcRegularFeatureEffect` CTR-splitting logic
  `[WEB: github.com/catboost/catboost catboost/libs/fstr/{feature_str.cpp,calc_fstr.cpp}, master HEAD, accessed 2026-07-17]`.
- **Most important constraints:**
  - CTR structural support (`ModelSplit::Ctr`, `apply.rs::predict_raw_cat`,
    `crate::ctr_data::CtrData`) is **already fully implemented and oracle-tested**
    on the current branch for catboost-rs's OWN trained categorical models
    (`crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs`) — this phase is
    additive to `fstr.rs` only, not a CTR-plumbing project.
  - The unmerged `feat/23-ctr-model-loading` branch/worktree (upstream `.cbm`
    CTR reconstruction, i.e. loading a model **trained by upstream CatBoost**
    that has CTR splits) is **confirmed NOT present** on this branch/working
    tree and is **not required** for this feature: FSTR-01's oracle model can
    be built the same way `tensor_ctr_e2e_oracle_test.rs` already does (train
    with `cb_train::train_cat`, lift via `Model::from_trained`), which needs no
    `.cbm` decode.
  - `Model` exposes **no flat/original feature-index map** spanning float + cat
    features (only `float_feature_borders: Vec<Vec<f64>>` and
    `ctr_data: Option<CtrData>`) — this was flagged explicitly as an "FSTR-01
    concern" and deliberately deferred by the immediately-preceding FSTR-03
    spec `[PROJECT: .planning/phases/18-extended-feature-importance/fstr-03-partial-dependence/SPEC.md §4,§9.3]`.
    Designing this index space is the central design decision of this phase.
- **Highest-risk findings:**
  1. The exact upstream CTR→feature-pair attribution rule for `Interaction`
     specifically (as opposed to the simpler `PredictionValuesChange`
     "regular effect" redistribution, which IS documented) is not fully pinned
     down from primary sources available to this research pass — the vendored
     `catboost-master/` tree in this repo does **not** contain
     `feature_str.cpp`/`calc_fstr.cpp` (0 hits), so the algorithm had to be
     fetched from GitHub `master` (not the pinned oracle version `1.2.10`);
     version drift between `master` and `1.2.10` is unverified. **[LOW confidence — flag for SPEC-time re-verification against the 1.2.10 tag specifically, or empirical fixture-driven reverse engineering, consistent with project convention elsewhere.]**
  2. `PredictionValuesChange` (`prediction_values_change()`) has the **exact
     same** one-line CTR-skip as `interaction()`, but is not itself named by
     any open requirement ID (`FSTR-01`/`02`/`03`) — whether this slice should
     also fix PVC's CTR gap (same underlying index-space machinery) or leave
     it for a separate, later requirement is an **open scoping question** for
     the spec author.

## Phase Requirements

### In Scope

- Extend `cb_model::fstr::interaction()` so that a model containing
  `ModelSplit::Ctr` splits (in either oblivious or non-symmetric trees)
  produces `Interaction` pairs attributing CTR-split effect to the underlying
  ORIGINAL (float and/or categorical) feature(s) in the CTR's combined
  projection, not silently dropping them
  `[CODEGRAPH: crates/cb-model/src/fstr.rs:316-321,448-455 the current `continue`-on-CTR skip]`.
- A dataset-free computation (`interaction(model: &Model) -> Vec<(usize,
  usize, f64)>` signature is unchanged — Interaction is explicitly
  "dataset-free" per the recovered requirement text)
  `[PROJECT: git-recovered .planning/REQUIREMENTS.md@a82289c "FSTR-01: ... pairwise split-cooccurrence over tree structure; dataset-free"]`.
- New oracle fixture(s) with a trained categorical (CTR-bearing) model,
  comparing `interaction()` output against upstream
  `get_feature_importance(type='Interaction')` at `≤1e-5`.
- A decision on the returned index space (float-feature index vs. a combined
  float+cat "regular" index vs. an internal per-distinct-split index) — see
  Open Questions.

### Acceptance Criteria

- `interaction()` on a CTR-bearing model no longer silently omits CTR-split
  contributions; the returned triples reproduce upstream's `Interaction`
  values within `1e-5` for at least one single-cat-feature CTR model and one
  combination (tensor) CTR model (mirroring `tensor_ctr` / `tensor_ctr_e2e`
  fixture coverage already in the repo)
  `[CODEGRAPH: crates/cb-oracle/fixtures/tensor_ctr, tensor_ctr_e2e]`.
- The existing float-only `interaction()` oracle assertions
  (`fstr_oracle_test.rs` item 2) remain green, byte-identical, unchanged
  (regression lock, `D-6.6-05`).
- No `unwrap`/`expect`/`panic`/`indexing_slicing` introduced (workspace-denied
  clippy restriction lints) `[VERIFIED: Cargo.toml:10-14]`.
- `cargo clippy -p cb-model --all-targets` and `cargo test -p cb-model` (+
  `-p cb-oracle`) pass with zero new warnings/failures (pre-existing
  `cb-oracle/src/model_json.rs:161` clippy nit and `ctr_data_roundtrip_test.rs`
  are baseline, not introduced by this slice —
  `[PROJECT: fstr-03-partial-dependence/PLAN.md "Not done" section]`).

### Out of Scope

- `LossFunctionChange` + CTR (**FSTR-02** — a separate, larger slice: it needs
  `shap_values`/`shap_interaction_values` to gain CTR support first, since
  `loss_function_change()` is built on top of `shap_values`
  `[CODEGRAPH: crates/cb-model/src/fstr.rs:503-505 calls shap_values]`, and
  `shap.rs` currently also skips CTR splits and sizes its output vector as
  `model.float_feature_borders.len()` only
  `[CODEGRAPH: crates/cb-model/src/shap.rs:728,1096-1097]`).
- Upstream `.cbm` CTR model **loading** (phase-23 `cbm-ctr-load` work) — not
  needed by this slice (see Constraints above) and not touched by it.
- SHAP / SHAP-interaction CTR support (`shap_values`, `shap_interaction_values`,
  `sage_values`, `prediction_diff`) — all currently numeric-only
  (`advanced_fstr` MODEL-05 family); untouched.
- Python / facade surfacing of `interaction()` beyond whatever already exists
  (the Rust facade `catboost-rs::Model::feature_importance` already calls
  `interaction` today for ALL models, float-only or not — extending its
  correctness for CTR models is exactly this slice's contribution; no NEW
  facade wiring is required) `[CODEGRAPH: crates/catboost-rs/src/model.rs:139-149]`.
  Python (`catboost-rs-py`) does not currently expose ANY `get_feature_importance`
  surface at all `[VERIFIED: grep 'get_feature_importance' crates/catboost-rs-py/src → 0 hits]`
  — wiring that up is a separate later DX task (same pattern as FSTR-03 was
  split into `fstr-03-partial-dependence` + `fstr-03-facade-python`).

### Open or Conflicting Requirements

1. **PredictionValuesChange CTR gap has no owning requirement ID.**
   `prediction_values_change()` skips CTR splits identically to
   `interaction()` (`[CODEGRAPH: crates/cb-model/src/fstr.rs:149-151,230-233]`),
   but no requirement text (`FSTR-01/02/03`) names it — it was presumably
   considered "MODEL-03 partial" and closed out for the float-only case in an
   earlier, already-complete phase. **The spec author must decide** whether
   this slice folds in the PVC CTR fix (same underlying machinery, near-zero
   marginal cost) or explicitly defers it (and if deferred, whether a new
   requirement ID should be minted).
2. **Exact CTR→index-space semantics for Interaction pairs is unresolved** —
   see Highest-risk findings #1 and the Recommended Architecture section
   below for the best-available (but unverified against the pinned 1.2.10
   oracle) upstream algorithm sketch.
3. **`.planning/REQUIREMENTS.md` and `.planning/ROADMAP.md` do not exist in the
   working tree** — they are git-recovered from commit `a82289c` per prior
   phases' own SOURCES.md ledgers and per this session's own verification
   (`git show a82289c:.planning/REQUIREMENTS.md` succeeds; the file is absent
   from `git status`/`ls`). Treat the recovered text as historical evidence,
   not a live authoritative file; the SPEC author should re-run the same
   recovery before citing line numbers.

## Project Constraints

- Source/test separation is mandatory (CLAUDE.md): no `#[cfg(test)] mod tests`
  body in a production `.rs` file; unit tests live in a sibling `_test.rs`
  mounted via `#[path = "..._test.rs"] mod tests;`
  `[PROJECT: CLAUDE.md "Source/Test Separation — Mandatory Rule"]`. Existing
  precedent to follow exactly: `crates/cb-model/src/ctr_data.rs:58-61`.
- Workspace-wide clippy restriction lints deny `unwrap_used`, `expect_used`,
  `panic`, `indexing_slicing` in production code (`[lints] workspace = true`)
  `[VERIFIED: Cargo.toml:10-14]`; **only enforced by `cargo clippy`, not `cargo
  build`** — a documented gotcha from the immediately-preceding FSTR-03 plan
  check `[PROJECT: fstr-03-partial-dependence/PLAN-CHECK.md MAJOR #2]`.
- All float reductions must route through `cb_core::sum_f64` (never a raw
  `.sum()`/`fold(0.0, ...)`) — D-08, already the pattern in `fstr.rs`.
- Parity bar: `≤1e-5` CPU oracle tolerance (`D-12`), the same bar used by every
  other `cb-model` oracle test.
- `unwrap()` strictly prohibited in production per top-level CLAUDE.md project
  constraints; `thiserror` for library errors, `anyhow` banned in `cb-model`
  (`D-14`, explicit `Cargo.toml` comment: "anyhow is intentionally absent (D-14
  structural ban). Do not add it.").
- No CubeCL / GPU dependency may enter `cb-model` (MODEL-02 boundary —
  `apply.rs` imports nothing from `cb-backend`); this feature is CPU/host-only
  and must preserve that boundary.
- Dependencies: "always use the latest crate versions" (top-level CLAUDE.md) —
  no NEW crate is anticipated for this slice (pure algorithm extension over
  already-present types).
- Known-red, pre-existing, environmental test suites to IGNORE (not this
  slice's responsibility): `cb-backend --lib` (CubeCL MLIR), `cb-train
  monotone_*`, `catboost-rs-py` (python3.14 link)
  `[PROJECT: fstr-03-partial-dependence/PLAN.md "Known-red suites"]`.

## Current Project Architecture

### Relevant subsystems and boundaries

- **`crates/cb-model/src/fstr.rs`** — the feature-importance module. Currently
  exports `prediction_values_change`, `interaction`, `loss_function_change`,
  `FeatureImportanceType` `[CODEGRAPH: crates/cb-model/src/lib.rs:35-37]`.
  `interaction()` (fstr.rs:288-355) has two accumulation arms already:
  - OBLIVIOUS arm (fstr.rs:293-329): literal pre-6.6 bit-indexed double loop
    over split levels, `D-6.6-05` BYTE-IDENTICAL lock.
  - NON-SYMMETRIC arm (fstr.rs:333-335, helpers 369-468): DFS over the
    node-graph accumulating signed per-path pair contributions, `D-6.6-10`.
  Both arms currently `continue`/skip whenever
  `ModelSplit::float_feature()` returns `None` (i.e., any `ModelSplit::Ctr`)
  `[CODEGRAPH: crates/cb-model/src/fstr.rs:316-321 (oblivious), 448-455 (non-symmetric)]`.
- **`crates/cb-model/src/model.rs`** — the canonical `Model`/`ModelSplit`/
  `CtrSplit` types. `ModelSplit::Ctr(CtrSplit)` is already a first-class,
  fully-matched split variant (model.rs:70-98); `CtrSplit` carries
  `projection: cb_train::TProjection` (model.rs:43-63), which in turn carries
  `cat_features: Vec<usize>` — the underlying categorical feature indices the
  CTR was built over `[CODEGRAPH: crates/cb-train/src/projection.rs:93-186]`.
- **`crates/cb-model/src/apply.rs`** — already fully supports CTR-split
  evaluation (`predict_raw_cat`, `passes_ctr_split`, `ctr_value_for_projection`,
  `ctr_value_for_combined_projection`) `[CODEGRAPH: crates/cb-model/src/apply.rs:37,170-199,386]`.
  This is the PROOF that CTR structural support is a solved, oracle-tested
  problem in this codebase already — `fstr.rs` is the one place still treating
  CTR as second-class.
- **`crates/cb-model/src/shap.rs`** — SHAP-family computations
  (`shap_values`, `shap_interaction_values`, `prediction_diff`, `sage_values`).
  Also currently numeric-only (sizes its output vector to
  `model.float_feature_borders.len()`, filters splits via `float_feature()`)
  `[CODEGRAPH: crates/cb-model/src/shap.rs:728,1043,1096-1097,1150,1170]`.
  Explicitly out of scope for FSTR-01 but is the hard dependency for FSTR-02.
- **`crates/cb-train`** — the trainer. `Model::from_trained` (model.rs:341)
  is the ONE place `ModelSplit::Ctr` values are constructed from a trained
  model's `AnySplit::Ctr` output; this is how catboost-rs's own trainer
  produces CTR-bearing canonical `Model`s (no `.cbm` decode involved).
  `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` and
  `multi_permutation_e2e_oracle_test.rs` are existing, PASSING, oracle-verified
  (≤1e-5, full multi-tree) end-to-end train→predict tests on categorical data
  — this is the reusable pattern for generating an FSTR-01 oracle fixture
  model without touching the unmerged CTR-loading branch.
- **`crates/cb-oracle`** — the oracle harness (`model_json.rs`,
  `compare.rs`). `SplitJson` already parses `split_type: "OnlineCtr"` from
  upstream `model.json` (field present, "ABSENT on OnlineCtr splits" comment
  for `float_feature_index`) `[CODEGRAPH: crates/cb-oracle/src/model_json.rs:24-41]`,
  and `CtrTableJson`/`ModelJson::ctr_data()` already parse the upstream
  `ctr_data` hash-map from `model.json` `[CODEGRAPH: crates/cb-oracle/src/model_json.rs:236-347,469-476]`.
  However, the EXISTING test helper `model_from_json` in
  `fstr_oracle_test.rs` (and `advanced_fstr_oracle_test.rs`) only builds
  `ModelSplit::Float` — there is **no existing helper that builds a canonical
  `Model` with `ModelSplit::Ctr` directly from a `ModelJson`** (JSON path); the
  only place `ModelSplit::Ctr` is built today is `Model::from_trained` (the
  cb-train lift path) and hand-written unit-test literals (`onnx_test.rs`,
  `export/onnx_test.rs`, `ctr_split_scoring_test.rs`). Building (or not
  needing) this JSON→CtrSplit bridge is a planning decision, not resolved
  here.

### Existing data/control flow

Upstream comparison flow used by every other `cb-model` oracle test (the
established, must-preserve pattern):

1. `cb-oracle/fixtures/<name>/gen_fixtures.py` trains a small model with a
   PINNED upstream `catboost==1.2.10` in an offline venv, dumps
   `model.json`/`.cbm`/`.npy` ground-truth arrays, commits them.
2. The Rust oracle test (`crates/cb-model/tests/*_oracle_test.rs`) loads the
   fixture, builds/loads a canonical `Model`, calls the function under test,
   and asserts `≤1e-5` against the committed `.npy` via
   `cb_oracle::compare::assert_abs_close`.

For CTR-bearing models specifically, `cb-train`'s existing e2e oracle tests
show a SECOND valid pattern that does not need upstream JSON/CBM parsing at
all: train via `cb_train::train_cat` with the SAME fixture/seed/params as an
existing committed upstream categorical fixture (`tensor_ctr_e2e`,
`multi_permutation_e2e`), lift with `Model::from_trained`, and the resulting
`Model` is already known (by that suite's own oracle assertions) to match
upstream `predict_raw` ≤1e-5. FSTR-01's oracle model can be built either way;
whichever is cheaper to wire is a PLAN-time decision.

### Existing reusable implementations

- `ModelSplit::float_feature()` — the existing float/CTR discriminator every
  fstr/shap function currently uses to skip CTR (`model.rs:82-88`). A new CTR
  accumulation arm should read `ModelSplit::Ctr(_)` directly (pattern-match),
  not repurpose this method.
- `interaction_add` (fstr.rs:265-274) — the shared insertion-order
  `(pairs, sums)` accumulator helper; reusable as-is for CTR-contributed pairs
  (same signature: `(a: usize, b: usize, contribution: f64)`).
- `TProjection.cat_features: Vec<usize>` (`cb-train/src/projection.rs:93-186`)
  — already the exact "which original cat features does this CTR span" answer
  needed for the attribution logic.
- `cb_core::sum_f64` — the mandated float-fold primitive.
- `cb_oracle::compare::assert_abs_close` — the mandated oracle comparator.

### Current conventions and patterns

- Every "arm" (oblivious vs non-symmetric) is a SEPARATE code path, never a
  refactor unifying them (`D-6.6-05`/`D-6.6-10` — preserve byte-identical
  legacy behavior for the float case while adding new code for new behavior).
- `#[must_use]` on every public pure function; doc comments cite the exact
  upstream C++ function + line range as "Source of truth."
- Oracle test files: top `#![allow(clippy::unwrap_used, clippy::expect_used,
  clippy::panic, clippy::indexing_slicing)]`, `const TOL: f64 = 1e-5;`, a local
  `fixture(rel: &str) -> PathBuf` helper resolving into `../cb-oracle/fixtures`.

## Standard Stack

| Name | Version (this project) | Existing/Proposed | Purpose here | Constraints | Usage | Doc finding |
|---|---|---|---|---|---|---|
| Rust | edition 2021 (cb-model), workspace floor 1.64 per top-level CLAUDE.md (cb-model itself declares `edition = "2021"`) | Existing | Implementation language | — | `[VERIFIED: crates/cb-model/Cargo.toml:4]` | — |
| `thiserror` | 2.0.18 (workspace-pinned) | Existing | Any new typed error variant (if `interaction()`'s signature needs to change — currently infallible, likely stays infallible) | — | Already used by `ModelError` | `[VERIFIED: Cargo.toml:20]` |
| `flatbuffers` | 25.12.19 | Existing | Not touched by this slice (no `.cbm` work) | — | `cb-model/Cargo.toml:29` | — |
| `serde`/`serde_json` | 1.0.228 / 1.0.150 (workspace-pinned) | Existing | Not touched (no new JSON shape) unless the SPEC opts to add a JSON→CtrSplit test bridge | — | `cb-model/Cargo.toml:32-33` | — |
| `ndarray`/`ndarray-npy` | 0.17.2 / 0.10.0 (workspace-pinned) | Existing (dev-dep) | Oracle `.npy` fixture I/O for the new test | — | `cb-model/Cargo.toml:47-48` (dev-deps) | — |
| `catboost` (Python, oracle generation) | `1.2.10` pinned (the project's oracle floor everywhere else) | Existing (external, offline venv) | Generate the ground-truth `Interaction` importance array for a CTR-bearing model | Not installed in the dev container by default (prior sessions installed it into a `uv`-managed venv for FSTR-03; same approach needed here) | `[PROJECT: fstr-03-partial-dependence/PLAN.md "T3 fixtures generated from real upstream"]` | `get_feature_importance(type='Interaction')` returns `(first_idx, second_idx, score)` triples; `type=EFstrType.Interaction`; no `data=` required for Interaction (dataset-free) unless leaf weights are missing `[CONTEXT7-CLI: /catboost/catboost "get_feature_importance" via `npx ctx7@latest docs /catboost/catboost "get_feature_importance type Interaction ..."`]` |
| `cb-train` (internal) | workspace-local | Existing | `train_cat`, `Model::from_trained`, `TProjection` — reused, not reimplemented | already a normal (non-dev) dependency of `cb-model` | `cb-model/Cargo.toml:24` | `[CODEGRAPH]` |
| `cb-oracle` (internal) | workspace-local | Existing (dev-dep) | Oracle comparator + `ModelJson`/`SplitJson`/`CtrTableJson` (if the JSON path is chosen) | dev-dependency only (`D-14`/`D-15` boundary) | `cb-model/Cargo.toml:44` | `[CODEGRAPH]` |

No new external dependency is anticipated. If the SPEC author decides the
oracle fixture needs `catboost` installed for `gen_fixtures.py`, that is an
offline/venv concern already solved by the FSTR-03 precedent, not a Cargo
dependency change.

## Dependency Analysis

- **Direct:** none added. This is an internal-only algorithm extension inside
  `cb-model`, which already depends on `cb-train` (for `TProjection`,
  `Model::from_trained`) and `cb-data` (categorical hashing) as NORMAL
  (non-dev) dependencies.
- **Transitive/peer:** none new.
- **Runtime/build/system:** none new; still CPU-only, still no CubeCL edge in
  `cb-model` (MODEL-02 boundary preserved).
- **Compatibility/migration:** `interaction()`'s public signature
  (`fn interaction(model: &Model) -> Vec<(usize, usize, f64)>`) is a candidate
  for an ADDITIVE, non-breaking change (same signature, richer/more-correct
  output) as long as the returned index space for existing float-only models
  is preserved bit-for-bit (regression risk — see Common Pitfalls). If the
  chosen index-space design widens the tuple's `usize` meaning for CTR models
  (e.g., float indices `0..n_float` followed by cat indices
  `n_float..n_float+n_cat`), that is still additive for float-only models
  (`n_cat == 0`, unchanged range) but IS a semantic change callers must be
  told about (facade/Python docs) once CTR models are involved.
- **Dependency additions/removals:** none.

## Recommended Architecture and Implementation Pattern

**Prescribed approach:** add a THIRD accumulation source inside
`interaction()` — a CTR-projection arm — that runs alongside (not replacing)
the existing oblivious and non-symmetric float arms, using the SAME
`interaction_add(&mut pairs, &mut sums, a, b, contribution)` accumulator so
the final `score = sum / total_effect * 100` normalization and descending sort
are shared, unmodified code.

1. **Component responsibilities**
   - `fstr.rs::interaction()` — unchanged float arms; add a CTR arm that, for
     every `ModelSplit::Ctr` encountered in a tree's splits (oblivious:
     `tree.splits`; non-symmetric: `tree.tree_splits`), resolves
     `ctr_split.projection.cat_features` to the set of underlying original
     categorical feature indices the split's projection spans.
   - **Design decision (planner-owned, not resolved by this research):**
     whether the CTR split's per-pair contribution (already computed
     structurally by the existing bit/DFS walk, since the value/leaf-weight
     math does not care about split KIND) is:
     (a) attributed to `(cat_feature_i, cat_feature_j)` pairs when the CTR is
     itself a combination of ≥2 cat features (direct — no upstream lookup
     needed, `TProjection.cat_features` already gives the member set), and/or
     (b) attributed to pairs between the CTR's member cat feature(s) and the
     OTHER split in the interacting pair when that other split is a plain
     float split (float ↔ cat interaction), following the same
     `GetFeature`/`featureToIdx` "distinct feature → one internal index, but a
     CTR's constituent original features receive redistributed effect"
     pattern upstream uses for `PredictionValuesChange`'s "regular" importance
     `[WEB: github.com/catboost/catboost calc_fstr.cpp `CalcRegularFeatureEffect`,
     "addEffect = effectWithSplit.first / featuresInSplit"; accessed 2026-07-17,
     master HEAD — NOT verified against the pinned 1.2.10 tag]`.
   - The exact interaction-specific rule (as opposed to the PVC "regular
     effect" rule, which the fetch above DOES document) needs a follow-up
     read of `feature_str.cpp`'s `CalcMostInteractingFeatures(model,
     featureToIdx)` overload with the SAME scrutiny, ideally against the
     1.2.10 tag specifically (`https://github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/feature_str.cpp` —
     not fetched in this pass; flagged for the SPEC author).
2. **Integration points:** none beyond `fstr.rs` itself; `interaction_add`
   signature unchanged; `interaction()`'s public signature unchanged (still
   `&Model -> Vec<(usize, usize, f64)>`, dataset-free).
3. **Data/control flow:** identical outer flow (per-tree accumulate → global
   normalize by `total_effect` → stable-sort descending) — only the per-tree
   accumulation gains a CTR-aware branch.
4. **Error/security/failure behavior:** stay infallible (`interaction()` has
   no `Result` today and no reason to add one); any malformed/empty
   projection degenerates to "contributes nothing" (mirrors the existing
   `count1 == 0.0 || count2 == 0.0` and `total_effect == 0.0` div-guard
   discipline, `T-04-04-03`).
5. **What must not be hand-rolled:**
   - Do NOT reimplement `calc_cat_feature_hash` / `fold_cat_hash` / CTR value
     lookup — `interaction()` needs only STRUCTURAL information
     (`TProjection.cat_features`), never a per-document CTR VALUE, so `ctr_data`
     lookup machinery in `apply.rs`/`ctr_data.rs` is irrelevant here and must
     not be pulled in.
   - Do NOT touch the existing oblivious/non-symmetric FLOAT accumulation
     loops — they are `D-6.6-05`/`D-6.6-10` byte-identical locks; add a
     parallel arm, never inline a CTR branch into the middle of the existing
     bit-indexed loop bodies in a way that could perturb float-only output
     ordering.
6. **Rejected alternative:** unifying float and CTR handling into one generic
   "feature-id" abstraction across the whole `fstr.rs`/`shap.rs` module in one
   pass — rejected as too large for one TDD slice; the project's own
   established pattern (FSTR-03 split into core + facade; MODEL-05 SHAP family
   landing as ITS OWN later slice after float PVC/Interaction/LossChange) is
   consistently ONE narrow capability per slice.

## Project Impact Scope

### Must Change

- `crates/cb-model/src/fstr.rs` — `interaction()` gains a CTR accumulation
  arm; `feature_count()` may need to widen (or a new CTR-aware sibling
  function/return-shape may be introduced) depending on the index-space
  decision. **Reason:** this IS the feature. **Downstream:** the Rust facade
  `catboost-rs::Model::feature_importance(Interaction)`
  (`crates/catboost-rs/src/model.rs:149`) automatically benefits (calls
  `interaction()` directly, no facade code change needed) — but ANY facade/py
  documentation describing the returned index meaning should be checked for
  accuracy once CTR models are in scope.
- New oracle fixture directory under `crates/cb-oracle/fixtures/` (e.g.
  `fstr_interaction_ctr/` or similar — naming is a plan-time choice) with
  `gen_fixtures.py`, `config.json`, ground-truth `.npy`, and either a
  `model.json`/`.cbm` OR reuse of an existing `tensor_ctr_e2e`-style fixture.
- A new integration oracle test in `crates/cb-model/tests/` (new file, e.g.
  `fstr_interaction_ctr_oracle_test.rs`) OR an addition to the existing
  `fstr_oracle_test.rs` — plan-time choice, but the existing file's docstring
  ("MODEL-03 partial") suggests it is the natural home if kept small.

### May Change

- `crates/cb-model/src/fstr.rs::prediction_values_change()` — ONLY if the SPEC
  author decides to fold in the PVC-CTR fix in this same slice (see Open
  Questions #1). If deferred, this file's PVC function is untouched.
- `crates/cb-model/src/lib.rs` — only if new public symbols are added (e.g., a
  new error type or a new function name); currently `interaction` is already
  `pub use`d, so a signature-preserving change needs no `lib.rs` edit.

### Verification Only

- `crates/cb-model/src/apply.rs`, `crates/cb-model/src/ctr_data.rs`,
  `crates/cb-model/src/model.rs` — read for the already-correct CTR structural
  representation; not modified by this slice.
- `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs`,
  `multi_permutation_e2e_oracle_test.rs` — used as evidence/pattern source for
  building a CTR-bearing test model; not modified.
- `crates/catboost-rs/src/model.rs::feature_importance` — verify it still
  compiles/behaves against the (signature-preserving) change; not expected to
  need edits.

### Explicitly Out of Scope

- `crates/cb-model/src/shap.rs` (all of MODEL-05: `shap_values`,
  `shap_interaction_values`, `prediction_diff`, `sage_values`) — FSTR-02's
  dependency, not this slice's.
- `feat/23-ctr-model-loading` branch content (`cbm.rs` CTR decode,
  `ctr_data.rs` `decode_ctr_model_parts` family) — unrelated, unmerged, not a
  blocker, not touched.
- `catboost-rs-py` — no `get_feature_importance` Python surface exists yet at
  all; wiring it is a separate DX task by the project's own established
  pattern (FSTR-03 core vs facade split).

## Do Not Hand-Roll

- `cb_core::sum_f64` for every float reduction (D-08) — do not use
  `.iter().sum()` or a raw fold.
- `interaction_add` — the existing pair accumulator; do not build a parallel
  `HashMap<(usize,usize), f64>` (the project's own convention explicitly
  avoids a hash map here "so the iteration order is deterministic").
- `TProjection.cat_features` / `combined_hash` / `fold_cat_hash` — already
  exist in `cb-train`; do not recompute cat-feature membership by re-parsing
  a `CtrSplit`'s other fields.
- `cb_oracle::compare::assert_abs_close` — the mandated oracle comparator;
  do not hand-roll a tolerance check.
- The existing `tensor_ctr_e2e_oracle_test.rs` train→lift pattern — do not
  invent a new way to materialize a CTR-bearing `Model` for testing when this
  one is already proven ≤1e-5 against upstream.

## Common Pitfalls and Risks

| # | Trigger | Consequence | Prevention | Verification |
|---|---|---|---|---|
| 1 | Perturbing the existing oblivious/non-symmetric FLOAT accumulation loops while adding the CTR arm | Silently breaks the `D-6.6-05`/`D-6.6-10` byte-identical regression lock for ALL existing float-only models (a real regression, not just a missing feature) | Add the CTR arm as a clearly separate code path/function; run the existing `fstr_oracle_test.rs` unmodified before/after as a regression gate | `cargo test -p cb-model --test fstr_oracle_test` unchanged pass |
| 2 | Assuming the vendored `catboost-master/` tree contains the CTR-interaction algorithm source | It does not (`0` hits for `feature_str.cpp`/`calc_fstr.cpp`) — any comment citing exact upstream line numbers for the CTR-interaction rule is unverifiable locally | Fetch from GitHub (ideally the `v1.2.10` tag, matching the oracle's pinned Python version) via WebFetch, or fall back to empirical fixture-driven reverse engineering (train a small CTR model, dump upstream's actual Interaction output, iterate the Rust implementation until it matches ≤1e-5) — the project has used the empirical approach successfully before (CTR value materialization, HNSW parity) | Cross-check the resulting Rust logic's output against a real `catboost==1.2.10` fixture, not against reasoning alone |
| 3 | Building the CTR-bearing oracle fixture via the upstream `.cbm`/JSON CTR-reconstruction path (assuming it needs `feat/23`) | Wastes effort chasing an unmerged branch; also risks accidentally depending on unmerged code if a future rebase/merge brings it in prematurely | Use the ALREADY-PROVEN `cb_train::train_cat` → `Model::from_trained` pattern from `tensor_ctr_e2e_oracle_test.rs` instead, confirmed independent of `feat/23` | `git log feat/23-ctr-model-loading` vs current branch `git merge-base`; confirm `cbm.rs`/`ctr_data.rs` on the working branch have NO CTR-decode functions (`grep -n "decode_ctr_model_parts\|reconstruct_model.*Ctr"`) |
| 4 | Widening `feature_count()`'s returned vector length or changing what a `usize` index MEANS in the interaction output tuple, without updating `catboost-rs`/docs | A caller reading `(feature_i, feature_j, score)` before/after this change could silently misinterpret a categorical-feature index as a float-feature index (or vice versa) if the index space changes shape | Make the index-space decision explicit and documented in the SPEC (§4 "Feature-index space", following the FSTR-03 SPEC's own precedent of a dedicated load-bearing subsection); keep `n_cat == 0` (float-only) behavior byte-identical | New unit test asserting a float-only model's `interaction()` output is IDENTICAL pre/post change |
| 5 | `interaction_dfs`'s existing `(0,0)` pure-leaf-node sentinel guard (`fstr.rs:444-446`, `WR-04`) silently also matching a legitimate CTR split whose split fields happen to collide with `(0,0)` | An added CTR arm inserted before/around this guard could reintroduce the exact WR-04 bug the comment describes (a placeholder mistaken for real feature 0) | Read `fstr.rs:439-446`'s existing comment carefully before touching the non-symmetric DFS; keep the sentinel check ordered exactly as today | Re-run the existing non-symmetric interaction oracle assertions (`fstr_oracle_test.rs` item 4) |
| 6 | Treating `LossFunctionChange` as trivially "the same fix" | It requires `shap_values`/`shap_interaction_values` CTR support first (a materially larger, separate slice) — conflating scope risks an oversized, unreviewable PR | Keep FSTR-01 and FSTR-02 as separate specs/plans, per the project's existing FSTR-03 core/facade split precedent | N/A (scoping discipline) |
| 7 | Assuming `catboost` Python package is available in this dev container for fixture (re)generation | It is not installed by default; FSTR-03's own PLAN.md required installing `catboost==1.2.10` into a `uv`-managed venv before fixtures could be generated | Budget venv setup time in the plan; document the exact install recipe used previously (referenced in `fstr-03-partial-dependence/PLAN.md` T3) | `python3 -c 'import catboost'` (expect `ModuleNotFoundError` until venv is created) |

## Testing and Verification Strategy

- **Unit tests:** none strictly required if `interaction()`'s CTR arm is
  simple enough to be fully exercised by the oracle test, but a targeted unit
  test on a hand-built tiny `Model` with one `ModelSplit::Ctr` (single-cat
  projection) verifying the CTR split contributes SOME non-zero, correctly
  attributed pair is recommended (mirrors `ctr_split_scoring_test.rs`'s
  hand-built-split pattern) — lives in a sibling `fstr_test.rs` per the
  source/test separation rule (currently `fstr.rs` has NO existing sibling
  test file — confirm whether one needs to be created, or whether all its
  current coverage is exclusively via the two integration oracle tests;
  `[VERIFIED: find crates/cb-model/src -iname 'fstr_test.rs' → no result]`).
- **Integration/oracle tests:** new oracle test(s) under `crates/cb-model/tests/`
  reproducing upstream `get_feature_importance(type='Interaction')` on a
  CTR-bearing model at `≤1e-5`, PLUS confirmation the existing float-only
  `fstr_oracle_test.rs`/`advanced_fstr_oracle_test.rs` assertions are
  unchanged.
- **Regression tests:** re-run full `cb-model` and `cb-train` suites (the
  latter to confirm `tensor_ctr_e2e_oracle_test.rs` /
  `multi_permutation_e2e_oracle_test.rs` remain green, since this slice reuses
  their model-construction pattern without modifying them).
- **Migration/data checks:** none (no serialization format touched).
- **Security/performance/operational checks:** none beyond the standard
  clippy restriction-lint gate; no new I/O, no new external dependency.
- **Exact project commands (verified pattern from the immediately-preceding
  FSTR-03 slice, applicable here unchanged):**
  ```
  cargo test -p cb-model                     # unit + oracle for this slice
  cargo test -p cb-model -p cb-oracle         # + comparator
  cargo test -p cb-train                      # confirm tensor_ctr_e2e / multi_permutation_e2e unaffected
  cargo clippy -p cb-model --all-targets      # RESTRICTION-LINT GATE (unwrap/expect/panic/indexing denied)
  ```
  `cargo build -p cb-model` does NOT enforce the restriction lints — use
  `cargo clippy` as the actual gate `[PROJECT: fstr-03-partial-dependence/PLAN-CHECK.md MAJOR #2]`.

## Planning Guidance

- **Suggested work boundaries / ordering:**
  1. Design/lock the feature-index-space contract for CTR-bearing
     `interaction()` output (the load-bearing decision — write it as its own
     SPEC subsection, following the FSTR-03 precedent's "Feature-index space
     (load-bearing — read first)" pattern).
  2. Decide and record the PVC-CTR scoping question (fold in or defer —
     Open Question #1) BEFORE writing acceptance tests, since it changes the
     spec's acceptance-scenario count.
  3. Build (or confirm reuse of) a CTR-bearing oracle fixture model —
     reuse `tensor_ctr_e2e`'s trained-model pattern first; only reach for a
     fresh `gen_fixtures.py` + upstream-venv round-trip if a NEW upstream
     `Interaction` ground-truth array is needed (it will be, since no existing
     fixture currently ships an `Interaction` array for a CTR model).
  4. Implement the CTR accumulation arm as an ADDITIVE, isolated code path;
     do not touch the existing float arms.
  5. Oracle-verify, then regression-verify (existing float fstr + cb-train
     CTR e2e suites).
- **Dependencies between implementation tasks:** the fixture/venv setup (step
  3) can proceed in parallel with the index-space design (step 1) once step 2
  is settled, since fixture generation needs to know what ground-truth shape
  to dump.
- **Decisions the planner must preserve:**
  - `interaction()` stays dataset-free (no new `&Pool`/columns parameter).
  - The float-only accumulation arms stay byte-identical (`D-6.6-05`/`D-6.6-10`).
  - No CubeCL/GPU dependency enters `cb-model` (MODEL-02).
  - `anyhow` stays banned in `cb-model` (`D-14`).
- **Items requiring a spike or explicit decision before implementation:**
  - The exact upstream CTR-interaction-pair attribution rule (Highest-risk
    finding #1) — recommend a short empirical spike: train a minimal 2-cat-
    feature CTR model with `catboost==1.2.10`, dump
    `get_feature_importance(type='Interaction')`, and compare candidate Rust
    attribution rules against it directly, rather than trusting the
    `master`-HEAD-sourced C++ reading alone.
  - Whether PVC-CTR is in-scope (Open Question #1).
  - Whether the oracle test file is a new file or an addition to
    `fstr_oracle_test.rs` (naming/organization, non-blocking).

## Addendum (post-scoping decision + v1.2.10-pinned spike)

**Scoping decision (user-confirmed):** PredictionValuesChange's CTR gap IS
folded into this slice alongside Interaction. Both `interaction()` and
`prediction_values_change()` in `fstr.rs` gain CTR accumulation arms in this
one requirement (superseding "Open or Conflicting Requirements" #1 and Open
Question #1 above — resolved, in-scope).

**Open Question #2 RESOLVED** via a dedicated `WebFetch` against the
`v1.2.10` tag specifically (matching the pinned oracle version exactly, not
`master` HEAD) — upgrades this from LOW/MEDIUM confidence to
`[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/{feature_str.cpp,calc_fstr.cpp}, accessed 2026-07-17]`:

1. **`Interaction` (`CalcFeatureInteraction`, calc_fstr.cpp):** for each
   internally-accumulated `(FirstFeature, SecondFeature, Score)` triple, each
   side is expanded to its list of ORIGINAL (external) feature indices — a
   plain float/one-hot split expands to a 1-element list
   (`layout.GetExternalFeatureIdx(FeatureIdx, Float|Categorical)`); an
   `OnlineCtr` split expands to the FULL list of its projection's
   `BinFeatures` (→ external float idx) + `CatFeatures` (→ external cat idx)
   — NOT `OneHotFeatures` for this function (Interaction's expansion omits
   one-hot; contrast with PVC below, which DOES include `OneHotFeatures`).
   The pair's `Score` is then distributed over the FULL CROSS-PRODUCT of
   `(external_f0 ∈ side0_list) × (external_f1 ∈ side1_list)`, each cell
   getting `Score / (side0_list.len() * side1_list.len())`, self-pairs
   (`f0 == f1`) skipped, and `(f0,f1)` order-normalized (`f0 < f1`) before
   accumulating into a `HashMap`-equivalent keyed by the EXTERNAL index pair.
   Exact code (v1.2.10, verbatim):
   ```cpp
   TVector<TVector<int>> internalToRegular;
   for (const auto& internalFeature : features) {   // features = {FirstFeature, SecondFeature}
       TVector<int> regularFeatures;
       if (internalFeature.Type == ESplitType::FloatFeature) {
           regularFeatures.push_back(layout.GetExternalFeatureIdx(internalFeature.FeatureIdx, EFeatureType::Float));
       } else {
           auto proj = internalFeature.Ctr.Base.Projection;
           for (auto& binFeature : proj.BinFeatures) {
               regularFeatures.push_back(layout.GetExternalFeatureIdx(binFeature.FloatFeature, EFeatureType::Float));
           }
           for (auto catFeature : proj.CatFeatures) {
               regularFeatures.push_back(layout.GetExternalFeatureIdx(catFeature, EFeatureType::Categorical));
           }
       }
       internalToRegular.push_back(regularFeatures);
   }
   double effect = effectWithFeaturePair.Score;
   for (int f0 : internalToRegular[0]) {
     for (int f1 : internalToRegular[1]) {
       if (f0 == f1) continue;
       if (f1 < f0) DoSwap(f0, f1);
       sumInteraction[{f0, f1}] += effect / (internalToRegular[0].ysize() * internalToRegular[1].ysize());
     }
   }
   ```
   **Index space:** `layout.GetExternalFeatureIdx(idx, type)` returns a single
   FLAT index shared across float+categorical features in ORIGINAL DATASET
   COLUMN ORDER (CatBoost's "external"/regular feature numbering — the same
   numbering `feature_names`/`get_feature_importance` present to Python
   callers). This means **catboost-rs's `Model` needs an equivalent
   original-column-order flat index for cat features**, since today only
   `float_feature_borders: Vec<Vec<f64>>` (float-local index) is exposed —
   confirms this IS the central design decision (per original research body).

2. **`PredictionValuesChange` (`CalcRegularFeatureEffect`, calc_fstr.cpp,
   FULL body fetched verbatim):** does NOT use a merged flat external index
   internally — it accumulates into FOUR SEPARATE per-type-local vectors
   (`catFeatureEffect[catFeaturesCount]`, `floatFeatureEffect[floatFeaturesCount]`,
   `textFeatureEffect[...]`, `embeddingFeatureEffect[...]`), where an
   `OnlineCtr` split divides its effect EQUALLY (no cross-product — single
   projection, not a pair) across `proj.BinFeatures.size() +
   proj.CatFeatures.size() + proj.OneHotFeatures.size()` (NOTE: PVC's
   redistribution DOES include `OneHotFeatures`, unlike Interaction's
   expansion above) — `addEffect = effectWithSplit.first / featuresInSplit`,
   added to `floatFeatureEffect[binFeature.FloatFeature]` /
   `catFeatureEffect[catIndex or oneHotFeature.CatFeatureIdx]` respectively.
   The four vectors are THEN concatenated (cat first, then float, then text,
   then embedding — see exact loop order in code) into one
   `TVector<TFeatureEffect>` where each element is an explicit
   `(Score, EFeatureType, per-type-local-index)` TUPLE (not a flat merged
   integer), then sorted descending by score. **So PVC's C++-internal
   representation is typed (kind, local-index) — the "flat external index"
   only exists at the Python-facing layer** (feature_names ordering); this
   project's own `prediction_values_change()` returning `Vec<(usize, f64)>`
   today (float-local index, since only floats existed) should decide whether
   to keep a `(FeatureType, local_idx)`-shaped result or adopt the same flat
   external-index convention as Interaction for consistency — **recommend the
   SPEC pick the FLAT EXTERNAL INDEX for BOTH functions' output**, since (a) it
   is what Interaction already structurally requires, (b) a single consistent
   convention avoids a footgun where two sibling functions in the same module
   use different index semantics, and (c) it is what Python-facing
   `get_feature_importance` ultimately exposes for both types anyway.
   Exact code (v1.2.10, verbatim, full function):
   ```cpp
   TVector<TFeatureEffect> CalcRegularFeatureEffect(
       const TVector<std::pair<double, TFeature>>& internalEffect,
       const TFullModel& model)
   {
       int catFeaturesCount = model.GetNumCatFeatures();
       int floatFeaturesCount = model.GetNumFloatFeatures();
       int textFeaturesCount = model.GetNumTextFeatures();
       int embeddingFeaturesCount = model.GetNumEmbeddingFeatures();
       TVector<double> catFeatureEffect(catFeaturesCount);
       TVector<double> floatFeatureEffect(floatFeaturesCount);
       TVector<double> textFeatureEffect(textFeaturesCount);
       TVector<double> embeddingFeatureEffect(embeddingFeaturesCount);
       for (const auto& effectWithSplit : internalEffect) {
           TFeature feature = effectWithSplit.second;
           switch (feature.Type) {
               case ESplitType::FloatFeature:
                   floatFeatureEffect[feature.FeatureIdx] += effectWithSplit.first;
                   break;
               case ESplitType::OneHotFeature:
                   catFeatureEffect[feature.FeatureIdx] += effectWithSplit.first;
                   break;
               case ESplitType::OnlineCtr: {
                   auto& proj = feature.Ctr.Base.Projection;
                   int featuresInSplit = proj.BinFeatures.ysize() + proj.CatFeatures.ysize()
                       + proj.OneHotFeatures.ysize();
                   double addEffect = effectWithSplit.first / featuresInSplit;
                   for (const auto& binFeature : proj.BinFeatures) {
                       floatFeatureEffect[binFeature.FloatFeature] += addEffect;
                   }
                   for (auto catIndex : proj.CatFeatures) {
                       catFeatureEffect[catIndex] += addEffect;
                   }
                   for (auto oneHotFeature : proj.OneHotFeatures) {
                       catFeatureEffect[oneHotFeature.CatFeatureIdx] += addEffect;
                   }
                   break;
               }
               case ESplitType::EstimatedFeature: { /* text/embedding, out of scope here */ }
           }
       }
       TVector<TFeatureEffect> regularFeatureEffect;
       for (int i = 0; i < catFeatureEffect.ysize(); ++i)
           regularFeatureEffect.push_back(TFeatureEffect(catFeatureEffect[i], EFeatureType::Categorical, i));
       for (int i = 0; i < floatFeatureEffect.ysize(); ++i)
           regularFeatureEffect.push_back(TFeatureEffect(floatFeatureEffect[i], EFeatureType::Float, i));
       // + text, embedding (out of scope: catboost-rs has no text/embedding features per prior phases)
       Sort(regularFeatureEffect.rbegin(), regularFeatureEffect.rend(), /* descending by Score, tie-break by Feature.Index desc */);
       return regularFeatureEffect;
   }
   ```

**Net effect on "Recommended Architecture":** the central shared prerequisite
for BOTH `interaction()`'s and `prediction_values_change()`'s CTR arms is a
single new helper — an original-dataset-column-order flat index resolver
(`external_float_index(local_idx) -> usize`, `external_cat_index(local_idx)
-> usize`, or one combined `fn external_feature_index(kind, local_idx) ->
usize`) — mirroring upstream `TFeaturesLayout::GetExternalFeatureIdx`. This
resolver is the ONE new piece of shared infrastructure this slice must add to
`cb-model` (in `fstr.rs` or a small new module); everything else
(cross-product distribution for Interaction, equal-split redistribution for
PVC) is a mechanical per-function translation of the verbatim C++ above.
Since `Model` today has no explicit original-column-order record, the SPEC
must define how `external_feature_index` is derived — the most direct source
is training-time feature order (`cb-train`'s `TProjection`/feature layout),
which needs a CodeGraph-verified check at PLAN time for whether `Model`
already retains this order implicitly (e.g., via insertion order of
`float_feature_borders` interleaved with an implied cat-feature list) or
whether a new field must be threaded through `Model::from_trained`.

**Confidence upgrade:** Open Question #2 moves from LOW to
**HIGH** (`[VERIFIED: WEB v1.2.10 tag]`, exact pinned version, not `master`
drift risk). Recommend the SPEC still require an empirical fixture check
(train a 2-cat-feature CTR model, compare against real
`catboost==1.2.10` `get_feature_importance` output) as the acceptance test —
verifying the implementation, not the algorithm understanding, which is now
solid.

## Open Questions

1. Should this slice also fix `prediction_values_change()`'s identical CTR
   skip (no requirement ID currently owns that gap), given it shares the same
   underlying index-space machinery, or should that be split into its own
   requirement/slice? **Blocks:** final acceptance-scenario count and spec
   scope boundary.
2. What is the EXACT upstream `Interaction`-specific (not `PredictionValuesChange`'s
   already-documented "regular effect") CTR-to-original-feature attribution
   rule, verified against the pinned oracle version `1.2.10` specifically
   (not `master` HEAD, which is what this research pass could access)? A
   dedicated `WebFetch` of
   `https://github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/feature_str.cpp`
   (and `calc_fstr.cpp` at the same tag) is recommended before finalizing the
   SPEC's behavioral contract. **Blocks:** the core algorithm design (§ above,
   "Recommended Architecture" point 1).
3. Should the new oracle fixture be a brand-new `gen_fixtures.py`/fixture
   directory, or can it be added as new arrays alongside the existing
   `tensor_ctr_e2e/` fixture (same model, new ground-truth array)? Either is
   viable; a plan-time choice, not a research blocker.
4. Does the returned `Vec<(usize, usize, f64)>` tuple shape need to change
   (e.g., to disambiguate a categorical index from a float index when both
   spaces are combined), or is a single combined index space (float indices
   first, then categorical) sufficient and matches upstream's own "regular"
   (flat) feature index convention? Needs confirmation against upstream's
   actual `Interaction`-mode CLI/Python output feature-index convention
   (untested in this research pass beyond the PVC "regular effect" reading).

## Sources

- **PageIndex:** queried via `get_folder_structure`/`browse_documents` is
  unavailable in this session's tool set for this task (no PageIndex MCP tools
  were exposed in the available toolset for this research pass) — relied
  instead on the LOCAL, already-indexed `.planning/phases/.../SPEC.md`,
  `PLAN.md`, `PLAN-CHECK.md` files (read directly), which is the same
  authoritative content the FSTR-03 spec itself notes was PageIndex-indexed
  (`folder id cmrhcxbtm000104jr3i5jzm0m`). **Confidence impact:** LOWERED to
  MEDIUM for any claim that would otherwise rely on a fresh PageIndex query;
  all such claims are instead grounded in direct file reads (`[PROJECT: ...]`)
  which is an equally strong local-evidence source per the Source Priority
  order.
- **CodeGraph MCP** (`codegraph_explore`), multiple queries against
  `/home/user/Documents/workspace/catboost_rs`:
  - `crates/cb-model/src/fstr.rs` (full file read via CodeGraph + `Read`)
  - `crates/cb-model/src/model.rs` (`ModelSplit`, `CtrSplit`, `NonSymmetricTree`)
  - `crates/cb-model/src/apply.rs` (CTR apply path)
  - `crates/cb-model/src/shap.rs` (CTR-skip confirmation)
  - `crates/cb-model/src/ctr_data.rs` (`ECtrType`, `CtrValueTable`)
  - `crates/cb-train/src/projection.rs` (`TProjection`, `cat_features`, `fold_cat_hash`, `combined_hash`)
  - `crates/cb-oracle/src/model_json.rs` (`SplitJson`, `CtrTableJson`, `ModelJson::ctr_data`)
  - `crates/cb-model/src/lib.rs` (public export surface)
  - `crates/catboost-rs/src/model.rs` (facade `feature_importance`)
- **Local repository evidence (`Read`/`Bash`/`grep`):**
  - `git branch -a`, `git worktree list`, `git log --oneline`, `git merge-base
    feat/18-fstr03-partial-dependence feat/23-ctr-model-loading`, `git diff
    --stat` between the two branches — established the CTR-loading branch's
    unmerged status.
  - `git show a82289c:.planning/REQUIREMENTS.md` — recovered historical
    requirement text (FSTR-01/02/03 wording).
  - `find .planning/phases -type f` — enumerated existing SPEC/PLAN/PLAN-CHECK/
    SOURCES conventions (`17-model-export/`, `18-extended-feature-importance/`,
    `23-ctr-model-loading/`).
  - `crates/cb-model/tests/fstr_oracle_test.rs`,
    `crates/cb-model/tests/advanced_fstr_oracle_test.rs` (full/partial reads).
  - `crates/cb-oracle/fixtures/{tensor_ctr,tensor_ctr_e2e,plain_ctr,one_hot_cat,
    ordered_ctr,advanced_fstr,fstr_loss_change,feature_importance}` directory
    listings.
  - `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` (head read).
  - `find catboost-master -iname "feature_str*"` etc. → 0 results (absence
    proof the vendored tree lacks the fstr C++ source).
  - `crates/cb-model/Cargo.toml`, root `Cargo.toml` (`[workspace.dependencies]`
    pinned versions), `crates/cb-model/src/lib.rs` module list.
- **Context7 CLI:**
  - `npx --yes ctx7@latest library "CatBoost"` → resolved `/catboost/catboost`.
  - `npx --yes ctx7@latest docs "/catboost/catboost" "get_feature_importance
    type Interaction PredictionValuesChange LossFunctionChange categorical CTR
    feature index mapping"` → confirmed `get_feature_importance` signature,
    `type=EFstrType.Interaction`, dataset requirement differences between
    `Interaction`/`PredictionValuesChange`/`LossFunctionChange`.
- **Web (official):**
  - `https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/fstr/feature_str.cpp`
    (fetched via WebFetch, master HEAD, accessed 2026-07-17) — confirmed
    `CalcMostInteractingFeatures` has a `featureToIdx`-taking overload used for
    CTR-bearing models.
  - `https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/fstr/calc_fstr.cpp`
    (fetched via WebFetch, master HEAD, accessed 2026-07-17) — confirmed
    `GetFeatureToIdxMap` (one flat index per distinct `TFeature`, CTR or float)
    and `CalcRegularFeatureEffect`'s CTR-effect-redistribution formula
    (`addEffect = effectWithSplit.first / featuresInSplit`).
  - **Caveat:** both fetches were against `master`, NOT the project's pinned
    oracle version `1.2.10`; not yet cross-checked against the `v1.2.10` tag.
  - `WebSearch` "catboost feature_str.cpp CalcFeatureImportance CTR
    GetOriginalFeatureIndex combination projection interaction" — supporting,
    lower-value general search (official docs pages catboost.ai, no
    additional primary-source detail beyond the two WebFetches above).

## Confidence Assessment

- **HIGH** (directly verified by project evidence/authoritative docs):
  - CTR structural support (`ModelSplit::Ctr`, `apply.rs`, `Model::from_trained`)
    already exists and is oracle-tested for catboost-rs's own trained
    categorical models.
  - `interaction()`/`prediction_values_change()` currently skip CTR splits
    (exact line numbers cited).
  - `feat/23-ctr-model-loading` is unmerged into the current branch/working
    tree (`.planning/phases/23-ctr-model-loading/` working-tree copy lacks the
    "Implementation evidence" section present on the `feat/23` branch tip;
    `cbm.rs`/`ctr_data.rs` on the current branch contain no CTR-decode
    functions).
  - `shap.rs` is also CTR-blind (independent confirmation FSTR-02 is a larger,
    separate slice).
  - Existing project conventions (SPEC/PLAN/PLAN-CHECK/SOURCES structure,
    clippy-gate gotcha, source/test separation mount pattern, oracle-fixture
    pattern, `1e-5` bar) — read directly from the immediately-preceding
    `fstr-03-partial-dependence` and `fstr-03-facade-python` artifacts.
  - `TProjection.cat_features` exists and is exactly the data needed to
    resolve a CTR split's underlying original feature(s).
- **MEDIUM** (supported by multiple reliable sources, not exercised locally):
  - Upstream's `GetFeatureToIdxMap`/`CalcRegularFeatureEffect` CTR-splitting
    logic (fetched from GitHub `master`, not the pinned `1.2.10` tag; not
    executed/tested against a real fixture in this research pass).
  - `get_feature_importance(type='Interaction')` Python signature/behavior
    (Context7-sourced, official docs, not executed locally — no `catboost`
    package installed in this environment).
- **LOW** (incomplete/conflicting/unavailable evidence, needs validation):
  - The EXACT `Interaction`-specific (as opposed to `PredictionValuesChange`'s
    documented "regular effect") CTR attribution rule — only the
    `featureToIdx`-overload's EXISTENCE was confirmed, not its full internal
    logic, and not against the pinned oracle version. Recommend a dedicated
    spike (WebFetch against the `v1.2.10` tag, or empirical fixture-driven
    reverse engineering) before the SPEC finalizes PDP-style failure-isolated
    behavioral specifications for this feature.
  - Whether `PredictionValuesChange`'s CTR gap is in- or out-of-scope for this
    requirement (no requirement ID currently owns it — an explicit
    project-owner decision, not a research-answerable fact).
