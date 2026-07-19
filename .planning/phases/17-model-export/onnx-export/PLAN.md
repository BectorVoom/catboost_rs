# TDD Implementation Plan — EXPORT-01 ONNX Export

> ## ⚠️ Planner Agent unavailable — this PLAN.md is a labeled FALLBACK
> The spec-tdd-planner-skill instructs invoking a project-installed agent
> named `planner`. No agent named `planner` exists in this environment's
> agent registry. The only locally-defined planning-shaped agent
> (`specification-planner`, `~/.claude/agents/specification-planner.md`) reads
> `./planning/settings.json` and writes to `./planning/phase/phase-XX-name/`
> — that is the **GSD-style planning skill's own agent**, and the
> spec-tdd-planner-skill's guardrails explicitly forbid depending on any GSD
> skill/command/workflow/agent. It was therefore NOT invoked.
> `[UNVERIFIED: Planner Agent unavailable]`
>
> This PLAN.md was authored directly by the parent (spec-tdd-planner-skill)
> session instead, using the same requirements, the persisted research
> (`research.md`), and `SPEC.md` — mirroring the structure, task-graph
> convention, and Red/Green/Refactor density of the most recent verified
> precedent in this repo, `../18-extended-feature-importance/fstr-03-partial-dependence/PLAN.md`.
> It MUST still pass the independent Plan Checker gate (`plan-checker`) before
> being described as ready for implementation — see `PLAN-CHECK.md`.

**Phase:** 17 (Model Export — ONNX + CoreML) · **Slice:** EXPORT-01
**Spec:** `./SPEC.md` (specs EXPORT-01a..EXPORT-01f) · **Requirement:** EXPORT-01
**Research:** `./research.md`
**Crates:** `cb-model` (new `export` submodule) · `catboost-rs` (facade) ·
`catboost-rs-py` (PyO3) · **Impact:** `local` (straight-line dependency chain,
no new crate, no `cb-train`/`cb-backend` edge).
**Parity bar:** structural only, this slice (§2/§9 risk 1 of SPEC.md) — NOT
the ≤1e-5 numeric bar; numeric ONNX-Runtime validation is EXPORT-03, a later
plan.

> Executor contract: strict Red → Green → Refactor per task. One spec per TDD
> cycle. **Source/test separation is mandatory** — no inline `#[cfg(test)] mod
> tests { … }` body in production `.rs`; unit tests go in a sibling
> `_test.rs` file wired via the sanctioned
> `#[cfg(test)] #[path = "..._test.rs"] mod tests;` mount (mirrors
> `crates/cb-model/src/ctr_data.rs:58-61`, and FSTR-03's T0). Integration
> tests (facade/PyO3 round trips) live under `crates/cb-model/tests/`,
> `crates/catboost-rs/tests/`, and the Python test suite respectively. **No
> `unwrap`/`expect`/`panic`/`indexing_slicing`** in production
> (workspace-denied `[VERIFIED: LOCAL Cargo.toml:10-14]`). Do **not** mark any
> task complete during planning.

## Validation commands (host CPU; avoids env-red suites)

```
cargo test -p cb-model export                    # unit tests for the new export module
cargo test -p cb-model                           # full cb-model suite, no regression
cargo test -p catboost-rs                        # facade save_onnx integration
cargo clippy -p cb-model --all-targets           # RESTRICTION-LINT GATE (unwrap/expect/panic/indexing denied)
cargo clippy -p catboost-rs --all-targets        # same gate for the facade addition
cargo build -p cb-model -p catboost-rs           # compile check only — does NOT enforce clippy restriction lints
cargo build -p catboost-rs-py                    # REQUIRED (T6-0 gate, added after plan-checker pass 3) —
                                                  # `cargo test -p catboost-rs-py` alone is NOT sufficient: dev-deps
                                                  # are visible to `cargo test`'s build but NOT to this real build,
                                                  # so a dev-dependency-only cb-model reference compiles under
                                                  # `test` and fails here — this command is what actually catches it.
```

Python (`catboost-rs-py`) — build + pytest, per this repo's existing
convention (exact invocation to be confirmed at T6 time against
`crates/catboost-rs-py`'s CI/test config; e.g. `maturin develop` then
`pytest crates/catboost-rs-py/tests/`).

> **Lint-gate correction (carried from FSTR-03 PLAN.md):** the workspace
> restriction lints are **clippy** lints — inert under `cargo build`, enforced
> only by `cargo clippy`. Use `cargo clippy` as the authoritative gate.

**Known-red suites to ignore** (pre-existing, environmental, unrelated to
this slice): `cb-backend --lib` (CubeCL MLIR), `cb-train monotone_*`,
`catboost-rs-py` full suite under a `python3.14` link mismatch
`[VERIFIED: LOCAL memory catboost-rs-preexisting-test-failures.md]`. If T6's
Python integration tests hit this SAME pre-existing link failure (unrelated
to `save_onnx`), record it as environmental and verify the Rust-level
`cb-model`/`catboost-rs` tests (T0–T5) are green as the primary completion
evidence for the underlying logic; do not treat a pre-existing, unrelated
Python-link failure as this slice's regression.

## Task graph (dependencies, not file order)

```
T0 scaffold (prost dep + vendored/generated ONNX bindings + module skeleton)
  ├─> T1 (EXPORT-01a guard, unit) ─────────────────────────┐
  └─> T2 (EXPORT-01b per-tree node builder, unit) ─┬─> T3 (EXPORT-01c regressor assembly, unit) ─┐
                                                    └─> T4 (EXPORT-01d classifier assembly, unit) ─┼─> T5 (EXPORT-01e serialize+entrypoint, unit) ─> T6 (EXPORT-01f facade+PyO3) ─> T7 refactor/docs/gate
                                                                                                    ┘
                                                       (T1 also feeds T5 — the guard is exercised by T5's entrypoint)
```

- **Parallelizable:** T1 and T2 after T0 (T1 touches only guard predicates on
  `Model` metadata; T2 touches only single-tree node-array construction — no
  shared mutable state, disjoint code paths within `export/onnx.rs`, though
  the SAME file, so the executor should still land T1 and T2 as two separate
  sequential commits/cycles rather than literally concurrent edits, per this
  project's single-executor TDD convention — "parallelizable" here means
  *no dependency ordering requirement*, not literal concurrent editing).
- **Parallelizable:** T3 and T4 after T2 (regressor vs classifier assembly
  are mutually exclusive code paths selected by `is_classifier`, same caveat
  as above).
- **Serial spine:** T0 → {T1,T2} → {T3,T4} → T5 → T6 → T7.

---

## T0 — Scaffold: `prost` dependency, vendored ONNX bindings, module skeleton (enabler, no spec)

- **Goal:** every later task has a compiling `cb-model::export` module, a
  working `prost`-generated `onnx` message namespace, and the typed error
  shell — WITHOUT yet implementing any guard/graph-building logic. Per SPEC
  §9 risk 8, this is its OWN independently-verifiable prerequisite step: "the
  generated file compiles and exposes the needed message types" must be
  true before T1 starts, not discovered mid-implementation.
- **Files:**
  - `Cargo.toml` (workspace root or `crates/cb-model/Cargo.toml`, per existing
    convention — verify which level other per-crate-only deps like
    `flatbuffers` are declared at, `[VERIFIED: CODEGRAPH crates/cb-model/Cargo.toml]`)
    — add `prost = "0.14"` (pin the exact version resolved at scaffold time;
    record it in a comment, matching the "always use latest crate versions"
    project constraint).
  - **One-time, OUT-OF-BAND generation step** (not part of `cargo build`,
    mirrors the `flatc`-then-commit precedent, `[VERIFIED: LOCAL
    crates/cb-model/src/lib.rs:51-89]`): fetch the official ONNX project's
    `onnx.proto`/`onnx-ml.proto` at a PINNED tagged release from
    `github.com/onnx/onnx` (record the exact tag/commit in a header comment
    of the generated file), and generate Rust bindings via `protox`
    (pure-Rust `protoc` replacement — **no system `protoc` binary is
    installed in this environment**, confirmed by `which protoc` → not
    found, so `protox` avoids adding a new system/build-time tool
    requirement, consistent with the project's existing aversion to growing
    its native build chain, `[VERIFIED: LOCAL CLAUDE.md Platform
    Requirements]`) + `prost-build`, run as a throwaway local script/example
    (NOT committed as a `build.rs` — the generation tool is dev-time-only,
    exactly like `flatc` is not a `cb-model` build dependency today).
  - commit the generated output as
    `crates/cb-model/src/generated/onnx_generated.rs`, wired into `lib.rs`
    via the SAME `flatc_module!`-style pattern already used for
    `model_generated`/`features_generated`/`ctr_data_generated`
    (`[VERIFIED: LOCAL crates/cb-model/src/lib.rs:67-89]`) — i.e. a
    `#[path]` module with the same `#[allow(...)]` block for generated-code
    lint exemptions (non_snake_case, clippy::all/pedantic/nursery/restriction,
    etc.), since `prost`-generated code carries the same non-idiomatic-name
    profile as `flatc` output. Confirm at scaffold time whether the
    project's existing macro can be reused verbatim or needs a `prost`-
    specific variant (the message TYPES differ from FlatBuffers' generated
    shapes, but the LINT-EXEMPTION need is identical).
  - create `crates/cb-model/src/export/mod.rs` (or a flat
    `crates/cb-model/src/export_onnx.rs` — pick ONE per SPEC §4's note that
    naming is a plan-time choice; this plan uses the `export/mod.rs` +
    `export/onnx.rs` directory shape since CoreML/EXPORT-02 will be a
    sibling file, per research.md's crate-placement rationale) containing:
    - `OnnxExportError` enum with the FIVE SPEC §4 variants
      (`CategoricalFeaturesUnsupported`, `NonObliviousTreesUnsupported`,
      `RegionTreesUnsupported`, `Encode(#[from] prost::EncodeError)`,
      `Io(#[from] std::io::Error)`).
    - a private `fn is_onnx_exportable(model: &Model) -> Result<(), OnnxExportError>`
      stub (signature only — body deferred to T1) as the SINGLE guard
      chokepoint (SPEC §4 "Do Not Hand-Roll").
    - a stub `pub fn export_onnx(model: &Model, path: &Path, is_classifier: bool) -> Result<(), OnnxExportError>`
      signature only (NOT a placeholder body that would let a later Red test
      pass vacuously — leave it `todo!()`-free by having it call the stub
      guard and then return an explicit "not yet implemented" typed error
      is WRONG too, since that's not in SPEC §4; instead, per the FSTR-03 T0
      precedent, define ONLY the types + signatures this task, and let
      T1/T5 force the function body via failing tests — i.e. do not give T0
      a function BODY at all beyond what's needed to compile; a `fn
      export_onnx(...) -> Result<(), OnnxExportError> { unimplemented!() }`
      is acceptable ONLY behind the test-only clippy allow, and ONLY if no
      T0-authored test calls it — confirm no test in T0 exercises this fn).
  - create empty `crates/cb-model/src/export/onnx_test.rs`, mounted via
    `#[cfg(test)] #[path = "onnx_test.rs"] mod tests;` at the bottom of
    `export/onnx.rs` (mirrors FSTR-03 T0's mount pattern exactly — the
    silent-false-green trap this guards against is identical here).
  - edit `crates/cb-model/src/lib.rs` — add `mod export;` and (deferred to
    T7) the `pub use export::{export_onnx, OnnxExportError};` line.
- **Validation:** `cargo build -p cb-model` compiles with the new module +
  generated bindings; `cargo test -p cb-model export` runs (zero tests,
  green) proving the `#[path]` mount is wired.
- **Completion evidence:** `onnx_generated.rs` exists, is committed, and
  `crates/cb-model` compiles importing at least `ModelProto`, `GraphProto`,
  `NodeProto`, `AttributeProto` from it; `OnnxExportError` compiles with all
  five variants; the module + test-file mount exist.
- **Risk carried forward:** if `protox` cannot resolve/parse the official
  ONNX `.proto` files without a system `protoc` (SPEC §9 risk 8, MEDIUM
  confidence in research.md), this task may need to fall back to installing
  a `protoc` binary in the dev/CI environment — if so, record that as a NEW
  build-time requirement explicitly (a deviation from the "no new
  build-time tool" goal) rather than silently absorbing it.

## T1 — EXPORT-01a guard: typed rejection of unsupported models (unit)

- **Spec:** EXPORT-01a. **Depends on:** T0. **Parallel with:** T2.
- **Red** — in `export/onnx_test.rs`:
  - `rejects_non_symmetric_tree_model` (AT-01a-1): hand-build a `Model` with
    one non-empty `non_symmetric_trees` entry (empty `oblivious_trees`,
    `region_trees`, `ctr_data: None`) → `is_onnx_exportable(&model)` returns
    `Err(OnnxExportError::NonObliviousTreesUnsupported)`.
  - `rejects_region_tree_model` (AT-01a-2): hand-build a `Model` with one
    non-empty `region_trees` entry, `non_symmetric_trees` empty →
    `Err(RegionTreesUnsupported)`.
  - `rejects_ctr_split_model` (AT-01a-3): hand-build an all-oblivious `Model`
    with one `ObliviousTree` containing a `ModelSplit::Ctr(..)` split (a
    minimal hand-built `CtrSplit` — reuse whatever helper `cb-model`'s
    existing CTR unit tests already construct one with, e.g.
    `crates/cb-model/src/ctr_data_test.rs` or `apply_test.rs`'s CTR fixture
    helper if one exists; do NOT hand-roll a second CTR-split constructor —
    grep first) → `Err(CategoricalFeaturesUnsupported)`.
  - `rejects_baked_ctr_data_with_no_ctr_split` (AT-01a-4): all-oblivious,
    all-`ModelSplit::Float` model, but `ctr_data: Some(..)` (an empty/minimal
    `CtrData` value) → `Err(CategoricalFeaturesUnsupported)`.
  - `accepts_float_only_oblivious_model` (AT-01a-5): all-oblivious, all-float,
    `ctr_data: None` → `Ok(())`.
  - `guard_order_non_oblivious_wins_over_ctr` (AT-01a-6): a model with BOTH a
    non-empty `non_symmetric_trees` AND a CTR split present →
    `Err(NonObliviousTreesUnsupported)` (proves check-order slot 1, not
    slot 3, fires — a dedicated order-proof test, not just "is an error").
  - **Expected initial failure:** `is_onnx_exportable` unimplemented →
    compile error (T0 stub) then, once stubbed to compile, wrong/no `Err`.
- **Green:** implement `is_onnx_exportable` per SPEC §4's checked order:
  (1) `!model.non_symmetric_trees.is_empty()` →
  `NonObliviousTreesUnsupported`; (2) `!model.region_trees.is_empty()` →
  `RegionTreesUnsupported`; (3) `model.ctr_data.is_some()` OR any
  `ObliviousTree.splits` contains `ModelSplit::Ctr(_)` →
  `CategoricalFeaturesUnsupported`; (4) else `Ok(())`. Use `.iter().any(...)`
  over nested splits — no indexing, no `unwrap`.
- **Refactor:** consider hoisting to `Model::is_float_only_oblivious(&self)
  -> bool` on `model.rs` if a future CoreML guard (EXPORT-02, out of this
  slice) would obviously reuse it — SPEC §4 "Do Not Hand-Roll" flags this as
  a *consideration*, not a requirement of this slice; do NOT block T1 on
  speculative CoreML wiring. If deferred, leave a `// NOTE:` pointing future
  CoreML work at this function instead of a second detector.
- **Validation:** `cargo test -p cb-model export`.
- **Completion evidence:** AT-01a-1..6 all green.

## T2 — EXPORT-01b oblivious tree → ONNX node arrays (unit, structural)

- **Spec:** EXPORT-01b. **Depends on:** T0. **Parallel with:** T1.
- **Red** — in `export/onnx_test.rs`:
  - `reversed_split_order_matches_hand_computed_mapping` (AT-01b-1): hand-build
    a depth-2 `ObliviousTree` with THREE DISTINCT `(feature, border)` splits
    at distinct positions (e.g. `splits = [Float{feature:2,border:9.0},
    Float{feature:0,border:1.0}, Float{feature:1,border:5.0}]`, i.e. index 0,
    1, 2). Compute the fragment for `tree_id=0`. Assert, PER ONNX DEPTH
    LEVEL, that the internal node at ONNX depth `d` carries
    `nodes_featureids[d] == splits[len-1-d].feature` and
    `nodes_values[d] == splits[len-1-d].border` — i.e. depth 0 (root) uses
    `splits[2]` (feature 1, border 5.0), depth 1 uses `splits[1]` (feature 0,
    border 1.0). This is the DEDICATED regression test for the
    reversed-split-order pitfall (SPEC §9 risk 1) — hand-computed, not
    round-tripped through any oracle.
  - `leaf_values_transcribed_verbatim_no_permutation` (AT-01b-2): same tree,
    assert the fragment's per-leaf contribution array equals
    `tree.leaf_values` in order, unchanged.
  - `branch_gt_mode_and_complete_binary_child_indexing` (AT-01b-3): a
    depth-3 hand-built tree; assert every internal node's `nodes_mode ==
    "BRANCH_GT"`, every leaf node's `nodes_mode == "LEAF"`, and
    `nodes_falsenodeids[i] == 2*i+1`, `nodes_truenodeids[i] == 2*i+2` for
    every internal node id `i`.
  - `depth_zero_tree_is_single_leaf_node` (AT-01b-4): a tree with
    `splits: vec![]`, `leaf_values: vec![v]` → fragment has exactly one node
    (`LEAF`, id 0), zero internal nodes.
  - **Expected initial failure:** the per-tree builder function doesn't
    exist yet (T0 provides no stub for it — first Red here is a compile
    error, expected and recorded).
- **Green:** implement a private fn, e.g.
  `fn build_tree_nodes(tree: &ObliviousTree, tree_id: i64) -> TreeNodeFragment`
  (struct name/shape is this task's own choice — bundle the seven parallel
  arrays `nodes_treeids/nodeids/featureids/modes/values/truenodeids/
  falsenodeids` plus the per-leaf contribution values, e.g. as a small
  private struct, not yet the full `onnx::NodeProto` — T3/T4 assemble that).
  Walk ONNX depth `d` in `0..k` (`k = tree.splits.len()`), reading
  `tree.splits.get(k - 1 - d)` (checked, no raw indexing — use
  `.checked_sub`/`.get` so a malformed depth never panics even though it's
  unreachable in practice), computing `2^d` internal nodes per level via
  the standard complete-binary-tree enumeration, then appending the `2^k`
  leaf nodes reading `tree.leaf_values` in order.
- **Refactor:** factor the complete-binary-tree node-id arithmetic
  (`2*i+1`/`2*i+2`) into a tiny named helper if it's duplicated across the
  internal-node loop and any later multi-tree assembly code (T3/T4) —
  DEFER this specific refactor to T7 if T3/T4 end up needing the same
  arithmetic, to avoid premature abstraction before the second call site
  exists.
- **Validation:** `cargo test -p cb-model export`.
- **Completion evidence:** AT-01b-1..4 all green — AT-01b-1 in particular
  is the load-bearing proof against the HIGH-risk reversed-order pitfall.

## T3 — EXPORT-01c whole-ensemble regressor assembly (unit, structural)

- **Spec:** EXPORT-01c. **Depends on:** T2. **Parallel with:** T4.
- **Red** — in `export/onnx_test.rs`:
  - `two_tree_regressor_assembly_preserves_boosting_order` (AT-01c-1): a
    2-tree `Model` (each tree distinct, from T2-style hand-built trees) →
    assemble the `TreeEnsembleRegressor` attribute set; assert
    `nodes_treeids` contains blocks for tree ids `0` and `1` in that order,
    and each tree's block matches an INDEPENDENTLY computed `build_tree_nodes`
    call for that same tree (i.e. re-derive expected values in the test via
    the T2 fn directly, not by re-typing literals — this is an integration
    check of T2 called twice, not a re-verification of T2's own logic).
  - `zero_bias_omits_base_values_attribute` (AT-01c-2): `model.bias == 0.0`
    → assert the assembled attribute set has NO `base_values` entry (look up
    by attribute name and assert `None`, not `Some([0.0])`).
  - `nonzero_bias_sets_base_values` (AT-01c-3): `model.bias == 2.5` → assert
    `base_values == [2.5]`.
  - `fixed_regressor_attributes` (AT-01c-4): any in-scope model → assert
    `op_type == "TreeEnsembleRegressor"`, `domain == "ai.onnx.ml"`,
    `post_transform == "NONE"`, `n_targets == 1`.
  - **Expected initial failure:** the assembly function doesn't exist yet.
- **Green:** implement `fn build_regressor_node(model: &Model) ->
  onnx::NodeProto` (or an equivalent intermediate struct later converted to
  `NodeProto` in T5 — this task's call): iterate `model.oblivious_trees`
  with `enumerate()` as `tree_id`, call T2's `build_tree_nodes` per tree,
  concatenate the seven arrays in tree order, set `n_targets=1`,
  `post_transform="NONE"`, conditionally push the `base_values` attribute
  only when `model.bias != 0.0`.
- **Refactor:** none beyond removing any duplication with T2's per-tree
  loop; keep the concatenation logic simple (`Vec::extend`).
- **Validation:** `cargo test -p cb-model export`.
- **Completion evidence:** AT-01c-1..4 green — AT-01c-2 is the dedicated
  regression test for the MEDIUM-risk zero-bias `base_values` omission
  pitfall (SPEC §9 / research.md Common Pitfalls item 3).

## T4 — EXPORT-01d whole-ensemble classifier assembly + ZipMap (unit, structural)

- **Spec:** EXPORT-01d. **Depends on:** T2. **Parallel with:** T3.
- **Red** — in `export/onnx_test.rs`:
  - `binary_classifier_asymmetric_base_values` (AT-01d-1): a 1-dim
    (`approx_dimension==1`) model with `bias == 1.0` → assert
    `base_values == [-1.0, 1.0]` (the asymmetric pair — the DEDICATED
    regression test distinguishing this from the regressor's single-value
    form; a naive reuse of T3's bias logic would wrongly emit `[1.0]` here).
  - `binary_classifier_uses_logistic` (AT-01d-2): same model →
    `post_transform == "LOGISTIC"`.
  - `multiclass_classifier_uses_softmax` (AT-01d-3): a small hand-built
    2-tree, `approx_dimension == 3` model → `post_transform == "SOFTMAX"`.
  - `multiclass_class_weights_use_dimension_major_indexing` (AT-01d-3b,
    ADDED after plan-checker review — CRITICAL: `post_transform` alone does
    NOT catch a leaf/class transposition bug). Upstream's `AddTree` reads
    the flat leaf-value buffer **leaf-major**
    (`leaf_values[leaf*dim+class]`, `[VERIFIED: WEB onnx_helpers.cpp:428-479]`),
    but this port's `ObliviousTree.leaf_values` is **dimension-major**
    (`leaf_values[class*n_leaves+leaf]`,
    `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:299-306]`) — these
    coincide at `dim==1` but NOT at `dim>1`. For the SAME 3-class model as
    AT-01d-3, hand-compute the expected per-class contribution array
    directly from `tree.leaf_values` using the dimension-major formula (NOT
    by calling the exporter itself) and assert the emitted `class_weights`
    (or equivalent attribute) matches it exactly, per class per leaf. A
    naive extension of T2's single-dim "iterate `leaf_values` in order"
    transcription to the multiclass case will silently transpose leaf/class
    and FAIL this specific test (while AT-01d-3 alone would have passed).
    If the exact upstream attribute name/shape for per-class weights is
    unclear from a first read of `onnx_helpers.cpp`, re-read the multiclass
    branch of `AddTree`/`ConvertTreeToOnnxGraph` before implementing — do
    NOT guess the layout. If genuinely blocked after that re-read, narrow
    THIS SLICE to `is_classifier=true` limited to `approx_dimension==1`
    only, mark AT-01d-3/AT-01d-3b `#[ignore = "multiclass leaf-layout needs
    a dedicated upstream re-read, see SPEC.md EXPORT-01d indexing note"]`,
    and record this explicitly and prominently in T7's completion evidence
    — do NOT silently ship a multiclass path with only AT-01d-3 (which
    cannot detect the transposition bug) as its coverage.
  - `zipmap_wired_to_classifier_probability_output` (AT-01d-4): assert the
    assembled graph fragment includes a `ZipMap` node (domain
    `"ai.onnx.ml"`) whose input name equals the `TreeEnsembleClassifier`
    node's probability-tensor output name (an exact string-equality check
    on the wiring, not just "a ZipMap node exists somewhere").
  - `class_labels_default_and_explicit` (AT-01d-5): `class_to_label` empty →
    `classlabels_int64s == [0, 1]`; `class_to_label == [0.0, 1.0]` (or
    another explicit pair) → `classlabels_int64s` equals that, cast to
    `i64`.
  - **Expected initial failure:** the classifier assembly function doesn't
    exist yet.
- **Green:** implement `fn build_classifier_nodes(model: &Model) ->
  (onnx::NodeProto /* TreeEnsembleClassifier */, onnx::NodeProto /* ZipMap */)`:
  reuse T2's `build_tree_nodes` per tree (same node-array transcription,
  extended per-tree fragment to carry `dim`-many target weights per leaf
  when `approx_dimension > 1`, per SPEC EXPORT-01d's multiclass note); set
  `post_transform` from `model.approx_dimension` (`1` → `"LOGISTIC"`,
  `>1` → `"SOFTMAX"`); for the binary case, when `model.bias != 0.0`, set
  `base_values = [-model.bias, model.bias]`; derive `classlabels_int64s`
  from `model.class_to_label` (cast `f64 as i64`, checked/lossless for the
  integer-valued labels this port stores) with an explicit `[0, 1]` default
  when empty; construct a sibling `ZipMap` node whose `input` is the
  classifier node's declared probability output name. **Per-class leaf
  contribution indexing (mandatory, see AT-01d-3b):** read
  `tree.leaf_values[class * n_leaves + leaf]` (dimension-major, per
  `crates/cb-model/src/model.rs:299-306`'s own documented layout) when
  emitting the class-`class` contribution for `leaf` — do NOT reuse T2's
  single-dim leaf iteration unmodified for `dim > 1`. Also set the
  `probability_tensor` output `ValueInfoProto`'s second dimension to `2`
  when `approx_dimension == 1` (matching upstream's `dims==1 ? 2 : dims`
  rule, `[VERIFIED: WEB onnx_helpers.cpp:526-531]`), not `1`.
- **Refactor:** factor any duplication between T3's and T4's per-tree
  concatenation loop into a small shared helper ONLY if the duplication is
  exact (both just concatenate `build_tree_nodes` output) — if the
  multiclass per-class target-weight extension makes T4's loop genuinely
  different from T3's, do NOT force a shared abstraction; keep them
  separate (SPEC's own "mutually exclusive by `is_classifier`" framing).
- **Validation:** `cargo test -p cb-model export`.
- **Completion evidence:** AT-01d-1..5 green (or AT-01d-3 explicitly
  `#[ignore]`d with the documented reason, per the note above — not silently
  skipped).

## T5 — EXPORT-01e metadata + serialization + public entry point (unit)

- **Spec:** EXPORT-01e. **Depends on:** T1, T3, T4.
- **Red** — in `export/onnx_test.rs`:
  - `guard_failure_writes_no_file` (AT-01e-1): a guard-failing model (e.g.
    the AT-01a-3 CTR model) + a `tempfile`/`tempdir`-provided path → call
    `export_onnx(&model, &path, false)`; assert `Err(_)` AND `!path.exists()`
    afterward (confirms no partial/empty file is created — SPEC's
    "guard runs to completion before any byte is written" invariant).
  - `regressor_round_trips_through_encode_decode` (AT-01e-2): a
    guard-passing multi-tree model → `export_onnx(&model, &path, false)` →
    `Ok(())`; read the file back, `prost::Message::decode` into
    `onnx::ModelProto`; assert `ir_version == 3`, exactly one
    `opset_import` entry `{domain: "ai.onnx.ml", version: 2}`, and that the
    decoded `TreeEnsembleRegressor` node's attributes match what T3's unit
    tests independently asserted for the SAME model (re-derive expected
    values via T3's `build_regressor_node` directly in the test, not
    hand-typed literals).
  - `classifier_round_trips_through_encode_decode` (AT-01e-3): same pattern
    with `is_classifier=true`, asserting the decoded graph contains BOTH the
    `TreeEnsembleClassifier` and `ZipMap` nodes matching T4's independently
    asserted output, AND (added after plan-checker review) that the decoded
    `probability_tensor` output's `ValueInfoProto` second dimension is `2`
    for a binary (`approx_dimension==1`) model.
  - `unwritable_path_returns_typed_io_error` (AT-01e-4): a guard-passing
    model + a path whose parent directory does not exist → `Err(OnnxExportError::Io(_))`,
    no panic.
  - **Expected initial failure:** `export_onnx` is still the T0 stub.
- **Green:** implement `export_onnx`: call `is_onnx_exportable` (T1) first,
  propagate its `Err` immediately (no file touched yet); branch on
  `is_classifier` to call T3's or T4's assembly fn(s); wrap the resulting
  node(s) into a `GraphProto` with the appropriate `input`/`output`
  `ValueInfoProto`s; build the full `ModelProto` (`ir_version=3`,
  `opset_import=[{domain:"ai.onnx.ml", version:2}]`, producer metadata);
  `prost::Message::encode` to a `Vec<u8>`; write via `std::fs::write` (or
  `File::create`+`Write::write_all`), propagating I/O errors via `?`
  (`#[from] std::io::Error` on `OnnxExportError::Io`).
- **Refactor:** none beyond ensuring the guard check is textually the FIRST
  statement in the function body (a structural/readability property this
  task's own code review should confirm, not just tested behaviorally).
- **Validation:** `cargo test -p cb-model export`.
- **Completion evidence:** AT-01e-1..4 green; `export_onnx` is the module's
  sole public entry point (verified by the T7 `pub use`).

## T6 — EXPORT-01f facade + Python surfacing

> **[ADDED after plan-checker review, pass 3 — BLOCKING prerequisite,
> applied directly without a 4th checker pass since this planning process
> caps automated revision at 3 passes; T0–T5 and the pass-1/pass-2 fixes
> above ARE checker-confirmed, but this specific step is NOT independently
> re-verified. Flag it for review at implementation time.]**
>
> **T6-0 — Promote `cb-model` from dev-only to a real `catboost-rs-py`
> dependency (do this FIRST, before any other T6 step).** Plan-checker pass
> 3 found that the `to_pyerr` `Export`-arm code below (Green PyO3 step)
> names `cb_model::OnnxExportError` directly inside
> `crates/catboost-rs-py/src/errors.rs`'s `to_pyerr` — a **production**
> function, not `#[cfg(test)]`-gated. But `crates/catboost-rs-py/Cargo.toml`
> currently declares `cb-model` under `[dev-dependencies]` only (with a
> comment stating it exists "only for `errors_test.rs`")
> `[VERIFIED: LOCAL crates/catboost-rs-py/Cargo.toml:38-46]`. Cargo's
> dev-dependency extern prelude is populated for `cargo test` compilation
> but NOT for the normal `--lib`/`cdylib` build `cargo build -p
> catboost-rs-py` / `maturin develop` / `maturin build` actually invoke —
> so `cargo test -p catboost-rs-py` would PASS while the real wheel build
> FAILS to compile, for every PyO3 method (`to_pyerr` is the shared
> chokepoint), not just `save_onnx`. Fix: move the `cb-model` line from
> `[dev-dependencies]` to `[dependencies]` in
> `crates/catboost-rs-py/Cargo.toml`, PRESERVING `default-features = false`
> (still required — prevents this dependency re-enabling `cb-backend`'s
> default `cpu` feature via `cb-model -> cb-train -> cb-backend` and
> unifying `cpu`+`rocm` under a workspace `--features rocm` build, the WR-03
> feature-unification landmine — re-verify this rationale holds for a
> REGULAR dependency, since the existing comment was written for the
> dev-dependency case only). Add `cargo build -p catboost-rs-py` (the real
> production build, NOT `cargo test`) to this task's Validation and to the
> plan-wide Validation commands section, specifically so a dev-dependency-
> vs-regular-dependency defect of this shape cannot hide behind `cargo
> test`'s inflated dependency graph again. Update SPEC.md §7 "Modified" list
> to include `crates/catboost-rs-py/Cargo.toml`.

- **Spec:** EXPORT-01f. **Depends on:** T5, T6-0 (above).
- **Red (Rust facade):**
  - `save_onnx_delegates_and_succeeds` (AT-01f-1a): a facade `Model` loaded
    from an existing float-only `.cbm` fixture (reuse an existing numeric
    fixture already committed under `crates/cb-oracle/fixtures/`, e.g. the
    same `numeric_tiny`-trained model FSTR-03 uses — do NOT create a new
    fixture for a structural test) → `.save_onnx(path, false)` → `Ok(())`,
    file exists. Location: `crates/catboost-rs/tests/onnx_facade_test.rs`
    (an external integration-test file), matching the existing
    `partial_dependence_facade_test.rs` precedent — this test only needs
    the already-`pub` `Model::load_cbm`, so the external-crate boundary is
    not a problem here.
  - `save_onnx_maps_guard_error` (AT-01f-1b): **[CORRECTED after
    plan-checker review, pass 2 — location is now MANDATED, not left open;
    the pass-1 fallback of "add a `#[cfg(test)]`-only constructor" is
    DELETED, it does not work.]** `crates/cb-model/src/cbm.rs`'s
    `reconstruct_model` and `crates/cb-model/src/json.rs`'s `from_doc` BOTH
    unconditionally set `ctr_data: None` and never construct
    `ModelSplit::Ctr` (CTR-model *loading* is separate, not-yet-merged work
    on `feat/23-ctr-model-loading`) — so no fixture loadable via
    `load_cbm`/`load_json` today can exercise the rejection path. This test
    hand-constructs a `cb_model::Model` value with a literal
    `ModelSplit::Ctr` split (same technique as EXPORT-01a's AT-01a-3) and
    wraps it via `Model::from_canonical` (`crates/catboost-rs/src/model.rs:38`,
    confirmed `pub(crate)`). **`pub(crate)` is reachable from an INTERNAL
    `#[cfg(test)]` module compiled as part of the `catboost-rs` crate, but
    NOT from `crates/catboost-rs/tests/` (a separate integration-test
    binary linking the library's normal, non-`cfg(test)` build).** This
    test therefore MUST live as an internal `#[cfg(test)]`-mounted module —
    mirror the existing `crates/catboost-rs/src/lib.rs:50-51 mod
    error_test;` precedent exactly: add `#[cfg(test)] mod onnx_test;` to
    `lib.rs` (or fold this one test into `error_test.rs` if that reads more
    naturally — a plan-time naming choice, not a constraint) and create
    `crates/catboost-rs/src/onnx_test.rs` (or the equivalent). Assert
    `.save_onnx(path, false)` → `Err(CatBoostError::Export(_))`.
  - **Expected initial failure:** `Model::save_onnx` doesn't exist / doesn't
    compile.
- **Green (Rust facade):**
  - `crates/catboost-rs/src/error.rs` — add the `Export(#[from]
    cb_model::OnnxExportError)` variant (exact text per SPEC §4).
  - `crates/catboost-rs/src/model.rs` — add
    `pub fn save_onnx(&self, path: &Path, is_classifier: bool) -> Result<(), CatBoostError>`
    delegating to `cb_model::export_onnx(&self.inner, path, is_classifier)?`
    (mirrors `save_cbm` at `model.rs:224-227` exactly).
- **Red (PyO3):**
  - `unfitted_regressor_save_onnx_raises_not_fitted` (AT-01f-2, pytest): an
    unfitted `CatBoostRegressor().save_onnx(path)` raises the same
    `NotFittedError` every other unfitted-estimator method raises.
  - `fitted_regressor_save_onnx_writes_file` (AT-01f-3, pytest): fit a
    `CatBoostRegressor` on numeric-only synthetic data, `.save_onnx(path)`,
    assert the file exists and is non-empty.
  - `fitted_classifier_save_onnx_emits_classifier_graph` (AT-01f-4, pytest):
    fit a `CatBoostClassifier` (Logloss default) on numeric-only synthetic
    binary data, `.save_onnx(path)`; assert (via a small Rust-side helper
    exposed for the test, OR a lightweight Python-side protobuf peek if
    `onnx`/`protobuf` happens to be available in the test venv — resolve
    which approach at task time and record the choice) that the graph
    contains `TreeEnsembleClassifier`+`ZipMap`, not `TreeEnsembleRegressor`.
  - `export_error_variants_map_to_correct_python_exceptions` (AT-01f-5, Rust
    unit test in `crates/catboost-rs-py/src/errors.rs`'s existing test
    module — added after plan-checker review, pass 2): assert
    `to_pyerr(&CatBoostError::Export(cb_model::OnnxExportError::CategoricalFeaturesUnsupported))`
    is a `CatBoostValueError`; same for the `NonObliviousTreesUnsupported`/
    `RegionTreesUnsupported` sub-variants; the `Io` sub-variant maps to
    `PyIOError`; the `Encode` sub-variant maps to base `CatBoostError` —
    one assertion per sub-variant, mirroring this file's existing
    per-variant test style (e.g. however `PartialDependence`'s mapping is
    already asserted, if it is).
  - **Expected initial failure:** `save_onnx` PyO3 method doesn't exist on
    either estimator; `to_pyerr` doesn't have an `Export` arm (compile
    error).
- **Green (PyO3):**
  - **FIRST** (mandatory prerequisite, SPEC §9 risk 7 — confirmed by
    plan-checker review to be a certain compiler requirement, not a "verify
    and maybe fix" hedge): `to_pyerr` in
    `crates/catboost-rs-py/src/errors.rs:104-116` is an EXHAUSTIVE `match`
    over all 7 current `CatBoostError` variants with NO wildcard arm —
    adding `CatBoostError::Export` as an 8th variant WILL NOT COMPILE until
    a matching arm is added. Add:
    ```rust
    FacadeError::Export(e) => match e {
        cb_model::OnnxExportError::CategoricalFeaturesUnsupported
        | cb_model::OnnxExportError::NonObliviousTreesUnsupported
        | cb_model::OnnxExportError::RegionTreesUnsupported => {
            CatBoostValueError::new_err(e.to_string())
        }
        cb_model::OnnxExportError::Io(io) => PyIOError::new_err(io.to_string()),
        cb_model::OnnxExportError::Encode(_) => CatBoostError::new_err(e.to_string()),
    },
    ```
    (per the mapping SPEC EXPORT-01f now specifies) BEFORE writing either
    PyO3 method below.
  - `crates/catboost-rs-py/src/regressor.rs` — add
    ```rust
    fn save_onnx(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let model = self.base.model.as_ref().ok_or_else(|| {
            not_fitted_err(py, "this CatBoostRegressor is not fitted yet; call `fit` before `save_onnx`")
        })?;
        py.detach(|| model.save_onnx(std::path::Path::new(path), false))
            .map_err(PyCbError)?;
        Ok(())
    }
    ```
  - `crates/catboost-rs-py/src/classifier.rs` — the same shape with
    `not_fitted_err(..., "...before `save_onnx`")` and
    `model.save_onnx(std::path::Path::new(path), true)`.
- **Refactor:** none — both PyO3 methods are intentionally near-duplicates
  differing only in the hardcoded bool and the not-fitted message; this
  mirrors the existing `predict`/`partial_dependence` per-estimator
  duplication pattern already present in this crate (`[VERIFIED: CODEGRAPH
  crates/catboost-rs-py/src/{regressor,classifier}.rs]`), not a new smell.
- **Validation:** `cargo build -p catboost-rs-py` (REAL production build —
  the T6-0 regression gate; must be run, not skipped in favor of `cargo
  test` alone) → `cargo test -p catboost-rs`; `cargo test -p catboost-rs-py
  errors::` (or equivalent) for AT-01f-5; Python build + pytest per the
  commands resolved at task time (see Validation commands section note).
- **Completion evidence:** AT-01f-1a/1b (Rust, with 1b in the internal
  `#[cfg(test)]` module), AT-01f-2..4 (pytest), and AT-01f-5 (Rust unit)
  all green.

## T7 — Refactor, public export, docs, full-slice gate

- **Depends on:** T1–T6.
- **Steps:**
  - `crates/cb-model/src/lib.rs` — `pub use export::{export_onnx, OnnxExportError};`.
  - Module-level doc comment on `export/onnx.rs` citing the upstream source
    (`onnx_helpers.cpp`) for the reversed-split-order + `BRANCH_GT` +
    `base_values` gating conventions, and stating the float-only/oblivious
    scope + the four typed rejections, mirroring `partial_dependence.rs`'s
    module doc density (`[VERIFIED: LOCAL crates/cb-model/src/partial_dependence.rs:1-52]`).
  - Resolve any deferred refactors flagged in T2/T3/T4 (the complete-binary-tree
    child-index helper, the T3/T4 shared-concatenation question) NOW that
    both call sites exist — extract only if the duplication is exact.
  - If AT-01d-3 (multiclass SOFTMAX) was `#[ignore]`d at T4, either resolve
    it now (re-read upstream, implement, un-ignore) or explicitly carry it
    forward in this task's completion evidence as a documented, intentional
    gap — do not leave it silently ignored without a note.
  - Confirm no `.unwrap()`/`.expect()`/`panic!`/raw-indexing crept into any
    new production file — grep as a smell check, `cargo clippy` as the
    authoritative gate.
- **Validation (full slice):**
  ```
  cargo clippy -p cb-model --all-targets      # restriction-lint gate (authoritative)
  cargo clippy -p catboost-rs --all-targets
  cargo build -p cb-model -p catboost-rs
  cargo test -p cb-model
  cargo test -p catboost-rs
  ```
  Plus the Python build+pytest command resolved in T6.
- **Completion evidence:** all EXPORT-01a..f acceptance tests green (or
  explicitly documented gap per above); restriction lints clean; SPEC §9
  risks 1/2/7/8 marked resolved or explicitly carried forward with a
  reason. Then flip the EXPORT-01 requirement checkbox in the (git-recovered,
  off-tree) `.planning/REQUIREMENTS.md` / `ROADMAP.md` — bookkeeping only,
  outside TDD; confirm the canonical revision before flipping, per the
  FSTR-03 PLAN.md precedent's own caveat.

## Traceability (task → spec → acceptance)

| Task | Spec | Acceptance tests | Kind |
|------|------|-------------------|------|
| T0 | (enabler) | compiles, generated bindings present | — |
| T1 | EXPORT-01a | AT-01a-1..6 | unit |
| T2 | EXPORT-01b | AT-01b-1..4 | unit (structural) |
| T3 | EXPORT-01c | AT-01c-1..4 | unit (structural) |
| T4 | EXPORT-01d | AT-01d-1..5 | unit (structural) |
| T5 | EXPORT-01e | AT-01e-1..4 | unit (structural, round-trip) |
| T6 | EXPORT-01f | AT-01f-1a/1b (Rust) + AT-01f-2..4 (pytest) + AT-01f-5 (Rust unit) | unit/integration |
| T7 | all | full-slice green | gate |

Every SPEC acceptance behavior has a Red task; every task references ≥1 spec ID.

## Deferred / explicitly out of this plan (see SPEC.md §2, §9)

- CoreML export (EXPORT-02) — separate future plan.
- Numeric ONNX-Runtime oracle validation (EXPORT-03) — separate future plan;
  requires adding `onnxruntime` to `crates/cb-oracle/generator/requirements.txt`
  (explicitly NOT touched by this plan).
- `CatBoostRanker.save_onnx` — not scoped into T6; a natural, low-risk
  follow-up if a future plan wants it (identical `is_classifier=false`
  shape to the regressor).
