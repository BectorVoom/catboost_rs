---
title: One-hot categorical training in the cb-train engine (device path + upstream-compatible model)
status: draft
format: markdown
spec_version: 4
updated_at: 2026-07-31T00:00:00Z
amendments:
  - "v2 (2026-07-31, from PLAN-CHECK.md pass 1): SPEC-OH-22 corrected to name BOTH scorers (production is find_optimal_split_partition_kernel, not find_optimal_split_kernel), to bound the fold by real bins, and to require lifting the trailing-border exclusion in both the kernel and the host belt; SPEC-OH-15 widened from shap_values to all four float_splits_of consumers and made explicit that float_splits_of must become fallible."
  - "v3 (2026-07-31, from PLAN-CHECK.md pass 2): the `one_hot == false` invariant in SPEC-OH-22 and SPEC-OH-23 changed from 'the kernel is byte-identical' to 'the kernel's OUTPUT is numerically identical'. Adding a #[comptime] parameter changes the generated kernel source by construction, so kernel byte-identity is not a testable property. Byte-identity claims about the trained MODEL on float-only paths (SPEC-OH-31) are unaffected and remain."
  - "v4 (2026-07-31, from PLAN-CHECK.md pass 3): SPEC-OH-22's real-bin bound no longer says `border < folds[feature]`. TCFeature.folds holds the uniform PADDED line width on the production path (session.rs:1363 -> cindex.rs:217), so that bound degenerated into the loop bound and admitted every padded bin. The constraint now mandates a separate real-cardinality array and explicitly forbids the unsafe repair of changing n_buckets_per_feature (which would alter packed words for every pool via feature_bits and break SPEC-OH-31)."
source_requirements:
  - "User request (2026-07-31): implement the unimplemented catboost engine and win on speed; item 1 of 3 (one-hot, then dead CTR params, then absent BoostParams)"
  - ".planning/plans/one-hot-categorical-training/research.md"
  - "Locked user decisions (2026-07-31): fix the mixed-kind split-order loss as an in-scope prerequisite; lift the device-eligibility blockers so one-hot reaches the GPU"
---

# One-hot categorical training in the `cb-train` engine

## 1. Context

`grow_one_hot_tree` (`crates/cb-train/src/tree.rs:3141`) exists with unit tests and
an oracle test, but has **no production caller** — its only non-test reference is a
re-export at `crates/cb-train/src/lib.rs:107`
`[VERIFIED: LOCAL research.md §1.1]`. `train_inner` never calls it, and
`params.one_hot_max_size` is read only at `boosting.rs:2711` and `:2725`, solely to
**exclude** low-cardinality categorical columns from CTR eligibility
`[VERIFIED: LOCAL research.md §1.2]`.

Consequently a column that `route_categorical` (`crates/cb-train/src/candidates.rs:92-104`)
sends to `EncodingPath::OneHot` — i.e. `1 < cardinality <= one_hot_max_size` — enters
no CTR projection **and** no one-hot split. It contributes nothing to the trained
model, silently. With the `CatBoostBuilder` default `one_hot_max_size = 2`, every
binary categorical column is silently dropped, while upstream CatBoost one-hot
encodes it. This breaks the ≤1e-5 upstream-parity bar on the most common
categorical data shape.

The existing `crates/cb-train/tests/one_hot_oracle_test.rs` does not detect this: it
builds its **own** test-local one-hot boosting driver rather than exercising
production code, and consumes no fixture at all — `crates/cb-oracle/fixtures/one_hot_cat/`
contains no model `[VERIFIED: LOCAL research.md §1.3, §4.1]`.

### 1.1 The pre-existing mixed-kind order defect (in scope by locked decision)

`cb_model::Model::from_trained` (`crates/cb-model/src/model.rs:326-363`) emits **all
float splits, then all CTR splits**, discarding the trainer's `GrownTree.level_kinds`,
which records the true per-level interleaving. Meanwhile
`leaf_index_for` (`crates/cb-model/src/apply.rs:208-215`) walks `tree.splits` in
**stored order** and treats split `i` as leaf-index bit `i`, and `.cbm` save/load
preserve that order 1:1 `[VERIFIED: LOCAL research.md §3.1-3.2]`.

For a tree whose level 0 is a CTR split and level 1 a float split, `level_kinds =
[Ctr, Float]` but the model stores `[Float, Ctr]`, transposing leaf indices 1 and 2 —
a genuine mis-prediction. The `tensor_ctr_e2e_oracle_test` suite passes 3/3 today
`[VERIFIED: local run 2026-07-31]`, so the defect is **latent, not currently
triggered** by committed fixtures. It is nonetheless on the live code path, and
one-hot must use the same lift. Copying the pattern would reproduce the defect for a
new split kind, so fixing it is a prerequisite, not an optional cleanup.

### 1.2 The value-space landmine

Upstream `.cbm` stores one-hot split values as raw `i32` `calc_cat_feature_hash`
outputs in `TOneHotFeature.Values`, **not** as bin ordinals — empirically confirmed by
producing a real catboost 1.2.10 one-hot model and byte-scanning it
`[VERIFIED: LOCAL research.md §4.4]`. The trainer-side `OneHotSplit.value: u32` is a
`cb_data::PerfectHash` **bin** (`crates/cb-train/src/tree.rs:125-136`). These are two
different value spaces. If both of our sides use the perfect-hash bin, our own oracle
passes while the produced `.cbm` is wrong for upstream — a silent interoperability
break that no in-repo test would catch.

### 1.3 Consumers that accept a new variant silently

`cb_model::ModelSplit` documents that every consumer matches exhaustively so no
consumer silently drops a split (T-05-09-03). That is **not true today** for six
sites, which would accept a third variant with no compiler error
`[VERIFIED: LOCAL research.md §3.3]`:

| site | today's behavior with a `OneHot` variant |
|---|---|
| `export/onnx.rs:106-113` guard | `matches!(.., Ctr(_))` → passes the guard, then `as_float()` → `None` → emits `(feature = 0, border = 0.0)` |
| `export/coreml.rs:106-113` guard | identical silent pass and `(0, 0.0)` emit |
| `json.rs:474-485` | `filter_map(as_float)` → silently DROPS the split, shortening the list → wrong depth/leaf indexing |
| `shap.rs:550-552` | `filter_map(as_float)` → silently drops → `tree_depth` shrinks → wrong SHAP |
| `fstr.rs:462-473` | `_ => false` wildcard → two one-hot splits on the SAME cat feature judged different internal features |
| `gpu_apply.rs:50-57` | guard passes, then the exhaustive match errors with a confusing message |

## 2. Scope and non-goals

### In scope
1. Ordered mixed-kind split representation end to end (trainer → model → `.cbm`),
   fixing the §1.1 order loss.
2. One-hot columns genuinely participate in oblivious tree growth via a **fused**
   level search (not `score_candidate_any`, see §9 R4).
3. `ModelSplit::OneHot` carrying the upstream raw-hash value space, applying
   correctly, and round-tripping through `.cbm`.
4. Deliberate handling at every consumer in §1.3 — a correct arm where implementable,
   a typed `UnsupportedModel`-class error where not. Never a silent drop.
5. The one-hot grow path runs on the **GPU device path** via comptime `one_hot` arms
   on the split *fold* and split *test*; the histogram fill is unchanged.
6. Oracle gate: ≤1e-5 vs upstream catboost 1.2.10 through **production** code.
7. Speed gate: Colab T4, matched config, faster than official CatBoost GPU.

### Non-goals
- **The materialize-as-synthetic-binary-float-columns shortcut is rejected** as the
  model representation. It needs zero kernel work and would meet the speed goal
  trivially, but the trained model then contains `ModelSplit::Float` splits on
  synthetic features, cannot be serialized as an upstream-readable one-hot `.cbm`, and
  pollutes the float-feature index space `[VERIFIED: LOCAL research.md §5.5]`. This is
  a locked decision, recorded so implementation cannot drift into it.
- ORD-06/ORD-07 combination-CTR split-search correctness
  (`.planning/phases/24-ctr-split-search-correctness/`) — separate tracked defect,
  distinct from the §1.1 ordering defect.
- One-hot on non-symmetric / Region / pairwise grow policies.
- ONNX/CoreML *implementation* of one-hot branches (guards only).
- `region_leaf`'s ignored `one_hot` flag (`apply.rs:283-303`).
- **Device-side CTR** — see SPEC-OH-26 and §9 R12.

## 3. Dependencies

Reused unchanged: `grow_one_hot_tree` (as a frozen correctness reference only),
`AnySplit` / `OneHotSplit` (`tree.rs:143`, `:131`), `route_categorical` /
`EncodingPath` (`candidates.rs:92`), `learn_set_cardinality`,
`distinct_bins_ascending` (`tree.rs:3037`), `cb_data::calc_cat_feature_hash`,
`cb_data::PerfectHash`, `build_combined_bins` (`cbm.rs:375-407`),
`build_bin_features` (`cbm.rs:85-93`), `build_ctr_features` (`cbm.rs:161-230`, the
pruning precedent). `cubecl = 0.10.0` `[VERIFIED: LOCAL Cargo.toml:38]`.

FlatBuffers bindings already generated
`[VERIFIED: LOCAL crates/cb-model/src/generated/model_generated.rs:1830-1902]`:
```
table TOneHotFeature { Index: i32 = -1; Values: [i32]; StringValues: [string]; }
TModelTrees.CatFeatures -> VT = 12;  TModelTrees.OneHotFeatures -> VT = 16
```

## 4. Typed contracts

```rust
// crates/cb-model/src/model.rs
pub enum ModelSplit {
    Float(Split),
    Ctr(CtrSplit),
    /// A one-hot `calc_cat_feature_hash(raw) == value_hash` equality test.
    /// `value_hash` is the UPSTREAM raw i32 hash space (§1.2), NOT a PerfectHash bin.
    OneHot(OneHotModelSplit),
}
pub struct OneHotModelSplit { pub cat_feature: usize, pub value_hash: i32 }

// crates/cb-train/src/boosting.rs — ObliviousTree gains ordered kinds
pub struct ObliviousTree {
    pub splits: Vec<Split>,
    pub ctr_splits: Vec<CtrSplitSpec>,
    pub one_hot_splits: Vec<OneHotSplit>,
    /// Per-level kind in TRUE level order; `splits`/`ctr_splits`/`one_hot_splits`
    /// are consumed in order as `level_kinds` is walked. len() == depth.
    pub level_kinds: Vec<LevelKind>,
}

// crates/cb-compute/src/runtime.rs — device seam carries the kind
pub struct DeviceGrownTree { pub splits: Vec<(u32, u32, bool /* one_hot */)>, /* … */ }
```

## 5. Failure-isolated behavioral specifications

### SPEC-OH-01 — `ObliviousTree` carries ordered mixed-kind levels
**Input:** a grown tree. **Output:** `ObliviousTree` with `level_kinds.len() == depth`.
**Given** a tree whose levels interleave kinds, **when** it is persisted, **then**
`level_kinds` records the true per-level kind in level order.
**Out of scope:** how `from_trained` consumes it (SPEC-OH-02).

### SPEC-OH-02 — `from_trained` emits splits in level order
**Given** a trained tree with `level_kinds = [Ctr, Float]`, **when**
`Model::from_trained` runs, **then** `model.splits == [Ctr(..), Float(..)]` — level
order, not kind-grouped order.
**Invariant:** for an all-float tree the emitted order is byte-identical to today.

### SPEC-OH-03 — the order fix is pinned by a CTR-at-level-0 regression test
**Given** a synthetic trained tree with a CTR split at level 0 and a float split at
level 1, **when** applied, **then** the predicted leaf matches the trainer's `leaf_of`
assignment for the same object.
**Why isolated:** this is the only test that would have caught §1.1; it must fail
against today's `from_trained` and pass after SPEC-OH-02.

### SPEC-OH-04 — one-hot-routed columns are identified in `train_inner`
**Given** categorical columns and `params.one_hot_max_size`, **when** `train_inner`
partitions them, **then** columns with `route_categorical(card, k) ==
EncodingPath::OneHot` are collected as an absolute-cat-index list, disjoint from the
CTR-eligible list.
**Invariant:** the existing `eligible_absolute` CTR list is unchanged.

### SPEC-OH-05 — one-hot columns reach the candidate feature matrix
**Given** the SPEC-OH-04 list, **when** the grow site builds its candidate set,
**then** each one-hot column contributes its `PerfectHash` bin column and its distinct
bins as equality candidates.

### SPEC-OH-06 — fused one-hot-aware level search
**Input:** float + one-hot candidates, der/weight, `scaled_l2`, `score_function`.
**Output:** one `AnySplit` per level.
**Given** a level, **when** the search runs, **then** it selects the best-scoring
candidate across both kinds using **histogram-fused** scoring — per-bin sums derived
once per feature, `left = bin_sums[value]`, `right = total - left` for one-hot — with
the same argmax and lowest-index tie-break as the float path.
**Explicit constraint:** `score_candidate_any` (`tree.rs:3052-3068`) is O(levels ×
candidates × n) with no histogram reuse and MUST NOT be used on the production path
(§9 R4). `grow_one_hot_tree` is retained only as a frozen correctness reference.

### SPEC-OH-07 — one-hot splits are persisted on the trained tree
**Given** a level that selected a one-hot split, **then** it appears in
`one_hot_splits` and its level records `LevelKind::OneHot`.

### SPEC-OH-08 — `ModelSplit::OneHot` exists with the upstream value space
**Given** the new variant, **then** `float_feature()` and `as_float()` both return
`None`, and `value_hash` is typed `i32` (the raw hash space).

### SPEC-OH-09 — bin → raw-hash mapping at the lift (the §1.2 landmine)
**Given** a trainer `OneHotSplit { feature, value }` where `value` is a `PerfectHash`
bin, **when** lifted, **then** the emitted `value_hash` is the **original
`calc_cat_feature_hash` i32** for that bin, recovered via the perfect-hash inverse —
never the bin ordinal.
**Acceptance:** the emitted `value_hash` for a known raw category equals
`calc_cat_feature_hash(raw)` computed independently in the test.
**Why isolated:** an in-repo-only oracle cannot detect this; the test must assert
against an independently computed hash.

### SPEC-OH-10 — `passes_split` applies a one-hot split
**Given** an object whose raw categorical value is `v`, **then** the split passes iff
`calc_cat_feature_hash(v) == value_hash`.

### SPEC-OH-11 — `.cbm` save emits `OneHotFeatures` with upstream offsets and pruning
**Given** a model with one-hot splits, **when** saved, **then** a `OneHotFeatures`
section is emitted containing **only the distinct `(cat_feature, value_hash)` pairs
the trees reference** (upstream prunes to used values), and the combined bin index
space is `Float → OneHot → Ctr`.
**Acceptance (empirically pinned, research.md §4.2):** for 1 float feature with 2
borders, cat 0 with 1 used value, cat 1 with 2 used values → global bins float `0,1`,
cat 0 `2`, cat 1 `3,4`.

### SPEC-OH-12 — `.cbm` load accepts one-hot splits
**Given** a `.cbm` containing one-hot splits, **when** loaded, **then** they decode to
`ModelSplit::OneHot`, replacing today's typed `"one-hot split unsupported (v1)"`
rejection (`cbm.rs:1041-1045`).
**Supersedes:** decision CTR-05 in `.planning/phases/23-ctr-model-loading/cbm-ctr-load/SPEC.md`;
`cb-model/src/cbm_test.rs::one_hot_split_index_is_typed_error` must be re-pointed, not
deleted silently.

### SPEC-OH-13 — an upstream-produced one-hot `.cbm` predicts within 1e-5
**Given** the committed upstream catboost 1.2.10 one-hot fixture, **when** loaded and
applied through production code, **then** predictions match the upstream reference
within 1e-5.
**Why isolated:** this is the ONLY specification that proves our encoding is
genuinely upstream-compatible rather than merely self-consistent (§1.2).

### SPEC-OH-14 — `.json` never silently drops a split
**Given** a model with one-hot splits, **when** `save_json` runs, **then** it either
emits the upstream `{cat_feature_index, value, split_type:"OneHotFeature"}` shape or
returns a typed error — never a shortened split list (`json.rs:474-485`).
**And** `decode_json` on an upstream one-hot json returns a typed "one-hot json
unsupported" error rather than today's misleading `missing field 'border'`.

### SPEC-OH-15 — SHAP never silently drops a one-hot split
**Given** a model with one-hot splits, **when** **any** consumer of `float_splits_of`
(`shap.rs:550-552`) runs, **then** it returns a typed unsupported error (or a correct
arm) — never a silently shortened `tree_depth`.

**The consumer set is four, not one (amendment, 2026-07-31):** `shap_values`
(`shap.rs:534`), `shap_interaction_values` (`:941`), `prediction_diff` (`:1137`), and
`sage_values` (`:1236`) — all `pub` and re-exported at `cb-model/src/lib.rs:58`. The
original wording named only `shap_values`, leaving `prediction_diff` and `sage_values`
silently wrong.

**Consequence — this is a deliberate signature change, not a local arm:** all four,
plus `fstr::loss_function_change` (`fstr.rs:788`), are **infallible today**. Satisfying
this specification requires `float_splits_of` itself to become fallible
(`Result<_, ShapUnsupported>`) so the compiler enforces every site, cascading through
four call sites: `fstr.rs:804`, `fstr.rs:846`, `catboost-rs/src/model.rs:292`, and
`catboost-rs/src/model.rs:393`. A reduced "boundary guard" variant that leaves the
inner APIs infallible is explicitly NOT acceptable — with only four call sites the
cascade is small and the compiler-enforced guarantee is the point.

### SPEC-OH-16 — ONNX and CoreML guards reject one-hot models
**Given** a model with one-hot splits, **when** export is attempted, **then** a typed
`OneHotSplitsUnsupported`-class error is returned.
**Why isolated:** today both guards test `matches!(.., Ctr(_))` and would pass a
one-hot model through to emit `(feature = 0, border = 0.0)` silently.

### SPEC-OH-17 — GPU-apply guard names one-hot explicitly
**Given** a one-hot model, **then** `gpu_apply` returns an explicit
`OneHotSplits` rejection rather than the confusing downstream match error.

### SPEC-OH-18 — fstr treats one-hot splits as cat-feature identities
**Given** two one-hot splits on the SAME cat feature with different values, **then**
`same_internal_feature` reports them as the same internal feature (upstream `TFeature`
identity ignores the value, exactly as it ignores a float border), and
`split_flat_indices` returns the flat cat index.
**Why isolated:** today's `_ => false` wildcard (`fstr.rs:462-473`) silently gets this
backwards, corrupting interaction/PVC.

### SPEC-OH-19 — partial dependence rejects one-hot models
**Given** a one-hot model, **then** `partial_dependence` returns a typed error rather
than operating on the float-only column space.

### SPEC-OH-20 — a cat-only pool is device-eligible
**Given** a pool with 0 float features and ≥1 one-hot column, **when** eligibility is
evaluated, **then** the `matrix.n_features() > 0` precondition
(`boosting.rs:3055-3122`, item 11) no longer forces a CPU fallback.
**Invariant:** every other precondition is unchanged.

### SPEC-OH-21 — device quantization emits one-hot bin columns
**Given** one-hot columns, **when** `quantize_feature_major` runs, **then** each
contributes a bin column with `cardinality` buckets in the shared uniform `n_bins`
line, and the packed `TCFeature.one_hot_feature` is set **truthfully** (it is
hard-coded `false` at `cindex.rs:226` today).

### SPEC-OH-22 — the split-scoring fold has a one-hot arm
**Given** `#[comptime] one_hot == true`, **then** the fold becomes
`left = bin_sums[value]`, `right = total - left` (instead of the float prefix fold),
preserving candidate indexing, argmax, and lowest-index tie-break, in **BOTH**
scorers:
- `find_optimal_split_partition_kernel` (`crates/cb-backend/src/kernels.rs:4506`) —
  the **production** resident path, reached via `score_partition_over_binsums`
  (`gpu_runtime/mod.rs:2961`) ← `grow_oblivious_tree_resident` (`mod.rs:3803`);
- `find_optimal_split_kernel` (`kernels.rs:3367`) — the non-resident slice entry.

**Why both (amendment, 2026-07-31):** the original wording named only
`find_optimal_split_kernel`. Patching only that one would leave the production path
unchanged while a parity test on the oracle path passed — a false green.

**Constraint — fold over REAL bins only:** the fold must be bounded by the feature's
**true cardinality**, not by the shared uniform `n_bins` line, so padded bins cannot
win a split.
**This bound must NOT be taken from `TCFeature.folds`.** On the production path
`session.rs:1363` packs with `vec![n_bins_line; eff_n_features]` — a uniform PADDED
width — and `cindex.rs:217` copies that straight into `TCFeature.folds`, so
`folds[feature] == n_bins_line` always and `border < folds[feature]` degenerates into
the loop bound itself, bounding nothing. A **separate real-cardinality array** must be
plumbed from quantization through the device config to the scorer. Note that the
obvious alternative — changing `n_buckets_per_feature` — is forbidden: `pack_cindex`
derives `bits = feature_bits(nb)` from it (`cindex.rs:181-200`), so it would change the
packed words for every pool and break the float-only byte-identity gate (SPEC-OH-31).
**Constraint — trailing-border exclusion must be lifted in TWO places:** the kernel
(`kernels.rs:4596-4604`) computes `max_border = n_bins_used - 1` and excludes it, and
a host belt repeats the exclusion at `gpu_runtime/mod.rs:3108-3112`
(`if (cand as usize) % n_bins >= n_bins_used - 1 { continue; }`). A one-hot equality
candidate on the last bin is legitimate, so lifting only one leaves the
highest-cardinality category unable to ever win.
**Invariant:** with `one_hot == false` the kernel's OUTPUT is **numerically identical**
to today, and the runtime `n_features` — which also fixes `leaf_stride` — is unchanged
on every launch. (Numerical, not byte, identity: adding comptime parameters changes the
generated kernel source by construction, so byte-identity of the kernel itself is not a
testable property; identity of the produced scores and chosen splits is.)
**Constraint:** the histogram **fill** is unchanged — only the fold differs.

### SPEC-OH-23 — the split-application test has a one-hot arm
**Given** `#[comptime] one_hot == true`, **then** `partition_split_kernel` tests
`read_bin(..) == value` instead of `> bin`.
**Invariant:** with `one_hot == false` the kernel's OUTPUT is **numerically identical**
to today — the same `new_leaf_of` routing for every object. (Numerical, not byte,
identity: adding the `#[comptime] one_hot: bool` parameter changes the generated kernel
source by construction, so byte-identity of the kernel itself is not a testable
property; identity of the produced document routing is.)

### SPEC-OH-24 — the device seam carries the split kind
**Given** a device-grown tree containing a one-hot split, **then**
`DeviceGrownTree.splits` conveys the kind (today `Vec<(u32, u32)>` carries none;
`region_path`'s 4-tuple is the in-repo precedent).

### SPEC-OH-25 — device one-hot training matches the CPU grower
**Given** an identical one-hot configuration, **then** the device-grown model matches
the CPU-grown model within 1e-5.
**Anti-false-pass guards (mandatory, both):** a `CountingGpu`-style wrapper proving
`grow_tree_on_device` returned a tree for **every** iteration — a silent CPU fallback
makes "device == CPU" trivially true — and an assertion that the trained model
actually contains ≥1 `ModelSplit::OneHot`.

### SPEC-OH-26 — one-hot × CTR in one pool is explicitly gated, not silently wrong
**Given** a pool with both one-hot-routed and CTR-routed columns, **then** training
either succeeds correctly or returns a typed error naming the unsupported
combination — never silently drops either kind.
**Note (scope):** the user's locked decision was to lift both device-eligibility
blockers. The float-count blocker is SPEC-OH-20. The CTR blocker
(`materialized_ctr_features.is_empty()` plus `DeviceTrainConfig::is_covered_regime()`
requiring `ctr.is_none()`) cannot be lifted without **device-side CTR support**, which
is a large adjacent feature, not a gate tweak. This specification therefore requires
only the honest gate; device CTR co-existence is deferred to the chained CTR plan and
recorded as §9 R12 for an explicit user decision.

### SPEC-OH-27 — RNG draw-order accounting is preserved
**Given** a configuration where `draws_active` (bootstrap / `random_strength`),
**when** the one-hot level search runs, **then** it consumes the per-level `randSeed`
and per-candidate `std_normal` draws exactly as
`greedy_tensor_search_oblivious_perturbed` does (`boosting.rs:3272-3277`).
**Why isolated:** one-hot changes the **candidate count**, hence the draw count; a
mismatch silently shifts every later tree's bootstrap sample. This is the same defect
class as the fixed `d7676b5` MVS bug.
**Blocking:** requires an instrumented upstream 1.2.10 run to establish ground truth
(§9 R5). If it cannot be established, one-hot × bootstrap must be typed-rejected
rather than guessed.

### SPEC-OH-28 — the oracle runs through production code
**Given** the committed one-hot fixture, **when** the oracle test runs, **then** it
drives production `train_cat` and matches upstream within 1e-5.
**Constraint:** the existing test-local driver in `one_hot_oracle_test.rs` is deleted,
not left alongside.
**Constraint:** the comparison uses the model's own `float_feature_borders()`, and
pins every knob whose builder default differs from catboost's raw dict-API default
(`random_strength = 0` is the known trap).

### SPEC-OH-29 — the fixture is frozen and generated in isolation
**Given** fixture generation, **then** `gen_fixtures.py` gains a dedicated
`--one-hot-only` flag placed **before** the `else: main()` fallthrough, mirroring
`--bootstrap-dev-only`, and the fixture is generated once with a pinned seed and
`thread_count=1` and committed.
**Mandatory guard:** `git status --short crates/cb-oracle/fixtures` must show ONLY the
intended new files; the task aborts otherwise.
**Rationale:** `gen_fixtures.py` has NO positional-scenario dispatch — an unrecognised
argv falls through to `main()` and regenerates the ENTIRE committed corpus, which is
run-to-run nondeterministic.

### SPEC-OH-30 — the Colab T4 speed gate
**Given** a matched config on a Colab T4, **when** catboost-rs and official CatBoost
GPU are both trained, **then** catboost-rs is faster.
**Constraint:** the runner must adopt the `CB_GPU_PROF` device-activation probe from
`bench/bootstrap_gpu/bootstrap_bench_colab.py` (greps for `CB_GPU_PROF tree` lines per
arm), which closes the "device activation not observable" caveat that
`quick_gpu_speed/bench.py` still carries.
**Baseline:** the numeric path is currently 1.05–1.19× faster than official CatBoost
GPU; one-hot must not inflate host prep (fit-prep ~226ms / quantize ~70ms / begin
~203ms already dominate ~150ms of device tree time).

### SPEC-OH-31 — the float-only path is unchanged (D-04)
**Given** a pool with no categorical columns, **then** the trained model is
byte-identical to today.
**Acceptance:** every existing float-only oracle suite passes **unmodified**.

## 6. Acceptance scenarios

| # | Scenario | Expected | Specs |
|---|---|---|---|
| A1 | Train on a binary categorical column with default `one_hot_max_size=2` | The column influences the model; ≥1 `ModelSplit::OneHot` | OH-04..07 |
| A2 | Train float-only | Byte-identical to today | OH-31 |
| A3 | Tree with CTR at level 0, float at level 1 | Applied leaf matches trainer `leaf_of` | OH-01..03 |
| A4 | Save one-hot model → `.cbm` | `OneHotFeatures` present, pruned, offsets `Float→OneHot→Ctr` | OH-11 |
| A5 | Load upstream 1.2.10 one-hot `.cbm` → predict | ≤1e-5 vs upstream reference | OH-12, OH-13 |
| A6 | Emitted `value_hash` for a known category | Equals independently computed `calc_cat_feature_hash` | OH-09 |
| A7 | ONNX / CoreML / SHAP / PDP / gpu_apply on a one-hot model | Typed error, never silent wrong numbers | OH-14..19 |
| A8 | Two one-hot splits on the same cat feature | Same internal feature in fstr | OH-18 |
| A9 | Cat-only pool | Device path engages, not CPU fallback | OH-20 |
| A10 | Device vs CPU one-hot training | ≤1e-5, with both anti-false-pass guards | OH-25 |
| A11 | Production one-hot oracle | ≤1e-5 vs upstream | OH-28, OH-29 |
| A12 | Colab T4 matched benchmark | catboost-rs faster than official CatBoost GPU | OH-30 |
| A13 | Mixed one-hot + CTR pool | Correct, or typed error — never a silent drop | OH-26 |

## 7. Impact scope

**Classification: cross-module** — `cb-train`, `cb-model`, `cb-compute`, `cb-backend`,
`cb-oracle`, plus the bench harness.

Full file/symbol table: research.md §9. Highest-traffic sites:
`cb-train/src/boosting.rs` (`ObliviousTree` at `:775-794`, partitioning at
`:2721-2749`, grower dispatch at `:4042-4200`, persist at `:4786-4824`, eligibility at
`:3055-3122`, quantize at `:2196-2233`), `cb-train/src/tree.rs` (`GrownTree`/`LevelKind`
at `:205-273`, level search at `:2919-3009`), `cb-model/src/{model,apply,cbm,json,shap,fstr,gpu_apply}.rs`,
`cb-model/src/export/{onnx,coreml}.rs`, `cb-backend/src/kernels.rs` (`:3367`, `:3731-3773`),
`cb-backend/src/gpu_runtime/{cindex,mod,session}.rs`, `cb-compute/src/runtime.rs`.

**Supersedes** decision CTR-05 in `.planning/phases/23-ctr-model-loading/cbm-ctr-load/SPEC.md`
and the save-side "v1 has no one-hot" offset assumption in `cbm-ctr-save/PLAN.md:36,46`.
**Ordering dependency:** `.planning/plans/catboost-builder-cat-features-routing/`
(in-flight, untracked) adds the `.one_hot_max_size(k)` builder setter that makes this
parameter reachable at all.

## 8. Compatibility and migration

- `ModelSplit` gains a third variant — a **breaking change for exhaustive external
  matchers**, deliberate and compiler-enforced for in-repo consumers.
- `.cbm` written by this change contains a `OneHotFeatures` section; models without
  one-hot splits are byte-identical to today.
- `.cbm` previously rejected with `"one-hot split unsupported (v1)"` now load — a
  widening, not a break.
- The §1.1 order fix changes emitted split order **only** for trees that mix kinds;
  all-float trees are unchanged (SPEC-OH-02 invariant).
- No CubeCL/runtime version change (`cubecl = 0.10.0`).

## 9. Risks and open questions

| # | Risk | Mitigation | Spec |
|---|---|---|---|
| R1 | Copying the CTR lift reproduces the order defect for one-hot | Ordered `level_kinds` end to end + a CTR-at-level-0 regression test | OH-01..03 |
| R2 | Value-space mismatch (PerfectHash bin vs raw i32 hash) — self-consistent but upstream-wrong | Assert against an independently computed hash; load a real upstream `.cbm` | OH-09, OH-13 |
| R3 | Six consumers accept a new variant with NO compiler error | Each gets an explicit spec and test | OH-14..19 |
| R4 | `grow_one_hot_tree` is O(levels × candidates × n) — cannot meet the speed goal | Fused search on the production path; keep it as a frozen reference | OH-06 |
| R5 | One-hot changes the per-level RNG **candidate/draw count**, shifting every later tree's bootstrap sample | Establish upstream ground truth by instrumented run; else typed-reject one-hot × bootstrap | OH-27 |
| R6 | Silent CPU fallback makes device==CPU trivially true | `CountingGpu` guard + assert ≥1 one-hot split | OH-25 |
| R7 | `gen_fixtures.py` unrecognised argv regenerates the whole corpus | Dedicated `--one-hot-only` flag + `git status` abort guard | OH-29 |
| R8 | Fixture nondeterminism | Generate once, pin seed + `thread_count=1`, commit | OH-29 |
| R9 | Builder defaults ≠ catboost raw-dict defaults (`random_strength`) | Pin every knob on BOTH sides | OH-28 |
| R10 | Uniform device `n_bins = max_f(borders+1)`; a high-cardinality one-hot (upstream allows up to 255) blows up the per-feature bin line | Bound `one_hot_max_size` on the device arm or fall back above a threshold | OH-21 |
| R11 | `--no-default-features --features rocm` test build is broken (pre-existing) | Use `--features rocm` (additive over `default=["cpu"]`), which builds | — |
| R12 | **Lifting the CTR device blocker requires device-side CTR support** — a large adjacent feature the "lift both" decision may not have priced in | Deferred to the chained CTR plan; this plan ships only the honest gate | OH-26 |

### Open questions
1. **R5 / SPEC-OH-27** is the one genuine blocker: does upstream's one-hot candidate
   set change the per-level RNG draw count? Needs an instrumented catboost 1.2.10 run
   (the technique used for `.planning/plans/bayesian-rng-draw-accounting/`). Until
   resolved, one-hot × bootstrap must be typed-rejected rather than guessed.
2. Does SPEC-OH-14 emit the real upstream one-hot json shape, or a typed error? Cheap
   either way; the plan must pick one, not leave it ambiguous.
3. R12: confirm with the user that device CTR co-existence is deferred.

## 10. Traceability and sources

- Research: `.planning/plans/one-hot-categorical-training/research.md` (797 lines,
  CodeGraph + local + empirical upstream probe).
- Locked user decisions 2026-07-31: fix the split-order loss in scope; lift the
  device-eligibility blockers; chained order one-hot → CTR params → absent BoostParams;
  per-feature device path + T4 speed gate.
- Empirical: a real catboost 1.2.10 one-hot `.cbm` was produced and byte-scanned to
  pin `TOneHotFeature.Values` as raw i32 hashes and the `Float→OneHot→Ctr` bin space
  (research.md §4.1-4.4). Our `load_cbm` returns exactly `one-hot split unsupported
  (v1)` on it, proving our offset math already matches.
- Local verification 2026-07-31: `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test`
  → 3 passed, confirming the §1.1 defect is latent rather than active.
- TreeFinder: no indexed document covers one-hot (index holds 3 unrelated documents).
  This SPEC is a **pending TreeFinder update**, staged locally.

## 11. Closing verification (release gate, not a plan task)

Full oracle suite on a CUDA GPU runner (Colab T4 via `~/.local/bin/colab`; Kaggle
`boomvector` as fallback), plus the SPEC-OH-30 speed comparison.
