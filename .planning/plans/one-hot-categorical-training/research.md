---
title: One-hot categorical training in cb-train (CPU + GPU device path)
kind: research
status: draft
updated_at: 2026-07-31
evidence_tags: "[VERIFIED: CODEGRAPH …] [VERIFIED: LOCAL <path:line>] [VERIFIED: CMD <command>] [VERIFIED: TREE_FINDER …] [INFERRED] [UNVERIFIED]"
scope_note: "Evidence gathering only. No production code changed. No plan tasks authored."
---

# Research — one-hot categorical training in the `cb-train` engine

## 0. TL;DR for the planner

1. **The gap claim is CORRECT.** `grow_one_hot_tree` has no production caller; a
   one-hot-routed categorical column is silently dropped from training.
2. **Two of your side-claims need correcting** (§1.4): the `rocm` build claim is
   only true for `--no-default-features --features rocm`, and the one-hot device
   plumbing is *further along* than stated (`TCFeature.one_hot_feature` and a
   `#[comptime] one_hot` pairwise-hist arm already exist).
3. **Split ORDER across mixed kinds IS significant, and the existing CTR
   precedent is already order-LOSSY** — `Model::from_trained` appends CTR splits
   *after* all float splits, discarding `GrownTree.level_kinds`. This is a
   pre-existing latent correctness bug that a one-hot phase **must not copy**
   (§3). Highest-risk item in this document.
4. **The upstream `.cbm` one-hot encoding is now empirically pinned** — I produced
   a real upstream 1.2.10 one-hot model and confirmed the combined bin-index
   space `Float → OneHot → Ctr` byte-for-byte through our own decoder (§4).
5. **`grow_one_hot_tree` is an O(candidates × n) reference implementation, not a
   performance path** — it cannot be the basis of a "beat CatBoost GPU" claim
   (§5.1). The performance story has to be the device path or a fused CPU
   histogram path.
6. **Two silently-wrong export sites** will accept a new `ModelSplit` variant
   without a compiler error and emit `(feature=0, border=0.0)` (§3.3).

---

## 1. Verification of the stated gap

### 1.1 `grow_one_hot_tree` has no production caller — CONFIRMED

`[VERIFIED: CMD rg -n "grow_one_hot_tree" --type rust -g '!target']` — exactly
five reference sites:

| site | kind |
|---|---|
| `crates/cb-train/src/tree.rs:3141` | the definition |
| `crates/cb-train/src/lib.rs:107` | `pub use` re-export |
| `crates/cb-train/src/tree_test.rs:67,83,118,130` | unit tests |
| `crates/cb-train/tests/one_hot_oracle_test.rs:54,139` | oracle test |
| `crates/cb-train/src/tree.rs:3140` (doc link) | doc reference |

`[VERIFIED: CODEGRAPH grow_one_hot_tree]` blast radius: "3 callers in
`crates/cb-train/src/lib.rs`" — i.e. only the re-export line, plus tests.
`train_inner` (`crates/cb-train/src/boosting.rs:2263`) never mentions it; its
grower dispatch (`boosting.rs:4042-4200`) has exactly four arms — `Region`,
`Lossguide|Depthwise`, `SymmetricTree`→{pairwise, CTR-aware, ordered, plain}
`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:4042-4200]`.

### 1.2 `one_hot_max_size` is read only to EXCLUDE from CTR — CONFIRMED

`[VERIFIED: CMD rg -n "one_hot" crates/cb-train/src/boosting.rs]` — inside
`train_inner` the parameter appears at exactly two sites:

- `boosting.rs:2709-2713` → `tensor_ctr_candidates(&cat_cardinalities, params.one_hot_max_size, params.max_ctr_complexity)`
- `boosting.rs:2721-2729` → `eligible_absolute` filter keeping only
  `route_categorical(card, one_hot_max_size) == EncodingPath::Ctr`

`[VERIFIED: LOCAL crates/cb-train/src/candidates.rs:92-104]` `route_categorical`
returns `OneHot` for `1 < card <= one_hot_max_size`. Nothing downstream consumes
`EncodingPath::OneHot`. `[VERIFIED: CMD rg -n "EncodingPath::OneHot"]` → only
`candidates.rs`, `candidates_test.rs`, `one_hot_oracle_test.rs:178`.

**The silent-drop is real:** a `train_cat` caller passing a cardinality-2
categorical column with the default `one_hot_max_size = 2`
`[VERIFIED: LOCAL crates/cb-train/src/candidates.rs:78-80 one_hot_max_size_default() == 2]`
gets a column that enters no CTR projection and no one-hot split; it contributes
nothing to the model and no error is raised.

Cross-check against upstream: with catboost 1.2.10 defaults, `one_hot_max_size`
is `2`, and a cardinality-3 categorical routes to `OnlineCtr`, not one-hot
`[VERIFIED: CMD .venv/bin/python — get_all_params()['one_hot_max_size'] == 2;
model.json split_type == "OnlineCtr"]`. So the repo's routing threshold matches
upstream exactly; only the one-hot *consumer* is missing.

### 1.3 The oracle test builds its own driver — CONFIRMED

`crates/cb-train/tests/one_hot_oracle_test.rs:90-169` defines
`train_one_hot_only(...)`, a test-local boosting driver calling
`grow_one_hot_tree` directly. It self-oracles against `cb_train::train` on
one-hot-**encoded binary float columns** — *not* against upstream
`[VERIFIED: LOCAL crates/cb-train/tests/one_hot_oracle_test.rs:15-37, 188-276]`.
It consumes **no fixture at all**; `crates/cb-oracle/fixtures/one_hot_cat/` is
mentioned only in the "why not" doc comment and contains only `.npy` anchors +
`config.json` — **no `.cbm`, no `model.json`**
`[VERIFIED: CMD ls crates/cb-oracle/fixtures/one_hot_cat/]`.

### 1.4 CORRECTIONS to the request's premises

| your claim | reality | evidence |
|---|---|---|
| "`cargo test -p cb-train --features rocm` does NOT build" | **`cargo test -p cb-train --features rocm --no-run` BUILDS FINE** (rocm is additive on top of `default = ["cpu"]`). The failure is under **`--no-default-features --features rocm`**: 10× `error[E0432]: unresolved import cb_backend::CpuBackend` | `[VERIFIED: CMD cargo test -p cb-train --features rocm --no-run]` → all test executables emitted; `[VERIFIED: CMD cargo test -p cb-train --no-default-features --features rocm --no-run]` → E0432 ×10 |
| "does the device split representation need a new split-kind field?" | Partly already there: `TCFeature.one_hot_feature: bool` exists as a **frozen descriptor contract**, currently always `false` and `#[allow(dead_code)]` | `[VERIFIED: LOCAL crates/cb-backend/src/gpu_runtime/cindex.rs:46-75, 226]` |
| "`ObliviousTree` doc claims one-hot paths produce ONLY float splits" | correct quote, and note `RegionTree`/`RegionLevel`/`DeviceGrownTree.region_path` **already carry a `one_hot: bool`** per level (unused, always `false`) | `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:840-843; crates/cb-model/src/model.rs:151-154; crates/cb-compute/src/runtime.rs:975-986]` |
| CLAUDE.md path `…/cubecl_error_guideline.md` | the actual artifact is a **directory** `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`; `INDEX.md` does exist | `[VERIFIED: CMD ls]` |
| CLAUDE.md refers to "AGENTS.md" rules | **no `AGENTS.md` exists in the repo** (maxdepth 3); the rules live inlined in `CLAUDE.md` | `[VERIFIED: CMD find … -name AGENTS.md]` |

---

## 2. Trainer wiring — where the one-hot path would hook in

### 2.1 What `train_inner` has at the grow site

`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:2678, 4042-4200]`

- `let matrix = FeatureMatrix::new(feature_values, feature_borders);` at
  `boosting.rs:2678` — **`FeatureMatrix::new` hardcodes `cat_bins: &[]`**
  `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:341-350]`. The struct already has
  a public `cat_bins: &'a [Vec<u32>]` field (`tree.rs:337`), so a struct-literal
  construction (exactly what `one_hot_oracle_test.rs:134-138` does) is the seam.
- `cat_columns: &[Vec<String>]` is in scope (`train_inner` param).
- `cb_data::perfect_hash_bins(&as_str)` is **already called** in `train_inner` to
  build `cat_eligible_buckets` for the CTR-eligible subset
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:2740-2749]` — the same
  primitive would build the one-hot bin columns for the `EncodingPath::OneHot`
  subset. **Do not hand-roll a second hashing loop.**
- Per-iteration the plain arm calls
  `greedy_tensor_search_oblivious_perturbed(&matrix, &score_weighted_der1,
  &score_weights, scaled_l2, params.depth, n, perturb, params.score_function,
  pen.as_ref())` `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:4185-4195]`.

### 2.2 Signature fit — an adapter is NOT enough

`grow_one_hot_tree(matrix, der1, weight, scaled_l2, depth, n_objects,
score_function) -> CbResult<GrownOneHotTree>`
`[VERIFIED: LOCAL crates/cb-train/src/tree.rs:3141-3163]`.

Missing relative to what the plain arm supplies and the boosting loop consumes:

| missing | consequence |
|---|---|
| `perturb: Option<Perturbation>` | `random_strength != 0` **and** any active bootstrap force `perturb = Some(..)` (`boosting.rs:4000-4018`); without it the RNG draw stream desyncs → every later tree's sample is wrong (`WR01-S7`) |
| `penalties: Option<&FeaturePenalties>` | FEAT-04 feature weights/penalties silently ignored |
| `min_data_in_leaf` | not honoured (plain oblivious arm also doesn't — parity by omission) |
| returns `GrownOneHotTree`, not `GrownTree` | the boosting loop consumes `grown.splits/.leaf_of/.ctr_splits/.level_kinds/.step_nodes/.node_id_to_leaf_id/.region_directions/.region_one_hot` (`boosting.rs:4207-4219, 4786-4824`) |

**Recommended shape (matches the CTR precedent exactly):** extend `GrownTree`
with `one_hot_splits: Vec<OneHotSplit>` + a third `LevelKind::OneHot { idx }`
variant, and grow through a `*_with_one_hot` sibling of
`greedy_tensor_search_oblivious_with_ctr` (`tree.rs:2919-3009`), which already
demonstrates the split-back-into-parallel-vectors + `level_kinds` pattern.
Keep `grow_one_hot_tree` as the frozen ≤1e-5 reference for a self-oracle.

### 2.3 Which `train_inner` features compose with one-hot

Derived from upstream behaviour + the repo's own gates:

| feature | composes? | why |
|---|---|---|
| Plain boosting | **yes** | the ORD-04 slice's own premise |
| bootstrap / sampling | **yes, must** | `score_weighted_der1` / `score_weights` are already the sampled channels; a one-hot grower must consume the same buffers, and must consume the per-level RNG draws in the same order |
| `random_strength` perturbation | **yes, must** | same draw-order argument |
| Ordered boosting | **defer** | `greedy_tensor_search_oblivious_ordered` is a separate per-segment scorer with no cat awareness; upstream *does* combine them, but this is a second scorer to extend |
| CTR (`has_ctr`) | **must compose eventually** | upstream mixes one-hot and CTR splits in one tree freely (my probe shows `OneHotFeature` and `FloatFeature` mixed; with higher-cardinality cats CTR joins). But the CTR arm is `greedy_tensor_search_oblivious_with_ctr`; a three-way candidate union is a bigger change. **Suggest: phase 1 = float + one-hot only, and make the (one-hot × CTR) combination a typed rejection, not a silent drop.** |
| non-symmetric grow policies (`Lossguide`/`Depthwise`) | **defer** | `leaf_wise_grower` emits bare `Split`; `NonSymmetricTree.splits: Vec<Split>` has no kind field; `cbm.rs:592-598` already rejects non-symmetric CTR save |
| `Region` grow policy | **defer** | `region_one_hot` flag exists but `region_leaf` (`apply.rs:283-303`) *ignores* it — it always uses `passes_split`'s `>` test. Wiring one-hot here would first require fixing that |
| pairwise scoring (`*Pairwise` losses) | **out of scope** | separate scorer; upstream has no `pairwise_hist_*_one_hot.cu` for the binary/half-byte families (`kernels.rs:1312-1313, 1428`) |
| eval sets | **needs work** | `tree_eval_contribution` (`boosting.rs:1920-1948`) walks `tree.splits` as floats only; `eval_matrices` are `FeatureMatrix::new(...)` (no cat) |
| multiclass / multilabel | **defer** | orthogonal |

### 2.4 The lift into `cb_train::ObliviousTree`

`boosting.rs:4809-4815` pushes `ObliviousTree { splits: grown.splits, ctr_splits,
leaf_values, leaf_weights }` — **`grown.level_kinds` is dropped on the floor.**
This is the crux of §3.

---

## 3. Split ORDER across mixed kinds — the highest-risk finding

### 3.1 Order IS significant

- Trainer: `assign_leaves_any` / `assign_leaf_of_averaging` walk the **level**
  list in order and call `leaf_index(&passes)` = forward bit order, split `i` →
  bit `i` `[VERIFIED: LOCAL crates/cb-train/src/tree.rs:290-300, 414-423;
  crates/cb-train/src/boosting.rs:1784-1825]`.
- Apply: `leaf_index_for` walks `tree.splits` **in stored order** and calls the
  same `leaf_index` `[VERIFIED: LOCAL crates/cb-model/src/apply.rs:208-215]`.
- `.cbm` save/load preserve `tree.splits` order 1:1 into/out of `TreeSplits`
  `[VERIFIED: LOCAL crates/cb-model/src/cbm.rs:646-668 (save), :1008-1067 (load)]`.

So the stored `Vec<ModelSplit>` order **is** the leaf-index bit order.

### 3.2 The existing CTR lift is order-LOSSY (pre-existing latent bug)

`[VERIFIED: LOCAL crates/cb-model/src/model.rs:326-363]`:

```
let mut splits: Vec<ModelSplit> = t.splits.iter().map(|s| ModelSplit::Float(*s)).collect();
for c in &t.ctr_splits { splits.push(ModelSplit::Ctr(...)); }
```

All float splits first, then all CTR splits. Meanwhile the trainer's
`grown.level_kinds` (`tree.rs:2959-2997`) records the true interleaving and
`assign_leaf_of_averaging` uses it to compute `leaf_of` → `leaf_values`.

**Therefore:** for a tree whose level 0 is a CTR split and level 1 a float split,
`level_kinds = [Ctr, Float]` but the model stores `[Float, Ctr]`. Leaf indices 1
and 2 are transposed at apply time. This is a genuine mis-prediction, not a
cosmetic issue.

- Confidence: **HIGH** on the code reading (three independent sites read; no
  compensating permutation of `leaf_values` exists anywhere on the path).
- Confidence: **MEDIUM** that it is *reachable today* — the committed CTR
  fixtures may happen to produce float-first or CTR-only trees. **Not verified
  empirically. Flag as a planner spike** (cheap: log `level_kinds` for the
  `tensor_ctr_e2e` fixture; if any tree is non-monotone, it is a live bug).

**Planning consequence:** do **not** mirror this pattern. `ObliviousTree` (both
`cb_train` and `cb_model` sides) must carry the ordered mixed-kind list, e.g. by
threading `level_kinds` onto `cb_train::ObliviousTree` and having `from_trained`
emit `Vec<ModelSplit>` in level order. Fixing the CTR ordering is arguably an
in-scope prerequisite — but note the ORD-06/ORD-07 combination-CTR *search* bug
(`.planning/phases/24-ctr-split-search-correctness/`) is explicitly out of scope
and is a **different** defect; this one is a lift/serialization ordering defect.

### 3.3 `ModelSplit` consumers — full enumeration

`[VERIFIED: CMD rg -n "ModelSplit::" --type rust -g '!target' | grep -v _test.rs]`
plus `[VERIFIED: CODEGRAPH ModelSplit]` (22 callers across 7 files).

**A. Compiler-forced (exhaustive `match` — adding a variant is a build error):**

| site | file:line | correct arm implementable? |
|---|---|---|
| `passes_split` (apply) | `cb-model/src/apply.rs:196-201` | **YES** — `cat_bin == value`; needs the object's cat bin. Note `cat_values: &[String]` is the raw string; the model must carry the value→bin mapping or store the raw hash (see §4.4) |
| `.cbm` save split emit | `cb-model/src/cbm.rs:656-668` | **YES** — needs a `OneHotFeatures` section + offset math (§4) |
| `flatten_oblivious_f64` (GPU apply) | `cb-model/src/gpu_apply.rs:117-138` | reject; guard must be widened first (see B) |
| `ModelSplit::float_feature` | `cb-model/src/model.rs:83-88` | **YES** → `None` (a one-hot split has no *float* index) |
| `ModelSplit::as_float` | `cb-model/src/model.rs:92-97` | **YES** → `None` |
| `split_flat_indices` (fstr) | `cb-model/src/fstr.rs:441-451` | **YES** → `vec![flat_cat_index(n_float, cat_feature)]` |
| fstr `cat_feature_count` closures | `cb-model/src/fstr.rs:163-166, 174-177` | **YES** → `Some(cat_feature)` |

**B. SILENTLY ACCEPTS a new variant (no compiler error) — must be hand-audited:**

| site | file:line | today's behaviour with a `OneHot` variant | honest fix |
|---|---|---|---|
| ONNX exportability guard | `cb-model/src/export/onnx.rs:106-113` | `matches!(split, ModelSplit::Ctr(_))` → **passes the guard** | add `OneHotSplitsUnsupported` variant to `OnnxExportError` (or implement `BRANCH_EQ`) |
| ONNX node build | `cb-model/src/export/onnx.rs:205-215` | `as_float()` → `None` → emits **`(feature = 0, border = 0.0)` silently** | must be unreachable after the guard |
| CoreML exportability guard | `cb-model/src/export/coreml.rs:106-113` | same silent pass | same |
| CoreML node build | `cb-model/src/export/coreml.rs:173-182` | same silent `(0, 0.0)` | same |
| GPU-apply guard | `cb-model/src/gpu_apply.rs:50-57` | `any(matches!(.., Ctr(_)))` → passes; then the exhaustive match in `flatten_oblivious_f64` errors (safe but with a confusing message) | add an explicit `GpuApplyUnsupported::OneHotSplits` arm |
| `save_json` oblivious splits | `cb-model/src/json.rs:474-485` | `filter_map(as_float)` → **silently DROPS the split**, shortening the split list → wrong depth/leaf indexing | typed error, or emit the upstream `{cat_feature_index, value, split_type:"OneHotFeature"}` shape |
| `save_json` non-symmetric | `cb-model/src/json.rs:399` | `and_then(as_float)` → `else` branch (checked) | audit |
| `save_json` region levels | `cb-model/src/json.rs:526-535` | `.ok_or_else(...)` → typed error today | fine |
| `decode_json` split load | `cb-model/src/json.rs:648-663` | `split_type` is **never read**; every split becomes `Float` | today an upstream one-hot json fails loudly on `missing field 'border'` (see §4.5) — acceptable, but the message is misleading |
| SHAP `float_splits_of` | `cb-model/src/shap.rs:550-552` | `filter_map(as_float)` → **silently drops**, so `tree_depth` shrinks and `subtree_weights` is built for the wrong depth → wrong SHAP | typed rejection (a `ShapUnsupported`) or a real arm |
| SHAP non-sym descent | `cb-model/src/shap.rs:725-733` | `float_feature()` → `None` → **stops descending** silently | audit |
| fstr `feature_count` | `cb-model/src/fstr.rs:130-151` | `filter_map(float_feature)` → one-hot doesn't widen the float vector (probably correct: it should widen the *cat* vector) | route through `cat_feature_count` |
| fstr `same_internal_feature` | `cb-model/src/fstr.rs:462-473` | has a `_ => false` wildcard → two one-hot splits on the SAME cat feature would be judged *different* internal features → wrong interaction/PVC | add `(OneHot, OneHot)` arm keyed on `cat_feature` (upstream `TFeature` identity ignores the value, exactly as it ignores the float border) |
| `sum_models` | `cb-model/src/model_sum.rs:24` | `splits: tree.splits.clone()` — kind-agnostic | fine |
| `partial_dependence` | `cb-model/src/partial_dependence.rs` | operates on `float_feature_borders` / float column space only (`:123, :229`) | needs a typed rejection for one-hot models |
| `staged_predict` | routes through `predict_raw*` | inherits `passes_split` | fine once A is done |
| `.cbm` load one-hot bin | `cb-model/src/cbm.rs:1041-1045` | **already a typed error**: `"one-hot split unsupported (v1)"` | replace with a real arm |
| `region_leaf` | `cb-model/src/apply.rs:283-303` | ignores `RegionLevel.one_hot` entirely | out of scope, but note it |

---

## 4. Upstream `.cbm` one-hot encoding — EMPIRICALLY PINNED

### 4.1 A real upstream one-hot model was produced

`[VERIFIED: CMD .venv/bin/python — catboost 1.2.10, Python 3.12.13]`. Config:
2 cat columns (cardinality 2 and 3) + 1 float column, `one_hot_max_size=5`,
`iterations=3, depth=3, max_ctr_complexity=0`. Artifacts in the scratchpad
(`onehot.cbm`, `onehot.json`, `onehot_preds.npy`). **Note the repo has NO such
fixture today** — `crates/cb-oracle/fixtures/one_hot_cat/` has no model at all;
`ctr_load/` has `simple.cbm`/`combo.cbm` but those are CTR, not one-hot
`[VERIFIED: CMD ls crates/cb-oracle/fixtures/{one_hot_cat,ctr_load}]`.

### 4.2 The combined bin-index space — CONFIRMED `Float → OneHot → Ctr`

Upstream `model.json` for the produced model
`[VERIFIED: CMD .venv/bin/python -m json …]`:

```
features_info.float_features  = [ { feature_index: 0, borders: [-0.23132047, 0.17452943] } ]
features_info.categorical_features = [
  { feature_index: 0, flat_feature_index: 1, values: [-1438285038] },
  { feature_index: 1, flat_feature_index: 2, values: [-1438285038, -1284790409] } ]

tree 0 splits: [ {cat 0, split_index 2, OneHotFeature, value -1438285038},
                 {cat 1, split_index 3, OneHotFeature, value -1438285038},
                 {cat 1, split_index 4, OneHotFeature, value -1284790409} ]
tree 1 splits: [ {cat 0, split_index 2, OneHot}, {cat 1, split_index 3, OneHot},
                 {float 0, split_index 0, border -0.23132047} ]
tree 2 splits: [ {cat 0, split_index 2, OneHot}, {cat 1, split_index 4, OneHot},
                 {float 0, split_index 1, border  0.17452943} ]
```

Offsets: float feature 0 has 2 borders → global bins **0,1**; cat 0 contributes
1 value → bin **2**; cat 1 contributes 2 values → bins **3,4**. That is exactly
`build_combined_bins`'s layout
`[VERIFIED: LOCAL crates/cb-model/src/cbm.rs:375-407]` — float bins in
feature-major/border-ascending order (`build_bin_features`, `cbm.rs:85-93`), then
one-hot bins in cat-feature-major / `Values`-array order, then CTR bins.

### 4.3 End-to-end confirmation through OUR decoder

`[VERIFIED: CMD cargo run (scratch crate depending on cb-model by path) — load_cbm(onehot.cbm)]`
→ `ERR malformed model: one-hot split unsupported (v1)`; framing
`magic=[CBM1] core_len=6984 total=6992` (tail = 0, i.e. **no CTR model-parts tail
for a pure one-hot model**).

This is a strong signal: the decoder classified the global split index into the
one-hot *range*, which is only possible if our float-prefix width and one-hot
offset match upstream's. A wrong offset would have produced a `BinKind::Float`
mis-decode or an out-of-range error instead.

### 4.4 The `TOneHotFeature` field layout

`[VERIFIED: LOCAL crates/cb-model/src/generated/model_generated.rs:1830-1902]`

```
table TOneHotFeature { Index: i32 = -1; Values: [i32]; StringValues: [string]; }
table TCatFeature   { Index: i32 = -1; FlatIndex: i32 = -1; FeatureId: string; UsedInModel: bool = true; }
TModelTrees.CatFeatures    -> VT = 12
TModelTrees.OneHotFeatures -> VT = 16
struct TOneHotSplit { Index: i32; Value: i32; }   // 8 bytes, used inside TFeatureCombination
```

`Values` are **`i32` categorical hashes** (`calc_cat_feature_hash` output), not
bin ordinals. Byte scan of `onehot.cbm` finds `-1438285038` at offsets 6860 and
6884 and `-1284790409` at 6864 — i.e. cat 1's `Values = [-1438285038,
-1284790409]` (contiguous) and cat 0's `Values = [-1438285038]`, with **no
`StringValues`** emitted `[VERIFIED: CMD python struct scan + strings]`.

**Pruning:** upstream emits only the one-hot values actually **used** by some
split, not the full cardinality. Probe: a cardinality-6 cat feature that never
won a split produced `categorical_features: [{feature_index:0, flat_feature_index:1}]`
with **no `values` key at all** and zero one-hot bins
`[VERIFIED: CMD .venv/bin/python probe]`. So the save path only needs to emit the
distinct `(cat_feature, value)` pairs the trees reference — mirroring exactly
what `build_ctr_features` already does for CTR identities
(`cb-model/src/cbm.rs:161-230`).

**Apply-side consequence:** the split test at apply time is
`calc_cat_feature_hash(raw_value) == Values[k]`, i.e. against the **raw i32
hash**, NOT against a first-seen `PerfectHash` bin. The trainer-side
`OneHotSplit.value: u32` is a `cb_data::PerfectHash` **bin**
`[VERIFIED: LOCAL crates/cb-train/src/tree.rs:125-136]`. **These are two
different value spaces** — the lift must map bin → the original hash. This is a
named landmine: if both sides use the perfect-hash bin, our own oracle passes and
the produced `.cbm` is unreadable/wrong for upstream.

### 4.5 `model.json` behaviour

Our `decode_json` fails **loudly** on an upstream one-hot json:
`[VERIFIED: CMD scratch probe — "model.json (de)serialization error: missing field 'border' at line 440 column 13"]`
because `SplitJson` requires `border` + `float_feature_index` with no serde
default (`json.rs:44-54`) and upstream's one-hot split carries
`{cat_feature_index, split_index, split_type, value}` instead. Acceptable today;
the message should be replaced with a typed "one-hot json unsupported" if json
one-hot is deferred.

### 4.6 One-hot vs CTR are mutually exclusive per column (matches our routing)

With catboost defaults (no `max_ctr_complexity=0` override), the same 2-cat
dataset emits **no `ctr_features` section at all** and all-one-hot trees
`[VERIFIED: CMD .venv/bin/python probe]`. Confirms the phase-1 simplification
"one-hot columns never also get CTRs" is upstream-faithful.

---

## 5. The GPU device path

### 5.1 `grow_one_hot_tree` cannot carry the performance goal

`score_candidate_any` (`tree.rs:3052-3068`) re-runs `assign_leaves_any` over
**all `n` objects for every candidate at every level** and re-reduces leaf stats
— O(levels × candidates × n) with no histogram reuse
`[VERIFIED: LOCAL crates/cb-train/src/tree.rs:3048-3130]`. Contrast
`select_level_plain` (`tree.rs:931-…`), which fuses per-feature histogram
derive + `O(n_bins)` prefix scoring inside one rayon task per feature
`[VERIFIED: LOCAL crates/cb-train/src/tree.rs:931-960]`.

**Planning consequence:** "fast enough to beat official CatBoost GPU" requires
either the device path or a fused-histogram CPU path. `grow_one_hot_tree` should
be retained only as the frozen ≤1e-5 correctness reference for a self-oracle.

### 5.2 `device_host_eligible` — full precondition list

`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:3055-3122]`

1. `group_spans.is_none()` (no ranking/grouped loss)
2. `ordered_learning_perm.is_none()` (Plain boosting only)
3. `materialized_ctr_features.is_empty()` **and** `structure_fold_columns.iter().all(Vec::is_empty)`
4. `!penalties_active`
5. `params.monotone_constraints.is_empty()`
6. `grow_policy ∈ {SymmetricTree, Depthwise, Lossguide, Region}`
7. `approx_dimension == 1 && !is_multiclass && !is_multilabel`
8. `bootstrap_type == No` **or** (`∈ {Bayesian,Bernoulli,Mvs,Poisson}` **and** `grow_policy == SymmetricTree`)
9. `params.random_strength == 0.0`
10. `eval_sets.is_empty()`
11. `matrix.n_features() > 0`  ← **a cat-only pool has 0 float features → device path OFF today**
12. `weights.iter().all(|&w| w == 1.0)` (WR-03: unweighted der)
13. `bias == 0.0` (CR-01: session seeds resident approx to zero)
14. `leaf_method ∈ {Gradient, Simple}` (CR-02)

Then `begin_device_training(...)` may still decline (`Ok(false)`).
`DeviceTrainConfig::is_covered_regime()` also requires `ctr.is_none()`
`[VERIFIED: LOCAL crates/cb-compute/src/runtime.rs:1142-1160]`.

Item **11** is a direct blocker for a one-hot-only pool; item **3** blocks any
mixed one-hot + CTR pool.

### 5.3 What the resident session uploads

- `quantize_feature_major(feature_values, feature_borders, n) -> (Vec<u32>, usize)`
  builds a **feature-major plain cindex** `bins[f*n + obj] =
  borders.partition_point(|b| v > b)` plus a **single uniform**
  `n_bins = max_f(borders[f].len() + 1)`
  `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:2196-2233]`.
  `n_bins` is a **comptime** for the hist/score kernels (`#[comptime] n_bins`,
  `kernels.rs:3374`), so all features share one bin-line width. A one-hot column
  of cardinality `c` slots in as a column with `c` buckets (padded); no schema
  change needed, only wasted histogram cells.
- `pack_cindex` (`gpu_runtime/cindex.rs:154-293`) repacks into grouped
  bit-packed words with per-feature `TCFeature { offset, mask, shift,
  first_fold_index, folds, one_hot_feature }`. **`one_hot_feature` is already in
  the frozen descriptor and is set to `false` at `cindex.rs:226`.** `read_bin` =
  `(words[offset + obj] >> shift) & mask` (`kernels.rs:2776`).
  `PackedCindex::device_arrays()` currently exports only `(offsets, shifts,
  masks)` — a fourth `one_hot` array (or a comptime/launch flag) would be needed.

### 5.4 What a device one-hot split needs

**Split application** — `partition_split_kernel` (`kernels.rs:3731-3773`) hard-codes
`if read_bin(...) > bin { new_leaf |= 1 << level_bit }`. One-hot needs `== value`.
Two viable shapes:

- add `#[comptime] one_hot: bool` and an if-as-statement branch (the **existing
  precedent** in `pairwise_hist_nonbinary_kernel`, `kernels.rs:1107, 1140`, which
  already carries exactly this comptime flag), or
- a second kernel. The comptime route is cheaper and matches house style.

Host launchers to touch: `launch_partition_split_into` and
`launch_partition_split_packed_into` (`gpu_runtime/mod.rs:2014-2078`).

**Split scoring** — `find_optimal_split_kernel` (`kernels.rs:3367-…`) folds
`left = Σ bins 0..=border`, `right = Σ bins border+1..n_bins` for candidate
`c = feature * n_bins + border`. For one-hot the fold becomes
`left = bin_sums[value]`, `right = total − left`. Same candidate indexing, same
argmax + lowest-index tie-break — a comptime `one_hot` arm again.
The resident LDS path (`partition_hist2_lds_kernel`,
`partition_hist2_nonbinary_kernel`, arbitrated by `hist_fill_path`,
`gpu_runtime/mod.rs:2588-2620`) produces the same 2-channel per-bin histogram, so
**the fill needs no change at all** — only the fold does. That is the cheapest
possible device change and the key reason the perf goal is plausible.

**Seam types to extend**: `DeviceGrownTree.splits: Vec<(u32, u32)>` carries no
kind `[VERIFIED: LOCAL crates/cb-compute/src/runtime.rs:931-935]`; needs a
parallel `one_hot: Vec<bool>` or a 3-tuple. `DeviceGrownTree.region_path`'s
4-tuple `(feature, bin, direction, one_hot)` is the in-repo precedent
(`runtime.rs:975-986`).

**CubeCL rules** (CLAUDE.md, from AGENTS.md): kernels generic over `F: Float`
(never a hard-coded float type); read
`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` before
writing kernel code; on a build error consult
`…/cubecl_error_solution_guide/` (directory — CLAUDE.md's `.md` filename is
stale). `cubecl = 0.10.0` `[VERIFIED: LOCAL Cargo.toml:38, Cargo.lock:1083-1085]`.
House constraints observed in existing kernels: **if-as-statement only** (no
if-expressions), grid-stride = `CUBE_COUNT * CUBE_DIM` (never a literal 32/64,
D-09), `f32::MIN` as the score sentinel (a literal `-inf` fails HIP/comgr JIT on
gfx1100) `[VERIFIED: LOCAL crates/cb-backend/src/kernels.rs:3396-3402]`.

### 5.5 The alternative that needs zero kernels

A one-hot column of cardinality `c` can be materialized as `c` binary float
columns with the single border `0.5` — exactly what `one_hot_encode` does in the
existing oracle test (`one_hot_oracle_test.rs:76-88`). That reuses the **entire**
float device path unchanged and would meet the speed goal trivially. Its cost:
the trained model then contains `ModelSplit::Float` splits on synthetic features
and **cannot be serialized as an upstream-readable one-hot `.cbm`**, and the
float-feature index space is polluted. Recommend **rejecting** it as the model
representation, but it is a legitimate *interim* internal representation if the
plan wants to decouple "training works" from "serializes as upstream one-hot".
Flag explicitly so the planner makes the call rather than drifting into it.

---

## 6. Oracle + fixture infrastructure

### 6.1 The `gen_fixtures.py` dispatch trap — CONFIRMED

`crates/cb-oracle/generator/gen_fixtures.py` is 3492 lines. The `__main__` block
(`:3449-3492`) is an `if/elif` chain over `sys.argv` with a terminal
**`else: main()`** `[VERIFIED: LOCAL crates/cb-oracle/generator/gen_fixtures.py:3449-3492]`.
An unrecognised flag therefore regenerates the **entire committed corpus** (65
fixture directories `[VERIFIED: CMD ls crates/cb-oracle/fixtures | wc -l]`).

Existing `--*-only` flags: `--wave1-only`, `--wave2-only`, `--wave3-only`,
`--multiclass-only`, `--multilabel-only`, `--mvs-seeds-only`,
`--bootstrap-dev-only`, `--multiquantile-only`.

**The pattern a new `--one-hot-only` must mirror** (from `gen_bootstrap_dev`,
`:863-905`):
1. a dedicated `gen_<scenario>() -> None` with a docstring stating *why* a new
   family exists and that it is **reachable only through the flag, never
   `main()`**;
2. reuse a frozen input dir where possible (never rewrite `inputs/`);
3. pin **every** knob catboost's raw dict API defaults differently from the Rust
   builder — `random_strength=0` in particular (see MEMORY: the
   `cv ORCH-01 random_strength` trap);
4. a `gen_<scenario>_only()` wrapper;
5. a new `elif "--one-hot-only" in sys.argv:` arm **before** the final `else`.

### 6.2 The existing one-hot oracle test

`crates/cb-train/tests/one_hot_oracle_test.rs` (320 lines):
- consumes **no fixture**;
- `one_hot_path_selection_boundary` — pure unit assertions on `route_categorical`;
- `one_hot_predict_matches_oracle_locked_float_reference` — self-oracle vs
  `cb_train::train` on one-hot-encoded binary columns, via `compare_stage(Stage::StagedApprox/Predictions)`;
- `no_permutation_in_one_hot_only_path` — determinism.

**Re-pointability:** the `one_hot_encode` reference and the
`predict_all` helper are directly reusable against a production
`train_cat(..., cat_columns)` call once the grow path is wired. The test-local
`train_one_hot_only` driver should then be **deleted**, not kept in parallel
(it hard-codes Gradient/RMSE/no-sampling and would drift). A genuine upstream
oracle (new fixture from §4.1) can and should replace the transitive lock.

### 6.3 Test-file conventions

Source/test separation is enforced as `#[cfg(test)] #[path = "<x>_test.rs"] mod tests;`
at the bottom of the production file (e.g. `cb-model/src/cbm.rs:61-66`,
`cb-model/src/gpu_apply.rs:168-170`) — **not** an inline `mod tests { }`.
Integration tests live in `crates/<crate>/tests/*_test.rs`.

---

## 7. Speed / bench infrastructure

### 7.1 `bench/quick_gpu_speed/bench.py`

`[VERIFIED: LOCAL bench/quick_gpu_speed/bench.py:1-105]`

- Kaggle CUDA script kernel; `WORK = "/kaggle/working"`.
- Workload: `SPEED_CONFIG = 300_000 rows × 50 features, seed 42`, `DEPTH=6`,
  `ITERS=30`, `LR=0.1`, `L2=3.0`, `BORDER_COUNT=32`, `RANDOM_SEED=42`.
- catboost_rs is built as a **maturin wheel with the `cuda` Cargo feature** and
  driven through the **real public Python `.fit()`**, not a cargo test.
- Official CatBoost is invoked as `task_type='GPU'` with a matched config.
- `build_eligibility_audit()` is a **static** audit of the `device_host_eligible`
  preconditions; the script states plainly that device activation is
  **not observable from Python** and a silent CPU fallback cannot be ruled out.

### 7.2 Current measured ratio (round 4, P100)

`[VERIFIED: LOCAL bench/quick_gpu_speed/kaggle-output-260716-r4c/result.json]`

| arm | s |
|---|---|
| catboost_rs RMSE | **1.2382** |
| catboost_rs Logloss | **1.2976** |
| official CatBoost GPU RMSE | 1.3000 |
| official CatBoost GPU Logloss | 1.3634 |
| XGBoost GPU hist RMSE / Logloss | 1.1229 / 1.0720 |
| sklearn HGB CPU RMSE | 2.7341 |

Speedups (competitor / catboost_rs): official CatBoost GPU **1.0499×** (RMSE) /
**1.0507×** (Logloss); XGBoost GPU **0.9069× / 0.8261×** (XGBoost still ahead).
GPU: Tesla P100-PCIE-16GB, driver 580.159.04, nvcc 12.8.

Stage attribution from `CB_GPU_PROF`: `fill 755.98 ms`, `score 193.08 ms`,
`stats_read 161.1 ms`, `split 65.57 ms`, `derive 84.62 ms` (of 2598.71 ms total),
plus fixed `fit-prep 225.86 ms`, `quantize 70.45 ms`, `begin 203.15 ms`.
**A one-hot fold change touches `score` (193 ms) and `split` (65 ms), not `fill`
(756 ms)** — so a correct one-hot arm should be roughly speed-neutral.

`kernel-metadata.json`: `id = yensen2/catboost-rs-quick-gpu-speed-check`,
`dataset_sources = ["yensen2/catboost-rs-quick-gpu-speed-src"]`.
**MEMORY warning:** the `yensen2` datasets are recorded as dead; the newer
bootstrap bench uses `id = boomvector/catboost-rs-bootstrap-gpu` with
**no dataset source** and a git-clone in-script
`[VERIFIED: LOCAL bench/bootstrap_gpu/kernel-metadata.json]`. A new bench should
follow the `boomvector` + git-clone pattern.

### 7.3 Colab T4

**There is no Colab sibling of `quick_gpu_speed/bench.py`.** The only Colab
runner is `bench/bootstrap_gpu/bootstrap_bench_colab.py` (21 588 bytes)
`[VERIFIED: CMD ls bench/bootstrap_gpu/]`, with output in
`bench/bootstrap_gpu/colab-t4-260730/{report.md,t4_results.txt}` (Tesla T4,
driver 580.82.07, CUDA 12.8, rustc 1.97.1, verdict ORACLE-PASS 26/26).

Its three deliberate deltas vs the Kaggle script `[VERIFIED: LOCAL bench/bootstrap_gpu/bootstrap_bench_colab.py:1-45]`:
1. `WORK = "/content/bench_out"`, `REPO = "/content/cbrs"` — source is the tree
   **already staged** by the driver (uploaded working tree), not a git clone;
2. no `only_No_is_gpu_eligible` caveat;
3. **the "device activation not observable" caveat is CLOSED**: Part B0 runs a
   short fit per arm under `CB_GPU_PROF=1` and greps for `CB_GPU_PROF tree`
   lines; an arm with no such line is labelled as not-device-run.

**To run the one-hot speed check on Colab T4**, port `quick_gpu_speed/bench.py`
onto that shape: swap `WORK`, stage the repo at `/content/cbrs`, and adopt the
`CB_GPU_PROF` activation probe (which also finally removes the
`activation_observable: false` caveat from the speed report).

---

## 8. Prior specs and plan overlap

### 8.1 TreeFinder

`[VERIFIED: TREE_FINDER search_hierarchy("one-hot categorical training encoding
path ORD-04")]` returned only `.planning/plans/snapshot-resume/SPEC.md`
(`document_id 1d68d3de-16aa-4790-9027-274ad5160038`),
`.planning/plans/snapshot-resume/PLAN.md` (`a5dcda3e-8246-4a3e-bb84-310290098a78`)
and `.planning/plans/xgboost-rust-rewrite/SPEC.md`
(`51f70f3d-f89f-4c57-bcae-35cbaaca8f94`) — **none relevant**. The index is sparse
as expected; on-disk `.planning/` is authoritative.

### 8.2 On-disk `.planning/`

`[VERIFIED: CMD rg -rn -i "one.hot|ORD-04|EncodingPath" .planning --glob '*.md' -l]`

| doc | relevance |
|---|---|
| **`.planning/plans/catboost-builder-cat-features-routing/{SPEC,PLAN,PLAN-CHECK,research}.md`** (**UNTRACKED / in-flight**, per `git status`) | **HIGH — ordering dependency.** Wires `cat_features` + CTR through the `CatBoostBuilder` facade and Python bindings, adds a `.one_hot_max_size(k)` setter (SPEC-CATF-01). Explicitly **does not** implement one-hot training. This phase should land **after** it (or explicitly declare the setter as a prerequisite) so `one_hot_max_size` is reachable at all. |
| `.planning/phases/23-ctr-model-loading/cbm-ctr-load/SPEC.md:44-46,160,191` | Owns decision **CTR-05**: one-hot splits are out of `.cbm` v1 and produce a typed `ModelError`; one-hot FEATURE tables are still counted for the bin offset. A one-hot phase **supersedes** CTR-05 — update, don't duplicate. Test `cb-model/src/cbm_test.rs::one_hot_split_index_is_typed_error` will need re-pointing. |
| `.planning/phases/23-ctr-model-loading/cbm-ctr-save/PLAN.md:36,46` | "v1 has no one-hot, so the CTR range begins right after the float bins" — the save-side offset assumption a one-hot phase must revise (`cbm.rs:524-526`). |
| `.planning/phases/17-model-export/onnx-export/SPEC.md:200` | states `ModelSplit` has exactly two variants and no one-hot; the ONNX guard must be revisited. |
| `.planning/phases/24-ctr-split-search-correctness/**` | **OUT OF SCOPE** per instruction (ORD-06/ORD-07 combination-CTR gating). Distinct from the §3.2 ordering defect. |
| `.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/**` | fstr/interaction cat-index conventions (`flat_cat_index`) a one-hot arm should reuse. |

No existing phase directory covers one-hot **training**.

---

## 9. Proposed impact scope

### Must change

| file / symbol | why |
|---|---|
| `crates/cb-train/src/candidates.rs` | expose the one-hot-routed absolute cat index list (mirror of `eligible_absolute`) |
| `crates/cb-train/src/boosting.rs:2678` | build `FeatureMatrix` with `cat_bins` for one-hot-routed columns (via `cb_data::perfect_hash_bins`, already imported at `:2745`) |
| `crates/cb-train/src/boosting.rs:2721-2749` | add the `EncodingPath::OneHot` partition alongside `eligible_absolute` |
| `crates/cb-train/src/boosting.rs:4042-4200` | grower dispatch: route to the one-hot-aware oblivious search when one-hot columns exist |
| `crates/cb-train/src/boosting.rs:4786-4824` | persist ordered mixed-kind splits onto `ObliviousTree` |
| `crates/cb-train/src/boosting.rs:775-794` (`ObliviousTree`) | add `one_hot_splits` + an ordered kind list (or a single ordered `Vec<AnySplitKind>`); **fix the order-loss of §3.2** |
| `crates/cb-train/src/tree.rs:205-273` (`GrownTree`, `LevelKind`) | add `one_hot_splits` + `LevelKind::OneHot` |
| `crates/cb-train/src/tree.rs:2919-3009` | new/extended one-hot-aware level search (fused, not `score_candidate_any`) |
| `crates/cb-model/src/model.rs:71-98` (`ModelSplit`) | third variant `OneHot { cat_feature, value_hash }` (raw i32 hash — §4.4) |
| `crates/cb-model/src/model.rs:326-363` (`from_trained`) | emit splits in **level order** |
| `crates/cb-model/src/apply.rs:196-201` | `passes_split` one-hot arm |
| `crates/cb-model/src/cbm.rs:85-114, 375-407, 519-668, 1032-1064` | `OneHotFeatures` + `CatFeatures` emit, offset math, load arm replacing the typed rejection |
| `crates/cb-model/src/fstr.rs:130-177, 441-473` | `feature_count`/`cat_feature_count`/`split_flat_indices`/`same_internal_feature` |
| `crates/cb-model/src/export/onnx.rs:106-113` & `coreml.rs:106-113` | widen the guards (**silently wrong today**) |
| `crates/cb-model/src/gpu_apply.rs:50-57` | explicit `OneHotSplits` rejection variant |
| `crates/cb-model/src/json.rs:474-485, 648-663` | stop silently dropping; typed error or real one-hot json shape |
| `crates/cb-model/src/shap.rs:550-552, 725-733` | typed rejection or a real arm (**silently wrong today**) |
| `crates/cb-oracle/generator/gen_fixtures.py` | `gen_one_hot()` + `gen_one_hot_only()` + `--one-hot-only` arm (before the `else`) |
| `crates/cb-oracle/fixtures/one_hot_train/` (new) | upstream 1.2.10 `.cbm` + `model.json` + preds + inputs |
| `crates/cb-train/tests/one_hot_oracle_test.rs` | re-point at production `train_cat`; delete the test-local driver |

### May change (device path — if the GPU goal is in this phase)

| file / symbol | why |
|---|---|
| `crates/cb-train/src/boosting.rs:3055-3122` | eligibility: `matrix.n_features() > 0` blocks a cat-only pool; one-hot admission arm |
| `crates/cb-train/src/boosting.rs:2196-2233` | quantization must also emit one-hot bin columns + bucket counts |
| `crates/cb-backend/src/gpu_runtime/cindex.rs:220-228, 93-109` | set `one_hot_feature` truthfully; export a 4th device array |
| `crates/cb-backend/src/kernels.rs:3367-…` (`find_optimal_split_kernel`) | `#[comptime] one_hot` fold arm (`left = bin_sums[v]`) |
| `crates/cb-backend/src/kernels.rs:3731-3773` (`partition_split_kernel`) | `#[comptime] one_hot` equality arm |
| `crates/cb-backend/src/gpu_runtime/mod.rs:2014-2078` | launcher pass-through |
| `crates/cb-backend/src/gpu_runtime/session.rs:940, 1551-1922` | resident grow loop + `DeviceGrownTree` emit |
| `crates/cb-compute/src/runtime.rs:931-935, 1083-1160` | `DeviceGrownTree.splits` kind field; `DeviceTrainConfig` one-hot arm + `is_covered_regime` |
| `bench/quick_gpu_speed/bench.py` (or a Colab sibling) | a one-hot matched-config arm |

### Verification only

- `crates/cb-model/src/model_sum.rs` (kind-agnostic clone)
- `crates/cb-model/src/predict.rs`, staged predict (route through `passes_split`)
- every float-only oracle test (D-04 byte-identity no-regression)
- `crates/catboost-rs/src/builder.rs:287` (the pinned `one_hot_max_size_default()`)

### Explicitly out of scope

- ORD-06/ORD-07 combination-CTR split-search correctness (`.planning/phases/24-…`)
- one-hot × CTR in the same tree (recommend a typed rejection in phase 1)
- one-hot on non-symmetric / Region / pairwise grow policies
- `region_leaf`'s ignored `one_hot` flag (`apply.rs:283-303`)
- ONNX/CoreML *implementation* of one-hot branches (guards only)

---

## 10. Risks, pitfalls, open questions

| # | item | severity |
|---|---|---|
| R1 | **Mixed-kind split ORDER loss in `from_trained`** (§3.2). Copying the CTR pattern for one-hot reproduces a silent mis-prediction. | **BLOCKING for design** |
| R2 | **Value-space mismatch**: trainer `OneHotSplit.value` is a `PerfectHash` bin; upstream `.cbm` stores the raw `calc_cat_feature_hash` i32. If both our sides use the bin, our oracle passes and the `.cbm` is wrong for upstream. | **HIGH** |
| R3 | **Silently-passing export guards** (`onnx.rs:110`, `coreml.rs:110`, `gpu_apply.rs:54`) and **silently-dropping filters** (`json.rs:477`, `shap.rs:551`, `fstr.rs:138`) — no compiler error. | **HIGH** |
| R4 | `grow_one_hot_tree` is O(candidates × n) per level — unusable for the speed goal (§5.1). | **HIGH** |
| R5 | RNG draw-order: the one-hot level search must consume the per-level `randSeed` + per-candidate `std_normal` draws exactly as `greedy_tensor_search_oblivious_perturbed` does whenever `draws_active`, or every later tree's bootstrap sample shifts (`WR01-S7`, `boosting.rs:3272-3277`). One-hot candidates change the **candidate count**, hence the draw count. Upstream ground truth needed. | **HIGH** |
| R6 | `device_host_eligible` requires `matrix.n_features() > 0` — a cat-only pool never reaches the device today. | MEDIUM |
| R7 | `gen_fixtures.py` unrecognised-argv → full corpus regeneration (§6.1). | MEDIUM (process) |
| R8 | Fixture nondeterminism: MEMORY records catboost quantization as run-to-run nondeterministic → freeze the produced one-hot fixture, don't regenerate casually. | MEDIUM |
| R9 | catboost raw-dict defaults ≠ builder defaults (`random_strength=0` in particular) — pin every knob on BOTH sides of any new fixture. | MEDIUM |
| R10 | `--no-default-features --features rocm` test build is broken (pre-existing, `CpuBackend` imported unconditionally by oracle tests). Any new rocm-gated test inherits this. | LOW (pre-existing) |
| R11 | Uniform `n_bins` on the device is `max_f(borders+1)`; adding a low-cardinality one-hot column is free, but a high-cardinality one-hot (`one_hot_max_size` up to 255 on upstream GPU) would blow up the per-feature bin line. | LOW–MEDIUM |

### Open questions the planner must resolve

1. **Is fixing the CTR split-order loss (§3.2) in scope, or is it split out as its
   own bug phase?** (MEMORY: the user wants full spec-tdd rigor per discovered
   upstream/latent bug, chained, with a check-in at each discovery.)
2. **Does the one-hot candidate set change upstream's per-level RNG draw count?**
   Requires an instrumented 1.2.10 run (the technique used for
   `.planning/plans/bayesian-rng-draw-accounting/`). Blocks any one-hot ×
   bootstrap/random_strength combination.
3. **Phase-1 model representation:** true `ModelSplit::OneHot` (upstream-compatible,
   touches ~15 sites) vs. synthetic binary float columns (zero kernel work, not
   upstream-serializable)? Recommend the former; the question must be explicit.
4. **One-hot × CTR in one tree** — typed rejection in phase 1, or a three-way
   candidate union now?
5. **Does the speed run go on Kaggle P100 (existing but `yensen2`-dataset-dead) or
   Colab T4 (needs a new runner)?**
6. **Where does the new fixture live** — a new `one_hot_train/` family, or extend
   `one_hot_cat/`? (Recommend new; `one_hot_cat/` is a frozen Wave-0 anchor.)

---

## 11. Confidence assessment

| finding | confidence |
|---|---|
| `grow_one_hot_tree` unreferenced in production; one-hot columns silently dropped | **HIGH** — codegraph + rg + full read of `train_inner`'s dispatch |
| Upstream `.cbm` bin space = `Float → OneHot → Ctr`, one-hot values are pruned raw i32 hashes | **HIGH** — real upstream 1.2.10 artifact produced and round-tripped through our own decoder |
| Upstream mixes one-hot and float splits within one tree, order-significant | **HIGH** — observed in trees 1 and 2 of the produced model |
| `from_trained` loses mixed-kind order (latent CTR bug) | **HIGH** on the code path; **MEDIUM** that it is reachable with committed fixtures (not empirically triggered) |
| Full `ModelSplit` consumer list and which sites are compiler-forced vs silent | **HIGH** — exhaustive rg + per-site read |
| `device_host_eligible` precondition list | **HIGH** — verbatim read |
| Device changes needed = fold arm + split-test arm; hist fill unchanged | **MEDIUM–HIGH** — kernel sources read; not prototyped |
| `grow_one_hot_tree` is O(candidates × n) and unfit for the perf goal | **HIGH** — source read; not benchmarked |
| Bench methodology + current 1.05× ratio vs official CatBoost GPU | **HIGH** — committed `result.json` |
| No Colab sibling for the speed bench | **HIGH** — directory listing |
| `--features rocm` builds; `--no-default-features --features rocm` does not | **HIGH** — both commands run |
| RNG draw-count impact of one-hot candidates | **LOW** — not investigated upstream; explicit open question |
| Whether upstream emits `StringValues` for string-valued cats in other configs | **LOW** — absent in my probe; only one config tested |
| TreeFinder coverage of this area | **HIGH (that it is empty)** |

---

## 12. Sources

**Project (local):** `CLAUDE.md`; `.planning/plans/catboost-builder-cat-features-routing/{SPEC,PLAN,PLAN-CHECK,research}.md`;
`.planning/phases/23-ctr-model-loading/cbm-ctr-{load,save}/{SPEC,PLAN}.md`;
`.planning/phases/17-model-export/onnx-export/SPEC.md`;
`.planning/phases/24-ctr-split-search-correctness/**`;
`bench/quick_gpu_speed/{bench.py,kernel-metadata.json,kaggle-output-260716-r4c/{result.json,report.md}}`;
`bench/bootstrap_gpu/{bootstrap_bench_colab.py,kernel-metadata.json,colab-t4-260730/report.md}`.

**CodeGraph queries:** `grow_one_hot_tree train_inner route_categorical EncodingPath one_hot_max_size`;
`GrownTree LevelKind CtrSplitSpec greedy_tensor_search_oblivious_with_ctr ctr_splits_for_tree`;
`predict_raw_one passes_split apply oblivious leaf index ModelSplit`.

**TreeFinder:** `search_hierarchy("one-hot categorical training encoding path ORD-04")` — no relevant hits.

**Source files read (verbatim):** `crates/cb-train/src/{boosting.rs,tree.rs,candidates.rs,lib.rs}`;
`crates/cb-train/tests/one_hot_oracle_test.rs`;
`crates/cb-model/src/{model.rs,apply.rs,cbm.rs,json.rs,fstr.rs,shap.rs,gpu_apply.rs,export/onnx.rs,export/coreml.rs,model_sum.rs,partial_dependence.rs}`;
`crates/cb-model/src/generated/model_generated.rs`;
`crates/cb-backend/src/{kernels.rs,gpu_runtime/{cindex.rs,mod.rs,session.rs}}`;
`crates/cb-compute/src/runtime.rs`; `crates/cb-oracle/generator/gen_fixtures.py`;
`crates/catboost-rs/src/{builder.rs,model.rs}`; `crates/catboost-rs-py/src/params.rs`.

**Commands run:**
`cargo test -p cb-train --features rocm --no-run`;
`cargo test -p cb-train --no-default-features --features rocm --no-run`;
scratch cargo crate `probe` depending on `cb-model` by path → `load_cbm`/`load_json` on the produced upstream artifacts;
`.venv/bin/python` (catboost 1.2.10, Python 3.12.13) producing `onehot.{cbm,json}`, `probe.{cbm,json}`, `default_ctr.json`, `d.json`;
`python struct` byte scan of `onehot.cbm`; `strings onehot.cbm`.

**Manifests / versions:** `Cargo.toml:38` + `Cargo.lock:1083-1085` → `cubecl 0.10.0`;
`crates/cb-model/Cargo.toml` → `flatbuffers 25.12.19`, `prost`, `serde`/`serde_json`, `thiserror`;
`.venv` → `catboost 1.2.10`, `numpy`, Python 3.12.13.

**External:** none needed — every claim was resolvable from the repository or
from a locally-executed upstream CatBoost 1.2.10 run.

**Note on artifacts:** the upstream one-hot `.cbm`/`.json` I produced live in the
session scratchpad, **not** in the repo. Nothing under `crates/`, `bench/`, or
`.planning/` was modified except this file.
