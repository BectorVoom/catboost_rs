---
title: TDD implementation plan — one-hot categorical training (device path + upstream-compatible model)
kind: plan
status: draft
revision: 5
spec: .planning/plans/one-hot-categorical-training/SPEC.md
research: .planning/plans/one-hot-categorical-training/research.md
plan_check: .planning/plans/one-hot-categorical-training/PLAN-CHECK.md
updated_at: 2026-07-31
task_count: 35
spec_count: 31
---

# TDD implementation plan — one-hot categorical training

Derived goal-backward from SPEC.md §6 (A1–A13) and §5 (SPEC-OH-01..31).
Every anchor below was re-verified against on-disk source (CodeGraph
`codegraph_explore` + line-ranged reads). Anchors that DIFFER from research.md
or SPEC.md are called out in §2 "Corrections".

**Revision 2** incorporated every finding of PLAN-CHECK pass 1 (4 CRITICAL,
11 MAJOR, 7 MINOR, plus its "Potential Bugs" and "Unverified Items" sections);
**§9** is that disposition table. **Revision 3** incorporates every finding of
PLAN-CHECK pass 2 (0 BLOCKER, 0 CRITICAL, **3 MAJOR, 4 MINOR**); **§9b** is that
disposition table, including a per-test placement-confirmation table for
MAJOR-C. **Revision 4** incorporates every finding of PLAN-CHECK pass 3
(0 BLOCKER, 0 CRITICAL, **2 MAJOR, 4 MINOR**); **§9c** is that disposition
table, including an end-to-end `real_folds` plumbing trace and the float-only
invariance argument. **Revision 5** incorporates every finding of PLAN-CHECK
pass 4 (0 BLOCKER, 0 CRITICAL, **1 MAJOR, 2 MINOR**) — all three text-only;
**§9d** is that disposition table. All SPEC amendments (v2, v3 and v4 — both
scorers / real-bin bound / both exclusions; the four SHAP consumers; the
`byte-identical` → `numerically identical` fix in SPEC-OH-22 **and** SPEC-OH-23;
and the "which `folds`" clarification) have been applied by the coordinator and
re-verified by the checker; `spec_version: 4`.
**Revision 5 requires no SPEC amendment.** **This plan does not edit SPEC.md.**

**Constraint recap that every task inherits (CLAUDE.md):** no `unwrap()` in
production; tests live in a sibling `*_test.rs` mounted with
`#[cfg(test)] #[path = "<x>_test.rs"] mod <module>;` at the bottom of the owning
module (precedents: `crates/cb-model/src/cbm.rs:65-66` → `mod tests;`,
`crates/cb-model/src/apply.rs:929-934` → `mod region_apply_test;`) or as
`crates/<crate>/tests/*_test.rs`; oracle bar ≤1e-5; CubeCL kernels
generics-float, if-as-statement only, grid-stride `CUBE_COUNT * CUBE_DIM`,
`f32::MIN` score sentinel.

**Mount-naming rule (mandatory, PLAN-CHECK MAJOR-9).** Every NEW test sibling
this plan creates MUST be mounted as `mod <file_stem>;` (not `mod tests;`), so
`cargo test -p <crate> --lib <parent_module>::<file_stem>` actually selects it.
Existing mounts are NOT renamed; commands against them use their real module
path. **Verified mount names** (`sed` of the line after each `#[path]`):

| file | mount | working filter |
|---|---|---|
| `cb-model/src/cbm.rs:65` | `mod tests;` | `cbm::tests` |
| `cb-model/src/fstr.rs:874` | `mod tests;` | `fstr::tests` |
| `cb-model/src/gpu_apply.rs:169` | `mod tests;` | `gpu_apply::tests` |
| `cb-model/src/partial_dependence.rs:325` | `mod tests;` | `partial_dependence::tests` |
| `cb-model/src/model_sum.rs:128` | `mod tests;` | `model_sum::tests` |
| `cb-model/src/apply.rs:929` | `mod region_apply_test;` | `apply::region_apply_test` |
| `cb-model/src/apply.rs:933` | `mod staged_predict_test;` | `apply::staged_predict_test` |
| `cb-model/src/export/onnx.rs:636` | `mod tests;` | `export::onnx::tests` |
| `cb-model/src/export/coreml.rs:307` | `mod tests;` | `export::coreml::tests` |
| `cb-train/src/tree.rs:92` | `mod general;` | `tree::general` |
| `cb-train/src/tree.rs:96` | `mod tie_break;` | `tree::tie_break` |
| `cb-train/src/boosting.rs:4954` | `mod tests;` | `boosting::tests` |
| `cb-train/src/boosting.rs:4958` | `mod boosting_device_fold_tests;` | `boosting::boosting_device_fold_tests` |
| `cb-train/src/candidates.rs:54` | `mod tests;` | `candidates::tests` |
| **`cb-oracle/src/lib.rs:33`** | `mod compare_test;` (CRATE ROOT) | `compare_test` |
| **`cb-oracle/src/lib.rs:35`** | `mod fixture_test;` (CRATE ROOT) | `fixture_test` |
| **`cb-oracle/src/lib.rs:37`** | `mod model_json_test;` (CRATE ROOT) | `model_json_test` |

**`cb-oracle` is the trap** (PLAN-CHECK pass-2 MAJOR-A): its three test siblings
are mounted at the **crate root** (`crates/cb-oracle/src/lib.rs:32-37`,
`#[cfg(test)] mod compare_test; / mod fixture_test; / mod model_json_test;`),
NOT under the module they test (`mod compare;` `:19`, `mod fixture;` `:21`,
`mod model_json;` `:22` are separate). So the filter is `model_json_test`,
**never** `model_json::tests`.

**Never** pass two positional filters to one `cargo test` invocation
(`--lib a --lib b` is malformed) — one invocation per filter.

### Device-test placement rule (mandatory, PLAN-CHECK pass-2 MAJOR-C)

`crates/cb-backend/tests/` is an **integration** directory: it sees only the
crate's `pub` surface. The symbols this plan's device Reds must exercise are NOT
public. Verified visibilities:

| symbol | declaration | reachable from `crates/cb-backend/tests/`? |
|---|---|---|
| `score_partition_over_binsums` | `gpu_runtime/mod.rs:2961` — bare `fn` | **NO** |
| `pack_cindex` | `gpu_runtime/cindex.rs:154` — `pub(crate) fn`, in `pub(crate) mod cindex` (`mod.rs:766`) | **NO** |
| `PackedCindex` / `::device_arrays` | `cindex.rs:81` / `:93` — `pub(crate)` type in a `pub(crate)` module | **NO** |
| `launch_partition_split_into` / `_packed_into` | `mod.rs:1916` / `:2014` — `pub(crate) fn` | **NO** |
| `GpuTrainSession` (the TYPE) | `session.rs:714` `pub struct`, module `mod session;` is private (`mod.rs:694`) but **re-exported** by `pub use session::*;` (`mod.rs:695`) | **yes** — `cb_backend::gpu_runtime::GpuTrainSession` resolves |
| the session's one-hot **observation point** (`one_hot_flags` / `real_folds` / `n_float` / derived `feature_lo`,`feature_hi`) | a `pub(crate)` accessor added by T27b | **NO** — which is why T27b's Red is a `gpu_runtime` sibling |
| `find_optimal_split_partition_kernel`, `partition_split_kernel` | `pub` in `pub mod kernels` (`lib.rs:19`) | yes — raw `::launch` only |
| `launch_find_optimal_split_pointwise` | `mod.rs:1295` — `pub` | yes, but routes to `find_optimal_split_kernel`, **not** the production partition scorer |
| `launch_apply_oblivious_f64` | `mod.rs:331` — `pub` | yes (what the one existing integration test uses) |

**Therefore every device Red in this plan lives as a `#[cfg(test)]` sibling
INSIDE `gpu_runtime`**, following the repo's own established pattern — plain
`#[cfg(test)] mod <name>;` in `crates/cb-backend/src/gpu_runtime/mod.rs` with
the file at `crates/cb-backend/src/gpu_runtime/<name>.rs`. Existing instances:
`mod ordered_test;` (`mod.rs:717`), `mod multiclass_test;` (`:725`),
`mod ranking_det_test;` (`:738`), `mod ranking_stoch_test;` (`:746`),
`mod session_residency;` (`:753`), `mod session_depth_gt1_test;` (`:760`).
A sibling at `gpu_runtime::<name>` reaches `super::score_partition_over_binsums`,
`super::launch_partition_split_packed_into`, `super::cindex::pack_cindex` and
`super::session::…` — all of which are visible to descendants of `gpu_runtime`.
Filter: `cargo test -p cb-backend --lib gpu_runtime::<name>`.

**No new file is added to `crates/cb-backend/tests/` by this plan**
(PLAN-CHECK pass-2 MINOR-b). That directory keeps exactly one file,
`apply_oblivious_launch_test.rs`, whose convention is: drive a `pub`
`cb_backend::gpu_runtime::*` entry point, run under the compile-time-selected
runtime (f64 on the default `cpu` backend), and assert against a host
reconstruction. Every `gpu_runtime` sibling this plan adds runs under that same
default `cpu` backend for its numeric assertions, with rocm-gated additions
where the existing siblings are (MEMORY `ginf01-gpu-inference-shipped`).

---

## 1. Resolutions to the three open questions

### Q1 — SPEC-OH-14: emit the real upstream one-hot json shape, or a typed error?

**Decision: typed error on BOTH directions (`save_json` and the json loader).**

Justification, from code actually read:

- `crates/cb-model/src/json.rs:474-485` builds `SplitJson { border,
  float_feature_index, split_index, split_type }` with `split_index` derived
  from `.enumerate()` over the **filtered** list. Emitting a real one-hot shape
  requires `SplitJson` to become an untagged/optional-field enum AND requires
  `split_index` to become the **global combined bin index** (`Float → OneHot →
  Ctr`, `cbm.rs:375-407`) rather than the current positional counter — because
  upstream's json `split_index` is the global bin index (research.md §4.2:
  cat 0 → `split_index 2`, cat 1 → `3`/`4`). That re-specifies the json
  split-index semantics for every float model too, putting SPEC-OH-31
  (float-only byte-identity) at risk for a non-goal.
- `from_doc` never reads `split_type` (the split-decode logic near
  `json.rs:648-663`; `decode_json` itself is `json.rs:813-816` — [C12]), so a
  real emit would round-trip **wrong** unless the decoder is rewritten too.
- SPEC.md §2 already lists "ONNX/CoreML *implementation* of one-hot branches
  (guards only)" as a non-goal; json one-hot is the same class of surface.
- The `.cbm` path (SPEC-OH-11/12/13) is the upstream-interop path and IS fully
  implemented; json is a secondary numeric export.

So: `save_json` returns `ModelError::Serialize("one-hot splits cannot be
represented in the numeric model.json schema (v1)")`, and the loader returns
`ModelError::Deserialize("one-hot json split unsupported (v1)")` **before**
serde reaches the misleading `missing field 'border'`.

**Scope boundary that PLAN-CHECK CRITICAL-1 forced into the open.** This
decision governs `cb_model`'s **model-representation** loader
(`crates/cb-model/src/json.rs`). It says nothing about
`crates/cb-oracle/src/model_json.rs`, a **separate, oracle-only** `model.json`
reader whose sole job is to hand a fixture's `float_feature_borders()` to a
comparison test. That reader MUST *tolerate* upstream one-hot documents —
otherwise SPEC-OH-28 / A11 cannot run at all. **T02b owns it**, and the two are
deliberately NOT unified: `cb-model` rejects (it would have to *represent* the
split), `cb-oracle` tolerates-and-skips (it only needs the float borders).

### Q2 — where does the fused one-hot level search live?

**Decision: extend `select_level_plain` (`tree.rs:931`) and
`select_level_perturbed` (`tree.rs:1059`) IN PLACE — return `AnySplit` instead
of `Split` — and extend `GrowScratch` (`tree.rs:697`) with the cat bin matrix,
cat histograms and cat bin width. No sibling grow function.**

1. The expensive machinery is `GrowScratch`: the feature-major bin matrix built
   once (`tree.rs:722-770`), the retained per-feature parent histograms, the
   subtraction trick (`derive_feature_level_hist`, `tree.rs:849`), and
   `advance_leaf_only` (`tree.rs:779-790`). A sibling would duplicate all of it
   **including the perturbed RNG draw contract** — the drift class that produced
   the `d7676b5` MVS bug.
2. The tie-break is a single strict-`>` first-wins over ONE flat candidate
   vector (`select_best_candidate`, `tree.rs:312-323`, **9 callers**, covered by
   `tree_tie_break_test.rs`). Splitting candidate generation across two
   functions makes the float-then-one-hot order an emergent property of the
   caller. `select_level_one_hot` (`tree.rs:3077-3130`) already establishes the
   required order (floats asc×border asc, THEN cat asc × bin asc) — preserve it
   verbatim, inside one function. **The argmax itself MUST remain
   `select_best_candidate`, generalized — never a second hand-rolled scan**
   (PLAN-CHECK MAJOR-6; see T18).
3. SPEC-OH-31 byte-identity is provable **by construction on the CPU path**:
   `FeatureMatrix::new` hard-codes `cat_bins: &[]` (`tree.rs:345-351`), so the
   added `0..n_cat` range is empty, no cat histogram is built, the candidate
   vector is unchanged, and `AnySplit::Float(s)` carries exactly today's `s`.
   The same fact covers the RSM draw loop (`tree.rs:610-614` iterates
   `matrix.n_features()`, float-only). **It does NOT transfer to the device** —
   see [C11] and T29b.
4. `greedy_tensor_search_oblivious_with_ctr` (`tree.rs:2919-3009`) is NOT the
   template for the *search* (per-candidate rescans, forbidden by SPEC-OH-06 /
   §9 R4). It IS the template for the **split-back** step (`tree.rs:2955-2997`).

`grow_one_hot_tree` (`tree.rs:3141`) stays as the frozen correctness reference
and keeps its `lib.rs:107` re-export.

### Q3 — must SPEC-OH-01..03 (the order fix) land strictly first?

**Decision: yes — strictly first among production-code tasks (T03, T04), after
the four evidence/enabler Wave-0 tasks (T00, T01a, T02, T02b), none of which
touch production Rust in `cb-train`/`cb-model`/`cb-backend`.**

1. **Write conflict.** `Model::from_trained` (`model.rs:326-426`, **24
   callers**) is rewritten by T04 and again by T21; `cb_train::ObliviousTree`
   (`boosting.rs:775-794`) by T03 and read by T19/T21.
2. **Evidence isolation.** The order fix is the ONLY change altering emitted
   bytes on an already-shipping path (mixed-kind CTR trees).
3. **Correctness dependency.** `leaf_index_for` (`apply.rs:208-215`) walks
   *stored* order, so a one-hot-at-level-0 / float-at-level-1 tree mis-predicts
   on day one under kind-grouped order.

**Regression evidence:** `tensor_ctr_e2e_oracle_test` stays 3/3; the four other
CTR oracles + `structure_fold_cycle_oracle_test`; full `cb-model` and
`cb-train`; and the new SPEC-OH-03 test must FAIL before (leaf 1↔2, `20.0` vs
`30.0`) and PASS after.

---

## 2. Corrections to SPEC.md / research.md anchors (verified)

C1–C10 were established in revision 1; **PLAN-CHECK independently confirmed C1,
C2, C3, C4, C5, C6 and C8**. C11–C15 are new.

| # | Claim | Verified reality | Impact |
|---|---|---|---|
| C1 | SPEC-OH-22 names `find_optimal_split_kernel` as *the* fold | **CONFIRMED.** Production is `find_optimal_split_partition_kernel` (`cb-backend/src/kernels.rs:4506`) ← `score_partition_over_binsums` (`gpu_runtime/mod.rs:2961`; launches `:3047`/`:3069`) ← `grow_oblivious_tree_resident` (`mod.rs:3803`, call `:3930-3932`) ← 2 callers in `session.rs`. `find_optimal_split_kernel` (`kernels.rs:3367`) is reached only via `score_over_binsums` (`mod.rs:1375`) ← `launch_find_optimal_split_pointwise_into` (`mod.rs:1319`). | T25 patches BOTH. SPEC amendment §10-A. |
| C2 | — | `kernels.rs:4524` `max_border = n_bins_used - 1`; `:4596-4604` `if border < max_border`. **CONFIRMED**, *and* a second **HOST BELT** at `gpu_runtime/mod.rs:3108-3112`: `if (cand as usize) % n_bins >= n_bins_used - 1 { continue; }`, doc'd as "the device kernel already excludes them; host belt, WR-05". | A one-hot equality candidate on the last bin is legitimate. **T25 lifts BOTH** (CRITICAL-3). |
| C3 | SPEC §4 types `value_hash: i32` | `calc_cat_feature_hash` returns **`u32`** (`cb-data/src/cat_hash.rs:362-365`); `TOneHotFeature.Values` is `[i32]`. | Bit-preserving `as i32`. |
| C4 | research §9 "`cbm.rs` … emit" | `TModelTreesArgs` (`cbm.rs:750-766`) emits `FloatFeatures` + `CtrFeatures` only — **no `CatFeatures`, no `OneHotFeatures`**; `cbm.rs:524`'s "v1 has no one-hot bins" is stale. | T07 adds both. |
| C5 | — | `BinKind::OneHot` is a **unit** variant (pushed `cbm.rs:389`, matched `:1041`). | T05 widens to `{ cat_feature, value_hash }`. |
| C6 | SPEC-OH-15 "typed error" | `shap_values` is **infallible** (`shap.rs:534`); so are `shap_interaction_values` (`:941`), **`prediction_diff` (`:1137`)**, **`sage_values` (`:1236`)** and `fstr::loss_function_change` (`fstr.rs:788`). `float_splits_of` (`shap.rs:550`) has TWO call sites: `:830` and **`:1139`**. Production callers: `fstr.rs:804`, `fstr.rs:846`, `catboost-rs/src/model.rs:292`, `:393` — **4**. | T10 makes `float_splits_of` fallible and cascades all four public surfaces; **the revision-1 escape hatch is DELETED** (MAJOR-1). SPEC amendment §10-B. |
| C7 | research §2.2 "`*_with_one_hot` sibling" | `select_level_plain`/`_perturbed` return `Split`; `advance_leaf_only` takes `&Split` (`tree.rs:779`). | Q2 supersedes. |
| C8 | research §2.1 `perfect_hash_bins` | **CONFIRMED.** `cat_hash.rs:471-479` builds a LOCAL `PerfectHash`, returns bins only; `remap_bounded` (`:444-459`) assigns `bin = map.len()` first-seen. | T17's zip-inverse is exact; T21's independent-hash assertion is the right oracle. |
| C9 | SPEC §9 R11 | `cargo test -p cb-train --features rocm --no-run` builds; `--no-default-features --features rocm` does not. | Additive form everywhere. |
| C10 | SPEC-OH-27 | Instrumented sources in-repo with `CB_INSTRUMENT_LOG`: `catboost-master/catboost/private/libs/algo/{train.cpp:6, greedy_tensor_search.cpp:4, yetirank_helpers.cpp:8}`. | T01a reuses the `bayesian-rng-draw-accounting` recipe. |
| **C11** | Rev-1 claimed OH-31 byte-identity is provable by construction | True for the **CPU** search only. The device builds a **single concatenated feature axis** (`quantize_feature_major` `boosting.rs:2196-2233` + `pack_cindex`). Also: `GpuTrainSession::begin` declines on `n == 0 \|\| n_features == 0 \|\| n_bins == 0` (`session.rs:1250-1253`), and pads via `pad_hist_line_bins(n_bins)` (`session.rs:1270-1274`) to a `{32,64,128,256}` family (rejection `gpu_runtime/mod.rs:2392-2403`). | **New T29b** = device byte-identity gate; T23/T24 name the padding site and assert a cat-only pool reaches the fill legally (C11 / MINOR-7). |
| **C12** | Rev-1 cited `decode_json` at `json.rs:648-663` | `decode_json` is `json.rs:813-816` (`serde_json::from_str::<ModelJsonDoc>(contents)?` → `from_doc`); the split-decode logic sits in `from_doc` near `:648-663`. A pre-check cannot be "a cheap targeted probe" — it is a full `serde_json::Value` pre-parse, or `#[serde(default)] border` + a `split_type` check inside `from_doc`. | T09 corrected (MINOR-3, MINOR-6). |
| **C13** | Rev-1 blocker B7 | `crates/cb-model/src/export/onnx_test.rs` **and** `coreml_test.rs` both exist and are mounted (`onnx.rs:636-637`, `coreml.rs:307-308`, both `mod tests;`). | T11's Reds live there; **B7 deleted** (MINOR-2). |
| **C14** | Rev-1 T11 named only the two `cb-model` guards | `catboost-rs-py/src/errors.rs:148-155` matches `OnnxExportError` **exhaustively** (4 or-patterned unsupported variants, `Io`, `Encode`, no `_`); `:164-171` the same for `CoreMlExportError`. Adding a variant is a hard `E0004`. **Verified safe by contrast:** `PdpError` reaches the facade via `#[from]` (`catboost-rs/src/error.rs:77`) with no exhaustive match, and `GpuApplyUnsupported` has none either. | T11 owns `errors.rs` and builds `catboost-rs-py`; T12/T14 recorded as safe (MAJOR-2). |
| **C15** | — | `from_trained` also builds `region_trees` with `RegionLevel { one_hot }` (`model.rs:399-408`) — an **unrelated** flag (the Region walk's equality marker, always `false` for the CPU float grower). | T04/T21 must not conflate it with `ModelSplit::OneHot`. |
| **C16** | PLAN-CHECK pass-1/2 (and revisions 2–3 of this plan) prescribed `TCFeature.folds` as the per-feature real-bin bound | **`TCFeature.folds` is the PADDED UNIFORM LINE WIDTH on the production path, never the cardinality.** Three links read in situ: (a) `crates/cb-backend/src/gpu_runtime/session.rs:1363` — `let n_buckets_per_feature = vec![n_bins_line; eff_n_features];` passed to `pack_cindex` at `:1364`, with its own comment "the resident fill / scorer address cells by the SAME padded width"; (b) `crates/cb-backend/src/gpu_runtime/cindex.rs:213-227` — `for (&nb, …) in n_buckets.iter().zip(…) { … let folds = u32::try_from(nb)?; … }`, i.e. `folds` is copied straight from that argument; (c) therefore `folds[f] == n_bins_line` for EVERY feature. **And the obvious repair is forbidden:** `pack_cindex`'s placement pass (`cindex.rs:181-200`) computes `let bits = feature_bits(nb)?;` per feature and derives `(group, shift, mask)` from it, so passing true cardinalities in `n_buckets_per_feature` would change the packed words for **every** pool including float-only — breaking T29b fn 1's frozen `packed_cindex.json`, the `kernels/cindex.rs` pack→read oracle, and the `read_bin` descriptors `launch_partition_split_packed_into` consumes. | `border < folds[feature]` would reduce to `border < n_bins_line` — the loop bound itself, i.e. **no bound at all**, leaving pass-1 MAJOR-3 (phantom padded one-hot candidates) unfixed. **A SEPARATE `real_folds` array is plumbed T24 → T27b → T25** (PLAN-CHECK pass-3 MAJOR-1); `n_buckets_per_feature` is NOT touched. |

---

## 3. Execution waves and dependency order

```text
WAVE 0 (evidence + enablers; T01a/T02 touch NO Rust; T00 adds ONE #[cfg(test)]
        sibling + mount in cb-backend; T02b is cb-oracle-only — all disjoint)
  T00   OH-31  float-only .cbm byte-identity baseline @ plan-base SHA
               + the DEVICE artifacts (packed cindex / scorer winners /
                 device_baseline.cbm) via a #[ignore]d gpu_runtime capture test
               + the accepted workspace test-failure baseline transcript
  T01a  OH-27  upstream RNG draw ground truth (evidence only)
  T02   OH-29  --one-hot-only fixture flag + committed one_hot_train/ fixture
  T02b  OH-28  cb-oracle model_json::SplitJson tolerates one-hot documents

WAVE 1 (STRICTLY FIRST for production code; serial)
  T03   OH-01  ObliviousTree carries ordered level kinds
  T04   OH-02/03  from_trained emits level order (pinned by CTR-at-level-0 test)

WAVE 2 (serial; the variant that breaks the build)
  T05   OH-08  ModelSplit::OneHot + conservative arms everywhere

WAVE 3 (parallel after T05 — disjoint files)
  T06 OH-10 apply.rs | T07 OH-11 cbm.rs | T09 OH-14 json.rs
  T10 OH-15 shap.rs(+fstr.rs) | T11 OH-16 export/*+catboost-rs-py
  T12 OH-17 gpu_apply.rs | T13 OH-18 fstr.rs | T14 OH-19 partial_dependence.rs
     (T10 and T13 are SERIAL against each other — both write fstr.rs)
  WAVE 3b (serial tails)
  T08 OH-12 cbm.rs load          [after T07]
  T15 OH-13 upstream .cbm oracle [after T08, T02, T02b]

WAVE 4 (trainer; serial — boosting.rs / tree.rs)
  T16  OH-04  train_inner partitions one-hot-routed columns   [after T03]
  T01b OH-27  consume the draw rule OR raise the typed rejection [after T16]
  T17  OH-05  one-hot columns reach the candidate matrix + bin->hash table
  T18  OH-06  fused one-hot-aware level search
  T19  OH-07  persist one-hot splits + LevelKind::OneHot
  T20  OH-26  one-hot x CTR typed gate
  T21  OH-09  bin -> raw-hash mapping at the lift   [after T19, T04, T05]
  T22  OH-28  production one-hot oracle             [after T21, T02, T02b]

WAVE 5 (device; serial)
  T23  OH-20  cat-only pool is device-eligible      [after T16]
  T24  OH-21  device quantization: one-hot bin columns + one_hot + folds arrays
  T25  OH-22  split-scoring fold one-hot arm (both scorers AND the host belt)
  T26  OH-23  partition_split_kernel equality arm
  T27  OH-24  DeviceGrownTree carries the split kind
  T27b OH-24/25  device session wiring (DeviceTrainConfig + begin + n_float)
  T28  OH-25  device-vs-CPU one-hot parity (both anti-false-pass guards)

WAVE 6 (gates)
  T29  OH-31  CPU/model float-only byte-identity gate (consumes T00)
  T29b OH-31  DEVICE float-only byte-identity gate
  T30  OH-30  Colab T4 speed gate + runner port
```

**Serialization owners** (extended per PLAN-CHECK MAJOR-4 / MAJOR-2). A task may
NOT run in parallel with any other task naming the same file.

| file | tasks |
|---|---|
| `crates/cb-train/src/boosting.rs` | T03, T16, **T01b**, T17, T19, T20, T23, T24 — plus **T21 as a READ-ONLY consumer** of the field T17 adds (T21 makes no `boosting.rs` edit) |
| `crates/cb-train/src/tree.rs` | T03, T17, T18, T19 |
| `crates/cb-model/src/model.rs` | T04, T05, T21 |
| `crates/cb-model/src/cbm.rs` | T05, T07, T08 |
| `crates/cb-model/src/shap.rs` | T10 |
| `crates/cb-model/src/fstr.rs` | T05, **T10 (cascade)**, T13 — T10 ↔ T13 SERIAL |
| `crates/cb-model/src/export/{onnx,coreml}.rs` | T11 |
| `crates/catboost-rs-py/src/errors.rs` | **T11** (E0004, [C14]) |
| `crates/catboost-rs/src/model.rs` | T10 (`:292`, `:393`) |
| `crates/cb-oracle/src/model_json.rs` | **T02b** |
| `crates/cb-backend/src/kernels.rs` | T25, T26 |
| `crates/cb-backend/src/gpu_runtime/mod.rs` | **T00 (test-mount only)**, T24, T25, T26, T27, **T27b**, **T29b** |
| `crates/cb-backend/src/gpu_runtime/cindex.rs` | **T24** |
| `crates/cb-backend/src/gpu_runtime/session.rs` | **T24, T27, T27b** |
| `crates/cb-backend/src/gpu_runtime/pairwise.rs` | **T26** (call-site sweep) |
| `crates/cb-backend/src/kernels/pointwise_hist.rs` | **T24** (sweep) |
| `crates/cb-backend/src/kernels/cindex.rs` | **T24** (sweep) |
| `crates/cb-backend/src/kernels/grow_loop.rs` | **T26** (sweep) |
| `crates/cb-backend/src/kernels/region_device.rs` | **T27** (sweep, OUT-OF-SCOPE grower) |
| `crates/cb-backend/src/kernels/nonsym_grow.rs` | **T27** (sweep, OUT-OF-SCOPE grower) |
| `crates/cb-compute/src/runtime.rs` | **T27, T27b** |
| `crates/cb-backend/src/gpu_backend.rs` | **T27b** |

**Blast-radius warning for the device wave (PLAN-CHECK MAJOR-4).**
`pack_cindex` (`cindex.rs:154`) has **13 callers**; `device_arrays`
(`cindex.rs:93`) **4**; `launch_partition_split_into` (`mod.rs:1916`) **5**;
`DeviceGrownTree` (`runtime.rs:931`) **17**. Several live in
`kernels/region_device.rs` and `kernels/nonsym_grow.rs`, which SPEC §2 puts OUT
OF SCOPE. **Every non-oblivious call site MUST pass the byte-unchanged default**
(`one_hot = false`, an all-false flags slice, `folds` = today's per-feature
bucket counts, `.2 == false` on the widened tuple), and T24/T26/T27 each
validate with `device_region_fit_test` AND `device_nonsym_fit_test`.

---

## 4. SPEC-ID → task coverage

| SPEC | Task(s) | SPEC | Task(s) |
|---|---|---|---|
| SPEC-OH-01 | T03 | SPEC-OH-17 | T12 |
| SPEC-OH-02 | T04 | SPEC-OH-18 | T13 |
| SPEC-OH-03 | T04 (the Red) | SPEC-OH-19 | T14 |
| SPEC-OH-04 | T16 | SPEC-OH-20 | T23 |
| SPEC-OH-05 | T17 | SPEC-OH-21 | T24 |
| SPEC-OH-06 | T18 | SPEC-OH-22 | T25 |
| SPEC-OH-07 | T19 | SPEC-OH-23 | T26 |
| SPEC-OH-08 | T05 | SPEC-OH-24 | T27, **T27b** |
| SPEC-OH-09 | T21 | SPEC-OH-25 | **T27b**, T28 |
| SPEC-OH-10 | T06 | SPEC-OH-26 | T20 |
| SPEC-OH-11 | T07 | SPEC-OH-27 | **T01a**, **T01b** |
| SPEC-OH-12 | T08 | SPEC-OH-28 | **T02b**, T22 |
| SPEC-OH-13 | T15 | SPEC-OH-29 | T02 |
| SPEC-OH-14 | T09 | SPEC-OH-30 | T30 |
| SPEC-OH-15 | T10 | SPEC-OH-31 | **T00**, T29, **T29b** |
| SPEC-OH-16 | T11 | | |

**35 tasks (T00, T01a, T01b, T02, T02b, T03–T30, T27b, T29b); all 31
specifications covered; every task references ≥1 spec.**
Acceptance mapping: A1→T19/T22, A2→T00/T29/T29b, A3→T04, A4→T07, A5→T08/T15,
A6→T21, A7→T09–T14, A8→T13, A9→T23, A10→T27b/T28, A11→T02b/T22, A12→T30,
A13→T20.

---

## 5. Tasks

### T00 — SPEC-OH-31 — capture the float-only byte-identity baseline at the plan-base SHA

**Goal.** Freeze, BEFORE any production change, (a) a float-only fit's `.cbm`
bytes and (b) the accepted workspace test-failure baseline, so SPEC-OH-31 /
A2 is proven against pre-change code rather than degenerating into a
self-comparison. *(PLAN-CHECK MAJOR-7 + "Unverified Items: workspace
baseline".)*

**Observable completion.** `crates/cb-oracle/fixtures/float_only_byte_identity/`
contains `baseline.cbm`, `inputs/`, a **`device/`** subdirectory
(`packed_cindex.json`, `scorer_winners.json`, `device_baseline.cbm`) and a
`README.md` recording the exact `git rev-parse HEAD` of the plan base; and
`.planning/plans/one-hot-categorical-training/baseline/workspace-test-baseline.txt`
records a full `cargo test --workspace` transcript with the known-failing
targets enumerated.

**Blocking / prerequisites.** None. **MUST run before T03.**

**Verified files and symbols.**
- `crates/cb-model/src/cbm.rs:750-766` — `TModelTreesArgs`; `CatFeatures` /
  `OneHotFeatures` will be `None` for a float-only model after T07, which is
  why the bytes are expected to be stable.
- `crates/cb-train/src/tree.rs:345-351` — `FeatureMatrix::new` hard-codes
  `cat_bins: &[]` (the CPU-side by-construction argument, [C11]).
- MEMORY `catboost-rs-preexisting-test-failures` records pre-existing failures
  in `cb-backend` (CubeCL MLIR), `cb-train` (monotone) and `catboost-rs-py`
  (python3.14 link). Without an enumerated baseline, T29's "full green
  `cargo test --workspace`" is unachievable as written.

**Red (SCAFFOLDING — a missing-artifact failure, not behavioral evidence).**
- File: `crates/cb-model/tests/float_only_byte_identity_test.rs` (new).
- Test fn: `float_only_cbm_bytes_match_the_frozen_plan_base_baseline`.
- Setup: load `crates/cb-oracle/fixtures/float_only_byte_identity/inputs/`,
  run the pinned float-only fit, `save_cbm` to a temp path, byte-compare against
  `baseline.cbm`.
- Expected initial failure: `No such file or directory` on `baseline.cbm`.

**Green (minimal).**
1. `git rev-parse HEAD` → record as `PLAN_BASE_SHA` in the fixture `README.md`.
2. Generate the inputs deterministically (pinned seed, `thread_count = 1`,
   `random_strength = 0`, `boost_from_average = false`, `bootstrap_type = No`,
   `iterations = 3`, `depth = 3`) and run the fit + `save_cbm` on the **current
   (pre-T03) tree**; commit `baseline.cbm` + inputs.
2b. **Capture the DEVICE artifacts (PLAN-CHECK pass-2 MAJOR-C).** The float-only
   device baseline cannot be produced through any public API — `pack_cindex`,
   `PackedCindex::device_arrays` and `score_partition_over_binsums` are
   `pub(crate)` / private (see the header's device-test placement rule). So T00
   **creates** the `src` sibling
   `crates/cb-backend/src/gpu_runtime/device_float_only_identity_test.rs`
   (mounted in `gpu_runtime/mod.rs` as
   `#[cfg(test)] mod device_float_only_identity_test;`) containing a single
   capture fn:
   ```rust
   #[test]
   #[ignore = "capture-only: run once at the plan-base SHA to freeze the fixture"]
   fn capture_float_only_device_artifacts() { … }
   ```
   which writes `device/packed_cindex.json` (the `(words, offsets, shifts,
   masks)` tuple) and `device/scorer_winners.json` (the per-level
   `(best_idx, best_gain)` pairs). Run it ONCE, on the pre-T03 tree:
   `cargo test -p cb-backend --lib gpu_runtime::device_float_only_identity_test -- --ignored`.
   Also produce `device/device_baseline.cbm` from a device-grown float-only fit
   through the public `train` + `save_cbm` API. Commit all three.
   **T29b later fills in the two assertion fns in the same file**; T00 leaves
   only the capture fn.
3. Run `cargo test --workspace 2>&1 | tee` into
   `baseline/workspace-test-baseline.txt`; in a sibling
   `baseline/README.md`, enumerate every failing target as the ACCEPTED
   baseline, with its failure reason.
4. Commit both, with the `git rev-parse HEAD` transcript in the commit body.

**Refactor.** The fixture is FROZEN: no later task may regenerate it. Add that
sentence to the fixture `README.md`. Regression scope: none (new files only).

**Validation.**
```
git rev-parse HEAD
cargo test -p cb-model --test float_only_byte_identity_test
cargo test -p cb-backend --lib gpu_runtime::device_float_only_identity_test -- --ignored
cargo test -p cb-backend --lib gpu_runtime::device_float_only_identity_test   # capture fn is #[ignore]d: 0 run, expected
git status --short crates/cb-oracle/fixtures   # ONLY float_only_byte_identity/
cargo build --workspace --all-targets          # the new sibling mount must compile
```

**Completion evidence.** The recorded SHA, the passing model test on the
pre-change tree, the three committed `device/` artifacts, and the enumerated
accepted-failure list.

**Parallelization.** Fully parallel with T01a, T02, T02b (disjoint files).
**Note on the "no production Rust" wave label:** T00 adds a `#[cfg(test)]`
sibling + its mount line to `crates/cb-backend/src/gpu_runtime/mod.rs`. That is
**test-only** Rust (no production behavior), and T00 runs strictly before the
device wave, so it cannot conflict with T24–T27b — but T00 is nonetheless listed
in §3's `gpu_runtime/mod.rs` serialization row so the ownership record is
complete.

---

### T01a — SPEC-OH-27 — upstream RNG draw-order ground truth (evidence only)

**Goal.** Establish, from a real instrumented CatBoost 1.2.10 run, whether a
one-hot-routed categorical column changes the per-level RNG draw count.
**This is the plan's only genuine blocker** (SPEC §9 R5; same defect class as
the fixed `d7676b5` MVS bug). *Split from revision-1's T01 per PLAN-CHECK
CRITICAL-4: this half produces evidence and touches NO production Rust; the
enforcement half is T01b, in Wave 4 after T16.*

**Observable completion.** EITHER
`.planning/plans/one-hot-categorical-training/instrumented-ground-truth/{one_hot.jsonl,
one_hot_no.jsonl, ONE_HOT_GROUND_TRUTH.md}` are committed and state the derived
draw rule, OR `ONE_HOT_GROUND_TRUTH.md` exists and records **`STATUS:
NOT-ESTABLISHED`** plus the reason — which is itself the input T01b consumes.

**Blocking / prerequisites.** None. **Blocks T01b and T18's perturbed arm.**

**Verified files and symbols.**
- `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp` — already
  instrumented (`CB_INSTRUMENT_LOG`, 4 hits); `train.cpp` 6 hits;
  `yetirank_helpers.cpp` 8 hits [C10].
- `.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/GROUND_TRUTH.md`
  — the build recipe (clang 22 + GNU `ld.bfd`, OpenSSL `no_fips`,
  `--start-group`/`--end-group`) and the reconstructed per-tree model
  `PRE(2) + Bootstrap(type) + Σ_levels[RSM(n_features) + CalcScores(1) +
  SelectBestCandidate(variable)] + Leaf(2)`.
- `crates/cb-train/src/tree.rs:610-614` — the current per-level RSM loop
  `for _ in 0..matrix.n_features() { p.rng.gen_rand_real1(); }`.
  **`n_features()` is FLOAT-only** (`tree.rs:355-357`); `n_cat_features()` is
  separate (`:361-363`). The open question is whether upstream's
  `SelectFeaturesForScoring` counts cat features.
- `crates/cb-train/src/tree.rs:1095-…` — `select_level_perturbed`'s
  `SelectBestCandidate` `std_normal` pass, one draw per LISTED feature.

**Red (SCAFFOLDING — a missing-artifact failure).**
- File: `crates/cb-train/tests/one_hot_draw_accounting_test.rs` (new).
- Test fn: `one_hot_ground_truth_artifact_is_present_and_states_a_verdict`.
- Setup: `std::fs::read_to_string` on a `CARGO_MANIFEST_DIR`-relative path to
  `ONE_HOT_GROUND_TRUTH.md`.
- Expected output: the file exists and contains either
  `RSM_RULE: n_float + n_one_hot`, `RSM_RULE: n_float`, or
  `STATUS: NOT-ESTABLISHED`.
- Expected initial failure: `No such file or directory`.
- *(The behavioral draw-count assertion lives in T01b, where a production
  consumer exists.)*

**Green (minimal).**
1. Rebuild the instrumented upstream 1.2.10 CLI following the
   `bayesian-rng-draw-accounting` recipe verbatim (**no new C++**).
2. Export the T02 fixture dataset to TSV with a `cat_features` declaration and
   run `catboost fit` under `CB_INSTRUMENT_LOG` for at minimum
   `bootstrap_type ∈ {No, Bayesian}` (Bayesian is the phase-sensitive one per
   the prior GROUND_TRUTH's "Why this resolves…" section).
3. Commit the jsonl traces and `ONE_HOT_GROUND_TRUTH.md` with the byte-exact
   per-tree call-count reconciliation and one of the three verdict lines above.
4. **If the build cannot be produced**, commit `ONE_HOT_GROUND_TRUTH.md` with
   `STATUS: NOT-ESTABLISHED` and the exact failure (missing toolchain, link
   error, …). **Do NOT guess a draw count.**

**Refactor.** None (evidence artifacts only). Regression scope: none — no Rust
production file is touched.

**Validation.**
```
cargo test -p cb-train --test one_hot_draw_accounting_test
```

**Completion evidence.** The committed traces + verdict line, or the
`NOT-ESTABLISHED` record with its reason.

**Parallelization.** Fully parallel with T00, T02, T02b.

---

### T02 — SPEC-OH-29 — `--one-hot-only` fixture generation, frozen and isolated

**Goal.** A committed upstream catboost 1.2.10 one-hot fixture family
`crates/cb-oracle/fixtures/one_hot_train/`, generated once, deterministically,
without touching any other committed fixture.

**Observable completion.** `git status --short crates/cb-oracle/fixtures` shows
ONLY files under `one_hot_train/`, and that directory contains a `.cbm`,
`model.json`, `preds.npy`, `X_float.npy`, `cat_cols.json`, `y.npy` and
`config.json` per scenario.

**Blocking / prerequisites.** None. Blocks T01a's dataset, T15, T22.

**Verified files and symbols.**
- `crates/cb-oracle/generator/gen_fixtures.py:3449-3492` — the `if/elif` chain
  over `sys.argv` with the terminal `else: main()`; 8 existing `--*-only` flags.
- `crates/cb-oracle/generator/gen_fixtures.py:863-905` — `gen_bootstrap_dev()`:
  the exact pattern to mirror (docstring stating why a new family exists and
  that it is reachable ONLY through the flag; reuse frozen inputs;
  `shared = {k: v for k, v in ISOLATING_PARAMS.items() …}`).
- `crates/cb-oracle/fixtures/one_hot_cat/` — contains NO model; a frozen Wave-0
  anchor. **Do not extend it.**
- `crates/cb-oracle/fixtures/ctr_load/` — the precedent for a model-bearing
  family (`simple.cbm`, `*_preds.npy`, `cat_cols.json`, `X_float.npy`, `y.npy`,
  `config.json`).
- `.venv` — catboost 1.2.10, pytest 9.1.1.

**Red (SCAFFOLDING).**
- File: `crates/cb-train/tests/one_hot_oracle_test.rs` (existing; the local
  driver is removed later, in T22).
- Test fn: `one_hot_train_fixture_is_present_and_wellformed`.
- Expected output: the fixture files exist, `config.json` parses, `preds.npy`
  has `n_rows` entries.
- Expected initial failure: directory missing.

**Green (minimal).**
1. `gen_one_hot()` with a docstring stating: a NEW family (never
   `one_hot_cat/`), reachable ONLY through `--one-hot-only`, never `main()`.
2. Scenarios:
   - `default_binary/` — 3 float columns + 1 **binary** cat column, DEFAULT
     `one_hot_max_size = 2` (SPEC §6 A1, the headline silent-drop case);
   - `multi/` — 1 float column (2 borders) + cat columns of cardinality 2 and 3,
     `one_hot_max_size = 5`, `max_ctr_complexity = 0` — matching the research.md
     §4.1 probe that pinned the encoding, so SPEC-OH-11's acceptance offsets
     (float `0,1`; cat 0 `2`; cat 1 `3,4`) are directly checkable.
   - **Ordering-discriminating requirement (PLAN-CHECK "Unverified Items"):**
     at least one cat feature's **first-referenced** value order MUST differ
     from its **ascending-hash** order, otherwise T07's `Values`-ordering pin
     (blocker B3) is vacuous. Verify this in the generator by computing
     `calc_cat_feature_hash`-equivalent values in Python and **asserting the two
     orders differ**; if they do not, perturb the category labels until they do,
     and record the chosen labels in `config.json`.
3. Params: `shared = {**ISOLATING_PARAMS, "iterations": 3, "depth": 3,
   "boost_from_average": False, "thread_count": 1, "random_seed": 0}` with
   `one_hot_max_size` per scenario, and **`random_strength = 0` pinned
   explicitly** (MEMORY `cv-orch01-random-strength-fixture`). Assert
   `model.get_all_params()` reads every pinned value back.
4. Save `.cbm`, `model.json`, `preds.npy` (raw formula values) and the inputs.
5. `gen_one_hot_only()` wrapper.
6. `elif "--one-hot-only" in sys.argv: gen_one_hot_only()` **immediately before**
   the final `else: main()`.
7. **Abort guard:** at the top of `gen_one_hot_only()` run
   `git status --short crates/cb-oracle/fixtures` via `subprocess`; after
   generation re-run it and `sys.exit(1)` loudly if any path outside
   `one_hot_train/` appears.

**Refactor.** No other `gen_*` may be called from `gen_one_hot()`; do NOT modify
`ISOLATING_PARAMS` (shared by every family). Regression scope: the abort guard
IS the regression scope.

**Validation.**
```
.venv/bin/python crates/cb-oracle/generator/gen_fixtures.py --one-hot-only
git status --short crates/cb-oracle/fixtures     # ONLY one_hot_train/ paths
cargo test -p cb-train --test one_hot_oracle_test
```

**Completion evidence.** The `git status` output in the commit body; the
first-referenced-vs-ascending-hash divergence printed by the generator.

**Parallelization.** Fully parallel with T00, T01a, T02b.

---

### T02b — SPEC-OH-28 (enabler) — `cb_oracle::model_json` tolerates upstream one-hot documents

**Goal.** Unblock the headline production oracle. The oracle-only `model.json`
reader must parse an upstream one-hot document and still yield
`float_feature_borders()`. *(PLAN-CHECK CRITICAL-1 — without this, T22 / A11
cannot run at all, and the defect would surface only after ~20 tasks.)*

**Observable completion.** `load_model_json` on
`crates/cb-oracle/fixtures/one_hot_train/<scenario>/model.json` succeeds, and
`float_feature_borders()` returns exactly the float features' borders (one-hot
splits contribute nothing).

**Blocking / prerequisites.** T02 (needs the fixture). **Blocks T15 and T22.**

**Verified files and symbols.**
- `crates/cb-oracle/src/model_json.rs:24-42` — verbatim:
  ```rust
  pub struct SplitJson {
      pub border: f64,                     // NO #[serde(default)]
      #[serde(default)] pub float_feature_index: i64,
      pub split_index: i64,
      pub split_type: String,              // "FloatFeature" | "OnlineCtr"
  }
  ```
  The doc comment already explains that `float_feature_index` was defaulted so
  `OnlineCtr` splits would parse — the identical argument now applies to
  `border` for `OneHotFeature` splits.
- Upstream's one-hot split object (research.md §4.2):
  `{"split_index":2,"cat_feature_index":0,"value":-1438285038,"split_type":"OneHotFeature"}`
  — **no `border` key**.
- `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs:215-219` — the established
  pattern SPEC-OH-28 mandates: `let model_json = load_model_json(&fixture(
  ".../model.json"))…; let borders = model_json.float_feature_borders();`
- `crates/cb-oracle/src/model_json_test.rs` — the existing test sibling (the Red
  home).
- **Scope note:** this is NOT the `cb-model` json loader (T09), which
  deliberately REJECTS one-hot (Q1). The two readers have different jobs.

**Red.**
- File: `crates/cb-oracle/src/model_json_test.rs` (existing; mounted at the
  **crate root**, `crates/cb-oracle/src/lib.rs:37` `mod model_json_test;` —
  so the test path is `model_json_test::<fn>`, **not** `model_json::tests`).
- Test fn: `upstream_one_hot_model_json_parses_and_yields_float_borders_only`.
- Setup: an inline minimal upstream document with one float split
  (`border`, `float_feature_index`, `split_type: "FloatFeature"`) and one
  one-hot split (`cat_feature_index`, `value`, `split_type: "OneHotFeature"`,
  **no `border`**), plus a `features_info.float_features` block.
- Expected output: `Ok(_)`, `doc.float_feature_borders()` equals the float
  borders, and `split.is_one_hot() == true` for the second split.
- Expected initial failure: `missing field 'border' at line … column …` — the
  exact defect CRITICAL-1 predicts for T22.

**Green (minimal).**
1. `#[serde(default)] pub border: f64` with a doc comment mirroring the existing
   `float_feature_index` rationale and naming `OneHotFeature`.
2. `#[serde(default)] pub cat_feature_index: Option<i64>` and
   `#[serde(default)] pub value: Option<i64>` (upstream's one-hot payload).
3. `impl SplitJson { #[must_use] pub fn is_one_hot(&self) -> bool {
   self.split_type == "OneHotFeature" } }`.
4. **Verified sufficient:** `ModelJson::float_feature_borders()`
   (`crates/cb-oracle/src/model_json.rs:461-467`) reads
   `self.features_info.float_features[..].borders` — **not** the splits — so
   `#[serde(default)] border` alone unblocks the parse. `FeaturesInfoJson
   .float_features` (`:230-234`), `FloatFeatureJson.borders` (`:219-226`) and
   `ModelJson.oblivious_trees` (`:329-330`) already carry `#[serde(default)]`,
   and `split_index` / `split_type` ARE present on upstream one-hot splits, so
   nothing else blocks it. Re-read the function before editing to confirm this
   is still true; if it ever derives borders from the splits, filter to
   `split_type == "FloatFeature"` so a one-hot split cannot inject a phantom
   `0.0` border.

**Refactor.** Do not add a `OneHotSplitJson` type — the oracle reader only needs
tolerate-and-skip. Regression scope: **every oracle test that parses
`model.json`** — CTR (`tensor_ctr_e2e_oracle_test`, `plain_ctr_oracle_test`,
`ordered_ctr_oracle_test`, `s_order_ctr_bins_oracle_test`) and float
(`slice_first_oracle_test`, `loss_oracle_test`, …).

**Validation.**
```
cargo test -p cb-oracle --lib model_json_test
cargo test -p cb-oracle
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test
cargo test -p cb-train --test slice_first_oracle_test
```
*(A second command once T02 has landed: a test that parses the real fixture,
not just the inline document.)*

**Completion evidence.** The `missing field 'border'` failure before, and the
real-fixture parse after.

**Parallelization.** Fully parallel with T00, T01a and all of Waves 1–3
(`crates/cb-oracle/src/` is named by no other task).

---

### T03 — SPEC-OH-01 — `ObliviousTree` carries ordered mixed-kind levels

**Goal.** The trainer-side `ObliviousTree` stops discarding
`GrownTree.level_kinds` and gains an (initially always empty) `one_hot_splits`
vector, so level order survives the trainer→model boundary.

**Observable completion.** For a CTR-mixed tree,
`ObliviousTree.level_kinds.len() == depth` in level order; for a float-only tree
it is EMPTY (byte-identical persist).

**Blocking / prerequisites.** T00 (the baseline must be frozen first). Blocks
T04, T16.

**Verified files and symbols.**
- `crates/cb-train/src/boosting.rs:775-794` — `pub struct ObliviousTree { splits:
  Vec<Split>, ctr_splits: Vec<CtrSplitSpec>, leaf_values, leaf_weights }`.
- `crates/cb-train/src/boosting.rs:4786-4790` — `let ctr_splits = if has_ctr {…}`.
- `crates/cb-train/src/boosting.rs:4809-4815` — the oblivious push;
  **`grown.level_kinds` is dropped here** (the §1.1 defect's origin).
- `crates/cb-train/src/tree.rs:205-273` — `GrownTree` (`level_kinds` at `:224`)
  and `enum LevelKind { Float(usize), Ctr { ctr_idx, border } }` at `:261-273`.
- `crates/cb-train/src/tree.rs:2955-2997` — the CTR split-back that POPULATES
  `level_kinds`; `tree.rs:652-661` — the plain search leaves it `Vec::new()`.
- `crates/cb-train/src/boosting.rs:4954-4955` — `#[path = "boosting_test.rs"]
  mod tests;` → filter `boosting::tests`.

**Red.**
- File: `crates/cb-train/src/boosting_test.rs` (existing).
- Test fn: `oblivious_tree_records_level_kinds_in_level_order`.
- Setup: extract the push into
  `fn oblivious_from_grown(grown, leaf_values, leaf_weights, ctr_splits) ->
  ObliviousTree` so it is unit-testable without a fit; construct a `GrownTree`
  with `level_kinds = [LevelKind::Ctr { ctr_idx: 0, border: 0.5 },
  LevelKind::Float(0)]`, one `ctr_splits` and one `splits` entry.
- Expected output: `tree.level_kinds == [Ctr{..}, Float(0)]` and
  `tree.one_hot_splits.is_empty()`.
- Expected initial failure: `error[E0609]/[E0560]` — no `level_kinds` field.

**Green (minimal).**
1. Add to `ObliviousTree`: `pub one_hot_splits: Vec<crate::tree::OneHotSplit>`
   and `pub level_kinds: Vec<crate::tree::LevelKind>`, both documented as
   "EMPTY when a tree's levels are all one kind — consumers fall back to the
   kind-grouped order, byte-identical to pre-change".
2. Extract `oblivious_from_grown` and thread `grown.level_kinds` +
   `grown.one_hot_splits` (the latter added in T19; `Vec::new()` for now).
3. Fix every `ObliviousTree { .. }` literal the compiler flags.

**Do NOT** here: add `LevelKind::OneHot` (T19) or touch `from_trained` (T04).

**Refactor.** `ObliviousTree` derives `Debug, Clone, PartialEq` — the two new
`Vec` fields must not break them. Regression scope: `cb-train` full, `cb-model`
full (the struct crosses the crate boundary via `from_trained`).

**Validation.**
```
cargo build --workspace --all-targets
cargo test -p cb-train --lib boosting::tests
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train
cargo test -p cb-model
cargo clippy --workspace --all-targets -- -D warnings
```

**Completion evidence.** New unit test passes; `tensor_ctr_e2e_oracle_test`
still 3/3; workspace clippy clean.

**Parallelization.** NONE — owns `boosting.rs` and `tree.rs`.

---

### T04 — SPEC-OH-02 + SPEC-OH-03 — `from_trained` emits splits in level order

**Goal.** Fix the pre-existing mixed-kind split-ORDER loss. SPEC-OH-03 is
definitionally the Red for SPEC-OH-02 (SPEC.md: "it must fail against today's
`from_trained` and pass after SPEC-OH-02"), so they are one TDD cycle.

**Observable completion.** For a trained tree with `level_kinds = [Ctr, Float]`,
`model.oblivious_trees[0].splits == [ModelSplit::Ctr(..), ModelSplit::Float(..)]`,
and the applied leaf equals the trainer's `leaf_of` for the same object.

**Blocking / prerequisites.** T03. Blocks T05, T21.

**Verified files and symbols.**
- `crates/cb-model/src/model.rs:326-426` — `from_trained`, **24 callers**,
  covered by ≥22 test files. Splits built float-first (`:331-335`) then
  CTR-appended (`:339-356`).
- `crates/cb-model/src/model.rs:399-408` — **[C15] WARNING:** the same function
  builds `region_trees` with `RegionLevel { one_hot }` — an UNRELATED flag (the
  Region walk's equality marker, always `false` for the CPU float grower). Do
  not conflate it with `ModelSplit::OneHot`.
- `crates/cb-model/src/apply.rs:208-215` — `leaf_index_for` walks `tree.splits`
  in STORED order, split `i` → bit `i`.
- `crates/cb-model/src/cbm.rs:646-668` (save) and `:1008-1067` (load) preserve
  that order 1:1.
- `crates/cb-train/src/tree.rs:290-301` — `leaf_index(passes)`: `idx |= 1 << i`.
- `crates/cb-train/src/tree.rs:414-423` — `assign_leaves_any`.

**Red (genuine, defect-demonstrating).**
- File: `crates/cb-model/tests/mixed_kind_split_order_test.rs` (new integration
  test — needs both `cb_train` and `cb_model`).
- Test fn: `ctr_at_level_zero_float_at_level_one_applies_to_the_trainer_leaf`.
- Setup: hand-construct a `cb_train::Model` with ONE `ObliviousTree` whose
  `level_kinds = [LevelKind::Ctr { ctr_idx: 0, border: b }, LevelKind::Float(0)]`,
  one `ctr_splits`, one float `splits`, `leaf_values = [10.0, 20.0, 30.0, 40.0]`;
  attach `ctr_data` via `cb_model::Model::with_ctr_data`.
- Input: one object that FAILS the CTR split and PASSES the float split →
  trainer leaf `0b10 == 2`.
- Expected output: `predict_raw` returns `30.0`.
- Expected initial failure: kind-grouped order stores `[Float, Ctr]`, so the
  float split becomes bit 0 → leaf `0b01 == 1` → `20.0`. **This 1↔2
  transposition IS the §1.1 defect and must be observed before the fix.**

**Green (minimal).** Rewrite the split assembly (`model.rs:331-356`):
```
if t.level_kinds.is_empty() {
    // unchanged legacy path: floats then ctrs (byte-identical)
} else {
    // walk level_kinds in order, indexing into t.splits / t.ctr_splits
    // (the one_hot arm lands in T21)
}
```
A `LevelKind` index out of range is a defensive skip — no panic, no `unwrap()`.

**Refactor.** The `t.level_kinds.is_empty()` branch makes the SPEC-OH-02
invariant structural. Keep it. Regression scope: all of `cb-model` + every CTR
oracle in `cb-train`.

**Validation.**
```
cargo test -p cb-model --test mixed_kind_split_order_test
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test tensor_ctr_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test
cargo test -p cb-train --test ordered_ctr_oracle_test
cargo test -p cb-train --test s_order_ctr_bins_oracle_test
cargo test -p cb-model
cargo test -p cb-train
cargo test -p cb-model --test float_only_byte_identity_test   # T00 gate stays green
```

**Completion evidence.** The Red's `20.0 != 30.0` failure line in the commit
body, its pass after, and `tensor_ctr_e2e_oracle_test` 3/3.

**Parallelization.** NONE. Owns `model.rs`.

---

### T05 — SPEC-OH-08 — `ModelSplit::OneHot` exists with the upstream value space

**Goal.** Introduce the third variant and give EVERY exhaustive consumer a
conservative, never-silent arm, so the workspace builds and no consumer drops a
split. Correct per-consumer behavior lands in T06–T14.

**Observable completion.** `cargo build --workspace --all-targets` green;
`ModelSplit::OneHot(_).float_feature()` and `.as_float()` both `None`;
`OneHotModelSplit.value_hash` typed `i32`.

**Blocking / prerequisites.** T04.

**Verified files and symbols (every exhaustive match; independently confirmed
by PLAN-CHECK).**
- `crates/cb-model/src/model.rs:71-98` — `enum ModelSplit` (2 variants today,
  22 callers across 7 files), `float_feature()`, `as_float()`.
- `crates/cb-model/src/apply.rs:196-201` — `passes_split`.
- `crates/cb-model/src/cbm.rs:656-668` — save split emit.
- `crates/cb-model/src/cbm.rs:1032-1064` — load `BinKind` match (one-hot typed
  error at `:1041-1045`).
- `crates/cb-model/src/gpu_apply.rs:117-138` — `flatten_oblivious_f64`.
- `crates/cb-model/src/fstr.rs:441-451` — `split_flat_indices`.
- `crates/cb-model/src/fstr.rs:163-166`, `:174-177` — the two
  `cat_feature_count` closures.
- `crates/cb-model/src/cbm.rs:385-392` — `build_combined_bins` pushes the unit
  `BinKind::OneHot` [C5].

**Red.**
- File: `crates/cb-model/src/model_test.rs` (new sibling; mount at the bottom of
  `model.rs` as `#[cfg(test)] #[path = "model_test.rs"] mod model_test;` — the
  `mod <file_stem>` rule, so `model::model_test` is a working filter).
- Test fn: `one_hot_split_has_no_float_identity`.
- Setup: `let s = ModelSplit::OneHot(OneHotModelSplit { cat_feature: 1,
  value_hash: -1_438_285_038_i32 });`
- Expected: `s.float_feature().is_none()`, `s.as_float().is_none()`,
  `s == s.clone()`.
- Expected initial failure: `error[E0599]: no variant named OneHot`.

**Green (minimal).**
1. `pub struct OneHotModelSplit { pub cat_feature: usize, pub value_hash: i32 }`
   (`Debug, Clone, Copy, PartialEq`) + the `OneHot(OneHotModelSplit)` variant,
   documented per SPEC §4: `value_hash` is the UPSTREAM raw i32
   `calc_cat_feature_hash` space, **not** a `PerfectHash` bin. `cat_feature` is
   the ABSOLUTE cat-column index.
2. `float_feature()` / `as_float()` → `None`.
3. Conservative arms (each replaced by a later task, none silent):
   - `apply.rs:196-201` → a private `passes_one_hot_split(model, s, cat_values)
     -> bool` stub returning `false`, marked `// T06 implements this`. **T06 is
     the next task in Wave 3 and MUST land before any release cut** — this is
     the one intentional intermediate-state stub in the plan.
   - `cbm.rs:656-668` → `ModelError::Serialize("one-hot split save unsupported
     (pending)")` (replaced by T07).
   - `gpu_apply.rs:117-138` → the existing typed unsupported path.
   - `fstr.rs:441-451` → take T13's contract immediately (a one-liner):
     `vec![flat_cat_index(n_float, oh.cat_feature)]`. Returning `vec![]` would
     be a silent drop and is forbidden.
   - `fstr.rs:163-166`, `:174-177` → `Some(oh.cat_feature)`.
4. Widen `BinKind::OneHot` to `OneHot { cat_feature: usize, value_hash: i32 }`
   [C5], populated in `build_combined_bins` (`cbm.rs:385-392`) from
   `OneHotFeature.Index` + `Values[k]`; keep the `:1041` typed error (T08
   replaces it).

**Refactor.** Add a `# Compatibility` doc note on `ModelSplit` recording the
breaking change for external exhaustive matchers (SPEC §8). No `#[allow]` may be
added anywhere.

**Validation.**
```
cargo build --workspace --all-targets
cargo test -p cb-model --lib model::model_test
cargo test -p cb-model --test model_sum_oracle_test
cargo test -p cb-model --test staged_predict_oracle_test
cargo test -p cb-model
cargo test -p catboost-rs
cargo clippy --workspace --all-targets -- -D warnings
```
*(`model_sum` / `staged_predict` are kept per PLAN-CHECK "Potential Bugs": both
are verified benign — `model_sum.rs` clones `tree.splits` kind-agnostically and
`staged_predict` routes through `passes_split` — and these commands lock that.)*

**Parallelization.** NONE. Gates all of Wave 3.

---

### T06 — SPEC-OH-10 — `passes_split` applies a one-hot split

**Goal.** An object passes a one-hot split iff
`calc_cat_feature_hash(raw_value) == value_hash`.

**Observable completion.** `predict_raw_cat` on a hand-built one-hot model
routes objects by the equality test.

**Blocking / prerequisites.** T05. **Must land before any release cut** (it
replaces T05's `false` stub).

**Verified files and symbols.**
- `crates/cb-model/src/apply.rs:196-201` — `passes_split(model, split, features,
  cat_values)`; `cat_values: &[String]` is the RAW per-object string column.
- `crates/cb-model/src/apply.rs:208-215` — `leaf_index_for`.
- `crates/cb-model/src/apply.rs:929-934` — the existing
  `mod region_apply_test;` / `mod staged_predict_test;` mounts (the
  `mod <file_stem>` precedent).
- `crates/cb-data/src/cat_hash.rs:362-365` — `calc_cat_feature_hash(&str) -> u32`
  [C3 — compare as `i32`].

**Red.**
- File: `crates/cb-model/src/apply_one_hot_test.rs` (new; mount as
  `#[path = "apply_one_hot_test.rs"] mod apply_one_hot_test;`).
- Test fn: `one_hot_split_passes_only_on_the_matching_raw_category`.
- Setup: a depth-1 `Model` whose single split is
  `ModelSplit::OneHot(OneHotModelSplit { cat_feature: 0,
  value_hash: calc_cat_feature_hash("b") as i32 })`, `leaf_values = [1.0, 2.0]`.
- Input: objects with raw cat values `["a"], ["b"], ["c"]`.
- Expected: `[1.0, 2.0, 1.0]`.
- Expected initial failure: T05's stub returns `false` → `[1.0, 1.0, 1.0]`.

**Green (minimal).** Implement `passes_one_hot_split`: bounds-checked
`cat_values.get(s.cat_feature)`, `calc_cat_feature_hash(v) as i32 ==
s.value_hash`; a missing column returns `false` (defensive, consistent with
`passes_float_split`). No `unwrap()`.

**Refactor.** Use the SAME cast direction as T07/T21. Regression scope:
`cb-model` apply / staged predict / region apply / CTR apply.

**Validation.**
```
cargo test -p cb-model --lib apply::apply_one_hot_test
cargo test -p cb-model --test apply_oracle_test
cargo test -p cb-model --test predict_oracle_test
cargo test -p cb-model --test staged_predict_oracle_test
cargo test -p cb-model
```

**Parallelization.** Parallel with T07, T09–T14 (owns `apply.rs`).

---

### T07 — SPEC-OH-11 — `.cbm` save emits `OneHotFeatures` with upstream offsets and pruning

**Goal.** A model with one-hot splits saves an upstream-shaped `.cbm`:
`CatFeatures` + `OneHotFeatures`, values pruned to those the trees reference,
combined bin space `Float → OneHot → Ctr`.

**Observable completion.** SPEC-OH-11's pinned acceptance (1 float feature with
2 borders, cat 0 with 1 used value, cat 1 with 2 used values → global bins float
`0,1`, cat 0 `2`, cat 1 `3,4`), **and** the saved `.cbm` loads in upstream
catboost 1.2.10 and predicts the same values (Verify).

**Blocking / prerequisites.** T05. Blocks T08.

**Verified files and symbols.**
- `crates/cb-model/src/cbm.rs:82-93` — `build_bin_features`.
- `crates/cb-model/src/cbm.rs:101-114` — `split_to_global_index`.
- `crates/cb-model/src/cbm.rs:161-230` — `build_ctr_features`: the **pruning +
  identity-grouping precedent** (`CtrIdentity`, `CtrIdentityKey`, `BTreeMap` for
  deterministic order, cumulative `offsets`). Mirror this shape.
- `crates/cb-model/src/cbm.rs:375-407` — `build_combined_bins` (the LOAD-side
  inverse; already walks `trees.OneHotFeatures()` at `:385-392`).
- `crates/cb-model/src/cbm.rs:519-526` — `build_core_blob` head; `:524`'s "v1 has
  no one-hot bins" comment must be corrected.
- `crates/cb-model/src/cbm.rs:656-668` — the split-emit `match`.
- `crates/cb-model/src/cbm.rs:750-766` — `TModelTreesArgs`: **no `CatFeatures`,
  no `OneHotFeatures` today** [C4].
- `crates/cb-model/src/generated/model_generated.rs:1830-1902` —
  `TOneHotFeature { Index: i32 = -1, Values: [i32], StringValues: [string] }`,
  `TCatFeature { Index, FlatIndex, FeatureId, UsedInModel }`,
  `TModelTrees.CatFeatures` VT=12, `OneHotFeatures` VT=16. **No FlatBuffers
  regeneration needed.**
- `crates/cb-model/src/cbm.rs:65-66` — `mod tests;` → filter `cbm::tests`.

**Red.**
- File: `crates/cb-model/src/cbm_test.rs` (existing).
- Test fn: `one_hot_save_emits_pruned_values_and_float_then_one_hot_bin_offsets`.
- Setup: the exact SPEC-OH-11 acceptance model — `float_feature_borders =
  vec![vec![b0, b1]]`, splits referencing cat 0 value `h0` and cat 1 values
  `h1`, `h2`, plus a cat-1 value `h3` referenced by NO split.
- Expected: after `save_cbm` + re-reading the FlatBuffers core, `OneHotFeatures`
  has `Index == 0, Values == [h0]` and `Index == 1, Values == [h1, h2]` (`h3`
  pruned); `CatFeatures` has 2 entries; emitted `TreeSplits` global indices are
  `2` (cat 0) and `3`/`4` (cat 1).
- Expected initial failure: `ModelError::Serialize("one-hot split save
  unsupported (pending)")` from T05's arm.

**Green (minimal).**
1. `struct OneHotIdentity { cat_feature: usize, values: Vec<i32> }` +
   `build_one_hot_features(model) -> OneHotFeaturePlan`, mirroring
   `build_ctr_features` (`cbm.rs:161-230`): collect the distinct
   `(cat_feature, value_hash)` pairs any tree split references, group by
   `cat_feature` ascending, and emit cumulative offsets.
   **Values ORDER (blocker B3):** pin it empirically against the T02 `multi/`
   scenario, whose generator guarantees first-referenced order ≠ ascending-hash
   order, so the pin is not vacuous. Record the winning rule in a doc comment.
2. `n_one_hot_bins = Σ values`; the CTR base becomes
   `n_float_bins + n_one_hot_bins` (correcting `cbm.rs:524` and the
   `cbm-ctr-save/PLAN.md:36,46` assumption SPEC §7 supersedes).
3. `one_hot_split_to_global_index(split, n_float_bins, &plan) -> Result<i32, _>`.
4. Emit `CatFeatures` (`TCatFeature { Index: c, FlatIndex: n_float + c,
   FeatureId: "", UsedInModel: true }`) and `OneHotFeatures` into
   `TModelTreesArgs` (`cbm.rs:757-762`). Both stay `None` when the model has no
   one-hot splits ⇒ **float-only wire bytes byte-identical** (SPEC-OH-31).
5. Replace T05's conservative save arm.

**Refactor.** Factor the `(feature, value) → global index` lookup so CTR and
one-hot share the cumulative-offset shape. Do NOT change `build_bin_features`.
Regression scope: `cbm_oracle_test`, `ctr_data_roundtrip_test`,
`non_symmetric_grower_roundtrip_oracle_test`, and T00's byte-identity gate.

**Validation.**
```
cargo test -p cb-model --lib cbm::tests
cargo test -p cb-model --test cbm_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test
cargo test -p cb-model --test float_only_byte_identity_test
cargo test -p cb-model
```
**Verify (upstream read-back — the R2 save-side guard).** Run manually (needs
`.venv`, not committed as a test):
```
.venv/bin/python -c "
from catboost import CatBoostRegressor
m = CatBoostRegressor(); m.load_model('<saved>.cbm')
print(m.predict(<pool>))"
```
compared against `cb_model::predict_raw_cat` on the same rows, ≤1e-5.
**If upstream refuses to load the file, this task is NOT done** — that is the
only check proving the emitted layout is genuinely upstream-compatible.

**Parallelization.** Parallel with T06, T09–T14. Serial before T08.

---

### T08 — SPEC-OH-12 — `.cbm` load accepts one-hot splits

**Goal.** Replace the typed `"one-hot split unsupported (v1)"` rejection with a
real decode into `ModelSplit::OneHot`.

**Observable completion.** A one-hot `.cbm` round-trips save→load→save
byte-identically, and the T02 upstream fixture loads.

**Blocking / prerequisites.** T07.

**Verified files and symbols.**
- `crates/cb-model/src/cbm.rs:1032-1064` — the `BinKind` decode match; the
  one-hot rejection at `:1041-1045`.
- `crates/cb-model/src/cbm.rs:375-407` — `build_combined_bins`, widened in T05.
- `.planning/phases/23-ctr-model-loading/cbm-ctr-load/SPEC.md` — decision CTR-05,
  superseded by SPEC §7.
- `crates/cb-model/src/cbm_test.rs::one_hot_split_index_is_typed_error` — must be
  **re-pointed, not deleted silently** (SPEC-OH-12).

**Red.**
- File: `crates/cb-model/src/cbm_test.rs`.
- Test fn: `one_hot_cbm_round_trips_to_model_split_one_hot`.
- Setup: take T07's model, `save_cbm` to a temp path, `load_cbm` back.
- Expected: `loaded.oblivious_trees[0].splits[0] == ModelSplit::OneHot(
  OneHotModelSplit { cat_feature: 0, value_hash: h0 })`; a second `save_cbm`
  is byte-identical.
- Expected initial failure: `ModelError::Deserialize("one-hot split unsupported
  (v1)")`.
- Second fn: rename/re-point `one_hot_split_index_is_typed_error` →
  `one_hot_split_index_decodes_to_one_hot_model_split`, with a comment citing
  SPEC-OH-12 as superseding CTR-05.

**Green (minimal).** Replace the `:1041` arm with
`crate::ModelSplit::OneHot(OneHotModelSplit { cat_feature: *cat_feature,
value_hash: *value_hash })`. `build_combined_bins` must read
`OneHotFeature.Index` as the **cat-feature index** — **assert this against the
T02 fixture (blocker B2); do not assume it is the flat index.** A negative
`Index` (the FlatBuffers default `-1`) is a typed `Deserialize` error, never a
silent 0.

**Refactor.** Load and save one-hot orderings must be exact inverses; doc-note
`build_combined_bins` as the inverse of `build_one_hot_features`. Regression
scope: `cbm_oracle_test`, `ctr_data_roundtrip_test`.

**Validation.**
```
cargo test -p cb-model --lib cbm::tests
cargo test -p cb-model --test cbm_oracle_test
cargo test -p cb-model
```

**Parallelization.** NONE with T07 (same file).

---

### T09 — SPEC-OH-14 — `.json` never silently drops a split

**Goal.** `save_json` returns a typed error on a one-hot model instead of
silently shortening the split list; the json loader returns a typed one-hot
error instead of the misleading `missing field 'border'` (Q1).

**Observable completion.** Both directions produce a `ModelError` naming
one-hot; no json document with a shortened split list can be produced.

**Blocking / prerequisites.** T05. **Not to be confused with T02b** — that
widens the *oracle-only* reader; this one *rejects* in the model loader.

**Verified files and symbols.**
- `crates/cb-model/src/json.rs:474-485` — `filter_map(ModelSplit::as_float)`
  then `.enumerate()` — the silent drop AND the positional `split_index`.
- `crates/cb-model/src/json.rs:399` — the non-symmetric `and_then(as_float)`
  path (guard it too).
- `crates/cb-model/src/json.rs:526-535` — region levels (already typed error).
- `crates/cb-model/src/json.rs:813-816` — **`decode_json` is HERE** [C12/MINOR-6]:
  `serde_json::from_str::<ModelJsonDoc>(contents)?` then `from_doc`. The
  split-decode logic is inside `from_doc`, near `:648-663`.
- `crates/cb-model/src/json.rs:44-54` — `SplitJson` requires `border` +
  `float_feature_index` with no serde default.
- `crates/cb-model/src/error.rs:19-25` (`Deserialize`), `:33-39` (`Serialize`).

**Red.**
- File: `crates/cb-model/tests/json_oracle_test.rs` (existing) — two fns.
- Fn 1: `save_json_rejects_one_hot_models_instead_of_shortening_splits`.
  Setup: a 2-level model, level 0 float, level 1 one-hot. Expected:
  `Err(ModelError::Serialize(msg))` with `msg.contains("one-hot")`. Initial
  failure: `Ok(_)` with a **1-element** `splits` array — assert that length in
  the failing run to document the defect, then flip.
- Fn 2: `json_load_rejects_an_upstream_one_hot_document_with_a_named_error`.
  Setup: the T02 fixture's `model.json`, or a minimal inline document carrying
  `"split_type": "OneHotFeature"` + `cat_feature_index` + `value`. Expected:
  `Err(ModelError::Deserialize(msg))` with `msg.contains("one-hot")`. Initial
  failure: `ModelError::Json(… missing field 'border' …)`.

**Green (minimal).**
1. `save_json`: before building `splits`, scan for
   `matches!(s, ModelSplit::OneHot(_))` and return
   `ModelError::Serialize("one-hot splits cannot be represented in the numeric
   model.json schema (v1)")`. Same guard on the non-symmetric arm (`:399`).
2. Loader guard — **[C12/MINOR-3] a "cheap targeted probe" is NOT achievable
   with `serde_json`.** Choose ONE and state it in the code:
   (a) a full `serde_json::Value` pre-parse in `decode_json` (`:813-816`) before
   the typed deserialization, walking
   `oblivious_trees[*].splits[*].split_type` — acceptable cost on an error path
   that is not hot; **or**
   (b) `#[serde(default)] border` on `SplitJson` plus an explicit
   `split_type == "OneHotFeature"` check inside `from_doc` that returns the
   typed error. **(b) is preferred** — it keeps one parse and puts the check
   where the split is actually interpreted. If (b) is taken, add a test that a
   *float* document with a genuinely missing `border` still fails loudly, so the
   default does not mask a malformed float split.

**Refactor.** Do NOT change `split_index` semantics (Q1). Add a doc note on
`save_json` recording that a real one-hot json emit would require re-specifying
`split_index` as the global combined bin index, and is deferred. Regression
scope: `json_oracle_test`, `class_params_roundtrip_test`.

**Validation.**
```
cargo test -p cb-model --test json_oracle_test
cargo test -p cb-model --test class_params_roundtrip_test
cargo test -p cb-model
```

**Parallelization.** Parallel with T06, T07, T10–T14 (owns `json.rs`).

---

### T10 — SPEC-OH-15 — SHAP never silently drops a one-hot split (all FOUR public surfaces)

**Goal.** Every consumer of `float_splits_of` returns a typed unsupported error
for a model carrying non-float splits, instead of a silently shortened
`tree_depth`. *(PLAN-CHECK MAJOR-1: revision 1 covered only 2 of 4 surfaces, and
its escape hatch is now DELETED — the real production call-site count is 4.)*

**Observable completion.** All four of `shap_values`, `shap_interaction_values`,
**`prediction_diff`** and **`sage_values`** return
`Err(ShapUnsupported::OneHotSplits)` on a one-hot model; float-only numbers
unchanged.

**Blocking / prerequisites.** T05. **SERIAL against T13** (both write
`fstr.rs`).

**Verified files and symbols [C6, confirmed by PLAN-CHECK].**
- `crates/cb-model/src/shap.rs:534` — `pub fn shap_values(model, cols,
  n_features) -> Vec<Vec<f64>>` (**infallible**).
- `crates/cb-model/src/shap.rs:550-552` — `fn float_splits_of(splits) ->
  Vec<Split>` = `filter_map(as_float)` — the silent drop. **TWO call sites:
  `:830` (`shap_values_fixed`) and `:1139` (inside `prediction_diff`).**
- `crates/cb-model/src/shap.rs:941` — `shap_interaction_values`.
- `crates/cb-model/src/shap.rs:1137` — **`pub fn prediction_diff`**.
- `crates/cb-model/src/shap.rs:1236` — **`pub fn sage_values`**.
- `crates/cb-model/src/lib.rs:58` — `pub use shap::{prediction_diff, sage_values,
  shap_interaction_values, shap_values};` — all four are shipped public API.
- `crates/cb-model/src/shap.rs:725-733` — the non-symmetric descent
  (`float_feature()` → `None` stops descending silently) — audit + guard.
- Production callers (grep, non-test): `crates/cb-model/src/fstr.rs:804`,
  `crates/cb-model/src/fstr.rs:846`, `crates/catboost-rs/src/model.rs:292`,
  `crates/catboost-rs/src/model.rs:393`. **Four — no escape hatch is warranted.**
- `crates/cb-model/src/fstr.rs:788` — `loss_function_change -> Vec<f64>`
  (`#[must_use]`), the cascade target.
- MEMORY `next-features-5plan-batch`: the `CatBoostError::to_pyerr` E0004 trap —
  match any new facade variant explicitly.

**Red (four fns, one per surface).**
- File: `crates/cb-model/tests/shap_oracle_test.rs` (existing).
- `shap_values_rejects_one_hot_models_instead_of_shrinking_tree_depth`
- `shap_interaction_values_rejects_one_hot_models`
- `prediction_diff_rejects_one_hot_models`
- `sage_values_rejects_one_hot_models`
- Setup (each): a depth-2 model with one float and one one-hot split.
- Expected: `Err(ShapUnsupported::OneHotSplits)`.
- Expected initial failure: an `Ok`-shaped result computed at depth 1. **Assert
  the wrong row width / a known-wrong value first** so the defect is documented,
  then flip the assertion.

**Green (minimal).**
1. `#[derive(Debug, thiserror::Error)] pub enum ShapUnsupported { OneHotSplits,
   CtrSplits }` — the CTR case is the SAME pre-existing silent drop and the same
   failure reason, so it belongs here, not in a separate spec.
2. **Make `float_splits_of` return `Result<Vec<Split>, ShapUnsupported>`**, so
   the compiler enforces the guard at BOTH call sites (`:830`, `:1139`) — this is
   what makes the coverage structural rather than a checklist.
3. Cascade to `Result<_, ShapUnsupported>`: `shap_values`,
   `shap_interaction_values`, `prediction_diff`, `sage_values`, and
   `fstr::loss_function_change`.
4. Fix the four production call sites; `crates/catboost-rs/src/model.rs:292` and
   `:393` already return `Result<_, CatBoostError>` — map the new error there,
   matching the new variant EXPLICITLY (E0004 trap).

**The revision-1 escape hatch is REMOVED.** With 4 call sites the signature
change is unambiguously the right path; a reduced-guarantee variant would leave
`prediction_diff` / `sage_values` silently wrong.

**Refactor.** Do not alter any float-only SHAP arithmetic. Regression scope:
`shap_oracle_test`, `advanced_fstr_oracle_test`, `fstr_oracle_test`,
`feature_selection_oracle_test` (cb-train), `catboost-rs`, `catboost-rs-py`.

**Validation.**
```
cargo test -p cb-model --test shap_oracle_test
cargo test -p cb-model --test advanced_fstr_oracle_test
cargo test -p cb-model --test fstr_oracle_test
cargo test -p cb-train --test feature_selection_oracle_test
cargo test -p catboost-rs
cargo test -p catboost-rs-py --no-run
cargo build --workspace --all-targets
```

**Parallelization.** Owns `shap.rs`; touches `fstr.rs` (`:788`, `:804`, `:846`)
and `catboost-rs/src/model.rs` — **SERIAL against T13**. Parallel with T06, T07,
T09, T11, T12, T14.

---

### T11 — SPEC-OH-16 — ONNX and CoreML guards reject one-hot models (and `catboost-rs-py` keeps building)

**Goal.** Both exportability guards reject a one-hot model with a typed error
instead of passing it through to emit `(feature = 0, border = 0.0)` — **and the
Python-binding error mapping is updated in the same task**, because both
`match`es there are exhaustive.

**Observable completion.** `is_onnx_exportable` / `is_coreml_exportable` return
`OneHotSplitsUnsupported`; `cargo build --workspace --all-targets` is green.

**Blocking / prerequisites.** T05. **This task will NOT surface as a `cb-model`
build failure** — both guards use `matches!(split, ModelSplit::Ctr(_))` and
accept a third variant silently. It must be written by hand. It **WILL** surface
as an `E0004` in `catboost-rs-py` [C14] — which is why that crate is in the
validation block.

**Verified files and symbols.**
- `crates/cb-model/src/export/onnx.rs:106-113` — `has_ctr_split = … .any(|split|
  matches!(split, ModelSplit::Ctr(_)))` then
  `if model.ctr_data.is_some() || has_ctr_split { return Err(
  OnnxExportError::CategoricalFeaturesUnsupported) }`.
- `crates/cb-model/src/export/onnx.rs:205-215` — node build via `as_float()` →
  `None` → `(0, 0.0)`.
- `crates/cb-model/src/export/coreml.rs:106-113` and `:173-182` — identical.
- `crates/cb-model/src/export/onnx.rs:636-637` — `#[path = "onnx_test.rs"]
  mod tests;` → filter `export::onnx::tests`. **[C13/MINOR-2]**
- `crates/cb-model/src/export/coreml.rs:307-308` — `mod tests;` → filter
  `export::coreml::tests`.
- **`crates/catboost-rs-py/src/errors.rs:148-155`** — matches
  `cb_model::OnnxExportError` EXHAUSTIVELY: `CategoricalFeaturesUnsupported |
  NonObliviousTreesUnsupported | RegionTreesUnsupported |
  NonIntegerClassLabelsUnsupported => CatBoostValueError`, then `Io =>
  PyIOError`, then `Encode => CatBoostError`. No `_` arm.
- **`crates/catboost-rs-py/src/errors.rs:164-171`** — the same shape for
  `CoreMlExportError` (`CategoricalFeaturesUnsupported |
  NonObliviousTreesUnsupported | RegionTreesUnsupported | MultiDimUnsupported`).
- **Verified SAFE by contrast (state this in the code review):**
  `PdpError` reaches the facade via `#[from]` at
  `crates/catboost-rs/src/error.rs:77` with no exhaustive match (so T14 is
  safe), and `GpuApplyUnsupported` has no exhaustive facade match (so T12 is
  safe).

**Red.**
- Files: `crates/cb-model/src/export/onnx_test.rs` and
  `crates/cb-model/src/export/coreml_test.rs` (both **exist**, [C13]).
- Test fns: `onnx_export_rejects_one_hot_models` /
  `coreml_export_rejects_one_hot_models`.
- Setup: a depth-1 model whose single split is `ModelSplit::OneHot`.
- Expected: `Err(OnnxExportError::OneHotSplitsUnsupported)` /
  `Err(CoreMlExportError::OneHotSplitsUnsupported)`.
- Expected initial failure: `Ok(())` from the guard; if the test drives the full
  export, a produced document containing a node with `feature == 0,
  border == 0.0`. **Assert that emitted `(0, 0.0)` first** to document the
  silent-wrong behavior, then flip.

**Green (minimal).**
1. Add `OneHotSplitsUnsupported` to both error enums and a dedicated
   `has_one_hot_split` scan in both guards, placed BEFORE the CTR check so the
   message is specific.
2. **Add the new arm to `catboost-rs-py/src/errors.rs:148-155` and `:164-171`**
   (join the existing or-pattern of guard-rejection variants →
   `CatBoostValueError`).
3. Leave the node builders untouched (unreachable after the guard).

**Refactor.** No extraction across the two export modules (deliberately
independent). Regression scope: `export::onnx::tests`, `export::coreml::tests`,
`coreml_export_test`, the ONNX oracle suite, and the whole workspace build.

**Validation.**
```
cargo test -p cb-model --lib export::onnx::tests
cargo test -p cb-model --lib export::coreml::tests
cargo test -p cb-model --test coreml_export_test
cargo test -p cb-model
cargo build --workspace --all-targets
cargo test -p catboost-rs-py --no-run
```

**Parallelization.** Owns `export/onnx.rs`, `export/coreml.rs` **and
`catboost-rs-py/src/errors.rs`**. Parallel with T06, T07, T09, T10, T12, T13,
T14.

---

### T12 — SPEC-OH-17 — GPU-apply guard names one-hot explicitly

**Goal.** `gpu_apply` returns an explicit `GpuApplyUnsupported::OneHotSplits`
instead of falling through the guard into the downstream match's confusing
message.

**Observable completion.** The error variant is `OneHotSplits`, produced by the
guard, not by `flatten_oblivious_f64`.

**Blocking / prerequisites.** T05.

**Verified files and symbols.**
- `crates/cb-model/src/gpu_apply.rs:50-57` — the guard
  (`matches!(.., Ctr(_))` → passes a one-hot model).
- `crates/cb-model/src/gpu_apply.rs:117-138` — `flatten_oblivious_f64`
  (exhaustive; T05 gave it the conservative arm).
- `crates/cb-model/src/gpu_apply.rs:169-170` — `#[path = "gpu_apply_test.rs"]
  mod tests;` → filter `gpu_apply::tests`.
- **[C14] safe:** `GpuApplyUnsupported` is not matched exhaustively anywhere in
  the facade or the Python bindings, so adding a variant breaks nothing.

**Red.**
- File: `crates/cb-model/src/gpu_apply_test.rs` (existing).
- Test fn: `gpu_apply_guard_names_one_hot_splits_explicitly`.
- Setup: a one-hot model; call the guard directly.
- Expected: `Err(GpuApplyUnsupported::OneHotSplits)`.
- Expected initial failure: `Ok(())` from the guard.

**Green (minimal).** Add the variant and a `has_one_hot_split` scan before the
CTR check.

**Refactor.** None. Regression scope: `gpu_apply::tests`,
`crates/cb-backend/tests/apply_oblivious_launch_test.rs`.

**Validation.**
```
cargo test -p cb-model --lib gpu_apply::tests
cargo test -p cb-backend --test apply_oblivious_launch_test
cargo build --workspace --all-targets
```

**Parallelization.** Owns `gpu_apply.rs`. Parallel with T06, T07, T09–T11, T13,
T14.

---

### T13 — SPEC-OH-18 — fstr treats one-hot splits as cat-feature identities

**Goal.** Two one-hot splits on the SAME cat feature are the same internal
feature (upstream `TFeature` identity ignores the value, exactly as it ignores a
float border), and `split_flat_indices` returns the flat cat index.

**Observable completion.** `same_internal_feature(OneHot{cat:1,v:a},
OneHot{cat:1,v:b}) == true`; `…{cat:2,..} == false`; `split_flat_indices` yields
`vec![flat_cat_index(n_float, cat_feature)]`.

**Blocking / prerequisites.** T05 (which already took the compiler-forced
`split_flat_indices` / `cat_feature_count` one-liners). This task adds the
NON-compiler-forced `same_internal_feature` arm and the tests. **SERIAL against
T10.**

**Verified files and symbols.**
- `crates/cb-model/src/fstr.rs:461-474` — `same_internal_feature` with the
  `_ => false` wildcard. With three variants, `(OneHot, OneHot)` falls into `_`
  → **silently backwards**.
- `crates/cb-model/src/fstr.rs:441-451` — `split_flat_indices` (exhaustive).
- `crates/cb-model/src/fstr.rs:130-151` — `feature_count` uses
  `filter_map(float_feature)`; a one-hot split correctly does NOT widen the
  float vector.
- `crates/cb-model/src/fstr.rs:163-177` — the two `cat_feature_count` closures.
- `crates/cb-model/src/fstr.rs:874-875` — `mod tests;` → filter `fstr::tests`.
- `flat_cat_index(n_float, c)` — the convention from
  `.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/`.

**Red.**
- File: `crates/cb-model/src/fstr_test.rs` (existing).
- Fn 1: `two_one_hot_splits_on_the_same_cat_feature_are_one_internal_feature`.
  Setup: `a = OneHot{cat_feature:1, value_hash:111}`,
  `b = OneHot{cat_feature:1, value_hash:222}`,
  `c = OneHot{cat_feature:2, value_hash:111}`.
  Expected: `same_internal_feature(&a,&b) == true`,
  `same_internal_feature(&a,&c) == false`,
  `split_flat_indices(&a, n_float) == vec![flat_cat_index(n_float, 1)]`.
  Expected initial failure: `same_internal_feature(&a,&b) == false` via `_`.
- Fn 2: `interaction_attributes_two_one_hot_splits_to_one_pair` — drive
  `interaction()` on a depth-2 one-hot tree; assert one self-interaction entry
  rather than two distinct internal features.

**Green (minimal).** Add
`(ModelSplit::OneHot(x), ModelSplit::OneHot(y)) => x.cat_feature == y.cat_feature`
BEFORE the `_` wildcard. Do NOT compare `value_hash`.

**Refactor.** Doc-note that the value is excluded for the same reason the float
border is. Replace the `_` wildcard with the explicit cross-kind arms **only if
behavior is unchanged** — doing so makes a fourth variant a build error.
Regression scope: `fstr_oracle_test`, `fstr_ctr_oracle_test`,
`advanced_fstr_oracle_test`.

**Validation.**
```
cargo test -p cb-model --lib fstr::tests
cargo test -p cb-model --test fstr_oracle_test
cargo test -p cb-model --test fstr_ctr_oracle_test
cargo test -p cb-model --test advanced_fstr_oracle_test
```

**Parallelization.** Owns `fstr.rs` — **SERIAL against T10**. Parallel with
T06, T07, T09, T11, T12, T14.

---

### T14 — SPEC-OH-19 — partial dependence rejects one-hot models

**Goal.** `partial_dependence` returns a typed error on a one-hot model rather
than operating on the float-only column space.

**Observable completion.** `Err(PdpError::OneHotSplitsUnsupported)`.

**Blocking / prerequisites.** T05.

**Verified files and symbols.**
- `crates/cb-model/src/partial_dependence.rs:287-291` — `pub fn
  partial_dependence(model, columns, features) -> Result<PartialDependence,
  PdpError>` (**already fallible** — a cheap task); `validate(...)` at `:292` is
  the insertion point.
- `crates/cb-model/src/partial_dependence.rs:76-…` — `enum PdpError`.
- `crates/cb-model/src/partial_dependence.rs:325-326` — `mod tests;` → filter
  `partial_dependence::tests`.
- **[C14] safe:** `PdpError` reaches the facade via `#[from]` at
  `crates/catboost-rs/src/error.rs:77` — no exhaustive match, so adding a
  variant breaks nothing.

**Red.**
- File: `crates/cb-model/src/partial_dependence_test.rs` (existing).
- Test fn: `partial_dependence_rejects_one_hot_models`.
- Setup: a one-hot model, `features = [0]`, one float column.
- Expected: `Err(PdpError::OneHotSplitsUnsupported)`.
- Expected initial failure: `Ok(PartialDependence { .. })` computed over the
  float column space alone.

**Green (minimal).** Add the variant and a `has_one_hot_split` scan inside
`validate`.

**Refactor.** None. Regression scope: `partial_dependence_oracle_test`.

**Validation.**
```
cargo test -p cb-model --lib partial_dependence::tests
cargo test -p cb-model --test partial_dependence_oracle_test
cargo build --workspace --all-targets
```

**Parallelization.** Owns `partial_dependence.rs`. Parallel with T06–T13.

---

### T15 — SPEC-OH-13 — an upstream-produced one-hot `.cbm` predicts within 1e-5

**Goal.** The ONLY specification that proves our encoding is genuinely
upstream-compatible rather than merely self-consistent (SPEC §1.2 / §9 R2, load
side).

**Observable completion.** The committed `one_hot_train/` fixture's upstream
`.cbm` loads through production `load_cbm` and `predict_raw_cat` matches
`preds.npy` within 1e-5.

**Blocking / prerequisites.** T08 (load arm), T02 (fixture), **T02b** (the
oracle-side `model.json` reader, if the test reads borders from `model.json`).

**Verified files and symbols.**
- `crates/cb-oracle/fixtures/one_hot_train/` — produced by T02.
- `crates/cb-model/src/cbm.rs:1032-1064` — the decode path.
- `crates/cb-model/src/apply.rs:196-201` — `passes_split` one-hot arm (T06).
- Precedent: `crates/cb-model/tests/cbm_oracle_test.rs` and the `ctr_load/`
  family consumed by the CTR load tests.
- research.md §4.3: our decoder ALREADY classifies the upstream one-hot split
  index into the one-hot range (it returns exactly `"one-hot split unsupported
  (v1)"` on the real artifact) — strong prior evidence the offset math matches.
  This task converts that into a prediction-level proof.

**Red.**
- File: `crates/cb-model/tests/one_hot_cbm_oracle_test.rs` (new).
- Test fn: `upstream_one_hot_cbm_predicts_within_1e5`.
- Setup: load `one_hot_train/<scenario>/model.cbm`, read `X_float.npy`,
  `cat_cols.json`, `preds.npy`.
- Expected: `max |ours - upstream| <= 1e-5` on every row, AND
  `model.oblivious_trees.iter().flat_map(|t| &t.splits).any(|s| matches!(s,
  ModelSplit::OneHot(_)))` — the fixture must actually exercise the variant.
- Expected initial failure: fixture missing (before T02) or `"one-hot split
  unsupported (v1)"` (before T08).

**Green (minimal).** No production change expected — this is a proof. A failure
localizes to T07/T08's offset or `Index` interpretation (blocker B2/B3) and must
be fixed in the owning task.

**Refactor.** None. Regression scope: `cargo test -p cb-model`.

**Validation.**
```
cargo test -p cb-model --test one_hot_cbm_oracle_test
cargo test -p cb-model
```

**Completion evidence.** The measured `max|diff|` in the commit body.

**Parallelization.** NONE (depends on T02, T02b, T08).

---

### T16 — SPEC-OH-04 — one-hot-routed columns are identified in `train_inner`

**Goal.** `train_inner` collects the absolute cat-feature indices that
`route_categorical` sends to `EncodingPath::OneHot`, disjoint from the existing
CTR-eligible list.

**Observable completion.** For cardinalities `[2, 5, 3]` and
`one_hot_max_size = 3`, the one-hot list is `[0, 2]` and `eligible_absolute` is
`[1]` — unchanged from today.

**Blocking / prerequisites.** T03. **Blocks T01b** (which needs this list).

**Verified files and symbols.**
- `crates/cb-train/src/boosting.rs:2702-2708` — `cat_cardinalities` via
  `learn_set_cardinality`.
- `crates/cb-train/src/boosting.rs:2721-2729` — `eligible_absolute` (the
  `EncodingPath::Ctr` filter) — the exact mirror to write.
- `crates/cb-train/src/boosting.rs:2740-2749` — `cat_eligible_buckets` built with
  `cb_data::perfect_hash_bins` (already imported).
- `crates/cb-train/src/candidates.rs:92-104` — `route_categorical(card, k)`;
  `:78-80` — `one_hot_max_size_default() == 2`.
- `crates/cb-train/src/candidates.rs:54` — the `candidates_test.rs` mount.
- `crates/cb-train/src/boosting.rs:4954-4955` — `mod tests;` → filter
  `boosting::tests`.

**Red (SCAFFOLDING — an extraction, `E0425`; no defect demonstrated).**
- File: `crates/cb-train/src/boosting_test.rs`.
- Test fn: `one_hot_routed_columns_are_partitioned_disjointly_from_ctr_eligible`.
- Setup: extract `fn partition_cat_columns(cards: &[u32], one_hot_max_size: u32)
  -> (Vec<usize> /*one_hot*/, Vec<usize> /*ctr*/)` and call it directly.
- Input/expected: `([2,5,3], 3) → (vec![0,2], vec![1])`;
  `([1,2], 2) → (vec![1], vec![])` (cardinality 1 routes to neither —
  `route_categorical` requires `1 < card`).
- Expected initial failure: `error[E0425]` — the function does not exist.

**Green (minimal).** Add `partition_cat_columns`; derive BOTH lists from it at
`boosting.rs:2721-2729`, keeping `eligible_absolute`'s value byte-identical.

**Refactor.** The two lists must be provably disjoint **by construction** — one
`match` on `EncodingPath`, not two independent filters. Regression scope: every
CTR oracle.

**Validation.**
```
cargo test -p cb-train --lib boosting::tests
cargo test -p cb-train --lib candidates::tests   # verified: candidates.rs:54-55 -> mod tests;
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test
cargo test -p cb-train --test ordered_ctr_oracle_test
cargo test -p cb-train
```

**Parallelization.** NONE (owns `boosting.rs`).

---

### T01b — SPEC-OH-27 — enforce the RNG draw-order contract (consume the rule, or typed-reject)

**Goal.** Make SPEC-OH-27's mandate **executable**: either the one-hot level
search consumes exactly the ground-truth per-level draw count, or one-hot ×
(bootstrap ≠ `No` OR `random_strength ≠ 0`) is typed-rejected. *(PLAN-CHECK
CRITICAL-4: revision 1 described this inside T01, which sat in a
"no production Rust" wave and needed a column list that T16 creates three waves
later — it was not schedulable.)*

**Observable completion.** EITHER a Rust test asserts the production per-level
draw count equals the T01a rule, OR `train_cat` on a one-hot pool with
`bootstrap_type != No` (or `random_strength != 0`) returns a typed
`CbError::Unsupported` naming the combination. **Never a guessed count.**

**Blocking / prerequisites.** **T01a** (the verdict) and **T16** (the one-hot
column list). Owns `crates/cb-train/src/boosting.rs`. Blocks T18's perturbed arm
and T30's benchmark config.

**Verified files and symbols.**
- `.planning/plans/one-hot-categorical-training/instrumented-ground-truth/ONE_HOT_GROUND_TRUTH.md`
  — T01a's verdict line (`RSM_RULE: …` or `STATUS: NOT-ESTABLISHED`).
- `crates/cb-train/src/tree.rs:610-614` — the per-level RSM draw loop over
  `matrix.n_features()` (FLOAT-only, `tree.rs:355-357`).
- `crates/cb-train/src/tree.rs:1095-…` — `select_level_perturbed`'s
  `SelectBestCandidate` `std_normal` pass, one draw per LISTED feature.
- `crates/cb-train/src/boosting.rs:4000-4018` — where `perturb` is assembled;
  `perturb.is_some()` ⇔ `draws_active` (`random_strength != 0` OR an active
  bootstrap).
- `crates/cb-train/src/boosting.rs:2721-2729` — T16's one-hot list, now in scope.
- `crates/cb-core/src/error.rs:86-92` — `CbError::Unsupported(String)`.
- `crates/cb-train/src/device_draw_replay.rs` + `device_draw_replay_test.rs:47`
  — the existing replay harness (the natural home for a draw-count assertion).

**Red — BRANCH A (T01a established the rule).**
- File: `crates/cb-train/tests/one_hot_draw_accounting_test.rs` (created by
  T01a).
- Test fn: `one_hot_perturbed_search_consumes_the_ground_truth_draw_count`.
- Setup: a 1-float + 1-binary-cat pool, `bootstrap_type = Bayesian`,
  `bagging_temperature = 1.0`, `random_seed = 0`, `thread_count = 1`,
  `iterations = 3`, `depth = 2`, `boost_from_average = false`; count RNG draws
  via the replay harness.
- Expected: per-level draws equal the `RSM_RULE` from `ONE_HOT_GROUND_TRUTH.md`
  (`n_float` or `n_float + n_one_hot`), and the resulting predictions match the
  T01a trace's tree-0/1/2 splits.
- Expected initial failure: the production loop draws `n_float` per level while
  the rule says `n_float + n_one_hot` (or vice versa) — a concrete count
  mismatch.

**Red — BRANCH B (T01a recorded `STATUS: NOT-ESTABLISHED`).**
- Same file. Test fn:
  `one_hot_with_active_draws_is_typed_rejected_until_ground_truth_exists`.
- Setup: `train_cat` on a one-hot pool with `bootstrap_type = Bayesian`.
- Expected: `Err(CbError::Unsupported(msg))` with `msg.contains("one-hot")` and
  `msg.contains("bootstrap")`.
- Expected initial failure: `Ok(model)` — the fit silently proceeds with a
  desynced draw stream (the `d7676b5` MVS defect class).

**Green (minimal).**
- **Branch A:** add `pub(crate) fn rsm_listed_feature_count(matrix:
  &FeatureMatrix) -> usize` in `tree.rs` returning the ground-truth rule, and
  consume it in BOTH `tree.rs:610-614` and the `SelectBestCandidate` pass — ONE
  function, so the two can never drift. (T18 then wires the one-hot candidates
  under the same rule.)
- **Branch B:** in `train_inner`, immediately after T16's partition, if the
  one-hot list is non-empty AND `perturb.is_some()`, return
  `CbError::Unsupported("one-hot categorical training is not supported with
  bootstrap_type != No or random_strength != 0: the upstream per-level RNG draw
  accounting for one-hot candidates has not been established (see
  .planning/plans/one-hot-categorical-training/instrumented-ground-truth/ONE_HOT_GROUND_TRUTH.md)")`.

**Instruction to T18 (both branches).** T18 step 3 must say explicitly what the
perturbed arm does: under **Branch A** it iterates
`rsm_listed_feature_count(matrix)`; under **Branch B** the perturbed arm is
*unreachable with one-hot columns* (the gate above fires first), so it keeps
iterating `matrix.n_features()` **unchanged**, and T18 adds an assertion that
one-hot candidates are never enumerated while `perturb.is_some()`.

**Refactor.** Under Branch B, the gate must live where BOTH the one-hot list and
`perturb` are in scope, so no future dispatch arm can bypass it. Regression
scope: every bootstrap oracle.

**Validation.**
```
cargo test -p cb-train --test one_hot_draw_accounting_test
cargo test -p cb-train --test bootstrap_oracle_test
cargo test -p cb-train --test bootstrap_dev_oracle_test
cargo test -p cb-train --test mvs_seeds_oracle_test
cargo test -p cb-train --test regularization_oracle_test
cargo test -p cb-train
```

**Completion evidence.** The branch taken, quoted from
`ONE_HOT_GROUND_TRUTH.md`; the draw counts or the typed error text; unchanged
bootstrap oracle results.

**Parallelization.** NONE (owns `boosting.rs`, and `tree.rs` under Branch A).

---

### T17 — SPEC-OH-05 — one-hot columns reach the candidate feature matrix

**Goal.** Each one-hot-routed column contributes its `PerfectHash` bin column to
`FeatureMatrix.cat_bins`, **and** the fit-wide bin→raw-hash table is built and
stored here. *(PLAN-CHECK MAJOR-10: the `cb_train::Model.one_hot_bin_to_hash`
field addition moves from T21 into T17, which already owns `boosting.rs` and
produces the table. T21 becomes a pure `cb-model` consumer.)*

**Observable completion.** `matrix.n_cat_features()` equals the one-hot list
length; `distinct_bins_ascending(matrix.cat_bins[c])` equals the column's
distinct bins; `one_hot_bin_to_hash[c][bin] == calc_cat_feature_hash(raw)` for
the raw value that produced `bin`; and
`one_hot_bin_to_hash[c].len() == cardinality(c)` **asserted**.

**Blocking / prerequisites.** T16 (and T01b, which precedes it on
`boosting.rs`).

**Verified files and symbols.**
- `crates/cb-train/src/boosting.rs:2678` — `let matrix =
  FeatureMatrix::new(feature_values, feature_borders);` — `new` hard-codes
  `cat_bins: &[]` (`tree.rs:345-351`). All `FeatureMatrix` fields are `pub`
  (`tree.rs:328-339`), so a struct literal is the seam.
- `crates/cb-train/src/boosting.rs:2740-2749` — the `perfect_hash_bins` reuse
  pattern.
- `crates/cb-data/src/cat_hash.rs:471-479` — `perfect_hash_bins(&[&str]) ->
  CbResult<Vec<u32>>`; the hash→bin map is **discarded** [C8, confirmed].
- `crates/cb-data/src/cat_hash.rs:444-459` — `PerfectHash::remap_bounded`
  assigns `bin = map.len()` on first sight ⇒ the zip-inverse is exact.
- `crates/cb-data/src/cat_hash.rs:362-365` — `calc_cat_feature_hash -> u32`.
- `crates/cb-train/src/tree.rs:3037-3046` — `distinct_bins_ascending`.
- `crates/cb-train/src/boosting.rs:775-794` — `cb_train::ObliviousTree` lives in
  this file, and so does `cb_train::Model` — hence the ownership move.

**Red (SCAFFOLDING for the extraction + one genuine assertion).**
- File: `crates/cb-train/src/boosting_test.rs`.
- Test fn: `one_hot_columns_build_bins_and_an_exact_bin_to_hash_inverse`.
- Setup: extract `fn build_one_hot_columns(cat_columns: &[Vec<String>],
  one_hot_abs: &[usize]) -> CbResult<(Vec<Vec<u32>>, Vec<Vec<u32>>)>` returning
  `(bins, hash_by_bin)`.
- Input: `cat_columns = [vec!["b","a","b","c"]]`, `one_hot_abs = [0]`.
- Expected: `bins == [[0, 1, 0, 2]]` (first-seen) and
  `hash_by_bin == [[h("b"), h("a"), h("c")]]`, **and**
  `hash_by_bin[0].len() == 3` (the cardinality invariant — PLAN-CHECK
  "Potential Bugs").
- Expected initial failure: `E0425` (extraction), then, once written naively
  without the length assertion, a genuine failure if the inverse is built from
  ascending hashes instead of first-seen bins.

**Green (minimal).**
1. `build_one_hot_columns`: call `perfect_hash_bins`, then zip the raw column
   with the returned bins and set `hash_by_bin[bin] = calc_cat_feature_hash(raw)`
   on first sight. **Assert `hash_by_bin[c].len() == cardinality(c)`** and
   return `CbError::Degenerate` if not — a length mismatch is an internal
   invariant violation, not a data condition. Bounds-checked `get`/`get_mut`, no
   `unwrap()`.
2. Replace `boosting.rs:2678` with the struct literal binding `cat_bins` (an
   empty `Vec` on the numeric path ⇒ `n_cat_features() == 0`, byte-identical).
3. **Add `pub one_hot_bin_to_hash: Vec<Vec<u32>>` to `cb_train::Model`** (indexed
   by one-hot column POSITION, then bin) **and** `pub one_hot_absolute:
   Vec<usize>` (position → ABSOLUTE cat-column index), populate both here, and
   fix every `Model { .. }` literal the compiler flags. T21 consumes them
   read-only.

**Refactor.** Do NOT add a second hashing loop — `perfect_hash_bins` is the one
sanctioned primitive (SPEC §3). Document that the table is valid only for the
exact learn-set column it was built from (bins are first-seen per column).
Regression scope: every float-only oracle (`cat_bins` must stay empty there) and
T00's byte-identity gate.

**Validation.**
```
cargo test -p cb-train --lib boosting::tests
cargo test -p cb-train --test slice_first_oracle_test
cargo test -p cb-train --test loss_oracle_test
cargo test -p cb-train --test leaf_methods_oracle_test
cargo test -p cb-train
cargo test -p cb-model --test float_only_byte_identity_test
cargo build --workspace --all-targets
```

**Parallelization.** NONE.

---

### T18 — SPEC-OH-06 — fused one-hot-aware level search

**Goal.** One-hot candidates are scored through the SAME per-feature histogram
machinery as floats (`left = bin_sums[value]`, `right = total - left`), inside
`select_level_plain` / `select_level_perturbed`, preserving the float-then-
one-hot enumeration order and **routing the argmax through a generalized
`select_best_candidate`** (PLAN-CHECK MAJOR-6). `score_candidate_any` MUST NOT
be on the production path (SPEC §9 R4).

**Observable completion.** A one-hot tree grown through
`greedy_tensor_search_oblivious_perturbed` is IDENTICAL to the same tree grown
through the frozen `grow_one_hot_tree` reference — the fused path is a pure
speedup, not a different algorithm — and `tree_tie_break_test.rs` pins
float-first on an exact float/one-hot score tie.

**Blocking / prerequisites.** T17, and **T01b** (which branch the perturbed arm
takes).

**Verified files and symbols.**
- `crates/cb-train/src/tree.rs:931-1041` — `select_level_plain`; the fused rayon
  `map_init(ScanScoreScratch::new, …)` pass at `:960-1021`; the flatten +
  `select_best_candidate` at `:1022-1033`.
- `crates/cb-train/src/tree.rs:1059-…` — `select_level_perturbed`; the fused
  pass at `:1095-…`, then the SERIAL feature-ascending RNG passes.
- `crates/cb-train/src/tree.rs:312-323` — **`select_best_candidate`**:
  `let mut best_score = MINIMAL_SCORE; … if candidate.score > best_score`
  (STRICT `>`, first-wins). **9 callers**; covered by
  `tree_tie_break_test.rs` and `tree_test.rs`.
- `crates/cb-train/src/tree.rs:697-714` — `GrowScratch { bins, leaf_of,
  feature_hists, n_bins, approx_dim, level }`.
- `crates/cb-train/src/tree.rs:722-770` — `GrowScratch::new`.
- `crates/cb-train/src/tree.rs:779-790` — `advance_leaf_only(&mut self, matrix,
  split: &Split, n_objects)`.
- `crates/cb-train/src/tree.rs:395-400` — `FeatureMatrix::passes_any`.
- `crates/cb-train/src/tree.rs:574-662` — `greedy_tensor_search_oblivious_perturbed`
  (RSM draws `:610-614`, dispatch `:628-647`, `GrownTree` build `:651-661`).
- `crates/cb-train/src/tree.rs:3077-3130` — `select_level_one_hot`: the frozen
  reference for the ENUMERATION ORDER.
- `crates/cb-train/src/tree.rs:849` — `derive_feature_level_hist` (the
  subtraction trick, reused verbatim for cat features).
- `crates/cb-train/src/tree.rs:92-105` — the four `tree_*` mounts
  (`mod general;`, `mod tie_break;`, …).

**Red.**
- File: `crates/cb-train/src/tree_one_hot_fused_test.rs` (new; mount after
  `tree.rs:105` as `#[path = "tree_one_hot_fused_test.rs"]
  mod tree_one_hot_fused_test;` — the `mod <file_stem>` rule, so
  `tree::tree_one_hot_fused_test` is a working filter).
- Fn 1: `fused_one_hot_search_matches_the_frozen_reference_grower`. Setup: 2
  float columns (3 borders each) + 2 cat bin columns (cardinality 2 and 3), 200
  objects, deterministic der1/weight, `depth = 3`, `perturb = None`,
  `score_function = L2`, `scaled_l2 = 3.0`. Expected: the reconstructed
  level-ordered `Vec<AnySplit>` equals `grow_one_hot_tree(...).splits` EXACTLY
  and `leaf_of` matches element-for-element. Initial failure: the fused path
  emits no one-hot split → the lists differ at level 0.
- Fn 2: `fused_one_hot_search_selects_the_last_bin_when_it_wins` — the CPU twin
  of the device [C2] guard: construct data where the HIGHEST cat bin is the
  winning equality value.
- Fn 3 (in **`crates/cb-train/src/tree_tie_break_test.rs`**, the existing
  `mod tie_break;` sibling): `float_and_one_hot_candidates_tie_breaks_to_float`
  — an exact score tie between a float border and a one-hot value must select
  the FLOAT (it is enumerated first). This is the MAJOR-6 guard.
- Fn 4 (Branch A only): `one_hot_perturbed_level_consumes_the_ground_truth_draws`.
  (Branch B: instead assert one-hot candidates are never enumerated while
  `perturb.is_some()`, because T01b's gate fires first.)

**Green (minimal).**
1. Extend `GrowScratch` with `cat_bins: Vec<u32>` (cat-major, built like `bins`
   at `:735-751`), `cat_n_bins: usize` (max cardinality; `1` when none), and
   `cat_hists: Vec<Option<BucketHistogram>>`.
2. In `select_level_plain`, after the float `per_feature` pass, add a SECOND
   rayon `map_init` pass over `0..matrix.n_cat_features()` emitting one
   candidate per **distinct bin** (`distinct_bins_ascending`, matching the CPU
   reference exactly — the device must mirror this, see T25/MAJOR-3), scored as
   `left = bin_sums[value]`, `right = total - left` through the SAME
   `split_score` calcer. Change the return type to
   `(AnySplit, Vec<Option<BucketHistogram>>, Vec<Option<BucketHistogram>>)`.
3. Mirror the insertion in `select_level_perturbed`, keeping every RNG draw in
   the SERIAL feature-ascending passes. **Draw-count behavior is dictated by
   T01b's branch:** Branch A → iterate `rsm_listed_feature_count(matrix)` in
   BOTH the RSM loop and the `SelectBestCandidate` pass; Branch B → leave both
   iterating `matrix.n_features()` **unchanged** (one-hot + `perturb.is_some()`
   is unreachable, gated in T01b) and add the assertion of Red fn 4.
4. Widen `advance_leaf_only` to `&AnySplit` (use `passes_any`).
5. In `greedy_tensor_search_oblivious_perturbed`, collect `Vec<AnySplit>`, then
   split back into `splits` / `one_hot_splits` / `level_kinds` using the
   `tree.rs:2955-2997` pattern; `leaf_of = assign_leaves_any(...)`.
6. **The argmax MUST be `select_best_candidate`, generalized** — e.g.
   `trait HasScore { fn score(&self) -> f64; }` plus
   `fn select_best<T: HasScore>(cands: &[T]) -> Option<&T>` seeded with
   `MINIMAL_SCORE` and using strict `>`; `Candidate` and the new
   `LevelCandidate { Float(Candidate), OneHot { feature, value, score } }` both
   implement it. **Do NOT hand-roll a second scan** — that is the drift class
   §1 Q2 warns about (MAJOR-6). `select_best_candidate`'s existing 9 callers
   must keep compiling unchanged.

**Refactor.** Reuse `derive_feature_level_hist` (`tree.rs:849`) for the cat
subtraction trick — do not write a second derivation. Delete nothing:
`grow_one_hot_tree` / `select_level_one_hot` / `score_candidate_any` remain the
frozen reference. Regression scope: all of `cb-train`, plus
`spike006_fused_parallel_test`, `rayon_determinism_test`, `perf_baseline_test`,
and T00's byte-identity gate.

**Validation.**
```
cargo test -p cb-train --lib tree::tree_one_hot_fused_test
cargo test -p cb-train --lib tree::tie_break
cargo test -p cb-train --lib tree::general
cargo test -p cb-train --test rayon_determinism_test
cargo test -p cb-train --test spike006_fused_parallel_test
cargo test -p cb-train --test perf_baseline_test
cargo test -p cb-train
cargo test -p cb-model --test float_only_byte_identity_test
cargo clippy --workspace --all-targets -- -D warnings
```

**Completion evidence.** The fused-vs-reference equality, the float-first
tie-break assertion, and `perf_baseline_test` showing no float-only regression.

**Parallelization.** NONE (owns `tree.rs`).

---

### T19 — SPEC-OH-07 — one-hot splits are persisted on the trained tree

**Goal.** A level that selected a one-hot split lands in
`ObliviousTree.one_hot_splits` with `LevelKind::OneHot` recorded at that level.

**Observable completion.** After a `train_cat` fit on a binary categorical
column with `one_hot_max_size = 2`, some tree has `one_hot_splits.len() >= 1`
and `level_kinds` containing `LevelKind::OneHot { .. }`.

**Blocking / prerequisites.** T18, T03.

**Verified files and symbols.**
- `crates/cb-train/src/tree.rs:256-273` — `enum LevelKind` (two variants today).
- `crates/cb-train/src/tree.rs:205-254` — `GrownTree` (needs `one_hot_splits`).
- `crates/cb-train/src/boosting.rs:4042-4200` — the grower dispatch; plain arm at
  `:4185-4195`.
- `crates/cb-train/src/boosting.rs:4786-4824` — the persist site
  (`oblivious_from_grown` after T03).
- `crates/cb-train/src/boosting.rs:4226-4232` — the FEAT-04 `used_features` loop
  over `grown.splits`: one-hot splits must NOT mark float-feature uses.

**Red (genuine — this is SPEC.md §1's headline defect made executable, A1).**
- File: `crates/cb-train/tests/one_hot_oracle_test.rs`.
- Test fn: `train_cat_produces_one_hot_splits_for_a_binary_categorical_column`.
- Setup: `train_cat` with 1 float column + 1 binary cat column,
  `one_hot_max_size = 2`, `depth = 2`, `iterations = 3`, `bootstrap_type = No`,
  `random_strength = 0.0`.
- Expected: `model.oblivious_trees.iter().any(|t| !t.one_hot_splits.is_empty())`
  and the matching `LevelKind::OneHot` entries.
- Expected initial failure: `one_hot_splits` is always empty — nothing populates
  it, i.e. the column is silently dropped.

**Green (minimal).**
1. Add `LevelKind::OneHot { one_hot_idx: usize }` and
   `GrownTree.one_hot_splits: Vec<OneHotSplit>` (documented EMPTY for every
   non-one-hot search).
2. Populate both in T18's split-back.
3. Thread `grown.one_hot_splits` through `oblivious_from_grown`.
4. Guard `boosting.rs:4226-4232` so only float splits mark `used_features`.

**Refactor.** `LevelKind` gains a third variant — every `match` on it becomes a
compile error, the desired forcing function. Regression scope: full `cb-train`,
full `cb-model`, T00's gate.

**Validation.**
```
cargo test -p cb-train --test one_hot_oracle_test
cargo test -p cb-train
cargo test -p cb-model
cargo test -p cb-model --test float_only_byte_identity_test
cargo build --workspace --all-targets
```

**Parallelization.** NONE.

---

### T20 — SPEC-OH-26 — one-hot × CTR in one pool is explicitly gated

**Goal.** A pool with BOTH one-hot-routed and CTR-routed cat columns either
trains correctly or returns a typed error naming the combination — never a
silent drop. **Device-side CTR co-existence is DEFERRED** (SPEC §9 R12); this
ships only the honest gate.

**Observable completion.** `train_cat` on such a pool returns
`CbError::Unsupported` naming both encodings.

**Blocking / prerequisites.** T19.

**Verified files and symbols.**
- `crates/cb-train/src/boosting.rs:2721-2729` — the two partitions (T16).
- `crates/cb-train/src/boosting.rs:4042-4200` — the dispatch: `has_ctr` selects
  `greedy_tensor_search_oblivious_with_ctr` (`:4140-4141`), else the plain
  perturbed arm (`:4185-4195`). There is **no three-way candidate union**.
- `crates/cb-train/src/tree.rs:2919-3009` — the CTR search takes no
  `cat_bins`/one-hot candidates.
- `crates/cb-compute/src/runtime.rs:1148-1161` — `is_covered_regime()` requires
  `ctr.is_none()`.
- `crates/cb-core/src/error.rs:86-92` — `CbError::Unsupported(String)`.

**MANDATORY pre-implementation check (PLAN-CHECK MINOR-4).** Before writing the
gate, enumerate EVERY test and fixture whose cat cardinalities span both routes
under its own `one_hot_max_size`, and assert the gate does not fire for any of
them. Verified starting points:
- `crates/cb-train/tests/one_hot_oracle_test.rs:217` pins
  `one_hot_max_size: 3`, and its doc at `:30` describes a deliberately **mixed**
  pool ("cat0 cardinality == one_hot_max_size → one-hot, cat1 == +1 → CTR") —
  exactly what this gate rejects. It survives today only because the test-local
  driver bypasses `train_inner`. **T22 rewrites that file**, so schedule the
  check to confirm the rewritten test does not trip the gate (or splits into two
  single-route pools).
- **Verified safe:** `tensor_ctr_e2e_oracle_test.rs:106` and
  `plain_ctr_oracle_test` pin `one_hot_max_size: 1` (nothing routes one-hot);
  `fstr_ctr` / `ctr_load` cardinalities are `[5, 4]`.
- Record the enumeration in the commit body.

**Red (genuine).**
- File: `crates/cb-train/tests/one_hot_oracle_test.rs`.
- Test fn: `one_hot_plus_ctr_pool_is_typed_rejected_not_silently_dropped`.
- Setup: `train_cat` with cat cardinalities `[2, 20]`, `one_hot_max_size = 2`.
- Expected: `Err(CbError::Unsupported(msg))` with `msg.contains("one-hot") &&
  msg.contains("CTR")`.
- Expected initial failure: `Ok(model)` in which the one-hot column silently
  contributes nothing — **assert
  `model.oblivious_trees.iter().all(|t| t.one_hot_splits.is_empty())` first** to
  document the silent drop, then flip.

**Green (minimal).** In `train_inner`, after T16's partition: if BOTH lists are
non-empty, return `CbError::Unsupported("training a pool with both
one-hot-routed and CTR-routed categorical columns is not yet supported
(device-side CTR co-existence is deferred): raise one_hot_max_size to route all
columns one-hot, or lower it to route all columns to CTR")`.

**Refactor.** Place the gate where BOTH lists are in scope so no future dispatch
arm bypasses it. Regression scope: every CTR oracle — assert explicitly that the
gate does NOT fire for them.

**Validation.**
```
cargo test -p cb-train --test one_hot_oracle_test
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test
cargo test -p cb-train --test ordered_ctr_oracle_test
cargo test -p cb-train --test s_order_ctr_bins_oracle_test
cargo test -p cb-model --test fstr_ctr_oracle_test
cargo test -p cb-train
```

**Parallelization.** NONE.

---

### T21 — SPEC-OH-09 — bin → raw-hash mapping at the lift (the §1.2 landmine)

**Goal.** The emitted `ModelSplit::OneHot.value_hash` is the ORIGINAL
`calc_cat_feature_hash` i32 for that bin — never the `PerfectHash` bin ordinal.

**Observable completion.** For a known raw category string, the emitted
`value_hash` equals `calc_cat_feature_hash(raw) as i32` computed
**independently in the test**.

**Blocking / prerequisites.** T19, T04, T05, **T17** (which now owns the
`one_hot_bin_to_hash` / `one_hot_absolute` fields — MAJOR-10). **This task edits
`cb-model/src/model.rs` ONLY; it is a read-only consumer of `boosting.rs`.**

**Verified files and symbols.**
- `crates/cb-train/src/tree.rs:125-136` — `OneHotSplit { feature: usize, value:
  u32 }` where `value` is a `PerfectHash` **bin** (the doc says so).
- `crates/cb-data/src/cat_hash.rs:362-365` — `calc_cat_feature_hash -> u32`
  [C3: `as i32` at the boundary].
- `crates/cb-model/src/model.rs:326-426` — `from_trained`, rewritten in T04 to
  walk `level_kinds`.
- `crates/cb-model/src/model.rs:399-408` — **[C15] WARNING:** `RegionLevel {
  one_hot }` is an UNRELATED flag. Do not conflate.
- `cb_train::Model.one_hot_bin_to_hash` / `.one_hot_absolute` — added by T17.

**Red (genuine).**
- File: `crates/cb-model/tests/one_hot_value_space_test.rs` (new integration
  test — it must depend on `cb_data` to compute the hash independently of the
  lift).
- Test fn: `lifted_one_hot_value_hash_equals_an_independently_computed_cat_hash`.
- Setup: `train_cat` on **two** cat columns where only the SECOND is
  one-hot-routed (so an absolute-vs-position index bug is visible), values e.g.
  `["alpha","beta","alpha", …]`; lift with `Model::from_trained`.
- Expected: for the emitted `ModelSplit::OneHot`,
  `value_hash == cb_data::calc_cat_feature_hash("alpha") as i32` (or `"beta"`,
  whichever won), computed IN the test — never read back out of the model; AND
  `value_hash != 0 && value_hash != 1` (**not** a bin ordinal); AND
  `cat_feature == 1` (the ABSOLUTE cat index, not the one-hot position).
- Expected initial failure: no `ModelSplit::OneHot` exists in the model (T04's
  level walk has no one-hot arm); after a naive arm, `value_hash == 0`.

**Green (minimal).**
1. Add the `LevelKind::OneHot` arm to `from_trained`'s level walk:
   `ModelSplit::OneHot(OneHotModelSplit {
      cat_feature: trained.one_hot_absolute[col],
      value_hash: trained.one_hot_bin_to_hash[col][split.value as usize] as i32 })`.
2. **The lookup cannot miss — T17's construction-time assertion IS the
   guarantee** (PLAN-CHECK pass-2 MINOR-a). T17 asserts
   `hash_by_bin[c].len() == cardinality(c)` when the table is built and returns
   `CbError::Degenerate` otherwise, so by the time `from_trained` runs, every
   `(col, bin)` a `LevelKind::OneHot` can name is in range. Therefore:
   - **Do NOT** add a "defensive skip" — that is a silent drop.
   - **Do NOT** "return the split built from some other value the lookup did
     produce" — that is a silently WRONG split. *(This option appeared in
     revision 2 and is deleted: it directly contradicts SPEC §1.3, which exists
     to eliminate exactly this class.)*
   - **Do** add a `debug_assert!` documenting the invariant and its owner
     (T17), and use bounds-checked `.get()` whose `None` branch is
     **unreachable by construction**; if the implementer wants a hard stop
     there, the only acceptable form is a typed failure, never a fabricated
     split.
   `from_trained` therefore stays `-> Self` and its 24 callers are untouched.
3. `cat_feature` MUST be the ABSOLUTE `cat_columns` index — `passes_split` (T06)
   indexes `cat_values` by absolute cat index.

**Refactor.** Cross-reference `cb_train::OneHotSplit.value` (bin space) from
`OneHotModelSplit.value_hash`'s doc (hash space). Regression scope: `cb-model`
full, `cb-train` full.

**Validation.**
```
cargo test -p cb-model --test one_hot_value_space_test
cargo test -p cb-model
cargo test -p cb-train
cargo build --workspace --all-targets
```

**Completion evidence.** The independently computed hash and the emitted
`value_hash` printed side by side; and a one-line confirmation that no
`from_trained` caller can construct a `cb_train::Model` bypassing T17's
table-construction assertion (B10).

**Parallelization.** NONE (owns `model.rs`).

---

### T22 — SPEC-OH-28 — the oracle runs through production code

**Goal.** The one-hot oracle drives production `train_cat` against the committed
upstream fixture at ≤1e-5, and the test-local driver is DELETED.

**Observable completion.** `one_hot_oracle_test.rs` contains no
`train_one_hot_only`, no `one_hot_only_scenario`, no `one_hot_encode` and no
direct `grow_one_hot_tree` call; the upstream comparison passes at ≤1e-5.

**Blocking / prerequisites.** T21, T02, **T02b** (without it `load_model_json`
fails on the fixture — PLAN-CHECK CRITICAL-1), T20.

**Verified files and symbols.**
- `crates/cb-train/tests/one_hot_oracle_test.rs` — 320 lines; the test-local
  driver `train_one_hot_only(...)` at `:90-169`; `grow_one_hot_tree` calls at
  `:54` and `:139`; `one_hot_encode` at `:76-88`; routing unit assertions at
  `:178`; `one_hot_max_size: 3` at `:217`; the self-oracle at `:188-276`.
- **`crates/cb-train/tests/one_hot_oracle_test.rs:284-292`** — verbatim:
  ```
  fn no_permutation_in_one_hot_only_path() {
      let (cat_bins, target) = one_hot_only_scenario();
      let a = train_one_hot_only(&cat_bins, &target, 3, 4, 2, 0.3, 3.0);
      let b = train_one_hot_only(&cat_bins, &target, 3, 4, 2, 0.3, 3.0);
  ```
  **This retained test IS the driver's only remaining consumer** — revision 1's
  "delete the driver, keep this test" was a contradiction (PLAN-CHECK MAJOR-11).
- `crates/cb-train/src/lib.rs:107` — the `grow_one_hot_tree` re-export (**KEEP**;
  `tree_test.rs:67,83,118,130` still exercises it as the frozen reference).
- `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs:215-219` — the mandated
  pattern: `load_model_json(&fixture(".../model.json"))` →
  `.float_feature_borders()`.
- `crates/cb-oracle/src/compare.rs` — `compare_stage`, `Stage::StagedApprox` /
  `Stage::Predictions` (used at `one_hot_oracle_test.rs:188-276` today).

**Red.**
- File: `crates/cb-train/tests/one_hot_oracle_test.rs` (rewritten).
- Test fn: `one_hot_train_matches_upstream_within_1e5`.
- Setup: load `one_hot_train/<scenario>/{X_float.npy, cat_cols.json, y.npy,
  config.json, preds.npy, model.json}`; take borders from
  `load_model_json(...).float_feature_borders()` (**the SPEC-OH-28 mandate**,
  enabled by T02b); drive `cb_train::train_cat` with the config's params; lift
  with `Model::from_trained`; predict with `cb_model::predict_raw_cat`.
- Expected: `max |ours - upstream| <= 1e-5` on both `StagedApprox` and
  `Predictions`, AND `model.oblivious_trees.iter().flat_map(|t| &t.splits)
  .any(|s| matches!(s, ModelSplit::OneHot(_)))`.
- Expected initial failure: the test does not exist / the fixture is not wired.

**Green (minimal).**
1. Write the oracle test above.
2. **DELETE** `train_one_hot_only`, `one_hot_only_scenario` and `one_hot_encode`
   (SPEC-OH-28: "the existing test-local driver is deleted, not left
   alongside").
3. **RE-POINT `no_permutation_in_one_hot_only_path` at production `train_cat`**
   (PLAN-CHECK MAJOR-11): two identical `train_cat` fits on the same one-hot
   pool must produce byte-identical models. The determinism guarantee is
   preserved, on the production path, without the driver.
4. Keep `one_hot_path_selection_boundary` (pure `route_categorical` unit
   assertions — no driver dependency).
5. Reconcile `one_hot_max_size: 3` at `:217`: the rewritten tests must use a
   single-route pool, or split into two, so T20's gate does not fire. Record
   which was chosen.

**Refactor.** Any residual gap > 1e-5 is a real defect in T18/T21 and must be
fixed there, not tolerated. Regression scope: `cargo test -p cb-train`.

**Validation.**
```
cargo test -p cb-train --test one_hot_oracle_test
rg -n "train_one_hot_only|one_hot_only_scenario|one_hot_encode|grow_one_hot_tree" crates/cb-train/tests/
   # must be EMPTY
rg -n "grow_one_hot_tree" crates/cb-train/src/    # must still show :3141, lib.rs:107, tree_test.rs
cargo test -p cb-train
```

**Completion evidence.** The measured `max|diff|` per stage; the empty `rg`
output over `tests/`; the non-empty `rg` over `src/` proving the frozen
reference survived.

**Parallelization.** NONE.

---

### T23 — SPEC-OH-20 — a cat-only pool is device-eligible

**Goal.** Lift the `matrix.n_features() > 0` precondition so a pool with 0 float
features and ≥1 one-hot column reaches the GPU. Every other precondition
unchanged. **And name the second, session-level blocker** so SPEC-OH-20's
0-float target is genuinely reachable (PLAN-CHECK MINOR-7 / C11).

**Observable completion.** For a cat-only pool with a one-hot column,
`device_host_eligible == true`; for a genuinely feature-less pool (0 float AND
0 one-hot) it stays `false`; and the session-level `n_features == 0 || n_bins ==
0` decline is documented as satisfied by T24's concatenated axis.

**Blocking / prerequisites.** T16.

**Verified files and symbols.**
- `crates/cb-train/src/boosting.rs:3055-3122` — the 14-clause
  `device_host_eligible`. **Clause 11 is `matrix.n_features() > 0` at `:3100`,
  present exactly once.** Clause 3
  (`materialized_ctr_features.is_empty() && structure_fold_columns.iter().all(
  Vec::is_empty)`) at `:3057-3058` is the CTR blocker SPEC-OH-26 leaves in place.
- `crates/cb-train/src/tree.rs:355-363` — `n_features()` (float) vs
  `n_cat_features()`.
- **`crates/cb-backend/src/gpu_runtime/session.rs:1250-1253`** — `GpuTrainSession
  ::begin` ALSO declines on `if n == 0 || n_features == 0 || n_bins == 0 {
  return Ok(None); }`. Under T24's layout `n_features` is the TOTAL
  (`n_float + n_cat`) and `n_bins = max(float n_bins, max cat cardinality)`, so
  a cat-only pool passes — **T24 must guarantee both, and T23 records the
  dependency here.**
- **`crates/cb-backend/src/gpu_runtime/session.rs:1270-1274`** — the padding
  site: `let n_bins_line = if nonsym_policy.is_none() && !region_active { match
  pad_hist_line_bins(n_bins) { Some(w) => w, None => return Ok(None) } } else {…}`;
  the `{32,64,128,256}` family rejection is at `gpu_runtime/mod.rs:2392-2403`.
  A cardinality-2 one-hot pool pads to `n_bins_line == 32`, which is legal.
- `crates/cb-train/src/boosting.rs:4954-4959` — `mod tests;` and
  `mod boosting_device_fold_tests;`.

**Red.**
- File: `crates/cb-train/src/boosting_device_fold_test.rs` (existing mount →
  filter `boosting::boosting_device_fold_tests`).
- Test fn: `cat_only_pool_with_one_hot_columns_is_device_eligible`.
- Setup: extract clause 11 into
  `fn has_any_scorable_feature(matrix: &FeatureMatrix) -> bool` and test it
  directly (the full expression needs a whole fit context).
- Input: (a) 0 float + 1 cat; (b) 2 float + 0 cat; (c) 0 float + 0 cat.
- Expected: `true`, `true`, `false`.
- Expected initial failure: case (a) returns `false`.

**Green (minimal).** Replace `matrix.n_features() > 0` at `:3100` with
`has_any_scorable_feature(&matrix)` = `matrix.n_features() > 0 ||
matrix.n_cat_features() > 0`. **Touch NO other clause.**

**Refactor.** Add a comment at `:3100` recording (a) that clause 3 (CTR) is
deliberately NOT lifted (SPEC §9 R12, SPEC-OH-26), and (b) the session-level
`n_features == 0 || n_bins == 0` decline at `session.rs:1250-1253` plus the
`pad_hist_line_bins` site at `:1270-1274`, so the next reader knows where the
0-float path is actually decided.

**Validation.**
```
cargo test -p cb-train --lib boosting::boosting_device_fold_tests
cargo test -p cb-train --test device_seam_test
cargo test -p cb-train --test device_oblivious_parity_probe_test
cargo test -p cb-train
cargo test -p cb-train --features rocm --no-run
```

**Parallelization.** NONE.

---

### T24 — SPEC-OH-21 — device quantization emits one-hot bin columns, `one_hot` flags AND per-feature `folds`

**Goal.** One-hot columns become device bin columns in the shared uniform
`n_bins` line; `TCFeature.one_hot_feature` is set TRUTHFULLY; **and a SEPARATE
per-feature real-cardinality array `real_folds` is produced**, because without
it the device scorer cannot bound one-hot candidates to real bins
(PLAN-CHECK pass-1 MAJOR-3) — and because `TCFeature.folds` cannot serve that
role (PLAN-CHECK pass-3 MAJOR-1 / correction [C16]).

**Observable completion.** `quantize_*` emits `n_float + n_one_hot` columns with
`n_bins = max(float n_bins, max cat cardinality)` **and a separate `real_folds`
per-feature cardinality array**; `pack_cindex` marks exactly the one-hot
columns; `PackedCindex::device_arrays()` exports **four** arrays
`(offsets, shifts, masks, one_hot_flags)`; and a cat-only pool reaches the fill
with a legal padded line width.

**Blocking / prerequisites.** T23.

**Verified files and symbols.**
- `crates/cb-train/src/boosting.rs:2196-2233` — `quantize_feature_major(
  feature_values, feature_borders, n) -> (Vec<u32>, usize)`; `n_bins =
  fold(0, |acc, b| acc.max(b.len() + 1))` at `:2203-2205` — **`0` for a pool with
  zero float features**, which `session.rs:1252` rejects. The call site is
  `boosting.rs:3129`.
- `crates/cb-backend/src/gpu_runtime/cindex.rs:46-75` — `TCFeature { offset,
  mask, shift, first_fold_index, folds, one_hot_feature }`, doc'd as
  "`one_hot_feature` selects EQUALITY (`== value`) vs THRESHOLD (`> bin`) split
  semantics downstream". **`#[allow(dead_code)]` is on the STRUCT (`:59`), not on
  the field** — `first_fold_index` (`:70`) is also unused [MINOR-1].
- `crates/cb-backend/src/gpu_runtime/cindex.rs:88-110` —
  `device_arrays() -> CbResult<(Vec<u32>, Vec<u32>, Vec<u32>)>`, **4 callers**.
- `crates/cb-backend/src/gpu_runtime/cindex.rs:211-228` — the descriptor emit;
  `folds` = per-feature `n_buckets` at `:217-219`; **`one_hot_feature: false`
  hard-coded at `:226`**.
- `crates/cb-backend/src/gpu_runtime/cindex.rs:117-129` — `feature_bits(
  n_buckets)`.
- `crates/cb-backend/src/gpu_runtime/cindex.rs:154` — `pack_cindex`,
  **13 callers** across `kernels/pointwise_hist.rs`, `kernels/cindex.rs`,
  `gpu_runtime/mod.rs`, `gpu_runtime/session.rs` [MAJOR-4].
- `crates/cb-backend/src/gpu_runtime/session.rs:1250-1253` (`n_features == 0 ||
  n_bins == 0` decline), `:1270-1274` (`pad_hist_line_bins`), `:1360-1363`
  (`let n_buckets_per_feature = vec![n_bins_line; eff_n_features];` — the
  packing width).
- `crates/cb-backend/src/gpu_runtime/mod.rs:2392-2403` — the `{32,64,128,256}`
  family gate.
- SPEC §9 R10: bound one-hot cardinality on the device **or fall back**.

**Red (five fns).**
- Files: a new `crates/cb-backend/src/gpu_runtime/cindex_one_hot_test.rs`
  (mount as `#[path = "cindex_one_hot_test.rs"] mod cindex_one_hot_test;` →
  filter `gpu_runtime::cindex::cindex_one_hot_test`) and
  `crates/cb-train/src/boosting_device_fold_test.rs`.
- Fn 1 `packed_cindex_marks_one_hot_features_truthfully`: `pack_cindex`
  over 2 float columns + 1 cat column with the one-hot flag set for index 2.
  Expected: `features[2].one_hot_feature == true`, `[0..2]` false;
  `device_arrays()` returns a **4**-tuple whose 4th element is `[0, 0, 1]`.
  Initial failure: 3-tuple arity (`E0308`) and `one_hot_feature` always `false`.
- **Fn 1b `packed_cindex_folds_is_the_padded_line_width_not_the_cardinality`**
  (the [C16] / pass-3 MAJOR-1 pin): call `pack_cindex` the way production does —
  `n_buckets_per_feature = vec![32; n_features]` (`session.rs:1363`) — over a
  cardinality-2 cat column, and assert `features[c].folds == 32`, **not** `2`.
  This test **documents that `TCFeature.folds` must never be used as a candidate
  bound** and fails loudly if a future change repurposes it. It has no Red→Green
  cycle of its own (it passes today); it is a **standing pin**, and its doc
  comment must say so.
- Fn 2 `quantize_emits_one_hot_bin_columns_in_the_shared_bin_line`: expected
  `bins.len() == (n_float + n_cat) * n`, `n_bins == max(float n_bins, max cat
  cardinality)`, and the cat stripe equals the `PerfectHash` bin column verbatim
  (no re-binning).
- Fn 3 `cat_only_pool_yields_a_nonzero_n_bins_and_a_legal_padded_line`
  (MINOR-7): 0 float + 1 cardinality-2 cat column. Expected `n_bins == 2`
  (**not `0`**) and `pad_hist_line_bins(2) == Some(32)`. Initial failure:
  `n_bins == 0` → `session.rs:1252` declines and SPEC-OH-20's 0-float target is
  unreachable.
- Fn 4 `one_hot_cardinality_above_the_device_bound_falls_back_to_the_cpu_grower`
  — see the decision below.

**Green (minimal).**
1. `quantize_feature_major_with_one_hot(feature_values, feature_borders,
   cat_bins, n) -> (Vec<u32>, usize, Vec<u32>)`: append the cat stripes AFTER
   the float stripes (so device feature index `n_float + c` is one-hot column
   `c` — the contiguous range T25 needs), compute
   `n_bins = max(float_n_bins, max_cat_cardinality).max(1)`, **and return the
   third value `real_folds` — see step 1b.**

   **Call-site rule (PLAN-CHECK pass-4 MAJOR-1a — read this before writing
   anything).** The device-quantize call site is **`boosting.rs:3129`**
   (`let (device_bins, device_n_bins) = if device_host_eligible { … }`), and
   there is exactly ONE. It **ALWAYS calls
   `quantize_feature_major_with_one_hot`**, on every device-eligible pool —
   passing an **empty `cat_bins` slice** when the pool has no one-hot columns.
   It therefore **ALWAYS populates `real_folds` to length `n_features`**,
   including on a float-only pool, where the value is `[borders[f].len()+1, …]`.
   `real_folds` is **never empty on a device-eligible fit**.

   **What "UNCHANGED" means here (the ambiguity pass-4 flagged).**
   `quantize_feature_major` (`boosting.rs:2196-2233`) is retained **with its
   body and signature unmodified** and is **delegated to** by the new function
   for the float prefix — so the float bin bytes are provably identical
   (SPEC-OH-31). It does **NOT** mean "the float-only path keeps calling the old
   2-tuple function". Reading it that way would leave nothing producing
   `real_folds` on a float-only fit, and T27b's unconditional
   `real_folds.len() == eff_n_features` assertion would then fail **every**
   float-only device fit — breaking `device_oblivious_parity_probe_test`,
   `device_bootstrap_parity_test`, `device_seam_test`, `bootstrap_dev_oracle_test`,
   `device_poisson_bootstrap_test`, `device_nonsym_fit_test`,
   `device_region_fit_test` and T29b (SPEC-OH-31's device half). **That reading
   is wrong and is forbidden.** T24 Red fn 5 pins the correct behavior by
   asserting `real_folds == [borders+1 …]` on a float-only pool.
1b. **`real_folds`: the per-feature REAL cardinality array (PLAN-CHECK pass-3
   MAJOR-1 / [C16]).** This is a **NEW, SEPARATE** host-side array — it is
   **NOT** `TCFeature.folds` and it does **NOT** come from `pack_cindex`:
   - float feature `f` → `feature_borders[f].len() + 1`;
   - one-hot column `c` (device index `n_float + c`) → that column's
     `PerfectHash` cardinality, i.e. `one_hot_bin_to_hash[c].len()` (T17 already
     asserts this equals the column's true cardinality);
   - length `n_float + n_cat`, element type `u32` (the device index type).
   **Two prohibitions, both load-bearing:**
   - **Do NOT reuse `TCFeature.folds`.** On the production path it is the padded
     uniform line width: `session.rs:1363` passes
     `vec![n_bins_line; eff_n_features]` into `pack_cindex`, and
     `cindex.rs:213-227` copies that argument straight into `folds`. So
     `folds[f] == n_bins_line` for every feature and `border < folds[feature]`
     is the loop bound itself — **no bound at all** ([C16]).
   - **Do NOT "fix" it by changing `n_buckets_per_feature`.**
     `pack_cindex`'s placement pass (`cindex.rs:181-200`) derives
     `bits = feature_bits(nb)` and hence `(group, shift, mask)` from that
     argument, so true cardinalities there would change the packed words for
     **every** pool including float-only, breaking T29b fn 1's frozen
     `packed_cindex.json`, the `kernels/cindex.rs` pack→read oracle, and the
     `read_bin` descriptors `launch_partition_split_packed_into` consumes.
   **Float-only invariance (must be shown, not assumed).** For a pool with no
   cat columns, `real_folds[f] == borders[f].len() + 1` — a *pure addition* that
   no existing launch reads: the scorer consults it only under the comptime
   `one_hot == true` arm (T25 constraint 3), and pass A / every float-only
   launch takes the `one_hot == false` arm whose eligibility stays
   `border < max_border`, byte-for-byte today's expression. Fn 5 below asserts
   this explicitly.
2. `pack_cindex` gains `one_hot: &[bool]` (length `n_features`) and sets
   `TCFeature.one_hot_feature` from it. **All 13 call sites must pass an
   all-`false` slice** except the oblivious one-hot path — the byte-unchanged
   default (MAJOR-4). **`n_buckets_per_feature` is NOT touched** (step 1b).
3. `device_arrays()` returns
   `(offsets, shifts, masks, one_hot_flags)` — a **4**-tuple; `one_hot_flags:
   Vec<u32>` (0/1) because the device index type is `u32`. Update all 4 callers.
   **`folds` is deliberately NOT added to this tuple**: `TCFeature.folds` is the
   padded line width and must never be used as a candidate bound ([C16]); the
   real bound travels as `real_folds` from step 1b via `DeviceTrainConfig`
   (T27b), not through `PackedCindex`. Add a doc line on `TCFeature.folds`
   saying exactly that, so the next reader does not repeat the mistake.
4. **`#[allow(dead_code)]` (pass-1 MINOR-1, corrected by pass-4 MINOR-2).**
   The allow sits on the STRUCT (`cindex.rs:59`). After this task
   **`one_hot_feature` becomes read** (via `device_arrays()`), but
   **`folds` and `first_fold_index` BOTH remain unread in the lib target** —
   verified across all of `crates/cb-backend/src/`, their only occurrences are
   the doc comments at `cindex.rs:48`/`:54` and the two writes at
   `:224`/`:225`. Step 3 deliberately removed `folds` from the tuple, so it is
   NOT consumed. Therefore **narrow the allow to BOTH `first_fold_index` AND
   `folds`** (two field-level `#[allow(dead_code)]`s), or keep the struct-level
   allow with an updated comment naming exactly those two as contract-only
   fields. **Narrowing to `first_fold_index` alone fails
   `cargo clippy --workspace --all-targets -- -D warnings` with
   `field is never read: folds`** — T24 Red fn 1b reads `folds`, but only under
   `#[cfg(test)]`, and the lib target is compiled without it.
5. **[SPEC §9 R10 / PLAN-CHECK "Potential Bugs"] Decision: FALLBACK, not abort.**
   SPEC §9 R10 says "Bound … **or fall back**", and returning
   `CbError::Unsupported` from `quantize_*` would ABORT the fit — a different,
   worse behavior. Implement it as an added **`device_host_eligible` clause** in
   `boosting.rs`: `&& one_hot_cardinalities.iter().all(|&c| c <=
   DEVICE_ONE_HOT_MAX_CARDINALITY)`, with
   `pub const DEVICE_ONE_HOT_MAX_CARDINALITY: usize` set so the padded line stays
   within the `{32,64,128,256}` family alongside the float bins. Fn 4 asserts the
   pool falls back to the CPU grower and still trains correctly — **not** that it
   errors.
6. **Fn 5 `real_folds_is_the_true_cardinality_and_float_only_is_unchanged`**
   (the pass-3 MAJOR-1 Red): for 2 float columns with 3 and 5 borders plus a
   cardinality-2 one-hot column, assert
   `real_folds == [4, 6, 2]` — and, on a **float-only** pool, assert both that
   `real_folds == [borders+1 …]` and that `(bins, n_bins)` are element-wise
   identical to plain `quantize_feature_major`'s output. Initial failure:
   `quantize_feature_major_with_one_hot` returns a 2-tuple (`E0308`).
   **Note:** `session.rs:1341` widens the feature axis with CTR columns
   (`eff_n_features`) when `ctr_is_covered`; SPEC-OH-26 gates one-hot × CTR, so
   in the one-hot regime `eff_n_features == n_features` — but T27b must still
   size/extend `real_folds` to `eff_n_features` (padding CTR columns with
   `n_bins_line`, which is inert because they are never one-hot).

**Refactor.** Regression scope: `cb-backend` full, `device_seam_test`,
`device_oblivious_parity_probe_test`, **`device_nonsym_fit_test`**,
**`device_region_fit_test`** (the out-of-scope growers touched by the 13-caller
sweep, MAJOR-4).

**Validation.**
```
cargo test -p cb-backend --lib gpu_runtime::cindex::cindex_one_hot_test
cargo test -p cb-backend
cargo test -p cb-train --lib boosting::boosting_device_fold_tests
cargo test -p cb-train --test device_seam_test
cargo test -p cb-train --test device_oblivious_parity_probe_test
cargo test -p cb-train --test device_nonsym_fit_test
cargo test -p cb-train --test device_region_fit_test
cargo test -p cb-train --features rocm --no-run
cargo clippy --workspace --all-targets -- -D warnings
```

**Parallelization.** NONE.

---

### T25 — SPEC-OH-22 — the split-scoring fold has a one-hot arm (BOTH scorers, BOTH exclusions)

**Goal.** With `one_hot == true`, the split scorer folds
`left = bin_sums[value]`, `right = total - left`, over **only the real bins of
that feature**, preserving candidate indexing, argmax and lowest-index
tie-break. With `one_hot == false` the scorer is numerically identical to today.
**The histogram FILL is unchanged.**

**Observable completion.** A device one-hot split score equals the CPU
`select_level_plain` one-hot candidate score to ≤1e-5 **at depth ≥ 2**; the
highest real bin CAN win **through `score_partition_over_binsums`**; padded /
non-existent bins can NEVER win; and the float-only scorer's
`best_idx`/`best_gain` are bit-identical to the pre-change values.

**Blocking / prerequisites.** T24 (needs `folds` + the one-hot flags).

**Verified files and symbols — read [C1], [C2] and MAJOR-3 before starting.**
- **`crates/cb-backend/src/kernels.rs:4506-4640`** —
  `find_optimal_split_partition_kernel`, the PRODUCTION resident scorer [C1].
  Key lines: `:4514` `#[comptime] n_bins`; `:4524`
  `let max_border = (n_bins_used as usize) - 1usize;`;
  **`:4525-4530`**
  ```
  let n_features_usize = n_features as usize;
  let n_parts_usize    = n_parts as usize;
  let n_candidates     = n_features_usize * n_bins_usize;
  let leaf_stride      = n_features_usize * n_bins_usize * 2usize;
  ```
  ← **the SAME `n_features` fixes BOTH the candidate bound and the per-partition
  row pitch** (CRITICAL-2); `:4548` `let border = c % n_bins_usize;`;
  `:4556-4581` the per-partition fold; `:4596-4604` `if border < max_border`.
- `crates/cb-backend/src/kernels.rs:3367-3500` — `find_optimal_split_kernel`
  (non-resident slice entry); fold `:3417-3436`; the `f32::MIN` sentinel
  rationale `:3387-3402`; the WR-05 trailing-border doc `:3355-3365`.
- `crates/cb-backend/src/gpu_runtime/mod.rs:1319-1460` —
  `launch_find_optimal_split_pointwise_into` (launches at `:1420`/`:1442`).
- `crates/cb-backend/src/gpu_runtime/mod.rs:2951-3130` —
  `score_partition_over_binsums`; kernel launches `:3047`/`:3069`; **the HOST
  BELT at `:3108-3112`**:
  ```
  // Trailing no-op border AND phantom padded borders (`border >= n_bins_used - 1`)
  // are all-LEFT no-op splits (the device kernel already excludes them; host belt, WR-05).
  if (cand as usize) % n_bins >= n_bins_used - 1 { continue; }
  ```
  followed by `let take = gain > best_gain || (gain == best_gain && cand <
  best_c);` — the lowest-index tie-break to preserve.
- `crates/cb-backend/src/gpu_runtime/mod.rs:2341` —
  `launch_partition_hist2_resident_into`: the histogram pitch spans the **full**
  concatenated feature axis.
- `crates/cb-backend/src/gpu_runtime/mod.rs:3930-3932` — the production call.
- `crates/cb-backend/src/kernels.rs:1107, 1140` —
  `pairwise_hist_nonbinary_kernel`'s existing `#[comptime] one_hot: bool` arm:
  the in-repo precedent.
- `crates/cb-train/src/tree.rs:3037-3046` — `distinct_bins_ascending`: the CPU
  reference enumerates **only bins actually present**.
- House rules: generics-float; if-as-STATEMENT only; grid-stride
  `CUBE_COUNT * CUBE_DIM`; `f32::MIN` sentinel (a literal `-inf` fails the
  gfx1100 HIP/comgr JIT). CubeCL manual
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`; on a
  build error consult
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`
  (a DIRECTORY — CLAUDE.md's `.md` filename is stale).

**Design (REVISED per CRITICAL-2, CRITICAL-3 and MAJOR-3).**

Two launches per level when one-hot columns exist:
- **pass A** — `one_hot = false`, `feature_lo = 0`, `feature_hi = n_float`;
- **pass B** — `one_hot = true`, `feature_lo = n_float`, `feature_hi = n_total`;
- the host finishes the argmax across the two winners with strict `>` and pass A
  first. **The cross-pass rule is exactly that: strict `>`, with pass A
  evaluated first, so a pass-B tie can never displace a float candidate** —
  which reproduces the CPU float-then-one-hot enumeration order. *(Precision
  note, PLAN-CHECK pass-3 MINOR-d: `cand < best_c` is the **per-pass**
  lowest-index tie-break living INSIDE `score_partition_over_binsums`
  (`mod.rs:3116`); across passes the host holds only two `BestSplit`s
  (`feature_id` / `bin_id` / `score` / `gain` — no candidate index), so the two
  mechanisms are distinct and must not be conflated. Keeping `c` absolute
  (constraint 1) is what makes the per-pass rule correct; "strict `>`, pass A
  first" is what makes the cross-pass rule correct, and it is sufficient on its
  own.)*

**Three non-negotiable constraints, each fixing a confirmed defect:**

1. **`n_features` MUST remain the FULL feature count on EVERY launch, AND the
   candidate index `c` MUST STAY ABSOLUTE** (CRITICAL-2 + PLAN-CHECK pass-2
   MAJOR-B). Add **two** comptime bounds — `#[comptime] feature_lo: u32` and
   `#[comptime] feature_hi: u32` — used ONLY to move the loop's **start and end**,
   never to renumber candidates:
   ```
   // `leaf_stride` keeps its FULL-`n_features` derivation — this is the CRITICAL-2
   // invariant. The old `n_candidates` binding is REPLACED by `hi` (both the loop
   // bound and the sentinel), so it is NOT re-declared: an unused binding is a hard
   // error under `cargo clippy --workspace --all-targets -- -D warnings`
   // (PLAN-CHECK pass-3 MINOR-c).
   let leaf_stride = n_features_usize * n_bins_usize * 2usize;  // UNCHANGED
   let lo = (feature_lo as usize) * n_bins_usize;
   let hi = (feature_hi as usize) * n_bins_usize;
   let mut my_idx = hi as u32;          // sentinel: this pass's own upper bound
   let mut c = ABSOLUTE_POS + lo;
   while c < hi {
       let feature = c / n_bins_usize;  // ABSOLUTE — unchanged from today
       let border  = c % n_bins_usize;  // unchanged
       …
       my_idx = c as u32;               // ABSOLUTE — unchanged from today
   ```
   **Why absolute is the required choice, not a preference.** Verified in situ:
   - `kernels.rs:4540` — the no-winner sentinel is `let mut my_idx = n_candidates
     as u32;`
   - `kernels.rs:4600-4603` — `if score > my_gain { my_gain = score; my_idx = c
     as u32; }`
   - `gpu_runtime/mod.rs:2998` — the HOST computes `n_candidates =
     n_features * n_bins` (the **FULL** space)
   - `gpu_runtime/mod.rs:3106-3123` — the host guards `if (cand as usize) >=
     n_candidates { continue; }`, ties with `gain > best_gain || (gain ==
     best_gain && cand < best_c)`, and decodes `let feature = (best_c as usize) /
     n_bins;` at `:3123`.

   A **relative** `c` (i.e. `n_candidates = (feature_hi - feature_lo) * n_bins`)
   would silently produce three distinct defects:
   1. **Phantom winner** — a pass-B block with no eligible candidate returns
      `my_idx = (feature_hi - feature_lo) * n_bins`, which is **strictly less
      than** the host's full `n_features * n_bins`, so the `>= n_candidates`
      guard does NOT skip it and a `f32::MIN`-adjacent gain is treated as a real
      candidate.
   2. **Wrong feature id** — a genuine pass-B winner decodes as
      `feature = c / n_bins = absolute_feature - feature_lo`, attributing a
      one-hot split to a **float** feature index.
   3. **Ill-defined cross-pass tie-break** — the host would compare a pass-A
      absolute index against a pass-B relative index; on an exact score tie the
      `cand < best_c` comparison is meaningless.

   Keeping `c` absolute leaves the host decode, the `>= n_candidates` range
   guard and the `cand < best_c` tie-break **byte-unchanged**, and still
   satisfies CRITICAL-2 because the *bound* is `feature_hi * n_bins` while
   `n_features` (hence `leaf_stride`) is untouched.
   **Sentinel rule under multi-pass:** each pass seeds `my_idx = hi` (its own
   upper bound), which is `<= n_features * n_bins`. The host's existing
   `>= n_candidates` guard therefore does NOT catch a pass-A/pass-B sentinel
   whose `hi < n_features * n_bins`. **So the host must additionally skip
   `cand >= hi` for the pass it is reducing** — pass its `hi` into
   `score_partition_over_binsums` (a plain `usize` argument) and tighten the
   guard to `if (cand as usize) >= pass_hi { continue; }`. For the float-only
   single-pass launch `pass_hi == n_candidates`, so the guard is
   byte-unchanged.
   The float-only launch is `feature_lo = 0, feature_hi = n_features`, which must
   be **shown** to collapse to today's arithmetic (`lo == 0`, `hi ==
   n_candidates`, `c = ABSOLUTE_POS`, sentinel `= n_candidates`). Apply the
   identical change to `find_optimal_split_kernel` (`kernels.rs:3367`), which
   shares the pattern.
2. **Lift the trailing-border exclusion in BOTH places** (CRITICAL-3): the
   kernel (`kernels.rs:4596-4604`) *and* the **host belt**
   (`gpu_runtime/mod.rs:3108-3112`). The belt must become `one_hot`-aware —
   skip `border >= n_bins_used - 1` only on the threshold pass. Lifting only the
   kernel leaves the highest category permanently unselectable.
3. **Bound one-hot candidates to REAL bins via `real_folds` — NOT
   `TCFeature.folds`** (pass-1 MAJOR-3, corrected by pass-3 MAJOR-1 / [C16]).
   The CPU reference enumerates `distinct_bins_ascending` (only present bins),
   while the device sweeps `0..n_bins` of the padded line. Eligibility becomes
   ```
   if one_hot  { border < real_folds[feature] }
   if !one_hot { border < max_border }
   ```
   where `real_folds` is the **separate per-feature cardinality array T24
   builds** (`quantize_feature_major_with_one_hot`, step 1b) and T27b uploads
   via `DeviceTrainConfig`. Without it, a cardinality-2 column in a 32-wide line
   contributes 30 phantom "all-objects-right" candidates that can tie or beat a
   real one, and device ≠ CPU.
   **Why not `TCFeature.folds` / `PackedCindex::device_arrays()`:** on the
   production path `session.rs:1363` packs with
   `vec![n_bins_line; eff_n_features]`, and `cindex.rs:213-227` copies that
   argument straight into `folds` — so `folds[f] == n_bins_line` for every
   feature and `border < folds[feature]` **is the loop bound**, i.e. no bound.
   Passing true cardinalities to `pack_cindex` instead is also forbidden: it
   would change `feature_bits`/`(group, shift, mask)` and therefore the packed
   words for every pool, breaking T29b's frozen `packed_cindex.json` ([C16]).
   **A hand-supplied `folds` in a unit test cannot detect this** (that is how
   the defect survived three review passes) — hence T28's
   production-session-driven assertion below.

**Red (six fns) — PLACEMENT IS LOAD-BEARING.**
- File: **`crates/cb-backend/src/gpu_runtime/one_hot_split_score_test.rs`**
  (new `src` sibling), mounted in `crates/cb-backend/src/gpu_runtime/mod.rs` as
  `#[cfg(test)] mod one_hot_split_score_test;` — the repo's own pattern
  (`mod ordered_test;` `:717`, `mod session_depth_gt1_test;` `:760`).
  **NOT `crates/cb-backend/tests/`** (PLAN-CHECK pass-2 MAJOR-C): the fns below
  must call `super::score_partition_over_binsums`, which is a **bare private
  `fn` at `gpu_runtime/mod.rs:2961`** and is unreachable from an integration
  test. A sibling at `gpu_runtime::one_hot_split_score_test` is a descendant of
  `gpu_runtime` and therefore sees it, plus `super::cindex::pack_cindex`
  (`pub(crate)` in `pub(crate) mod cindex`). Runs under the default `cpu`
  backend for its numeric assertions.
  Filter: `cargo test -p cb-backend --lib gpu_runtime::one_hot_split_score_test`.
- Fn 1 `one_hot_fold_matches_the_cpu_equality_score_at_depth_two` — **`n_parts =
  4` (depth ≥ 2), NOT 1** (CRITICAL-2: a single partition cannot expose the
  `leaf_stride` defect). Synthetic `bin_sums` for `n_float = 1`, `n_cat = 2`.
  Expected: the device winner equals the CPU `left = bin_sums[v], right = total
  - left` argmax for every `v`, ≤1e-5, on every partition.
- Fn 2 `one_hot_highest_real_bin_can_win_through_score_partition_over_binsums` —
  **drives `score_partition_over_binsums`, NOT the raw kernel** (CRITICAL-3),
  and asserts the returned `BestSplit.bin_id == real_folds[feature] - 1`.
  Initial failure: the host belt discards it even after the kernel arm lands.
- Fn 3 `one_hot_padded_bins_never_win` (pass-1 MAJOR-3): a cardinality-2 one-hot
  column in a 32-wide padded line, with a padded bin seeded to the highest
  score, driven with `real_folds = [.., 2]`. Expected: the winner is a REAL bin.
  Initial failure: a phantom bin wins.
  **Known blind spot, closed elsewhere (pass-3 MAJOR-1):** this fn hand-supplies
  `real_folds`, so it cannot detect a wrong *data source* on the production
  path — which is exactly how the `TCFeature.folds` mistake survived three
  review passes. The production-path assertion is **T28's
  `device_one_hot_parity_with_a_padded_and_a_gap_bin`, driven through
  `begin_device_training`**. State this cross-reference in this fn's doc
  comment.
- **Fn 4 `one_hot_winner_reports_the_absolute_device_feature_index`**
  (MAJOR-B guard): with `n_float = 3` and the winning one-hot column at absolute
  device feature index `4`, assert
  `BestSplit.feature_id == 4` — **not** `4 - feature_lo`. Initial failure under a
  relative index space: `feature_id == 1`, i.e. a one-hot split attributed to a
  float feature.
- **Fn 5 `pass_b_with_no_eligible_candidate_produces_no_winner`**
  (MAJOR-B guard): a pass-B launch whose every one-hot candidate is ineligible
  (all `border >= real_folds[feature]`). Expected: `Ok(None)` from
  `score_partition_over_binsums`. Initial failure under a relative sentinel: the
  sentinel `(feature_hi - feature_lo) * n_bins` slips past the host's
  `>= pass_hi` guard and is reported as a phantom winner.
- Fn 6 `float_only_scorer_output_is_numerically_identical_after_the_one_hot_arm`
  — a frozen `bin_sums` + expected `(best_idx, best_gain)` captured BEFORE the
  change, asserted with `==` on raw bits. Must pass before AND after.
  *(Name and claim aligned to SPEC v3, PLAN-CHECK pass-3 MINOR-d: this asserts
  **numeric identity of the kernel's OUTPUT**, not byte-identity of the kernel —
  adding comptime parameters changes the generated kernel source by
  construction, so only output identity is testable.)*

**Green (minimal).**
1. Add `#[comptime] one_hot: bool`, `#[comptime] feature_lo: u32`,
   `#[comptime] feature_hi: u32` to BOTH scorers, and a
   **`real_folds: &Array<u32>`** binding for the one-hot eligibility bound
   (Design constraint 3 — **not** `TCFeature.folds`, see [C16]).
2. As if-as-STATEMENT comptime arms — **this bullet mirrors Design constraint 1
   verbatim; if the two ever disagree, constraint 1 governs**
   (PLAN-CHECK pass-3 MAJOR-2):
   ```
   // candidate range: ABSOLUTE (Design constraint 1). `feature_lo`/`feature_hi`
   // move only the loop's start/end; candidates are NEVER renumbered.
   let leaf_stride = n_features_usize * n_bins_usize * 2usize;  // UNCHANGED (full n_features)
   let lo = (feature_lo as usize) * n_bins_usize;
   let hi = (feature_hi as usize) * n_bins_usize;
   let mut my_idx = hi as u32;          // sentinel: this pass's own upper bound
   let mut c = ABSOLUTE_POS + lo;
   while c < hi {
       let feature = c / n_bins_usize;  // UNCHANGED from today
       let border  = c % n_bins_usize;  // UNCHANGED from today
       …
       my_idx = c as u32;               // UNCHANGED from today
   ```
   **The relative form `n_candidates = (feature_hi - feature_lo) * n_bins` with
   `feature = feature_lo + c / n_bins` is FORBIDDEN** — it is what Reds fn 4 and
   fn 5 exist to catch, and Design constraint 1 spends ~40 lines proving it
   produces a phantom winner, a wrong feature id and an ill-defined tie-break.
   Do NOT re-declare an `n_candidates`/`n_candidates_abs` binding: `hi` is both
   the loop bound and the sentinel, and an unused binding is a hard error under
   `-D warnings` (MINOR-c).
   - fold: `if one_hot { left = bins[border]; right = total - left }` else the
     existing prefix fold — accumulating `total` in the SAME single bin loop
     (one pass, no second read of `bin_sums`).
   - eligibility: `if one_hot { border < real_folds[feature] }`;
     `if !one_hot { border < max_border }`.
3. Extend `launch_find_optimal_split_pointwise_into` and
   `score_partition_over_binsums` with the new comptime args, the
   **`real_folds`** handle **and the `pass_hi` sentinel bound** (constraint 1);
   make the **host belt** (`mod.rs:3108-3112`) `one_hot`-aware and tighten the
   range guard at `mod.rs:3106-3107` from `>= n_candidates` to `>= pass_hi`;
   leave the decode `let feature = (best_c as usize) / n_bins;` (`mod.rs:3123`)
   and the per-pass `cand < best_c` tie-break (`mod.rs:3116`)
   **byte-unchanged**; add the two-pass driver + the cross-pass host argmax
   (strict `>`, pass A first) in `grow_oblivious_tree_resident` (`mod.rs:3930`).

**Refactor.** Do NOT touch `partition_hist2_lds_kernel`,
`partition_hist2_nonbinary_kernel` or `hist_fill_path` — the fill is unchanged,
which is the entire reason the SPEC-OH-30 speed goal is plausible (research.md
§7.2: `fill 755.98 ms` vs `score 193.08 ms`). Regression scope: `cb-backend`
full + the device parity probes + the two out-of-scope growers.

**Validation.**
```
cargo test -p cb-backend --lib gpu_runtime::one_hot_split_score_test
cargo test -p cb-backend
cargo test -p cb-train --test device_oblivious_parity_probe_test
cargo test -p cb-train --test device_seam_test
cargo test -p cb-train --test device_nonsym_fit_test
cargo test -p cb-train --test device_region_fit_test
cargo test -p cb-train --features rocm --no-run
cargo test -p cb-backend --features rocm            # local gfx1151
```

**Parallelization.** NONE (owns `kernels.rs` + `gpu_runtime/mod.rs`, and adds
the `mod one_hot_split_score_test;` mount to `gpu_runtime/mod.rs`).

---

### T26 — SPEC-OH-23 — the split-application test has a one-hot arm

**Goal.** With `#[comptime] one_hot == true`, `partition_split_kernel` tests
`read_bin(..) == value` instead of `> bin`. With `one_hot == false` it is
numerically identical to today.

**Observable completion.** Device doc-routing for a one-hot split matches the
CPU `FeatureMatrix::passes_one_hot` assignment for every object.

**Blocking / prerequisites.** T25 (both own `kernels.rs`).

**Verified files and symbols.**
- `crates/cb-backend/src/kernels.rs:3731-3770` — `partition_split_kernel`;
  **`:3764-3766`** `if read_bin(cindex, offset, obj_u, shift, mask) > bin {
  new_leaf = new_leaf | (1u32 << level_bit); }`; grid-stride
  `CUBE_COUNT * (CUBE_DIM as usize)` at `:3752`; the generics-float keeper
  `let _ = der1.len();` at `:3746`.
- `crates/cb-backend/src/kernels.rs:2776` — `read_bin` =
  `(words[offset + obj] >> shift) & mask`.
- **ONE kernel, TWO launchers** (confirmed by PLAN-CHECK, so a single-kernel
  patch does reach production):
  `launch_partition_split_into` (`gpu_runtime/mod.rs:1916`, f32 `:1964`,
  f64 `:1981`) — **5 callers** in `gpu_runtime/pairwise.rs`,
  `kernels/grow_loop.rs`, `mod.rs`; and
  `launch_partition_split_packed_into` (`mod.rs:2014`, f32 `:2045`, f64 `:2062`)
  — the PRODUCTION launcher, called from `grow_oblivious_tree_resident` at
  `mod.rs:3965-3978` with `(split_offset, split_shift, split_mask,
  split.bin_id, level)`.
- `crates/cb-train/src/tree.rs:380-385` — `passes_one_hot` (the CPU contract).

**Red — PLACEMENT IS LOAD-BEARING.**
- File: **`crates/cb-backend/src/gpu_runtime/one_hot_partition_split_test.rs`**
  (new `src` sibling), mounted in `gpu_runtime/mod.rs` as
  `#[cfg(test)] mod one_hot_partition_split_test;`.
  **NOT `crates/cb-backend/tests/`** (PLAN-CHECK pass-2 MAJOR-C): the fns drive
  `super::launch_partition_split_packed_into` (**`pub(crate)`**,
  `mod.rs:2014`) and build their input with `super::cindex::pack_cindex`
  (**`pub(crate)`** in `pub(crate) mod cindex`, `cindex.rs:154`) — neither is
  reachable from an integration test. As a `gpu_runtime` descendant the sibling
  sees both, so **the test does NOT need to hand-roll the packed words**. Runs
  under the default `cpu` backend.
  Filter: `cargo test -p cb-backend --lib gpu_runtime::one_hot_partition_split_test`.
- Fn 1 `one_hot_partition_split_routes_on_equality`: `pack_cindex` over one cat
  column with bins `[0,1,2,1,0]`, then `launch_partition_split_packed_into` with
  `value = 1`, `level_bit = 0`. Expected `new_leaf_of == [0, 1, 0, 1, 0]`.
  Initial failure: `[0, 1, 1, 1, 0]` — the `> bin` test lets bin 2 through.
- Fn 2 `float_partition_split_is_unchanged_after_the_one_hot_arm`: a frozen
  input/output pair asserted with `==` (a **numeric**-output identity assertion,
  per the applied SPEC-OH-22 amendment).

**Green (minimal).** Add `#[comptime] one_hot: bool`; the routing becomes
```
let b = read_bin(cindex, offset, obj_u, shift, mask);
if one_hot  { if b == bin { new_leaf = new_leaf | (1u32 << level_bit); } }
if !one_hot { if b >  bin { new_leaf = new_leaf | (1u32 << level_bit); } }
```
(if-as-statement only; the comptime resolves one arm away). Thread the flag
through BOTH launchers and set it from the chosen split's kind at
`mod.rs:3965`. **All 5 `launch_partition_split_into` callers must pass
`one_hot = false`** — the byte-unchanged default (MAJOR-4); two of them are in
the out-of-scope `pairwise.rs` / `grow_loop.rs`.

**Refactor.** None — keep the kernel minimal. Regression scope: `cb-backend`
full + the device parity probes + the two out-of-scope growers.

**Validation.**
```
cargo test -p cb-backend --lib gpu_runtime::one_hot_partition_split_test
cargo test -p cb-backend
cargo test -p cb-train --test device_oblivious_parity_probe_test
cargo test -p cb-train --test device_nonsym_fit_test
cargo test -p cb-train --test device_region_fit_test
cargo test -p cb-backend --features rocm
```

**Parallelization.** NONE (owns `kernels.rs` after T25; also
`gpu_runtime/pairwise.rs`, `kernels/grow_loop.rs`, and adds the
`mod one_hot_partition_split_test;` mount to `gpu_runtime/mod.rs`).

---

### T27 — SPEC-OH-24 — the device seam carries the split kind

**Goal.** `DeviceGrownTree.splits` conveys whether each level's split is
one-hot, and `cb-train`'s device fold materializes `LevelKind::OneHot` +
`one_hot_splits` from it.

**Observable completion.** A device-grown one-hot tree lifts to an
`ObliviousTree` with the same `level_kinds` / `one_hot_splits` shape the CPU
grower produces.

**Blocking / prerequisites.** T25, T26, T19.

**Verified files and symbols.**
- `crates/cb-compute/src/runtime.rs:931-935` — `pub splits: Vec<(u32, u32)>`
  ("Pass test: `quantized_bin[feature] > bin_id`") — carries NO kind.
  **`DeviceGrownTree` has 17 callers** across `cb-compute/src/lib.rs`,
  `kernels/region_device.rs`, `kernels/nonsym_grow.rs`,
  `gpu_runtime/session.rs` + 2 more [MAJOR-4].
- `crates/cb-compute/src/runtime.rs:975-993` — `region_path: Vec<(u32, u32,
  bool, bool)>` — the **in-repo precedent** for a tuple carrying `one_hot`.
- `crates/cb-compute/src/runtime.rs:927-929` — the hard constraint: "PLAIN HOST
  types — no `cubecl` / `cb-backend` type may appear on this struct (T-10-04
  feature-unification landmine)".
- `crates/cb-backend/src/gpu_runtime/mod.rs:3289-3298` — the backend-local
  `GrownTree { splits: Vec<(u32, u32)>, leaf_of, leaf_values, part_stats }`.
- `crates/cb-backend/src/gpu_runtime/mod.rs:3946` —
  `splits.push((split.feature_id, split.bin_id));`
- `crates/cb-backend/src/gpu_runtime/mod.rs:4052-4058` — the `GrownTree` return.
- `crates/cb-backend/src/gpu_runtime/session.rs:1871` — the
  `grow_oblivious_tree_resident` call; `:1922-1923` — `Ok(DeviceGrownTree {
  splits: tree.splits, … })`.
- `crates/cb-train/src/boosting.rs:4042-4200` — the device fold that converts a
  `DeviceGrownTree` into a `GrownTree`.

**Red (SCAFFOLDING — an arity `E0308` — plus one behavioral assertion).**
- File: `crates/cb-train/tests/device_seam_test.rs` (existing).
- Test fn: `device_grown_one_hot_tree_conveys_the_split_kind`.
- Setup: a stub / cpu-backend `Runtime` returning a `DeviceGrownTree` with one
  one-hot level; run the cb-train device fold.
- Expected: the produced `GrownTree` has
  `level_kinds == [LevelKind::OneHot { one_hot_idx: 0 }]` and one
  `one_hot_splits` entry whose `feature` is the **ABSOLUTE cat index** and whose
  `value` is the bin.
- Expected initial failure: tuple arity (`E0308`), then — once widened — a
  behavioral failure if the device feature index is not mapped back through
  `n_float`.

**Green (minimal).**
1. Widen `DeviceGrownTree.splits` to `Vec<(u32, u32, bool)>` —
   `(feature_index, bin_id, one_hot)` — mirroring `region_path`'s precedent and
   documenting that `false` is the byte-unchanged float meaning. **All 17
   callers must compile with `false`** (MAJOR-4); the two in
   `kernels/region_device.rs` and `kernels/nonsym_grow.rs` are OUT-OF-SCOPE
   growers and must be byte-unchanged.
2. Widen the backend-local `GrownTree.splits` identically; push the kind at
   `mod.rs:3946` from T25's two-pass winner.
3. In the cb-train fold, map device feature index `>= n_float` back to the
   ABSOLUTE cat index (the inverse of T24's layout, using T17's
   `one_hot_absolute`) and emit `LevelKind::OneHot` + an `OneHotSplit`.

**Refactor.** Do NOT introduce a parallel `Vec<bool>` — the tuple keeps the kind
and the split inseparable (what `region_path` got right). Ensure NO
`cubecl`/`cb-backend` type enters `cb-compute::runtime`.

**Validation.**
```
cargo test -p cb-train --test device_seam_test
cargo test -p cb-train --test device_nonsym_fit_test
cargo test -p cb-train --test device_region_fit_test
cargo test -p cb-backend
cargo test -p cb-train
cargo build --workspace --all-targets
cargo test -p cb-train --features rocm --no-run
```

**Parallelization.** NONE.

---

### T27b — SPEC-OH-24 / SPEC-OH-25 — device session wiring (`DeviceTrainConfig` + `begin` + the `n_float` boundary)

**Goal.** Own the concrete device-integration surface that revision 1 left
unassigned inside T28's "wire whatever remains". *(PLAN-CHECK MAJOR-5: this is
the largest single piece of device integration and was estimated as a test
task.)*

**Observable completion.** The one-hot flags, **the `real_folds` cardinality
array** and the `n_float` boundary reach the resident scorer;
`is_covered_regime()` admits the one-hot regime; and a session-level test proves
they arrive with the RIGHT VALUES (a cardinality-2 column must show
`real_folds[c] == 2`, **not** the padded `n_bins_line`).

**Blocking / prerequisites.** T27. **Blocks T28**, which then stays a pure
parity gate.

**Verified files and symbols.**
- `crates/cb-compute/src/runtime.rs:1251-1283` —
  `Runtime::begin_device_training` has a **fixed argument list**
  (`bins_feature_major`, `n_features`, `n_bins`, `config`) with **no one-hot
  channel**.
- `crates/cb-compute/src/runtime.rs:1082-1121` — `DeviceTrainConfig` fields;
  `:1123-1140` its `Default`; `:1148-1161` `is_covered_regime()` (requires
  `ctr.is_none()`, `bootstrap_type == No`, `!sample_from_host`, …).
- `crates/cb-compute/src/runtime.rs:927-929` — **PLAIN HOST types only** on the
  seam structs; the new channel must be plain host data.
- `crates/cb-backend/src/gpu_backend.rs:273-287` — `GpuTrainSession::begin`
  forwards the same argument list.
- `crates/cb-backend/src/gpu_runtime/session.rs:1250-1253` — the
  `n == 0 || n_features == 0 || n_bins == 0` decline; `:1270-1274`
  `pad_hist_line_bins`; `:1360-1363`
  `let n_buckets_per_feature = vec![n_bins_line; eff_n_features];`;
  `:1484` and `:1889` where `n_bins_line` is stored / read.
- `crates/cb-backend/src/gpu_runtime/mod.rs:3803` —
  `grow_oblivious_tree_resident` needs the `n_float` boundary to choose
  `feature_lo` / `feature_hi` for T25's two passes.
- `crates/cb-train/src/boosting.rs:3129` — where the host builds
  `(device_bins, device_n_bins)` and the `DeviceTrainConfig`.

**Red — PLACEMENT IS LOAD-BEARING.**
- File: **`crates/cb-backend/src/gpu_runtime/one_hot_session_wiring_test.rs`**
  (new `src` sibling), mounted in `gpu_runtime/mod.rs` as
  `#[cfg(test)] mod one_hot_session_wiring_test;` — the same pattern as the
  existing `mod session_residency;` (`:753`) and `mod session_depth_gt1_test;`
  (`:760`), which are the in-repo precedent for session-level tests.
  **NOT `crates/cb-train/tests/device_seam_test.rs` and NOT
  `crates/cb-backend/tests/`** (PLAN-CHECK pass-2 MAJOR-C): `GpuTrainSession`
  lives in the **private** `mod session;` (`gpu_runtime/mod.rs:694`), so it is
  invisible to both integration-test directories. A sibling at
  `gpu_runtime::one_hot_session_wiring_test` is a descendant of `gpu_runtime` and
  reaches `super::session::…` (and `super::GpuTrainSession` via the
  `pub use session::*;` at `mod.rs:695`) — exactly how `session_depth_gt1_test`
  and `session_residency` already do it.
  Filter: `cargo test -p cb-backend --lib gpu_runtime::one_hot_session_wiring_test`.
- Test fn: `one_hot_flags_real_folds_and_n_float_reach_the_resident_scorer`.
- Setup: 1 float column (31 borders → `n_bins = 32`) + 2 **cardinality-2**
  one-hot columns, so the padded line width `n_bins_line` is `32` while the real
  cardinality is `2`; build the `DeviceTrainConfig` the production host builds
  (`one_hot_flags = [false, true, true]`, `real_folds = [32, 2, 2]`,
  `n_float = 1`); open a `GpuTrainSession` via `begin`.
- Expected: the session stored `one_hot_flags == [false, true, true]`,
  **`real_folds == [32, 2, 2]`** and derived `feature_lo = 1, feature_hi = 3`
  for pass B.
  **The load-bearing assertion is `real_folds[1] == 2`, not `32`** — that is
  precisely the value `TCFeature.folds` would have supplied
  (`session.rs:1363` packs `vec![n_bins_line; eff_n_features]`), and asserting
  it here catches a regression to the inert bound at the seam rather than as an
  unlocalized ≤1e-5 gap in T28 ([C16] / pass-3 MAJOR-1).
- **If no observation point exists, add a `pub(crate) fn` accessor on
  `GpuTrainSession`** returning the stored
  `(one_hot_flags, real_folds, n_float, feature_lo, feature_hi)` — `pub(crate)`
  is reachable from the sibling and adds no public surface. Do NOT rely on a
  `CB_GPU_PROF` string probe for this assertion.
- Expected initial failure: `DeviceTrainConfig` has no `one_hot_flags` /
  `real_folds` field (`E0560`), and `begin_device_training` has no channel to
  carry them.

**Green (minimal).**
1. Add to `DeviceTrainConfig` (PLAIN HOST types only):
   `pub one_hot_flags: Vec<bool>` (length `n_features`),
   **`pub real_folds: Vec<u32>`** (the per-feature REAL cardinality array T24
   step 1b produces — length `n_features`), and `pub n_float: usize`.
   **Two separate questions, do not conflate them (PLAN-CHECK pass-4
   MAJOR-1a):**
   - **Source compatibility** — all three fields get an empty / `false` /
     `0` `Default`, so every existing `DeviceTrainConfig { .. }` literal that
     uses `..Default::default()` keeps compiling byte-unchanged. This is about
     *construction sites*, not runtime values.
   - **Runtime value on a device-eligible fit** — `real_folds` is **NEVER
     empty**. T24's call-site rule makes `boosting.rs:3129` always call
     `quantize_feature_major_with_one_hot` (empty `cat_bins` on a float-only
     pool), so a float-only fit carries `real_folds == [borders[f].len()+1, …]`
     of length `n_features`, exactly as the §9b trace's `produce` **and**
     `carry` rows state and as T24 Red fn 5 asserts. The `Default`-empty value
     is only ever seen by a construction site that never reaches
     `begin_device_training`.
   **`real_folds` is NOT `TCFeature.folds`** ([C16] / pass-3 MAJOR-1): the
   latter is the padded uniform line width on the production path
   (`session.rs:1363` → `cindex.rs:213-227`) and must never bound a candidate.
   `PackedCindex::device_arrays()` deliberately does not carry it (T24 step 3).
   **Sizing against the CTR-widened axis:** `session.rs:1341` widens the feature
   axis to `eff_n_features` when `ctr_is_covered`. SPEC-OH-26 gates one-hot ×
   CTR, so in the one-hot regime `eff_n_features == n_features`; nonetheless
   T27b must **extend `real_folds` to `eff_n_features`**, padding any CTR
   columns with `n_bins_line` (inert — a CTR column is never `one_hot`, so the
   scorer never reads its entry). Assert
   `real_folds.len() == eff_n_features` before upload; a mismatch is a typed
   `CbError::LengthMismatch`, never a silent truncation.
   **Keep this assertion UNCONDITIONAL** (pass-4 MAJOR-1a): with T24's call-site
   rule it is satisfied on every device-eligible path including float-only, and
   it remains the guard that fails loud if `real_folds` is ever empty while
   one-hot columns are present. Do NOT weaken it to
   `if !real_folds.is_empty()` — that would restore exactly the silently-inert
   bound [C16] exists to eliminate.
   **Float-only invariance:** with no one-hot columns, `one_hot_flags` is
   all-`false`, so the scorer only ever takes the `one_hot == false` arm whose
   eligibility is the unchanged `border < max_border`; `real_folds` is uploaded
   but never read. Existing launches are therefore numerically unchanged —
   locked by T29b fn 2 and by T27b's full-device-suite validation.
2. Add a `one_hot` arm to `is_covered_regime()`: an all-`false` `one_hot_flags`
   is the unchanged covered regime; a non-empty one-hot set is covered ONLY when
   `ctr.is_none()` still holds (SPEC-OH-26 guarantees it) — **do NOT relax
   `ctr.is_none()`**.
3. Thread `one_hot_flags` + **`real_folds`** + `n_float` through
   `begin_device_training` (trait default), `gpu_backend.rs:273-287` and
   `session.rs`; store them on the session alongside `n_bins_line` /
   `n_features: eff_n_features` (`session.rs:1484-1485`), and **upload
   `real_folds` as the scorer's `real_folds: &Array<u32>` binding** (T25 Green
   step 1). Do NOT route it through `PackedCindex` — that type carries the
   padded `folds` and is not the source of truth here ([C16]).
4. Derive `feature_lo`/`feature_hi` in `grow_oblivious_tree_resident` from
   `n_float` and pass them to T25's two launches, alongside each pass's
   `pass_hi`.

**Refactor.** Every existing `DeviceTrainConfig { .. }` literal must keep
compiling via `..Default::default()` where it already does. Regression scope:
ALL device tests — the covered-regime gate is depended on by every existing
device oracle.

**Validation.**
```
cargo test -p cb-backend --lib gpu_runtime::one_hot_session_wiring_test
cargo test -p cb-backend --lib gpu_runtime::session_depth_gt1_test
cargo test -p cb-backend --lib gpu_runtime::session_residency
cargo test -p cb-train --test device_seam_test
cargo test -p cb-train --test device_oblivious_parity_probe_test
cargo test -p cb-train --test device_bootstrap_parity_test
cargo test -p cb-train --test device_nonsym_fit_test
cargo test -p cb-train --test device_region_fit_test
cargo test -p cb-backend
cargo build --workspace --all-targets
cargo test -p cb-train --features rocm
```

**Parallelization.** NONE (owns `cb-compute/src/runtime.rs`,
`gpu_backend.rs`, `gpu_runtime/session.rs`, `gpu_runtime/mod.rs`, and adds the
`mod one_hot_session_wiring_test;` mount).

---

### T28 — SPEC-OH-25 — device one-hot training matches the CPU grower (pure parity gate)

**Goal.** The device-grown one-hot model matches the CPU-grown model within
1e-5, with BOTH mandatory anti-false-pass guards. *(Reduced to a pure gate:
all wiring now belongs to T24/T25/T26/T27/T27b — PLAN-CHECK MAJOR-5.)*

**Observable completion.** Parity ≤1e-5, a `CountingGpu` wrapper proving
`grow_tree_on_device` returned a tree for EVERY iteration, and an assertion that
the trained model contains ≥1 `ModelSplit::OneHot`.

**Blocking / prerequisites.** T27b, T19, T21.

**Verified files and symbols.**
- `crates/cb-train/tests/device_bootstrap_parity_test.rs` — the existing
  `CountingGpu`-style precedent (SPEC §9 R6 names it); reuse its wrapper shape.
- `crates/cb-train/tests/device_oblivious_parity_probe_test.rs`,
  `device_nonsym_fit_test.rs`, `device_region_fit_test.rs` — fit-parity shapes.
- `crates/cb-compute/src/runtime.rs:1164-…` — the `Runtime` trait to wrap.
- `crates/cb-compute/src/runtime.rs:1148-1161` — `is_covered_regime()` (must be
  satisfied; `ctr.is_none()` holds because SPEC-OH-26 gates the mixed pool).
- `crates/cb-train/src/boosting.rs:3055-3122` — every clause the setup must
  satisfy: no groups, Plain, no CTR, no penalties, no monotone,
  `SymmetricTree`, `approx_dimension == 1`, `bootstrap_type == No`,
  `random_strength == 0.0`, no eval sets, `has_any_scorable_feature`, unit
  weights, `bias == 0.0`, `leaf_method ∈ {Gradient, Simple}`.
- Local hardware: ROCm gfx1151 for real device runs; the cpu backend runs the
  same kernels under plain `cargo test`.

**Red (four assertions, three of them guards).**
- File: `crates/cb-train/tests/device_one_hot_parity_test.rs` (new).
- Test fn: `device_one_hot_training_matches_cpu_within_1e5`.
- Setup: a `CountingGpu<R>` wrapper counting `grow_tree_on_device` calls that
  returned `Ok(Some(_))`; a pool with 1 float column + 2 one-hot columns,
  `iterations = 5`, `depth = 3`, `bootstrap_type = No`, `random_strength = 0.0`,
  `boost_from_average = false`, unit weights, `leaf_method = Gradient`.
- Expected:
  1. `counter.device_trees == 5` (**guard 1** — a silent CPU fallback makes
     "device == CPU" trivially true);
  2. `model.oblivious_trees.iter().flat_map(|t| &t.splits).filter(|s|
     matches!(s, ModelSplit::OneHot(_))).count() >= 1` (**guard 2**);
  3. `max |device_pred - cpu_pred| <= 1e-5`.
- **Second test fn (pass-1 MAJOR-3 + pass-3 MAJOR-1 — the ONLY assertion in the
  plan that can catch a wrong `real_folds` DATA SOURCE), mandatory:**
  `device_one_hot_parity_with_a_padded_and_a_gap_bin` — **it MUST run through
  the production path `train`/`train_cat` → `device_host_eligible` →
  `begin_device_training` → `grow_oblivious_tree_resident`, never a
  hand-supplied `real_folds`.** That is the whole point: T25 Red fn 3
  hand-supplies the array and therefore passes even when production feeds the
  padded line width, which is how the `TCFeature.folds` defect survived three
  review passes ([C16]). Configure it so the two differ maximally — 1 float
  column with 31 borders (`n_bins = 32`, `n_bins_line = 32`) plus a
  **cardinality-2** one-hot column whose *padded* bins would otherwise be
  eligible, and arrange the der/weight data so a padded bin scores highest.
  Expected: device == CPU within 1e-5, i.e. the winner is a REAL bin.
  Initial failure if `real_folds` is wired to the padded width: a phantom
  "all-objects-right" candidate wins device-side only.
  Second sub-case, same fn: a column with an **interior bin absent from the
  training data** (a gap), so the CPU and device candidate sets differ if the
  `real_folds` rule is wrong in a second, independent way.
  **Do NOT add a session read-back assertion here** (PLAN-CHECK pass-4
  MINOR-1): T28's file is a **cb-train integration** test, and T27b's
  `real_folds` accessor is `pub(crate)` on `cb_backend`'s `GpuTrainSession`
  (itself owned privately by `GpuBackend`, `gpu_backend.rs:296`
  `*self.session.borrow_mut() = session;`), so it is unreachable from here —
  the same visibility class as pass-2 MAJOR-C. **Failure localization lives in
  `gpu_runtime::one_hot_session_wiring_test`** (T27b's Red, which asserts
  `real_folds == [32, 2, 2]` at the seam). Coverage is complete without it:
  T24 Red fn 5 proves the producer emits true cardinalities, T27b's Red proves
  the seam carries them, and this fn proves the end-to-end production-path
  parity. State that cross-reference in this fn's doc comment.
- Third test fn (CRITICAL-2 depth case):
  `device_one_hot_parity_at_depth_three_for_a_mixed_pool` — exercises
  `leaf_stride` addressing on partitions ≥ 1.
- Expected initial failures: guard 1 (`device_trees == 0`) if the device
  declines a cat-bearing pool; guard 2 if the device tree carries no one-hot
  split; the ≤1e-5 assertion for the padded/gap and depth cases.

**Green (minimal).** **No new wiring should be needed** — if any is, it belongs
to the owning task (T24/T25/T26/T27/T27b), which must be re-opened rather than
patched here. Do NOT relax `is_covered_regime`'s `ctr.is_none()`.

**Refactor.** Both anti-false-pass guards must be `assert!`s in the test body,
not comments — SPEC-OH-25 makes them mandatory. Regression scope: all device
tests.

**Validation.**
```
cargo test -p cb-train --test device_one_hot_parity_test
cargo test -p cb-train --test device_bootstrap_parity_test
cargo test -p cb-train --test device_oblivious_parity_probe_test
cargo test -p cb-train --test device_nonsym_fit_test
cargo test -p cb-train --test device_region_fit_test
cargo test -p cb-train
cargo test -p cb-train --features rocm            # real gfx1151 device run
```

**Completion evidence.** The device-tree count, the one-hot split count, and the
measured `max|diff|` for all three cases, from the ROCm run.

**Parallelization.** NONE.

---

### T29 — SPEC-OH-31 — the CPU / model float-only path is unchanged (D-04)

**Goal.** Prove that a pool with no categorical columns produces a byte-identical
trained model, against the **frozen T00 baseline** (not a self-comparison), and
that every existing float-only oracle suite passes **UNMODIFIED**.

**Observable completion.** `float_only_byte_identity_test` passes against
`baseline.cbm` captured at the plan-base SHA; `git diff --stat` shows zero
changed lines in every float-only oracle test file; and `cargo test --workspace`
shows **no NEW failures** relative to T00's recorded accepted baseline.

**Blocking / prerequisites.** T00 (the baseline artifact), T28 (the last
production change).

**Verified files and symbols.**
- `crates/cb-oracle/fixtures/float_only_byte_identity/` — T00's frozen baseline
  + its `README.md` recording `PLAN_BASE_SHA`.
- `.planning/plans/one-hot-categorical-training/baseline/workspace-test-baseline.txt`
  — T00's accepted-failure enumeration (MEMORY
  `catboost-rs-preexisting-test-failures`: `cb-backend` CubeCL MLIR, `cb-train`
  monotone, `catboost-rs-py` python3.14 link).
- `crates/cb-model/src/cbm.rs:750-766` — `CatFeatures` / `OneHotFeatures` are
  `None` for a float-only model after T07 ⇒ wire bytes unchanged by
  construction.
- `crates/cb-train/src/tree.rs:345-351` — `FeatureMatrix::new` hard-codes
  `cat_bins: &[]` ⇒ the CPU search's added loop range is empty [C11].
- Float-only oracle suites that must be edited by NO task:
  `slice_first_oracle_test`, `loss_oracle_test`, `leaf_methods_oracle_test`,
  `regularization_oracle_test`, `overfit_oracle_test`, `permutation_oracle_test`,
  `wave1/2/3_*_oracle_test`, `multiclass_oracle_test`, `multilabel_oracle_test`,
  `multiquantile_oracle_test`, `bootstrap_oracle_test`,
  `bootstrap_dev_oracle_test`, `mvs_seeds_oracle_test`,
  `non_symmetric_grower_oracle_test`, `region_e2e_test`,
  `rayon_determinism_test`, `perf_baseline_test`; plus `cb-model`'s
  `apply_oracle_test`, `cbm_oracle_test`, `json_oracle_test`,
  `shap_oracle_test`, `fstr_oracle_test`, `predict_oracle_test`.
  *(Exception, declared: `shap_oracle_test`, `advanced_fstr_oracle_test` and
  `fstr_oracle_test` gain NEW one-hot-rejection fns in T10/T13. Their EXISTING
  fns must be unchanged — assert on the diff hunks, not the file.)*

**Red.**
- File: `crates/cb-model/tests/float_only_byte_identity_test.rs` (created in
  T00).
- Test fn: `float_only_cbm_bytes_match_the_frozen_plan_base_baseline`.
- Expected: byte-identical output.
- Expected failure (if any): a byte diff, which localizes the regression to a
  specific task via bisection over the wave order.

**Green (minimal).** No production change is expected. Any diff is a genuine
regression and must be fixed in the owning task, not accommodated here.
**Regenerating `baseline.cbm` is FORBIDDEN** (it would turn the test into a
tautology — PLAN-CHECK MAJOR-7).

**Refactor.** None.

**Validation.**
```
cargo test -p cb-model --test float_only_byte_identity_test
git diff --stat <PLAN_BASE_SHA>..HEAD -- 'crates/**/tests/**' 'crates/**/*_test.rs'
   # every float-only oracle test file: ZERO changed lines, except the
   # declared T10/T13 additive-only hunks
cargo test --workspace 2>&1 | tee /tmp/ws-after.txt
diff <(grep -E '^(test result|failures:)' .planning/plans/one-hot-categorical-training/baseline/workspace-test-baseline.txt) \
     <(grep -E '^(test result|failures:)' /tmp/ws-after.txt)
   # NO NEW failing target relative to the accepted baseline
cargo clippy --workspace --all-targets -- -D warnings
```

**Completion evidence.** The `git diff --stat` proving no float-only oracle test
was edited, and the baseline diff showing no new failing target.

**Parallelization.** NONE (a whole-plan gate).

---

### T29b — SPEC-OH-31 — the DEVICE float-only path is unchanged

**Goal.** Supply the device-side byte-identity gate that the CPU
"empty-loop-range" argument does **not** provide. *(PLAN-CHECK C11: on the
device, `quantize_feature_major` + `pack_cindex` build ONE concatenated feature
axis, so there is no analogous emptiness guarantee.)*

**Observable completion.** For a float-only pool, the device path produces
(a) a bit-identical packed cindex + descriptor table, (b) bit-identical
per-level `(best_idx, best_gain)` from the scorer, and (c) a byte-identical
trained model — all against artifacts frozen at the plan-base SHA.

**Blocking / prerequisites.** T28. Runs alongside T29.

**Verified files and symbols.**
- `crates/cb-train/src/boosting.rs:2196-2233` — `quantize_feature_major` (kept
  UNCHANGED for the float-only call path by T24).
- `crates/cb-backend/src/gpu_runtime/cindex.rs:154-228` — `pack_cindex` and the
  `TCFeature` emit; T24 adds `one_hot: &[bool]` (all-`false` here) and exports
  `folds`.
- `crates/cb-backend/src/kernels.rs:4506-4640` — the scorer; T25's float-only
  launch must be `one_hot = false, feature_lo = 0, feature_hi = n_features` with
  the FULL runtime `n_features`, which must **collapse to today's arithmetic**.
- `crates/cb-backend/src/kernels.rs:3731-3770` — `partition_split_kernel`; T26's
  `one_hot = false` arm must be numerically identical.
- `crates/cb-backend/src/gpu_runtime/mod.rs:3108-3112` — the host belt; with
  `one_hot = false` its skip condition must be unchanged.
- `crates/cb-compute/src/runtime.rs:1148-1161` — `is_covered_regime()`; an
  all-`false` `one_hot_flags` must be the unchanged covered regime (T27b).

**Red — PLACEMENT IS LOAD-BEARING.**
- File (fns 1–2): **`crates/cb-backend/src/gpu_runtime/device_float_only_identity_test.rs`**
  (new `src` sibling), mounted in `gpu_runtime/mod.rs` as
  `#[cfg(test)] mod device_float_only_identity_test;`.
  **NOT `crates/cb-backend/tests/`** (PLAN-CHECK pass-2 MAJOR-C): fn 1 calls
  `super::cindex::pack_cindex` + `PackedCindex::device_arrays` (both
  **`pub(crate)`** in the **`pub(crate)`** `cindex` module) and fn 2 calls
  `super::score_partition_over_binsums` (a **bare private `fn`**) — neither is
  reachable from an integration test.
  Filter: `cargo test -p cb-backend --lib gpu_runtime::device_float_only_identity_test`.
- Fn 1 `packed_cindex_for_a_float_only_pool_is_bit_identical`: a frozen
  `(words, offsets, shifts, masks)` tuple read from
  `crates/cb-oracle/fixtures/float_only_byte_identity/device/packed_cindex.json`
  (captured by **T00**, see below), asserted with `==`.
- Fn 2 `float_only_scorer_winner_is_numerically_identical_per_level`: frozen
  per-level `(best_idx, best_gain)` pairs from
  `…/device/scorer_winners.json`, asserted on raw bits — the SAME frozen data
  T25's Red fn 6 uses, promoted to a standing gate. *(A **numeric**-output
  identity assertion, per the applied SPEC-OH-22 amendment.)*
- Fn 3 (in `crates/cb-train/tests/device_oblivious_parity_probe_test.rs`,
  existing — a fit-level test through the public `train`/`save_cbm` API, so an
  integration test is correct here):
  `device_float_only_model_bytes_match_the_frozen_baseline` — a device-grown
  float-only fit, saved with `save_cbm`, byte-compared against a frozen
  `…/device/device_baseline.cbm`. **This one IS a model byte-identity claim and
  stays byte-identical** (SPEC-OH-31).
- Expected initial failure: fixture files missing (before T00 runs); after that,
  none until the device wave — the value of this task is that it FAILS if
  T24/T25/T26/T27b perturb the float path.

**Green (minimal).** No production change expected.

**How T00 captures the frozen device artifacts (PLAN-CHECK pass-2 MAJOR-C —
previously unspecified and unreachable).** The capture must run through the same
`pub(crate)` symbols, so it lives in the SAME `gpu_runtime` sibling as a
`#[ignore]`d writer test:
```rust
// crates/cb-backend/src/gpu_runtime/device_float_only_identity_test.rs
#[test]
#[ignore = "capture-only: run once at the plan-base SHA to freeze the fixture"]
fn capture_float_only_device_artifacts() { /* writes packed_cindex.json,
   scorer_winners.json into crates/cb-oracle/fixtures/float_only_byte_identity/device/ */ }
```
**T00 runs it once, at the plan-base SHA**, with
`cargo test -p cb-backend --lib gpu_runtime::device_float_only_identity_test -- --ignored`,
and commits the emitted files alongside `baseline.cbm`. The `device_baseline.cbm`
for fn 3 is produced by the same T00 run through the public fit API. Because the
sibling file must exist for the capture, **T00 creates it** (with the capture fn
and `#[ignore]`d/stubbed assertions) and **T29b fills in fns 1–2**. This is
recorded in T00's Green step 2.

**Refactor.** None.

**Validation.**
```
cargo test -p cb-backend --lib gpu_runtime::device_float_only_identity_test
cargo test -p cb-train --test device_oblivious_parity_probe_test
cargo test -p cb-backend --features rocm
cargo test -p cb-train --features rocm
```

**Completion evidence.** Bit-identical cindex, scorer winners and model bytes
against the plan-base artifacts.

**Parallelization.** Parallel with T29 (disjoint test files); both gate T30.

---

### T30 — SPEC-OH-30 — the Colab T4 speed gate

**Goal.** On a Colab T4 with a **matched** config, catboost-rs trains a one-hot
workload FASTER than official CatBoost GPU, with device activation OBSERVED
rather than assumed.

**Observable completion.** A committed report showing catboost-rs faster than
official CatBoost GPU on both arms, with a `CB_GPU_PROF tree` line captured for
every catboost-rs arm, and **every knob pinned identically on both sides and
read back**.

**Blocking / prerequisites.** T29, T29b. **Also depends on T01b's branch** — see
the config note below.

**Verified files and symbols.**
- `bench/quick_gpu_speed/bench.py` — the Kaggle-shaped runner
  (`WORK = "/kaggle/working"`, `SPEED_CONFIG = 300_000 × 50`, `DEPTH=6`,
  `ITERS=30`, `LR=0.1`, `L2=3.0`, `BORDER_COUNT=32`, `RANDOM_SEED=42`; a maturin
  wheel with the `cuda` Cargo feature driven through the real Python `.fit()`).
  Its `build_eligibility_audit()` is a STATIC audit and the script states
  plainly that device activation is not observable from Python.
- `bench/bootstrap_gpu/bootstrap_bench_colab.py` — the ONLY Colab-shaped runner:
  `WORK = "/content/bench_out"`, `REPO = "/content/cbrs"` (repo STAGED by the
  driver, not git-cloned), and Part B0's `CB_GPU_PROF=1` short fit per arm
  grepping for `CB_GPU_PROF tree` lines — the probe that CLOSES the activation
  caveat.
- `bench/bootstrap_gpu/kernel-metadata.json` — `id = boomvector/…`, no dataset
  source (the `yensen2` datasets are dead; MEMORY
  `kaggle-colab-gpu-runner-facts`).
- `bench/quick_gpu_speed/kaggle-output-260716-r4c/result.json` — the baseline:
  catboost_rs RMSE `1.2382` vs official CatBoost GPU `1.3000` (1.0499×); stage
  attribution `fill 755.98 / score 193.08 / split 65.57 / derive 84.62 ms` plus
  fixed `fit-prep 225.86 / quantize 70.45 / begin 203.15 ms`.
- `crates/cb-backend/src/gpu_runtime/mod.rs:4039-4049` — the
  `eprintln!("CB_GPU_PROF tree …")` the probe greps for.
- `crates/cb-train/src/boosting.rs:3055-3122` — the eligibility clauses the
  bench config must satisfy: **`params.random_strength == 0.0`**,
  **`bias == 0.0`** (⇒ `boost_from_average = False`), unit weights,
  `leaf_method ∈ {Gradient, Simple}`, no eval sets.
- Runner: `~/.local/bin/colab new -s NAME --gpu T4` (MEMORY
  `kaggle-colab-gpu-runner-facts`; accelerator CAN be pinned; beware the
  orphaned-runtime deadlock). Kaggle `boomvector` is the fallback.

**Matched-config requirements (PLAN-CHECK MAJOR-8 — revision 1 left
`bootstrap_type` unpinned).**
- **Pin `bootstrap_type` EXPLICITLY and IDENTICALLY on both sides.** Official
  CatBoost's GPU default is `Bayesian`; leaving it unset would compare
  catboost-rs (which the eligibility gate pushes toward `No`) against an
  official run doing Bayesian sampling — strictly more work per tree — and
  inflate the reported speedup.
- **Pin `random_strength=0`, `boost_from_average=False`** (both are hard
  eligibility clauses; an unpinned value silently drops catboost-rs to the CPU
  grower).
- **Assert `model.get_all_params()` reads every pinned value back on the
  official side** (MEMORY `cv-orch01-random-strength-fixture` trap).
- **T01b dependency:** if T01b took **Branch B** (the typed rejection), one-hot ×
  bootstrap ≠ `No` is rejected outright, so the benchmark config is
  **constrained to `bootstrap_type='No'` on BOTH sides**, and the report MUST
  state that constraint explicitly rather than silently choosing it. If T01b
  took Branch A, pin whichever type is chosen — identically on both sides.

**Red.**
- File: `bench/one_hot_gpu_speed/one_hot_bench_colab.py` (new).
- The "test" is the runner, whose emitted `result.json` must satisfy
  `speedup_official_catboost_gpu > 1.0` for BOTH the RMSE and Logloss arms and
  `activation_observable == True` for every catboost-rs arm.
- Setup: port `quick_gpu_speed/bench.py` onto the `bootstrap_bench_colab.py`
  shape — `WORK = "/content/bench_out"`, repo staged at `/content/cbrs`, plus
  Part B0's `CB_GPU_PROF` probe.
- Input: `SPEED_CONFIG` widened with categorical columns — 300 000 rows ×
  45 float + 5 **binary** categorical, `one_hot_max_size = 2` — with the pinned
  knobs above on both sides; official CatBoost run with `task_type='GPU'`, the
  same `cat_features`, and the same `one_hot_max_size`.
- Expected initial failure: the runner does not exist; and on first real run the
  most likely genuine failure is `activation_observable == False` (a silent CPU
  fallback), which the probe now makes visible instead of hiding.

**Green (minimal).** Write the runner; run it on Colab T4; commit
`bench/one_hot_gpu_speed/colab-t4-<date>/{report.md, result.json,
t4_results.txt}` and `kernel-metadata.json` following the `boomvector` +
staged-repo pattern. The report must record the pinned `bootstrap_type`, the
`get_all_params()` read-back, and (if applicable) the T01b Branch-B constraint.

**Refactor.** Do NOT modify `bench/quick_gpu_speed/bench.py` or
`bench/bootstrap_gpu/*` — the new runner is a sibling. If the speed gate fails,
look FIRST at host prep (`fit-prep` / `quantize` / `begin` already dominate
device tree time), not the fold — SPEC-OH-30's explicit constraint is that
one-hot must not inflate host prep.

**Validation.**
```
cargo test -p cb-train --features rocm          # correctness on gfx1151 first
~/.local/bin/colab new -s onehot-speed --gpu T4
# then the staged-repo flow from bench/bootstrap_gpu/bootstrap_bench_colab.py
python -c "import json;d=json.load(open('bench/one_hot_gpu_speed/colab-t4-<date>/result.json'));print(d)"
```

**Completion evidence.** The committed `report.md` with both speedup ratios > 1,
per-arm `CB_GPU_PROF tree` evidence, and the pinned-config read-back.

**Parallelization.** NONE (final gate).

---

## 6. Global validation commands (verified against this repo)

```
cargo build --workspace --all-targets
cargo test --workspace          # compare against T00's accepted-failure baseline
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p catboost-rs-py --no-run                     # the E0004 gate [C14]
cargo test -p cb-train --features rocm --no-run           # BUILDS (additive over default=["cpu"])
cargo test -p cb-backend --features rocm                  # local gfx1151 device run
.venv/bin/python crates/cb-oracle/generator/gen_fixtures.py --one-hot-only
git status --short crates/cb-oracle/fixtures
```

**NEVER use** `cargo test -p cb-train --no-default-features --features rocm` —
pre-existing 10× `error[E0432]: unresolved import cb_backend::CpuBackend`
(SPEC §9 R11, [C9]).

**NEVER use** `cargo test -p <crate> --lib <file_stem>` unless `<file_stem>` is
the actual module path — see the mount table in the header. A non-matching
filter reports `0 passed; N filtered out` with a **green exit code**
(PLAN-CHECK MAJOR-9).

---

## 7. Unresolved blockers and unverified assumptions

| # | Item | Status | Owner |
|---|---|---|---|
| B1 | **Does upstream list cat features in `SelectFeaturesForScoring`?** Prior ground truth pins `RSM = n_features` for **float** features only (`tree.rs:610-614`). | **GENUINE BLOCKER**, scheduled first. T01a produces a verdict; **T01b enforces it either way** — Branch A consumes the rule, Branch B typed-rejects one-hot × (bootstrap ≠ No OR random_strength ≠ 0). **Never a guessed count.** | T01a, T01b |
| B2 | `TOneHotFeature.Index` == cat-feature index is *inferred* from research §4.2, not byte-verified. | T08 asserts it against the T02 fixture before the load arm is trusted. | T08 |
| B3 | The `OneHotFeatures.Values` ORDER rule (first-referenced vs ascending-hash vs ascending-bin) is not pinned. | T07 pins it empirically; **T02 now guarantees the fixture makes the two orders differ**, so the pin is not vacuous (PLAN-CHECK "Unverified Items"). | T02, T07 |
| B4 | *(retired)* SPEC-OH-15 cascade size. | **RESOLVED** — 4 production call sites (`fstr.rs:804`, `:846`, `catboost-rs/src/model.rs:292`, `:393`). The escape hatch is deleted; the signature change is unambiguous. | T10 |
| B5 | The comptime device design (T25) is derived from source reading, **not prototyped**. CubeCL may reject `#[comptime] u32` as a loop base/bound. | The invariants are the gate, not the shape: full runtime `n_features` for `leaf_stride`; float-only launch bit-identical (T25 Red fn 4 + T29b). Consult `cubecl_error_solution_guide/` on any build error. | T25 |
| B6 | SPEC-OH-30 assumes the fold change is roughly speed-neutral (`score 193` + `split 65` of 2599 ms; `fill 756` unchanged). Not measured with one-hot columns. | T30 measures it; the named risk is host-prep inflation, not the fold. | T30 |
| B7 | *(retired)* "no ONNX test sibling exists". | **RESOLVED, [C13]** — `crates/cb-model/src/export/onnx_test.rs` exists and is mounted. | T11 |
| B8 | SPEC §7 records an ordering dependency on `.planning/plans/catboost-builder-cat-features-routing/` (still UNTRACKED) which adds the `.one_hot_max_size(k)` builder setter. | This plan's Rust work does not need it (tests drive `BoostParams` directly), but the feature is not user-reachable through `CatBoostBuilder` until that plan lands. | plan-level |
| B9 | SPEC-OH-22 names the wrong kernel for the production path [C1]. | Plan covers both; **SPEC amendment §10-A** proposed. | T25 |
| **B10** | *(reduced, PLAN-CHECK pass-2 MINOR-a)* The `one_hot_bin_to_hash` lookup in `from_trained` must be miss-free. | **RESOLVED by construction:** T17 asserts `hash_by_bin[c].len() == cardinality(c)` and returns `CbError::Degenerate` otherwise, so the lookup cannot miss; `from_trained` stays `-> Self` and its 24 callers are untouched. Residual check: **confirm no `from_trained` caller can construct a `cb_train::Model` bypassing T17's assertion** (a one-line `codegraph_explore "cb_train::Model"` confirmation in T21's commit body). | T17, T21 |
| **B11** | `DEVICE_ONE_HOT_MAX_CARDINALITY`'s numeric value is not yet derived — it must keep the padded line inside `{32,64,128,256}` alongside the float bins. | T24 derives it from `pad_hist_line_bins` + the float `n_bins`, and Red fn 4 asserts the **fallback** (not an abort). | T24 |
| **B12** | Whether a `DeviceTrainConfig` extension can be made without perturbing `is_covered_regime` for existing device oracles. | T27b's validation runs the full device suite; an all-`false` `one_hot_flags` must be the unchanged covered regime. | T27b |

**Explicitly out of scope, untouched:**
`.planning/phases/24-ctr-split-search-correctness/` (ORD-06/ORD-07). No file
under that phase is referenced by any task. The device Region and non-symmetric
growers (`kernels/region_device.rs`, `kernels/nonsym_grow.rs`) are touched ONLY
by mechanical default-value call-site sweeps in T24/T26/T27, each gated by
`device_region_fit_test` + `device_nonsym_fit_test`.

---

## 8. Process attestation

No GSD skill, slash command, workflow, or sub-agent was invoked in producing
this plan or this revision. Tooling used: `codegraph_explore` (CodeGraph MCP),
`Read`, read-only `Bash` (`rg` / `ls` / `sed` inspection, plus one `mv` of the
revision-1 plan into the session scratchpad), and `Write`/`Edit` for this file.
No production code, test, fixture, or configuration file was created or
modified. `SPEC.md` was read and left byte-unchanged (28478 B).

---

## 9. PLAN-CHECK disposition (finding → task → how addressed)

| Finding | Task(s) | How addressed |
|---|---|---|
| **CRITICAL-1** T22/A11 blocked by `cb_oracle::model_json::SplitJson` requiring `border` | **T02b** (new, Wave 0) | New task widening `crates/cb-oracle/src/model_json.rs`: `#[serde(default)] border`, `Option<cat_feature_index>`, `Option<value>`, `SplitJson::is_one_hot()`; Red in the existing `model_json_test.rs` asserting the exact `missing field 'border'` failure first. T15 and T22 now list T02b as a prerequisite. §1 Q1 records why `cb-model` rejects while `cb-oracle` tolerates. |
| **CRITICAL-2** T25's two-pass design cannot bound pass A (`n_features` fixes both `n_candidates` and `leaf_stride`) | **T25** | Design rewritten: `#[comptime] feature_lo` **AND** `feature_hi`; `n_candidates = (feature_hi - feature_lo) * n_bins`; **`n_features` stays the FULL count on every launch** so `leaf_stride` is unchanged; float-only launch = `(0, n_features)` and must be shown to collapse. Applied to `find_optimal_split_kernel` too. Red fn 1 now uses **`n_parts = 4` (depth ≥ 2)**; T28 gains a depth-3 mixed-pool case. |
| **CRITICAL-3** the trailing-border exclusion exists in the HOST BELT as well as the kernel | **T25** | Both sites named verbatim (`kernels.rs:4596-4604` and `gpu_runtime/mod.rs:3108-3112`), both lifted, the belt made `one_hot`-aware. Red fn 2 **drives `score_partition_over_binsums`** (not the raw kernel) and asserts `BestSplit.bin_id == folds[feature] - 1`. Recorded in [C2]. |
| **CRITICAL-4** SPEC-OH-27's fallback not schedulable at T01's position | **T01a** + **T01b** (split) | T01a = Wave 0, evidence only, emits a verdict line (`RSM_RULE:` or `STATUS: NOT-ESTABLISHED`). **T01b = Wave 4, after T16, owns `boosting.rs`**, with two fully-specified Red/Green branches. T01b is in the `boosting.rs` serialization row. T18 step 3 now states explicitly what the perturbed arm does under each branch. |
| **MAJOR-1** SPEC-OH-15 misses `prediction_diff` / `sage_values` | **T10** | `float_splits_of` becomes `Result<Vec<Split>, ShapUnsupported>` so the compiler enforces the guard at BOTH call sites (`shap.rs:830`, `:1139`); all four public surfaces cascade; a Red fn per surface. **The escape hatch is DELETED** (4 real call sites). Recorded in [C6]. |
| **MAJOR-2** T11 breaks `catboost-rs-py` (E0004); validation never builds it | **T11** | `catboost-rs-py/src/errors.rs:148-155` and `:164-171` added to T11's files, Green and serialization row; validation adds `cargo build --workspace --all-targets` and `cargo test -p catboost-rs-py --no-run`. T12/T14 explicitly recorded as SAFE with evidence (`PdpError` via `#[from]` at `catboost-rs/src/error.rs:77`; no exhaustive `GpuApplyUnsupported` match). Recorded in [C14]. |
| **MAJOR-3** device one-hot candidates include phantom/padded bins | **T24**, **T25**, **T28** | T24 exports a per-feature `folds` array from `device_arrays()` (now a 5-tuple). T25's eligibility becomes `if one_hot { border < folds[feature] }`. T25 Red fn 3 seeds a padded bin to the highest score. T28 adds a padded-plus-gap-bin divergence case. |
| **MAJOR-4** 13/5/17-caller blast radius; five files missing from the ownership table | **§3**, **T24**, **T26**, **T27** | §3's serialization table extended with nine device files + `catboost-rs-py/src/errors.rs`; a "Blast-radius warning" block added with the caller counts; T24/T26/T27 each instruct that every non-oblivious call site passes the byte-unchanged default, and each validates with `device_region_fit_test` AND `device_nonsym_fit_test`. |
| **MAJOR-5** device session wiring unowned ("wire whatever remains") | **T27b** (new) | New task owning `DeviceTrainConfig.{one_hot_flags, n_float}` (plain host types, `runtime.rs:927-929`), the `begin_device_training` signature across the trait / `gpu_backend.rs:273-287` / `session.rs`, the `is_covered_regime` arm, and the `feature_lo`/`feature_hi` derivation in `grow_oblivious_tree_resident`. **T28 reduced to a pure parity gate** whose Green says any missing wiring re-opens the owning task. |
| **MAJOR-6** T18 introduces a second argmax | **T18** | Green step 6 rewritten: the argmax MUST be `select_best_candidate`, generalized via a `HasScore` trait (`MINIMAL_SCORE` seed, strict `>` preserved; its 9 callers keep compiling). New Red fn 3 in the existing `tree_tie_break_test.rs` (`mod tie_break;`) asserts a float/one-hot exact tie resolves to FLOAT. |
| **MAJOR-7** the byte-identity baseline must exist before T03 but T29 runs last | **T00** (new) | New Wave-0 task capturing `baseline.cbm` + inputs + the **device** artifacts at the plan-base SHA, with `git rev-parse HEAD` recorded in a fixture `README.md`, plus the accepted workspace-failure transcript. T03 lists T00 as a prerequisite; T29 only CONSUMES it and forbids regeneration. |
| **MAJOR-8** T30 leaves `bootstrap_type` unpinned | **T30** | A dedicated "Matched-config requirements" block: pin `bootstrap_type` identically on both sides, pin `random_strength=0` and `boost_from_average=False` (hard eligibility clauses), assert `get_all_params()` reads every value back, and state the T01b-Branch-B constraint (`bootstrap_type='No'`) explicitly in the report. |
| **MAJOR-9** 11 validation commands filter on non-existent module names | **header**, **all tasks** | A verified mount-name table added to the header (`cbm::tests`, `fstr::tests`, `gpu_apply::tests`, `partial_dependence::tests`, `model_sum::tests`, `apply::region_apply_test`, `apply::staged_predict_test`, `export::onnx::tests`, `export::coreml::tests`, `tree::general`, `tree::tie_break`, `boosting::tests`, `boosting::boosting_device_fold_tests`). Every `--lib` filter in every task rewritten to a real module path; the malformed two-filter form eliminated; a mandatory `mod <file_stem>;` rule for every NEW sibling, with matching filters (`model::model_test`, `apply::apply_one_hot_test`, `tree::tree_one_hot_fused_test`, `gpu_runtime::cindex::cindex_one_hot_test`). Repeated as a global rule in §6. |
| **MAJOR-10** T21 must edit `boosting.rs` but is declared `model.rs`-only | **T17**, **T21**, **§3** | The `cb_train::Model.one_hot_bin_to_hash` (+ `one_hot_absolute`) field addition **moved into T17**, which already owns `boosting.rs` and produces the table. T21 is now a pure `cb-model` consumer, and §3 lists it in the `boosting.rs` row explicitly marked **read-only (no edit)**. |
| **MAJOR-11** T22 deletes a helper a retained test depends on | **T22** | The contradiction is resolved: `no_permutation_in_one_hot_only_path` is **re-pointed at production `train_cat`** (two identical fits must be byte-identical), and `train_one_hot_only`, `one_hot_only_scenario` and `one_hot_encode` are deleted together. The `:284-292` evidence is quoted in the task. |
| **MINOR-1** `#[allow(dead_code)]` is on the struct, not the field | **T24** | Green step 4: keep behavior correct by **narrowing** the allow to `first_fold_index` alone and removing the struct-level allow — removing it wholesale would fail `-D warnings` on `first_fold_index`. |
| **MINOR-2** an ONNX test sibling already exists | **T11**, **§2 [C13]** | T11's Reds point at `crates/cb-model/src/export/onnx_test.rs` and `coreml_test.rs`; blocker **B7 retired**. |
| **MINOR-3** T09's decode guard cannot be "a cheap targeted probe" | **T09** | Green step 2 now offers two explicit implementations — a full `serde_json::Value` pre-parse, or (**preferred**) `#[serde(default)] border` + a `split_type` check inside `from_doc` — plus a test that a float document with a genuinely missing `border` still fails loudly. |
| **MINOR-4** T20's gate vs `one_hot_max_size=3` fixtures | **T20** | A MANDATORY pre-implementation enumeration added, naming `one_hot_oracle_test.rs:217`/`:30` as the mixed-pool case and recording `tensor_ctr_e2e_oracle_test.rs:106` / `plain_ctr_oracle_test` (`one_hot_max_size: 1`) and `fstr_ctr`/`ctr_load` (`[5,4]`) as verified safe; T22 must state how it reconciles `:217`. |
| **MINOR-5** several Reds are structural tautologies | **T00, T01a, T02, T16, T17, T22, T27** | Each such Red is now explicitly labelled **"(SCAFFOLDING — …)"** so it is not mistaken for behavioral evidence; T17 and T27 additionally gained a genuine behavioral assertion alongside the structural one. |
| **MINOR-6** `decode_json` is at `json.rs:813`, not `:648` | **T09**, **§2 [C12]** | Corrected: `decode_json` = `json.rs:813-816`; the split-decode logic lives in `from_doc` near `:648-663`. |
| **MINOR-7** device `n_bins` family for a cat-only pool | **T23**, **T24** | The padding site is now named: `session.rs:1270-1274` `pad_hist_line_bins(n_bins)`, family gate `gpu_runtime/mod.rs:2392-2403`, plus the `n_features == 0 \|\| n_bins == 0` decline at `session.rs:1250-1253`. T24 Red fn 3 asserts a cat-only pool yields `n_bins == 2` (not `0`) and pads to `32`. |
| **Potential bug** — `DEVICE_ONE_HOT_MAX_CARDINALITY`: error vs fallback conflated | **T24** | Decided explicitly per SPEC §9 R10 ("or fall back"): a **`device_host_eligible` clause**, NOT a `CbError::Unsupported` from `quantize_*` (which would abort the fit). Red fn 4 asserts the CPU fallback trains correctly. |
| **Potential bug** — `one_hot_bin_to_hash` length / missing-entry silent skip | **T17**, **T21** | T17 asserts `hash_by_bin[c].len() == cardinality(c)` at construction and returns `CbError::Degenerate` otherwise; T21's "defensive skip" is REMOVED. *(Pass-2 MINOR-a further deleted the "return a split built from some other value" option — see the pass-2 table below.)* |
| **Potential bug** — `model_sum` / `staged_predict` benign | **T05** | Confirmed benign; T05's validation now keeps `cargo test -p cb-model --test model_sum_oracle_test` and `--test staged_predict_oracle_test` as standing locks. |
| **Unverified** — workspace baseline (pre-existing failures) | **T00**, **T29** | T00 records a `cargo test --workspace` transcript enumerating the accepted failures; **T29's acceptance becomes "no NEW failing target relative to that baseline"**, with a concrete `diff` command, instead of "full green". |
| **Unverified** — `OneHotFeatures.Values` ordering pin could be vacuous | **T02**, **T07** | T02's generator must guarantee at least one cat feature whose first-referenced order differs from its ascending-hash order, **asserting it in Python** and recording the labels in `config.json`; T07 pins the rule against that scenario. |
| **Unverified** — device `n_bins` padding site | **T23**, **T24** | Located and cited: `session.rs:1270-1274`. |
| **Unverified** — `TOneHotFeature.Index` semantics | **T08** | Unchanged mitigation (assert against the T02 fixture); tracked as B2. |
| **Unverified** — CubeCL `#[comptime] u32` as a loop base | **T25** | Unchanged posture (the invariant is the gate, not the shape); tracked as B5, now with T29b as a standing regression gate. |
| **Note** — `from_trained` also carries an unrelated `RegionLevel { one_hot }` | **T04**, **T21**, **§2 [C15]** | Warning added to both tasks and recorded as [C15]. |
| **Order change 1** new T00 in Wave 0 | done | §3 wave block + T03 prerequisite. |
| **Order change 2** split T01 | done | T01a (Wave 0) / T01b (Wave 4, after T16). |
| **Order change 3** new T02b | done | Wave 0; T15 and T22 depend on it. |
| **Order change 4** move the field to T17 | done | §3 + T17 Green step 3 + T21 prerequisites. |
| **Order change 5** new T27b | done | Wave 5, between T27 and T28. |
| **Order change 6** T11 must build `catboost-rs-py` | done | T11 validation + §3 ownership row. |
| **Order change 7** extend the serialization table | done | §3, ten additional files. |

---

## 9b. PLAN-CHECK pass-2 disposition (3 MAJOR + 4 MINOR)

| Finding | Task(s) | How addressed |
|---|---|---|
| **MAJOR-A** — T02b's own filter `model_json::tests` is dead (a MAJOR-9 relapse in the task that closes CRITICAL-1) | **T02b**, header | Filter corrected to `cargo test -p cb-oracle --lib model_json_test`. The header mount table gains a **`cb-oracle` block** with all three crate-root mounts verified in situ (`crates/cb-oracle/src/lib.rs:33` `mod compare_test;`, `:35` `mod fixture_test;`, `:37` `mod model_json_test;` — separate from `mod compare;` `:19`, `mod fixture;` `:21`, `mod model_json;` `:22`), plus an explicit "`cb-oracle` is the trap" callout. T02b's Red now states the mount and the resulting `model_json_test::<fn>` path. `cb-train/src/candidates.rs:54 → mod tests;` also added to the table (MINOR-d). |
| **MAJOR-B** — the kernel emitted a RELATIVE candidate index + sentinel while the host decodes an ABSOLUTE one | **T25** | Design constraint 1 rewritten to keep **`c` ABSOLUTE**: `feature_lo`/`feature_hi` move only the loop's start/end (`lo = feature_lo * n_bins`, `hi = feature_hi * n_bins`, `c = ABSOLUTE_POS + lo`, `while c < hi`), `feature = c / n_bins` and `my_idx = c` stay **unchanged from today**, and `n_candidates_abs` / `leaf_stride` stay derived from the full runtime `n_features`. All three named failure modes are quoted in the plan with their anchors (`kernels.rs:4540`, `:4600-4603`, `mod.rs:2998`, `:3106-3123`). The **sentinel** is made unambiguous under multi-pass: each pass seeds `my_idx = hi` and the host receives a `pass_hi` argument, tightening `>= n_candidates` to `>= pass_hi` (byte-unchanged for the float-only single pass). The cross-pass tie-break is now the existing `cand < best_c` on absolute indices, agreeing with float-then-one-hot order **by construction**. Green step 3 updated. **Two new Reds:** fn 4 `one_hot_winner_reports_the_absolute_device_feature_index` (asserts `feature_id == 4`, not `4 - feature_lo`) and fn 5 `pass_b_with_no_eligible_candidate_produces_no_winner`. |
| **MAJOR-C** — four Reds placed in integration tests must exercise private / `pub(crate)` symbols | **header**, **T00**, **T25**, **T26**, **T27b**, **T29b** | A new header section **"Device-test placement rule"** tabulates the eight verified visibilities (`score_partition_over_binsums` bare `fn` `mod.rs:2961`; `pack_cindex` / `PackedCindex` / `device_arrays` `pub(crate)` in `pub(crate) mod cindex` `mod.rs:766`; both partition-split launchers `pub(crate)` `:1916`/`:2014`; `GpuTrainSession` in the **private** `mod session;` `:694`; the two `pub` kernels; `launch_find_optimal_split_pointwise` `pub` `:1295`; `launch_apply_oblivious_f64` `pub` `:331`) and mandates the repo's own `gpu_runtime` `src`-sibling pattern (`mod ordered_test;` `:717`, `mod multiclass_test;` `:725`, `mod ranking_det_test;` `:738`, `mod ranking_stoch_test;` `:746`, `mod session_residency;` `:753`, `mod session_depth_gt1_test;` `:760`). **All four Reds relocated** — see the placement confirmation table below. **T00 now owns the device capture** via a `#[ignore]`d `capture_float_only_device_artifacts` fn in the same `gpu_runtime` sibling, run once at the plan-base SHA with `-- --ignored`, writing `device/{packed_cindex.json, scorer_winners.json, device_baseline.cbm}`; T29b fills in the two assertion fns. §3's `gpu_runtime/mod.rs` row now lists T00 (test-mount only) and T29b, and T00's Parallelization note records the test-only Rust edit. |
| **MINOR-a** — T21 offered an option its own closing sentence forbids | **T21**, **B10** | The "return the split built from a value the lookup DID produce" option is **deleted** (it is a silently wrong split). Green step 2 now states that T17's construction-time assertion (`hash_by_bin[c].len() == cardinality(c)` → `CbError::Degenerate`) IS the guarantee, so the lookup cannot miss, `from_trained` stays `-> Self`, and its 24 callers are untouched; a `debug_assert!` documents the invariant and names T17 as its owner. **B10 reduced** to "confirm no `from_trained` caller can construct a `cb_train::Model` bypassing T17", and T21's completion evidence updated to match. |
| **MINOR-b** — `crates/cb-backend/tests/` conventions unstated | **header** | Stated explicitly: **this plan adds NO file to `crates/cb-backend/tests/`** (MAJOR-C moved all three out). That directory keeps its single file `apply_oblivious_launch_test.rs`, whose convention is recorded — drive a `pub` `cb_backend::gpu_runtime::*` entry point, run under the compile-time-selected runtime (f64 on the default `cpu` backend), assert against a host reconstruction. Every new `gpu_runtime` sibling runs under that same default `cpu` backend, rocm-gated where the existing siblings are. |
| **MINOR-c** — SPEC-OH-22 "byte-identical" vs §10-A "numerically identical" | **T25**, **T26**, **T29b**, **§10-A** | The SPEC change was applied by the coordinator. Plan wording aligned: every identity claim about a **KERNEL** now says *numerically identical / numeric-output identity* (T25 Red fn 6, T26 Red fn 2, T29b Red fn 2 all annotated with the reason: adding comptime parameters changes the generated kernel source by construction, so byte-identity of the kernel is not a testable property while identity of produced scores and chosen splits is). Byte-identity claims about the **trained MODEL** on float-only paths (SPEC-OH-31 / T00 / T29 / T29b fn 3) are unchanged and explicitly re-affirmed. |
| **MINOR-d** — T16's hedge is resolvable | **T16**, header | Hedge dropped; the comment now reads `# verified: candidates.rs:54-55 -> mod tests;`, and the row is added to the header mount table. |

### MAJOR-C placement confirmation — every relocated test, against the real visibility

| Red | NEW location + mount | Symbols it must reach | Why it now compiles |
|---|---|---|---|
| T25 fns 1–6 | `crates/cb-backend/src/gpu_runtime/one_hot_split_score_test.rs`; `#[cfg(test)] mod one_hot_split_score_test;` in `gpu_runtime/mod.rs` | `super::score_partition_over_binsums` (bare `fn`, `mod.rs:2961`), `super::cindex::pack_cindex` (`pub(crate)`) | a descendant module of `gpu_runtime` sees its private items and any `pub(crate)` item in the crate |
| T26 fns 1–2 | `crates/cb-backend/src/gpu_runtime/one_hot_partition_split_test.rs`; mounted in `gpu_runtime/mod.rs` | `super::launch_partition_split_packed_into` (`pub(crate)`, `mod.rs:2014`), `super::cindex::pack_cindex` | same — and the test no longer needs to hand-roll packed words |
| T27b | `crates/cb-backend/src/gpu_runtime/one_hot_session_wiring_test.rs`; mounted in `gpu_runtime/mod.rs` | `super::session::…` / `super::GpuTrainSession` (private `mod session;`, `mod.rs:694`, re-exported `pub use session::*;` `:695`) | exactly how the existing `session_residency` (`:753`) and `session_depth_gt1_test` (`:760`) siblings reach it; a `pub(crate)` accessor may be added for the observation point |
| T29b fns 1–2 **and T00's capture** | `crates/cb-backend/src/gpu_runtime/device_float_only_identity_test.rs`; mounted in `gpu_runtime/mod.rs` (**file created by T00**) | `super::cindex::{pack_cindex, PackedCindex}` (`pub(crate)`), `super::score_partition_over_binsums` (private) | same; the capture fn is `#[ignore]`d and run once with `-- --ignored` at the plan-base SHA |
| T29b fn 3 | **stays** in `crates/cb-train/tests/device_oblivious_parity_probe_test.rs` | public `train` + `save_cbm` only | a fit-level test through the public API — an integration test is correct here |
| T24 (unchanged) | `crates/cb-backend/src/gpu_runtime/cindex_one_hot_test.rs`, `#[path]`-mounted inside `cindex.rs` | `pack_cindex`, `PackedCindex::device_arrays` (both in scope in `cindex.rs`) | already correct in revision 2 — the reference placement PLAN-CHECK pointed at |

**Every relocated filter updated:** `gpu_runtime::one_hot_split_score_test`,
`gpu_runtime::one_hot_partition_split_test`,
`gpu_runtime::one_hot_session_wiring_test`,
`gpu_runtime::device_float_only_identity_test`
(and T24's unchanged `gpu_runtime::cindex::cindex_one_hot_test`).

---

## 9c. PLAN-CHECK pass-3 disposition (2 MAJOR + 4 MINOR)

| Finding | Task(s) | How addressed |
|---|---|---|
| **MAJOR-1** — the `folds` bound does not bound: `TCFeature.folds` is the PADDED line width on the production path, so `border < folds[feature]` is a no-op and pass-1 MAJOR-3 was never actually fixed | **T24**, **T25**, **T27b**, **T28**, **§2 [C16]** | New correction **[C16]** records the three verified links (`session.rs:1363` `vec![n_bins_line; eff_n_features]` → `pack_cindex` `:1364`; `cindex.rs:213-227` copies that argument into `folds`; ⇒ `folds[f] == n_bins_line` always) **and** the prohibition on the obvious repair (`cindex.rs:181-200` derives `bits = feature_bits(nb)` and hence `(group, shift, mask)` from `n_buckets_per_feature`, so changing it alters the packed words for every pool and breaks T29b's frozen `packed_cindex.json`). **A SEPARATE `real_folds` array is plumbed T24 → T27b → T25:** T24 Green step **1b** builds it host-side in `quantize_feature_major_with_one_hot` (float `f` → `borders[f].len()+1`; one-hot `c` → `one_hot_bin_to_hash[c].len()`), returned as a third value; T24 step 3 **shrinks `device_arrays()` back to a 4-tuple** and adds a doc line on `TCFeature.folds` saying it must never bound a candidate; T27b carries `real_folds` on `DeviceTrainConfig` + `begin_device_training` and uploads it as the scorer binding; T25 Design constraint 3 and Green step 1/2 consume `real_folds`, never `folds`. **Production-path assertion added** (the blind spot that let this survive three passes): T28's `device_one_hot_parity_with_a_padded_and_a_gap_bin` is now explicitly required to run through `train_cat` → `begin_device_training` → `grow_oblivious_tree_resident` with **no hand-supplied array**, on a 31-border float column (`n_bins_line = 32`) plus a cardinality-2 one-hot column with a padded bin arranged to score highest, plus a session read-back asserting `real_folds[n_float+c] == 2`. T27b's Red likewise asserts `real_folds == [32, 2, 2]`. T25 Red fn 3 keeps its hand-supplied form but its doc comment now names T28 as the production-path counterpart. New T24 Red **fn 1b** pins `folds == 32` (not `2`) when packed the production way, so a future reader cannot re-confuse the two. |
| **MAJOR-2** — T25's Green step 2 still specified the RELATIVE index its own Design rejects | **T25** | Green step 2's first bullet replaced with the Design's absolute-`c` block verbatim (`lo`/`hi`, `my_idx = hi`, `c = ABSOLUTE_POS + lo`, `while c < hi`, `feature = c / n_bins`, `my_idx = c`, `leaf_stride` off full `n_features`), prefixed with "**this bullet mirrors Design constraint 1 verbatim; if the two ever disagree, constraint 1 governs**" so they cannot drift again, and followed by an explicit "**the relative form … is FORBIDDEN**" naming Reds fn 4/5 as its detectors. Green step 3 also updated to say `real_folds` + `pass_hi` and to name the per-pass tie-break at `mod.rs:3116`. |
| **MINOR-a** — `spec_version: 2` | — | **Already fixed by the coordinator** (`SPEC.md` now reads `spec_version: 3`). No plan action; recorded here for completeness. |
| **MINOR-b** — the header visibility table wrongly marks `GpuTrainSession` unreachable, contradicting the plan's own placement table | header | Row split in two and corrected against source: **the TYPE is reachable** (`session.rs:714` `pub struct`, re-exported by `pub use session::*;` at `mod.rs:695`), while **the one-hot observation point** (`one_hot_flags` / `real_folds` / `n_float` / derived `feature_lo`,`feature_hi`, a `pub(crate)` accessor added by T27b) is **not** — which is the actual reason T27b's Red is a `gpu_runtime` sibling. The two tables now agree. |
| **MINOR-c** — `n_candidates_abs` declared and never used under `-D warnings` | **T25** | Dropped from both the Design snippet and Green step 2, with an inline comment explaining that `hi` serves as both loop bound and sentinel so the old `n_candidates` binding is *replaced*, not re-declared, and that an unused binding is a hard error under `cargo clippy --workspace --all-targets -- -D warnings`. The CRITICAL-2 invariant is now carried solely by `leaf_stride`'s full-`n_features` derivation, which is the property that actually matters. The one remaining prose reference (`<= n_candidates_abs`) rewritten to `<= n_features * n_bins`. |
| **MINOR-d** — the cross-pass tie-break sentence conflates two mechanisms; Red fn 6's name claims bit-identity | **T25** | The Design sentence now states only the correct-and-sufficient cross-pass rule — "**strict `>`, with pass A evaluated first, so a pass-B tie can never displace a float candidate**" — with a precision note that `cand < best_c` (`mod.rs:3116`) is the **per-pass** rule living inside `score_partition_over_binsums`, that across passes the host holds only two `BestSplit`s (no candidate index), and that the two must not be conflated. Red fn 6 renamed `float_only_scorer_output_is_numerically_identical_after_the_one_hot_arm`, aligned to SPEC v3. |

**Reading §9 and §9b after revision 4.** Those tables are the *historical*
record of what revisions 2 and 3 did; two of their cells describe wording that
revision 4 has since superseded, and the current text governs:
- §9's CRITICAL-2 row and §9b's MAJOR-B row quote
  `n_candidates = (feature_hi − feature_lo) * n_bins` and `n_candidates_abs` —
  both **replaced** by the absolute-`c` / `hi`-only form (pass-3 MAJOR-2 and
  MINOR-c above).
- §9b's MAJOR-B row says the cross-pass tie-break "is the existing
  `cand < best_c` on absolute indices" — **superseded** by MINOR-d above: the
  cross-pass rule is "strict `>`, pass A first"; `cand < best_c` is per-pass.
- §9's MAJOR-3 row and §9b's MAJOR-3 reference say `device_arrays()` exports
  `folds` — **superseded** by [C16] / pass-3 MAJOR-1: it exports a 4-tuple and
  the bound travels as `real_folds` via `DeviceTrainConfig`.

**Pass-3 CLOSED items left undisturbed:** MAJOR-A (filter re-audit), MAJOR-B's
Design (the float-only collapse table, the `pass_hi` sentinel argument), MAJOR-C
(test placement + T00's `#[ignore]` capture + T29b's chain), and pass-2
MINORs a–d. No edit in revision 4 touches any of them except MAJOR-B's Green,
which is MAJOR-2's required fix, and the `n_candidates_abs` line, which is
MINOR-c's.

---

## 9d. PLAN-CHECK pass-4 disposition (1 MAJOR + 2 MINOR)

| Finding | Task(s) | How addressed |
|---|---|---|
| **MAJOR-1a** — the trace's `carry` row said `real_folds` is `empty (Default)` on a float-only pool, while the `produce` row, T24 step 1b and T24 Red fn 5 all say `[borders+1, …]`; with T27b's **unconditional** length assertion, the "empty" reading fails **every** float-only device fit | **T24** Green step 1, **§9b trace** | Both halves fixed. (a) **T24 Green step 1 gains an explicit "Call-site rule"**: the device-quantize call site is `boosting.rs:3129` and there is exactly ONE; it **ALWAYS** calls `quantize_feature_major_with_one_hot`, passing an **empty `cat_bins` slice** on a float-only pool, and therefore **ALWAYS populates `real_folds` to length `n_features`** — `real_folds` is **never empty on a device-eligible fit**. (b) The genuinely ambiguous "UNCHANGED" sentence is disambiguated: it means `quantize_feature_major`'s **body and signature are unmodified and it is delegated to** for the float prefix (so the float bin bytes are provably identical, SPEC-OH-31); it does **NOT** mean the float-only path keeps calling the 2-tuple version. That wrong reading is named, its exact consequence spelled out (seven device oracles + T29b would fail on `CbError::LengthMismatch`), and **declared forbidden**. (c) The trace's `carry` row float-only cell corrected from `empty (Default)` to **`[borders+1, …]`, length `n_features`, NEVER empty on a device-eligible fit**, matching the `produce` row and T24 Red fn 5. The `Default`-empty remark is moved out of that cell into T27b step 1 as a separate **source-compatibility** statement (it answers "do existing struct literals still compile", not "what is the runtime value"). (d) **T27b's length assertion stays unconditional** — with (a) it holds on every path, and it remains the guard that catches an empty `real_folds` when one-hot columns exist. |
| **MINOR-1** — T28's session read-back needs a `pub(crate)` accessor on `cb_backend`'s `GpuTrainSession` but T28 is a **cb-train integration** test (pass-2 MAJOR-C's visibility class, re-appearing in an assertion revision 4 added) | **T28** | The read-back assertion is **deleted** from T28, with an in-place note recording why it is unreachable (`pub(crate)` on `cb_backend`; the session is owned privately by `GpuBackend`, `gpu_backend.rs:296`) and pointing at **`gpu_runtime::one_hot_session_wiring_test`** (T27b's Red) as where failure localization lives. **Coverage is unchanged**: T24 Red fn 5 proves the producer emits true cardinalities, T27b's Red proves the seam carries `[32, 2, 2]`, and T28's production-path ≤1e-5 parity proves the end-to-end property. Only the localization nicety moved. |
| **MINOR-2** — T24 step 4's allow-narrowing is stale after step 3 removed `folds` from the tuple | **T24** step 4 | Corrected: after this task **`one_hot_feature` becomes read** (via `device_arrays()`), but **`folds` AND `first_fold_index` both remain unread in the lib target** — verified across `crates/cb-backend/src/`, their only occurrences are the doc comments at `cindex.rs:48`/`:54` and the two writes at `:224`/`:225`. So the allow is narrowed to **both** fields (or the struct-level allow is kept with a comment naming exactly those two as contract-only). The step now states explicitly that narrowing to `first_fold_index` alone **fails `-D warnings` with `field is never read: folds`**, and that T24 Red fn 1b's read of `folds` does not help because it is `#[cfg(test)]`-only while the lib target compiles without it. |

**Pass-4 CLOSED items left undisturbed:** the `real_folds` mechanism itself
(producer → carrier → uploader → consumer), the production-path blind-spot
closure in T28, the float-only comptime-elision argument (which the checker
confirmed the repo already relies on at `kernels.rs:4582`/`:4592`,
`if score_fn == comptime!(SCORE_FN_COSINE)`), CTR-padding inertness, the
fail-loud behavior of an empty `real_folds` when one-hot columns exist, SPEC v4,
MAJOR-2, all pass-3 MINORs, and MAJOR-A / MAJOR-B-Design / MAJOR-C. Revision 5
is text-only and touches none of them.

**Note on `mod.rs:3122` (checker "no finding", recorded so it is not
re-litigated):** the post-loop `if (best_c as usize) < n_candidates` winner test
is left **unchanged** and remains correct under the two-pass scheme —
`best_c` starts at `u32::MAX` and is only ever assigned from a `cand` that
already passed `>= pass_hi`, and `pass_hi <= n_candidates`. T25 must not touch
it.

### `real_folds` plumbing — end-to-end trace

| stage | task | what it does | float-only value | read by anything float-only? |
|---|---|---|---|---|
| produce | **T24** step 1b | `quantize_feature_major_with_one_hot(...) -> (bins, n_bins, real_folds)`; float `f` → `borders[f].len()+1`, one-hot `c` → `one_hot_bin_to_hash[c].len()` | `[borders+1, …]` | no |
| carry | **T27b** step 1 | `DeviceTrainConfig.real_folds: Vec<u32>` (plain host type), extended to `eff_n_features` with `n_bins_line` for CTR columns, length-asserted | **`[borders+1, …]`, length `n_features`** — identical to the produce row; **NEVER empty on a device-eligible fit** | no |
| upload | **T27b** step 3 | through `begin_device_training` → `gpu_backend.rs:273-287` → `session.rs`, stored beside `n_bins_line` (`:1484-1485`), bound as `real_folds: &Array<u32>` | uploaded, never read | no |
| consume | **T25** constraint 3 / Green 1–2 | `if one_hot { border < real_folds[feature] }` — **only under the comptime `one_hot == true` arm** | n/a — pass A / float-only launches take `one_hot == false`, whose eligibility stays the byte-unchanged `border < max_border` | **no** |
| prove | **T24** fn 5, **T27b** Red, **T28** fn 2, **T29b** fn 2 | values correct; seam values correct; production-path parity; float-only scorer output numerically identical | — | — |

**Float-only invariance argument (why no existing launch changes):** the only
new *read* of `real_folds` is inside the `one_hot == true` comptime arm.
`one_hot` is `false` for pass A and for every single-pass float-only launch, and
CubeCL resolves the comptime branch away, so the emitted eligibility expression
for those launches is exactly today's `border < max_border`. `real_folds` is
uploaded but never sampled. This is asserted by **T24 Red fn 5** (float-only
`(bins, n_bins)` element-wise identical to plain `quantize_feature_major`),
**T25 Red fn 6** (frozen `(best_idx, best_gain)` numerically identical),
**T29b fns 1–2** (frozen packed cindex + scorer winners at the plan-base SHA),
and **T27b's full-device-suite validation**.

---

## 10. SPEC amendments

Both amendments proposed in revision 2 **have been applied by the coordinator**
(§10-A and §10-B below are retained as the record of what was requested and
why), together with the pass-2 MINOR-c wording fix: SPEC-OH-22's
`one_hot == false` invariant now reads "the kernel's OUTPUT is **numerically
identical**", with a note that adding comptime parameters changes the generated
kernel source by construction, so byte-identity of the kernel is not a testable
property while identity of produced scores and chosen splits is. This plan's
task wording is aligned to that (MINOR-c row above).

**§10-C (raised in revision 3) has been APPLIED** — SPEC v3 carries the
`numerically identical` wording in **both** OH-22 and OH-23.

**§10-D (raised in revision 4) has ALSO been APPLIED as SPEC v4** — SPEC-OH-22
now names the bound as the feature's **true cardinality**, forbids
`TCFeature.folds` with the evidence chain (`session.rs:1363` packs
`vec![n_bins_line; eff_n_features]` → `cindex.rs:217` copies it into `folds`),
and forbids the unsafe repair for the right reason (`pack_cindex` derives
`bits = feature_bits(nb)` at `cindex.rs:181-200`, so changing
`n_buckets_per_feature` would alter packed words for every pool and break
SPEC-OH-31). `spec_version: 4`. The checker re-verified it as correct and
sufficient with no new gap introduced.

**Revision 5 requires NO SPEC amendment.** All of pass 4's findings
(1 MAJOR + 2 MINOR) are plan-text defects, fixed in §9d. Every amendment
§10-A..§10-D requested across revisions 2–4 is applied and checker-confirmed;
§10-A..§10-D below are retained only as the record of what was requested and
why.

### §10-D — SPEC-OH-22 should name WHICH `folds` bounds the one-hot fold — APPLIED (SPEC v4)

**Current text (SPEC-OH-22, the v2 amendment's added constraint):**
> … fold over only the real bins of that feature (`border < folds[feature]`) …

**Proposed clarification:**
> … fold over only the real bins of that feature — bounded by the per-feature
> **real cardinality** (the number of distinct quantized values the feature
> actually has), which is **NOT** `TCFeature.folds`: on the production path that
> field carries the padded uniform line width, because
> `crates/cb-backend/src/gpu_runtime/session.rs:1363` packs with
> `vec![n_bins_line; eff_n_features]` and `cindex.rs:213-227` copies that
> argument straight into `folds`. The bound must therefore travel as a separate
> host-computed cardinality array, and `n_buckets_per_feature` must NOT be
> changed to carry it (that would alter `feature_bits` → `(group, shift, mask)`
> and hence the packed words for every pool, including float-only).

### §10-C — SPEC-OH-23 had the same `byte-identical` kernel claim MINOR-c fixed in SPEC-OH-22 — APPLIED (SPEC v3)

**Current text (`SPEC.md:340-343`, SPEC-OH-23):**
> **Given** `#[comptime] one_hot == true`, **then** `partition_split_kernel` tests
> `read_bin(..) == value` instead of `> bin`.
> **Invariant:** with `one_hot == false` the kernel is byte-identical to today.

**Proposed replacement (mirroring the wording already applied to SPEC-OH-22 at
`SPEC.md:333-337`):**
> **Given** `#[comptime] one_hot == true`, **then** `partition_split_kernel` tests
> `read_bin(..) == value` instead of `> bin`.
> **Invariant:** with `one_hot == false` the kernel's OUTPUT is **numerically
> identical** to today — the same `new_leaf_of` routing for every object.
> (Numerical, not byte, identity: T26 adds a `#[comptime] one_hot: bool`
> parameter, which changes the generated kernel source by construction, so
> byte-identity of the kernel itself is not a testable property; identity of the
> produced doc routing is.)

**Why it matters:** T26's Red fn 2
(`float_partition_split_is_unchanged_after_the_one_hot_arm`) asserts a frozen
input/output pair — an OUTPUT identity check. Under the current SPEC-OH-23
wording that test does not literally discharge the stated invariant, and an
implementer trying to satisfy the literal text would be chasing an unachievable
property. The plan text is already aligned to the numeric reading; only the SPEC
sentence is stale.

### §10-A — SPEC-OH-22 named the wrong kernel (blocker B9, correction [C1]) — APPLIED

**Current text (SPEC.md §5, SPEC-OH-22):**
> **Given** `#[comptime] one_hot == true`, **then** `find_optimal_split_kernel`
> folds `left = bin_sums[value]`, `right = total - left` (instead of the float
> prefix fold), preserving candidate indexing, argmax, and lowest-index
> tie-break.

**Proposed replacement:**
> **Given** `#[comptime] one_hot == true`, **then** BOTH split scorers —
> `find_optimal_split_partition_kernel` (`crates/cb-backend/src/kernels.rs:4506`,
> the PRODUCTION resident path, reached via `score_partition_over_binsums`
> `gpu_runtime/mod.rs:2961` ← `grow_oblivious_tree_resident` `mod.rs:3803`) and
> `find_optimal_split_kernel` (`kernels.rs:3367`, the non-resident slice entry
> reached via `launch_find_optimal_split_pointwise_into` `mod.rs:1319`) — fold
> `left = bin_sums[value]`, `right = total - left` (instead of the float prefix
> fold), over only the real bins of that feature (`border < folds[feature]`),
> preserving candidate indexing, argmax, and lowest-index tie-break.
> **Invariant:** with `one_hot == false` both kernels are numerically identical
> to today, and the runtime `n_features` argument — which also fixes the
> per-partition `leaf_stride` — is unchanged on every launch.
> **Constraint:** the histogram **fill** is unchanged; and the trailing-border
> exclusion must be lifted for one-hot in BOTH the kernel
> (`kernels.rs:4596-4604`) and the host belt (`gpu_runtime/mod.rs:3108-3112`).

### §10-B — SPEC-OH-15 demands a typed error from infallible APIs (correction [C6])

**Current text (SPEC.md §5, SPEC-OH-15):**
> **Given** a model with one-hot splits, **when** `shap_values` runs, **then** it
> returns a typed unsupported error (or a correct arm) — never a silently
> shortened `tree_depth` (`shap.rs:550-552`).

**Proposed replacement:**
> **Given** a model with one-hot splits, **when** ANY consumer of
> `float_splits_of` (`crates/cb-model/src/shap.rs:550-552`) runs — namely
> `shap_values` (`:534`), `shap_interaction_values` (`:941`), `prediction_diff`
> (`:1137`) and `sage_values` (`:1236`), all four `pub` and re-exported at
> `cb-model/src/lib.rs:58` — **then** it returns a typed unsupported error, never
> a silently shortened `tree_depth`.
> **Note:** all four are **infallible** (`-> Vec<…>`) today, as is
> `fstr::loss_function_change` (`fstr.rs:788`), so satisfying this specification
> requires a deliberate signature change to `Result<_, ShapUnsupported>`
> cascading through four production call sites (`fstr.rs:804`, `fstr.rs:846`,
> `catboost-rs/src/model.rs:292`, `:393`). `float_splits_of` itself becomes
> fallible so the compiler enforces the guard at both of its call sites
> (`shap.rs:830`, `:1139`).









