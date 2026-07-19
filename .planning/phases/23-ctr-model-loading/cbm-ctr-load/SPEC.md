---
title: "CTR model loading — decode_cbm reconstructs ctr_data + ModelSplit::Ctr"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-17T00:00:00Z
phase: 23
requirement_ids:
  - CTRLOAD-01
source_requirements:
  - "Gap analysis: no public path loads/predicts an upstream categorical .cbm (load_cbm sets ctr_data:None, drops CTR splits). Foundational for CTR predict parity + later FSTR-01."
---

# CTR model loading (`.cbm`)

> **Draft.** Make `cb_model::decode_cbm`/`load_cbm` reconstruct the per-tree
> `ModelSplit::Ctr` splits AND the baked `Model.ctr_data` value tables from an
> upstream `catboost==1.2.10` `.cbm`, so a categorical model loads and predicts in
> Rust to ≤1e-5 via the EXISTING `predict_raw_cat` apply path (no apply changes).

## 1. Context

Today `decode_cbm` (`crates/cb-model/src/cbm.rs:386`) → `reconstruct_model` (:494)
builds float bins only, lifts only `ModelSplit::Float`, and sets `ctr_data: None`
`[VERIFIED: LOCAL cbm.rs:533,571,612]`. A serialized tree's `TreeSplits[i]` are
GLOBAL indices into a combined bin table ordered **FloatFeatures → OneHotFeatures →
CtrFeatures**; any non-float index currently ERRORS or is dropped `[VERIFIED: LOCAL
cbm.rs:565]`. The in-memory CTR structures (`CtrSplit`, `CtrData`, `CtrValueTable`)
and the apply path (`predict_raw_cat` → `passes_ctr_split`) already exist and are
oracle-locked via the trainer path (`from_trained`+`with_ctr_data`); this slice makes
the LOADER an alternative producer of the SAME structures
`[VERIFIED: research §6 apply-compatibility proof]`.

## 2. Scope / non-goals

### In (v1)
- Oblivious (symmetric) trees.
- SimpleCtr (1 cat feature) AND CombinationCtr (≥2 cat features) projections.
- All CTR types a default categorical model emits (`Borders`, `Counter`,
  `FeatureFreq`) and all priors — a real categorical `.cbm` always references several.
- Load-and-predict parity (`load_cbm` → `predict_raw_cat`) ≤1e-5.

### Out / deferred (typed-error, never silent mis-load)
- **One-hot SPLITS** in trees — no `ModelSplit::OneHot` variant exists
  (`apply.rs:196` matches only Float/Ctr); a `.cbm` carrying one-hot split indices →
  typed `ModelError`. (One-hot FEATURE tables are still COUNTED for the bin offset.)
- Non-symmetric / Region CTR trees; multiclass CTR (`TargetClassesCount>2` /
  `TargetBorderClassifierIdx>1` collision); mean-type CTRs
  (`BinarizedTargetMeanValue`/`FloatTargetMeanValue`) unless a fixture forces one;
  CTR **saving** (`save_cbm` stays no-tail — load-only slice).

## 3. Dependencies (verified)

| Dependency | Interface | Evidence |
|-----------|-----------|----------|
| Target structs | `CtrSplit` (model.rs:43), `ModelSplit::Ctr` (:75), `Model.ctr_data` (:298) | `[VERIFIED: CODEGRAPH]` |
| CTR value table | `CtrValueTable` (ctr_data.rs:151), `CtrData` (:283), `ECtrType::from_i8` (:96), `ctr_base_key` (:294), `Prior` (:128) | `[VERIFIED: LOCAL]` |
| Projection ctor | `cb_train::TProjection::from_features(&[usize])` (projection.rs:121) — sorts+dedups | `[VERIFIED: LOCAL; cb-model deps cb-train]` |
| FlatBuffers accessors | `TModelTrees.{CtrFeatures,OneHotFeatures}`, `TCtrFeature{Ctr,Borders}`, `TModelCtr{Base,TargetBorderIdx,PriorNum,PriorDenom,Shift,Scale}`, `TModelCtrBase{FeatureCombination,CtrType}`, `TFeatureCombination.CatFeatures`, `TCtrValueTable{ModelCtrBase,IndexHashRaw,CTRBlob,CounterDenominator,TargetClassesCount}` | `[VERIFIED: LOCAL generated/*]` |
| Apply consumer | `predict_raw_cat` (apply.rs:386), `passes_ctr_split` (:157), `ctr_table_key` (:126) — UNCHANGED | `[VERIFIED: research §6]` |

## 4. Typed contracts (no public signature change)

`load_cbm(path) -> Result<Model, ModelError>` and `decode_cbm(&[u8]) -> Result<Model,
ModelError>` keep their signatures; a categorical `.cbm` now returns a `Model` with
`ctr_data: Some(..)` and `ModelSplit::Ctr` splits. New internal helpers:

```rust
// classified combined bin entry (extends today's float-only BinFeature)
enum BinKind { Float { feature: usize, border: f64 },
               OneHot,                                   // counted, not representable
               Ctr { ctr_feature: usize, border_index: usize } }
// build the FloatFeatures→OneHotFeatures→CtrFeatures ordered table:
fn build_combined_bins(trees: &TModelTrees) -> Result<Vec<BinKind>, ModelError>;
// parse the appended model-parts tail into CtrData:
fn decode_ctr_model_parts(tail: &[u8]) -> Result<CtrData, ModelError>;
```

All new parsing is bounds-checked before slicing, uses verifying FlatBuffers
accessors (never `_unchecked`), and returns `ModelError::Deserialize`/`SchemaVersion`
on malformed input — no panic/unwrap/indexing (workspace deny-lints).

## 5. Failure-isolated behavioral specs

### CTR-01 — Combined bin-feature classification
- **Responsibility:** build the `FloatFeatures→OneHotFeatures→CtrFeatures` ordered bin
  table and classify each `TreeSplits[i]` global index by range. Isolates index math.
- **Behavior:** `gidx < n_float_bins` → `Float` (byte-identical to today);
  `< n_float+n_onehot` → `OneHot` (COUNTED, → typed error if a *tree split* references
  it); else → `Ctr{ctr_feature, border_index}` by walking `CtrFeatures` accumulating
  `Borders.len()`. Out-of-range gidx → `ModelError::Deserialize`.
- **AT:** unit over a hand-built `TModelTrees` (float-only stays Float; a Ctr-range
  index classifies to the right `(ctr_feature,border_index)`; one-hot-range → error).

### CTR-02 — CtrSplit reconstruction from the core flatbuffer
- **Responsibility:** for a Ctr-range split, build `CtrSplit` from the referenced
  `TCtrFeature`/`TModelCtr` (projection=`from_features(CatFeatures)`, ctr_type, prior,
  target_border_idx, border=`Borders[k]`, shift, scale). Isolates the field mapping.
- **AT:** unit — decoded `CtrSplit` fields equal the flatbuffer accessors (f32→f64
  casts; projection sorted/deduped).

### CTR-03 — ctr_data model-parts parsing
- **Responsibility:** parse the tail (`u32 count`, then `u32 size + TCtrValueTable`×count)
  into `CtrData` keyed by `ctr_base_key(ctr_type, projection.cat_features())`; decode
  `IndexHashRaw` (12-byte `(u64 hash, u32 idx)` slots, empty=`0xFFFF…`) → `hashes`
  (bucket_count = #non-empty slots, AUTHORITATIVE; the non-empty `idx` set must be
  exactly `0..bucket_count`) and `CTRBlob` per ctr_type (Borders/Buckets
  `i32×TargetClassesCount`; Counter/FeatureFreq `i32×1` + `counter_denominator`; blob
  length cross-checked against `bucket_count*width`). **Mean-type CTRs
  (`BinarizedTargetMeanValue`/`FloatTargetMeanValue`) are REJECTED with a typed error in
  v1** — their `TCtrMeanHistory` byte layout is not empirically dissected and no fixture
  exercises it (§2 defer). Isolates the tail parser.
- **AT:** unit over the committed `simple.cbm` tail — table count, per-table
  bucket_count, a known `(hash → int_counts)` slot match the empirically-dissected
  values (§ research 2b).

### CTR-04 — End-to-end load-and-predict parity (integration oracle)
- **Responsibility:** `load_cbm(simple.cbm)`/`combo.cbm` → `predict_raw_cat(model,
  float_cols, cat_cols)` reproduces upstream `predict(RawFormulaVal)` ≤1e-5.
- **AT (oracle):** `assert_abs_close(upstream_preds, rust_preds, 1e-5)` for the
  SimpleCtr and Combination fixtures. This is the invariant that proves the hash-fold
  reuse (research §9 top risk).

### CTR-05 — Robustness / typed rejection
- **Responsibility:** each of these returns a typed `ModelError` — never panic/drop:
  truncated/missing tail; an `IndexHashRaw` index set that isn't exactly `0..bucket_count`
  (gap/duplicate/out-of-range `blob_index`); a `CTRBlob` length ≠ `bucket_count*width`; a
  mean-type CTR part; a tree one-hot split index; a `(ctr_type,projection)` table-key
  collision; and `CtrFeatures` present with an empty/absent tail.
- **AT:** unit — truncated tail bytes → `Err`; a gap/dup index → `Err`; a mean-type part
  → `Err`; a synthesized one-hot split → `Err`; a duplicate key → `Err`; and the
  regression lock — a numeric `.cbm` still loads with `ctr_data: None`.

## 6. Acceptance roll-up

| Scenario | Spec | Kind | Bar |
|----------|------|------|-----|
| bin index → Float/OneHot/Ctr | CTR-01 | unit | exact |
| CtrSplit fields == flatbuffer | CTR-02 | unit | exact |
| tail → CtrData (dissected values) | CTR-03 | unit | exact |
| SimpleCtr + Combination load-predict | CTR-04 | oracle | ≤1e-5 |
| malformed/one-hot/collision → typed Err; numeric stays None | CTR-05 | unit | typed Err |

## 7. Impact

`local` (crate `cb-model`, file `cbm.rs` + a new parser; `ctr_data.rs` reuse). New
oracle fixture dir `crates/cb-oracle/fixtures/ctr_load/`. Numeric `reconstruct_model`
stays byte-identical (existing `cbm_oracle_test` round-trip + apply/json/fstr oracles
must stay green). The `catboost-rs` facade `load_cbm` inherits CTR support for free.
No public signature change; no apply change; `save_cbm` unchanged (load-only).

## 8. Risks

1. **[TOP] Hash-fold invariant** — decoded `hashes` (from `IndexHashRaw`) must equal
   apply's `fold_cat_hash(0, calc_cat_feature_hash(v))`. HIGH-confidence by
   transitivity (trainer oracle-locked, same magic), but the loader is a NEW `hashes`
   producer → CTR-04 is the end-to-end confirmation. `[research §9.1]`
2. **Dense-hash contiguity / empty marker** — assume `blob_index ∈ 0..bucket_count-1`,
   empty=`0xFFFF…`; validate + error otherwise. `[research §9.2]`
3. **One-hot / multiclass / mean-CTR** out of v1 — reject with typed error, don't
   silently mis-load. `[research §9.4/9.3/9.5]`
4. **save asymmetry** — CTR models don't survive `save→load` in v1; never extend the
   round-trip test to CTR models until save lands. `[research §9.6]`

## 9. Traceability

Research: `scratchpad/research-ctr-load.md` (empirical `.cbm` dissection, catboost
1.2.10). Fixtures pre-generated: `scratchpad/ctr_probe/{simple,combo}.{cbm,json},
*_preds.npy,X.npy,y.npy` — to be re-generated by a committed `gen_fixtures.py`.
Apply-compat: research §6. Integration points: research §8.

## 10. Implementation evidence (Phase 23 execution, `cbm-ctr-load/PLAN.md` T0-T5)

**No PageIndex-indexed document exists for this spec** (verified via
`browse_documents`/`get_document` against the `catboost_rs` PageIndex folder —
it holds only the unrelated Phase-18 `SPEC.md`); this local file is the sole
authoritative record, per the same convention Phase-18's `SPEC.md` documents
("PENDING — no writable PageIndex spec corpus is indexed for this repo").
`status: draft` in the frontmatter is UNCHANGED (lifecycle/approval is a
separate decision from implementation completion).

All five specs' acceptance tests pass in this execution and are considered
**implemented**:

| Spec | Evidence |
|------|----------|
| CTR-01 | `cb-model/src/cbm_test.rs::classify_bins_places_float_onehot_ctr_ranges`, `::classify_bins_numeric_only_model_matches_float_only_order` |
| CTR-02 | `cb-model/src/cbm_test.rs::ctr_split_from_flatbuffer_maps_fields`, `::ctr_split_from_sorts_and_dedups_combination_projection`, `::ctr_split_from_out_of_range_border_index_is_typed_error` |
| CTR-03 | `cb-model/src/cbm_test.rs::decode_ctr_parts_matches_dissected_simple`, `::decode_ctr_parts_matches_dissected_combo` |
| CTR-04 | `cb-model/tests/cbm_oracle_test.rs::cbm_load_upstream_ctr_simple_applies_within_tol`, `::cbm_load_upstream_ctr_combo_applies_within_tol` — max\|diff\| = 0 for both `simple` and `combo` (well within the 1e-5 bar) |
| CTR-05 | `cb-model/src/cbm_test.rs::decode_ctr_parts_truncated_tail_is_typed_error`, `::decode_ctr_parts_empty_tail_is_typed_error`, `::decode_ctr_parts_gap_in_bucket_indices_is_typed_error`, `::decode_ctr_parts_duplicate_bucket_index_is_typed_error`, `::decode_ctr_parts_blob_length_mismatch_is_typed_error`, `::decode_ctr_parts_mean_ctr_type_is_typed_error`, `::decode_ctr_parts_duplicate_table_key_is_typed_error`, `::one_hot_split_index_is_typed_error`, `::ctr_features_present_with_empty_tail_is_typed_error`, `::non_symmetric_model_with_ctr_features_is_typed_error`, `::numeric_only_model_still_decodes_with_ctr_data_none` (regression lock) |

Symbols touched: `crates/cb-model/src/cbm.rs` (`BinKind`, `build_combined_bins`,
`ctr_split_from`, `reconstruct_model`, `decode_cbm`); `crates/cb-model/src/ctr_data.rs`
(`decode_ctr_model_parts`, `decode_one_ctr_value_table`, `decode_index_hash_raw`,
`decode_ctr_blob`). Apply path (`apply.rs`, `ctr_data.rs` value-lookup) UNCHANGED.

Verification commands (verbatim output in the execution report):
`cargo test -p cb-model cbm`, `cargo test -p cb-model --test cbm_oracle_test`,
`cargo test -p cb-model` (full crate, no regressions), `cargo clippy -p cb-model
--lib --no-deps` (zero errors; zero new warnings from this slice's code).

Completed: 2026-07-16T22:49:59Z. Worktree:
`catboost_rs-worktrees/23-ctr-model-loading` branch `feat/23-ctr-model-loading`.
