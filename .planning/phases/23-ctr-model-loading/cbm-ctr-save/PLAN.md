# TDD Plan — CTR model **saving** (`save_cbm` emits CtrFeatures + model-parts tail)

**Inverse of** `./..//cbm-ctr-load/{SPEC,PLAN}.md` (CTR-01..05). **Crate:** `cb-model`
(`cbm.rs` save path + a new tail encoder in `ctr_data.rs`). **Apply path UNCHANGED.**
Parity bar 1e-5.

> Executor contract: strict Red→Green→Refactor, one spec per cycle. Source/test
> separation (unit tests in the existing sibling `#[cfg(test)] #[path] mod tests;`
> = `cbm_test.rs` / `ctr_data_test.rs`; oracle tests under `tests/`). NO
> `unwrap`/`expect`/`panic`/`indexing_slicing` in production; checked casts
> (`u32/i32/i64::try_from` → typed `ModelError`, never `as` truncation on a
> length/index/count). Lint gate = `cargo clippy -p cb-model --lib --no-deps`
> (NOT `cargo build`).

## Scope

### In (v1) — the mirror of the load slice
- Oblivious (symmetric) trees carrying `ModelSplit::Ctr` splits.
- SimpleCtr (1 cat feature) AND CombinationCtr (≥2 cat features) projections.
- CTR types a default categorical model emits: `Borders`, `Counter`, `FeatureFreq`
  (+ `Buckets` if present). All priors.
- **Round-trip parity**: `load_cbm(upstream ctr .cbm)` → `save_cbm` → `load_cbm` →
  `predict_raw_cat` reproduces the upstream preds ≤1e-5, AND the reloaded `Model`
  is structurally `==` the first-loaded `Model`.

### Out / deferred (typed-error or documented non-goal — never silent mis-save)
- **Upstream re-read of our saved CTR `.cbm`** is a NON-GOAL for v1 (documented).
  We emit an `IndexHashRaw` in the *minimal dense-hash form our own loader accepts*
  (one slot per bucket, `blob_index == bucket position`, no empty slots); upstream's
  exact `TDenseHash` power-of-two probing layout is NOT replicated in v1. The parity
  bar is Rust round-trip, mirroring what the load slice guarantees in reverse.
- Mean-type CTRs (`BinarizedTargetMeanValue`/`FloatTargetMeanValue`) — typed error
  on save (their `TCtrMeanHistory` layout is undissected; the load slice already
  rejects them, so no such `Model` can reach save from a load, but a hand-built one
  could → reject).
- One-hot splits; non-symmetric / Region CTR trees; multiclass CTR
  (`target_classes_count > 2`). Non-symmetric trees carrying CTR splits → typed error
  (today the non-symmetric save arm only handles `as_float`; a CTR split there must
  error, not silently write `0`).
- A CTR split whose CTR table is ABSENT from `model.ctr_data` → typed error (never
  emit a CtrFeature whose apply-time table lookup would miss).

## Key inverse facts (dissected from the load path — cite in code comments)

1. **Combined bin order** (`build_combined_bins`, cbm.rs:152): global split index space
   is `FloatFeatures bins → OneHotFeatures bins → CtrFeatures bins`. v1 has no one-hot,
   so a CTR split's global index = `n_float_bins + ctr_feature_offset + border_index`,
   where `ctr_feature_offset = Σ Borders.len()` of preceding `CtrFeatures`.
2. **`TCtrFeature` grouping**: the load path reconstructs one `CtrSplit` per
   `(ctr_feature c, border_index k)`. Inverting: a `TCtrFeature` is one unique CTR
   IDENTITY = `(projection, ctr_type, prior, target_border_idx, shift, scale)` with the
   SORTED-ASCENDING set of distinct `border` values used by the trees. Each tree
   `CtrSplit` maps to `(c = index of its identity, k = index of its border in that
   identity's sorted Borders)`. Any deterministic identity ordering round-trips (the
   loader rebuilds bins from the wire order we write); order identities by a stable key
   (e.g. `ctr_base_key` then prior/target_border_idx/shift/scale) for reproducibility.
3. **`TCtrFeature` wire fields** (inverse of `ctr_split_from`, cbm.rs:199):
   `Ctr = TModelCtr{ Base = TModelCtrBase{ FeatureCombination = TFeatureCombination{
   CatFeatures = projection.cat_features() as i32 }, CtrType = ECtrType.as_i8() },
   TargetBorderIdx, PriorNum = prior.num as f32, PriorDenom = prior.denom as f32,
   Shift = shift as f32, Scale = scale as f32 }`, `Borders = sorted borders as f32`.
   Use `crate::model_generated::ncat_boost_fbs::{TCtrFeatureArgs, TModelCtrArgs,
   TModelCtrBaseArgs, TFeatureCombinationArgs}` (builders confirmed present).
4. **Model-parts tail** (inverse of `decode_ctr_model_parts`, ctr_data.rs:527):
   `u32 LE part_count` then per table `u32 LE part_size` + a `TCtrValueTable`
   FlatBuffers buffer (its OWN finished buffer, length-prefixed). One part per entry in
   `model.ctr_data.tables` (BTreeMap → deterministic order; keys are already unique →
   no collision). `TCtrValueTable` from
   `crate::ctr_data_generated::ncat_boost_fbs::{TCtrValueTableArgs, TModelCtrBaseArgs,
   TFeatureCombinationArgs}`:
   - `ModelCtrBase`: `CtrType = ctr_type.as_i8()` (wrap into that module's transparent
     `ECtrType(i8)`), `FeatureCombination.CatFeatures` = the table's projection cat
     features. Recover the projection from the table KEY (`ctr:type=<i8>:proj=<f0>,…`)
     or carry it — parse the `proj=` CSV back to `Vec<i32>` (empty proj → empty vector).
   - `TargetClassesCount = table.target_classes_count as i32` (2 for Borders/Buckets,
     0 for Counter/FeatureFreq — matches the load-side wire value).
   - `CounterDenominator = table.counter_denominator as i32` (checked `i32::try_from`).
   - `IndexHashRaw`: `bucket_count = table.hashes.len()` slots, slot `i` =
     `hashes[i] as u64 LE (8B) ++ (i as u32) LE (4B)` (12B). No empty markers. Round-
     trips through `decode_index_hash_raw` (non-empty idx set = exactly `0..bucket_count`).
     REJECT with a typed error if any `hashes[i] == 0xFFFF_FFFF_FFFF_FFFF` (would read
     back as an empty slot — corrupts bucket_count).
   - `CTRBlob`: raw LE `i32` array, row-major `bucket_count × width`,
     `width = is_counter ? 1 : target_classes_count`; element = `i32::try_from(count)`
     (checked — a count exceeding i32 is a typed error, never `as` truncation). Cross-
     check each `int_counts[i].len() == width` → else typed error.
   - REJECT `is_mean()` tables (typed error).

## Task graph
```
T1 (CtrFeatures grouping + global index) ─┐
                                          ├─> T4 (wire into save_cbm) ─> T5 (round-trip oracle) ─> T6 (robustness)
T2 (TCtrFeature fbb build) ───────────────┤
T3 (tail encoder: TCtrValueTable + parts) ┘
```

## T1 — CtrFeatures grouping + combined global index (unit, `cbm_test.rs`)
- **Red:** `group_ctr_features_assigns_stable_indices` — build a `Model` (or a
  `Vec<&CtrSplit>`) with 2 identities, one used at 2 borders, the other at 1; assert
  grouping yields 2 identities, sorted-ascending distinct borders per identity, and a
  `(&CtrSplit) → (ctr_feature, border_index)` map consistent with
  `n_float + offset + border_index`. A float split still maps via the existing
  `split_to_global_index`.
- **Green:** add `fn build_ctr_features(model: &Model) -> Result<CtrFeaturePlan, ModelError>`
  producing (a) an ordered `Vec<CtrIdentity{projection, ctr_type, prior,
  target_border_idx, shift, scale, borders: Vec<f64>}>` and (b) a lookup from a
  `CtrSplit` (by identity+border) to its `(ctr_feature, border_index)`; plus
  `fn ctr_split_to_global_index(split:&CtrSplit, n_float_bins, plan) -> Result<i32,_>`.
  Group by an identity key that folds `border` OUT; collect+sort+dedup borders
  (`f64::total_cmp`, dedup by `to_bits`). Bounds/overflow via checked casts.
- **Validation:** `cargo test -p cb-model cbm group_ctr`.

## T2 — TCtrFeature FlatBuffers build (unit)
- **Red:** `tctr_feature_roundtrips_via_ctr_split_from` — build ONE `TCtrFeature`
  buffer from a `CtrIdentity`, wrap it with `root`-style access (or serialize into a
  minimal `TModelTrees` and read `CtrFeatures().get(0)`), call the EXISTING
  `ctr_split_from(tcf, k)` for each border index, assert the returned `CtrSplit` equals
  the original (projection, ctr_type, prior, target_border_idx, border=borders[k],
  shift, scale). This proves T2 is the exact inverse of load CTR-02.
- **Green:** `fn build_tctr_feature(fbb, identity) -> WIPOffset<TCtrFeature>` per fact 3.
- **Validation:** `cargo test -p cb-model cbm tctr_feature`.

## T3 — Tail encoder: TCtrValueTable + model-parts framing (unit, `ctr_data_test.rs`)
- **Red:** `encode_ctr_model_parts_roundtrips_via_decode` — take a `CtrData` with a
  Borders table (width 2) and a Counter table (width 1, counter_denominator set), call
  the new `encode_ctr_model_parts`, feed the bytes to the EXISTING
  `decode_ctr_model_parts`, assert the decoded `CtrData == ` the original. Also a
  `encode_rejects_mean_table` (mean → `Err`) and `encode_rejects_marker_hash`
  (a bucket hash `== 0xFFFF…` → `Err`).
- **Green:** in `ctr_data.rs`:
  - `fn build_tctr_value_table(fbb, key, table) -> Result<WIPOffset<TCtrValueTable>,_>`
    (fact 4; recover projection from `key`, or add a small `parse_ctr_base_key(key) ->
    Result<(ECtrType, Vec<i32>),_>` helper — reject a malformed key). Reject mean;
    checked casts; build `IndexHashRaw`/`CTRBlob` as `fbb.create_vector::<u8>`.
  - `pub fn encode_ctr_model_parts(ctr_data:&CtrData) -> Result<Vec<u8>,ModelError>`:
    `u32 count` then per table finish an INDEPENDENT `FlatBufferBuilder`, length-prefix
    its `finished_data()` with `u32 part_size`, append. (Each part is its own finished
    buffer — matches `root_as_tctr_value_table(part)` reading one table per part.)
  - Do NOT touch the existing self-describing `encode_ctr_data`/`decode_ctr_data`
    (unrelated LE blob, ctr_data.rs:855) — this is the UPSTREAM parts format.
- **Validation:** `cargo test -p cb-model ctr_data encode_ctr_model_parts`.

## T4 — Wire into `save_cbm` / `build_core_blob`
- **Green (`cbm.rs`):**
  - In `build_core_blob`: if `model` carries any `ModelSplit::Ctr` (or
    `model.ctr_data.is_some()`), compute `plan = build_ctr_features(model)?` and
    `n_float_bins = bins.len()`. Emit the `CtrFeatures` vector into `TModelTreesArgs`
    (`add_CtrFeatures`) built from `plan.identities` via `build_tctr_feature`.
  - In the **oblivious** split loop: replace the `if let Some(float_split)=…` that
    SKIPS CTR splits with a match — `ModelSplit::Float` → `split_to_global_index`;
    `ModelSplit::Ctr(c)` → `ctr_split_to_global_index(c, n_float_bins, &plan)`. (Both
    push exactly one `tree_splits` entry, so `TreeSizes`/offsets are unchanged.)
  - **Non-symmetric** arm: a CTR split (`split.as_float()` is `None` AND the node is
    not a pure leaf) → typed error "non-symmetric CTR save unsupported (v1)".
  - `save_cbm`: after building `core`, if `model.ctr_data.is_some()` append the tail —
    `out.extend(encode_ctr_model_parts(ctr_data)?)` AFTER the 8-byte frame + core (the
    tail lives at `buf[8 + core_len..]`, exactly where `decode_cbm` reads it). The
    `core_len` field still counts ONLY the FlatBuffers core (NOT the tail) — verify
    against `decode_cbm`'s `buf.get(8+declared..)`.
  - A numeric-only model (no CTR splits, `ctr_data None`) emits NO `CtrFeatures` and NO
    tail → **byte-identical** to today (regression lock).
- **Validation:** existing `cargo test -p cb-model --test cbm_oracle_test` numeric
  round-trip + `apply/json/fstr` oracles stay green.

## T5 — Round-trip oracle (integration, `cbm_oracle_test.rs`)
- **Red:** `cbm_ctr_simple_save_reload_applies_within_tol` and `_combo` — mirror
  `assert_upstream_ctr_load_parity` (:229): `load_cbm(ctr_load/{s}.cbm)` → `save_cbm`
  to a unique tmp → `load_cbm` back → `predict_raw_cat(&reloaded, &float, &cat)` ==
  `{s}_preds.npy` ≤1e-5; ALSO `assert_eq!(first_loaded, reloaded)` (structural).
- **Green:** falls out of T4. **Validation:** `cargo test -p cb-model --test cbm_oracle_test ctr`.

## T6 — Robustness / typed rejection (unit)
- **Red (`cbm_test.rs` / `ctr_data_test.rs`):** save a `Model` with a CTR split whose
  table is missing from `ctr_data` → `Err`; a mean-type table → `Err`; a count
  exceeding `i32` → `Err`; a marker-valued hash → `Err`; a non-symmetric tree with a
  CTR split → `Err`; and the regression lock — a numeric `Model` saves with NO tail
  (`load_cbm` reads it back `ctr_data == None`, bytes unchanged from pre-slice).
- **Green:** ensure each guard returns `ModelError`, none panic.
- **Validation:** `cargo test -p cb-model cbm && cargo test -p cb-model ctr_data &&
  cargo clippy -p cb-model --lib --no-deps`.

## Traceability
| Task | Inverse of | Kind |
|------|-----------|------|
| T1 | CTR-01 (combined bins) | unit |
| T2 | CTR-02 (CtrSplit) | unit |
| T3 | CTR-03 (tail parse) | unit |
| T4 | decode_cbm/reconstruct wiring | integration |
| T5 | CTR-04 (load-predict) | oracle |
| T6 | CTR-05 (robustness) | unit |
