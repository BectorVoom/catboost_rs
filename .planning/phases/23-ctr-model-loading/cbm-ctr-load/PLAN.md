# TDD Plan — CTR model loading (`decode_cbm` reconstructs CTR)

> ## Execution status (2026-07-17) — ✅ COMPLETE (T0–T5, specs CTR-01..05)
> Plan-checker PASS (2/2). Implemented on branch `feat/23-ctr-model-loading`.
> - **T0** frozen fixtures `crates/cb-oracle/fixtures/ctr_load/` (simple + combo `.cbm`
>   + upstream `RawFormulaVal` preds + float/cat cols + `gen_fixtures.py`). Deviation:
>   catboost's border/quantization is run-to-run nondeterministic even at `random_seed`/
>   `thread_count=1`, so `gen_fixtures.py` is best-effort provenance, NOT a CI-regen path;
>   the committed `.cbm` bytes (n=400, 1 float, 2 cat) are frozen ground truth.
> - **T1** `build_combined_bins` (Float/OneHot/Ctr) · **T2** `ctr_split_from` ·
>   **T3** `decode_ctr_model_parts` (+ `decode_index_hash_raw`/`decode_ctr_blob`; mean
>   types rejected; bucket-count authority; index-completeness) in `cbm.rs`/`ctr_data.rs`,
>   unit-tested in sibling `cbm_test.rs` · **T4** wired into `decode_cbm`/`reconstruct_model`
>   + CTR-04 oracle · **T5** robustness (mean/one-hot/collision/malformed/gap → typed
>   `ModelError`; numeric regression lock).
> - **Result:** CTR-04 **max|diff| = 0** (exact) for simple AND combo via the UNCHANGED
>   `predict_raw_cat`. 18 unit + CTR oracle tests green; full `cargo test -p cb-model` = 0
>   fail; `cargo clippy -p cb-model --lib --no-deps` = 0 new errors; `apply.rs` untouched.
> - **Out of v1 (typed-error):** one-hot splits, non-symmetric/multiclass/mean CTR, CTR
>   *saving* (`save_cbm` no-tail — CTR models don't survive save→load yet). Facade
>   `Model::load_cbm` inherits CTR load for free. This also unblocks a future FSTR-01.

**Spec:** `./SPEC.md` (CTR-01..05). **Crate:** `cb-model` (`cbm.rs` + new parser;
`ctr_data.rs` reuse). **Apply path UNCHANGED.** Parity bar 1e-5.

> Executor contract: strict Red→Green→Refactor, one spec per cycle. Source/test
> separation (unit tests in a sibling `#[cfg(test)] #[path] mod tests;`; oracle tests
> under `tests/`). NO `unwrap`/`expect`/`panic`/`indexing_slicing` in production;
> bounds-check BEFORE slicing; verifying FlatBuffers accessors only (never
> `_unchecked`). Lint gate = `cargo clippy -p cb-model --lib --no-deps` (NOT `cargo
> build`). Fixtures need the `catboost==1.2.10` venv at
> `…/scratchpad/cbenv/bin/python`.

## Validation
```
cargo test  -p cb-model cbm            # unit + oracle for this slice
cargo test  -p cb-model --test cbm_oracle_test
cargo clippy -p cb-model --lib --no-deps
```
Numeric path must stay byte-identical: `cbm_oracle_test` (round-trip + upstream
binclf load), `apply_oracle_test`, `json_oracle_test`, `advanced_fstr_oracle_test`
must stay green.

## Task graph
```
T0 fixtures ─┬─> T3 (ctr_data parse) ─┐
             └─> T4 (e2e oracle) <─────┼── T1 (bin classify) ─> T2 (CtrSplit build) ─┘
                                        └─> T5 (robustness)
```
T1→T2 serial (classification underpins CtrSplit build); T0→T3, T0→T4; T4 integrates
T1/T2/T3; T5 after T4.

## T0 — Oracle fixtures (enabler) — FROZEN, fully pinned
- **Files (new):** `crates/cb-oracle/fixtures/ctr_load/gen_fixtures.py` (+ committed
  artifacts). The script MUST be **fully deterministic and pinned** so the model bytes
  (and thus T3's dissected constants) are reproducible — pin EVERY input, not just the
  model params:
  - **Data:** `np.random.RandomState(0)`, `n=200`, 2 float cols + 2 cat cols; target
    `y = ((f0 + (c0==2) + noise) > 0.5)` (the exact cat-DRIVEN construction from the
    research probe `train.py` — a weak cat signal yields ZERO CTR splits, research §7).
    Record the exact feature layout + `y` formula in a header comment.
  - **Params:** `bootstrap_type="No", l2_leaf_reg=3.0, learning_rate=0.1,
    random_seed=0, random_strength=0, thread_count=1, depth=4, iterations=10,
    logging_level="Silent", allow_writing_files=False`, `cat_features=[2,3]`.
  - **simple**: `max_ctr_complexity=1`. **combo**: `max_ctr_complexity=2`.
  - Save `simple.cbm`/`combo.cbm` (+ `.json` for cross-check); dump
    `predict(X, prediction_type='RawFormulaVal')` → `*_preds.npy`; save float columns
    `X_float.npy` and cat columns as **A4 plain-integer strings** (`str(int(v))`, the
    same stringification apply's `calc_cat_feature_hash` expects) → `cat_cols.json`;
    `config.json` (params, `cat_features`, `n_ctrs`, per-part `ctr_type` list).
- **FROZEN:** commit the generated `.cbm`/`.npy`/`.json` as ground-truth artifacts;
  **CI does NOT regenerate them** (no catboost in CI). `gen_fixtures.py` documents
  reproducibility only. T3's byte-level constants are dissected from the COMMITTED
  `simple.cbm` (record them in the test with a comment tying each to the committed file).
- **Verify:** re-load each `.npy`; assert the committed model actually has `OnlineCtr`
  splits and record the tail parts' `ctr_type` set (expect {0 Borders, 4 Counter}) in
  `config.json` so MAJOR-2's mean-rejection scope is evidenced.
- **Done:** committed `.cbm`/`.npy`/`.json` + loadable; `config.json` lists ctr_types.

## T1 — CTR-01 combined bin classification (unit)
- **Spec:** CTR-01. **Depends:** T0 (only for a real `trees` in the unit test; can also
  hand-build).
- **Red** (`cbm_test.rs` sibling unit): `classify_bins_places_float_onehot_ctr_ranges`
  — build/borrow a `TModelTrees` with `k` float borders + `m` ctr borders (no one-hot);
  assert a float-range gidx → `BinKind::Float{feature,border}`, a ctr-range gidx →
  `BinKind::Ctr{ctr_feature,border_index}`, and an out-of-range gidx → `Err`. If a
  one-hot table exists, a one-hot-range index → the `OneHot` kind.
- **Green:** add `enum BinKind { Float{feature,border}, OneHot, Ctr{ctr_feature,border_index} }`
  and `fn build_combined_bins(trees:&TModelTrees) -> Result<Vec<BinKind>, ModelError>`:
  push one `Float` per `FloatFeatures[f].Borders`, one `OneHot` per
  `OneHotFeatures[o].Values` (COUNTED only), one `Ctr{ctr_feature=c, border_index=k}`
  per `CtrFeatures[c].Borders[k]`. Keep the existing float-only `build_bin_features`
  for the SAVE path untouched.
- **Refactor:** none. **Validation:** `cargo test -p cb-model cbm classify`.

## T2 — CTR-02 CtrSplit reconstruction (unit)
- **Spec:** CTR-02. **Depends:** T1.
- **Red** (`cbm_test.rs`): `ctr_split_from_flatbuffer_maps_fields` — from a
  `CtrFeatures[c]` + border index, build a `CtrSplit` and assert
  `projection.cat_features() == sorted(CatFeatures)`, `ctr_type == from_i8(CtrType)`,
  `prior == (PriorNum,PriorDenom)`, `target_border_idx == TargetBorderIdx`,
  `border == Borders[k] as f64`, `shift/scale` cast. (Build the `TModelTrees` from the
  committed `simple.cbm` core, or hand-serialize a minimal `TCtrFeature`.)
- **Green:** `fn ctr_split_from(feature: TCtrFeature, border_index: usize) ->
  Result<CtrSplit, ModelError>`: read `Ctr()` (`TModelCtr`) → `Base()`
  (`TModelCtrBase`) → `FeatureCombination().CatFeatures()`; `TProjection::from_features`;
  `ECtrType::from_i8`; `Prior{num,denom}`; f32→f64 casts; `Borders().get(k)`. All
  checked `.get`/`ok_or_else` — typed error on any missing field.
  - **Imports (MINOR-1):** the CORE-side accessors `TCtrFeature`/`TModelCtr`/
    `TModelCtrBase`/`TFeatureCombination` come from `crate::model_generated::ncat_boost_fbs`
    (same module `reconstruct_model` already uses). `TModelCtrBase.CtrType()` returns
    the generated transparent `ECtrType(pub i8)`, so convert via
    `crate::ctr_data::ECtrType::from_i8(base.CtrType().0)` (note the `.0`).
- **Validation:** `cargo test -p cb-model cbm ctr_split`.

## T3 — CTR-03 ctr_data model-parts parsing (unit)
- **Spec:** CTR-03. **Depends:** T0.
- **Red** (`cbm_test.rs`): `decode_ctr_parts_matches_dissected_simple` — read the
  COMMITTED `simple.cbm` (frozen T0 artifact), take the tail (`buf[8+declared..]`), call
  `decode_ctr_model_parts`, and assert the values dissected FROM THE COMMITTED FILE
  (record each with a comment tying it to that file; the research-probe values are
  `table count == 3`, the `(Borders, proj=[0])` table `bucket_count==5` with slot
  `hash 0xdae37c9ea4a59432 → int_counts [36,47]`, the `(Counter, proj=[0])` table
  `counter_denominator==84` — re-confirm against the committed bytes at implementation
  time, they change if the fixture changes). Also assert the parsed `hashes` indices are
  exactly `0..bucket_count` (completeness).
- **Green:** `fn decode_ctr_model_parts(tail:&[u8]) -> Result<CtrData, ModelError>`:
  - **Imports (MINOR-1):** the tail structs are a DIFFERENT generated module —
    `crate::ctr_data_generated::ncat_boost_fbs::{root_as_tctr_value_table, TCtrValueTable}`
    (NOT `model_generated`); its `ModelCtrBase().CtrType()` also returns
    `ECtrType(pub i8)` → `ECtrType::from_i8(base.CtrType().0)`.
  - Read `u32 count`; per part read `u32 size` (bounds-check `size` against remaining)
    + verifying `root_as_tctr_value_table`.
  - **IndexHashRaw** (12-byte `(u64 hash LE, u32 idx LE)` slots, empty marker
    `hash==0xFFFF_FFFF_FFFF_FFFF`): collect the non-empty `(idx, hash)` pairs;
    `bucket_count = n_nonempty` is AUTHORITATIVE (MINOR-3); build `hashes` sized
    `bucket_count`; validate the non-empty `idx` set is **exactly `0..bucket_count` with
    no gap/duplicate** (track a filled `bool` per slot; MINOR-2) → else
    `ModelError::Deserialize`.
  - **CTRBlob** per ctr_type: `Borders`/`Buckets` → `i32` width `w = TargetClassesCount`
    (binclf 2); `Counter`/`FeatureFreq` → **force `w = 1`** (their `TargetClassesCount==0`;
    never divide by 0), `counter_denominator` from the field; **cross-check**
    `blob.len()/4 == bucket_count * w` → else error (MINOR-3).
  - **REJECT mean ctr_types** (`BinarizedTargetMeanValue`/`FloatTargetMeanValue`, and any
    ctr_type outside {Borders,Buckets,Counter,FeatureFreq}) with a typed
    `ModelError::Deserialize("mean/target-mean CTR unsupported (v1)")` — the
    `TCtrMeanHistory` byte layout is NOT empirically dissected and no fixture exercises
    it (SPEC §2 defers it; MAJOR-2). Do NOT fill `CtrValueTable.mean` from unverified bytes.
  - Key each table by `ctr_base_key(ctr_type, projection.cat_features())`; **error on a
    duplicate key** (collision → CTR-05). Bounds-check every slice; no `_unchecked`. Do
    NOT reuse the unrelated LE-blob `decode_ctr_data` (research §4 NOTE).
- **Validation:** `cargo test -p cb-model cbm decode_ctr_parts`.

## T4 — wire into decode_cbm + CTR-04 end-to-end oracle
- **Spec:** CTR-04. **Depends:** T1, T2, T3.
- **Red** (`crates/cb-model/tests/cbm_oracle_test.rs`, mirror
  `cbm_load_upstream_binclf_applies_within_tol` :173): `cbm_load_upstream_ctr_simple_applies_within_tol`
  and `_combo` — `load_cbm(ctr_load/simple.cbm)` → `predict_raw_cat(&model,
  &float_cols, &cat_cols)`; `assert_abs_close(upstream_preds, rust, 1e-5)`. Fails today
  (CTR splits dropped → decode error or wrong prediction).
- **Green:** 
  - `decode_cbm`: capture the tail `buf.get(8usize.saturating_add(declared)..)`;
    pass it to `reconstruct_model`.
  - `reconstruct_model(trees, class_to_label, tail: &[u8])`: replace `build_bin_features`
    with `build_combined_bins`; in the split loop, match the classified bin — `Float` →
    existing `ModelSplit::Float`; `Ctr{c,k}` → `ModelSplit::Ctr(ctr_split_from(CtrFeatures[c],k))`;
    `OneHot` → `ModelError::Deserialize("one-hot split unsupported")` (CTR-05). Set
    `ctr_data = if CtrFeatures non-empty { Some(decode_ctr_model_parts(tail)?) } else { None }`.
    **A present-but-empty/absent tail when `CtrFeatures` is non-empty is a typed error**
    (`decode_ctr_model_parts(&[])` reads `count` from too-few bytes → `ModelError`);
    `decode_cbm` passes `buf.get(8+declared..).unwrap_or(&[])` so the empty slice reaches
    the parser as an error, never a panic.
  - Non-symmetric early-return path: if that model carries `CtrFeatures`, return a typed
    "non-symmetric CTR unsupported (v1)" error rather than silently `None`.
- **Refactor:** keep the numeric (`CtrFeatures` empty) branch byte-identical (bins are
  all Float; ctr_data None). **Validation:** `cargo test -p cb-model --test cbm_oracle_test`.

## T5 — CTR-05 robustness / typed rejection (unit)
- **Spec:** CTR-05. **Depends:** T4.
- **Red** (`cbm_test.rs`): truncated tail (drop last N bytes) → `Err`; a `.cbm` whose
  tree references a one-hot-range index → `Err` (synthesize or note if no fixture);
  `blob_index ≥ bucket_count` OR a gap/duplicate index set (not exactly `0..bucket_count`)
  → `Err` (MINOR-2); a `CTRBlob` length not matching `bucket_count*width` → `Err`
  (MINOR-3); a mean ctr_type part → `Err` (MAJOR-2); a duplicate `(ctr_type,proj)` part
  → `Err`; and the regression lock — a NUMERIC `.cbm` still decodes with
  `ctr_data == None`.
- **Green:** ensure each guard above returns `ModelError`, none panic.
- **Validation:** `cargo test -p cb-model cbm` (+ full slice `cargo clippy`).

## Traceability
| Task | Spec | Kind |
|------|------|------|
| T0 | (enabler) | fixtures |
| T1 | CTR-01 | unit |
| T2 | CTR-02 | unit |
| T3 | CTR-03 | unit |
| T4 | CTR-04 | oracle |
| T5 | CTR-05 | unit |
