---
title: TDD implementation plan — Part 1 waves W4–W5 (tasks E18–E23)
parent: ./PLAN.md
spec: ./SPEC.md
status: ready-for-implementation
---

# Part 1 — Engine, waves W4–W5

Continuation of `./PLAN.md` and `./PLAN-W2-W3.md`. §3 (shared conventions), §3.1
(mutation-check protocol), §3.2 (verified commands) and §4 (waves + edge list) of
`./PLAN.md` apply verbatim.

---

## WAVE W4 — `.cbm` mean-CTR codec (LOCKED USER DECISION)

> Lifting the v1 mean restriction is a real serialization sub-phase. It has its
> own byte-identity oracle (E00, already frozen in W0) so the non-mean regression
> gate is **not** a tautology, and it is proven against an **upstream-produced**
> `.cbm` rather than by self-comparison alone (risk R7).
>
> The two rejection sites, verified verbatim:
> - **DECODE** — `decode_one_ctr_value_table`:
>   `if ctr_type.is_mean() { return Err(ModelError::Deserialize("mean/target-mean CTR unsupported (v1, MAJOR-2)")) }`
> - **ENCODE** — `build_tctr_value_table`:
>   `if table.ctr_type.is_mean() { return Err(ModelError::Serialize("mean/target-mean CTR unsupported on save (v1, MAJOR-2)")) }`
>
> `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs, both read verbatim]`

---

### E18 — Upstream BTMV `.cbm` fixture (data lands before the codec)

- **Specs:** SPEC-CTRT-15 (data half); acceptance **A8** second half
- **Blocked by:** E13 (its generator already trains the BTMV model). **Blocks:** E19.
- **Parallelizable:** **YES** with E17 and with all of W5 — owns only the fixture
  directory; no production code.

**Goal / observable completion condition.** `crates/cb-oracle/fixtures/ctr_btmv_simple/model.cbm`
exists, is committed, and is **proven** to contain a mean-typed CTR table — so
E19's Red is a real upstream-format failure, not a self-produced one.

**Files**
- Modify: `crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py` (add the
  `model.cbm` emission + the new assertion)
- Create + COMMIT: `crates/cb-oracle/fixtures/ctr_btmv_simple/model.cbm`
- Modify: `crates/cb-oracle/fixtures/ctr_btmv_simple/config.json` (`npy_schema`
  gains a `model.cbm` entry)

**Exact verified files/symbols to touch**
- `.cbm` is the interop target and the existing corpus already commits upstream
  `.cbm` files — `crates/cb-oracle/fixtures/one_hot_train/default_binary/model.cbm`
  with the `npy_schema` note `"model.cbm": "upstream one-hot model, FlatBuffers (the interop target)"`
  `[VERIFIED: LOCAL, read in full]`, and
  `crates/cb-oracle/fixtures/ctr_load/{simple,combo}.cbm`
  `[VERIFIED: LOCAL ctr_load/gen_fixtures.py docstring]`.
- Generator call: `model.save_model(path, format="cbm")`.
- The wire type to assert: `ECtrType::BinarizedTargetMeanValue = 2`
  `[VERIFIED: LOCAL crates/cb-model/src/generated/ctr_data_generated.rs:204]`.

**MANDATORY anti-false-pass guard (added to the generator):**
```python
# The .cbm must actually carry a mean-typed CTR table, else E19 is untestable.
reloaded = catboost.CatBoost().load_model(cbm_path)
j = json.loads(reloaded.... )   # or re-read the sibling model.json
assert any(c["ctr_type"] == "BinarizedTargetMeanValue"
           for c in j["features_info"]["ctrs"]), \
    "model.cbm carries no BinarizedTargetMeanValue CTR — E19/E20 would be vacuous"
```
Plus the §3 corpus-cleanliness guard.

**Red — DOUBLE-GENERATION DETERMINISM CHECK (the falsifiability requirement for a
data-only task).** Regenerate twice into the scratchpad and `diff -r`.
**Expected:** empty. **If `predictions.npy` differs, STOP AND REPORT** — a
nondeterministic reference cannot be an oracle. Because the fixture is
categorical-only, float-quantization nondeterminism is structurally excluded.

**Green.** Emit `model.cbm` from the already-frozen E13 training run (do NOT
retrain — reuse the same `CatBoost` object in the same generator invocation so the
`.cbm` and the `model.json` / `predictions.npy` describe the identical model).
Commit.

**Refactor constraints + required regression scope**
- Constraint: E13's five existing artifacts must be **byte-unchanged** by this
  task — assert with `git diff --stat crates/cb-oracle/fixtures/ctr_btmv_simple`
  showing only `model.cbm` (new) and `config.json` (schema line). If any `.npy`
  changed, the generator retrained: **STOP AND REPORT**.
- Regression scope: `cargo test -p cb-train --test ctr_btmv_simple_oracle_test`.

**Validation**
```bash
.venv/bin/python crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py
git status --porcelain crates/cb-oracle/fixtures     # ONLY ctr_btmv_simple/*
git diff --stat crates/cb-oracle/fixtures/ctr_btmv_simple   # only config.json
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
```

**Completion evidence.** `model.cbm` committed; the generator's mean-type assertion
passing; `git diff --stat` proving no `.npy` moved.

---

### E19 — `.cbm` DECODES mean-type CTR tables

- **Specs:** SPEC-CTRT-15 (decode half); acceptance **A8**
- **Blocked by:** E11, E18. **Blocks:** E20.
- **Parallelizable:** **NO** — owns `crates/cb-model/src/ctr_data.rs`.

**Goal / observable completion condition.** An upstream `.cbm` carrying a
`BinarizedTargetMeanValue` table **loads** and **predicts within 1e-5**, replacing
the `ModelError::Deserialize("mean/target-mean CTR unsupported (v1, MAJOR-2)")`
rejection.

**Files**
- Modify: `crates/cb-model/src/ctr_data.rs`
- Modify: `crates/cb-model/src/ctr_data_test.rs` (exists
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data_test.rs]`)
- Create: `crates/cb-model/tests/ctr_mean_cbm_oracle_test.rs`

**Exact verified files/symbols to touch**
- **The rejection to replace:** in `decode_one_ctr_value_table`,
  ```rust
  let ctr_type = ECtrType::from_i8(base.CtrType().0)?;
  if ctr_type.is_mean() {
      return Err(ModelError::Deserialize(
          "mean/target-mean CTR unsupported (v1, MAJOR-2)".to_owned(),
      ));
  }
  ```
  `[VERIFIED: LOCAL, read verbatim]`.
- `fn decode_ctr_blob(vt, bucket_count, width) -> Result<Vec<Vec<i64>>, ModelError>`
  at `crates/cb-model/src/ctr_data.rs:711-745` — reads `CTRBlob` as a raw
  little-endian **`i32`** array, checks `blob.len() % 4 == 0` and
  `n_i32 == bucket_count * width` `[VERIFIED: LOCAL, read verbatim]`. The mean
  decoder is its **sibling**, not a modification of it.
- The mean wire layout: upstream `TCtrMeanHistory { float Sum; int Count; }`
  (`catboost/libs/model/online_ctr.h:380-401`) ⇒ the `CTRBlob` byte array is
  `f32 Sum` + `i32 Count` **pairs**, 8 bytes per bucket
  `[INFERRED: research §F.4 + SPEC-CTRT-14 — read from the C++ struct, NOT from a
  hex dump of an upstream .cbm]`.
- **MANDATORY STRIDE-AMBIGUITY BRANCH (the inference above is NOT verified).**
  This repo's **own** self-describing CTR format uses `f32 LE Sum ; i64 LE Count` —
  a **12-byte** stride `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:947]`. If
  upstream's `.cbm` also used a 12-byte stride, an 8-byte decoder would fail the
  length check on the E18 fixture. Therefore the decoder MUST carry an explicit
  two-branch probe:
  1. try stride **8** (`f32` Sum + `i32` Count): accept iff
     `blob.len() % 8 == 0 && blob.len() / 8 == bucket_count`;
  2. if that fails, try stride **12** (`f32` Sum + `i64` Count): check
     `blob.len() % 12 == 0 && blob.len() / 12 == bucket_count`;
  3. **if stride 12 matches, STOP AND REPORT** — do NOT silently adopt it. A
     12-byte upstream stride invalidates SPEC-CTRT-14's stated wire layout and E20's
     8-byte encoder, and both must be re-specified before proceeding. Report the
     observed `blob.len()`, the `bucket_count`, and which stride matched.
  4. if **neither** matches, return the typed
     `ModelError::Deserialize` naming `blob.len()`, `bucket_count`, and both
     candidate strides.
  **Never silently pick one.**
- `width` is computed as `if ctr_type.is_counter() { 1 } else { target_classes_count }`
  with a `width == 0` rejection `[VERIFIED: LOCAL, read verbatim]` — the mean
  branch must **bypass** that width logic entirely (a mean table has one
  `(Sum, Count)` pair per bucket regardless of `TargetClassesCount`).
- `CtrValueTable.mean: Vec<(f32, i64)>` already exists
  `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:169-170]`, with
  `mean_at(bucket) -> Option<(f32, i64)>` (`:192-196`) and the apply arm
  `ECtrType::BinarizedTargetMeanValue | ECtrType::FloatTargetMeanValue =>
  Calc(Sum, Count)` (`:249-256`) — **the apply path is already complete**.
- `decode_index_hash_raw(vt)` supplies `bucket_count` and is type-agnostic
  `[VERIFIED: LOCAL, called at decode_one_ctr_value_table]`.

**CodeGraph evidence.** `ECtrType` (`crates/cb-model/src/ctr_data.rs:74`) has
**8 callers** in `cbm.rs`, `lib.rs`, `model.rs` with covering tests
`ctr_data_test.rs`, `export/coreml_test.rs`, `export/onnx_test.rs`, `fstr_test.rs`
(+1) `[VERIFIED: CODEGRAPH]`. `is_mean()` (`:118`) is consumed by exactly the two
rejection sites plus the JSON serde `[VERIFIED: LOCAL grep]`.

**Red**
- File: `crates/cb-model/tests/ctr_mean_cbm_oracle_test.rs`
- Test fn 1: `upstream_btmv_cbm_loads_without_a_mean_rejection`
  Setup: `cb_model::load_cbm(fixture("ctr_btmv_simple/model.cbm"))`.
  Expected: `Ok(model)` with `model.ctr_data.is_some()`, and the mean table found:
  ```rust
  let t = model.ctr_data.as_ref().unwrap().tables.values()
      .find(|t| t.ctr_type == ECtrType::BinarizedTargetMeanValue)
      .expect("no BTMV table in the loaded .cbm");
  assert_eq!(t.mean.len(), t.hashes.len(), "one (Sum, Count) pair per bucket");
  assert!(t.int_counts.is_empty(), "a mean table carries no int_counts");
  assert!(t.mean.iter().any(|&(s, c)| s != 0.0 && c != 0),
          "all-zero mean table — the blob was not actually decoded");
  ```
  The third assertion is the anti-vacuity guard: a decoder that returns an
  all-zero `mean` of the right length would otherwise pass.
- Test fn 2: `upstream_btmv_cbm_predicts_within_1e_minus_5`
  Setup: the loaded model + `ctr_btmv_simple/X_cat.npy` (stringified) →
  `cb_model::predict_raw_cat`.
  Expected: `max |ours − ctr_btmv_simple/predictions.npy| <= 1e-5`, reported; plus
  the non-degeneracy guard `assert!(ours.iter().any(|v| *v != ours[0]))`.
- Test fn 3 (`ctr_data_test.rs`): `mean_blob_decoder_rejects_a_length_matching_neither_stride`
  Expected: for a blob length that is a multiple of **neither** 8 nor 12 (e.g.
  `bucket_count * 8 + 3`), `ModelError::Deserialize` naming the length, the bucket
  count and **both** candidate strides — the mean blob's analogue of
  `decode_ctr_blob`'s `% 4` guard. Never a panic.
- Test fn 3b (`ctr_data_test.rs`, **the stride-ambiguity pin**):
  `mean_blob_decoder_reports_rather_than_silently_accepting_a_twelve_byte_stride`
  Expected: for a blob of exactly `bucket_count * 12` bytes, the decoder does NOT
  return an `Ok` decoded at stride 12 — it returns a typed `Deserialize` whose
  message says the 12-byte (`f32 Sum ; i64 Count`) stride matched and that
  SPEC-CTRT-14's inferred 8-byte layout must be re-specified before proceeding
  (`crates/cb-model/src/ctr_data.rs:947` is this repo's own 12-byte precedent).
- **EXPECTED INITIAL FAILURE:** test fn 1 —
  `Err(ModelError::Deserialize("mean/target-mean CTR unsupported (v1, MAJOR-2)"))`
  surfaced as ``called `Result::unwrap()` on an `Err` value`` in the test harness.
- Run: `cargo test -p cb-model --test ctr_mean_cbm_oracle_test`

**Green (minimal implementation intent).** Replace the `is_mean()` rejection with a
branch: when `ctr_type.is_mean()`, call a new sibling
`fn decode_ctr_mean_blob(vt, bucket_count) -> Result<Vec<(f32, i64)>, ModelError>`
that
- reads `CTRBlob` bytes;
- **runs the stride-ambiguity branch above**: stride 8 first; on a length-check
  failure, test stride 12 and — if 12 matches — **STOP AND REPORT** rather than
  decoding; if neither matches, return a typed `Deserialize` naming `blob.len()`,
  `bucket_count` and **both** candidate strides;
- per bucket (stride-8 path) reads `f32::from_le_bytes` then `i32::from_le_bytes`,
  pushing `(sum, i64::from(count))`;
- uses checked `.get(..)` + `<[u8; 4]>::try_from(..).ok()` exactly as
  `decode_ctr_blob` does — **no indexing, no panic**.
Then set `int_counts: Vec::new()`, `mean: decoded`, and skip the `width == 0`
rejection for mean types.

**Refactor constraints + required regression scope**
- **Constraint (the branch predicate — MANDATORY, no implementer discretion):** the
  mean codec branch **MUST key on `ctr_type.is_mean()`**, which covers
  **`FloatTargetMeanValue` as well as `BinarizedTargetMeanValue`** — **NOT** on
  `BinarizedTargetMeanValue` alone. Rationale, recorded so it cannot be re-litigated:
  the inverted `crates/cb-model/src/ctr_data_test.rs` test that E20 rewrites
  (`encode_ctr_model_parts_rejects_mean_table`, `:197-212`) builds a
  **`FloatTargetMeanValue`** table, while the E18 training fixture is
  **`BinarizedTargetMeanValue`** — so a `BinarizedTargetMeanValue`-only branch
  passes one and fails the other (in either direction), and the two would silently
  disagree about which tables the codec accepts.
- Constraint: `decode_ctr_blob` must be left byte-identical (non-mean path
  unchanged) — E00's baseline gate proves it.
- Constraint: do NOT touch the ENCODE side in this task (that is E20), so E13's
  `btmv_save_cbm_is_a_typed_rejection_until_e20` still passes here.
- Regression scope: `cargo test -p cb-model` in full, especially
  `cbm_oracle_test`, `json_oracle_test`, `ctr_data_roundtrip_test`,
  `float_only_byte_identity_test`, `ctr_nonmean_byte_identity_test`, and the
  `ctr_load` consumers.

**Validation**
```bash
cargo test -p cb-model --test ctr_mean_cbm_oracle_test
cargo test -p cb-model --lib ctr_data_test
cargo test -p cb-model --test cbm_oracle_test --test json_oracle_test \
  --test ctr_data_roundtrip_test --test float_only_byte_identity_test \
  --test ctr_nonmean_byte_identity_test --test fstr_ctr_oracle_test
cargo test -p cb-model
cargo clippy -p cb-model --all-targets
```

**Completion evidence.** Upstream `.cbm` loads **at the recorded stride** (state
explicitly which of 8 / 12 matched the E18 fixture — if 12, this task STOPS AND
REPORTS instead of completing); the three structural assertions (including the
all-zero anti-vacuity guard) pass; ≤1e-5 with the recorded max-divergence; the
malformed-blob and 12-byte-stride typed rejections; all `cb-model` targets green
including both byte-identity baselines.

---

### E20 — `.cbm` ENCODES mean-type CTR tables + round-trip + the non-mean gate

- **Specs:** SPEC-CTRT-14, SPEC-CTRT-15 (round-trip half), SPEC-CTRT-16;
  acceptance **A8**, **A9**
- **Blocked by:** E00, E19. **Blocks:** none.
- **Parallelizable:** **NO** — owns `crates/cb-model/src/ctr_data.rs`.

**Goal / observable completion condition.** `save_cbm` encodes a mean table as
`f32 Sum` + `i32 Count` pairs into the raw `CTRBlob`; `save → load → save` is
**byte-identical**; and the E00 frozen non-mean baseline is **still byte-identical**
(the regression gate that makes this change safe).

**Files**
- Modify: `crates/cb-model/src/ctr_data.rs`
- Modify: **`crates/cb-model/src/ctr_data_test.rs`** — **INVERT the existing green
  test** `encode_ctr_model_parts_rejects_mean_table` at
  `crates/cb-model/src/ctr_data_test.rs:197-212`
  `[VERIFIED: LOCAL, read verbatim — it asserts `encode_ctr_model_parts(&CtrData {
  tables }).is_err()`]`. Lifting the encode rejection **breaks this test**, and it
  was not listed anywhere in the plan before this revision. **Do NOT delete it.**
  Rename it to `encode_ctr_model_parts_round_trips_a_mean_table` and rewrite the
  body as `encode_ctr_model_parts(&CtrData { tables })` → `decode_ctr_model_parts`
  → `assert_eq!` on the recovered `mean` vector (plus `assert!(!mean.is_empty())`
  as the anti-vacuity guard). The unit coverage of the encoder's mean path must be
  **inverted, not removed**.
- Modify: `crates/cb-model/tests/ctr_mean_cbm_oracle_test.rs`
- Modify: `crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs` (flip E13's
  save-rejection test into a save-success test)

**Exact verified files/symbols to touch**
- **The rejection to replace:** at the top of `build_tctr_value_table`,
  ```rust
  if table.ctr_type.is_mean() {
      return Err(ModelError::Serialize(
          "mean/target-mean CTR unsupported on save (v1, MAJOR-2)".to_owned(),
      ));
  }
  ```
  `[VERIFIED: LOCAL, read verbatim]`.
- **TWO doc blocks state the mean rejection, and BOTH must be updated in the same
  edit or each becomes a documentation lie:**
  1. the encoder's module doc block (immediately above `parse_ctr_base_key`) at
     `crates/cb-model/src/ctr_data.rs:756-758`, which states
     `"Mean tables and marker-valued hashes are rejected (v1) — never a silent mis-save."`
     `[VERIFIED: LOCAL, read verbatim]`;
  2. the **`# Errors` doc block on `build_tctr_value_table` at
     `crates/cb-model/src/ctr_data.rs:801-806`**, which states
     `"ModelError::Serialize on a mean-type table (undissected TCtrMeanHistory layout, v1)"`
     `[VERIFIED: LOCAL, read verbatim]`. This is a **public `# Errors` contract**
     and it was not listed in the plan before this revision.
- `IndexHashRaw` is written as one 12-byte `(u64 hash LE, u32 blob_index = bucket
  position LE)` slot per bucket, with the
  `hash == EMPTY_HASH_MARKER (0xFFFF_FFFF_FFFF_FFFF)` rejection
  `[VERIFIED: LOCAL, read verbatim]` — **type-agnostic; reuse unchanged for mean
  tables**.
- The per-bucket count-width check (`width = counter?1:target_classes_count`) and
  the `i32` overflow rejections in the `CTRBlob` writer must be **bypassed** for
  mean tables and replaced by their `(f32, i64)` analogue: reject
  `count > i32::MAX` with a typed `Serialize` naming the bucket.
- `TCtrValueTable.CTRBlob` is `Vector<u8>` in the generated schema
  (`VT_CTRBLOB = 8`) `[VERIFIED: CODEGRAPH crates/cb-model/src/generated/ctr_data_generated.rs:2232,2270-2275]`
  — a raw byte array, so an 8-byte-per-bucket mean layout needs **no schema
  change**.
- `TargetClassesCount` for a mean table: upstream writes the real class count;
  the decoder (E19) ignores it for mean types. Write `table.target_classes_count`
  unchanged so an upstream-written and a repo-written table agree.

**CodeGraph evidence.** `build_tctr_value_table` is reached only from the `.cbm`
encoder path in `crates/cb-model/src/ctr_data.rs`; `cbm.rs` consumes the finished
parts `[VERIFIED: CODEGRAPH `Ctr` in `crates/cb-model/src/cbm.rs`]`. Encode is
therefore isolated from the JSON serde, which already handles mean tables with
stride 3 `[VERIFIED: research §F.4]`.

**Red**
- File: `crates/cb-model/tests/ctr_mean_cbm_oracle_test.rs`
- Test fn 4: `mean_ctr_cbm_save_load_save_is_byte_identical`
  Setup: load `ctr_btmv_simple/model.cbm` (E19 path) → `save_cbm` → `load_cbm` →
  `save_cbm` again.
  Expected: `assert_eq!(first_save, second_save)` byte-for-byte, plus
  `assert_eq!(reloaded.ctr_data, once_loaded.ctr_data)` proving the mean vectors
  survived exactly (`Vec<(f32, i64)>` compares bitwise under `PartialEq` for
  non-NaN `f32`; assert no `NaN` first).
- Test fn 5: `mean_blob_is_eight_bytes_per_bucket`
  Expected: the produced `CTRBlob` length `== 8 * bucket_count` — pins the wire
  layout so a future stride change is a test failure, not a silent
  incompatibility.
- Test fn 6: `saving_a_mean_table_whose_count_exceeds_i32_is_a_typed_error`
  Expected: `ModelError::Serialize` naming the bucket. Never a panic.
- File: `crates/cb-model/src/ctr_data_test.rs` — **the INVERTED existing test**
  Rename `encode_ctr_model_parts_rejects_mean_table` (`:197-212`) to
  `encode_ctr_model_parts_round_trips_a_mean_table` and replace its
  `assert!(… .is_err())` with:
  ```rust
  let parts = encode_ctr_model_parts(&CtrData { tables })
      .expect("mean tables are encodable after SPEC-CTRT-14");
  let back = decode_ctr_model_parts(&parts).expect("decode");
  let t = back.tables.values().next().expect("one table");
  assert!(!t.mean.is_empty(), "anti-vacuity: an empty mean vector would round-trip trivially");
  assert_eq!(t.mean, original_mean, "the (f32, i64) pairs must survive exactly");
  ```
  **This test is REQUIRED, not optional:** without it the only unit coverage of the
  encoder's mean path disappears the moment the rejection is lifted.
- File: `crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs`
- Test fn 4 (**flipped** from E13): rename
  `btmv_save_cbm_is_a_typed_rejection_until_e20` →
  `btmv_trained_model_round_trips_through_cbm`, asserting `save_cbm(&trained)` is
  `Ok` and the reloaded model predicts within 1e-5 of the in-memory one.
- **THE REGRESSION GATE (must be re-run, unmodified, as part of this task's Red
  and Green):** `cargo test -p cb-model --test ctr_nonmean_byte_identity_test`
  (E00) and `--test float_only_byte_identity_test`.
- **EXPECTED INITIAL FAILURE:** test fn 4 —
  `Err(ModelError::Serialize("mean/target-mean CTR unsupported on save (v1, MAJOR-2)"))`.
- Run: `cargo test -p cb-model --test ctr_mean_cbm_oracle_test`

**Green (minimal implementation intent).** Replace the `is_mean()` rejection with a
branch that writes the mean `CTRBlob`: per bucket, `sum.to_le_bytes()` (4 bytes)
followed by `i32::try_from(count).map_err(|_| ModelError::Serialize(...))?.to_le_bytes()`
(4 bytes). Everything else in `build_tctr_value_table` — `IndexHashRaw`, the
`EMPTY_HASH_MARKER` guard, `CounterDenominator`, `TargetClassesCount`,
`ModelCtrBase` — is unchanged. **Update BOTH doc blocks in the same edit**: the
module block at `crates/cb-model/src/ctr_data.rs:756-758` **and** the
`# Errors` block on `build_tctr_value_table` at
`crates/cb-model/src/ctr_data.rs:801-806`. **Invert**
`crates/cb-model/src/ctr_data_test.rs:197-212` rather than deleting it.

**Refactor constraints + required regression scope**
- **Constraint (the branch predicate — MANDATORY, no implementer discretion):** the
  mean encode branch **MUST key on `table.ctr_type.is_mean()`**, which covers
  **`FloatTargetMeanValue` as well as `BinarizedTargetMeanValue`** — **NOT** on
  `BinarizedTargetMeanValue` alone. Rationale, recorded so it cannot be
  re-litigated: the test this task **inverts**
  (`encode_ctr_model_parts_rejects_mean_table`,
  `crates/cb-model/src/ctr_data_test.rs:197-212`) builds a **`FloatTargetMeanValue`**
  table, while E18/E19's training fixture is **`BinarizedTargetMeanValue`** — so a
  `BinarizedTargetMeanValue`-only branch passes one and fails the other, and the
  encoder would disagree with E19's decoder about which tables are accepted.
  The E19 decode branch carries the identical constraint; the two must match.
- **Constraint (SPEC-CTRT-16, the whole point):** the non-mean encode path must
  produce **byte-identical** output. E00's frozen baseline is the gate and it was
  captured **before** any codec change, so it is not a self-comparison.
- **Mandatory mutation check on the gate (§3.1):** temporarily change the mean
  branch's stride from 8 to 12 bytes; **expected failure** is test fn 5
  (`left: 12*B, right: 8*B`) and test fn 4 (round-trip length mismatch) while
  `ctr_nonmean_byte_identity_test` stays **green** — proving the two paths are
  genuinely independent. Revert manually.
- Regression scope: `cargo test -p cb-model` in full + `cargo test -p cb-train`
  (E13's flipped test) + `cargo test --workspace` compared to the
  `.planning/plans/one-hot-categorical-training/baseline/` gate.

**Validation**
```bash
cargo test -p cb-model --test ctr_mean_cbm_oracle_test
cargo test -p cb-model --lib ctr_data_test          # the INVERTED round-trip test
cargo test -p cb-model --test ctr_nonmean_byte_identity_test \
  --test float_only_byte_identity_test --test cbm_oracle_test \
  --test json_oracle_test --test ctr_data_roundtrip_test
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
cargo test -p cb-model -p cb-train
cargo clippy -p cb-model --all-targets
```

**Completion evidence.** Round-trip byte identity; the 8-byte stride pin; the
`i32` overflow typed rejection; **E00's non-mean baseline still byte-identical**;
the recorded stride-mutation failure/revert proving the gate is falsifiable; E13's
flipped save test green; **`encode_ctr_model_parts_round_trips_a_mean_table` green
in `crates/cb-model/src/ctr_data_test.rs`** (the inverted former rejection test —
it must exist under the new name, not be deleted); and **both** doc blocks
(`ctr_data.rs:756-758` and the `# Errors` block at `:801-806`) updated.

---

## WAVE W5 — `counter_calc_method`

> **The honesty constraint.** `counter_calc_method` is **unobservable without an
> eval set** — measured `maxdiff = 0.000e+00` learn-only vs `4.010e-01` with an
> eval set `[VERIFIED: research §B, EXPERIMENT probe6.py/probe7.py]`. A learn-only
> test passes trivially and proves nothing: **it is forbidden.** Either the
> eval-set fixture lands (E23) or an explicit deferral is recorded — never a
> learn-only "pass".
>
> **The structural blocker discovered by this plan:** `EvalSet` carries only
> `feature_values` and `target` — **no categorical columns**
> `[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:1951-1959]`. E21 removes it.

---

### E21 — `EvalSet` carries categorical columns; `train_cat_with_eval_sets` exists

- **Specs:** enabling task for SPEC-CTRT-17
- **Blocked by:** E11. **Blocks:** E22.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/boosting.rs`.

**Goal / observable completion condition.** `EvalSet` gains
`cat_columns: &'a [Vec<String>]`; a public `train_cat_with_eval_sets` exists with
the same shape as `train_with_eval_sets` but returning `(Model, BakedCtrData)`; all
6 existing `EvalSet` literal sites compile unchanged in behavior.

**Files**
- Modify: `crates/cb-train/src/boosting.rs`
- Modify: `crates/cb-train/src/lib.rs` (export `train_cat_with_eval_sets`)
- Modify: `crates/cb-train/tests/eval_metrics_oracle_test.rs` (4 literal sites)
- Modify: `crates/cb-train/tests/overfit_oracle_test.rs` (1 literal site)

**Exact verified files/symbols to touch**
- `pub struct EvalSet<'a> { pub feature_values: &'a [Vec<f32>], pub target: &'a [f64] }`
  at `crates/cb-train/src/boosting.rs:1951-1959` — **no derives**
  `[VERIFIED: LOCAL sed -n '1944,1960p']`.
- **All 6 struct-literal sites, enumerated** `[VERIFIED: LOCAL grep -rn "EvalSet {"]`:
  `crates/cb-train/tests/eval_metrics_oracle_test.rs:120,124,228,232`;
  `crates/cb-train/tests/overfit_oracle_test.rs:213`;
  `crates/cb-train/src/boosting.rs:2093`. Each adds `cat_columns: &[]`.
- `EvalSet` is referenced in only **4 files**: `crates/cb-train/src/lib.rs` (the
  `pub use` at `:102`), `crates/cb-train/src/boosting.rs`, and the two test files
  `[VERIFIED: LOCAL grep -rln]`.
- `pub fn train_with_eval_sets<R: Runtime>(runtime, feature_values,
  feature_borders, target, weights, params, staged_out, eval_sets, history)
  -> CbResult<Model>` at `crates/cb-train/src/boosting.rs:2139-2149`, which calls
  `train_inner(..)` and **discards** the baked data
  (`let (model, _baked) = train_inner(…)` at `:2153`) `[VERIFIED: LOCAL]`.
- `fn train_inner(..., eval_sets: &[EvalSet], ...)` at
  `crates/cb-train/src/boosting.rs:2555-2567` — **already accepts eval sets**
  `[VERIFIED: LOCAL]`, so the wrapper is the only missing piece.
- `pub fn train_cat<R: Runtime>(runtime, feature_values, feature_borders,
  cat_columns, target, weights, params, staged_out) -> CbResult<(Model, BakedCtrData)>`
  at `crates/cb-train/src/boosting.rs:2236-2245` — the shape to mirror
  `[VERIFIED: LOCAL]`.

**CodeGraph evidence for ordering.** E11 must precede this task because the eval
set only matters for a Counter table, and the per-type bake is what makes a Counter
table exist. `train_inner`'s existing `eval_sets` parameter means this task adds a
field and a wrapper, not a control-flow change.

**Red**
- File: `crates/cb-train/tests/eval_metrics_oracle_test.rs` (extend) — or a new
  `crates/cb-train/tests/train_cat_eval_sets_test.rs`
- Test fn: `train_cat_with_eval_sets_accepts_categorical_eval_columns`
- Setup: a 30-row learn set with 1 cat column (cardinality 6) and a 20-row eval set
  with the SAME cat column shape; `simple_ctr: ECtrType::Counter`;
  `counter_calc_method: CounterCalcMethod::SkipTest`.
- Expected: `Ok((model, baked))`, `baked.tables` non-empty, and
  `eval_sets[0].cat_columns.len() == 1` readable from the constructed value.
  Plus a length-mismatch case: an eval set whose `cat_columns[0].len()` disagrees
  with its `target.len()` returns `CbError::LengthMismatch { .. }` (typed, no
  panic).
- **EXPECTED INITIAL FAILURE:**
  `error[E0560]: struct 'EvalSet' has no field named 'cat_columns'`, then
  `error[E0425]: cannot find function 'train_cat_with_eval_sets' in crate 'cb_train'`.
- Run: `cargo test -p cb-train --test train_cat_eval_sets_test`

**Green (minimal implementation intent).**
1. Add `pub cat_columns: &'a [Vec<String>]` to `EvalSet`; update the 6 literal
   sites with `cat_columns: &[]` (behavior-preserving: an empty slice is exactly
   today's semantics).
2. Add `pub fn train_cat_with_eval_sets<R: Runtime>(…, cat_columns, …, eval_sets,
   history) -> CbResult<(Model, BakedCtrData)>` delegating to `train_inner` and
   **returning** the baked data (unlike `train_with_eval_sets`, which discards it).
3. Add a length-consistency check for each eval set's cat columns against its
   target length, returning `CbError::LengthMismatch`.
4. Export from `crates/cb-train/src/lib.rs` next to `train_with_eval_sets` (`:102`).
5. **DEFINE THE BUCKET-SPACE RULE (mandatory; without it SPEC-CTRT-17 is
   under-specified, not merely untested).**
   **OWNERSHIP — read this first: step 5 is a SPECIFICATION, consumed by E22. It
   produces NO code in E21 beyond the doc comment on `materialize_ctr_feature`.**
   The implementation (widening the remap, the `extra_cat_columns` parameter and
   both call-site threads) is **E22's**, which owns
   `crates/cb-train/src/ctr/ctr_feature.rs` — see E22 Green step 0. E21 must not
   change `materialize_ctr_feature`'s signature or body.
   Under `CounterCalcMethod::Full`, the combined-projection hash **and the
   first-seen `HashMap<u64,u32>` remap** in `materialize_ctr_feature`
   (`crates/cb-train/src/ctr/ctr_feature.rs:183-196`, inside the function at
   `:124`) are built over the
   **CONCATENATION of the learn cat columns and EVERY eval set's cat columns**, in
   the order `learn ++ eval[0] ++ eval[1] ++ …` — mirroring upstream, which tallies
   over a `hashArr` built across learn **and every test set** and, under `Full`,
   sets `uniqValuesCounts.CounterCount = leafCount`
   (`online_ctr.cpp:716-729`, research §A.3).
   **Consequence, stated explicitly:** an eval-only categorical value that never
   appears in the learn set gets **its own bucket**, so `bucket_count` (`leafCount`)
   GROWS. Under `SkipTest` the remap is built over the learn columns **only** and
   `bucket_count` is unchanged — that is today's behavior and must stay
   byte-identical.
   **The learn-document OUTPUT column stays indexed by the LEARN slice.** The eval
   documents contribute to the tally and to the bucket space; they do NOT produce
   output rows. `column.bins.len()` and `column.ctr_value.len()` must equal the
   learn document count under BOTH settings — assert it.

**Refactor constraints + required regression scope**
- Constraint: `train`, `train_with_eval_sets` and `train_cat` signatures are
  **unchanged** — no caller outside `cb-train` moves.
- Constraint: with `cat_columns: &[]` on every eval set, behavior must be
  byte-identical — `eval_metrics_oracle_test` and `overfit_oracle_test` are the
  gate.
- Regression scope: `cargo test -p cb-train` in full + the 3 one-hot targets
  (`boosting.rs` is touched).

**Validation**
```bash
cargo test -p cb-train --test train_cat_eval_sets_test
cargo test -p cb-train --test eval_metrics_oracle_test --test overfit_oracle_test
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test \
  --test device_one_hot_parity_test
cargo test -p cb-train
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** All 6 literal sites updated; both eval-set oracles green
unchanged; the typed length-mismatch case green; 3 one-hot targets green; **the
`Full` bucket-space rule (Green step 5) written down in the code as a doc comment
on `materialize_ctr_feature`, naming `online_ctr.cpp:716-729` and stating that an
eval-only category widens `bucket_count` while the output column stays indexed by
the learn slice.**

---

### E22 — Thread `counter_calc_method` into the Counter total and the final bake

- **Specs:** SPEC-CTRT-17 (threading half)
- **Blocked by:** E21. **Blocks:** E23.
- **Parallelizable:** **NO** — owns `crates/cb-train/src/boosting.rs`,
  `crates/cb-train/src/ctr/{online,final_ctr,bake,ctr_feature}.rs`.
  (`ctr_feature.rs` was last owned by E09, nine tasks earlier, so no conflict.)

**Goal / observable completion condition.** `SkipTest` counts learn documents only;
`Full` counts learn **+ every eval set** in BOTH effect sites — the online
`CountOnlineCTRTotal` sample range and the final-CTR `totalSampleCount`. The flag
is read from `params.counter_calc_method`, which today has **zero reads** in
`boosting.rs`.

**Files**
- Modify: `crates/cb-train/src/ctr/online.rs` (`online_counter_column` gains the
  extra-sample bins)
- Modify: `crates/cb-train/src/ctr/final_ctr.rs` (make E11's
  `counter_calc_skip_test` parameter real)
- Modify: `crates/cb-train/src/ctr/bake.rs` (thread it through `bake_ctr_table`)
- Modify: `crates/cb-train/src/boosting.rs` (read `params.counter_calc_method`;
  thread the eval cat columns from BOTH `materialize_ctr_feature` call sites)
- **Modify: `crates/cb-train/src/ctr/ctr_feature.rs` — E22 OWNS THIS FILE.** It is
  where E21 Green step 5's `Full` bucket-space rule is actually implemented (E21
  only wrote the doc comment). E22 already owns `crates/cb-train/src/boosting.rs`,
  and E09 — which also owns `ctr_feature.rs` — precedes E22 by nine tasks, so
  serialization is unchanged and no new conflict is created. See Green step 0.
- Modify: `crates/cb-train/src/ctr/online_test.rs`,
  `crates/cb-train/src/ctr/final_ctr_test.rs`
- Modify: `crates/cb-train/tests/ctr_feature_materialize_test.rs` (existing target;
  it is where test fn 4 lives — `materialize_ctr_feature` is reachable there)
- Modify: `crates/cb-train/tests/ctr_split_scoring_test.rs` — **mechanical, forced
  by this task's further widening of both signatures.** Update the TWO
  `materialize_ctr_feature` call sites at `:384` and `:394` (E09 already widened
  them; E22 adds `extra_cat_columns`, passed as an **empty** slice — the `SkipTest`
  default, byte-identical behavior) and the THREE `bake_ctr_table` call sites at
  `:542`, `:576`, `:645` (E11 already widened them; E22 adds
  `counter_calc_skip_test = true` plus the eval cat columns, empty — again the
  `SkipTest` default, byte-identical behavior)
  `[VERIFIED: LOCAL crates/cb-train/tests/ctr_split_scoring_test.rs:384, :394, :542, :576, :645]`.
  **CHANGE NO ASSERTION.** This file is **one of the eleven SPEC-CTRT-18 oracle
  targets** (PLAN.md §3.2); **weakening or deleting any assertion in it is
  FORBIDDEN.**

**Exact verified files/symbols to touch**
- `pub enum CounterCalcMethod { #[default] SkipTest, Full }` at
  `crates/cb-train/src/ctr/mod.rs:129-136`
  `[VERIFIED: LOCAL, read verbatim]`; `BoostParams.counter_calc_method` at
  `crates/cb-train/src/boosting.rs:272`; `counter_calc_method_default()` returns
  `SkipTest` (`:480-482`) — **matches upstream** `[VERIFIED: research §D.2]`.
- **Zero reads today:** `grep 'params.counter_calc_method' crates/cb-train/src/boosting.rs`
  → nothing `[VERIFIED: LOCAL; research §0/§F]`.
- **Exactly two effect sites, both Counter-only** `[VERIFIED: research §B]`:
  1. online — the sample range for `CountOnlineCTRTotal`: learn-only (`SkipTest`)
     vs `hashArr.size()` = learn + all test sets (`Full`)
     (`online_ctr.cpp:716-729`);
  2. final bake — `totalSampleCount += Data.GetTestSampleCount()` **only when**
     `ctrType == Counter && counterCalcMethod == Full` (`online_ctr.cpp:956-960`).
- It does **not** affect `FeatureFreq`, `Borders`, `Buckets`, or BTMV
  `[VERIFIED: research §B]`.
- `final_ctr.rs:70-73`'s existing doc claim — "in this whole-learn-set build there
  are no test documents, so the flag does not change the counts" — is **CORRECT**
  and must be **narrowed**, not deleted: it is true only with no eval set
  `[VERIFIED: research §B]`.
- Measured ground truth to reproduce `[VERIFIED: research §B, EXPERIMENT probe7.py]`:
  ```
  Full     → counts [18, 19, 18, 20, 22, 21], CounterDenominator = 22  (Σ = 100 = 60 learn + 40 test)
  SkipTest → counts [ 8, 14,  8, 14, 13, 11], CounterDenominator = 14  (Σ =  60 = learn only)
  ```

**CodeGraph evidence.** `build_final_ctr` has **14 callers** in `ctr/bake.rs` and
`ctr/mod.rs` with covering tests `ctr_data_roundtrip_test.rs` and
`ctr/final_ctr_test.rs` `[VERIFIED: CODEGRAPH]`; E11 already widened its signature,
so this task changes semantics only — no second signature churn.

**Red**
- File: `crates/cb-train/src/ctr/online_test.rs`
- Test fn 1: `counter_column_full_includes_eval_bins_skip_test_does_not`
  Setup: learn `bins = [0,0,1]` and eval `bins = [1,1,1]`.
  Expected:
  `SkipTest` → totals `[2, 1]`, denominator `2`;
  `Full`     → totals `[2, 4]`, denominator `4`;
  plus `assert_ne!(skip_result, full_result)` — the anti-vacuity guard.
- File: `crates/cb-train/src/ctr/final_ctr_test.rs`
- Test fn 2: `feature_freq_total_sample_count_ignores_counter_calc_method`
  Expected: `FeatureFreq`'s `counter_denominator` is identical under both settings
  (the flag is Counter-only), while `Counter`'s differs — asserted in the same
  test so a blanket implementation fails.
- Test fn 3: `borders_and_btmv_tables_are_unchanged_by_counter_calc_method`
  Expected: byte-identical `FinalCtrTable` under both settings.
- File: **`crates/cb-train/tests/ctr_feature_materialize_test.rs`** (NOT
  `ctr/online_test.rs`) — test fn 4 is stated over categorical **STRINGS**, and
  `online_counter_column(bins, extra_bins, bucket_count)` takes **pre-remapped
  bins**: it has no access to strings and cannot produce a bucket count of its own,
  so the assertion below is inexpressible there. `materialize_ctr_feature` — which
  owns the remap — is reachable from this existing integration target. **The
  bins-level half of the rule (totals / MAX denominator over
  `online_counter_column`) stays in `ctr/online_test.rs` as test fn 1 above.**
- Test fn 4 (**the EVAL-ONLY UNSEEN CATEGORY case — mandatory; it is the one E22's
  original `learn [0,0,1]` / eval `[1,1,1]` fixture does NOT exercise**):
  `an_eval_only_unseen_category_widens_both_the_bucket_count_and_the_denominator`
  Setup: learn cat column `["a","a","b"]`, eval cat column `["c","c","c"]` passed as
  `extra_cat_columns` to `materialize_ctr_feature` — the value `"c"` appears in
  **NO** learn document.
  Expected under `CounterCalcMethod::Full`:
  ```rust
  assert_eq!(bucket_count_full, 3,
      "an eval-only category must get its OWN bucket under Full \
       (uniqValuesCounts.CounterCount = leafCount, online_ctr.cpp:716-729)");
  assert_eq!(bucket_count_skiptest, 2,
      "under SkipTest the bucket space is learn-only and MUST be unchanged");
  assert_eq!(denominator_full, 3, "the MAX denominator sees the widened space");
  assert_eq!(denominator_skiptest, 2);
  assert_eq!(column_full.bins.len(), 3,
      "the OUTPUT column stays indexed by the LEARN slice — eval documents \
       contribute to the tally and the bucket space, never to output rows");
  ```
- **EXPECTED INITIAL FAILURE:** test fn 1 —
  ``assertion `left == right` failed: left: [2, 1], right: [2, 4]`` for the `Full`
  case, because `online_counter_column` (E06) deliberately has no extra-sample
  parameter. Test fn 4 — a compile error first
  (`error[E0061]`: `materialize_ctr_feature` has no `extra_cat_columns` parameter),
  then, once the parameter lands but before the remap is widened,
  ``assertion `left == right` failed: left: 2, right: 3`` on `bucket_count_full`.
  **Additionally `error[E0061]` at the five call sites in
  `crates/cb-train/tests/ctr_split_scoring_test.rs` — `materialize_ctr_feature` at
  `:384`, `:394` and `bake_ctr_table` at `:542`, `:576`, `:645` — as soon as the
  widened signatures land. That target does not build until the five mechanical
  argument additions in Files land; compile-forced, not behavioral.**
- Run: `cargo test -p cb-train --lib ctr::online_test -- counter_calc`,
  `cargo test -p cb-train --lib ctr::final_ctr_test -- counter_calc` and
  `cargo test -p cb-train --test ctr_feature_materialize_test -- eval_only_unseen`

**Green (minimal implementation intent).**
0. **IMPLEMENT E21 Green step 5's `Full` bucket-space rule — this is the step that
   makes the rule real; E21 only documented it.** In
   `crates/cb-train/src/ctr/ctr_feature.rs`:
   - widen `materialize_ctr_feature` (`:124` — **7 parameters on disk today**, 9
     after E09 adds `ctr_type` / `target_border_idx`) with an
     `extra_cat_columns: &[Vec<String>]` parameter (the concatenated eval-set cat
     columns; EMPTY under `SkipTest`);
   - fold the extra documents into the combined-key / first-seen remap — the local
     `HashMap<u64,u32>` loop at `crates/cb-train/src/ctr/ctr_feature.rs:183-196` —
     **AFTER** the learn documents, so learn bin numbering is byte-identical to
     today and an eval-only value can only ever receive a NEW, higher bin;
   - **keep the OUTPUT column indexed by the LEARN slice**: `column.bins.len()` and
     `column.ctr_value.len()` stay equal to the learn document count under BOTH
     settings. The extra documents contribute to the bucket space and the tally
     only, never to output rows;
   - thread the new parameter from **BOTH** production call sites —
     `crates/cb-train/src/boosting.rs:3238` (structure folds) and `:3274`
     (averaging fold) — passing an **empty** slice under
     `CounterCalcMethod::SkipTest`.
   This is a single remap shared by the bins and the totals; deriving a second,
   independent bucket space inside `online_counter_column` is **FORBIDDEN** (it
   would index the per-document bins into a different space than the
   totals/denominator — exactly the non-local divergence E23's ladder step 0 exists
   to catch).
1. `online_counter_column(bins, extra_bins, bucket_count)` — `extra_bins` is the
   concatenated eval-set bin slice, EMPTY under `SkipTest`. Both the per-bucket
   tally and the MAX denominator see `bins ∪ extra_bins`; the per-document output
   is still indexed by the LEARN bins only.
   **`bucket_count` GROWS under `Full`, and so does the `MAX` denominator — stated
   here so no implementer has to guess.** The bins fed to this function come from
   E21 Green step 5's rule, implemented by Green step 0 above: under `Full` the
   first-seen `HashMap<u64,u32>` remap
   (`crates/cb-train/src/ctr/ctr_feature.rs:183-196`) is built over
   `learn ++ every eval set`, so an eval-only categorical value receives its **own**
   bucket and `bucket_count == leafCount` over the widened space
   (`uniqValuesCounts.CounterCount = leafCount`, `online_ctr.cpp:716-729`).
   Since the denominator is `max` over the per-bucket totals, that new bucket can
   raise it. Under `SkipTest` the remap and therefore `bucket_count` are learn-only
   and byte-identical to today. Test fn 4 is the gate for both halves.
2. `build_final_ctr`'s Counter arm: when `!counter_calc_skip_test`, the totals
   already include eval documents (they were accumulated), so only the
   `counter_denominator` rule needs the widened input — mirror `online_ctr.cpp:956-960`.
3. `bake_ctr_table` gains `counter_calc_skip_test: bool` and the eval cat columns.
4. `boosting.rs` reads `params.counter_calc_method` once, converts to the bool
   (`matches!(m, CounterCalcMethod::SkipTest)`), and threads it to both the
   materialization and the bake.
5. Narrow the `final_ctr.rs:70-73` doc comment to say the flag is a no-op **only
   when there is no eval set**, citing the measured `maxdiff = 0.000e+00` vs
   `4.010e-01`.

**Refactor constraints + required regression scope**
- Constraint: with no eval set, output must be **byte-identical** under both
  settings — that is the measured ground truth and the D-04 no-op proof.
- Constraint: the flag must NOT touch Borders / Buckets / BTMV / FeatureFreq
  (test fns 2 and 3 enforce this).
- Regression scope: **all 11 CTR oracles + 3 one-hot targets + E12/E13/E15/E16/E17
  fixtures**.

**Validation**
```bash
cargo test -p cb-train --lib ctr::
cargo test -p cb-train --test ctr_feature_materialize_test   # test fn 4 lives here
cargo test -p cb-train --test ctr_counter_simple_oracle_test \
  --test ctr_btmv_simple_oracle_test --test ctr_buckets_simple_oracle_test \
  --test ctr_borders_multiprior_oracle_test --test ctr_mixed_simple_vs_combo_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test --test ordered_ctr_oracle_test \
  --test tensor_ctr_oracle_test --test tensor_ctr_e2e_oracle_test \
  --test s_order_ctr_bins_oracle_test --test ctr_split_scoring_test \
  --test ctr_feature_materialize_test --test multi_permutation_e2e_oracle_test \
  --test multi_permutation_fold_oracle_test
cargo test -p cb-model --test ctr_data_roundtrip_test --test fstr_ctr_oracle_test
cargo test -p cb-train --test one_hot_oracle_test --test one_hot_draw_accounting_test \
  --test device_one_hot_parity_test
cargo test -p cb-train -p cb-model
cargo clippy -p cb-train --all-targets
```

**Completion evidence.** `grep 'params.counter_calc_method' crates/cb-train/src/boosting.rs`
returns **non-zero** matches (zero today); the **four** tests green — the three unit
tests in `ctr/online_test.rs` / `ctr/final_ctr_test.rs` (the `assert_ne!`
anti-vacuity guard and the type-isolation assertions) plus **test fn 4 in
`crates/cb-train/tests/ctr_feature_materialize_test.rs`, pinning the
eval-only-unseen-category widening of both `bucket_count` and the `MAX`
denominator while the output column stays learn-length**;
`materialize_ctr_feature` carries the `extra_cat_columns` parameter and both
`boosting.rs` call sites (`:3238`, `:3274`) thread it; every existing oracle green.
**`git diff crates/cb-train/tests/ctr_split_scoring_test.rs` shows exactly the five
widened call sites (`materialize_ctr_feature` `:384`, `:394`; `bake_ctr_table`
`:542`, `:576`, `:645`) on top of the earlier tasks' edits and NOTHING else** — no
assertion added, removed, weakened or reworded.

---

### E23 — `counter_full_eval` fixture + the eval-set ≤1e-5 gate (or a RECORDED deferral)

- **Specs:** SPEC-CTRT-17 (parity half); acceptance **A6**
- **Blocked by:** E22. **Blocks:** F00 (start of Part 2).
- **Parallelizable:** **YES** with E17/E18 — owns a new fixture directory and a new
  test target.

**Goal / observable completion condition.** EITHER an upstream fixture trained
**with an eval set** at `counter_calc_method="Full"` passes at ≤1e-5 **and** is
proven to differ from the `SkipTest` run; **OR** an explicit, written deferral is
recorded in `SPEC.md` §7 A6 and in `crates/cb-train/src/ctr/final_ctr.rs`'s doc
comment. **A learn-only test is FORBIDDEN** — it passes trivially
(`maxdiff = 0.000e+00`) and proves nothing.

**Files (success path)**
- Create: `crates/cb-oracle/fixtures/ctr_counter_full_eval/gen_fixtures.py`
- Create + COMMIT: `.../{X_cat.npy,y.npy,X_cat_eval.npy,y_eval.npy,model_full.json,model_skiptest.json,predictions_full.npy,predictions_skiptest.npy,config.json}`
- Create: `crates/cb-train/tests/ctr_counter_full_eval_oracle_test.rs`

**Files (deferral path)**
- Modify: `.planning/plans/ctr-type-engine-and-facade-routing/SPEC.md` (§7 row A6)
- Modify: `crates/cb-train/src/ctr/final_ctr.rs` (doc comment)
- Create: `crates/cb-train/tests/ctr_counter_full_eval_deferral_test.rs` — a test
  that asserts the threading exists (`params.counter_calc_method` changes the baked
  `counter_denominator` when an eval set is supplied) **without** claiming upstream
  parity.

**Exact verified files/symbols to touch**
- `train_cat_with_eval_sets` (E21) and `EvalSet.cat_columns` (E21).
- `cb_train::CounterCalcMethod::{SkipTest, Full}`
  `[VERIFIED: LOCAL crates/cb-train/src/ctr/mod.rs:129-136]`.
- No CTR fixture in the corpus currently carries an eval set
  `[VERIFIED: research §B]` — this is genuinely new fixture shape.

**Fixture configuration.** The §3 isolating set, categorical-only, 60 learn rows +
40 eval rows (matching the measured probe scale), `simple_ctr = ["Counter:Prior=0.5"]`,
`combinations_ctr = []`, `max_ctr_complexity = 1`, generated **twice** — once with
`counter_calc_method="Full"` and once with `"SkipTest"` — plus `eval_set=(X_eval, y_eval)`.

**MANDATORY anti-false-pass guard (generator) — the discriminator itself:**
```python
maxdiff = float(np.max(np.abs(pred_full - pred_skiptest)))
assert maxdiff > 1e-3, (
    f"counter_calc_method is UNOBSERVABLE on this fixture (maxdiff={maxdiff:.3e}). "
    "A fixture where Full and SkipTest agree cannot test SPEC-CTRT-17. "
    "Research measured 4.010e-01 with a 40-row eval set — widen the eval set or "
    "strengthen the categorical signal. DO NOT weaken this assertion.")
assert any(c["ctr_type"] == "Counter" for c in ctrs_full)
```
Plus the §3 corpus-cleanliness guard.

**Red**
- File: `crates/cb-train/tests/ctr_counter_full_eval_oracle_test.rs`
- Test fn 1: `counter_full_with_eval_set_matches_upstream_within_1e_minus_5`
  Params: `counter_calc_method: CounterCalcMethod::Full`, trained through
  `train_cat_with_eval_sets` with the eval set's cat columns supplied.
  Expected: `max |ours − predictions_full.npy| <= 1e-5`, reported.
- Test fn 2: `counter_skiptest_with_eval_set_matches_upstream_within_1e_minus_5`
  Same with `SkipTest` against `predictions_skiptest.npy`.
- Test fn 3 (**the discriminator**, mirroring the generator guard):
  `full_and_skiptest_predictions_actually_differ`
  Expected: `assert!(maxdiff_ours > 1e-3)` with a message quoting the research
  measurement (`0.000e+00` learn-only vs `4.010e-01` with eval). **Without this,
  two identical wrong answers pass both parity gates.**
- Test fn 4: `baked_counter_denominator_is_larger_under_full`
  Expected: `denominator_full > denominator_skiptest` — the structural twin of the
  measured `22` vs `14`.
- **EXPECTED INITIAL FAILURE:** `No such file or directory` on
  `ctr_counter_full_eval/X_cat.npy`; after generation and before E22, test fn 3
  fails with `maxdiff_ours = 0.000e+00` — the exact trivial-pass failure mode this
  spec exists to prevent.
- Run: `cargo test -p cb-train --test ctr_counter_full_eval_oracle_test`

**Green (minimal implementation intent).** No production change (delivered by
E21/E22). If the ≤1e-5 gate fails, run the ladder below.

**Localization ladder (STOP AND REPORT at the first hit)**
0. **EVAL-ONLY UNSEEN CATEGORICAL VALUE ⇒ BUCKET-SPACE DIVERGENCE. Check this
   FIRST.** Compute the set difference between the eval set's categorical values and
   the learn set's. If it is non-empty, verify against E21 Green step 5's rule: under
   `Full` each eval-only value must have received its **own** bucket, so
   `bucket_count` and the `MAX` denominator are **larger** than the learn-only space
   (`uniqValuesCounts.CounterCount = leafCount`, `online_ctr.cpp:716-729`). A repo
   that maps such a value to **no** bucket, or to a spurious existing one, produces a
   non-local numeric divergence with no other symptom. Compare
   `len(model_full.json → ctr_data[…].hashes)` against the repo's `bucket_count`.
   **STOP AND REPORT** on a mismatch — the defect is in the remap E22 Green step 0
   widened inside `materialize_ctr_feature`
   (`crates/cb-train/src/ctr/ctr_feature.rs:183-196`), not in the parity arithmetic.
1. Compare the baked Counter `int_counts` + `counter_denominator` against
   `model_full.json → ctr_data`. A count mismatch ⇒ the eval documents are not
   entering the tally.
2. Compare `SkipTest` alone. If `SkipTest` passes and `Full` fails, the defect is
   isolated to the eval-set sample range — report with both denominators.
3. If **both** fail identically, the defect predates W5 and is in E06/E11 —
   re-run `ctr_counter_simple` (E12) to confirm; **STOP AND REPORT**.

**Deferral path (only if the fixture cannot be made to discriminate).** Record, in
all three places, verbatim: *"SPEC-CTRT-17 is THREADED but NOT parity-verified.
It is unobservable without an eval set (measured `maxdiff = 0.000e+00` learn-only
vs `4.010e-01` with a 40-row eval set, research §B). No eval-set CTR fixture could
be produced that discriminates the two settings, so no learn-only test is claimed
— a learn-only test would pass trivially and prove nothing."* The deferral test
(structural, non-parity) is still MANDATORY.

**Refactor constraints + required regression scope**
- Constraint: **never** ship a learn-only `counter_calc_method` test.
- Regression scope: the 5 new CTR fixtures + all 11 existing CTR oracles.

**Validation**
```bash
.venv/bin/python crates/cb-oracle/fixtures/ctr_counter_full_eval/gen_fixtures.py
git status --porcelain crates/cb-oracle/fixtures   # ONLY ctr_counter_full_eval/*
cargo test -p cb-train --test ctr_counter_full_eval_oracle_test
cargo test -p cb-train --test ctr_counter_simple_oracle_test
cargo test -p cb-train -p cb-model
# WAVE-CLOSING GATE for all of Part 1:
cargo test --workspace --no-fail-fast 2>&1 | tail -40
#   compare against .planning/plans/one-hot-categorical-training/baseline/
#   RULE: no target that passes there may fail here.
#   `exact_quantile_weighted_matches_cpu` is FLAKY (~2/5) — not a regression.
```

**Completion evidence.** Either (a) nine committed artifacts + four tests green
with the recorded `maxdiff_ours` proving discrimination, or (b) the three recorded
deferral texts + the structural deferral test green. Plus the Part-1-closing
`cargo test --workspace --no-fail-fast` transcript compared line-by-line against
the accepted baseline.

---

> **Part 1 ends here.** Part 2 (tasks F00–F23), the SPEC-ID → task coverage tables,
> the risk register and the unresolved-blocker list are in `./PLAN-PART2.md`.
