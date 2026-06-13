# Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice) - Pattern Map

**Mapped:** 2026-06-14
**Files analyzed:** 13 new/modified files
**Analogs found:** 13 / 13

> Mechanism note: this map follows the **DECISION REVISION (2026-06-14)** in 05-CONTEXT.md — the per-object oracle is **transcribe-then-self-oracle** (a dependency-free `ordered_oracle.cpp`, zero catboost includes), NOT TU linking. The closest analog for that harness is therefore `cityhash_oracle.cpp` (which is itself a transcription, not a link), confirmed by 05-RESEARCH.md ESCALATION.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cb-train/src/permutation.rs` (new) | service | transform (RNG → index perm) | `crates/cb-core/src/rng.rs` (consumer) + `crates/cb-train/src/bootstrap.rs` (RNG-draw orchestration) | role-match |
| `crates/cb-train/src/fold.rs` (new) | service | batch (body/tail prefix state) | `crates/cb-train/src/bootstrap.rs` | role-match |
| `crates/cb-train/src/ctr/online.rs` (new) | service | event-driven (read-before-increment per doc) | `crates/cb-train/src/boosting.rs` (`accumulate_leaf_weights`, sequential host accumulation) | role-match |
| `crates/cb-train/src/ctr/calc_ctr.rs` (new) | utility | transform (triple → ui8 bin) | `crates/cb-model/src/apply.rs` (`binarize_feature`) | role-match |
| `crates/cb-train/src/ctr/final_ctr.rs` (new) | service | batch (whole-set table build) | `crates/cb-train/src/boosting.rs` (`accumulate_leaf_weights`) | role-match |
| `crates/cb-train/src/projection.rs` (new) | utility | transform (combination enum + hash fold) | `crates/cb-data/src/cat_hash.rs` (`calc_cat_feature_hash`, `PerfectHash`) | role-match |
| `crates/cb-train/src/candidates.rs` (new) | service | transform (one-hot vs CTR routing) | `crates/cb-train/src/tree.rs` (`select_best_candidate`, candidate enumeration) | role-match |
| `crates/cb-train/src/boosting.rs` (modify) | service | request-response (train driver) | itself (extend `train`/iteration loop) | exact (self-extend) |
| `crates/cb-train/src/tree.rs` (modify) | service | transform (categorical splits) | itself (extend `Split`/candidate gen) | exact (self-extend) |
| `crates/cb-model/src/ctr_data.rs` (new) | model | CRUD (serde over FlatBuffers) | `crates/cb-model/src/cbm.rs` (FlatBuffers save/load) | role-match |
| `crates/cb-model/src/apply.rs` (modify) | service | request-response (inference CTR Calc) | itself (extend `predict_raw_one`) | exact (self-extend) |
| `crates/cb-oracle/src/compare.rs` (modify) | utility | request-response (stage gate) | itself (add `Stage` variants) | exact (self-extend) |
| `crates/cb-oracle/src/model_json.rs` (modify) | model | CRUD (parse `ctr_data`) | itself (extend serde structs) | exact (self-extend) |
| `crates/cb-oracle/generator/ordered_oracle.cpp` (new) | test/harness | file-I/O (config+`.npy` in → per-object `.npy` out) | `crates/cb-oracle/generator/cityhash_oracle.cpp` | role-match (transcription precedent) |

---

## Pattern Assignments

### `crates/cb-train/src/permutation.rs` (service, transform) — ORD-01, D-03 linchpin

**Analog:** `crates/cb-core/src/rng.rs` (the RNG primitive — already bit-exact) + `crates/cb-train/src/bootstrap.rs` (how the trainer draws on the persistent RNG).

**RNG API to consume** (`crates/cb-core/src/rng.rs:141-228`) — do NOT re-port; Fisher-Yates draws `uniform(bound)`:
```rust
pub struct TFastRng64 { /* ... */ }
impl TFastRng64 {
    pub fn from_seed(seed: u64) -> Self { /* ... */ }
    pub fn gen_rand(&mut self) -> u64 { /* ... */ }
    pub fn uniform(&mut self, bound: u64) -> u64 { /* ... */ }     // bit-exact GenUniform
    pub fn try_uniform(&mut self, bound: u64) -> CbResult<u64> { /* fallible variant */ }
}
```

**RNG-draw-phase discipline to mirror** (`crates/cb-train/src/boosting.rs:43-54`) — the existing code documents exact per-iteration draw counts against the upstream RNG phase. The permutation generator must document its Fisher-Yates draw order the same way (RESEARCH `shuffle.h:24-32`, block size 1 for N<1000 — RESEARCH Open Q3):
```rust
/// Per-iteration PRE-bootstrap draws on the persistent RNG (train.cpp:208,211):
/// the fold pick (`Rand.GenRand() % foldCount`) and the derivative-seed draw ...
const PRE_TREE_DRAWS: usize = 2;
```

**Output type:** `Vec<i32>` permutation indices per fold → `Stage::Permutation` (exact integer compare). Lock this stage FIRST (D-03) before any value stage runs.

---

### `crates/cb-train/src/ctr/online.rs` (service, event-driven) — ORD-03, the no-leakage core

**Analog:** `crates/cb-train/src/boosting.rs::accumulate_leaf_weights` (`:283-293`) — sequential host bucket accumulation with the sanctioned ordered reduction.

**Bucket-accumulation pattern** (`crates/cb-train/src/boosting.rs:283-293`):
```rust
fn accumulate_leaf_weights(leaf_of: &[usize], weights: &[f64], n_leaves: usize) -> Vec<f64> {
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if let (Some(bucket), Some(&w)) = (members.get_mut(leaf), weights.get(i)) {
            bucket.push(w);                       // checked .get only — indexing_slicing deny
        }
    }
    members.iter().map(|bucket| sum_f64(bucket)).collect()   // ordered sum (D-08)
}
```

**Critical adaptation — read-before-increment** (RESEARCH Pattern 1, `online_ctr.cpp:168-184/300-307`): for each doc in permutation order READ the bucket's accumulated `(goodCount, totalCount)`, compute the CTR, THEN increment. The "read before increment" IS the no-leakage property — the exact bug the per-object oracle exists to catch.

**Parity discipline (RESEARCH Anti-Pattern):** the **integer** class counts (`TCtrHistory::N[2]` are `int`) are exact-integer accumulation and are EXEMPT from `sum_f64` (compare exactly at `Stage::OnlineCtr` num/denom). Only **float** sums (`TCtrMeanHistory::Sum`, priors, leaf ders) route through `cb_core::sum_f64` (D-08).

**Categorical hashing source (D Carried-Forward + RESEARCH Anti-Pattern):** consume `cb_data::calc_cat_feature_hash` (see `crates/cb-data/src/cat_hash.rs:362`), NEVER a model's `ctr_data` hash_map.

---

### `crates/cb-train/src/ctr/calc_ctr.rs` (utility, transform) — ORD-03

**Analog:** `crates/cb-model/src/apply.rs::binarize_feature` (`:44-47`) — the count-of-borders quantization shape.

**Quantization pattern** (`crates/cb-model/src/apply.rs:45-47`):
```rust
pub fn binarize_feature(raw: f64, borders: &[f64]) -> usize {
    borders.iter().filter(|&&b| raw > b).count()     // STRICT > — load-bearing
}
```

**Critical adaptation — two distinct CTR normalizations** (RESEARCH Pattern 2 + Pitfall 1):
- **Online (training)** `CalcCTR` (`online_ctr.h:128-131`): denom is hard `+1` → `ctr = (countInClass + prior) / (totalCount + 1)`; then `(ctr + shift) / norm * borderCount`.
- **Inference (model-side)** `TModelCtr::Calc` (`online_ctr.h:289-292`): `(countInClass + PriorNum) / (totalCount + PriorDenom)`; then `(ctr + Shift) * Scale`.
These coincide only when `PriorDenom == 1` (default priors). Keep them as **separate functions** — do not share the code path.

---

### `crates/cb-train/src/projection.rs` (utility, transform) — ORD-05

**Analog:** `crates/cb-data/src/cat_hash.rs` — `calc_cat_feature_hash` (`:362`) + `city_hash_64` (`:272`) + `PerfectHash` (`:389`). The projection `CalcHash`/`MultiHash` fold (RESEARCH, `ctr_provider.h:65-78`) builds on these primitives. `GetFullProjectionLength` / `max_ctr_complexity` gate per RESEARCH (`projection.h:138-144`).

**Reuse, never re-port** (`crates/cb-data/src/cat_hash.rs:362,389`):
```rust
pub fn calc_cat_feature_hash(s: &str) -> u32 { /* CityHash64 & 0xffffffff */ }
pub struct PerfectHash { /* first-seen perfect-hash bins */ }
```

---

### `crates/cb-train/src/candidates.rs` (service, transform) — ORD-04 one-hot path selection

**Analog:** `crates/cb-train/src/tree.rs` — `select_best_candidate` + the `Candidate` enumeration (`:62-71`, plus the strict first-wins tie-break documented at `:1-46`).

**Candidate struct + strict-first-wins discipline to extend** (`crates/cb-train/src/tree.rs:63-71`):
```rust
pub struct Candidate {
    pub feature: usize,
    pub border: f64,
    pub score: f64,
}
// tree.rs header: SelectBestCandidate uses strict `if (gain > bestGain)` over a
// FIXED iteration order (feature asc, border asc); FIRST max wins. Do NOT use >=.
```

**Critical adaptation — one-hot vs CTR threshold (RESEARCH Pitfall 3):** one-hot used when `1 < count <= one_hot_max_size` (inclusive); CTR used when `count > one_hot_max_size` (exclusive). Cardinality is learn-set-only unique count. Permutation is needed in Plain mode too whenever any cat feature `> one_hot_max_size` (RESEARCH Pitfall 2) — the D-04 one-hot-only first slice has genuinely NO permutation.

---

### `crates/cb-train/src/boosting.rs` (modify, service) — ORD-02 ordered boosting

**Analog:** itself — extend the train driver. Existing structures to build on:

**Train entry + iteration loop** (`crates/cb-train/src/boosting.rs:338,565`):
```rust
pub fn train<R: Runtime>( /* ... */ ) -> CbResult<...> { /* ... */ }
// ...
for iter in 0..params.iterations { /* grow one tree, update approx */ }
```
**`BoostParams` extension point** (`:57-119`): add `boosting_type` (Plain/Ordered — pin explicitly, NOT auto; RESEARCH Pitfall 6), `permutation_count`, `fold_len_multiplier`, `one_hot_max_size`, `max_ctr_complexity`, CTR config. Mirror the per-field doc-comment-with-upstream-citation style already in the struct.

**Critical adaptation — body/tail ordered prefix** (RESEARCH Pattern 3, `fold.cpp:35-41,156-198` + `approx_calcer.cpp:566-600`): leaf deltas estimated on the body, applied to the tail, so a tail doc's approximant never depends on itself. Output `ordered_approx_iter{t}` per iteration → `Stage::OrderedApprox` ≤1e-5. Every approx sum via `cb_core::sum_f64` (D-08); the `accumulate_leaf_weights`/`compute_leaf_deltas` patterns (`:204-293`) are the leaf-stat reduction template.

---

### `crates/cb-model/src/ctr_data.rs` (new, model) — `ctr_data` serde over committed bindings

**Analog:** `crates/cb-model/src/cbm.rs` — FlatBuffers build (`save_cbm` `:119`) + VERIFYING decode (`decode_cbm` `:240`).

**Committed bindings to use (do NOT re-derive `.fbs`)** — `crates/cb-model/src/generated/ctr_data_generated.rs`, `ECtrType` (`:199-231`):
```rust
pub struct ECtrType(pub i8);
impl ECtrType {
    pub const Borders: Self = Self(0);
    pub const Buckets: Self = Self(1);
    pub const BinarizedTargetMeanValue: Self = Self(2);
    pub const FloatTargetMeanValue: Self = Self(3);
    pub const Counter: Self = Self(4);
    pub const FeatureFreq: Self = Self(5);
}
```

**VERIFYING decode + bounds-before-slice pattern to copy** (`crates/cb-model/src/cbm.rs:240-270`):
```rust
pub fn decode_cbm(buf: &[u8]) -> Result<Model, ModelError> {
    if buf.get(0..4) != Some(CBM1.as_slice()) { /* typed err, never index */ }
    let size_bytes: [u8; 4] = buf.get(4..8).and_then(|s| <[u8;4]>::try_from(s).ok())
        .ok_or_else(|| ModelError::Deserialize("truncated .cbm size field".to_owned()))?;
    let declared = u32::from_le_bytes(size_bytes) as usize;
    let core = buf.get(8..8usize.saturating_add(declared))   // BOUND before slice (V5)
        .ok_or_else(|| ModelError::Deserialize(/* ... */))?;
    let model_core = root_as_tmodel_core(core)               // VERIFYING accessor, not _unchecked
        .map_err(|e| ModelError::Deserialize(format!("corrupt FlatBuffers TModelCore: {e}")))?;
    // ...
}
```
**Security (RESEARCH V5):** bounds-check bucket index `< leafCount` before indexing the CTR blob; malformed `IndexHashRaw`/`CTRBlob`/`CounterDenominator`/`TargetClassesCount` → typed `ModelError`, never panic.

---

### `crates/cb-model/src/apply.rs` (modify, service) — model-side CTR Calc at inference

**Analog:** itself — extend `predict_raw_one` (`:62-81`).

**Existing apply accumulation to extend** (`crates/cb-model/src/apply.rs:62-81`):
```rust
fn predict_raw_one(model: &Model, features: &[f32]) -> f64 {
    let leaf_contributions: Vec<f64> = model.oblivious_trees.iter().map(|tree| {
        let passes: Vec<bool> = tree.splits.iter().map(|s| passes_split(s, features)).collect();
        let leaf = leaf_index(&passes);
        tree.leaf_values.get(leaf).copied().unwrap_or(0.0)     // checked .get
    }).collect();
    model.bias + sum_f64(&leaf_contributions)                  // ordered sum, bias once
}
```

**Critical adaptation (RESEARCH "Model-side inference apply"):** per-type `Calc(cic, tot) = (cic + PriorNum)/(tot + PriorDenom); return (ctr + Shift) * Scale`. Missing bucket → `emptyVal = Calc(0, denom)` (Counter) or `Calc(0,0)` (others) — replicate the not-found→empty path (V5, `static_ctr_provider.cpp:115-119`), never OOB index. Projection hashes via `cb_data::calc_cat_feature_hash` (the projection.rs helper), NEVER the model's stored hash_map.

---

### `crates/cb-oracle/src/compare.rs` (modify, utility) — add Stage variants

**Analog:** itself — add to the `Stage` enum (`:9-21`). The `compare_stage` gate (`:70-96`) is reused unchanged.

**Existing enum to extend** (`crates/cb-oracle/src/compare.rs:9-21`):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Borders, Splits, LeafValues, StagedApprox, Predictions,
    // ADD (RESEARCH D-02 schema): Permutation, OnlineCtr, OrderedApprox
}
```
`Stage::Permutation` is integer-exact (compare with a `==` path, not the 1e-5 `assert_abs_close`); `OnlineCtr`/`OrderedApprox` use the existing ≤1e-5 `compare_stage`. The non-finite-diff guard at `:44` (`!(diff <= tol)`) is the template — preserve it.

---

### `crates/cb-oracle/src/model_json.rs` (modify, model) — parse `ctr_data`

**Analog:** itself — add a `#[derive(Deserialize)]` struct mirroring the existing borders-only structs (`SplitJson` `:22-33`, `FloatFeatureJson` `:54-62`) and an accessor like `float_feature_borders` (`:127-133`).

**Serde-struct + accessor pattern to copy** (`crates/cb-oracle/src/model_json.rs:54-62, 127-133`):
```rust
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FloatFeatureJson {
    pub feature_index: i64,
    #[serde(default)] pub borders: Vec<f64>,    // #[serde(default)] keeps old fixtures parsing
}
// accessor:
pub fn float_feature_borders(&self) -> Vec<Vec<f64>> {
    self.features_info.float_features.iter().map(|f| f.borders.clone()).collect()
}
```
Add a `CtrDataJson` struct (`TCtrValueTable` blobs — counts per bucket; interpret per `static_ctr_provider.cpp:14-126`) with `#[serde(default)]` so Phase-3/4 fixtures without `ctr_data` keep parsing (RESEARCH A5). Stay `unwrap`/`expect`-free (T-03-00-01).

---

### `crates/cb-oracle/generator/ordered_oracle.cpp` (new, harness) — transcribe-then-self-oracle

**Analog:** `crates/cb-oracle/generator/cityhash_oracle.cpp` — the transcription precedent (ZERO catboost includes; the only viable mechanism per RESEARCH ESCALATION).

**Header-doc + zero-include + build-line convention to copy** (`crates/cb-oracle/generator/cityhash_oracle.cpp:1-29`):
```cpp
// <name>.cpp — authoritative <X> oracle.
//
// A standalone, dependency-free transcription of Yandex CatBoost's <algo>
// (catboost-master/.../<file>:<lines>). WHY THIS EXISTS: ... I/O: ...
// Build: g++ -O2 -std=c++17 <name>.cpp -o <name>
#include <cstdint>
#include <cstring>
#include <string>
#include <iostream>
// NO catboost headers.
```

**Transcribe verbatim (RESEARCH fallback step 1):** the `Shuffle` Fisher-Yates loop (`shuffle.h:24-32`) over a transcribed `TFastRng64`; online CTR read-before-increment + `CalcCTR` (`online_ctr.cpp:168-184/300-307` + `online_ctr.h:102-131`); body/tail prefix (`fold.cpp:35-41,156-198`); ordered approximant prefix (`approx_calcer.cpp:566-600`).

**Output `.npy` per RESEARCH D-02 schema** (read by `crates/cb-oracle/src/fixture.rs::load_f64_vec` `:21`): `permutation_fold{k}.npy [N] int32`; `ctr_good_count.npy`/`ctr_total_count.npy [N] int32`; `ctr_value.npy [N|N,n_prior] f64`; `ordered_approx_iter{t}.npy [N] f64`.

**Self-oracle anchors (RESEARCH fallback step 2):** permutation cross-checked vs the oracle-locked `cb_core::TFastRng64` Rust reproduction (D-03 ground truth); final CTR vs the trained model's `ctr_data` blob; ordered approx validated indirectly (final-prediction parity + monotone num/denom + identity-permutation degeneration — A2 residual risk, accepted).

---

## Shared Patterns

### Ordered float summation (D-08, parity-critical)
**Source:** `crates/cb-core/src/rng.rs` companion `cb_core::sum_f64` / `sum_f32_in_f64` (used in `crates/cb-train/src/boosting.rs:292`, `crates/cb-model/src/apply.rs:80`)
**Apply to:** ALL new CTR running float sums, priors, ordered approximants, leaf stats.
```rust
members.iter().map(|bucket| sum_f64(bucket)).collect()   // never iter().sum() / fold(0.0,…)
```
**Exemption (RESEARCH Anti-Pattern):** integer class counts (`TCtrHistory::N[2]`) are exact-int accumulation — NOT routed through `sum_f64`, compared exactly.

### Checked access / panic-free (INFRA-02 deny-lints, V5)
**Source:** `crates/cb-train/src/boosting.rs:283-293`, `crates/cb-model/src/cbm.rs:240-270`, `crates/cb-model/src/apply.rs:76`
**Apply to:** ALL new files. Use `.get`/`.get_mut`/`u32::try_from`; no `unwrap`/`expect`/`panic`/raw indexing in production. Tests opt out via the lib.rs `#![cfg_attr(test, allow(...))]` (`crates/cb-train/src/lib.rs:1`).

### Typed errors, no `anyhow` (CI grep)
**Source:** `crates/cb-oracle/src/error.rs:9-79` (`OracleError`), `crates/cb-model/src/error.rs` (`ModelError`)
**Apply to:** new `cb-train`/`cb-model`/`cb-oracle` code — `thiserror` enum variants; `anyhow` structurally banned.

### Source/test separation (CLAUDE.md / AGENTS.md, MANDATORY)
**Source:** `crates/cb-train/src/tree.rs:35-46` (sibling `#[path]` test modules), `crates/cb-train/src/lib.rs`
**Apply to:** every new source file — tests live in dedicated `*_test.rs` siblings, mounted via `#[cfg(test)] #[path = "..._test.rs"] mod ...;`. NO inline `#[cfg(test)]` test bodies in production files.

### Categorical hashing single source (D Carried-Forward, RESEARCH Anti-Pattern)
**Source:** `crates/cb-data/src/cat_hash.rs:362` (`calc_cat_feature_hash`)
**Apply to:** every CTR/projection consumer (online.rs, projection.rs, apply.rs) — NEVER a model's `ctr_data` hash_map.

### Frozen committed fixtures (D-09)
**Source:** `crates/cb-oracle/src/fixture.rs` (`load_f64_vec` `:21`, `FixtureConfig` `:28-36`), existing layout e.g. `crates/cb-oracle/fixtures/cat_hash/{*.npy,config.json}`
**Apply to:** new categorical fixtures (D-08) under `crates/cb-oracle/fixtures/` — `.npy` + `config.json` (`seed`, `catboost_version:"1.2.10"`, `thread_count:1`); generators run offline, never in CI.

---

## No Analog Found

None — every new/modified file maps to an existing analog. The genuinely novel algorithm content (read-before-increment ordering, body/tail prefix, projection enumeration) is transcribed from the vendored `catboost-master/` 1.2.10 source per 05-RESEARCH.md, but the Rust *structure* in each case follows an existing in-repo analog above.

---

## Metadata

**Analog search scope:** `crates/cb-train/src/`, `crates/cb-model/src/`, `crates/cb-oracle/src/`, `crates/cb-oracle/generator/`, `crates/cb-core/src/`, `crates/cb-data/src/`, `crates/cb-oracle/fixtures/`
**Files scanned:** ~25 (read in full: compare.rs, model_json.rs, cityhash_oracle.cpp, apply.rs, fixture.rs, model.rs head; targeted: boosting.rs, tree.rs, cbm.rs, ctr_data_generated.rs, rng.rs, cat_hash.rs, error.rs)
**Pattern extraction date:** 2026-06-14
