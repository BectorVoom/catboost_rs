//! Model-side `ctr_data` (de)serialize + per-type inference apply (ORD-03,
//! Security V5).
//!
//! # What this is
//!
//! The `ctr_data` section of a CatBoost model maps each CTR-base key (one per
//! `(projection, ctr_type, prior)` the trees reference) to a [`CtrValueTable`] —
//! the whole-learn-set bucket counts the model bakes ([`cb_train::FinalCtrTable`]
//! is the trainer-side producer) plus the `CounterDenominator`. At inference the
//! model hashes a document's categorical projection to a bucket, reads that
//! bucket's counts, and applies the per-type `Calc(cic, tot)` to recover the CTR
//! value the tree split tests against.
//!
//! # Wire forms
//!
//! - **`model.json`:** the upstream `{ "hash_map": [hash, n0, n1, …],
//!   "hash_stride": k, "counter_denominator": d }` flat heterogeneous array
//!   (`json_model_helpers.cpp:475-482`) — the SAME shape `cb-oracle`'s
//!   `CtrTableJson` parses. [`CtrValueTable::to_json`] / [`CtrValueTable::from_json`].
//! - **`.cbm`:** a self-describing little-endian binary blob (the model-parts
//!   region after the FlatBuffers core). [`encode_ctr_data`] / [`decode_ctr_data`]
//!   mirror the `cbm.rs` bounds-before-slice discipline (every declared length is
//!   bounded against the remaining bytes BEFORE slicing — Security V5,
//!   T-05-04-V5).
//!
//! # Per-type inference Calc (`static_ctr_provider.cpp:52-122`)
//!
//! - **Borders (binclf):** `Calc(history[1], history[0] + history[1])`.
//! - **Buckets:** `Calc(history[targetBorderIdx], Σ classes)`.
//! - **Mean (Binarized/Float):** `Calc(Sum, Count)`.
//! - **Counter / FeatureFreq:** `Calc(total[bucket], CounterDenominator)`.
//! - **missing bucket → emptyVal:** `Calc(0, denom)` (Counter) / `Calc(0, 0)`
//!   (others) — the not-found→empty path (`static_ctr_provider.cpp:115-119`),
//!   NEVER an OOB index (T-05-04-01).
//!
//! # Categorical projection hashing (Anti-Pattern)
//!
//! A document's bucket is found by hashing its categorical projection via
//! [`cb_data::calc_cat_feature_hash`] — NEVER the model's STORED `ctr_data`
//! hash_map (which holds CTR-projection hashes, a different thing). See
//! [`CtrValueTable::bucket_for_hash`].
//!
//! # Parity discipline
//!
//! Bounds-checked decode (every blob length validated before slice); malformed /
//! oversized blob or out-of-range bucket index → typed [`ModelError`], never a
//! panic (Security V5). Float sums via `cb_core::sum_f64`. No `unwrap`/`expect`/
//! raw-index in production; no `anyhow`.

use std::collections::BTreeMap;

use cb_core::sum_f64;
use serde::{Deserialize, Serialize};

use crate::ctr_data_generated::ncat_boost_fbs::{root_as_tctr_value_table, TCtrValueTable};
use crate::error::ModelError;

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file).
#[cfg(test)]
#[path = "ctr_data_test.rs"]
mod tests;

/// The six CTR types, mirroring the upstream `ECtrType` (`ctr_type.h`) i8
/// discriminants — the SAME values as `cb_train::ECtrType` and the generated
/// `ctr_data_generated::ECtrType`. Defined here so the `cb-model` serde owns its
/// own typed enum; [`Self::from_i8`] / [`Self::as_i8`] map losslessly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum ECtrType {
    /// Class-count borders CTR (`= 0`).
    Borders = 0,
    /// Class-count buckets CTR (`= 1`).
    Buckets = 1,
    /// Binarized-target mean (`= 2`).
    BinarizedTargetMeanValue = 2,
    /// Raw float-target mean (`= 3`).
    FloatTargetMeanValue = 3,
    /// Counter CTR (`= 4`): bucket total / MAX bucket total.
    Counter = 4,
    /// Feature-frequency CTR (`= 5`): bucket total / total sample count.
    FeatureFreq = 5,
}

impl ECtrType {
    /// The upstream i8 discriminant.
    #[must_use]
    pub fn as_i8(self) -> i8 {
        self as i8
    }

    /// Reconstruct from the upstream i8 discriminant; an unknown value is a typed
    /// error (no panic — Security V5, T-05-04-V5).
    ///
    /// # Errors
    /// [`ModelError::Deserialize`] if `value` is not a known CTR type.
    pub fn from_i8(value: i8) -> Result<Self, ModelError> {
        match value {
            0 => Ok(Self::Borders),
            1 => Ok(Self::Buckets),
            2 => Ok(Self::BinarizedTargetMeanValue),
            3 => Ok(Self::FloatTargetMeanValue),
            4 => Ok(Self::Counter),
            5 => Ok(Self::FeatureFreq),
            other => Err(ModelError::Deserialize(format!(
                "unknown ECtrType discriminant {other}"
            ))),
        }
    }

    /// Whether this type stores per-bucket FLOAT mean histories (`Sum`/`Count`)
    /// rather than integer counts.
    #[must_use]
    pub fn is_mean(self) -> bool {
        matches!(self, Self::BinarizedTargetMeanValue | Self::FloatTargetMeanValue)
    }

    /// Whether this type uses the shared `CounterDenominator` (Counter /
    /// FeatureFreq) rather than a per-bucket total.
    #[must_use]
    pub fn is_counter(self) -> bool {
        matches!(self, Self::Counter | Self::FeatureFreq)
    }
}

/// A CTR prior `(num, denom)` for the inference `Calc` (`PriorNum`/`PriorDenom`).
/// The in-scope fixtures pin `denom == 1` (RESEARCH A6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prior {
    /// The additive numerator (`PriorNum`).
    pub num: f64,
    /// The denominator weight (`PriorDenom`).
    pub denom: f64,
}

impl Prior {
    /// A unit-denominator prior `num / 1`.
    #[must_use]
    pub fn unit(num: f64) -> Self {
        Self { num, denom: 1.0 }
    }
}

/// One CTR table baked into the model: the per-bucket counts (keyed by the
/// projection hash), the per-bucket stride (1 hash + `stride-1` counts), and the
/// `CounterDenominator`.
///
/// The buckets are stored as `(hash, counts)` pairs in insertion order (the
/// `hash_map` flat-array order); a `hash -> index` map gives O(log n) lookup at
/// inference. For mean types the two "counts" per bucket are
/// `[Sum.to_bits() as i64-reinterpreted, Count]` — see [`Self::mean_at`].
#[derive(Debug, Clone, PartialEq)]
pub struct CtrValueTable {
    /// The CTR type this table serves.
    pub ctr_type: ECtrType,
    /// Number of target classes (`TargetClassesCount`, 2 for binclf). Used by
    /// the Borders/Buckets per-class indexing.
    pub target_classes_count: usize,
    /// Per-bucket projection hashes, in `hash_map` order.
    pub hashes: Vec<u64>,
    /// Per-bucket integer counts (class counts for Borders/Buckets, the single
    /// bucket total for Counter/FeatureFreq), one inner vector per bucket of
    /// length `hash_stride - 1`. Empty for mean types.
    pub int_counts: Vec<Vec<i64>>,
    /// Per-bucket mean `(Sum, Count)` for the mean types; empty otherwise.
    pub mean: Vec<(f32, i64)>,
    /// `CounterDenominator` (Counter = max bucket total, FeatureFreq = total
    /// sample count); `0` for the non-counter types.
    pub counter_denominator: i64,
}

impl CtrValueTable {
    /// Find the bucket index for a projection `hash` (the inference lookup),
    /// returning `None` for the not-found→empty path
    /// (`static_ctr_provider.cpp:115-119`) — NEVER an OOB index.
    #[must_use]
    pub fn bucket_for_hash(&self, hash: u64) -> Option<usize> {
        self.hashes.iter().position(|&h| h == hash)
    }

    /// The integer counts of bucket `bucket`, bounds-checked (`None` if out of
    /// range — never a panic, T-05-04-01).
    #[must_use]
    pub fn counts_at(&self, bucket: usize) -> Option<&[i64]> {
        self.int_counts.get(bucket).map(Vec::as_slice)
    }

    /// The `(Sum, Count)` of mean bucket `bucket`, bounds-checked.
    #[must_use]
    pub fn mean_at(&self, bucket: usize) -> Option<(f32, i64)> {
        self.mean.get(bucket).copied()
    }

    /// Apply the per-type inference `Calc` for a document whose projection hashes
    /// to `hash`, with the given `prior` and `(shift, scale)` normalization
    /// (`static_ctr_provider.cpp:52-122` + `online_ctr.h:289-292`).
    ///
    /// `target_border_idx` selects the Buckets per-class numerator (the binarized
    /// target border the CTR is computed against, default 0 for the head class).
    ///
    /// A missing bucket returns the empty value: `Calc(0, CounterDenominator)`
    /// for Counter/FeatureFreq, `Calc(0, 0)` otherwise — the no-OOB not-found
    /// path. Returns a finite `f64`.
    #[must_use]
    pub fn calc_for_hash(
        &self,
        hash: u64,
        prior: Prior,
        shift: f64,
        scale: f64,
        target_border_idx: usize,
    ) -> f64 {
        let bucket = self.bucket_for_hash(hash);
        let (cic, tot) = self.numerator_denominator(bucket, target_border_idx);
        calc_inference(cic, tot, prior, shift, scale)
    }

    /// Compute the `(countInClass, totalCount)` pair for `bucket` per the CTR
    /// type — the heart of the per-type apply. A `None` bucket yields the empty
    /// `(0, denom)` (Counter) / `(0, 0)` (others) pair.
    fn numerator_denominator(&self, bucket: Option<usize>, target_border_idx: usize) -> (f64, f64) {
        match self.ctr_type {
            ECtrType::Borders => {
                // Calc(history[1], history[0] + history[1]).
                match bucket.and_then(|b| self.counts_at(b)) {
                    Some(counts) => {
                        let n0 = counts.first().copied().unwrap_or(0) as f64;
                        let n1 = counts.get(1).copied().unwrap_or(0) as f64;
                        (n1, n0 + n1)
                    }
                    None => (0.0, 0.0),
                }
            }
            ECtrType::Buckets => {
                // Calc(history[targetBorderIdx], Σ classes).
                match bucket.and_then(|b| self.counts_at(b)) {
                    Some(counts) => {
                        let cic = counts.get(target_border_idx).copied().unwrap_or(0) as f64;
                        // Σ over classes via the order-locked sum (D-08).
                        let as_f64: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
                        (cic, sum_f64(&as_f64))
                    }
                    None => (0.0, 0.0),
                }
            }
            ECtrType::BinarizedTargetMeanValue | ECtrType::FloatTargetMeanValue => {
                // Calc(Sum, Count).
                match bucket.and_then(|b| self.mean_at(b)) {
                    Some((sum, count)) => (f64::from(sum), count as f64),
                    None => (0.0, 0.0),
                }
            }
            ECtrType::Counter | ECtrType::FeatureFreq => {
                // Calc(total[bucket], CounterDenominator); missing → Calc(0, denom).
                let total = bucket
                    .and_then(|b| self.counts_at(b))
                    .and_then(|c| c.first().copied())
                    .unwrap_or(0) as f64;
                (total, self.counter_denominator as f64)
            }
        }
    }
}

/// The inference `Calc(cic, tot)` (`online_ctr.h:289-292`):
/// `(cic + PriorNum) / (tot + PriorDenom)` then `(ctr + Shift) * Scale`. Guards a
/// zero denominator (no div-by-zero / NaN). The model-side form (denom is
/// `+ PriorDenom`, NOT the online `+1`).
#[must_use]
pub fn calc_inference(cic: f64, tot: f64, prior: Prior, shift: f64, scale: f64) -> f64 {
    let denom = tot + prior.denom;
    let ctr = if denom == 0.0 {
        0.0
    } else {
        (cic + prior.num) / denom
    };
    (ctr + shift) * scale
}

/// The whole `ctr_data` section: the CTR-base key → [`CtrValueTable`] map
/// (`json_model_helpers.cpp:524`). A `BTreeMap` gives a deterministic key order
/// for reproducible round-trips.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CtrData {
    /// The CTR tables, keyed by upstream CTR-base string.
    pub tables: BTreeMap<String, CtrValueTable>,
}

/// The canonical `(projection, ctr_type)` CTR-base key — the SAME form
/// `crate::apply::ctr_table_key` reconstructs at apply time:
/// `"ctr:type=<i8>:proj=<f0>,<f1>,…"` over the projection's SORTED cat-feature
/// members. Defined here (not only in `apply`) so the trainer-side bake lift uses
/// the IDENTICAL key the apply lookup expects (Plan 05-14).
#[must_use]
pub fn ctr_base_key(ctr_type: ECtrType, cat_features: &[usize]) -> String {
    let members: Vec<String> = cat_features.iter().map(usize::to_string).collect();
    format!("ctr:type={}:proj={}", ctr_type.as_i8(), members.join(","))
}

impl CtrData {
    /// Lift a trainer-side [`cb_train::BakedCtrData`] (ORD-05, Plan 05-14) into the
    /// canonical model-side `ctr_data`: each baked whole-set table becomes a
    /// [`CtrValueTable`] keyed by the canonical [`ctr_base_key`] the apply path
    /// reconstructs. The per-bucket combined projection hashes + class counts are
    /// carried verbatim; the inference `(Shift, Scale)` ride on the model's
    /// `CtrSplit` (via `from_trained`), not the table, so they are not duplicated
    /// here.
    #[must_use]
    pub fn from_baked(baked: &cb_train::BakedCtrData) -> Self {
        let mut tables = BTreeMap::new();
        for t in &baked.tables {
            let ctr_type = ECtrType::from_i8(t.ctr_type).unwrap_or(ECtrType::Borders);
            let key = ctr_base_key(ctr_type, t.projection.cat_features());
            tables.insert(
                key,
                CtrValueTable {
                    ctr_type,
                    target_classes_count: t.target_classes_count,
                    hashes: t.hashes.clone(),
                    int_counts: t.int_counts.clone(),
                    mean: Vec::new(),
                    counter_denominator: t.counter_denominator,
                },
            );
        }
        Self { tables }
    }
}

// ---------------------------------------------------------------------------
// model.json serde (the upstream hash_map / hash_stride / counter_denominator
// flat heterogeneous array — the SAME shape cb-oracle's CtrTableJson parses).
// ---------------------------------------------------------------------------

/// The serde shape of one CTR table in `model.json` (`json_model_helpers.cpp`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CtrTableJson {
    /// Flat heterogeneous `[hash_string, count, count, …]` repeated every
    /// `hash_stride` elements.
    #[serde(default)]
    pub hash_map: Vec<serde_json::Value>,
    /// Elements per bucket (1 hash + `hash_stride - 1` counts).
    #[serde(default)]
    pub hash_stride: i64,
    /// `CounterDenominator`; `0` for non-counter types.
    #[serde(default)]
    pub counter_denominator: i64,
    /// The CTR type discriminant (i8), so the JSON round-trips the type.
    #[serde(default)]
    pub ctr_type: i8,
    /// `TargetClassesCount` for the per-class indexing.
    #[serde(default)]
    pub target_classes_count: i64,
}

impl CtrValueTable {
    /// Serialize to the `model.json` flat-array shape. For the integer types each
    /// bucket emits `[hash, c0, c1, …]`; for the mean types each bucket emits
    /// `[hash, Sum_bits_as_f64, Count]` — the `Sum` is carried as its exact f64
    /// value so the round-trip is loss-free.
    #[must_use]
    pub fn to_json(&self) -> CtrTableJson {
        let mut hash_map: Vec<serde_json::Value> = Vec::new();
        let stride: i64;
        if self.ctr_type.is_mean() {
            stride = 3; // hash + Sum + Count.
            for (i, &(sum, count)) in self.mean.iter().enumerate() {
                let hash = self.hashes.get(i).copied().unwrap_or(0);
                hash_map.push(serde_json::json!(hash.to_string()));
                hash_map.push(serde_json::json!(f64::from(sum)));
                hash_map.push(serde_json::json!(count));
            }
        } else {
            // stride = 1 hash + per-bucket count width.
            let width = self.int_counts.first().map_or(0, Vec::len);
            stride = 1 + width as i64;
            for (i, counts) in self.int_counts.iter().enumerate() {
                let hash = self.hashes.get(i).copied().unwrap_or(0);
                hash_map.push(serde_json::json!(hash.to_string()));
                for &c in counts {
                    hash_map.push(serde_json::json!(c));
                }
            }
        }
        CtrTableJson {
            hash_map,
            hash_stride: stride,
            counter_denominator: self.counter_denominator,
            ctr_type: self.ctr_type.as_i8(),
            target_classes_count: self.target_classes_count as i64,
        }
    }

    /// Parse from the `model.json` flat-array shape. Ragged blobs, non-integer
    /// count slots, or a bad CTR type → typed [`ModelError`] (never a panic,
    /// T-05-04-V5).
    ///
    /// # Errors
    /// [`ModelError::Deserialize`] on a ragged stride, a non-numeric slot, an
    /// unparsable hash, or an unknown CTR type.
    pub fn from_json(json: &CtrTableJson) -> Result<Self, ModelError> {
        let ctr_type = ECtrType::from_i8(json.ctr_type)?;
        let stride = json.hash_stride;
        if stride <= 0 {
            return Ok(Self {
                ctr_type,
                target_classes_count: json.target_classes_count.max(0) as usize,
                hashes: Vec::new(),
                int_counts: Vec::new(),
                mean: Vec::new(),
                counter_denominator: json.counter_denominator,
            });
        }
        let stride = stride as usize;
        if !json.hash_map.len().is_multiple_of(stride) {
            return Err(ModelError::Deserialize(format!(
                "ctr_data hash_map length {} is not a multiple of hash_stride {stride}",
                json.hash_map.len()
            )));
        }

        let mut hashes = Vec::with_capacity(json.hash_map.len() / stride);
        let mut int_counts = Vec::new();
        let mut mean = Vec::new();

        for bucket in json.hash_map.chunks_exact(stride) {
            // bucket[0] is the hash string; bucket[1..] are the counts.
            let hash_str = bucket
                .first()
                .and_then(|v| v.as_str().map(str::to_owned))
                .ok_or_else(|| {
                    ModelError::Deserialize("ctr_data bucket missing hash string".to_owned())
                })?;
            let hash: u64 = hash_str.parse().map_err(|_| {
                ModelError::Deserialize(format!("ctr_data hash {hash_str:?} is not a u64"))
            })?;
            hashes.push(hash);

            if ctr_type.is_mean() {
                // [hash, Sum, Count].
                let sum = bucket
                    .get(1)
                    .and_then(serde_json::Value::as_f64)
                    .ok_or_else(|| {
                        ModelError::Deserialize("ctr_data mean Sum slot non-numeric".to_owned())
                    })?;
                let count = bucket
                    .get(2)
                    .and_then(serde_json::Value::as_i64)
                    .ok_or_else(|| {
                        ModelError::Deserialize("ctr_data mean Count slot non-integer".to_owned())
                    })?;
                mean.push((sum as f32, count));
            } else {
                let mut counts = Vec::with_capacity(stride - 1);
                for slot in bucket.iter().skip(1) {
                    let value = slot.as_i64().ok_or_else(|| {
                        ModelError::Deserialize("ctr_data count slot non-integer".to_owned())
                    })?;
                    counts.push(value);
                }
                int_counts.push(counts);
            }
        }

        Ok(Self {
            ctr_type,
            target_classes_count: json.target_classes_count.max(0) as usize,
            hashes,
            int_counts,
            mean,
            counter_denominator: json.counter_denominator,
        })
    }
}

// ---------------------------------------------------------------------------
// Upstream `.cbm` model-parts tail parser (CTR-03) — a DIFFERENT wire format
// from the self-describing LE blob below (`decode_ctr_data` / `encode_ctr_data`
// are cb-model's OWN round-trip format and are never used to read an upstream
// `.cbm`). The model-parts region appended after the `TModelCore` FlatBuffers
// core (`cbm.rs` framing doc) is:
//
//   u32 LE  part_count
//   repeat part_count times:
//     u32 LE  part_size ; part_size bytes = a `TCtrValueTable` FlatBuffers
//                          table (verified via `root_as_tctr_value_table`,
//                          never `_unchecked`)
//
// Each `TCtrValueTable` carries `ModelCtrBase` (ctr_type + combined-projection
// cat features), `IndexHashRaw` (a dense-hash byte blob: 12-byte slots of
// `(u64 hash LE, u32 blob_index LE)`, empty marker `hash ==
// 0xFFFF_FFFF_FFFF_FFFF`), `CTRBlob` (a raw `i32` LE array, width
// `TargetClassesCount` for Borders/Buckets or forced `1` for Counter/
// FeatureFreq, whose `TargetClassesCount` wire field is `0`), and
// `CounterDenominator`. Every declared length is bounds-checked against the
// remaining bytes BEFORE slicing (Security V5); mean-type CTRs
// (`BinarizedTargetMeanValue`/`FloatTargetMeanValue`) are rejected (v1,
// SPEC §2/MAJOR-2 — their `TCtrMeanHistory` byte layout is not empirically
// dissected and no fixture exercises it).
// ---------------------------------------------------------------------------

/// Byte width of one `IndexHashRaw` dense-hash slot: `u64 hash LE` (8 bytes)
/// followed by `u32 blob_index LE` (4 bytes).
const INDEX_HASH_SLOT_LEN: usize = 12;

/// The `IndexHashRaw` empty-slot sentinel hash value.
const EMPTY_HASH_MARKER: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Parse the appended upstream model-parts tail into the canonical [`CtrData`]
/// (CTR-03): `u32 part_count` then `part_count * (u32 part_size +
/// TCtrValueTable FlatBuffers table)`. Every length is bounds-checked against
/// the remaining bytes BEFORE slicing; the VERIFYING `root_as_tctr_value_table`
/// accessor is used (never `_unchecked`). Each table is keyed by
/// [`ctr_base_key`] over `(ctr_type, projection.cat_features())` — the SAME
/// form [`crate::apply`]'s `ctr_table_key` reconstructs at apply time.
///
/// # Errors
/// [`ModelError::Deserialize`] on: a truncated/missing part_count or part
/// header; a declared part size exceeding the remaining bytes; a corrupt
/// FlatBuffers `TCtrValueTable`; a mean-type CTR (`BinarizedTargetMeanValue`/
/// `FloatTargetMeanValue`, deferred — v1); an `IndexHashRaw` whose non-empty
/// `blob_index` set is not EXACTLY `0..bucket_count` (a gap, duplicate, or
/// out-of-range index); a `CTRBlob` byte length that does not cross-check
/// against `bucket_count * width`; or a duplicate `(ctr_type, projection)`
/// table key.
pub fn decode_ctr_model_parts(tail: &[u8]) -> Result<CtrData, ModelError> {
    let count_bytes: [u8; 4] = tail
        .get(0..4)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .ok_or_else(|| ModelError::Deserialize("truncated ctr model-parts count".to_owned()))?;
    let count = u32::from_le_bytes(count_bytes);

    let mut pos: usize = 4;
    let mut tables = BTreeMap::new();

    for _ in 0..count {
        let size_bytes: [u8; 4] = tail
            .get(pos..pos.saturating_add(4))
            .and_then(|s| <[u8; 4]>::try_from(s).ok())
            .ok_or_else(|| ModelError::Deserialize("truncated ctr model-part size".to_owned()))?;
        let size = u32::from_le_bytes(size_bytes) as usize;
        pos = pos.saturating_add(4);

        let part = tail.get(pos..pos.saturating_add(size)).ok_or_else(|| {
            ModelError::Deserialize(format!(
                "declared ctr model-part size {size} exceeds available {} bytes",
                tail.len().saturating_sub(pos)
            ))
        })?;
        pos = pos.saturating_add(size);

        let vt = root_as_tctr_value_table(part).map_err(|e| {
            ModelError::Deserialize(format!("corrupt FlatBuffers TCtrValueTable: {e}"))
        })?;

        let (key, table) = decode_one_ctr_value_table(&vt)?;
        if tables.insert(key.clone(), table).is_some() {
            return Err(ModelError::Deserialize(format!(
                "duplicate ctr_data table key {key:?} (ctr_type/projection collision, CTR-05)"
            )));
        }
    }

    Ok(CtrData { tables })
}

/// Decode one `TCtrValueTable` FlatBuffers part into its canonical
/// `(key, CtrValueTable)` pair (CTR-03 field mapping).
fn decode_one_ctr_value_table(
    vt: &TCtrValueTable<'_>,
) -> Result<(String, CtrValueTable), ModelError> {
    let base = vt.ModelCtrBase().ok_or_else(|| {
        ModelError::Deserialize("TCtrValueTable missing ModelCtrBase".to_owned())
    })?;
    // The generated `ECtrType` here is the transparent `ECtrType(pub i8)` tuple
    // from `ctr_data_generated` (a SEPARATE self-contained schema module from
    // `model_generated`'s copy) — convert via `.0` (MINOR-1).
    let ctr_type = ECtrType::from_i8(base.CtrType().0)?;
    if ctr_type.is_mean() {
        return Err(ModelError::Deserialize(
            "mean/target-mean CTR unsupported (v1, MAJOR-2)".to_owned(),
        ));
    }

    let combination = base.FeatureCombination().ok_or_else(|| {
        ModelError::Deserialize("TModelCtrBase missing FeatureCombination".to_owned())
    })?;
    let mut cat_features: Vec<usize> = Vec::new();
    if let Some(cats) = combination.CatFeatures() {
        for i in 0..cats.len() {
            let raw = cats.get(i);
            let idx = usize::try_from(raw).map_err(|_| {
                ModelError::Deserialize(format!("negative CatFeatures index {raw}"))
            })?;
            cat_features.push(idx);
        }
    }
    let projection = cb_train::TProjection::from_features(&cat_features);
    let key = ctr_base_key(ctr_type, projection.cat_features());

    let hashes = decode_index_hash_raw(vt)?;
    let bucket_count = hashes.len();
    let target_classes_count = usize::try_from(vt.TargetClassesCount())
        .map_err(|_| ModelError::Deserialize("negative TargetClassesCount".to_owned()))?;
    // Counter/FeatureFreq wire TargetClassesCount is 0 (a single bucket total,
    // not a per-class count); force width 1 so we never divide by zero
    // (MINOR-3).
    let width = if ctr_type.is_counter() {
        1
    } else {
        target_classes_count
    };
    if width == 0 {
        return Err(ModelError::Deserialize(format!(
            "ctr_type {ctr_type:?} has zero-width CTRBlob (TargetClassesCount {target_classes_count})"
        )));
    }
    let int_counts = decode_ctr_blob(vt, bucket_count, width)?;
    let counter_denominator = i64::from(vt.CounterDenominator());

    Ok((
        key,
        CtrValueTable {
            ctr_type,
            target_classes_count,
            hashes,
            int_counts,
            mean: Vec::new(),
            counter_denominator,
        },
    ))
}

/// Decode `IndexHashRaw` (a dense-hash byte blob: 12-byte `(u64 hash LE, u32
/// blob_index LE)` slots, empty marker `hash == 0xFFFF_FFFF_FFFF_FFFF`) into
/// the per-bucket `hashes` vector, sized to the AUTHORITATIVE `bucket_count`
/// (the number of non-empty slots, MINOR-3). The non-empty `blob_index` set
/// MUST be exactly `0..bucket_count` — a gap, duplicate, or out-of-range index
/// is a typed error, never a silent truncation or OOB write (MINOR-2).
fn decode_index_hash_raw(vt: &TCtrValueTable<'_>) -> Result<Vec<u64>, ModelError> {
    let raw: Vec<u8> = vt
        .IndexHashRaw()
        .map(|v| v.iter().collect())
        .unwrap_or_default();
    if !raw.len().is_multiple_of(INDEX_HASH_SLOT_LEN) {
        return Err(ModelError::Deserialize(format!(
            "IndexHashRaw length {} is not a multiple of {INDEX_HASH_SLOT_LEN}",
            raw.len()
        )));
    }
    let n_slots = raw.len() / INDEX_HASH_SLOT_LEN;

    let mut non_empty: Vec<(usize, u64)> = Vec::new();
    for s in 0..n_slots {
        let off = s.saturating_mul(INDEX_HASH_SLOT_LEN);
        let slot = raw
            .get(off..off.saturating_add(INDEX_HASH_SLOT_LEN))
            .ok_or_else(|| ModelError::Deserialize("IndexHashRaw slot out of range".to_owned()))?;
        let hash_bytes: [u8; 8] = slot
            .get(0..8)
            .and_then(|b| <[u8; 8]>::try_from(b).ok())
            .ok_or_else(|| ModelError::Deserialize("IndexHashRaw hash read failed".to_owned()))?;
        let idx_bytes: [u8; 4] = slot
            .get(8..12)
            .and_then(|b| <[u8; 4]>::try_from(b).ok())
            .ok_or_else(|| ModelError::Deserialize("IndexHashRaw idx read failed".to_owned()))?;
        let hash = u64::from_le_bytes(hash_bytes);
        if hash == EMPTY_HASH_MARKER {
            continue;
        }
        let idx = u32::from_le_bytes(idx_bytes) as usize;
        non_empty.push((idx, hash));
    }

    let bucket_count = non_empty.len();
    let mut hashes = vec![0u64; bucket_count];
    let mut filled = vec![false; bucket_count];
    for (idx, hash) in non_empty {
        let Some(filled_slot) = filled.get_mut(idx) else {
            return Err(ModelError::Deserialize(format!(
                "IndexHashRaw bucket index {idx} out of range (bucket_count {bucket_count})"
            )));
        };
        if *filled_slot {
            return Err(ModelError::Deserialize(format!(
                "IndexHashRaw duplicate bucket index {idx}"
            )));
        }
        *filled_slot = true;
        if let Some(slot) = hashes.get_mut(idx) {
            *slot = hash;
        }
    }
    if filled.iter().any(|&f| !f) {
        return Err(ModelError::Deserialize(
            "IndexHashRaw bucket indices are not exactly 0..bucket_count (gap detected)"
                .to_owned(),
        ));
    }
    Ok(hashes)
}

/// Decode `CTRBlob` (a raw little-endian `i32` array) into the per-bucket
/// `int_counts`, cross-checking the byte length against `bucket_count * width`
/// (MINOR-3) before slicing.
fn decode_ctr_blob(
    vt: &TCtrValueTable<'_>,
    bucket_count: usize,
    width: usize,
) -> Result<Vec<Vec<i64>>, ModelError> {
    let blob: Vec<u8> = vt.CTRBlob().map(|v| v.iter().collect()).unwrap_or_default();
    if !blob.len().is_multiple_of(4) {
        return Err(ModelError::Deserialize(format!(
            "CTRBlob length {} is not a multiple of 4",
            blob.len()
        )));
    }
    let n_i32 = blob.len() / 4;
    let expected = bucket_count.saturating_mul(width);
    if n_i32 != expected {
        return Err(ModelError::Deserialize(format!(
            "CTRBlob element count {n_i32} does not match bucket_count*width ({bucket_count}*{width}={expected})"
        )));
    }
    let mut int_counts: Vec<Vec<i64>> = Vec::with_capacity(bucket_count);
    for b in 0..bucket_count {
        let mut counts = Vec::with_capacity(width);
        for j in 0..width {
            let i = b.saturating_mul(width).saturating_add(j);
            let off = i.saturating_mul(4);
            let bytes: [u8; 4] = blob
                .get(off..off.saturating_add(4))
                .and_then(|s| <[u8; 4]>::try_from(s).ok())
                .ok_or_else(|| ModelError::Deserialize("CTRBlob read out of range".to_owned()))?;
            counts.push(i64::from(i32::from_le_bytes(bytes)));
        }
        int_counts.push(counts);
    }
    Ok(int_counts)
}

// ---------------------------------------------------------------------------
// .cbm-style binary blob (de)serialize — bounds-before-slice (Security V5,
// mirrors cbm.rs:240-270). Self-describing little-endian layout:
//
//   u32 LE  table_count
//   repeat table_count times:
//     u32 LE key_len ; key_len bytes (UTF-8 key)
//     i8      ctr_type
//     u32 LE  target_classes_count
//     i64 LE  counter_denominator
//     u8      is_mean (1 = mean histories, 0 = int counts)
//     u32 LE  bucket_count
//     u32 LE  stride            (count width per bucket; mean => 2)
//     repeat bucket_count:
//       u64 LE hash
//       if is_mean: f32 LE Sum ; i64 LE Count
//       else:       stride * (i64 LE count)
//
// Every declared length is BOUNDED against the remaining bytes BEFORE slicing.
// ---------------------------------------------------------------------------

/// A cursor over an untrusted blob that reads fixed-width little-endian values
/// with bounds checks, never panicking on a short / oversized buffer
/// (Security V5, T-05-04-V5).
struct BlobReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> BlobReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ModelError> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            ModelError::Deserialize("ctr_data blob length overflow".to_owned())
        })?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| {
            ModelError::Deserialize(format!(
                "ctr_data blob truncated: need {n} bytes at {} of {}",
                self.pos,
                self.buf.len()
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        self.take(1)?
            .first()
            .copied()
            .ok_or_else(|| ModelError::Deserialize("ctr_data u8 read failed".to_owned()))
    }

    fn i8(&mut self) -> Result<i8, ModelError> {
        Ok(self.u8()? as i8)
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let b = self.take(4)?;
        let arr: [u8; 4] = b.try_into().map_err(|_| {
            ModelError::Deserialize("ctr_data u32 read failed".to_owned())
        })?;
        Ok(u32::from_le_bytes(arr))
    }

    fn i64(&mut self) -> Result<i64, ModelError> {
        let b = self.take(8)?;
        let arr: [u8; 8] = b.try_into().map_err(|_| {
            ModelError::Deserialize("ctr_data i64 read failed".to_owned())
        })?;
        Ok(i64::from_le_bytes(arr))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        let b = self.take(8)?;
        let arr: [u8; 8] = b.try_into().map_err(|_| {
            ModelError::Deserialize("ctr_data u64 read failed".to_owned())
        })?;
        Ok(u64::from_le_bytes(arr))
    }

    fn f32(&mut self) -> Result<f32, ModelError> {
        let b = self.take(4)?;
        let arr: [u8; 4] = b.try_into().map_err(|_| {
            ModelError::Deserialize("ctr_data f32 read failed".to_owned())
        })?;
        Ok(f32::from_le_bytes(arr))
    }
}

/// A soft cap on a single declared collection length, so a hostile blob cannot
/// drive a huge pre-allocation before the per-element reads bound it
/// (DoS guard, T-05-04-02). The bound is generous (16M) — far above any real
/// model — and the actual reads still bound every byte.
const MAX_DECLARED_LEN: usize = 16 * 1024 * 1024;

fn bounded_len(declared: u32) -> Result<usize, ModelError> {
    let n = declared as usize;
    if n > MAX_DECLARED_LEN {
        return Err(ModelError::Deserialize(format!(
            "ctr_data declared length {n} exceeds cap {MAX_DECLARED_LEN}"
        )));
    }
    Ok(n)
}

/// Encode `ctr_data` to the self-describing little-endian blob (the model-parts
/// region of a `.cbm`).
#[must_use]
pub fn encode_ctr_data(ctr_data: &CtrData) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(ctr_data.tables.len() as u32).to_le_bytes());
    for (key, table) in &ctr_data.tables {
        let key_bytes = key.as_bytes();
        out.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(key_bytes);
        out.push(table.ctr_type.as_i8() as u8);
        out.extend_from_slice(&(table.target_classes_count as u32).to_le_bytes());
        out.extend_from_slice(&table.counter_denominator.to_le_bytes());

        let is_mean = table.ctr_type.is_mean();
        out.push(u8::from(is_mean));
        let bucket_count = if is_mean {
            table.mean.len()
        } else {
            table.int_counts.len()
        };
        out.extend_from_slice(&(bucket_count as u32).to_le_bytes());
        let stride = if is_mean {
            2
        } else {
            table.int_counts.first().map_or(0, Vec::len)
        };
        out.extend_from_slice(&(stride as u32).to_le_bytes());

        for i in 0..bucket_count {
            let hash = table.hashes.get(i).copied().unwrap_or(0);
            out.extend_from_slice(&hash.to_le_bytes());
            if is_mean {
                let (sum, count) = table.mean.get(i).copied().unwrap_or((0.0, 0));
                out.extend_from_slice(&sum.to_le_bytes());
                out.extend_from_slice(&count.to_le_bytes());
            } else if let Some(counts) = table.int_counts.get(i) {
                for j in 0..stride {
                    let c = counts.get(j).copied().unwrap_or(0);
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
    }
    out
}

/// Decode the self-describing little-endian `ctr_data` blob, validating every
/// declared length BEFORE slicing (Security V5, T-05-04-V5). A truncated /
/// oversized / malformed blob → typed [`ModelError`], never a panic.
///
/// # Errors
/// [`ModelError::Deserialize`] on truncation, an oversized declared length, a
/// bad CTR type, or a bad UTF-8 key.
pub fn decode_ctr_data(buf: &[u8]) -> Result<CtrData, ModelError> {
    let mut reader = BlobReader::new(buf);
    let table_count = bounded_len(reader.u32()?)?;
    let mut tables = BTreeMap::new();

    for _ in 0..table_count {
        let key_len = bounded_len(reader.u32()?)?;
        let key_bytes = reader.take(key_len)?;
        let key = std::str::from_utf8(key_bytes)
            .map_err(|_| ModelError::Deserialize("ctr_data key is not UTF-8".to_owned()))?
            .to_owned();

        let ctr_type = ECtrType::from_i8(reader.i8()?)?;
        let target_classes_count = bounded_len(reader.u32()?)?;
        let counter_denominator = reader.i64()?;
        let is_mean = reader.u8()? != 0;
        let bucket_count = bounded_len(reader.u32()?)?;
        let stride = bounded_len(reader.u32()?)?;

        let mut hashes = Vec::with_capacity(bucket_count.min(MAX_DECLARED_LEN));
        let mut int_counts = Vec::new();
        let mut mean = Vec::new();

        for _ in 0..bucket_count {
            let hash = reader.u64()?;
            hashes.push(hash);
            if is_mean {
                let sum = reader.f32()?;
                let count = reader.i64()?;
                mean.push((sum, count));
            } else {
                let mut counts = Vec::with_capacity(stride);
                for _ in 0..stride {
                    counts.push(reader.i64()?);
                }
                int_counts.push(counts);
            }
        }

        tables.insert(
            key,
            CtrValueTable {
                ctr_type,
                target_classes_count,
                hashes,
                int_counts,
                mean,
                counter_denominator,
            },
        );
    }

    Ok(CtrData { tables })
}
