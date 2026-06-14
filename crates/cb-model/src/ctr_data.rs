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
