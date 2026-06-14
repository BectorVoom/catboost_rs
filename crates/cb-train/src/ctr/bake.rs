//! Whole-set inference CTR-table bake (ORD-05, Plan 05-14).
//!
//! After the boosting loop chooses its tensor-CTR splits, each chosen
//! `(projection, ctr_type, prior)` needs its WHOLE-LEARN-SET inference
//! `CtrValueTable` baked into the model's `ctr_data` so the apply path can
//! reproduce the upstream CTR value per category. This is the THIRD CTR
//! materialization (research "Summary"): the structure search uses the identity
//! learning fold, leaf VALUES use the shuffled averaging fold, and APPLY uses the
//! whole-set TOTALS (`online_ctr.cpp:916-939` `CalcFinalCtrsImpl`, the completed
//! per-bucket class counts — NOT a read-before-increment prefix).
//!
//! # What this builds
//!
//! For one chosen projection it accumulates the whole learn set into per-bucket
//! `[N0, N1]` class counts keyed on the COMBINED projection hash
//! ([`TProjection::combined_hash`], the SAME fold the apply path
//! `ctr_value_for_combined_projection` reconstructs), then exposes them as a
//! [`BakedCtrTable`] carrying the per-bucket combined hashes + class counts and
//! the inference `(Shift, Scale)` derived from the prior PAIR.
//!
//! # Scale / Shift derivation (`CalcNormalization`, `online_ctr.cpp:102-111`)
//!
//! `(shift, norm) = calc_normalization(prior_num)`; the baked `Shift = shift`,
//! `Scale = ctr_border_count / norm`. For the in-scope `Borders:Prior=0.5/1`
//! fixture: `calc_normalization(0.5)` → `shift=0`, `norm=1` → `Shift=0`,
//! `Scale = 15 / 1 = 15`, matching `model.json` `ctrs[0].{shift,scale}`.
//!
//! # No new histogram (reuse the locked primitive)
//!
//! The whole-set accumulation reuses [`crate::ctr::online::accumulate_online`]
//! over a synthesized COMBINED-key string column (one distinct string per distinct
//! combined hash, in first-seen order) and [`crate::ctr::final_ctr::build_final_ctr`]
//! — the SAME whole-set producer the single-feature path uses — so this does not
//! reinvent the per-bucket histogram. The bucket→combined-hash mapping is tracked
//! alongside (first-seen order matches `accumulate_online`'s `PerfectHash` bin
//! order) so the baked `hashes` are the combined projection keys the apply fold
//! reproduces.
//!
//! # Parity discipline
//!
//! Per-feature hashes via [`cb_data::calc_cat_feature_hash`] (never a model's
//! stored CTR hash map). Counts are checked i64 accumulation bounded by `N`
//! (WR-02). Checked `.get` only; no `unwrap`/`expect`/panic; no `anyhow`.

use std::collections::HashMap;

use cb_core::{CbError, CbResult};
use cb_data::calc_cat_feature_hash;

use crate::ctr::calc_ctr::calc_normalization;
use crate::ctr::final_ctr::build_final_ctr;
use crate::ctr::online::accumulate_online;
use crate::ctr::ECtrType;
use crate::projection::TProjection;

/// One baked whole-set inference CTR table for a chosen projection: the per-bucket
/// combined projection hashes + class counts (Borders `[N0, N1]`), the inference
/// `(Shift, Scale)`, and the prior PAIR. `cb-model` lifts this into a
/// `CtrValueTable` keyed by the canonical `(projection, ctr_type)` key (the SAME
/// key the apply path reconstructs).
#[derive(Debug, Clone, PartialEq)]
pub struct BakedCtrTable {
    /// The combined categorical projection (sorted member set) this table serves.
    pub projection: TProjection,
    /// The CTR type i8 discriminant (the SAME values as [`ECtrType`] /
    /// `cb_model::ECtrType`).
    pub ctr_type: i8,
    /// The number of target classes (`TargetClassesCount`, 2 for binclf).
    pub target_classes_count: usize,
    /// Per-bucket combined projection hashes (the apply-time lookup key folds the
    /// SAME `fold_cat_hash` over the document's member hashes — `combined_hash`).
    pub hashes: Vec<u64>,
    /// Per-bucket integer class counts `[N0, N1, …]` (Borders/Buckets), one inner
    /// vector per bucket. Empty for the mean / counter types.
    pub int_counts: Vec<Vec<i64>>,
    /// `CounterDenominator` (Counter / FeatureFreq); `0` otherwise.
    pub counter_denominator: i64,
    /// The inference `Shift` (`calc_normalization(prior_num)` → `shift`).
    pub shift: f64,
    /// The inference `Scale` (`ctr_border_count / norm`).
    pub scale: f64,
    /// The CTR prior numerator (`PriorNum`), carried as a PAIR.
    pub prior_num: f64,
    /// The CTR prior denominator (`PriorDenom`).
    pub prior_denom: f64,
}

/// The baked `ctr_data` the trainer exposes from [`crate::train_cat`]: one
/// [`BakedCtrTable`] per DISTINCT chosen CTR split. `cb-model` converts it into a
/// `cb_model::CtrData` keyed by the canonical apply key.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BakedCtrData {
    /// The baked tables in chosen order.
    pub tables: Vec<BakedCtrTable>,
}

/// Bake the WHOLE-LEARN-SET inference CTR table for one chosen `projection`
/// (`CalcFinalCtrsImpl`, `online_ctr.cpp:916-939` — completed per-bucket class
/// counts, NOT the prefix). Accumulates over the COMBINED projection hash and
/// derives the inference `(Shift, Scale)` from the prior PAIR.
///
/// `cat_columns[c]` is cat feature `c`'s per-object value column (A4 string form);
/// `target_class[i]` is object `i`'s binclf class in `[0, classes)`;
/// `classes` is the target-class count (2 for binclf); `ctr_border_count` is the
/// Borders CTR border count (15); `prior_num`/`prior_denom` are the prior PAIR.
///
/// # Errors
/// - [`CbError::Degenerate`] if `cat_columns` is empty, `target_class` is shorter
///   than the columns, a projection member is out of range, or `prior_denom == 0`.
pub fn bake_ctr_table(
    cat_columns: &[Vec<String>],
    projection: &TProjection,
    target_class: &[usize],
    classes: usize,
    ctr_border_count: usize,
    prior_num: f64,
    prior_denom: f64,
) -> CbResult<BakedCtrTable> {
    if cat_columns.is_empty() {
        return Err(CbError::Degenerate(
            "bake_ctr_table: no categorical columns supplied".to_owned(),
        ));
    }
    if prior_denom == 0.0 {
        return Err(CbError::Degenerate(
            "bake_ctr_table: prior_denom must be non-zero".to_owned(),
        ));
    }
    // The document count is the first column's length; every member column must be
    // at least that long.
    let n = cat_columns.first().map_or(0, Vec::len);
    for &member in projection.cat_features() {
        let Some(col) = cat_columns.get(member) else {
            return Err(CbError::Degenerate(
                "bake_ctr_table: projection member out of range for cat_columns".to_owned(),
            ));
        };
        if col.len() < n {
            return Err(CbError::Degenerate(
                "bake_ctr_table: cat column shorter than document count".to_owned(),
            ));
        }
    }
    if target_class.len() < n {
        return Err(CbError::Degenerate(
            "bake_ctr_table: target_class shorter than document count".to_owned(),
        ));
    }

    // 1. Per-document COMBINED projection hash (the SAME key fold the apply path
    //    reconstructs): fold each member feature's per-document cat hash via
    //    TProjection::combined_hash. NEVER a model's stored CTR hash map.
    let feature_count = cat_columns.len();
    let mut combined_keys: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut feature_hashes: Vec<u32> = Vec::with_capacity(feature_count);
        for col in cat_columns {
            let value = col.get(i).map_or("", String::as_str);
            feature_hashes.push(calc_cat_feature_hash(value));
        }
        combined_keys.push(projection.combined_hash(&feature_hashes));
    }

    // 2. Synthesize a per-document string column with one DISTINCT token per
    //    distinct combined hash (the decimal hash string), tracking the combined
    //    hash in FIRST-SEEN order. accumulate_online's PerfectHash remaps these to
    //    dense first-seen bins, so the resulting per-bucket class counts align
    //    bin-for-bin with `first_seen_hashes` (the apply lookup is by hash, so this
    //    ordering only needs to be internally consistent).
    let mut first_seen: HashMap<u64, usize> = HashMap::with_capacity(n);
    let mut first_seen_hashes: Vec<u64> = Vec::new();
    let key_strings: Vec<String> = combined_keys
        .iter()
        .map(|&k| {
            let next = first_seen_hashes.len();
            first_seen.entry(k).or_insert_with(|| {
                first_seen_hashes.push(k);
                next
            });
            k.to_string()
        })
        .collect();
    let key_refs: Vec<&str> = key_strings.iter().map(String::as_str).collect();

    // 3. Whole-set per-bucket class counts via the SHARED accumulate_online +
    //    build_final_ctr producer (the inference TOTALS — completed counts, NOT a
    //    read-before-increment prefix). `target` (raw float) is irrelevant for the
    //    Borders class counts; pass a zero vector of matching length.
    let target_zero = vec![0.0_f64; n];
    let target_class_n = target_class.get(..n).unwrap_or(target_class).to_vec();
    let acc = accumulate_online(&key_refs, &target_class_n, &target_zero, classes, classes)?;
    let final_table = build_final_ctr(&acc, ECtrType::Borders);

    // 4. Reshape the flat bucket-major class counts into per-bucket `[N0, N1, …]`,
    //    keyed by the first-seen combined hash. accumulate_online's PerfectHash
    //    assigns bins in first-seen order, the SAME order as `first_seen_hashes`.
    let bucket_count = final_table.bucket_count;
    // The synthesized key strings are bijective with the distinct combined hashes,
    // so the PerfectHash bin count MUST equal the distinct combined-hash count. A
    // mismatch would mean two distinct combined hashes collided under
    // calc_cat_feature_hash of their decimal strings (a 32-bit collision — degenerate
    // for any realistic dataset); surface it as a typed error rather than silently
    // mis-aligning the baked counts.
    if bucket_count != first_seen_hashes.len() {
        return Err(CbError::Degenerate(
            "bake_ctr_table: combined-hash key-string collision broke bucket alignment".to_owned(),
        ));
    }
    let mut hashes: Vec<u64> = Vec::with_capacity(bucket_count);
    let mut int_counts: Vec<Vec<i64>> = Vec::with_capacity(bucket_count);
    for b in 0..bucket_count {
        let hash = first_seen_hashes.get(b).copied().unwrap_or(0);
        let start = b.saturating_mul(classes);
        let counts: Vec<i64> = (0..classes)
            .map(|c| final_table.int_counts.get(start + c).copied().unwrap_or(0))
            .collect();
        hashes.push(hash);
        int_counts.push(counts);
    }

    // Derive (Shift, Scale) from the prior PAIR (calc_normalization(prior_num)):
    // Shift = shift; Scale = ctr_border_count / norm (Borders:0.5/1 → 0, 15).
    let (shift, norm) = calc_normalization(prior_num);
    let scale = if norm == 0.0 {
        0.0
    } else {
        ctr_border_count as f64 / norm
    };

    Ok(BakedCtrTable {
        projection: projection.clone(),
        ctr_type: ECtrType::Borders.as_i8(),
        target_classes_count: classes,
        hashes,
        int_counts,
        counter_denominator: 0,
        shift,
        scale,
        prior_num,
        prior_denom,
    })
}
