//! Final (whole-set) CTR table build — the per-bucket counts baked into the
//! model's `ctr_data` section, one [`FinalCtrTable`] per [`ECtrType`]
//! (RESEARCH "Final CTR table build", `online_ctr.cpp:916-939`).
//!
//! # Whole-set table (`CalcFinalCtrsImpl`)
//!
//! For each document, accumulate into its bucket (`online_ctr.cpp:916-939`):
//! - **Borders / Buckets** (multi-class): `++ctrIntArray[classes * elemId +
//!   targetClass]` — the per-bucket class counts.
//! - **BinarizedTargetMeanValue:** `ctrMean[elemId].Add(targetClass /
//!   targetBorderCount)`.
//! - **FloatTargetMeanValue:** `ctrMean[elemId].Add(target)` (raw) — the
//!   final-CTR path ONLY (NOT in the online dispatch, Pitfall 5).
//! - **Counter / FeatureFreq:** `++ctrIntArray[elemId]` (the bucket total).
//!
//! Then the per-type denominator (Pitfall 4):
//! - **Counter:** `CounterDenominator = *MaxElement(counts)` (the MAX bucket
//!   total).
//! - **FeatureFreq:** `CounterDenominator = totalSampleCount` (the DISTINCT total
//!   sample count) — the two share the `++count` accumulation but DIFFER only in
//!   this denominator.
//!
//! # Six types (D-07)
//!
//! All six [`ECtrType`] values build through [`build_final_ctr`]; the whole-set
//! accumulator ([`crate::ctr::online::OnlineCtrAccumulator`]) already holds every
//! histogram, so each type reads whichever it needs.
//!
//! # Parity discipline
//!
//! Integer counts are EXACT integer sums; the per-bucket mean `Sum` is the
//! running `f32` from [`crate::ctr::online::TCtrMeanHistory`]. The only
//! parity-critical FLOAT reduction here (the Counter max / FeatureFreq total are
//! integer) is absent — there is no float WHOLE-vector sum in the table build.
//! Checked access only; no `unwrap`/`expect`/panic/raw index; no `anyhow`.

use crate::ctr::online::OnlineCtrAccumulator;
use crate::ctr::ECtrType;

/// One built whole-set CTR table for a single [`ECtrType`] — the per-bucket
/// counts and the type-specific denominator, the data baked into the model's
/// `ctr_data` section (`TCtrValueTable`).
#[derive(Debug, Clone, PartialEq)]
pub struct FinalCtrTable {
    /// The CTR type this table was built for.
    pub ctr_type: ECtrType,
    /// The number of target classes (`TargetClassesCount`, 2 for binclf).
    pub target_classes_count: usize,
    /// Per-bucket integer counts, flattened `[bucket0_class0, bucket0_class1,
    /// bucket1_class0, …]` for the class types, or `[bucket0_total,
    /// bucket1_total, …]` for Counter/FeatureFreq. Empty for the mean types
    /// (which carry `mean_sum`/`mean_count` instead).
    pub int_counts: Vec<i64>,
    /// Per-bucket mean sums (BinarizedTargetMeanValue/FloatTargetMeanValue);
    /// empty for the non-mean types.
    pub mean_sum: Vec<f32>,
    /// Per-bucket mean counts (paired with [`Self::mean_sum`]); empty for the
    /// non-mean types.
    pub mean_count: Vec<i64>,
    /// `CounterDenominator` — the MAX bucket total (Counter) or the total sample
    /// count (FeatureFreq); `0` for the types that carry no counter denominator.
    pub counter_denominator: i64,
    /// The number of distinct buckets.
    pub bucket_count: usize,
}

/// Build the whole-set [`FinalCtrTable`] for `ctr_type` from the completed
/// online accumulator (`CalcFinalCtrsImpl`, `online_ctr.cpp:916-939`).
///
/// `counter_calc_skip_test` selects the `CounterCalcMethod` semantics
/// (default `SkipTest`, Pitfall 4) — in this whole-learn-set build there are no
/// test documents, so the flag does not change the counts; it is recorded for the
/// model's `CounterCalcMethod` field and reserved for the tensor-CTR path.
#[must_use]
pub fn build_final_ctr(acc: &OnlineCtrAccumulator, ctr_type: ECtrType) -> FinalCtrTable {
    let classes = acc.classes;
    let bucket_count = acc.bucket_count;

    let mut table = FinalCtrTable {
        ctr_type,
        target_classes_count: classes,
        int_counts: Vec::new(),
        mean_sum: Vec::new(),
        mean_count: Vec::new(),
        counter_denominator: 0,
        bucket_count,
    };

    match ctr_type {
        ECtrType::Borders | ECtrType::Buckets => {
            // ++ctrIntArray[classes * elemId + targetClass] — per-bucket class
            // counts, flattened in bucket-major order.
            let mut flat = Vec::with_capacity(bucket_count.saturating_mul(classes));
            for hist in &acc.class_histories {
                for c in 0..classes {
                    flat.push(hist.n.get(c).copied().unwrap_or(0));
                }
            }
            table.int_counts = flat;
        }
        ECtrType::BinarizedTargetMeanValue => {
            // ctrMean[elemId] for Add(targetClass / targetBorderCount).
            table.mean_sum = acc.binarized_mean.iter().map(|m| m.sum).collect();
            table.mean_count = acc.binarized_mean.iter().map(|m| m.count).collect();
        }
        ECtrType::FloatTargetMeanValue => {
            // ctrMean[elemId] for Add(target) raw — final-CTR path ONLY.
            table.mean_sum = acc.float_mean.iter().map(|m| m.sum).collect();
            table.mean_count = acc.float_mean.iter().map(|m| m.count).collect();
        }
        ECtrType::Counter => {
            // ++ctrIntArray[elemId]; CounterDenominator = *MaxElement(counts)
            // — the MAX bucket total (Pitfall 4).
            table.int_counts = acc.total_counts.clone();
            table.counter_denominator = acc.total_counts.iter().copied().max().unwrap_or(0);
        }
        ECtrType::FeatureFreq => {
            // ++ctrIntArray[elemId]; CounterDenominator = totalSampleCount — the
            // DISTINCT total sample count (Pitfall 4: differs from Counter only
            // here). totalSampleCount is the number of accumulated documents
            // (sum of every bucket total).
            table.int_counts = acc.total_counts.clone();
            table.counter_denominator = acc.total_counts.iter().copied().sum();
        }
    }

    table
}
