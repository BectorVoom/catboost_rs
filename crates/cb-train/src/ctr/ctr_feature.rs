//! Combined-projection ONLINE CTR-feature materialization (ORD-05, Plan 05-11
//! Task 1) — the per-fold online-CTR-during-growth path of upstream
//! `greedy_tensor_search.cpp` AddTreeCtrs.
//!
//! A tensor / simple CTR candidate is materialized into a per-document feature
//! column the oblivious tree search can split on. This is NOT new CTR math (D-05):
//! it folds each document's per-feature categorical hashes into one combined
//! projection key ([`TProjection::combined_hash`]), remaps the combined keys to
//! dense first-seen bins (the perfect-hash remap the online accumulation keys on),
//! runs the EXISTING read-before-increment online prefix
//! ([`crate::ctr::online::online_ctr_prefix_binclf`]) over those bins, then
//! quantizes each document's online CTR value to an integer CTR bin against the
//! Borders quantizer ([`crate::ctr::calc_ctr::calc_ctr_online_bin`]).
//!
//! # No leakage (the load-bearing property)
//!
//! The materialized value for a document is the read-BEFORE-increment prefix —
//! `(good + prior) / (total + 1)` computed over only the documents that precede
//! it in the permutation, NEVER its own label. This is inherited verbatim from
//! `online_ctr_prefix_binclf`; this module does not reimplement the prefix loop.
//!
//! # Prior as a num/denom PAIR (not a lossy scalar)
//!
//! The column carries the prior as the `prior_num` / `prior_denom` PAIR so it
//! matches [`crate::CtrSplitSpec`] (`tree.rs`, fields `prior_num`/`prior_denom`)
//! and `cb_model::CtrSplit.prior: Prior` (`num`/`denom`) — the Plan 05-12 bake
//! threads BOTH halves through `calc_normalization` / `from_trained`, so the
//! denominator is never silently lost. The SCALAR fed to the online prefix is
//! derived as `prior_num / prior_denom` (for the fixture `0.5 / 1.0 == 0.5`,
//! exact; a non-unit denominator divides correctly rather than collapsing).
//!
//! # Source of truth
//!
//! - `catboost/private/libs/algo/greedy_tensor_search.cpp` (AddTreeCtrs +
//!   the per-fold online CTR computed during growth) — the structural site this
//!   materialization fills.
//! - `online_ctr.cpp:300-307` — the read-before-increment prefix
//!   ([`crate::ctr::online::online_ctr_prefix_binclf`], reused verbatim).
//! - `online_ctr.h:128-131` — the `(ctr + shift) / norm * borderCount` online
//!   quantizer ([`crate::ctr::calc_ctr::calc_ctr_online_bin`], the implicit
//!   `float -> ui8` cast performed by the truncation here).
//! - `ctr_provider.h:65-78` — the combined-projection `CalcHash` fold
//!   ([`TProjection::combined_hash`]).
//!
//! # Parity discipline
//!
//! Per-feature hashes come from the single categorical-hash source
//! [`cb_data::calc_cat_feature_hash`] — NEVER a model's stored CTR hash map
//! (RESEARCH Anti-Pattern). Checked access only; no fallible-extraction or
//! aborting calls and no application-error crate in production (CLAUDE.md).
//! Tests live in the dedicated `tests/ctr_feature_materialize_test.rs` integration
//! file (source/test separation) — no embedded test module in this production file.

use std::collections::HashMap;

use cb_core::{CbError, CbResult};
use cb_data::calc_cat_feature_hash;

use crate::ctr::calc_ctr::{calc_ctr_online, calc_ctr_online_bin};
use crate::ctr::online::{
    online_class_prefix_column, online_counter_column, online_mean_prefix, SIMPLE_CLASSES_COUNT,
};
use crate::ctr::ECtrType;
use crate::projection::TProjection;

/// The per-document materialized CTR-feature column for one candidate projection
/// (ORD-05 / D-05): the combined-projection online CTR feature quantized to
/// Borders bins, carrying the prior as a num/denom PAIR matching
/// [`crate::CtrSplitSpec`] / `cb_model::CtrSplit`.
#[derive(Debug, Clone, PartialEq)]
pub struct CtrFeatureColumn {
    /// The combined categorical projection (sorted member set) this column was
    /// materialized over.
    pub projection: TProjection,
    /// The CTR type i8 discriminant this column was materialized for (the SAME
    /// values as [`ECtrType`] / `cb_model::ECtrType`).
    pub ctr_type: i8,
    /// The Buckets per-class numerator selector this column was materialized for
    /// (`GetTargetBorderCount`'s border index). `0` for every non-Buckets type.
    pub target_border_idx: usize,
    /// The CTR prior numerator (`PriorNum`), carried as a PAIR (never pre-divided).
    pub prior_num: f64,
    /// The CTR prior denominator (`PriorDenom`); `1.0` for the default priors.
    pub prior_denom: f64,
    /// The per-document quantized integer CTR bins in OBJECT order, each in
    /// `[0, ctr_border_count]` (the truncated [`calc_ctr_online_bin`]).
    pub bins: Vec<u32>,
    /// The per-document raw online CTR values in OBJECT order (the
    /// read-before-increment `(good + prior) / (total + 1)` prefix), kept for the
    /// materialization test's value-relation assertion and Plan 05-12 scoring.
    pub ctr_value: Vec<f64>,
    /// The number of DISTINCT combined-projection buckets (the projection's
    /// categorical cardinality — `TOnlineCtrUniqValuesCounts::Count`,
    /// `ctrs.h:50`). Used by the `model_size_reg` cat-feature-weight penalty in the
    /// structure search (`GetCatFeatureWeight`, greedy_tensor_search.cpp:908-932) to
    /// down-weight high-cardinality (e.g. combination) CTR candidates so they do not
    /// out-score a lower-cardinality simple CTR on a thin margin.
    pub bucket_count: usize,
}

/// Materialize a per-document combined-projection online CTR feature column for
/// one candidate `projection`, under the single learn `permutation`.
///
/// Steps (reusing the locked primitives — no re-derived CTR math):
/// 1. Per document `i`, fold each projection-member feature's
///    `calc_cat_feature_hash(&cat_columns[member][i])` into the combined key via
///    [`TProjection::combined_hash`] (members visited in the projection's sorted
///    order). NEVER a model's stored CTR hash map.
/// 2. Remap the combined keys to dense first-seen bins (insertion-order
///    `HashMap<u64, u32>` — the perfect-hash remap the online accumulation keys
///    on).
/// 3. Derive the scalar online prior `prior_num / prior_denom` and run the
///    EXISTING [`online_ctr_prefix_binclf`] over the combined bins to get the
///    per-document read-before-increment `(good, total, value)` in OBJECT order.
/// 4. Quantize each document's online CTR value to a CTR bin via
///    [`calc_ctr_online_bin`] truncated to `u32` (the implicit `float -> ui8`
///    cast), stored as `bins`; the raw online value is stored as `ctr_value`.
///
/// `target_class[i]` is object `i`'s binclf class in `[0, 2)`;
/// `ctr_border_count` is the Borders CTR border count
/// ([`crate::ctr_border_count_default`] = 15).
///
/// # The `counter_calc_method = Full` bucket-space rule (E21 spec, E22 impl)
///
/// Under [`crate::CounterCalcMethod::Full`], the combined-projection hash and
/// the first-seen `HashMap<u64, u32>` remap in step 2 are built over the
/// **CONCATENATION of the learn cat columns and EVERY eval set's cat columns**,
/// in the order `learn ++ eval[0] ++ eval[1] ++ …` — mirroring upstream, which
/// tallies over a `hashArr` built across learn AND every test set and, under
/// `Full`, sets `uniqValuesCounts.CounterCount = leafCount`
/// (`online_ctr.cpp:716-729`). Consequence: an eval-only categorical value that
/// never appears in the learn set gets ITS OWN bucket, so `bucket_count`
/// (`leafCount`) GROWS. Under `SkipTest` the remap is built over the learn
/// columns only and `bucket_count` is unchanged — today's behavior,
/// byte-identical. In BOTH settings the learn-document OUTPUT column stays
/// indexed by the LEARN slice: eval documents contribute to the tally and the
/// bucket space but produce no output rows, so `bins.len()` and
/// `ctr_value.len()` always equal the learn document count.
///
/// # Errors
/// - [`CbError::Degenerate`] if `cat_columns` is empty, a member column is
///   shorter than the permutation implies, or `prior_denom == 0`.
/// - Propagated from [`online_ctr_prefix_binclf`] (length / permutation-range
///   checks).
#[allow(clippy::too_many_arguments)]
pub fn materialize_ctr_feature(
    cat_columns: &[Vec<String>],
    projection: &TProjection,
    permutation: &[i32],
    target_class: &[usize],
    prior_num: f64, prior_denom: f64,
    ctr_border_count: usize,
    ctr_type: ECtrType,
    target_border_idx: usize,
    extra_cat_columns: &[Vec<String>],
) -> CbResult<CtrFeatureColumn> {
    // The document count is the permutation length (every learn document appears
    // once); the member columns must each be at least that long.
    let n = permutation.len();
    if cat_columns.is_empty() {
        return Err(CbError::Degenerate(
            "materialize_ctr_feature: no categorical columns supplied".to_owned(),
        ));
    }
    if prior_denom == 0.0 {
        return Err(CbError::Degenerate(
            "materialize_ctr_feature: prior_denom must be non-zero".to_owned(),
        ));
    }
    if target_class.len() < n {
        return Err(CbError::Degenerate(
            "materialize_ctr_feature: target_class shorter than permutation".to_owned(),
        ));
    }
    // Each projection member must index a column long enough for n documents.
    for &member in projection.cat_features() {
        let Some(col) = cat_columns.get(member) else {
            return Err(CbError::Degenerate(
                "materialize_ctr_feature: projection member out of range for cat_columns"
                    .to_owned(),
            ));
        };
        if col.len() < n {
            return Err(CbError::Degenerate(
                "materialize_ctr_feature: cat column shorter than permutation".to_owned(),
            ));
        }
    }

    // 1. Combined-projection key per document (folding each member's per-document
    //    cat hash via TProjection::combined_hash). The full per-feature hash
    //    vector for document i is built across ALL cat columns so combined_hash's
    //    member indices address into it directly (it indexes by absolute feature).
    let feature_count = cat_columns.len();
    let mut combined_keys: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut feature_hashes: Vec<u32> = Vec::with_capacity(feature_count);
        for col in cat_columns {
            // Checked .get — the projection-member columns were length-validated
            // above; non-member columns only need to be hashable when present.
            let value = col.get(i).map_or("", String::as_str);
            feature_hashes.push(calc_cat_feature_hash(value));
        }
        combined_keys.push(projection.combined_hash(&feature_hashes));
    }

    // 2. Remap combined keys to dense first-seen bins (insertion-order remap).
    let mut remap: HashMap<u64, u32> = HashMap::with_capacity(n);
    let mut combined_bins: Vec<u32> = Vec::with_capacity(n);
    for &key in &combined_keys {
        let next = remap.len();
        // The bin count cannot exceed n documents, so `next as u32` is loss-free
        // for any realistic dataset; guard the u32 bound defensively (no panic).
        if next >= u32::MAX as usize {
            return Err(CbError::OutOfRange(
                "materialize_ctr_feature: more than u32::MAX distinct combined keys".to_owned(),
            ));
        }
        let bin = *remap.entry(key).or_insert(next as u32);
        combined_bins.push(bin);
    }

    // 2b. The `counter_calc_method = Full` bucket-space rule (E22 implementing
    //     E21's spec; see the doc comment above). COUNTER-ONLY: upstream widens
    //     `uniqValuesCounts.CounterCount` — the Counter bucket space — to
    //     `leafCount` over the learn ++ every-eval hash array
    //     (`online_ctr.cpp:716-729`), while every other type's unique count
    //     stays learn-only. Eval keys extend the SAME first-seen remap AFTER
    //     the learn documents (learn bin numbering byte-identical; an eval-only
    //     value can only ever receive a NEW, higher bin) and produce
    //     `extra_bins` for the tally — never output rows. `extra_cat_columns`
    //     is EMPTY under `SkipTest` (the caller's gate), making this a no-op.
    let mut extra_bins: Vec<u32> = Vec::new();
    if ctr_type == ECtrType::Counter && !extra_cat_columns.is_empty() {
        let m = extra_cat_columns.iter().map(Vec::len).max().unwrap_or(0);
        let extra_count = extra_cat_columns.len();
        for i in 0..m {
            let mut feature_hashes: Vec<u32> = Vec::with_capacity(extra_count);
            for col in extra_cat_columns {
                let value = col.get(i).map_or("", String::as_str);
                feature_hashes.push(calc_cat_feature_hash(value));
            }
            let key = projection.combined_hash(&feature_hashes);
            let next = remap.len();
            if next >= u32::MAX as usize {
                return Err(CbError::OutOfRange(
                    "materialize_ctr_feature: more than u32::MAX distinct combined keys"
                        .to_owned(),
                ));
            }
            let bin = *remap.entry(key).or_insert(next as u32);
            extra_bins.push(bin);
        }
    }

    // 3. Scalar online prior from the PAIR (prior_denom == 1 ⇒ scalar == num).
    let prior_scalar = prior_num / prior_denom;
    // The read-before-increment online prefix over the combined bins (OBJECT
    // order). Reused verbatim — no re-derived prefix loop. target_class is sliced
    // to n so the length contract matches the permutation/bins.
    let target_class_n = target_class.get(..n).unwrap_or(target_class);

    // 3b. Per-type dispatch (SPEC-CTRT-06/-07/-08). Each arm supplies the
    //     per-document (numerator, denominator) INPUTS; the quantizer below is
    //     type-agnostic and stays byte-identical.
    let bucket_count = remap.len();
    let (nums, denoms, ctr_value, quantize_in_f32) = match ctr_type {
        ECtrType::Borders | ECtrType::Buckets => {
            // Borders at target_border_idx == 0 is EXACTLY the pre-dispatch
            // `online_ctr_prefix_binclf` path (E05's firewall proves the
            // equivalence bit-for-bit).
            let prefix = online_class_prefix_column(
                permutation,
                &combined_bins,
                target_class_n,
                SIMPLE_CLASSES_COUNT,
                target_border_idx,
                ctr_type,
                prior_scalar,
            )?;
            let nums: Vec<f64> = prefix.good.iter().map(|&g| g as f64).collect();
            (nums, prefix.total, prefix.value, false)
        }
        ECtrType::BinarizedTargetMeanValue => {
            let prefix = online_mean_prefix(
                permutation,
                &combined_bins,
                target_class_n,
                SIMPLE_CLASSES_COUNT,
                prior_scalar,
            )?;
            let nums: Vec<f64> = prefix.sum.iter().map(|&s| f64::from(s)).collect();
            // BTMV quantizes in f32, mirroring upstream's all-`float` CalcCTR.
            (nums, prefix.count, prefix.value, true)
        }
        ECtrType::Counter => {
            // Whole-set totals with the CONSTANT max-bucket denominator; no
            // permutation dependence at all (ctr_type.cpp:43-56). Under
            // `counter_calc_method = Full` the eval bins join the tally and the
            // MAX (step 2b); the output stays learn-indexed.
            let (totals, denominator) =
                online_counter_column(&combined_bins, &extra_bins, bucket_count);
            let nums: Vec<f64> = totals.iter().map(|&t| t as f64).collect();
            let denoms: Vec<i64> = vec![denominator; totals.len()];
            let value: Vec<f64> = nums
                .iter()
                .map(|&num| calc_ctr_online(num, denominator, prior_scalar))
                .collect();
            (nums, denoms, value, false)
        }
        ECtrType::FloatTargetMeanValue | ECtrType::FeatureFreq => {
            // Defence in depth — E02 already rejects these at `train_inner`, but a
            // direct caller must not get a silently-wrong column either.
            return Err(CbError::Unsupported(format!(
                "Ctr type {ctr_type:?} is not implemented on CPU yet \
                 (upstream catboost_options.cpp:504-509)"
            )));
        }
    };

    // 4. Quantize each document's online CTR value to an integer CTR bin (the
    //    implicit float -> ui8 truncation). Inputs are in OBJECT order.
    let mut bins: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let good = nums.get(i).copied().unwrap_or(0.0);
        let total = denoms.get(i).copied().unwrap_or(0);
        let bin_f = if quantize_in_f32 {
            // Upstream's CalcCTR is all-`float` for the mean types; compute in f32
            // and widen only at the end (research §A.0 caveat).
            let norm = (total as f32) + 1.0f32;
            let ctr = (good as f32 + prior_scalar as f32) / norm;
            f64::from(ctr * ctr_border_count as f32)
        } else {
            calc_ctr_online_bin(good, total, prior_scalar, ctr_border_count)
        };
        // Truncate toward zero, clamp into [0, ctr_border_count] (a degenerate
        // negative or over-range float maps to the nearest valid bin — no panic,
        // no out-of-range cast). The CTR is non-negative and bounded by the
        // quantizer, so the clamp only guards arithmetic edge cases.
        let truncated = bin_f.trunc();
        let clamped = if truncated < 0.0 {
            0u32
        } else if truncated > ctr_border_count as f64 {
            ctr_border_count as u32
        } else {
            truncated as u32
        };
        bins.push(clamped);
    }

    Ok(CtrFeatureColumn {
        projection: projection.clone(),
        ctr_type: ctr_type.as_i8(),
        target_border_idx,
        prior_num,
        prior_denom,
        bins,
        ctr_value,
        // Distinct combined-projection buckets (the projection cardinality) — the
        // size of the first-seen remap.
        bucket_count,
    })
}
