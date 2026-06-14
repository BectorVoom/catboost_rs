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

use crate::ctr::calc_ctr::calc_ctr_online_bin;
use crate::ctr::online::online_ctr_prefix_binclf;
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
    /// The CTR type i8 discriminant (Borders head — the combinations_ctr /
    /// simple_ctr default; the SAME values as [`ECtrType`] / `cb_model::ECtrType`).
    pub ctr_type: i8,
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

    // 3. Scalar online prior from the PAIR (prior_denom == 1 ⇒ scalar == num).
    let prior_scalar = prior_num / prior_denom;
    // The read-before-increment online prefix over the combined bins (OBJECT
    // order). Reused verbatim — no re-derived prefix loop. target_class is sliced
    // to n so the length contract matches the permutation/bins.
    let target_class_n = target_class.get(..n).unwrap_or(target_class);
    let prefix =
        online_ctr_prefix_binclf(permutation, &combined_bins, target_class_n, prior_scalar)?;

    // 4. Quantize each document's online CTR value to an integer CTR bin (the
    //    implicit float -> ui8 truncation). good[i]/total[i] are in OBJECT order.
    let mut bins: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let good = prefix.good.get(i).copied().unwrap_or(0);
        let total = prefix.total.get(i).copied().unwrap_or(0);
        let bin_f = calc_ctr_online_bin(good as f64, total, prior_scalar, ctr_border_count);
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
        // Borders head — the combinations_ctr / simple_ctr default family.
        ctr_type: ECtrType::Borders.as_i8(),
        prior_num,
        prior_denom,
        bins,
        ctr_value: prefix.value,
    })
}
