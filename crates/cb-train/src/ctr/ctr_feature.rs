//! Combined-projection ONLINE CTR-feature materialization (ORD-05, Plan 05-11
//! Task 1) — the per-fold online-CTR-during-growth path of upstream
//! `greedy_tensor_search.cpp` AddTreeCtrs.
//!
//! TODO(05-11 GREEN): implement materialize_ctr_feature.

use cb_core::{CbError, CbResult};

use crate::projection::TProjection;

/// The per-document materialized CTR-feature column for one candidate projection.
#[derive(Debug, Clone, PartialEq)]
pub struct CtrFeatureColumn {
    /// The combined categorical projection.
    pub projection: TProjection,
    /// The CTR type i8 discriminant.
    pub ctr_type: i8,
    /// The CTR prior numerator (`PriorNum`).
    pub prior_num: f64,
    /// The CTR prior denominator (`PriorDenom`).
    pub prior_denom: f64,
    /// The per-document quantized CTR bins.
    pub bins: Vec<u32>,
    /// The per-document raw online CTR values.
    pub ctr_value: Vec<f64>,
}

/// Materialize a per-document combined-projection online CTR feature column.
///
/// # Errors
/// Not yet implemented (RED).
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn materialize_ctr_feature(
    cat_columns: &[Vec<String>],
    projection: &TProjection,
    permutation: &[i32],
    target_class: &[usize],
    prior_num: f64,
    prior_denom: f64,
    ctr_border_count: usize,
) -> CbResult<CtrFeatureColumn> {
    Err(CbError::Degenerate("materialize_ctr_feature: not yet implemented".to_owned()))
}
