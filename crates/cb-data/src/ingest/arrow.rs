//! Arrow ingestion (D-06): validate an Arrow `Float64Array` column at the trust
//! boundary and read it into the SAME owned `Vec<f64>` the owned-`Vec` path
//! produces.
//!
//! The shared validation/read routine [`arrow_f64_column`] is the single funnel
//! both the Arrow [`IngestSource`] impl and the Polars path
//! ([`crate::ingest::polars`]) call — Polars rechunks each `Series` to one
//! contiguous Arrow chunk, then routes through this exact code (no duplicated
//! validation logic, D-05).
//!
//! Validation rejects, as a typed [`cb_core::CbError`] (never a panic, never a
//! blind index):
//! - a non-`Float64` column dtype → [`CbError::Dtype`] (threat T-02-13);
//! - a `NaN` (or a `null`) in a column declared categorical →
//!   [`CbError::NanInCategorical`] (threat T-02-14);
//! - a column-length disagreement → [`CbError::LengthMismatch`] (threat T-02-13).
//!
//! # Null handling (CR-02 / threat T-02-14)
//!
//! Arrow stores nulls out-of-band in a validity bitmap, with the value buffer
//! at a null slot holding undefined payload (commonly `0.0`). Reading
//! `values()` alone would silently turn a `null` into `0.0`, bypassing the
//! missing-value pipeline. This module therefore consults the validity bitmap:
//! a `null` in a numeric column becomes `f64::NAN` (the missing-value
//! representation the quantizer's NanMode handling expects), and a `null` in a
//! column declared categorical is rejected exactly like a smuggled `NaN`.
//!
//! No `unwrap` / `expect` / `panic` / `[]`-indexing appears in this module
//! (Shared Pattern C).

use arrow::array::{Array, ArrayRef, Float64Array};
use arrow::datatypes::DataType;
use cb_core::{CbError, CbResult};

use crate::ingest::IngestSource;
use crate::pool::Pool;

/// Validate an Arrow column is a contiguous `Float64` array and read it into an
/// owned `Vec<f64>` — the same buffer the owned-`Vec` path yields.
///
/// When `categorical` is `true`, any `NaN` OR `null` in the column is rejected
/// with [`CbError::NanInCategorical`] (a missing value must never reach a
/// categorical hash). For a numeric column, `null` entries are normalized to
/// `f64::NAN` so they flow through the quantizer's NanMode handling (CR-02).
///
/// # Errors
///
/// - [`CbError::Dtype`] if `column.data_type()` is not `Float64`.
/// - [`CbError::Ingestion`] if the validated column cannot be downcast to a
///   `Float64Array` (a defensive guard; dtype is checked first).
/// - [`CbError::NanInCategorical`] if `categorical` is set and the column holds a
///   `NaN` or a `null`.
pub fn arrow_f64_column(
    column: &ArrayRef,
    column_index: usize,
    categorical: bool,
) -> CbResult<Vec<f64>> {
    if column.data_type() != &DataType::Float64 {
        return Err(CbError::Dtype {
            expected: "Float64",
            got: format!("{:?}", column.data_type()),
        });
    }

    // dtype is Float64, so the downcast must succeed; the `ok_or_else` keeps the
    // path panic-free rather than calling the panicking `as_primitive`.
    let typed = column
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| CbError::Ingestion {
            message: format!("column {column_index}: Float64 downcast failed"),
        })?;

    // A categorical column must never carry a missing value: a `null` (tracked
    // in the validity bitmap, NOT the value buffer) is rejected exactly like a
    // smuggled `NaN`. Checking `null_count()` first catches the bitmap case the
    // raw `values()` scan below cannot see (CR-02 / threat T-02-14).
    if categorical {
        if typed.null_count() > 0 {
            return Err(CbError::NanInCategorical {
                column: column_index,
            });
        }
        for &v in typed.values().iter() {
            if v.is_nan() {
                return Err(CbError::NanInCategorical {
                    column: column_index,
                });
            }
        }
    }

    // Numeric path. When the column carries no nulls, the value buffer is a
    // faithful view and is copied once. When it has nulls, the null slots hold
    // undefined payload, so each null must be explicitly materialized as
    // `f64::NAN` (the missing-value representation) by consulting the validity
    // bitmap via `is_null` (CR-02).
    if typed.null_count() == 0 {
        Ok(typed.values().to_vec())
    } else {
        Ok((0..typed.len())
            .map(|i| {
                if typed.is_null(i) {
                    f64::NAN
                } else {
                    typed.value(i)
                }
            })
            .collect())
    }
}

/// An Arrow-backed dataset source: float feature columns (one `ArrayRef` per
/// feature) plus a label column, ingested through the shared Arrow validation
/// path into a [`Pool`].
///
/// `categorical_features` marks which float columns are declared categorical so
/// a smuggled `NaN` is rejected (threat T-02-14). Construct with
/// [`ArrowColumns::new`].
#[derive(Debug, Clone)]
pub struct ArrowColumns {
    float_features: Vec<ArrayRef>,
    categorical_features: Vec<bool>,
    label: ArrayRef,
}

impl ArrowColumns {
    /// Build from Arrow float feature columns and a label column.
    ///
    /// `categorical_features[i]` flags float column `i` as categorical (NaN
    /// rejected). A shorter / empty `categorical_features` treats the missing
    /// tail as non-categorical.
    #[must_use]
    pub fn new(
        float_features: Vec<ArrayRef>,
        categorical_features: Vec<bool>,
        label: ArrayRef,
    ) -> Self {
        Self {
            float_features,
            categorical_features,
            label,
        }
    }
}

impl IngestSource for ArrowColumns {
    fn into_pool(self) -> CbResult<Pool> {
        let n_rows = self.label.len();

        let mut float_columns: Vec<Vec<f64>> = Vec::with_capacity(self.float_features.len());
        for (index, column) in self.float_features.iter().enumerate() {
            if column.len() != n_rows {
                return Err(CbError::LengthMismatch {
                    column: format!("float_feature[{index}]"),
                    expected: n_rows,
                    actual: column.len(),
                });
            }
            let categorical = self
                .categorical_features
                .get(index)
                .copied()
                .unwrap_or(false);
            float_columns.push(arrow_f64_column(column, index, categorical)?);
        }

        let label = arrow_f64_column(&self.label, usize::MAX, false)?;

        Ok(Pool::from_validated_columns(
            n_rows,
            float_columns,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            label,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
    }
}
