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
//! - a `NaN` in a column declared categorical → [`CbError::NanInCategorical`]
//!   (threat T-02-14);
//! - a column-length disagreement → [`CbError::LengthMismatch`] (threat T-02-13).
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
/// When `categorical` is `true`, any `NaN` in the column is rejected with
/// [`CbError::NanInCategorical`] (a `NaN` must never reach a categorical hash).
///
/// # Errors
///
/// - [`CbError::Dtype`] if `column.data_type()` is not `Float64`.
/// - [`CbError::Ingestion`] if the validated column cannot be downcast to a
///   `Float64Array` (a defensive guard; dtype is checked first).
/// - [`CbError::NanInCategorical`] if `categorical` is set and the column holds a
///   `NaN`.
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

    // Zero-copy view of the contiguous backing buffer; copied once into the
    // owned column the Pool takes ownership of.
    let values = typed.values();

    if categorical {
        for &v in values.iter() {
            if v.is_nan() {
                return Err(CbError::NanInCategorical {
                    column: column_index,
                });
            }
        }
    }

    Ok(values.to_vec())
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
