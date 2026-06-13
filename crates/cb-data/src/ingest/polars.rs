//! Polars ingestion (D-05): a Polars `DataFrame` rides the SHARED Arrow
//! validation path after `rechunk()`.
//!
//! Each declared float column is `rechunk()`-ed to a single contiguous chunk,
//! its `f64` values are taken as a contiguous slice (`cont_slice()` — a
//! non-contiguous column is a typed [`cb_core::CbError::Ingestion`], never a
//! panic), and that slice is wrapped into an Arrow `Float64Array` and routed
//! through [`crate::ingest::arrow::arrow_f64_column`] — the EXACT same
//! dtype/NaN-in-categorical validation the Arrow path uses (no duplicated
//! validation logic, D-05). The result is identical owned columns to both the
//! Arrow and owned-`Vec` paths.
//!
//! No `unwrap` / `expect` / `panic` / `[]`-indexing appears in this module
//! (Shared Pattern C).

use arrow::array::{ArrayRef, Float64Array};
use cb_core::{CbError, CbResult};
use polars::prelude::DataFrame;

use crate::ingest::arrow::arrow_f64_column;
use crate::ingest::IngestSource;
use crate::pool::Pool;

/// A Polars-backed dataset source: the column names of the float features and
/// the label, resolved against a [`DataFrame`] and ingested through the shared
/// Arrow validation path into a [`Pool`].
///
/// `categorical_features` marks which float columns are declared categorical so
/// a smuggled `NaN` is rejected (threat T-02-14). Construct with
/// [`PolarsColumns::new`].
#[derive(Debug, Clone)]
pub struct PolarsColumns {
    frame: DataFrame,
    float_feature_names: Vec<String>,
    categorical_features: Vec<bool>,
    label_name: String,
}

impl PolarsColumns {
    /// Build from a `DataFrame`, the ordered float feature column names, the
    /// per-feature categorical flags, and the label column name.
    #[must_use]
    pub fn new(
        frame: DataFrame,
        float_feature_names: Vec<String>,
        categorical_features: Vec<bool>,
        label_name: impl Into<String>,
    ) -> Self {
        Self {
            frame,
            float_feature_names,
            categorical_features,
            label_name: label_name.into(),
        }
    }
}

/// Rechunk a named `f64` Polars column to one contiguous chunk and wrap it as an
/// Arrow `Float64Array` — the bridge onto the shared Arrow validation path.
///
/// # Errors
///
/// - [`CbError::Ingestion`] if the column is missing, is not `f64`, or (after
///   `rechunk`) is still non-contiguous.
fn column_to_arrow(frame: &DataFrame, name: &str) -> CbResult<ArrayRef> {
    let column = frame.column(name).map_err(|e| CbError::Ingestion {
        message: format!("column `{name}`: {e}"),
    })?;

    // One contiguous chunk, then a typed f64 view of it (D-05: rechunk -> shared
    // Arrow path). `cont_slice` errors (not panics) on a multi-chunk / nullable
    // column, surfaced as a typed ingestion error.
    let rechunked = column.rechunk();
    let f64_chunked = rechunked.f64().map_err(|e| CbError::Ingestion {
        message: format!("column `{name}` is not f64: {e}"),
    })?;
    let slice = f64_chunked.cont_slice().map_err(|e| CbError::Ingestion {
        message: format!("column `{name}` is non-contiguous after rechunk: {e}"),
    })?;

    Ok(std::sync::Arc::new(Float64Array::from(slice.to_vec())) as ArrayRef)
}

impl IngestSource for PolarsColumns {
    fn into_pool(self) -> CbResult<Pool> {
        let n_rows = self.frame.height();

        let mut float_columns: Vec<Vec<f64>> =
            Vec::with_capacity(self.float_feature_names.len());
        for (index, name) in self.float_feature_names.iter().enumerate() {
            let arrow_column = column_to_arrow(&self.frame, name)?;
            let categorical = self
                .categorical_features
                .get(index)
                .copied()
                .unwrap_or(false);
            float_columns.push(arrow_f64_column(&arrow_column, index, categorical)?);
        }

        let label_arrow = column_to_arrow(&self.frame, &self.label_name)?;
        let label = arrow_f64_column(&label_arrow, usize::MAX, false)?;

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
