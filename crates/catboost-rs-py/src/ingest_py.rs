//! Multi-source ingestion (PYAPI-04): NumPy / Pandas / Arrow / Polars ->
//! [`OwnedColumns`], with strict D-12 validation and the D-11 own-before-detach
//! discipline that makes PYAPI-06 buffer-safety a code property.
//!
//! All four sources converge on the EXISTING
//! [`catboost_rs::OwnedColumns`]/[`catboost_rs::IngestSource`] seam (the cb-data
//! `ingest` mod doc already anticipates this phase) — no new ingestion seam, no
//! re-implemented length checks. Every adapter COPIES the borrowed Python buffer
//! into owned `Vec`s BEFORE returning, so no `PyReadonlyArray` / Arrow capsule
//! borrow ever crosses a `py.detach()` boundary (D-11; threat T-08-08).
//!
//! Validation is strict (D-12): float64 / non-contiguous / ambiguous-object /
//! nullable inputs are rejected with a typed [`CatBoostValueError`] carrying an
//! actionable message (threats T-08-09 / T-08-10) — never silently coerced.
//!
//! # Dispatch order
//!
//! 1. An object exposing the Arrow PyCapsule interface (`__arrow_c_stream__` or
//!    `__arrow_c_array__`) — a pyarrow `Table` or a Polars `DataFrame` — goes
//!    through the single Arrow path (`arrow_to_owned`).
//! 2. A Pandas `DataFrame` (has `.to_numpy` + `.columns` + `.dtypes`) goes through
//!    the Pandas path (`pandas_to_owned`), which validates object columns.
//! 3. Anything else is treated as a NumPy ndarray (`numpy_to_owned`).

use arrow::array::{Array, Float32Array};
use arrow::datatypes::DataType;
use catboost_rs::OwnedColumns;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3_arrow::input::AnyRecordBatch;

use crate::errors::CatBoostValueError;

/// The single ingestion entry point: dispatch on the Python type of `x` and
/// return fully-owned [`OwnedColumns`].
///
/// `x` may be a NumPy ndarray, a Pandas `DataFrame`, a pyarrow `Table`, or a
/// Polars `DataFrame`. `y` (the optional label) follows the same float32 NumPy
/// contract on every path. `cat_features` lists the column indices that are
/// declared categorical (so a non-numeric Pandas column is allowed instead of
/// rejected); columns are read as raw strings into `OwnedColumns`'s cat slot.
///
/// In EVERY branch the data is copied/converted into owned `Vec`s before
/// returning — no borrow escapes (D-11). The caller may `py.detach()` for compute
/// immediately afterward.
///
/// # Errors
/// [`CatBoostValueError`] on any dtype / layout / shape / nullability / ambiguous
/// -object mismatch (D-12 strict; no silent coercion).
pub(crate) fn ingest_to_owned(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
    cat_features: Option<&[usize]>,
) -> PyResult<OwnedColumns> {
    // 1. Arrow PyCapsule interface (pyarrow Table OR Polars DataFrame share it).
    if has_arrow_capsule(py, x)? {
        return arrow_to_owned(x, y, cat_features);
    }
    // 2. Pandas DataFrame (duck-typed: a NumPy ndarray has none of these).
    if is_pandas_dataframe(py, x)? {
        return pandas_to_owned(py, x, y, cat_features);
    }
    // 3. NumPy ndarray (the 08-01 path, now raising the typed error).
    numpy_to_owned(x, y)
}

/// `true` if `obj` exposes the Arrow PyCapsule interface (`__arrow_c_stream__`
/// for a Table/DataFrame, or `__arrow_c_array__` for a single batch). Both
/// pyarrow `Table` and Polars `DataFrame` implement `__arrow_c_stream__`.
fn has_arrow_capsule(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.hasattr(intern!(py, "__arrow_c_stream__"))?
        || obj.hasattr(intern!(py, "__arrow_c_array__"))?)
}

/// `true` if `obj` duck-types as a Pandas `DataFrame` (the columns/dtypes/to_numpy
/// triple a NumPy ndarray does not have).
fn is_pandas_dataframe(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.hasattr(intern!(py, "columns"))?
        && obj.hasattr(intern!(py, "dtypes"))?
        && obj.hasattr(intern!(py, "to_numpy"))?)
}

/// Copy a C-contiguous float32 NumPy feature matrix (and optional label) into an
/// owned [`OwnedColumns`] (column-major SoA, cast f32 -> f64).
///
/// `x` must be a 2-D `(n_rows, n_features)` C-contiguous `float32` array; `y`,
/// when present, must be a 1-D `(n_rows,)` C-contiguous `float32` array. Returns a
/// typed [`CatBoostValueError`] (D-12 strict; no silent coercion) with an
/// actionable message for any dtype / layout / shape mismatch. The returned
/// columns are fully owned, so the caller may release the GIL immediately
/// afterward (D-11).
///
/// # Errors
/// [`CatBoostValueError`] if `x`/`y` are not C-contiguous float32 of the expected
/// rank, or if `y`'s length does not match `x`'s row count.
pub(crate) fn numpy_to_owned(
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<OwnedColumns> {
    // Borrow X as a read-only float32 2-D array. `extract` fails on a wrong dtype
    // / rank; map it to the typed CatBoostValueError with an actionable message
    // (D-12: reject float64, never coerce — threat T-08-09).
    let float_cols = numpy_matrix_to_cols(x)?;
    let n_rows = float_cols.first().map_or(0, Vec::len);
    let label = label_to_owned(y, n_rows)?;
    Ok(OwnedColumns::new(float_cols, label))
}

/// Validate a 2-D C-contiguous float32 NumPy array and copy it column-major into
/// owned `Vec<Vec<f64>>` (cast f32 -> f64). Shared by the NumPy and Pandas
/// (numeric block) paths so the strict D-12 contract is identical. The result is
/// fully owned — no borrow escapes (D-11).
///
/// # Errors
/// [`CatBoostValueError`] if the array is not a 2-D C-contiguous float32 array.
fn numpy_matrix_to_cols(x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let x_arr: PyReadonlyArray2<f32> = x.extract().map_err(|_| {
        CatBoostValueError::new_err(
            "X must be a 2-D float32 NumPy array; pass `X.astype(np.float32)` \
             (catboost-rs requires float32 input — no silent precision coercion)",
        )
    })?;
    if !x_arr.is_c_contiguous() {
        return Err(CatBoostValueError::new_err(
            "X must be C-contiguous; pass `np.ascontiguousarray(X, dtype=np.float32)`",
        ));
    }

    let view = x_arr.as_array();
    let (n_rows, n_features) = match view.shape() {
        [r, c] => (*r, *c),
        _ => {
            return Err(CatBoostValueError::new_err(
                "X must be a 2-D (n_rows, n_features) array",
            ))
        }
    };

    // Column-major SoA copy (cast f32 -> f64). Row/col are derived from `shape`,
    // so indexing is in range.
    let mut float_cols: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for col in 0..n_features {
        let mut column = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            column.push(f64::from(view[[row, col]]));
        }
        float_cols.push(column);
    }
    Ok(float_cols)
}

/// Read the optional float32 1-D NumPy label into an owned `Vec<f64>`, validating
/// the contiguity / dtype / length contract. An absent `y` yields an empty label
/// (the unsupervised / predict case). Shared by every ingest path so the label
/// contract is identical regardless of the feature source.
///
/// # Errors
/// [`CatBoostValueError`] if `y` is not a C-contiguous 1-D float32 array, or if
/// its length differs from `n_rows`.
fn label_to_owned(y: Option<&Bound<'_, PyAny>>, n_rows: usize) -> PyResult<Vec<f64>> {
    let Some(y_any) = y else {
        return Ok(Vec::new());
    };
    let y_arr: PyReadonlyArray1<f32> = y_any.extract().map_err(|_| {
        CatBoostValueError::new_err(
            "y must be a 1-D float32 NumPy array; pass `y.astype(np.float32)`",
        )
    })?;
    if !y_arr.is_c_contiguous() {
        return Err(CatBoostValueError::new_err(
            "y must be C-contiguous; pass `np.ascontiguousarray(y, dtype=np.float32)`",
        ));
    }
    let y_view = y_arr.as_array();
    if y_view.len() != n_rows {
        return Err(CatBoostValueError::new_err(format!(
            "y length ({}) does not match X row count ({n_rows})",
            y_view.len()
        )));
    }
    Ok(y_view.iter().map(|&v| f64::from(v)).collect())
}

/// Convert a Pandas `DataFrame` into owned columns.
///
/// Numeric columns are extracted as a single C-contiguous float32 NumPy matrix
/// (`df[numeric].to_numpy(dtype=float32)`) and copied; non-numeric (object /
/// string) columns are allowed ONLY when their positional index appears in
/// `cat_features` (read as raw strings into the cat slot). A non-numeric column
/// NOT in `cat_features` is ambiguous and rejected by name with a
/// `cat_features` suggestion (threat T-08-10). Everything is owned before
/// returning (D-11).
///
/// [`OwnedColumns`] stores float and categorical features as two SEPARATE blocks
/// with no per-original-index column ordering. To keep `cat_features` indices (and
/// every other positional contract — `ignored_features`, monotone constraints,
/// feature weights, SHAP attribution, `feature_names_`) meaningful, categorical
/// columns MUST be trailing: every declared categorical index must be greater than
/// every numeric index. An interleaved categorical column (one sitting BETWEEN
/// numeric columns) would silently reorder the surviving numeric features relative
/// to their source positions, corrupting feature alignment (CR-01). Rather than
/// reorder silently, such a frame is rejected with an actionable error.
///
/// # Errors
/// [`CatBoostValueError`] if an object/string column is present without being
/// listed in `cat_features`, if a declared categorical column is interleaved
/// between numeric columns, or on any label-contract violation.
fn pandas_to_owned(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
    cat_features: Option<&[usize]>,
) -> PyResult<OwnedColumns> {
    let cats = cat_features.unwrap_or(&[]);
    let columns = df.getattr(intern!(py, "columns"))?;
    let n_cols = columns.len()?;

    // Identify each column's kind by positional index. A column is "categorical"
    // iff its index is listed in cat_features. Any other non-numeric column is
    // ambiguous (D-12 / T-08-10).
    let dtypes = df.getattr(intern!(py, "dtypes"))?;
    let mut numeric_names: Vec<Bound<'_, PyAny>> = Vec::new();
    let mut cat_names: Vec<Bound<'_, PyAny>> = Vec::new();
    // Track whether a categorical column has been seen yet so an interleaved
    // numeric-after-categorical column (i.e. a non-trailing categorical block) is
    // rejected rather than silently reordered (CR-01).
    let mut seen_cat = false;

    for idx in 0..n_cols {
        let col_name = columns.get_item(idx)?;
        let is_cat = cats.contains(&idx);
        if is_cat {
            seen_cat = true;
            cat_names.push(col_name);
            continue;
        }
        if seen_cat {
            // A numeric column AFTER a categorical column means the categorical
            // block is interleaved, not trailing. Splitting into independent
            // float/cat blocks here would collapse the surviving numeric columns
            // to new positions and scramble every positional feature contract
            // (CR-01) — reject instead of silently reordering.
            return Err(CatBoostValueError::new_err(format!(
                "categorical columns must be trailing: column index {idx} ('{}') is \
                 numeric but appears AFTER a categorical column listed in \
                 `cat_features`. catboost-rs ingestion preserves positional feature \
                 indices, so reorder the DataFrame to place all `cat_features` \
                 columns last (ordered numeric/categorical interleaving is not yet \
                 supported)",
                col_name.str()?,
            )));
        }
        // dtype.kind in {b,i,u,f} is numeric; otherwise it is object/other.
        let dtype = dtypes.get_item(&col_name)?;
        let kind: String = dtype.getattr(intern!(py, "kind"))?.extract()?;
        if matches!(kind.as_str(), "b" | "i" | "u" | "f") {
            numeric_names.push(col_name);
        } else {
            return Err(CatBoostValueError::new_err(format!(
                "column '{}' has a non-numeric dtype (kind '{kind}'); if it is a \
                 categorical feature pass its column index in `cat_features=[...]`, \
                 otherwise convert it to float32",
                col_name.str()?,
            )));
        }
    }

    // Numeric block: select the numeric columns, materialize a float32 numpy
    // matrix, and route through the strict NumPy path (which copies + owns).
    let float_cols: Vec<Vec<f64>> = if numeric_names.is_empty() {
        Vec::new()
    } else {
        let np_module = py.import(intern!(py, "numpy"))?;
        let float32 = np_module.getattr(intern!(py, "float32"))?;
        let selector = pyo3::types::PyList::new(py, &numeric_names)?;
        let sub = df.get_item(selector)?;
        // .to_numpy(dtype=np.float32) — a contiguous owned float32 matrix.
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item(intern!(py, "dtype"), float32)?;
        let arr = sub.call_method(intern!(py, "to_numpy"), (), Some(&kwargs))?;
        // numpy_matrix_to_cols validates + copies (label handled separately below).
        numpy_matrix_to_cols(&arr)?
    };

    // Categorical block: read each declared categorical column as strings.
    let mut cat_cols: Vec<Vec<String>> = Vec::with_capacity(cat_names.len());
    for col_name in &cat_names {
        let series = df.get_item(col_name)?;
        // Use .astype(str).tolist() to obtain owned Python strings.
        let str_series = series.call_method1(intern!(py, "astype"), ("str",))?;
        let values = str_series.call_method0(intern!(py, "tolist"))?;
        let col: Vec<String> = values.extract()?;
        cat_cols.push(col);
    }

    let n_rows = pandas_n_rows(py, df)?;
    let label = label_to_owned(y, n_rows)?;
    let mut owned = OwnedColumns::new(float_cols, label);
    if !cat_cols.is_empty() {
        owned = owned.with_cat_features(cat_cols);
    }
    Ok(owned)
}

/// Row count of a Pandas `DataFrame` via `len(df)`.
fn pandas_n_rows(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<usize> {
    let builtins = py.import(intern!(py, "builtins"))?;
    let len = builtins.call_method1(intern!(py, "len"), (df,))?;
    len.extract()
}

/// Convert a pyarrow `Table` or a Polars `DataFrame` (one shared Arrow-capsule
/// path) into owned columns.
///
/// Each feature column is validated to be Arrow `Float32` with `null_count == 0`
/// (D-12 / threat T-08-10) and copied (cast f32 -> f64) into an owned column;
/// nothing borrows the capsule after this function returns (D-11). Arrow/Polars
/// categorical columns are not yet ingested, so a non-empty `cat_features` is
/// REJECTED (rather than silently dropped) — this keeps the Arrow path consistent
/// with the Pandas path, where the same logical DataFrame must yield the same
/// feature set regardless of source (CR-01). Declared-categorical Arrow columns
/// land in a later plan.
///
/// # Errors
/// [`CatBoostValueError`] if `cat_features` is non-empty (unsupported on the Arrow
/// path), if any column is not Float32, carries a null, or if the table cannot be
/// imported via the Arrow C Data Interface.
fn arrow_to_owned(
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
    cat_features: Option<&[usize]>,
) -> PyResult<OwnedColumns> {
    // Reject (do not silently drop) declared categorical columns on the Arrow path
    // so an Arrow/Polars table and the equivalent Pandas DataFrame never diverge in
    // their feature set (CR-01 consistency).
    if cat_features.is_some_and(|c| !c.is_empty()) {
        return Err(CatBoostValueError::new_err(
            "cat_features is not yet supported on the Arrow/Polars ingest path; \
             categorical Arrow/Polars columns are not yet ingested (convert to a \
             Pandas DataFrame with trailing categorical columns, or cast the columns \
             to Float32)",
        ));
    }
    // Import via the Arrow PyCapsule C Data Interface (pyo3-arrow handles capsule
    // lifetime). AnyRecordBatch::FromPyObject accepts anything with
    // __arrow_c_stream__ / __arrow_c_array__ (pyarrow Table and Polars DataFrame).
    let any_batch: AnyRecordBatch = x.extract().map_err(|e| {
        CatBoostValueError::new_err(format!(
            "could not import the Arrow/Polars table via the Arrow C Data Interface: {e}"
        ))
    })?;
    let table = any_batch
        .into_table()
        .map_err(|e| CatBoostValueError::new_err(e.to_string()))?;
    let (batches, schema) = table.into_inner();

    let n_features = schema.fields().len();
    let mut float_cols: Vec<Vec<f64>> = vec![Vec::new(); n_features];

    for field in schema.fields() {
        if field.data_type() != &DataType::Float32 {
            return Err(CatBoostValueError::new_err(format!(
                "Arrow/Polars column '{}' has dtype {:?}; catboost-rs requires Float32 \
                 (cast with .cast(pa.float32()) / .cast(pl.Float32))",
                field.name(),
                field.data_type(),
            )));
        }
    }

    // Concatenate the (possibly multi-chunk) batches column by column into owned
    // Vec<f64>. Each chunk is validated null-free (T-08-10).
    for batch in &batches {
        for col_idx in 0..n_features {
            let column = batch.column(col_idx);
            if column.null_count() > 0 {
                return Err(CatBoostValueError::new_err(format!(
                    "Arrow/Polars column '{}' contains nulls; catboost-rs does not \
                     support nullable feature columns (drop or impute the nulls first)",
                    schema.field(col_idx).name(),
                )));
            }
            let typed = column.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                CatBoostValueError::new_err(format!(
                    "Arrow/Polars column '{}' could not be read as Float32",
                    schema.field(col_idx).name(),
                ))
            })?;
            let dst = &mut float_cols[col_idx];
            dst.reserve(typed.len());
            for v in typed.values().iter() {
                dst.push(f64::from(*v));
            }
        }
    }

    let n_rows = float_cols.first().map_or(0, Vec::len);
    let label = label_to_owned(y, n_rows)?;
    Ok(OwnedColumns::new(float_cols, label))
}
