//! NumPy -> [`OwnedColumns`] ingestion for the 08-01 walking-skeleton slice.
//!
//! The smoke slice is deliberately strict (D-12): it REQUIRES a C-contiguous
//! float32 2-D array for `X` and a C-contiguous float32 1-D array for `y`, and
//! rejects anything else with a plain [`PyValueError`]. The typed
//! `CatBoostValueError` taxonomy and the multi-dtype / Pandas / Arrow / Polars
//! adapters land in plans 08-02 / 08-03.
//!
//! Per D-11 (own-before-detach), the borrowed NumPy buffer is COPIED into an
//! owned `Vec<Vec<f64>>` (and `Vec<f64>` label) BEFORE the caller releases the
//! GIL — no `PyReadonlyArray` borrow ever crosses a `py.detach()` boundary.

use catboost_rs::OwnedColumns;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Copy a C-contiguous float32 NumPy feature matrix (and optional label) into an
/// owned [`OwnedColumns`] (column-major SoA, cast f32 -> f64).
///
/// `x` must be a 2-D `(n_rows, n_features)` C-contiguous `float32` array; `y`,
/// when present, must be a 1-D `(n_rows,)` C-contiguous `float32` array. Returns
/// a [`PyValueError`] for any dtype / layout / shape mismatch (D-12 strict; no
/// silent coercion). The returned columns are fully owned, so the caller may
/// release the GIL immediately afterward (D-11).
///
/// # Errors
/// [`PyValueError`] if `x`/`y` are not C-contiguous float32 of the expected rank,
/// or if `y`'s length does not match `x`'s row count.
pub(crate) fn numpy_to_owned(
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<OwnedColumns> {
    // Borrow X as a read-only float32 2-D array. `extract` fails (TypeError) on a
    // wrong dtype / rank; map it to a ValueError with an actionable message.
    let x_arr: PyReadonlyArray2<f32> = x.extract().map_err(|_| {
        PyValueError::new_err(
            "X must be a 2-D float32 NumPy array; pass `X.astype(np.float32)` \
             (catboost-rs requires float32 input — no silent precision coercion)",
        )
    })?;
    if !x_arr.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "X must be C-contiguous; pass `np.ascontiguousarray(X, dtype=np.float32)`",
        ));
    }

    let view = x_arr.as_array();
    let shape = view.shape();
    let (n_rows, n_features) = match shape {
        [r, c] => (*r, *c),
        _ => {
            return Err(PyValueError::new_err(
                "X must be a 2-D (n_rows, n_features) array",
            ))
        }
    };

    // Column-major SoA copy (cast f32 -> f64). `view[[row, col]]` is bounds-safe
    // (ndarray panics out of range, but row/col are derived from `shape`).
    let mut float_cols: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for col in 0..n_features {
        let mut column = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            column.push(f64::from(view[[row, col]]));
        }
        float_cols.push(column);
    }

    let label: Vec<f64> = match y {
        Some(y_any) => {
            let y_arr: PyReadonlyArray1<f32> = y_any.extract().map_err(|_| {
                PyValueError::new_err(
                    "y must be a 1-D float32 NumPy array; pass `y.astype(np.float32)`",
                )
            })?;
            if !y_arr.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "y must be C-contiguous; pass `np.ascontiguousarray(y, dtype=np.float32)`",
                ));
            }
            let y_view = y_arr.as_array();
            if y_view.len() != n_rows {
                return Err(PyValueError::new_err(format!(
                    "y length ({}) does not match X row count ({n_rows})",
                    y_view.len()
                )));
            }
            y_view.iter().map(|&v| f64::from(v)).collect()
        }
        None => Vec::new(),
    };

    Ok(OwnedColumns::new(float_cols, label))
}
