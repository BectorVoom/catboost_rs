//! The PYAPI-05 typed-exception taxonomy: one facade [`catboost_rs::CatBoostError`]
//! variant -> one specific, catchable Python exception.
//!
//! # Taxonomy
//!
//! - [`CatBoostError`] (base) subclasses `pyo3::exceptions::PyException`.
//! - `CatBoostParameterError` subclasses [`CatBoostError`] — raised by the
//!   `params.rs` registry validator (D-05/D-07) for an unknown / not-yet-implemented
//!   kwarg. It has NO facade-enum source (it is a binding-side validation error).
//! - `CatBoostValueError` subclasses [`CatBoostError`] — a dtype / layout / value
//!   boundary failure (D-12) and the target of `FeatureMismatch` / `Deserialize` /
//!   `SchemaVersion`.
//! - `NotFittedError` subclasses BOTH [`CatBoostError`] AND `ValueError`, so
//!   sklearn's not-fitted path (`check_estimator`) recognizes it.
//!
//! # Orphan rule (E0117)
//!
//! `catboost_rs::CatBoostError` and `pyo3::PyErr` are BOTH foreign to this crate,
//! so `impl From<catboost_rs::CatBoostError> for PyErr` is illegal. We wrap the
//! facade error in the LOCAL newtype [`PyCbError`] and impl `From<PyCbError> for
//! PyErr`; call sites convert with `.map_err(PyCbError)?`. The free function
//! [`to_pyerr`] is the single chokepoint performing the variant match.

use catboost_rs::CatBoostError as FacadeError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

create_exception!(catboost_rs, CatBoostError, PyException);
create_exception!(catboost_rs, CatBoostParameterError, CatBoostError);
create_exception!(catboost_rs, CatBoostValueError, CatBoostError);
// `NotFittedError` must subclass BOTH `CatBoostError` (taxonomy lineage) AND
// `ValueError` (so sklearn's `check_is_fitted` not-fitted path recognizes it).
// PyO3's `create_exception!` macro accepts only a single parent, so the macro
// cannot express this multiple-inheritance. We instead build `NotFittedError`
// dynamically in [`register`] with the two bases `(CatBoostError, ValueError)`
// via `PyType.__call__` (the Python `type(name, bases, dict)` 3-arg form), then
// add it to the module. Raising it from Rust is done by name-lookup on the
// module (see [`not_fitted_err`]).

/// The dynamically-built `NotFittedError` type (bases `(CatBoostError,
/// ValueError)`), cached per-interpreter so [`not_fitted_err`] can raise it
/// without re-importing the module.
static NOT_FITTED: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Build the `NotFittedError` type with the two bases `(CatBoostError,
/// ValueError)` via Python's 3-arg `type(name, bases, dict)`.
fn build_not_fitted(py: Python<'_>) -> PyResult<Py<PyType>> {
    let bases = PyTuple::new(
        py,
        [
            py.get_type::<CatBoostError>(),
            py.get_type::<PyValueError>(),
        ],
    )?;
    let ns = PyDict::new(py);
    let type_obj = py.get_type::<PyType>();
    let cls = type_obj.call1(("NotFittedError", bases, ns))?;
    let cls_ty = cls.cast_into::<PyType>()?;
    Ok(cls_ty.unbind())
}

/// The cached `NotFittedError` type for this interpreter (built on first use).
///
/// # Errors
/// Propagates any failure constructing the type object.
pub(crate) fn not_fitted_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    NOT_FITTED
        .get_or_try_init(py, || build_not_fitted(py))
        .map(|ty| ty.bind(py))
}

/// Raise a `NotFittedError` carrying `msg` (the typed not-fitted sentinel for
/// `predict`/`fit` ordering, sklearn parity).
pub(crate) fn not_fitted_err(py: Python<'_>, msg: &str) -> PyErr {
    fn build(py: Python<'_>, msg: &str) -> PyResult<PyErr> {
        let ty = not_fitted_type(py)?;
        let instance = ty.call1((msg,))?;
        Ok(PyErr::from_value(instance))
    }
    build(py, msg).unwrap_or_else(|e| e)
}

/// Local newtype wrapping the foreign facade error so `From<_> for PyErr` is
/// orphan-legal (E0117). Convert at call sites with `.map_err(PyCbError)?`.
pub(crate) struct PyCbError(pub(crate) FacadeError);

impl From<PyCbError> for PyErr {
    fn from(err: PyCbError) -> Self {
        to_pyerr(&err.0)
    }
}

/// Map a facade [`catboost_rs::CatBoostError`] onto its specific Python exception
/// (PYAPI-05, one variant -> one exception). The single conversion chokepoint.
///
/// - `FeatureMismatch` -> `CatBoostValueError` (a bad-input value error).
/// - `Deserialize` / `SchemaVersion` -> `CatBoostValueError` (malformed /
///   unsupported model = value error).
/// - `Io` -> `PyIOError` (a file-I/O failure surfaces as the stdlib I/O error).
/// - `Train` / `Model` -> base `CatBoostError` (internal training / model error).
/// - `Export` (EXPORT-01f) -> per `OnnxExportError` sub-variant: the four
///   guard-rejection variants (`CategoricalFeaturesUnsupported` /
///   `NonObliviousTreesUnsupported` / `RegionTreesUnsupported` /
///   `NonIntegerClassLabelsUnsupported`) map to `CatBoostValueError` (the
///   model itself is the "bad input" to the export operation, mirroring
///   `PartialDependence`'s own mapping); `Io` mirrors the top-level `Io`
///   arm's own mapping (`PyIOError`); `Encode` maps to the base
///   `CatBoostError` (an internal/unexpected failure, mirroring `Train`/`Model`
///   — not user-input-driven).
pub(crate) fn to_pyerr(err: &FacadeError) -> PyErr {
    match err {
        FacadeError::FeatureMismatch(m) => CatBoostValueError::new_err(m.clone()),
        FacadeError::Deserialize(m) | FacadeError::SchemaVersion(m) => {
            CatBoostValueError::new_err(m.clone())
        }
        FacadeError::Io(io) => PyIOError::new_err(io.to_string()),
        FacadeError::Train(c) => CatBoostError::new_err(c.to_string()),
        FacadeError::Model(m) => CatBoostError::new_err(m.to_string()),
        // An invalid partial-dependence request is a bad-input value error
        // (like FeatureMismatch): arity / out-of-range / duplicate / empty.
        FacadeError::PartialDependence(e) => CatBoostValueError::new_err(e.to_string()),
        FacadeError::Export(e) => match e {
            cb_model::OnnxExportError::CategoricalFeaturesUnsupported
            | cb_model::OnnxExportError::NonObliviousTreesUnsupported
            | cb_model::OnnxExportError::RegionTreesUnsupported
            | cb_model::OnnxExportError::NonIntegerClassLabelsUnsupported => {
                CatBoostValueError::new_err(e.to_string())
            }
            cb_model::OnnxExportError::Io(io) => PyIOError::new_err(io.to_string()),
            cb_model::OnnxExportError::Encode(_) => CatBoostError::new_err(e.to_string()),
        },
    }
}

/// Register the four exception types in the `#[pymodule]` so they are importable
/// as `catboost_rs.CatBoostError` etc. (catchable from Python).
///
/// # Errors
/// Propagates any failure adding a type object to the module.
pub(crate) fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CatBoostError", py.get_type::<CatBoostError>())?;
    m.add(
        "CatBoostParameterError",
        py.get_type::<CatBoostParameterError>(),
    )?;
    m.add("CatBoostValueError", py.get_type::<CatBoostValueError>())?;
    m.add("NotFittedError", not_fitted_type(py)?.clone())?;
    Ok(())
}
