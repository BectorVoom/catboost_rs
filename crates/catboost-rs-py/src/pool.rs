//! Native `Pool` `#[pyclass]` (PYAPI-03) — mirrors upstream `catboost.Pool`.
//!
//! A `Pool` bundles a feature matrix with its label and optional grouping /
//! weighting / pairing metadata, so a user can pass a single explicit `Pool` to
//! `fit` / `predict` instead of a bare framework object. The constructor routes
//! `data` + `label` through the shared [`crate::ingest_py::ingest_to_owned`]
//! adapter (so a `Pool` accepts the SAME NumPy / Pandas / Arrow / Polars sources
//! the estimators do) and attaches optional columns via the
//! [`catboost_rs::OwnedColumns`] `with_*` builder.
//!
//! All length / range validation is INHERITED from
//! [`catboost_rs::OwnedColumns::into_pool`] (the existing ingest seam) — never
//! re-implemented here (threat T-08-11). `__new__` keeps the build cheap by
//! storing the `OwnedColumns`; the length check fires lazily in [`Pool::to_pool`]
//! (the crate-internal entry the estimators call), surfacing
//! `CbError::LengthMismatch` as a [`crate::errors::CatBoostValueError`].

use catboost_rs::{IngestSource, OwnedColumns, Pool as FacadePool};
use pyo3::prelude::*;

use crate::errors::CatBoostValueError;
use crate::ingest_py::ingest_to_owned;

/// CatBoost-mirror `Pool`: an owned feature matrix + label (+ optional metadata),
/// constructed from any supported source and convertible into the facade
/// [`catboost_rs::Pool`] via the inherited `into_pool()` validation.
#[pyclass(name = "Pool")]
pub struct Pool {
    /// The owned columns; cloned into a facade `Pool` on demand (so the same
    /// `Pool` object can drive both `fit` and `predict`).
    owned: OwnedColumns,
    /// Cached row count (`num_row`), captured at construction from the feature
    /// matrix so the getter needs no rebuild.
    n_rows: usize,
    /// Cached feature-column count (`num_col`).
    n_cols: usize,
}

#[pymethods]
impl Pool {
    /// Construct a `Pool`, mirroring upstream `Pool.__init__` (RESEARCH 421-428).
    ///
    /// `data` + `label` go through the shared ingest adapter (NumPy / Pandas /
    /// Arrow / Polars); the optional columns attach via the `OwnedColumns`
    /// `with_*` chain. The supported optional columns in this slice are
    /// `cat_features` (column indices declared categorical), `weight`, `group_id`,
    /// `subgroup_id`, and `baseline`; `text_features` / `embedding_features` /
    /// `pairs` / `feature_names` are accepted in the signature (upstream parity)
    /// but not yet ingested (a later plan wires them). No length validation runs
    /// here — it fires in [`Pool::to_pool`] (inherited from `into_pool()`).
    ///
    /// # Errors
    /// [`CatBoostValueError`] on any dtype / layout / ambiguous-object /
    /// nullability failure from the ingest adapter (D-12).
    #[new]
    #[pyo3(signature = (
        data,
        label = None,
        cat_features = None,
        text_features = None,
        embedding_features = None,
        weight = None,
        group_id = None,
        subgroup_id = None,
        pairs = None,
        baseline = None,
        feature_names = None,
        thread_count = -1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        label: Option<&Bound<'_, PyAny>>,
        cat_features: Option<Vec<usize>>,
        text_features: Option<&Bound<'_, PyAny>>,
        embedding_features: Option<&Bound<'_, PyAny>>,
        weight: Option<Vec<f64>>,
        group_id: Option<Vec<u64>>,
        subgroup_id: Option<Vec<u64>>,
        pairs: Option<&Bound<'_, PyAny>>,
        baseline: Option<Vec<f64>>,
        feature_names: Option<&Bound<'_, PyAny>>,
        thread_count: i64,
    ) -> PyResult<Self> {
        // Signature-parity arguments not yet ingested in this slice. Bind them so
        // the upstream-mirror signature compiles; later plans wire them.
        let _ = (
            text_features,
            embedding_features,
            pairs,
            feature_names,
            thread_count,
        );

        // Own-before-detach: ingest_to_owned COPIES every buffer into owned
        // columns (D-11). `cat_features` declares which columns are categorical.
        let mut owned = ingest_to_owned(py, data, label, cat_features.as_deref())?;

        // Attach the supported optional columns via the inherited with_* chain.
        if let Some(w) = weight {
            owned = owned.with_weights(w);
        }
        if let Some(g) = group_id {
            owned = owned.with_group_id(g);
        }
        if let Some(sg) = subgroup_id {
            owned = owned.with_subgroup_id(sg);
        }
        if let Some(b) = baseline {
            owned = owned.with_baseline(b);
        }

        // Cache the shape from the owned columns (validation-free; no rebuild
        // needed for the getters). `into_pool()` length validation is deferred to
        // `to_pool`.
        let (n_rows, n_cols) = owned.feature_shape();
        Ok(Self {
            owned,
            n_rows,
            n_cols,
        })
    }

    /// Number of objects (rows) in the pool (upstream `Pool.num_row`).
    #[getter]
    fn num_row(&self) -> usize {
        self.n_rows
    }

    /// Number of feature columns (upstream `Pool.num_col`).
    #[getter]
    fn num_col(&self) -> usize {
        self.n_cols
    }
}

impl Pool {
    /// Materialize the facade [`catboost_rs::Pool`], inheriting the
    /// `OwnedColumns::into_pool()` length / range validation (threat T-08-11). The
    /// owned columns are cloned so the `Pool` pyclass can be reused for both `fit`
    /// and `predict`.
    ///
    /// # Errors
    /// [`CatBoostValueError`] if the inherited length check fails (e.g. a label
    /// shorter than the feature matrix).
    pub(crate) fn to_pool(&self) -> PyResult<FacadePool> {
        self.owned
            .clone()
            .into_pool()
            .map_err(|e| CatBoostValueError::new_err(e.to_string()))
    }
}
