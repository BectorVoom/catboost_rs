//! User-supplied custom objective / metric trait pair (LOSS-07, Rust half;
//! D-6.4-05). A Rust caller plugs a `CustomObjective` (per-object first/second
//! derivative) and/or a `CustomMetric` (eval) into the SAME `Loss`/`EvalMetric`
//! dispatch the built-in losses ride (D-6.2-03 — one code path), via an
//! `Arc<dyn>` newtype seam.
//!
//! # Mirror of the Python callback contract (D-09 — PyO3 DEFERRED to Phase 8)
//!
//! The trait signatures intentionally mirror the catboost Python callback
//! contract (`core.py:4867`):
//! - `CustomObjective::calc_ders_range(approxes, targets, weights) -> [(der1,
//!   der2)]` ← Python `calc_ders_range`.
//! - `CustomMetric::evaluate(approxes, target, weight) -> (error_sum,
//!   weight_sum)`, `get_final_error(error, weight)`, `is_max_optimal()` ←
//!   Python `CustomMetric.evaluate` / `get_final_error` / `is_max_optimal`.
//!
//! so the Phase-8 PyO3 bridge is a THIN adapter wrapping the SAME trait. No PyO3
//! / `pyo3` dependency is added here (D-09).
//!
//! # The `dyn`-in-derived-enum mechanism (06.4-RESEARCH Strand 3)
//!
//! [`Loss`](crate::Loss) derives `Debug + Clone + PartialEq` and `EvalMetric`
//! (in `cb-train`) historically derived `Copy`. A bare `Box<dyn ...>` field
//! breaks all of those (`dyn Trait` is neither `Clone` nor `PartialEq` nor
//! `Copy`). The solution is an **`Arc<dyn>` newtype** with a manual `Clone`
//! (cheap `Arc` refcount bump), a manual `Debug` (`"<dyn>"`), and a manual
//! pointer-identity `PartialEq` (`Arc::ptr_eq` — two distinct trait objects are
//! never "equal", which is the only sound equality for an opaque closure). `Arc`
//! (not `Box`) is required so the same objective instance is shared cheaply
//! across the train loop's `.clone()` / by-value call sites.
//!
//! # No-panic / parity discipline (CLAUDE.md, D-08)
//!
//! Every trait method returns [`CbResult`](cb_core::CbResult): a user objective
//! signals failure with a typed error, never an `unwrap`/`panic!`/`anyhow`
//! (T-06.4D-01 — a panicking objective cannot poison the train loop). The
//! consumer path (`compute_gradients` / `EvalMetric::eval`) REJECTS non-finite
//! ders/errors (T-06.4D-02 — Tampering: NaN/Inf must not reach leaf estimation).

use std::sync::Arc;

use cb_core::CbResult;

/// A user-supplied training objective: per-object first and second derivatives
/// of a scalar loss, mirroring the catboost Python `calc_ders_range`.
///
/// The implementor fills `ders[i] = (der1_i, der2_i)` for every object `i`,
/// where `der1` is the gradient and `der2` the (diagonal) hessian — the SAME
/// `(der1, der2)` convention as the built-in losses in [`crate::loss`] (e.g.
/// RMSE `der1 = target - approx`, `der2 = -1`; Logloss `der1 = target -
/// sigmoid(approx)`, `der2 = -p*(1-p)`). The rest of the pipeline (leaf
/// estimation, tree search) is loss-agnostic and consumes the resulting
/// derivative buffer unchanged. The leaf-estimation method is the
/// CALLER-SELECTED [`crate::runtime`] leaf method (the Builder default is
/// `Gradient`, which uses only der1); the user-supplied der2 is consumed ONLY
/// when the caller selects the Newton leaf method. Provide der2 if you intend to
/// train with Newton; under Gradient the der2 column is ignored (WR-02).
///
/// `Send + Sync` is required so the handle can live inside a `Loss` shared
/// across the (single-threaded in this phase, thread-safe by contract) train
/// loop and a future GPU runtime.
pub trait CustomObjective: Send + Sync {
    /// Fill `ders[i] = (der1, der2)` for each object `i` from its raw
    /// approximant `approxes[i]` and label `targets[i]`.
    ///
    /// `weights` is either empty (every weight is the upstream-convention
    /// `1.0`) or per-object (`weights.len() == approxes.len()`). The implementor
    /// applies the weight itself if it needs a weighted derivative (the built-in
    /// losses apply the per-object weight AFTER the unweighted scalar der; a
    /// custom objective may follow the same convention).
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if the input lengths disagree, or for any
    /// objective-specific precondition violation. The implementor MUST NOT
    /// `panic!`/`unwrap` (T-06.4D-01); the consumer additionally rejects any
    /// non-finite der it produces (T-06.4D-02).
    fn calc_ders_range(
        &self,
        approxes: &[f64],
        targets: &[f64],
        weights: &[f64],
        ders: &mut [(f64, f64)],
    ) -> CbResult<()>;
}

/// A user-supplied evaluation metric, mirroring the catboost Python
/// `CustomMetric` callback (`evaluate` / `get_final_error` / `is_max_optimal`).
///
/// [`evaluate`](CustomMetric::evaluate) returns the accumulated `(error_sum,
/// weight_sum)` numerator/denominator over the eval set;
/// [`get_final_error`](CustomMetric::get_final_error) reduces them to the final
/// scalar (e.g. `error / weight`); [`is_max_optimal`](CustomMetric::is_max_optimal)
/// gives the optimization direction (`true` = larger is better, like AUC;
/// `false` = smaller is better, like Logloss/RMSE).
pub trait CustomMetric: Send + Sync {
    /// Accumulate `(error_sum, weight_sum)` over the eval set's raw approximants.
    ///
    /// `weight` is empty (uniform `1.0`) or per-object. The caller divides
    /// `error_sum` by `weight_sum` via [`get_final_error`](Self::get_final_error)
    /// after rejecting a non-finite / non-positive `weight_sum`.
    ///
    /// # Errors
    /// Returns a [`cb_core::CbError`] if the input lengths disagree or a
    /// metric-specific precondition fails. No `panic!`/`unwrap` (T-06.4D-01).
    fn evaluate(&self, approxes: &[f64], target: &[f64], weight: &[f64]) -> CbResult<(f64, f64)>;

    /// Reduce the accumulated `(error, weight)` to the final metric value (e.g.
    /// `error / weight`, or `sqrt(error / weight)` for an RMSE-style metric).
    fn get_final_error(&self, error: f64, weight: f64) -> f64;

    /// `true` if a LARGER value of this metric is better (e.g. AUC), `false` if
    /// SMALLER is better (e.g. Logloss / RMSE). Drives the overfitting detector
    /// / best-model direction.
    fn is_max_optimal(&self) -> bool;
}

/// An `Arc<dyn CustomObjective>` newtype that is `Clone + Debug + PartialEq`, so
/// it can be a field of the derived [`Loss`](crate::Loss) enum (06.4-RESEARCH
/// Strand 3). Equality is `Arc::ptr_eq` pointer identity — two distinct
/// objective instances are never equal (the only sound equality for an opaque
/// trait object). `Clone` is a cheap `Arc` refcount bump; `Debug` prints
/// `"CustomObjectiveHandle(<dyn>)"`.
#[derive(Clone)]
pub struct CustomObjectiveHandle(pub Arc<dyn CustomObjective>);

impl CustomObjectiveHandle {
    /// Wrap a custom objective in a shareable handle.
    #[must_use]
    pub fn new(objective: Arc<dyn CustomObjective>) -> Self {
        Self(objective)
    }
}

impl PartialEq for CustomObjectiveHandle {
    /// Pointer-identity equality (`Arc::ptr_eq`): two distinct trait objects are
    /// never equal. There is no value-equality for an opaque closure, so this is
    /// the only sound `PartialEq` — it lets [`Loss`](crate::Loss) keep its
    /// `#[derive(PartialEq)]`.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for CustomObjectiveHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CustomObjectiveHandle(<dyn>)")
    }
}

/// An `Arc<dyn CustomMetric>` newtype that is `Clone + Debug + PartialEq` (same
/// `Arc::ptr_eq` mechanism as [`CustomObjectiveHandle`]), so it can be a field
/// of the `EvalMetric` enum once `EvalMetric` drops `Copy` (Arc is not `Copy`;
/// 06.4-RESEARCH Pitfall 7).
#[derive(Clone)]
pub struct CustomMetricHandle(pub Arc<dyn CustomMetric>);

impl CustomMetricHandle {
    /// Wrap a custom metric in a shareable handle.
    #[must_use]
    pub fn new(metric: Arc<dyn CustomMetric>) -> Self {
        Self(metric)
    }
}

impl PartialEq for CustomMetricHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for CustomMetricHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CustomMetricHandle(<dyn>)")
    }
}
