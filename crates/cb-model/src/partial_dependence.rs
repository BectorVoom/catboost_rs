//! Partial dependence for one or two float features (FSTR-03 / PDP-01..05).
//!
//! # Upstream averaging formula
//!
//! For a target feature set `x_S` and complementary features `x_C`, partial
//! dependence is `f_{x_S}(x_S) = (1/n) · Σ_{i=1..n} f(x_S, x_C^{(i)})`: the
//! target feature(s) are pinned to a grid value while every other feature
//! keeps its actual per-object dataset value, the model is applied
//! (`RawFormulaVal` via `crate::predict_raw`), and the result is averaged
//! over all `n` dataset objects
//! `[VERIFIED: CONTEXT7 /catboost/catboost "partial dependence function
//! estimated by calculating averages of the model's predictions across the
//! training data"]`. A one-feature request is a 1-D curve over that feature's
//! grid; a two-feature request is a 2-D grid over the Cartesian product,
//! row-major (`features[0]` outer, `features[1]` inner)
//! `[VERIFIED: CONTEXT7 /catboost/catboost plot_partial_dependence]`.
//!
//! # Float-only scope (index space)
//!
//! `features` and `columns` are both indexed in FLOAT-feature-index space:
//! `f ∈ 0..n_float` where `n_float = model.float_feature_borders.len()`, and
//! `columns[f]` is that float feature's per-object `f32` column (the exact SoA
//! layout `crate::predict_raw` consumes). `cb_model::Model` stores no
//! flat-feature / feature-kind map, so this is the only index space the API
//! can honor; categorical / CTR / text / embedding target features are out of
//! scope for this slice (deferred follow-up) and are indistinguishable, from
//! `Model` metadata alone, from an out-of-range float index — both surface as
//! [`PdpError::FeatureIndexOutOfRange`]
//! `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313]`.
//!
//! # Typed rejections (PDP-05)
//!
//! Every invalid request is rejected with a typed [`PdpError`] variant, in a
//! deterministic check order (arity → column shape → empty dataset → feature
//! range → duplicate) — see `SPEC.md` §5. No panic, no `unwrap`/`expect`, no
//! raw indexing (workspace-denied restriction lints).
//!
//! # Grid derivation (PDP-02) — per-bin representatives
//!
//! Upstream computes partial dependence **per BIN**: a float feature with
//! borders `b_0 < … < b_{k-1}` (k = n_borders) has `k+1` bins
//! (`(-inf,b_0], (b_0,b_1], …, (b_{k-1},+inf)`) and `plot_partial_dependence`
//! returns one averaged value per bin (`_calc_partial_dependence`,
//! catboost 1.2.10 `core.py:4041`). Because a model's prediction depends only on
//! which bin the feature falls in, the representative value of bin `i` can be
//! any point interior to it; [`grid_for_feature`] uses
//! `[b_0 - 1, (b_0+b_1)/2, …, (b_{k-2}+b_{k-1})/2, b_{k-1} + 1]` — length `k+1`,
//! strictly ascending. Feeding `grid[i]` through the averaging engine reproduces
//! upstream bin `i` (empirically verified to <1e-15 on the numeric_tiny model;
//! oracle-locked ≤1e-5 by `partial_dependence_oracle_test.rs`). A feature the
//! model never split on (empty borders) is rejected upstream ("not used in
//! model"); here it yields a benign single-point grid rather than an error.

use crate::apply::predict_raw;
use crate::model::Model;
use cb_core::sum_f64;

/// Result of a partial-dependence computation for one or two features.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialDependence {
    /// The 1 or 2 target float-feature indices, in the order requested.
    pub features: Vec<usize>,
    /// Per-target-feature ascending grid values (`grids.len() == features.len()`).
    pub grids: Vec<Vec<f64>>,
    /// Averaged `RawFormulaVal`, row-major over the Cartesian product of `grids`.
    /// Single feature: `values.len() == grids[0].len()`.
    /// Two features:   `values.len() == grids[0].len() * grids[1].len()`,
    ///                 index `a*grids[1].len() + b` = `(grids[0][a], grids[1][b])`.
    pub values: Vec<f64>,
}

/// Typed failure at the partial-dependence boundary (no panic, no unwrap).
/// Every variant is reachable and implementable from [`Model`] + `columns`
/// alone (SPEC.md §4/§5).
#[derive(Debug, thiserror::Error)]
pub enum PdpError {
    /// A requested feature index is `>=` the model's float-feature count. Also
    /// the (only) outcome for a categorical/CTR target, which has no float
    /// index.
    #[error("feature index {index} out of range (model has {n_float} float features)")]
    FeatureIndexOutOfRange { index: usize, n_float: usize },
    /// `features.len()` is not 1 or 2 (upstream partial dependence is 1-2
    /// features).
    #[error("partial dependence supports 1 or 2 features, got {requested}")]
    UnsupportedFeatureArity { requested: usize },
    /// A two-feature request named the SAME float feature twice; a 2-D PD
    /// surface over `(f, f)` is degenerate. Conservative API precondition of
    /// this slice.
    #[error(
        "partial dependence over two features requires distinct features; feature {index} was given twice"
    )]
    DuplicateFeature { index: usize },
    /// `columns` does not match the model's float-feature layout: the count is
    /// not `n_float`, or the columns are ragged (unequal lengths). Guards the
    /// silent NaN-pad hazard (`predict_raw_cat` at apply.rs:404-407) — a
    /// short/narrow column would otherwise average garbage.
    #[error(
        "columns malformed: expected {expected_float_features} equal-length float columns, got {actual}"
    )]
    MalformedColumns {
        expected_float_features: usize,
        actual: String,
    },
    /// The dataset has no objects (`columns` empty, or every column length 0).
    #[error("dataset has no objects")]
    EmptyDataset,
}

/// PDP-05: validate `(model, columns, features)` against the SPEC §5
/// deterministic check order: (1) arity → (2) column shape → (3) empty
/// dataset → (4) per-feature range → (5) duplicate. Returns the common object
/// count `n_obj` on success — the single guard [`pdp_curve_single`] relies on
/// to never NaN-pad a short/missing column
/// `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407]`.
fn validate(model: &Model, columns: &[Vec<f32>], features: &[usize]) -> Result<usize, PdpError> {
    // (1) arity: 1 or 2 features only.
    if features.len() != 1 && features.len() != 2 {
        return Err(PdpError::UnsupportedFeatureArity {
            requested: features.len(),
        });
    }

    let n_float = model.float_feature_borders.len();

    // (2) column shape: exactly n_float columns, all equal length
    // (rectangular). `columns == []` falls in here too (0 != n_float for the
    // in-scope n_float >= 1), NOT EmptyDataset (SPEC §5 order).
    let first_len = columns.first().map(Vec::len);
    let is_rectangular = columns.iter().all(|c| Some(c.len()) == first_len);
    if columns.len() != n_float || !is_rectangular {
        let lengths: Vec<usize> = columns.iter().map(Vec::len).collect();
        let actual = format!("{} columns, lengths {lengths:?}", columns.len());
        return Err(PdpError::MalformedColumns {
            expected_float_features: n_float,
            actual,
        });
    }

    // (3) empty dataset: correct shape, but zero rows.
    let n_obj = first_len.unwrap_or(0);
    if n_obj == 0 {
        return Err(PdpError::EmptyDataset);
    }

    // (4) feature range: first offending index, in request order.
    for &f in features {
        if f >= n_float {
            return Err(PdpError::FeatureIndexOutOfRange { index: f, n_float });
        }
    }

    // (5) duplicate (2-feature only).
    if features.len() == 2 {
        if let (Some(&a), Some(&b)) = (features.first(), features.get(1)) {
            if a == b {
                return Err(PdpError::DuplicateFeature { index: a });
            }
        }
    }

    Ok(n_obj)
}

/// Shared per-grid-point PDP step, factored out of what was previously
/// hand-duplicated column-override + zero-object-guard + predict+average logic
/// (once in [`pdp_curve_single`], twice nested in [`pdp_curve_pair`]) — a future
/// fix to this logic (e.g. the NaN-pad guard this module's doc comment warns
/// about) now only needs to be applied once.
///
/// **Precondition (enforced by [`validate`] before either curve engine ever
/// runs in the public [`partial_dependence`] path):** `working.len() == n_float`,
/// every column has the same length `n_obj`, and every `overrides` feature index
/// is `< n_float` — because these hold, `predict_raw` never NaN-pads a
/// short/missing column
/// `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407]`.
///
/// Overwrites each `(feature, value)` pair's ENTIRE column in `working` (every
/// object gets the same pinned value — a no-op for an out-of-range `feature`,
/// same as the pre-refactor `if let Some(col) = ...` guard), then, unless
/// `n_obj == 0`, calls [`predict_raw`] and folds the mean through [`sum_f64`]
/// (D-08 — never a raw `.sum()`). `working` is the caller's REUSED working-column
/// buffer (mutated in place, not re-cloned per call — same allocation-reuse
/// discipline as before the refactor).
fn pdp_averaged_prediction(
    model: &Model,
    working: &mut [Vec<f32>],
    overrides: &[(usize, f64)],
    n_obj: usize,
) -> f64 {
    for &(feature, value) in overrides {
        if let Some(col) = working.get_mut(feature) {
            let v = value as f32;
            for slot in col.iter_mut() {
                *slot = v;
            }
        }
    }
    if n_obj == 0 {
        return 0.0;
    }
    let preds = predict_raw(model, working);
    sum_f64(&preds) / (n_obj as f64)
}

/// PDP-01: the single-feature averaging engine over an explicit grid.
///
/// For each grid point, forms a working column set identical to `columns`
/// except that the whole `feature` column is overridden to `grid[k] as f32`
/// (via [`pdp_averaged_prediction`]). Pure: the caller's `columns` are not
/// mutated (the override happens on an owned working copy, reused across grid
/// points). Output length is exactly `grid.len()`, in grid order.
fn pdp_curve_single(model: &Model, columns: &[Vec<f32>], feature: usize, grid: &[f64]) -> Vec<f64> {
    let n_obj = columns.first().map_or(0, Vec::len);
    // One reusable working buffer (T1 refactor): clone the column set once,
    // then overwrite only the target column's contents per grid point,
    // rather than re-cloning every column on every iteration.
    let mut working: Vec<Vec<f32>> = columns.to_vec();
    grid.iter()
        .map(|&v| pdp_averaged_prediction(model, &mut working, &[(feature, v)], n_obj))
        .collect()
}

/// PDP-02: derive the per-bin representative grid for a float `feature` from its
/// stored borders (see the module-level "Grid derivation" note). Length is
/// `n_borders + 1` (one representative per bin), strictly ascending. An empty
/// (never-split) feature yields the single-point grid `[0.0]`; an out-of-range
/// index (unreachable in the validated public path) yields an empty grid.
fn grid_for_feature(model: &Model, feature: usize) -> Vec<f64> {
    let borders = match model.float_feature_borders.get(feature) {
        Some(b) => b,
        None => return Vec::new(),
    };
    if borders.is_empty() {
        return vec![0.0];
    }
    let mut grid = Vec::with_capacity(borders.len() + 1);
    // Bin 0: any value <= b_0; use b_0 - 1.0 (strictly below the first border).
    if let Some(&first) = borders.first() {
        grid.push(first - 1.0);
    }
    // Interior bins: midpoint of consecutive borders.
    for pair in borders.windows(2) {
        if let (Some(&lo), Some(&hi)) = (pair.first(), pair.get(1)) {
            grid.push((lo + hi) / 2.0);
        }
    }
    // Last bin: any value > b_{k-1}; use b_{k-1} + 1.0.
    if let Some(&last) = borders.last() {
        grid.push(last + 1.0);
    }
    grid
}

/// PDP-04: the two-feature averaging engine over the Cartesian product of two
/// grids, row-major (`f1` outer, `f2` inner: index `a*g2.len()+b`). Same
/// validated precondition as [`pdp_curve_single`]; `f1 != f2` (validate rejects
/// duplicates). Overrides BOTH target columns per grid point (via
/// [`pdp_averaged_prediction`]); the mean folds through [`sum_f64`] (D-08).
fn pdp_curve_pair(
    model: &Model,
    columns: &[Vec<f32>],
    features: (usize, usize),
    grids: (&[f64], &[f64]),
) -> Vec<f64> {
    let (f1, f2) = features;
    let (g1, g2) = grids;
    let n_obj = columns.first().map_or(0, Vec::len);
    let mut working: Vec<Vec<f32>> = columns.to_vec();
    let mut out = Vec::with_capacity(g1.len().saturating_mul(g2.len()));
    for &a in g1 {
        for &b in g2 {
            out.push(pdp_averaged_prediction(model, &mut working, &[(f1, a), (f2, b)], n_obj));
        }
    }
    out
}

/// Public entry point (composes PDP-05 validation + PDP-02 grid + PDP-01/PDP-04
/// engines). `columns[f]` is float feature `f`'s per-object `f32` column (SoA,
/// as `predict_raw` consumes); `columns.len()` MUST equal
/// `model.float_feature_borders.len()` (validated). Values match upstream
/// `plot_partial_dependence` per-bin averages within `1e-5`.
///
/// # Errors
/// A typed [`PdpError`] for any input that fails the PDP-05 deterministic
/// check order.
pub fn partial_dependence(
    model: &Model,
    columns: &[Vec<f32>],
    features: &[usize],
) -> Result<PartialDependence, PdpError> {
    validate(model, columns, features)?;
    // Arity is 1 or 2 (guaranteed by `validate`); slice patterns avoid indexing.
    match features {
        [f] => {
            let grid = grid_for_feature(model, *f);
            let values = pdp_curve_single(model, columns, *f, &grid);
            Ok(PartialDependence {
                features: vec![*f],
                grids: vec![grid],
                values,
            })
        }
        [f1, f2] => {
            let g1 = grid_for_feature(model, *f1);
            let g2 = grid_for_feature(model, *f2);
            let values = pdp_curve_pair(model, columns, (*f1, *f2), (&g1, &g2));
            Ok(PartialDependence {
                features: vec![*f1, *f2],
                grids: vec![g1, g2],
                values,
            })
        }
        // Unreachable after `validate`'s arity check; keeps the match total
        // without a panic.
        _ => Err(PdpError::UnsupportedFeatureArity {
            requested: features.len(),
        }),
    }
}

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md —
// no test body in this production file). Mirrors ctr_data.rs:58-61.
#[cfg(test)]
#[path = "partial_dependence_test.rs"]
mod tests;
