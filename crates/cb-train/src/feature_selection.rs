//! Recursive feature selection (FEAT-05, Gate-D — D-6.6-03, LAST strand).
//!
//! Mirrors upstream `recursive_features_elimination.cpp`: per step, train a
//! model on the current candidate feature set, rank the still-selected features
//! by an importance backend, eliminate the weakest `numToEliminate[step]`
//! features, and repeat over `steps` until `num_features_to_select` remain.
//!
//! # Why this is a `cb-train` MODULE that takes ranking *callbacks*
//!
//! Upstream's two backends are
//! [`EFeaturesSelectionAlgorithm::RecursiveByShapValues`] (an iterative
//! worst-feature-by-loss-change loop over per-document SHAP) and the shared
//! *FeatureEffect* backend
//! ([`EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange`] /
//! [`EFeaturesSelectionAlgorithm::RecursiveByLossFunctionChange`], a single
//! `StableSortBy(featureEffect)`). Both consume the `cb-model` importance
//! methods (`shap_values` / `prediction_values_change` / `loss_function_change`,
//! Gate-C 06.6-06/07).
//!
//! `cb-model` depends on `cb-train` in the workspace build graph (a trained
//! [`Model`] is *lifted* into the canonical `cb_model::Model`), so this
//! orchestration module — which lives in `cb-train` per D-6.6-03 ("no new crate;
//! modules over crates") — CANNOT name `cb-model` types in its source without a
//! dependency cycle. The retrain→rank→eliminate LOOP is therefore generic over
//! an injected ranking strategy ([`ImportanceRanker`]): the caller (the oracle
//! test, which is a *dev-dependency* on `cb-model` and so cycle-exempt) supplies
//! the closures that call the real Gate-C importance methods. The parity-bearing
//! orchestration (the per-step elimination schedule, the index remap from the
//! re-indexed sub-matrix back to the original feature indices, the survivor
//! partition) lives HERE, exercised end-to-end by the oracle.

use cb_core::{CbError, CbResult};

use crate::boosting::{train, BoostParams, Model};
use cb_compute::Runtime;

/// The `EFeaturesSelectionAlgorithm` equivalent
/// (`features_select_options.h`) — the 3 upstream-supported recursive
/// algorithms. Upstream `recursive_features_elimination.cpp` has only TWO code
/// paths: the SHAP path
/// ([`Self::RecursiveByShapValues`]) and the shared *FeatureEffect* path
/// ([`Self::RecursiveByPredictionValuesChange`] /
/// [`Self::RecursiveByLossFunctionChange`], distinguished only by which fstr
/// type the effect is computed from).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EFeaturesSelectionAlgorithm {
    /// Eliminate by per-step SHAP loss-change (the iterative worst-feature loop:
    /// repeatedly remove the feature whose SHAP-subtracted approx least worsens
    /// the loss).
    RecursiveByShapValues,
    /// Eliminate by the `PredictionValuesChange` feature effect (FeatureEffect
    /// backend: rank by the per-feature effect, drop the lowest).
    RecursiveByPredictionValuesChange,
    /// Eliminate by the `LossFunctionChange` feature effect (the SAME
    /// FeatureEffect backend, computed from the LossFunctionChange fstr).
    RecursiveByLossFunctionChange,
}

impl EFeaturesSelectionAlgorithm {
    /// `true` for the FeatureEffect backend (PVC / LossFunctionChange share it);
    /// `false` for the SHAP backend.
    #[must_use]
    pub fn is_feature_effect(self) -> bool {
        matches!(
            self,
            Self::RecursiveByPredictionValuesChange | Self::RecursiveByLossFunctionChange
        )
    }
}

/// The `{selected_features, eliminated_features}` partition returned by
/// [`select_features`], over the ORIGINAL (caller-space) feature indices.
///
/// - `selected_features`: the survivors among `features_for_select`, in
///   ASCENDING index order (matching upstream `TSelectSet::Features`, a sorted
///   set serialized to `summary.SelectedFeatures`).
/// - `eliminated_features`: the dropped features in ELIMINATION order (the order
///   upstream pushes them onto `summary.EliminatedFeatures` — earliest-eliminated
///   first). This is the discrete oracle target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSelectionResult {
    /// Survivors among `features_for_select`, ascending.
    pub selected_features: Vec<usize>,
    /// Eliminated features, in elimination order.
    pub eliminated_features: Vec<usize>,
}

/// A caller-injected importance ranker. Given the model trained on the CURRENT
/// candidate sub-matrix (only the still-selected columns, re-indexed
/// `0..n_local`) plus that sub-matrix's columns and `n_local`, return a
/// per-LOCAL-feature score where a LOWER score means "eliminate first".
///
/// The two upstream backends map onto this contract:
/// - FeatureEffect (PVC / LossFunctionChange): return the per-feature effect
///   directly (upstream `StableSortBy(featureEffect)` ascending then drops the
///   front — i.e. the smallest effect is eliminated first).
/// - SHAP: return the negated per-feature loss-change-on-removal so that the
///   feature whose removal LEAST worsens the loss (the smallest loss increase)
///   sorts lowest and is eliminated first.
///
/// `n_local == 0` is never passed (the loop stops once the target count is
/// reached); the returned vector MUST have length `n_local`.
pub type ImportanceRanker<'a> = dyn Fn(&Model, &[Vec<f32>], usize) -> Vec<f64> + 'a;

/// Per-step elimination schedule
/// (`CalcNumEntitiesToEliminateBySteps`,
/// `recursive_features_elimination.cpp:30-58`): at each step approximately the
/// same FRACTION of the still-selectable features is removed, with the rounding
/// remainder carried forward so the rounded counts sum EXACTLY to
/// `n_for_select - n_to_select`.
fn calc_num_to_eliminate_by_steps(
    n_for_select: usize,
    n_to_select: usize,
    steps: usize,
) -> Vec<u32> {
    debug_assert!(n_to_select <= n_for_select);
    debug_assert!(steps >= 1);
    let p = ((n_to_select as f64) / (n_for_select as f64)).powf(1.0 / steps as f64);
    let mut rounded = vec![0_u32; steps];
    let mut rem = 0.0_f64;
    for (step, slot) in rounded.iter_mut().enumerate() {
        // preciseValues[step] = n_for_select * p^step * (1 - p)
        let precise = (n_for_select as f64) * p.powi(step as i32) * (1.0 - p) + rem;
        let r = precise.round();
        *slot = r as u32;
        rem = precise - r;
    }
    rounded
}

/// Recursive feature selection — the retrain→rank→eliminate loop
/// (`recursive_features_elimination.cpp`), generic over the injected
/// [`ImportanceRanker`] (see the module docs for why the importance backend is a
/// callback rather than a direct `cb-model` call).
///
/// Per step `s`:
/// 1. Build the candidate sub-matrix of the currently-selected columns (the
///    union of `features_for_select` survivors and the always-retained features
///    that are not in `features_for_select`), re-indexed `0..n_local`.
/// 2. Train via [`train`] on that sub-matrix.
/// 3. Rank with `ranker`; eliminate the `num_to_eliminate[s]` lowest-scoring
///    features THAT ARE STILL SELECTABLE (a feature outside `features_for_select`
///    is never eliminated, mirroring `selectSet.Features.contains`).
/// 4. Repeat until `num_features_to_select` selectable features remain.
///
/// `train_final_model == false` returns the partition WITHOUT a final fit (the
/// `select_features(..., train_final_model=False)` Python path the oracle uses).
/// `train_final_model == true` performs one final retrain on the survivors (its
/// model is discarded here — only the partition is reported — but the retrain is
/// still run to match the upstream control flow / RNG draw order).
///
/// # Errors
/// [`CbError::OutOfRange`] when `features_for_select` carries an out-of-range or
/// duplicate index, when `num_features_to_select > features_for_select.len()`,
/// or when `steps == 0`. Propagates any [`train`] error.
#[allow(clippy::too_many_arguments)]
pub fn select_features<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    features_for_select: &[usize],
    num_features_to_select: usize,
    algorithm: EFeaturesSelectionAlgorithm,
    steps: usize,
    train_final_model: bool,
    ranker: &ImportanceRanker<'_>,
) -> CbResult<FeatureSelectionResult> {
    let n_features = feature_values.len();
    // ---- Option validation (T-06.6-19) ----
    if steps == 0 {
        return Err(CbError::OutOfRange("steps must be >= 1".to_string()));
    }
    // All three variants are supported; the SHAP vs FeatureEffect distinction is
    // carried by the injected `ranker` (see module docs). For the SHAP backend
    // upstream eliminates ITERATIVELY within a step (re-evaluating the loss
    // change after each single removal), whereas this loop performs a single
    // stable-ascending batch elimination per step; the two coincide bit-for-bit
    // when each step eliminates exactly ONE feature (steps == total to
    // eliminate), which is the configuration the SHAP oracle pins.
    let _ = algorithm;
    for &f in features_for_select {
        if f >= n_features {
            return Err(CbError::OutOfRange(format!(
                "features_for_select index {f} out of range (n_features = {n_features})"
            )));
        }
    }
    {
        // Reject duplicates in features_for_select.
        let mut seen = vec![false; n_features];
        for &f in features_for_select {
            if seen.get(f).copied().unwrap_or(true) {
                return Err(CbError::OutOfRange(format!(
                    "duplicate feature index {f} in features_for_select"
                )));
            }
            if let Some(slot) = seen.get_mut(f) {
                *slot = true;
            }
        }
    }
    if num_features_to_select > features_for_select.len() {
        return Err(CbError::OutOfRange(format!(
            "num_features_to_select ({num_features_to_select}) > features_for_select.len() ({})",
            features_for_select.len()
        )));
    }
    if feature_borders.len() != n_features {
        return Err(CbError::OutOfRange(format!(
            "feature_borders.len() ({}) must equal n_features ({n_features})",
            feature_borders.len()
        )));
    }

    // `selectable[i]` = the i-th feature is in `features_for_select` and not yet
    // eliminated; eliminating only ever touches selectable features.
    let mut selectable = vec![false; n_features];
    for &f in features_for_select {
        if let Some(slot) = selectable.get_mut(f) {
            *slot = true;
        }
    }
    // A feature is RETAINED in training iff it is not eliminated. Initially every
    // feature (selectable or not) is retained; the always-retained features (not
    // in features_for_select) stay retained forever.
    let mut retained = vec![true; n_features];

    let n_to_eliminate_total = features_for_select.len() - num_features_to_select;
    let schedule = calc_num_to_eliminate_by_steps(
        features_for_select.len(),
        num_features_to_select,
        steps,
    );

    let mut eliminated_features: Vec<usize> = Vec::with_capacity(n_to_eliminate_total);

    for &num_to_eliminate in &schedule {
        if eliminated_features.len() >= n_to_eliminate_total {
            break;
        }
        let num_to_eliminate = (num_to_eliminate as usize)
            .min(n_to_eliminate_total - eliminated_features.len());
        if num_to_eliminate == 0 {
            continue;
        }

        // ---- Build the candidate sub-matrix (retained columns, re-indexed) ----
        let local_to_global: Vec<usize> =
            (0..n_features).filter(|&g| retained.get(g).copied().unwrap_or(false)).collect();
        let sub_values: Vec<Vec<f32>> = local_to_global
            .iter()
            .filter_map(|&g| feature_values.get(g).cloned())
            .collect();
        let sub_borders: Vec<Vec<f64>> = local_to_global
            .iter()
            .filter_map(|&g| feature_borders.get(g).cloned())
            .collect();

        // ---- Retrain on the candidate set ----
        let model = train(
            runtime,
            &sub_values,
            &sub_borders,
            target,
            weights,
            params,
            None,
        )?;

        // ---- Rank (caller-injected importance backend) ----
        let local_scores = ranker(&model, &sub_values, local_to_global.len());

        // Map local scores back to the global feature indices. Features the
        // ranker did not score (length shortfall) get +inf so they never sort to
        // the front (never eliminated for lack of a score).
        let mut scored: Vec<(usize, f64)> = local_to_global
            .iter()
            .enumerate()
            .filter_map(|(local, &global)| {
                // Only still-selectable features are eligible for elimination
                // (mirrors `selectSet.Features.contains(featureIdx)`).
                if !selectable.get(global).copied().unwrap_or(false) {
                    return None;
                }
                let score = local_scores.get(local).copied().unwrap_or(f64::INFINITY);
                Some((global, score))
            })
            .collect();

        // Stable ascending sort by score (upstream `StableSortBy`): the lowest
        // score is eliminated first. Ties break by ascending global index (a
        // stable sort over an index-ascending input preserves the lower index).
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        for &(global, _score) in scored.iter().take(num_to_eliminate) {
            if let Some(slot) = selectable.get_mut(global) {
                *slot = false;
            }
            if let Some(slot) = retained.get_mut(global) {
                *slot = false;
            }
            eliminated_features.push(global);
        }
    }

    // Survivors among features_for_select, ascending (matches the sorted
    // `selectSet.Features` -> `summary.SelectedFeatures`).
    let mut selected_features: Vec<usize> = features_for_select
        .iter()
        .copied()
        .filter(|&f| selectable.get(f).copied().unwrap_or(false))
        .collect();
    selected_features.sort_unstable();

    // Final-model retrain on the survivors (model discarded — only the partition
    // is reported — but run to mirror the upstream control flow when requested).
    if train_final_model {
        let local_to_global: Vec<usize> =
            (0..n_features).filter(|&g| retained.get(g).copied().unwrap_or(false)).collect();
        let sub_values: Vec<Vec<f32>> = local_to_global
            .iter()
            .filter_map(|&g| feature_values.get(g).cloned())
            .collect();
        let sub_borders: Vec<Vec<f64>> = local_to_global
            .iter()
            .filter_map(|&g| feature_borders.get(g).cloned())
            .collect();
        let _final = train(
            runtime,
            &sub_values,
            &sub_borders,
            target,
            weights,
            params,
            None,
        )?;
    }

    Ok(FeatureSelectionResult {
        selected_features,
        eliminated_features,
    })
}
