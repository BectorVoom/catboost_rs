//! Weighted model merge (`sum_models`, SPEC `sum_models`): combine N compatible
//! float-only oblivious models into one by scaling each model's leaf
//! contributions (and bias) by its weight and concatenating the ensembles.
//!
//! First slice: N >= 1 oblivious, float-only (`ctr_data == None`) models with
//! identical `float_feature_borders`, `approx_dimension`, and `class_to_label`
//! — [`validate`] rejects any input violating these, so they are true, checked
//! preconditions. A standard trained/loaded float [`Model`] additionally has NO
//! separate leaf scale (apply is `bias + sum(leaf_values)`, no scale multiply)
//! — this "scale==1" property is therefore ASSUMED, not checked: `Model` has no
//! scale field to check, so an input that somehow violated it would be
//! UNDETECTABLE here (SPEC.md §2, §9 R1). See `.planning/plans/sum-models/SPEC.md`.

use cb_core::sum_f64;

use crate::error::ModelError;
use crate::model::{Model, ObliviousTree};

/// Scale every leaf value in `tree` by `w`, leaving `splits` and
/// `leaf_weights` unchanged (R3: `leaf_weights` carried unscaled — this
/// slice predicts only).
fn scaled_tree(tree: &ObliviousTree, w: f64) -> ObliviousTree {
    ObliviousTree {
        splits: tree.splits.clone(),
        leaf_values: tree.leaf_values.iter().map(|v| w * v).collect(),
        leaf_weights: tree.leaf_weights.clone(),
    }
}

/// Validate that `models`/`weights` are mergeable, returning the reference
/// (first) model on success. Checks, in order: non-empty `models`; a
/// non-empty `weights` whose length matches `models.len()`; every model is
/// oblivious-only (no non-symmetric / region trees) and float-only (no
/// `ctr_data`); every model agrees with the first on `float_feature_borders`,
/// `approx_dimension`, and `class_to_label`. Never panics — every access is
/// checked (`.first`/`.get`), no `[]` indexing.
fn validate<'a>(models: &[&'a Model], weights: &[f64]) -> Result<&'a Model, ModelError> {
    let Some(&first) = models.first() else {
        return Err(ModelError::Merge("no models to sum".to_owned()));
    };

    if !weights.is_empty() && weights.len() != models.len() {
        return Err(ModelError::Merge(format!(
            "weights.len() ({}) must equal models.len() ({}) when weights is non-empty",
            weights.len(),
            models.len()
        )));
    }

    for (i, &m) in models.iter().enumerate() {
        if !m.non_symmetric_trees.is_empty() || !m.region_trees.is_empty() {
            return Err(ModelError::Merge(format!(
                "model {i} is non-oblivious (non_symmetric_trees / region_trees non-empty); \
                 sum_models only supports oblivious models in this slice"
            )));
        }
        if m.ctr_data.is_some() {
            return Err(ModelError::Merge(format!(
                "model {i} carries ctr_data; sum_models only supports float-only models \
                 in this slice"
            )));
        }
        if m.float_feature_borders != first.float_feature_borders {
            return Err(ModelError::Merge(format!(
                "model {i} disagrees with model 0 on float_feature_borders"
            )));
        }
        if m.approx_dimension != first.approx_dimension {
            return Err(ModelError::Merge(format!(
                "model {i} disagrees with model 0 on approx_dimension \
                 ({} vs {})",
                m.approx_dimension, first.approx_dimension
            )));
        }
        if m.class_to_label != first.class_to_label {
            return Err(ModelError::Merge(format!(
                "model {i} disagrees with model 0 on class_to_label"
            )));
        }
    }

    Ok(first)
}

/// Combine `models` into one weighted-sum model. `weights[i]` scales model
/// `i`'s leaf contributions; when `weights` is empty, every model gets
/// weight `1.0`.
///
/// # Errors
/// [`ModelError::Merge`] if: `models` is empty; `weights` is non-empty and its
/// length != `models.len()`; any model is non-oblivious (`non_symmetric_trees`
/// / `region_trees` non-empty); any model carries `ctr_data` (`Some`); the
/// models disagree on `float_feature_borders`, `approx_dimension`, or
/// `class_to_label`.
pub fn sum_models(models: &[&Model], weights: &[f64]) -> Result<Model, ModelError> {
    let first = validate(models, weights)?;

    let effective_weights: Vec<f64> = if weights.is_empty() {
        vec![1.0; models.len()]
    } else {
        weights.to_vec()
    };

    let mut oblivious_trees: Vec<ObliviousTree> = Vec::new();
    let mut weighted_biases: Vec<f64> = Vec::new();
    for (m, &w) in models.iter().zip(effective_weights.iter()) {
        oblivious_trees.extend(m.oblivious_trees.iter().map(|t| scaled_tree(t, w)));
        weighted_biases.push(w * m.bias);
    }

    // `non_symmetric_trees`/`region_trees`/`ctr_data` are provably empty/`None`
    // here — `validate` above rejects any model where they are not — so they
    // are constructed fresh rather than cloned from `first`, which would
    // wrongly suggest they might carry real merged data.
    Ok(Model {
        oblivious_trees,
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: sum_f64(&weighted_biases),
        float_feature_borders: first.float_feature_borders.clone(),
        ctr_data: None,
        approx_dimension: first.approx_dimension,
        class_to_label: first.class_to_label.clone(),
    })
}

#[cfg(test)]
#[path = "model_sum_test.rs"]
mod tests;
