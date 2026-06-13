//! Auto class weights (Balanced / SqrtBalanced) and per-object / per-class
//! weight resolution (DATA-08).
//!
//! A bit-faithful port of upstream CatBoost's
//! `catboost-master/catboost/private/libs/target/calc_class_weights.cpp`.
//!
//! **Summary class weight** (`CalculateSummaryClassWeight`, :29-55): for each
//! class `c`, sum the per-object weights of every object in class `c`. Upstream
//! accumulates these in a `double` per block; under `thread_count == 1` there is
//! exactly one block, so it degenerates to one sequential `f64` fold. Every such
//! sum routes through the sanctioned [`cb_core::sum_f64`] primitive (D-07 /
//! D-08), never a raw summation form. The `double` total is stored back as a
//! `float`.
//!
//! **Weight function** (`GetWeightFunction`, :11-27): with
//! `MINIMAL_CLASS_WEIGHT = 1e-8` (:9), Balanced returns
//! `w > 1e-8 ? max / w : 1.0` and SqrtBalanced returns
//! `w > 1e-8 ? sqrt(max / w) : 1.0`, where `max` is the largest summary class
//! weight and the arithmetic is `f32` (upstream's lambdas take/return `float`).
//! The `1e-8` floor guards the degenerate / empty class: a class with
//! (near-)zero summary weight yields `1.0` rather than dividing by zero — no
//! panic, no `inf`.
//!
//! No `unwrap` / `expect` / `panic` / `[]`-indexing appears in this module
//! (Shared Pattern C); fallible paths return [`cb_core::CbResult`].

use cb_core::{sum_f64, CbError, CbResult};

/// Upstream `MINIMAL_CLASS_WEIGHT` (`calc_class_weights.cpp:9`): the floor below
/// which a class is treated as empty and assigned weight `1.0`.
pub const MINIMAL_CLASS_WEIGHT: f32 = 1e-8;

/// Per-class summary weight: for each class `c`, the `f64` sum of the per-object
/// weights of every object assigned to `c` (`CalculateSummaryClassWeight`).
///
/// `target_classes[i]` is object `i`'s class index; `item_weights[i]` is its
/// weight (pass an all-ones slice for the unweighted case). The per-class sums
/// are accumulated through [`cb_core::sum_f64`] to honor D-07 / D-08.
///
/// # Errors
///
/// - [`CbError::OutOfRange`] if `class_count == 0`, if `target_classes` and
///   `item_weights` differ in length, or if any class index is `>= class_count`.
pub fn summary_class_weights(
    target_classes: &[usize],
    item_weights: &[f64],
    class_count: usize,
) -> CbResult<Vec<f64>> {
    if class_count == 0 {
        return Err(CbError::OutOfRange("class_count must be > 0".to_string()));
    }
    if target_classes.len() != item_weights.len() {
        return Err(CbError::LengthMismatch {
            column: "item_weights".to_string(),
            expected: target_classes.len(),
            actual: item_weights.len(),
        });
    }

    // Bucket each object's weight under its class, preserving object order so the
    // per-class fold matches upstream's single-block sequential accumulation.
    let mut per_class: Vec<Vec<f64>> = vec![Vec::new(); class_count];
    for (class, &weight) in target_classes.iter().zip(item_weights.iter()) {
        match per_class.get_mut(*class) {
            Some(bucket) => bucket.push(weight),
            None => {
                return Err(CbError::OutOfRange(format!(
                    "class index {class} out of range for class_count {class_count}"
                )))
            }
        }
    }

    Ok(per_class
        .iter()
        .map(|bucket| sum_f64(bucket))
        .collect::<Vec<f64>>())
}

/// The largest summary class weight, as `f32` (upstream's `maxSummaryClassWeight`
/// is a `float`). Returns `0.0` for an empty slice (no class).
fn max_summary_weight_f32(summary: &[f64]) -> f32 {
    summary
        .iter()
        .map(|&w| w as f32)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0)
}

/// Balanced auto class weights (`GetWeightFunction::Balanced`): per class `c`,
/// `w_c > 1e-8 ? max / w_c : 1.0`, computed in `f32`.
///
/// # Errors
///
/// Propagates [`summary_class_weights`] errors.
pub fn balanced_class_weights(
    target_classes: &[usize],
    item_weights: &[f64],
    class_count: usize,
) -> CbResult<Vec<f32>> {
    let summary = summary_class_weights(target_classes, item_weights, class_count)?;
    let max = max_summary_weight_f32(&summary);
    Ok(summary
        .iter()
        .map(|&w| {
            let w = w as f32;
            if w > MINIMAL_CLASS_WEIGHT {
                max / w
            } else {
                1.0
            }
        })
        .collect())
}

/// SqrtBalanced auto class weights (`GetWeightFunction::SqrtBalanced`): per class
/// `c`, `w_c > 1e-8 ? sqrt(max / w_c) : 1.0`, computed in `f32` (`sqrtf`).
///
/// # Errors
///
/// Propagates [`summary_class_weights`] errors.
pub fn sqrt_balanced_class_weights(
    target_classes: &[usize],
    item_weights: &[f64],
    class_count: usize,
) -> CbResult<Vec<f32>> {
    let summary = summary_class_weights(target_classes, item_weights, class_count)?;
    let max = max_summary_weight_f32(&summary);
    Ok(summary
        .iter()
        .map(|&w| {
            let w = w as f32;
            if w > MINIMAL_CLASS_WEIGHT {
                (max / w).sqrt()
            } else {
                1.0
            }
        })
        .collect())
}

/// Resolve a per-object weight vector from per-class weights and explicit
/// per-object weights.
///
/// Upstream multiplies the (optional) explicit per-object weight by the class
/// weight of the object's class. `per_object_weights` may be empty (treated as
/// all-ones); `class_weights` may be empty (treated as all-ones — no class
/// reweighting). The result is one `f64` weight per object.
///
/// # Errors
///
/// - [`CbError::LengthMismatch`] if a non-empty `per_object_weights` differs in
///   length from `target_classes`.
/// - [`CbError::OutOfRange`] if a class index has no entry in a non-empty
///   `class_weights`.
pub fn resolve_object_weights(
    class_weights: &[f32],
    per_object_weights: &[f64],
    target_classes: &[usize],
) -> CbResult<Vec<f64>> {
    if !per_object_weights.is_empty() && per_object_weights.len() != target_classes.len() {
        return Err(CbError::LengthMismatch {
            column: "per_object_weights".to_string(),
            expected: target_classes.len(),
            actual: per_object_weights.len(),
        });
    }

    let mut resolved: Vec<f64> = Vec::with_capacity(target_classes.len());
    for (object_index, &class) in target_classes.iter().enumerate() {
        let base = per_object_weights.get(object_index).copied().unwrap_or(1.0);
        let class_weight = if class_weights.is_empty() {
            1.0_f64
        } else {
            match class_weights.get(class) {
                Some(&cw) => f64::from(cw),
                None => {
                    return Err(CbError::OutOfRange(format!(
                        "class index {class} out of range for {} class weights",
                        class_weights.len()
                    )))
                }
            }
        };
        resolved.push(base * class_weight);
    }
    Ok(resolved)
}
