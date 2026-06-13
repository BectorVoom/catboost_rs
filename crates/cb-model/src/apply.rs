//! Pure-Rust CPU oblivious-tree apply path (MODEL-02).
//!
//! Turns the canonical [`crate::Model`] into predictions WITHOUT any GPU
//! toolchain: this file imports nothing from the backend / GPU compute crate
//! (MODEL-02 — the apply path runs on a machine with no GPU dependency present).
//! It is the substrate the prediction-type transforms ([`crate::predict`]), SHAP
//! / fstr (Plan 04), and the Builder facade (Plan 05) consume.
//!
//! # Source of truth (RESEARCH Pattern 2)
//!
//! - **Step A — binarize** (`quantization.h:130-138`): a float feature's bin index
//!   is the COUNT of borders `b` for which `rawValue > b` (STRICT `>`). This is the
//!   exact `cb-data` / Phase-2 binarization semantics
//!   (`bin = borders.filter(|b| raw > b).count()`).
//! - **Step B — leaf index** (`evaluator_impl.cpp:26-50`): forward bit order over
//!   the tree's ordered splits — split `i` contributes bit `i`. Each split is a
//!   `value > border` test on one float feature; the per-split `passes` bool feeds
//!   `cb_train::leaf_index` (the SAME forward-bit-order evaluator the trainer
//!   used, so the leaf ordering matches `model.json`).
//! - **Step C — accumulate + bias** (`evaluator_impl.cpp:155-172`,
//!   `eval_processing.h:179 ApplyScaleAndBias`): `raw = bias + Σ_trees
//!   leaf_values[leaf]`. Leaf values are ALREADY `learning_rate`-scaled (Phase-3
//!   finding) — add them directly, never re-scale. The model `bias` is added
//!   EXACTLY ONCE (single source of truth, RESEARCH Pitfall 6): leaf values are
//!   bias-free, so a model with bias `B` and tree-sum `S` predicts `B + S`.
//!
//! # Parity discipline
//!
//! The per-object leaf-sum routes through `cb_core::sum_f64` (D-08 — never a raw
//! `iter().sum()` / `fold(0.0, …)`). All leaf / border access is checked
//! `.get` (no panics, `indexing_slicing` deny — T-04-02-01).

use cb_core::sum_f64;
use cb_train::{leaf_index, Split};

use crate::Model;

/// The bin index of `raw` against ascending `borders`: the COUNT of borders the
/// value strictly exceeds (`raw > b`), Step A (`quantization.h:138`).
///
/// STRICT `>` is load-bearing — it reproduces upstream's `<`/`<=` border
/// semantics (the exact `cb-data` Phase-2 binarization). A value below every
/// border bins to `0`; a value above every border bins to `borders.len()`.
#[must_use]
pub fn binarize_feature(raw: f64, borders: &[f64]) -> usize {
    borders.iter().filter(|&&b| raw > b).count()
}

/// Whether an object passes one split: the split's `value > border` test on its
/// float feature (Step B). Out-of-range feature indices return `false`
/// defensively (the loaded model supplies valid indices) — checked `.get` only.
fn passes_split(split: &Split, features: &[f32]) -> bool {
    features
        .get(split.feature)
        .is_some_and(|&v| f64::from(v) > split.border)
}

/// Apply every oblivious tree to one object and accumulate `bias + Σ_trees
/// leaf_values[leaf]` (Steps B + C). The per-tree leaf contributions are summed
/// host-side through [`sum_f64`] (D-08); `bias` is added exactly once afterward
/// (RESEARCH Pitfall 6).
fn predict_raw_one(model: &Model, features: &[f32]) -> f64 {
    let leaf_contributions: Vec<f64> = model
        .oblivious_trees
        .iter()
        .map(|tree| {
            // Step B: forward-bit-order leaf index over this tree's splits.
            let passes: Vec<bool> = tree
                .splits
                .iter()
                .map(|s| passes_split(s, features))
                .collect();
            let leaf = leaf_index(&passes);
            // Step C (per tree): the leaf value (already learning_rate-scaled).
            // Out-of-range leaf indices contribute 0.0 (T-04-02-01 checked access).
            tree.leaf_values.get(leaf).copied().unwrap_or(0.0)
        })
        .collect();
    // Σ_trees via the order-locked sum (D-08), THEN + bias exactly once.
    model.bias + sum_f64(&leaf_contributions)
}

/// Apply `model` to a numeric feature view, returning the per-object
/// `RawFormulaVal` (the raw logit / score), in object order (MODEL-02).
///
/// `feature_values[f]` is float feature `f`'s per-object `f32` column (the SoA
/// layout the trainer and the Plan-05 Builder feed). The number of objects is
/// taken from the FIRST column; an empty model (no float features) yields one
/// `bias`-only prediction per object only when a column count can be determined,
/// otherwise an empty vector.
///
/// This is the pure-Rust apply path: it touches NO backend / GPU compute
/// symbol, so it runs with no GPU toolchain present (MODEL-02).
#[must_use]
pub fn predict_raw(model: &Model, feature_values: &[Vec<f32>]) -> Vec<f64> {
    let n_objects = feature_values.first().map_or(0, Vec::len);
    (0..n_objects)
        .map(|obj| {
            // Gather this object's per-feature values into a contiguous row
            // (checked `.get`; a short column contributes nothing for that
            // feature, so its split test reads a missing value -> `false`).
            let row: Vec<f32> = feature_values
                .iter()
                .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
                .collect();
            predict_raw_one(model, &row)
        })
        .collect()
}
