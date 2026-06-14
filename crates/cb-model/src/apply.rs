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
use cb_data::calc_cat_feature_hash;
use cb_train::{fold_cat_hash, leaf_index};

use crate::ctr_data::{calc_inference, CtrValueTable, Prior};
use crate::model::{CtrSplit, ModelSplit};
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

/// Compute one CTR value for a document's simple (single-feature) categorical
/// projection at inference (`static_ctr_provider.cpp:52-122`).
///
/// The projection categorical VALUE (already in the A4 string form,
/// [`cb_data::stringify_int_category`] for integer-coded values) is hashed via
/// [`cb_data::calc_cat_feature_hash`] — the single categorical-hash source,
/// NEVER the model's STORED `ctr_data` hash_map (RESEARCH Anti-Pattern). The
/// table's [`CtrValueTable::calc_for_hash`] then applies the per-type
/// `Calc(cic, tot)` with the `prior` / `(shift, scale)` normalization, replicating
/// the not-found→empty path (a missing bucket returns the empty value, never an
/// OOB index — T-05-04-01).
///
/// `target_border_idx` selects the Buckets per-class numerator (default `0`).
#[must_use]
pub fn ctr_value_for_projection(
    table: &CtrValueTable,
    cat_value: &str,
    prior: Prior,
    shift: f64,
    scale: f64,
    target_border_idx: usize,
) -> f64 {
    let hash = u64::from(calc_cat_feature_hash(cat_value));
    table.calc_for_hash(hash, prior, shift, scale, target_border_idx)
}

/// Compute one CTR value for a document's TENSOR / COMBINATION projection at
/// inference (ORD-05 / D-05). A tensor CTR is the SAME per-type
/// [`CtrValueTable::calc_for_hash`] apply as the single-feature path
/// ([`ctr_value_for_projection`]) — only the KEY changes: it is the COMBINED
/// projection hash folding each member feature's per-document categorical hash,
/// NOT a single feature's hash.
///
/// `cat_values` holds the document's projection-member categorical VALUES (each
/// already in the A4 string form, [`cb_data::stringify_int_category`] for
/// integer-coded values) IN THE PROJECTION'S SORTED MEMBER ORDER. Each is hashed
/// via [`cb_data::calc_cat_feature_hash`] — the single categorical-hash source,
/// NEVER the model's STORED `ctr_data` hash_map (RESEARCH Anti-Pattern) — and
/// folded into the combined ui64 key via [`cb_train::fold_cat_hash`]
/// (`ctr_provider.h:65-78` `CalcHash`, the sign-extended `(ui64)(int)` cast). The
/// table's [`CtrValueTable::calc_for_hash`] then applies the per-type
/// `Calc(cic, tot)` with the not-found→empty path (a missing combined bucket
/// returns the empty value, never an OOB index — T-05-06-V5). A single-element
/// `cat_values` degenerates to the simple-CTR combined key (one fold over the
/// empty `0` seed), so the simple and tensor paths share one keyspace.
///
/// `target_border_idx` selects the Buckets per-class numerator (default `0`).
#[must_use]
pub fn ctr_value_for_combined_projection(
    table: &CtrValueTable,
    cat_values: &[&str],
    prior: Prior,
    shift: f64,
    scale: f64,
    target_border_idx: usize,
) -> f64 {
    // Fold each member feature's per-document ui32 cat hash into the combined key,
    // starting from the 0 seed (ctr_provider.h:70 `ui64 result = 0`), in the
    // supplied (projection-sorted) member order.
    let mut combined: u64 = 0;
    for &cat_value in cat_values {
        combined = fold_cat_hash(combined, calc_cat_feature_hash(cat_value));
    }
    table.calc_for_hash(combined, prior, shift, scale, target_border_idx)
}

/// The CTR-base table key a [`CtrSplit`] reconstructs from its
/// `(projection, ctr_type)` — used to look up the baked [`CtrValueTable`] in the
/// model's `ctr_data` map. The key form is
/// `"ctr:type=<i8>:proj=<f0>,<f1>,…"` over the projection's SORTED cat-feature
/// members. The trainer-side bake uses the SAME canonical form so the apply-time
/// reconstruction matches the stored key byte-for-byte.
#[must_use]
fn ctr_table_key(split: &CtrSplit) -> String {
    let members: Vec<String> = split
        .projection
        .cat_features()
        .iter()
        .map(usize::to_string)
        .collect();
    format!("ctr:type={}:proj={}", split.ctr_type.as_i8(), members.join(","))
}

/// Whether an object passes one float split (`value > border`, Step B).
/// Out-of-range feature indices return `false` defensively (the loaded model
/// supplies valid indices) — checked `.get` only.
fn passes_float_split(feature: usize, border: f64, features: &[f32]) -> bool {
    features
        .get(feature)
        .is_some_and(|&v| f64::from(v) > border)
}

/// Whether an object passes one CTR split (`ctr_value > border`, ORD-05 / D-05).
///
/// The object's combined-projection CTR value is computed from its RAW cat values
/// (gathered in the projection's sorted member order from `cat_values`) by folding
/// each member's [`cb_data::calc_cat_feature_hash`] via [`cb_train::fold_cat_hash`]
/// into the combined key (NEVER the model's stored `ctr_data` hash_map — RESEARCH
/// Anti-Pattern), then looking up the baked [`CtrValueTable`] (keyed by
/// `(projection, ctr_type)`) with the not-found→empty bounds-safe path
/// ([`ctr_value_for_combined_projection`]). A missing table or a missing combined
/// bucket returns the empty CTR value ([`calc_inference`] over a `None` bucket),
/// never an OOB index (T-05-09-V5).
///
/// `cat_values[c]` is the object's raw categorical VALUE for cat feature `c`
/// (already in the A4 string form, [`cb_data::stringify_int_category`] for
/// integer-coded values); the projection members index into it.
fn passes_ctr_split(model: &Model, split: &CtrSplit, cat_values: &[String]) -> bool {
    // Gather the projection-member raw cat values in the projection's sorted
    // order (checked `.get`; a missing member is dropped — the fold then sees
    // fewer members, which the not-found→empty path tolerates).
    let members: Vec<&str> = split
        .projection
        .cat_features()
        .iter()
        .filter_map(|&f| cat_values.get(f).map(String::as_str))
        .collect();

    // Look up the baked table by the reconstructed (projection, ctr_type) key.
    let key = ctr_table_key(split);
    let ctr_value = match model.ctr_data.as_ref().and_then(|d| d.tables.get(&key)) {
        // The combined-projection CTR value: fold each member's
        // calc_cat_feature_hash via fold_cat_hash into the combined key, then the
        // per-type Calc over the baked table with the bounds-safe not-found→empty
        // bucket path (T-05-09-V5).
        Some(table) => ctr_value_for_combined_projection(
            table,
            &members,
            split.prior,
            /* shift = */ 0.0,
            /* scale = */ 1.0,
            split.target_border_idx,
        ),
        // No baked table for this split: the not-found→empty CTR value
        // (`Calc(0, 0)` over the prior). Bounds-safe — never an OOB index.
        None => calc_inference(0.0, 0.0, split.prior, 0.0, 1.0),
    };
    ctr_value > split.border
}

/// Whether an object passes one [`ModelSplit`] (Step B): a [`ModelSplit::Float`]
/// keeps the existing `value > border` path byte-for-byte; a [`ModelSplit::Ctr`]
/// evaluates the combined-projection CTR value against the split border
/// ([`passes_ctr_split`]).
fn passes_split(model: &Model, split: &ModelSplit, features: &[f32], cat_values: &[String]) -> bool {
    match split {
        ModelSplit::Float(s) => passes_float_split(s.feature, s.border, features),
        ModelSplit::Ctr(c) => passes_ctr_split(model, c, cat_values),
    }
}

/// Apply every oblivious tree to one object and accumulate `bias + Σ_trees
/// leaf_values[leaf]` (Steps B + C). The per-tree leaf contributions are summed
/// host-side through [`sum_f64`] (D-08); `bias` is added exactly once afterward
/// (RESEARCH Pitfall 6).
///
/// `features` is the object's contiguous float-feature row; `cat_values` its raw
/// categorical values (empty for the numeric-only apply path, where no CTR split
/// is ever evaluated — the `ModelSplit::Float` path is byte-for-byte unchanged).
fn predict_raw_one(model: &Model, features: &[f32], cat_values: &[String]) -> f64 {
    let leaf_contributions: Vec<f64> = model
        .oblivious_trees
        .iter()
        .map(|tree| {
            // Step B: forward-bit-order leaf index over this tree's splits.
            let passes: Vec<bool> = tree
                .splits
                .iter()
                .map(|s| passes_split(model, s, features, cat_values))
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
/// symbol, so it runs with no GPU toolchain present (MODEL-02). Models with CTR
/// splits need the categorical columns — use [`predict_raw_cat`] for those.
#[must_use]
pub fn predict_raw(model: &Model, feature_values: &[Vec<f32>]) -> Vec<f64> {
    predict_raw_cat(model, feature_values, &[])
}

/// Apply `model` to a numeric + categorical feature view (ORD-05 / D-05),
/// returning the per-object `RawFormulaVal` in object order.
///
/// `feature_values[f]` is float feature `f`'s per-object `f32` column;
/// `cat_columns[c]` is categorical feature `c`'s per-object raw VALUE column
/// (already in the A4 string form). A [`ModelSplit::Ctr`] reads its
/// projection-member cat values from `cat_columns` and looks up the baked
/// `ctr_data` table (the combined-projection CTR apply); a [`ModelSplit::Float`]
/// keeps the existing numeric path. The numeric-only [`predict_raw`] is the
/// `cat_columns = &[]` special case (no CTR split is ever evaluated, so the
/// Float-only apply is byte-for-byte unchanged).
#[must_use]
pub fn predict_raw_cat(
    model: &Model,
    feature_values: &[Vec<f32>],
    cat_columns: &[Vec<String>],
) -> Vec<f64> {
    let n_float = feature_values.first().map_or(0, Vec::len);
    let n_cat = cat_columns.first().map_or(0, Vec::len);
    let n_objects = n_float.max(n_cat);
    (0..n_objects)
        .map(|obj| {
            // Gather this object's per-feature float row (checked `.get`; a short
            // column reads NaN, which fails every strict `> border` test).
            let row: Vec<f32> = feature_values
                .iter()
                .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
                .collect();
            // Gather this object's raw categorical values (checked `.get`).
            let cats: Vec<String> = cat_columns
                .iter()
                .map(|col| col.get(obj).cloned().unwrap_or_default())
                .collect();
            predict_raw_one(model, &row, &cats)
        })
        .collect()
}
