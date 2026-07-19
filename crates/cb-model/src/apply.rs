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

use crate::ctr_data::{calc_inference, ctr_base_key, CtrValueTable, Prior};
use crate::error::ModelError;
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
    // Delegate to the shared canonical key (the SAME form the trainer-side bake
    // lift `CtrData::from_baked` uses — Plan 05-14), so the bake key matches the
    // apply key byte-for-byte.
    ctr_base_key(split.ctr_type, split.projection.cat_features())
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
            split.shift,
            split.scale,
            split.target_border_idx,
        ),
        // No baked table for this split: the not-found→empty CTR value
        // (`Calc(0, 0)` over the prior), STILL scaled by the split's Shift/Scale
        // so an absent combined bucket lands in the same border space as the baked
        // borders (Plan 05-14 — never the hardcoded 0.0/1.0). Bounds-safe.
        None => calc_inference(0.0, 0.0, split.prior, split.shift, split.scale),
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

/// The forward-bit-order leaf index of one object in one tree (Step B): the
/// dimension-AGNOSTIC structural leaf the object lands in. Shared verbatim by the
/// scalar ([`predict_raw_one`]) and N-dim ([`predict_raw_multi`]) apply paths so
/// the leaf selection is identical (CR-01: only the leaf-VALUE read differs by
/// dimension, never the leaf index).
fn leaf_index_for(model: &Model, tree: &crate::ObliviousTree, features: &[f32], cat_values: &[String]) -> usize {
    let passes: Vec<bool> = tree
        .splits
        .iter()
        .map(|s| passes_split(model, s, features, cat_values))
        .collect();
    leaf_index(&passes)
}

/// The flat-node pointer-walk leaf index of one object in one NON-SYMMETRIC
/// (Lossguide / Depthwise) tree (FEAT-06, RESEARCH Pattern 1, transcribing
/// `evaluator_impl.cpp:726-742`).
///
/// Starts at the tree-local node `0`, and at each node takes
/// `diff = passes_split(node) ? right_subtree_diff : left_subtree_diff`, advances
/// `index = index + diff` (i64 arithmetic — T-06.6-13), and HALTS as soon as the
/// chosen `diff == 0` (the `TerminalMarker`; a one-sided `(d, 0)` / `(0, d)` node
/// halts on its zero side). The halted node's per-node `node_id_to_leaf_id` entry
/// (LOCAL, 0-based within this tree — see the decode in `cbm.rs`) is the leaf
/// index returned. Returns `None` for a malformed graph (out-of-range index,
/// missing split, an interior `u32::MAX` leaf slot, or a non-terminating walk),
/// so the caller contributes `0.0` rather than panicking (T-06.6-12 — no OOB, no
/// infinite loop on crafted input).
///
/// ALL node access is checked `.get` (`indexing_slicing` deny); the walk is bounded
/// by `node_count` iterations so a cyclic / escaping graph cannot loop forever.
fn leaf_index_nonsym(
    model: &Model,
    tree: &crate::NonSymmetricTree,
    features: &[f32],
    cat_values: &[String],
) -> Option<usize> {
    let node_count = tree.step_nodes.len();
    let mut index: i64 = 0;
    // Hard upper bound: a valid walk visits each node at most once, so more than
    // `node_count` steps means a cyclic / malformed graph — reject (no infinite
    // loop, T-06.6-12).
    for _ in 0..=node_count {
        let idx = usize::try_from(index).ok()?;
        // Validate index < node_count via checked `.get` (T-06.6-13).
        let &(left_diff, right_diff) = tree.step_nodes.get(idx)?;
        let split = tree.tree_splits.get(idx)?;
        let diff: i64 = if passes_split(model, split, features, cat_values) {
            i64::from(right_diff)
        } else {
            i64::from(left_diff)
        };
        index = index.checked_add(diff)?;
        if diff == 0 {
            // Halt: read this node's LOCAL leaf id (a `u32::MAX` here marks a
            // pure-interior node that should never be a halt point → malformed).
            let leaf_id = *tree.node_id_to_leaf_id.get(idx)?;
            if leaf_id == u32::MAX {
                return None;
            }
            return usize::try_from(leaf_id).ok();
        }
    }
    None
}

/// The walk-until-diverge leaf index of one object in one REGION tree (GPUT-18 /
/// D-03a, transcribing `add_model_value.cu::AddRegionImpl` / `ComputeRegionBinsImpl`).
///
/// Walks the path levels from the root: `bin = 0`; at each level compute
/// `split = passes_split(level)`, and while `split == expected_direction` advance
/// (`bin += 1`); on the FIRST mismatch the path diverges and the walk HALTS. The
/// returned leaf is `bin` — the depth reached along the region path (`0..=depth`).
/// A depth-`d` region therefore has exactly `d + 1` reachable leaves.
///
/// All leaf/level access is checked, so a malformed region (missing level, short
/// `leaf_values`) yields a bounded walk and the caller contributes `0.0` rather than
/// panicking (T-12-03 — mirror of [`leaf_index_nonsym`]'s defensive discipline). The
/// walk is intrinsically bounded by the level count (no cyclic-graph risk — a path
/// visits each level at most once).
fn region_leaf(
    model: &Model,
    tree: &crate::RegionTree,
    features: &[f32],
    cat_values: &[String],
) -> usize {
    let mut bin: usize = 0;
    for level in &tree.levels {
        // `passes` is the `value > border` (or CTR) test; `one_hot` equality is
        // reserved for the device / categorical region path (Plan 04) and never
        // emitted by the CPU float grower, so the float `>` test is authoritative
        // here. The walk continues while the test matches the stored direction.
        let passes = passes_split(model, &level.split, features, cat_values);
        if passes == level.expected_direction {
            bin += 1;
        } else {
            break;
        }
    }
    bin
}

/// Apply every tree to one object and accumulate `bias + Σ_trees
/// leaf_values[leaf]` (Steps B + C), branching per-tree on the model's tree
/// VARIANT (D-6.6-05): an oblivious model walks `oblivious_trees` via the EXISTING
/// forward-bit-order [`leaf_index_for`] (BYTE-IDENTICAL); a non-symmetric model
/// walks `non_symmetric_trees` via the pointer-walk [`leaf_index_nonsym`]. A model
/// is EITHER all-oblivious or all-non-symmetric, so exactly one loop contributes.
///
/// The per-tree leaf contributions are summed host-side through [`sum_f64`] (D-08);
/// `bias` is added exactly once afterward (RESEARCH Pitfall 6).
///
/// `features` is the object's contiguous float-feature row; `cat_values` its raw
/// categorical values (empty for the numeric-only apply path, where no CTR split
/// is ever evaluated — the `ModelSplit::Float` path is byte-for-byte unchanged).
fn predict_raw_one(model: &Model, features: &[f32], cat_values: &[String]) -> f64 {
    // Oblivious arm: the existing forward-bit-order path, BYTE-IDENTICAL (D-6.6-05).
    let oblivious: Vec<f64> = model
        .oblivious_trees
        .iter()
        .map(|tree| {
            // Step B: forward-bit-order leaf index over this tree's splits.
            let leaf = leaf_index_for(model, tree, features, cat_values);
            // Step C (per tree): the leaf value (already learning_rate-scaled).
            // Out-of-range leaf indices contribute 0.0 (T-04-02-01 checked access).
            tree.leaf_values.get(leaf).copied().unwrap_or(0.0)
        })
        .collect();
    // Non-symmetric arm: the flat-node pointer-walk (FEAT-06). Empty for an
    // oblivious model, so this contributes nothing there.
    let non_symmetric: Vec<f64> = model
        .non_symmetric_trees
        .iter()
        .map(|tree| {
            leaf_index_nonsym(model, tree, features, cat_values)
                .and_then(|leaf| tree.leaf_values.get(leaf).copied())
                .unwrap_or(0.0)
        })
        .collect();
    // Region arm: the walk-until-diverge path (GPUT-18). Empty for an oblivious /
    // non-symmetric model, so this contributes nothing there. The bin leaf index
    // is checked against `leaf_values` (a short buffer contributes 0.0, no panic).
    let region: Vec<f64> = model
        .region_trees
        .iter()
        .map(|tree| {
            let leaf = region_leaf(model, tree, features, cat_values);
            tree.leaf_values.get(leaf).copied().unwrap_or(0.0)
        })
        .collect();
    // Σ_trees via the order-locked sum (D-08), THEN + bias exactly once.
    model.bias + sum_f64(&oblivious) + sum_f64(&non_symmetric) + sum_f64(&region)
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
    // Multi-output models (`approx_dimension > 1`) route through the dim-aware
    // accumulator (CR-01). A scalar model (`<= 1`) keeps the BYTE-IDENTICAL scalar
    // `predict_raw_one` path (D-04 on the public apply surface).
    if model.approx_dimension > 1 {
        return predict_raw_multi_cat(model, feature_values, cat_columns);
    }
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

/// The upstream `staged_predict` stage tree-counts for the half-open interval
/// `[start, end)` stepping by `step` (`step >= 1`, `start < end`): the counts
/// `start + step, start + 2·step, …` that stay `< end`, then ALWAYS `end` as the
/// final stage. A pure, side-effect-free schedule generator so SP-02's inclusion
/// rule is reasoned about independently of the per-object accumulation.
///
/// All arithmetic is saturating (workspace deny `arithmetic_side_effects` is not
/// on, but overflow would be a silent bug, not a panic).
fn stage_counts(start: usize, end: usize, step: usize) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut count = start.saturating_add(step);
    while count < end {
        counts.push(count);
        count = count.saturating_add(step);
    }
    // The final stage is always the full `end` (upstream always includes it).
    counts.push(end);
    counts
}

/// The cumulative raw approx of ONE object over the first `end` oblivious trees:
/// `bias + Σ_{t<end} tree_t.leaf_values[leaf_index_for(t, x)]`, accumulated
/// through the order-locked [`sum_f64`] (D-08), `+ bias` once. Byte-identical to
/// the oblivious arm of [`predict_raw_one`] truncated to the first `end` trees;
/// `cat_values` is empty on the float-only staged path (no CTR split evaluated).
fn prefix_row_value(model: &Model, features: &[f32], cat_values: &[String], end: usize) -> f64 {
    let contributions: Vec<f64> = model
        .oblivious_trees
        .iter()
        .take(end)
        .map(|tree| {
            let leaf = leaf_index_for(model, tree, features, cat_values);
            tree.leaf_values.get(leaf).copied().unwrap_or(0.0)
        })
        .collect();
    model.bias + sum_f64(&contributions)
}

/// Raw approx over an increasing prefix of the ensemble (SCALAR oblivious models).
/// Returns one row per stage; each row is length `n_objects` in object order.
/// Stages are the tree counts `min(ntree_start + eval_period, ntree_end), …,
/// ntree_end`, where `ntree_end == 0` is treated as `oblivious_trees.len()`.
/// `eval_period == 0` is treated as `1`. An out-of-range `ntree_start >= end`
/// yields an empty `Vec`.
///
/// # Contract (UNGUARDED — caller's responsibility)
/// This function does NOT validate model shape: it only accumulates
/// `model.oblivious_trees` over the scalar leaf-value path. On a multi-dimension
/// (`approx_dimension > 1`), non-symmetric, Region, or CTR model it returns
/// SILENTLY WRONG output (dropped dimensions / ignored trees), not an error. The
/// scalar-oblivious guard lives at the facade (SP-03); any direct `cb-model`
/// caller MUST ensure the model is scalar + oblivious + float-only before calling.
#[must_use]
pub fn predict_raw_staged(
    model: &Model,
    feature_values: &[Vec<f32>],
    ntree_start: usize,
    ntree_end: usize,
    eval_period: usize,
) -> Vec<Vec<f64>> {
    let n_trees = model.oblivious_trees.len();
    // `ntree_end == 0` means "all trees"; otherwise clamp to the tree count.
    let end = if ntree_end == 0 {
        n_trees
    } else {
        ntree_end.min(n_trees)
    };
    if ntree_start >= end {
        return Vec::new();
    }
    // `eval_period == 0` degenerates to a step of 1 (one stage per tree).
    let step = eval_period.max(1);
    let counts = stage_counts(ntree_start, end, step);

    // Float-only staged path: no categorical columns are consulted. The object
    // count is the FIRST float column's length (matching `predict_raw_cat`).
    let n_objects = feature_values.first().map_or(0, Vec::len);
    // Gather each object's contiguous float row ONCE (checked `.get`; a short
    // column reads NaN, which fails every strict `> border` test) and reuse it
    // across every stage.
    let rows: Vec<Vec<f32>> = (0..n_objects)
        .map(|obj| {
            feature_values
                .iter()
                .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
                .collect()
        })
        .collect();

    counts
        .iter()
        .map(|&count| {
            rows.iter()
                .map(|row| prefix_row_value(model, row, &[], count))
                .collect()
        })
        .collect()
}

/// Apply `model` to a numeric feature view, returning the DIMENSION-MAJOR raw
/// approx of length `approx_dimension * n` (output index `d * n + i`) — the public
/// N-dim apply for multi-output models (MultiClass / MultiClassOneVsAll /
/// MultiLogloss / MultiCrossEntropy / MultiQuantile), closing CR-01.
///
/// This is the producer [`crate::apply_multiclass_prediction`] consumes: a loaded
/// model.json / .cbm multi-output model predicts ALL `approx_dimension` outputs,
/// not just dimension 0. At `approx_dimension <= 1` the output is BYTE-IDENTICAL
/// to [`predict_raw`] (the multi accumulator collapses to the scalar path — same
/// leaf index, same `bias + leaf_value` per object), preserving D-04 on the
/// public surface. Models with CTR splits need the categorical columns — use the
/// `cat_columns` form via [`predict_raw_multi_cat`] is internal; this numeric
/// wrapper passes `&[]`.
/// # WARNING — multi-dim-bias models (WR-03)
///
/// This entry point seeds EVERY output dimension with the single scalar
/// [`Model::bias`] (= dim-0 bias). It is therefore WRONG for a model whose
/// `scale_and_bias[1]` is a genuine per-dimension bias vector —
/// RMSEWithUncertainty's `[mean, 0.5*log(var)]` (LOSS-06 / D-6.4-04) — because the
/// dim-1 log-scale starting bias is silently dropped (yielding a wrong
/// `exp(2*log-scale)` variance). For such models use
/// [`predict_raw_multi_biased`] with the per-dim bias read from
/// `model.json`'s `scale_and_bias[1]`.
#[must_use]
pub fn predict_raw_multi(model: &Model, feature_values: &[Vec<f32>]) -> Vec<f64> {
    predict_raw_multi_cat(model, feature_values, &[])
}

/// Apply `model` to a numeric feature view with a PER-DIMENSION bias, returning
/// the DIMENSION-MAJOR raw approx of length `approx_dimension * n` (output index
/// `d * n + i`) — the N-dim apply for models whose `scale_and_bias[1]` is a true
/// per-dimension bias vector (e.g. RMSEWithUncertainty's `[mean, 0.5*log(var)]`,
/// LOSS-06 / D-6.4-04).
///
/// The project's scalar [`Model::bias`] keeps only dim-0 (06.4-02 decision — no
/// per-dim bias struct field), so [`predict_raw_multi`] seeds EVERY dim with that
/// one scalar. That is WRONG for a genuinely multi-dim-bias model (the dim-1
/// log-scale bias is dropped). This variant seeds output slot `d` with
/// `per_dim_bias[d]` instead (the `treeStart == 0` per-dim bias path,
/// `scale_and_bias.cpp:42`), leaving the leaf accumulation identical. A shorter
/// `per_dim_bias` reads `0.0` for the missing dims (checked `.get`).
///
/// At `approx_dimension <= 1` with a single-element `per_dim_bias` this is
/// byte-identical to [`predict_raw_multi`] (same leaf index, same `bias +
/// leaf_value`); the scalar D-04 surface is unaffected.
#[must_use]
pub fn predict_raw_multi_biased(
    model: &Model,
    feature_values: &[Vec<f32>],
    per_dim_bias: &[f64],
) -> Vec<f64> {
    let dim = model.approx_dimension.max(1);
    let n = feature_values.first().map_or(0, Vec::len);

    // dim-major output `out[d * n + i]`, seeded with the PER-DIM bias per slot.
    let mut out = vec![0.0_f64; dim.saturating_mul(n)];
    for d in 0..dim {
        let bias_d = per_dim_bias.get(d).copied().unwrap_or(0.0);
        for i in 0..n {
            if let Some(slot) = out.get_mut(d.saturating_mul(n).saturating_add(i)) {
                *slot = bias_d;
            }
        }
    }

    for obj in 0..n {
        let row: Vec<f32> = feature_values
            .iter()
            .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
            .collect();
        let cats: Vec<String> = Vec::new();
        // Per dimension, accumulate this object's per-tree leaf values over the
        // FULL tree sequence (the per-dim bias seed already holds + bias_d).
        let contributions = apply_tree_slice_one(model, &row, &cats, 0, model.oblivious_trees.len(), dim);
        for d in 0..dim {
            if let Some(slot) = out.get_mut(d.saturating_mul(n).saturating_add(obj)) {
                *slot += contributions.get(d).copied().unwrap_or(0.0);
            }
        }
    }
    out
}

/// The dim-aware accumulator backing [`predict_raw_multi`] (and the multi-output
/// branch of [`predict_raw_cat`]).
///
/// Per object: compute the forward-bit-order `leaf` ONCE ([`leaf_index_for`] —
/// the leaf index is dimension-agnostic). Per tree, set
/// `n_leaves = tree.leaf_values.len() / dim` (checked; a non-multiple yields a
/// 0-contribution, never a panic) and, for each `d in 0..dim`, read
/// `tree.leaf_values.get(d * n_leaves + leaf)` (the DIMENSION-MAJOR in-memory
/// buffer — the loaded model un-transposes the LEAF-MAJOR wire form to dim-major).
/// The per-(dim, object) tree contributions are summed through [`sum_f64`] (D-08),
/// then `model.bias` is added exactly once per output slot (single scalar bias
/// this wave, RESEARCH Pitfall 6). All access is checked `.get` (`indexing_slicing`
/// deny, T-6.2-06-01).
#[must_use]
fn predict_raw_multi_cat(
    model: &Model,
    feature_values: &[Vec<f32>],
    cat_columns: &[Vec<String>],
) -> Vec<f64> {
    let dim = model.approx_dimension.max(1);
    let n_float = feature_values.first().map_or(0, Vec::len);
    let n_cat = cat_columns.first().map_or(0, Vec::len);
    let n = n_float.max(n_cat);

    // dim-major output `out[d * n + i]`, seeded with `bias` per slot (single
    // scalar bias this wave — added exactly once, RESEARCH Pitfall 6). NOTE
    // (WR-03): every dim is seeded with the SAME scalar dim-0 bias, so this path
    // is WRONG for a genuine per-dim-bias model (e.g. RMSEWithUncertainty's dim-1
    // log-scale bias) — those callers must use `predict_raw_multi_biased`.
    let mut out = vec![model.bias; dim.saturating_mul(n)];

    for obj in 0..n {
        // This object's float row + raw cat values (checked `.get`; a short float
        // column reads NaN, failing every strict `> border` test).
        let row: Vec<f32> = feature_values
            .iter()
            .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
            .collect();
        let cats: Vec<String> = cat_columns
            .iter()
            .map(|col| col.get(obj).cloned().unwrap_or_default())
            .collect();

        // Per dimension, accumulate this object's per-tree leaf values through the
        // order-locked sum (D-08), then the seeded `bias` already holds the +bias.
        // A model is EITHER all-oblivious or all-non-symmetric, so exactly one of
        // the two arms contributes (D-6.6-05); the oblivious arm is byte-identical.
        for d in 0..dim {
            let mut contributions: Vec<f64> = model
                .oblivious_trees
                .iter()
                .map(|tree| {
                    let leaf = leaf_index_for(model, tree, &row, &cats);
                    // n_leaves = leaf_values.len() / dim (checked); a non-multiple
                    // makes n_leaves a floor and the read may fall out of range,
                    // which contributes 0.0 (T-6.2-06-01) — never a panic.
                    let n_leaves = tree.leaf_values.len() / dim;
                    tree.leaf_values
                        .get(d.saturating_mul(n_leaves).saturating_add(leaf))
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();
            // Non-symmetric arm: the flat-node pointer-walk, reading the
            // DIMENSION-MAJOR per-tree `leaf_values` at `d * n_leaves + leaf`
            // (FEAT-06; empty for an oblivious model).
            contributions.extend(model.non_symmetric_trees.iter().map(|tree| {
                // A ragged per-tree buffer (`len % dim != 0`, only possible from a
                // malformed loaded model) would make `n_leaves` a floor and
                // mis-stride dimension d>0; contribute 0.0 instead of a
                // plausible-but-wrong value (WR-02). Valid models are unaffected.
                let n_leaves = if dim == 0 || tree.leaf_values.len() % dim != 0 {
                    0
                } else {
                    tree.leaf_values.len() / dim
                };
                leaf_index_nonsym(model, tree, &row, &cats)
                    .and_then(|leaf| {
                        tree.leaf_values
                            .get(d.saturating_mul(n_leaves).saturating_add(leaf))
                            .copied()
                    })
                    .unwrap_or(0.0)
            }));
            if let Some(slot) = out.get_mut(d.saturating_mul(n).saturating_add(obj)) {
                *slot += sum_f64(&contributions);
            }
        }
    }
    out
}

/// Accumulate one object's per-dimension leaf-value sum over the tree SLICE
/// `[tree_start, tree_end)` — the dimension-major `[d0, d1, …]` contribution of
/// that tree range, WITHOUT any bias (the VE bias is applied once by
/// [`apply_virtual_ensembles`], matching upstream `ApplyScaleAndBias`'s
/// `treeStart > 0` no-bias path, `scale_and_bias.cpp:13`).
///
/// `dim = model.approx_dimension.max(1)`; `row` / `cats` are the object's gathered
/// float / categorical values. All per-(dim) Σ route through [`sum_f64`] (D-08).
/// Out-of-range tree indices are skipped (checked `.get`); a leaf index out of a
/// tree's range contributes `0.0` (T-04-02-01) — never a panic.
fn apply_tree_slice_one(
    model: &Model,
    row: &[f32],
    cats: &[String],
    tree_start: usize,
    tree_end: usize,
    dim: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(dim);
    for d in 0..dim {
        let contributions: Vec<f64> = (tree_start..tree_end)
            .filter_map(|ti| model.oblivious_trees.get(ti))
            .map(|tree| {
                let leaf = leaf_index_for(model, tree, row, cats);
                let n_leaves = tree.leaf_values.len() / dim;
                tree.leaf_values
                    .get(d.saturating_mul(n_leaves).saturating_add(leaf))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        out.push(sum_f64(&contributions));
    }
    out
}

/// Per-tree leaf statistics over a dataset (upstream `CollectLeavesStatistics`,
/// `catboost/libs/fstr/util.cpp` v1.2.10): apply every tree to every object via
/// the SAME leaf-assignment walk the predict path uses ([`leaf_index_for`] /
/// [`leaf_index_nonsym`]) and add `1.0` per document to the landed leaf's slot
/// (unit document weights — upstream falls back to `+= 1.0` when the pool's
/// weights are trivial, which is the only mode this port supports).
///
/// The outer Vec lists oblivious trees first, then non-symmetric trees (a model
/// is one kind or the other in practice); each inner Vec has one slot per
/// STRUCTURAL leaf (`leaf_values.len() / approx_dimension`). This is the
/// dataset-weights substrate of
/// [`crate::prediction_values_change_with_data`] — for online-CTR models
/// these statistics genuinely differ from the stored training-time
/// `leaf_weights`, because training assigns documents by online (prefix) CTR
/// values while this walk uses the final baked tables. A malformed
/// non-symmetric walk ([`leaf_index_nonsym`] `None`) contributes nothing for
/// that document, mirroring the apply path's defensive discipline.
#[must_use]
pub fn collect_leaves_statistics(
    model: &Model,
    feature_values: &[Vec<f32>],
    cat_columns: &[Vec<String>],
) -> Vec<Vec<f64>> {
    let dim = model.approx_dimension.max(1);
    let mut stats: Vec<Vec<f64>> = model
        .oblivious_trees
        .iter()
        .map(|t| vec![0.0_f64; t.leaf_values.len() / dim])
        .chain(
            model
                .non_symmetric_trees
                .iter()
                .map(|t| vec![0.0_f64; t.leaf_values.len() / dim]),
        )
        .collect();

    let n_float_objs = feature_values.first().map_or(0, Vec::len);
    let n_cat_objs = cat_columns.first().map_or(0, Vec::len);
    let n_objects = n_float_objs.max(n_cat_objs);
    let nonsym_base = model.oblivious_trees.len();

    for obj in 0..n_objects {
        // Gather rows exactly as `predict_raw_cat` does (checked `.get`; a
        // short float column reads NaN, failing every strict `> border` test).
        let row: Vec<f32> = feature_values
            .iter()
            .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
            .collect();
        let cats: Vec<String> = cat_columns
            .iter()
            .map(|col| col.get(obj).cloned().unwrap_or_default())
            .collect();

        for (tree_idx, tree) in model.oblivious_trees.iter().enumerate() {
            let leaf = leaf_index_for(model, tree, &row, &cats);
            if let Some(slot) = stats.get_mut(tree_idx).and_then(|s| s.get_mut(leaf)) {
                *slot += 1.0;
            }
        }
        for (tree_idx, tree) in model.non_symmetric_trees.iter().enumerate() {
            if let Some(leaf) = leaf_index_nonsym(model, tree, &row, &cats) {
                if let Some(slot) =
                    stats.get_mut(nonsym_base + tree_idx).and_then(|s| s.get_mut(leaf))
                {
                    *slot += 1.0;
                }
            }
        }
    }
    stats
}

/// Apply the trained model's virtual-ensemble slicing (`apply.cpp:526-600`),
/// producing the per-ensemble RAW approx matrix in OBJECT-MAJOR
/// `(n, virtual_ensembles_count, approx_dimension)` flat layout — value at index
/// `i * (V * dim) + e * dim + d`. These are the RAW per-ensemble dims (dim 0 mean,
/// dim 1 log-scale for RMSEWithUncertainty); the variance transform `exp(2*x)` is
/// applied later by [`crate::apply_prediction_type`] (`VirtEnsembles` /
/// `TotalUncertainty`).
///
/// Transcribes upstream EXACTLY (default `unshrinkCoef = 1`, A2 — `model_shrink_rate
/// = 0`; non-zero constant-shrink is a documented v2 follow-up, NOT a silent
/// omission):
/// - `end = model.oblivious_trees.len()` (all trees; upstream `ntree_end = 0`).
/// - `eval_period = end / (2 * V)` (INTEGER division).
/// - Guard `eval_period > 0 && eval_period * V < end`, else a `ModelError`
///   ("Not enough trees …", `CB_ENSURE` equivalent — NOT a panic, T-06.4C-01).
///   `V = 0` also errors (eval_period would div-by-zero / the guard fails).
/// - `begin = end - eval_period * V`.
/// - Ensemble 0 seeds from the apply of trees `[0, begin)` PLUS the per-dim
///   `bias` (the `treeStart == 0` bias path, `scale_and_bias.cpp:13`).
/// - For `v in 0..V`: add the apply of trees `[begin + v*eval_period,
///   begin + (v+1)*eval_period)` (NO bias — `treeStart > 0`) to the running
///   ensemble approx; for all but the last ensemble, copy the running sum forward
///   to seed the next ensemble (the `copyerLambda` with `copyToNextEnsemble`).
///
/// `bias` is the per-dimension model bias (`scale_and_bias[1]`, length
/// `approx_dimension`); a shorter slice reads `0.0` for the missing dims (checked
/// `.get`). All Σ route through [`sum_f64`] (D-08). All access is checked `.get`
/// (`indexing_slicing` deny) — out-of-range tree / leaf indices never panic.
///
/// # Errors
/// [`ModelError::Deserialize`] when the model has fewer than `2*V + 1` trees for
/// the requested `virtual_ensembles_count` (or `V == 0`).
pub fn apply_virtual_ensembles(
    model: &Model,
    feature_values: &[Vec<f32>],
    bias: &[f64],
    virtual_ensembles_count: usize,
) -> Result<Vec<f64>, ModelError> {
    let dim = model.approx_dimension.max(1);
    let end = model.oblivious_trees.len();
    let v = virtual_ensembles_count;

    // evalPeriod = end / (2*V) (integer); guard evalPeriod>0 && evalPeriod*V < end
    // (need >= 2V+1 trees). V == 0 makes the denominator 0 -> reject up front.
    if v == 0 {
        return Err(ModelError::Deserialize(
            "virtual_ensembles_count must be > 0".to_owned(),
        ));
    }
    let eval_period = end / (2 * v);
    if eval_period == 0 || eval_period.saturating_mul(v) >= end {
        return Err(ModelError::Deserialize(format!(
            "Not enough trees in model for {v} virtual ensembles \
             (have {end}, need >= {})",
            2 * v + 1
        )));
    }
    let begin = end - eval_period * v;

    let n_float = feature_values.first().map_or(0, Vec::len);
    let n = n_float;

    // Object-major output `(n, V, dim)`: out[i*(V*dim) + e*dim + d].
    let mut out = vec![0.0_f64; n.saturating_mul(v).saturating_mul(dim)];

    for obj in 0..n {
        // This object's float row (checked `.get`; a short column reads NaN,
        // failing every strict `> border` test). No categorical columns on the VE
        // numeric apply path (the RMSEWithUncertainty fixture is numeric-only).
        let row: Vec<f32> = feature_values
            .iter()
            .map(|col| col.get(obj).copied().unwrap_or(f32::NAN))
            .collect();
        let cats: Vec<String> = Vec::new();

        // Ensemble 0 base: apply trees [0, begin) PLUS the per-dim bias (the
        // treeStart == 0 bias path). This becomes the running approx seed.
        let base = apply_tree_slice_one(model, &row, &cats, 0, begin, dim);
        let mut running: Vec<f64> = (0..dim)
            .map(|d| base.get(d).copied().unwrap_or(0.0) + bias.get(d).copied().unwrap_or(0.0))
            .collect();

        // Each ensemble adds its evalPeriod-tree slice (NO bias, treeStart > 0),
        // then the running sum is the ensemble's approx; copy-forward seeds the
        // next ensemble (copyerLambda, copyToNextEnsemble for all but the last).
        for e in 0..v {
            let slice_start = begin + e.saturating_mul(eval_period);
            let slice_end = (slice_start + eval_period).min(end);
            let slice = apply_tree_slice_one(model, &row, &cats, slice_start, slice_end, dim);
            for d in 0..dim {
                // running[d] += slice[d] (the per-dim ensemble accumulation).
                if let Some(r) = running.get_mut(d) {
                    *r = sum_f64(&[*r, slice.get(d).copied().unwrap_or(0.0)]);
                }
                // Store this ensemble's approx (object-major (n, V, dim)).
                let idx = obj
                    .saturating_mul(v)
                    .saturating_mul(dim)
                    .saturating_add(e.saturating_mul(dim))
                    .saturating_add(d);
                if let Some(slot) = out.get_mut(idx) {
                    *slot = running.get(d).copied().unwrap_or(0.0);
                }
            }
            // The running sum carries forward to seed the next ensemble (it is NOT
            // reset) — `unshrinkCoef = 1` (default A2 path), so no rescale.
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "region_apply_test.rs"]
mod region_apply_test;

#[cfg(test)]
#[path = "staged_predict_test.rs"]
mod staged_predict_test;
