//! Online (Plain-mode, whole-set) CTR accumulation — the per-bucket class-count
//! and target-sum histograms every CTR type is computed from (ORD-03, D-06).
//!
//! # Plain vs Ordered (D-06 key isolation)
//!
//! This module accumulates the WHOLE learn set into each bucket (no per-object
//! prefix): for every document, its categorical bucket's class count / target
//! sum is incremented, and the final-CTR table (see [`crate::ctr::final_ctr`])
//! reads the completed totals. This is the Plain-mode target statistic locked
//! BEFORE the ordered (read-before-increment) per-object prefix of Wave 5, so a
//! later divergence localizes to the ordering math, never the CTR math.
//!
//! The upstream read-before-increment template
//! (`online_ctr.cpp:168-184/300-307`) reads the prefix counts for a document's
//! bucket THEN increments — the prefix IS the no-leakage property. Plain mode is
//! that loop run to completion (whole set), so each bucket holds its full counts.
//! The bucket-accumulation shape mirrors
//! `boosting.rs::accumulate_leaf_weights` (bucket members in object order), but
//! the bucket key is the categorical bin, not the leaf.
//!
//! # Source of truth
//!
//! - `online_ctr.cpp:300-307` (`CalcQuantizedCtrs`, binclf simple path):
//!   `goodCount=elem[1]; totalCount=elem[0]+elem[1]; ++elem[targetClass]` — the
//!   `N[0]`/`N[1]` neg/pos class counts of [`TCtrHistory`].
//! - `online_ctr.cpp:916-939` (`CalcFinalCtrsImpl`, whole-set): Borders/Buckets
//!   accumulate `++ctrIntArray[targetClassesCount * elemId + targetClass]`;
//!   BinarizedTargetMeanValue `Add(targetClass / targetBorderCount)`;
//!   FloatTargetMeanValue `Add(target)` (raw); Counter/FeatureFreq `++count`.
//!
//! # Parity discipline
//!
//! Integer class counts ([`TCtrHistory::n`]) are EXACT integer accumulation —
//! they do NOT route through `sum_f64` (RESEARCH Anti-Pattern caveat: only FLOAT
//! sums do). The float [`TCtrMeanHistory::sum`] is a single running `f32` add
//! per the upstream `TCtrMeanHistory::Add` (`online_ctr.h:373-376`), matching
//! upstream's per-element float accumulation bit-for-bit; the parity-critical
//! whole-vector reductions in [`crate::ctr::final_ctr`] use `cb_core::sum_f64`.
//! Categorical hashing is via [`cb_data::calc_cat_feature_hash`] +
//! [`cb_data::PerfectHash`] — NEVER a model's `ctr_data` hash_map
//! (D Carried-Forward). Checked access only; no `unwrap`/`expect`/panic/raw
//! index; no `anyhow`.

use cb_core::{CbError, CbResult};
use cb_data::{calc_cat_feature_hash, PerfectHash};

use crate::ctr::calc_ctr::calc_ctr_online;

/// The number of target classes for the simple binary-classification CTR path
/// (`online_ctr.cpp` `SIMPLE_CLASSES_COUNT == 2`). The neg/pos class counts are
/// `N[0]`/`N[1]`.
pub const SIMPLE_CLASSES_COUNT: usize = 2;

/// Per-bucket integer class-count history (`TCtrHistory`, `online_ctr.h:357-367`):
/// `N[targetClassesCount]` class counts. For binclf `N[0]`/`N[1]` are the neg/pos
/// counts. Exact integer accumulation (NOT a float sum — RESEARCH Anti-Pattern
/// caveat).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TCtrHistory {
    /// Per-class counts, length `targetClassesCount` (2 for binclf).
    pub n: Vec<i64>,
}

impl TCtrHistory {
    /// A zeroed history with `classes` class slots.
    #[must_use]
    pub fn new(classes: usize) -> Self {
        Self {
            n: vec![0; classes],
        }
    }

    /// Total count summed over every class (`elem[0] + elem[1] + …`). Exact
    /// integer sum (not a float reduction).
    #[must_use]
    pub fn total(&self) -> i64 {
        self.n.iter().sum()
    }

    /// Increment the count of `class` by one (`++elem[targetClass]`). A class
    /// index out of range is ignored (checked access; the caller binarizes the
    /// target into `[0, classes)` so this never drops in practice).
    pub fn increment(&mut self, class: usize) {
        if let Some(slot) = self.n.get_mut(class) {
            *slot += 1;
        }
    }
}

/// Per-bucket float target-sum history (`TCtrMeanHistory`, `online_ctr.h:369-401`):
/// a running `f32` `Sum` and an `i32` `Count`. Used by BinarizedTargetMeanValue
/// (`Add(targetClass / targetBorderCount)`) and FloatTargetMeanValue
/// (`Add(target)`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TCtrMeanHistory {
    /// Running sum of added target values (`f32` to match upstream's
    /// per-element float accumulation, `online_ctr.h:373`).
    pub sum: f32,
    /// Number of added values.
    pub count: i64,
}

impl TCtrMeanHistory {
    /// Add one target value (`TCtrMeanHistory::Add`, `online_ctr.h:373-376`):
    /// `Sum += target; ++Count`. The single-element `f32` add matches upstream
    /// bit-for-bit (the parity-critical WHOLE-vector reductions route through
    /// `cb_core::sum_f64` in [`crate::ctr::final_ctr`]).
    pub fn add(&mut self, target: f32) {
        self.sum += target;
        self.count += 1;
    }
}

/// The accumulated whole-set CTR histograms for one categorical feature: the
/// per-bucket class-count histories (Borders/Buckets), the per-bucket mean
/// histories (BinarizedTargetMeanValue/FloatTargetMeanValue), the per-bucket
/// total counts (Counter/FeatureFreq), and the bucket-defining perfect-hash map.
///
/// All four are filled in ONE pass over the learn set (whole-set, Plain mode);
/// the per-type final-CTR table reads whichever of them its type needs
/// ([`crate::ctr::final_ctr`]). The `bins` give each object its bucket index; the
/// `bucket_count` is the number of distinct buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineCtrAccumulator {
    /// Per-bucket class-count histories (length `bucket_count`).
    pub class_histories: Vec<TCtrHistory>,
    /// Per-bucket mean histories for the binarized-target-mean path
    /// (`Add(targetClass / targetBorderCount)`), length `bucket_count`.
    pub binarized_mean: Vec<TCtrMeanHistory>,
    /// Per-bucket mean histories for the raw float-target-mean path
    /// (`Add(target)`), length `bucket_count`.
    pub float_mean: Vec<TCtrMeanHistory>,
    /// Per-bucket total document counts (Counter/FeatureFreq numerator),
    /// length `bucket_count`.
    pub total_counts: Vec<i64>,
    /// The per-object bucket index (perfect-hash bin) in object order.
    pub bins: Vec<u32>,
    /// The number of distinct buckets (`bins.iter().max() + 1`).
    pub bucket_count: usize,
    /// The number of target classes (2 for binclf).
    pub classes: usize,
}

/// Accumulate the WHOLE learn set into the per-bucket CTR histograms (Plain
/// mode, D-06). `column` holds each object's categorical value already in the A4
/// string form ([`cb_data::stringify_int_category`] for integer-coded values);
/// `target_class[i]` is object `i`'s binarized target class in `[0, classes)`;
/// `target[i]` is its raw float target (for FloatTargetMeanValue);
/// `target_border_count` is the binarized-target divisor (for
/// BinarizedTargetMeanValue, `Add(targetClass / targetBorderCount)`).
///
/// One pass: hash each value to its perfect-hash bin
/// ([`cb_data::calc_cat_feature_hash`] + [`cb_data::PerfectHash`], never a model
/// `ctr_data` hash_map), then increment that bucket's class count, mean sums, and
/// total count. The bucket histograms are COMPLETE on return (whole set), ready
/// for the per-type final-CTR table build.
///
/// # Errors
/// - [`CbError::Degenerate`] if `column`, `target_class`, and `target` differ in
///   length, or `target_border_count == 0`.
/// - [`CbError::OutOfRange`] propagated from [`cb_data::PerfectHash::remap`] if
///   the column exceeds the `u32::MAX` distinct-value bound.
pub fn accumulate_online(
    column: &[&str],
    target_class: &[usize],
    target: &[f64],
    classes: usize,
    target_border_count: usize,
) -> CbResult<OnlineCtrAccumulator> {
    let n = column.len();
    if target_class.len() != n || target.len() != n {
        return Err(CbError::Degenerate(
            "ctr accumulate: column / target_class / target length mismatch".to_owned(),
        ));
    }
    if target_border_count == 0 {
        return Err(CbError::Degenerate(
            "ctr accumulate: target_border_count must be non-zero".to_owned(),
        ));
    }

    // First pass: hash + remap each value to its perfect-hash bin, recording the
    // per-object bins and the distinct-bucket count.
    let mut ph = PerfectHash::new();
    let mut bins: Vec<u32> = Vec::with_capacity(n);
    for &value in column {
        let hash = calc_cat_feature_hash(value);
        bins.push(ph.remap(hash)?);
    }
    let bucket_count = ph.len();

    let mut class_histories = vec![TCtrHistory::new(classes); bucket_count];
    let mut binarized_mean = vec![TCtrMeanHistory::default(); bucket_count];
    let mut float_mean = vec![TCtrMeanHistory::default(); bucket_count];
    let mut total_counts = vec![0i64; bucket_count];

    // Whole-set accumulation pass (Plain mode — no prefix). Checked `.get` only.
    let divisor = target_border_count as f32;
    for i in 0..n {
        let Some(&bin) = bins.get(i) else { continue };
        let bucket = bin as usize;
        let Some(&class) = target_class.get(i) else {
            continue;
        };
        let Some(&raw_target) = target.get(i) else {
            continue;
        };

        if let Some(hist) = class_histories.get_mut(bucket) {
            hist.increment(class);
        }
        if let Some(mean) = binarized_mean.get_mut(bucket) {
            // BinarizedTargetMeanValue: Add(targetClass / targetBorderCount).
            mean.add(class as f32 / divisor);
        }
        if let Some(mean) = float_mean.get_mut(bucket) {
            // FloatTargetMeanValue: Add(target) raw.
            mean.add(raw_target as f32);
        }
        if let Some(total) = total_counts.get_mut(bucket) {
            *total += 1;
        }
    }

    Ok(OnlineCtrAccumulator {
        class_histories,
        binarized_mean,
        float_mean,
        total_counts,
        bins,
        bucket_count,
        classes,
    })
}

/// The per-object ONLINE (ordered, read-before-increment) binclf CTR over a
/// permutation — the per-object prefix statistic the `plain_ctr` fixture dumps
/// (`online_ctr.cpp:300-307`, `CalcQuantizedCtrs` simple binclf path).
///
/// For each position `p` in permutation order, the document `doc = permutation[p]`
/// READS its bucket's accumulated prefix counts BEFORE its own label is added:
/// `good = N[1]`, `total = N[0] + N[1]`, then `ctr = (good + prior) / (total + 1)`
/// ([`calc_ctr_online`]), then `++N[targetClass[doc]]`. The READ-BEFORE-INCREMENT
/// is the no-leakage property — a document's CTR never sees its own label.
///
/// Even in Plain BOOSTING mode this online prefix is computed within the single
/// permutation whenever a cat feature exceeds `one_hot_max_size` (`hasCtrs`,
/// RESEARCH Pitfall 2) — that is exactly the `plain_ctr` scenario.
///
/// Returns the per-object `(good_count, total_count, ctr_value)` in OBJECT order
/// (indexed by `doc`, not by permutation position), matching the fixture's
/// `ctr_good_count` / `ctr_total_count` / `ctr_value` `.npy` schema (D-02).
///
/// # Parameters
/// - `permutation[p]` — the object index at learn-order position `p`.
/// - `bins[doc]` — object `doc`'s categorical bucket (perfect-hash bin).
/// - `target_class[doc]` — object `doc`'s binarized class in `[0, 2)`.
/// - `prior` — the additive CTR prior numerator (e.g. `0.5`).
///
/// # Errors
/// [`CbError::Degenerate`] if `bins` / `target_class` are shorter than the
/// permutation implies, or a permutation index is out of range.
pub fn online_ctr_prefix_binclf(
    permutation: &[i32],
    bins: &[u32],
    target_class: &[usize],
    prior: f64,
) -> CbResult<OnlineCtrPrefix> {
    let n = permutation.len();
    if bins.len() != n || target_class.len() != n {
        return Err(CbError::Degenerate(
            "online_ctr_prefix: permutation / bins / target_class length mismatch".to_owned(),
        ));
    }

    // Per-bucket [N0, N1] prefix counts; the bucket count bounds the histogram.
    let bucket_count = bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut counts: Vec<[i64; SIMPLE_CLASSES_COUNT]> = vec![[0, 0]; bucket_count];

    let mut good = vec![0i64; n];
    let mut total = vec![0i64; n];
    let mut value = vec![0f64; n];

    for &doc_i in permutation {
        let doc = doc_i as usize;
        let Some(&bin) = bins.get(doc) else {
            return Err(CbError::Degenerate(
                "online_ctr_prefix: permutation index out of range for bins".to_owned(),
            ));
        };
        let Some(&class) = target_class.get(doc) else {
            return Err(CbError::Degenerate(
                "online_ctr_prefix: permutation index out of range for target_class".to_owned(),
            ));
        };
        let bucket = bin as usize;
        let elem = counts.get(bucket);
        // READ the prefix counts BEFORE incrementing (online_ctr.cpp:303-304).
        let (n0, n1) = elem.map_or((0, 0), |e| (e[0], e[1]));
        let g = n1; // good = N[1] (pos class)
        let t = n0 + n1; // total = N[0] + N[1]
        if let Some(slot) = good.get_mut(doc) {
            *slot = g;
        }
        if let Some(slot) = total.get_mut(doc) {
            *slot = t;
        }
        if let Some(slot) = value.get_mut(doc) {
            *slot = calc_ctr_online(g as f64, t, prior);
        }
        // INCREMENT after read (learn set): ++N[targetClass[doc]].
        if let Some(elem) = counts.get_mut(bucket) {
            if let Some(c) = elem.get_mut(class) {
                *c += 1;
            }
        }
    }

    Ok(OnlineCtrPrefix { good, total, value })
}

/// The per-object online (ordered) CTR prefix vectors in OBJECT order
/// (`online_ctr_prefix_binclf`): the integer numerator/denominator and the f64
/// CTR value per document, matching the `plain_ctr` fixture's D-02 `.npy` schema.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineCtrPrefix {
    /// Per-object good count `N[1]` read BEFORE the document's own label
    /// (`ctr_good_count.npy`).
    pub good: Vec<i64>,
    /// Per-object total count `N[0] + N[1]` read before the label
    /// (`ctr_total_count.npy`).
    pub total: Vec<i64>,
    /// Per-object online CTR value `(good + prior) / (total + 1)`
    /// (`ctr_value.npy`).
    pub value: Vec<f64>,
}

/// The ORDERED (per-permutation) online CTR for one permutation — the focused
/// delta of Wave 5 over the Plain-mode whole-set CTR of 05-04 (D-05/D-06). It is
/// the SAME read-before-increment prefix loop ([`online_ctr_prefix_binclf`]) but
/// computed UNDER A SPECIFIC PERMUTATION (the per-fold order, `online_ctr.cpp`
/// `CalcOnlineCTRClasses` runs once per learn permutation). Ordered boosting
/// drives one ordered CTR per learning fold; the `ordered_ctr` fixture commits
/// fold-0's per-object prefix (and the fold-0 / fold-1 permutations themselves
/// for the D-03 gate).
///
/// Beyond the OBJECT-order `good`/`total`/`value` ([`OnlineCtrPrefix`]), this
/// also returns the running `(num, denom)` along the PERMUTATION (the prefix
/// read by each successive document, in learn order) — the internal-consistency
/// anchor the per-object oracle asserts MONOTONE non-decreasing (a document only
/// ever sees more predecessors as the prefix grows; a non-monotone running count
/// would betray an out-of-order accumulation, the silent-leakage signature).
///
/// # Parameters
/// As [`online_ctr_prefix_binclf`]: `permutation[p]` is the object at learn-order
/// position `p`; `bins[doc]` the bucket; `target_class[doc]` the binclf class;
/// `prior` the additive numerator.
///
/// # Errors
/// Propagated from [`online_ctr_prefix_binclf`] (length / range checks).
pub fn ordered_ctr_per_permutation(
    permutation: &[i32],
    bins: &[u32],
    target_class: &[usize],
    prior: f64,
) -> CbResult<OrderedCtrPrefix> {
    // The per-object prefix is exactly the read-before-increment loop; recompute
    // it AND capture the running (num, denom) read at each permutation step for
    // the monotone internal-consistency anchor (per-bucket prefixes grow as the
    // permutation advances, so the per-step read for a fixed bucket is monotone).
    let prefix = online_ctr_prefix_binclf(permutation, bins, target_class, prior)?;

    // Running num/denom AT EACH permutation STEP (learn order), i.e. the prefix
    // value each successive document reads — indexed by permutation position.
    let n = permutation.len();
    let bucket_count = bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut counts: Vec<[i64; SIMPLE_CLASSES_COUNT]> = vec![[0, 0]; bucket_count];
    let mut step_num = Vec::with_capacity(n);
    let mut step_denom = Vec::with_capacity(n);
    for &doc_i in permutation {
        let doc = doc_i as usize;
        let Some(&bin) = bins.get(doc) else {
            return Err(CbError::Degenerate(
                "ordered_ctr: permutation index out of range for bins".to_owned(),
            ));
        };
        let Some(&class) = target_class.get(doc) else {
            return Err(CbError::Degenerate(
                "ordered_ctr: permutation index out of range for target_class".to_owned(),
            ));
        };
        let bucket = bin as usize;
        let (n0, n1) = counts.get(bucket).map_or((0, 0), |e| (e[0], e[1]));
        step_num.push(n1);
        step_denom.push(n0 + n1);
        if let Some(elem) = counts.get_mut(bucket) {
            if let Some(c) = elem.get_mut(class) {
                *c += 1;
            }
        }
    }

    Ok(OrderedCtrPrefix {
        prefix,
        step_num,
        step_denom,
    })
}

/// The ordered (per-permutation) online CTR result: the OBJECT-order per-object
/// prefix plus the PERMUTATION-order running `(num, denom)` (the prefix read at
/// each learn-order step) for the monotone internal-consistency anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedCtrPrefix {
    /// The per-object `good`/`total`/`value` (OBJECT order) — matches the
    /// `ordered_ctr` fixture's `ctr_good_count`/`ctr_total_count`/`ctr_value`.
    pub prefix: OnlineCtrPrefix,
    /// The running good count read at each PERMUTATION step (learn order). For a
    /// fixed bucket this is monotone non-decreasing across that bucket's steps.
    pub step_num: Vec<i64>,
    /// The running total count read at each PERMUTATION step (learn order).
    pub step_denom: Vec<i64>,
}

impl OrderedCtrPrefix {
    /// True iff, within EACH bucket, the running `(num, denom)` read along the
    /// permutation is monotone non-decreasing — the no-out-of-order anchor.
    /// `bins[permutation[p]]` keys each step to its bucket.
    #[must_use]
    pub fn per_bucket_monotone(&self, permutation: &[i32], bins: &[u32]) -> bool {
        let bucket_count = bins.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut last_num = vec![i64::MIN; bucket_count];
        let mut last_denom = vec![i64::MIN; bucket_count];
        for (p, &doc_i) in permutation.iter().enumerate() {
            let doc = doc_i as usize;
            let Some(&bin) = bins.get(doc) else {
                return false;
            };
            let bucket = bin as usize;
            let (Some(&num), Some(&denom)) = (self.step_num.get(p), self.step_denom.get(p)) else {
                return false;
            };
            let (Some(ln), Some(ld)) = (last_num.get_mut(bucket), last_denom.get_mut(bucket))
            else {
                return false;
            };
            if num < *ln || denom < *ld {
                return false;
            }
            *ln = num;
            *ld = denom;
        }
        true
    }
}
