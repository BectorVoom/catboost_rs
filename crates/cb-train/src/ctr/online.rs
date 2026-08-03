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
/// The per-object BinarizedTargetMeanValue prefix vectors in OBJECT order: the
/// running `f32` `Sum`, the `i64` `Count`, and the resulting CTR value per
/// document (SPEC-CTRT-07).
///
/// # Why a distinct type and not [`OnlineCtrPrefix`]
/// `OnlineCtrPrefix::good` is `Vec<i64>`. Reusing it would force an i64
/// truncation of the f32 `Sum` — precisely the silent-widening failure this
/// producer exists to prevent. The `f32` width of `sum` is load-bearing and
/// mirrors upstream `TCtrMeanHistory::Sum` (`online_ctr.h:373-376`).
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineMeanPrefix {
    /// The running target-sum read by each document, in OBJECT order. `f32` to
    /// match upstream's accumulator width bit-for-bit.
    pub sum: Vec<f32>,
    /// The running count read by each document, in OBJECT order.
    pub count: Vec<i64>,
    /// The per-document CTR value `calc_ctr_online(sum, count, prior)`.
    pub value: Vec<f64>,
}

/// The read-before-increment BinarizedTargetMeanValue prefix over
/// [`TCtrMeanHistory`] (SPEC-CTRT-07).
///
/// Each document READS its bucket's `(Sum, Count)` and only then folds its own
/// binarized target in — the no-leakage property (`online_ctr.cpp:168-185`). The
/// added value is `targetClass / targetBorderCount` where upstream passes
/// `targetClassesCount - 1` (`online_ctr.cpp:762`), so for binclf it is exactly
/// `targetClass ∈ {0.0f, 1.0f}`.
///
/// `Sum` is accumulated in **f32**, matching upstream; see
/// `online_test::btmv_sum_is_accumulated_in_f32_not_f64`.
///
/// # Errors
/// [`CbError::Degenerate`] on a length mismatch between `permutation`, `bins` and
/// `target_class`, or a permutation index out of range for either.
pub fn online_mean_prefix(
    permutation: &[i32],
    bins: &[u32],
    target_class: &[usize],
    classes: usize,
    prior: f64,
) -> CbResult<OnlineMeanPrefix> {
    let n = permutation.len();
    if bins.len() != n || target_class.len() != n {
        return Err(CbError::Degenerate(
            "online_mean_prefix: permutation / bins / target_class length mismatch".to_owned(),
        ));
    }

    let bucket_count = bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut hists: Vec<TCtrMeanHistory> = vec![TCtrMeanHistory::default(); bucket_count];

    let mut sum = vec![0f32; n];
    let mut count = vec![0i64; n];
    let mut value = vec![0f64; n];

    // targetBorderCount = targetClassesCount - 1 (online_ctr.cpp:762), floored at
    // 1 so a degenerate `classes` can never divide by zero.
    let divisor = classes.saturating_sub(1).max(1) as f32;

    for &doc_i in permutation {
        let doc = doc_i as usize;
        let Some(&bin) = bins.get(doc) else {
            return Err(CbError::Degenerate(
                "online_mean_prefix: permutation index out of range for bins".to_owned(),
            ));
        };
        let Some(&class) = target_class.get(doc) else {
            return Err(CbError::Degenerate(
                "online_mean_prefix: permutation index out of range for target_class".to_owned(),
            ));
        };
        let bucket = bin as usize;

        // READ before incrementing.
        let (s, c) = hists.get(bucket).map_or((0.0f32, 0i64), |h| (h.sum, h.count));
        if let Some(slot) = sum.get_mut(doc) {
            *slot = s;
        }
        if let Some(slot) = count.get_mut(doc) {
            *slot = c;
        }
        if let Some(slot) = value.get_mut(doc) {
            *slot = calc_ctr_online(f64::from(s), c, prior);
        }

        // INCREMENT after read: Add(targetClass / targetBorderCount).
        if let Some(h) = hists.get_mut(bucket) {
            h.add(class as f32 / divisor);
        }
    }

    Ok(OnlineMeanPrefix { sum, count, value })
}

/// The read-before-increment class-count prefix COLUMN for any class-prefix CTR
/// type, deriving every document's `(numerator, denominator)` **exclusively**
/// through [`online_class_prefix`] (SPEC-CTRT-06).
///
/// This is the generalization of [`online_ctr_prefix_binclf`]: at
/// `(ECtrType::Borders, target_border_idx = 0, classes = 2)` the two produce
/// bit-identical output (pinned by
/// `online_test::class_prefix_column_at_borders_b0_equals_the_binclf_prefix`).
///
/// # Errors
/// - [`CbError::Degenerate`] on a length mismatch or an out-of-range permutation
///   index.
/// - [`CbError::Degenerate`] if `ctr_type` is [`ECtrType::Counter`](crate::ctr::ECtrType::Counter):
///   its denominator is the MAX bucket total, which no single bucket's class
///   counts can produce. Use [`online_counter_column`] instead. A checked misuse,
///   never a silently wrong column.
pub fn online_class_prefix_column(
    permutation: &[i32],
    bins: &[u32],
    target_class: &[usize],
    classes: usize,
    target_border_idx: usize,
    ctr_type: crate::ctr::ECtrType,
    prior: f64,
) -> CbResult<OnlineCtrPrefix> {
    if matches!(ctr_type, crate::ctr::ECtrType::Counter) {
        return Err(CbError::Degenerate(
            "Counter is not a class-prefix CTR type; use online_counter_column".to_owned(),
        ));
    }

    let n = permutation.len();
    if bins.len() != n || target_class.len() != n {
        return Err(CbError::Degenerate(
            "online_class_prefix_column: permutation / bins / target_class length mismatch"
                .to_owned(),
        ));
    }

    let bucket_count = bins.iter().copied().max().map_or(0, |m| m as usize + 1);

    // Per-bucket class counts as ONE FLAT allocation of `bucket_count * classes`,
    // row-major by bucket — deliberately NOT `Vec<TCtrHistory>`.
    //
    // `TCtrHistory` owns a `Vec<i64>`, so a `Vec<TCtrHistory>` costs one heap
    // allocation PER BUCKET plus a 24-byte Vec header each. Measured over a
    // 500k-bucket column that is 48.2 MB above baseline versus 28.6 MB for the
    // flat form — 1.7x the working set, and 500k mallocs instead of one. The
    // pre-existing `online_ctr_prefix_binclf` this function generalizes uses the
    // same flat shape (`Vec<[i64; SIMPLE_CLASSES_COUNT]>`); high memory efficiency
    // is a first-class constraint here (CLAUDE.md), so the generic form must not
    // regress it.
    let Some(counts_len) = bucket_count.checked_mul(classes) else {
        return Err(CbError::OutOfRange(
            "online_class_prefix_column: bucket_count * classes overflows".to_owned(),
        ));
    };
    let mut counts: Vec<i64> = vec![0; counts_len];

    let mut good = vec![0i64; n];
    let mut total = vec![0i64; n];
    let mut value = vec![0f64; n];

    for &doc_i in permutation {
        let doc = doc_i as usize;
        let Some(&bin) = bins.get(doc) else {
            return Err(CbError::Degenerate(
                "online_class_prefix_column: permutation index out of range for bins".to_owned(),
            ));
        };
        let Some(&class) = target_class.get(doc) else {
            return Err(CbError::Degenerate(
                "online_class_prefix_column: permutation index out of range for target_class"
                    .to_owned(),
            ));
        };
        let bucket = bin as usize;
        let start = bucket.saturating_mul(classes);
        let end = start.saturating_add(classes);

        // READ the prefix counts BEFORE incrementing, through the ONE generic
        // numerator rule — this function contains no numerator arithmetic of its own.
        let slots: &[i64] = counts.get(start..end).unwrap_or(&[]);
        let (num, denom) = online_class_prefix(slots, target_border_idx, ctr_type);
        if let Some(slot) = good.get_mut(doc) {
            *slot = num as i64;
        }
        if let Some(slot) = total.get_mut(doc) {
            *slot = denom;
        }
        if let Some(slot) = value.get_mut(doc) {
            *slot = calc_ctr_online(num, denom, prior);
        }

        // INCREMENT after read. `class < classes` is checked FIRST: without it a
        // malformed class index would land in the NEXT bucket's row rather than
        // being dropped, silently corrupting a different bucket's counts.
        if class < classes {
            if let Some(slot) = counts.get_mut(start.saturating_add(class)) {
                *slot += 1;
            }
        }
    }

    Ok(OnlineCtrPrefix { good, total, value })
}

/// The Counter per-document column: each document's WHOLE-SET bucket total, plus
/// the constant MAX bucket total that serves as the shared denominator
/// (SPEC-CTRT-08).
///
/// # Not a prefix
/// Counter is permutation-INdependent
/// (`IsPermutationDependentCtrType(Counter) == false`, `ctr_type.cpp:43-56`).
/// Every document sees its bucket's FULL count — **including its own row** — so
/// there is no read-before-increment and no leakage question. This function
/// therefore takes **no permutation parameter at all**: permutation invariance is
/// structural here rather than merely asserted.
///
/// # Upstream
/// `online_ctr.cpp:503-562` (the Counter column build) and `:934-936` (the
/// denominator = MAX bucket total, the same rule
/// [`crate::ctr::final_ctr::build_final_ctr`]'s Counter arm already applies to the
/// baked table).
///
/// # Not handled here
/// `counter_calc_method` — which widens the counted sample range from learn-only
/// to learn + every eval set (`online_ctr.cpp:714-729`) — is deliberately NOT a
/// parameter of this function. It lands in E22, once `EvalSet` can carry
/// categorical columns at all.
///
/// Returns `(per_document_bucket_total, max_bucket_total)`. An empty `bins`
/// yields `(vec![], 0)` — a zero denominator is returned plainly rather than
/// producing a division by zero downstream.
#[must_use]
pub fn online_counter_column(
    bins: &[u32],
    extra_bins: &[u32],
    bucket_count: usize,
) -> (Vec<i64>, i64) {
    let mut totals = vec![0i64; bucket_count];
    // `extra_bins` (E22 / SPEC-CTRT-17): the concatenated EVAL-set bins under
    // `counter_calc_method = Full` — upstream's `CountOnlineCTRTotal` sample
    // range spans the learn + every-test-set `hashArr` (`online_ctr.cpp:
    // 716-729`), so eval documents join the per-bucket tally AND the MAX
    // denominator. EMPTY under `SkipTest` — byte-identical to the learn-only
    // behavior. The per-document OUTPUT column below is indexed by the LEARN
    // `bins` only: eval documents never produce output rows.
    for &bin in bins.iter().chain(extra_bins) {
        if let Some(slot) = totals.get_mut(bin as usize) {
            *slot += 1;
        }
    }

    let denominator = totals.iter().copied().max().unwrap_or(0);

    let column = bins
        .iter()
        .map(|&bin| totals.get(bin as usize).copied().unwrap_or(0))
        .collect();

    (column, denominator)
}

/// The ONE generic classes-prefix producer: given a bucket's per-class prefix
/// `counts`, the target-border index `b` and the CTR type, return that bucket's
/// `(numerator, denominator)` pair (SPEC-CTRT-04).
///
/// # Upstream
/// `UpdateGoodCount` (`online_ctr.cpp:115-121`):
/// ```text
/// if (ctrType == Buckets) *goodCount = curCount; else *goodCount -= curCount;
/// ```
/// applied cumulatively over `border = 0..targetBorderCount` starting from
/// `goodCount = Total`. For a single border index `b` that collapses to:
///
/// - [`Buckets`](crate::ctr::ECtrType::Buckets) → `N[b]`
/// - everything else → `Total - Σ_{c ≤ b} N[c]`
///
/// The denominator is always the bucket total. The read-before-increment
/// ORDER (`online_ctr.cpp:168-185`) is the CALLER's responsibility — this
/// function is pure and sees only an already-read prefix.
///
/// # Not for `Counter`
/// [`Counter`](crate::ctr::ECtrType::Counter) is **not** a class-prefix type: its
/// numerator is the whole-set bucket total and its denominator is the MAX bucket
/// total, neither of which is derivable from one bucket's class counts. It must
/// never be passed here; `online_counter_column` owns that path.
///
/// # Safety of the arithmetic
/// Access is via checked `.get` only — an out-of-range `b` contributes `0`
/// rather than panicking — and the cumulative subtraction saturates, so a
/// malformed `counts` can never underflow.
#[must_use]
pub fn online_class_prefix(
    counts: &[i64],
    target_border_idx: usize,
    ctr_type: crate::ctr::ECtrType,
) -> (f64, i64) {
    let total: i64 = counts.iter().sum();

    let num = if matches!(ctr_type, crate::ctr::ECtrType::Buckets) {
        counts.get(target_border_idx).copied().unwrap_or(0)
    } else {
        let head: i64 = (0..=target_border_idx)
            .map(|c| counts.get(c).copied().unwrap_or(0))
            .sum();
        total.saturating_sub(head)
    };

    (num as f64, total)
}

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
        // READ the prefix counts BEFORE incrementing (online_ctr.cpp:303-304).
        // Routed through the ONE generic classes-prefix producer (E05/SPEC-CTRT-05):
        // binclf Borders IS `online_class_prefix(&[N0, N1], 0, Borders)`, proven
        // bit-identical over an exhaustive grid in `online_test.rs`.
        let slots: &[i64] = counts.get(bucket).map_or(&[][..], |e| &e[..]);
        let (g_f64, t) = online_class_prefix(slots, 0, crate::ctr::ECtrType::Borders);
        let g = g_f64 as i64; // good = N[1] (pos class)
        if let Some(slot) = good.get_mut(doc) {
            *slot = g;
        }
        if let Some(slot) = total.get_mut(doc) {
            *slot = t;
        }
        if let Some(slot) = value.get_mut(doc) {
            *slot = calc_ctr_online(g_f64, t, prior);
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
        // Re-routed through the SAME generic producer as `online_ctr_prefix_binclf`
        // above. Both loops MUST use it: this second derivation is independent, so
        // re-routing only one of the two would let them silently diverge (E05).
        let slots: &[i64] = counts.get(bucket).map_or(&[][..], |e| &e[..]);
        let (num, denom) = online_class_prefix(slots, 0, crate::ctr::ECtrType::Borders);
        step_num.push(num as i64);
        step_denom.push(denom);
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
