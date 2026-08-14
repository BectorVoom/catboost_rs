//! The `feature_border_type` binarizer family (`EBorderSelectionType`) — the
//! seven float-feature border-selection algorithms CatBoost exposes.
//!
//! # Source of truth
//!
//! `library/cpp/grid_creator/binarization.cpp` (v1.2.10), `MakeBinarizer`:
//!
//! | `EBorderSelectionType` | upstream binarizer | strategy | penalty |
//! |---|---|---|---|
//! | `GreedyLogSum` | `TGreedyBinarizer<MaxSumLog>` | greedy heap | `-log(w+1e-8)` |
//! | `GreedyMinEntropy` | `TGreedyBinarizer<MinEntropy>` | greedy heap | `w*log(w+1e-8)` |
//! | `MaxLogSum` | `TExactBinarizer<MaxSumLog>` | exact DP | `-log(w+1e-8)` |
//! | `MinEntropy` | `TExactBinarizer<MinEntropy>` | exact DP | `w*log(w+1e-8)` |
//! | `Median` | `TMedianBinarizer` | equal-frequency quantiles | — |
//! | `Uniform` | `TUniformBinarizer` | equal-width | — |
//! | `UniformAndQuantiles` | `TMedianPlusUniformBinarizer` | half of each | — |
//!
//! The greedy pair lives in [`crate::borders`] (it predates this module and is
//! the catboost default); this module adds the other five algorithms and the
//! dispatching entry point [`select_borders_f32`].
//!
//! # f32 / f64 discipline (RESEARCH Pitfall 2)
//!
//! Border VALUES are computed in `f32` throughout, matching upstream's `float`
//! feature storage; only penalty accumulators and DP costs are `f64`. The three
//! midpoint formulas are deliberately NOT interchangeable and are transcribed
//! verbatim, because each rounds differently:
//!
//! - greedy `LeftBorder`: `0.5f * a + 0.5f * b`  (two multiplies, then add)
//! - `RegularBorder`:     `(a + b) * .5f`        (add, then one multiply)
//! - exact-DP border:     `(a + b) / 2`          (add, then divide)
//!
//! # Limitation: `initialBorders` is not modelled
//!
//! Every upstream binarizer takes an optional `initialBorders` vector (the
//! `input_borders` parameter, which snaps generated borders onto a previously
//! saved grid). `input_borders` is not implemented in this crate, so the
//! `TMaybe<TVector<float>>` argument is always `Nothing()` here and the
//! corresponding branches are omitted rather than stubbed. When `input_borders`
//! lands it must be threaded through [`regular_border`] and [`exact_split`].

use crate::borders::{
    borders_from_values_with_penalty, finalize_border_set, gather_non_nan, sort_f32_ascending,
    PenaltyType, MAX_SUBSET_SIZE_FOR_BUILD_BORDERS,
};

/// The `feature_border_type` parameter (`EBorderSelectionType`, upstream
/// `enums.h`). The legal-value set is probed from the installed catboost 1.2.10
/// wheel, whose enum parser rejects an unknown token with:
/// "Valid options are: 'Median', 'GreedyLogSum', 'UniformAndQuantiles',
/// 'MinEntropy', 'MaxLogSum', 'Uniform', 'GreedyMinEntropy'."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EBorderSelectionType {
    /// Equal-FREQUENCY borders: the value at each `(i+1)/(n+1)` quantile
    /// position (`TMedianBinarizer`).
    Median,
    /// Greedy heap split under the `MaxSumLog` penalty — the catboost default
    /// and the only type implemented before the `feature_border_type` wave.
    #[default]
    GreedyLogSum,
    /// Half equal-frequency + half equal-WIDTH borders
    /// (`TMedianPlusUniformBinarizer`).
    UniformAndQuantiles,
    /// Exact dynamic-programming split under the `MinEntropy` penalty.
    MinEntropy,
    /// Exact dynamic-programming split under the `MaxSumLog` penalty.
    MaxLogSum,
    /// Equal-WIDTH borders between the column min and max (`TUniformBinarizer`).
    Uniform,
    /// Greedy heap split under the `MinEntropy` penalty.
    GreedyMinEntropy,
}

impl EBorderSelectionType {
    /// Parse the upstream spelling (exact, case-sensitive — matching the
    /// enum parser, which is case-sensitive).
    ///
    /// Returns `None` for any token outside the legal set; the caller turns that
    /// into a typed configuration error listing the legal values.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "Median" => Some(Self::Median),
            "GreedyLogSum" => Some(Self::GreedyLogSum),
            "UniformAndQuantiles" => Some(Self::UniformAndQuantiles),
            "MinEntropy" => Some(Self::MinEntropy),
            "MaxLogSum" => Some(Self::MaxLogSum),
            "Uniform" => Some(Self::Uniform),
            "GreedyMinEntropy" => Some(Self::GreedyMinEntropy),
            _ => None,
        }
    }

    /// The upstream spelling of this variant.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Median => "Median",
            Self::GreedyLogSum => "GreedyLogSum",
            Self::UniformAndQuantiles => "UniformAndQuantiles",
            Self::MinEntropy => "MinEntropy",
            Self::MaxLogSum => "MaxLogSum",
            Self::Uniform => "Uniform",
            Self::GreedyMinEntropy => "GreedyMinEntropy",
        }
    }

    /// Every legal value, in the order the wheel's enum parser lists them —
    /// used to build error messages and to drive the oracle matrix.
    #[must_use]
    pub fn all() -> [Self; 7] {
        [
            Self::Median,
            Self::GreedyLogSum,
            Self::UniformAndQuantiles,
            Self::MinEntropy,
            Self::MaxLogSum,
            Self::Uniform,
            Self::GreedyMinEntropy,
        ]
    }
}

/// Select up to `max_borders` borders for one already-narrowed `f32` feature
/// column under `border_type`.
///
/// Mirrors [`crate::select_borders_greedy_logsum_f32`] in every respect other
/// than the algorithm: over-cap columns are sub-sampled to
/// [`MAX_SUBSET_SIZE_FOR_BUILD_BORDERS`] with the same fixed-seed draw, NaNs are
/// dropped, and `nan_sentinel` optionally prepends the NanMode(`Min`)
/// `f32::MIN` sentinel. Passing [`EBorderSelectionType::GreedyLogSum`] routes to
/// the pre-existing greedy path and returns byte-identical borders.
#[must_use]
pub fn select_borders_f32(
    column: &[f32],
    max_borders: usize,
    border_type: EBorderSelectionType,
    nan_sentinel: bool,
) -> Vec<f64> {
    let values: Vec<f32> = if column.len() > MAX_SUBSET_SIZE_FOR_BUILD_BORDERS {
        let sample = crate::sample_indices_for_build_borders(
            column.len(),
            MAX_SUBSET_SIZE_FOR_BUILD_BORDERS,
        );
        gather_non_nan(column, &sample)
    } else {
        column.iter().copied().filter(|v| !v.is_nan()).collect()
    };
    select_borders_from_values(values, max_borders, border_type, nan_sentinel)
}

/// [`select_borders_f32`] over an `f64` column (narrowing to `f32` first, the
/// same `v as f32` every other entry point performs).
#[must_use]
pub fn select_borders(
    column: &[f64],
    max_borders: usize,
    border_type: EBorderSelectionType,
    nan_sentinel: bool,
) -> Vec<f64> {
    let narrowed: Vec<f32> = column.iter().map(|&v| v as f32).collect();
    select_borders_f32(&narrowed, max_borders, border_type, nan_sentinel)
}

/// Dispatch the (already sampled, already NaN-free) value multiset to its
/// binarizer and finalize the border set.
///
/// # The sentinel consumes one border of the budget
///
/// `border_count` is the TOTAL stored-border budget for the feature. When the
/// column carries NaNs, upstream spends one of those borders on the NanMode
/// sentinel and hands the binarizer only `border_count - 1` — so a NaN column at
/// `border_count = 8` gets 7 real borders plus the sentinel, not 8 plus the
/// sentinel. (`QuantizeParams` already documents this as "the binarizer reserves
/// one border for the sentinel internally"; nothing enforced it, because the
/// pre-existing fixture ran at a saturating budget where the reservation is
/// unobservable.)
fn select_borders_from_values(
    mut values: Vec<f32>,
    border_count: usize,
    border_type: EBorderSelectionType,
    nan_sentinel: bool,
) -> Vec<f64> {
    let max_borders = crate::borders::real_border_budget(border_count, nan_sentinel);
    match border_type {
        EBorderSelectionType::GreedyLogSum => borders_from_values_with_penalty(
            &mut values,
            max_borders,
            nan_sentinel,
            PenaltyType::MaxSumLog,
        ),
        EBorderSelectionType::GreedyMinEntropy => borders_from_values_with_penalty(
            &mut values,
            max_borders,
            nan_sentinel,
            PenaltyType::MinEntropy,
        ),
        EBorderSelectionType::Uniform => {
            // TUniformBinarizer reads only the min and max, and does NOT sort.
            let borders = uniform_borders(&values, max_borders);
            finalize_border_set(borders, nan_sentinel)
        }
        EBorderSelectionType::Median => {
            sort_f32_ascending(&mut values);
            let borders = generate_median_borders(&values, max_borders);
            finalize_border_set(borders, nan_sentinel)
        }
        EBorderSelectionType::UniformAndQuantiles => {
            sort_f32_ascending(&mut values);
            let borders = median_plus_uniform_borders(&values, max_borders);
            finalize_border_set(borders, nan_sentinel)
        }
        EBorderSelectionType::MaxLogSum => {
            let borders = exact_borders(&mut values, max_borders, PenaltyType::MaxSumLog);
            finalize_border_set(borders, nan_sentinel)
        }
        EBorderSelectionType::MinEntropy => {
            let borders = exact_borders(&mut values, max_borders, PenaltyType::MinEntropy);
            finalize_border_set(borders, nan_sentinel)
        }
    }
}

/// `TUniformBinarizer::BestSplit` — equal-WIDTH borders.
///
/// ```text
/// auto [minIter, maxIter] = MinMaxElement(values.begin(), values.end());
/// if (minValue == maxValue) return TQuantization();
/// for (int i = 0; i < maxBordersCount; ++i) {
///     double currentValue = minValue + (i + 1) * (maxValue - minValue) / (maxBordersCount + 1);
///     borders.insert(currentValue);
/// }
/// ```
///
/// The right-hand side is entirely `float` / `int` in C++, so the arithmetic is
/// performed in `f32` and only the (discarded) assignment widens — reproduced
/// here as `f32` arithmetic. A degenerate (constant) column yields no borders.
fn uniform_borders(values: &[f32], max_borders: usize) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for &v in values {
        if v < min_value {
            min_value = v;
        }
        if v > max_value {
            max_value = v;
        }
    }
    if min_value == max_value {
        return Vec::new();
    }
    let mut out: Vec<f32> = Vec::with_capacity(max_borders);
    for i in 0..max_borders {
        // minValue + (i + 1) * (maxValue - minValue) / (maxBordersCount + 1)
        let step = (i + 1) as f32 * (max_value - min_value);
        out.push(min_value + step / (max_borders + 1) as f32);
    }
    out
}

/// `GenerateMedianBorders` — equal-FREQUENCY (quantile) borders over the SORTED
/// value vector (duplicates included).
///
/// ```text
/// ui64 total = featureValues.size();
/// if (total == 0 || featureValues.front() == featureValues.back()) return {};
/// for (int i = 0; i < maxBordersCount; ++i) {
///     ui64 i1 = (i + 1) * total / (maxBordersCount + 1);
///     i1 = Min(i1, total - 1);
///     float val1 = featureValues[i1];
///     if (val1 != featureValues[0]) result.insert(RegularBorder(val1, featureValues, ...));
/// }
/// ```
///
/// The index arithmetic is exact `u64` integer math (NOT float quantiles).
fn generate_median_borders(sorted: &[f32], max_borders: usize) -> Vec<f32> {
    let total = sorted.len() as u64;
    let (Some(&first), Some(&last)) = (sorted.first(), sorted.last()) else {
        return Vec::new();
    };
    if total == 0 || first == last {
        return Vec::new();
    }
    let mut out: Vec<f32> = Vec::with_capacity(max_borders);
    for i in 0..max_borders {
        // ui64 i1 = (i + 1) * total / (maxBordersCount + 1);
        let mut i1 = (i as u64 + 1) * total / (max_borders as u64 + 1);
        i1 = i1.min(total - 1);
        let Some(&val1) = sorted.get(i1 as usize) else {
            continue;
        };
        if val1 != first {
            out.push(regular_border(val1, sorted));
        }
    }
    out
}

/// `TMedianPlusUniformBinarizer::BestSplit` — `maxBorders - maxBorders/2`
/// quantile borders PLUS `maxBorders/2` equal-width ones.
///
/// ```text
/// int halfBorders = maxBordersCount / 2;
/// borders = GenerateMedianBorders(values, ..., maxBordersCount - halfBorders);
/// float minValue = values.front(), maxValue = values.back();
/// for (int i = 0; i < halfBorders; ++i) {
///     float val = minValue + (i + 1) * (maxValue - minValue) / (halfBorders + 1);
///     borders.insert(RegularBorder(val, values, ...));
/// }
/// ```
///
/// Note the uniform half here is SNAPPED onto the data through
/// [`regular_border`] (unlike [`uniform_borders`], which emits the raw
/// equal-width value), and that both halves land in the same dedup set — which
/// is why this type routinely returns FEWER than `max_borders` borders.
fn median_plus_uniform_borders(sorted: &[f32], max_borders: usize) -> Vec<f32> {
    let (Some(&min_value), Some(&max_value)) = (sorted.first(), sorted.last()) else {
        return Vec::new();
    };
    if min_value == max_value {
        return Vec::new();
    }
    let half_borders = max_borders / 2;
    let mut out = generate_median_borders(sorted, max_borders - half_borders);
    for i in 0..half_borders {
        // float val = minValue + (i + 1) * (maxValue - minValue) / (halfBorders + 1);
        let step = (i + 1) as f32 * (max_value - min_value);
        let val = min_value + step / (half_borders + 1) as f32;
        out.push(regular_border(val, sorted));
    }
    out
}

/// `RegularBorder(border, sortedValues, initialBorders)` — snap a candidate
/// border onto the midpoint of the straddling pair of observed values.
///
/// ```text
/// lowerBound = LowerBound(sortedValues.begin(), sortedValues.end(), border);
/// if (lowerBound == end)   return Max(2.f * back,  back  + 1.f);   // always false
/// if (lowerBound == begin) return Min(.5f * front, 2.f * front);   // always true
/// float res = (lowerBound[0] + lowerBound[-1]) * .5f;
/// if (res == lowerBound[0]) res = lowerBound[-1];  // wrong-side rounding
/// return res;
/// ```
///
/// The `initialBorders` branches are omitted — see the module doc.
fn regular_border(border: f32, sorted: &[f32]) -> f32 {
    let (Some(&front), Some(&back)) = (sorted.first(), sorted.last()) else {
        return border;
    };
    // LowerBound: first index whose value is >= border.
    let idx = sorted.partition_point(|&v| v < border);
    if idx == sorted.len() {
        // Binarizing to always false.
        return (2.0_f32 * back).max(back + 1.0_f32);
    }
    if idx == 0 {
        // Binarizing to always true.
        return (0.5_f32 * front).min(2.0_f32 * front);
    }
    let hi = sorted.get(idx).copied().unwrap_or(back);
    let lo = sorted.get(idx - 1).copied().unwrap_or(front);
    // float res = (lowerBound[0] + lowerBound[-1]) * .5f;
    let mut res = (hi + lo) * 0.5_f32;
    if res == hi {
        // Wrong side rounding (should be very scarce).
        res = lo;
    }
    res
}

/// `TExactBinarizer<Penalty>::BestSplit` — the guaranteed-optimum binarizer.
///
/// Groups the sorted values into UNIQUE values with their object counts
/// (`GroupAndSortValues`), runs the dynamic program over those weights, then
/// emits `(values[t] + values[t+1]) / 2` for every threshold `t` that is not the
/// last unique value.
fn exact_borders(values: &mut [f32], max_borders: usize, penalty: PenaltyType) -> Vec<f32> {
    sort_f32_ascending(values);
    let (unique, weights) = group_and_sort_values(values);
    let thresholds = exact_split(&weights, max_borders, penalty);

    let mut out: Vec<f32> = Vec::with_capacity(thresholds.len());
    for t in thresholds {
        // if (t + 1 != values.size()) borders.insert((values[t] + values[t + 1]) / 2);
        if t + 1 != unique.len() {
            let (Some(&a), Some(&b)) = (unique.get(t), unique.get(t + 1)) else {
                continue;
            };
            out.push((a + b) / 2.0_f32);
        }
    }
    out
}

/// `GroupAndSortValues` over an ALREADY-sorted value slice: collapse runs of
/// equal values into (unique value, object count) pairs. Counts are the bin
/// weights the exact DP minimizes over.
fn group_and_sort_values(sorted: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut unique: Vec<f32> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    for &v in sorted {
        if unique.last().copied() == Some(v) {
            if let Some(w) = weights.last_mut() {
                *w += 1.0;
            }
        } else {
            unique.push(v);
            weights.push(1.0);
        }
    }
    (unique, weights)
}

/// The exact dynamic program (`BestSplit(weights, maxBordersCount, thresholds,
/// mode)`) returning the optimal bin-END indices into the unique-value vector.
///
/// # The recurrence
///
/// With `bins = max_borders + 1` and `sweights` the inclusive prefix sums of the
/// per-unique-value weights, upstream stores layer `l` (a partition into `l + 2`
/// bins) in a COMPRESSED index `j` that denotes absolute end position `l + 1 + j`
/// — which is what the trailing `thresholds[l] += l` fixup undoes:
///
/// ```text
/// error_0[i]  = Penalty(sweights[i])                                  // one bin, ends at i
/// error_l[j]  = min over i <= j of  error_{l-1}[i]
///                 + Penalty(sweights[l + 1 + j] - sweights[l + i])
/// ```
///
/// # Why divide-and-conquer is exact here
///
/// Upstream's `E_RLM2` mode is an ACCELERATION of the same recurrence, not a
/// different objective: it exploits the Monge/quadrangle property to avoid the
/// O(n^2) scan. Both penalties are convex in the bin weight, so
/// `cost(i, j) = Penalty(sweights[j'] - sweights[i'])` satisfies the quadrangle
/// inequality and the optimal `i` is non-decreasing in `j`. This implementation
/// uses the standard divide-and-conquer optimization over that monotonicity,
/// giving the identical optimum in `O(dsize * log(dsize))` per layer.
///
/// # Tie-breaking is ASYMMETRIC, and it decides the answer
///
/// With unit object weights the optimum is massively degenerate: partitioning
/// `n` equal-weight values into `k` bins costs the same for ANY arrangement of
/// the `floor(n/k)` / `ceil(n/k)` bin sizes, so which optimal partition comes
/// back is decided purely by tie-breaking. Upstream uses two DIFFERENT rules:
///
/// - the per-layer scan (`E_Base` line 248, `E_Base2` line 277) compares
///   `if (newError <= bestError)` — NON-strict, so scanning `i` ascending the
///   LARGEST minimizing index wins;
/// - the final match (line 640 / 649) compares `if (newError < bestError)` —
///   strict, so there the SMALLEST minimizing index wins.
///
/// Using one rule for both (the obvious reading) reproduces neither catboost's
/// first border nor its later ones.
fn exact_split(weights: &[f32], max_borders: usize, penalty: PenaltyType) -> Vec<usize> {
    let wsize = weights.len();
    let bins = max_borders + 1;
    if bins <= 1 || wsize <= 1 {
        return Vec::new();
    }
    // At or past saturation every adjacent unique-value boundary is a border,
    // and the DP's compressed index range (wsize - bins + 1) would be empty.
    if bins >= wsize {
        return (0..wsize - 1).collect();
    }

    // Inclusive prefix sums, accumulated in f32 exactly as upstream's
    // `TVector<TWeightType> sweights(weights)` with TWeightType = float. Object
    // counts stay well inside f32's exact-integer range (2^24), so this is exact
    // for the unweighted path while remaining faithful for a weighted one.
    let mut sweights: Vec<f32> = Vec::with_capacity(wsize);
    let mut running = 0.0_f32;
    for &w in weights {
        running += w;
        sweights.push(running);
    }
    let sw = |i: usize| -> f64 { f64::from(sweights.get(i).copied().unwrap_or(0.0)) };

    let dsize = wsize - bins + 1;
    // Layer 0: a single bin covering [0..=i].
    let mut prev: Vec<f64> = (0..dsize).map(|i| penalty.apply(sw(i))).collect();
    let mut cur: Vec<f64> = vec![0.0; dsize];
    let mut best_solutions: Vec<Vec<u32>> = Vec::with_capacity(bins.saturating_sub(2));
    // The two-loop scratch (`bs1`/`bs2`, `e1`/`e2`) E_RLM2 keeps across its
    // forward / inverted passes (binarization.cpp:236-237).
    let mut bs1: Vec<usize> = vec![0; dsize];
    let mut bs2: Vec<usize> = vec![0; dsize];
    let mut e1: Vec<f64> = vec![0.0; dsize];
    let mut e2: Vec<f64> = vec![0.0; dsize];

    for l in 0..bins - 2 {
        let mut argmin: Vec<u32> = vec![0; dsize];
        rlm2_layer(
            l, dsize, &prev, &sweights, penalty, &mut bs1, &mut bs2, &mut e1, &mut e2, &mut argmin,
            &mut cur,
        );
        best_solutions.push(argmin);
        std::mem::swap(&mut prev, &mut cur);
    }

    // Traceback. `prev` now holds the final layer (bins - 1 bins placed); the
    // last bin closes at the final position, i.e. compressed j = dsize - 1.
    let mut thresholds: Vec<usize> = vec![0; bins - 1];
    let l = bins - 2;
    let j = dsize - 1;
    let mut best_index = 0usize;
    let mut best_error = prev.first().copied().unwrap_or(f64::INFINITY)
        + penalty.apply(sw(l + j + 1) - sw(l));
    for i in 1..=j {
        let new_error = prev.get(i).copied().unwrap_or(f64::INFINITY)
            + penalty.apply(sw(l + j + 1) - sw(l + i));
        if new_error < best_error {
            best_error = new_error;
            best_index = i;
        }
    }
    if let Some(slot) = thresholds.get_mut(bins - 2) {
        *slot = best_index;
    }
    for layer in (1..=l).rev() {
        best_index = best_solutions
            .get(layer - 1)
            .and_then(|row| row.get(best_index))
            .map_or(0, |&v| v as usize);
        if let Some(slot) = thresholds.get_mut(layer - 1) {
            *slot = best_index;
        }
    }

    // Undo the compressed indexing: thresholds[l] += l.
    for (idx, t) in thresholds.iter_mut().enumerate() {
        *t += idx;
    }
    thresholds
}

/// Divide-and-conquer optimization of one DP layer over a monotone argmin.
///
/// The `Eps` slack E_RLM2 compares errors with (`binarization.cpp:219`). It is
/// NOT a rounding guard that could be dropped: the optimum is degenerate under
/// unit weights, and `Eps` is what decides which of the equally-good partitions
/// each monotone scan settles on. Changing it changes the emitted borders.
const RLM2_EPS: f64 = 1e-12;

/// One layer of upstream's `E_RLM2` dynamic program
/// (`binarization.cpp:450-592`), transcribed loop-for-loop.
///
/// `E_RLM2` computes the same recurrence as the reference `E_Base2` branch but
/// reaches it through a forward pass and an inverted pass that each advance a
/// MONOTONE pointer, then repairs any interval where the two disagree
/// (`while (bs1[k] + 1 < bs2[k])`). That is what makes it "almost always
/// `O(wsize * bins)`" instead of quadratic.
///
/// It is transcribed rather than replaced by a cleaner equivalent (e.g. a
/// divide-and-conquer Monge optimization) because the two are NOT
/// interchangeable in this application: they agree on the optimal COST but not
/// on WHICH optimal partition they return, and with unit weights essentially
/// every partition into equal-size bins is optimal. A D&C implementation with
/// upstream's `<=` tie rule matches catboost's first few borders and then
/// diverges — the Eps-slack scans below are the actual selection rule.
#[allow(clippy::too_many_arguments)]
fn rlm2_layer(
    l: usize,
    dsize: usize,
    prev: &[f64],
    sweights: &[f32],
    penalty: PenaltyType,
    bs1: &mut [usize],
    bs2: &mut [usize],
    e1: &mut [f64],
    e2: &mut [f64],
    argmin: &mut [u32],
    cur: &mut [f64],
) {
    let sw = |i: usize| -> f64 { f64::from(sweights.get(i).copied().unwrap_or(0.0)) };
    let pv = |i: usize| -> f64 { prev.get(i).copied().unwrap_or(f64::INFINITY) };
    // prevError[i] + Penalty(sweights[l + j + 1] - sweights[l + i])
    let cost = |j: usize, i: usize| -> f64 { pv(i) + penalty.apply(sw(l + j + 1) - sw(l + i)) };
    // The inverted pass's mirrored cost:
    // prevError[dsize-i-1] + Penalty(sweights[l+dsize-j] - sweights[l+dsize-i-1])
    let icost = |j: usize, i: usize| -> f64 {
        let back = dsize.saturating_sub(i + 1);
        pv(back) + penalty.apply(sw(l + dsize - j) - sw(l + back))
    };
    let get = |v: &[usize], i: usize| -> usize { v.get(i).copied().unwrap_or(0) };
    let getf = |v: &[f64], i: usize| -> f64 { v.get(i).copied().unwrap_or(f64::INFINITY) };
    let set = |v: &mut [usize], i: usize, x: usize| {
        if let Some(s) = v.get_mut(i) {
            *s = x;
        }
    };
    let setf = |v: &mut [f64], i: usize, x: f64| {
        if let Some(s) = v.get_mut(i) {
            *s = x;
        }
    };

    // ---- First forward loop (binarization.cpp:451-467) ----
    {
        let mut i = 0usize;
        for j in 0..dsize {
            let mut best_error = cost(j, i);
            i += 1;
            while i <= j {
                let new_error = cost(j, i);
                if new_error > best_error + RLM2_EPS {
                    break;
                }
                best_error = new_error;
                i += 1;
            }
            i = i.saturating_sub(1);
            set(bs1, j, i);
            setf(e1, j, best_error);
        }
    }

    // ---- First inverted loop (binarization.cpp:469-504) ----
    {
        let mut i = 0usize;
        for j in 0..dsize {
            i = i.max(j);
            let back_j = dsize - j - 1;
            let maxi = dsize - get(bs1, back_j) - 1;
            if i + 1 >= maxi {
                set(bs2, back_j, get(bs1, back_j));
                setf(e2, back_j, getf(e1, back_j));
                i = maxi;
                continue;
            }
            let mut best_error = getf(e1, back_j);
            while i + 1 < maxi {
                let new_error = icost(j, i);
                if new_error + RLM2_EPS < best_error {
                    best_error = new_error;
                    break;
                }
                i += 1;
            }
            if i + 1 >= maxi {
                i = maxi;
            } else {
                i += 1;
                while i + 1 < maxi {
                    let new_error = icost(j, i);
                    if new_error > best_error + RLM2_EPS {
                        break;
                    }
                    best_error = new_error;
                    i += 1;
                }
                i = i.saturating_sub(1);
            }
            set(bs2, back_j, dsize - i - 1);
            setf(e2, back_j, best_error);
        }
    }

    // ---- Repair pass (binarization.cpp:506-592) ----
    for k in 0..dsize {
        while get(bs1, k) + 1 < get(bs2, k) {
            // Rebuild required.
            let mut maxj = dsize;

            // Forward loop (binarization.cpp:511-549).
            {
                let mut i = get(bs1, k) + 2;
                for j in k..maxj {
                    if i <= get(bs1, j) {
                        maxj = j;
                        break;
                    }
                    let maxi = get(bs2, j);
                    if i + 1 >= maxi {
                        i = maxi;
                        set(bs1, j, i);
                        setf(e1, j, getf(e2, j));
                        continue;
                    }
                    let mut best_error = getf(e2, j);
                    while i + 1 < maxi {
                        let new_error = cost(j, i);
                        if new_error + RLM2_EPS < best_error {
                            best_error = new_error;
                            break;
                        }
                        i += 1;
                    }
                    if i + 1 >= maxi {
                        i = maxi;
                    } else {
                        i += 1;
                        while i + 1 < maxi {
                            let new_error = cost(j, i);
                            if new_error > best_error + RLM2_EPS {
                                break;
                            }
                            best_error = new_error;
                            i += 1;
                        }
                        i = i.saturating_sub(1);
                    }
                    set(bs1, j, i);
                    setf(e1, j, best_error);
                }
            }

            // Inverted loop (binarization.cpp:551-587).
            {
                let j1 = dsize - maxj;
                let j2 = dsize - k;
                let mut i = dsize - get(bs2, dsize.saturating_sub(j1 + 1)) - 1 + 2;
                for j in j1..j2 {
                    let back_j = dsize.saturating_sub(j + 1);
                    let maxi = dsize - get(bs1, back_j) - 1;
                    if i + 1 >= maxi {
                        set(bs2, back_j, get(bs1, back_j));
                        setf(e2, back_j, getf(e1, back_j));
                        i = maxi;
                        continue;
                    }
                    let mut best_error = getf(e1, back_j);
                    while i + 1 < maxi {
                        let new_error = icost(j, i);
                        if new_error + RLM2_EPS < best_error {
                            best_error = new_error;
                            break;
                        }
                        i += 1;
                    }
                    if i + 1 >= maxi {
                        i = maxi;
                    } else {
                        i += 1;
                        while i + 1 < maxi {
                            let new_error = icost(j, i);
                            if new_error > best_error + RLM2_EPS {
                                break;
                            }
                            best_error = new_error;
                            i += 1;
                        }
                        i = i.saturating_sub(1);
                    }
                    set(bs2, back_j, dsize - i - 1);
                    setf(e2, back_j, best_error);
                }
            }
        }
        // Everything is fine now!
        if let Some(slot) = argmin.get_mut(k) {
            *slot = get(bs1, k) as u32;
        }
        setf(cur, k, getf(e1, k));
    }
}
