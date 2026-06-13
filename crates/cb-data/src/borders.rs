//! GreedyLogSum greedy binarizer (DATA-03) — the parity-critical float-feature
//! border selector, transcribed bit-for-bit from upstream CatBoost.
//!
//! # Source of truth
//!
//! `library/cpp/grid_creator/binarization.cpp`:
//! - `Penalty<EPenaltyType::MaxSumLog>(w) = -log(w + 1e-8)` (line 178-181).
//! - `TFeatureBin` / `IFeatureBin` greedy split over the **sorted object values**
//!   (lines 1320-1425): each bin's best split is found by probing the
//!   lower/upper bound of the bin's middle value; the split score is
//!   `-Penalty(left) - Penalty(right) + Penalty(total)` over object counts
//!   (`CalcSplitScore`, line 1398-1407).
//! - `GreedySplit` (lines 1499-1520): a max-heap of bins keyed by best-split
//!   score; while `splits.size() <= maxBorders && top.CanSplit()`, pop the
//!   top, split it, push both halves; then collect `LeftBorder` of every
//!   non-first bin.
//! - `LeftBorder` (line 1357-1371): the border value is computed in **f32** as
//!   `0.5f * values[start-1] + 0.5f * values[start]`.
//! - Borders are collected into a `THashSet<float>` then sorted ascending; the
//!   IEEE `-0.0f` is normalized to `+0.0f`.
//!
//! # f64 vs f32 discipline (RESEARCH Pitfall 2)
//!
//! Penalty accumulators and split scores are computed in `f64`; border *values*
//! are computed in `f32` (then widened to `f64` for the oracle comparison).
//! Mixing these silently shifts every downstream bin boundary (threat T-02-06).
//!
//! # Summation routing (D-07 / D-08)
//!
//! Object counts are exact small integers, so the per-bin penalty argument is an
//! exact `count as f64`; there is no float *summation* of weights in the
//! unweighted path. Where this module does fold floats, it routes through the
//! sanctioned reduction primitive [`cb_core::sum_f64`] rather than any raw
//! fold — see [`total_object_weight`].

use cb_core::sum_f64;

/// The `MaxSumLog` penalty: `-log(count + 1e-8)` (binarization.cpp:180). The
/// `1e-8` epsilon guards `log(0)` for an empty side. Computed in `f64`.
#[must_use]
pub fn penalty_maxsumlog(count: f64) -> f64 {
    // double Penalty<EPenaltyType::MaxSumLog>(double weight) { return -log(weight + 1e-8); }
    -(count + 1e-8).ln()
}

/// Total object weight over a slice of per-object weights, routed through the
/// sanctioned reduction primitive ([`cb_core::sum_f64`]) so this module never
/// spells a raw float fold (D-07 / D-08). In the unweighted
/// path every object weight is `1.0`, so this returns the object count as an
/// `f64` — but the *summation order* still flows through the audited primitive,
/// matching upstream's `double` weight accumulation
/// (binarization.cpp:803-815).
#[must_use]
fn total_object_weight(weights: &[f64]) -> f64 {
    sum_f64(weights)
}

/// One greedy feature bin over `values[start..end]` (`TFeatureBin`).
///
/// `values` is the full **sorted** object-value slice; `start`/`end` are indices
/// into it. `best_split` and `best_score` cache the best probe point found by
/// [`Bin::update_best_split`].
struct Bin {
    start: usize,
    end: usize,
    best_split: usize,
    best_score: f64,
}

impl Bin {
    /// `TFeatureBin(binStart, binEnd, featuresStart)` — construct then
    /// immediately compute the best split (`UpdateBestSplitProperties`).
    fn new(values: &[f32], start: usize, end: usize) -> Self {
        let mut bin = Self {
            start,
            end,
            best_split: start,
            best_score: 0.0,
        };
        bin.update_best_split(values);
        bin
    }

    /// `IFeatureBin::CanSplit`: a real split was found strictly inside the bin.
    fn can_split(&self) -> bool {
        self.start != self.best_split && self.end != self.best_split
    }

    /// `IFeatureBin::IsFirst`: this bin starts at index 0, so it has no left
    /// border to emit.
    fn is_first(&self) -> bool {
        self.start == 0
    }

    /// `IFeatureBin::CalcSplitScore` (binarization.cpp:1398-1407). Counts are bin
    /// object counts cast to `f64`; the score is
    /// `-Penalty(left) - Penalty(right) + Penalty(total)`. A split at the bin
    /// boundary is `-inf` (never chosen).
    fn calc_split_score(&self, split_pos: usize) -> f64 {
        if split_pos == self.start || split_pos == self.end {
            return f64::NEG_INFINITY;
        }
        // leftPartScore = -Penalty(splitPos - BinStart);
        let left = -penalty_maxsumlog((split_pos - self.start) as f64);
        // rightPartScore = -Penalty(BinEnd - splitPos);
        let right = -penalty_maxsumlog((self.end - split_pos) as f64);
        // currBinScore = -Penalty(BinEnd - BinStart);
        let curr = -penalty_maxsumlog((self.end - self.start) as f64);
        // return leftPartScore + rightPartScore - currBinScore;
        left + right - curr
    }

    /// `TFeatureBin::UpdateBestSplitProperties` (binarization.cpp:1409-1424):
    /// probe the lower bound (in `[start, mid)`) and upper bound (in
    /// `[mid, end)`) of the middle value, keep the higher-scoring of the two
    /// (ties favor the lower bound, matching `scoreLeft >= scoreRight`).
    fn update_best_split(&mut self, values: &[f32]) {
        // const int mid = BinStart + (BinEnd - BinStart) / 2;
        let mid = self.start + (self.end - self.start) / 2;
        // float midValue = *(FeaturesStart + mid);
        let mid_value = values.get(mid).copied().unwrap_or(f32::NAN);

        // lb = LowerBound(FeaturesStart + BinStart, FeaturesStart + mid, midValue)
        let lb = lower_bound(values, self.start, mid, mid_value);
        // ub = UpperBound(FeaturesStart + mid, FeaturesStart + BinEnd, midValue)
        let ub = upper_bound(values, mid, self.end, mid_value);

        let score_left = self.calc_split_score(lb);
        let score_right = self.calc_split_score(ub);
        // BestSplit = scoreLeft >= scoreRight ? lb : ub;
        if score_left >= score_right {
            self.best_split = lb;
            self.best_score = score_left;
        } else {
            self.best_split = ub;
            self.best_score = score_right;
        }
    }

    /// `TFeatureBin::Split` (binarization.cpp:1387-1395): carve off the left half
    /// `[start, best_split)` as a new bin, advance this bin's start to
    /// `best_split`, recompute its best split, and return the left bin.
    fn split(&mut self, values: &[f32]) -> Self {
        let left = Self::new(values, self.start, self.best_split);
        self.start = self.best_split;
        self.update_best_split(values);
        left
    }

    /// `IFeatureBin::LeftBorder` for a non-first bin (binarization.cpp:1368-1370):
    /// the border value in **f32**, `0.5f * values[start-1] + 0.5f * values[start]`.
    fn left_border(&self, values: &[f32]) -> f32 {
        let prev = values.get(self.start - 1).copied().unwrap_or(f32::NAN);
        let cur = values.get(self.start).copied().unwrap_or(f32::NAN);
        // float borderValue = 0.5f * (*(FeaturesStart + BinStart - 1));
        // borderValue += 0.5f * (*(FeaturesStart + BinStart));
        let mut border = 0.5_f32 * prev;
        border += 0.5_f32 * cur;
        border
    }
}

/// `LowerBound` over `values[lo..hi]` for `target` (first index whose value is
/// `>= target`). Returns an index in `[lo, hi]`.
fn lower_bound(values: &[f32], lo: usize, hi: usize, target: f32) -> usize {
    let mut left = lo;
    let mut right = hi;
    while left < right {
        let mid = left + (right - left) / 2;
        if values.get(mid).copied().unwrap_or(f32::NAN) < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// `UpperBound` over `values[lo..hi]` for `target` (first index whose value is
/// `> target`). Returns an index in `[lo, hi]`.
fn upper_bound(values: &[f32], lo: usize, hi: usize, target: f32) -> usize {
    let mut left = lo;
    let mut right = hi;
    while left < right {
        let mid = left + (right - left) / 2;
        if values.get(mid).copied().unwrap_or(f32::NAN) <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// Select up to `max_borders` GreedyLogSum borders for one float feature column.
///
/// `column` is the raw (unsorted, possibly NaN-containing) object values for a
/// single feature, as `f64` (the [`crate::Pool`] storage type). The values are
/// narrowed to `f32` (CatBoost's feature storage type), NaNs are filtered out,
/// and the result is the sorted-ascending border set, each border widened back
/// to `f64` for oracle comparison.
///
/// `nan_sentinel` optionally prepends the NanMode `f32::MIN` sentinel border at
/// index 0 (mirroring upstream's stored-border NaN handling under
/// `nan_mode=Min`); pass `None` for a NaN-free / `nan_mode=Max` feature. The
/// caller decides per-feature whether the sentinel is present (it is
/// config-dependent — see `borders_quant/config.json`, A1/A3).
#[must_use]
pub fn select_borders_greedy_logsum(
    column: &[f64],
    max_borders: usize,
    nan_sentinel: bool,
) -> Vec<f64> {
    // Narrow to f32 (feature storage type), drop NaNs, sort ascending.
    let mut values: Vec<f32> = column
        .iter()
        .map(|&v| v as f32)
        .filter(|v| !v.is_nan())
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Each non-NaN object carries unit weight; the total weight is accumulated
    // through the audited reduction primitive (D-07) — for the unweighted path
    // this equals the object count, but the routing is the parity contract.
    let unit_weights = vec![1.0_f64; values.len()];
    let total_weight = total_object_weight(&unit_weights);
    let borders_f32 = greedy_split(&values, max_borders, total_weight);

    // THashSet<float> -> Sort ascending, normalizing -0.0f to +0.0f.
    let mut sorted: Vec<f32> = borders_f32
        .into_iter()
        .map(|b| if b == 0.0_f32 { 0.0_f32 } else { b })
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup();

    let mut out: Vec<f64> = Vec::with_capacity(sorted.len() + usize::from(nan_sentinel));
    if nan_sentinel {
        // NanMode(Min) sentinel: numeric_limits<float>::lowest() prepended.
        out.push(f64::from(f32::MIN));
    }
    out.extend(sorted.into_iter().map(f64::from));
    out
}

/// `GreedySplit` (binarization.cpp:1499-1520): a max-heap of bins keyed by best
/// split score. While the heap holds `<= max_borders` bins and the top bin can
/// split, pop it, split it, and push both halves; then collect the left border
/// of every non-first bin into a dedup set.
fn greedy_split(values: &[f32], max_borders: usize, total_weight: f64) -> Vec<f32> {
    // total_weight equals values.len() in the unweighted path; assert the
    // reduction routed through cb_core agrees with the slice length so a future
    // weighted variant cannot silently desync the count.
    debug_assert_eq!(total_weight as usize, values.len());
    let _ = total_weight;
    if values.len() < 2 {
        return Vec::new();
    }

    // std::priority_queue<TBinType> is a max-heap on Score(). We model it as a
    // Vec scanned for the max each iteration; the bin counts here are tiny
    // (<= max_borders + 1), so a linear scan is fine and avoids the Ord/NaN
    // hazards of BinaryHeap on f64 scores.
    let mut bins: Vec<Bin> = vec![Bin::new(values, 0, values.len())];

    // while (splits.size() <= maxBordersCount && splits.top().CanSplit())
    while bins.len() <= max_borders {
        // Find the current top (max best_score).
        let Some(top_idx) = arg_max_score(&bins) else {
            break;
        };
        if !bins.get(top_idx).map(Bin::can_split).unwrap_or(false) {
            break;
        }
        // pop top, split, push left + the mutated top.
        if let Some(mut top) = pop_at(&mut bins, top_idx) {
            let left = top.split(values);
            bins.push(left);
            bins.push(top);
        } else {
            break;
        }
    }

    // Collect LeftBorder of every non-first bin.
    let mut borders: Vec<f32> = Vec::with_capacity(bins.len());
    for bin in &bins {
        if !bin.is_first() {
            borders.push(bin.left_border(values));
        }
    }
    borders
}

/// Index of the bin with the maximum `best_score` (the priority-queue top).
/// Ties keep the first occurrence, mirroring a stable max selection.
fn arg_max_score(bins: &[Bin]) -> Option<usize> {
    let mut best_idx: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, bin) in bins.iter().enumerate() {
        if best_idx.is_none() || bin.best_score > best_score {
            best_idx = Some(idx);
            best_score = bin.best_score;
        }
    }
    best_idx
}

/// Remove and return the bin at `index` (`swap_remove`-free to avoid disturbing
/// the relative order, which keeps tie-breaking deterministic).
fn pop_at(bins: &mut Vec<Bin>, index: usize) -> Option<Bin> {
    if index < bins.len() {
        Some(bins.remove(index))
    } else {
        None
    }
}

