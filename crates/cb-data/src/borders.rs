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
    // Narrow to f32 (feature storage type), drop NaNs, sort ascending. The UNSTABLE
    // sort is byte-identical here: keys equal under the comparator are either
    // bit-identical f32s or the {-0.0, +0.0} pair, and every downstream consumer is
    // order-insensitive across equal keys — `left_border`'s midpoint of any {-0.0, +0.0}
    // adjacency is +0.0 in either order, the greedy split scores depend only on the
    // sorted VALUES at each index, and the emitted border set normalizes -0.0 to +0.0
    // before dedup. pdqsort avoids the stable merge sort's O(n/2) allocation on the hot
    // fit-prep path.
    let mut values: Vec<f32> = column
        .iter()
        .map(|&v| v as f32)
        .filter(|v| !v.is_nan())
        .collect();
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
///
/// # Heap tie-break parity (WR-01)
///
/// Upstream uses `std::priority_queue<TBinType>` whose `operator<` compares
/// **only** `Score()` (binarization.cpp:1345-1351). On equal scores, which bin
/// the heap pops is fixed by the binary-heap array structure, not insertion
/// order. We therefore reproduce the C++ STL binary max-heap operations
/// (`std::push_heap` / `std::pop_heap`, libstdc++ `__push_heap` / `__adjust_heap`
/// semantics) bit-for-bit in [`heap_push`] / [`heap_pop`] rather than a
/// first-occurrence linear scan, so the SET of bins that get split — and hence
/// the final sorted border set — matches upstream even when scores tie.
fn greedy_split(values: &[f32], max_borders: usize, total_weight: f64) -> Vec<f32> {
    // total_weight equals values.len() in the unweighted path; assert the
    // reduction routed through cb_core agrees with the slice length so a future
    // weighted variant cannot silently desync the count.
    debug_assert_eq!(total_weight as usize, values.len());
    let _ = total_weight;
    if values.len() < 2 {
        return Vec::new();
    }

    // std::priority_queue<TBinType> backed by a Vec maintained in binary-max-heap
    // order on `best_score` (the STL heap invariant). `heap_push`/`heap_pop`
    // reproduce libstdc++'s push_heap/pop_heap so tie-break pops match upstream.
    let mut heap: Vec<Bin> = Vec::new();
    heap_push(&mut heap, Bin::new(values, 0, values.len()));

    // while (splits.size() <= maxBordersCount && splits.top().CanSplit())
    while heap.len() <= max_borders {
        // splits.top() is the heap root.
        if !heap.first().map(Bin::can_split).unwrap_or(false) {
            break;
        }
        // auto top = splits.top(); splits.pop();
        let Some(mut top) = heap_pop(&mut heap) else {
            break;
        };
        // auto left = top.Split(); splits.push(left); splits.push(top);
        let left = top.split(values);
        heap_push(&mut heap, left);
        heap_push(&mut heap, top);
    }

    // Collect LeftBorder of every non-first bin. The collection order is
    // irrelevant (the caller dedups into a sorted set), so iterating the heap
    // array directly is equivalent to draining `splits` top-by-top.
    let mut borders: Vec<f32> = Vec::with_capacity(heap.len());
    for bin in &heap {
        if !bin.is_first() {
            borders.push(bin.left_border(values));
        }
    }
    borders
}

/// `std::push_heap(first, last, comp)` (libstdc++ `__push_heap`): sift the
/// just-appended last element up toward the root while its parent compares less
/// (`comp(parent, value)`), reproducing the STL heap's exact placement so tied
/// scores settle into the same array positions as upstream.
fn heap_push(heap: &mut Vec<Bin>, value: Bin) {
    heap.push(value);
    let mut hole = heap.len() - 1;
    // __push_heap: while (holeIndex > topIndex && comp(arr[parent], value))
    while hole > 0 {
        let parent = (hole - 1) / 2;
        // Compare parent against the value currently sitting at `hole`.
        let parent_lt_value = {
            // SAFETY of indexing: parent < hole < heap.len(); both in bounds.
            let (p, h) = (parent, hole);
            match (heap.get(p), heap.get(h)) {
                (Some(pp), Some(hh)) => pp.best_score < hh.best_score,
                _ => false,
            }
        };
        if !parent_lt_value {
            break;
        }
        heap.swap(parent, hole);
        hole = parent;
    }
}

/// `std::pop_heap(first, last, comp)` then `pop_back()` (libstdc++ `__pop_heap`
/// with `__adjust_heap`): move the root out, then sift the former last element
/// down from the root, always descending into the larger child, exactly as the
/// STL does, so the next `top()` matches upstream's heap on tied scores.
fn heap_pop(heap: &mut Vec<Bin>) -> Option<Bin> {
    let len = heap.len();
    if len == 0 {
        return None;
    }
    // Move the root to the back and shrink; `result` is the popped top.
    let last = len - 1;
    heap.swap(0, last);
    let result = heap.pop();
    let new_len = heap.len();
    if new_len > 1 {
        // __adjust_heap from the root over [0, new_len): sift the value now at
        // index 0 down, picking the larger child each step (comp(child, other)).
        adjust_heap(heap, 0, new_len);
    }
    result
}

/// libstdc++ `std::__adjust_heap(first, holeIndex, len, value)` specialized to
/// `holeIndex == 0` with `value` already placed at `heap[0]`. Sifts that value
/// down: at each level it moves the larger of the two children up into the hole,
/// then descends. Children of `i` are `2*i+1` and `2*i+2`; on a tie between the
/// two children the STL picks the **right** child (`comp(child[2*i+1],
/// child[2*i+2])` chooses the second when the first is not greater).
fn adjust_heap(heap: &mut [Bin], start: usize, len: usize) {
    let mut hole = start;
    loop {
        let right = 2 * hole + 2;
        if right >= len {
            break;
        }
        // secondChild = right; if comp(arr[right], arr[right-1]) secondChild = left.
        let left = right - 1;
        let larger = match (heap.get(left), heap.get(right)) {
            (Some(l), Some(r)) => {
                if r.best_score < l.best_score {
                    left
                } else {
                    right
                }
            }
            _ => break,
        };
        let hole_lt_larger = match (heap.get(hole), heap.get(larger)) {
            (Some(h), Some(g)) => h.best_score < g.best_score,
            _ => break,
        };
        if !hole_lt_larger {
            break;
        }
        heap.swap(hole, larger);
        hole = larger;
    }
    // Handle a lone left child (no right child) at the bottom level.
    let left = 2 * hole + 1;
    if left == len - 1 {
        let hole_lt_left = match (heap.get(hole), heap.get(left)) {
            (Some(h), Some(l)) => h.best_score < l.best_score,
            _ => return,
        };
        if hole_lt_left {
            heap.swap(hole, left);
        }
    }
}

