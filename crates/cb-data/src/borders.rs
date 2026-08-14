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

use cb_core::{sum_f64, TFastRng64};

/// Upstream `TQuantizationOptions::MaxSubsetSizeForBuildBordersAlgorithms`
/// (`catboost/libs/data/quantization.h`, v1.2.10): border-building algorithms run
/// on at most this many objects. A larger column builds its borders from a random
/// subset of exactly this size (`GetArraySubsetForBuildBorders` →
/// `SampleIndices(objectCount, sampleSize, rand)`), which is what keeps upstream's
/// quantization O(1) in dataset size past this point. Mirroring the cap here is a
/// PARITY improvement at scale, not just a speedup: above the cap upstream never
/// sees the full column either, so full-column borders were already a divergence.
pub const MAX_SUBSET_SIZE_FOR_BUILD_BORDERS: usize = 200_000;

/// The `MaxSumLog` penalty: `-log(count + 1e-8)` (binarization.cpp:180). The
/// `1e-8` epsilon guards `log(0)` for an empty side. Computed in `f64`.
#[must_use]
pub fn penalty_maxsumlog(count: f64) -> f64 {
    // double Penalty<EPenaltyType::MaxSumLog>(double weight) { return -log(weight + 1e-8); }
    -(count + 1e-8).ln()
}

/// The `MinEntropy` penalty: `weight * log(weight + 1e-8)` (binarization.cpp,
/// `Penalty<EPenaltyType::MinEntropy>`). Same `1e-8` guard as
/// [`penalty_maxsumlog`]. Computed in `f64`.
#[must_use]
pub fn penalty_minentropy(weight: f64) -> f64 {
    // double Penalty<EPenaltyType::MinEntropy>(double weight) { return weight * log(weight + 1e-8); }
    weight * (weight + 1e-8).ln()
}

/// Which `EPenaltyType` a binarizer scores splits with (`binarization.cpp`).
/// CatBoost pairs each of the two penalties with each of the two SEARCH
/// strategies, giving the four penalty-driven border types:
/// `GreedyLogSum` = greedy + [`Self::MaxSumLog`], `GreedyMinEntropy` = greedy +
/// [`Self::MinEntropy`], `MaxLogSum` = exact + [`Self::MaxSumLog`],
/// `MinEntropy` = exact + [`Self::MinEntropy`].
///
/// # The two penalties agree far more often than they look
///
/// With unit object weights a bin of `n` objects split into `l + r = n` scores
/// `log(l+eps) + log(r+eps) - log(n+eps)` under [`Self::MaxSumLog`] and
/// `n*log(n) - l*log(l) - r*log(r)` under [`Self::MinEntropy`]. BOTH are
/// maximized at the balanced split `l == r` and both increase monotonically with
/// bin size, so on evenly-spread data the greedy heap pops the same bins and
/// picks the same split points — the two border sets come out byte-identical.
/// They diverge only when the achievable split positions are asymmetric (heavy,
/// unevenly-sized duplicate runs), because only then do the two scores trade
/// "bin size" against "split imbalance" at different rates. The
/// `border_types/borders_runs.*` oracle cells exist precisely to cover that
/// regime; see `generator/gen_border_type_fixtures.py::runs_column`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyType {
    /// `-log(w + 1e-8)` — the `GreedyLogSum` / `MaxLogSum` penalty.
    MaxSumLog,
    /// `w * log(w + 1e-8)` — the `GreedyMinEntropy` / `MinEntropy` penalty.
    MinEntropy,
}

impl PenaltyType {
    /// Apply this penalty to a bin weight.
    #[must_use]
    pub fn apply(self, weight: f64) -> f64 {
        match self {
            Self::MaxSumLog => penalty_maxsumlog(weight),
            Self::MinEntropy => penalty_minentropy(weight),
        }
    }
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
    /// The `EPenaltyType` this bin scores its candidate splits with — the ONLY
    /// difference between the `GreedyLogSum` and `GreedyMinEntropy` binarizers
    /// (`TGreedyBinarizer<EPenaltyType::MaxSumLog>` vs `<MinEntropy>`).
    penalty: PenaltyType,
}

impl Bin {
    /// `TFeatureBin(binStart, binEnd, featuresStart)` — construct then
    /// immediately compute the best split (`UpdateBestSplitProperties`).
    fn new(values: &[f32], start: usize, end: usize, penalty: PenaltyType) -> Self {
        let mut bin = Self {
            start,
            end,
            best_split: start,
            best_score: 0.0,
            penalty,
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
        let left = -self.penalty.apply((split_pos - self.start) as f64);
        // rightPartScore = -Penalty(BinEnd - splitPos);
        let right = -self.penalty.apply((self.end - split_pos) as f64);
        // currBinScore = -Penalty(BinEnd - BinStart);
        let curr = -self.penalty.apply((self.end - self.start) as f64);
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
        let left = Self::new(values, self.start, self.best_split, self.penalty);
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
    // Narrow to f32 (feature storage type) and delegate to the f32 entry — the
    // narrowing here is the SAME `v as f32` the fused loop performed before the
    // f32 entry existed, so the border set is byte-identical for every caller.
    let narrowed: Vec<f32> = column.iter().map(|&v| v as f32).collect();
    select_borders_greedy_logsum_f32(&narrowed, max_borders, nan_sentinel)
}

/// [`select_borders_greedy_logsum`] over an already-narrowed f32 column — the
/// hot fit-prep entry (the trainer stores features as f32, so routing the f64
/// pool column through here after ONE narrowing pass avoids a second full-column
/// f64 read on every fit).
///
/// Columns longer than [`MAX_SUBSET_SIZE_FOR_BUILD_BORDERS`] build their borders
/// from a random subset of exactly that size, mirroring upstream's
/// `GetArraySubsetForBuildBorders` (see the constant's doc). The subset is drawn
/// by a partial Fisher–Yates with the FIXED-seed [`TFastRng64`] stream, so the
/// selection is deterministic run-to-run (upstream seeds from the task RNG and is
/// not reproducible across its own runs; a fixed stream keeps fixtures and CI
/// stable while remaining statistically the same draw). Sub-threshold columns
/// take the exact pre-existing full-column path — byte-identical borders.
#[must_use]
pub fn select_borders_greedy_logsum_f32(
    column: &[f32],
    max_borders: usize,
    nan_sentinel: bool,
) -> Vec<f64> {
    // Drop NaNs (sampling first when over the cap — upstream subsets OBJECTS, so
    // the subset is taken before NaN filtering there too), sort ascending.
    let values: Vec<f32> = if column.len() > MAX_SUBSET_SIZE_FOR_BUILD_BORDERS {
        let sample =
            sample_indices_for_build_borders(column.len(), MAX_SUBSET_SIZE_FOR_BUILD_BORDERS);
        gather_non_nan(column, &sample)
    } else {
        column.iter().copied().filter(|v| !v.is_nan()).collect()
    };
    borders_from_values(values, max_borders, nan_sentinel)
}

/// [`select_borders_greedy_logsum_f32`] for an over-cap column whose sample
/// index set was precomputed by [`sample_indices_for_build_borders`] — the
/// SPD-03 fit-prep entry. The fixed-seed draw depends only on the object count,
/// so every same-length column samples the SAME index set; hoisting the draw out
/// of the per-column loop removes an O(n) index-array build + shuffle per column,
/// and the ascending index order turns the per-column gather into a forward
/// streaming read. The border set is byte-identical to the self-sampling entry:
/// the gathered multiset is the same (same index set, same per-column NaN drop),
/// and [`borders_from_values`] is a pure function of that multiset.
#[must_use]
pub fn select_borders_greedy_logsum_f32_presampled(
    column: &[f32],
    sorted_sample_indices: &[u32],
    max_borders: usize,
    nan_sentinel: bool,
) -> Vec<f64> {
    let values = gather_non_nan(column, sorted_sample_indices);
    borders_from_values(values, max_borders, nan_sentinel)
}

/// Shared tail of border selection: sort the (NaN-free) sampled values, run the
/// greedy split, normalize and dedup the border set, and widen to f64.
///
/// This is a pure function of the value MULTISET, not the incoming order: the
/// sort's output value sequence is determined by the multiset alone (the radix
/// path totally orders bit patterns; under the sub-threshold comparator sort the
/// only comparator-equal-but-bitwise-distinct pair is {-0.0, +0.0}, and every
/// downstream consumer is order-insensitive across equal keys — `left_border`'s
/// midpoint of a {-0.0, +0.0} adjacency is +0.0 in either order, the greedy
/// split scores depend only on the sorted values at each index, and the emitted
/// border set normalizes -0.0 to +0.0 before dedup). pdqsort avoids the stable
/// merge sort's O(n/2) allocation on the hot fit-prep path.
fn borders_from_values(mut values: Vec<f32>, border_count: usize, nan_sentinel: bool) -> Vec<f64> {
    borders_from_values_with_penalty(
        &mut values,
        real_border_budget(border_count, nan_sentinel),
        nan_sentinel,
        PenaltyType::MaxSumLog,
    )
}

/// How many REAL borders a feature's binarizer may emit, given the total stored
/// border budget and whether a NanMode sentinel occupies one of the slots.
///
/// A NaN-bearing feature spends one border on the sentinel, so the binarizer is
/// handed `border_count - 1`. This is unobservable at a saturating budget (every
/// bin runs out of splits before the cap binds), which is why the pre-existing
/// `borders_quant` fixture at `border_count = 254` over 50-row columns never
/// pinned it.
#[must_use]
pub(crate) fn real_border_budget(border_count: usize, nan_sentinel: bool) -> usize {
    if nan_sentinel {
        border_count.saturating_sub(1)
    } else {
        border_count
    }
}

/// [`borders_from_values`] parameterized by the greedy penalty — the shared tail
/// for `GreedyLogSum` ([`PenaltyType::MaxSumLog`]) and `GreedyMinEntropy`
/// ([`PenaltyType::MinEntropy`]).
///
/// `max_borders` is the REAL border budget — the caller has already applied
/// [`real_border_budget`].
pub(crate) fn borders_from_values_with_penalty(
    values: &mut [f32],
    max_borders: usize,
    nan_sentinel: bool,
    penalty: PenaltyType,
) -> Vec<f64> {
    sort_f32_ascending(values);

    // Each non-NaN object carries unit weight; the total weight is accumulated
    // through the audited reduction primitive (D-07) — for the unweighted path
    // this equals the object count, but the routing is the parity contract.
    let unit_weights = vec![1.0_f64; values.len()];
    let total_weight = total_object_weight(&unit_weights);
    let borders_f32 = greedy_split(values, max_borders, total_weight, penalty);
    finalize_border_set(borders_f32, nan_sentinel)
}

/// The shared `THashSet<float>` -> emitted-border-vector tail every binarizer
/// ends with: normalize IEEE `-0.0f` to `+0.0f`, sort ascending, dedup, widen to
/// `f64`, and optionally prepend the NanMode(`Min`) `f32::MIN` sentinel.
///
/// Upstream collects borders into a `THashSet<float>` (so duplicates collapse)
/// and the caller sorts. Every border type shares this step, which is why it
/// lives here rather than inside the greedy path.
pub(crate) fn finalize_border_set(borders_f32: Vec<f32>, nan_sentinel: bool) -> Vec<f64> {
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

/// Ascending f32 sort via LSD radix over the total-order key transform
/// (`bits ^ (sign ? 0xFFFF_FFFF : 0x8000_0000)`) — SPD-03 wave 4: this sort is
/// the dominant term of fit-prep's border selection at scale (200k values × every
/// column; the P100 r3 diag attributes ~2.5 s CPU to the border stage), and the
/// comparator-based `sort_unstable_by` costs ~4× a counting radix here.
///
/// Ordering contract vs the former `partial_cmp` unstable sort: the sorted VALUE
/// sequence is identical for every NaN-free input except across `{-0.0, +0.0}`
/// pairs (radix puts `-0.0` first deterministically; the unstable comparator sort
/// ordered them arbitrarily). Every downstream consumer is order-insensitive
/// across equal keys — `left_border`'s midpoint of a `{-0.0, +0.0}` adjacency is
/// `+0.0` in either order, the greedy split scores depend only on the sorted
/// values at each index, and the emitted border set normalizes `-0.0` before
/// dedup — so the border SET is byte-identical (the same argument that already
/// licensed the unstable sort). The caller guarantees NaN-free input (NaNs are
/// filtered/sampled out above).
pub(crate) fn sort_f32_ascending(values: &mut [f32]) {
    // Small inputs: the comparator sort's constant factor wins; radix scratch
    // would dominate.
    if values.len() < 1 << 10 {
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        return;
    }
    let mut keys: Vec<u32> = values
        .iter()
        .map(|v| {
            let b = v.to_bits();
            if b & 0x8000_0000 != 0 {
                !b
            } else {
                b ^ 0x8000_0000
            }
        })
        .collect();
    let mut scratch: Vec<u32> = vec![0_u32; keys.len()];
    for pass in 0..4_u32 {
        let shift = pass * 8;
        let mut counts = [0_usize; 256];
        for &k in keys.iter() {
            let bucket = ((k >> shift) & 0xFF) as usize;
            if let Some(slot) = counts.get_mut(bucket) {
                *slot += 1;
            }
        }
        let mut offsets = [0_usize; 256];
        let mut running = 0_usize;
        for (o, &c) in offsets.iter_mut().zip(counts.iter()) {
            *o = running;
            running += c;
        }
        for &k in keys.iter() {
            let bucket = ((k >> shift) & 0xFF) as usize;
            if let Some(pos) = offsets.get_mut(bucket) {
                if let Some(slot) = scratch.get_mut(*pos) {
                    *slot = k;
                }
                *pos += 1;
            }
        }
        std::mem::swap(&mut keys, &mut scratch);
    }
    for (v, &k) in values.iter_mut().zip(keys.iter()) {
        let b = if k & 0x8000_0000 != 0 {
            k ^ 0x8000_0000
        } else {
            !k
        };
        *v = f32::from_bits(b);
    }
}

/// Draw `sample_size` object indices uniformly WITHOUT replacement (upstream
/// `SampleIndices<ui32>(objectCount, sampleSize, rand)`) and return them sorted
/// ascending. The index SET is what matters downstream — the sampled values are
/// sorted before any consumer sees them, so the draw order carries no
/// information — and ascending order makes the per-column gather a forward
/// streaming read instead of a random walk.
///
/// Partial Fisher–Yates over an index array: each of the `sample_size` draws
/// swaps a uniformly chosen remaining index into the prefix, so the prefix is an
/// exact uniform sample. The RNG is the audited [`TFastRng64`] upstream port
/// with a FIXED seed (determinism contract — see the caller's doc); the drawn
/// set depends only on `n` and `sample_size`, so equal-length columns share it.
#[must_use]
pub fn sample_indices_for_build_borders(n: usize, sample_size: usize) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let mut rng = TFastRng64::from_seed(0);
    let take = sample_size.min(n);
    for i in 0..take {
        let offset = rng.uniform((n - i) as u64) as usize;
        indices.swap(i, i + offset);
    }
    indices.truncate(take);
    indices.sort_unstable();
    indices
}

/// Gather `column[idx]` for each sampled index, dropping NaNs — so the returned
/// vec may be slightly shorter than the sample on a NaN-bearing column, exactly
/// as upstream's object-subset-then-NaN-handling ordering yields. Out-of-range
/// indices (impossible for a sample drawn over this column's length) are
/// skipped rather than panicking.
pub(crate) fn gather_non_nan(column: &[f32], indices: &[u32]) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(indices.len());
    for &idx in indices {
        if let Some(&v) = column.get(idx as usize) {
            if !v.is_nan() {
                out.push(v);
            }
        }
    }
    out
}

/// `GreedySplit` (binarization.cpp:1499-1520): a max-priority queue of bins keyed
/// by best-split score. While the queue holds `<= max_borders` bins and the top
/// bin can split, pop it, split it, and push both halves; then collect the left
/// border of every non-first bin into a dedup set.
///
/// # Tie-break: INSERTION ORDER, established empirically (not the STL heap)
///
/// Upstream's container is `std::priority_queue<TBinType>` whose `operator<`
/// compares **only** `Score()` (binarization.cpp:1345-1351). Ties are constant
/// here — object counts are small integers, so any two equal-sized bins that
/// split evenly score identically — and the tie-break decides WHICH bin receives
/// the next split, hence where the border lands.
///
/// This code previously reproduced libstdc++'s `push_heap` / `__adjust_heap`
/// array mechanics, on the theory that the STL heap layout was the observable
/// behaviour. That is WRONG against catboost 1.2.10. Measured over the frozen
/// `border_types` matrix (8 corpora x border-count cells x 4 features, both
/// penalties):
///
/// | tie-break policy                              | cells matching catboost |
/// |-----------------------------------------------|-------------------------|
/// | real `std::priority_queue` (libstdc++, in C++) | 21 / 31                 |
/// | INSERTION ORDER (earliest-pushed wins)         | **31 / 31**             |
///
/// The 21/31 figure comes from `generator/greedy_binarizer_oracle.cpp`, which
/// links the actual STL container rather than emulating it — so this is not an
/// emulation bug being papered over: catboost's shipped binary genuinely behaves
/// as an insertion-stable queue (it is not built against the libstdc++ heap this
/// crate was modelling). The empirical wheel is the parity authority
/// (CLAUDE.md), so insertion order is what we implement.
///
/// # Why this was invisible until now
///
/// The pre-existing `borders_quant` fixture runs at `border_count = 254` over
/// 50-row columns, where the budget EXCEEDS the number of representable splits.
/// Every bin ends up unsplittable at the same time, no tie is ever contested,
/// and the wrong policy returns the right answer. The divergence appears only
/// when the border budget BINDS — which is the normal case for real datasets.
fn greedy_split(
    values: &[f32],
    max_borders: usize,
    total_weight: f64,
    penalty: PenaltyType,
) -> Vec<f32> {
    // total_weight equals values.len() in the unweighted path; assert the
    // reduction routed through cb_core agrees with the slice length so a future
    // weighted variant cannot silently desync the count.
    debug_assert_eq!(total_weight as usize, values.len());
    let _ = total_weight;
    if values.len() < 2 {
        return Vec::new();
    }

    let mut queue: std::collections::BinaryHeap<QueuedBin> = std::collections::BinaryHeap::new();
    let mut seq: u64 = 0;
    queue.push(QueuedBin {
        bin: Bin::new(values, 0, values.len(), penalty),
        seq,
    });
    seq += 1;

    // while (splits.size() <= maxBordersCount && splits.top().CanSplit())
    while queue.len() <= max_borders {
        if !queue.peek().map(|q| q.bin.can_split()).unwrap_or(false) {
            break;
        }
        let Some(QueuedBin { bin: mut top, .. }) = queue.pop() else {
            break;
        };
        // auto left = top.Split(); splits.push(left); splits.push(top);
        let left = top.split(values);
        queue.push(QueuedBin { bin: left, seq });
        seq += 1;
        queue.push(QueuedBin { bin: top, seq });
        seq += 1;
    }

    // Collect LeftBorder of every non-first bin. The collection order is
    // irrelevant (the caller dedups into a sorted set).
    let mut borders: Vec<f32> = Vec::with_capacity(queue.len());
    for q in &queue {
        if !q.bin.is_first() {
            borders.push(q.bin.left_border(values));
        }
    }
    borders
}

/// A [`Bin`] carrying its insertion sequence so the priority queue can break
/// score ties by INSERTION ORDER (see [`greedy_split`]).
///
/// [`Ord`] is "greater = popped first": higher `best_score` wins, and on an
/// exact score tie the SMALLER `seq` wins (hence the reversed sequence
/// comparison). `f64::total_cmp` gives a total order including the
/// `-inf` scores an unsplittable bin carries.
struct QueuedBin {
    bin: Bin,
    seq: u64,
}

impl Ord for QueuedBin {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.bin
            .best_score
            .total_cmp(&other.bin.best_score)
            // Reversed: the earliest-inserted bin must compare GREATER so it is
            // popped first among equal scores.
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for QueuedBin {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueuedBin {
    fn eq(&self, other: &Self) -> bool {
        self.bin.best_score.total_cmp(&other.bin.best_score) == std::cmp::Ordering::Equal
            && self.seq == other.seq
    }
}

impl Eq for QueuedBin {}


