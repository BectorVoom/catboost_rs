//! Host-side ordered bucket reduction — the parity-critical step that folds the
//! backend's per-object scatter contributions into per-bin / per-leaf totals
//! through `cb_core::sum_f64` in canonical object order (D-02/D-05). The
//! `cb-backend` kernel does ONLY the order-independent per-object work; THIS is
//! where the order-sensitive SUM happens, so the 1e-5 oracle bar stays
//! deterministic.
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/score_calcers.cpp` / `online_predictor.h` —
//! `TBucketStats { SumWeightedDelta, SumWeight }`. Each leaf/bucket accumulates
//! the per-object first-derivative ("weighted delta") and weight; the L2 score
//! calcer (`score.rs`) and the Gradient leaf delta (`leaf.rs`) consume these
//! reduced totals.
//!
//! # Summation routing (D-07 / D-08)
//!
//! Every bin total is produced by [`cb_core::sum_f64`] over the per-object
//! contributions GATHERED in canonical object order. No raw iterator-sum or
//! zero-seeded float fold is spelled here (D-08); the gather builds an ordered
//! `Vec` and hands it to the single sanctioned primitive.

use cb_core::sum_f64;

/// A single leaf/bucket's reduced statistics (`TBucketStats` analogue): the
/// summed first-derivative ("weighted delta") and the summed weight of its member
/// objects.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LeafStats {
    /// Σ der1[i] over the leaf's member objects (the "weighted delta"). In the
    /// unweighted path the per-object weight is folded into `der1` already, so
    /// this is the plain derivative sum.
    pub sum_weighted_delta: f64,
    /// Σ weight[i] over the leaf's member objects (the leaf's object count in the
    /// unweighted path).
    pub sum_weight: f64,
}

/// Reduce per-object contributions into one bucket per leaf index.
///
/// `leaf_of[i]` is object `i`'s leaf index (`0..n_leaves`); `der1[i]` its
/// first-derivative contribution; `weight[i]` its weight. For each leaf the
/// member objects are gathered in ascending object order and summed via
/// [`cb_core::sum_f64`] (D-05 canonical order), producing a [`LeafStats`] per
/// leaf. `der1`, `weight`, and `leaf_of` MUST be the same length (`n` objects);
/// any object whose leaf index is `>= n_leaves` is ignored defensively rather
/// than panicking (the trainer guarantees valid indices).
#[must_use]
pub fn reduce_leaf_stats(
    leaf_of: &[usize],
    der1: &[f64],
    weight: &[f64],
    n_leaves: usize,
) -> Vec<LeafStats> {
    // Gather each leaf's per-object contributions in ascending object order, then
    // fold each gathered Vec through the single sanctioned reduction primitive so
    // the SUM order is exactly upstream's thread_count==1 object order (D-05).
    let mut delta_members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
    let mut weight_members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];

    for (i, &leaf) in leaf_of.iter().enumerate() {
        if leaf >= n_leaves {
            continue;
        }
        // der1/weight are parallel to leaf_of; a missing entry is treated as 0.0
        // (defensive — the trainer passes equal-length slices).
        let d = der1.get(i).copied().unwrap_or(0.0);
        let w = weight.get(i).copied().unwrap_or(0.0);
        if let Some(slot) = delta_members.get_mut(leaf) {
            slot.push(d);
        }
        if let Some(slot) = weight_members.get_mut(leaf) {
            slot.push(w);
        }
    }

    (0..n_leaves)
        .map(|leaf| {
            let deltas = delta_members.get(leaf).map_or(&[][..], Vec::as_slice);
            let weights = weight_members.get(leaf).map_or(&[][..], Vec::as_slice);
            LeafStats {
                sum_weighted_delta: sum_f64(deltas),
                sum_weight: sum_f64(weights),
            }
        })
        .collect()
}

/// Reduce per-object weighted second derivatives into one `Σ der2*weight` per
/// leaf index, through `cb_core::sum_f64` in canonical object order (D-05).
///
/// This is the Newton-method companion to [`reduce_leaf_stats`]: the boosting
/// loop needs the leaf's summed second derivative for `newton_leaf_delta`'s
/// `-sum_der2 + scaledL2` denominator. `weighted_der2[i]` is object `i`'s
/// `der2 * weight` (the elementwise product the host computes); `leaf_of[i]` its
/// leaf index. Objects whose leaf index is `>= n_leaves` are ignored defensively
/// (the trainer guarantees valid indices). Kept separate from [`LeafStats`] so the
/// score path (which never reads `der2`) is untouched.
#[must_use]
pub fn reduce_leaf_der2(
    leaf_of: &[usize],
    weighted_der2: &[f64],
    n_leaves: usize,
) -> Vec<f64> {
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if leaf >= n_leaves {
            continue;
        }
        let d = weighted_der2.get(i).copied().unwrap_or(0.0);
        if let Some(slot) = members.get_mut(leaf) {
            slot.push(d);
        }
    }
    (0..n_leaves)
        .map(|leaf| {
            let ds = members.get(leaf).map_or(&[][..], Vec::as_slice);
            sum_f64(ds)
        })
        .collect()
}

/// Gather each leaf's per-member residuals (as `f32`, matching upstream's
/// `TVector<float> leafSamples`) and weights, in ascending object order — the
/// input the Exact-method quantile (`exact_leaf_delta`) consumes per leaf.
///
/// `leaf_of[i]` is object `i`'s leaf index; `residuals[i] = target_i - approx_i`
/// (the host computes the residual in `f64`, this widens it through `f32` to match
/// upstream's float sample buffer); `weight[i]` its object weight. Returns, per
/// leaf, the `(residuals, weights)` member vectors. Objects whose leaf index is
/// `>= n_leaves` are ignored defensively. No float SUM of derivatives is spelled
/// here (the only later accumulation is the quantile's weight scan inside
/// [`crate::leaf::exact_leaf_delta`]).
#[must_use]
pub fn collect_leaf_residuals(
    leaf_of: &[usize],
    residuals: &[f64],
    weight: &[f64],
    n_leaves: usize,
) -> Vec<(Vec<f32>, Vec<f64>)> {
    let mut out: Vec<(Vec<f32>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); n_leaves];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if leaf >= n_leaves {
            continue;
        }
        let r = residuals.get(i).copied().unwrap_or(0.0) as f32;
        let w = weight.get(i).copied().unwrap_or(1.0);
        if let Some(slot) = out.get_mut(leaf) {
            slot.0.push(r);
            slot.1.push(w);
        }
    }
    out
}

// ----------------------------------------------------------------------------
// CPU split-finding histogram primitives (PERF-01 / PERF-02, Phase 21).
//
// The parity-critical DATA-PRODUCTION layer: a per-(leaf, feature, bin)
// 2+-channel `TBucketStats` histogram built by ONE object-order pass, the
// `O(n_bins)` prefix scan that turns a feature's buckets into the per-border
// `LeafStats` array fed to the UNCHANGED score math, and the subtraction trick
// (child = parent − sibling). Everything routes every float sum through
// [`cb_core::sum_f64`] in canonical order (D-05/D-08) so the ≤1e-5 oracle bar is
// preserved. Pure host Rust — cubecl-free AND rayon-free (D-03); NOT a dependency
// on `cb-backend` (the device `pointwise_hist.rs::host_reference_hist2` is the
// READ-ONLY template transcribed here, never imported).
// ----------------------------------------------------------------------------

/// A per-`(leaf, feature, bin)` bucket-statistics histogram — the host
/// `TBucketStats` analogue (`calc_score_cache.h:72-95`).
///
/// # Frozen flat layout (mirrors the device `pointwise_hist.rs:44-49`)
///
/// ```text
/// index(leaf, feature, bin, channel) =
///     ((leaf * n_features + feature) * n_bins + bin) * n_channels + channel
/// ```
///
/// `n_channels = approx_dimension + 1`: channels `0..approx_dimension` hold
/// `Σ der1[d]` (the "weighted delta" per output dimension), channel
/// `approx_dimension` holds `Σ weight` (shared across dimensions). Each cell is
/// the ordered [`cb_core::sum_f64`] of its member objects' contributions gathered
/// in ascending object order — so the histogram carries exactly the same reduced
/// totals [`reduce_leaf_stats`] would produce for the (leaf, feature, bin)
/// partition, generalized from leaf to (leaf, feature, bin).
#[derive(Debug, Clone, PartialEq)]
pub struct BucketHistogram {
    /// Flat channel data in the frozen layout above.
    data: Vec<f64>,
    /// Number of leaves in the CURRENT partition.
    n_leaves: usize,
    /// Number of (float) features whose bins are histogrammed.
    n_features: usize,
    /// Number of bins per feature (`n_borders + 1`).
    n_bins: usize,
    /// Number of channels (`approx_dimension` delta channels + 1 weight channel).
    n_channels: usize,
}

impl BucketHistogram {
    /// Number of leaves in the partition this histogram was built over.
    #[must_use]
    pub fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Number of features histogrammed.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of bins per feature (`n_borders + 1`).
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Number of channels (`approx_dimension + 1`).
    #[must_use]
    pub fn n_channels(&self) -> usize {
        self.n_channels
    }

    /// Number of delta (per-dimension `Σ der1`) channels (`n_channels - 1`).
    #[must_use]
    pub fn approx_dimension(&self) -> usize {
        self.n_channels.saturating_sub(1)
    }

    /// The flat base offset of cell `(leaf, feature, bin)`, or `None` if any index
    /// is out of range (defensive — no raw indexing, workspace deny
    /// `indexing_slicing`).
    fn cell_base(&self, leaf: usize, feature: usize, bin: usize) -> Option<usize> {
        if leaf >= self.n_leaves || feature >= self.n_features || bin >= self.n_bins {
            return None;
        }
        // (leaf * n_features + feature) * n_bins + bin) * n_channels
        Some(((leaf * self.n_features + feature) * self.n_bins + bin) * self.n_channels)
    }

    /// The contiguous flat block holding ALL `(bin, channel)` cells of
    /// `(leaf, feature)` — length `n_bins * n_channels`, or `None` if the
    /// `(leaf, feature)` index is out of range. Within the returned slice, cell
    /// `(bin, channel)` lives at offset `bin * n_channels + channel` (the frozen
    /// layout: `leaf`/`feature` are the outer indices, so a fixed `(leaf, feature)`
    /// is one contiguous run). This lets the `O(n_bins)` prefix scan read a whole
    /// feature row with a SINGLE base computation instead of the per-cell flat-index
    /// multiply chain of [`channel`](Self::channel) — a pure constant-factor scan
    /// speedup (PERF-01), byte-identical values (T-21-01 defensive `get`).
    fn feature_block(&self, leaf: usize, feature: usize) -> Option<&[f64]> {
        let base = self.cell_base(leaf, feature, 0)?;
        let len = self.n_bins.checked_mul(self.n_channels)?;
        self.data.get(base..base.checked_add(len)?)
    }

    /// The value of one channel of cell `(leaf, feature, bin)`. Out-of-range
    /// indices return `0.0` (an absent cell contributes nothing — mirrors the
    /// empty-leaf `LeafStats::default()` convention).
    #[must_use]
    pub fn channel(&self, leaf: usize, feature: usize, bin: usize, channel: usize) -> f64 {
        if channel >= self.n_channels {
            return 0.0;
        }
        self.cell_base(leaf, feature, bin)
            .and_then(|base| self.data.get(base + channel))
            .copied()
            .unwrap_or(0.0)
    }

    /// The subtraction trick (`TBucketStats::Remove`, `calc_score_cache.h:88`):
    /// per-cell, per-channel `self - other`, yielding the sibling histogram
    /// (`sibling = parent.remove(child)`, upstream `scoring.cpp:315` FixUpStats).
    ///
    /// The two histograms MUST share the same shape; on a shape mismatch the
    /// receiver is returned unchanged (defensive — the trainer always subtracts a
    /// same-shape child, and a mismatch is a caller bug rather than a panic
    /// condition, T-21-02). Every subtraction is a plain f64 `-=` in the frozen
    /// cell order, matching upstream's own subtraction so the rounding is
    /// parity-faithful (RESEARCH Pitfall 2).
    #[must_use]
    pub fn remove(&self, other: &BucketHistogram) -> BucketHistogram {
        if self.n_leaves != other.n_leaves
            || self.n_features != other.n_features
            || self.n_bins != other.n_bins
            || self.n_channels != other.n_channels
            || self.data.len() != other.data.len()
        {
            return self.clone();
        }
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        BucketHistogram {
            data,
            n_leaves: self.n_leaves,
            n_features: self.n_features,
            n_bins: self.n_bins,
            n_channels: self.n_channels,
        }
    }

    /// Elementwise `self + other` in a SINGLE pass (one allocation) — the
    /// disjoint-slot merge a level transition needs (the FALSE child occupies the
    /// low slots, the TRUE child the high slots, so their non-zero cells never
    /// overlap). The two histograms MUST share the same shape; on a mismatch the
    /// receiver is returned unchanged (defensive, T-21-02).
    ///
    /// Each cell is a plain f64 `+` in the frozen cell order — bit-identical to the
    /// `a − (0 − b)` triple-`remove` identity it replaces (`a − (−b) == a + b` in
    /// IEEE), but one pass / one allocation instead of three, which is the dominant
    /// per-level cost at large `n_bins` (PERF-01).
    #[must_use]
    pub fn add(&self, other: &BucketHistogram) -> BucketHistogram {
        if self.n_leaves != other.n_leaves
            || self.n_features != other.n_features
            || self.n_bins != other.n_bins
            || self.n_channels != other.n_channels
            || self.data.len() != other.data.len()
        {
            return self.clone();
        }
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        BucketHistogram {
            data,
            n_leaves: self.n_leaves,
            n_features: self.n_features,
            n_bins: self.n_bins,
            n_channels: self.n_channels,
        }
    }

    /// Fused `relocate(self, shift) − minus` in a SINGLE allocation: relocate this
    /// (parent) histogram into the `n_new_leaves` layout (leaf `p` → slot `p+shift`)
    /// and subtract `minus` (already in the `n_new_leaves` layout). This is the
    /// subtraction-trick larger-sibling derivation (WR-04) without the intermediate
    /// relocated-parent allocation.
    ///
    /// Bit-identical to `self.relocate(n_new_leaves, shift).remove(minus)`: each cell
    /// is `relocated_parent − minus`, computed here as `(−minus) + relocated_parent`
    /// which equals `relocated_parent − minus` exactly in IEEE (`−minus` is an exact
    /// negation, add is commutative). Shape mismatch falls back to the explicit
    /// two-step path (defensive, T-21-02).
    #[must_use]
    pub fn relocate_sub(
        &self,
        n_new_leaves: usize,
        shift: usize,
        minus: &BucketHistogram,
    ) -> BucketHistogram {
        let per_leaf = self
            .n_features
            .checked_mul(self.n_bins)
            .and_then(|x| x.checked_mul(self.n_channels))
            .unwrap_or(0);
        let total = n_new_leaves.checked_mul(per_leaf).unwrap_or(0);
        if minus.n_leaves != n_new_leaves
            || minus.n_features != self.n_features
            || minus.n_bins != self.n_bins
            || minus.n_channels != self.n_channels
            || minus.data.len() != total
        {
            // Defensive: explicit relocate-then-remove (same bits, one extra alloc).
            return self.relocate(n_new_leaves, shift).remove(minus);
        }
        // Start from −minus, then fold in the relocated parent blocks.
        let mut data: Vec<f64> = minus.data.iter().map(|&b| -b).collect();
        for p in 0..self.n_leaves {
            let dest = match p.checked_add(shift) {
                Some(d) if d < n_new_leaves => d,
                _ => continue,
            };
            let src_start = match p.checked_mul(per_leaf) {
                Some(s) => s,
                None => continue,
            };
            let dst_start = match dest.checked_mul(per_leaf) {
                Some(s) => s,
                None => continue,
            };
            let src = match self.data.get(src_start..src_start.saturating_add(per_leaf)) {
                Some(s) => s,
                None => continue,
            };
            if let Some(dst) = data.get_mut(dst_start..dst_start.saturating_add(per_leaf)) {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s;
                }
            }
        }
        BucketHistogram {
            data,
            n_leaves: n_new_leaves,
            n_features: self.n_features,
            n_bins: self.n_bins,
            n_channels: self.n_channels,
        }
    }

    /// Fused `self + relocate(other, shift)` in a SINGLE allocation: add `other`'s
    /// cells, relocated by `shift` (leaf `p` → slot `p+shift`), into a clone of
    /// `self`. This reunites the smaller sibling's second (high-slot) copy with the
    /// larger sibling without a separate relocation allocation.
    ///
    /// Bit-identical to `self.add(&other.relocate(self.n_leaves, shift))`: each cell
    /// is `self + relocated_other` in the same operand order. Shape mismatch on the
    /// per-leaf block width falls back to that explicit path (defensive, T-21-02).
    #[must_use]
    pub fn add_relocated(&self, other: &BucketHistogram, shift: usize) -> BucketHistogram {
        if self.n_features != other.n_features
            || self.n_bins != other.n_bins
            || self.n_channels != other.n_channels
        {
            return self.add(&other.relocate(self.n_leaves, shift));
        }
        let per_leaf = self
            .n_features
            .checked_mul(self.n_bins)
            .and_then(|x| x.checked_mul(self.n_channels))
            .unwrap_or(0);
        let mut data = self.data.clone();
        for p in 0..other.n_leaves {
            let dest = match p.checked_add(shift) {
                Some(d) if d < self.n_leaves => d,
                _ => continue,
            };
            let src_start = match p.checked_mul(per_leaf) {
                Some(s) => s,
                None => continue,
            };
            let dst_start = match dest.checked_mul(per_leaf) {
                Some(s) => s,
                None => continue,
            };
            let src = match other.data.get(src_start..src_start.saturating_add(per_leaf)) {
                Some(s) => s,
                None => continue,
            };
            if let Some(dst) = data.get_mut(dst_start..dst_start.saturating_add(per_leaf)) {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s;
                }
            }
        }
        BucketHistogram {
            data,
            n_leaves: self.n_leaves,
            n_features: self.n_features,
            n_bins: self.n_bins,
            n_channels: self.n_channels,
        }
    }

    /// Relocate this histogram's leaves into a larger `n_new_leaves` partition,
    /// copying source leaf `p`'s whole cell block into destination leaf `p + shift`.
    ///
    /// This is a byte-identical MEMCPY of already-folded cells — NO re-summation and
    /// NO reorder, so the relocated histogram carries the exact same bits as the
    /// source (parity-SAFE). It exists so [`GrowScratch::advance`] can produce the
    /// parent histogram in the child (n_new_leaves) layout WITHOUT a from-scratch
    /// object rebuild, then derive the larger sibling via [`remove`](Self::remove)
    /// (the subtraction trick's actual payoff, WR-04). Destination leaves outside
    /// `[shift, shift + n_leaves)` stay zero. On an index-arithmetic overflow the
    /// result is an empty degenerate histogram rather than a panic (T-21-02); source
    /// leaves whose destination falls outside the new partition are skipped
    /// defensively (no raw indexing, T-21-01).
    #[must_use]
    pub fn relocate(&self, n_new_leaves: usize, shift: usize) -> BucketHistogram {
        // Per-leaf contiguous block width (leaf is the OUTERMOST flat index).
        let per_leaf = self
            .n_features
            .checked_mul(self.n_bins)
            .and_then(|x| x.checked_mul(self.n_channels))
            .unwrap_or(0);
        let total = n_new_leaves.checked_mul(per_leaf).unwrap_or(0);
        let mut data = vec![0.0_f64; total];
        for p in 0..self.n_leaves {
            let dest = match p.checked_add(shift) {
                Some(d) if d < n_new_leaves => d,
                _ => continue,
            };
            let src_start = match p.checked_mul(per_leaf) {
                Some(s) => s,
                None => continue,
            };
            let dst_start = match dest.checked_mul(per_leaf) {
                Some(s) => s,
                None => continue,
            };
            let src = match self
                .data
                .get(src_start..src_start.saturating_add(per_leaf))
            {
                Some(s) => s,
                None => continue,
            };
            if let Some(dst) = data.get_mut(dst_start..dst_start.saturating_add(per_leaf)) {
                dst.copy_from_slice(src);
            }
        }
        BucketHistogram {
            data,
            n_leaves: n_new_leaves,
            n_features: self.n_features,
            n_bins: self.n_bins,
            n_channels: self.n_channels,
        }
    }
}

/// The bin of `value` under ascending `borders`: the count of borders STRICTLY
/// LESS than `value` (an upper-bound), consistent with the split test
/// `f64::from(value) > border` (`FeatureMatrix::passes_float`, `tree.rs:360-365`).
///
/// With `borders` ascending, object `obj` passes border `k` (`value > borders[k]`)
/// exactly when `borders[k] < value`, i.e. when `k < bin_of(borders, value)`. So a
/// split at border index `b` puts the FALSE child (`value <= borders[b]`) at
/// `bins <= b` and the TRUE child (`value > borders[b]`) at `bins > b` — the
/// boundary the prefix scan relies on (RESEARCH Pitfall 4). Values equal to a
/// border land in the lower bucket (strict `<`), below-min lands in bin `0`, and
/// above-max lands in bin `borders.len()` (`= n_bins - 1`).
#[must_use]
pub fn bin_of(borders: &[f64], value: f32) -> usize {
    let v = f64::from(value);
    borders.iter().filter(|&&b| b < v).count()
}

/// Build the per-`(leaf, feature, bin)` [`BucketHistogram`] in ONE object-order
/// pass (transcribes the device `host_reference_hist2`, `pointwise_hist.rs:106-163`,
/// generalizing [`reduce_leaf_stats`] from `leaf` to `(leaf, feature, bin)`).
///
/// - `bins` is the quantized bin matrix laid out FEATURE-major:
///   `bins[feature * n_objects + obj]` is object `obj`'s bin for `feature`
///   (produced by [`bin_of`]). `n_objects` is inferred from `leaf_of.len()`.
/// - `der1` is the DIMENSION-major first-derivative buffer:
///   `der1[d * n_objects + obj]`, length `approx_dimension * n_objects`.
/// - `weight` is per-object (length `n_objects`), shared across dimensions.
/// - `leaf_of[obj]` is object `obj`'s CURRENT-partition leaf index
///   (`0..n_leaves`); objects with a leaf `>= n_leaves` are ignored defensively.
///
/// Each cell's contributions are GATHERED in ascending object order, then folded
/// through [`cb_core::sum_f64`] (D-05) — never a raw iterator sum (D-08). Channel
/// `d` (`0..approx_dimension`) is `Σ der1[d]`, channel `approx_dimension` is
/// `Σ weight`. Out-of-range bins/objects are skipped (no raw indexing — workspace
/// deny `indexing_slicing`, T-21-01). On an index-arithmetic overflow of the flat
/// length the histogram is returned empty rather than panicking (T-21-02); the
/// depth cap (`MAX_DEPTH=16`, `tree.rs:100`) bounds `n_leaves` well below that.
///
/// The eight parameters mirror the frozen `(bins, der1, weight, leaf_of, shape…)`
/// contract of the device template (`pointwise_hist.rs`); bundling them into a
/// struct would obscure that one-to-one correspondence, so the arity allow is
/// deliberate.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn build_bucket_histogram(
    bins: &[u32],
    der1: &[f64],
    weight: &[f64],
    leaf_of: &[usize],
    n_leaves: usize,
    n_features: usize,
    n_bins: usize,
    approx_dimension: usize,
) -> BucketHistogram {
    let n_objects = leaf_of.len();
    let n_channels = approx_dimension + 1;
    // Checked flat length (T-21-02): overflow → empty degenerate histogram (no panic).
    let total = n_leaves
        .checked_mul(n_features)
        .and_then(|x| x.checked_mul(n_bins))
        .and_then(|x| x.checked_mul(n_channels))
        .unwrap_or(0);

    // Scatter-add each per-object contribution directly into ONE flat accumulator
    // in ascending OBJECT order (WR-02, PERF-03): no per-cell nested-Vec gather.
    // Because `scatter_add_f64` is the SCATTER form of the same left-to-right `+=`
    // fold as `cb_core::sum_f64`, folding a cell's members by repeated scatter-add in
    // this object-outer / feature-inner order is byte-identical to gathering them and
    // calling `sum_f64` (D-05/D-08 preserved with zero per-cell heap allocation).
    let mut data = vec![0.0_f64; total];

    for obj in 0..n_objects {
        let leaf = match leaf_of.get(obj) {
            Some(&l) if l < n_leaves => l,
            _ => continue,
        };
        let w = weight.get(obj).copied().unwrap_or(0.0);
        for feature in 0..n_features {
            let bin = match bins.get(feature * n_objects + obj) {
                Some(&b) => b as usize,
                None => continue,
            };
            if bin >= n_bins {
                continue;
            }
            let cell_base = ((leaf * n_features + feature) * n_bins + bin) * n_channels;
            for d in 0..approx_dimension {
                let dval = der1.get(d * n_objects + obj).copied().unwrap_or(0.0);
                cb_core::scatter_add_f64(&mut data, cell_base + d, dval);
            }
            cb_core::scatter_add_f64(&mut data, cell_base + approx_dimension, w);
        }
    }

    BucketHistogram {
        data,
        n_leaves,
        n_features,
        n_bins,
        n_channels,
    }
}

/// The `O(n_bins)` prefix scan for ONE candidate border on `feature`: split every
/// existing leaf into its FALSE child (`bins <= border`) and TRUE child
/// (`bins > border`) and emit the per-dimension canonical-leaf-order [`LeafStats`]
/// array ready for [`crate::multi_dim_split_score`] / [`crate::l2_split_score`].
///
/// The returned `Vec` is indexed `[dimension]`; each inner `Vec<LeafStats>` is in
/// canonical leaf order of length `2 * hist.n_leaves()`: index `parent` is the
/// FALSE child of parent leaf `parent`, index `parent + n_leaves` is its TRUE
/// child. This matches the forward-bit `leaf_index` convention (`tree.rs:284`)
/// where the new candidate occupies the highest bit
/// (`leaf = parent + (candidate ? n_leaves_parent : 0)`) — so the leaf ORDER is
/// byte-for-byte the `assign_leaves(chosen ++ candidate)` order the current
/// `score_candidate` produces (the forward-bit ORDER is unaffected by the
/// running-prefix summation and stays exact).
///
/// # TRUE-side strategy (the parity-relevant summation order, 21-06)
///
/// Each parent's per-bin row is combined in a SINGLE ascending pass (matching
/// upstream `CalcScoresForLeaf`'s running `trueStats = total − falseStats`
/// complement, `score_calcers.cpp`):
/// - **FALSE side** is a left-to-right running prefix `acc_false += bin[b]` in
///   ascending bin order. This is BIT-IDENTICAL to the previous
///   `sum_f64(bins[0..=border])` — a running `+=` from `acc = 0.0` in ascending
///   order IS the [`cb_core::sum_f64`] fold — so the FALSE side carries ZERO
///   parity risk.
/// - **TRUE side** is `true = total − acc_false`, where
///   `total = sum_f64(all bins ascending)` is computed ONCE per (parent, channel).
///   This `total − prefix` complement is `O(n_bins)` (not the old `O(n_bins²)`
///   per-border suffix re-sum) but it CHANGES the TRUE-side f64 bits versus the
///   previous ascending suffix `sum_f64(bins[border+1..n])`. So the FALSE side is
///   bit-identical while the TRUE side is only **≤1e-5 oracle-equivalent** (the
///   `total − prefix` reorder, gated by the full atomic oracle suite, 21-06).
#[must_use]
pub fn scan_border_to_leaf_stats(
    hist: &BucketHistogram,
    feature: usize,
    border: usize,
    approx_dimension: usize,
) -> Vec<Vec<LeafStats>> {
    let n_parent = hist.n_leaves();
    let n_bins = hist.n_bins();
    let n_channels = hist.n_channels();
    let weight_channel = approx_dimension; // channel index of Σ weight
    let mut out: Vec<Vec<LeafStats>> =
        vec![vec![LeafStats::default(); 2 * n_parent]; approx_dimension.max(1)];

    for parent in 0..n_parent {
        // The whole (parent, feature) block, read ONCE (constant-factor scan win):
        // cell (bin, channel) is at `bin * n_channels + channel`. A None block means
        // the parent/feature is out of range → all-zero row (default LeafStats).
        let block = match hist.feature_block(parent, feature) {
            Some(b) => b,
            None => continue,
        };
        // Per-bin weight column, gathered in bin order for the ordered sum_f64.
        let bin_weight: Vec<f64> = (0..n_bins)
            .map(|bin| block.get(bin * n_channels + weight_channel).copied().unwrap_or(0.0))
            .collect();
        // FALSE = ascending running prefix over bins 0..=border (bit-identical to
        // sum_f64(bins[0..=border])). TRUE = total − prefix (the reordered complement).
        let total_w = sum_f64(&bin_weight);
        let w_false = running_prefix(&bin_weight, border);
        let w_true = total_w - w_false;

        for d in 0..approx_dimension {
            let bin_delta: Vec<f64> = (0..n_bins)
                .map(|bin| block.get(bin * n_channels + d).copied().unwrap_or(0.0))
                .collect();
            let total_d = sum_f64(&bin_delta);
            let d_false = running_prefix(&bin_delta, border);
            let d_true = total_d - d_false;
            let false_stats = LeafStats {
                sum_weighted_delta: d_false,
                sum_weight: w_false,
            };
            let true_stats = LeafStats {
                sum_weighted_delta: d_true,
                sum_weight: w_true,
            };
            if let Some(row) = out.get_mut(d) {
                if let Some(slot) = row.get_mut(parent) {
                    *slot = false_stats;
                }
                if let Some(slot) = row.get_mut(parent + n_parent) {
                    *slot = true_stats;
                }
            }
        }
    }
    out
}

/// Ascending left-to-right running prefix `Σ row[0..=border]` — the FALSE-side
/// fold. Folding a slice's leading `border+1` elements with a zero-seeded `+=` in
/// ascending order is byte-for-byte [`cb_core::sum_f64`] of that leading run, so
/// the FALSE side stays bit-identical to the previous `sum_f64(bins[0..=border])`.
/// Out-of-range bins are skipped defensively (no raw indexing, T-21-01).
fn running_prefix(row: &[f64], border: usize) -> f64 {
    let mut acc = 0.0_f64;
    let last = border.min(row.len().saturating_sub(1));
    for b in 0..=last {
        acc += row.get(b).copied().unwrap_or(0.0);
    }
    acc
}

/// The full `O(n_borders · n_bins)` scan across all `n_borders` candidate
/// thresholds of `feature`, returning one per-dimension canonical-leaf-order
/// [`LeafStats`] set PER border. The result is indexed `[border][dimension]`;
/// `result[b]` is directly consumable by [`crate::multi_dim_split_score`] for
/// candidate border `b`.
///
/// This is the PRIMARY scan (WR-03): each parent's per-channel bin row is gathered
/// AT MOST ONCE per (parent, feature) — OUTSIDE the border loop — and its `total`
/// is summed once; the borders are then walked with a single carried running prefix
/// `acc_false += row[b]`, emitting `false = acc_false`, `true = total − acc_false`
/// per border. No per-border re-gather and no per-border `Vec` allocation (the old
/// path was `O(n_bins²)` and allocated fresh rows per border). The TRUE-side
/// `total − prefix` reorder is the exact one documented on [`scan_border_to_leaf_stats`]
/// (FALSE bit-identical, TRUE ≤1e-5 oracle-equivalent, gated by the 21-06 atomic
/// oracle suite).
#[must_use]
pub fn scan_borders_to_leaf_stats(
    hist: &BucketHistogram,
    feature: usize,
    n_borders: usize,
    approx_dimension: usize,
) -> Vec<Vec<Vec<LeafStats>>> {
    let n_parent = hist.n_leaves();
    let n_bins = hist.n_bins();
    let n_channels = hist.n_channels();
    let dim = approx_dimension.max(1);
    let weight_channel = approx_dimension; // channel index of Σ weight
    let mut out: Vec<Vec<Vec<LeafStats>>> =
        vec![vec![vec![LeafStats::default(); 2 * n_parent]; dim]; n_borders];

    // Reused per-parent gather scratch (allocated ONCE, not per border — PERF-03).
    let mut bin_weight = vec![0.0_f64; n_bins];
    let mut bin_delta = vec![vec![0.0_f64; n_bins]; approx_dimension];

    for parent in 0..n_parent {
        // The whole (parent, feature) block, read ONCE (constant-factor scan win):
        // cell (bin, channel) is at `bin * n_channels + channel`. A None block means
        // the parent/feature is out of range → all-zero row (default LeafStats).
        let block = match hist.feature_block(parent, feature) {
            Some(b) => b,
            None => continue,
        };
        // Gather each channel's per-bin row ONCE per (parent, feature) from the block.
        for bin in 0..n_bins {
            if let Some(slot) = bin_weight.get_mut(bin) {
                *slot = block.get(bin * n_channels + weight_channel).copied().unwrap_or(0.0);
            }
        }
        let total_w = sum_f64(&bin_weight);
        for d in 0..approx_dimension {
            for bin in 0..n_bins {
                let v = block.get(bin * n_channels + d).copied().unwrap_or(0.0);
                if let Some(row) = bin_delta.get_mut(d) {
                    if let Some(slot) = row.get_mut(bin) {
                        *slot = v;
                    }
                }
            }
        }
        // Precompute per-dimension totals once.
        let total_d: Vec<f64> = (0..approx_dimension)
            .map(|d| sum_f64(bin_delta.get(d).map_or(&[][..], Vec::as_slice)))
            .collect();

        // Single carried running prefix per channel across borders (FALSE side).
        let mut acc_false_w = 0.0_f64;
        let mut acc_false_d = vec![0.0_f64; approx_dimension];
        for b in 0..n_borders {
            acc_false_w += bin_weight.get(b).copied().unwrap_or(0.0);
            let w_false = acc_false_w;
            let w_true = total_w - acc_false_w;
            for d in 0..approx_dimension {
                let inc = bin_delta.get(d).and_then(|r| r.get(b)).copied().unwrap_or(0.0);
                let acc = match acc_false_d.get_mut(d) {
                    Some(a) => {
                        *a += inc;
                        *a
                    }
                    None => continue,
                };
                let td = total_d.get(d).copied().unwrap_or(0.0);
                let false_stats = LeafStats {
                    sum_weighted_delta: acc,
                    sum_weight: w_false,
                };
                let true_stats = LeafStats {
                    sum_weighted_delta: td - acc,
                    sum_weight: w_true,
                };
                if let Some(row) = out.get_mut(b).and_then(|per_dim| per_dim.get_mut(d)) {
                    if let Some(slot) = row.get_mut(parent) {
                        *slot = false_stats;
                    }
                    if let Some(slot) = row.get_mut(parent + n_parent) {
                        *slot = true_stats;
                    }
                }
            }
        }
    }
    out
}

/// FUSED single-pass border scan + split score (PERF-01 constant recovery,
/// PERF-03): returns the per-border candidate score directly, WITHOUT ever
/// materializing the full `Vec<Vec<Vec<LeafStats>>>` that
/// [`scan_borders_to_leaf_stats`] builds and WITHOUT the per-candidate score-side
/// `Vec<f64>` [`crate::multi_dim_split_score`] would allocate. For each candidate
/// border `b` (0-indexed, corresponding to bin `b`), the `2·n_parent` per-dimension
/// [`LeafStats`] are computed inline from a per-parent running prefix
/// (FALSE = running prefix, TRUE = per-parent total − prefix — the EXACT values
/// [`scan_borders_to_leaf_stats`] emits) into a single REUSED `per_dim` scratch, then
/// scored through [`crate::score::multi_dim_split_score_into`] with caller-reused
/// `num`/`den` fold buffers. The candidate order is border-ascending (index 0..
/// `n_borders`); `result[b]` is the score for border `b`.
///
/// # Bit-identity (the parity guard, 21-07)
/// The per-border score is byte-for-byte identical to
/// `multi_dim_split_score(score_function, &scan_borders_to_leaf_stats(..)[b],
/// scaled_l2)`. The fold is a FRESH `sum_f64` over all `2·n_parent` leaves in
/// canonical dimension-then-leaf order PER border — NO running/incremental num/den
/// is carried across borders (a cross-border reorder is FORBIDDEN, D-08 + the ≤1e-5
/// crux). Only the FALSE-side per-parent weight/delta running prefix accumulates
/// across borders (the same ascending `+=` `scan_borders_to_leaf_stats` already does,
/// bit-identical); the TRUE side is `total − prefix` (the already-gated 21-06 reorder,
/// unchanged). The fusion changes ONLY allocation lifetime, not arithmetic order —
/// `fused_scan_score_bit_identical` (histogram_test) asserts `to_bits()` equality.
///
/// # Performance shape (measured, PERF-01 honest framing)
/// At the CB_PERF harness (n=10000, nf=20, depth=6) per-tree time decomposes as a
/// flat floor ≈1.7ms (binning + the `O(n·nf)` scatter build) + ~0.026 ms/bin. The
/// n_bins-linear term IS this `O(n_bins·n_leaves·n_features)` split-scoring pass,
/// which is n-INDEPENDENT (the histogram cell count does not depend on the row
/// count) and hence ALGORITHMICALLY IRREDUCIBLE. Flatness across the n_bins sweep
/// requires n ≫ n_bins·n_leaves, which FAILS here (n_bins·n_leaves ≈ 16K > 10K) and
/// would only hold at n ≥ 100k; official CatBoost is itself ~2.1× (not flat) at this
/// size. This fused path recovers the parity-safe ALLOCATION constant (eliminating
/// the giant materialization + per-candidate score Vecs) — it does NOT, and cannot,
/// make the sweep flat at this harness size.
#[must_use]
pub fn scan_and_score_borders(
    hist: &BucketHistogram,
    feature: usize,
    n_borders: usize,
    approx_dimension: usize,
    score_function: crate::runtime::EScoreFunction,
    scaled_l2: f64,
) -> Vec<f64> {
    use crate::score::multi_dim_split_score_into;

    let n_parent = hist.n_leaves();
    let n_bins = hist.n_bins();
    let n_channels = hist.n_channels();
    let dim = approx_dimension.max(1);
    let weight_channel = approx_dimension; // channel index of Σ weight

    // Per-parent totals (`sum_f64` over the whole weight / delta column, ONCE per
    // parent — the same total `scan_borders_to_leaf_stats` computes). Flat arrays:
    // `total_w[parent]`, `total_d[parent * dim + d]`.
    let mut total_w = vec![0.0_f64; n_parent];
    let mut total_d = vec![0.0_f64; n_parent.saturating_mul(dim)];
    // Reused per-parent gather column for the total's ordered `sum_f64`.
    let mut col = vec![0.0_f64; n_bins];
    for parent in 0..n_parent {
        let block = match hist.feature_block(parent, feature) {
            Some(b) => b,
            None => continue, // out-of-range parent/feature → zero totals (empty row).
        };
        for bin in 0..n_bins {
            if let Some(slot) = col.get_mut(bin) {
                *slot = block.get(bin * n_channels + weight_channel).copied().unwrap_or(0.0);
            }
        }
        if let Some(slot) = total_w.get_mut(parent) {
            *slot = sum_f64(&col);
        }
        for d in 0..approx_dimension {
            for bin in 0..n_bins {
                if let Some(slot) = col.get_mut(bin) {
                    *slot = block.get(bin * n_channels + d).copied().unwrap_or(0.0);
                }
            }
            if let Some(slot) = total_d.get_mut(parent * dim + d) {
                *slot = sum_f64(&col);
            }
        }
    }

    // Per-parent FALSE-side running prefixes, carried across the border loop (the
    // ONLY cross-border accumulation — bit-identical ascending `+=`, the same one
    // `scan_borders_to_leaf_stats` does). `acc_false_w[parent]`,
    // `acc_false_d[parent * dim + d]`.
    let mut acc_false_w = vec![0.0_f64; n_parent];
    let mut acc_false_d = vec![0.0_f64; n_parent.saturating_mul(dim)];
    // REUSED per-border LeafStats scratch: `per_dim[d]` is `[2·n_parent]` leaves
    // (indices 0..n_parent = FALSE children, n_parent..2·n_parent = TRUE children),
    // in the exact canonical order `multi_dim_split_score` folds (dimension-then-leaf).
    let mut per_dim: Vec<Vec<LeafStats>> = vec![vec![LeafStats::default(); 2 * n_parent]; dim];
    // REUSED score-fold buffers (num / den) — zero per-candidate heap allocation.
    let mut num_scratch: Vec<f64> = Vec::new();
    let mut den_scratch: Vec<f64> = Vec::new();

    let mut scores: Vec<f64> = Vec::with_capacity(n_borders);
    for b in 0..n_borders {
        for parent in 0..n_parent {
            // Advance this parent's FALSE prefix by bin `b` (or leave it if the
            // parent/feature block is absent — its stats stay zero, matching the
            // `scan_borders_to_leaf_stats` `continue`).
            let block = hist.feature_block(parent, feature);
            let w_inc = block
                .and_then(|blk| blk.get(b * n_channels + weight_channel))
                .copied()
                .unwrap_or(0.0);
            let w_false = match acc_false_w.get_mut(parent) {
                Some(a) => {
                    *a += w_inc;
                    *a
                }
                None => continue,
            };
            let w_true = total_w.get(parent).copied().unwrap_or(0.0) - w_false;
            for d in 0..approx_dimension {
                let d_inc = block
                    .and_then(|blk| blk.get(b * n_channels + d))
                    .copied()
                    .unwrap_or(0.0);
                let acc = match acc_false_d.get_mut(parent * dim + d) {
                    Some(a) => {
                        *a += d_inc;
                        *a
                    }
                    None => continue,
                };
                let td = total_d.get(parent * dim + d).copied().unwrap_or(0.0);
                let false_stats = LeafStats {
                    sum_weighted_delta: acc,
                    sum_weight: w_false,
                };
                let true_stats = LeafStats {
                    sum_weighted_delta: td - acc,
                    sum_weight: w_true,
                };
                if let Some(row) = per_dim.get_mut(d) {
                    if let Some(slot) = row.get_mut(parent) {
                        *slot = false_stats;
                    }
                    if let Some(slot) = row.get_mut(parent + n_parent) {
                        *slot = true_stats;
                    }
                }
            }
        }
        // FRESH per-border fold over all 2·n_parent leaves in dimension-then-leaf
        // order (the reused scratch is cleared+refilled inside `_into`).
        let score = multi_dim_split_score_into(
            &mut num_scratch,
            &mut den_scratch,
            score_function,
            &per_dim,
            scaled_l2,
        );
        scores.push(score);
    }
    scores
}
