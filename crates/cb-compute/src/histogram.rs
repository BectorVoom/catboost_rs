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

    // Gather each (cell, channel) member list in ascending OBJECT order, then fold
    // each through the single sanctioned ordered primitive (the reduce_leaf_stats
    // shape, generalized). Scratch-buffer REUSE across levels is deferred to a
    // later wave (21-05, PERF-03); this primitive allocates fresh for clarity.
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); total];

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
                if let Some(slot) = members.get_mut(cell_base + d) {
                    slot.push(dval);
                }
            }
            if let Some(slot) = members.get_mut(cell_base + approx_dimension) {
                slot.push(w);
            }
        }
    }

    let mut data = vec![0.0_f64; total];
    for (i, slot) in data.iter_mut().enumerate() {
        let contributions = members.get(i).map_or(&[][..], Vec::as_slice);
        *slot = sum_f64(contributions);
    }

    BucketHistogram {
        data,
        n_leaves,
        n_features,
        n_bins,
        n_channels,
    }
}
