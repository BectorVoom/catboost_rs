//! T22 / FPP-19 — the ORDERED per-segment split scorer.
//!
//! # Why this lives here and not in `crates/cb-backend/tests/`
//!
//! Every fn drives `super::score_ordered_over_segment_binsums`, a bare PRIVATE `fn` in
//! `gpu_runtime/mod.rs`; an integration test under `tests/` sees the public surface only.
//! Same placement rationale as `one_hot_split_score_test`.
//!
//! NOT built under the default `cpu` backend: `find_optimal_split_ordered_kernel` grid-strides
//! over the `CUBE_COUNT` builtin, which cubecl-cpu rejects outright ("Unsupported builtin was
//! used: CubeCount") — every fn here would fail for that reason alone, indistinguishable from
//! a real regression. Run with a real device:
//!
//! ```text
//! cargo test -p cb-backend --no-default-features --features rocm \
//!     --lib gpu_runtime::ordered_split_score_test
//! ```
//!
//! Note this needs no `Atomic<u64>`: the histograms are built HOST-side in the fill's exact
//! layout, so only the scorer is under test.
//!
//! # What is under test
//!
//! `score_candidate_ordered` (`cb-train/src/tree.rs:2383`) scores a candidate as
//!
//! ```text
//!     Σ_s  l2_split_score( per-leaf stats over the permutation PREFIX [0, tail_finish_s),
//!                          scale_l2_reg(l2, body_sum_weight_s, body_finish_s) )
//! ```
//!
//! The reference below is a TRANSCRIPTION of that (cb-backend must not depend on cb-train —
//! the T-10-04 landmine), driven from raw objects. The device side is fed the same data as
//! `n_segments` concatenated PREFIX histograms in the partition fill's exact layout.
//!
//! The bar (PLAN T22): **integer equality on the chosen `(feature, border)`**, plus ε=1e-4 on
//! the summed score. The choice must be exact; the totals agree only to float tolerance
//! because the kernel runs one accumulator over `(seg, part, left/right)` while the CPU folds
//! per-leaf terms into a per-segment score and then sums those.

use super::{score_ordered_over_segment_binsums, BestSplit};
use crate::kernels::REDUCE_FIXEDPOINT_SCALE_F64;
use crate::SelectedRuntime;

/// `n = 30`, `fold_len_multiplier = 2.0` — the worked example documented at
/// `cb-train/src/fold.rs:135`: segments `[(1,2), (2,4), (4,8), (8,16), (16,30)]`.
const N: usize = 30;
const N_FEATURES: usize = 3;
/// The PADDED histogram line width the fill dispatches (the `{32,64,128,256}` family).
const N_BINS: usize = 32;
/// The ACTUAL quantized bin count — borders `0..N_BINS_USED-1` are the real candidates.
const N_BINS_USED: usize = 4;
const L2: f64 = 3.0;

/// The frozen `(body_finish, tail_finish)` pairs for `N = 30, multiplier = 2.0`.
/// Transcribed from `body_tail_segments`, NOT recomputed — this is the fixture.
const SEGMENTS: [(usize, usize); 5] = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 30)];

/// Host replica of the kernel's `fixedpoint_encode` (`round(v · 2^30) → i64 → u64`).
fn encode(v: f64) -> u64 {
    ((v * REDUCE_FIXEDPOINT_SCALE_F64).round() as i64) as u64
}

fn client() -> cubecl::client::ComputeClient<SelectedRuntime> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    <SelectedRuntime as cubecl::Runtime>::client(&device)
}

/// The fixture: per-object quantized bin per feature, der1, weight, and the learn
/// permutation. All deterministic closed forms — no RNG, so the fixture is frozen by
/// construction and a failure is reproducible.
struct Fixture {
    /// `bins[feature][obj]` in `0..N_BINS_USED`.
    bins: Vec<Vec<usize>>,
    der1: Vec<f64>,
    weight: Vec<f64>,
    /// The learn permutation: `permutation[position] = object id`.
    permutation: Vec<usize>,
}

impl Fixture {
    fn new() -> Self {
        // Three features with DIFFERENT bin patterns so the argmin has a genuine choice, and
        // magnitudes kept small so every fixed-point channel sum stays far under the
        // `|Σ| < 2^33` encode contract.
        let bins: Vec<Vec<usize>> = (0..N_FEATURES)
            .map(|f| {
                (0..N)
                    .map(|i| match f {
                        0 => i % N_BINS_USED,
                        1 => (i / 4) % N_BINS_USED,
                        _ => (i * 7 + 3) % N_BINS_USED,
                    })
                    .collect()
            })
            .collect();
        let der1 = (0..N)
            .map(|i| ((i as f64) * 0.37).sin() * 0.5 + 0.05 * (i as f64 % 3.0))
            .collect();
        let weight = (0..N).map(|i| 0.5 + 0.5 * ((i % 5) as f64) / 4.0).collect();
        // A genuinely non-identity permutation (a stride-7 walk over 30 is a full cycle since
        // gcd(7, 30) == 1), so a prefix over POSITIONS is not a prefix over object ids — which
        // is the property that would silently pass if the code confused the two.
        let permutation = (0..N).map(|p| (p * 7) % N).collect();
        Self {
            bins,
            der1,
            weight,
            permutation,
        }
    }

    /// `body_sum_weights`' transcription: `Σ weight[0..body_finish]` in ORIGINAL index order
    /// (`weights.get(..body_finish)` — deliberately NOT the permuted order).
    fn body_sum_weight(&self, body_finish: usize) -> f64 {
        self.weight.iter().take(body_finish).sum()
    }

    /// `scale_l2_reg(l2, sum_all_weights, doc_count)`.
    fn scaled_l2(&self, body_finish: usize) -> f64 {
        if body_finish == 0 {
            L2
        } else {
            L2 * (self.body_sum_weight(body_finish) / body_finish as f64)
        }
    }

    fn per_segment_lambda(&self) -> Vec<f64> {
        SEGMENTS
            .iter()
            .map(|&(body_finish, _)| self.scaled_l2(body_finish))
            .collect()
    }
}

/// The FROZEN CPU L2 leaf term (`add_leaf_plain` / `cb_leaf_score_term`):
/// `avg = sum / (w + lambda)` under the `w > 0` guard, then `term = avg * sum`.
fn leaf_term(d: f64, w: f64, lambda: f64) -> f64 {
    if w > 0.0 {
        d * d / (w + lambda)
    } else {
        0.0
    }
}

/// CPU reference for ONE candidate `(feature, border)`: the transcription of
/// `score_candidate_ordered`, driven from raw objects.
///
/// `part_of[obj]` is the partition the ALREADY-CHOSEN splits put the object in (all zeros at
/// level 0). The candidate adds one bit: `bin > border` ⇒ the right child.
fn cpu_ordered_score(
    fx: &Fixture,
    part_of: &[usize],
    n_parts: usize,
    feature: usize,
    border: usize,
) -> f64 {
    let mut segment_scores: Vec<f64> = Vec::with_capacity(SEGMENTS.len());
    for &(body_finish, tail_finish) in SEGMENTS.iter() {
        let lambda = fx.scaled_l2(body_finish);
        // `ordered_segment_leaf_stats` walks the permutation PREFIX [0, tail_finish) — it
        // takes `body_finish` and discards it (`let _ = body_finish;`, tree.rs:2317).
        let mut sum_d = vec![0.0_f64; n_parts * 2];
        let mut sum_w = vec![0.0_f64; n_parts * 2];
        for p in 0..tail_finish.min(fx.permutation.len()) {
            let obj = match fx.permutation.get(p) {
                Some(&o) => o,
                None => continue,
            };
            let bin = fx.bins.get(feature).and_then(|c| c.get(obj)).copied().unwrap_or(0);
            let part = part_of.get(obj).copied().unwrap_or(0);
            let side = usize::from(bin > border);
            let leaf = part * 2 + side;
            if let Some(slot) = sum_d.get_mut(leaf) {
                *slot += fx.der1.get(obj).copied().unwrap_or(0.0);
            }
            if let Some(slot) = sum_w.get_mut(leaf) {
                *slot += fx.weight.get(obj).copied().unwrap_or(0.0);
            }
        }
        // Per-segment L2 score over every leaf, left-then-right within each partition.
        let mut acc = 0.0;
        for leaf in 0..n_parts * 2 {
            acc += leaf_term(
                sum_d.get(leaf).copied().unwrap_or(0.0),
                sum_w.get(leaf).copied().unwrap_or(0.0),
                lambda,
            );
        }
        segment_scores.push(acc);
    }
    segment_scores.iter().sum()
}

/// The CPU reference argmin over the SAME candidate enumeration the device sweeps: feature
/// ascending, border ascending, strict `>` (first wins), trailing/padded borders excluded.
fn cpu_best_candidate(fx: &Fixture, part_of: &[usize], n_parts: usize) -> (usize, usize, f64) {
    let mut best = (usize::MAX, usize::MAX, f64::NEG_INFINITY);
    for feature in 0..N_FEATURES {
        for border in 0..N_BINS_USED - 1 {
            let score = cpu_ordered_score(fx, part_of, n_parts, feature, border);
            if score > best.2 {
                best = (feature, border, score);
            }
        }
    }
    best
}

/// Build the `n_segments` concatenated PREFIX histograms in the partition fill's layout:
/// `bin_sums[seg * seg_stride + part * (n_features * n_bins * 2) + (f * n_bins + bin) * 2 + ch]`,
/// channel 0 = Σ weight, channel 1 = Σ der1.
///
/// Segment `s`'s slice accumulates the permutation prefix `[0, tail_finish_s)` — exactly what
/// `launch_partition_hist2_resident_into` produces when driven with `indices = permutation`
/// and `n = tail_finish_s` (the T21 finding that removes the need for any segmented fill
/// kernel).
fn build_segment_bin_sums(fx: &Fixture, part_of: &[usize], n_parts: usize) -> Vec<u64> {
    let leaf_stride = N_FEATURES * N_BINS * 2;
    let seg_stride = n_parts * leaf_stride;
    let mut acc = vec![0.0_f64; SEGMENTS.len() * seg_stride];
    for (s, &(_, tail_finish)) in SEGMENTS.iter().enumerate() {
        for p in 0..tail_finish.min(fx.permutation.len()) {
            let obj = match fx.permutation.get(p) {
                Some(&o) => o,
                None => continue,
            };
            let part = part_of.get(obj).copied().unwrap_or(0);
            let d = fx.der1.get(obj).copied().unwrap_or(0.0);
            let w = fx.weight.get(obj).copied().unwrap_or(0.0);
            for f in 0..N_FEATURES {
                let bin = fx.bins.get(f).and_then(|c| c.get(obj)).copied().unwrap_or(0);
                let base = s * seg_stride + part * leaf_stride + (f * N_BINS + bin) * 2;
                if let Some(slot) = acc.get_mut(base) {
                    *slot += w;
                }
                if let Some(slot) = acc.get_mut(base + 1) {
                    *slot += d;
                }
            }
        }
    }
    acc.into_iter().map(encode).collect()
}

/// Drive the device ordered scorer over a prepared segmented histogram.
fn device_best(fx: &Fixture, part_of: &[usize], n_parts: usize) -> Option<BestSplit> {
    let bin_sums = build_segment_bin_sums(fx, part_of, n_parts);
    let client = client();
    let handle = client.create(cubecl::bytes::Bytes::from_elems(bin_sums));
    let real_folds = vec![N_BINS_USED as u32; N_FEATURES];
    match score_ordered_over_segment_binsums(
        &client,
        handle,
        n_parts,
        N_BINS,
        N_BINS_USED,
        N_FEATURES,
        &fx.per_segment_lambda(),
        &real_folds,
    ) {
        Ok(best) => best,
        Err(e) => panic!("ordered scorer returned an error: {e}"),
    }
}

/// Fn 1 — LEVEL 0 (`n_parts == 1`): the device ordered scorer must choose the SAME
/// `(feature, border)` as the transcribed `select_level_ordered`, and agree on the score.
#[test]
fn ordered_scorer_matches_cpu_reference_at_level_0() {
    let fx = Fixture::new();
    let part_of = vec![0usize; N];
    let (cpu_f, cpu_b, cpu_score) = cpu_best_candidate(&fx, &part_of, 1);

    let best = match device_best(&fx, &part_of, 1) {
        Some(b) => b,
        None => panic!("device ordered scorer found no candidate at level 0"),
    };

    assert_eq!(
        (best.feature_id as usize, best.bin_id as usize),
        (cpu_f, cpu_b),
        "level 0: device chose (f={}, b={}), CPU reference chose (f={cpu_f}, b={cpu_b})",
        best.feature_id,
        best.bin_id,
    );
    let diff = (f64::from(best.gain) - cpu_score).abs();
    assert!(
        diff <= 1e-4,
        "level 0 summed score: device {} vs CPU {cpu_score} (|diff| = {diff} > 1e-4)",
        best.gain,
    );
}

/// Fn 2 — LEVEL 1 (`n_parts == 2`): the per-partition row pitch is load-bearing. A scorer
/// that derived `leaf_stride` from anything but the FULL feature count, or that forgot the
/// `seg * seg_stride` outer pitch, reads a neighbouring partition's or segment's row and
/// silently scores the wrong histogram — which this catches and level 0 cannot.
#[test]
fn ordered_scorer_matches_cpu_reference_at_level_1() {
    let fx = Fixture::new();
    // One already-chosen split defines the two partitions: feature 1, border 1.
    let part_of: Vec<usize> = (0..N)
        .map(|obj| {
            let bin = fx.bins.get(1).and_then(|c| c.get(obj)).copied().unwrap_or(0);
            usize::from(bin > 1)
        })
        .collect();
    let (cpu_f, cpu_b, cpu_score) = cpu_best_candidate(&fx, &part_of, 2);

    let best = match device_best(&fx, &part_of, 2) {
        Some(b) => b,
        None => panic!("device ordered scorer found no candidate at level 1"),
    };

    assert_eq!(
        (best.feature_id as usize, best.bin_id as usize),
        (cpu_f, cpu_b),
        "level 1: device chose (f={}, b={}), CPU reference chose (f={cpu_f}, b={cpu_b})",
        best.feature_id,
        best.bin_id,
    );
    let diff = (f64::from(best.gain) - cpu_score).abs();
    assert!(
        diff <= 1e-4,
        "level 1 summed score: device {} vs CPU {cpu_score} (|diff| = {diff} > 1e-4)",
        best.gain,
    );
}

/// Fn 3 — the segments must be NESTED PREFIXES, not disjoint ranges.
///
/// This is the T21 correction expressed as a test. Building the histogram from the DISJOINT
/// ranges `[body_finish, tail_finish)` — the reading `WAVE7-SPIKES.md` assumed — produces a
/// different summed score than the prefix `[0, tail_finish)`. If the two ever agreed, this
/// fixture would not be discriminating and the other two tests would prove less than they
/// claim.
#[test]
fn disjoint_range_segments_would_score_differently_than_nested_prefixes() {
    let fx = Fixture::new();
    let part_of = vec![0usize; N];

    let prefix_score = cpu_ordered_score(&fx, &part_of, 1, 0, 1);

    // The same fold, but each segment restricted to [body_finish, tail_finish).
    let mut disjoint_total = 0.0;
    for &(body_finish, tail_finish) in SEGMENTS.iter() {
        let lambda = fx.scaled_l2(body_finish);
        let mut sum_d = vec![0.0_f64; 2];
        let mut sum_w = vec![0.0_f64; 2];
        for p in body_finish..tail_finish.min(fx.permutation.len()) {
            let obj = match fx.permutation.get(p) {
                Some(&o) => o,
                None => continue,
            };
            let bin = fx.bins.first().and_then(|c| c.get(obj)).copied().unwrap_or(0);
            let side = usize::from(bin > 1);
            if let Some(slot) = sum_d.get_mut(side) {
                *slot += fx.der1.get(obj).copied().unwrap_or(0.0);
            }
            if let Some(slot) = sum_w.get_mut(side) {
                *slot += fx.weight.get(obj).copied().unwrap_or(0.0);
            }
        }
        for leaf in 0..2 {
            disjoint_total += leaf_term(
                sum_d.get(leaf).copied().unwrap_or(0.0),
                sum_w.get(leaf).copied().unwrap_or(0.0),
                lambda,
            );
        }
    }

    assert!(
        (prefix_score - disjoint_total).abs() > 1e-3,
        "the prefix and disjoint readings scored the same ({prefix_score} vs {disjoint_total}); \
         this fixture cannot discriminate the T21 correction"
    );
}

/// Fn 4 — THE INTEGRATION TEST: build the per-segment histograms **on the device** via
/// [`super::launch_partition_hist2_prefix_into`] and still match the CPU ordered reference.
///
/// Fns 1–2 fed the scorer histograms assembled HOST-side, which proves the scorer's segment
/// fold but assumes the fill can produce those histograms. This closes that gap, and it is the
/// test that actually exercises the T21 finding: the UNCHANGED partition fill, driven with
/// `indices = permutation` and `n_visit = tail_finish_s`, IS the segmented fill.
///
/// The specific bug it guards is the `n` vs `n_visit` split. The fill declares `der1`,
/// `weight` and `leaf_of` with length `n` and bounds its loop by `indices.len()`. Passing a
/// single shortened `n` for both — the obvious way to write this — reads those object arrays
/// out of bounds for every permutation position whose object id exceeds the prefix length.
/// The stride-7 permutation makes that reachable at the very first segment.
///
/// Needs `Atomic<u64>`, so unlike fns 1–3 this one is rocm/cuda-only in substance as well as
/// in build configuration.
#[test]
fn device_prefix_fill_feeds_the_ordered_scorer() {
    use super::cindex::pack_cindex;
    use super::{launch_copy_u64_block, launch_partition_hist2_prefix_into};

    let fx = Fixture::new();
    let part_of = vec![0usize; N];
    let n_parts = 1usize;
    let client = client();

    // Feature-major bins, one bucket-width per feature equal to the padded line so the
    // packed field and the histogram line agree exactly.
    let mut bins: Vec<u32> = Vec::with_capacity(N_FEATURES * N);
    for f in 0..N_FEATURES {
        for obj in 0..N {
            bins.push(fx.bins.get(f).and_then(|c| c.get(obj)).copied().unwrap_or(0) as u32);
        }
    }
    let n_buckets = vec![N_BINS; N_FEATURES];
    let one_hot = vec![false; N_FEATURES];
    let packed = match pack_cindex(&bins, &n_buckets, &one_hot, N) {
        Ok(p) => p,
        Err(e) => panic!("pack_cindex failed: {e}"),
    };
    let (offsets, shifts, masks, _flags) = match packed.device_arrays() {
        Ok(a) => a,
        Err(e) => panic!("device_arrays failed: {e}"),
    };

    let der1_f: Vec<f64> = fx.der1.clone();
    let weight_f: Vec<f64> = fx.weight.clone();
    let der1_h = client.create(cubecl::bytes::Bytes::from_elems(der1_f));
    let weight_h = client.create(cubecl::bytes::Bytes::from_elems(weight_f));
    let words_h = client.create(cubecl::bytes::Bytes::from_elems(packed.words.clone()));
    let offsets_h = client.create(cubecl::bytes::Bytes::from_elems(offsets));
    let shifts_h = client.create(cubecl::bytes::Bytes::from_elems(shifts));
    let masks_h = client.create(cubecl::bytes::Bytes::from_elems(masks));
    // `indices` IS the learn permutation — the whole trick.
    let perm_u32: Vec<u32> = fx.permutation.iter().map(|&p| p as u32).collect();
    let perm_h = client.create(cubecl::bytes::Bytes::from_elems(perm_u32));
    let leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; N]));

    let leaf_stride = N_FEATURES * N_BINS * 2;
    let seg_stride = n_parts * leaf_stride;
    let total_len = SEGMENTS.len() * seg_stride;
    let packed_hist = client.empty(total_len * std::mem::size_of::<u64>());

    for (s, &(_, tail_finish)) in SEGMENTS.iter().enumerate() {
        let seg_h = match launch_partition_hist2_prefix_into(
            &client,
            der1_h.clone(),
            weight_h.clone(),
            words_h.clone(),
            offsets_h.clone(),
            shifts_h.clone(),
            masks_h.clone(),
            perm_h.clone(),
            leaf_of_h.clone(),
            packed.words.len(),
            N,
            tail_finish,
            N_BINS,
            N_FEATURES,
            /* level = */ 0,
        ) {
            Ok(h) => h,
            Err(e) => panic!("prefix fill failed for segment {s}: {e}"),
        };
        if let Err(e) = launch_copy_u64_block(
            &client,
            seg_h,
            seg_stride,
            packed_hist.clone(),
            total_len,
            s * seg_stride,
        ) {
            panic!("copy into segment slot {s} failed: {e}");
        }
    }

    let real_folds = vec![N_BINS_USED as u32; N_FEATURES];
    let best = match score_ordered_over_segment_binsums(
        &client,
        packed_hist,
        n_parts,
        N_BINS,
        N_BINS_USED,
        N_FEATURES,
        &fx.per_segment_lambda(),
        &real_folds,
    ) {
        Ok(Some(b)) => b,
        Ok(None) => panic!("device found no candidate over the device-filled histograms"),
        Err(e) => panic!("ordered scorer errored over device-filled histograms: {e}"),
    };

    let (cpu_f, cpu_b, cpu_score) = cpu_best_candidate(&fx, &part_of, n_parts);
    assert_eq!(
        (best.feature_id as usize, best.bin_id as usize),
        (cpu_f, cpu_b),
        "device-filled: chose (f={}, b={}), CPU reference chose (f={cpu_f}, b={cpu_b})",
        best.feature_id,
        best.bin_id,
    );
    let diff = (f64::from(best.gain) - cpu_score).abs();
    assert!(
        diff <= 1e-4,
        "device-filled summed score: {} vs CPU {cpu_score} (|diff| = {diff} > 1e-4)",
        best.gain,
    );
}
