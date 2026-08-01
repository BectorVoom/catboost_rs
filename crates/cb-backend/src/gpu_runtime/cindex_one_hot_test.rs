//! T24 / SPEC-OH-21 — the packed cindex marks one-hot features truthfully, and
//! `TCFeature.folds` is pinned as the PADDED LINE WIDTH (never a candidate bound).
//!
//! # Why this lives here and not in `crates/cb-backend/tests/`
//!
//! Both fns call `super::cindex::pack_cindex` and `PackedCindex::device_arrays`,
//! which are `pub(crate)` inside a `pub(crate)` module — an integration test under
//! `tests/` links against the crate's PUBLIC surface only and cannot reach them. A
//! `#[cfg(test)] mod` declared in `gpu_runtime/mod.rs` is a descendant of
//! `gpu_runtime`, so it sees both (the same placement `session_residency` and
//! `session_depth_gt1_test` already use).

use super::cindex::pack_cindex;

/// The one-hot flag must reach `TCFeature.one_hot_feature` and be exported to the
/// device through `device_arrays()`'s FOURTH array, so the scorer can select
/// EQUALITY (`bin == value`) instead of THRESHOLD (`bin > value`) semantics.
///
/// A parallel host-side `Vec<bool>` would let the flag and the descriptor drift; the
/// flag rides the descriptor for the same reason `region_path` carries its own
/// one-hot bit inside the tuple.
#[test]
fn packed_cindex_marks_one_hot_features_truthfully() {
    let n = 8usize;
    // 2 float columns (32 buckets each) + 1 one-hot cat column (cardinality 2).
    let n_buckets = vec![32usize, 32, 2];
    let one_hot = vec![false, false, true];
    let mut bins = Vec::with_capacity(n_buckets.len() * n);
    for (f, &nb) in n_buckets.iter().enumerate() {
        for i in 0..n {
            bins.push(((i + f) % nb) as u32);
        }
    }

    let packed = pack_cindex(&bins, &n_buckets, &one_hot, n).expect("pack_cindex must succeed");

    let flags: Vec<bool> = packed.features.iter().map(|f| f.one_hot_feature).collect();
    assert_eq!(
        flags,
        vec![false, false, true],
        "`TCFeature.one_hot_feature` must mirror the caller's per-feature flag"
    );

    let (_offsets, _shifts, _masks, one_hot_flags) =
        packed.device_arrays().expect("device_arrays must succeed");
    assert_eq!(
        one_hot_flags,
        vec![0u32, 0, 1],
        "`device_arrays()` must export the one-hot flags as the FOURTH `u32` array \
         (the device index type is `u32`, so a `bool` array is not uploadable)"
    );
}

/// STANDING PIN (no Red→Green cycle of its own — it passes today and must keep
/// passing): `TCFeature.folds` is the PADDED UNIFORM LINE WIDTH on the production
/// path, NOT the feature's real cardinality.
///
/// `session.rs` packs with `vec![n_bins_line; eff_n_features]`, and `pack_cindex`
/// copies that argument straight into `folds`. So `folds[f] == n_bins_line` for
/// EVERY feature, and an eligibility test `border < folds[feature]` is the loop
/// bound itself — i.e. no bound at all. The real one-hot candidate bound travels
/// separately as `real_folds` (T24 step 1b → `DeviceTrainConfig`, T27b).
///
/// This test exists so that a future change repurposing `folds` as a candidate
/// bound fails loudly here instead of surfacing as an unlocalized ≤1e-5 device-vs-CPU
/// gap.
#[test]
fn packed_cindex_folds_is_the_padded_line_width_not_the_cardinality() {
    let n = 8usize;
    let n_features = 2usize;
    // Exactly how production calls it: ONE uniform padded line width for every
    // feature, regardless of the column's true cardinality.
    let n_bins_line = 32usize;
    let n_buckets_per_feature = vec![n_bins_line; n_features];
    let one_hot = vec![false, true];
    // The cat column's REAL cardinality is 2 — its bins only ever take 0 or 1.
    let mut bins = Vec::with_capacity(n_features * n);
    for i in 0..n {
        bins.push((i % n_bins_line) as u32);
    }
    for i in 0..n {
        bins.push((i % 2) as u32);
    }

    let packed =
        pack_cindex(&bins, &n_buckets_per_feature, &one_hot, n).expect("pack_cindex must succeed");

    let cat = packed
        .features
        .get(1)
        .expect("the cat feature descriptor must exist");
    assert_eq!(
        cat.folds, 32,
        "`TCFeature.folds` is the PADDED line width the caller supplied, NOT the \
         cardinality-2 truth — which is exactly why it must never bound a one-hot \
         candidate. The real bound is `real_folds`, built by \
         `quantize_feature_major_with_one_hot` and carried on `DeviceTrainConfig`."
    );
    assert!(
        cat.one_hot_feature,
        "the flag must still be set — `folds` being inert is a separate fact"
    );
}
