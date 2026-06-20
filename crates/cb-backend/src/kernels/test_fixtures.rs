//! IN-04: shared `#[cfg(test)]` fixture-construction primitives for the Phase 7.5
//! cross-oracle test modules (`kernels::score_split`, `kernels::grow_loop`).
//!
//! The five fixture builders across those two files (`make_fixture`,
//! `make_boosting_fixture`, `make_ranking_fixture`, `make_score_fixture`,
//! `make_pairwise_fixture`) are NOT byte-identical — they have distinct names and produce
//! different oracle inputs (der1- vs target-based ramps, with/without a weight channel,
//! with/without a global competitor-pair list). But they share the SAME deterministic
//! construction of each individual channel. These primitives factor ONLY that genuinely
//! shared per-channel construction; each builder still composes the exact same bytes it
//! produced before (the consolidation PRESERVES every oracle input — see the
//! per-primitive contract below).
//!
//! All `pub(super)` so the sibling `#[cfg(test)] mod`s under `kernels` can compose them
//! without widening any non-test API surface.

/// The centred ramp `der1[k] = (k as f64) - (n as f64) / 2.0`, length `n`.
///
/// This is the IDENTICAL expression every builder used for its primary signal channel:
/// the der1 ramp (`make_fixture`/`make_score_fixture`/`make_ranking_fixture`/
/// `make_pairwise_fixture`) and the regression target ramp (`make_boosting_fixture` —
/// the boosting input is the SAME `(k - n/2)` ramp, only semantically a target). Folded
/// left-to-right in object order, byte-for-byte what the inlined `.map(...)` produced.
pub(super) fn ramp_centred(n: usize) -> Vec<f64> {
    (0..n).map(|k| (k as f64) - (n as f64) / 2.0).collect()
}

/// The non-trivial per-object weight channel `weight[k] = 0.5 + ((k % 5) as f64) * 0.25`,
/// length `n` (never all-1, so the weight denominator is a real sum).
///
/// IDENTICAL across `make_fixture`, `make_boosting_fixture`, `make_ranking_fixture`,
/// `make_score_fixture`. (`make_pairwise_fixture` has NO weight channel and does not call
/// this — its output is unchanged.)
pub(super) fn weight_mod5(n: usize) -> Vec<f64> {
    (0..n).map(|k| 0.5 + ((k % 5) as f64) * 0.25).collect()
}

/// The feature-major quantized `cindex` (`cindex[feature * n + obj]`, length
/// `n_features * n`): feature 0's bins climb monotonically with the object index
/// (`((obj * n_bins) / n.max(1)).min(n_bins - 1)` — aligning the ramp with the bin axis
/// for a clear best border), every other feature gets the deterministic lower-gain spread
/// `(obj * (feature + 2) + feature) % n_bins`.
///
/// IDENTICAL across all five builders (the pairwise builder named the object count
/// `n_objects`, but `n == n_objects`, so the bytes match exactly).
pub(super) fn cindex_feature_major(n: usize, n_features: usize, n_bins: usize) -> Vec<u32> {
    let mut cindex = vec![0u32; n_features * n];
    for feature in 0..n_features {
        for obj in 0..n {
            let bin = if feature == 0 {
                ((obj * n_bins) / n.max(1)).min(n_bins - 1)
            } else {
                (obj * (feature + 2) + feature) % n_bins
            };
            cindex[feature * n + obj] = bin as u32;
        }
    }
    cindex
}

/// The object visiting order `indices[k] = k`, length `n`. IDENTICAL across all five.
pub(super) fn indices_identity(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// The global competitor-pair list (the PairLogit/ranking adjacency): for each winner `w`
/// the losers `l` in the sliding window `(w + 1)..(w + 4).min(n)`, with the non-trivial
/// weight `0.5 + ((w + l) % 5) as f64 * 0.25`. Returns `(pair_i, pair_j, pair_weight)`.
///
/// IDENTICAL across `make_ranking_fixture` and `make_pairwise_fixture` (the only two
/// builders with a pair list); both produced these exact bytes inline.
pub(super) fn competitor_pairs(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut pair_i: Vec<u32> = Vec::new();
    let mut pair_j: Vec<u32> = Vec::new();
    let mut pair_weight: Vec<f64> = Vec::new();
    for w in 0..n {
        for l in (w + 1)..(w + 4).min(n) {
            pair_i.push(w as u32);
            pair_j.push(l as u32);
            pair_weight.push(0.5 + ((w + l) % 5) as f64 * 0.25);
        }
    }
    (pair_i, pair_j, pair_weight)
}
