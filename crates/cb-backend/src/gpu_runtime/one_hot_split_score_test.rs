//! T25 / SPEC-OH-22 — the split scorer's one-hot (EQUALITY) fold arm.
//!
//! # Why this lives here and not in `crates/cb-backend/tests/`
//!
//! Every fn drives `super::score_partition_over_binsums`, a bare PRIVATE `fn` in
//! `gpu_runtime/mod.rs`. An integration test under `tests/` links against the crate's
//! public surface only and cannot reach it. A `#[cfg(test)] mod` declared in
//! `gpu_runtime/mod.rs` is a descendant of `gpu_runtime`, so it sees it — the same
//! placement `session_residency` and `session_depth_gt1_test` already use. The numeric
//! assertions run under the default `cpu` backend.
//!
//! # What the one-hot arm is
//!
//! With `one_hot == false` the fold is a PREFIX sum: `left = Σ bins[0..=border]`,
//! `right = Σ bins[border+1..]`, i.e. the threshold split `bin > border`. With
//! `one_hot == true` it is an EQUALITY split: `left = bins[value]`,
//! `right = total - left`, matching the CPU `FeatureMatrix::passes_one_hot`
//! (`IsTrueOneHotFeature(featureValue, splitValue) = featureValue == splitValue`).
//!
//! The histogram FILL is untouched — only the fold over it changes.

use cb_core::CbResult;

use super::{score_partition_over_binsums, BestSplit, SCORE_FN_L2};
use crate::kernels::REDUCE_FIXEDPOINT_SCALE_F64;
use crate::SelectedRuntime;

/// Host replica of the kernel's `fixedpoint_encode` (`round(v · 2^30) → i64 → u64`), so a
/// test can build a `bin_sums` buffer in exactly the layout the partition fill writes.
fn encode(v: f64) -> u64 {
    ((v * REDUCE_FIXEDPOINT_SCALE_F64).round() as i64) as u64
}

/// Build a partition histogram in the fill's layout:
/// `bin_sums[part * (n_features * n_bins * 2) + (feature * n_bins + bin) * 2 + channel]`,
/// channel 0 = Σ weight, channel 1 = Σ der1.
fn build_bin_sums(
    n_parts: usize,
    n_features: usize,
    n_bins: usize,
    cell: impl Fn(usize, usize, usize) -> (f64, f64),
) -> Vec<u64> {
    let mut out = vec![0u64; n_parts * n_features * n_bins * 2];
    for part in 0..n_parts {
        for f in 0..n_features {
            for b in 0..n_bins {
                let (w, d) = cell(part, f, b);
                let base = part * (n_features * n_bins * 2) + (f * n_bins + b) * 2;
                if let Some(slot) = out.get_mut(base) {
                    *slot = encode(w);
                }
                if let Some(slot) = out.get_mut(base + 1) {
                    *slot = encode(d);
                }
            }
        }
    }
    out
}

/// The CPU reference for ONE one-hot candidate `(feature, value)`: the L2 split score
/// summed over every active partition, with `left = bins[value]`, `right = total - left`.
fn cpu_one_hot_score(
    sums: &[(f64, f64)],
    n_parts: usize,
    n_bins: usize,
    value: usize,
    lambda: f64,
) -> f64 {
    // The FROZEN CPU L2 leaf term (`cb_leaf_score_term` / `cb-compute::score`):
    // `avg = sum / (w + lambda)` under a `w > 0` guard, then `term = avg * sum`.
    let term = |d: f64, w: f64| -> f64 {
        if w > 0.0 {
            d * d / (w + lambda)
        } else {
            0.0
        }
    };
    let mut acc = 0.0;
    for part in 0..n_parts {
        let mut total_w = 0.0;
        let mut total_d = 0.0;
        let mut left_w = 0.0;
        let mut left_d = 0.0;
        for b in 0..n_bins {
            let (w, d) = sums.get(part * n_bins + b).copied().unwrap_or((0.0, 0.0));
            total_w += w;
            total_d += d;
            if b == value {
                left_w = w;
                left_d = d;
            }
        }
        acc += term(left_d, left_w);
        acc += term(total_d - left_d, total_w - left_w);
    }
    acc
}

fn client() -> cubecl::client::ComputeClient<SelectedRuntime> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    <SelectedRuntime as cubecl::Runtime>::client(&device)
}

/// Drive `score_partition_over_binsums` for ONE pass over `[feature_lo, feature_hi)`.
#[allow(clippy::too_many_arguments)]
fn score_pass(
    bin_sums: &[u64],
    n_parts: usize,
    n_bins: usize,
    n_bins_used: usize,
    n_features: usize,
    real_folds: &[u32],
    one_hot: bool,
    feature_lo: usize,
    feature_hi: usize,
    lambda: f64,
) -> CbResult<Option<BestSplit>> {
    let client = client();
    let handle = client.create(cubecl::bytes::Bytes::from_elems(bin_sums.to_vec()));
    score_partition_over_binsums(
        &client,
        handle,
        n_parts,
        n_bins,
        n_bins_used,
        n_features,
        lambda,
        SCORE_FN_L2,
        real_folds,
        one_hot,
        feature_lo,
        feature_hi,
    )
}

/// Fn 1 — the one-hot fold must reproduce the CPU equality score AT DEPTH >= 2.
///
/// `n_parts = 4` (not 1) is load-bearing: the per-partition row pitch is
/// `leaf_stride = n_features * n_bins * 2` derived from the FULL feature count. A
/// single-partition test cannot expose a `leaf_stride` computed from a pass-narrowed
/// feature count, which is exactly the defect the "keep `n_features` full" constraint
/// prevents.
#[test]
fn one_hot_fold_matches_the_cpu_equality_score_at_depth_two() {
    let n_parts = 4usize;
    let n_float = 1usize;
    let n_cat = 2usize;
    let n_features = n_float + n_cat;
    let n_bins = 4usize;
    let lambda = 1.0_f64;
    let real_folds = vec![n_bins as u32, 4, 4];

    // Deterministic, distinguishable per (part, feature, bin) cells.
    let cell = |part: usize, f: usize, b: usize| -> (f64, f64) {
        let w = 1.0 + (part + f + b) as f64 * 0.25;
        let d = ((part * 7 + f * 3 + b) % 5) as f64 - 2.0;
        (w, d)
    };
    let bin_sums = build_bin_sums(n_parts, n_features, n_bins, cell);

    let best = score_pass(
        &bin_sums,
        n_parts,
        n_bins,
        n_bins,
        n_features,
        &real_folds,
        /* one_hot = */ true,
        /* feature_lo = */ n_float,
        /* feature_hi = */ n_features,
        lambda,
    )
    .expect("scorer must not error")
    .expect("a one-hot pass over 2 cat features must produce a winner");

    // Independent CPU argmax over the same candidate set.
    let mut cpu_best = (f64::NEG_INFINITY, usize::MAX, usize::MAX);
    for f in n_float..n_features {
        let sums: Vec<(f64, f64)> = (0..n_parts)
            .flat_map(|p| (0..n_bins).map(move |b| (p, b)))
            .map(|(p, b)| cell(p, f, b))
            .collect();
        for v in 0..n_bins {
            let s = cpu_one_hot_score(&sums, n_parts, n_bins, v, lambda);
            if s > cpu_best.0 {
                cpu_best = (s, f, v);
            }
        }
    }

    assert_eq!(
        (best.feature_id as usize, best.bin_id as usize),
        (cpu_best.1, cpu_best.2),
        "the device one-hot winner must be the CPU equality-fold argmax"
    );
    assert!(
        (f64::from(best.gain) - cpu_best.0).abs() <= 1e-5 * cpu_best.0.abs().max(1.0),
        "device gain {} vs CPU {} exceeds 1e-5",
        best.gain,
        cpu_best.0
    );
}

/// Fn 2 — the HIGHEST REAL BIN must be able to win, THROUGH
/// `score_partition_over_binsums` (not the raw kernel).
///
/// The threshold path excludes `border >= n_bins_used - 1` in TWO places: the kernel and
/// the host belt. Lifting only the kernel leaves the last category permanently
/// unselectable, and the raw-kernel test would not notice.
#[test]
fn one_hot_highest_real_bin_can_win_through_score_partition_over_binsums() {
    let n_parts = 1usize;
    let n_features = 1usize;
    let n_bins = 4usize;
    let real_folds = vec![4u32];
    let lambda = 1.0_f64;

    // Concentrate an extreme derivative on the HIGHEST real bin (index 3) so it is the
    // unambiguous equality-split winner.
    let bin_sums = build_bin_sums(n_parts, n_features, n_bins, |_p, _f, b| {
        if b == 3 {
            (1.0, 100.0)
        } else {
            (1.0, 0.0)
        }
    });

    let best = score_pass(
        &bin_sums,
        n_parts,
        n_bins,
        n_bins,
        n_features,
        &real_folds,
        true,
        0,
        n_features,
        lambda,
    )
    .expect("scorer must not error")
    .expect("the highest real bin must be a selectable candidate");

    assert_eq!(
        best.bin_id,
        real_folds[0] - 1,
        "the highest REAL bin must be selectable on the one-hot pass — the trailing-border \
         exclusion belongs to the THRESHOLD path only, and it lives in the host belt as \
         well as the kernel"
    );
}

/// Fn 3 — padded bins can NEVER win.
///
/// **Known blind spot, closed elsewhere:** this fn HAND-SUPPLIES `real_folds`, so it
/// cannot detect a wrong DATA SOURCE on the production path (e.g. wiring the padded
/// `TCFeature.folds` in). That assertion is
/// `device_one_hot_parity_with_a_padded_and_a_gap_bin` in
/// `crates/cb-train/tests/device_one_hot_parity_test.rs`, which runs through
/// `train` → `begin_device_training` → `grow_oblivious_tree_resident`.
#[test]
fn one_hot_padded_bins_never_win() {
    let n_parts = 1usize;
    let n_features = 1usize;
    let n_bins = 32usize; // the padded line width
    let real_cardinality = 2usize; // the column's TRUE cardinality
    let real_folds = vec![real_cardinality as u32];
    let lambda = 1.0_f64;

    // Seed a PADDED bin (index 17, well past the real cardinality) with the highest
    // possible score. A scorer that sweeps the whole padded line will pick it.
    let bin_sums = build_bin_sums(n_parts, n_features, n_bins, |_p, _f, b| {
        if b == 17 {
            (1.0, 1000.0)
        } else if b < real_cardinality {
            (1.0, 1.0)
        } else {
            (0.0, 0.0)
        }
    });

    let best = score_pass(
        &bin_sums,
        n_parts,
        n_bins,
        n_bins,
        n_features,
        &real_folds,
        true,
        0,
        n_features,
        lambda,
    )
    .expect("scorer must not error");

    if let Some(split) = best {
        assert!(
            (split.bin_id as usize) < real_cardinality,
            "a PADDED bin ({}) won the one-hot pass; candidates must be bounded by \
             `real_folds` ({real_cardinality}), never by the padded line width",
            split.bin_id
        );
    }
}

/// Fn 4 — the winner must report the ABSOLUTE device feature index.
///
/// Under a relative candidate space (`n_candidates = (feature_hi - feature_lo) * n_bins`)
/// a genuine pass-B winner would decode as `feature = absolute - feature_lo`, attributing
/// a one-hot split to a FLOAT feature.
#[test]
fn one_hot_winner_reports_the_absolute_device_feature_index() {
    let n_parts = 1usize;
    let n_float = 3usize;
    let n_features = 5usize; // device features 3 and 4 are one-hot
    let n_bins = 4usize;
    let real_folds = vec![4u32, 4, 4, 4, 4];
    let lambda = 1.0_f64;

    // Only device feature 4 carries signal.
    let bin_sums = build_bin_sums(n_parts, n_features, n_bins, |_p, f, b| {
        if f == 4 && b == 1 {
            (1.0, 50.0)
        } else if f == 4 {
            (1.0, 0.0)
        } else {
            (1.0, 0.0)
        }
    });

    let best = score_pass(
        &bin_sums,
        n_parts,
        n_bins,
        n_bins,
        n_features,
        &real_folds,
        true,
        n_float,
        n_features,
        lambda,
    )
    .expect("scorer must not error")
    .expect("device feature 4 carries a real candidate");

    assert_eq!(
        best.feature_id, 4,
        "the pass must report the ABSOLUTE device feature index (4), not the \
         pass-relative one (4 - feature_lo = 1), which would attribute a one-hot \
         split to a float feature"
    );
}

/// Fn 5 — a pass with no eligible candidate must produce NO winner.
///
/// Each pass seeds its no-winner sentinel to its OWN upper bound `hi`, which is `<=` the
/// full candidate count. The host must therefore skip `cand >= pass_hi`, not just
/// `cand >= n_features * n_bins`; otherwise a pass-B sentinel slips through the range
/// guard and an `f32::MIN`-adjacent gain is reported as a real winner.
#[test]
fn pass_b_with_no_eligible_candidate_produces_no_winner() {
    let n_parts = 1usize;
    let n_float = 1usize;
    let n_features = 2usize;
    let n_bins = 4usize;
    // The one-hot column's real cardinality is 0 — EVERY border is `>= real_folds[1]`,
    // so no candidate on the pass-B range is eligible.
    let real_folds = vec![4u32, 0];
    let lambda = 1.0_f64;

    let bin_sums = build_bin_sums(n_parts, n_features, n_bins, |_p, _f, _b| (1.0, 1.0));

    let best = score_pass(
        &bin_sums,
        n_parts,
        n_bins,
        n_bins,
        n_features,
        &real_folds,
        true,
        n_float,
        n_features,
        lambda,
    )
    .expect("scorer must not error");

    assert!(
        best.is_none(),
        "a pass whose every candidate is ineligible must return None, not a phantom \
         winner decoded from the pass's own sentinel"
    );
}

/// Fn 6 — the FLOAT-ONLY scorer output is numerically identical after the one-hot arm.
///
/// This asserts identity of the kernel's OUTPUT, not byte-identity of the kernel:
/// adding comptime parameters changes the generated kernel source by construction, so
/// only output identity is testable. The expected pair was captured from this exact
/// input before the one-hot arm landed; it must hold before AND after.
#[test]
fn float_only_scorer_output_is_numerically_identical_after_the_one_hot_arm() {
    let n_parts = 2usize;
    let n_features = 3usize;
    let n_bins = 8usize;
    let lambda = 2.0_f64;
    // On the float-only path production supplies `real_folds = [borders+1, …]`; it is
    // uploaded but NEVER read, because the `one_hot == false` arm keeps the unchanged
    // `border < max_border` eligibility.
    let real_folds = vec![n_bins as u32; n_features];

    let cell = |part: usize, f: usize, b: usize| -> (f64, f64) {
        let w = 1.0 + ((part * 5 + f * 3 + b) % 7) as f64;
        let d = ((part * 11 + f * 13 + b * 3) % 9) as f64 - 4.0;
        (w, d)
    };
    let bin_sums = build_bin_sums(n_parts, n_features, n_bins, cell);

    let best = score_pass(
        &bin_sums,
        n_parts,
        n_bins,
        n_bins,
        n_features,
        &real_folds,
        /* one_hot = */ false,
        /* feature_lo = */ 0,
        /* feature_hi = */ n_features,
        lambda,
    )
    .expect("scorer must not error")
    .expect("a float-only level must produce a winner");

    // The independent CPU reference: the PREFIX (threshold) fold, excluding the trailing
    // border, over the same cells. Bit-for-bit reproducible from the constants above.
    // The FROZEN CPU L2 leaf term (`cb_leaf_score_term` / `cb-compute::score`):
    // `avg = sum / (w + lambda)` under a `w > 0` guard, then `term = avg * sum`.
    let term = |d: f64, w: f64| -> f64 {
        if w > 0.0 {
            d * d / (w + lambda)
        } else {
            0.0
        }
    };
    let mut cpu_best = (f64::NEG_INFINITY, usize::MAX, usize::MAX);
    for f in 0..n_features {
        for border in 0..n_bins.saturating_sub(1) {
            let mut acc = 0.0;
            for part in 0..n_parts {
                let (mut lw, mut ld, mut rw, mut rd) = (0.0, 0.0, 0.0, 0.0);
                for b in 0..n_bins {
                    let (w, d) = cell(part, f, b);
                    if b <= border {
                        lw += w;
                        ld += d;
                    } else {
                        rw += w;
                        rd += d;
                    }
                }
                acc += term(ld, lw);
                acc += term(rd, rw);
            }
            if acc > cpu_best.0 {
                cpu_best = (acc, f, border);
            }
        }
    }

    assert_eq!(
        (best.feature_id as usize, best.bin_id as usize),
        (cpu_best.1, cpu_best.2),
        "the float-only winner must be unchanged by the one-hot arm"
    );
    assert!(
        (f64::from(best.gain) - cpu_best.0).abs() <= 1e-5 * cpu_best.0.abs().max(1.0),
        "float-only gain {} vs reference {}",
        best.gain,
        cpu_best.0
    );
}
