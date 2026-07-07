//! Phase 21.5 Plan 01 — RED-first BYTE-IDENTITY guard (D-04/D-05/D-06).
//!
//! Locks the fused single-feature accumulate+score primitive
//! ([`fused_feature_scan_and_score`]) to the pre-change two-pass path
//! (`build_bucket_histogram`(all features) + `scan_and_score_borders`(feature))
//! BEFORE `tree.rs` (Plans 02/03) moves accumulation into the parallel region.
//!
//! Two guards, both asserting `f64::to_bits` equality (NOT approx / NOT ≤1e-5):
//!   1. `fused_equals_two_pass_level0` — level-0 fresh build, on TWO contrasting
//!      shapes (wide-feature nf=20/nbins=128 AND low-feature nf=8/nbins=254). Bins
//!      are feature-major, so a 1-feature build over a contiguous column folds each
//!      cell's members in the SAME ascending-object-order `sum_f64` as the
//!      whole-partition build → byte-for-byte identical cells and scores (D-03).
//!   2. `fused_subtraction_trick_equals_full_build` — a 2-leaf parent split into 4
//!      leaves, deriving the per-feature children via the subtraction trick
//!      (smaller sibling built into a reusable buffer, larger = `relocate_sub`,
//!      reunited via `add_relocated`) and asserting it is `to_bits`-equal, cell by
//!      cell, to a fresh whole-partition 4-leaf build — then that scoring both is
//!      identical too (D-06, the path Plan 02 relies on). Integer derivatives keep
//!      the `parent − false == true` subtraction bit-exact (mirrors the existing
//!      `bucket_histogram_remove_equals_fresh_sibling` convention).
//!
//! This guard is a STANDARD (always-run) regression net — NOT `CB_PERF`-gated — and
//! MUST stay green through Plans 02/03.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use cb_compute::{
    build_bucket_histogram, build_bucket_histogram_into, fused_feature_scan_and_score,
    scan_and_score_borders, EScoreFunction, FusedFeatureScratch,
};

/// Lifted from `spike006_fused_parallel_test.rs::make_inputs`: same splitmix64
/// generator — feature-major bins + a per-object, DIMENSION-major der1
/// (`der1[d * n + i]`, `dim` output channels) + unit weight + a mid-tree partition.
/// For `dim == 1` this produces EXACTLY the same `der1` values as the original
/// single-dimension generator (`(f + 0) % 2 == 0` reduces to `f % 2 == 0`), so the
/// existing dim=1 callers are unaffected byte-for-byte (WR-01).
fn make_inputs(
    n: usize,
    nf: usize,
    nbins: usize,
    n_leaves: usize,
    dim: usize,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut bins = vec![0u32; nf * n];
    let mut der1 = vec![0.0f64; dim.max(1) * n];
    for f in 0..nf {
        for i in 0..n {
            let mut z = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((f as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let v = (z >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            bins[f * n + i] = ((v * nbins as f64) as usize).min(nbins - 1) as u32;
            if f < 5 {
                for d in 0..dim.max(1) {
                    der1[d * n + i] += v * if (f + d) % 2 == 0 { 1.0 } else { -1.0 };
                }
            }
        }
    }
    let weight = vec![1.0f64; n];
    let leaf_of: Vec<usize> = (0..n).map(|i| i % n_leaves).collect();
    (bins, der1, weight, leaf_of)
}

/// One level-0 shape: fused per-feature scores == two-pass scores, byte-for-byte,
/// for EVERY (feature, border), at the given `dim` (`approx_dimension`). One
/// `FusedFeatureScratch` is reused across features (D-07) — proving reuse does not
/// perturb the bits. `dim >= 2` exercises the multiclass fused path (WR-01): the
/// per-dimension FALSE prefix / `total_d` indexing in `scan_and_score_borders_into`
/// (`histogram.rs`) and the per-channel build in `build_bucket_histogram_into`.
fn check_level0(n: usize, nf: usize, nbins: usize, n_leaves: usize, dim: usize) {
    let sf = EScoreFunction::L2;
    let l2 = 3.0f64;
    let n_borders = nbins - 1;
    let (bins, der1, weight, leaf_of) = make_inputs(n, nf, nbins, n_leaves, dim);

    // Reference: serial whole-partition build over ALL features, then per-feature score.
    let hist = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, n_leaves, nf, nbins, dim);

    let mut scratch = FusedFeatureScratch::new();
    for f in 0..nf {
        let reference = scan_and_score_borders(&hist, f, n_borders, dim, sf, l2);
        // Candidate: fused 1-feature build+score over feature f's contiguous column.
        let col = &bins[f * n..(f + 1) * n];
        let fused = fused_feature_scan_and_score(
            &mut scratch,
            col,
            &der1,
            &weight,
            &leaf_of,
            n_leaves,
            nbins,
            dim,
            n_borders,
            sf,
            l2,
        );
        assert_eq!(
            reference.len(),
            fused.len(),
            "border count mismatch at feature {f} (shape n={n} nf={nf} nbins={nbins} dim={dim})"
        );
        for b in 0..reference.len() {
            assert_eq!(
                reference[b].to_bits(),
                fused[b].to_bits(),
                "fused diverged from two-pass at (feature {f}, border {b}) \
                 for shape n={n} nf={nf} nbins={nbins} n_leaves={n_leaves} dim={dim}"
            );
        }
    }
}

#[test]
fn fused_equals_two_pass_level0() {
    // Wide-feature AND low-feature/high-bin — the two contrasting shapes (D-05).
    check_level0(10_000, 20, 128, 32, 1);
    check_level0(40_000, 8, 254, 16, 1);
}

/// WR-01: the byte-identity guard above only ever exercised `dim == 1`, but
/// `select_level_plain` / `select_level_perturbed` are reachable with
/// `approx_dimension > 1` (oblivious multiclass training). Lock the fused path at
/// `dim >= 2` on the same two contrasting shapes, so a per-dimension channel-offset
/// or `total_d` mis-indexing regression trips this always-run net.
#[test]
fn fused_equals_two_pass_level0_multiclass() {
    check_level0(10_000, 20, 128, 32, 3);
    check_level0(40_000, 8, 254, 16, 2);
}

#[test]
fn fused_subtraction_trick_equals_full_build() {
    // Integer derivatives + unit weights → the `parent − false == true` subtraction
    // is bit-exact (integer f64 sums/differences under 2^53 are exact), so the
    // subtraction-trick derivation is byte-identical to a fresh build (mirrors
    // `bucket_histogram_remove_equals_fresh_sibling`).
    let nf = 6usize;
    let n = 64usize;
    let nbins = 8usize;
    let dim = 1usize;
    let n_channels = dim + 1; // per the frozen layout: der channels + 1 weight channel
    let sf = EScoreFunction::L2;
    let l2 = 3.0f64;
    let n_borders = nbins - 1;

    // Feature-major integer bins; a deterministic spread across the bin range.
    let mut bins = vec![0u32; nf * n];
    for f in 0..nf {
        for i in 0..n {
            bins[f * n + i] = ((i * 7 + f * 3) % nbins) as u32;
        }
    }
    // Integer der1 in {-5..5}, exact under f64.
    let der1: Vec<f64> = (0..n).map(|i| (i % 11) as f64 - 5.0).collect();
    let weight = vec![1.0f64; n];

    // Base 2-leaf parent partition.
    let leaf_of2: Vec<usize> = (0..n).map(|i| i % 2).collect();
    // Split on feature 0 at border index `sb` (FALSE: bin <= sb, TRUE: bin > sb).
    let sb = 3usize;
    let passes: Vec<bool> = (0..n).map(|i| bins[i] as usize > sb).collect();
    // Forward-bit 4-leaf assignment: FALSE child stays at `parent`, TRUE child at
    // `parent + n_parent` (the candidate takes the high bit; n_parent = 2).
    let leaf_of4: Vec<usize> = (0..n)
        .map(|i| leaf_of2[i] + if passes[i] { 2 } else { 0 })
        .collect();
    // Smaller sibling = the FALSE children in the 2-leaf layout; TRUE objects are
    // sent out of range (leaf 2 >= n_leaves=2) so they are ignored on this build.
    let leaf_of_false: Vec<usize> = (0..n)
        .map(|i| if passes[i] { 2 } else { leaf_of2[i] })
        .collect();

    let mut small_data: Vec<f64> = Vec::new();
    for f in 0..nf {
        let col = &bins[f * n..(f + 1) * n];
        // Fresh whole-partition 4-leaf build for this feature (the reference).
        let full_f = build_bucket_histogram(col, &der1, &weight, &leaf_of4, 4, 1, nbins, dim);
        // Parent (2-leaf) and the smaller (FALSE) sibling (2-leaf).
        let parent2 = build_bucket_histogram(col, &der1, &weight, &leaf_of2, 2, 1, nbins, dim);
        let small = build_bucket_histogram(col, &der1, &weight, &leaf_of_false, 2, 1, nbins, dim);

        // Lock the buffer-reusing `build_bucket_histogram_into` to the same cells as
        // the allocating `build_bucket_histogram` for the smaller sibling (D-07):
        // the fields are private, so compare via the public `channel` accessor and
        // the known frozen flat layout `((leaf*1 + 0)*nbins + bin)*n_channels + c`.
        build_bucket_histogram_into(
            &mut small_data,
            col,
            &der1,
            &weight,
            &leaf_of_false,
            2,
            1,
            nbins,
            dim,
        );
        for leaf in 0..2 {
            for bin in 0..nbins {
                for c in 0..n_channels {
                    let idx = (leaf * nbins + bin) * n_channels + c;
                    assert_eq!(
                        small_data[idx].to_bits(),
                        small.channel(leaf, 0, bin, c).to_bits(),
                        "build_bucket_histogram_into cell mismatch at feature {f} \
                         leaf {leaf} bin {bin} channel {c}"
                    );
                }
            }
        }

        // Subtraction-trick derivation of the 4-leaf histogram:
        //   larger (TRUE, high slots) = relocate(parent, +2) − relocate(FALSE, +2)
        //   derived                    = larger + relocate(FALSE, +0)
        let minus = small.relocate(4, 2);
        let larger = parent2.relocate_sub(4, 2, &minus);
        let derived = larger.add_relocated(&small, 0);

        // Cell-by-cell byte-identity with the fresh full build.
        for leaf in 0..4 {
            for bin in 0..nbins {
                for c in 0..n_channels {
                    assert_eq!(
                        derived.channel(leaf, 0, bin, c).to_bits(),
                        full_f.channel(leaf, 0, bin, c).to_bits(),
                        "subtraction-trick cell diverged at feature {f} \
                         leaf {leaf} bin {bin} channel {c}"
                    );
                }
            }
        }

        // Scoring the derived histogram equals scoring the freshly-built one.
        let s_derived = scan_and_score_borders(&derived, 0, n_borders, dim, sf, l2);
        let s_full = scan_and_score_borders(&full_f, 0, n_borders, dim, sf, l2);
        assert_eq!(s_derived.len(), s_full.len(), "score length mismatch at feature {f}");
        for b in 0..s_full.len() {
            assert_eq!(
                s_derived[b].to_bits(),
                s_full[b].to_bits(),
                "subtraction-trick score diverged at feature {f} border {b}"
            );
        }
    }
}
