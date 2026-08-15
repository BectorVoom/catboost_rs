#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
//! WR-01 `WR01-S7`: the host draw replay must leave the persistent training RNG
//! in the EXACT phase a real CPU oblivious grow would have left it in.
//!
//! The oracle here is the REAL grower (`greedy_tensor_search_oblivious_perturbed`)
//! driven with `score_st_dev = 0.0` — the `random_strength == 0` regime the WR-01
//! eligibility gate admits — compared by `TFastRng64::raw_state()` and
//! `call_count()`. Comparing the RNG STATE (not a draw count) is what makes this a
//! real oracle: `std_normal`'s uniform consumption is data-dependent, so a replay
//! that merely advanced by an arithmetic count would pass a count assertion and
//! still desynchronise.

use super::{replay_grow_draws, ReplayPolicy};
use crate::tree::{greedy_tensor_search_oblivious_perturbed, FeatureMatrix, Perturbation};
use cb_compute::EScoreFunction;
use cb_core::TFastRng64;

/// A deterministic, separable synthetic pool: `n` objects, `n_features` float
/// columns with `borders_per_feature` ascending borders each.
fn synth(
    n: usize,
    n_features: usize,
    borders_per_feature: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let values: Vec<Vec<f32>> = (0..n_features)
        .map(|f| {
            (0..n)
                .map(|i| ((i * (f + 3) + f) % 97) as f32 / 97.0)
                .collect()
        })
        .collect();
    let borders: Vec<Vec<f64>> = (0..n_features)
        .map(|_| {
            (1..=borders_per_feature)
                .map(|b| b as f64 / (borders_per_feature + 1) as f64)
                .collect()
        })
        .collect();
    // A target-shaped der1 with structure, so the grower makes real (non-degenerate)
    // split choices; the replay must hold regardless of WHICH splits are chosen.
    let der1: Vec<f64> = (0..n)
        .map(|i| ((i % 13) as f64) * 0.25 - 1.5 + ((i % 7) as f64) * 0.1)
        .collect();
    let weight = vec![1.0_f64; n];
    (values, borders, der1, weight)
}

/// Run the real grower once on a fresh RNG and return its post-grow RNG state.
fn real_grow_state(
    seed: u64,
    n: usize,
    n_features: usize,
    borders_per_feature: usize,
    depth: usize,
) -> ([u64; 4], u64) {
    let (values, borders, der1, weight) = synth(n, n_features, borders_per_feature);
    let matrix = FeatureMatrix::new(&values, &borders);
    let mut rng = TFastRng64::from_seed(seed);
    let grown = greedy_tensor_search_oblivious_perturbed(
        &matrix,
        &der1,
        &weight,
        1.0,
        depth,
        n,
        Some(Perturbation {
            rng: &mut rng,
            // The WR-01 in-scope regime: sampling active, `random_strength == 0`,
            // so the perturbation is a numeric no-op but the draws still happen.
            score_st_dev: 0.0,
            score_type: cb_compute::ERandomScoreType::NormalWithModelSizeDecrease,
            // `rsm = 1.0` keeps every feature a candidate, so this test
            // observes the unchanged draw stream it was written against.
            rsm: 1.0,
        }),
        EScoreFunction::Cosine,
        None,
    )
    .expect("the synthetic pool must grow a tree");
    assert_eq!(
        grown.splits.len(),
        depth,
        "the grower must choose one split per level"
    );
    (rng.raw_state(), rng.call_count())
}

/// Run only the replay on a fresh RNG and return its post-replay state.
fn replay_state(seed: u64, n_features: usize, depth: usize) -> ([u64; 4], u64) {
    let mut rng = TFastRng64::from_seed(seed);
    replay_grow_draws(&mut rng, ReplayPolicy::SymmetricTree, depth, n_features);
    (rng.raw_state(), rng.call_count())
}

#[test]
fn replay_matches_the_real_grower_rng_state_across_shapes() {
    // (seed, n, n_features, borders_per_feature, depth)
    let shapes: &[(u64, usize, usize, usize, usize)] = &[
        (0, 64, 3, 4, 1),
        (1, 128, 4, 7, 3),
        (17, 256, 6, 15, 4),
        (42, 512, 8, 31, 6),
        (7, 200, 2, 3, 2),
    ];
    for &(seed, n, nf, bpf, depth) in shapes {
        let (real_state, real_calls) = real_grow_state(seed, n, nf, bpf, depth);
        let (rep_state, rep_calls) = replay_state(seed, nf, depth);
        assert_eq!(
            rep_state, real_state,
            "replay RNG state must equal the real grower's for \
             (seed={seed}, n={n}, n_features={nf}, borders={bpf}, depth={depth})"
        );
        // The call count is a weaker check than the state, but it localises a
        // failure: a state mismatch WITH an equal count means a wrong draw KIND.
        assert_eq!(
            rep_calls, real_calls,
            "replay draw count must equal the real grower's for \
             (seed={seed}, depth={depth}, n_features={nf})"
        );
        // Guard against a vacuous pass: sampling-active grows DO consume draws.
        assert!(
            real_calls > 0,
            "the perturbed grower must consume draws (shape seed={seed})"
        );
    }
}

#[test]
fn replay_counts_borderless_features_because_the_grower_does() {
    // A border-less ("unused-but-quantized") float feature is a LISTED candidate
    // upstream: it draws its RSM `GenRandReal1` and its `SelectBestCandidate`
    // normal even though it can never win. If the replay filtered such features
    // out, this shape would desynchronise.
    let n = 96;
    let depth = 3;
    let values: Vec<Vec<f32>> = (0..4)
        .map(|f| (0..n).map(|i| ((i * (f + 2)) % 89) as f32 / 89.0).collect())
        .collect();
    // Feature 1 and 3 have NO borders.
    let borders: Vec<Vec<f64>> = vec![
        vec![0.25, 0.5, 0.75],
        Vec::new(),
        vec![0.2, 0.4, 0.6, 0.8],
        Vec::new(),
    ];
    let der1: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) * 0.3 - 1.2).collect();
    let weight = vec![1.0_f64; n];
    let matrix = FeatureMatrix::new(&values, &borders);

    let mut real_rng = TFastRng64::from_seed(3);
    greedy_tensor_search_oblivious_perturbed(
        &matrix,
        &der1,
        &weight,
        1.0,
        depth,
        n,
        Some(Perturbation {
            rng: &mut real_rng,
            score_st_dev: 0.0,
            score_type: cb_compute::ERandomScoreType::NormalWithModelSizeDecrease,
            // `rsm = 1.0` keeps every feature a candidate, so this test
            // observes the unchanged draw stream it was written against.
            rsm: 1.0,
        }),
        EScoreFunction::Cosine,
        None,
    )
    .expect("two bordered features are enough to grow");

    let mut rep_rng = TFastRng64::from_seed(3);
    // 4 LISTED features — not the 2 bordered ones.
    replay_grow_draws(&mut rep_rng, ReplayPolicy::SymmetricTree, depth, 4);

    assert_eq!(
        rep_rng.raw_state(),
        real_rng.raw_state(),
        "border-less features must be counted as listed candidates in the replay"
    );
}

#[test]
fn replay_is_a_no_op_at_zero_depth() {
    // Zero levels ⇒ the grower's level loop never runs ⇒ nothing is consumed.
    for nf in [0usize, 5] {
        let mut rng = TFastRng64::from_seed(99);
        let before = (rng.raw_state(), rng.call_count());
        replay_grow_draws(&mut rng, ReplayPolicy::SymmetricTree, 0, nf);
        assert_eq!(
            (rng.raw_state(), rng.call_count()),
            before,
            "replay must consume nothing at depth=0, n_features={nf}"
        );
    }
}

#[test]
fn replay_at_zero_features_still_takes_the_per_level_rand_seed() {
    // A feature-less matrix is excluded by the WR-01 eligibility gate
    // (`matrix.n_features() > 0`), so this shape is unreachable in a real fit.
    // It is pinned anyway because it isolates the ONE per-level main-stream draw
    // that is NOT proportional to the feature count: `select_level_perturbed`
    // takes its `CalcScores` randSeed BEFORE iterating candidates, so the
    // faithful replay is `depth` draws — not zero. Asserting zero here would
    // encode a draw model the grower does not have.
    let depth = 4;
    let mut rng = TFastRng64::from_seed(99);
    replay_grow_draws(&mut rng, ReplayPolicy::SymmetricTree, depth, 0);
    assert_eq!(
        rng.call_count(),
        depth as u64,
        "n_features=0 must still consume one randSeed `gen_rand` per level"
    );

    let mut expected = TFastRng64::from_seed(99);
    for _ in 0..depth {
        let _ = expected.gen_rand();
    }
    assert_eq!(
        rng.raw_state(),
        expected.raw_state(),
        "the zero-feature replay must be exactly `depth` bare `gen_rand` draws"
    );
}

#[test]
fn replay_is_multi_tree_composable() {
    // The real risk this phase carries is CUMULATIVE drift: tree k's leftover
    // phase seeds tree k+1's `bootstrap()`. Replaying T trees in sequence must
    // therefore equal T sequential real grows on one continuous stream.
    let (n, nf, bpf, depth, trees) = (192usize, 5usize, 7usize, 3usize, 4usize);
    let (values, borders, der1, weight) = synth(n, nf, bpf);
    let matrix = FeatureMatrix::new(&values, &borders);

    let mut real_rng = TFastRng64::from_seed(2024);
    for tree in 0..trees {
        greedy_tensor_search_oblivious_perturbed(
            &matrix,
            &der1,
            &weight,
            1.0,
            depth,
            n,
            Some(Perturbation {
                rng: &mut real_rng,
                score_st_dev: 0.0,
                score_type: cb_compute::ERandomScoreType::NormalWithModelSizeDecrease,
                // `rsm = 1.0` keeps every feature a candidate, so this test
                // observes the unchanged draw stream it was written against.
                rsm: 1.0,
            }),
            EScoreFunction::Cosine,
            None,
        )
        .unwrap_or_else(|e| panic!("tree {tree} must grow: {e}"));
    }

    let mut rep_rng = TFastRng64::from_seed(2024);
    for _ in 0..trees {
        replay_grow_draws(&mut rep_rng, ReplayPolicy::SymmetricTree, depth, nf);
    }

    assert_eq!(
        rep_rng.raw_state(),
        real_rng.raw_state(),
        "the replay must stay phase-exact across {trees} consecutive trees"
    );
}

// ─── BLOCKER B-1 (blocks T11/T15): the replay must be GROW-POLICY aware ─────────────────

/// B-1. `replay_grow_draws` replays the OBLIVIOUS level search's draw shape: `depth`
/// levels × (`n_features` uniforms + 1 `gen_rand` + `n_features` normals). The
/// non-symmetric and Region CPU growers consume a DIFFERENT number — specifically
/// **zero**: `leaf_wise_grower` and `region_grower` take no `Perturbation` and no
/// `TFastRng64` at all (verified against their signatures), so on a Depthwise / Lossguide
/// / Region fit the CPU branch's per-tree draw count is 0.
///
/// Until T11 the gate forbade sampling × non-symmetric, so the two could never co-occur
/// and the mismatch was unreachable. T11 relaxes exactly that, which makes this live: an
/// unconditional oblivious replay would consume draws the CPU branch never consumes, and
/// the NEXT tree's `bootstrap()` would read a different RNG phase — a tree-2 divergence
/// owned by no other task in the phase. (Precedent for this exact failure class: the MVS
/// tree-2 gap, root-caused to fabricated RNG draws.)
///
/// This test pins the CPU side of the contract: a real `region_grower` /
/// `leaf_wise_grower` call must leave the RNG *untouched*.
#[test]
fn non_oblivious_cpu_growers_consume_zero_draws() {
    use crate::tree::{leaf_wise_grower, region_grower, LeafWisePolicy};

    let n = 48usize;
    let nf = 3usize;
    let (values, borders, der1, weight) = synth(n, nf, 7);
    let matrix = FeatureMatrix::new(&values, &borders);

    let rng = TFastRng64::from_seed(4242);
    let before_state = rng.raw_state();
    let before_calls = rng.call_count();

    region_grower(&matrix, &der1, &weight, 1.0, 3, 1, n, EScoreFunction::Cosine)
        .expect("region grow must succeed");
    for policy in [LeafWisePolicy::Depthwise, LeafWisePolicy::Lossguide] {
        leaf_wise_grower(policy, &matrix, &der1, &weight, 1.0, 3, 8, 1, n, EScoreFunction::Cosine)
            .expect("leaf-wise grow must succeed");
    }

    assert_eq!(
        rng.raw_state(),
        before_state,
        "the Region / leaf-wise CPU growers must not advance the training RNG at all — \
         they take no Perturbation, so the device replay for those policies must be a no-op"
    );
    assert_eq!(rng.call_count(), before_calls, "…and must draw nothing");
}

/// B-1, the device side: the replay helper must consume NOTHING for a non-oblivious
/// policy, and the oblivious draw shape must be unchanged.
///
/// Pre-fix, `replay_grow_draws` had no policy parameter and always replayed `depth`
/// oblivious levels, so this fails to compile / over-consumes.
#[test]
fn replay_is_a_no_op_for_non_oblivious_policies() {
    let depth = 4usize;
    let nf = 3usize;

    for policy in [
        ReplayPolicy::Region,
        ReplayPolicy::Depthwise,
        ReplayPolicy::Lossguide,
    ] {
        let mut rng = TFastRng64::from_seed(777);
        let before_state = rng.raw_state();
        let before_calls = rng.call_count();
        replay_grow_draws(&mut rng, policy, depth, nf);
        assert_eq!(
            rng.raw_state(),
            before_state,
            "{policy:?}: the CPU grower for this policy draws nothing, so the replay must \
             leave the stream untouched — replaying the oblivious shape here would \
             desynchronise the NEXT tree's bootstrap()"
        );
        assert_eq!(rng.call_count(), before_calls, "{policy:?}: no draws");
    }

    // …and the oblivious shape is unchanged: it must still consume.
    let mut rng = TFastRng64::from_seed(777);
    let before_calls = rng.call_count();
    replay_grow_draws(&mut rng, ReplayPolicy::SymmetricTree, depth, nf);
    assert!(
        rng.call_count() > before_calls,
        "the SymmetricTree replay must still consume the oblivious level-search draws"
    );
}
