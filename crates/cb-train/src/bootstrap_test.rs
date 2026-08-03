//! Draw-sequence and dispatch unit tests for [`crate::bootstrap`] (TRAIN-04).
//!
//! The parity contract is the EXACT draw ORDER (Pitfall 4 / threat T-03-03-01):
//! these tests re-derive the expected sample weights / control mask directly
//! from the bitstream-validated [`cb_core::TFastRng64`] primitives
//! (`from_seed` / `advance` / `gen_rand` / `gen_rand_real1`) and assert the
//! [`crate::bootstrap::bootstrap`] output matches them across >= 2 reseed blocks
//! for a fixed seed. Kept in a dedicated `*_test.rs` file (source/test
//! separation, CLAUDE.md / AGENTS.md).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
// The fast-log reference transcribes upstream's verbatim C constants; same
// rationale as the production helper (see bootstrap.rs `fast_log2f`).
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use cb_core::TFastRng64;

use crate::bootstrap::{
    bootstrap, BootstrapResult, EBootstrapType, BAYESIAN_BLOCK_SIZE,
};

/// Re-derive the Bayesian per-block weight stream the way upstream does, using
/// ONLY the validated RNG primitives, so the reference is independent of the
/// implementation under test.
/// Independent transcription of upstream `FastLog2f`/`FastLogf`
/// (`library/cpp/fast_log/fast_log.h`) for the reference; the implementation
/// under test must reproduce the SAME bit-manipulation approximation.
fn ref_fast_logf(value: f32) -> f32 {
    let vx_i = value.to_bits();
    let mx = f32::from_bits((vx_i & 0x007F_FFFF) | 0x3f00_0000);
    let mut y = vx_i as f32;
    y *= 1.192_092_895_507_812_5e-7_f32;
    let log2 =
        y - 124.225_514_99_f32 - 1.498_030_302_f32 * mx - 1.725_879_99_f32 / (0.352_088_706_8_f32 + mx);
    0.693_147_18_f32 * log2
}

fn expected_bayesian(n: usize, bagging_temperature: f32, random_seed: u64) -> Vec<f64> {
    let mut main = TFastRng64::from_seed(random_seed);
    let rand_seed = main.gen_rand();
    let mut weights = vec![1.0_f64; n];
    let block_count = n.div_ceil(BAYESIAN_BLOCK_SIZE);
    for block_idx in 0..block_count {
        let mut r = TFastRng64::from_seed(rand_seed.wrapping_add(block_idx as u64));
        r.advance(10);
        let begin = block_idx * BAYESIAN_BLOCK_SIZE;
        let end = usize::min(begin + BAYESIAN_BLOCK_SIZE, n);
        for w in weights[begin..end].iter_mut() {
            let u = r.gen_rand_real1();
            let bw: f32 = -ref_fast_logf((u as f32) + 1e-100_f32);
            *w = f64::from(bw.powf(bagging_temperature));
        }
    }
    weights
}

/// Bayesian draws reproduce the per-1000-block reseed across >= 2 blocks: the
/// 1500-object stream (blocks `[0,1000)` and `[1000,1500)`) must match the
/// independently re-derived reference exactly, AND the two blocks must differ
/// (proving the reseed actually happened, not a single continuous stream).
#[test]
fn bayesian_draw_sequence_matches_reference_across_two_blocks() {
    let n = 1500;
    let temp = 1.0_f32;
    let seed = 0_u64;
    let ders = vec![0.0_f64; n]; // Bayesian ignores derivatives.

    let mut rng = TFastRng64::from_seed(seed);
    let BootstrapResult {
        sample_weights,
        control,
    } = bootstrap(EBootstrapType::Bayesian, &ders, 1.0, temp, None, &mut rng).unwrap();

    let expected = expected_bayesian(n, temp, seed);
    assert_eq!(sample_weights.len(), n);
    assert_eq!(sample_weights, expected, "Bayesian weights must match the per-block reference");
    // Bayesian leaves control all-true (BernoulliSampleRate == 1, no draw).
    assert!(control.iter().all(|&c| c));

    // The per-block reseed makes block 0 and block 1 distinct streams: the first
    // weight of block 1 (object 1000) is NOT the 1000th draw of a continuous
    // single-block stream — assert the two blocks' first weights differ.
    assert_ne!(sample_weights[0], sample_weights[1000]);
}

/// `bagging_temperature == 0` short-circuits Bayesian to all-`1.0` with no draws.
#[test]
fn bayesian_zero_temperature_is_identity() {
    let n = 1500;
    let ders = vec![0.0_f64; n];
    let mut rng = TFastRng64::from_seed(0);
    let res = bootstrap(EBootstrapType::Bayesian, &ders, 1.0, 0.0, None, &mut rng).unwrap();
    assert!(res.sample_weights.iter().all(|&w| w == 1.0));
    assert!(res.control.iter().all(|&c| c));
}

/// Re-derive the Bernoulli control mask: SEQUENTIAL `GenRandReal1() < subsample`
/// over the SAME continuous main stream (no per-block reseed).
fn expected_bernoulli_control(n: usize, subsample: f64, random_seed: u64) -> Vec<bool> {
    let mut rng = TFastRng64::from_seed(random_seed);
    let rate = f64::from(subsample as f32);
    (0..n).map(|_| rng.gen_rand_real1() < rate).collect()
}

/// Bernoulli control matches the sequential single-stream reference across the
/// whole 1500-object range (spanning both 1000-blocks); sample weights stay
/// `1.0` (the subsample lives in the control, not the weights).
#[test]
fn bernoulli_control_sequential_matches_reference() {
    let n = 1500;
    let subsample = 0.8;
    let seed = 0_u64;
    let ders = vec![0.0_f64; n];

    let mut rng = TFastRng64::from_seed(seed);
    let res = bootstrap(EBootstrapType::Bernoulli, &ders, subsample, 0.0, None, &mut rng).unwrap();

    let expected = expected_bernoulli_control(n, subsample, seed);
    assert_eq!(res.control, expected);
    assert!(res.sample_weights.iter().all(|&w| w == 1.0));
    // ~80% selected; assert it is neither all-true nor all-false and spans blocks.
    let selected = res.control.iter().filter(|&&c| c).count();
    assert!(selected > n * 7 / 10 && selected < n);
}

/// `subsample == 1.0` makes Bernoulli select every object with no draw.
#[test]
fn bernoulli_full_subsample_selects_all_without_draw() {
    let n = 100;
    let ders = vec![0.0_f64; n];
    let mut rng = TFastRng64::from_seed(0);
    // A clone of the RNG must be UNADVANCED after a full-subsample Bernoulli call.
    let mut probe = TFastRng64::from_seed(0);
    let res = bootstrap(EBootstrapType::Bernoulli, &ders, 1.0, 0.0, None, &mut rng).unwrap();
    assert!(res.control.iter().all(|&c| c));
    // rng must not have advanced: its next draw equals the fresh probe's draw.
    assert_eq!(rng.gen_rand(), probe.gen_rand());
}

/// `No` is the identity: all weights `1.0`, all selected, ZERO RNG draws (the
/// RNG is left completely unadvanced).
#[test]
fn no_bootstrap_is_identity_and_draws_nothing() {
    let n = 1500;
    let ders = vec![1.0_f64; n];
    let mut rng = TFastRng64::from_seed(0);
    let mut probe = TFastRng64::from_seed(0);
    let res = bootstrap(EBootstrapType::No, &ders, 1.0, 0.0, None, &mut rng).unwrap();
    assert!(res.sample_weights.iter().all(|&w| w == 1.0));
    assert!(res.control.iter().all(|&c| c));
    assert_eq!(rng.gen_rand(), probe.gen_rand(), "No must not advance the RNG");
}

/// MVS with `subsample == 1.0` is all-`1.0` weights with no draw; with a real
/// subsample it produces an importance-weighted, partially-zeroed mask whose
/// nonzero weights are `>= 1` (each is `1/probability`, `probability <= 1`).
#[test]
fn mvs_full_subsample_is_identity_and_real_subsample_is_importance_weighted() {
    let n = 2000; // single MVS block (< 8192).
    // Varied gradient magnitudes so the threshold is non-degenerate.
    let ders: Vec<f64> = (0..n).map(|i| (i as f64 % 13.0) - 6.0).collect();

    let mut rng_full = TFastRng64::from_seed(0);
    let mut probe = TFastRng64::from_seed(0);
    let full = bootstrap(EBootstrapType::Mvs, &ders, 1.0, 0.0, None, &mut rng_full).unwrap();
    assert!(full.sample_weights.iter().all(|&w| w == 1.0));
    assert_eq!(rng_full.gen_rand(), probe.gen_rand(), "MVS subsample=1 draws nothing");

    let mut rng = TFastRng64::from_seed(0);
    let res = bootstrap(EBootstrapType::Mvs, &ders, 0.5, 0.0, None, &mut rng).unwrap();
    assert_eq!(res.sample_weights.len(), n);
    // Some objects dropped (weight 0), some kept; control mirrors weight>eps.
    let kept = res.sample_weights.iter().filter(|&&w| w > 0.0).count();
    assert!(kept > 0 && kept < n);
    for (&w, &c) in res.sample_weights.iter().zip(res.control.iter()) {
        assert_eq!(c, w > f64::from(f32::EPSILON));
        if w > 0.0 {
            // 1/probability with probability in (0, 1] -> weight >= 1.
            assert!(w >= 1.0 - 1e-9);
        }
    }
}

/// Poisson is rejected on the CPU path (mirrors upstream
/// `bootstrap_options.cpp`): the dispatch returns an error, never panics.
#[test]
fn poisson_is_rejected_on_cpu() {
    let ders = vec![0.0_f64; 10];
    let mut rng = TFastRng64::from_seed(0);
    let res = bootstrap(EBootstrapType::Poisson, &ders, 0.8, 0.0, None, &mut rng);
    assert!(res.is_err(), "Poisson must be rejected on CPU");
}

/// `MVS-S1`: an MVS `Bootstrap()` call consumes EXACTLY ONE main-stream draw.
///
/// Upstream takes a single `randSeed = rand->GenRand()` (`mvs.cpp:174`) and nothing
/// else: with `performRandomChoice = false`, `TCalcScoreFold::Sample` goes down the
/// `SetControlNoZeroWeighted` branch (`calc_score_cache.cpp:742-748`), which never
/// touches `rand`, and `CalcWeightedData` (`tensor_search_helpers.cpp:442-485`) is
/// draw-free.
///
/// This is the contract no oracle can express. An oracle only sees the *consequence*
/// — a wrong per-tree RNG phase flipping a split argmax somewhere down the boosting
/// run — and only for the seed/bias combinations where the wrong subset happens to
/// change an argmax at all. That is exactly how a 2-draw-per-tree defect survived
/// with the committed `bootstrap/mvs` oracle green: it is one of the configurations
/// that passed *despite* it. Pinning the draw count directly makes the defect
/// impossible to reintroduce silently.
///
/// All four legs assert the SAME single behaviour — the per-call main-stream draw
/// count — so a failure has one principal cause.
#[test]
fn mvs_bootstrap_consumes_exactly_one_main_stream_draw() {
    let n = 1500; // one MVS block (< 8192); the fixture's object count.
    // Varied gradient magnitudes so the threshold is non-degenerate.
    let ders: Vec<f64> = (0..n).map(|i| (i as f64 % 13.0) - 6.0).collect();
    let seed = 0_u64;

    // (1) The count itself — the assertion whose failure names the defect.
    let mut rng = TFastRng64::from_seed(seed);
    let _ = bootstrap(EBootstrapType::Mvs, &ders, 0.8, 0.0, None, &mut rng).unwrap();
    assert_eq!(
        rng.call_count(),
        1,
        "an MVS bootstrap() call must consume exactly ONE main-stream draw (the \
         rand_seed); the per-block sample streams branch off it via \
         TFastRng64::from_seed(rand_seed + block_idx) and never touch the main stream"
    );

    // (2) The state, not just the count: a wrong draw KIND would keep the count
    //     right and still desynchronise every later tree.
    let mut probe = TFastRng64::from_seed(seed);
    let _ = probe.gen_rand();
    assert_eq!(
        rng.raw_state(),
        probe.raw_state(),
        "the one draw must be a bare gen_rand() on the main stream"
    );

    // (3) Zero-draw regression leg: `subsample >= 1.0` short-circuits to the
    //     identity sample and must consume nothing at all.
    let mut rng_full = TFastRng64::from_seed(seed);
    let _ = bootstrap(EBootstrapType::Mvs, &ders, 1.0, 0.0, None, &mut rng_full).unwrap();
    assert_eq!(
        rng_full.call_count(),
        0,
        "MVS at subsample >= 1.0 must draw nothing (the identity short-circuit)"
    );

    // (4) Accumulation leg: the real hazard is CUMULATIVE per-tree drift, so pin
    //     the count across consecutive calls on ONE continuous stream.
    let mut rng_multi = TFastRng64::from_seed(seed);
    for _ in 0..3 {
        let _ = bootstrap(EBootstrapType::Mvs, &ders, 0.8, 0.0, None, &mut rng_multi).unwrap();
    }
    assert_eq!(
        rng_multi.call_count(),
        3,
        "three consecutive MVS bootstrap() calls must consume exactly three draws; \
         any surplus is a per-tree phase drift that desynchronises every later tree"
    );
}

/// `MVS-S4` / `MVS-S5`: the two f32 transcription fidelities in the MVS sampler.
///
/// `TMvsSampler::SampleRate` is a `float` (`mvs.h:47`), so upstream's per-block
/// threshold-search target `SampleRate * blockSize` is an f32 product. Computing it in
/// f64 shifts the target by up to ~2.4e-4 at realistic block sizes, which can move the
/// threshold and therefore which objects survive sampling.
///
/// The `(0.8, 8192)` case is the no-regression leg: scaling by a power of two is
/// already exact, so it must hold both before and after. `(0.8, 1500)` and
/// `(0.8, 3616)` are the discriminating cases — an f64 product gives `1200.0000178813934`
/// and `2892.800043106079` respectively.
#[test]
fn mvs_f32_transcription_targets_and_weight_narrowing() {
    use crate::bootstrap::mvs_block_sample_size;

    // MVS-S4: the block threshold target reproduces upstream's `float * ui32`.
    assert_eq!(mvs_block_sample_size(0.8, 1500), 1200.0);
    assert_eq!(mvs_block_sample_size(0.8, 8192), 6553.600_097_656_25);
    assert_eq!(mvs_block_sample_size(0.8, 3616), 2892.800_048_828_125);

    // MVS-S5: every stored MVS weight round-trips through f32 losslessly, mirroring
    // upstream's `TVector<float> SampleWeights` (`fold.h:217`).
    let n = 2000;
    let ders: Vec<f64> = (0..n).map(|i| (i as f64 % 13.0) - 6.0).collect();
    let mut rng = TFastRng64::from_seed(0);
    let res = bootstrap(EBootstrapType::Mvs, &ders, 0.5, 0.0, None, &mut rng).unwrap();
    let kept = res.sample_weights.iter().filter(|&&w| w > 0.0).count();
    assert!(kept > 0 && kept < n, "the sample must be non-degenerate");
    for (&w, &c) in res.sample_weights.iter().zip(res.control.iter()) {
        assert_eq!(
            w,
            f64::from(w as f32),
            "MVS sample weights must be f32-representable (fold.h:217)"
        );
        // The narrowing must not disturb the control mask: a kept weight is 1/p >= 1,
        // far above f32::EPSILON, and a dropped one is bit-exact 0.0.
        assert_eq!(c, w > f64::from(f32::EPSILON));
    }
}
