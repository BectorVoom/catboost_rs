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
