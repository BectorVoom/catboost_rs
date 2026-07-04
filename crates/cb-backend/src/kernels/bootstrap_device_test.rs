//! Self-oracle for the device bootstrap draw (Phase 12 Plan 06, GPUT-09, Pattern F): the device
//! RNG draw ([`crate::kernels::bootstrap_device`]) must reproduce the FROZEN CPU sample computed
//! on the validated [`cb_core::TFastRng64`] (the Phase-1 oracle-tested PRNG) from a PINNED seed —
//! bit-for-bit for the Bernoulli keep-mask, ≤1e-4 for the Bayesian weights (the device uses the
//! exact `ln` vs upstream's `FastLogf` approximation, within the ε=1e-4 device bar), and
//! determinism-only for Poisson (upstream REJECTS Poisson on CPU, so there is NO CPU oracle, D-11).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernels live in the
//! production `kernels::bootstrap_device` module; ALL assertions / `.unwrap()` / indexing live
//! here. The reference is the VALIDATED `cb_core::TFastRng64` (a normal cb-backend dep — the
//! landmine is cb-TRAIN, not cb-core), so this is NON-tautological: it holds the device `#[cube]`
//! u64 transcription (gfx1100 wave32 arithmetic) against an independently-validated host RNG.
//!
//! Runs over [`crate::SelectedRuntime`]. The serial u64 RNG kernels are validated on ROCm in-env
//! (gfx1100); the cpu/wgpu backends cannot execute the u64/f64 serial draw (documented, the same
//! by-design limitation as the resident-grow oracles), so the assertions SKIP off rocm/cuda
//! (WR-01 anti-false-pass — a cpu run must not silently "pass" without exercising the device).

use cb_core::TFastRng64;

use crate::kernels::bootstrap_device::{
    device_score_stddev, draw_bootstrap_weights_host, DeviceBootstrapKind,
};

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device draw actually runs on this backend (u64/f64 serial RNG → rocm/cuda only).
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// `FastLog2f` (`library/cpp/fast_log/fast_log.h:62-76`) — the upstream base-2 log APPROXIMATION,
/// transcribed here to compute the FROZEN CPU Bayesian reference (the device uses the exact `ln`;
/// this reference proves the device stays within ε=1e-4 of the true upstream approximation). This
/// mirrors `cb_train::bootstrap::fast_log2f` (transcribed, not imported — no cb-train dep even in
/// the test).
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn fast_log2f(value: f32) -> f32 {
    let vx_i = value.to_bits();
    let mx = f32::from_bits((vx_i & 0x007F_FFFF) | 0x3f00_0000);
    let mut y = vx_i as f32;
    y *= 1.192_092_895_507_812_5e-7_f32;
    y - 124.225_514_99_f32 - 1.498_030_302_f32 * mx - 1.725_879_99_f32 / (0.352_088_706_8_f32 + mx)
}

/// `FastLogf` (`fast_log.h:84-86`): `0.69314718 * FastLog2f(value)`.
#[allow(clippy::approx_constant)]
fn fast_logf(value: f32) -> f32 {
    0.693_147_18_f32 * fast_log2f(value)
}

/// Frozen CPU Bernoulli keep-mask (`SetSampledControl`, `calc_score_cache.cpp:1196`): sequential
/// `control[i] = gen_rand_real1() < BernoulliSampleRate` off the continuous main stream, widened
/// to the 0/1 sample weight the device produces.
fn cpu_bernoulli(seed: u64, subsample: f64, n: usize) -> ([u64; 4], Vec<f64>) {
    let mut rng = TFastRng64::from_seed(seed);
    let base = rng.raw_state();
    let rate = f64::from(subsample as f32);
    let weights = (0..n)
        .map(|_| f64::from(u8::from(rng.gen_rand_real1() < rate)))
        .collect();
    (base, weights)
}

/// Frozen CPU Bayesian weights (`GenerateRandomWeights`, `tensor_search_helpers.cpp:327`): one
/// main-stream `rand_seed = rng.GenRand()`, then per 1000-element block
/// `r = from_seed(rand_seed + block).advance(10)` and `w = (-FastLogf(u + 1e-100))^temp`.
fn cpu_bayesian(seed: u64, temp: f32, n: usize) -> (u64, Vec<f64>) {
    let mut rng = TFastRng64::from_seed(seed);
    let rand_seed = rng.gen_rand();
    let block_size = 1000usize;
    let mut weights = vec![0.0_f64; n];
    let block_count = n.div_ceil(block_size);
    for block_idx in 0..block_count {
        let mut br = TFastRng64::from_seed(rand_seed.wrapping_add(block_idx as u64));
        br.advance(10);
        let begin = block_idx * block_size;
        let end = usize::min(begin + block_size, n);
        for w in weights.get_mut(begin..end).unwrap_or(&mut []) {
            let u = br.gen_rand_real1();
            let ww: f32 = -fast_logf((u as f32) + 1e-100_f32);
            *w = f64::from(ww.powf(temp));
        }
    }
    (rand_seed, weights)
}

#[test]
fn bernoulli_keep_mask_matches_frozen_cpu_sample_bit_for_bit() {
    if !device_backend_active() {
        eprintln!("[bootstrap bernoulli] skipped — device draw needs rocm/cuda (u64/f64 serial)");
        return;
    }
    for (seed, subsample, n) in [(17u64, 0.5_f64, 37usize), (42, 0.7, 128), (123, 0.3, 250)] {
        let (base, expected) = cpu_bernoulli(seed, subsample, n);
        let device = draw_bootstrap_weights_host(
            DeviceBootstrapKind::Bernoulli,
            base,
            0,
            subsample,
            0.0,
            n,
        )
        .unwrap();
        assert_eq!(device.len(), expected.len(), "length mismatch (seed {seed})");
        // Integer-exact: the keep-mask is 0/1, so bit-for-bit equality is required.
        let kept_dev: usize = device.iter().filter(|&&w| w > 0.5).count();
        let kept_cpu: usize = expected.iter().filter(|&&w| w > 0.5).count();
        println!(
            "[bootstrap bernoulli seed={seed} rate={subsample}] kept device={kept_dev} cpu={kept_cpu} / {n}"
        );
        assert_eq!(
            device, expected,
            "device Bernoulli keep-mask diverged from frozen CPU sample (seed {seed})"
        );
    }
}

#[test]
fn bayesian_weights_match_frozen_cpu_sample_within_epsilon() {
    if !device_backend_active() {
        eprintln!("[bootstrap bayesian] skipped — device draw needs rocm/cuda (u64/f64 serial)");
        return;
    }
    // Two seeds; one n spanning >1 block (1000) to exercise the per-block reseed.
    for (seed, temp, n) in [(17u64, 1.0_f32, 64usize), (7, 0.5, 1536)] {
        let (rand_seed, expected) = cpu_bayesian(seed, temp, n);
        // The device takes rand_seed directly (the ONE main-stream draw); base_state is unused.
        let device = draw_bootstrap_weights_host(
            DeviceBootstrapKind::Bayesian,
            [0; 4],
            rand_seed,
            1.0,
            f64::from(temp),
            n,
        )
        .unwrap();
        assert_eq!(device.len(), expected.len(), "length mismatch (seed {seed})");
        let max_div = device
            .iter()
            .zip(expected.iter())
            .map(|(&d, &c)| (d - c).abs())
            .fold(0.0_f64, f64::max);
        println!("[bootstrap bayesian seed={seed} temp={temp} n={n}] max_div={max_div:.3e}");
        assert!(
            max_div <= TOL,
            "device Bayesian weights diverged from frozen CPU sample: max_div={max_div:.3e} > {TOL:.0e} (seed {seed})"
        );
    }
}

#[test]
fn poisson_weights_are_deterministic_and_nonnegative_integers() {
    if !device_backend_active() {
        eprintln!("[bootstrap poisson] skipped — device draw needs rocm/cuda (u64/f64 serial)");
        return;
    }
    // No CPU oracle (upstream rejects Poisson on CPU, D-11) — validate determinism + shape only.
    let base = TFastRng64::from_seed(2024).raw_state();
    let n = 200usize;
    let a =
        draw_bootstrap_weights_host(DeviceBootstrapKind::Poisson, base, 0, 1.0, 0.0, n).unwrap();
    let b =
        draw_bootstrap_weights_host(DeviceBootstrapKind::Poisson, base, 0, 1.0, 0.0, n).unwrap();
    assert_eq!(a, b, "device Poisson draw is not deterministic for a pinned seed");
    for &w in &a {
        assert!(w >= 0.0, "Poisson weight must be non-negative");
        assert!((w - w.round()).abs() < 1e-9, "Poisson weight must be an integer count");
    }
    let mean = a.iter().sum::<f64>() / n as f64;
    println!("[bootstrap poisson] n={n} mean_count={mean:.3} (Poisson(1) ⇒ ~1.0)");
}

#[test]
fn bernoulli_leaves_match_transitively_via_identical_mask() {
    // "leaves ≤1e-4" transitively: an identical keep-mask ⇒ identical weighted leaf averages.
    // A one-leaf mean over residuals weighted by the mask is bit-identical when the masks match,
    // which the bit-for-bit test already guarantees; here we assert the leaf VALUE equality
    // explicitly on a small residual set (no GPU needed for the transitive arithmetic).
    if !device_backend_active() {
        eprintln!("[bootstrap leaves] skipped — device draw needs rocm/cuda");
        return;
    }
    let seed = 99u64;
    let n = 64usize;
    let subsample = 0.6;
    let (base, cpu_mask) =
        cpu_bernoulli(seed, subsample, n);
    let dev_mask =
        draw_bootstrap_weights_host(DeviceBootstrapKind::Bernoulli, base, 0, subsample, 0.0, n)
            .unwrap();
    let residuals: Vec<f64> = (0..n).map(|k| ((k as f64) * 0.31).sin() * 3.0 - 0.5).collect();
    let leaf = |mask: &[f64]| -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (r, m) in residuals.iter().zip(mask.iter()) {
            num += r * m;
            den += m;
        }
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    };
    let lv_dev = leaf(&dev_mask);
    let lv_cpu = leaf(&cpu_mask);
    println!("[bootstrap leaves] device={lv_dev:.8} cpu={lv_cpu:.8} div={:.3e}", (lv_dev - lv_cpu).abs());
    assert!((lv_dev - lv_cpu).abs() <= TOL, "device-sampled leaf value diverged from CPU-sampled");
}

#[test]
fn score_stddev_is_deterministic_and_matches_population_stddev() {
    // Random-strength jitter SCALE = random_strength * populationStdDev(scores), computed with the
    // ordered (deterministic) reduction. Host-arithmetic check (no GPU): repeated calls are
    // identical, and the value equals the closed-form population stddev.
    let scores = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5];
    let rs = 0.75;
    let a = device_score_stddev(&scores, rs);
    let b = device_score_stddev(&scores, rs);
    assert_eq!(a, b, "score stddev must be deterministic");
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let var = scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / n;
    let expected = rs * var.sqrt();
    assert!((a - expected).abs() <= 1e-9, "score stddev {a} != expected {expected}");
    // Degenerate: <2 scores or zero strength ⇒ no jitter.
    assert_eq!(device_score_stddev(&[1.0], rs), 0.0);
    assert_eq!(device_score_stddev(&scores, 0.0), 0.0);
}
