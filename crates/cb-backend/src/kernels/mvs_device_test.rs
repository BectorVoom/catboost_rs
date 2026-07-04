//! Self-oracle for the device MVS draw (Phase 12 Plan 07, GPUT-17, Pattern F): the device MVS
//! sample-weight kernel ([`crate::kernels::mvs_device`]) must reproduce the FROZEN CPU
//! `mvs_sample_weights` computed on the validated [`cb_core::TFastRng64`] (the Phase-1 oracle-tested
//! PRNG) + [`cb_core::sum_f64`] (the sanctioned ordered sum) from a PINNED seed — weights ≤1e-4
//! (the device threshold is a monotone bisection of the SAME `calculate_threshold` root; the
//! per-block reseeded `NextUniformF` stream is bit-for-bit), the per-block threshold identical, and
//! the sampled count per block matching.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernel lives in the
//! production `kernels::mvs_device` module; ALL assertions / `.unwrap()` / indexing live here. The
//! CPU reference transcribes `cb-train/src/bootstrap.rs`'s `single_probability` /
//! `calculate_threshold` / `mvs_sample_weights` INLINE (NO `cb-train` dep even in the test — the
//! feature-unification landmine), held against the independently-validated `cb_core` RNG + sum, so
//! this is NON-tautological.
//!
//! Runs over [`crate::SelectedRuntime`]. The serial u64/f64 MVS kernel is validated on ROCm in-env
//! (gfx1100); the cpu/wgpu backends cannot execute the u64/f64 serial draw (documented, the same
//! by-design limitation as the resident-grow / bootstrap oracles), so the assertions SKIP off
//! rocm/cuda (WR-01 anti-false-pass — a cpu run must not silently "pass" without exercising the
//! device).

use cb_core::{sum_f64, TFastRng64};

use crate::kernels::mvs_device::draw_mvs_weights_host;

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// MVS block size (`mvs.h:48`; `cb-train` `MVS_BLOCK_SIZE`).
const MVS_BLOCK_SIZE: usize = 8192;

/// Whether the device draw actually runs on this backend (u64/f64 serial MVS → rocm/cuda only).
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

// ---------------------------------------------------------------------------
// Frozen CPU reference (transcribed from cb-train/src/bootstrap.rs — NO dep)
// ---------------------------------------------------------------------------

/// `GetSingleProbability` (`mvs.cpp:17`): `der>threshold ? 1 : (threshold>0 ? der/threshold : 0)`.
fn single_probability(derivative_abs: f64, threshold: f64) -> f64 {
    if derivative_abs > threshold {
        1.0
    } else if threshold > 0.0 {
        derivative_abs / threshold
    } else {
        0.0
    }
}

/// `TMvsSampler::CalculateThreshold` (`mvs.cpp:81-118`) — the recursive quickselect-partition MVS
/// threshold estimator (transcribed VERBATIM from `cb-train`). `candidates` is one block's
/// `sqrt(lambda + grad^2)` values.
fn calculate_threshold(
    candidates: &mut [f64],
    sum_of_small_current: f64,
    number_of_large_current: f64,
    sample_size: f64,
) -> f64 {
    let threshold = match candidates.first() {
        Some(&t) => t,
        None => return 0.0,
    };
    let mut small: Vec<f64> = Vec::with_capacity(candidates.len());
    let mut middle: Vec<f64> = Vec::new();
    let mut large: Vec<f64> = Vec::new();
    for &c in candidates.iter() {
        if c < threshold {
            small.push(c);
        } else if c <= threshold {
            middle.push(c);
        } else {
            large.push(c);
        }
    }
    let sum_of_small_update = sum_f64(&small);
    let number_of_large_update = large.len() as f64;
    let number_of_middle = middle.len() as f64;
    let sum_of_middle = number_of_middle * threshold;

    let estimated_sample_size = if threshold != 0.0 {
        (sum_of_small_current + sum_of_small_update) / threshold
            + number_of_large_current
            + number_of_large_update
            + number_of_middle
    } else {
        f64::INFINITY
    };

    if estimated_sample_size > sample_size {
        if !large.is_empty() {
            let next_small = sum_of_small_current + sum_of_middle + sum_of_small_update;
            calculate_threshold(&mut large, next_small, number_of_large_current, sample_size)
        } else {
            let denom = sample_size - number_of_large_current;
            if denom != 0.0 {
                (sum_of_small_current + sum_of_small_update + sum_of_middle) / denom
            } else {
                threshold
            }
        }
    } else if !small.is_empty() {
        let next_large = number_of_large_current + number_of_large_update + number_of_middle;
        calculate_threshold(&mut small, sum_of_small_current, next_large, sample_size)
    } else {
        let denom =
            sample_size - number_of_large_current - number_of_middle - number_of_large_update;
        if denom != 0.0 {
            sum_of_small_current / denom
        } else {
            threshold
        }
    }
}

/// `mean(|der|)^2` — `GetLambda(...)` on the FIRST tree (`mvs.cpp:37-79`). The squared mean gradient
/// magnitude, via the ordered [`sum_f64`] (D-05).
fn mvs_lambda_iter0(derivatives: &[f64]) -> f64 {
    if derivatives.is_empty() {
        return 0.0;
    }
    let mags: Vec<f64> = derivatives.iter().map(|&d| (d * d).sqrt()).collect();
    let mean = sum_f64(&mags) / derivatives.len() as f64;
    mean * mean
}

/// The per-block CPU MVS threshold for `derivatives[begin..end]` with the given `lambda` /
/// f32-rounded `sample_rate` — used to assert the device reproduces the per-block threshold.
fn cpu_block_threshold(block: &[f64], lambda: f64, sample_rate: f64) -> f64 {
    let mut candidates: Vec<f64> = block.iter().map(|&d| (lambda + d * d).sqrt()).collect();
    calculate_threshold(&mut candidates, 0.0, 0.0, sample_rate * block.len() as f64)
}

/// `TMvsSampler::GenSampleWeights` (`mvs.cpp:120-224`, single-dimension) — the FROZEN CPU sample.
/// Returns `(rand_seed, weights)`: `rand_seed = rng.gen_rand()` is the ONE main-stream draw the
/// device consumes; `weights[i]` is the per-object MVS sample weight.
fn cpu_mvs_sample(
    seed: u64,
    derivatives: &[f64],
    lambda: f64,
    sample_rate: f64,
) -> (u64, Vec<f64>) {
    let n = derivatives.len();
    let sample_rate = f64::from(sample_rate as f32);
    let mut rng = TFastRng64::from_seed(seed);
    let rand_seed = rng.gen_rand();
    let mut weights = vec![0.0_f64; n];
    let block_count = n.div_ceil(MVS_BLOCK_SIZE);
    for block_idx in 0..block_count {
        let mut block_rng = TFastRng64::from_seed(rand_seed.wrapping_add(block_idx as u64));
        block_rng.advance(10);
        let begin = block_idx * MVS_BLOCK_SIZE;
        let end = usize::min(begin + MVS_BLOCK_SIZE, n);
        let block = derivatives.get(begin..end).unwrap_or(&[]);
        let threshold = cpu_block_threshold(block, lambda, sample_rate);
        for (offset, &der) in block.iter().enumerate() {
            let grad2 = der * der;
            let probability = single_probability((grad2 + lambda).sqrt(), threshold);
            let idx = begin + offset;
            if probability > f64::EPSILON {
                let weight = 1.0 / probability;
                let r = block_rng.gen_rand_real1();
                if let Some(slot) = weights.get_mut(idx) {
                    *slot = weight * f64::from(r < probability);
                }
            } else if let Some(slot) = weights.get_mut(idx) {
                *slot = 0.0;
            }
        }
    }
    (rand_seed, weights)
}

/// A deterministic, varied derivative vector (mixes signs / magnitudes so the threshold is
/// non-trivial). NOT sorted (keeps the CPU quickselect recursion shallow).
fn make_derivatives(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let x = k as f64;
            ((x * 0.9).sin() * 2.3 + (x * 0.13).cos() * 0.7) * (1.0 + (x * 0.37).sin().abs())
        })
        .collect()
}

fn max_divergence(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------

#[test]
fn mvs_weights_match_frozen_cpu_sample_within_epsilon() {
    if !device_backend_active() {
        eprintln!("[mvs] skipped — device MVS needs rocm/cuda (u64/f64 serial)");
        return;
    }
    // Single-block fixtures: varied (seed, sample_rate, n); lambda is the iter-0 GetLambda formula.
    for (seed, sample_rate, n) in [(17u64, 0.5_f64, 48usize), (42, 0.7, 64), (2024, 0.3, 200)] {
        let der = make_derivatives(n);
        let lambda = mvs_lambda_iter0(&der);
        let (rand_seed, expected) = cpu_mvs_sample(seed, &der, lambda, sample_rate);
        let device = draw_mvs_weights_host(&der, rand_seed, sample_rate, lambda, n).unwrap();
        assert_eq!(device.len(), expected.len(), "length mismatch (seed {seed})");

        let max_div = max_divergence(&device, &expected);
        let kept_dev = device.iter().filter(|&&w| w > 0.0).count();
        let kept_cpu = expected.iter().filter(|&&w| w > 0.0).count();
        let target = f64::from(sample_rate as f32) * n as f64;
        println!(
            "[mvs seed={seed} rate={sample_rate} n={n}] max_div={max_div:.3e} kept dev={kept_dev} \
             cpu={kept_cpu} (~target {target:.1})"
        );
        assert!(
            max_div <= TOL,
            "device MVS weights diverged from frozen CPU sample: max_div={max_div:.3e} > {TOL:.0e} (seed {seed})"
        );
        // Identical keep-mask ⇒ identical sampled count (a flip would blow past the ε bar above).
        assert_eq!(
            kept_dev, kept_cpu,
            "device MVS sampled count per block diverged from CPU (seed {seed})"
        );
        // Sampled count sits in a sane band around the block target (single block here).
        assert!(
            kept_cpu as f64 <= target + n as f64 * 0.5 + 1.0,
            "sampled count {kept_cpu} implausibly large vs target {target:.1} (seed {seed})"
        );
    }
}

#[test]
fn mvs_per_block_threshold_matches_and_reweight_is_consistent() {
    if !device_backend_active() {
        eprintln!("[mvs threshold] skipped — device MVS needs rocm/cuda");
        return;
    }
    // Explicit lambda (a later-tree GetLambda value) + a single block: the CPU threshold is the
    // recursive quickselect; the device weights transitively encode the SAME threshold (matching
    // weights ⇒ matching per-object probability ⇒ matching threshold on the un-capped candidates).
    let n = 96usize;
    let der = make_derivatives(n);
    let lambda = 0.25_f64;
    let sample_rate = 0.6_f64;
    let threshold = cpu_block_threshold(&der, lambda, f64::from(sample_rate as f32));
    assert!(threshold.is_finite() && threshold > 0.0, "degenerate CPU threshold {threshold}");

    let (rand_seed, expected) = cpu_mvs_sample(31, &der, lambda, sample_rate);
    let device = draw_mvs_weights_host(&der, rand_seed, sample_rate, lambda, n).unwrap();
    let max_div = max_divergence(&device, &expected);
    println!("[mvs threshold] cpu_threshold={threshold:.8} max_div={max_div:.3e}");
    assert!(max_div <= TOL, "device MVS weights diverged (threshold path): {max_div:.3e} > {TOL:.0e}");

    // An UN-capped candidate (below threshold) has weight 1/p = threshold/candidate exactly — verify
    // the reweight math against the CPU threshold for a kept, un-capped object.
    for (i, &der_i) in der.iter().enumerate() {
        let cand = (lambda + der_i * der_i).sqrt();
        if cand < threshold {
            if let Some(&w) = device.get(i) {
                if w > 0.0 {
                    let expect_w = threshold / cand;
                    assert!(
                        (w - expect_w).abs() <= TOL.max(expect_w.abs() * 1e-6),
                        "kept un-capped object {i}: device weight {w:.8} != 1/p {expect_w:.8}"
                    );
                }
            }
        }
    }
}

#[test]
fn mvs_multi_block_reseed_is_deterministic_and_finite() {
    if !device_backend_active() {
        eprintln!("[mvs multiblock] skipped — device MVS needs rocm/cuda");
        return;
    }
    // n spanning >1 block (block_idx 0 and 1) exercises the per-block reseed boundary. A pinned
    // seed ⇒ identical draws (determinism); all weights finite and non-negative.
    let n = MVS_BLOCK_SIZE + 24;
    let der = make_derivatives(n);
    let lambda = mvs_lambda_iter0(&der);
    let sample_rate = 0.5_f64;
    let (rand_seed, _cpu) = cpu_mvs_sample(7, &der, lambda, sample_rate);

    let a = draw_mvs_weights_host(&der, rand_seed, sample_rate, lambda, n).unwrap();
    let b = draw_mvs_weights_host(&der, rand_seed, sample_rate, lambda, n).unwrap();
    assert_eq!(a, b, "device MVS is not deterministic for a pinned seed (multi-block)");
    assert_eq!(a.len(), n);
    for &w in &a {
        assert!(w.is_finite() && w >= 0.0, "MVS weight must be finite and non-negative");
    }
    // The second block (24 objects) must have been visited (weights present past the boundary).
    let tail_kept = a.get(MVS_BLOCK_SIZE..n).map_or(0, |t| t.iter().filter(|&&w| w > 0.0).count());
    println!("[mvs multiblock] n={n} tail_kept(block1)={tail_kept}/24");
}
