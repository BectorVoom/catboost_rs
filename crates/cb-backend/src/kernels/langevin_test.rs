//! Self-oracle for the device Langevin / SGLB draw (Phase 13 Plan 09, GPUT-20, Pattern F): the
//! device AddLangevinNoise kernel ([`crate::kernels::langevin`]) must reproduce the FROZEN CPU
//! `coefficient · std_normal` sequence computed on the validated [`cb_core::TFastRng64`] (the Phase-1
//! oracle-tested PRNG) + [`cb_core::std_normal`] (the Phase-1 oracle-tested Marsaglia-polar draw) from
//! a PINNED seed — the noised der ≤1e-4 (the device transcribes the SAME per-element reseed
//! `from_seed(rand_seed + i).advance(10)` + Marsaglia-polar rejection loop bit-for-bit), the
//! per-element draw count matching (a divergent count shifts that element's value past the ε bar).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernel lives in the
//! production `kernels::langevin` module; ALL assertions / `.unwrap()` / indexing live here. The CPU
//! reference calls the independently-validated `cb_core::std_normal` on the independently-validated
//! `cb_core::TFastRng64` — NO `cb-train` dep even in the test (the feature-unification landmine), so
//! this is NON-tautological.
//!
//! Runs over [`crate::SelectedRuntime`]. The serial u64/f64 Langevin kernel is validated on ROCm
//! in-env (gfx1100); the cpu/wgpu backends cannot execute the u64/f64 serial draw (documented, the
//! same by-design limitation as the MVS / bootstrap oracles), so the numeric assertions SKIP off
//! rocm/cuda (WR-01 anti-false-pass — a cpu run must not silently "pass" without exercising the
//! device). The whole file is `not(feature = "wgpu")` (the wgpu backend has no f64/u64 channel).

#![cfg(not(feature = "wgpu"))]

use cb_compute::Loss;
use cb_core::{std_normal, TFastRng64};

use crate::kernels::langevin::{draw_langevin_host, langevin_covered_loss};

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device draw actually runs on this backend (u64/f64 serial Langevin → rocm/cuda only).
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

// ---------------------------------------------------------------------------
// Frozen CPU reference (calls the Phase-1-validated cb_core primitives — NO dep)
// ---------------------------------------------------------------------------

/// The FROZEN CPU Langevin sample: for each element `i`, reseed a fresh [`TFastRng64`] from
/// `rand_seed + i`, `advance(10)`, draw ONE [`std_normal`], and add `coefficient · normal` to the
/// input der. Returns the noised der (the exact sequence the device must reproduce). The per-element
/// reseed mirrors the device kernel's `from_seed(rand_seed + i).advance(10)`.
fn cpu_langevin_noise(derivatives: &[f64], rand_seed: u64, coefficient: f64) -> Vec<f64> {
    derivatives
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let mut rng = TFastRng64::from_seed(rand_seed.wrapping_add(i as u64));
            rng.advance(10);
            d + coefficient * std_normal(&mut rng)
        })
        .collect()
}

/// The per-element Marsaglia-polar draw COUNT (number of `(x, y)` uniform PAIRS consumed) the CPU
/// rejection loop takes for element `i`. Reproduces `cb_core::std_normal`'s accept condition
/// (`0 < x*x + y*y <= 1`) over the SAME reseeded stream, counting iterations — the load-bearing
/// draw-order quantity (Pitfall 4). A device consuming a different count for element `i` would produce
/// a different element-`i` value (caught by the ε assertion below), because each element reseeds
/// independently.
fn cpu_langevin_draw_count(rand_seed: u64, i: usize) -> usize {
    let mut rng = TFastRng64::from_seed(rand_seed.wrapping_add(i as u64));
    rng.advance(10);
    let mut pairs = 0usize;
    loop {
        let x = rng.gen_rand_real1() * 2.0 - 1.0;
        let y = rng.gen_rand_real1() * 2.0 - 1.0;
        let r = x * x + y * y;
        pairs += 1;
        if !(r > 1.0 || r <= 0.0) {
            return pairs;
        }
    }
}

/// A deterministic, varied derivative vector (mixes signs / magnitudes). NOT sorted.
fn make_derivatives(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let x = k as f64;
            (x * 0.9).sin() * 2.3 + (x * 0.13).cos() * 0.7
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

/// Test 1: with a pinned seed, the device noised der reproduces the frozen CPU
/// `coefficient · std_normal` sequence within ε=1e-4.
#[test]
fn langevin_noised_der_matches_frozen_cpu_sequence_within_epsilon() {
    if !device_backend_active() {
        eprintln!("[langevin] skipped — device Langevin needs rocm/cuda (u64/f64 serial)");
        return;
    }
    for (rand_seed, coefficient, n) in
        [(17u64, 0.05_f64, 48usize), (42, 0.20, 64), (2024, 0.011, 200)]
    {
        let der = make_derivatives(n);
        let expected = cpu_langevin_noise(&der, rand_seed, coefficient);
        let device = draw_langevin_host(&der, rand_seed, coefficient, n).unwrap();
        assert_eq!(device.len(), expected.len(), "length mismatch (seed {rand_seed})");

        let max_div = max_divergence(&device, &expected);
        println!("[langevin seed={rand_seed} coef={coefficient} n={n}] max_div={max_div:.3e}");
        assert!(
            max_div <= TOL,
            "device Langevin noised der diverged from frozen CPU sequence: \
             max_div={max_div:.3e} > {TOL:.0e} (seed {rand_seed})"
        );
        // Every element must be finite (the noise is added to a finite der).
        for &v in &device {
            assert!(v.is_finite(), "device Langevin produced a non-finite noised der (seed {rand_seed})");
        }
    }
}

/// Test 2: the per-element draw COUNT matches the CPU rejection-loop consumption; a divergent count
/// is detected. The count itself is the load-bearing draw-order quantity (Pitfall 4); because each
/// element reseeds independently, a device count divergence for element `i` shifts element `i`'s
/// value past the ε bar — so the value match below IS the per-element count guard, and the counts are
/// additionally asserted to be well-formed (every element draws at least one `(x, y)` pair).
#[test]
fn langevin_per_element_draw_count_is_wellformed_and_reproduced() {
    let rand_seed = 31u64;
    let coefficient = 0.1_f64;
    let n = 80usize;

    // The CPU per-element draw counts are all >= 1 (the Marsaglia-polar loop always draws at least
    // one pair) and bounded in practice (expected ≈1.27 pairs; a pathological count would be a red
    // flag). This is a pure-CPU sanity fence on the draw-order model — runs on every backend.
    let mut total_pairs = 0usize;
    for i in 0..n {
        let pairs = cpu_langevin_draw_count(rand_seed, i);
        assert!(pairs >= 1, "element {i} consumed no Marsaglia-polar pair (draw-order model broken)");
        assert!(pairs <= 1000, "element {i} draw count {pairs} implausibly large (RNG model broken)");
        total_pairs += pairs;
    }
    let mean_pairs = total_pairs as f64 / n as f64;
    println!("[langevin draw-count] n={n} mean_pairs={mean_pairs:.3} (expected ≈1.27)");

    if !device_backend_active() {
        eprintln!("[langevin draw-count] device value-match skipped — needs rocm/cuda");
        return;
    }
    // The device value match is the actual per-element count-divergence detector: a device that
    // consumed a different number of draws for any element would produce a different element value.
    let der = make_derivatives(n);
    let expected = cpu_langevin_noise(&der, rand_seed, coefficient);
    let device = draw_langevin_host(&der, rand_seed, coefficient, n).unwrap();
    let max_div = max_divergence(&device, &expected);
    println!("[langevin draw-count] value max_div={max_div:.3e}");
    assert!(
        max_div <= TOL,
        "device Langevin draw count/order diverged from CPU (value max_div={max_div:.3e} > {TOL:.0e})"
    );
}

/// Test 3: empty-`n` constructs the no-op launcher WITHOUT a 0-length `read_one` (HIP fault
/// avoidance); and a `*Pairwise` + Langevin config is NOT covered (`langevin_covered_loss` false for
/// `is_pairwise_scoring`) — the "PairLogit+Langevin → CPU fallback" guard (A4) — while a covered
/// pointwise der loss IS covered. Runs on every backend (no device launch for the empty / predicate
/// paths).
#[test]
fn langevin_empty_noop_and_pairwise_declines() {
    // Empty-n: the host wrapper short-circuits to an empty Vec with NO device launch / read_one.
    let out = draw_langevin_host(&[], 7, 0.1, 0).unwrap();
    assert!(out.is_empty(), "empty-n Langevin must return an empty noised der (no 0-len read_one)");

    // Covered pointwise der family → Langevin covered.
    assert!(langevin_covered_loss(&Loss::Rmse), "RMSE must be Langevin-covered");
    assert!(langevin_covered_loss(&Loss::Logloss), "Logloss must be Langevin-covered");
    assert!(langevin_covered_loss(&Loss::CrossEntropy), "CrossEntropy must be Langevin-covered");

    // Pairwise oracle → Langevin NOT supported (A4 upstream CB_ENSURE): PairLogit+Langevin → CPU.
    assert!(
        !langevin_covered_loss(&Loss::PairLogitPairwise),
        "PairLogitPairwise+Langevin must decline (Langevin unsupported on the pairwise oracle, A4)"
    );
    assert!(
        !langevin_covered_loss(&Loss::YetiRankPairwise { permutations: 1, decay: 0.99 }),
        "YetiRankPairwise+Langevin must decline (pairwise oracle, A4)"
    );
    // A non-covered pointwise loss (Quantile) is also not Langevin-covered (covered-family gate).
    assert!(
        !langevin_covered_loss(&Loss::Quantile { alpha: 0.5, delta: 1e-6 }),
        "Quantile is outside the covered Langevin der family"
    );
}
