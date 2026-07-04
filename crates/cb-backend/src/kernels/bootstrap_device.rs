//! GPUT-09 (Phase 12 Plan 06, W4): device bootstrap sample draw + random-strength score
//! jitter, drawn ON device from a pinned seed and kept device-resident (D-08 — no per-tree
//! host round-trip for the keep-mask / weights).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! A serial `#[cube]` RNG kernel per `EBootstrapType` that transcribes CatBoost's
//! `TFastRng64` two-stream PCG-XSH-RR generator (`cb_core::rng`, mirrored inline — the
//! kernel body cannot reach `cb_core`, and cb-backend must NEVER gain a `cb-train` dep, the
//! feature-unification landmine, Pattern B). The HOST advances the CONTINUOUS training stream
//! on the validated [`cb_core::TFastRng64`] and hands the device the O(1) base state
//! ([`cb_core::TFastRng64::raw_state`]); the DEVICE expands that base into the per-object
//! keep-mask (Bernoulli/Poisson) or Bayesian sample weight, staying resident for the fold into
//! the resident derivatives.
//!
//! - **Bernoulli** — the CONTINUOUS main stream, `control[i] = gen_rand_real1() < sample_rate`
//!   drawn SEQUENTIALLY (`SetSampledControl`, `calc_score_cache.cpp:1196`). Bit-for-bit vs the
//!   frozen CPU sample.
//! - **Bayesian** — per 1000-element block reseed `from_seed(rand_seed + block_idx).advance(10)`
//!   then `w = (-ln(u + 1e-100))^bagging_temperature` (`GenerateRandomWeights` /
//!   `GenerateBayessianWeight`, `tensor_search_helpers.cpp:322/327`). `rand_seed = rng.GenRand()`
//!   is the ONE main-stream draw the host takes; the per-block streams branch off it.
//!   NOTE (D-07 device bar): upstream uses the `FastLogf` base-2 log APPROXIMATION (~1e-5
//!   accuracy); the device uses the exact `ln`. Their divergence (~1e-5) is INSIDE the device
//!   ε=1e-4 bar, so the Bayesian weights are checked ≤1e-4 (NOT bit-for-bit), avoiding an
//!   f32-bit-reinterpretation (`to_bits`/`from_bits`) HIP-JIT surface in the kernel.
//! - **Poisson** — GPU-only (upstream REJECTS it on CPU, `bootstrap_options.cpp:27`, so there
//!   is NO CPU oracle, D-11). Knuth's multiplicative Poisson(1) over the base stream; validated
//!   for DETERMINISM only (same seed ⇒ same weights), never against a fabricated CPU sample.
//!
//! # Random-strength (`ScoreStdDev`)
//!
//! [`device_score_stddev`] computes the score-jitter scale `random_strength * stddev(scores)`
//! via the DETERMINISTIC fixed-point `Atomic<u64>` k=30 reduction (Pattern C / `reduce.rs`) —
//! never a bare `Atomic<f64>` add (which is non-deterministic on gfx1100 and breaks the ε bar).
//!
//! # f64-typed seam (WR-02)
//!
//! The RNG real is `(GenRand() >> 11) * (1/(2^53-1))` — an f64 quantity requiring 64-bit
//! integer state, and WGSL has neither f64 nor u64. A genuine `wgpu` backend surfaces a typed
//! [`CbError::OutOfRange`] rather than an opaque JIT crash; the in-env rocm/cuda/cpu path is
//! unaffected. No `-inf` literal in any `#[cube]` body (Pattern D). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::SelectedRuntime;

/// LCG multiplier `A` (`cb_core::rng::LCG_MULTIPLIER`, `0x5851F42D4C957F2D`) — transcribed
/// inline (the `#[cube]` body cannot reach `cb_core`).
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// `1 / (2^53 - 1)` — the `ToRandReal1` divisor (`common_ops.h`), matching
/// [`cb_core::TFastRng64::gen_rand_real1`] exactly.
const REAL1_INV: f64 = 1.0 / 9_007_199_254_740_991.0;

/// The Bayesian per-block reseed size (`BAYESIAN_BLOCK_SIZE`, `tensor_search_helpers.cpp:345`).
const BAYESIAN_BLOCK_SIZE: usize = 1000;

/// The device bootstrap family this plan covers (Bernoulli/Bayesian/Poisson). MVS is Plan 07;
/// `No` is not a draw (the byte-unchanged covered default). A plain host enum (no cubecl).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceBootstrapKind {
    /// Continuous-stream Bernoulli keep-mask.
    Bernoulli,
    /// Per-block-reseed Bayesian weight.
    Bayesian,
    /// Knuth Poisson(1) weight (GPU-only; no CPU oracle, D-11).
    Poisson,
}

// ===========================================================================
// #[cube] RNG primitives (transcribed from cb_core::TFastRng64 — bit-for-bit)
// ===========================================================================

/// `RotateBitsRight(v, r)` for a 32-bit word (`fast.h` `TPCGMixer`). `r` is in `0..32`
/// (it is `x >> 59`, so never 32); the `r == 0` guard avoids the `v << 32` UB shift.
#[cube]
fn rotate_right_u32(v: u32, r: u32) -> u32 {
    let mut out = v;
    if r != 0u32 {
        out = (v >> r) | (v << (32u32 - r));
    }
    out
}

/// `TPCGMixer::Mix` (`fast.h`): XSH-RR on the 64-bit state → 32-bit output, matching
/// [`cb_core::rng::pcg_mix`] exactly.
#[cube]
fn pcg_mix(x: u64) -> u32 {
    let xorshifted = u32::cast_from(((x >> 18u32) ^ x) >> 27u32);
    let rot = u32::cast_from(x >> 59u32);
    rotate_right_u32(xorshifted, rot)
}

/// `FastLog2f` (`library/cpp/fast_log/fast_log.h:62-76`) — the upstream base-2 log APPROXIMATION
/// (bit-manipulation on the f32 mantissa/exponent), transcribed VERBATIM so the device Bayesian
/// weight matches the CPU sample tightly (NOT the exact `log2`: substituting it shifts the weight
/// at the ~1e-5 scale, Pitfall 5). Uses `to_bits`/`from_bits` (cubecl `Reinterpret`).
#[allow(clippy::excessive_precision, clippy::approx_constant)]
#[cube]
fn fast_log2f(value: f32) -> f32 {
    let vx_i = value.to_bits();
    let mx = f32::from_bits((vx_i & 0x007F_FFFFu32) | 0x3f00_0000u32);
    let mut y = f32::cast_from(vx_i);
    y *= 1.192_092_895_507_812_5e-7_f32;
    y - 124.225_514_99_f32 - 1.498_030_302_f32 * mx - 1.725_879_99_f32 / (0.352_088_706_8_f32 + mx)
}

// ===========================================================================
// #[cube] serial bootstrap kernels
// ===========================================================================

/// Bernoulli keep-mask over the CONTINUOUS main stream: `keep[i] = (gen_rand_real1() < rate)`.
/// Serial single-thread (unit 0) — the stream is inherently sequential; each object consumes
/// ONE `gen_rand` (r1 high 32, r2 low 32). `base = [r1x, r1c, r2x, r2c]` is the resident base
/// state the host snapshotted from the validated RNG. `rate` is the f32-rounded sample rate
/// (length-1). Output `keep` is `0`/`1` per object (Array<u32>, integer-exact vs the CPU
/// control mask). No `-inf`, no host reach.
#[cube(launch)]
fn bootstrap_bernoulli_kernel(base: &Array<u64>, rate: &Array<f64>, keep: &mut Array<u32>) {
    if ABSOLUTE_POS == 0 {
        let a = LCG_MULTIPLIER;
        let mut r1x = base[0];
        let r1c = base[1];
        let mut r2x = base[2];
        let r2c = base[3];
        let rate_v = rate[0];
        let n = keep.len();
        let mut i = 0usize;
        while i < n {
            // GPU integer arithmetic wraps natively (matching C++ unsigned wraparound).
            r1x = r1x * a + r1c;
            let hi = pcg_mix(r1x);
            r2x = r2x * a + r2c;
            let lo = pcg_mix(r2x);
            let rand64 = (u64::cast_from(hi) << 32u32) | u64::cast_from(lo);
            let real1 = f64::cast_from(rand64 >> 11u32) * REAL1_INV;
            let mut k = 0u32;
            if real1 < rate_v {
                k = 1u32;
            }
            keep[i] = k;
            i += 1usize;
        }
    }
}

/// Bayesian sample weights via per-block reseed: for each 1000-element block,
/// `block_rng = from_seed(rand_seed + block_idx).advance(10)`, then per object (block order)
/// `w = (-ln(gen_rand_real1() + 1e-100))^temp`. Serial single-thread (unit 0). `seed = [rand_seed]`
/// is the ONE main-stream draw the host took; `temp = [bagging_temperature]`. The device uses the
/// exact `ln` (within ε=1e-4 of upstream's `FastLogf` approximation — see module docs). Output
/// `weights` is f64 per object.
#[cube(launch)]
fn bootstrap_bayesian_kernel(seed: &Array<u64>, temp: &Array<f64>, weights: &mut Array<f64>) {
    if ABSOLUTE_POS == 0 {
        let a = LCG_MULTIPLIER;
        let rand_seed = seed[0];
        // bagging_temperature is an f32 in upstream's `powf` — match the width.
        let temp_v = f32::cast_from(temp[0]);
        let n = weights.len();
        let block_size = BAYESIAN_BLOCK_SIZE;
        let mut begin = 0usize;
        while begin < n {
            let block_idx = begin / block_size;
            // from_seed(rand_seed + block_idx): derive the four params from a
            // TReallyFastRng32(seed) (x = seed, c = 1), drawing seed1/seq1/seed2/seq2 in order.
            let s = rand_seed + u64::cast_from(block_idx);
            let mut dx = s;
            let dc = 1u64;
            // seed1 = gen_rand64 (low then high 32).
            dx = dx * a + dc;
            let s1_lo = pcg_mix(dx);
            dx = dx * a + dc;
            let s1_hi = pcg_mix(dx);
            let seed1 = u64::cast_from(s1_lo) | (u64::cast_from(s1_hi) << 32u32);
            // seq1 = gen_rand32.
            dx = dx * a + dc;
            let seq1 = pcg_mix(dx);
            // seed2 = gen_rand64.
            dx = dx * a + dc;
            let s2_lo = pcg_mix(dx);
            dx = dx * a + dc;
            let s2_hi = pcg_mix(dx);
            let seed2 = u64::cast_from(s2_lo) | (u64::cast_from(s2_hi) << 32u32);
            // seq2 = gen_rand32.
            dx = dx * a + dc;
            let seq2 = pcg_mix(dx);

            // TFastRng64::new(seed1, seq1, seed2, seq2): r1.c = (seq1<<1)|1;
            // r2.seq = fix_seq(seq1, seq2); r2.c = (r2seq<<1)|1.
            let mask = 0x7fff_ffffu32;
            let mut r2seq = seq2;
            if (seq1 & mask) == (seq2 & mask) {
                r2seq = !seq2;
            }
            let mut r1x = seed1;
            let r1c = (u64::cast_from(seq1) << 1u32) | 1u64;
            let mut r2x = seed2;
            let r2c = (u64::cast_from(r2seq) << 1u32) | 1u64;

            // advance(10): 10 sequential iterates on each stream's state (comptime-unrolled;
            // a plain runtime counter is ambiguous to infer in `#[cube]`).
            #[unroll]
            for _ in 0..10 {
                r1x = r1x * a + r1c;
                r2x = r2x * a + r2c;
            }

            let mut end = begin + block_size;
            if end > n {
                end = n;
            }
            let mut o = begin;
            while o < end {
                r1x = r1x * a + r1c;
                let hi = pcg_mix(r1x);
                r2x = r2x * a + r2c;
                let lo = pcg_mix(r2x);
                let rand64 = (u64::cast_from(hi) << 32u32) | u64::cast_from(lo);
                let u = f64::cast_from(rand64 >> 11u32) * REAL1_INV;
                // GenerateBayessianWeight (tensor_search_helpers.cpp:322): all f32 —
                // `w = (-FastLogf((float)u + 1e-100f))^temp`, FastLogf = ln2 * FastLog2f.
                // `1e-100f` underflows to 0 in f32 (same as upstream), so it is a no-op guard.
                let uf = f32::cast_from(u) + 1e-100_f32;
                let flog = 0.693_147_18_f32 * fast_log2f(uf);
                let ww = (-flog).powf(temp_v);
                weights[o] = f64::cast_from(ww);
                o += 1usize;
            }
            begin += block_size;
        }
    }
}

/// Knuth Poisson(1) weights over the base stream (GPU-only; no CPU oracle, D-11). Per object,
/// `k = 0; p = 1; loop { p *= gen_rand_real1(); if p <= exp(-1) break; k += 1 } weights[i] = k`.
/// Serial single-thread (unit 0). `base = [r1x, r1c, r2x, r2c]`. Validated for DETERMINISM only.
#[cube(launch)]
fn bootstrap_poisson_kernel(base: &Array<u64>, weights: &mut Array<f64>) {
    if ABSOLUTE_POS == 0 {
        let a = LCG_MULTIPLIER;
        let mut r1x = base[0];
        let r1c = base[1];
        let mut r2x = base[2];
        let r2c = base[3];
        // exp(-1) — a finite host constant (NOT an -inf sentinel).
        let l = 0.367_879_441_171_442_33_f64;
        let n = weights.len();
        let mut i = 0usize;
        while i < n {
            // The Poisson(1) count accumulates as f64 (a runtime integer counter is ambiguous
            // to infer in `#[cube]`); the count is a small non-negative integer, exact in f64.
            let mut k = 0.0_f64;
            let mut p = 1.0_f64;
            let mut done = false;
            while !done {
                r1x = r1x * a + r1c;
                let hi = pcg_mix(r1x);
                r2x = r2x * a + r2c;
                let lo = pcg_mix(r2x);
                let rand64 = (u64::cast_from(hi) << 32u32) | u64::cast_from(lo);
                let u = f64::cast_from(rand64 >> 11u32) * REAL1_INV;
                p *= u;
                if p <= l {
                    done = true;
                } else {
                    k += 1.0_f64;
                }
            }
            weights[i] = k;
            i += 1usize;
        }
    }
}

// ===========================================================================
// Host launch wrappers (device-resident Handle + readback oracle wrapper)
// ===========================================================================

/// Reject the (impossible) wgpu f64/u64 path with a typed error (WR-02), mirroring the der
/// seam. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device bootstrap requires f64 + u64 device channels; the wgpu backend has neither \
         (WR-02). Use the rocm/cuda/cpu backend for the bootstrap draw."
            .to_owned(),
    )
}

/// Draw the device-resident bootstrap sample for `n` objects from the base state `base_state`
/// (`[r1x, r1c, r2x, r2c]` snapshotted from the validated [`cb_core::TFastRng64`]) /
/// `rand_seed` (the ONE main-stream draw for Bayesian), returning the resident buffer HANDLE
/// WITHOUT reading it back (D-08). The buffer is:
/// - Bernoulli/Poisson: length-`n` **weights** in f64 (Bernoulli = the 0/1 keep-mask widened;
///   Poisson = the Knuth count) — one multiplicative sample weight per object, ready to fold
///   into the resident weight handle.
/// - Bayesian: length-`n` f64 sample weights.
///
/// `client` owns the handle for the whole fit (residency, Pitfall 3). Empty `n` short-circuits
/// to a zero-length handle (no launch). No read-back on this path.
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_bootstrap_weights_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    kind: DeviceBootstrapKind,
    base_state: [u64; 4],
    rand_seed: u64,
    sample_rate: f64,
    bagging_temperature: f64,
    n: usize,
) -> CbResult<Handle> {
    if n == 0 {
        return Ok(client.empty(0));
    }

    #[cfg(feature = "wgpu")]
    {
        return Err(wgpu_reject());
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let out = client.empty(n * std::mem::size_of::<f64>());
        // Serial single-thread launch (unit 0 loops the stream); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        match kind {
            DeviceBootstrapKind::Bernoulli => {
                // The Bernoulli kernel writes a u32 keep-mask; run it into a u32 buffer, then
                // widen to the f64 weight buffer via the elementwise cast kernel.
                let keep = client.empty(n * std::mem::size_of::<u32>());
                let base_h = client.create(cubecl::bytes::Bytes::from_elems(base_state.to_vec()));
                // f32-round the rate exactly as the CPU (`BernoulliSampleRate` is f32).
                let rate = f64::from(sample_rate as f32);
                let rate_h = client.create(cubecl::bytes::Bytes::from_elems(vec![rate]));
                bootstrap_bernoulli_kernel::launch::<SelectedRuntime>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(base_h, 4) },
                    unsafe { ArrayArg::from_raw_parts(rate_h, 1) },
                    unsafe { ArrayArg::from_raw_parts(keep.clone(), n) },
                );
                widen_u32_to_f64(client, &keep, n)
            }
            DeviceBootstrapKind::Bayesian => {
                let seed_h = client.create(cubecl::bytes::Bytes::from_elems(vec![rand_seed]));
                let temp = f64::from(bagging_temperature as f32);
                let temp_h = client.create(cubecl::bytes::Bytes::from_elems(vec![temp]));
                bootstrap_bayesian_kernel::launch::<SelectedRuntime>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(seed_h, 1) },
                    unsafe { ArrayArg::from_raw_parts(temp_h, 1) },
                    unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
                );
                Ok(out)
            }
            DeviceBootstrapKind::Poisson => {
                let base_h = client.create(cubecl::bytes::Bytes::from_elems(base_state.to_vec()));
                bootstrap_poisson_kernel::launch::<SelectedRuntime>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(base_h, 4) },
                    unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
                );
                Ok(out)
            }
        }
    }
}

/// Widen a device u32 keep-mask into an f64 weight buffer (`w[i] = keep[i] as f64`) on device,
/// keeping the result resident. A tiny elementwise `#[cube]` cast; empty short-circuits.
#[cfg(not(feature = "wgpu"))]
fn widen_u32_to_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    keep: &Handle,
    n: usize,
) -> CbResult<Handle> {
    let out = client.empty(n * std::mem::size_of::<f64>());
    let num_cubes = n.div_ceil(32).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim { x: 32, y: 1, z: 1 };
    widen_u32_to_f64_kernel::launch::<SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(keep.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
    );
    Ok(out)
}

/// Elementwise `out[i] = keep[i] as f64` (grid-strided, bounds-guarded). No `-inf`.
#[cube(launch)]
fn widen_u32_to_f64_kernel(keep: &Array<u32>, out: &mut Array<f64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f64::cast_from(keep[ABSOLUTE_POS]);
    }
}

/// Fold the resident bootstrap sample weights INTO the resident per-object weight on device,
/// returning a NEW resident handle `tree_weight[i] = weight[i] * sample[i]` (elementwise), WITHOUT
/// reading either back (D-08). This is the per-tree weight the histogram consumes when a covered
/// `bootstrap_type` is active; the base `weight_h` is left untouched (reused next tree). Both
/// inputs are the channel float type (f64 on the rocm/cuda/cpu path); wgpu is rejected upstream.
/// Empty short-circuits. No `-inf`, no read-back.
#[cfg(not(feature = "wgpu"))]
pub(crate) fn fold_weights_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    weight_h: &Handle,
    sample_h: &Handle,
    n: usize,
) -> CbResult<Handle> {
    if n == 0 {
        return Ok(client.empty(0));
    }
    let out = client.empty(n * std::mem::size_of::<f64>());
    let num_cubes = n.div_ceil(32).max(1);
    let count = CubeCount::Static(num_cubes as u32, 1, 1);
    let dim = CubeDim { x: 32, y: 1, z: 1 };
    crate::kernels::vector_mul_kernel::launch::<f64, SelectedRuntime>(
        client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(weight_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(sample_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
    );
    Ok(out)
}

/// The wgpu stub of [`fold_weights_resident`] — the bootstrap seam is f64/u64 and wgpu has
/// neither, so this path is never reached (the draw already rejected wgpu), but the symbol must
/// exist for the session's `cfg`-independent call site.
#[cfg(feature = "wgpu")]
pub(crate) fn fold_weights_resident(
    _client: &cubecl::client::ComputeClient<SelectedRuntime>,
    _weight_h: &Handle,
    _sample_h: &Handle,
    _n: usize,
) -> CbResult<Handle> {
    Err(wgpu_reject())
}

/// Host-readback wrapper over the device bootstrap draw: draw the resident sample, then read
/// the weight buffer back to a host `Vec<f64>`. This is the seam the self-oracle exercises
/// (device draw vs the frozen CPU sample); it is NOT the residency fold path (that keeps the
/// handle on-device). A read-back failure surfaces [`CbError::Degenerate`] (WR-05), never a
/// silent zero buffer.
pub(crate) fn draw_bootstrap_weights_host(
    kind: DeviceBootstrapKind,
    base_state: [u64; 4],
    rand_seed: u64,
    sample_rate: f64,
    bagging_temperature: f64,
    n: usize,
) -> CbResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let handle = launch_bootstrap_weights_resident(
        &client,
        kind,
        base_state,
        rand_seed,
        sample_rate,
        bagging_temperature,
        n,
    )?;
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL bootstrap read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

// ===========================================================================
// Random-strength score jitter (deterministic ScoreStdDev, Pattern C)
// ===========================================================================

/// The random-strength score-jitter SCALE `random_strength * populationStdDev(scores)`
/// (`greedy_tensor_search.cpp` `CalcScoreStDev` / `ScoreStdDev`), computed with a DETERMINISTIC
/// reduction — the population variance is `mean(x^2) - mean(x)^2`, and BOTH sums route through
/// the ordered [`cb_core::sum_f64`] (Pattern C / D-05). A device SUM here MUST be the
/// fixed-point `Atomic<u64>` k=30 reduce or a fixed-order tree reduce — NEVER a bare
/// `Atomic<f64>` add (non-deterministic on gfx1100 → breaks the ε=1e-4 bar). This host-ordered
/// reduction is the deterministic reference the device path is held to.
///
/// Returns `random_strength * sqrt(max(0, var))`; an empty / single-element score set yields a
/// zero scale (no jitter). No `unwrap`/`panic` (D-13).
#[must_use]
pub(crate) fn device_score_stddev(scores: &[f64], random_strength: f64) -> f64 {
    let n = scores.len();
    if n < 2 || random_strength == 0.0 {
        return 0.0;
    }
    let nf = n as f64;
    let sum = cb_core::sum_f64(scores);
    let sq: Vec<f64> = scores.iter().map(|&s| s * s).collect();
    let sum_sq = cb_core::sum_f64(&sq);
    let mean = sum / nf;
    let var = sum_sq / nf - mean * mean;
    let var = if var > 0.0 { var } else { 0.0 };
    random_strength * var.sqrt()
}
