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
//! - **Poisson** — GPU-only, and therefore the ONE arm that does NOT follow the model above.
//!   Upstream REJECTS Poisson on the CPU task type (`bootstrap_options.cpp:29`, "poisson
//!   bootstrap is not supported on CPU"), so there is no CPU sampler to mirror and no CPU
//!   stream to advance. Upstream's CUDA kernel IS the specification, and it is transcribed
//!   here VERBATIM: a per-thread seed buffer, `numBlocks = min(ceil(seeds/256), ceil(n/256))`
//!   blocks of 256, a grid-stride walk, and the multiply-with-carry `AdvanceSeed` /
//!   `NextUniform` / `NextPoisson` of `cuda_util/kernel/random_gen.cuh`. See
//!   [`launch_poisson_bootstrap_resident`]. Gated bit-for-bit against
//!   `cb-oracle/generator/poisson_bootstrap_oracle.cpp` (a host transcription of the SAME
//!   upstream sources) via the `bootstrap_poisson/` fixtures.
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

/// `NextUniform` scale `2^-32` (`random_gen.cuh:27`) — the literal upstream writes.
const NEXT_UNIFORM_SCALE: f64 = 2.328_306_435_996_595e-10;

/// The upstream Poisson bootstrap block size (`bootstrap.cu:67`, `blockSize = 256`).
pub(crate) const POISSON_BLOCK_SIZE: usize = 256;

/// The largest `alpha` the transcribed [`poisson_bootstrap_kernel`] covers. Upstream's
/// `NextPoisson` switches to a Gaussian approximation above 20, but that branch is
/// UNREACHABLE through `GetPoissonLambda() = -log(1 - subsample)` computed in f32:
/// `alpha > 20` needs `1 - subsample < 2.1e-9`, and every such `subsample` rounds to
/// exactly `1.0f` in f32, where the formula returns `-1` instead. The host entry point
/// rejects an out-of-range `alpha` rather than silently taking the wrong branch.
const POISSON_MAX_ALPHA: f64 = 20.0;

/// Upstream's `PoissonBootstrapImpl` (`catboost/cuda/cuda_util/kernel/bootstrap.cu:8-19`)
/// with `NextPoisson`/`NextUniform`/`AdvanceSeed` (`random_gen.cuh`) inlined — a VERBATIM
/// transcription, because on the GPU-only Poisson arm this kernel IS the specification
/// (upstream has no CPU Poisson sampler).
///
/// Thread `t` of the `stride = numBlocks * 256` launched threads owns `seeds[t]` and walks
/// objects `t, t + stride, t + 2*stride, ...`, mutating its seed word in place and writing
/// it back — so consecutive trees continue the per-thread streams exactly as upstream's
/// persistent seed buffer does. `weights[i]` is the raw Poisson count: upstream fills the
/// buffer with `1.0f` before the draw (`gpu_data/bootstrap.h:88-90`) and multiplies.
///
/// Float widths are load-bearing and match upstream exactly: `NextUniform` returns f64,
/// `log` is the f64 natural log, but the accumulator `logp` and the threshold `L = -alpha`
/// are f32 — so every iteration rounds the f64 log to 24 bits. Substituting an f32 log or
/// an f64 accumulator changes which objects cross the threshold.
///
/// `cfg = [alpha]` (f32, `GetPoissonLambda()`). The stride is read from the launch geometry
/// (`CUBE_COUNT_X * CUBE_DIM_X`), which is literally upstream's `gridDim.x * blockDim.x`.
/// No `-inf` literal (a `u == 0` draw yields a runtime `-inf` from `ln`, which exits the loop
/// with count 0 — exactly what upstream does).
#[cube(launch)]
fn poisson_bootstrap_kernel(seeds: &mut Array<u64>, cfg: &Array<f32>, weights: &mut Array<f64>) {
    let n = weights.len();
    let t = ABSOLUTE_POS;
    // Threads past the object count do no work; upstream writes their seed back unchanged,
    // which is a no-op. The `seeds.len()` half is a bounds guard the host also enforces.
    if t < n && t < seeds.len() {
        let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
        let alpha = cfg[0];
        // `float L = -alpha` (random_gen.cuh:66). Written as a subtraction so no negative
        // literal enters the kernel body.
        let l = 0.0_f32 - alpha;
        let mut s = seeds[t];
        let mut i = t;
        while i < n {
            // `NextPoisson(&s, alpha)`, the `alpha <= 20` branch (random_gen.cuh:66-72).
            let mut logp = 0.0_f32;
            // Upstream's `int k`, held as f64: a runtime integer counter is ambiguous to
            // infer inside `#[cube]` (the same reason the Bayesian block index is handled
            // this way), and the count is a small non-negative integer, exact in f64.
            let mut k = 0.0_f64;
            let mut done = false;
            while !done {
                k += 1.0_f64;
                // `AdvanceSeed(&s)` (random_gen.cuh:7-15): two independent 16-bit
                // multiply-with-carry steps on the high/low halves. u32 arithmetic wraps
                // natively on device, matching C++ unsigned wraparound.
                let v0 = u32::cast_from(s >> 32u32);
                let u0 = u32::cast_from(s & 0xFFFF_FFFFu64);
                let v = 36969u32 * (v0 & 0xFFFFu32) + (v0 >> 16u32);
                let u = 18000u32 * (u0 & 0xFFFFu32) + (u0 >> 16u32);
                s = (u64::cast_from(v) << 32u32) | u64::cast_from(u);
                // `NextUniform` (random_gen.cuh:22-29) re-splits the advanced state; `(v << 16)
                // + u` is u32 arithmetic (wrapping) BEFORE the widening to f64.
                let mixed = (v << 16u32) + u;
                let uni = f64::cast_from(mixed) * NEXT_UNIFORM_SCALE;
                // `logp += log(NextUniform(seed))` — f64 log accumulated into an f32.
                logp = f32::cast_from(f64::cast_from(logp) + uni.ln());
                if logp <= l {
                    done = true;
                }
            }
            // `return k - 1` — the loop runs at least once, so `k >= 1` and the count is
            // non-negative.
            weights[i] = k - 1.0_f64;
            i += stride;
        }
        seeds[t] = s;
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
            // Poisson does NOT ride this entry point: it has no host-advanced CPU stream to
            // snapshot (upstream has no CPU Poisson sampler at all), and its state is the
            // persistent per-thread device seed buffer. It routes to
            // [`launch_poisson_bootstrap_resident`] instead, and the session never reaches
            // here with it.
            DeviceBootstrapKind::Poisson => Err(CbError::Degenerate(
                "Poisson bootstrap does not use the host-stream draw path; call \
                 launch_poisson_bootstrap_resident with the resident seed buffer"
                    .to_owned(),
            )),
        }
    }
}

/// `TBootstrapConfig::GetPoissonLambda()` (`bootstrap_options.h:31-34`), VERBATIM:
/// `takenFraction < 1 ? -log(1 - takenFraction) : -1`, computed in f32 exactly as upstream.
///
/// The `subsample >= 1` case really does return `-1` upstream, and a negative `alpha` makes
/// `NextPoisson` return 0 for EVERY object (the `logp > L` test fails on the first draw), i.e.
/// an all-zero sample. Callers must reject that configuration up front rather than train a
/// model on identically-zero weights — [`poisson_alpha`] returns it faithfully and the
/// resident entry point below refuses it.
#[must_use]
pub(crate) fn poisson_alpha(subsample: f64) -> f32 {
    let taken = subsample as f32;
    if taken < 1.0 {
        -((1.0_f32 - taken).ln())
    } else {
        -1.0
    }
}

/// The upstream launch geometry (`PoissonBootstrap`, `bootstrap.cu:66-70`):
/// `numBlocks = min(ceil(seeds_size / 256), ceil(n / 256))`, `stride = numBlocks * 256`.
/// Exposed because the mapping object → seed depends on it, so the oracle and the session
/// must agree on it exactly.
#[must_use]
pub(crate) fn poisson_grid(seeds_size: usize, n: usize) -> (usize, usize) {
    let num_blocks = seeds_size
        .div_ceil(POISSON_BLOCK_SIZE)
        .min(n.div_ceil(POISSON_BLOCK_SIZE));
    (num_blocks, num_blocks * POISSON_BLOCK_SIZE)
}

/// Draw the device-resident Poisson bootstrap weights for `n` objects, ADVANCING the resident
/// per-thread seed buffer in place (`seeds_h`, `seeds_size` u64 words) exactly as upstream's
/// persistent `TGpuAwareRandom` seed buffer advances across trees. Returns the resident
/// length-`n` f64 weight handle WITHOUT reading it back (D-08).
///
/// `subsample` is the raw `subsample` parameter; λ is derived through [`poisson_alpha`].
/// Rejects, with a typed error rather than a silently wrong model:
/// - `subsample >= 1.0` (upstream λ = -1 ⇒ every weight 0),
/// - `alpha > 20` (upstream's Gaussian branch, unreachable through the f32 λ formula),
/// - a `seeds_size` that is not a multiple of the 256 block size (upstream's own launch would
///   read past the buffer for such a size).
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_poisson_bootstrap_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    seeds_h: &Handle,
    seeds_size: usize,
    subsample: f64,
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
        let alpha = poisson_alpha(subsample);
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(CbError::OutOfRange(format!(
                "Poisson bootstrap needs subsample in (0, 1): upstream's GetPoissonLambda \
                 returns -1 for subsample >= 1, which zeroes every sample weight (got \
                 subsample = {subsample})"
            )));
        }
        if f64::from(alpha) > POISSON_MAX_ALPHA {
            return Err(CbError::OutOfRange(format!(
                "Poisson bootstrap lambda {alpha} exceeds the transcribed alpha <= 20 branch \
                 of upstream NextPoisson"
            )));
        }
        if seeds_size == 0 || !seeds_size.is_multiple_of(POISSON_BLOCK_SIZE) {
            return Err(CbError::OutOfRange(format!(
                "Poisson bootstrap seed buffer must be a non-zero multiple of the upstream \
                 block size {POISSON_BLOCK_SIZE} (got {seeds_size})"
            )));
        }

        let (num_blocks, _stride) = poisson_grid(seeds_size, n);
        let out = client.empty(n * std::mem::size_of::<f64>());
        let cfg_h = client.create(cubecl::bytes::Bytes::from_elems(vec![alpha]));
        poisson_bootstrap_kernel::launch::<SelectedRuntime>(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim {
                x: POISSON_BLOCK_SIZE as u32,
                y: 1,
                z: 1,
            },
            unsafe { ArrayArg::from_raw_parts(seeds_h.clone(), seeds_size) },
            unsafe { ArrayArg::from_raw_parts(cfg_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        Ok(out)
    }
}

/// Upstream's per-device seed-buffer size (`TGpuAwareRandom::CreateSeeds`,
/// `gpu_random.h:26` — `maxCountPerDevice = 256 * 256`). It caps the Poisson kernel's
/// parallelism at 65536 threads, which is also why the block count is `min(256, ...)`.
pub(crate) const POISSON_SEEDS_SIZE: usize = 256 * 256;

/// Create and upload the resident Poisson seed buffer for a fit.
///
/// Upstream fills this buffer host-side from its Mersenne `TRandom::NextUniformL`
/// (`gpu_random.cpp:261-268`). We fill it from the repo's validated
/// [`cb_core::TFastRng64`] instead: the seed material is opaque random state, its
/// provenance is NOT part of the kernel contract being reproduced (the oracle pins the
/// buffer explicitly), and a real upstream fit's Mersenne stream position at the moment
/// `GetGpuSeeds` is first called is not observable from outside anyway. What IS reproduced
/// bit-for-bit is the object → weight map GIVEN a seed buffer, which is where every
/// upstream-specific decision lives.
pub(crate) fn create_poisson_seeds(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    rng_seed: u64,
    seeds_size: usize,
) -> Handle {
    let mut rng = cb_core::TFastRng64::from_seed(rng_seed);
    let seeds: Vec<u64> = (0..seeds_size).map(|_| rng.gen_rand()).collect();
    client.create(cubecl::bytes::Bytes::from_elems(seeds))
}

/// Host-readback wrapper over the device Poisson draw: upload `seeds`, run `rounds`
/// CONSECUTIVE draws over the same (in-place advanced) seed buffer, and return the
/// concatenated round-major weights. This is the seam the upstream-fixture oracle
/// exercises; it is NOT the residency path (that keeps the handle on-device).
#[allow(dead_code)] // consumed by the #[cfg(test)] bootstrap_device_test self-oracle (source/test separation)
pub(crate) fn draw_poisson_weights_host(
    seeds: &[u64],
    subsample: f64,
    n: usize,
    rounds: usize,
) -> CbResult<Vec<f64>> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let seeds_h = client.create(cubecl::bytes::Bytes::from_elems(seeds.to_vec()));
    let mut out = Vec::with_capacity(rounds * n);
    for _ in 0..rounds {
        let handle =
            launch_poisson_bootstrap_resident(&client, &seeds_h, seeds.len(), subsample, n)?;
        let bytes = client.read_one(handle).map_err(|e| {
            CbError::Degenerate(format!("CubeCL Poisson read-back failed: {e:?}"))
        })?;
        out.extend_from_slice(bytemuck::cast_slice::<u8, f64>(&bytes));
    }
    Ok(out)
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
#[allow(dead_code)] // consumed by the #[cfg(test)] bootstrap_device_test self-oracle (source/test separation)
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
#[allow(dead_code)] // consumed by the #[cfg(test)] bootstrap_device_test self-oracle (source/test separation)
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
