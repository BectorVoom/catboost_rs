//! GPUT-17 (Phase 12 Plan 07, W5): Minimal-Variance Sampling (MVS) on device — CatBoost's DEFAULT
//! GPU sampler. MVS is inherently a device reduction over the RESIDENT derivatives: per-block
//! (`BlockSize = 8192`) it finds the optimal threshold over the `sqrt(lambda + der^2)` candidates,
//! then reweights each object by the inverse keep-probability. The keep-mask / weights stay
//! device-resident (D-08 — no per-tree host round-trip); the fold into the resident derivatives
//! happens on device (`fold_weights_resident`, shared with Plan 06).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! A serial `#[cube]` kernel that transcribes `cb-train/src/bootstrap.rs`'s MVS semantics
//! (`single_probability`, `calculate_threshold`, `mvs_sample_weights`) INLINE — the kernel body
//! cannot reach `cb_core`, and cb-backend must NEVER gain a `cb-train` dep (the feature-unification
//! landmine, Pattern B). Per block:
//!
//! 1. **candidate** `c_i = sqrt(lambda + der_i^2)` over the block's resident derivatives.
//! 2. **threshold** — the block threshold `μ` solving `Σ_i min(1, c_i/μ) = sample_rate·blockSize`.
//!    This is the SAME root the CPU `calculate_threshold` (recursive quickselect) returns; the
//!    device finds it by a DETERMINISTIC monotone bisection (`F(μ)` is continuous, strictly
//!    decreasing → unique root). "Match the threshold SEMANTICS, not the algorithm"
//!    (`CATBOOST_CUDA_KERNELS_DESIGN.md` §6.1). Every block SUM is a SERIAL single-thread
//!    accumulation (unit 0) — inherently order-fixed / deterministic, so NO device atomic (and in
//!    particular NO bare `Atomic<f64>`) is used; the in-env `Atomic<u64>`-advertisement regression
//!    that blocks the resident histogram does NOT affect this kernel (it runs in-env like Plan
//!    06's RNG / Plan 08's CTR serial kernels).
//! 3. **reweight** — `p = single_probability(c_i, μ) = c_i>μ ? 1 : c_i/μ`; `weight = (1/p)` when
//!    `NextUniformF < p` else `0`. The per-block RNG is the CPU's exact stream:
//!    `block_rng = from_seed(rand_seed + block_idx).advance(10)` (`rand_seed` is the ONE
//!    main-stream `GenRand()` the host takes), and the draw is CONDITIONAL on `p > f64::EPSILON`
//!    (matching the CPU — a `p ≤ ε` object consumes NO draw, keeping the stream phase aligned).
//!
//! # f64-typed seam (WR-02)
//!
//! The RNG real is `(GenRand() >> 11) · (1/(2^53-1))` — an f64 quantity over 64-bit state; WGSL has
//! neither f64 nor u64, so a genuine `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather
//! than an opaque JIT crash. The in-env rocm/cuda/cpu path is unaffected. No `-inf` literal in any
//! `#[cube]` body (Pattern D — a finite `f64` bound is used for the bisection bracket). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::SelectedRuntime;

/// LCG multiplier `A` (`cb_core::rng::LCG_MULTIPLIER`, `0x5851F42D4C957F2D`) — transcribed inline
/// (the `#[cube]` body cannot reach `cb_core`), matching [`crate::kernels::bootstrap_device`].
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// `1 / (2^53 - 1)` — the `ToRandReal1` divisor (`common_ops.h`), matching
/// [`cb_core::TFastRng64::gen_rand_real1`] exactly.
const REAL1_INV: f64 = 1.0 / 9_007_199_254_740_991.0;

/// MVS block size (`mvs.h:48` `const ui32 BlockSize = 8192`; `cb-train` `MVS_BLOCK_SIZE`).
const MVS_BLOCK_SIZE: usize = 8192;

/// `f64::EPSILON` (`2^-52`) — the CPU `mvs_sample_weights` keep-probability floor
/// (`if probability > f64::EPSILON`). A finite host constant transcribed inline (NOT an `-inf`
/// sentinel); a `p ≤ ε` object gets weight `0` and consumes NO RNG draw.
const F64_EPSILON: f64 = 2.220_446_049_250_313e-16;

/// Bisection iteration budget for the threshold root-find. `100` halvings shrink the bracket by
/// `2^-100` (relative), so the device threshold matches the CPU `calculate_threshold` root to full
/// `f64` precision — the reweight probabilities agree far inside the ε=1e-4 device bar, and no
/// object's pinned `NextUniformF` draw straddles the (identical) probability, so the keep-mask does
/// not flip. Perf is an MVP concern (Plan 06 precedent); the covered fixtures are single-block.
const MVS_BISECTION_ITERS: u32 = 100;

// ===========================================================================
// #[cube] RNG primitives (transcribed from cb_core::TFastRng64 — bit-for-bit,
// mirroring crate::kernels::bootstrap_device so the per-block reseed matches)
// ===========================================================================

/// `RotateBitsRight(v, r)` for a 32-bit word (`fast.h` `TPCGMixer`). `r = x >> 59` is in `0..32`
/// (never 32); the `r == 0` guard avoids the `v << 32` UB shift.
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

// ===========================================================================
// #[cube] serial MVS kernel
// ===========================================================================

/// Serial (unit 0) MVS sample-weight kernel over the resident derivatives. `params = [lambda,
/// sample_rate]` (`sample_rate` already f32-rounded by the host); `seed = [rand_seed]` (the ONE
/// main-stream `GenRand()` the host took). Output `weights[i]` is the per-object multiplicative
/// MVS sample weight (`0` == dropped), device-resident, ready to fold into the resident weight.
///
/// Per block: bracketed bisection finds the threshold `μ` (deterministic serial sums), then the
/// per-block reseeded RNG reweights each object with a CONDITIONAL `NextUniformF` draw (`p > ε`).
/// No `-inf` literal; no host reach.
#[cube(launch)]
fn mvs_sample_kernel(
    der: &Array<f64>,
    params: &Array<f64>,
    seed: &Array<u64>,
    weights: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let a = LCG_MULTIPLIER;
        let lambda = params[0];
        let rate = params[1];
        let rand_seed = seed[0];
        let n = weights.len();
        let block_size = MVS_BLOCK_SIZE;
        let eps = F64_EPSILON;

        let mut begin = 0usize;
        while begin < n {
            let block_idx = begin / block_size;
            let mut end = begin + block_size;
            if end > n {
                end = n;
            }
            let bs = end - begin;

            // --- Pass 1: total candidate mass + max candidate (bisection bracket). ---
            let mut total_sum = 0.0_f64;
            let mut max_cand = 0.0_f64;
            let mut j = begin;
            while j < end {
                let d = der[j];
                let cand = (lambda + d * d).sqrt();
                total_sum += cand;
                if cand > max_cand {
                    max_cand = cand;
                }
                j += 1usize;
            }

            // sample_size = sample_rate · blockSize (rate is f32-rounded on the host).
            let sample_size = rate * f64::cast_from(u64::cast_from(bs));

            // --- Threshold: bisect F(μ) = Σ min(1, c_i/μ) = sample_size (F strictly decreasing).
            // Bracket: F(0+) = #{c_i>0} ≥ sample_size (rate<1) and F(hi) < sample_size for
            // hi = total_sum/sample_size + max_cand. Degenerate (all-zero candidates or a
            // non-positive target) leaves threshold 0 → every p ≤ ε → weight 0 (matches the CPU
            // all-zero-derivative path, which likewise draws nothing).
            let mut threshold = 0.0_f64;
            if max_cand > 0.0 && sample_size > 0.0 {
                let mut lo = 0.0_f64;
                let mut hi = total_sum / sample_size + max_cand;
                let mut it = 0u32;
                while it < MVS_BISECTION_ITERS {
                    let mid = 0.5_f64 * (lo + hi);
                    let mut f = 0.0_f64;
                    let mut k = begin;
                    while k < end {
                        let d = der[k];
                        let cand = (lambda + d * d).sqrt();
                        let mut p = 1.0_f64;
                        if cand < mid {
                            p = cand / mid;
                        }
                        f += p;
                        k += 1usize;
                    }
                    // F too large ⇒ μ too small ⇒ raise the floor; else lower the ceiling.
                    if f > sample_size {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    it += 1u32;
                }
                threshold = 0.5_f64 * (lo + hi);
            }

            // --- Per-block reseed: block_rng = from_seed(rand_seed + block_idx).advance(10). ---
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

            // TFastRng64::new: r1.c = (seq1<<1)|1; r2.seq = fix_seq(seq1, seq2); r2.c = (r2seq<<1)|1.
            let mask = 0x7fff_ffffu32;
            let mut r2seq = seq2;
            if (seq1 & mask) == (seq2 & mask) {
                r2seq = !seq2;
            }
            let mut r1x = seed1;
            let r1c = (u64::cast_from(seq1) << 1u32) | 1u64;
            let mut r2x = seed2;
            let r2c = (u64::cast_from(r2seq) << 1u32) | 1u64;

            // advance(10): 10 sequential iterates on each stream (comptime-unrolled).
            #[unroll]
            for _ in 0..10 {
                r1x = r1x * a + r1c;
                r2x = r2x * a + r2c;
            }

            // --- Reweight: single_probability + conditional NextUniformF draw. ---
            let mut o = begin;
            while o < end {
                let d = der[o];
                let cand = (lambda + d * d).sqrt();
                // single_probability(cand, threshold): cand>μ ? 1 : (μ>0 ? cand/μ : 0).
                let mut p = 0.0_f64;
                if cand > threshold {
                    p = 1.0_f64;
                } else if threshold > 0.0 {
                    p = cand / threshold;
                }
                let mut w = 0.0_f64;
                if p > eps {
                    // Draw ONLY when p > ε (matching the CPU stream phase).
                    r1x = r1x * a + r1c;
                    let hi_w = pcg_mix(r1x);
                    r2x = r2x * a + r2c;
                    let lo_w = pcg_mix(r2x);
                    let rand64 = (u64::cast_from(hi_w) << 32u32) | u64::cast_from(lo_w);
                    let r = f64::cast_from(rand64 >> 11u32) * REAL1_INV;
                    let mut keep = 0.0_f64;
                    if r < p {
                        keep = 1.0_f64;
                    }
                    w = keep / p;
                }
                weights[o] = w;
                o += 1usize;
            }

            begin += block_size;
        }
    }
}

// ===========================================================================
// Host launch wrappers (device-resident Handle + readback oracle wrapper)
// ===========================================================================

/// Reject the (impossible) wgpu f64/u64 path with a typed error (WR-02), mirroring
/// [`crate::kernels::bootstrap_device`]. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device MVS requires f64 + u64 device channels; the wgpu backend has neither (WR-02). \
         Use the rocm/cuda/cpu backend for the MVS draw."
            .to_owned(),
    )
}

/// Draw the device-resident MVS sample weights for the resident derivatives `der_h` (`n` objects),
/// returning the resident buffer HANDLE WITHOUT reading it back (D-08). `rand_seed` is the ONE
/// main-stream `GenRand()` the host took; `sample_rate` is f32-rounded here (exactly as the CPU);
/// `lambda` is `GetLambda(...)` supplied by the caller. `client` owns the handle for the whole fit
/// (residency, Pitfall 3). Empty `n` short-circuits to a zero-length handle (no launch).
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_mvs_weights_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der_h: &Handle,
    rand_seed: u64,
    sample_rate: f64,
    lambda: f64,
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
        // f32-round the rate exactly as the CPU (`TMvsSampler::SampleRate` is f32).
        let rate = f64::from(sample_rate as f32);
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![lambda, rate]));
        let seed_h = client.create(cubecl::bytes::Bytes::from_elems(vec![rand_seed]));
        // Serial single-thread launch (unit 0 loops the blocks); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        mvs_sample_kernel::launch::<SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(params_h, 2) },
            unsafe { ArrayArg::from_raw_parts(seed_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n) },
        );
        Ok(out)
    }
}

/// Host-readback wrapper over the device MVS draw: upload the derivatives, draw the resident sample
/// weights, then read the buffer back to a host `Vec<f64>`. This is the seam the self-oracle
/// exercises (device MVS vs the frozen CPU `mvs_sample_weights`); it is NOT the residency fold path
/// (that keeps the handle on-device). A read-back failure surfaces [`CbError::Degenerate`] (WR-05),
/// never a silent zero buffer.
pub(crate) fn draw_mvs_weights_host(
    derivatives: &[f64],
    rand_seed: u64,
    sample_rate: f64,
    lambda: f64,
    n: usize,
) -> CbResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let der_h = client.create(cubecl::bytes::Bytes::from_elems(derivatives.to_vec()));
    let handle =
        launch_mvs_weights_resident(&client, &der_h, rand_seed, sample_rate, lambda, n)?;
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL MVS read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}
