//! GPUT-20 (Phase 13 Plan 09, W9): Langevin / SGLB seeded-Gaussian noise on device — the SMALLEST
//! device family, landed last (D-09). `AddLangevinNoise` adds a per-element seeded Gaussian
//! (`coefficient · std_normal(seed_i)`) to the RESIDENT reduced-derivative buffer on device,
//! layering on the existing der residency (Phase 12 Plan 06/07 sampling precedent). The mask never
//! crosses the seam: the noised der stays a resident device handle (D-08 — no per-tree host
//! round-trip of the n-length buffer).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! A serial `#[cube]` kernel (unit 0, mirroring [`crate::kernels::mvs_device`] /
//! [`crate::kernels::bootstrap_device`]) that, per element `i`, RESEEDS a fresh
//! [`cb_core::TFastRng64`] from `rand_seed + i`, `advance(10)`s it, draws ONE standard normal via
//! the Marsaglia-polar rejection loop transcribed INLINE (`util/random/normal.h:11-24`, the exact
//! `cb_core::normal::std_normal` draw order — the kernel body cannot reach `cb_core`, and cb-backend
//! must NEVER gain a `cb-train` dep, the feature-unification landmine), and adds
//! `coefficient · normal` to the resident der. The per-element reseed makes the noise embarrassingly
//! parallel in principle; the serial loop is chosen to match the MVS / bootstrap device precedent
//! (single-block fixtures, MVP scope) and to keep the draw stream trivially deterministic.
//!
//! # Draw-order fidelity (RESEARCH Pitfall 4)
//!
//! The Marsaglia-polar loop consumes a VARIABLE, even number of `gen_rand_real1` uniforms per sample
//! (it draws `(x, y)` PAIRS and rejects any pair outside the open unit disc). The device transcribes
//! that loop bit-for-bit — a wrong draw order or count would shift the sample beyond the ε=1e-4
//! device bar. Because each element reseeds from its OWN `rand_seed + i`, a per-element count
//! divergence is isolated to that element's value (not propagated), so the self-oracle's per-element
//! value match IS the per-element draw-count guard.
//!
//! # Termination / DoS-safety (threat T-13-17)
//!
//! The rejection loop retries until `(x, y)` lands strictly inside the open unit disc
//! (`0 < r <= 1`). `gen_rand_real1()` returns a value in `[0, 1]`, so the accept region has positive
//! measure — the expected iteration count is `4/π ≈ 1.27`, bounded, with no infinite loop on
//! well-formed draws (the [`crate::kernels::bootstrap_device`] Poisson-loop precedent uses the SAME
//! unbounded-with-flag shape).
//!
//! # f64-typed seam (WR-02)
//!
//! The RNG real is `(GenRand() >> 11) · (1/(2^53-1))` — an f64 quantity over 64-bit state; WGSL has
//! neither f64 nor u64, so a genuine `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather
//! than an opaque JIT crash. The in-env rocm/cuda/cpu path is unaffected. NO infinity literal in any
//! `#[cube]` body (Pattern D — the rejection loop uses only finite arithmetic). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_compute::Loss;
use cb_core::{CbError, CbResult};

use crate::SelectedRuntime;

/// LCG multiplier `A` (`cb_core::rng::LCG_MULTIPLIER`, `0x5851F42D4C957F2D`) — transcribed inline
/// (the `#[cube]` body cannot reach `cb_core`), matching [`crate::kernels::mvs_device`] /
/// [`crate::kernels::bootstrap_device`].
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// `1 / (2^53 - 1)` — the `ToRandReal1` divisor (`common_ops.h`), matching
/// [`cb_core::TFastRng64::gen_rand_real1`] exactly.
const REAL1_INV: f64 = 1.0 / 9_007_199_254_740_991.0;

// ===========================================================================
// #[cube] RNG primitives (transcribed from cb_core::TFastRng64 — bit-for-bit,
// mirroring crate::kernels::mvs_device so the per-element reseed matches)
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
// #[cube] serial Langevin (AddLangevinNoise) kernel
// ===========================================================================

/// Serial (unit 0) Langevin seeded-Gaussian kernel over the resident derivatives. `params =
/// [coefficient]` (the caller's SGLB/`DiffusionTemperature`-derived noise scale); `seed =
/// [rand_seed]` (the ONE main-stream `GenRand()` the host took). Adds `coefficient · std_normal_i`
/// to each resident der IN PLACE, where `std_normal_i` is drawn from a fresh
/// `from_seed(rand_seed + i).advance(10)` stream via the inline Marsaglia-polar loop. The noised der
/// stays device-resident (no `read_one`). No `-inf` literal; no `cb_core` reach.
#[cube(launch)]
fn langevin_noise_kernel(der: &mut Array<f64>, params: &Array<f64>, seed: &Array<u64>) {
    if ABSOLUTE_POS == 0 {
        let a = LCG_MULTIPLIER;
        let coefficient = params[0];
        let rand_seed = seed[0];
        let n = der.len();

        let mut i = 0usize;
        while i < n {
            // --- Per-element reseed: elem_rng = from_seed(rand_seed + i).advance(10). ---
            // (Verbatim transcription of TFastRng64::from_seed + ::new, mirroring mvs_device.)
            let s = rand_seed + u64::cast_from(i);
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

            // --- Marsaglia-polar std_normal (normal.h:11-24), draw order bit-for-bit. ---
            // Unbounded rejection loop with an accept flag (the bootstrap Poisson-loop shape); the
            // accept region has positive measure so it terminates in ≈1.27 iterations in expectation.
            let mut accepted = false;
            let mut result = 0.0_f64;
            while !accepted {
                // x = gen_rand_real1() * 2 - 1  (r1 supplies the high 32 bits, r2 the low 32).
                r1x = r1x * a + r1c;
                let hx = pcg_mix(r1x);
                r2x = r2x * a + r2c;
                let lx = pcg_mix(r2x);
                let rx64 = (u64::cast_from(hx) << 32u32) | u64::cast_from(lx);
                let ux = f64::cast_from(rx64 >> 11u32) * REAL1_INV;
                let x = ux * 2.0_f64 - 1.0_f64;
                // y = gen_rand_real1() * 2 - 1.
                r1x = r1x * a + r1c;
                let hy = pcg_mix(r1x);
                r2x = r2x * a + r2c;
                let ly = pcg_mix(r2x);
                let ry64 = (u64::cast_from(hy) << 32u32) | u64::cast_from(ly);
                let uy = f64::cast_from(ry64 >> 11u32) * REAL1_INV;
                let y = uy * 2.0_f64 - 1.0_f64;
                // r = x*x + y*y; accept iff 0 < r <= 1.
                let r = x * x + y * y;
                if !(r > 1.0_f64 || r <= 0.0_f64) {
                    // return x * sqrt(-2 * ln(r) / r).
                    result = x * (-2.0_f64 * r.ln() / r).sqrt();
                    accepted = true;
                }
            }

            der[i] += coefficient * result;
            i += 1usize;
        }
    }
}

// ===========================================================================
// Host launch wrappers (device-resident Handle + readback oracle wrapper)
// ===========================================================================

/// Reject the (impossible) wgpu f64/u64 path with a typed error (WR-02), mirroring
/// [`crate::kernels::mvs_device`] / [`crate::kernels::bootstrap_device`]. Kept in one place so every
/// entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device Langevin noise requires f64 + u64 device channels; the wgpu backend has neither \
         (WR-02). Use the rocm/cuda/cpu backend for the Langevin draw."
            .to_owned(),
    )
}

/// Add the device-resident Langevin seeded Gaussian to the resident derivatives `der_h` (`n`
/// objects) IN PLACE, returning the (same) resident buffer HANDLE WITHOUT reading it back (D-08).
/// `rand_seed` is the ONE main-stream `GenRand()` the host took; `coefficient` is the caller's SGLB
/// noise scale. `client` owns the handle for the whole fit (residency, Pitfall 3). Empty `n`
/// short-circuits to a zero-length handle (no launch, never a 0-len `read_one`).
///
/// # Errors
/// [`CbError::OutOfRange`] on the wgpu backend (no f64/u64 channel, WR-02).
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn launch_langevin_resident(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    der_h: &Handle,
    rand_seed: u64,
    coefficient: f64,
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
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![coefficient]));
        let seed_h = client.create(cubecl::bytes::Bytes::from_elems(vec![rand_seed]));
        // Serial single-thread launch (unit 0 loops the objects); one cube, one unit.
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: 1, y: 1, z: 1 };
        langevin_noise_kernel::launch::<SelectedRuntime>(
            client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(der_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(params_h, 1) },
            unsafe { ArrayArg::from_raw_parts(seed_h, 1) },
        );
        Ok(der_h.clone())
    }
}

/// Host-readback wrapper over the device Langevin draw: upload the derivatives, add the resident
/// seeded Gaussian, then read the buffer back to a host `Vec<f64>`. This is the seam the self-oracle
/// exercises (device Langevin vs the frozen CPU `coefficient · std_normal` sequence); it is NOT the
/// residency fold path (that keeps the handle on-device). A read-back failure surfaces
/// [`CbError::Degenerate`] (WR-05), never a silent zero buffer.
///
/// # Errors
/// [`CbError`] propagated from the device launch / read-back.
pub(crate) fn draw_langevin_host(
    derivatives: &[f64],
    rand_seed: u64,
    coefficient: f64,
    n: usize,
) -> CbResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
    let der_h = client.create(cubecl::bytes::Bytes::from_elems(derivatives.to_vec()));
    let handle = launch_langevin_resident(&client, &der_h, rand_seed, coefficient, n)?;
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("CubeCL Langevin read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}

/// Whether `loss` is device-covered by the Langevin/SGLB noise path (GPUT-20, Pattern A). Langevin
/// adds noise to the POINTWISE reduced derivatives, so it is covered for the covered pointwise der
/// family (RMSE / Logloss / CrossEntropy) and EXPLICITLY NOT supported on the pairwise oracle
/// (`is_pairwise_scoring` — upstream `pairwise_oracle.h` `CB_ENSURE`s Langevin is not supported for
/// the pairwise leaf/scoring path, A4). A `*Pairwise` (e.g. PairLogit-pairwise) + Langevin config
/// therefore falls back to CPU (`Ok(None)`) at the session gate. This predicate is the one exercised
/// directly by the langevin self-oracle (the private session `LangevinState` gate consumes it).
#[must_use]
pub(crate) fn langevin_covered_loss(loss: &Loss) -> bool {
    if cb_compute::is_pairwise_scoring(loss) {
        // A4: Langevin NOT supported on the pairwise oracle — decline (→ CPU fallback).
        return false;
    }
    matches!(loss, Loss::Rmse | Loss::Logloss | Loss::CrossEntropy)
}
