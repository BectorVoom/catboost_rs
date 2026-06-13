//! `TFastRng64` — an exact, bit-for-bit Rust port of CatBoost's PCG-based fast
//! PRNG (`catboost-master/util/random/fast.h`, `lcg_engine.h`, `common_ops.h`).
//!
//! # Non-cryptographic
//!
//! **`TFastRng64` is NOT cryptographically secure.** It is a deterministic,
//! seekable linear-congruential / PCG generator whose sole purpose is to
//! reproduce upstream CatBoost's raw random bitstream for parity testing and for
//! the sampling / permutation logic later phases build on. Its output is fully
//! predictable from the seed. **Never** use it to generate secrets, tokens,
//! nonces, or anything where unpredictability matters (RESEARCH Security Domain
//! V6, threat T-01-06).
//!
//! # Parity contract
//!
//! Two 32-bit PCG-XSH-RR LCG streams are concatenated into 64 bits exactly as in
//! `fast.h`. Every multiply / add uses `wrapping_*` so the arithmetic matches
//! C++'s defined unsigned wraparound and cannot trigger a debug-overflow panic
//! (RESEARCH Pitfall 5, threat T-01-04). Validated against the vendored vectors
//! in `fast_ut.cpp` (see `rng_test.rs`).

use crate::error::{CbError, CbResult};

/// LCG multiplier `A` shared by every stream (`6364136223846793005`,
/// i.e. `0x5851F42D4C957F2D`) — `fast.h` `TLcgIterator<ui64, ULL(6364136223846793005)>`.
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// PCG output mixer (`TPCGMixer::Mix` in `fast.h`): XSH-RR on the 64-bit state,
/// producing a 32-bit result.
#[inline]
fn pcg_mix(x: u64) -> u32 {
    // const ui32 xorshifted = ((x >> 18u) ^ x) >> 27u;
    let xorshifted = (((x >> 18) ^ x) >> 27) as u32;
    // const ui32 rot = x >> 59u;
    let rot = (x >> 59) as u32;
    // return RotateBitsRight(xorshifted, rot);
    xorshifted.rotate_right(rot)
}

/// `NPrivate::LcgAdvance` (`lcg_engine.cpp`): jump the LCG state forward by
/// `delta` steps in O(log delta), using the closed form
/// `seed[n] = A**n * seed[0] + (A**n - 1)/(A - 1) * addend`. All arithmetic is
/// wrapping, matching the C++ unsigned semantics exactly.
#[inline]
fn lcg_advance(seed: u64, lcg_base: u64, lcg_addend: u64, delta: u64) -> u64 {
    // Find the highest set bit of delta (T mask).
    let mut mask: u64 = 1;
    while mask != (1u64 << 63) && (mask << 1) <= delta {
        mask <<= 1;
    }
    let mut apow: u64 = 1; // A**m
    let mut adiv: u64 = 0; // (A**m - 1)/(A - 1)
    while mask != 0 {
        // m *= 2
        adiv = adiv.wrapping_mul(apow.wrapping_add(1));
        apow = apow.wrapping_mul(apow);
        if delta & mask != 0 {
            // m++
            adiv = adiv.wrapping_add(apow);
            apow = apow.wrapping_mul(lcg_base);
        }
        mask >>= 1;
    }
    seed.wrapping_mul(apow)
        .wrapping_add(lcg_addend.wrapping_mul(adiv))
}

/// One PCG-XSH-RR 32-bit stream: a 64-bit LCG state `x` with per-stream odd
/// addend `c`, mixed down to 32 bits on output. Mirrors `TLcgRngBase` composed
/// with `TLcgIterator` and `TPCGMixer`.
#[derive(Clone, Copy)]
struct Lcg32 {
    /// LCG state `X` (`TLcgRngBase::X`).
    x: u64,
    /// Per-stream addend `C = (seq << 1) | 1` (`TLcgIterator::C`), always odd.
    c: u64,
}

impl Lcg32 {
    /// `TLcgIterator(seq)` + state seed: `C = (seq << 1) | 1`, `X = seed`.
    #[inline]
    fn new(seed: u64, seq: u32) -> Self {
        let c = ((seq as u64) << 1) | 1;
        Self { x: seed, c }
    }

    /// `TReallyFastRng32(seed)`: the fixed-stream engine whose addend is `1`
    /// (`TFastLcgIterator<ui64, A, ULL(1)>`).
    #[inline]
    fn new_really_fast(seed: u64) -> Self {
        Self { x: seed, c: 1 }
    }

    /// `Iterate(x) = x*A + C` (`TLcgIterator::Iterate`), wrapping.
    #[inline]
    fn iterate(&self, x: u64) -> u64 {
        x.wrapping_mul(LCG_MULTIPLIER).wrapping_add(self.c)
    }

    /// `TLcgRngBase::GenRand`: `Mix(X = Iterate(X))` — iterate the state in
    /// place, then mix to 32 bits.
    #[inline]
    fn gen_rand32(&mut self) -> u32 {
        self.x = self.iterate(self.x);
        pcg_mix(self.x)
    }

    /// `TCommonRNG::GenRand64` for a 32-bit engine (`ToRand64`): low 32 bits from
    /// the first `GenRand`, high 32 bits from the second —
    /// `((ui64)x) | (((ui64)rng.GenRand()) << 32)`.
    #[inline]
    fn gen_rand64(&mut self) -> u64 {
        let low = self.gen_rand32() as u64;
        let high = self.gen_rand32() as u64;
        low | (high << 32)
    }

    /// `TLcgRngBase::Advance`: jump the state forward by `delta` steps.
    #[inline]
    fn advance(&mut self, delta: u64) {
        self.x = lcg_advance(self.x, LCG_MULTIPLIER, self.c, delta);
    }
}

/// `FixSeq` (`fast.cpp`): force the two streams onto distinct sequences. If the
/// low 31 bits of `seq1` and `seq2` collide, flip `seq2` to `~seq2`.
#[inline]
fn fix_seq(seq1: u32, seq2: u32) -> u32 {
    let mask = (!0u32) >> 1;
    if (seq1 & mask) == (seq2 & mask) {
        !seq2
    } else {
        seq2
    }
}

/// Exact port of CatBoost's `TFastRng64` (two PCG-XSH-RR 32-bit LCGs
/// concatenated to 64 bits). See the module docs for the non-cryptographic
/// caveat and parity contract.
#[derive(Clone, Copy)]
pub struct TFastRng64 {
    r1: Lcg32,
    r2: Lcg32,
}

impl TFastRng64 {
    /// Four-argument constructor (`TFastRng64(seed1, seq1, seed2, seq2)`):
    /// stream 1 takes `seq1` verbatim; stream 2's sequence is passed through
    /// [`fix_seq`] to guarantee the two streams differ.
    #[must_use]
    pub fn new(seed1: u64, seq1: u32, seed2: u64, seq2: u32) -> Self {
        Self {
            r1: Lcg32::new(seed1, seq1),
            r2: Lcg32::new(seed2, fix_seq(seq1, seq2)),
        }
    }

    /// One-argument constructor (`TFastRng64(ui64 seed)` via `TArgs`): derive the
    /// four parameters from a single seed by drawing them, in order, from a
    /// `TReallyFastRng32(seed)` — `Seed1 = GenRand64()`, `Seq1 = GenRand()`,
    /// `Seed2 = GenRand64()`, `Seq2 = GenRand()`.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut derive = Lcg32::new_really_fast(seed);
        let seed1 = derive.gen_rand64();
        let seq1 = derive.gen_rand32();
        let seed2 = derive.gen_rand64();
        let seq2 = derive.gen_rand32();
        Self::new(seed1, seq1, seed2, seq2)
    }

    /// `TFastRng64::GenRand`: `(R1.GenRand() << 32) | R2.GenRand()`. Note the
    /// order — stream 1 supplies the high 32 bits, stream 2 the low 32 bits.
    #[inline]
    pub fn gen_rand(&mut self) -> u64 {
        let x = self.r1.gen_rand32() as u64;
        let y = self.r2.gen_rand32() as u64;
        (x << 32) | y
    }

    /// `TFastRng64::Advance`: advance both underlying streams by `delta`.
    #[inline]
    pub fn advance(&mut self, delta: u64) {
        self.r1.advance(delta);
        self.r2.advance(delta);
    }

    /// Fallible uniform draw in `[0, bound)` via rejection sampling
    /// (`NPrivate::GenUniform`). Ports the C++ `Y_ABORT_UNLESS(max > 0)`
    /// precondition into a `Result`: a zero `bound` returns
    /// [`CbError::InvalidBound`] instead of panicking (D-13, threat T-01-05).
    ///
    /// # Errors
    ///
    /// Returns [`CbError::InvalidBound`] when `bound == 0`.
    pub fn try_uniform(&mut self, bound: u64) -> CbResult<u64> {
        if bound == 0 {
            return Err(CbError::InvalidBound { bound });
        }
        // randmax = RandMax() - RandMax() % bound, RandMax() == u64::MAX.
        let randmax = u64::MAX - (u64::MAX % bound);
        loop {
            let rand = self.gen_rand();
            if rand < randmax {
                return Ok(rand % bound);
            }
        }
    }

    /// Infallible uniform draw in `[0, bound)`. A zero `bound` has no valid
    /// output, so — rather than panic (`clippy::panic` is denied) — this returns
    /// `0` for the degenerate case; callers needing the error should use
    /// [`Self::try_uniform`]. For every valid `bound > 0` this is exactly the
    /// upstream `Uniform` result.
    #[inline]
    pub fn uniform(&mut self, bound: u64) -> u64 {
        self.try_uniform(bound).unwrap_or(0)
    }
}
