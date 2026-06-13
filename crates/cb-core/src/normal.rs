//! `std_normal` — an exact, bit-for-bit Rust port of CatBoost's
//! `StdNormalDistribution` (`catboost-master/util/random/normal.h:11-24`), the
//! Marsaglia-polar / Box-Muller standard-normal draw over [`TFastRng64`].
//!
//! # Why a hand port (not `rand_distr`)
//!
//! The `random_strength` split-score perturbation (`TRandomScore::GetInstance`)
//! draws this normal PER scored split candidate. Parity hinges on the exact
//! *draw sequence*: the rejection loop consumes a VARIABLE number of
//! `gen_rand_real1()` uniforms per normal — a different sampler (e.g.
//! `rand_distr::Normal`, which uses the ziggurat algorithm) would consume a
//! different number of uniforms in a different order, desynchronising every
//! subsequent draw and picking different splits (RESEARCH Pitfall 3,
//! Don't-Hand-Roll table). So the upstream loop is transcribed verbatim; no
//! third-party sampler is permitted.
//!
//! # Parity contract (`normal.h:11-24`)
//!
//! ```text
//! do {
//!     x = GenRandReal1() * 2 - 1;
//!     y = GenRandReal1() * 2 - 1;
//!     r = x*x + y*y;
//! } while (r > 1 || r <= 0);
//! return x * sqrt(-2 * log(r) / r);
//! ```
//!
//! Each `GenRandReal1()` is the [`TFastRng64::gen_rand_real1`] primitive
//! (`(GenRand() >> 11) * (1 / (2^53 - 1))`, `common_ops.h:19`). The loop draws
//! uniforms in `(x, y)` PAIRS and rejects any pair landing outside the open unit
//! disc, so the RNG advances by an even number of `gen_rand` calls per normal.
//! `std::log` is the natural log (`f64::ln`); `std::sqrt` is `f64::sqrt`.

use crate::rng::TFastRng64;

/// Draw one standard-normal sample (`StdNormalDistribution<double>`,
/// `normal.h:11-24`) from `rng` via the Marsaglia-polar rejection loop, consuming
/// a variable number of [`TFastRng64::gen_rand_real1`] draws in the exact upstream
/// order. Used by the `random_strength` perturbation
/// (`TRandomScore::GetInstance` → `NormalDistribution(rng, 0, stDev)`).
///
/// # Termination
///
/// The loop retries until `(x, y)` lands strictly inside the open unit disc
/// (`0 < r <= 1`). `gen_rand_real1()` returns a value in `[0, 1]`, so each
/// rejected iteration advances `TFastRng64` and the accept region has positive
/// measure — the expected number of iterations is `4/π ≈ 1.27`, bounded, with no
/// infinite loop on well-formed draws (threat T-03-04-02).
#[must_use]
pub fn std_normal(rng: &mut TFastRng64) -> f64 {
    loop {
        // x = (T)rng.GenRandReal1() * T(2) - T(1);
        let x = rng.gen_rand_real1() * 2.0 - 1.0;
        // y = (T)rng.GenRandReal1() * T(2) - T(1);
        let y = rng.gen_rand_real1() * 2.0 - 1.0;
        // r = x * x + y * y;
        let r = x * x + y * y;
        // } while (r > T(1) || r <= T(0));
        if !(r > 1.0 || r <= 0.0) {
            // return x * std::sqrt(-T(2) * std::log(r) / r);
            return x * (-2.0 * r.ln() / r).sqrt();
        }
    }
}
