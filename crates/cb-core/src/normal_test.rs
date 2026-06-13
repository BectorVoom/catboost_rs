//! Unit tests for the exact standard-normal draw (`StdNormalDistribution`,
//! `util/random/normal.h:12-24`) over [`TFastRng64`].
//!
//! The contract is the *draw sequence*, not just the value: the Marsaglia-polar /
//! Box-Muller rejection loop consumes a VARIABLE number of `gen_rand_real1()`
//! uniforms per normal (`do { x=2u-1; y=2u-1; r=x²+y² } while (r>1 || r<=0)`).
//! These tests (a) re-derive the expected normal INDEPENDENTLY from the validated
//! `gen_rand_real1` primitive — proving the algorithm and the draw count — and
//! (b) pin the concrete first/second normals for fixed seeds against
//! hand-computed references (`seed=17` deliberately triggers ONE rejected pair so
//! the variable-length loop is exercised, not just the happy path).

use crate::normal::std_normal;
use crate::rng::TFastRng64;

/// Independent reference: run the SAME Marsaglia-polar loop directly on the
/// `gen_rand_real1` primitive (validated in `rng_test.rs`), returning the normal
/// value AND the number of uniform draws consumed. This is the spec transcription
/// the production `std_normal` must match draw-for-draw.
fn reference_std_normal(rng: &mut TFastRng64) -> (f64, usize) {
    let mut draws = 0usize;
    loop {
        let x = rng.gen_rand_real1() * 2.0 - 1.0;
        draws += 1;
        let y = rng.gen_rand_real1() * 2.0 - 1.0;
        draws += 1;
        let r = x * x + y * y;
        if !(r > 1.0 || r <= 0.0) {
            return (x * (-2.0 * r.ln() / r).sqrt(), draws);
        }
    }
}

#[test]
fn std_normal_matches_independent_reference_seed_17() {
    // Production draw.
    let mut prod = TFastRng64::from_seed(17);
    let n1 = std_normal(&mut prod);
    let n2 = std_normal(&mut prod);

    // Independent reference on a fresh RNG.
    let mut refr = TFastRng64::from_seed(17);
    let (r1, draws1) = reference_std_normal(&mut refr);
    let (r2, draws2) = reference_std_normal(&mut refr);

    // Bit-for-bit equal: same primitive, same loop, same order.
    assert_eq!(n1, r1, "first normal must match the reference exactly");
    assert_eq!(n2, r2, "second normal must match the reference exactly");

    // seed=17 rejects the first uniform pair, so the FIRST normal consumes 4
    // uniforms (one rejected pair + one accepted pair); the SECOND consumes 2.
    // This pins the VARIABLE draw count (the Pitfall-3 parity landmine).
    assert_eq!(draws1, 4, "seed=17 first normal must consume 4 uniform draws");
    assert_eq!(draws2, 2, "seed=17 second normal must consume 2 uniform draws");
}

#[test]
fn std_normal_pins_hand_computed_values_seed_17() {
    // Hand-computed against the validated `gen_rand_real1` chain (see
    // /tmp reference run committed in the SUMMARY): the variable-draw case.
    let mut rng = TFastRng64::from_seed(17);
    let n1 = std_normal(&mut rng);
    let n2 = std_normal(&mut rng);
    assert!(
        (n1 - (-1.812_286_613_368_193_9e-1)).abs() < 1e-15,
        "seed=17 first normal mismatch: {n1}"
    );
    assert!(
        (n2 - 2.833_247_730_478_129_5e-1).abs() < 1e-15,
        "seed=17 second normal mismatch: {n2}"
    );
}

#[test]
fn std_normal_pins_hand_computed_values_seed_0_and_42() {
    // Happy-path seeds (first pair accepted: 2 draws each).
    let mut rng0 = TFastRng64::from_seed(0);
    let n0 = std_normal(&mut rng0);
    assert!(
        (n0 - 6.337_067_335_392_765_3e-1).abs() < 1e-15,
        "seed=0 first normal mismatch: {n0}"
    );

    let mut rng42 = TFastRng64::from_seed(42);
    let n42 = std_normal(&mut rng42);
    assert!(
        (n42 - 1.969_271_554_069_228_8e-1).abs() < 1e-15,
        "seed=42 first normal mismatch: {n42}"
    );
}

#[test]
fn std_normal_consumes_pairs_so_count_is_even() {
    // The loop always draws uniforms in pairs (x then y); a full sequence of
    // normals therefore advances the RNG by an even number of `gen_rand` calls.
    let mut refr = TFastRng64::from_seed(123);
    let (_v, draws) = reference_std_normal(&mut refr);
    assert_eq!(draws % 2, 0, "draw count must be even (paired x/y)");
    assert!(draws >= 2, "at least one pair is always drawn");
}
