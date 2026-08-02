//! Bit-exactness tests for [`crate::fast_approx`] against reference vectors
//! generated from the REAL upstream implementations
//! (`tests/fixtures/fast_approx_ref_generator.cpp.txt`, compiled against the
//! v1.2.10 `fast_exp_avx2.avx2.cpp.o` object and re-compiled upstream sources
//! on 2026-08-03). Every pair must match to the BIT — a tolerance here would
//! defeat the entire point of porting approximate transcendentals.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use super::{apply_learning_rate_exp, fast_exp, fast_logf, fmath_expd, logloss_ders_exp};

fn ref_lines() -> Vec<(String, u64, u64)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("fast_approx_ref.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reference vectors must load from {path:?}: {e}"));
    text.lines()
        .map(|line| {
            let mut it = line.split_whitespace();
            let name = it.next().expect("name").to_owned();
            let a = u64::from_str_radix(it.next().expect("in"), 16).expect("in hex");
            let b = u64::from_str_radix(it.next().expect("out"), 16).expect("out hex");
            (name, a, b)
        })
        .collect()
}

#[test]
fn fast_exp_matches_upstream_bit_for_bit() {
    let mut checked = 0usize;
    for (name, in_bits, out_bits) in ref_lines() {
        if name != "fast_exp" {
            continue;
        }
        let x = f64::from_bits(in_bits);
        let got = fast_exp(x);
        assert_eq!(
            got.to_bits(),
            out_bits,
            "fast_exp({x:e}) = {got:e}, upstream {:e}",
            f64::from_bits(out_bits)
        );
        checked += 1;
    }
    assert!(checked > 1000, "vacuous: only {checked} fast_exp vectors");
}

#[test]
fn fmath_expd_matches_upstream_bit_for_bit() {
    let mut checked = 0usize;
    for (name, in_bits, out_bits) in ref_lines() {
        if name != "fmath_expd" {
            continue;
        }
        let x = f64::from_bits(in_bits);
        let got = fmath_expd(x);
        assert_eq!(
            got.to_bits(),
            out_bits,
            "fmath_expd({x:e}) = {got:e}, upstream {:e}",
            f64::from_bits(out_bits)
        );
        checked += 1;
    }
    assert!(checked > 1000, "vacuous: only {checked} fmath_expd vectors");
}

#[test]
fn fast_logf_matches_upstream_bit_for_bit() {
    let mut checked = 0usize;
    for (name, in_bits, out_bits) in ref_lines() {
        if name != "fast_logf" {
            continue;
        }
        #[allow(clippy::cast_possible_truncation)]
        let x = f32::from_bits(in_bits as u32);
        let got = fast_logf(x);
        #[allow(clippy::cast_possible_truncation)]
        let want = f32::from_bits(out_bits as u32);
        assert_eq!(got.to_bits(), want.to_bits(), "fast_logf({x:e}) = {got:e}, upstream {want:e}");
        checked += 1;
    }
    assert!(checked > 500, "vacuous: only {checked} fast_logf vectors");
}

/// The composed per-doc learning-rate pipeline reproduces the instrumented
/// upstream fold-approx value observed in the 2026-08-02 localization run:
/// leaf delta `-0.0096153846153846159` (= -0.5/52), lr `0.1` — upstream stored
/// fold approx `0.99903645…` (its ln = `-0.00096401…`), NOT the exact
/// `exp(-0.1 * 0.0096153846…) = 0.99903892…`. The exact-exp value differs in
/// the 6th decimal — the two must be distinguishable, and the composed pipeline
/// must land on upstream's side.
#[test]
fn apply_learning_rate_reproduces_the_instrumented_fold_approx() {
    let delta_exp = fmath_expd(-0.009_615_384_615_384_616);
    let applied = apply_learning_rate_exp(delta_exp, 0.1);
    // Instrumented CLI printed the stored fold approx as 0.99903645 (8 sig
    // digits) for a doc starting at exp-approx 1.0.
    assert!(
        (applied - 0.999_036_45).abs() < 5e-9,
        "composed pipeline produced {applied:.10}, instrumented upstream stored 0.99903645"
    );
    let exact = f64::exp(-0.1 * 0.009_615_384_615_384_616);
    assert!(
        (applied - exact).abs() > 1e-7,
        "the approximate pipeline must be DISTINGUISHABLE from exact exp \
         (applied {applied:.12}, exact {exact:.12}) — if these agree the port is vacuous"
    );
}

#[test]
fn logloss_ders_exp_matches_the_upstream_rounding_order() {
    // e = 1 (approx 0) → p = 0.5 exactly, der1 = ±0.5 exactly.
    let (d1, d2) = logloss_ders_exp(1.0, 1.0);
    assert_eq!(d1, 0.5);
    assert_eq!(d2, -0.25);
    // p computed as 1 - 1/(1+e) — for e where e/(1+e) rounds differently the
    // two forms diverge in the last ulp; pin the upstream form.
    let e = 0.997_216_149_567_183_2_f64;
    let p_upstream = 1.0 - 1.0 / (1.0 + e);
    let (d1, _) = logloss_ders_exp(e, 0.0);
    assert_eq!(d1.to_bits(), (-p_upstream).to_bits());
}
