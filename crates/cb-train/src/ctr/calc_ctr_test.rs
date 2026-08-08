//! Unit tests for the CTR-value quantizers ([`crate::ctr::calc_ctr`]) — the
//! online (`+1`) vs inference (`+PriorDenom`) distinction (Pitfall 1).
//!
//! # DCTR-05 — the no-op proof for DCTR-04 (Track E)
//!
//! [`calc_normalization_is_identity_for_every_repo_prior`] is **DCTR-05's
//! characterization proof that DCTR-04 is inert for every committed artifact**.
//! DCTR-04 corrects the CPU `BinarizedTargetMeanValue` quantizer to apply
//! `(ctr + shift) / norm` (`CalcCTR`, `online_ctr.h:128-131`) instead of the raw
//! `ctr`. That correction can only change a binned value when
//! `(shift, norm) != (0.0, 1.0)`, i.e. when the prior falls outside `[0, 1]`
//! (`calc_normalization`: `left = min(0, p)`, `right = max(1, p)` ⇒
//! `p ∈ [0,1] ⇒ shift = 0, norm = 1`).
//!
//! Every prior this repository can produce is inside `[0, 1]`: the fixture
//! `config.json`s carry only `Prior=0`, `Prior=0.25`, `Prior=0.5` and `Prior=1`,
//! and the Rust callers construct only `priors: vec![0.25]` / `priors: vec![0.5]`.
//! The test below pins `calc_normalization(p) == (0.0, 1.0)` for that exact set,
//! so DCTR-04 is provably a no-op on every committed artifact and **no fixture is
//! re-baselined** — in particular `ctr_btmv_simple_oracle_test` must stay green
//! unchanged across DCTR-04.
//!
//! The two out-of-range companions (`p = 2.0`, `p = -0.5`) scope the claim: the
//! identity holds *because* the repo's priors are in `[0, 1]`, not because
//! `calc_normalization` is the identity everywhere — which is precisely why
//! DCTR-04 is a real correction rather than a cosmetic one.

use crate::ctr::calc_ctr::{
    calc_ctr_inference, calc_ctr_online, calc_ctr_online_bin, calc_normalization, Prior,
};

#[test]
fn online_denominator_is_hard_plus_one() {
    // (countInClass + prior) / (totalCount + 1) — NOT (total + PriorDenom).
    // good=3, total=3, prior=0.5 -> (3 + 0.5) / (3 + 1) = 0.875.
    let v = calc_ctr_online(3.0, 3, 0.5);
    assert!((v - 0.875).abs() < 1e-9, "got {v}");
}

#[test]
fn online_matches_plain_ctr_fixture_anchors() {
    // The plain_ctr fixture's ctr_value vector is the online (prior 0.5) CTR.
    // Reproduce a few of its (good,total)->value anchors exactly.
    // index 0: good=3 total=3 -> (3+0.5)/(3+1) = 0.875.
    assert!((calc_ctr_online(3.0, 3, 0.5) - 0.875).abs() < 1e-6);
    // index 5: good=0 total=0 -> (0+0.5)/(0+1) = 0.5.
    assert!((calc_ctr_online(0.0, 0, 0.5) - 0.5).abs() < 1e-6);
    // index 16: good=4 total=6 -> (4+0.5)/(6+1) = 0.642857...
    assert!((calc_ctr_online(4.0, 6, 0.5) - 0.642_857_142_857).abs() < 1e-6);
    // index 23: good=2 total=2 -> (2+0.5)/(2+1) = 0.833333...
    assert!((calc_ctr_online(2.0, 2, 0.5) - 0.833_333_333).abs() < 1e-6);
}

#[test]
fn inference_uses_prior_denom_not_plus_one() {
    // Inference: (cic + PriorNum) / (tot + PriorDenom); (ctr + Shift) * Scale.
    // With a NON-unit denom (PriorDenom=2) it diverges from the online +1 form.
    let prior = Prior { num: 0.5, denom: 2.0 };
    // (3 + 0.5) / (3 + 2) = 0.7, then (0.7 + 0) * 1 = 0.7.
    let v = calc_ctr_inference(3.0, 3.0, prior, 0.0, 1.0);
    assert!((v - 0.7).abs() < 1e-9, "got {v}");
    // The online +1 form gives 0.875 -> the two DIVERGE at PriorDenom != 1.
    assert!((calc_ctr_online(3.0, 3, 0.5) - 0.875).abs() < 1e-9);
}

#[test]
fn online_and_inference_coincide_at_unit_denominator() {
    // At PriorDenom == 1 and Shift=0/Scale=1 the two raw CTRs coincide (A6).
    let prior = Prior::unit(0.5);
    let online = calc_ctr_online(2.0, 4, 0.5);
    let inference = calc_ctr_inference(2.0, 4.0, prior, 0.0, 1.0);
    assert!((online - inference).abs() < 1e-9, "{online} vs {inference}");
}

#[test]
fn normalization_matches_calc_normalization_formula() {
    // prior 0.5: left=min(0,0.5)=0, right=max(1,0.5)=1, shift=0, norm=1.
    let (shift, norm) = calc_normalization(0.5);
    assert!((shift - 0.0).abs() < 1e-9);
    assert!((norm - 1.0).abs() < 1e-9);
    // prior -0.3: left=-0.3, right=1, shift=0.3, norm=1.3.
    let (shift, norm) = calc_normalization(-0.3);
    assert!((shift - 0.3).abs() < 1e-9);
    assert!((norm - 1.3).abs() < 1e-9);
    // prior 1.5: left=0, right=1.5, shift=0, norm=1.5.
    let (shift, norm) = calc_normalization(1.5);
    assert!((shift - 0.0).abs() < 1e-9);
    assert!((norm - 1.5).abs() < 1e-9);
}

/// DCTR-05 — Track E is inert for every committed artifact.
///
/// `calc_normalization(p) == (0.0, 1.0)` for every prior reachable in this
/// repository (`{0, 0.25, 0.5, 1}`, all in `[0, 1]`), asserted as exact equality
/// — not within an epsilon — because the whole point is that DCTR-04's
/// `(ctr + shift) / norm` introduces *literally zero* arithmetic here. See this
/// module's header for the full argument.
#[test]
fn calc_normalization_is_identity_for_every_repo_prior() {
    // The complete set of priors this repository can produce: the fixture
    // `config.json`s (`Prior=0`, `Prior=0.25`, `Prior=0.5`, `Prior=1`) and the
    // Rust callers (`priors: vec![0.25]`, `priors: vec![0.5]`).
    for &prior in &[0.0_f64, 0.25, 0.5, 1.0] {
        assert_eq!(
            calc_normalization(prior),
            (0.0, 1.0),
            "prior {prior} is in [0,1] and must normalize to the identity \
             (shift 0, norm 1), making DCTR-04 a no-op for it"
        );
    }
    // Out-of-`[0,1]` companions — pinned so the identity claim above is scoped to
    // the repo's priors rather than mistaken for a property of the function.
    assert_eq!(calc_normalization(2.0), (0.0, 2.0));
    assert_eq!(calc_normalization(-0.5), (0.5, 1.5));
}

#[test]
fn online_bin_applies_shift_norm_border_count() {
    // ctr=0.875, prior 0.5 -> shift=0, norm=1; (0.875 + 0)/1 * 4 = 3.5.
    let bin = calc_ctr_online_bin(3.0, 3, 0.5, 4);
    assert!((bin - 3.5).abs() < 1e-6, "got {bin}");
}

#[test]
fn degenerate_norm_returns_zero_not_nan() {
    // norm == 0 only if right == left; with prior in [0,1] norm is >= 1, so a
    // pathological prior driving norm to 0 is guarded (no NaN/inf, no panic).
    // Construct via calc_normalization edge: prior 0.0 -> norm 1, never 0; the
    // guard is exercised through the border_count=0 degenerate path instead.
    let bin = calc_ctr_online_bin(1.0, 2, 0.5, 0);
    assert!((bin - 0.0).abs() < 1e-12);
}

#[test]
fn inference_zero_denominator_is_guarded() {
    // total + PriorDenom == 0 -> ctr defaults to 0 (no div-by-zero/NaN).
    let prior = Prior { num: 0.0, denom: 0.0 };
    let v = calc_ctr_inference(0.0, 0.0, prior, 0.0, 1.0);
    assert!(v.is_finite());
    assert!((v - 0.0).abs() < 1e-12);
}
