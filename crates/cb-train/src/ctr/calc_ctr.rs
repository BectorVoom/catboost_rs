//! CTR-value quantization — the online (training) and inference (model-side)
//! forms as SEPARATE functions (RESEARCH Pitfall 1).
//!
//! # Two distinct denominators (Pitfall 1 — the load-bearing distinction)
//!
//! The online (training) [`calc_ctr_online`] uses denominator `totalCount + 1`
//! (`online_ctr.h:128-131` — the denom is a HARD `+1`). The inference
//! (model-side) [`calc_ctr_inference`] uses `(countInClass + PriorNum) /
//! (totalCount + PriorDenom)` then `(ctr + Shift) * Scale`
//! (`online_ctr.h:289-292`). They COINCIDE numerically only when
//! `PriorDenom == 1` (the default unit-denominator priors `0/1`, `0.5/1`,
//! `1/1`); the in-scope fixtures pin denom = 1 so they align, but the two are
//! implemented as separate code paths so a non-unit `PriorDenom` is handled
//! correctly (the divergence Pitfall 1 warns about).
//!
//! # Normalization (`CalcNormalization`, `online_ctr.cpp:102-111`)
//!
//! `left = min(0, prior); right = max(1, prior); shift = -left; norm =
//! right - left`. The online form maps the quantized CTR into a `[0, border]`
//! bin via `(ctr + shift) / norm * borderCount`; the inference form applies the
//! per-CTR `Shift`/`Scale` baked into the model (which encode the same
//! normalization).
//!
//! # Parity discipline
//!
//! These are per-element scalar quantizers (no vector reduction), so no
//! `sum_f64` is involved; the inputs (counts, priors) come from the
//! integer-exact / `sum_f64`-reduced upstream paths. Checked arithmetic; no
//! `unwrap`/`expect`/panic; no `anyhow`.

/// A CTR prior `(num, denom)` (`TPrior`, `cat_feature_options.cpp:118-138`):
/// the additive `PriorNum` numerator and the `PriorDenom` denominator weight.
/// Default unit-denominator priors are `0/1`, `0.5/1`, `1/1` (Borders/Buckets/
/// BinarizedTargetMeanValue) and `0/1` (Counter/FeatureFreq/FloatTargetMean).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prior {
    /// The additive numerator (`PriorNum`).
    pub num: f64,
    /// The denominator weight (`PriorDenom`); `1.0` for the default priors.
    pub denom: f64,
}

impl Prior {
    /// A unit-denominator prior `num / 1` (the default-prior shape; the in-scope
    /// fixtures pin denom = 1 so online `+1` and inference `+PriorDenom`
    /// coincide, RESEARCH A6).
    #[must_use]
    pub fn unit(num: f64) -> Self {
        Self { num, denom: 1.0 }
    }
}

/// The `(shift, norm)` normalization pair from a prior
/// (`CalcNormalization`, `online_ctr.cpp:102-111`): `left = min(0, prior);
/// right = max(1, prior); shift = -left; norm = right - left`.
///
/// The `prior` here is the scalar prior numerator used by the online quantizer
/// (the `PriorNum` for the single-prior path).
#[must_use]
pub fn calc_normalization(prior: f64) -> (f64, f64) {
    let left = f64::min(0.0, prior);
    let right = f64::max(1.0, prior);
    let shift = -left;
    let norm = right - left;
    (shift, norm)
}

/// The ONLINE (training) CTR value (`CalcCTR`, `online_ctr.h:128-131`):
/// `ctr = (countInClass + prior) / (totalCount + 1)` — the denominator is a HARD
/// `+1` (NOT `+ PriorDenom`; that is the inference form). This is the raw CTR
/// before the `(ctr + shift) / norm * borderCount` quantization.
///
/// `count_in_class` is the bucket's good count (e.g. `N[1]` for binclf Borders);
/// `total_count` its total; `prior` the additive numerator.
#[must_use]
pub fn calc_ctr_online(count_in_class: f64, total_count: i64, prior: f64) -> f64 {
    // (countInClass + prior) / (totalCount + 1) — online denom is hard +1.
    (count_in_class + prior) / (total_count as f64 + 1.0)
}

/// The ONLINE quantized CTR bin (`CalcCTR`, `online_ctr.h:128-131`, full form):
/// `(ctr + shift) / norm * borderCount` where `ctr` is [`calc_ctr_online`] and
/// `(shift, norm)` come from [`calc_normalization`]. Returns the floating bin
/// position (the caller truncates to `ui8` exactly as upstream's implicit
/// `float -> ui8` cast).
///
/// `border_count` is `0` → the quantizer returns `0.0` (no borders, degenerate).
#[must_use]
pub fn calc_ctr_online_bin(
    count_in_class: f64,
    total_count: i64,
    prior: f64,
    border_count: usize,
) -> f64 {
    let ctr = calc_ctr_online(count_in_class, total_count, prior);
    let (shift, norm) = calc_normalization(prior);
    if norm == 0.0 {
        return 0.0;
    }
    (ctr + shift) / norm * border_count as f64
}

/// The INFERENCE (model-side) CTR value (`TModelCtr::Calc`,
/// `online_ctr.h:289-292`): `ctr = (countInClass + PriorNum) /
/// (totalCount + PriorDenom)` then `(ctr + Shift) * Scale`. This is the
/// model-side form the `cb-model` apply path uses; it COINCIDES with
/// [`calc_ctr_online_bin`] only when `PriorDenom == 1` (Pitfall 1).
///
/// `count_in_class` / `total_count` are the bucket's good/total counts (or
/// `Sum`/`Count` for mean types, `total`/`CounterDenominator` for Counter);
/// `prior` carries `(PriorNum, PriorDenom)`; `shift`/`scale` are the baked
/// per-CTR `Shift`/`Scale`.
#[must_use]
pub fn calc_ctr_inference(
    count_in_class: f64,
    total_count: f64,
    prior: Prior,
    shift: f64,
    scale: f64,
) -> f64 {
    // ctr = (cic + PriorNum) / (tot + PriorDenom); return (ctr + Shift) * Scale.
    let denom = total_count + prior.denom;
    let ctr = if denom == 0.0 {
        0.0
    } else {
        (count_in_class + prior.num) / denom
    };
    (ctr + shift) * scale
}
