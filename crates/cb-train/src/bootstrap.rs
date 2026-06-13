//! Bootstrap / sampling (TRAIN-04) — per-iteration sample-weight and document
//! selection seeded by the Phase-1 [`cb_core::TFastRng64`], reproducing
//! upstream catboost 1.2.10's EXACT draw order.
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/tensor_search_helpers.cpp` (`Bootstrap`,
//! `GenerateRandomWeights`, `GenerateBayessianWeight`),
//! `catboost/private/libs/algo/calc_score_cache.cpp` (`SetSampledControl`,
//! `SetControlNoZeroWeighted`), `catboost/private/libs/algo/mvs.cpp`
//! (`TMvsSampler::GenSampleWeights`, `CalculateThreshold`), and
//! `catboost/private/libs/algo/greedy_tensor_search.cpp` (`DoBootstrap`,
//! per-tree `SamplingFrequency=PerTree` default).
//!
//! # CPU / object-sampling / non-pairwise contract (this slice)
//!
//! With the first-slice isolating params (`thread_count=1`, `Rsm=1`,
//! `random_strength=0`, object sampling unit, non-pairwise RMSE loss,
//! `SamplingFrequency=PerTree`) bootstrap runs ONCE PER TREE on the persistent
//! [`cb_core::TFastRng64`] (seeded `random_seed`); the draw stream is
//! CONTINUOUS across iterations — never reseeded per tree. The five upstream
//! dispatch arms (object sampling, non-pairwise):
//!
//! - [`EBootstrapType::No`]    — `SampleWeights` all `1.0`; `control` all `true`;
//!   ZERO RNG draws.
//! - [`EBootstrapType::Bayesian`] — `GenerateRandomWeights`: `rand_seed =
//!   rng.gen_rand()` (one draw on the main stream), then per 1000-element block
//!   `r = TFastRng64::from_seed(rand_seed + block_idx).advance(10)` and per
//!   object `w = (-ln(r.gen_rand_real1() + 1e-100))^bagging_temperature`
//!   (`powf`). `control` all `true` (BernoulliSampleRate == 1).
//! - [`EBootstrapType::Bernoulli`] — `SampleWeights` all `1.0`; the object
//!   subsample lives in `SetSampledControl`, which draws SEQUENTIALLY from the
//!   SAME continuous main stream (NO per-block reseed):
//!   `control[i] = rng.gen_rand_real1() < subsample`.
//! - [`EBootstrapType::Mvs`] — `performRandomChoice = false`; single
//!   8192-element block; `lambda` from the mean gradient magnitude (iter 0) or
//!   the previous tree's mean leaf L2 norm; per-block threshold via
//!   [`calculate_threshold`]; `SampleWeights[i] = (1/p) * (r.gen_rand_real1() <
//!   p)`; `control[i] = SampleWeights[i] > eps`.
//! - [`EBootstrapType::Poisson`] — UNSUPPORTED on CPU (upstream
//!   `bootstrap_options.cpp:27-33` throws "poisson bootstrap is not supported on
//!   CPU"). The dispatch surfaces [`CbError::Degenerate`]; there is no CPU
//!   oracle for it (D-11).
//!
//! # Parity discipline
//!
//! All draws go through [`cb_core::TFastRng64`] ONLY (never `rand`); all sums
//! route through `cb_core::sum_f64`. No `unwrap`/`expect`/panic in production.

use cb_core::{sum_f64, CbError, CbResult, TFastRng64};

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file), mounted as a child module
// so `cargo test -p cb-train bootstrap` selects them.
#[cfg(test)]
#[path = "bootstrap_test.rs"]
mod tests;

/// Block size for the per-block reseed in `GenerateRandomWeights` (Bayesian) and
/// `SetSampledControl` enumeration (`tensor_search_helpers.cpp:345`).
pub const BAYESIAN_BLOCK_SIZE: usize = 1000;

/// MVS block size (`mvs.h:48` `const ui32 BlockSize = 8192`).
pub const MVS_BLOCK_SIZE: usize = 8192;

/// The five upstream `EBootstrapType` values (CPU). Poisson is accepted as a
/// variant for dispatch completeness but is unsupported on the CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EBootstrapType {
    /// No sampling — every weight `1.0`, every object selected, no draws.
    No,
    /// Bayesian bagging — per-block reseed Bayesian weight.
    Bayesian,
    /// Bernoulli object subsample via the sequential `control` mask.
    Bernoulli,
    /// Minimal-Variance Sampling (the CPU default).
    Mvs,
    /// Poisson — GPU-only; rejected on CPU (mirrors upstream).
    Poisson,
}

/// The result of one `Bootstrap()` call: a per-object multiplicative
/// `sample_weight` (applied to gradients AND weights in the score/leaf math) and
/// a per-object `control` mask (`true` == the object participates in the split
/// SCORE histograms). Both are in the SAME object order the caller supplies the
/// derivatives in (the learn-permutation order upstream uses).
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapResult {
    /// Per-object sample weight (`fold->SampleWeights`).
    pub sample_weights: Vec<f64>,
    /// Per-object score-fold control mask (`TCalcScoreFold::Control`).
    pub control: Vec<bool>,
}

impl BootstrapResult {
    /// The trivial `No`-bootstrap result: all weights `1.0`, all selected.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            sample_weights: vec![1.0; n],
            control: vec![true; n],
        }
    }
}

/// `FastLog2f` (`library/cpp/fast_log/fast_log.h:62-76`) — a bit-manipulation
/// base-2 log APPROXIMATION (accuracy ~1e-5), NOT the exact `log2`. Bayesian
/// parity REQUIRES this exact approximation: substituting `f32::log2` shifts the
/// weight at the ~1e-5 scale and breaks the oracle. Transcribed verbatim.
// `excessive_precision` / `approx_constant` are intentionally allowed on the
// two fast-log helpers: the literals are VERBATIM transcriptions of upstream's
// C source constants (`fast_log.h`). Trimming them to the shortest round-tripping
// decimal (excessive_precision) or substituting `f32::consts::LN_2`
// (approx_constant) would change the exact f32 bit pattern and break Bayesian
// parity at the ~1e-5 oracle bound. The constant `0.69314718` is upstream's own
// truncated `ln(2)` literal — NOT the standard-library constant.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
#[inline]
fn fast_log2f(value: f32) -> f32 {
    // Constants transcribed at FULL upstream precision (fast_log.h:73-75); a
    // truncated literal shifts the result by ~1e-7 and compounds into the
    // Bayesian leaf values past the 1e-5 oracle bound.
    let vx_i = value.to_bits();
    let mx = f32::from_bits((vx_i & 0x007F_FFFF) | 0x3f00_0000);
    let mut y = vx_i as f32;
    y *= 1.192_092_895_507_812_5e-7_f32;
    y - 124.225_514_99_f32 - 1.498_030_302_f32 * mx - 1.725_879_99_f32 / (0.352_088_706_8_f32 + mx)
}

/// `FastLogf` (`fast_log.h:84-86`): `0.69314718 * FastLog2f(value)`.
#[allow(clippy::approx_constant)]
#[inline]
fn fast_logf(value: f32) -> f32 {
    0.693_147_18_f32 * fast_log2f(value)
}

/// `GenerateBayessianWeight` (`tensor_search_helpers.cpp:322-325`):
/// `w = -FastLogf(GenRandReal1() + 1e-100)` then `powf(w, baggingTemperature)`.
///
/// The exact widths matter: the draw is `f64`, `+ 1e-100` and `FastLogf` are
/// `f32`, and `powf(w, baggingTemperature)` is `f32`; the result widens to `f64`
/// for the host-side reduction. `FastLogf` is the bit-manipulation approximation
/// above — NOT `ln` (Bayesian parity hinges on this, ~1e-5 sensitivity).
#[inline]
fn bayesian_weight(bagging_temperature: f32, rng: &mut TFastRng64) -> f64 {
    let u = rng.gen_rand_real1();
    let w: f32 = -fast_logf((u as f32) + 1e-100_f32);
    f64::from(w.powf(bagging_temperature))
}

/// `GenerateRandomWeights` (`tensor_search_helpers.cpp:327`) for object sampling:
/// `baggingTemperature == 0` short-circuits to all-`1.0`; otherwise
/// `randSeed = rand->GenRand()` (one draw on `rng`), then per 1000-element block
/// `r = TFastRng64::from_seed(randSeed + blockIdx).advance(10)` supplies each
/// object's Bayesian weight in block order.
fn generate_random_weights(n: usize, bagging_temperature: f32, rng: &mut TFastRng64) -> Vec<f64> {
    if bagging_temperature == 0.0 {
        return vec![1.0; n];
    }
    let rand_seed = rng.gen_rand();
    let mut weights = vec![1.0_f64; n];
    let block_count = n.div_ceil(BAYESIAN_BLOCK_SIZE);
    for block_idx in 0..block_count {
        let mut block_rng = TFastRng64::from_seed(rand_seed.wrapping_add(block_idx as u64));
        block_rng.advance(10);
        let begin = block_idx * BAYESIAN_BLOCK_SIZE;
        let end = usize::min(begin + BAYESIAN_BLOCK_SIZE, n);
        for w in weights.get_mut(begin..end).unwrap_or(&mut []) {
            *w = bayesian_weight(bagging_temperature, &mut block_rng);
        }
    }
    weights
}

/// `SetSampledControl` (`calc_score_cache.cpp:1196-1200`) for object sampling:
/// `control[i] = rand->GenRandReal1() < BernoulliSampleRate`, drawn SEQUENTIALLY
/// from the same `rng` (no per-block reseed). `BernoulliSampleRate` is an `f32`
/// upstream (`calc_score_cache.cpp:262`), so the `subsample` threshold is the
/// f32-rounded value promoted back to `f64` for the comparison — matching the
/// exact promotion `(double)GenRandReal1() < (float)rate`. `subsample == 1.0`
/// fills `true` with no draw (`BernoulliSampleRate == 1` early-return).
fn set_sampled_control(n: usize, subsample: f64, rng: &mut TFastRng64) -> Vec<bool> {
    if subsample >= 1.0 {
        return vec![true; n];
    }
    let rate = f64::from(subsample as f32);
    (0..n).map(|_| rng.gen_rand_real1() < rate).collect()
}

/// `GetSingleProbability` (`mvs.cpp:17`):
/// `der > threshold ? 1.0 : der / threshold`.
#[inline]
fn single_probability(derivative_abs: f64, threshold: f64) -> f64 {
    if derivative_abs > threshold {
        1.0
    } else if threshold > 0.0 {
        derivative_abs / threshold
    } else {
        0.0
    }
}

/// `TMvsSampler::CalculateThreshold` (`mvs.cpp:81-118`) — a recursive
/// quickselect-style partition over the per-object gradient magnitudes that
/// finds the MVS threshold for one block. `candidates` is the block's
/// `sqrt(lambda + grad^2)` values; the routine mutates the slice in place
/// (matching upstream's `std::partition` on the iterator range). Returns the
/// threshold for the block.
fn calculate_threshold(
    candidates: &mut [f64],
    sum_of_small_current: f64,
    number_of_large_current: f64,
    sample_size: f64,
) -> f64 {
    let threshold = match candidates.first() {
        Some(&t) => t,
        None => return 0.0,
    };
    // std::partition: [begin, middle_begin) = (< threshold);
    // [middle_begin, middle_end) = (== threshold); [middle_end, end) = (> ...).
    let mut small: Vec<f64> = Vec::with_capacity(candidates.len());
    let mut middle: Vec<f64> = Vec::new();
    let mut large: Vec<f64> = Vec::new();
    for &c in candidates.iter() {
        if c < threshold {
            small.push(c);
        } else if c <= threshold {
            middle.push(c);
        } else {
            large.push(c);
        }
    }
    let sum_of_small_update = sum_f64(&small);
    let number_of_large_update = large.len() as f64;
    let number_of_middle = middle.len() as f64;
    let sum_of_middle = number_of_middle * threshold;

    let estimated_sample_size = if threshold != 0.0 {
        (sum_of_small_current + sum_of_small_update) / threshold
            + number_of_large_current
            + number_of_large_update
            + number_of_middle
    } else {
        f64::INFINITY
    };

    if estimated_sample_size > sample_size {
        if !large.is_empty() {
            // middle_end != candidates_end -> recurse on the large part.
            let next_small = sum_of_small_current + sum_of_middle + sum_of_small_update;
            calculate_threshold(&mut large, next_small, number_of_large_current, sample_size)
        } else {
            let denom = sample_size - number_of_large_current;
            if denom != 0.0 {
                (sum_of_small_current + sum_of_small_update + sum_of_middle) / denom
            } else {
                threshold
            }
        }
    } else if !small.is_empty() {
        // middle_begin != candidates_begin -> recurse on the small part.
        let next_large = number_of_large_current + number_of_large_update + number_of_middle;
        calculate_threshold(&mut small, sum_of_small_current, next_large, sample_size)
    } else {
        let denom =
            sample_size - number_of_large_current - number_of_middle - number_of_large_update;
        if denom != 0.0 {
            sum_of_small_current / denom
        } else {
            threshold
        }
    }
}

/// `TMvsSampler::GenSampleWeights` (`mvs.cpp:120-224`) for the single-dimension
/// (RMSE/Logloss) plain-boosting CPU path. `derivatives[i]` is the per-object
/// weighted first derivative (`fold->BodyTailArr[0].WeightedDerivatives[0]`);
/// `lambda` is `GetLambda(...)` (the squared mean gradient magnitude on iter 0,
/// or the squared previous-tree mean leaf L2 norm). Returns the per-object
/// `SampleWeights`. `sample_rate == 1.0` short-circuits to all-`1.0`.
fn mvs_sample_weights(
    derivatives: &[f64],
    lambda: f64,
    sample_rate: f64,
    rng: &mut TFastRng64,
) -> Vec<f64> {
    let n = derivatives.len();
    if sample_rate >= 1.0 {
        return vec![1.0; n];
    }
    // `TMvsSampler::SampleRate` is an `f32` (mvs.h:47); the block sample-size
    // target `SampleRate * blockSize` and the per-object probability comparison
    // use the f32-rounded rate promoted to `f64`.
    let sample_rate = f64::from(sample_rate as f32);
    let rand_seed = rng.gen_rand();
    let mut weights = vec![0.0_f64; n];
    let block_count = n.div_ceil(MVS_BLOCK_SIZE);
    for block_idx in 0..block_count {
        let mut block_rng = TFastRng64::from_seed(rand_seed.wrapping_add(block_idx as u64));
        block_rng.advance(10);
        let begin = block_idx * MVS_BLOCK_SIZE;
        let end = usize::min(begin + MVS_BLOCK_SIZE, n);
        let block = derivatives.get(begin..end).unwrap_or(&[]);
        let block_size = block.len();

        // thresholdCandidates[idx] = sqrt(lambda + der^2) over the block.
        let mut candidates: Vec<f64> = block.iter().map(|&d| (lambda + d * d).sqrt()).collect();
        let threshold = calculate_threshold(
            &mut candidates,
            0.0,
            0.0,
            sample_rate * block_size as f64,
        );

        for (offset, &der) in block.iter().enumerate() {
            let grad2 = der * der;
            let probability = single_probability((grad2 + lambda).sqrt(), threshold);
            let idx = begin + offset;
            if probability > f64::EPSILON {
                let weight = 1.0 / probability;
                let r = block_rng.gen_rand_real1();
                if let Some(slot) = weights.get_mut(idx) {
                    *slot = weight * f64::from(r < probability);
                }
            } else if let Some(slot) = weights.get_mut(idx) {
                *slot = 0.0;
            }
        }
    }
    weights
}

/// `CalculateMeanGradValue` (`mvs.cpp:37-65`): mean over objects of
/// `sqrt(der^2)` == `|der|` for the single-dimension case. Routed through the
/// sanctioned ordered `sum_f64` (D-05). Returns `0.0` for an empty input.
fn mean_grad_value(derivatives: &[f64]) -> f64 {
    if derivatives.is_empty() {
        return 0.0;
    }
    let mags: Vec<f64> = derivatives.iter().map(|&d| (d * d).sqrt()).collect();
    sum_f64(&mags) / derivatives.len() as f64
}

/// The MVS `lambda` (`mvs.cpp:67-79` `GetLambda`): on the FIRST tree (no prior
/// leaf values) it is `mean(|der|)^2`; on later trees it is
/// `mean_last_iter_leaf_l2_norm^2` (`CalculateLastIterMeanLeafValue`), supplied
/// by the caller via `prev_leaf_mean_l2`.
fn mvs_lambda(derivatives: &[f64], prev_leaf_mean_l2: Option<f64>) -> f64 {
    match prev_leaf_mean_l2 {
        Some(mean) => mean * mean,
        None => {
            let mean = mean_grad_value(derivatives);
            mean * mean
        }
    }
}

/// `CalculateLastIterMeanLeafValue` (`mvs.cpp:21-35`): mean over leaves of the
/// per-leaf L2 norm of the (single-dimension) leaf values. For the
/// one-dimensional case this is `mean(|leaf_value|)`. Used by the caller to feed
/// `prev_leaf_mean_l2` into [`bootstrap`] for trees after the first.
#[must_use]
pub fn last_iter_mean_leaf_value(leaf_values: &[f64]) -> f64 {
    if leaf_values.is_empty() {
        return 0.0;
    }
    let norms: Vec<f64> = leaf_values.iter().map(|&v| (v * v).sqrt()).collect();
    sum_f64(&norms) / leaf_values.len() as f64
}

/// One `Bootstrap()` call (`tensor_search_helpers.cpp:487`) for the CPU,
/// object-sampling, non-pairwise path.
///
/// `derivatives[i]` is the per-object WEIGHTED first derivative in the SAME
/// order the caller will reduce histograms in (MVS reads it; the other arms
/// ignore it). `prev_leaf_mean_l2` is `Some(mean L2 norm of the previous tree's
/// leaf values)` for trees after the first (MVS lambda), else `None`. `rng` is
/// the persistent, continuously-advancing training RNG.
///
/// # Errors
/// [`CbError::Degenerate`] for [`EBootstrapType::Poisson`] (unsupported on CPU,
/// mirroring upstream `bootstrap_options.cpp`).
pub fn bootstrap(
    bootstrap_type: EBootstrapType,
    derivatives: &[f64],
    subsample: f64,
    bagging_temperature: f32,
    prev_leaf_mean_l2: Option<f64>,
    rng: &mut TFastRng64,
) -> CbResult<BootstrapResult> {
    let n = derivatives.len();
    match bootstrap_type {
        EBootstrapType::No => Ok(BootstrapResult::identity(n)),
        EBootstrapType::Bayesian => {
            let sample_weights = generate_random_weights(n, bagging_temperature, rng);
            // BernoulliSampleRate == 1 for Bayesian -> control all true, no draw.
            Ok(BootstrapResult {
                sample_weights,
                control: vec![true; n],
            })
        }
        EBootstrapType::Bernoulli => {
            // SampleWeights all 1.0 (Fill); the subsample lives in the control.
            let control = set_sampled_control(n, subsample, rng);
            Ok(BootstrapResult {
                sample_weights: vec![1.0; n],
                control,
            })
        }
        EBootstrapType::Mvs => {
            let lambda = mvs_lambda(derivatives, prev_leaf_mean_l2);
            let sample_weights = mvs_sample_weights(derivatives, lambda, subsample, rng);
            if subsample < 1.0 {
                // MVS uses `performRandomChoice=false` (calc_score_cache.cpp:752),
                // so its `sampledDocs->Sample` keeps the full doc set and the score
                // path consumes two additional `GenRand()` draws on the main stream
                // relative to the Bernoulli/`SetSampledControl` path. Reproduce them
                // so the next tree's MVS reseed lands on the correct RNG phase
                // (verified end-to-end against the MVS oracle). With subsample==1.0
                // MVS draws nothing at all (the early-return identity path).
                rng.gen_rand();
                rng.gen_rand();
            }
            // performRandomChoice = false -> control = weight > eps
            // (SetControlNoZeroWeighted, calc_score_cache.cpp:1203-1211).
            let control: Vec<bool> = sample_weights
                .iter()
                .map(|&w| w > f64::from(f32::EPSILON))
                .collect();
            Ok(BootstrapResult {
                sample_weights,
                control,
            })
        }
        EBootstrapType::Poisson => Err(CbError::Degenerate(
            "poisson bootstrap is not supported on CPU".to_owned(),
        )),
    }
}
