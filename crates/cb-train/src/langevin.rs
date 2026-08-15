//! Langevin / SGLB gradient noise — a transcription of upstream's
//! `catboost/private/libs/algo_helpers/langevin_utils.cpp` (v1.2.10).
//!
//! Stochastic Gradient Langevin Boosting perturbs the gradients with seeded
//! Gaussian noise so the boosting trajectory samples from a posterior rather than
//! descending to a point estimate. Upstream injects that noise at TWO separate
//! places, and both are needed for parity:
//!
//! 1. [`add_noise_to_derivatives`] — the per-object weighted derivatives, applied
//!    inside `DoBootstrap` right after the bootstrap sample is drawn
//!    (`greedy_tensor_search.cpp:760`). These derivatives drive SPLIT SCORING, so
//!    this noise changes the tree STRUCTURE.
//! 2. [`add_noise_to_leaf_der_sums`] / [`add_noise_to_leaf_newton_sums`] — the
//!    per-leaf derivative sums, applied inside the leaf-estimation loop
//!    (`approx_calcer.cpp:768`). This noise changes the LEAF VALUES of a tree
//!    whose structure is already fixed.
//!
//! # The block-seeded stream (the part that is easy to get wrong)
//!
//! The object-derivative noise does NOT reseed per element. It blocks the range
//! by a COMPILE-TIME CONSTANT ([`LANGEVIN_BLOCK_SIZE`] = upstream's
//! `CB_THREAD_LIMIT` = 128, `private/libs/options/restrictions.h:59`), seeds one
//! `TFastRng64` per block from `randomSeed + blockIdx`, and draws sequentially
//! within the block in index order:
//!
//! ```text
//! for blockIdx in 0 .. ceil(n / 128):
//!     rng = TFastRng64(randomSeed + blockIdx)
//!     for idx in [blockIdx*128, min((blockIdx+1)*128, n)):
//!         der[idx] += coef * std_normal(rng)
//! ```
//!
//! Because the block size is a CONSTANT and not the thread count, the result is
//! identical at any `thread_count` — which is exactly what makes upstream's
//! `thread_count` numerically inert, measured at 1/2/4/8/16 threads. Reproducing
//! it with a per-element reseed, or with blocks sized by the runtime thread
//! count, would give a different (and thread-dependent) model.
//!
//! > NOTE: the pre-existing DEVICE kernel `cb_backend::kernels::langevin` reseeds
//! > PER ELEMENT (`from_seed(rand_seed + i).advance(10)`), which is a different
//! > stream from this one. Its self-oracle only compares it against a CPU replica
//! > of its own rule, so the divergence is invisible there. The device grower
//! > declines Langevin for that reason; see `device_host_eligible`.
//!
//! # The noise rate
//!
//! `CalcLangevinNoiseRate(dt, lr) = sqrt(2.0 / lr / dt)` — note the noise
//! DECREASES as the diffusion temperature rises. Both arguments are `float`
//! upstream, so they are narrowed to `f32` here before the division; skipping
//! that narrowing shifts the coefficient in the 8th significant digit, which is
//! above this project's 1e-5 parity bar once it compounds over iterations.
//!
//! A `diffusion_temperature` of exactly `0` returns early from EVERY entry point:
//! no noise AND no RNG draws.

use cb_core::{std_normal, TFastRng64};

/// Upstream's `CB_THREAD_LIMIT` (`private/libs/options/restrictions.h:59`), used
/// as the BLOCK SIZE of `TSimpleIndexRangesGenerator(TIndexRange(n), blockSize)`.
///
/// This is a constant, NOT the thread count — see the module doc for why that
/// distinction is what makes the noise thread-invariant.
pub const LANGEVIN_BLOCK_SIZE: usize = 128;

/// `CalcLangevinNoiseRate(float diffusionTemperature, float learningRate)`
/// (`langevin_utils.cpp:16`): `sqrt(2.0 / learningRate / diffusionTemperature)`.
///
/// Both arguments are `float` upstream. They are narrowed to `f32` and widened
/// back before the division so the division operands are bit-identical to
/// upstream's; the division and `sqrt` themselves are `double`, matching the
/// `2.0` literal's type.
#[must_use]
pub fn langevin_noise_rate(diffusion_temperature: f64, learning_rate: f64) -> f64 {
    let dt = f64::from(diffusion_temperature as f32);
    let lr = f64::from(learning_rate as f32);
    (2.0 / lr / dt).sqrt()
}

/// `AddLangevinNoiseToDerivatives` (`langevin_utils.cpp:20`): add
/// `coef * std_normal` to every derivative, over the block-seeded stream
/// described in the module doc.
///
/// `derivatives` may be a DIMENSION-MAJOR buffer (`[d * n + i]`); upstream's
/// multi-dimensional overload re-blocks per dimension over the PER-OBJECT count,
/// so callers with `approx_dimension > 1` must call this once per dimension
/// slice rather than passing the flat buffer.
///
/// A `diffusion_temperature` of `0` is a no-op that consumes no randomness.
pub fn add_noise_to_derivatives(
    derivatives: &mut [f64],
    diffusion_temperature: f64,
    learning_rate: f64,
    random_seed: u64,
) {
    if diffusion_temperature == 0.0 {
        return;
    }
    let coef = langevin_noise_rate(diffusion_temperature, learning_rate);
    for (block_idx, block) in derivatives.chunks_mut(LANGEVIN_BLOCK_SIZE).enumerate() {
        // `TFastRng64 blockRng(randomSeed + blockIdx)` — wrapping matches the C++
        // `ui64` addition, which is defined to wrap.
        let mut rng = TFastRng64::from_seed(random_seed.wrapping_add(block_idx as u64));
        for der in block.iter_mut() {
            *der += coef * std_normal(&mut rng);
        }
    }
}

/// `AddLangevinNoiseToLeafDerivativesSum` (`langevin_utils.cpp:82`), the
/// `ELeavesEstimation::Gradient` variant.
///
/// ONE `TFastRng64` seeded from `random_seed` serves ALL leaves, drawn in leaf
/// order. A leaf whose summed weight is below `1e-9` is SKIPPED and takes NO
/// draw, so an empty leaf shifts every later leaf's noise — the skip is part of
/// the stream, not an optimization.
///
/// The per-leaf scale is `coef * sqrt(sum_weight + scaled_l2_regularizer)`.
pub fn add_noise_to_leaf_der_sums(
    sum_der: &mut [f64],
    sum_weights: &[f64],
    diffusion_temperature: f64,
    learning_rate: f64,
    scaled_l2_regularizer: f64,
    random_seed: u64,
) {
    if diffusion_temperature == 0.0 {
        return;
    }
    let coef = langevin_noise_rate(diffusion_temperature, learning_rate);
    let mut rng = TFastRng64::from_seed(random_seed);
    for (leaf, der) in sum_der.iter_mut().enumerate() {
        let weight = sum_weights.get(leaf).copied().unwrap_or(0.0);
        // `if (sum.SumWeights < 1e-9) continue;` — no draw for a skipped leaf.
        if weight < 1e-9 {
            continue;
        }
        let scaled_coef = coef * (weight + scaled_l2_regularizer).sqrt();
        *der += scaled_coef * std_normal(&mut rng);
    }
}

/// `AddLangevinNoiseToLeafNewtonSum` (`langevin_utils.cpp:117`), the
/// `ELeavesEstimation::Newton` variant.
///
/// Identical to [`add_noise_to_leaf_der_sums`] except that the per-leaf scale
/// uses the second derivative, `coef * sqrt(|sum_der2| + scaled_l2_regularizer)`.
/// The SKIP predicate still reads the summed WEIGHT, not `sum_der2`.
pub fn add_noise_to_leaf_newton_sums(
    sum_der: &mut [f64],
    sum_der2: &[f64],
    sum_weights: &[f64],
    diffusion_temperature: f64,
    learning_rate: f64,
    scaled_l2_regularizer: f64,
    random_seed: u64,
) {
    if diffusion_temperature == 0.0 {
        return;
    }
    let coef = langevin_noise_rate(diffusion_temperature, learning_rate);
    let mut rng = TFastRng64::from_seed(random_seed);
    for (leaf, der) in sum_der.iter_mut().enumerate() {
        let weight = sum_weights.get(leaf).copied().unwrap_or(0.0);
        if weight < 1e-9 {
            continue;
        }
        let der2 = sum_der2.get(leaf).copied().unwrap_or(0.0);
        let scaled_coef = coef * (der2.abs() + scaled_l2_regularizer).sqrt();
        *der += scaled_coef * std_normal(&mut rng);
    }
}

/// The `model_shrink_rate` upstream applies when `langevin` is on and the user
/// did NOT set one (`0.001`, stored as a `float` — `get_all_params` reports
/// `0.0010000000474974513`, i.e. `f32(0.001)` widened).
///
/// An explicit `model_shrink_rate` OVERRIDES this (measured: `langevin=True,
/// model_shrink_rate=0.5` resolves to `0.5`, and an explicit `0.0` resolves to
/// `0` and trains differently from the default). `posterior_sampling` is the
/// exception — it overrides even an explicit value, see
/// [`posterior_sampling_shrink_rate`].
#[must_use]
pub fn langevin_default_model_shrink_rate() -> f64 {
    f64::from(0.001_f32)
}

/// The `diffusion_temperature` upstream applies when `langevin` is on and the
/// user did NOT set one (`10000`).
#[must_use]
pub fn langevin_default_diffusion_temperature() -> f64 {
    10000.0
}

/// `posterior_sampling`'s diffusion temperature: the LEARN SET SIZE.
///
/// Measured against catboost 1.2.10 at n ∈ {50, 200, 777} — the resolved
/// `diffusion_temperature` is exactly `n` in every case.
#[must_use]
pub fn posterior_sampling_diffusion_temperature(learn_sample_count: usize) -> f64 {
    learn_sample_count as f64
}

/// `posterior_sampling`'s model shrink rate: `1 / (2n)`, stored as a `float`.
///
/// Measured at n ∈ {50, 200, 777}: `0.01`, `0.0025`, `0.000643500650767237` —
/// the last one is `f32(1/1554)`, which is why this narrows.
#[must_use]
pub fn posterior_sampling_shrink_rate(learn_sample_count: usize) -> f64 {
    f64::from((1.0 / (2.0 * learn_sample_count as f64)) as f32)
}

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file), mounted as a child module
// so `cargo test -p cb-train langevin` selects them.
#[cfg(test)]
#[path = "langevin_test.rs"]
mod tests;
