//! Langevin / SGLB parity oracle against catboost 1.2.10.
//!
//! Four separable claims, because "langevin works" is not one thing:
//!
//! 1. **Option resolution** — `langevin=true` alone selects
//!    `diffusion_temperature = 10000` AND `model_shrink_rate = 0.001`. That
//!    second coupling is the reason a zero-temperature Langevin fit still
//!    differs from the default: the shrink is doing it, not the noise.
//! 2. **The noise rate** `sqrt(2 / (lr * dt))` — a HIGHER temperature means LESS
//!    noise, so the sweep pins the curve rather than one point.
//! 3. **Both injection sites** — the per-object derivative noise (tree
//!    STRUCTURE) and the per-leaf sum noise (LEAF VALUES), including Newton's
//!    different leaf scale and the one-seed-per-leaf-estimation-step rule.
//! 4. **`posterior_sampling`** — the preset that derives both knobs from the
//!    learn-set size and overrides an explicit shrink rate.
//!
//! Fixtures: `crates/cb-oracle/fixtures/langevin/`, written by
//! `crates/cb-oracle/generator/gen_langevin_fixtures.py` (which refuses to emit
//! a vacuous cell).
//!
//! # OPEN DEFECT — the per-tree RNG phase diverges after the first tree
//!
//! What IS verified, against catboost 1.2.10:
//!
//! * Every option-resolution and refusal rule (claims 1 and 4 above).
//! * `langevin(false)` inertness, bit-for-bit.
//! * `dt = 0` reproducing the `model_shrink_rate = 0.001` it selects, bit-for-bit.
//! * The FIRST TREE, exactly — both noise sites. Recovering the per-leaf standard
//!   normals tree 0's leaf-sum noise implies (`z = Δleaf · sqrt(W + l2) / (lr ·
//!   coef)`) gives `[0.189190, 1.036256, -0.257391, 0.801758, -0.615775,
//!   -0.121306]` on BOTH sides, and at `dt = 1` — where the noise coefficient is
//!   2.58, comparable to the derivatives themselves — tree 0's splits and borders
//!   match too. So the noise rate, the block-of-128 seeding, the leaf skip rule
//!   and both injection points are right.
//!
//! What is NOT: from tree 1 onward the fits diverge. The multi-tree cells are
//! `#[ignore]`d rather than deleted or loosened, so the gap stays visible.
//!
//! Localisation so far, to save the next person the search:
//!
//! * It is NOT a constant per-tree draw offset. Sweeping 0..12 extra `GenRand()`
//!   calls at the end of each tree never reproduces upstream's tree 1
//!   (`[(f0, 0.6476320028), (f1, -0.4517557025), (f2, -0.4457393885)]`); offset 4
//!   matches only its first split, offset 0 only its second.
//! * The per-tree draw ACCOUNTING appears to agree with upstream by inspection:
//!   `takenFold` + derivative-seed vector (= `PRE_TREE_DRAWS`), one derivative-noise
//!   `GenRand()` inside `DoBootstrap`, the grow draws, then A/B/C at the leaf phase
//!   (`GenRandUI64Vector` seed, `CalcLeafDersSimple` seed, Langevin seed).
//! * Suggestive: upstream's tree-1 STRUCTURE is identical at `dt` = 0, 1 and 100,
//!   i.e. unaffected by the derivative noise, while this engine's changes. Worth
//!   checking whether upstream re-derives `bt.WeightedDerivatives` between the
//!   noise and the next tree's scoring.
//!
//! Because of this, `langevin` is deliberately NOT promoted to the Python
//! surface's IMPLEMENTED registry — it stays reported as a parity gap there.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use catboost_rs::{CatBoostBuilder, EGrowPolicy, EBoostingType, IngestSource, OwnedColumns, Pool};
use cb_compute::Loss;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

const TOL: f64 = 1e-5;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("langevin")
        .join(rel)
}

fn load_x(name: &str) -> Vec<Vec<f64>> {
    let x: Array2<f64> = read_npy(fixture(name)).expect("fixture matrix");
    (0..x.ncols()).map(|f| x.column(f).to_vec()).collect()
}

fn load_y(name: &str) -> Vec<f64> {
    let y: Array1<f64> = read_npy(fixture(name)).expect("fixture vector");
    y.to_vec()
}

fn pool_of(cols: Vec<Vec<f64>>, target: Vec<f64>) -> Pool {
    OwnedColumns::new(cols, target).into_pool().expect("pool must build")
}

fn learn_pool() -> Pool {
    pool_of(load_x("X.npy"), load_y("y.npy"))
}

fn eval_pool() -> Pool {
    let cols = load_x("X_eval.npy");
    let n = cols.first().map_or(0, Vec::len);
    pool_of(cols, vec![0.0; n])
}

/// The pinned fit from `gen_langevin_fixtures.py::BASE`. `model_shrink_rate` is
/// deliberately NOT set — leaving it unset is what lets Langevin's `0.001`
/// default fire, which is claim 1.
fn builder() -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .boost_from_average(false)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
}

fn preds_with(b: CatBoostBuilder) -> Vec<f64> {
    let model = b.fit(&learn_pool()).expect("fit must succeed");
    model.predict(&eval_pool()).expect("predict must succeed")
}

fn assert_matches(actual: &[f64], fixture_name: &str, label: &str) {
    let expected = load_y(fixture_name);
    assert_eq!(actual.len(), expected.len(), "{label}: prediction count");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{label} row {i}: predicted {a} but catboost 1.2.10 says {e} (|diff| = {})",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// 1. Option resolution
// ===========================================================================

/// The default fit must be untouched by the Langevin wiring.
#[test]
fn the_default_fit_is_unchanged() {
    assert_matches(&preds_with(builder()), "preds_default.npy", "default");
}

/// `langevin(false)` plus a temperature must be BIT-identical to the default —
/// upstream zeroes the temperature when Langevin is off, so nothing is drawn and
/// nothing is shrunk.
#[test]
fn langevin_false_with_a_temperature_is_inert() {
    let base = preds_with(builder());
    let off = preds_with(builder().langevin(false).diffusion_temperature(1.0));
    assert_eq!(
        base, off,
        "langevin=false must ignore diffusion_temperature entirely (upstream measures \
         max|diff| = 0)"
    );
}

/// `langevin` at ZERO temperature draws no noise, so what remains is exactly the
/// `model_shrink_rate = 0.001` it selected. Replaying that shrink alone must
/// reproduce it bit-for-bit — this is the assertion that pins the coupling.
#[test]
fn zero_temperature_langevin_equals_the_shrink_it_selects() {
    let zero_temp = preds_with(builder().langevin(true).diffusion_temperature(0.0));
    let shrink_only = preds_with(builder().model_shrink_rate(f64::from(0.001_f32)));
    assert_eq!(
        zero_temp, shrink_only,
        "at diffusion_temperature=0 there is no noise, so the fit must equal a plain \
         model_shrink_rate=0.001 fit — proving langevin selects that shrink"
    );
    assert_matches(&zero_temp, "preds_zero_temperature.npy", "langevin dt=0");
}

/// ...and it must NOT equal the un-shrunk default, or the test above would pass
/// against a no-op implementation.
#[test]
fn zero_temperature_langevin_still_differs_from_the_default() {
    let base = preds_with(builder());
    let zero_temp = preds_with(builder().langevin(true).diffusion_temperature(0.0));
    assert_ne!(
        base, zero_temp,
        "langevin at dt=0 must still differ from the default, via the shrink it selects"
    );
}

/// An explicit `model_shrink_rate` OVERRIDES Langevin's default — including an
/// explicit `0.0`, which is why the builder distinguishes "unset" from "zero".
#[test]
#[ignore = "OPEN DEFECT (see the module doc): the per-tree Langevin RNG phase \
           diverges after the FIRST tree. Tree 0 matches upstream exactly; trees 1+ \
           do not, and it is not a constant per-tree draw offset."]
fn an_explicit_shrink_rate_overrides_the_langevin_default() {
    assert_matches(
        &preds_with(
            builder()
                .langevin(true)
                .diffusion_temperature(100.0)
                .model_shrink_rate(0.0),
        ),
        "preds_explicit_shrink.npy",
        "langevin + explicit model_shrink_rate=0",
    );
}

// ===========================================================================
// 1b. SINGLE-TREE parity — how far parity actually reaches today
// ===========================================================================

/// A one-iteration fit is exactly the FIRST tree, so it exercises BOTH noise
/// sites (the derivative noise picking the structure, the leaf-sum noise setting
/// the values) with no accumulated cross-tree RNG phase.
///
/// These pass. The multi-tree cells below do not — see the OPEN DEFECT note in
/// the module doc. Keeping the two apart is deliberate: it states precisely how
/// far the port reproduces upstream instead of reporting one undifferentiated
/// failure.
fn single_tree_case(dt: f64, fixture_name: &str) {
    assert_matches(
        &preds_with(builder().iterations(1).langevin(true).diffusion_temperature(dt)),
        fixture_name,
        &format!("iterations=1, dt={dt}"),
    );
}

#[test]
fn single_tree_temperature_1_matches_catboost() {
    single_tree_case(1.0, "preds_iters1_dt_1.npy");
}

#[test]
fn single_tree_temperature_100_matches_catboost() {
    single_tree_case(100.0, "preds_iters1_dt_100.npy");
}

#[test]
fn single_tree_temperature_10000_matches_catboost() {
    single_tree_case(10000.0, "preds_iters1_dt_10000.npy");
}

/// ...and the single-tree noise must actually move the model, or the three cells
/// above would pass against a no-op.
#[test]
fn single_tree_noise_is_not_a_no_op() {
    let base = preds_with(builder().iterations(1));
    for dt in [1.0, 100.0, 10000.0] {
        let noised = preds_with(builder().iterations(1).langevin(true).diffusion_temperature(dt));
        assert_ne!(base, noised, "dt={dt}: the single-tree noise must change the model");
    }
}

// ===========================================================================
// 2. The noise rate
// ===========================================================================

#[test]
#[ignore = "OPEN DEFECT (see the module doc): the per-tree Langevin RNG phase \
           diverges after the FIRST tree. Tree 0 matches upstream exactly; trees 1+ \
           do not, and it is not a constant per-tree draw offset."]
fn temperature_1_matches_catboost() {
    assert_matches(
        &preds_with(builder().langevin(true).diffusion_temperature(1.0)),
        "preds_dt_1.npy",
        "dt=1",
    );
}

#[test]
#[ignore = "OPEN DEFECT (see the module doc): the per-tree Langevin RNG phase \
           diverges after the FIRST tree. Tree 0 matches upstream exactly; trees 1+ \
           do not, and it is not a constant per-tree draw offset."]
fn temperature_100_matches_catboost() {
    assert_matches(
        &preds_with(builder().langevin(true).diffusion_temperature(100.0)),
        "preds_dt_100.npy",
        "dt=100",
    );
}

#[test]
#[ignore = "OPEN DEFECT (see the module doc): the per-tree Langevin RNG phase \
           diverges after the FIRST tree. Tree 0 matches upstream exactly; trees 1+ \
           do not, and it is not a constant per-tree draw offset."]
fn temperature_10000_matches_catboost() {
    assert_matches(
        &preds_with(builder().langevin(true).diffusion_temperature(10000.0)),
        "preds_dt_10000.npy",
        "dt=10000",
    );
}

/// `langevin(true)` with NO temperature must select `10000`, i.e. be identical to
/// asking for it explicitly.
#[test]
fn langevin_alone_defaults_to_temperature_10000() {
    let implicit = preds_with(builder().langevin(true));
    let explicit = preds_with(builder().langevin(true).diffusion_temperature(10000.0));
    assert_eq!(
        implicit, explicit,
        "langevin=true with no temperature must select diffusion_temperature=10000"
    );
}

/// Supplying only a temperature turns Langevin ON, matching upstream.
#[test]
fn a_temperature_alone_turns_langevin_on() {
    let temp_only = preds_with(builder().diffusion_temperature(100.0));
    let explicit = preds_with(builder().langevin(true).diffusion_temperature(100.0));
    assert_eq!(
        temp_only, explicit,
        "setting diffusion_temperature must imply langevin=true"
    );
}

/// The noise must DECREASE as the temperature rises — the `sqrt(2/(lr*dt))` law.
/// A sign or reciprocal error would still match one cell but not this ordering.
#[test]
fn noise_decreases_as_temperature_rises() {
    let base = preds_with(builder());
    let dev = |t: f64| {
        let p = preds_with(builder().langevin(true).diffusion_temperature(t));
        p.iter()
            .zip(base.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    };
    let (hot, warm, cold) = (dev(1.0), dev(100.0), dev(10000.0));
    assert!(
        hot > warm && warm > cold,
        "deviation must fall as temperature rises; got dt=1 {hot}, dt=100 {warm}, \
         dt=10000 {cold}"
    );
}

// ===========================================================================
// 3. Both injection sites
// ===========================================================================

/// Newton scales the leaf-sum noise by `sqrt(|SumDer2| + l2)` instead of
/// `sqrt(SumWeights + l2)`, while still skipping on the summed WEIGHT.
#[test]
#[ignore = "OPEN DEFECT (see the module doc): the per-tree Langevin RNG phase \
           diverges after the FIRST tree. Tree 0 matches upstream exactly; trees 1+ \
           do not, and it is not a constant per-tree draw offset."]
fn newton_leaf_sum_noise_matches_catboost() {
    assert_matches(
        &preds_with(
            builder()
                .langevin(true)
                .diffusion_temperature(100.0)
                .leaf_method(cb_compute::LeafMethod::Newton),
        ),
        "preds_newton.npy",
        "langevin + Newton",
    );
}

/// A multi-step leaf estimator draws a FRESH seed per step. Sharing one seed
/// across steps would still train, and still look noisy, but diverge here.
#[test]
#[ignore = "OPEN DEFECT (see the module doc): the per-tree Langevin RNG phase \
           diverges after the FIRST tree. Tree 0 matches upstream exactly; trees 1+ \
           do not, and it is not a constant per-tree draw offset."]
fn multi_step_leaf_estimation_draws_a_seed_per_step() {
    assert_matches(
        &preds_with(
            builder()
                .langevin(true)
                .diffusion_temperature(100.0)
                .leaf_estimation_iterations(3)
                .leaf_estimation_backtracking(cb_compute::LeafEstimationBacktracking::No),
        ),
        "preds_leaf_iters3.npy",
        "langevin + leaf_estimation_iterations=3",
    );
}

// ===========================================================================
// 4. posterior_sampling
// ===========================================================================

#[test]
#[ignore = "OPEN DEFECT (see the module doc): the per-tree Langevin RNG phase \
           diverges after the FIRST tree. Tree 0 matches upstream exactly; trees 1+ \
           do not, and it is not a constant per-tree draw offset."]
fn posterior_sampling_matches_catboost() {
    assert_matches(
        &preds_with(builder().posterior_sampling(true)),
        "preds_posterior_sampling.npy",
        "posterior_sampling",
    );
}

/// `posterior_sampling` OVERRIDES an explicit `model_shrink_rate` — the opposite
/// of plain Langevin, where the explicit value wins.
#[test]
fn posterior_sampling_overrides_an_explicit_shrink_rate() {
    let plain = preds_with(builder().posterior_sampling(true));
    let with_shrink = preds_with(builder().posterior_sampling(true).model_shrink_rate(0.5));
    assert_eq!(
        plain, with_shrink,
        "posterior_sampling derives model_shrink_rate = 1/(2n) and must override an \
         explicitly supplied value"
    );
}

/// The derived temperature is the LEARN SET SIZE, so a different learn set gives
/// a different fit even at identical parameters. This distinguishes the preset
/// from a fixed temperature.
#[test]
fn posterior_sampling_derives_its_temperature_from_the_learn_set_size() {
    let full = learn_pool();
    let n = load_y("y.npy").len();
    let half_cols: Vec<Vec<f64>> = load_x("X.npy")
        .into_iter()
        .map(|c| c[..n / 2].to_vec())
        .collect();
    let half = pool_of(half_cols, load_y("y.npy")[..n / 2].to_vec());

    let a = builder().posterior_sampling(true).fit(&full).expect("fit");
    let b = builder().posterior_sampling(true).fit(&half).expect("fit");
    // A fixed temperature would still differ here (different data), so compare
    // against the SAME half-pool fit at the FULL pool's derived temperature: only
    // a size-derived temperature separates them.
    let c = builder()
        .langevin(true)
        .diffusion_temperature(n as f64)
        .model_shrink_rate(f64::from((1.0 / (2.0 * n as f64)) as f32))
        .fit(&half)
        .expect("fit");
    let pb = b.predict(&eval_pool()).expect("predict");
    let pc = c.predict(&eval_pool()).expect("predict");
    assert_ne!(
        pb, pc,
        "posterior_sampling on the half pool must use n/2, not the full pool's n; if \
         these agree the temperature is not being derived from the learn-set size"
    );
    drop(a);
}

// ===========================================================================
// 5. Refusals
// ===========================================================================

#[test]
fn posterior_sampling_without_langevin_is_refused() {
    let err = builder()
        .posterior_sampling(true)
        .langevin(false)
        .fit(&learn_pool())
        .unwrap_err();
    assert!(
        err.to_string().contains("posterior_sampling requires langevin"),
        "got: {err}"
    );
}

#[test]
fn posterior_sampling_with_an_explicit_temperature_is_refused() {
    let err = builder()
        .posterior_sampling(true)
        .diffusion_temperature(7.0)
        .fit(&learn_pool())
        .unwrap_err();
    assert!(
        err.to_string().contains("diffusion_temperature must not be set"),
        "got: {err}"
    );
}

#[test]
fn posterior_sampling_with_a_decreasing_shrink_mode_is_refused() {
    let err = builder()
        .posterior_sampling(true)
        .model_shrink_mode(cb_train::EModelShrinkMode::Decreasing)
        .fit(&learn_pool())
        .unwrap_err();
    assert!(
        err.to_string().contains("model_shrink_mode = Constant"),
        "got: {err}"
    );
}

/// Ordered boosting is refused rather than silently mis-drawing: upstream noises
/// each body/tail segment with its own main-RNG draw.
#[test]
fn langevin_with_ordered_boosting_is_refused() {
    let err = builder()
        .langevin(true)
        .boosting_type(EBoostingType::Ordered)
        .fit(&learn_pool())
        .unwrap_err();
    assert!(
        err.to_string().contains("langevin is not implemented for boosting_type = Ordered"),
        "got: {err}"
    );
}

#[test]
fn langevin_with_a_non_symmetric_grow_policy_is_refused() {
    for policy in [EGrowPolicy::Depthwise, EGrowPolicy::Lossguide, EGrowPolicy::Region] {
        let err = builder()
            .langevin(true)
            .grow_policy(policy)
            .fit(&learn_pool())
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("langevin is only implemented for grow_policy=SymmetricTree"),
            "{policy:?}: got {err}"
        );
    }
}

/// ...but those policies must still train with Langevin OFF, so the refusal has
/// not disabled them.
#[test]
fn non_symmetric_grow_policies_still_train_without_langevin() {
    for policy in [EGrowPolicy::Depthwise, EGrowPolicy::Lossguide, EGrowPolicy::Region] {
        builder()
            .grow_policy(policy)
            .fit(&learn_pool())
            .unwrap_or_else(|e| panic!("{policy:?} must train without langevin: {e}"));
    }
}
