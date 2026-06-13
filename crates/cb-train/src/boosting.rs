//! Plain gradient-boosting loop (TRAIN-01) — drives [`crate::tree`] over the
//! generic `cb-compute` [`Runtime`] boundary to grow symmetric oblivious trees
//! with Gradient leaf estimation, oracle-locked to upstream catboost 1.2.10.
//!
//! # Source of truth
//!
//! `catboost/libs/train_lib/train_model.cpp` (boosting driver) +
//! `online_predictor.h` (leaf math):
//! - Starting approx (`CalcOptimumConstApprox`, Pitfall 2): for RMSE
//!   `boost_from_average=true` the starting approx is the target MEAN, stored as
//!   the model bias; for Logloss `boost_from_average=false` the starting approx
//!   is `0` (bias `0`).
//! - Per iteration: `compute_gradients(approx, target)` → grow one oblivious tree
//!   → Gradient leaf delta `CalcAverage(sumDer, sumWeight, scaledL2)` over each
//!   leaf's members (ordered `sum_f64`, D-05) → store `learning_rate * delta` as
//!   the leaf value → `approx[i] += leaf_value[leaf(i)]`.
//! - `leaf_estimation_iterations = 1` for this slice (auto-forced; Pitfall 5).
//!
//! # Parity discipline
//!
//! Every leaf SUM routes through `cb_core::sum_f64` (via
//! `cb_compute::reduce_leaf_stats`). The leaf values STORED already include the
//! `learning_rate` factor, matching the upstream `model.json` `leaf_values` the
//! oracle compares against. Degenerate inputs surface as [`CbError`]; no
//! `unwrap`/`expect`/raw float fold in production (deny-lints + D-08).

use cb_compute::{
    collect_leaf_residuals, exact_leaf_delta, gradient_leaf_delta, newton_leaf_delta,
    reduce_leaf_der2, reduce_leaf_stats, scale_l2_reg, score_st_dev, simple_leaf_delta, LeafMethod,
    Loss, Runtime, QUANTILE_ALPHA, QUANTILE_DELTA,
};
use cb_core::{sum_f64, CbError, CbResult, TFastRng64};

use crate::bootstrap::{bootstrap, last_iter_mean_leaf_value, EBootstrapType};
use crate::tree::{
    check_depth, greedy_tensor_search_oblivious_perturbed, FeatureMatrix, Perturbation, Split,
};

/// Per-iteration PRE-bootstrap draws on the persistent RNG (train.cpp:208,211):
/// the fold pick (`Rand.GenRand() % foldCount`) and the derivative-seed draw
/// (`Rand.GenRand()` feeding `GenRandUI64Vector`). Consumed only when sampling
/// is active so the bootstrap draws land on the correct RNG phase every tree.
const PRE_TREE_DRAWS: usize = 2;

/// Per-iteration POST-bootstrap draws BEYOND the `depth` per-level
/// `CalcScores` random-strength seed draws (greedy_tensor_search.cpp:884): the
/// depth loop evaluates `depth + 1` candidate levels (the final level finds no
/// improving split and breaks), so `CalcScores` draws one extra `GenRand()`.
/// Verified end-to-end against the Bernoulli oracle (post = depth + 1).
const POST_TREE_EXTRA_DRAWS: usize = 1;

/// Parameters for the plain boosting loop (the D-07 simplified isolating set).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoostParams {
    /// Which loss / objective (RMSE or Logloss).
    pub loss: Loss,
    /// Number of boosting iterations (trees).
    pub iterations: usize,
    /// Tree depth (number of splits per tree; `2^depth` leaves).
    pub depth: usize,
    /// Learning rate scaling every leaf delta.
    pub learning_rate: f64,
    /// L2 leaf regularization (`l2_leaf_reg`).
    pub l2_leaf_reg: f64,
    /// Split-score perturbation strength (`random_strength`, TRAIN-05). `0.0`
    /// disables the perturbation (no normal draws — the first-slice path);
    /// non-zero turns on the per-candidate `TRandomScore::GetInstance` normal
    /// draw over the persistent RNG.
    pub random_strength: f64,
    /// Whether to start from the per-loss optimum constant approx (the target
    /// mean for RMSE), stored as the model bias. `false` starts from `0`.
    pub boost_from_average: bool,
    /// Which leaf-estimation method computes the per-leaf deltas (TRAIN-03 /
    /// D-09). The first-slice path is [`LeafMethod::Gradient`].
    pub leaf_method: LeafMethod,
    /// Bootstrap / sampling type (TRAIN-04). The first-slice path is
    /// [`EBootstrapType::No`].
    pub bootstrap_type: EBootstrapType,
    /// Object subsample fraction (`subsample`), used by Bernoulli and MVS. `1.0`
    /// disables subsampling. Ignored by `No`/`Bayesian`.
    pub subsample: f64,
    /// Bayesian bagging temperature (`bagging_temperature`); `0.0` makes Bayesian
    /// weights all `1.0`. Ignored by the other types.
    pub bagging_temperature: f32,
    /// The training random seed seeding the persistent sampling RNG
    /// (`random_seed`). Only consumed when `bootstrap_type != No`.
    pub random_seed: u64,
}

/// One trained oblivious tree: the ordered splits and the per-leaf values
/// (already scaled by `learning_rate`, matching upstream `model.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct ObliviousTree {
    /// The ordered splits (feature + border) defining the symmetric structure.
    pub splits: Vec<Split>,
    /// Leaf values in canonical forward-bit-order, length `2^depth`.
    pub leaf_values: Vec<f64>,
}

/// A trained plain-boosted model: the boosting-order trees plus the starting
/// approx (`boost_from_average`) stored as the model bias.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// The oblivious trees in boosting (iteration) order.
    pub oblivious_trees: Vec<ObliviousTree>,
    /// The starting approx / model bias.
    pub bias: f64,
}

impl Model {
    /// Per-tree split borders flattened in tree order (for
    /// `compare_stage(Stage::Splits, …)`).
    #[must_use]
    pub fn split_borders(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|t| t.splits.iter().map(|s| s.border))
            .collect()
    }

    /// Per-tree leaf values flattened in tree order (for
    /// `compare_stage(Stage::LeafValues, …)`).
    #[must_use]
    pub fn leaf_values(&self) -> Vec<f64> {
        self.oblivious_trees
            .iter()
            .flat_map(|t| t.leaf_values.iter().copied())
            .collect()
    }
}

/// Compute the starting approx (and model bias): the target mean for RMSE with
/// `boost_from_average`, else `0` (Pitfall 2). The mean is folded through the
/// sanctioned `sum_f64` primitive (D-05).
fn starting_approx(params: &BoostParams, target: &[f64]) -> f64 {
    if params.boost_from_average && matches!(params.loss, Loss::Rmse) && !target.is_empty() {
        sum_f64(target) / target.len() as f64
    } else {
        0.0
    }
}

/// Compute the per-leaf deltas for the selected [`LeafMethod`] (TRAIN-03 / D-09).
///
/// Gradient/Newton/Simple are closed-form over each leaf's ordered reduced sums
/// (`cb_core::sum_f64` via `reduce_leaf_stats` / `reduce_leaf_der2`, D-05). Exact
/// takes the weighted median of each leaf's per-member residuals
/// (`target - approx`) via the quantile-style optimum. `weighted_der1[i]` is
/// `der1*weight`; `der2[i]` the per-object second derivative (weighted below for
/// the Newton sum); `approx`/`target` the running approximant/labels.
#[allow(clippy::too_many_arguments)]
fn compute_leaf_deltas(
    method: LeafMethod,
    leaf_of: &[usize],
    weighted_der1: &[f64],
    der2: &[f64],
    weights: &[f64],
    approx: &[f64],
    target: &[f64],
    scaled_l2: f64,
    n_leaves: usize,
) -> Vec<f64> {
    match method {
        LeafMethod::Gradient => {
            let stats = reduce_leaf_stats(leaf_of, weighted_der1, weights, n_leaves);
            stats
                .iter()
                .map(|s| gradient_leaf_delta(s.sum_weighted_delta, s.sum_weight, scaled_l2))
                .collect()
        }
        LeafMethod::Simple => {
            let stats = reduce_leaf_stats(leaf_of, weighted_der1, weights, n_leaves);
            stats
                .iter()
                .map(|s| simple_leaf_delta(s.sum_weighted_delta, s.sum_weight, scaled_l2))
                .collect()
        }
        LeafMethod::Newton => {
            let stats = reduce_leaf_stats(leaf_of, weighted_der1, weights, n_leaves);
            // Newton needs Σ der2*weight per leaf; build the weighted-der2 column
            // (elementwise product the host folds), then reduce ordered (D-05).
            let weighted_der2: Vec<f64> = der2
                .iter()
                .zip(weights.iter())
                .map(|(&d, &w)| d * w)
                .collect();
            let sum_der2 = reduce_leaf_der2(leaf_of, &weighted_der2, n_leaves);
            stats
                .iter()
                .zip(sum_der2.iter())
                .map(|(s, &d2)| newton_leaf_delta(s.sum_weighted_delta, d2, scaled_l2))
                .collect()
        }
        LeafMethod::Exact => {
            // Exact: per-leaf weighted median of residuals r_i = target_i -
            // approx_i (MAE / Quantile alpha=0.5, delta=1e-6). scaled_l2 is unused
            // (Exact has no L2 term — it is a rank statistic, not an average).
            let residuals: Vec<f64> = approx
                .iter()
                .zip(target.iter())
                .map(|(&a, &t)| t - a)
                .collect();
            let members = collect_leaf_residuals(leaf_of, &residuals, weights, n_leaves);
            members
                .iter()
                .map(|(r, w)| exact_leaf_delta(r, w, QUANTILE_ALPHA, QUANTILE_DELTA))
                .collect()
        }
    }
}

/// Train a plain-boosted oblivious-tree model over the generic runtime `R`.
///
/// `feature_values[f]` is float feature `f`'s per-object `f32` column;
/// `feature_borders[f]` its ascending candidate borders (the model's float-feature
/// borders). `target`/`weights` are per-object; `staged_out`, when `Some`, is
/// filled with the per-iteration staged approximants (flat, `iterations * n`).
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `params.depth > MAX_DEPTH`.
/// - [`CbError::Degenerate`] on an empty dataset or a level with no candidate
///   split.
/// - Any error the runtime's `compute_gradients` surfaces.
pub fn train<R: Runtime>(
    runtime: &R,
    feature_values: &[Vec<f32>],
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
    params: &BoostParams,
    mut staged_out: Option<&mut Vec<f64>>,
) -> CbResult<Model> {
    check_depth(params.depth)?;

    let n = target.len();
    if n == 0 {
        return Err(CbError::Degenerate("empty target".to_owned()));
    }

    // Per-object weights: default to 1.0 when no weights are supplied.
    let weights: Vec<f64> = if weights.is_empty() {
        vec![1.0; n]
    } else {
        weights.to_vec()
    };
    let sum_all_weights = sum_f64(&weights);

    let bias = starting_approx(params, target);
    let mut approx = vec![bias; n];

    let matrix = FeatureMatrix {
        feature_values,
        feature_borders,
    };

    let n_leaves = 1usize << params.depth;
    let mut trees: Vec<ObliviousTree> = Vec::with_capacity(params.iterations);

    // Persistent, continuously-advancing sampling RNG (`LearnProgress->Rand`,
    // seeded `random_seed`). Only consumed when bootstrap_type != No (Bayesian /
    // Bernoulli / MVS). The draw stream is NOT reseeded per tree (Pitfall 4).
    //
    // The bootstrap draws are NOT the only consumers of the persistent RNG:
    // upstream's per-iteration boosting body advances `LearnProgress->Rand` in a
    // FIXED pattern around each tree's `DoBootstrap` (train.cpp:206-243,
    // greedy_tensor_search.cpp:884,1916). Reproducing the draw ORDER (the parity
    // contract) requires consuming those non-bootstrap draws in the exact same
    // sequence so the bootstrap draws land on the correct RNG state every tree:
    //   * PRE-bootstrap, per iteration (train.cpp:208,211): `Rand.GenRand()`
    //     (fold pick `% foldCount`) + `Rand.GenRand()` (seed for
    //     `GenRandUI64Vector`) = [`PRE_TREE_DRAWS`] draws.
    //   * POST-bootstrap, per depth level (greedy_tensor_search.cpp:884):
    //     `CalcScores` draws ONE `Rand.GenRand()` per level (the
    //     random-strength seed, consumed even at `random_strength=0`) = `depth`
    //     draws per tree.
    let mut rng = TFastRng64::from_seed(params.random_seed);
    // The persistent RNG is consumed when EITHER sampling is active (bootstrap !=
    // No) OR the `random_strength` perturbation is on. With perturbation the
    // per-level `randSeed` draw and the `SelectBestCandidate` normal draws are
    // consumed INLINE by the perturbed tree search (in exact upstream order), so
    // the bulk POST per-level draws must NOT be applied in that case.
    let perturb_active = params.random_strength != 0.0;
    let draws_active = !matches!(params.bootstrap_type, EBootstrapType::No) || perturb_active;
    // MVS lambda for trees after the first uses the previous tree's mean leaf L2
    // norm (`CalculateLastIterMeanLeafValue`); `None` on the first tree.
    let mut prev_leaf_mean_l2: Option<f64> = None;

    for iter in 0..params.iterations {
        // 1. Per-object derivatives (UN-reduced; D-02) via the runtime kernel.
        let ders = runtime.compute_gradients(params.loss, &approx, target)?;

        // Weighted gradient contribution per object: der1 * weight (the
        // histogram-scatter elementwise product; the host reduces it ordered).
        let weighted_der1: Vec<f64> = ders
            .der1
            .iter()
            .zip(weights.iter())
            .map(|(&d, &w)| d * w)
            .collect();

        // 1a. PRE-bootstrap per-iteration draws (train.cpp:208,211): keep the RNG
        //     phase-aligned with upstream before the per-tree Bootstrap.
        if draws_active {
            for _ in 0..PRE_TREE_DRAWS {
                rng.gen_rand();
            }
        }

        // 1b. Bootstrap / sampling (TRAIN-04): once per tree, on the continuous
        //     RNG. MVS reads the weighted derivatives; the others ignore them.
        let sampled = bootstrap(
            params.bootstrap_type,
            &weighted_der1,
            params.subsample,
            params.bagging_temperature,
            prev_leaf_mean_l2,
            &mut rng,
        )?;

        // The SAMPLE WEIGHTS and CONTROL mask affect ONLY the SPLIT SCORING
        // (the `sampledDocs` histogram path); LEAF VALUES are estimated on the
        // FULL, UN-sampled AveragingFold derivatives (verified against upstream:
        // Bayesian/MVS sample weights never enter `CalcLeafValues`). So:
        //   * SCORE path: der1*weight*sampleWeight, restricted to control-true
        //     objects (zero score weight excludes an object from the ordered
        //     histogram reduction, exactly as `sampledDocs` drops it).
        //   * LEAF path: the raw weighted_der1 / weights (no sampling) —
        //     unchanged from the first slice.
        let score_weighted_der1: Vec<f64> = weighted_der1
            .iter()
            .zip(sampled.sample_weights.iter())
            .zip(sampled.control.iter())
            .map(|((&d, &sw), &c)| if c { d * sw } else { 0.0 })
            .collect();
        let score_weights: Vec<f64> = weights
            .iter()
            .zip(sampled.sample_weights.iter())
            .zip(sampled.control.iter())
            .map(|((&w, &sw), &c)| if c { w * sw } else { 0.0 })
            .collect();

        // 2. Grow one oblivious tree using the L2 split score over the ordered
        //    leaf-stat reduction (sampled subset / sample-weighted). When
        //    `random_strength != 0`, the per-candidate `TRandomScore` normal
        //    perturbation is drawn from the persistent RNG in upstream order
        //    (`scoreStDev = random_strength * derivativesStDevFromZero *
        //    modelSizeMultiplier`, `modelLength = iter * learning_rate`); the
        //    perturbation uses the SCORE-path weighted derivatives (the same fold
        //    `derivativesStDevFromZero` is computed over upstream).
        let scaled_l2 = scale_l2_reg(params.l2_leaf_reg, sum_all_weights, n);
        let perturb = if perturb_active {
            let model_length = iter as f64 * params.learning_rate;
            let std_dev = score_st_dev(params.random_strength, &score_weighted_der1, model_length);
            Some(Perturbation {
                rng: &mut rng,
                score_st_dev: std_dev,
            })
        } else {
            None
        };
        let grown = greedy_tensor_search_oblivious_perturbed(
            &matrix,
            &score_weighted_der1,
            &score_weights,
            scaled_l2,
            params.depth,
            n,
            perturb,
        )?;

        // 3. Leaf values via the selected estimation method (TRAIN-03 / D-09),
        //    scaled by learning_rate (stored value matches model.json). Leaf
        //    estimation uses the FULL fold (all objects) with the RAW (un-sampled)
        //    derivatives/weights. Every reduction over leaf members routes through
        //    cb_core::sum_f64 (D-05).
        let leaf_deltas = compute_leaf_deltas(
            params.leaf_method,
            &grown.leaf_of,
            &weighted_der1,
            &ders.der2,
            &weights,
            &approx,
            target,
            scaled_l2,
            n_leaves,
        );
        let leaf_values: Vec<f64> = leaf_deltas
            .iter()
            .map(|&delta| params.learning_rate * delta)
            .collect();

        // 4. Update approx: approx[i] += leaf_value[leaf(i)].
        for (i, &leaf) in grown.leaf_of.iter().enumerate() {
            if let (Some(a), Some(&lv)) = (approx.get_mut(i), leaf_values.get(leaf)) {
                *a += lv;
            }
        }

        // Record the staged approximant for this iteration (raw value / logit).
        if let Some(out) = staged_out.as_deref_mut() {
            out.extend_from_slice(&approx);
        }

        // POST per-tree draws. Two distinct main-RNG consumers run AFTER the tree
        // structure is grown:
        //   (a) the per-level `CalcScores` randSeed (greedy_tensor_search.cpp:884)
        //       — ONE `Rand.GenRand()` per level; and
        //   (b) the leaf-estimation seed (train.cpp:303,
        //       `GenRandUI64Vector(foldCount, Rand.GenRand())`) — ONE
        //       `Rand.GenRand()` per TREE, drawn once the tree is built.
        // When the perturbation is OFF but sampling is on, (a) is not observable
        // individually, so the prior wave folds (a)+(b) into a single bulk
        // `depth + 1` advance that keeps the next tree's Bootstrap phase-aligned.
        // When the perturbation is ON, the perturbed search ALREADY consumed (a)'s
        // randSeed AND the `SelectBestCandidate` normal draws inline in exact
        // upstream order, so only (b) — the single leaf-estimation seed draw —
        // remains to be consumed here (train.cpp:303). This source-faithful draw
        // locks the FIRST tree end-to-end (splits + leaf values <= 1e-5); a
        // per-tree main-RNG phase drift remains for tree-1+ (the variable-length
        // normal-draw accounting could not be localized at tree granularity
        // without C++ instrumentation of `LearnProgress->Rand` — escalated to
        // D-11 / Open Q4, see the regularization oracle test header and SUMMARY).
        if perturb_active {
            for _ in 0..POST_TREE_EXTRA_DRAWS {
                rng.gen_rand();
            }
        } else if draws_active {
            for _ in 0..(params.depth + POST_TREE_EXTRA_DRAWS) {
                rng.gen_rand();
            }
        }

        // MVS lambda for the NEXT tree uses this tree's mean leaf L2 norm
        // (`CalculateLastIterMeanLeafValue`, mvs.cpp:21-35) over the stored
        // (learning_rate-scaled) leaf values.
        prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&leaf_values));

        trees.push(ObliviousTree {
            splits: grown.splits,
            leaf_values,
        });
    }

    Ok(Model {
        oblivious_trees: trees,
        bias,
    })
}
