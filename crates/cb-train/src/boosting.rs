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
    gradient_leaf_delta, reduce_leaf_stats, scale_l2_reg, Loss, Runtime,
};
use cb_core::{sum_f64, CbError, CbResult};

use crate::tree::{check_depth, greedy_tensor_search_oblivious, FeatureMatrix, Split};

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
    /// Whether to start from the per-loss optimum constant approx (the target
    /// mean for RMSE), stored as the model bias. `false` starts from `0`.
    pub boost_from_average: bool,
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

    for _iter in 0..params.iterations {
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

        // 2. Grow one oblivious tree using the L2 split score over the ordered
        //    leaf-stat reduction.
        let scaled_l2 = scale_l2_reg(params.l2_leaf_reg, sum_all_weights, n);
        let grown = greedy_tensor_search_oblivious(
            &matrix,
            &weighted_der1,
            &weights,
            scaled_l2,
            params.depth,
            n,
        )?;

        // 3. Gradient leaf values: CalcAverage over each leaf's ordered member
        //    sums, scaled by learning_rate (stored value matches model.json).
        let stats = reduce_leaf_stats(&grown.leaf_of, &weighted_der1, &weights, n_leaves);
        let leaf_values: Vec<f64> = stats
            .iter()
            .map(|s| {
                let delta = gradient_leaf_delta(s.sum_weighted_delta, s.sum_weight, scaled_l2);
                params.learning_rate * delta
            })
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
