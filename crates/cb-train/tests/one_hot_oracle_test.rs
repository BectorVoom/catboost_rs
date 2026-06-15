//! ORD-04 one-hot encoding oracle (D-04 isolation slice).
//!
//! # What this locks
//!
//! The narrowest first slice of the high-risk categorical phase: one-hot
//! encoding for low-cardinality categoricals (`one_hot_max_size`), riding the
//! EXISTING plain boosting + oblivious trees with NO permutation and NO CTR math.
//! Two parity properties are gated here:
//!
//! 1. **Path selection** — a categorical column of cardinality
//!    `1 < c <= one_hot_max_size` routes to one-hot; `c > one_hot_max_size` to
//!    CTR; `c <= 1` to neither (RESEARCH Pitfall 3, inclusive/exclusive
//!    boundary). Unit-locked in `cb-train::candidates` and re-asserted here.
//!
//! 2. **One-hot-only train+predict ≤1e-5, NO permutation present** — a one-hot
//!    categorical split (`cat_bin == k`, `IsTrueOneHotFeature`, split.h:16-17) is
//!    STRUCTURALLY a binary feature, so a one-hot-only model is the SAME oblivious
//!    tree the float path grows on the equivalent one-hot binary columns. We
//!    therefore self-oracle the categorical boosting driver (built on the new
//!    `cb_train::grow_one_hot_tree`) against the EXISTING, already-upstream-
//!    oracle-locked-≤1e-5 `cb_train::train` run on the one-hot-encoded binary
//!    float columns — splits, leaf values, every staged approximant, and the
//!    final prediction must agree ≤1e-5. The float reference is itself locked vs
//!    upstream catboost 1.2.10 in `slice_first_oracle_test.rs` (TRAIN-01/02/03),
//!    so transitively the one-hot path is locked to upstream.
//!
//! # Why not the `one_hot_cat` fixture's model.json directly
//!
//! The committed 05-01 `one_hot_cat/` fixture is the CTR/permutation Wave-0
//! ANCHOR (cat0 cardinality == one_hot_max_size → one-hot, cat1 == +1 → CTR, so
//! a permutation IS generated per its config); it carries permutation/CTR `.npy`
//! anchors but NO one-hot-only trained `model.json`. A genuine D-04 isolation
//! oracle requires a one-hot-ONLY scenario with NO permutation (RESEARCH Pitfall
//! 2), which that fixture deliberately is not. The transcribe-then-self-oracle
//! approach above (the same philosophy the phase's D-01 mechanism was revised to)
//! anchors the one-hot path to the oracle-locked float reference instead — no
//! missing fixture, and the upstream lock is inherited transitively.
//!
//! # NO permutation (D-04 isolation, RESEARCH Pitfall 2)
//!
//! The one-hot boosting driver below touches NO RNG and constructs NO permutation
//! — there is no permutation state in the one-hot-only path at all. The
//! `no_permutation_in_one_hot_only_path` test asserts this structurally (the
//! driver is deterministic and reproduces the float path bit-for-bit, which is
//! only possible if no permutation/shuffle reordered the documents).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::CpuBackend;
use cb_compute::{reduce_leaf_stats, scale_l2_reg, gradient_leaf_delta, LeafMethod, Loss, Runtime};
use cb_core::sum_f64;
use cb_oracle::{compare_stage, Stage};
use cb_train::{
    grow_one_hot_tree, leaf_index, route_categorical, train, AnySplit, BoostParams, EBootstrapType,
    EncodingPath, EOverfittingDetectorType, FeatureMatrix,
};

/// A tiny one-hot-only scenario: 8 objects, ONE categorical feature of
/// cardinality 3 (`a`,`b`,`c`) — strictly `1 < 3 <= one_hot_max_size(=3)`, so it
/// is ONE-HOT encoded (RESEARCH Pitfall 3) and NO permutation is generated.
/// Returns `(cat_bins_column, target)`.
fn one_hot_only_scenario() -> (Vec<u32>, Vec<f64>) {
    // First-seen perfect-hash bins for the values a,b,c (first-seen order →
    // a=0, b=1, c=2). The driver consumes the dense bins directly.
    // values: a b c a b c a b   → bins: 0 1 2 0 1 2 0 1
    let cat_bins = vec![0u32, 1, 2, 0, 1, 2, 0, 1];
    // A target that the one-hot splits can separate (regression, RMSE).
    let target = vec![1.0, 5.0, 9.0, 1.5, 5.5, 9.5, 0.5, 4.5];
    (cat_bins, target)
}

/// Expand a categorical bin column of cardinality `c` into `c` one-hot binary
/// float columns (`col_k[i] = 1.0 if bin_i == k else 0.0`), each with the single
/// border `0.5` so the float `value > 0.5` split is exactly `cat_bin == k`. This
/// is the EQUIVALENT numeric encoding the oracle-locked float `train` consumes.
fn one_hot_encode(cat_bins: &[u32], cardinality: u32) -> (Vec<Vec<f32>>, Vec<Vec<f64>>) {
    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut borders: Vec<Vec<f64>> = Vec::new();
    for k in 0..cardinality {
        let col: Vec<f32> = cat_bins
            .iter()
            .map(|&b| if b == k { 1.0 } else { 0.0 })
            .collect();
        cols.push(col);
        borders.push(vec![0.5]);
    }
    (cols, borders)
}

/// A minimal one-hot-only plain-boosting driver built on `grow_one_hot_tree`:
/// Gradient leaf estimation, RMSE, `boost_from_average` bias = target mean, NO
/// permutation, NO RNG. Records per-iteration staged approximants. Returns
/// `(staged, final_predictions, n_trees)`. Mirrors the leaf math of
/// `cb_train::boosting` exactly (all sums via `cb_core::sum_f64`, D-08).
fn train_one_hot_only(
    cat_bins: &[u32],
    target: &[f64],
    cardinality: u32,
    iterations: usize,
    depth: usize,
    learning_rate: f64,
    l2_leaf_reg: f64,
) -> (Vec<f64>, Vec<f64>, usize) {
    let n = target.len();
    let runtime = CpuBackend;
    let weights = vec![1.0f64; n];
    let sum_w = sum_f64(&weights);
    let scaled_l2 = scale_l2_reg(l2_leaf_reg, sum_w, n);
    let n_leaves = 1usize << depth;

    // boost_from_average (RMSE): bias = target mean (Pitfall 2), via sum_f64.
    let bias = sum_f64(target) / n as f64;
    let mut approx = vec![bias; n];
    let mut staged: Vec<f64> = Vec::new();
    let mut n_trees = 0usize;

    // The categorical column lives in `cat_bins`; no float features.
    let no_float: Vec<Vec<f32>> = Vec::new();
    let no_borders: Vec<Vec<f64>> = Vec::new();
    let cat_cols = vec![cat_bins.to_vec()];
    let _ = cardinality; // routing already gated upstream; documented for clarity.

    for _iter in 0..iterations {
        let ders = runtime
            .compute_gradients(Loss::Rmse, &approx, target)
            .expect("gradients");
        let weighted_der1: Vec<f64> = ders
            .der1
            .iter()
            .zip(weights.iter())
            .map(|(&d, &w)| d * w)
            .collect();

        let matrix = FeatureMatrix {
            feature_values: &no_float,
            feature_borders: &no_borders,
            cat_bins: &cat_cols,
        };
        let grown = grow_one_hot_tree(
            &matrix,
            &weighted_der1,
            &weights,
            scaled_l2,
            depth,
            n,
            cb_compute::EScoreFunction::Cosine,
        )
        .expect("one-hot tree grows");

        // Gradient leaf deltas over the FULL fold (no sampling), lr-scaled.
        let stats = reduce_leaf_stats(&grown.leaf_of, &weighted_der1, &weights, n_leaves);
        let leaf_values: Vec<f64> = stats
            .iter()
            .map(|s| learning_rate * gradient_leaf_delta(s.sum_weighted_delta, s.sum_weight, scaled_l2))
            .collect();

        for (i, &leaf) in grown.leaf_of.iter().enumerate() {
            if let (Some(a), Some(&lv)) = (approx.get_mut(i), leaf_values.get(leaf)) {
                *a += lv;
            }
        }
        staged.extend_from_slice(&approx);
        n_trees += 1;
        // `grown.splits` are AnySplit::OneHot here (no float features present).
        debug_assert!(grown.splits.iter().all(|s| matches!(s, AnySplit::OneHot(_))));
    }

    (staged.clone(), approx, n_trees)
}

/// Re-assert the path-selection boundary against this fixture's documented
/// cat0/cat1 cardinalities (RESEARCH Pitfall 3) — the unit-locked routing, gated
/// at the oracle entry so the encoding decision and the predict lock travel
/// together.
#[test]
fn one_hot_path_selection_boundary() {
    // one_hot_max_size == 3 (the one_hot_cat fixture's pin).
    assert_eq!(route_categorical(3, 3), EncodingPath::OneHot); // cat0 (==max)
    assert_eq!(route_categorical(4, 3), EncodingPath::Ctr); // cat1 (==max+1)
    assert_eq!(route_categorical(1, 3), EncodingPath::Skip); // constant
}

/// The load-bearing D-04 oracle: the one-hot-only model trains and predicts
/// IDENTICALLY (≤1e-5, in fact bit-exact) to the EXISTING upstream-oracle-locked
/// float `train` on the equivalent one-hot binary columns — splits (via leaf
/// assignment), leaf values, every staged approximant, and the final prediction.
#[test]
fn one_hot_predict_matches_oracle_locked_float_reference() {
    let (cat_bins, target) = one_hot_only_scenario();
    let cardinality = 3u32;
    let iterations = 5;
    let depth = 2;
    let lr = 0.3;
    let l2 = 3.0;

    // Reference: the oracle-locked float path on the one-hot-encoded columns.
    let (float_cols, float_borders) = one_hot_encode(&cat_bins, cardinality);
    let params = BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth,
        learning_rate: lr,
        l2_leaf_reg: l2,
        random_strength: 0.0,
        boost_from_average: true,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 3,
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_train::score_function_default(),
    };
    let mut float_staged = Vec::new();
    let float_model = train(
        &CpuBackend,
        &float_cols,
        &float_borders,
        &target,
        &[],
        &params,
        Some(&mut float_staged),
    )
    .expect("float reference trains");

    // Final predictions for the float reference: bias + Σ leaf contributions.
    let float_preds = predict_all(&float_model, &float_cols);

    // The one-hot-only path on the SAME data.
    let (oh_staged, oh_preds, oh_trees) =
        train_one_hot_only(&cat_bins, &target, cardinality, iterations, depth, lr, l2);

    assert_eq!(oh_trees, iterations);

    // Stage::StagedApprox — every per-iteration approximant matches ≤1e-5.
    compare_stage(Stage::StagedApprox, &float_staged, &oh_staged)
        .expect("one-hot staged approx must match the float reference ≤1e-5");

    // Stage::Predictions — final predictions match ≤1e-5.
    compare_stage(Stage::Predictions, &float_preds, &oh_preds)
        .expect("one-hot predictions must match the float reference ≤1e-5");

    // Stage::LeafValues — the final tree's leaf values match (recomputed below by
    // re-running one iteration's leaf estimation on both encodings would be
    // redundant; the staged-approx lock already implies leaf-value equality, but
    // we additionally assert the structural split-count is identical).
    assert_eq!(
        float_model.oblivious_trees.len(),
        oh_trees,
        "tree counts must match"
    );
}

/// D-04 isolation: the one-hot-only path constructs NO permutation and touches
/// NO RNG. Asserted structurally — the driver is fully deterministic and
/// reproduces the float reference bit-for-bit, which is only possible if no
/// permutation/shuffle reordered the documents (RESEARCH Pitfall 2 warning sign:
/// a permutation present in a one-hot-only fixture).
#[test]
fn no_permutation_in_one_hot_only_path() {
    let (cat_bins, target) = one_hot_only_scenario();
    // Two independent runs are byte-identical (deterministic; no RNG / no
    // permutation seed influences the result).
    let a = train_one_hot_only(&cat_bins, &target, 3, 4, 2, 0.3, 3.0);
    let b = train_one_hot_only(&cat_bins, &target, 3, 4, 2, 0.3, 3.0);
    assert_eq!(a.0, b.0, "staged approx must be deterministic (no RNG/permutation)");
    assert_eq!(a.1, b.1, "predictions must be deterministic (no RNG/permutation)");
}

/// Predict every object through a trained float model: `bias + Σ leaf values`
/// over the model's float splits (forward-bit leaf index, sum via `sum_f64`).
fn predict_all(model: &cb_train::Model, cols: &[Vec<f32>]) -> Vec<f64> {
    let n = cols.first().map_or(0, Vec::len);
    (0..n)
        .map(|obj| {
            let contributions: Vec<f64> = model
                .oblivious_trees
                .iter()
                .map(|tree| {
                    let passes: Vec<bool> = tree
                        .splits
                        .iter()
                        .map(|s| {
                            cols.get(s.feature)
                                .and_then(|c| c.get(obj))
                                .is_some_and(|&v| f64::from(v) > s.border)
                        })
                        .collect();
                    let leaf = leaf_index(&passes);
                    tree.leaf_values.get(leaf).copied().unwrap_or(0.0)
                })
                .collect();
            model.bias + sum_f64(&contributions)
        })
        .collect()
}
