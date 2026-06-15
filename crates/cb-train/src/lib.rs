#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-train` — the plain gradient-boosting training core (TRAIN-01/02/03).
//!
//! Grows symmetric oblivious trees ([`tree`]) over the generic `cb-compute`
//! `Runtime` boundary and drives the plain boosting loop ([`boosting`]) so a user
//! can train RMSE + Logloss models on the CPU whose splits, leaf values, and
//! staged approximants match upstream catboost 1.2.10 to <= 1e-5.
//!
//! # Parity discipline
//!
//! Every parity-critical float SUM routes through `cb_core::sum_f64` (via
//! `cb-compute`); the split tie-break is the strict `>` first-wins rule
//! (Pitfall 1); depth is capped against `2^depth` overflow (T-03-01-02). No
//! `unwrap`/`expect`/raw float fold in production (deny-lints + D-08).

mod autolr;
mod bootstrap;
mod boosting;
mod candidates;
mod ctr;
mod fold;
mod metrics;
mod overfit;
mod permutation;
mod projection;
mod tree;

pub use autolr::{coefficients as autolr_coefficients, guess as autolr_guess, TargetType};
pub use bootstrap::{
    bootstrap, last_iter_mean_leaf_value, BootstrapResult, EBootstrapType, BAYESIAN_BLOCK_SIZE,
    MVS_BLOCK_SIZE,
};
pub use candidates::{
    learn_set_cardinality, one_hot_max_size_default, route_categorical, route_column,
    tensor_ctr_candidates, CtrCandidate, EncodingPath,
};
pub use ctr::{
    bake_ctr_table, BakedCtrData, BakedCtrTable,
    accumulate_online, build_final_ctr, calc_ctr_inference, calc_ctr_online, calc_ctr_online_bin,
    calc_normalization, materialize_ctr_feature, online_ctr_prefix_binclf,
    ordered_ctr_per_permutation, CounterCalcMethod, CtrFeatureColumn, ECtrType, FinalCtrTable,
    OnlineCtrAccumulator, OnlineCtrPrefix, OrderedCtrPrefix, Prior, TCtrHistory, TCtrMeanHistory,
    SIMPLE_CLASSES_COUNT,
};
pub use fold::{
    body_sum_weights, body_tail_boundaries, body_tail_segments, create_folds, learning_fold_count,
    plain_fold_body_tail, select_min_batch_size, select_tail_size, Fold,
};
pub use metrics::{EvalMetric, EvalMetricHistory};
pub use overfit::{BestModelTracker, EOverfittingDetectorType, OverfittingDetector};
pub use permutation::{
    averaging_ctr_permutation, create_shuffled_indices, fisher_yates_permutation, fold_block_size,
    permutations, PERMUTATION_BLOCK_SIZE_THRESHOLD,
};
pub use projection::{calc_hash, enumerate_projections, fold_cat_hash, TProjection};
pub use boosting::{
    boosting_type_default, combinations_ctr_default, combinations_ctr_priors_default,
    counter_calc_method_default, ctr_border_count_default, fold_len_multiplier_default,
    has_time_default, max_ctr_complexity_default, model_size_reg_default, need_shuffle,
    ordered_approx_delta_simple,
    permutation_count_default, score_function_default,
    simple_ctr_default, simple_ctr_priors_default, train, train_cat, train_with_eval,
    train_with_eval_sets, BoostParams, EBoostingType, EvalSet, Model, ObliviousTree,
};
pub use tree::{
    check_depth, greedy_tensor_search_oblivious, greedy_tensor_search_oblivious_ordered,
    greedy_tensor_search_oblivious_with_ctr, grow_one_hot_tree, leaf_index, select_best_candidate,
    AnySplit, Candidate, CtrSplitSpec, FeatureMatrix, GrownOneHotTree, GrownTree, LevelKind,
    OneHotSplit, Split, MAX_DEPTH,
};
