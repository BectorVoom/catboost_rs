#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! `cb-compute` — the pure-generic compute boundary and host-side training math
//! (D-03/D-04). This crate defines the abstract [`Runtime`] / [`Float`] traits
//! the `cb-backend` CubeCL `CpuRuntime` implements, plus the host-side
//! parity-critical math: per-loss derivatives ([`loss`]), the ordered bucket
//! reduction ([`histogram`]), the L2 split score ([`score`]), and the
//! L2-regularized leaf estimation ([`leaf`]).
//!
//! # No `cubecl` (D-03)
//!
//! This crate MUST NOT depend on `cubecl`. The concrete runtime is bound only in
//! `cb-backend`; the GPU runtimes attach additively in Phase 7 without touching
//! this crate or `cb-train`.
//!
//! # Parity discipline (D-02/D-05/D-08)
//!
//! Every parity-critical float SUM in this crate routes through
//! `cb_core::sum_f64` in canonical object order. The backend kernels do ONLY
//! order-independent elementwise work; the order-sensitive reduction lives here.

mod custom;
mod embedding_calcers;
mod histogram;
mod lda_linalg;
mod leaf;
mod loss;
mod pairwise_scoring;
mod ranking_der;
mod runtime;
mod score;
mod text_calcers;

pub use custom::{
    CustomMetric, CustomMetricHandle, CustomObjective, CustomObjectiveHandle,
};
pub use embedding_calcers::{
    between_matrix, total_scatter, IncrementalCloud, KnnCalcer, KnnCloud, LdaCalcer,
    LDA_DEFAULT_REG,
};
pub use histogram::{
    collect_leaf_residuals, reduce_leaf_der2, reduce_leaf_stats, LeafStats,
};
pub use lda_linalg::{
    calculate_projection, jacobi_symmetric_eig, reduce_generalized, sgemv_rowmajor, SymmetricEig,
};
pub use leaf::{
    build_monotonic_linear_orders, calc_average, calc_monotonic_leaf_deltas, exact_leaf_delta,
    gradient_leaf_delta, logcosh_exact_leaf_delta, newton_leaf_delta, pairwise_cholesky_solve,
    scale_l2_reg, simple_leaf_delta, solve_symmetric_newton, LeafMethod,
};
pub use loss::{
    calc_softmax, cross_entropy_der1, cross_entropy_der2, expectile_der1, expectile_der2,
    focal_der1, focal_der2, huber_der1, huber_der2, logcosh_der1, logcosh_der2, logloss_der1,
    logloss_der2, lq_der1, lq_der2, mae_der1, mae_der2, mape_der1, mape_der2,
    lambdamart_pair_grad, multi_crossentropy_ders, multiclass_onevsall_ders, pairlogit_pair_prob,
    poisson_der1, poisson_der2, quantile_der1,
    quantile_der2, queryrmse_der, querysoftmax_der, rmse_der1,
    rmse_der2, rmse_with_uncertainty_ders, sigmoid, softmax_ders, tweedie_der1, tweedie_der2,
    FOCAL_P_MIN, QUANTILE_ALPHA, QUANTILE_DELTA,
};
pub use pairwise_scoring::{
    calculate_pairwise_score, compute_der_sums, compute_pair_weight_statistics,
    BucketPairWeightStatistics,
};
pub use ranking_der::{
    calc_ders_for_queries, group_reduce_weighted, is_pairwise_scoring, is_plain_only,
    Competitor as RankingCompetitor, GroupSpan,
};
pub use runtime::{
    DeviceGrownTree, Derivatives, EScoreFunction, Float, LambdaMartMetric, Loss, Runtime,
    StochasticRankMetric,
    EXPECTILE_ALPHA_DEFAULT, QUERYSOFTMAX_BETA_DEFAULT, QUERYSOFTMAX_LAMBDA_DEFAULT,
    STOCHASTIC_RANK_LAMBDA_DEFAULT, STOCHASTIC_RANK_MU_DEFAULT,
    STOCHASTIC_RANK_NUM_ESTIMATIONS_DEFAULT, STOCHASTIC_RANK_NU_DEFAULT,
    STOCHASTIC_RANK_SIGMA_DEFAULT, YETIRANK_DECAY_DEFAULT, YETIRANK_MAGIC_CONST,
    YETIRANK_PERMUTATIONS_DEFAULT,
};
pub use score::{
    add_leaf_plain, cosine_split_score, derivatives_std_dev_from_zero, l2_split_score,
    multi_dim_split_score, random_score_instance, score_st_dev, MINIMAL_SCORE,
};
pub use text_calcers::{
    bag_of_words_compute, bm25_compute, naive_bayes_compute, Bm25State, NaiveBayesState,
    BM25_DEFAULT_B, BM25_DEFAULT_K, BM25_DEFAULT_TRUNCATE_BORDER, NAIVE_BAYES_DEFAULT_PRIOR,
    NAIVE_BAYES_SEEN_TOKENS_PRIOR,
};

#[cfg(test)]
mod embedding_calcers_test;
#[cfg(test)]
mod histogram_test;
#[cfg(test)]
mod lda_linalg_test;
#[cfg(test)]
mod leaf_test;
#[cfg(test)]
mod loss_test;
#[cfg(test)]
mod score_test;
#[cfg(test)]
mod text_calcers_test;
