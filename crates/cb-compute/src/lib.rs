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

mod histogram;
mod leaf;
mod loss;
mod ranking_der;
mod runtime;
mod score;

pub use histogram::{
    collect_leaf_residuals, reduce_leaf_der2, reduce_leaf_stats, LeafStats,
};
pub use leaf::{
    calc_average, exact_leaf_delta, gradient_leaf_delta, logcosh_exact_leaf_delta,
    newton_leaf_delta, pairwise_cholesky_solve, scale_l2_reg, simple_leaf_delta,
    solve_symmetric_newton, LeafMethod,
};
pub use loss::{
    calc_softmax, cross_entropy_der1, cross_entropy_der2, expectile_der1, expectile_der2,
    focal_der1, focal_der2, huber_der1, huber_der2, logcosh_der1, logcosh_der2, logloss_der1,
    logloss_der2, lq_der1, lq_der2, mae_der1, mae_der2, mape_der1, mape_der2,
    lambdamart_pair_grad, multi_crossentropy_ders, multiclass_onevsall_ders, pairlogit_pair_prob,
    poisson_der1, poisson_der2, quantile_der1,
    quantile_der2, queryrmse_der, querysoftmax_der, rmse_der1,
    rmse_der2, sigmoid, softmax_ders, tweedie_der1, tweedie_der2, FOCAL_P_MIN, QUANTILE_ALPHA,
    QUANTILE_DELTA,
};
pub use ranking_der::{
    calc_ders_for_queries, group_reduce_weighted, is_pairwise_scoring, is_plain_only,
    Competitor as RankingCompetitor, GroupSpan,
};
pub use runtime::{
    Derivatives, EScoreFunction, Float, LambdaMartMetric, Loss, Runtime, EXPECTILE_ALPHA_DEFAULT,
    QUERYSOFTMAX_BETA_DEFAULT, QUERYSOFTMAX_LAMBDA_DEFAULT,
};
pub use score::{
    add_leaf_plain, cosine_split_score, derivatives_std_dev_from_zero, l2_split_score,
    multi_dim_split_score, random_score_instance, score_st_dev, MINIMAL_SCORE,
};

#[cfg(test)]
mod histogram_test;
#[cfg(test)]
mod leaf_test;
#[cfg(test)]
mod loss_test;
#[cfg(test)]
mod score_test;
