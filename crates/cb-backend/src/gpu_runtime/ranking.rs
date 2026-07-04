//! GPUT-22 (Phase 13 Plan 04, Wave 4): the deterministic **query/listwise objective device
//! driver** — QueryRMSE, QuerySoftMax, and QueryCrossEntropy — transcribing the CPU
//! `cb_compute::ranking_der::calc_ders_for_queries` der math onto the device path over the
//! Plan-03 shared query-grouping infrastructure ([`crate::kernels::query_helper`]). This is the
//! amortization payoff of that grouping substrate: the group means / group max / per-query bias
//! removal the ranking der needs are ALL computed by the Plan-03 kernels; this driver adds only the
//! per-objective pointwise der arithmetic (RMSE), the per-query softmax der (QuerySoftMax), and the
//! bounded per-query shift search (QueryCrossEntropy).
//!
//! # What lives here (production, NOT `#[cfg(test)]`)
//!
//! - [`query_rmse_ders_host`] — QueryRMSE: `RemoveGroupMeans` on the resident residual (the
//!   `queryAvrg` mean-removal from the Plan-03 group-means kernel) then the pointwise
//!   `der1 = (residual − queryAvrg)·w`, `der2 = −w` in [`ranking_rmse_der_kernel`].
//! - [`query_softmax_ders_host`] — QuerySoftMax: the per-query softmax der using the Plan-03
//!   `ComputeGroupMax` shift + the weighted exp-share `p`, in [`query_softmax_der_kernel`].
//! - [`query_cross_entropy_shifts_host`] — QueryCrossEntropy's per-query bisection + Newton shift
//!   search ([`query_cross_entropy_shift_kernel`]), a BOUNDED root-find (Open Q3). This arm is
//!   INDEPENDENTLY gated: it is deferred (not in the covered ranking set, [`ranking_objective_covered`])
//!   because it has no `cb_compute::ranking_der` CPU der oracle / `Loss` variant yet, so it can be
//!   shipped structurally without over-running QueryRMSE / QuerySoftMax.
//!
//! # Parity discipline — the covered regime is uniform-weight
//!
//! The covered device ranking regime is UNIFORM object weights (`weights` empty → `1.0` column, the
//! same expansion [`crate::kernels::query_helper::compute_group_means_host`] performs). Under
//! uniform weights the Plan-03 `ComputeGroupMax` (max over ALL objects) equals the CPU's
//! max-over-`weight>0` seed, and the group means fixed-point reduction matches
//! `group_reduce_weighted` / `cb_core::sum_f64` within ε=1e-4 (proven by the Plan-03 self-oracle).
//! Each per-query serial der loop walks `[begin, end)` in ascending doc order — the SAME order the
//! CPU `calc_ders_for_queries` per-group loop uses — so the serial single-thread accumulation is
//! deterministic and order-matches the CPU reference by construction (no atomic needed).
//!
//! # f64/u64-typed seam (WR-01)
//!
//! The der accumulates in f64 and the group infra uses u64 fixed-point / u64 RNG; WGSL has neither
//! f64 nor u64, so a genuine `wgpu` backend surfaces a typed [`CbError::OutOfRange`] rather than an
//! opaque JIT crash (the [`crate::kernels::query_helper`] / [`crate::kernels::mvs_device`]
//! precedent). No `-inf` literal in any `#[cube]` body (the shift search uses a FINITE bracket). No
//! `unwrap`/`expect`/`panic`/indexing in production (workspace lints + D-13).

use cubecl::prelude::*;
use cubecl::server::Handle;

use cb_core::{CbError, CbResult};

use crate::kernels::query_helper::{
    compute_group_ids_host, compute_group_max_host, compute_group_means_host,
    remove_group_means_host,
};
#[cfg(not(feature = "wgpu"))]
use crate::kernels::exact_quantile::segmented_radix_sort;
use crate::SelectedRuntime;

/// The deterministic query/listwise objective this driver computes the der for. QueryRMSE and
/// QuerySoftMax are the COVERED deterministic arms (a `cb_compute::ranking_der` CPU der oracle
/// exists for each — [`ranking_objective_covered`] returns `true`). QueryCrossEntropy is the
/// INDEPENDENTLY-DEFERRED arm (Open Q3): its bounded per-query shift search is landed structurally
/// but it has no CPU der oracle / `Loss` variant yet, so it is NOT covered (the session gate returns
/// `Ok(None)` for it without disabling QueryRMSE / QuerySoftMax).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RankingObjective {
    /// QueryRMSE — per-query mean-removed residual (`error_functions.h:879-933`).
    QueryRmse,
    /// QuerySoftMax — per-query weighted softmax der (`error_functions.cpp:540-576`).
    QuerySoftMax {
        /// The inverse-temperature `Beta` (`QuerySoftMax:beta`, default `1.0`).
        beta: f64,
        /// The L2 regularization `LambdaReg` (`QuerySoftMax:lambda`, default `0.01`).
        lambda: f64,
    },
    /// QueryCrossEntropy — per-query bisection/Newton shift search (Open Q3, INDEPENDENTLY gated
    /// off: no CPU der oracle yet, so it is deferred rather than shipped as unverified der parity).
    QueryCrossEntropy,
    /// YetiRank (Phase 13 Plan 05, D-08) — the STOCHASTIC in-query bootstrap-sampled listwise
    /// objective (pointwise leaf). Per bootstrap iteration the device perturbs each doc's
    /// exp-approx with `uni/(1.000001−uni)` (inline PCG, pinned seed), sorts within each query
    /// (`segmented_radix_sort`), accumulates the Classic decayed pairwise weights, then the
    /// accumulated competitors feed the pairwise-logit der — reproducing the CPU
    /// `yetirank_sample_pairs` + `calc_ders_for_queries` stream bit-for-bit (Pitfall 4).
    YetiRank {
        /// Noise permutation count (`permutations`, default 10; validated `>= 1`).
        permutations: u32,
        /// Classic-weight geometric decay (`decay`, default 0.85; validated in `[0, 1]`).
        decay: f64,
    },
    /// PFound-F (Phase 13 Plan 05, D-08) — the STOCHASTIC pairwise-leaf listwise objective
    /// (`YetiRankPairwise`, the GPU `pfound_f` arm). Shares the SAME sampled-pair RNG stream +
    /// competitor der as [`RankingObjective::YetiRank`]; only the leaf path differs (Cholesky vs
    /// pointwise), decided later in boosting — NOT in the der. So the device der is identical.
    PFoundF {
        /// Noise permutation count (`permutations`, default 10; validated `>= 1`).
        permutations: u32,
        /// Classic-weight geometric decay (`decay`, default 0.85; validated in `[0, 1]`).
        decay: f64,
    },
}

/// Whether `objective` is in the COVERED deterministic ranking set (has a `cb_compute::ranking_der`
/// CPU der oracle this driver is self-oracled against). QueryRMSE / QuerySoftMax → `true`;
/// QueryCrossEntropy → `false` (Open Q3, independently deferred behind its own `Ok(None)` gate so
/// QueryRMSE / QuerySoftMax ship even though QueryCrossEntropy's der oracle is not landed).
#[must_use]
pub(crate) fn ranking_objective_covered(objective: RankingObjective) -> bool {
    match objective {
        RankingObjective::QueryRmse
        | RankingObjective::QuerySoftMax { .. }
        // Phase 13 Plan 05 (D-08): the stochastic pair is COVERED — its device der reproduces the
        // FROZEN pinned-seed CPU `yetirank_sample_pairs` + `calc_ders_for_queries` reference
        // bit-for-bit at ε=1e-4 (the `ranking_stoch_test` self-oracle).
        | RankingObjective::YetiRank { .. }
        | RankingObjective::PFoundF { .. } => true,
        RankingObjective::QueryCrossEntropy => false,
    }
}

/// QueryCrossEntropy shift-search BISECTION iteration budget (Open Q3, T-13-07 DoS mitigation): a
/// FIXED count (mirrors the `mvs_device` `MVS_BISECTION_ITERS` monotone-bracket precedent) so the
/// per-query loop can never run unbounded. Upstream `query_cross_entropy.cu` runs 8 bisection
/// halvings; a `2·BRACKET / 2^8` bracket is < 0.24 before Newton refines.
const QCE_SHIFT_BISECTION_ITERS: u32 = 8;

/// QueryCrossEntropy shift-search NEWTON refinement budget (Open Q3, T-13-07): a FIXED count of
/// Newton steps after the bisection bracket (upstream runs 5). Bounded — never unbounded.
const QCE_SHIFT_NEWTON_ITERS: u32 = 5;

/// The FINITE shift-search bracket `[−BRACKET, +BRACKET]` (Pattern D — NO `-inf` literal in a
/// `#[cube]` body). A logit shift of ±30 saturates every sigmoid, bracketing any feasible root.
const QCE_SHIFT_BRACKET: f64 = 30.0;

// ===========================================================================
// #[cube] ranking der kernels
// ===========================================================================

/// QueryRMSE pointwise der over the per-query MEAN-REMOVED residual `centered[d] = (target[d] −
/// approx[d]) − queryAvrg[qid]` (the `RemoveGroupMeans` output from the Plan-03 grouping infra) and
/// the per-object `weights[d]`. Doc-parallel (`ABSOLUTE_POS`-indexed):
/// `der1[d] = centered[d]·w`, `der2[d] = −1·w` (`error_functions.h:901-907`
/// `TQueryRmseError::CalcDersForQueries` — the querywise der folds the weight IN). Order-independent
/// (each doc writes its own slot).
#[cube(launch)]
fn ranking_rmse_der_kernel(
    centered: &Array<f64>,
    weights: &Array<f64>,
    der1: &mut Array<f64>,
    der2: &mut Array<f64>,
) {
    let d = ABSOLUTE_POS;
    if d < centered.len() {
        let w = weights[d];
        der1[d] = centered[d] * w;
        der2[d] = -1.0f64 * w;
    }
}

/// QuerySoftMax per-query der (`error_functions.cpp:540-576` `TQuerySoftMaxError`). Serial (unit 0)
/// per query, consuming the Plan-03 `ComputeGroupMax` shift `group_max[g]` (the max-shift before
/// `exp`, applied per query). `params = [beta, lambda]`. For each query `g` over `[begin, end)`:
/// - `sumWTargets = Σ_{w>0, t>0} w·t`; if `≤ 0` every der in the query is `0` (no divide).
/// - else `sumExpApprox = Σ exp(Beta·(approx − maxApprox))·w`, `p = expApprox·w / sumExpApprox`,
///   `der2 = Beta·sumWTargets·(Beta·p·(p−1) − LambdaReg)`, `der1 = Beta·(−sumWTargets·p + w·t)` for
///   `w>0`; `der1 = der2 = 0` for `w≤0`.
/// The serial `[begin, end)` walk matches the CPU per-group ascending order exactly (deterministic).
#[cube(launch)]
fn query_softmax_der_kernel(
    approx: &Array<f64>,
    target: &Array<f64>,
    weights: &Array<f64>,
    q_offsets: &Array<u32>,
    group_max: &Array<f64>,
    params: &Array<f64>,
    der1: &mut Array<f64>,
    der2: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let beta = params[0];
        let lambda = params[1];
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            let max_a = group_max[g];

            // sumWTargets = Σ_{w>0, t>0} w·t (max-shift is applied below; this term is shift-free).
            let mut sum_wt = 0.0f64;
            let mut d = begin;
            while d < end {
                let w = weights[d as usize];
                let t = target[d as usize];
                if w > 0.0f64 {
                    if t > 0.0f64 {
                        sum_wt += w * t;
                    }
                }
                d += 1u32;
            }

            if sum_wt > 0.0f64 {
                // sumExpApprox = Σ exp(Beta·(approx − maxApprox))·w.
                let mut sum_exp = 0.0f64;
                let mut de = begin;
                while de < end {
                    let w = weights[de as usize];
                    let a = approx[de as usize];
                    sum_exp += f64::exp(beta * (a - max_a)) * w;
                    de += 1u32;
                }
                let mut dd = begin;
                while dd < end {
                    let w = weights[dd as usize];
                    let mut e1 = 0.0f64;
                    let mut e2 = 0.0f64;
                    if w > 0.0f64 {
                        if sum_exp > 0.0f64 {
                            let a = approx[dd as usize];
                            let t = target[dd as usize];
                            let p = (f64::exp(beta * (a - max_a)) * w) / sum_exp;
                            e2 = beta * sum_wt * (beta * p * (p - 1.0f64) - lambda);
                            e1 = beta * (-sum_wt * p + w * t);
                        }
                    }
                    der1[dd as usize] = e1;
                    der2[dd as usize] = e2;
                    dd += 1u32;
                }
            } else {
                // sumWTargets ≤ 0 → every der in the query is 0 (error_functions.cpp:571-576).
                let mut dz = begin;
                while dz < end {
                    der1[dz as usize] = 0.0f64;
                    der2[dz as usize] = 0.0f64;
                    dz += 1u32;
                }
            }
            g += 1usize;
        }
    }
}

/// QueryCrossEntropy per-query BOUNDED shift search (Open Q3; T-13-07 DoS mitigation). Serial (unit
/// 0) per query. Finds the per-query logit shift `b` solving the monotone equation
/// `F(b) = Σ w·sigmoid(approx + b) = Σ w·target` by a FIXED-count bisection over the FINITE bracket
/// `[−BRACKET, +BRACKET]` (no `-inf` literal, Pattern D) followed by a FIXED-count Newton refine
/// (`F'(b) = Σ w·p·(1−p)`). Both loop counts are compile-time constants — the search can never run
/// unbounded (the DoS threat). `shift_out[g]` is the per-query shift.
///
/// NOTE: this arm is INDEPENDENTLY DEFERRED ([`ranking_objective_covered`] returns `false` for
/// QueryCrossEntropy) — the shift search is a genuine, self-consistent root-find (verifiable
/// `F(shift) ≈ Σ w·t`), but the full QueryCrossEntropy der has no `cb_compute::ranking_der` CPU
/// oracle / `Loss` variant yet, so the session gate declines it rather than shipping unverified der
/// parity. Landing the bounded search here keeps QueryCrossEntropy ready without over-running the
/// covered QueryRMSE / QuerySoftMax arms.
#[cube(launch)]
fn query_cross_entropy_shift_kernel(
    approx: &Array<f64>,
    target: &Array<f64>,
    weights: &Array<f64>,
    q_offsets: &Array<u32>,
    params: &Array<f64>,
    shift_out: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        // The FINITE bracket is a RUNTIME scalar (params[0]) — NOT the comptime const directly:
        // seeding `lo`/`hi` from a comptime f64 makes them comptime-typed, which then clashes when
        // reassigned from the runtime `mid` (a `NativeExpand<f64>` type mismatch). The host passes
        // `QCE_SHIFT_BRACKET` in as runtime data (the `mvs_device` runtime-bracket precedent).
        let bracket = params[0];
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];

            // Target mass T = Σ w·target (the value F(b) must reach).
            let mut t_mass = 0.0f64;
            let mut d = begin;
            while d < end {
                t_mass += weights[d as usize] * target[d as usize];
                d += 1u32;
            }

            // Bisection over the FINITE bracket; F(b) = Σ w·sigmoid(approx + b) is monotone in b.
            let mut lo = 0.0f64 - bracket;
            let mut hi = bracket;
            let mut it = 0u32;
            while it < QCE_SHIFT_BISECTION_ITERS {
                let mid = 0.5f64 * (lo + hi);
                let mut f = 0.0f64;
                let mut k = begin;
                while k < end {
                    let z = approx[k as usize] + mid;
                    let p = 1.0f64 / (1.0f64 + f64::exp(0.0f64 - z));
                    f += weights[k as usize] * p;
                    k += 1u32;
                }
                if f < t_mass {
                    lo = mid;
                } else {
                    hi = mid;
                }
                it += 1u32;
            }
            let mut b = 0.5f64 * (lo + hi);

            // Newton refine (bounded): b ← b − (F(b) − T) / F'(b).
            let mut nit = 0u32;
            while nit < QCE_SHIFT_NEWTON_ITERS {
                let mut f = 0.0f64;
                let mut fp = 0.0f64;
                let mut k = begin;
                while k < end {
                    let z = approx[k as usize] + b;
                    let p = 1.0f64 / (1.0f64 + f64::exp(0.0f64 - z));
                    let w = weights[k as usize];
                    f += w * p;
                    fp += w * p * (1.0f64 - p);
                    k += 1u32;
                }
                if fp > 0.0f64 {
                    let num = f - t_mass;
                    let step = num / fp;
                    b = b - step;
                }
                nit += 1u32;
            }
            shift_out[g] = b;
            g += 1usize;
        }
    }
}

// ===========================================================================
// Host launch wrappers
// ===========================================================================

/// Reject the (impossible) wgpu f64/u64 path with a typed error (WR-01), mirroring
/// [`crate::kernels::query_helper`]. Kept in one place so every entry point agrees.
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError {
    CbError::OutOfRange(
        "device ranking der requires f64 + u64 device channels; the wgpu backend has neither \
         (WR-01). Use the rocm/cuda/cpu backend for the query der."
            .to_owned(),
    )
}

/// The selected-runtime client (one per call, mirroring [`crate::kernels::query_helper`]).
#[cfg(not(feature = "wgpu"))]
fn selected_client() -> cubecl::client::ComputeClient<SelectedRuntime> {
    let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
    <SelectedRuntime as cubecl::Runtime>::client(&device)
}

/// Expand the caller's `weights` to a uniform `1.0` column when empty (the covered regime), else
/// validate the length. Mirrors [`crate::kernels::query_helper::compute_group_means_host`] so the
/// der driver and the group reductions agree on the weight column.
fn weight_column(weights: &[f64], n: usize) -> CbResult<Vec<f64>> {
    if weights.is_empty() {
        return Ok(vec![1.0; n]);
    }
    if weights.len() != n {
        return Err(CbError::Degenerate(format!(
            "ranking der: weights len {} != n {n}",
            weights.len()
        )));
    }
    Ok(weights.to_vec())
}

/// Validate the flat ranking der inputs (shared by every objective). `approx` / `target` are length
/// `n`; `q_offsets` has `n_groups + 1` entries covering `[0, n)`.
fn validate_ranking_inputs(approx: &[f64], target: &[f64], q_offsets: &[u32]) -> CbResult<()> {
    if approx.len() != target.len() {
        return Err(CbError::Degenerate(format!(
            "ranking der: approx len {} != target len {}",
            approx.len(),
            target.len()
        )));
    }
    if q_offsets.is_empty() {
        return Err(CbError::Degenerate(
            "ranking der: q_offsets must have n_groups + 1 entries (got 0)".to_owned(),
        ));
    }
    Ok(())
}

/// QueryRMSE device der: the per-query mean-removed residual then the pointwise der. Composes the
/// Plan-03 grouping infra — `ComputeGroupMeans` (the `queryAvrg` numerator/denominator fixed-point
/// reduction), `ComputeGroupIds`, `RemoveGroupMeans` (`residual − queryAvrg`) — with
/// [`ranking_rmse_der_kernel`]. Returns `(der1, der2)` (each length `n`), matching
/// `cb_compute::calc_ders_for_queries` for `Loss::QueryRmse` within ε=1e-4.
pub(crate) fn query_rmse_ders_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    validate_ranking_inputs(approx, target, q_offsets)?;
    let n = approx.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (weights, q_offsets);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let weight_col = weight_column(weights, n)?;
        // residual[i] = target[i] - approx[i] (input prep; the group reduction is on-device).
        let residuals: Vec<f64> = approx
            .iter()
            .zip(target.iter())
            .map(|(&a, &t)| t - a)
            .collect();
        // queryAvrg per group = Σ(residual·w)/Σw (Plan-03 fixed-point group-means reduction).
        let query_avrg = compute_group_means_host(&residuals, &weight_col, q_offsets)?;
        // qids[d] = the doc's query id (Plan-03 ComputeGroupIds).
        let qids = compute_group_ids_host(q_offsets, n)?;
        // centered[d] = residual[d] - queryAvrg[qids[d]] (Plan-03 RemoveGroupMeans).
        let centered = remove_group_means_host(&residuals, &qids, &query_avrg)?;

        // Pointwise der: der1 = centered·w, der2 = -w.
        let client = selected_client();
        let centered_h = client.create(cubecl::bytes::Bytes::from_elems(centered));
        let w_h = client.create(cubecl::bytes::Bytes::from_elems(weight_col));
        let der1 = client.empty(n * std::mem::size_of::<f64>());
        let der2 = client.empty(n * std::mem::size_of::<f64>());
        let block = 64u32;
        let cubes = n.div_ceil(block as usize).max(1) as u32;
        ranking_rmse_der_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim { x: block, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(centered_h, n) },
            unsafe { ArrayArg::from_raw_parts(w_h, n) },
            unsafe { ArrayArg::from_raw_parts(der1.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(der2.clone(), n) },
        );
        let der1_v = read_f64(&client, der1, "query_rmse_der1")?;
        let der2_v = read_f64(&client, der2, "query_rmse_der2")?;
        Ok((der1_v, der2_v))
    }
}

/// QuerySoftMax device der: the per-query weighted softmax der using the Plan-03 `ComputeGroupMax`
/// shift. Returns `(der1, der2)` (each length `n`), matching `cb_compute::calc_ders_for_queries` for
/// `Loss::QuerySoftMax { lambda, beta }` within ε=1e-4 (covered uniform-weight regime).
pub(crate) fn query_softmax_ders_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
    beta: f64,
    lambda: f64,
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    validate_ranking_inputs(approx, target, q_offsets)?;
    let n = approx.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (weights, q_offsets, beta, lambda);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let weight_col = weight_column(weights, n)?;
        // Per-query max approx (Plan-03 ComputeGroupMax) — the max-shift before exp. Uniform-weight
        // covered regime: max over all objects == the CPU max-over-(weight>0) seed.
        let group_max = compute_group_max_host(approx, q_offsets)?;
        let n_groups = q_offsets.len().saturating_sub(1);

        let client = selected_client();
        let approx_h = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
        let target_h = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
        let w_h = client.create(cubecl::bytes::Bytes::from_elems(weight_col));
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        let max_h = client.create(cubecl::bytes::Bytes::from_elems(group_max));
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![beta, lambda]));
        let der1 = client.empty(n * std::mem::size_of::<f64>());
        let der2 = client.empty(n * std::mem::size_of::<f64>());
        // Serial single-thread launch (unit 0 loops the queries).
        query_softmax_der_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(approx_h, n) },
            unsafe { ArrayArg::from_raw_parts(target_h, n) },
            unsafe { ArrayArg::from_raw_parts(w_h, n) },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(max_h, n_groups) },
            unsafe { ArrayArg::from_raw_parts(params_h, 2) },
            unsafe { ArrayArg::from_raw_parts(der1.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(der2.clone(), n) },
        );
        let der1_v = read_f64(&client, der1, "query_softmax_der1")?;
        let der2_v = read_f64(&client, der2, "query_softmax_der2")?;
        Ok((der1_v, der2_v))
    }
}

/// QueryCrossEntropy per-query BOUNDED shift search (Open Q3; INDEPENDENTLY deferred). Returns the
/// per-query logit shifts (length `n_groups`) solving `Σ w·sigmoid(approx + shift) = Σ w·target` via
/// the fixed-count bisection + Newton [`query_cross_entropy_shift_kernel`]. This is a genuine,
/// self-consistent root-find (the returned shift satisfies the equation) — NOT a der claim; the full
/// QueryCrossEntropy der stays gated off until its CPU oracle lands.
pub(crate) fn query_cross_entropy_shifts_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
) -> CbResult<Vec<f64>> {
    validate_ranking_inputs(approx, target, q_offsets)?;
    let n = approx.len();
    let n_groups = q_offsets.len().saturating_sub(1);
    if n == 0 || n_groups == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = (weights, q_offsets);
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let weight_col = weight_column(weights, n)?;
        let client = selected_client();
        let approx_h = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
        let target_h = client.create(cubecl::bytes::Bytes::from_elems(target.to_vec()));
        let w_h = client.create(cubecl::bytes::Bytes::from_elems(weight_col));
        let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
        // The FINITE shift bracket passed as runtime data (see the kernel comment).
        let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![QCE_SHIFT_BRACKET]));
        let out = client.empty(n_groups * std::mem::size_of::<f64>());
        query_cross_entropy_shift_kernel::launch::<SelectedRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            unsafe { ArrayArg::from_raw_parts(approx_h, n) },
            unsafe { ArrayArg::from_raw_parts(target_h, n) },
            unsafe { ArrayArg::from_raw_parts(w_h, n) },
            unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
            unsafe { ArrayArg::from_raw_parts(params_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), n_groups) },
        );
        read_f64(&client, out, "query_cross_entropy_shift")
    }
}

// ===========================================================================
// Phase 13 Plan 05 (GPUT-22, D-08): the STOCHASTIC ranking pair — YetiRank / PFound-F.
//
// The device reproduces the CPU `cb_train::yetirank::sample_pairs` +
// `cb_compute::calc_ders_for_queries` (YetiRank arm = pairwise-logit der over the sampled
// competitors) stream BIT-FOR-BIT under the pinned-seed / frozen-fixture discipline (D-06):
//
//   1. host derives the per-query inner Gumbel seed inline (the single-block `derive_query_seeds`
//      2-level `TFastRng64` chain — O(1) base state per query, NO per-iteration host RNG readback,
//      the `bootstrap_device` precedent);
//   2. the device `#[cube]` [`yetirank_perturb_kernel`] re-expands each query's `TFastRng64` INLINE
//      (transcribed PCG, mirroring `kernels::mvs_device`) and, per permutation, per doc (in the EXACT
//      CPU draw order — perm-major, doc-ascending), draws `gen_rand_real1`, CASTS it to `f32`,
//      perturbs `exp(approx)` by the f32 ratio `u/(1.000001−u)` (Pitfall 4: the f32 round is
//      load-bearing), producing the resident `perturbed[perm·n + d]`;
//   3. host sorts each query DESCENDING per permutation by reusing [`segmented_radix_sort`] (a
//      STABLE 2-pass 64-bit LSD radix over the non-negative-f64 bit pattern → full-precision order),
//      accumulates the Classic decayed pairwise weights (f32 storage — the parity contract), then the
//      accumulated competitors feed the transcribed pairwise-logit der.
//
// The RNG draw COUNT (`permutations · query_size` per query) is asserted by the self-oracle — a
// divergent count desyncs every subsequent draw beyond ε=1e-4 (T-13-10). No `cb-train` dep (the
// feature-unification landmine); the sampler + der are transcribed inline. No `-inf` literal in the
// `#[cube]` body (the perturbation is a finite ratio). No `unwrap`/`expect`/`panic`/indexing in
// production (workspace lints + D-13).
// ===========================================================================

/// LCG multiplier `A` (`cb_core::rng::LCG_MULTIPLIER`) transcribed inline (the `#[cube]` body cannot
/// reach `cb_core`), matching [`crate::kernels::mvs_device`] / [`crate::kernels::bootstrap_device`].
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// `1 / (2^53 − 1)` — the `ToRandReal1` divisor, matching [`cb_core::TFastRng64::gen_rand_real1`].
const REAL1_INV: f64 = 1.0 / 9_007_199_254_740_991.0;

/// The Gumbel-noise `1.000001` denominator guard (`yetirank_helpers.cpp:149-152`): `expApprox *=
/// u / (1.000001 − u)`. The ratio is evaluated in `f32` (both operands `f32`) — the f32 round is
/// LOAD-BEARING for the ≤1e-4 gate (an all-f64 ratio drifts the sampled weights ~1e-8, Pitfall 4).
const YETI_NOISE_GUARD: f32 = 1.000_001;

/// The Classic-weight magic constant `0.15` ("Like in GPU", `yetirank_helpers.cpp:198`).
const YETI_MAGIC_CONST: f64 = 0.15;

/// `RotateBitsRight(v, r)` for a 32-bit word (`fast.h` `TPCGMixer`), matching
/// [`crate::kernels::mvs_device`]. `r = x >> 59` is in `0..32` (never 32); the `r == 0` guard avoids
/// the `v << 32` UB shift.
#[cube]
fn rank_rotate_right_u32(v: u32, r: u32) -> u32 {
    let mut out = v;
    if r != 0u32 {
        out = (v >> r) | (v << (32u32 - r));
    }
    out
}

/// `TPCGMixer::Mix` (`fast.h`): XSH-RR on the 64-bit state → 32-bit output, matching
/// [`cb_core::rng::pcg_mix`] exactly.
#[cube]
fn rank_pcg_mix(x: u64) -> u32 {
    let xorshifted = u32::cast_from(((x >> 18u32) ^ x) >> 27u32);
    let rot = u32::cast_from(x >> 59u32);
    rank_rotate_right_u32(xorshifted, rot)
}

/// Serial (unit 0) YetiRank/PFound-F in-query bootstrap perturbation kernel. Per query `g` (in group
/// order) it re-expands `TFastRng64::from_seed(seeds[g])` INLINE, then per permutation, per doc `d`
/// in `[begin, end)` (the EXACT CPU draw order) draws one `gen_rand_real1`, casts it to `f32`, and
/// writes `perturbed[p·n + d] = exp(approx[d]) · f32(u/(1.000001−u))`. `params = [permutations]`.
/// The draw stream is CONTINUOUS across permutations within a query (the CPU `for perm { for doc }`
/// order), so a per-doc jump is unnecessary — the serial loop advances the stream in place.
#[cube(launch)]
fn yetirank_perturb_kernel(
    approx: &Array<f64>,
    q_offsets: &Array<u32>,
    seeds: &Array<u64>,
    params: &Array<u32>,
    perturbed: &mut Array<f64>,
) {
    if ABSOLUTE_POS == 0 {
        let a = LCG_MULTIPLIER;
        let perms = params[0];
        let n = approx.len();
        let n_groups = q_offsets.len() - 1;
        let mut g = 0usize;
        while g < n_groups {
            let begin = q_offsets[g];
            let end = q_offsets[g + 1];
            let s = seeds[g];

            // from_seed(s): the `TReallyFastRng32(seed)` derive stream (addend `c == 1`) yields
            // seed1 = GenRand64, seq1 = GenRand32, seed2 = GenRand64, seq2 = GenRand32.
            let dc = 1u64;
            let mut dx = s;
            dx = dx * a + dc;
            let s1_lo = rank_pcg_mix(dx);
            dx = dx * a + dc;
            let s1_hi = rank_pcg_mix(dx);
            let seed1 = u64::cast_from(s1_lo) | (u64::cast_from(s1_hi) << 32u32);
            dx = dx * a + dc;
            let seq1 = rank_pcg_mix(dx);
            dx = dx * a + dc;
            let s2_lo = rank_pcg_mix(dx);
            dx = dx * a + dc;
            let s2_hi = rank_pcg_mix(dx);
            let seed2 = u64::cast_from(s2_lo) | (u64::cast_from(s2_hi) << 32u32);
            dx = dx * a + dc;
            let seq2 = rank_pcg_mix(dx);

            // TFastRng64::new: r1.c = (seq1<<1)|1; r2.seq = FixSeq(seq1, seq2); r2.c = (r2seq<<1)|1.
            let mask = 0x7fff_ffffu32;
            let mut r2seq = seq2;
            if (seq1 & mask) == (seq2 & mask) {
                r2seq = !seq2;
            }
            let mut r1x = seed1;
            let r1c = (u64::cast_from(seq1) << 1u32) | 1u64;
            let mut r2x = seed2;
            let r2c = (u64::cast_from(r2seq) << 1u32) | 1u64;

            let mut p = 0u32;
            while p < perms {
                let base = (p as usize) * n;
                let mut d = begin;
                while d < end {
                    // gen_rand() = (r1.GenRand() << 32) | r2.GenRand(); real1 = (gen>>11)·REAL1_INV.
                    r1x = r1x * a + r1c;
                    let hi = rank_pcg_mix(r1x);
                    r2x = r2x * a + r2c;
                    let lo = rank_pcg_mix(r2x);
                    let rand64 = (u64::cast_from(hi) << 32u32) | u64::cast_from(lo);
                    let real1 = f64::cast_from(rand64 >> 11u32) * REAL1_INV;
                    // The Gumbel ratio in f32 (both operands f32) — the load-bearing round.
                    let uf = f32::cast_from(real1);
                    let ratio = uf / (YETI_NOISE_GUARD - uf);
                    let expa = f64::exp(approx[d as usize]);
                    perturbed[base + d as usize] = expa * f64::cast_from(ratio);
                    d += 1u32;
                }
                p += 1u32;
            }
            g += 1usize;
        }
    }
}

/// The per-query inner Gumbel seeds for a single-block (`blockCount == 1`) fit — the host-side
/// transcription of `cb_train::yetirank::derive_query_seeds` (`yetirank_helpers.cpp:365-389` +
/// `restorable_rng.cpp:3-9`), reusing the sanctioned [`cb_core::TFastRng64`] (NOT a `cb-train` dep):
/// `blockSeed = TFastRng64(random_seed).GenRand()`, then per query `querySeed =
/// TFastRng64(blockSeed).GenRand()`. This is the O(1) base state the device kernel re-expands.
#[must_use]
pub(crate) fn derive_query_seeds_inline(random_seed: u64, group_count: usize) -> Vec<u64> {
    let mut seed_rng = cb_core::TFastRng64::from_seed(random_seed);
    let block_seed = seed_rng.gen_rand();
    let mut block_rng = cb_core::TFastRng64::from_seed(block_seed);
    (0..group_count).map(|_| block_rng.gen_rand()).collect()
}

/// The pairwise-logit pair probability `p = exp(loser)/(exp(loser)+exp(winner))` (else `0.5` when
/// both exps underflow) — the host transcription of `cb_compute::pairlogit_pair_prob` (kept inline so
/// the stochastic der matches the CPU `pairlogit_group_der` scatter bit-for-bit; `cb_compute` is a
/// normal dep, but transcribing avoids re-exporting a private helper).
#[cfg(not(feature = "wgpu"))]
#[inline]
fn pairlogit_pair_prob_local(winner_approx: f64, loser_approx: f64) -> f64 {
    let exp_loser = loser_approx.exp();
    let exp_winner = winner_approx.exp();
    let denom = exp_loser + exp_winner;
    if denom > 0.0 {
        exp_loser / denom
    } else {
        0.5
    }
}

/// The DESCENDING per-query sort order (global doc indices) for one permutation's `perturbed` slice,
/// reusing [`segmented_radix_sort`] (the acceptance-criterion sort reuse). `perturbed` values are all
/// NON-NEGATIVE (`exp(approx) > 0`, ratio `>= 0`), so their raw IEEE-754 `f64` bit patterns are
/// MONOTONE. Radix-sorting the BITWISE-COMPLEMENTED key (`!bits`) ASCENDING in a single stable 2-pass
/// 64-bit LSD radix (low 32 bits then high 32 bits) therefore yields the full DESCENDING value order
/// directly, and — because the radix is STABLE and equal values share an equal complemented key —
/// tied documents stay in ORIGINAL index order, exactly matching the upstream stable descending sort
/// (`yetirank_helpers.cpp:326-331`). A plain reverse-of-stable-ascending would instead flip ties into
/// reversed-index order (the WR-01 parity divergence), so we complement the key rather than reverse.
#[cfg(not(feature = "wgpu"))]
fn descending_order_per_query(perturbed: &[f64], q_offsets: &[u32]) -> CbResult<Vec<u32>> {
    let n = perturbed.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut head = vec![0u32; n];
    for w in q_offsets.windows(2) {
        let b = *w.first().unwrap_or(&0) as usize;
        if let Some(slot) = head.get_mut(b) {
            *slot = 1;
        }
    }
    // Complement the radix key so ONE stable ASCENDING pass already yields the DESCENDING value
    // order with tie order preserved (WR-01). Non-negative f64 bit patterns are monotone, so `!bits`
    // inverts the ordering while keeping equal keys equal; the stable radix then leaves tied docs in
    // ORIGINAL index order — exactly the upstream stable descending sort, not a tie-flipping reverse.
    let ord: Vec<u64> = perturbed.iter().map(|&v| !v.to_bits()).collect();
    let lo: Vec<u32> = ord.iter().map(|&b| b as u32).collect();
    let hi: Vec<u32> = ord.iter().map(|&b| (b >> 32) as u32).collect();
    let idx0: Vec<u32> = (0..n as u32).collect();
    // Pass 1: stable sort by the low 32 bits.
    let (_sk, order_lo) = segmented_radix_sort(&head, &lo, &idx0)?;
    // Reorder the high 32 bits by pass-1's permutation, then stable-sort by them (carry the perm).
    let hi_re: Vec<u32> = order_lo
        .iter()
        .map(|&i| hi.get(i as usize).copied().unwrap_or(0))
        .collect();
    let (_sk2, order) = segmented_radix_sort(&head, &hi_re, &order_lo)?;
    // `order` is now DESCENDING by expApprox within each query (ties in original index order).
    Ok(order)
}

/// `AddWeight` (`yetirank_helpers.cpp:185-191`): route the pair weight to the higher-relevance doc as
/// the winner (`f32` storage + accumulation, the parity contract). Ties (`==`) add nothing.
#[cfg(not(feature = "wgpu"))]
#[inline]
fn yeti_add_weight(
    first: usize,
    second: usize,
    relev_first: f32,
    relev_second: f32,
    weight: f32,
    cw: &mut [Vec<f32>],
) {
    if relev_first > relev_second {
        if let Some(row) = cw.get_mut(first) {
            if let Some(cell) = row.get_mut(second) {
                *cell += weight;
            }
        }
    } else if relev_first < relev_second {
        if let Some(row) = cw.get_mut(second) {
            if let Some(cell) = row.get_mut(first) {
                *cell += weight;
            }
        }
    }
}

/// The STOCHASTIC YetiRank / PFound-F device der (Phase 13 Plan 05, D-08), shared by both public
/// arms (PFound-F == `YetiRankPairwise` rides the SAME sampled-pair stream + competitor der; only the
/// leaf path differs, which is decided later in boosting — NOT here). Returns `(der1, der2, draws)`,
/// where `draws == permutations · n` is the consumed `gen_rand_real1` count (asserted by the
/// self-oracle: a divergent count silently desyncs every value, T-13-10). Reproduces the CPU
/// `yetirank_sample_pairs` + `pairlogit_group_der` reference for the pinned `random_seed` bit-for-bit
/// at ε=1e-4. The covered regime is uniform per-group weight (`1.0`); the per-object `weights` are
/// accepted for signature symmetry but the pairwise der folds the weight into the PAIR (upstream).
#[cfg(not(feature = "wgpu"))]
fn yetirank_sample_der_core(
    approx: &[f64],
    target: &[f64],
    _weights: &[f64],
    q_offsets: &[u32],
    permutations: u32,
    decay: f64,
    random_seed: u64,
) -> CbResult<(Vec<f64>, Vec<f64>, usize)> {
    validate_ranking_inputs(approx, target, q_offsets)?;
    let n = approx.len();
    let n_groups = q_offsets.len().saturating_sub(1);
    if n == 0 || n_groups == 0 || permutations == 0 {
        return Ok((vec![0.0; n], vec![0.0; n], 0));
    }

    // (1) Per-query inner Gumbel seeds (host, O(1) base state) — the single-block derivation.
    let seeds = derive_query_seeds_inline(random_seed, n_groups);

    // (2) Device: perturb exp(approx) per permutation/doc in the exact CPU draw order (inline PCG).
    let client = selected_client();
    let approx_h = client.create(cubecl::bytes::Bytes::from_elems(approx.to_vec()));
    let off_h = client.create(cubecl::bytes::Bytes::from_elems(q_offsets.to_vec()));
    let seeds_h = client.create(cubecl::bytes::Bytes::from_elems(seeds.clone()));
    let params_h = client.create(cubecl::bytes::Bytes::from_elems(vec![permutations]));
    let total = permutations as usize * n;
    let perturbed_h = client.empty(total * std::mem::size_of::<f64>());
    yetirank_perturb_kernel::launch::<SelectedRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(approx_h, n) },
        unsafe { ArrayArg::from_raw_parts(off_h, q_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(seeds_h, n_groups) },
        unsafe { ArrayArg::from_raw_parts(params_h, 1) },
        unsafe { ArrayArg::from_raw_parts(perturbed_h.clone(), total) },
    );
    // One bulk read-back of the perturbed values (NOT a per-iteration RNG readback — the RNG stays
    // device-resident; this is the same data-crossing class as the deterministic driver's der).
    let perturbed = read_f64(&client, perturbed_h, "yetirank_perturbed")?;

    // (3) Host: per permutation sort DESCENDING (segmented_radix_sort) + Classic decayed weights.
    let mut der1 = vec![0.0_f64; n];
    let mut der2 = vec![0.0_f64; n];
    let draws = permutations as usize * n;
    for w in q_offsets.windows(2) {
        let begin = *w.first().unwrap_or(&0) as usize;
        let end = *w.get(1).unwrap_or(&0) as usize;
        let qs = end.saturating_sub(begin);
        if qs == 0 {
            continue;
        }
        // competitorsWeights[winner_local][loser_local], f32 (the parity contract).
        let mut cw = vec![vec![0.0_f32; qs]; qs];
        for p in 0..permutations as usize {
            let base = p * n;
            let seg = perturbed.get(base + begin..base + end).unwrap_or(&[]);
            // Descending order (LOCAL indices) for this permutation's single-query slice
            // (`local_offsets` = one segment `[0, qs)`) via the reused `segmented_radix_sort`.
            let local_offsets = vec![0u32, qs as u32];
            let order_global = descending_order_per_query(seg, &local_offsets)?;
            // CalcWeightsClassic (yetirank_helpers.cpp:193-205): decayed |Δrelev| along the order.
            let mut decay_coef = 1.0_f64;
            for doc_id in 1..qs {
                let first = order_global.get(doc_id - 1).copied().unwrap_or(0) as usize;
                let second = order_global.get(doc_id).copied().unwrap_or(0) as usize;
                #[allow(clippy::cast_possible_truncation)]
                let rf = target.get(begin + first).copied().unwrap_or(0.0) as f32;
                #[allow(clippy::cast_possible_truncation)]
                let rs = target.get(begin + second).copied().unwrap_or(0.0) as f32;
                #[allow(clippy::cast_possible_truncation)]
                let pair_weight =
                    (YETI_MAGIC_CONST * decay_coef * f64::from((rf - rs).abs())) as f32;
                yeti_add_weight(first, second, rf, rs, pair_weight, &mut cw);
                decay_coef *= decay;
            }
        }

        // (4) Normalize `queryWeight · cw / permutations` (f32) → competitors, then pairlogit der.
        // queryWeight is the covered uniform 1.0. The pairwise-logit der is a fixed-order scatter
        // (winner raises own der, lowers each loser's) — reproduces `pairlogit_group_der` exactly.
        let denom = permutations as f32;
        for winner in 0..qs {
            let winner_approx = approx.get(begin + winner).copied().unwrap_or(0.0);
            let mut winner_der = 0.0_f64;
            let mut winner_second = 0.0_f64;
            for loser in 0..qs {
                let cwl = cw
                    .get(winner)
                    .and_then(|row| row.get(loser))
                    .copied()
                    .unwrap_or(0.0);
                let w_f32 = 1.0_f32 * cwl / denom;
                if w_f32 != 0.0 {
                    let loser_approx = approx.get(begin + loser).copied().unwrap_or(0.0);
                    let p = pairlogit_pair_prob_local(winner_approx, loser_approx);
                    let wp = f64::from(w_f32);
                    winner_der += wp * p;
                    winner_second += wp * p * (p - 1.0);
                    if let Some(d) = der1.get_mut(begin + loser) {
                        *d -= wp * p;
                    }
                    if let Some(d) = der2.get_mut(begin + loser) {
                        *d += wp * p * (p - 1.0);
                    }
                }
            }
            if let Some(d) = der1.get_mut(begin + winner) {
                *d += winner_der;
            }
            if let Some(d) = der2.get_mut(begin + winner) {
                *d += winner_second;
            }
        }
    }
    Ok((der1, der2, draws))
}

/// YetiRank device der (Phase 13 Plan 05, D-08): the stochastic in-query bootstrap-sampled listwise
/// der (POINTWISE leaf). Returns `(der1, der2)` (each length `n`), reproducing the CPU
/// `yetirank_sample_pairs` + `calc_ders_for_queries` reference for the pinned `random_seed`
/// bit-for-bit at ε=1e-4. See [`yetirank_sample_der_core`].
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn yetirank_ders_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
    permutations: u32,
    decay: f64,
    random_seed: u64,
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    #[cfg(feature = "wgpu")]
    {
        validate_ranking_inputs(approx, target, q_offsets)?;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let (d1, d2, _draws) =
            yetirank_sample_der_core(approx, target, weights, q_offsets, permutations, decay, random_seed)?;
        Ok((d1, d2))
    }
}

/// PFound-F device der (Phase 13 Plan 05, D-08): the stochastic PAIRWISE-leaf listwise der
/// (`YetiRankPairwise`, the GPU `pfound_f` arm). Shares the SAME sampled-pair stream + competitor der
/// as [`yetirank_ders_host`] (only the leaf path differs — Cholesky vs pointwise — decided later in
/// boosting, NOT here), so the device der is identical. Returns `(der1, der2)`.
#[cfg_attr(feature = "wgpu", allow(unused_variables))]
pub(crate) fn pfound_f_ders_host(
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    q_offsets: &[u32],
    permutations: u32,
    decay: f64,
    random_seed: u64,
) -> CbResult<(Vec<f64>, Vec<f64>)> {
    #[cfg(feature = "wgpu")]
    {
        validate_ranking_inputs(approx, target, q_offsets)?;
        return Err(wgpu_reject());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let (d1, d2, _draws) =
            yetirank_sample_der_core(approx, target, weights, q_offsets, permutations, decay, random_seed)?;
        Ok((d1, d2))
    }
}

/// The consumed `gen_rand_real1` draw COUNT for a stochastic ranking fit — `permutations · n`
/// (perm-major, doc-ascending, per query). Exposed so the self-oracle asserts the device stream
/// length matches the CPU exactly (a divergent count silently shifts every value, T-13-10).
#[must_use]
pub(crate) fn yetirank_draw_count(n: usize, permutations: u32) -> usize {
    permutations as usize * n
}

/// Read a resident `f64` handle back to host, mapping a failure to [`CbError::Degenerate`] (WR-05),
/// never a silent zero buffer.
#[cfg(not(feature = "wgpu"))]
fn read_f64(
    client: &cubecl::client::ComputeClient<SelectedRuntime>,
    handle: Handle,
    who: &str,
) -> CbResult<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| CbError::Degenerate(format!("ranking {who} f64 read-back failed: {e:?}")))?;
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes).to_vec())
}
